#!/usr/bin/env python
"""MS12 -- within-overpass water-surface heterogeneity (Section 4.4, Table 3, Figure 4), one source.

Why this exists. The heterogeneity numbers of the manuscript (0.117 -> 0.397 m) come from
scripts/phase20_fragmentation_report.py, but Figure 4 was drawn from a different sample -- the
33 slope-sample overpasses, where p95 - p05 is taken over SIX beam-median points spread along
20-70 km of centreline, so it mostly measures slope x span (median ~1.8 m after the breach) --
and the "planar trend removed" numbers (0.103 -> 0.269 m) had no script in the repository at all.
This script computes both versions from one per-date table and draws Figure 4 from it.

Per date (unit = one ICESat-2 acquisition date in the footprint, >= 30 segments):
  * PRE_BREACH: every ATL13 segment of the date over the pool (kakhovka_atl13_segments, EGG2015
    frame: h_wgs84 + free2mean - zeta_EGG2015) -- the pool is one water body;
  * BREACH_DRAWDOWN / POST_BREACH: ATL13 segments classified as MAIN_CHANNEL, CONNECTED_SIDE_CHANNEL
    or TRIBUTARY (atl13_water_classification.parquet); residual ponds and flooded depressions are
    excluded, so the metric describes the connected river system, not the spread between separate
    water bodies (the same rule as phase20_fragmentation_report.atl13_wse_by_date).
  * raw:       p95 - p05 of the date's water-surface elevations;
  * detrended: p95 - p05 of the residuals about a robust plane (soft-L1 least squares in
               EPSG:32636 metres: wse ~ a + b*E + c*N), i.e. the spread left once the date's own
               gradient is removed.
Contrast PRE vs POST on the per-date values: difference of medians, 95 % bootstrap interval and
two-sided permutation test (the helpers of phase20_fragmentation_report), Mann-Whitney U.

Outputs
    outputs/paper/validation/ms12_heterogeneity_by_date.csv
    outputs/paper/validation/ms12_heterogeneity_summary.csv
    outputs/paper/figures/F04_heterogeneity.png / .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from swot_dnipro import config as CFG  # noqa: E402

OUT = ROOT / "outputs/paper/validation"
FIG = ROOT / "outputs/paper/figures"
BREACH = pd.Timestamp(CFG.BREACH_DATE)
CONNECTED = {"MAIN_CHANNEL", "CONNECTED_SIDE_CHANNEL", "TRIBUTARY"}
MIN_SEG = 30
COLOURS = {"PRE_BREACH": "#1f6f8b", "BREACH_DRAWDOWN": "#b5651d", "POST_BREACH": "#2e7d32"}
LABELS = {"PRE_BREACH": "before the breach", "BREACH_DRAWDOWN": "2023 drawdown", "POST_BREACH": "after the breach"}


def p95_p05(v: np.ndarray) -> float:
    return float(np.percentile(v, 95) - np.percentile(v, 5))


def detrended_range(lon: np.ndarray, lat: np.ndarray, h: np.ndarray) -> float:
    import pyproj
    e, n = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform(lon, lat)
    e, n = (np.asarray(e) - np.mean(e)) / 1e3, (np.asarray(n) - np.mean(n)) / 1e3      # km, centred
    A = np.column_stack([np.ones_like(e), e, n])
    x0 = np.linalg.lstsq(A, h, rcond=None)[0]
    fit = optimize.least_squares(lambda x: A @ x - h, x0, loss="soft_l1", f_scale=0.05)
    return p95_p05(h - A @ fit.x)


def per_date() -> pd.DataFrame:
    from swot_dnipro.vertical import sample_grid
    rows = []
    pre = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet")
    pre["dt"] = pd.to_datetime(pre["time"], utc=True, errors="coerce").dt.tz_localize(None)
    pre = pre[pre.dt < BREACH].dropna(subset=["lat", "lon", "h_wgs84_m", "dt"])
    z = sample_grid(CFG.EGG2015_TIF, pre.lon.values, pre.lat.values)
    pre["wse_m"] = pre.h_wgs84_m.values + CFG.free2mean(pre.lat.values) - z
    pre["date"] = pre.dt.dt.normalize()
    for d, g in pre.groupby("date"):
        if len(g) >= MIN_SEG:
            rows.append(dict(date=d, period="PRE_BREACH", n_segments=len(g), sample="whole pool",
                             range_m=p95_p05(g.wse_m.values),
                             range_detrended_m=detrended_range(g.lon.values, g.lat.values, g.wse_m.values)))
    cl = pd.read_parquet(CFG.TABLES / "atl13_water_classification.parquet")
    cl["date"] = pd.to_datetime(cl["date"])
    for (d, per), g in cl.groupby(["date", "period"]):
        w = g[g.water_class.isin(CONNECTED)]
        if len(w) >= MIN_SEG:
            rows.append(dict(date=d, period=per, n_segments=len(w), sample="connected channels",
                             range_m=p95_p05(w.wse_m.values),
                             range_detrended_m=detrended_range(w.lon.values, w.lat.values, w.wse_m.values)))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def contrast(df: pd.DataFrame, col: str) -> dict:
    import phase20_fragmentation_report as P20
    P20.RNG = np.random.default_rng(CFG.SEED)
    a = df.loc[df.period == "PRE_BREACH", col].values
    b = df.loc[df.period == "POST_BREACH", col].values
    row = P20.regime_row(a, b, col, col)
    return row


def figure(df: pd.DataFrame, summ: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    for per in ("PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"):
        g = df[df.period == per]
        ax.scatter(g.date, g.range_m, s=14, color=COLOURS[per], alpha=0.8, lw=0,
                   label=f"{LABELS[per]} (n = {len(g)})", zorder=3)
    raw = summ.set_index("metric").loc["range_m"]
    for per, med in (("PRE_BREACH", raw.median_pre), ("POST_BREACH", raw.median_post)):
        g = df[df.period == per]
        ax.hlines(med, g.date.min(), g.date.max(), color=COLOURS[per], lw=2.2, zorder=4)
        ax.annotate(f"median {med:.3f} m", (g.date.max(), med), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=COLOURS[per])
    ax.axvline(BREACH, color="0.45", ls="--", lw=1)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("within-date WSE range, p95 − p05 (m)")
    ax.set_title("Water-surface heterogeneity by ICESat-2 date", loc="left", fontsize=11)
    ax.grid(axis="y", which="major", color="0.9")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F04_heterogeneity.{ext}")


def main() -> None:
    df = per_date()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ms12_heterogeneity_by_date.csv", index=False)
    summ = pd.DataFrame([contrast(df, "range_m"), contrast(df, "range_detrended_m")])
    summ.to_csv(OUT / "ms12_heterogeneity_summary.csv", index=False)
    figure(df, summ)
    print(df.period.value_counts().to_dict())
    print(summ[["metric", "n_pre", "n_post", "median_pre", "median_post", "diff_post_minus_pre",
                "diff_ci_lo", "diff_ci_hi", "permutation_p", "mannwhitney_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
