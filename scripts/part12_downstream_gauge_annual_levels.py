#!/usr/bin/env python
"""PART 12 — mean annual water level at Kherson and Nova Kakhovka, in EVRF2019.

Turns the user's own instruction into a small, auditable table: take the mean
annual level at each gauge, referenced to its own gauge zero, converted
Baltic (BS-77) -> EVRF2019. No new vertical-datum math happens here -- that
step is `vertical.gauge_evrf2019()`, already applied to every observation by
`part1_gauge_rereference.py` (`H_evrf2019_m = zero_bs77_m + stage_m +
delta_epsg9902_m`, EPSG:9902 grid, official geodetic transform only, no
empirical alignment). This script only aggregates that existing per-observation
series to one number per (station, regime, year), then to one number per
(station, regime).

WHY MEAN-OF-ANNUAL-MEANS, NOT A FLAT POOLED MEAN. Kherson has 2,176 daily-ish
observations unevenly spread 2019-2025 (a real gap in 2024's yearbook, filled
only by a thinner 2025 term store); Nova Kakhovka has 1,096, stopping hard at
2021-12-31. A flat mean over all rows would silently let the years with denser
sampling dominate. Averaging within each calendar year first, then averaging
the yearly means, weights every year the series actually covers equally.

REGIMES. Following the project's own three-period split (see
`scripts/discover_swot.py`'s `REGIMES`): PRE_BREACH ends 2023-06-05,
BREACH_DRAWDOWN (06-06 .. 08-31) is excluded from both means -- a transient,
not a state -- POST_BREACH starts 2023-09-01.

NOVA KAKHOVKA POST_BREACH IS NaN, ON PURPOSE. No observation exists anywhere
in the source after 2021-12-31 (confirmed in `part1`'s own summary). The
reservoir gauge lost its meaning when the pool it measured drained. This row
is written with an explicit note, never dropped, never filled with a proxy --
"missing slope support is NaN, never 0" (`outputs/planning/13`, standing
rules).

Outputs
-------
outputs/tables/part12_downstream_annual_levels.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

LEVELS_PQ = ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet"
STATIONS = {80805: "Kherson", 80977: "Nova Kakhovka"}
REGIMES = [("PRE_BREACH", pd.Timestamp("2000-01-01"), pd.Timestamp("2023-06-05")),
           ("BREACH_DRAWDOWN", pd.Timestamp("2023-06-06"), pd.Timestamp("2023-08-31")),
           ("POST_BREACH", pd.Timestamp("2023-09-01"), pd.Timestamp("2100-01-01"))]


def assign_regime(dates: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, a, b in REGIMES:
        out[(dates >= a) & (dates <= b)] = name
    return out


def annual_mean_of_means(d: pd.DataFrame) -> tuple[float, float, int, int]:
    """mean-of-annual-means, its spread, n years, n obs -- see module docstring."""
    yearly = d.groupby(d.date.dt.year)["H_evrf2019_m"].mean()
    if yearly.empty:
        return np.nan, np.nan, 0, 0
    return float(yearly.mean()), float(yearly.std(ddof=0)), int(len(yearly)), int(len(d))


def main() -> None:
    print("=" * 78)
    print("PART 12 — mean annual level, Kherson + Nova Kakhovka, EVRF2019")
    print("=" * 78)
    if not LEVELS_PQ.exists():
        raise SystemExit(f"missing {LEVELS_PQ} -- run part1_gauge_rereference.py first")

    lv = pd.read_parquet(LEVELS_PQ)
    lv = lv[lv.station_id.isin(STATIONS) & (lv.qc == "ok")].copy()
    lv["regime"] = assign_regime(lv.date)

    rows = []
    for sid, name in STATIONS.items():
        d = lv[lv.station_id == sid]
        for regime, _, _ in REGIMES:
            if regime == "BREACH_DRAWDOWN":
                continue  # transient, not a state -- never averaged
            dr = d[d.regime == regime]
            mean_of_means, sd_annual, n_years, n_obs = annual_mean_of_means(dr)
            pooled = float(dr.H_evrf2019_m.mean()) if len(dr) else np.nan
            note = ""
            if n_obs == 0:
                note = "no observations in this regime in any source"
            rows.append(dict(
                station_id=sid, name_en=name, regime=regime,
                mean_annual_evrf2019_m=mean_of_means,
                pooled_mean_evrf2019_m=pooled,
                sd_annual_m=sd_annual, n_years=n_years, n_obs=n_obs,
                zero_bs77_m=float(d.zero_bs77_m.iloc[0]) if len(d) else np.nan,
                delta_epsg9902_m=float(d.delta_epsg9902_m.iloc[0]) if len(d) else np.nan,
                notes=note))
            print(f"  {name:16s} {regime:16s}  "
                  f"{'NaN' if np.isnan(mean_of_means) else f'{mean_of_means:7.3f} m':>9s}"
                  f"  (pooled {'NaN' if np.isnan(pooled) else f'{pooled:7.3f} m':>9s})"
                  f"  n_years={n_years:2d}  n_obs={n_obs:5d}  {note}")

    out = pd.DataFrame(rows)
    out_path = CFG.TABLES / "part12_downstream_annual_levels.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
