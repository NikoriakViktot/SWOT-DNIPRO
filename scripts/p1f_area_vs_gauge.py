#!/usr/bin/env python
"""P1E (coverage timeline) + P1F (satellite A(H) vs Rozumivka design/first
dataset), 2026-09-11 operator correction.

P1F is bounded to 2019-01-01..2023-06-05 because the Rozumivka gauge record
(all_water_levels_common_frame.csv, station:80959) starts 2019-01-01
(operator-confirmed) -- NOT because the Sentinel catalogue search (P1A) was
itself limited to that window (it was not; P1A found scenes from 2017-01-04).
Water_area_km2 is filled only for the 3 dates P1C actually measured a mask
for (2023-05-16/05-19/06-05); every other date is DESIGN ONLY (null area,
ready to fill once/if downloaded) -- per the operator's explicit
"do not yet use it to tune masks, prepare the dataset" instruction.

Outputs
-------
outputs/tables/prebreach_area_vs_rozumivka_level.csv
outputs/figures/P1_prebreach_scene_coverage_timeline.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

GAUGE_START = "2019-01-01"
CHECK_DATE = "2023-06-05"
INK, BLUE, RED, AMBER, GREEN = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e"


def load_gauge():
    g = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    roz = g[(g.station_or_domain == "station:80959") & (g.source == "gauge")].copy()
    roz["date"] = pd.to_datetime(roz.date).dt.strftime("%Y-%m-%d")
    return roz.set_index("date")[["raw_level_m", "raw_vertical_frame",
                                  "transformed_level_m", "common_vertical_frame",
                                  "corrected_level_m", "empirical_correction_m"]]


def main() -> None:
    cand = pd.read_csv(CFG.TABLES / "prebreach_fullpool_scene_candidates.csv")
    roz = load_gauge()

    # =========================================================== P1E timeline
    cand["dt"] = pd.to_datetime(cand.date)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1.2]})

    ax = axes[0]
    full = cand[cand.n_tiles_present == 4]
    partial = cand[cand.n_tiles_present < 4]
    ax.scatter(partial.dt, partial.native_coverage_fraction, s=10, color=GREY if False else "#c9c9c9",
              label=f"partial tile set (n={len(partial)})", zorder=2)
    sc = ax.scatter(full.dt, full.native_coverage_fraction, s=14,
                    c=full.est_cloud_over_reservoir_pct_weighted, cmap="RdYlGn_r",
                    vmin=0, vmax=60, label=f"all 4 tiles found (n={len(full)})", zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("est. cloud over reservoir (%, area-weighted whole-scene metadata)")
    ax.axvline(pd.Timestamp(CHECK_DATE), color=RED, lw=1.5, ls="--",
              label=f"{CHECK_DATE} (primary checkpoint)")
    ax.axvline(pd.Timestamp("2023-06-06"), color=INK, lw=1.2, ls=":", label="breach")
    ax.set_ylabel("native geometric coverage\nfraction of reservoir")
    ax.set_title("P1E -- pre-breach Sentinel-2 scene coverage timeline, "
                 "2017-2023-06-05 (NOT mixing catalogue availability with "
                 "valid optical observation -- see caption)", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.02, 1.05)

    ax = axes[1]
    ax.bar(cand.dt, cand.n_tiles_present, width=3, color=BLUE, alpha=0.6)
    ax.axhline(4, color=INK, lw=0.8, ls=":")
    ax.set_ylabel("n tiles found\nin catalogue")
    ax.set_ylim(0, 4.5)

    ax = axes[2]
    roz_plot = roz.copy()
    roz_plot.index = pd.to_datetime(roz_plot.index)
    ax.plot(roz_plot.index, roz_plot.corrected_level_m, color=GREEN, lw=0.8, alpha=0.8)
    ax.set_ylabel("Rozumivka level\n(m, EVRF2019)")
    ax.set_xlabel("date")
    ax.axvline(pd.Timestamp(GAUGE_START), color=AMBER, lw=1, ls="--",
              label=f"gauge record starts {GAUGE_START}")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("A fully-tiled date with 80% cloud is NOT a 100%-observed reservoir -- "
                 "top panel colour is cloud, position is TILE coverage; the two are kept separate.",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    out_fig = CFG.FIG / "P1_prebreach_scene_coverage_timeline.png"
    fig.savefig(out_fig, dpi=150)
    print(f"-> {out_fig}")

    # =========================================================== P1F dataset
    cons_path = CFG.TABLES / "sa2_multidate_consensus.csv"
    cons = pd.read_csv(cons_path).set_index("date") if cons_path.exists() else pd.DataFrame()

    f = cand[(cand.dt >= GAUGE_START) & (cand.dt <= CHECK_DATE)].copy()
    rows = []
    for r in f.itertuples():
        g = roz.loc[r.date] if r.date in roz.index else None
        water_area = (cons.loc[r.date, "observed_water_area_km2"]
                      if (len(cons) and r.date in cons.index) else np.nan)
        n_mos = (int(cons.loc[r.date, "n_tiles_actually_mosaicked"])
                if (len(cons) and r.date in cons.index) else 0)
        rows.append({
            "date": r.date,
            "sentinel_datetime": r.date,  # STAC datetime already date-only in P1A summary; scene-level datetime lives in prebreach_sentinel_discovery_full.csv
            "gauge_date": r.date,  # daily gauge series, same-day by construction
            "H_Rozumivka_BS77": g.raw_level_m if g is not None else np.nan,
            "delta_to_EVRF2019_m": (g.transformed_level_m - g.raw_level_m) if g is not None else np.nan,
            "H_Rozumivka_EVRF2019": g.transformed_level_m if g is not None else np.nan,
            "water_area_km2": water_area,
            "water_area_source": ("P1C measured (n_tiles=%d)" % n_mos) if np.isfinite(water_area) else "NOT YET MEASURED -- design only, needs download",
            "native_coverage_fraction": r.native_coverage_fraction,
            "est_cloud_over_reservoir_pct": r.est_cloud_over_reservoir_pct_weighted,
            "temporal_match_quality": "same-day (daily gauge series)",
        })
    out = pd.DataFrame(rows).sort_values("date")
    out_path = CFG.TABLES / "prebreach_area_vs_rozumivka_level.csv"
    out.to_csv(out_path, index=False)
    print(f"-> {out_path}  ({len(out)} candidate dates, gauge-bounded "
          f"{GAUGE_START}..{CHECK_DATE})")
    n_measured = out.water_area_km2.notna().sum()
    print(f"   water_area_km2 actually measured (P1C): {n_measured} / {len(out)}")
    print(f"   remaining {len(out) - n_measured} rows are DESIGN ONLY (null area) "
          f"pending download approval -- not used to tune anything")
    if n_measured:
        print(out[out.water_area_km2.notna()][
            ["date", "H_Rozumivka_EVRF2019", "water_area_km2", "native_coverage_fraction"]
        ].to_string())


if __name__ == "__main__":
    main()
