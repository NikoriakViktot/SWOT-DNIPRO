#!/usr/bin/env python
"""P1F v2 -- populate satellite A(H) with real measured areas from the
targeted download round, then compare against the historical Table 19 A(H)
curve (2026-09-11 operator correction, Steps 9-10).

VERTICAL SYSTEM, explicit (Step 10): historical Table 19
(data/historical/historical_level_area_volume.csv) is HISTORICAL_BALTIC
(BS-42/BS-77, not stated, not converted in the source). Rozumivka gauge
levels used everywhere else in this project (all_water_levels_common_frame.csv)
are already EVRF2019. The bridge used here is the SAME empirical constant
already established and used elsewhere in this project
(hist22_low_water_search.py: BS_TO_EVRF = 0.185 m) -- not invented for this
comparison. Both original and bridged historical columns are kept so nothing
is silently overlaid.

CAVEAT carried into the figure and printed output: Table 19's water_level_m
is a UNIFORM dam-referenced design level (one number for the whole pool);
Rozumivka is a POINT gauge ~130 km from the dam. The project's own prior
finding (PRE_BREACH longitudinal slope ~0.04 cm/km, within-date WSE range
0.09 m) supports treating them as comparable for the PRE_BREACH period
specifically, but this is a modelling choice, not a proven identity -- stated
outright, not assumed.

CRITICAL (operator instruction): Sentinel masks are NEVER fit to Table 19.
This is a comparison of two independent measurements, reported with absolute
and relative differences, not a calibration.

Outputs
-------
outputs/tables/prebreach_area_vs_rozumivka_level.csv  (UPDATED with real areas)
outputs/tables/satellite_vs_historical_hypsometry.csv
outputs/figures/P1_satellite_vs_historical_AH.png
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

BS_TO_EVRF = 0.185   # established elsewhere in this project (hist22), reused not reinvented
INK, BLUE, RED, AMBER, GREEN = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e"


def main() -> None:
    ah = pd.read_csv(CFG.TABLES / "prebreach_area_vs_rozumivka_level.csv")
    cons = pd.read_csv(CFG.TABLES / "sa2_multidate_consensus.csv")
    valid = cons[cons.status == "VALID"].set_index("date")
    print(f"VALID measured dates available to populate A(H): {len(valid)}")

    ah = ah.set_index("date")
    for d in valid.index:
        if d in ah.index:
            ah.loc[d, "water_area_km2"] = valid.loc[d, "observed_water_area_km2"]
            ah.loc[d, "water_area_source"] = "P1C v2 measured (VALID, full tile coverage)"
    # Step 11 (explicit, operator): 2023-05-16 (1/4 tile, IoU~0.012) and
    # 2023-05-19 (1/4 tile, IoU~0.16) are partial observations left over from
    # the FIRST pilot run of p1c_sa2_consensus.py, BEFORE this targeted round.
    # They are not in this round's 18-date VALID set (never selected for
    # targeted download) and their stale, badly-wrong area values (27.5 km2
    # and 354.9 km2 against an expected ~2200 km2) must not silently persist
    # in the "measured" column just because nothing here overwrote them.
    KNOWN_INVALID = ["2023-05-16", "2023-05-19"]
    for d in KNOWN_INVALID:
        if d in ah.index and ah.loc[d, "water_area_source"] != (
                "P1C v2 measured (VALID, full tile coverage)"):
            ah.loc[d, "water_area_km2"] = np.nan
            ah.loc[d, "water_area_source"] = "INVALID_FOR_FULL_DOMAIN_IOU (1/4 tile pilot observation, excluded per Step 11)"
    ah = ah.reset_index()
    ah.to_csv(CFG.TABLES / "prebreach_area_vs_rozumivka_level.csv", index=False)
    n_measured = ah.water_area_km2.notna().sum()
    print(f"prebreach_area_vs_rozumivka_level.csv: {n_measured} / {len(ah)} rows now have a "
          f"real measured water_area_km2")

    sat = ah[ah.water_area_km2.notna()].copy()
    sat = sat.sort_values("H_Rozumivka_EVRF2019")
    print("\nMeasured satellite A(H) points:")
    print(sat[["date", "H_Rozumivka_EVRF2019", "water_area_km2"]].to_string(index=False))

    # ---- historical Table 19, vertical-bridged, columns kept explicit -----
    hist = pd.read_csv(CFG.ROOT / "data/historical/historical_level_area_volume.csv")
    hist = hist.rename(columns={"water_level_m": "H_historical_original",
                                "surface_area_km2": "A_historical_km2"})
    hist["historical_vertical_system"] = hist["vertical_datum"]
    hist["H_historical_common_EVRF2019_approx"] = hist.H_historical_original + BS_TO_EVRF
    hist = hist.sort_values("H_historical_common_EVRF2019_approx")

    # ---- compare: interpolate historical curve at each satellite H --------
    h_grid = hist.H_historical_common_EVRF2019_approx.values
    a_grid = hist.A_historical_km2.values
    sat["A_historical_interp_km2"] = np.interp(
        sat.H_Rozumivka_EVRF2019, h_grid, a_grid,
        left=np.nan, right=np.nan)
    sat["abs_diff_km2"] = sat.water_area_km2 - sat.A_historical_interp_km2
    sat["rel_diff_pct"] = sat.abs_diff_km2 / sat.A_historical_interp_km2 * 100

    out_cmp = CFG.TABLES / "satellite_vs_historical_hypsometry.csv"
    sat[["date", "H_Rozumivka_EVRF2019", "water_area_km2", "A_historical_interp_km2",
        "abs_diff_km2", "rel_diff_pct"]].to_csv(out_cmp, index=False)
    print(f"\n-> {out_cmp}")
    in_range = sat[sat.H_Rozumivka_EVRF2019.between(h_grid.min(), h_grid.max())]
    if len(in_range):
        print(f"\nWithin historical H range [{h_grid.min():.2f},{h_grid.max():.2f}] "
              f"(n={len(in_range)}):")
        print(f"  abs_diff_km2: median={in_range.abs_diff_km2.median():+.1f}  "
              f"range=[{in_range.abs_diff_km2.min():+.1f},{in_range.abs_diff_km2.max():+.1f}]")
        print(f"  rel_diff_pct: median={in_range.rel_diff_pct.median():+.2f}%  "
              f"range=[{in_range.rel_diff_pct.min():+.2f}%,{in_range.rel_diff_pct.max():+.2f}%]")
        below = in_range[in_range.H_Rozumivka_EVRF2019 < 15.7]
        above = in_range[in_range.H_Rozumivka_EVRF2019 >= 16.1]
        if len(below):
            print(f"  LOW-level behaviour (H<15.7): median rel_diff = "
                  f"{below.rel_diff_pct.median():+.2f}% (n={len(below)})")
        if len(above):
            print(f"  HIGH-level behaviour (H>=16.1): median rel_diff = "
                  f"{above.rel_diff_pct.median():+.2f}% (n={len(above)})")
        # monotonicity of the satellite series itself
        s2 = sat.dropna(subset=["water_area_km2"]).sort_values("H_Rozumivka_EVRF2019")
        mono = (s2.water_area_km2.diff().dropna() >= -5).mean()  # tolerate small non-monotone noise
        print(f"  satellite A(H) monotonicity: {mono*100:.0f}% of consecutive H-ordered "
              f"steps are non-decreasing (tolerance 5 km2)")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(hist.H_historical_common_EVRF2019_approx, hist.A_historical_km2,
           color=INK, lw=2, marker="o", ms=4,
           label=f"historical Table 19 (BS-77 + {BS_TO_EVRF} m bridge)")
    ax.scatter(sat.H_Rozumivka_EVRF2019, sat.water_area_km2, color=BLUE, s=55,
              zorder=5, label="satellite A(H), this round (n=%d)" % len(sat))
    for r in sat.itertuples():
        ax.annotate(r.date[5:], (r.H_Rozumivka_EVRF2019, r.water_area_km2),
                   fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.axhline(2174.7, color=RED, lw=1, ls=":", label="SA_2 area (2174.7 km2)")
    ax.set_xlabel("H, m EVRF2019 (Rozumivka gauge / historical BS-77+0.185 bridge)")
    ax.set_ylabel("reservoir surface area, km2")
    ax.set_title("P1F v2 -- satellite A(H) vs. historical Table 19 A(H)\n"
                "NOT fit to each other -- independent measurements, compared",
                fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_fig = CFG.FIG / "P1_satellite_vs_historical_AH.png"
    fig.savefig(out_fig, dpi=160)
    print(f"-> {out_fig}")


if __name__ == "__main__":
    main()
