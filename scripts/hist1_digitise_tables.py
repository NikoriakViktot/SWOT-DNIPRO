#!/usr/bin/env python
"""HISTORICAL 1 — digitise the Kakhovka reservoir design/field tables.

Source: a printed monograph on the Dnipro reservoirs (photographed pages).
    Table 19  water level -> surface area -> volume, by reach and total
    Table 20  longitudinal free-surface elevations for nine discharges
    Table 21  morphometric characteristics of the five reservoir reaches
    Fig. 13   longitudinal profile (bed, free surface, gauge positions)
    Fig. 16   calculated and OBSERVED longitudinal free-surface curves

These are transcribed from photographs, which is the single largest error
source in this whole chain. Transcription is therefore verified against the
internal arithmetic of the tables themselves -- reach volumes must sum to the
total, area at NPG must equal area at GMO plus the drying area, useful volume
must equal NPG minus GMO -- and the script REFUSES TO WRITE if any of that
fails. This caught a real inconsistency: reach 2 area 582 vs the 532 its own
arithmetic requires -- a typo in the SOURCE, not in the transcription (see the
Table 21 block below).

VERTICAL DATUM: the source predates EVRF2019 and states no datum explicitly.
Soviet reservoir design documents of this period use the Baltic 1977 system
(or its predecessor BS-42, which differs by a few centimetres). It is recorded
here as HISTORICAL_BALTIC and is NOT converted. Do not compare these numbers
with EVRF2019 heights until that is resolved -- see hist3.

Outputs
-------
data/historical/historical_level_area_volume.csv
data/historical/historical_reservoir_reaches.csv
data/historical/historical_longitudinal_wse.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

OUT = ROOT / "data/historical"
DATUM = "HISTORICAL_BALTIC (BS-42/BS-77, not stated in source, NOT converted)"
SRC = "Dnipro reservoirs monograph, photographed pages (Tables 19-21, Figs 13-16)"

# Design levels named in the Table 19 notes. These are the reason the table
# matters here: the reservoir has a documented low navigation level.
DESIGN_LEVELS = {
    18.0: "",
    17.5: "NUF - highest forced level",
    16.0: "NPG - normal impoundment level",
    14.0: "UNS - navigation drawdown level",
    12.7: "GMO - dead-volume level",
}

# ---------------------------------------------------------------- Table 19 --
# level_m, area_km2, V1..V5, V_total   (NaN where the photograph is unreadable)
T19 = [
    (18.0, 2222, None, None, None, None, None, 22.57),
    (17.5, 2205, None, None, None, None, None, 21.46),
    (17.0, 2188, None, None, None, None, None, 20.36),
    (16.5, 2172, 6.90, 5.64, 2.79, 3.56, 0.38, 19.27),
    (16.0, 2155, 6.65, 5.38, 2.60, 3.21, 0.35, 18.19),
    (15.5, 2133, 6.40, 5.12, 2.42, 2.86, 0.32, 17.12),
    (15.0, 2110, 6.16, 4.85, 2.24, 2.52, 0.29, 16.06),
    (14.5, 2077, 5.92, 4.59, 2.06, 2.18, 0.26, 15.01),
    (14.0, 2041, 5.68, 4.32, 1.88, 1.87, 0.23, 13.98),
    (13.5, 1984, 5.44, 4.06, 1.71, 1.57, 0.20, 12.98),
    (13.0, 1916, 5.20, 3.80, 1.55, 1.27, 0.18, 12.00),
    (12.7, 1876, 5.06, 3.65, 1.45, 1.12, 0.16, 11.44),
    (12.0, 1774, 4.73, 3.29, 1.22, 0.78, 0.13, 10.15),
    (11.5, 1693, 4.50, 3.03, 1.06, 0.58, 0.11, 9.28),
    (11.0, 1613, 4.28, 2.78, 0.92, 0.40, 0.09, 8.46),
    (10.5, 1528, 4.04, 2.52, 0.78, 0.26, 0.07, 7.67),
    (10.0, 1443, 3.82, 2.27, 0.65, 0.15, 0.06, 6.95),
]

# ---------------------------------------------------------------- Table 21 --
# Reach 2's NPG area is PRINTED as 582 -- confirmed on a later square-on scan,
# so this is not a misreading on my part, as I first recorded it. The SOURCE is
# internally inconsistent there: 518 (at GMO) + 14 (drying) = 532, the NPG
# column then sums to 2205 instead of the stated 2155, and 532 is the only value
# consistent with BOTH the row sum and the column total. The other two
# candidates fail: making the drying area 64 breaks the drying total, and making
# the GMO area 568 breaks the GMO total. So 532 is used and 582 is treated as a
# typographical error in the source.
T21 = {
    "reach_id": [1, 2, 3, 4, 5],
    "extent": ["Kakhovka HPP - Babyne", "Babyne - Nikopol",
               "Nikopol - Verkhnia Tarasivka", "Verkhnia Tarasivka - Blahovishchenka",
               "Blahovishchenka - Dnipro HPP"],
    "width_typical_km": ["4-6", "11-15", "7-11", "15-20", "1-2"],
    "width_mean_km": [5.4, 12.4, 9.3, np.nan, 1.4],
    "width_max_km": [7.2, 18.7, 13.0, 24.2, 2.3],
    "width_min_km": [3.0, np.nan, 4.0, np.nan, np.nan],
    "depth_typical_m": ["12-16", "9-12", "6-9", "3-6", "4-7"],
    "depth_mean_m": [13.4, 10.1, 7.2, 4.6, 4.9],
    "depth_max_m": [36, 25, 22, 13, 18],
    "area_npg_km2": [495, 532, 363, 693, 72],
    "area_gmo_km2": [469, 518, 329, 513, 47],
    "area_drying_km2": [26, 14, 34, 180, 25],
    "volume_npg_km3": [6.65, 5.38, 2.60, 3.21, 0.35],
    "volume_gmo_km3": [5.06, 3.65, 1.45, 1.12, 0.16],
    "volume_useful_km3": [1.59, 1.73, 1.15, 2.09, 0.19],
}
T21_TOTAL = dict(width_mean_km=9.0, width_max_km=24.2, width_min_km=3.0,
                 depth_mean_m=8.5, depth_max_m=36, area_npg_km2=2155,
                 area_gmo_km2=1876, area_drying_km2=279, volume_npg_km3=18.19,
                 volume_gmo_km3=11.44, volume_useful_km3=6.75, length_km=238)

# ---------------------------------------------------------------- Table 20 --
# "Coordinates of the free-surface curves of the Kakhovka reservoir at the
# normal impoundment horizon at the dam, for various discharges."
#
# 135 numbers off a rotated photograph, and unlike Tables 19/21 this one has NO
# internal checksum. It is therefore marked lower confidence and validated only
# by monotonicity, which is a weak test: it catches gross errors, not a 16.23
# misread as 16.25. Treat individual cells as approximate; the PROFILE SHAPE is
# what this table is used for.
T20_Q = [23800, 19200, 16200, 14200, 13100, 11400, 9400, 7300, 800]
T20_LABEL = ["Q0.1%", "Q0.3%", "Q1%", "Q2%", "Q3%", "Q5%", "Q10%", "Q20%", "Qmin"]
# name, chainage_km (None = not yet established), row of 9 elevations
T20 = [
    ("Kakhovka HPP",        0.0,  [16.00, 16.00, 16.00, 16.00, 16.00, 16.00, 16.00, 16.00, 16.00]),
    ("Mykhailivka",         None, [16.02, 16.01, 16.00, 16.00, 16.00, 16.00, 16.00, 16.00, 16.00]),
    ("Hornostaivka",        None, [16.10, 16.06, 16.04, 16.03, 16.02, 16.01, 16.01, 16.00, 16.00]),
    ("Kachkarivka",         None, [16.13, 16.09, 16.07, 16.05, 16.04, 16.02, 16.01, 16.00, 16.00]),
    ("km 79",               79.0, [16.23, 16.15, 16.11, 16.09, 16.08, 16.05, 16.03, 16.01, 16.00]),
    ("Otmet",               None, [16.30, 16.20, 16.15, 16.12, 16.10, 16.07, 16.04, 16.02, 16.00]),
    ("km 117",              117.0,[16.39, 16.25, 16.19, 16.16, 16.14, 16.10, 16.05, 16.02, 16.00]),
    ("Nikopol",             None, [16.47, 16.30, 16.23, 16.19, 16.18, 16.12, 16.07, 16.03, 16.00]),
    ("Illinske",            None, [16.67, 16.42, 16.30, 16.25, 16.23, 16.16, 16.09, 16.04, 16.00]),
    ("Verkhnia Tarasivka",  None, [16.91, 16.61, 16.42, 16.32, 16.30, 16.22, 16.13, 16.06, 16.00]),
    ("Bilenke",             None, [17.28, 16.89, 16.67, 16.48, 16.42, 16.33, 16.21, 16.09, 16.01]),
    ("Lysa Hora",           None, [17.87, 17.39, 17.06, 16.81, 16.72, 16.58, 16.36, 16.16, 16.01]),
    ("Rozumivka",           None, [18.68, 18.32, 18.09, 17.94, 17.85, 17.70, 17.41, 16.87, 16.02]),
    ("Zaporizhzhia",        None, [19.66, 19.25, 18.94, 18.73, 18.60, 18.38, 18.04, 17.55, 16.02]),
    ("Dnipro HPP",          None, [22.23, 21.42, 20.80, 20.36, 20.12, 19.74, 19.24, 18.62, 16.04]),
]


def check(cond, msg):
    if not cond:
        raise SystemExit(f"TRANSCRIPTION CHECK FAILED: {msg}")
    print(f"  ok  {msg}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- Table 19 -------------------------------------------
    print("=== Table 19: level / area / volume ===")
    t19 = pd.DataFrame(T19, columns=["water_level_m", "surface_area_km2"] +
                       [f"volume_reach{i}_km3" for i in range(1, 6)] +
                       ["volume_total_km3"])
    vc = [f"volume_reach{i}_km3" for i in range(1, 6)]
    have = t19[vc].notna().all(axis=1)
    resid = (t19.loc[have, vc].sum(axis=1) - t19.loc[have, "volume_total_km3"]).abs()
    check(resid.max() <= 0.011,
          f"reach volumes sum to the total on all {have.sum()} complete rows "
          f"(max |residual| {resid.max():.3f} km3)")
    check(t19.water_level_m.is_monotonic_decreasing, "levels are ordered")
    check(t19.surface_area_km2.is_monotonic_decreasing,
          "area decreases with level")
    check(t19.volume_total_km3.is_monotonic_decreasing,
          "volume decreases with level")
    # dV/dH must be bracketed by the surface areas at the two ends of the step
    d = t19.sort_values("water_level_m")
    dv = np.diff(d.volume_total_km3.values) * 1e9
    dh = np.diff(d.water_level_m.values)
    a_lo, a_hi = d.surface_area_km2.values[:-1] * 1e6, d.surface_area_km2.values[1:] * 1e6
    implied = dv / dh
    check(bool(np.all(implied >= a_lo * 0.97) and np.all(implied <= a_hi * 1.03)),
          "dV/dH is consistent with the tabulated surface areas")

    t19["design_level"] = t19.water_level_m.map(DESIGN_LEVELS).fillna("")
    t19["vertical_datum"] = DATUM
    t19["source"] = SRC + " / Table 19"
    t19.to_csv(OUT / "historical_level_area_volume.csv", index=False)
    print(f"  -> {OUT/'historical_level_area_volume.csv'}  ({len(t19)} rows)")
    print(f"\n  DESIGN LEVELS RECORDED IN THE SOURCE:")
    for lv, nm in sorted(DESIGN_LEVELS.items(), reverse=True):
        if nm:
            v = float(t19.loc[t19.water_level_m == lv, "volume_total_km3"].iloc[0])
            a = float(t19.loc[t19.water_level_m == lv, "surface_area_km2"].iloc[0])
            print(f"    {lv:5.1f} m  {nm:<34} {a:>6,.0f} km2  {v:>6.2f} km3")

    # ---------------- Table 21 -------------------------------------------
    print("\n=== Table 21: morphometric reaches ===")
    t21 = pd.DataFrame(T21)
    check(bool((t21.area_gmo_km2 + t21.area_drying_km2 == t21.area_npg_km2).all()),
          "area at NPG = area at GMO + drying area, every reach")
    check(abs(t21.area_npg_km2.sum() - T21_TOTAL["area_npg_km2"]) <= 1,
          f"reach areas at NPG sum to {T21_TOTAL['area_npg_km2']:,} km2")
    check(abs(t21.area_gmo_km2.sum() - T21_TOTAL["area_gmo_km2"]) <= 1,
          f"reach areas at GMO sum to {T21_TOTAL['area_gmo_km2']:,} km2")
    check(abs(t21.area_drying_km2.sum() - T21_TOTAL["area_drying_km2"]) <= 1,
          f"drying areas sum to {T21_TOTAL['area_drying_km2']:,} km2")
    check(bool((( t21.volume_npg_km3 - t21.volume_gmo_km3
                 - t21.volume_useful_km3).abs() <= 0.011).all()),
          "useful volume = NPG - GMO, every reach")
    for c, tot in [("volume_npg_km3", 18.19), ("volume_gmo_km3", 11.44),
                   ("volume_useful_km3", 6.75)]:
        check(abs(t21[c].sum() - tot) <= 0.011, f"{c} sums to {tot} km3")
    # cross-table: Table 21 at NPG/GMO must reproduce Table 19 at 16.0/12.7
    for lv, col in [(16.0, "volume_npg_km3"), (12.7, "volume_gmo_km3")]:
        t19v = float(t19.loc[t19.water_level_m == lv, "volume_total_km3"].iloc[0])
        check(abs(t21[col].sum() - t19v) <= 0.011,
              f"Table 21 {col} agrees with Table 19 at {lv} m ({t19v} km3)")
    for lv, col in [(16.0, "area_npg_km2"), (12.7, "area_gmo_km2")]:
        t19a = float(t19.loc[t19.water_level_m == lv, "surface_area_km2"].iloc[0])
        check(abs(t21[col].sum() - t19a) <= 1,
              f"Table 21 {col} agrees with Table 19 at {lv} m ({t19a:,.0f} km2)")

    t21["vertical_datum"] = DATUM
    t21["source"] = SRC + " / Table 21"
    t21["length_km"] = np.nan          # per-reach lengths not legible; total 238
    t21["total_length_km_all_reaches"] = T21_TOTAL["length_km"]
    t21.to_csv(OUT / "historical_reservoir_reaches.csv", index=False)
    print(f"  -> {OUT/'historical_reservoir_reaches.csv'}  ({len(t21)} reaches)")

    # ---------------- Table 20 -------------------------------------------
    print("\n=== Table 20: longitudinal free-surface elevations ===")
    rows = []
    for order, (name, ch, vals) in enumerate(T20):
        for q, lab, h in zip(T20_Q, T20_LABEL, vals):
            rows.append({"location": name, "order_from_dam": order,
                         "chainage_km": ch, "Q_m3_s": q, "Q_label": lab,
                         "H_dam_m": 16.00, "WSE_historical_m": h,
                         "curve_type": "design discharge profile",
                         "observed_or_calculated": "calculated",
                         "source_table": "Table 20", "source_figure": "",
                         "vertical_datum": DATUM, "transcription_confidence":
                         "medium - no internal checksum available"})
    t20 = pd.DataFrame(rows)

    piv = t20.pivot(index="order_from_dam", columns="Q_label",
                    values="WSE_historical_m")[T20_LABEL]
    bad = [(T20[i][0], c) for i in piv.index for c in T20_LABEL
           if i > 0 and piv.loc[i, c] < piv.loc[i - 1, c] - 1e-9]
    check(not bad, f"elevation never decreases going upstream, all 9 discharges"
          + (f" -- offenders: {bad}" if bad else ""))
    dec = [(T20[i][0], T20_LABEL[j]) for i in piv.index
           for j in range(len(T20_LABEL) - 1)
           if piv.loc[i, T20_LABEL[j]] < piv.loc[i, T20_LABEL[j + 1]] - 1e-9]
    check(not dec, "at every point, backwater rises with discharge"
          + (f" -- offenders: {dec}" if dec else ""))
    check(bool((piv.loc[0] == 16.00).all()),
          "every profile is pinned to 16.00 m at the dam, as the table states")

    t20.to_csv(OUT / "historical_longitudinal_wse.csv", index=False)
    print(f"  -> {OUT/'historical_longitudinal_wse.csv'}  ({len(t20)} rows, "
          f"{t20.location.nunique()} locations x {len(T20_Q)} discharges)")

    # ---------------- what the profile actually looks like ----------------
    print("\n=== the historical longitudinal profile, in one view ===")
    print(f"{'location':<22}" + "".join(f"{l:>8}" for l in T20_LABEL))
    for i, (name, ch, _) in enumerate(T20):
        print(f"{name:<22}" + "".join(f"{piv.loc[i, l]:>8.2f}" for l in T20_LABEL))

    print("\n  rise from the dam to the head of the reservoir:")
    for lab in T20_LABEL:
        print(f"    {lab:<7} Q={dict(zip(T20_LABEL,T20_Q))[lab]:>6,} m3/s  ->  "
              f"Zaporizhzhia {piv.loc[13, lab] - 16.0:+.2f} m, "
              f"Verkhnia Tarasivka {piv.loc[9, lab] - 16.0:+.2f} m")
    print("\n  So the reservoir was NOT a level plane. Its longitudinal shape is a")
    print("  function of discharge: at low flow it is flat to 2-4 cm over 240 km,")
    print("  at high flow the upper reach stands metres above the dam. Any")
    print("  comparison with an ICESat-2 profile must condition on discharge.")


if __name__ == "__main__":
    main()
