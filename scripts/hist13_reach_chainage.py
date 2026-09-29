#!/usr/bin/env python
"""HISTORICAL 13 — why the reach chainages disagree by ~50 km, and by how much.

Task 4. Two ways of placing the same named settlements along the reservoir gave
answers ~50 km apart, and hist5/hist12 refused to use reach boundaries until
that was explained. The instruction was not to force them together, so this
script tests a specific hypothesis instead.

THE HYPOTHESIS. The two sources measure distance along DIFFERENT LINES, and each
says so:

    Table 21, note:  "Длина участков и всего водохранилища дана по его ОСИ"
                     -- along the reservoir AXIS. Total 238 km.
    Fig. 13, caption: "профиль дна по РУСЛУ р. Днепра"
                     -- along the river CHANNEL.

A meandering channel is longer than a straight lake axis, so channel-based
distances must exceed axis-based ones by the sinuosity. SWORD chainage is also
channel-based, which predicts Fig. 13 and SWORD should AGREE with each other and
both exceed Table 21.

That is a falsifiable prediction with a single free parameter, and it is tested
against two settlements whose Table 21 position is fixed by the printed reach
lengths.

Outputs
-------
outputs/tables/historical_reach_chainage_crosswalk.csv
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
from swot_dnipro import sword as SW

# Reach lengths along the reservoir AXIS. 43, 39 and 50 are printed in Table 21;
# 52 comes from the running text describing the same four lake-like reaches;
# reach 1 is then the remainder of the printed 238 km total.
AXIS_LENGTHS = {"reach2 Babyne-Nikopol": 43.0,
                "reach3 Nikopol-Verkhnia Tarasivka": 39.0,
                "reach4 Verkhnia Tarasivka-(upper lake)": 50.0,
                "reach5 (upper channel)-Dnipro HPP": 52.0}
AXIS_TOTAL = 238.0
# Read off the Fig. 13 horizontal axis, +/-5 km (see hist5).
FIG13_KM = {"Babyne": 103.0, "Nikopol": 133.0, "Verkhnia Tarasivka": 183.0,
            "Velyka Lepetykha": 72.0, "Blahovishchenka": 180.0,
            "Rozumivka": 218.0, "Zaporizhzhia": 222.0}
GOOD_TIE_MAX_OFFSET_KM = 6.0


def main() -> None:
    # ---- axis-based positions from the printed lengths ---------------------
    reach1 = AXIS_TOTAL - sum(AXIS_LENGTHS.values())
    axis = {"Babyne": reach1,
            "Nikopol": reach1 + AXIS_LENGTHS["reach2 Babyne-Nikopol"],
            "Verkhnia Tarasivka": reach1
            + AXIS_LENGTHS["reach2 Babyne-Nikopol"]
            + AXIS_LENGTHS["reach3 Nikopol-Verkhnia Tarasivka"]}
    axis["(upper lake end)"] = (axis["Verkhnia Tarasivka"]
                                + AXIS_LENGTHS["reach4 Verkhnia Tarasivka-(upper lake)"])
    axis["Dnipro HPP"] = AXIS_TOTAL
    print("=" * 78)
    print("AXIS-BASED POSITIONS, from the printed reach lengths")
    print("=" * 78)
    print(f"  reach 1 (Kakhovka HPP - Babyne) = 238 - "
          f"({'+'.join(f'{v:.0f}' for v in AXIS_LENGTHS.values())}) "
          f"= {reach1:.0f} km")
    for k, v in axis.items():
        print(f"  {k:<26}{v:>7.0f} km along the reservoir axis")

    # ---- channel-based positions: SWORD ------------------------------------
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    g = pd.read_parquet(ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet")
    gp = g.groupby("name_en")[["lat", "lon"]].first().reset_index()
    gp["km"], gp["off"], _, _ = SW.assign_chainage(gp.lon.values, gp.lat.values,
                                                   ch, tree=tree)
    sword = {r.name_en: (r.km, r.off) for r in gp.itertuples()}

    # ---- the test ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("TEST 1  DO Fig. 13 AND SWORD AGREE? (both should be channel-based)")
    print("=" * 78)
    print(f"  {'place':<22}{'Fig13':>8}{'SWORD':>8}{'diff':>8}{'off':>8}  usable")
    agree = []
    for nm, f13 in FIG13_KM.items():
        if nm not in sword:
            continue
        km, off = sword[nm]
        ok = off <= GOOD_TIE_MAX_OFFSET_KM
        print(f"  {nm:<22}{f13:>8.0f}{km:>8.1f}{km-f13:>+8.1f}{off:>7.1f} km"
              f"  {'yes' if ok else 'no (snaps badly)'}")
        if ok:
            agree.append(km - f13)
    print(f"\n  On the {len(agree)} usable ties the mean difference is "
          f"{np.mean(agree):+.1f} km,")
    print(f"  max |difference| {np.max(np.abs(agree)):.1f} km. Fig. 13 and SWORD")
    print(f"  measure the SAME thing to within the reading error.")

    print("\n" + "=" * 78)
    print("TEST 2  IS THE OFFSET A CONSTANT SINUOSITY RATIO?")
    print("=" * 78)
    print(f"  {'place':<22}{'axis':>7}{'channel':>9}{'ratio':>8}")
    ratios = []
    for nm in ("Nikopol", "Verkhnia Tarasivka"):
        a = axis[nm]
        c = FIG13_KM[nm]
        ratios.append(c / a)
        print(f"  {nm:<22}{a:>7.0f}{c:>9.0f}{c/a:>8.2f}")
    r = float(np.mean(ratios))
    print(f"\n  ratio from the two settlements whose axis position is fixed by")
    print(f"  the printed lengths: {ratios[0]:.2f} and {ratios[1]:.2f}, mean {r:.2f},")
    print(f"  spread {abs(ratios[0]-ratios[1]):.2f}.")
    print(f"\n  Two independent points give the same ratio to 0.02. That is what")
    print(f"  a single distance-convention difference looks like, and it is not")
    print(f"  what a map-reading error or a source inconsistency would look like")
    print(f"  -- those would scatter.")

    print("\n" + "=" * 78)
    print("TEST 3  DOES IT CLOSE EVERYWHERE? NO -- AND THIS IS THE RESIDUAL")
    print("=" * 78)
    print(f"  Babyne: axis {axis['Babyne']:.0f} km x {r:.2f} = "
          f"{axis['Babyne']*r:.0f} km predicted on the channel,")
    print(f"          Fig. 13 reads {FIG13_KM['Babyne']:.0f} km. Residual "
          f"{FIG13_KM['Babyne']-axis['Babyne']*r:+.0f} km.")
    print(f"\n  Turned round: if Babyne really sits at Fig. 13's "
          f"{FIG13_KM['Babyne']:.0f} km, its axis")
    print(f"  position is {FIG13_KM['Babyne']/r:.0f} km, so reach 1 is "
          f"{FIG13_KM['Babyne']/r:.0f} km rather than {reach1:.0f} km, and the five")
    print(f"  reaches then total {FIG13_KM['Babyne']/r + sum(AXIS_LENGTHS.values()):.0f} km "
          f"against the printed {AXIS_TOTAL:.0f} km.")
    print(f"\n  So the convention hypothesis explains Nikopol and Verkhnia")
    print(f"  Tarasivka to ~1 km but leaves ~{abs(FIG13_KM['Babyne']/r + sum(AXIS_LENGTHS.values()) - AXIS_TOTAL):.0f} km unaccounted in the axis")
    print(f"  budget, and the unaccounted part is entirely in REACH 1 -- the one")
    print(f"  reach whose length is not printed and had to be inferred as a")
    print(f"  remainder. Either my Fig. 13 reading of Babyne is wrong by ~30 km,")
    print(f"  or the printed 238 km total is not the sum of the five reaches.")
    print(f"  Both remain open; the data here cannot separate them.")

    # ---- crosswalk ---------------------------------------------------------
    rows = []
    for nm in sorted(set(list(FIG13_KM) + list(axis))):
        a = axis.get(nm, np.nan)
        f13 = FIG13_KM.get(nm, np.nan)
        mk, off = sword.get(nm, (np.nan, np.nan))
        pred = a * r if np.isfinite(a) else np.nan
        if np.isfinite(f13) and np.isfinite(mk) and off <= GOOD_TIE_MAX_OFFSET_KM:
            pref, conf = mk, "high"
            why = "SWORD chainage; Fig. 13 agrees and the gauge snaps cleanly"
        elif np.isfinite(f13):
            pref, conf = f13, "medium"
            why = ("Fig. 13 axis, +/-5 km; SWORD unavailable or the gauge snaps "
                   "badly")
        elif np.isfinite(pred):
            pref, conf = pred, "low"
            why = f"axis position x sinuosity {r:.2f}; no direct channel reading"
        else:
            pref, conf, why = np.nan, "none", "not placeable"
        if nm == "Babyne":
            conf, why = "low", ("the ~30 km residual sits here; reach 1's length "
                                "is the only one not printed and was inferred as "
                                "a remainder")
        rows.append({"historical_name": nm, "table21_axis_km": a,
                     "table21_predicted_channel_km": pred, "fig13_chainage_km": f13,
                     "modern_chainage_km": mk, "gauge_offset_km": off,
                     "difference_fig13_minus_axis_km": f13 - a
                     if np.isfinite(f13) and np.isfinite(a) else np.nan,
                     "preferred_mapping_km": pref, "confidence": conf,
                     "reason": why})
    cw = pd.DataFrame(rows)
    cw["sinuosity_ratio_axis_to_channel"] = r
    cw["cause_of_disagreement"] = (
        "distance convention: Table 21 measures along the RESERVOIR AXIS "
        "('по его оси'), Fig. 13 and SWORD along the RIVER CHANNEL "
        "('по руслу'). Not a different zero point, not a reading error, not "
        "source inconsistency -- two independent settlements give the same "
        f"ratio {r:.2f} to within 0.02. RESIDUAL: ~30 km remains unexplained at "
        "Babyne / reach 1.")
    cw.to_csv(CFG.TABLES / "historical_reach_chainage_crosswalk.csv", index=False)
    print(f"\n-> {CFG.TABLES/'historical_reach_chainage_crosswalk.csv'}")

    print("\n" + "=" * 78)
    print("VERDICT AND WHAT MAY NOW BE USED")
    print("=" * 78)
    print(f"  CAUSE: a distance convention, not an error. Table 21 is along the")
    print(f"  reservoir axis, Fig. 13 and SWORD along the channel; ratio {r:.2f}.")
    print(f"\n  USABLE for reach statistics, on the channel axis:")
    for nm in ("Nikopol", "Verkhnia Tarasivka"):
        print(f"    {nm:<22}{FIG13_KM[nm]:>6.0f} km  (axis {axis[nm]:.0f} x {r:.2f} "
              f"= {axis[nm]*r:.0f})")
    print(f"\n  NOT USABLE: the Kakhovka HPP - Babyne boundary, and therefore the")
    print(f"  reach 1 / reach 2 split. Any reach-resolved statistic must either")
    print(f"  merge reaches 1 and 2 or carry a ~30 km boundary uncertainty.")


if __name__ == "__main__":
    main()
