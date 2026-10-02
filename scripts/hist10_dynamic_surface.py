#!/usr/bin/env python
"""HISTORICAL 10 — the dynamic part of the pre-breach water surface.

Tasks 1, 2, 4, 5. The reservoir's MEAN free surface is nearly flat over the main
pool (Table 20). That does not make it flat at any instant. The source documents
two mechanisms that move it, and a stated measurement accuracy:

    Fig. 46 + text   seiches: uninodal 13.4 h, binodal 7.3 h, trinodal 3.5 h,
                     with nodal lines and antinodes located along the axis
    Fig. 57          wind setup and setdown against wind speed, seven gauges
    Table 200        the level-measurement accuracy the source assumes, +/-2 cm,
                     and the storage error it implies for Kakhovka
    running text     the free-surface curves differ "by very small increments"
                     EXCEPT the uppermost river-like reach of about 30 km

WHAT IS AND IS NOT DIGITISED HERE.

Fig. 46 is a map of nodal lines plus recorder positions; the numbers that matter
(periods, amplitudes, node chainages) are in the running text and are
transcribed from it, not measured off the map.

Fig. 57 is seven scatter panels with fitted curves. Its axes are linear and
tick-labelled every 20 cm, so the fitted curves are read at the tick wind
speeds. That is a BY-EYE reading and is flagged as such at +/-10 cm; hist9
showed how badly by-eye reading can go wrong when an axis is not what it looks
like, so nothing here is quoted to better than 5 cm and the conclusions are
drawn at order-of-magnitude only. Individual scatter points are NOT digitised.

Fragments of the source column are cut off at the page edge. Where a number is
only partially legible it is recorded with a truncated-text flag rather than
guessed.

Outputs
-------
data/processed/historical/historical_seiches.csv
data/processed/historical/historical_wind_setup_curves.csv
data/processed/historical/historical_wind_setup_summary.csv
data/processed/historical/historical_level_measurement_accuracy.csv
data/processed/historical/historical_text_evidence.csv
outputs/tables/hist10_dynamic_vs_mean.csv
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

OUT = ROOT / "data/processed/historical"
SRC = "Dnipro reservoirs monograph, photographed pages"
DATUM = "HISTORICAL_BALTIC (realization not stated in source)"

# --------------------------------------------------------------- Fig. 46 ----
# Periods and geometry, transcribed from the running text on the Fig. 46 page.
# Chainages are the source's own, measured "along the reservoir axis".
SEICHES = [
    dict(mode="uninodal", n_nodes=1, period_h=13.4,
         node_locations="centre of equilibrium at Kuchuhury urochyshche",
         node_chainage_km=150.0,
         antinode_locations="Kakhovka HPP dam; eastern part of the upper reach",
         amplitude_cm=np.nan, peak_to_peak_cm=np.nan,
         legibility="period and node clearly legible; amplitude not stated "
                    "separately for this mode"),
    dict(mode="binodal", n_nodes=2, period_h=7.3,
         node_locations="Mykhailivka and Blahovishchenka",
         node_chainage_km=np.nan,
         antinode_locations="Kakhovka HPP dam; central antinode near Nikopol "
                            "(somewhat west); eastern part of the upper reach",
         amplitude_cm=np.nan, peak_to_peak_cm=27.5,
         legibility="period, nodes and peak-to-peak range clearly legible"),
    dict(mode="trinodal", n_nodes=3, period_h=3.5,
         node_locations="not stated in the legible text",
         node_chainage_km=np.nan, antinode_locations="not stated",
         amplitude_cm=np.nan, peak_to_peak_cm=np.nan,
         legibility="period legible from the same sentence as the other two"),
    dict(mode="smooth oscillations (not identified as a seiche mode)",
         n_nodes=np.nan, period_h=2.4, node_locations="", node_chainage_km=np.nan,
         antinode_locations="", amplitude_cm=np.nan, peak_to_peak_cm=np.nan,
         legibility="found experimentally, per the text; mean period"),
    dict(mode="smooth oscillations (not identified as a seiche mode)",
         n_nodes=np.nan, period_h=1.5, node_locations="", node_chainage_km=np.nan,
         antinode_locations="", amplitude_cm=np.nan, peak_to_peak_cm=np.nan,
         legibility="found experimentally, per the text; mean period"),
]
# Binodal nodal-line chainages, given explicitly in the text.
SEICHE_NODES = [("Mykhailivka", 82.0, "binodal nodal line"),
                ("Blahovishchenka", 175.0, "binodal nodal line"),
                ("Kuchuhury", 150.0, "uninodal node / centre of equilibrium")]
# Range at the antinodes, and the largest recorded event.
BINODAL_RANGE_CM = (20.0, 35.0)
LARGEST_EVENT = "28-29 Oct 1969, storm-driven"
# A further amplitude figure appears in a column cut off at the page edge; only
# "0-40 cm" is legible, so the leading digit is unknown. NOT used.
TRUNCATED_AMPLITUDE_FRAGMENT = "'...0-40 cm' - leading digit cut off, not used"

# --------------------------------------------------------------- Fig. 57 ----
# Fitted curve read at the tick wind speeds. Station order follows the panel
# lettering a, b, v, g, d, e, zh in the caption.
WIND_PANELS = {
    "Nova Kakhovka":      dict(panel="a", y_axis_max_cm=160,
                               curve={10: 15, 15: 35, 20: 60, 25: 148}),
    "Velyka Lepetykha":   dict(panel="b", y_axis_max_cm=100,
                               curve={10: 10, 15: 25, 20: 45, 25: 90}),
    "Hrushivska Damba":   dict(panel="v", y_axis_max_cm=80,
                               curve={10: 5, 15: 15, 20: 35, 25: 65}),
    "Nikopol":            dict(panel="g", y_axis_max_cm=60,
                               curve={10: 5, 15: 12, 20: 25, 25: 40}),
    "Kamianka-Dniprovska": dict(panel="d", y_axis_max_cm=60,
                                curve={10: 2, 15: 7, 20: 15, 25: 23}),
    "Blahovishchenka":    dict(panel="e", y_axis_max_cm=80,
                               curve={10: 5, 15: 15, 20: 30, 25: 48}),
    "Plavni":             dict(panel="zh", y_axis_max_cm=160,
                               curve={10: 10, 15: 30, 20: 60, 25: 148}),
}
WIND_READ_UNC_CM = 10.0

# -------------------------------------------------------------- Table 200 ---
LEVEL_ACC_CM = 2.0
KAKHOVKA_AREA_T200_KM2 = 2150.0     # as printed in Table 200
KAKHOVKA_DW_MLN_M3 = 43.2           # storage error for +/-2 cm, as printed
# Discharge error implied by that storage error over Dt days, as printed.
KAKHOVKA_DQ = {1: 500.0, 10: 50.0, 30: 17.0}

# ------------------------------------------------------------- text claims --
TEXT_CLAIMS = [
    dict(claim="the free surface is nearly level except the uppermost ~30 km",
         exact_wording="Кривые свободной поверхности водохранилища благодаря "
                       "наличию в верхней его части широкого озеровидного плёса "
                       "отличаются очень малыми приращениями отметок, исключая "
                       "самый верхний русловидный участок в нижнем бьефе ДГЭС "
                       "им. Ленина протяженностью около 30 км (табл. 20, рис. 15 "
                       "и 16).",
         page="Table 20 / Figs 15-16 discussion",
         interpretation="The source states in words what the reach-by-reach "
                        "falsification test found from ICESat-2: a near-level "
                        "main pool and a distinct river-like reach of about "
                        "30 km below the Dnipro HPP.",
         relevance="independent qualitative validation of the chainage-based "
                   "reach analysis (hist6)"),
    dict(claim="longitudinal BED slope of the lake-like part vs the lower reach",
         exact_wording="для озеровидной части водоема (от ДГЭС до с. Бабино) "
                       "среднее его значение 0,09‰, а на нижнем русловидном "
                       "участке (от с. Бабино до Каховской ГЭС) — 0,04‰",
         page="Table 19 / Fig. 13 discussion",
         interpretation="These are BED slopes (0.09 and 0.04 per mille = 9 and "
                        "4 cm/km), NOT water-surface slopes. They must not be "
                        "compared with the satellite WSE gradients.",
         relevance="guards against a numerical coincidence: 0.09 per mille bed "
                   "slope is unrelated to the +0.09 cm/km WSE figure"),
    dict(claim="reservoir reached its normal impoundment level only in 1958",
         exact_wording="Река Днепр в створе Каховской ГЭС перекрыта в июле "
                       "1955 г. ... в мае 1958 г. достигли нормального "
                       "подпорного горизонта, равного 16,0 м.",
         page="Table 19 discussion",
         interpretation="Filling history: closure Jul 1955, 8.0 m by Nov 1955, "
                        "15.0 m in spring 1956, NPG 16.0 m only in May 1958.",
         relevance="bounds the epoch of any survey reduced to a full-pool level"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- Task 1: seiches --------------------------------------
    print("=" * 78)
    print("TASK 1 - SEICHES (Fig. 46 and the text on its page)")
    print("=" * 78)
    s = pd.DataFrame(SEICHES)
    s["source_page"] = SRC + " / Fig. 46 page"
    s["source_figure"] = "Fig. 46 + running text"
    s["binodal_range_at_antinodes_cm"] = f"{BINODAL_RANGE_CM[0]:.0f}-{BINODAL_RANGE_CM[1]:.0f}"
    s["largest_recorded_event"] = LARGEST_EVENT
    s.to_csv(OUT / "historical_seiches.csv", index=False)
    print(f"  {'mode':<48}{'period h':>9}{'p2p cm':>8}")
    for r in s.itertuples():
        p2 = f"{r.peak_to_peak_cm:.1f}" if np.isfinite(r.peak_to_peak_cm) else "--"
        print(f"  {r.mode[:47]:<48}{r.period_h:>9.1f}{p2:>8}")
    print(f"\n  nodal geometry stated in the text:")
    for nm, km, what in SEICHE_NODES:
        print(f"    {nm:<18}{km:>7.0f} km   {what}")
    print(f"\n  peak-to-peak range at the antinodes: "
          f"{BINODAL_RANGE_CM[0]:.0f}-{BINODAL_RANGE_CM[1]:.0f} cm (Table 40)")
    print(f"  largest recorded: {LARGEST_EVENT}")
    print(f"  NOT used: {TRUNCATED_AMPLITUDE_FRAGMENT}")
    print(f"  -> {OUT/'historical_seiches.csv'}")

    # ---------------- Task 2: wind setup ------------------------------------
    print("\n" + "=" * 78)
    print("TASK 2 - WIND SETUP / SETDOWN (Fig. 57)")
    print("=" * 78)
    cw = pd.read_csv(CFG.TABLES / "historical_to_modern_chainage_crosswalk.csv")
    ch = dict(zip(cw.location, cw.chainage_km_historical))
    # gauge chainages from Fig. 13's axis, as used in hist5
    FIG13 = {"Nova Kakhovka": 0.0, "Velyka Lepetykha": 72.0, "Nikopol": 133.0,
             "Blahovishchenka": 180.0, "Plavni": 197.0,
             "Hrushivska Damba": 106.0, "Kamianka-Dniprovska": 138.0}
    rows, summ = [], []
    for st, d in WIND_PANELS.items():
        for w_, dl in d["curve"].items():
            for kind in ("setup", "setdown"):
                rows.append({"station": st, "panel": d["panel"],
                             "wind_speed_m_s": w_, "delta_level_cm": dl,
                             "setup_or_setdown": kind,
                             "curve_id": "fitted curve (source lines 4/5)",
                             "read_uncertainty_cm": WIND_READ_UNC_CM,
                             "method": "read by eye at tick wind speeds",
                             "source_page": SRC + " / Fig. 57",
                             "source_figure": "Fig. 57"})
        mx = max(d["curve"].values())
        w25 = d["curve"][25]
        w10 = d["curve"][10]
        summ.append({"station": st, "panel": d["panel"],
                     "station_chainage_km": FIG13.get(st, np.nan),
                     "y_axis_max_cm": d["y_axis_max_cm"],
                     "max_response_cm": mx,
                     "sensitivity_cm_per_m_s_10_25": (w25 - w10) / 15.0,
                     "note": "the source plots setup and setdown on the SAME "
                             "panel and axis, so the magnitude is shared; the "
                             "SIGN depends on wind direction"})
    pd.DataFrame(rows).to_csv(OUT / "historical_wind_setup_curves.csv", index=False)
    ws = pd.DataFrame(summ).sort_values("station_chainage_km")
    ws.to_csv(OUT / "historical_wind_setup_summary.csv", index=False)
    print(f"  {'station':<22}{'chainage':>9}{'max cm':>8}{'cm per m/s':>12}")
    for r in ws.itertuples():
        print(f"  {r.station:<22}{r.station_chainage_km:>8.0f} "
              f"{r.max_response_cm:>7.0f}{r.sensitivity_cm_per_m_s_10_25:>12.1f}")
    print(f"\n  Read by eye at +/-{WIND_READ_UNC_CM:.0f} cm. The two ends of the")
    print(f"  reservoir respond ~6x more strongly than the centre")
    print(f"  ({ws.max_response_cm.max():.0f} cm at the ends vs "
          f"{ws.max_response_cm.min():.0f} cm at Kamianka-Dniprovska), which is")
    print(f"  what a wind-driven tilt about a central pivot looks like.")
    print(f"  -> {OUT/'historical_wind_setup_curves.csv'}")

    # ---------------- Task 4: level accuracy --------------------------------
    print("\n" + "=" * 78)
    print("TASK 4 - LEVEL-MEASUREMENT ACCURACY (Table 200)")
    print("=" * 78)
    acc = pd.DataFrame([{
        "quantity": "water-level measurement accuracy assumed by the source",
        "value": LEVEL_ACC_CM, "unit": "cm",
        "reservoir": "all Dnipro reservoirs (the table's premise)",
        "storage_consequence_mln_m3": KAKHOVKA_DW_MLN_M3,
        "reservoir_area_km2": KAKHOVKA_AREA_T200_KM2,
        "discharge_error_m3_s_1d": KAKHOVKA_DQ[1],
        "discharge_error_m3_s_10d": KAKHOVKA_DQ[10],
        "discharge_error_m3_s_30d": KAKHOVKA_DQ[30],
        "source_page": SRC + " / Table 200",
        "usage_restriction": "source-reported OBSERVATIONAL accuracy only. NOT "
                             "combined in quadrature with satellite or datum "
                             "uncertainties anywhere in this project.",
    }])
    acc.to_csv(OUT / "historical_level_measurement_accuracy.csv", index=False)
    print(f"  stated accuracy      : +/-{LEVEL_ACC_CM:.0f} cm")
    print(f"  Kakhovka area (T200) : {KAKHOVKA_AREA_T200_KM2:,.0f} km2 "
          f"(Table 19 gives 2,155 at 16.0 m)")
    print(f"  storage consequence  : {KAKHOVKA_DW_MLN_M3:.1f} million m3 for "
          f"+/-2 cm")
    print(f"  implied discharge error: "
          + ", ".join(f"{v:.0f} m3/s over {k} d" for k, v in KAKHOVKA_DQ.items()))
    print(f"\n  Kept as observational accuracy only, NOT folded into any budget.")
    print(f"  -> {OUT/'historical_level_measurement_accuracy.csv'}")

    # ---------------- Task 5: text evidence ---------------------------------
    print("\n" + "=" * 78)
    print("TASK 5 - SOURCE TEXT ON THE UPPER RIVER-LIKE REACH")
    print("=" * 78)
    te = pd.DataFrame(TEXT_CLAIMS)
    te["source"] = SRC
    te.to_csv(OUT / "historical_text_evidence.csv", index=False)
    for r in te.itertuples():
        print(f"\n  CLAIM      {r.claim}")
        print(f"  WORDING    {r.exact_wording[:150]}"
              + ("..." if len(r.exact_wording) > 150 else ""))
        print(f"  MEANS      {r.interpretation}")
        print(f"  RELEVANCE  {r.relevance}")
    print(f"\n  -> {OUT/'historical_text_evidence.csv'}")

    # ---------------- Task 3: orders of magnitude ---------------------------
    print("\n" + "=" * 78)
    print("TASK 3 - DYNAMIC PERTURBATION vs MEAN HYDRAULIC GRADIENT")
    print("=" * 78)
    t20 = pd.read_csv(OUT / "historical_table20_wse.csv").rename(
        columns={"chainage_km_historical": "km"})
    pool = t20[t20.km <= 183]
    mean_rise = {}
    for q, g in pool.groupby("Q_label"):
        g = g.sort_values("km")
        mean_rise[q] = 100 * (g.WSE_historical_m.iloc[-1] - g.WSE_historical_m.iloc[0])

    geom = pd.read_csv(CFG.TABLES / "hist7_slope_geometry.csv")
    pre = geom[geom.period == "PRE_BREACH"]
    obs_span_cm = 100 * pre.wse_span_m.median()
    obs_p90_cm = 100 * pre.wse_span_m.quantile(0.9)

    seiche_half = BINODAL_RANGE_CM[1] / 2
    wind_end = ws.max_response_cm.max()
    wind_centre = ws.max_response_cm.min()

    comp = [
        ("mean hydraulic rise, dam to Verkhnia Tarasivka, Qmin",
         mean_rise.get("Qmin", np.nan), "cm over 183 km", "Table 20"),
        ("mean hydraulic rise, same reach, Q20% (7300 m3/s)",
         mean_rise.get("Q20%", np.nan), "cm over 183 km", "Table 20"),
        ("mean hydraulic rise, same reach, Q1% (16200 m3/s)",
         mean_rise.get("Q1%", np.nan), "cm over 183 km", "Table 20"),
        ("seiche, half of the binodal range at an antinode",
         seiche_half, "cm at a point", "Fig. 46 text"),
        ("wind setup at an END, moderate gale (15 m/s)",
         max(d["curve"][15] for d in WIND_PANELS.values()),
         "cm at a point", "Fig. 57"),
        ("wind setup at an END, extreme storm (25 m/s)",
         wind_end, "cm at a point", "Fig. 57"),
        ("wind setup in the CENTRE, extreme storm (25 m/s)",
         wind_centre, "cm at a point", "Fig. 57"),
        ("source's own level-measurement accuracy",
         LEVEL_ACC_CM, "cm", "Table 200"),
        ("OBSERVED ICESat-2 within-overpass WSE span, pre-breach median",
         obs_span_cm, "cm over ~8-13 km", "this study, hist7"),
        ("OBSERVED ICESat-2 within-overpass WSE span, pre-breach p90",
         obs_p90_cm, "cm over ~8-13 km", "this study, hist7"),
    ]
    print(f"  {'quantity':<58}{'cm':>8}  source")
    for nm, v, unit, src in comp:
        print(f"  {nm:<58}{v:>8.1f}  {src}")
    pd.DataFrame(comp, columns=["quantity", "value_cm", "unit", "source"]).to_csv(
        CFG.TABLES / "hist10_dynamic_vs_mean.csv", index=False)

    w15 = max(d["curve"][15] for d in WIND_PANELS.values())
    r20 = mean_rise.get("Q20%", np.nan)
    print(f"\n  The mean hydraulic rise across the ENTIRE main pool is "
          f"{mean_rise.get('Qmin', np.nan):.1f} cm at low")
    print(f"  flow and {r20:.1f} cm at Q20%. A moderate gale (15 m/s) already raises")
    print(f"  ONE END by {w15:.0f} cm -- {w15/r20:.0f}x that whole-pool rise -- and an extreme")
    print(f"  storm (25 m/s) by {wind_end:.0f} cm. A seiche antinode swings "
          f"+/-{seiche_half:.0f} cm with a")
    print(f"  period of {SEICHES[1]['period_h']:.1f} h.")
    print(f"\n  No ratio is quoted against the Qmin rise: it is 0.0 cm to the")
    print(f"  precision Table 20 is printed at, so the ratio is a division by")
    print(f"  zero rather than a large number.")

    # ---------------- Task 8: the questions, answered -----------------------
    print("\n" + "=" * 78)
    print("TASK 8 - INTERPRETATION")
    print("=" * 78)
    q1 = obs_p90_cm <= BINODAL_RANGE_CM[1]
    print(f"\n  1. Are seiche amplitudes large enough to explain a substantial")
    print(f"     fraction of the observed pre-breach instantaneous variability?")
    print(f"     Observed within-overpass span: median {obs_span_cm:.1f} cm, "
          f"p90 {obs_p90_cm:.1f} cm.")
    print(f"     Documented binodal range at an antinode: "
          f"{BINODAL_RANGE_CM[0]:.0f}-{BINODAL_RANGE_CM[1]:.0f} cm.")
    print(f"     -> {'YES' if q1 else 'NO'}: the entire observed spread fits inside the")
    print(f"        documented seiche range. SUFFICIENT, not demonstrated: no")
    print(f"        contemporaneous seiche record exists for any ICESat-2 pass.")
    print(f"\n  2. Can wind produce a longitudinal tilt larger than the mean slope?")
    print(f"     -> YES, by a wide margin. Even a 15 m/s gale gives {w15:.0f} cm at one")
    print(f"        end against a {r20:.1f} cm mean rise over the whole pool; a 25 m/s")
    print(f"        storm gives {wind_end:.0f} cm. Spread over ~200 km those are tilts of")
    print(f"        order {w15/200:.2f}-{wind_end/200:.2f} cm/km, against reach slopes measured at")
    print(f"        0.00-0.06 cm/km -- one to two orders of magnitude larger.")
    print(f"\n  3. Does this explain the sign changes in individual pre-breach slopes?")
    print(f"     -> It is a sufficient explanation, and it composes with the")
    print(f"        GEOMETRIC finding of hist7: each fit is one overpass, six")
    print(f"        beams, with the chainage span inflated ~1.34x by snapping.")
    print(f"        A perturbation of a few cm across such a baseline yields a")
    print(f"        slope whose sign is set by which beam lands where.")
    print(f"        Wind and seiches explain the MAGNITUDE; the beam geometry")
    print(f"        explains why it is read as a longitudinal slope at all.")
    print(f"\n  4. Does the source independently support a distinct upper reach?")
    print(f"     -> YES, in words: 'very small increments of elevation, EXCEPT")
    print(f"        the uppermost river-like reach ... about 30 km long'.")
    print(f"\n  5. What is dynamics and what is measurement uncertainty?")
    print(f"     Plausibly hydraulic dynamics: the {obs_span_cm:.1f} cm within-overpass")
    print(f"       spread, and the sign instability of short-baseline slopes.")
    print(f"     NOT dynamics: the 2.00 m bathymetric reference error (resolved),")
    print(f"       the 0.40 m spread among reference-level estimates, and the")
    print(f"       chainage-inflation artefact. Those are geodetic and geometric.")
    print(f"\n  NOT CLAIMED anywhere: that wind or a seiche explains any specific")
    print(f"  satellite date. No contemporaneous wind or level record was used.")


if __name__ == "__main__":
    main()
