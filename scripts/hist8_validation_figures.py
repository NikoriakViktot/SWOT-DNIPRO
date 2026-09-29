#!/usr/bin/env python
"""HISTORICAL 8 — the remaining validation figures: V1, V3, V4, V6.

Task 8. These belong in the VALIDATION section, before the main
reservoir-to-river result.

  V1  the vertical-reference framework: what frame each dataset is native to,
      what is applied to bring it to EVRF2019, and how big that step is
  V3  the historical free-surface profiles (Table 20, nine discharges), with
      Fig. 16's two field-measured curves marked as awaiting a square-on scan
  V4  ICESat-2 pre-breach against those profiles -- drawn so that the SCALE
      MISMATCH is the message, because after hist7 an ICESat-2 "longitudinal
      profile" is known to be a single cross-section
  V6  Kherson as the local anchor: gauge, SWOT and ICESat-2 against each other

V4 deliberately does NOT draw a fitted ICESat-2 longitudinal profile. hist7
showed every per-date fit is one overpass, six beams across ~3-10 km of ground,
with the chainage span inflated ~1.34x by snapping onto a meandering centreline.
Drawing those as profiles would assert exactly the thing that is false.

Outputs
-------
outputs/figures/V1_vertical_reference_framework.png
outputs/figures/V3_historical_free_surface_profiles.png
outputs/figures/V4_icesat_pre_vs_historical.png
outputs/figures/V6_kherson_anchor_summary.png
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
PURPLE = "#6b4c8a"
QCOL = {"Qmin": BLUE, "Q20%": GREEN, "Q10%": "#5a8f6a", "Q5%": AMBER,
        "Q3%": "#c08a3a", "Q2%": "#c4762f", "Q1%": "#c1553a",
        "Q0.3%": "#b03a2e", "Q0.1%": RED}


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"{name}.{e}", dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/(name + '.png')}")


# ----------------------------------------------------------------- V1 -------
def v1():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(14, 100); ax.axis("off")

    rows = [
        ("Reservoir & downstream gauges", BLUE,
         "stage in cm above a LOCAL zero\non the historical Baltic system\n"
         "(zeros: +12.00 m reservoir, −5.00 m downstream)",
         "H = zero + stage,  then\n+ Δ EPSG:9902 (ua_2019z)\nsampled PER STATION",
         "+0.172 … +0.216 m\nin this reach"),
        ("ICESat-2  ATL13 / ATL03 / ATL08", GREEN,
         "ellipsoidal height, WGS84 / ITRF2020\nTIDE-FREE permanent tide",
         "+ tide_earth_free2mean\n− ζ EGG2015 quasigeoid\nthen per-station c",
         "free2mean ≈ +0.036 m\nζ ≈ +24 m\nc = −0.13 … −0.22 m"),
        ("SWOT  PIXC / RiverSP", PURPLE,
         "ellipsoidal height, geoid field\nsupplied in the product",
         "h = height − solid_earth_tide\n− load_tide_fes − pole_tide\nNEVER subtract `geoid`",
         "subtracting geoid would\ncost 23.9 m of error"),
        ("Historical bathymetry  (S-57 SOUNDG)", AMBER,
         "DEPTH below an UNRECORDED level\nno VERDAT, no SORDAT, no datum named",
         "reference = УНС 14.00 m\n(working hypothesis)\n+ Δ EPSG:9902 per sounding",
         "datum → 14.19 m EVRF2019\n(14.162 … 14.216)"),
    ]
    y = 84
    for name, col, native, applied, mag in rows:
        ax.add_patch(FancyBboxPatch((1, y - 7), 25, 13, boxstyle="round,pad=0.4",
                                    fc=col, alpha=0.14, ec=col, lw=1.6))
        ax.text(2.2, y + 4.2, name, fontsize=10.2, fontweight="bold", color=col)
        ax.text(2.2, y - 0.3, native, fontsize=8.1, color=INK, va="center")
        ax.text(31, y - 0.3, applied, fontsize=8.4, color=INK, va="center",
                family="monospace")
        ax.text(60, y - 0.3, mag, fontsize=8.1, color=GREY, va="center")
        ax.add_patch(FancyArrowPatch((26.5, y - 0.3), (30, y - 0.3),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=col, lw=1.8))
        ax.add_patch(FancyArrowPatch((56.5, y - 0.3), (59, y - 0.3),
                                     arrowstyle="-", color=GREY, lw=0.9))
        ax.add_patch(FancyArrowPatch((78, y - 0.3), (85, 42),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=col, lw=1.8, alpha=0.65,
                                     connectionstyle="arc3,rad=0.12"))
        y -= 19

    ax.add_patch(FancyBboxPatch((80, 33), 18, 18, boxstyle="round,pad=0.5",
                                fc=INK, alpha=0.90, ec=INK))
    ax.text(89, 45, "EVRF2019", fontsize=15, fontweight="bold", color="white",
            ha="center")
    ax.text(89, 41, "EPSG:9389", fontsize=9, color="white", ha="center")
    ax.text(89, 37.5, "zero-tide\nAmsterdam (NAP)", fontsize=8, color="white",
            ha="center")

    ax.text(1, 96, "V1 · The vertical-reference framework",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(1, 93.2, "Four datasets, four native vertical frames. None can be "
            "compared with another until all four are in one frame — that is the "
            "starting problem, not a detail.",
            fontsize=9.4, color=GREY)
    ax.text(31, 89.2, "APPLIED", fontsize=8.6, fontweight="bold", color=GREY)
    ax.text(60, 89.2, "MAGNITUDE", fontsize=8.6, fontweight="bold", color=GREY)
    ax.text(1, 18, "Kept strictly separate throughout: (1) the official geodetic "
            "transformation, (2) any empirical local alignment, (3) independent "
            "validation.\nThe historical bathymetry reference is a WORKING "
            "HYPOTHESIS — the source names the level but does not say the "
            "soundings were reduced to it.",
            fontsize=8.6, color=RED)
    save(fig, "V1_vertical_reference_framework")


# ----------------------------------------------------------------- V3 -------
def v3(h):
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.8),
                           gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    for q in ["Q0.1%", "Q0.3%", "Q1%", "Q2%", "Q3%", "Q5%", "Q10%", "Q20%", "Qmin"]:
        s = h[h.Q_label == q].sort_values("km")
        qq = int(s.Q_m3_s.iloc[0])
        a.plot(s.km, s.WSE_historical_m, "-o", color=QCOL[q], lw=1.9, ms=4,
               label=f"{q}  Q = {qq:,} m³/s")
    a.axhline(16.0, color=GREY, lw=1, ls=":")
    a.axvspan(180, 245, color=AMBER, alpha=0.10)
    a.text(183, 21.4, "backwater limb", fontsize=8.8, color=AMBER)
    a.set_xlabel("chainage from the dam (km)")
    a.set_ylabel("free-surface elevation (m, historical Baltic — NOT converted)")
    a.legend(fontsize=7.8, ncol=2, loc="upper left")
    a.set_title("Table 20 · calculated free-surface profiles, all nine discharges",
                fontsize=11.5, loc="left")
    a.grid(alpha=0.22)

    a = ax[1]
    a.axis("off")
    a.text(0, 1, (
        "WHAT THIS FIGURE IS, AND IS NOT\n\n"
        "These nine curves are CALCULATED design profiles,\n"
        "tabulated numerically in Table 20 and pinned to\n"
        "16.00 m at the dam.\n\n"
        "Fig. 16 of the source carries FOUR curves:\n"
        "  1  calculated, Q = 8 400 m³/s     → in Table 20\n"
        "  2  FIELD-MEASURED, 22–25 Apr 1970 → not digitised\n"
        "  3  calculated, Q0.1% = 23 800     → in Table 20\n"
        "  4  FIELD-MEASURED, 28–30 Apr 1970 → not digitised\n\n"
        "Curves 2 and 4 are the only field measurements in\n"
        "the source and the only truly independent check on\n"
        "the design profiles. They are NOT digitised here.\n\n"
        "Why: the photograph is taken at an angle and the\n"
        "horizontal gridlines converge, so pixel position is\n"
        "not linear in elevation. Reading curve 3 by eye and\n"
        "checking it against its own Table 20 column shows a\n"
        "systematic error of about +0.2 m in the flat reach.\n"
        "The entire pool signal being tested is 0.00–0.06 m.\n\n"
        "A by-eye digitisation would be worse than no data,\n"
        "so none is supplied. A flatbed scan, or a photograph\n"
        "taken square-on, would make it reliable.\n\n"
        "What Fig. 16 does establish without digitisation:\n"
        "curve 2 lies ON curve 1 over the whole length — the\n"
        "design profile was confirmed in the field in 1970."),
        va="top", ha="left", fontsize=8.7, family="monospace",
        color=INK, transform=a.transAxes)

    fig.suptitle("V3 · The reservoir's documented free surface", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "V3_historical_free_surface_profiles")


# ----------------------------------------------------------------- V4 -------
def v4(h, pre, geom):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.9))

    # (a) the historical envelope with each ICESat-2 overpass drawn at its OWN
    #     baseline length and its OWN measured tilt -- the scale mismatch is
    #     the point, so it is drawn to scale rather than fitted through.
    a = ax[0]
    for q in ["Qmin", "Q20%", "Q10%", "Q5%", "Q1%", "Q0.1%"]:
        s = h[h.Q_label == q].sort_values("km")
        a.plot(s.km, s.WSE_historical_m - 16.0, "-", color=QCOL[q], lw=1.9,
               label=f"historical {q}")
    g = geom[geom.period == "PRE_BREACH"]
    for i, r in enumerate(g.itertuples()):
        c0 = r.chainage_span_km
        mid = 0.0
        a.plot([0, 0], [0, 0], alpha=0)      # keep autoscale sane
    for r in g.itertuples():
        # each overpass: a horizontal bar of its chainage span, height = the WSE
        # span it actually resolved, placed at its reach midpoint
        lo = float(r.reach_km.split("-")[0])
        hi = float(r.reach_km.split("-")[1])
        xc = (lo + hi) / 2
        a.plot([xc - r.chainage_span_km / 2, xc + r.chainage_span_km / 2],
               [r.wse_span_m, r.wse_span_m], color=INK, lw=1.4, alpha=0.5,
               solid_capstyle="butt")
    a.plot([], [], color=INK, lw=1.6, alpha=0.6,
           label="one ICESat-2 overpass:\nbaseline length × WSE span resolved")
    a.set_yscale("symlog", linthresh=0.05)
    a.set_xlabel("chainage from the dam (km)")
    a.set_ylabel("rise above the dam / WSE span (m)")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.22)
    a.set_title("The scale mismatch: overpass baselines vs the signal",
                fontsize=11.2, loc="left")

    # (b) the raw pre-breach cloud, coloured by year, to show that between-date
    #     LEVEL variation dwarfs any longitudinal structure
    a = ax[1]
    pre = pre.copy()
    pre["yr"] = pd.to_datetime(pre.date).dt.year
    for y, c in zip(sorted(pre.yr.unique()),
                    [BLUE, GREEN, AMBER, RED, PURPLE, GREY, "#8a5a12"]):
        s = pre[pre.yr == y]
        a.scatter(s.chain_km, s.wse_m, s=9, color=c, alpha=0.55, lw=0, label=str(y))
    # This axis is EVRF2019, so the historical NPG cannot be drawn at 16.00:
    # 16.00 m historical Baltic is 16.185 m EVRF2019 via the median EPSG:9902
    # offset. Drawing it at 16.00 would be the exact frame confusion this whole
    # figure set exists to prevent.
    NPG_EVRF = 16.00 + 0.185
    a.axhline(NPG_EVRF, color=INK, lw=1.2, ls="--")
    a.text(3, NPG_EVRF + 0.05, f"НПГ = {NPG_EVRF:.2f} m EVRF2019 "
           f"(16.00 m historical Baltic + Δ EPSG:9902)", fontsize=8, color=INK)
    a.set_xlabel("chainage from the dam (km)")
    a.set_ylabel("ICESat-2 WSE (m, EVRF2019)")
    a.legend(fontsize=8, ncol=2, title="pass year", title_fontsize=8)
    a.grid(alpha=0.22)
    a.set_title("Between-date level variation dominates the cloud",
                fontsize=11.2, loc="left")

    fig.suptitle("V4 · ICESat-2 pre-breach against the documented profiles   ·   "
                 "no fitted ICESat-2 longitudinal profile is drawn, and why",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "V4_icesat_pre_vs_historical")


# ----------------------------------------------------------------- V6 -------
def v6():
    pilot = pd.read_csv(CFG.TABLES / "kherson_swot_icesat_sensor_comparison_pilot.csv")
    anch = pd.read_csv(CFG.TABLES / "part4_anchor_summary.csv")
    val = pd.read_csv(CFG.TABLES / "swot_validation_against_gauge_and_icesat.csv")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2),
                           gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))

    a = ax[0]
    r = pilot.iloc[0]
    items = [("SWOT − gauge", r.median_R_SWOT_m, r.R_SWOT_ci95_low,
              r.R_SWOT_ci95_high, int(r.n_SWOT_dates), PURPLE),
             ("ICESat-2 − gauge", r.median_R_ICESat_m, r.R_ICESat_ci95_low,
              r.R_ICESat_ci95_high, int(r.n_ICESat_dateRGT), GREEN)]
    for i, (nm, v, lo, hi, n, c) in enumerate(items):
        a.plot([lo, hi], [i, i], color=c, lw=8, alpha=0.4, solid_capstyle="butt")
        a.plot(v, i, "o", color=c, ms=13)
        a.annotate(f"{v:+.3f} m\nn={n}", (v, i), textcoords="offset points",
                   xytext=(0, 16), ha="center", fontsize=9, color=c,
                   fontweight="bold")
    a.axvline(0, color=INK, lw=1.6, ls="--")
    a.set_yticks(range(len(items))); a.set_yticklabels([i[0] for i in items],
                                                       fontsize=10)
    a.set_ylim(-0.6, 1.7); a.set_xlabel("residual against the Kherson gauge (m)")
    a.grid(axis="x", alpha=0.25)
    a.set_title("Kherson pilot: each sensor against the gauge",
                fontsize=11.2, loc="left")
    a.text(0.02, 0.03, "pilot subset, tide-harmonised, matched dates only",
           transform=a.transAxes, fontsize=7.6, color=GREY, va="bottom")

    a = ax[1]
    a.axhline(0, color=INK, lw=1.4, ls="--")
    for rad, grp in val.dropna(subset=["R_swot_minus_gauge_m"]).groupby("radius_km"):
        a.scatter([rad] * len(grp), grp.R_swot_minus_gauge_m, s=42, color=PURPLE,
                  alpha=0.6, lw=0)
        a.plot(rad, grp.R_swot_minus_gauge_m.median(), "_", color=INK, ms=26, mew=2.5)
    a.set_xscale("log")
    a.set_xticks(sorted(val.radius_km.unique()))
    a.set_xticklabels([f"{v:g}" for v in sorted(val.radius_km.unique())])
    a.set_xlabel("PIXC averaging radius (km)")
    a.set_ylabel("SWOT − gauge (m)")
    a.grid(alpha=0.25)
    # Report the swing rather than calling it "stable": the medians move 0.08 m
    # across the radii, which is not nothing at this magnitude.
    med_by_r = (val.dropna(subset=["R_swot_minus_gauge_m"])
                .groupby("radius_km").R_swot_minus_gauge_m.median())
    swing = float(med_by_r.max() - med_by_r.min())
    a.set_title(f"Radius sensitivity: medians swing {swing:.3f} m",
                fontsize=11.2, loc="left")
    a.text(0.02, 0.03, "per-granule table, all dates and tiles —\na DIFFERENT "
           "subset from the left panel", transform=a.transAxes, fontsize=7.6,
           color=GREY, va="bottom")

    a = ax[2]
    a.axis("off")
    d = r.Delta_sensor_m
    a.text(0, 1, (
        "V6 · KHERSON AS THE LOCAL ANCHOR\n\n"
        f"SWOT − gauge      {r.median_R_SWOT_m:+.3f} m\n"
        f"                  [{r.R_SWOT_ci95_low:+.3f}, {r.R_SWOT_ci95_high:+.3f}]"
        f"  NMAD {r.NMAD_R_SWOT_m:.3f}\n"
        f"ICESat-2 − gauge  {r.median_R_ICESat_m:+.3f} m\n"
        f"                  [{r.R_ICESat_ci95_low:+.3f}, {r.R_ICESat_ci95_high:+.3f}]"
        f"  NMAD {r.NMAD_R_ICESat_m:.3f}\n\n"
        f"Δ sensor          {d:+.3f} m\n"
        f"  if tides NOT removed  "
        f"{r.Delta_sensor_if_tides_NOT_removed_m:+.3f} m\n\n"
        "The gauge zero (−5.00 m) and the EPSG:9902 step\n"
        "CANCEL in these differences, so Δ sensor is\n"
        "independent of the datum question entirely.\n\n"
        "ANCHOR STATUS (part4)\n"
        + "\n".join(
            f"  {x.name:<10} c = {x.c_pre_2km_m:+.3f} "
            f"[{x.c_ci_lo:+.3f}, {x.c_ci_hi:+.3f}], n={int(x.n_matchups)}"
            for x in anch.itertuples())
        + "\n\nNOT ESTABLISHED\n"
        "  post-breach c at either station\n"
        "  ICESat-2 never comes within 13 km of Kherson\n"
        "  Rozumivka has no SWOT coverage\n\n"
        "The Kherson correction is NOT transferable and is\n"
        "never applied outside its own reach."),
        va="top", ha="left", fontsize=8.6, family="monospace", color=INK,
        transform=a.transAxes)

    fig.suptitle("V6 · Local gauge anchor and cross-sensor agreement at Kherson",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "V6_kherson_anchor_summary")


def main() -> None:
    h = pd.read_csv(ROOT / "data/processed/historical/historical_table20_wse.csv")
    h = h.rename(columns={"chainage_km_historical": "km"})
    p = pd.read_csv(ROOT / "outputs/tables/kakhovka_longitudinal_profiles.csv")
    pre = p[p.period == "PRE_BREACH"]
    geom = pd.read_csv(CFG.TABLES / "hist7_slope_geometry.csv")

    v1()
    v3(h)
    v4(h, pre, geom)
    v6()
    print("\nV1, V3, V4, V6 built. Task 8 complete except for the Fig. 16 field")
    print("curves, which stay blocked on a square-on scan of that page.")


if __name__ == "__main__":
    main()
