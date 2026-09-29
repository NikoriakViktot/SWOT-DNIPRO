#!/usr/bin/env python
"""Render every publication figure from outputs/figure_data/. No science here.

Run:  python scripts/make_figures.py [FigNN ...]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import (
    C, MARKERS, bootstrap_ci, graticule, nmad, north_arrow, panel_label, save,
    scale_bar, use_style, zero_line,
)

use_style()
FD, FIG = CFG.FIGDATA, CFG.FIGURES if hasattr(CFG, "FIGURES") else CFG.FIG
M = CFG.CRS_METRIC


def _read(name, **kw):
    return pd.read_csv(FD / name, **kw)


def _gauges_gdf():
    rows = [(i, n, lo, la, "reservoir") for i, n, lo, la in CFG.RESERVOIR_GAUGES]
    i, n, lo, la = CFG.KHERSON_GAUGE
    rows.append((i, n, lo, la, "downstream"))
    df = pd.DataFrame(rows, columns=["id", "name", "lon", "lat", "kind"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                            crs=CFG.CRS_GEOG).to_crs(M)


# =========================================================================== #
# Fig 01 — study area / data coverage                                         #
# =========================================================================== #
def fig01():
    res = gpd.read_file(CFG.RESERVOIR_GEOJSON).to_crs(M)
    g = _gauges_gdf()
    dam = gpd.GeoDataFrame(geometry=gpd.points_from_xy([CFG.KAKHOVKA_DAM[0]], [CFG.KAKHOVKA_DAM[1]]),
                           crs=CFG.CRS_GEOG).to_crs(M)

    # ICESat-2 track locations actually used
    tracks = []
    for p, lbl in [(CFG.KAKHOVKA_ATL13, "reservoir"), (CFG.KHERSON_ATL13, "downstream")]:
        d = pd.read_parquet(p)
        d = d.dropna(subset=["lat_mean", "lon_mean"])
        tracks.append(gpd.GeoDataFrame(d.assign(zone=lbl),
                      geometry=gpd.points_from_xy(d.lon_mean, d.lat_mean),
                      crs=CFG.CRS_GEOG).to_crs(M))
    trk = pd.concat(tracks, ignore_index=True)

    # SWOT PIXC pilot footprint (actual pixel extent of the downloaded tile)
    px = pd.read_csv(FD / "SFig11_pixc_pixels_20230405_allclasses.csv.gz")
    pxg = gpd.GeoDataFrame(geometry=gpd.points_from_xy(px.lon, px.lat), crs=CFG.CRS_GEOG).to_crs(M)
    cov = _read("Fig12_coverage_timeline.csv")

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    res.plot(ax=ax, fc=C["reservoir"], ec="#5d92a8", lw=0.6, alpha=0.85, zorder=1)
    ax.scatter(pxg.geometry.x, pxg.geometry.y, s=0.12, c=C["swot"], alpha=0.10,
               lw=0, zorder=2, rasterized=True)
    sub = trk[trk.zone == "reservoir"]
    ax.scatter(sub.geometry.x, sub.geometry.y, s=1.4, c=C["icesat"], alpha=0.28, lw=0,
               zorder=3, rasterized=True)
    sub = trk[trk.zone == "downstream"]
    ax.scatter(sub.geometry.x, sub.geometry.y, s=1.4, c=C["icesat"], alpha=0.28, lw=0,
               zorder=3, rasterized=True)

    gr = g[g.kind == "reservoir"]
    ax.scatter(gr.geometry.x, gr.geometry.y, s=52, c=C["gauge"], marker=MARKERS["gauge"],
               ec="white", lw=0.8, zorder=6, label="Reservoir gauges (n=6, series ends 2021)")
    gk = g[g.kind == "downstream"]
    ax.scatter(gk.geometry.x, gk.geometry.y, s=95, c=C["accent"], marker=MARKERS["gauge"],
               ec="black", lw=0.9, zorder=7, label="Kherson gauge 80805 (pilot site, 2023 daily)")
    ax.scatter(dam.geometry.x, dam.geometry.y, s=175, c=C["dam"], marker=MARKERS["dam"],
               ec="white", lw=0.7, zorder=7, label="Kakhovka dam (breach 2023-06-06)")

    for _, r in g.iterrows():
        off = 5200 if r["kind"] == "reservoir" else -9000
        ax.annotate(r["name"], (r.geometry.x, r.geometry.y), xytext=(6, off / 1000),
                    textcoords="offset points", fontsize=6.6, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72))

    h = [Line2D([], [], ls="none", marker="s", ms=6, mfc=C["reservoir"], mec="#5d92a8",
                label="Former Kakhovka Reservoir (2 175 km²)"),
         Line2D([], [], ls="none", marker="o", ms=4, color=C["swot"], alpha=0.6,
                label=f"SWOT PIXC pilot swath (tile 001_237R, {int(cov.downloaded.sum())}/57 dates)"),
         Line2D([], [], ls="none", marker="o", ms=4, color=C["icesat"], alpha=0.6,
                label="ICESat-2 ATL13 beam-pass locations"),
         Line2D([], [], ls="none", marker="s", ms=6, color=C["gauge"],
                label="Reservoir gauges (n=6, series ends 2021)"),
         Line2D([], [], ls="none", marker="s", ms=7, color=C["accent"], mec="black",
                label="Kherson gauge 80805 (pilot site)"),
         Line2D([], [], ls="none", marker="*", ms=11, color=C["dam"],
                label="Kakhovka dam (breach 2023-06-06)")]
    ax.legend(handles=h, loc="upper left", fontsize=6.8, ncol=1)

    ax.set_aspect("equal")
    graticule(ax, M, step=0.5)
    scale_bar(ax, loc="lower right")
    north_arrow(ax, loc=(0.045, 0.42))
    ax.set_title("Observation systems over the former Kakhovka Reservoir and lower Dnipro")
    ax.set_xlabel(f"CRS: {M} (WGS 84 / UTM 36N)")

    # inset: Ukraine locator
    ins = ax.inset_axes([0.735, 0.035, 0.25, 0.30])
    world = gpd.GeoDataFrame(geometry=[], crs=CFG.CRS_GEOG)
    ins.add_patch(mpatches.Rectangle((22.1, 44.3), 18.1, 8.2, fc="#eef1f2", ec="#8b9497", lw=0.6))
    b = res.to_crs(CFG.CRS_GEOG).total_bounds
    ins.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                     fc=C["bad"], ec=C["bad"], lw=0.8, alpha=0.85))
    ins.set_xlim(22, 40.3); ins.set_ylim(44.2, 52.5)
    ins.set_xticks([24, 32, 40]); ins.set_yticks([46, 52])
    ins.set_xticklabels(["24°E", "32°E", "40°E"], fontsize=5.2)
    ins.set_yticklabels(["46°N", "52°N"], fontsize=5.2)
    ins.tick_params(length=1.6, pad=1)
    ins.grid(False)
    ins.set_title("locator (Ukraine extent)", fontsize=5.8, pad=1.5)
    save(fig, "Fig01_study_area_data_coverage", FIG)

    # SFig01 — track geometry detail around Kherson
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(pxg.geometry.x, pxg.geometry.y, s=0.5, c=C["swot"], alpha=0.16, lw=0, rasterized=True)
    sub = trk[trk.zone == "downstream"]
    ax.scatter(sub.geometry.x, sub.geometry.y, s=6, c=C["icesat"], alpha=0.5, lw=0, rasterized=True)
    ax.scatter(gk.geometry.x, gk.geometry.y, s=110, c=C["accent"], marker="s", ec="black",
               lw=1.0, zorder=6)
    for r_km, ls in [(1, "-"), (2, "--"), (5, ":")]:
        circ = gk.geometry.iloc[0].buffer(r_km * 1000).exterior
        ax.plot(*circ.xy, color="black", lw=0.8, ls=ls, zorder=7, label=f"{r_km} km radius")
    ax.set_aspect("equal")
    ax.set_xlim(gk.geometry.x.iloc[0] - 22000, gk.geometry.x.iloc[0] + 22000)
    ax.set_ylim(gk.geometry.y.iloc[0] - 15000, gk.geometry.y.iloc[0] + 15000)
    graticule(ax, M, step=0.2)
    scale_bar(ax, loc="lower left")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("SWOT PIXC and ICESat-2 track geometry at the Kherson pilot site")
    ax.set_xlabel(f"CRS: {M}   ·   SWOT pixels: 2023-04-05 granule, all classes")
    save(fig, "SFig01_swot_icesat_track_geometry", FIG)
    print("  Fig01 + SFig01")


# =========================================================================== #
# Fig 02 — vertical reference chain schematic                                 #
# =========================================================================== #
def fig02():
    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    ax.set_xlim(-0.1, 3.6); ax.set_ylim(0, 10.6); ax.axis("off"); ax.grid(False)

    def box(x, y, txt, fc, w=1.0, h=0.62, fs=7.3, bold=False, ec="#33393d", ls="-"):
        ax.add_patch(mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.035", fc=fc, ec=ec, lw=0.9, ls=ls, zorder=3))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4,
                fontweight="bold" if bold else "normal")

    def arrow(x, y0, y1, lbl=None, color="#33393d"):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color), zorder=2)
        if lbl:
            ax.text(x + 0.075, (y0 + y1) / 2, lbl, fontsize=6.6, va="center", ha="left",
                    style="italic", color="#2c3e50")

    cols = {"gauge": 0.42, "icesat": 1.72, "swot": 3.02}
    for k, x in cols.items():
        ttl = {"gauge": "GAUGE", "icesat": "ICESat-2 ATL13", "swot": "SWOT PIXC"}[k]
        ax.text(x, 10.25, ttl, ha="center", fontsize=9.5, fontweight="bold", color=C[k])

    # --- gauge branch
    x = cols["gauge"]
    box(x, 9.4, "stage (cm above gauge zero)", "#f2f4f5")
    arrow(x, 9.09, 8.51, "+ 12.000 m (accepted zero)")
    box(x, 8.2, "H  (BS-77 normal height)", "#f2f4f5")
    arrow(x, 7.89, 7.31, "+ Δ  EPSG:9902")
    box(x, 7.0, "H  EVRF2019\n(EPSG:9389, zero-tide)", "#dcece4", bold=True)
    ax.text(x, 6.30, "EPSG:9902 is a BS-77 → EVRF2019\nnormal-height offset grid —\nNOT a quasigeoid (acc. 0.068 m)",
            ha="center", va="top", fontsize=6.4, color=C["bad"],
            bbox=dict(boxstyle="round,pad=0.28", fc="#fdf3f2", ec=C["bad"], lw=0.7))

    # --- ICESat branch
    x = cols["icesat"]
    box(x, 9.4, "ATL13  ht_water_surf\nellipsoidal, WGS 84", "#f2f4f5")
    ax.text(x, 9.78, "TIDE-FREE", fontsize=6.6, color=C["bad"],
            va="center", ha="center", fontweight="bold")
    arrow(x, 9.09, 8.51, "tides ALREADY\napplied in ATL03")
    box(x, 8.2, "+ tide_earth_free2mean\n0.06029 − 0.180873 sin²φ", "#fdf1e3")
    arrow(x, 7.89, 7.31, "permanent-tide harmonisation")
    box(x, 7.0, "h  mean/zero-tide crust", "#eef4f8")
    arrow(x, 6.69, 6.11, "− ζ  EGG2015  (zero-tide)")
    box(x, 5.8, "H  EGG2015", "#eef4f8")
    arrow(x, 5.49, 4.91, "+ empirical c(x,y)")
    box(x, 4.6, "EVRF2019-compatible\nempirical WSE", "#dcece4", bold=True)

    # --- SWOT branch
    x = cols["swot"]
    box(x, 9.4, "PIXC  height\nRAW ellipsoidal, WGS 84", "#f2f4f5")
    arrow(x, 9.09, 8.51, "− solid_earth_tide\n− load_tide_fes − pole_tide")
    box(x, 8.2, "h  tide-corrected\n(MEAN-tide crust)", "#eaf1f7")
    ax.text(x, 7.72, "verified −0.0008 m vs RiverSP (n=1023)", fontsize=6.3, color=C["ok"],
            va="center", ha="center", fontweight="bold")
    arrow(x, 7.89, 7.31, "already in the\nEGG2015 convention")
    box(x, 7.0, "h  mean/zero-tide crust", "#eef4f8")
    arrow(x, 6.69, 6.11, "− ζ  EGG2015  (zero-tide)")
    box(x, 5.8, "H  EGG2015", "#eef4f8")

    # common frame
    ax.add_patch(mpatches.FancyBboxPatch((0.85, 3.05), 2.6, 0.62,
                 boxstyle="round,pad=0.04", fc="#e8eef3", ec=C["swot"], lw=1.3, zorder=3))
    ax.text(2.15, 3.36, "COMMON COMPARISON FRAME  —  H  EGG2015", ha="center", va="center",
            fontsize=8.6, fontweight="bold", zorder=4, color=C["swot"])
    for x0 in (cols["icesat"], cols["swot"]):
        ax.annotate("", xy=(2.15, 3.68), xytext=(x0, 5.49 if x0 == cols["swot"] else 4.29),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C["swot"],
                                    connectionstyle="arc3,rad=0.12"), zorder=2)
    ax.annotate("", xy=(1.2, 3.05), xytext=(0.42, 6.69),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C["gauge"], ls="--",
                                connectionstyle="arc3,rad=-0.16"), zorder=2)

    warn = ("EGM2008 (PIXC `geoid`, mean-tide) and EGG2015 (zero-tide quasigeoid) are NOT interchangeable.\n"
            "PIXC `geoid` is reported for reference and is NOT subtracted from PIXC `height` in this chain.\n"
            "EPSG:9902 is never applied to an ellipsoidal or geoid-referenced height.")
    ax.text(1.75, 1.95, warn, ha="center", va="center", fontsize=7.0, color=C["bad"],
            bbox=dict(boxstyle="round,pad=0.42", fc="#fdf3f2", ec=C["bad"], lw=0.9))
    ax.text(1.75, 0.75, "Permanent-tide systems:  ATL13 = tide-free · SWOT (post-correction) = mean-tide · "
            "EGG2015 & EVRF2019 = zero-tide.\nFor crustal ellipsoidal heights zero-tide ≡ mean-tide, "
            "so only the ICESat-2 branch needs harmonisation.",
            ha="center", va="center", fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.34", fc="#f4f6f7", ec="#aeb6ba", lw=0.7))
    ax.set_title("Vertical-reference chains for the three observation systems", pad=8)
    save(fig, "Fig02_vertical_reference_chain", FIG)
    print("  Fig02")


# =========================================================================== #
# Fig 03 — PIXC/RiverSP sign validation                                       #
# =========================================================================== #
def fig03():
    d = _read("Fig03_pixc_riversp_sign_validation.csv")
    per = _read("Fig03_pixc_riversp_sign_validation_pernode.csv")
    main = d[d.variant != "D"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = np.arange(len(main))
    cols = [C["ok"] if v == "A" else C["muted"] for v in main.variant]
    for i, r in main.iterrows():
        vals = per[f"diff_{r.variant}_m"]
        parts = ax.violinplot([vals], positions=[i], widths=0.62, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(cols[i]); b.set_alpha(0.22); b.set_edgecolor("none")
    ax.errorbar(xs, main.median_diff_m,
                yerr=[main.median_diff_m - main.ci95_low_m, main.ci95_high_m - main.median_diff_m],
                fmt="o", ms=7, lw=1.6, capsize=4, color="none", ecolor="#33393d",
                mfc="none", zorder=5)
    ax.scatter(xs, main.median_diff_m, s=62, c=cols, ec="black", lw=0.9, zorder=6)
    zero_line(ax)
    for i, r in main.iterrows():
        ax.annotate(f"{r.median_diff_m:+.4f} m\nNMAD {r.nmad_m:.4f}",
                    (i, r.median_diff_m), xytext=(0, 26 if i == 0 else -34),
                    textcoords="offset points", ha="center", fontsize=7,
                    fontweight="bold" if r.variant == "A" else "normal",
                    color=C["ok"] if r.variant == "A" else "#33393d")
    ax.annotate("ACCEPTED CHAIN", (0, main.median_diff_m[0]), xytext=(0, 58),
                textcoords="offset points", ha="center", fontsize=7.6,
                fontweight="bold", color=C["ok"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"({v})\n{c}" for v, c in zip(main.variant, main.chain)], fontsize=7.4)
    ax.set_ylabel("median  PIXC h − RiverSP h  [m]")
    ax.set_ylim(-0.06, 0.20)
    ax.set_title("PIXC tide-correction sign, validated against RiverSP on the same cycle/pass")
    ax.text(0.985, 0.03, f"n = {int(main.n_nodes[0])} RiverSP nodes  ·  cycle 482 / pass 001 / 2023-04-05\n"
            "≥50 open-water PIXC pixels within 300 m of each node  ·  error bars: 95 % bootstrap CI of the median",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c8cccf", lw=0.6))

    dvar = d[d.variant == "D"].iloc[0]
    ins = ax.inset_axes([0.635, 0.60, 0.34, 0.34])
    ins.bar([0], [dvar.median_diff_m], width=0.5, color=C["bad"], alpha=0.85)
    ins.axhline(0, color="k", lw=0.8)
    ins.set_xticks([0]); ins.set_xticklabels(["(D) height − geoid"], fontsize=6.2)
    ins.set_ylabel("m", fontsize=6.2)
    ins.tick_params(labelsize=6)
    ins.set_title(f"geoid blunder: {dvar.median_diff_m:.2f} m", fontsize=6.6, color=C["bad"], pad=2)
    ins.grid(False)
    save(fig, "Fig03_pixc_riversp_sign_validation", FIG)
    print("  Fig03")


# =========================================================================== #
# Fig 04 — permanent-tide effect on the six correctors                        #
# =========================================================================== #
def fig04():
    d = _read("Fig04_permanent_tide_correctors.csv").sort_values("c_original_m")
    s = _read("Fig04_regional_median_summary.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        ax.plot([r.c_original_m, r.c_harmonised_m], [i, i], color="#95a5a6", lw=1.9, zorder=2)
        ax.annotate(f"Δ {r.delta_c_m:+.4f}", (max(r.c_original_m, r.c_harmonised_m), i),
                    xytext=(9, 0), textcoords="offset points", va="center", fontsize=6.6,
                    color="#5d6d7e")
    ax.scatter(d.c_original_m, y, s=68, c="white", ec=C["icesat"], lw=1.7, zorder=4,
               label="original  (ATL13 tide-free, unharmonised)")
    ax.scatter(d.c_harmonised_m, y, s=68, c=C["icesat"], ec="black", lw=0.8, zorder=5,
               label="harmonised  (+ tide_earth_free2mean)")
    ax.axvline(s.original_m, color=C["icesat"], ls=":", lw=1.1, alpha=0.75,
               label=f"regional median, original = {s.original_m:+.4f} m")
    ax.axvline(s.harmonised_m, color=C["icesat"], ls="-", lw=1.3, alpha=0.9,
               label=f"regional median, harmonised = {s.harmonised_m:+.4f} m")
    ax.set_yticks(y); ax.set_yticklabels([f"{n}\n({i})" for n, i in zip(d.name_en, d.station_id)],
                                         fontsize=7.2)
    ax.set_xlabel("empirical correction  c = H$_{gauge,EVRF2019}$ − H$_{ICESat,EGG2015}$   [m]")
    ax.set_title("Permanent-tide harmonisation shifts every station corrector by ≈ +0.037 m")
    ax.legend(loc="lower right", fontsize=6.8)
    ax.text(0.015, 0.97, f"shift of regional median = {s.shift_m:+.4f} m\n"
            f"station-to-station NMAD: {s.station_to_station_nmad_original_m:.4f} → "
            f"{s.station_to_station_nmad_harmonised_m:.4f} m\n"
            "vertical frame: EGG2015 (zero-tide) · n = 6 stations",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig04_permanent_tide_effect_correctors", FIG)
    print("  Fig04")


# =========================================================================== #
# Fig 05 — Kherson pilot time series                                          #
# =========================================================================== #
def fig05():
    sw = _read("Fig05_kherson_timeseries_swot.csv", parse_dates=["date", "swot_utc"])
    ic = _read("Fig05_kherson_timeseries_icesat.csv", parse_dates=["date", "icesat_utc"])
    ic = ic[ic.qc_flag == "ok"]
    gd = pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET)
    gd = gd[gd.stat_type == "daily"].copy()
    gd["date"] = pd.to_datetime(gd["date"])
    d9902 = float(sw.delta_epsg9902_m.iloc[0])
    gd = gd[(gd.date >= "2023-03-20") & (gd.date <= "2023-06-12")]
    gd["H"] = gd["water_level_m_abs"] + d9902

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    ax = axes[0]
    ax.step(gd.date, gd.H, where="mid", color=C["gauge"], lw=1.0, alpha=0.85,
            label="gauge 80805 — daily value, epoch unresolved")
    ax.scatter(gd.date, gd.H, s=7, c=C["gauge"], alpha=0.55, zorder=3)
    ax.errorbar(sw.swot_utc, sw.H_SWOT_EGG2015_m, yerr=sw.nmad_m, fmt="o", ms=7,
                color=C["swot"], ecolor=C["swot"], capsize=3, lw=1.2, mec="black", mew=0.6,
                zorder=5, label="SWOT PIXC (instantaneous; error bar = within-scene NMAD)")
    ax.scatter(ic.icesat_utc, ic.H_meantide_m, s=62, marker="^", c=C["icesat"],
               ec="black", lw=0.6, zorder=5,
               label="ICESat-2 ATL13 (instantaneous, permanent-tide harmonised)")
    for _, r in sw.iterrows():
        ax.annotate(r.swot_utc.strftime("%H:%M"), (r.swot_utc, r.H_SWOT_EGG2015_m),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=5.8, color=C["swot"])
    for _, r in ic.iterrows():
        ax.annotate(r.icesat_utc.strftime("%H:%M"), (r.icesat_utc, r.H_meantide_m),
                    xytext=(0, -14), textcoords="offset points", ha="center",
                    fontsize=5.8, color=C["icesat"])
    ax.set_ylabel("water-surface elevation\nin common frame  [m]")
    ax.legend(loc="upper left", fontsize=6.7)
    ax.set_title("Kherson pilot — gauge, SWOT and ICESat-2 in the EGG2015 frame (6 SWOT dates)")
    panel_label(ax, "a")
    ax.text(0.995, 0.03, "gauge: H = stage$_{BS77}$ + Δ EPSG:9902  (EVRF2019)\n"
            "satellites: H = h − ζ$_{EGG2015}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c8cccf", lw=0.6))

    ax = axes[1]
    zero_line(ax)
    ax.scatter(sw.swot_utc, -sw.c_SWOT_m, s=58, c=C["swot"], ec="black", lw=0.6,
               zorder=5, label=f"SWOT − gauge  (median {np.median(-sw.c_SWOT_m):+.3f} m, n={len(sw)})")
    ax.scatter(ic.icesat_utc, -ic.c_ICESat_m, s=58, marker="^", c=C["icesat"], ec="black",
               lw=0.6, zorder=5,
               label=f"ICESat-2 − gauge  (median {np.median(-ic.c_ICESat_m):+.3f} m, n={len(ic)})")
    for _, r in sw.iterrows():
        ax.plot([r.swot_utc, r.swot_utc], [0, -r.c_SWOT_m], color=C["swot"], lw=0.8, alpha=0.45, zorder=2)
    for _, r in ic.iterrows():
        ax.plot([r.icesat_utc, r.icesat_utc], [0, -r.c_ICESat_m], color=C["icesat"], lw=0.8,
                alpha=0.45, zorder=2)
    ax.set_ylabel("satellite − gauge  [m]")
    ax.set_xlabel("2023 (UTC)")
    ax.legend(loc="upper left", fontsize=6.7)
    panel_label(ax, "b")
    ax.text(0.995, 0.05, "residuals are satellite minus gauge; gauge epoch unresolved (daily)\n"
            "Kherson is wind-setup dominated (±0.3–0.5 m)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color="#5d6d7e")
    fig.autofmt_xdate()
    save(fig, "Fig05_kherson_pilot_timeseries", FIG)
    print("  Fig05")


# =========================================================================== #
# Fig 06 — same-day three-way, 2023-04-05 (rebuilt: true geometry)            #
# =========================================================================== #
def fig06():
    three = _read("Fig06_three_way_20230405.csv")
    px = pd.read_csv(FD / "Fig11_pixc_pixels_20230405_accepted.csv.gz")
    co = _read("Fig16_colocated_pairs_persegment.csv")
    dt_h = float(three.dt_swot_minus_icesat_h.iloc[0])
    g = _gauges_gdf(); gk = g[g.id == 80805].iloc[0]

    dist_med = float(co.dist_to_gauge_km.median())
    fig = plt.figure(figsize=(10.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    pg = gpd.GeoDataFrame(px, geometry=gpd.points_from_xy(px.lon, px.lat),
                          crs=CFG.CRS_GEOG).to_crs(M)
    vlo, vhi = np.nanpercentile(px.H_EGG2015_m, [5, 95])
    sc = ax.scatter(pg.geometry.x, pg.geometry.y, s=0.5, c=px.H_EGG2015_m, cmap="viridis",
                    vmin=vlo, vmax=vhi, lw=0, rasterized=True, zorder=3)
    cg = gpd.GeoDataFrame(co, geometry=gpd.points_from_xy(co.lon, co.lat),
                          crs=CFG.CRS_GEOG).to_crs(M)
    ax.scatter(cg.geometry.x, cg.geometry.y, s=9, marker="^", c=C["icesat"], ec="none",
               alpha=0.85, zorder=6, label=f"ICESat-2 ATL13 segments co-located with SWOT (n={len(co)})")
    ax.scatter([gk.geometry.x], [gk.geometry.y], s=115, marker="s", c=C["accent"],
               ec="black", lw=1.0, zorder=8, label="gauge 80805 (Kherson)")
    ax.plot(*gk.geometry.buffer(1000).exterior.xy, color="black", lw=1.1, zorder=8,
            label="1 km SWOT aggregation radius")
    # link showing the separation between the gauge and the co-located pairs
    cx, cy = cg.geometry.x.median(), cg.geometry.y.median()
    ax.annotate("", xy=(cx, cy), xytext=(gk.geometry.x, gk.geometry.y),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.4, color=C["bad"], ls="--"), zorder=9)
    ax.text((cx + gk.geometry.x) / 2, (cy + gk.geometry.y) / 2,
            f"  {dist_med:.0f} km", color=C["bad"], fontsize=8.5,
            fontweight="bold", rotation=38, ha="center", va="bottom", zorder=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.88, pad=0.012, aspect=26)
    cb.set_label("H$_{SWOT,EGG2015}$  [m]", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_aspect("equal")
    xs = list(cg.geometry.x) + [gk.geometry.x]
    ys = list(cg.geometry.y) + [gk.geometry.y]
    padx = 0.10 * (max(xs) - min(xs)); pady = 0.16 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - padx, max(xs) + padx)
    ax.set_ylim(min(ys) - pady, max(ys) + pady)
    graticule(ax, M, step=0.2)
    scale_bar(ax, loc="lower right")
    ax.legend(loc="lower left", fontsize=6.0)
    ax.set_title(f"Spatial — the two sensors meet {dist_med:.0f} km from the gauge")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 24); ax.set_ylim(0.35, 3.15)
    ax.set_yticks([0.8, 1.7, 2.6])
    ax.set_yticklabels(["gauge 80805", "ICESat-2\nATL13", "SWOT\nPIXC"], fontsize=7.5)
    ax.add_patch(mpatches.Rectangle((0, 0.55), 24, 0.5, fc=C["gauge"], alpha=0.16,
                                    ec=C["gauge"], lw=0.8, ls="--"))
    ax.text(12, 0.8, "daily value — epoch unresolved", ha="center", va="center",
            fontsize=7, color=C["gauge"], style="italic")
    t_ic = pd.Timestamp(three.utc.iloc[1]); t_sw = pd.Timestamp(three.utc.iloc[2])
    h_ic = t_ic.hour + t_ic.minute / 60; h_sw = t_sw.hour + t_sw.minute / 60
    ax.scatter([h_ic], [1.7], s=130, marker="^", c=C["icesat"], ec="black", lw=0.8, zorder=5)
    ax.scatter([h_sw], [2.6], s=130, marker="o", c=C["swot"], ec="black", lw=0.8, zorder=5)
    ax.annotate(t_ic.strftime("%H:%M:%S UTC"), (h_ic, 1.7), xytext=(0, -17),
                textcoords="offset points", ha="center", fontsize=7, color=C["icesat"])
    ax.annotate(t_sw.strftime("%H:%M:%S UTC"), (h_sw, 2.6), xytext=(0, 13),
                textcoords="offset points", ha="center", fontsize=7, color=C["swot"])
    ax.annotate("", xy=(h_sw, 2.16), xytext=(h_ic, 2.16),
                arrowprops=dict(arrowstyle="<|-|>", lw=1.3, color=C["bad"]))
    ax.text((h_ic + h_sw) / 2, 2.26, f"Δt = {dt_h:.2f} h", ha="center", fontsize=8.4,
            fontweight="bold", color=C["bad"])
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of 2023-04-05 (UTC)")
    ax.set_title("Temporal — SAME-DAY, NOT SIMULTANEOUS", color=C["bad"])
    panel_label(ax, "b")
    ax.text(0.5, 0.10, f"gauge {three.value_m.iloc[0]:.3f} m (EVRF2019, AT the gauge)\n"
            f"co-located pair {dist_med:.0f} km downstream:  "
            f"ICESat-2 {co.H_ICESat_EGG2015_m.median():.3f} m  vs  "
            f"SWOT {co.H_SWOT_EGG2015_m.median():.3f} m\n"
            f"difference {co.diff_swot_minus_icesat_m.median():+.3f} m",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f6f7", ec="#aeb6ba", lw=0.7))
    fig.suptitle("Three-way comparison, 2023-04-05 — gauge · SWOT PIXC · ICESat-2", y=1.01)
    save(fig, "Fig06_three_way_20230405", FIG)
    print("  Fig06")


# =========================================================================== #
# Fig 16 — spatially co-located cross-sensor comparison                       #
# =========================================================================== #
def fig16():
    seg = _read("Fig16_colocated_pairs_persegment.csv")
    beam = _read("Fig16_colocated_pairs_perbeam.csv")
    summ = _read("Fig16_colocated_summary.csv").iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7),
                             gridspec_kw={"width_ratios": [1.15, 1, 0.95], "wspace": 0.30})

    ax = axes[0]
    lim = (0.25, 0.85)
    ax.plot(lim, lim, color="k", lw=0.9, ls="--", zorder=2, label="1:1")
    for b, grp in seg.groupby("beam"):
        ax.scatter(grp.H_ICESat_EGG2015_m, grp.H_SWOT_EGG2015_m, s=7, alpha=0.55,
                   lw=0, label=f"{b} (n={len(grp)})")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("H$_{ICESat-2,EGG2015}$  [m]")
    ax.set_ylabel("H$_{SWOT,EGG2015}$  [m]")
    ax.set_title("Co-located pairs, 1:1")
    ax.legend(fontsize=5.7, loc="upper left", ncol=2, markerscale=1.8)
    panel_label(ax, "a")

    ax = axes[1]
    core = seg.diff_swot_minus_icesat_m.clip(-0.45, 0.10)
    n_out = int(((seg.diff_swot_minus_icesat_m < -0.45) |
                 (seg.diff_swot_minus_icesat_m > 0.10)).sum())
    ax.hist(core, bins=45, color=C["swot"], alpha=0.7, ec="white", lw=0.3)
    ax.set_xlim(-0.45, 0.10)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.axvline(summ.median_diff_m, color=C["bad"], lw=1.6)
    ax.annotate(f"median\n{summ.median_diff_m:+.4f} m", (summ.median_diff_m, ax.get_ylim()[1] * 0.86),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=7.4,
                fontweight="bold", color=C["bad"])
    ax.set_xlabel("H$_{SWOT}$ − H$_{ICESat-2}$  [m]")
    ax.set_ylabel(f"ATL13 segments (n={int(summ.n_segments)})")
    ax.set_title("Difference distribution")
    if n_out:
        ax.text(0.03, 0.97, f"{n_out} segment(s) outside\nthe plotted range",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.2, color="#5d6d7e")
    panel_label(ax, "b")

    ax = axes[2]
    y = np.arange(len(beam))[::-1]
    ax.scatter(beam.diff_m, y, s=70, c=C["swot"], ec="black", lw=0.8, zorder=5)
    ax.errorbar(beam.diff_m, y, xerr=beam.within_beam_nmad_m, fmt="none",
                ecolor=C["swot"], capsize=3, lw=1.1, zorder=4)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.axvspan(summ.ci95_low_m, summ.ci95_high_m, color=C["accent"], alpha=0.20, zorder=1)
    ax.axvline(summ.median_diff_m, color=C["bad"], lw=1.5, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(beam.beam, fontsize=7.5)
    for i, r in beam.iterrows():
        ax.annotate(f"n={int(r.n_segments)}", (r.diff_m, y[i]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=6.2)
    ax.set_xlabel("H$_{SWOT}$ − H$_{ICESat-2}$  [m]")
    ax.set_title("By beam (independent unit)")
    ax.set_xlim(-0.30, 0.04)
    panel_label(ax, "c")
    ax.text(0.03, 0.30, f"median {summ.median_diff_m:+.4f} m\n"
            f"95 % CI [{summ.ci95_low_m:+.3f}, {summ.ci95_high_m:+.3f}]\n"
            f"NMAD {summ.nmad_m:.4f} · n = {int(summ.n_independent_beams)} beams\n"
            "CI excludes zero",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c8cccf", lw=0.6))

    fig.suptitle("SWOT PIXC vs ICESat-2 ATL13, spatially co-located — no gauge involved "
                 f"(2023-04-05, match radius {summ.match_radius_km:g} km, Δt {summ.median_dt_hours:.2f} h)",
                 y=1.03, fontsize=9.5)
    save(fig, "Fig16_colocated_cross_sensor", FIG)
    print("  Fig16")


# =========================================================================== #
# Fig 07 — sensor residual forest                                             #
# =========================================================================== #
def fig07():
    d = _read("Fig07_sensor_residuals.csv")
    d = d[d.statistic.str.startswith("median (satellite")].reset_index(drop=True)
    d["med"] = -d.median_c_m          # display as satellite − gauge
    d["lo"] = -d.ci95_high_m
    d["hi"] = -d.ci95_low_m
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    y = np.arange(len(d))[::-1]
    cols = [C["swot"], C["icesat"]]
    for i, r in d.iterrows():
        ax.plot([r.lo, r.hi], [y[i], y[i]], lw=2.4, color=cols[i], solid_capstyle="round", zorder=3)
        ax.plot([r.lo, r.lo], [y[i] - .1, y[i] + .1], lw=1.4, color=cols[i], zorder=3)
        ax.plot([r.hi, r.hi], [y[i] - .1, y[i] + .1], lw=1.4, color=cols[i], zorder=3)
        ax.scatter([r.med], [y[i]], s=95, c=cols[i], ec="black", lw=0.9, zorder=5)
        ax.annotate(f"{r.med:+.3f} m   [{r.lo:+.3f}, {r.hi:+.3f}]\nNMAD {r.nmad_m:.3f} · n = {r['n']} {r.independent_unit}",
                    (r.hi, y[i]), xytext=(12, 0), textcoords="offset points", va="center", fontsize=6.9)
    lo_ov, hi_ov = max(d.lo), min(d.hi)
    if lo_ov <= hi_ov:
        ax.axvspan(lo_ov, hi_ov, color=C["accent"], alpha=0.16, zorder=1)
        ax.text((lo_ov + hi_ov) / 2, 1.42, f"95 % CIs OVERLAP\n[{lo_ov:+.3f}, {hi_ov:+.3f}] m",
                ha="center", fontsize=7.2, color="#a06000", fontweight="bold")
    ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels(d.sensor, fontsize=8)
    ax.set_ylim(-0.75, 1.95)
    ax.set_xlim(-0.12, 0.52)
    ax.set_xlabel("satellite WSE − gauge WSE   [m]      (common frame: EGG2015, permanent-tide harmonised)")
    ax.set_title("Kherson pilot — residual against the gauge, by sensor")
    ax.text(0.995, 0.04, "bars: 95 % percentile bootstrap CI of the median (seed 42)\n"
            "difference is NOT statistically significant at this sample size\n"
            "CAUTION: SWOT is aggregated within 1 km of the gauge; ICESat-2 beam-passes are a\n"
            "median 29.8 km away (19–60 km) — an AOI-scale quantity, NOT gauge-collocated.\n"
            "For a like-for-like cross-sensor number see Fig. 16.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig07_sensor_residual_forest", FIG)
    print("  Fig07")


# =========================================================================== #
# Fig 08 — harmonisation sensitivity                                          #
# =========================================================================== #
def fig08():
    d = _read("Fig08_harmonisation_variants.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    x = np.arange(len(d))
    cols = [C["bad"], C["accent"], C["ok"]]
    ax.bar(x, d.delta_sensor_m, width=0.5, color=cols, alpha=0.88, ec="black", lw=0.7)
    zero_line(ax)
    for i, r in d.iterrows():
        ax.annotate(f"{r.delta_sensor_m:+.4f} m", (i, r.delta_sensor_m),
                    xytext=(0, -15 if r.delta_sensor_m < 0 else 6), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold", color="white" if i == 2 else "black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.variant}\n{r.description}" for r in d.itertuples()], fontsize=7)
    ax.set_ylabel("Δ$_{sensor}$ = median R$_{SWOT}$ − median R$_{ICESat}$   [m]")
    ax.set_title("Effect of vertical-reference harmonisation on the cross-sensor difference")
    notes = {"V0": "physically inconsistent", "V1": "incomplete permanent-tide harmonisation",
             "V2": "PREFERRED BRANCH"}
    for i, r in d.iterrows():
        ax.annotate(notes[r.variant], (i, 0.012), ha="center", fontsize=6.7,
                    fontweight="bold" if r.variant == "V2" else "normal", color=cols[i])
    ax.set_ylim(-0.34, 0.06)
    ax.text(0.015, 0.05, "R = H$_{sat,EGG2015}$ − gauge stage; the gauge zero (−5.00 m) and\n"
            "Δ EPSG:9902 are common constants and cancel in Δ$_{sensor}$.\n"
            f"n = {int(d.n_SWOT[0])} SWOT overpasses, {int(d.n_ICESat[0])} ICESat date×RGT   ·   "
            f"95 % CIs overlap in every variant.\n"
            "This diagnostic is gauge-mediated and spatially mismatched (ICESat ~30 km from the\n"
            "gauge): read it for the RELATIVE effect of each harmonisation step, not as Δ$_{sensor}$.",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig08_vertical_harmonisation_sensitivity", FIG)
    print("  Fig08")


# =========================================================================== #
# Fig 09 — water-mask sensitivity                                             #
# =========================================================================== #
def fig09():
    d = _read("Fig09_water_mask_sensitivity.csv")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})
    x = np.arange(len(d))
    cols = [C["ok"] if a else C["muted"] for a in d.accepted]
    ax = axes[0]
    ax.errorbar(x, d.median_residual_m, yerr=d.nmad_residual_m, fmt="none",
                ecolor="#33393d", capsize=4, lw=1.2, zorder=3)
    ax.scatter(x, d.median_residual_m, s=85, c=cols, ec="black", lw=0.9, zorder=5)
    zero_line(ax)
    for i, r in d.iterrows():
        ax.annotate(f"{r.median_residual_m:+.3f}", (i, r.median_residual_m),
                    xytext=(0, 13), textcoords="offset points", ha="center", fontsize=7,
                    fontweight="bold" if r.accepted else "normal")
    ax.annotate("ACCEPTED", (0, d.median_residual_m[0]), xytext=(0, -22),
                textcoords="offset points", ha="center", fontsize=7.2,
                fontweight="bold", color=C["ok"])
    ax.set_ylabel("SWOT − gauge  [m]")
    ax.set_title("SWOT water-mask sensitivity at Kherson (radius 1 km, 6 dates)")
    panel_label(ax, "a")
    ax.text(0.99, 0.95, "error bars: NMAD across the 6 dates", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5)

    ax = axes[1]
    ax.bar(x, d.median_n_px, width=0.55, color=cols, alpha=0.85, ec="black", lw=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("accepted pixels\nper scene (median, log)")
    for i, r in d.iterrows():
        ax.annotate(f"{int(r.median_n_px):,}", (i, r.median_n_px), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"class {r.strategy}\n{r.description}" for r in d.itertuples()], fontsize=6.6)
    panel_label(ax, "b")
    ax.text(0.99, 0.93, "PIXC classification (PDD D-56411): 3 water_near_land, 4 open_water,\n"
            "5 dark_water, 6 low_coh_water_near_land, 7 open_low_coh_water",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.1)
    save(fig, "Fig09_water_mask_sensitivity", FIG)
    print("  Fig09")


# =========================================================================== #
# Fig 10 — radius sensitivity                                                 #
# =========================================================================== #
def fig10():
    d = _read("Fig10_radius_sensitivity.csv")
    raw = _read("Fig14_distance_residuals.csv")
    sw = raw[raw.sensor == "SWOT"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for r_km, grp in sw.groupby("radius_km"):
        ax.scatter([r_km] * len(grp), grp.residual_vs_gauge_m, s=16, c=C["swot"],
                   alpha=0.35, lw=0, zorder=3)
    ax.errorbar(d.radius_km, d.median_residual_m, yerr=d.nmad_residual_m, fmt="o-",
                ms=7, color=C["swot"], ecolor=C["swot"], capsize=4, lw=1.4, mec="black",
                mew=0.7, zorder=5, label="median across 6 dates ± NMAD")
    zero_line(ax)
    ax.axvline(1.0, color=C["ok"], ls="--", lw=1.2, alpha=0.85)
    ax.annotate("accepted radius\n1 km", (1.0, ax.get_ylim()[1]), xytext=(6, -14),
                textcoords="offset points", fontsize=7, color=C["ok"], fontweight="bold",
                va="top")
    ax.set_xscale("log")
    ax.set_xticks(d.radius_km); ax.set_xticklabels([f"{v:g}" for v in d.radius_km])
    ax.set_xlabel("aggregation radius from the Kherson gauge  [km]")
    ax.set_ylabel("SWOT − gauge  [m]")
    ax.set_title("Radius sensitivity — widening the window samples the channel gradient")
    ax2 = ax.twinx()
    ax2.plot(d.radius_km, d.median_n_px, color=C["muted"], ls=":", lw=1.3, marker="s", ms=4)
    ax2.set_yscale("log"); ax2.set_ylabel("median accepted pixels (log)", color=C["muted"], fontsize=7.5)
    ax2.tick_params(axis="y", labelcolor=C["muted"], labelsize=6.8); ax2.grid(False)
    ax.legend(loc="lower left", fontsize=7)
    drift = d.median_residual_m.iloc[-1] - d.median_residual_m.iloc[0]
    ax.text(0.985, 0.05, f"median drifts {drift:+.3f} m between 0.5 and 10 km\n"
            "light points: individual overpasses",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig10_radius_sensitivity_kherson", FIG)
    print("  Fig10")


# =========================================================================== #
# Fig 11 — PIXC pixel maps                                                    #
# =========================================================================== #
def fig11():
    px = pd.read_csv(FD / "Fig11_pixc_pixels_20230405_accepted.csv.gz")
    g = _gauges_gdf(); gk = g[g.id == 80805].iloc[0]
    pg = gpd.GeoDataFrame(px, geometry=gpd.points_from_xy(px.lon, px.lat),
                          crs=CFG.CRS_GEOG).to_crs(M)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sc = ax.scatter(pg.geometry.x, pg.geometry.y, s=0.8, c=px.H_EGG2015_m, cmap="viridis",
                    vmin=0.0, vmax=1.0, lw=0, rasterized=True)
    ax.scatter([gk.geometry.x], [gk.geometry.y], s=140, marker="s", c=C["accent"],
               ec="black", lw=1.0, zorder=7, label="gauge 80805")
    for r_km, ls in [(1, "-"), (2, "--"), (5, ":")]:
        ax.plot(*gk.geometry.buffer(r_km * 1000).exterior.xy, color="black", lw=0.9, ls=ls,
                zorder=7, label=f"{r_km} km")
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("H$_{SWOT,EGG2015}$  [m]", fontsize=7.5)
    ax.set_aspect("equal")
    graticule(ax, M, step=0.1)
    scale_bar(ax, loc="lower left"); north_arrow(ax)
    ax.legend(loc="upper right", fontsize=6.8)
    ax.set_title("SWOT PIXC accepted water pixels, 2023-04-05 (cycle 482 / pass 001 / tile 237R)")
    ax.set_xlabel(f"CRS: {M}   ·   classification = 4 (open_water), n = {len(px):,} pixels")
    save(fig, "Fig11_pixc_wse_map_20230405", FIG)

    ac = pd.read_csv(FD / "SFig11_pixc_pixels_20230405_allclasses.csv.gz")
    ag = gpd.GeoDataFrame(ac, geometry=gpd.points_from_xy(ac.lon, ac.lat),
                          crs=CFG.CRS_GEOG).to_crs(M)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    palette = {1: "#d9d2c5", 2: "#c2b280", 3: "#7fb3d5", 4: C["swot"],
               5: "#5d3a9b", 6: "#e8a33d", 7: "#48c9b0"}
    for cls, col in palette.items():
        m = ac.classification == cls
        if m.sum() == 0:
            continue
        ax.scatter(ag.geometry.x[m], ag.geometry.y[m], s=0.7, c=col, lw=0, alpha=0.75,
                   rasterized=True, label=f"{cls} {CFG.PIXC_CLASS[cls]} (n={int(m.sum()):,})")
    ax.scatter([gk.geometry.x], [gk.geometry.y], s=140, marker="s", c=C["accent"],
               ec="black", lw=1.0, zorder=7)
    ax.plot(*gk.geometry.buffer(1000).exterior.xy, color="black", lw=1.0, zorder=7)
    ax.set_aspect("equal")
    graticule(ax, M, step=0.1); scale_bar(ax, loc="lower left")
    ax.legend(loc="upper right", fontsize=6.2, markerscale=6)
    ax.set_title("PIXC classification, 2023-04-05 — why the water mask matters")
    ax.set_xlabel(f"CRS: {M}   ·   random subsample of {len(ac):,} pixels for rendering")
    save(fig, "SFig11_pixc_water_class_map_20230405", FIG)
    print("  Fig11 + SFig11")


# =========================================================================== #
# Fig 12 — pre-breach coverage timeline                                       #
# =========================================================================== #
def fig12():
    d = _read("Fig12_coverage_timeline.csv", parse_dates=["date", "swot_utc"])
    ic = _read("Fig12_coverage_timeline_icesat.csv", parse_dates=["date", "icesat_utc"])
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    dn = d[~d.downloaded]; dy = d[d.downloaded]
    ax.scatter(dn.date, [2] * len(dn), s=34, marker="o", c="white", ec=C["swot"], lw=1.0,
               zorder=4, label=f"SWOT PIXC available, not downloaded (n={len(dn)})")
    ax.scatter(dy.date, [2] * len(dy), s=68, marker="o", c=C["swot"], ec="black", lw=0.8,
               zorder=5, label=f"SWOT PIXC downloaded — pilot (n={len(dy)})")
    ax.scatter(ic.date, [1.4] * len(ic), s=54, marker="^", c=C["icesat"], ec="black",
               lw=0.6, zorder=5, label=f"ICESat-2 ATL13 at Kherson (n={len(ic)} date×RGT)")
    gd = pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET)
    gd = gd[gd.stat_type == "daily"].copy(); gd["date"] = pd.to_datetime(gd["date"])
    gd = gd[(gd.date >= d.date.min()) & (gd.date <= pd.Timestamp("2023-06-12"))]
    ax.scatter(gd.date, [0.8] * len(gd), s=8, marker="|", c=C["gauge"], lw=1.0, zorder=4,
               label=f"Kherson gauge 80805, daily (n={len(gd)})")
    same = d[d.icesat_same_day]
    for _, r in same.iterrows():
        ax.plot([r.date, r.date], [1.4, 2], color=C["accent"], lw=1.6, zorder=3)
    if len(same):
        ax.scatter(same.date, [1.7] * len(same), s=90, marker="*", c=C["accent"],
                   ec="black", lw=0.6, zorder=6,
                   label=f"same-day SWOT + ICESat-2 (n={len(same)})")
    br = pd.Timestamp(CFG.BREACH_DATE)
    ax.axvline(br, color=C["bad"], lw=1.6, zorder=2)
    ax.annotate("dam breach\n2023-06-06", (br, 2.42), xytext=(6, 0), textcoords="offset points",
                fontsize=7, color=C["bad"], fontweight="bold", va="top")
    ax.set_yticks([0.8, 1.4, 2]); ax.set_yticklabels(["gauge", "ICESat-2", "SWOT"], fontsize=8)
    ax.set_ylim(0.45, 2.65)
    ax.set_xlim(d.date.min() - pd.Timedelta(days=3), pd.Timestamp("2023-06-12"))
    ax.set_xlabel("2023")
    ax.set_title("Pre-breach observation inventory over the Kherson pilot site (cal/val 1-day repeat)")
    ax.legend(loc="upper left", fontsize=6.4, ncol=2)
    fig.autofmt_xdate()
    save(fig, "Fig12_swot_prebreach_coverage", FIG)
    print("  Fig12")


# =========================================================================== #
# Fig 13 — residual vs temporal separation                                    #
# =========================================================================== #
def fig13():
    d = _read("Fig13_time_separation.csv", parse_dates=["swot_utc", "icesat_utc"])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    sd = d[d.same_day]; od = d[~d.same_day]
    ax.scatter(od.abs_dt_hours, od.diff_swot_minus_icesat_m, s=34, c=C["muted"], alpha=0.6,
               ec="none", label=f"different-day pairs (n={len(od)})")
    ax.scatter(sd.abs_dt_hours, sd.diff_swot_minus_icesat_m, s=95, c=C["accent"],
               ec="black", lw=0.9, zorder=5, label=f"SAME-DAY pairs (n={len(sd)})")
    zero_line(ax)
    for _, r in sd.iterrows():
        ax.annotate(f"{r.swot_utc:%Y-%m-%d}\nΔt={r.abs_dt_hours:.1f} h",
                    (r.abs_dt_hours, r.diff_swot_minus_icesat_m), xytext=(8, 8),
                    textcoords="offset points", fontsize=6.4)
    ax.set_xlabel("|temporal separation|  SWOT − ICESat-2  [hours]")
    ax.set_ylabel("H$_{SWOT}$ − H$_{ICESat}$  [m]   (EGG2015, harmonised)")
    ax.set_title("Cross-sensor difference against temporal separation")
    ax.legend(loc="upper right", fontsize=7)
    ax.text(0.015, 0.04, f"n = {len(d)} pairs from {d.swot_utc.dt.date.nunique()} SWOT × "
            f"{d.icesat_utc.dt.date.nunique()} ICESat dates.\n"
            "No regression fitted — n is too small and pairs are not independent.\n"
            "Kherson is wind-setup dominated (±0.3–0.5 m), so temporal separation is\n"
            "expected to dominate the scatter.",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig13_residual_vs_time_separation", FIG)
    print("  Fig13")


# =========================================================================== #
# Fig 14 — residual vs distance                                               #
# =========================================================================== #
def fig14():
    d = _read("Fig14_distance_residuals.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    sw = d[d.sensor == "SWOT"]; ic = d[d.sensor == "ICESat-2"]
    ax.scatter(sw.median_dist_km, sw.residual_vs_gauge_m, s=34, c=C["swot"], alpha=0.65,
               ec="none", label=f"SWOT PIXC (n={len(sw)} scene×radius)")
    ax.scatter(ic.median_dist_km, ic.residual_vs_gauge_m, s=70, marker="^", c=C["icesat"],
               ec="black", lw=0.6, zorder=5, label=f"ICESat-2 ATL13 (n={len(ic)} date×RGT)")
    zero_line(ax)
    med = sw.groupby("radius_km").agg(x=("median_dist_km", "median"),
                                      y=("residual_vs_gauge_m", "median")).reset_index()
    ax.plot(med.x, med.y, color=C["swot"], lw=1.4, ls="--", zorder=4,
            label="SWOT median by aggregation radius")
    ax.set_xlabel("median distance of accepted observations from the gauge  [km]")
    ax.set_ylabel("satellite − gauge  [m]")
    ax.set_title("Residual against spatial separation — channel-gradient contamination")
    ax.legend(loc="upper left", fontsize=6.9)
    ax.text(0.985, 0.04, "straight-line distance in EPSG:32636; along-channel distance\n"
            "would be preferable but requires a SWORD centreline for the estuary.\n"
            "Gauge epoch unresolved (daily value).",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig14_residual_vs_distance", FIG)
    print("  Fig14")


# =========================================================================== #
# Fig 15 — reservoir corrector control points                                 #
# =========================================================================== #
def fig15():
    d = _read("Fig15_corrector_points.csv")
    s = _read("Fig04_regional_median_summary.csv").iloc[0]
    res = gpd.read_file(CFG.RESERVOIR_GEOJSON).to_crs(M)
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat),
                         crs=CFG.CRS_GEOG).to_crs(M)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    res.plot(ax=ax, fc=C["reservoir"], ec="#5d92a8", lw=0.6, alpha=0.8, zorder=1)
    sc = ax.scatter(g.geometry.x, g.geometry.y, s=210, c=d.c_harmonised_m, cmap="RdYlBu_r",
                    vmin=-0.20, vmax=-0.08, ec="black", lw=1.1, zorder=6)
    dam = gpd.GeoSeries([gpd.points_from_xy([CFG.KAKHOVKA_DAM[0]], [CFG.KAKHOVKA_DAM[1]])[0]],
                        crs=CFG.CRS_GEOG).to_crs(M)
    ax.scatter(dam.x, dam.y, s=175, marker="*", c=C["dam"], ec="white", lw=0.7, zorder=7,
               label="Kakhovka dam")
    for _, r in g.iterrows():
        ax.annotate(f"{r['name_en']}\nc = {r.c_harmonised_m:+.3f} m\nNMAD {r.empirical_nmad_m:.3f}"
                    f" · n={int(r.n_matchups)}",
                    (r.geometry.x, r.geometry.y), xytext=(9, 9), textcoords="offset points",
                    fontsize=6.4, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#aeb6ba",
                              lw=0.55, alpha=0.9))
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("harmonised c  [m]", fontsize=7.5)
    ax.set_aspect("equal")
    graticule(ax, M, step=0.5); scale_bar(ax, loc="lower left"); north_arrow(ax)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Empirical EVRF2019 correctors at the six reservoir control gauges "
                 "(permanent-tide harmonised)")
    ax.set_xlabel(f"CRS: {M}   ·   c = H$_{{gauge,EVRF2019}}$ − H$_{{ICESat,EGG2015}}$, PRE_BREACH")
    ax.text(0.015, 0.035, f"regional median c = {s.harmonised_m:+.4f} m "
            f"(was {s.original_m:+.4f} m before harmonisation)\n"
            f"station-to-station NMAD = {s.station_to_station_nmad_harmonised_m:.4f} m · n = 6 controls\n"
            "NO interpolated surface: 6 controls do not support one (see LOSO-CV)",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "Fig15_reservoir_corrector_control_points", FIG)
    print("  Fig15")


ALL = {f"Fig{n:02d}": fn for n, fn in [
    (1, fig01), (2, fig02), (3, fig03), (4, fig04), (5, fig05), (6, fig06), (7, fig07),
    (8, fig08), (9, fig09), (10, fig10), (11, fig11), (12, fig12), (13, fig13),
    (14, fig14), (15, fig15), (16, fig16)]}

# =========================================================================== #
# Fig 17 — independent SWOT validation of the gauge-derived correction        #
# =========================================================================== #
def fig17():
    d = _read("Fig17_colocated_perdate.csv", parse_dates=["date", "swot_utc", "icesat_utc"])
    summ = _read("Fig17_independent_validation_summary.csv")
    c = float(summ.c_gauge_harmonised_m.iloc[0])
    d = d.sort_values("date").reset_index(drop=True)
    ok = d.qc_flag == "ok"

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.05, 1.05, 0.95], "wspace": 0.30})
    x = np.arange(len(d))
    lab = [f"{r.date:%Y-%m-%d}\nΔt={r.dt_hours:+.1f} h" for r in d.itertuples()]

    # (a) before correction
    ax = axes[0]
    zero_line(ax)
    ax.axhline(c, color=C["icesat"], ls="-", lw=1.4,
               label=f"gauge-derived c = {c:+.4f} m")
    for i, r in d.iterrows():
        mk = "o" if r.qc_flag == "ok" else "X"
        ax.errorbar(i, r.d_date_m, yerr=r.within_date_nmad_m, fmt=mk, ms=10,
                    color=C["swot"] if r.qc_flag == "ok" else C["muted"],
                    ecolor="#33393d", capsize=4, lw=1.2, mec="black", mew=0.8, zorder=5)
        ax.annotate(f"{r.d_date_m:+.3f}", (i, r.d_date_m), xytext=(0, 15),
                    textcoords="offset points", ha="center", fontsize=7.2, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.8)
    ax.set_ylabel("d = H$_{SWOT}$ − H$_{ICESat-2}$  [m]")
    ax.set_title("(a) before correction")
    ax.set_ylim(-0.30, 0.42)
    ax.legend(loc="upper left", fontsize=6.6)
    panel_label(ax, "a")

    # (b) after correction
    ax = axes[1]
    zero_line(ax)
    for i, r in d.iterrows():
        mk = "o" if r.qc_flag == "ok" else "X"
        ax.errorbar(i, r.e_date_m, yerr=r.within_date_nmad_m, fmt=mk, ms=10,
                    color=C["accent"] if r.qc_flag == "ok" else C["muted"],
                    ecolor="#33393d", capsize=4, lw=1.2, mec="black", mew=0.8, zorder=5)
        ax.annotate(f"{r.e_date_m:+.3f}", (i, r.e_date_m), xytext=(0, 15),
                    textcoords="offset points", ha="center", fontsize=7.2, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.8)
    ax.set_ylabel("e = H$_{SWOT}$ − (H$_{ICESat-2}$ + c)  [m]")
    ax.set_title("(b) after the gauge-derived correction")
    ax.set_ylim(-0.30, 0.42)
    panel_label(ax, "b")
    b = summ.iloc[0]; a = summ.iloc[1]
    ax.text(0.5, 0.02, f"|median| {abs(b.median_m):.3f} → {abs(a.median_m):.3f} m\n"
            f"MAE {b.mean_abs_m:.3f} → {a.mean_abs_m:.3f} m\n"
            f"RMSE {b.rmse_m:.3f} → {a.rmse_m:.3f} m\n"
            "the correction does NOT improve agreement",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.6, color=C["bad"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdf3f2", ec=C["bad"], lw=0.8))

    # (c) d against temporal separation — the diagnostic that now matters
    ax = axes[2]
    zero_line(ax)
    ax.axhline(c, color=C["icesat"], ls="-", lw=1.2, alpha=0.8)
    for i, r in d.iterrows():
        mk = "o" if r.qc_flag == "ok" else "X"
        ax.errorbar(abs(r.dt_hours), r.d_date_m, yerr=r.within_date_nmad_m, fmt=mk, ms=10,
                    color=C["swot"] if r.qc_flag == "ok" else C["muted"],
                    ecolor="#33393d", capsize=4, lw=1.2, mec="black", mew=0.8, zorder=5)
        ax.annotate(f"{r.date:%m-%d}", (abs(r.dt_hours), r.d_date_m), xytext=(7, 6),
                    textcoords="offset points", fontsize=6.6)
    ax.set_xlabel("|temporal separation| SWOT − ICESat-2  [h]")
    ax.set_ylabel("d  [m]")
    ax.set_title("(c) d against Δt")
    ax.set_xlim(-1, 17); ax.set_ylim(-0.30, 0.42)
    panel_label(ax, "c")
    ax.text(0.97, 0.96, "the best-matched overpass\n(Δt = 2.3 h) gives d ≈ 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6, color=C["ok"],
            fontweight="bold")

    h = [Line2D([], [], ls="none", marker="o", ms=7, color=C["swot"], mec="black",
                label="co-located overpass (ICESat-2 QC ok)"),
         Line2D([], [], ls="none", marker="X", ms=8, color=C["muted"], mec="black",
                label="ICESat-2 beam spread > 1 m (suspect)"),
         Line2D([], [], color="#33393d", lw=1.2, label="error bar: within-overpass NMAD")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Independent SWOT test of the gauge-derived ICESat-2 correction — "
                 f"one point per co-located overpass (n = {len(d)})", y=1.02, fontsize=9.5)
    save(fig, "Fig17_independent_swot_validation", FIG)
    print("  Fig17")


ALL["Fig17"] = fig17


if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    print("Rendering figures ...")
    for k in want:
        try:
            ALL[k]()
        except Exception as exc:  # keep going, report at the end
            import traceback
            print(f"  !! {k} FAILED: {exc}")
            traceback.print_exc(limit=3)
    print(f"Done -> {FIG}")


