#!/usr/bin/env python
"""QA STAGE 4 — where, exactly, does the result come from?

analysis_domain_map.png     what each candidate was classified as, and the zone
icesat2_temporal_coverage.png   which year each confirmed observation belongs to
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import contextily as ctx
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
# OSM asks for an identifying User-Agent and serves an "Access blocked" PLACEHOLDER
# IMAGE -- not an HTTP error -- to clients that do not send one. contextily's own
# default is a random "contextily-<hex>", so it must be overridden.
#
# `ctx.add_basemap(..., headers=...)` does NOT do this: add_basemap has no such
# parameter, so the dict is forwarded to imshow and raises AttributeError, which
# the try/except below then swallowed -- the map simply came out with no basemap
# and no message. The supported hook is the module-level constant.
ctx.tile.USER_AGENT = ("SWOT-DNIPRO-research-figure/1.0 "
                       "(static maps, a few tiles at zoom 8)")
TO_LL = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform
BREACH = pd.Timestamp("2023-06-06")
CCOL = {"DRY_EXPOSED_BED": GREEN, "WATER": BLUE,
        "SHORELINE_AMBIGUOUS": AMBER, "UNKNOWN": "#c9c9c9"}


def base(ax, zoom=8):
    ax.set_xlim(33.25, 35.45); ax.set_ylim(46.68, 47.95)
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", zoom=zoom, attribution=False,
                        source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        ax.text(0.995, 0.006, "© OpenStreetMap contributors", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.4, color="#333", zorder=20,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
    except Exception as e:
        # Loud, with the message: a bare type name hid an AttributeError from an
        # unsupported kwarg for a whole run of figures.
        print(f"   !! BASEMAP FAILED -- map has no basemap: {type(e).__name__}: {e}")
    ax.set_xlabel("°E"); ax.grid(alpha=0.15, color="white")


def outline(ax, geom, **kw):
    for g in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        x, y = g.exterior.xy
        ax.plot(x, y, **kw)


def main() -> None:
    m = pd.read_csv(CFG.TABLES / "qa3_s7_classification.csv", parse_dates=["date"])
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet")
    fp_m = shape(json.load(open(CFG.FIGDATA / "P20_reservoir_footprint.geojson"))["geometry"])
    fp = shp_transform(TO_LL, fp_m)
    zone = gpd.read_file(ROOT / "data/processed/bathymetry/qa3_analysis_zone.gpkg").to_crs(4326)

    s6 = m[(m.h_canopy == 0) & (m.veg_ph_count == 0) & (m.gnd_ph_count >= 50)
           & (m.match_dist_m <= 50)]
    s7 = s6[(s6.dry_class == "DRY_EXPOSED_BED") & (s6.date > BREACH)]

    # ---------------- analysis domain -----------------------------------
    fig, ax = plt.subplots(figsize=(15, 8))
    base(ax)
    # The pool is drawn as an OUTLINE ONLY and the zone as the one filled shape.
    # Filling both made a 10 % zone read as near-total coverage: at this scale a
    # pale pool fill and a hatched zone are not separable by eye, so the figure
    # overstated what 102 points support. Only one thing on this map is filled,
    # and it is the thing whose area is the point of the figure.
    outline(ax, fp, color=INK, lw=1.6, zorder=6)
    zone.plot(ax=ax, facecolor=GREEN, edgecolor=GREEN, alpha=0.55, lw=0.8, zorder=5)
    ax.scatter(sd.lon[::6], sd.lat[::6], s=0.9, color="#7a5c2e", alpha=0.30,
               zorder=4, lw=0)
    for k, c in CCOL.items():
        s = m[m.dry_class == k]
        if len(s):
            ax.scatter(s.lon, s.lat, s=6, color=c, alpha=0.55, lw=0, zorder=7)
    ax.scatter(s7.lon, s7.lat, s=52, facecolor="none", edgecolor=RED, lw=1.5, zorder=9)

    ax.legend(handles=[
        Patch(fc='none', ec=INK, lw=1.6, label="former reservoir pool (5 Jun 2023), "
              f"{fp_m.area/1e6:,.0f} km² — outline only"),
        Patch(fc=GREEN, alpha=.55, ec=GREEN, label=f"S7 analysis zone, 1 km buffer "
              f"({zone.to_crs(CFG.CRS_METRIC).area.sum()/1e6:,.0f} km², "
              f"{100*zone.to_crs(CFG.CRS_METRIC).area.sum()/fp_m.area:.0f} % of the pool)"),
        Line2D([], [], ls="", marker=".", color="#7a5c2e", label="old bathymetric soundings"),
        Line2D([], [], ls="", marker="o", color=CCOL["DRY_EXPOSED_BED"],
               label=f"candidate: confirmed dry ({(m.dry_class=='DRY_EXPOSED_BED').sum():,})"),
        Line2D([], [], ls="", marker="o", color=CCOL["WATER"],
               label=f"candidate: WATER on the pass date ({(m.dry_class=='WATER').sum():,})"),
        Line2D([], [], ls="", marker="o", color=CCOL["SHORELINE_AMBIGUOUS"],
               label=f"candidate: shoreline ambiguous ({(m.dry_class=='SHORELINE_AMBIGUOUS').sum():,})"),
        Line2D([], [], ls="", marker="o", color=CCOL["UNKNOWN"],
               label=f"candidate: no water mask within 20 d ({(m.dry_class=='UNKNOWN').sum():,})"),
        Line2D([], [], ls="", marker="o", mfc="none", mec=RED, mew=1.5,
               label=f"S7 CONFIRMED exposed bed ({len(s7)} pts, {s7.track.nunique()} tracks)"),
    ], fontsize=8.6, loc="upper left", framealpha=0.95)
    # The drawn zone uses ONE buffer radius, and the eye reads a merged buffer
    # union as more coverage than it is. The choice of radius is a modelling
    # decision, so the alternatives are put on the map itself rather than left in
    # a table the reader of the figure will not see.
    ax.text(0.985, 0.975, "coverage depends on the buffer radius:\n"
            "  250 m →   18 km²  (0.8 %)\n"
            "  500 m →   65 km²  (2.9 %)\n"
            "    1 km →  225 km²  (10.3 %)   ← drawn\n"
            "    2 km →  719 km²  (32.8 %)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            family="monospace", color=INK, zorder=21,
            bbox=dict(fc="white", ec=GREY, alpha=0.93, pad=4))
    ax.set_ylabel("°N")
    ax.set_title("QA4 · Analysis domain: what the −1.8 m actually rests on",
                 fontsize=14, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.012, "Geometric membership of the former pool is not enough: every "
            "candidate is checked against the Sentinel-2 water mask nearest in time to "
            "its own ICESat-2 pass.\nOnly the red-circled points are confirmed dry "
            "exposed bed and enter S7.", transform=ax.transAxes, fontsize=9,
            color=GREY, va="bottom")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"QA4_analysis_domain_map.{e}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/'QA4_analysis_domain_map.png'}")

    # ---------------- temporal coverage ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2),
                             gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax = axes[0]; base(ax)
    outline(ax, fp, color=INK, lw=1.4, zorder=6)
    ycol = {2023: "#c1402a", 2024: "#b07d27", 2025: "#236f8c"}
    for y, c in ycol.items():
        s = s7[s7.date.dt.year == y]
        if len(s):
            ax.scatter(s.lon, s.lat, s=46, color=c, alpha=0.9, lw=0.6,
                       edgecolor="white", zorder=8,
                       label=f"{y}  ({len(s)} pts, {s.track.nunique()} tracks)")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.set_ylabel("°N")
    ax.set_title("S7 confirmed observations by year", fontsize=12, loc="left")

    ax2 = axes[1]
    # top edge from the data: a hardcoded ceiling makes pd.cut return NaN for
    # anything beyond it, and groupby then drops those points silently.
    cbins = np.arange(0, np.ceil(s7.chainage_km.max() / 20) * 20 + 1, 20)
    cut = pd.cut(s7.chainage_km, cbins)
    assert cut.notna().all(), "chainage bins do not cover every S7 point"
    q = s7.groupby(cut, observed=True)
    g = q.agg(n=("dH", "size"), med=("dH", "median"))
    mid = [i.mid for i in g.index]
    ax2.barh(mid, g.n, height=15, color=GREY, alpha=0.5)
    ax2.set_xlabel("S7 points per 20 km reach"); ax2.set_ylabel("chainage from dam (km)")
    ax2.grid(alpha=0.25)
    a3 = ax2.twiny()
    # Reaches beyond the impounded pool are drawn detached: the 240-260 km cluster
    # is one track on one day, and joining it to the line would read as a
    # longitudinal trend that a single overpass cannot support.
    POOL_MAX_KM = 200.0
    inp = [m_ <= POOL_MAX_KM for m_ in mid]
    a3.plot([v for v, k in zip(g.med, inp) if k],
            [v for v, k in zip(mid, inp) if k], "o-", color=RED, ms=7, lw=2)
    a3.plot([v for v, k in zip(g.med, inp) if not k],
            [v for v, k in zip(mid, inp) if not k], "o", mfc="none", mec=AMBER,
            mew=2, ms=8, label="1 track, 1 day — flagged")
    if not all(inp):
        a3.legend(fontsize=7.4, loc="lower right")
    a3.axvline(s7.dH.median(), color=BLUE, ls="--", lw=1.6)
    a3.set_xlabel("median dH per reach (m)", color=RED)
    a3.set_xlim(-4, 1)
    ax2.set_title("dH is not confined to one reach", fontsize=12, loc="left")
    fig.suptitle(f"QA4 · Temporal and longitudinal coverage of S7   "
                 f"({len(s7)} points, {s7.track.nunique()} tracks, "
                 f"{s7.date.min():%b %Y}–{s7.date.max():%b %Y}, "
                 f"chainage {s7.chainage_km.min():.0f}–{s7.chainage_km.max():.0f} km)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"QA4_temporal_coverage.{e}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/'QA4_temporal_coverage.png'}")

    print("\n=== S7 per 20 km reach ===")
    print(g.to_string())

    # coverage is a modelling choice, so report the sensitivity rather than one number
    from shapely.ops import unary_union as _uu
    pm = gpd.GeoSeries(gpd.points_from_xy(s7.lon, s7.lat), crs=4326).to_crs(CFG.CRS_METRIC)
    print("\n=== coverage sensitivity to the buffer radius ===")
    rows = []
    for rad in (250, 500, 1000, 2000):
        z = _uu(pm.buffer(rad).values).intersection(fp_m)
        rows.append({"buffer_m": rad, "zone_km2": z.area / 1e6,
                     "coverage_pct": 100 * z.area / fp_m.area})
        print(f"  {rad:>5} m buffer -> {z.area/1e6:>7,.0f} km2  "
              f"({100*z.area/fp_m.area:>5.1f} % of the former pool)")
    pd.DataFrame(rows).to_csv(CFG.TABLES / "qa4_coverage_sensitivity.csv", index=False)
    print("\n  The result is anchored on 102 points along 28 tracks spanning")
    print(f"  chainage {s7.chainage_km.min():.0f}-{s7.chainage_km.max():.0f} km, i.e. the full")
    print("  length of the reservoir, but sampling only a few per cent of its AREA.")


if __name__ == "__main__":
    main()
