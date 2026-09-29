#!/usr/bin/env python
"""M1-M3 — maps of the drained lakebed.

M1  ATL08 bare-earth bed elevation (the surface that did not exist before)
M2  Sentinel-2 surface classification, pre-breach vs post-breach
M3  Kakhovka_SA_2.geojson vs the data-driven footprint -- settles the polygon's epoch
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
from matplotlib.colors import BoundaryNorm, ListedColormap
import contextily as ctx
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from affine import Affine
from rasterio import features
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD
from swot_dnipro import watermask as WM

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
SA2 = Path("/mnt/e/data_swot/Kakhovka_SA_2.geojson")
# both moved to the bulk volume; the repo-disk paths no longer exist, so the
# M2 figure could not be regenerated until these were re-pointed
MASKS = CFG.BULK_ROOT / "data_swot/processed/water_masks"
NDVI_DIR = CFG.BULK_ROOT / "data_swot/processed/spectral_indices"
POST_WSE = 5.20
OSM = {"User-Agent": "SWOT-DNIPRO-research-figure/1.0 (static maps, few tiles)"}
TO_LL = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform


def basemap(ax, zoom=9):
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", zoom=zoom, attribution=False,
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        headers=OSM, alpha=0.55)
        ax.text(0.995, 0.006, "© OpenStreetMap contributors", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.6, color="#333", zorder=12,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))
    except Exception as e:
        print(f"    [basemap unavailable: {type(e).__name__}]")


def outlines():
    sa2 = gpd.read_file(SA2).to_crs("EPSG:4326").geometry.union_all()
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp_m = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
            SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    return sa2, shp_transform(TO_LL, fp_m), fp_m


def draw_outline(ax, geom, **kw):
    gs = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for g in gs:
        x, y = g.exterior.xy
        ax.plot(x, y, **kw)


# ------------------------------------------------------------------ M1 ----
def m1_bed(sa2, fp_ll):
    d = pd.read_parquet(ROOT / "data/processed/atl08/kakhovka_atl08_terrain.parquet")
    bed = d[d.surface_class == "exposed_bed"]
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(33.25, 35.45); ax.set_ylim(46.68, 47.95)
    basemap(ax)
    draw_outline(ax, sa2, color=INK, lw=1.6, zorder=4, alpha=0.9)

    lv = [0, 2, 4, 5.2, 7, 9, 11, 13, 16, 20]
    cmap = ListedColormap(["#08306b", "#2171b5", "#6baed6", "#c6dbef",
                           "#fee0b6", "#fdae61", "#e6772e", "#b2521f", "#7f3b12"])
    s = ax.scatter(bed.lon, bed.lat, c=bed.H_terrain_common_m, s=3.4,
                   cmap=cmap, norm=BoundaryNorm(lv, cmap.N), zorder=6, lw=0)
    cb = fig.colorbar(s, ax=ax, pad=0.012, ticks=lv, extend="both")
    cb.set_label("ATL08 bare-earth bed elevation  (m, EGG2015 common frame)", fontsize=9.5)
    cb.ax.axhline(POST_WSE, color=RED, lw=2.4)
    cb.ax.text(1.9, POST_WSE, f" post-breach\n water {POST_WSE:.1f} m", color=RED,
               fontsize=8, va="center", transform=cb.ax.get_yaxis_transform())

    below = (bed.H_terrain_common_m <= POST_WSE)
    ax.set_title("M1 · The bare-earth surface of the drained Kakhovka lakebed",
                 fontsize=14, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.012,
            f"{len(bed):,} ATL08 ground segments, {bed.date.nunique()} dates, "
            f"{bed.date.min():%b %Y}–{bed.date.max():%b %Y}. Blue = below the residual water "
            f"level ({below.mean()*100:.0f} % of the bed), orange = above it "
            f"({(~below).mean()*100:.0f} %).\nThis surface did not exist as a measurable "
            f"dataset before the breach: GLO-30 captured the filled reservoir as flat water "
            f"at ~16 m.", transform=ax.transAxes, fontsize=9, color=GREY, va="bottom")
    ax.set_xlabel("°E"); ax.set_ylabel("°N"); ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"M1_lakebed_elevation.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"-> {CFG.FIG/'M1_lakebed_elevation.png'}")


# ------------------------------------------------------------------ M2 ----
def m2_classes(sa2):
    wm = pd.read_csv(CFG.TABLES / "water_mask_summary.csv")
    wm["dt"] = pd.to_datetime(wm.sensing_time, format="%Y%m%dT%H%M%S")
    wm["date"] = wm.dt.dt.strftime("%Y-%m-%d")
    picks = [("2023-06-05", "PRE-breach  ·  5 Jun 2023"),
             ("2025-08-31", "POST-breach  ·  31 Aug 2025")]
    cols = ["#00000000", BLUE, GREEN, "#9bbf7a", AMBER]
    cmap = ListedColormap(cols)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for ax, (dt, ttl) in zip(axes, picks):
        g = wm[wm.date == dt]
        drawn = False
        for r in g.itertuples():
            npz = MASKS / f"{r.name}.npz"
            tif = NDVI_DIR / f"{r.name}_ndvi.tif"
            if not (npz.exists() and tif.exists()):
                continue
            z = np.load(npz)
            with rasterio.open(tif) as src:
                ndvi = src.read(1)
                tr, crs = src.transform, src.crs
            ndwi, mndwi, scl, water = (z["ndwi"].astype("f4"), z["mndwi"].astype("f4"),
                                       z["scl"], z["mask"])
            valid = ((ndvi > -1) & (ndwi > -1) & (mndwi > -1)
                     & ~np.isin(scl, WM.SCL_REJECT))
            cls = np.zeros(ndvi.shape, "u1")
            cls[valid] = 4
            cls[valid & (ndvi >= 0.15)] = 3
            cls[valid & (ndvi >= 0.30)] = 2
            cls[valid & water] = 1
            step = 3
            c = cls[::step, ::step]
            h, w = c.shape
            t2 = tr * Affine.scale(step, step)
            xs = t2.c + np.arange(w) * t2.a
            ys = t2.f + np.arange(h) * t2.e
            tf = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            X, Y = np.meshgrid(xs, ys)
            lon, lat = tf.transform(X, Y)
            ax.pcolormesh(lon, lat, np.ma.masked_where(c == 0, c), cmap=cmap,
                          vmin=0, vmax=4, shading="nearest", zorder=5, rasterized=True)
            drawn = True
        if not drawn:
            ax.text(.5, .5, "no scene", transform=ax.transAxes, ha="center")
        ax.set_xlim(33.25, 35.45); ax.set_ylim(46.68, 47.95)
        basemap(ax, zoom=8)
        draw_outline(ax, sa2, color=INK, lw=1.5, zorder=7)
        ax.set_title(ttl, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel("°E"); ax.grid(alpha=0.15, color="white")
    axes[0].set_ylabel("°N")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(fc=BLUE, label="water"),
                            Patch(fc=GREEN, label="vegetation  NDVI ≥ 0.30"),
                            Patch(fc="#9bbf7a", label="sparse veg  0.15–0.30"),
                            Patch(fc=AMBER, label="bare  NDVI < 0.15")],
                   fontsize=9, loc="lower right", framealpha=0.95)
    fig.suptitle("M2 · Sentinel-2 surface classification of the former pool "
                 "(water mask unchanged; NDVI splits the non-water remainder)",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"M2_surface_classification.{e}", dpi=190, bbox_inches="tight")
    plt.close(fig); print(f"-> {CFG.FIG/'M2_surface_classification.png'}")


# ------------------------------------------------------------------ M3 ----
def m3_polygon(sa2, fp_ll, fp_m):
    sa2_m = shp_transform(
        pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform,
        sa2)
    inter = sa2_m.intersection(fp_m); union = sa2_m.union(fp_m)
    iou = inter.area / union.area
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(33.25, 35.45); ax.set_ylim(46.68, 47.95)
    basemap(ax, zoom=8)
    for g in (fp_ll.geoms if fp_ll.geom_type == "MultiPolygon" else [fp_ll]):
        x, y = g.exterior.xy
        ax.fill(x, y, color=BLUE, alpha=0.30, zorder=5)
    draw_outline(ax, fp_ll, color=BLUE, lw=1.4, zorder=6)
    draw_outline(ax, sa2, color=RED, lw=2.0, zorder=7)
    ax.plot([], [], color=BLUE, lw=6, alpha=0.45,
            label=f"data-driven footprint, largest connected water 5 Jun 2023 "
                  f"({fp_m.area/1e6:,.0f} km²)")
    ax.plot([], [], color=RED, lw=2.2,
            label=f"Kakhovka_SA_2.geojson ({sa2_m.area/1e6:,.0f} km²)")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax.set_title("M3 · The reservoir polygon's epoch, settled", fontsize=14,
                 fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.012,
            f"Kakhovka_SA_2 was carried as 'epoch uncertain, decorative only'. Against an "
            f"independently derived full-pool observation it gives IoU = {iou:.3f}, "
            f"{100*inter.area/sa2_m.area:.1f} % of it inside the data footprint and "
            f"{100*inter.area/fp_m.area:.1f} % the other way, with areas matching to "
            f"{abs(sa2_m.area-fp_m.area)/fp_m.area*100:.1f} %.\nIts attributes carry "
            f"VolumeCorr = 17 455 (≈17.5 km³, the reservoir's full volume). "
            f"It is a PRE-BREACH FULL-POOL mask.",
            transform=ax.transAxes, fontsize=9, color=GREY, va="bottom")
    ax.set_xlabel("°E"); ax.set_ylabel("°N"); ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"M3_polygon_epoch.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"-> {CFG.FIG/'M3_polygon_epoch.png'}")
    pd.DataFrame([{"sa2_km2": sa2_m.area / 1e6, "footprint_km2": fp_m.area / 1e6,
                   "intersection_km2": inter.area / 1e6, "iou": iou,
                   "sa2_inside_footprint_pct": 100 * inter.area / sa2_m.area,
                   "footprint_inside_sa2_pct": 100 * inter.area / fp_m.area}]
                 ).to_csv(CFG.TABLES / "reservoir_polygon_comparison.csv", index=False)


def main():
    sa2, fp_ll, fp_m = outlines()
    m3_polygon(sa2, fp_ll, fp_m)
    m1_bed(sa2, fp_ll)
    m2_classes(sa2)


if __name__ == "__main__":
    main()
