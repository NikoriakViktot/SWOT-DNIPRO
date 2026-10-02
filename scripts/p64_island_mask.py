#!/usr/bin/env python
"""P64 -- islands and never-inundated land inside the Kakhovka pool polygon, as an explicit CONSTRAINT for the bed kriging.

Why (user, 2026-09-19: "острови як обмеження потрібно робити щоб там не було кригінга ... не видаляти острови, вони мають
залишитись з фабдем"). hist14 krige the bed over everything inside the registry subzones, islands included, so interpolated
bathymetry is written under dry land; p55 then stacks a gap fill and a feather on top of the same cells. p62/p63 measured the
result: 30.8 km² of gap fill sits 21.8 m BELOW FABDEM and 133 km² of feather 4.4 m below, at cells where FABDEM agrees with
GEDI ground to ~0.5 m. This mask is the constraint that stops the kriging there. Nothing is deleted from the terrain: the
island keeps its own FABDEM elevation in every product that represents terrain (p55 source class 3), the bed reconstruction
simply declines to invent a bathymetric value under dry land.

Three criteria, all from data, no hand-drawn geometry; a cell is an island only if all three agree.
  OPTICAL   never observed as water in the pre-breach S2 record (2019-2022, 9 seasonal windows), water share normalised by the
            OBSERVED area: NO_DATA never votes "dry" (memory rule feedback-not-observed-is-not-dry).
  SUPPORT   minimum optical support (user's requirement): n_valid >= MIN_OBS summed over the windows AND present in >= MIN_YEARS
            distinct years. Below that the cell is UNKNOWN, never "island".
  ELEVATION FABDEM above POOL_WS_MAX_M. Inside the pool polygon FABDEM is the pre-breach water surface: measured p1 14.46 m,
            p95 14.57 m, p99 22.65 m, so 17.0 m separates the water surface from islands and valley slopes (p50 32.3 m there).

RESOLUTION. FABDEM is 1 arcsec native = 21.0 m in x and 30.7 m in y at 47.3 N, so the project's 20 m frame oversamples it and a
classification decision taken on resampled values is not independent information (user: "потрібно робити так само як і в фабдем
а там напевно 25 або 30 м"). The mask is therefore built and stored on a 30 m grid -- the same cell size as hist14's finest grid
-- and every consumer resamples it with NEAREST, never bilinear.

Outputs
  outputs/rasters/zone1/zone1_pool_islands_30m.tif   uint8: 0 water/bed, 1 island, 2 UNKNOWN (insufficient optical support)
  outputs/tables/p64_island_mask.csv                 areas per criterion, intersection, UNKNOWN, sensitivity to the threshold and MIN_OBS
  outputs/figures/p64_island_mask.png                the mask on an Esri satellite basemap (rule feedback-check-geometry-on-a-basemap)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

CELL = 30.0                       # FABDEM native ground sample (30.7 m N-S at 47.3 N); also hist14's finest grid
POOL_WS_MAX_M = 17.0              # above this, FABDEM inside the pool is terrain, not the pre-breach water surface
WATER_SHARE_MAX = 5.0             # "never water": observed water share of the window, per cent
MIN_OBS = 12                      # minimum valid S2 observations summed over the pre-breach windows
MIN_YEARS = 2                     # ... and present in at least this many distinct pre-breach years
MIN_COMPONENT_CELLS = 4           # drop island specks smaller than this (4 cells of 30 m = 0.0036 km2)
WINDOW_BUF_M = 100.0              # classify a little OUTSIDE the registry polygon too: p55 clips the pool bed to polygon+60 m,
                                  # and that annulus is bank. A gate that is spatial must cover every cell the bed can reach.
MAX_HOLE_CELLS = 2                # close pinholes inside an island
ANN = ROOT / "outputs/rasters/zone1/annual"
WINDOWS = [(2019, "leafon"), (2020, "leafon"), (2021, "spring"), (2021, "leafon"), (2021, "autumn"),
           (2022, "spring"), (2022, "leafon"), (2022, "autumn"), (2022, "winter")]
FAB = CFG.BULK_ROOT / "terrain/ZONE_1_KAKHOVKA_LOWER_DNIPRO/fabdem_evrf2019_20m.tif"
OUT_TIF = ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif"


def to_grid(path: Path, shape, transform, resampling, dtype="f4", src_nodata=None):
    """Read a raster onto the mask grid. NEAREST for anything that feeds a decision."""
    dst = np.full(shape, np.nan, "f4")
    with rasterio.open(path) as ds:
        reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=transform, dst_crs=CFG.CRS_METRIC,
                  resampling=resampling, src_nodata=ds.nodata if src_nodata is None else src_nodata, dst_nodata=np.nan)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-max", type=float, default=POOL_WS_MAX_M)
    ap.add_argument("--min-obs", type=int, default=MIN_OBS)
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    pool = SD.load_utm("reservoir_full_pool_prebreach"); win = pool.buffer(WINDOW_BUF_M)
    g = SD.build_grid(win, CELL, what="p64 island mask")
    gx, gy = g["gx"], g["gy"]; shape = (len(gy), len(gx))
    tr = from_origin(gx[0] - CELL / 2, gy[-1] + CELL / 2, CELL, CELL)   # build_grid returns gy ASCENDING; a north-up raster starts at gy[-1]
    SD.assert_covers(win, gx, gy, what="p64 island mask", cell=CELL)
    in_win = features.rasterize([(win, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    in_pool = features.rasterize([(pool, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    px = CELL * CELL / 1e6
    print(f"grid {shape[1]}x{shape[0]} at {CELL:.0f} m; pool {in_pool.sum() * px:,.1f} km2, "
          f"analysis window (+{WINDOW_BUF_M:.0f} m) {in_win.sum() * px:,.1f} km2")

    # ---------------- optical: never observed as water, with support
    n_obs = np.zeros(shape, "f4"); n_wet_win = np.zeros(shape, "f4"); n_win_obs = np.zeros(shape, "f4"); years_seen = {}
    for yr, season in WINDOWS:
        ws = ANN / f"zone1_water_share_{yr}_{season}_20m.tif"; nv = ANN / f"zone1_n_valid_{yr}_{season}_20m.tif"
        if not (ws.exists() and nv.exists()):
            print(f"  missing window {yr} {season} -- skipped"); continue
        share = to_grid(ws, shape, tr, Resampling.nearest, src_nodata=255)
        valid = to_grid(nv, shape, tr, Resampling.nearest, src_nodata=0)
        obs = np.isfinite(valid) & (valid > 0) & np.isfinite(share)
        n_obs += np.where(obs, valid, 0.0); n_win_obs += obs
        n_wet_win += obs & (share > WATER_SHARE_MAX)                      # observed AND wet in this window
        years_seen.setdefault(yr, np.zeros(shape, bool))
        years_seen[yr] |= obs
    n_years = np.sum([v for v in years_seen.values()], axis=0).astype("f4")
    support = (n_obs >= a.min_obs) & (n_years >= MIN_YEARS)
    never_water = support & (n_wet_win == 0)
    print(f"optical: median support {np.median(n_obs[in_pool]):.0f} obs / {np.median(n_years[in_pool]):.0f} years inside the pool; "
          f"cells with support {support[in_pool].sum() * px:,.1f} km2, never water {never_water[in_pool].sum() * px:,.1f} km2")

    # ---------------- elevation: FABDEM above the pre-breach water surface (NEAREST: a decision, not a height)
    fab = to_grid(FAB, shape, tr, Resampling.nearest)
    high = np.isfinite(fab) & (fab > a.ws_max)
    print(f"elevation: FABDEM > {a.ws_max:.1f} m inside the pool {np.logical_and(high, in_pool).sum() * px:,.1f} km2 "
          f"(FABDEM in pool: p1 {np.nanpercentile(fab[in_pool], 1):.2f}, p50 {np.nanpercentile(fab[in_pool], 50):.2f}, p95 {np.nanpercentile(fab[in_pool], 95):.2f} m)")

    # ---------------- combine + clean
    island = in_win & never_water & high
    lab, n = ndimage.label(island)
    if n:
        sizes = np.bincount(lab.ravel()); small = np.isin(lab, np.flatnonzero(sizes < MIN_COMPONENT_CELLS)) & (lab > 0)
        island &= ~small
    filled = ndimage.binary_fill_holes(island); holes = filled & ~island
    hl, hn = ndimage.label(holes)
    if hn:
        hs = np.bincount(hl.ravel()); island |= np.isin(hl, np.flatnonzero(hs <= MAX_HOLE_CELLS)) & (hl > 0)
    unknown = in_win & ~support & high & ~island                          # high ground we cannot corroborate optically
    out = np.where(island, 1, np.where(unknown, 2, 0)).astype("u1"); out[~in_win] = 0
    nl = ndimage.label(island)[1]
    print(f"LAND inside the window: {island.sum() * px:,.2f} km2 in {nl} components "
          f"({(island & in_pool).sum() * px:,.2f} km2 inside the registry polygon = islands, "
          f"{(island & ~in_pool).sum() * px:,.2f} km2 in the +{WINDOW_BUF_M:.0f} m annulus = bank); "
          f"UNKNOWN (high but unsupported) {unknown.sum() * px:,.2f} km2")

    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="uint8", crs=CFG.CRS_METRIC,
                transform=tr, nodata=255, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT_TIF, "w", **prof) as ds:
        ds.write(out, 1)
        ds.update_tags(producer="p64_island_mask.py", classes="0 water/bed | 1 land (island inside the polygon, bank in the annulus) (never water AND FABDEM > threshold AND supported) | 2 UNKNOWN (high but optically unsupported)",
                       cell_m=f"{CELL:.0f}", note="FABDEM native 1 arcsec = 21.0 x 30.7 m at 47.3N; consumers must resample NEAREST",
                       pool_ws_max_m=f"{a.ws_max}", water_share_max_pct=f"{WATER_SHARE_MAX}", min_obs=f"{a.min_obs}", min_years=f"{MIN_YEARS}",
                       windows=";".join(f"{y}_{s}" for y, s in WINDOWS))

    # ---------------- sensitivity table
    rows = [dict(quantity="pool polygon", km2=round(in_pool.sum() * px, 2)),
            dict(quantity=f"analysis window (pool +{WINDOW_BUF_M:.0f} m)", km2=round(in_win.sum() * px, 2)),
            dict(quantity="LAND inside the registry polygon (islands)", km2=round((island & in_pool).sum() * px, 2)),
            dict(quantity="LAND in the annulus (bank the bed spills onto)", km2=round((island & ~in_pool).sum() * px, 2)),
            dict(quantity="optical support (n_obs >= MIN_OBS and >= 2 years)", km2=round((in_win & support).sum() * px, 2)),
            dict(quantity="never observed as water (supported)", km2=round((in_win & never_water).sum() * px, 2)),
            dict(quantity=f"FABDEM > {a.ws_max} m", km2=round((in_win & high).sum() * px, 2)),
            dict(quantity="ISLAND = both criteria, cleaned", km2=round(island.sum() * px, 2), n_components=nl),
            dict(quantity="UNKNOWN (high, optically unsupported)", km2=round(unknown.sum() * px, 2))]
    for t in (16.0, 17.0, 18.0, 19.0):
        h = np.isfinite(fab) & (fab > t)
        rows.append(dict(quantity=f"sensitivity: island at FABDEM > {t} m", km2=round((in_win & never_water & h).sum() * px, 2), threshold_m=t))
    for mo in (6, 12, 20, 30):
        s = (n_obs >= mo) & (n_years >= MIN_YEARS); nw = s & (n_wet_win == 0)
        rows.append(dict(quantity=f"sensitivity: island at MIN_OBS = {mo}", km2=round((in_win & nw & high).sum() * px, 2), min_obs=mo))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p64_island_mask.csv", index=False)
    print(T.to_string(index=False))

    # ---------------- figure on a satellite basemap (mandatory geometry check)
    if a.no_figure:
        return
    try:
        import contextily as cx
        import geopandas as gpd
        shapes = [s for s, v in features.shapes(island.astype("u1"), mask=island, transform=tr) if v == 1]
        from shapely.geometry import shape as shp
        G = gpd.GeoDataFrame(geometry=[shp(s) for s in shapes], crs=CFG.CRS_METRIC).to_crs(3857)
        P = gpd.GeoSeries([pool], crs=CFG.CRS_METRIC).to_crs(3857)
        fig, ax = plt.subplots(figsize=(20, 11))
        P.boundary.plot(ax=ax, color="#1f78b4", lw=1.0, label="registry pool polygon")
        G.plot(ax=ax, facecolor="#e31a1c", edgecolor="#e31a1c", alpha=0.75, label=f"islands ({island.sum() * px:.1f} km²)")
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs=3857, attribution_size=6)
        ax.set_title(f"P64 islands inside the pre-breach pool: never water 2019–2022 AND FABDEM > {a.ws_max:.0f} m, {CELL:.0f} m grid")
        ax.legend(loc="lower right", fontsize=9); ax.set_axis_off()
        fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p64_island_mask.png", dpi=110)
        print("-> outputs/figures/p64_island_mask.png")
    except Exception as e:
        print(f"figure skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
