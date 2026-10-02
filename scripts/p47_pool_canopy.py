#!/usr/bin/env python
"""P47 -- canopy height and overgrowth of the former Kakhovka pool from ICESat-2 (plan 15, user request 2026-09-19).

Source: SlideRule/PhoREAL ATL08-type segments at 20 m (data_swot/processed/atl08/kakhovka_atl08_20m_raw.parquet
2023-09..2025-12 and kakhovka_atl08_20m_2026_raw.parquet when pulled). QC exactly as the audit pilot
(pilots/icesat2/build_segment_layers.py): terrain_ok = gnd_ph_count >= 4 & snowcover == 1 & |h_te| < 500;
canopy_ok = terrain_ok & veg_ph_count >= 3 & 0.5 <= h_canopy <= 45 & canopy_openness finite;
canopy_none = terrain_ok & veg_ph_count == 0 (a valid 0 m). h_canopy is height ABOVE local ground (no datum).

Products (former water surface of the pool only; ZONE_1 pool window grid from p43):
  outputs/tables/p47_pool_canopy_by_year.csv          per leaf-on season (Jun-Sep) and per bed-recession zone:
                                                      n segments, n canopy_ok, share with canopy >= 2 m / >= 4 m,
                                                      h_canopy p25/p50/p75/p95 of canopy_ok, n RGT passes
  outputs/rasters/roughness/pool/zone1_poolwin_icesat2_canopy_h_mean_<year>_250m.tif   mean h_canopy (canopy_ok U canopy_none) per 250 m cell
  outputs/rasters/roughness/pool/zone1_poolwin_icesat2_overgrowth_share_<year>_250m.tif share of segments with canopy >= 2 m (%), 255 no track
  outputs/rasters/roughness/pool/zone1_poolwin_icesat2_n_segments_<year>_250m.tif
  data/processed/atl08/pool_canopy_segments_20m.gpkg  (layer per year) the QC'd segments with all fields
Caveat (audit C-29): cross-year pairs on the same footprint do not exist (off-pointing), so year-to-year
differences are differences of SAMPLES on the same surface, not repeat measurements.
Filters (2026-09-19): --months (leaf-on 6-9 default, or all: season kept in `leaf_on`); dry_at_pass -- 2023 segments only after
the recession zone of the cell dried (p43 zoning dates), 2024+ only where the year's leaf-on water_share <= 50 %. The 2023
"canopy" share (2 %) is at the noise floor of the still-water control (`never` zone 0.2-1.6 %): do not publish 2023 heights.
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
import geopandas as gpd
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

ATL = CFG.BULK_ROOT / "data_swot/processed/atl08"
RO = ROOT / "outputs/rasters/roughness/pool"
CELL = 250.0
COLS = ["time", "rgt", "cycle", "gt", "spot", "segment_id", "h_te_median", "h_canopy", "h_max_canopy", "canopy_openness", "veg_ph_count", "gnd_ph_count", "snowcover", "solar_elevation", "geometry"]


TAG = "icesat2"


def load(pattern: str) -> gpd.GeoDataFrame:
    parts = []
    for f in sorted(ATL.glob(pattern)):
        t = pq.read_table(f, columns=COLS).to_pandas()
        g = gpd.GeoDataFrame(t, geometry=gpd.GeoSeries.from_wkb(t.pop("geometry"), crs="EPSG:4326")).to_crs(CFG.CRS_METRIC)
        g["source_file"] = f.name; parts.append(g)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CFG.CRS_METRIC)
    g["year"] = pd.to_datetime(g.time).dt.year; g["month"] = pd.to_datetime(g.time).dt.month
    g["terrain_ok"] = (g.gnd_ph_count >= 4) & (g.snowcover == 1) & (g.h_te_median.abs() < 500)
    g["canopy_ok"] = g.terrain_ok & (g.veg_ph_count >= 3) & (g.h_canopy >= 0.5) & (g.h_canopy <= 45) & np.isfinite(g.canopy_openness)
    g["canopy_none"] = g.terrain_ok & (g.veg_ph_count == 0)
    g["strong_beam"] = g.spot.isin([1, 3, 5])
    return g


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--pattern", default="kakhovka_atl08_20m*_raw.parquet"); ap.add_argument("--tag", default="icesat2")
    ap.add_argument("--months", default="6-9", help="'6-9' = leaf-on only (default); 'all' = every pass (user 2026-09-19: use all tracks; season kept as a field)")
    a = ap.parse_args(); global TAG; TAG = a.tag
    m0, m1 = (1, 12) if a.months == "all" else tuple(int(v) for v in a.months.split("-"))
    g = load(a.pattern); g["leaf_on"] = g.month.between(6, 9)
    with rasterio.open(RO / "zone1_poolwin_former_water_surface_mask.tif") as ds:
        fw = ds.read(1) == 1; tr = ds.transform; shape = fw.shape
    with rasterio.open(RO / "zone1_poolwin_bed_recession_zoning_2023.tif") as ds:
        zoning = ds.read(1)
    cols, rows = ~tr * (g.geometry.x.values, g.geometry.y.values)
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    inb = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    g = g[inb].copy(); cols, rows = cols[inb], rows[inb]
    g["in_former_water"] = fw[rows, cols]; g["recession_zone"] = zoning[rows, cols]
    g = g[g.in_former_water & g.month.between(m0, m1)].copy()
    # dry-at-pass (user 2026-09-19: 2023 "canopy" over still-wet bed is water/noise, take only after the water receded):
    # 2023 -> the cell's recession zone must have dried before the pass (p43 zoning: fast <= 06-30, mid <= 08-06, slow <= 09-08, never = water);
    # 2024+ -> the cell must not be water in that year's leaf-on S2 composite (water_share > 50 %)
    dry_by = {1: pd.Timestamp("2023-06-30"), 2: pd.Timestamp("2023-08-06"), 3: pd.Timestamp("2023-09-08")}
    t = pd.to_datetime(g.time).dt.tz_localize(None)
    dry = np.ones(len(g), bool)
    y23 = g.year.values == 2023
    for zc, d in dry_by.items():
        dry[y23 & (g.recession_zone.values == zc) & (t.values < np.datetime64(d))] = False
    dry[y23 & (g.recession_zone.values == 4)] = False
    for y in sorted(set(g.year.values) - {2023}):
        f = ROOT / "outputs/rasters/zone1/annual" / f"zone1_water_share_{y}_leafon_20m.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as ds:
            ws = ds.read(1); wtr = ds.transform
        c2, r2 = ~wtr * (g.geometry.x.values, g.geometry.y.values); c2 = np.floor(c2).astype(int); r2 = np.floor(r2).astype(int)
        ib = (c2 >= 0) & (c2 < ws.shape[1]) & (r2 >= 0) & (r2 < ws.shape[0]); v = np.zeros(len(g), "u1"); v[ib] = ws[r2[ib], c2[ib]]
        dry[(g.year.values == y) & (v != 255) & (v > 50)] = False
    g["dry_at_pass"] = dry
    print(f"dry-at-pass filter: kept {dry.sum()} of {len(g)} segments; 2023 kept {int((dry & y23).sum())} of {int(y23.sum())}")
    g = g[g.dry_at_pass].copy()
    ZL = {1: "fast", 2: "mid", 3: "slow", 4: "never", 0: "n/a"}
    rows_out = []
    for (y, z), s in g.groupby(["year", "recession_zone"]):
        ok = s[s.canopy_ok]; lab = s[s.canopy_ok | s.canopy_none]
        rows_out.append(dict(year=int(y), recession_zone=ZL.get(int(z), str(z)), n_segments=len(s), n_terrain_ok=int(s.terrain_ok.sum()), n_canopy_ok=len(ok), n_canopy_none=int(s.canopy_none.sum()),
                             n_passes_rgt_cycle=int(s.groupby(["rgt", "cycle"]).ngroups), share_canopy_ge2m=round(float((lab.h_canopy >= 2).mean()), 4) if len(lab) else np.nan,
                             share_canopy_ge4m=round(float((lab.h_canopy >= 4).mean()), 4) if len(lab) else np.nan,
                             h_canopy_p25=round(float(ok.h_canopy.quantile(.25)), 2) if len(ok) else np.nan, h_canopy_p50=round(float(ok.h_canopy.median()), 2) if len(ok) else np.nan,
                             h_canopy_p75=round(float(ok.h_canopy.quantile(.75)), 2) if len(ok) else np.nan, h_canopy_p95=round(float(ok.h_canopy.quantile(.95)), 2) if len(ok) else np.nan,
                             h_max_canopy_p95=round(float(ok.h_max_canopy.quantile(.95)), 2) if len(ok) else np.nan))
    for (y, lo), s in g.groupby(["year", "leaf_on"]):
        ok = s[s.canopy_ok]; lab = s[s.canopy_ok | s.canopy_none]
        rows_out.append(dict(year=int(y), recession_zone="ALL_leafon" if lo else "ALL_leafoff", n_segments=len(s), n_terrain_ok=int(s.terrain_ok.sum()), n_canopy_ok=len(ok), n_canopy_none=int(s.canopy_none.sum()),
                             n_passes_rgt_cycle=int(s.groupby(["rgt", "cycle"]).ngroups), share_canopy_ge2m=round(float((lab.h_canopy >= 2).mean()), 4) if len(lab) else np.nan,
                             share_canopy_ge4m=round(float((lab.h_canopy >= 4).mean()), 4) if len(lab) else np.nan,
                             h_canopy_p25=round(float(ok.h_canopy.quantile(.25)), 2) if len(ok) else np.nan, h_canopy_p50=round(float(ok.h_canopy.median()), 2) if len(ok) else np.nan,
                             h_canopy_p75=round(float(ok.h_canopy.quantile(.75)), 2) if len(ok) else np.nan, h_canopy_p95=round(float(ok.h_canopy.quantile(.95)), 2) if len(ok) else np.nan,
                             h_max_canopy_p95=round(float(ok.h_max_canopy.quantile(.95)), 2) if len(ok) else np.nan))
    for y, s in g.groupby("year"):
        ok = s[s.canopy_ok]; lab = s[s.canopy_ok | s.canopy_none]
        rows_out.append(dict(year=int(y), recession_zone="ALL", n_segments=len(s), n_terrain_ok=int(s.terrain_ok.sum()), n_canopy_ok=len(ok), n_canopy_none=int(s.canopy_none.sum()),
                             n_passes_rgt_cycle=int(s.groupby(["rgt", "cycle"]).ngroups), share_canopy_ge2m=round(float((lab.h_canopy >= 2).mean()), 4) if len(lab) else np.nan,
                             share_canopy_ge4m=round(float((lab.h_canopy >= 4).mean()), 4) if len(lab) else np.nan,
                             h_canopy_p25=round(float(ok.h_canopy.quantile(.25)), 2) if len(ok) else np.nan, h_canopy_p50=round(float(ok.h_canopy.median()), 2) if len(ok) else np.nan,
                             h_canopy_p75=round(float(ok.h_canopy.quantile(.75)), 2) if len(ok) else np.nan, h_canopy_p95=round(float(ok.h_canopy.quantile(.95)), 2) if len(ok) else np.nan,
                             h_max_canopy_p95=round(float(ok.h_max_canopy.quantile(.95)), 2) if len(ok) else np.nan))
    out = pd.DataFrame(rows_out).sort_values(["year", "recession_zone"]); out.to_csv(CFG.TABLES / f"p47_pool_canopy_by_year_{TAG}.csv", index=False)
    print(out[out.recession_zone == "ALL"].to_string(index=False))
    # 250 m rasters per year
    x0, y1 = tr.c, tr.f; ny, nx = int(np.ceil(shape[0] * 20 / CELL)), int(np.ceil(shape[1] * 20 / CELL)); tr250 = from_origin(x0, y1, CELL, CELL)
    lab = g[g.canopy_ok | g.canopy_none]
    cx = ((lab.geometry.x.values - x0) // CELL).astype(int); cy = ((y1 - lab.geometry.y.values) // CELL).astype(int)
    for y in sorted(lab.year.unique()):
        m = (lab.year.values == y) & (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        n = np.zeros((ny, nx), "i4"); hs = np.zeros((ny, nx), "f8"); ge2 = np.zeros((ny, nx), "i4")
        np.add.at(n, (cy[m], cx[m]), 1); np.add.at(hs, (cy[m], cx[m]), lab.h_canopy.values[m]); np.add.at(ge2, (cy[m], cx[m]), (lab.h_canopy.values[m] >= 2).astype(int))
        for name, arr, dtype, nd, q in ((f"{TAG}_canopy_h_mean_{y}_250m", np.where(n > 0, hs / np.maximum(n, 1), -9999).astype("f4"), "float32", -9999, "mean h_canopy [m above ground] of QC'd segments (canopy_ok U canopy_none), leaf-on"),
                                        (f"{TAG}_overgrowth_share_{y}_250m", np.where(n > 0, np.round(100 * ge2 / np.maximum(n, 1)), 255).astype("u1"), "uint8", 255, "share of segments with canopy >= 2 m, % (255 = no track)"),
                                        (f"{TAG}_n_segments_{y}_250m", np.clip(n, 0, 65535).astype("u2"), "uint16", 0, "QC'd segments per cell")):
            with rasterio.open(RO / f"zone1_poolwin_{name}.tif", "w", driver="GTiff", height=ny, width=nx, count=1, dtype=dtype, crs=CFG.CRS_METRIC, transform=tr250, nodata=nd, compress="deflate") as ds:
                ds.write(arr, 1); ds.update_tags(quantity=q, year=str(y), source="SlideRule PhoREAL atl08p 20 m", producer="p47_pool_canopy.py", caveat="on-track sample, not a wall-to-wall map; no repeat footprints across years (audit C-29)")
    gp = ATL / f"pool_canopy_segments_{TAG}.gpkg"
    for y, s in g.groupby("year"):
        s.drop(columns=["time"]).assign(utc=pd.to_datetime(s.time).dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_file(gp, layer=f"leafon_{y}", driver="GPKG")
    print(f"-> {gp}; rasters in {RO}")


if __name__ == "__main__":
    main()
