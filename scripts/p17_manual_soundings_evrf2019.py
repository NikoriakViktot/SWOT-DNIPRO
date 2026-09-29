#!/usr/bin/env python
"""P17 — manual chart soundings, converted to EVRF2019 bed elevation.

The 1,415 hand-clicked points (`depth_manual/manual_depth_points_POINT.gpkg`,
13 physically implausible points already removed this session) are, per the
user's own confirmation, already in the Baltic height system -- read off a
paper chart, EPSG:32636 horizontally. So a depth below that Baltic reference
converts to a Baltic bed elevation by simple negation, and from there to
EVRF2019 by the same EPSG:9902 grid step used for the gauges:

    H_bed_bs77_m     = -depth_m
    H_bed_evrf2019_m = H_bed_bs77_m + delta_epsg9902_m(lon, lat)

UNLIKE THE GAUGES, THIS IS SAMPLED PER POINT, NOT PER STATION. The gauges are
two fixed locations; these 1,415 points are spread across ~140 km, so one
`delta_epsg9902_m` value would not be valid everywhere. `load_grid`/
`sample_grid` from `part1_gauge_rereference.py` already do exactly this kind
of bilinear grid sampling -- reused here, not reimplemented.

ONE RESIDUAL CAVEAT, CARRIED FORWARD, NOT RESOLVED. The user said "Baltic",
not "BS-77" specifically. Older Soviet river charts sometimes carry BS-42
instead. EPSG:9902 transforms from BS-77. This is recorded as an explicit
column on every point (`vertical_datum_assumption`) rather than silently
assumed away -- if a BS-42/BS-77 offset at these dates/locations turns out to
be non-negligible, every downstream row that used this file is traceable.

Outputs
-------
data/processed/bathymetry/manual_soundings_evrf2019.parquet
outputs/tables/p17_manual_soundings_evrf2019_summary.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from part1_gauge_rereference import load_grid, sample_grid

POINTS = Path("/mnt/f/data_kakhovka_dem_swot/depth_manual/manual_depth_points_POINT.gpkg")
OUT_PQ = ROOT / "data/processed/bathymetry/manual_soundings_evrf2019.parquet"
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")


def median_nn_spacing_m(gdf: gpd.GeoDataFrame) -> float:
    if len(gdf) < 2:
        return float("nan")
    xy = np.c_[gdf.geometry.x.values, gdf.geometry.y.values]
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def main() -> None:
    print("=" * 78)
    print("P17 — manual chart soundings -> EVRF2019 bed elevation")
    print("=" * 78)

    gdf = gpd.read_file(POINTS)
    n_total = len(gdf)
    d = gdf[~gdf.geometry.isna() & ~gdf.depth_m.isna()].copy()
    print(f"  {n_total} rows -> {len(d)} usable (geometry + depth_m)")

    lon = d.geometry.to_crs(4326).x.values
    lat = d.geometry.to_crs(4326).y.values
    z, head = load_grid()
    delta = sample_grid(z, head, lon, lat)

    d["H_bed_bs77_m"] = -d.depth_m
    d["delta_epsg9902_m"] = delta
    d["H_bed_evrf2019_m"] = d.H_bed_bs77_m + d.delta_epsg9902_m
    d["vertical_datum_assumption"] = (
        "Baltic system per user confirmation (paper chart); "
        "BS-77 assumed for the EPSG:9902 step, BS-42 not independently ruled out")

    for z_name in ZONES:
        geom = SD.load_utm(z_name)
        d[f"in_{z_name}"] = d.geometry.within(geom)

    x = d.geometry.x.values
    y = d.geometry.y.values
    out = pd.DataFrame({
        "x": x, "y": y, "lon": lon, "lat": lat,
        "sheet_id": d.sheet_id.values, "raw_label": d.raw_label.values,
        "depth_m": d.depth_m.values, "confidence": d.confidence.values,
        "H_bed_bs77_m": d.H_bed_bs77_m.values,
        "delta_epsg9902_m": d.delta_epsg9902_m.values,
        "H_bed_evrf2019_m": d.H_bed_evrf2019_m.values,
        "vertical_datum_assumption": d.vertical_datum_assumption.values,
    })
    for z_name in ZONES:
        out[z_name] = d[f"in_{z_name}"].values

    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PQ, index=False)
    print(f"\n-> {OUT_PQ}  ({len(out):,} points)")

    print(f"\n  H_bed_evrf2019_m: median {out.H_bed_evrf2019_m.median():.2f} m, "
          f"range {out.H_bed_evrf2019_m.min():.2f} .. {out.H_bed_evrf2019_m.max():.2f} m")

    rows = []
    for z_name in ZONES:
        sub = gpd.GeoDataFrame(out[out[z_name]], geometry=d[d[f"in_{z_name}"]].geometry.values)
        n = len(sub)
        spacing = median_nn_spacing_m(sub) if n >= 2 else float("nan")
        rows.append(dict(
            zone=z_name, n_points=n,
            depth_min_m=float(sub.depth_m.min()) if n else np.nan,
            depth_max_m=float(sub.depth_m.max()) if n else np.nan,
            median_nn_spacing_m=spacing))
        print(f"  {z_name:32s} n={n:5d}  median NN spacing {spacing:7.1f} m")

    sm = pd.DataFrame(rows)
    sm_path = CFG.TABLES / "p17_manual_soundings_evrf2019_summary.csv"
    sm.to_csv(sm_path, index=False)
    print(f"\n-> {sm_path}")


if __name__ == "__main__":
    main()
