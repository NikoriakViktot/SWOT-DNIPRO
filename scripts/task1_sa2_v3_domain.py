#!/usr/bin/env python
"""TASK 1 (operator, 2026-09-11) -- revise the SA_2 study-domain polygon to
include the upstream hydrological control node: gauge 80957 (Zaporizhzhia,
right bank, 35.08208E/47.87204N) and the DniproHES/Zaporizhzhia river
cross-section.

Method (NOT an arbitrary rectangle):
  1. Start from SA_2 (reservoir_full_pool_prebreach).
  2. Add every pixel classified SYSTEMATIC_SA2_OMISSION (class 4) by the
     18-date multi-date consensus (p1c_v2_full_consensus.py) -- this is the
     concrete water evidence the disagreement map already showed
     concentrated almost entirely at the NE/Zaporizhzhia tip.
  3. Add any CONSISTENTLY_WATER (class 5) pixel not already inside SA_2.
  4. Explicitly guarantee gauge 80957 is enclosed: if it falls outside the
     step-3 union, connect it with a buffered corridor from the nearest
     point on the union boundary (never a disconnected blob).
  5. Add a river-corridor buffer around the straight line from the SA_2 NE
     tip through the gauge and a short distance past it (approximating the
     DniproHES tailrace vicinity, since no independently verified DniproHES
     dam-axis coordinate exists in this project -- documented, not hidden).

Outputs
-------
data/processed/study_domain/SA_2_v3.gpkg
outputs/analysis/upstream_boundary_revision.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from rasterio import features
from shapely.geometry import Point, LineString, shape
from shapely.ops import transform as shp_transform, unary_union, nearest_points

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

RASTER_DIR = CFG.ROOT / "data" / "processed" / "study_domain"
OUT_DIR = CFG.ROOT / "data" / "processed" / "study_domain"
ANALYSIS_DIR = CFG.ROOT / "outputs" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

GAUGE_80957 = Point(35.08208, 47.87204)   # lon, lat, EPSG:4326
CORRIDOR_BUFFER_M = 1500.0                 # half-width of the connecting corridor
CORRIDOR_EXTEND_M = 3000.0                 # how far past the gauge to extend upstream
CLOSE_BUFFER_M = 150.0                     # small buffer to close raster-vector gaps

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform
_TF_M_TO_4326 = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform


def polygonize_class(path: Path, values: set[int]):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        transform = ds.transform
    mask = np.isin(arr, list(values))
    if not mask.any():
        return None
    polys = [shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask,
             transform=transform) if v == 1]
    return unary_union(polys) if polys else None


def main() -> None:
    sa2 = SD.load("reservoir_full_pool_prebreach")
    sa2_m = shp_transform(_TF_4326_TO_M, sa2)
    old_area_km2 = sa2_m.area / 1e6

    cls_path = RASTER_DIR / "p1c_v2_class.tif"
    if not cls_path.exists():
        raise SystemExit(f"missing {cls_path} -- run p1c_v2_full_consensus.py first")

    omission = polygonize_class(cls_path, {4})       # SYSTEMATIC_SA2_OMISSION
    consistently_water = polygonize_class(cls_path, {5})  # CONSISTENTLY_WATER

    pieces = [sa2_m]
    if omission is not None:
        pieces.append(omission.buffer(CLOSE_BUFFER_M))
    if consistently_water is not None:
        pieces.append(consistently_water.buffer(CLOSE_BUFFER_M))
    union1 = unary_union(pieces).buffer(0)

    gauge_m = shp_transform(_TF_4326_TO_M, GAUGE_80957)
    gauge_inside_step1 = union1.contains(gauge_m)
    dist_to_union1 = gauge_m.distance(union1)
    print(f"after adding omission+consistently-water evidence: area="
          f"{union1.area/1e6:,.1f} km2, gauge_80957 inside={gauge_inside_step1}, "
          f"distance to boundary={dist_to_union1:.0f} m")

    # NE-most point of the union so far (approximation of "upstream tip")
    ext = union1.exterior if hasattr(union1, "exterior") else max(
        union1.geoms, key=lambda g: g.area).exterior
    coords = np.array(ext.coords)
    ne_idx = np.argmax(coords[:, 0] + coords[:, 1])  # crude NE-most proxy (max x+y)
    ne_point_m = coords[ne_idx]

    # corridor connecting the current NE tip -> gauge -> a short distance beyond
    line = LineString([ne_point_m, (gauge_m.x, gauge_m.y)])
    # extend the line past the gauge by CORRIDOR_EXTEND_M along its own bearing
    dx = gauge_m.x - ne_point_m[0]
    dy = gauge_m.y - ne_point_m[1]
    length = (dx**2 + dy**2) ** 0.5
    if length > 0:
        ux, uy = dx / length, dy / length
        far_point = (gauge_m.x + ux * CORRIDOR_EXTEND_M, gauge_m.y + uy * CORRIDOR_EXTEND_M)
        line = LineString([ne_point_m, (gauge_m.x, gauge_m.y), far_point])
    corridor = line.buffer(CORRIDOR_BUFFER_M)

    sa2_v3_m = unary_union([union1, corridor]).buffer(0)
    # never let the corridor construction escape the vetted study domain
    study_m = shp_transform(_TF_4326_TO_M, SD.load("study_domain_full"))
    sa2_v3_m = sa2_v3_m.intersection(study_m.buffer(5000))

    new_area_km2 = sa2_v3_m.area / 1e6
    added_km2 = new_area_km2 - old_area_km2
    gauge_inside_final = sa2_v3_m.contains(gauge_m)
    dist_final = gauge_m.distance(sa2_v3_m.boundary) if gauge_inside_final else gauge_m.distance(sa2_v3_m)

    sa2_v3 = shp_transform(_TF_M_TO_4326, sa2_v3_m)
    minx, miny, maxx, maxy = sa2_v3.bounds
    upstream_most = (maxx, maxy)  # NE corner of bounds, proxy for "upstream-most"

    print(f"\nSA_2_v3: old_area={old_area_km2:,.1f} km2  new_area={new_area_km2:,.1f} km2  "
          f"added={added_km2:,.1f} km2")
    print(f"gauge 80957 inside SA_2_v3: {gauge_inside_final}  "
          f"distance to boundary: {dist_final:.0f} m")
    print(f"upstream-most bound (lon,lat): {upstream_most}")
    print(f"SA_2_v3 bounds: lon [{minx:.4f},{maxx:.4f}]  lat [{miny:.4f},{maxy:.4f}]")
    print(f"(reference) SA_2 original bounds: lon max {sa2.bounds[2]:.4f}  lat max {sa2.bounds[3]:.4f}")

    # ---- write output -------------------------------------------------------
    gdf = gpd.GeoDataFrame({"name": ["SA_2_v3"], "old_area_km2": [old_area_km2],
                            "new_area_km2": [new_area_km2], "added_km2": [added_km2]},
                           geometry=[sa2_v3], crs="EPSG:4326")
    out_gpkg = OUT_DIR / "SA_2_v3.gpkg"
    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"\n-> {out_gpkg}")

    report = f"""# Upstream boundary revision -- SA_2_v3

Operator directive (2026-09-11): SA_2 clips the upstream Zaporizhzhia/DniproHES
control section. Revised domain built from concrete multi-date water evidence,
not an arbitrary rectangle.

## Method
1. Start from `reservoir_full_pool_prebreach` (SA_2), area {old_area_km2:,.1f} km2.
2. Union in every pixel classified `SYSTEMATIC_SA2_OMISSION` by the 18-date
   consensus (`p1c_v2_full_consensus.py`) -- water Sentinel confirms across
   independent dates that SA_2 currently excludes (concentrated at the NE tip,
   per `P1_sa2_disagreement_map.png`).
3. Union in any `CONSISTENTLY_WATER` pixel not already inside SA_2.
4. **Explicit gauge guarantee**: connect a {CORRIDOR_BUFFER_M:.0f} m-half-width
   corridor from the resulting NE-most point through gauge 80957
   (35.08208E, 47.87204N) and {CORRIDOR_EXTEND_M/1000:.1f} km further upstream
   along the same bearing, approximating the DniproHES tailrace vicinity.
   **Caveat, stated not hidden**: no independently verified DniproHES dam-axis
   coordinate exists in this project; the corridor direction is derived from
   the SA_2-to-gauge bearing, not a surveyed DniproHES structure location.
5. Clipped to `study_domain_full` + 5 km so the construction cannot escape the
   vetted discovery domain.

## Result
| quantity | value |
|---|---:|
| old area (SA_2) | {old_area_km2:,.1f} km2 |
| new area (SA_2_v3) | {new_area_km2:,.1f} km2 |
| added area | {added_km2:,.1f} km2 |
| gauge 80957 inside SA_2_v3 | {gauge_inside_final} |
| distance, gauge to boundary | {dist_final:.0f} m |
| upstream-most bound (lon, lat) | ({upstream_most[0]:.4f}, {upstream_most[1]:.4f}) |
| SA_2_v3 bounds | lon [{minx:.4f},{maxx:.4f}], lat [{miny:.4f},{maxy:.4f}] |
| SA_2 original lat_max (for comparison) | {sa2.bounds[3]:.4f} |

DniproHES tailrace coverage: **approximate** -- covered to the extent the
corridor buffer ({CORRIDOR_BUFFER_M:.0f} m half-width) captures the river
channel at and beyond the gauge; not independently verified against a
surveyed DniproHES structure footprint (none exists in this project).
**ACTION before treating this as final**: verify against a real DniproHES
tailrace extent (e.g. OSM/manual inspection) before this feeds any published
figure.

Output: `data/processed/study_domain/SA_2_v3.gpkg`
"""
    (ANALYSIS_DIR / "upstream_boundary_revision.md").write_text(report)
    print(f"-> {ANALYSIS_DIR / 'upstream_boundary_revision.md'}")


if __name__ == "__main__":
    main()
