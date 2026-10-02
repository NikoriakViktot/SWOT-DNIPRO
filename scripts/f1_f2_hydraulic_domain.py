#!/usr/bin/env python
"""F1 + F2 -- formal hydraulic_longitudinal_domain.gpkg with explicit
non-overlapping zone classes, and a DniproHES control zone built from the
real surveyed axis (not the earlier gauge-buffer approximation).

Outputs
-------
data/processed/study_domain/hydraulic_longitudinal_domain.gpkg
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
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union, split, transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STUDY_DIR = CFG.ROOT / "data" / "processed" / "study_domain"
CONTROL_ZONE_BUFFER_M = 400.0  # analysis convenience, NOT a physical dam-width
                               # statement -- documented per F2's instruction

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def main() -> None:
    SD.assert_projected_32636  # exists; not force-wired into every legacy call yet
    sa2_m = shp_transform(_TF_4326_TO_M, SD.load("reservoir_full_pool_prebreach"))
    dam = gpd.read_file(STUDY_DIR / "dniprohes_control_axis.gpkg").geometry.iloc[0]
    assert gpd.read_file(STUDY_DIR / "dniprohes_control_axis.gpkg").crs.to_epsg() == 32636

    with rasterio.open(STUDY_DIR / "upper_prebreach_water_frequency.tif") as ds:
        freq = ds.read(1); tr = ds.transform
        assert ds.crs.to_epsg() == 32636, "F2/F1 CRS contract: raster must be EPSG:32636"

    persistent = freq >= 0.8
    polys = [shape(g) for g, v in features.shapes(persistent.astype("uint8"),
             mask=persistent, transform=tr) if v == 1]
    persistent_poly = unary_union(polys)

    # F2: control zone = buffer around the real axis, documented width
    control_zone = dam.buffer(CONTROL_ZONE_BUFFER_M)

    # split persistent water by the (extended) real axis, as in the previous pass
    coords = list(dam.coords)
    p0, p1 = np.array(coords[0]), np.array(coords[-1])
    d = p1 - p0; d = d / np.linalg.norm(d); far = 50000
    ext_line = LineString([tuple(p0 - d * far), tuple(p1 + d * far)])
    parts = list(split(persistent_poly, ext_line).geoms)

    def side(pt):
        a, b = np.array(dam.coords[0]), np.array(dam.coords[-1])
        ap = np.array([pt.x, pt.y]) - a; ab = b - a
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        return "kakhovka" if cross < 0 else "dniprovske"  # verified against SA_2 centroid previously

    kakhovka_pieces, dniprovske_pieces = [], []
    for p in parts:
        if p.distance(sa2_m) > 5000:
            continue  # far-field split artefact, excluded as before
        if sa2_m.contains(p.centroid):
            continue  # already inside SA_2 -- not part of the candidate zone
        (kakhovka_pieces if side(p.centroid) == "kakhovka" else dniprovske_pieces).append(p)

    # enforce strict non-overlap, priority: reservoir core > control zone >
    # backwater candidate > dniprovske upstream (F1: classes must not overlap)
    control_zone_excl = control_zone.difference(sa2_m)
    upper_backwater_candidate = (unary_union(kakhovka_pieces)
                                 .difference(control_zone_excl).difference(sa2_m))
    dniprovske_upstream = (unary_union(dniprovske_pieces)
                           .difference(control_zone_excl).difference(sa2_m)
                           .difference(upper_backwater_candidate))

    rows = [
        {"zone_id": 4, "zone_name": "KAKHOVKA_RESERVOIR_CORE", "geometry": sa2_m,
         "source": "Kakhovka_SA_2.geojson (unchanged)", "confidence": "authoritative"},
        {"zone_id": 3, "zone_name": "KAKHOVKA_UPPER_BACKWATER_CANDIDATE",
         "geometry": upper_backwater_candidate,
         "source": "18-date pre-breach persistence (>=0.8) minus SA_2 minus control zone, "
                   "Kakhovka side of real DniproHES axis",
         "confidence": "candidate, NOT promoted to reservoir core"},
        {"zone_id": 2, "zone_name": "DNIPROHES_CONTROL_ZONE", "geometry": control_zone_excl,
         "source": f"damb_DniproGES.shp axis, {CONTROL_ZONE_BUFFER_M:.0f}m buffer "
                   f"(analysis convenience, not a physical dam-width statement)",
         "confidence": "high (real surveyed axis)"},
        {"zone_id": 1, "zone_name": "DNIPROVSKE_RESERVOIR_UPSTREAM",
         "geometry": dniprovske_upstream,
         "source": "18-date pre-breach persistence (>=0.8), Dniprovske side of real axis",
         "confidence": "out of scope -- different reservoir system, informational only"},
    ]
    gdf = gpd.GeoDataFrame(rows, crs=CFG.CRS_METRIC)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf["crs"] = "EPSG:32636"
    gdf["notes"] = ""
    gdf.loc[gdf.zone_name == "KAKHOVKA_UPPER_BACKWATER_CANDIDATE", "notes"] = \
        "supersedes the earlier 16.2 km2 gauge-buffer estimate; to be split further in F6"

    # non-overlap check
    for i in range(len(gdf)):
        for j in range(i + 1, len(gdf)):
            ov = gdf.geometry.iloc[i].intersection(gdf.geometry.iloc[j]).area / 1e6
            if ov > 0.01:
                print(f"  WARNING: {gdf.zone_name.iloc[i]} / {gdf.zone_name.iloc[j]} "
                      f"overlap {ov:.3f} km2")

    out = STUDY_DIR / "hydraulic_longitudinal_domain.gpkg"
    gdf.to_file(out, driver="GPKG")
    print(f"-> {out}")
    print(gdf[["zone_id", "zone_name", "area_km2", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
