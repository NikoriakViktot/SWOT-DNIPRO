#!/usr/bin/env python
"""F7 + F8 -- local metric chainage, s=0 at the real DniproHES control axis,
positive downstream (toward Kakhovka). Valid as a LOCAL LINEAR approximation
near the control section (the upper AOI, ~10 km radius) -- not a full
high-fidelity centerline for the whole 250 km reservoir. Beyond the upper
AOI, SWORD chain_km (existing project convention) remains the reference,
carried as a separate field, never overwritten.

A SWORD chainage artefact was found at the exact control-section location
(lon 35.0877, lat 47.8702 -> chain_km=263.5, a clear non-monotonic spike
against neighbours reading ~247.5/247.7) -- this is the concrete reason a
local chainage was needed here rather than trusting SWORD at this specific
point.

Outputs
-------
outputs/tables/upper_chainage_controls.csv
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
import pandas as pd
import pyproj
from shapely.geometry import Point
from shapely.ops import nearest_points, transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STUDY_DIR = CFG.ROOT / "data" / "processed" / "study_domain"
GAUGE_80957 = (35.08208, 47.87204)

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def main() -> None:
    dam = gpd.read_file(STUDY_DIR / "dniprohes_control_axis.gpkg").geometry.iloc[0]
    sa2_m = shp_transform(_TF_4326_TO_M, SD.load("reservoir_full_pool_prebreach"))
    gauge_m = Point(*_TF_4326_TO_M(*GAUGE_80957))
    hyd = gpd.read_file(STUDY_DIR / "hydraulic_longitudinal_domain.gpkg")

    a = np.array(dam.coords[0]); b = np.array(dam.coords[-1])
    ab = b - a
    flow_dir = np.array([ab[1], -ab[0]])
    flow_dir = flow_dir / np.linalg.norm(flow_dir)
    mid = np.array(dam.centroid.coords[0])

    def s_local_km(pt: Point) -> float:
        v = np.array([pt.x, pt.y]) - mid
        return float(np.dot(v, flow_dir) / 1000.0)

    # sign calibration: SA_2 (known Kakhovka/downstream) must be positive
    sign = 1 if s_local_km(sa2_m.centroid) > 0 else -1

    p_sa2_bound, _ = nearest_points(sa2_m.boundary, dam)
    backwater = hyd[hyd.zone_name == "KAKHOVKA_UPPER_BACKWATER_CANDIDATE"].geometry.iloc[0]
    dniprovske = hyd[hyd.zone_name == "DNIPROVSKE_RESERVOIR_UPSTREAM"].geometry.iloc[0]

    def extent_s(geom):
        if geom.is_empty:
            return (np.nan, np.nan)
        coords = []
        gg = geom.geoms if hasattr(geom, "geoms") else [geom]
        for g in gg:
            coords.extend(list(g.exterior.coords))
        svals = [sign * s_local_km(Point(c)) for c in coords]
        return (min(svals), max(svals))

    bw_min, bw_max = extent_s(backwater)
    dn_min, dn_max = extent_s(dniprovske)

    rows = [
        {"point": "DniproHES control axis", "s_local_km": 0.0,
         "definition": "origin, s=0 by construction", "sword_chain_km": None,
         "uncertainty_note": "axis is a real 1492m surveyed line; s=0 taken at its midpoint"},
        {"point": "gauge 80957", "s_local_km": round(sign * s_local_km(gauge_m), 3),
         "definition": "local linear projection onto flow direction perpendicular to dam axis",
         "sword_chain_km": 247.512,
         "uncertainty_note": "285m raw distance from axis; nearest reliable SWORD node "
                             "(the geometrically nearest node, chain_km=263.5, is a known "
                             "non-monotonic outlier and was excluded -- see note below)"},
        {"point": "SA_2 nearest boundary point", "s_local_km": round(sign * s_local_km(p_sa2_bound), 3),
         "definition": "nearest point on SA_2 boundary to the dam axis",
         "sword_chain_km": None,
         "uncertainty_note": f"distance to axis = {p_sa2_bound.distance(dam):.3f} m -- "
                             f"SA_2's own digitized boundary touches the control axis almost exactly"},
        {"point": "KAKHOVKA_UPPER_BACKWATER_CANDIDATE extent", "s_local_km": f"{bw_min:.2f} .. {bw_max:.2f}",
         "definition": "min/max s_local across the candidate zone's vertices",
         "sword_chain_km": None, "uncertainty_note": "gradual, not a sharp boundary -- "
                             "reported as a range per operator instruction (F8: do not force "
                             "a sharp transition where evidence indicates a gradual backwater)"},
        {"point": "DNIPROVSKE_RESERVOIR_UPSTREAM extent", "s_local_km": f"{dn_min:.2f} .. {dn_max:.2f}",
         "definition": "min/max s_local across the zone's vertices (out of scope, informational)",
         "sword_chain_km": None, "uncertainty_note": "different reservoir system"},
    ]
    out = pd.DataFrame(rows)
    out_path = CFG.TABLES / "upper_chainage_controls.csv"
    out.to_csv(out_path, index=False)
    print(f"-> {out_path}")
    print(out.to_string(index=False))

    print("\nSWORD glitch note (excluded from any chainage calculation):")
    print("  lon=35.0877, lat=47.8702, chain_km=263.515 -- non-monotonic spike, "
          "neighbouring nodes read chain_km~247.5/247.7. Kept in the raw SWORD "
          "parquet for transparency, never used as a control-point reference.")


if __name__ == "__main__":
    main()
