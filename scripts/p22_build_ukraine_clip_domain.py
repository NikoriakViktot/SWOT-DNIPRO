#!/usr/bin/env python
"""Build the Ukraine archive-clip domain for the continent-wide SWOT products.

Why this exists
---------------
SWOT RiverSP and LakeSP are distributed per continent-wide overpass ("_EU_" in
the filename): one granule covers every reach or lake in Europe for that pass.
The archive pulled for this project is 402 GB of LakeSP plus 81 GB of RiverSP,
and almost all of it is outside Ukraine -- the first Unassigned granule
inspected spans 54.95-71.38 N, i.e. Scandinavia and the Arctic, and contributes
nothing here.

This domain is the filter used to cut that archive down. It is a STORAGE
container, in the same sense as ``study_domain_full``: it must never be used to
clip a scientific product, because a national boundary has no hydraulic
meaning.

Why not a country polygon
-------------------------
Natural Earth 10m admin-0 was tried first and is the WRONG instrument, in two
independent ways, both verified before this file was written:

1. It is a LAND polygon. ``contains()`` returns False for Kherson (32.61,
   46.63) and for Ochakiv on the liman (31.55, 46.61), because both sit on
   water. SWOT LakeSP/RiverSP observe water surfaces, so clipping them to a
   land boundary would delete the river, the liman and the reservoir -- the
   entire subject of this project.
2. It assigns Crimea to ``ADMIN='Russia'``, a political determination this
   project has no reason to encode in its data-retention rule, and which would
   silently drop Crimean and Azov observations.

Composition
-----------
Union of authoritative sources only -- no hand-drawn geometry anywhere:

  * ``ukraine_basins.geojson`` (companion repo): the 16 official Ukrainian
    river basin districts (райони басейнів річок). Hydrographic rather than
    political, so it covers river mouths and includes Crimea on drainage
    grounds rather than excluding it on political ones.
  * all four registered analysis zones, loaded through the registry. The
    basins alone cover only 71.1 % of ZONE_3_DNIPRO_BUG_ESTUARY, because the
    basin districts stop at the coastline while ZONE_3 extends into the
    Dnipro-Bug liman; adding the zones repairs exactly that.
  * a 20 km margin, buffered in EPSG:32636 per the repo CRS policy, never in
    degrees.

Validation is by SHAPE, not by area: every registered zone must come out at
100 % covered, and named control points at the country's extremes must fall
inside. Both are asserted below, so a future edit that quietly shrinks the
domain fails here rather than in a deletion.

Outputs
-------
data/processed/domains/ukraine_swot_clip_domain.geojson   (EPSG:4326)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

BASINS = CFG.ICESAT_ROOT / "data" / "1_data" / "data" / "ukraine_basins.geojson"
OUT = ROOT / "data" / "processed" / "domains" / "ukraine_swot_clip_domain.geojson"
MARGIN_M = 20_000
ZONES = ["ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY"]

#: Control points that must survive the clip. Kherson and Ochakiv are the two
#: that Natural Earth's land polygon fails; the rest pin the country's corners.
CONTROLS = {
    "Kherson": (32.61, 46.63),
    "Ochakiv (liman)": (31.55, 46.61),
    "Kinburn spit": (31.50, 46.50),
    "Nova Kakhovka dam": (33.374, 46.775),
    "Simferopol": (34.10, 44.95),
    "Uzhhorod (W)": (22.30, 48.60),
    "Luhansk (E)": (39.30, 48.57),
}


def main() -> None:
    if not BASINS.exists():
        raise SystemExit(f"basin districts not found: {BASINS}")

    b = gpd.read_file(BASINS).to_crs(CFG.CRS_METRIC)
    print(f"  basin districts : {len(b)} features from {BASINS.name}")

    parts = [b.geometry.union_all()]
    for z in ZONES:
        parts.append(SD.load_utm(z))
    u_utm = unary_union(parts).buffer(MARGIN_M)
    u = gpd.GeoSeries([u_utm], crs=CFG.CRS_METRIC).to_crs(CFG.CRS_GEOG).iloc[0]

    print(f"  margin          : {MARGIN_M/1000:.0f} km, buffered in {CFG.CRS_METRIC}")
    print(f"  area            : {u_utm.area/1e6:,.0f} km2")
    print(f"  bounds          : {[round(v, 3) for v in u.bounds]}")

    print("\n  coverage of every registered zone (by shape, not area):")
    bad = []
    for z in ZONES:
        g = SD.load(z)
        pct = g.intersection(u).area / g.area * 100
        print(f"    {z:<32} {pct:6.2f} %")
        if pct < 99.99:
            bad.append((z, pct))

    print("\n  control points:")
    for nm, (x, y) in CONTROLS.items():
        inside = u.contains(Point(x, y))
        print(f"    {nm:<20} {'OK' if inside else 'OUTSIDE'}")
        if not inside:
            bad.append((nm, 0.0))

    if bad:
        raise SystemExit(
            "GATE FAILED — the clip domain does not cover: "
            + ", ".join(f"{n} ({p:.2f}%)" for n, p in bad)
            + ". Refusing to write a domain that would delete project data.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"name": ["ukraine_swot_clip_domain"]},
                     geometry=[u], crs=CFG.CRS_GEOG).to_file(OUT, driver="GeoJSON")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
