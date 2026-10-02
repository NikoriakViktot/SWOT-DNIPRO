#!/usr/bin/env python
"""P1A -- complete pre-breach Sentinel-2 L2A catalogue discovery (2026-09-11
operator correction). Metadata only, nothing downloaded.

Supersedes treating the local 17-file cache (2023-05-16 onward) as the
complete pre-breach record. Queries against ``study_domain_full``
(spatial_domains.py), NOT any of the three retired hard-coded bboxes
(RES_BBOX_4326 / RESERVOIR_WKT / discover_swot.py's box), for the full
2017-01-01 .. 2023-06-05 window. Uses the Planetary Computer STAC API
(open, no credentials -- same endpoint already proven reliable in
hist22_low_water_search.py).

Outputs
-------
data/catalog/prebreach_sentinel_discovery_full.csv
"""
from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
START, END = "2017-01-01", "2023-06-05"
COLLECTION = "sentinel-2-l2a"
CATALOG = CFG.ROOT / "data" / "catalog"
CATALOG.mkdir(parents=True, exist_ok=True)
LOCAL_S2 = CFG.ROOT / "data" / "raw" / "sentinel"  # symlink target checked separately

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def stac_search(bbox, start, end, limit=250):
    """Paginate a POST STAC search, following the 'next' link."""
    body = {"collections": [COLLECTION], "bbox": list(bbox), "limit": limit,
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"}
    items, url, payload = [], STAC, body
    n_pages = 0
    while url:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode() if payload else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload else "GET")
        with urllib.request.urlopen(req, timeout=120) as resp:
            j = json.loads(resp.read())
        feats = j.get("features", [])
        items.extend(feats)
        n_pages += 1
        nxt = next((L for L in j.get("links", []) if L.get("rel") == "next"), None)
        if nxt is None or not feats:
            break
        url = nxt["href"]
        # a 'next' link from PC STAC is a full GET url with a token; switch mode
        payload = nxt.get("body")
        if payload is None:
            # GET-style pagination
            req_method = nxt.get("method", "GET")
            if req_method == "GET":
                payload = None
        print(f"    page {n_pages}: {len(feats)} features (running total {len(items)})")
        if n_pages > 60:
            print("    !! stopping at 60 pages as a safety cap")
            break
    return items


def find_local_scenes() -> set[str]:
    """Product names of everything already cached, from both known locations."""
    names = set()
    for base in (CFG.ROOT / "data" / "raw" / "sentinel",
                 Path("/mnt/e/data_swot/sentinel")):
        if base.exists():
            for p in base.glob("*.SAFE.zip"):
                names.add(p.name.replace(".SAFE.zip", ""))
    return names


def main() -> None:
    reservoir = SD.load("reservoir_full_pool_prebreach")
    reservoir_m = shp_transform(_TF_4326_TO_M, reservoir)
    reservoir_area_km2 = reservoir_m.area / 1e6
    bbox = SD.as_bbox("study_domain_full")
    print(f"study_domain_full bbox: {bbox}")
    print(f"reservoir_full_pool_prebreach area: {reservoir_area_km2:,.1f} km2")

    print(f"\nquerying Planetary Computer STAC, {COLLECTION}, {START}..{END} ...")
    items = stac_search(bbox, START, END)
    print(f"\ntotal features returned: {len(items)}")
    if not items:
        raise SystemExit("STAC search returned zero items -- check network/bbox")

    local = find_local_scenes()
    print(f"local cached SAFE products found: {len(local)}")

    rows = []
    for it in items:
        p = it["properties"]
        date = p["datetime"][:10]
        tile = p.get("s2:mgrs_tile", "")
        cloud = p.get("eo:cloud_cover")
        proc_level = p.get("s2:processing_baseline", "")
        try:
            g = shape(it["geometry"])
            g_m = shp_transform(_TF_4326_TO_M, g)
            inter_m = g_m.intersection(reservoir_m)
            inter_km2 = inter_m.area / 1e6
            geom_wkt = g.wkt  # actual footprint polygon, not just its bbox
        except Exception:
            inter_km2 = float("nan")
            geom_wkt = ""
        frac = inter_km2 / reservoir_area_km2 if reservoir_area_km2 else float("nan")
        pid = it["id"]
        cached = any(pid.startswith(n) or n.startswith(pid) for n in local) or (
            any(f"_T{tile}_" in n and n.split("_")[2][:8] == date.replace("-", "")
                for n in local) if tile else False)
        rows.append({
            "date": date, "datetime": p["datetime"], "product_id": pid,
            "tile_id": tile, "processing_level": "L2A",
            "processing_baseline": proc_level,
            "cloud_cover_metadata": cloud,
            "native_bounds": json.dumps(it["bbox"]),
            "native_geometry_wkt": geom_wkt,
            "intersection_area_km2": round(inter_km2, 2),
            "reservoir_coverage_fraction": round(frac, 4),
            "cached_locally": bool(cached),
            "local_path": "", "download_required": not bool(cached),
            "provider": "planetarycomputer/sentinel-2-l2a",
        })

    df = pd.DataFrame(rows).sort_values(["date", "tile_id"]).reset_index(drop=True)
    out = CATALOG / "prebreach_sentinel_discovery_full.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(df)} rows)")
    print(f"date range found: {df.date.min()} .. {df.date.max()}")
    print(f"unique dates: {df.date.nunique()}, unique tiles: {df.tile_id.nunique()}")
    print("tiles seen:", sorted(df.tile_id.unique()))
    print(f"cached_locally=True: {int(df.cached_locally.sum())} / {len(df)}")
    print("\nEarliest 5 rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
