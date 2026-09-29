#!/usr/bin/env python
"""F3 -- discover post-breach Sentinel-2 scenes covering the upper
DniproHES transition AOI (needs tile T36UXU, which has never been
downloaded for any post-breach date -- the existing 77-scene post-breach
cache only has T36TWS/TWT/TXT).

Metadata only, no download here.

Outputs
-------
data/catalog/postbreach_upper_discovery.csv
outputs/tables/f3_upper_candidate_dates.csv
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

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
START, END = "2023-06-06", "2025-12-31"
COLLECTION = "sentinel-2-l2a"
REQUIRED_TILES = {"36TXT", "36UXU"}  # corrected after discovery: the upper AOI
                                     # (34.95-35.22E, 47.75-47.96N) does not
                                     # overlap T36TWS/T36TWT at all -- verified
                                     # empirically (0 hits for those tiles)

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def stac_search(bbox, start, end, limit=250):
    body = {"collections": [COLLECTION], "bbox": list(bbox), "limit": limit,
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"}
    items, url, payload = [], STAC, body
    while url:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode() if payload else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload else "GET")
        with urllib.request.urlopen(req, timeout=120) as resp:
            j = json.loads(resp.read())
        items.extend(j.get("features", []))
        nxt = next((L for L in j.get("links", []) if L.get("rel") == "next"), None)
        if nxt is None:
            break
        url, payload = nxt["href"], nxt.get("body")
    return items


def find_local_scenes():
    names = set()
    for base in (CFG.ROOT / "data" / "raw" / "sentinel_p1_targeted",
                CFG.ROOT / "data" / "raw" / "sentinel", Path("/mnt/e/data_swot/sentinel")):
        if base.exists():
            for p in base.glob("*.SAFE.zip"):
                names.add(p.name.replace(".SAFE.zip", ""))
    return names


def main() -> None:
    aoi_m = None
    import geopandas as gpd
    aoi = gpd.read_file(CFG.ROOT / "data/processed/study_domain/upper_dniprohes_transition_aoi.gpkg")
    aoi_ll = aoi.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = aoi_ll.total_bounds
    print(f"upper AOI bbox: {minx:.4f},{miny:.4f},{maxx:.4f},{maxy:.4f}")

    print(f"querying {COLLECTION}, {START}..{END} ...")
    items = stac_search((minx, miny, maxx, maxy), START, END)
    print(f"total features: {len(items)}")

    local = find_local_scenes()
    rows = []
    for it in items:
        p = it["properties"]
        tile = p.get("s2:mgrs_tile", "")
        if tile not in REQUIRED_TILES:
            continue
        date = p["datetime"][:10]
        cloud = p.get("eo:cloud_cover")
        pid = it["id"]
        cached = any(pid.startswith(n) or n.startswith(pid) for n in local)
        rows.append({"date": date, "tile_id": tile, "product_id": pid,
                    "cloud_cover_metadata": cloud, "cached_locally": cached})

    df = pd.DataFrame(rows).sort_values(["date", "tile_id"])
    out_cat = CFG.ROOT / "data" / "catalog" / "postbreach_upper_discovery.csv"
    df.to_csv(out_cat, index=False)
    print(f"-> {out_cat} ({len(df)} rows)")

    # candidate dates: all 4 required tiles present that day
    piv = df.groupby("date").tile_id.apply(lambda s: set(s))
    full = piv[piv.apply(lambda s: REQUIRED_TILES.issubset(s))]
    print(f"\ndates with all 4 required tiles in catalogue: {len(full)}")

    cand = df[df.date.isin(full.index)].copy()
    cloud_by_date = cand.groupby("date").cloud_cover_metadata.mean().rename("mean_cloud_pct")
    cached_by_date = cand.groupby("date").cached_locally.all().rename("all_cached")
    summary = pd.concat([cloud_by_date, cached_by_date], axis=1).reset_index()
    summary["date_dt"] = pd.to_datetime(summary.date)
    summary["month"] = summary.date_dt.dt.month
    summary["season"] = summary.month.map(lambda m: {12:"winter",1:"winter",2:"winter",
                                                      3:"spring",4:"spring",5:"spring",
                                                      6:"summer",7:"summer",8:"summer",
                                                      9:"autumn",10:"autumn",11:"autumn"}[m])
    summary = summary.sort_values("mean_cloud_pct")
    out_sum = CFG.TABLES / "f3_upper_candidate_dates.csv"
    summary.to_csv(out_sum, index=False)
    print(f"-> {out_sum}")
    print(f"\nlow-cloud (<30%), one per season where possible, top rows:")
    print(summary[summary.mean_cloud_pct < 30].groupby("season").head(6)
         [["date", "mean_cloud_pct", "season", "all_cached"]].to_string(index=False))

    already_full = summary.all_cached.sum()
    print(f"\ndates already fully cached (incl. T36UXU): {already_full}")
    print(f"dates needing at least T36UXU download: {(~summary.all_cached).sum()}")


if __name__ == "__main__":
    main()
