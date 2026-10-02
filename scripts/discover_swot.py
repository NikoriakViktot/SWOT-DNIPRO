#!/usr/bin/env python
"""Phase 2 discovery — what SWOT genuinely intersects the study AOIs, and how big is it.

Metadata only: no granule is downloaded. Footprints are tested against real AOI
polygons, and tile-based products (PIXC, Raster) are trusted for spatial filtering
while the continent-scale SP products (RiverSP/LakeSP) are counted separately
because their CMR footprints cover all of Europe and cannot be filtered spatially.

Writes data/catalog/granules_discovered.parquet.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, box

from swot_dnipro import config as CFG

BASE = "https://cmr.earthdata.nasa.gov/search/granules.json"
CATALOG = CFG.ROOT / "data" / "catalog"

#: typical granule sizes measured from what we already hold / HEAD requests
TYPICAL_MB = {"SWOT_L2_HR_PIXC_2.0": 385.0, "SWOT_L2_HR_Raster_2.0": 25.0,
              "SWOT_L2_HR_RiverSP_2.0": 30.0, "SWOT_L2_HR_LakeSP_2.0": 25.0,
              "SWOT_L2_HR_PIXCVec_2.0": 45.0}

AOIS = {
    "kakhovka_reservoir": box(33.3524, 46.7556, 35.3439, 47.7774),
    "lower_dnipro_kherson": box(31.9, 46.35, 33.45, 47.05),
}
REGIMES = [("PRE_BREACH", "2023-03-25", "2023-06-05"),
           ("BREACH_DRAWDOWN", "2023-06-06", "2023-08-31"),
           ("POST_BREACH", "2023-09-01", "2026-09-07")]


def polys(g):
    out = []
    for p in (g.get("polygons") or []):
        f = p[0] if isinstance(p, list) else p
        v = list(map(float, f.split()))
        out.append(Polygon([(v[i + 1], v[i]) for i in range(0, len(v) - 1, 2)]))
    for bx in (g.get("boxes") or []):
        s, w, n, e = map(float, bx.split())
        out.append(box(w, s, e, n))
    return out


def query(short_name, bbox, start, end):
    out, page, sess = [], 1, requests.Session()
    while True:
        r = sess.get(BASE, params={"short_name": short_name, "bounding_box": bbox,
                                   "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
                                   "page_size": 2000, "page_num": page}, timeout=120)
        r.raise_for_status()
        e = r.json()["feed"].get("entry") or []
        out += e
        if len(e) < 2000:
            break
        page += 1
        if page > 15:
            break
    return out


def main() -> None:
    CATALOG.mkdir(parents=True, exist_ok=True)
    rows = []
    for aoi_name, aoi in AOIS.items():
        w, s, e, n = aoi.bounds
        bbox = f"{w},{s},{e},{n}"
        for sn in ["SWOT_L2_HR_PIXC_2.0", "SWOT_L2_HR_Raster_2.0",
                   "SWOT_L2_HR_RiverSP_2.0", "SWOT_L2_HR_LakeSP_2.0"]:
            tiled = sn in ("SWOT_L2_HR_PIXC_2.0", "SWOT_L2_HR_Raster_2.0")
            for regime, a, b in REGIMES:
                try:
                    ents = query(sn, bbox, a, b)
                except Exception as exc:
                    print(f"  !! {sn} {aoi_name} {regime}: {exc}")
                    continue
                for g in ents:
                    P = polys(g)
                    # tiled products: require a real footprint intersection
                    hit = (any(p.intersects(aoi) for p in P) if (tiled and P) else True)
                    if not hit:
                        continue
                    gid = g.get("producer_granule_id", "")
                    parts = gid.split("_")
                    rows.append({
                        "aoi": aoi_name, "short_name": sn, "regime": regime,
                        "granule_id": gid, "time_start": g.get("time_start"),
                        "time_end": g.get("time_end"),
                        "cycle": parts[4] if len(parts) > 4 else "",
                        "pass": parts[5] if len(parts) > 5 else "",
                        "tile": parts[6] if len(parts) > 6 else "",
                        "spatially_filtered": tiled,
                        "typical_mb": TYPICAL_MB.get(sn, 30.0),
                    })
    df = pd.DataFrame(rows).drop_duplicates(["short_name", "granule_id"])
    df["date"] = pd.to_datetime(df.time_start).dt.date
    df.to_parquet(CATALOG / "granules_discovered.parquet", index=False)

    held = {p.name for p in CFG.PIXC_DIR.glob("*.nc")}
    df["already_held"] = df.granule_id.isin(held)

    print(f"discovered {len(df):,} unique granules over the two AOIs\n")
    piv = (df.groupby(["short_name", "regime"])
             .agg(n=("granule_id", "size"),
                  GB=("typical_mb", lambda x: round(x.sum() / 1024, 1)),
                  dates=("date", "nunique"))
             .reset_index())
    print(piv.to_string(index=False))
    print("\n=== totals by product ===")
    tot = (df.groupby("short_name")
             .agg(n=("granule_id", "size"), est_GB=("typical_mb", lambda x: round(x.sum() / 1024, 1)),
                  first=("date", "min"), last=("date", "max")))
    print(tot.to_string())
    print(f"\nestimated FULL acquisition: {df.typical_mb.sum()/1024:.0f} GB")
    print(f"already held: {int(df.already_held.sum())} granules")
    print("\nNOTE: RiverSP/LakeSP counts are NOT spatially filtered — their CMR footprints "
          "are continent-scale (all of Europe), so every European granule matches the bbox. "
          "Their true AOI relevance can only be established after reading each file.")
    tiled = df[df.spatially_filtered]
    print(f"\nSpatially-verified (PIXC + Raster) only: {len(tiled)} granules, "
          f"{tiled.typical_mb.sum()/1024:.0f} GB")
    print(tiled.groupby(["short_name", "aoi", "regime"]).size().to_string())


if __name__ == "__main__":
    main()
