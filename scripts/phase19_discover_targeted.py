#!/usr/bin/env python
"""Phase 19 STEP 2 — targeted Sentinel discovery for the HIGH/MEDIUM feasibility dates.

Metadata only. Reads outputs/tables/postbreach_profile_feasibility.csv, takes every
HIGH and MEDIUM date, and searches CDSE for Sentinel-2 L2A (all three tiles that
carry ATL13: T36TWS/T36TWT/T36TXT) within +/-DAYS, plus Sentinel-1 GRD as a
cloud fallback.  Picks the best (lowest-cloud, then closest-in-time) S2 scene per
(date, tile) and writes an augmented selection that fetch_sentinel.py can consume.

Outputs
-------
data/catalog/sentinel_targeted_discovered.csv     every candidate (S2 + S1)
data/catalog/sentinel2_selected_targeted.csv      best S2 per (date, tile), NOT already cached
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from swot_dnipro import config as CFG

CAT = CFG.ROOT / "data" / "catalog"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
RESERVOIR_WKT = "POLYGON((33.35 46.75,35.35 46.75,35.35 47.79,33.35 47.79,33.35 46.75))"
TILES = ("T36TWS", "T36TWT", "T36TXT")
DAYS = 12
MAX_CLOUD_S2 = 70.0


def token() -> str:
    c = json.loads((pathlib.Path.home() / ".config/cdse/credentials.json").read_text())
    r = requests.post(TOKEN_URL, timeout=60, data={
        "client_id": "cdse-public", "username": c["username"],
        "password": c["password"], "grant_type": "password"})
    r.raise_for_status()
    return r.json()["access_token"]


def search(collection, start, end, tok, cloud=None):
    f = (f"Collection/Name eq '{collection}' "
         f"and OData.CSC.Intersects(area=geography'SRID=4326;{RESERVOIR_WKT}') "
         f"and ContentDate/Start gt {start}T00:00:00.000Z "
         f"and ContentDate/Start lt {end}T23:59:59.999Z")
    if cloud is not None:
        f += (" and Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' "
              f"and a/OData.CSC.DoubleAttribute/Value lt {cloud})")
    out, url = [], None
    params = {"$filter": f, "$top": 200, "$orderby": "ContentDate/Start asc",
              "$expand": "Attributes"}
    while True:
        r = requests.get(url or ODATA, params=None if url else params,
                         headers={"Authorization": f"Bearer {tok}"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        out += j.get("value", [])
        url = j.get("@odata.nextLink")
        if not url or len(out) > 2000:
            break
    return out


def main() -> None:
    fz = pd.read_csv(CFG.TABLES / "postbreach_profile_feasibility.csv", parse_dates=["date"])
    targets = fz[fz.priority.isin(["HIGH", "MEDIUM"])].sort_values("date")
    print(f"targets: {len(targets)} HIGH/MEDIUM dates "
          f"({targets.date.min().date()}..{targets.date.max().date()})")

    try:
        cached = set(pd.read_csv(CFG.TABLES / "sentinel_scene_inventory.csv").name)
    except FileNotFoundError:
        cached = set()

    tok = token()
    print("CDSE auth OK\n")

    rows = []
    for t in targets.itertuples():
        a = (t.date - pd.Timedelta(days=DAYS)).strftime("%Y-%m-%d")
        b = (t.date + pd.Timedelta(days=DAYS)).strftime("%Y-%m-%d")
        for coll, cloud in (("SENTINEL-2", MAX_CLOUD_S2), ("SENTINEL-1", None)):
            try:
                res = search(coll, a, b, tok, cloud)
            except Exception as exc:
                print(f"  !! {coll} {t.date.date()}: {exc}")
                continue
            for p in res:
                name = p.get("Name", "")
                if coll == "SENTINEL-2" and "MSIL2A" not in name:
                    continue
                if coll == "SENTINEL-1" and "_GRDH_" not in name and "_GRD_" not in name:
                    continue
                tile = name.split("_")[5] if coll == "SENTINEL-2" and len(name.split("_")) > 5 else ""
                if coll == "SENTINEL-2" and tile not in TILES:
                    continue
                cc = next((a_["Value"] for a_ in p.get("Attributes", [])
                           if a_.get("Name") == "cloudCover"), None)
                st = p.get("ContentDate", {}).get("Start")
                rows.append({
                    "collection": coll, "target_date": t.date.date(), "priority": t.priority,
                    "regime": t.period, "tile": tile, "name": name, "product_id": p.get("Id"),
                    "sensing_start": st, "size_bytes": int(p.get("ContentLength") or 0),
                    "cloud_cover": cc, "online": p.get("Online"),
                    "days_from_target": abs((pd.to_datetime(st).tz_localize(None) - t.date).days),
                    "already_cached": name in cached,
                })

    df = pd.DataFrame(rows).drop_duplicates(["collection", "name", "target_date"])
    CAT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAT / "sentinel_targeted_discovered.csv", index=False)

    s2 = df[df.collection == "SENTINEL-2"].copy()
    s1 = df[df.collection == "SENTINEL-1"].copy()
    print("=== Sentinel-2 candidates per target date / tile (lowest cloud first) ===")
    for d, g in s2.groupby("target_date"):
        have = sorted(g[g.already_cached].tile.unique())
        cov = {tl: g[g.tile == tl].cloud_cover.min() for tl in TILES if (g.tile == tl).any()}
        print(f"  {d}  cached tiles={have or '-'}  best cloud by tile: "
              + "  ".join(f"{k}={v:.0f}%" for k, v in cov.items()))

    # best NEW S2 scene per (date, tile): lowest cloud, then closest in time
    new = s2[~s2.already_cached].copy()
    best = (new.sort_values(["target_date", "tile", "cloud_cover", "days_from_target"])
              .groupby(["target_date", "tile"]).first().reset_index())
    keep = ["target_date", "tile", "regime", "name", "product_id", "sensing_start",
            "size_bytes", "cloud_cover", "days_from_target"]
    best[keep].to_csv(CAT / "sentinel2_selected_targeted.csv", index=False)

    print(f"\nNEW S2 scenes to fetch: {len(best)}  "
          f"({best.size_bytes.sum()/1e9:.1f} GB)  -> data/catalog/sentinel2_selected_targeted.csv")
    print(best[["target_date", "tile", "cloud_cover", "days_from_target",
                "size_bytes"]].assign(GB=lambda d: (d.size_bytes/1e9).round(2))
          .drop(columns="size_bytes").to_string(index=False))

    print(f"\nSentinel-1 GRD available for {s1.target_date.nunique()} of {len(targets)} "
          f"target dates (fallback where S2 cloud is high):")
    for d, g in s1.groupby("target_date"):
        near = g[g.days_from_target <= 4]
        print(f"  {d}: {len(g)} scenes, {len(near)} within 4 d")


if __name__ == "__main__":
    main()
