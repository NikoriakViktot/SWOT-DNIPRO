#!/usr/bin/env python
"""Corrected Sentinel-2 selection — by TILE COVERAGE of the ATL13 observations.

Why this replaces the first selection
-------------------------------------
The first pass searched scenes intersecting the reservoir *bounding box* and then
sorted by cloud. That returned tiles which only clip a corner of the AOI: measured
against the 112 497 post-breach ATL13 segments, tile T36TXS and T36UWU each contain
**zero** of them, yet they were selected because they were cloud-free.

Coverage measured from the tiles already downloaded:

    T36TXT  42.4 %   lon 34.31-35.80  lat 46.83-47.85   upper reservoir
    T36TWS  15.2 %   lon 33.00-34.44  lat 45.96-46.95   dam / lower end
    T36UXU   1.5 %
    T36TXS   0.0 %   <- selected before, useless
    T36UWU   0.0 %   <- selected before, useless

T36TWT (lon ~33.0-34.5, lat ~46.83-47.85) is the missing middle-reservoir tile.

So selection is now restricted to the tiles that actually contain the observations,
and cloud is used only to choose between scenes *within* those tiles.
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

CATALOG = CFG.ROOT / "data" / "catalog"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
#: tiles that actually contain post-breach ATL13 observations
TILES = ("T36TWT", "T36TXT", "T36TWS")
MAX_CLOUD = 40.0        # relaxed: coverage matters more than a pristine sky
WINDOW_DAYS = 7


def token() -> str:
    c = json.loads((pathlib.Path.home() / ".config/cdse/credentials.json").read_text())
    r = requests.post(TOKEN_URL, timeout=60, data={
        "client_id": "cdse-public", "username": c["username"],
        "password": c["password"], "grant_type": "password"})
    r.raise_for_status()
    return r.json()["access_token"]


#: polygon over the reservoir body where the ATL13 observations actually lie
CORE_WKT = "POLYGON((33.05 46.80,35.75 46.80,35.75 47.88,33.05 47.88,33.05 46.80))"


def search(start: str, end: str, tok: str) -> list[dict]:
    f = (f"Collection/Name eq 'SENTINEL-2' "
         f"and OData.CSC.Intersects(area=geography'SRID=4326;{CORE_WKT}') "
         f"and ContentDate/Start gt {start}T00:00:00.000Z "
         f"and ContentDate/Start lt {end}T23:59:59.999Z "
         f"and Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' "
         f"and a/OData.CSC.DoubleAttribute/Value lt {MAX_CLOUD})")
    r = requests.get(ODATA, params={"$filter": f, "$top": 400, "$expand": "Attributes",
                                    "$orderby": "ContentDate/Start asc"},
                     headers={"Authorization": f"Bearer {tok}"}, timeout=180)
    r.raise_for_status()
    return r.json().get("value", [])


def main() -> None:
    tok = token()
    sl = pd.read_csv(CFG.FIGDATA / "FigG_perdate_slopes_robust.csv", parse_dates=["date"])
    targets = sl[sl.period.isin(["BREACH_DRAWDOWN", "POST_BREACH"])][["date", "period"]]
    extra = pd.DataFrame({"date": pd.to_datetime(["2023-05-20", "2023-06-10"]),
                          "period": ["PRE_BREACH", "BREACH_DRAWDOWN"]})
    targets = pd.concat([targets, extra]).drop_duplicates("date").sort_values("date")

    rows = []
    for t in targets.itertuples():
        a = (t.date - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        b = (t.date + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        try:
            res = search(a, b, tok)
        except Exception as exc:
            print(f"  !! {t.date.date()}: {exc}")
            continue
        for p in res:
            n = p.get("Name", "")
            if "MSIL2A" not in n:
                continue
            parts = n.split("_")
            tile = parts[5] if len(parts) > 5 else ""
            if tile not in TILES:
                continue
            cc = next((x["Value"] for x in p.get("Attributes", [])
                       if x.get("Name") == "cloudCover"), None)
            st = p.get("ContentDate", {}).get("Start")
            rows.append({"target_date": t.date.date(), "regime": t.period, "tile": tile,
                         "name": n, "product_id": p.get("Id"), "sensing_start": st,
                         "size_bytes": int(p.get("ContentLength") or 0),
                         "cloud_cover": cc,
                         "days_from_target": abs((pd.to_datetime(st).tz_localize(None)
                                                  - t.date).days)})
    df = pd.DataFrame(rows).drop_duplicates(["name", "target_date"])
    if df.empty:
        print("nothing found"); return

    # best scene per (target_date, tile): fewest clouds, then closest in time
    best = (df.sort_values(["target_date", "tile", "cloud_cover", "days_from_target"])
              .groupby(["target_date", "tile"]).first().reset_index())
    best.to_csv(CATALOG / "sentinel2_selected_by_tile.csv", index=False)

    held = {p.name.replace(".zip", "") for p in (CFG.ROOT / "data/raw/sentinel").glob("*.zip")}
    best["already_held"] = best.name.isin(held)
    todo = best[~best.already_held]

    print(f"candidates: {len(df)}   selected: {len(best)}  "
          f"({best.target_date.nunique()} dates x {best.tile.nunique()} tiles)")
    print(best.groupby("tile").agg(n=("name", "size"),
                                   GB=("size_bytes", lambda x: round(x.sum() / 1e9, 2)),
                                   median_cloud=("cloud_cover", "median")).to_string())
    print(f"\nalready held: {int(best.already_held.sum())}")
    print(f"TO DOWNLOAD: {len(todo)} scenes, {todo.size_bytes.sum()/1e9:.2f} GB")
    print("\nper target date:")
    print(best.pivot_table(index="target_date", columns="tile", values="cloud_cover",
                          aggfunc="first").round(2).to_string())


if __name__ == "__main__":
    main()
