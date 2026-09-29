#!/usr/bin/env python
"""Phase 11 discovery — Sentinel scenes for post-breach water masks.

Metadata only; nothing is downloaded. Targets the ICESat-2 profile dates that carry
the reservoir->river result, because a water mask is only useful where we also have a
longitudinal WSE profile to classify (Phase 19: main channel vs residual ponds).

Auth: CDSE Keycloak password grant -> access token (30 min).
Catalogue: OData v1.
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

#: former reservoir footprint, as a closed WKT polygon (lon lat order)
RESERVOIR_WKT = ("POLYGON((33.35 46.75,35.35 46.75,35.35 47.79,33.35 47.79,33.35 46.75))")
MAX_CLOUD = 25.0
WINDOW_DAYS = 5


def token() -> str:
    c = json.loads((pathlib.Path.home() / ".config/cdse/credentials.json").read_text())
    r = requests.post(TOKEN_URL, timeout=60, data={
        "client_id": "cdse-public", "username": c["username"],
        "password": c["password"], "grant_type": "password"})
    r.raise_for_status()
    return r.json()["access_token"]


def search(collection: str, start: str, end: str, tok: str, cloud=None) -> list[dict]:
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
    tok = token()
    print("CDSE auth OK\n")

    # dates that carry the reservoir->river result
    sl = pd.read_csv(CFG.FIGDATA / "FigG_perdate_slopes_robust.csv", parse_dates=["date"])
    targets = sl[sl.period.isin(["BREACH_DRAWDOWN", "POST_BREACH"])].sort_values("date")
    # plus a pre-breach reference and the breach itself
    extra = pd.DataFrame({"date": pd.to_datetime(["2023-05-20", "2023-06-10"]),
                          "period": ["PRE_BREACH", "BREACH_DRAWDOWN"]})
    targets = pd.concat([targets[["date", "period"]], extra]).sort_values("date")

    rows = []
    for coll, cloud in (("SENTINEL-2", MAX_CLOUD), ("SENTINEL-1", None)):
        for t in targets.itertuples():
            a = (t.date - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
            b = (t.date + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
            try:
                res = search(coll, a, b, tok, cloud)
            except Exception as exc:
                print(f"  !! {coll} {t.date.date()}: {exc}")
                continue
            for p in res:
                name = p.get("Name", "")
                if coll == "SENTINEL-2" and "MSIL2A" not in name:
                    continue
                if coll == "SENTINEL-1" and "GRD" not in name:
                    continue
                cc = next((a_["Value"] for a_ in p.get("Attributes", [])
                           if a_.get("Name") == "cloudCover"), None)
                rows.append({
                    "collection": coll, "target_date": t.date.date(), "regime": t.period,
                    "name": name, "product_id": p.get("Id"),
                    "sensing_start": p.get("ContentDate", {}).get("Start"),
                    "size_bytes": int(p.get("ContentLength") or 0),
                    "cloud_cover": cc, "online": p.get("Online"),
                    "days_from_target": abs((pd.to_datetime(
                        p.get("ContentDate", {}).get("Start")).tz_localize(None) - t.date).days),
                })
    df = pd.DataFrame(rows).drop_duplicates(["collection", "name", "target_date"])
    CATALOG.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CATALOG / "sentinel_discovered.parquet", index=False)
    df.to_csv(CATALOG / "sentinel_discovered.csv", index=False)

    print(f"discovered {len(df):,} scenes over the reservoir AOI\n")
    if df.empty:
        return
    print(df.groupby(["collection", "regime"]).agg(
        scenes=("name", "size"), GB=("size_bytes", lambda x: round(x.sum() / 1e9, 1)),
        target_dates=("target_date", "nunique")).to_string())

    # best S2 scene per target date
    s2 = df[(df.collection == "SENTINEL-2") & df.cloud_cover.notna()].copy()
    if len(s2):
        best = (s2.sort_values(["target_date", "cloud_cover", "days_from_target"])
                  .groupby("target_date").first().reset_index())
        best.to_csv(CATALOG / "sentinel2_best_per_date.csv", index=False)
        print(f"\n=== best (lowest-cloud) S2 L2A within ±{WINDOW_DAYS} d of each profile date ===")
        print(best[["target_date", "regime", "sensing_start", "cloud_cover",
                    "days_from_target", "size_bytes"]].assign(
            GB=lambda d: (d.size_bytes / 1e9).round(2)).drop(columns="size_bytes").to_string(index=False))
        print(f"\nminimal acquisition (1 best S2 scene per profile date): "
              f"{best.size_bytes.sum()/1e9:.1f} GB for {len(best)} scenes")


if __name__ == "__main__":
    main()
