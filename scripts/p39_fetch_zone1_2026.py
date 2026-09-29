#!/usr/bin/env python
"""P39 -- fetch the 2026 leaf-on Sentinel-2 scenes for the ZONE_1 (reservoir) eastern tiles.

Audit 20260918T175108Z, GAP-001: no 2026 scene exists on 36TWS/36TWT/36TXS/36TXT anywhere on
disk, so the former reservoir bed has no 2026 optical state (succession, roughness CURRENT_2026).
The western tiles (ZONE_2/3/4) already have 50 scenes for 2026.

Discovery is done here directly against CDSE OData (cloudCover attribute) because no frozen
manifest covers 2026; the download itself reuses p1_targeted_fetch's machinery (token box,
serial connection, retry/backoff, verify), with RAW monkey-patched to the bulk volume
(`$BULK_ROOT/data_swot/sentinel`, where the other eastern scenes live) -- bulk never goes to
the repo disk (plan 15 WP0.3).

Selection rule (fixed before running): per tile, every L2A product between --start and --end
with cloudCover < --max-cloud, then keep at most --per-month per tile-month, preferring the
lowest cloud; a (date, tile) already on disk is skipped by already_have().

Usage
-----
python scripts/p39_fetch_zone1_2026.py --dry-run
python scripts/p39_fetch_zone1_2026.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import requests

from swot_dnipro import config as CFG
import p1_targeted_fetch as P1F

EASTERN = ("36TWS", "36TWT", "36TXS", "36TXT")
DEST = CFG.BULK_ROOT / "data_swot" / "sentinel"


def discover(tile: str, start: str, end: str, max_cloud: float, tok: str) -> list[dict]:
    f = (f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL2A') and contains(Name,'_T{tile}_') "
         f"and ContentDate/Start gt {start}T00:00:00.000Z and ContentDate/Start lt {end}T23:59:59.999Z "
         f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
         f"and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})")
    out, url, params = [], P1F.ODATA, {"$filter": f, "$top": 200, "$expand": "Attributes"}
    while url:
        r = requests.get(url, params=params, headers={"Authorization": f"Bearer {tok}"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        for p in j.get("value", []):
            cc = next((a["Value"] for a in p.get("Attributes", []) if a.get("Name") == "cloudCover"), None)
            out.append(dict(tile_id=tile, name=p["Name"], date=p["ContentDate"]["Start"][:10], cloud=cc,
                            size_gb=round(int(p.get("ContentLength") or 0) / 1e9, 2)))
        url, params = j.get("@odata.nextLink"), None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--max-cloud", type=float, default=20.0)
    ap.add_argument("--per-month", type=int, default=3, help="max scenes per tile per month (lowest cloud first)")
    ap.add_argument("--min-gb", type=float, default=0.5, help="drop products smaller than this (partial tiles)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    P1F.RAW = DEST                      # bulk volume, same store as the other eastern scenes
    box = P1F.TokenBox()
    cand = []
    for t in EASTERN:
        cand += discover(t, a.start, a.end, a.max_cloud, box.get())
    C = pd.DataFrame(cand)
    if C.empty:
        print("no candidates on CDSE for the selection rule"); return
    C = C[C.size_gb >= a.min_gb]           # orbit-edge slivers (36TXS 0.04 GB, 36TXT 0.15 GB) are not scenes
    C = C.sort_values(["tile_id", "date", "cloud"]).drop_duplicates(["tile_id", "date"])
    C["month"] = C.date.str[:7]
    C = C.sort_values(["tile_id", "month", "cloud"]).groupby(["tile_id", "month"]).head(a.per_month).sort_values(["date", "tile_id"])
    C["on_disk"] = [str(P1F.already_have(d, t) or "") for d, t in zip(C.date, C.tile_id)]
    plan_path = CFG.TABLES / "p39_zone1_2026_fetch_plan.csv"
    C.to_csv(plan_path, index=False)
    todo = C[C.on_disk == ""]
    print(f"candidates {len(cand)} -> selected {len(C)} (date,tile); on disk {int((C.on_disk != '').sum())}; "
          f"to fetch {len(todo)} = {todo.size_gb.sum():.1f} GB -> {plan_path}")
    print(C[["date", "tile_id", "cloud", "size_gb", "on_disk"]].to_string(index=False))
    if a.dry_run or todo.empty:
        return
    rows = [P1F.fetch_one(r, box) for r in todo.itertuples()]
    log = pd.DataFrame(rows)
    log_path = CFG.ROOT / "data" / "catalog" / "p39_zone1_2026_fetch_log.csv"
    log.to_csv(log_path, index=False)
    print(f"\n-> {log_path}\n{log.status.value_counts().to_string()}")


if __name__ == "__main__":
    main()
