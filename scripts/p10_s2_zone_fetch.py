#!/usr/bin/env python
"""P10 fetch — download the WESTERN_S2_TILE_SET scenes p10 froze but never
pulled, for one zone's manifest at a time.

p10_s2_zone_manifest.py measured coverage and wrote a frozen, stratified
manifest per zone (outputs/tables/p10_zone_*_s2_manifest.csv) -- selection
only, no bytes moved. This is the fetch step, deliberately separate so a
~440 GB pull across three zones is never one unreviewable action.

Reuses p1_targeted_fetch.py's CDSE machinery (token refresh, one-connection-
at-a-time retry/backoff -- CDSE 429'd at 5 concurrent in this project before)
rather than reimplementing it. The only thing changed is WHERE it writes and
WHAT it reads:

    RAW is monkey-patched to <S2_ZONE_FETCH_DIR> (drive F by default, not the
    repo disk p1_targeted_fetch.py itself uses) -- the project convention is
    bulk downloads go to F, and p1_targeted_fetch.py's own repo-disk choice
    was a documented one-off for a ~79 GB pull, not a rule for a ~440 GB one.

    Rows come from the p10 manifest's `tiles` column (pipe-separated, e.g.
    "36TVS|36TVT"), expanded one row per (date, tile) and restricted to
    WESTERN_S2_TILE_SET -- the eastern tiles in a ZONE_4 row are already on
    disk and are skipped here on purpose.

Usage
-----
python scripts/p10_s2_zone_fetch.py --zone ZONE_2_KHERSON_DELTA
python scripts/p10_s2_zone_fetch.py --zone ZONE_2_KHERSON_DELTA --dry-run
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from swot_dnipro import config as CFG
import p1_targeted_fetch as P1F

WESTERN = {"36TUS", "36TUT", "36TVS", "36TVT"}
S2_ZONE_FETCH_DIR = Path("/mnt/f/data_kakhovka_dem_swot/s2_zone_fetch")
MANIFESTS = {
    "ZONE_2_KHERSON_DELTA": "p10_zone_2_kherson_delta_s2_manifest.csv",
    "ZONE_3_DNIPRO_BUG_ESTUARY": "p10_zone_3_dnipro_bug_estuary_s2_manifest.csv",
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY": "p10_zone_4_dam_to_kherson_floodway_s2_manifest.csv",
}


def rows_for(zone: str) -> pd.DataFrame:
    man = pd.read_csv(CFG.TABLES / MANIFESTS[zone])
    pairs = set()
    for _, r in man.iterrows():
        for t in str(r.tiles).split("|"):
            if t in WESTERN:
                pairs.add((r.date, t))
    return pd.DataFrame(sorted(pairs), columns=["date", "tile_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=list(MANIFESTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = rows_for(args.zone)
    print("=" * 78)
    print(f"P10 fetch — {args.zone}, WESTERN_S2_TILE_SET only")
    print("=" * 78)
    print(f"  manifest: {MANIFESTS[args.zone]}")
    print(f"  {len(todo)} unique (date, tile) pairs to fetch")
    print(f"  destination: {S2_ZONE_FETCH_DIR}")
    if args.dry_run:
        print(todo.to_string(index=False))
        return

    S2_ZONE_FETCH_DIR.mkdir(parents=True, exist_ok=True)
    P1F.RAW = S2_ZONE_FETCH_DIR  # redirect the reused fetch machinery to F

    box = P1F.TokenBox()
    rows = []
    with ThreadPoolExecutor(max_workers=P1F.N_WORKERS) as ex:
        futs = {ex.submit(P1F.fetch_one, r, box): r for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(f"[{i}/{len(todo)}] done", flush=True)

    out = pd.DataFrame(rows)
    out_path = CFG.ROOT / "data" / "catalog" / f"p10_{args.zone.lower()}_s2_fetch_log.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print(out.status.value_counts().to_string() if len(out) else "nothing processed")


if __name__ == "__main__":
    main()
