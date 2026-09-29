#!/usr/bin/env python
"""Phase 19 STEP 2 — download the targeted S2 scenes from phase19_discover_targeted.py.

Reuses fetch_sentinel.py's auth / download / verify.  Differences:
  * source list = data/catalog/sentinel2_selected_targeted.csv
  * raw ZIPs go to data/raw/sentinel/  (symlinked to /mnt/e/data_swot/sentinel — E: drive)
  * MERGES into outputs/tables/sentinel_scene_inventory.csv (never truncates the
    existing 63-scene inventory)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from swot_dnipro import config as CFG
from fetch_sentinel import KeepAuth, token, verify, RAW

SEL = CFG.ROOT / "data" / "catalog" / "sentinel2_selected_targeted.csv"
INV = CFG.TABLES / "sentinel_scene_inventory.csv"


def main() -> None:
    sel = pd.read_csv(SEL)
    if sel.empty:
        print("nothing to fetch — sentinel2_selected_targeted.csv is empty")
        return
    print(f"targeted selection: {len(sel)} scenes, {sel.size_bytes.sum()/1e9:.2f} GB, "
          f"tiles {sorted(sel.tile.unique())}\nRAW -> {RAW.resolve()}", flush=True)

    inv = pd.read_csv(INV) if INV.exists() else pd.DataFrame()
    have = set(inv.name) if len(inv) else set()

    s = KeepAuth()
    tok = token()
    s.headers.update({"Authorization": f"Bearer {tok}"})
    t_token = time.time()

    new_rows = []
    for i, r in enumerate(sel.itertuples(), 1):
        zpath = RAW / f"{r.name}.zip"
        status = "CACHED"
        if not zpath.exists():
            if time.time() - t_token > 1500:
                tok = token(); s.headers.update({"Authorization": f"Bearer {tok}"})
                t_token = time.time()
            url = (f"https://catalogue.dataspace.copernicus.eu/odata/v1/"
                   f"Products({r.product_id})/$value")
            t0 = time.time()
            try:
                with s.get(url, stream=True, timeout=1800) as resp:
                    resp.raise_for_status()
                    tmp = zpath.with_suffix(".zip.part")
                    n = 0
                    with open(tmp, "wb") as fh:
                        for ch in resp.iter_content(1 << 20):
                            fh.write(ch); n += len(ch)
                    tmp.replace(zpath)
                status = f"DOWNLOADED {n/1e6:.0f}MB {n/1e6/max(time.time()-t0,.01):.1f}MB/s"
            except Exception as exc:
                print(f"[{i:2d}/{len(sel)}] FAIL {r.name}: {exc}", flush=True)
                continue
        v = verify(zpath, int(r.size_bytes))
        new_rows.append({"target_date": r.target_date, "regime": r.regime, "name": r.name,
                         "product_id": r.product_id, "tile": r.tile,
                         "sensing_start": r.sensing_start, "cloud_cover": r.cloud_cover,
                         "days_from_target": r.days_from_target, "status": status,
                         "local_path": str(zpath), **v})
        print(f"[{i:2d}/{len(sel)}] {r.target_date} {r.tile} cloud={r.cloud_cover:6.2f}% "
              f"{status} size_ok={v['size_match']} bands={v['bands_present']}", flush=True)

    add = pd.DataFrame(new_rows)
    merged = pd.concat([inv[~inv.name.isin(add.name)] if len(inv) else inv, add],
                       ignore_index=True)
    merged.to_csv(INV, index=False)
    ok = merged[(merged.valid_zip == True) & (merged.all_bands_ok == True)]
    print(f"\ninventory now {len(merged)} scenes ({len(ok)} verified OK), "
          f"{merged.size_bytes.sum()/1e9:.2f} GB  -> {INV}")
    print(f"added/updated this run: {len(add)}")


if __name__ == "__main__":
    main()
