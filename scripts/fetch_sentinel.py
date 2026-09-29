#!/usr/bin/env python
"""Phase 19 step 1-4 — download and verify the 21 selected Sentinel-2 L2A scenes.

Raw ZIPs are immutable: written once to data/raw/sentinel/, never rewritten.
Every scene is verified (size, zip validity, required bands, tile, time, SHA-256)
and registered in outputs/tables/sentinel_scene_inventory.csv.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from swot_dnipro import config as CFG

#: Raw scenes live on the bulk volume, not the repo disk (see config.BULK_ROOT).
#: This used to be CFG.ROOT/"data"/"raw"/"sentinel". The 83 scenes this script
#: verified are on the bulk volume under data_swot/sentinel, so the repo-disk
#: path found no cache and a re-run would have re-fetched ~68 GB onto the wrong
#: disk, on top of contradicting the project's own storage rule.
RAW = CFG.BULK_ROOT / "data_swot" / "sentinel"
RAW.mkdir(parents=True, exist_ok=True)
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
NEEDED_BANDS = ("B03", "B08", "B11", "SCL")


class KeepAuth(requests.Session):
    """CDSE redirects catalogue -> download host; requests strips Authorization on a
    host change, but CDSE requires it to survive the redirect."""

    def rebuild_auth(self, prepared, response):
        return


def token() -> str:
    c = json.loads((pathlib.Path.home() / ".config/cdse/credentials.json").read_text())
    r = requests.post(TOKEN_URL, timeout=60, data={
        "client_id": "cdse-public", "username": c["username"],
        "password": c["password"], "grant_type": "password"})
    r.raise_for_status()
    return r.json()["access_token"]


def verify(zpath: Path, expected_size: int) -> dict:
    """Repair 2026-09-20 (finding F10): `is_zipfile` and `namelist` read only the archive's directory, so a payload
    corrupted in place -- same length, same directory -- passed every flag here while `testzip` reported the member as
    bad. The per-member CRC is now part of the verdict, and `complete` is the single field a consumer should read.
    The SHA-256 is provenance only: no independent expected digest is published for these products.
    """
    out = {"size_bytes": zpath.stat().st_size, "size_match": zpath.stat().st_size == expected_size,
           "valid_zip": False, "bands_present": "", "all_bands_ok": False, "crc_ok": False,
           "crc_bad_member": "", "sha256": "", "complete": False}
    if not zipfile.is_zipfile(zpath):
        return out
    out["valid_zip"] = True
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        bad = z.testzip()
    out["crc_ok"] = bad is None
    out["crc_bad_member"] = bad or ""
    found = [b for b in NEEDED_BANDS if any(b in n and n.endswith(".jp2") for n in names)]
    out["bands_present"] = "+".join(found)
    out["all_bands_ok"] = len(found) == len(NEEDED_BANDS)
    h = hashlib.sha256()
    with open(zpath, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    out["sha256"] = h.hexdigest()
    out["complete"] = bool(out["size_match"] and out["valid_zip"] and out["crc_ok"] and out["all_bands_ok"])
    return out


def main() -> None:
    best = pd.read_csv(CFG.ROOT / "data/catalog/sentinel2_selected_by_tile.csv")
    # Operator decision 2026-09-07: acquire all three tiles that carry post-breach
    # ATL13 observations — T36TWT + T36TXT (reservoir body, 42.4 % + middle) plus
    # T36TWS (dam / lower end, another 15 % of points). Maximum coverage set,
    # ~50 GB. See scripts/reselect_sentinel.py for the coverage measurement.
    best = best[best.tile.isin(("T36TWT", "T36TXT", "T36TWS"))].reset_index(drop=True)
    print(f"selection: {len(best)} scenes, "
          f"{best.size_bytes.sum()/1e9:.2f} GB, tiles {sorted(best.tile.unique())}", flush=True)
    s = KeepAuth()
    tok = token()
    s.headers.update({"Authorization": f"Bearer {tok}"})
    t_token = time.time()

    rows = []
    for i, r in enumerate(best.itertuples(), 1):
        zpath = RAW / f"{r.name}.zip"
        status = "CACHED"
        if not zpath.exists():
            if time.time() - t_token > 1500:          # refresh before the 1800 s expiry
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
                print(f"[{i:2d}/{len(best)}] FAIL {r.name}: {exc}", flush=True)
                rows.append({"target_date": r.target_date, "regime": r.regime,
                             "name": r.name, "product_id": r.product_id,
                             "sensing_start": r.sensing_start, "cloud_cover": r.cloud_cover,
                             "days_from_target": r.days_from_target,
                             "status": f"FAILED: {type(exc).__name__}", "size_bytes": 0,
                             "size_match": False, "valid_zip": False, "bands_present": "",
                             "all_bands_ok": False, "sha256": "", "local_path": ""})
                continue
        v = verify(zpath, int(r.size_bytes))
        tile = r.name.split("_")[5] if len(r.name.split("_")) > 5 else ""
        rows.append({"target_date": r.target_date, "regime": r.regime, "name": r.name,
                     "product_id": r.product_id, "tile": tile,
                     "sensing_start": r.sensing_start, "cloud_cover": r.cloud_cover,
                     "days_from_target": r.days_from_target, "status": status,
                     "local_path": str(zpath), **v})
        print(f"[{i:2d}/{len(best)}] {r.target_date} {tile} cloud={r.cloud_cover:6.3f}% "
              f"{status} size_ok={v['size_match']} bands={v['bands_present']}", flush=True)
        # incremental write so a mid-run crash keeps the verified rows
        pd.DataFrame(rows).to_csv(CFG.TABLES / "sentinel_scene_inventory.csv", index=False)

    inv = pd.DataFrame(rows)
    inv.to_csv(CFG.TABLES / "sentinel_scene_inventory.csv", index=False)
    ok = inv[inv.get("complete", False) == True]          # noqa: E712 -- column may be absent on an all-failed run
    print(f"\nverified OK: {len(ok)}/{len(inv)}   total {inv.size_bytes.sum()/1e9:.2f} GB")
    if len(ok) < len(inv):
        print("PROBLEM SCENES:")
        cols = [c for c in ("target_date", "name", "status", "size_match", "valid_zip",
                            "crc_ok", "crc_bad_member", "bands_present") if c in inv]
        print(inv[~inv.index.isin(ok.index)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
