#!/usr/bin/env python
"""P1 STEP 1 + STEP 6-7 -- download only the granules listed in
outputs/tables/P1_targeted_download_manifest.csv that are not already cached.

CDSE OData is queried fresh per (date, tile) to get the exact download Id --
the manifest's product_id column came from the Planetary Computer STAC
catalogue (P1A) and is NOT the same identifier space as CDSE's, so it is
used only to confirm date/tile, never passed directly to the download URL.

Downloads go to data/raw/sentinel_p1_targeted/ (a NEW local directory on the
WSL-root filesystem, ~663 GB free) -- deliberately NOT the existing
data/raw/sentinel symlink target (/mnt/e/data_swot/sentinel, verified this
session at 22 GB free / 91% full). Do not point new downloads at E: until
that drive is cleaned up.

Raw ZIPs are immutable: written once, verified, never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import requests

from swot_dnipro import config as CFG

RAW = CFG.ROOT / "data" / "raw" / "sentinel_p1_targeted"
RAW.mkdir(parents=True, exist_ok=True)
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
NEEDED_BANDS = ("B03", "B08", "B11", "SCL")
# CDSE download endpoint 429'd immediately at 5 concurrent connections
# (verified this session -- every 3rd+ simultaneous request failed). Serial,
# one connection at a time, is what actually completed the test download.
N_WORKERS = 1
MAX_RETRIES = 6


class KeepAuth(requests.Session):
    def rebuild_auth(self, prepared, response):
        return


def token() -> str:
    c = json.loads((pathlib.Path.home() / ".config/cdse/credentials.json").read_text())
    r = requests.post(TOKEN_URL, timeout=60, data={
        "client_id": "cdse-public", "username": c["username"],
        "password": c["password"], "grant_type": "password"})
    r.raise_for_status()
    return r.json()["access_token"]


def cdse_find(date: str, tile: str, tok: str) -> dict | None:
    """Find the CDSE product for one (date, tile). Returns None if not found."""
    f = (f"Collection/Name eq 'SENTINEL-2' "
         f"and contains(Name,'MSIL2A') and contains(Name,'{tile}') "
         f"and ContentDate/Start gt {date}T00:00:00.000Z "
         f"and ContentDate/Start lt {date}T23:59:59.999Z")
    r = requests.get(ODATA, params={"$filter": f, "$top": 20},
                     headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    r.raise_for_status()
    vals = r.json().get("value", [])
    if not vals:
        return None
    # prefer the newest processing baseline if duplicates exist
    vals.sort(key=lambda p: p.get("Name", ""))
    return vals[-1]


def verify(zpath: Path, expected_size: int | None = None) -> dict:
    """Completeness of a SAFE zip (repair 2026-09-20, findings F02/F10).

    `is_zipfile` and `namelist` read only the archive's directory: a payload corrupted in place, keeping its length and its
    directory intact, passed all of them (reproduced on a synthetic archive whose B03 member was altered by one byte). The
    CRC stored per member is the check that catches it, so `testzip` is now the decisive criterion. A truncated download is
    caught earlier and more cheaply -- the end-of-central-directory record sits at the end of the file, so a partial zip is
    not a zip at all -- which is why completeness no longer rests on an arbitrary size floor.

    The SHA-256 is recorded for provenance only. It is not compared with anything, because no independent expected digest is
    published for these products; the CRCs inside the archive are the integrity evidence that does exist.
    """
    st = zpath.stat()
    out = {"size_bytes": st.st_size, "size_match": (expected_size is None or st.st_size == expected_size),
           "expected_size": expected_size or 0, "valid_zip": False, "bands_present": "", "all_bands_ok": False,
           "crc_ok": False, "crc_bad_member": "", "sha256": "", "complete": False, "mtime": st.st_mtime}
    if not zipfile.is_zipfile(zpath):
        return out
    out["valid_zip"] = True
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        bad = z.testzip()                       # first member failing its stored CRC, else None
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
    out["complete"] = bool(out["valid_zip"] and out["crc_ok"] and out["all_bands_ok"] and out["size_match"])
    return out


def sidecar(zpath: Path) -> Path:
    return zpath.with_name(zpath.name + ".verified.json")


def verified(zpath: Path, expected_size: int | None = None) -> dict:
    """verify(), memoised in a sidecar so a multi-GB archive is CRC-checked once rather than on every run.

    The sidecar is trusted only while the file it describes still has the same size and mtime; anything else re-verifies.
    """
    sc = sidecar(zpath)
    try:
        st = zpath.stat()
        rec = json.loads(sc.read_text())
        if rec.get("size_bytes") == st.st_size and abs(float(rec.get("mtime", -1)) - st.st_mtime) < 1e-6:
            if expected_size is None or rec.get("expected_size") in (0, expected_size):
                return rec
    except Exception:
        pass
    rec = verify(zpath, expected_size)
    try:
        sc.write_text(json.dumps(rec, indent=1))
    except Exception:
        pass
    return rec


#: Every directory a Sentinel-2 SAFE zip has ever been written to in this
#: project. Searched in order, so a scene already on disk is never re-fetched
#: just because a later run chose a different destination. ``/mnt/e/...`` is a
#: drive this project no longer uses and is kept only so an old mount still
#: counts as a hit; the bulk-volume entries are where scenes actually are.
_SCENE_SEARCH_PATH = (
    CFG.BULK_ROOT / "data_swot" / "sentinel",     # fetch_sentinel.py (83 scenes)
    CFG.BULK_ROOT / "s2_zone_fetch",              # p10_s2_zone_fetch.py (western)
    CFG.ROOT / "data" / "raw" / "sentinel_p1_targeted",
    CFG.ROOT / "data" / "raw" / "sentinel",
    Path("/mnt/e/data_swot/sentinel"),
)


def already_have(date: str, tile: str, min_bytes: int = 1_000_000) -> Path | None:
    """A cached hit must actually be a complete, readable archive.

    Repair 2026-09-20 (finding F02). This used to accept any file with the right name and at least 500 MB, so an aborted
    download -- which `fetch_one` wrote straight to the final path -- came back as ALREADY_CACHED on the next run and was
    never retried. A sparse 500 MB file that was not a zip at all passed. Completeness is now decided by the archive's own
    CRCs (see `verify`), memoised in a sidecar so the cost is paid once per file. `min_bytes` survives only as a cheap
    screen against obvious stubs; it is deliberately far below any real product, since a valid scene smaller than the old
    floor used to be rejected as if it were partial.
    """
    for base in (RAW, *_SCENE_SEARCH_PATH):
        try:
            if not base.exists():
                continue
            cand = sorted(base.glob(f"*{date.replace('-', '')}T*_T{tile}_*.SAFE.zip"))
        except OSError as exc:
            # A retired mount point raises rather than returning False: /mnt/e is listed above only so an old
            # mount still counts as a hit, and an unmounted one must not abort the whole search.
            print(f"  {date} {tile}: search path {base} unavailable ({type(exc).__name__}) -- skipped", flush=True)
            continue
        for p in cand:
            if p.stat().st_size < min_bytes:
                print(f"  {date} {tile}: ignoring stub {p.name} ({p.stat().st_size} B)", flush=True)
                continue
            v = verified(p)
            if v["complete"]:
                return p
            print(f"  {date} {tile}: {p.name} is on disk but INCOMPLETE "
                  f"(valid_zip={v['valid_zip']} crc_ok={v['crc_ok']} bands={v['bands_present']}"
                  f"{' bad=' + v['crc_bad_member'] if v['crc_bad_member'] else ''}) -- it will be re-fetched", flush=True)
    return None


class TokenBox:
    """Thread-safe token holder -- CDSE tokens last 1800 s; several workers
    downloading at ~6 MB/s each can run past that, so refresh centrally."""
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._tok = token()
        self._t0 = time.time()

    def get(self) -> str:
        with self._lock:
            if time.time() - self._t0 > 1400:
                self._tok = token()
                self._t0 = time.time()
            return self._tok


def fetch_one(r, box: TokenBox) -> dict:
    existing = already_have(r.date, r.tile_id)
    if existing is not None:
        print(f"  {r.date} {r.tile_id}: already cached at {existing}", flush=True)
        return {"date": r.date, "tile_id": r.tile_id, "status": "ALREADY_CACHED",
                "path": str(existing)}
    tok = box.get()
    p = cdse_find(r.date, r.tile_id, tok)
    if p is None:
        print(f"  {r.date} {r.tile_id}: NOT FOUND on CDSE", flush=True)
        return {"date": r.date, "tile_id": r.tile_id, "status": "NOT_FOUND_CDSE"}
    name, pid, size = p["Name"], p["Id"], int(p.get("ContentLength") or 0)
    zpath = RAW / f"{name.replace('.SAFE', '')}.SAFE.zip"
    part = zpath.with_name(zpath.name + ".part")
    if zpath.exists() and verified(zpath, size)["complete"]:
        return {"date": r.date, "tile_id": r.tile_id, "status": "CACHED_THIS_RUN", "path": str(zpath)}
    url = f"{ODATA}({pid})/$value"
    print(f"  {r.date} {r.tile_id}: downloading {name} ({size/1e9:.2f} GB) ...", flush=True)
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        s = KeepAuth()
        s.headers.update({"Authorization": f"Bearer {box.get()}"})
        try:
            with s.get(url, stream=True, timeout=900, allow_redirects=True) as resp:
                if resp.status_code == 429:
                    wait = min(30 * attempt, 180)
                    print(f"  .. {r.date} {r.tile_id}: 429, backing off {wait}s "
                          f"(attempt {attempt}/{MAX_RETRIES})", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                # Write to `.part` and rename only after verification: the final path must never hold an unfinished
                # archive, or the next run's cache check sees it and stops retrying (finding F02).
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"  !! {r.date} {r.tile_id}: download failed after {attempt} "
                      f"attempts: {type(exc).__name__}: {exc}", flush=True)
                return {"date": r.date, "tile_id": r.tile_id, "status": f"FAILED:{exc}"}
            wait = min(20 * attempt, 120)
            print(f"  .. {r.date} {r.tile_id}: {type(exc).__name__}, retry in {wait}s "
                  f"(attempt {attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
    else:
        return {"date": r.date, "tile_id": r.tile_id, "status": "FAILED:429_exhausted_retries"}
    dt = time.time() - t0
    v = verify(part, size)
    if v["complete"]:
        os.replace(part, zpath)                       # atomic publication of an immutable raw product
        sidecar(zpath).write_text(json.dumps({**v, "mtime": zpath.stat().st_mtime}, indent=1))
        status = "OK"
    else:
        status = "VERIFY_FAILED"
        print(f"  !! {r.date} {r.tile_id}: verification failed -- size_match={v['size_match']} "
              f"valid_zip={v['valid_zip']} crc_ok={v['crc_ok']} bands={v['bands_present']} "
              f"{'bad_member=' + v['crc_bad_member'] if v['crc_bad_member'] else ''}; "
              f"leaving {part.name} in place, the final path is NOT written", flush=True)
    print(f"  {r.date} {r.tile_id}: -> {status}  {size/1e9:.2f} GB in {dt:.0f}s "
          f"({size/1e6/max(dt,1):.1f} MB/s)  bands={v['bands_present']}", flush=True)
    return {"date": r.date, "tile_id": r.tile_id, "status": status,
            "path": str(zpath if status == "OK" else part),
            "size_bytes": v["size_bytes"], "sha256": v["sha256"], "bands_present": v["bands_present"],
            "crc_ok": v["crc_ok"], "crc_bad_member": v["crc_bad_member"]}


def main() -> None:
    man = pd.read_csv(CFG.TABLES / "P1_targeted_download_manifest.csv")
    todo = man[man.download_required].drop_duplicates(["date", "tile_id"])
    print(f"manifest: {len(man)} rows, {len(todo)} unique (date,tile) to fetch, "
          f"running {N_WORKERS} concurrent downloads")

    box = TokenBox()
    rows = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_one, r, box): r for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(f"[{i}/{len(todo)}] done", flush=True)

    out = pd.DataFrame(rows)
    out_path = CFG.ROOT / "data" / "catalog" / "P1_targeted_fetch_log.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print(out.status.value_counts().to_string() if len(out) else "nothing processed")


if __name__ == "__main__":
    main()
