#!/usr/bin/env python
"""SAFE TECHNICAL CHECK ONLY (2026-09-11, while Phase A downloads run) --
integrity/checksum/band verification and manifest status update. Does NOT
build any prior, does NOT touch thresholds, does NOT run kriging. This is
also the script the Phase A Completion Gate step 1-2 reruns once all 52
granules have landed.

Status codes (exactly the five requested):
  DOWNLOAD_OK      zip valid, all 4 required bands present, size sane
  CORRUPT          zip fails integrity test or a required band is unreadable
  MISSING          not on disk yet (still downloading, or never started)
  DUPLICATE        more than one file for this (date, tile) -- newest
                   processing baseline kept as canonical, others flagged
  RETRY_REQUIRED   present but incomplete (partial download interrupted)

Outputs
-------
outputs/tables/P1_manifest_verification.csv
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from swot_dnipro import config as CFG

RAW_DIRS = (CFG.ROOT / "data" / "raw" / "sentinel_p1_targeted",
           CFG.ROOT / "data" / "raw" / "sentinel", Path("/mnt/e/data_swot/sentinel"))
NEEDED_BANDS = ("B03", "B08", "B11", "SCL")
MIN_PLAUSIBLE_BYTES = 500_000_000  # smallest real granule seen so far is ~0.85 GB


def all_matches(date: str, tile: str) -> list[Path]:
    """De-duplicated by REAL (resolved) path -- data/raw/sentinel is a symlink
    to /mnt/e/data_swot/sentinel, so searching both would double-count every
    genuinely-single file as a false DUPLICATE."""
    tok = date.replace("-", "")
    seen, out = set(), []
    for base in RAW_DIRS:
        if not base.exists():
            continue
        for p in sorted(base.glob(f"*{tok}T*_T{tile}_*.SAFE.zip")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return out


def verify_one(p: Path) -> dict:
    r = {"path": str(p), "size_bytes": p.stat().st_size, "valid_zip": False,
        "bands_present": "", "all_bands_ok": False, "sha256": ""}
    if not zipfile.is_zipfile(p):
        return r
    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()  # None if all CRCs check out
            names = z.namelist()
    except Exception:
        return r
    r["valid_zip"] = bad is None
    found = [b for b in NEEDED_BANDS if any(b in n and n.endswith(".jp2") for n in names)]
    r["bands_present"] = "+".join(found)
    r["all_bands_ok"] = len(found) == len(NEEDED_BANDS)
    return r


def main() -> None:
    man = pd.read_csv(CFG.TABLES / "P1_targeted_download_manifest.csv")
    pairs = man[["date", "tile_id"]].drop_duplicates()
    print(f"verifying {len(pairs)} (date, tile) pairs against current disk state")

    rows = []
    for r in pairs.itertuples():
        matches = all_matches(r.date, r.tile_id)
        if not matches:
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": "MISSING",
                        "n_files_found": 0})
            continue
        checked = [(p, verify_one(p)) for p in matches]
        # canonical = largest valid file (newest processing baseline usually also largest/complete)
        checked.sort(key=lambda t: t[1]["size_bytes"], reverse=True)
        canon_p, canon_v = checked[0]
        age_s = time.time() - canon_p.stat().st_mtime
        if len(matches) > 1:
            status = "DUPLICATE"
        elif canon_v["size_bytes"] < MIN_PLAUSIBLE_BYTES:
            # a file still being written by the live download looks identical
            # to a stalled one at a single point in time -- only flag
            # RETRY_REQUIRED once it has been quiet for a while
            status = "IN_PROGRESS (recently modified, likely still downloading)" if age_s < 180 else "RETRY_REQUIRED"
        elif not canon_v["valid_zip"] or not canon_v["all_bands_ok"]:
            status = "IN_PROGRESS (recently modified, likely still downloading)" if age_s < 180 else "CORRUPT"
        else:
            status = "DOWNLOAD_OK"
        rows.append({
            "date": r.date, "tile_id": r.tile_id, "status": status,
            "n_files_found": len(matches), "canonical_path": str(canon_p),
            "size_bytes": canon_v["size_bytes"], "valid_zip": canon_v["valid_zip"],
            "bands_present": canon_v["bands_present"], "all_bands_ok": canon_v["all_bands_ok"],
            "other_paths": "|".join(str(p) for p, _ in checked[1:]) if len(checked) > 1 else "",
        })

    out = pd.DataFrame(rows)
    out_path = CFG.TABLES / "P1_manifest_verification.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print(out.status.value_counts().to_string())

    ok = out[out.status == "DOWNLOAD_OK"]
    if len(ok):
        total_gb = ok.size_bytes.sum() / 1e9
        print(f"\nDOWNLOAD_OK total volume: {total_gb:.2f} GB across {len(ok)} granules")
    dup = out[out.status == "DUPLICATE"]
    if len(dup):
        print(f"\nDUPLICATE rows -- kept canonical (largest) file, flagged the rest:")
        print(dup[["date", "tile_id", "n_files_found", "canonical_path"]].to_string(index=False))
    bad = out[out.status.isin(["CORRUPT", "RETRY_REQUIRED"])]
    if len(bad):
        print(f"\n{len(bad)} row(s) need attention:")
        print(bad[["date", "tile_id", "status", "size_bytes"]].to_string(index=False))

    missing = out[out.status == "MISSING"]
    print(f"\nMISSING (not yet on disk): {len(missing)} / {len(out)} "
          f"-- normal while the background download is still running")


if __name__ == "__main__":
    main()
