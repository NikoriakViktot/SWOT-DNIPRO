#!/usr/bin/env python
"""QA — re-point the fetch logs at where their files actually are.

The repository used to live at ``/home/niko/projects/SWOT-DNIPRO`` and now
lives at ``/home/niko/repo/SWOT-DNIPRO``; separately, the raw Sentinel-2
scenes were moved off the repo disk onto the bulk volume. Three registries
still carried the old absolute paths, so none of their 169 rows resolved:

    outputs/tables/sentinel_scene_inventory.csv        87 rows
    data/catalog/P1_targeted_fetch_log.csv             52 rows
    data/catalog/F4_upper_postbreach_fetch_log.csv     30 rows

No bytes were lost -- every registered scene is on the bulk volume -- but the
registries carry the SHA-256 and band checks that make those bytes auditable,
and a registry whose paths do not resolve cannot be re-checked at all.

This relocates by FILENAME, never by rewriting the old prefix: the file moved
between volumes, so string surgery on the prefix would invent a path rather
than find one. A row is only rewritten when exactly one candidate exists on
disk; ambiguous and genuinely-missing rows are reported and left untouched.

Size is verified against the recorded ``size_bytes`` where the registry has
it. SHA-256 is NOT re-computed here -- that is ~80 GB of reads and belongs in
its own deliberate run (``--verify-sha``).

Outputs
-------
Rewrites the three registries in place (a ``.bak`` copy is written first) and
prints a per-file summary.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from swot_dnipro import config as CFG

#: Every directory a raw scene may legitimately sit in today.
SEARCH_DIRS = (
    CFG.BULK_ROOT / "data_swot" / "sentinel",
    CFG.BULK_ROOT / "s2_zone_fetch",
    ROOT / "data" / "raw" / "sentinel_p1_targeted",
    ROOT / "data" / "raw" / "sentinel",
)

REGISTRIES = (
    (CFG.TABLES / "sentinel_scene_inventory.csv", "local_path"),
    (ROOT / "data" / "catalog" / "P1_targeted_fetch_log.csv", "path"),
    (ROOT / "data" / "catalog" / "F4_upper_postbreach_fetch_log.csv", "path"),
)


def build_index() -> dict[str, list[Path]]:
    """filename -> every place it exists."""
    idx: dict[str, list[Path]] = {}
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.zip"):
            idx.setdefault(p.name, []).append(p)
    return idx


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def relocate(csv: Path, col: str, idx: dict[str, list[Path]],
             verify_sha: bool) -> dict:
    if not csv.exists():
        return dict(registry=csv.name, error="missing")
    D = pd.read_csv(csv)
    if col not in D.columns:
        return dict(registry=csv.name, error=f"no column {col!r}")

    ok_before = sum(Path(str(p)).exists() for p in D[col].dropna())
    moved = missing = ambiguous = size_bad = sha_bad = sha_ok = 0
    new_paths = []
    for raw in D[col]:
        p = Path(str(raw))
        if p.exists():
            new_paths.append(str(p))
            continue
        cands = idx.get(p.name, [])
        if len(cands) == 1:
            new_paths.append(str(cands[0]))
            moved += 1
        else:
            new_paths.append(str(raw))
            if len(cands) > 1:
                ambiguous += 1
                print(f"    AMBIGUOUS {p.name}: {len(cands)} copies")
            else:
                missing += 1
                print(f"    NOT FOUND {p.name}")
    D[col] = new_paths

    if "size_bytes" in D.columns:
        for _, r in D.iterrows():
            q = Path(str(r[col]))
            if q.exists() and pd.notna(r.size_bytes):
                if q.stat().st_size != int(r.size_bytes):
                    size_bad += 1
                    print(f"    SIZE MISMATCH {q.name}")

    if verify_sha and "sha256" in D.columns:
        for _, r in D.iterrows():
            q = Path(str(r[col]))
            if q.exists() and isinstance(r.sha256, str) and len(r.sha256) == 64:
                if sha256(q) == r.sha256:
                    sha_ok += 1
                else:
                    sha_bad += 1
                    print(f"    SHA MISMATCH {q.name}")

    if moved:
        shutil.copy2(csv, csv.with_suffix(csv.suffix + ".bak"))
        D.to_csv(csv, index=False)

    ok_after = sum(Path(str(p)).exists() for p in D[col].dropna())
    return dict(registry=csv.name, rows=len(D), resolved_before=ok_before,
                resolved_after=ok_after, relocated=moved, not_found=missing,
                ambiguous=ambiguous, size_mismatch=size_bad,
                sha_ok=sha_ok, sha_mismatch=sha_bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-sha", action="store_true",
                    help="re-compute SHA-256 for every resolved row (~80 GB of reads)")
    a = ap.parse_args()

    print("=" * 78)
    print("QA — relocating fetch registries")
    print("=" * 78)
    idx = build_index()
    print(f"  indexed {len(idx)} scene filenames across {len(SEARCH_DIRS)} directories")
    for d in SEARCH_DIRS:
        print(f"    {'OK ' if d.exists() else 'absent'}  {d}")

    out = []
    for csv, col in REGISTRIES:
        print(f"\n  {csv.name}")
        out.append(relocate(csv, col, idx, a.verify_sha))

    S = pd.DataFrame(out)
    print("\n" + "=" * 78)
    print(S.to_string(index=False))
    dest = CFG.TABLES / "qa_fetch_log_relocation.csv"
    S.to_csv(dest, index=False)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
