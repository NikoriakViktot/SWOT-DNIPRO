#!/usr/bin/env python
"""Phase 19 steps 3-4 — raw-data manifest.

Raw Sentinel-2 ZIPs live on the external volume (data/raw/sentinel is a symlink to
E:\\data_swot\\sentinel) and are treated as immutable: this script only reads them.
It records SHA-256, size, sensing time, tile and band inventory for every raw file
and rolls the Sentinel + SWORD + reference inputs into one manifest.
"""
from __future__ import annotations

import hashlib
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from swot_dnipro import config as CFG

RAW_S2 = CFG.ROOT / "data" / "raw" / "sentinel"
SWORD_DIR = CFG.ROOT / "data" / "reference" / "river_network"
CATALOG = CFG.ROOT / "data" / "catalog"


def sha256(p: Path, cap_gb: float = 8.0) -> str:
    if p.stat().st_size > cap_gb * 1e9:
        return "SKIPPED_TOO_LARGE"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bands(p: Path) -> str:
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
        return "+".join(b for b in ("B03", "B08", "B11", "SCL")
                        if any(f"_{b}_" in n and n.endswith(".jp2") for n in names))
    except Exception:
        return ""


def main() -> None:
    rows = []
    for p in sorted(RAW_S2.glob("*.zip")):
        parts = p.name.split("_")
        rows.append({
            "family": "Sentinel-2 L2A", "file": p.name,
            "path": str(p.resolve()), "bytes": p.stat().st_size,
            "sensing_time": parts[2] if len(parts) > 2 else "",
            "tile": parts[5] if len(parts) > 5 else "",
            "valid_zip": zipfile.is_zipfile(p), "bands": bands(p),
            "sha256": sha256(p),
            "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        })
    for p in sorted(SWORD_DIR.rglob("*")):
        if p.is_file() and p.suffix in (".nc", ".zip", ".gpkg", ".shp"):
            rows.append({
                "family": "SWORD v16", "file": p.name, "path": str(p.resolve()),
                "bytes": p.stat().st_size, "sensing_time": "", "tile": "",
                "valid_zip": zipfile.is_zipfile(p) if p.suffix == ".zip" else "",
                "bands": "", "sha256": sha256(p),
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            })
    for label, p in [("EGG2015 quasigeoid", CFG.EGG2015_TIF),
                     ("EPSG:9902 grid", CFG.UA2019Z_ASC)]:
        p = Path(p)
        if p.exists():
            rows.append({"family": "reference", "file": p.name, "path": str(p.resolve()),
                         "bytes": p.stat().st_size, "sensing_time": "", "tile": "",
                         "valid_zip": "", "bands": "", "sha256": sha256(p),
                         "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime,
                                                             timezone.utc).isoformat()})

    df = pd.DataFrame(rows)
    CATALOG.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG / "raw_data_manifest.csv", index=False)
    (RAW_S2 / "MANIFEST.csv").write_text(
        df[df.family == "Sentinel-2 L2A"].to_csv(index=False))

    print(f"manifest: {len(df)} raw files, {df.bytes.sum()/1e9:.1f} GB")
    print(df.groupby("family").agg(files=("file", "size"),
                                   GB=("bytes", lambda x: round(x.sum() / 1e9, 2))).to_string())
    s2 = df[df.family == "Sentinel-2 L2A"]
    bad = s2[(s2.valid_zip != True) | (s2.bands != "B03+B08+B11+SCL")]
    if len(bad):
        print(f"\n!! {len(bad)} Sentinel ZIPs incomplete:")
        print(bad[["file", "valid_zip", "bands"]].to_string(index=False))
    else:
        print(f"\nall {len(s2)} Sentinel ZIPs: valid + 4 bands present")
    print(f"\n-> {CATALOG/'raw_data_manifest.csv'}\n-> {RAW_S2/'MANIFEST.csv'}")


if __name__ == "__main__":
    main()
