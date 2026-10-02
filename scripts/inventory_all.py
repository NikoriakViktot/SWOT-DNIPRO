#!/usr/bin/env python
"""Phase 0 — complete inventory of every dataset across all three project trees.

Produces outputs/tables/full_dataset_inventory.{csv,parquet} and the numbers that
feed outputs/reports/full_data_acquisition_plan.md. Read-only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from swot_dnipro import config as CFG

ROOTS = {
    "SWOT-DNIPRO": Path("/home/niko/repo/SWOT-DNIPRO"),
    "icesat2-atl13-kakhovka": Path("/home/niko/repo/icesat2-atl13-kakhovka"),
    "OneDrive-legacy": Path("/mnt/c/Users/5302/OneDrive/PhD/kachovka_phd/kod/swot_dnipro"),
}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
             ".ipynb_checkpoints", ".idea", "node_modules"}
HASH_CAP = 256 * 1024 * 1024   # hash files up to 256 MB fully


def classify(p: Path, root: Path) -> tuple[str, str]:
    """Classify from the path RELATIVE to the tree root, so that the root
    directory name (e.g. ``icesat2-atl13-kakhovka``) cannot match everything."""
    s, n = str(p.relative_to(root)).lower(), p.name.lower()
    if "pixc" in n and n.endswith(".nc"):
        return "SWOT", "PIXC"
    if "riversp" in s:
        return "SWOT", "RiverSP"
    if "lakesp" in s:
        return "SWOT", "LakeSP"
    if "raster" in s and n.endswith(".nc"):
        return "SWOT", "Raster"
    if "atl13" in s:
        return "ICESat-2", "ATL13"
    if "atl03" in s:
        return "ICESat-2", "ATL03"
    if "egg_2015" in n or "egg2015" in n:
        return "geodesy", "EGG2015"
    if "ua_2019z" in n:
        return "geodesy", "EPSG:9902 grid"
    if "yearbook" in n or "post_id=" in s or "dm_h" in s or "reservoir" in n:
        return "gauge", "hydrological record"
    if "sword" in s:
        return "river network", "SWORD"
    if any(k in s for k in ("sentinel", "s1_", "s2_")):
        return "Sentinel", "imagery"
    if any(k in s for k in ("era5", "meteo", "wind")):
        return "meteo", "reanalysis/obs"
    if n.endswith((".geojson", ".shp", ".gpkg")):
        return "reference", "vector geometry"
    if n.endswith((".py", ".ipynb", ".sh", ".ps1", ".toml", ".yaml", ".yml", ".cfg")):
        return "code", "source"
    if n.endswith((".md", ".pdf", ".docx", ".txt")):
        return "docs", "documentation"
    if n.endswith((".csv", ".parquet", ".duckdb")):
        return "derived", "table"
    return "other", "other"


def sha256(p: Path, size: int) -> str:
    if size > HASH_CAP:
        return f"SKIPPED_GT_{HASH_CAP // 1024 // 1024}MB"
    h = hashlib.sha256()
    try:
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def main() -> None:
    rows = []
    for label, root in ROOTS.items():
        if not root.exists():
            print(f"  !! {label}: {root} NOT PRESENT")
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in fn:
                p = Path(dp) / f
                try:
                    st = p.stat()
                except Exception:
                    continue
                fam, prod = classify(p, root)
                rows.append({
                    "tree": label, "relative_path": str(p.relative_to(root)),
                    "family": fam, "product": prod, "extension": p.suffix.lower(),
                    "size_bytes": st.st_size,
                    "mtime": pd.Timestamp(st.st_mtime, unit="s").isoformat(),
                    "sha256": sha256(p, st.st_size),
                })
    df = pd.DataFrame(rows)

    # duplicate detection across trees
    real = df[~df.sha256.str.startswith(("SKIPPED", "ERR"))]
    dup = real.groupby("sha256").agg(n_copies=("tree", "size"),
                                     trees=("tree", lambda x: ";".join(sorted(set(x)))))
    df = df.merge(dup, on="sha256", how="left")
    df["is_duplicate"] = df.n_copies.fillna(1) > 1

    CFG.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(CFG.TABLES / "full_dataset_inventory.csv", index=False)
    df.to_parquet(CFG.TABLES / "full_dataset_inventory.parquet", index=False)

    gb = lambda b: b / 1e9
    print(f"total files: {len(df):,}   total size: {gb(df.size_bytes.sum()):.2f} GB\n")
    print("=== by tree ===")
    t = df.groupby("tree").agg(n_files=("size_bytes", "size"),
                               GB=("size_bytes", lambda x: round(gb(x.sum()), 3)))
    print(t.to_string())
    print("\n=== by family ===")
    fam = (df.groupby("family")
             .agg(n_files=("size_bytes", "size"),
                  GB=("size_bytes", lambda x: round(gb(x.sum()), 3)))
             .sort_values("GB", ascending=False))
    print(fam.to_string())
    print("\n=== SWOT products held ===")
    sw = df[df.family == "SWOT"]
    if len(sw):
        print(sw.groupby("product").agg(n=("size_bytes", "size"),
                                        GB=("size_bytes", lambda x: round(gb(x.sum()), 3))).to_string())
    print("\n=== cross-tree duplicates ===")
    d = df[df.is_duplicate & (df.n_copies > 1)]
    print(f"{len(d)} files with an identical copy elsewhere, "
          f"{gb(d.size_bytes.sum()):.3f} GB total (redundant ~{gb(d.size_bytes.sum())/2:.3f} GB)")
    if len(d):
        print(d.groupby("trees").size().to_string())

    print("\n=== disk ===")
    for mp, name in [("/", "WSL root"), ("/mnt/c", "C:"), ("/mnt/d", "D:"), ("/mnt/g", "G:")]:
        try:
            u = shutil.disk_usage(mp)
            print(f"  {name:9s} total {gb(u.total):7.1f} GB   free {gb(u.free):7.1f} GB")
        except Exception:
            print(f"  {name:9s} unavailable")


if __name__ == "__main__":
    main()
