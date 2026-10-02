#!/usr/bin/env python
"""Clip the continent-wide SWOT RiverSP / LakeSP archive to Ukraine.

RiverSP and LakeSP ship one granule per continent-wide overpass, so the local
archive (402 GB LakeSP + 81 GB RiverSP) is overwhelmingly outside the study
area -- the first Unassigned granule inspected spans 54.95-71.38 N, i.e.
Scandinavia and the Arctic.

This writes a Ukraine-only copy and TOUCHES NOTHING IN data/raw. Deleting or
archiving the originals is a separate, deliberate step, taken only after the
manifest written here has been reviewed: a clip that silently dropped the
study area would otherwise be discovered after the evidence was gone.

Method
------
Two stages, because reading 8 452 shapefiles in full is the expensive part:

1. bounds-only rejection via ``pyogrio.read_info`` -- no geometry is loaded.
   A granule whose total bounds miss the domain bbox is skipped outright.
2. for survivors, a bbox-pushdown read followed by an exact ``intersects``
   against the domain geometry.

The domain comes from the registry (``ukraine_swot_clip_domain``), never from
a literal or a local file path -- see scripts/p22_build_ukraine_clip_domain.py
for why a country polygon is the wrong instrument and what this one is built
from instead.

Output format is shapefile, matching the input, so any existing ``*.shp`` glob
keeps working. The saving comes from dropping features, not from re-encoding.

Outputs
-------
<dest>/<product>/<year>/<month>/<granule>.shp   (only where features survive)
outputs/tables/p22_ukraine_clip_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd
import pyogrio

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

PRODUCTS = ("swot_l2_hr_lakesp_2.0", "swot_l2_hr_riversp_2.0")
SRC_ROOT = ROOT / "data" / "raw"


def granule_bytes(shp: Path) -> int:
    """A shapefile is 5 files; size means all of them."""
    return sum(p.stat().st_size for p in shp.parent.glob(shp.stem + ".*")
               if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(CFG.BULK_ROOT / "swot_ua"),
                    help="destination root (default: bulk volume, per the "
                         "project's storage rule)")
    ap.add_argument("--product", choices=PRODUCTS, default=None,
                    help="restrict to one product (default: both)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N granules -- use for a sizing run")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage-1 bounds screen only; writes nothing")
    a = ap.parse_args()

    dom = SD.load("ukraine_swot_clip_domain")
    bbox = SD.as_bbox("ukraine_swot_clip_domain")
    dest = Path(a.dest)
    products = [a.product] if a.product else list(PRODUCTS)

    print("=" * 78)
    print("P22 — clipping the SWOT archive to Ukraine")
    print("=" * 78)
    print(f"  domain : ukraine_swot_clip_domain (registry)")
    print(f"  bbox   : {[round(v, 3) for v in bbox]}")
    print(f"  dest   : {dest}")
    print(f"  mode   : {'DRY RUN (stage 1 only)' if a.dry_run else 'WRITE'}")

    shps = []
    for prod in products:
        found = sorted((SRC_ROOT / prod).rglob("*.shp"))
        print(f"  {prod:<26} {len(found):5d} granules")
        shps += [(prod, p) for p in found]
    if a.limit:
        shps = shps[:a.limit]
        print(f"  --limit {a.limit}: processing {len(shps)} granules")

    rows = []
    t0 = time.time()
    kept = skipped = empty = failed = 0
    for i, (prod, shp) in enumerate(shps, 1):
        nbytes = granule_bytes(shp)
        try:
            info = pyogrio.read_info(shp)
            tb = info.get("total_bounds")
            n_src = int(info.get("features", 0))
        except Exception as ex:
            failed += 1
            rows.append(dict(product=prod, granule=shp.stem, stage="INFO_FAILED",
                             n_src=-1, n_kept=-1, bytes_src=nbytes, bytes_kept=0,
                             error=f"{type(ex).__name__}: {str(ex)[:80]}"))
            continue

        # stage 1 -- bounds-only rejection, no geometry read
        if tb is None or tb[0] > bbox[2] or tb[2] < bbox[0] \
                or tb[1] > bbox[3] or tb[3] < bbox[1]:
            skipped += 1
            rows.append(dict(product=prod, granule=shp.stem, stage="BBOX_SKIP",
                             n_src=n_src, n_kept=0, bytes_src=nbytes,
                             bytes_kept=0, error=""))
            if i % 250 == 0:
                print(f"  [{i}/{len(shps)}] kept {kept} skip {skipped} "
                      f"empty {empty}  {time.time()-t0:.0f}s", flush=True)
            continue

        if a.dry_run:
            rows.append(dict(product=prod, granule=shp.stem, stage="WOULD_READ",
                             n_src=n_src, n_kept=-1, bytes_src=nbytes,
                             bytes_kept=0, error=""))
            continue

        # stage 2 -- bbox pushdown, then exact intersection
        try:
            g = gpd.read_file(shp, bbox=bbox)
            if len(g):
                g = g[g.intersects(dom)]
        except Exception as ex:
            failed += 1
            rows.append(dict(product=prod, granule=shp.stem, stage="READ_FAILED",
                             n_src=n_src, n_kept=-1, bytes_src=nbytes,
                             bytes_kept=0, error=f"{type(ex).__name__}: {str(ex)[:80]}"))
            continue

        if not len(g):
            empty += 1
            rows.append(dict(product=prod, granule=shp.stem, stage="NO_FEATURES",
                             n_src=n_src, n_kept=0, bytes_src=nbytes,
                             bytes_kept=0, error=""))
            continue

        rel = shp.relative_to(SRC_ROOT)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        g.to_file(out)
        kept += 1
        rows.append(dict(product=prod, granule=shp.stem, stage="CLIPPED",
                         n_src=n_src, n_kept=len(g), bytes_src=nbytes,
                         bytes_kept=granule_bytes(out), error=""))
        if i % 100 == 0:
            print(f"  [{i}/{len(shps)}] kept {kept} skip {skipped} empty {empty} "
                  f"{time.time()-t0:.0f}s", flush=True)

    M = pd.DataFrame(rows)
    out_csv = CFG.TABLES / "p22_ukraine_clip_manifest.csv"
    M.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print(M.stage.value_counts().to_string())
    src_gb = M.bytes_src.sum() / 1e9
    kept_gb = M.bytes_kept.sum() / 1e9
    print(f"\n  source   {src_gb:8.1f} GB over {len(M)} granules")
    if not a.dry_run:
        print(f"  clipped  {kept_gb:8.1f} GB  ({100*kept_gb/max(src_gb,1e-9):.2f} % of source)")
        print(f"  features {int(M.n_src[M.n_src>0].sum()):,} -> "
              f"{int(M.n_kept[M.n_kept>0].sum()):,}")
    print(f"  elapsed  {time.time()-t0:.0f}s")
    print(f"\n-> {out_csv}")
    print("\nNOTHING in data/raw was modified. Review the manifest before "
          "archiving or deleting the originals.")


if __name__ == "__main__":
    main()
