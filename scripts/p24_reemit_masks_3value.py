#!/usr/bin/env python
"""P24 -- re-emit every cached Sentinel-2 water mask as a THREE-valued GeoTIFF.

The problem
-----------
`phase19_watermasks.py` and `p1g_build_masks.py` declared ``nodata=255`` on
every `<scene>_water.tif` but wrote the boolean mask as 0/1. Cloud, cloud
shadow, snow and no-data (``watermask.SCL_REJECT``) were folded into
``water=False`` and left the file as 0 -- arithmetically identical to dry bed.
`phase20_water_objects.py` then measured coverage as ``mos != 255``, which is
~1.0 for a fully clouded tile. Two production dates show the symptom directly:
2023-05-18 and 2023-08-06 report `footprint_observed_fraction` = 0.50 with zero
water bodies -- "a full reservoir with no water".

This is the project's own rule, violated: NO_DATA must never become a dry vote
(feedback-not-observed-is-not-dry).

Why a re-emit, not a rebuild
----------------------------
The 162 cached `.npz` sidecars all carry `scl`, so the observed/unobserved
distinction was never lost -- only the GeoTIFF dropped it. Re-deriving
``valid = ~isin(scl, SCL_REJECT)`` from the npz and rewriting the tif costs
seconds per scene and touches no Sentinel-2 zip. Rebuilding from the ~89 GB of
SAFE archives would give the identical result at ~100x the cost.

The water RULE is unchanged. Only the third value is added.

What is written
---------------
* `<scene>_water.tif`  uint8, ``0 = observed land, 1 = observed water,
  255 = not observed``; identical CRS/transform/tags as before plus a
  ``values`` tag stating the contract.
* `<scene>.npz`  the same file with a ``valid`` array appended (all other
  arrays byte-identical).
* the previous two-valued tif is moved, not deleted, to
  ``<MASKS>/_legacy_2value/`` so the change is reversible and auditable.
* `outputs/tables/p24_mask_reemit_manifest.csv` -- one row per scene with
  the fraction of pixels set to 255 and a check against the ``valid_frac``
  already recorded in `water_mask_summary.csv` (computed from scl at
  `phase19_watermasks.py:104` -- the same quantity, so they must agree).

Usage
-----
python scripts/p24_reemit_masks_3value.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from swot_dnipro import config as CFG
from swot_dnipro import watermask as WM

MASKS = CFG.BULK_ROOT / "data_swot" / "processed" / "water_masks"
LEGACY = MASKS / "_legacy_2value"
NODATA = np.uint8(255)


def reemit_one(npz_path: Path, dry_run: bool) -> dict:
    tif_path = npz_path.with_name(npz_path.stem + "_water.tif")
    with np.load(npz_path, allow_pickle=True) as z:
        mask = z["mask"].astype(bool)
        scl = z["scl"].astype("i2")
        a = z["affine"]
        crs = str(z["crs"])
        tile = str(z["tile"]) if "tile" in z.files else ""
        sensing = str(z["sensing"]) if "sensing" in z.files else ""
        keep = {k: z[k] for k in z.files}

    valid = ~np.isin(scl, WM.SCL_REJECT)
    arr = np.where(valid, mask.astype("uint8"), NODATA).astype("uint8")
    frac_nodata = float((~valid).mean())
    frac_water_obs = float(mask[valid].mean()) if valid.any() else np.nan

    # what the previous file said, for the record
    old_unique, old_nodata = None, None
    if tif_path.exists():
        with rasterio.open(tif_path) as src:
            old_nodata = src.nodata
            old_unique = np.unique(src.read(1)).tolist()

    row = dict(scene=npz_path.stem, tile=tile, sensing_time=sensing,
               n_px=int(mask.size), frac_not_observed=frac_nodata,
               frac_water_of_observed=frac_water_obs,
               old_tif_unique=str(old_unique), old_tif_nodata=old_nodata,
               status="DRY_RUN" if dry_run else "")
    if dry_run:
        return row

    # Move the ORIGINAL two-valued tif aside exactly once. On a re-run the file
    # in the working directory is already the re-emitted one; moving it would
    # overwrite the legacy original with its own replacement.
    if tif_path.exists() and not (LEGACY / tif_path.name).exists():
        LEGACY.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tif_path), str(LEGACY / tif_path.name))

    transform = Affine(*a[:6])
    with rasterio.open(tif_path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="uint8", crs=crs,
                       transform=transform, compress="deflate", nodata=255) as dst:
        dst.write(arr, 1)
        dst.update_tags(1, sensing_time=sensing, tile=tile,
                        method="NDWI>0 & MNDWI>0 & SCL-permitted",
                        values="0=land 1=water 255=not_observed(SCL_REJECT)",
                        reemitted_by="p24_reemit_masks_3value.py 2026-09-16")

    keep["valid"] = valid
    # np.savez APPENDS ".npz" to any name not already ending in it. The first
    # run used "<stem>.npz.tmp", got "<stem>.npz.tmp.npz" on disk, and the
    # replace() of the non-existent "<stem>.npz.tmp" raised on all 162 scenes
    # -- after the tif had already been re-emitted. The tifs were fine; the
    # manifest and the `valid` array in the npz were not written.
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **keep)
    tmp.replace(npz_path)
    row["status"] = "REEMITTED"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    npzs = sorted(MASKS.glob("*.npz"))
    if a.limit:
        npzs = npzs[:a.limit]
    print("=" * 78)
    print(f"P24 -- re-emit {len(npzs)} masks as 0/1/255   "
          f"({'DRY RUN' if a.dry_run else 'WRITE'})")
    print("=" * 78)
    print(f"  masks : {MASKS}")
    print(f"  legacy: {LEGACY}")

    t0 = time.time()
    rows = []
    for i, p in enumerate(npzs, 1):
        try:
            rows.append(reemit_one(p, a.dry_run))
        except Exception as ex:
            rows.append(dict(scene=p.stem, status=f"FAILED {type(ex).__name__}: {str(ex)[:80]}"))
        if i % 20 == 0 or i == len(npzs):
            print(f"  [{i}/{len(npzs)}] {time.time()-t0:.0f}s", flush=True)

    M = pd.DataFrame(rows)
    for c in ("frac_not_observed", "frac_water_of_observed"):
        if c not in M:                       # every row failed -> still write a manifest
            M[c] = np.nan

    # cross-check against the valid_frac phase19 already computed from scl
    summ_path = CFG.TABLES / "water_mask_summary.csv"
    if summ_path.exists():
        S = pd.read_csv(summ_path)[["name", "valid_frac"]]
        M = M.merge(S, left_on="scene", right_on="name", how="left").drop(columns="name")
        M["valid_frac_expected"] = M.valid_frac
        M["valid_frac_reemitted"] = 1.0 - M.frac_not_observed
        M["valid_frac_abs_diff"] = (M.valid_frac_reemitted - M.valid_frac_expected).abs()
        M = M.drop(columns="valid_frac")

    out = CFG.TABLES / "p24_mask_reemit_manifest.csv"
    M.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print(M.status.str.split(" ").str[0].value_counts().to_string())
    ok = M[M.status == "REEMITTED"] if not a.dry_run else M
    if len(ok):
        print(f"\n  not-observed fraction: median {ok.frac_not_observed.median():.3f}, "
              f"max {ok.frac_not_observed.max():.3f}")
        if "valid_frac_abs_diff" in ok:
            chk = ok.valid_frac_abs_diff.dropna()
            if len(chk):
                print(f"  agreement with water_mask_summary.valid_frac: n={len(chk)}, "
                      f"max |diff| {chk.max():.4f}  "
                      f"{'OK' if chk.max() < 1e-3 else 'MISMATCH -- investigate'}")
            else:
                print("  agreement check: no scene in this run has a water_mask_summary "
                      "row (the p1/f4-targeted scenes are not in the phase19 summary)")
            print(f"  scenes without a summary row: {int(ok.valid_frac_expected.isna().sum())} "
                  f"of {len(ok)} (expected: 162 npz vs 89 summary rows)")
    print(f"\n-> {out}")
    if a.dry_run:
        print("\nDRY RUN: nothing written.")


if __name__ == "__main__":
    main()
