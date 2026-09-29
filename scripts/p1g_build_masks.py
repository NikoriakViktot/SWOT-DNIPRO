#!/usr/bin/env python
"""P1 STEP 7 -- build water masks ONLY for the P1_targeted_download_manifest.csv
scenes (2026-09-11 operator correction, second round). Reuses
swot_dnipro.watermask (the same NDWI/MNDWI/SCL method as phase19_watermasks.py)
so the targeted masks are methodologically identical to the canonical cache --
just built for a small, deliberately chosen subset instead of the full 1323-date
sweep (that full rebuild is P2, explicitly NOT started here).

Reads from data/raw/sentinel_p1_targeted/ (new downloads) AND the existing
data/raw/sentinel/ symlink (already-cached scenes), writes to the SAME
canonical data/processed/water_masks/ location phase19_watermasks.py uses, so
p1c_sa2_consensus.py's existing mosaic_water_mask() picks everything up with
no changes.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import watermask as WM

#: Masks live on the bulk volume (see config.BULK_ROOT); the repo-disk path this
#: used to point at no longer exists. RAW_DIRS lists every place a SAFE zip has
#: actually been written, bulk volume first.
MASKS = CFG.BULK_ROOT / "data_swot" / "processed" / "water_masks"
MASKS.mkdir(parents=True, exist_ok=True)
RAW_DIRS = (CFG.BULK_ROOT / "data_swot" / "sentinel",
           CFG.BULK_ROOT / "s2_zone_fetch",
           CFG.ROOT / "data" / "raw" / "sentinel_p1_targeted",
           CFG.ROOT / "data" / "raw" / "sentinel")


def find_zip(date: str, tile: str) -> Path | None:
    tok = date.replace("-", "")
    for base in RAW_DIRS:
        if not base.exists():
            continue
        hits = sorted(base.glob(f"*{tok}T*_T{tile}_*.SAFE.zip"))
        hits = [h for h in hits if h.stat().st_size > 500_000_000]
        if hits:
            return hits[0]
    return None


def _geotiff(path: Path, wm: WM.WaterMask) -> None:
    """0 = observed land, 1 = observed water, 255 = not observed. The same
    three-valued contract as phase19_watermasks._geotiff -- see there for why
    the earlier 0/1-only write made cloud indistinguishable from dry bed."""
    import rasterio
    a = wm.transform
    valid = wm.valid if wm.valid is not None else ~np.isin(wm.scl, WM.SCL_REJECT)
    arr = np.where(valid, wm.mask.astype("uint8"), np.uint8(255)).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=wm.mask.shape[0],
                       width=wm.mask.shape[1], count=1, dtype="uint8",
                       crs=wm.crs, transform=a, compress="deflate", nodata=255) as dst:
        dst.write(arr, 1)
        dst.update_tags(1, sensing_time=wm.sensing_time, tile=wm.tile,
                        method="NDWI>0 & MNDWI>0 & SCL-permitted",
                        values="0=land 1=water 255=not_observed(SCL_REJECT)")


def _npz(path: Path, wm: WM.WaterMask, lab: np.ndarray) -> None:
    a = wm.transform
    np.savez_compressed(
        path, mask=wm.mask, labels=lab.astype("i4"),
        valid=wm.valid if wm.valid is not None else ~np.isin(wm.scl, WM.SCL_REJECT),
        affine=np.array([a.a, a.b, a.c, a.d, a.e, a.f], float),
        crs=str(wm.crs), tile=wm.tile, sensing=wm.sensing_time,
        ndwi=wm.ndwi.astype("f2"), mndwi=wm.mndwi.astype("f2"), scl=wm.scl.astype("i1"))


def main() -> None:
    man = pd.read_csv(CFG.TABLES / "P1_targeted_download_manifest.csv")
    pairs = man[["date", "tile_id"]].drop_duplicates()
    print(f"building masks for {len(pairs)} (date, tile) pairs")

    rows = []
    for i, r in enumerate(pairs.itertuples(), 1):
        zp = find_zip(r.date, r.tile_id)
        if zp is None:
            print(f"  [{i}/{len(pairs)}] {r.date} {r.tile_id}: NOT ON DISK, skipping "
                  f"(download may have failed -- check P1_targeted_fetch_log.csv)")
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": "NO_RAW_FILE"})
            continue
        stem = zp.name.replace(".zip", "")
        npz_path = MASKS / f"{stem}.npz"
        tif_path = MASKS / f"{stem}_water.tif"
        if npz_path.exists() and tif_path.exists():
            print(f"  [{i}/{len(pairs)}] {r.date} {r.tile_id}: mask already cached ({stem})")
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": "ALREADY_MASKED",
                        "scene": stem})
            continue
        try:
            wm = WM.build_mask(zp, 0.0, 0.0)
            lab, nlab = WM.label_water_bodies(wm)
        except Exception as exc:
            print(f"  [{i}/{len(pairs)}] {r.date} {r.tile_id}: FAILED building mask: "
                  f"{type(exc).__name__}: {exc}")
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": f"FAILED:{exc}"})
            continue
        _npz(npz_path, wm, lab)
        _geotiff(tif_path, wm)
        px_m = abs(wm.transform.a)
        water_km2 = float(wm.mask.sum() * px_m**2 / 1e6)
        print(f"  [{i}/{len(pairs)}] {r.date} {r.tile_id}: water={water_km2:.1f} km2, "
              f"{nlab} components -> {stem}")
        rows.append({"date": r.date, "tile_id": r.tile_id, "status": "OK", "scene": stem,
                    "water_km2": water_km2, "n_components": int(nlab)})

    out = pd.DataFrame(rows)
    out_path = CFG.ROOT / "data" / "catalog" / "P1_targeted_masks_log.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print(out.status.value_counts().to_string() if len(out) else "nothing processed")


if __name__ == "__main__":
    main()
