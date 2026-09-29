#!/usr/bin/env python
"""Phase 19 step 6 — build and cache a Sentinel-2 water mask for every selected scene.

For each verified L2A scene:
  * NDWI  = (B03 - B08)/(B03 + B08)          green / NIR
  * MNDWI = (B03 - B11)/(B03 + B11)          green / SWIR
  * water = NDWI>t AND MNDWI>t AND SCL not in {cloud, cirrus, shadow, snow, sat, nodata}
            (SCL==water is also trusted with a relaxed index)
  * connected components (8-connectivity) -> hydraulic connectivity

Outputs
-------
data/processed/water_masks/<scene>.npz         mask(bool 20 m) + labels + affine + crs
data/processed/water_masks/<scene>_water.tif   single-band GeoTIFF of the mask
outputs/tables/water_mask_summary.csv          one row per scene
outputs/tables/water_mask_sensitivity.csv      threshold sweep (step 6: "test sensitivity")

Nothing here reads ICESat-2 or the +3.31 cm/km benchmark — masks are built blind.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import watermask as WM

#: Scenes and masks live on the bulk volume (config.BULK_ROOT). These two used
#: to point at the repo disk; the data had moved to F and the repo paths no
#: longer existed, so a re-run would have re-fetched 68 GB and rebuilt every
#: mask onto the wrong disk.
RAW = CFG.BULK_ROOT / "data_swot" / "sentinel"
MASKS = CFG.BULK_ROOT / "data_swot" / "processed" / "water_masks"
MASKS.mkdir(parents=True, exist_ok=True)

#: threshold pairs for the sensitivity sweep; the middle one is the operating point
SWEEP = [(-0.10, -0.10), (-0.05, -0.05), (0.0, 0.0), (0.05, 0.05), (0.10, 0.10),
         (0.0, -0.10), (0.0, 0.10)]
OPERATING = (0.0, 0.0)


def _save_npz(path: Path, wm: WM.WaterMask, lab: np.ndarray) -> None:
    a = wm.transform
    np.savez_compressed(
        path,
        mask=wm.mask, labels=lab.astype("i4"),
        valid=wm.valid if wm.valid is not None else ~np.isin(wm.scl, WM.SCL_REJECT),
        affine=np.array([a.a, a.b, a.c, a.d, a.e, a.f], float),
        crs=str(wm.crs), tile=wm.tile, sensing=wm.sensing_time,
        ndwi=wm.ndwi.astype("f2"), mndwi=wm.mndwi.astype("f2"), scl=wm.scl.astype("i1"))


def _geotiff(path: Path, wm: WM.WaterMask) -> None:
    """Three-valued mask: 0 = observed land, 1 = observed water, 255 = NOT OBSERVED.

    ``nodata=255`` was declared here from the start but never written: the bool
    mask went out as 0/1, so cloud, shadow and no-data became 0 = land. Every
    downstream coverage figure computed as ``mos != 255`` was therefore ~1.0
    regardless of cloud. Writing 255 where ``valid`` is False is what makes that
    coverage figure mean what it says."""
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


def main() -> None:
    inv = pd.read_csv(CFG.TABLES / "sentinel_scene_inventory.csv")
    inv = inv[(inv.get("valid_zip", False)) & (inv.get("all_bands_ok", False))].copy()
    if inv.empty:
        raise SystemExit("no verified scenes in sentinel_scene_inventory.csv — run fetch_sentinel.py")
    inv = inv.drop_duplicates("name")
    print(f"verified scenes to mask: {len(inv)}")

    # incremental: keep rows for scenes whose mask is already cached AND already
    # summarised, rebuild only what is new (a re-run after a targeted fetch)
    def _old(path):
        try:
            d = pd.read_csv(CFG.TABLES / path)
            return {n: g for n, g in d.groupby("name")}
        except FileNotFoundError:
            return {}
    old_summ, old_sens = _old("water_mask_summary.csv"), _old("water_mask_sensitivity.csv")

    summ, sens = [], []
    for i, r in enumerate(inv.itertuples(), 1):
        zp = RAW / f"{r.name}.zip"
        if not zp.exists():
            print(f"  [{i:2d}] MISSING {zp.name}"); continue
        npz = MASKS / f"{r.name}.npz"
        if (npz.exists() and (MASKS / f"{r.name}_water.tif").exists()
                and r.name in old_summ and r.name in old_sens):
            summ.append(old_summ[r.name].iloc[0].to_dict())
            sens.extend(old_sens[r.name].to_dict("records"))
            print(f"  [{i:2d}/{len(inv)}] CACHED {r.name}")
            continue
        wm = WM.build_mask(zp, *OPERATING)
        lab, nlab = WM.label_water_bodies(wm)
        _save_npz(npz, wm, lab)
        _geotiff(MASKS / f"{r.name}_water.tif", wm)

        px_m = abs(wm.transform.a)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        largest = int(sizes.max()) if sizes.size > 1 else 0
        valid = ~np.isin(wm.scl, WM.SCL_REJECT)
        summ.append({
            "name": r.name, "tile": wm.tile, "sensing_time": wm.sensing_time,
            "target_date": getattr(r, "target_date", ""), "regime": getattr(r, "regime", ""),
            "cloud_cover_meta": getattr(r, "cloud_cover", np.nan),
            "px_m": px_m, "grid": f"{wm.mask.shape[1]}x{wm.mask.shape[0]}",
            "valid_frac": float(valid.mean()),
            "water_px": int(wm.mask.sum()), "water_km2": float(wm.mask.sum() * px_m**2 / 1e6),
            "n_components": int(nlab),
            "largest_component_px": largest,
            "largest_component_km2": float(largest * px_m**2 / 1e6),
        })
        for tn, tm in SWEEP:
            w2 = WM.build_mask(zp, tn, tm)
            sens.append({"name": r.name, "tile": wm.tile, "ndwi_thr": tn, "mndwi_thr": tm,
                         "water_px": int(w2.mask.sum()),
                         "water_km2": float(w2.mask.sum() * px_m**2 / 1e6)})
        s0 = [x for x in sens if x["name"] == r.name and
              x["ndwi_thr"] == 0.0 and x["mndwi_thr"] == 0.0][0]
        rng = [x["water_km2"] for x in sens if x["name"] == r.name]
        print(f"  [{i:2d}/{len(inv)}] {wm.tile} {wm.sensing_time}  "
              f"water {s0['water_km2']:7.1f} km2  comps {nlab:5d}  "
              f"sweep range {min(rng):.0f}-{max(rng):.0f} km2")

    sm = pd.DataFrame(summ)
    sm.to_csv(CFG.TABLES / "water_mask_summary.csv", index=False)
    sn = pd.DataFrame(sens)
    sn.to_csv(CFG.TABLES / "water_mask_sensitivity.csv", index=False)

    # sensitivity headline: how much does total water area move with the threshold?
    piv = sn.pivot_table(index="name", columns=["ndwi_thr", "mndwi_thr"], values="water_km2")
    base = piv[(0.0, 0.0)]
    rel = piv.sub(base, axis=0).div(base, axis=0).abs()
    print(f"\nwater-area sensitivity to threshold (median |Δ| vs NDWI=MNDWI=0):")
    for col in piv.columns:
        print(f"  NDWI>{col[0]:+.2f} MNDWI>{col[1]:+.2f} : "
              f"median {rel[col].median()*100:5.1f}%   max {rel[col].max()*100:5.1f}%")
    print(f"\n-> {CFG.TABLES/'water_mask_summary.csv'}")
    print(f"-> {CFG.TABLES/'water_mask_sensitivity.csv'}")
    print(f"-> {MASKS}/  ({len(list(MASKS.glob('*.npz')))} npz, "
          f"{len(list(MASKS.glob('*_water.tif')))} tif)")


if __name__ == "__main__":
    main()
