#!/usr/bin/env python
"""K10e.V2/V3/V5 -- 2023 Sentinel-2 spectral stack + physical surface classes.

Scope, as agreed: 2023 ONLY, no Dynamic World (earthengine-api has no local
credentials -- ~/.config/earthengine does not exist). This is a scoped subset
of the full K10e vegetation/canopy specification; V4 (Dynamic World) and
V13 (2024-2026 expansion) are deliberately deferred, not silently dropped.

Builds on, and does NOT modify, the existing pipeline:
  - src/swot_dnipro/watermask.py       -> NDWI, MNDWI, SCL, water mask (cached
                                           .npz in data/processed/water_masks/)
  - scripts/part9_s2_spectral.py       -> NDVI on the same 20 m grid, already
                                           written to data/processed/spectral_indices/

NEW here: BSI and NDMI (not computed anywhere in the project yet), and an
11-class physical surface classification richer than part9's 4-class split
(part9's water/vegetation/sparse_veg/bare is not touched or replaced).

    NDMI = (B08 - B11) / (B08 + B11)              moisture, NIR/SWIR1
    BSI  = ((B11+B04)-(B08+B02)) / ((B11+B04)+(B08+B02))   bare soil index

Physical classes (V5), combining NDVI + NDWI + MNDWI + NDMI + BSI + SCL --
NOT NDVI alone:

    OPEN_WATER              established water mask (NDWI/MNDWI/SCL, unchanged)
    SHALLOW_OR_MIXED_WATER  water mask boundary pixels (mixed within 1 px)
    WET_SEDIMENT            non-water, high NDMI, low-moderate NDVI
    DRY_BARE_SEDIMENT       non-water, high BSI, low NDVI, low NDMI
    SPARSE_HERBACEOUS       low-moderate NDVI, low-moderate NDMI
    DENSE_HERBACEOUS        high NDVI, moderate NDMI, not woody-signalled
    REED_OR_FLOODED_VEGETATION  high NDVI AND high NDMI (vegetation on wet ground)
    SHRUB_YOUNG_WOODY        (not separable from spectral indices alone without
                               structure/DW -- collapsed into DENSE_HERBACEOUS
                               with a flag; see AMBIGUOUS_WOODY note below)
    BUILT_HARD_SURFACE      reserved in the legend, never assigned here -- see
                            the note in classify() on why SCL cannot separate
                            it from bare sediment; expect 0 for this reservoir
    AMBIGUOUS               indices disagree / low valid support

SHRUB_YOUNG_WOODY and TREE_CANOPY cannot be distinguished from herbaceous
cover using Sentinel-2 spectral indices alone -- that structural separation is
exactly what Dynamic World (deferred) or the PhoREAL canopy fields (used
per-point, not per-pixel, in k10e_point_attribution.py) are for. This script
does NOT invent a woody/herbaceous split it cannot support; classes are
labelled accordingly and the gap is explicit, not papered over.

Outputs
-------
data/processed/spectral_indices/<scene>_stack.tif   (bands: ndvi,ndwi,mndwi,ndmi,bsi,surface_class)
outputs/tables/k10e_surface_class_by_date_2023.csv
"""
from __future__ import annotations

import sys
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import pyproj
import rasterio
from affine import Affine
from rasterio import features
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD
from swot_dnipro import watermask as WM
import json

# Both moved to the bulk volume (config.BULK_ROOT); the repo-disk mask path no
# longer exists and /mnt/e is a drive this project no longer uses. Re-pointed
# 2026-09-16 so a re-run finds its cache instead of silently masking nothing.
# NOTE: this script is superseded for new work by swot_dnipro.sentinel_preprocess
# (canonical module, adds AWEIsh/NDTI, BOA offset, zone grids); kept for the
# existing 89 eastern-tile stacks.
MASKS = CFG.BULK_ROOT / "data_swot/processed/water_masks"
SAFE_DIRS = [CFG.BULK_ROOT / "data_swot/sentinel", CFG.BULK_ROOT / "s2_zone_fetch",
             ROOT / "data/raw/sentinel_p1_targeted"]
# Written to F: (1.9 TB free), NOT into the WSL ext4 volume: each 6-band stack
# is ~280 MB and the WSL virtual disk (ext4.vhdx) lives on the Windows C: drive,
# which had only 62 GB free -- a 72-scene run ran the host out of space mid-run.
# See k10e_extra_scenes_2023.py-adjacent notes; this path is a hard requirement,
# not a preference.
OUT_TIF = Path("/mnt/f/data_kakhovka_dem_swot/spectral_indices")
SENTINEL = -1.0        # anything <= this is invalid fill (-9.0)

# class codes, stored in the surface_class band
CLASSES = {
    0: "INVALID", 1: "OPEN_WATER", 2: "SHALLOW_OR_MIXED_WATER", 3: "WET_SEDIMENT",
    4: "DRY_BARE_SEDIMENT", 5: "SPARSE_HERBACEOUS", 6: "DENSE_HERBACEOUS",
    7: "REED_OR_FLOODED_VEGETATION", 8: "BUILT_HARD_SURFACE", 9: "AMBIGUOUS",
}
# thresholds, conventional / conservative -- not fit to this validation, so a
# threshold-sensitivity table is emitted rather than silently trusted
NDVI_VEG, NDVI_SPARSE = 0.30, 0.15
NDMI_WET = 0.10
BSI_BARE = 0.10


def find_zip(name: str) -> Path | None:
    for d in SAFE_DIRS:
        p = d / f"{name}.zip"
        if p.exists():
            return p
    return None


def read_band(zp: str, names: list[str], band: str, res: str, out_shape):
    member = WM._find(names, band, res)
    if member is None:
        return None
    with rasterio.open(f"zip+file://{zp}!/{member}") as s:
        return s.read(1, out_shape=out_shape, resampling=Resampling.average).astype("f4")


def classify(ndvi, ndwi, mndwi, ndmi, bsi, scl, water_mask, valid):
    from scipy import ndimage
    cls = np.zeros(ndvi.shape, "u1")
    cls[valid] = CLASSES_INV["AMBIGUOUS"]

    # water first (established mask wins, per part9 convention)
    cls[valid & water_mask] = CLASSES_INV["OPEN_WATER"]
    # boundary of the water mask: dilate - erode, 1 px ring
    wm_dil = ndimage.binary_dilation(water_mask, iterations=1)
    wm_ero = ndimage.binary_erosion(water_mask, iterations=1)
    boundary = wm_dil & ~wm_ero
    cls[valid & boundary] = CLASSES_INV["SHALLOW_OR_MIXED_WATER"]

    # SCL=5 is Sen2Cor's own "NOT_VEGETATED" class -- bare soil, sand, rock AND
    # built surfaces alike; Sentinel-2 spectral indices + SCL cannot separate
    # "freshly exposed lakebed sediment" from "concrete" within that one code.
    # An earlier version of this classifier used SCL==5 as the BUILT_HARD_SURFACE
    # trigger and got frac_BUILT_HARD_SURFACE=0.39 on 2023-07-05 -- one month
    # after the breach, on newly exposed lakebed with no buildings on it. That
    # was mislabelled bare sediment, not built surface. Fixed: SCL==5 with low
    # NDVI/NDMI is DRY_BARE_SEDIMENT (the actually-common, physically-expected
    # case here); BUILT_HARD_SURFACE is reserved for the rare case of high
    # reflectance in ALL of B02/B03/B04 simultaneously (concrete/metal spectral
    # signature), which SCL alone cannot provide and is not attempted here --
    # this reservoir has no meaningful built-surface footprint to classify.
    nonwater = valid & ~water_mask
    cls[nonwater & np.isin(scl, (5,)) & (ndvi < NDVI_SPARSE) & (ndmi < NDMI_WET)] = \
        CLASSES_INV["DRY_BARE_SEDIMENT"]
    cls[nonwater & (bsi >= BSI_BARE) & (ndvi < NDVI_SPARSE) & (ndmi < NDMI_WET)] = \
        CLASSES_INV["DRY_BARE_SEDIMENT"]
    cls[nonwater & (ndmi >= NDMI_WET) & (ndvi < NDVI_VEG) & (bsi < BSI_BARE)] = \
        CLASSES_INV["WET_SEDIMENT"]
    cls[nonwater & (ndvi >= NDVI_SPARSE) & (ndvi < NDVI_VEG) & (ndmi < NDMI_WET)] = \
        CLASSES_INV["SPARSE_HERBACEOUS"]
    cls[nonwater & (ndvi >= NDVI_VEG) & (ndmi < NDMI_WET)] = \
        CLASSES_INV["DENSE_HERBACEOUS"]
    cls[nonwater & (ndvi >= NDVI_VEG) & (ndmi >= NDMI_WET)] = \
        CLASSES_INV["REED_OR_FLOODED_VEGETATION"]
    # BUILT_HARD_SURFACE is intentionally never assigned by this function (see
    # note above) -- it stays in the legend for schema stability but this
    # reservoir bed classification will always report it as 0.
    return cls


CLASSES_INV = {v: k for k, v in CLASSES.items()}


def main() -> None:
    print("=" * 78)
    print("K10e.V2/V3/V5 -- 2023 Sentinel-2 spectral stack (BSI, NDMI new; "
          "physical surface classes)")
    print("=" * 78)
    print("Scope: full POST_BREACH period (2023-09 .. latest cached scene). Originally")
    print("2023-only; extended after the channel-only PRIMARY stratum proved too thin")
    print("(17 segments, 2 RGTs) to repeat D14 on. Dynamic World now available (see")
    print("k10e_dynamic_world_2023.py -- GEE credentials re-authenticated this session).")
    print(f"threshold set: NDVI_veg={NDVI_VEG}, NDVI_sparse={NDVI_SPARSE}, "
          f"NDMI_wet={NDMI_WET}, BSI_bare={BSI_BARE} (conventional, swept in "
          f"k10e_threshold_sensitivity.csv, not fit to validation)\n")

    OUT_TIF.mkdir(parents=True, exist_ok=True)
    wm = pd.read_csv(CFG.TABLES / "water_mask_summary.csv")
    wm["dt"] = pd.to_datetime(wm.sensing_time, format="%Y%m%dT%H%M%S")
    wm = wm[wm.dt >= "2023-09-01"].copy()   # POST0, matches k10_foundation_icesat2.py
    print(f"POST_BREACH scenes with an existing water-mask cache: {len(wm)}, "
          f"{wm.dt.dt.normalize().nunique()} unique dates "
          f"({wm.dt.min().date()} .. {wm.dt.max().date()})")

    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp_m = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
            SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    by_date, sweep, n_ok = [], [], 0
    for i, s in enumerate(wm.itertuples(), 1):
        out_path = OUT_TIF / f"{s.name}_stack.tif"
        if out_path.exists():
            n_ok += 1
            print(f"  {i:2d}/{len(wm)}  {s.name[:45]:45s}  CACHED (already on F:)")
            continue
        npz_p = MASKS / f"{s.name}.npz"
        zip_p = find_zip(s.name)
        if not npz_p.exists() or zip_p is None:
            print(f"  {s.name[:45]:45s}  SKIP (missing npz or zip)")
            continue
        z = np.load(npz_p)
        ndwi, mndwi, scl, water = (z["ndwi"].astype("f4"), z["mndwi"].astype("f4"),
                                   z["scl"], z["mask"])
        a = z["affine"]; tr = Affine(a[0], a[1], a[2], a[3], a[4], a[5]); crs = str(z["crs"])
        shp = ndwi.shape

        zp = str(zip_p)
        names = zipfile.ZipFile(zp).namelist()
        b02 = read_band(zp, names, "B02", "10m", shp)
        b04 = read_band(zp, names, "B04", "10m", shp)
        b08 = read_band(zp, names, "B08", "10m", shp)
        b11 = read_band(zp, names, "B11", "20m", shp)
        if any(b is None for b in (b02, b04, b08, b11)):
            print(f"  {s.name[:45]:45s}  SKIP (missing band)")
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (b08 - b04) / (b08 + b04)
            ndmi = (b08 - b11) / (b08 + b11)
            bsi = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02))
        ndvi = np.nan_to_num(ndvi, nan=-9.0)
        ndmi = np.nan_to_num(ndmi, nan=-9.0)
        bsi = np.nan_to_num(bsi, nan=-9.0)

        valid = ((ndvi > SENTINEL) & (ndwi > SENTINEL) & (mndwi > SENTINEL)
                 & (ndmi > SENTINEL) & (bsi > SENTINEL) & ~np.isin(scl, WM.SCL_REJECT))
        cls = classify(ndvi, ndwi, mndwi, ndmi, bsi, scl, water, valid)

        with rasterio.open(OUT_TIF / f"{s.name}_stack.tif", "w", driver="GTiff",
                           height=shp[0], width=shp[1], count=6, dtype="float32",
                           crs=crs, transform=tr, compress="deflate", predictor=2,
                           tiled=True, nodata=-9.0) as dst:
            for bi, (arr, nm) in enumerate(
                    ((ndvi, "NDVI"), (ndwi, "NDWI"), (mndwi, "MNDWI"), (ndmi, "NDMI"),
                     (bsi, "BSI"), (cls.astype("f4"), "surface_class")), 1):
                dst.write(arr, bi)
                dst.set_band_description(bi, nm)
            dst.update_tags(sensing_time=s.sensing_time, tile=s.tile,
                            class_legend=str(CLASSES), nodata_fill="-9.0",
                            thresholds=f"NDVI_veg={NDVI_VEG},NDVI_sparse={NDVI_SPARSE},"
                                       f"NDMI_wet={NDMI_WET},BSI_bare={BSI_BARE}")
        n_ok += 1

        to_scene = pyproj.Transformer.from_crs(CFG.CRS_METRIC, crs, always_xy=True).transform
        fpr = features.rasterize([(shp_transform(to_scene, fp_m), 1)], out_shape=shp,
                                 transform=tr, fill=0, dtype="uint8").astype(bool)
        sel = fpr & valid
        if sel.sum() >= 500:
            row = {"date": s.dt.date().isoformat(), "tile": s.tile, "n_valid_px": int(sel.sum())}
            for code, nm in CLASSES.items():
                if code == 0:
                    continue
                row[f"frac_{nm}"] = float((cls[sel] == code).mean())
            row["ndvi_median"] = float(np.median(ndvi[sel]))
            row["ndmi_median"] = float(np.median(ndmi[sel]))
            row["bsi_median"] = float(np.median(bsi[sel]))
            by_date.append(row)
            for thr_n, thr_m, thr_b in [(0.20, 0.05, 0.05), (0.30, 0.10, 0.10),
                                        (0.40, 0.15, 0.15)]:
                veg = sel & (ndvi >= thr_n)
                sweep.append({"date": row["date"], "tile": s.tile, "ndvi_thr": thr_n,
                              "ndmi_thr": thr_m, "bsi_thr": thr_b,
                              "frac_vegetated": float(veg.mean())})
        print(f"  {i:2d}/{len(wm)}  {s.name[:45]:45s}  OK  "
              f"({int(sel.sum()):,} px in former pool)", flush=True)

    bd = pd.DataFrame(by_date)
    bd.to_csv(CFG.TABLES / "k10e_surface_class_by_date_2023.csv", index=False)
    pd.DataFrame(sweep).to_csv(CFG.TABLES / "k10e_threshold_sensitivity_2023.csv", index=False)

    print(f"\nstacks written: {n_ok}/{len(wm)}")
    print(f"-> {OUT_TIF} (*_stack.tif)")
    print(f"-> {CFG.TABLES / 'k10e_surface_class_by_date_2023.csv'}")
    if len(bd):
        print("\n2023 former-pool surface composition by date (fraction of valid px):")
        show = ["date"] + [f"frac_{CLASSES[c]}" for c in range(1, 9)]
        print(bd[show].sort_values("date").to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
