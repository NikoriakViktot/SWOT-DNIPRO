#!/usr/bin/env python
"""PART 9 — Sentinel-2 spectral indices (NDWI, MNDWI, NDVI) and a surface classification.

NDWI and MNDWI are already computed and stored by phase19_watermasks.py inside the
per-scene .npz, on the 20 m grid (B03_10m averaged //2). NDVI is not, because
watermask.py never reads B04 -- so B04 and B08 are re-read here with the SAME
averaging, which puts NDVI on the identical grid (verified: 5490x5490 match).

    NDWI  = (B03 - B08) / (B03 + B08)      water, green/NIR      [from npz]
    MNDWI = (B03 - B11) / (B03 + B11)      water, green/SWIR     [from npz]
    NDVI  = (B08 - B04) / (B08 + B04)      vegetation, NIR/red   [computed here]

TWO TRAPS, both handled:
  * invalid pixels are stored as -9.0, NOT NaN -> every statistic gates on > -1
  * the water mask IS defined as NDWI>0 & MNDWI>0 & SCL, so "fraction of water
    pixels with MNDWI>0" is ~1 BY CONSTRUCTION and is not evidence of anything.
    The informative quantities are the MAGNITUDE inside water, the values
    OUTSIDE the mask, and the water-to-bed separation.

The established water mask is NOT redefined. Classification only splits the
non-water remainder by NDVI, using conventional breakpoints reported with a
sensitivity sweep.

Outputs
-------
data/processed/spectral_indices/<scene>_ndvi.tif
outputs/tables/s2_spectral_class_by_date.csv
outputs/tables/s2_index_stats_by_class.csv
outputs/tables/s2_ndvi_threshold_sensitivity.csv
"""
from __future__ import annotations

import json
import sys
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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

MASKS = ROOT / "data/processed/water_masks"
SAFE = Path("/mnt/e/data_swot/sentinel")
OUT_TIF = ROOT / "data/processed/spectral_indices"
BREACH = pd.Timestamp("2023-06-06")
POST0 = pd.Timestamp("2023-09-01")
NDVI_VEG, NDVI_SPARSE = 0.30, 0.15        # conventional breakpoints, swept below
SENTINEL = -1.0                            # anything <= this is invalid (-9.0 fill)


def read_ndvi(zip_path: Path):
    """NDVI on the same grid the water mask uses: B04/B08 at 10 m, averaged //2."""
    zp = str(zip_path)
    names = zipfile.ZipFile(zp).namelist()
    b04, b08 = WM._find(names, "B04", "10m"), WM._find(names, "B08", "10m")
    if not (b04 and b08):
        return None

    def rd(member):
        with rasterio.open(f"zip+file://{zp}!/{member}") as s:
            return s.read(1, out_shape=(s.height // 2, s.width // 2),
                          resampling=Resampling.average).astype("f4")

    r, n = rd(b04), rd(b08)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (n - r) / (n + r)
    return np.nan_to_num(v, nan=-9.0)


def main() -> None:
    OUT_TIF.mkdir(parents=True, exist_ok=True)
    wm = pd.read_csv(CFG.TABLES / "water_mask_summary.csv")
    wm["dt"] = pd.to_datetime(wm.sensing_time, format="%Y%m%dT%H%M%S")
    wm["date"] = wm.dt.dt.strftime("%Y-%m-%d")
    wm["period"] = np.where(wm.dt < BREACH, "PRE_BREACH",
                            np.where(wm.dt < POST0, "DRAWDOWN", "POST_BREACH"))

    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp_m = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
            SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    rows, stats, sweep = [], [], []
    for i, s in enumerate(wm.itertuples(), 1):
        npz_p = MASKS / f"{s.name}.npz"
        zip_p = SAFE / f"{s.name}.zip"
        if not npz_p.exists() or not zip_p.exists():
            continue
        z = np.load(npz_p)
        ndwi, mndwi = z["ndwi"].astype("f4"), z["mndwi"].astype("f4")
        scl, water = z["scl"], z["mask"]
        a = z["affine"]
        tr = Affine(a[0], a[1], a[2], a[3], a[4], a[5])
        crs = str(z["crs"])

        ndvi = read_ndvi(zip_p)
        if ndvi is None or ndvi.shape != ndwi.shape:
            print(f"  {s.name[:40]}: NDVI grid mismatch, skipped"); continue

        with rasterio.open(OUT_TIF / f"{s.name}_ndvi.tif", "w", driver="GTiff",
                           height=ndvi.shape[0], width=ndvi.shape[1], count=1,
                           dtype="float32", crs=crs, transform=tr,
                           compress="deflate", predictor=2, tiled=True,
                           nodata=-9.0) as dst:
            dst.write(ndvi, 1)
            dst.update_tags(1, index="NDVI=(B08-B04)/(B08+B04)", grid="B03_10m averaged //2",
                            sensing_time=s.sensing_time, tile=s.tile, nodata_fill="-9.0")

        # ---- footprint in this scene's own grid ---------------------------
        to_scene = pyproj.Transformer.from_crs(CFG.CRS_METRIC, crs, always_xy=True).transform
        fpr = features.rasterize([(shp_transform(to_scene, fp_m), 1)],
                                 out_shape=ndvi.shape, transform=tr,
                                 fill=0, dtype="uint8").astype(bool)

        valid = (ndvi > SENTINEL) & (ndwi > SENTINEL) & (mndwi > SENTINEL) \
            & ~np.isin(scl, WM.SCL_REJECT)
        # the established mask is used as-is; only the NON-water remainder is split
        cls = np.full(ndvi.shape, 0, "u1")          # 0 invalid
        cls[valid] = 4                              # bare
        cls[valid & (ndvi >= NDVI_SPARSE)] = 3      # sparse veg
        cls[valid & (ndvi >= NDVI_VEG)] = 2         # vegetation
        cls[valid & water] = 1                      # water (mask wins)

        names = {1: "water", 2: "vegetation", 3: "sparse_veg", 4: "bare"}
        for inside, zone in ((fpr, "former_pool"), (~fpr, "outside_pool")):
            sel = inside & valid
            n_sel = int(sel.sum())
            if n_sel < 500:
                continue
            row = {"date": s.date, "tile": s.tile, "period": s.period, "zone": zone,
                   "n_valid_px": n_sel, "valid_fraction": float((inside & valid).sum()
                                                                / max(int(inside.sum()), 1))}
            for k, nm in names.items():
                row[f"frac_{nm}"] = float((cls[sel] == k).mean())
            row["ndvi_median"] = float(np.median(ndvi[sel]))
            row["mndwi_median"] = float(np.median(mndwi[sel]))
            row["ndwi_median"] = float(np.median(ndwi[sel]))
            rows.append(row)

            for k, nm in names.items():
                m = sel & (cls == k)
                if m.sum() < 200:
                    continue
                stats.append({"date": s.date, "tile": s.tile, "period": s.period,
                              "zone": zone, "class": nm, "n_px": int(m.sum()),
                              **{f"{ix}_{q}": float(np.percentile(arr[m], p))
                                 for ix, arr in (("ndvi", ndvi), ("ndwi", ndwi),
                                                 ("mndwi", mndwi))
                                 for q, p in (("p10", 10), ("median", 50), ("p90", 90))}})

        # NDVI threshold sensitivity, inside the former pool only
        sel = fpr & valid & ~water
        if sel.sum() > 500:
            for thr in (0.20, 0.25, 0.30, 0.35, 0.40):
                sweep.append({"date": s.date, "tile": s.tile, "period": s.period,
                              "ndvi_threshold": thr,
                              "frac_vegetated_nonwater": float((ndvi[sel] >= thr).mean())})
        if i % 10 == 0:
            print(f"  {i}/{len(wm)} scenes", flush=True)

    by_date = pd.DataFrame(rows)
    by_date.to_csv(CFG.TABLES / "s2_spectral_class_by_date.csv", index=False)
    pd.DataFrame(stats).to_csv(CFG.TABLES / "s2_index_stats_by_class.csv", index=False)
    sw = pd.DataFrame(sweep)
    sw.to_csv(CFG.TABLES / "s2_ndvi_threshold_sensitivity.csv", index=False)

    print(f"\n-> {CFG.TABLES/'s2_spectral_class_by_date.csv'} ({len(by_date)} rows)")
    print(f"-> {OUT_TIF} ({len(list(OUT_TIF.glob('*_ndvi.tif')))} NDVI GeoTIFFs)\n")

    fp = by_date[by_date.zone == "former_pool"]
    print("=== former pool: surface composition by period (median over scenes) ===")
    print(fp.groupby("period")[["frac_water", "frac_vegetation", "frac_sparse_veg",
                                "frac_bare", "ndvi_median", "mndwi_median"]]
          .median().round(3).to_string())
    print("\n=== exposed bed (post-breach, former pool) vegetated fraction by year ===")
    post = fp[fp.period == "POST_BREACH"].copy()
    post["year"] = pd.to_datetime(post.date).dt.year
    print(post.groupby("year").agg(n_scenes=("date", "size"),
                                   veg=("frac_vegetation", "median"),
                                   sparse=("frac_sparse_veg", "median"),
                                   bare=("frac_bare", "median"),
                                   water=("frac_water", "median")).round(3).to_string())
    print("\n=== NDVI threshold sensitivity (post-breach, non-water, in pool) ===")
    sp = sw[sw.period == "POST_BREACH"].copy()
    sp["year"] = pd.to_datetime(sp.date).dt.year
    print(sp.pivot_table(index="year", columns="ndvi_threshold",
                         values="frac_vegetated_nonwater", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
