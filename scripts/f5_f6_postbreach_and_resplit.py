#!/usr/bin/env python
"""F5 + F6 -- post-breach persistence for the upper AOI, pre/post
difference, and re-evaluation of the KAKHOVKA_UPPER_BACKWATER_CANDIDATE
(12.3 km2) into BACKWATER_CONFIRMED / ACTIVE_CHANNEL_ONLY / STAGE_DEPENDENT
/ AMBIGUOUS.

Outputs
-------
data/processed/study_domain/upper_postbreach_water_frequency.tif
outputs/figures/U_pre_post_upper_transition.png
outputs/tables/upper_backwater_candidate_resplit.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio import features
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

MASKS = CFG.ROOT / "data" / "processed" / "water_masks"
STUDY_DIR = CFG.ROOT / "data" / "processed" / "study_domain"
_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def mosaic_for_date(date_token: str, tiles=("36TXT", "36UXU")):
    files = []
    for t in tiles:
        files += sorted(MASKS.glob(f"*{date_token}T*_T{t}_*_water.tif"))
    if not files:
        return None, None
    srcs = [rasterio.open(f) for f in files]
    arr, transform = merge(srcs, nodata=255, resampling=Resampling.nearest)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    return arr[0], transform, crs


def main() -> None:
    sel = pd.read_csv(CFG.TABLES / "f3_selected_dates.csv")
    with rasterio.open(STUDY_DIR / "upper_prebreach_water_frequency.tif") as ds:
        pre_freq = ds.read(1); dst_transform = ds.transform; dst_crs = ds.crs
        H, W = pre_freq.shape
        pre_freq = np.where(pre_freq < 0, np.nan, pre_freq)

    water_count = np.zeros((H, W), dtype=np.int16)
    obs_count = np.zeros((H, W), dtype=np.int16)
    n_dates_used = 0
    for d in sel.date:
        tok = d.replace("-", "")
        arr, transform, crs = mosaic_for_date(tok)
        if arr is None:
            print(f"  {d}: no mask (raw missing), skipped")
            continue
        dst_water = np.zeros((H, W), dtype="uint8")
        reproject(source=(arr == 1).astype("uint8"), destination=dst_water,
                 src_transform=transform, src_crs=crs,
                 dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.max)
        dst_obs = np.zeros((H, W), dtype="uint8")
        reproject(source=(arr != 255).astype("uint8"), destination=dst_obs,
                 src_transform=transform, src_crs=crs,
                 dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.max)
        water_count += dst_water * dst_obs
        obs_count += dst_obs
        n_dates_used += 1
    print(f"post-breach dates actually mosaicked: {n_dates_used}")

    with np.errstate(divide="ignore", invalid="ignore"):
        post_freq = np.where(obs_count > 0, water_count / np.maximum(obs_count, 1), np.nan)

    out_tif = STUDY_DIR / "upper_postbreach_water_frequency.tif"
    with rasterio.open(out_tif, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs=dst_crs, transform=dst_transform,
                       nodata=-1, compress="deflate") as dst:
        dst.write(np.where(np.isnan(post_freq), -1, post_freq).astype("float32"), 1)
    print(f"-> {out_tif}")

    delta = post_freq - pre_freq  # NaN where either side lacks observation

    # ---- F6: re-split the backwater candidate ------------------------------
    hyd = gpd.read_file(STUDY_DIR / "hydraulic_longitudinal_domain.gpkg")
    cand = hyd[hyd.zone_name == "KAKHOVKA_UPPER_BACKWATER_CANDIDATE"].geometry.iloc[0]
    cand_mask = features.geometry_mask([cand], out_shape=(H, W), transform=dst_transform,
                                       invert=True)

    both_obs = cand_mask & ~np.isnan(pre_freq) & ~np.isnan(post_freq)
    pre_only = cand_mask & ~np.isnan(pre_freq) & np.isnan(post_freq)
    neither = cand_mask & np.isnan(pre_freq) & np.isnan(post_freq)

    persistent_both = both_obs & (pre_freq >= 0.7) & (post_freq >= 0.7)
    active_channel_only = both_obs & (pre_freq < 0.3) & (post_freq >= 0.7)  # new post-breach channel, not pre-breach backwater
    stage_dependent = both_obs & (~persistent_both) & (~active_channel_only) & \
                      ((pre_freq >= 0.2) | (post_freq >= 0.2))
    low_conf = both_obs & (pre_freq < 0.2) & (post_freq < 0.2)

    def km2(mask):
        return float(mask.sum()) * abs(dst_transform.a) * abs(dst_transform.e) / 1e6

    rows = [
        {"class": "BACKWATER_CONFIRMED", "area_km2": km2(persistent_both),
         "definition": "persistent (>=0.7) both pre- and post-breach -- genuine continuously "
                       "flooded backwater, not created/destroyed by the breach"},
        {"class": "ACTIVE_CHANNEL_ONLY", "area_km2": km2(active_channel_only),
         "definition": "rare/absent pre-breach (<0.3) but persistent post-breach (>=0.7) -- "
                       "post-breach active channel migration/widening, not pre-breach reservoir water"},
        {"class": "STAGE_DEPENDENT", "area_km2": km2(stage_dependent),
         "definition": "intermediate/variable frequency on at least one side -- shoreline-like, "
                       "water-level dependent"},
        {"class": "AMBIGUOUS_LOW_SIGNAL", "area_km2": km2(low_conf),
         "definition": "low frequency both sides -- weak/noisy signal, not classified further"},
        {"class": "INSUFFICIENT_OBSERVATION_PREBREACH_ONLY", "area_km2": km2(pre_only),
         "definition": "has pre-breach data but no post-breach observation reached this pixel "
                       "(2025-07-06 T36TXT gap, or edge-of-tile effects)"},
        {"class": "INSUFFICIENT_OBSERVATION_NEITHER", "area_km2": km2(neither),
         "definition": "no observation on either side"},
    ]
    out_tbl = pd.DataFrame(rows)
    out_csv = CFG.TABLES / "upper_backwater_candidate_resplit.csv"
    out_tbl.to_csv(out_csv, index=False)
    total_cand_km2 = km2(cand_mask)
    print(f"\ncandidate zone total: {total_cand_km2:.2f} km2")
    print(out_tbl.to_string(index=False))
    print(f"-> {out_csv}")

    # ---- figure -------------------------------------------------------------
    minx, maxy = dst_transform * (0, 0)
    maxx, miny = dst_transform * (W, H)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for ax, arr, title, cmap in ((axes[0], pre_freq, "pre-breach (18 dates)", "Blues"),
                                 (axes[1], post_freq, f"post-breach ({n_dates_used} dates)", "Blues"),
                                 (axes[2], delta, "delta (post - pre)", "RdBu_r")):
        vmin, vmax = (0, 1) if cmap == "Blues" else (-1, 1)
        im = ax.imshow(arr, extent=(minx, maxx, miny, maxy), origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8)
        gg = cand.geoms if hasattr(cand, "geoms") else [cand]
        for g in gg:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="black", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    fig.suptitle("U_pre_post_upper_transition -- black outline = "
                "KAKHOVKA_UPPER_BACKWATER_CANDIDATE (12.3 km2, pre-resplit)", fontsize=11)
    fig.tight_layout()
    out_fig = CFG.FIG / "U_pre_post_upper_transition.png"
    fig.savefig(out_fig, dpi=150)
    print(f"-> {out_fig}")


if __name__ == "__main__":
    main()
