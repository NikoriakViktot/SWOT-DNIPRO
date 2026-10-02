#!/usr/bin/env python
"""P1C v2 -- true multi-date SA_2 consensus (2026-09-11 operator correction,
second round, Step 8). Supersedes p1c_sa2_consensus.py's 3-date pilot: now
covers every date in P1_targeted_download_manifest.csv whose masks have been
built by p1g_build_masks.py.

Step 11 (explicit): 2023-05-16 and 2023-05-19 are excluded from the IoU
statistics -- 1/4-tile partial observations, flagged
INVALID_FOR_FULL_DOMAIN_IOU, kept in the table for transparency but never
averaged into "how well does SA_2 match reality".

Outputs
-------
outputs/tables/sa2_multidate_consensus.csv   (REPLACES the 3-date pilot)
outputs/figures/P1_sa2_disagreement_map.png  (CONSISTENTLY WATER / VARIABLE
    SHORELINE / CONSISTENTLY OUTSIDE / SYSTEMATIC SA_2 OMISSION / SYSTEMATIC
    SA_2 COMMISSION)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio import features
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

MASKS = CFG.ROOT / "data" / "processed" / "water_masks"
INK, BLUE, RED, AMBER, GREEN, PURP = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#7d5ba6"
KNOWN_PARTIAL = {"2023-05-16": 1, "2023-05-19": 1}  # tiles actually cached before this round
MAIN_TILES = ("36TWS", "36TWT", "36TXT")
CHECK_DATE = "2023-06-05"
COARSE_RES = 100.0  # m, common comparison grid
MIN_DATES_FOR_CLASSIFICATION = 3  # Phase A Completion Gate item 5: a cell touched
                                   # by fewer VALID dates than this is
                                   # INSUFFICIENT_OBSERVATION, never silently
                                   # folded into CONSISTENTLY_* on a thin sample

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform
_TF_M_TO_4326 = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform


def dates_from_manifest():
    man = pd.read_csv(CFG.TABLES / "P1_targeted_download_manifest.csv")
    return sorted(man.date.unique())


def mosaic_for_date(date: str):
    tok = date.replace("-", "")
    files = sorted(MASKS.glob(f"*{tok}T*_water.tif"))
    if not files:
        return None, None, []
    srcs = [rasterio.open(f) for f in files]
    arr, transform = merge(srcs, nodata=255, resampling=Resampling.nearest)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    tiles = sorted({f.name.split("_")[5] for f in files})
    return arr[0], transform, tiles, crs


def n_required_tiles(date: str) -> int:
    if date in KNOWN_PARTIAL:
        return 4  # what a FULL observation would have needed that date
    return 4 if date == CHECK_DATE else 3


def boundary_distance_stats(a, b):
    if a is None or a.is_empty or b is None or b.is_empty:
        return np.nan, np.nan
    ba = a.boundary
    n, length = 400, a.boundary.length
    if length == 0:
        return np.nan, np.nan
    d = np.array([ba.interpolate(i / n * length).distance(b.boundary) for i in range(n)])
    return float(np.median(d)), float(np.percentile(d, 90))


def main() -> None:
    reservoir = SD.load("reservoir_full_pool_prebreach")
    reservoir_m = shp_transform(_TF_4326_TO_M, reservoir)
    study_m = shp_transform(_TF_4326_TO_M, SD.load("study_domain_full"))
    sa2_area_km2 = reservoir_m.area / 1e6

    dates = dates_from_manifest()
    print(f"dates in manifest: {len(dates)}")

    # ---- common coarse grid for the disagreement map ----------------------
    minx, miny, maxx, maxy = reservoir_m.buffer(3000).bounds
    minx, miny = np.floor(minx / COARSE_RES) * COARSE_RES, np.floor(miny / COARSE_RES) * COARSE_RES
    maxx, maxy = np.ceil(maxx / COARSE_RES) * COARSE_RES, np.ceil(maxy / COARSE_RES) * COARSE_RES
    W, H = int((maxx - minx) / COARSE_RES), int((maxy - miny) / COARSE_RES)
    dst_transform = from_origin(minx, maxy, COARSE_RES, COARSE_RES)
    dst_crs = CFG.CRS_METRIC
    print(f"common grid: {W}x{H} @ {COARSE_RES:.0f} m")

    rows = []
    water_count = np.zeros((H, W), dtype=np.int16)
    obs_count = np.zeros((H, W), dtype=np.int16)  # per-pixel: how many VALID
                                                   # dates actually observed this cell

    for date in dates:
        arr, transform, tiles, crs = mosaic_for_date(date)
        if arr is None:
            print(f"  {date}: no masks built yet, skipping")
            continue
        water_geom = None
        try:
            lab, n = ndimage.label(arr == 1, structure=np.ones((3, 3)))
            polys = [shape(g) for g, v in features.shapes(
                (arr == 1).astype("uint8"), mask=(arr == 1), transform=transform) if v == 1]
            water_geom = unary_union(polys) if polys else None
        except Exception as exc:
            print(f"  {date}: polygonize failed ({exc})")
        if water_geom is not None:
            water_geom = water_geom.intersection(study_m)

        n_req = n_required_tiles(date)
        n_have = len(tiles)
        complete = (n_have >= n_req) or (date in KNOWN_PARTIAL and n_have >= n_req)
        # KNOWN_PARTIAL dates are, by construction, never complete this round
        if date in KNOWN_PARTIAL:
            complete = False

        if water_geom is None or water_geom.is_empty:
            iou = omission = commission = med_d = p90_d = np.nan
            obs_km2 = 0.0
        else:
            inter = water_geom.intersection(reservoir_m)
            union = water_geom.union(reservoir_m)
            iou = inter.area / union.area if union.area else np.nan
            omission = reservoir_m.difference(water_geom).area / 1e6
            commission = water_geom.difference(reservoir_m).area / 1e6
            obs_km2 = water_geom.area / 1e6
            med_d, p90_d = boundary_distance_stats(water_geom, reservoir_m)

        status = "VALID" if complete else "INVALID_FOR_FULL_DOMAIN_IOU"
        rows.append({
            "date": date, "n_tiles_present": n_have, "n_tiles_required": n_req,
            "tiles": "|".join(tiles), "status": status,
            "IoU": round(iou, 4) if np.isfinite(iou) else np.nan,
            "observed_water_area_km2": round(obs_km2, 1),
            "SA2_area_km2": round(sa2_area_km2, 1),
            "omission_area_km2": round(omission, 1) if np.isfinite(omission) else np.nan,
            "commission_area_km2": round(commission, 1) if np.isfinite(commission) else np.nan,
            "shoreline_distance_median_m": round(med_d, 1) if np.isfinite(med_d) else np.nan,
            "shoreline_distance_p90_m": round(p90_d, 1) if np.isfinite(p90_d) else np.nan,
        })
        print(f"  {date}: status={status}  tiles={n_have}/{n_req}  "
              f"IoU={iou if np.isfinite(iou) else float('nan'):.4f}  "
              f"omission={omission:.1f}  commission={commission:.1f}" if np.isfinite(iou) else
              f"  {date}: status={status}  tiles={n_have}/{n_req}  no water geometry")

        if complete and water_geom is not None and not water_geom.is_empty:
            dst_water = np.zeros((H, W), dtype="uint8")
            reproject(source=(arr == 1).astype("uint8"), destination=dst_water,
                     src_transform=transform, src_crs=crs,
                     dst_transform=dst_transform, dst_crs=dst_crs,
                     resampling=Resampling.max)
            # actual observed footprint for THIS date (not assumed grid-wide) --
            # arr!=255 is nodata; reproject that too so obs_count reflects real
            # per-pixel coverage, matching Phase A Completion Gate item 5
            dst_obs = np.zeros((H, W), dtype="uint8")
            reproject(source=(arr != 255).astype("uint8"), destination=dst_obs,
                     src_transform=transform, src_crs=crs,
                     dst_transform=dst_transform, dst_crs=dst_crs,
                     resampling=Resampling.max)
            water_count += dst_water * dst_obs
            obs_count += dst_obs

    cons = pd.DataFrame(rows)
    out = CFG.TABLES / "sa2_multidate_consensus.csv"
    cons.to_csv(out, index=False)
    print(f"\n-> {out}  ({len(cons)} dates total)")

    valid = cons[cons.status == "VALID"]
    print(f"\nVALID dates for IoU statistics: {len(valid)} / {len(cons)}")
    if len(valid):
        print(f"  IoU: median={valid.IoU.median():.4f}  IQR=[{valid.IoU.quantile(.25):.4f},"
              f"{valid.IoU.quantile(.75):.4f}]  min={valid.IoU.min():.4f}  max={valid.IoU.max():.4f}")
        print(f"  commission_area_km2: median={valid.commission_area_km2.median():.1f}  "
              f"range=[{valid.commission_area_km2.min():.1f},{valid.commission_area_km2.max():.1f}]")
        print(f"  omission_area_km2: median={valid.omission_area_km2.median():.1f}  "
              f"range=[{valid.omission_area_km2.min():.1f},{valid.omission_area_km2.max():.1f}]")

    # ---- disagreement classification map -----------------------------------
    n_valid = int((cons.status == "VALID").sum())
    if n_valid == 0:
        print("no VALID dates yet -- skipping disagreement map (masks still building?)")
        return
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_water = np.where(obs_count > 0, water_count / np.maximum(obs_count, 1), 0.0)
    sa2_mask = features.geometry_mask([reservoir_m], out_shape=(H, W),
                                      transform=dst_transform, invert=True)
    cls = np.zeros((H, W), dtype="uint8")
    # 0=outside study domain, 1=CONSISTENTLY OUTSIDE, 2=SYSTEMATIC SA2 COMMISSION,
    # 3=VARIABLE SHORELINE, 4=SYSTEMATIC SA2 OMISSION, 5=CONSISTENTLY WATER,
    # 6=INSUFFICIENT_OBSERVATION (Phase A Completion Gate item 5 -- a cell
    # touched by fewer than MIN_DATES_FOR_CLASSIFICATION VALID dates is never
    # silently folded into a CONSISTENTLY_* class)
    study_mask = features.geometry_mask([study_m], out_shape=(H, W),
                                        transform=dst_transform, invert=True)
    insufficient = obs_count < MIN_DATES_FOR_CLASSIFICATION
    always_water = (frac_water >= 0.9) & ~insufficient
    never_water = (frac_water <= 0.1) & ~insufficient
    variable = study_mask & ~insufficient & ~always_water & ~never_water

    cls[study_mask & insufficient] = 6                # INSUFFICIENT_OBSERVATION (checked first)
    cls[study_mask & never_water & ~sa2_mask] = 1     # CONSISTENTLY OUTSIDE
    cls[study_mask & always_water & ~sa2_mask] = 2    # SYSTEMATIC SA2 COMMISSION
    cls[variable] = 3                                  # VARIABLE SHORELINE
    cls[study_mask & never_water & sa2_mask] = 4      # SYSTEMATIC SA2 OMISSION
    cls[study_mask & always_water & sa2_mask] = 5     # CONSISTENTLY WATER

    labels = ["outside study domain", "CONSISTENTLY OUTSIDE", "SYSTEMATIC SA_2 COMMISSION",
             "VARIABLE SHORELINE", "SYSTEMATIC SA_2 OMISSION", "CONSISTENTLY WATER",
             "INSUFFICIENT OBSERVATION"]
    colors = ["#ffffff", "#e8e8e8", AMBER, "#f5deb3", RED, BLUE, "#c8b8e8"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(11, 9))
    ext = (minx, maxx, miny, maxy)
    im = ax.imshow(cls, extent=ext, origin="upper", cmap=cmap, norm=norm)
    xs, ys = reservoir_m.exterior.xy if hasattr(reservoir_m, "exterior") else (
        list(reservoir_m.geoms)[0].exterior.xy)
    ax.plot(xs, ys, color=INK, lw=1.8, label="SA_2 boundary")
    cbar = fig.colorbar(im, ax=ax, ticks=range(7), pad=0.01)
    cbar.ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"P1C v2 -- SA_2 disagreement classification, n={n_valid} VALID dates "
                f"(full 3-4 tile coverage only)", fontsize=11, loc="left")
    ax.set_xlabel("Easting (m, EPSG:32636)"); ax.set_ylabel("Northing (m)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_fig = CFG.FIG / "P1_sa2_disagreement_map.png"
    fig.savefig(out_fig, dpi=160)
    print(f"-> {out_fig}")
    tot = study_mask.sum()
    for v, lab in zip(range(1, 7), labels[1:]):
        pct = (cls == v).sum() / tot * 100 if tot else 0
        print(f"  {lab}: {(cls==v).sum()*COARSE_RES**2/1e6:.1f} km2 ({pct:.2f}% of study domain)")

    # persist the accumulator rasters so downstream scripts (SA_2 domain
    # revision) can reuse the multi-date water evidence without re-running
    # the full per-date reprojection loop
    raster_dir = CFG.ROOT / "data" / "processed" / "study_domain"
    raster_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("water_count", water_count), ("obs_count", obs_count), ("class", cls)):
        with rasterio.open(raster_dir / f"p1c_v2_{name}.tif", "w", driver="GTiff",
                           height=H, width=W, count=1, dtype="int16",
                           crs=dst_crs, transform=dst_transform, compress="deflate",
                           nodata=-1) as dst:
            dst.write(arr.astype("int16"), 1)
    print(f"-> {raster_dir}/p1c_v2_{{water_count,obs_count,class}}.tif  (n_valid={n_valid})")


if __name__ == "__main__":
    main()
