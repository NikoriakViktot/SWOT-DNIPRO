#!/usr/bin/env python
"""P1C -- multi-date consensus VALIDATION of Kakhovka_SA_2.geojson (2026-09-11
operator correction). Compares each available pre-breach water mask against
reservoir_full_pool_prebreach = SA_2. This is validation, not fitting: SA_2 is
never modified here.

Scope this run: only the pre-breach dates ALREADY CACHED locally qualify for
genuine pixel-level comparison without a new download (2023-05-16, 2023-05-19,
2023-06-05 -- 3 dates, all within 3 weeks of the breach). This is explicitly
NOT the >=5-date, multi-year consensus the design calls for; the next 10
highest-ranked, 100%-coverage, ~0% cloud dates from P1B
(outputs/tables/prebreach_fullpool_scene_candidates.csv) span 2019-2023 and
are the recommended download list to extend this to a genuine multi-year
consensus (see the P0/P1 return-to-operator summary).

Outputs
-------
outputs/tables/sa2_multidate_consensus.csv
outputs/figures/P1_best_prebreach_fullpool_masks.png
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
from rasterio import features
from rasterio.merge import merge
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

MASKS = CFG.ROOT / "data" / "processed" / "water_masks"
INK, BLUE, RED, AMBER, GREEN = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e"

#: (date, [tile substrings]) -- the only pre-breach dates with a cached raster today
CACHED_PREBREACH = {
    "2023-05-16": ["20230516T083601"],
    "2023-05-19": ["20230519T084601"],
    "2023-06-05": ["20230605T083601"],
}

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform
_TF_M_TO_4326 = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform


def mosaic_water_mask(date_token: str):
    files = sorted(MASKS.glob(f"*{date_token}*_water.tif"))
    if not files:
        return None, None, []
    srcs = [rasterio.open(f) for f in files]
    arr, transform = merge(srcs, nodata=255, resampling=Resampling.nearest)
    for s in srcs:
        s.close()
    tiles_used = sorted({f.name.split("_")[5] for f in files})
    return arr[0], transform, tiles_used


def polygonize(mask, transform):
    lab, n = ndimage.label(mask == 1, structure=np.ones((3, 3)))
    if n == 0:
        return None
    polys = [shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=(mask == 1),
             transform=transform) if v == 1]
    return unary_union(polys) if polys else None


def boundary_distance_stats(a, b):
    """Median/p90 distance from sampled boundary vertices of `a` to `b`'s boundary."""
    if a is None or a.is_empty or b is None or b.is_empty:
        return np.nan, np.nan
    ba = a.boundary
    n = 400
    dists = []
    length = ba.length
    if length == 0:
        return np.nan, np.nan
    for i in range(n):
        pt = ba.interpolate(i / n * length)
        dists.append(pt.distance(b.boundary))
    d = np.array(dists)
    return float(np.median(d)), float(np.percentile(d, 90))


def main() -> None:
    reservoir = SD.load("reservoir_full_pool_prebreach")
    reservoir_m = shp_transform(_TF_4326_TO_M, reservoir)
    sa2_area_km2 = reservoir_m.area / 1e6
    # a single Sentinel-2 tile (110x110 km) extends far past the reservoir --
    # e.g. T36TWS alone reaches south to the Black Sea coast / Dnipro-Buh
    # liman. Restrict the water polygon to study_domain_full BEFORE computing
    # commission/omission/IoU, or unrelated water bodies swamp the comparison.
    study_domain_m = shp_transform(_TF_4326_TO_M, SD.load("study_domain_full"))

    rows, masks_for_fig = [], []
    for date, tokens in CACHED_PREBREACH.items():
        arr, transform, tiles_used = mosaic_water_mask(tokens[0])
        if arr is None:
            print(f"  {date}: no cached water-mask rasters found, skipping")
            continue
        water_geom = polygonize(arr, transform)  # already EPSG:32636
        if water_geom is None or water_geom.is_empty:
            print(f"  {date}: mask produced no water polygon, skipping")
            continue
        water_geom = water_geom.intersection(study_domain_m)
        inter = water_geom.intersection(reservoir_m)
        union = water_geom.union(reservoir_m)
        iou = inter.area / union.area if union.area else np.nan
        omission = reservoir_m.difference(water_geom).area / 1e6
        commission = water_geom.difference(reservoir_m).area / 1e6
        obs_area_km2 = water_geom.area / 1e6
        med_d, p90_d = boundary_distance_stats(water_geom, reservoir_m)
        valid = (arr != 255)
        # cloud/valid fraction WITHIN whatever this raster grid overlaps --
        # this is NOT reservoir-wide coverage (a single 110km tile's grid
        # only spans part of the reservoir on 2023-05-16/05-19), so it only
        # answers "was the overlapping part cloudy", not "was the whole
        # reservoir observed". Reservoir-wide coverage is the tile-geometry
        # fraction already computed in P1B (native_coverage_fraction).
        cov_mask = features.geometry_mask([reservoir_m], out_shape=arr.shape,
                                          transform=transform, invert=True)
        raster_valid_fraction_within_overlap = float(
            (valid & cov_mask).sum() / max(cov_mask.sum(), 1))
        rows.append({
            "date": date, "IoU": round(iou, 4),
            "observed_water_area_km2": round(obs_area_km2, 1),
            "SA2_area_km2": round(sa2_area_km2, 1),
            "omission_area_km2": round(omission, 1),
            "commission_area_km2": round(commission, 1),
            "shoreline_distance_median_m": round(med_d, 1) if np.isfinite(med_d) else None,
            "shoreline_distance_p90_m": round(p90_d, 1) if np.isfinite(p90_d) else None,
            "raster_valid_fraction_within_overlap": round(raster_valid_fraction_within_overlap, 4),
            "n_pixels_valid": int(valid.sum()), "raster_shape": str(arr.shape),
            "tiles_actually_mosaicked": "|".join(tiles_used),
            "n_tiles_actually_mosaicked": len(tiles_used),
        })
        masks_for_fig.append((date, water_geom))
        print(f"  {date}: IoU={iou:.4f}  omission={omission:.1f} km2  "
              f"commission={commission:.1f} km2  "
              f"raster_valid_within_overlap={raster_valid_fraction_within_overlap:.3f}  "
              f"shoreline_dist med/p90={med_d:.0f}/{p90_d:.0f} m")

    cons = pd.DataFrame(rows)
    cand_path = CFG.TABLES / "prebreach_fullpool_scene_candidates.csv"
    if cand_path.exists():
        cand = pd.read_csv(cand_path)[["date", "native_coverage_fraction", "n_tiles_present"]]
        cons = cons.merge(cand, on="date", how="left").rename(
            columns={"native_coverage_fraction": "reservoir_native_coverage_fraction"})
        cons["note"] = np.where(
            cons.n_tiles_actually_mosaicked < 4,
            "PARTIAL LOCAL TILE COVERAGE this date (only " +
            cons.n_tiles_actually_mosaicked.astype(str) +
            "/4 required tiles cached) -- omission_area_km2 mostly reflects "
            "un-observed area, NOT genuine SA_2 disagreement. The catalogue "
            "(P1B) confirms a full 4-tile acquisition set EXISTS for this "
            "date and could be downloaded to complete the test. Only trust "
            "IoU/commission from dates with n_tiles_actually_mosaicked==4.",
            "")
    out = CFG.TABLES / "sa2_multidate_consensus.csv"
    cons.to_csv(out, index=False)
    print(f"\n-> {out}  (n={len(cons)} dates -- LIMITED to what is already cached; "
          f"see docstring for the recommended next-download list)")
    if len(cons):
        print(f"\nCONSENSUS SUMMARY (n={len(cons)}):")
        print(f"  IoU: mean={cons.IoU.mean():.4f}  min={cons.IoU.min():.4f}  max={cons.IoU.max():.4f}")
        print(f"  omission_area_km2: mean={cons.omission_area_km2.mean():.1f}")
        print(f"  commission_area_km2: mean={cons.commission_area_km2.mean():.1f}")

    # ---- figure: side-by-side masks over SA_2 outline --------------------
    n = len(masks_for_fig)
    if n == 0:
        print("no masks to plot"); return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5))
    if n == 1:
        axes = [axes]
    for ax, (date, water_geom) in zip(axes, masks_for_fig):
        def plot_poly(geom, ax, **kw):
            geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
            for g in geoms:
                if g.is_empty:
                    continue
                xs, ys = g.exterior.xy
                ax.fill(xs, ys, **kw)
        plot_poly(reservoir_m, ax, facecolor="none", edgecolor=INK, lw=2.2, zorder=3)
        plot_poly(water_geom, ax, facecolor=BLUE, alpha=0.45, edgecolor=BLUE, lw=0.8, zorder=2)
        row = cons[cons.date == date].iloc[0]
        ax.set_title(f"{date}\nIoU={row.IoU:.3f}  omission={row.omission_area_km2:.0f} km2  "
                     f"commission={row.commission_area_km2:.0f} km2", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(handles=[
        plt.Line2D([0], [0], color=INK, lw=2.2, label="SA_2 (authoritative)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, alpha=0.45, edgecolor=BLUE,
                      label="observed water (this date)")],
        fontsize=8, loc="lower left")
    fig.suptitle("P1C -- multi-date consensus validation of Kakhovka_SA_2.geojson "
                 f"(n={n}, cached dates only)", fontsize=12)
    fig.tight_layout()
    out_fig = CFG.FIG / "P1_best_prebreach_fullpool_masks.png"
    fig.savefig(out_fig, dpi=160)
    print(f"-> {out_fig}")


if __name__ == "__main__":
    main()
