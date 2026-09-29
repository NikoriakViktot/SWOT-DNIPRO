#!/usr/bin/env python
"""P20 — export the ZONE_2/ZONE_4 PRE_BREACH CV-selected bed DEM as GeoTIFF.

CANONICAL SURFACE = WHICHEVER METHOD WON BLOCKED CV (p19), NOT ALWAYS OK. For
both zones this session that was RBF. `ok_variance` is exported as a
DIAGNOSTIC of the OK method specifically (its Lagrange-system variance) --
it is not the canonical surface's own uncertainty and is deliberately kept
OUT of `confidence_class`, which is built only from method-agnostic signals
(blocked-CV error already reported in p19's table, local roughness, distance
to a real sounding, and now whether a cell's nearest shoreline anchor was
itself extrapolated) so it describes the RBF product actually shipped.

Mirrors hist20_export_dem.py's layer set, plus one addition and one honest
scoping change forced by much sparser data than the reservoir had (527-626
soundings over 1,398-6,920 km2, vs. the reservoir's 7,514 over 2,340 km2 --
roughly an order of magnitude sparser per km2 for ZONE_4):

  method_spread        max-min across the 4 cross-validated interpolators.
                        A LOWER BOUND on uncertainty, not a kriging variance
                        -- same caveat as hist20, carried in the band tag.
  local_roughness       sd of bed elevation over the 8 nearest OTHER soundings.
  dist_to_sounding      distance to the nearest real sounding (not a shoreline
                        pseudo-point).
  ok_variance (NEW)     real kriging variance from ok_predict's Lagrange
                        system (p19). Exported IN ADDITION to method_spread,
                        not instead of it, because with this few points the
                        four smooth interpolators are more likely to agree
                        with each other in genuinely under-constrained cells,
                        understating uncertainty exactly where it matters.
  confidence_class      1/2/3, thresholds computed FROM THIS DATASET's own
                        roughness/distance distribution (tertiles), not
                        copied from hist18's reservoir-specific cut points --
                        those were fitted to a 357 m-spacing, 7,514-point
                        dataset and reusing them unexamined here would be
                        exactly the "one domain's method on a different
                        domain" mistake this project keeps auditing for.

TWO MASKS, NOT ONE -- CAUGHT BY REVIEW ON A MAP, NOT A NUMBER. A registry
zone (ZONE_2/ZONE_4) is an acquisition/analysis AOI, not a water mask
(`spatial_domains.yaml` says so explicitly). The distance-to-sounding
"support" mask below only ever addressed VALUE plausibility (is a prediction
close enough to real data to trust its number); it says nothing about
SPATIAL plausibility (should a bed elevation exist at this location AT
ALL). A cell can sit well within reach of a sounding and still be a field,
an island, or a town -- and the first shipped rasters did exactly that,
visible directly on a basemap. So:

    FINAL_MASK = SUPPORT_MASK (distance to sounding) INTERSECT WATER_MASK
                 (dnipro_water_domain ∩ zone)

never FINAL_MASK = SUPPORT_MASK alone, and never the registry zone polygon.
SUPPORT_MASK: cells farther than DIST_MASK_KM from the nearest sounding are
excluded regardless of the water mask -- the grid may legitimately cover the
whole registry polygon, and "every cell in it is a trustworthy bed estimate"
is a different, false claim. DIST_MASK_KM is set from the fitted variogram
range (p19's own logic: a CV block beyond the range scores extrapolation,
not interpolation) at 1x the range, capped at a few km. WATER_MASK:
`dnipro_water_domain` (chosen over the alternative `water_prebreach` S1
observation -- see the code comment at the mask build for the full
comparison and why) rasterized onto the SAME grid as the bed surface,
exported separately as `zone24_pre_breach_water_mask_{ZONE}_PRE_BREACH_{cell}m.tif` so the
clip is directly auditable in GIS, not just asserted in a log line.

CELL. `--cell` must match the p19 run being exported (default 250, the
canonical/scientific grid; 50/30 are display-only refinements, same role as
hist20_export_dem.py's — not separately re-verified for this sparser
dataset). The cell size is read back from the .npz's own grid spacing and
asserted against `--cell`, never assumed, and tags every output filename so
a fine-grid run can never silently overwrite the canonical 250 m rasters.

Outputs (outputs/rasters/zone24/)
----------------------------------
zone24_bed_{method}_{ZONE}_PRE_BREACH_{cell}m.tif   (4, one per interpolator)
zone24_bed_method_spread_{ZONE}_PRE_BREACH_{cell}m.tif
zone24_bed_local_roughness_{ZONE}_PRE_BREACH_{cell}m.tif
zone24_bed_dist_to_sounding_{ZONE}_PRE_BREACH_{cell}m.tif
zone24_bed_ok_variance_{ZONE}_PRE_BREACH_{cell}m.tif
zone24_bed_confidence_class_{ZONE}_PRE_BREACH_{cell}m.tif
"""
from __future__ import annotations

import argparse
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
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG

REGIME = "PRE_BREACH"
CANONICAL_CELL = 250.0
DIST_MASK_CAP_KM = 5.0
RASTERS = ROOT / "outputs/rasters/zone24"


def write_tif(path, arr2d, tr, nodata, dtype="float32"):
    """`arr2d` is built with row 0 = gy[0] (south), ascending -- but a
    GeoTIFF's row 0 is its northernmost row (from_origin's convention). Must
    flip before writing or every raster reads back mirrored north-south
    (caught by p21's own verification, where the "kept" cells landed 30+ km
    north of every sounding that supposedly constrained them). Same fix as
    hist20_export_dem.py's write_tif, same reason, cited not re-derived."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.flipud(np.asarray(arr2d))
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0],
                       width=a.shape[1], count=1, dtype=dtype,
                       crs=CFG.CRS_METRIC, transform=tr, nodata=nodata,
                       compress="deflate", predictor=2, tiled=True) as ds:
        ds.write(a.astype(dtype), 1)
    print(f"  -> {path}")


def to_grid(values, tgt, inside, gx, gy, nodata=np.nan):
    """Scatter flat inside-cell values back onto the full (ny, nx) grid."""
    ny, nx = len(gy), len(gx)
    full = np.full(ny * nx, nodata, dtype="float64")
    full[inside] = values
    return full.reshape(ny, nx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True,
                    choices=["ZONE_2_KHERSON_DELTA", "ZONE_4_DAM_TO_KHERSON_FLOODWAY"])
    ap.add_argument("--cell", type=float, default=CANONICAL_CELL,
                    help="must match the --cell the p19 run used")
    args = ap.parse_args()
    zone = args.zone
    cell_arg = args.cell
    tag = f"{zone}_{REGIME}"
    cell_suffix = f"{cell_arg:.0f}m"

    print("=" * 78)
    print(f"P20 — export bed surface GeoTIFFs, {tag}, {cell_suffix}")
    print("=" * 78)

    npz_path = (CFG.BULK_ROOT / "data_swot/processed/bathymetry" /
               f"zone24_bed_surface_{tag}_{cell_suffix}.npz")
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path} -- run "
                         f"p19_zone24_bed_surface.py --zone {zone} --cell {cell_arg:.0f} first")
    z = np.load(npz_path, allow_pickle=True)
    gx, gy, inside, tgt = z["gx"], z["gy"], z["inside"], z["tgt"]
    CELL = float(gx[1] - gx[0])  # from the actual grid, never assumed
    if abs(CELL - cell_arg) > 1e-6:
        raise SystemExit(f"grid cell {CELL:.1f} m in {npz_path.name} does not "
                         f"match --cell {cell_arg:.0f}")
    best = str(z["best_method"])
    vg_range_m = float(z["variogram_range_m"])
    dist_to_sounding = z["dist_to_sounding"]
    methods = ("IDW", "OK", "RBF", "LINEAR")
    preds = {m: z[f"pred_{m}"] for m in methods}
    ok_var = z["pred_OK_variance"]
    clip_amount_best = z[f"clip_amount_{best}"]
    clip_n_sounding_best = z[f"clip_n_sounding_nn_{best}"]
    clip_n_shoreline_best = z[f"clip_n_shoreline_nn_{best}"]

    tr = from_origin(float(gx[0]), float(gy[-1]), CELL, CELL)
    dist_mask_km = min(vg_range_m / 1000.0, DIST_MASK_CAP_KM)
    support_masked = dist_to_sounding > dist_mask_km * 1000.0
    print(f"  distance (VALUE-support) mask: {dist_mask_km:.2f} km (min of fitted "
          f"variogram range {vg_range_m/1000:.1f} km and a {DIST_MASK_CAP_KM:.0f} km cap)")
    print(f"  {support_masked.sum():,}/{len(support_masked):,} cells "
          f"({100*support_masked.mean():.1f}%) masked NoData -- beyond any "
          f"sounding's trustworthy reach")

    # ---- SPATIAL mask: a bed DEM may only exist where PRE_BREACH water
    # actually was, never just "distance from a sounding is small enough".
    # A registry zone (ZONE_2/ZONE_4) is an acquisition/analysis AOI, NOT a
    # water mask (spatial_domains.yaml's own provenance says so explicitly);
    # the distance-support mask alone still let the surface bleed onto dry
    # floodplain, fields and islands within reach of a sounding, which a
    # reviewer caught directly on a map. Two PRE_BREACH-appropriate
    # candidates exist in this project:
    #   dnipro_water_domain     ESA WorldCover v200 2021, class 80+90,
    #                           CONNECTIVITY-derived -- islands fall out as
    #                           interior rings by construction, not by a
    #                           later hole-fill step. Single-epoch (2021,
    #                           pre-breach), already the geometry p18 used
    #                           to place shoreline pseudo-points, so the mask
    #                           boundary stays consistent with the elevation
    #                           constraint's own boundary.
    #   water_prebreach (p0r/p0w)  observed S1 classification, 2023-06-01/02
    #                           (2 dates, right before the breach). Larger
    #                           than dnipro_water_domain in both zones
    #                           (ZONE_2 433 vs 309 km2, ZONE_4 755 vs 592 km2)
    #                           -- plausibly real extra wet area, but SAR
    #                           water classification in a reed/braided delta
    #                           is this project's own documented hardest
    #                           case (ZONE_2's registry provenance: "dark-SAR
    #                           =water degrades" here), from only 2 dates,
    #                           never cross-validated against another date.
    # CHOSEN: dnipro_water_domain -- connectivity-validated channel topology
    # (matches what a reviewer expects: channels + straits + open water,
    # islands excluded, not a flood-pulse blob), and geometric consistency
    # with the shoreline constraint already governing this same surface.
    from swot_dnipro import spatial_domains as SD
    wdom = SD.load_utm("dnipro_water_domain").intersection(SD.load_utm(zone))
    import shapely
    is_water = shapely.contains_xy(wdom, tgt[:, 0], tgt[:, 1])
    land_masked = ~is_water
    print(f"  SPATIAL (water-domain) mask: dnipro_water_domain ∩ {zone} = "
          f"{wdom.area/1e6:,.1f} km2; {land_masked.sum():,}/{len(land_masked):,} "
          f"cells ({100*land_masked.mean():.1f}%) are land -- masked "
          f"regardless of interpolation support")

    masked = support_masked | land_masked
    print(f"  FINAL mask = support ∪ land: {masked.sum():,}/{len(masked):,} "
          f"({100*masked.mean():.1f}%) excluded, "
          f"{(~masked).sum():,} cells kept")

    RASTERS.mkdir(parents=True, exist_ok=True)
    write_tif(RASTERS / f"zone24_pre_breach_water_mask_{tag}_{cell_suffix}.tif",
              to_grid(is_water.astype(np.float32), tgt, inside, gx, gy, nodata=0.0),
              tr, 0.0, dtype="uint8")
    stack = np.stack([np.where(masked, np.nan, preds[m]) for m in methods])
    method_spread = stack.max(0) - stack.min(0)

    snd = pd.read_parquet(ROOT / "data/processed/bathymetry/manual_soundings_evrf2019.parquet")
    snd = snd[snd[zone]]
    xy_snd = np.c_[snd.x.values, snd.y.values]
    z_snd = snd.H_bed_evrf2019_m.values
    tree = cKDTree(xy_snd)
    d8, i8 = tree.query(xy_snd, k=min(9, len(xy_snd)))
    local_roughness_at_soundings = np.array(
        [np.std(z_snd[i8[k, 1:]]) for k in range(len(xy_snd))])
    _, i_nn = tree.query(tgt, k=1)
    local_roughness = local_roughness_at_soundings[i_nn]
    local_roughness = np.where(masked, np.nan, local_roughness)

    dist_masked = np.where(masked, np.nan, dist_to_sounding)

    # near_extrapolated_shoreline: whether a cell's NEAREST shoreline pseudo-
    # point (p18) was itself flagged extrapolated (>15 km from any real
    # water-level anchor). This is the "support class / extrapolation mask"
    # signal for the canonical RBF surface -- a cell can sit close to a
    # sounding yet still be governed, near the boundary, by a shoreline
    # constraint that was itself a guess.
    sh = pd.read_parquet(ROOT / f"data/processed/bathymetry/"
                         f"zone24_shore_pseudopoints_{REGIME}.parquet")
    sh = sh[sh.zone == zone]
    sh_tree = cKDTree(np.c_[sh.x.values, sh.y.values])
    _, i_sh = sh_tree.query(tgt, k=1)
    near_extrapolated = sh.extrapolated.values[i_sh].astype(float)
    near_extrapolated = np.where(masked, np.nan, near_extrapolated)
    write_tif(RASTERS / f"zone24_bed_near_extrapolated_shoreline_{tag}_{cell_suffix}.tif",
              to_grid(near_extrapolated, tgt, inside, gx, gy), tr, np.nan)

    for m in methods:
        arr = to_grid(np.where(masked, np.nan, preds[m]), tgt, inside, gx, gy)
        write_tif(RASTERS / f"zone24_bed_{m}_{tag}_{cell_suffix}.tif", arr, tr, np.nan)
    write_tif(RASTERS / f"zone24_bed_method_spread_{tag}_{cell_suffix}.tif",
              to_grid(method_spread, tgt, inside, gx, gy), tr, np.nan)
    write_tif(RASTERS / f"zone24_bed_local_roughness_{tag}_{cell_suffix}.tif",
              to_grid(local_roughness, tgt, inside, gx, gy), tr, np.nan)
    write_tif(RASTERS / f"zone24_bed_dist_to_sounding_{tag}_{cell_suffix}.tif",
              to_grid(dist_masked, tgt, inside, gx, gy), tr, np.nan)
    write_tif(RASTERS / f"zone24_bed_ok_variance_{tag}_{cell_suffix}.tif",
              to_grid(np.where(masked, np.nan, ok_var), tgt, inside, gx, gy), tr, np.nan)

    # ---- envelope-guard clip diagnostics for the SHIPPED (best) method ----
    # Per the review that caught the RBF overshoot: the guard is now part of
    # the estimator, not a cosmetic safety net, so how much of the surface it
    # actually altered -- and whether that came from real soundings or the
    # softer shoreline constraint -- is provenance, not an implementation
    # detail. Exported here; p21 check 9 turns it into the summary numbers.
    write_tif(RASTERS / f"zone24_bed_clip_amount_{tag}_{cell_suffix}.tif",
              to_grid(np.where(masked, np.nan, clip_amount_best), tgt, inside, gx, gy),
              tr, np.nan)
    clip_mask = (clip_amount_best > 1e-6) & ~masked
    n_clip = int(clip_mask.sum())
    print(f"\n  envelope-guard clip diagnostics ({best}, the shipped method):")
    if n_clip:
        ca = clip_amount_best[clip_mask]
        print(f"    {n_clip:,}/{int((~masked).sum()):,} kept cells "
              f"({100*n_clip/(~masked).sum():.1f}%) were clipped -- "
              f"|clip| median {np.median(ca):.2f} m, p95 {np.percentile(ca,95):.2f} m, "
              f"max {ca.max():.2f} m")
        print(f"    among clipped cells' {8} nearest neighbours: median "
              f"{np.median(clip_n_sounding_best[clip_mask]):.0f} real soundings, "
              f"{np.median(clip_n_shoreline_best[clip_mask]):.0f} shoreline pseudo-points")
    else:
        print("    0 kept cells clipped")

    # ---- confidence class, thresholds from THIS dataset's own tertiles -----
    # Method-agnostic indicators only -- deliberately NOT ok_variance, which
    # is a diagnostic of the OK method alone and would misrepresent the
    # canonical RBF surface's own reliability if folded in here.
    rq = np.nanpercentile(local_roughness, [33, 67])
    dq = np.nanpercentile(dist_masked, [33, 67])
    print(f"\n  confidence thresholds (this dataset): roughness tertiles "
          f"{rq[0]:.2f}/{rq[1]:.2f} m, distance tertiles "
          f"{dq[0]:.0f}/{dq[1]:.0f} m")
    cls = np.full(local_roughness.shape, 2, dtype="float64")  # MODERATE default
    high = (local_roughness <= rq[0]) & (dist_masked <= dq[0]) & (near_extrapolated == 0)
    lower = (local_roughness >= rq[1]) | (dist_masked >= dq[1]) | (near_extrapolated == 1)
    cls[high] = 1
    cls[lower & ~high] = 3
    cls[masked] = np.nan
    n_downgraded_by_extrap = int(((near_extrapolated == 1) & (cls == 3)
                                  & ~((local_roughness >= rq[1]) | (dist_masked >= dq[1]))).sum())
    print(f"  {n_downgraded_by_extrap:,} cells downgraded to LOWER solely because "
          f"their nearest shoreline anchor was extrapolated (p18)")
    for k, name in ((1, "HIGH"), (2, "MODERATE"), (3, "LOWER")):
        n = int((cls == k).sum())
        print(f"    class {k} ({name}): {n:,} cells "
              f"({100*n/np.isfinite(cls).sum():.1f}%)")
    write_tif(RASTERS / f"zone24_bed_confidence_class_{tag}_{cell_suffix}.tif",
              to_grid(cls, tgt, inside, gx, gy), tr, np.nan)

    print(f"\n  preferred method for this run: {best}")
    print(f"-> {RASTERS}")


if __name__ == "__main__":
    main()
