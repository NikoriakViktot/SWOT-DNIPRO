#!/usr/bin/env python
"""HISTORICAL 20 — export the bed surface as GeoTIFF, with confidence layers.

The reconstructed bed has existed only as a compact .npz of inside-only float32
arrays, which is what every script in this project indexes but which no GIS can
open. This writes standard north-up single-band GeoTIFFs in EPSG:32636.

WHY MORE THAN ONE LAYER. hist18 showed that the prediction error is strongly
heterogeneous -- the worst 5 % of observations carry 51 % of the squared error,
concentrated where the bed is rough, where it is deeply incised, and within
~500 m of the shoreline. Shipping a single bed_elevation.tif invites the reader
to treat a channel wall and a smooth floodplain platform as equally reliable.
So the bed goes out with the three measured drivers of its error and a
rule-based confidence class built from them.

WHAT THE UNCERTAINTY LAYER IS, AND IS NOT. `method_spread` is the range across
the four independently cross-validated interpolators at each cell. It is a
measured quantity with no model behind it, and it is a LOWER BOUND on the true
uncertainty: four smooth interpolators fed the same sparse points agree with
each other more than any of them agrees with the bed. It is NOT a kriging
variance, and it must not be read as a standard error.

The canonical scientific grid remains 250 m: hist17 showed the integrated area
statistics move by only -0.06 pp between 250 m and 30 m, so the fine grids add
resolution for display, not information.

Outputs (outputs/rasters/)
--------------------------
kakhovka_bed_OK_epoch_{250,50,30}m.tif    bed elevation, m EVRF2019
kakhovka_bed_method_spread_250m.tif       inter-interpolator range, m
kakhovka_bed_local_roughness_250m.tif     sd of bed over 8 nearest soundings, m
kakhovka_bed_dist_to_sounding_250m.tif    distance to nearest sounding, m
kakhovka_bed_confidence_class_250m.tif    1 high / 2 moderate / 3 lower
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import shape

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

OUT = ROOT / "outputs/rasters"
OUT.mkdir(parents=True, exist_ok=True)
BATH = CFG.BULK_ROOT / "data_swot/processed/bathymetry"
NODATA = -9999.0
METHOD, VARIANT = "OK", "epoch"        # the preferred surface and constraint
CANON_CELL = 250.0
# Thresholds for the confidence class, taken from hist18's decomposition bins
# rather than chosen here. Each is the point where measured RMSE changes band.
SHORE_NEAR_M = 500.0        # RMSE 4.24 m inside this, 1.18 m beyond 4 km
SHORE_FAR_M = 2000.0        # RMSE 2.24 m in 1-2 km, 1.49 m in 2-4 km
ROUGH_Q_HI = 0.75           # roughest decile 5.46 m vs 0.88 m in Q1
ROUGH_Q_LO = 0.50
DIST_FAR_M = 750.0          # beyond the p90 sounding spacing (555 m)
DIST_NEAR_M = 500.0


def write_tif(path, arr2d, gx, gy, cell, nodata=NODATA, dtype="float32",
              descr=""):
    """One north-up single-band GeoTIFF.

    The grids are stored with gy ASCENDING (row 0 = south), which is why the
    array is flipped: a GeoTIFF's first row is its northernmost.
    """
    a = np.flipud(np.asarray(arr2d))
    a = np.where(np.isfinite(a), a, nodata).astype(dtype)
    tr = from_origin(gx[0] - cell / 2, gy[-1] + cell / 2, cell, cell)
    with rasterio.open(
            path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
            count=1, dtype=dtype, crs=CFG.CRS_METRIC, transform=tr,
            nodata=nodata, compress="LZW", tiled=True,
            blockxsize=256, blockysize=256) as ds:
        ds.write(a, 1)
        if descr:
            ds.set_band_description(1, descr)
        ds.update_tags(1, DESCRIPTION=descr)
    n = int(np.isfinite(np.asarray(arr2d)).sum())
    print(f"  -> {path.name:<46}{a.shape[1]}x{a.shape[0]}  "
          f"{n:,} valid  {path.stat().st_size/1e6:.1f} MB")


def as_raster(flat_inside, ins_idx, nx, ny):
    full = np.full(nx * ny, np.nan, np.float64)
    full[ins_idx] = flat_inside
    return full.reshape(ny, nx)


def main() -> None:
    # ---- soundings, for the two data-geometry layers -----------------------
    sd = pd.read_parquet(BATH / "kakhovka_soundings_evrf2019.parquet")
    sd = sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0)).groupby(
        ["_kx", "_ky"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"),
        H_bed_evrf2019_m=("H_bed_evrf2019_m", "mean"))
    sxy = np.c_[sd.x.values, sd.y.values]
    sz = sd.H_bed_evrf2019_m.values
    stree = cKDTree(sxy)
    # local roughness AT EACH SOUNDING, exactly as hist18 defines it
    _, i8 = stree.query(sxy, k=9)
    s_rough = sz[i8[:, 1:]].std(axis=1)
    rtree_rough = cKDTree(sxy)
    print(f"{len(sd):,} soundings; local roughness "
          f"{s_rough.min():.2f}..{s_rough.max():.2f} m "
          f"(Q50 {np.median(s_rough):.2f})")

    # was the P20 footprint: the distance-to-shore band below was measured
    # against a boundary 9.4 km short of the real eastern shore
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    fp_bnd = shapely.boundary(fp)

    files = sorted(BATH.glob("kakhovka_bed_surface_*m.npz"),
                   key=lambda p: -float(p.stem.split("_")[-1][:-1]))
    if not files:
        raise SystemExit("no bed surfaces on disk -- run hist14 first")

    for f in files:
        npz = np.load(f, allow_pickle=True)
        cell = float(npz["cell_m"])
        gx, gy, ins = npz["gx"], npz["gy"], npz["ins_idx"]
        # Refuse a truncated surface rather than publish from it. The bed-surface
        # npz files were written on the P20 footprint (E max 668,540-668,670) while
        # the registry domain reaches 678,000 m; consuming one silently truncated
        # every product below. Regenerate with hist14 after its domain fix.
        _fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                           SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
        SD.assert_covers(_fp, gx, gy, what=str("bed surface npz (hist20)"),
                         cell=float(npz["cell_m"]) if "cell_m" in npz else 0.0)
        nx, ny = len(gx), len(gy)
        pref = str(npz["preferred"])
        if pref != METHOD:
            print(f"\n!! {f.name}: preferred method is {pref}, not {METHOD} -- "
                  f"exporting {METHOD} anyway, as the canonical choice")
        print(f"\n{cell:.0f} m grid: {nx}x{ny}, {len(ins):,} cells inside "
              f"({len(ins)*(cell/1e3)**2:,.0f} km2)")

        bed = npz[f"surf_{VARIANT}_{METHOD}"].astype(np.float64)
        write_tif(OUT / f"kakhovka_bed_{METHOD}_{VARIANT}_{cell:.0f}m.tif",
                  as_raster(bed, ins, nx, ny), gx, gy, cell,
                  descr=f"Kakhovka pre-breach bed elevation, m EVRF2019 normal "
                        f"heights; ordinary kriging, shoreline constrained at "
                        f"17.08 m (2023-06-05 waterline); soundings reduced to "
                        f"UNS 14.00 m; blocked-1km CV RMSE 2.76 m, "
                        f"NMAD 1.32 m, bias -0.03 m")

        if abs(cell - CANON_CELL) > 1:
            print("     (confidence layers are written for the canonical "
                  "250 m grid only)")
            continue

        # ---------- cell coordinates, straight off the flat index -----------
        cx = gx[ins % nx]
        cy = gy[ins // nx]
        cxy = np.c_[cx, cy]

        # ---------- layer: inter-interpolator spread ------------------------
        stack = np.vstack([npz[f"surf_{VARIANT}_{m}"].astype(np.float64)
                           for m in ("IDW", "OK", "RBF", "LINEAR")])
        spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        print(f"     method spread: median {np.nanmedian(spread):.2f} m, "
              f"p90 {np.nanpercentile(spread,90):.2f} m, "
              f"max {np.nanmax(spread):.2f} m")
        write_tif(OUT / f"kakhovka_bed_method_spread_{cell:.0f}m.tif",
                  as_raster(spread, ins, nx, ny), gx, gy, cell,
                  descr="Range across 4 cross-validated interpolators (IDW, "
                        "OK, RBF, LINEAR), m. A LOWER BOUND on uncertainty, "
                        "NOT a kriging variance: smooth interpolators on "
                        "sparse points agree with each other more than with "
                        "the bed.")

        # ---------- layer: local roughness, nearest-sounding value ----------
        _, inn = rtree_rough.query(cxy, k=1)
        rough = s_rough[inn]
        write_tif(OUT / f"kakhovka_bed_local_roughness_{cell:.0f}m.tif",
                  as_raster(rough, ins, nx, ny), gx, gy, cell,
                  descr="Local bed roughness: sd of bed elevation over the 8 "
                        "soundings nearest to the sounding nearest this cell, "
                        "m. hist18: the best-populated predictor of "
                        "prediction error (0.88 m RMSE in the smoothest "
                        "quartile vs 5.52 m in the roughest decile).")

        # ---------- layer: distance to nearest sounding ---------------------
        dsound, _ = stree.query(cxy, k=1)
        write_tif(OUT / f"kakhovka_bed_dist_to_sounding_{cell:.0f}m.tif",
                  as_raster(dsound, ins, nx, ny), gx, gy, cell,
                  descr="Distance from cell centre to the nearest sounding, m. "
                        "Median sounding spacing is 357 m (p90 555 m).")

        # ---------- layer: confidence class ---------------------------------
        dshore = shapely.distance(shapely.points(cx, cy), fp_bnd)
        rq_hi = np.quantile(s_rough, ROUGH_Q_HI)
        rq_lo = np.quantile(s_rough, ROUGH_Q_LO)
        cls = np.full(len(ins), 2, np.float64)              # MODERATE default
        high = ((rough <= rq_lo) & (dshore > SHORE_FAR_M)
                & (dsound <= DIST_NEAR_M))
        lower = ((rough > rq_hi) | (dshore < SHORE_NEAR_M)
                 | (dsound > DIST_FAR_M))
        cls[high] = 1
        cls[lower] = 3                                      # lower wins
        for k, lab in ((1, "HIGH"), (2, "MODERATE"), (3, "LOWER")):
            sel = cls == k
            print(f"     class {k} {lab:<9}{100*sel.mean():>5.1f}% of area   "
                  f"median spread {np.nanmedian(spread[sel]):.2f} m   "
                  f"median roughness {np.median(rough[sel]):.2f} m")
        write_tif(OUT / f"kakhovka_bed_confidence_class_{cell:.0f}m.tif",
                  as_raster(cls, ins, nx, ny), gx, gy, cell, dtype="int16",
                  nodata=0,
                  descr="1 HIGH: roughness <= median AND >2 km from shore AND "
                        "<=500 m from a sounding. 3 LOWER: roughness in the "
                        "top quartile OR <500 m from shore OR >750 m from a "
                        "sounding. 2 MODERATE otherwise. Thresholds are the "
                        "band edges of hist18's measured error decomposition, "
                        "not free parameters.")

        # sanity: the class must actually separate the measured spread
        m1 = np.nanmedian(spread[cls == 1])
        m3 = np.nanmedian(spread[cls == 3])
        print(f"     check: median method spread rises {m1:.2f} -> {m3:.2f} m "
              f"from class 1 to class 3 ({m3/m1:.1f}x) — the class separates "
              f"an INDEPENDENT measure of uncertainty")
        if not m3 > m1:
            print("     !! WARNING: the confidence class does not order the "
                  "method spread. Do not ship it.")

    print(f"\nCanonical scientific grid: {CANON_CELL:.0f} m. The 50 m and 30 m "
          f"rasters are for display:\nhist17 showed the integrated exposure "
          f"fraction moves only -0.06 pp between them.")
    print("\nQuote with the DEM, never the RMSE alone:")
    print("  RMSE 2.76 m | NMAD 1.32 m | bias -0.03 m | median |e| 0.89 m")
    print("  and: the worst 5 % of observations account for 51 % of the "
          "squared prediction error.")


if __name__ == "__main__":
    main()
