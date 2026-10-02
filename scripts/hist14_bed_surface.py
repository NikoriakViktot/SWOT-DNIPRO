#!/usr/bin/env python
"""HISTORICAL 14 — an area-weighted bed surface, chosen by validation.

Tasks 1 and 3. The point-count exposure test (hist12/E) was invalid because the
soundings follow the navigable channel and under-sample the shallow margins, so
a point sample cannot measure an area fraction. This builds the surface that
can, and picks between interpolators on cross-validation rather than on how
smooth they look.

Four interpolators, all on the same grid and domain:

    IDW        inverse distance, power 2, 12 neighbours
    OK         ordinary kriging, local, with a variogram fitted by SciKit-GStat
    RBF        thin-plate spline, 50 neighbours
    LINEAR     barycentric on the Delaunay triangulation

CROSS-VALIDATION IS SPATIALLY BLOCKED, at two scales, and the scale decides the
ranking. The fitted variogram range is ~2 km, so a 5 km block holds out cells
lying BEYOND the correlation range of any training point: that scores
extrapolation, which no method can pass, and is not how this surface is used.
Median sounding spacing is 357 m, so 1 km blocks test interpolation across the
gaps the surface actually bridges. Both are reported; the method is chosen on
the 1 km score. Random holdout is reported too, and is optimistic, because
soundings sit metres apart along survey lines.

SHORELINE CONSTRAINT. The first version of this script had NONE, and that was a
defect with a direct effect on the exposure fraction it feeds. The soundings
stop short of the shore: only 6.3 % lie within 250 m of the footprint boundary,
the closest are ~167 m out, and the 250 nearest-to-shore soundings have a median
bed of 12.15 m. The waterline itself is at ~17.1 m. Left unconstrained, an
interpolator carries 12 m outward to the shore and models the margin ~5 m too
deep -- and the drying fraction counts cells above 12.885 m, so it would be
biased LOW exactly at the margins where drying happens. RBF even extrapolated
to +67.9 m there.

The constraint is therefore boundary pseudo-points at the waterline elevation,
and that elevation is fixed INDEPENDENTLY of anything this test is compared
against. The footprint is dated 2023-06-05 with an area of 2 191.8 km2. Two
unrelated routes agree on its level:

    Rozumivka gauge, 2023-06-05        16.88 m BS-77  =  17.08 m EVRF2019
    Table 19 area curve at 2 192 km2      ~17.1 m BS-77

so the 2023-06-05 waterline sits about 1 m ABOVE the design NPG of 16.0 m -- the
pool was raised that spring. A constraint at NPG would itself have been wrong.
Three variants are built and reported: no constraint, NPG, and the epoch
waterline. The level is NOT chosen to make the exposure fractions match
Table 21.

No channel geometry is used as a constraint. The SWORD line enters only as a
diagnostic axis for reporting residuals, never as an input to any surface, so it
cannot act as a hidden tuning target.

PARALLELISM, with an honest note on when it earns its keep. Profiling found the
cost was never the kriging: all 57,527 local solves take ~3 s, and batching them
gains only 1.2x. The 20-minute runtime was ONE library call --
GeoSeries.within() against a 34,831-vertex polygon, ~1,805 s for this grid
against 0.02 s for shapely.contains_xy, with identical masks. That is fixed
below, and at the default 250 m cell the whole script runs in a couple of
minutes on one core, where --jobs buys almost nothing.

It is provided because it buys a great deal as soon as the grid is refined: cell
size enters as 1/cell^2, so --cell 50 is 25x the targets and kriging then
dominates. Both the CV folds and the grid chunks are embarrassingly parallel.

    --jobs N     worker processes (default 1)
    --cell M     grid cell size in metres (default 250)
    --verify     check that the parallel path reproduces the serial one

Implementation notes that matter for correctness:
  * with --jobs > 1 the BLAS thread limit is pinned to 1 BEFORE numpy is
    imported. Without that, N processes each spawning 32 BLAS threads on a
    32-thread machine thrash and can run SLOWER than serial;
  * the big arrays are shared by fork, not pickled, so per-task payloads stay
    at a few tens of kB;
  * every random draw -- fold assignment and the variogram subsample -- happens
    in the PARENT, so results do not depend on --jobs. --verify checks that.

Outputs, at the canonical 250 m cell
-----------------------------------
outputs/tables/hist14_interpolator_cv.csv
outputs/tables/hist14_surface_summary.csv
data/processed/bathymetry/kakhovka_bed_surface_250m.npz
outputs/figures/V12_bed_surface_validation.png

Any other --cell tags the surface, the summary and the figure with the cell size
(`..._50m`), so a fine-grid run cannot overwrite the canonical outputs and the
resolution is always readable off the filename. hist17 reads every surface on
disk and reports whether the answer depends on the cell size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# BLAS pinning has to happen BEFORE numpy is imported, and the default fork
# start method means children inherit whatever the parent set up. So the job
# count is read straight off argv here, ahead of any heavy import.
_JOBS = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--jobs" and _i + 1 < len(sys.argv):
        _JOBS = int(sys.argv[_i + 1])
    elif _a.startswith("--jobs="):
        _JOBS = int(_a.split("=", 1)[1])
if _JOBS != 1:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import warnings

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree
from shapely.geometry import shape

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from shapely.ops import unary_union
from swot_dnipro import sword as SW

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
BLOCK_KM_LIST = [1.0, 5.0]
N_FOLDS = 10
IDW_K, IDW_P = 12, 2.0
OK_K = 32
OK_BATCH = 4000              # kriging systems solved per batched call
RBF_K = 50
SHORE_STEP_M = 250.0
SHORE_VARIANTS = {"none": None, "npg": 16.00 + 0.185, "epoch": 17.08}
GMO_EVRF = 12.70 + 0.185
CHUNK = 4000                 # grid targets per parallel task
RNG = np.random.default_rng(CFG.SEED)

# Shared, read-only, inherited by fork. Never pickled.
_S: dict = {}


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


# --------------------------------------------------------------- methods ----
def _idw(xy_tr, z_tr, xy_te, vg=None):
    d, i = cKDTree(xy_tr).query(xy_te, k=min(IDW_K, len(xy_tr)))
    d = np.atleast_2d(d); i = np.atleast_2d(i)
    w = 1.0 / np.maximum(d, 1e-6) ** IDW_P
    return (w * z_tr[i]).sum(1) / w.sum(1)


def spherical(h, rng_, sill, nug):
    h = np.asarray(h, float)
    out = np.where(h <= rng_,
                   nug + (sill - nug) * (1.5 * h / rng_ - 0.5 * (h / rng_) ** 3),
                   sill)
    return np.where(h == 0, 0.0, out)


def _ok(xy_tr, z_tr, xy_te, vg):
    """Local ordinary kriging, batched.

    Three guards, all learned the hard way. With the fitted nugget at 0.00 the
    kriging matrix is singular wherever two training points coincide, and
    np.linalg.solve does not raise on a singular system -- it returns garbage,
    which produced a cross-validated RMSE of 5.7e13 m and looked like a real
    number. So: a nugget FLOOR, a regularised diagonal, and any solution whose
    weights are not finite or do not sum to ~1 is replaced by inverse distance.

    Systems are stacked and solved in batches rather than one at a time. That is
    only ~1.2x on its own, but it removes the Python-level loop, which is what
    makes chunked parallel execution worth having at fine cell sizes.
    """
    rng_, sill, nug = vg
    nug = max(nug, 0.01 * sill)
    K = min(OK_K, len(xy_tr))
    _, idx = cKDTree(xy_tr).query(xy_te, k=K)
    idx = np.atleast_2d(idx)
    out = np.empty(len(xy_te))
    for a in range(0, len(xy_te), OK_BATCH):
        b = slice(a, min(a + OK_BATCH, len(xy_te)))
        ii = idx[b]
        P = xy_tr[ii]                                   # (n, K, 2)
        n = P.shape[0]
        A = np.zeros((n, K + 1, K + 1))
        A[:, :K, :K] = spherical(
            np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1),
            rng_, sill, nug)
        A[:, :K, K] = 1.0
        A[:, K, :K] = 1.0
        A[:, np.arange(K), np.arange(K)] += 1e-8 * sill
        rhs = np.zeros((n, K + 1))
        dt = np.linalg.norm(P - xy_te[b][:, None, :], axis=-1)
        rhs[:, :K] = spherical(dt, rng_, sill, nug)
        rhs[:, K] = 1.0
        try:
            # NumPy 2 no longer infers "stack of vectors" when b.ndim ==
            # a.ndim - 1; it reads rhs as a matrix and the solve fails on the
            # core dimensions. An explicit trailing axis restores the intent.
            w = np.linalg.solve(A, rhs[:, :, None])[:, :K, 0]
        except np.linalg.LinAlgError:
            w = np.full((n, K), np.nan)
        # The Lagrange row FORCES sum(w) = 1 even on a singular system, so the
        # sum check alone cannot detect a degenerate neighbourhood: the weights
        # come back summing to 1 with magnitudes in the thousands. In hist24
        # that produced a leave-one-contour-out RMSE of 224.8 m against 3.6 m
        # at a neighbouring densification. Ordinary-kriging weights on a
        # well-posed system stay of order unity.
        bad = (~np.isfinite(w).all(1) | (np.abs(w.sum(1) - 1) > 0.05)
               | (np.abs(w).max(1) > 5.0))
        if bad.any():
            wi = 1.0 / np.maximum(dt[bad], 1e-6) ** 2
            w[bad] = wi / wi.sum(1, keepdims=True)
        out[b] = (w * z_tr[ii]).sum(1)
    return out


def _rbf(xy_tr, z_tr, xy_te, vg=None):
    f = RBFInterpolator(xy_tr, z_tr, neighbors=min(RBF_K, len(xy_tr)),
                        kernel="thin_plate_spline", smoothing=1.0)
    return f(xy_te)


def _linear(xy_tr, z_tr, xy_te, vg=None):
    v = griddata(xy_tr, z_tr, xy_te, method="linear")
    bad = ~np.isfinite(v)
    if bad.any():
        v[bad] = griddata(xy_tr, z_tr, xy_te[bad], method="nearest")
    return v


METHODS = {"IDW": _idw, "OK": _ok, "RBF": _rbf, "LINEAR": _linear}


# ----------------------------------------------------------- parallel glue --
def _task_cv(arg):
    """One (scheme, method, fold): predict the held-out soundings."""
    scheme, method, fold = arg
    te = _S["folds"][scheme] == fold
    if te.sum() == 0 or (~te).sum() < 50:
        return scheme, method, np.where(te)[0], np.array([])
    xy, z = _S["xy"], _S["z"]
    pred = METHODS[method](xy[~te], z[~te], xy[te], _S["vg"])
    return scheme, method, np.where(te)[0], pred - z[te]


def _task_mask(arg):
    """One ROW STRIP of the point-in-polygon test.

    Strips, not flat slabs: at 30 m the full meshgrid is 19.3 M cells and two
    float64 coordinate arrays would be 310 MB before any surface exists, on a
    machine with 12 GB free. Each worker builds only its own strip.
    """
    r0, r1 = arg
    GX, GY = np.meshgrid(_S["gx"], _S["gy"][r0:r1])
    return r0, shapely.contains_xy(_S["fp"], GX.ravel(), GY.ravel())


def _task_grid(arg):
    """One (variant, method, chunk): predict a slab of grid targets."""
    var, method, lo, hi = arg
    xt, zt = _S["train"][var]
    return var, method, lo, hi, METHODS[method](xt, zt, _S["tgt"][lo:hi],
                                                _S["vg"])


def run(tasks, fn, jobs, label):
    """Map fn over tasks, in processes when jobs > 1."""
    if jobs <= 1:
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        raise SystemExit("--jobs relies on the fork start method to share the "
                         "sounding arrays without pickling them")
    print(f"  [{label}: {len(tasks)} tasks over {jobs} processes]")
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


ISLAND_MASK = ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif"


def _island_flags(gx, gy):
    """True where p64 says the cell is an island, flat in the same (y-slow, x-fast, gy ascending) order as `inside`.

    Sampled NEAREST at any cell size: the mask is a classification, and FABDEM's native sample is 21 x 31 m, so the
    30 m mask must never be resampled by averaging. Returns None when the mask has not been built yet.
    """
    if not ISLAND_MASK.exists():
        return None
    import rasterio
    with rasterio.open(ISLAND_MASK) as ds:
        arr = ds.read(1); t = ds.transform
    col = np.floor((np.asarray(gx) - t.c) / t.a).astype(np.int64)
    row = np.floor((np.asarray(gy) - t.f) / t.e).astype(np.int64)
    vc = (col >= 0) & (col < arr.shape[1]); vr = (row >= 0) & (row < arr.shape[0])
    sub = np.zeros((len(gy), len(gx)), bool)
    if vr.any() and vc.any():
        sub[np.ix_(vr, vc)] = arr[np.ix_(row[vr], col[vc])] == 1
    return sub.ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cell", type=float, default=250.0)
    ap.add_argument("--verify", action="store_true",
                    help="confirm the parallel path reproduces the serial one")
    args = ap.parse_args()
    jobs = max(1, args.jobs)
    CELL = args.cell
    print(f"jobs={jobs}, cell={CELL:.0f} m"
          + ("  (BLAS pinned to 1 thread per process)" if jobs > 1 else ""))

    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                "kakhovka_soundings_evrf2019.parquet")
    # Exact duplicate coordinates make the kriging matrix singular. There are
    # only 5, and they are averaged rather than dropped so no location is lost.
    n0 = len(sd)
    sd = sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0)).groupby(
        ["_kx", "_ky"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), lon=("lon", "mean"),
        lat=("lat", "mean"), H_bed_evrf2019_m=("H_bed_evrf2019_m", "mean"))
    if len(sd) != n0:
        print(f"averaged {n0-len(sd)} exact duplicate coordinates "
              f"({n0:,} -> {len(sd):,})")
    xy = np.c_[sd.x.values, sd.y.values]
    z = sd.H_bed_evrf2019_m.values
    # THE DOMAIN COMES FROM THE REGISTRY, NOT FROM P20_reservoir_footprint.
    # This line was the source of the whole eastern-truncation chain. The P20
    # footprint stops at E 668,540 m while the authoritative domain reaches
    # E 678,000 m, so every surface this script wrote was cut by ~9.4 km on the
    # east and ~4.1 km on the north -- and hist15, hist17, hist20 and hist24 all
    # inherited that grid, which means published GeoTIFFs and validation tables
    # inherited it too.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    print(f"{len(sd):,} soundings, bed {z.min():.2f}..{z.max():.2f} m EVRF2019")
    print(f"domain: registry KAKHOVKA_RESERVOIR_CORE + "
          f"FORMER_RESERVOIR_TRANSITION, {fp.area/1e6:,.0f} km2")
    print(f"        E {fp.bounds[0]:,.0f}..{fp.bounds[2]:,.0f}  "
          f"N {fp.bounds[1]:,.0f}..{fp.bounds[3]:,.0f}")

    import skgstat as skg
    sub = RNG.choice(len(xy), min(1500, len(xy)), replace=False)
    V = skg.Variogram(xy[sub], z[sub], model="spherical", n_lags=20,
                      maxlag=0.35, normalize=False)
    vg = (float(V.parameters[0]), float(V.parameters[1]),
          float(V.parameters[2]) if len(V.parameters) > 2 else 0.0)
    print(f"\nspherical variogram on a subsample: range {vg[0]/1000:.1f} km, "
          f"sill {vg[1]:.2f}, nugget {vg[2]:.2f} m2")

    # ---- fold assignment: READ, never drawn --------------------------------
    # The partition used to come from RNG here, which made the CV score depend
    # on how many other draws this script happened to make first: hist18,
    # drawing five block scales instead of two, got 2.784 m where this script
    # got 2.761 m from the same seed. The partition is now DATA
    # (hist19_freeze_cv_folds.py), so the score is reproducible whatever the
    # call order. Row alignment is asserted, not assumed.
    fp_folds = (CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                       "hist14_cv_fold_assignments.parquet")
    if not fp_folds.exists():
        raise SystemExit(f"missing {fp_folds.name} -- run "
                         f"scripts/hist19_freeze_cv_folds.py first")
    fa = pd.read_parquet(fp_folds)
    if len(fa) != len(sd) or not (np.allclose(fa.x.values, sd.x.values)
                                  and np.allclose(fa.y.values, sd.y.values)):
        raise SystemExit(
            "fold file does not align with the soundings row-for-row. The "
            "soundings were probably regenerated; re-run hist19.")
    folds = {}
    for bk in BLOCK_KM_LIST:
        col = f"block_{bk:g}km_fold".replace(".", "p")
        folds[f"blocked{bk:.0f}km"] = fa[col].values.astype(int)
        print(f"spatial blocks: {bk:.0f} km -> {N_FOLDS} folds "
              f"(frozen, seed {int(fa.seed.iloc[0])})")
    folds["random"] = fa.random_fold.values.astype(int)

    _S.update(xy=xy, z=z, vg=vg, folds=folds)

    # ---- cross-validation --------------------------------------------------
    tasks = [(s, m, f) for s in folds for m in METHODS for f in range(N_FOLDS)]
    res = run(tasks, _task_cv, jobs, "CV")
    acc = {(s, m): np.full(len(xy), np.nan) for s in folds for m in METHODS}
    for s, m, ix, r in res:
        if len(r):
            acc[(s, m)][ix] = r
    rows = []
    for (s, m), r in acc.items():
        rr = r[np.isfinite(r)]
        rows.append({"scheme": s, "method": m, "n": len(rr),
                     "RMSE_m": float(np.sqrt((rr ** 2).mean())),
                     "MAE_m": float(np.abs(rr).mean()),
                     "bias_m": float(rr.mean()), "NMAD_m": nmad(rr)})
    # Label what each scheme is FOR, so a reader of the CSV cannot mistake the
    # 5 km stress test or the optimistic random holdout for the primary score.
    ROLE = {f"blocked{BLOCK_KM_LIST[0]:.0f}km":
            "PRIMARY - blocks matched to the ~357 m data spacing",
            f"blocked{BLOCK_KM_LIST[-1]:.0f}km":
            "stress test - blocks exceed the ~2 km variogram range, so this "
            "scores extrapolation, not interpolation",
            "random": "optimistic reference - survey lines put a near-twin of "
                      "each held-out point in the training set"}
    rows = [{**r, "role": ROLE[r["scheme"]]} for r in rows]
    cv = pd.DataFrame(rows).sort_values(["scheme", "method"])
    for r in cv.itertuples():
        print(f"  {r.scheme:<14}{r.method:<8}RMSE {r.RMSE_m:>6.3f}  "
              f"MAE {r.MAE_m:>6.3f}  bias {r.bias_m:>+7.3f}  "
              f"NMAD {r.NMAD_m:>6.3f}  (n={r.n:,})")
    cv.to_csv(CFG.TABLES / "hist14_interpolator_cv.csv", index=False)

    MAIN = f"blocked{BLOCK_KM_LIST[0]:.0f}km"
    bl = cv[cv.scheme == MAIN].set_index("method")
    hard = cv[cv.scheme == f"blocked{BLOCK_KM_LIST[-1]:.0f}km"].set_index("method")
    rd = cv[cv.scheme == "random"].set_index("method")
    print(f"\n  random holdout is optimistic by "
          f"{(bl.RMSE_m/rd.RMSE_m).median():.2f}x against 1 km blocks and "
          f"{(hard.RMSE_m/rd.RMSE_m).median():.2f}x against 5 km.")
    print(f"  The 5 km scheme holds out cells beyond the {vg[0]/1000:.1f} km variogram")
    print(f"  range, so it scores extrapolation and no method can pass it. The")
    print(f"  choice uses {MAIN}, matched to the ~357 m data spacing.")
    best = bl.RMSE_m.idxmin()
    print(f"\n  preferred surface: {best}  (blocked RMSE "
          f"{bl.loc[best,'RMSE_m']:.3f} m, bias {bl.loc[best,'bias_m']:+.3f} m)")
    print("  blocked RMSE ranking: "
          + ", ".join(f"{m} {bl.loc[m,'RMSE_m']:.2f}"
                      for m in bl.RMSE_m.sort_values().index))

    # ---- residual structure ------------------------------------------------
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    _, off, _, _ = SW.assign_chainage(sd.lon.values, sd.lat.values, ch, tree=tree)
    r_best = acc[(MAIN, best)]
    print(f"\n  residual structure of {best} ({MAIN}), by distance from the channel:")
    bb = pd.DataFrame({"off": off, "r": r_best})
    for i, g in bb.groupby(pd.cut(bb.off, [0, 1, 3, 6, 10, 100]), observed=True):
        print(f"    {str(i):<12}n={len(g):>5,}  bias {g.r.mean():+.3f}  "
              f"RMSE {np.sqrt((g.r**2).mean()):.3f} m")
    print("  residuals are largest near the channel, where the bed is steepest --")
    print("  a resolution limit, not a bias in the method.")

    # ---- grid --------------------------------------------------------------
    x0, y0, x1, y1 = fp.bounds
    gx = np.arange(x0, x1 + CELL, CELL)
    gy = np.arange(y0, y1 + CELL, CELL)
    # refuse to write a grid narrower than the domain, ever again
    SD.assert_covers(fp, gx, gy, what=f"hist14 {CELL:.0f} m grid", cell=CELL)
    # shapely.contains_xy, NOT GeoSeries.within: the footprint has 34,831
    # vertices and the geopandas path takes ~1,805 s for this grid. That one
    # line was the original runtime of this script -- the kriging it looked like
    # is 3 s.
    #
    # contains_xy is still the SERIAL bottleneck once the grid is refined: 14 s
    # at 250 m and 87 s at 100 m, which is why --jobs plateaued at ~1.5x. It is
    # therefore chunked across processes too. rasterio.features.rasterize would
    # be ~2000x faster, but it disagrees with the exact test on 0.06-0.09 % of
    # cells, all on the boundary -- precisely where the exposure fraction is
    # sensitive -- so it is NOT used. Exactness is kept and parallelised.
    ncell = len(gx) * len(gy)
    _S.update(fp=fp, gx=gx, gy=gy)
    rows_per = max(1, int(1_500_000 / len(gx)))
    mtasks = [(r0, min(r0 + rows_per, len(gy)))
              for r0 in range(0, len(gy), rows_per)]
    parts = dict((r0, v) for r0, v in
                 run(mtasks, _task_mask, jobs if len(mtasks) > 1 else 1,
                     "domain mask"))
    inside = np.concatenate([parts[r0] for r0, _ in mtasks])
    # ---- ISLANDS ARE A CONSTRAINT, NOT A TARGET ---------------------------
    # The registry subzones are the WATER BODY's outline, but they enclose land
    # that was never under water -- Khortytsia above all. Kriging a bed under
    # dry land produced the -22 m artefacts p62/p63 measured in the seamless
    # DEM. The bed reconstruction therefore declines to predict there. Nothing
    # is deleted from the terrain: those cells keep their own FABDEM elevation
    # in every product that represents terrain (p55 source class 3), and the
    # class is recorded as provenance, never as a hole. The mask is sampled
    # NEAREST, because it is a decision and not a height.
    n_all = int(inside.sum())
    isl = _island_flags(gx, gy)
    if isl is not None:
        inside &= ~isl
        n_isl = n_all - int(inside.sum())
        print(f"island constraint: {n_isl:,} cells excluded from the kriging "
              f"targets ({n_isl*(CELL/1e3)**2:,.2f} km2) -- {ISLAND_MASK.name}")
    else:
        print(f"island constraint: {ISLAND_MASK} NOT FOUND -- kriging the whole "
              f"registry domain (run scripts/p64_island_mask.py first)")
    ins_idx = np.flatnonzero(inside)
    # Target coordinates straight from the flat index -- no full meshgrid.
    tgt = np.c_[gx[ins_idx % len(gx)], gy[ins_idx // len(gx)]]
    print(f"\ngrid {len(gx)}x{len(gy)} at {CELL:.0f} m, {len(ins_idx):,} cells "
          f"inside ({len(ins_idx)*(CELL/1e3)**2:,.0f} km2)")

    bnd = fp.boundary
    n_sh = max(int(bnd.length / SHORE_STEP_M), 100)
    sh = np.array([bnd.interpolate(t, normalized=True).coords[0]
                   for t in np.linspace(0, 1, n_sh, endpoint=False)])
    print(f"shoreline pseudo-points: {len(sh):,} along {bnd.length/1e3:,.0f} km "
          f"of outline at {SHORE_STEP_M:.0f} m spacing")

    train = {v: ((xy, z) if lev is None
                 else (np.vstack([xy, sh]),
                       np.concatenate([z, np.full(len(sh), lev)])))
             for v, lev in SHORE_VARIANTS.items()}
    _S.update(train=train, tgt=tgt)
    nt = len(tgt)

    gtasks = [(v, m, lo, min(lo + CHUNK, nt))
              for v in SHORE_VARIANTS for m in METHODS
              for lo in range(0, nt, CHUNK)]
    gres = run(gtasks, _task_grid, jobs, "grid")
    # INSIDE-ONLY, float32. Full float64 grids would be 1.86 GB at 30 m before
    # compression, on a machine with 12 GB free; every consumer immediately
    # indexes by `inside` anyway, so the mask plus flat arrays is both smaller
    # and a closer fit to how the data are used. float32 keeps 0.001 m, far
    # finer than the 2.76 m cross-validated RMSE.
    store = {f"surf_{v}_{m}": np.full(nt, np.nan, np.float32)
             for v in SHORE_VARIANTS for m in METHODS}
    for v, m, lo, hi, val in gres:
        store[f"surf_{v}_{m}"][lo:hi] = val

    for var, lev in SHORE_VARIANTS.items():
        print(f"\n  constraint '{var}'"
              + (f" at {lev:.2f} m EVRF2019" if lev else " (soundings only)")
              + f", {len(train[var][0]):,} training points")
        for m in METHODS:
            g2 = store[f"surf_{var}_{m}"]
            print(f"    {m:<8}{np.nanmin(g2):+7.2f} .. {np.nanmax(g2):+7.2f} m, "
                  f"mean {np.nanmean(g2):+.2f}")
    print(f"\n  effect of the constraint on the {best} surface:")
    for var in SHORE_VARIANTS:
        g2 = store[f"surf_{var}_{best}"]
        g2 = g2[np.isfinite(g2)]
        print(f"    {var:<6} mean {g2.mean():+.2f} m   cells above GMO "
              f"{GMO_EVRF:.3f} m: {100*(g2 >= GMO_EVRF).mean():.1f} %")
    print("  That last column is the quantity hist15 reports, so the constraint")
    print("  is not cosmetic -- it moves the answer.")

    # ---- optional: parallel == serial --------------------------------------
    if args.verify:
        print("\n  [verify] recomputing 3 CV tasks and 2 grid chunks serially")
        for t in [(MAIN, "OK", 0), (MAIN, "IDW", 3), ("random", "LINEAR", 7)]:
            s_, m_, ix, r = _task_cv(t)
            assert np.allclose(acc[(s_, m_)][ix], r, equal_nan=True), t
        for t in gtasks[:2]:
            v, m, lo, hi, val = _task_grid(t)
            # `store` holds inside-only flat arrays, indexed by position in
            # ins_idx -- NOT full grids. lo:hi is already that slice; going
            # through ins_idx again would index the array with grid offsets.
            got = store[f"surf_{v}_{m}"][lo:hi]
            assert np.allclose(got, val, equal_nan=True), t
        print("  [verify] identical -- results do not depend on --jobs")

    # bulk product -> bulk drive, so the consumers read the file this run
    # wrote. Writing it to the repo disk while hist15/17/20/24 read the bulk
    # copy is how a stale truncated surface survived a regeneration.
    out_npz = (CFG.BULK_ROOT / "data_swot/processed/bathymetry"
               / f"kakhovka_bed_surface_{CELL:.0f}m.npz")
    # The npz is cell-tagged, so the table and figure must be too, or a fine
    # grid silently overwrites the canonical 250 m outputs and nothing in the
    # filename says which resolution is on disk. 250 m keeps the bare name so
    # existing references to it stay valid.
    TAG = "" if abs(CELL - 250.0) < 1 else f"_{CELL:.0f}m"
    np.savez_compressed(
        out_npz,
        gx=gx, gy=gy, ins_idx=ins_idx.astype(np.int64),
        n_inside=len(ins_idx), preferred=best,
        cell_m=CELL, crs=str(CFG.CRS_METRIC),
        shore_variants=np.array(list(SHORE_VARIANTS), dtype=object),
        shore_levels=np.array([SHORE_VARIANTS[k] or np.nan
                               for k in SHORE_VARIANTS]),
        **store)
    pd.DataFrame([{"variant": v, "method": m, "cell_m": CELL,
                   "cells": int(np.isfinite(store[f"surf_{v}_{m}"]).sum()),
                   "mean_m": float(np.nanmean(store[f"surf_{v}_{m}"])),
                   "min_m": float(np.nanmin(store[f"surf_{v}_{m}"])),
                   "max_m": float(np.nanmax(store[f"surf_{v}_{m}"])),
                   "preferred": m == best}
                  for v in SHORE_VARIANTS for m in METHODS]).to_csv(
        CFG.TABLES / f"hist14_surface_summary{TAG}.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ext = [gx[0] / 1e3, gx[-1] / 1e3, gy[0] / 1e3, gy[-1] / 1e3]
    # Rebuild a display raster, strided so a 19 M-cell grid does not go into
    # imshow at full size.
    st = max(1, int(np.ceil(max(len(gx), len(gy)) / 1400)))

    def as_raster(flat):
        g = np.full(len(gx) * len(gy), np.nan, np.float32)
        g[ins_idx] = flat
        return g.reshape(len(gy), len(gx))[::st, ::st]

    for a, m in zip(np.ravel(ax)[:4], METHODS):
        im = a.imshow(as_raster(store[f"surf_epoch_{m}"]), origin="lower",
                      extent=ext,
                      cmap="viridis", vmin=-8, vmax=17)
        a.set_title(f"{m}   blocked RMSE {bl.loc[m,'RMSE_m']:.2f} m"
                    + ("   <- preferred" if m == best else ""),
                    fontsize=10.5, loc="left")
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=np.ravel(ax)[:4].tolist(), shrink=0.6,
                 label="bed elevation (m, EVRF2019), shoreline-constrained")

    a = np.ravel(ax)[4]
    w = 0.27
    yy = np.arange(len(bl))
    for k, (tab, lab, c) in enumerate(
            [(bl, f"blocked {BLOCK_KM_LIST[0]:.0f} km", BLUE),
             (hard, f"blocked {BLOCK_KM_LIST[1]:.0f} km", AMBER),
             (rd, "random", GREY)]):
        a.barh(yy + (k - 1) * w, tab.loc[bl.index].RMSE_m.values, height=w,
               color=c, label=lab)
    a.set_yticks(yy); a.set_yticklabels(bl.index, fontsize=9)
    a.set_xlabel("cross-validated RMSE (m)")
    a.legend(fontsize=8)
    a.set_title("Block size decides the ranking", fontsize=10.5, loc="left")
    a.grid(axis="x", alpha=0.25)

    a = np.ravel(ax)[5]
    a.scatter(off, r_best, s=2, color=GREY, alpha=0.25, lw=0)
    q = bb.groupby(pd.cut(bb.off, np.arange(0, 26, 1)), observed=True).r
    mid = [i.mid for i in q.median().index]
    a.plot(mid, q.median(), "-", color=RED, lw=2.2, label="median")
    a.plot(mid, q.apply(lambda v: np.sqrt((v ** 2).mean())), "--", color=INK,
           lw=1.8, label="RMSE")
    a.axhline(0, color=INK, lw=1)
    a.set_xlim(0, 25); a.set_ylim(-12, 12)
    a.set_xlabel("distance from the channel line (km)")
    a.set_ylabel(f"{best} residual (m)")
    a.legend(fontsize=8.4)
    a.set_title("Residuals concentrate where the bed is steepest",
                fontsize=10.5, loc="left")

    fig.suptitle("V12 · Bed surface interpolation, chosen by spatially blocked "
                 "cross-validation", fontsize=12.5, y=1.0)
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V12_bed_surface_validation{TAG}.{e}", dpi=175,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/f'V12_bed_surface_validation{TAG}.png'}")
    print(f"-> {CFG.TABLES/f'hist14_surface_summary{TAG}.csv'}")
    # The cross-validation is on the sounding points, not on the grid, so it is
    # the same file whatever --cell is: untagged on purpose.
    print(f"-> {CFG.TABLES/'hist14_interpolator_cv.csv'}  (cell-independent)")
    print(f"-> {out_npz}")


if __name__ == "__main__":
    main()
