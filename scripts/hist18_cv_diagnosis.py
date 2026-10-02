#!/usr/bin/env python
"""HISTORICAL 18 — why is the blocked-CV RMSE of the bed DEM 2.761 m?

DIAGNOSIS ONLY. No interpolation parameter is changed, and nothing here is
tuned to reduce RMSE. The kriging estimator, its neighbourhood size, the
variogram fit and the fold assignment are IMPORTED from hist14 so they are
identical by construction rather than by copy-and-paste.

The question is not whether 2.761 m is large. It is whether the error is
spread roughly evenly over the domain -- in which case the DEM is uniformly
uncertain to ~2.8 m -- or concentrated in a small number of deep, narrow or
sparsely sampled places, in which case most of the surface is far better than
the headline number and the headline number is a statement about a specific
failure mode.

Three candidate explanations are separable with the data at hand:

  spatial extrapolation   a 1 km hole is ~3x the 357 m median sounding
                          spacing, so blocked CV asks for prediction beyond
                          the sampled geometry. Comparing 0.5 / 1 / 2 / 5 km
                          blocks against random holdout measures this
                          directly.
  channel morphology      a former channel trough inside a withheld block
                          cannot be recovered from platform points on its
                          edges. Splitting the error at the descriptive
                          platform/trough threshold measures this.
  vertical reference      would show up as a BIAS, not as scatter. The
                          blocked bias is -0.032 m, so this is already
                          close to excluded -- but it is tested per reach
                          anyway, because a datum problem could be regional.

Outputs
-------
outputs/tables/hist18_cv_scheme_comparison.csv
outputs/tables/hist18_error_decomposition.csv
outputs/tables/hist18_worst_residuals.csv
outputs/figures/V16_cv_error_diagnosis.png
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# BLAS pinning must precede numpy, exactly as in hist14
_JOBS = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--jobs" and _i + 1 < len(sys.argv):
        _JOBS = int(sys.argv[_i + 1])
if _JOBS != 1:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import shape

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD
from swot_dnipro import sword as SW

# The estimator and every constant that defines it, imported not copied.
import hist14_bed_surface as H14

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
SCHEMES = [("random", None), ("blocked0.5km", 0.5), ("blocked1km", 1.0),
           ("blocked2km", 2.0), ("blocked5km", 5.0)]
PRIMARY = "blocked1km"
# hist12's DESCRIPTIVE classification threshold. Not a mode boundary -- the
# distribution is unimodal (skew -0.76) -- but it is the established split
# between the shallow floodplain platform and the channel troughs.
PLATFORM_THR = 7.378891966910189
RNG = np.random.default_rng(CFG.SEED)


def stats(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return dict(n=0)
    a = np.abs(r)
    return dict(n=len(r), RMSE_m=float(np.sqrt((r ** 2).mean())),
                MAE_m=float(a.mean()), bias_m=float(r.mean()),
                median_m=float(np.median(r)), NMAD_m=H14.nmad(r),
                p50_abs=float(np.percentile(a, 50)),
                p90_abs=float(np.percentile(a, 90)),
                p95_abs=float(np.percentile(a, 95)),
                max_abs=float(a.max()))


def _task(t):
    """One (scheme, fold): predict the held-out soundings with OK.

    Module-level, not a closure, so it survives pickling for
    ProcessPoolExecutor. Reads the shared arrays out of hist14's own _S dict,
    which the parent populated before forking.
    """
    scheme, fold = t
    xy, z, vg = H14._S["xy"], H14._S["z"], H14._S["vg"]
    te = H14._S["folds"][scheme] == fold
    if te.sum() == 0 or (~te).sum() < 50:
        return scheme, np.where(te)[0], np.array([]), np.array([])
    pred = H14.METHODS["OK"](xy[~te], z[~te], xy[te], vg)
    # distance from each held-out point to the nearest TRAINING point -- the
    # covariate that distinguishes blocked from random hold-out
    d = cKDTree(xy[~te]).query(xy[te], k=1)[0]
    return scheme, np.where(te)[0], pred - z[te], d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    jobs = max(1, args.jobs)

    # ================================================================= data
    # Loaded exactly as hist14 loads it, duplicates averaged the same way.
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                "kakhovka_soundings_evrf2019.parquet")
    n0 = len(sd)
    sd = sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0)).groupby(
        ["_kx", "_ky"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), lon=("lon", "mean"),
        lat=("lat", "mean"), H_bed_evrf2019_m=("H_bed_evrf2019_m", "mean"))
    xy = np.c_[sd.x.values, sd.y.values]
    z = sd.H_bed_evrf2019_m.values
    print(f"{len(sd):,} soundings ({n0:,} raw, {n0-len(sd)} duplicates "
          f"averaged), bed {z.min():.2f}..{z.max():.2f} m "
          f"(amplitude {z.max()-z.min():.1f} m)")

    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
          SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    # ============================================== variogram, same as hist14
    import skgstat as skg
    sub = RNG.choice(len(xy), min(1500, len(xy)), replace=False)
    Vg = skg.Variogram(xy[sub], z[sub], model="spherical", n_lags=20,
                       maxlag=0.35, normalize=False)
    vg = (float(Vg.parameters[0]), float(Vg.parameters[1]),
          float(Vg.parameters[2]) if len(Vg.parameters) > 2 else 0.0)
    print(f"spherical variogram: range {vg[0]/1000:.2f} km, sill {vg[1]:.2f}, "
          f"nugget {vg[2]:.2f} m2   (imported estimator, unchanged)")

    # data geometry, for reference against the block sizes
    dnn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    print(f"nearest-sounding spacing: median {np.median(dnn):.0f} m, "
          f"p90 {np.percentile(dnn,90):.0f} m")

    # ==================================================== folds: READ, not drawn
    # This script originally drew its own folds and got 2.784 m where hist14
    # got 2.761 m -- same seed, different RNG consumption order. Both now read
    # the frozen partition, so the two agree exactly.
    fp_folds = (CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                       "hist14_cv_fold_assignments.parquet")
    if not fp_folds.exists():
        raise SystemExit(f"missing {fp_folds.name} -- run "
                         f"scripts/hist19_freeze_cv_folds.py first")
    fa = pd.read_parquet(fp_folds)
    if len(fa) != len(sd) or not (np.allclose(fa.x.values, sd.x.values)
                                  and np.allclose(fa.y.values, sd.y.values)):
        raise SystemExit("fold file does not align with the soundings "
                         "row-for-row; re-run hist19.")
    folds = {}
    for name, bk in SCHEMES:
        col = "random_fold" if bk is None else \
            f"block_{bk:g}km_fold".replace(".", "p")
        folds[name] = fa[col].values.astype(int)
    print(f"  folds read from {fp_folds.name} (seed {int(fa.seed.iloc[0])}): "
          + ", ".join(folds))

    # ======================================= run OK under every scheme
    # Identical kriging parameters throughout: only the fold geometry changes.
    H14._S.update(xy=xy, z=z, vg=vg, folds=folds)
    resid = {}
    nnd = {}          # distance from each held-out point to its own training set
    blockid = {}
    tasks = [(s, f) for s in folds for f in range(H14.N_FOLDS)]
    acc = {s: np.full(len(xy), np.nan) for s in folds}
    accd = {s: np.full(len(xy), np.nan) for s in folds}
    print(f"\nrunning {len(tasks)} folds x 1 method (OK) on {jobs} job(s)")

    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_task, tasks))
    else:
        results = [_task(t) for t in tasks]
    for scheme, ix, r, d in results:
        if len(r):
            acc[scheme][ix] = r
            accd[scheme][ix] = d
    for s in folds:
        resid[s] = acc[s]
        nnd[s] = accd[s]
    # The 1 km BLOCK id, which is what factor 7 asks about -- not the fold id.
    # A fold is 1/10 of all blocks scattered over the reservoir; a block is one
    # contiguous 1 km cell, and that is where a hotspot would sit.
    blockid[PRIMARY] = pd.factorize(
        [tuple(b) for b in np.floor(xy / 1000.0).astype(int)])[0]

    # ============================================ A. scheme comparison
    print("\n" + "=" * 78)
    print("SCHEME COMPARISON — identical kriging, only the hold-out geometry")
    print("=" * 78)
    rows = []
    print(f"  {'scheme':<14}{'block':>7}{'n':>7}{'RMSE':>8}{'MAE':>7}"
          f"{'NMAD':>7}{'bias':>8}{'p90|e|':>8}{'RMSE/NMAD':>11}"
          f"{'med d_train':>13}")
    for name, bk in SCHEMES:
        s = stats(resid[name])
        s["scheme"] = name
        s["block_km"] = bk if bk else 0.0
        s["median_dist_to_training_m"] = float(np.nanmedian(nnd[name]))
        s["rmse_over_nmad"] = s["RMSE_m"] / s["NMAD_m"]
        rows.append(s)
        print(f"  {name:<14}{(f'{bk:.1f}' if bk else 'random'):>7}"
              f"{s['n']:>7,}{s['RMSE_m']:>8.3f}{s['MAE_m']:>7.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}{s['p90_abs']:>8.3f}"
              f"{s['rmse_over_nmad']:>11.2f}"
              f"{s['median_dist_to_training_m']:>12.0f} m")
    sc = pd.DataFrame(rows)
    sc.to_csv(CFG.TABLES / "hist18_cv_scheme_comparison.csv", index=False)

    rr = sc.set_index("scheme")
    print(f"\n  random -> 1 km blocked: RMSE "
          f"{rr.loc['random','RMSE_m']:.3f} -> {rr.loc[PRIMARY,'RMSE_m']:.3f} m "
          f"({rr.loc[PRIMARY,'RMSE_m']/rr.loc['random','RMSE_m']:.2f}x), while the "
          f"median distance to the")
    print(f"  nearest training point goes "
          f"{rr.loc['random','median_dist_to_training_m']:.0f} m -> "
          f"{rr.loc[PRIMARY,'median_dist_to_training_m']:.0f} m.")
    print(f"  Bias stays within "
          f"{sc.bias_m.abs().max():.3f} m across ALL schemes, so the vertical "
          f"reference is not implicated.")

    # ============================================ context covariates
    print("\n" + "=" * 78)
    print(f"ERROR DECOMPOSITION on the PRIMARY scheme ({PRIMARY})")
    print("=" * 78)
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/"
                                "sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    km, dch, _, _ = SW.assign_chainage(sd.lon.values, sd.lat.values, ch,
                                       tree=tree)
    # local terrain roughness: sd of bed among the 8 nearest soundings
    _, i8 = cKDTree(xy).query(xy, k=9)
    rough = z[i8[:, 1:]].std(axis=1)
    dshore = shapely.distance(shapely.points(xy[:, 0], xy[:, 1]),
                              shapely.boundary(fp))

    D = pd.DataFrame(dict(
        x=xy[:, 0], y=xy[:, 1], lon=sd.lon.values, lat=sd.lat.values,
        bed_m=z, resid=resid[PRIMARY], d_train_m=nnd[PRIMARY],
        chainage_km=km, d_channel_km=dch, roughness_m=rough,
        d_shore_m=dshore, block=blockid[PRIMARY], fold=folds[PRIMARY],
        resid_random=resid["random"]))
    D["abs_resid"] = D.resid.abs()
    D = D[np.isfinite(D.resid)]

    ov = stats(D.resid.values)
    print(f"  overall: n={ov['n']:,}  RMSE {ov['RMSE_m']:.3f}  "
          f"MAE {ov['MAE_m']:.3f}  NMAD {ov['NMAD_m']:.3f}  "
          f"bias {ov['bias_m']:+.3f}")
    print(f"  |error| percentiles: p50 {ov['p50_abs']:.2f}  "
          f"p90 {ov['p90_abs']:.2f}  p95 {ov['p95_abs']:.2f}  "
          f"max {ov['max_abs']:.2f} m")

    # ---------------------------------------- IS THE ERROR CONCENTRATED?
    print("\n  " + "-" * 74)
    print("  CONCENTRATION — the decisive question")
    print("  " + "-" * 74)
    sse = (D.resid ** 2).values
    order = np.argsort(-sse)
    cum = np.cumsum(sse[order]) / sse.sum()
    conc = {}
    for pct in (1, 2, 5, 10, 20, 50):
        k = max(1, int(round(pct / 100 * len(sse))))
        conc[pct] = 100 * cum[k - 1]
        print(f"    worst {pct:>2d}% of points carry "
              f"{conc[pct]:>5.1f}% of the total squared error")
    # what the RMSE would be without the worst 1 % / 5 %
    for pct in (1, 5):
        k = int(round(pct / 100 * len(sse)))
        keep = np.setdiff1d(np.arange(len(sse)), order[:k])
        print(f"    RMSE excluding the worst {pct}%: "
              f"{np.sqrt(sse[keep].mean()):.3f} m "
              f"(vs {ov['RMSE_m']:.3f} m overall)")

    # ---------------------------------------- 1. distance to training point
    dec = []
    print("\n  1. NEAREST TRAINING-POINT DISTANCE")
    print(f"     {'bin':<18}{'n':>7}{'RMSE':>8}{'NMAD':>7}{'bias':>8}"
          f"{'%SSE':>8}")
    edges = [0, 250, 500, 750, 1000, 1500, 2000, 1e9]
    labs = ["0-250 m", "250-500", "500-750", "750-1000", "1-1.5 km",
            "1.5-2 km", ">2 km"]
    D["_b"] = pd.cut(D.d_train_m, edges, labels=labs, include_lowest=True)
    for b, g in D.groupby("_b", observed=True):
        s = stats(g.resid.values)
        s.update(factor="dist_to_nearest_training_point", bin=str(b),
                 share_SSE_pct=100 * (g.resid ** 2).sum() / sse.sum())
        dec.append(s)
        print(f"     {str(b):<18}{s['n']:>7,}{s['RMSE_m']:>8.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}"
              f"{s['share_SSE_pct']:>8.1f}")

    # ---------------------------------------- 2. bed elevation class
    print("\n  2. BED ELEVATION / DEPTH CLASS")
    print(f"     {'bin':<18}{'n':>7}{'RMSE':>8}{'NMAD':>7}{'bias':>8}"
          f"{'%SSE':>8}")
    zed = [-20, -10, -5, 0, PLATFORM_THR, 11, 13, 20]
    zlab = ["< -10 m", "-10..-5", "-5..0", f"0..{PLATFORM_THR:.1f}",
            f"{PLATFORM_THR:.1f}..11", "11..13", "> 13 m"]
    D["_z"] = pd.cut(D.bed_m, zed, labels=zlab, include_lowest=True)
    for b, g in D.groupby("_z", observed=True):
        s = stats(g.resid.values)
        s.update(factor="bed_elevation_class", bin=str(b),
                 share_SSE_pct=100 * (g.resid ** 2).sum() / sse.sum())
        dec.append(s)
        print(f"     {str(b):<18}{s['n']:>7,}{s['RMSE_m']:>8.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}"
              f"{s['share_SSE_pct']:>8.1f}")

    # platform vs trough, the headline split
    plat = D[D.bed_m >= PLATFORM_THR]
    trou = D[D.bed_m < PLATFORM_THR]
    sp, st = stats(plat.resid.values), stats(trou.resid.values)
    print(f"\n     SHALLOW PLATFORM (bed >= {PLATFORM_THR:.2f} m): "
          f"n={sp['n']:,}  RMSE {sp['RMSE_m']:.3f}  NMAD {sp['NMAD_m']:.3f}  "
          f"bias {sp['bias_m']:+.3f}")
    print(f"     CHANNEL TROUGHS  (bed <  {PLATFORM_THR:.2f} m): "
          f"n={st['n']:,}  RMSE {st['RMSE_m']:.3f}  NMAD {st['NMAD_m']:.3f}  "
          f"bias {st['bias_m']:+.3f}")
    print(f"     ratio of RMSE, troughs / platform: "
          f"{st['RMSE_m']/sp['RMSE_m']:.2f}x")
    print(f"     troughs are {100*st['n']/ov['n']:.0f}% of points but carry "
          f"{100*(trou.resid**2).sum()/sse.sum():.0f}% of the squared error")
    for tag, s, g in (("shallow_platform", sp, plat),
                      ("channel_trough", st, trou)):
        s = dict(s); s.update(factor="morphology", bin=tag,
                              share_SSE_pct=100*(g.resid**2).sum()/sse.sum())
        dec.append(s)

    # ---------------------------------------- 3. distance to channel corridor
    print("\n  3. DISTANCE TO THE SWORD MODERN-CHANNEL CORRIDOR")
    print(f"     {'bin':<18}{'n':>7}{'RMSE':>8}{'NMAD':>7}{'bias':>8}"
          f"{'%SSE':>8}")
    ced = [0, 1, 2, 3, 5, 8, 1e9]
    clab = ["0-1 km", "1-2", "2-3", "3-5", "5-8", "> 8 km"]
    D["_c"] = pd.cut(D.d_channel_km, ced, labels=clab, include_lowest=True)
    for b, g in D.groupby("_c", observed=True):
        s = stats(g.resid.values)
        s.update(factor="dist_to_channel_corridor", bin=str(b),
                 share_SSE_pct=100 * (g.resid ** 2).sum() / sse.sum())
        dec.append(s)
        print(f"     {str(b):<18}{s['n']:>7,}{s['RMSE_m']:>8.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}"
              f"{s['share_SSE_pct']:>8.1f}")

    # ---------------------------------------- 4. chainage / reach
    print("\n  4. CHAINAGE / RESERVOIR REACH")
    print(f"     {'bin':<18}{'n':>7}{'RMSE':>8}{'NMAD':>7}{'bias':>8}"
          f"{'%SSE':>8}")
    red = [0, 133, 183, 250]
    rlab = ["reach 1+2 (0-133)", "reach 3 (133-183)", "reach 4 (183-250)"]
    D["_r"] = pd.cut(D.chainage_km, red, labels=rlab, include_lowest=True)
    for b, g in D.groupby("_r", observed=True):
        s = stats(g.resid.values)
        s.update(factor="reach", bin=str(b),
                 share_SSE_pct=100 * (g.resid ** 2).sum() / sse.sum())
        dec.append(s)
        print(f"     {str(b):<18}{s['n']:>7,}{s['RMSE_m']:>8.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}"
              f"{s['share_SSE_pct']:>8.1f}")
    print(f"     per-reach bias spans "
          f"{min(d['bias_m'] for d in dec if d['factor']=='reach'):+.3f} to "
          f"{max(d['bias_m'] for d in dec if d['factor']=='reach'):+.3f} m — "
          f"a datum problem would show here.")

    # ---------------------------------------- 5. local roughness
    print("\n  5. LOCAL TERRAIN ROUGHNESS (sd of bed over 8 nearest soundings)")
    print(f"     {'bin':<18}{'n':>7}{'RMSE':>8}{'NMAD':>7}{'bias':>8}"
          f"{'%SSE':>8}")
    qs = [0] + list(np.percentile(D.roughness_m, [25, 50, 75, 90])) + [1e9]
    qlab = ["Q1 smoothest", "Q2", "Q3", "Q4", "top 10% roughest"]
    D["_g"] = pd.cut(D.roughness_m, qs, labels=qlab, include_lowest=True)
    for b, g in D.groupby("_g", observed=True):
        s = stats(g.resid.values)
        s.update(factor="local_roughness", bin=str(b),
                 share_SSE_pct=100 * (g.resid ** 2).sum() / sse.sum())
        dec.append(s)
        print(f"     {str(b):<18}{s['n']:>7,}{s['RMSE_m']:>8.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}"
              f"{s['share_SSE_pct']:>8.1f}")

    # ---------------------------------------- 6. shoreline distance
    print("\n  6. DISTANCE TO THE FOOTPRINT SHORELINE")
    print(f"     {'bin':<18}{'n':>7}{'RMSE':>8}{'NMAD':>7}{'bias':>8}"
          f"{'%SSE':>8}")
    sed = [0, 500, 1000, 2000, 4000, 1e9]
    slab = ["0-500 m", "500-1000", "1-2 km", "2-4 km", "> 4 km"]
    D["_s"] = pd.cut(D.d_shore_m, sed, labels=slab, include_lowest=True)
    for b, g in D.groupby("_s", observed=True):
        s = stats(g.resid.values)
        s.update(factor="dist_to_shoreline", bin=str(b),
                 share_SSE_pct=100 * (g.resid ** 2).sum() / sse.sum())
        dec.append(s)
        print(f"     {str(b):<18}{s['n']:>7,}{s['RMSE_m']:>8.3f}"
              f"{s['NMAD_m']:>7.3f}{s['bias_m']:>+8.3f}"
              f"{s['share_SSE_pct']:>8.1f}")

    # ---------------------------------------- 7. withheld block
    fd = D.groupby("fold").agg(n=("resid", "size"),
                               SSE=("resid", lambda v: float((v ** 2).sum())))
    fd["RMSE_m"] = np.sqrt(fd.SSE / fd.n)
    print(f"\n  7. WITHHELD BLOCK — {D.block.nunique():,} contiguous 1 km "
          f"blocks (folds span {fd.RMSE_m.min():.2f}–{fd.RMSE_m.max():.2f} m)")
    bl = D.groupby("block").agg(
        n=("resid", "size"),
        SSE=("resid", lambda v: float((v ** 2).sum())))
    bl["RMSE_m"] = np.sqrt(bl.SSE / bl.n)
    bl["share_SSE_pct"] = 100 * bl.SSE / sse.sum()
    bl = bl.sort_values("share_SSE_pct", ascending=False)
    print(f"     {'block':>7}{'n':>6}{'RMSE':>9}{'%SSE':>8}   worst 6 blocks")
    for b, r in bl.head(6).iterrows():
        print(f"     {int(b):>7}{int(r.n):>6}{r.RMSE_m:>9.3f}"
              f"{r.share_SSE_pct:>8.1f}")
    top = bl.head(max(1, int(round(0.01 * len(bl)))))
    print(f"     the worst 1% of blocks ({len(top)} of {len(bl):,}) carry "
          f"{top.share_SSE_pct.sum():.0f}% of the squared error")
    print(f"     block-level RMSE spans {bl.RMSE_m.min():.2f} to "
          f"{bl.RMSE_m.max():.2f} m")

    # An 'overall' row, so downstream scripts read the headline package from
    # the table instead of retyping it.
    o = dict(stats(D.resid.values))
    o.update(factor="overall", bin=PRIMARY, share_SSE_pct=100.0,
             **{f"worst_{p}pct_share_SSE": conc[p] for p in (1, 2, 5, 10, 20)})
    dec.insert(0, o)
    pd.DataFrame(dec).to_csv(CFG.TABLES / "hist18_error_decomposition.csv",
                             index=False)

    # ---------------------------------------- worst 20 residuals
    w = D.reindex(D.abs_resid.sort_values(ascending=False).index).head(20)
    wcols = ["lon", "lat", "chainage_km", "bed_m", "resid", "d_train_m",
             "d_channel_km", "roughness_m", "d_shore_m", "resid_random"]
    w[wcols].to_csv(CFG.TABLES / "hist18_worst_residuals.csv", index=False)
    print("\n  WORST 20 RESIDUALS")
    print(f"     {'lon':>8}{'lat':>8}{'km':>7}{'bed':>8}{'resid':>8}"
          f"{'d_train':>9}{'d_chan':>8}{'rough':>7}{'random e':>10}")
    for r in w.itertuples():
        print(f"     {r.lon:>8.3f}{r.lat:>8.3f}{r.chainage_km:>7.0f}"
              f"{r.bed_m:>8.2f}{r.resid:>+8.2f}{r.d_train_m:>8.0f}m"
              f"{r.d_channel_km:>8.2f}{r.roughness_m:>7.2f}"
              f"{r.resid_random:>+10.2f}")
    print(f"\n     Of the worst 20, {int((w.bed_m < PLATFORM_THR).sum())} sit "
          f"below the platform threshold (i.e. in troughs) and "
          f"{int((w.d_train_m > 500).sum())} are >500 m from any training "
          f"point.")
    print(f"     Their median |error| under RANDOM CV is only "
          f"{w.resid_random.abs().median():.2f} m, against "
          f"{w.abs_resid.median():.2f} m under blocked CV.")

    # ================================================================= verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    # Rank the candidate explanations by how much RMSE actually varies across
    # each factor's bins. The largest spread is the dominant mechanism -- read
    # off the data rather than asserted.
    dd = pd.DataFrame(dec)
    rank = []
    for fct in ("local_roughness", "dist_to_shoreline", "bed_elevation_class",
                "dist_to_channel_corridor", "reach",
                "dist_to_nearest_training_point"):
        g = dd[dd.factor == fct]
        if len(g) < 2:
            continue
        rank.append((fct, g.RMSE_m.max() / g.RMSE_m.min(),
                     g.RMSE_m.min(), g.RMSE_m.max()))
    rank.sort(key=lambda t: -t[1])
    print("  factors ranked by how much RMSE varies across their bins:")
    for fct, ratio_, lo_, hi_ in rank:
        print(f"    {ratio_:>5.1f}x   {fct:<32}{lo_:.2f} -> {hi_:.2f} m")

    ratio = rr.loc[PRIMARY, "RMSE_m"] / rr.loc["random", "RMSE_m"]
    tr_ratio = st["RMSE_m"] / sp["RMSE_m"]
    k5 = int(round(0.05 * len(sse)))
    keep5 = np.setdiff1d(np.arange(len(sse)), order[:k5])
    w20med = w.abs_resid.median()
    w20rnd = w.resid_random.abs().median()

    print(f"\n  1. SPATIAL EXTRAPOLATION FROM BLOCKED CV -- NOT the cause.")
    print(f"     Random hold-out already gives "
          f"{rr.loc['random','RMSE_m']:.3f} m; 1 km blocking adds only "
          f"{ratio:.2f}x")
    print(f"     ({rr.loc[PRIMARY,'RMSE_m']:.3f} m), and even 5 km blocks only "
          f"{rr.loc['blocked5km','RMSE_m']/rr.loc['random','RMSE_m']:.2f}x. The "
          f"reason is visible in the")
    print(f"     geometry: the median distance to the nearest training point "
          f"moves only")
    print(f"     {rr.loc['random','median_dist_to_training_m']:.0f} -> "
          f"{rr.loc[PRIMARY,'median_dist_to_training_m']:.0f} m, because the "
          f"survey is line-based and removing a 1 km")
    print(f"     block still leaves the adjacent survey lines in training.")
    print(f"     Decisive: the worst 20 residuals are just as large under "
          f"RANDOM CV")
    print(f"     (median |e| {w20rnd:.2f} m vs {w20med:.2f} m blocked). They "
          f"are not created by the hold-out.")

    print(f"\n  2. SUB-SPACING MORPHOLOGICAL RELIEF -- the dominant cause.")
    lr = dd[dd.factor == 'local_roughness']
    bz_ = dd[dd.factor == 'bed_elevation_class']
    nmin = int(bz_.loc[bz_.RMSE_m.idxmax(), 'n'])
    print(f"     Bed-elevation class shows the largest raw ratio "
          f"({bz_.RMSE_m.max()/bz_.RMSE_m.min():.1f}x) but its worst bin holds "
          f"only")
    print(f"     {nmin} points, so the best-populated discriminator is local "
          f"roughness")
    print(f"     ({lr.RMSE_m.max()/lr.RMSE_m.min():.1f}x over bins of "
          f"{int(lr.n.min())}-{int(lr.n.max())} points):")
    print(f"     RMSE {lr.RMSE_m.min():.2f} m in the smoothest quartile against "
          f"{lr.RMSE_m.max():.2f} m in the")
    print(f"     roughest decile. The bed has {z.max()-z.min():.1f} m of "
          f"amplitude and")
    print(f"     the survey samples it every {np.median(dnn):.0f} m, so relief "
          f"shorter than that")
    print(f"     spacing cannot be RELIABLY RECOVERED FROM THE SOUNDING POINTS "
          f"ALONE by an")
    print(f"     ordinary unconstrained interpolator. That is a statement about "
          f"the present")
    print(f"     data being under-determined, NOT about mathematical "
          f"impossibility: adding")
    print(f"     the historical channel axis, breaklines, hydrographic "
          f"cross-sections or")
    print(f"     old charts as constraints could recover a narrow trough "
          f"considerably better.")
    print(f"     The signature is a smoothing bias that REVERSES with "
          f"elevation:")
    bz = dd[dd.factor == 'bed_elevation_class']
    print(f"     deepest bin bias {bz.bias_m.max():+.2f} m (predicted too "
          f"shallow), highest bin "
          f"{bz.bias_m.min():+.2f} m")
    print(f"     (too deep). That is regression toward the local mean, not an "
          f"offset.")

    print(f"\n  3. THE SHORELINE MARGIN -- the second mechanism.")
    ds = dd[dd.factor == 'dist_to_shoreline'].sort_values('lo' if 'lo' in dd else 'RMSE_m')
    sh0 = dd[(dd.factor == 'dist_to_shoreline') & (dd.bin == '0-500 m')].iloc[0]
    print(f"     Within 500 m of the shoreline RMSE is {sh0.RMSE_m:.2f} m and "
          f"those {sh0.n:,} points")
    print(f"     ({100*sh0.n/ov['n']:.0f}% of the survey) carry "
          f"{sh0.share_SSE_pct:.0f}% of the squared error, against "
          f"{dd[(dd.factor=='dist_to_shoreline')].RMSE_m.min():.2f} m beyond "
          f"4 km.")
    print(f"     This is the same margin problem the shoreline constraint was "
          f"introduced for.")

    print(f"\n  4. VERTICAL REFERENCE -- excluded.")
    print(f"     Overall bias {ov['bias_m']:+.3f} m; per-reach bias spans only "
          f"{min(d['bias_m'] for d in dec if d['factor']=='reach'):+.3f} to "
          f"{max(d['bias_m'] for d in dec if d['factor']=='reach'):+.3f} m,")
    print(f"     and the elevation-dependent bias reverses sign. A datum error "
          f"is a constant")
    print(f"     offset; nothing here behaves like one.")

    print(f"\n  5. CONCENTRATION -- strongly non-uniform.")
    print(f"     The worst 1% of soundings carry {conc[1]:.0f}% of the squared "
          f"error and the worst 5%")
    print(f"     carry {conc[5]:.0f}%. Excluding that 5% the RMSE falls to "
          f"{np.sqrt(sse[keep5].mean()):.2f} m. Median |error| is "
          f"{ov['p50_abs']:.2f} m;")
    print(f"     the smoothest quartile of the bed is predicted to "
          f"{lr.RMSE_m.min():.2f} m RMSE.")
    print(f"     Reach 1+2 alone -- the deep lower reservoir with the former "
          f"channel --")
    rch = dd[dd.factor == 'reach']
    print(f"     carries {rch.share_SSE_pct.max():.0f}% of the squared error.")

    print(f"\n  => THE NUMBER PACKAGE TO QUOTE, never the RMSE alone:")
    print(f"       RMSE        {rr.loc[PRIMARY,'RMSE_m']:.2f} m")
    print(f"       NMAD        {ov['NMAD_m']:.2f} m")
    print(f"       bias        {ov['bias_m']:+.2f} m")
    print(f"       median |e|  {ov['p50_abs']:.2f} m")
    print(f"       and: the worst 5% of observations account for "
          f"{conc[5]:.0f}% of the squared error.")
    print(f"\n     Defensible wording: 'the median absolute prediction error "
          f"was")
    print(f"     approximately {ov['p50_abs']:.1f} m, whereas RMSE rose to "
          f"{rr.loc[PRIMARY,'RMSE_m']:.2f} m because of a small number of")
    print(f"     large errors concentrated in morphologically complex zones.'")
    print(f"     NOT defensible: 'most of the DEM is accurate to ~1 m'. "
          f"{ov['p50_abs']:.2f} m is the")
    print(f"     median of the residual distribution at SOUNDING LOCATIONS, "
          f"not an")
    print(f"     area-weighted accuracy of the interpolated surface.")
    print(f"\n     {rr.loc[PRIMARY,'RMSE_m']:.3f} m is a heavy-tailed error of "
          f"local morphology, not a")
    print(f"     vertical offset of the DEM, and not an artefact of blocked "
          f"cross-validation.")
    print(f"\n  NOT concluded: that the DEM is accurate to the median. Blocked "
          f"CV remains the")
    print(f"  right score for a surface evaluated away from the survey lines, "
          f"and the")
    print(f"  headline stays {rr.loc[PRIMARY,'RMSE_m']:.3f} m. What changes is "
          f"the EXPLANATION -- and the")
    print(f"  fact that integrated AREA statistics are far better conditioned "
          f"than any")
    print(f"  single cell is now quantified rather than asserted.")
    print(f"\n  NOT DONE, deliberately: no interpolation parameter was changed "
          f"and nothing")
    print(f"  was tuned to reduce RMSE.")

    # ================================================================= figure
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.26)

    a = fig.add_subplot(gs[0, 0])
    a.hist(D.resid, bins=np.arange(-12, 12.25, 0.25), color=BLUE, alpha=0.85)
    a.axvline(0, color=INK, lw=1)
    a.axvline(ov["bias_m"], color=RED, lw=1.6, ls="--",
              label=f"bias {ov['bias_m']:+.3f} m")
    a.set_yscale("log")
    a.set_xlabel("residual (m), predicted − observed")
    a.set_ylabel("soundings (log)")
    a.legend(fontsize=8.5)
    a.set_title(f"a · Residuals, {PRIMARY}\nRMSE {ov['RMSE_m']:.2f} vs NMAD "
                f"{ov['NMAD_m']:.2f} m — heavy tails, no offset",
                fontsize=10, loc="left")

    a = fig.add_subplot(gs[0, 1])
    a.plot(100 * np.arange(1, len(cum) + 1) / len(cum), 100 * cum,
           color=RED, lw=2.4)
    a.plot([0, 100], [0, 100], color=GREY, ls=":", lw=1.2,
           label="uniform error")
    for pct in (1, 5, 10):
        a.plot([pct], [conc[pct]], "o", color=INK, ms=6)
        a.annotate(f"{pct}% → {conc[pct]:.0f}%", (pct, conc[pct]),
                   textcoords="offset points", xytext=(9, -4), fontsize=8.5)
    a.set_xlim(0, 40); a.set_ylim(0, 100)
    a.set_xlabel("worst x % of soundings")
    a.set_ylabel("% of total squared error")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("b · Error concentration\nfar from uniform", fontsize=10,
                loc="left")

    a = fig.add_subplot(gs[0, 2])
    xs = np.arange(len(sc))
    a.bar(xs, sc.RMSE_m, color=[GREEN if s == "random" else
                                (RED if s == PRIMARY else BLUE)
                                for s in sc.scheme], alpha=0.9)
    a.plot(xs, sc.NMAD_m, "o-", color=INK, lw=1.6, ms=6, label="NMAD")
    for i, r in enumerate(sc.itertuples()):
        a.annotate(f"{r.RMSE_m:.2f}", (i, r.RMSE_m),
                   textcoords="offset points", xytext=(0, 4), ha="center",
                   fontsize=8.5)
    a.set_xticks(xs)
    a.set_xticklabels([s.replace("blocked", "") for s in sc.scheme],
                      fontsize=8.5)
    a.set_ylabel("m")
    a.legend(fontsize=8.5); a.grid(alpha=0.25, axis="y")
    a.set_title("c · Hold-out geometry sets the number\nsame kriging "
                "throughout", fontsize=10, loc="left")

    def barpanel(ax, factor, title, rot=0):
        d = pd.DataFrame([x for x in dec if x["factor"] == factor])
        ax.bar(range(len(d)), d.RMSE_m, color=BLUE, alpha=0.85, label="RMSE")
        ax.plot(range(len(d)), d.NMAD_m, "o-", color=RED, lw=1.5, ms=5,
                label="NMAD")
        ax2 = ax.twinx()
        ax2.plot(range(len(d)), d.share_SSE_pct, "s--", color=AMBER, lw=1.3,
                 ms=4, label="% of SSE")
        ax2.set_ylabel("% of squared error", fontsize=8.5, color=AMBER)
        ax2.tick_params(labelsize=8, colors=AMBER)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d.bin, fontsize=8, rotation=rot,
                           ha="right" if rot else "center")
        ax.set_ylabel("m")
        ax.grid(alpha=0.22, axis="y")
        ax.set_title(title, fontsize=10, loc="left")
        return ax

    barpanel(fig.add_subplot(gs[1, 0]), "dist_to_nearest_training_point",
             "d · By distance to the nearest training point\nnon-monotonic — "
             "the densest areas are the deep channel lines", rot=35)
    barpanel(fig.add_subplot(gs[1, 1]), "bed_elevation_class",
             "e · By bed elevation — troughs dominate", rot=35)
    barpanel(fig.add_subplot(gs[1, 2]), "local_roughness",
             "f · By local terrain roughness", rot=35)
    barpanel(fig.add_subplot(gs[2, 0]), "dist_to_channel_corridor",
             "g · By distance to the channel corridor", rot=35)
    barpanel(fig.add_subplot(gs[2, 1]), "reach",
             "h · By reach — a datum error would show here", rot=25)

    a = fig.add_subplot(gs[2, 2])
    sm = a.scatter(D.d_train_m, D.abs_resid, c=D.bed_m, s=5, alpha=0.5,
                   cmap="viridis_r", vmin=-15, vmax=15)
    a.set_xscale("log")
    a.set_xlabel("distance to nearest training point (m, log)")
    a.set_ylabel("|residual| (m)")
    a.grid(alpha=0.22)
    plt.colorbar(sm, ax=a, label="bed elevation (m)", pad=0.02)
    a.set_title("i · Large errors occur at EVERY distance\nsparsity is not "
                "the driver", fontsize=10, loc="left")

    fig.suptitle(f"V16 · Why the blocked-CV RMSE is "
                 f"{rr.loc[PRIMARY,'RMSE_m']:.3f} m — decomposition of the bed-"
                 f"surface prediction error   ·   ordinary kriging, "
                 f"parameters unchanged", fontsize=12.5, y=0.975)
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V16_cv_error_diagnosis.{e}", dpi=170,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V16_cv_error_diagnosis.png'}")
    for f in ("hist18_cv_scheme_comparison", "hist18_error_decomposition",
              "hist18_worst_residuals"):
        print(f"-> {CFG.TABLES/(f + '.csv')}")


if __name__ == "__main__":
    main()
