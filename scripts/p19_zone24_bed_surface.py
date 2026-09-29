#!/usr/bin/env python
"""P19 — CV-selected PRE_BREACH bed-elevation DEM for ZONE_2 or ZONE_4.

NAMING, DELIBERATE. Four interpolators are fit and cross-validated (IDW, OK,
RBF, LINEAR); the one with the lowest blocked-CV RMSE is written as the
canonical surface. For both zones this session that was RBF, not OK -- so
"kriged bed DEM" is not an accurate name for the shipped product. The correct
name is "PRE_BREACH interpolated bed-elevation DEM, RBF selected by spatial
blocked cross-validation" (or "CV-selected bed DEM" for short). OK stays in
the comparison and its Lagrange-system variance is exported (see p20) as a
diagnostic of the OK METHOD specifically -- it is not the uncertainty of the
canonical RBF surface, and must not be read as such.

ACCURACY IS QUOTED ON SOUNDINGS ONLY (2026-09-16, audit F-01/F-06). The
cross-validation set stacks 527 / 626 real soundings with 854 / 2,110 shoreline
pseudo-points from p18 -- values of a smooth 1-D water-surface profile placed on
a WorldCover-2021 outline, ~28 % of them held flat beyond 15 km from any anchor.
A 2-D interpolator reproduces that constructed curve almost tautologically, so
the pooled CV RMSE (1.84 / 2.93 m) was dominated by the easy half of the sample
and understated the error on the bed by ~1.7x. `cv_score` now returns pooled AND
stratified metrics; `best_method` is chosen on `RMSE_snd_m`; the summary table
carries `best_rmse_soundings_m` (3.19 / 3.99 m, bias +0.84 / +1.15 m -- the
surface sits too shallow where depth was actually measured) as the accuracy
figure and `best_rmse_pooled_m` only for continuity. For both zones RBF remains
the selected method under either metric.

PRE_BREACH ONLY. The manual soundings (p17) are static, pre-2023 chart depths
-- they describe the channel bed BEFORE the breach re-incised it. There is no
post-breach depth measurement anywhere in this project's inputs, so a
POST_BREACH bed surface cannot legitimately be built from them (the project's
own standing rule: pre- and post-breach bathymetry are different surfaces and
must never be merged or substituted for each other). p18's POST_BREACH
water-surface profile is a real product on its own; it does not feed a bed
DEM here. A POST_BREACH bed surface needs new depth data (survey or ICESat-2
exposed-terrain during a drawdown) -- future work, not attempted.

METHOD PORTED FROM hist14_bed_surface.py, NOT ITS NUMBERS. The OK solver
(`_ok`), the IDW/RBF/LINEAR alternatives and the guards against a degenerate
kriging system are imported verbatim -- generic numerics, not reservoir-
specific (same rule p0w_zone2_orbit_aware_composite.py already applied this
session to ZONE_4's water-mask method). Real kriging variance comes from
`ok_predict(..., want_var=True)`, imported from k9_predictive_validation.py.
Everything domain-specific is re-measured fresh here: the footprint (registry
ZONE_2/ZONE_4, never KAKHOVKA_RESERVOIR_CORE), the shoreline constraint
(p18's chainage-varying elevations, not one constant), the variogram range,
and the CV block sizes chosen from that range.

SHORELINE CONSTRAINT. p18's pseudo-points (250 m spacing, one elevation per
point from the chainage profile) are stacked with the real soundings as extra
training data, exactly hist14's mechanism -- just with a per-point elevation
instead of one constant. Points p18 flagged `extrapolated=True` (>15 km from
the nearest real water-level anchor) are still used -- dropping them would
leave holes exactly where the constraint matters most -- but are kept in a
separate column through to `p20`'s uncertainty export.

EMODNET, ZONE_2 ONLY, AS A COVARIATE ONLY, NEVER AS BACKGROUND ELEVATION.
Per the user's explicit decision, EMODnet's own datum stays unreconciled
(LAT). It may enter only as a de-meaned shape covariate in a regression-
kriging trend: `C(x) = EMODnet_elev(x) - local_median(EMODnet_elev, 2 km)`,
fit `z ~ a + b*C` on the soundings, krige the RESIDUAL. Whether this beats
plain isotropic OK is decided by the same blocked CV as everything else here,
not assumed -- ZONE_4 has ~0% EMODnet depth coverage (confirmed by direct
sampling this session) and never gets this term.

Usage
-----
python scripts/p19_zone24_bed_surface.py --zone ZONE_2_KHERSON_DELTA
python scripts/p19_zone24_bed_surface.py --zone ZONE_4_DAM_TO_KHERSON_FLOODWAY

Outputs
-------
data/processed/bathymetry/zone24_bed_surface_{ZONE}_PRE_BREACH_250m.npz  (F:)
outputs/tables/p19_interpolator_cv_{ZONE}_PRE_BREACH.csv
outputs/tables/p19_surface_summary_{ZONE}_PRE_BREACH.csv
outputs/figures/P19_bed_surface_validation_{ZONE}_PRE_BREACH.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from hist14_bed_surface import METHODS, spherical, nmad, OK_K, OK_BATCH  # noqa: F401
from k9_predictive_validation import ok_predict

REGIME = "PRE_BREACH"
CANONICAL_CELL = 250.0   # scientific grid; 50/30 m are display-only refinements
                        # (hist17's own finding for the reservoir: integrated
                        # area stats move ~0.06 pp between 250 m and 30 m --
                        # not re-verified for this sparser dataset, carried as
                        # the same operating assumption, tag the filename so
                        # a fine-grid run can never silently overwrite 250 m)
N_FOLDS = 10
EMODNET_TIF = ROOT / "data/processed/emodnet/D0_liman_background.tif"
EMODNET_TREND_RADIUS_M = 2000.0
RNG = np.random.default_rng(CFG.SEED)


def load_soundings(zone: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/processed/bathymetry/manual_soundings_evrf2019.parquet")
    d = d[d[zone]].copy()
    n0 = len(d)
    d = d.assign(_kx=d.x.round(0), _ky=d.y.round(0)).groupby(
        ["_kx", "_ky"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"),
        H_bed_evrf2019_m=("H_bed_evrf2019_m", "mean"))
    if len(d) != n0:
        print(f"  averaged {n0-len(d)} exact-duplicate coordinates -> {len(d)}")
    return d


def load_shore_points(zone: str) -> pd.DataFrame:
    sh = pd.read_parquet(
        ROOT / f"data/processed/bathymetry/zone24_shore_pseudopoints_{REGIME}.parquet")
    return sh[sh.zone == zone].copy()


def emodnet_covariate(xy: np.ndarray) -> np.ndarray:
    """C(x) = EMODnet elevation minus its own local (2 km) median -- a shape
    covariate only. NaN wherever EMODnet has no data at that location."""
    with rasterio.open(EMODNET_TIF) as src:
        elev = src.read(1)
        tr = src.transform
        e_val = np.array([v[0] for v in src.sample(
            [(x, y) for x, y in xy], indexes=[1])])
        rows, cols = np.where(np.isfinite(elev))
        if len(rows) == 0:
            return np.full(len(xy), np.nan)
        xs, ys = rasterio.transform.xy(tr, rows, cols)
        tree = cKDTree(np.c_[xs, ys])
        d, idx = tree.query(xy, k=1)
        # local median over a 2 km neighbourhood of valid EMODnet cells
        local_med = np.full(len(xy), np.nan)
        near_idx = tree.query_ball_point(xy, r=EMODNET_TREND_RADIUS_M)
        for i, nb in enumerate(near_idx):
            if nb:
                local_med[i] = np.median(elev[rows[nb], cols[nb]])
    return e_val - local_med


ENVELOPE_K = 8   # matches k9_predictive_validation.py's own OK "stay inside the
                 # local data envelope" neighbourhood size


def clip_to_local_envelope(xy_tr: np.ndarray, z_tr: np.ndarray,
                           xy_te: np.ndarray, pred: np.ndarray,
                           k: int = ENVELOPE_K, is_shore_tr: np.ndarray = None,
                           return_diag: bool = False):
    """Clip each prediction to the [min, max] of its K nearest TRAINING points.

    k9's ok_predict() already does this for OK specifically ("a local OK
    prediction must stay inside the local data envelope... anything outside
    it means the system was degenerate"). RBF/IDW/LINEAR, imported verbatim
    from hist14_bed_surface.py, have no such guard -- and RBF's thin-plate
    spline has no bound on its output at all. That produced a shipped
    ZONE_2 raster reading up to +95.9 m in a delta whose soundings top out
    at -0.79 m: RBF scored well on blocked CV (which only checks accuracy AT
    held-out sounding locations) while oscillating wildly at OTHER grid
    cells, which CV never looks at. This generalises k9's guard to every
    method, applied post-hoc rather than reworking each interpolator.

    NAMING CONSEQUENCE. The guard is now part of the estimator, not a
    cosmetic safety net -- z(x) is constrained to [min(z_8NN), max(z_8NN)]
    by construction, and that local envelope (not the global catastrophic-
    failure check in p21) is what actually rules out a +80 m excursion mid-
    delta. So the shipped method is "local-envelope-constrained RBF selected
    by blocked spatial CV", not bare "RBF" -- a reader who reconstructs a
    plain RBFInterpolator without this clip reproduces the +188 m bug this
    guard exists to prevent. See p21 check 9 for how much of the surface the
    guard actually altered (`is_shore_tr`/`return_diag`, below) -- if a large
    fraction of cells is clipped, that is a materially different claim than
    "RBF with a safety net for rare excursions", and must be reported as such,
    not silently folded into "RBF selected"."""
    K = min(k, len(xy_tr))
    _, idx = cKDTree(xy_tr).query(xy_te, k=K)
    idx = np.atleast_2d(idx)
    nbr_z = z_tr[idx]
    lo, hi = nbr_z.min(1), nbr_z.max(1)
    clipped = np.clip(pred, lo, hi)
    if not return_diag:
        return clipped
    n_shore = is_shore_tr[idx].sum(1) if is_shore_tr is not None else np.full(len(pred), np.nan)
    diag = dict(clip_amount_m=np.abs(pred - clipped),
               n_sounding_nn=K - n_shore, n_shoreline_nn=n_shore)
    return clipped, diag


def block_folds(xy: np.ndarray, block_km: float, seed: int) -> np.ndarray:
    block = np.floor(xy / (block_km * 1000.0)).astype(int)
    uniq, inv = np.unique(block, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq)) % N_FOLDS
    return perm[inv]


def _stats(r: np.ndarray, suffix: str = "") -> dict:
    """RMSE / MAE / bias / NMAD of a residual vector; NaN when it is empty.

    ``suffix`` = "" gives the historical pooled names (``n, RMSE_m, ...``) so
    every existing reader keeps working; ``"_snd"`` / ``"_shore"`` give the
    stratified columns."""
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {f"n{suffix}": 0, f"RMSE{suffix}_m": np.nan, f"MAE{suffix}_m": np.nan,
                f"bias{suffix}_m": np.nan, f"NMAD{suffix}_m": np.nan}
    return {f"n{suffix}": int(len(r)), f"RMSE{suffix}_m": float(np.sqrt((r ** 2).mean())),
            f"MAE{suffix}_m": float(np.abs(r).mean()), f"bias{suffix}_m": float(r.mean()),
            f"NMAD{suffix}_m": nmad(r)}


def summarise_resid(resid: np.ndarray, is_shore: np.ndarray | None) -> dict:
    """Pooled metrics PLUS the split by observation type.

    The pooled RMSE used to be the only number this script reported, and it was
    the number the surface's accuracy was quoted at (1.84 / 2.93 m). But 62-77 %
    of the CV set are shoreline pseudo-points: values of a smooth 1-D water-
    surface profile interpolated along chain_km and placed on a WorldCover
    outline, ~28 % of them held flat beyond 15 km from any anchor. A 2-D
    interpolator reproduces a smooth 1-D curve almost tautologically, so the
    pooled figure is dominated by the easy, constructed half of the sample.
    Against the real depth observations the error is ~1.7x larger and carries a
    positive bias (surface too shallow). Reporting both, and selecting the
    method on the soundings-only figure, is the whole point of this function
    (audit 20260916T093000Z, findings F-01 / F-06)."""
    out = _stats(resid)
    if is_shore is not None:
        out.update(_stats(resid[~is_shore], "_snd"))
        out.update(_stats(resid[is_shore], "_shore"))
    return out


def cv_score(xy: np.ndarray, z: np.ndarray, vg, folds: np.ndarray, method: str,
             is_shore: np.ndarray | None = None) -> dict:
    resid = np.full(len(z), np.nan)
    for f in range(N_FOLDS):
        te = folds == f
        if te.sum() == 0 or (~te).sum() < 10:
            continue
        pred = METHODS[method](xy[~te], z[~te], xy[te], vg)
        pred = clip_to_local_envelope(xy[~te], z[~te], xy[te], pred)
        resid[te] = pred - z[te]
    return summarise_resid(resid, is_shore)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True,
                    choices=["ZONE_2_KHERSON_DELTA", "ZONE_4_DAM_TO_KHERSON_FLOODWAY"])
    ap.add_argument("--cell", type=float, default=CANONICAL_CELL,
                    help="grid cell size in metres (default 250, canonical/"
                         "scientific; 50/30 are display-only refinements)")
    args = ap.parse_args()
    zone = args.zone
    CELL = args.cell
    tag = f"{zone}_{REGIME}"

    print("=" * 78)
    print(f"P19 — CV-selected bed-elevation DEM, {tag}")
    print("=" * 78)

    snd = load_soundings(zone)
    shp = load_shore_points(zone)
    print(f"  soundings: {len(snd)}, bed {snd.H_bed_evrf2019_m.min():.2f} .. "
          f"{snd.H_bed_evrf2019_m.max():.2f} m EVRF2019")
    print(f"  shoreline pseudo-points: {len(shp)} "
          f"({int(shp.extrapolated.sum())} extrapolated)")
    if len(snd) < 30:
        raise SystemExit(f"only {len(snd)} soundings in {zone} -- not enough to "
                         f"krige credibly; stop rather than fabricate a surface")

    xy_snd = np.c_[snd.x.values, snd.y.values]
    z_snd = snd.H_bed_evrf2019_m.values
    xy_sh = np.c_[shp.x.values, shp.y.values]
    z_sh = shp.elevation_evrf2019_m.values

    xy = np.vstack([xy_snd, xy_sh])
    z = np.concatenate([z_snd, z_sh])
    is_shore = np.concatenate([np.zeros(len(snd), bool), np.ones(len(shp), bool)])

    import skgstat as skg
    sub = RNG.choice(len(xy), min(1500, len(xy)), replace=False)
    V = skg.Variogram(xy[sub], z[sub], model="spherical", n_lags=20, normalize=False)
    vg = (float(V.parameters[0]), float(V.parameters[1]),
         float(V.parameters[2]) if len(V.parameters) > 2 else 0.0)
    print(f"\n  spherical variogram: range {vg[0]/1000:.2f} km, sill {vg[1]:.2f}, "
          f"nugget {vg[2]:.2f} m2")

    tree = cKDTree(xy)
    nn = tree.query(xy, k=2)[0][:, 1]
    block_primary = max(0.5, round(np.median(nn) / 200.0) * 0.2)   # ~ a few x spacing
    block_stress = max(block_primary * 4, vg[0] / 1000.0 * 1.5)
    print(f"  block CV scales: primary {block_primary:.2f} km "
          f"(matched to median spacing {np.median(nn):.0f} m), "
          f"stress {block_stress:.2f} km (~1.5x variogram range)")

    methods = dict(METHODS)
    use_emodnet = False
    C = None
    if zone == "ZONE_2_KHERSON_DELTA":
        C = emodnet_covariate(xy)
        n_cov = int(np.isfinite(C).sum())
        print(f"\n  EMODnet covariate available at {n_cov}/{len(xy)} points "
              f"({100*n_cov/len(xy):.0f}%)")
        if n_cov >= 30:
            ok_c = np.isfinite(C)
            b, a = np.polyfit(C[ok_c], z[ok_c], 1)
            r2 = np.corrcoef(C[ok_c], z[ok_c])[0, 1] ** 2
            print(f"  z ~ {a:.3f} + {b:.3f}*C, R2={r2:.3f} "
                  f"(shape covariate only, never EMODnet's own elevation)")
            use_emodnet = True

    schemes = {"blocked_primary": block_folds(xy, block_primary, CFG.SEED),
              "blocked_stress": block_folds(xy, block_stress, CFG.SEED + 1),
              "random": RNG.integers(0, N_FOLDS, len(xy))}

    rows = []
    for scheme, folds in schemes.items():
        for m in methods:
            rows.append(dict(scheme=scheme, method=m,
                             **cv_score(xy, z, vg, folds, m, is_shore)))
        if use_emodnet:
            ok_c = np.isfinite(C)
            resid_full = np.full(len(z), np.nan)
            for f in range(N_FOLDS):
                te = folds == f
                tr = (~te) & ok_c
                if te.sum() == 0 or tr.sum() < 10:
                    continue
                b, a = np.polyfit(C[tr], z[tr], 1)
                r_tr = z[tr] - (a + b * C[tr])
                pred_r = METHODS["OK"](xy[tr], r_tr, xy[te], vg)
                trend = np.where(np.isfinite(C[te]), a + b * np.nan_to_num(C[te]), np.nan)
                pred = np.where(np.isfinite(C[te]), trend + pred_r,
                               METHODS["OK"](xy[~te], z[~te], xy[te], vg))
                resid_full[te] = pred - z[te]
            # through the same aggregator as every other method -- an inline
            # copy here would lack RMSE_snd_m and silently break the selection
            rows.append(dict(scheme=scheme, method="OK_EMODNET_COVARIATE",
                            **summarise_resid(resid_full, is_shore)))

    cv = pd.DataFrame(rows)
    ROLE = {"blocked_primary": "PRIMARY - blocks matched to local point spacing",
           "blocked_stress": "stress test - approaches/exceeds the variogram range",
           "random": "optimistic reference"}
    cv["role"] = cv.scheme.map(ROLE)
    cv_path = CFG.TABLES / f"p19_interpolator_cv_{tag}.csv"
    cv.to_csv(cv_path, index=False)
    print("\n  CV summary (blocked_primary) -- soundings-only is the accuracy "
          "figure; pooled shown for continuity:")
    bl = cv[cv.scheme == "blocked_primary"].set_index("method")
    for m, r in bl.iterrows():
        print(f"    {m:24s} SND  RMSE {r.RMSE_snd_m:6.3f}  bias {r.bias_snd_m:+6.3f}  "
              f"n={r.n_snd:.0f}   | shore RMSE {r.RMSE_shore_m:6.3f} n={r.n_shore:.0f}"
              f"   | pooled RMSE {r.RMSE_m:6.3f}")
    # method selection on the SOUNDINGS metric. The pooled metric used to
    # decide, and it favours whatever fits the smooth constructed shoreline
    # curve best -- not whatever predicts the bed best.
    best = bl.RMSE_snd_m.idxmin()
    best_pooled = bl.RMSE_m.idxmin()
    best_name = (f"local-envelope-constrained {best}" if best != "OK_EMODNET_COVARIATE"
                else best)
    print(f"\n  preferred method: {best_name} selected by blocked spatial CV on "
          f"SOUNDINGS (RMSE_snd {bl.loc[best,'RMSE_snd_m']:.3f} m, bias "
          f"{bl.loc[best,'bias_snd_m']:+.3f} m; pooled {bl.loc[best,'RMSE_m']:.3f} m)")
    if best_pooled != best:
        print(f"  NOTE: the pooled metric would have picked {best_pooled} "
              f"(pooled {bl.loc[best_pooled,'RMSE_m']:.3f} m, soundings "
              f"{bl.loc[best_pooled,'RMSE_snd_m']:.3f} m) -- selection changed.")
    print("  see the clip diagnostics below for how much of the surface the "
          "envelope guard actually altered (p21 check 9 reports this per raster)")

    # ---- build the production grid ---------------------------------------
    fp = SD.load_utm(zone)
    gr = SD.build_grid(fp, CELL, what=f"{tag} bed surface grid")
    gx, gy = gr["gx"], gr["gy"]
    GX, GY = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(fp, GX.ravel(), GY.ravel())
    tgt = np.c_[GX.ravel()[inside], GY.ravel()[inside]]
    print(f"\n  grid {len(gx)}x{len(gy)} at {CELL:.0f} m, {inside.sum():,} cells inside")

    pred = {}
    clip_diag = {}
    for m in ("IDW", "OK", "RBF", "LINEAR"):
        raw = METHODS[m](xy, z, tgt, vg)
        pred[m], diag = clip_to_local_envelope(xy, z, tgt, raw,
                                               is_shore_tr=is_shore, return_diag=True)
        clip_diag[m] = diag
        n_clipped = int(np.sum(diag["clip_amount_m"] > 1e-6))
        if n_clipped:
            ca = diag["clip_amount_m"][diag["clip_amount_m"] > 1e-6]
            print(f"  {m}: {n_clipped:,}/{len(tgt):,} cells "
                  f"({100*n_clipped/len(tgt):.1f}%) clipped -- |clip| median "
                  f"{np.median(ca):.2f} m, p95 {np.percentile(ca,95):.2f} m, "
                  f"max {ca.max():.2f} m (raw range "
                  f"{raw.min():.1f}..{raw.max():.1f} m)")
            nb_sh = diag["n_shoreline_nn"][diag["clip_amount_m"] > 1e-6]
            nb_sn = diag["n_sounding_nn"][diag["clip_amount_m"] > 1e-6]
            print(f"       among clipped cells' {ENVELOPE_K} nearest neighbours: "
                  f"median {np.median(nb_sn):.0f} real soundings, "
                  f"{np.median(nb_sh):.0f} shoreline pseudo-points")
    var_ok = ok_predict(xy, z, tgt, vg, want_var=True)
    if isinstance(var_ok, tuple):
        _, pred["OK_variance"] = var_ok
    else:
        pred["OK_variance"] = np.full(len(tgt), np.nan)

    dist_to_sounding = cKDTree(xy_snd).query(tgt, k=1)[0]

    out_dir = CFG.BULK_ROOT / "data_swot/processed/bathymetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"zone24_bed_surface_{tag}_{CELL:.0f}m.npz"
    diag_arrays = {}
    for m, d in clip_diag.items():
        diag_arrays[f"clip_amount_{m}"] = d["clip_amount_m"]
        diag_arrays[f"clip_n_sounding_nn_{m}"] = d["n_sounding_nn"]
        diag_arrays[f"clip_n_shoreline_nn_{m}"] = d["n_shoreline_nn"]
    np.savez_compressed(npz_path, gx=gx, gy=gy, inside=inside, tgt=tgt,
                        best_method=best, dist_to_sounding=dist_to_sounding,
                        variogram_range_m=vg[0], variogram_sill=vg[1],
                        variogram_nugget=vg[2],
                        n_soundings=len(snd), n_shore_points=len(shp),
                        **{f"pred_{k}": v for k, v in pred.items()},
                        **diag_arrays)
    print(f"\n-> {npz_path}")

    summ = pd.DataFrame([dict(
        zone=zone, regime=REGIME, n_soundings=len(snd), n_shore_points=len(shp),
        n_shore_extrapolated=int(shp.extrapolated.sum()),
        variogram_range_km=vg[0] / 1000.0, variogram_sill=vg[1], variogram_nugget=vg[2],
        block_primary_km=block_primary, block_stress_km=block_stress,
        best_method=best,
        best_method_by_pooled=best_pooled,
        # the accuracy figure: error against real depth observations
        best_rmse_soundings_m=float(bl.loc[best, "RMSE_snd_m"]),
        best_bias_soundings_m=float(bl.loc[best, "bias_snd_m"]),
        best_nmad_soundings_m=float(bl.loc[best, "NMAD_snd_m"]),
        n_cv_soundings=int(bl.loc[best, "n_snd"]),
        # against the constructed shoreline pseudo-points, kept separate
        best_rmse_shore_m=float(bl.loc[best, "RMSE_shore_m"]),
        best_bias_shore_m=float(bl.loc[best, "bias_shore_m"]),
        # the historical pooled number, renamed so nobody quotes it as accuracy
        best_rmse_pooled_m=float(bl.loc[best, "RMSE_m"]),
        used_emodnet_covariate=use_emodnet,
        grid_cells_inside=int(inside.sum()), cell_m=CELL)])
    summ_path = CFG.TABLES / (f"p19_surface_summary_{tag}.csv" if CELL == CANONICAL_CELL
                              else f"p19_surface_summary_{tag}_{CELL:.0f}m.csv")
    summ.to_csv(summ_path, index=False)
    print(f"-> {summ_path}")
    print(f"-> {cv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = axes[0].scatter(tgt[:, 0], tgt[:, 1], c=pred[best], s=2, cmap="viridis")
    axes[0].scatter(xy_snd[:, 0], xy_snd[:, 1], c="red", s=4, label="soundings")
    axes[0].set_title(f"{tag}\nbest method: {best}")
    axes[0].legend()
    axes[0].set_aspect("equal")
    fig.colorbar(sc, ax=axes[0], label="bed elevation, EVRF2019 (m)")

    xb = np.arange(len(bl.index))
    axes[1].bar(xb - 0.2, bl.RMSE_snd_m, width=0.4, label="soundings only")
    axes[1].bar(xb + 0.2, bl.RMSE_m, width=0.4, alpha=0.45, label="pooled (incl. shore pseudo-points)")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(bl.index)
    axes[1].set_ylabel("blocked-primary CV RMSE (m)")
    axes[1].set_title("interpolator comparison (selection on soundings)")
    axes[1].legend(fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig_path = CFG.FIG / (f"P19_bed_surface_validation_{tag}.png" if CELL == CANONICAL_CELL
                         else f"P19_bed_surface_validation_{tag}_{CELL:.0f}m.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"-> {fig_path}")


if __name__ == "__main__":
    main()
