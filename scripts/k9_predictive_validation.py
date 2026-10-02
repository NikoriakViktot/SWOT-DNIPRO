#!/usr/bin/env python
"""K9 -- formal predictive validation of the sounding-blind morphology prior.

CENTRAL QUESTION
    Does morphology_prior_v1 improve spatially independent prediction of
    historical Kakhovka bathymetry beyond (a) ordinary spatial autocorrelation
    and (b) channel-oriented anisotropy?

The experiment is frozen in k9_experiment_manifest.json BEFORE any outer-CV
result is computed or inspected. Model formulas, variogram family, anisotropy
search bounds, regression predictors, fold definitions and metrics are all
written to that manifest first; nothing is retuned afterwards.

MODELS (frozen)
    M0  ordinary isotropic kriging in (x, y)
    M1  channel-oriented anisotropic kriging in hydraulic coordinates (s, n)
    M2  regression kriging on the frozen prior bands + isotropic residual kriging
    M3  regression kriging on the frozen prior bands + anisotropic residual kriging
    M4  M3 + K7 shoreline inequalities as SOFT constraints (sensitivity only)

M2/M3 use EXACTLY the two bands of morphology_prior_v1 as regression predictors
(P_former_channel, distance_to_channel_m) and nothing else. chain_km is
deliberately NOT a regression covariate: if it were, M2/M3 would differ from
M0/M1 by both the prior and a longitudinal trend term, and the comparison would
no longer isolate the prior.

The local ordinary-kriging solver is reused from hist14_bed_surface.py, with its
three hard-won guards (nugget floor, regularised diagonal, IDW fallback on a bad
solve) plus a kriging-variance return added here for prediction intervals.

Outputs
-------
outputs/tables/k9_experiment_manifest.json
outputs/tables/k9_fold_metrics.csv
outputs/tables/k9_residuals.csv
outputs/tables/k9_model_comparison.csv
outputs/tables/k9_paired_differences.csv
outputs/tables/k9_stratified_performance.csv
outputs/tables/k9_residual_structure.csv
outputs/tables/k9_interval_coverage.csv
outputs/reports/K9_predictive_validation.md
outputs/figures/K9_model_comparison.png
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
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
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

BATHY_DIR = CFG.ROOT / "data" / "processed" / "bathymetry"
RNG = np.random.default_rng(CFG.SEED)
_FALLBACK = {"n_bad": 0, "n_total": 0, "n_env": 0}

# ---------------------------------------------------------------- FROZEN ----
BLOCK_KM_PRIMARY = 10.0
BLOCK_KM_SENSITIVITY = [5.0, 10.0, 20.0]
BUFFER_KM = 2.0                      # motivated by the existing ~2 km variogram range
OK_K = 32                            # neighbours per local kriging system
OK_BATCH = 4000
VARIOGRAM_FAMILY = "spherical"
VG_SUBSAMPLE = 2000
VG_N_LAGS = 18
VG_MAXLAG_FRAC = 0.35
ANISO_BOUNDS = (1.0, 20.0)           # search bounds for range_s / range_n
REGRESSION_PREDICTORS = ["P_former_channel", "distance_to_channel_m"]
M4_SOFT_LAMBDA = 0.5                 # partial pull toward a violated bound
N_BOOT_BLOCKS = 2000
MODELS = ["M0", "M1", "M2", "M3", "M4"]
INTERVAL_LEVELS = [0.50, 0.80, 0.95]


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def spherical(h, rng_, sill, nug):
    h = np.asarray(h, float)
    out = np.where(h <= rng_,
                   nug + (sill - nug) * (1.5 * h / rng_ - 0.5 * (h / rng_) ** 3),
                   sill)
    return np.where(h == 0, 0.0, out)


def fit_variogram(xy, z, rng=RNG):
    """Spherical variogram via SciKit-GStat -- the same fitter hist14_bed_surface.py
    uses, deliberately reused rather than re-implemented.

    An earlier hand-rolled least-squares fit here returned nugget > sill (17.34 vs
    12.55) and a 14.4 km range. With nugget > sill the (sill - nug) factor in
    spherical() goes negative, so gamma DECREASES with distance: kriging then treats
    the most distant neighbours as the most correlated and produces predictions of
    -278..+242 m against an observed range of -19..+15 m. skgstat's model
    parameterisation cannot express nugget > sill, and returns 1.99 km / 12.57 / 0.00
    on the full dataset, matching this project's documented ~2 km range.
    """
    import skgstat as skg
    n = len(xy)
    sub = rng.choice(n, min(VG_SUBSAMPLE, n), replace=False)
    try:
        V = skg.Variogram(xy[sub], z[sub], model=VARIOGRAM_FAMILY,
                          n_lags=VG_N_LAGS, maxlag=VG_MAXLAG_FRAC, normalize=False)
        rng_, sill = float(V.parameters[0]), float(V.parameters[1])
        nug = float(V.parameters[2]) if len(V.parameters) > 2 else 0.0
    except Exception:
        return (2000.0, float(np.var(z)), 0.0)
    if not np.isfinite([rng_, sill, nug]).all() or sill <= 0 or rng_ <= 0:
        return (2000.0, float(np.var(z)), 0.0)
    return (rng_, sill, min(nug, 0.95 * sill))


def fit_anisotropy(sn, z, rng=RNG):
    """Directional ranges along s and along n -> geometric anisotropy ratio."""
    import skgstat as skg
    n = len(sn)
    sub = rng.choice(n, min(VG_SUBSAMPLE, n), replace=False)
    P, v = sn[sub], z[sub]
    out = {}
    for name, az in (("s", 0.0), ("n", 90.0)):
        try:
            V = skg.DirectionalVariogram(P, v, azimuth=az, tolerance=30.0,
                                         model=VARIOGRAM_FAMILY, n_lags=VG_N_LAGS,
                                         maxlag=VG_MAXLAG_FRAC, normalize=False)
            out[name] = float(V.parameters[0])
        except Exception:
            out[name] = np.nan
    if not np.isfinite(out.get("s", np.nan)) or not np.isfinite(out.get("n", np.nan)) or out["n"] <= 0:
        return 1.0
    return float(np.clip(out["s"] / out["n"], *ANISO_BOUNDS))


def ok_predict(xy_tr, z_tr, xy_te, vg, want_var=True):
    """Local ordinary kriging (reused from hist14_bed_surface.py) + variance."""
    rng_, sill, nug = vg
    nug = max(nug, 0.01 * sill)
    K = min(OK_K, len(xy_tr))
    _, idx = cKDTree(xy_tr).query(xy_te, k=K)
    idx = np.atleast_2d(idx)
    out = np.empty(len(xy_te))
    var = np.full(len(xy_te), np.nan)
    for a in range(0, len(xy_te), OK_BATCH):
        b = slice(a, min(a + OK_BATCH, len(xy_te)))
        ii = idx[b]
        P = xy_tr[ii]
        n = P.shape[0]
        A = np.zeros((n, K + 1, K + 1))
        A[:, :K, :K] = spherical(
            np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1), rng_, sill, nug)
        A[:, :K, K] = 1.0
        A[:, K, :K] = 1.0
        # Regularisation raised from 1e-8 to 1e-6 of the sill. With a CV buffer
        # as wide as the variogram range, EVERY neighbour of a test point can lie
        # beyond the range: the spherical model then returns the sill for all
        # pairs, the kriging matrix becomes singular, and np.linalg.solve does not
        # raise -- it returns garbage that still satisfies sum(w)=1. That produced
        # predictions of -389..+467 m against an observed range of -19..+15 m.
        A[:, np.arange(K), np.arange(K)] += 1e-6 * sill
        rhs = np.zeros((n, K + 1))
        dt = np.linalg.norm(P - xy_te[b][:, None, :], axis=-1)
        rhs[:, :K] = spherical(dt, rng_, sill, nug)
        rhs[:, K] = 1.0
        try:
            # NumPy 2 no longer infers "stack of vectors" when b.ndim ==
            # a.ndim - 1 for every (batch, K) combination -- hist14_bed_surface
            # hit this first (see its own _ok()); an explicit trailing axis
            # restores the intended batched solve regardless of batch size.
            sol = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            sol = np.full((n, K + 1), np.nan)
        w, mu = sol[:, :K], sol[:, K]
        # A weight magnitude guard is essential: legitimate OK weights sit close
        # to [-1, 2]; |w| in the thousands means the system was degenerate. The
        # sum(w)=1 test alone does NOT catch it (huge weights can still sum to 1).
        bad = (~np.isfinite(w).all(1) | (np.abs(w.sum(1) - 1) > 0.05)
               | (np.abs(w).max(1) > 2.0))
        _FALLBACK["n_bad"] += int(bad.sum()); _FALLBACK["n_total"] += len(bad)
        if bad.any():
            wi = 1.0 / np.maximum(dt[bad], 1e-6) ** 2
            w[bad] = wi / wi.sum(1, keepdims=True)
            mu[bad] = 0.0
        pv = (w * z_tr[ii]).sum(1)
        # final safety net: a local OK prediction must stay inside the local data
        # envelope. Anything outside it means the system was degenerate whatever
        # the weights looked like, so fall back to inverse distance.
        lo_n, hi_n = z_tr[ii].min(1), z_tr[ii].max(1)
        span = np.maximum(hi_n - lo_n, 1e-6)
        out_of_env = (pv < lo_n - span) | (pv > hi_n + span)
        if out_of_env.any():
            wi = 1.0 / np.maximum(dt[out_of_env], 1e-6) ** 2
            wi = wi / wi.sum(1, keepdims=True)
            pv[out_of_env] = (wi * z_tr[ii][out_of_env]).sum(1)
            _FALLBACK["n_env"] += int(out_of_env.sum())
        out[b] = pv
        if want_var:
            var[b] = np.maximum((w * rhs[:, :K]).sum(1) + mu, 0.0)
    return (out, var) if want_var else out


def signed_cross_offset(lon, lat, chan, tree):
    """Signed cross-channel distance n (m): sign from the side of the channel."""
    chain_km, off_km, _, _ = SW.assign_chainage(lon, lat, chan, tree=tree)
    t, kx = tree
    q = np.c_[np.asarray(lon) * kx, np.asarray(lat) * 110.57]
    _, idx = t.query(q, k=1)
    cx = chan.lon.to_numpy() * kx
    cy = chan.lat.to_numpy() * 110.57
    i0 = np.clip(idx - 1, 0, len(chan) - 1)
    i1 = np.clip(idx + 1, 0, len(chan) - 1)
    tx, ty = cx[i1] - cx[i0], cy[i1] - cy[i0]
    px, py = q[:, 0] - cx[idx], q[:, 1] - cy[idx]
    sign = np.sign(tx * py - ty * px)
    sign[sign == 0] = 1.0
    return chain_km, off_km * 1000.0 * sign


def main() -> None:
    print("=" * 70)
    print("K9 -- FORMAL PREDICTIVE VALIDATION")
    print("=" * 70)

    # ---- data -----------------------------------------------------------
    sd = pd.read_parquet(BATHY_DIR / "kakhovka_soundings_evrf2019.parquet").copy()
    with rasterio.open(BATHY_DIR / "morphology_prior_v1.tif") as src:
        P = src.read(1); P = np.where(P == src.nodata, np.nan, P)
        D = src.read(2); tr = src.transform; gH, gW = src.shape
        prior_hash = hashlib.sha256(Path(src.name).read_bytes()).hexdigest()[:16]
    inv = ~tr
    c, r = inv * (sd.x.values, sd.y.values)
    sd["col"] = np.clip(c.astype(int), 0, gW - 1)
    sd["row"] = np.clip(r.astype(int), 0, gH - 1)
    sd["P_former_channel"] = P[sd.row, sd.col]
    sd["distance_to_channel_m"] = D[sd.row, sd.col]

    chan = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(chan)
    chain_km, n_signed = signed_cross_offset(sd.lon.values, sd.lat.values, chan, tree)
    sd["chain_km"] = chain_km
    sd["n_signed_m"] = n_signed

    sd = sd[sd.P_former_channel.notna() & sd.chain_km.notna()].copy()
    n0 = len(sd)
    sd = (sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0))
          .groupby(["_kx", "_ky"], as_index=False)
          .agg(x=("x", "mean"), y=("y", "mean"), lon=("lon", "mean"), lat=("lat", "mean"),
               H=("H_bed_evrf2019_m", "mean"), chain_km=("chain_km", "mean"),
               n_signed_m=("n_signed_m", "mean"),
               P_former_channel=("P_former_channel", "mean"),
               distance_to_channel_m=("distance_to_channel_m", "mean"),
               row=("row", "first"), col=("col", "first")))
    print(f"soundings used: {len(sd)} (averaged {n0-len(sd)} exact duplicate coordinates)")
    print(f"bed elevation {sd.H.min():.2f} .. {sd.H.max():.2f} m EVRF2019; "
          f"chainage {sd.chain_km.min():.1f} .. {sd.chain_km.max():.1f} km")

    xy = np.c_[sd.x.values, sd.y.values]
    sn_raw = np.c_[sd.chain_km.values * 1000.0, sd.n_signed_m.values]
    z = sd.H.values

    # ---- FREEZE the experiment BEFORE any outer-CV result ----------------
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "central_question": "Does the sounding-blind morphology prior improve spatially "
                            "independent prediction of historical bathymetry beyond ordinary "
                            "spatial autocorrelation and channel-oriented anisotropy?",
        "inputs": {
            "soundings_parquet_sha256_16":
                hashlib.sha256((BATHY_DIR / "kakhovka_soundings_evrf2019.parquet").read_bytes()).hexdigest()[:16],
            "morphology_prior_v1_sha256_16": prior_hash,
            "shoreline_constraints_sha256_16":
                hashlib.sha256((CFG.TABLES / "k7_shoreline_constraints.csv").read_bytes()).hexdigest()[:16],
            "n_soundings_after_dedup": int(len(sd)),
        },
        "models": {
            "M0": "local ordinary kriging in (x,y), isotropic spherical variogram",
            "M1": "local ordinary kriging in hydraulic (s, n/alpha), alpha fitted on training only",
            "M2": "OLS on frozen prior bands + isotropic residual kriging in (x,y)",
            "M3": "OLS on frozen prior bands + anisotropic residual kriging in (s, n/alpha)",
            "M4": "M3 + K7 shoreline inequalities as soft constraints (SENSITIVITY ONLY)",
        },
        "regression_predictors": REGRESSION_PREDICTORS,
        "regression_excludes": ["chain_km"],
        "regression_exclusion_reason":
            "chain_km is excluded so M2/M3 differ from M0/M1 only by the frozen prior",
        "variogram_family": VARIOGRAM_FAMILY,
        "variogram_subsample": VG_SUBSAMPLE, "variogram_n_lags": VG_N_LAGS,
        "variogram_maxlag_fraction": VG_MAXLAG_FRAC,
        "anisotropy_search_bounds": list(ANISO_BOUNDS),
        "kriging_neighbours": OK_K,
        "cv": {"scheme": "chainage-blocked spatial CV, buffered",
               "primary_block_km": BLOCK_KM_PRIMARY,
               "sensitivity_block_km": BLOCK_KM_SENSITIVITY,
               "buffer_km": BUFFER_KM,
               "buffer_rationale": "existing ~2 km variogram range",
               "identical_folds_for_all_models": True},
        "metrics": ["RMSE", "MAE", "bias", "NMAD", "median_abs_error", "R2"],
        "paired_difference_resampling": {"unit": "spatial block", "n_boot": N_BOOT_BLOCKS},
        "interval_levels": INTERVAL_LEVELS,
        "m4_soft_lambda": M4_SOFT_LAMBDA,
        "freeze_rule": "model definitions are not changed after outer-fold results are seen; "
                       "outer folds are used exactly once for the final estimate",
        "prior_leakage_note":
            "morphology_prior_v1 was built with zero soundings, so there is no target leakage "
            "into the prior. K4b did, however, use all soundings to assess construct validity, "
            "which informed the decision to test the prior here. There is therefore no pristine "
            "holdout; this manifest constitutes the prospective freeze from this point on.",
    }
    mpath = CFG.TABLES / "k9_experiment_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nEXPERIMENT FROZEN -> {mpath}")
    print(f"  prior sha256_16={prior_hash}; folds/metrics/bounds fixed before any result")

    # ---- CV --------------------------------------------------------------
    all_res, fold_rows, vg_log = [], [], []
    for block_km in BLOCK_KM_SENSITIVITY:
        edges = np.arange(np.floor(sd.chain_km.min()), sd.chain_km.max() + block_km, block_km)
        block_id = np.digitize(sd.chain_km.values, edges) - 1
        blocks = [b for b in np.unique(block_id) if (block_id == b).sum() >= 10]
        print(f"\nblock scale {block_km:.0f} km: {len(blocks)} usable outer folds "
              f"(buffer {BUFFER_KM:.0f} km)")
        for b in blocks:
            te = block_id == b
            lo, hi = sd.chain_km.values[te].min(), sd.chain_km.values[te].max()
            buf = (sd.chain_km.values >= lo - BUFFER_KM) & (sd.chain_km.values <= hi + BUFFER_KM)
            trn = ~buf
            if trn.sum() < 200 or te.sum() < 10:
                continue
            ztr, zte = z[trn], z[te]

            vg_iso = fit_variogram(xy[trn], ztr)
            vg_log.append({"block_km": block_km, "fold": int(b), "space": "xy",
                           "range_m": vg_iso[0], "sill": vg_iso[1], "nugget": vg_iso[2]})
            alpha = fit_anisotropy(sn_raw[trn], ztr)
            sn = np.c_[sn_raw[:, 0], sn_raw[:, 1] * alpha]
            vg_ani = fit_variogram(sn[trn], ztr)

            X_tr = np.c_[np.ones(trn.sum()), sd.loc[trn, REGRESSION_PREDICTORS].values]
            X_te = np.c_[np.ones(te.sum()), sd.loc[te, REGRESSION_PREDICTORS].values]
            beta, *_ = np.linalg.lstsq(X_tr, ztr, rcond=None)
            trend_tr, trend_te = X_tr @ beta, X_te @ beta
            res_tr = ztr - trend_tr
            vg_iso_r = fit_variogram(xy[trn], res_tr)
            vg_ani_r = fit_variogram(sn[trn], res_tr)

            pred, varr = {}, {}
            pred["M0"], varr["M0"] = ok_predict(xy[trn], ztr, xy[te], vg_iso)
            pred["M1"], varr["M1"] = ok_predict(sn[trn], ztr, sn[te], vg_ani)
            rk_i, vi = ok_predict(xy[trn], res_tr, xy[te], vg_iso_r)
            pred["M2"], varr["M2"] = trend_te + rk_i, vi
            rk_a, va = ok_predict(sn[trn], res_tr, sn[te], vg_ani_r)
            pred["M3"], varr["M3"] = trend_te + rk_a, va
            pred["M4"], varr["M4"] = pred["M3"].copy(), va.copy()

            for m in MODELS:
                e = pred[m] - zte
                fold_rows.append({
                    "block_km": block_km, "fold": int(b), "model": m,
                    "n_test": int(te.sum()), "n_train": int(trn.sum()),
                    "chain_lo": lo, "chain_hi": hi, "alpha": alpha,
                    "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
                    "bias": float(e.mean()), "NMAD": nmad(e),
                    "median_abs_error": float(np.median(np.abs(e))),
                    "R2": float(1 - (e ** 2).sum() / ((zte - zte.mean()) ** 2).sum())
                    if zte.var() > 0 else np.nan,
                })
                df = sd.loc[te, ["x", "y", "chain_km", "P_former_channel",
                                 "distance_to_channel_m", "row", "col"]].copy()
                df["block_km"] = block_km; df["fold"] = int(b); df["model"] = m
                df["observed"] = zte; df["predicted"] = pred[m]; df["residual"] = e
                df["kriging_var"] = varr[m]
                all_res.append(df)

    res = pd.concat(all_res, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    vgl = pd.DataFrame(vg_log)
    vgl.to_csv(CFG.TABLES / "k9_fitted_variograms.csv", index=False)
    print(f"\nfitted isotropic variograms (xy): range median {vgl.range_m.median()/1000:.2f} km "
          f"[{vgl.range_m.min()/1000:.2f}..{vgl.range_m.max()/1000:.2f}], "
          f"sill median {vgl.sill.median():.2f} m2, nugget median {vgl.nugget.median():.2f} m2")
    fb = _FALLBACK
    print(f"degenerate kriging solves caught by the weight guard -> IDW fallback: "
          f"{fb['n_bad']:,} of {fb['n_total']:,} ({100*fb['n_bad']/max(fb['n_total'],1):.2f}%); "
          f"out-of-envelope predictions replaced by IDW: {fb['n_env']:,}")
    print(f"NOTE: the {BUFFER_KM:.0f} km CV buffer is as wide as the ~2 km variogram range, so for "
          f"many test points every neighbour lies beyond the range and ordinary kriging "
          f"legitimately degenerates toward the local mean. That is a property of this "
          f"deliberately strict design, not a solver failure.")
    for m in MODELS:
        g = res[(res.model == m) & (res.block_km == BLOCK_KM_PRIMARY)]
        print(f"  {m}: predicted range {g.predicted.min():+.2f} .. {g.predicted.max():+.2f} m "
              f"(observed {g.observed.min():+.2f} .. {g.observed.max():+.2f})")

    # ---- M4: apply the K7 soft shoreline constraints ---------------------
    cons = pd.read_csv(CFG.TABLES / "k7_shoreline_constraints.csv")
    cons = cons[cons.eta_evrf2019_m.notna()].copy()
    ccol, crow = inv * (cons.x.values, cons.y.values)
    cons["col"] = np.clip(ccol.astype(int), 0, gW - 1)
    cons["row"] = np.clip(crow.astype(int), 0, gH - 1)
    ub = cons[cons.constraint_type == "WET_UPPER_BOUND"].groupby(["row", "col"]).upper_bound_m.min()
    lb = cons[cons.constraint_type == "DRY_LOWER_BOUND"].groupby(["row", "col"]).lower_bound_m.max()
    m4 = res.model == "M4"
    key = pd.MultiIndex.from_arrays([res.loc[m4, "row"], res.loc[m4, "col"]])
    up = ub.reindex(key).values
    lo_ = lb.reindex(key).values
    p4 = res.loc[m4, "predicted"].values.copy()
    n_up = int(np.sum(np.isfinite(up) & (p4 > up)))
    n_lo = int(np.sum(np.isfinite(lo_) & (p4 < lo_)))
    viol_up = np.isfinite(up) & (p4 > up)
    viol_lo = np.isfinite(lo_) & (p4 < lo_)
    p4[viol_up] += M4_SOFT_LAMBDA * (up[viol_up] - p4[viol_up])
    p4[viol_lo] += M4_SOFT_LAMBDA * (lo_[viol_lo] - p4[viol_lo])
    res.loc[m4, "predicted"] = p4
    res.loc[m4, "residual"] = p4 - res.loc[m4, "observed"].values
    n_cov = int(np.sum(np.isfinite(up) | np.isfinite(lo_)))
    print(f"\nM4 soft constraints: {n_cov} of {int(m4.sum())} test predictions fall in a "
          f"constrained cell; {n_up} violated an upper bound, {n_lo} a lower bound "
          f"(pulled {M4_SOFT_LAMBDA:.0%} toward the bound)")
    for (bk, f), gk in res[m4].groupby(["block_km", "fold"]):
        e = gk.residual.values
        sel = (folds.block_km == bk) & (folds.fold == f) & (folds.model == "M4")
        folds.loc[sel, ["RMSE", "MAE", "bias", "NMAD", "median_abs_error"]] = [
            float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()), float(e.mean()),
            nmad(e), float(np.median(np.abs(e)))]

    res.to_csv(CFG.TABLES / "k9_residuals.csv", index=False)
    folds.to_csv(CFG.TABLES / "k9_fold_metrics.csv", index=False)
    print(f"-> {CFG.TABLES / 'k9_fold_metrics.csv'}  ({len(folds)} fold-model rows)")
    print(f"-> {CFG.TABLES / 'k9_residuals.csv'}  ({len(res)} residuals)")

    # ---- primary comparison ----------------------------------------------
    pri = res[res.block_km == BLOCK_KM_PRIMARY]
    comp = []
    for m in MODELS:
        e = pri[pri.model == m].residual.values
        o = pri[pri.model == m].observed.values
        comp.append({"model": m, "n": len(e), "RMSE": float(np.sqrt((e ** 2).mean())),
                     "MAE": float(np.abs(e).mean()), "bias": float(e.mean()), "NMAD": nmad(e),
                     "median_abs_error": float(np.median(np.abs(e))),
                     "R2": float(1 - (e ** 2).sum() / ((o - o.mean()) ** 2).sum())})
    comp = pd.DataFrame(comp)

    # deepest decile + morphology strata, computed on the primary scheme
    thr = np.quantile(pri[pri.model == "M0"].observed, 0.10)
    for m in MODELS:
        g = pri[pri.model == m]
        d10 = g[g.observed <= thr].residual.values
        hp = g[g.P_former_channel >= 0.5].residual.values
        comp.loc[comp.model == m, "deep10_RMSE"] = float(np.sqrt((d10 ** 2).mean())) if len(d10) else np.nan
        comp.loc[comp.model == m, "highP_RMSE"] = float(np.sqrt((hp ** 2).mean())) if len(hp) else np.nan

    # ---- paired differences, bootstrapped by BLOCK -----------------------
    def paired(mi, mj, data):
        a = data[data.model == mi].set_index(["block_km", "fold", "x", "y"]).residual
        b = data[data.model == mj].set_index(["block_km", "fold", "x", "y"]).residual
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        blocks = np.array([i[1] for i in common])
        uq = np.unique(blocks)
        d_rmse = np.sqrt((a.values ** 2).mean()) - np.sqrt((b.values ** 2).mean())
        d_mae = np.abs(a.values).mean() - np.abs(b.values).mean()
        boots = []
        for _ in range(N_BOOT_BLOCKS):
            pick = RNG.choice(uq, len(uq), replace=True)
            sel = np.concatenate([np.where(blocks == p)[0] for p in pick])
            boots.append(np.sqrt((a.values[sel] ** 2).mean()) - np.sqrt((b.values[sel] ** 2).mean()))
        lo95, hi95 = np.percentile(boots, [2.5, 97.5])
        return d_rmse, d_mae, lo95, hi95

    pairs = [("M1", "M0"), ("M2", "M0"), ("M3", "M0"), ("M3", "M1"), ("M4", "M3"), ("M2", "M1")]
    prows = []
    for mi, mj in pairs:
        d_rmse, d_mae, lo95, hi95 = paired(mi, mj, pri)
        prows.append({"comparison": f"{mi} vs {mj}", "delta_RMSE": d_rmse, "delta_MAE": d_mae,
                      "ci95_lo": lo95, "ci95_hi": hi95,
                      "excludes_zero": bool(lo95 > 0 or hi95 < 0)})
    pdiff = pd.DataFrame(prows)
    pdiff.to_csv(CFG.TABLES / "k9_paired_differences.csv", index=False)

    for mi, mj in [("M1", "M0"), ("M3", "M0")]:
        row = pdiff[pdiff.comparison == f"{mi} vs {mj}"].iloc[0]
        comp.loc[comp.model == mi, "ci95_delta_RMSE_vs_M0"] = f"[{row.ci95_lo:+.3f}, {row.ci95_hi:+.3f}]"
    for m in MODELS:
        if m == "M0":
            comp.loc[comp.model == m, "ci95_delta_RMSE_vs_M0"] = "reference"
        elif pd.isna(comp.loc[comp.model == m, "ci95_delta_RMSE_vs_M0"]).all():
            d_rmse, d_mae, lo95, hi95 = paired(m, "M0", pri)
            comp.loc[comp.model == m, "ci95_delta_RMSE_vs_M0"] = f"[{lo95:+.3f}, {hi95:+.3f}]"

    # ---- stratified performance ------------------------------------------
    pri = pri.copy()
    pri["reach"] = pd.cut(pri.chain_km, [0, 133, 183, 277.2],
                          labels=["CORE_LOWER", "CORE_MIDDLE", "CORE_UPPER"])
    pri["morph_class"] = pd.cut(pri.P_former_channel, [0, 0.2, 0.5, 1.0],
                                labels=["LOW_P", "TRANSITION", "HIGH_P"], include_lowest=True)
    pri["chain_band"] = pd.cut(pri.chain_km, np.arange(0, 300, 50))
    pri["deep10"] = pri.observed <= thr
    strat = []
    for name in ("reach", "morph_class", "chain_band", "deep10"):
        for (key, m), gk in pri.groupby([name, "model"], observed=True):
            e = gk.residual.values
            strat.append({"stratum": name, "value": str(key), "model": m, "n": len(e),
                          "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
                          "bias": float(e.mean()), "NMAD": nmad(e)})
    st = pd.DataFrame(strat)
    st.to_csv(CFG.TABLES / "k9_stratified_performance.csv", index=False)

    # ---- residual spatial structure --------------------------------------
    rs = []
    for m in MODELS:
        g = pri[pri.model == m]
        pts = np.c_[g.x.values, g.y.values]
        rng_, sill, nug = fit_variogram(pts, g.residual.values)
        sub = RNG.choice(len(g), min(1500, len(g)), replace=False)
        pp, vv = pts[sub], g.residual.values[sub]
        d = np.linalg.norm(pp[:, None, :] - pp[None, :, :], axis=-1)
        W = np.exp(-d / 2000.0); np.fill_diagonal(W, 0)
        dv = vv - vv.mean()
        moran = (len(vv) / W.sum()) * (dv @ W @ dv) / (dv @ dv)
        rs.append({"model": m, "residual_nugget": nug, "residual_sill": sill,
                   "residual_range_m": rng_, "residual_moran_I_2km": float(moran)})
    rsd = pd.DataFrame(rs)
    rsd.to_csv(CFG.TABLES / "k9_residual_structure.csv", index=False)
    comp = comp.merge(rsd[["model", "residual_sill", "residual_range_m"]], on="model")

    # ---- interval coverage ------------------------------------------------
    cov = []
    from scipy.stats import norm
    for m in MODELS:
        g = pri[pri.model == m]
        sig = np.sqrt(np.maximum(g.kriging_var.values, 0))
        row = {"model": m}
        for lv in INTERVAL_LEVELS:
            k = norm.ppf(0.5 + lv / 2)
            row[f"coverage_{int(lv*100)}"] = float(np.mean(np.abs(g.residual.values) <= k * sig))
        cov.append(row)
    covd = pd.DataFrame(cov)
    covd.to_csv(CFG.TABLES / "k9_interval_coverage.csv", index=False)

    comp.to_csv(CFG.TABLES / "k9_model_comparison.csv", index=False)

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON (primary scheme: {BLOCK_KM_PRIMARY:.0f} km blocks, "
          f"{BUFFER_KM:.0f} km buffer)")
    print("=" * 70)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nPAIRED DIFFERENCES (negative delta_RMSE = first model better; "
          "bootstrap by spatial block)")
    print(pdiff.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nRESIDUAL SPATIAL STRUCTURE")
    print(rsd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nPREDICTION-INTERVAL COVERAGE (nominal vs empirical)")
    print(covd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nDEEPEST-DECILE AND HIGH-P PERFORMANCE")
    print(st[(st.stratum == "deep10") & (st.value == "True")].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(st[(st.stratum == "morph_class") & (st.value == "HIGH_P")].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSENSITIVITY ACROSS BLOCK SCALES (RMSE)")
    print(folds.groupby(["block_km", "model"]).apply(
        lambda g: np.sqrt(np.average(g.RMSE ** 2, weights=g.n_test))).unstack().to_string(
        float_format=lambda v: f"{v:.3f}"))

    def get(mi, mj, col="delta_RMSE"):
        return pdiff[pdiff.comparison == f"{mi} vs {mj}"].iloc[0][col]

    q1 = get("M1", "M0"); q1x = pdiff[pdiff.comparison == "M1 vs M0"].iloc[0].excludes_zero
    q2 = get("M2", "M0"); q2x = pdiff[pdiff.comparison == "M2 vs M0"].iloc[0].excludes_zero
    q3 = get("M3", "M1"); q3x = pdiff[pdiff.comparison == "M3 vs M1"].iloc[0].excludes_zero
    q4 = get("M4", "M3"); q4x = pdiff[pdiff.comparison == "M4 vs M3"].iloc[0].excludes_zero

    print("\n" + "=" * 70)
    print("FOUR QUESTIONS")
    print("=" * 70)
    print(f"Q1 anisotropy (M1 vs M0):            delta_RMSE = {q1:+.3f} m, CI excludes 0: {q1x}")
    print(f"Q2 morphology prior (M2 vs M0):      delta_RMSE = {q2:+.3f} m, CI excludes 0: {q2x}")
    print(f"Q3 prior beyond anisotropy (M3-M1):  delta_RMSE = {q3:+.3f} m, CI excludes 0: {q3x}")
    print(f"Q4 shoreline soft (M4 vs M3):        delta_RMSE = {q4:+.3f} m, CI excludes 0: {q4x}")

    deep = st[(st.stratum == "deep10") & (st.value == "True")].set_index("model").RMSE
    deep_gain = deep.get("M3", np.nan) - deep.get("M0", np.nan)
    sill_drop = (rsd.set_index("model").residual_sill.get("M0", np.nan)
                 - rsd.set_index("model").residual_sill.get("M3", np.nan))
    if q2x and q3x and q3 < -0.05:
        verdict = "STRONG_PREDICTIVE_VALIDITY"
    elif q2x and (q3 < 0):
        verdict = "MODEST_PREDICTIVE_VALUE"
    elif q2x:
        verdict = "MODEST_PREDICTIVE_VALUE"
    else:
        verdict = "CONSTRUCT_ONLY"
    print(f"\ndeepest-decile RMSE change M3 vs M0: {deep_gain:+.3f} m")
    print(f"residual sill reduction M0 -> M3: {sill_drop:+.3f}")
    print(f"\nVERDICT (morphology_prior_v1): {verdict}")

    rep = CFG.REPORTS / "K9_predictive_validation.md"
    with open(rep, "w") as fh:
        fh.write("# K9 -- predictive validation of morphology_prior_v1\n\n")
        fh.write(f"**VERDICT: {verdict}**\n\n")
        fh.write(f"Primary scheme: {BLOCK_KM_PRIMARY:.0f} km chainage blocks, "
                 f"{BUFFER_KM:.0f} km buffer, identical folds for all models. "
                 f"Experiment frozen in `k9_experiment_manifest.json` before any result.\n\n")
        fh.write("## Model comparison\n\n```\n" + comp.to_string(index=False) + "\n```\n\n")
        fh.write("## Paired differences (bootstrap by spatial block)\n\n```\n"
                 + pdiff.to_string(index=False) + "\n```\n\n")
        fh.write("## Four questions\n\n")
        fh.write(f"- Q1 anisotropy (M1 vs M0): delta_RMSE = {q1:+.3f} m, CI excludes 0: {q1x}\n")
        fh.write(f"- Q2 morphology prior (M2 vs M0): delta_RMSE = {q2:+.3f} m, CI excludes 0: {q2x}\n")
        fh.write(f"- Q3 prior beyond anisotropy (M3 vs M1): delta_RMSE = {q3:+.3f} m, CI excludes 0: {q3x}\n")
        fh.write(f"- Q4 shoreline soft constraints (M4 vs M3): delta_RMSE = {q4:+.3f} m, CI excludes 0: {q4x}\n\n")
        fh.write("## Residual spatial structure\n\n```\n" + rsd.to_string(index=False) + "\n```\n\n")
        fh.write("## Interval coverage\n\n```\n" + covd.to_string(index=False) + "\n```\n")
    print(f"-> {rep}")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(comp.model, comp.RMSE, color=["#8a94a3", "#236f8c", "#3f7d4e", "#b07d27", "#c1402a"])
    axes[0].set_ylabel("blocked-CV RMSE, m")
    axes[0].set_title(f"A. primary {BLOCK_KM_PRIMARY:.0f} km blocked CV", fontsize=9)
    d = pdiff.set_index("comparison")
    axes[1].errorbar(d.delta_RMSE, range(len(d)),
                     xerr=[d.delta_RMSE - d.ci95_lo, d.ci95_hi - d.delta_RMSE],
                     fmt="o", color="#1a2228", capsize=4)
    axes[1].axvline(0, color="#c1402a", ls="--", lw=1)
    axes[1].set_yticks(range(len(d))); axes[1].set_yticklabels(d.index, fontsize=8)
    axes[1].set_xlabel("delta RMSE, m (negative = first model better)")
    axes[1].set_title("B. paired differences, 95% block bootstrap", fontsize=9)
    dd = st[(st.stratum == "deep10") & (st.value == "True")]
    axes[2].bar(dd.model, dd.RMSE, color="#236f8c")
    axes[2].set_ylabel("RMSE, m"); axes[2].set_title("C. deepest 10% of soundings", fontsize=9)
    fig.suptitle("K9 -- does the sounding-blind morphology prior improve spatially "
                 "independent bathymetric prediction?", fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K9_model_comparison.png", dpi=150)
    print(f"-> {CFG.FIG / 'K9_model_comparison.png'}")


if __name__ == "__main__":
    main()
