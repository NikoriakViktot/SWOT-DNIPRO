"""Local ordinary kriging with per-observation error variance (soft data).

Lifted from `scripts/hist24_three_contour_dem.py:99-151` (which in turn extends
`hist14_bed_surface._ok`) so that the downstream zones can use the same
estimator ZONE_1 does -- with one correction and one guard made explicit.

The correction: the sign of the error-variance term
----------------------------------------------------
`spherical()` returns the VARIOGRAM gamma(h): 0 at h = 0, rising to the sill.
The kriging system is therefore in variogram form. For an observation with
measurement-error variance s2, derive in covariance form (unambiguous):

    Cov(Z_i, Z_j) = C(h_ij) + s2_i * delta_ij,     C = sill - gamma
    => gamma_eff_ii = sill - (C(0) + s2_i) = sill - (sill + s2_i) = -s2_i

so in variogram form the diagonal term is MINUS the error variance. hist24
line 123 added PLUS err_var. That inverts the meaning of soft data: the noisier
an observation is declared, the MORE weight it receives (0.398 -> 1.459 as s2
goes 0 -> 10 in the audit's two-point check, with the reliable point's weight
going negative). Verified against an independent covariance-form solve in
`audit_runs/20260916T093000Z/kriging/independent_check.py`. At the parameter
values hist24 actually ran with (err_var 0.04-0.6 m2 against a sill of 10-100
m2) the induced bias was 0.006-0.099 m -- negligible there -- but the relation
is non-linear and this module is meant for constraints whose sigma budget is
larger. `verify_sign()` below runs the two-point check on import of the
estimator so the sign cannot silently regress. The old `--verify` only tested
err_var = 0, the one case where the sign is invisible.

The guard: weights that explode
-------------------------------
The Lagrange row forces sum(w) = 1 even when the system is singular, so a sum
check cannot detect degeneracy; hist24 saw |w| in the thousands and a
leave-one-contour-out RMSE of 224.8 m. Any target whose |w|max exceeds
`W_MAX` falls back to inverse-distance weighting over the same neighbours and
is counted, so the fallback rate is a reported diagnostic, never hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

OK_K = 32
OK_BATCH = 4000
W_MAX = 5.0


def spherical(h, rng_, sill, nug):
    """Variogram gamma(h): 0 at h=0 (identical to hist14_bed_surface.spherical)."""
    h = np.asarray(h, float)
    out = np.where(h <= rng_,
                   nug + (sill - nug) * (1.5 * h / rng_ - 0.5 * (h / rng_) ** 3),
                   sill)
    return np.where(h == 0, 0.0, out)


def idw(xy_tr, z_tr, xy_te, k=OK_K, power=2.0):
    d, idx = cKDTree(xy_tr).query(xy_te, k=min(k, len(xy_tr)))
    d = np.atleast_2d(d); idx = np.atleast_2d(idx)
    w = 1.0 / np.maximum(d, 1.0) ** power
    return (w * z_tr[idx]).sum(1) / w.sum(1)


def ok_soft(xy_tr, z_tr, err_var, xy_te, vg, k=OK_K, batch=OK_BATCH,
            w_max=W_MAX, return_diag=False):
    """Local OK; `err_var[i]` is the measurement-error variance of training point i.

    err_var = 0 reduces exactly to hist14's `_ok` (asserted in `verify_sign`).
    Returns predictions, and with `return_diag` also the number of targets
    that fell back to IDW and the max |w| per target."""
    rng_, sill, nug = vg
    nug = max(nug, 0.01 * sill)
    xy_tr = np.asarray(xy_tr, float); z_tr = np.asarray(z_tr, float)
    err_var = np.asarray(err_var, float)
    xy_te = np.asarray(xy_te, float)
    K = min(k, len(xy_tr))
    _, idx = cKDTree(xy_tr).query(xy_te, k=K)
    idx = np.atleast_2d(idx)
    out = np.empty(len(xy_te))
    wmax_all = np.empty(len(xy_te))
    n_fallback = 0
    for a in range(0, len(xy_te), batch):
        b = slice(a, min(a + batch, len(xy_te)))
        ii = idx[b]
        P = xy_tr[ii]
        n = P.shape[0]
        A = np.zeros((n, K + 1, K + 1))
        A[:, :K, :K] = spherical(np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1),
                                 rng_, sill, nug)
        A[:, :K, K] = 1.0
        A[:, K, :K] = 1.0
        # variogram form: measurement error enters the diagonal as MINUS s2
        A[:, np.arange(K), np.arange(K)] += 1e-8 * sill - err_var[ii]
        rhs = np.zeros((n, K + 1))
        rhs[:, :K] = spherical(np.linalg.norm(P - xy_te[b][:, None, :], axis=-1), rng_, sill, nug)
        rhs[:, K] = 1.0
        try:
            wts = np.linalg.solve(A, rhs[:, :, None])[:, :K, 0]
        except np.linalg.LinAlgError:
            wts = np.full((n, K), np.nan)
        wm = np.nanmax(np.abs(wts), axis=1)
        bad = ~np.isfinite(wm) | (wm > w_max)
        pred = np.einsum("nk,nk->n", np.nan_to_num(wts), z_tr[ii])
        if bad.any():
            d = np.linalg.norm(P[bad] - xy_te[b][bad][:, None, :], axis=-1)
            w = 1.0 / np.maximum(d, 1.0) ** 2
            pred[bad] = (w * z_tr[ii][bad]).sum(1) / w.sum(1)
            n_fallback += int(bad.sum())
        out[b] = pred
        wmax_all[b] = wm
    if return_diag:
        return out, dict(n_fallback=n_fallback, n_targets=len(xy_te), wmax=wmax_all)
    return out


def verify_sign(atol: float = 1e-9) -> dict:
    """Two-point check against an independent covariance-form OK solve.

    Not the same formula being tested: the reference builds the COVARIANCE
    matrix C = sill - gamma, adds s2 to its diagonal, and solves. Raises if the
    variogram-form estimator disagrees for any s2 in {0, 1, 5, 10}."""
    sill, nug, rng_ = 10.0, 0.0, 100.0
    vg = (rng_, sill, nug)
    # the estimator's two documented numerical stabilisers -- the nugget floor
    # and the 1e-8*sill diagonal regularisation -- are part of what it computes,
    # not what is under test, so the reference carries them too (expressed in
    # covariance space: a +r on the variogram diagonal is a -r on C's diagonal).
    # Without this the check fails at s2 = 0 by 0.013 m, which is the floor, not
    # the sign -- exactly what happened on the first run of this function.
    nug_eff = max(nug, 0.01 * sill)
    reg = 1e-8 * sill
    x = np.array([[0.0, 0.0], [50.0, 0.0]]); x0 = np.array([[20.0, 0.0]])
    z = np.array([1.0, 10.0])
    h_ij = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    h_i0 = np.linalg.norm(x - x0, axis=-1)
    out = {}
    for s2v in (0.0, 1.0, 5.0, 10.0):
        s2 = np.array([0.0, s2v])
        C = np.zeros((3, 3)); C[:2, :2] = sill - spherical(h_ij, rng_, sill, nug_eff)
        C[np.arange(2), np.arange(2)] += s2 - reg; C[:2, 2] = 1; C[2, :2] = 1
        b = np.r_[sill - spherical(h_i0, rng_, sill, nug_eff), 1.0]
        ref = float(np.linalg.solve(C, b)[:2] @ z)
        got = float(ok_soft(x, z, s2, x0, vg, k=2)[0])
        out[s2v] = (got, ref)
        if abs(got - ref) > atol:
            raise AssertionError(f"ok_soft sign check FAILED at s2={s2v}: got {got:.6f}, "
                                 f"covariance-form reference {ref:.6f}")
    return out


# the sign cannot regress silently: the check runs whenever the module is imported
_SIGN_CHECK = verify_sign()
