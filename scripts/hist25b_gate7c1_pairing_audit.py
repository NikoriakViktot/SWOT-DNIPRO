#!/usr/bin/env python
"""GATE 7C1 — correspondence / pairing validation, before any H1/H2/H3 re-run.

WHY THIS GATE EXISTS. The same-date controls returned d_raw = -0.0 m on
2019-03-18, which looks decisive until you notice the pairing fraction was
28.3%. That result therefore only says

    the part of the shoreline the matcher COULD pair has no systematic offset

and says nothing about the other 71.7%. Only 9.4% of the unpaired vertices sit
near an S2 coverage gap, so the rest is a genuine correspondence failure, not
missing optical data. With P90(|d_raw|) at 243-261 m on SAME-DATE scenes, the
dominant open risk is no longer the estimator -- it is the correspondence
model. Area agreement of 0.8-1.3% does NOT validate shoreline geometry: two
contours can enclose near-identical area and still differ locally through bays,
islands, side channels, vegetation edges and disconnected pieces.

FIVE CHECKS
  1. Pairing fraction by shoreline LENGTH, not by vertex count. Vertex counts
     depend on densification; length is physical.
  2. Paired / unpaired map with an explicit reason per vertex.
  3. Symmetric matching, S1->S2 and S2->S1.
  4. Sensitivity to max_match_distance over 25..250 m.
  5. Normal-intersection matching instead of nearest-point.

THE SYMMETRY SIGN, which is a trap. Expressed in the frozen convention
("positive = SAR water extent LARGER than optical") the two directions use
OPPOSITE inside-tests:
    S1->S2: a SAR vertex inside optical water  -> SAR extent smaller -> NEGATIVE
    S2->S1: an optical vertex inside SAR water -> SAR extent larger  -> POSITIVE
so a correct implementation gives d_12 ~ +d_21, NOT d_12 ~ -d_21. The
antisymmetric expectation is what you get from applying one sign rule blindly
in both directions. Both are reported below so the distinction is visible
rather than argued.

TWO END-TO-END TESTS ON REAL KAKHOVKA GEOMETRY, not circles and half-planes.
  A. Offset injection. Take the signed distance field D of the real cleaned
     2019 optical water body -- islands, narrow channels, concave bays and all
     -- and form F = D - delta. The zero level of that field is EXACTLY the
     real shoreline displaced by delta along its own normal, because D is a
     true distance field. Push it through the production matcher and check
     recovery for delta = +/-5, 10, 20, 40 m.
  B. Stage sign. Build a synthetic optical field displaced by L = dH/s and
     confirm end-to-end that d_residual = d_raw - d_stage returns to zero with
     the right sign. The same-date controls could NOT test this: dH was
     exactly 0.000, so the stage correction had zero lever arm.

Outputs
-------
outputs/tables/gate7c1_pairing_length.csv
outputs/tables/gate7c1_threshold_sensitivity.csv
outputs/tables/gate7c1_synthetic_recovery.csv
outputs/figures/historical_bathymetry/png/gate7c1_pairing_map.png
outputs/figures/historical_bathymetry/png/gate7c1_sensitivity.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from swot_dnipro import config as CFG
from hist25b_gate6_event_qualification import load_frozen_manifest
from hist25b_gate7c_v2_same_date_controls import (
    CELL, CONTROL_DATES, FIGDIR, SAS, build_fields, build_grid, clean_water)

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
THRESHOLDS_M = (25.0, 50.0, 75.0, 100.0, 150.0, 250.0)
NORMAL_MAX_M = 250.0       # search half-length along the cross-shore normal
NORMAL_STEP_M = 2.0        # ray sampling; sub-pixel comes from interpolation
GAP_NEAR_M = 200.0
INJECT_M = (-40.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 40.0)
STAGE_DH_M = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
STAGE_SLOPE = 0.00293      # the slope measured on 2019-03-18, m/m


# ----------------------------------------------------------------- geometry --
def contour_points(F, valid, G):
    """Contour vertices WITH a per-vertex representative length and tangent.

    The length weight is computed on the FULL polyline before the validity
    filter, so that dropping vertices never inflates the length of the ones
    that remain. Returns x, y, row, col, seg_length_m, normal_x, normal_y."""
    x0, y1 = G["x0"], G["y1"]
    out = []
    for c in find_contours(np.nan_to_num(F, nan=1e6), level=0.0):
        r, cc = c[:, 0], c[:, 1]
        if len(r) < 3:
            continue
        x = x0 + (cc + 0.5) * CELL
        y = y1 - (r + 0.5) * CELL
        seg = np.hypot(np.diff(x), np.diff(y))
        wl = np.zeros(len(x))
        wl[:-1] += seg / 2.0
        wl[1:] += seg / 2.0
        tx, ty = np.gradient(x), np.gradient(y)
        tn = np.hypot(tx, ty)
        tn[tn == 0] = 1.0
        # unit normal = tangent rotated by 90 deg; direction is irrelevant
        # because the ray is searched both ways
        nxv, nyv = -ty / tn, tx / tn
        ri = np.clip(np.round(r).astype(int), 0, F.shape[0] - 1)
        ci = np.clip(np.round(cc).astype(int), 0, F.shape[1] - 1)
        k = valid[ri, ci]
        if not k.any():
            continue
        out.append(np.c_[x[k], y[k], ri[k], ci[k], wl[k], nxv[k], nyv[k]])
    return np.vstack(out) if out else np.empty((0, 7))


def sample_field(F, valid, X, Y, G):
    """Bilinear F at metric coordinates; NaN where the reference is invalid.

    F carries NaN outside optical coverage. Interpolating that directly would
    poison whole neighbourhoods, and replacing NaN with a large positive number
    would manufacture a sign change between water and nodata. So the validity
    mask is interpolated alongside and any sample that is not fully supported
    is returned as NaN -- no optical evidence, rather than a fake crossing."""
    rr = (G["y1"] - Y) / CELL - 0.5
    cc = (X - G["x0"]) / CELL - 0.5
    co = np.vstack([rr.ravel(), cc.ravel()])
    Ff = np.where(valid, np.nan_to_num(F, nan=0.0), 0.0).astype(np.float64)
    v = map_coordinates(Ff, co, order=1, mode="constant", cval=0.0)
    s = map_coordinates(valid.astype(np.float64), co, order=1,
                        mode="constant", cval=0.0)
    v[s < 0.999] = np.nan
    return v.reshape(X.shape)


# ------------------------------------------------------------------ matchers --
def match_nearest(A, F_ref, valid_ref):
    """Nearest-vertex distance from every A vertex to the reference contour.

    Returns the UNSIGNED distance and the reference field value at A, so that
    any threshold can be applied afterwards without re-running the search."""
    return A[:, :2], F_ref[A[:, 2].astype(int), A[:, 3].astype(int)]


def match_normal(A, F_ref, valid_ref, G):
    """Signed cross-shore distance by NORMAL INTERSECTION, sub-pixel.

    For each comparison vertex the reference field is sampled along the local
    cross-shore normal and the zero crossing NEAREST to the vertex is taken,
    with the crossing located by linear interpolation between the two
    bracketing samples. A vertex whose normal does not cross the reference
    contour within NORMAL_MAX_M is UNPAIRED -- an unmatched vertex is missing
    evidence, not a measured displacement, and substituting the nearest point
    on some unrelated branch of the shoreline is exactly the failure mode this
    gate exists to catch.

    Returns signed distance (NaN where unpaired) in the frozen convention."""
    t = np.arange(-NORMAL_MAX_M, NORMAL_MAX_M + NORMAL_STEP_M, NORMAL_STEP_M)
    n = len(A)
    d = np.full(n, np.nan)
    CH = 20000
    for a in range(0, n, CH):
        b = min(a + CH, n)
        X = A[a:b, 0, None] + t[None, :] * A[a:b, 5, None]
        Y = A[a:b, 1, None] + t[None, :] * A[a:b, 6, None]
        V = sample_field(F_ref, valid_ref, X, Y, G)
        s0, s1 = V[:, :-1], V[:, 1:]
        cross = np.isfinite(s0) & np.isfinite(s1) & ((s0 <= 0) != (s1 <= 0))
        # sub-pixel position of each crossing, then keep the one nearest t = 0
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = s0 / (s0 - s1)
        tc = np.where(cross, t[:-1][None, :] + frac * NORMAL_STEP_M, np.inf)
        j = np.argmin(np.abs(tc), axis=1)
        best = tc[np.arange(b - a), j]
        d[a:b] = np.where(np.isfinite(best), np.abs(best), np.nan)
    return d


def apply_sign(dist, f_ref_at_A, direction):
    """Signed distance in the frozen convention: + = SAR extent LARGER.

    The inside-test is OPPOSITE in the two directions, and getting this wrong
    is what makes a correct matcher look antisymmetric:
      s1_to_s2 : SAR vertex inside optical water  -> SAR smaller -> NEGATIVE
      s2_to_s1 : optical vertex inside SAR water  -> SAR larger  -> POSITIVE"""
    inside = f_ref_at_A < 0
    if direction == "s1_to_s2":
        return np.where(inside, -1.0, +1.0) * dist
    if direction == "s2_to_s1":
        return np.where(inside, +1.0, -1.0) * dist
    raise ValueError(direction)


def stats(d, wl):
    """Robust summary plus the LENGTH actually represented by the sample."""
    k = np.isfinite(d)
    if k.sum() < 50:
        return dict(n=int(k.sum()), length_km=float(wl[k].sum() / 1000),
                    median=np.nan, nmad=np.nan, p90=np.nan, p95=np.nan)
    v = d[k]
    m = float(np.median(v))
    return dict(n=int(k.sum()), length_km=float(wl[k].sum() / 1000),
                median=m, nmad=float(1.4826 * np.median(np.abs(v - m))),
                p90=float(np.percentile(np.abs(v), 90)),
                p95=float(np.percentile(np.abs(v), 95)))


# ---------------------------------------------------------------- synthetic --
def signed_distance(W):
    """True signed distance to the boundary of mask W, metres, <0 inside.

    Because D is a genuine distance field, the zero level of D - delta is the
    real shoreline displaced by exactly delta along its own normal. That makes
    delta an ANALYTIC ground truth on fully realistic geometry -- islands,
    narrow channels, concave bays -- which a circle or a half-plane cannot
    exercise.

    THE HALF-CELL IS NOT OPTIONAL. The EDT measures to the nearest opposite
    pixel CENTRE, so a water cell touching land reports 20 m when its centre is
    only 10 m inside the interface. Writing D = dout - din therefore produces a
    field that jumps -20 -> +20 across one cell: gradient 2.0 m/m, not 1.0, and
    every injected offset comes back at half its true value. That version of
    this test reported recovery of 2.30 / 4.62 / 10.00 / 31.85 m for injections
    of 5 / 10 / 20 / 40 m and looked like a matcher failure; it was the ground
    truth that was wrong, the same error as the three discarded mask-built
    constructions in Gate 7C0. Subtracting the half cell on each side restores
    gradient 1.0 and exact recovery."""
    din = ndimage.distance_transform_edt(W) * CELL
    dout = ndimage.distance_transform_edt(~W) * CELL
    D = np.where(W, -(din - 0.5 * CELL), dout - 0.5 * CELL)
    return D.astype(np.float32)


def check_distance_field(D, W):
    """Refuse to run the synthetic tests on a field that is not a distance field.

    The gradient of a true signed distance field is 1.0 everywhere away from
    the medial axis. Asserting it here is what would have caught the half-cell
    error immediately instead of letting it masquerade as a matcher failure."""
    gy, gx = np.gradient(D, CELL)
    near = ndimage.binary_dilation(W, np.ones((3, 3), bool), 2) & ~ \
        ndimage.binary_erosion(W, np.ones((3, 3), bool), 2)
    g = float(np.median(np.hypot(gx[near], gy[near])))
    print(f"    |grad D| near the boundary = {g:.3f} m/m (must be 1.000)")
    if abs(g - 1.0) > 0.05:
        raise SystemExit(f"synthetic ground truth is not a distance field "
                         f"(|grad D| = {g:.3f}); injected offsets would be "
                         f"recovered at {1/g:.2f}x and misread as matcher bias")
    return g


def synthetic_recovery(D, dom, G):
    """Offset injection and stage-sign recovery through the PRODUCTION matcher."""
    from hist25b_gate7c_v2_same_date_controls import boundary_band
    rows = []
    band_ref, _ = boundary_band(D, dom)
    for delta in INJECT_M:
        Fc = D - delta
        band_c, _ = boundary_band(Fc, dom)
        A = contour_points(Fc, band_c, G)
        if len(A) < 200:
            continue
        d = apply_sign(match_normal(A, D, dom, G),
                       D[A[:, 2].astype(int), A[:, 3].astype(int)], "s1_to_s2")
        st = stats(d, A[:, 4])
        rows.append(dict(test="offset_injection", truth_m=delta,
                         recovered_median=st["median"], nmad=st["nmad"],
                         p90=st["p90"], n=st["n"],
                         pair_frac_length=float(
                             A[np.isfinite(d), 4].sum() / A[:, 4].sum()),
                         error_m=st["median"] - delta))
        print(f"    offset {delta:+6.1f} m -> recovered "
              f"{st['median']:+7.2f} m  err {st['median']-delta:+6.2f}  "
              f"NMAD {st['nmad']:5.2f}  f_L {A[np.isfinite(d),4].sum()/A[:,4].sum():.3f}")
    for dH in STAGE_DH_M:
        L = dH / STAGE_SLOPE            # optical displaced outward by L
        F_opt = D - L
        band_c, _ = boundary_band(D, dom)
        A = contour_points(D, band_c, G)
        band_r, _ = boundary_band(F_opt, dom)
        d_raw = apply_sign(match_normal(A, F_opt, dom, G),
                           sample_field(F_opt, dom, A[:, 0], A[:, 1], G),
                           "s1_to_s2")
        d_stage = -dH / STAGE_SLOPE
        res = d_raw - d_stage
        st_r, st_s = stats(d_raw, A[:, 4]), stats(res, A[:, 4])
        rows.append(dict(test="stage_sign", truth_m=-L,
                         recovered_median=st_r["median"], nmad=st_r["nmad"],
                         p90=st_r["p90"], n=st_r["n"],
                         pair_frac_length=float(
                             A[np.isfinite(d_raw), 4].sum() / A[:, 4].sum()),
                         error_m=st_r["median"] + L, delta_H=dH,
                         d_stage=d_stage, d_residual_median=st_s["median"]))
        print(f"    dH {dH:+5.2f} m (L {L:+7.1f} m) -> d_raw "
              f"{st_r['median']:+7.1f}  d_stage {d_stage:+7.1f}  "
              f"residual {st_s['median']:+6.2f} m")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main --
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("GATE 7C1 — correspondence / pairing validation")
    print("=" * 78)
    print(f"  git {commit}; H1/H2/H3 NOT run; hist24 NOT run")
    print("  the same-date median is only as good as what the matcher paired")

    M, meta = load_frozen_manifest()
    G = build_grid()
    tok = json.loads(urllib.request.urlopen(SAS, timeout=90).read())["token"]

    length_rows, sens_rows, maps = [], [], {}
    for d in CONTROL_DATES:
        print(f"\n  {d}")
        FL = build_fields(d, G, M, tok)
        if FL is None:
            continue
        F_sar, F_opt = FL["F_sar"], FL["F_opt"]
        obs, cov_s2 = FL["obs"], FL["cov_s2"]
        band_sar, band_opt = FL["band_sar"], FL["band_opt"]

        # ---- 0. scene radiometric separability -----------------------------
        # How far apart the anchor classes sit in the discriminant is the
        # cheapest available predictor of whether the mask will come out
        # connected. ERA5 wind does NOT explain the difference between these
        # two dates: 2019-03-18 was the CALMER scene (2.2 m/s at the ~04 UTC
        # descending overpass against 3.8 m/s on 2023-02-20) and yet it is the
        # fragmented one, so the wind-roughening explanation is falsified here.
        sep = float(FL["anchor_land"] - FL["anchor_water"])
        lab, ncomp = ndimage.label(FL["W_sar"], structure=np.ones((3, 3), int))
        szs = ndimage.sum(np.ones_like(lab), lab, range(1, ncomp + 1))
        frac_big = float(szs.max() / szs.sum()) if ncomp else np.nan
        print(f"    LDA class separation {sep:.2f}; SAR mask {ncomp} components, "
              f"largest holds {100*frac_big:.0f}% of the water area")
        length_rows.append(dict(control_date=d, matcher="scene_quality",
                                lda_separation=sep, n_components=int(ncomp),
                                largest_component_frac=frac_big))

        # ---- 1. eligible shoreline length, and what cleaning removed --------
        raw_sar = contour_points(F_sar, obs, G)
        A = contour_points(F_sar, band_sar, G)          # eligible SAR vertices
        B = contour_points(F_opt, band_opt, G)          # eligible optical
        L_raw = raw_sar[:, 4].sum() / 1000
        L_elig = A[:, 4].sum() / 1000
        print(f"    raw SAR zero-contour {L_raw:8,.0f} km; after cleaning "
              f"{L_elig:8,.0f} km  ({100*(1-L_elig/L_raw):.1f}% removed as "
              f"speckle, not shoreline)")
        print(f"    eligible optical contour {B[:,4].sum()/1000:8,.0f} km")

        # ---- 3+5. both directions, both matchers ---------------------------
        f_at_A = F_opt[A[:, 2].astype(int), A[:, 3].astype(int)]
        f_at_B = F_sar[B[:, 2].astype(int), B[:, 3].astype(int)]
        treeB = cKDTree(B[:, :2])
        treeA = cKDTree(A[:, :2])
        nnA, _ = treeB.query(A[:, :2], k=1)
        nnB, _ = treeA.query(B[:, :2], k=1)
        norA = match_normal(A, F_opt, cov_s2, G)
        norB = match_normal(B, F_sar, obs, G)

        for name, vert, dist, fref, direction in (
                ("s1_to_s2_nearest", A, nnA, f_at_A, "s1_to_s2"),
                ("s2_to_s1_nearest", B, nnB, f_at_B, "s2_to_s1"),
                ("s1_to_s2_normal", A, norA, f_at_A, "s1_to_s2"),
                ("s2_to_s1_normal", B, norB, f_at_B, "s2_to_s1")):
            sd = apply_sign(dist, fref, direction)
            use = sd.copy()
            if name.endswith("nearest"):
                use[dist > NORMAL_MAX_M] = np.nan   # same budget as the normal
            st = stats(use, vert[:, 4])
            fl = vert[np.isfinite(use), 4].sum() / vert[:, 4].sum()
            length_rows.append(dict(control_date=d, matcher=name,
                                    f_L=float(fl), f_vertices=float(
                                        np.isfinite(use).mean()), **st))
            print(f"    {name:20s} f_L {100*fl:5.1f}%  median "
                  f"{st['median']:+7.1f}  NMAD {st['nmad']:6.1f}  "
                  f"P90 {st['p90']:6.1f}  P95 {st['p95']:6.1f}")

        # ---- 4. threshold sensitivity --------------------------------------
        sdA_n = apply_sign(nnA, f_at_A, "s1_to_s2")
        sdA_r = apply_sign(norA, f_at_A, "s1_to_s2")
        for thr in THRESHOLDS_M:
            for nm, dd, sd in (("nearest", nnA, sdA_n), ("normal", norA, sdA_r)):
                v = np.where(dd <= thr, sd, np.nan)
                st = stats(v, A[:, 4])
                sens_rows.append(dict(control_date=d, matcher=nm,
                                      threshold_m=thr,
                                      f_L=float(A[np.isfinite(v), 4].sum()
                                                / A[:, 4].sum()), **st))

        # ---- 2. reason per unpaired vertex ---------------------------------
        gap_m = ndimage.distance_transform_edt(cov_s2) * CELL
        gi = np.clip(((G["y1"] - A[:, 1]) / CELL).astype(int), 0, G["ny"] - 1)
        gj = np.clip(((A[:, 0] - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
        near_gap = gap_m[gi, gj] <= GAP_NEAR_M
        reason = np.where(np.isfinite(sdA_r), "paired",
                          np.where(near_gap, "unpaired_s2_gap",
                                   np.where(nnA <= NORMAL_MAX_M,
                                            "unpaired_no_normal_crossing",
                                            "unpaired_max_distance")))
        wl = A[:, 4]
        print("    unpaired reasons, by LENGTH:")
        for rn in ("paired", "unpaired_s2_gap", "unpaired_no_normal_crossing",
                   "unpaired_max_distance"):
            k = reason == rn
            print(f"      {rn:32s} {100*wl[k].sum()/wl.sum():5.1f}%  "
                  f"({wl[k].sum()/1000:7,.0f} km)")
            length_rows.append(dict(control_date=d, matcher=f"reason:{rn}",
                                    f_L=float(wl[k].sum() / wl.sum()),
                                    n=int(k.sum()),
                                    length_km=float(wl[k].sum() / 1000)))
        maps[d] = dict(X=A[:, 0], Y=A[:, 1], reason=reason, d=sdA_r,
                       cov=cov_s2, W=FL["W_opt"])

    LR = pd.DataFrame(length_rows)
    SR = pd.DataFrame(sens_rows)
    LR.to_csv(CFG.TABLES / "gate7c1_pairing_length.csv", index=False)
    SR.to_csv(CFG.TABLES / "gate7c1_threshold_sensitivity.csv", index=False)

    # ---- end-to-end synthetic tests on the real 2019 geometry --------------
    print("\n  END-TO-END on the REAL cleaned 2019 shoreline "
          "(islands, channels, bays):")
    FL = build_fields(CONTROL_DATES[0], G, M, tok)
    Wc = clean_water(FL["W_opt"])
    D = signed_distance(Wc)
    check_distance_field(D, Wc)
    dom = G["inside"]
    SYN = synthetic_recovery(D, dom, G)
    SYN.to_csv(CFG.TABLES / "gate7c1_synthetic_recovery.csv", index=False)

    make_figures(maps, SR, SYN)

    print("\n" + "=" * 78)
    print("GATE 7C1 SUMMARY")
    print("=" * 78)
    print(LR[LR.matcher.str.contains("_to_")].to_string(index=False))
    print("\nSTOP. H1/H2/H3 not started; hist24 not started.")


def make_figures(maps, SR, SYN):
    if maps:
        fig, axes = plt.subplots(1, len(maps), figsize=(8 * len(maps), 8))
        axes = np.atleast_1d(axes)
        cols = {"paired": GREEN, "unpaired_s2_gap": AMBER,
                "unpaired_no_normal_crossing": RED,
                "unpaired_max_distance": PURPLE}
        for ax, (d, m) in zip(axes, maps.items()):
            for rn, c in cols.items():
                k = m["reason"] == rn
                if k.any():
                    ax.scatter(m["X"][k] / 1000, m["Y"][k] / 1000, s=0.4,
                               c=c, label=f"{rn} ({100*k.mean():.0f}%)",
                               linewidths=0)
            ax.set_title(f"{d} · pairing outcome along the SAR shoreline",
                         color=INK)
            ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
            ax.set_aspect("equal")
            ax.legend(markerscale=14, fontsize=8, loc="upper left")
        fig.suptitle("Gate 7C1 · where the matcher succeeds and where it does not",
                     color=INK)
        fig.tight_layout()
        p = FIGDIR / "gate7c1_pairing_map.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"-> {p}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for nm, ls in (("nearest", "--"), ("normal", "-")):
        for d, c in zip(SR.control_date.unique(), (BLUE, RED)):
            s = SR[(SR.matcher == nm) & (SR.control_date == d)]
            axes[0].plot(s.threshold_m, 100 * s.f_L, ls, color=c, marker="o",
                         label=f"{d} {nm}")
            axes[1].plot(s.threshold_m, s["median"], ls, color=c, marker="o",
                         label=f"{d} {nm}")
            axes[2].plot(s.threshold_m, s.p95, ls, color=c, marker="o",
                         label=f"{d} {nm}")
    for ax, t in zip(axes, ("paired length fraction (%)",
                            "median d_raw (m)", "P95 |d_raw| (m)")):
        ax.set_xlabel("max match distance (m)"); ax.set_title(t, color=INK)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[1].axhline(0, color=INK, lw=0.8)
    fig.suptitle("Gate 7C1 · a median that drifts with the threshold is "
                 "selection bias", color=INK)
    fig.tight_layout()
    p = FIGDIR / "gate7c1_sensitivity.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")

    inj = SYN[SYN.test == "offset_injection"]
    if not inj.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        lim = max(abs(inj.truth_m).max() * 1.3, 10)
        ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1, ls="--",
                label="1:1")
        ax.errorbar(inj.truth_m, inj.recovered_median, yerr=inj.nmad,
                    fmt="o", color=BLUE, capsize=3,
                    label="production matcher on real geometry")
        ax.set_xlabel("injected normal offset (m)")
        ax.set_ylabel("recovered median d_raw (m)")
        ax.set_title("Gate 7C1 · offset injection on the real 2019 shoreline",
                     color=INK)
        ax.grid(alpha=0.3); ax.legend(); ax.set_aspect("equal")
        fig.tight_layout()
        p = FIGDIR / "gate7c1_offset_injection.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
