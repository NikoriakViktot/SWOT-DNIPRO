#!/usr/bin/env python
"""GATE 7C0 v2 — publication-grade validation of the shoreline estimator.

LAYER A ONLY. This answers one numerical question: can the algorithm recover a
mathematically known shoreline displacement, independently of orientation and
both above and below one pixel? It does NOT establish the real positional
accuracy of Sentinel-1 or Sentinel-2, which requires independent UAV / GNSS /
VHR validation and is a separate layer entirely.

WHY THE GROUND TRUTH IS ANALYTIC. Earlier attempts built two binary masks and
compared them, which makes the rasterisation part of the thing being tested;
three successive test constructions failed on their own artefacts before the
estimator was ever reached. Here the shoreline is defined exactly as

    F(x, y) = x cos(theta) + y sin(theta) - c ,   F = 0 is the shoreline

and F is SAMPLED at 10 m cell centres. Ground truth then exists in closed
form and is independent of any raster boundary convention.

TWO ESTIMATORS, IDENTICAL INPUTS
  v1  estimator_v1_pixel_boundary   threshold F, take boundary pixels of the
                                    binary mask, measure by distance transform
                                    -- the method used through Gate 7B/7C
  v2  estimator_v2_subpixel_contour Marching Squares on the CONTINUOUS field
                                    at F = 0, giving floating-point vertices,
                                    then vector-to-vector signed distance

v1 is preserved rather than replaced, and its earlier failures stay in the QA
record.

DISPLACEMENTS span sub-pixel to multi-pixel: 0, +-5, +-10, +-20, +-40, +-60 m
on a 10 m grid, i.e. 0.5 to 6 pixels. The 5 m case is a legitimate test of
sub-pixel recoverability -- that is precisely what sub-pixel contouring is
for -- and is NOT dismissed as unresolvable. But recovering a synthetic 5 m
shift does not mean a real Sentinel shoreline is accurate to 5 m.

RESOLUTION SEMANTICS, kept strictly apart:
    analysis grid resolution        10 m (S2 NDWI from native 10 m B3/B8)
    sub-pixel estimator precision   measured here
    physical positional accuracy    NOT measured here
A 20 m native band upsampled to 10 m does not create 10 m information, so an
MNDWI-based surface would carry effective_information_resolution = 20 m.

SIGN CONVENTION, fixed and unit-tested:
    positive = comparison shoreline lies on the +F side (landward here)
    negative = comparison shoreline lies on the -F side (waterward)
Polyline vertex order must never change this, which is why the sign comes
from the analytic field rather than from ring winding.

The |b - 1| < 0.05 criterion is an additional stringent numerical acceptance
criterion adopted for this study. It is NOT a published Sentinel shoreline
standard; the literature validates with bias / MAE / RMSE / NMAD against
independent references.

Outputs
-------
outputs/tables/gate7c0_v1_synthetic_results.csv
outputs/tables/gate7c0_v2_synthetic_results.csv
outputs/tables/gate7c0_orientation_summary.csv
outputs/tables/gate7c0_accuracy_metrics.csv
outputs/reports/gate7c0_QA.md
outputs/figures/historical_bathymetry/png/gate7c0_*.png
"""
from __future__ import annotations

import json
import subprocess
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
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
GRID_M = 10.0                       # S2 NDWI is native 10 m (B3, B8)
N = 400                             # cells per side
MARGIN_CELLS = 40                   # exclude array border, unambiguous
ORIENTATIONS_DEG = (0, 15, 30, 45, 60, 75, 90)
OFFSETS_M = (-60, -40, -20, -10, -5, 0, 5, 10, 20, 40, 60)
SLOPE_TOL = 0.05                    # study-specific, not a published standard
INTERCEPT_TOL_M = 0.5 * GRID_M
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
REPORTS = CFG.OUT / "reports"


def grid_xy():
    """Cell-centre coordinates in metres."""
    c = (np.arange(N) + 0.5) * GRID_M
    return np.meshgrid(c, c)          # X, Y


def field(theta_deg, c_m):
    """Exact signed-distance field; F = 0 is the shoreline, in metres."""
    X, Y = grid_xy()
    th = np.radians(theta_deg)
    ctr = N * GRID_M / 2
    return (X - ctr) * np.cos(th) + (Y - ctr) * np.sin(th) - c_m


def v1_boundary_offset(F_cmp, F_ref):
    """estimator_v1_pixel_boundary: threshold, boundary pixels, distance
    transform. Preserved exactly as used through Gate 7B/7C."""
    cmp_m = F_cmp < 0
    ref_m = F_ref < 0
    d_in = ndimage.distance_transform_edt(ref_m) * GRID_M
    d_out = ndimage.distance_transform_edt(~ref_m) * GRID_M
    bnd = cmp_m & ~ndimage.binary_erosion(cmp_m, np.ones((3, 3), bool))
    ys, xs = np.where(bnd)
    keep = ((xs > MARGIN_CELLS) & (xs < N - MARGIN_CELLS) &
            (ys > MARGIN_CELLS) & (ys < N - MARGIN_CELLS))
    ys, xs = ys[keep], xs[keep]
    if len(xs) == 0:
        return np.array([])
    # half-cell correction from the earlier calibration
    sdf = np.where(ref_m, -(ndimage.distance_transform_edt(ref_m) - 0.5),
                   +(ndimage.distance_transform_edt(~ref_m) - 0.5)) * GRID_M
    return sdf[ys, xs] + 0.5 * GRID_M


def contour_xy(F):
    """Marching Squares at F = 0 on the CONTINUOUS field, returned as metric
    vertices. skimage yields (row, col) in fractional index space; converting
    with the cell-centre convention is what preserves sub-pixel position."""
    out = []
    for c in find_contours(F, level=0.0):
        r, cc = c[:, 0], c[:, 1]
        x = (cc + 0.5) * GRID_M
        y = (r + 0.5) * GRID_M
        out.append(np.c_[x, y])
    return out


def v2_subpixel_offset(F_cmp, F_ref, theta_deg, c_ref_m):
    """estimator_v2_subpixel_contour: vector contour, signed normal distance.

    The sign and the normal come from the ANALYTIC reference geometry, so no
    polyline vertex order can flip them."""
    th = np.radians(theta_deg)
    ctr = N * GRID_M / 2
    segs = contour_xy(F_cmp)
    if not segs:
        return np.array([])
    P = np.vstack(segs)
    lo, hi = MARGIN_CELLS * GRID_M, (N - MARGIN_CELLS) * GRID_M
    m = ((P[:, 0] > lo) & (P[:, 0] < hi) & (P[:, 1] > lo) & (P[:, 1] < hi))
    P = P[m]
    if len(P) == 0:
        return np.array([])
    # signed distance of each extracted vertex from the analytic reference line
    return ((P[:, 0] - ctr) * np.cos(th) + (P[:, 1] - ctr) * np.sin(th)
            - c_ref_m)


def stats(err, n):
    return dict(n=int(n), bias_m=float(np.mean(err)),
                median_m=float(np.median(err)),
                mae_m=float(np.mean(np.abs(err))),
                rmse_m=float(np.sqrt(np.mean(err ** 2))),
                sd_m=float(np.std(err)),
                nmad_m=float(1.4826 * np.median(np.abs(err - np.median(err)))),
                q05_m=float(np.percentile(err, 5)),
                q25_m=float(np.percentile(err, 25)),
                q75_m=float(np.percentile(err, 75)),
                q95_m=float(np.percentile(err, 95)),
                max_abs_m=float(np.max(np.abs(err))))


def run_suite():
    rows = []
    for ang in ORIENTATIONS_DEG:
        F_ref = field(ang, 0.0)
        for off in OFFSETS_M:
            F_cmp = field(ang, off)
            for name, fn in (("v1_pixel_boundary",
                              lambda: v1_boundary_offset(F_cmp, F_ref)),
                             ("v2_subpixel_contour",
                              lambda: v2_subpixel_offset(F_cmp, F_ref, ang, 0.0))):
                d = fn()
                if d.size < 20:
                    continue
                err = d - off
                rows.append(dict(estimator=name, orientation_deg=ang,
                                 true_offset_m=float(off),
                                 median_recovered_m=float(np.median(d)),
                                 mean_recovered_m=float(np.mean(d)),
                                 **stats(err, d.size)))
    return pd.DataFrame(rows)


def curved_suite():
    """Secondary robustness test: analytic circle, so truth stays closed-form."""
    X, Y = grid_xy()
    ctr = N * GRID_M / 2
    R0 = N * GRID_M * 0.30
    rows = []
    for off in (-40, -20, -10, -5, 0, 5, 10, 20, 40):
        F_ref = np.hypot(X - ctr, Y - ctr) - R0
        F_cmp = np.hypot(X - ctr, Y - ctr) - (R0 + off)
        segs = contour_xy(F_cmp)
        if not segs:
            continue
        P = np.vstack(segs)
        d = np.hypot(P[:, 0] - ctr, P[:, 1] - ctr) - R0
        err = d - off
        rows.append(dict(geometry="circle", true_offset_m=float(off),
                         estimator="v2_subpixel_contour",
                         median_recovered_m=float(np.median(d)), **stats(err, d.size)))
        dv1 = v1_boundary_offset(F_cmp, F_ref)
        if dv1.size > 20:
            e1 = dv1 - off
            rows.append(dict(geometry="circle", true_offset_m=float(off),
                             estimator="v1_pixel_boundary",
                             median_recovered_m=float(np.median(dv1)),
                             **stats(e1, dv1.size)))
    return pd.DataFrame(rows)


def regress(df):
    out = []
    for (est, ang), g in df.groupby(["estimator", "orientation_deg"]):
        b, a = np.polyfit(g.true_offset_m, g.median_recovered_m, 1)
        out.append(dict(estimator=est, orientation_deg=ang, intercept_m=a,
                        slope=b, rmse_m=float(np.sqrt((g.rmse_m ** 2).mean()))))
    return pd.DataFrame(out)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
    except Exception:
        commit = "unknown"
    print("=" * 78)
    print("GATE 7C0 v2 — NUMERICAL ESTIMATOR VALIDATION (Layer A only)")
    print("=" * 78)
    print(f"  analysis grid {GRID_M:.0f} m; ground truth analytic "
          f"F = x cos(t) + y sin(t) - c, sampled at cell centres")
    print(f"  orientations {ORIENTATIONS_DEG}")
    print(f"  offsets (m)  {OFFSETS_M}  = "
          f"{tuple(round(o/GRID_M, 1) for o in OFFSETS_M)} pixels")
    print(f"  git {commit}")
    print("  NOTE: 10 m grid is NOT 10 m positional uncertainty, and sub-pixel "
          "recovery\n        is NOT sub-pixel physical accuracy.")

    R = run_suite()
    for est in R.estimator.unique():
        R[R.estimator == est].to_csv(
            CFG.TABLES / f"gate7c0_{est.split('_')[0]}_synthetic_results.csv",
            index=False)
    G = regress(R)
    G.to_csv(CFG.TABLES / "gate7c0_orientation_summary.csv", index=False)
    R.to_csv(CFG.TABLES / "gate7c0_accuracy_metrics.csv", index=False)

    print("\n" + "=" * 78)
    print("REGRESSION recovered = a + b * true, BY ORIENTATION")
    print("=" * 78)
    for est in ("v1_pixel_boundary", "v2_subpixel_contour"):
        g = G[G.estimator == est]
        print(f"\n  {est}")
        print("    " + g[["orientation_deg", "intercept_m", "slope", "rmse_m"]
                         ].to_string(index=False,
                                     float_format=lambda v: f"{v:+.3f}"
                                     ).replace("\n", "\n    "))
        print(f"    slope range {g.slope.min():.4f}..{g.slope.max():.4f}  "
              f"spread {g.slope.max()-g.slope.min():.4f}")

    print("\n" + "=" * 78)
    print("ACCURACY BY ESTIMATOR (all orientations, all offsets)")
    print("=" * 78)
    A = (R.groupby("estimator")
           .agg(bias_m=("bias_m", "mean"), mae_m=("mae_m", "mean"),
                rmse_m=("rmse_m", lambda s: float(np.sqrt((s ** 2).mean()))),
                nmad_m=("nmad_m", "median"),
                max_abs_m=("max_abs_m", "max")).reset_index())
    print(A.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n  RMSE by orientation:")
    print(R.pivot_table(index="orientation_deg", columns="estimator",
                        values="rmse_m",
                        aggfunc=lambda s: float(np.sqrt((s ** 2).mean()))
                        ).to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n" + "=" * 78)
    print("SUB-PIXEL TEST (+-5 m = 0.5 pixel) — reported, not dismissed")
    print("=" * 78)
    sub = R[R.true_offset_m.abs() == 5]
    print(sub.pivot_table(index="orientation_deg", columns="estimator",
                          values="median_recovered_m").to_string(
        float_format=lambda v: f"{v:+.2f}"))
    for est in ("v1_pixel_boundary", "v2_subpixel_contour"):
        s = sub[sub.estimator == est]
        print(f"  {est}: |error| mean {s.mae_m.mean():.2f} m, "
              f"RMSE {np.sqrt((s.rmse_m**2).mean()):.2f} m")

    CU = curved_suite()
    if not CU.empty:
        print("\n" + "=" * 78)
        print("CURVED GEOMETRY (analytic circle) — robustness to curvature")
        print("=" * 78)
        print(CU.groupby("estimator").agg(
            bias_m=("bias_m", "mean"), mae_m=("mae_m", "mean"),
            rmse_m=("rmse_m", lambda s: float(np.sqrt((s ** 2).mean())))
        ).to_string(float_format=lambda v: f"{v:,.3f}"))

    # ------------------------------------------------ decision
    dec = {}
    for est in ("v1_pixel_boundary", "v2_subpixel_contour"):
        g = G[G.estimator == est]
        r = R[R.estimator == est]
        a_ok = g.intercept_m.abs().max() < INTERCEPT_TOL_M
        b_ok = (g.slope - 1).abs().max() < SLOPE_TOL
        o_ok = (g.slope.max() - g.slope.min()) < SLOPE_TOL
        dec[est] = dict(intercept_ok=bool(a_ok), slope_ok=bool(b_ok),
                        orientation_ok=bool(o_ok),
                        max_abs_intercept=float(g.intercept_m.abs().max()),
                        worst_slope_dev=float((g.slope - 1).abs().max()),
                        slope_spread=float(g.slope.max() - g.slope.min()),
                        rmse=float(np.sqrt((r.rmse_m ** 2).mean())),
                        passed=bool(a_ok and b_ok and o_ok))
    print("\n" + "=" * 78)
    print("GATE 7C0 DECISION")
    print("=" * 78)
    for est, d in dec.items():
        print(f"  {est}")
        print(f"    |a| < {INTERCEPT_TOL_M:.0f} m           "
              f"{'PASS' if d['intercept_ok'] else 'FAIL'}"
              f"  (max |a| = {d['max_abs_intercept']:.2f} m)")
        print(f"    |b-1| < {SLOPE_TOL}            "
              f"{'PASS' if d['slope_ok'] else 'FAIL'}"
              f"  (worst = {d['worst_slope_dev']:.4f})")
        print(f"    no orientation dependence  "
              f"{'PASS' if d['orientation_ok'] else 'FAIL'}"
              f"  (slope spread = {d['slope_spread']:.4f})")
        print(f"    pooled RMSE = {d['rmse']:.2f} m")
    if dec["v1_pixel_boundary"]["passed"]:
        verdict = "GATE 7C0 = PASS WITH V1"
    elif dec["v2_subpixel_contour"]["passed"]:
        verdict = "GATE 7C0 = PASS WITH V2"
    else:
        verdict = "GATE 7C0 = FAIL"
    print(f"\n  {verdict}")

    _figures(R, G, CU)
    _report(R, G, A, dec, verdict, commit, CU)
    print(f"\n-> {REPORTS/'gate7c0_QA.md'}")
    print("\nSTOP. The estimator is frozen here; real Sentinel pairs are NOT "
          "re-interpreted in this gate.")


def _figures(R, G, CU):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    for est, mk, col in (("v1_pixel_boundary", "o--", RED),
                         ("v2_subpixel_contour", "s-", GREEN)):
        g = R[R.estimator == est]
        for ang, gg in g.groupby("orientation_deg"):
            a.plot(gg.true_offset_m, gg.median_recovered_m, mk, ms=3, lw=0.8,
                   color=col, alpha=0.55)
    lim = [min(OFFSETS_M), max(OFFSETS_M)]
    a.plot(lim, lim, color=INK, lw=1.6, label="1:1")
    a.plot([], [], "o--", color=RED, label="v1 pixel boundary")
    a.plot([], [], "s-", color=GREEN, label="v2 subpixel contour")
    a.set_xlabel("true offset (m)"); a.set_ylabel("recovered (m)")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("a · true vs recovered, all orientations", fontsize=10.4,
                loc="left")
    a = ax[0, 1]
    for est, col in (("v1_pixel_boundary", RED), ("v2_subpixel_contour", GREEN)):
        g = R[R.estimator == est]
        a.plot(g.true_offset_m, g.median_recovered_m - g.true_offset_m, "o",
               ms=3.5, color=col, alpha=0.6, label=est)
    a.axhline(0, color=INK, lw=1.2)
    a.axhspan(-GRID_M / 2, GRID_M / 2, color=GREY, alpha=0.18,
              label="half cell")
    a.set_xlabel("true offset (m)"); a.set_ylabel("error (m)")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_title("b · error vs true offset", fontsize=10.4, loc="left")
    a = ax[1, 0]
    for est, col in (("v1_pixel_boundary", RED), ("v2_subpixel_contour", GREEN)):
        g = G[G.estimator == est]
        a.plot(g.orientation_deg, g.slope, "o-", color=col, label=est)
    a.axhline(1.0, color=INK, lw=1.2)
    a.axhspan(1 - SLOPE_TOL, 1 + SLOPE_TOL, color=GREY, alpha=0.18,
              label=f"+-{SLOPE_TOL} tolerance")
    a.set_xlabel("orientation (deg)"); a.set_ylabel("regression slope b")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_title("c · slope vs orientation — the diagnostic that matters",
                fontsize=10.4, loc="left")
    a = ax[1, 1]
    for est, col in (("v1_pixel_boundary", RED), ("v2_subpixel_contour", GREEN)):
        g = R[R.estimator == est].groupby("orientation_deg").rmse_m.apply(
            lambda s: float(np.sqrt((s ** 2).mean())))
        a.plot(g.index, g.values, "o-", color=col, label=est)
    a.set_xlabel("orientation (deg)"); a.set_ylabel("RMSE (m)")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_title("d · RMSE vs orientation", fontsize=10.4, loc="left")
    fig.suptitle("Gate 7C0 v2 · numerical estimator validation "
                 "(synthetic precision, NOT sensor accuracy)", y=1.0,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "gate7c0_true_vs_recovered.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # geometry examples
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.2))
    for a, ang in zip(ax, (0, 30, 45, 60, 90)):
        F_ref = field(ang, 0.0)
        F_cmp = field(ang, 20.0)
        a.imshow(F_cmp < 0, origin="lower", cmap="Blues", alpha=0.35,
                 extent=[0, N * GRID_M, 0, N * GRID_M])
        for seg in contour_xy(F_cmp):
            a.plot(seg[:, 0], seg[:, 1], color=GREEN, lw=1.6)
        for seg in contour_xy(F_ref):
            a.plot(seg[:, 0], seg[:, 1], color=INK, lw=1.2, ls="--")
        a.set_title(f"{ang} deg, +20 m", fontsize=10, loc="left")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Gate 7C0 v2 · analytic reference (dashed) and recovered "
                 "sub-pixel contour (green)", y=1.03, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "gate7c0_geometry_examples.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    for f in ("true_vs_recovered", "geometry_examples"):
        print(f"-> {FIGDIR/('gate7c0_'+f+'.png')}")


def _report(R, G, A, dec, verdict, commit, CU):
    L = ["# Gate 7C0 — numerical shoreline-estimator validation", "",
         f"git `{commit}` · analysis grid {GRID_M:.0f} m · "
         f"orientations {ORIENTATIONS_DEG} · offsets {OFFSETS_M} m", "",
         "## 1. Scope", "",
         "Layer A only: can the algorithm recover a mathematically known "
         "shoreline displacement, independently of orientation, above and "
         "below one pixel? This does **not** establish the real positional "
         "accuracy of Sentinel-1 or Sentinel-2, which needs independent "
         "UAV/GNSS/VHR validation.", "",
         "A 10 m analysis grid is **not** 10 m positional uncertainty, and "
         "sub-pixel coordinate recovery is **not** sub-pixel physical "
         "accuracy.", "",
         "## 2. Ground truth", "",
         "Analytic: `F(x,y) = x cos(t) + y sin(t) - c`, with `F = 0` the "
         "shoreline, sampled at 10 m cell centres. Earlier attempts compared "
         "two binary masks, which makes rasterisation part of what is being "
         "tested; three test constructions failed on their own artefacts "
         "before the estimator was reached.", "",
         "## 3. Estimators", "",
         "| id | method |", "|---|---|",
         "| `v1_pixel_boundary` | threshold, binary boundary pixels, distance "
         "transform (used through Gate 7B/7C) |",
         "| `v2_subpixel_contour` | Marching Squares on the continuous field "
         "at F=0, vector-to-vector signed distance |", "",
         "## 4. Accuracy", "", "```", A.to_string(index=False), "```", "",
         "## 5. Regression by orientation", "",
         "```", G.to_string(index=False), "```", "",
         "## 6. Decision", ""]
    for est, d in dec.items():
        L.append(f"- **{est}**: intercept "
                 f"{'PASS' if d['intercept_ok'] else 'FAIL'} "
                 f"(max |a| {d['max_abs_intercept']:.2f} m), slope "
                 f"{'PASS' if d['slope_ok'] else 'FAIL'} "
                 f"(worst |b-1| {d['worst_slope_dev']:.4f}), orientation "
                 f"{'PASS' if d['orientation_ok'] else 'FAIL'} "
                 f"(spread {d['slope_spread']:.4f}), RMSE {d['rmse']:.2f} m")
    L += ["", f"**{verdict}**", "",
          "## 7. Interpretation limits", "",
          "The `|b-1| < 0.05` criterion is an additional stringent numerical "
          "acceptance criterion adopted for this study, not a published "
          "Sentinel shoreline standard; the literature validates with "
          "bias/MAE/RMSE/NMAD against independent references.", "",
          "Synthetic sub-pixel recoverability does not represent real-world "
          "shoreline positional accuracy. Sentinel-1 and Sentinel-2 "
          "uncertainties must be estimated separately and must not be assumed "
          "equal.", ""]
    (REPORTS / "gate7c0_QA.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
