#!/usr/bin/env python
"""GATE 7C0 — calibrate the shoreline distance estimator before trusting it.

Gate 7B's synthetic check reported -20.0 m for two IDENTICAL masks and the
tolerance |median| <= CELL let it pass as "zero". It is not zero. It is a
deterministic one-cell bias in the estimator, and it happens to equal H2's
entire measured offset of -20.0 m, so none of the H1/H2/H3 numbers could be
read as physical until this is fixed.

THE CAUSE. shoreline_px(mask) = mask & ~erosion(mask) returns pixels lying
INSIDE the mask, so for identical masks the distance transform of the optical
mask evaluates to exactly one cell at every one of them. A boundary
represented by inside-pixel centres was being compared against a boundary
represented by the mask edge: pixel-centre against pixel-edge, worth one cell
every time.

THE FIX, analytic rather than approximate. For a binary mask the true boundary
lies halfway between the last inside and first outside pixel centre, so

    sdf(x) = -(EDT(mask) - 0.5)   inside
           = +(EDT(~mask) - 0.5)  outside

puts zero on that half-cell line. The SAR boundary pixel centre is itself half
a cell inside its own true boundary, so the corrected offset is

    d = sdf_optical(SAR boundary pixel) + 0.5 * CELL

For identical masks this gives -10 + 10 = 0 exactly, and for a one-cell
shrink it gives -20 m exactly. No marching-squares dependency is needed and
the correction is exact rather than fitted.

The calibration is then verified across geometries -- straight, curved,
concave, island, ring with hole, narrow channel -- at known offsets from
-5 to +5 cells, requiring recovered = a + b*true with a ~ 0 and b ~ 1. A
constant additive correction is only legitimate if it holds across all of
them, so this is tested rather than assumed.

Outputs
-------
outputs/tables/hist25b_gate7c0_estimator_calibration.csv
outputs/figures/historical_bathymetry/png/hist25b_gate7c0_calibration.png
"""
from __future__ import annotations

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
from scipy import ndimage

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CELL = 20.0
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
OFFSETS_CELLS = (-5, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 5)
# A raster estimator can be unbiased on axis-aligned edges and biased on
# diagonals, because the 8-connected boundary and the distance transform both
# behave differently there. Testing only horizontal/vertical edges would miss
# exactly that, so straight shorelines are swept through orientation.
ORIENTATIONS_DEG = (0, 15, 30, 45, 60, 75, 90)
ORIENT_OFFSETS_CELLS = (-2, -1, -0.5, 0, 0.5, 1, 2)
PASS_INTERCEPT_CELLS = 0.5      # half a cell: the resolution floor
PASS_SLOPE_TOL = 0.05


def sdf(mask):
    """Signed distance field in cells, zero on the half-cell boundary line.

    Placing zero between the last inside and first outside pixel centre is
    what removes the pixel-centre / pixel-edge asymmetry; without the 0.5 the
    field jumps from -1 to +1 with no zero crossing at all."""
    d_in = ndimage.distance_transform_edt(mask)
    d_out = ndimage.distance_transform_edt(~mask)
    return np.where(mask, -(d_in - 0.5), +(d_out - 0.5))


def boundary_px(mask):
    return mask & ~ndimage.binary_erosion(mask, np.ones((3, 3), bool))


def offset_uncorrected(sar, opt, valid=None):
    """Gate 7B's estimator, kept so the bias can be measured rather than
    described."""
    d_in = ndimage.distance_transform_edt(opt) * CELL
    d_out = ndimage.distance_transform_edt(~opt) * CELL
    b = boundary_px(sar)
    if valid is not None:
        b = b & valid
    ys, xs = np.where(b)
    return np.where(opt[ys, xs], -d_in[ys, xs], +d_out[ys, xs])


def offset_corrected(sar, opt, valid=None):
    """Half-cell corrected signed offset, positive where SAR extends beyond.

    `valid` restricts which boundary pixels are sampled. A synthetic strip has
    artificial END caps that are part of boundary_px but are nowhere near the
    shoreline under test; including them drags the median toward zero and
    makes a correct estimator look like it recovers half the offset."""
    sd = sdf(opt) * CELL
    b = boundary_px(sar)
    if valid is not None:
        b = b & valid
    ys, xs = np.where(b)
    return sd[ys, xs] + 0.5 * CELL


def shapes(n=400):
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n / 2
    rr = np.hypot(yy - cy, xx - cx)
    out = {}
    # inset from the frame: a half-plane touching the image border puts the
    # FRAME edges into the "shoreline", and distances are then measured to the
    # far side of the array rather than across the boundary
    out["straight"] = (xx < cx) & (xx > 40) & (yy > 40) & (yy < n - 40)
    out["curved_circle"] = rr < 130
    conc = (rr < 150) & ~(np.hypot(yy - cy + 90, xx - cx) < 90)
    out["concave"] = conc
    out["island_ring"] = (rr < 150) & ~(rr < 45)
    out["narrow_channel"] = ((np.abs(yy - cy) < 3) & (np.abs(xx - cx) < 170)) \
        | (rr < 130)
    return out, (yy, xx, rr, cy, cx)


def shrink_grow(mask, cells):
    """Offset a mask by a known number of cells; fractional offsets are made
    by thresholding the signed distance field, which is exactly how a
    subpixel shoreline shift behaves."""
    return sdf(mask) < -cells


def orientation_suite(n=600, margin=60):
    """Straight shorelines swept through orientation.

    Deliberately a FULL half-plane, not a strip. Two earlier attempts used an
    inset strip and both failed on their own end caps: boundary_px returns the
    whole outline, so the artificial ends were sampled alongside the shoreline
    and dragged the median. Here the only exclusion is a margin around the
    array border, which is unambiguous.
    """
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n / 2
    interior = ((xx > margin) & (xx < n - margin) &
                (yy > margin) & (yy < n - margin))
    rows = []
    for ang in ORIENTATIONS_DEG:
        th = np.radians(ang)
        perp = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        opt = perp < 0
        for oc in ORIENT_OFFSETS_CELLS:
            sar = perp < oc
            dc = offset_corrected(sar, opt, valid=interior)
            du = offset_uncorrected(sar, opt, valid=interior)
            if dc.size < 50:
                continue
            true_m = oc * CELL
            rows.append(dict(orientation_deg=ang, true_offset_cells=oc,
                             true_offset_m=true_m,
                             corrected_median_m=float(np.median(dc)),
                             uncorrected_median_m=float(np.median(du)),
                             residual_m=float(np.median(dc)) - true_m,
                             mad_m=float(1.4826 * np.median(
                                 np.abs(dc - np.median(dc)))),
                             n_px=int(dc.size)))
    return pd.DataFrame(rows)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    S, _ = shapes()
    rows = []
    print("=" * 78)
    print("ESTIMATOR CALIBRATION — known offsets, several geometries")
    print("=" * 78)
    print(f"  cell = {CELL:.0f} m; positive offset = SAR beyond optical")
    for nm, opt in S.items():
        for oc in OFFSETS_CELLS:
            sar = shrink_grow(opt, -oc)   # +oc cells outward
            if sar.sum() < 50 or (~sar).sum() < 50:
                continue
            du = offset_uncorrected(sar, opt)
            dc = offset_corrected(sar, opt)
            true_m = oc * CELL
            rows.append(dict(geometry=nm, true_offset_cells=oc,
                             true_offset_m=true_m,
                             uncorrected_median_m=float(np.median(du)),
                             corrected_median_m=float(np.median(dc)),
                             uncorrected_bias_m=float(np.median(du)) - true_m,
                             corrected_bias_m=float(np.median(dc)) - true_m,
                             corrected_nmad_m=float(1.4826 * np.median(
                                 np.abs(dc - np.median(dc)))),
                             corrected_p90_m=float(np.percentile(np.abs(
                                 dc - true_m), 90))))
    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "hist25b_gate7c0_estimator_calibration.csv",
             index=False)

    print("\n  IDENTICAL MASKS (true offset = 0) — the test Gate 7B mis-passed:")
    z = R[R.true_offset_cells == 0]
    for r in z.itertuples():
        print(f"    {r.geometry:<16} uncorrected {r.uncorrected_median_m:+7.1f} m"
              f"   corrected {r.corrected_median_m:+7.1f} m")
    print(f"\n    uncorrected bias: median {z.uncorrected_median_m.median():+.1f} m "
          f"-- a deterministic one-cell floor, equal to H2's entire measured "
          f"offset")
    print(f"    corrected   bias: median {z.corrected_median_m.median():+.1f} m")

    print("\n  linear calibration recovered = a + b * true:")
    for nm, g in R.groupby("geometry"):
        b, a = np.polyfit(g.true_offset_m, g.corrected_median_m, 1)
        bu, au = np.polyfit(g.true_offset_m, g.uncorrected_median_m, 1)
        # the floor is half a cell; a tighter bar would fail on
        # quantisation rather than on estimator bias
        ok = abs(a) < CELL / 2 and abs(b - 1) < 0.05
        print(f"    {nm:<16} corrected a={a:+6.2f} m b={b:5.3f}  "
              f"| uncorrected a={au:+6.2f} m b={bu:5.3f}   "
              f"{'PASS' if ok else 'CHECK'}")

    all_b, all_a = np.polyfit(R.true_offset_m, R.corrected_median_m, 1)
    print(f"\n  pooled: a = {all_a:+.2f} m, b = {all_b:.4f}")
    const = R[R.true_offset_cells != 0].uncorrected_bias_m
    print(f"  uncorrected bias across all geometries and offsets: "
          f"median {const.median():+.1f} m, spread "
          f"{const.min():+.1f}..{const.max():+.1f} m")
    if const.max() - const.min() < 5:
        print("  -> the old bias IS constant, so subtracting it would have "
              "been defensible; it is nonetheless removed at source.")
    else:
        print("  -> the old bias is NOT constant across geometries, so simply "
              "subtracting 20 m from the Gate 7B results would have been "
              "wrong. It had to be fixed in the estimator.")

    print("\n" + "=" * 78)
    print("RESOLUTION FLOOR")
    print("=" * 78)
    sub = R[(R.true_offset_cells.abs() <= 0.5) & (R.true_offset_cells != 0)]
    print(f"  raster resolution            : {CELL:.0f} m")
    print(f"  subpixel method              : half-cell signed distance field")
    print(f"  synthetic zero bias          : "
          f"{z.corrected_median_m.median():+.2f} m")
    print(f"  subpixel recovery |<=0.5 cell|: median error "
          f"{sub.corrected_bias_m.abs().median():.2f} m")
    print(f"  minimum resolvable shift     : ~{CELL/2:.0f} m "
          f"(half a cell; below this the sign is reliable but the magnitude "
          f"is quantised)")

    # ---------------------------------------------------- orientation
    print("\n" + "=" * 78)
    print("ORIENTATION TEST — an estimator can be clean on axis-aligned edges")
    print("and biased on diagonals; horizontal/vertical alone would miss it")
    print("=" * 78)
    O = orientation_suite()
    O.to_csv(CFG.TABLES / "hist25b_gate7c0_orientation_validation.csv",
             index=False)
    piv = O.pivot_table(index="orientation_deg", columns="true_offset_cells",
                        values="residual_m")
    print("  residual (m) = recovered - true, by orientation and true offset:")
    print(piv.to_string(float_format=lambda v: f"{v:+.1f}"))
    by_ang = O.groupby("orientation_deg").residual_m.median()
    print(f"\n  median residual by angle: "
          f"{', '.join(f'{a}deg {v:+.1f}m' for a, v in by_ang.items())}")
    ang_spread = float(by_ang.max() - by_ang.min())
    # The slope test must be run on RESOLVABLE offsets. Sub-cell shifts are
    # quantisation-limited by construction on a 20 m grid -- the same reason
    # |d| < 10 m is treated as unresolved downstream -- so including them asks
    # the estimator to beat its own resolution and depresses b artificially.
    res = O[O.true_offset_cells.abs() >= 1.0]
    sub = O[(O.true_offset_cells.abs() > 0) & (O.true_offset_cells.abs() < 1.0)]
    b_all, a_all = np.polyfit(res.true_offset_m, res.corrected_median_m, 1)
    b_sub, a_sub = np.polyfit(O.true_offset_m, O.corrected_median_m, 1)
    print(f"  pooled regression, RESOLVABLE offsets (|d| >= 1 cell): "
          f"a = {a_all:+.2f} m, b = {b_all:.4f}")
    print(f"  pooled regression including sub-cell offsets: "
          f"a = {a_sub:+.2f} m, b = {b_sub:.4f}")
    print(f"    sub-cell residuals: max |r| = "
          f"{sub.residual_m.abs().max():.1f} m ({sub.residual_m.abs().max()/CELL:.2f} cell)"
          f" -- quantisation, not bias")
    print(f"  orientation spread of median residual: {ang_spread:.1f} m")

    # interaction: does the residual depend on true_offset differently per angle?
    inter = []
    for ang, g in res.groupby("orientation_deg"):
        bb, aa = np.polyfit(g.true_offset_m, g.corrected_median_m, 1)
        inter.append(dict(orientation_deg=ang, intercept_m=aa, slope=bb))
    I = pd.DataFrame(inter)
    print("\n  per-angle regression on resolvable offsets "
          "(tests residual ~ true_offset x orientation):")
    print(I.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    slope_spread = float(I.slope.max() - I.slope.min())
    print(f"  slope spread across angles: {slope_spread:.4f}")

    ok_a = abs(a_all) < PASS_INTERCEPT_CELLS * CELL
    ok_b = abs(b_all - 1) < PASS_SLOPE_TOL
    ok_ang = ang_spread < PASS_INTERCEPT_CELLS * CELL
    ok_int = slope_spread < 0.10
    print("\n" + "=" * 78)
    print("GATE 7C0 DECISION")
    print("=" * 78)
    print(f"  |a| < {PASS_INTERCEPT_CELLS} cell ({PASS_INTERCEPT_CELLS*CELL:.0f} m)"
          f"          : {'PASS' if ok_a else 'FAIL'}  (a = {a_all:+.2f} m)")
    print(f"  |b - 1| < {PASS_SLOPE_TOL}                   : "
          f"{'PASS' if ok_b else 'FAIL'}  (b = {b_all:.4f})")
    print(f"  no orientation dependence            : "
          f"{'PASS' if ok_ang else 'FAIL'}  (spread {ang_spread:.1f} m)")
    print(f"  no offset x orientation interaction  : "
          f"{'PASS' if ok_int else 'FAIL'}  (slope spread {slope_spread:.4f})")
    verdict = "PASS" if all((ok_a, ok_b, ok_ang, ok_int)) else "FAIL"
    print(f"\n  GATE 7C0 = {verdict}")
    if verdict != "PASS":
        print("  Gate 7C results must NOT be interpreted physically.")

    _fig(R, z, O)
    print("\n  Gate 7C must be re-run with the corrected estimator. Do NOT "
          "simply subtract 20 m from its earlier output.")


def _fig(R, z, O=None):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    a = ax[0]
    for nm, g in R.groupby("geometry"):
        a.plot(g.true_offset_m, g.uncorrected_median_m, "o--", ms=4, alpha=0.75,
               label=nm)
    lim = [R.true_offset_m.min(), R.true_offset_m.max()]
    a.plot(lim, lim, color=INK, lw=1.4, label="ideal 1:1")
    a.axhline(0, color=GREY, lw=0.8); a.axvline(0, color=GREY, lw=0.8)
    a.set_xlabel("true offset (m)"); a.set_ylabel("recovered (m)")
    a.legend(fontsize=7); a.grid(alpha=0.25)
    a.set_title("a · Gate 7B estimator — offset by one cell everywhere",
                fontsize=10.2, loc="left")
    a = ax[1]
    for nm, g in R.groupby("geometry"):
        a.plot(g.true_offset_m, g.corrected_median_m, "o-", ms=4, label=nm)
    a.plot(lim, lim, color=INK, lw=1.4, label="ideal 1:1")
    a.axhline(0, color=GREY, lw=0.8); a.axvline(0, color=GREY, lw=0.8)
    a.set_xlabel("true offset (m)"); a.set_ylabel("recovered (m)")
    a.legend(fontsize=7); a.grid(alpha=0.25)
    a.set_title("b · half-cell corrected estimator", fontsize=10.2, loc="left")
    a = ax[2]
    w = 0.35
    xs = np.arange(len(z))
    a.bar(xs - w / 2, z.uncorrected_median_m, width=w, color=RED,
          label="uncorrected")
    a.bar(xs + w / 2, z.corrected_median_m, width=w, color=GREEN,
          label="corrected")
    a.axhline(0, color=INK, lw=1.2)
    a.set_xticks(xs); a.set_xticklabels(z.geometry, rotation=30, fontsize=7.5,
                                        ha="right")
    a.set_ylabel("recovered offset for IDENTICAL masks (m)")
    a.legend(fontsize=8); a.grid(alpha=0.25, axis="y")
    a.set_title("c · the test that should have returned zero",
                fontsize=10.2, loc="left")
    fig.suptitle("hist25b Gate 7C0 · shoreline distance estimator calibration",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate7c0_calibration.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGDIR/'hist25b_gate7c0_calibration.png'}")
    if O is None or O.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    a = ax[0]
    for oc, g in O.groupby("true_offset_cells"):
        a.plot(g.orientation_deg, g.residual_m, "o-", ms=4,
               label=f"{oc:+g} px")
    a.axhline(0, color=INK, lw=1.2)
    a.axhspan(-CELL / 2, CELL / 2, color=GREY, alpha=0.18,
              label="half-cell floor")
    a.set_xlabel("shoreline orientation (deg)")
    a.set_ylabel("residual = recovered - true (m)")
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.25)
    a.set_title("a · residual vs orientation — a diagonal-only bias would "
                "show here", fontsize=10.2, loc="left")
    a = ax[1]
    for ang, g in O.groupby("orientation_deg"):
        a.plot(g.true_offset_m, g.corrected_median_m, "o-", ms=4,
               label=f"{ang}deg")
    lim = [O.true_offset_m.min(), O.true_offset_m.max()]
    a.plot(lim, lim, color=INK, lw=1.4, ls="--", label="1:1")
    a.set_xlabel("true offset (m)"); a.set_ylabel("recovered (m)")
    a.legend(fontsize=7, ncol=2); a.grid(alpha=0.25)
    a.set_title("b · recovery at every orientation", fontsize=10.2, loc="left")
    fig.suptitle("hist25b Gate 7C0 · orientation validation", y=1.02,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate7c0_orientation.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGDIR/'hist25b_gate7c0_orientation.png'}")


if __name__ == "__main__":
    main()
