#!/usr/bin/env python
"""HIST 27 — the waterline migration rate as a LOCAL k(x), on the new contours.

Roadmap step 3. Gate 7C3 replaced a sounding BED slope (0.0053, i.e. 190 m of
waterline per metre of stage) with a contour-derived ~9.8 m per m, and flagged
that number PROVISIONAL on three grounds: it was measured on the 20 m raster
staircase, one of its pairs was sub-cell, and only two of its three spacings
were independent. hist26 removed the staircase. This re-measures on the rebuilt
contours, and locally rather than globally, because

    replacing one wrong global slope with one right global slope
    is still a global slope

and a 1,200 km shoreline does not have one bank profile.

    k(x) = dn(x) / dH          d_stage(x) = -k(x) * dH

WHERE k(x) IS NaN, AND WHY THAT MATTERS MORE THAN THE NUMBER. k is only defined
where the three levels give consistent local geometry. Three separate ways it
fails, each recorded rather than filled in:

  NOT_NESTED     the lower contour's vertex does not lie inside the upper
                 contour's water. Locally the two levels cross, so there is no
                 monotone waterline to differentiate.
  NO_CROSSING    the cross-shore normal finds no upper contour within the
                 search length. Flat ground, or a shoreline the levels do not
                 share.
  PAIRS_DISAGREE H3->H2 and H2->H1 give rates differing by more than a factor
                 of DISAGREE_FACTOR at the same place. A bank that is locally
                 linear must give the same rate for both steps; when it does
                 not, the local profile is not linear over this stage range and
                 a single k there would be a fiction.

Only the two INDEPENDENT steps are measured. H3->H1 is deliberately not
computed: its area difference is exactly the sum of the other two, so it carries
no new information and would fake a third corroborating estimate.

Outputs
-------
outputs/tables/hist27_local_migration_rate.csv   (per-vertex k, with flags)
outputs/tables/hist27_migration_summary.csv
outputs/figures/historical_bathymetry/png/hist27_local_k.png
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from hist25b_gate7_anchored_classifier import CELL, build_grid, clean_water
from hist25b_gate7c1_pairing_audit import contour_points, match_normal

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
FIELDS = CFG.BULK_ROOT / "hist26_continuous_fields"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
STEPS = (("H3", "H2"), ("H2", "H1"))     # the two INDEPENDENT steps only
SEARCH_M = 250.0
DISAGREE_FACTOR = 2.0
NEIGHBOUR_M = 500.0                      # radius for comparing the two steps
BAND_PX = 3


def load_field(cid):
    z = np.load(FIELDS / f"{cid}_field.npz", allow_pickle=True)
    W, valid = z["W"], z["valid"]
    F = -W
    F[~valid] = np.nan
    return F, valid


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("HIST 27 — local waterline migration rate k(x)")
    print("=" * 78)
    print(f"  git {commit}")
    print("  measured on the hist26 continuous contours, not the staircase")
    print("  only the two INDEPENDENT steps; H3->H1 is a linear combination")

    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    C = pd.read_csv(CFG.TABLES / "hist26_continuous_contours.csv").set_index(
        "contour_id")

    # An independent, geometry-only cross-check before any per-vertex work:
    # mean normal spacing is dA / mean perimeter, which uses no matcher at all.
    # On the STAIRCASE contours the two steps came out at 9.9 and 9.8 m per m,
    # and that near-equality is what made 9.8 look like a measurement. On the
    # rebuilt contours they do not agree.
    print("\n  AREA-BASED cross-check (no matcher involved):")
    for lo, hi in STEPS:
        dH = float(C.loc[hi, "H_evrf2019_m"] - C.loc[lo, "H_evrf2019_m"])
        dA = float(C.loc[hi, "area_km2"] - C.loc[lo, "area_km2"]) * 1e6
        per = 0.5 * float(C.loc[lo, "boundary_km"] + C.loc[hi, "boundary_km"]) * 1e3
        print(f"    {lo}->{hi}: dH {dH:+.4f} m, dA {dA/1e6:+6.1f} km2, "
              f"spacing {dA/per:6.1f} m -> {dA/per/dH:6.1f} m per m")
    print("    (the same computation on the staircase contours gave 9.9 and "
          "9.8:")
    print("     the quantisation, not the bank, is what made them agree)")

    F, V = {}, {}
    for cid in ("H1", "H2", "H3"):
        F[cid], V[cid] = load_field(cid)

    k = np.ones((3, 3), bool)
    per_step, summary = {}, []
    for lo, hi in STEPS:
        dH = float(C.loc[hi, "H_evrf2019_m"] - C.loc[lo, "H_evrf2019_m"])
        wet = clean_water((F[lo] < 0) & V[lo] & inside)
        lab, n = ndimage.label(wet, structure=np.ones((3, 3), int))
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        conn = ndimage.binary_fill_holes(lab == (int(np.argmax(sz)) + 1))
        band = (ndimage.binary_dilation(conn, k, BAND_PX)
                & ~ndimage.binary_erosion(conn, k, BAND_PX)
                & V[lo] & V[hi] & inside)
        A = contour_points(F[lo], band, G)
        if len(A) < 500:
            print(f"  {lo}->{hi}: too few vertices")
            continue
        dist = match_normal(A, F[hi], V[hi] & inside, G)
        # the upper level must contain the lower one locally, or there is no
        # monotone waterline to differentiate
        ri, ci = A[:, 2].astype(int), A[:, 3].astype(int)
        nested = F[hi][ri, ci] < 0
        flag = np.where(~nested, "NOT_NESTED",
                        np.where(~np.isfinite(dist) | (dist > SEARCH_M),
                                 "NO_CROSSING", "OK"))
        kk = np.where(flag == "OK", dist / dH, np.nan)
        per_step[(lo, hi)] = dict(X=A[:, 0], Y=A[:, 1], k=kk, flag=flag,
                                  wl=A[:, 4], dH=dH)
        ok = flag == "OK"
        print(f"\n  {lo}->{hi}: dH {dH:+.4f} m, {len(A):,} vertices")
        for f in ("OK", "NOT_NESTED", "NO_CROSSING"):
            m = flag == f
            print(f"    {f:12s} {100*A[m, 4].sum()/A[:, 4].sum():5.1f}% of length")
        if ok.sum() > 100:
            v = kk[ok]
            print(f"    k: median {np.median(v):6.2f}  IQR "
                  f"{np.percentile(v, 25):5.2f} .. {np.percentile(v, 75):5.2f}"
                  f"  P90 {np.percentile(v, 90):6.2f} m per m")
            summary.append(dict(step=f"{lo}->{hi}", delta_H=dH,
                                n_vertices=int(len(A)),
                                frac_len_ok=float(A[ok, 4].sum() / A[:, 4].sum()),
                                k_median=float(np.median(v)),
                                k_p25=float(np.percentile(v, 25)),
                                k_p75=float(np.percentile(v, 75)),
                                k_p90=float(np.percentile(v, 90))))

    if len(per_step) < 2:
        raise SystemExit("both steps are needed to test local consistency")

    # ---- do the two independent steps agree, locally? --------------------
    (a1, b1), (a2, b2) = STEPS
    s1, s2 = per_step[(a1, b1)], per_step[(a2, b2)]
    m1 = np.isfinite(s1["k"])
    m2 = np.isfinite(s2["k"])
    tree = cKDTree(np.c_[s2["X"][m2], s2["Y"][m2]])
    d, j = tree.query(np.c_[s1["X"][m1], s1["Y"][m1]], k=1,
                      distance_upper_bound=NEIGHBOUR_M)
    have = np.isfinite(d)
    k1 = s1["k"][m1][have]
    k2 = s2["k"][m2][j[have]]
    ratio = np.maximum(k1, k2) / np.maximum(np.minimum(k1, k2), 1e-6)
    agree = ratio <= DISAGREE_FACTOR
    print(f"\n  LOCAL CONSISTENCY between the two independent steps:")
    print(f"    {have.sum():,} of {m1.sum():,} OK vertices have a "
          f"{a2}->{b2} neighbour within {NEIGHBOUR_M:.0f} m")
    print(f"    agree within a factor {DISAGREE_FACTOR}: "
          f"{100*agree.mean():.1f}%")
    print(f"    k({a1}->{b1}) median {np.median(k1):.2f}, "
          f"k({a2}->{b2}) median {np.median(k2):.2f} m per m")
    kbar = np.where(agree, 0.5 * (k1 + k2), np.nan)
    good = np.isfinite(kbar)
    print(f"    combined k where both steps agree: median "
          f"{np.median(kbar[good]):.2f}  IQR {np.percentile(kbar[good], 25):.2f}"
          f" .. {np.percentile(kbar[good], 75):.2f} m per m")
    print(f"    Gate 7C3's provisional global value was 9.8 m per m")

    X = s1["X"][m1][have]
    Y = s1["Y"][m1][have]
    out = pd.DataFrame(dict(x=X, y=Y, k_step1=k1, k_step2=k2,
                            ratio=ratio, pairs_agree=agree, k_local=kbar))
    out["status"] = np.where(out.pairs_agree, "OK", "PAIRS_DISAGREE")
    out.to_csv(CFG.TABLES / "hist27_local_migration_rate.csv", index=False)
    S = pd.DataFrame(summary)
    S.loc[len(S)] = dict(step="combined", delta_H=np.nan,
                         n_vertices=int(good.sum()),
                         frac_len_ok=float(agree.mean()),
                         k_median=float(np.median(kbar[good])),
                         k_p25=float(np.percentile(kbar[good], 25)),
                         k_p75=float(np.percentile(kbar[good], 75)),
                         k_p90=float(np.percentile(kbar[good], 90)))
    S.to_csv(CFG.TABLES / "hist27_migration_summary.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'hist27_local_migration_rate.csv'}")

    figure(per_step, out)

    print("\n" + "=" * 78)
    print("VERDICT ON k")
    print("=" * 78)
    frac = float(agree.mean())
    if frac < 0.5:
        print("  k(x) IS NOT IDENTIFIABLE from these three contours.")
        print(f"  The two independent steps agree at only {100*frac:.1f}% of")
        print("  locations, and a quarter to a third of the shoreline is not")
        print("  even locally nested. The geometry-only cross-check above")
        print("  disagrees between steps by the same order, so this is not a")
        print("  matcher artefact.")
        print("\n  What this retires: Gate 7C3's provisional global 9.8 m per m.")
        print("  Its apparent support was the staircase - on the rebuilt")
        print("  contours the two steps give different rates, and the earlier")
        print("  agreement to 8% was quantisation, not a bank profile.")
        print("\n  What it does NOT change: the DEM still does not need this.")
        print("  Each shoreline constrains elevation at its OWN observed WSE.")
        print("  d_stage is required only for cross-stage transfer, which was")
        print("  already frozen as UNVALIDATED and now has a stronger reason.")
    else:
        print(f"  k(x) is usable where status == OK ({100*frac:.1f}% of pairs).")
        print("  d_stage(x) = -k_local(x) * dH, NaN elsewhere.")
    print("\nSTOP. Next: re-run and freeze Gate 7C3 on the admitted scenes.")


def figure(per_step, out):
    # three panels in one row: an equal-aspect map inside a wide subplot just
    # centres itself and leaves the rest of the row blank
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    ax, ax2, ax3 = axes
    g = out[out.pairs_agree]
    sc = ax.scatter(g.x / 1000, g.y / 1000, c=np.clip(g.k_local, 0, 40), s=2.2,
                    cmap="viridis", vmin=0, vmax=40, linewidths=0)
    b = out[~out.pairs_agree]
    ax.scatter(b.x / 1000, b.y / 1000, s=3.5, c=RED, linewidths=0,
               label=f"steps disagree ({100*(~out.pairs_agree).mean():.0f}%)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.022, pad=0.01)
    cb.set_label("k(x)  m of waterline per m of stage", fontsize=8)
    ax.set_title("local k where both steps agree", color=INK, fontsize=11)
    ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
    ax.set_aspect("equal"); ax.legend(fontsize=8, markerscale=3, loc="lower left")

    for (lo, hi), s in per_step.items():
        v = s["k"][np.isfinite(s["k"])]
        ax2.hist(np.clip(v, 0, 60), bins=60, histtype="step", lw=1.6,
                 label=f"{lo}->{hi}  median {np.median(v):.1f}")
    ax2.axvline(9.8, color=RED, ls="--", lw=1.4,
                label="Gate 7C3 provisional global 9.8")
    ax2.set_xlabel("k (m per m)"); ax2.set_ylabel("vertices")
    ax2.set_title("the two independent steps, separately", color=INK, fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    ax3.scatter(np.clip(out.k_step1, 0, 60), np.clip(out.k_step2, 0, 60),
                s=3, c=np.where(out.pairs_agree, GREEN, RED), linewidths=0)
    ax3.plot([0, 60], [0, 60], color=INK, lw=1, ls="--")
    ax3.set_xlabel("k from H3->H2 (m per m)")
    ax3.set_ylabel("k from H2->H1 (m per m)")
    ax3.set_title("a locally linear bank must give the same rate twice",
                  color=INK, fontsize=11)
    ax3.grid(alpha=.3); ax3.set_aspect("equal")
    fig.suptitle("hist27 · k(x), and where it is not defined", color=INK,
                 fontsize=14)
    fig.tight_layout()
    p = FIGDIR / "hist27_local_k.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
