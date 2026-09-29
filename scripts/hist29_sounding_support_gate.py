#!/usr/bin/env python
"""HIST 29 — does an INDEPENDENT waterline exist at each stage? TIN bracketing.

Replaces a criterion of mine that was wrong in principle. hist28 asked how many
soundings lie within 0.5 m of a stage, and I then mis-stated even that: the code
printed 39 soundings within 0.5 m of H2 and I wrote it up as 0. Both the metric
and the write-up are corrected here.

PROXIMITY IS NOT BRACKETING. An independent isobath at level H exists only where
the sounded surface actually CROSSES H:

    z_low < H < z_high     within one triangle of the sounding TIN

If every sounding near a stage lies below it, no interpolation produces a
measured contour there -- only an extrapolated one, which is not independent
control and must not be called it. H2 is the case in point: 39 soundings sit
within 0.5 m of 15.444 m, and ZERO triangles bracket it, because the highest
sounding in the whole survey is 15.32 m.

THE SUPPORT FILTER IS NOT OPTIONAL EITHER. A Delaunay triangulation of scattered
soundings spans every survey gap, including the convex hull, with slivers up to
139 km long. Those triangles bracket levels they have no business bracketing.
The soundings have a median nearest-neighbour spacing of 357 m (p90 555 m), and
the triangle longest-edge distribution has p50 689 m and p75 1041 m, so
MAX_EDGE_M = 1000 admits the normal survey geometry and rejects gap-spanning
slivers. The sensitivity of every count to that choice is reported rather than
hidden.

WHAT THIS GATE DECIDES, per contour:

    DIRECT       the TIN brackets the stage over a usable isobath length, so an
                 independent, sounding-only reference contour exists
    UNAVAILABLE  it does not; the contour cannot be validated against the
                 soundings at its own elevation, and saying so is the result

Nothing here uses S1, S2, any H1/H2/H3 shoreline, the P20 footprint, or any
shoreline boundary condition. That isolation is the point: a reference built
partly from the thing being tested is not a reference.

Outputs
-------
outputs/tables/hist29_sounding_support.csv
data/processed/bathymetry/sounding_isobaths.gpkg
outputs/figures/historical_bathymetry/png/hist29_support.png
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
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3")
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
GPKG = ROOT / "data/processed/bathymetry/sounding_isobaths.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
MAX_EDGE_M = 1000.0
EDGE_SENSITIVITY = (500.0, 1000.0, 2000.0, 5000.0)
MIN_SUPPORTED_KM = 20.0        # below this an isobath is a fragment, not control
STAGES = {"H3": 14.188588, "H2": 15.444365, "H1": 17.103588}


def tin_isobath(xy, z, tri, keep, H):
    """Segments where the sounded surface crosses H, inside supported triangles.

    Linear interpolation along the two crossing edges of each bracketing
    triangle. No smoothing, no gridding: gridding first would let an
    interpolator invent a crossing where the soundings have none."""
    segs, grads = [], []
    for t in tri[keep]:
        p = xy[t]
        v = z[t]
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = v[a], v[b]
            if (za - H) * (zb - H) < 0:
                f = (H - za) / (zb - za)
                pts.append(p[a] + f * (p[b] - p[a]))
        if len(pts) == 2 and not np.allclose(pts[0], pts[1]):
            segs.append(LineString(pts))
            # plane through the three vertices -> the LOCAL slope at this
            # isobath, which is the right quantity for converting a vertical
            # uncertainty into a horizontal one
            A = np.c_[p[:, 0] - p[0, 0], p[:, 1] - p[0, 1], np.ones(3)]
            try:
                c, *_ = np.linalg.lstsq(A, v, rcond=None)
                grads.append(float(np.hypot(c[0], c[1])))
            except Exception:
                grads.append(np.nan)
    return segs, np.array(grads)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("HIST 29 — independent waterline support, by TIN bracketing")
    print("=" * 78)
    print(f"  git {commit}")
    print("  soundings only: no S1, no S2, no H1/H2/H3 shoreline, no P20")

    P = pd.read_parquet(PRIMARY)
    xy = np.c_[P.x.to_numpy(), P.y.to_numpy()]
    z = P.H_bed_evrf2019_m.to_numpy()
    nn, _ = cKDTree(xy).query(xy, k=2)
    print(f"\n  {len(P):,} soundings; nearest-neighbour spacing median "
          f"{np.median(nn[:, 1]):.0f} m, p90 {np.percentile(nn[:, 1], 90):.0f} m")
    print(f"  bed elevation range {z.min():.2f} .. {z.max():.2f} m EVRF2019")

    T = Delaunay(xy)
    tri = T.simplices
    edge = np.max([np.linalg.norm(xy[tri[:, a]] - xy[tri[:, b]], axis=1)
                   for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    print(f"  {len(tri):,} triangles; longest edge p50 {np.percentile(edge,50):.0f} m, "
          f"p75 {np.percentile(edge,75):.0f} m, max {edge.max()/1000:.0f} km")
    print(f"  support filter: longest edge <= {MAX_EDGE_M:.0f} m")

    rows, lines = [], {}
    for cid, H in STAGES.items():
        zt = z[tri]
        brack = (zt.min(1) < H) & (zt.max(1) > H)
        keep = brack & (edge <= MAX_EDGE_M)
        segs, grads = tin_isobath(xy, z, tri, keep, H)
        L = sum(s.length for s in segs) / 1000.0
        merged = linemerge(unary_union(segs)) if segs else None
        if merged is not None:
            lines[cid] = merged
        sens = {f"n_bracketing_edge_{int(e)}m": int((brack & (edge <= e)).sum())
                for e in EDGE_SENSITIVITY}
        r = dict(
            contour_id=cid, stage_m=H,
            n_within_0_10m=int((np.abs(z - H) <= 0.10).sum()),
            n_within_0_25m=int((np.abs(z - H) <= 0.25).sum()),
            n_within_0_50m=int((np.abs(z - H) <= 0.50).sum()),
            n_below_stage=int((z < H).sum()),
            n_above_stage=int((z > H).sum()),
            n_bracketing_triangles_all=int(brack.sum()),
            n_bracketing_triangles=int(keep.sum()),
            supported_isobath_length_km=float(L),
            local_slope_median=float(np.nanmedian(grads)) if len(grads) else np.nan,
            **sens)
        r["absolute_validation"] = (
            "DIRECT" if (keep.sum() > 0 and L >= MIN_SUPPORTED_KM)
            else "UNAVAILABLE")
        rows.append(r)

        print(f"\n  {cid} at {H:.3f} m")
        print(f"    soundings within  0.10 / 0.25 / 0.50 m : "
              f"{r['n_within_0_10m']:>5d} / {r['n_within_0_25m']:>5d} / "
              f"{r['n_within_0_50m']:>5d}")
        print(f"    soundings below / above the stage      : "
              f"{r['n_below_stage']:>5d} / {r['n_above_stage']:>5d}")
        print(f"    bracketing triangles, all / supported  : "
              f"{r['n_bracketing_triangles_all']:>5d} / "
              f"{r['n_bracketing_triangles']:>5d}")
        print(f"      by max-edge " + "  ".join(
            f"{int(e)}m:{sens[f'n_bracketing_edge_{int(e)}m']}"
            for e in EDGE_SENSITIVITY))
        print(f"    supported isobath length               : {L:8.1f} km")
        if len(grads) and np.isfinite(grads).any():
            print(f"    local slope at the isobath (median)    : "
                  f"{np.nanmedian(grads):.4f} m/m")
        print(f"    -> absolute_validation = {r['absolute_validation']}")
        if r["n_within_0_50m"] > 0 and r["n_bracketing_triangles_all"] == 0:
            print(f"       NOTE: {r['n_within_0_50m']} soundings sit within "
                  f"0.5 m of this stage and NONE bracket it.")
            print("       Proximity is not bracketing; an isobath here would "
                  "be extrapolation.")

    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "hist29_sounding_support.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'hist29_sounding_support.csv'}")

    if lines:
        GPKG.parent.mkdir(parents=True, exist_ok=True)
        if GPKG.exists():
            GPKG.unlink()
        gpd.GeoDataFrame(
            dict(contour_id=list(lines), stage_m=[STAGES[c] for c in lines]),
            geometry=[lines[c] for c in lines],
            crs=CFG.CRS_METRIC).to_file(GPKG, layer="sounding_isobaths",
                                        driver="GPKG")
        print(f"-> {GPKG}")

    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    for r in R.itertuples():
        print(f"  {r.contour_id}: absolute_validation = {r.absolute_validation}"
              + ("" if r.absolute_validation == "DIRECT" else
                 f"   (no bracketing; highest sounding {z.max():.2f} m "
                 f"< {r.stage_m:.2f} m)"))
    direct = R[R.absolute_validation == "DIRECT"]
    if direct.empty:
        raise SystemExit("no stage has independent sounding support")
    print(f"\n  Only {', '.join(direct.contour_id)} can be validated against "
          f"the soundings at its own elevation.")
    print("  The others are NOT invalid -- they are externally unvalidated, and")
    print("  must be carried as soft constraints with that flag, never as")
    print("  independently validated true waterlines.")
    figure(R, xy, z, lines)
    print("\nSTOP. Next: S2_H3 and S1_H3 against this sounding-only isobath.")


def figure(R, xy, z, lines):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.4))
    sc = ax[0].scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, c=z, s=2,
                       cmap="viridis", linewidths=0)
    cb = fig.colorbar(sc, ax=ax[0], fraction=.03, pad=.01)
    cb.set_label("sounded bed, EVRF2019 (m)", fontsize=8)
    for cid, g in lines.items():
        gpd.GeoSeries([g]).plot(ax=ax[0], color=RED, linewidth=1.2)
    ax[0].set_title("soundings and the supported H3 isobath (red)", color=INK)
    ax[0].set_xlabel("easting (km)"); ax[0].set_ylabel("northing (km)")
    ax[0].set_aspect("equal")

    x = np.arange(len(R))
    ax[1].bar(x - .2, R.n_within_0_50m, .4, color=GREY,
              label="soundings within 0.5 m")
    ax[1].bar(x + .2, R.n_bracketing_triangles, .4, color=BLUE,
              label=f"triangles bracketing the stage (edge <= {MAX_EDGE_M:.0f} m)")
    for i, r in enumerate(R.itertuples()):
        ax[1].text(i, max(r.n_within_0_50m, r.n_bracketing_triangles) + 12,
                   r.absolute_validation, ha="center", fontsize=9,
                   color=GREEN if r.absolute_validation == "DIRECT" else RED)
    ax[1].set_xticks(x); ax[1].set_xticklabels(
        [f"{r.contour_id}\n{r.stage_m:.2f} m" for r in R.itertuples()])
    ax[1].set_ylabel("count")
    ax[1].set_title("proximity is not bracketing", color=INK)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("hist29 · where an independent waterline actually exists",
                 color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "hist29_support.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
