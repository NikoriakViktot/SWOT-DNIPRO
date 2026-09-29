#!/usr/bin/env python
"""GATE 7C2b — the written admission cut, from the 26-scene distribution.

Gate 7C2 deliberately set no thresholds; this file sets them, once, in writing,
with each cut tied to a stated requirement and placed at a gap the data actually
has rather than at a round number.

TWO DECISIONS, NOT ONE. A scene can be unusable for area and volume while its
outer shoreline is still a valid geometric constraint, and the reverse:

    shoreline_constraint_usable   may this scene's external shoreline be used
                                  as an elevation constraint at its own stage
    area_volume_contributor       may this scene feed the per-stage A(H) / V(H)
                                  multi-orbit composite (a single S1 swath never
                                  spans the zone, so this is never a per-scene
                                  verdict on the curve itself)

ROLES, NOT DELETION. Seven scenes observe 1.8-6.7% of the domain and see 26-128
km of shoreline against 459-1932 km for the rest -- six of the seven are relative
orbit 87, a swath that barely clips the reservoir. They are not bad scenes, they
are small ones. They are admitted with constraint_scope = LOCAL: usable as
elevation constraints where they do observe, never as the basis of a
reservoir-wide statistic. Discarding a valid acquisition for being small would
throw away real measurements.

NMAD IS NOT AN ADMISSION CRITERION, and this is the most consequential call
here. The NMAD distribution is bimodal -- 15 scenes below 21 m, 11 above 25 --
but the split is a TARGET effect, not a scene-quality effect: 9 of the 11 high
scenes are H3 (median 30.1 m, against 15.3 for H1 and 17.9 for H2), because H3
is the lowest contour with 1,855 island rings and 29% of its boundary on them.
Cutting on NMAD would remove the lowest stage almost entirely and bias the
hypsometry of the DEM that these constraints exist to build. NMAD therefore
propagates into the per-constraint horizontal uncertainty, where it belongs, and
never into a yes/no.

NO MEDIAN ANYWHERE. Per 12_GATE_7C_ROLE_FREEZE.md, median_*_vs_raster_ref is
absolute position against a 20 m raster staircase and is not admissible for
anything. Only spreads, differences and counts are used below.

Outputs
-------
outputs/tables/gate7c2b_admission.csv
outputs/figures/historical_bathymetry/png/gate7c2b_admission.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CELL = 20.0
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"

# Each cut: (column, comparison, value, the gap it sits in, the requirement).
# A threshold with no requirement behind it is a preference, not a criterion.
CUTS = [
    ("f_L_100", ">=", 0.60, "0.492 -> 0.702",
     "below this the paired shoreline is a minority of the eligible line, so "
     "the statistic describes a selected fragment rather than the shoreline"),
    ("median_drift", "<=", 6.0, "5.503 -> 7.245",
     "a median that moves more than ~0.3 cell as the match threshold is "
     "relaxed is being selected by the threshold, not measured"),
    ("abs_bidirectional", "<=", 12.0, "9.913 -> 16.441",
     "the two matching directions must agree to better than ~0.6 cell, or the "
     "correspondence is direction-dependent and neither direction is trusted"),
]
SCOPE_OBS = 0.35          # gap 0.067 -> 0.699; basin-wide vs local
AREA_OBS = 0.95           # A(H) needs the basin observed, not most of it
AREA_LARGEST = 0.99       # one water body, not a fragmented one
AREA_RINT = 0.50          # interior boundary must not rival the shoreline


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("GATE 7C2b — admission cut")
    print("=" * 78)
    print(f"  git {commit}")

    Q = pd.read_csv(CFG.TABLES / "gate7c2_scene_quality.csv")
    Q = Q[Q.event_id.notna()].copy()
    Q["abs_bidirectional"] = Q.bidirectional_difference.abs()

    print(f"\n  {len(Q)} scenes; no median_*_vs_raster_ref column is used")
    print("\n  CUTS")
    ok = pd.Series(True, index=Q.index)
    for col, op, val, gap, why in CUTS:
        keep = Q[col] <= val if op == "<=" else Q[col] >= val
        keep = keep.fillna(False)
        print(f"    {col:20s} {op} {val:<6.2f}  gap {gap:<18s} "
              f"drops {int((~keep).sum())}")
        print(f"      {why}")
        ok &= keep
    Q["shoreline_constraint_usable"] = ok

    Q["constraint_scope"] = np.where(Q.obs_fraction >= SCOPE_OBS,
                                     "BASIN", "LOCAL")
    # A per-SCENE area_volume flag was misleading: only 2 of 26 scenes reach
    # 95% coverage on their own and H1 had none, which reads as "no area curve
    # is possible" when the truth is that a single S1 swath does not span an
    # 11,412 km2 zone. The flag is therefore what a scene can CONTRIBUTE to a
    # multi-orbit composite; whether the composite closes is a per-STAGE
    # question, answered by union_coverage().
    Q["area_volume_contributor"] = (
        Q.shoreline_constraint_usable
        & (Q.constraint_scope == "BASIN")
        & (Q.core_largest_fraction >= AREA_LARGEST)
        & (Q.internal_external_ratio <= AREA_RINT)).fillna(False)

    # NMAD becomes an uncertainty, never a gate. The shoreline NMAD is a
    # horizontal scatter; the elevation uncertainty it implies needs the local
    # migration rate, which Gate 7C3 left PROVISIONAL, so it is not computed
    # here -- only the horizontal term is carried forward.
    Q["sigma_x_shoreline_m"] = Q.nmad_100

    print(f"\n  shoreline_constraint_usable : {int(Q.shoreline_constraint_usable.sum())}"
          f" / {len(Q)}")
    print(f"    of which BASIN scope       : "
          f"{int((Q.shoreline_constraint_usable & (Q.constraint_scope=='BASIN')).sum())}")
    print(f"    of which LOCAL scope       : "
          f"{int((Q.shoreline_constraint_usable & (Q.constraint_scope=='LOCAL')).sum())}")
    print(f"  area_volume_contributor     : "
          f"{int(Q.area_volume_contributor.sum())} / {len(Q)}")
    print("    (a single S1 swath never spans the zone; A(H) is a per-stage")
    print("     multi-orbit composite, closed below, not a per-scene property)")

    print("\n  by target (the cut must not empty a stage):")
    t = Q.groupby("target").agg(
        n=("event_id", "size"),
        shoreline_ok=("shoreline_constraint_usable", "sum"),
        basin=("constraint_scope", lambda s: int((s == "BASIN").sum())),
        area_contrib=("area_volume_contributor", "sum"),
        sigma_x_median=("sigma_x_shoreline_m", "median"))
    print(t.round(1).to_string())
    if (t.shoreline_ok == 0).any():
        raise SystemExit("a target stage has no admitted shoreline: the cut "
                         "would bias the hypsometry and must be revisited")

    union_coverage(Q)

    print("\n  rejected scenes and why:")
    for r in Q[~Q.shoreline_constraint_usable].itertuples():
        why = []
        for col, op, val, _, _ in CUTS:
            v = getattr(r, col)
            bad = (v > val) if op == "<=" else (v < val)
            if pd.isna(v) or bad:
                why.append(f"{col}={v:.3f}")
        print(f"    {r.date} {r.target} orb{r.relative_orbit:<4d} "
              f"{', '.join(why)}")

    keep_cols = ["event_id", "date", "target", "relative_orbit",
                 "shoreline_constraint_usable", "constraint_scope",
                 "area_volume_contributor", "sigma_x_shoreline_m",
                 "f_L_100", "median_drift", "abs_bidirectional",
                 "obs_fraction", "core_largest_fraction",
                 "internal_external_ratio", "lda_separation"]
    out = Q[[c for c in keep_cols if c in Q.columns]]
    out.to_csv(CFG.TABLES / "gate7c2b_admission.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'gate7c2b_admission.csv'}")
    figure(Q)
    print("\nSTOP. Next on the critical path: rebuild H1/H2/H3 as continuous "
          "contours.")


def union_coverage(Q):
    """Does the UNION of admitted scenes at each stage observe the basin?

    Only 2 of 26 single scenes reach 95% coverage, and H1 has none, so A(H) and
    V(H) cannot come from single events. That is not a failure of the cut, it is
    a statement about Sentinel-1 swath geometry over an 11,412 km2 zone: the
    reservoir needs several relative orbits to be seen whole. The decision this
    forces is architectural -- area and volume must be built from a multi-orbit
    composite per stage, with the per-orbit disagreement in the overlaps
    reported, not from whichever single scene happens to be widest."""
    from shapely.ops import unary_union
    from swot_dnipro import spatial_domains as SD
    from hist25b_gate7_anchored_classifier import CACHE, build_grid
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    print("\n  UNION coverage per stage, admitted scenes only:")
    for tgt, grp in Q[Q.shoreline_constraint_usable].groupby("target"):
        u = np.zeros(inside.shape, bool)
        orbs, n = [], 0
        for r in grp.itertuples():
            p = CACHE / f"{r.event_id}.npz"
            if not p.exists():
                continue
            u |= np.load(p)["cov"]
            orbs.append(int(r.relative_orbit)); n += 1
        cov = float((u & inside).sum() / inside.sum())
        print(f"    {tgt}: {n} scenes over orbits {sorted(set(orbs))} "
              f"-> union covers {100*cov:.1f}% of the domain "
              f"{'OK for A(H)' if cov >= AREA_OBS else 'STILL SHORT'}")


def figure(Q):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    cmap = {"H1": AMBER, "H2": GREEN, "H3": BLUE}
    c = Q.target.map(cmap)
    m = Q.shoreline_constraint_usable
    for a, (x, y, xl, yl, vx, vy) in zip(ax, [
            ("f_L_100", "median_drift", "f_L at 100 m",
             "median drift over 25-250 m (m)", 0.60, 6.0),
            ("f_L_100", "abs_bidirectional", "f_L at 100 m",
             "|bidirectional difference| (m)", 0.60, 12.0),
            ("obs_fraction", "sigma_x_shoreline_m", "observed fraction of domain",
             "sigma_x from NMAD (m) - NOT a cut", SCOPE_OBS, None)]):
        a.scatter(Q.loc[m, x], Q.loc[m, y], c=c[m], s=80, edgecolor=INK,
                  linewidth=.6, label="admitted")
        a.scatter(Q.loc[~m, x], Q.loc[~m, y], c="none", s=100,
                  edgecolor=RED, linewidth=1.6, label="rejected")
        if vx is not None:
            a.axvline(vx, color=INK, ls="--", lw=1)
        if vy is not None:
            a.axhline(vy, color=INK, ls="--", lw=1)
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3)
    ax[2].set_title("NMAD is a target effect (H3 high), so it sets sigma_x\n"
                    "and never admission", color=INK, fontsize=10)
    ax[0].set_title("stability vs paired length", color=INK, fontsize=10)
    ax[1].set_title("direction agreement vs paired length", color=INK,
                    fontsize=10)
    h = [plt.Line2D([], [], marker="o", ls="", color=v, label=k)
         for k, v in cmap.items()]
    h.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=RED,
                        label="rejected"))
    ax[0].legend(handles=h, fontsize=8)
    fig.suptitle("Gate 7C2b · the admission cut, placed at gaps the data has",
                 color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "gate7c2b_admission.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
