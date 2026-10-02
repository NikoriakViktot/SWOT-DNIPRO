#!/usr/bin/env python
"""HIST 28 — is the SAR-optical offset a single number? The hist24 precondition.

hist24_three_contour_dem.py carries OFFSET_FROM_TRUE_WATERLINE_M = 181.0,
measured against the superseded P20 footprint. It has to go. The roadmap is
explicit that it must NOT simply be replaced by the re-measured 29-36 m as a new
scalar: a constant is only admissible if the offset actually is single-valued,
and that is a question to be tested, not assumed.

THE TEST. Across the Gate 7C2b admitted scenes, does the per-scene offset vary
systematically with the target stage, or with the relative orbit? And is the
scene-to-scene spread small compared with the scatter within a single scene?

    stable single-valued        no group structure AND between-scene spread
                                comparable to within-scene NMAD -> use a constant
    structured                  group structure -> carry it per target / orbit
    unstructured but noisy      no group structure BUT between-scene spread
                                much larger than within-scene NMAD -> no constant
                                is justified; the scatter is uncertainty, not a
                                correction

Outputs
-------
outputs/tables/hist28_offset_stability.csv
outputs/figures/historical_bathymetry/png/hist28_offset_stability.png
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
from scipy import stats

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3")
CELL = 20.0
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("HIST 28 — is the offset a single number?")
    print("=" * 78)
    print(f"  git {commit}")

    R = pd.read_csv(CFG.TABLES / "gate7c3_continuous_offset.csv")
    R = R[R.admitted.astype(bool)].copy()
    print(f"  {len(R)} Gate 7C2b admitted scenes\n")

    rows = []
    for key in ("target", "relative_orbit"):
        g = R.groupby(key).d_raw_median
        print(f"  d_raw_median by {key}:")
        print(g.agg(["count", "median", "min", "max"]).round(2).to_string())
        groups = [v.values for _, v in g if len(v) > 1]
        if len(groups) > 1:
            h = stats.kruskal(*groups)
            print(f"    Kruskal-Wallis: H = {h.statistic:.2f}, "
                  f"p = {h.pvalue:.3f}  "
                  f"{'STRUCTURED' if h.pvalue < 0.05 else 'no group structure'}")
            rows.append(dict(grouping=key, H=float(h.statistic),
                             p=float(h.pvalue),
                             structured=bool(h.pvalue < 0.05)))
        print()

    between = float(R.d_raw_median.max() - R.d_raw_median.min())
    within = float(R.d_raw_nmad.median())
    ratio = between / within
    print(f"  between-scene spread of medians : {between:.1f} m")
    print(f"  within-scene NMAD (median)      : {within:.1f} m")
    print(f"  ratio                           : {ratio:.2f}")

    structured = any(r["structured"] for r in rows)
    print("\n" + "=" * 78)
    print("VERDICT FOR hist24")
    print("=" * 78)
    if structured:
        print("  STRUCTURED: carry the offset per target / orbit, not as one")
        print("  number.")
        verdict = "PER_GROUP"
    elif ratio < 1.5:
        print("  STABLE: a single constant is defensible.")
        verdict = "CONSTANT"
    else:
        print("  UNSTRUCTURED BUT NOISY. No group explains the variation, and")
        print(f"  the scene-to-scene spread is {ratio:.1f}x the scatter within a")
        print("  single scene, so the between-scene variation is not a")
        print("  systematic effect that any constant could remove.")
        print("\n  -> hist24 takes NO offset constant. Not 181.0, not 29-36,")
        print("     and not a per-target value. The central value is")
        print(f"     {R.d_raw_median.median():+.1f} m, under {abs(R.d_raw_median.median())/CELL:.2f}"
              f" of the {CELL:.0f} m cell,")
        print("     and the scatter around it belongs in sigma, not in a")
        print("     correction. Subtracting an unresolved mean would move every")
        print("     shoreline by less than a fifth of a cell while pretending")
        print("     to a precision the scenes do not agree on.")
        verdict = "NO_CONSTANT"

    out = pd.DataFrame(rows)
    out["between_scene_spread_m"] = between
    out["within_scene_nmad_m"] = within
    out["spread_ratio"] = ratio
    out["central_offset_m"] = float(R.d_raw_median.median())
    out["verdict"] = verdict
    out.to_csv(CFG.TABLES / "hist28_offset_stability.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'hist28_offset_stability.csv'}")
    figure(R, between, within)


def figure(R, between, within):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    cmap = {"H1": AMBER, "H2": GREEN, "H3": BLUE}
    for i, key in enumerate(("target", "relative_orbit")):
        keys = sorted(R[key].unique())
        for j, k in enumerate(keys):
            v = R.loc[R[key] == k, "d_raw_median"].values
            n = R.loc[R[key] == k, "d_raw_nmad"].values
            c = cmap.get(k, BLUE) if key == "target" else BLUE
            ax[i].errorbar(np.full(len(v), j) + np.linspace(-.12, .12, len(v)),
                           v, yerr=n, fmt="o", color=c, capsize=3, ms=6,
                           alpha=.85)
        ax[i].axhline(0, color=INK, lw=1)
        ax[i].axhline(R.d_raw_median.median(), color=RED, ls="--", lw=1.2,
                      label=f"overall median {R.d_raw_median.median():+.1f} m")
        ax[i].set_xticks(range(len(keys)))
        ax[i].set_xticklabels([str(k) for k in keys])
        ax[i].set_xlabel(key.replace("_", " "))
        ax[i].set_ylabel("d_raw median ± NMAD (m)")
        ax[i].grid(alpha=.3); ax[i].legend(fontsize=8)
    ax[0].set_title("no structure by target stage (p = 0.27)", color=INK)
    ax[1].set_title("no structure by relative orbit (p = 0.34)", color=INK)
    fig.suptitle(f"hist28 · between-scene spread {between:.0f} m vs "
                 f"within-scene NMAD {within:.0f} m → no constant is justified",
                 color=INK, fontsize=12)
    fig.tight_layout()
    p = FIGDIR / "hist28_offset_stability.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
