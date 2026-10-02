#!/usr/bin/env python
"""MS14 -- Figure 3 (F02_slope_per_overpass.png), per-overpass slope, from its table.

The figure was carried from earlier work without a script; its numbers matched the table
(medians +0.090 / +3.314 cm/km, 14 of 33 intervals beyond the axis, widest 225 cm/km) but its
panel note pointed at a section that no longer exists and its legend covered the breach label.
This draws it from outputs/figure_data/FigG_perdate_slopes_robust.csv (33 overpasses, span >= 20 km):
(a) every overpass with its own Theil-Sen interval on a bounded axis, the number of intervals
reaching beyond it printed on the panel; (b) the pre and post distributions with their medians.

Outputs  outputs/paper/figures/F02_slope_per_overpass.png / .pdf
         outputs/paper/validation/ms14_slope_figure_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/figure_data/FigG_perdate_slopes_robust.csv"
FIG = ROOT / "outputs/paper/figures"
VAL = ROOT / "outputs/paper/validation"
YLIM = (-20, 30)
BREACH = pd.Timestamp("2023-06-06")
COL = {"PRE_BREACH": "#1f6f8b", "BREACH_DRAWDOWN": "#b5651d", "POST_BREACH": "#2e7d32"}
LAB = {"PRE_BREACH": "before the breach", "BREACH_DRAWDOWN": "2023 drawdown", "POST_BREACH": "after the breach"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = pd.read_csv(SRC, parse_dates=["date"])
    beyond = int(((g.ts_lo_cm_km < YLIM[0]) | (g.ts_hi_cm_km > YLIM[1])).sum())
    widest = float((g.ts_hi_cm_km - g.ts_lo_cm_km).max())
    med = g.groupby("period").slope_theilsen_cm_km.median()
    pos = g[g.period == "POST_BREACH"].slope_theilsen_cm_km.gt(0).sum()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200, gridspec_kw=dict(width_ratios=[2.4, 1]))
    for per in ("PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"):
        s = g[g.period == per]
        lo = np.clip(s.ts_lo_cm_km, *YLIM)
        hi = np.clip(s.ts_hi_cm_km, *YLIM)
        ax.vlines(s.date, lo, hi, color=COL[per], lw=1, alpha=0.7)
        ax.scatter(s.date, s.slope_theilsen_cm_km, s=22, color=COL[per], zorder=3,
                   label=f"{LAB[per]} (n = {len(s)})")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.axvline(BREACH, color="0.5", ls="--", lw=1)
    ax.text(BREACH, YLIM[1] * 0.97, " breach", color="0.4", fontsize=8, va="top")
    ax.set_ylim(*YLIM)
    ax.set_ylabel("longitudinal slope (cm/km)")
    ax.set_title("a · every overpass, with its own Theil–Sen interval", loc="left", fontsize=10)
    ax.text(0.01, 0.02, f"{beyond} of {len(g)} intervals extend beyond the axis (widest {widest:.0f} cm/km)",
            transform=ax.transAxes, fontsize=7.5, color="0.4")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 0.93))

    rng = np.random.default_rng(0)
    for i, per in enumerate(("PRE_BREACH", "POST_BREACH")):
        v = g[g.period == per].slope_theilsen_cm_km.values
        bx.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=COL[per], alpha=0.25, lw=0.8), medianprops=dict(color=COL[per], lw=2),
                   whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
        bx.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COL[per], zorder=3)
        bx.text(i + 0.3, med[per], f"{med[per]:+.2f}", color=COL[per], fontsize=8, va="center")
    bx.axhline(0, color="0.2", lw=0.8)
    bx.set_xticks([0, 1], ["before", "after"])
    bx.set_ylabel("cm/km")
    bx.set_title(f"b · the distributions\n({pos} of 14 after the breach positive)", loc="left", fontsize=10)
    for z in (ax, bx):
        z.grid(axis="y", color="0.92")
        for s_ in ("top", "right"):
            z.spines[s_].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F02_slope_per_overpass.{ext}")
    pd.DataFrame([dict(statistic="median_pre_cm_km", value=med["PRE_BREACH"]),
                  dict(statistic="median_post_cm_km", value=med["POST_BREACH"]),
                  dict(statistic="intervals_beyond_axis", value=beyond),
                  dict(statistic="widest_interval_cm_km", value=widest),
                  dict(statistic="post_positive", value=int(pos))]).to_csv(VAL / "ms14_slope_figure_summary.csv", index=False)
    print(f"medians {med['PRE_BREACH']:+.3f} / {med['POST_BREACH']:+.3f}; beyond {beyond}/{len(g)}; widest {widest:.0f}; post positive {pos}/14")


if __name__ == "__main__":
    main()
