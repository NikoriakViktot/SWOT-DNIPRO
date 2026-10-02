#!/usr/bin/env python
"""V7 — what ATL03/ATL08 adds: an independent check of the vertical chain, and
the bare-earth surface of the drained lakebed."""
from __future__ import annotations

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

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
PRE_WSE, POST_WSE = 15.96, 5.20          # ATL13 branch, independent of ATL08


def main() -> None:
    d = pd.read_parquet(ROOT / "data/processed/atl08/kakhovka_atl08_terrain.parquet")
    bed = d[d.surface_class == "exposed_bed"]
    pool = d[d.surface_class == "pool_water_surface"]
    land = d[d.surface_class == "outside_pool_land"]

    fig = plt.figure(figsize=(14.5, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.36, wspace=0.28)

    # ---- (a) elevation distributions ------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    bins = np.arange(-2, 100, 1.0)
    ax.hist(land.H_terrain_common_m, bins=bins, color=GREY, alpha=0.55,
            label=f"land outside the pool  (n={len(land):,})")
    ax.hist(bed.H_terrain_common_m, bins=bins, color=AMBER, alpha=0.85,
            label=f"exposed lakebed, post-breach  (n={len(bed):,})")
    ax.hist(pool.H_terrain_common_m, bins=bins, color=BLUE, alpha=0.85,
            label=f"filled pool surface, pre-breach  (n={len(pool):,})")
    ax.axvline(PRE_WSE, color=BLUE, ls="--", lw=1.6)
    ax.axvline(POST_WSE, color=RED, ls="--", lw=1.6)
    ax.set_yscale("log")
    ytop = ax.get_ylim()[1]
    ax.text(PRE_WSE + 1.2, ytop * 0.06, f"ATL13 pre-breach\npool {PRE_WSE:.2f} m",
            color=BLUE, fontsize=8.6, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.4))
    ax.text(POST_WSE - 1.2, ytop * 0.06, f"ATL13 post-breach\nwater {POST_WSE:.2f} m",
            color=RED, fontsize=8.6, ha="right", va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.4))
    ax.set_xlim(-2, 100)
    ax.set_xlabel("terrain height  (m, EGG2015 common frame)")
    ax.set_ylabel("ATL08 segments (log)")
    ax.legend(fontsize=8.6, loc="upper right"); ax.grid(alpha=0.22)
    ax.set_title("(a) ATL08 terrain by surface class", fontsize=11, loc="left")

    # ---- (b) the cross-check --------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    m = pool.H_terrain_common_m.median()
    ax2.hist(pool.H_terrain_common_m, bins=np.arange(13, 19, 0.05), color=BLUE, alpha=0.8)
    ax2.axvline(PRE_WSE, color=RED, ls="--", lw=2)
    ax2.axvline(m, color=INK, lw=1.4)
    ax2.set_xlim(13.5, 18.5)
    ax2.set_xlabel("m, common frame"); ax2.set_ylabel("segments")
    ax2.grid(alpha=0.22)
    ax2.set_title("(b) independent cross-check", fontsize=11, loc="left")
    ax2.text(0.03, 0.60, f"ATL08 over the filled pool\nmedian {m:.3f} m\n"
                         f"ATL13 water product  {PRE_WSE:.2f} m\n"
                         f"difference  {m - PRE_WSE:+.3f} m",
             transform=ax2.transAxes, va="top", fontsize=8.6,
             bbox=dict(fc="white", ec=GREY, alpha=0.94))

    # ---- (c) bed hypsometry ---------------------------------------------
    ax3 = fig.add_subplot(gs[1, :2])
    v = np.sort(bed.H_terrain_common_m.values)
    frac = np.arange(1, len(v) + 1) / len(v) * 100
    ax3.plot(v, frac, "-", color=AMBER, lw=2.4)
    ax3.axvline(POST_WSE, color=RED, ls="--", lw=1.8)
    ax3.axvline(PRE_WSE, color=BLUE, ls="--", lw=1.4)
    above = (bed.H_terrain_common_m > POST_WSE).mean() * 100
    ax3.axhspan(100 - above, 100, color=AMBER, alpha=0.16)
    ax3.text(POST_WSE + 0.6, 50, f"{above:.0f} % of the exposed bed lies ABOVE\n"
                                 f"the post-breach water level",
             fontsize=10, color="#8a5a12", fontweight="bold")
    ax3.set_xlim(-1, 22); ax3.set_ylim(0, 100)
    ax3.set_xlabel("bed elevation  (m, common frame)")
    ax3.set_ylabel("% of bed below this level")
    ax3.grid(alpha=0.22)
    ax3.set_title("(c) why the water fragments: bed hypsometry vs the residual water level",
                  fontsize=11, loc="left")

    # ---- (d) vegetation on the bed ---------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    yr = bed.groupby(bed.dt.dt.year).h_canopy.apply(lambda s: (s > 0).mean() * 100)
    n = bed.groupby(bed.dt.dt.year).size()
    ax4.bar(yr.index.astype(str), yr.values, color=GREEN, alpha=0.85, width=0.6)
    for x, (y, nn) in enumerate(zip(yr.values, n.values)):
        ax4.text(x, y + 0.4, f"{y:.1f}%\nn={nn:,}", ha="center", fontsize=8.4)
    ax4.set_ylim(0, max(yr.values) * 1.35)
    ax4.set_ylabel("% of bed segments with canopy > 0")
    ax4.grid(axis="y", alpha=0.22)
    ax4.set_title("(d) vegetation colonising the bed", fontsize=11, loc="left")

    fig.suptitle("V7 · ATL03/ATL08 (SlideRule PhoREAL): the bare-earth surface of the "
                 "drained Kakhovka lakebed", fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "ATL08 classifies ATL03 photons into ground / canopy / top-of-canopy. "
             "Panel (b) is a genuine independent check: ATL08 land processing over the "
             "pre-breach pool reproduces the ATL13 water product to 11 mm. "
             "Panel (d) trend is more trustworthy than the canopy heights — see caveats.",
             ha="center", fontsize=8.6, color=GREY)
    for ext in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V7_atl08_lakebed.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/'V7_atl08_lakebed.png'}")

    pd.DataFrame({
        "quantity": ["ATL08 pool surface median", "ATL13 pre-breach pool", "difference",
                     "bed p05", "bed p25", "bed median", "bed p75", "bed p95",
                     "post-breach water (ATL13)", "% bed above post-breach water"],
        "value_m": [m, PRE_WSE, m - PRE_WSE,
                    *[np.percentile(bed.H_terrain_common_m, q) for q in (5, 25, 50, 75, 95)],
                    POST_WSE, above]}).to_csv(CFG.FIGDATA / "V7_atl08_lakebed.csv", index=False)


if __name__ == "__main__":
    main()
