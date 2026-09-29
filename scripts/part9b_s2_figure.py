#!/usr/bin/env python
"""V8 — Sentinel-2 spectral indices and surface classification of the former pool."""
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
CCOL = {"water": BLUE, "vegetation": GREEN, "sparse_veg": "#9bbf7a", "bare": AMBER}


def main() -> None:
    d = pd.read_csv(CFG.TABLES / "s2_spectral_class_by_date.csv", parse_dates=["date"])
    st = pd.read_csv(CFG.TABLES / "s2_index_stats_by_class.csv")
    sw = pd.read_csv(CFG.TABLES / "s2_ndvi_threshold_sensitivity.csv", parse_dates=["date"])
    d["month"], d["year"] = d.date.dt.month, d.date.dt.year
    fp = d[d.zone == "former_pool"]
    out = d[d.zone == "outside_pool"]

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.40, wspace=0.30)

    # (a) composition by period
    ax = fig.add_subplot(gs[0, 0])
    order = ["PRE_BREACH", "DRAWDOWN", "POST_BREACH"]
    comp = fp.groupby("period")[[f"frac_{c}" for c in CCOL]].median().reindex(order)
    bottom = np.zeros(len(comp))
    for c in CCOL:
        v = comp[f"frac_{c}"].values
        ax.bar(range(len(comp)), v, bottom=bottom, color=CCOL[c], label=c, width=0.62)
        for i, (b, h) in enumerate(zip(bottom, v)):
            if h > 0.06:
                ax.text(i, b + h / 2, f"{h*100:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += v
    ax.set_xticks(range(len(comp)))
    ax.set_xticklabels(["pre-breach", "drawdown", "post-breach"], fontsize=9)
    ax.set_ylabel("fraction of the former pool"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_title("(a) surface composition of the former pool", fontsize=11, loc="left")

    # (b) spectral separation
    ax2 = fig.add_subplot(gs[0, 1])
    sub = st[st.zone == "former_pool"]
    for k, c in CCOL.items():
        g = sub[sub["class"] == k]
        if not len(g):
            continue
        ax2.scatter(g.mndwi_median, g.ndvi_median, s=26, color=c, alpha=0.7,
                    lw=0, label=k)
    ax2.axvline(0, color=INK, lw=1, ls=":"); ax2.axhline(0, color=INK, lw=1, ls=":")
    ax2.set_xlabel("MNDWI median"); ax2.set_ylabel("NDVI median")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.22)
    ax2.set_title("(b) the classes separate spectrally", fontsize=11, loc="left")

    # (c) the control: inside vs outside, peak season only
    ax3 = fig.add_subplot(gs[0, 2])
    ps_in = fp[(fp.period == "POST_BREACH") & fp.month.between(5, 9)]
    ps_out = out[(out.period == "POST_BREACH") & out.month.between(5, 9)]
    a = ps_in.groupby("year").frac_vegetation.median()
    b = ps_out.groupby("year").frac_vegetation.median()
    na = ps_in.groupby("year").size(); nb = ps_out.groupby("year").size()
    ax3.plot(a.index, a.values * 100, "o-", ms=10, lw=2.6, color=GREEN,
             label="former pool (exposed bed)")
    ax3.plot(b.index, b.values * 100, "s--", ms=9, lw=2.2, color=AMBER,
             label="surrounding steppe (control)")
    for x, y, n in zip(a.index, a.values * 100, na.values):
        ax3.annotate(f"{y:.0f}%\nn={n}", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8.4, color=GREEN)
    for x, y, n in zip(b.index, b.values * 100, nb.values):
        ax3.annotate(f"{y:.0f}%\nn={n}", (x, y), textcoords="offset points",
                     xytext=(0, -26), ha="center", fontsize=8.4, color="#8a5a12")
    ax3.set_xticks(a.index); ax3.set_ylim(-4, 72)
    ax3.set_ylabel("% vegetated (NDVI ≥ 0.30)")
    ax3.legend(fontsize=8.4, loc="center right"); ax3.grid(alpha=0.22)
    ax3.set_title("(c) peak season only (May–Sep)", fontsize=11, loc="left")

    # (d) water fraction over time
    ax4 = fig.add_subplot(gs[1, :2])
    for per, c in (("PRE_BREACH", BLUE), ("DRAWDOWN", AMBER), ("POST_BREACH", RED)):
        g = fp[fp.period == per]
        ax4.scatter(g.date, g.frac_water * 100, s=48, color=c, alpha=0.8,
                    edgecolor="white", lw=0.8, label=per.replace("_", "-").lower())
        ax4.scatter(g.date, g.frac_vegetation * 100, s=30, color=GREEN, alpha=0.55,
                    marker="^", lw=0)
    ax4.axvline(pd.Timestamp("2023-06-06"), color=RED, ls="--", lw=1.6)
    ax4.text(pd.Timestamp("2023-06-10"), 92, "breach", color=RED, fontsize=9)
    ax4.plot([], [], "^", color=GREEN, label="vegetated (NDVI ≥ 0.30)")
    ax4.set_ylabel("% of the former pool"); ax4.set_ylim(-3, 100)
    ax4.legend(fontsize=8.4, ncol=4, loc="upper right"); ax4.grid(alpha=0.22)
    ax4.set_title("(d) water gives way to vegetation inside the former pool",
                  fontsize=11, loc="left")

    # (e) NDVI threshold sensitivity
    ax5 = fig.add_subplot(gs[1, 2])
    sp = sw[(sw.period == "POST_BREACH")].copy()
    sp["year"] = sp.date.dt.year
    sp = sp[sp.date.dt.month.between(5, 9)]
    piv = sp.pivot_table(index="ndvi_threshold", columns="year",
                         values="frac_vegetated_nonwater", aggfunc="median")
    for y in piv.columns:
        ax5.plot(piv.index, piv[y] * 100, "o-", ms=6, label=str(y))
    ax5.set_xlabel("NDVI threshold"); ax5.set_ylabel("% non-water vegetated")
    ax5.legend(fontsize=8.4, title="year", title_fontsize=8)
    ax5.grid(alpha=0.22)
    ax5.set_title("(e) threshold sensitivity", fontsize=11, loc="left")

    fig.suptitle("V8 · Sentinel-2 NDWI / MNDWI / NDVI and the surface classification "
                 "of the former Kakhovka pool", fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "The established water mask (NDWI>0 ∧ MNDWI>0 ∧ SCL) is used unchanged; NDVI only "
             "splits the non-water remainder. Panel (c) is the control that matters: the exposed "
             "bed greens while the surrounding steppe browns over the same months, so the trend "
             "is not regional phenology. 2023 is sampled by 3 September scenes only.",
             ha="center", fontsize=8.6, color=GREY)
    for ext in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V8_s2_spectral_classification.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/'V8_s2_spectral_classification.png'}")


if __name__ == "__main__":
    main()
