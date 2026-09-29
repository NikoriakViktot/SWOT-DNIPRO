#!/usr/bin/env python
"""FigG — the reservoir->river transition, four panels."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import C, panel_label, save, use_style, zero_line
use_style()
FD, FIG = CFG.FIGDATA, CFG.FIG
PC = {"PRE_BREACH": "#1b6ca8", "BREACH_DRAWDOWN": "#e08214", "POST_BREACH": "#b03a2e"}


def main():
    pts = pd.read_csv(FD / "FigD_kakhovka_profile_points.csv")
    sl = pd.read_csv(FD / "FigG_perdate_slopes_robust.csv")
    st = pd.read_csv(FD / "FigG_transition_statistics.csv")
    ts = st[st.estimator == "Theil-Sen"].iloc[0]

    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.26)

    # --- A/B/C: representative profiles per regime -------------------------
    for j, (per, ttl) in enumerate([("PRE_BREACH", "PRE-BREACH"),
                                    ("BREACH_DRAWDOWN", "DRAWDOWN"),
                                    ("POST_BREACH", "POST-BREACH")]):
        ax = fig.add_subplot(gs[0, j])
        s = sl[sl.period == per].sort_values("span_km", ascending=False).head(4)
        for r in s.itertuples():
            g = pts[(pts.date == r.date) & (pts.period == per)].sort_values("chain_km")
            y = g.wse_m - np.median(g.wse_m)
            ax.scatter(g.chain_km, y, s=22, alpha=0.85, ec="black", lw=0.3, zorder=4,
                       label=f"{r.date}  {r.slope_theilsen_cm_km:+.2f}")
            xs = np.array([g.chain_km.min(), g.chain_km.max()])
            ax.plot(xs, (r.slope_theilsen_cm_km / 100) * (xs - np.median(g.chain_km)),
                    lw=1.1, alpha=0.65, zorder=3,
                    color=ax.collections[-1].get_facecolor()[0])
        zero_line(ax, 0)
        ax.set_title(f"{ttl}\nmedian Theil-Sen "
                     f"{np.median(sl[sl.period==per].slope_theilsen_cm_km):+.2f} cm/km",
                     color=PC[per], fontsize=8.5)
        ax.set_xlabel("chainage from dam [km]", fontsize=7.5)
        if j == 0: ax.set_ylabel("WSE − date median  [m]")
        ax.set_ylim(-2.6, 2.6); ax.set_xlim(60, 285)
        ax.legend(fontsize=5.4, loc="upper left", title="date · cm/km", title_fontsize=5.6)
        panel_label(ax, "abc"[j])
    fig.text(0.5, 0.985, "one point = one ICESat-2 beam-pass · one fitted slope per date "
             "(the independent unit) · residual ponds sit above the channel at the same "
             "chainage, so vertical scatter grows ~20× after the breach",
             ha="center", fontsize=6.6, color="#5d6d7e", style="italic")

    # --- D: slope time series ----------------------------------------------
    ax = fig.add_subplot(gs[1, :2])
    sl["dt"] = pd.to_datetime(sl.date)
    for per in PC:
        s = sl[sl.period == per]
        ax.scatter(s.dt, s.slope_theilsen_cm_km, s=52, c=PC[per], ec="black", lw=0.6,
                   zorder=5, label=f"{per.replace('_',' ').title()} (n={len(s)})")
    br = pd.Timestamp(CFG.BREACH_DATE)
    ax.axvline(br, color=C["bad"], lw=2.0, zorder=3)
    ax.annotate("dam breach\n2023-06-06", (br, -1.6), xytext=(-8, 0),
                textcoords="offset points", fontsize=7.4, color=C["bad"],
                fontweight="bold", va="center", ha="right")
    zero_line(ax, 0)
    for per, x0, x1 in [("PRE_BREACH", sl.dt.min(), br),
                        ("POST_BREACH", pd.Timestamp("2023-09-01"), sl.dt.max())]:
        m = np.median(sl[sl.period == per].slope_theilsen_cm_km)
        ax.plot([x0, x1], [m, m], color=PC[per], lw=1.8, ls="--", alpha=0.85, zorder=4)
        ax.annotate(f"median {m:+.2f}", (x1, m), xytext=(-4, 6), textcoords="offset points",
                    ha="right", fontsize=7, color=PC[per], fontweight="bold")
    ax.set_ylabel("longitudinal WSE slope, Theil-Sen  [cm/km]")
    ax.set_xlabel("date")
    ax.set_title("Longitudinal slope of the former Kakhovka Reservoir, 2018–2025")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.set_ylim(-3, 12)
    panel_label(ax, "d")

    # --- E: distribution + tests -------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    order = ["PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"]
    rng = np.random.default_rng(CFG.SEED)
    for i, per in enumerate(order):
        v = sl[sl.period == per].slope_theilsen_cm_km
        ax.scatter(np.full(len(v), i) + rng.normal(0, .07, len(v)), v, s=40, c=PC[per],
                   ec="black", lw=0.5, alpha=0.9, zorder=5)
        m = np.median(v)
        ax.plot([i - .3, i + .3], [m, m], color="black", lw=2.4, zorder=6)
        ax.annotate(f"{m:+.2f}", (i, m), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7.4, fontweight="bold")
    zero_line(ax, 0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["PRE", "DRAW", "POST"], fontsize=8)
    ax.set_ylabel("Theil-Sen slope [cm/km]")
    ax.set_title("Slope by regime")
    ax.set_ylim(-3, 13.5)
    ax.text(0.5, 0.80,
            f"Δ = {ts.diff_post_minus_pre_cm_km:+.2f} cm/km\n"
            f"95 % CI [{ts.diff_ci95_low:+.2f}, {ts.diff_ci95_high:+.2f}]\n"
            f"permutation p = {ts.permutation_p:.4f}\n"
            f"sign test: PRE {ts.pre_positive} (p={ts.pre_sign_test_p:.2f})\n"
            f"           POST {ts.post_positive} (p={ts.post_sign_test_p:.4f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c8cccf", lw=0.6))
    panel_label(ax, "e")

    fig.suptitle("Satellite altimetry records the hydraulic transformation of the former "
                 "Kakhovka Reservoir\nfrom an impounded pool to a river-dominated system",
                 y=1.02, fontsize=10.5)
    save(fig, "FigG_reservoir_to_river_transition", FIG)
    print("  FigG")


if __name__ == "__main__":
    main()
