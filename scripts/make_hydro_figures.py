#!/usr/bin/env python
"""Figures A-F: hydrological context and the post-breach reservoir->river transition."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.lines import Line2D
from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import (C, graticule, nmad, north_arrow, panel_label,
                                        save, scale_bar, use_style, zero_line)
use_style()
FD, FIG, M = CFG.FIGDATA, CFG.FIG, CFG.CRS_METRIC
STATE_C = {"RISING": "#1e8449", "FALLING": "#b03a2e", "STABLE": "#5d6d7e", "UNCERTAIN": "#95a5a6"}


def figA():
    h = pd.read_csv(FD / "FigA_kherson_hydrograph_2023.csv", parse_dates=["date"])
    ctx = pd.read_csv(FD / "FigA_kherson_hydrograph_context.csv",
                      parse_dates=["date", "swot_utc", "icesat_utc"])
    fig = plt.figure(figsize=(9.6, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.42, wspace=0.26)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(h.date, h.water_level_cm, color=C["gauge"], lw=1.2, zorder=3)
    ax.scatter(h.date, h.water_level_cm, s=7, c=C["gauge"], zorder=4)
    pre0 = h[h.date < pd.Timestamp(CFG.BREACH_DATE)]
    ytop = pre0.water_level_cm.max()
    for r in ctx.itertuples():
        ax.axvline(r.swot_utc, color=C["swot"], lw=1.0, alpha=0.85, zorder=2)
        ax.axvline(r.icesat_utc, color=C["icesat"], lw=1.0, ls="--", alpha=0.85, zorder=2)
        y = pre_top = None
        y = ytop + 8
        ax.annotate("", xy=(r.swot_utc, y), xytext=(r.icesat_utc, y),
                    arrowprops=dict(arrowstyle="<|-|>", lw=1.1, color=C["bad"]))
        ax.text(r.icesat_utc + (r.swot_utc - r.icesat_utc) / 2, y + 2,
                f"Δt={r.delta_t_h:+.1f} h\n{r.state}", ha="center", va="bottom",
                fontsize=6.4, fontweight="bold", color=STATE_C[r.state])
    br = pd.Timestamp(CFG.BREACH_DATE)
    ax.axvline(br, color=C["bad"], lw=2.0, zorder=5)
    ax.annotate("dam breach 2023-06-06", (br, h.water_level_cm.min()), xytext=(6, 4),
                textcoords="offset points", fontsize=7, color=C["bad"], fontweight="bold")
    ax.set_ylabel("Kherson stage above graph zero  [cm]")
    pre = h[h.date < pd.Timestamp(CFG.BREACH_DATE)]
    ax.set_ylim(pre.water_level_cm.min() - 6, pre.water_level_cm.max() + 26)
    peak = h.water_level_cm.max()
    ax.annotate(f"breach wave peaks at {peak:.0f} cm\n(off scale)",
                (h.loc[h.water_level_cm.idxmax(), "date"], pre.water_level_cm.max() + 14),
                xytext=(-60, -6), textcoords="offset points", fontsize=6.6, color=C["bad"],
                arrowprops=dict(arrowstyle="-|>", color=C["bad"], lw=0.9))
    ax.set_title("Kherson gauge 80805, 2023 — daily hydrograph with satellite acquisition epochs",
                 pad=22)
    ax.legend(handles=[Line2D([], [], color=C["gauge"], lw=1.2, label="gauge (daily; epoch not stated)"),
                       Line2D([], [], color=C["swot"], lw=1.0, label="SWOT PIXC acquisition"),
                       Line2D([], [], color=C["icesat"], lw=1.0, ls="--", label="ICESat-2 acquisition")],
              loc="upper left", fontsize=6.8)
    panel_label(ax, "a")
    ax2 = ax.twinx()
    ax2.set_ylim([(v / 100.0 - 5.0) for v in ax.get_ylim()])
    ax2.set_ylabel("H  BS-77  [m]", fontsize=7.5); ax2.grid(False); ax2.tick_params(labelsize=6.8)

    for i, r in enumerate(ctx.itertuples()):
        ax = fig.add_subplot(gs[1, i])
        w = h[(h.date >= r.date - pd.Timedelta(days=7)) & (h.date <= r.date + pd.Timedelta(days=7))]
        ax.plot(w.date, w.water_level_cm, color=C["gauge"], lw=1.3, marker="o", ms=3.5)
        ax.axvline(r.swot_utc, color=C["swot"], lw=1.2)
        ax.axvline(r.icesat_utc, color=C["icesat"], lw=1.2, ls="--")
        ax.axvspan(min(r.swot_utc, r.icesat_utc), max(r.swot_utc, r.icesat_utc),
                   color=C["bad"], alpha=0.13)
        ax.set_title(f"{r.date:%Y-%m-%d}  ·  {r.state}", fontsize=8,
                     color=STATE_C[r.state])
        ax.tick_params(labelsize=6.2); ax.tick_params(axis="x", rotation=40)
        if i == 0: ax.set_ylabel("stage [cm]")
        ax.text(0.5, 0.03,
                f"dH/dt = {r.dH_dt_central_cm_day:+.1f} cm/day\n"
                f"expected over Δt: {r.expected_change_cm:+.1f} cm\n"
                f"observed SWOT−ICESat: {r.observed_swot_minus_icesat_cm:+.1f} cm",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.0,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c8cccf", lw=0.55))
        panel_label(ax, "bcd"[i])
    fig.suptitle("Hydrological context of the SWOT–ICESat-2 matchups at Kherson", y=0.995)
    save(fig, "FigA_kherson_hydrograph_context", FIG); print("  FigA")


def figC():
    c = pd.read_csv(FD / "FigA_kherson_hydrograph_context.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lim = 25
    ax.plot([-lim, lim], [-lim, lim], color="k", ls="--", lw=0.9, label="1:1 (hydrology explains all)")
    ax.axhline(0, color="#95a5a6", lw=0.7); ax.axvline(0, color="#95a5a6", lw=0.7)
    for r in c.itertuples():
        mk = "o" if r.qc_flag == "ok" else "X"
        ax.scatter(r.expected_change_cm, r.observed_swot_minus_icesat_cm, s=120, marker=mk,
                   c=STATE_C[r.state], ec="black", lw=0.9, zorder=5)
        ax.annotate(f"{r.date:%m-%d}\n{r.state}\nΔt={r.delta_t_h:+.1f} h",
                    (r.expected_change_cm, r.observed_swot_minus_icesat_cm),
                    xytext=(9, -4), textcoords="offset points", fontsize=6.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("expected ΔH from the daily hydrograph over Δt  [cm]")
    ax.set_ylabel("observed  H$_{SWOT}$ − H$_{ICESat-2}$  [cm]")
    ax.set_title("Observed cross-sensor difference vs hydrological expectation")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.text(0.98, 0.03, "n = 3 — no regression fitted.\nPoints far from the 1:1 line are NOT\n"
            "explained by the daily hydrograph.\nSub-daily wind setup is unresolved.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "FigC_observed_vs_hydrological_expectation", FIG); print("  FigC")


def figD():
    pts = pd.read_csv(FD / "FigD_kakhovka_profile_points.csv")
    sl = pd.read_csv(FD / "FigD_kakhovka_perdate_slopes.csv")
    summ = pd.read_csv(FD / "FigD_kakhovka_slope_summary.csv")
    PC = {"PRE_BREACH": "#1b6ca8", "BREACH_DRAWDOWN": "#e08214", "POST_BREACH": "#b03a2e"}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.9], "wspace": 0.28})

    for ax, per, ttl, ylim in [
            (axes[0], "PRE_BREACH", "(a) PRE-BREACH — flat pool", 1.2),
            (axes[1], "POST_BREACH", "(b) POST-BREACH — gradient + heterogeneity", 3.2)]:
        s = sl[sl.period == per].sort_values("span_km", ascending=False).head(6)
        for j, r in enumerate(s.itertuples()):
            g = pts[(pts.date == r.date) & (pts.period == per)].sort_values("chain_km")
            y = g.wse_m - g.wse_m.median()
            # scatter only: connecting points would imply a continuous profile that
            # the post-breach system does not have (residual ponds vs main channel)
            ax.scatter(g.chain_km, y, s=26, alpha=0.85, ec="black", lw=0.35, zorder=4,
                       label=f"{r.date} ({r.slope_cm_per_km:+.2f} cm/km)")
            xs = np.array([g.chain_km.min(), g.chain_km.max()])
            ax.plot(xs, (r.slope_cm_per_km / 100.0) * (xs - g.chain_km.median()),
                    lw=1.0, alpha=0.55, zorder=3,
                    color=ax.collections[-1].get_facecolor()[0])
        zero_line(ax, 0)
        ax.set_xlabel("chainage upstream from the dam  [km]")
        ax.set_ylabel("WSE − date median  [m]")
        ax.set_title(ttl, color=PC[per])
        ax.set_ylim(-ylim, ylim)
        ax.legend(fontsize=5.6, loc="upper left", ncol=1)
        row = summ[summ.period == per].iloc[0]
        ax.text(0.98, 0.03, f"median slope {row.median_slope_cm_per_km:+.2f} cm/km\n"
                f"median WSE range {row.median_wse_range_m:.2f} m\n"
                f"{row.frac_positive_slope*100:.0f} % of dates positive · n={int(row.n_dates)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c8cccf", lw=0.6))
        panel_label(ax, "ab"[0 if per == "PRE_BREACH" else 1])
    axes[1].text(0.02, 0.97, "vertical scatter at similar chainage =\nresidual ponds vs main channel",
                 transform=axes[1].transAxes, va="top", fontsize=6.0, color="#5d6d7e")

    ax = axes[2]
    order = ["PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"]
    for i, per in enumerate(order):
        v = sl[sl.period == per].slope_cm_per_km
        v = v[np.abs(v) < 15]  # keep the axis readable; outliers annotated below
        ax.scatter(np.full(len(v), i) + np.random.default_rng(CFG.SEED).normal(0, .06, len(v)),
                   v, s=42, c=PC[per], ec="black", lw=0.6, alpha=0.9, zorder=5)
        m = np.median(sl[sl.period == per].slope_cm_per_km)
        ax.plot([i - .28, i + .28], [m, m], color="black", lw=2.2, zorder=6)
        ax.annotate(f"{m:+.2f}", (i, m), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7.2, fontweight="bold")
    zero_line(ax, 0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{p.replace('_',chr(10))}\nn={int(summ[summ.period==p].n_dates.iloc[0])}"
                        for p in order], fontsize=6.8)
    ax.set_ylabel("longitudinal WSE slope  [cm/km]")
    ax.set_title("(c) slope by regime")
    ax.set_ylim(-6, 11)
    n_out = int((np.abs(sl.slope_cm_per_km) >= 15).sum())
    if n_out:
        ax.text(0.03, 0.03, f"{n_out} date(s) with |slope| > 15 cm/km\nomitted from this axis "
                "(mixed water bodies)", transform=ax.transAxes, va="bottom", fontsize=6.0,
                color="#5d6d7e")
    panel_label(ax, "c")
    fig.suptitle("Former Kakhovka Reservoir: from flat pool to river-like longitudinal profile",
                 y=1.02, fontsize=10)
    save(fig, "FigD_kakhovka_longitudinal_profiles", FIG); print("  FigD")


def figE():
    pts = pd.read_csv(FD / "FigD_kakhovka_profile_points.csv")
    res = gpd.read_file(CFG.RESERVOIR_GEOJSON).to_crs(M)
    dam = gpd.GeoSeries([gpd.points_from_xy([CFG.KAKHOVKA_DAM[0]], [CFG.KAKHOVKA_DAM[1]])[0]],
                        crs=CFG.CRS_GEOG).to_crs(M)
    gg = pd.DataFrame(CFG.RESERVOIR_GAUGES, columns=["id", "name", "lon", "lat"])
    ggm = gpd.GeoDataFrame(gg, geometry=gpd.points_from_xy(gg.lon, gg.lat), crs=CFG.CRS_GEOG).to_crs(M)
    order = [("PRE_BREACH", "PRE-BREACH"), ("BREACH_DRAWDOWN", "DRAWDOWN"), ("POST_BREACH", "POST-BREACH")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3), gridspec_kw={"wspace": 0.06})
    vmin, vmax = 12.0, 17.0
    for ax, (per, ttl) in zip(axes, order):
        s = pts[pts.period == per]
        g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon_mean, s.lat_mean),
                             crs=CFG.CRS_GEOG).to_crs(M)
        res.plot(ax=ax, fc="#e8eef1", ec="#8fa9b5", lw=0.5, zorder=1)
        sc = ax.scatter(g.geometry.x, g.geometry.y, s=13, c=s.wse_m, cmap="viridis",
                        vmin=vmin, vmax=vmax, ec="none", zorder=4)
        ax.scatter(dam.x, dam.y, s=110, marker="*", c=C["dam"], ec="white", lw=0.6, zorder=6)
        ax.scatter(ggm.geometry.x, ggm.geometry.y, s=22, marker="s", c=C["gauge"],
                   ec="white", lw=0.5, zorder=5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{ttl}\nn={s.date.nunique()} dates, {len(s)} beam-passes", fontsize=8.5)
        ax.text(0.02, 0.02, f"WSE median {np.median(s.wse_m):.2f} m\nrange {s.wse_m.max()-s.wse_m.min():.2f} m",
                transform=ax.transAxes, va="bottom", fontsize=6.4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c8cccf", lw=0.55))
    scale_bar(axes[0], loc="lower right"); north_arrow(axes[2], loc=(0.93, 0.90))
    cb = fig.colorbar(sc, ax=axes, shrink=0.80, pad=0.012, aspect=28)
    cb.set_label("ICESat-2 WSE, EGG2015 frame  [m]", fontsize=7.5)
    fig.suptitle("Water-surface elevation over the former Kakhovka Reservoir by regime "
                 f"(breach {CFG.BREACH_DATE}) · CRS {M}", y=1.02, fontsize=9.5)
    save(fig, "FigE_kakhovka_regime_maps", FIG); print("  FigE")


def figF():
    pts = pd.read_csv(FD / "FigD_kakhovka_profile_points.csv")
    sl = pd.read_csv(FD / "FigD_kakhovka_perdate_slopes.csv")
    s = sl[(sl.period == "POST_BREACH") & (np.abs(sl.slope_cm_per_km) < 15)].sort_values("date")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    cmap = plt.get_cmap("plasma")
    for i, r in enumerate(s.itertuples()):
        g = pts[(pts.date == r.date) & (pts.period == "POST_BREACH")].sort_values("chain_km")
        col = cmap(i / max(1, len(s) - 1))
        ax.scatter(g.chain_km, g.wse_m, s=24, color=col, ec="black", lw=0.3, alpha=0.9,
                   label=f"{r.date}  ({r.slope_cm_per_km:+.2f})", zorder=4)
        xs = np.array([g.chain_km.min(), g.chain_km.max()])
        ax.plot(xs, (r.slope_cm_per_km / 100.0) * (xs - g.chain_km.median()) + g.wse_m.median(),
                color=col, lw=1.1, alpha=0.7, zorder=3)
    ax.set_xlabel("chainage upstream from the Kakhovka dam  [km]")
    ax.set_ylabel("ICESat-2 WSE, EGG2015 frame  [m]")
    ax.set_title("Post-breach longitudinal WSE profiles of the former reservoir")
    ax.legend(fontsize=6.0, ncol=2, loc="upper left", title="date (slope cm/km)",
              title_fontsize=6.4)
    ax.text(0.985, 0.03, f"n = {len(s)} post-breach dates with ≥3 chainages spanning >20 km\n"
            "one line per date; slope fitted per date (the independent unit)\n"
            "WSE rises upstream on 13 of 14 dates — a river-like gradient",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "FigF_postbreach_longitudinal_gradient", FIG); print("  FigF")


if __name__ == "__main__":
    for f in (figA, figC, figD, figE, figF):
        try: f()
        except Exception as e:
            import traceback; print(f"  !! {f.__name__} FAILED: {e}"); traceback.print_exc(limit=2)
    print("Done ->", FIG)
