#!/usr/bin/env python
"""Phase 19 figures 1-7.

1  Sentinel-2 water extent  PRE / DRAWDOWN / POST
2  classified post-breach water system (main channel / side channels / residual water)
3  representative longitudinal profiles  PRE / DRAWDOWN / POST all-water / POST channel-only
4  time series of Theil-Sen channel slope, with the 2023-06-06 breach line
5  PRE vs DRAW vs POST channel-slope distributions
6  residual-water elevation relative to the reconstructed channel
7  all-water slope vs channel-only slope, by date
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import (C, graticule, nmad, north_arrow, panel_label,
                                        save, scale_bar, use_style, zero_line)

use_style()
FD, FIG, M = CFG.FIGDATA, CFG.FIG, CFG.CRS_METRIC
PROC = CFG.ROOT / "data" / "processed"
MASKS = PROC / "water_masks"
CLS_COL = {"MAIN_CHANNEL": "#1b6ca8", "CONNECTED_SIDE_CHANNEL": "#48c9b0",
           "RESIDUAL_POND": "#b03a2e", "FLOODED_DEPRESSION": "#e08214",
           "TRIBUTARY": "#7d3c98", "UNKNOWN": "#c4ccd1"}
PC = {"PRE_BREACH": "#1b6ca8", "BREACH_DRAWDOWN": "#e08214", "POST_BREACH": "#b03a2e"}


def _cl():
    d = pd.read_parquet(CFG.TABLES / "atl13_water_classification.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d


def _load_mask(name):
    from affine import Affine
    z = np.load(MASKS / f"{name}.npz", allow_pickle=True)
    return z["mask"], Affine(*z["affine"]), str(z["crs"])


def fig1_water_extent():
    summ = CFG.TABLES / "water_mask_summary.csv"
    if not summ.exists():
        print("  Fig1 skipped (no water_mask_summary.csv)"); return
    s = pd.read_csv(summ)
    s["sensing_dt"] = pd.to_datetime(s.sensing_time, format="%Y%m%dT%H%M%S")
    # one representative T36TWT scene per regime (largest connected water body = channel)
    picks = []
    for reg, lab in [("PRE_BREACH", "PRE-BREACH"), ("BREACH_DRAWDOWN", "DRAWDOWN"),
                     ("POST_BREACH", "POST-BREACH")]:
        sub = s[(s.regime == reg) & (s.tile == "T36TWT")]
        if sub.empty:
            sub = s[s.regime == reg]
        if sub.empty:
            continue
        picks.append((sub.sort_values("cloud_cover_meta").iloc[0], lab))
    if not picks:
        print("  Fig1 skipped (no scenes)"); return
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.6))
    if len(picks) == 1:
        axes = [axes]
    for ax, (row, lab) in zip(axes, picks):
        mask, aff, crs = _load_mask(row["name"])
        ax.imshow(mask, cmap="Blues", vmin=0, vmax=1.4, interpolation="nearest",
                  extent=[aff.c, aff.c + aff.a * mask.shape[1],
                          aff.f + aff.e * mask.shape[0], aff.f])
        ax.set_title(f"{lab}\n{pd.Timestamp(row.sensing_time[:8]).date()}  "
                     f"water {row.water_km2:,.0f} km²  cloud {row.cloud_cover_meta:.1f}%",
                     fontsize=8.2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Sentinel-2 water extent, tile T36TWT (mid former reservoir)  ·  "
                 "NDWI>0 ∧ MNDWI>0 ∧ SCL-permitted", y=1.02, fontsize=9.2)
    save(fig, "P19_Fig1_water_extent_pre_draw_post", FIG)
    print("  Fig1")


def fig2_classification():
    cl = _cl()
    post = cl[(cl.period == "POST_BREACH") & (cl.image_name != "")]
    if post.empty:
        print("  Fig2 skipped (no classified post-breach points)"); return
    ch = pd.read_parquet(PROC / "profiles" / "sword_dnipro_channel.parquet")
    g = gpd.GeoDataFrame(post, geometry=gpd.points_from_xy(post.lon, post.lat),
                         crs=4326).to_crs(M)
    chg = gpd.GeoDataFrame(ch, geometry=gpd.points_from_xy(ch.lon, ch.lat),
                           crs=4326).to_crs(M)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.scatter(chg.geometry.x, chg.geometry.y, s=1.4, c="#3b6e8f", alpha=0.5, lw=0,
               zorder=2, label="SWORD Dnipro nodes")
    order = ["UNKNOWN", "TRIBUTARY", "FLOODED_DEPRESSION", "RESIDUAL_POND",
             "CONNECTED_SIDE_CHANNEL", "MAIN_CHANNEL"]
    for k in order:
        m = (post.water_class == k).values
        if m.sum() == 0:
            continue
        ax.scatter(g.geometry.x[m], g.geometry.y[m], s=5 if k != "UNKNOWN" else 2,
                   c=CLS_COL[k], alpha=0.5 if k == "UNKNOWN" else 0.85, lw=0,
                   zorder=3 + order.index(k), label=f"{k} (n={int(m.sum()):,})", rasterized=True)
    dam = gpd.GeoSeries(gpd.points_from_xy([CFG.KAKHOVKA_DAM[0]], [CFG.KAKHOVKA_DAM[1]]),
                        crs=4326).to_crs(M)
    ax.scatter(dam.x, dam.y, s=170, marker="*", c=C["dam"], ec="white", lw=0.7, zorder=20)
    ax.set_aspect("equal"); graticule(ax, M, 0.5); scale_bar(ax, loc="lower right")
    north_arrow(ax)
    ax.legend(loc="upper left", fontsize=6.3, markerscale=2.4)
    ax.set_title("Post-breach ATL13 observations classified by water-body type")
    ax.set_xlabel("Sentinel-2 NDWI+MNDWI+SCL mask · SWORD v16 topology · connected-component "
                  "connectivity · WSE consistency")
    save(fig, "P19_Fig2_classified_water_system", FIG)
    print("  Fig2")


def fig3_profiles():
    prof = pd.read_csv(FD / "P19_profiles.csv", parse_dates=["date"])
    cl = _cl()
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.7), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    panels = [("PRE_BREACH", "PRE_CHANNEL", "(a) PRE-BREACH channel"),
              ("BREACH_DRAWDOWN", "MAIN_CHANNEL", "(b) DRAWDOWN channel"),
              ("POST_BREACH", "ALL_WATER", "(c) POST all-water"),
              ("POST_BREACH", "MAIN_CHANNEL", "(d) POST channel-only")]
    for j, (per, sub, ttl) in enumerate(panels):
        ax = axes[j]
        s = prof[(prof.period == per) & (prof.subset == sub)]
        if s.empty:
            ax.text(.5, .5, "no qualifying dates", transform=ax.transAxes, ha="center")
            ax.set_title(ttl, fontsize=8.5); continue
        s = s.sort_values("span_km", ascending=False).head(4)
        for r in s.itertuples():
            if sub == "PRE_CHANNEL":
                xs = np.array([r.chain_min_km, r.chain_max_km])
                ax.plot(xs, (r.slope_theilsen_cm_km / 100) * (xs - xs.mean()), lw=1.4,
                        label=f"{r.date:%Y-%m-%d} {r.slope_theilsen_cm_km:+.2f}")
                continue
            g = cl[cl.date == r.date]
            g = g[g.is_water == 1.0] if sub == "ALL_WATER" else g[g.water_class == sub]
            if g.empty:
                continue
            y = g.wse_m - np.median(g.wse_m)
            ax.scatter(g.chain_km, y, s=5, alpha=0.5, lw=0, rasterized=True,
                       label=f"{r.date:%Y-%m-%d} {r.slope_theilsen_cm_km:+.2f}")
            xs = np.array([g.chain_km.min(), g.chain_km.max()])
            ax.plot(xs, (r.slope_theilsen_cm_km / 100) * (xs - np.median(g.chain_km)),
                    lw=1.2, alpha=0.8, color=ax.collections[-1].get_facecolor()[0])
        zero_line(ax, 0)
        med = prof[(prof.period == per) & (prof.subset == sub)].slope_theilsen_cm_km.median()
        ax.set_title(f"{ttl}\nmedian {med:+.2f} cm/km", fontsize=8.5, color=PC[per])
        ax.set_xlabel("chainage from dam [km]", fontsize=7.5)
        ax.legend(fontsize=5.4, loc="upper left")
        panel_label(ax, "abcd"[j])
    axes[0].set_ylabel("WSE − date median  [m]")
    axes[0].set_ylim(-2.6, 2.6)
    fig.suptitle("Longitudinal profiles: SWORD chainage, Theil-Sen fits, one line per date",
                 y=1.03, fontsize=9.5)
    save(fig, "P19_Fig3_profiles_by_regime", FIG)
    print("  Fig3")


def _channel_prof():
    prof = pd.read_csv(FD / "P19_profiles.csv", parse_dates=["date"])
    return prof[prof.subset.isin(["MAIN_CHANNEL", "PRE_CHANNEL"])].copy()


def fig4_timeseries():
    ch = _channel_prof()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for per in PC:
        s = ch[ch.period == per]
        if s.empty:
            continue
        ax.scatter(s.date, s.slope_theilsen_cm_km, s=54, c=PC[per], ec="black", lw=0.6,
                   zorder=5, label=f"{per.replace('_',' ').title()} (n={len(s)})")
        ax.errorbar(s.date, s.slope_theilsen_cm_km,
                    yerr=[s.slope_theilsen_cm_km - s.ts_lo_cm_km,
                          s.ts_hi_cm_km - s.slope_theilsen_cm_km],
                    fmt="none", ecolor=PC[per], alpha=0.35, lw=0.8, zorder=4)
    br = pd.Timestamp(CFG.BREACH_DATE)
    ax.axvline(br, color=C["bad"], lw=2.0, zorder=3)
    ax.annotate("dam breach 2023-06-06", (br, ax.get_ylim()[1]), xytext=(6, -12),
                textcoords="offset points", fontsize=7.6, color=C["bad"], fontweight="bold")
    zero_line(ax, 0)
    for per in ("PRE_BREACH", "POST_BREACH"):
        s = ch[ch.period == per]
        if len(s) < 2:
            continue
        m = s.slope_theilsen_cm_km.median()
        ax.plot([s.date.min(), s.date.max()], [m, m], color=PC[per], lw=1.8, ls="--", zorder=4)
        ax.annotate(f"median {m:+.2f}", (s.date.max(), m), xytext=(-4, 7),
                    textcoords="offset points", ha="right", fontsize=7,
                    color=PC[per], fontweight="bold")
    ax.set_ylabel("MAIN-CHANNEL Theil-Sen slope  [cm/km]")
    ax.set_title("Channel-only longitudinal slope through the breach")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "P19_Fig4_channel_slope_timeseries", FIG)
    print("  Fig4")


def fig5_distributions():
    ch = _channel_prof()
    st = pd.read_csv(FD / "P19_statistics.csv") if (FD / "P19_statistics.csv").exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    order = ["PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"]
    rng = np.random.default_rng(CFG.SEED)
    for i, per in enumerate(order):
        v = ch[ch.period == per].slope_theilsen_cm_km
        if v.empty:
            continue
        ax.scatter(np.full(len(v), i) + rng.normal(0, .07, len(v)), v, s=42, c=PC[per],
                   ec="black", lw=0.5, zorder=5)
        m = np.median(v)
        ax.plot([i - .3, i + .3], [m, m], color="black", lw=2.4, zorder=6)
        ax.annotate(f"{m:+.2f}", (i, m), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=7.6, fontweight="bold")
    zero_line(ax, 0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PRE", "DRAW", "POST"], fontsize=8.5)
    ax.set_ylabel("MAIN-CHANNEL Theil-Sen slope [cm/km]")
    ax.set_title("Channel slope by regime  (unit = one date)")
    if len(st):
        r = st[st.comparison.str.contains("MAIN_CHANNEL vs") & (st.estimator == "Theil-Sen")]
        if len(r):
            r = r.iloc[0]
            ax.text(0.5, 0.98, f"Δ = {r.diff_cm_km:+.2f} cm/km  "
                    f"95% CI [{r.diff_ci_lo:+.2f}, {r.diff_ci_hi:+.2f}]\n"
                    f"permutation p = {r.permutation_p:.4f}\n"
                    f"sign  PRE {r.pre_positive}  /  POST {r.post_positive}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "P19_Fig5_channel_slope_distributions", FIG)
    print("  Fig5")


def fig6_residual():
    p = FD / "P19_residual_offsets.csv"
    if not p.exists():
        print("  Fig6 skipped (no residual-water offsets)"); return
    r = pd.read_csv(p, parse_dates=["date"])
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8),
                             gridspec_kw={"width_ratios": [1.1, 1.2, 1], "wspace": 0.3})
    ax = axes[0]
    ax.hist(r.delta_H_residual_m.clip(-2, 5), bins=45, color=C["bad"], alpha=0.75,
            ec="white", lw=0.3)
    med = r.delta_H_residual_m.median()
    ax.axvline(0, color="k", lw=1.0, ls="--")
    ax.axvline(med, color="black", lw=1.8)
    ax.annotate(f"median {med:+.2f} m", (med, ax.get_ylim()[1] * .88), xytext=(6, 0),
                textcoords="offset points", fontsize=7.4, fontweight="bold")
    ax.set_xlabel(r"$\Delta H = H_{residual} - \hat H_{channel}(s)$  [m]")
    ax.set_ylabel(f"ATL13 segments (n={len(r):,})")
    ax.set_title("Residual water vs channel")
    panel_label(ax, "a")

    ax = axes[1]
    for k in ("RESIDUAL_POND", "FLOODED_DEPRESSION"):
        s = r[r.water_class == k]
        if s.empty:
            continue
        ax.scatter(s.chain_km, s.delta_H_residual_m, s=13, c=CLS_COL[k], alpha=0.6, lw=0,
                   label=f"{k} (n={len(s):,})", rasterized=True)
    zero_line(ax, 0)
    ax.set_xlabel("chainage from dam [km]")
    ax.set_ylabel(r"$\Delta H$ above channel [m]")
    ax.set_title("Offset vs position")
    ax.legend(fontsize=6.4, loc="upper left")
    ax.set_ylim(-2, 5)
    panel_label(ax, "b")

    ax = axes[2]
    pers = CFG.TABLES / "residual_water_persistence.csv"
    if pers.exists():
        b = pd.read_csv(pers)
        ax.scatter(b.n_dates, b.median_delta_m, s=30, c=C["muted"], alpha=0.55, lw=0,
                   label=f"all locations (n={len(b)})")
        m = b[b.n_dates > 1]
        if len(m):
            ax.scatter(m.n_dates, m.median_delta_m, s=64,
                       c=np.where(m.sign_stable, C["bad"], C["muted"]), ec="black", lw=0.6,
                       zorder=5, label=f">1 date (n={len(m)}, sign-stable {int(m.sign_stable.sum())})")
        zero_line(ax, 0)
        ax.set_xlabel("dates the water body is observed")
        ax.set_ylabel(r"median $\Delta H$ [m]")
        ax.set_title("Temporal persistence")
        ax.legend(fontsize=6.2)
    panel_label(ax, "c")
    fig.suptitle("Residual water bodies relative to the reconstructed Dnipro channel profile",
                 y=1.03, fontsize=9.5)
    save(fig, "P19_Fig6_residual_water_offsets", FIG)
    print("  Fig6")


def fig7_compare():
    p = FD / "P19_allwater_vs_channel.csv"
    if not p.exists():
        print("  Fig7 skipped"); return
    d = pd.read_csv(p, parse_dates=["date"])
    d = d[d.period != "PRE_BREACH"]
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    hi = max(12, np.nanmax(d[["slope_all_water_cm_km", "slope_channel_cm_km"]].values) * 1.1)
    lim = (-2, hi)
    ax.plot(lim, lim, color="k", ls="--", lw=0.9, label="1:1 (classification changes nothing)")
    for per, col in PC.items():
        s = d[d.period == per]
        if s.empty:
            continue
        ax.scatter(s.slope_all_water_cm_km, s.slope_channel_cm_km, s=64, c=col, ec="black",
                   lw=0.6, zorder=5, label=f"{per.replace('_',' ').title()} (n={len(s)})")
    zero_line(ax, 0); ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("ALL-WATER Theil-Sen slope  [cm/km]")
    ax.set_ylabel("MAIN-CHANNEL-only Theil-Sen slope  [cm/km]")
    ax.set_title("Does the gradient survive channel-only classification?")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.text(0.98, 0.03, f"median all-water {d.slope_all_water_cm_km.median():+.2f}\n"
            f"median channel  {d.slope_channel_cm_km.median():+.2f}\n"
            f"median Δ {d.delta_slope_cm_km.median():+.2f} cm/km",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c8cccf", lw=0.6))
    save(fig, "P19_Fig7_allwater_vs_channel", FIG)
    print("  Fig7")


if __name__ == "__main__":
    for f in (fig1_water_extent, fig2_classification, fig3_profiles, fig4_timeseries,
              fig5_distributions, fig6_residual, fig7_compare):
        try:
            f()
        except Exception as e:
            import traceback
            print(f"  !! {f.__name__}: {e}")
            traceback.print_exc(limit=2)
    print("Done ->", FIG)
