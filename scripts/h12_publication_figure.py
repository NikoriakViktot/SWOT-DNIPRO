#!/usr/bin/env python
"""Publication-ready composite figure: top = map of the three
KAKHOVKA_RESERVOIR_CORE sub-zones (+ the two ATL13-unsampled upper zones,
shown greyed out for honest scope), bottom = PRE/POST slope distributions
per zone with Hodges-Lehmann shift + bootstrap CI.

Outputs
-------
outputs/figures/H12_publication_figure.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.gridspec import GridSpec
from scipy import stats as sstats
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

RNG = np.random.default_rng(42)
N_BOOT = 3000
REACH_BREAKS = (0, 133, 183, 277.2)
REACH_LABELS = ("CORE_LOWER", "CORE_MIDDLE", "CORE_UPPER")
ZONE_COLORS = {"CORE_LOWER": "#236f8c", "CORE_MIDDLE": "#3f7d4e", "CORE_UPPER": "#b07d27"}
INK, GREY = "#1a2228", "#c9c9c9"

_TF = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def hodges_lehmann(a, b):
    return np.median(np.subtract.outer(b, a).ravel())


def bootstrap_ci(pre, post, n_boot=N_BOOT):
    boots = [np.median(RNG.choice(post, len(post), replace=True)) -
            np.median(RNG.choice(pre, len(pre), replace=True)) for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def main() -> None:
    df = pd.read_csv(CFG.FIGDATA / "FigD_kakhovka_profile_points.csv")
    df = df[df.qc_pass].copy()
    df["core_reach"] = pd.cut(df.chain_km, REACH_BREAKS, labels=REACH_LABELS, include_lowest=True)

    zrows = []
    for (d, per, reach), g in df.groupby(["date", "period", "core_reach"], observed=True):
        if len(g) < 3:
            continue
        ts = sstats.theilslopes(g.wse_m.values, g.chain_km.values).slope * 100
        zrows.append({"date": d, "period": per, "zone": reach, "slope_cm_km": ts})
    zdf = pd.DataFrame(zrows)

    fig = plt.figure(figsize=(11, 13))
    gs = GridSpec(2, 1, height_ratios=[1.1, 1], hspace=0.28)

    # ---- top: zone map -------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    sa2_m = shp_transform(_TF, SD.load("reservoir_full_pool_prebreach"))
    hyd = gpd.read_file(CFG.ROOT / "data/processed/study_domain/hydraulic_longitudinal_domain.gpkg")

    def plot_poly(g, ax, **kw):
        gg = g.geoms if hasattr(g, "geoms") else [g]
        for p in gg:
            if p.is_empty:
                continue
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, **kw)

    # colour SA_2 by reach using the point-cloud's chain_km bounds (approximate,
    # visual aid only -- exact reach boundaries are 1-D along chainage, not a
    # precise 2-D split of the polygon; this is disclosed in the caption)
    plot_poly(sa2_m, ax0, facecolor="#dcdcdc", edgecolor=INK, lw=1.2, zorder=1)
    for reach, color in ZONE_COLORS.items():
        sub = df[df.core_reach == reach]
        xs, ys = _TF(sub.lon_mean.values, sub.lat_mean.values)
        ax0.scatter(xs, ys, s=10, color=color, alpha=0.6, label=reach, zorder=3)

    # grey out the two unsampled upper zones, honestly
    for zn in ("DNIPROHES_CONTROL_ZONE", "KAKHOVKA_UPPER_BACKWATER_CANDIDATE"):
        geom = hyd[hyd.zone_name == zn].geometry.iloc[0]
        plot_poly(geom, ax0, facecolor=GREY, edgecolor="grey", lw=1, alpha=0.9, zorder=2)
    ax0.plot([], [], color="grey", lw=6, alpha=0.9,
            label="DniproHES control / upper backwater candidate\n(0 ATL13 points -- NOT covered by this result)")

    ax0.set_aspect(1 / np.cos(np.radians(47.3)) if False else "equal")
    ax0.set_xlabel("Easting, m (EPSG:32636)"); ax0.set_ylabel("Northing, m")
    ax0.set_title("A. KAKHOVKA_RESERVOIR_CORE sub-zones (ATL13 profile points shown)\n"
                 "grey = unsampled upper transition zones, out of scope for this result",
                 loc="left", fontsize=10)
    ax0.legend(fontsize=7.5, loc="upper left", framealpha=0.95)

    # ---- bottom: PRE/POST distributions + HL shift ----------------------
    ax1 = fig.add_subplot(gs[1])
    positions = []
    labels = []
    p = 0
    stats_text = []
    for reach in REACH_LABELS:
        pre = zdf[(zdf.zone == reach) & (zdf.period == "PRE_BREACH")].slope_cm_km.dropna().values
        post = zdf[(zdf.zone == reach) & (zdf.period == "POST_BREACH")].slope_cm_km.dropna().values
        for label, vals, off in ((f"{reach}\nPRE (n={len(pre)})", pre, 0),
                                 (f"{reach}\nPOST (n={len(post)})", post, 0.8)):
            jitter = RNG.uniform(-0.15, 0.15, size=len(vals))
            color = ZONE_COLORS[reach] if off else "#9fb8c2"
            alpha = 0.85 if off else 0.5
            ax1.scatter(np.full(len(vals), p + off) + jitter, vals, color=color, alpha=alpha,
                       s=28, zorder=3, edgecolor="none")
            ax1.scatter([p + off], [np.median(vals)], color="black", marker="_", s=400, zorder=5)
        hl = hodges_lehmann(pre, post)
        ci = bootstrap_ci(pre, post)
        stats_text.append(f"{reach}: HL shift = {hl:+.2f} cm/km, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        ax1.annotate(f"HL={hl:+.1f}\n[{ci[0]:+.1f},{ci[1]:+.1f}]", xy=(p + 0.4, max(pre.max(), post.max()) + 3),
                    ha="center", fontsize=8, color=INK)
        positions += [p, p + 0.8]
        labels += [f"{reach}\nPRE", "POST"]
        p += 2.2
    ax1.axhline(0, color="grey", lw=0.8, ls=":")
    ax1.set_xticks(positions); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Theil-Sen local slope, cm/km")
    ax1.set_title("B. PRE vs POST slope distributions by zone, with Hodges-Lehmann shift + 95% bootstrap CI",
                 loc="left", fontsize=9.5)
    fig.text(0.5, 0.495, "\n".join(stats_text), ha="center", va="top", fontsize=8.5, color=INK)
    fig.suptitle("Spatial heterogeneity of the post-breach longitudinal slope shift\n"
                "(within the sampled former Kakhovka Reservoir core only)", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = CFG.FIG / "H12_publication_figure.png"
    fig.savefig(out, dpi=170)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
