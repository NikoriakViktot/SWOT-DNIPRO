#!/usr/bin/env python
"""MS6 — paper 1 map figures that carry the analysis zones.

Figure 1 (F00_study_area) and Figure S1 (F18_swot_icesat_whole_zone_map) were
drawn on the registry download containers ZONE_1..ZONE_4, which overlap and
whose F was the 10 km-buffered flood envelope. Both are redrawn here on the
paper zones R/F/D/E of scripts/ms5_paper1_zones.py (disjoint, in flow order),
with the June 2023 Sentinel-1 flood envelope shown as an event layer inside F,
never as a zone. The generator of the v5 figures was not kept in any
repository; this script is now their source.

    python scripts/ms6_paper1_figures.py            # both figures
    python scripts/ms6_paper1_figures.py --only F00

Outputs: outputs/paper/figures/F00_study_area.{png,pdf}
         outputs/paper/figures/F18_swot_icesat_whole_zone_map.{png,pdf}
         (F18 also writes its crossing table, see ms6b)
"""
from __future__ import annotations

import argparse
import json
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import p30_sea_post_yearbooks_2023 as P30

OUT = ROOT / "outputs/paper/figures"
MANIFEST = ROOT / "outputs/paper/zones/paper1_zones_manifest.json"
ZONES = ("R_FORMER_KAKHOVKA_RESERVOIR", "F_LOWER_DNIPRO_FLOODWAY",
         "D_KHERSON_DELTA", "E_DNIPRO_BUG_ESTUARY")
ZSTYLE = {  # fill, edge
    "R": ("#dbe9f1", "#5b8fa8"),
    "F": ("#f6ddd6", "#c1402a"),
    "D": ("#e3eedc", "#3f7d4e"),
    "E": ("#e8e1f0", "#7a4f9e"),
}
ZTEXT = {
    "R": "R  Reservoir: former Kakhovka Reservoir",
    "F": "F  Floodway: lower Dnipro, dam to Kherson",
    "D": "D  Delta: Kherson delta",
    "E": "E  Estuary: Dnipro–Buh estuary",
}
WATER = "#7aa3bd"
PRE, POST = "#f5c518", "#d7191c"   # yellow before the breach, red after: both read on blue water
GAUGE = "#b5651d"
DAM = "#c0392b"
# The four 2023 yearbook posts below the dam that the paper uses (Section 5.12)
POST_IDS = (80807, 98032, 98025, 98022)   # Kasperivka, Stanislav, Parutyne, Ochakiv
MYKOLAIV = 98027

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def zones_4326() -> dict[str, gpd.GeoSeries]:
    man = json.loads(MANIFEST.read_text())
    return {man["zones"][k]["letter"]: gpd.GeoSeries([SD.load(k)], crs=4326)
            for k in ZONES}, man


def _n(v) -> str:
    """Thousands separated by a space, as in the manuscript."""
    return f"{v:,.0f}".replace(",", " ")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT.relative_to(ROOT)}/{name}.png")


# --------------------------------------------------------------- Figure 1 --
def fig_study_area() -> None:
    Z, man = zones_4326()
    water = gpd.GeoSeries([SD.load("dnipro_water_domain")], crs=4326)
    env = gpd.GeoSeries([SD.load("event_flood_2023_s1_envelope")], crs=4326)
    F = Z["F"].iloc[0]
    env_in_f = env.intersection(F)

    seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet",
                          columns=["time", "lon", "lat"])
    post = seg.time >= pd.Timestamp(CFG.BREACH_DATE, tz="UTC")

    fig, ax = plt.subplots(figsize=(10, 7.2))
    for L, g in Z.items():
        fc, ec = ZSTYLE[L]
        g.plot(ax=ax, fc=fc, ec=ec, lw=0.9, zorder=1)
    Z["R"].boundary.plot(ax=ax, color=ZSTYLE["R"][1], lw=0.8, zorder=6)
    water.plot(ax=ax, fc=WATER, ec="none", alpha=0.85, zorder=2)
    env_in_f.plot(ax=ax, fc=ZSTYLE["F"][1], ec="none", alpha=0.45, zorder=3)
    ax.scatter(seg.lon[~post], seg.lat[~post], s=0.5, c=PRE, lw=0, zorder=4,
               rasterized=True)
    ax.scatter(seg.lon[post], seg.lat[post], s=0.5, c=POST, lw=0, zorder=5,
               rasterized=True)

    # zone letters at a representative interior point
    for L, g in Z.items():
        p = g.iloc[0].representative_point() if L != "R" else None
        if L == "R":   # beside the reservoir's widest part, clear of the gauges
            x, y = 34.62, 47.38
        elif L == "D":  # below the delta posts' labels
            x, y = 32.45, 46.42
        else:
            x, y = p.x, p.y
        ax.text(x, y, L, fontsize=22, weight="bold", color=ZSTYLE[L][1],
                ha="center", va="center", alpha=0.9, zorder=8)

    # SWOT calibration-orbit repeat reach (outlet -> Kherson, daily)
    dlon, dlat = CFG.KAKHOVKA_DAM
    klon, klat = CFG.KHERSON_GAUGE[2:]
    ax.plot([dlon, klon], [dlat, klat], ls="--", c=DAM, lw=1.6, zorder=6)

    gauges = [(n, lo, la) for _, n, lo, la in CFG.RESERVOIR_GAUGES]
    gauges.append(CFG.KHERSON_GAUGE[1:])
    m = P30.POSTS[MYKOLAIV]
    gauges.append(("Mykolaiv (Southern Bug)", m["lon"], m["lat"]))
    for n, lo, la in gauges:
        ax.scatter(lo, la, s=70, c=GAUGE, ec="k", lw=0.6, zorder=9)
        ax.annotate(n, (lo, la), xytext=(6, 5), textcoords="offset points",
                    fontsize=8, zorder=10)
    for pid in POST_IDS:
        m = P30.POSTS[pid]
        ax.scatter(m["lon"], m["lat"], s=55, marker="s", fc="w", ec=GAUGE,
                   lw=1.6, zorder=9)
        ax.annotate(m["name_en"], (m["lon"], m["lat"]),
                    xytext={98032: (-22, -16)}.get(pid, (6, -11)),
                    textcoords="offset points", fontsize=8, zorder=10)
    ax.scatter(dlon, dlat, s=150, marker="v", c=DAM, ec="k", lw=0.6, zorder=11)

    ax.set_aspect(1 / np.cos(np.deg2rad(47.0)))
    ax.set_xlabel("°E")
    ax.set_ylabel("°N")

    zl = [Patch(fc=ZSTYLE[L][0], ec=ZSTYLE[L][1],
                label=f"{ZTEXT[L]} ({_n(man['zones'][k]['area_km2'])} km²)")
          for L, k in zip("RFDE", ZONES)]
    leg1 = ax.legend(handles=zl, loc="upper left", fontsize=7.6, frameon=False,
                     title="analysis zones (disjoint, upstream → downstream)",
                     title_fontsize=8, alignment="left")
    ax.add_artist(leg1)
    n_pre, n_post = int((~post).sum()), int(post.sum())
    ol = [
        Patch(fc=WATER, ec="none", label="pre-breach water (WorldCover 2021)"),
        Patch(fc=ZSTYLE["F"][1], ec="none", alpha=0.45,
              label="June 2023 Sentinel-1 flood envelope in F (event layer)"),
        Line2D([], [], ls="", marker="o", ms=5, c=PRE,
               label=f"ICESat-2 ATL13  pre-breach (n={_n(n_pre)})"),
        Line2D([], [], ls="", marker="o", ms=5, c=POST,
               label=f"ICESat-2 ATL13  post-breach (n={_n(n_post)})"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=GAUGE, mec="k",
               label="hydrological gauge"),
        Line2D([], [], ls="", marker="s", ms=7, mfc="w", mec=GAUGE, mew=1.6,
               label="2023 yearbook post below the dam (Section 5.12)"),
        Line2D([], [], ls="", marker="v", ms=10, mfc=DAM, mec="k",
               label="Kakhovka dam (breached 6 Jun 2023)"),
        Line2D([], [], ls="--", c=DAM, lw=1.6,
               label="SWOT calibration-orbit repeat (outlet → Kherson, daily)"),
    ]
    ax.legend(handles=ol, loc="lower right", fontsize=7.6, frameon=False)
    _save(fig, "F00_study_area")


# --------------------------------------------------------------- Figure S1 --
def fig_swot_icesat_map() -> None:
    import ms6b_swot_icesat_crossings as XS
    XS.figure(zones_4326()[0], ZSTYLE, _save)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["F00", "F18"])
    a = ap.parse_args()
    if a.only in (None, "F00"):
        fig_study_area()
    if a.only in (None, "F18"):
        fig_swot_icesat_map()


if __name__ == "__main__":
    main()
