#!/usr/bin/env python
"""V0 — the gauge network and the datum problem, in EN and UK.

The opening figure: every gauge is a Baltic-1977 stage above its own local zero,
and each needs its own EPSG:9902 correction to reach EVRF2019. Nothing can be
compared with a satellite until that is done.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import contextily as ctx
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREY = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#8a94a3"

# The OSM tile policy requires a User-Agent that identifies the application;
# contextily's default ("contextily-<random hex>") is refused and silently
# returns an "Access blocked" placeholder image instead of an error.
# Put your own contact string here if you publish the figure.
OSM_HEADERS = {"User-Agent": "SWOT-DNIPRO-research-figure/1.0 "
                             "(static gauge map, a few tiles at zoom 8)"}
TO_LL = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform

TXT = {
    "en": dict(
        title="The starting problem: the gauges are not in the satellites' system",
        sub=("Every gauge reports a stage in centimetres above its own local zero on the "
             "Baltic 1977 datum.\nEach zero and each EPSG:9902 correction is different — "
             "so no gauge can be compared with a satellite height until it is converted."),
        res="reservoir gauges  ·  zero 12.00 m BS-77",
        down="downstream gauges  ·  zero −5.00 m BS-77",
        dam="Kakhovka dam (breached 6 Jun 2023)",
        pool="former reservoir pool (5 Jun 2023)",
        river="Dnipro channel (SWORD v16)",
        nodata="no level series available",
        cols=["gauge", "zero BS-77", "Δ EPSG:9902", "zero in EVRF2019", "record"],
        note=("Δ EPSG:9902 varies by 4.4 cm across this reach alone, so a single national "
              "constant would be wrong. The two gauges whose record spans the breach are "
              "shown in bold — they are the only in-situ anchors for the event."),
        spans="spans the breach",
        attrib="basemap © OpenStreetMap contributors (ODbL)"),
    "uk": dict(
        title="Вихідна проблема: гідропости не в системі супутників",
        sub=("Кожен пост дає стан у сантиметрах над власним нулем у Балтійській системі 1977 р.\n"
             "Нулі різні й поправка EPSG:9902 різна — доки їх не звести, жоден пост не можна "
             "порівняти з супутниковою висотою."),
        res="пости водосховища  ·  нуль 12.00 м БС-77",
        down="пости нижче греблі  ·  нуль −5.00 м БС-77",
        dam="Каховська гребля (підрив 6 черв. 2023)",
        pool="чаша водосховища (5 черв. 2023)",
        river="русло Дніпра (SWORD v16)",
        nodata="ряду рівня немає",
        cols=["пост", "нуль БС-77", "Δ EPSG:9902", "нуль у EVRF2019", "ряд"],
        note=("Δ EPSG:9902 змінюється на 4.4 см лише в межах цієї ділянки — тож єдина "
              "загальнонаціональна константа була б хибною. Жирним — два пости, чиї ряди "
              "перетинають підрив: це єдині наземні якорі події."),
        spans="перетинає підрив",
        attrib="підкладка © OpenStreetMap contributors (ODbL)"),
}
NAMES = {
    "en": {80977: "Nova Kakhovka", 80971: "Velyka Lepetykha", 80964: "Nikopol",
           80963: "Blahovishchenka", 80961: "Plavni", 80959: "Rozumivka",
           80805: "Kherson", 98027: "Mykolaiv"},
    "uk": {80977: "Нова Каховка", 80971: "В. Лепетиха", 80964: "Нікополь",
           80963: "Благовіщенка", 80961: "Плавні", 80959: "Розумівка",
           80805: "Херсон", 98027: "Миколаїв"},
}


def build(lang: str) -> None:
    t, nm = TXT[lang], NAMES[lang]
    st = pd.read_csv(CFG.TABLES / "gauge_vertical_reference_summary.csv")
    st["date_min"] = pd.to_datetime(st.date_min)
    st["date_max"] = pd.to_datetime(st.date_max)

    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
          SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    # The axes are drawn in lon/lat (channel, dam, gauges), so the registry
    # polygon -- which is metric -- is projected once, here, before plotting.
    fp_ll = shp_transform(TO_LL, fp)
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.15, 1], hspace=0.16)
    ax = fig.add_subplot(gs[0])

    for geom in (fp_ll.geoms if fp_ll.geom_type == "MultiPolygon" else [fp_ll]):
        x, y = geom.exterior.xy
        ax.fill(x, y, color=BLUE, alpha=0.30, zorder=3)
        ax.plot(x, y, color="#0d3d52", lw=1.4, alpha=0.95, zorder=3)
    ax.plot([], [], color=BLUE, lw=6, alpha=0.45, label=t["pool"])
    ax.plot(ch.lon, ch.lat, ".", ms=1.1, color="#2b2b2b", alpha=0.75, zorder=4)
    ax.plot([], [], "-", color="#2b2b2b", lw=1.4, label=t["river"])

    dlon, dlat = CFG.KAKHOVKA_DAM
    ax.plot(dlon, dlat, "v", ms=16, color=RED, mec="white", mew=2.0, zorder=8,
            label=t["dam"])

    for _, s in st.iterrows():
        has = s.n_obs > 0
        spans = bool(s.spans_breach)
        c = BLUE if s.domain == "reservoir" else AMBER
        if not has:
            ax.plot(s.lon, s.lat, "o", ms=11, mfc="white", mec="#5a5a5a", mew=2.0, zorder=7)
        else:
            ax.plot(s.lon, s.lat, "o", ms=16 if spans else 11, color=c,
                    mec="white", mew=2.2, zorder=7)
            if spans:
                ax.plot(s.lon, s.lat, "o", ms=25, mfc="none", mec="white", mew=3.4, zorder=6)
                ax.plot(s.lon, s.lat, "o", ms=25, mfc="none", mec=RED, mew=2.2, zorder=7)
        # explicit per-station label placement (dx, dy in points, ha)
        LBL = {80805: (0, -44, "center"), 98027: (0, -44, "center"),
               80977: (-14, -44, "right"), 80971: (0, 24, "center"),
               80964: (-10, 24, "center"), 80963: (0, 24, "center"),
               80959: (-6, 26, "center"), 80961: (16, -44, "left")}
        dx, dy, ha = LBL.get(s.station_id, (0, 24, "center"))
        ax.annotate(f"{nm[s.station_id]}\n{s.station_id}  Δ={s.delta_epsg9902_m:+.3f}",
                    (s.lon, s.lat), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.4,
                    fontweight="bold" if spans else "normal",
                    color=INK if has else GREY,
                    zorder=9,
                    bbox=dict(fc="white", ec="#cfcfcf", lw=0.5, alpha=0.90, pad=2.0))

    ax.plot([], [], "o", ms=10, color=BLUE, label=t["res"])
    ax.plot([], [], "o", ms=10, color=AMBER, label=t["down"])
    ax.plot([], [], "o", ms=10, mfc="none", mec=RED, mew=2, label=t["spans"])
    ax.plot([], [], "o", ms=9, mfc="white", mec=GREY, mew=1.6, label=t["nodata"])
    ax.set_xlim(31.7, 35.75); ax.set_ylim(46.28, 48.25)
    # OpenStreetMap basemap, warped from Web Mercator to the plotting CRS by
    # contextily. Tiles are a licensed third-party product: attribution below is
    # required, and the vector layers stay readable without it.
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", zoom=8, attribution=False,
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        headers=OSM_HEADERS, alpha=0.82)
        ax.text(0.995, 0.008, t["attrib"], transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.2, color="#333333", zorder=10,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
    except Exception as e:                       # offline -> plain background
        print(f"     [basemap unavailable: {type(e).__name__}]")
    ax.set_xlabel("°E"); ax.set_ylabel("°N")
    ax.grid(alpha=0.18, color="white", lw=0.6)
    ax.legend(fontsize=8.2, loc="upper left", framealpha=0.96, ncol=2,
              facecolor="white", edgecolor="#bbbbbb",
              borderpad=0.6, columnspacing=1.1)
    ax.set_title(t["title"] + "\n", fontsize=14.5, fontweight="bold", color=INK,
                 loc="left", pad=34)
    ax.text(0, 1.018, t["sub"], transform=ax.transAxes, fontsize=9.3, color=GREY,
            va="bottom", linespacing=1.45)

    # ---- table ---------------------------------------------------------
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    x0 = [0.0, 0.235, 0.375, 0.525, 0.70]
    for j, h in enumerate(t["cols"]):
        axt.text(x0[j], 0.97, h, fontsize=9.6, fontweight="bold", color=INK)
    axt.plot([0, 1], [0.925, 0.925], color=INK, lw=1.3)
    y = 0.845
    for _, s in st.sort_values(["domain", "station_id"]).iterrows():
        has = s.n_obs > 0
        spans = bool(s.spans_breach)
        c = INK if has else GREY
        fw = "bold" if spans else "normal"
        rec = (f"{s.date_min:%Y}–{s.date_max:%Y}  ({int(s.n_obs):,} d)"
               if has else t["nodata"])
        if spans:
            rec += "  ★"
        vals = [f"{nm[s.station_id]}  ({s.station_id})", f"{s.zero_bs77_m:+.2f} m",
                f"{s.delta_epsg9902_m:+.4f} m", f"{s.nominal_zero_evrf2019_m:+.3f} m", rec]
        for j, v in enumerate(vals):
            axt.text(x0[j], y, v, fontsize=8.9, color=c, fontweight=fw,
                     family="monospace" if j in (1, 2, 3) else None)
        y -= 0.105
    axt.text(0, y - 0.03, t["note"], fontsize=8.6, color=RED, va="top", wrap=True)
    axt.set_xlim(0, 1); axt.set_ylim(y - 0.16, 1.02)

    out = CFG.FIG / f"V0_gauge_network_{lang}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}.png")


if __name__ == "__main__":
    for lg in ("en", "uk"):
        build(lg)
