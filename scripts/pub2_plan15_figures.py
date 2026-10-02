#!/usr/bin/env python
"""PUB2 -- publication figures for the plan-15 results (user 2026-09-19: "графіки та карти для публікації").

Seven figures, project style (swot_dnipro.plotting.style), EPSG:32636, scale bar + north arrow + graticule on every map,
vector PDF + 400 dpi PNG in outputs/figures/publication/plan15/. Nothing is computed here: every panel draws a product that a
numbered script wrote (p40, p42, p43, p47, p49, p50, p51, p51b) and the caption text names it.

  fig01_system_map          the breach as one system: reservoir bed A (2023 recession zoning) + below-dam floodplain B (June-2023 flood)
  fig02_two_domains         why A and B are two study zones: pre-breach state, land cover, index trajectories, separability
  fig03_indices_2022_2026   NDVI / MNDWI / AWEIsh / class over both domains, last normal year (2022) vs current (2026)
  fig04_flood_rf            observation-constrained vs prior-assisted RF flood reconstruction + common-support skill + permutation importance
  fig05_flood_area_compare  published flood-area estimates by type vs this study
  fig06_pool_succession     class areas by state, Manning n by state, water occurrence 2023-24, ICESat-2 canopy
  fig07_manning_maps        n_base maps of the bed (2023 / 2024 / 2026), delta n, floodplain n 2026
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource, BoundaryNorm
from matplotlib.patches import Patch
from shapely.geometry import Point

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro.plotting import style as ST
from swot_dnipro.plotting import maps as M

OUT = ROOT / "outputs/figures/publication/plan15"
FLOOD = CFG.BULK_ROOT / "floodplain"; TERR = CFG.BULK_ROOT / "terrain"; RO = ROOT / "outputs/rasters/roughness"
Z1, Z2, Z3, Z4 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY"
ZN = {Z1: 1, Z2: 2, Z3: 3, Z4: 4}
CA, CB = "#1f4e79", "#7a3b00"
STATES = ["BREACH_2023", "FIRST_EXPOSURE_2023", "STATE_2024", "STATE_2025", "CURRENT_2026"]
STATE_LBL = {"BREACH_2023": "Jun 2023\n(breach)", "FIRST_EXPOSURE_2023": "Jul–Sep\n2023", "STATE_2024": "2024", "STATE_2025": "2025", "CURRENT_2026": "2026"}


# ----------------------------------------------------------------------------- helpers
def dec(path: Path, k: int, nodata_extra=(), scale=None):
    with rasterio.open(path) as ds:
        h, w = max(ds.height // k, 1), max(ds.width // k, 1)
        a = ds.read(1, out_shape=(h, w), resampling=Resampling.nearest).astype("f4")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        for e in nodata_extra:
            a[a == e] = np.nan
        tr = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        ext = (tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f)
    return (a * scale if scale else a), tr, ext


def mask_to(geom, a, tr):
    return np.where(features.rasterize([(geom, 1)], out_shape=a.shape, transform=tr, fill=0, dtype="uint8").astype(bool), a, np.nan)


def hillshade(ax, zones, k=8, alpha=0.55, extent=None):
    """One hillshade mosaic on a common coarse grid over *extent* (per-zone frames overlap and would show as rectangles)."""
    from rasterio.warp import reproject
    from rasterio.transform import from_origin
    x0, x1, y0, y1 = extent; cell = 20.0 * k; nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
    tr = from_origin(x0, y1, cell, cell); dem = np.full((ny, nx), np.nan, "f4")
    for z in zones:
        f = TERR / z / "fabdem_20m.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as ds:
            dst = np.full((ny, nx), np.nan, "f4")
            reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=tr, dst_crs=ds.crs, resampling=Resampling.average, dst_nodata=np.nan)
        dst[dst == -9999] = np.nan; dem = np.where(np.isnan(dem), dst, dem)
    fill = np.nanmedian(dem) if np.isfinite(dem).any() else 0.0; hs = LightSource(azdeg=315, altdeg=45).hillshade(np.where(np.isfinite(dem), dem, fill), vert_exag=8, dx=cell, dy=cell)
    hs = np.where(np.isfinite(dem), hs, np.nan)
    ax.imshow(hs, extent=(x0, x0 + nx * cell, y1 - ny * cell, y1), cmap="gray", vmin=0, vmax=1, alpha=alpha, interpolation="bilinear", zorder=1)


def furniture(ax, extent, scale_km=None, arrow=True, arrow_loc=(0.955, 0.93)):
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal"); ax.grid(False)
    ST.scale_bar(ax, scale_km); ST.graticule(ax, CFG.CRS_METRIC, step=0.5)
    if arrow:
        ST.north_arrow(ax, loc=arrow_loc)
    ax.tick_params(labelsize=6.5)


def points():
    dam = gpd.GeoSeries([Point(*CFG.KAKHOVKA_DAM)], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]
    kh = gpd.GeoSeries([Point(CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3])], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]
    return dam, kh


def outlines(ax, A, B, dam, kh, lw=0.8):
    gpd.GeoSeries([A], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color=CA, lw=lw, zorder=6)
    gpd.GeoSeries([B], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color=CB, lw=lw, zorder=6)
    ax.plot(dam.x, dam.y, marker=ST.MARKERS["dam"], color=ST.C["dam"], ms=9, mec="k", mew=0.4, zorder=8)
    ax.plot(kh.x, kh.y, marker=ST.MARKERS["gauge"], color=ST.C["gauge"], ms=5, zorder=8)


def sys_extent(A, B, pad=4000):
    b = np.array([A.bounds, B.bounds]); return (b[:, 0].min() - pad, b[:, 2].max() + pad, b[:, 1].min() - pad, b[:, 3].max() + pad)


# ----------------------------------------------------------------------------- fig 1
def fig01(A, B, dam, kh):
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); ex = sys_extent(A, B)
    hillshade(ax, (Z1, Z4, Z2, Z3), k=10, extent=ex)
    a, tr, ext = dec(RO / "pool/zone1_poolwin_bed_recession_zoning_2023.tif", 5, nodata_extra=(0,)); a = mask_to(A, a, tr)
    zc = ListedColormap(["#fee08b", "#fdae61", "#d73027", "#08519c"])
    ax.imshow(a, extent=ext, cmap=zc, vmin=0.5, vmax=4.5, interpolation="nearest", zorder=3)
    for z in (Z2, Z4):
        d, dtr, dext = dec(FLOOD / z / f"{z}_domain_mask.tif", 3); ax.imshow(np.where(d == 1, 1.0, np.nan), extent=dext, cmap=ListedColormap(["#c7e9c0"]), vmin=0, vmax=1, interpolation="nearest", zorder=2)
        f, ftr, fext = dec(FLOOD / z / f"{z}_flood_envelope_clean.tif", 3, nodata_extra=(255,)); f = mask_to(B, f, ftr)
        ax.imshow(np.where(f == 1, 1.0, np.nan), extent=fext, cmap=ListedColormap(["#2b8cbe"]), vmin=0, vmax=1, interpolation="nearest", zorder=3)
    for z, col in ((Z3, "#888888"),):
        gpd.GeoSeries([SD.load_utm(z)], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color=col, lw=0.5, ls=":", zorder=5)
    outlines(ax, A, B, dam, kh)
    ax.annotate("Kakhovka dam", (dam.x, dam.y), xytext=(8, 6), textcoords="offset points", fontsize=7)
    ax.annotate("Kherson", (kh.x, kh.y), xytext=(6, -10), textcoords="offset points", fontsize=7)
    ax.text(0.56, 0.93, "A — reservoir bed (drained June 2023)", transform=ax.transAxes, fontsize=8, color=CA, weight="bold")
    ax.text(0.02, 0.40, "B — below-dam floodplain\n(flooded June 2023, then receded)", transform=ax.transAxes, fontsize=8, color=CB, weight="bold")
    handles = [Patch(fc="#fee08b", label="A: bed exposed by 30 Jun 2023"), Patch(fc="#fdae61", label="A: exposed by 6 Aug 2023"), Patch(fc="#d73027", label="A: exposed by 8 Sep 2023"),
               Patch(fc="#08519c", label="A: not exposed in 2023 (channel, lakes)"), Patch(fc="#c7e9c0", label=f"B: floodplain domain ({B.area/1e6:,.0f} km²)".replace(",", " ")), Patch(fc="#2b8cbe", label="B: June-2023 flood (S1 ≥ 2 events ∪ S2)"),
               plt.Line2D([], [], color="#888888", ls=":", lw=0.8, label="Dnipro–Buh estuary zone")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5)
    furniture(ax, ex, 50, arrow_loc=(0.06, 0.93))
    ax.set_title("The Kakhovka breach as one system: the bed that drained (A) and the floodplain that flooded (B)", fontsize=9)
    ST.save(fig, "fig01_system_map", OUT); print("fig01")


# ----------------------------------------------------------------------------- fig 2
def fig02():
    F = pd.read_csv(CFG.TABLES / "p49_domain_features_by_year.csv"); S = pd.read_csv(CFG.TABLES / "p49_domain_separability_2022.csv")
    years = sorted(F.year.unique())
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    # (a) separability bars (Cliff's delta) 2022
    ax = axes[0, 0]; s = S.copy(); s["feature"] = s.feature.str.replace("_2022", "").str.replace("water_share", "water share"); s = s.sort_values("cliffs_delta")
    ax.barh(s.feature, s.cliffs_delta, color=np.where(s.cliffs_delta > 0, CA, CB)); ST.zero_line(ax); ax.set_xlabel("Cliff's δ (A − B), leaf-on 2022"); ax.set_title("Separability before the breach", fontsize=8); ST.panel_label(ax, "a")
    # (b,c,d) trajectories
    for ax, v, letter in ((axes[0, 1], "NDVI", "b"), (axes[0, 2], "MNDWI", "c"), (axes[1, 0], "AWEIsh", "d")):
        for dom, col, lbl in (("A_reservoir_bed", CA, "A reservoir bed"), ("B_floodplain", CB, "B floodplain")):
            t = F[(F.domain == dom) & (F["index"] == v)].sort_values("year")
            ax.plot(t.year, t.p50, "-o", color=col, ms=3, label=lbl); ax.fill_between(t.year, t.p25, t.p75, color=col, alpha=0.18, lw=0)
        ax.axvline(2023.4, color=ST.C["bad"], ls="--", lw=0.8); ax.set_xticks(years); ax.tick_params(axis="x", labelsize=6.5); ax.set_title(f"{v} (leaf-on median ± IQR)", fontsize=8); ST.panel_label(ax, letter)
        if letter == "b":
            ax.legend(fontsize=6.5, loc="lower right"); ax.text(2023.45, ax.get_ylim()[1], "breach", color=ST.C["bad"], fontsize=6.5, va="top")
    # (e) water share
    ax = axes[1, 1]
    for dom, col, lbl in (("A_reservoir_bed", CA, "A"), ("B_floodplain", CB, "B")):
        t = F[(F.domain == dom) & (F["index"] == "water_share")].sort_values("year"); ax.plot(t.year, t.p50, "-o", color=col, ms=3, label=lbl); ax.fill_between(t.year, t.p25, t.p75, color=col, alpha=0.18, lw=0)
    ax.axvline(2023.4, color=ST.C["bad"], ls="--", lw=0.8); ax.set_xticks(years); ax.tick_params(axis="x", labelsize=6.5); ax.set_ylabel("water share, % of dates", fontsize=7.5); ax.set_title("Leaf-on water share", fontsize=8); ST.panel_label(ax, "e")
    # (f) Manning n by state
    ax = axes[1, 2]; P = pd.read_csv(CFG.TABLES / "p43_class_areas_pool.csv"); P = P[P.state.isin(STATES)].set_index("state").loc[STATES]
    Bf = pd.read_csv(CFG.TABLES / "p43_class_areas_below_dam_floodplain.csv")
    x = np.arange(len(STATES)); ax.plot(x, P.n_base_area_weighted, "-o", color=CA, ms=3, label="A reservoir bed")
    for grid, ls_, lbl in (("zone4", "-", "B floodplain (dam–Kherson)"), ("zone2", "--", "B floodplain (delta)")):
        b = Bf[(Bf.grid == grid) & Bf.state.isin(STATES)].set_index("state").reindex(STATES); ax.plot(x, b.n_base_area_weighted, ls_, marker="s", color=CB, ms=3, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(["Jun\n2023", "Jul–Sep\n2023", "2024", "2025", "2026"], fontsize=6.5); ax.set_ylabel("area-weighted Manning n", fontsize=7.5); ax.set_title("Roughness by state", fontsize=8); ax.legend(fontsize=5.5, loc="center right"); ST.panel_label(ax, "f")
    fig.tight_layout(w_pad=1.2); ST.save(fig, "fig02_two_domains", OUT); print("fig02")


# ----------------------------------------------------------------------------- fig 3
def fig03(A, B, dam, kh):
    vars_ = ("NDVI", "MNDWI", "AWEIsh", "class_mode"); years = (2022, 2026); ex = sys_extent(A, B)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), layout="constrained")
    ims = {}
    for i, y in enumerate(years):
        for j, v in enumerate(vars_):
            ax = axes[i, j]
            for z, geom in ((Z1, A), (Z2, B), (Z4, B)):
                f = ROOT / f"outputs/rasters/zone{ZN[z]}/annual/zone{ZN[z]}_{v}_{y}_leafon_20m.tif"
                if not f.exists():
                    continue
                if v == "class_mode":
                    a, tr, ext = dec(f, 8, nodata_extra=(0,)); a = mask_to(geom, a, tr); cm_, nm_ = M.class_cmap(); ims[v] = ax.imshow(a, extent=ext, cmap=cm_, norm=nm_, interpolation="nearest", zorder=2)
                else:
                    a, tr, ext = dec(f, 8, scale=1e-4); a = mask_to(geom, a, tr); sc = M.INDEX_SCALE[v]; ims[v] = ax.imshow(a, extent=ext, cmap=sc["cmap"], vmin=sc["vmin"], vmax=sc["vmax"], interpolation="nearest", zorder=2)
            outlines(ax, A, B, dam, kh, lw=0.4); ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{v.replace('class_mode', 'surface class')} · {y}", fontsize=8); ST.panel_label(ax, "abcdefgh"[i * 4 + j])
    for j, v in enumerate(vars_[:3]):
        cb = fig.colorbar(ims[v], ax=axes[:, j], orientation="horizontal", shrink=0.85, pad=0.02, aspect=28); cb.ax.tick_params(labelsize=6); cb.set_label(v, fontsize=7)
    cbc = fig.colorbar(ims["class_mode"], ax=axes[:, 3], orientation="horizontal", shrink=0.85, pad=0.02, aspect=28, boundaries=np.arange(0.5, 8.5), values=list(range(1, 8)))
    cbc.set_ticks(range(1, 8)); cbc.set_ticklabels(["water", "shallow /\nmixed", "wet\nsediment", "bare\nsediment", "sparse\nherb.", "dense\nherb.", "reed /\nflooded veg."]); cbc.ax.tick_params(labelsize=5, length=0); cbc.set_label("surface class (S2 rule set)", fontsize=7)
    ST.scale_bar(axes[1, 3], 50, loc="lower right")
    fig.suptitle("Optical state of both domains: last normal-regime year (2022) vs current (2026), leaf-on medians", fontsize=8.5)
    ST.save(fig, "fig03_indices_2022_2026", OUT); print("fig03")


# ----------------------------------------------------------------------------- fig 4
def fig04(A, B, dam, kh):
    Mt = pd.read_csv(CFG.TABLES / "p51b_common_support_metrics.csv"); I = pd.read_csv(CFG.TABLES / "p51b_permutation_importance.csv")
    fig = plt.figure(figsize=(7.2, 6.4)); gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1))
    bx = B.bounds; exB = (bx[0] - 2000, bx[2] + 2000, bx[1] - 2000, bx[3] + 2000)
    for k, (tag, title, letter) in enumerate((("_post_delta", "Observation-constrained (post-breach predictors only)", "a"), ("", "Prior-assisted reconstruction (all predictors)", "b"))):
        ax = fig.add_subplot(gs[0, k]); hillshade(ax, (Z4, Z2), k=6, alpha=0.4, extent=exB)
        for z in (Z2, Z4):
            d, dtr, dext = dec(FLOOD / z / f"{z}_domain_mask.tif", 2); ax.imshow(np.where(d == 1, 1.0, np.nan), extent=dext, cmap=ListedColormap(["#e5e5e5"]), vmin=0, vmax=1, interpolation="nearest", zorder=2)
            c, ctr, cext = dec(FLOOD / z / f"{z}_rf_flood_class{tag}_20m.tif", 2); c = mask_to(B, c, ctr)
            sup, _, _ = dec(FLOOD / z / f"{z}_rf_flood_prob_post_delta_20m.tif", 2)          # 255 = no post-breach optics
            nosup = np.isnan(sup)
            if tag == "":
                ax.imshow(np.where((c == 1) & ~nosup, 1.0, np.nan), extent=cext, cmap=ListedColormap(["#08519c"]), vmin=0, vmax=1, interpolation="nearest", zorder=3)
                ax.imshow(np.where((c == 1) & nosup, 1.0, np.nan), extent=cext, cmap=ListedColormap(["#9ecae1"]), vmin=0, vmax=1, interpolation="nearest", zorder=3)
            else:
                ax.imshow(np.where(nosup & (d == 1), 1.0, np.nan), extent=dext, cmap=ListedColormap(["#bdbdbd"]), vmin=0, vmax=1, interpolation="nearest", zorder=3)
                ax.imshow(np.where(c == 1, 1.0, np.nan), extent=cext, cmap=ListedColormap(["#08519c"]), vmin=0, vmax=1, interpolation="nearest", zorder=4)
        outlines(ax, A, B, dam, kh, lw=0.5); furniture(ax, exB, 20, arrow=(k == 1), arrow_loc=(0.06, 0.9)); ax.set_title(title, fontsize=8); ST.panel_label(ax, letter)
        if k == 0:
            ax.legend(handles=[Patch(fc="#08519c", label=f"flooded, observed ({pd.read_csv(CFG.TABLES / 'p51_rf_areas_post_delta.csv').rf_flood_km2.sum():.0f} km²)"), Patch(fc="#bdbdbd", label="no usable post-breach optics"), Patch(fc="#e5e5e5", label="domain, not flooded")], loc="upper center", fontsize=6, bbox_to_anchor=(0.5, -0.16), ncol=2)
        else:
            ax.legend(handles=[Patch(fc="#08519c", label="flooded, observed"), Patch(fc="#9ecae1", label="flooded, filled by learned prior"), Patch(fc="#e5e5e5", label="domain, not flooded")], loc="upper center", fontsize=6, bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax = fig.add_subplot(gs[1, 0]); t = Mt[Mt.support == "S1_ge1_post_date"].set_index("variant").loc[["all", "post_delta", "pre_only"]]; t2 = Mt[Mt.support == "S2_ge2_post_dates"].set_index("variant").loc[["all", "post_delta", "pre_only"]]
    x = np.arange(3); w = 0.36; ax.bar(x - w / 2, t.AP, w, color="#2171b5", label=f"≥ 1 post-breach date (n = {int(t.n.iloc[0]):,})"); ax.bar(x + w / 2, t2.AP, w, color="#e6550d", label=f"≥ 2 dates (n = {int(t2.n.iloc[0]):,})")
    for i in range(3):
        ax.text(x[i] - w / 2, t.AP.iloc[i] + 0.01, f"AUC {t.AUC.iloc[i]:.3f}", ha="center", fontsize=5.5, rotation=90, va="bottom"); ax.text(x[i] + w / 2, t2.AP.iloc[i] + 0.01, f"{t2.AUC.iloc[i]:.3f}", ha="center", fontsize=5.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(["all\npredictors", "post-breach\nonly", "pre-breach\nonly"], fontsize=7); ax.set_ylabel("average precision (OOF, 5-km blocks)"); ax.set_ylim(0, 1.32); ax.legend(fontsize=6, loc="upper right"); ax.set_title("Skill on the same cells (common support)"); ST.panel_label(ax, "c")
    ax = fig.add_subplot(gs[1, 1]); ia = I[I.variant == "all"].pivot(index="index", columns="epoch", values="AP_drop_mean").fillna(0)[["post", "delta", "pre"]].sort_values("post", ascending=False)
    ia.plot.bar(ax=ax, stacked=True, color=["#2171b5", "#74c476", "#969696"], width=0.7); ax.set_ylabel("drop in OOF AP when permuted"); ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=45, labelsize=7); ax.legend(title="predictor epoch", fontsize=6, title_fontsize=6.5); ax.set_title("Grouped permutation importance (all predictors)"); ST.panel_label(ax, "d")
    fig.tight_layout(); ST.save(fig, "fig04_flood_rf", OUT); print("fig04")


# ----------------------------------------------------------------------------- fig 5
def fig05():
    A_ = pd.read_csv(CFG.TABLES / "p51_rf_areas.csv"); Ap = pd.read_csv(CFG.TABLES / "p51_rf_areas_post_delta.csv"); prov = pd.read_csv(CFG.TABLES / "p42_domain_provenance.csv")
    rf_all = float(A_.rf_flood_km2.sum()); rf_obs = float(Ap.rf_flood_km2.sum()); cov = float(Ap.predicted_km2.sum() / Ap.domain_km2.sum()); pers = float(prov.persistent_ge2_km2.sum() + prov.optical_s2_km2.sum())
    rows = [("Lischenko & Filipovych 2024 — Landsat, 3 days after the breach", 620.3, "observation-based estimate"),
            ("Kadam et al. 2024 — HEC-RAS 1D", 681, "hydraulic model scenario"), ("Kadam et al. 2024 — HEC-RAS 2D, 300 m breach", 823, "hydraulic model scenario"), ("Kadam et al. 2024 — HEC-RAS 2D, 600 m breach", 873, "hydraulic model scenario"),
            ("This study — RF, prior-assisted reconstruction (all predictors)", rf_all, "RF reconstruction with learned prior"), (f"This study — RF, observation-constrained ({cov:.0%} of domain covered)", rf_obs, "post-breach-observation-constrained RF"),
            ("This study — S1 persistent (≥ 2 events) ∪ S2 optical", pers, "persistence product (peak removed)")]
    col = {"observation-based estimate": "#1a9850", "hydraulic model scenario": "#fdae61", "RF reconstruction with learned prior": "#2166ac", "post-breach-observation-constrained RF": "#67a9cf", "persistence product (peak removed)": "#999999"}
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[col[r[2]] for r in rows], edgecolor="k", lw=0.3)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 8, yi, f"{r[1]:.0f} km²", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.8); ax.set_xlabel("flood-affected area below the Kakhovka dam, km²"); ax.set_xlim(0, 1000)
    ax.legend(handles=[Patch(fc=c, ec="k", lw=0.3, label=k) for k, c in col.items()], fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.set_title("Flood-area estimates are different quantities: agreement in magnitude is not validation", fontsize=8.5)
    fig.tight_layout(); ST.save(fig, "fig05_flood_area_compare", OUT); print("fig05")


# ----------------------------------------------------------------------------- fig 6
def fig06(A):
    P = pd.read_csv(CFG.TABLES / "p43_class_areas_pool.csv"); P = P[P.state.isin(STATES)].set_index("state").loc[STATES]
    groups = {"open / intermittent water": ["open_water_km2", "intermittent_water_km2"], "wet sediment": ["wet_sediment_km2"], "bare sand / silt": ["bare_sand_silt_km2"], "herbaceous (sparse + dense)": ["sparse_herbaceous_km2", "dense_herbaceous_km2"],
              "reed / tall herb": ["reed_tall_herb_km2"], "young woody, sparse": ["young_woody_sparse_km2"], "young woody, dense": ["young_woody_dense_km2"], "legacy woody / built": ["mature_woody_legacy_km2", "built_km2"]}
    gcol = ["#1b6ca8", "#8c6d31", "#d9c58b", "#b8d98d", "#1e6f3f", "#a1d99b", "#238b45", "#7d7d7d"]
    fig = plt.figure(figsize=(7.2, 6.8)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0]); bottom = np.zeros(len(STATES)); x = np.arange(len(STATES))
    for (lbl, cols), c in zip(groups.items(), gcol):
        v = P[cols].sum(1).values; ax.bar(x, v, 0.7, bottom=bottom, color=c, label=lbl, edgecolor="white", lw=0.3); bottom += v
    ax.set_xticks(x); ax.set_xticklabels([STATE_LBL[s] for s in STATES], fontsize=6.5); ax.set_ylabel("area, km²"); ax.set_title("Surface classes by state", fontsize=8); ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, frameon=False); ST.panel_label(ax, "a")
    ax2 = ax.twinx(); ax2.plot(x, P.n_base_area_weighted, "-o", color="k", ms=3); ax2.set_ylabel("area-weighted n"); ax2.grid(False)
    # (b) water occurrence 2023-06-20..2024-09-26
    ax = fig.add_subplot(gs[0, 1]); a, tr, ext = dec(RO / "pool/zone1_water_occurrence_class_LISCH_water3_20m.tif", 8, nodata_extra=(0,)); a = mask_to(A, a, tr)
    cm = ListedColormap(["#f7f7f7", "#c6dbef", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]); im = ax.imshow(a, extent=ext, cmap=cm, vmin=0.5, vmax=6.5, interpolation="nearest", zorder=2)
    gpd.GeoSeries([A], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color=CA, lw=0.5, zorder=5); bx = A.bounds; furniture(ax, (bx[0] - 3000, bx[2] + 3000, bx[1] - 3000, bx[3] + 3000), 25)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=range(1, 7)); cb.ax.set_yticklabels(["never", "1–25 %", "25–50 %", "50–75 %", "75–99 %", "always"], fontsize=6); ax.set_title("Water occurrence, Jun 2023 – Sep 2024", fontsize=8); ST.panel_label(ax, "b")
    # (c) ICESat-2 canopy
    ax = fig.add_subplot(gs[1, 0]); T = pd.read_csv(CFG.TABLES / "p47_pool_canopy_by_year_icesat2_lowcnf.csv"); T = T[T.recession_zone == "ALL"].sort_values("year")
    xs = np.arange(len(T)); ax.bar(xs, 100 * T.share_canopy_ge2m, 0.5, color="#a1d99b", label="segments with canopy ≥ 2 m, %"); ax.set_ylabel("share of QC'd segments, %"); ax.set_xticks(xs); ax.set_xticklabels([f"{y}\nn = {n/1000:.1f} k" for y, n in zip(T.year, T.n_segments)], fontsize=6.5)
    ax2 = ax.twinx(); ax2.errorbar(xs, T.h_canopy_p50, yerr=[T.h_canopy_p50 - T.h_canopy_p25, T.h_canopy_p75 - T.h_canopy_p50], fmt="o-", color=ST.C["icesat"], ms=3, capsize=2, label="canopy height p50 (IQR), m"); ax2.set_ylabel("h_canopy, m above ground"); ax2.grid(False)
    ax.text(0, 100 * T.share_canopy_ge2m.iloc[0] + 0.4, "noise floor", fontsize=6, ha="center", color="#555")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left"); ax.set_title("ICESat-2 canopy on the drained bed", fontsize=8); ST.panel_label(ax, "c")
    # (d) re-flooded spring 2024
    ax = fig.add_subplot(gs[1, 1]); a, tr, ext = dec(RO / "pool/zone1_reflooded_spring2024_20m.tif", 8, nodata_extra=(255,)); a = mask_to(A, a, tr)
    ax.imshow(np.where(a == 0, 1.0, np.nan), extent=ext, cmap=ListedColormap(["#eeeeee"]), vmin=0, vmax=1, interpolation="nearest", zorder=2); ax.imshow(np.where(a == 1, 1.0, np.nan), extent=ext, cmap=ListedColormap(["#4292c6"]), vmin=0, vmax=1, interpolation="nearest", zorder=3)
    gpd.GeoSeries([A], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color=CA, lw=0.5, zorder=5); furniture(ax, (bx[0] - 3000, bx[2] + 3000, bx[1] - 3000, bx[3] + 3000), 25, arrow=False)
    ax.legend(handles=[Patch(fc="#4292c6", label="re-flooded Mar–May 2024 after exposure in 2023 (988 km²)"), Patch(fc="#eeeeee", label="stayed exposed")], fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 0.93)); ax.set_title("Re-flooding, spring 2024", fontsize=8); ST.panel_label(ax, "d")
    ST.save(fig, "fig06_pool_succession", OUT); print("fig06")


# ----------------------------------------------------------------------------- fig 7
def fig07(A, B, dam, kh):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), layout="constrained"); bx = A.bounds; exA = (bx[0] - 3000, bx[2] + 3000, bx[1] - 3000, bx[3] + 3000)
    bb = B.bounds; exB = (bb[0] - 2000, bb[2] + 2000, bb[1] - 2000, bb[3] + 2000); vmin, vmax = 0.02, 0.16
    for ax, st, letter in zip(axes[0], ("BREACH_2023", "STATE_2024", "CURRENT_2026"), "abc"):
        a, tr, ext = dec(RO / f"pool/zone1_poolwin_manning_n_base_{st}.tif", 6); a = mask_to(A, a, tr)
        im = ax.imshow(a, extent=ext, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2); gpd.GeoSeries([A], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color="k", lw=0.4, zorder=5)
        furniture(ax, exA, 25, arrow=False); ax.set_title(f"Bed n · {STATE_LBL[st].replace(chr(10), ' ')}", fontsize=8); ST.panel_label(ax, letter)
    for ax, st, letter in zip(axes[1, 1:], ("BREACH_2023", "CURRENT_2026"), "ef"):
        for z in (2, 4):
            f = RO / f"below_dam_floodplain/zone{z}_manning_n_base_{st}.tif"
            if f.exists():
                a, tr, ext = dec(f, 3); a = mask_to(B, a, tr); im2 = ax.imshow(a, extent=ext, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2)
        outlines(ax, A, B, dam, kh, lw=0.4); furniture(ax, exB, 20, arrow=False); ax.set_title(f"Floodplain n · {STATE_LBL[st].replace(chr(10), ' ')}", fontsize=8); ST.panel_label(ax, letter)
    cb = fig.colorbar(im, ax=[*axes[0], *axes[1, 1:]], orientation="vertical", shrink=0.6, pad=0.01, aspect=30); cb.set_label("Manning n (base)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax = axes[1, 0]; a, tr, ext = dec(RO / "pool/zone1_poolwin_dn_base_BREACH_2023_CURRENT_2026.tif", 6); a = mask_to(A, a, tr)
    im3 = ax.imshow(a, extent=ext, cmap="RdBu_r", vmin=-0.1, vmax=0.1, interpolation="nearest", zorder=2); gpd.GeoSeries([A], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color="k", lw=0.4, zorder=5); furniture(ax, exA, 25, arrow=False)
    cb3 = fig.colorbar(im3, ax=ax, orientation="horizontal", shrink=0.8, pad=0.03, aspect=25); cb3.set_label("Δn, Jun 2023 → 2026", fontsize=6.5); cb3.ax.tick_params(labelsize=6)
    ax.set_title("Bed: change in n", fontsize=8); ST.panel_label(ax, "d")
    ST.save(fig, "fig07_manning_maps", OUT); print("fig07")


# ----------------------------------------------------------------------------- fig 8
def fig08(A, B, dam, kh):
    """FABDEM elevation as the background of the flood reconstruction, with a zoom on the left bank south of Kherson (user 2026-09-19)."""
    from rasterio.warp import reproject
    from rasterio.transform import from_origin
    bb = B.bounds; exB = (bb[0] - 2000, bb[2] + 2000, bb[1] - 2000, bb[3] + 2000)
    zoom = (462e3, 482e3, 5146e3, 5166e3)          # (x0, x1, y0, y1) like every extent here
    def dem_mosaic(extent, k):
        x0, x1, y0, y1 = extent; cell = 20.0 * k; nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1; tr = from_origin(x0, y1, cell, cell); dem = np.full((ny, nx), np.nan, "f4")
        for z in (Z4, Z2):
            with rasterio.open(TERR / z / "fabdem_20m.tif") as ds:
                dst = np.full((ny, nx), np.nan, "f4"); reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=tr, dst_crs=ds.crs, resampling=Resampling.average, dst_nodata=np.nan)
            dst[dst == -9999] = np.nan; dem = np.where(np.isnan(dem), dst, dem)
        return dem, (x0, x0 + nx * cell, y1 - ny * cell, y1)
    def flood_layers(ax, k):
        for z in (Z2, Z4):
            c, ctr, cext = dec(FLOOD / z / f"{z}_rf_flood_class_20m.tif", k); c = mask_to(B, c, ctr)
            sup, _, _ = dec(FLOOD / z / f"{z}_rf_flood_prob_post_delta_20m.tif", k); nosup = np.isnan(sup)
            ax.imshow(np.where((c == 1) & ~nosup, 1.0, np.nan), extent=cext, cmap=ListedColormap(["#08306b"]), vmin=0, vmax=1, interpolation="nearest", zorder=3, alpha=0.9)
            ax.imshow(np.where((c == 1) & nosup, 1.0, np.nan), extent=cext, cmap=ListedColormap(["#6baed6"]), vmin=0, vmax=1, interpolation="nearest", zorder=3, alpha=0.9)
    fig = plt.figure(figsize=(7.2, 6.6)); gs = fig.add_gridspec(2, 2, height_ratios=(1.05, 1), width_ratios=(1, 1))
    ax = fig.add_subplot(gs[0, :]); dem, dext = dem_mosaic(exB, 3); ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(np.where(np.isfinite(dem), np.clip(dem, 0, 20), 0), cmap=plt.get_cmap("terrain"), vmin=0, vmax=20, blend_mode="soft", vert_exag=4, dx=60, dy=60)
    rgb[..., 3] = np.where(np.isfinite(dem), 1.0, 0.0); ax.imshow(rgb, extent=dext, interpolation="bilinear", zorder=1)
    flood_layers(ax, 2); outlines(ax, A, B, dam, kh, lw=0.5); furniture(ax, exB, 20, arrow_loc=(0.06, 0.9))
    ax.add_patch(matplotlib.patches.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, ec="k", lw=0.9, ls="--", zorder=8))
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(0, 20)); cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01); cb.set_label("FABDEM elevation, m (EGM2008), clipped at 20 m", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax.legend(handles=[Patch(fc="#08306b", label="flooded June 2023 — observed (post-breach optics)"), Patch(fc="#6baed6", label="flooded — filled by the learned prior (no post-breach optics)"), plt.Line2D([], [], color=CB, lw=0.8, label="floodplain domain (HAND < h₀(x))"), plt.Line2D([], [], color="k", ls="--", lw=0.8, label="zoom (panel b)")], fontsize=6, loc="lower left"); ax.set_title("June-2023 flood reconstruction on the FABDEM terrain", fontsize=8.5); ST.panel_label(ax, "a")
    ax = fig.add_subplot(gs[1, 0]); dem, dext = dem_mosaic(zoom, 1)
    ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
    try:                                                                  # satellite basemap (user 2026-09-19): Esri World Imagery tiles via contextily
        import contextily as cx
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs=CFG.CRS_METRIC, zoom=12, attribution=False, zorder=1)
        ax.text(0.01, 0.01, "Basemap: Esri World Imagery", transform=ax.transAxes, fontsize=4.5, color="white", zorder=9)
    except Exception as e:                                                # offline: fall back to the shaded DEM
        print("basemap unavailable:", str(e)[:60])
        rgb = ls.shade(np.where(np.isfinite(dem), np.clip(dem, 0, 12), 0), cmap=plt.get_cmap("terrain"), vmin=0, vmax=12, blend_mode="soft", vert_exag=6, dx=20, dy=20); rgb[..., 3] = np.where(np.isfinite(dem), 1.0, 0.0)
        ax.imshow(rgb, extent=dext, interpolation="bilinear", zorder=1)
    flood_layers(ax, 1)
    ax.contour(np.linspace(dext[0], dext[1], dem.shape[1]), np.linspace(dext[3], dext[2], dem.shape[0]), np.where(np.isfinite(dem), dem, 99), levels=[5.6], colors="red", linewidths=0.7, zorder=6)
    gpd.GeoSeries([B], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color="yellow", lw=0.7, zorder=6); ax.plot(kh.x, kh.y, marker=ST.MARKERS["gauge"], color="white", mec="k", ms=5, zorder=8)
    furniture(ax, zoom, 5, arrow=False)
    ax.legend(handles=[Patch(fc="#08306b", label="flooded, observed"), Patch(fc="#6baed6", label="flooded, prior-filled"), plt.Line2D([], [], color="yellow", lw=0.8, label="floodplain domain"), plt.Line2D([], [], color="red", lw=0.8, label="5.6 m contour (Kherson peak stage, 8 Jun 2023)")], fontsize=5, loc="upper right"); ax.set_title("Left bank south of Kherson on satellite imagery", fontsize=8); ST.panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, 1]); 
    for z, col, lbl in ((Z4, "#2171b5", "dam–Kherson reach"), (Z2, "#e6550d", "delta")):
        d, dtr, dext2 = dec(FLOOD / z / f"{z}_domain_mask.tif", 2); c, _, _ = dec(FLOOD / z / f"{z}_rf_flood_class_20m.tif", 2)
        with rasterio.open(TERR / z / "fabdem_20m.tif") as ds:
            e = ds.read(1, out_shape=d.shape, resampling=Resampling.average).astype("f4"); e[e == ds.nodata] = np.nan
        ind = (d == 1) & np.isfinite(e); fl = ind & (c == 1)
        bins = np.arange(-2, 22, 1.0); ax.hist(e[ind], bins, histtype="step", color=col, lw=1.0, label=f"domain, {lbl}"); ax.hist(e[fl], bins, color=col, alpha=0.35, lw=0, label=f"RF flooded, {lbl}")
    ax.axvline(5.56, color="red", lw=0.8, ls="--"); ax.text(5.8, ax.get_ylim()[1] * 0.9, "Kherson peak\n5.56 m", color="red", fontsize=6); ax.set_xlabel("FABDEM elevation, m"); ax.set_ylabel("cells (×20 m ×2 decimation)"); ax.legend(fontsize=5.5); ax.set_title("Elevation of the domain and of the reconstructed flood", fontsize=8); ST.panel_label(ax, "c")
    fig.tight_layout(); ST.save(fig, "fig08_flood_on_dem", OUT); print("fig08")


def main():
    ST.use_style(); OUT.mkdir(parents=True, exist_ok=True)
    A = SD.load_utm("reservoir_full_pool_prebreach"); B = SD.load_utm("below_dam_floodplain"); dam, kh = points()
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8"]
    for w in which:
        {"1": lambda: fig01(A, B, dam, kh), "2": fig02, "3": lambda: fig03(A, B, dam, kh), "4": lambda: fig04(A, B, dam, kh), "5": fig05, "6": lambda: fig06(A), "7": lambda: fig07(A, B, dam, kh), "8": lambda: fig08(A, B, dam, kh)}[w]()
    print("->", OUT)


if __name__ == "__main__":
    main()
