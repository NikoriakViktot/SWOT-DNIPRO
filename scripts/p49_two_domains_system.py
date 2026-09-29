#!/usr/bin/env python
"""P49 -- the breach-and-flood system as ONE map, the two domains as separately demonstrated populations, joined with the optical indices.

User request 2026-09-19: (1) maps of the cut and the rectangles as they look now; (2) explain and PROVE that the reservoir bed (pool) and the
below-dam floodplain are two different study zones with different conditions; (3) one map of the whole breach + flood zone; (4) join with the
optical indices (AWEIsh, MNDWI, NDWI, NDVI, NDMI, BSI, NDTI).

Domains (registry only, no literals): A = `reservoir_full_pool_prebreach` (the pool that drained), B = `below_dam_floodplain` (the land that flooded).
Sources: p40 annual composites (zone1 for A; zone4, zone2 for B), zone*_water_frac_PRE_BREACH, WorldCover 2021 frames, p42 flood rasters
($BULK_ROOT/floodplain/<ZONE>/), p43 bed-recession zoning (A), p43 transition summary (Manning n).

Outputs
  outputs/figures/p49_system_map.png                     pool by recession zone + floodplain by June-2023 flood, dam, Kherson (no construction lines: user 2026-09-19)
  outputs/figures/p49_system_indices_<year>_leafon.png   all 7 indices + class mode over BOTH domains on one map (years: --years)
  outputs/figures/p49_domain_evidence.png                why two zones: pre-breach state, land cover, index trajectories, Manning n, separability
  outputs/tables/p49_domain_features_by_year.csv         per domain x year x index: n, p25/p50/p75
  outputs/tables/p49_domain_separability_<year>.csv      per index: KS, Cliff's delta, single-feature AUC (domain A vs B, leaf-on <year>)
  outputs/tables/p49_flood_event_areas.csv               cleaned S1 area per event and zone (comparator for peak flood areas in the literature)
Method notes: points are random inside each domain polygon (SEED 42; N per domain --n); sampling is nearest-cell on the 20 m composites; INT16
indices are x1e4; nodata never enters a statistic (NO_DATA is not a value).
"""
from __future__ import annotations

import argparse
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
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Point
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro.plotting import maps as M

IDX = ("NDVI", "MNDWI", "NDWI", "AWEIsh", "NDMI", "BSI", "NDTI")
ZF = {"ZONE_1": ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", 1), "ZONE_4": ("ZONE_4_DAM_TO_KHERSON_FLOODWAY", 4), "ZONE_2": ("ZONE_2_KHERSON_DELTA", 2)}
FLOOD = CFG.BULK_ROOT / "floodplain"
FIG = ROOT / "outputs/figures"
RECT_NOTE = "CUT_RECTS in p42_below_dam_floodplain.py"
WCN = {10: "trees", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare", 80: "water", 90: "wetland", 95: "mangrove", 0: "n/a"}


def annual(zkey: str, var: str, year: int, window: str = "leafon") -> Path:
    n = ZF[zkey][1]
    return ROOT / "outputs/rasters" / f"zone{n}" / "annual" / f"zone{n}_{var}_{year}_{window}_20m.tif"


def sample(path: Path, xs: np.ndarray, ys: np.ndarray, scale: float | None = None, nodata_extra=()) -> np.ndarray:
    """Nearest-cell values at points (NaN outside/nodata)."""
    out = np.full(len(xs), np.nan)
    if not path.exists():
        return out
    with rasterio.open(path) as ds:
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys)
        r = np.asarray(r); c = np.asarray(c)
        ok = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width)
        if not ok.any():
            return out
        a = ds.read(1)
        v = a[r[ok], c[ok]].astype("f8")
        bad = np.zeros(v.shape, bool)
        if ds.nodata is not None:
            bad |= v == ds.nodata
        for e in nodata_extra:
            bad |= v == e
        v[bad] = np.nan
        out[ok] = v * (scale or 1.0)
    return out


def random_points(geom, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); x0, y0, x1, y1 = geom.bounds; xs, ys = [], []
    import shapely
    while len(xs) < n:
        px = rng.uniform(x0, x1, 4 * n); py = rng.uniform(y0, y1, 4 * n)
        m = shapely.contains_xy(geom, px, py); xs.extend(px[m]); ys.extend(py[m])
    return np.array(xs[:n]), np.array(ys[:n])


def decimated(path: Path, k: int, scale=None, nodata_extra=()):
    with rasterio.open(path) as ds:
        h, w = ds.height // k, ds.width // k
        a = ds.read(1, out_shape=(h, w), resampling=Resampling.nearest).astype("f4")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        for e in nodata_extra:
            a[a == e] = np.nan
        tr = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        ext = [tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f]
    return a * (scale or 1.0), tr, ext


def domain_mask(geom, shape, tr) -> np.ndarray:
    return features.rasterize([(geom, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)


def draw_domain_layers(ax, var: str, year: int, A, B, window: str = "leafon", k: int = 8):
    """The composite of var/year over BOTH domains on one axes (A from zone1, B from zone4 then zone2 where zone4 has no coverage)."""
    kind = "idx" if var in IDX else "class" if var == "class_mode" else "share"
    for zkey, geom in (("ZONE_1", A), ("ZONE_2", B), ("ZONE_4", B)):   # zone4 drawn last (main floodplain); zone2 first (delta)
        f = annual(zkey, var, year, window)
        if not f.exists():
            continue
        a, tr, ext = decimated(f, k, scale=1e-4 if kind == "idx" else None, nodata_extra=(0,) if kind == "class" else (255,) if kind == "share" else ())
        m = domain_mask(geom, a.shape, tr); a = np.where(m, a, np.nan)
        if kind == "idx":
            sc = M.INDEX_SCALE[var]; ax.imshow(a, extent=ext, cmap=sc["cmap"], vmin=sc["vmin"], vmax=sc["vmax"], interpolation="nearest", zorder=2)
        elif kind == "class":
            cm_, nm_ = M.class_cmap(); ax.imshow(a, extent=ext, cmap=cm_, norm=nm_, interpolation="nearest", zorder=2)
        else:
            ax.imshow(a, extent=ext, cmap="Blues", vmin=0, vmax=100, interpolation="nearest", zorder=2)
    return kind


def outlines(ax, A, B, dam, kh, rects=True):
    gpd.GeoSeries([A], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color="#1f4e79", lw=0.7, zorder=5)
    gpd.GeoSeries([B], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color="#7a3b00", lw=0.7, zorder=5)
    ax.plot(dam.x, dam.y, "k^", ms=7, zorder=6); ax.plot(kh.x, kh.y, "ks", ms=5, zorder=6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def system_extent(A, B):
    x0 = min(A.bounds[0], B.bounds[0]); y0 = min(A.bounds[1], B.bounds[1]); x1 = max(A.bounds[2], B.bounds[2]); y1 = max(A.bounds[3], B.bounds[3])
    return x0 - 3000, x1 + 3000, y0 - 3000, y1 + 3000


def cut_rects():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p42", ROOT / "scripts/p42_below_dam_floodplain.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return list(getattr(m, "CUT_RECTS", []))


def fig_system_map(A, B, dam, kh):
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ex = system_extent(A, B); ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3])
    # A: bed recession zoning (2023), inside the pool polygon
    z = ROOT / "outputs/rasters/roughness/pool/zone1_poolwin_bed_recession_zoning_2023.tif"
    a, tr, ext = decimated(z, 6, nodata_extra=(0,)); a = np.where(domain_mask(A, a.shape, tr), a, np.nan)
    zc = ListedColormap(["#08306b", "#2171b5", "#9ecae1", "#fdd49e"])   # fast / mid / slow / never -> dark = drained first? colours below
    zc = ListedColormap(["#fee08b", "#fdae61", "#d73027", "#08519c"])   # fast, mid, slow (exposure sooner = paler) ; never = water
    ax.imshow(a, extent=ext, cmap=zc, vmin=0.5, vmax=4.5, interpolation="nearest", zorder=2)
    # B: June-2023 flood (clean envelope) on the floodplain domain
    ax.add_patch(matplotlib.patches.PathPatch(matplotlib.path.Path([(0, 0)]), alpha=0)) if False else None
    for zkey in ("ZONE_2", "ZONE_4"):
        zn = ZF[zkey][0]; f = FLOOD / zn / f"{zn}_flood_envelope_clean.tif"; dm = FLOOD / zn / f"{zn}_domain_mask.tif"
        if dm.exists():
            d, dtr, dext = decimated(dm, 3); ax.imshow(np.where(d == 1, 1.0, np.nan), extent=dext, cmap=ListedColormap(["#c7e9c0"]), vmin=0, vmax=1, interpolation="nearest", zorder=2)
        if f.exists():
            d, dtr, dext = decimated(f, 3, nodata_extra=(255,)); inside = domain_mask(B, d.shape, dtr)
            ax.imshow(np.where((d == 1) & inside, 1.0, np.nan), extent=dext, cmap=ListedColormap(["#2b8cbe"]), vmin=0, vmax=1, interpolation="nearest", zorder=3)
    outlines(ax, A, B, dam, kh)
    ax.annotate("Kakhovka dam", (dam.x, dam.y), xytext=(6, 8), textcoords="offset points", fontsize=9, zorder=8); ax.annotate("Kherson", (kh.x, kh.y), xytext=(6, -12), textcoords="offset points", fontsize=9, zorder=8)
    ax.text(0.55, 0.90, "A · reservoir bed (drained)\ncolour = when the bed left the water (2023 recession)", transform=ax.transAxes, fontsize=10, color="#1f4e79", weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85), zorder=9)
    ax.text(0.02, 0.30, "B · below-dam floodplain (flooded, then receded)\nblue = June-2023 flood (S1 ≥ 2 events ∪ S2), green = domain", transform=ax.transAxes, fontsize=10, color="#7a3b00", weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85), zorder=9)
    ax.legend(handles=[Patch(fc="#fee08b", label="A: fast (≤ 06-30)"), Patch(fc="#fdae61", label="A: mid (≤ 08-06)"), Patch(fc="#d73027", label="A: slow (≤ 09-08)"), Patch(fc="#08519c", label="A: never exposed 2023 (water/channel)"),
                       Patch(fc="#c7e9c0", label="B: floodplain domain"), Patch(fc="#2b8cbe", label="B: June-2023 flood")], loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title("The Kakhovka breach as one system: the reservoir bed that drained (A) and the floodplain that flooded (B) — same event, different processes (EPSG:32636)", fontsize=12)
    out = FIG / "p49_system_map.png"; fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); print("->", out)


def fig_indices(A, B, dam, kh, year: int):
    vars_ = IDX + ("class_mode",)
    fig, axes = plt.subplots(2, 4, figsize=(26, 11.5))
    ex = system_extent(A, B)
    for ax, v in zip(axes.ravel(), vars_):
        ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3]); kind = draw_domain_layers(ax, v, year, A, B); outlines(ax, A, B, dam, kh)
        if kind == "idx":
            sc = M.INDEX_SCALE[v]; sm = plt.cm.ScalarMappable(cmap=sc["cmap"], norm=plt.Normalize(sc["vmin"], sc["vmax"])); plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        ax.set_title(f"{v} · leaf-on {year}", fontsize=11)
    fig.suptitle(f"Optical indices over both domains, leaf-on {year} (20 m composites, decimated ×8 for the figure; outline blue = A reservoir bed, brown = B floodplain)", fontsize=13)
    fig.tight_layout(); out = FIG / f"p49_system_indices_{year}_leafon.png"; fig.savefig(out, dpi=90); plt.close(fig); print("->", out)


def points_table(A, B, n: int, years) -> pd.DataFrame:
    xa, ya = random_points(A, n); xb, yb = random_points(B, n, seed=43)
    zone4 = SD.load_utm(ZF["ZONE_4"][0]); rows = []
    for dom, xs, ys in (("A_reservoir_bed", xa, ya), ("B_floodplain", xb, yb)):
        d = pd.DataFrame(dict(domain=dom, x=xs, y=ys))
        if dom.startswith("A"):
            d["src"] = "ZONE_1"
        else:
            import shapely
            d["src"] = np.where(shapely.contains_xy(zone4, xs, ys), "ZONE_4", "ZONE_2")
        rows.append(d)
    P = pd.concat(rows, ignore_index=True)
    for y in years:
        for v in IDX:
            col = np.full(len(P), np.nan)
            for zk in ("ZONE_1", "ZONE_4", "ZONE_2"):
                m = (P.src == zk).values
                if m.any():
                    col[m] = sample(annual(zk, v, y), P.x.values[m], P.y.values[m], scale=1e-4)
            P[f"{v}_{y}"] = col
        ws = np.full(len(P), np.nan)
        for zk in ("ZONE_1", "ZONE_4", "ZONE_2"):
            m = (P.src == zk).values
            if m.any():
                ws[m] = sample(annual(zk, "water_share", y), P.x.values[m], P.y.values[m], nodata_extra=(255,))
        P[f"water_share_{y}"] = ws
    for zk in ("ZONE_1", "ZONE_4", "ZONE_2"):
        m = (P.src == zk).values
        if not m.any():
            continue
        n_ = ZF[zk][1]; zn = ZF[zk][0]
        P.loc[m, "pre_water_frac"] = sample(ROOT / f"outputs/rasters/zone{n_}/zone{n_}_water_frac_PRE_BREACH_20m.tif", P.x.values[m], P.y.values[m], nodata_extra=(255,))
        P.loc[m, "worldcover2021"] = sample(CFG.BULK_ROOT / "worldcover_frames" / zn / "wc_2021_20m.tif", P.x.values[m], P.y.values[m])
        if zk != "ZONE_1":
            P.loc[m, "hand_m"] = sample(FLOOD / zn / f"{zn}_hand_m.tif", P.x.values[m], P.y.values[m], nodata_extra=(-9999,))
            P.loc[m, "flood_n_events"] = sample(FLOOD / zn / f"{zn}_flood_n_events.tif", P.x.values[m], P.y.values[m], nodata_extra=(255,))
    ma = (P.domain == "A_reservoir_bed").values
    P.loc[ma, "recession_zone"] = sample(ROOT / "outputs/rasters/roughness/pool/zone1_poolwin_bed_recession_zoning_2023.tif", P.x.values[ma], P.y.values[ma], nodata_extra=(0,))
    return P


def separability(P: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    for v in IDX + ("water_share",):
        col = f"{v}_{year}" if v != "water_share" else f"water_share_{year}"
        a = P.loc[P.domain.str.startswith("A"), col].dropna().values; b = P.loc[P.domain.str.startswith("B"), col].dropna().values
        if len(a) < 50 or len(b) < 50:
            continue
        ks = stats.ks_2samp(a, b); u = stats.mannwhitneyu(a, b); auc = u.statistic / (len(a) * len(b)); delta = 2 * auc - 1
        rows.append(dict(feature=col, n_A=len(a), n_B=len(b), median_A=round(float(np.median(a)), 3), median_B=round(float(np.median(b)), 3), KS=round(float(ks.statistic), 3), cliffs_delta=round(float(delta), 3), AUC_A_gt_B=round(float(auc), 3)))
    return pd.DataFrame(rows)


def fig_evidence(P: pd.DataFrame, years, sep: pd.DataFrame, norm_year: int):
    cA, cB = "#1f4e79", "#7a3b00"
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    A = P[P.domain.str.startswith("A")]; B = P[P.domain.str.startswith("B")]
    ax = axes[0, 0]; bins = np.linspace(0, 100, 21)
    ax.hist(A.pre_water_frac.dropna(), bins, alpha=0.7, color=cA, label=f"A reservoir bed (n={A.pre_water_frac.notna().sum()})", density=True); ax.hist(B.pre_water_frac.dropna(), bins, alpha=0.7, color=cB, label=f"B floodplain (n={B.pre_water_frac.notna().sum()})", density=True)
    ax.set_xlabel("PRE-breach water frequency, % of observed dates"); ax.set_ylabel("density"); ax.set_title("(a) before the breach: A was open water, B was land"); ax.legend(fontsize=9)
    ax = axes[0, 1]; cats = [10, 20, 30, 40, 50, 60, 80, 90]
    sa = [np.mean(A.worldcover2021.dropna() == c) * 100 for c in cats]; sb = [np.mean(B.worldcover2021.dropna() == c) * 100 for c in cats]; x = np.arange(len(cats))
    ax.bar(x - 0.2, sa, 0.4, color=cA, label="A"); ax.bar(x + 0.2, sb, 0.4, color=cB, label="B"); ax.set_xticks(x); ax.set_xticklabels([WCN[c] for c in cats], rotation=30); ax.set_ylabel("% of points"); ax.set_title("(b) ESA WorldCover 2021 land cover"); ax.legend()
    ax = axes[0, 2]
    data, labels, cols = [], [], []
    for v in ("NDVI", "MNDWI", "BSI"):
        c = f"{v}_{norm_year}"
        for nm, D, col in (("A", A, cA), ("B", B, cB)):
            data.append(D[c].dropna().values); labels.append(f"{v}\n{nm}"); cols.append(col)
    bp = ax.boxplot(data, tick_labels=labels, whis=(5, 95), showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.5); ax.set_title(f"(c) leaf-on {norm_year} (last normal-regime year): index distributions"); ax.grid(alpha=0.3)
    for i, v in enumerate(("NDVI", "MNDWI", "AWEIsh")):
        ax = axes[1, i] if i < 2 else None
        if ax is None:
            continue
        for nm, D, col in (("A reservoir bed", A, cA), ("B floodplain", B, cB)):
            med = [D[f"{v}_{y}"].median() for y in years]; q1 = [D[f"{v}_{y}"].quantile(0.25) for y in years]; q3 = [D[f"{v}_{y}"].quantile(0.75) for y in years]
            ax.plot(years, med, "-o", color=col, label=nm); ax.fill_between(years, q1, q3, color=col, alpha=0.18)
        ax.axvline(2022.9, color="red", ls="--", lw=1); ax.text(2022.93, ax.get_ylim()[0], " breach 06-2023", color="red", fontsize=8, va="bottom")
        ax.set_title(f"({'de'[i]}) {v}, leaf-on median ± IQR"); ax.set_xticks(years); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax = axes[1, 2]; ax.axis("off")
    t = sep.copy(); t["feature"] = t.feature.str.replace(f"_{norm_year}", "")
    txt = f"(f) separability of A vs B, leaf-on {norm_year}\n\n" + t[["feature", "median_A", "median_B", "KS", "cliffs_delta"]].to_string(index=False)
    txt += "\n\nManning n (p43, area-weighted):\n  A bed: 0.034 (06-2023) → 0.044 → 0.062 → 0.083 → 0.086 (2026)\n  B floodplain: ≈ 0.080–0.083, no net change"
    txt += "\n\nProcess:\n  A = state change (water → bare → woody), monotone\n  B = transient inundation, returns to the pre-breach land use"
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=9)
    fig.suptitle("Why A (reservoir bed) and B (below-dam floodplain) are two study zones — and not one", fontsize=14)
    fig.tight_layout(); out = FIG / "p49_domain_evidence.png"; fig.savefig(out, dpi=100); plt.close(fig); print("->", out)


def flood_event_areas() -> pd.DataFrame:
    rows = []
    for zk in ("ZONE_4", "ZONE_2"):
        zn = ZF[zk][0]
        for f in sorted((FLOOD / zn).glob(f"{zn}_flood_extent_2023-*.tif")):
            with rasterio.open(f) as ds:
                a = ds.read(1)
            ev = f.stem.split("flood_extent_")[1]
            rows.append(dict(zone=zn, event=ev, area_km2=round(float((a == 1).sum()) * 0.0004, 1), observed_km2=round(float((a != 255).sum()) * 0.0004, 1), note="cleaned S1 water (HAND<h0(x), not pre-breach water, not Inhulets), single event"))
        with rasterio.open(FLOOD / zn / f"{zn}_flood_envelope_clean.tif") as ds:
            a = ds.read(1)
        rows.append(dict(zone=zn, event="PERSISTENT(>=2 S1)|S2 optical", area_km2=round(float((a == 1).sum()) * 0.0004, 1), observed_km2=np.nan, note="persistence removes the peak; not comparable with single-date peak areas"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--years", type=int, nargs="*", default=[2021, 2022, 2023, 2024, 2025, 2026]); ap.add_argument("--map-years", type=int, nargs="*", default=[2022, 2026]); ap.add_argument("--norm-year", type=int, default=2022)
    ap.add_argument("--skip-maps", action="store_true"); a = ap.parse_args()
    A = SD.load_utm("reservoir_full_pool_prebreach"); B = SD.load_utm("below_dam_floodplain")
    dam = gpd.GeoSeries([Point(*CFG.KAKHOVKA_DAM)], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]; kh = gpd.GeoSeries([Point(CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3])], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]
    ev = flood_event_areas(); ev.to_csv(CFG.TABLES / "p49_flood_event_areas.csv", index=False); print(ev.to_string(index=False))
    P = points_table(A, B, a.n, a.years)
    rows = []
    for dom, D in P.groupby("domain"):
        for y in a.years:
            for v in IDX + ("water_share",):
                c = f"{v}_{y}" if v != "water_share" else f"water_share_{y}"; s = D[c].dropna()
                rows.append(dict(domain=dom, year=y, index=v, n=len(s), p25=s.quantile(.25) if len(s) else np.nan, p50=s.median() if len(s) else np.nan, p75=s.quantile(.75) if len(s) else np.nan))
    pd.DataFrame(rows).round(4).to_csv(CFG.TABLES / "p49_domain_features_by_year.csv", index=False)
    sep = separability(P, a.norm_year); sep.to_csv(CFG.TABLES / f"p49_domain_separability_{a.norm_year}.csv", index=False); print(sep.to_string(index=False))
    fig_evidence(P, a.years, sep, a.norm_year)
    if not a.skip_maps:
        fig_system_map(A, B, dam, kh)
        for y in a.map_years:
            fig_indices(A, B, dam, kh, y)


if __name__ == "__main__":
    main()
