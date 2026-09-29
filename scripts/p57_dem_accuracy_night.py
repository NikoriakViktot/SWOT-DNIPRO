#!/usr/bin/env python
"""P57 -- accuracy of the DEM products against NIGHT ICESat-2 ATL08 ground segments (user 2026-09-19): RMSE, MAE, bias, LE90 (and LE95, NMAD).

Reference: ATL08/PhoREAL terrain, solar_elevation < 0, gnd_ph_count >= 8, snowcover == 1, chain h_te + free2mean - zeta_EGG2015 + c
(kakhovka pull carries the chain; lower_dnipro / liman pulls get it on the fly, as p56). residual = DEM - ICESat-2, m EVRF2019.
Products
  A  FABDEM -> EVRF2019 (p56), land outside the former pool (frames ZONE_4 / ZONE_2 / ZONE_3), WorldCover != water
  B  reservoir bed DEM v1 (hist20) on the bed exposed 2023-07..2024-12 (dry at pass; p52 rule) -- the bed is the terrain now
  C  seamless DEM (p55 per-zone 20 m) on A + B points together, i.e. the product that goes to HEC-RAS
LE90 = 90th percentile of |residual|; LE95 likewise. Strata: all / by zone / by product / by WorldCover / low terrain (< 5 m) / by year.
Outputs: outputs/tables/p57_dem_accuracy_night.csv, outputs/figures/p57_dem_accuracy_night.png
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
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import vertical as VT

ATLD = CFG.BULK_ROOT / "data_swot/processed/atl08"; TERR = CFG.BULK_ROOT / "terrain"; SEAM = CFG.BULK_ROOT / "dem_seamless"; RO = ROOT / "outputs/rasters/roughness/pool"
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY")
WCN = {10: "trees", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare", 80: "water", 90: "wetland"}


def sample(path, xs, ys):
    with rasterio.open(path) as ds:
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys); r = np.asarray(r); c = np.asarray(c); ok = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width); a = ds.read(1); out = np.full(len(xs), np.nan)
        v = a[r[ok], c[ok]].astype("f8")
        if ds.nodata is not None:
            v[v == ds.nodata] = np.nan
        out[ok] = v
    return out


def metrics(r, label, **extra):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 20:
        return dict(set=label, N=len(r), **extra)
    a = np.abs(r)
    return dict(set=label, N=len(r), RMSE=round(float(np.sqrt((r ** 2).mean())), 3), MAE=round(float(a.mean()), 3), bias=round(float(r.mean()), 3), median=round(float(np.median(r)), 3),
                LE90=round(float(np.percentile(a, 90)), 3), LE95=round(float(np.percentile(a, 95)), 3), NMAD=round(float(1.4826 * np.median(a - np.median(r)) if False else 1.4826 * np.median(np.abs(r - np.median(r)))), 3), **extra)


def load_points():
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean()); tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)
    cols = ["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "H_ice", "in_former_pool", "pull"]
    t = pd.read_parquet(ATLD / "kakhovka_atl08_terrain.parquet", columns=["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "H_terrain_common_m", "in_former_pool"])
    t["date"] = pd.to_datetime(t.date); t["H_ice"] = t.H_terrain_common_m + c; t["pull"] = "kakhovka"; parts = [t[cols]]
    for name in ("lower_dnipro_atl08_20m_raw.parquet", "liman_atl08_20m_raw.parquet"):
        e = pd.read_parquet(ATLD / name, columns=["time", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "geometry"]); g = gpd.GeoSeries.from_wkb(e.pop("geometry")); e["lon"], e["lat"] = g.x.values, g.y.values
        e["date"] = pd.to_datetime(e.pop("time")).dt.tz_localize(None); e = e[(e.gnd_ph_count >= 8) & (e.h_te_median.abs() < 500)].copy()
        e["H_ice"] = e.h_te_median.values + CFG.free2mean(e.lat.values) - VT.sample_grid(CFG.EGG2015_TIF, e.lon.values, e.lat.values) + c; e["in_former_pool"] = False; e["pull"] = name.split("_atl08")[0]; parts.append(e[cols])
    P = pd.concat(parts, ignore_index=True); P = P[(P.solar_elevation < 0) & (P.gnd_ph_count >= 8) & (P.snowcover == 1) & (P.h_te_median.abs() < 500) & np.isfinite(P.H_ice)].copy()
    P["x"], P["y"] = tf.transform(P.lon.values, P.lat.values); P["year"] = P.date.dt.year
    return P, c


def main():
    P, c = load_points(); print(f"night QC segments: {len(P):,} (c_EGG2015->EVRF2019 {c:+.3f})", flush=True)
    # dry-at-pass flag for the pool (p52 rule)
    with rasterio.open(RO / "zone1_poolwin_bed_recession_zoning_2023.tif") as ds:
        zon = ds.read(1); tr = ds.transform
    cc, rr = ~tr * (P.x.values, P.y.values); cc = np.floor(cc).astype(int); rr = np.floor(rr).astype(int); ok = (cc >= 0) & (cc < zon.shape[1]) & (rr >= 0) & (rr < zon.shape[0]); z = np.zeros(len(P), "u1"); z[ok] = zon[rr[ok], cc[ok]]; P["rzone"] = z
    dry_by = {1: pd.Timestamp("2023-06-30"), 2: pd.Timestamp("2023-08-06"), 3: pd.Timestamp("2023-09-08")}
    pool_pts = P.in_former_pool & np.isin(P.rzone, (1, 2, 3)) & (P.date >= "2023-07-01") & (P.date <= "2024-12-31")
    dry = np.zeros(len(P), bool)
    for k, d in dry_by.items():
        dry |= pool_pts.values & (P.rzone.values == k) & (P.date.values >= np.datetime64(d))
    ws = sample(ROOT / "outputs/rasters/zone1/annual/zone1_water_share_2024_leafon_20m.tif", P.x.values, P.y.values); dry &= ~((P.year.values == 2024) & np.isfinite(ws) & (ws != 255) & (ws > 50))
    B = P[dry].copy(); B["product"] = "B reservoir bed DEM (dry bed 2023-24)"; B["dem"] = sample(ROOT / "outputs/rasters/kakhovka_bed_OK_epoch_50m.tif", B.x.values, B.y.values); B["zone"] = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"; B["wc"] = sample(CFG.BULK_ROOT / "worldcover_frames/ZONE_1_KAKHOVKA_LOWER_DNIPRO/wc_2021_20m.tif", B.x.values, B.y.values)
    parts = [B]
    L = P[~P.in_former_pool].copy()
    for zn in ("ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY"):
        fab = sample(TERR / zn / "fabdem_evrf2019_20m.tif", L.x.values, L.y.values); wc = sample(CFG.BULK_ROOT / "worldcover_frames" / zn / "wc_2021_20m.tif", L.x.values, L.y.values)
        ok = np.isfinite(fab) & (fab > -5) & (fab < 80) & (wc != 80); g = L[ok].copy(); g["dem"] = fab[ok]; g["wc"] = wc[ok]; g["zone"] = zn; g["product"] = "A FABDEM->EVRF2019 (land below dam)"; parts.append(g)
    V = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["zone", "date", "x", "y"]); V["res"] = V.dem - V.H_ice
    # C: seamless per-zone 20 m at the same points
    V["seam"] = np.nan
    for zn in ZONES:
        m = (V.zone == zn).values
        if m.any():
            V.loc[m, "seam"] = sample(SEAM / f"{zn}_dem_evrf2019_20m.tif", V.x.values[m], V.y.values[m])
    V["res_seam"] = V.seam - V.H_ice
    rows = [metrics(V.res_seam, "C seamless DEM (p55) -- ALL night points (land below dam + exposed bed)", product="C")]
    for prod, g in V.groupby("product"):
        rows.append(metrics(g.res, prod, product=prod[0])); rows.append(metrics(g.res_seam, f"C seamless on the same points as {prod[0]}", product="C"))
    for zn, g in V.groupby("zone"):                                        # per zone (user 2026-09-19): the source product and the seamless DEM
        rows.append(metrics(g.res, f"{g['product'].iloc[0][0]} source DEM, {zn}", product=g['product'].iloc[0][0], zone=zn)); rows.append(metrics(g.res_seam, f"C seamless, {zn}", product="C", zone=zn))
        for k, gg in g.groupby("wc"):
            if len(gg) >= 200:
                rows.append(metrics(gg.res_seam, f"C seamless, {zn}, WorldCover {WCN.get(int(k), int(k))}", product="C", zone=zn))
    low = V[V.dem < 5]; rows.append(metrics(low[low["product"].str.startswith("A")].res, "A FABDEM, low floodplain (< 5 m)", product="A")); rows.append(metrics(low.res_seam, "C seamless, low terrain (< 5 m) incl. exposed bed", product="C"))
    for k, g in V[V["product"].str.startswith("A")].groupby("wc"):
        rows.append(metrics(g.res, f"A FABDEM, WorldCover {WCN.get(int(k), int(k))}", product="A"))
    for y, g in V.groupby("year"):
        rows.append(metrics(g.res_seam, f"C seamless, year {y}", product="C"))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p57_dem_accuracy_night.csv", index=False); print(T.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    for prod, g, col in ((None, V[V["product"].str.startswith("A")], "#7a3b00"), (None, V[V["product"].str.startswith("B")], "#1f4e79")):
        ax.hist(g.res.clip(-6, 6), bins=96, histtype="step", lw=1.4, color=col, label=f"{g['product'].iloc[0]} n={len(g):,}")
    ax.hist(V.res_seam.clip(-6, 6), bins=96, color="#2ca25f", alpha=0.35, label=f"C seamless DEM n={int(np.isfinite(V.res_seam).sum()):,}"); ax.axvline(0, color="k"); ax.set_xlabel("DEM − ICESat-2 (night ground), m EVRF2019"); ax.legend(fontsize=8); ax.set_title("residual distributions")
    ax = axes[1]; tt = T[T.set.str.startswith(("A FABDEM->", "B reservoir", "C seamless DEM (p55) -- ALL", "C seamless, low", "A FABDEM, low"))]
    x = np.arange(len(tt)); w = 0.2
    for i, (k, col) in enumerate((("RMSE", "#2171b5"), ("MAE", "#6baed6"), ("LE90", "#e6550d"), ("bias", "#7a3b00"))):
        ax.bar(x + (i - 1.5) * w, tt[k], w, color=col, label=k)
    ax.set_xticks(x); ax.set_xticklabels([s[:38] for s in tt.set], rotation=20, ha="right", fontsize=7); ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("m"); ax.legend(); ax.set_title("night ICESat-2 accuracy metrics")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p57_dem_accuracy_night.png", dpi=100); print("-> outputs/figures/p57_dem_accuracy_night.png")


if __name__ == "__main__":
    main()
