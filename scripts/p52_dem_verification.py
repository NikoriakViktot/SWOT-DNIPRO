#!/usr/bin/env python
"""P52 -- verification of the existing DEMs before the seamless rebuild (plan 16, WP-A; user 2026-09-19).

A. Reservoir bed DEM (kakhovka_bed_OK_epoch_50m, EVRF2019) vs ICESat-2 ATL08 terrain on the EXPOSED bed, Jul 2023 - Dec 2024
   (independent: never used in the DEM). residual = H_DEM - H_ICESat2 (positive = DEM too high). Vertical chain as k10:
   H_EVRF2019 = H_terrain_common (h_te + free2mean - zeta_EGG2015) + c_EGG2015_to_EVRF2019 (mean of the six reservoir stations).
   QC: gnd_ph_count >= 4, snowcover == 1, |h| < 500; dry at pass (p43 recession zoning dates for 2023; leaf-on water share <= 50 % for 2024).
   Stats: global, by year, leaf-on/off, by recession zone, by distance to the nearest sounding; 2 km block map.
B. Zone DEMs (zone{2,3,4}_bed_canonical_PRE_BREACH_30m): datum check at the pre-breach shoreline. If chart depths are depths below the
   mean water surface (user 2026-09-19) the bed at the 2019-2022 shoreline must sit at the mean annual level (MAL ~ 0 m BS77 at
   Kherson); with p17's `-depth` convention it sits ~1-2 m too low. Sampled on a 1-cell ring inside the water polygon
   (water_share leaf-on 2019..2022 >= 50 % in every year).
C. FABDEM (EGM2008) vs ICESat-2 EVRF2019 on stable land outside the pool -> empirical c_EGM2008_to_EVRF2019 (median, NMAD).
D. Seam at the dam: pool DEM vs zone-4 DEM within 1.5 km of the dam.
Outputs: outputs/tables/p52_pool_dem_vs_icesat2.csv, p52_pool_dem_residual_blocks_2km.csv, p52_zone_dem_shoreline_datum.csv,
         p52_fabdem_datum_offset.csv, p52_dam_seam.csv; outputs/figures/p52_dem_verification.png
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
import rasterio
from rasterio import features
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

ATL = CFG.BULK_ROOT / "data_swot/processed/atl08/kakhovka_atl08_terrain.parquet"
POOL_DEM = ROOT / "outputs/rasters/kakhovka_bed_OK_epoch_50m.tif"
SND = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
RO = ROOT / "outputs/rasters/roughness/pool"
ZONES = {2: "ZONE_2_KHERSON_DELTA", 3: "ZONE_3_DNIPRO_BUG_ESTUARY", 4: "ZONE_4_DAM_TO_KHERSON_FLOODWAY"}
MAL_BS77 = {2019: -0.07, 2020: -0.10, 2021: 0.01, 2022: -0.01}      # Kherson 80805 daily means (p42/p52 check), m BS77
KHERSON_DELTA_9902 = 0.22                                            # BS77 -> EVRF2019 at Kherson (k5: epsg9902_offset ~0.216 at Nova Kakhovka; same order at Kherson)


def stats(res, label, **extra):
    r = np.asarray(res, float); r = r[np.isfinite(r)]
    if len(r) == 0:
        return dict(set=label, N=0, **extra)
    return dict(set=label, N=len(r), RMSE=round(float(np.sqrt((r ** 2).mean())), 3), MAE=round(float(np.abs(r).mean()), 3), bias=round(float(r.mean()), 3), median=round(float(np.median(r)), 3),
                NMAD=round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3), P05=round(float(np.percentile(r, 5)), 3), P95=round(float(np.percentile(r, 95)), 3), **extra)


def sample(path, xs, ys):
    with rasterio.open(path) as ds:
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys); r = np.asarray(r); c = np.asarray(c)
        ok = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width); a = ds.read(1); out = np.full(len(xs), np.nan)
        v = a[r[ok], c[ok]].astype("f8")
        if ds.nodata is not None:
            v[v == ds.nodata] = np.nan
        out[ok] = v
    return out


def part_a():
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean())
    t = pd.read_parquet(ATL, columns=["date", "rgt", "cycle", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "H_terrain_common_m", "in_former_pool"])
    t["date"] = pd.to_datetime(t.date); t = t[(t.date >= "2023-07-01") & (t.date <= "2024-12-31") & t.in_former_pool & (t.gnd_ph_count >= 4) & (t.snowcover == 1) & (t.h_te_median.abs() < 500)].copy()
    t["H_evrf"] = t.H_terrain_common_m + c
    tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); t["x"], t["y"] = tf.transform(t.lon.values, t.lat.values)
    with rasterio.open(RO / "zone1_poolwin_former_water_surface_mask.tif") as ds:
        fw = ds.read(1) == 1; tr = ds.transform
    with rasterio.open(RO / "zone1_poolwin_bed_recession_zoning_2023.tif") as ds:
        zon = ds.read(1)
    cc, rr = ~tr * (t.x.values, t.y.values); cc = np.floor(cc).astype(int); rr = np.floor(rr).astype(int); ok = (cc >= 0) & (cc < fw.shape[1]) & (rr >= 0) & (rr < fw.shape[0])
    t = t[ok].copy(); cc, rr = cc[ok], rr[ok]; t["former_water"] = fw[rr, cc]; t["zone"] = zon[rr, cc]; t = t[t.former_water].copy()
    dry_by = {1: pd.Timestamp("2023-06-30"), 2: pd.Timestamp("2023-08-06"), 3: pd.Timestamp("2023-09-08")}
    dry = np.ones(len(t), bool); y23 = t.date.dt.year.values == 2023
    for z, d in dry_by.items():
        dry[y23 & (t.zone.values == z) & (t.date.values < np.datetime64(d))] = False
    dry[t.zone.values == 4] = False
    f = ROOT / "outputs/rasters/zone1/annual/zone1_water_share_2024_leafon_20m.tif"
    if f.exists():
        ws = sample(f, t.x.values, t.y.values); dry[(t.date.dt.year.values == 2024) & np.isfinite(ws) & (ws != 255) & (ws > 50)] = False
    t = t[dry].copy(); t["H_dem"] = sample(POOL_DEM, t.x.values, t.y.values); t["res"] = t.H_dem - t.H_evrf; t = t[np.isfinite(t.res)].copy()
    s = pd.read_parquet(SND); tree = cKDTree(np.c_[s.x.values, s.y.values]); t["dist_snd_m"], _ = tree.query(np.c_[t.x.values, t.y.values])
    t["leaf_on"] = t.date.dt.month.between(6, 9); t["year"] = t.date.dt.year
    rows = [stats(t.res, "ALL 2023-07..2024-12, dry bed", c_egg2015_to_evrf2019=round(c, 4), n_rgt=int(t.rgt.nunique()))]
    for y, g in t.groupby("year"):
        rows.append(stats(g.res, f"year {y}"))
    for lo, g in t.groupby("leaf_on"):
        rows.append(stats(g.res, "leaf-on" if lo else "leaf-off"))
    ZL = {1: "fast", 2: "mid", 3: "slow"}
    for z, g in t.groupby("zone"):
        rows.append(stats(g.res, f"recession zone {ZL.get(int(z), z)}"))
    for lo, hi in ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 1e9)):
        g = t[(t.dist_snd_m >= lo) & (t.dist_snd_m < hi)]; rows.append(stats(g.res, f"dist to sounding {lo:.0f}-{min(hi, 99999):.0f} m"))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p52_pool_dem_vs_icesat2.csv", index=False); print(T.to_string(index=False))
    bx = np.floor(t.x / 2000) * 2000; by = np.floor(t.y / 2000) * 2000
    B = t.assign(bx=bx, by=by).groupby(["bx", "by"]).res.agg(n="size", median="median", mean="mean", nmad=lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))).reset_index(); B = B[B.n >= 20]
    B.to_csv(CFG.TABLES / "p52_pool_dem_residual_blocks_2km.csv", index=False)
    return t, B


def part_b():
    rows = []
    for n, zn in ZONES.items():
        dem = ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_canonical_PRE_BREACH_30m.tif"
        if not dem.exists():
            continue
        with rasterio.open(dem) as ds:
            D = ds.read(1).astype("f4"); D[D == ds.nodata] = np.nan; tr = ds.transform; shp = D.shape
        water = None
        for y in (2019, 2020, 2021, 2022):
            f = ROOT / f"outputs/rasters/zone{n}/annual/zone{n}_water_share_{y}_leafon_20m.tif"
            if not f.exists():
                continue
            with rasterio.open(f) as ds:
                w = ds.read(1, out_shape=shp, resampling=rasterio.enums.Resampling.nearest)
            wy = (w != 255) & (w >= 50); water = wy if water is None else (water & wy)
        if water is None:
            continue
        inner = water & ~ndimage.binary_erosion(water, iterations=1); ring2 = ndimage.binary_erosion(water, iterations=1) & ~ndimage.binary_erosion(water, iterations=3)
        core = ndimage.binary_erosion(water, iterations=10)
        for lab, m in (("shoreline ring (1 cell inside 2019-22 water)", inner), ("2-3 cells inside", ring2), ("core (>10 cells inside)", core)):
            v = D[m & np.isfinite(D)]
            rows.append(dict(zone=zn, where=lab, n=len(v), dem_p10=round(float(np.percentile(v, 10)), 2) if len(v) else np.nan, dem_p50=round(float(np.median(v)), 2) if len(v) else np.nan, dem_p90=round(float(np.percentile(v, 90)), 2) if len(v) else np.nan,
                             expected_at_shoreline_evrf2019=round(np.mean(list(MAL_BS77.values())) + KHERSON_DELTA_9902, 2), note="p17 convention H=-depth; user 2026-09-19: H should be MAL - depth"))
        vw = D[water & np.isfinite(D)]; vo = D[~water & np.isfinite(D)]
        rows.append(dict(zone=zn, where="DEM cells OUTSIDE the 2019-22 water polygon", n=len(vo), dem_p10=round(float(np.percentile(vo, 10)), 2) if len(vo) else np.nan, dem_p50=round(float(np.median(vo)), 2) if len(vo) else np.nan, dem_p90=round(float(np.percentile(vo, 90)), 2) if len(vo) else np.nan,
                         expected_at_shoreline_evrf2019=np.nan, note=f"share of DEM area outside the pre-breach water: {len(vo)/max(len(vo)+len(vw),1):.2f} -> the kriging was not bounded by the shoreline"))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p52_zone_dem_shoreline_datum.csv", index=False); print(T.to_string(index=False))
    return T


def part_c():
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean())
    t = pd.read_parquet(ATL, columns=["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "H_terrain_common_m", "in_former_pool", "h_canopy", "veg_ph_count"])
    t["date"] = pd.to_datetime(t.date); t = t[(t.date >= "2019-01-01") & ~t.in_former_pool & (t.gnd_ph_count >= 8) & (t.snowcover == 1) & (t.h_te_median.abs() < 500) & (t.veg_ph_count < 2)].copy()   # bare/sparse stable land
    t["H_evrf"] = t.H_terrain_common_m + c; tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); t["x"], t["y"] = tf.transform(t.lon.values, t.lat.values)
    rows = []
    for zn in ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA"):
        f = CFG.BULK_ROOT / "terrain" / zn / "fabdem_20m.tif"
        v = sample(f, t.x.values, t.y.values); ok = np.isfinite(v) & (v > 0.5) & (v < 60); r = v[ok] - t.H_evrf.values[ok]
        rows.append(stats(r, f"FABDEM - ICESat2_EVRF2019, stable bare land, {zn}"))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p52_fabdem_datum_offset.csv", index=False); print(T.to_string(index=False))
    return T


def part_d():
    import geopandas as gpd
    from shapely.geometry import Point
    dam = gpd.GeoSeries([Point(*CFG.KAKHOVKA_DAM)], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]
    xs = dam.x + np.arange(-3000, 3001, 50.0); ys = np.full(xs.shape, dam.y)
    with rasterio.open(POOL_DEM) as ds:
        pass
    p = sample(POOL_DEM, xs, ys); z4 = sample(ROOT / "outputs/rasters/zone4/zone4_bed_canonical_PRE_BREACH_30m.tif", xs, ys); fab = sample(CFG.BULK_ROOT / "terrain/ZONE_4_DAM_TO_KHERSON_FLOODWAY/fabdem_20m.tif", xs, ys)
    T = pd.DataFrame(dict(dx_from_dam_m=xs - dam.x, pool_dem=p, zone4_dem=z4, fabdem=fab)); T.to_csv(CFG.TABLES / "p52_dam_seam.csv", index=False)
    print("dam seam: pool DEM valid cells W-E:", int(np.isfinite(p).sum()), "| zone4 valid:", int(np.isfinite(z4).sum()), "| both:", int((np.isfinite(p) & np.isfinite(z4)).sum()))
    return T


def main():
    t, B = part_a(); TB = part_b(); TC = part_c(); TD = part_d()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]; ax.hist(t.res.clip(-8, 8), bins=80, color="#2171b5"); ax.axvline(0, color="k"); ax.set_xlabel("pool DEM − ICESat-2 (EVRF2019), m"); ax.set_title(f"(a) pool DEM vs ICESat-2 dry bed 2023-07..2024-12: n={len(t):,}, RMSE {np.sqrt((t.res**2).mean()):.2f} m, bias {t.res.mean():+.2f}, NMAD {1.4826*np.median(np.abs(t.res-t.res.median())):.2f}")
    ax = axes[0, 1]; sc = ax.scatter(B.bx + 1000, B.by + 1000, c=B["median"], cmap="RdBu_r", vmin=-3, vmax=3, s=12, marker="s"); plt.colorbar(sc, ax=ax, label="median residual, m (2 km blocks, n ≥ 20)"); ax.set_aspect("equal"); ax.set_title("(b) where the pool DEM is too high (red) / too low (blue)")
    ax = axes[1, 0]; tb = TB[TB["where"].str.startswith("shoreline")]; ax.bar(range(len(tb)), tb.dem_p50, color="#7a3b00"); ax.axhline(tb.expected_at_shoreline_evrf2019.iloc[0], color="red", ls="--", label="expected = MAL 2019-22 (Kherson) in EVRF2019")
    ax.set_xticks(range(len(tb))); ax.set_xticklabels([z.split("_")[1] for z in tb.zone]); ax.set_ylabel("zone DEM at the pre-breach shoreline, m EVRF2019 (p50)"); ax.legend(); ax.set_title("(c) datum check: bed at the 2019-22 shoreline should equal the mean water level")
    ax = axes[1, 1]; ax.plot(TD.dx_from_dam_m, TD.pool_dem, label="pool DEM"); ax.plot(TD.dx_from_dam_m, TD.zone4_dem, label="zone-4 DEM"); ax.plot(TD.dx_from_dam_m, TD.fabdem, label="FABDEM (EGM2008)", alpha=0.6); ax.axvline(0, color="k", ls=":"); ax.set_xlabel("E-W distance from the dam, m"); ax.set_ylabel("m"); ax.legend(); ax.set_title("(d) seam at the dam (W-E profile through the dam point)")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p52_dem_verification.png", dpi=100); print("-> outputs/figures/p52_dem_verification.png")


if __name__ == "__main__":
    main()
