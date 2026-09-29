#!/usr/bin/env python
"""P56 -- FABDEM into the project's single vertical frame, and its validation below the dam with night-time ICESat-2 ground points
(user 2026-09-19: "FABDEM через геоїд до сфероїда, потім до EGG2015 і поправку; валідація нижче дамби нічними ground-точками ATL08").

Vertical chain, identical in form to the ICESat-2 terrain chain of k10 (config.free2mean, EGG2015 quasigeoid, c_EGG2015->EVRF2019):
    FABDEM  H_EGM2008 (orthometric, tide-free)
      + N_EGM2008(lon, lat)            EGM2008 geoid undulation (PROJ, EPSG:3855 -> EPSG:4979, network grid us_nga_egm08_25)
      = h_ell                          WGS84 ellipsoidal height
      + free2mean(lat)                 permanent-tide harmonisation to the mean-tide system (as ATL03 -> k10)
      - zeta_EGG2015(lon, lat)         EGG2015 quasigeoid (sibling repo egg_2015.tif)
      + c_EGG2015_to_EVRF2019          mean of the six reservoir stations (outputs/tables/egg2015_to_evrf2019_by_station.csv)
      = H_EVRF2019                     -> terrain/<ZONE>/fabdem_evrf2019_20m.tif
    ICESat-2 ATL08 terrain: H_terrain_common_m (= h_te + free2mean - zeta_EGG2015, k10) + c  -> the same frame.

Validation (ZONE_4 dam-Kherson and ZONE_2 delta frames, land outside the former pool; night shots primary because the ATL08 ground
finder is cleaner without solar background): residual = FABDEM_EVRF2019 - ICESat2_EVRF2019 (positive = FABDEM too high).
RMSE / MAE / bias / median / NMAD, night vs day, by zone, by WorldCover 2021 class, by canopy height class, by period (pre / post breach:
the floodplain terrain should not have changed except where the flood eroded or deposited), 2 km block medians.
Outputs: terrain/<ZONE>/fabdem_evrf2019_20m.tif (4 zones), outputs/tables/p56_fabdem_chain_samples.csv, p56_fabdem_vs_icesat2_below_dam.csv,
         p56_fabdem_residual_blocks_2km.csv, outputs/figures/p56_fabdem_validation.png
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
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer, network
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import vertical as VT

network.set_network_enabled(True)
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY")
VAL_ZONES = ("ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY")
EXTRA_PULLS = ("lower_dnipro_atl08_20m_raw.parquet", "liman_atl08_20m_raw.parquet")   # 20 m PhoREAL pulls west of 33.4E (2022-2025): same chain applied on the fly
ATL = CFG.BULK_ROOT / "data_swot/processed/atl08/kakhovka_atl08_terrain.parquet"
LATTICE_M = 2000.0
WCN = {10: "trees", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare", 80: "water", 90: "wetland"}


def c_evrf() -> float:
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); return float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean())


def geoid_fields(ds, tf_ll):
    """N_EGM2008 and zeta_EGG2015 on a coarse lattice over the raster, returned as interpolators in (row, col)."""
    t = Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
    xs = np.arange(ds.bounds.left, ds.bounds.right + LATTICE_M, LATTICE_M); ys = np.arange(ds.bounds.top, ds.bounds.bottom - LATTICE_M, -LATTICE_M)
    X, Y = np.meshgrid(xs, ys); lon, lat = tf_ll.transform(X.ravel(), Y.ravel())
    _, _, N = t.transform(lon, lat, np.zeros(lon.shape)); N = np.asarray(N).reshape(X.shape)
    zeta = np.asarray(VT.sample_grid(CFG.EGG2015_TIF, lon, lat)).reshape(X.shape)
    f2m = np.asarray(CFG.free2mean(lat)).reshape(X.shape)
    rows = np.asarray(rasterio.transform.rowcol(ds.transform, np.full(len(ys), xs[0]), ys)[0], float); cols = np.asarray(rasterio.transform.rowcol(ds.transform, xs, np.full(len(xs), ys[0]))[1], float)
    mk = lambda A: RegularGridInterpolator((rows, cols), A, bounds_error=False, fill_value=None)
    return mk(N), mk(zeta), mk(f2m), dict(N_p50=float(np.median(N)), zeta_p50=float(np.median(zeta)), free2mean_p50=float(np.median(f2m)), N_range=(float(N.min()), float(N.max())), zeta_range=(float(zeta.min()), float(zeta.max())))


def convert_zone(zone: str, c: float) -> dict:
    src = CFG.BULK_ROOT / "terrain" / zone / "fabdem_20m.tif"; dst = CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif"
    tf_ll = Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True)
    with rasterio.open(src) as ds:
        fN, fZ, fF, info = geoid_fields(ds, tf_ll); prof = ds.profile.copy(); prof.update(dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(dst, "w", **prof) as out:
            for _, win in ds.block_windows(1):
                a = ds.read(1, window=win).astype("f4"); nd = (a == ds.nodata) | ~np.isfinite(a)
                rr = np.arange(win.row_off, win.row_off + win.height) + 0.5; cc = np.arange(win.col_off, win.col_off + win.width) + 0.5; RR, CC = np.meshgrid(rr, cc, indexing="ij"); pts = np.c_[RR.ravel(), CC.ravel()]
                corr = (fN(pts) + fF(pts) - fZ(pts) + c).reshape(a.shape).astype("f4")
                out.write(np.where(nd, -9999.0, a + corr).astype("f4"), 1, window=win)
            out.update_tags(vertical_datum="EVRF2019 (empirically aligned)", chain="H_EGM2008 + N_EGM2008(PROJ us_nga_egm08_25) + free2mean(lat) - zeta_EGG2015 + c_EGG2015_to_EVRF2019", c_egg2015_to_evrf2019=f"{c:+.4f}",
                            N_egm2008_median=f"{info['N_p50']:.3f}", zeta_egg2015_median=f"{info['zeta_p50']:.3f}", free2mean_median=f"{info['free2mean_p50']:.4f}", producer="p56_fabdem_evrf2019.py", source="FABDEM v1.2 (terrain/<zone>/fabdem_20m.tif)")
    info.update(zone=zone, c=c, total_shift_median=round(info["N_p50"] + info["free2mean_p50"] - info["zeta_p50"] + c, 3)); print(f"{zone}: N_EGM2008 {info['N_range'][0]:.2f}..{info['N_range'][1]:.2f}, zeta_EGG2015 {info['zeta_range'][0]:.2f}..{info['zeta_range'][1]:.2f}, free2mean {info['free2mean_p50']:+.3f}, c {c:+.3f} -> shift {info['total_shift_median']:+.3f} m", flush=True)
    return info


def stats(res, label, **extra):
    r = np.asarray(res, float); r = r[np.isfinite(r)]
    if len(r) == 0:
        return dict(set=label, N=0, **extra)
    return dict(set=label, N=len(r), RMSE=round(float(np.sqrt((r ** 2).mean())), 3), MAE=round(float(np.abs(r).mean()), 3), bias=round(float(r.mean()), 3), median=round(float(np.median(r)), 3), NMAD=round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3), P05=round(float(np.percentile(r, 5)), 3), P95=round(float(np.percentile(r, 95)), 3), **extra)


def sample(path, xs, ys):
    with rasterio.open(path) as ds:
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys); r = np.asarray(r); c = np.asarray(c); ok = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width); a = ds.read(1); out = np.full(len(xs), np.nan)
        v = a[r[ok], c[ok]].astype("f8")
        if ds.nodata is not None:
            v[v == ds.nodata] = np.nan
        out[ok] = v
    return out


def validate(c: float):
    t = pd.read_parquet(ATL, columns=["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "veg_ph_count", "H_terrain_common_m", "in_former_pool", "rgt"])
    t["date"] = pd.to_datetime(t.date); t = t[~t.in_former_pool & (t.gnd_ph_count >= 8) & (t.snowcover == 1) & (t.h_te_median.abs() < 500)].copy(); t["H_ice"] = t.H_terrain_common_m + c; t["pull"] = "kakhovka"
    extra = []
    for name in EXTRA_PULLS:                                                   # the western pulls carry no vertical chain yet: apply k10's chain here
        f = ATL.parent / name
        if not f.exists():
            continue
        e = pd.read_parquet(f, columns=["time", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "veg_ph_count", "geometry"])
        import geopandas as gpd
        g = gpd.GeoSeries.from_wkb(e.pop("geometry")); e["lon"], e["lat"] = g.x.values, g.y.values; e["date"] = pd.to_datetime(e.pop("time")).dt.tz_localize(None)
        e = e[(e.gnd_ph_count >= 8) & (e.snowcover == 1) & (e.h_te_median.abs() < 500)].copy()
        zeta = VT.sample_grid(CFG.EGG2015_TIF, e.lon.values, e.lat.values); e["H_ice"] = e.h_te_median.values + CFG.free2mean(e.lat.values) - zeta + c
        e["rgt"] = -1; e["pull"] = name.split("_atl08")[0]; extra.append(e[np.isfinite(e.H_ice)]); print(f"  extra pull {name}: {len(e):,} QC segments", flush=True)
    if extra:
        t = pd.concat([t[["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "veg_ph_count", "H_ice", "rgt", "pull"]]] + [x[["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "solar_elevation", "h_canopy", "veg_ph_count", "H_ice", "rgt", "pull"]] for x in extra], ignore_index=True)
    tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); t["x"], t["y"] = tf.transform(t.lon.values, t.lat.values); t["night"] = t.solar_elevation < 0
    t["period"] = np.where(t.date < CFG.BREACH_DATE, "pre-breach", "post-breach"); t["canopy_class"] = pd.cut(t.h_canopy.fillna(0), [-1, 0.5, 2, 5, 100], labels=["no canopy", "0.5-2 m", "2-5 m", "> 5 m"])
    parts = []
    for zn in VAL_ZONES:
        fab = sample(CFG.BULK_ROOT / "terrain" / zn / "fabdem_evrf2019_20m.tif", t.x.values, t.y.values); wc = sample(CFG.BULK_ROOT / "worldcover_frames" / zn / "wc_2021_20m.tif", t.x.values, t.y.values)
        ok = np.isfinite(fab) & (fab > -5) & (fab < 80); g = t[ok].copy(); g["fab"] = fab[ok]; g["wc"] = wc[ok]; g["zone"] = zn; g["res"] = g.fab - g.H_ice; parts.append(g)
    V = pd.concat(parts, ignore_index=True); V = V[V.wc != 80]                        # FABDEM on water is meaningless
    V = V.drop_duplicates(subset=["zone", "date", "x", "y"])
    rows = [stats(V[V.night].res, "ALL below dam, NIGHT (primary)", n_rgt=int(V[V.night].rgt.nunique())), stats(V[~V.night].res, "ALL below dam, day")]
    for zn, g in V[V.night].groupby("zone"):
        rows.append(stats(g.res, f"night, {zn}"))
    for pl, g in V[V.night].groupby("pull"):
        rows.append(stats(g.res, f"night, pull {pl}"))
    for p, g in V[V.night].groupby("period"):
        rows.append(stats(g.res, f"night, {p}"))
    for k, g in V[V.night].groupby("wc"):
        rows.append(stats(g.res, f"night, WorldCover {WCN.get(int(k), int(k))}"))
    for k, g in V[V.night].groupby("canopy_class", observed=True):
        rows.append(stats(g.res, f"night, ICESat-2 canopy {k}"))
    for lo, hi in ((-5, 2), (2, 5), (5, 10), (10, 80)):
        g = V[V.night & (V.fab >= lo) & (V.fab < hi)]; rows.append(stats(g.res, f"night, FABDEM {lo}..{hi} m"))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p56_fabdem_vs_icesat2_below_dam.csv", index=False); print(T.to_string(index=False))
    Vn = V[V.night]; B = Vn.assign(bx=np.floor(Vn.x / 2000) * 2000, by=np.floor(Vn.y / 2000) * 2000).groupby(["bx", "by"]).res.agg(n="size", median="median", nmad=lambda r: 1.4826 * np.median(np.abs(r - np.median(r)))).reset_index(); B = B[B.n >= 15]
    B.to_csv(CFG.TABLES / "p56_fabdem_residual_blocks_2km.csv", index=False)
    return V, T, B


def main():
    c = c_evrf()
    if "--validate-only" not in sys.argv:
        infos = [convert_zone(z, c) for z in ZONES]; pd.DataFrame(infos).to_csv(CFG.TABLES / "p56_fabdem_chain_samples.csv", index=False)
    V, T, B = validate(c)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11)); Vn = V[V.night]
    ax = axes[0, 0]; ax.hist(Vn.res.clip(-6, 6), bins=80, color="#2171b5", alpha=0.8, label=f"night n={len(Vn):,}"); ax.hist(V[~V.night].res.clip(-6, 6), bins=80, histtype="step", color="#e6550d", label=f"day n={int((~V.night).sum()):,}"); ax.axvline(0, color="k")
    r0 = T.iloc[0]; ax.set_title(f"(a) FABDEM(EVRF2019) − ICESat-2(EVRF2019), land below the dam: night RMSE {r0.RMSE:.2f} m, bias {r0.bias:+.2f}, NMAD {r0.NMAD:.2f}"); ax.set_xlabel("residual, m (positive = FABDEM too high)"); ax.legend()
    ax = axes[0, 1]; sc = ax.scatter(B.bx + 1000, B.by + 1000, c=B["median"], cmap="RdBu_r", vmin=-2, vmax=2, s=14, marker="s"); plt.colorbar(sc, ax=ax, label="median residual, m (night, 2 km blocks, n ≥ 15)"); ax.set_aspect("equal"); ax.set_title("(b) spatial pattern of the FABDEM error")
    ax = axes[1, 0]; tw = T[T.set.str.startswith("night, WorldCover")]; ax.bar(range(len(tw)), tw.bias, color="#7a3b00", label="bias"); ax.errorbar(range(len(tw)), tw.bias, yerr=tw.NMAD, fmt="none", ecolor="k", capsize=3, label="± NMAD"); ax.set_xticks(range(len(tw))); ax.set_xticklabels([s.split("WorldCover ")[1] for s in tw.set], rotation=30); ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("m"); ax.set_title("(c) by WorldCover 2021 class (night)"); ax.legend()
    ax = axes[1, 1]; tc = T[T.set.str.startswith("night, ICESat-2 canopy")]; ax.bar(range(len(tc)), tc.bias, color="#238b45"); ax.errorbar(range(len(tc)), tc.bias, yerr=tc.NMAD, fmt="none", ecolor="k", capsize=3); ax.set_xticks(range(len(tc))); ax.set_xticklabels([s.split("canopy ")[1] for s in tc.set]); ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("m"); ax.set_title("(d) by ICESat-2 canopy height at the segment (FABDEM is forest-removed)")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p56_fabdem_validation.png", dpi=100); print("-> outputs/figures/p56_fabdem_validation.png")


if __name__ == "__main__":
    main()
