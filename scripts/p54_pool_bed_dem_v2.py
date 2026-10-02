#!/usr/bin/env python
"""P54 -- reservoir bed DEM v2, channel first (plan 16, WP-C; user 2026-09-19): the old Dnipro channel that re-emerged after the breach is
kriged on its own, along its axis, before the rest of the bed.

Why: the pool DEM (hist14/hist20, isotropic OK on 7 514 soundings) smears the incised channel sideways -- p52 shows the largest residuals
against ICESat-2 in the 'fast' recession zone and near the channel. A channel is anisotropic: bed elevation varies slowly along the thalweg
and quickly across it.

Steps
  1. channel mask (20 m, pool window grid): p43 bed recession zoning == 4 (never exposed in 2023) UNION p50 water-occurrence classes 5-6
     (water >= 75 % of the 30 dates 2023-06..2024-09); binary closing 3x3, components < 0.5 km2 dropped; dilated by CHANNEL_HALO_M.
  2. centreline: SWORD nodes inside the pool, main stem only (nodes whose chain-order neighbours are < 1 km away; 9 tributary jumps dropped),
     ordered by chain_km -> LineString.
  3. channel soundings = soundings inside the (dilated) channel mask; densified along the centreline: every node gets a thalweg point
     H = min(H) of soundings within R_THALWEG (a channel thalweg is its deepest line), sigma 0.8 m; nodes without soundings interpolated
     along the line (sigma 1.2 m). Points enter kriging in channel-fitted coordinates (s along, ANISO*d across) -- kriging.ok_soft.
  4. surface: channel cells kriged in channel-fitted space; everywhere else the hist20 50 m bed DEM (resampled to 20 m); 100 m feather at
     the channel edge. Cells of the former water surface never covered by either stay NaN (p55 fills them from the shoreline).
  5. validation: (a) blocked CV on channel soundings, channel-fitted OK vs the hist20 surface sampled at the same soundings (the honest
     comparison: does the anisotropic channel model beat the existing DEM on held-out depths?); (b) ICESat-2 dry-bed 2023-07..2024-12 (p52
     method) on the WHOLE pool for v1 vs v2 -- the channel itself is under water, so this only checks that v2 did not degrade the exposed bed.
Outputs
  outputs/rasters/kakhovka_bed_v2_channel_first_20m.tif (+ _source: 1 channel kriging, 2 hist20 bed, 3 feather), outputs/tables/p54_channel_cv.csv,
  p54_pool_v1_vs_v2_icesat2.csv, p54_channel_points.parquet, outputs/figures/p54_pool_channel_dem.png
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import kriging as KR
from swot_dnipro import spatial_domains as SD
import p19_zone24_bed_surface as P19
from p28_zone_bed_surface import fit_variogram
from p53_below_dam_bed_dem import channel_fitted

RO = ROOT / "outputs/rasters/roughness/pool"
SND = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
POOL_V1 = ROOT / "outputs/rasters/kakhovka_bed_OK_epoch_50m.tif"
SWORD = CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet"
ATL = CFG.BULK_ROOT / "data_swot/processed/atl08/kakhovka_atl08_terrain.parquet"
CHANNEL_HALO_M, MIN_COMPONENT_KM2, R_THALWEG, FEATHER_M = 100.0, 0.5, 500.0, 100.0
SIGMA_SND, SIGMA_THAL, SIGMA_THAL_INTERP = 0.3, 0.8, 1.2


def sample(path, xs, ys):
    with rasterio.open(path) as ds:
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys); r = np.asarray(r); c = np.asarray(c); ok = (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width); a = ds.read(1); out = np.full(len(xs), np.nan)
        v = a[r[ok], c[ok]].astype("f8")
        if ds.nodata is not None:
            v[v == ds.nodata] = np.nan
        out[ok] = v
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--aniso", type=float, default=5.0); a = ap.parse_args(); t0 = time.time()
    with rasterio.open(RO / "zone1_poolwin_bed_recession_zoning_2023.tif") as ds:
        zon = ds.read(1); tr = ds.transform; shape = zon.shape; cell = ds.res[0]
    occ = np.zeros(shape, "u1"); f50 = RO / "zone1_water_occurrence_class_LISCH_water3_20m.tif"
    with rasterio.open(f50) as ds:
        reproject(source=rasterio.band(ds, 1), destination=occ, dst_transform=tr, dst_crs=CFG.CRS_METRIC, resampling=Resampling.nearest, dst_nodata=0)
    pool = SD.load_utm("reservoir_full_pool_prebreach"); from rasterio import features
    pool_m = features.rasterize([(pool, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    ch = ((zon == 4) | (occ >= 5)) & pool_m; ch = ndimage.binary_closing(ch, structure=np.ones((3, 3)))
    lab, n = ndimage.label(ch); sz = np.bincount(lab.ravel()) * cell * cell / 1e6; ch = np.isin(lab, np.flatnonzero(sz >= MIN_COMPONENT_KM2)[1:]) if n else ch
    ch_d = ndimage.binary_dilation(ch, iterations=int(CHANNEL_HALO_M / cell)) & pool_m
    print(f"channel mask: {ch.sum()*cell*cell/1e6:.0f} km2 ({n} components before the {MIN_COMPONENT_KM2} km2 filter), dilated {ch_d.sum()*cell*cell/1e6:.0f} km2", flush=True)
    # centreline: SWORD main stem inside the pool
    sw = pd.read_parquet(SWORD).sort_values("chain_km").reset_index(drop=True); tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); sw["x"], sw["y"] = tf.transform(sw.lon.values, sw.lat.values)
    sw = sw[shapely.contains_xy(pool.buffer(500), sw.x.values, sw.y.values)].reset_index(drop=True)
    d = np.hypot(np.diff(sw.x), np.diff(sw.y)); keep = np.ones(len(sw), bool); keep[1:] &= d < 1000; keep[:-1] &= np.r_[d < 1000]
    sw = sw[keep].reset_index(drop=True); line = shapely.LineString(np.c_[sw.x.values, sw.y.values]); print(f"centreline: {len(sw)} SWORD nodes, {line.length/1e3:.0f} km")
    # soundings
    s = pd.read_parquet(SND); s = s.rename(columns={"H_bed_evrf2019_m": "H"})[["x", "y", "H"]].dropna()   # BED elevation (H_ref is the 14 m reference level)
    cc, rr = ~tr * (s.x.values, s.y.values); cc = np.floor(cc).astype(int); rr = np.floor(rr).astype(int); ok = (cc >= 0) & (cc < shape[1]) & (rr >= 0) & (rr < shape[0]); inch = np.zeros(len(s), bool); inch[ok] = ch_d[rr[ok], cc[ok]]
    cs = s[inch].copy(); print(f"soundings in the channel: {len(cs)} of {len(s)}")
    # thalweg densification along the centreline
    ts = cKDTree(np.c_[cs.x.values, cs.y.values]); th = np.full(len(sw), np.nan)
    for i, (x, y) in enumerate(zip(sw.x.values, sw.y.values)):
        idx = ts.query_ball_point([x, y], R_THALWEG)
        if len(idx) >= 2:
            th[i] = np.min(cs.H.values[idx])
    have = np.isfinite(th); sig = np.full(len(sw), np.nan)
    if have.any():
        lo, hi = np.flatnonzero(have).min(), np.flatnonzero(have).max(); ar = np.arange(len(sw)); interp = ~have & (ar > lo) & (ar < hi)
        sdist = shapely.line_locate_point(line, shapely.points(sw.x.values, sw.y.values)); th[interp] = np.interp(sdist[interp], sdist[have], th[have]); sig[have] = SIGMA_THAL; sig[interp] = SIGMA_THAL_INTERP
    thal = pd.DataFrame(dict(x=sw.x.values, y=sw.y.values, H=th, sigma=sig)).dropna(); print(f"thalweg points: {len(thal)} ({int(have.sum())} from soundings within {R_THALWEG:.0f} m)")
    P = pd.concat([cs.assign(sigma=SIGMA_SND, source="sounding"), thal.assign(source="thalweg")], ignore_index=True); P.to_parquet(CFG.TABLES / "p54_channel_points.parquet", index=False)
    xy_real = np.c_[P.x.values, P.y.values].astype(float); z_all = P.H.values.astype(float); ev_all = P.sigma.values.astype(float) ** 2; is_snd = (P.source == "sounding").values
    T = lambda xy: channel_fitted(line, xy, a.aniso); xy_all = T(xy_real)
    vg = fit_variogram(xy_all[is_snd], z_all[is_snd], CFG.SEED); print(f"variogram (channel soundings, channel-fitted): range {vg[0]/1e3:.2f} km, sill {vg[1]:.2f}, nugget {vg[2]:.2f}")
    # CV: channel-fitted OK (with thalweg) vs hist20 v1 at the same held-out soundings
    xy_snd, z_snd, xy_snd_real = xy_all[is_snd], z_all[is_snd], xy_real[is_snd]; nn = cKDTree(xy_snd_real).query(xy_snd_real, k=2)[0][:, 1]; block_km = max(0.5, round(np.median(nn) / 200.0) * 0.2); folds = P19.block_folds(xy_snd_real, block_km, CFG.SEED)
    resid = np.full(len(z_snd), np.nan)
    for f in range(P19.N_FOLDS):
        te = folds == f
        if te.sum() == 0:
            continue
        xy_tr = np.vstack([xy_snd[~te], xy_all[~is_snd]]); z_tr = np.concatenate([z_snd[~te], z_all[~is_snd]]); ev = np.concatenate([np.full((~te).sum(), SIGMA_SND ** 2), ev_all[~is_snd]])
        resid[te] = KR.ok_soft(xy_tr, z_tr, ev, xy_snd[te], vg) - z_snd[te]
    v1 = sample(POOL_V1, xy_snd_real[:, 0], xy_snd_real[:, 1]) - z_snd
    def st(r, lab):
        r = r[np.isfinite(r)]; return dict(model=lab, n=len(r), RMSE=round(float(np.sqrt((r ** 2).mean())), 3), MAE=round(float(np.abs(r).mean()), 3), bias=round(float(r.mean()), 3), NMAD=round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3))
    cv = pd.DataFrame([st(resid, f"channel-fitted OK_SOFT (aniso {a.aniso}), blocked CV on channel soundings"), st(v1, "hist20 pool DEM v1 sampled at the same soundings (in-sample for v1)")]); cv["block_km"] = block_km; cv.to_csv(CFG.TABLES / "p54_channel_cv.csv", index=False); print(cv.to_string(index=False))
    # surface: channel cells
    rr, cc = np.nonzero(ch_d); tx = tr.c + (cc + 0.5) * cell; ty = tr.f - (rr + 0.5) * cell; pred, diag = KR.ok_soft(xy_all, z_all, ev_all, T(np.c_[tx, ty]), vg, return_diag=True)
    chan = np.full(shape, np.nan, "f4"); chan[rr, cc] = pred; print(f"channel surface: {len(rr):,} cells, {diag['n_fallback']} IDW fallbacks ({time.time()-t0:.0f}s)")
    v1r = np.full(shape, np.nan, "f4")
    with rasterio.open(POOL_V1) as ds:
        reproject(source=rasterio.band(ds, 1), destination=v1r, dst_transform=tr, dst_crs=CFG.CRS_METRIC, resampling=Resampling.bilinear, src_nodata=ds.nodata, dst_nodata=np.nan)
    has_c = np.isfinite(chan); dist = ndimage.distance_transform_edt(~has_c) * cell; w = np.clip(1 - dist / FEATHER_M, 0, 1)
    idx = ndimage.distance_transform_edt(~has_c, return_indices=True, return_distances=False); near_c = chan[idx[0], idx[1]]; del idx
    dem = np.where(has_c, chan, np.where(np.isfinite(v1r), (1 - w) * v1r + w * np.where(np.isfinite(near_c), near_c, v1r), np.nan)).astype("f4")
    src = np.where(has_c, 1, np.where(np.isfinite(v1r), np.where(w > 0, 3, 2), 0)).astype("u1"); dem[~pool_m] = np.nan; src[~pool_m] = 0
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, crs=CFG.CRS_METRIC, transform=tr, compress="deflate", tiled=True)
    with rasterio.open(ROOT / "outputs/rasters/kakhovka_bed_v2_channel_first_20m.tif", "w", dtype="float32", nodata=-9999.0, **prof) as ds:
        ds.write(np.nan_to_num(dem, nan=-9999.0), 1); ds.update_tags(quantity="bed elevation m EVRF2019", method=f"channel (p43 never-exposed U p50 occurrence>=75 %) kriged in channel-fitted coordinates (SWORD main stem, aniso {a.aniso}) with thalweg densification; elsewhere hist20 OK bed; {FEATHER_M:.0f} m feather", producer="p54_pool_bed_dem_v2.py")
    with rasterio.open(ROOT / "outputs/rasters/kakhovka_bed_v2_source_20m.tif", "w", dtype="uint8", nodata=0, **prof) as ds:
        ds.write(src, 1); ds.update_tags(values="1 channel kriging; 2 hist20 bed DEM v1; 3 feather", producer="p54_pool_bed_dem_v2.py")
    # ICESat-2 dry bed check v1 vs v2 (p52 method)
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean())
    t = pd.read_parquet(ATL, columns=["date", "lon", "lat", "h_te_median", "gnd_ph_count", "snowcover", "H_terrain_common_m", "in_former_pool"]); t["date"] = pd.to_datetime(t.date)
    t = t[(t.date >= "2023-07-01") & (t.date <= "2024-12-31") & t.in_former_pool & (t.gnd_ph_count >= 4) & (t.snowcover == 1)].copy(); t["H"] = t.H_terrain_common_m + c; t["x"], t["y"] = tf.transform(t.lon.values, t.lat.values)
    cc2, rr2 = ~tr * (t.x.values, t.y.values); cc2 = np.floor(cc2).astype(int); rr2 = np.floor(rr2).astype(int); ok = (cc2 >= 0) & (cc2 < shape[1]) & (rr2 >= 0) & (rr2 < shape[0]); t = t[ok].copy(); rr2, cc2 = rr2[ok], cc2[ok]
    t["z"] = zon[rr2, cc2]; t = t[np.isin(t.z, (1, 2, 3))]; dry_by = {1: "2023-06-30", 2: "2023-08-06", 3: "2023-09-08"}; t = t[[(r.date >= pd.Timestamp(dry_by[int(r.z)])) for r in t.itertuples()]]
    cc2, rr2 = ~tr * (t.x.values, t.y.values); cc2 = np.floor(cc2).astype(int); rr2 = np.floor(rr2).astype(int)
    r1 = v1r[rr2, cc2] - t.H.values; r2 = dem[rr2, cc2] - t.H.values; srcv = src[rr2, cc2]
    rows = [st(r1, "v1 hist20, all dry-bed segments"), st(r2, "v2 channel-first, all dry-bed segments")]
    for k, lab in ((1, "channel cells"), (3, "feather cells"), (2, "unchanged v1 cells")):
        m = srcv == k; rows += [st(r1[m], f"v1 on {lab}"), st(r2[m], f"v2 on {lab}")]
    V = pd.DataFrame(rows); V.to_csv(CFG.TABLES / "p54_pool_v1_vs_v2_icesat2.csv", index=False); print(V.to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)); k = 4; ext = (tr.c, tr.c + shape[1] * cell, tr.f - shape[0] * cell, tr.f)
    axes[0].imshow(np.where(pool_m[::k, ::k], 0.92, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1); axes[0].imshow(np.where(ch[::k, ::k], 1, np.nan), extent=ext, cmap="Blues", vmin=0, vmax=1.3); axes[0].plot(sw.x, sw.y, "r-", lw=0.6); axes[0].set_title("old channel mask (never exposed ∪ water ≥ 75 %) and SWORD main stem"); axes[0].set_aspect("equal")
    im = axes[1].imshow(dem[::k, ::k], extent=ext, cmap="viridis", vmin=-15, vmax=16); plt.colorbar(im, ax=axes[1], fraction=0.03, label="m EVRF2019"); axes[1].set_title("bed DEM v2, channel first"); axes[1].set_aspect("equal")
    dd = (dem - v1r)[::k, ::k]; im = axes[2].imshow(dd, extent=ext, cmap="RdBu_r", vmin=-4, vmax=4); plt.colorbar(im, ax=axes[2], fraction=0.03, label="v2 − v1, m"); axes[2].set_title("where the channel model changed the bed"); axes[2].set_aspect("equal")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p54_pool_channel_dem.png", dpi=90); print(f"-> outputs/figures/p54_pool_channel_dem.png ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
