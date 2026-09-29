#!/usr/bin/env python
"""P53 -- below-dam bed DEM v2 (plan 16, WP-B; user 2026-09-19): chart depths relative to the mean water level, kriging bounded by the
pre-breach (2019-2022) shoreline, points densified along the channel before kriging.

What changes against p28 (whose canonical surfaces put the bed 2-6 m below the shoreline and, in ZONE_4, 86 % outside the water; p52):
  1. datum   : H_bed = MAL(x) - depth. MAL(x) = mean annual level 2019-2022 at Kherson (80805, BS77) + delta_EPSG9902(point)
               + G * s(x), G = +0.0014 m/km (p27 / Finding 6), s = chainage upstream of Kherson (SWORD). p17 used H = -depth + delta,
               i.e. MAL = 0: the difference is MAL_Kherson (-0.04 m) + G*s -- centimetres here, so the datum was not the main defect.
  2. boundary: the kriging target set is ONLY the pre-breach water polygon (leaf-on water_share >= 50 % in every year 2019-2022 with a
               composite; p40); nothing is predicted on land. The p27 dated shoreline vertices (H ~ MAL, sigma 0.2-0.4 m) enter as SOFT
               data, always (p28's 'SND_ONLY wins the CV' rule is dropped on purpose: the user wants the shoreline honoured).
  3. densify : (a) SWORD nodes (~300 m apart) get a thalweg point H = MAL - max depth of the soundings within R_THALWEG, sigma 1.0 m;
               nodes without soundings are linearly interpolated along chainage (sigma 1.5 m); (b) ZONE_3 only: EMODnet DTM 2024
               sampled on a 250 m lattice where no sounding lies within 1 km (sigma 1.0 m, --emodnet).
  4. estimator: kriging.ok_soft, variogram on the soundings only (p28.fit_variogram), soundings err_var = SIGMA_SND^2 (chart reading).
  5. validation: blocked CV on soundings (p19.block_folds), OK_SOFT with all constraints vs soundings-only OK; plus the p52 shoreline
               check (bed on the 1-cell ring inside the water polygon vs MAL).
Outputs
  outputs/rasters/zone<N>/zone<N>_bed_v2_PRE_BREACH_30m.tif      bed elevation, m EVRF2019, NaN outside the water polygon / support
  outputs/rasters/zone<N>/zone<N>_bed_v2_source_30m.tif          1 sounding-supported (<= SUPPORT_M), 2 densified/shore-supported only
  outputs/tables/p53_bed_v2_cv_<ZONE>.csv, p53_bed_v2_summary.csv (append), p53_bed_v2_points_<ZONE>.parquet (all training points + source)
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
import geopandas as gpd
import rasterio
import shapely
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject
import rasterio.profiles
import rasterio.transform
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer

from swot_dnipro import config as CFG
from swot_dnipro import kriging as KR
from swot_dnipro import spatial_domains as SD
import p19_zone24_bed_surface as P19
from p28_zone_bed_surface import fit_variogram

ZONES = {"ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4, "ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3}
CELL = 30.0
MAL_KHERSON_BS77 = -0.04        # mean of the 2019-2022 daily means at 80805 (-0.07, -0.10, +0.01, -0.01), m BS77
G_M_PER_KM = 0.0014             # pre-breach water-surface gradient (p27, Finding 6)
SIGMA_SND = 0.30                # chart reading / datum error of a hand-digitised depth, m
SIGMA_THALWEG, SIGMA_THALWEG_INTERP, SIGMA_EMODNET = 1.0, 1.5, 1.0
R_THALWEG = 1000.0              # soundings within this radius of a SWORD node define its thalweg depth
SUPPORT_M = 2000.0              # cells farther than this from any sounding are 'densified/shore-supported only' (source 2), kept but flagged
THIN_SHORE = 7                  # every 7th p27 vertex (~1 km): 3 651 vertices at thin 3 drowned the 626 soundings in every K=32 neighbourhood (CV bias +3.6 m)
SIGMA_SHORE_MIN = 1.5           # a 20 m shoreline pixel on a bank slope is a bed elevation to ~1-2 m, not to the 0.2-0.4 m stage sigma of p27
SWORD = CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet"
EMODNET = CFG.BULK_ROOT / "data_swot/processed/emodnet/D0_liman_background.tif"
# --- repair 2026-09-20: domain permission and boundary conditioning -------------------------------------------
P66_CLASS = "outputs/rasters/zone{n}/prebreach/prebreach_class.tif"   # 1 CORE 2 LEVEL_DEPENDENT 3 LAND 4 ISLAND 5 UNCERTAIN
CORE_CLASSES = (1,)             # only corroborated persistent water may carry a bathymetric prediction
LAND_CLASSES = (3, 4)           # LAND and ISLAND are the ONLY valid counterpart of a true shoreline
MAX_NEAREST_M = SUPPORT_M       # hard gate now, not a flag: no sounding within this -> not a prediction target
MIN_NEIGHBORS = 3               # ... and at least this many soundings within SUPPORT_RADIUS_M
SUPPORT_RADIUS_M = SUPPORT_M
STATUS = {0: "NoData / bathymetry prohibited", 1: "supported interpolated bathymetry",
          2: "true-shore-conditioned bathymetry", 4: "IDW fallback (count only, see tags)"}
SUPPORT_SOURCES = ("sounding", "emodnet_dtm2024")   # observations that count as bathymetric SUPPORT: the measured depths
                          # and, where the run actually uses them, the EMODnet lattice samples. Soundings-only support made
                          # the gate drop 83 % of the ZONE_3 estuary even in the production configuration whose solver was
                          # already being fed those EMODnet points. Shoreline vertices and SWORD thalweg nodes are NOT
                          # support: they are constraints derived from the boundary and the centreline, not independent
                          # observations of the bed.


def channel_fitted(line, xy: np.ndarray, aniso: float) -> np.ndarray:
    """Channel-fitted coordinates (Merwade-type): s = distance along the SWORD centreline, d = signed distance across it.
    Kriging in (s, aniso*d) makes the across-channel range 1/aniso of the along-channel one, which is what a river bed looks like:
    a shoreline 300 m across the channel is then as 'far' as a point 1.5 km along it (aniso 5)."""
    pts = shapely.points(xy[:, 0], xy[:, 1]); sdist = shapely.line_locate_point(line, pts); on = shapely.line_interpolate_point(line, sdist); ahead = shapely.line_interpolate_point(line, np.minimum(sdist + 1.0, line.length))
    ox, oy = shapely.get_x(on), shapely.get_y(on); tx, ty = shapely.get_x(ahead) - ox, shapely.get_y(ahead) - oy; nrm = np.hypot(tx, ty) + 1e-9
    d = ((xy[:, 0] - ox) * (-ty) + (xy[:, 1] - oy) * tx) / nrm            # signed perpendicular offset (left of the downstream direction positive)
    return np.c_[sdist, aniso * d]


def kherson_chain(sw: pd.DataFrame) -> float:
    kx, ky = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform(CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3])
    i = cKDTree(np.c_[sw.x.values, sw.y.values]).query([kx, ky])[1]; return float(sw.chain_km.values[i])


def mal_evrf(delta9902: np.ndarray, s_km: np.ndarray) -> np.ndarray:
    return MAL_KHERSON_BS77 + delta9902 + G_M_PER_KM * s_km


def water_polygon_mask(zone: str, n: int, G: dict) -> tuple[np.ndarray, list]:
    """Pre-breach water: leaf-on water_share >= 50 % in ANY year 2019-2022 with a composite (the intersection over years collapsed to
    14 km2 in ZONE_4 because 2019/2022 have 1-3 dates with partial coverage), UNION the p27 dated PRE_BREACH shoreline polygon."""
    water, years = np.zeros((G["ny"], G["nx"]), bool), []
    for y in (2019, 2020, 2021, 2022):
        f = ROOT / f"outputs/rasters/zone{n}/annual/zone{n}_water_share_{y}_leafon_20m.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as ds:
            w = np.full((G["ny"], G["nx"]), 255, "u1")
            reproject(source=rasterio.band(ds, 1), destination=w, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC, resampling=Resampling.nearest, dst_nodata=255)
        water |= (w != 255) & (w >= 50); years.append(y)
    gp = ROOT / f"data/processed/bathymetry/zone_shore_contours_{zone}_PRE_BREACH.gpkg"
    if gp.exists():
        poly = gpd.read_file(gp, layer="contours").geometry.union_all()
        water |= features.rasterize([(poly, 1)], out_shape=water.shape, transform=G["transform"], fill=0, dtype="uint8").astype(bool); years.append("p27")
    # the below-dam DEM must never enter the reservoir: cut the registry pool (+200 m) and everything east of the dam + 1 km
    # (found 2026-09-19: the ZONE_4 frame reaches 569 km E and the lower 40 km of the pool were kriged from below-dam soundings)
    from shapely.geometry import Point
    pool = SD.load_utm("reservoir_full_pool_prebreach").buffer(200.0)
    water &= ~features.rasterize([(pool, 1)], out_shape=water.shape, transform=G["transform"], fill=0, dtype="uint8").astype(bool)
    dam = gpd.GeoSeries([Point(*CFG.KAKHOVKA_DAM)], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]; xs_ = G["x0"] + (np.arange(G["nx"]) + 0.5) * CELL; water[:, xs_ > dam.x + 1000.0] = False
    water = ndimage.binary_opening(water, iterations=1)                     # drop 1-cell specks
    return water, years


def core_domain_mask(zone: str, n: int, G: dict) -> np.ndarray:
    """p66 CORE classes on p53's grid. Grid mismatch is an explicit error, resampling is nearest and logged."""
    p = ROOT / P66_CLASS.format(n=n)
    if not p.exists():
        raise SystemExit(f"missing p66 class raster {p} -- run scripts/p66_prebreach_water_domain.py --zones {zone} first")
    with rasterio.open(p) as ds:
        if ds.crs != CFG.CRS_METRIC:
            raise SystemExit(f"{p.name}: CRS {ds.crs} != {CFG.CRS_METRIC}")
        gb = (G["x0"], G["y1"] - G["ny"] * CELL, G["x0"] + G["nx"] * CELL, G["y1"])
        # The two grids are built on the same polygon at 20 m and 30 m with cell-centre registration, so their extents
        # can differ by less than one cell. A shortfall up to one cell is tolerated AND LOGGED; anything larger is an
        # error. Uncovered cells receive class 0, which is not in CORE_CLASSES, so they carry no bathymetry: the
        # tolerance fails closed.
        short = max(ds.bounds.left - gb[0], ds.bounds.bottom - gb[1], gb[2] - ds.bounds.right, gb[3] - ds.bounds.top, 0.0)
        if short > CELL:
            raise SystemExit(f"{p.name} bounds {tuple(round(v) for v in ds.bounds)} fall {short:.0f} m short of the p53 grid "
                             f"{tuple(round(v) for v in gb)} (more than one {CELL:.0f} m cell)")
        if short > 0:
            print(f"  p66 raster is {short:.0f} m (< 1 cell) short of the p53 grid at the frame edge; those cells become class 0 = no bathymetry")
        print(f"  p66 classes: {p.name}, {ds.res[0]:.0f} m -> {CELL:.0f} m by NEAREST (class raster, never averaged)")
        cls = np.zeros((G["ny"], G["nx"]), "u1")
        reproject(source=rasterio.band(ds, 1), destination=cls, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                  resampling=Resampling.nearest, src_nodata=ds.nodata, dst_nodata=0)
    return cls


def shore_masks(core: np.ndarray, valid: np.ndarray, cls: np.ndarray):
    """(true_shore, artificial_edge). A taper to MAL is a SHORELINE boundary condition, so it is allowed only where
    the CORE boundary actually meets corroborated land. Every other edge of the prediction domain -- the limit of
    observational support, an UNCERTAIN neighbour, the frame edge -- is artificial and must not be tapered."""
    land = np.isin(cls, LAND_CLASSES)
    core_boundary = core & ~ndimage.binary_erosion(core, iterations=1, border_value=1)
    true_shore = core_boundary & ndimage.binary_dilation(land, iterations=1)
    valid_boundary = valid & ~ndimage.binary_erosion(valid, iterations=1, border_value=1)
    return true_shore, valid_boundary & ~true_shore


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zone", required=True, choices=list(ZONES)); ap.add_argument("--emodnet", action="store_true"); ap.add_argument("--shore-mode", choices=("points", "taper"), default="taper", help="points: p27 vertices as soft data; taper: kriging on soundings+thalweg only, then the surface is blended linearly to MAL within TAPER_M of the shoreline (boundary honoured without drowning the soundings)")
    ap.add_argument("--taper-m", type=float, default=150.0)
    ap.add_argument("--thalweg", choices=("on", "off"), default="off", help="off (default): the SWORD parquet is not a monotonic centreline (474 jumps > 2 km; branches interleaved by chain_km), so along-chainage interpolation and max-depth thalweg points are unreliable (CV RMSE 2.48 vs 2.29, bias -0.41) -- kept only as an experiment")
    ap.add_argument("--domain", choices=("core", "legacy"), default="core", help="core (default): p66 CORE + hard sounding support decide the prediction targets; legacy: the pre-repair union water polygon, kept runnable for the ablation baseline")
    ap.add_argument("--aniso", type=float, default=1.0, help="across/along-channel anisotropy in channel-fitted coordinates (1 = isotropic in x,y)"); a = ap.parse_args()
    zone, n = a.zone, ZONES[a.zone]; t0 = time.time()
    fp = SD.load_utm(zone); gr = SD.build_grid(fp, CELL, what=f"{zone} bed v2 grid"); gx, gy = gr["gx"], gr["gy"]
    G = dict(nx=len(gx), ny=len(gy), x0=float(gx.min() - CELL / 2), y1=float(gy.max() + CELL / 2)); G["transform"] = rasterio.transform.from_origin(G["x0"], G["y1"], CELL, CELL)   # rows run N->S regardless of build_grid's gy order
    # --- soundings with the MAL datum
    m = pd.read_parquet(ROOT / "data/processed/bathymetry/manual_soundings_evrf2019.parquet"); m = m[m[zone]].copy()
    sw = pd.read_parquet(SWORD); tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); sw["x"], sw["y"] = tf.transform(sw.lon.values, sw.lat.values)
    sw = sw.sort_values("chain_km").reset_index(drop=True); ck = kherson_chain(sw); tree_sw = cKDTree(np.c_[sw.x.values, sw.y.values])
    m["s_km"] = sw.chain_km.values[tree_sw.query(np.c_[m.x.values, m.y.values])[1]] - ck
    m["H"] = mal_evrf(m.delta_epsg9902_m.values, m.s_km.values) - m.depth_m.values
    m = m.assign(_kx=m.x.round(0), _ky=m.y.round(0)).groupby(["_kx", "_ky"], as_index=False).agg(x=("x", "mean"), y=("y", "mean"), H=("H", "mean"), depth_m=("depth_m", "mean"), s_km=("s_km", "mean"))
    print(f"{zone}: {len(m)} soundings, depth {m.depth_m.min():.0f}..{m.depth_m.max():.0f} m, H {m.H.min():.1f}..{m.H.max():.1f} m EVRF2019; MAL(x) - (-depth+delta) = {np.mean(MAL_KHERSON_BS77 + G_M_PER_KM*m.s_km):+.3f} m on average", flush=True)
    # --- water polygon 2019-2022
    water, years = water_polygon_mask(zone, n, G); print(f"  pre-breach water polygon (legacy union): {water.sum()*CELL*CELL/1e6:.0f} km2 from leaf-on {years}")
    if a.domain == "core":
        cls = core_domain_mask(zone, n, G); core = np.isin(cls, CORE_CLASSES)
        print(f"  p66 CORE: {core.sum()*CELL*CELL/1e6:.1f} km2 (legacy union admitted {water.sum()*CELL*CELL/1e6:.1f} km2; "
              f"{(water & ~core).sum()*CELL*CELL/1e6:.1f} km2 of it is NOT corroborated persistent water)")
    else:
        cls = np.zeros((G["ny"], G["nx"]), "u1"); core = water.copy()
        print("  domain = legacy union water polygon (ablation baseline A)")
    # --- shoreline soft constraints (p27), thinned
    v = gpd.read_file(ROOT / f"data/processed/bathymetry/zone_shore_contours_{zone}_PRE_BREACH.gpkg", layer="vertices").iloc[::THIN_SHORE]
    shore = pd.DataFrame(dict(x=v.x.values, y=v.y.values, H=v.H_evrf2019_m.values, sigma=np.maximum(v.sigma_m.values, SIGMA_SHORE_MIN), source="shore_p27"))
    # --- thalweg densification along SWORD nodes inside the zone
    inside_sw = shapely.contains_xy(fp, sw.x.values, sw.y.values); nodes = sw[inside_sw].copy()
    ts = cKDTree(np.c_[m.x.values, m.y.values]); th = np.full(len(nodes), np.nan)
    for i, (x, y) in enumerate(zip(nodes.x.values, nodes.y.values)):
        idx = ts.query_ball_point([x, y], R_THALWEG)
        if len(idx) >= 2:
            th[i] = np.max(m.depth_m.values[idx])
    have = np.isfinite(th); depth = th.copy(); sigma = np.full(len(nodes), np.nan)
    if have.any():
        lo, hi = np.flatnonzero(have).min(), np.flatnonzero(have).max(); ar = np.arange(len(nodes)); interp = ~have & (ar > lo) & (ar < hi)
        depth[interp] = np.interp(nodes.chain_km.values[interp], nodes.chain_km.values[have], th[have]); sigma[have] = SIGMA_THALWEG; sigma[interp] = SIGMA_THALWEG_INTERP
    nodes = nodes.assign(thalweg_depth=depth, sigma=sigma, s_km=nodes.chain_km - ck); nodes = nodes[np.isfinite(nodes.thalweg_depth)].copy()
    d9902 = float(np.median(pd.read_parquet(ROOT / "data/processed/bathymetry/manual_soundings_evrf2019.parquet").delta_epsg9902_m))
    nodes["H"] = mal_evrf(d9902, nodes.s_km.values) - nodes.thalweg_depth.values
    thal = pd.DataFrame(dict(x=nodes.x.values, y=nodes.y.values, H=nodes.H.values, sigma=nodes.sigma.values, source="thalweg_sword"))
    print(f"  thalweg points: {len(thal)} SWORD nodes ({int((nodes.sigma == SIGMA_THALWEG).sum())} from soundings within {R_THALWEG:.0f} m, rest interpolated along chainage)")
    # --- EMODnet lattice (ZONE_3)
    emo = pd.DataFrame(dict(x=np.zeros(0), y=np.zeros(0), H=np.zeros(0), sigma=np.zeros(0), source=np.array([], dtype=object)))
    if a.emodnet and EMODNET.exists():
        lx = np.arange(gx.min(), gx.max(), 250.0); ly = np.arange(gy.min(), gy.max(), 250.0); LX, LY = np.meshgrid(lx, ly); px, py = LX.ravel(), LY.ravel()
        cc, rr = ~G["transform"] * (px, py); cc = np.floor(cc).astype(int); rr = np.floor(rr).astype(int); ok = (cc >= 0) & (cc < G["nx"]) & (rr >= 0) & (rr < G["ny"]); ok[ok] &= water[rr[ok], cc[ok]]
        far = ts.query(np.c_[px, py])[0] > 1000.0; sel = ok & far
        with rasterio.open(EMODNET) as ds:
            ev = np.array([q[0] for q in ds.sample(np.c_[px[sel], py[sel]])], "f8")
        good = np.isfinite(ev) & (ev > -40) & (ev < 1)
        emo = pd.DataFrame(dict(x=px[sel][good], y=py[sel][good], H=ev[good], sigma=SIGMA_EMODNET, source="emodnet_dtm2024")); print(f"  EMODnet lattice points: {len(emo)}")
    # --- training set
    snd = pd.DataFrame(dict(x=m.x.values, y=m.y.values, H=m.H.values, sigma=SIGMA_SND, source="sounding"))
    if a.shore_mode == "taper":
        shore = shore.iloc[0:0]
    if a.thalweg == "off":
        thal = thal.iloc[0:0]
    P = pd.concat([snd, shore, thal, emo], ignore_index=True); P.to_parquet(CFG.TABLES / f"p53_bed_v2_points_{zone}.parquet", index=False)
    xy_real = np.c_[P.x.values.astype(float), P.y.values.astype(float)]; z_all = P.H.values.astype(float); ev_all = P.sigma.values.astype(float) ** 2; is_snd = (P.source == "sounding").values
    line = shapely.LineString(np.c_[sw.x.values, sw.y.values])                 # SWORD main channel, sorted by chainage
    T = (lambda xy: channel_fitted(line, xy, a.aniso)) if a.aniso != 1.0 else (lambda xy: xy)
    xy_all = T(xy_real); print(f"  coordinates: {'channel-fitted (s, %.0f*d)' % a.aniso if a.aniso != 1.0 else 'x, y'}")
    vg = fit_variogram(xy_all[is_snd], snd.H.values, CFG.SEED); print(f"  variogram (soundings): range {vg[0]/1e3:.2f} km, sill {vg[1]:.2f}, nugget {vg[2]:.2f}")
    # --- blocked CV on soundings
    xy_snd = xy_all[is_snd]; xy_snd_real = xy_real[is_snd]; z_snd = snd.H.values; nn = cKDTree(xy_snd_real).query(xy_snd_real, k=2)[0][:, 1]; block_km = max(0.5, round(np.median(nn) / 200.0) * 0.2); folds = P19.block_folds(xy_snd_real, block_km, CFG.SEED)
    rows = []
    for label, use_cons in (("OK_SOFT sounding" + ("+thalweg" if len(thal) else "") + ("+shore" if len(shore) else "") + ("+emodnet" if len(emo) else ""), True), ("OK soundings only", False)):
        resid = np.full(len(z_snd), np.nan)
        for f in range(P19.N_FOLDS):
            te = folds == f
            if te.sum() == 0:
                continue
            tr = ~te
            xy_tr = np.vstack([xy_snd[tr], xy_all[~is_snd]]) if use_cons else xy_snd[tr]; z_tr = np.concatenate([z_snd[tr], z_all[~is_snd]]) if use_cons else z_snd[tr]
            ev = np.concatenate([np.full(tr.sum(), SIGMA_SND ** 2), ev_all[~is_snd]]) if use_cons else np.full(tr.sum(), SIGMA_SND ** 2)
            resid[te] = KR.ok_soft(xy_tr, z_tr, ev, xy_snd[te], vg) - z_snd[te]
        r = resid[np.isfinite(resid)]; rows.append(dict(zone=zone, model=label, n=len(r), RMSE=float(np.sqrt((r ** 2).mean())), MAE=float(np.abs(r).mean()), bias=float(r.mean()), NMAD=float(1.4826 * np.median(np.abs(r - np.median(r)))), block_km=block_km))
        print(f"  CV {label}: RMSE {rows[-1]['RMSE']:.3f} bias {rows[-1]['bias']:+.3f} NMAD {rows[-1]['NMAD']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(CFG.TABLES / f"p53_bed_v2_cv_{zone}.csv", index=False)
    # --- observational support decided BEFORE kriging: an unsupported cell is not a target at all
    r0, c0 = np.nonzero(core); cand_real = np.c_[G["x0"] + (c0 + 0.5) * CELL, G["y1"] - (r0 + 0.5) * CELL]
    is_sup = np.isin(P.source.values, SUPPORT_SOURCES)                      # the observations THIS run actually gave the solver
    sup_xy = xy_real[is_sup]; sup_tree = cKDTree(sup_xy)
    print(f"  support set: {int(is_sup.sum()):,} observations ({', '.join(f'{k} {int((P.source.values == k).sum()):,}' for k in SUPPORT_SOURCES if (P.source.values == k).any())})")
    d1 = sup_tree.query(cand_real, k=1)[0]
    n_loc = np.asarray(sup_tree.query_ball_point(cand_real, r=SUPPORT_RADIUS_M, return_length=True))
    supp = (d1 <= MAX_NEAREST_M) & (n_loc >= MIN_NEIGHBORS) if a.domain == "core" else np.ones(len(d1), bool)
    valid_domain = np.zeros(core.shape, bool); valid_domain[r0[supp], c0[supp]] = True
    dist_raster = np.full(core.shape, np.nan, "f4"); dist_raster[r0, c0] = d1
    print(f"  support gate: {supp.sum():,} of {len(supp):,} CORE cells keep a target "
          f"({valid_domain.sum()*CELL*CELL/1e6:.1f} km2); dropped {int((~supp).sum()):,} "
          f"(d1 > {MAX_NEAREST_M:.0f} m or fewer than {MIN_NEIGHBORS} soundings within {SUPPORT_RADIUS_M:.0f} m)")
    # --- production surface: kriging sees ONLY the admissible targets
    rr, cc = np.nonzero(valid_domain); tx = G["x0"] + (cc + 0.5) * CELL; ty = G["y1"] - (rr + 0.5) * CELL; tgt_real = np.c_[tx, ty]; tgt = T(tgt_real)
    surf = np.full(core.shape, np.nan, "f4")
    if len(tgt):
        pred, diag = KR.ok_soft(xy_all, z_all, ev_all, tgt, vg, return_diag=True); surf[rr, cc] = pred
    else:
        pred = np.zeros(0); diag = {"n_fallback": 0}
    taper_zone = np.zeros(core.shape, bool)
    if a.shore_mode == "taper" and len(tgt):
        mal_t = mal_evrf(d9902, sw.chain_km.values[tree_sw.query(tgt_real)[1]] - ck)
        if a.domain == "core":
            # distance_transform_edt knows no semantics: it returns the distance to the nearest ZERO of whatever mask it
            # is given, so the old call treated every edge of the water mask as a shoreline. A taper to MAL is a
            # shoreline boundary condition and is applied only where the nearest boundary is a validated shoreline.
            true_shore, art_edge = shore_masks(core, valid_domain, cls)
            d_true = ndimage.distance_transform_edt(~true_shore) * CELL if true_shore.any() else np.full(core.shape, np.inf, "f4")
            d_art = ndimage.distance_transform_edt(~art_edge) * CELL if art_edge.any() else np.full(core.shape, np.inf, "f4")
            tz = valid_domain & (d_true <= a.taper_m) & (d_true < d_art)
            u = np.clip(d_true / a.taper_m, 0.0, 1.0); w3 = u * u * (3.0 - 2.0 * u)      # smoothstep: no slope break at the end
            sel = tz[rr, cc]
            surf[rr[sel], cc[sel]] = mal_t[sel] + w3[rr[sel], cc[sel]] * (pred[sel] - mal_t[sel])
            taper_zone = tz
            print(f"  true-shore taper {a.taper_m:.0f} m: shoreline cells {int(true_shore.sum()):,}, artificial edge {int(art_edge.sum()):,}, "
                  f"tapered {int(tz.sum()):,} of {int(valid_domain.sum()):,} ({tz.sum()/max(valid_domain.sum(),1):.0%} of the surface)")
        else:
            dshore = ndimage.distance_transform_edt(water) * CELL
            wgt = np.clip((dshore - CELL) / a.taper_m, 0, 1)[rr, cc]
            surf[rr, cc] = wgt * pred + (1 - wgt) * mal_t
            taper_zone[rr[wgt < 1], cc[wgt < 1]] = True
            print(f"  LEGACY taper {a.taper_m:.0f} m applied to {int((wgt < 1).sum()):,} of {len(wgt):,} cells ({(wgt < 1).mean():.0%} of the surface)")
    dist_snd = sup_tree.query(tgt_real, k=1)[0] if len(tgt) else np.zeros(0)
    status = np.zeros(core.shape, "u1"); status[rr, cc] = 1; status[taper_zone & valid_domain] = 2
    status[~np.isfinite(surf)] = 0
    print(f"  surface: {len(tgt):,} cells, {diag['n_fallback']} IDW fallbacks; "
          f"status 1 {int((status==1).sum()):,}, status 2 {int((status==2).sum()):,}")
    # shoreline check (p52 B) on the new surface
    inner = valid_domain & ~ndimage.binary_erosion(valid_domain, iterations=1, border_value=1); deep = ndimage.binary_erosion(valid_domain, iterations=10)
    exp_shore = float(np.mean(mal_evrf(d9902, np.array([0.0]))))
    ring = surf[inner & np.isfinite(surf)]; corev = surf[deep & np.isfinite(surf)]
    # NOT an independent acceptance test: where the taper fires it forces the edge to MAL and this check then only
    # confirms the assignment. Acceptance is land/island false positives against p66 classes (see p53b_land_fpr.py).
    print(f"  [diagnostic, not independent] ring p50 {np.median(ring) if len(ring) else float('nan'):+.2f} m "
          f"(expected ~{exp_shore:+.2f}); interior p50 {np.median(corev) if len(corev) else float('nan'):+.2f} m")
    prof = rasterio.profiles.DefaultGTiffProfile(count=1, dtype="float32", width=G["nx"], height=G["ny"], crs=CFG.CRS_METRIC, transform=G["transform"], nodata=-9999.0, compress="deflate")
    out = ROOT / f"outputs/rasters/zone{n}"; out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out / f"zone{n}_bed_v2_PRE_BREACH_30m.tif", "w", **prof) as ds:
        ds.write(np.nan_to_num(surf, nan=-9999.0), 1)
        ds.update_tags(quantity="bed elevation, m EVRF2019 (H = MAL(x) - chart depth)", datum="EVRF2019 via EPSG:9902; MAL Kherson 2019-2022 = -0.04 m BS77 + G*s", boundary=f"pre-breach water polygon, leaf-on water_share>=50 % in {years}",
                       coordinates=f"channel-fitted, anisotropy {a.aniso}" if a.aniso != 1.0 else "x,y", shore_mode=f"{a.shore_mode} ({a.taper_m:.0f} m)", training=f"soundings {len(snd)} (sigma {SIGMA_SND}), shore vertices {len(shore)} (p27, thin {THIN_SHORE}), thalweg nodes {len(thal)}, emodnet {len(emo)}", variogram=f"spherical range {vg[0]:.0f} m sill {vg[1]:.2f} nugget {vg[2]:.2f}", producer="p53_below_dam_bed_dem.py")
    p2 = dict(prof); p2.update(dtype="uint8", nodata=0)
    with rasterio.open(out / f"zone{n}_bed_v2_status_30m.tif", "w", **p2) as ds:
        ds.write(status, 1); ds.update_tags(values="; ".join(f"{k} {v}" for k, v in STATUS.items()), domain=a.domain,
                                            contract="a finite elevation is NOT sufficient: consumers must require status in (1,2)",
                                            idw_fallbacks=str(diag["n_fallback"]), producer="p53_below_dam_bed_dem.py")
    with rasterio.open(out / f"zone{n}_bed_v2_source_30m.tif", "w", **p2) as ds:                    # kept for backward compatibility
        ds.write(np.where(status == 0, 0, np.where(status == 2, 2, 1)).astype("u1"), 1)
        ds.update_tags(values="1 supported interpolated; 2 true-shore conditioned; 0 no bathymetry", producer="p53_below_dam_bed_dem.py")
    with rasterio.open(out / f"zone{n}_bed_v2_domain_class_30m.tif", "w", **p2) as ds:
        ds.write(cls, 1); ds.update_tags(values="p66 prebreach classes: 1 CORE 2 LEVEL_DEPENDENT 3 LAND 4 ISLAND 5 UNCERTAIN", producer="p53_below_dam_bed_dem.py")
    p3 = dict(prof); p3.update(dtype="float32", nodata=-9999.0)
    with rasterio.open(out / f"zone{n}_bed_v2_support_distance_30m.tif", "w", **p3) as ds:
        ds.write(np.nan_to_num(dist_raster, nan=-9999.0), 1)
        ds.update_tags(quantity="distance to the nearest sounding, m", gate=f"target requires d1 <= {MAX_NEAREST_M:.0f} m and >= {MIN_NEIGHBORS} support observations within {SUPPORT_RADIUS_M:.0f} m", support_sources="|".join(SUPPORT_SOURCES), producer="p53_below_dam_bed_dem.py")
    summ = dict(zone=zone, domain=a.domain, cell_m=CELL, aniso=a.aniso, shore_mode=a.shore_mode, taper_m=a.taper_m, soundings=len(snd), shore_vertices=len(shore), thalweg_nodes=len(thal), emodnet_points=len(emo),
                water_km2=round(water.sum() * CELL * CELL / 1e6, 1), core_km2=round(float(core.sum()) * CELL * CELL / 1e6, 2), valid_km2=round(float(valid_domain.sum()) * CELL * CELL / 1e6, 2),
                cells_valid=int(valid_domain.sum()), cells_taper=int(taper_zone.sum()), taper_fraction=round(float(taper_zone.sum() / max(valid_domain.sum(), 1)), 3), idw_fallbacks=int(diag["n_fallback"]),
                sounding_supported_share=round(float(np.mean(dist_snd <= SUPPORT_M)) if len(dist_snd) else float("nan"), 3),
                cv_rmse_ok_soft=rows[0]["RMSE"], cv_rmse_snd_only=rows[1]["RMSE"], cv_bias_snd_only=rows[1]["bias"], n_soundings=len(snd), shoreline_ring_p50=float(np.median(ring)) if len(ring) else float("nan"), expected_shoreline=exp_shore, core_p50=float(np.median(corev)) if len(corev) else float("nan"), vg_range_m=vg[0], years="|".join(map(str, years)))
    sp = CFG.TABLES / "p53_bed_v2_summary.csv"; S = pd.concat([pd.read_csv(sp), pd.DataFrame([summ])], ignore_index=True) if sp.exists() else pd.DataFrame([summ]); S = S.drop_duplicates(["zone", "domain"], keep="last") if "domain" in S else S.drop_duplicates("zone", keep="last"); S.to_csv(sp, index=False)
    print(f"-> {out / f'zone{n}_bed_v2_PRE_BREACH_30m.tif'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
