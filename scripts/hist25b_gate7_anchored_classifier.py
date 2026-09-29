#!/usr/bin/env python
"""GATE 7 — anchored SAR water classification on the corrected domain.

Proceeds from the frozen Gate-6 manifest. M0 (Otsu) is a frozen baseline that
is never tuned; M1-M3 are the candidate anchored models, predeclared below.
M4-M5 are deliberately not implemented: Otsu already failed in 7 of 8 scenes
for want of a bimodal histogram, so a wide sweep now would buy model shopping
rather than insight.

TWO INDEPENDENT AXES. Spatial completeness and classification quality are
separate and must not be traded against each other. An event covering 80% of
the domain can still produce an excellent shoreline on that 80%, so coverage
never enters a classifier score, and >=95% coverage is not a quality
criterion. Pixels outside valid coverage are NO_DATA and are never treated as
LAND, never scored, and never used to close a contour.

WHAT SAR PRODUCES HERE. A water-land boundary, not a bed elevation. A
shoreline becomes an elevation constraint only where the Gate-6 eligibility
already holds: PREBREACH_IMPOUNDED, WSE resolved, WSE semantics resolved, and
a target-stage role allowed. Active-drawdown imagery stays available for
transition morphology and is never promoted to H1/H2/H3 geometry.

THE SCIENTIFIC TARGET is not how much water gets classified. It is

    L_new = length of trustworthy shoreline inside the 15.32-17.08 m band
            that neither the primary soundings nor CMAP2020 constrain

The primary survey stops at 15.3153 m and CMAP2020 stops 0.0995 m BELOW that,
so 1.77 m of upper slope currently has no internal support at all. If SAR puts
defensible shoreline there, it is a stronger addition than CMAP ever was.

MODELS (predeclared, parameters fixed before any metric is computed)
    M0  Otsu on log10 VV and VH, the inherited engine, frozen baseline
    M1  anchored 1-D VV threshold, midpoint of the two anchor medians in dB
    M2  anchored 1-D VH threshold, same construction
    M3  anchored 2-D VV+VH linear discriminant from the anchor covariances

Outputs
-------
outputs/tables/hist25b_gate7_model_comparison.csv
outputs/tables/hist25b_gate7_event_metrics.csv
outputs/tables/hist25b_gate7_topology_qa.csv
outputs/tables/hist25b_gate7_upper_margin_support.csv
outputs/figures/historical_bathymetry/png/hist25b_gate7_*.png
"""
from __future__ import annotations

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
from rasterio.features import rasterize as rio_rasterize
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import phase19_s1_watermask as S1
from hist25b_gate6_event_qualification import load_frozen_manifest

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CELL = 20.0
CACHE = CFG.S1_CACHE / "ZONE_1_reservoir_corrected"
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
CONTOURS_CMAP = ROOT / "data/processed/bathymetry/cmap2020_contours_utm.gpkg"
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"

WATER_CORE_BUFFER_M = 500.0     # inside the lowest contour, away from any shore
LAND_BUFFER_M = 100.0           # outside the highest contour
EDGE_BAND_M = 60.0              # ambiguous belt either side of a contour
HOLDOUT_BLOCK_KM = 10.0         # spatial blocks for anchor holdout
H_PRIMARY_MAX = 15.3153
H_SHORE = 17.0836
# Primary operational redundancy radius. 250/1000/2000 m are reported as
# sensitivity. The ~2 km variogram range is deliberately NOT used to define
# source independence: it is a spatial correlation scale of the bed surface,
# not the distance at which an existing sounding stops making a new shoreline
# redundant. Conflating the two would silently inflate or deflate L_new.
UNSUPPORTED_M = 500.0
REDUNDANCY_RADII_M = (250.0, 500.0, 1000.0, 2000.0)


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-6))


def build_grid(fp):
    b = fp.bounds
    x0, y0 = np.floor(b[0] / CELL) * CELL, np.floor(b[1] / CELL) * CELL
    x1, y1 = np.ceil(b[2] / CELL) * CELL, np.ceil(b[3] / CELL) * CELL
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    tr = from_origin(x0, y1, CELL, CELL)
    inside = rio_rasterize([(fp, 1)], out_shape=(ny, nx), transform=tr,
                           fill=0, dtype="uint8").astype(bool)
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, nx=nx, ny=ny, tr=tr, inside=inside)


MIN_PART_KM2 = 0.05        # speckle fragments are not shoreline
MIN_HOLE_KM2 = 0.05


def clean_water(mask):
    """Drop speckle fragments and pinholes before measuring a shoreline.

    Taking the boundary of a raw classified mask counts the perimeter of every
    fragment: with ~4,000 components that returns 13,000 km of "shoreline"
    against a real reservoir perimeter of ~1,540 km. Cleaning first is what
    makes the length physical. This is NOT largest_component -- genuine
    secondary water bodies are kept."""
    px = CELL ** 2 / 1e6
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n:
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * px
        keep = np.zeros(n + 1, bool)
        keep[1:] = sz >= MIN_PART_KM2
        mask = keep[lab]
    holes = ndimage.binary_fill_holes(mask) & ~mask
    hl, hn = ndimage.label(holes, structure=np.ones((3, 3), int))
    if hn:
        hs = ndimage.sum(np.ones_like(hl), hl, range(1, hn + 1)) * px
        fill = np.zeros(hn + 1, bool)
        fill[1:] = hs < MIN_HOLE_KM2
        mask = mask | fill[hl]
    return mask


def shoreline_px(mask):
    """Boundary pixels of a mask, without closing across NO_DATA."""
    return mask & ~ndimage.binary_erosion(mask, np.ones((3, 3), bool))


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    M, meta = load_frozen_manifest()
    print(f"Gate-6 manifest verified: sha256 {meta['sha256'][:16]}..., "
          f"{len(M)} selected events")

    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    print(f"grid {G['nx']}x{G['ny']} at {CELL:.0f} m; domain "
          f"{inside.sum()*CELL**2/1e6:,.0f} km2")

    # ------------------------------------------------ anchors (5 classes)
    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")
    inv = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv"
                      ).sort_values("H_evrf2019_m")
    lowest, highest = inv.contour_id.iloc[0], inv.contour_id.iloc[-1]

    def rast(g):
        return rio_rasterize([(g, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0, dtype="uint8").astype(bool)
    w_lo, w_hi = rast(gdf.loc[lowest, "geometry"]), rast(gdf.loc[highest, "geometry"])
    d_in_lo = ndimage.distance_transform_edt(w_lo) * CELL
    d_out_hi = ndimage.distance_transform_edt(~w_hi) * CELL
    d_edge = ndimage.distance_transform_edt(~shoreline_px(w_hi)) * CELL

    # WorldCover wetland, so reed margins are not represented only by generic
    # land/water anchors
    wet = np.zeros(inside.shape, bool)
    try:
        import p0b_build_dnipro_water_domain as P0B
        import rasterio
        from rasterio.warp import Resampling as WR, reproject
        tok = P0B.sas_token()
        for f in P0B.search_worldcover([32.2, 46.5, 35.5, 48.0]):
            tmp = np.zeros(inside.shape, np.uint8)
            with rasterio.open(f["assets"]["map"]["href"] + "?" + tok) as ds:
                reproject(source=rasterio.band(ds, 1), destination=tmp,
                          dst_transform=G["tr"], dst_crs="EPSG:32636",
                          dst_nodata=0, resampling=WR.nearest)
            wet |= (tmp == 90)
        print(f"WorldCover herbaceous wetland: {wet.sum()*CELL**2/1e6:,.0f} km2")
    except Exception as ex:
        print(f"wetland anchor unavailable ({type(ex).__name__}); "
              f"VEGETATED_WETLAND class will be empty")

    A_WATER = w_lo & (d_in_lo > WATER_CORE_BUFFER_M) & ~wet
    A_LAND = inside & ~w_hi & (d_out_hi > LAND_BUFFER_M) & ~wet
    A_EDGE = inside & (d_edge <= EDGE_BAND_M)
    A_WET = inside & wet
    print(f"\nanchors: CONFIDENT_WATER {A_WATER.sum()*CELL**2/1e6:7,.0f} km2 | "
          f"CONFIDENT_LAND {A_LAND.sum()*CELL**2/1e6:7,.0f} km2")
    print(f"         AMBIGUOUS_EDGE {A_EDGE.sum()*CELL**2/1e6:7,.0f} km2 | "
          f"VEGETATED_WETLAND {A_WET.sum()*CELL**2/1e6:7,.0f} km2")
    print("  anchors exclude the edge belt, so training never sees the "
          "boundary the models are judged on")

    # spatial holdout blocks: anchors and validation must not share ground
    bx = ((np.arange(G["nx"]) * CELL) // (HOLDOUT_BLOCK_KM * 1000)).astype(int)
    by = ((np.arange(G["ny"]) * CELL) // (HOLDOUT_BLOCK_KM * 1000)).astype(int)
    blk = (by[:, None] * 1000 + bx[None, :])
    train_blk = (blk % 2 == 0)
    print(f"  spatial holdout: {HOLDOUT_BLOCK_KM:.0f} km blocks, "
          f"{100*train_blk.mean():.0f}% train / {100*(~train_blk).mean():.0f}% test")

    # ------------------------------------------------ models
    print("\n" + "=" * 78)
    print("MODELS (parameters fixed before any metric is computed)")
    print("=" * 78)
    print("  M0  Otsu on log10 VV and VH -- inherited engine, frozen baseline")
    print("  M1  anchored 1-D VV threshold  (midpoint of anchor medians, dB)")
    print("  M2  anchored 1-D VH threshold  (same construction)")
    print("  M3  anchored 2-D VV+VH linear discriminant from anchor covariance")

    rows, ev_rows, topo, shorelines = [], [], [], {}
    elig = M[M.scientific_roles.fillna("").str.contains("TARGET_STAGE_GEOMETRY")
             & (M.temporal_regime == "PREBREACH_IMPOUNDED")
             & M.WSE_RESOLVED.astype(bool)
             & M.wse_semantics_resolved.astype(bool)]
    print(f"\n  events eligible for target-stage geometry: {len(elig)} of {len(M)}")
    print("  (classification runs on all; only these may become elevation "
          "constraints)")

    for r in M.itertuples():
        npz = CACHE / f"{r.event_id}.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        obs = cov & inside
        if obs.sum() < 5000:
            del vv, vh, cov
            continue
        vvd, vhd = db(vv), db(vh)

        aw = A_WATER & obs
        al = A_LAND & obs
        if aw.sum() < 500 or al.sum() < 500:
            del vv, vh, cov, vvd, vhd
            continue
        tr_w, tr_l = aw & train_blk, al & train_blk
        te_w, te_l = aw & ~train_blk, al & ~train_blk
        if min(tr_w.sum(), tr_l.sum(), te_w.sum(), te_l.sum()) < 200:
            del vv, vh, cov, vvd, vhd
            continue

        preds = {}
        # M0 frozen baseline
        t_vv = float(np.clip(S1.otsu_threshold(vv[obs]), S1.OTSU_MIN, S1.OTSU_MAX))
        t_vh = float(np.clip(S1.otsu_threshold(vh[obs]), S1.VH_MIN, S1.VH_MAX))
        p = np.zeros(inside.shape, bool)
        p[obs] = (vv[obs] < t_vv) & (vh[obs] < t_vh)
        preds["M0_otsu_baseline"] = p
        # M1 / M2 anchored 1-D
        for nm, arr in (("M1_anchored_VV", vvd), ("M2_anchored_VH", vhd)):
            thr = 0.5 * (np.median(arr[tr_w]) + np.median(arr[tr_l]))
            p = np.zeros(inside.shape, bool)
            p[obs] = arr[obs] < thr
            preds[nm] = p
        # M3 anchored 2-D linear discriminant
        Xw = np.c_[vvd[tr_w], vhd[tr_w]]
        Xl = np.c_[vvd[tr_l], vhd[tr_l]]
        mw, ml = Xw.mean(0), Xl.mean(0)
        Sw = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
        wv = np.linalg.solve(Sw, ml - mw)
        cut = 0.5 * (wv @ mw + wv @ ml)
        p = np.zeros(inside.shape, bool)
        p[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) < cut
        preds["M3_anchored_VV_VH_LDA"] = p

        for nm, p in preds.items():
            tp = float((p & te_w).sum()); fn = float((~p & te_w).sum())
            fp_ = float((p & te_l).sum()); tn = float((~p & te_l).sum())
            prec = tp / max(tp + fp_, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            iou = tp / max(tp + fp_ + fn, 1)
            ev_rows.append(dict(event_id=r.event_id, model=nm,
                                relative_orbit=int(r.relative_orbit),
                                temporal_regime=r.temporal_regime,
                                spatial_status=r.spatial_status,
                                radiometric_quality=r.radiometric_quality,
                                coverage=float(r.valid_coverage_fraction),
                                precision=prec, recall=rec, f1=f1, iou=iou,
                                water_km2=float((p & inside).sum()) * CELL ** 2 / 1e6))
            if nm not in shorelines:
                shorelines[nm] = {}
            shorelines[nm][r.event_id] = (p & inside).copy()
        del vv, vh, cov, vvd, vhd, preds
    E = pd.DataFrame(ev_rows)
    E.to_csv(CFG.TABLES / "hist25b_gate7_event_metrics.csv", index=False)
    print(f"\n  scored {E.event_id.nunique()} events x {E.model.nunique()} models")

    # ------------------------------------------------ model comparison
    print("\n" + "=" * 78)
    print("MODEL COMPARISON — bootstrap unit is the EVENT, never the pixel")
    print("=" * 78)
    agg = (E.groupby("model")
             .agg(events=("event_id", "nunique"),
                  precision=("precision", "median"),
                  recall=("recall", "median"), f1=("f1", "median"),
                  iou=("iou", "median"),
                  f1_p10=("f1", lambda s: float(np.percentile(s, 10))),
                  f1_iqr=("f1", lambda s: float(np.percentile(s, 75) -
                                                np.percentile(s, 25))),
                  water_km2=("water_km2", "median"))
             .reset_index().sort_values("f1", ascending=False))
    agg.to_csv(CFG.TABLES / "hist25b_gate7_model_comparison.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n  f1_p10 and f1_iqr measure stability across events; a model that "
          "wins on median but collapses on the worst events is not usable.")
    print("\n  per-orbit median F1:")
    print(E.pivot_table(index="model", columns="relative_orbit", values="f1",
                        aggfunc="median").to_string(
        float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------ topology QA
    print("\n" + "=" * 78)
    print("TOPOLOGY QA — metrics cannot see a swallowed island")
    print("=" * 78)
    kh = gpd.GeoSeries([Point(35.06, 47.84)], crs=4326).to_crs(32636).iloc[0]
    kc = int((kh.x - G["x0"]) / CELL); kr = int((G["y1"] - kh.y) / CELL)
    for nm, per in shorelines.items():
        isl, ncomp = [], []
        for eid, p in per.items():
            lab, n = ndimage.label(p, structure=np.ones((3, 3), int))
            ncomp.append(n)
            filled = ndimage.binary_fill_holes(p)
            holes = filled & ~p
            hl, hn = ndimage.label(holes, structure=np.ones((3, 3), int))
            big = 0
            if hn:
                sz = ndimage.sum(np.ones_like(hl), hl, range(1, hn + 1))
                big = int((sz * CELL ** 2 / 1e6 > 1.0).sum())
            isl.append(big)
        topo.append(dict(model=nm, median_components=float(np.median(ncomp)),
                         median_islands_gt1km2=float(np.median(isl)),
                         max_islands=int(np.max(isl)) if isl else 0))
        print(f"  {nm:<24} components {np.median(ncomp):8,.0f} | "
              f"islands >1 km2 retained {np.median(isl):4.0f}")
    pd.DataFrame(topo).to_csv(CFG.TABLES / "hist25b_gate7_topology_qa.csv",
                              index=False)

    # ------------------------------------------------ L_new
    print("\n" + "=" * 78)
    print("UPPER-MARGIN SUPPORT — the quantity that matters for the DEM")
    print("=" * 78)
    P = pd.read_parquet(PRIMARY)
    ptree = cKDTree(np.c_[P.x.to_numpy(), P.y.to_numpy()])
    try:
        CM = gpd.read_file(CONTOURS_CMAP, layer="cmap2020_contours")
        cpts = np.vstack([np.asarray(g.coords) for g in CM.geometry
                          if g.geom_type == "LineString"])
        ctree = cKDTree(cpts)
    except Exception:
        ctree = None
    best = agg.model.iloc[0]

    def connectivity(mask, obs):
        """Split classified water into components that may inherit the
        reservoir WSE and those that may not.

        A disconnected oxbow, pond or floodplain pool can be classified
        perfectly as WATER and still have no right to z_bed = H_reservoir.
        Only the body hydraulically continuous with the reservoir at that
        epoch may become a target-stage elevation constraint. Components
        touching NO_DATA are AMBIGUOUS, because their continuation is simply
        unobserved."""
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
        if n == 0:
            return mask & False, mask & False, mask & False
        core_ids = set(np.unique(lab[A_WATER & mask])) - {0}
        nodata_touch = ndimage.binary_dilation(~obs, np.ones((3, 3), bool))
        amb_ids = set(np.unique(lab[nodata_touch & mask])) - {0} - core_ids
        keep = np.zeros(n + 1, bool); keep[list(core_ids)] = True
        amb = np.zeros(n + 1, bool); amb[list(amb_ids)] = True
        conn = keep[lab]
        ambg = amb[lab]
        disc = mask & ~conn & ~ambg
        return conn, disc, ambg

    U_MAP_EVENT = None          # first eligible event drives the map
    map_cache = {}
    up_rows, conn_rows = [], []
    for eid in shorelines[best]:
        r = M[M.event_id == eid].iloc[0]
        if eid not in set(elig.event_id):
            continue
        # the target contour is carried in scientific_roles, e.g.
        # "H2_TARGET_STAGE_GEOMETRY_010"; the manifest has no nearest_target
        # column of its own
        roles = str(r.get("scientific_roles", ""))
        tgt = next((t for t in ("H1", "H2", "H3")
                    if f"{t}_TARGET_STAGE_GEOMETRY" in roles), "")
        zz = np.load(CACHE / f"{eid}.npz")
        obs_e = zz["cov"] & inside
        del zz
        cw = clean_water(shorelines[best][eid])
        conn, disc, ambg = connectivity(cw, obs_e)
        px = CELL ** 2 / 1e6
        conn_rows.append(dict(event_id=eid, contour=tgt,
                              connected_km2=float(conn.sum()) * px,
                              disconnected_km2=float(disc.sum()) * px,
                              ambiguous_km2=float(ambg.sum()) * px,
                              n_disconnected=int(ndimage.label(
                                  disc, structure=np.ones((3, 3), int))[1])))
        # ONLY the reservoir-connected body may become an elevation constraint
        sh = shoreline_px(conn)
        ys, xs = np.where(sh)
        if len(xs) == 0:
            continue
        X = G["x0"] + (xs + 0.5) * CELL
        Y = G["y1"] - (ys + 0.5) * CELL
        dp, _ = ptree.query(np.c_[X, Y], k=1)
        dc = (ctree.query(np.c_[X, Y], k=1)[0] if ctree is not None
              else np.full(len(X), 1e9))
        L_elig = len(X) * CELL / 1000
        row = dict(event_id=eid, contour=tgt, WSE_m=float(r.WSE_nominal),
                   WSE_unc_m=float(r.WSE_uncertainty_total_nominal),
                   eligible_shoreline_km=L_elig,
                   median_dist_sounding_m=float(np.median(dp)))
        for rad in REDUNDANCY_RADII_M:
            i = int(rad)
            row[f"Lnew_sound_{i}m"] = float((dp > rad).sum()) * CELL / 1000
            row[f"Lnew_cmap_{i}m"] = float((dc > rad).sum()) * CELL / 1000
            both = float(((dp > rad) & (dc > rad)).sum()) * CELL / 1000
            row[f"Lnew_both_{i}m"] = both
            row[f"fnew_both_{i}m"] = both / L_elig if L_elig else np.nan
        up_rows.append(row)
        if eid == U_MAP_EVENT or U_MAP_EVENT is None:
            map_cache["eid"] = eid
            map_cache["conn"] = conn.copy()
            map_cache["disc"] = disc.copy()
            map_cache["X"], map_cache["Y"] = X, Y
            map_cache["dp"], map_cache["dc"] = dp, dc
    U = pd.DataFrame(up_rows)
    CN = pd.DataFrame(conn_rows)
    U.to_csv(CFG.TABLES / "hist25b_gate7_upper_margin_support.csv", index=False)
    CN.to_csv(CFG.TABLES / "hist25b_gate7_connectivity.csv", index=False)
    if not len(U):
        print("  no eligible target-stage event produced a shoreline")
        return
    print(f"  model: {best}; eligible target-stage events: {len(U)}")
    print("\n  HYDRAULIC CONNECTIVITY (only the reservoir-connected body may")
    print("  inherit the reservoir WSE):")
    print(f"    connected    median {CN.connected_km2.median():8,.0f} km2")
    print(f"    disconnected median {CN.disconnected_km2.median():8,.0f} km2 "
          f"-- classified WATER, but no right to z_bed = H_reservoir")
    print(f"    ambiguous    median {CN.ambiguous_km2.median():8,.0f} km2 "
          f"-- touches NO_DATA, continuation unobserved")
    print(f"\n  eligible connected shoreline: median "
          f"{U.eligible_shoreline_km.median():,.0f} km per event")
    print("\n  REDUNDANCY-DISTANCE SENSITIVITY (500 m is primary; the others")
    print("  are sensitivity, and the 2 km variogram range is NOT used as a")
    print("  definition of source independence)")
    hdr = (f"    {'radius':>8} | {'vs soundings':>22} | {'vs CMAP':>22} | "
           f"{'vs BOTH':>22} | {'f_new':>14}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for rad in REDUNDANCY_RADII_M:
        i = int(rad)
        s_, c_, b_ = (U[f"Lnew_sound_{i}m"], U[f"Lnew_cmap_{i}m"],
                      U[f"Lnew_both_{i}m"])
        f_ = U[f"fnew_both_{i}m"]
        star = " *" if rad == UNSUPPORTED_M else "  "
        print(f"    {i:>6}m{star}| {s_.median():7,.0f} "
              f"[{s_.quantile(.1):5,.0f}-{s_.quantile(.9):5,.0f}] | "
              f"{c_.median():7,.0f} [{c_.quantile(.1):5,.0f}-{c_.quantile(.9):5,.0f}] | "
              f"{b_.median():7,.0f} [{b_.quantile(.1):5,.0f}-{b_.quantile(.9):5,.0f}] | "
              f"{100*f_.median():6.1f}% [{100*f_.quantile(.1):4.0f}-"
              f"{100*f_.quantile(.9):4.0f}]")
    b500 = U[f"Lnew_both_{int(UNSUPPORTED_M)}m"]
    b2000 = U["Lnew_both_2000m"]
    ratio = b2000.median() / max(b500.median(), 1e-9)
    print(f"\n  robustness: L_new at 2000 m is {100*ratio:.0f}% of its value "
          f"at 500 m")
    if ratio > 0.5:
        print("  -> the conclusion survives the strictest radius: SAR adds "
              "upper-margin constraint that neither source supports.")
    else:
        print("  -> novelty collapses with radius: SAR mainly DENSIFIES "
              "geometry between existing support, rather than reaching "
              "unsupported ground. State it that way.")

    _figures(agg, E, U, shorelines, best, G, A_WATER, A_LAND, P)
    _maps(map_cache, G, P, ctree, best, CN)
    print("\nSTOP before modifying the hist24 DEM.")


def _maps(mc, G, P, ctree, best, CN):
    """Maps: what the classifier produced, what connectivity removed, and
    where the shoreline is genuinely unsupported."""
    if not mc:
        print("  (no map event cached)")
        return
    ext = [G["x0"] / 1000, G["x1"] / 1000, G["y0"] / 1000, G["y1"] / 1000]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2), sharex=True, sharey=True)
    a = ax[0]
    canvas = np.zeros(mc["conn"].shape, np.uint8)
    canvas[G["inside"]] = 1
    canvas[mc["disc"]] = 2
    canvas[mc["conn"]] = 3
    a.imshow(canvas, extent=ext, origin="upper", interpolation="nearest",
             cmap=matplotlib.colors.ListedColormap(
                 ["#ffffff", "#eef1f3", AMBER, BLUE]), vmin=0, vmax=3)
    a.set_title(f"a · {mc['eid']}  —  {best}\n"
                "blue = reservoir-connected, amber = disconnected water "
                "(no right to reservoir WSE)", fontsize=9.6, loc="left")
    a = ax[1]
    a.imshow(np.where(G["inside"], 0.25, 0), extent=ext, origin="upper",
             cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    a.plot(P.x / 1000, P.y / 1000, ".", ms=0.4, color=GREY, alpha=0.6)
    new = (mc["dp"] > UNSUPPORTED_M) & (mc["dc"] > UNSUPPORTED_M)
    a.plot(mc["X"][~new] / 1000, mc["Y"][~new] / 1000, ".", ms=0.35,
           color="#9fb4c2")
    a.plot(mc["X"][new] / 1000, mc["Y"][new] / 1000, ".", ms=0.7, color=RED)
    a.set_title(f"b · red = shoreline >{UNSUPPORTED_M:.0f} m from BOTH "
                f"soundings and CMAP\ngrey dots = the 7,514 soundings",
                fontsize=9.6, loc="left")
    a = ax[2]
    sc = a.scatter(mc["X"] / 1000, mc["Y"] / 1000,
                   c=np.minimum(mc["dp"], 3000), s=0.5, cmap="magma_r")
    fig.colorbar(sc, ax=a, label="distance to nearest sounding (m, capped 3 km)")
    a.set_title("c · how far each shoreline pixel sits from real bathymetry",
                fontsize=9.6, loc="left")
    for a in ax:
        a.set_xlabel("easting (km)")
    ax[0].set_ylabel("northing (km)")
    fig.suptitle("hist25b Gate 7 · connectivity filter and upper-margin "
                 "support", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "hist25b_gate7_maps.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def _figures(agg, E, U, shorelines, best, G, A_WATER, A_LAND, P):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    a = ax[0]
    order = agg.model.tolist()
    dat = [E[E.model == m].f1.dropna() for m in order]
    a.boxplot(dat, tick_labels=[m.replace("_", "\n") for m in order],
              showfliers=False)
    a.set_ylabel("F1 on held-out anchor blocks")
    a.set_title("a · per-event F1 — spread matters as much as the median",
                fontsize=10.2, loc="left")
    a.grid(alpha=0.25, axis="y"); a.tick_params(labelsize=7.5)
    a = ax[1]
    for m, c in zip(order, (BLUE, GREEN, AMBER, PURPLE, RED)):
        s = E[E.model == m]
        a.scatter(s.coverage, s.f1, s=22, color=c, label=m, alpha=0.8)
    a.set_xlabel("spatial coverage fraction"); a.set_ylabel("F1")
    a.legend(fontsize=7); a.grid(alpha=0.25)
    a.set_title("b · coverage vs classification quality\n"
                "two independent axes; low coverage is not low quality",
                fontsize=10.2, loc="left")
    a = ax[2]
    if len(U):
        a.bar(range(len(U)), U[f'Lnew_both_{int(UNSUPPORTED_M)}m'], color=GREEN)
        a.set_xticks(range(len(U)))
        a.set_xticklabels(U.event_id.str[:10], rotation=90, fontsize=6.5)
        a.set_ylabel(f"km beyond soundings AND CMAP ({UNSUPPORTED_M:.0f} m)")
    a.set_title("c · L_new, the quantity that matters for the DEM",
                fontsize=10.2, loc="left")
    a.grid(alpha=0.25, axis="y")
    fig.suptitle("hist25b Gate 7 · anchored SAR classification", y=1.02,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate7_models.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGDIR/'hist25b_gate7_models.png'}")


if __name__ == "__main__":
    main()
