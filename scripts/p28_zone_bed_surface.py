#!/usr/bin/env python
"""P28 -- PRE_BREACH bed-elevation DEM for ZONE_2 / ZONE_3 / ZONE_4 with
mask-derived shoreline constraints as SOFT data, and honest cross-validation.

What changes against p19, and why
---------------------------------
p19 constrains the surface with p18's pseudo-points: an undated WorldCover-2021
outline carrying a mean-annual level, stacked with the soundings as HARD
training data with no error variance (audit F-18), and validated on a CV set
that is 62-77 % those same pseudo-points (F-01). ZONE_1's DEM (hist24) instead
feeds dated Sentinel-2 shorelines at gauge-known levels to kriging as SOFT
constraints with a per-point sigma budget, through `ok_soft`.

This script brings zones 2/3/4 to that standard:

  * constraints  = p27's dated S2 shoreline vertices (x, y, H, sigma) -- the
                   hist23/hist26 chain on the p25 zone stacks;
  * estimator    = `swot_dnipro.kriging.ok_soft`, sign-corrected (F-04), with
                   the |w|max guard and a sign check that runs on import;
  * validation   = `p19.summarise_resid` stratified by observation type;
                   `best_method` is selected on RMSE_snd_m (real depth) only;
  * ZONE_3       = built for the first time (634 soundings; no DEM existed).

Three arms, so the effect of the constraint source is measurable:
    SND_ONLY          soundings alone                (IDW OK RBF LINEAR)
    SND+SHORE_S2      + p27 vertices as SOFT (OK_SOFT) and, for comparison,
                      as HARD (RBF, OK, LINEAR)      <- PRIMARY arm
    SND+PSEUDO_P18    + p18 pseudo-points as HARD    (p19's design, zones 2/4 only)

The variogram is fitted on the SOUNDINGS only: the constraints are not bed
measurements and must not shape the spatial model of the bed.

The local-envelope clip (p19) is retained for every hard-data method and its
clipped fraction and 8-NN composition are reported per method, because with
soft constraints the envelope is now bounded by observed waterlines instead
of a synthetic line -- that is the point, and it must be visible.

What it still cannot claim: a POST_BREACH bed. There is no post-breach depth
measurement in the project (p19 docstring), and that has not changed.

Outputs
-------
BULK_ROOT/data_swot/processed/bathymetry/zone_bed_surface_<ZONE>_PRE_BREACH_<cell>m.npz
outputs/tables/p28_interpolator_cv_<ZONE>.csv      (arm x method x scheme, stratified)
outputs/tables/p28_surface_summary_<ZONE>.csv
outputs/tables/p28_constraint_dominance_<ZONE>.csv (share of constraints among K-NN, by shore distance)
outputs/rasters/zone<N>/zone<N>_bed_<method>_<arm>_PRE_BREACH_<cell>m.tif   (best surface)
outputs/rasters/zone<N>/zone<N>_bed_constraint_share_PRE_BREACH_<cell>m.tif
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import kriging as KR
from swot_dnipro import spatial_domains as SD
import p19_zone24_bed_surface as P19          # load_soundings, load_shore_points, METHODS,
                                              # clip_to_local_envelope, block_folds,
                                              # summarise_resid, N_FOLDS, emodnet_covariate

ZONES = {"ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
REGIME = "PRE_BREACH"
HARD_METHODS = ("IDW", "OK", "RBF", "LINEAR")
ARMS = ("SND_ONLY", "SND+SHORE_S2", "SND+PSEUDO_P18")
PRIMARY_ARM = "SND+SHORE_S2"
CONSTRAINT_GPKG = ROOT / "data" / "processed" / "bathymetry"


def load_s2_constraints(zone: str):
    gp = CONSTRAINT_GPKG / f"zone_shore_contours_{zone}_{REGIME}.gpkg"
    if not gp.exists():
        return None
    v = gpd.read_file(gp, layer="vertices")
    return dict(xy=np.c_[v.x.values, v.y.values], z=v.H_evrf2019_m.values.astype(float),
                s=v.sigma_m.values.astype(float), n=len(v), groups=v.group.nunique(),
                sigma_median=float(np.median(v.sigma_m)))


def fit_variogram(xy, z, seed):
    import skgstat as skg
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(xy), min(1500, len(xy)), replace=False)
    V = skg.Variogram(xy[sub], z[sub], model="spherical", n_lags=20, normalize=False)
    p = V.parameters
    return (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)


def thin(cons: dict | None, every: int) -> dict | None:
    """Every k-th vertex: densify() emits vertices in ring order at 150 m, so
    every=3 is ~450 m, every=7 ~1 km. The first ZONE_2 run showed 13,835
    shoreline vertices against 527 soundings drowning every K=32
    neighbourhood; spacing is the first knob to turn before anything cleverer."""
    if cons is None or every <= 1:
        return cons
    sl = slice(None, None, every)
    return dict(xy=cons["xy"][sl], z=cons["z"][sl], s=cons["s"][sl],
                n=len(cons["z"][sl]), groups=cons.get("groups"), sigma_median=cons.get("sigma_median"))


def cv_arm(xy_snd, z_snd, cons, folds_snd, vg, arm: str, hard: bool, soft: bool,
           dist_shore_snd=None):
    """Blocked CV. Folds are defined on SOUNDINGS; constraints are always training.

    The held-out unit is a block of soundings -- the only real depth data -- so
    every residual scored here is a residual on a measurement. Constraint
    points never appear in the test set (they are not measurements), which is
    the reason the stratified split of p19 is unnecessary here: RMSE_m IS
    RMSE_snd_m. Both names are written so the tables line up with p19's."""
    rows = []
    n = len(xy_snd)
    methods = list(HARD_METHODS) if hard else []
    if soft:
        methods.append("OK_SOFT")
    for m in methods:
        resid = np.full(n, np.nan)
        fb_total = 0
        for f in range(P19.N_FOLDS):
            te = folds_snd == f
            if te.sum() == 0 or (~te).sum() < 10:
                continue
            xy_tr, z_tr = xy_snd[~te], z_snd[~te]
            if cons is not None:
                xy_tr = np.vstack([xy_tr, cons["xy"]]); z_tr = np.concatenate([z_tr, cons["z"]])
            if m == "OK_SOFT":
                ev = np.concatenate([np.zeros((~te).sum()), cons["s"] ** 2])
                pred, diag = KR.ok_soft(xy_tr, z_tr, ev, xy_snd[te], vg, return_diag=True)
                fb_total += diag["n_fallback"]
            else:
                pred = P19.METHODS[m](xy_tr, z_tr, xy_snd[te], vg)
                pred = P19.clip_to_local_envelope(xy_tr, z_tr, xy_snd[te], pred)
            resid[te] = pred - z_snd[te]
        st = P19.summarise_resid(resid, None)
        st.update({f"{k}_snd_m": st[f"{k}_m"] for k in ("RMSE", "MAE", "bias", "NMAD")})
        st["n_snd"] = st["n"]
        # the question a soundings-only CV cannot answer alone: do constraints
        # help NEAR the shore (where the soundings are sparse) even if they hurt
        # far from it? Stratify the sounding residuals by distance to shoreline.
        if dist_shore_snd is not None:
            for lab, lo, hi in (("near", 0, 500), ("mid", 500, 1500), ("far", 1500, np.inf)):
                sel = (dist_shore_snd >= lo) & (dist_shore_snd < hi)
                r = resid[sel]; r = r[np.isfinite(r)]
                st[f"n_{lab}"] = int(len(r))
                st[f"RMSE_{lab}_m"] = float(np.sqrt((r ** 2).mean())) if len(r) else np.nan
                st[f"bias_{lab}_m"] = float(r.mean()) if len(r) else np.nan
        rows.append(dict(arm=arm, method=m, n_constraints=0 if cons is None else cons["n"],
                         soft=(m == "OK_SOFT"), idw_fallback_targets=fb_total, **st))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=list(ZONES))
    ap.add_argument("--cell", type=float, default=250.0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--thin", default="1,3,7",
                    help="vertex thinning factors for the S2 arm (1=150 m, 3~450 m, 7~1 km)")
    a = ap.parse_args()
    zone, CELL = a.zone, a.cell
    # 250 m is the statistical canon and keeps the untagged table names that p29
    # reads; any finer grid tags its tables so it cannot overwrite them
    TAG = "" if abs(CELL - 250.0) < 1 else f"_{CELL:.0f}m"
    arms = [x for x in a.arms.split(",") if x in ARMS]
    thins = [int(t) for t in a.thin.split(",") if t.strip()]
    n_zone = ZONES[zone]
    t0 = time.time()
    print("=" * 78)
    print(f"P28 -- {zone} {REGIME} bed DEM with soft S2 shoreline constraints")
    print("=" * 78)

    snd = P19.load_soundings(zone)
    xy_snd = np.c_[snd.x.values, snd.y.values]
    z_snd = snd.H_bed_evrf2019_m.values.astype(float)
    print(f"  soundings {len(snd)}  bed {z_snd.min():.2f}..{z_snd.max():.2f} m")
    if len(snd) < 30:
        raise SystemExit("fewer than 30 soundings -- refuse to build a surface")

    s2c = load_s2_constraints(zone)
    if s2c is None:
        print("  no p27 S2 constraints for this zone -- SND+SHORE_S2 arm skipped")
        arms = [x for x in arms if x != "SND+SHORE_S2"]
    else:
        print(f"  S2 shoreline vertices {s2c['n']:,} in {s2c['groups']} group(s), "
              f"H {s2c['z'].min():.2f}..{s2c['z'].max():.2f} m, sigma median {s2c['sigma_median']:.3f} m")
    p18 = None
    try:
        sh = P19.load_shore_points(zone)
        if len(sh):
            p18 = dict(xy=np.c_[sh.x.values, sh.y.values], z=sh.elevation_evrf2019_m.values.astype(float),
                       s=np.full(len(sh), np.nan), n=len(sh))
            print(f"  p18 pseudo-points {len(sh):,} (comparison arm, HARD as in p19)")
    except Exception:
        pass
    if p18 is None:
        arms = [x for x in arms if x != "SND+PSEUDO_P18"]

    vg = fit_variogram(xy_snd, z_snd, CFG.SEED)
    print(f"  variogram on soundings: range {vg[0]/1000:.2f} km, sill {vg[1]:.2f}, nugget {vg[2]:.2f}")
    nn = cKDTree(xy_snd).query(xy_snd, k=2)[0][:, 1]
    block_km = max(0.5, round(np.median(nn) / 200.0) * 0.2)
    folds = P19.block_folds(xy_snd, block_km, CFG.SEED)
    print(f"  blocked folds on soundings: {block_km:.2f} km blocks, {P19.N_FOLDS} folds")

    # distance from each sounding to the nearest S2 shoreline vertex (for the
    # near / mid / far split of the CV residuals)
    dshore = (cKDTree(s2c["xy"]).query(xy_snd, k=1)[0] if s2c is not None else None)
    if dshore is not None:
        print(f"  soundings within 500 m of the S2 shoreline: {int((dshore<500).sum())}, "
              f"500-1500 m: {int(((dshore>=500)&(dshore<1500)).sum())}, >1500 m: {int((dshore>=1500).sum())}")

    rows = []
    for arm in arms:
        if arm == "SND+SHORE_S2":
            for k in thins:
                label = arm if k == 1 else f"{arm}_thin{k}"
                rows += cv_arm(xy_snd, z_snd, thin(s2c, k), folds, vg, label, hard=True, soft=True,
                               dist_shore_snd=dshore)
                print(f"  CV {label}: done ({time.time()-t0:.0f}s)")
        else:
            cons = {"SND_ONLY": None, "SND+PSEUDO_P18": p18}[arm]
            rows += cv_arm(xy_snd, z_snd, cons, folds, vg, arm, hard=True, soft=False,
                           dist_shore_snd=dshore)
            print(f"  CV {arm}: done ({time.time()-t0:.0f}s)")
    cv = pd.DataFrame(rows)
    cv["scheme"] = "blocked_soundings"
    cv_path = CFG.TABLES / f"p28_interpolator_cv_{zone}{TAG}.csv"
    cv.to_csv(cv_path, index=False)
    cols = ["arm", "method", "n_constraints", "RMSE_snd_m", "bias_snd_m"]
    cols += [c for c in ("RMSE_near_m", "RMSE_mid_m", "RMSE_far_m", "n_near") if c in cv]
    print("\n  " + cv[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))

    # selection: best RMSE at the soundings across ALL arms. If a constraint arm
    # cannot beat soundings-only on the only real depth data, it does not get
    # to be the canonical surface -- however desirable a constrained shoreline
    # is in principle. The near/far split above is what says whether the
    # constraints earn their place near the shore.
    ibest = cv.RMSE_snd_m.idxmin()
    best_m, best_arm = cv.loc[ibest, "method"], cv.loc[ibest, "arm"]
    prim = cv[cv.arm == best_arm].set_index("method")
    thin_k = int(best_arm.split("thin")[1]) if "thin" in best_arm else 1
    if s2c is not None and best_arm.startswith("SND+SHORE_S2"):
        s2c = thin(s2c, thin_k)
    print(f"\n  selected: {best_arm} / {best_m}  RMSE_snd {prim.loc[best_m,'RMSE_snd_m']:.3f} m  "
          f"bias {prim.loc[best_m,'bias_snd_m']:+.3f} m")
    ref = cv[(cv.arm == "SND_ONLY")].set_index("method")
    if best_m in ref.index:
        print(f"  vs SND_ONLY same method: RMSE_snd {ref.loc[best_m,'RMSE_snd_m']:.3f} m")
    if "SND+PSEUDO_P18" in set(cv.arm):
        p18r = cv[cv.arm == "SND+PSEUDO_P18"].set_index("method")
        if best_m in p18r.index:
            print(f"  vs p19 design (hard p18 pseudo-points), same method: RMSE_snd "
                  f"{p18r.loc[best_m,'RMSE_snd_m']:.3f} m")

    # ---- production surface --------------------------------------------------
    fp = SD.load_utm(zone)
    gr = SD.build_grid(fp, CELL, what=f"{zone} bed grid")
    gx, gy = gr["gx"], gr["gy"]
    GX, GY = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(fp, GX.ravel(), GY.ravel())
    tgt = np.c_[GX.ravel()[inside], GY.ravel()[inside]]
    print(f"\n  grid {len(gx)}x{len(gy)} at {CELL:.0f} m, {inside.sum():,} cells inside")
    cons = (None if best_arm == "SND_ONLY" else p18 if best_arm == "SND+PSEUDO_P18" else s2c)
    xy_tr, z_tr = xy_snd, z_snd
    if cons is not None:
        xy_tr = np.vstack([xy_tr, cons["xy"]]); z_tr = np.concatenate([z_tr, cons["z"]])
    is_con = np.concatenate([np.zeros(len(xy_snd), bool), np.ones(len(xy_tr) - len(xy_snd), bool)])
    if best_m == "OK_SOFT":
        ev = np.concatenate([np.zeros(len(xy_snd)), cons["s"] ** 2])
        surf, diag = KR.ok_soft(xy_tr, z_tr, ev, tgt, vg, return_diag=True)
        clip_frac = 0.0
        print(f"  OK_SOFT: {diag['n_fallback']} of {diag['n_targets']:,} targets fell back to IDW")
    else:
        raw = P19.METHODS[best_m](xy_tr, z_tr, tgt, vg)
        surf, d = P19.clip_to_local_envelope(xy_tr, z_tr, tgt, raw, is_shore_tr=is_con, return_diag=True)
        clip_frac = float(np.mean(d["clip_amount_m"] > 1e-6))
        print(f"  {best_m}: {100*clip_frac:.1f} % of cells clipped by the local envelope")
    # constraint share among K nearest training points, by distance to shoreline
    K = 32
    _, idx = cKDTree(xy_tr).query(tgt, k=min(K, len(xy_tr)))
    con_share = is_con[idx].mean(1)
    dist_shore = (cKDTree(cons["xy"]).query(tgt, k=1)[0] if cons is not None
                  else np.full(len(tgt), np.nan))
    dom = []
    for lo, hi in ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 1e9)):
        s = (dist_shore >= lo) & (dist_shore < hi)
        if s.any():
            dom.append(dict(zone=zone, arm=best_arm, method=best_m, shore_dist_m=f"{lo}-{hi if hi<1e9 else 'inf'}",
                            n_cells=int(s.sum()), constraint_share_median=float(np.median(con_share[s])),
                            constraint_share_p90=float(np.percentile(con_share[s], 90))))
    pd.DataFrame(dom).to_csv(CFG.TABLES / f"p28_constraint_dominance_{zone}{TAG}.csv", index=False)

    # ---- product support -----------------------------------------------------
    # The interpolator returns a value for every grid cell inside the zone
    # polygon, land included; the first ZONE_2 render showed the steppe and the
    # city at -1..-7 m. The canonical raster is masked to (a) the observed
    # PRE_BREACH water body -- the p27 shoreline polygon, which is exactly the
    # role a shoreline is good for: the product's boundary, not a bed sample --
    # and (b) cells within SUPPORT_KM of a sounding, beyond which the surface is
    # extrapolation. The unmasked surface is kept as a diagnostic raster.
    SUPPORT_KM = min(vg[0] / 1000.0, 3.0)
    dist_snd = cKDTree(xy_snd).query(tgt, k=1)[0]
    in_water = np.ones(len(tgt), bool)
    gp = CONSTRAINT_GPKG / f"zone_shore_contours_{zone}_{REGIME}.gpkg"
    if gp.exists():
        wpoly = gpd.read_file(gp, layer="contours").geometry.union_all()
        in_water = shapely.contains_xy(wpoly, tgt[:, 0], tgt[:, 1])
    support = in_water & (dist_snd <= SUPPORT_KM * 1000.0)
    surf_canon = np.where(support, surf, np.nan)
    print(f"  support: inside observed water {in_water.mean()*100:.1f} % of cells, "
          f"within {SUPPORT_KM:.1f} km of a sounding {(dist_snd<=SUPPORT_KM*1000).mean()*100:.1f} %, "
          f"canonical cells {support.sum():,} ({support.mean()*100:.1f} %)")

    out_dir = CFG.BULK_ROOT / "data_swot" / "processed" / "bathymetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"zone_bed_surface_{zone}_{REGIME}_{CELL:.0f}m.npz"
    np.savez_compressed(npz, gx=gx, gy=gy, inside=inside, tgt=tgt, surf=surf,
                        surf_canonical=surf_canon, support=support, dist_to_sounding_m=dist_snd,
                        support_km=SUPPORT_KM,
                        best_method=best_m, best_arm=best_arm, constraint_share=con_share,
                        dist_to_shore_m=dist_shore, variogram=np.array(vg),
                        n_soundings=len(snd), n_constraints=0 if cons is None else cons["n"])
    # rasters
    rdir = ROOT / "outputs" / "rasters" / f"zone{n_zone}"
    rdir.mkdir(parents=True, exist_ok=True)
    ny, nx = len(gy), len(gx)
    def to_grid(v, fill=np.nan):
        g = np.full(ny * nx, fill, "f4"); g[inside] = v; return g.reshape(ny, nx)[::-1, :]
    tr = from_origin(float(gx[0] - CELL / 2), float(gy[-1] + CELL / 2), CELL, CELL)
    prof = dict(driver="GTiff", height=ny, width=nx, count=1, dtype="float32", crs=CFG.CRS_METRIC,
                transform=tr, compress="deflate", nodata=-9999.0)
    tag = dict(zone=zone, regime=REGIME, arm=best_arm, method=best_m, cell_m=str(CELL),
               n_soundings=str(len(snd)), n_constraints=str(0 if cons is None else cons["n"]),
               constraints="p27 dated S2 shoreline vertices, SOFT" if best_arm == "SND+SHORE_S2" else best_arm,
               rmse_soundings_m=f"{prim.loc[best_m,'RMSE_snd_m']:.3f}", producer="p28_zone_bed_surface.py")
    for name, arr, extra in (
            (f"zone{n_zone}_bed_canonical_{REGIME}_{CELL:.0f}m.tif", to_grid(surf_canon, np.nan),
             dict(role=("CANONICAL: " if not TAG else "DISPLAY grid, 250 m is the statistical canon: ")
                       + "masked to the p27 observed water body and to "
                       f"<= {SUPPORT_KM:.1f} km from a sounding")),
            (f"zone{n_zone}_bed_{best_m}_{best_arm.replace('+','_')}_{REGIME}_{CELL:.0f}m_unmasked.tif",
             to_grid(surf, np.nan), dict(role="DIAGNOSTIC: unmasked interpolator output over the whole zone; NOT a bathymetry product on land")),
            (f"zone{n_zone}_bed_constraint_share_{REGIME}_{CELL:.0f}m.tif", to_grid(con_share, np.nan),
             dict(role="share of constraint points among the 32 nearest training points")),
            (f"zone{n_zone}_bed_dist_to_sounding_{REGIME}_{CELL:.0f}m.tif", to_grid(dist_snd, np.nan),
             dict(role="metres to the nearest sounding (support)"))):
        with rasterio.open(rdir / name, "w", **prof) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999.0).astype("f4"), 1); dst.update_tags(**tag, **extra)
    summ = pd.DataFrame([dict(zone=zone, regime=REGIME, cell_m=CELL, n_soundings=len(snd),
                              n_s2_constraints=0 if s2c is None else s2c["n"],
                              s2_sigma_median_m=np.nan if s2c is None else s2c["sigma_median"],
                              n_p18_pseudopoints=0 if p18 is None else p18["n"],
                              variogram_range_km=vg[0] / 1000, variogram_sill=vg[1], variogram_nugget=vg[2],
                              block_km=block_km, arms="|".join(arms), best_arm=best_arm, best_method=best_m,
                              best_rmse_soundings_m=float(prim.loc[best_m, "RMSE_snd_m"]),
                              best_bias_soundings_m=float(prim.loc[best_m, "bias_snd_m"]),
                              best_nmad_soundings_m=float(prim.loc[best_m, "NMAD_snd_m"]),
                              clip_fraction=clip_frac, grid_cells_inside=int(inside.sum()),
                              support_km=SUPPORT_KM, canonical_cells=int(support.sum()),
                              canonical_km2=float(support.sum() * CELL ** 2 / 1e6))])
    summ.to_csv(CFG.TABLES / f"p28_surface_summary_{zone}{TAG}.csv", index=False)
    print(f"\n-> {npz}\n-> {cv_path}\n-> {rdir}\n  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
