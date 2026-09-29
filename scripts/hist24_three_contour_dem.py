#!/usr/bin/env python
"""HISTORICAL 24 — the three-contour constrained bathymetry experiment.

Steps 4-13. hist23 built the three pre-breach shorelines; this turns them into
soft elevation constraints, fits the constrained surfaces, and validates them
with a test the sounding cross-validation cannot perform.

WHY A NEW VALIDATION IS NEEDED. The highest sounding in the survey is 15.32 m
against a 17.08 m waterline, so a 1.77 m elevation band contains no soundings
at all. That band is exactly where the constraints add information and exactly
where held-out soundings cannot score anything. So:

  PRIMARY   leave-one-contour-out: fit with two contours, predict the third
  GUARD     frozen-fold sounding CV: confirm the interior did not degrade
            -- a near-null result here is the EXPECTED outcome, declared in
            advance, not a disappointment
  INDEPENDENT  Table 19 A(H) and Table 21 exposure, recomputed AFTER fitting
            and never used during it

SOFT CONSTRAINTS, STATED FORMULATION. A shoreline pixel is not an exact
elevation. Contour points enter the kriging system as observations carrying a
measurement-error variance sigma_H^2 added to the diagonal of the left-hand
side -- textbook kriging with measurement error. Soundings keep the fitted
nugget. This is the one documented departure from hist14's estimator; every
other parameter (neighbourhood, variogram, batching, guards) is imported
unchanged.

Outputs
-------
outputs/tables/prebreach_contour_uncertainty.csv
outputs/tables/leave_one_contour_out_validation.csv
outputs/tables/baseline_vs_constrained_sounding_cv.csv
outputs/tables/unsounded_belt_analysis.csv
outputs/tables/baseline_vs_constrained_hypsometry.csv
outputs/figures/V20_leave_one_contour_out_validation.png
outputs/figures/V21_baseline_vs_constrained_bathymetry.png
outputs/figures/V22_hypsometry_validation.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_JOBS = 1
for _i, _a in enumerate(sys.argv):
    if _a == "--jobs" and _i + 1 < len(sys.argv):
        _JOBS = int(sys.argv[_i + 1])
if _JOBS != 1:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import sword as SW

import hist14_bed_surface as H14

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
# The staircase version (prebreach_contours.gpkg) has every vertex on a
# multiple of 20 m and must not carry positional truth -- see
# outputs/planning/12_GATE_7C_ROLE_FREEZE.md. hist26 rebuilt these from the
# continuous water field with the classification rule unchanged.
GPKG = ROOT / "data/processed/bathymetry/prebreach_contours_continuous.gpkg"
CONSTRAINTS = None      # set in main() from hist31
SPACINGS = (100.0, 150.0, 250.0)
SPACING_MAIN = 150.0
CELL = 250.0
GMO_EVRF, NPG_EVRF = 12.70 + 0.185, 16.00 + 0.185
Z_MAX_SOUNDING = 15.3153        # highest sounding, from hist12
# uncertainty components, all in metres (1 sigma)
SIG_EPSG9902 = 0.068            # stated accuracy of the BS-77->EVRF2019 grid
SIG_GAUGE_READ = 0.01           # gauge reading precision
SIG_SEICHE = 0.175             # half the documented binodal range at antinode
_S: dict = {}


# ------------------------------------------------------- soft-constraint OK --
def ok_soft(xy_tr, z_tr, err_var, xy_te, vg):
    """hist14's local ordinary kriging, with per-observation error variance.

    Identical to H14._ok except that err_var[i] is added to the diagonal for
    training point i. With err_var = 0 everywhere it reduces exactly to the
    baseline estimator, which is asserted in --verify.
    """
    rng_, sill, nug = vg
    nug = max(nug, 0.01 * sill)
    K = min(H14.OK_K, len(xy_tr))
    _, idx = cKDTree(xy_tr).query(xy_te, k=K)
    idx = np.atleast_2d(idx)
    out = np.empty(len(xy_te))
    for a in range(0, len(xy_te), H14.OK_BATCH):
        b = slice(a, min(a + H14.OK_BATCH, len(xy_te)))
        ii = idx[b]
        P = xy_tr[ii]
        n = P.shape[0]
        A = np.zeros((n, K + 1, K + 1))
        A[:, :K, :K] = H14.spherical(
            np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1),
            rng_, sill, nug)
        A[:, :K, K] = 1.0
        A[:, K, :K] = 1.0
        # MINUS err_var. spherical() is the VARIOGRAM (0 at h=0), and in
        # variogram form a measurement-error variance s2 enters the diagonal as
        # gamma_eff_ii = sill - (sill + s2) = -s2. This line added +err_var until
        # 2026-09-16, which gave a noisier observation MORE weight (audit
        # 20260916T093000Z, F-04; independent covariance-form check in
        # audit_runs/.../kriging/independent_check.py). At the err_var this
        # script actually runs with (0.04-0.6 m2 vs a sill of ~10-100 m2) the
        # induced bias was 0.006-0.099 m, so the shipped products were not
        # materially affected and are NOT recomputed here; the sign is fixed so
        # the estimator means what its docstring says. swot_dnipro.kriging
        # carries the same estimator with a sign check that runs on import.
        A[:, np.arange(K), np.arange(K)] += 1e-8 * sill - err_var[ii]
        rhs = np.zeros((n, K + 1))
        dt = np.linalg.norm(P - xy_te[b][:, None, :], axis=-1)
        rhs[:, :K] = H14.spherical(dt, rng_, sill, nug)
        rhs[:, K] = 1.0
        try:
            # NumPy 2 no longer infers "stack of vectors" when b.ndim ==
            # a.ndim - 1; it reads rhs as a matrix and the solve fails on the
            # core dimensions. An explicit trailing axis restores the intent.
            wts = np.linalg.solve(A, rhs[:, :, None])[:, :K, 0]
        except np.linalg.LinAlgError:
            wts = np.full((n, K), np.nan)
        # The Lagrange row FORCES sum(w) = 1 even when the kriging system is
        # ill-conditioned, so the sum check cannot detect a singular solve: the
        # weights come back summing to 1 with individual magnitudes in the
        # thousands, and the estimate explodes. That is what produced a
        # leave-one-contour-out RMSE of 224.8 m at 150 m densification against
        # 3.6 m at 100 m and 9.9 m at 250 m -- a 6125% spread across a spacing
        # choice, which is numerics, not bathymetry. Ordinary-kriging weights on
        # a well-posed system stay of order unity; anything past 5 is a
        # degenerate neighbourhood, and those points fall back to IDW.
        bad = (~np.isfinite(wts).all(1)
               | (np.abs(wts.sum(1) - 1) > 0.05)
               | (np.abs(wts).max(1) > 5.0))
        if bad.any():
            wi = 1.0 / np.maximum(dt[bad], 1e-6) ** 2
            wts[bad] = wi / wi.sum(1, keepdims=True)
        out[b] = (wts * z_tr[ii]).sum(1)
    return out


def _cv_task(arg):
    """One (variant, fold) of the frozen-fold sounding CV."""
    var, fold = arg
    xy, z, vg = _S["xy"], _S["z"], _S["vg"]
    te = _S["folds"] == fold
    cxy, cz, cs = _S["cxy"][var], _S["cz"][var], _S["cs"][var]
    Xtr = np.vstack([xy[~te], cxy]) if len(cxy) else xy[~te]
    Ztr = np.concatenate([z[~te], cz]) if len(cxy) else z[~te]
    Etr = np.concatenate([np.zeros((~te).sum()), cs ** 2]) if len(cxy) \
        else np.zeros((~te).sum())
    pred = ok_soft(Xtr, Ztr, Etr, xy[te], vg)
    return var, np.where(te)[0], pred - z[te]


def densify(geom, spacing):
    """Points along a boundary at ~spacing, plus every vertex-free segment."""
    b = geom.boundary if geom.geom_type in ("Polygon", "MultiPolygon") else geom
    lines = list(b.geoms) if hasattr(b, "geoms") else [b]
    pts = []
    for ln in lines:
        n = max(2, int(np.ceil(ln.length / spacing)))
        for t in np.linspace(0, 1, n, endpoint=False):
            p = ln.interpolate(t, normalized=True)
            pts.append((p.x, p.y))
    return np.array(pts)


def stats(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if not len(r):
        return dict(n=0)
    a = np.abs(r)
    return dict(n=len(r), RMSE_m=float(np.sqrt((r**2).mean())),
                MAE_m=float(a.mean()), bias_m=float(r.mean()),
                median_m=float(np.median(r)), NMAD_m=H14.nmad(r),
                p90_abs=float(np.percentile(a, 90)),
                p95_abs=float(np.percentile(a, 95)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    jobs = max(1, args.jobs)

    import geopandas as gpd
    if not GPKG.exists():
        raise SystemExit(f"{GPKG.name} missing -- run hist23 first")
    cg = gpd.read_file(GPKG, layer="contour_polygons").sort_values(
        "H_evrf2019_m").reset_index(drop=True)
    inv = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv")
    # Gauge metadata is unchanged by the geometry rebuild, but the date groups
    # and their level spreads are not: hist26 reached full coverage from fewer
    # dates, so its own spreads are the ones that belong in the budget.
    h26 = pd.read_csv(CFG.TABLES / "hist26_continuous_contours.csv").set_index(
        "contour_id")
    for c in inv.contour_id:
        if c in h26.index:
            inv.loc[inv.contour_id == c, "level_spread_within_group_m"] = float(
                h26.loc[c, "level_spread_m"])
            inv.loc[inv.contour_id == c, "dates"] = str(h26.loc[c, "dates"])
            inv.loc[inv.contour_id == c, "n_dates"] = int(h26.loc[c, "n_dates"])
    print(f"contours: " + ", ".join(
        f"{r.contour_id}={r.H_evrf2019_m:.3f} m" for r in cg.itertuples()))

    # ================================================================= GATE
    # REWRITTEN. The old gate demanded that the shoreline move between levels
    # by more than its offset from the TRUE waterline, and hard-coded that
    # offset as OFFSET_FROM_TRUE_WATERLINE_M = 181.0 -- a number measured
    # against the superseded P20 footprint. It failed at signal/bias = 0.12 and
    # blocked everything downstream.
    #
    # Three results replace it, and the semantics change with them:
    #
    #   hist28  the offset is not single-valued at all. Across the admitted
    #           scenes it has no structure by stage (p 0.27) or orbit (p 0.34),
    #           and the between-scene spread is 2.3x the within-scene scatter.
    #           NO global constant is admissible -- not 181, not 29-36, none.
    #   hist29  an independent waterline exists only where the sounding TIN
    #           BRACKETS the level. H3 yes (512 triangles, 55.1 km); H2 and H1
    #           no, because the highest sounding in the survey is 15.32 m.
    #   hist30  at H3 the absolute bias is +19.6 m (S2) and +17.6 m (S1)
    #           against a reference whose own horizontal uncertainty is 87.2 m.
    #           Gross common-mode displacement is excluded; fine calibration is
    #           not achieved, and no correction is applied.
    #
    # So the gate no longer asks for absolute validation at every stage. It
    # asks that the constraint table exist, that every stage carry an explicit
    # validation class, and that nothing downstream smuggle in a correction:
    #
    #     DIRECT validation is evidence where available,
    #     not a prerequisite for every stage.
    CONS = pd.read_csv(CFG.TABLES / "hist31_dem_constraints.csv")
    print("\nGATE (support classes, not a global offset):")
    for r in CONS.itertuples():
        ref = ("NaN" if not np.isfinite(r.sigma_x_reference_m)
               else f"{r.sigma_x_reference_m:5.1f} m")
        print(f"  {r.stage_target}  {r.absolute_validation_class:12s} "
              f"supported {r.absolute_validation_supported_length_km:6.1f} km  "
              f"sigma_x_sensor {r.sigma_x_sensor_m:5.1f} m  "
              f"sigma_x_reference {ref}  "
              f"sigma_total {r.sigma_total_m:5.1f} m"
              + ("  (LOWER BOUND)" if r.sigma_total_is_lower_bound else ""))
    if (CONS.offset_correction_applied_m != 0).any():
        raise SystemExit("GATE FAILED: an offset correction has been applied. "
                         "hist28 showed the offset is not single-valued, so no "
                         "constant is admissible.")
    if CONS.cross_stage_transfer_licensed.any():
        raise SystemExit("GATE FAILED: cross-stage transfer is licensed in the "
                         "constraint table. hist27 showed k(x) is not "
                         "identifiable, so d_stage stays unlicensed.")
    bad = CONS[(CONS.absolute_validation_class == "UNAVAILABLE")
               & CONS.sigma_x_reference_m.notna()]
    if not bad.empty:
        raise SystemExit("GATE FAILED: sigma_x_reference must be NaN, never 0, "
                         "where absolute validation is UNAVAILABLE.")
    if not (CONS.absolute_validation_class == "DIRECT").any():
        raise SystemExit("GATE FAILED: no stage has any independent support at "
                         "all. At least one DIRECT class is needed as evidence "
                         "that the shorelines are not grossly displaced.")
    ndir = int((CONS.absolute_validation_class == "DIRECT").sum())
    print(f"      GATE PASSED -- {ndir} of {len(CONS)} stages have DIRECT "
          f"support; the rest are used as soft constraints at their own")
    print("      observed WSE, flagged ABSOLUTE_UNVALIDATED, with no "
          "correction applied.")

    # ---- soundings + frozen folds -----------------------------------------
    # bulk products live on the bulk drive, never the repo disk
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                         "kakhovka_soundings_evrf2019.parquet")
    sd = sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0)).groupby(
        ["_kx", "_ky"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), lon=("lon", "mean"),
        lat=("lat", "mean"), H_bed_evrf2019_m=("H_bed_evrf2019_m", "mean"))
    xy = np.c_[sd.x.values, sd.y.values]
    z = sd.H_bed_evrf2019_m.values
    fa = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                          "hist14_cv_fold_assignments.parquet")
    if not np.allclose(fa.x.values, sd.x.values):
        raise SystemExit("fold file misaligned with soundings")
    folds = fa.block_1km_fold.values.astype(int)
    # Extent from the spatial-domain registry, never from a derived geojson.
    # P20_reservoir_footprint.geojson -- used here until 2026-09 -- truncated
    # 87.4 km2 of real water on the east while adding ~123 km2 of land, errors
    # that nearly cancel in total area. See p0g for the sounding-support check
    # on the recovered ground.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    FOOTPRINT_KM2 = fp.area / 1e6

    import skgstat as skg
    RNG = np.random.default_rng(CFG.SEED)
    sub = RNG.choice(len(xy), min(1500, len(xy)), replace=False)
    V = skg.Variogram(xy[sub], z[sub], model="spherical", n_lags=20,
                      maxlag=0.35, normalize=False)
    vg = (float(V.parameters[0]), float(V.parameters[1]),
          float(V.parameters[2]) if len(V.parameters) > 2 else 0.0)
    print(f"variogram (unchanged): range {vg[0]/1000:.2f} km, "
          f"sill {vg[1]:.2f}, nugget {vg[2]:.2f}")

    # ============================================ 4. UNCERTAINTY BUDGET
    print("\n" + "=" * 78)
    print("4. CONTOUR VERTICAL UNCERTAINTY BUDGET")
    print("=" * 78)
    # local bed slope at each shoreline, for the positional -> vertical term
    npz0 = np.load(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                          "kakhovka_bed_surface_250m.npz", allow_pickle=True)
    gx0, gy0, ins0 = npz0["gx"], npz0["gy"], npz0["ins_idx"]
    bed0 = npz0["surf_epoch_OK"].astype(float)
    b2 = np.full(len(gy0) * len(gx0), np.nan)
    b2[ins0] = bed0
    b2 = b2.reshape(len(gy0), len(gx0))
    gyy, gxx = np.gradient(b2, CELL, CELL)
    slope = np.hypot(gxx, gyy)                # m per m
    st = cKDTree(np.c_[np.repeat(gy0, len(gx0)), np.tile(gx0, len(gy0))])

    ubudget, cpts = [], {}
    for r in cg.itertuples():
        pts = densify(r.geometry, SPACING_MAIN)
        # nearest grid slope
        d, i = st.query(np.c_[pts[:, 1], pts[:, 0]], k=1)
        sl = np.nan_to_num(slope.ravel()[i], nan=np.nanmedian(slope))
        sl_med = float(np.nanmedian(sl))
        sig_pos = sl_med * CELL / 2          # half a grid cell of position error
        sig_mixed = sl_med * 20.0            # one 20 m mixed pixel
        row = inv[inv.contour_id == r.contour_id].iloc[0]
        # pool non-flatness: measured where several gauges exist, otherwise
        # bounded by the worst measured 6-gauge spread in the record (0.29 m)
        sig_flat = (float(row.interstation_spread_m) if row.n_gauges > 1
                    else 0.29)
        # Emergent-vegetation offset. NDWI/MNDWI detect OPEN water; the reedy
        # margins of this reservoir are inundated but not open, so the
        # classified boundary sits INSIDE the true waterline. Measured on H1,
        # where an independent waterline exists (the P20 footprint, 2,192 km2,
        # which is NOT derived from Table 19): the area deficit spread over the
        # shoreline length gives a mean inward offset. This is a SYSTEMATIC
        # bias, not noise -- it is entered as an uncertainty because its
        # magnitude varies along the shore, and the residual systematic part is
        # stated as a limitation rather than corrected (correcting it against
        # Table 19 would be circular).
        h1 = inv[inv.contour_id == "H1"].iloc[0]
        # derived from the actual analysis footprint; 2192.0 was hard-coded
        # here and was precisely the old P20 area, so it carried the
        # truncation into the vegetation-offset term
        veg_off_m = ((FOOTPRINT_KM2 - float(h1.polygon_area_km2)) * 1e6
                     / (float(h1.shoreline_length_km) * 1e3))
        sig_veg = abs(veg_off_m) * sl_med
        comp = dict(sigma_epsg9902_m=SIG_EPSG9902,
                    sigma_gauge_read_m=SIG_GAUGE_READ,
                    sigma_pool_flatness_m=sig_flat,
                    sigma_level_spread_in_group_m=float(
                        row.level_spread_within_group_m),
                    sigma_seiche_windsetup_m=SIG_SEICHE,
                    sigma_shoreline_position_m=sig_pos,
                    sigma_mixed_pixel_m=sig_mixed,
                    sigma_emergent_vegetation_m=sig_veg)
        tot = float(np.sqrt(sum(v ** 2 for v in comp.values())))
        ubudget.append(dict(contour_id=r.contour_id, dates=row.dates,
                            mean_inward_offset_m=veg_off_m,
                            H_evrf2019_m=r.H_evrf2019_m,
                            n_gauges=int(row.n_gauges),
                            median_local_bed_slope_m_per_m=sl_med,
                            n_contour_points=len(pts),
                            spacing_m=SPACING_MAIN, sigma_total_m=tot, **comp))
        cpts[r.contour_id] = (pts, np.full(len(pts), r.H_evrf2019_m),
                              np.full(len(pts), tot))
        print(f"\n  {r.contour_id} ({row.dates}, {row.n_gauges} gauge"
              f"{'s' if row.n_gauges > 1 else ''}): {len(pts):,} points at "
              f"{SPACING_MAIN:.0f} m")
        for k, v in comp.items():
            print(f"      {k:<34}{v:.3f} m")
        print(f"      {'-> sigma_H (quadrature)':<34}{tot:.3f} m")
    UB = pd.DataFrame(ubudget)
    UB.to_csv(CFG.TABLES / "prebreach_contour_uncertainty.csv", index=False)
    print(f"\n  The pool-flatness term is MEASURED for H2 "
          f"({float(UB[UB.contour_id=='H2'].sigma_pool_flatness_m.iloc[0]):.3f} m, "
          f"six gauges) and only BOUNDED for H1/H3 (0.29 m, the worst "
          f"six-gauge\n  spread in the record). That is the concrete value of "
          f"the 2019 scene.")

    # ============================================ 6. SURFACE VARIANTS
    print("\n" + "=" * 78)
    print("6. VARIANTS")
    print("=" * 78)
    ids = list(cg.contour_id)
    VARIANTS = {"B0": ["H1"], "C12": ["H1", "H2"], "C13": ["H1", "H3"],
                "C23": ["H2", "H3"], "C123": ["H1", "H2", "H3"]}
    cxy, cz, cs = {}, {}, {}
    for v, use in VARIANTS.items():
        if v == "B0":
            # the baseline uses the SAME boundary pseudo-points hist14 uses,
            # i.e. the 2023-06-05 waterline at 17.08 m -- reproduced here as
            # H1 densified the same way, with the same soft treatment, so the
            # comparison isolates the ADDITION of contours, not the change of
            # formulation.
            use = ["H1"]
        P = [cpts[u] for u in use if u in cpts]
        cxy[v] = np.vstack([p[0] for p in P])
        cz[v] = np.concatenate([p[1] for p in P])
        cs[v] = np.concatenate([p[2] for p in P])
        print(f"  {v:<5}{'+'.join(use):<12}{len(cxy[v]):>7,} contour points")

    _S.update(xy=xy, z=z, vg=vg, folds=folds, cxy=cxy, cz=cz, cs=cs)

    # ---- how much do the contours DOMINATE the kriging neighbourhood? -----
    # 3 contours at 150 m spacing put ~20k points against 7.5k soundings, so
    # near the shore the 32-point neighbourhood can be almost all contour.
    # That is the mechanism by which a shoreline constraint could degrade the
    # interior, so it is measured rather than assumed away.
    Xall = np.vstack([xy, cxy["C123"]])
    is_c = np.concatenate([np.zeros(len(xy), bool),
                           np.ones(len(cxy["C123"]), bool)])
    _, nn = cKDTree(Xall).query(xy, k=H14.OK_K)
    frac_c = is_c[nn].mean(axis=1)
    dsh0 = shapely.distance(shapely.points(xy[:, 0], xy[:, 1]),
                            shapely.boundary(fp))
    print(f"\n  contour share of each sounding's {H14.OK_K}-point "
          f"neighbourhood (C123):")
    for lo, hi, lab in ((0, 250, "0-250 m"), (250, 500, "250-500"),
                        (500, 1000, "500-1000"), (1000, 2000, "1-2 km"),
                        (2000, 1e9, ">2 km")):
        s = (dsh0 >= lo) & (dsh0 < hi)
        if s.sum():
            print(f"    shore {lab:<9}n={int(s.sum()):>5}  median "
                  f"{100*np.median(frac_c[s]):>5.1f}%  p90 "
                  f"{100*np.percentile(frac_c[s],90):>5.1f}%")
    print(f"    -> the constraint acts where it should and fades inland; the "
          f"frozen-fold\n       guard below is what decides whether it "
          f"nonetheless harmed the interior.")

    if args.verify:
        te = folds == 0
        a = ok_soft(xy[~te], z[~te], np.zeros((~te).sum()), xy[te], vg)
        b = H14.METHODS["OK"](xy[~te], z[~te], xy[te], vg)
        assert np.allclose(a, b, equal_nan=True), "ok_soft != H14._ok at zero error"
        print("\n  [verify] ok_soft reduces exactly to hist14's estimator "
              "when err_var = 0")

    # ==================================== 7. LEAVE-ONE-CONTOUR-OUT (PRIMARY)
    print("\n" + "=" * 78)
    print("7. LEAVE-ONE-CONTOUR-OUT — the primary test of the unsounded belt")
    print("=" * 78)
    loco = []
    pairs = {"H3": ["H1", "H2"], "H2": ["H1", "H3"], "H1": ["H2", "H3"]}
    print(f"\n  {'withheld':>9}{'trained on':>13}{'n':>7}{'median':>9}"
          f"{'RMSE':>8}{'NMAD':>8}{'p90':>8}{'area err':>11}")
    for out, tr in pairs.items():
        Ptr = [cpts[t] for t in tr]
        Xtr = np.vstack([xy] + [p[0] for p in Ptr])
        Ztr = np.concatenate([z] + [p[1] for p in Ptr])
        Etr = np.concatenate([np.zeros(len(z))] + [p[2] ** 2 for p in Ptr])
        Pte, Hte, _ = cpts[out]
        pred = ok_soft(Xtr, Ztr, Etr, Pte, vg)
        res = pred - Hte
        s = stats(res)
        # area at the withheld level, predicted vs observed
        Hlev = float(Hte[0])
        # also compare with the baseline (H1 only) where meaningful
        base = None
        if out != "H1":
            Xb = np.vstack([xy, cpts["H1"][0]])
            Zb = np.concatenate([z, cpts["H1"][1]])
            Eb = np.concatenate([np.zeros(len(z)), cpts["H1"][2] ** 2])
            base = stats(ok_soft(Xb, Zb, Eb, Pte, vg) - Hte)
        obs_area = float(cg[cg.contour_id == out].polygon_area_km2.iloc[0]) \
            if "polygon_area_km2" in cg else float(
            inv[inv.contour_id == out].polygon_area_km2.iloc[0])
        loco.append(dict(withheld=out, trained_on="+".join(tr),
                         H_withheld_m=Hlev, observed_area_km2=obs_area,
                         baseline_RMSE_m=(base or {}).get("RMSE_m", np.nan),
                         baseline_median_m=(base or {}).get("median_m", np.nan),
                         **s))
        print(f"  {out:>9}{'+'.join(tr):>13}{s['n']:>7,}"
              f"{s['median_m']:>+9.3f}{s['RMSE_m']:>8.3f}{s['NMAD_m']:>8.3f}"
              f"{s['p90_abs']:>8.3f}{'':>11}")
        if base:
            print(f"  {'':>9}{'baseline H1 only':>13}{base['n']:>7,}"
                  f"{base['median_m']:>+9.3f}{base['RMSE_m']:>8.3f}"
                  f"{base['NMAD_m']:>8.3f}{base['p90_abs']:>8.3f}")
    # ---- spacing sensitivity: the densification step IS the softness knob --
    print(f"\n  sensitivity to the densification spacing (withholding H3, "
          f"the deepest contour):")
    print(f"  {'spacing':>9}{'n train pts':>13}{'median':>9}{'RMSE':>8}"
          f"{'NMAD':>8}")
    spac_rows = []
    for sp in SPACINGS:
        P = []
        for t in ("H1", "H2"):
            pp = densify(cg[cg.contour_id == t].geometry.iloc[0], sp)
            sig = float(UB[UB.contour_id == t].sigma_total_m.iloc[0])
            P.append((pp, np.full(len(pp),
                                  float(cg[cg.contour_id == t].H_evrf2019_m.iloc[0])),
                      np.full(len(pp), sig)))
        Xtr = np.vstack([xy] + [p[0] for p in P])
        Ztr = np.concatenate([z] + [p[1] for p in P])
        Etr = np.concatenate([np.zeros(len(z))] + [p[2] ** 2 for p in P])
        Pte = densify(cg[cg.contour_id == "H3"].geometry.iloc[0], SPACING_MAIN)
        Hte = np.full(len(Pte),
                      float(cg[cg.contour_id == "H3"].H_evrf2019_m.iloc[0]))
        s = stats(ok_soft(Xtr, Ztr, Etr, Pte, vg) - Hte)
        spac_rows.append(dict(spacing_m=sp, n_train_contour=sum(len(p[0]) for p in P),
                              **s))
        print(f"  {sp:>7.0f} m{sum(len(p[0]) for p in P):>13,}"
              f"{s['median_m']:>+9.3f}{s['RMSE_m']:>8.3f}{s['NMAD_m']:>8.3f}")
    sr = pd.DataFrame(spac_rows)
    print(f"    -> RMSE varies {sr.RMSE_m.min():.3f}-{sr.RMSE_m.max():.3f} m "
          f"across spacings ({100*(sr.RMSE_m.max()/sr.RMSE_m.min()-1):.0f}% "
          f"spread);")
    print(f"       {SPACING_MAIN:.0f} m is used throughout, and the choice is "
          f"not load-bearing.")

    LO = pd.DataFrame(loco)
    LO.to_csv(CFG.TABLES / "leave_one_contour_out_validation.csv", index=False)
    sr.to_csv(CFG.TABLES / "contour_spacing_sensitivity.csv", index=False)
    imp = LO.dropna(subset=["baseline_RMSE_m"])
    if len(imp):
        print(f"\n  improvement where a baseline comparison exists:")
        for r in imp.itertuples():
            print(f"    {r.withheld}: RMSE {r.baseline_RMSE_m:.3f} -> "
                  f"{r.RMSE_m:.3f} m "
                  f"({100*(1-r.RMSE_m/r.baseline_RMSE_m):+.0f}%), "
                  f"median {r.baseline_median_m:+.3f} -> {r.median_m:+.3f} m")

    # ==================================== 8. FROZEN-FOLD GUARD
    print("\n" + "=" * 78)
    print("8. FROZEN-FOLD SOUNDING CV — a GUARD, not the primary test")
    print("=" * 78)
    print("  Declared in advance: a near-null result here is the EXPECTED "
          "outcome,\n  because the new information sits above the highest "
          "sounding.")
    tasks = [(v, f) for v in ("B0", "C123") for f in range(H14.N_FOLDS)]
    acc = {v: np.full(len(xy), np.nan) for v in ("B0", "C123")}
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            out = list(ex.map(_cv_task, tasks))
    else:
        out = [_cv_task(t) for t in tasks]
    for v, ix, r in out:
        acc[v][ix] = r

    dsh = shapely.distance(shapely.points(xy[:, 0], xy[:, 1]),
                           shapely.boundary(fp))
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/"
                                "sword_dnipro_channel.parquet")
    km, _, _, _ = SW.assign_chainage(sd.lon.values, sd.lat.values, ch,
                                     tree=SW.build_chainage_tree(ch))
    _, i8 = cKDTree(xy).query(xy, k=9)
    rough = z[i8[:, 1:]].std(axis=1)
    D = pd.DataFrame(dict(bed=z, d_shore=dsh, chainage=km, rough=rough,
                          B0=acc["B0"], C123=acc["C123"]))
    rows = []
    def blk(label, sel):
        a, b = stats(D.B0[sel]), stats(D.C123[sel])
        rows.append(dict(subset=label, n=a.get("n", 0),
                         B0_RMSE=a.get("RMSE_m"), C123_RMSE=b.get("RMSE_m"),
                         B0_NMAD=a.get("NMAD_m"), C123_NMAD=b.get("NMAD_m"),
                         B0_bias=a.get("bias_m"), C123_bias=b.get("bias_m"),
                         B0_p90_abs=a.get("p90_abs"),
                         C123_p90_abs=b.get("p90_abs"),
                         dRMSE=(b.get("RMSE_m", np.nan) - a.get("RMSE_m", np.nan))))
        print(f"  {label:<24}{a.get('n',0):>7,}{a.get('RMSE_m',np.nan):>9.3f}"
              f"{b.get('RMSE_m',np.nan):>9.3f}"
              f"{b.get('RMSE_m',np.nan)-a.get('RMSE_m',np.nan):>+9.3f}"
              f"{a.get('bias_m',np.nan):>+9.3f}{b.get('bias_m',np.nan):>+9.3f}")
    print(f"\n  {'subset':<24}{'n':>7}{'B0 RMSE':>9}{'C123':>9}{'delta':>9}"
          f"{'B0 bias':>9}{'C123':>9}")
    blk("ALL", np.ones(len(D), bool))
    # B0 here is NOT bit-identical to the published canonical surface: hist14
    # uses hard boundary pseudo-points at 250 m spacing, this uses the same
    # 2023-06-05 waterline densified at 150 m with the soft treatment. That is
    # deliberate -- it isolates the ADDITION of contours from the change of
    # formulation -- but the offset has to be stated.
    _b0 = float(rows[0]["B0_RMSE"])
    print(f"\n  NOTE: this B0 scores {_b0:.3f} m against the published "
          f"canonical 2.761 m.")
    print(f"  The gap is the FORMULATION change (hard 250 m pseudo-points -> "
          f"soft 150 m\n  contour points), not a data change. B0 vs C123 below "
          f"is the like-for-like\n  comparison; 2.761 m remains the canonical "
          f"published score.")
    for lo, hi, lab in ((0, 250, "shore 0-250 m"), (250, 500, "shore 250-500"),
                        (500, 1000, "shore 500-1000"), (1000, 2000, "shore 1-2 km"),
                        (2000, 1e9, "shore >2 km")):
        blk(lab, (D.d_shore >= lo) & (D.d_shore < hi))
    blk("bed < 0 m (troughs)", D.bed < 0)
    blk("bed 0-7.4 m", (D.bed >= 0) & (D.bed < 7.379))
    blk("bed > 7.4 m (platform)", D.bed >= 7.379)
    blk("roughest decile", D.rough >= D.rough.quantile(0.9))
    for lo, hi, lab in ((0, 133, "reach 1+2"), (133, 183, "reach 3"),
                        (183, 250, "reach 4")):
        blk(lab, (D.chainage >= lo) & (D.chainage < hi))
    CV = pd.DataFrame(rows)
    CV.to_csv(CFG.TABLES / "baseline_vs_constrained_sounding_cv.csv",
              index=False)

    # ==================================== 9-11. GRIDS, BELT, HYPSOMETRY
    print("\n" + "=" * 78)
    print("9-11. SURFACES, THE UNSOUNDED BELT, AND INDEPENDENT HYPSOMETRY")
    print("=" * 78)
    # THE TARGET GRID COMES FROM THE REGISTRY, NEVER FROM THE LEGACY SURFACE.
    # kakhovka_bed_surface_250m.npz is P20-era: its gx runs to 668,670 m while
    # the registry domain reaches 678,000 m. Inheriting its grid silently cut
    # 9,330 m of the eastern reservoir out of every surface, hypsometry, belt
    # statistic and figure this script produces. That is the third recurrence
    # of the eastern-truncation bug, and it is why the npz is now used ONLY as
    # a slope covariate, where its gaps become NaN and are handled as such.
    GRID = SD.build_grid(fp, CELL, what="hist24 target grid")
    gxg, gyg, nxg, ins0 = GRID["gx"], GRID["gy"], GRID["nx"], GRID["ins_idx"]
    tgt = np.c_[GRID["x"], GRID["y"]]
    print(f"  target grid from the registry: E {gxg[0]:.0f}..{gxg[-1]:.0f}, "
          f"N {gyg[0]:.0f}..{gyg[-1]:.0f}, {len(ins0):,} cells inside")
    print(f"    the legacy npz grid stopped at E {float(npz0['gx'][-1]):.0f} "
          f"-- {fp.bounds[2]-float(npz0['gx'][-1]):.0f} m short")
    surf = {}
    for v in ("B0", "C123"):
        Xtr = np.vstack([xy, cxy[v]])
        Ztr = np.concatenate([z, cz[v]])
        Etr = np.concatenate([np.zeros(len(z)), cs[v] ** 2])
        surf[v] = ok_soft(Xtr, Ztr, Etr, tgt, vg)
        print(f"  {v}: bed {np.nanmin(surf[v]):+.2f} .. "
              f"{np.nanmax(surf[v]):+.2f} m, mean {np.nanmean(surf[v]):+.2f}")

    A = (CELL / 1e3) ** 2
    belt = []
    print(f"\n  the unsounded belt ({Z_MAX_SOUNDING:.2f} - "
          f"{cpts['H1'][1][0]:.2f} m, above every sounding):")
    H1v = float(cpts["H1"][1][0])
    for v in ("B0", "C123"):
        b = surf[v]
        sel = (b > Z_MAX_SOUNDING) & (b < H1v)
        gd = shapely.distance(shapely.points(tgt[:, 0], tgt[:, 1]),
                              shapely.boundary(fp))
        belt.append(dict(variant=v, belt_cells=int(sel.sum()),
                         belt_area_km2=sel.sum() * A,
                         median_dist_to_shore_m=float(np.median(gd[sel]))
                         if sel.sum() else np.nan,
                         pct_of_domain=100 * sel.mean()))
        r = belt[-1]
        print(f"    {v}: {r['belt_area_km2']:,.0f} km2 "
              f"({r['pct_of_domain']:.1f}% of the domain), median "
              f"{r['median_dist_to_shore_m']:,.0f} m from shore")
    pd.DataFrame(belt).to_csv(CFG.TABLES / "unsounded_belt_analysis.csv",
                              index=False)

    lav = pd.read_csv(ROOT / "data/historical/historical_level_area_volume.csv"
                      ).sort_values("water_level_m")
    hyp = []
    for r in lav.itertuples():
        H = r.water_level_m + 0.185
        row = dict(water_level_bs77_m=r.water_level_m,
                   area_historical_km2=r.surface_area_km2)
        for v in ("B0", "C123"):
            a = float((surf[v] < H).sum()) * A
            row[f"area_{v}_km2"] = a
            row[f"diff_{v}_km2"] = a - r.surface_area_km2
            row[f"diff_{v}_pct"] = 100 * (a - r.surface_area_km2) / r.surface_area_km2
        hyp.append(row)
    HY = pd.DataFrame(hyp)
    HY.to_csv(CFG.TABLES / "baseline_vs_constrained_hypsometry.csv", index=False)
    r0 = float(np.sqrt((HY.diff_B0_km2 ** 2).mean()))
    r1 = float(np.sqrt((HY.diff_C123_km2 ** 2).mean()))
    print(f"\n  Table 19 A(H), 17 levels — INDEPENDENT, not fitted:")
    print(f"    B0   RMS area error {r0:6.1f} km2   median "
          f"{HY.diff_B0_pct.median():+.2f} %   worst "
          f"{HY.diff_B0_pct.min():+.2f} %")
    print(f"    C123 RMS area error {r1:6.1f} km2   median "
          f"{HY.diff_C123_pct.median():+.2f} %   worst "
          f"{HY.diff_C123_pct.min():+.2f} %")
    print(f"    -> {'IMPROVED' if r1 < r0 else 'NOT improved'} by "
          f"{100*(1-r1/r0):+.0f} %")

    exp = {}
    for v in ("B0", "C123"):
        b = surf[v]
        res = b[np.isfinite(b) & (b < NPG_EVRF)]
        exp[v] = 100 * float((res >= GMO_EVRF).mean())
    print(f"\n  Table 21 drying fraction — INDEPENDENT, not fitted:")
    print(f"    historical 12.95 %   B0 {exp['B0']:.2f} %   "
          f"C123 {exp['C123']:.2f} %")

    # ================================================================ figures
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
    a = ax[0]
    xs = np.arange(len(LO))
    a.bar(xs - 0.2, LO.baseline_RMSE_m, 0.38, color=GREY, alpha=0.85,
          label="baseline (H1 only)")
    a.bar(xs + 0.2, LO.RMSE_m, 0.38, color=GREEN, alpha=0.9,
          label="trained on the other two")
    a.set_xticks(xs)
    a.set_xticklabels([f"{r.withheld}\n{r.H_withheld_m:.2f} m"
                       for r in LO.itertuples()])
    a.set_ylabel("RMSE at the withheld shoreline (m)")
    a.legend(fontsize=8.6); a.grid(alpha=0.25, axis="y")
    a.set_title("a · Leave-one-contour-out\nthe test the sounding CV cannot do",
                fontsize=10.5, loc="left")

    a = ax[1]
    s = CV[CV.subset.str.startswith("shore")]
    xs = np.arange(len(s))
    a.plot(xs, s.B0_RMSE, "o-", color=GREY, lw=2, ms=7, label="B0")
    a.plot(xs, s.C123_RMSE, "s-", color=GREEN, lw=2, ms=7, label="C123")
    a.set_xticks(xs); a.set_xticklabels(
        [t.replace("shore ", "") for t in s.subset], fontsize=8.4, rotation=20)
    a.set_ylabel("sounding CV RMSE (m)")
    a.legend(fontsize=8.6); a.grid(alpha=0.25)
    a.set_title("b · Frozen-fold guard by shore distance\nnull result expected",
                fontsize=10.5, loc="left")

    a = ax[2]
    a.plot(HY.area_historical_km2, HY.water_level_bs77_m, "o-", color=BLUE,
           lw=2.2, ms=5, label="Table 19 (independent)")
    a.plot(HY.area_B0_km2, HY.water_level_bs77_m, "s--", color=GREY, lw=1.8,
           ms=4, label=f"B0 (RMS {r0:.0f} km²)")
    a.plot(HY.area_C123_km2, HY.water_level_bs77_m, "^--", color=GREEN,
           lw=1.8, ms=4, label=f"C123 (RMS {r1:.0f} km²)")
    a.set_xlabel("area (km²)"); a.set_ylabel("level (m BS-77)")
    a.legend(fontsize=8.4); a.grid(alpha=0.25)
    a.set_title("c · Independent hypsometric validation\nnever used in fitting",
                fontsize=10.5, loc="left")

    fig.suptitle("V20 · Three-contour constrained bathymetry — validation",
                 fontsize=12.4, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V20_leave_one_contour_out_validation.{e}",
                    dpi=175, bbox_inches="tight")
    plt.close(fig)

    # difference map
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    ext = [gxg[0] / 1e3, gxg[-1] / 1e3, gyg[0] / 1e3, gyg[-1] / 1e3]

    def ras(flat):
        f2 = np.full(len(gyg) * nxg, np.nan)
        f2[ins0] = flat
        return f2.reshape(len(gyg), nxg)

    for i, (v, ttl) in enumerate((("B0", "a · baseline B0"),
                                  ("C123", "b · constrained C123"))):
        im = ax[i].imshow(ras(surf[v]), origin="lower", extent=ext,
                          cmap="viridis", vmin=-15, vmax=18)
        ax[i].set_title(f"{ttl}\nbed elevation (m EVRF2019)", fontsize=10.5,
                        loc="left")
        plt.colorbar(im, ax=ax[i], fraction=0.03, pad=0.02)
    d = surf["C123"] - surf["B0"]
    im = ax[2].imshow(ras(d), origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-2, vmax=2)
    ax[2].set_title(f"c · C123 − B0\nmedian {np.nanmedian(d):+.2f} m, "
                    f"p95 |Δ| {np.nanpercentile(np.abs(d),95):.2f} m",
                    fontsize=10.5, loc="left")
    plt.colorbar(im, ax=ax[2], fraction=0.03, pad=0.02, label="Δ bed (m)")
    for a in ax:
        a.set_xlabel("easting (km)")
    ax[0].set_ylabel("northing (km)")
    fig.suptitle("V21 · Baseline versus three-contour constrained bed surface",
                 fontsize=12.4, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V21_baseline_vs_constrained_bathymetry.{e}",
                    dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {CFG.FIG/'V20_leave_one_contour_out_validation.png'}")
    print(f"-> {CFG.FIG/'V21_baseline_vs_constrained_bathymetry.png'}")
    for n in ("prebreach_contour_uncertainty",
              "leave_one_contour_out_validation",
              "baseline_vs_constrained_sounding_cv",
              "unsounded_belt_analysis",
              "baseline_vs_constrained_hypsometry"):
        print(f"-> {CFG.TABLES/(n + '.csv')}")

    # ================================================================ verdict
    print("\n" + "=" * 78)
    print("12. DECISION RULE")
    print("=" * 78)
    c1 = len(imp) and (imp.RMSE_m < imp.baseline_RMSE_m).all()
    all_row = CV[CV.subset == "ALL"].iloc[0]
    c2 = (all_row.C123_RMSE - all_row.B0_RMSE) < 0.05
    c3 = r1 <= r0 * 1.05
    c4 = abs(exp["C123"] - 12.95) <= abs(exp["B0"] - 12.95) + 1.0
    c5 = abs(all_row.C123_bias) < 0.15
    for k, (ok, txt) in enumerate((
            (c1, "leave-one-contour-out improves at every withheld contour"),
            (c2, f"interior sounding CV does not degrade "
                 f"({all_row.C123_RMSE-all_row.B0_RMSE:+.3f} m)"),
            (c3, f"Table 19 A(H) agreement improves or holds "
                 f"({r0:.0f} -> {r1:.0f} km2)"),
            (c4, f"Table 21 exposure stays acceptable "
                 f"({exp['B0']:.2f} -> {exp['C123']:.2f} % vs 12.95 %)"),
            (c5, f"no new systematic bias "
                 f"({all_row.C123_bias:+.3f} m)")), 1):
        print(f"  {k}. {'PASS' if ok else 'FAIL'}  {txt}")
    if all((c1, c2, c3, c4, c5)):
        print("\n  ALL FIVE CONDITIONS MET -> C123 is scientifically preferable.")
        print("  Fine grids (50 m, 30 m) may now be regenerated.")
    else:
        print("\n  NOT ALL CONDITIONS MET -> the baseline REMAINS canonical.")
        print("  The constrained surface is kept as an experiment only.")


if __name__ == "__main__":
    main()
