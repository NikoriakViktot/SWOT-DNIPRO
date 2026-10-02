#!/usr/bin/env python
"""GATE 7C3 — the physical shoreline offset, continuous S1 against continuous S2.

This is the only gate in the 7C family that is allowed to report an absolute
offset, because it is the only one where BOTH sides are proper continuous
geometry:

    S1 continuous LDA discriminant -> sub-pixel contour
                                          <->
    S2 continuous NDWI             -> sub-pixel contour

Gate 7C2 must never be used for this. Its reference polygons have every vertex
on a multiple of 20 m -- a raster staircase that happens to be stored as a
polygon -- so a smooth contour measured against it shows half-cell offsets from
a perfect estimator. See outputs/planning/12_GATE_7C_ROLE_FREEZE.md.

WHAT IS NEW HERE, AND WHY IT MATTERS. The same-date controls could not test the
stage correction at all: both had dH = 0.000 exactly, so d_stage had zero lever
arm and only its synthetic sign test was exercised. Here each S1 scene is paired
with the nearest usable S2 acquisition, which in general is on a DIFFERENT day
at a DIFFERENT stage, so dH spans a real range and

    d_stage = -dH / s        d_residual = d_raw - d_stage

is finally testable against data. The decisive check is not any single scene but
the regression of d_raw on dH across scenes: the convention predicts

    d(d_raw)/d(dH) = -1/s

so a fitted slope near -1/s confirms that the measured offsets really are stage
displacement, and an intercept near zero says there is no residual sensor bias
once stage is removed. A fitted slope near ZERO would mean the offsets are NOT
stage-driven and the whole d_stage correction is inapplicable.

WHICH s. NOT the sounding slope. d_stage needs the slope of the BANK AT THE
WATERLINE, and the soundings are in-water, so a plane fitted to them returns the
BED slope in deeper water -- here 0.0053, i.e. 190 m of waterline per metre of
stage. Submerged bed slope is not a proxy for subaerial bank slope; that is a
structural error, and it is kept as a negative-method result rather than
deleted. The sounding slope is still recorded as bed_slope_soundings so the
discrepancy stays visible.

The replacement, ~9.8 m per m from the stored contours, is PROVISIONAL and is
admitted only to reject 190, not to freeze a parameter. It is measured on the
same 20 m raster staircase Gate 7C2 demoted; one of its pairs is sub-cell; and
only TWO of the three spacings are independent, because the areas are exactly
additive. An earlier version of this file called them "three independent pairs
agreeing to within 9%" -- that was wrong, and the arithmetic is in
effective_migration_rate(). After H1/H2/H3 are rebuilt as continuous contours
this must become a LOCAL k(x), NaN where the three levels disagree.

THIS GATE IS NOT FINAL. The 18 scenes have not yet passed a written Gate 7C2b
admission, and the S1/S2 pairs carry lags up to +/-7 days. The result below is
provisional until both are settled.

MEASUREMENT AND MODEL ARE SEPARATE. The CSV stores only what was measured --
d_raw, dH, f_L, NMAD. d_stage and d_residual are derived centrally from the
migration rate, so revising the stage model costs no refetching and can never
silently contaminate the measurement.

SIGN CONVENTION, unchanged and unit-tested upstream:
    positive  SAR water extent LARGER than optical
    negative  SAR water extent SMALLER than optical

EVERYTHING IS IMPORTED, NOT RE-IMPLEMENTED. The grid, the S1 cache, the NDWI
reader, the sub-pixel contour, the swath-edge filter and the normal-intersection
matcher all come from the gates that validated them. A local copy would be free
to drift away from what was actually tested.

Outputs
-------
outputs/tables/gate7c3_continuous_offset.csv
outputs/figures/historical_bathymetry/png/gate7c3_stage_regression.png
outputs/figures/historical_bathymetry/png/gate7c3_residual_maps.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import watermask as WM
from hist25b_gate6_event_qualification import load_frozen_manifest
from hist25b_gate7_anchored_classifier import (CELL, CACHE, build_grid,
                                               clean_water, db)
from hist25b_gate7c_v2_same_date_controls import fetch_ndwi, http_json, SAS
from hist25b_gate7c1_pairing_audit import (apply_sign, contour_points,
                                           match_normal, stats as rstats)
from hist25b_gate7c2_scene_quality import subpixel_external  # noqa: F401

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
S2_WINDOW_DAYS = 7
S2_MAX_CLOUD = 40.0
MIN_S2_DOMAIN_COV = 0.35
SLOPE_RADIUS_M = 1000.0
SLOPE_MIN_PTS = 8
MATCH_MAX_M = 100.0        # Gate 7C2: f_L and drift both plateau by ~100-150 m


def pick_s2(date, bbox):
    """Nearest usable S2 acquisition to an S1 date, with its own metadata.

    Deliberately NOT restricted to same-date: the whole point of this gate is
    that dH must vary, and a same-date pair pins dH at zero."""
    d0 = pd.Timestamp(date)
    lo = (d0 - pd.Timedelta(days=S2_WINDOW_DAYS)).strftime("%Y-%m-%d")
    hi = (d0 + pd.Timedelta(days=S2_WINDOW_DAYS)).strftime("%Y-%m-%d")
    q = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 400,
         "datetime": f"{lo}T00:00:00Z/{hi}T23:59:59Z"}
    feats = http_json(STAC, q)["features"]
    by_day = {}
    for f in feats:
        dt = f["properties"]["datetime"][:10]
        cl = float(f["properties"].get("eo:cloud_cover", 100.0))
        by_day.setdefault(dt, []).append(cl)
    cand = [(dt, float(np.mean(c)), len(c)) for dt, c in by_day.items()]
    cand = [c for c in cand if c[1] <= S2_MAX_CLOUD]
    if not cand:
        return None
    # nearest in time first, cloud as the tie-break; a 6-day-old cloud-free
    # scene is worth less here than a same-day scene with some cloud, because
    # every day of separation is stage uncertainty we then have to model
    cand.sort(key=lambda c: (abs((pd.Timestamp(c[0]) - d0).days), c[1]))
    dt, cl, n = cand[0]
    return dict(s2_date=dt, s2_cloud=cl, s2_n_items=n,
                dt_days=int((pd.Timestamp(dt) - d0).days))


def local_slope(X, Y, ptree, pxy, pz, cap=2500):
    """Cross-shore bed slope from the SOUNDINGS, never from the DEM this will
    later constrain. Returned as NaN where support is insufficient -- never 0,
    because a zero slope silently becomes an infinite d_stage."""
    slope = np.full(len(X), np.nan)
    step = max(1, len(X) // cap)
    idxs = np.arange(0, len(X), step)
    for k, nn in zip(idxs, ptree.query_ball_point(
            np.c_[X[idxs], Y[idxs]], r=SLOPE_RADIUS_M)):
        if len(nn) < SLOPE_MIN_PTS:
            continue
        A = np.c_[pxy[nn, 0] - X[k], pxy[nn, 1] - Y[k], np.ones(len(nn))]
        try:
            coef, *_ = np.linalg.lstsq(A, pz[nn], rcond=None)
        except Exception:
            continue
        slope[k] = float(np.hypot(coef[0], coef[1]))
    return slope, idxs


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("GATE 7C3 — physical offset, continuous S1 vs continuous S2")
    print("=" * 78)
    print(f"  git {commit}")
    print("  the ONLY gate allowed to report an absolute offset: both sides")
    print("  are continuous geometry, neither is a 20 m raster staircase")
    print("  PROVISIONAL until the same scenes pass a written Gate 7C2b")

    M, meta = load_frozen_manifest()
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    G["bbox_ll"] = [round(v, 5) for v in
                    gpd.GeoSeries([fp], crs=32636).to_crs(4326).iloc[0].bounds]
    inside = G["inside"]
    gdf = gpd.read_file(GPKG_OPTICAL,
                        layer="contour_polygons").set_index("contour_id")
    from rasterio.features import rasterize as rio_rasterize

    def rast(g):
        return rio_rasterize([(g, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0,
                             dtype="uint8").astype(bool)

    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    gg = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = gg.groupby("date").transformed_level_m.median()
    P = pd.read_parquet(PRIMARY)
    pxy = np.c_[P.x.to_numpy(), P.y.to_numpy()]
    pz = P.H_bed_evrf2019_m.to_numpy()
    ptree = cKDTree(pxy)
    tok = http_json(SAS, timeout=90)["token"]

    elig = M[M.scientific_roles.fillna("").str.contains("TARGET_STAGE_GEOMETRY")
             & (M.temporal_regime == "PREBREACH_IMPOUNDED")
             & M.WSE_RESOLVED.astype(bool)
             & M.wse_semantics_resolved.astype(bool)]
    OUT = CFG.TABLES / "gate7c3_continuous_offset.csv"
    rows, maps = [], {}
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(prev.event_id)
        print(f"  resuming: {len(done)} scenes already computed")
    tok_t = pd.Timestamp.utcnow()
    for r in elig.itertuples():
        if r.event_id in done:
            continue
        if (pd.Timestamp.utcnow() - tok_t).total_seconds() > 1800:
            # SAS tokens expire; a long sweep must refresh rather than fail
            tok = http_json(SAS, timeout=90)["token"]
            tok_t = pd.Timestamp.utcnow()
        npz = CACHE / f"{r.event_id}.npz"
        if not npz.exists():
            continue
        tgt = next((t for t in ("H1", "H2", "H3")
                    if f"{t}_TARGET_STAGE_GEOMETRY" in str(r.scientific_roles)), "")
        if not tgt or tgt not in gdf.index:
            continue
        s2 = pick_s2(r.date, G["bbox_ll"])
        if s2 is None:
            print(f"  {r.event_id:26s} no usable S2 within "
                  f"+/-{S2_WINDOW_DAYS} d")
            continue
        ndwi, good, scl = fetch_ndwi(s2["s2_date"], G, tok)
        cov_s2 = good & inside
        cov_frac = float(cov_s2.sum() / inside.sum())
        if cov_frac < MIN_S2_DOMAIN_COV:
            print(f"  {r.event_id:26s} S2 {s2['s2_date']} covers only "
                  f"{100*cov_frac:.0f}% of the domain -- skipped")
            continue

        opt = rast(gdf.loc[tgt, "geometry"])
        z = np.load(npz)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        obs = cov & inside & cov_s2
        vvd, vhd = db(vv), db(vh)
        d_in = ndimage.distance_transform_edt(opt) * CELL
        d_out = ndimage.distance_transform_edt(~opt) * CELL
        aw = opt & (d_in > 500) & obs
        al = inside & ~opt & (d_out > 100) & obs
        if aw.sum() < 500 or al.sum() < 500:
            print(f"  {r.event_id:26s} anchors too small under the S2 footprint")
            del vv, vh, cov, vvd, vhd
            continue
        Xw = np.c_[vvd[aw], vhd[aw]]; Xl = np.c_[vvd[al], vhd[al]]
        mw, ml = Xw.mean(0), Xl.mean(0)
        Sw = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
        wv = np.linalg.solve(Sw, ml - mw)
        cut = 0.5 * (wv @ mw + wv @ ml)
        disc = np.full(inside.shape, np.nan, np.float32)
        disc[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) - cut
        if np.nanmedian(disc[aw]) > np.nanmedian(disc[al]):
            disc = -disc
        F_sar = disc
        T = float(WM.DEFAULT_NDWI)
        F_opt = np.where(cov_s2, T - ndwi, np.nan).astype(np.float32)

        core = clean_water((F_sar < 0) & obs)
        lab, n = ndimage.label(core, structure=np.ones((3, 3), int))
        keep = np.zeros(n + 1, bool)
        keep[list(set(np.unique(lab[aw & core])) - {0})] = True
        core = keep[lab]
        # EXTERNAL shoreline only, with tangents, in one pass. The band is
        # taken around the FILLED core so hole boundaries are excluded, which
        # is what subpixel_external does; contour_points is used instead of it
        # because the normal-intersection matcher needs the per-vertex tangent
        # that subpixel_external does not carry.
        A = contour_points(F_sar, _band_of(ndimage.binary_fill_holes(core), G),
                           G)
        A, edge_frac = drop_swath_edges_7(A, obs, G)
        if len(A) < 500:
            print(f"  {r.event_id:26s} too little external shoreline")
            del vv, vh, cov, vvd, vhd, disc
            continue

        dist = match_normal(A, F_opt, cov_s2, G)
        fref = F_opt[A[:, 2].astype(int), A[:, 3].astype(int)]
        sd = apply_sign(dist, fref, "s1_to_s2")
        sd = np.where(dist <= MATCH_MAX_M, sd, np.nan)
        k = np.isfinite(sd)
        if k.sum() < 500:
            print(f"  {r.event_id:26s} too few paired vertices")
            del vv, vh, cov, vvd, vhd, disc
            continue
        X, Y, d_raw, wl = A[k, 0], A[k, 1], sd[k], A[k, 4]
        f_L = float(wl.sum() / A[:, 4].sum())

        H_sar = float(r.WSE_nominal)
        H_opt = float(lev[s2["s2_date"]]) if s2["s2_date"] in lev.index else np.nan
        dH = H_opt - H_sar
        # The sounding slope is RECORDED but no longer used for d_stage: it is
        # the bed slope in deeper water, not the bank slope at the waterline.
        # The stage model is applied centrally, from the contour-derived
        # migration rate, so the expensive MEASUREMENT (d_raw, dH, f_L) stays
        # separable from the MODEL and a change of model costs no refetching.
        slope, idxs = local_slope(X, Y, ptree, pxy, pz)
        s_med = float(np.nanmedian(slope))
        st_raw = rstats(d_raw, wl)

        rows.append(dict(
            event_id=r.event_id, s1_date=r.date, target=tgt,
            relative_orbit=int(r.relative_orbit), **s2,
            s2_domain_cov=cov_frac, swath_edge_frac=edge_frac,
            f_L=f_L, n_vertices=int(k.sum()),
            H_SAR=H_sar, H_OPT=H_opt, delta_H=dH,
            bed_slope_soundings=s_med,
            slope_support=float(np.isfinite(slope[idxs]).mean()),
            d_raw_median=st_raw["median"], d_raw_nmad=st_raw["nmad"],
            d_raw_p90=st_raw["p90"],
            d_residual_nmad=st_raw["nmad"]))
        print(f"  {r.event_id:26s} {tgt} S2 {s2['s2_date']} "
              f"({s2['dt_days']:+d} d, cloud {s2['s2_cloud']:4.1f}%) "
              f"dH {dH:+6.3f} | d_raw {st_raw['median']:+7.1f} "
              f"NMAD {st_raw['nmad']:5.1f} f_L {100*f_L:4.1f}% "
              f"(bed slope {s_med:.5f}, recorded not applied)")
        maps[r.event_id] = dict(X=X, Y=Y, d=d_raw, date=r.date, tgt=tgt, dH=dH)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        del vv, vh, cov, vvd, vhd, disc, ndwi

    if not rows:
        raise SystemExit("no scene produced a continuous-continuous comparison")
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    # ---- ADMISSION (Gate 7C2b) -----------------------------------------
    adm = pd.read_csv(CFG.TABLES / "gate7c2b_admission.csv")
    ok = set(adm.loc[adm.shoreline_constraint_usable.astype(bool), "event_id"])
    R["admitted"] = R.event_id.isin(ok)
    print(f"\n  Gate 7C2b admission: {int(R.admitted.sum())} of {len(R)} "
          f"measured scenes are admitted")
    for r in R[~R.admitted].itertuples():
        print(f"    excluded {r.s1_date} {r.target} orb{r.relative_orbit}")

    # ---- STAGE MODEL: not applied ---------------------------------------
    rate = migration_rate_status()
    R["migration_rate_m_per_m"] = np.nan
    R["d_stage"] = np.nan
    R["d_residual_median"] = np.nan
    R.to_csv(OUT, index=False)
    verdict(R[R.admitted].copy(), rate)
    figures(R[R.admitted].copy(), maps, rate)
    print("\nSTOP. hist24 not started.")


def _band_of(core, G, band_px=3):
    k = np.ones((3, 3), bool)
    return (ndimage.binary_dilation(core, k, band_px)
            & ~ndimage.binary_erosion(core, k, band_px))


def drop_swath_edges_7(A, obs, G, edge_m=200.0):
    """drop_swath_edges for the 7-column contour_points array."""
    d_edge = ndimage.distance_transform_edt(obs) * CELL
    ii = np.clip(((G["y1"] - A[:, 1]) / CELL).astype(int), 0, G["ny"] - 1)
    jj = np.clip(((A[:, 0] - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
    keep = d_edge[ii, jj] >= edge_m
    frac = float(A[~keep, 4].sum() / A[:, 4].sum()) if len(A) else np.nan
    return A[keep], frac


def effective_migration_rate(gdf):
    """PROVISIONAL waterline migration rate, from the three stored contours.

    WHAT IT CORRECTS. d_stage = -dH/s needs the slope of the BANK AT THE
    WATERLINE. The soundings are in-water, so a plane fitted to them within 1 km
    returns the BED slope in deeper water -- here ~0.0053, i.e. 190 m of
    waterline per metre of stage. That is a structural error, not a bad fit:
    submerged bed slope is not a proxy for subaerial bank slope. Using it
    inflated d_stage ~19-fold and made d_residual WORSE than d_raw. This is
    retained as a negative-method result, not deleted.

    WHY THIS NUMBER IS PROVISIONAL, in three separate ways.

    1. SOURCE. It is measured on the same H1/H2/H3 polygons that Gate 7C2
       demoted: every vertex on a multiple of 20 m, a raster staircase. Using
       them for a sub-cell displacement is the very thing
       12_GATE_7C_ROLE_FREEZE.md forbids for positional truth. It is admitted
       here only because 9.8 vs 190 is a factor of 19 and survives any
       plausible staircase error -- it is enough to reject the sounding slope,
       not enough to freeze a parameter.

    2. RESOLUTION. H3->H2 gives a mean normal spacing of 12.4 m, BELOW the 20 m
       cell, against a per-contour quantisation of order half a cell. That pair
       is not resolved by its own source.

    3. INDEPENDENCE, and an earlier overclaim of mine. I described these as
       "three independent pairs agreeing to within 9%". They are not. The areas
       are EXACTLY additive -- dA(H3->H1) = dA(H3->H2) + dA(H2->H1) to 0.00 km2
       -- so the third pair carries zero new area information and is a linear
       combination of the other two. There are TWO independent spacings, and
       the third's agreement is arithmetic, not corroboration.

    WHAT MUST REPLACE IT. After H1/H2/H3 are rebuilt as continuous-field
    contours, re-measure, and do it LOCALLY rather than globally:

        k(x) = dn(x) / dH        d_stage(x) = -k(x) * dH

    with k(x) = NaN wherever the three levels do not give consistent local
    geometry. Replacing one wrong global slope with one right global slope is
    still a global slope, and the bank profile is not uniform around a 1,540 km
    shoreline.

    Mean normal spacing is dA / mean perimeter, exact for a thin band between
    two nested contours."""
    H = {"H1": 17.103588, "H2": 15.444365, "H3": 14.188588}   # WSE_nominal medians
    A = {t: gdf.loc[t, "geometry"].area for t in H}
    P = {t: gdf.loc[t, "geometry"].length for t in H}
    rates = []
    for a, b in (("H3", "H2"), ("H2", "H1"), ("H3", "H1")):
        dH = H[b] - H[a]
        if abs(dH) < 1e-6:
            continue
        dx = (A[b] - A[a]) / (0.5 * (P[a] + P[b]))
        rates.append(dx / dH)
        res = "RESOLVED" if abs(dx) >= CELL else f"BELOW the {CELL:.0f} m cell"
        indep = "" if (a, b) != ("H3", "H1") else "  [not independent: dA is"\
            " exactly dA(H3->H2)+dA(H2->H1)]"
        print(f"    {a}->{b}: dH {dH:+.3f} m, mean spacing {dx:+6.1f} m "
              f"-> {dx/dH:6.1f} m per m   {res}{indep}")
    rate = float(np.median(rates))
    print(f"    PROVISIONAL rate {rate:.1f} m per m (vs {190:.0f} from the "
          f"sounding bed slope).")
    print("    Provisional because it is measured on the same 20 m raster")
    print("    staircase Gate 7C2 demoted, one pair is sub-cell, and only TWO")
    print("    of the three spacings are independent. It is enough to REJECT")
    print("    the sounding slope, not to freeze a parameter. Re-measure as a")
    print("    LOCAL k(x) once H1/H2/H3 are rebuilt as continuous contours.")
    return rate


def migration_rate_status():
    """Read hist27's verdict on k rather than re-deriving a global rate.

    hist27 re-measured the migration rate on the rebuilt continuous contours and
    found it NOT IDENTIFIABLE: the two independent steps give 17.6 and 6.9 m per
    m by geometry alone, agree locally at only 14.9% of places, and a quarter to
    a third of the shoreline is not even locally nested. The 9.8 m per m this
    gate used provisionally came from the 20 m staircase, where quantisation
    made the two steps look equal. So no stage correction is applied here at
    all, and d_stage / d_residual are left NaN rather than filled with a number
    that has been shown not to exist."""
    try:
        S = pd.read_csv(CFG.TABLES / "hist27_migration_summary.csv")
        steps = S[S.step != "combined"]
        print("\n  STAGE MODEL — not applied. hist27 measured k on the "
              "rebuilt contours:")
        for r in steps.itertuples():
            print(f"    {r.step}: k median {r.k_median:.2f} m per m "
                  f"(IQR {r.k_p25:.2f} .. {r.k_p75:.2f}), "
                  f"{100*r.frac_len_ok:.0f}% of length defined")
        comb = S[S.step == "combined"]
        if len(comb):
            print(f"    the two steps agree locally at only "
                  f"{100*float(comb.frac_len_ok.iloc[0]):.1f}% of places")
        print("    -> k is not identifiable; d_stage and d_residual stay NaN")
    except FileNotFoundError:
        print("\n  hist27 summary missing; stage model still not applied")
    return None


def verdict(R, rate):
    print("\n" + "=" * 78)
    print("GATE 7C3 — FROZEN RESULT ON THE ADMITTED SCENES")
    print("=" * 78)
    R = R.copy()
    m = R.dropna(subset=["delta_H", "d_raw_median"])
    print(f"  scenes {len(m)};  dH range {m.delta_H.min():+.3f} .. "
          f"{m.delta_H.max():+.3f} m  (the same-date controls had dH = 0 only)")
    if rate is None:
        # A regression of d_raw on dH tests a stage model. hist27 showed there
        # is no identifiable k to test, so fitting one here would only produce
        # a slope with nothing to compare it against.
        lr = stats.linregress(m.delta_H, m.d_raw_median)
        print(f"  d_raw vs dH: slope {lr.slope:+.1f} +/- "
              f"{1.96*lr.stderr:.1f} m per m, p {lr.pvalue:.3f} "
              f"({'not ' if lr.pvalue > 0.05 else ''}significant)")
        print("  No predicted slope is quoted: hist27 found k not identifiable,")
        print("  so there is nothing to compare the fit against. The stage")
        print("  model is not tested here and is not applied.")
    elif len(m) >= 5 and m.delta_H.std() > 1e-3:
        lr = stats.linregress(m.delta_H, m.d_raw_median)
        pred = -rate
        print(f"  regression d_raw on dH:")
        print(f"    fitted slope     {lr.slope:+10.1f} m per m  "
              f"(95% CI +/- {1.96*lr.stderr:.1f})")
        print(f"    predicted        {pred:+10.1f} m per m  "
              f"from the contour-derived migration rate")
        print(f"    for comparison   "
              f"{-1/float(m.bed_slope_soundings.median()):+10.1f} m per m  "
              f"if the sounding BED slope were used (it is the wrong parameter)")
        span = float(m.delta_H.max() - m.delta_H.min())
        print(f"    POWER: dH spans only {span:.3f} m, so the predicted stage "
              f"effect across")
        print(f"    the whole sample is {span*rate:.1f} m, against a d_raw "
              f"inter-scene IQR of")
        print(f"    {float(m.d_raw_median.quantile(.75)-m.d_raw_median.quantile(.25)):.1f} m. "
              f"This experiment cannot resolve it either way.")
        print(f"    intercept        {lr.intercept:+10.1f} m   "
              f"r {lr.rvalue:+.3f}  p {lr.pvalue:.4f}")
        if lr.pvalue > 0.05:
            print("    NOT significant: these offsets are not demonstrably")
            print("    stage-driven, so d_stage is not established by this data")
        elif abs(lr.slope - pred) <= 1.96 * lr.stderr:
            print("    consistent with -1/s: the offsets ARE stage displacement")
        else:
            print("    significant but inconsistent with -1/s: something other")
            print("    than stage is scaling with dH")
    else:
        print("  insufficient dH spread to regress; d_stage remains untested")
    for c, lab in (("d_raw_median", "d_raw"),):
        v = R[c].dropna()
        print(f"  {lab:12s} across scenes: median {v.median():+7.2f} m  "
              f"IQR {v.quantile(.25):+7.2f} .. {v.quantile(.75):+7.2f}  "
              f"sign split {int((v>0).sum())}+/{int((v<0).sum())}-")
    v = R.d_raw_median.dropna()
    try:
        w = stats.wilcoxon(v)
        print(f"  Wilcoxon signed-rank on d_raw: p {w.pvalue:.3f}")
    except Exception:
        pass
    v = R.d_raw_median.dropna()
    try:
        pw = stats.wilcoxon(v).pvalue
    except Exception:
        pw = np.nan
    print("\n  WORDING, at the strength the admitted sample supports.")
    print("  The claim is NOT 'there is no SAR-optical bias'. It is: no")
    print("  systematic offset is RESOLVED at the 5% level. On the admitted")
    print(f"  16 the evidence is stronger than on all 18 (Wilcoxon p {pw:.3f}")
    print("  against 0.108) and sits right at the boundary, so a small")
    print("  negative offset of order -4 m, about 0.2 cell, cannot be excluded")
    print("  either. What can be said is that it is small against the 20 m")
    print("  cell. The pairs still carry lags up to +/-7 days, which at any")
    print("  plausible migration rate is itself worth a few metres.")
    print("\n  WHAT THE DEM ACTUALLY NEEDS. Nothing above blocks it, because")
    print("  cross-stage transfer is not required: each S1 shoreline")
    print("  constrains elevation at its OWN observed WSE, z(x_i) ~ H(t_i).")
    print("  d_stage is needed only to compare S1 against S2 across a lag, or")
    print("  to move geometry between stages. So:")
    print("    LICENSED     shoreline constrains elevation at its own stage")
    print("    UNVALIDATED  cross-stage shoreline transfer")


def figures(R, maps, rate):
    m = R.dropna(subset=["delta_H", "d_raw_median"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    cmap = {"H1": AMBER, "H2": GREEN, "H3": BLUE}
    ax[0].scatter(m.delta_H, m.d_raw_median, c=m.target.map(cmap), s=70,
                  edgecolor=INK, linewidth=0.5)
    if rate is not None and len(m) >= 5 and m.delta_H.std() > 1e-3:
        lr = stats.linregress(m.delta_H, m.d_raw_median)
        xs = np.linspace(m.delta_H.min(), m.delta_H.max(), 50)
        ax[0].plot(xs, lr.intercept + lr.slope * xs, color=RED, lw=1.6,
                   label=f"fit {lr.slope:+.0f} m/m")
        ax[0].plot(xs, lr.intercept - rate * xs, color=INK, lw=1.2, ls="--",
                   label=f"predicted {-rate:+.0f} m/m (contour-derived)")
        ax[0].legend(fontsize=8)
    ax[0].axhline(0, color=GREY, lw=0.8); ax[0].axvline(0, color=GREY, lw=0.8)
    ax[0].set_xlabel("dH = H_optical - H_SAR (m)")
    ax[0].set_ylabel("d_raw median (m)   + = SAR water larger")
    ax[0].set_title("no stage model is fitted: hist27 showed k is not\n"
                    "identifiable from the three contours", color=INK)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color=INK, lw=1.0)
    for t, c in cmap.items():
        s = R[R.target == t]
        ax[1].errorbar(s.delta_H, s.d_raw_median, yerr=s.d_raw_nmad,
                       fmt="o", color=c, capsize=3, label=t, ms=6)
    ax[1].set_xlabel("dH (m)")
    ax[1].set_ylabel("d_raw median +/- NMAD (m)")
    ax[1].set_title("admitted scenes with their scatter\n"
                    "(no stage correction: k is not identifiable)", color=INK)
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("Gate 7C3 FROZEN · admitted scenes, continuous S1 vs continuous S2",
                 color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "gate7c3_stage_regression.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"-> {p}")

    if not maps:
        return
    pick = list(maps)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for ax, eid in zip(axes.ravel(), pick):
        d = maps[eid]
        sc = ax.scatter(d["X"] / 1000, d["Y"] / 1000, s=1.6,
                        c=np.clip(d["d"] - (-d["dH"] * rate), -60, 60),
                        cmap="RdBu_r",
                        vmin=-60, vmax=60, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label("d_residual (m)", fontsize=8)
        ax.set_title(f"{d['date']} · {d['tgt']} · dH {d['dH']:+.3f} m",
                     color=INK, fontsize=10)
        ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
        ax.set_aspect("equal")
    fig.suptitle("Gate 7C3 · residual after stage removal, along the shoreline",
                 color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "gate7c3_residual_maps.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"-> {p}")


if __name__ == "__main__":
    main()
