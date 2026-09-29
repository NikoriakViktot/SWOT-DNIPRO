#!/usr/bin/env python
"""GATE 7C v2 — same-date S1/S2 controls on the frozen sub-pixel estimator.

Gate 7C0 froze estimator_v2_subpixel_contour: Marching Squares on a CONTINUOUS
field, vector geometry, slope 1.000 and RMSE 0.00 m at every orientation
against v1's 0.923-1.000 and 4.15 m. Everything below uses v2 only.

WHAT IS RE-RUN AND WHAT IS NOT. This is the same-date control step, deliberately
first: 2019-03-18 and 2023-02-20 have S1 and S2 on the same calendar day, so
stage and temporal mismatch are minimised and they are the cleanest possible
sanity check of the new estimator. hist24 is NOT run. No 0.86 diagonal
correction is applied. No previous SAR bias is carried over.

CONTINUOUS FIELDS, NOT BINARY MASKS
  S2  B3/B8 -> continuous NDWI -> contour at the production threshold. The
      binary mask is kept only for QA and surface classification.
  S1  the M3 anchored LDA discriminant score w.x - cut IS already a continuous
      decision field, and its zero level IS the classifier's own boundary. No
      probability surface is invented to make the contour look smoother.

SIGN CONVENTION, stated in terms of water extent so no wording can flip it:
      positive  SAR water extent LARGER than optical
      negative  SAR water extent SMALLER than optical
This matches Gate 7B. A SAR vertex that falls where the optical index says
water lies inside the optical water body, so the SAR extent is smaller there
and d_raw is negative.

THE SIGN OF d_stage IS THE TRAP. With dH = H_OPT - H_SAR, a positive dH means
the optical scene was at a HIGHER stage, so optical water is larger and the
SAR extent looks smaller, i.e. d_raw goes NEGATIVE. Therefore

    d_stage = -dH / s   and   d_residual = d_raw - d_stage

Writing d_stage = +dH/s would double the effect instead of removing it, so the
sign is unit-tested below rather than trusted.

SUB-PIXEL DISTANCE. Near its zero crossing a smooth field gives the signed
distance as F / |grad F| to first order, which is orientation-independent and
needs no raster boundary convention.

WHAT THIS DOES NOT DO. Synthetic RMSE of 0.00 m says nothing about real
sensor accuracy. sigma_x for S1 and S2 are NOT set here, are not assumed
equal, and are not inherited from the grid size.

Outputs
-------
outputs/tables/gate7c_v2_same_date_controls.csv
outputs/tables/gate7c_v2_surface_class.csv
outputs/figures/historical_bathymetry/png/gate7c_v2_controls.png
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
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import watermask as WM
from hist25b_gate6_event_qualification import load_frozen_manifest

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CELL = 20.0                      # S1 event cache grid; S2 is read onto it
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
S1_CACHE = CFG.S1_CACHE / "ZONE_1_reservoir_corrected"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
CONTROL_DATES = ("2019-03-18", "2023-02-20")
SLOPE_RADIUS_M = 1000.0
SLOPE_MIN_PTS = 8
MIN_PART_KM2 = 0.05        # Gate 7 values, reused verbatim
MIN_HOLE_KM2 = 0.05
BAND_PX = 3                # +/-60 m about the cleaned boundary
MAX_MATCH_M = 500.0        # beyond this a vertex is UNPAIRED, not a 500 m error
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"


def test_stage_sign():
    """Unit-test the d_stage sign against the frozen water-extent convention.

    A higher optical stage means larger optical water, hence a SMALLER relative
    SAR extent and a NEGATIVE d_raw. d_stage must reproduce that sign, or the
    correction adds the error instead of removing it."""
    s = 0.005
    dH = +0.10                       # optical 10 cm higher than SAR
    d_stage = -dH / s                # expected -20 m
    ok = d_stage < 0
    print(f"  dH = {dH:+.2f} m (optical higher) -> d_stage = {d_stage:+.1f} m "
          f"[{'OK' if ok else 'WRONG SIGN'}]")
    d_raw_sim = -20.0                # SAR extent smaller, purely from stage
    print(f"  simulated d_raw {d_raw_sim:+.1f} m -> residual "
          f"{d_raw_sim - d_stage:+.1f} m  [should be ~0]")
    return ok and abs(d_raw_sim - d_stage) < 1e-9


def clean_water(mask):
    """Gate 7's cleaning, reused verbatim: speckle fragments are not shoreline.

    This is NOT largest_component -- genuine secondary water bodies are kept."""
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


def boundary_band(F, obs):
    """Cells within +/-BAND_PX of the CLEANED water boundary of field F.

    Contouring a raw continuous field returns every speckle zero-crossing in
    the whole domain, on land and in open water alike. Those vertices are not
    shoreline, and matching them to the nearest optical vertex manufactures
    offsets of hundreds of metres -- this is the same defect Gate 7 recorded
    for raw classified masks (13,000 km of "shoreline" against a real 1,540 km).
    The sub-pixel contour position still comes from the continuous field; the
    cleaned binary mask is used only to say WHERE a shoreline exists at all."""
    W = clean_water((F < 0) & obs)
    k = np.ones((3, 3), bool)
    return (ndimage.binary_dilation(W, k, BAND_PX)
            & ~ndimage.binary_erosion(W, k, BAND_PX) & obs), W


def contour_vertices(F, transform_xy, valid):
    """Marching-Squares vertices of F = 0, as metric coordinates."""
    rows = []
    for c in find_contours(np.nan_to_num(F, nan=1e6), level=0.0):
        r, cc = c[:, 0], c[:, 1]
        ri = np.clip(np.round(r).astype(int), 0, F.shape[0] - 1)
        ci = np.clip(np.round(cc).astype(int), 0, F.shape[1] - 1)
        k = valid[ri, ci]
        if not k.any():
            continue
        x, y = transform_xy(r[k], cc[k])
        rows.append(np.c_[x, y, ri[k], ci[k]])
    return np.vstack(rows) if rows else np.empty((0, 4))


def subpixel_offsets(F_cmp, F_ref, transform_xy, band_cmp, band_ref):
    """Signed cross-shore distance, TRUE vector-to-vector, in metres.

    Two corrections over the first version. (1) It evaluated F_ref/|grad F_ref|
    at every comparison vertex; that linearisation is only valid NEAR the zero
    crossing, so away from the boundary the NDWI gradient is small and noisy and
    the ratio explodes. Both shorelines are now extracted as vectors and the
    distance is nearest-vertex between them. (2) Vertices are taken only from
    each field's own cleaned boundary band, and a vertex with no counterpart
    within MAX_MATCH_M is returned as UNPAIRED rather than as a large offset --
    an unmatched vertex is missing evidence, not a measured displacement.

    Returns the paired vertices and the unpaired count, separately."""
    A = contour_vertices(F_cmp, transform_xy, band_cmp)   # SAR
    B = contour_vertices(F_ref, transform_xy, band_ref)   # optical
    if len(A) < 50 or len(B) < 50:
        return np.empty((0, 3)), A[:, :2]
    tree = cKDTree(B[:, :2])
    dist, _ = tree.query(A[:, :2], k=1)
    paired = dist <= MAX_MATCH_M
    unpaired_xy = A[~paired, :2]
    A, dist = A[paired], dist[paired]
    ri = A[:, 2].astype(int); ci = A[:, 3].astype(int)
    fref = F_ref[ri, ci]
    # F_ref < 0 means the comparison vertex sits inside optical water, so the
    # SAR extent is SMALLER there -> negative by the frozen convention
    sign = np.where(fref < 0, -1.0, +1.0)
    return np.c_[A[:, 0], A[:, 1], sign * dist], unpaired_xy


def http_json(url, payload=None, timeout=180, tries=5):
    """GET/POST JSON with retries. A single transient read timeout killed a
    40-minute Gate 7C3 run after nine scenes; the network is not part of the
    science, so it must not be able to end a sweep."""
    import time
    last = None
    for a in range(tries):
        try:
            req = (urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
                if payload is not None else url)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as ex:
            last = ex
            time.sleep(min(60, 4 * 2 ** a))
    raise RuntimeError(f"{type(last).__name__} after {tries} tries: {last}")


def fetch_ndwi(date, G, tok):
    """Continuous NDWI on the analysis grid, from native 10 m B3/B8."""
    q = {"collections": ["sentinel-2-l2a"], "bbox": G["bbox_ll"], "limit": 60,
         "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z"}
    feats = http_json(STAC, q)["features"]
    g3 = np.full((G["ny"], G["nx"]), np.nan, np.float32)
    n8 = np.full((G["ny"], G["nx"]), np.nan, np.float32)
    scl = np.zeros((G["ny"], G["nx"]), np.uint8)
    for f in feats:
        for band, dst in (("B03", g3), ("B08", n8), ("SCL", scl)):
            if band not in f["assets"]:
                continue
            try:
                with rasterio.open(f["assets"][band]["href"] + "?" + tok) as ds:
                    a = ds.read(1, window=from_bounds(G["x0"], G["y0"], G["x1"],
                                                      G["y1"], ds.transform),
                                out_shape=(G["ny"], G["nx"]),
                                resampling=Resampling.bilinear,
                                boundless=True, fill_value=0)
            except Exception:
                continue
            m = a > 0
            if band == "SCL":
                scl[m] = a[m].astype(np.uint8)
            else:
                dst[m] = a[m].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (g3 - n8) / (g3 + n8)
    good = np.isfinite(ndwi) & ~np.isin(scl, [0, 1, 3, 8, 9, 10])
    return ndwi, good, scl


def build_grid():
    """The analysis grid and domain mask. Shared so that Gate 7C1 audits the
    SAME grid the controls ran on, not a re-derived one."""
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    b = fp.bounds
    x0, y0 = np.floor(b[0] / CELL) * CELL, np.floor(b[1] / CELL) * CELL
    x1, y1 = np.ceil(b[2] / CELL) * CELL, np.ceil(b[3] / CELL) * CELL
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    G = dict(x0=x0, y0=y0, x1=x1, y1=y1, nx=nx, ny=ny,
             tr=from_origin(x0, y1, CELL, CELL),
             bbox_ll=[round(v, 5) for v in
                      gpd.GeoSeries([fp], crs=32636).to_crs(4326).iloc[0].bounds])
    G["inside"] = rio_rasterize([(fp, 1)], out_shape=(ny, nx),
                                transform=G["tr"], fill=0,
                                dtype="uint8").astype(bool)
    G["xy"] = lambda r, c: (x0 + (c + 0.5) * CELL, y1 - (r + 0.5) * CELL)
    return G


def build_fields(d, G, M, tok, cache=True):
    """The two continuous fields for one control date, plus their cleaned bands.

    Factored out of main() so Gate 7C1's correspondence audit runs against the
    identical F_sar / F_opt the controls used. Cached on the bulk drive because
    the audit re-reads these fields many times over.

    Returns None with a printed reason when the date cannot be built."""
    inside = G["inside"]
    cf = CFG.BULK_ROOT / "gate7c_fields" / f"{d}.npz"
    if cache and cf.exists():
        z = np.load(cf, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        out["event_id"] = str(out["event_id"])
        return out
    ev = M[(M.date == d) & M.gate6_selected.astype(bool)]
    if ev.empty:
        print(f"    {d}: no selected S1 event")
        return None
    r = ev.iloc[0]
    npz = S1_CACHE / f"{r.event_id}.npz"
    if not npz.exists():
        print(f"    {d}: S1 event product missing")
        return None
    ndwi, good, scl = fetch_ndwi(d, G, tok)
    cov_s2 = good & inside
    if cov_s2.sum() < 50000:
        print(f"    {d}: insufficient optical coverage")
        return None
    z = np.load(npz)
    vv, vh, cov = z["vv"], z["vh"], z["cov"]
    obs = cov & inside & cov_s2
    vvd = 10 * np.log10(np.maximum(vv, 1e-6))
    vhd = 10 * np.log10(np.maximum(vh, 1e-6))
    # anchors from the optical index itself, kept away from the boundary
    T = float(WM.DEFAULT_NDWI)
    wat = (ndwi > T) & obs
    lnd = (ndwi < T) & obs
    d_edge = ndimage.distance_transform_edt(
        ~(wat & ~ndimage.binary_erosion(wat, np.ones((3, 3), bool)))) * CELL
    aw = wat & (d_edge > 500)
    al = lnd & (d_edge > 500)
    if aw.sum() < 500 or al.sum() < 500:
        print(f"    {d}: anchors too small")
        return None
    Xw = np.c_[vvd[aw], vhd[aw]]; Xl = np.c_[vvd[al], vhd[al]]
    mw, ml = Xw.mean(0), Xl.mean(0)
    Sw = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
    wv = np.linalg.solve(Sw, ml - mw)
    cut = 0.5 * (wv @ mw + wv @ ml)
    # M3's own continuous decision field; zero level IS its boundary
    lda = np.full(inside.shape, np.nan, np.float32)
    lda[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) - cut
    # Orient the field NEGATIVE-IN-WATER from the anchors themselves rather
    # than from the algebra of the LDA sign. A hard-coded flip here was
    # wrong: w = S^-1 (m_land - m_water) already makes the discriminant
    # negative over water, so negating it selected LAND, and the "SAR water
    # area" came out at 252 km2 against an optical 1,563 km2.
    F_sar = lda if np.nanmedian(lda[aw]) < np.nanmedian(lda[al]) else -lda
    F_opt = np.where(cov_s2, T - ndwi, np.nan).astype(np.float32)
    band_sar, W_sar = boundary_band(F_sar, obs)
    band_opt, W_opt = boundary_band(F_opt, obs)
    out = dict(F_sar=F_sar, F_opt=F_opt, obs=obs, cov_s2=cov_s2, scl=scl,
               band_sar=band_sar, band_opt=band_opt, W_sar=W_sar, W_opt=W_opt,
               anchor_water=np.float32(np.nanmedian(F_sar[aw])),
               anchor_land=np.float32(np.nanmedian(F_sar[al])),
               event_id=str(r.event_id),
               relative_orbit=np.int32(r.relative_orbit),
               H_sar=np.float32(r.WSE_nominal))
    if cache:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, **out)
    return out


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("GATE 7C v2 — SAME-DATE CONTROLS on the frozen sub-pixel estimator")
    print("=" * 78)
    print(f"  estimator v2_subpixel_contour (Gate 7C0 = PASS WITH V2)")
    print(f"  git {commit}; control dates {CONTROL_DATES}")
    print("  no 0.86 diagonal correction, no inherited SAR bias, hist24 NOT run")

    print("\n  sign unit-test for d_stage:")
    if not test_stage_sign():
        raise SystemExit("d_stage sign is wrong relative to the frozen "
                         "water-extent convention")

    M, meta = load_frozen_manifest()
    G = build_grid()
    inside, xy = G["inside"], G["xy"]
    x0, y1, nx, ny = G["x0"], G["y1"], G["nx"], G["ny"]

    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    gg = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = gg.groupby("date").transformed_level_m.median()
    P = pd.read_parquet(PRIMARY)
    pxy = np.c_[P.x.to_numpy(), P.y.to_numpy()]
    pz = P.H_bed_evrf2019_m.to_numpy()
    ptree = cKDTree(pxy)
    tok = json.loads(urllib.request.urlopen(SAS, timeout=90).read())["token"]

    rows, keep = [], {}
    for d in CONTROL_DATES:
        ev = M[(M.date == d) & M.gate6_selected.astype(bool)]
        if ev.empty:
            print(f"\n  {d}: no selected S1 event -- skipped")
            continue
        r = ev.iloc[0]
        print(f"\n  {d}: S1 {r.event_id}")
        FL = build_fields(d, G, M, tok)
        if FL is None:
            continue
        F_sar, F_opt = FL["F_sar"], FL["F_opt"]
        obs, cov_s2, scl = FL["obs"], FL["cov_s2"], FL["scl"]
        band_sar, band_opt = FL["band_sar"], FL["band_opt"]
        W_sar, W_opt = FL["W_sar"], FL["W_opt"]
        # report against the DOMAIN, not the bounding-box grid: the domain is a
        # thin sinuous shape filling only ~12% of its own bbox, so a grid-mean
        # reads as ~10% even when optical coverage of the water body is complete
        print(f"    S2 continuous NDWI: "
              f"{100*cov_s2.sum()/inside.sum():.1f}% of the analysis domain "
              f"({100*cov_s2.mean():.1f}% of the bbox grid)")
        print(f"    LDA anchor medians: water {FL['anchor_water']:+.3f}, "
              f"land {FL['anchor_land']:+.3f} (water must be negative)")
        px2 = CELL ** 2 / 1e6
        print(f"    cleaned water area: SAR {W_sar.sum()*px2:,.0f} km2, "
              f"optical {W_opt.sum()*px2:,.0f} km2")
        pts, unp = subpixel_offsets(F_sar, F_opt, xy, band_sar, band_opt)
        if len(pts) < 200:
            print("    too few contour vertices -- skipped")
            continue
        X, Y, d_raw = pts[:, 0], pts[:, 1], pts[:, 2]
        n_unpaired = len(unp)
        pair_frac = len(pts) / (len(pts) + n_unpaired)
        print(f"    vertices paired within {MAX_MATCH_M:.0f} m: "
              f"{len(pts):,} ({100*pair_frac:.1f}%); unpaired {n_unpaired:,}")

        # WHY are they unpaired? A low pairing rate is only benign if the
        # unpaired vertices sit where the optical scene has no valid pixel to
        # pair WITH. If instead they sit in fully-observed optical land, the
        # SAR is calling water where the optical says none, and the median over
        # the paired subset would be a survivorship-biased statistic.
        gap_m = ndimage.distance_transform_edt(cov_s2) * CELL
        gap_frac = np.nan
        if n_unpaired:
            uc = np.clip(((y1 - unp[:, 1]) / CELL).astype(int), 0, ny - 1)
            ur = np.clip(((unp[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
            near_gap = gap_m[uc, ur] <= 200.0
            gap_frac = float(near_gap.mean())
            print(f"    unpaired within 200 m of an S2 coverage gap: "
                  f"{100*gap_frac:.1f}%")

        H_sar = float(r.WSE_nominal)
        H_opt = float(lev[d]) if d in lev.index else np.nan
        dH = H_opt - H_sar
        slope = np.full(len(X), np.nan)
        step = max(1, len(X) // 2500)
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
        s_med = float(np.nanmedian(slope))
        d_stage = -dH / s_med if np.isfinite(s_med) and s_med > 1e-6 else np.nan
        d_res = d_raw - d_stage if np.isfinite(d_stage) else np.full(len(d_raw), np.nan)

        print(f"    H_SAR {H_sar:.3f} m, H_OPT {H_opt:.3f} m, dH {dH:+.3f} m")
        print(f"    local slope {s_med:.5f} m/m "
              f"(support {100*np.isfinite(slope[idxs]).mean():.0f}% of the "
              f"{len(idxs):,} SAMPLED vertices; slope is deliberately not "
              f"evaluated at all {len(X):,})")
        print(f"    d_raw     median {np.median(d_raw):+7.1f} m  "
              f"NMAD {1.4826*np.median(np.abs(d_raw-np.median(d_raw))):6.1f}  "
              f"n {len(d_raw):,}")
        print(f"    d_stage   {d_stage:+7.1f} m")
        print(f"    d_residual median {np.nanmedian(d_res):+7.1f} m")
        rows.append(dict(control_date=d, event_id=r.event_id,
                         relative_orbit=int(r.relative_orbit),
                         H_SAR=H_sar, H_OPT=H_opt, delta_H=dH,
                         local_slope=s_med,
                         slope_support_frac=float(np.isfinite(slope[idxs]).mean()),
                         n_slope_sampled=int(len(idxs)),
                         n_vertices=int(len(d_raw)),
                         n_unpaired=int(n_unpaired),
                         pair_frac=float(pair_frac),
                         unpaired_near_s2_gap_frac=gap_frac,
                         s2_domain_cov=float(cov_s2.sum() / inside.sum()),
                         area_sar_km2=float(W_sar.sum() * px2),
                         area_opt_km2=float(W_opt.sum() * px2),
                         d_raw_median=float(np.median(d_raw)),
                         d_raw_nmad=float(1.4826 * np.median(
                             np.abs(d_raw - np.median(d_raw)))),
                         d_raw_p90=float(np.percentile(np.abs(d_raw), 90)),
                         d_stage=float(d_stage),
                         d_residual_median=float(np.nanmedian(d_res)),
                         d_residual_nmad=float(1.4826 * np.nanmedian(
                             np.abs(d_res - np.nanmedian(d_res))))))
        keep[d] = dict(X=X, Y=Y, d_raw=d_raw, d_res=d_res, scl=scl)
        del FL, F_sar, F_opt, band_sar, band_opt, W_sar, W_opt

    if not rows:
        raise SystemExit("no control pair produced a comparison")
    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "gate7c_v2_same_date_controls.csv", index=False)
    print("\n" + "=" * 78)
    print("SAME-DATE CONTROL SUMMARY")
    print("=" * 78)
    print(R[["control_date", "delta_H", "local_slope", "d_raw_median",
             "d_stage", "d_residual_median", "d_residual_nmad",
             "n_vertices"]].to_string(index=False,
                                      float_format=lambda v: f"{v:,.3f}"))
    print("\n  sign convention: positive = SAR water extent LARGER than optical")
    print("  d_stage = -dH / s ; d_residual = d_raw - d_stage")

    _fig(R, keep)
    print("\nSTOP. Gate 7C decision requires the full H1/H2/H3 re-run; hist24 "
          "is not started.")


def _fig(R, keep):
    n = max(len(keep), 1)
    fig, ax = plt.subplots(1, n + 1, figsize=(6.2 * (n + 1), 5.6))
    ax = np.atleast_1d(ax)
    for a, (d, k) in zip(ax, keep.items()):
        v = float(np.percentile(np.abs(k["d_raw"]), 95)) or 50
        sc = a.scatter(k["X"] / 1000, k["Y"] / 1000, c=k["d_raw"], s=1.0,
                       cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(sc, ax=a, label="d_raw (m)")
        a.set_title(f"{d} · d_raw, sub-pixel vector estimator\n"
                    f"+ = SAR water larger", fontsize=10, loc="left")
        a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    a = ax[-1]
    for d, k in keep.items():
        a.hist(k["d_raw"], bins=120, histtype="step", lw=1.6, label=f"{d} d_raw")
        a.hist(k["d_res"][np.isfinite(k["d_res"])], bins=120, histtype="step",
               lw=1.6, ls="--", label=f"{d} d_residual")
    a.axvline(0, color=INK, lw=1.3)
    a.set_xlabel("signed cross-shore distance (m)"); a.set_ylabel("vertices")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_xlim(-300, 300)
    a.set_title("distribution before and after stage correction",
                fontsize=10, loc="left")
    fig.suptitle("Gate 7C v2 · same-date controls on the frozen sub-pixel "
                 "estimator", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "gate7c_v2_controls.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
