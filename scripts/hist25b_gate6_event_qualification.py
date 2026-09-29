#!/usr/bin/env python
"""HIST25B GATE 6 — event fetch, mosaicking and radiometric qualification.

No classification happens here. Gate 6 turns the Gate-5 inventory into a
frozen, quality-labelled population of OBSERVATION EVENTS that a classifier
may later use.

WHAT CHANGED FROM GATE 5, AND WHY
---------------------------------
1. 12.204 m is NOT a gauge floor. Rozumivka's accepted reference is
   H0 = 12.000 m and the EPSG:9902 correction is ~ +0.203588 m, so a
   transformed value near 12.204 m is numerically a recorded stage near
   ZERO. Gauge zero is a vertical reference only -- not the bed, not a
   physical lower limit, and not evidence the station cannot represent
   lower water. The real problem is semantic: after the 19-day gap the
   record returns at exactly ~12.204 m and stays constant. Those events are
   therefore flagged zero_stage_plateau_flag / wse_semantics_resolved=False
   and kept in the archive, but they may not act as shoreline-elevation
   constraints. No gauge_floor_flag is created, and nothing here asserts
   the true post-breach WSE was 12.204 m, nor that it could not be lower.

2. Chronology is not hydraulic state. The 895-day POSTBREACH_TRANSITION
   bucket is gone. TEMPORAL regime now has four classes only
   (PREBREACH_IMPOUNDED / BREACH_ONSET / ACTIVE_DRAWDOWN /
   POSTBREACH_UNRESOLVED). The SPATIAL hydraulic state (RIVER_CHANNEL,
   RESIDUAL_WATER, EXPOSED_BED, FLOODED_VEGETATION, BACKWATER) is a
   separate axis, left UNRESOLVED here because it must come from spatial
   evidence, not from the Rozumivka hydrograph.

3. Intra-day WSE uncertainty is explicit. The gauge series is daily with
   every timestamp 00:00:00, so obs_time_known = FALSE. Two uncertainty
   models are carried, both consequences of unknown observation time rather
   than measurements of error:
       u_intraday_nominal      = 0.5 * |dH_daily|
       u_intraday_conservative = max(|dH_prev_day|, |dH_next_day|)
   Pre-breach this is negligible; during ACTIVE_DRAWDOWN it dominates.
   2023-06-08 (dH/dt ~ -0.75 m/day) keeps excellent shoreline GEOMETRY
   while its ELEVATION LABEL becomes substantially less certain. Good
   geometry != precise WSE label, and that is not grounds to drop a scene.

4. Raw stage proximity is separated from eligibility. RAW_STAGE_PROXIMITY
   answers only "how close was nominal WSE to H1/H2/H3". ELIGIBLE_TARGET_
   STAGE_GEOMETRY additionally requires PREBREACH_IMPOUNDED, resolved WSE,
   and resolved WSE semantics. An ACTIVE_DRAWDOWN scene is never promoted
   to target-stage geometry just because its instantaneous level is
   numerically close: LEVEL proximity != HYDRAULIC STATE equivalence.

5. The unit of analysis is the observation event. 2,655 STAC assets form
   ~1,416 physical overpasses. Assets sharing sat:absolute_orbit are slices
   of one pass -- verified here by checking the acquisition time span is
   minutes, not hours -- and are mosaicked into ONE event product before
   any statistic is computed.

INCIDENCE ANGLE IS NOT AVAILABLE. The Planetary Computer sentinel-1-rtc
item exposes only vv, vh, tilejson and rendered_preview. There is no
incidence-angle band and no per-pixel angle metadata, so
incidence_angle_median/iqr are emitted as NaN with a recorded reason rather
than approximated from orbit geometry.

Outputs
-------
outputs/tables/hist25b_raw_stage_proximity.csv
outputs/tables/hist25b_eligible_target_stage_geometry.csv
outputs/tables/hist25b_event_grouping_verification.csv
outputs/tables/hist25b_event_radiometric_qa.csv
outputs/tables/hist25b_pass2_topup_strata.csv
outputs/tables/hist25b_gate6_event_manifest.csv
outputs/tables/hist25b_gate6_manifest_hash.json
outputs/figures/historical_bathymetry/png/hist25b_gate6_*.png
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")

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
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CELL = 20.0
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
BBOX = [32.2, 46.5, 35.5, 48.0]   # discovery only; overlap is
                                  # screened against the real domain
# Event mosaics are bulk data and live on drive F. The corrected domain uses
# a different grid from the P20 one, so the old cache cannot be reused.
CACHE = CFG.S1_CACHE / "ZONE_1_reservoir_corrected"
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"

BREACH = pd.Timestamp("2023-06-06")
MARGIN_SLOPE = 0.03
SIG_STATIC = float(np.sqrt(0.068 ** 2 + 0.010 ** 2 + 0.175 ** 2))   # 0.188 m
MAX_GAUGE_GAP_DAYS = 3
ANOMALY_RATE = 0.20
CUM_LOSS_MIN = 0.50
OBS_TIME_KNOWN = False

# Rozumivka vertical reference: gauge zero, NOT a bed or a measurement floor
ROZUMIVKA_H0 = 12.000
EPSG9902_CORRECTION = 0.203588
ZERO_STAGE_WSE = ROZUMIVKA_H0 + EPSG9902_CORRECTION          # ~12.2036 m
ZERO_STAGE_TOL = 0.02

STAGE_TIERS = (("STAGE_TIER_010", 0.10), ("STAGE_TIER_020", 0.20),
               ("STAGE_TIER_030", 0.30))
MAX_EVENTS_PER_CELL = 2        # per stage tier x regime x relative orbit;
                               # Pass 2 tops up strata that QA thins out
BORDER_DIST_M = 100.0
BORDER_DARK_DB = -28.0
STABLE_REF_MIN_M, STABLE_REF_MAX_M = 800.0, 3000.0


MANIFEST_CSV = CFG.TABLES / "hist25b_gate6_event_manifest.csv"
MANIFEST_HASH = CFG.TABLES / "hist25b_gate6_manifest_hash.json"


def load_frozen_manifest(selected_only=True, verify=True):
    """THE single entry point for every downstream gate.

    Gate 7 and later must obtain their event population from here and never
    from an intermediate table, so the classifier's input is provably the
    population Gate 6 froze. The sha256 recorded at freeze time is re-checked
    on every load; a mismatch raises rather than silently proceeding.
    """
    meta = json.loads(MANIFEST_HASH.read_text())
    if verify:
        h = hashlib.sha256(MANIFEST_CSV.read_bytes()).hexdigest()
        if h != meta["sha256"]:
            raise RuntimeError(
                f"Gate-6 manifest has changed since it was frozen.\n"
                f"  recorded   {meta['sha256']}\n  recomputed {h}\n"
                f"Re-run Gate 6 rather than editing the manifest in place.")
    M = pd.read_csv(MANIFEST_CSV)
    if selected_only:
        M = M[M.gate6_selected.astype(bool)].reset_index(drop=True)
    return M, meta


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-6))


def sas_token():
    return json.loads(urllib.request.urlopen(SAS, timeout=90).read())["token"]


def stac_day(date):
    q = {"collections": ["sentinel-1-rtc"], "bbox": BBOX, "limit": 500,
         "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z"}
    r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())["features"]


def build_grid(fp):
    x0 = np.floor(fp.bounds[0] / CELL) * CELL
    y0 = np.floor(fp.bounds[1] / CELL) * CELL
    x1 = np.ceil(fp.bounds[2] / CELL) * CELL
    y1 = np.ceil(fp.bounds[3] / CELL) * CELL
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    tr = from_origin(x0, y1, CELL, CELL)
    inside = rio_rasterize([(fp, 1)], out_shape=(ny, nx), transform=tr,
                           fill=0, dtype="uint8").astype(bool)
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, nx=nx, ny=ny, tr=tr, inside=inside)


def gauge_series():
    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    g = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = g.groupby("date").transformed_level_m.median().sort_index()
    lev.index = pd.to_datetime(lev.index)
    return lev.sort_index()


def temporal_regime(lev):
    """Four TEMPORAL classes only. Spatial hydraulic state is a separate
    axis and is not inferred from the hydrograph."""
    reg = pd.Series("POSTBREACH_UNRESOLVED", index=lev.index, dtype=object)
    d = lev.diff()
    gaps = lev.index.to_series().diff().dt.days.fillna(1)
    reg[lev.index < BREACH] = "PREBREACH_IMPOUNDED"
    if BREACH in reg.index:
        reg[BREACH] = "BREACH_ONSET"
    cum, stopped = 0.0, False
    for t in lev.index[lev.index > BREACH]:
        if gaps[t] > MAX_GAUGE_GAP_DAYS:
            stopped = True
            continue
        rate = d.get(t, np.nan)
        if not stopped and np.isfinite(rate) and rate < -ANOMALY_RATE:
            cum += -rate
            reg[t] = "ACTIVE_DRAWDOWN"
    return reg


def wse_fields(ts, lev, reg):
    """Attach WSE with an explicit method and two intra-day uncertainty
    models. Returns WSE_UNRESOLVED rather than inventing precision."""
    day = pd.Timestamp(ts.date())
    out = dict(obs_time_known=OBS_TIME_KNOWN,
               gauge_match_method="WSE_UNRESOLVED", WSE_nominal=np.nan,
               WSE_uncertainty_static=SIG_STATIC,
               WSE_uncertainty_intraday_nominal=np.nan,
               WSE_uncertainty_intraday_conservative=np.nan,
               WSE_uncertainty_total_nominal=np.nan,
               WSE_uncertainty_total_conservative=np.nan,
               WSE_RESOLVED=False, wse_interpolated=False,
               gauge_time_before_days=np.nan, gauge_time_after_days=np.nan,
               max_gap_hours=np.nan, dHdt_m_per_day=np.nan,
               zero_stage_plateau_flag=False, wse_semantics_resolved=False,
               wse_semantics_note="")
    before = lev.index[lev.index <= day]
    after = lev.index[lev.index >= day]
    if not len(before) or not len(after):
        out["wse_semantics_note"] = "No bracketing daily gauge observation."
        return out
    lo, hi = before[-1], after[0]
    out["gauge_time_before_days"] = (day - lo).days
    out["gauge_time_after_days"] = (hi - day).days
    gap = (hi - lo).days
    out["max_gap_hours"] = float(max(gap, 0) * 24)

    idx = list(lev.index)
    pos = idx.index(lo)
    d_prev = abs(lev.iloc[pos] - lev.iloc[pos - 1]) if pos > 0 else np.nan
    d_next = abs(lev.iloc[pos + 1] - lev.iloc[pos]) if pos + 1 < len(idx) else np.nan
    d_daily = np.nanmax([d_prev, d_next]) if np.isfinite([d_prev, d_next]).any() else 0.0
    u_nom = 0.5 * (d_prev if np.isfinite(d_prev) else d_daily)
    u_con = np.nanmax([d_prev, d_next]) if np.isfinite([d_prev, d_next]).any() else np.nan
    out["dHdt_m_per_day"] = float(d_daily)
    out["WSE_uncertainty_intraday_nominal"] = float(u_nom)
    out["WSE_uncertainty_intraday_conservative"] = float(u_con)

    if day in lev.index:
        out.update(gauge_match_method="daily_observation_same_date",
                   WSE_nominal=float(lev[day]), WSE_RESOLVED=True)
    elif gap <= MAX_GAUGE_GAP_DAYS:
        f = (day - lo).days / gap
        out.update(gauge_match_method="interpolated_between_daily",
                   WSE_nominal=float(lev[lo] + f * (lev[hi] - lev[lo])),
                   WSE_RESOLVED=True, wse_interpolated=True)
    else:
        out["wse_semantics_note"] = (
            f"Gauge gap of {gap} days exceeds {MAX_GAUGE_GAP_DAYS}; "
            "WSE not resolvable.")
        return out

    out["WSE_uncertainty_total_nominal"] = float(
        np.sqrt(SIG_STATIC ** 2 + np.nan_to_num(u_nom) ** 2))
    out["WSE_uncertainty_total_conservative"] = float(
        np.sqrt(SIG_STATIC ** 2 + np.nan_to_num(u_con) ** 2))

    # zero-stage plateau: a recorded stage near zero, repeated. This is a
    # SEMANTIC problem, not a physical floor.
    if abs(out["WSE_nominal"] - ZERO_STAGE_WSE) < ZERO_STAGE_TOL:
        out["zero_stage_plateau_flag"] = True
        out["wse_semantics_resolved"] = False
        out["wse_semantics_note"] = (
            "Repeated stage near zero after breach; physical meaning "
            "unresolved. Gauge zero is a vertical reference, not a bed "
            "elevation or a lower measurement limit.")
    else:
        out["wse_semantics_resolved"] = True
    return out


def spatial_prescreen(years, fp_utm):
    """Predict each event's footprint overlap from STAC geometry, BEFORE any
    download. Spatial eligibility is an a-priori property of orbit geometry,
    not something to be discovered by fetching pixels -- and it is NOT a
    radiometric property, so it must not enter the radiometric denominator.

    Gate 5 inventoried via a STAC bbox far larger than the reservoir, so an
    acquisition can intersect that bbox while missing the AOI completely.
    Without this screen, Pass-1 strata spend slots on orbits that physically
    cannot see the analysis footprint.
    """
    # Keyed by the domain: the prescreen measures overlap against a specific
    # geometry, so a cache written for a different footprint must not be
    # reused. The corrected domain adds 87 km2 on the east, which changes the
    # overlap of exactly the orbits that reach it.
    cache = CFG.TABLES / (f"hist25b_event_spatial_prescreen_"
                          f"{SD.geom_hash(SD.load('dnipro_water_domain'))}.csv")
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for year in years:
        for q0, q1 in (("01-01", "03-31"), ("04-01", "06-30"),
                       ("07-01", "09-30"), ("10-01", "12-31")):
            feats = stac_range(f"{year}-{q0}", f"{year}-{q1}")
            for f in feats:
                p = f["properties"]
                if p.get("sat:relative_orbit") is None or not f.get("geometry"):
                    continue
                rows.append(dict(date=p["datetime"][:10],
                                 relative_orbit=int(p["sat:relative_orbit"]),
                                 orbit_state=p.get("sat:orbit_state", "UNKNOWN"),
                                 geometry=shp_shape(f["geometry"])))
            print(f"    prescreen {year} {q0[:2]}-{q1[:2]}: {len(feats)} items",
                  flush=True)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(32636)
    out = []
    for (d, orb, st), g in gdf.groupby(["date", "relative_orbit", "orbit_state"]):
        u = g.geometry.union_all() if hasattr(g.geometry, "union_all") \
            else g.geometry.unary_union
        frac = float(u.intersection(fp_utm).area / fp_utm.area)
        out.append(dict(date=d, relative_orbit=orb, orbit_state=st,
                        predicted_overlap_fraction=frac))
    P = pd.DataFrame(out)
    P.to_csv(cache, index=False)
    return P


def stac_range(d0, d1):
    out, token = [], None
    while True:
        q = {"collections": ["sentinel-1-rtc"], "bbox": BBOX, "limit": 500,
             "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z"}
        if token:
            q["token"] = token
        r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                   headers={"Content-Type": "application/json"})
        j = json.loads(urllib.request.urlopen(r, timeout=180).read())
        out += j["features"]
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        token = nxt[0].get("body", {}).get("token")
        if not token:
            break
    return out


def read_asset(item, tok, G, pol):
    with rasterio.open(item["assets"][pol]["href"] + "?" + tok) as ds:
        arr = ds.read(1, window=from_bounds(G["x0"], G["y0"], G["x1"], G["y1"],
                                            ds.transform),
                      out_shape=(G["ny"], G["nx"]),
                      resampling=Resampling.average,
                      boundless=True, fill_value=np.nan)
    return arr.astype(np.float32)


def build_event(event_id, items, tok, G):
    """Mosaic every asset of one physical overpass into ONE event product,
    measuring the calibration discrepancy across mosaic seams on the way."""
    ny, nx = G["ny"], G["nx"]
    npz = CACHE / f"{event_id}.npz"
    if npz.exists():
        z = np.load(npz)
        # A CACHED PRODUCT IS ONLY VALID ON THE GRID IT WAS FETCHED ON.
        # Repairing ZONE_2's geometry (p0x removed two polygons that belonged
        # to the Southern Bug) changed its grid from 5825 to 3830 rows, and the
        # next script to read the cache died on a broadcast error. A shape
        # mismatch means the domain moved under the cache, so the cache is
        # stale by definition and is rebuilt rather than reused -- silently
        # reusing it, or cropping it to fit, would put pixels in the wrong
        # place with no error at all.
        if z["vv"].shape != (ny, nx):
            print(f"      {event_id}: cached on {z['vv'].shape}, grid is "
                  f"{(ny, nx)} -- domain changed, refetching", flush=True)
            del z
            npz.unlink()
        else:
            return z["vv"], z["vh"], z["cov"], float(z["seam"])
    vv = np.full((ny, nx), np.nan, np.float32)
    vh = np.full((ny, nx), np.nan, np.float32)
    seams = []
    for it in sorted(items, key=lambda f: f["properties"]["datetime"]):
        t0 = time.time()
        try:
            a_vv = read_asset(it, tok, G, "vv")
            a_vh = read_asset(it, tok, G, "vh")
        except Exception as ex:
            print(f"        {it['id'][:46]} READ FAIL {type(ex).__name__}")
            continue
        m = np.isfinite(a_vv) & (a_vv > 0) & np.isfinite(a_vh) & (a_vh > 0)
        overlap = m & np.isfinite(vv)
        if overlap.sum() > 5000:
            seams.append(float(np.median(np.abs(db(a_vv[overlap]) -
                                                db(vv[overlap])))))
        new = m & ~np.isfinite(vv)
        vv[new] = a_vv[new]
        vh[new] = a_vh[new]
        print(f"        {it['id'][:46]} [{time.time()-t0:.0f}s]")
        del a_vv, a_vh, m, overlap, new
    cov = np.isfinite(vv) & np.isfinite(vh)
    seam = float(np.median(seams)) if seams else np.nan
    np.savez_compressed(npz, vv=vv, vh=vh, cov=cov, seam=seam)
    return vv, vh, cov, seam


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    # Extent from the registry, never from a derived geojson. The reservoir
    # water is the SA_2-bounded core plus the corrected water just outside
    # that envelope; P20 truncated 87.4 km2 of it on the east.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    lev = gauge_series()
    reg_daily = temporal_regime(lev)

    inv3 = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv")
    inv3 = inv3.sort_values("H_evrf2019_m").reset_index(drop=True)
    targets = {r.contour_id: float(r.H_evrf2019_m) for r in inv3.itertuples()}

    print("=" * 78)
    print("STEP 1-2 — GATE-5 ASSETS, WSE SEMANTICS CORRECTED")
    print("=" * 78)
    A = pd.read_csv(CFG.TABLES / "hist25b_s1_asset_inventory.csv")
    print(f"  {len(A)} raw STAC assets (frozen from Gate 5)")
    print(f"  Rozumivka H0 = {ROZUMIVKA_H0:.3f} m, EPSG:9902 correction "
          f"+{EPSG9902_CORRECTION:.6f} m -> zero stage ~ {ZERO_STAGE_WSE:.4f} m")
    print("  12.204 m is a recorded stage near ZERO, NOT a gauge floor.")
    print("  No gauge_floor_flag is created anywhere in this script.")

    # ---------------------------------------------- STEP 7: event grouping
    print("\n" + "=" * 78)
    print("STEP 7 — VERIFY THAT GROUPED ASSETS ARE ONE PHYSICAL OVERPASS")
    print("=" * 78)
    A["sensing_dt"] = pd.to_datetime(A.sensing_datetime)
    grp = A.groupby(["date", "relative_orbit", "orbit_state"])
    ev = grp.agg(asset_count=("asset_id", "size"),
                 acquisition_start=("sensing_dt", "min"),
                 acquisition_end=("sensing_dt", "max"),
                 platform=("platform", "first"),
                 polarisations=("polarisations", "first")).reset_index()
    ev["acquisition_time_span_minutes"] = (
        (ev.acquisition_end - ev.acquisition_start).dt.total_seconds() / 60.0)
    ev["event_id"] = (ev.date + "_orb" + ev.relative_orbit.astype(str) + "_"
                      + ev.orbit_state.str[:3].str.upper())
    span = ev.acquisition_time_span_minutes
    print(f"  {len(A)} assets -> {len(ev)} events "
          f"({len(A)/len(ev):.2f} assets per event)")
    print(f"  acquisition time span within an event: median {span.median():.2f} min, "
          f"p95 {span.quantile(.95):.2f} min, max {span.max():.2f} min")
    bad = ev[span > 30]
    print(f"  events whose assets span > 30 min (would NOT be one pass): {len(bad)}")
    ev["single_pass_verified"] = span <= 30
    ev.to_csv(CFG.TABLES / "hist25b_event_grouping_verification.csv", index=False)

    # ---------------------------------------------- WSE + regime per event
    wf = pd.DataFrame([wse_fields(t, lev, reg_daily)
                       for t in pd.to_datetime(ev.date)])
    E = pd.concat([ev.reset_index(drop=True), wf], axis=1)
    E["temporal_regime"] = [reg_daily.get(pd.Timestamp(d), "POSTBREACH_UNRESOLVED")
                            for d in pd.to_datetime(E.date)]
    E["spatial_hydraulic_state"] = "UNRESOLVED"
    E["spatial_state_note"] = ("Requires spatial evidence (S1/S2 extent, "
                               "channel connectivity, ICESat-2 longitudinal "
                               "WSE); not inferable from the gauge.")
    for c, t in targets.items():
        E[f"dH_{c}_m"] = E.WSE_nominal - t
    print("\n  temporal regimes across events:")
    print(E.temporal_regime.value_counts().to_string())
    print(f"\n  WSE_RESOLVED: {int(E.WSE_RESOLVED.sum())} / {len(E)}")
    print(f"  zero_stage_plateau_flag: {int(E.zero_stage_plateau_flag.sum())}")
    print(f"  wse_semantics_resolved: {int(E.wse_semantics_resolved.sum())}")

    # ---------------------------------------------- STEP 3-4: two tables
    print("\n" + "=" * 78)
    print("STEP 3 — RAW_STAGE_PROXIMITY (nominal level only, no eligibility)")
    print("=" * 78)
    raw_rows, elig_rows = [], []
    for c, t in targets.items():
        for tier, tol in STAGE_TIERS:
            near = E[E.WSE_nominal.notna() & ((E.WSE_nominal - t).abs() <= tol)]
            raw_rows.append(dict(contour_id=c, target_wse_m=t, stage_tier=tier,
                                 tolerance_m=tol,
                                 horizontal_error_m=tol / MARGIN_SLOPE,
                                 events=len(near),
                                 n_orbits=near.relative_orbit.nunique(),
                                 orbits="|".join(map(str, sorted(
                                     int(o) for o in near.relative_orbit.unique())))))
            el = near[(near.temporal_regime == "PREBREACH_IMPOUNDED")
                      & near.WSE_RESOLVED & near.wse_semantics_resolved]
            elig_rows.append(dict(contour_id=c, target_wse_m=t, stage_tier=tier,
                                  tolerance_m=tol,
                                  horizontal_error_m=tol / MARGIN_SLOPE,
                                  eligible_events=len(el),
                                  n_orbits=el.relative_orbit.nunique(),
                                  orbits="|".join(map(str, sorted(
                                      int(o) for o in el.relative_orbit.unique()))),
                                  excluded_by_regime=len(near) - len(el)))
            print(f"  {c} {tier} (|dH|<={tol:.2f}): raw {len(near):3d} events, "
                  f"eligible {len(el):3d}  (excluded by regime/semantics: "
                  f"{len(near)-len(el)})")
    RAW = pd.DataFrame(raw_rows)
    ELG = pd.DataFrame(elig_rows)
    RAW.to_csv(CFG.TABLES / "hist25b_raw_stage_proximity.csv", index=False)
    ELG.to_csv(CFG.TABLES / "hist25b_eligible_target_stage_geometry.csv",
               index=False)

    # ---------------------------------------------- roles
    def roles_for(r):
        out = []
        tr = r.temporal_regime
        if tr == "PREBREACH_IMPOUNDED":
            out += ["PREBREACH_REFERENCE", "RADIOMETRIC_CALIBRATION"]
        elif tr == "BREACH_ONSET":
            out += ["BREACH_ONSET_OBSERVATION", "HYDRAULIC_TRANSITION",
                    "SHORELINE_STATE", "TEMPORAL_CHANGE_REFERENCE"]
        elif tr == "ACTIVE_DRAWDOWN":
            out += ["HYDRAULIC_TRANSITION", "CHANNEL_EXPOSURE",
                    "SHORELINE_STATE", "FLOODED_VEGETATION_DIAGNOSTIC",
                    "FUTURE_HYDRAULIC_MODEL_VALIDATION",
                    "TEMPORAL_CHANGE_REFERENCE"]
        else:
            out += ["POSTBREACH_UNRESOLVED", "SHORELINE_STATE",
                    "TEMPORAL_CHANGE_REFERENCE"]
        if not r.WSE_RESOLVED:
            out.append("WSE_UNRESOLVED")
        if r.zero_stage_plateau_flag:
            out.append("WSE_SEMANTICS_UNRESOLVED")
        if (tr == "PREBREACH_IMPOUNDED" and r.WSE_RESOLVED
                and r.wse_semantics_resolved):
            for c, t in targets.items():
                for tier, tol in STAGE_TIERS:
                    if abs(r.WSE_nominal - t) <= tol:
                        out.append(f"{c}_TARGET_STAGE_GEOMETRY_{tier[-3:]}")
                        break
        out.append("CLASSIFIER_DIAGNOSTIC")
        return "|".join(dict.fromkeys(out))
    E["scientific_roles"] = [roles_for(r) for r in E.itertuples()]

    # ---------------------------------------------- spatial pre-screen
    print("\n" + "=" * 78)
    print("SPATIAL PRE-SCREEN — eligibility is known BEFORE download")
    print("=" * 78)
    P = spatial_prescreen(sorted({int(d[:4]) for d in E.date}), fp)
    E = E.merge(P, on=["date", "relative_orbit", "orbit_state"], how="left")
    E["predicted_overlap_fraction"] = E.predicted_overlap_fraction.fillna(0.0)
    E["spatial_status_predicted"] = np.where(
        E.predicted_overlap_fraction < 0.001, "NO_OVERLAP",
        np.where(E.predicted_overlap_fraction < 0.95, "PARTIAL_OVERLAP",
                 "VALID_OVERLAP"))
    print(E.spatial_status_predicted.value_counts().to_string())
    print("\n  predicted overlap by relative orbit (median):")
    ob = (E.groupby("relative_orbit")
            .agg(events=("event_id", "size"),
                 median_overlap=("predicted_overlap_fraction", "median"),
                 no_overlap=("spatial_status_predicted",
                             lambda s: int((s == "NO_OVERLAP").sum())))
            .reset_index())
    print(ob.to_string(index=False))
    print("\n  NO_OVERLAP is a SPATIAL property of orbit geometry, not a SAR")
    print("  defect; such events never enter Pass-1 and are excluded from the")
    print("  radiometric denominator (radiometric_quality = NOT_EVALUATED).")

    # ---------------------------------------------- STEP 5: Pass-1 strata
    print("\n" + "=" * 78)
    print("STEP 5 — PASS 1 SELECTION: stage tier x temporal regime x orbit")
    print("        (radiometric quality is NOT an a-priori stratum)")
    print("=" * 78)
    E_sel = E[E.spatial_status_predicted != "NO_OVERLAP"]
    sel = []
    for c, t in targets.items():
        for tier, tol in STAGE_TIERS:
            el = E_sel[(E_sel.temporal_regime == "PREBREACH_IMPOUNDED")
                   & E_sel.WSE_RESOLVED & E_sel.wse_semantics_resolved
                   & ((E_sel.WSE_nominal - t).abs() <= tol)].copy()
            if el.empty:
                continue
            el["abs_dH"] = (el.WSE_nominal - t).abs()
            pick = (el.sort_values(["relative_orbit", "abs_dH"])
                      .groupby("relative_orbit").head(MAX_EVENTS_PER_CELL).copy())
            pick["stratum"] = f"{c}_{tier}"
            pick["select_reason"] = f"{c}_target_stage_{tier}"
            sel.append(pick)
    for tr in ("BREACH_ONSET", "ACTIVE_DRAWDOWN"):
        s = E_sel[E_sel.temporal_regime == tr].copy()
        if len(s):
            s["stratum"] = tr
            s["select_reason"] = f"{tr.lower()}_branch"
            sel.append(s)
    pu = E_sel[E_sel.temporal_regime == "POSTBREACH_UNRESOLVED"].copy()
    if len(pu):
        pu = pu.sort_values("date").groupby("relative_orbit").head(1).copy()
        pu["stratum"] = "POSTBREACH_UNRESOLVED"
        pu["select_reason"] = "postbreach_state_reference"
        sel.append(pu)
    SEL = (pd.concat(sel).drop_duplicates("event_id").reset_index(drop=True))
    print(SEL.groupby("select_reason").size().to_string())
    print(f"\n  Pass-1 events to fetch: {len(SEL)} of {len(E)} inventoried")

    # ---------------------------------------------- anchors + stable ref AOI
    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")
    def rast(cid):
        return rio_rasterize([(gdf.loc[cid, "geometry"], 1)],
                             out_shape=(G["ny"], G["nx"]), transform=G["tr"],
                             fill=0, dtype="uint8").astype(bool)
    w_hi = rast(inv3.contour_id.iloc[-1])
    d_fp = ndimage.distance_transform_edt(~inside) * CELL
    stable_ref = (~inside & (d_fp > STABLE_REF_MIN_M) & (d_fp < STABLE_REF_MAX_M))
    print(f"\n  STABLE terrestrial reference AOI: "
          f"{stable_ref.sum()*CELL**2/1e6:,.0f} km2 outside the former reservoir")

    # ---------------------------------------------- STEP 6-9: fetch + QA
    print("\n" + "=" * 78)
    print("STEP 6-9 — FETCH ALL ASSETS PER EVENT, MOSAIC, THEN QA")
    print("=" * 78)
    tok = sas_token()
    qa = []
    for i, r in enumerate(SEL.itertuples(), 1):
        print(f"\n  [{i}/{len(SEL)}] {r.event_id}  {r.asset_count} assets  "
              f"span {r.acquisition_time_span_minutes:.1f} min  "
              f"[{r.temporal_regime}]  {r.select_reason}")
        items = [f for f in stac_day(r.date)
                 if f["properties"].get("sat:relative_orbit") == r.relative_orbit
                 and f["properties"]["datetime"][:10] == r.date]
        if not items:
            print("      no STAC items resolved -- skipped")
            continue
        try:
            vv, vh, cov, seam = build_event(r.event_id, items, tok, G)
        except Exception as ex:
            print(f"      EVENT BUILD FAILED {type(ex).__name__}: {ex}")
            continue
        obs = cov & inside
        frac = float(obs.sum()) / inside.sum()
        if obs.sum() < 1000:
            qa.append(dict(event_id=r.event_id, date=r.date,
                           relative_orbit=int(r.relative_orbit),
                           orbit_state=r.orbit_state,
                           temporal_regime=r.temporal_regime,
                           select_reason=r.select_reason, stratum=r.stratum,
                           asset_count=int(r.asset_count),
                           acquisition_time_span_minutes=float(
                               r.acquisition_time_span_minutes),
                           valid_coverage_fraction=frac,
                           nodata_fraction=1.0 - frac,
                           rejection_reason="no_overlap_with_footprint"))
            print(f"      coverage {frac:.3f} -- REJECT (no overlap)")
            del vv, vh, cov, obs
            continue
        vvd, vhd = db(vv[obs]), db(vh[obs])
        edge = obs & (ndimage.distance_transform_edt(cov) * CELL < BORDER_DIST_M)
        border_noise = (float((db(vv[edge]) < BORDER_DARK_DB).sum()) /
                        max(float(obs.sum()), 1.0)) if edge.any() else 0.0
        sref = stable_ref & cov
        qa.append(dict(
            event_id=r.event_id, date=r.date,
            relative_orbit=int(r.relative_orbit), orbit_state=r.orbit_state,
            temporal_regime=r.temporal_regime, select_reason=r.select_reason,
            stratum=r.stratum, asset_count=int(r.asset_count),
            acquisition_time_span_minutes=float(r.acquisition_time_span_minutes),
            valid_coverage_fraction=frac, nodata_fraction=1.0 - frac,
            VV_valid_fraction=float((np.isfinite(vv) & inside).sum()) / inside.sum(),
            VH_valid_fraction=float((np.isfinite(vh) & inside).sum()) / inside.sum(),
            incidence_angle_median=np.nan, incidence_angle_iqr=np.nan,
            incidence_angle_available=False,
            incidence_angle_reason="not provided by source RTC product",
            VV_median=float(np.median(vvd)),
            VV_p05=float(np.percentile(vvd, 5)),
            VV_p95=float(np.percentile(vvd, 95)),
            VH_median=float(np.median(vhd)),
            VH_p05=float(np.percentile(vhd, 5)),
            VH_p95=float(np.percentile(vhd, 95)),
            VV_VH_ratio_median=float(np.median(vvd - vhd)),
            border_noise_fraction=border_noise,
            mosaic_seam_metric=seam,
            seam_metric_applicable=bool(np.isfinite(seam)),
            seam_metric_reason=("" if np.isfinite(seam) else
                                "GRD slices abut without overlap; metric undefined"),
            stable_reference_VV_median=float(np.median(db(vv[sref])))
                if sref.sum() > 1000 else np.nan,
            stable_reference_VH_median=float(np.median(db(vh[sref])))
                if sref.sum() > 1000 else np.nan))
        print(f"      cov {frac:.3f}  VV {qa[-1]['VV_median']:6.2f} dB  "
              f"VH {qa[-1]['VH_median']:6.2f} dB  "
              f"seam {seam if np.isfinite(seam) else float('nan'):.2f} dB  "
              f"border {100*border_noise:.2f}%")
        del vv, vh, cov, obs, edge

    QA = pd.DataFrame(qa)
    if QA.empty:
        raise SystemExit("no events qualified")

    # ---------------------------------------------- STEP 10: quality labels
    print("\n" + "=" * 78)
    print("STEP 10 — EVENT-LEVEL RADIOMETRIC QUALITY")
    print("=" * 78)
    for pol in ("VV", "VH"):
        col = f"stable_reference_{pol}_median"
        if col in QA:
            mu, sd = QA[col].median(), 1.4826 * (QA[col] - QA[col].median()).abs().median()
            QA[f"stable_reference_{pol}_zscore"] = (QA[col] - mu) / (sd if sd > 0 else np.nan)

    def has_text(v):
        """A flag is set only by a non-empty STRING. Never by NaN: float('nan')
        is truthy in Python, so a bare `if r.get(col)` marks every row once any
        single row has populated that column."""
        return isinstance(v, str) and v.strip() != ""

    def num(v):
        """A numeric test fires only on a finite number."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None

    def classify(r):
        """Return (spatial_status, radiometric_quality, reason, flags).

        Coverage and radiometry are SEPARATE axes. An event that misses the
        analysis footprint is not a bad SAR image -- it is simply not
        applicable to this spatial task, so its radiometry is NOT_EVALUATED
        and it never enters the radiometric denominator.
        """
        if has_text(r.get("rejection_reason")):
            return ("NO_OVERLAP", "NOT_EVALUATED",
                    "outside_analysis_footprint", "")
        cov = num(r.get("valid_coverage_fraction"))
        if cov is None or cov <= 0.001:
            return ("NO_OVERLAP", "NOT_EVALUATED",
                    "outside_analysis_footprint", "")
        spatial = "VALID_OVERLAP" if cov >= 0.95 else "PARTIAL_OVERLAP"

        flags = []
        bn = num(r.get("border_noise_fraction"))
        if bn is not None and bn > 0.05:
            flags.append("border_noise_high")
        # an inapplicable metric must not penalise the event
        if bool(r.get("seam_metric_applicable", False)):
            sm = num(r.get("mosaic_seam_metric"))
            if sm is not None and sm > 1.5:
                flags.append("mosaic_seam_gt_1.5dB")
        for pol in ("VV", "VH"):
            z = num(r.get(f"stable_reference_{pol}_zscore"))
            if z is not None and abs(z) > 3.0:
                flags.append(f"stable_ref_{pol}_zscore_gt3")
        # incidence angle is absent from the product for EVERY event; that is
        # a source limitation, never a per-event degradation

        fl = "|".join(flags)
        if len(flags) >= 2:
            return spatial, "DEGRADED", fl, fl
        if flags:
            return spatial, "USABLE", "", fl
        if cov < 0.50:
            return spatial, "USABLE", "", "partial_coverage"
        return spatial, "GOOD", "", ""

    lab = QA.apply(lambda r: pd.Series(
        classify(r), index=["spatial_status", "radiometric_quality",
                            "rejection_reason", "quality_flags"]), axis=1)
    QA = QA.drop(columns=[c for c in ("spatial_status", "radiometric_quality",
                                      "rejection_reason", "quality_flags")
                          if c in QA.columns])
    QA = pd.concat([QA, lab], axis=1)
    QA.to_csv(CFG.TABLES / "hist25b_event_radiometric_qa.csv", index=False)
    print("  spatial_status (coverage axis):")
    print(QA.spatial_status.value_counts().to_string())
    print("\n  radiometric_quality (radiometry axis):")
    print(QA.radiometric_quality.value_counts().to_string())
    evald = QA[QA.radiometric_quality != "NOT_EVALUATED"]
    print(f"\n  radiometric denominator excludes NO_OVERLAP: "
          f"{len(evald)} of {len(QA)} events evaluated")
    print("  incidence_angle_available = False for ALL events "
          "(source RTC product limitation, not a per-event defect)")
    napp = int((~QA.seam_metric_applicable.astype(bool)).sum())
    print(f"  seam_metric_applicable = False for {napp}/{len(QA)} events "
          "(GRD slices abut without overlap); these are NOT penalised")
    print("\n  no event is silently removed; REJECT rows are retained with a "
          "rejection_reason")

    # ---------------------------------------------- STEP 11: Pass-2 strata
    print("\n" + "=" * 78)
    print("STEP 11 — PASS-2 TOP-UP STRATA (quality as a POST-fetch criterion)")
    print("=" * 78)
    QA["usable"] = QA.radiometric_quality.isin(["GOOD", "USABLE"])
    cov_mat = (QA.groupby(["stratum", "temporal_regime", "relative_orbit",
                           "spatial_status"])
                 .agg(events=("event_id", "size"), usable=("usable", "sum"),
                      good=("radiometric_quality",
                            lambda s: int((s == "GOOD").sum())))
                 .reset_index())
    # a cell is only short if it has no usable event AND is spatially applicable
    cov_mat["needs_topup"] = ((cov_mat.usable == 0) &
                              (cov_mat.spatial_status != "NO_OVERLAP"))
    cov_mat.to_csv(CFG.TABLES / "hist25b_pass2_topup_strata.csv", index=False)
    need = cov_mat[cov_mat.needs_topup]
    print(f"  stage x regime x orbit x quality matrix: {len(cov_mat)} cells")
    print(f"  spatially-applicable cells with NO usable event: {len(need)}")
    if len(need):
        print(need.to_string(index=False))
    else:
        print("  -> no Pass-2 top-up required; every spatially-applicable "
              "stratum x orbit cell has at least one usable event")

    # ---------------------------------------------- STEP 13: freeze manifest
    MAN = E.merge(QA.drop(columns=[c for c in ("date", "relative_orbit",
                                               "orbit_state", "temporal_regime",
                                               "asset_count",
                                               "acquisition_time_span_minutes")
                                   if c in QA.columns]),
                  on="event_id", how="left")
    MAN["gate6_selected"] = MAN.event_id.isin(SEL.event_id)
    MAN["datetime"] = MAN.acquisition_start
    man_path = MANIFEST_CSV
    MAN.to_csv(man_path, index=False)
    h = hashlib.sha256(man_path.read_bytes()).hexdigest()
    meta = dict(manifest=str(man_path), sha256=h,
                n_assets=int(len(A)), n_events=int(len(E)),
                n_selected=int(len(SEL)), n_qualified=int(len(QA)),
                assets_per_event=float(len(A) / len(E)),
                stage_tiers={t: v for t, v in STAGE_TIERS},
                margin_slope_m_per_m=MARGIN_SLOPE,
                wse_static_sigma_m=SIG_STATIC,
                zero_stage_wse_m=ZERO_STAGE_WSE,
                obs_time_known=OBS_TIME_KNOWN,
                incidence_angle_available=False,
                frozen_utc=pd.Timestamp.utcnow().isoformat())
    MANIFEST_HASH.write_text(json.dumps(meta, indent=2))
    print(f"\n  manifest frozen: sha256 {h[:16]}...")

    # ================================================ FIGURES AND MAPS
    _figures(E, QA, RAW, ELG, lev, reg_daily, targets, G, SEL, tok, stable_ref)

    print("\n" + "=" * 78)
    print("GATE 6 COMPLETE — STOP BEFORE CLASSIFICATION")
    print("=" * 78)
    for p in ("hist25b_raw_stage_proximity", "hist25b_eligible_target_stage_geometry",
              "hist25b_event_grouping_verification", "hist25b_event_radiometric_qa",
              "hist25b_pass2_topup_strata", "hist25b_gate6_event_manifest"):
        print(f"-> {CFG.TABLES/(p+'.csv')}")
    print(f"-> {CFG.TABLES/'hist25b_gate6_manifest_hash.json'}")


def _figures(E, QA, RAW, ELG, lev, reg_daily, targets, G, SEL, tok, stable_ref):
    """Gate-6 diagnostic figures and the event-coverage maps."""
    # --- fig 1: stage-time inventory with temporal regime ------------------
    fig, ax = plt.subplots(2, 1, figsize=(13.6, 8.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    a = ax[0]
    a.plot(lev.index, lev.values, color=GREY, lw=0.8, zorder=1)
    cols = {"PREBREACH_IMPOUNDED": BLUE, "BREACH_ONSET": PURPLE,
            "ACTIVE_DRAWDOWN": RED, "POSTBREACH_UNRESOLVED": AMBER}
    for rg, c in cols.items():
        m = reg_daily == rg
        if m.any():
            a.scatter(lev.index[m], lev.values[m], s=4, color=c, label=rg, zorder=2)
    for cid, t in targets.items():
        a.axhline(t, color=GREEN, ls=":", lw=1.1)
        a.text(lev.index[5], t + 0.06, cid, fontsize=8, color=GREEN)
    a.axhline(ZERO_STAGE_WSE, color=RED, ls="-.", lw=1.2)
    a.text(lev.index[5], ZERO_STAGE_WSE + 0.06,
           "gauge zero (vertical reference, NOT a floor)", fontsize=7.4, color=RED)
    a.set_ylabel("WSE (m EVRF2019)")
    a.legend(fontsize=7.4, ncol=4, loc="lower left")
    a.grid(alpha=0.22)
    a.set_title("a · temporal regime only — spatial hydraulic state is a "
                "separate axis, left UNRESOLVED", fontsize=10.4, loc="left")
    a = ax[1]
    res = E[E.WSE_RESOLVED]
    unres = E[~E.WSE_RESOLVED]
    a.scatter(pd.to_datetime(res.date), res.WSE_nominal, s=7, color=BLUE,
              label=f"WSE resolved (n={len(res)})")
    zp = E[E.zero_stage_plateau_flag]
    if len(zp):
        a.scatter(pd.to_datetime(zp.date), zp.WSE_nominal, s=26, color=RED,
                  marker="s", label=f"zero-stage plateau, semantics "
                                    f"unresolved (n={len(zp)})")
    a.set_ylabel("WSE at event (m)"); a.set_xlabel("date")
    a.legend(fontsize=7.6); a.grid(alpha=0.22)
    a.set_title("b · observation events on the stage-time plane",
                fontsize=10.4, loc="left")
    fig.suptitle("hist25b Gate 6 · stage-time event inventory", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate6_stage_time_events.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    # --- fig 2: raw proximity vs eligibility -------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=False)
    for a, cid in zip(ax, targets):
        r = RAW[RAW.contour_id == cid].set_index("stage_tier")
        e = ELG[ELG.contour_id == cid].set_index("stage_tier")
        x = np.arange(len(r))
        a.bar(x - 0.2, r.events, width=0.4, color=GREY, label="raw stage proximity")
        a.bar(x + 0.2, e.eligible_events, width=0.4, color=GREEN,
              label="eligible target-stage geometry")
        a.set_xticks(x); a.set_xticklabels([s[-3:] for s in r.index])
        a.set_xlabel("stage tier (cm)")
        a.set_title(f"{cid}  ({targets[cid]:.3f} m)", fontsize=10.2, loc="left")
        a.grid(alpha=0.25, axis="y")
        for xi, (rv, evv) in enumerate(zip(r.events, e.eligible_events)):
            a.text(xi - 0.2, rv, str(rv), ha="center", va="bottom", fontsize=7.5)
            a.text(xi + 0.2, evv, str(evv), ha="center", va="bottom", fontsize=7.5)
    ax[0].set_ylabel("observation events")
    ax[0].legend(fontsize=7.8)
    fig.suptitle("hist25b Gate 6 · LEVEL proximity is not HYDRAULIC STATE "
                 "equivalence", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate6_proximity_vs_eligibility.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    # --- fig 3: radiometric QA --------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13.4, 8.6))
    a = ax[0, 0]
    qc = {"GOOD": GREEN, "USABLE": BLUE, "DEGRADED": AMBER, "REJECT": RED}
    for q, c in qc.items():
        s = QA[QA.radiometric_quality == q]
        if len(s):
            a.scatter(s.valid_coverage_fraction, s.VV_median, s=46, color=c, label=q)
    a.set_xlabel("valid coverage fraction"); a.set_ylabel("VV median (dB)")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_title("a · coverage vs backscatter, by quality label",
                fontsize=10.2, loc="left")
    a = ax[0, 1]
    for q, c in qc.items():
        s = QA[QA.radiometric_quality == q]
        if len(s):
            a.scatter(s.VV_median, s.VH_median, s=46, color=c, label=q)
    a.set_xlabel("VV median (dB)"); a.set_ylabel("VH median (dB)")
    a.grid(alpha=0.25)
    a.set_title("b · VV/VH event radiometry", fontsize=10.2, loc="left")
    a = ax[1, 0]
    sm = QA.mosaic_seam_metric.dropna()
    if len(sm):
        a.hist(sm, bins=24, color=BLUE, alpha=0.8)
        a.axvline(1.5, color=RED, ls="--", lw=1.4, label="DEGRADED threshold")
        a.legend(fontsize=8)
    a.set_xlabel("mosaic seam |ΔVV| (dB)"); a.set_ylabel("events")
    a.grid(alpha=0.25)
    a.set_title("c · calibration discrepancy across mosaic seams",
                fontsize=10.2, loc="left")
    a = ax[1, 1]
    reg_order = ["PREBREACH_IMPOUNDED", "BREACH_ONSET", "ACTIVE_DRAWDOWN",
                 "POSTBREACH_UNRESOLVED"]
    tab = (QA.groupby(["temporal_regime", "radiometric_quality"]).size()
             .unstack(fill_value=0).reindex(reg_order).fillna(0))
    bot = np.zeros(len(tab))
    for q, c in qc.items():
        if q in tab.columns:
            a.bar(range(len(tab)), tab[q], bottom=bot, color=c, label=q)
            bot += tab[q].values
    a.set_xticks(range(len(tab)))
    a.set_xticklabels([s.replace("_", "\n") for s in tab.index], fontsize=7.2)
    a.set_ylabel("events"); a.legend(fontsize=8); a.grid(alpha=0.25, axis="y")
    a.set_title("d · quality by temporal regime", fontsize=10.2, loc="left")
    fig.suptitle("hist25b Gate 6 · event-level radiometric qualification",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate6_radiometric_qa.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    # --- fig 4: MAPS — per-event coverage and the stable reference AOI -----
    inside = G["inside"]
    ext = [G["x0"] / 1000, G["x1"] / 1000, G["y0"] / 1000, G["y1"] / 1000]
    show = (QA.sort_values(["temporal_regime", "valid_coverage_fraction"],
                           ascending=[True, False])
              .drop_duplicates("relative_orbit").head(6))
    n = max(len(show), 1)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.3 * nrow),
                           squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, r in zip(ax.ravel(), show.itertuples()):
        npz = CACHE / f"{r.event_id}.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        cov = z["cov"]
        canvas = np.zeros(inside.shape, np.uint8)
        canvas[stable_ref] = 1
        canvas[inside] = 2
        canvas[inside & cov] = 3
        a.axis("on")
        a.imshow(canvas, extent=ext, origin="upper", interpolation="nearest",
                 cmap=matplotlib.colors.ListedColormap(
                     ["#ffffff", "#dfe4ea", "#f2c9c0", BLUE]), vmin=0, vmax=3)
        a.set_title(f"{r.event_id}\norb {r.relative_orbit} · "
                    f"cov {100*r.valid_coverage_fraction:.0f}% · "
                    f"{r.radiometric_quality}", fontsize=8.4, loc="left")
        a.set_xlabel("easting (km)", fontsize=7.5)
        a.set_ylabel("northing (km)", fontsize=7.5)
        a.tick_params(labelsize=6.8)
        del z, cov, canvas
    fig.suptitle("hist25b Gate 6 · observation-event footprints  "
                 "(blue = observed reservoir, pink = unobserved, "
                 "grey = stable terrestrial reference AOI)",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate6_event_coverage_maps.png", dpi=165,
                bbox_inches="tight")
    plt.close(fig)

    # --- fig 5: MAP — cumulative observation redundancy --------------------
    stack = np.zeros(inside.shape, np.int16)
    pre = QA[(QA.temporal_regime == "PREBREACH_IMPOUNDED") &
             QA.radiometric_quality.isin(["GOOD", "USABLE"])]
    for r in pre.itertuples():
        npz = CACHE / f"{r.event_id}.npz"
        if npz.exists():
            z = np.load(npz)
            stack += (z["cov"] & inside).astype(np.int16)
            del z
    fig, a = plt.subplots(figsize=(9.6, 7.4))
    m = np.where(inside, stack, np.nan)
    im = a.imshow(m, extent=ext, origin="upper", cmap="viridis",
                  interpolation="nearest")
    cb = fig.colorbar(im, ax=a, shrink=0.84)
    cb.set_label("number of usable pre-breach events observing the pixel")
    a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    a.set_title("hist25b Gate 6 · observation redundancy over the former "
                "reservoir\nusable pre-breach events only; low values mark "
                "strata needing Pass-2 top-up", fontsize=10.6, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate6_observation_redundancy_map.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    for f in ("stage_time_events", "proximity_vs_eligibility", "radiometric_qa",
              "event_coverage_maps", "observation_redundancy_map"):
        print(f"-> {FIGDIR/('hist25b_gate6_'+f+'.png')}")


if __name__ == "__main__":
    main()
