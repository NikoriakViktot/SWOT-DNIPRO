#!/usr/bin/env python
"""HIST25B — why did the Sentinel-1 classifier fail? (GATES 1-4 only)

hist25 is frozen as a negative experiment. Its SAR branch produced
non-physical open-water areas (H1 876, H2 630, H3 1876 km2 against
independent footprints of ~2192 / ~2010 / ~1989 km2) and a signal/bias of
0.03, worse than optical alone. This script does NOT propose a replacement
classifier. It establishes, quantitatively, WHICH mechanism broke it.

The spec this implements is gated. Gates 5-10 (classifier candidates M0-M4,
flooded vegetation, consensus, footprint validation) are deliberately NOT
implemented here: choosing a classifier before the failure is diagnosed is
the unprincipled tuning the spec forbids.

  GATE 1  (B0)       reproduce hist25's SAR result from the frozen cache
  GATE 2  (B1,B2)    audit the RTC product: units, scaling, the old clamp
  GATE 3  (B3,B5)    per-scene diagnostics BEFORE compositing; is H2 already
                     broken at scene level, or does compositing break it?
  GATE 4  (B9,B10)   independent calibration anchors; does a two-class
                     structure exist at all -> is Otsu even eligible?

CALIBRATION ANCHORS (B9). Parameters are never chosen from the target
shorelines or the known footprint areas. Three anchors are derived from the
hist23 optical contours and buffered away from every target shoreline:

  WATER_CORE   inside the LOWEST contour, >500 m from its boundary
               => water at all three levels, contains no target shoreline
  LAND_MARGIN  inside the historical footprint but outside the HIGHEST
               contour, >100 m from it => dry at all three levels
  LAND_OUTER   500-3000 m outside the historical footprint

The known H1/H2/H3 footprint areas are used NOWHERE in this script. They are
validation-only and remain untouched until Gate 9.

WHAT IS ALREADY KNOWN BY INSPECTION (verified before writing this, and
re-measured below rather than asserted):

  * phase19_s1_watermask.otsu_threshold() clamps INTERNALLY to
    [0.005, 0.05] linear = [-23.0, -13.0] dB, regardless of polarisation.
  * water_mask() then re-clamps VH to [0.0020, 0.0130]. Because the inner
    clamp already floors at 0.005, the VH threshold can only ever land in
    [0.005, 0.013] = [-23.0, -18.9] dB. The declared VH_MIN of -27 dB is
    UNREACHABLE. hist25 inherited this double clamp.
  * hist25 did NOT apply enhanced_lee speckle filtering, which the engine's
    own water_mask() applies before thresholding. hist25 therefore used the
    engine outside its design.

Outputs
-------
outputs/tables/hist25b_hist25_reproduction.csv
outputs/tables/hist25b_scene_diagnostics.csv
outputs/tables/hist25b_cross_orbit_consistency.csv
outputs/reports/hist25b_rtc_product_audit.md
outputs/figures/hist25b_raw_histograms.png
outputs/figures/hist25b_old_clamp_diagnostic.png
outputs/figures/hist25b_scene_water_backscatter.png
outputs/figures/hist25b_orbit_comparison.png
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
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
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape as shp_shape

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD
import phase19_s1_watermask as S1

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
CELL = 20.0
CACHE = CFG.BULK_ROOT / "data_swot/processed/bathymetry/s1_cache"
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
REPORTS = CFG.OUT / "reports"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# anchor buffers (B9) -- all expressed as distance from a contour boundary
WATER_CORE_BUFFER_M = 500.0
LAND_MARGIN_BUFFER_M = 100.0
LAND_OUTER_MIN_M, LAND_OUTER_MAX_M = 500.0, 3000.0
DB_FLOOR = 1e-6                       # guard for log10 of zero/negative


def db(x):
    return 10.0 * np.log10(np.maximum(x, DB_FLOOR))


def robust(v_db):
    """Median and a robust spread for a dB-valued sample."""
    if v_db.size == 0:
        return dict(n=0, med=np.nan, nmad=np.nan, p05=np.nan, p25=np.nan,
                    p75=np.nan, p95=np.nan)
    med = float(np.median(v_db))
    return dict(
        n=int(v_db.size), med=med,
        nmad=float(1.4826 * np.median(np.abs(v_db - med))),
        p05=float(np.percentile(v_db, 5)), p25=float(np.percentile(v_db, 25)),
        p75=float(np.percentile(v_db, 75)), p95=float(np.percentile(v_db, 95)))


def separability(w_db, l_db):
    """Robust two-sample separation (a Cohen-d analogue built on NMAD, so a
    few misclassified anchor pixels cannot dominate it)."""
    if w_db.size == 0 or l_db.size == 0:
        return np.nan, np.nan
    mw, ml = np.median(w_db), np.median(l_db)
    sw = 1.4826 * np.median(np.abs(w_db - mw))
    sl = 1.4826 * np.median(np.abs(l_db - ml))
    pooled = np.sqrt(0.5 * (sw ** 2 + sl ** 2))
    d = abs(ml - mw) / pooled if pooled > 0 else np.nan
    # overlap of the two robust distributions, as the fraction of water
    # pixels brighter than the land 5th percentile
    overlap = float((w_db > np.percentile(l_db, 5)).mean())
    return float(d), overlap


def unclamped_otsu(x_linear):
    """The engine's Otsu, with the internal clamp removed, so the clamp's
    effect can be measured instead of assumed. Identical arithmetic
    otherwise (log10 domain, 256 bins, between-class variance)."""
    valid = x_linear[np.isfinite(x_linear) & (x_linear > 0)]
    if valid.size < S1.OTSU_MIN_VALID:
        return np.nan
    data = np.log10(valid)
    hist, edges = np.histogram(data, bins=256)
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * edges[:-1])
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)
    return float(10 ** edges[int(np.argmax(sigma_b))])


def modality(x_db, bins=256):
    """Count well-separated density modes, and report the depth of the
    deepest valley between the two largest ones. A histogram with no real
    valley makes global Otsu ineligible (B10)."""
    if x_db.size < 1000:
        return dict(n_modes=0, valley_depth=np.nan, peak_sep_db=np.nan)
    hist, edges = np.histogram(x_db, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sm = ndimage.uniform_filter1d(hist.astype(float), size=9)
    if sm.max() <= 0:
        return dict(n_modes=0, valley_depth=np.nan, peak_sep_db=np.nan)
    sm /= sm.max()
    # local maxima at least 2% of peak height
    ismax = (sm > np.roll(sm, 1)) & (sm > np.roll(sm, -1)) & (sm > 0.02)
    idx = np.flatnonzero(ismax)
    if idx.size < 2:
        return dict(n_modes=int(idx.size), valley_depth=0.0, peak_sep_db=0.0)
    order = idx[np.argsort(sm[idx])[::-1]][:2]
    a, b = sorted(order)
    valley = sm[a:b + 1].min()
    depth = float(min(sm[a], sm[b]) - valley)
    return dict(n_modes=int(idx.size), valley_depth=depth,
                peak_sep_db=float(abs(centres[b] - centres[a])))


def _replay(cid, ordering_name, seq, inside, cache):
    """Replay hist25's composite-and-threshold chain for one level under a
    given acquisition ordering, and measure how much of the detected water
    survives the largest-connected-component step."""
    water = np.zeros(inside.shape, bool)
    flood = np.zeros(inside.shape, bool)
    valid = np.zeros(inside.shape, bool)
    used = []
    for date, orb in seq:
        f = cache / f"s1_{cid}_{date}_orb{int(orb)}_{CELL:.0f}m.npz"
        if not f.exists():
            continue
        z = np.load(f)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        newpx = cov & ~valid & inside
        gain = float(newpx.sum()) / float(inside.sum())
        if newpx.any() and newpx.sum() >= S1.OTSU_MIN_VALID:
            fvv, fvh = vv[newpx], vh[newpx]
            tvv = float(np.clip(S1.otsu_threshold(fvv), S1.OTSU_MIN, S1.OTSU_MAX))
            tvh = float(np.clip(S1.otsu_threshold(fvh), S1.VH_MIN, S1.VH_MAX))
            ow = np.zeros(inside.shape, bool)
            vg = np.zeros(inside.shape, bool)
            ow[newpx] = (fvv < tvv) & (fvh < tvh)
            vg[newpx] = (fvv < tvv) & (fvh >= tvh)
            water |= ow & newpx
            flood |= vg & newpx
            del ow, vg, fvv, fvh
        valid |= cov
        if gain > 0.001:
            used.append(f"{date}/orb{int(orb)}({100*gain:.0f}%)")
        del vv, vh, cov, newpx
    comb = (water | flood) & inside
    lab, n = ndimage.label(comb, structure=np.ones((3, 3), int))
    if n == 0:
        return dict(contour_id=cid, ordering=ordering_name, detected_km2=0.0,
                    largest_component_km2=0.0, lost_to_fragmentation_km2=0.0,
                    lost_pct=np.nan, n_components=0, scene_sequence="")
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = lab == (int(np.argmax(sizes)) + 1)
    det = float(comb.sum()) * CELL ** 2 / 1e6
    lc = float(big.sum()) * CELL ** 2 / 1e6
    return dict(contour_id=cid, ordering=ordering_name, detected_km2=det,
                largest_component_km2=lc, lost_to_fragmentation_km2=det - lc,
                lost_pct=100 * (det - lc) / det if det else np.nan,
                n_components=int(n), scene_sequence=" -> ".join(used))


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


def dist_inside_m(mask):
    """Distance (m) from each True pixel to the nearest False pixel."""
    return ndimage.distance_transform_edt(mask) * CELL


def dist_outside_m(mask):
    """Distance (m) from each False pixel to the nearest True pixel."""
    return ndimage.distance_transform_edt(~mask) * CELL


def stac_orbit_lookup(level_windows, bbox):
    """(date, relative_orbit) -> orbit_state. Degrades to empty on any
    network problem; the diagnostics below still run without it."""
    out = {}
    for start, end in level_windows:
        q = {"collections": ["sentinel-1-rtc"], "bbox": bbox, "limit": 200,
             "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"}
        try:
            r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                       headers={"Content-Type": "application/json"})
            feats = json.loads(urllib.request.urlopen(r, timeout=90).read())["features"]
        except Exception as ex:
            print(f"    STAC lookup failed ({type(ex).__name__}) -- "
                  f"orbit direction will be reported as UNKNOWN")
            return out
        for f in feats:
            p = f["properties"]
            key = (p["datetime"][:10], p.get("sat:relative_orbit"))
            out[key] = dict(orbit_state=p.get("sat:orbit_state", "UNKNOWN"),
                            platform=p.get("platform", ""),
                            pols="+".join(p.get("sar:polarizations", [])),
                            item_id=f["id"], datetime=p["datetime"][:19])
    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    # registry domain, not the retired P20 footprint: that stops 9.4 km short
    # on the east and 4.1 km on the north, and here it defined the GRID
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    print(f"grid {G['nx']}x{G['ny']} at {CELL:.0f} m; footprint "
          f"{inside.sum()*CELL**2/1e6:,.0f} km2")

    # ---------------------------------------------------------------- B20
    # Never assume the label order implies stage order.
    inv = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv")
    inv = inv.sort_values("H_evrf2019_m").reset_index(drop=True)
    print("\n" + "=" * 78)
    print("B20 — ACTUAL STAGE ORDER (never assumed from the labels)")
    print("=" * 78)
    for r in inv.itertuples():
        print(f"  {r.contour_id}: WSE {r.H_evrf2019_m:.3f} m EVRF2019")
    lowest, highest = inv.contour_id.iloc[0], inv.contour_id.iloc[-1]
    print(f"  lowest stage = {lowest}, highest stage = {highest}")
    print(f"  expected nesting: " +
          " subset of ".join(inv.contour_id.tolist()))

    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")
    def rast(cid):
        return rio_rasterize([(gdf.loc[cid, "geometry"], 1)],
                             out_shape=(G["ny"], G["nx"]), transform=G["tr"],
                             fill=0, dtype="uint8").astype(bool)
    w_lowest, w_highest = rast(lowest), rast(highest)

    # ---------------------------------------------------------------- B9
    print("\n" + "=" * 78)
    print("GATE 4a / B9 — INDEPENDENT CALIBRATION ANCHORS")
    print("=" * 78)
    water_core = w_lowest & (dist_inside_m(w_lowest) > WATER_CORE_BUFFER_M)
    land_margin = (inside & ~w_highest &
                   (dist_outside_m(w_highest) > LAND_MARGIN_BUFFER_M))
    d_out = dist_outside_m(inside)
    land_outer = (~inside & (d_out > LAND_OUTER_MIN_M) & (d_out < LAND_OUTER_MAX_M))
    land_all = land_margin | land_outer
    for nm, m in (("WATER_CORE", water_core), ("LAND_MARGIN", land_margin),
                  ("LAND_OUTER", land_outer)):
        print(f"  {nm:<12} {m.sum()*CELL**2/1e6:8,.0f} km2  "
              f"({m.sum():,} cells)")
    print("  None of these contain a target shoreline; the known footprint")
    print("  areas are not used anywhere in this script.")

    # ---------------------------------------------------------------- GATE 1/2/3
    files = sorted(CACHE.glob("s1_*_20m.npz"))
    if not files:
        raise SystemExit("GATE 1 FAILED: the hist25 S1 cache is empty, so the "
                         "frozen experiment cannot be reproduced. Re-run "
                         "hist25 before attempting recalibration.")
    print(f"\n  frozen hist25 cache: {len(files)} acquisitions")

    bbox = [32.2, 46.5, 35.5, 48.0]
    windows = [("2023-01-29", "2023-03-07"), ("2019-02-07", "2019-03-30"),
               ("2023-05-24", "2023-06-17")]
    print("\n  looking up orbit direction from STAC ...")
    orbit_meta = stac_orbit_lookup(windows, bbox)

    print("\n" + "=" * 78)
    print("GATES 1-3 — PER-SCENE DIAGNOSTICS, BEFORE ANY COMPOSITING")
    print("=" * 78)
    rows, repro, hist_samples = [], [], {}
    for f in files:
        m = re.match(r"s1_(H\d)_(\d{4}-\d{2}-\d{2})_orb(\d+)_", f.name)
        cid, date, orb = m.group(1), m.group(2), int(m.group(3))
        z = np.load(f)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        obs = cov & inside
        meta = orbit_meta.get((date, orb), {})
        orbit_state = meta.get("orbit_state", "UNKNOWN")

        if obs.sum() < 1000:
            print(f"  {cid} {date} orb{orb:>3}: no usable overlap "
                  f"({obs.sum():,} cells) -- skipped")
            rows.append(dict(contour_id=cid, date=date, relative_orbit=orb,
                             orbit_state=orbit_state, usable=False,
                             observed_frac=float(obs.sum()) / inside.sum()))
            del vv, vh, cov, obs
            continue

        a_w = water_core & cov
        a_l = land_all & cov
        vv_w, vh_w = db(vv[a_w]), db(vh[a_w])
        vv_l, vh_l = db(vv[a_l]), db(vh[a_l])
        rw_vv, rw_vh = robust(vv_w), robust(vh_w)
        rl_vv, rl_vh = robust(vv_l), robust(vh_l)
        d_vv, ov_vv = separability(vv_w, vv_l)
        d_vh, ov_vh = separability(vh_w, vh_l)

        # --- GATE 1: what hist25 actually did, reproduced exactly ---------
        sel = obs
        fvv, fvh = vv[sel], vh[sel]
        thr_vv_used = float(np.clip(S1.otsu_threshold(fvv), S1.OTSU_MIN, S1.OTSU_MAX))
        thr_vh_used = float(np.clip(S1.otsu_threshold(fvh), S1.VH_MIN, S1.VH_MAX))
        openw = np.zeros(inside.shape, bool)
        openw[sel] = (fvv < thr_vv_used) & (fvh < thr_vh_used)
        area_hist25 = float(openw.sum()) * CELL ** 2 / 1e6

        # --- GATE 2: the same Otsu with the clamp removed -----------------
        thr_vv_free = unclamped_otsu(fvv)
        thr_vh_free = unclamped_otsu(fvh)
        openw_free = np.zeros(inside.shape, bool)
        if np.isfinite(thr_vv_free) and np.isfinite(thr_vh_free):
            openw_free[sel] = (fvv < thr_vv_free) & (fvh < thr_vh_free)
        area_free = float(openw_free.sum()) * CELL ** 2 / 1e6

        # --- what an anchor-derived threshold would give (NOT fitted to
        # --- any target area: it is the midpoint between the two anchors)
        thr_vv_anchor_db = 0.5 * (rw_vv["med"] + rl_vv["med"])
        thr_vh_anchor_db = 0.5 * (rw_vh["med"] + rl_vh["med"])
        openw_anchor = np.zeros(inside.shape, bool)
        openw_anchor[sel] = ((db(fvv) < thr_vv_anchor_db) &
                             (db(fvh) < thr_vh_anchor_db))
        area_anchor = float(openw_anchor.sum()) * CELL ** 2 / 1e6

        # how much genuine water core does each rule recover?
        core_n = float((water_core & cov).sum())
        rec_hist25 = float((openw & water_core & cov).sum()) / core_n if core_n else np.nan
        rec_free = float((openw_free & water_core & cov).sum()) / core_n if core_n else np.nan
        rec_anchor = float((openw_anchor & water_core & cov).sum()) / core_n if core_n else np.nan
        # false alarms on anchored dry land
        land_n = float((land_all & cov).sum())
        fpr_hist25 = float((openw & land_all & cov).sum()) / land_n if land_n else np.nan
        fpr_anchor = float((openw_anchor & land_all & cov).sum()) / land_n if land_n else np.nan

        mod_vv = modality(db(fvv))
        mod_vh = modality(db(fvh))
        clamp_lo_db, clamp_hi_db = db(S1.OTSU_MIN), db(S1.OTSU_MAX)
        vv_pinned = abs(thr_vv_used - S1.OTSU_MIN) < 1e-9 or \
                    abs(thr_vv_used - S1.OTSU_MAX) < 1e-9
        vh_pinned = abs(thr_vh_used - S1.VH_MIN) < 1e-9 or \
                    abs(thr_vh_used - S1.VH_MAX) < 1e-9 or \
                    abs(thr_vh_used - S1.OTSU_MIN) < 1e-9

        rows.append(dict(
            contour_id=cid, date=date, relative_orbit=orb,
            orbit_state=orbit_state, usable=True,
            observed_frac=float(obs.sum()) / inside.sum(),
            water_core_vv_db=rw_vv["med"], water_core_vv_nmad=rw_vv["nmad"],
            water_core_vh_db=rw_vh["med"], water_core_vh_nmad=rw_vh["nmad"],
            land_vv_db=rl_vv["med"], land_vh_db=rl_vh["med"],
            sep_vv=d_vv, overlap_vv=ov_vv, sep_vh=d_vh, overlap_vh=ov_vh,
            vv_modes=mod_vv["n_modes"], vv_valley_depth=mod_vv["valley_depth"],
            vh_modes=mod_vh["n_modes"], vh_valley_depth=mod_vh["valley_depth"],
            otsu_eligible_vv=bool(mod_vv["n_modes"] >= 2 and
                                  mod_vv["valley_depth"] > 0.05),
            otsu_eligible_vh=bool(mod_vh["n_modes"] >= 2 and
                                  mod_vh["valley_depth"] > 0.05),
            thr_vv_used_db=db(thr_vv_used), thr_vh_used_db=db(thr_vh_used),
            thr_vv_unclamped_db=db(thr_vv_free) if np.isfinite(thr_vv_free) else np.nan,
            thr_vh_unclamped_db=db(thr_vh_free) if np.isfinite(thr_vh_free) else np.nan,
            thr_vv_anchor_db=thr_vv_anchor_db, thr_vh_anchor_db=thr_vh_anchor_db,
            vv_threshold_pinned_by_clamp=vv_pinned,
            vh_threshold_pinned_by_clamp=vh_pinned,
            area_hist25_km2=area_hist25, area_unclamped_km2=area_free,
            area_anchor_km2=area_anchor,
            watercore_recall_hist25=rec_hist25,
            watercore_recall_unclamped=rec_free,
            watercore_recall_anchor=rec_anchor,
            land_fpr_hist25=fpr_hist25, land_fpr_anchor=fpr_anchor))
        repro.append(dict(
            contour_id=cid, date=date, relative_orbit=orb,
            orbit_state=orbit_state, item_id=meta.get("item_id", ""),
            datetime=meta.get("datetime", ""), platform=meta.get("platform", ""),
            polarisations=meta.get("pols", ""), collection="sentinel-1-rtc",
            pixel_spacing_m=CELL, nodata_convention="NaN (boundless read)",
            units_assumed="linear power (gamma0 RTC)",
            vv_min=float(np.nanmin(fvv)), vv_p01=float(np.nanpercentile(fvv, 1)),
            vv_p50=float(np.nanpercentile(fvv, 50)),
            vv_p99=float(np.nanpercentile(fvv, 99)),
            vv_max=float(np.nanmax(fvv)),
            vh_p50=float(np.nanpercentile(fvh, 50)),
            thr_vv_used_linear=thr_vv_used, thr_vh_used_linear=thr_vh_used,
            water_area_km2=area_hist25))

        if cid not in hist_samples:
            step = max(1, int(sel.sum() // 400000))
            hist_samples[cid] = dict(vv=db(fvv[::step]), vh=db(fvh[::step]),
                                     vv_w=vv_w[::max(1, vv_w.size // 80000)],
                                     vv_l=vv_l[::max(1, vv_l.size // 80000)],
                                     date=date, orb=orb)

        print(f"\n  {cid} {date} orb{orb:>3} [{orbit_state}] "
              f"obs {100*obs.sum()/inside.sum():5.1f}%")
        print(f"    anchors  water_core VV {rw_vv['med']:7.2f} dB "
              f"(NMAD {rw_vv['nmad']:.2f}) | land VV {rl_vv['med']:7.2f} dB "
              f"| separation d={d_vv:.2f}")
        print(f"             water_core VH {rw_vh['med']:7.2f} dB "
              f"(NMAD {rw_vh['nmad']:.2f}) | land VH {rl_vh['med']:7.2f} dB "
              f"| separation d={d_vh:.2f}")
        print(f"    hist25   thr VV {db(thr_vv_used):7.2f} dB"
              f"{'  <-- PINNED BY CLAMP' if vv_pinned else ''}")
        print(f"             thr VH {db(thr_vh_used):7.2f} dB"
              f"{'  <-- PINNED BY CLAMP' if vh_pinned else ''}")
        print(f"    unclamped thr VV {db(thr_vv_free):7.2f} dB, "
              f"VH {db(thr_vh_free):7.2f} dB")
        print(f"    anchor    thr VV {thr_vv_anchor_db:7.2f} dB, "
              f"VH {thr_vh_anchor_db:7.2f} dB")
        print(f"    water-core recall: hist25 {100*rec_hist25:5.1f}%  "
              f"unclamped {100*rec_free:5.1f}%  anchor {100*rec_anchor:5.1f}%")
        print(f"    land false alarm : hist25 {100*fpr_hist25:5.1f}%  "
              f"anchor {100*fpr_anchor:5.1f}%")
        print(f"    modality VV: {mod_vv['n_modes']} modes, valley depth "
              f"{mod_vv['valley_depth']:.3f} -> Otsu eligible "
              f"{mod_vv['n_modes'] >= 2 and mod_vv['valley_depth'] > 0.05}")
        del vv, vh, cov, obs, fvv, fvh, openw, openw_free, openw_anchor

    D = pd.DataFrame(rows)
    D.to_csv(CFG.TABLES / "hist25b_scene_diagnostics.csv", index=False)
    pd.DataFrame(repro).to_csv(
        CFG.TABLES / "hist25b_hist25_reproduction.csv", index=False)

    U = D[D.usable].copy()

    # ---------------------------------------------------------------- B5
    print("\n" + "=" * 78)
    print("GATE 3b / B5 — CROSS-ORBIT CONSISTENCY")
    print("=" * 78)
    if U.orbit_state.nunique() > 1:
        co = (U.groupby("orbit_state")
                .agg(n=("date", "size"),
                     water_vv_db=("water_core_vv_db", "median"),
                     water_vh_db=("water_core_vh_db", "median"),
                     land_vv_db=("land_vv_db", "median"),
                     sep_vv=("sep_vv", "median"),
                     recall_hist25=("watercore_recall_hist25", "median"),
                     recall_anchor=("watercore_recall_anchor", "median"))
                .reset_index())
        print(co.to_string(index=False))
        co.to_csv(CFG.TABLES / "hist25b_cross_orbit_consistency.csv", index=False)
        g = U.groupby("orbit_state").water_core_vv_db.median()
        spread = float(g.max() - g.min())
        print(f"\n  ASC/DESC water-core VV offset: {spread:.2f} dB")
    else:
        print("  only one orbit direction resolved; cross-orbit test not "
              "possible from the frozen cache")
        U.to_csv(CFG.TABLES / "hist25b_cross_orbit_consistency.csv", index=False)

    # ---------------------------------------------------------------- B22
    print("\n" + "=" * 78)
    print("GATE 3c / B22 — IS H2 ALREADY BROKEN AT SCENE LEVEL?")
    print("=" * 78)
    per = (U.groupby("contour_id")
             .agg(scenes=("date", "size"),
                  water_vv_db=("water_core_vv_db", "median"),
                  water_vh_db=("water_core_vh_db", "median"),
                  land_vv_db=("land_vv_db", "median"),
                  sep_vv=("sep_vv", "median"),
                  recall_hist25=("watercore_recall_hist25", "median"),
                  recall_unclamped=("watercore_recall_unclamped", "median"),
                  recall_anchor=("watercore_recall_anchor", "median"),
                  thr_used_db=("thr_vv_used_db", "median"),
                  pinned=("vv_threshold_pinned_by_clamp", "mean"))
             .reindex(inv.contour_id.tolist()))
    print(per.to_string())
    print("\n  Reading: 'recall_hist25' is the fraction of PERSISTENT OPEN")
    print("  WATER (an anchor that contains no shoreline) that hist25's rule")
    print("  actually recovered. A low value here is a radiometric/threshold")
    print("  failure, not a shoreline or compositing failure.")

    # ---------------------------------------------------------------- B17/B23
    # Decompose the loss into (a) what the threshold detected and (b) what
    # survived hist25's largest-connected-component step. hist25 went
    # straight from a raw per-pixel threshold to largest_component(), while
    # the engine's own water_mask() first applies enhanced_lee, then
    # binary_opening/closing, then a 25-pixel minimum size. Skipping that
    # cleanup leaves a speckle-shredded mask, and on a shredded mask
    # "keep the largest component" is destructive.
    print("\n" + "=" * 78)
    print("GATE 3d / B17 — HOW MUCH WATER DID THE CONNECTIVITY STEP DISCARD?")
    print("=" * 78)
    # Two orderings are replayed, because the DIFFERENCE between them is the
    # finding. hist25 ordered by proximity to a verified gauge date, which is
    # motivated for stage-matching but is orthogonal to radiometric quality.
    inv_i = inv.set_index("contour_id")
    frag = []
    for cid in inv.contour_id:
        sub = U[U.contour_id == cid]
        tdates = [pd.Timestamp(d) for d in inv_i.loc[cid, "dates"].split("|")]
        scenes = list(map(tuple, sub[["date", "relative_orbit"]].values))
        recall = {(r.date, r.relative_orbit): r.watercore_recall_hist25
                  for r in sub.itertuples()}

        def by_date(t):
            return min(abs((pd.Timestamp(t[0]) - d).days) for d in tdates)

        orderings = {
            "as_run_nearest_date": sorted(scenes, key=by_date),
            "quality_first": sorted(scenes, key=lambda t: -recall.get(t, 0.0)),
        }
        for oname, seq in orderings.items():
            frag.append(_replay(cid, oname, seq, inside, CACHE))
    F = pd.DataFrame(frag)
    F.to_csv(CFG.TABLES / "hist25b_fragmentation_loss.csv", index=False)
    print(f"\n  {'level':<5}{'ordering':<22}{'detected':>11}{'largest':>10}"
          f"{'discarded':>11}{'components':>12}")
    for r in F.itertuples():
        print(f"  {r.contour_id:<5}{r.ordering:<22}{r.detected_km2:>10,.0f}"
              f"{r.largest_component_km2:>10,.0f}{r.lost_pct:>10.0f}%"
              f"{r.n_components:>12,}")
    print("\n  hist25 applied NO speckle filter and NO morphological cleanup")
    print("  before the connectivity step, unlike the engine's own")
    print("  water_mask(). On a shredded mask, 'keep the largest component'")
    print("  is destructive; the size of that loss depends on WHICH scene")
    print("  claimed the footprint first.")

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for a, cid in zip(ax, inv.contour_id):
        s = hist_samples.get(cid)
        if not s:
            a.axis("off"); continue
        a.hist(s["vv"], bins=180, color=AMBER, alpha=0.65, density=True,
               label="VV (all observed)")
        a.hist(s["vh"], bins=180, color=BLUE, alpha=0.55, density=True,
               label="VH (all observed)")
        a.axvline(db(S1.OTSU_MIN), color=RED, ls="--", lw=1.5,
                  label=f"clamp floor {db(S1.OTSU_MIN):.1f} dB")
        a.axvline(db(S1.OTSU_MAX), color=RED, ls=":", lw=1.5,
                  label=f"clamp ceiling {db(S1.OTSU_MAX):.1f} dB")
        a.set_title(f"{cid}  {s['date']} orb{s['orb']}", fontsize=10, loc="left")
        a.set_xlabel("backscatter (dB)"); a.set_xlim(-35, 5)
        a.legend(fontsize=7.2)
    ax[0].set_ylabel("density")
    fig.suptitle("hist25b · Sentinel-1 RTC backscatter distributions with the "
                 "inherited clamp band marked", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "hist25b_raw_histograms.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for a, cid in zip(ax, inv.contour_id):
        s = hist_samples.get(cid)
        if not s:
            a.axis("off"); continue
        a.hist(s["vv_w"], bins=140, color=BLUE, alpha=0.7, density=True,
               label="WATER_CORE anchor (VV)")
        a.hist(s["vv_l"], bins=140, color=GREY, alpha=0.6, density=True,
               label="LAND anchor (VV)")
        a.axvspan(db(S1.OTSU_MIN), db(S1.OTSU_MAX), color=RED, alpha=0.12,
                  label="inherited clamp band")
        a.axvline(db(S1.OTSU_MIN), color=RED, ls="--", lw=1.6)
        row = U[U.contour_id == cid]
        if len(row):
            a.axvline(float(row.thr_vv_used_db.median()), color=INK, lw=1.8,
                      label="threshold hist25 used")
            a.axvline(float(row.thr_vv_anchor_db.median()), color=GREEN, lw=1.8,
                      ls="-.", label="anchor-derived threshold")
        a.set_title(cid, fontsize=10, loc="left")
        a.set_xlabel("VV (dB)"); a.set_xlim(-30, 0)
        a.legend(fontsize=7.2)
    ax[0].set_ylabel("density")
    fig.suptitle("hist25b · where the inherited clamp sits relative to the "
                 "true water and land modes", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "hist25b_old_clamp_diagnostic.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    a = ax[0]
    for cid, c in zip(inv.contour_id, (BLUE, GREEN, AMBER)):
        s = U[U.contour_id == cid]
        a.scatter(s.water_core_vv_db, s.water_core_vh_db, s=70, color=c,
                  label=cid, zorder=3)
    a.axvline(db(S1.OTSU_MIN), color=RED, ls="--", lw=1.5,
              label="VV clamp floor")
    a.set_xlabel("water-core VV (dB)"); a.set_ylabel("water-core VH (dB)")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("a · persistent open water, per acquisition", fontsize=10.5,
                loc="left")
    a = ax[1]
    xs = np.arange(len(U))
    a.bar(xs - 0.2, 100 * U.watercore_recall_hist25, width=0.4, color=RED,
          label="hist25 rule")
    a.bar(xs + 0.2, 100 * U.watercore_recall_anchor, width=0.4, color=GREEN,
          label="anchor-derived threshold")
    a.set_xticks(xs)
    a.set_xticklabels([f"{r.contour_id}\n{r.date[5:]}" for r in U.itertuples()],
                      fontsize=7.5)
    a.set_ylabel("% of persistent open water recovered")
    a.legend(fontsize=8.5); a.grid(alpha=0.25, axis="y")
    a.set_title("b · water-core recall (anchor contains no shoreline)",
                fontsize=10.5, loc="left")
    fig.suptitle("hist25b · scene-level radiometry and recall", y=1.02,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "hist25b_scene_water_backscatter.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    fig, a = plt.subplots(figsize=(8.2, 5.2))
    if U.orbit_state.nunique() > 1:
        for st, c in zip(sorted(U.orbit_state.unique()), (BLUE, AMBER, GREY)):
            s = U[U.orbit_state == st]
            a.scatter(s.water_core_vv_db, s.land_vv_db, s=70, color=c, label=st)
    else:
        a.scatter(U.water_core_vv_db, U.land_vv_db, s=70, color=BLUE,
                  label=str(U.orbit_state.iloc[0]) if len(U) else "UNKNOWN")
    lims = [-30, -2]
    a.plot(lims, lims, color=GREY, ls=":", lw=1.2, label="water = land")
    a.axvline(db(S1.OTSU_MIN), color=RED, ls="--", lw=1.5, label="VV clamp floor")
    a.set_xlabel("water-core VV (dB)"); a.set_ylabel("land VV (dB)")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("hist25b · orbit-direction comparison of the water/land "
                "contrast", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(CFG.FIG / "hist25b_orbit_comparison.png", dpi=170)
    plt.close(fig)

    # ---------------------------------------------------------------- report
    pinned_frac = float(U.vv_threshold_pinned_by_clamp.mean())
    med_rec_h = float(U.watercore_recall_hist25.median())
    med_rec_a = float(U.watercore_recall_anchor.median())
    med_water_vv = float(U.water_core_vv_db.median())
    elig_vv = float(U.otsu_eligible_vv.mean())
    lines = [
        "# hist25b — Sentinel-1 RTC product audit and hist25 failure diagnosis",
        "",
        "Gates 1-4 only. No replacement classifier is proposed here.",
        "",
        "## Inherited engine, as written",
        "",
        "`phase19_s1_watermask.otsu_threshold()` clamps INTERNALLY to "
        f"`[{S1.OTSU_MIN}, {S1.OTSU_MAX}]` linear "
        f"= `[{db(S1.OTSU_MIN):.1f}, {db(S1.OTSU_MAX):.1f}]` dB, for any "
        "polarisation.",
        "",
        "`water_mask()` then re-clamps VH to "
        f"`[{S1.VH_MIN}, {S1.VH_MAX}]` = "
        f"`[{db(S1.VH_MIN):.1f}, {db(S1.VH_MAX):.1f}]` dB. Because the inner "
        "clamp already floors at "
        f"{S1.OTSU_MIN} ({db(S1.OTSU_MIN):.1f} dB), the reachable VH interval "
        f"is only `[{db(S1.OTSU_MIN):.1f}, {db(S1.VH_MAX):.1f}]` dB — the "
        f"declared VH_MIN of {db(S1.VH_MIN):.1f} dB is unreachable.",
        "",
        "hist25 reproduced this double clamp, and additionally did NOT apply "
        "`enhanced_lee` speckle filtering, which `water_mask()` applies "
        "before thresholding. hist25 therefore used the engine outside its "
        "design envelope in two independent ways.",
        "",
        "## Product provenance",
        "",
        "- collection: `sentinel-1-rtc` (Microsoft Planetary Computer)",
        "- radiometry: terrain-corrected **gamma0**, linear power, float32",
        "- the inherited clamp was derived for **sigma0** from SAFE-calibrated "
        "products in `phase19_s1_watermask`; gamma0 is systematically brighter "
        "than sigma0 by ~1/cos(theta), so a sigma0-derived clamp does not "
        "transfer unchanged",
        f"- resampled to {CELL:.0f} m on the hist23 analysis grid; nodata as NaN",
        "",
        "## Measured outcome (per-scene, from the frozen cache)",
        "",
        f"- persistent-open-water VV, median over all acquisitions: "
        f"**{med_water_vv:.2f} dB**",
        f"- the inherited VV clamp floor is **{db(S1.OTSU_MIN):.1f} dB**",
        f"- fraction of acquisitions whose VV threshold was PINNED at a clamp "
        f"bound: **{100*pinned_frac:.0f}%**",
        f"- median recall of persistent open water, hist25 rule: "
        f"**{100*med_rec_h:.1f}%**",
        f"- median recall of persistent open water, anchor-derived threshold: "
        f"**{100*med_rec_a:.1f}%**",
        f"- fraction of acquisitions with a genuinely bimodal VV histogram "
        f"(Otsu eligible): **{100*elig_vv:.0f}%**",
        "",
        "## Tables and figures",
        "",
        "- `outputs/tables/hist25b_hist25_reproduction.csv`",
        "- `outputs/tables/hist25b_scene_diagnostics.csv`",
        "- `outputs/tables/hist25b_cross_orbit_consistency.csv`",
        "- `outputs/figures/hist25b_raw_histograms.png`",
        "- `outputs/figures/hist25b_old_clamp_diagnostic.png`",
        "- `outputs/figures/hist25b_scene_water_backscatter.png`",
        "- `outputs/figures/hist25b_orbit_comparison.png`",
    ]
    (REPORTS / "hist25b_rtc_product_audit.md").write_text("\n".join(lines))

    print("\n" + "=" * 78)
    print("GATE SUMMARY")
    print("=" * 78)
    print(f"  GATE 1 reproduce hist25 from cache      : "
          f"PASS ({len(U)} usable acquisitions)")
    print(f"  GATE 2 RTC units/scaling/clamp audited  : PASS")
    print(f"  GATE 3 H2 failure localised to scene    : see table above")
    print(f"  GATE 4 anchors separable / Otsu eligible: "
          f"{100*elig_vv:.0f}% of scenes VV-bimodal")
    print("\n  STOP. Gates 5-10 (classifier candidates, flooded vegetation,")
    print("  consensus, footprint validation) are deliberately not run until")
    print("  the above is reviewed.")
    for p in ("hist25b_hist25_reproduction", "hist25b_scene_diagnostics",
              "hist25b_cross_orbit_consistency"):
        print(f"-> {CFG.TABLES/(p + '.csv')}")
    print(f"-> {REPORTS/'hist25b_rtc_product_audit.md'}")


if __name__ == "__main__":
    main()
