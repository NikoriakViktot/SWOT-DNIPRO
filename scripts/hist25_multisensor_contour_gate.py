#!/usr/bin/env python
"""HISTORICAL 25 — multi-sensor shoreline agreement gate.

hist23 built three independent pre-breach optical shorelines (H1/H2/H3) and
showed their nesting is geometrically sound (99.05/99.89/99.87% on the common
observed domain). hist24 then showed the geometry is not the problem: the
optical shoreline barely MOVES between levels (21 m over a 2.86 m span)
against an independently measured 181 m inward bias from the emergent-
vegetation belt (signal/bias = 0.12). NDWI/MNDWI detects OPEN water, and on
this reservoir the true waterline is held by flooded reed, not open water.

THIS SCRIPT DOES NOT REPLACE OPTICAL WITH RADAR. Sentinel-1 responds to
flooded vegetation through double-bounce, which is a different, complementary
failure mode (it can see through canopy that optical cannot see through, and
it can miss open water that is wind-roughened or frozen). The two sensors are
kept as two independent shoreline estimates per level, and AGREEMENT between
them -- not either one alone -- is what is allowed to constrain the DEM later.
This mirrors the "level of confidence" method of Chenier et al. 2019 (IJGI
8(1):48): several independent techniques are combined by where they agree,
not averaged blindly, and pixels below agreement are excluded rather than
guessed at.

WHAT IS NOT TRANSFERRED FROM THAT PAPER: their +/-1 m depth-agreement
tolerance. That number comes from a different sensor pair (WorldView-2 stereo
photogrammetry vs. multibeam sonar) at a different site and a different
physical quantity (vertical depth, not horizontal shoreline position). Using
it here would be exactly the kind of unexamined literal transfer this
project's history warns against. Section 1 below derives a tolerance from
THIS pipeline's own sensors and THIS reservoir's own measured quantities.

STRUCTURE
  hist23  : mask geometry is nested and consistent            (PASSED)
  hist24  : optical shoreline signal < vegetation bias         (FAILED, gated)
  hist25  : does an independent SAR contour change that verdict, when
            combined with optical through explicit, tolerance-based
            agreement rather than substitution?                (THIS SCRIPT)

This script builds contours, measures agreement, and recomputes the
signal/bias gate for optical-only, SAR-only and consensus. It does NOT fit a
DEM. If the consensus gate still fails, the shoreline-constrained branch
stops here exactly as hist24 already stops for optical alone.

Outputs
-------
outputs/tables/hist25_optical_sar_contour_agreement.csv
outputs/tables/hist25_water_level_consistency.csv
outputs/tables/hist25_signal_bias_comparison.csv
data/processed/bathymetry/hist25_contour_confidence.gpkg
data/processed/bathymetry/hist25_flooded_vegetation_candidates.gpkg
data/processed/bathymetry/hist25_three_level_consensus.gpkg
data/processed/bathymetry/hist25_agreement_distance_map.tif
outputs/figures/hist25_H{1,2,3}_optical_vs_sar.png
outputs/figures/hist25_disagreement_map.png
outputs/figures/hist25_confidence_map.png
outputs/figures/hist25_signal_bias_comparison.png
"""
from __future__ import annotations

import json
import math
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
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize as rio_rasterize
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import phase19_s1_watermask as S1          # existing Otsu engine, unchanged

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
CELL = 20.0                                 # identical grid to hist23
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
CACHE = CFG.BULK_ROOT / "data_swot/processed/bathymetry/s1_cache"
BATHY = ROOT / "data/processed/bathymetry"
FOOTPRINT_KM2 = 2192.0                      # independent P20 footprint at H1
LEVEL_ORDER = ["H3", "H2", "H1"]            # low -> high
PAD_DAYS = 12                               # search window padding around the
                                             # optical-verified dates for that level
COV_TARGET = 0.995                          # stop compositing once this much observed
MODERATE_MULT = 3.0                         # CONF_MODERATE band = [tau, MODERATE_MULT*tau)

# ============================================================================
# SECTION 1 — PROJECT-SPECIFIC AGREEMENT TOLERANCE (NOT Chenier et al.'s 1 m)
# ============================================================================
# Components, all independently sourced from THIS pipeline's own sensors and
# THIS reservoir's own measurements -- never from the literal accuracy figure
# of an unrelated stereo-photogrammetry study.
#
#   quantisation   : both sensors are resampled onto the SAME 20 m analysis
#                    grid used throughout hist23-25. A boundary's true
#                    position within a grid cell is unresolved; for a
#                    uniform distribution across one cell the standard
#                    deviation is cell/sqrt(12). Two independent
#                    resamplings (S2's own and S1's own) combine in
#                    quadrature.
#   geolocation    : documented MISSION geolocation-accuracy specifications,
#                    not a value measured in this repository. Flagged as an
#                    external input, exactly like hist24's SIG_EPSG9902.
#     S2 L1C/L2A   : <=12.5 m absolute geolocation (ESA mission requirements)
#     S1 RTC       : ~1 analysis-grid pixel residual co-registration after
#                    terrain correction on Planetary Computer's RTC product;
#                    taken conservatively as 6 m (mid-point of commonly
#                    reported 3-10 m absolute location error for S1 GRD/RTC).
#   level smear    : hist23 already measures, per contour, how much the
#                    composited dates disagree in level (level_spread /
#                    "spread_used"). A shoreline built from dates that are not
#                    perfectly level-flat is smeared horizontally by
#                    spread / slope. The slope used is hist24's own
#                    physically-diagnosed margin slope (~0.03 m/m) -- not the
#                    internally-inconsistent 0.14 m/m hist24 explicitly
#                    flagged as an artifact of the 20 m pixel, not geometry.
S2_GEOLOC_M = 12.5
S1_RTC_GEOLOC_M = 6.0
MARGIN_SLOPE = 0.03                          # hist24: physically diagnosed margin slope

# water-level uncertainty components, identical to hist24 (1 sigma, metres)
SIG_EPSG9902 = 0.068
SIG_GAUGE_READ = 0.01
SIG_SEICHE = 0.175


def compute_tau(spread_used_m: float) -> dict:
    quant = math.sqrt(2) * (CELL / math.sqrt(12))
    geoloc = math.sqrt(S2_GEOLOC_M ** 2 + S1_RTC_GEOLOC_M ** 2)
    base = math.sqrt(quant ** 2 + geoloc ** 2)
    smear = spread_used_m / MARGIN_SLOPE
    return dict(quant_m=quant, geoloc_m=geoloc, base_m=base,
                smear_m=smear, tau_m=base + smear)


# ============================================================================
# grid / STAC / caching helpers
# ============================================================================
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


def sas_token():
    return json.loads(urllib.request.urlopen(SAS, timeout=90).read())["token"]


def stac_search(start, end, bbox):
    q = {"collections": ["sentinel-1-rtc"], "bbox": bbox, "limit": 200,
         "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"}
    r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())["features"]


def poly_from_mask(m, tr, min_km2=0.5):
    geoms = [shp_shape(s) for s, v in rio_shapes(m.astype(np.uint8), mask=m,
                                                 transform=tr) if v == 1]
    if not geoms:
        return None
    p = unary_union(geoms).buffer(0)
    if p.geom_type == "MultiPolygon":
        keep = [q for q in p.geoms if q.area > min_km2 * 1e6]
        p = unary_union(keep) if keep else p
    return p


def polygonize_filtered(mask, tr, min_km2=0.02):
    """Polygonize a (possibly very fragmented) boolean raster WITHOUT
    unary_union. Small components are dropped by pixel count before
    vectorising, not after -- unary_union on tens of thousands of raw
    fragments (as a naive `poly_from_mask` on an unfiltered mask would do)
    is the actual cost, not the vectorisation itself. Returns a list of
    polygons, one per surviving connected component (not merged)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s * CELL ** 2 / 1e6 >= min_km2]
    if not keep:
        return []
    filt = np.isin(lab, keep)
    return [shp_shape(s) for s, v in rio_shapes(filt.astype(np.uint8), mask=filt,
                                                transform=tr) if v == 1]


def largest_component(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def dist_to_boundary_m(mask):
    """Unsigned distance (m) from every pixel to the nearest crossing of
    ``mask``'s boundary, at 20 m grid resolution."""
    din = ndimage.distance_transform_edt(mask)
    dout = ndimage.distance_transform_edt(~mask)
    return np.where(mask, din, dout) * CELL


def _cover(pts, cov, x0, y1, cell):
    j = np.array([int((q.x - x0) // cell) for q in pts])
    i = np.array([int((y1 - q.y) // cell) for q in pts])
    ok = (i >= 0) & (i < cov.shape[0]) & (j >= 0) & (j < cov.shape[1])
    out = np.zeros(len(pts), bool)
    out[ok] = cov[i[ok], j[ok]]
    return out


def one_way_distances(test_poly, ref_line, cov, x0, y1, cell, n=3000):
    """Distance from sampled points on test_poly's boundary to ref_line,
    restricted to points that fall on an observed pixel."""
    b = test_poly.boundary
    pts = [b.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
    keep = _cover(pts, cov, x0, y1, cell)
    return np.array([q.distance(ref_line) for q, k in zip(pts, keep) if k])


def boundary_pair_stats(poly_a, poly_b, obs_both, x0, y1, cell, n=3000):
    """Symmetric distance statistics between two polygon boundaries, on the
    domain observed by both. Returns None if too few samples survive."""
    d_ab = one_way_distances(poly_a, poly_b.boundary, obs_both, x0, y1, cell, n)
    d_ba = one_way_distances(poly_b, poly_a.boundary, obs_both, x0, y1, cell, n)
    if len(d_ab) < 50 or len(d_ba) < 50:
        return None
    pooled = np.concatenate([d_ab, d_ba])
    med = float(np.median(pooled))
    return dict(
        n_a=len(d_ab), n_b=len(d_ba),
        median_m=med,
        nmad_m=float(1.4826 * np.median(np.abs(pooled - med))),
        p90_m=float(np.percentile(pooled, 90)),
        p95_m=float(np.percentile(pooled, 95)),
        hausdorff_m=float(max(d_ab.max(), d_ba.max())),
    )


def nesting_stats(polys: dict, valids: dict, seq=LEVEL_ORDER, cell=CELL):
    """Pairwise 'is lower nested inside higher' on the domain both observed,
    same convention as hist23: percentage of the lower polygon's area that
    falls inside the higher polygon, restricted to their common domain."""
    common = None
    for cid in seq:
        common = valids[cid] if common is None else (common & valids[cid])
    rows = []
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            lo, hi = seq[i], seq[j]
            if polys.get(lo) is None or polys.get(hi) is None:
                continue
            tr = _GRID["tr"]; ny, nx = _GRID["ny"], _GRID["nx"]
            a = rio_rasterize([(polys[lo], 1)], out_shape=(ny, nx), transform=tr,
                              fill=0, dtype="uint8").astype(bool) & common
            b = rio_rasterize([(polys[hi], 1)], out_shape=(ny, nx), transform=tr,
                              fill=0, dtype="uint8").astype(bool) & common
            if a.sum() == 0:
                continue
            pct = 100.0 * float((a & b).sum()) / float(a.sum())
            rows.append(dict(lower=lo, higher=hi, pct_lower_inside_higher=pct,
                             common_km2=float(common.sum()) * cell ** 2 / 1e6))
    return pd.DataFrame(rows), common


# ============================================================================
# SECTION 2 (loading) — optical contours already built by hist23
# ============================================================================
def load_optical(inv: pd.DataFrame):
    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")
    poly, valid, water, dates_used = {}, {}, {}, {}
    for cid in LEVEL_ORDER:
        poly[cid] = gdf.loc[cid, "geometry"]
        ds = gdf.loc[cid, "dates"].split("|")
        dates_used[cid] = ds
        v = np.zeros((_GRID["ny"], _GRID["nx"]), bool)
        for d in ds:
            npz = ROOT / f"data/processed/bathymetry/contour_cache/wm_{d}_{CELL:.0f}m.npz"
            z = np.load(npz)
            v |= z["valid"]
        valid[cid] = v & _GRID["inside"]
        water[cid] = rio_rasterize([(poly[cid], 1)], out_shape=(_GRID["ny"], _GRID["nx"]),
                                   transform=_GRID["tr"], fill=0,
                                   dtype="uint8").astype(bool) & _GRID["inside"]
    return poly, valid, water, dates_used


# ============================================================================
# SECTION 2 (building) — SAR contours composited across each level's window
# ============================================================================
def sar_composite(cid, target_dates, tok, bbox):
    x0, y0, x1, y1 = _GRID["x0"], _GRID["y0"], _GRID["x1"], _GRID["y1"]
    nx, ny, tr, inside = _GRID["nx"], _GRID["ny"], _GRID["tr"], _GRID["inside"]
    tdates = pd.to_datetime(sorted(target_dates))
    start = (tdates.min() - pd.Timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
    end = (tdates.max() + pd.Timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
    feats = stac_search(start, end, bbox)
    rows = [dict(item_id=f["id"], date=f["properties"]["datetime"][:10],
                 relative_orbit=f["properties"].get("sat:relative_orbit"))
            for f in feats]
    if not rows:
        print(f"    {cid}: no Sentinel-1 RTC items in {start}..{end}")
        return None
    inv = pd.DataFrame(rows)
    acq = (inv.groupby(["date", "relative_orbit"]).agg(n=("item_id", "size"))
              .reset_index())
    acq["lag_days"] = acq.date.apply(
        lambda d: float(np.min(np.abs((pd.Timestamp(d) - tdates).days))))
    acq = acq.sort_values("lag_days")
    print(f"    {cid}: window {start}..{end}, {len(acq)} candidate acquisitions")

    water = np.zeros((ny, nx), bool)
    flood = np.zeros((ny, nx), bool)
    valid = np.zeros((ny, nx), bool)
    used = []
    for r in acq.itertuples():
        date, orb = r.date, int(r.relative_orbit)
        npzf = CACHE / f"s1_{cid}_{date}_orb{orb}_{CELL:.0f}m.npz"
        if npzf.exists():
            z = np.load(npzf)
            vv, vh, cov = z["vv"], z["vh"], z["cov"]
        else:
            items = [f for f in feats
                     if f["properties"]["datetime"][:10] == date
                     and f["properties"].get("sat:relative_orbit") == orb]
            vv = np.full((ny, nx), np.nan, np.float32)
            vh = np.full((ny, nx), np.nan, np.float32)
            for it in items:
                a = it["assets"]
                t0 = time.time()
                try:
                    for pol, dst in (("vv", vv), ("vh", vh)):
                        with rasterio.open(a[pol]["href"] + "?" + tok) as ds:
                            arr = ds.read(
                                1, window=from_bounds(x0, y0, x1, y1, ds.transform),
                                out_shape=(ny, nx), resampling=Resampling.average,
                                boundless=True, fill_value=np.nan)
                        m = np.isfinite(arr) & (arr > 0)
                        dst[m] = arr[m]
                except Exception as ex:
                    print(f"      {it['id'][:40]}: READ FAIL {type(ex).__name__}")
                    continue
                print(f"      {it['id'][:40]}  [{time.time()-t0:.0f}s]")
            cov = np.isfinite(vv) & np.isfinite(vh)
            np.savez_compressed(npzf, vv=vv, vh=vh, cov=cov)

        newpx = cov & ~valid & inside
        gain = float(newpx.sum()) / max(1, inside.sum())
        if newpx.any():
            sel = newpx
            fvv, fvh = vv[sel], vh[sel]
            if sel.sum() >= S1.OTSU_MIN_VALID:
                thr_vv = float(np.clip(S1.otsu_threshold(fvv), S1.OTSU_MIN, S1.OTSU_MAX))
                thr_vh = float(np.clip(S1.otsu_threshold(fvh), S1.VH_MIN, S1.VH_MAX))
                openw = np.zeros((ny, nx), bool)
                veg = np.zeros((ny, nx), bool)
                openw[sel] = (fvv < thr_vv) & (fvh < thr_vh)
                veg[sel] = (fvv < thr_vv) & (fvh >= thr_vh)
                water |= openw & newpx
                flood |= veg & newpx
        valid |= cov
        used.append(dict(date=date, orbit=orb, lag_days=r.lag_days, gain=gain))
        print(f"      {date} orbit {orb}: +{100*gain:5.1f}% newly observed "
              f"(cumulative {100*(valid&inside).sum()/inside.sum():5.1f}%), "
              f"{r.lag_days:.1f} d from a verified level date")
        if (valid & inside).sum() / inside.sum() > COV_TARGET:
            print(f"      -> coverage complete, remaining acquisitions not needed")
            break
    return dict(water=water & inside, flood=flood & inside, valid=valid & inside,
               used=pd.DataFrame(used))


# ============================================================================
# main
# ============================================================================
def main() -> None:
    global _GRID
    CACHE.mkdir(parents=True, exist_ok=True)
    # registry domain, not the retired P20 footprint: that stops 9.4 km short
    # on the east and 4.1 km on the north, and here it defined the GRID
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    _GRID = build_grid(fp)
    print(f"grid {_GRID['nx']}x{_GRID['ny']} at {CELL:.0f} m; footprint "
          f"{_GRID['inside'].sum()*CELL**2/1e6:,.0f} km2")

    inv = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv").set_index("contour_id")
    optical_poly, optical_valid, optical_water, optical_dates = load_optical(inv)

    # =============================================== SECTION 1 — TOLERANCE
    print("\n" + "=" * 78)
    print("SECTION 1 — PROJECT-SPECIFIC AGREEMENT TOLERANCE (not Chenier's 1 m)")
    print("=" * 78)
    tau = {}
    for cid in LEVEL_ORDER:
        spread = float(inv.loc[cid, "level_spread_within_group_m"])
        t = compute_tau(spread)
        tau[cid] = t["tau_m"]
        print(f"  {cid}: quant {t['quant_m']:.1f} m + geoloc {t['geoloc_m']:.1f} m "
              f"(RSS) = base {t['base_m']:.1f} m; level-spread {spread:.3f} m / "
              f"slope {MARGIN_SLOPE} m/m = smear {t['smear_m']:.1f} m "
              f"-> tau_{cid} = {t['tau_m']:.1f} m")

    # =============================================== SECTION 2 — SAR CONTOURS
    print("\n" + "=" * 78)
    print("SECTION 2 — SENTINEL-1 CONTOURS, COMPOSITED PER LEVEL")
    print("=" * 78)
    bbox = [32.2, 46.5, 35.5, 48.0]
    tok = sas_token()
    sar_water, sar_flood, sar_valid, sar_poly, sar_used = {}, {}, {}, {}, {}
    for cid in LEVEL_ORDER:
        r = sar_composite(cid, optical_dates[cid], tok, bbox)
        if r is None:
            sar_water[cid] = np.zeros_like(_GRID["inside"])
            sar_flood[cid] = np.zeros_like(_GRID["inside"])
            sar_valid[cid] = np.zeros_like(_GRID["inside"])
            sar_poly[cid] = None
            continue
        sar_water[cid], sar_flood[cid], sar_valid[cid] = r["water"], r["flood"], r["valid"]
        sar_used[cid] = r["used"]
        combined = largest_component((r["water"] | r["flood"]) & _GRID["inside"])
        sar_poly[cid] = poly_from_mask(combined, _GRID["tr"])
        obs = float(sar_valid[cid].sum()) / _GRID["inside"].sum()
        area = float(combined.sum()) * CELL ** 2 / 1e6
        print(f"    {cid}: SAR (open+flooded-veg) observed fraction {obs:.3f}, "
              f"area {area:,.0f} km2")

    # =============================================== SECTION 3+4 — AGREEMENT
    print("\n" + "=" * 78)
    print("SECTIONS 2-4 — OPTICAL/SAR DISTANCE, CONFIDENCE, FLOODED VEGETATION")
    print("=" * 78)
    agree_rows, conf_rasters, veg_polys, disagreement_rasters = [], {}, {}, {}
    for cid in LEVEL_ORDER:
        obs_o = optical_valid[cid]
        obs_s = sar_valid[cid]
        both = obs_o & obs_s
        only_o = obs_o & ~obs_s
        only_s = obs_s & ~obs_o
        nodata = _GRID["inside"] & ~obs_o & ~obs_s
        opt_w = optical_water[cid]
        sar_w = sar_water[cid] | sar_flood[cid]

        dist_opt = dist_to_boundary_m(opt_w)
        dist_sar = dist_to_boundary_m(sar_w)
        xor = both & (opt_w != sar_w)
        disagreement_m = np.zeros(_GRID["inside"].shape, np.float32)
        disagreement_m[xor] = np.minimum(dist_opt, dist_sar)[xor]
        disagreement_rasters[cid] = disagreement_m

        conf = np.zeros(_GRID["inside"].shape, np.uint8)   # 0 NO_DATA
        conf[nodata] = 0
        conf[only_o | only_s] = 3                          # CONF_LOW
        conf[both & ~xor] = 1                               # CONF_HIGH (agree)
        conf[xor & (disagreement_m < tau[cid])] = 1         # within tolerance
        conf[xor & (disagreement_m >= tau[cid]) &
             (disagreement_m < MODERATE_MULT * tau[cid])] = 2   # CONF_MODERATE
        conf[xor & (disagreement_m >= MODERATE_MULT * tau[cid])] = 4  # CONFLICT
        conf_rasters[cid] = conf

        flooded_veg = obs_o & obs_s & (~opt_w) & sar_flood[cid] & _GRID["inside"]
        veg_polys[cid] = polygonize_filtered(flooded_veg, _GRID["tr"], min_km2=0.005)
        veg_km2 = float(flooded_veg.sum()) * CELL ** 2 / 1e6

        stats = boundary_pair_stats(optical_poly[cid], sar_poly[cid], both,
                                    _GRID["x0"], _GRID["y1"], CELL) \
                if sar_poly.get(cid) is not None else None

        n_tot = float(_GRID["inside"].sum())
        row = dict(
            contour_id=cid, tau_m=tau[cid],
            observed_optical_frac=float(obs_o.sum()) / n_tot,
            observed_sar_frac=float(obs_s.sum()) / n_tot,
            observed_both_frac=float(both.sum()) / n_tot,
            pct_conf_high=100 * float((conf == 1).sum()) / n_tot,
            pct_conf_moderate=100 * float((conf == 2).sum()) / n_tot,
            pct_conf_low=100 * float((conf == 3).sum()) / n_tot,
            pct_conflict=100 * float((conf == 4).sum()) / n_tot,
            # NOT (conf == 0).sum(): conf default-initialises to 0 over the
            # WHOLE grid, not just the footprint, so that would count every
            # pixel of Ukraine outside the reservoir as "NO_DATA". `nodata`
            # is already restricted to _GRID["inside"].
            pct_no_data=100 * float(nodata.sum()) / n_tot,
            flooded_vegetation_candidate_km2=veg_km2,
        )
        if stats:
            row.update({f"boundary_{k}": v for k, v in stats.items()})
        agree_rows.append(row)
        print(f"\n  {cid}: HIGH {row['pct_conf_high']:.1f}%  MODERATE "
              f"{row['pct_conf_moderate']:.1f}%  LOW {row['pct_conf_low']:.1f}%  "
              f"CONFLICT {row['pct_conflict']:.1f}%  NO_DATA {row['pct_no_data']:.1f}%")
        print(f"      flooded-vegetation candidates (optical=dry, SAR=inundated): "
              f"{veg_km2:,.1f} km2")
        if stats:
            print(f"      optical<->SAR boundary distance: median "
                  f"{stats['median_m']:.0f} m, NMAD {stats['nmad_m']:.0f} m, "
                  f"p90 {stats['p90_m']:.0f} m, Hausdorff {stats['hausdorff_m']:.0f} m "
                  f"(tau = {tau[cid]:.0f} m)")

    AG = pd.DataFrame(agree_rows)
    AG.to_csv(CFG.TABLES / "hist25_optical_sar_contour_agreement.csv", index=False)

    # =============================================== SECTION 5 — NESTING
    print("\n" + "=" * 78)
    print("SECTION 5 — THREE-LEVEL NESTING, EACH SENSOR AND THE CONSENSUS")
    print("=" * 78)
    opt_nest, _ = nesting_stats(optical_poly, optical_valid)
    sar_nest, _ = nesting_stats(sar_poly, sar_valid)
    print("\n  optical nesting:")
    print(opt_nest.to_string(index=False) if len(opt_nest) else "    (insufficient data)")
    print("\n  SAR nesting:")
    print(sar_nest.to_string(index=False) if len(sar_nest) else "    (insufficient data)")

    # =============================================== SECTION 6 — CONSENSUS
    print("\n" + "=" * 78)
    print("SECTION 6 — CONSENSUS CONTOUR (HIGH + physically-supported MODERATE "
          "+ flooded-vegetation candidates; plain single-sensor LOW/CONFLICT "
          "excluded)")
    print("=" * 78)
    consensus_poly, consensus_valid = {}, {}
    for cid in LEVEL_ORDER:
        conf = conf_rasters[cid]
        opt_w = optical_water[cid]
        sar_w = sar_water[cid] | sar_flood[cid]
        support = ((conf == 1) & opt_w) | ((conf == 2) & (opt_w | sar_w))
        flooded_veg = optical_valid[cid] & sar_valid[cid] & (~opt_w) & \
                      sar_flood[cid] & _GRID["inside"]
        support = support | flooded_veg
        comp = largest_component(support & _GRID["inside"])
        consensus_poly[cid] = poly_from_mask(comp, _GRID["tr"])
        consensus_valid[cid] = optical_valid[cid] | sar_valid[cid]
        area = float(comp.sum()) * CELL ** 2 / 1e6
        print(f"  {cid}: consensus area {area:,.0f} km2 "
              f"(includes {float(flooded_veg.sum())*CELL**2/1e6:,.1f} km2 "
              f"flooded-vegetation-only support)")
    cons_nest, _ = nesting_stats(consensus_poly, consensus_valid)
    print("\n  consensus nesting:")
    print(cons_nest.to_string(index=False) if len(cons_nest) else "    (insufficient data)")

    # =============================================== SECTION 7 — SIGNAL/BIAS
    print("\n" + "=" * 78)
    print("SECTION 7 — SIGNAL/BIAS GATE: OPTICAL vs SAR vs CONSENSUS")
    print("=" * 78)

    def signal_bias(polys, valids, label):
        pairs = []
        for i in range(len(LEVEL_ORDER)):
            for j in range(i + 1, len(LEVEL_ORDER)):
                lo, hi = LEVEL_ORDER[i], LEVEL_ORDER[j]
                if polys.get(lo) is None or polys.get(hi) is None:
                    continue
                both = valids[lo] & valids[hi]
                st = boundary_pair_stats(polys[lo], polys[hi], both,
                                         _GRID["x0"], _GRID["y1"], CELL)
                if st is None:
                    continue
                dH = abs(float(inv.loc[hi, "H_evrf2019_m"]) -
                        float(inv.loc[lo, "H_evrf2019_m"]))
                pairs.append(dict(pair=f"{hi}-{lo}", dH=dH, med=st["median_m"]))
        if not pairs:
            return None
        P = pd.DataFrame(pairs)
        full = P.loc[P.dH.idxmax()]
        if polys.get("H1") is None:
            return None
        bias = one_way_distances(polys["H1"], fp.boundary, valids["H1"],
                                 _GRID["x0"], _GRID["y1"], CELL)
        if len(bias) < 50:
            return None
        bias_m = float(np.median(bias))
        signal_m = float(full.med)
        ratio = signal_m / bias_m if bias_m > 0 else float("inf")
        return dict(method=label, signal_m=signal_m, dH=float(full.dH),
                   bias_m=bias_m, signal_bias_ratio=ratio, pairs=P)

    sb_opt = signal_bias(optical_poly, optical_valid, "optical")
    sb_sar = signal_bias(sar_poly, sar_valid, "SAR")
    sb_con = signal_bias(consensus_poly, consensus_valid, "consensus")
    sb_rows = [r for r in (sb_opt, sb_sar, sb_con) if r is not None]
    if not sb_rows:
        raise SystemExit("SECTION 7 could not be computed for any method -- "
                         "insufficient jointly-observed shoreline samples. "
                         "Check the SAR composite coverage before re-running.")
    SB = pd.DataFrame([{k: v for k, v in r.items() if k != "pairs"} for r in sb_rows])
    SB.to_csv(CFG.TABLES / "hist25_signal_bias_comparison.csv", index=False)
    for r in sb_rows:
        print(f"  {r['method']:<10} signal {r['signal_m']:.0f} m over dH "
              f"{r['dH']:.2f} m, bias {r['bias_m']:.0f} m -> "
              f"signal/bias = {r['signal_bias_ratio']:.2f}")

    verdict = "FAIL"
    con_ratio = sb_con["signal_bias_ratio"] if sb_con else 0.0
    nesting_ok = (len(cons_nest) > 0 and
                 (cons_nest.pct_lower_inside_higher > 97).all())
    agreement_ok = (AG.pct_conf_high + AG.pct_conf_moderate).min() > 40 \
        if len(AG) else False
    if con_ratio > 1.0 and nesting_ok and agreement_ok:
        verdict = "PASS"
    elif sb_opt and con_ratio > sb_opt["signal_bias_ratio"]:
        verdict = "PARTIAL"
    print(f"\n  VERDICT: {verdict}")
    print(f"    consensus signal/bias > 1        : {con_ratio > 1.0} "
          f"({con_ratio:.2f})")
    print(f"    consensus nesting > 97% each pair : {nesting_ok}")
    print(f"    HIGH+MODERATE coverage > 40% each level: {agreement_ok}")
    if verdict == "FAIL":
        print("\n  Consensus does not clear the gate. Do not proceed to DEM "
              "fitting on these contours -- the canonical DEM stays as it is.")
    elif verdict == "PARTIAL":
        print("\n  Consensus improves on optical alone but does not clear "
              "signal/bias = 1 outright. Report as an open question, do not "
              "silently round up to PASS.")
    else:
        print("\n  Consensus clears the gate. Contours may proceed to "
              "hist26 as SOFT constraints (never hard), restricted to "
              "CONF_HIGH/CONF_MODERATE support only.")

    # =============================================== SECTION 8 — LEVEL TABLE
    sig_h = math.sqrt(SIG_EPSG9902 ** 2 + SIG_GAUGE_READ ** 2 + SIG_SEICHE ** 2)
    lvl_rows = []
    for cid in LEVEL_ORDER:
        lvl_rows.append(dict(
            contour_id=cid, H_evrf2019_m=float(inv.loc[cid, "H_evrf2019_m"]),
            vertical_datum="EVRF2019", H_sigma_m=sig_h,
            optical_dates="|".join(optical_dates[cid]),
            sar_dates="|".join(sar_used[cid].date) if cid in sar_used and
                len(sar_used[cid]) else "",
            optical_time_diff_days=0.0,
            sar_time_diff_days=float(sar_used[cid].lag_days.iloc[0])
                if cid in sar_used and len(sar_used[cid]) else float("nan"),
        ))
    LV = pd.DataFrame(lvl_rows)
    LV.to_csv(CFG.TABLES / "hist25_water_level_consistency.csv", index=False)

    # =============================================== OUTPUTS — vector/raster
    rows = []
    for cid in LEVEL_ORDER:
        if optical_poly[cid] is not None:
            rows.append(dict(contour_id=cid, sensor="optical",
                             geometry=optical_poly[cid]))
        if sar_poly.get(cid) is not None:
            rows.append(dict(contour_id=cid, sensor="SAR", geometry=sar_poly[cid]))
        if consensus_poly.get(cid) is not None:
            rows.append(dict(contour_id=cid, sensor="consensus",
                             geometry=consensus_poly[cid]))
    if rows:
        gpd.GeoDataFrame(rows, crs=CFG.CRS_METRIC).to_file(
            BATHY / "hist25_three_level_consensus.gpkg",
            layer="contours_by_sensor", driver="GPKG")

    conf_labels = {0: "NO_DATA", 1: "CONF_HIGH", 2: "CONF_MODERATE",
                  3: "CONF_LOW", 4: "CONFLICT"}
    # HIGH/NO_DATA dominate by area with a noisy, fragmented boundary (mostly
    # trivial background agreement far from any shoreline) -- coarser filter.
    # MODERATE/LOW/CONFLICT are the scientifically interesting minority
    # classes near the margin -- keep sub-hectare detail.
    conf_min_km2 = {0: 0.5, 1: 0.5, 2: 0.02, 3: 0.02, 4: 0.02}
    conf_rows = []
    for cid in LEVEL_ORDER:
        for code, name in conf_labels.items():
            # restrict to the footprint explicitly: conf==0 is also the
            # array's default value outside _GRID["inside"], which is not
            # "NO_DATA inside the reservoir" and must not be vectorised.
            m = (conf_rasters[cid] == code) & _GRID["inside"]
            for g in polygonize_filtered(m, _GRID["tr"], min_km2=conf_min_km2[code]):
                conf_rows.append(dict(contour_id=cid, confidence=name, geometry=g))
    if conf_rows:
        gpd.GeoDataFrame(conf_rows, crs=CFG.CRS_METRIC).to_file(
            BATHY / "hist25_contour_confidence.gpkg",
            layer="confidence_classes", driver="GPKG")

    veg_rows = [dict(contour_id=cid, geometry=g)
               for cid in LEVEL_ORDER for g in veg_polys.get(cid, [])]
    if veg_rows:
        gpd.GeoDataFrame(veg_rows, crs=CFG.CRS_METRIC).to_file(
            BATHY / "hist25_flooded_vegetation_candidates.gpkg",
            layer="flooded_vegetation_candidates", driver="GPKG")

    prof = dict(driver="GTiff", height=_GRID["ny"], width=_GRID["nx"], count=3,
               dtype="float32", crs=CFG.CRS_METRIC, transform=_GRID["tr"],
               nodata=-1.0, compress="deflate")
    with rasterio.open(BATHY / "hist25_agreement_distance_map.tif", "w", **prof) as ds:
        for k, cid in enumerate(LEVEL_ORDER, start=1):
            band = np.where(_GRID["inside"], disagreement_rasters[cid], -1.0)
            ds.write(band.astype(np.float32), k)
            ds.set_band_description(k, f"{cid}_disagreement_m")
    print(f"\n-> {(BATHY/'hist25_agreement_distance_map.tif').relative_to(ROOT)}")
    for n in ("hist25_three_level_consensus", "hist25_contour_confidence",
             "hist25_flooded_vegetation_candidates"):
        p = BATHY / f"{n}.gpkg"
        if p.exists():
            print(f"-> {p}")
    for n in ("hist25_optical_sar_contour_agreement",
             "hist25_water_level_consistency", "hist25_signal_bias_comparison"):
        print(f"-> {CFG.TABLES/(n+'.csv')}")

    # =============================================== FIGURES
    for cid in LEVEL_ORDER:
        fig, ax = plt.subplots(figsize=(7.2, 6.6))
        if optical_poly[cid] is not None:
            xs, ys = optical_poly[cid].exterior.xy if optical_poly[cid].geom_type == \
                "Polygon" else optical_poly[cid].geoms[0].exterior.xy
            ax.plot(np.array(xs)/1e3, np.array(ys)/1e3, color=RED, lw=1.6,
                   label="optical")
        if sar_poly.get(cid) is not None:
            g = sar_poly[cid]
            gs = g.geoms if g.geom_type == "MultiPolygon" else [g]
            for k, gg in enumerate(gs):
                xs, ys = gg.exterior.xy
                ax.plot(np.array(xs)/1e3, np.array(ys)/1e3, color=BLUE, lw=1.6,
                       label="SAR (open+flooded-veg)" if k == 0 else None)
        if consensus_poly.get(cid) is not None:
            g = consensus_poly[cid]
            gs = g.geoms if g.geom_type == "MultiPolygon" else [g]
            for k, gg in enumerate(gs):
                xs, ys = gg.exterior.xy
                ax.plot(np.array(xs)/1e3, np.array(ys)/1e3, color=GREEN, lw=1.2,
                       ls="--", label="consensus" if k == 0 else None)
        ax.set_aspect("equal"); ax.legend(fontsize=8.5)
        ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
        ax.set_title(f"{cid} — optical vs SAR shoreline "
                    f"(H={float(inv.loc[cid,'H_evrf2019_m']):.2f} m EVRF2019)",
                    fontsize=10.6, loc="left")
        fig.tight_layout()
        fig.savefig(CFG.FIG / f"hist25_{cid}_optical_vs_sar.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for a, cid in zip(ax, LEVEL_ORDER):
        d = np.where(_GRID["inside"], disagreement_rasters[cid], np.nan)
        im = a.imshow(d, cmap="magma", vmin=0, vmax=max(50.0, tau[cid]*3))
        a.set_title(f"{cid} disagreement (tau={tau[cid]:.0f} m)", fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.04, label="m")
    fig.suptitle("hist25 · optical-SAR shoreline disagreement width", y=1.02)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "hist25_disagreement_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    cmap = matplotlib.colors.ListedColormap(
        [GREY, GREEN, AMBER, BLUE, RED])       # NO_DATA/HIGH/MODERATE/LOW/CONFLICT
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for a, cid in zip(ax, LEVEL_ORDER):
        m = np.where(_GRID["inside"], conf_rasters[cid], np.nan)
        a.imshow(m, cmap=cmap, vmin=0, vmax=4)
        a.set_title(cid, fontsize=10); a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
              (GREY, GREEN, AMBER, BLUE, RED)]
    fig.legend(handles, ["NO_DATA", "CONF_HIGH", "CONF_MODERATE", "CONF_LOW",
                        "CONFLICT"], loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("hist25 · multi-sensor shoreline confidence classes", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(CFG.FIG / "hist25_confidence_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ys = np.arange(len(SB))
    ax.barh(ys, SB.signal_bias_ratio, color=[RED, BLUE, GREEN][:len(SB)])
    ax.axvline(1.0, color=INK, ls="--", lw=1.4, label="gate: signal/bias = 1")
    ax.set_yticks(ys); ax.set_yticklabels(SB.method)
    ax.set_xlabel("signal / bias")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="x")
    ax.set_title("hist25 · does consensus clear the signal/bias gate?",
                fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(CFG.FIG / "hist25_signal_bias_comparison.png", dpi=170)
    plt.close(fig)

    print("\n-> outputs/figures/hist25_H{1,2,3}_optical_vs_sar.png")
    print("-> outputs/figures/hist25_disagreement_map.png")
    print("-> outputs/figures/hist25_confidence_map.png")
    print("-> outputs/figures/hist25_signal_bias_comparison.png")


_GRID: dict = {}

if __name__ == "__main__":
    main()
