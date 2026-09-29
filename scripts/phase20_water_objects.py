#!/usr/bin/env python
"""Phase 20 STEP 1-3 — date-level water-body objects, fragmentation metrics, channel
connectivity, from the Sentinel-2 water masks (scripts/phase19_watermasks.py).

Objective (operator, 2026-09-08): the paper's PRIMARY hypothesis moves from "post-breach
longitudinal channel slope" (Phase 19, weak, n=4-6) to "loss of the continuous reservoir
water surface and emergence of a fragmented river-pond-exposed-bed system". Quantify it,
do not assume it.

Former-reservoir footprint = the largest connected water component of the 2023-06-05
Sentinel-2 mosaic (one day before the breach, pool still full), clipped to the reservoir
bounding box. This is data-driven; it does NOT use the SWORD line for the footprint
because SWORD's Dnipro nodes here span 16 reach_ids and are not spatially monotonic in
`chain_km` (islands split the stem). SWORD is used only as points for chainage
(KD-tree nearest node) and connectivity.

Outputs
-------
outputs/tables/water_body_objects.parquet         one row per (date, connected component)
outputs/tables/fragmentation_metrics_by_date.csv  N bodies, largest fraction, F-index, size classes
outputs/tables/channel_connectivity_by_date.csv   SWORD-topology connected-channel length / gaps
outputs/figure_data/P20_reservoir_footprint.geojson
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# F4 — cap BLAS/OpenMP threads BEFORE numpy is imported. On a loaded box the
# default (one thread per core, 77 here) thrashes cache and slows this down.
# No effect on results.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio import features, windows
from rasterio.enums import Resampling
from rasterio.merge import merge
from scipy import ndimage
from shapely.geometry import LineString, mapping, shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

#: Masks live on the bulk volume; the repo-disk path this pointed at no longer
#: exists (data moved to F). Same class of defect as fetch_sentinel.py's RAW.
MASKS = CFG.BULK_ROOT / "data_swot" / "processed" / "water_masks"
T = CFG.TABLES
M = CFG.CRS_METRIC                        # EPSG:32636
PIX_M = 20.0
# The reservoir extent comes from the registry, never from a literal. The
# tuple that stood here -- (33.35, 46.65, 35.20, 47.95) -- was the P20-era box
# that truncates ~87 km2 of real water at the eastern end (spatial_domains.yaml
# header; feedback-never-truncate-reservoir-east). Replaced 2026-09-16 as
# plan step B-2, AFTER the mask fix had been run on the old box (B-1), so the
# two effects on the fragmentation tables are attributable separately.
from swot_dnipro import spatial_domains as SD
# "Reservoir" = the registry's reservoir-only extent (CORE + TRANSITION, the two
# subzones hist23/hist26/k10e use), not the whole ZONE_1 whose bbox reaches
# Kherson and let the 2023-06-05 largest component grow to 2,337 km2 (+145).
def _reservoir_bbox_4326():
    from shapely.ops import unary_union as _uu
    u = _uu([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
             SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    x0, y0, x1, y1 = u.bounds
    xs, ys = _TF_TO_LL.transform([x0, x1, x1, x0], [y0, y0, y1, y1])
    return (min(xs), min(ys), max(xs), max(ys))
RES_BBOX_4326 = None   # resolved lazily in _bbox_m(); _TF_TO_LL is defined below
RES_CHAIN_MIN, RES_CHAIN_MAX = 0.0, 240.0
MMU_KM2 = 0.05
MMU_KM2_ALT = (0.02, 0.10)
COVERAGE_ACCEPTABLE = 0.50   # footprint_observed_fraction QC bands
COVERAGE_HIGH = 0.80
CHANNEL_TOUCH_M = 750.0
GAP_TOL_KM = 5.0
BIN_KM = 2.0
FOOTPRINT_DATE = "2023-06-05"

_TF_TO_M = pyproj.Transformer.from_crs("EPSG:4326", M, always_xy=True)
_TF_TO_LL = pyproj.Transformer.from_crs(M, "EPSG:4326", always_xy=True)


def sword_nodes():
    # was `CFG.CFG.BULK_ROOT` -- a doubled attribute that raised AttributeError
    # on every call, so this script could not have run in its committed state
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    ch = ch[(ch.chain_km >= RES_CHAIN_MIN - 5) & (ch.chain_km <= RES_CHAIN_MAX + 5)].copy()
    ch["x"], ch["y"] = _TF_TO_M.transform(ch.lon.values, ch.lat.values)
    return ch.sort_values("chain_km").reset_index(drop=True)


def date_mosaic(files):
    srcs = [rasterio.open(f) for f in files]
    arr, transform = merge(srcs, nodata=255, resampling=Resampling.nearest)
    for s in srcs:
        s.close()
    return arr[0], transform


def build_footprint(wm) -> "shapely.geometry.base.BaseGeometry":
    g = wm[wm.date == FOOTPRINT_DATE]
    files = [MASKS / f"{n}_water.tif" for n in g.name if (MASKS / f"{n}_water.tif").exists()]
    mos, transform = date_mosaic(files)
    x0, y0, x1, y1 = _bbox_m()
    H, W = mos.shape
    ext = features.geometry_mask  # noqa (silence linters)
    # bbox clip in pixel space
    inv = ~transform
    c0, r1 = inv * (x0, y0); c1, r0 = inv * (x1, y1)
    r0, r1 = sorted((int(r0), int(r1))); c0, c1 = sorted((int(c0), int(c1)))
    r0, c0 = max(r0, 0), max(c0, 0); r1, c1 = min(r1, H), min(c1, W)
    water = np.zeros_like(mos, bool)
    water[r0:r1, c0:c1] = mos[r0:r1, c0:c1] == 1
    lab, n = ndimage.label(water, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    main = int(sizes.argmax())
    fp = ndimage.binary_dilation(lab == main, iterations=10)      # +200 m
    polys = [shape(gm) for gm, v in features.shapes(fp.astype("uint8"), mask=fp,
                                                    transform=transform) if v == 1]
    geom = unary_union(polys)
    area = geom.area / 1e6
    print(f"footprint from {FOOTPRINT_DATE}: {area:,.0f} km2  (largest CC of the full pool)")
    import json
    (CFG.FIGDATA / "P20_reservoir_footprint.geojson").write_text(json.dumps({
        "type": "Feature", "properties": {"date": FOOTPRINT_DATE, "area_km2": area,
                                          "crs": M}, "geometry": mapping(geom)}))
    return geom


def _bbox_m():
    lo, la, hi, ha = RES_BBOX_4326 or _reservoir_bbox_4326()
    xs, ys = _TF_TO_M.transform([lo, hi, hi, lo], [la, la, ha, ha])
    return min(xs), min(ys), max(xs), max(ys)


def coverage_class(frac: float) -> str:
    """QC class for how much of the former pool a date actually observes."""
    if not np.isfinite(frac):
        return "PARTIAL"
    if frac >= COVERAGE_HIGH:
        return "HIGH"
    if frac >= COVERAGE_ACCEPTABLE:
        return "ACCEPTABLE"
    return "PARTIAL"


def objects_for_date(d, per, files, footprint, nodes, chain_tree=None):
    mos, transform = date_mosaic(files)
    H, W = mos.shape
    fp = features.rasterize([(footprint, 1)], out_shape=(H, W), transform=transform,
                            fill=0, dtype="uint8").astype(bool)

    # ---- coverage QC -------------------------------------------------------
    # The mosaic spans only THIS date's tiles, so the footprint must be measured
    # against its own fixed total area, never against the in-mosaic remainder.
    # A date that sees 8 % of the pool cannot be compared with one that sees all
    # of it; every F-metric below is conditional on this number.
    valid = mos != 255                          # 255 = nodata in the merged mosaic
    total_aoi_km2 = footprint.area / 1e6
    total_aoi_pixels = int(round(total_aoi_km2 * 1e6 / PIX_M ** 2))
    valid_pixels = int((valid & fp).sum())
    observed_aoi_km2 = valid_pixels * PIX_M ** 2 / 1e6
    frac = observed_aoi_km2 / total_aoi_km2 if total_aoi_km2 else np.nan
    cov = {"date": d, "period": per,
           "n_tiles": len(files),
           "total_aoi_km2": total_aoi_km2,
           "total_aoi_pixels": total_aoi_pixels,
           "valid_pixels": valid_pixels,
           "observed_aoi_km2": observed_aoi_km2,
           "footprint_observed_fraction": frac,
           "coverage_class": coverage_class(frac)}

    water = (mos == 1) & fp
    lab, n = ndimage.label(water, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(), 0, transform, lab, valid, cov

    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    # main component = water nearest the dam
    dx, dy = _TF_TO_M.transform(*CFG.KAKHOVKA_DAM)
    inv = ~transform
    dcol, drow = inv * (dx, dy)
    # F3 — one np.nonzero + one argmin (was: full scan with argmin computed twice).
    # np.nonzero is row-major like np.where, so ties break identically.
    ys, xs = np.nonzero(lab)
    if ys.size:
        _j = np.argmin((xs - dcol) ** 2 + (ys - drow) ** 2)
        main_id = int(lab[ys[_j], xs[_j]])
    else:
        main_id = 0

    # F1 — bounding box per label, so no step below ever scans the full raster.
    # ndimage.find_objects returns one slice tuple per label (index k-1) and costs
    # ~0.1 s for 50k labels; `lab == k` over the whole 110 Mpx mosaic was the
    # single dominant cost of this function.
    slices = ndimage.find_objects(lab)

    nx, ny = nodes.x.values, nodes.y.values
    rows = []
    for k in range(1, n + 1):
        npix = int(sizes[k])
        if npix < 9:
            continue
        sl = slices[k - 1]
        if sl is None:
            continue
        r0, c0 = sl[0].start, sl[1].start
        sub = lab[sl] == k                       # component-sized, not raster-sized
        sub_transform = windows.transform(
            windows.Window.from_slices(sl[0], sl[1]), transform)
        area_km2 = npix * PIX_M ** 2 / 1e6
        polys = [shape(g) for g, v in features.shapes(sub.astype("uint8"),
                 mask=sub, transform=sub_transform) if v == 1]
        geom = unary_union(polys)
        mrr = geom.minimum_rotated_rectangle
        try:
            mx, my = mrr.exterior.coords.xy
            e = [np.hypot(mx[i + 1] - mx[i], my[i + 1] - my[i]) for i in range(4)]
            major, minor = max(e), max(min(e), 1.0)
        except Exception:
            major = minor = np.sqrt(geom.area)
        yy, xx = np.nonzero(sub)
        yy = yy + r0                             # back to global pixel coordinates
        xx = xx + c0
        step = max(1, len(xx) // 400)
        pmx, pmy = transform * (xx[::step] + 0.5, yy[::step] + 0.5)
        lon, lat = _TF_TO_LL.transform(pmx, pmy)
        cch, cdist, _, _ = SW.assign_chainage(lon, lat, nodes, tree=chain_tree)
        cch = cch[np.isfinite(cch)]
        # nearest SWORD node distance for this body
        cx, cy = geom.centroid.x, geom.centroid.y
        d_sword = np.sqrt(((nx - cx) ** 2 + (ny - cy) ** 2).min()) / 1000.0
        rows.append({
            "date": d, "period": per, "comp_id": k, "n_pix": npix, "area_km2": area_km2,
            "perimeter_m": geom.length, "centroid_x": cx, "centroid_y": cy,
            "major_axis_m": major, "minor_axis_m": minor, "elongation": major / minor,
            "compactness": 4 * np.pi * geom.area / max(geom.length ** 2, 1.0),
            "dist_to_sword_km": d_sword,
            "on_main_channel": d_sword * 1000 < CHANNEL_TOUCH_M,
            "is_main_component": k == main_id,
            "chain_min_km": float(cch.min()) if cch.size else np.nan,
            "chain_max_km": float(cch.max()) if cch.size else np.nan,
            "chain_span_km": float(cch.max() - cch.min()) if cch.size else np.nan,
            "wkt": geom.wkt if area_km2 > MMU_KM2 else "",
        })
    df = pd.DataFrame(rows)
    df["is_noise"] = df.area_km2 < MMU_KM2
    for _k in ("footprint_observed_fraction", "coverage_class"):
        df[_k] = cov[_k]          # QC travels with every object row
    return df, main_id, transform, lab, valid, cov


def _reach_from_dam(flags):
    """Contiguous run of True 2 km bins measured from the dam, tolerating gaps
    up to GAP_TOL_KM. Returns (reach_km, max_gap_km)."""
    reach = gap = max_gap = 0.0
    for i, c in enumerate(flags):
        if c:
            reach = (i + 1) * BIN_KM; gap = 0.0
        else:
            gap += BIN_KM; max_gap = max(max_gap, gap)
            if gap > GAP_TOL_KM:
                break
    return reach, max_gap


def connectivity(d, lab, main_id, transform, nodes, valid=None):
    """Connected Dnipro length from the dam, plus how much channel the scene could
    have seen at all.

    A date whose tiles cover only part of the pool can only ever report a short
    connected length, so the absolute kilometres are not comparable between dates.
    `connected_channel_fraction` = connected / observable is, and is the number to
    analyse; the absolute lengths are kept for continuity.
    """
    nd = nodes[(nodes.chain_km >= RES_CHAIN_MIN) & (nodes.chain_km <= RES_CHAIN_MAX)]
    inv = ~transform
    col, row = inv * (nd.x.values, nd.y.values)
    col = np.floor(col).astype(int); row = np.floor(row).astype(int)
    H, W = lab.shape
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    rr = np.clip(row, 0, H - 1); cc = np.clip(col, 0, W - 1)

    # observable = node falls inside this date's mosaic AND on a non-nodata pixel
    obs = np.zeros(len(nd), bool)
    if valid is None:
        obs[ok] = True
    else:
        obs[ok] = valid[rr[ok], cc[ok]]

    comp = np.zeros(len(nd), int)
    comp[ok] = lab[rr[ok], cc[ok]]
    nd = nd.assign(on_water=comp > 0, in_main=(comp == main_id) & (main_id != 0),
                   observable=obs)

    bins = np.arange(RES_CHAIN_MIN, RES_CHAIN_MAX + BIN_KM, BIN_KM)
    idx = [(nd.chain_km >= b) & (nd.chain_km < b + BIN_KM) for b in bins[:-1]]
    conn_bins = np.array([nd[m].in_main.any() for m in idx])
    obs_bins = np.array([nd[m].observable.any() for m in idx])

    reach, max_gap = _reach_from_dam(conn_bins)
    observable_km, _ = _reach_from_dam(obs_bins)
    return {"date": d,
            "connected_channel_len_km": reach,
            "observable_channel_len_km": observable_km,
            "connected_channel_fraction": (reach / observable_km) if observable_km else np.nan,
            "frac_reservoir_connected": reach / (RES_CHAIN_MAX - RES_CHAIN_MIN),
            "max_gap_km": max_gap,
            "n_swrd_observable": int(nd.observable.sum()),
            "n_swrd_on_water": int(nd.on_water.sum()),
            "n_swrd_in_main": int(nd.in_main.sum())}


def main() -> None:
    wm = pd.read_csv(T / "water_mask_summary.csv")
    wm["dt"] = pd.to_datetime(wm["sensing_time"], format="%Y%m%dT%H%M%S")
    wm["date"] = wm["dt"].dt.strftime("%Y-%m-%d")
    wm["period"] = np.where(wm.dt < "2023-06-06", "PRE_BREACH",
                            np.where(wm.dt < "2023-09-01", "DRAWDOWN", "POST_BREACH"))

    nodes = sword_nodes()
    # F2 — the chainage KD-tree depends only on `nodes`; build it once here
    # instead of once per connected component (was ~8.8k rebuilds per date).
    chain_tree = SW.build_chainage_tree(nodes)
    footprint = build_footprint(wm)
    fp_area = footprint.area / 1e6

    all_obj, frag, conn = [], [], []
    for (d, per), g in wm.groupby(["date", "period"]):
        files = [MASKS / f"{n}_water.tif" for n in g.name if (MASKS / f"{n}_water.tif").exists()]
        if not files:
            continue
        df, main_id, transform, lab, valid, cov = objects_for_date(
            d, per, files, footprint, nodes, chain_tree=chain_tree)
        # A date with no component >= 9 px inside the footprint is still an
        # observation (usually a partial swath) and is kept, flagged by coverage.
        if not df.empty:
            all_obj.append(df)
        if df.empty:
            df = pd.DataFrame({"area_km2": pd.Series(dtype=float),
                               "is_noise": pd.Series(dtype=bool)})
        real = df[~df.is_noise]
        tot = real.area_km2.sum() if len(real) else 0.0
        largest = real.area_km2.max() if len(real) else 0.0
        crow = connectivity(d, lab, main_id, transform, nodes, valid=valid)
        conn.append(crow)
        row = {"date": d, "period": per, "n_water_bodies": len(real),
               "total_water_area_km2": tot, "largest_component_area_km2": largest,
               "largest_component_fraction": largest / tot if tot else np.nan,
               "fragmentation_index": 1 - largest / tot if tot else np.nan,
               "footprint_water_fraction": tot / fp_area,
               "n_bodies_gt_0p1": int((real.area_km2 > 0.1).sum()),
               "n_bodies_gt_1": int((real.area_km2 > 1).sum()),
               "n_bodies_gt_5": int((real.area_km2 > 5).sum())}
        for mm in MMU_KM2_ALT:
            row[f"n_bodies_mmu_{mm}"] = int((df.area_km2 >= mm).sum()) if not df.empty else 0
        # coverage QC is a first-class part of every date row: no F-metric below
        # is interpretable without it.
        row.update({k: v for k, v in cov.items() if k not in ("date", "period")})
        frag.append(row)
        print(f"  {d} {per:11s} cov={cov['footprint_observed_fraction']:5.2f} "
              f"{cov['coverage_class']:<10s} bodies={len(real):3d} tot={tot:7.1f} km2 "
              f"largest={largest:7.1f} ({(row['largest_component_fraction'] or 0)*100:4.0f}%) "
              f"F={row['fragmentation_index']:.2f} "
              f"conn={crow['connected_channel_len_km']:5.0f}/"
              f"{crow['observable_channel_len_km']:.0f}km "
              f"({crow['connected_channel_fraction']:.2f})", flush=True)

    pd.concat(all_obj, ignore_index=True).to_parquet(T / "water_body_objects.parquet", index=False)
    fr = pd.DataFrame(frag).sort_values("date")
    fr.to_csv(T / "fragmentation_metrics_by_date.csv", index=False)
    fr.to_csv(CFG.FIGDATA / "P20_fragmentation.csv", index=False)
    cn = pd.DataFrame(conn).sort_values("date")
    cn.to_csv(T / "channel_connectivity_by_date.csv", index=False)
    cn.to_csv(CFG.FIGDATA / "P20_connectivity.csv", index=False)

    print(f"\n-> {T/'fragmentation_metrics_by_date.csv'} ({len(fr)} dates)")
    print("\n=== coverage classes ===")
    print(fr.groupby(["period", "coverage_class"]).size().to_string())
    print("\n=== by period x coverage (date = unit) ===")
    mg = fr.merge(cn, on="date")
    for lo, tag in ((0.0, "ALL"), (COVERAGE_ACCEPTABLE, ">=0.50"), (COVERAGE_HIGH, ">=0.80")):
        s = mg[mg.footprint_observed_fraction >= lo]
        print(f"\n-- coverage {tag} (n={len(s)}) --")
        print(s.groupby("period").agg(
            n_dates=("date", "nunique"),
            med_bodies=("n_water_bodies", "median"),
            med_largest_frac=("largest_component_fraction", "median"),
            med_F=("fragmentation_index", "median"),
            med_conn_km=("connected_channel_len_km", "median"),
            med_conn_frac=("connected_channel_fraction", "median")).to_string())


if __name__ == "__main__":
    main()
