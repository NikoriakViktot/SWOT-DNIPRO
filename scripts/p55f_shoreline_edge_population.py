#!/usr/bin/env python
"""P55f -- do the residual source-2 errors form a SHORELINE-EDGE population? DIAGNOSIS ONLY, nothing is written to a product.

After the p55e admissibility repair the source-2 tail is entirely inside the registry pool polygon and concentrated near its
inner edge: 676 night-ICESat-2 segments with hist20 confidence 3 within 250 m of the boundary carry a 2.96 % tail while the
interior carries 0.04 %. The open question is deliberately NOT "why did p55 admit a bed here?" -- p55 admitted it because the
registry polygon says the cell is inside the water body, which is the correct positive criterion. The question is:

    do these points belong to hist20's BATHYMETRIC DOMAIN at all,
    or does the polygon formally enclose the shoreline bench, after which kriging quite legitimately interpolates a bed there?

Those are different defects with different repairs -- a domain/conditioning change in hist20 versus a solver change -- so the
axis that separates them has to be measured, not assumed. Five candidate axes are stratified against the same tail definition:

    distance to the polygon boundary     is it an edge effect at all
    FABDEM - the pool water surface      the decisive one if the tail sits in a narrow band where the ground IS at full pool
    FABDEM slope                         bank steepness
    local relief over 200 m              bank relief
    distance to the nearest sounding     the support axis, already known to be adequate here (100+ soundings within 2 km)

Each axis is reported as quintiles (n, tail share, bias, RMSE) and as a rank AUC for predicting tail membership, with the
orientation stated explicitly. A covariate that separates the tail while support does not is evidence for a shoreline-aware
admissibility domain rather than for a different interpolator.

Note on evidence strength (user, 2026-09-20): `share_fabdem_above_ws = 0` after the new gate partly follows from the gate
itself and is only a sanity check. The independent evidence is the ICESat-2 tail and its monotone rise towards the boundary,
which is what this script measures.
Outputs: outputs/tables/p55f_edge_axes_quintiles.csv, p55f_edge_axis_separation.csv, p55f_edge_band_crosstab.csv,
         outputs/tables/p55f_edge_band_area.csv
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy.spatial import cKDTree
from shapely import points as _pts, contains as _contains
from shapely.geometry import MultiLineString

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

spec = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P63)
P57 = P63.P57

SEAM = CFG.BULK_ROOT / "dem_seamless"
CONF = ROOT / "outputs/rasters/kakhovka_bed_confidence_class_250m.tif"
DIST_SND = ROOT / "outputs/rasters/kakhovka_bed_dist_to_sounding_250m.tif"
POOL_POLY = SD.load_utm("reservoir_full_pool_prebreach")
BOUNDARY_STEP_M = 10.0
TAIL_M = -5.0                     # the tail definition used throughout p63/p55e
INTERIOR_M = 1000.0               # "well inside": used only to MEASURE the pool water surface, not as a gate
EDGE_BAND_M = 250.0               # the belt the residual tail lives in, from p55e's distance strata
WS_BAND_M = 1.0                   # |FABDEM - pool water surface| for "the ground IS at full pool"
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")


def sample_at(path: Path, x, y):
    out = np.full(len(x), np.nan)
    if not Path(path).exists():
        return out
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("f8")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        c, r = ~ds.transform * (x, y)
        c = np.floor(c).astype(int); r = np.floor(r).astype(int)
        ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
        out[ok] = a[r[ok], c[ok]]
    return out


def boundary_tree():
    b = POOL_POLY.boundary
    lines = list(b.geoms) if isinstance(b, MultiLineString) else [b]
    pts = []
    for ln in lines:
        n = max(int(np.ceil(ln.length / BOUNDARY_STEP_M)), 2)
        pts.append(np.array([[p.x, p.y] for p in (ln.interpolate(t) for t in np.linspace(0.0, ln.length, n))]))
    return cKDTree(np.vstack(pts))


def terrain_axes(V):
    """FABDEM slope and 200 m relief at each point, computed on each zone's own 20 m frame."""
    V["fabdem_slope_deg"] = np.nan; V["fabdem_relief_200m_m"] = np.nan
    for zn in ZONES:
        m = (V.zone == zn).values
        if not m.any():
            continue
        p = CFG.BULK_ROOT / "terrain" / zn / "fabdem_evrf2019_20m.tif"
        with rasterio.open(p) as ds:
            a = ds.read(1).astype("f4")
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            tr, cell = ds.transform, abs(ds.transform.a)
        filled = np.nan_to_num(a, nan=float(np.nanmedian(a)))
        gy, gx = np.gradient(filled, cell, cell)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("f4"); del gy, gx, filled
        c, r = ~tr * (V.x.values[m], V.y.values[m])
        c = np.floor(c).astype(int); r = np.floor(r).astype(int)
        ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
        sl = np.full(m.sum(), np.nan); rel = np.full(m.sum(), np.nan)
        ri, ci = r[ok], c[ok]
        sl[ok] = slope[ri, ci]
        k = 5                                          # +/- 5 cells = +/- 100 m
        for j, (rr, cc) in enumerate(zip(ri, ci)):
            w = a[max(rr - k, 0):rr + k + 1, max(cc - k, 0):cc + k + 1]
            if np.isfinite(w).any():
                rel[np.flatnonzero(ok)[j]] = float(np.nanmax(w) - np.nanmin(w))
        V.loc[m, "fabdem_slope_deg"] = sl; V.loc[m, "fabdem_relief_200m_m"] = rel
        del a, slope
        print(f"  terrain axes on {zn}: {int(m.sum()):,} points", flush=True)
    return V


def build():
    tree = boundary_tree()
    V = P63.build_set_C()
    V = V[np.isfinite(V.res) & (V.src == 2)].copy()
    x = V.x.values.astype("f8"); y = V.y.values.astype("f8")
    V["inside_pool"] = _contains(POOL_POLY, _pts(x, y))
    V["dist_to_boundary_m"] = tree.query(np.c_[x, y], k=1)[0]
    V["hist20_confidence"] = sample_at(CONF, x, y)
    V["dist_to_sounding_m"] = sample_at(DIST_SND, x, y)
    V = terrain_axes(V)
    # The pool water surface AS FABDEM RECORDS IT, measured on the interior of the same population rather than assumed.
    # This is a reference level, not a gate: nothing is admitted or rejected by it here.
    interior = V.dist_to_boundary_m > INTERIOR_M
    ws = float(np.nanmedian(V.fab[interior])) if interior.any() else np.nan
    V["fab_minus_pool_ws_m"] = V.fab - ws
    V["is_tail"] = V.res < TAIL_M   # never name it `tail`: it shadows DataFrame.tail
    return V, ws


def quintiles(V, axes):
    rows = []
    for name, col in axes:
        v = V[col].values
        g = V[np.isfinite(v)]
        if len(g) < 50:
            continue
        q = pd.qcut(g[col], 5, duplicates="drop")
        for lab, sub in g.groupby(q, observed=True):
            rows.append(dict(axis=name, bin=f"{lab.left:.2f} .. {lab.right:.2f}", n_points=len(sub),
                             tail_share=round(float(sub.is_tail.mean()), 4), n_tail=int(sub.is_tail.sum()),
                             bias=round(float(sub.res.mean()), 3), RMSE=round(float(np.sqrt((sub.res ** 2).mean())), 3),
                             worst=round(float(sub.res.min()), 2)))
    return pd.DataFrame(rows)


def separation(V, axes):
    """Rank AUC of each axis for predicting tail membership. ORIENTATION: AUC > 0.5 means a HIGHER value of the axis makes a
    tail point MORE likely. AUC 0.5 is no separation. `sep` is |AUC - 0.5| * 2, so 0 = none and 1 = complete."""
    from scipy import stats
    rows = []
    t = V.is_tail.values
    for name, col in axes:
        v = V[col].values
        ok = np.isfinite(v)
        a, b = v[ok & t], v[ok & ~t]
        if len(a) < 10 or len(b) < 10:
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = float(u.statistic) / (len(a) * len(b))
        rows.append(dict(axis=name, n_tail=len(a), n_rest=len(b), auc_higher_value_means_tail=round(auc, 3),
                         sep=round(abs(auc - 0.5) * 2, 3), p_value=f"{u.pvalue:.2e}",
                         median_tail=round(float(np.median(a)), 2), median_rest=round(float(np.median(b)), 2)))
    return pd.DataFrame(rows).sort_values("sep", ascending=False)


def band_crosstab(V, ws):
    """The decisive test. Inside the edge belt, split by whether the GROUND is at the full-pool level."""
    rows = []
    edge = V.dist_to_boundary_m <= EDGE_BAND_M
    at_ws = np.abs(V.fab_minus_pool_ws_m) <= WS_BAND_M
    for lab, m in (("edge belt (<= 250 m) & ground AT full pool (|FABDEM - ws| <= 1 m)", edge & at_ws),
                   ("edge belt (<= 250 m) & ground NOT at full pool", edge & ~at_ws),
                   ("interior (> 250 m) & ground AT full pool", ~edge & at_ws),
                   ("interior (> 250 m) & ground NOT at full pool", ~edge & ~at_ws)):
        g = V[m.values]
        if not len(g):
            continue
        rows.append(dict(stratum=lab, n_points=len(g), n_tail=int(g.is_tail.sum()),
                         tail_share=round(float(g.is_tail.mean()), 4), bias=round(float(g.res.mean()), 3),
                         RMSE=round(float(np.sqrt((g.res ** 2).mean())), 3), worst=round(float(g.res.min()), 2),
                         fabdem_tail_share=round(float((g.fab_res < TAIL_M).mean()), 4),
                         dist_to_sounding_p50=round(float(np.nanmedian(g.dist_to_sounding_m)), 1),
                         slope_p50=round(float(np.nanmedian(g.fabdem_slope_deg)), 2)))
    # the same split restricted to the population the user named
    c3 = V[(V.hist20_confidence == 3) & edge]
    for lab, m in (("confidence 3 & edge belt & ground AT full pool", np.abs(c3.fab_minus_pool_ws_m) <= WS_BAND_M),
                   ("confidence 3 & edge belt & ground NOT at full pool", np.abs(c3.fab_minus_pool_ws_m) > WS_BAND_M)):
        g = c3[m.values]
        if not len(g):
            continue
        rows.append(dict(stratum=lab, n_points=len(g), n_tail=int(g.is_tail.sum()),
                         tail_share=round(float(g.is_tail.mean()), 4), bias=round(float(g.res.mean()), 3),
                         RMSE=round(float(np.sqrt((g.res ** 2).mean())), 3), worst=round(float(g.res.min()), 2),
                         fabdem_tail_share=round(float((g.fab_res < TAIL_M).mean()), 4),
                         dist_to_sounding_p50=round(float(np.nanmedian(g.dist_to_sounding_m)), 1),
                         slope_p50=round(float(np.nanmedian(g.fabdem_slope_deg)), 2)))
    return pd.DataFrame(rows)


def band_area(ws):
    """How much source-2 AREA sits in the shoreline band, i.e. what a shoreline-aware domain would actually cost."""
    tree = boundary_tree()
    rows = []
    for zone in ZONES:
        sp = SEAM / f"{zone}_dem_source_20m.tif"
        if not sp.exists():
            continue
        with rasterio.open(sp) as ds:
            src = ds.read(1); tr, shape = ds.transform, ds.shape; cell = abs(tr.a)
        rr, cc = np.nonzero(src == 2); del src
        if not len(rr):
            continue
        px = cell * cell / 1e6
        x, y = rasterio.transform.xy(tr, rr, cc)
        x = np.asarray(x, "f8"); y = np.asarray(y, "f8")
        with rasterio.open(CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif") as ds:
            a = ds.read(1).astype("f4")
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            fab = a[rr, cc]; del a
        del rr, cc
        d = tree.query(np.c_[x, y], k=1)[0]
        conf = sample_at(CONF, x, y)
        edge = d <= EDGE_BAND_M
        at_ws = np.abs(fab - ws) <= WS_BAND_M
        for lab, m in (("all source 2", np.ones(len(x), bool)),
                       ("edge belt <= 250 m", edge),
                       ("edge belt & ground at full pool", edge & at_ws),
                       ("edge belt & ground at full pool & confidence 3", edge & at_ws & (conf == 3)),
                       ("edge belt & confidence 3", edge & (conf == 3))):
            rows.append(dict(zone=zone, stratum=lab, n_cells=int(m.sum()), km2=round(float(m.sum()) * px, 2),
                             fabdem_p50=round(float(np.nanmedian(fab[m])), 2) if m.any() else np.nan))
        del x, y, fab, d, conf
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    V, ws = build()
    print(f"\nsource-2 validation points: {len(V):,}; tail (< {TAIL_M:.0f} m) {int(V.is_tail.sum())} = {100*V.is_tail.mean():.2f} %")
    print(f"pool water surface as FABDEM records it, measured on the interior (> {INTERIOR_M:.0f} m from the boundary): {ws:.2f} m EVRF2019")
    axes = [("distance to polygon boundary (m)", "dist_to_boundary_m"),
            ("FABDEM - pool water surface (m)", "fab_minus_pool_ws_m"),
            ("FABDEM slope (deg)", "fabdem_slope_deg"),
            ("local relief over 200 m (m)", "fabdem_relief_200m_m"),
            ("distance to nearest sounding (m)", "dist_to_sounding_m")]
    S = separation(V, axes)
    print("\nAXIS SEPARATION -- rank AUC for tail membership (AUC > 0.5: a HIGHER value makes a tail point more likely)")
    print(S.to_string(index=False)); S.to_csv(CFG.TABLES / "p55f_edge_axis_separation.csv", index=False)
    Q = quintiles(V, axes)
    print("\nQUINTILES per axis")
    print(Q.to_string(index=False)); Q.to_csv(CFG.TABLES / "p55f_edge_axes_quintiles.csv", index=False)
    C = band_crosstab(V, ws)
    print("\nDECISIVE CROSSTAB -- edge belt x whether the GROUND stands at the full-pool level")
    print(C.to_string(index=False)); C.to_csv(CFG.TABLES / "p55f_edge_band_crosstab.csv", index=False)
    A = band_area(ws)
    print("\nAREA COST -- how much source-2 area a shoreline-aware domain would touch")
    print(A.to_string(index=False)); A.to_csv(CFG.TABLES / "p55f_edge_band_area.csv", index=False)
    V.to_csv(CFG.TABLES / "p55f_source2_points_with_axes.csv", index=False)
    print("\n-> outputs/tables/p55f_{edge_axis_separation,edge_axes_quintiles,edge_band_crosstab,edge_band_area,source2_points_with_axes}.csv")


if __name__ == "__main__":
    main()
