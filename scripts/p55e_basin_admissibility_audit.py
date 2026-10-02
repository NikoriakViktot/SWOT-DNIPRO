#!/usr/bin/env python
"""P55e -- is the hist20 edge tail systemic? DIAGNOSIS ONLY, nothing is written to any product.

The three worst source-2 residuals (-20.2, -18.0, -17.9 m; p55d) share one signature: hist20 confidence class 3, within 20 m of
the pre-breach pool boundary, and two of the three OUTSIDE the registry polygon altogether -- admitted only by p55's 60 m buffer
`pool_in0`. Support is not the problem there (99-106 soundings within 2 km, nearest 216-243 m). The admissibility rule is:

    p55 asks  "is this cell NOT an island?"   ->  bed_allowed = ~(p64 class == 1),  NaN outside the p64 window -> allowed
    it should ask  "is this cell confirmed basin?"

Absence of an island veto is not evidence of basin membership. Three points do not establish a systemic defect, so this script
censuses EVERY source-2 cell of the seamless DEM and cross-tabulates it by the two axes the three outliers picked out:

    basin membership   IN_POOL  (inside the registry polygon)  |  ANNULUS  (only inside the 60 m buffer p55 adds)
    hist20 confidence  the 250 m confidence-class raster, 3 being the weakest

and then measures the same strata against the night ICESat-2 ground segments of p63's set C. Two questions are answered
separately, because they have different consequences:
    AREA      how much bed was written under each rule (how big is the exposure)
    ACCURACY  does the residual tail actually concentrate in the suspected stratum (is the rule the cause)
Outputs: outputs/tables/p55e_source2_area_by_stratum.csv, p55e_source2_accuracy_by_stratum.csv
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
from shapely.geometry import MultiLineString

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

spec = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P63)
P57 = P63.P57

SEAM = CFG.BULK_ROOT / "dem_seamless"
CONF = ROOT / "outputs/rasters/kakhovka_bed_confidence_class_250m.tif"
ISL = ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif"
DIST_SND = ROOT / "outputs/rasters/kakhovka_bed_dist_to_sounding_250m.tif"
POOL_POLY = SD.load_utm("reservoir_full_pool_prebreach")
POOL_BUF_M = 60.0                 # p55's `POOL_POLY_BUF`: the rule under audit
BOUNDARY_STEP_M = 10.0            # densification of the pool boundary for the distance lookup
POOL_WS_MAX_M = 17.0              # inside the pool FABDEM is the pre-breach water surface; above this it is terrain
ZONES_WITH_POOL_BED = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")


def sample_at(path: Path, x, y):
    """Nearest-cell lookup of a raster at scattered metric coordinates (same idiom as P57.sample)."""
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
    """A KD-tree of the pool boundary, densified to BOUNDARY_STEP_M, for exact distance-to-shoreline at millions of cells."""
    b = POOL_POLY.boundary
    lines = list(b.geoms) if isinstance(b, MultiLineString) else [b]
    pts = []
    for ln in lines:
        n = max(int(np.ceil(ln.length / BOUNDARY_STEP_M)), 2)
        d = np.linspace(0.0, ln.length, n)
        pts.append(np.array([[p.x, p.y] for p in (ln.interpolate(t) for t in d)]))
    P = np.vstack(pts)
    return cKDTree(P), len(P)


def membership(x, y, pool_in_flat, tree):
    """Basin membership as p55 decides it, split into the class the registry confirms and the class only the buffer admits."""
    d = tree.query(np.c_[x, y], k=1)[0]
    cls = np.where(pool_in_flat, "IN_POOL", np.where(d <= POOL_BUF_M, "ANNULUS", "OUTSIDE_BUFFER"))
    return cls, d


def census():
    tree, nb = boundary_tree()
    print(f"pool boundary densified to {nb:,} vertices at {BOUNDARY_STEP_M:.0f} m", flush=True)
    rows = []
    for zone in ZONES_WITH_POOL_BED:
        sp = SEAM / f"{zone}_dem_source_20m.tif"
        if not sp.exists():
            continue
        with rasterio.open(sp) as ds:
            src = ds.read(1); tr, shape = ds.transform, ds.shape; cell = abs(tr.a)
        rr, cc = np.nonzero(src == 2)
        del src
        if not len(rr):
            continue
        px_km2 = cell * cell / 1e6
        x, y = rasterio.transform.xy(tr, rr, cc)
        x = np.asarray(x, "f8"); y = np.asarray(y, "f8")
        pool_in = features.rasterize([(POOL_POLY, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        pin = pool_in[rr, cc]; del pool_in
        cls, dbnd = membership(x, y, pin, tree)
        with rasterio.open(SEAM / f"{zone}_dem_evrf2019_20m.tif") as ds:
            a = ds.read(1).astype("f4"); a[a == ds.nodata] = np.nan; bed = a[rr, cc]; del a
        with rasterio.open(CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif") as ds:
            a = ds.read(1).astype("f4")
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            fab = a[rr, cc]; del a
        del rr, cc
        conf = sample_at(CONF, x, y); isl = sample_at(ISL, x, y); dsnd = sample_at(DIST_SND, x, y)
        print(f"  {zone}: {len(x):,} source-2 cells = {len(x) * px_km2:,.1f} km2", flush=True)
        for c in ("IN_POOL", "ANNULUS", "OUTSIDE_BUFFER"):
            for k in (1.0, 2.0, 3.0, np.nan):
                m = (cls == c) & (np.isnan(conf) if np.isnan(k) else (conf == k))
                if not m.any():
                    continue
                rows.append(dict(
                    zone=zone, membership=c, hist20_confidence=("no class" if np.isnan(k) else int(k)),
                    n_cells=int(m.sum()), km2=round(float(m.sum()) * px_km2, 2),
                    bed_p50=round(float(np.nanmedian(bed[m])), 2),
                    fabdem_p50=round(float(np.nanmedian(fab[m])), 2),
                    fabdem_minus_bed_p50=round(float(np.nanmedian(fab[m] - bed[m])), 2),
                    share_fabdem_above_ws=round(float(np.nanmean(fab[m] > POOL_WS_MAX_M)), 4),
                    dist_to_boundary_p50=round(float(np.median(dbnd[m])), 1),
                    dist_to_sounding_p50=round(float(np.nanmedian(dsnd[m])), 1) if np.isfinite(dsnd[m]).any() else np.nan,
                    p64_share_class0=round(float(np.nanmean(isl[m] == 0)), 4),
                    p64_share_unknown=round(float(np.nanmean(isl[m] == 2)), 4),
                    p64_share_no_window=round(float(np.mean(~np.isfinite(isl[m]))), 4)))
        del x, y, bed, fab, conf, isl, dsnd, cls, dbnd
    return pd.DataFrame(rows)


def accuracy():
    """The same strata, measured against night ICESat-2 ground. Sign as in p57: residual = DEM - ICESat-2 ground."""
    tree, _ = boundary_tree()
    V = P63.build_set_C()
    V = V[np.isfinite(V.res) & (V.src == 2)].copy()
    if V.empty:
        return pd.DataFrame()
    x = V.x.values.astype("f8"); y = V.y.values.astype("f8")
    from shapely import points as _pts, contains as _contains
    inside = _contains(POOL_POLY, _pts(x, y))
    cls, dbnd = membership(x, y, inside, tree)
    V["membership"] = cls; V["dist_to_boundary_m"] = dbnd
    V["hist20_confidence"] = sample_at(CONF, x, y)
    V["p64_class"] = sample_at(ISL, x, y)
    V["dist_to_sounding_m"] = sample_at(DIST_SND, x, y)
    rows = []

    def stat(label, g, extra=None):
        if len(g) < 20:
            return
        res = g.res.values; fr = g.fab_res.values; fr = fr[np.isfinite(fr)]
        r = dict(stratum=label, n_points=len(g),
                 bias=round(float(res.mean()), 3), median=round(float(np.median(res)), 3),
                 RMSE=round(float(np.sqrt((res ** 2).mean())), 3),
                 LE90=round(float(np.percentile(np.abs(res), 90)), 3),
                 worst=round(float(res.min()), 2),
                 share_below_5m=round(float((res < -5).mean()), 4),
                 fabdem_bias=round(float(fr.mean()), 3) if len(fr) else np.nan,
                 fabdem_share_below_5m=round(float((fr < -5).mean()), 4) if len(fr) else np.nan,
                 dist_to_boundary_p50=round(float(np.median(g.dist_to_boundary_m)), 1),
                 dist_to_sounding_p50=round(float(np.nanmedian(g.dist_to_sounding_m)), 1))
        r.update(extra or {})
        rows.append(r)

    stat("ALL source 2", V)
    for c in ("IN_POOL", "ANNULUS", "OUTSIDE_BUFFER"):
        stat(f"membership = {c}", V[V.membership == c])
    for k in (1.0, 2.0, 3.0):
        stat(f"hist20 confidence = {int(k)}", V[V.hist20_confidence == k])
    for c in ("IN_POOL", "ANNULUS"):
        for k in (1.0, 2.0, 3.0):
            stat(f"{c} & confidence {int(k)}", V[(V.membership == c) & (V.hist20_confidence == k)])
    # the exact signature of the three worst points, as a stratum rather than as three cases
    belt = V[(V.hist20_confidence == 3) & (V.dist_to_boundary_m <= 250)]
    stat("confidence 3 & within 250 m of the pool boundary", belt)
    stat("confidence 3 & within 250 m & NOT inside the polygon", belt[belt.membership != "IN_POOL"])
    for lo, hi in ((0, 50), (50, 100), (100, 250), (250, 1000), (1000, 1e9)):
        stat(f"distance to boundary {lo}-{hi if hi < 1e9 else 'inf'} m",
             V[(V.dist_to_boundary_m >= lo) & (V.dist_to_boundary_m < hi)])
    return pd.DataFrame(rows), V


def main():
    pd.set_option("display.width", 250)
    print("AREA CENSUS -- every source-2 cell of the seamless DEM")
    A = census()
    A.to_csv(CFG.TABLES / "p55e_source2_area_by_stratum.csv", index=False)
    print(A.to_string(index=False))
    print("\nACCURACY -- the same strata against night ICESat-2 ground (residual = DEM - ICESat-2)")
    B, V = accuracy()
    if len(B):
        B.to_csv(CFG.TABLES / "p55e_source2_accuracy_by_stratum.csv", index=False)
        print(B.to_string(index=False))
        V[["zone", "x", "y", "date", "res", "fab_res", "membership", "dist_to_boundary_m",
           "hist20_confidence", "p64_class", "dist_to_sounding_m"]].to_csv(
            CFG.TABLES / "p55e_source2_points.csv", index=False)
    print("\n-> outputs/tables/p55e_source2_{area_by_stratum,accuracy_by_stratum,points}.csv")


if __name__ == "__main__":
    main()
