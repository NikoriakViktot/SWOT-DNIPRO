#!/usr/bin/env python
"""P67 -- the decisive test: is the source-1 defect caused by the DOMAIN rather than by the interpolator?

p65 established that p53's residual against night ICESat-2 keys on ground elevation (Spearman -0.874): fine below 2 m, -8.0 m
with a 100 % tail between 5 and 10 m, -32.0 m above 10 m. p66 rebuilt where a bed may exist from the pre-breach water record
itself. If the catastrophic residuals sit in the part of p53's old domain that the new one excludes, then the interpolator was
never the problem -- it was asked to produce a bed over land (user, 2026-09-20).

    OLD = p53's own domain: UNION over 2019..2022 of (leaf-on water_share >= 50 %) UNIONed with the digitised p27 contour
    NEW = p66 CORE + LEVEL_DEPENDENT (pre-breach water occurrence, stage-stratified, with a support rule)
    test: of p65's points with residual < -5 m, what share lies in OLD \\ NEW?

Reported per zone and pooled, together with the elevation profile of the difference, so the claim is falsifiable: if the tail
were spread evenly across OLD, the domain would be exonerated.
Output: outputs/tables/p67_old_vs_new_domain.csv
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
from rasterio.enums import Resampling
from rasterio.warp import reproject

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

spec = importlib.util.spec_from_file_location("p53", ROOT / "scripts/p53_below_dam_bed_dem.py")
P53 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P53)
spec2 = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(P63)
P57 = P63.P57
ZONES = {"ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
TAIL_M = -5.0
NEW_CLASSES = (1, 2)        # CORE, LEVEL_DEPENDENT


def old_grid(zone):
    g = SD.build_grid(SD.load_utm(zone), P53.CELL, what=f"p53 grid {zone}")
    gx, gy = g["gx"], g["gy"]
    G = dict(nx=len(gx), ny=len(gy), x0=float(gx.min() - P53.CELL / 2), y1=float(gy.max() + P53.CELL / 2))
    G["transform"] = rasterio.transform.from_origin(G["x0"], G["y1"], P53.CELL, P53.CELL)
    return G


def main():
    V = P63.build_set_C(); S1 = V[V.src == 1].copy(); S1["res"] = S1.dem - S1.H_ice
    rows = []
    for zone, n in ZONES.items():
        G = old_grid(zone)
        try:
            old, years = P53.water_polygon_mask(zone, n, G)
        except Exception as e:
            print(f"{zone}: old domain unavailable ({type(e).__name__}: {e})"); continue
        cp = ROOT / f"outputs/rasters/zone{n}/prebreach/prebreach_class.tif"
        if not cp.exists():
            print(f"{zone}: p66 class raster missing"); continue
        new_cls = np.zeros(old.shape, "u1")
        with rasterio.open(cp) as ds:
            reproject(source=rasterio.band(ds, 1), destination=new_cls, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        new = np.isin(new_cls, NEW_CLASSES)
        # WHERE p53 ACTUALLY WROTE A BED is not the water polygon: --shore-mode taper extends the surface 150 m BEYOND it,
        # blending to MAL. That band is outside `old`, so a test against the polygon alone cannot see it.
        bed_extent = np.zeros(old.shape, bool)
        bp = ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_PRE_BREACH_30m.tif"
        if bp.exists():
            be = np.zeros(old.shape, "f4")
            with rasterio.open(bp) as ds:
                reproject(source=rasterio.band(ds, 1), destination=be, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                          resampling=Resampling.nearest, src_nodata=ds.nodata, dst_nodata=np.nan)
            bed_extent = np.isfinite(be)
        taper = bed_extent & ~old
        uncertain = new_cls == 5
        px = P53.CELL * P53.CELL / 1e6
        diff = old & ~new
        g = S1[S1.zone == zone]
        if len(g):
            rc = rasterio.transform.rowcol(G["transform"], g.x.values, g.y.values)
            r_, c_ = np.asarray(rc[0]), np.asarray(rc[1])
            ok = (r_ >= 0) & (r_ < old.shape[0]) & (c_ >= 0) & (c_ < old.shape[1])
            in_diff = np.zeros(len(g), bool); in_diff[ok] = diff[r_[ok], c_[ok]]
            in_unc = np.zeros(len(g), bool); in_unc[ok] = uncertain[r_[ok], c_[ok]]
            in_taper = np.zeros(len(g), bool); in_taper[ok] = taper[r_[ok], c_[ok]]
            in_poly = np.zeros(len(g), bool); in_poly[ok] = old[r_[ok], c_[ok]]
            tail = (g.res < TAIL_M).values
            rows.append(dict(zone=zone, old_km2=round(float(old.sum()) * px, 1), new_km2=round(float(new.sum()) * px, 1),
                             old_minus_new_km2=round(float(diff.sum()) * px, 1),
                             uncertain_km2=round(float(uncertain.sum()) * px, 1),
                             n_points=len(g), n_tail=int(tail.sum()),
                             tail_in_old_minus_new=round(float(in_diff[tail].mean()), 3) if tail.any() else np.nan,
                             nontail_in_old_minus_new=round(float(in_diff[~tail].mean()), 3) if (~tail).any() else np.nan,
                             tail_in_uncertain=round(float(in_unc[tail].mean()), 3) if tail.any() else np.nan,
                             median_ground_tail_m=round(float(g.H_ice[tail].median()), 2) if tail.any() else np.nan,
                             taper_km2=round(float(taper.sum()) * px, 1),
                             tail_in_taper=round(float(in_taper[tail].mean()), 3) if tail.any() else np.nan,
                             nontail_in_taper=round(float(in_taper[~tail].mean()), 3) if (~tail).any() else np.nan,
                             tail_in_water_polygon=round(float(in_poly[tail].mean()), 3) if tail.any() else np.nan,
                             nontail_in_water_polygon=round(float(in_poly[~tail].mean()), 3) if (~tail).any() else np.nan))
            print(f"{zone}: OLD {old.sum() * px:,.0f} km2 -> NEW {new.sum() * px:,.0f} km2, difference {diff.sum() * px:,.0f} km2 "
                  f"(UNCERTAIN {uncertain.sum() * px:,.0f}); tail points in the difference "
                  f"{in_diff[tail].mean() if tail.any() else float('nan'):.0%} vs non-tail {in_diff[~tail].mean():.0%}"
                  f" | TAPER band {taper.sum() * px:,.0f} km2: tail {in_taper[tail].mean() if tail.any() else float('nan'):.0%} vs non-tail {in_taper[~tail].mean():.0%}", flush=True)
    if rows:
        T = pd.DataFrame(rows)
        pooled = dict(zone="POOLED", old_km2=T.old_km2.sum(), new_km2=T.new_km2.sum(), old_minus_new_km2=T.old_minus_new_km2.sum(),
                      uncertain_km2=T.uncertain_km2.sum(), n_points=int(T.n_points.sum()), n_tail=int(T.n_tail.sum()))
        w = T.n_tail.values
        pooled["tail_in_old_minus_new"] = round(float(np.average(T.tail_in_old_minus_new.fillna(0), weights=np.where(w > 0, w, 1e-9))), 3)
        pooled["nontail_in_old_minus_new"] = round(float(np.average(T.nontail_in_old_minus_new.fillna(0), weights=T.n_points - T.n_tail + 1e-9)), 3)
        for c in ("tail_in_taper", "nontail_in_taper", "tail_in_water_polygon", "nontail_in_water_polygon"):
            wts = w if c.startswith("tail") else (T.n_points - T.n_tail).values
            pooled[c] = round(float(np.average(T[c].fillna(0), weights=np.where(wts > 0, wts, 1e-9))), 3)
        T = pd.concat([T, pd.DataFrame([pooled])], ignore_index=True)
        T.to_csv(CFG.TABLES / "p67_old_vs_new_domain.csv", index=False)
        print("\n" + T.to_string(index=False))
        print("\n-> outputs/tables/p67_old_vs_new_domain.csv")


if __name__ == "__main__":
    main()
