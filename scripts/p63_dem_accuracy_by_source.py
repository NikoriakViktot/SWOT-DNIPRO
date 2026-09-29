#!/usr/bin/env python
"""P63 -- the stratification p57 was missing: night ICESat-2 accuracy of the seamless DEM BY DEM-SOURCE CLASS.

Why this exists. p57 reports one headline number for the seamless DEM (set C: 841 752 night ground segments, LE90 1.036 m) and
stratifies it by zone, product, WorldCover class, elevation band and year -- but never by the PROVENANCE of the cell the point
falls on. p62 (GEDI) then found that two of p55's five source classes are metres below FABDEM at places where FABDEM itself agrees
with GEDI ground to ~0.5 m. This script re-creates p57's set C exactly (same loader, same dry-at-pass rule, same A/B selection, so
the reproduced LE90 must come out at 1.036) and adds `dem_source` as the stratum, which answers one question:

    how much of the headline LE90 is actually a statement about the parts of the DEM this project BUILT,
    as opposed to the FABDEM that it merely carried through?

Sign convention as in p57: residual = DEM - ICESat-2 ground, so a NEGATIVE bias means the DEM sits BELOW the real ground (a hole).
Source classes are p55's: 1 zone bed v2 (p53), 2 reservoir bed (hist20), 3 FABDEM->EVRF2019 (p56), 4 FABDEM feathered within 100 m
of a bathymetric edge, 5 former-pool gap fill between the bed DEM and the 16 m shoreline.
Output: outputs/tables/p63_dem_accuracy_by_source.csv
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

from swot_dnipro import config as CFG

spec = importlib.util.spec_from_file_location("p57", ROOT / "scripts/p57_dem_accuracy_night.py")
P57 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P57)

SRC_NAME = {1: "1 zone bed v2 (p53)", 2: "2 reservoir bed (hist20)", 3: "3 FABDEM -> EVRF2019 (p56)", 4: "4 FABDEM feathered (100 m)", 5: "5 former-pool gap fill"}
LAND_ZONES = ("ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY")


def source_areas():
    """km2 per source class, summed over every zone raster of the seamless DEM."""
    out = {}
    for p in sorted(P57.SEAM.glob("*_dem_source_20m.tif")):
        with rasterio.open(p) as ds:
            a = ds.read(1); px = abs(ds.transform.a * ds.transform.e) / 1e6
            u, n = np.unique(a[a > 0], return_counts=True)
            for k, cnt in zip(u, n):
                out[int(k)] = out.get(int(k), 0.0) + float(cnt) * px
    return out


def build_set_C():
    """p57's validation set C, re-created verbatim: B = dry exposed bed, A = land below the dam, both sampled on the seamless DEM."""
    P, c = P57.load_points()
    with rasterio.open(P57.RO / "zone1_poolwin_bed_recession_zoning_2023.tif") as ds:
        zon = ds.read(1); tr = ds.transform
    cc, rr = ~tr * (P.x.values, P.y.values); cc = np.floor(cc).astype(int); rr = np.floor(rr).astype(int)
    ok = (cc >= 0) & (cc < zon.shape[1]) & (rr >= 0) & (rr < zon.shape[0]); z = np.zeros(len(P), "u1"); z[ok] = zon[rr[ok], cc[ok]]; P["rzone"] = z
    dry_by = {1: pd.Timestamp("2023-06-30"), 2: pd.Timestamp("2023-08-06"), 3: pd.Timestamp("2023-09-08")}
    pool_pts = P.in_former_pool & np.isin(P.rzone, (1, 2, 3)) & (P.date >= "2023-07-01") & (P.date <= "2024-12-31")
    dry = np.zeros(len(P), bool)
    for k, d in dry_by.items():
        dry |= pool_pts.values & (P.rzone.values == k) & (P.date.values >= np.datetime64(d))
    ws = P57.sample(ROOT / "outputs/rasters/zone1/annual/zone1_water_share_2024_leafon_20m.tif", P.x.values, P.y.values)
    dry &= ~((P.year.values == 2024) & np.isfinite(ws) & (ws != 255) & (ws > 50))
    B = P[dry].copy(); B["zone"] = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"; B["product"] = "B exposed bed"; parts = [B]
    L = P[~P.in_former_pool].copy()
    for zn in LAND_ZONES:
        fab = P57.sample(P57.TERR / zn / "fabdem_evrf2019_20m.tif", L.x.values, L.y.values)
        wc = P57.sample(CFG.BULK_ROOT / "worldcover_frames" / zn / "wc_2021_20m.tif", L.x.values, L.y.values)
        m = np.isfinite(fab) & (fab > -5) & (fab < 80) & (wc != 80)
        g = L[m].copy(); g["zone"] = zn; g["product"] = "A land below the dam"; parts.append(g)
    V = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["zone", "date", "x", "y"])
    V["dem"] = np.nan; V["src"] = np.nan; V["fab"] = np.nan
    for zn in V.zone.unique():
        m = (V.zone == zn).values
        V.loc[m, "dem"] = P57.sample(P57.SEAM / f"{zn}_dem_evrf2019_20m.tif", V.x.values[m], V.y.values[m])
        V.loc[m, "src"] = P57.sample(P57.SEAM / f"{zn}_dem_source_20m.tif", V.x.values[m], V.y.values[m])
        V.loc[m, "fab"] = P57.sample(P57.TERR / zn / "fabdem_evrf2019_20m.tif", V.x.values[m], V.y.values[m])
    V["res"] = V.dem - V.H_ice
    V["fab_res"] = V.fab - V.H_ice          # FABDEM control: discriminates a bad DEM from a bad validation target
    return V[np.isfinite(V.res)].copy()


def main():
    V = build_set_C(); areas = source_areas(); tot_area = sum(areas.values())
    le90_all = float(np.percentile(np.abs(V.res), 90))
    print(f"set C re-created: {len(V):,} night segments (p57 reports 841,752); LE90 {le90_all:.3f} m (p57/B7.1 reports 1.036)")
    rows = []
    for k, name in SRC_NAME.items():
        g = V[V.src == k]; a = areas.get(k, 0.0)
        r = dict(dem_source=name, area_km2=round(a, 1), area_share=round(100 * a / tot_area, 3), n_points=len(g),
                 point_share=round(100 * len(g) / len(V), 3), points_per_100km2=round(len(g) / a * 100, 1) if a else np.nan)
        if len(g) >= 30:
            res = g.res.values; ab = np.abs(res)
            r.update(bias=round(float(res.mean()), 3), median=round(float(np.median(res)), 3), RMSE=round(float(np.sqrt((res ** 2).mean())), 3),
                     LE90=round(float(np.percentile(ab, 90)), 3), LE95=round(float(np.percentile(ab, 95)), 3),
                     NMAD=round(float(1.4826 * np.median(np.abs(res - np.median(res)))), 3), worst=round(float(res.min()), 2),
                     share_below_5m=round(float((res < -5).mean()), 4), validated="yes")
            fr = g.fab_res.values; fr = fr[np.isfinite(fr)]
            if len(fr) >= 30:
                # CONTROL. FABDEM at the same points. If BOTH the seamless DEM and FABDEM sit far below ICESat-2, the
                # validation target is water, not ground, and the bed is not on trial. If FABDEM is near zero while the
                # seamless DEM is not, the defect is in what p55 wrote.
                # The discriminator must look at the TAIL, not only the centre: a bed surface can have a sane median and still
                # carry a heavy negative tail. Compare like with like -- the share of points more than 5 m below the reference.
                tail, fab_tail = float((res < -5).mean()), float((fr < -5).mean())
                r.update(fabdem_bias=round(float(fr.mean()), 3), fabdem_median=round(float(np.median(fr)), 3),
                         fabdem_LE90=round(float(np.percentile(np.abs(fr), 90)), 3), fabdem_share_below_5m=round(fab_tail, 4),
                         verdict=("validation target likely WATER (both the product and FABDEM sit below ICESat-2)" if (tail > 0.02 and fab_tail > 0.02)
                                  else "DEM DEFECT: FABDEM agrees with ICESat-2 here, the product does not" if (tail > 0.02 >= fab_tail)
                                  else "consistent"))
        else:
            r.update(validated="NO -- not enough night ground segments to estimate accuracy at all")
        rows.append(r)
    T = pd.DataFrame(rows)
    T.loc[len(T)] = dict(dem_source="ALL (the headline B7.1 number)", area_km2=round(tot_area, 1), area_share=100.0, n_points=len(V), point_share=100.0,
                         bias=round(float(V.res.mean()), 3), median=round(float(np.median(V.res)), 3), RMSE=round(float(np.sqrt((V.res ** 2).mean())), 3),
                         LE90=round(le90_all, 3), LE95=round(float(np.percentile(np.abs(V.res), 95)), 3),
                         NMAD=round(float(1.4826 * np.median(np.abs(V.res - np.median(V.res)))), 3), worst=round(float(V.res.min()), 2),
                         share_below_5m=round(float((V.res < -5).mean()), 4), validated="yes")
    T.to_csv(CFG.TABLES / "p63_dem_accuracy_by_source.csv", index=False)
    print(T.to_string(index=False))
    built = V[V.src.isin([1, 2, 4, 5])]
    print(f"\nthe parts this project BUILT (sources 1, 2, 4, 5): {len(built):,} points = {100 * len(built) / len(V):.2f} % of the validation sample, "
          f"LE90 {np.percentile(np.abs(built.res), 90):.2f} m; the carried-through FABDEM (source 3) is {100 * (V.src == 3).mean():.2f} % of the points with LE90 {np.percentile(np.abs(V[V.src == 3].res), 90):.2f} m")
    print("-> outputs/tables/p63_dem_accuracy_by_source.csv")


if __name__ == "__main__":
    main()
