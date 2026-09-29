#!/usr/bin/env python
"""P55c -- two open questions after the p53 domain repair. DIAGNOSIS ONLY, nothing is changed.

Q1  The seamless DEM's worst night-ICESat-2 residual is still -24.7 m while the repaired source 1 now bottoms out at
    -6.27 m. Which component produces it? Dumped with full provenance so the next repair is aimed, not guessed.
Q2  ZONE_1 and ZONE_4 still show seam medians of -3.87 and -3.56 m. "The pool bed meeting FABDEM land" is an assertion,
    not a measurement. Every seam pixel is therefore classified by what the bathymetry actually abuts:
      A REAL SHORELINE   -- the neighbour is corroborated LAND (p66 LAND/ISLAND below the dam, or, inside the pool,
                            FABDEM standing above the pre-breach water surface). A metre-scale step there can be a real
                            bank and must NOT be smoothed to <= 1 m.
      B ARTIFICIAL/DATA  -- the neighbour is water evidence or NoData: the end of bathymetry, of support, of the domain
                            or of the frame. A step there is a product artefact.
    A single global MAX_SEAM_STEP_M gate cannot tell these apart, which is the same conceptual error the taper had.
Output: outputs/tables/p55c_worst_residual.csv, p55c_seam_by_boundary_type.csv
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
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

spec = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P63)
P57 = P63.P57
SEAM = CFG.BULK_ROOT / "dem_seamless"
ZN = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
POOL_WS_MAX_M = 17.0
BATHY_SRC = (1, 2, 5)


def to_frame(path, shape, transform, resampling=Resampling.bilinear, fill=np.nan, dtype="f4"):
    dst = np.full(shape, fill, dtype)
    with rasterio.open(path) as ds:
        reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=transform, dst_crs=CFG.CRS_METRIC,
                  resampling=resampling, src_nodata=ds.nodata, dst_nodata=fill)
    return dst


def q1_worst():
    V = P63.build_set_C(); V = V[np.isfinite(V.res)].copy()
    w = V.loc[V.res.idxmin()].copy()
    zone = w.zone; n = ZN[zone]
    row = dict(x=round(float(w.x), 1), y=round(float(w.y), 1), zone=zone, residual_m=round(float(w.res), 2),
               dem_m=round(float(w.dem), 2), icesat2_ground_m=round(float(w.H_ice), 2), fabdem_m=round(float(w.fab), 2),
               fabdem_minus_icesat2_m=round(float(w.fab - w.H_ice), 2), p55_source_code=int(w.src), date=str(w.date)[:10])
    row["p55_source"] = {1: "zone bed v2 (p53)", 2: "reservoir bed (hist20)", 3: "FABDEM", 4: "FABDEM feathered", 5: "pool gap fill"}.get(int(w.src), "?")
    for name, p in (("p53_prediction_status", ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_status_30m.tif"),
                    ("p66_domain_class", ROOT / f"outputs/rasters/zone{n}/prebreach/prebreach_class.tif"),
                    ("distance_to_support_m", ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_support_distance_30m.tif")):
        row[name] = float(P57.sample(p, np.array([w.x]), np.array([w.y]))[0]) if p.exists() else np.nan
    # the ten worst, so one outlier is not mistaken for a pattern
    top = V.nsmallest(10, "res")[["zone", "x", "y", "res", "dem", "H_ice", "fab", "src"]].round(2)
    return row, top


def q2_seams():
    rows = []
    for zone, n in ZN.items():
        sp = SEAM / f"{zone}_dem_source_20m.tif"; dp = SEAM / f"{zone}_dem_evrf2019_20m.tif"
        if not sp.exists():
            continue
        with rasterio.open(sp) as ds:
            src = ds.read(1); tr, shape = ds.transform, ds.shape
        with rasterio.open(dp) as ds:
            dem = ds.read(1).astype("f4"); dem[dem == ds.nodata] = np.nan
        fab = to_frame(CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif", shape, tr)
        fab_gate = to_frame(CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif", shape, tr, Resampling.nearest)
        bathy = np.isin(src, BATHY_SRC)
        if not bathy.any():
            continue
        # water evidence: p66 CORE below the dam, or FABDEM at the pre-breach water surface inside the pool polygon
        water_ev = np.zeros(shape, bool)
        for m in (2, 3, 4):
            cp = ROOT / f"outputs/rasters/zone{m}/prebreach/prebreach_class.tif"
            if cp.exists():
                water_ev |= to_frame(cp, shape, tr, Resampling.nearest, fill=0, dtype="u1") == 1
        pool = features.rasterize([(SD.load_utm("reservoir_full_pool_prebreach"), 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        water_ev |= pool & np.isfinite(fab_gate) & (fab_gate <= POOL_WS_MAX_M)
        land_ev = np.isfinite(fab) & ~water_ev                       # corroborated land: FABDEM present and not water evidence
        boundary = bathy & ~ndimage.binary_erosion(bathy, iterations=1, border_value=1)
        near_land = ndimage.binary_dilation(land_ev & ~bathy, iterations=1)
        near_gap = ndimage.binary_dilation(~bathy & ~np.isfinite(dem), iterations=1)
        typeA = boundary & near_land
        typeB = boundary & ~near_land
        step = dem - fab
        px = abs(tr.a * tr.e) / 1e6
        for lab, m in (("A real shoreline (bed | corroborated land)", typeA), ("B artificial / data boundary (bed | water evidence or NoData)", typeB)):
            v = step[m & np.isfinite(step)]
            rows.append(dict(zone=zone, boundary_type=lab, n_cells=int(m.sum()), km2=round(float(m.sum()) * px, 2),
                             share=round(float(m.sum() / max(boundary.sum(), 1)), 3),
                             step_p50=round(float(np.median(v)), 2) if len(v) else np.nan,
                             step_p10=round(float(np.percentile(v, 10)), 2) if len(v) else np.nan,
                             step_p90=round(float(np.percentile(v, 90)), 2) if len(v) else np.nan,
                             share_step_gt_1m=round(float(np.mean(np.abs(v) > 1.0)), 3) if len(v) else np.nan,
                             touches_nodata=int((m & near_gap).sum())))
        print(f"  {zone}: boundary {int(boundary.sum()):,} cells -> A {int(typeA.sum()):,} / B {int(typeB.sum()):,}", flush=True)
    return pd.DataFrame(rows)


def main():
    print("Q1 -- the worst residual in the seamless DEM")
    row, top = q1_worst()
    pd.DataFrame([row]).to_csv(CFG.TABLES / "p55c_worst_residual.csv", index=False)
    for k, v in row.items():
        print(f"  {k:26s} {v}")
    print("\n  ten worst residuals:"); print(top.to_string(index=False))
    print("\nQ2 -- seam pixels classified by what the bathymetry abuts")
    S = q2_seams()
    if len(S):
        S.to_csv(CFG.TABLES / "p55c_seam_by_boundary_type.csv", index=False)
        pd.set_option("display.width", 220)
        print(S.to_string(index=False))
    print("\n-> outputs/tables/p55c_{worst_residual,seam_by_boundary_type}.csv")


if __name__ == "__main__":
    main()
