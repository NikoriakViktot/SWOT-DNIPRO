#!/usr/bin/env python
"""P55d -- the two tasks left after p53 was frozen. DIAGNOSIS ONLY.

TASK A  hist20 outlier audit. The three worst night-ICESat-2 residuals carried by p55 source 2 (reservoir bed, hist20)
        are -20.16, -17.98 and -17.92 m. Each is attributed against: ICESat-2 ground, the hist20 bed, FABDEM, distance
        to the nearest hist20 sounding, distance to the pre-breach pool boundary, hist20's own confidence class, local
        FABDEM slope, the neighbouring bathymetry and whether the point sits in the surveyed channel. The question is
        which of five mechanisms produced it: extrapolation, a vertical datum error, a wrong domain, an old
        interpolation artefact, or a genuine bathymetric depression.
TASK B  Artificial seam audit. True shorelines are left alone. Only boundary type B from p55c is examined -- the seam
        pixels where the bathymetry abuts water evidence or NoData rather than corroborated land -- and each is given a
        subtype so the fill decision can be made per class rather than by one global step threshold:
          data termination   the neighbour is a declared CORE gap: the product stops where knowledge stops
          source switch      the neighbour is FABDEM standing on water evidence: a bed hands over to a water surface
          frame edge         the neighbour is outside the zone frame
Outputs: outputs/tables/p55d_hist20_outliers.csv, p55d_artificial_seam_subtypes.csv
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
SND = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"


def to_frame(path, shape, transform, resampling=Resampling.bilinear, fill=np.nan, dtype="f4"):
    dst = np.full(shape, fill, dtype)
    with rasterio.open(path) as ds:
        reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=transform, dst_crs=CFG.CRS_METRIC,
                  resampling=resampling, src_nodata=ds.nodata, dst_nodata=fill)
    return dst


def task_a(n_worst=3):
    V = P63.build_set_C(); V = V[np.isfinite(V.res) & (V.src == 2)].copy()
    W = V.nsmallest(n_worst, "res").copy()
    pool = SD.load_utm("reservoir_full_pool_prebreach")
    sd = pd.read_parquet(SND, columns=["x", "y", "H_bed_evrf2019_m"]) if SND.exists() else None
    tree = cKDTree(np.c_[sd.x.values, sd.y.values]) if sd is not None else None
    bedr = ROOT / "outputs/rasters/kakhovka_bed_OK_epoch_50m.tif"
    conf = ROOT / "outputs/rasters/kakhovka_bed_confidence_class_250m.tif"
    zone1 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
    with rasterio.open(CFG.BULK_ROOT / "terrain" / zone1 / "fabdem_evrf2019_20m.tif") as ds:
        fab1 = ds.read(1).astype("f4"); fab1[fab1 == ds.nodata] = np.nan; tr1, cell1 = ds.transform, abs(ds.transform.a)
    gy_, gx_ = np.gradient(np.nan_to_num(fab1, nan=float(np.nanmedian(fab1))), cell1, cell1)
    slope1 = np.degrees(np.arctan(np.hypot(gx_, gy_)))
    rows = []
    for _, w in W.iterrows():
        r = dict(zone=w.zone, x=round(float(w.x), 1), y=round(float(w.y), 1), residual_m=round(float(w.res), 2),
                 icesat2_ground_m=round(float(w.H_ice), 2), seamless_dem_m=round(float(w.dem), 2), fabdem_m=round(float(w.fab), 2),
                 fabdem_minus_icesat2_m=round(float(w.fab - w.H_ice), 2), date=str(w.date)[:10])
        r["hist20_bed_m"] = round(float(P57.sample(bedr, np.array([w.x]), np.array([w.y]))[0]), 2) if bedr.exists() else np.nan
        r["hist20_confidence_class"] = float(P57.sample(conf, np.array([w.x]), np.array([w.y]))[0]) if conf.exists() else np.nan
        if tree is not None:
            d, i = tree.query([w.x, w.y], k=8)
            r["dist_to_nearest_sounding_m"] = round(float(d[0]), 1)
            r["n_soundings_within_2km"] = int(len(tree.query_ball_point([w.x, w.y], r=2000.0)))
            r["nearest_sounding_bed_m"] = round(float(sd.H_bed_evrf2019_m.values[i[0]]), 2)
            r["status"] = "extrapolation (no sounding within 2 km)" if r["n_soundings_within_2km"] == 0 else "interpolation"
        p = SD.load_utm("reservoir_full_pool_prebreach")
        from shapely.geometry import Point
        pt = Point(float(w.x), float(w.y))
        r["inside_pool_polygon"] = bool(p.contains(pt)); r["dist_to_pool_boundary_m"] = round(float(p.boundary.distance(pt)), 1)
        r["p64_island_class"] = float(P57.sample(ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif", np.array([w.x]), np.array([w.y]))[0])
        rc = rasterio.transform.rowcol(tr1, w.x, w.y)
        rr_, cc_ = int(np.asarray(rc[0])), int(np.asarray(rc[1]))
        if 0 <= rr_ < fab1.shape[0] and 0 <= cc_ < fab1.shape[1]:
            r["fabdem_slope_deg"] = round(float(slope1[rr_, cc_]), 2)
            sl = (slice(max(rr_ - 5, 0), rr_ + 6), slice(max(cc_ - 5, 0), cc_ + 6))
            r["fabdem_relief_200m_m"] = round(float(np.nanmax(fab1[sl]) - np.nanmin(fab1[sl])), 2)
        nb = P57.sample(bedr, np.array([w.x - 100, w.x + 100, w.x, w.x]), np.array([w.y, w.y, w.y - 100, w.y + 100]))
        r["neighbouring_bed_m"] = "; ".join("nan" if not np.isfinite(v) else f"{v:.2f}" for v in nb)
        rows.append(r)
    return pd.DataFrame(rows)


def task_b():
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
        water_ev = np.zeros(shape, bool)
        for m in (2, 3, 4):
            cp = ROOT / f"outputs/rasters/zone{m}/prebreach/prebreach_class.tif"
            if cp.exists():
                water_ev |= to_frame(cp, shape, tr, Resampling.nearest, fill=0, dtype="u1") == 1
        pool = features.rasterize([(SD.load_utm("reservoir_full_pool_prebreach"), 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        water_ev |= pool & np.isfinite(fab_gate) & (fab_gate <= POOL_WS_MAX_M)
        land_ev = np.isfinite(fab) & ~water_ev
        boundary = bathy & ~ndimage.binary_erosion(bathy, iterations=1, border_value=1)
        typeB = boundary & ~ndimage.binary_dilation(land_ev & ~bathy, iterations=1)
        nodata_nb = ndimage.binary_dilation(~bathy & ~np.isfinite(dem), iterations=1)
        fabwater_nb = ndimage.binary_dilation((~bathy) & np.isin(src, (3, 4)) & water_ev, iterations=1)
        edge = np.zeros(shape, bool); edge[0, :] = edge[-1, :] = True; edge[:, 0] = edge[:, -1] = True
        frame_nb = ndimage.binary_dilation(edge, iterations=1)
        step = dem - fab
        px = abs(tr.a * tr.e) / 1e6
        sub = [("data termination (neighbour is a declared CORE gap)", typeB & nodata_nb),
               ("source switch (neighbour is FABDEM on water evidence)", typeB & fabwater_nb & ~nodata_nb),
               ("frame edge", typeB & frame_nb & ~nodata_nb & ~fabwater_nb)]
        acc = np.zeros(shape, bool)
        for lab, m in sub:
            acc |= m
            v = step[m & np.isfinite(step)]
            rows.append(dict(zone=zone, subtype=lab, n_cells=int(m.sum()), km2=round(float(m.sum()) * px, 2),
                             step_p50=round(float(np.median(v)), 2) if len(v) else np.nan,
                             step_p10=round(float(np.percentile(v, 10)), 2) if len(v) else np.nan,
                             share_step_gt_1m=round(float(np.mean(np.abs(v) > 1.0)), 3) if len(v) else np.nan,
                             inside_src=";".join(str(int(k)) for k in np.unique(src[m])[:4])))
        rest = typeB & ~acc
        if rest.any():
            v = step[rest & np.isfinite(step)]
            rows.append(dict(zone=zone, subtype="other / unclassified", n_cells=int(rest.sum()), km2=round(float(rest.sum()) * px, 2),
                             step_p50=round(float(np.median(v)), 2) if len(v) else np.nan,
                             step_p10=round(float(np.percentile(v, 10)), 2) if len(v) else np.nan,
                             share_step_gt_1m=round(float(np.mean(np.abs(v) > 1.0)), 3) if len(v) else np.nan,
                             inside_src=";".join(str(int(k)) for k in np.unique(src[rest])[:4])))
        print(f"  {zone}: type B {int(typeB.sum()):,} cells", flush=True)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 230)
    print("TASK A -- hist20 outlier audit (p55 source 2)")
    A = task_a(); A.to_csv(CFG.TABLES / "p55d_hist20_outliers.csv", index=False)
    print(A.T.to_string())
    print("\nTASK B -- artificial seam subtypes (true shorelines excluded)")
    B = task_b()
    if len(B):
        B.to_csv(CFG.TABLES / "p55d_artificial_seam_subtypes.csv", index=False)
        print(B.to_string(index=False))
    print("\n-> outputs/tables/p55d_{hist20_outliers,artificial_seam_subtypes}.csv")


if __name__ == "__main__":
    main()
