#!/usr/bin/env python
"""Parts B & C — longitudinal water-surface profiles.

Part B  Lower Dnipro (Kherson): how large is the along-channel WSE gradient, and can
        it explain the ICESat-2 minus gauge difference seen when ICESat-2 observations
        sit 19-60 km from the gauge?

Part C  Former Kakhovka Reservoir: was the pre-breach pool flat, and does a
        river-like longitudinal gradient appear after 2023-06-06?

Chainage
--------
A centreline is built from the reservoir polygon by principal-axis binning (60 bins,
centroid of the water in each bin), giving an ordered polyline; chainage is measured
from the dam. All geometry in EPSG:32636.

Statistical unit
----------------
**One date** (a beam-pass profile). Segments within a beam-pass are correlated, and
beam-passes on one date share the atmosphere and the water state, so a per-date slope
is fitted and the *distribution of slopes across dates* is the result.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import nmad

MIN_CHAIN_POINTS = 3
MIN_SPAN_KM = 20.0
CHAIN_SEP_KM = 5.0


def reservoir_centreline():
    res = gpd.read_file(CFG.RESERVOIR_GEOJSON).to_crs(CFG.CRS_METRIC)
    geom = res.geometry.union_all()
    xy = shapely.get_coordinates(geom)
    c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    t = (xy - c) @ vt[0]
    bins = np.linspace(t.min(), t.max(), 60)
    nodes = [xy[(t >= a) & (t < b)].mean(0)
             for a, b in zip(bins[:-1], bins[1:]) if ((t >= a) & (t < b)).sum() > 3]
    line = LineString(np.array(nodes))
    dam = gpd.GeoSeries([Point(CFG.KAKHOVKA_DAM)], crs=CFG.CRS_GEOG).to_crs(CFG.CRS_METRIC)[0]
    return line, line.project(dam), res


def n_distinct(v, sep=CHAIN_SEP_KM):
    v = np.sort(np.asarray(v))
    keep = [v[0]]
    for x in v[1:]:
        if x - keep[-1] > sep:
            keep.append(x)
    return len(keep)


def fit_profiles(df, group_cols=("date",)):
    """Per-date least-squares slope of WSE against chainage."""
    out = []
    for key, g in df.groupby(list(group_cols)):
        key = key if isinstance(key, tuple) else (key,)
        if len(g) < MIN_CHAIN_POINTS:
            continue
        span = g.chain_km.max() - g.chain_km.min()
        nch = n_distinct(g.chain_km.values)
        if nch < MIN_CHAIN_POINTS or span < MIN_SPAN_KM:
            continue
        x, y = g.chain_km.to_numpy(float), g.wse_m.to_numpy(float)
        slope, icpt = np.polyfit(x, y, 1)
        pred = slope * x + icpt
        ss = float(np.sum((y - pred) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        rec = dict(zip(group_cols, key))
        rec.update({
            "n_points": len(g), "n_distinct_chainages": nch, "span_km": span,
            "chain_min_km": g.chain_km.min(), "chain_max_km": g.chain_km.max(),
            # sign: positive = WSE increases upstream (normal river), reported as cm/km
            "slope_cm_per_km": slope * 100.0,
            "slope_m_per_100km": slope * 100.0,
            "r2": 1 - ss / sst if sst > 0 else np.nan,
            "residual_nmad_m": nmad(y - pred),
            "wse_median_m": float(np.median(y)),
            "wse_range_m": float(y.max() - y.min()),
            "wse_nmad_m": nmad(y),
        })
        out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    line, dam_s, res = reservoir_centreline()
    print(f"centreline {line.length/1000:.0f} km, dam at chainage 0")

    # ---------------- Part C: reservoir ---------------------------------------
    ic = pd.read_parquet(CFG.KAKHOVKA_ATL13)
    ic["dt"] = pd.to_datetime(ic["datetime"], utc=True)
    ic["date"] = ic["dt"].dt.date
    ic = ic.dropna(subset=["lat_mean", "lon_mean", "median_wse_evrs_m"])
    g = gpd.GeoDataFrame(ic, geometry=gpd.points_from_xy(ic.lon_mean, ic.lat_mean),
                         crs=CFG.CRS_GEOG).to_crs(CFG.CRS_METRIC)
    ic["chain_km"] = [abs(line.project(p) - dam_s) / 1000 for p in g.geometry]
    # harmonise ATL13 to the mean/zero-tide crust (as elsewhere in this project)
    ic["wse_m"] = ic["median_wse_evrs_m"] + CFG.free2mean(ic["lat_mean"])
    ic[["date", "period", "rgt", "beam", "chain_km", "wse_m", "n_points",
        "nmad_m", "lat_mean", "lon_mean"]].to_csv(
        CFG.TABLES / "kakhovka_longitudinal_profiles.csv", index=False)
    ic.to_csv(CFG.FIGDATA / "FigD_kakhovka_profile_points.csv", index=False)

    prof = fit_profiles(ic[["date", "period", "chain_km", "wse_m"]], ("date", "period"))
    prof.to_csv(CFG.FIGDATA / "FigD_kakhovka_perdate_slopes.csv", index=False)

    rows = []
    for per in ["PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"]:
        s = prof[prof.period == per]
        if s.empty:
            continue
        rows.append({
            "period": per, "n_dates": len(s),
            "median_slope_cm_per_km": float(np.median(s.slope_cm_per_km)),
            "nmad_slope_cm_per_km": nmad(s.slope_cm_per_km),
            "p05_slope": float(np.percentile(s.slope_cm_per_km, 5)),
            "p95_slope": float(np.percentile(s.slope_cm_per_km, 95)),
            "frac_positive_slope": float((s.slope_cm_per_km > 0).mean()),
            "median_r2": float(np.median(s.r2)),
            "median_wse_range_m": float(np.median(s.wse_range_m)),
            "median_wse_nmad_m": float(np.median(s.wse_nmad_m)),
            "median_span_km": float(np.median(s.span_km)),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(CFG.TABLES / "kakhovka_pre_post_slope_summary.csv", index=False)
    summ.to_csv(CFG.FIGDATA / "FigD_kakhovka_slope_summary.csv", index=False)

    # classification per post-breach date (C4)
    cls = []
    for r in prof.itertuples():
        crit = {
            "monotonic_gradient": abs(r.slope_cm_per_km) > 1.0,
            "gradient_downstream_positive": r.slope_cm_per_km > 1.0,
            "linear_fit_r2_gt_0.5": r.r2 > 0.5,
            "wse_range_gt_1m": r.wse_range_m > 1.0,
        }
        score = sum(crit.values())
        verdict = ("river-dominated" if score >= 3 else
                   "transitional" if score == 2 else
                   "reservoir-like" if r.n_points >= MIN_CHAIN_POINTS else "insufficient data")
        cls.append({"date": r.date, "period": r.period, "slope_cm_per_km": r.slope_cm_per_km,
                    "r2": r.r2, "wse_range_m": r.wse_range_m, "span_km": r.span_km,
                    **{f"crit_{k}": v for k, v in crit.items()},
                    "criteria_met": score, "classification": verdict})
    pd.DataFrame(cls).to_csv(
        CFG.TABLES / "kakhovka_postbreach_channel_classification.csv", index=False)

    print("\n=== PART C: longitudinal slope of the former Kakhovka Reservoir ===")
    print(summ.to_string(index=False))
    print("\nper-date slopes by period (cm/km):")
    for per in ["PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"]:
        s = prof[prof.period == per]
        if len(s):
            print(f"  {per:16s} n={len(s):2d}  "
                  f"{np.round(np.sort(s.slope_cm_per_km.values), 2).tolist()}")
    print("\nclassification counts:")
    print(pd.DataFrame(cls).groupby(["period", "classification"]).size().to_string())

    # ---------------- Part B: lower Dnipro gradient ---------------------------
    seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kherson_atl13_segments.parquet")
    seg["dt"] = pd.to_datetime(seg["time"], utc=True, errors="coerce")
    seg["date"] = seg["dt"].dt.date
    kh = gpd.GeoSeries([Point(CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3])],
                       crs=CFG.CRS_GEOG).to_crs(CFG.CRS_METRIC)[0]
    sub = seg[(seg.dt >= "2023-01-01") & (seg.dt < CFG.BREACH_DATE)].copy()
    gs = gpd.GeoDataFrame(sub, geometry=gpd.points_from_xy(sub.lon, sub.lat),
                          crs=CFG.CRS_GEOG).to_crs(CFG.CRS_METRIC)
    sub["dist_km"] = gs.geometry.distance(kh) / 1000.0
    from swot_dnipro.vertical import sample_grid
    z = sample_grid(CFG.EGG2015_TIF, sub.lon.values, sub.lat.values)
    sub["wse_m"] = sub.h_wgs84_m.values + CFG.free2mean(sub.lat.values) - z
    b = []
    for (d,), grp in sub.groupby(["date"]):
        if len(grp) < 50 or grp.dist_km.max() - grp.dist_km.min() < 10:
            continue
        sl, ic_ = np.polyfit(grp.dist_km, grp.wse_m, 1)
        b.append({"date": d, "n_segments": len(grp),
                  "dist_min_km": grp.dist_km.min(), "dist_max_km": grp.dist_km.max(),
                  "slope_cm_per_km": sl * 100, "wse_at_gauge_extrap_m": ic_,
                  "wse_median_m": float(np.median(grp.wse_m)),
                  "residual_nmad_m": nmad(grp.wse_m - (sl * grp.dist_km + ic_))})
    bl = pd.DataFrame(b)
    bl.to_csv(CFG.FIGDATA / "FigB_lower_dnipro_gradient.csv", index=False)
    print("\n=== PART B: lower-Dnipro WSE gradient (ICESat-2, 2023 pre-breach) ===")
    print(bl.to_string(index=False))
    if len(bl):
        print(f"\n  median gradient {np.median(bl.slope_cm_per_km):+.2f} cm/km  "
              f"-> over the 29.8 km median ICESat-gauge separation that is "
              f"{np.median(bl.slope_cm_per_km) * 29.8 / 100:+.3f} m")


if __name__ == "__main__":
    main()
