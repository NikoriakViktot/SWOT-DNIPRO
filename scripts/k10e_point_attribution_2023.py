#!/usr/bin/env python
"""K10e.V6/V7/V9/V10/V11 -- attach the spectral stack to every accepted 20 m
ICESat-2 segment, build vegetation/canopy risk, and split
PRIMARY_GROUND_VALIDATION_SET from VEGETATION_DIAGNOSTIC_SET.

Scope: ALL accepted POST_BREACH segments (k10d_modern_elevation_points_20m.
parquet, period == POST_BREACH -- 2023-09 through the latest cached scene).
Originally scoped to 2023 only; extended once the 2023-only channel PRIMARY
stratum proved too thin (17 segments, 2 RGTs) to repeat D14 on. Dynamic World
risk is added afterwards by k10e_merge_and_rescore_2023.py, not here.

V6 -- date-aware attribution: for each ICESat-2 segment, walk candidate scene
DATES in order of |delta_time_days| (not just the single nearest date), and
take the first scene TILE at that date whose raster actually covers the point
-- tiles are non-overlapping strips, so "nearest date globally" is not the
same as "nearest date that covers this point".

V7 -- 3x3 neighbourhood: center, median, min, max, sd, computed on the native
20 m stack grid (same grid the k10d ICESat-2 foundation's morphology raster
sampling uses elsewhere in this project).

V9 -- risk scores, built ONLY from evidence actually available here (spectral
indices + PhoREAL-derived canopy fields from ATL03 photons, both already
QC'd in k10d). No Dynamic World, no persistence across dates (that needs the
wall-to-wall succession product, not built yet) -- both gaps are recorded in
the manifest rather than silently assumed away.

V10 -- two output products, not one filtered set:
    A. PRIMARY_GROUND_VALIDATION_SET  (strict)
    B. VEGETATION_DIAGNOSTIC_SET      (everything else that passed water QC)

V11 -- ground-photon threshold sensitivity at 20 m, evaluated empirically via
inter-cycle dispersion (no independent truth exists to compute RMSE against,
so the observable proxy is: does raising the photon floor reduce the vertical
spread among repeat-cycle observations of the same canonical ~20 m cell?).

Outputs
-------
data/processed/current_bed/k10e_post_breach_vegetation_context.parquet  (all POST_BREACH accepted, V6+V7+V9)
outputs/tables/k10e_primary_ground_validation_post_breach.csv
outputs/tables/k10e_vegetation_diagnostic_post_breach.csv
outputs/tables/k10e_photon_threshold_sensitivity_20m.csv
"""
from __future__ import annotations

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

CUR = CFG.ROOT / "data" / "processed" / "current_bed"
STACKS = Path("/mnt/f/data_kakhovka_dem_swot/spectral_indices")  # see k10e_spectral_stack_2023.py
BAND_NAMES = ["NDVI", "NDWI", "MNDWI", "NDMI", "BSI", "surface_class"]
NODATA = -9.0
SHORE_BUFFER_M = 50.0
MAX_DELTA_DAYS = 25.0          # covers every 2023 water-QC-ok segment once the Sep/Oct/Nov
                               # scene gap is closed (max nearest-scene delta = 23 days);
                               # beyond this a "contemporaneous" spectral read is not trusted


class SceneStack:
    __slots__ = ("path", "date", "arr", "transform", "inv", "H", "W")

    def __init__(self, path, date):
        self.path, self.date = path, date
        with rasterio.open(path) as src:
            self.arr = src.read()               # (6, H, W)
            self.transform = src.transform
            self.H, self.W = src.height, src.width
        self.inv = ~self.transform

    def sample(self, x, y):
        """Center value + 3x3 window stats for every band, at metric (x, y)
        reprojected into this raster's own CRS by the caller."""
        col, row = self.inv * (x, y)
        col, row = int(np.floor(col)), int(np.floor(row))
        n = len(BAND_NAMES)
        center = np.full(n, np.nan)
        med = np.full(n, np.nan)
        mn = np.full(n, np.nan)
        mx = np.full(n, np.nan)
        sd = np.full(n, np.nan)
        if not (0 <= row < self.H and 0 <= col < self.W):
            return None
        c0 = self.arr[0, row, col]
        if c0 <= NODATA:
            return None
        r0, r1 = max(row - 1, 0), min(row + 2, self.H)
        c0_, c1_ = max(col - 1, 0), min(col + 2, self.W)
        win = self.arr[:, r0:r1, c0_:c1_]
        for b in range(n):
            v = win[b]
            valid = v[v > NODATA]
            center[b] = self.arr[b, row, col]
            if len(valid):
                med[b] = np.median(valid)
                mn[b] = valid.min()
                mx[b] = valid.max()
                sd[b] = valid.std() if len(valid) > 1 else 0.0
        return center, med, mn, mx, sd


def main() -> None:
    print("=" * 78)
    print("K10e.V6/V7/V9/V10/V11 -- POST_BREACH vegetation/canopy context for 20 m ICESat-2")
    print("=" * 78)

    g = pd.read_parquet(CUR / "k10d_modern_elevation_points_20m.parquet")
    g = g[g.period == "POST_BREACH"].copy().reset_index(drop=True)
    print(f"POST_BREACH accepted (QC-cascade-passing) segments: {len(g):,} "
          f"({g.date.min().date()} .. {g.date.max().date()})")
    ok = g.exposed_ground_validation_ok.sum()
    print(f"  of which night+DRY+>={SHORE_BUFFER_M:.0f}m (existing water QC): {ok:,}")

    # ---------------- load scene stacks, grouped by date --------------------
    import pyproj
    tf_cache = {}

    def to_scene_xy(x, y, crs):
        if crs not in tf_cache:
            tf_cache[crs] = pyproj.Transformer.from_crs(CFG.CRS_METRIC, crs, always_xy=True)
        return tf_cache[crs].transform(x, y)

    tif_paths = sorted(STACKS.glob("*_stack.tif"))
    print(f"spectral stacks found: {len(tif_paths)}")

    scenes = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            sens = src.tags().get("sensing_time", "")
        try:
            date = pd.to_datetime(sens, format="%Y%m%dT%H%M%S")
        except Exception:
            continue
        scenes.append({"path": p, "date": date, "crs": None})
    sdf = pd.DataFrame(scenes)
    sdf["day"] = sdf.date.dt.normalize()
    unique_days = sorted(sdf.day.unique())
    print(f"unique scene dates: {len(unique_days)} -> "
          f"{[str(d.date()) for d in unique_days]}")

    # No persistent raster cache: each tile is visited exactly once (the day
    # loop below is chronological and each day's candidate set only shrinks),
    # so caching every opened SceneStack accumulates unboundedly -- 89 tiles at
    # ~700 MB each (5490x5490 x 6 float32 bands, uncompressed once read) is
    # tens of GB and was killed by the OOM killer (exit 137) on the first full
    # POST_BREACH run. Open, use within this scope, let it be freed.
    def get_stack(path, date):
        return SceneStack(path, date)

    n = len(g)
    out = {f"{b.lower()}_center": np.full(n, np.nan) for b in BAND_NAMES[:-1]}
    out.update({f"{b.lower()}_3x3_median": np.full(n, np.nan) for b in BAND_NAMES[:-1]})
    out.update({f"{b.lower()}_3x3_min": np.full(n, np.nan) for b in BAND_NAMES[:-1]})
    out.update({f"{b.lower()}_3x3_max": np.full(n, np.nan) for b in BAND_NAMES[:-1]})
    out.update({f"{b.lower()}_3x3_sd": np.full(n, np.nan) for b in BAND_NAMES[:-1]})
    out["surface_class_code"] = np.full(n, np.nan)
    out["sentinel_datetime"] = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    out["delta_time_sentinel_days"] = np.full(n, np.nan)
    filled = np.zeros(n, bool)

    icesat_dates = g.date.values.astype("datetime64[ns]")
    gx, gy = g.x.values, g.y.values

    # order candidate days by how close they are, PER SEGMENT, but iterate day
    # by day (all segments whose current-best delta improves at this day) to
    # avoid opening every raster for every one of 11k+ points individually
    day_arr = np.array(unique_days, dtype="datetime64[ns]")
    best_delta = np.full(n, np.inf)
    for day in day_arr:
        delta = np.abs((icesat_dates - day) / np.timedelta64(1, "D"))
        candidate = (delta < best_delta) & (delta <= MAX_DELTA_DAYS)
        if not candidate.any():
            continue
        tiles_today = sdf[sdf.day == day]
        idxs = np.where(candidate)[0]
        for _, srow in tiles_today.iterrows():
            st = get_stack(srow.path, srow.date)
            with rasterio.open(srow.path) as src:
                crs = src.crs
            sx, sy = to_scene_xy(gx[idxs], gy[idxs], crs)
            for k, i in enumerate(idxs):
                res = st.sample(sx[k], sy[k])
                if res is None:
                    continue
                center, med, mn, mx, sd = res
                if center[0] <= NODATA:      # NDVI invalid at this pixel
                    continue
                for bi, bname in enumerate(BAND_NAMES[:-1]):
                    out[f"{bname.lower()}_center"][i] = center[bi]
                    out[f"{bname.lower()}_3x3_median"][i] = med[bi]
                    out[f"{bname.lower()}_3x3_min"][i] = mn[bi]
                    out[f"{bname.lower()}_3x3_max"][i] = mx[bi]
                    out[f"{bname.lower()}_3x3_sd"][i] = sd[bi]
                out["surface_class_code"][i] = center[-1]
                out["sentinel_datetime"][i] = srow.date
                out["delta_time_sentinel_days"][i] = delta[i]
                best_delta[i] = delta[i]
                filled[i] = True
        print(f"  day {pd.Timestamp(day).date()}: cumulative attributed "
              f"{int(filled.sum()):,}/{n:,}", flush=True)

    for k, v in out.items():
        g[k] = v
    print(f"\nattribution coverage: {int(filled.sum()):,}/{n:,} "
          f"({100*filled.sum()/n:.1f}%) within {MAX_DELTA_DAYS:.0f} days of a scene")

    CLASS_NAMES = {0: "INVALID", 1: "OPEN_WATER", 2: "SHALLOW_OR_MIXED_WATER",
                  3: "WET_SEDIMENT", 4: "DRY_BARE_SEDIMENT", 5: "SPARSE_HERBACEOUS",
                  6: "DENSE_HERBACEOUS", 7: "REED_OR_FLOODED_VEGETATION",
                  8: "BUILT_HARD_SURFACE", 9: "AMBIGUOUS"}
    g["surface_class"] = g.surface_class_code.map(
        lambda c: CLASS_NAMES.get(int(c), "UNATTRIBUTED") if np.isfinite(c) else "UNATTRIBUTED")
    print("\nsurface class distribution (POST_BREACH accepted segments):")
    print(g.surface_class.value_counts().to_string())

    # ---------------- V9: risk scores, no Dynamic World ----------------------
    print("\n" + "-" * 78)
    print("V9 -- vegetation/canopy risk (spectral + PhoREAL canopy fields only)")
    print("-" * 78)
    ndvi = g.ndvi_center.values
    ndmi = g.ndmi_center.values
    sd3 = g.ndvi_3x3_sd.values
    hc = g.h_canopy.values                    # PhoREAL, from ATL03 photons directly
    mean_c = g.h_mean_canopy.values
    veg_ph_frac = np.where(g.ph_count.values > 0,
                           g.veg_ph_count.values / np.maximum(g.ph_count.values, 1), 0.0)

    def clip01(v):
        return np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0)

    # vegetation_risk_score: continuous 0-1, driven by NDVI, 0 for water/bare
    g["vegetation_risk_score"] = clip01((ndvi - 0.10) / (0.50 - 0.10))
    # reed/herbaceous: high NDVI AND high NDMI together (flooded/lush vegetation)
    g["reed_herbaceous_risk_score"] = clip01(
        np.minimum((ndvi - 0.15) / 0.35, (ndmi - 0.0) / 0.30))
    # woody canopy risk: STRUCTURAL evidence only, independent of the spectral
    # branch by design -- PhoREAL h_canopy / h_mean_canopy / vegetation-photon
    # fraction come straight from ATL03 photon geometry, not from Sentinel-2.
    # No multi-date persistence term: that requires the wall-to-wall succession
    # raster (V12/V15), not built in this scoped pass -- recorded as a gap.
    g["woody_canopy_risk_score"] = clip01(
        0.6 * clip01(hc / 3.0) + 0.4 * clip01(veg_ph_frac / 0.3))
    # mixed-pixel risk: 3x3 spectral heterogeneity + shoreline proximity +
    # SHALLOW_OR_MIXED_WATER class, i.e. "can this pixel's class be trusted at all"
    g["mixed_pixel_risk_score"] = clip01(
        np.maximum(clip01(sd3 / 0.20),
                  np.where(g.surface_class == "SHALLOW_OR_MIXED_WATER", 1.0, 0.0)))
    g["woody_canopy_risk_note"] = "structural (PhoREAL ATL03 photons), no multi-date persistence term"

    for c in ["vegetation_risk_score", "reed_herbaceous_risk_score",
             "woody_canopy_risk_score", "mixed_pixel_risk_score"]:
        print(f"  {c}: median {np.nanmedian(g[c]):.3f}, p90 {np.nanpercentile(g[c],90):.3f}")

    # ---------------- V10: PRIMARY vs DIAGNOSTIC -----------------------------
    print("\n" + "-" * 78)
    print("V10 -- PRIMARY_GROUND_VALIDATION_SET vs VEGETATION_DIAGNOSTIC_SET")
    print("-" * 78)
    LOW_RISK = 0.20
    water_qc_ok = g.exposed_ground_validation_ok
    low_risk = ((g.vegetation_risk_score < LOW_RISK) & (g.woody_canopy_risk_score < LOW_RISK)
                & (g.mixed_pixel_risk_score < LOW_RISK))
    good_class = g.surface_class.isin(["DRY_BARE_SEDIMENT", "SPARSE_HERBACEOUS"])
    attributed = filled
    primary = water_qc_ok & low_risk & good_class & attributed
    diagnostic = water_qc_ok & ~primary

    g["k10e_split"] = np.select(
        [primary, diagnostic, ~water_qc_ok],
        ["PRIMARY_GROUND_VALIDATION_SET", "VEGETATION_DIAGNOSTIC_SET", "REJECTED_BY_WATER_QC"],
        default="REJECTED_BY_WATER_QC")

    print(f"  water-QC accepted (existing K10d rule): {int(water_qc_ok.sum()):,}")
    print(f"  PRIMARY_GROUND_VALIDATION_SET (+ low veg/canopy/mixed risk + "
          f"DRY_BARE_SEDIMENT/SPARSE_HERBACEOUS + spectrally attributed): "
          f"{int(primary.sum()):,}")
    print(f"  VEGETATION_DIAGNOSTIC_SET (water-QC ok, but vegetation/canopy/mixed "
          f"risk or unattributed): {int(diagnostic.sum()):,}")
    print(f"    of which unattributed (no scene within {MAX_DELTA_DAYS:.0f} d): "
          f"{int((water_qc_ok & ~attributed).sum()):,}")
    print(g[water_qc_ok].groupby("surface_class").size().sort_values(ascending=False)
          .to_string())

    # like-for-like comparison: same POST_BREACH population, with vs without the
    # vegetation/canopy QC added in this script (unlike the earlier 2023-only
    # run, this IS the same population k10d's "1,613 segments, 9 RGTs" figure
    # was computed on, so the two numbers are now directly comparable).
    chan_water_qc = water_qc_ok & (g.P_channel >= 0.5)
    chan_primary = primary & (g.P_channel >= 0.5)
    print(f"\n  channel stratum (P_channel>=0.5), full POST_BREACH population:")
    print(f"    water-QC ok, NO vegetation QC:  {int(chan_water_qc.sum()):,} segments, "
          f"{g[chan_water_qc].rgt.nunique()} RGTs  (of which "
          f"{int((chan_water_qc & attributed).sum())} spectrally attributed)")
    print(f"    + vegetation/canopy QC (PRIMARY): {int(chan_primary.sum()):,} segments, "
          f"{g[chan_primary].rgt.nunique()} RGTs")

    gp = CUR / "k10e_post_breach_vegetation_context.parquet"
    g.to_parquet(gp, index=False)
    g[primary].to_csv(CFG.TABLES / "k10e_primary_ground_validation_post_breach.csv", index=False)
    g[diagnostic].to_csv(CFG.TABLES / "k10e_vegetation_diagnostic_post_breach.csv", index=False)
    print(f"\n-> {gp}")
    print(f"-> {CFG.TABLES / 'k10e_primary_ground_validation_post_breach.csv'}")
    print(f"-> {CFG.TABLES / 'k10e_vegetation_diagnostic_post_breach.csv'}")

    # ---------------- V11: photon-count sensitivity --------------------------
    print("\n" + "-" * 78)
    print("V11 -- ground-photon threshold sensitivity at 20 m")
    print("-" * 78)
    print("No independent truth exists, so RMSE-vs-threshold cannot be computed directly.")
    print("Proxy: vertical dispersion among repeat-cycle observations of the same ~20 m")
    print("canonical along-track cell (same rgt, gt, x_atc rounded to 20 m), as a function")
    print("of the ground-photon floor applied.")
    gg = g[water_qc_ok].copy()
    gg["cell"] = gg.rgt.astype(str) + "_" + gg["gt"].astype(str) + "_" + \
        (gg.x_atc.round(-1) // 20 * 20).astype(int).astype(str)
    rows = []
    for thr in (3, 5, 10, 15, 20):
        sub = gg[gg.gnd_ph_count >= thr]
        rep = sub.groupby("cell").filter(lambda d: len(d) >= 2)
        disp = rep.groupby("cell").H_terrain_EVRF2019_empirical_m.std()
        rows.append({"gnd_ph_threshold": thr, "n_accepted": len(sub),
                     "n_RGT": sub.rgt.nunique(),
                     "n_repeat_cells": disp.notna().sum(),
                     "median_intercycle_sd_m": float(disp.median()) if len(disp) else np.nan,
                     "p90_intercycle_sd_m": float(disp.quantile(.9)) if len(disp) else np.nan})
    pts = pd.DataFrame(rows)
    pts.to_csv(CFG.TABLES / "k10e_photon_threshold_sensitivity_20m.csv", index=False)
    print(pts.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {CFG.TABLES / 'k10e_photon_threshold_sensitivity_20m.csv'}")


if __name__ == "__main__":
    main()
