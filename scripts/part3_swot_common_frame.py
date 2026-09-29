#!/usr/bin/env python
"""PART 3 — SWOT PIXC in the common vertical frame, compared with gauge and ICESat-2.

Validated chain (PIXC PDD D-56411 Rev C p.53; sign-verified against RiverSP to
-0.0008 m over 1023 nodes):

    h_SWOT        = height - solid_earth_tide - load_tide_fes - pole_tide
    H_SWOT_common = h_SWOT - zeta_EGG2015

PIXC `height` is already ellipsoidal; subtracting `geoid` instead is a ~23.9 m
blunder and is never done here.

Comparisons are made against
  * the Kherson gauge in EVRF2019 (Part 1), absolute, no datum shifted away
  * ICESat-2 in the same common frame (Part 2)

Outputs
-------
outputs/tables/swot_validation_against_gauge_and_icesat.csv
data/processed/gauges/swot_levels_common_frame.parquet
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import netCDF4 as nc
import pyproj

from swot_dnipro import config as CFG
from swot_dnipro.vertical import sample_grid

PIXC_DIR = ROOT / "data/raw/pixc_nova_kakhovka"
GAUGES = ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet"
ICE_LV = ROOT / "data/processed/gauges/icesat_levels_common_frame.parquet"
OUT_PQ = ROOT / "data/processed/gauges/swot_levels_common_frame.parquet"

KHERSON = 80805
OPEN_WATER = 4                     # PIXC classification: open_water
RADII_KM = (0.5, 1.0, 2.0, 5.0)
RNG = np.random.default_rng(CFG.SEED)
_GEOD = pyproj.Geod(ellps="WGS84")


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    m = [np.median(RNG.choice(x, len(x), True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def read_granule(path, lat0, lon0, max_km=6.0):
    d = nc.Dataset(path)
    g = d.groups["pixel_cloud"]
    cls = np.asarray(g.variables["classification"][:])
    keep = cls == OPEN_WATER
    if not keep.any():
        return pd.DataFrame()
    lat = np.asarray(g.variables["latitude"][:])[keep]
    lon = np.asarray(g.variables["longitude"][:])[keep]
    _, _, dist = _GEOD.inv(np.full(len(lat), lon0), np.full(len(lat), lat0), lon, lat)
    near = dist <= max_km * 1000.0
    if not near.any():
        return pd.DataFrame()
    idx = np.where(keep)[0][near]

    def v(name):
        return np.asarray(g.variables[name][:])[idx].astype(float)

    h = v("height") - v("solid_earth_tide") - v("load_tide_fes") - v("pole_tide")
    out = pd.DataFrame({"lat": lat[near], "lon": lon[near], "h_swot_m": h,
                        "dist_km": dist[near] / 1000.0,
                        "geoid_egm2008_m": v("geoid")})
    out["zeta_egg2015_m"] = sample_grid(CFG.EGG2015_TIF, out.lon.values, out.lat.values)
    out["H_common_m"] = out.h_swot_m - out.zeta_egg2015_m
    d.close()
    return out.dropna(subset=["H_common_m"])


def main() -> None:
    gv = pd.read_parquet(GAUGES)
    kh = gv[(gv.station_id == KHERSON) & (gv.qc == "ok")]
    lat0, lon0 = float(kh.lat.iloc[0]), float(kh.lon.iloc[0])
    ice = pd.read_parquet(ICE_LV) if ICE_LV.exists() else pd.DataFrame()

    files = sorted(PIXC_DIR.glob("SWOT_L2_HR_PIXC_*.nc"))
    print(f"{len(files)} PIXC granules; Kherson gauge at {lat0:.5f}, {lon0:.5f}\n")

    per_px, rows = [], []
    for f in files:
        m = re.search(r"PIXC_(\d+)_(\d+)_(\w+)_(\d{8}T\d{6})", f.name)
        cyc, pas, tile, stamp = m.groups()
        utc = pd.to_datetime(stamp, format="%Y%m%dT%H%M%S")
        date = utc.normalize()
        px = read_granule(f, lat0, lon0)
        if px.empty:
            print(f"{date:%Y-%m-%d}  no open-water pixels within 6 km"); continue
        px["date"], px["swot_utc"], px["cycle"], px["pass"] = date, utc, int(cyc), int(pas)
        per_px.append(px)

        gq = kh[kh.date == date]
        h_gauge = float(gq.H_evrf2019_m.iloc[0]) if len(gq) else np.nan
        stage = float(gq.stage_m.iloc[0]) if len(gq) else np.nan
        for R in RADII_KM:
            s = px[px.dist_km <= R]
            if len(s) < 30:
                continue
            hs = float(np.median(s.H_common_m))
            iq = ice[(ice.station_id == KHERSON) & (ice.date == date)] if len(ice) else ice
            h_ice = float(np.median(iq.H_common_m)) if len(iq) else np.nan
            rows.append({
                "date": date, "swot_utc": utc, "cycle": int(cyc), "pass": int(pas),
                "tile": tile, "radius_km": R, "n_px": len(s),
                "median_dist_km": float(s.dist_km.median()),
                "H_SWOT_common_m": hs, "within_scene_nmad_m": nmad(s.H_common_m.values),
                "gauge_stage_m": stage, "H_gauge_evrf2019_m": h_gauge,
                "R_swot_minus_gauge_m": hs - h_gauge if np.isfinite(h_gauge) else np.nan,
                "H_ICESat_common_m": h_ice,
                "d_swot_minus_icesat_m": hs - h_ice if np.isfinite(h_ice) else np.nan,
                "gauge_epoch": "daily value, epoch not stated"})
        n1 = int((px.dist_km <= 1.0).sum())
        print(f"{date:%Y-%m-%d} {utc:%H:%M} cyc{cyc}  open-water px<=1km: {n1:5d}  "
              f"H_SWOT_common(1km)="
              f"{np.median(px[px.dist_km<=1.0].H_common_m) if n1 else float('nan'):.3f} m  "
              f"gauge_EVRF2019={h_gauge:.3f} m")

    val = pd.DataFrame(rows)
    val.to_csv(CFG.TABLES / "swot_validation_against_gauge_and_icesat.csv", index=False)
    if per_px:
        allpx = pd.concat(per_px, ignore_index=True)
        OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
        allpx[["date", "swot_utc", "cycle", "pass", "lat", "lon", "dist_km",
               "h_swot_m", "zeta_egg2015_m", "H_common_m"]].to_parquet(OUT_PQ, index=False)

    print("\n=== SWOT vs Kherson gauge (EVRF2019), by aggregation radius ===")
    print(f"{'R km':>5}{'n over':>8}{'median R':>11}{'NMAD':>9}{'CI95':>22}")
    for R in RADII_KM:
        s = val[(val.radius_km == R)].R_swot_minus_gauge_m.dropna()
        if len(s) < 2:
            continue
        lo, hi = boot_ci(s.values)
        print(f"{R:>5.1f}{len(s):>8}{np.median(s):>+11.4f}{nmad(s.values):>9.4f}"
              f"   [{lo:+.4f}, {hi:+.4f}]")

    d = val[(val.radius_km == 1.0)].d_swot_minus_icesat_m.dropna()
    print(f"\nSWOT - ICESat-2 (same day, 1 km): n={len(d)}"
          + (f"  median {np.median(d):+.4f} m" if len(d) else "  (no same-day pairs)"))
    print(f"\n-> {CFG.TABLES/'swot_validation_against_gauge_and_icesat.csv'} ({len(val)} rows)")


if __name__ == "__main__":
    main()
