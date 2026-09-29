#!/usr/bin/env python
"""PART 8 — ATL03/ATL08 terrain and canopy over the former Kakhovka pool.

SlideRule `atl08p` runs PhoREAL on ATL03 photons carrying the ATL08 ground /
canopy / top-of-canopy labels, and returns one row per 100 m segment with a
median TERRAIN height and canopy metrics.

Why this matters here: the Sentinel-1 branch was closed because no bare-earth
DEM of the drained lakebed exists (GLO-30 captured the *filled* reservoir at a
flat ~16 m). Post-breach ATL08 ground returns inside the former pool ARE that
missing surface, and the canopy metrics measure the vegetation colonising it.

Vertical chain, identical to the ATL13 branch:

    H_common = h_te_median + tide_earth_free2mean(lat) - zeta_EGG2015

ATL03 photon heights are tide-free (ATL03 ATBD v007 p.8), EGG2015 is zero-tide.

Outputs
-------
data/processed/atl08/kakhovka_atl08_terrain.parquet
outputs/tables/atl08_terrain_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyproj
import sliderule
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from sliderule import icesat2

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD
from swot_dnipro.vertical import sample_grid

OUT_PQ = ROOT / "data/processed/atl08/kakhovka_atl08_terrain.parquet"
BBOX = (33.30, 46.70, 35.40, 47.90)          # same AOI as the ATL13 branch
BREACH = pd.Timestamp("2023-06-06")
POST0 = pd.Timestamp("2023-09-01")
TO_LL = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform
TO_UTM = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def poly_from_bbox(b):
    lo, la, hi, ha = b
    return [{"lon": lo, "lat": la}, {"lon": hi, "lat": la}, {"lon": hi, "lat": ha},
            {"lon": lo, "lat": ha}, {"lon": lo, "lat": la}]


def pull(years) -> pd.DataFrame:
    sliderule.init("slideruleearth.io", verbose=False)
    poly = poly_from_bbox(BBOX)
    out = []
    for y0, y1 in years:
        parms = {
            "poly": poly, "t0": f"{y0}-01-01", "t1": f"{y1}-01-01",
            "srt": icesat2.SRT_LAND, "cnf": icesat2.CNF_SURFACE_HIGH,
            "atl08_class": ["atl08_ground", "atl08_canopy", "atl08_top_of_canopy"],
            "len": 100, "res": 100,
            "phoreal": {"binsize": 1.0, "geoloc": "center",
                        "use_abs_h": False, "send_waveform": False},
        }
        t = time.time()
        try:
            g = icesat2.atl08p(parms)
        except Exception as e:
            print(f"  {y0}: FAILED {type(e).__name__}: {str(e)[:90]}")
            continue
        if len(g):
            g = g.reset_index()
            out.append(g)
        print(f"  {y0}: {len(g):>7,} segments in {time.time()-t:>5.0f}s", flush=True)
    if not out:
        raise SystemExit("no ATL08 data returned")
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true",
                    help="skip the SlideRule pull and re-derive from the raw parquet")
    a = ap.parse_args()

    raw_pq = OUT_PQ.with_name("kakhovka_atl08_raw.parquet")
    if a.use_cache and raw_pq.exists():
        df = pd.read_parquet(raw_pq)
        print(f"cache: {len(df):,} segments")
    else:
        print(f"SlideRule atl08p (PhoREAL) over {BBOX}")
        df = pull([(y, y + 1) for y in range(2018, 2026)])
        OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_pq, index=False)
        print(f"-> {raw_pq}  ({len(df):,} segments)")

    # geometry -> plain columns
    if "geometry" in df.columns:
        df["lon"] = df.geometry.x if hasattr(df.geometry, "x") else df.lon
        df["lat"] = df.geometry.y if hasattr(df.geometry, "y") else df.lat
        df = df.drop(columns=["geometry"])
    tcol = "time" if "time" in df.columns else df.columns[0]
    df["dt"] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["dt", "lat", "lon", "h_te_median"])
    df["date"] = df.dt.dt.normalize()

    # ---- common vertical frame, same chain as ATL13 ------------------------
    z = sample_grid(CFG.EGG2015_TIF, df.lon.values, df.lat.values)
    df["zeta_egg2015_m"] = z
    df["free2mean_m"] = CFG.free2mean(df.lat.values)
    df["H_terrain_common_m"] = df.h_te_median + df.free2mean_m - z
    df["H_canopy_top_common_m"] = df.H_terrain_common_m + df.h_canopy

    df["period"] = np.where(df.dt < BREACH, "PRE_BREACH",
                            np.where(df.dt < POST0, "BREACH_DRAWDOWN", "POST_BREACH"))

    # ---- inside the former pool? -------------------------------------------
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
          SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    from shapely import points as _pts, contains as _contains
    # fp is in CFG.CRS_METRIC, the segments are lon/lat: move the POINTS into the
    # metric CRS rather than bending the polygon, per the project CRS policy.
    _px, _py = TO_UTM(df.lon.values, df.lat.values)
    df["in_former_pool"] = _contains(fp, _pts(_px, _py))

    # A pre-breach segment inside the pool is a WATER surface, not the bed:
    # ATL08 terrain there is the pool top. Only post-breach segments inside the
    # footprint expose the bed, and that distinction is carried explicitly.
    df["surface_class"] = np.where(
        ~df.in_former_pool, "outside_pool_land",
        np.where(df.period == "PRE_BREACH", "pool_water_surface", "exposed_bed"))

    df.to_parquet(OUT_PQ, index=False)
    print(f"-> {OUT_PQ}  ({len(df):,} segments)")

    # ---- summary -----------------------------------------------------------
    rows = []
    for (per, sc), g in df.groupby(["period", "surface_class"]):
        veg = g[g.h_canopy > 0]
        rows.append({
            "period": per, "surface_class": sc, "n_segments": len(g),
            "n_dates": g.date.nunique(),
            "terrain_p05_m": float(np.percentile(g.H_terrain_common_m, 5)),
            "terrain_median_m": float(g.H_terrain_common_m.median()),
            "terrain_p95_m": float(np.percentile(g.H_terrain_common_m, 95)),
            "frac_with_canopy": float((g.h_canopy > 0).mean()),
            "canopy_median_m": float(veg.h_canopy.median()) if len(veg) else np.nan,
            "canopy_p95_m": float(np.percentile(veg.h_canopy, 95)) if len(veg) else np.nan,
            "gnd_ph_median": float(g.gnd_ph_count.median()),
            "veg_ph_median": float(g.veg_ph_count.median())})
    s = pd.DataFrame(rows).sort_values(["surface_class", "period"])
    s.to_csv(CFG.TABLES / "atl08_terrain_summary.csv", index=False)

    print(f"-> {CFG.TABLES/'atl08_terrain_summary.csv'}\n")
    with pd.option_context("display.width", 200):
        print(s.to_string(index=False))

    bed = df[df.surface_class == "exposed_bed"]
    if len(bed):
        print(f"\n=== exposed lakebed: the surface that did not exist before ===")
        print(f"{len(bed):,} ATL08 ground segments over {bed.date.nunique()} dates, "
              f"{bed.date.min():%Y-%m-%d}..{bed.date.max():%Y-%m-%d}")
        print(f"bed elevation  p05 {np.percentile(bed.H_terrain_common_m,5):.1f}  "
              f"median {bed.H_terrain_common_m.median():.1f}  "
              f"p95 {np.percentile(bed.H_terrain_common_m,95):.1f} m EVRS")
        yr = bed.groupby(bed.dt.dt.year).agg(
            n=("h_canopy", "size"), veg_frac=("h_canopy", lambda v: (v > 0).mean()),
            canopy_med=("h_canopy", lambda v: v[v > 0].median() if (v > 0).any() else np.nan),
            canopy_p95=("h_canopy", lambda v: np.percentile(v[v > 0], 95) if (v > 0).any() else np.nan))
        print("\nvegetation colonising the exposed bed, by year:")
        print(yr.round(3).to_string())


if __name__ == "__main__":
    main()
