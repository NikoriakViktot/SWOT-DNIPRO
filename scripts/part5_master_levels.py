#!/usr/bin/env python
"""PART 5 — one master table of every water level in a common comparable frame.

Provenance is never collapsed. Three distinct things stay in three distinct
columns and are never summed silently:

  transformed_level_m   official geodetic transformation only
                        (gauge: BS-77 -> EVRF2019 via EPSG:9902;
                         satellite: ellipsoidal -> EGG2015 common frame)
  empirical_correction_m  station/domain-specific empirical alignment, applied
                        ONLY where it was actually estimated. The Kherson
                        constant is never applied to the reservoir and vice versa.
  corrected_level_m     transformed + empirical (NaN where no constant is
                        defensible, never silently 0)

Outputs
-------
data/processed/master/all_water_levels_common_frame.parquet
outputs/tables/all_water_levels_common_frame.csv
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
import pyproj

from swot_dnipro import config as CFG

GD = ROOT / "data/processed/gauges"
OUT_PQ = ROOT / "data/processed/master/all_water_levels_common_frame.parquet"
BREACH = pd.Timestamp("2023-06-06")
TIE_RADIUS_KM = 2.0
_GEOD = pyproj.Geod(ellps="WGS84")

COLS = ["source", "source_type", "station_or_domain", "domain", "datetime", "date",
        "lat", "lon", "raw_level_m", "raw_vertical_frame", "transformed_level_m",
        "common_vertical_frame", "empirical_correction_m", "correction_source",
        "corrected_level_m", "comparable_to_gauge", "n_obs", "spread_m",
        "period", "qc", "notes"]


def main() -> None:
    gv = pd.read_parquet(GD / "gauge_levels_evrf2019.parquet")
    ice_st = pd.read_parquet(GD / "icesat_levels_common_frame.parquet")
    swpx = pd.read_parquet(GD / "swot_levels_common_frame.parquet")
    al = pd.read_csv(CFG.TABLES / "icesat_station_alignment_constants.csv")

    # PRE-breach, station-local constants only; nothing pooled, nothing borrowed
    cst = al[(al.period == "PRE_BREACH") & (al.radius_km == TIE_RADIUS_KM)]
    cmap = dict(zip(cst.station_id, cst.alignment_constant_m))
    cci = {r.station_id: (r.ci95_low_m, r.ci95_high_m, r.n_matchups)
           for r in cst.itertuples()}

    frames = []

    # ---------------- gauges -------------------------------------------------
    g = gv.copy()
    frames.append(pd.DataFrame({
        "source": "gauge", "source_type": "in_situ",
        "station_or_domain": "station:" + g.station_id.astype(str),
        "domain": g.domain, "datetime": g.date, "date": g.date,
        "lat": g.lat, "lon": g.lon,
        "raw_level_m": g.stage_m,
        "raw_vertical_frame": "BS-77 stage above gauge zero",
        "transformed_level_m": g.H_evrf2019_m,
        "common_vertical_frame": "EVRF2019 (EPSG:9389, zero-tide)",
        "empirical_correction_m": 0.0,
        "correction_source": "none — official geodetic transform only (EPSG:9902)",
        "corrected_level_m": g.H_evrf2019_m,
        "comparable_to_gauge": True,
        "n_obs": 1, "spread_m": np.nan,
        "period": np.where(g.date < BREACH, "PRE_BREACH", "POST_BREACH"),
        "qc": g.qc,
        "notes": "zero_bs77=" + g.zero_bs77_m.round(2).astype(str)
                 + "; delta_9902=" + g.delta_epsg9902_m.round(4).astype(str)
                 + "; " + g.series}))

    # ---------------- ICESat-2, station-local (tie-comparable) ---------------
    st_meta = gv.groupby("station_id")[["lat", "lon", "domain"]].first()
    i = ice_st.copy()
    i = i.merge(st_meta, left_on="station_id", right_index=True, how="left")
    corr = i.station_id.map(cmap)
    corr = corr.where(i.period == "PRE_BREACH")          # no POST constant is defensible
    note = []
    for r in i.itertuples():
        if r.period == "PRE_BREACH" and r.station_id in cci:
            lo, hi, n = cci[r.station_id]
            note.append(f"c from station {r.station_id} PRE, n={n}, CI[{lo:+.3f},{hi:+.3f}]")
        else:
            note.append("no defensible post-breach constant at this station "
                        "(ICESat water coverage collapses after the breach)")
    frames.append(pd.DataFrame({
        "source": "ICESat-2", "source_type": "satellite_altimetry",
        "station_or_domain": "station:" + i.station_id.astype(str),
        "domain": i.domain, "datetime": i.date, "date": i.date,
        "lat": i.lat, "lon": i.lon,
        "raw_level_m": i.H_common_raw_m,
        "raw_vertical_frame": "EGG2015 (h_ATL13 - zeta), tide-free ATL03",
        "transformed_level_m": i.H_common_m,
        "common_vertical_frame": "EGG2015 common frame, permanent-tide harmonised",
        "empirical_correction_m": corr,
        "correction_source": np.where(corr.notna(),
                                      "station-specific c_station (Part 2)", "none"),
        "corrected_level_m": i.H_common_m + corr,
        "comparable_to_gauge": True,
        "n_obs": i.n_segments, "spread_m": i.within_nmad_m,
        "period": i.period, "qc": "ok",
        "notes": note}))

    # ---------------- SWOT, per overpass x radius ----------------------------
    kh = gv[gv.station_id == 80805]
    lat0, lon0 = float(kh.lat.iloc[0]), float(kh.lon.iloc[0])
    for R in (0.5, 1.0, 2.0, 5.0):
        s = swpx[swpx.dist_km <= R]
        if s.empty:
            continue
        a = s.groupby(["date", "swot_utc"]).agg(
            H=("H_common_m", "median"), h_raw=("h_swot_m", "median"),
            n=("H_common_m", "size"),
            sp=("H_common_m", lambda v: 1.4826 * np.median(np.abs(v - np.median(v))))
        ).reset_index()
        frames.append(pd.DataFrame({
            "source": "SWOT", "source_type": "satellite_altimetry",
            "station_or_domain": f"station:80805@{R:g}km",
            "domain": "downstream", "datetime": a.swot_utc, "date": a.date,
            "lat": lat0, "lon": lon0,
            "raw_level_m": a.h_raw,
            "raw_vertical_frame": "ellipsoidal, PIXC height minus solid-earth/load/pole tides",
            "transformed_level_m": a.H,
            "common_vertical_frame": "EGG2015 common frame",
            "empirical_correction_m": np.nan,
            "correction_source": "none — independently validated, not corrected",
            "corrected_level_m": np.nan,
            "comparable_to_gauge": True,
            "n_obs": a.n, "spread_m": a.sp,
            "period": "PRE_BREACH", "qc": "ok",
            "notes": f"open-water pixels within {R:g} km of gauge 80805"}))

    m = pd.concat(frames, ignore_index=True)[COLS].sort_values(
        ["source", "station_or_domain", "datetime"]).reset_index(drop=True)

    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT_PQ, index=False)
    m.to_csv(CFG.TABLES / "all_water_levels_common_frame.csv", index=False)

    print(f"-> {OUT_PQ}\n-> {CFG.TABLES/'all_water_levels_common_frame.csv'}")
    print(f"\n{len(m):,} rows\n")
    print("=== rows by source x period ===")
    print(m.pivot_table(index="source", columns="period", values="date",
                        aggfunc="count", fill_value=0).to_string())
    print("\n=== empirical corrections actually applied (station-specific only) ===")
    ap = (m[m.empirical_correction_m.notna() & (m.empirical_correction_m != 0)]
          .groupby(["source", "station_or_domain", "domain"])
          .agg(n=("date", "size"), c=("empirical_correction_m", "first")).reset_index())
    print(ap.to_string(index=False) if len(ap) else "  none")
    print("\n=== provenance classes ===")
    print(m.groupby(["source", "correction_source"]).size().to_string())


if __name__ == "__main__":
    main()
