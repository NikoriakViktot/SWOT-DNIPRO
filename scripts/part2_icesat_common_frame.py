#!/usr/bin/env python
"""PART 2 — ICESat-2 in the common vertical frame + LOCAL alignment constants.

    H_common      = h_ATL13 + tide_earth_free2mean(lat) - zeta_EGG2015     (harmonised)
    H_common_raw  = h_ATL13                             - zeta_EGG2015     (as the
                                                          ICESat-2 project ships it)

    c_station = median( H_gauge_EVRF2019 - H_ICESat_common )

Constants are computed SEPARATELY per station and per period. No single regional
correction is imposed, and the Kherson constant is never applied to the reservoir.

Outputs
-------
outputs/tables/icesat_station_alignment_constants.csv
outputs/tables/icesat_station_matchups.csv
data/processed/gauges/icesat_levels_common_frame.parquet
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

ICE = CFG.ICESAT_ROOT
GAUGES = ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet"
OUT_PQ = ROOT / "data/processed/gauges/icesat_levels_common_frame.parquet"
BREACH = pd.Timestamp("2023-06-06")
RNG = np.random.default_rng(CFG.SEED)
RADII_KM = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
MIN_SEG = 20                     # segments required for a local beam-pass level

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


def load_icesat(domain):
    f = "kherson_atl13_evrs.parquet" if domain == "downstream" else "kakhovka_atl13_evrs.parquet"
    d = pd.read_parquet(ICE / "data/processed" / f,
                        columns=["time", "lat", "lon", "rgt", "beam", "h_wgs84_m",
                                 "zeta_egg2015_m", "H_evrs_egg2015_m", "period",
                                 "water_mask_pass"])
    d = d[d.water_mask_pass] if "water_mask_pass" in d else d
    d["dt"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
    d["date"] = d.dt.dt.normalize()
    # permanent-tide harmonisation: ATL03 heights are tide-free, EGG2015 is zero-tide
    d["free2mean_m"] = CFG.free2mean(d.lat.values)
    d["H_common_raw_m"] = d.H_evrs_egg2015_m                      # project convention
    d["H_common_m"] = d.H_evrs_egg2015_m + d.free2mean_m          # harmonised
    return d


def local_pass_levels(seg, lat0, lon0, radius_km):
    """One level per (date, rgt, beam) from segments inside the radius."""
    _, _, dist = _GEOD.inv(np.full(len(seg), lon0), np.full(len(seg), lat0),
                           seg.lon.values, seg.lat.values)
    s = seg.loc[dist <= radius_km * 1000.0].copy()
    if s.empty:
        return pd.DataFrame()
    s["dist_km"] = dist[dist <= radius_km * 1000.0] / 1000.0
    g = s.groupby(["date", "rgt", "beam"]).agg(
        n_segments=("H_common_m", "size"),
        H_common_m=("H_common_m", "median"),
        H_common_raw_m=("H_common_raw_m", "median"),
        within_nmad_m=("H_common_m", lambda v: nmad(v)),
        mean_dist_km=("dist_km", "mean"),
        period=("period", "first")).reset_index()
    return g[g.n_segments >= MIN_SEG]


def main() -> None:
    gv = pd.read_parquet(GAUGES)
    gv = gv[gv.qc == "ok"]                       # fill_suspect rows never enter a tie
    st = (gv.groupby(["station_id", "name_en", "domain", "lat", "lon"])
            .size().reset_index(name="n_gauge_obs"))

    ice = {d: load_icesat(d) for d in ("reservoir", "downstream")}

    rows, matchups, levels = [], [], []
    for _, s in st.iterrows():
        seg = ice[s.domain]
        g = gv[gv.station_id == s.station_id][["date", "H_evrf2019_m", "stage_m"]]
        for R in RADII_KM:
            lv = local_pass_levels(seg, s.lat, s.lon, R)
            if lv.empty:
                continue
            lv["station_id"] = s.station_id
            lv["radius_km"] = R
            m = lv.merge(g, on="date", how="inner")
            if m.empty:
                continue
            m["d_raw_m"] = m.H_evrf2019_m - m.H_common_raw_m
            m["d_harm_m"] = m.H_evrf2019_m - m.H_common_m
            m["station_id"] = s.station_id
            m["name_en"] = s.name_en
            m["domain"] = s.domain
            matchups.append(m)
            if R == 2.0:
                levels.append(lv)

            for per, sub in [("PRE_BREACH", m[m.date < BREACH]),
                             ("POST_BREACH", m[m.date >= BREACH]),
                             ("ALL", m)]:
                if len(sub) < 3:
                    continue
                lo, hi = boot_ci(sub.d_harm_m.values)
                rows.append({
                    "station_id": s.station_id, "name_en": s.name_en,
                    "domain": s.domain, "period": per,
                    "radius_km": R, "radius_m": R * 1000,
                    "n_matchups": len(sub), "n_dates": sub.date.nunique(),
                    "n_rgts": sub.rgt.nunique(),
                    "time_window": f"{sub.date.min():%Y-%m-%d}..{sub.date.max():%Y-%m-%d}",
                    "alignment_constant_m": float(np.median(sub.d_harm_m)),
                    "alignment_constant_raw_m": float(np.median(sub.d_raw_m)),
                    "nmad_m": nmad(sub.d_harm_m.values),
                    "ci95_low_m": lo, "ci95_high_m": hi,
                    "median_H_icesat_common_m": float(np.median(sub.H_common_m)),
                    "median_H_gauge_evrf2019_m": float(np.median(sub.H_evrf2019_m)),
                    "notes": ""})

    al = pd.DataFrame(rows)
    mu = pd.concat(matchups, ignore_index=True) if matchups else pd.DataFrame()
    al.to_csv(CFG.TABLES / "icesat_station_alignment_constants.csv", index=False)
    mu.to_csv(CFG.TABLES / "icesat_station_matchups.csv", index=False)
    if levels:
        lvdf = pd.concat(levels, ignore_index=True)
        OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
        lvdf.to_parquet(OUT_PQ, index=False)

    # ---- regression against the ICESat-2 project's own Phase 2c values --------
    ref = pd.read_csv(ICE / "outputs/tables/egg2015_to_evrf2019_by_station.csv")
    print("=== regression vs ICESat-2 project Phase 2c (PRE_BREACH, its own radius) ===")
    print(f"{'station':<18}{'R km':>5}{'ref c_C':>10}{'ours raw':>10}{'diff':>9}")
    for _, r in ref.iterrows():
        q = al[(al.station_id == r.station_id) & (al.period == "PRE_BREACH")
               & (al.radius_km == r.reported_radius_km)]
        if q.empty:
            print(f"{r.name_en:<18}{r.reported_radius_km:>5.0f}{r.c_station_C_dailymean_m:>10.4f}"
                  f"{'--':>10}{'--':>9}")
            continue
        ours = float(q.alignment_constant_raw_m.iloc[0])
        print(f"{r.name_en:<18}{r.reported_radius_km:>5.0f}"
              f"{r.c_station_C_dailymean_m:>10.4f}{ours:>10.4f}"
              f"{ours - r.c_station_C_dailymean_m:>+9.4f}")

    print("\n=== alignment constants, radius 2 km (harmonised) ===")
    v = al[(al.radius_km == 2.0)].sort_values(["domain", "station_id", "period"])
    cols = ["station_id", "name_en", "domain", "period", "n_matchups", "n_dates",
            "alignment_constant_m", "nmad_m", "ci95_low_m", "ci95_high_m", "time_window"]
    with pd.option_context("display.width", 220):
        print(v[cols].to_string(index=False))
    print(f"\n-> {CFG.TABLES/'icesat_station_alignment_constants.csv'} ({len(al)} rows)")
    print(f"-> {CFG.TABLES/'icesat_station_matchups.csv'} ({len(mu)} rows)")


if __name__ == "__main__":
    main()
