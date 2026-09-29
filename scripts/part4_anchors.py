#!/usr/bin/env python
"""PART 4 — Kherson and Rozumivka as LOCAL anchors, and whether pooling is allowed.

Six comparisons, each with its own geometry stated:

  1 gauge <-> SWOT        at the gauge (both genuinely within 1 km)
  2 gauge <-> ICESat-2    at each station (2 km tie radius)
  3 SWOT  <-> ICESat-2    where the two sensors ACTUALLY overlap, not at the
                          gauge: ICESat never comes closer than 13 km to the
                          Kherson gauge, so a gauge-mediated sensor difference
                          is spatially confounded and is not computed.
  4 temporal separation
  5 spatial separation
  6 expected hydrological contribution from the daily gauge slope

Then: is a single pooled correction defensible, or must it stay per domain?

Outputs
-------
outputs/tables/part4_anchor_summary.csv
outputs/tables/part4_colocated_swot_icesat.csv
outputs/tables/part4_pooling_test.csv
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
from scipy import stats
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro.vertical import sample_grid

GD = ROOT / "data/processed/gauges"
PIXC_DIR = ROOT / "data/raw/pixc_nova_kakhovka"
KHERSON, ROZUMIVKA = 80805, 80959
MATCH_M, MIN_PX = 500.0, 30
RNG = np.random.default_rng(CFG.SEED)
_GEOD = pyproj.Geod(ellps="WGS84")
_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    m = [np.median(RNG.choice(x, len(x), True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------- #
# 3. SWOT <-> ICESat where they overlap
# --------------------------------------------------------------------------- #
def colocate():
    ice = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kherson_atl13_evrs.parquet",
                          columns=["time", "lat", "lon", "beam", "rgt",
                                   "h_wgs84_m", "zeta_egg2015_m", "H_evrs_egg2015_m"])
    ice["dt"] = pd.to_datetime(ice["time"], utc=True).dt.tz_localize(None)
    ice["date"] = ice.dt.dt.normalize()
    ice["H_common_m"] = ice.H_evrs_egg2015_m + CFG.free2mean(ice.lat.values)

    rows = []
    for f in sorted(PIXC_DIR.glob("SWOT_L2_HR_PIXC_*.nc")):
        m = re.search(r"PIXC_(\d+)_(\d+)_\w+_(\d{8}T\d{6})", f.name)
        cyc, pas, stamp = m.groups()
        sutc = pd.to_datetime(stamp, format="%Y%m%dT%H%M%S")
        date = sutc.normalize()
        seg = ice[ice.date == date]
        if len(seg) < MIN_PX:
            continue

        d = nc.Dataset(f); g = d.groups["pixel_cloud"]
        cls = np.asarray(g.variables["classification"][:])
        k = cls == 4
        plat = np.asarray(g.variables["latitude"][:])[k]
        plon = np.asarray(g.variables["longitude"][:])[k]
        # restrict to the ICESat bounding box + 2 km, then match
        b = (seg.lat.min() - 0.03, seg.lat.max() + 0.03,
             seg.lon.min() - 0.03, seg.lon.max() + 0.03)
        w = (plat >= b[0]) & (plat <= b[1]) & (plon >= b[2]) & (plon <= b[3])
        if w.sum() < MIN_PX:
            d.close(); continue
        idx = np.where(k)[0][w]

        def v(n_):
            return np.asarray(g.variables[n_][:])[idx].astype(float)
        h = v("height") - v("solid_earth_tide") - v("load_tide_fes") - v("pole_tide")
        plat, plon = plat[w], plon[w]
        z = sample_grid(CFG.EGG2015_TIF, plon, plat)
        H = h - z
        d.close()

        px, py = _TO_M.transform(plon, plat)
        sx, sy = _TO_M.transform(seg.lon.values, seg.lat.values)
        tree = cKDTree(np.c_[px, py])
        nbr = tree.query_ball_point(np.c_[sx, sy], r=MATCH_M)
        keep = [i for i, nb in enumerate(nbr) if len(nb) >= MIN_PX]
        if not keep:
            continue
        s = seg.iloc[keep].copy()
        s["H_swot_m"] = [np.median(H[nbr[i]]) for i in keep]
        s["d_m"] = s.H_swot_m - s.H_common_m
        s["dist_gauge_km"] = _GEOD.inv(np.full(len(s), 32.61203), np.full(len(s), 46.62375),
                                       s.lon.values, s.lat.values)[2] / 1000.0
        per_beam = s.groupby("beam").d_m.median()
        iutc = s.dt.min()
        rows.append({"date": date, "swot_utc": sutc, "icesat_utc": iutc,
                     "dt_hours": (sutc - iutc).total_seconds() / 3600.0,
                     "cycle": int(cyc), "n_segments": len(s), "n_beams": s.beam.nunique(),
                     "median_dist_to_gauge_km": float(s.dist_gauge_km.median()),
                     "d_swot_minus_icesat_m": float(np.median(s.d_m)),
                     "per_beam_nmad_m": nmad(per_beam.values),
                     "within_overpass_nmad_m": nmad(s.d_m.values)})
        print(f"  {date:%Y-%m-%d}  n_seg={len(s):5d} beams={s.beam.nunique()} "
              f"dt={rows[-1]['dt_hours']:+6.2f} h  dist~{rows[-1]['median_dist_to_gauge_km']:.0f} km"
              f"  d={rows[-1]['d_swot_minus_icesat_m']:+.4f} m")
    return pd.DataFrame(rows)


def main() -> None:
    gv = pd.read_parquet(GD / "gauge_levels_evrf2019.parquet")
    gv = gv[gv.qc == "ok"]
    al = pd.read_csv(CFG.TABLES / "icesat_station_alignment_constants.csv")
    sw = pd.read_csv(CFG.TABLES / "swot_validation_against_gauge_and_icesat.csv")

    print("=== 3. SWOT <-> ICESat-2 where the sensors overlap ===")
    co = colocate()
    co.to_csv(CFG.TABLES / "part4_colocated_swot_icesat.csv", index=False)

    # ---- 6. hydrological context at Kherson on the co-located dates ---------
    kh = gv[gv.station_id == KHERSON].sort_values("date").set_index("date")
    kh["dHdt_cm_day"] = kh.stage_m.diff().shift(-1) * 100
    if len(co):
        co["gauge_dHdt_cm_day"] = co.date.map(kh.dHdt_cm_day)
        co["expected_dH_cm"] = co.gauge_dHdt_cm_day * co.dt_hours / 24.0
        co.to_csv(CFG.TABLES / "part4_colocated_swot_icesat.csv", index=False)

    # ---- anchor summary ----------------------------------------------------
    rows = []
    for sid, name, dom in [(KHERSON, "Kherson", "downstream"),
                           (ROZUMIVKA, "Rozumivka", "reservoir")]:
        a = al[(al.station_id == sid) & (al.period == "PRE_BREACH") & (al.radius_km == 2.0)]
        ar = al[(al.station_id == sid) & (al.period == "PRE_BREACH")]
        g = gv[gv.station_id == sid]
        rows.append({
            "station_id": sid, "name": name, "domain": dom,
            "gauge_span": f"{g.date.min():%Y-%m-%d}..{g.date.max():%Y-%m-%d}",
            "gauge_n_days": len(g), "gauge_spans_breach": bool(g.date.max() > pd.Timestamp("2023-06-06")),
            "c_pre_2km_m": float(a.alignment_constant_m.iloc[0]) if len(a) else np.nan,
            "c_ci_lo": float(a.ci95_low_m.iloc[0]) if len(a) else np.nan,
            "c_ci_hi": float(a.ci95_high_m.iloc[0]) if len(a) else np.nan,
            "c_nmad_m": float(a.nmad_m.iloc[0]) if len(a) else np.nan,
            "n_matchups": int(a.n_matchups.iloc[0]) if len(a) else 0,
            "c_radius_swing_m": float(ar.alignment_constant_m.max() - ar.alignment_constant_m.min())
                                 if len(ar) > 1 else np.nan,
            "swot_available": sid == KHERSON,
            "c_post_established": False,
            "notes": ("SWOT co-located within 1 km; ICESat never closer than 13 km"
                      if sid == KHERSON else
                      "in-reservoir anchor; ICESat post-breach coverage collapses "
                      "(1 segment within 2 km)")})
    ans = pd.DataFrame(rows)
    ans.to_csv(CFG.TABLES / "part4_anchor_summary.csv", index=False)

    # ---- can one pooled correction serve everything? -----------------------
    pre = al[(al.period == "PRE_BREACH") & (al.radius_km == 2.0)]
    res = pre[pre.domain == "reservoir"].alignment_constant_m.values
    khv = pre[pre.station_id == KHERSON].alignment_constant_m.values
    lo_r, hi_r = boot_ci(res)
    kh_lo = float(pre[pre.station_id == KHERSON].ci95_low_m.iloc[0])
    kh_hi = float(pre[pre.station_id == KHERSON].ci95_high_m.iloc[0])
    kh_med = float(np.median(khv))

    # every verdict below is COMPUTED from the intervals, never asserted
    overlap = min(hi_r, kh_hi) - max(lo_r, kh_lo)          # <0 => disjoint
    zero_in_kh = kh_lo <= 0.0 <= kh_hi
    zero_in_res = lo_r <= 0.0 <= hi_r
    # with 5 stations vs 1 the smallest attainable rank-test p is 1/6, so a rank
    # test cannot reach significance here whatever the data say -- reported, not used
    u = stats.mannwhitneyu(res, khv, alternative="two-sided") if len(khv) else None
    p_floor = 1.0 / len(res) if len(res) else np.nan

    pool = [{
        "test": "reservoir stations pooled (station = unit)",
        "n": len(res), "median_m": float(np.median(res)), "nmad_m": nmad(res),
        "ci95_low_m": lo_r, "ci95_high_m": hi_r,
        "zero_inside_ci": zero_in_res,
        "verdict": f"station-to-station NMAD {nmad(res):.3f} m; CI width "
                   f"{hi_r - lo_r:.3f} m; zero {'inside' if zero_in_res else 'outside'} CI"},
        {"test": "Kherson (downstream)", "n": len(khv), "median_m": kh_med,
         "nmad_m": np.nan, "ci95_low_m": kh_lo, "ci95_high_m": kh_hi,
         "zero_inside_ci": zero_in_kh,
         "verdict": f"zero {'inside' if zero_in_kh else 'outside'} CI "
                    f"-> {'no correction needed' if zero_in_kh else 'correction needed'}"},
        {"test": "reservoir vs Kherson: may they be pooled?",
         "n": len(res) + len(khv), "median_m": float(np.median(res) - kh_med),
         "nmad_m": np.nan, "ci95_low_m": np.nan, "ci95_high_m": np.nan,
         "zero_inside_ci": np.nan,
         "verdict": (f"CI overlap {overlap:+.3f} m -> "
                     f"{'DISJOINT, do not pool' if overlap < 0 else 'overlapping'}; "
                     f"rank test uninformative here (n=5 vs 1, p floor {p_floor:.3f}"
                     + (f", observed p={u.pvalue:.3f})" if u is not None else ")"))}]
    pt = pd.DataFrame(pool)
    pt.to_csv(CFG.TABLES / "part4_pooling_test.csv", index=False)

    print("\n=== 1-2,4-5. anchor summary ===")
    with pd.option_context("display.width", 220):
        print(ans.drop(columns=["notes"]).to_string(index=False))
    print("\n=== gauge <-> SWOT at Kherson (radius ladder) ===")
    for R in sorted(sw.radius_km.unique()):
        s = sw[sw.radius_km == R].R_swot_minus_gauge_m.dropna()
        lo, hi = boot_ci(s.values)
        print(f"  R={R:>4.1f} km  n={len(s)}  median {np.median(s):+.4f}  "
              f"NMAD {nmad(s.values):.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print("\n=== can one correction serve all stations? ===")
    print(pt.to_string(index=False))
    print(f"\nreservoir stations: {np.round(res,4).tolist()}")
    print(f"Kherson: {np.round(khv,4).tolist()}"
          + (f"  Mann-Whitney p={u.pvalue:.4f}" if u is not None else ""))


if __name__ == "__main__":
    main()
