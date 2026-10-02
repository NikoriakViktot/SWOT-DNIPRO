#!/usr/bin/env python
"""Compute every figure's machine-readable source table into outputs/figure_data/.

All science lives here or in ``src/swot_dnipro``; figure scripts only plot.
Run:  python scripts/make_figure_data.py
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import bootstrap_ci, nmad
from swot_dnipro.vertical import (
    atl13_to_mean_tide, haversine_km, pixc_ellipsoidal_height, read_pixc, sample_grid,
)

FD = CFG.FIGDATA
FD.mkdir(parents=True, exist_ok=True)
KH_LON, KH_LAT = CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3]
SCRATCH = Path("/tmp/claude-1000/-home-niko-projects-icesat2-atl13-kakhovka/"
               "11c576e3-cf8a-4ab9-8372-bf5a24c5336e/scratchpad")


def _gauge_daily() -> pd.DataFrame:
    g = pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET)
    g = g[g["stat_type"] == "daily"].copy()
    g["date"] = pd.to_datetime(g["date"])
    return g[["date", "water_level_cm", "water_level_m_abs"]].rename(
        columns={"water_level_m_abs": "stage_bs77_m"})


def _delta9902(lon, lat) -> float:
    return float(sample_grid(CFG.UA2019Z_ASC, [lon], [lat])[0])


# --------------------------------------------------------------------------- #
# Fig 03 — PIXC/RiverSP sign validation                                        #
# --------------------------------------------------------------------------- #
def fig03():
    shp = glob.glob(str(SCRATCH / "rvsign" / "*482_001*" / "*.shp"))
    if not shp:
        print("  [fig03] RiverSP granule not present — skipped")
        return
    rv = gpd.read_file(shp[0], bbox=(32.3, 46.3, 33.5, 47.2))
    rv = rv[(rv["wse"] > -1e10) & (rv["geoid_hght"] > -1e10) & (rv["node_q"] <= 2)]

    f = sorted(CFG.PIXC_DIR.glob("*482_001*.nc"))[0]
    d = read_pixc(f, bbox=(32.3, 46.3, 33.5, 47.2))
    t = d["solid_earth_tide"] + d["load_tide_fes"] + d["pole_tide"]
    variants = {
        "A": ("height − (st+lt+pt)", d["height"] - t, "hypothesised / documented"),
        "B": ("height", d["height"], "no tide correction"),
        "C": ("height + (st+lt+pt)", d["height"] + t, "wrong sign"),
        "D": ("height − geoid", d["height"] - d["geoid"], "geoid blunder"),
    }
    rows = []
    for _, n in rv.iterrows():
        dist = haversine_km(d["lon"], d["lat"], n["lon"], n["lat"])
        m = dist < 0.3
        if m.sum() < 50:
            continue
        ref = n["wse"] + n["geoid_hght"]
        rec = {"node_id": n["node_id"], "n_px": int(m.sum()), "h_riversp_m": ref,
               "geoid_riversp_m": n["geoid_hght"], "geoid_pixc_m": float(np.nanmedian(d["geoid"][m]))}
        for k, (_, arr, _) in variants.items():
            rec[f"diff_{k}_m"] = float(np.nanmedian(arr[m])) - ref
        rows.append(rec)
    per = pd.DataFrame(rows)
    per.to_csv(FD / "Fig03_pixc_riversp_sign_validation_pernode.csv", index=False)

    summ = [{"variant": k, "chain": lbl, "interpretation": note,
             "median_diff_m": float(np.median(per[f"diff_{k}_m"])),
             "nmad_m": nmad(per[f"diff_{k}_m"]),
             "ci95_low_m": bootstrap_ci(per[f"diff_{k}_m"], seed=CFG.SEED)[0],
             "ci95_high_m": bootstrap_ci(per[f"diff_{k}_m"], seed=CFG.SEED)[1],
             "n_nodes": len(per)}
            for k, (lbl, _, note) in variants.items()]
    pd.DataFrame(summ).to_csv(FD / "Fig03_pixc_riversp_sign_validation.csv", index=False)
    print(f"  [fig03] n={len(per)} nodes; A median={summ[0]['median_diff_m']:+.4f} m")


# --------------------------------------------------------------------------- #
# Fig 04 / Fig 15 — permanent-tide effect on the six correctors                #
# --------------------------------------------------------------------------- #
def fig04_fig15():
    src = pd.read_csv(CFG.CORRECTOR_BY_STATION)
    src["free2mean_m"] = CFG.free2mean(src["lat"])
    # H_ICESat rises/falls by free2mean, so c = H_gauge - H_ICESat moves by -free2mean
    src["c_original_m"] = src["c_station_m"]
    src["c_harmonised_m"] = src["c_station_m"] - src["free2mean_m"]
    src["delta_c_m"] = src["c_harmonised_m"] - src["c_original_m"]
    cols = ["station_id", "name_en", "slug", "lat", "lon", "delta_epsg9902_m",
            "n_matchups", "n_dates", "n_rgts", "empirical_nmad_m",
            "bootstrap_ci95_low_m", "bootstrap_ci95_high_m",
            "free2mean_m", "c_original_m", "c_harmonised_m", "delta_c_m"]
    out = src[cols].sort_values("lat")
    out.to_csv(FD / "Fig04_permanent_tide_correctors.csv", index=False)
    out.to_csv(FD / "Fig15_corrector_points.csv", index=False)

    summary = pd.DataFrame([{
        "quantity": "regional median c",
        "original_m": float(np.median(src["c_original_m"])),
        "harmonised_m": float(np.median(src["c_harmonised_m"])),
        "shift_m": float(np.median(src["c_harmonised_m"]) - np.median(src["c_original_m"])),
        "station_to_station_nmad_original_m": nmad(src["c_original_m"]),
        "station_to_station_nmad_harmonised_m": nmad(src["c_harmonised_m"]),
        "n_stations": len(src)}])
    summary.to_csv(FD / "Fig04_regional_median_summary.csv", index=False)
    print(f"  [fig04] regional median {summary.original_m[0]:+.4f} -> {summary.harmonised_m[0]:+.4f} m")


# --------------------------------------------------------------------------- #
# Kherson pilot core — Figs 05, 07, 08, 13, 14                                 #
# --------------------------------------------------------------------------- #
def _swot_per_date(radius_km=1.0, classes=(4,)):
    rows = []
    for f in sorted(CFG.PIXC_DIR.glob("*.nc")):
        d = read_pixc(f, bbox=(32.3, 46.3, 33.5, 47.2), water_classes=classes)
        dist = haversine_km(d["lon"], d["lat"], KH_LON, KH_LAT)
        m = dist < radius_km
        if m.sum() < 20:
            continue
        z = sample_grid(CFG.EGG2015_TIF, d["lon"][m], d["lat"][m])
        H = d["h_ell"][m] - z
        H_raw = d["height"][m] - z
        t = pd.Timestamp(d["time_granule_start"]).tz_localize(None)
        rows.append({
            "granule": f.name, "cycle": d["cycle"], "pass": d["pass"], "tile": d["tile"],
            "date": t.normalize(), "swot_utc": t, "radius_km": radius_km,
            "water_classes": "+".join(map(str, classes)), "n_px": int(m.sum()),
            "H_SWOT_EGG2015_m": float(np.nanmedian(H)),
            "H_SWOT_EGG2015_raw_m": float(np.nanmedian(H_raw)),
            "nmad_m": nmad(H), "p05_m": float(np.nanpercentile(H, 5)),
            "p95_m": float(np.nanpercentile(H, 95)),
            "median_tide_term_m": float(np.nanmedian(
                (d["solid_earth_tide"] + d["load_tide_fes"] + d["pole_tide"])[m])),
            "median_zeta_egg2015_m": float(np.nanmedian(z)),
            "median_geoid_egm2008_m": float(np.nanmedian(d["geoid"][m])),
            "median_dist_km": float(np.median(dist[m])),
        })
    return pd.DataFrame(rows)


def _icesat_per_date():
    ic = pd.read_parquet(CFG.KHERSON_ATL13)
    ic["dt"] = pd.to_datetime(ic["datetime"], utc=True).dt.tz_localize(None)
    ic["date"] = ic["dt"].dt.normalize()
    pre = ic[(ic["dt"] >= "2023-01-01") & (ic["dt"] < CFG.BREACH_DATE)]
    per = pre.groupby(["date", "rgt"]).agg(
        icesat_utc=("dt", "min"), n_beams=("beam", "nunique"), n_points=("n_points", "sum"),
        H_tidefree_m=("median_wse_evrs_m", "median"),
        beam_spread_m=("median_wse_evrs_m", lambda x: x.max() - x.min()),
        lat=("lat_mean", "median"), lon=("lon_mean", "median")).reset_index()
    per["free2mean_m"] = CFG.free2mean(per["lat"])
    per["H_meantide_m"] = atl13_to_mean_tide(per["H_tidefree_m"], per["lat"])
    per["dist_to_gauge_km"] = haversine_km(per["lon"], per["lat"], KH_LON, KH_LAT)
    per["qc_flag"] = np.where(per["beam_spread_m"] > 1.0, "BEAM_SPREAD_GT_1M", "ok")
    return per


def kherson_core():
    d9902 = _delta9902(KH_LON, KH_LAT)
    g = _gauge_daily()
    sw = _swot_per_date().merge(g, on="date", how="left")
    ic = _icesat_per_date().merge(g, on="date", how="left")

    sw["delta_epsg9902_m"] = d9902
    sw["H_gauge_EVRF2019_m"] = sw["stage_bs77_m"] + d9902
    sw["c_SWOT_m"] = sw["H_gauge_EVRF2019_m"] - sw["H_SWOT_EGG2015_m"]
    sw["R_SWOT_m"] = sw["H_SWOT_EGG2015_m"] - sw["stage_bs77_m"]
    sw["gauge_epoch"] = "daily value — epoch unresolved"

    ic["delta_epsg9902_m"] = d9902
    ic["H_gauge_EVRF2019_m"] = ic["stage_bs77_m"] + d9902
    ic["c_ICESat_m"] = ic["H_gauge_EVRF2019_m"] - ic["H_meantide_m"]
    ic["R_ICESat_m"] = ic["H_meantide_m"] - ic["stage_bs77_m"]

    sw.to_csv(FD / "Fig05_kherson_timeseries_swot.csv", index=False)
    ic.to_csv(FD / "Fig05_kherson_timeseries_icesat.csv", index=False)

    clean = ic[ic["qc_flag"] == "ok"]
    # Fig 07 forest
    forest = []
    for label, x, n_unit in [("SWOT (PIXC)", sw["c_SWOT_m"], "SWOT overpasses"),
                             ("ICESat-2 (ATL13)", clean["c_ICESat_m"], "date x RGT")]:
        lo, hi = bootstrap_ci(x, seed=CFG.SEED)
        forest.append({"sensor": label, "statistic": "median (satellite − gauge, sign-flipped c)",
                       "median_c_m": float(np.nanmedian(x)), "nmad_m": nmad(x),
                       "ci95_low_m": lo, "ci95_high_m": hi, "n": int(np.isfinite(x).sum()),
                       "independent_unit": n_unit})
    for label, x, n_unit in [("SWOT (PIXC)", sw["R_SWOT_m"], "SWOT overpasses"),
                             ("ICESat-2 (ATL13)", clean["R_ICESat_m"], "date x RGT")]:
        lo, hi = bootstrap_ci(x, seed=CFG.SEED)
        forest.append({"sensor": label, "statistic": "median R = H_sat − stage",
                       "median_c_m": float(np.nanmedian(x)), "nmad_m": nmad(x),
                       "ci95_low_m": lo, "ci95_high_m": hi, "n": int(np.isfinite(x).sum()),
                       "independent_unit": n_unit})
    pd.DataFrame(forest).to_csv(FD / "Fig07_sensor_residuals.csv", index=False)

    # Fig 08 harmonisation variants
    var = []
    combos = [
        ("V0", sw["H_SWOT_EGG2015_raw_m"] - sw["stage_bs77_m"], clean["H_tidefree_m"] - clean["stage_bs77_m"],
         "SWOT raw height vs ATL13 tide-free", "physically inconsistent (time-variable tides)"),
        ("V1", sw["R_SWOT_m"], clean["H_tidefree_m"] - clean["stage_bs77_m"],
         "SWOT tide-corrected vs ATL13 tide-free", "incomplete permanent-tide harmonisation"),
        ("V2", sw["R_SWOT_m"], clean["R_ICESat_m"],
         "+ ATL13 → mean/zero-tide crust", "preferred branch — fully consistent with EGG2015"),
    ]
    for name, s, i, desc, note in combos:
        ls, hs = bootstrap_ci(s, seed=CFG.SEED)
        li, hi_ = bootstrap_ci(i, seed=CFG.SEED)
        ms, mi = float(np.nanmedian(s)), float(np.nanmedian(i))
        var.append({"variant": name, "description": desc, "assessment": note,
                    "median_R_SWOT_m": ms, "R_SWOT_ci95_low": ls, "R_SWOT_ci95_high": hs,
                    "nmad_SWOT_m": nmad(s), "n_SWOT": int(np.isfinite(s).sum()),
                    "median_R_ICESat_m": mi, "R_ICESat_ci95_low": li, "R_ICESat_ci95_high": hi_,
                    "nmad_ICESat_m": nmad(i), "n_ICESat": int(np.isfinite(i).sum()),
                    "delta_sensor_m": ms - mi,
                    "ci_overlap": bool(max(ls, li) <= min(hs, hi_))})
    pd.DataFrame(var).to_csv(FD / "Fig08_harmonisation_variants.csv", index=False)

    # Fig 13 — residual vs temporal separation (SWOT x ICESat pairs)
    pairs = []
    for _, a in sw.iterrows():
        for _, b in clean.iterrows():
            dt_h = (a["swot_utc"] - b["icesat_utc"]).total_seconds() / 3600.0
            if abs(dt_h) > 24 * 14:
                continue
            pairs.append({"swot_utc": a["swot_utc"], "icesat_utc": b["icesat_utc"],
                          "dt_hours": dt_h, "abs_dt_hours": abs(dt_h),
                          "same_day": a["date"] == b["date"],
                          "H_SWOT_m": a["H_SWOT_EGG2015_m"], "H_ICESat_m": b["H_meantide_m"],
                          "diff_swot_minus_icesat_m": a["H_SWOT_EGG2015_m"] - b["H_meantide_m"],
                          "icesat_rgt": b["rgt"]})
    pd.DataFrame(pairs).sort_values("abs_dt_hours").to_csv(FD / "Fig13_time_separation.csv", index=False)

    # Fig 14 — residual vs distance from gauge
    rec = []
    for r in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        s = _swot_per_date(radius_km=r)
        s = s.merge(g, on="date", how="left")
        s["residual_m"] = s["H_SWOT_EGG2015_m"] - (s["stage_bs77_m"] + d9902)
        for _, row in s.iterrows():
            rec.append({"sensor": "SWOT", "radius_km": r, "date": row["date"],
                        "n_px": row["n_px"], "median_dist_km": row["median_dist_km"],
                        "H_m": row["H_SWOT_EGG2015_m"], "nmad_m": row["nmad_m"],
                        "residual_vs_gauge_m": row["residual_m"]})
    for _, row in ic.iterrows():
        rec.append({"sensor": "ICESat-2", "radius_km": np.nan, "date": row["date"],
                    "n_px": row["n_points"], "median_dist_km": row["dist_to_gauge_km"],
                    "H_m": row["H_meantide_m"], "nmad_m": np.nan,
                    "residual_vs_gauge_m": row["H_meantide_m"] - row["H_gauge_EVRF2019_m"]})
    dist_df = pd.DataFrame(rec)
    dist_df.to_csv(FD / "Fig14_distance_residuals.csv", index=False)

    # Fig 10 — radius sensitivity summary
    rad = (dist_df[dist_df.sensor == "SWOT"].groupby("radius_km")
           .agg(median_residual_m=("residual_vs_gauge_m", "median"),
                nmad_residual_m=("residual_vs_gauge_m", nmad),
                median_H_m=("H_m", "median"), median_n_px=("n_px", "median"),
                n_dates=("date", "nunique")).reset_index())
    rad.to_csv(FD / "Fig10_radius_sensitivity.csv", index=False)
    print(f"  [kherson] SWOT n={len(sw)} dates, ICESat n={len(clean)} date×RGT, "
          f"Δ_sensor V2={var[2]['delta_sensor_m']:+.4f} m, CI overlap={var[2]['ci_overlap']}")
    return sw, ic


# --------------------------------------------------------------------------- #
# Fig 09 — water-mask sensitivity                                              #
# --------------------------------------------------------------------------- #
def fig09():
    d9902 = _delta9902(KH_LON, KH_LAT)
    g = _gauge_daily()
    strategies = [("4", (4,), "open_water only (ACCEPTED)"),
                  ("3+4", (3, 4), "+ water_near_land"),
                  ("4+7", (4, 7), "+ open_low_coh_water"),
                  ("3+4+6+7", (3, 4, 6, 7), "+ all low-coherence"),
                  ("3+4+5+6+7", (3, 4, 5, 6, 7), "+ dark water")]
    rows = []
    for key, cls, note in strategies:
        s = _swot_per_date(radius_km=1.0, classes=cls).merge(g, on="date", how="left")
        s["residual_m"] = s["H_SWOT_EGG2015_m"] - (s["stage_bs77_m"] + d9902)
        rows.append({"strategy": key, "classes": key, "description": note,
                     "median_residual_m": float(np.nanmedian(s["residual_m"])),
                     "nmad_residual_m": nmad(s["residual_m"]),
                     "median_n_px": float(np.median(s["n_px"])),
                     "total_n_px": int(s["n_px"].sum()),
                     "median_H_m": float(np.nanmedian(s["H_SWOT_EGG2015_m"])),
                     "median_within_date_nmad_m": float(np.median(s["nmad_m"])),
                     "n_dates": len(s),
                     "accepted": key == "4"})
    pd.DataFrame(rows).to_csv(FD / "Fig09_water_mask_sensitivity.csv", index=False)
    print(f"  [fig09] {len(rows)} mask strategies")


# --------------------------------------------------------------------------- #
# Fig 12 — pre-breach coverage timeline                                        #
# --------------------------------------------------------------------------- #
def fig12():
    hits = json.load(open(SCRATCH / "pixc_hits.json"))
    # SWOT_L2_HR_PIXC_<cycle>_<pass>_<tile>_<start>_... -> idx 4,5,6,7
    have = {pd.Timestamp(p.name.split("_")[7][:8]).normalize()
            for p in CFG.PIXC_DIR.glob("*.nc")}
    g = _gauge_daily()
    gauge_dates = set(g["date"])
    ic = _icesat_per_date()
    ic_dates = set(ic["date"])
    rows = []
    for h in hits:
        gid = h["producer_granule_id"]
        parts = gid.split("_")
        t = pd.Timestamp(h["time_start"]).tz_localize(None)
        d = t.normalize()
        rows.append({"date": d, "swot_utc": t, "cycle": parts[4], "pass": parts[5],
                     "tile": parts[6], "granule": gid,
                     "downloaded": d in have,
                     "kherson_gauge_available": d in gauge_dates,
                     "icesat_same_day": d in ic_dates})
    df = pd.DataFrame(rows).sort_values("swot_utc").drop_duplicates("date")
    df.to_csv(FD / "Fig12_coverage_timeline.csv", index=False)
    ic[["date", "icesat_utc", "rgt", "n_beams", "qc_flag"]].to_csv(
        FD / "Fig12_coverage_timeline_icesat.csv", index=False)
    print(f"  [fig12] {len(df)} SWOT dates, {df.downloaded.sum()} downloaded, "
          f"{df.icesat_same_day.sum()} with same-day ICESat")


# --------------------------------------------------------------------------- #
# Fig 06 / 11 — 2023-04-05 pixels                                              #
# --------------------------------------------------------------------------- #
def fig06_fig11():
    f = sorted(CFG.PIXC_DIR.glob("*482_001*.nc"))[0]
    box = (32.45, 46.52, 32.80, 46.75)
    acc = read_pixc(f, bbox=box, water_classes=(4,))
    allc = read_pixc(f, bbox=box, water_classes=None)
    z = sample_grid(CFG.EGG2015_TIF, acc["lon"], acc["lat"])
    df = pd.DataFrame({
        "lon": acc["lon"], "lat": acc["lat"], "classification": acc["classification"],
        "height_m": acc["height"], "h_ell_tidecorr_m": acc["h_ell"],
        "zeta_egg2015_m": z, "H_EGG2015_m": acc["h_ell"] - z,
        "dist_to_gauge_km": haversine_km(acc["lon"], acc["lat"], KH_LON, KH_LAT)})
    df.to_csv(FD / "Fig11_pixc_pixels_20230405_accepted.csv.gz", index=False, compression="gzip")
    ac = pd.DataFrame({"lon": allc["lon"], "lat": allc["lat"],
                       "classification": allc["classification"],
                       "water_frac": allc["water_frac"]})
    ac.sample(min(len(ac), 200000), random_state=CFG.SEED).to_csv(
        FD / "SFig11_pixc_pixels_20230405_allclasses.csv.gz", index=False, compression="gzip")

    ic = _icesat_per_date()
    day_ic = ic[ic["date"] == pd.Timestamp("2023-04-05")]
    sw = pd.read_csv(FD / "Fig05_kherson_timeseries_swot.csv", parse_dates=["date", "swot_utc"])
    day_sw = sw[sw["date"] == pd.Timestamp("2023-04-05")]
    g = _gauge_daily()
    day_g = g[g["date"] == pd.Timestamp("2023-04-05")]
    dt_h = (day_sw["swot_utc"].iloc[0] - day_ic["icesat_utc"].iloc[0]).total_seconds() / 3600
    three = pd.DataFrame([
        {"source": "gauge 80805 (Kherson)", "utc": "", "epoch_resolved": False,
         "epoch_note": "daily value — epoch unresolved",
         "value_m": float(day_g["stage_bs77_m"].iloc[0]) + _delta9902(KH_LON, KH_LAT),
         "quantity": "H_gauge_EVRF2019", "n": 1},
        {"source": "ICESat-2 ATL13 (RGT %d)" % day_ic["rgt"].iloc[0],
         "utc": str(day_ic["icesat_utc"].iloc[0]), "epoch_resolved": True, "epoch_note": "instantaneous",
         "value_m": float(day_ic["H_meantide_m"].iloc[0]),
         "quantity": "H_ICESat_EGG2015 (mean-tide harmonised)", "n": int(day_ic["n_points"].iloc[0])},
        {"source": "SWOT PIXC (482/001)", "utc": str(day_sw["swot_utc"].iloc[0]),
         "epoch_resolved": True, "epoch_note": "instantaneous",
         "value_m": float(day_sw["H_SWOT_EGG2015_m"].iloc[0]),
         "quantity": "H_SWOT_EGG2015", "n": int(day_sw["n_px"].iloc[0])},
    ])
    three["dt_swot_minus_icesat_h"] = dt_h
    three["comparison_type"] = "SAME-DAY, NOT SIMULTANEOUS"
    three.to_csv(FD / "Fig06_three_way_20230405.csv", index=False)
    print(f"  [fig06] Δt(SWOT−ICESat) = {dt_h:.2f} h; accepted px = {len(df)}")


if __name__ == "__main__":
    print("Building figure data ...")
    fig03()
    fig04_fig15()
    kherson_core()
    fig09()
    fig12()
    fig06_fig11()
    print(f"Done -> {FD}")
