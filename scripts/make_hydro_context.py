#!/usr/bin/env python
"""Part A — Kherson hydrological context for every co-located satellite overpass.

Question
--------
Can the observed SWOT-ICESat-2 differences be explained by real water-level change
during the Δt between the two acquisitions, rather than by a vertical-datum offset?

Data reality
------------
The primary source is the yearbook table ``80805U_2023.xls`` ("Таблиця 1.2. Рівень
води, см", gauge zero −5.00 m BS-77) — a **day × month grid with one value per day
and no stated observation time**. No sub-daily, term (08/20) or hourly record for
2023 exists anywhere in the local archive. Every ``dH/dt`` below is therefore a
*daily* slope used as an approximation, and the expected change over a sub-daily Δt
is an extrapolation from it, not an interpolation. It is reported as a diagnostic,
never subtracted to produce a "corrected" datum bias.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

#: |slope| below this is indistinguishable from gauge resolution/noise.
#: The gauge reports whole centimetres; day-to-day differences of 1-2 cm are at
#: that level, so 3 cm/day is used as the RISING/FALLING threshold.
STABLE_THRESHOLD_CM_DAY = 3.0
GAUGE_RESOLUTION_CM = 1.0


def load_daily() -> pd.DataFrame:
    g = pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET)
    g = g[g["stat_type"] == "daily"].copy()
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return g[["date", "water_level_cm", "water_level_m_abs"]]


def slope_cm_day(g: pd.DataFrame, day: pd.Timestamp, half_window: int) -> float:
    """Robust-ish least-squares slope over ±half_window days around *day*."""
    w = g[(g.date >= day - pd.Timedelta(days=half_window)) &
          (g.date <= day + pd.Timedelta(days=half_window))]
    if len(w) < 3:
        return np.nan
    x = (w.date - day).dt.total_seconds().to_numpy() / 86400.0
    y = w.water_level_cm.to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def classify(s: float) -> str:
    if not np.isfinite(s):
        return "UNCERTAIN"
    if abs(s) < STABLE_THRESHOLD_CM_DAY:
        return "STABLE"
    return "RISING" if s > 0 else "FALLING"


def main() -> None:
    g = load_daily()
    per = pd.read_csv(CFG.FIGDATA / "Fig17_colocated_perdate.csv",
                      parse_dates=["date", "swot_utc", "icesat_utc"])

    # ---------------- A1 data inventory --------------------------------------
    inv = [{
        "source": "data/1_data/data/dm_H/herson_h/80805U_2023.xls (Таблиця 1.2)",
        "start_date": "2023-01-01", "end_date": "2023-12-31",
        "temporal_resolution": "daily (1 value/day)",
        "timezone": "not stated in the table",
        "variable": "Рівень води (stage above graph zero −5.00 m BS-77)",
        "units": "cm", "n_records": 365,
        "quality_notes": "PRIMARY yearbook source; day x month grid; NO observation "
                         "epoch stated; no term (08/20) or hourly columns present"},
       {"source": "data/1_data/data/parquet/dm_H/80805_yearbook.parquet",
        "start_date": "2019-01-01", "end_date": "2023-12-31",
        "temporal_resolution": "daily + monthly mean/min/max",
        "timezone": "inherited, not stated",
        "variable": "water_level_cm, water_level_m_abs", "units": "cm / m",
        "n_records": int(len(pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET))),
        "quality_notes": "derived from the .xls above; water_level_m_abs = -5.00 + cm/100"},
       {"source": "data/1_data/data/parquet/post_id=80805/year=2025.parquet",
        "start_date": "2025", "end_date": "2025",
        "temporal_resolution": "term readings (h_08/h_20)", "timezone": "Europe/Kyiv",
        "variable": "h_08, h_20", "units": "cm", "n_records": -1,
        "quality_notes": "2025 ONLY — does not cover the 2023 pre-breach SWOT window"},
       {"source": "sub-daily / hourly / storm telegrams for 2023",
        "start_date": "", "end_date": "", "temporal_resolution": "NONE FOUND",
        "timezone": "", "variable": "", "units": "", "n_records": 0,
        "quality_notes": "ABSENT from both repositories; searched dm_H, csv, h_level, "
                         "parquet (all versions), ingestion jobs, Excel originals"}]
    pd.DataFrame(inv).to_csv(CFG.TABLES / "kherson_80805_data_inventory.csv", index=False)

    # ---------------- A2 hydrograph context ----------------------------------
    rows = []
    for r in per.itertuples():
        day = r.date
        H = {k: g.loc[g.date == day + pd.Timedelta(days=k), "water_level_cm"]
             for k in (-1, 0, 1)}
        H = {k: (float(v.iloc[0]) if len(v) else np.nan) for k, v in H.items()}
        central = (H[1] - H[-1]) / 2.0 if np.isfinite(H[1]) and np.isfinite(H[-1]) else np.nan
        s3, s5, s7 = (slope_cm_day(g, day, w) for w in (1, 2, 3))
        # SWOT is the later acquisition when dt_hours > 0
        dt_days = r.dt_hours / 24.0
        exp_cm = central * dt_days if np.isfinite(central) else np.nan
        obs_cm = r.d_date_m * 100.0
        rows.append({
            "date": day.date(), "swot_utc": r.swot_utc, "icesat_utc": r.icesat_utc,
            "delta_t_h": r.dt_hours, "swot_is_later": bool(r.dt_hours > 0),
            "H_prev_cm": H[-1], "H_day_cm": H[0], "H_next_cm": H[1],
            "dH_previous_cm": H[0] - H[-1], "dH_next_cm": H[1] - H[0],
            "dH_dt_central_cm_day": central,
            "slope_3d_cm_day": s3, "slope_5d_cm_day": s5, "slope_7d_cm_day": s7,
            "state": classify(central),
            "expected_change_cm": exp_cm,
            "observed_swot_minus_icesat_cm": obs_cm,
            "remaining_after_hydro_cm": obs_cm - exp_cm if np.isfinite(exp_cm) else np.nan,
            "hydro_explains_sign": (bool(np.sign(exp_cm) == np.sign(obs_cm))
                                    if np.isfinite(exp_cm) and obs_cm != 0 else False),
            "within_overpass_nmad_cm": r.within_date_nmad_m * 100.0,
            "qc_flag": r.qc_flag,
            "epoch_caveat": "daily gauge value, no stated epoch; sub-daily extrapolated "
                            "from a daily slope — diagnostic only",
        })
    ctx = pd.DataFrame(rows)
    ctx.to_csv(CFG.TABLES / "kherson_hydrograph_context.csv", index=False)
    ctx.to_csv(CFG.FIGDATA / "FigA_kherson_hydrograph_context.csv", index=False)

    adj = ctx[["date", "delta_t_h", "state", "dH_dt_central_cm_day", "slope_5d_cm_day",
               "expected_change_cm", "observed_swot_minus_icesat_cm",
               "remaining_after_hydro_cm", "hydro_explains_sign", "qc_flag"]]
    adj.to_csv(CFG.TABLES / "swot_icesat_hydrological_adjustment.csv", index=False)

    # full 2023 hydrograph for the figure
    w = g[(g.date >= "2023-03-01") & (g.date <= "2023-06-20")].copy()
    w["H_evrf2019_m"] = w["water_level_m_abs"] + float(
        pd.read_csv(CFG.FIGDATA / "Fig05_kherson_timeseries_swot.csv").delta_epsg9902_m.iloc[0])
    w.to_csv(CFG.FIGDATA / "FigA_kherson_hydrograph_2023.csv", index=False)

    # ---------------- report --------------------------------------------------
    print("Kherson 80805 — sub-daily data for 2023: NONE FOUND (daily yearbook only)")
    print(f"STABLE threshold: |slope| < {STABLE_THRESHOLD_CM_DAY:g} cm/day "
          f"(gauge resolution {GAUGE_RESOLUTION_CM:g} cm)\n")
    cols = ["date", "delta_t_h", "H_prev_cm", "H_day_cm", "H_next_cm",
            "dH_dt_central_cm_day", "slope_5d_cm_day", "state"]
    print(ctx[cols].to_string(index=False))
    print()
    print("HYDROLOGICAL EXPECTATION vs OBSERVATION  (cm)")
    print(ctx[["date", "delta_t_h", "state", "expected_change_cm",
               "observed_swot_minus_icesat_cm", "remaining_after_hydro_cm",
               "hydro_explains_sign"]].to_string(index=False))
    print()
    for r in ctx.itertuples():
        verdict = ("hydrology has the SAME sign as the observation"
                   if r.hydro_explains_sign else
                   "hydrology has the OPPOSITE sign — cannot explain the observation")
        print(f"  {r.date}: Δt={r.delta_t_h:+.2f} h, {r.state}, "
              f"expected {r.expected_change_cm:+.1f} cm vs observed "
              f"{r.observed_swot_minus_icesat_cm:+.1f} cm -> {verdict}")
    print("\nContext: day-to-day |dH| at Kherson 2023-03-01..breach — median 3.0, "
          "p90 9.0, max 20.0 cm/day (n=96).")
    print("An 18 cm change is a ~99th-percentile daily excursion here, so the daily record "
          "does NOT\nmake a hydrodynamic explanation of 2023-04-05 look routine. See "
          "outputs/tables/kherson_daily_variability.csv")


if __name__ == "__main__":
    main()
