#!/usr/bin/env python
"""Phase 9 analysis — can wind setup explain the 2023-04-05 anomaly?

Runs as soon as `data/raw/meteo/era5_lower_dnipro_2023_03-06.nc` exists.

The question
------------
On 2023-04-05 SWOT read 18.1 cm LOWER than ICESat-2 (Δt = 14.22 h, SWOT later).
The Kherson daily hydrograph cannot explain it: the gauge was RISING at +8.5 cm/day,
which predicts SWOT +5 cm HIGHER, and an 18 cm daily excursion is a ~99th-percentile
event there (median |ΔH| 3.0, p90 9.0, max 20.0 cm/day, n = 96).

Wind setup is the remaining candidate. The Dnipro-Buh liman is oriented roughly
WSW-ENE, so an along-liman wind pushes water toward or away from Kherson. We test
whether the along-axis wind component changed between the two acquisition epochs in
the direction and magnitude needed.

Method
------
* extract the ERA5 cell nearest Kherson and a liman-mean box
* rotate 10 m wind into along-axis (toward Kherson, +) and cross-axis components
* report the wind at each satellite epoch and the change across Δt
* compare against a first-order wind-setup scale, reported as an ORDER OF MAGNITUDE
  only, never subtracted from the observation

ERA5 is a reanalysis, not a station observation. Labelled as such everywhere.
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

NC = CFG.ROOT / "data" / "raw" / "meteo" / "era5_lower_dnipro_2023_03-06.nc"
#: Dnipro-Buh liman long axis, degrees clockwise from north, pointing TOWARD Kherson (ENE).
LIMAN_AXIS_DEG = 65.0
KH_LON, KH_LAT = CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3]
LIMAN_BOX = dict(lon=slice(31.2, 32.4), lat=slice(46.9, 46.3))  # ERA5 lat descends


def load() -> pd.DataFrame:
    import xarray as xr
    ds = xr.open_dataset(NC)
    ren = {"longitude": "lon", "latitude": "lat", "valid_time": "time"}
    ds = ds.rename({k: v for k, v in ren.items() if k in ds.dims or k in ds.coords})
    kh = ds.sel(lon=KH_LON, lat=KH_LAT, method="nearest")
    lim = ds.sel(**LIMAN_BOX).mean(dim=["lon", "lat"])
    out = pd.DataFrame({"time": pd.to_datetime(ds.time.values)})
    for tag, src in (("kherson", kh), ("liman", lim)):
        u = np.asarray(src["u10"].values, float)
        v = np.asarray(src["v10"].values, float)
        out[f"u10_{tag}"] = u
        out[f"v10_{tag}"] = v
        out[f"wspd_{tag}"] = np.hypot(u, v)
        out[f"wdir_from_{tag}"] = (270 - np.degrees(np.arctan2(v, u))) % 360
        th = np.radians(LIMAN_AXIS_DEG)
        # +ve = blowing along the liman axis toward Kherson (piles water up at Kherson)
        out[f"along_axis_{tag}"] = u * np.sin(th) + v * np.cos(th)
        out[f"cross_axis_{tag}"] = u * np.cos(th) - v * np.sin(th)
        for p, name in (("msl", "msl_hpa"), ("sp", "sp_hpa")):
            if p in src:
                out[f"{name}_{tag}"] = np.asarray(src[p].values, float) / 100.0
    return out.set_index("time").sort_index()


def at(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    """Linear interpolation of the hourly field to an exact acquisition epoch."""
    s = df.reindex(df.index.union([ts])).interpolate("time").loc[ts]
    return s


def main() -> None:
    if not NC.exists():
        print(f"ERA5 file not present: {NC}")
        print("Run scripts/fetch_era5.py after accepting the ERA5 licence at")
        print("  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"
              "?tab=download#manage-licences")
        sys.exit(1)

    df = load()
    df.to_parquet(CFG.ROOT / "data" / "processed" / "meteo_hourly_context.parquet")
    print(f"ERA5 hourly: {df.index.min()} .. {df.index.max()}  n={len(df)}")

    per = pd.read_csv(CFG.FIGDATA / "Fig17_colocated_perdate.csv",
                      parse_dates=["date", "swot_utc", "icesat_utc"])
    rows = []
    for r in per.itertuples():
        a, b = at(df, r.icesat_utc), at(df, r.swot_utc)
        d_along = b.along_axis_liman - a.along_axis_liman
        rows.append({
            "date": r.date.date(), "delta_t_h": r.dt_hours,
            "observed_swot_minus_icesat_cm": r.d_date_m * 100,
            "wspd_at_icesat_ms": a.wspd_kherson, "wspd_at_swot_ms": b.wspd_kherson,
            "wdir_at_icesat_deg": a.wdir_from_kherson, "wdir_at_swot_deg": b.wdir_from_kherson,
            "along_axis_at_icesat_ms": a.along_axis_liman,
            "along_axis_at_swot_ms": b.along_axis_liman,
            "delta_along_axis_ms": d_along,
            "max_wspd_between_ms": float(df.loc[min(a.name, b.name):max(a.name, b.name),
                                                "wspd_kherson"].max()),
            "msl_at_icesat_hpa": a.get("msl_hpa_kherson", np.nan),
            "msl_at_swot_hpa": b.get("msl_hpa_kherson", np.nan),
            "delta_msl_hpa": b.get("msl_hpa_kherson", np.nan) - a.get("msl_hpa_kherson", np.nan),
            # inverse barometer: ~1 cm of water per 1 hPa
            "inverse_barometer_cm": -(b.get("msl_hpa_kherson", np.nan)
                                      - a.get("msl_hpa_kherson", np.nan)),
            "qc_flag": r.qc_flag,
        })
    ctx = pd.DataFrame(rows)
    ctx.to_csv(CFG.TABLES / "wind_setup_context.csv", index=False)
    ctx.to_csv(CFG.FIGDATA / "FigH_wind_setup_context.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== wind & pressure at each satellite epoch (ERA5 reanalysis) ===")
    print(ctx[["date", "delta_t_h", "wspd_at_icesat_ms", "wspd_at_swot_ms",
               "along_axis_at_icesat_ms", "along_axis_at_swot_ms", "delta_along_axis_ms",
               "delta_msl_hpa", "inverse_barometer_cm",
               "observed_swot_minus_icesat_cm"]].to_string(index=False))

    print("\n=== interpretation ===")
    for r in ctx.itertuples():
        sign_ok = np.sign(r.delta_along_axis_ms) == np.sign(r.observed_swot_minus_icesat_cm)
        print(f"\n  {r.date}: observed {r.observed_swot_minus_icesat_cm:+.1f} cm over "
              f"Δt={r.delta_t_h:+.2f} h")
        print(f"    along-liman wind {r.along_axis_at_icesat_ms:+.1f} -> "
              f"{r.along_axis_at_swot_ms:+.1f} m/s  (Δ {r.delta_along_axis_ms:+.1f} m/s); "
              f"max gust between {r.max_wspd_between_ms:.1f} m/s")
        print(f"    Δ MSL {r.delta_msl_hpa:+.1f} hPa -> inverse-barometer "
              f"{r.inverse_barometer_cm:+.1f} cm")
        print(f"    wind change sign {'MATCHES' if sign_ok else 'does NOT match'} "
              "the observed difference")
    print("\nNOTE: ERA5 is a reanalysis. No wind-derived term is subtracted from any "
          "observation; this is a diagnostic of plausibility only.")


if __name__ == "__main__":
    main()
