#!/usr/bin/env python
"""MS8 -- can wind setup over the Dnipro-Buh liman explain the 5 April 2023 Kherson anomaly?

The manuscript (v6, Sections 5.3, 7, S3.5, S4) says that no wind or pressure record exists
for 5 April 2023, when SWOT read 18.1 cm LOWER than ICESat-2 at Kherson (SWOT 21:26 UTC,
ICESat-2 07:13 UTC, dt = 14.2 h) while the gauge was rising (+5.9 cm expected). That
statement is wrong about this repository: p0e (ZONE_3 wind catalogue) holds ERA5 hourly
10 m wind and mean-sea-level pressure over the liman, 2019-2023, 20 cells of 0.25 deg
(46.25-47.25 N, 31.5-32.25 E), fetched from the Open-Meteo archive API
(outputs/tables/p0e_zone3_era5_hourly.csv). The planned CDS download of
scripts/fetch_era5.py never ran, so scripts/analyse_wind_setup.py never produced a table.

This script runs the same diagnostic on the p0e table:
  * wind resolved into the along-liman component (axis 65 deg clockwise from north, pointing
    ENE toward Kherson; +ve piles water up at Kherson), liman mean and the cell nearest
    Kherson (46.5 N 32.25 E, about 27 km west of the gauge -- Kherson itself is outside the
    p0e grid);
  * the hourly series interpolated to each satellite epoch and the change across dt;
  * a first-order setup scale  d_eta = rho_a C_D U|U| F / (rho_w g h)  with F = 60 km,
    h = 5 m, C_D = 1.5e-3, reported as an ORDER OF MAGNITUDE only and never subtracted from
    any observation;
  * the inverse-barometer change (about 1 cm per hPa).

ERA5 is a reanalysis at ~28 km, not a station observation; the gauge sits in the river,
where the local fetch is far shorter than the liman's. The result is a plausibility
diagnostic.

Outputs
    outputs/paper/validation/ms8_wind_setup_20230405.csv
    outputs/paper/validation/ms8_wind_setup_20230405.md
    outputs/paper/validation/ms8_era5_extract.csv   hourly rows of the p0e table within +-2 days of
                                                    every colocated date (all cells), with the sha256
                                                    of the full table in its header comment; when the
                                                    full table is absent (release snapshot) the script
                                                    reads this extract instead and reproduces the numbers
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ERA5 = ROOT / "outputs/tables/p0e_zone3_era5_hourly.csv"
PERDATE = ROOT / "outputs/figure_data/Fig17_colocated_perdate.csv"
OUT = ROOT / "outputs/paper/validation"
EXTRACT = OUT / "ms8_era5_extract.csv"
WINDOW_DAYS = 2
LIMAN_AXIS_DEG = 65.0            # as in scripts/analyse_wind_setup.py
KHERSON_CELL = (46.5, 32.25)     # nearest p0e cell to the Kherson gauge (46.624 N, 32.612 E)
FETCH_M, DEPTH_M, CD, RHO_A, RHO_W, G = 60e3, 5.0, 1.5e-3, 1.2, 1000.0, 9.81


def setup_cm(u_along: float) -> float:
    return 100 * RHO_A * CD * u_along * abs(u_along) * FETCH_M / (RHO_W * G * DEPTH_M)


def load_era5(per: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """The full p0e table when present (and refresh the extract), else the tracked extract."""
    import hashlib
    if ERA5.exists():
        d = pd.read_csv(ERA5, parse_dates=["time"])
        sha = hashlib.sha256(ERA5.read_bytes()).hexdigest()[:16]
        keep = np.zeros(len(d), bool)
        for t in pd.to_datetime(per.date):
            keep |= (d.time >= t - pd.Timedelta(days=WINDOW_DAYS)) & (d.time < t + pd.Timedelta(days=WINDOW_DAYS + 1))
        OUT.mkdir(parents=True, exist_ok=True)
        with EXTRACT.open("w") as f:
            f.write(f"# extract of {ERA5.relative_to(ROOT)} (sha256 {sha}), +-{WINDOW_DAYS} d around the colocated dates\n")
            d[keep].to_csv(f, index=False)
        return d, f"{ERA5.relative_to(ROOT)} (sha256 {sha})"
    if EXTRACT.exists():
        d = pd.read_csv(EXTRACT, comment="#", parse_dates=["time"])
        return d, f"{EXTRACT.relative_to(ROOT)} (tracked extract; {EXTRACT.read_text().splitlines()[0][2:]})"
    sys.exit(f"neither {ERA5} nor {EXTRACT} exists")


def main() -> None:
    per = pd.read_csv(PERDATE, parse_dates=["date", "swot_utc", "icesat_utc"])
    d, source = load_era5(per)
    th = np.radians(270.0 - d.wind_dir)              # meteorological "from" -> vector "to"
    d["u"], d["v"] = d.wind_speed * np.cos(th), d.wind_speed * np.sin(th)
    ax = np.radians(LIMAN_AXIS_DEG)
    d["along"] = d.u * np.sin(ax) + d.v * np.cos(ax)
    cols = ["along", "wind_speed", "mslp"]
    series = {"liman_mean": d.groupby("time")[cols].mean(),
              "cell_nearest_kherson": d[(d.lat == KHERSON_CELL[0]) & (d.lon == KHERSON_CELL[1])]
              .set_index("time")[cols]}
    rows = []
    for r in per.itertuples():
        for tag, s in series.items():
            s = s.sort_index()
            s = s.reindex(s.index.union([r.icesat_utc, r.swot_utc])).interpolate("time")
            a, b = s.loc[r.icesat_utc], s.loc[r.swot_utc]
            lo, hi = min(r.icesat_utc, r.swot_utc), max(r.icesat_utc, r.swot_utc)
            rows.append(dict(
                date=r.date.date(), field=tag, dt_hours=r.dt_hours,
                observed_swot_minus_icesat_cm=100 * r.d_date_m,
                along_axis_at_icesat_ms=a.along, along_axis_at_swot_ms=b.along,
                delta_along_axis_ms=b.along - a.along,
                wspd_at_icesat_ms=a.wind_speed, wspd_at_swot_ms=b.wind_speed,
                max_wspd_between_ms=s.loc[lo:hi, "wind_speed"].max(),
                setup_scale_at_icesat_cm=setup_cm(a.along), setup_scale_at_swot_cm=setup_cm(b.along),
                setup_scale_change_cm=setup_cm(b.along) - setup_cm(a.along),
                mslp_at_icesat_hpa=a.mslp, mslp_at_swot_hpa=b.mslp,
                inverse_barometer_cm=-(b.mslp - a.mslp),
                sign_matches_observation=bool(np.sign(setup_cm(b.along) - setup_cm(a.along))
                                              == np.sign(r.d_date_m)),
                qc_flag=r.qc_flag))
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "ms8_wind_setup_20230405.csv", index=False)

    lines = ["# MS8 -- wind setup over the liman at the 5 April 2023 Kherson anomaly", "",
             f"Source: {source} (ERA5 hourly via Open-Meteo, p0e; reanalysis, ~28 km, "
             f"{d.groupby(['lat', 'lon']).ngroups} cells {d.lat.min()}-{d.lat.max()} N, {d.lon.min()}-{d.lon.max()} E). "
             f"Epochs from `{PERDATE.relative_to(ROOT)}`. Along-axis: {LIMAN_AXIS_DEG:g} deg, +ve toward Kherson. "
             f"Setup scale: F = {FETCH_M / 1e3:.0f} km, h = {DEPTH_M:g} m, C_D = {CD:g} (order of magnitude only).", ""]
    for r in t.itertuples():
        lines += [f"## {r.date} -- {r.field}", "",
                  f"- observed SWOT - ICESat-2: {r.observed_swot_minus_icesat_cm:+.1f} cm over dt = {r.dt_hours:+.1f} h",
                  f"- along-liman wind: {r.along_axis_at_icesat_ms:+.1f} m/s at the ICESat-2 epoch -> "
                  f"{r.along_axis_at_swot_ms:+.1f} m/s at the SWOT epoch (change {r.delta_along_axis_ms:+.1f} m/s; "
                  f"max speed between {r.max_wspd_between_ms:.1f} m/s)",
                  f"- setup scale: {r.setup_scale_at_icesat_cm:+.0f} cm -> {r.setup_scale_at_swot_cm:+.0f} cm "
                  f"(change {r.setup_scale_change_cm:+.0f} cm); inverse barometer {r.inverse_barometer_cm:+.1f} cm",
                  f"- sign of the wind change {'MATCHES' if r.sign_matches_observation else 'does NOT match'} the observation", ""]
    k = t[(t.field == "liman_mean")].iloc[0]
    lines += ["## Reading", "",
              f"Between the two epochs the along-liman wind turned from {k.along_axis_at_icesat_ms:+.1f} m/s (blowing water "
              f"away from Kherson, setdown) to {k.along_axis_at_swot_ms:+.1f} m/s (toward Kherson, setup). Wind setup "
              f"therefore predicts a RISE of order {k.setup_scale_change_cm:+.0f} cm at Kherson across dt, the same sign as the "
              f"gauge trend (+5.9 cm) and opposite to the observed {k.observed_swot_minus_icesat_cm:+.1f} cm. Pressure changed by "
              f"{-k.inverse_barometer_cm:+.1f} hPa. On the ERA5 reanalysis, wind setup does not explain the anomaly; it makes "
              "the discrepancy larger. The anomaly remains unexplained. The manuscript sentence 'no wind or pressure record "
              "exists' should read: no station record; the ERA5 reanalysis over the liman (p0e) shows a wind change of the "
              "wrong sign.", ""]
    (OUT / "ms8_wind_setup_20230405.md").write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(t[["date", "field", "observed_swot_minus_icesat_cm", "along_axis_at_icesat_ms", "along_axis_at_swot_ms",
             "setup_scale_change_cm", "inverse_barometer_cm", "sign_matches_observation"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
