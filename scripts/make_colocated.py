#!/usr/bin/env python
"""Independent SWOT validation of the gauge-derived ICESat-2 vertical correction.

Design
------
The two sensors are matched **to each other**, never through the gauge: for every
ATL13 segment, take the median of SWOT open-water pixels within 500 m on the same
day. This removes the ~30 km spatial mismatch that invalidated the earlier
gauge-mediated comparison (ICESat-2 has no ATL13 segments within 10 km of the
Kherson gauge in the 2023 pre-breach window).

Statistical unit
----------------
**One co-located SWOT-ICESat-2 overpass (= one date).** Segments and beams from a
single date/RGT are pseudo-replicates of one overpass: they show how stable the
offset is *within* a scene, not how reproducible it is *between* scenes. Segment- or
beam-level bootstrap must NOT be used to claim a global sensor bias, so the
per-date statistics are the headline and the within-date spread is reported
separately as a quality indicator.

Quantities
----------
    d_date = median(H_SWOT - H_ICESat)                       # raw sensor offset
    e_date = median(H_SWOT - (H_ICESat + c_gauge))           # residual AFTER the
                                                             # gauge-derived correction
with ``c_gauge = -0.1321 m``, the permanent-tide-harmonised regional median from the
six reservoir control gauges. SWOT played no part in deriving it, so this is a
genuinely independent test.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import netCDF4 as nc
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import bootstrap_ci, nmad
from swot_dnipro.vertical import haversine_km, read_pixc, sample_grid

MATCH_RADIUS_KM = 0.5
MIN_PIXELS = 30

#: Permanent-tide-harmonised regional median corrector from the six reservoir
#: control gauges (outputs/figure_data/Fig04_regional_median_summary.csv).
#: Derived from gauges + ICESat-2 only — SWOT is not involved.
C_GAUGE_HARMONISED_M = -0.1321


def main() -> None:
    seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kherson_atl13_segments.parquet")
    seg["dt"] = pd.to_datetime(seg["time"], utc=True, errors="coerce").dt.tz_localize(None)
    seg["date"] = seg["dt"].dt.normalize()

    per_seg = []
    for f in sorted(CFG.PIXC_DIR.glob("*.nc")):
        with nc.Dataset(str(f)) as ds:
            t = pd.Timestamp(ds.getncattr("time_granule_start")).tz_localize(None)
            cyc, pas, tile = (int(ds.getncattr("cycle_number")), int(ds.getncattr("pass_number")),
                              str(ds.getncattr("tile_name")))
        day = seg[seg["date"] == t.normalize()]
        if day.empty:
            continue
        box = (day.lon.min() - 0.05, day.lat.min() - 0.05,
               day.lon.max() + 0.05, day.lat.max() + 0.05)
        d = read_pixc(f, bbox=box, water_classes=(4,))
        if d["n_kept"] < 100:
            continue
        H_swot_px = d["h_ell"] - sample_grid(CFG.EGG2015_TIF, d["lon"], d["lat"])
        zi = sample_grid(CFG.EGG2015_TIF, day.lon.values, day.lat.values)
        # ATL13 is tide-free -> harmonise to the mean/zero-tide crust EGG2015 expects
        H_ice = day.h_wgs84_m.values + CFG.free2mean(day.lat.values) - zi

        for lat, lon, beam, rgt, dt_i, Hi in zip(day.lat.values, day.lon.values,
                                                 day.beam.values, day.rgt.values,
                                                 day.dt.values, H_ice):
            m = haversine_km(d["lon"], d["lat"], lon, lat) < MATCH_RADIUS_KM
            if m.sum() < MIN_PIXELS:
                continue
            per_seg.append({
                "date": t.normalize(), "swot_utc": t, "icesat_utc": pd.Timestamp(dt_i),
                "cycle": cyc, "pass": pas, "tile": tile, "rgt": rgt, "beam": beam,
                "lat": lat, "lon": lon,
                "dist_to_gauge_km": float(haversine_km([lon], [lat], CFG.KHERSON_GAUGE[2],
                                                       CFG.KHERSON_GAUGE[3])[0]),
                "H_ICESat_EGG2015_m": float(Hi),
                "H_SWOT_EGG2015_m": float(np.nanmedian(H_swot_px[m])),
                "n_swot_px": int(m.sum()),
            })
    ps = pd.DataFrame(per_seg)
    if ps.empty:
        print("no co-located pairs")
        return
    ps["d_m"] = ps.H_SWOT_EGG2015_m - ps.H_ICESat_EGG2015_m
    ps["e_m"] = ps.H_SWOT_EGG2015_m - (ps.H_ICESat_EGG2015_m + C_GAUGE_HARMONISED_M)
    ps["dt_hours"] = (ps.swot_utc - ps.icesat_utc).dt.total_seconds() / 3600.0
    ps.to_csv(CFG.FIGDATA / "Fig17_colocated_pairs_persegment.csv", index=False)

    # --- secondary: per beam (within-overpass stability only) -----------------
    pb = (ps.groupby(["date", "rgt", "beam"])
            .agg(n_segments=("lat", "size"), d_m=("d_m", "median"),
                 within_beam_nmad_m=("d_m", nmad)).reset_index())
    pb.to_csv(CFG.FIGDATA / "Fig17_colocated_pairs_perbeam.csv", index=False)

    # --- PRIMARY: per date = one independent co-located overpass --------------
    rows = []
    for (date,), grp in ps.groupby(["date"]):
        beams = grp.groupby("beam")["d_m"].median()
        rows.append({
            "date": date, "swot_utc": grp.swot_utc.iloc[0], "icesat_utc": grp.icesat_utc.min(),
            "swot_cycle": grp.cycle.iloc[0], "swot_pass": grp["pass"].iloc[0],
            "swot_tile": grp.tile.iloc[0], "icesat_rgt": int(grp.rgt.iloc[0]),
            "dt_hours": float(grp.dt_hours.median()),
            "abs_dt_hours": float(abs(grp.dt_hours.median())),
            "n_segments": len(grp), "n_beams": grp.beam.nunique(),
            "median_n_swot_px": float(grp.n_swot_px.median()),
            "total_swot_px": int(grp.n_swot_px.sum()),
            "median_dist_to_gauge_km": float(grp.dist_to_gauge_km.median()),
            "H_ICESat_m": float(grp.H_ICESat_EGG2015_m.median()),
            "H_SWOT_m": float(grp.H_SWOT_EGG2015_m.median()),
            "d_date_m": float(grp.d_m.median()),
            "within_date_nmad_m": nmad(grp.d_m),
            "between_beam_nmad_m": nmad(beams) if len(beams) > 2 else np.nan,
            "p05_m": float(np.nanpercentile(grp.d_m, 5)),
            "p95_m": float(np.nanpercentile(grp.d_m, 95)),
            "e_date_m": float(grp.e_m.median()),
        })
    pd_ = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # carry the ICESat-2 beam-spread QC flag through
    icq = pd.read_csv(CFG.FIGDATA / "Fig12_coverage_timeline_icesat.csv", parse_dates=["date"])
    pd_ = pd_.merge(icq[["date", "rgt", "qc_flag"]].rename(columns={"rgt": "icesat_rgt"}),
                    on=["date", "icesat_rgt"], how="left")
    pd_["qc_flag"] = pd_["qc_flag"].fillna("ok")
    pd_.to_csv(CFG.FIGDATA / "Fig17_colocated_perdate.csv", index=False)

    # --- summary over the independent unit (dates) ---------------------------
    n = len(pd_)
    def blk(col):
        x = pd_[col]
        out = {"median_m": float(np.median(x)), "nmad_m": nmad(x),
               "mean_abs_m": float(np.mean(np.abs(x))),
               "rmse_m": float(np.sqrt(np.mean(x ** 2))),
               "min_m": float(x.min()), "max_m": float(x.max()), "n_dates": n}
        if n >= 4:
            lo, hi = bootstrap_ci(x, seed=CFG.SEED)
            out["ci95_low_m"], out["ci95_high_m"] = lo, hi
        else:
            out["ci95_low_m"] = out["ci95_high_m"] = np.nan
        return out
    summ = pd.DataFrame([
        {"quantity": "d = H_SWOT - H_ICESat (before correction)", **blk("d_date_m")},
        {"quantity": f"e = H_SWOT - (H_ICESat + c), c = {C_GAUGE_HARMONISED_M:+.4f} m",
         **blk("e_date_m")},
    ])
    summ["independent_unit"] = "one co-located SWOT-ICESat overpass (date)"
    summ["c_gauge_harmonised_m"] = C_GAUGE_HARMONISED_M
    summ["gauge_used_for_matching"] = False
    summ.to_csv(CFG.FIGDATA / "Fig17_independent_validation_summary.csv", index=False)

    # --- report ---------------------------------------------------------------
    print(f"co-located overpasses (INDEPENDENT UNIT): n = {n}")
    print(pd_[["date", "swot_cycle", "icesat_rgt", "dt_hours", "n_beams", "n_segments",
               "H_ICESat_m", "H_SWOT_m", "d_date_m", "within_date_nmad_m",
               "e_date_m"]].to_string(index=False))
    print()
    for _, r in summ.iterrows():
        ci = (f"  CI95 [{r.ci95_low_m:+.4f}, {r.ci95_high_m:+.4f}]"
              if np.isfinite(r.ci95_low_m) else "  (CI not reported: n < 4 dates)")
        print(f"{r.quantity}\n   median {r.median_m:+.4f} m   NMAD {r.nmad_m:.4f}   "
              f"MAE {r.mean_abs_m:.4f}   RMSE {r.rmse_m:.4f}   n={int(r.n_dates)}{ci}")
    d0, e0 = summ.iloc[0], summ.iloc[1]
    print(f"\ngauge-derived correction c = {C_GAUGE_HARMONISED_M:+.4f} m "
          f"(from gauges + ICESat-2 only; SWOT not involved)")
    print(f"  |median| : {abs(d0.median_m):.4f} -> {abs(e0.median_m):.4f} m  "
          f"({'REDUCED' if abs(e0.median_m) < abs(d0.median_m) else 'NOT reduced'})")
    print(f"  MAE      : {d0.mean_abs_m:.4f} -> {e0.mean_abs_m:.4f} m")
    print(f"  RMSE     : {d0.rmse_m:.4f} -> {e0.rmse_m:.4f} m")
    print(f"  between-date NMAD is unchanged by a constant offset: "
          f"{d0.nmad_m:.4f} -> {e0.nmad_m:.4f} m")


if __name__ == "__main__":
    main()
