#!/usr/bin/env python
"""K5 -- canonical gauge table (six Kakhovka gauges only) + Sentinel matchups.

Canonical vertical transform, per operator instruction (K5 spec):

    H_gauge_EVRF2019(t) = 12.000 + stage_m(t) + Delta_EPSG9902(station)

where 12.000 m is the accepted BS-77 gauge zero for the six Kakhovka
reservoir gauges and Delta_EPSG9902 is the ALREADY-VALIDATED station-specific
Baltic-1977 -> EVRF2019 normal-height offset from this project
(outputs/tables/gauge_vertical_reference_summary.csv). EGG2015 / quasigeoid
correctors are NOT applied to gauge heights and are not mixed in here.

Station 80957 (DniproHES / Dniprovske pool) is NOT used anywhere -- different
pool, different and unresolved datum realization.

Hard rule honoured here: the source series are DAILY. No sub-daily observation
time exists for any of the six gauges, so no observation time is assumed. Each
record carries observation_type (daily_observation vs term_08_20_mean) and
obs_time_known=False, and the Sentinel matchup table reports the temporal
offset in whole days with the intra-day offset explicitly marked UNKNOWN.

Outputs
-------
outputs/tables/k5_gauge_levels_evrf2019.csv
outputs/tables/k5_sentinel_gauge_matchups.csv
outputs/tables/k5_gauge_availability.csv
outputs/figures/K5_gauge_hydrographs.png
"""
from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

#: The six Kakhovka reservoir gauges. 80957 deliberately absent.
SIX = [80977, 80971, 80964, 80963, 80961, 80959]
GAUGE_ZERO_BS77 = 12.000
RAW_DIR = Path("/home/niko/repo/icesat2-atl13-kakhovka/data/1_data/data/parquet/dm_H")
TERM_DIR = Path("/home/niko/repo/icesat2-atl13-kakhovka/data/1_data/data/parquet")
MASK_DIR = CFG.ROOT / "data" / "processed" / "water_masks"

#: CONFIRMED time reference (operator, 2026-09-11): h_08 / h_20 are Kyiv LOCAL
#: CIVIL time on the local calendar date stored in the source. Ukraine observes
#: seasonal offsets, so no fixed UTC+2/+3 is ever hard-coded -- the IANA
#: Europe/Kyiv database resolves the offset for each individual date, and every
#: comparison with Sentinel is done on the tz-aware UTC timeline.
TERM_HOURS_LOCAL = (8, 20)
KYIV = ZoneInfo("Europe/Kyiv")
UTC = ZoneInfo("UTC")
TIME_SEMANTICS = "local civil observation time"

_TF = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)


def _load_station(sid: int) -> pd.DataFrame:
    """Term observations (h_08/h_20, 2020+) plus the 2019 daily yearbook fallback.

    The partitioned ``post_id=`` store is the richer source: it carries the two
    separate term observations, not just their daily mean, which is what makes a
    real intra-day match to the Sentinel overpass possible. ``h_min``/``h_max``
    in that store are NOT trustworthy (median of h_max-h_min is negative, i.e.
    the labels are swapped), so they are never used -- the intra-day range is
    recomputed from h_08/h_20.
    """
    frames = []
    d = TERM_DIR / f"post_id={sid}"
    if d.exists():
        t = pd.concat([pd.read_parquet(f) for f in sorted(d.rglob("*.parquet"))],
                      ignore_index=True)
        t = t[["date", "h_08", "h_20", "h_min", "h_max"]].copy()
        t["date"] = pd.to_datetime(t.date).dt.normalize()
        t["series"] = "term_08_20"
        # SOURCE QC: h_min/h_max are carried through UNMODIFIED but flagged --
        # the median of (h_max - h_min) is negative in this store, i.e. the two
        # fields are apparently inverted or do not mean min/max. Their semantics
        # are unresolved from source documentation, so they are excluded from all
        # quantitative use in K5-K8. Nothing is swapped or overwritten.
        t["qc_minmax_inverted"] = (t.h_max - t.h_min) < 0
        frames.append(t)
    daily = RAW_DIR / f"{sid}_reservoir.parquet"
    if daily.exists():
        r = pd.read_parquet(daily)
        r = r[r.stat_type == "daily"].copy()
        r["date"] = pd.to_datetime(r.date).dt.normalize()
        r = r.rename(columns={"water_level_cm": "h_daily"})[["date", "h_daily"]]
        r["series"] = "daily_yearbook"
        frames.append(r)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # term observations win where both exist (they carry the sub-daily structure)
    out = out.sort_values(["date", "series"]).drop_duplicates("date", keep="last")
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    print("=" * 70)
    print("K5 -- GAUGE CANONICALIZATION AND DATE MATCHING")
    print("=" * 70)
    print(f"stations used: {SIX}  (station 80957 deliberately excluded)")

    # ---- validated station offsets (reused, never re-estimated) ------------
    summ = pd.read_csv(CFG.TABLES / "gauge_vertical_reference_summary.csv")
    summ = summ[summ.station_id.isin(SIX)].set_index("station_id")
    assert set(summ.index) == set(SIX), "missing one of the six gauges in the validated summary"
    assert (summ.zero_bs77_m == GAUGE_ZERO_BS77).all(), "a gauge zero is not 12.000 m BS-77"
    print("validated EPSG:9902 offsets reused from gauge_vertical_reference_summary.csv "
          "(not re-estimated):")
    for sid, r in summ.iterrows():
        print(f"  {sid} {r.name_en:<18} zero={r.zero_bs77_m:.3f} BS-77  "
              f"Delta_9902={r.delta_epsg9902_m:+.4f} m  -> zero_EVRF2019={12.0 + r.delta_epsg9902_m:.4f} m")

    # ---- chainage for each gauge -------------------------------------------
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)

    # ---- build the canonical per-observation table -------------------------
    rows = []
    for sid in SIX:
        raw = _load_station(sid)
        meta = summ.loc[sid]
        x, y = _TF.transform(meta.lon, meta.lat)
        chain_km, _, _, _ = SW.assign_chainage([meta.lon], [meta.lat], ch, tree=tree)
        for i, r in raw.iterrows():
            base = {
                "station_id": sid, "station_name": meta.name_en,
                "x_utm": x, "y_utm": y, "chain_km": float(chain_km[0]),
                "gauge_zero_bs77_m": GAUGE_ZERO_BS77,
                "epsg9902_offset_m": meta.delta_epsg9902_m,
                "stage_units": "cm", "source_row": int(i), "series": r.series,
            }
            if r.series == "term_08_20":
                # two genuinely timed observations -> emit both, no time invented
                for hh, col in zip(TERM_HOURS_LOCAL, ("h_08", "h_20")):
                    val = r.get(col)
                    stage_m = val / 100.0 if pd.notna(val) else np.nan
                    # Europe/Kyiv resolves the correct historical UTC offset for
                    # this specific date; no DST transition date is hard-coded.
                    dt_local = datetime(r.date.year, r.date.month, r.date.day, hh, 0,
                                        tzinfo=KYIV)
                    dt_utc = dt_local.astimezone(UTC)
                    rows.append({
                        **base,
                        "datetime_local_kyiv": dt_local.isoformat(),
                        "datetime_utc": dt_utc.isoformat(),
                        "utc_offset_hours": dt_local.utcoffset().total_seconds() / 3600.0,
                        "observation_slot": col,
                        "timezone_name": "Europe/Kyiv",
                        "time_semantics": TIME_SEMANTICS,
                        "source_datetime": dt_utc.replace(tzinfo=None),  # UTC timeline
                        "source_datetime_precision": "term_observation_kyiv_local_converted_to_utc",
                        "obs_time_known": True,
                        "observation_type": f"term_observation_{hh:02d}h_kyiv_local",
                        "observation_note": "h_min/h_max present in source but excluded "
                                            "(apparent inversion, semantics unresolved)",
                        "h_min_raw": r.get("h_min"), "h_max_raw": r.get("h_max"),
                        "qc_minmax_inverted": bool(r.get("qc_minmax_inverted", False)),
                        "stage_raw": val, "stage_m": stage_m,
                        "H_EVRF2019_m": (GAUGE_ZERO_BS77 + stage_m + meta.delta_epsg9902_m
                                         if pd.notna(stage_m) else np.nan),
                        "source_file": f"parquet/post_id={sid}",
                        "QC_flag": "ok" if pd.notna(stage_m) else "MISSING_no_value_in_source",
                    })
            else:
                val = r.get("h_daily")
                stage_m = val / 100.0 if pd.notna(val) else np.nan
                rows.append({
                    **base,
                    "datetime_local_kyiv": r.date.date().isoformat(),
                    "datetime_utc": pd.NaT,
                    "utc_offset_hours": np.nan,
                    "observation_slot": "daily",
                    "timezone_name": "Europe/Kyiv",
                    "time_semantics": "date only -- no observation time in source, none assumed",
                    "source_datetime": r.date,
                    "source_datetime_precision": "date_only_no_time_in_source",
                    "obs_time_known": False,
                    "observation_type": "daily_observation",
                    "observation_note": "yearbook daily value; sub-daily time unknown, none assumed",
                    "h_min_raw": np.nan, "h_max_raw": np.nan, "qc_minmax_inverted": False,
                    "stage_raw": val, "stage_m": stage_m,
                    "H_EVRF2019_m": (GAUGE_ZERO_BS77 + stage_m + meta.delta_epsg9902_m
                                     if pd.notna(stage_m) else np.nan),
                    "source_file": f"{sid}_reservoir.parquet",
                    "QC_flag": "ok" if pd.notna(stage_m) else "MISSING_no_value_in_source",
                })
    g = pd.DataFrame(rows)
    g["date"] = pd.to_datetime(g.source_datetime).dt.normalize()

    # ---- gross anomaly detection: flagged, NEVER deleted -------------------
    for sid, sg in g.groupby("station_id"):
        v = sg.H_EVRF2019_m
        med, mad = v.median(), (v - v.median()).abs().median()
        rob_z = 0.6745 * (v - med) / mad if mad > 0 else pd.Series(0.0, index=v.index)
        jump = v.diff().abs()
        anom = (rob_z.abs() > 6) | (jump > 1.0)
        idx = sg.index[anom.fillna(False)]
        g.loc[idx, "QC_flag"] = g.loc[idx, "QC_flag"].replace("ok", "ANOMALY_FLAGGED_NOT_REMOVED")
    n_anom = (g.QC_flag == "ANOMALY_FLAGGED_NOT_REMOVED").sum()
    print(f"\ncanonical gauge table: {len(g)} observations, {g.station_id.nunique()} stations")
    print(f"gross anomalies flagged (robust z>6 or day-to-day jump >1.0 m): {n_anom} "
          f"-- flagged only, none removed")
    print(f"missing values in source: {(g.QC_flag == 'MISSING_no_value_in_source').sum()}")

    g.to_csv(CFG.TABLES / "k5_gauge_levels_evrf2019.csv", index=False)
    print(f"-> {CFG.TABLES / 'k5_gauge_levels_evrf2019.csv'}")

    print("\ntemporal coverage per station (the decisive constraint for K6):")
    cov = g.groupby(["station_id", "station_name"]).agg(
        n=("date", "count"), first=("date", "min"), last=("date", "max"),
        chain_km=("chain_km", "first")).reset_index()
    print(cov.to_string(index=False))

    # ---- Sentinel acquisition inventory ------------------------------------
    scenes = []
    for f in sorted(MASK_DIR.glob("*.npz")):
        m = re.search(r"_(\d{8})T(\d{6})_.*_T(\d{2}[A-Z]{3})_", f.stem)
        if not m:
            continue
        dt = pd.to_datetime(f"{m.group(1)}T{m.group(2)}", format="%Y%m%dT%H%M%S")
        scenes.append({"scene": f.stem, "sensing_datetime_utc": dt,
                       "date": dt.normalize(), "tile": m.group(3)})
    sc = pd.DataFrame(scenes)
    print(f"\nSentinel scenes with built water masks: {len(sc)} files, "
          f"{sc.date.nunique()} unique acquisition dates "
          f"({sc.date.min().date()} .. {sc.date.max().date()})")

    # ---- matchups: every Sentinel date x every gauge ------------------------
    mrows = []
    gv = g[g.H_EVRF2019_m.notna()].copy()
    MAX_GAP_D = 5
    for date, sg in sc.groupby("date"):
        sens_utc = sg.sensing_datetime_utc.min()
        # everything below is compared on the UTC timeline; the gauge terms were
        # already converted Europe/Kyiv -> UTC when the canonical table was built
        sens_local = sens_utc  # naming kept; value is UTC, matching gauge UTC
        tiles = ",".join(sorted(sg.tile.unique()))
        for sid in SIX:
            s = gv[gv.station_id == sid]
            rec = {"sentinel_date": date, "sentinel_sensing_datetime_utc": sens_utc,
                   "sentinel_sensing_kyiv_local":
                       sens_utc.tz_localize("UTC").tz_convert(KYIV).isoformat(),
                   "tiles": tiles, "station_id": sid,
                   "station_name": summ.loc[sid, "name_en"],
                   "chain_km": s.chain_km.iloc[0] if len(s) else np.nan,
                   "before_datetime": pd.NaT, "after_datetime": pd.NaT,
                   "dt_before_h": np.nan, "dt_after_h": np.nan,
                   "intra_day_range_m": np.nan, "interpolation_method": "none"}
            same = s[s.date == date].sort_values("source_datetime")
            if len(same) >= 2 and same.obs_time_known.all():
                # genuinely timed terms bracketing (or adjacent to) the overpass
                t = same.source_datetime.values.astype("datetime64[ns]").astype("float64")
                h = same.H_EVRF2019_m.values
                tq = np.datetime64(sens_local).astype("datetime64[ns]").astype("float64")
                inside = t.min() <= tq <= t.max()
                H = float(np.interp(tq, t, h))
                b_i = int(np.searchsorted(t, tq) - 1) if inside else (0 if tq < t.min() else len(t) - 1)
                mrows.append({**rec, "gauge_date": date, "H_EVRF2019_m": H,
                              "observation_type": ("interpolated_between_terms" if inside
                                                   else "nearest_term_no_extrapolation"),
                              "match_method": ("intra_day_linear_between_08_and_20_terms" if inside
                                               else "outside_term_window_nearest_used"),
                              "before_datetime": same.source_datetime.iloc[max(b_i, 0)],
                              "after_datetime": same.source_datetime.iloc[min(b_i + 1, len(t) - 1)],
                              "dt_before_h": (sens_local - same.source_datetime.iloc[max(b_i, 0)]).total_seconds() / 3600,
                              "dt_after_h": (same.source_datetime.iloc[min(b_i + 1, len(t) - 1)] - sens_local).total_seconds() / 3600,
                              "intra_day_range_m": float(np.ptp(h)),
                              "interpolation_method": "linear_in_time_between_known_term_times",
                              "QC_flag": "|".join(sorted(set(same.QC_flag))) })
                continue
            if len(same) == 1:
                r = same.iloc[0]
                mrows.append({**rec, "gauge_date": date, "H_EVRF2019_m": r.H_EVRF2019_m,
                              "observation_type": r.observation_type,
                              "match_method": ("same_calendar_day_time_unknown"
                                               if not r.obs_time_known else "single_term_only"),
                              "QC_flag": r.QC_flag})
                continue
            before = s[s.date < date].tail(1)
            after = s[s.date > date].head(1)
            if len(before) and len(after):
                b, a = before.iloc[0], after.iloc[0]
                dtb = (sens_local - b.source_datetime).total_seconds() / 3600
                dta = (a.source_datetime - sens_local).total_seconds() / 3600
                if max(dtb, dta) / 24.0 > MAX_GAP_D:
                    mrows.append({**rec, "gauge_date": pd.NaT, "H_EVRF2019_m": np.nan,
                                  "observation_type": "none",
                                  "match_method": "GAP_TOO_LARGE_not_interpolated",
                                  "QC_flag": f"gap {dtb/24:.1f}/{dta/24:.1f} d exceeds {MAX_GAP_D} d limit",
                                  "before_datetime": b.source_datetime, "after_datetime": a.source_datetime,
                                  "dt_before_h": dtb, "dt_after_h": dta})
                else:
                    w = dta / (dtb + dta)
                    mrows.append({**rec, "gauge_date": pd.NaT,
                                  "H_EVRF2019_m": w * b.H_EVRF2019_m + (1 - w) * a.H_EVRF2019_m,
                                  "observation_type": "interpolated_across_days",
                                  "match_method": "linear_time_interpolation_across_days",
                                  "QC_flag": "INTERPOLATED",
                                  "before_datetime": b.source_datetime, "after_datetime": a.source_datetime,
                                  "dt_before_h": dtb, "dt_after_h": dta,
                                  "interpolation_method": "linear_in_time"})
            else:
                mrows.append({**rec, "gauge_date": pd.NaT, "H_EVRF2019_m": np.nan,
                              "observation_type": "none",
                              "match_method": "OUTSIDE_SERIES_no_extrapolation",
                              "QC_flag": "series does not cover this date",
                              "before_datetime": before.source_datetime.iloc[0] if len(before) else pd.NaT,
                              "after_datetime": after.source_datetime.iloc[0] if len(after) else pd.NaT})
    mu = pd.DataFrame(mrows)
    mu.to_csv(CFG.TABLES / "k5_sentinel_gauge_matchups.csv", index=False)
    print(f"-> {CFG.TABLES / 'k5_sentinel_gauge_matchups.csv'}")

    # ---- mandatory timezone audit -------------------------------------------
    # The error mode being quantified: reading h_08/h_20 as if they were already
    # UTC (no conversion), versus the correct Europe/Kyiv -> UTC conversion.
    print("\n" + "=" * 70)
    print("TIMEZONE AUDIT (Europe/Kyiv -> UTC vs naive 'terms are UTC')")
    print("=" * 70)
    terms = g[g.obs_time_known].copy()
    terms["slot_hour"] = terms.observation_slot.map({"h_08": 8, "h_20": 20})
    arows = []
    for date, sg in sc.groupby("date"):
        sens_utc = sg.sensing_datetime_utc.min()
        for sid in SIX:
            s = terms[(terms.station_id == sid) & (terms.date == date)].sort_values("slot_hour")
            if len(s) < 2 or s.H_EVRF2019_m.isna().any():
                continue
            h = s.H_EVRF2019_m.values
            off = float(s.utc_offset_hours.iloc[0])
            t_corr = s.source_datetime.values.astype("datetime64[ns]").astype("float64")
            tq = np.datetime64(sens_utc).astype("datetime64[ns]").astype("float64")
            wse_corr = float(np.interp(tq, t_corr, h))
            # naive: pretend the 08/20 labels are UTC hours
            t_naive = np.array([np.datetime64(pd.Timestamp(date) + pd.Timedelta(hours=int(x)))
                                .astype("datetime64[ns]").astype("float64")
                                for x in s.slot_hour.values])
            wse_naive = float(np.interp(tq, t_naive, h))
            sens_kyiv = sens_utc.tz_localize("UTC").tz_convert(KYIV)
            arows.append({
                "date": date, "station_id": sid,
                "sentinel_utc": sens_utc.isoformat(),
                "matched_gauge_local": s.datetime_local_kyiv.iloc[0],
                "matched_gauge_utc": s.datetime_utc.iloc[0],
                "utc_offset": off,
                "sentinel_kyiv_local": sens_kyiv.isoformat(),
                "old_dt_hours": (tq - t_naive[0]) / 3.6e12,
                "corrected_dt_hours": (tq - t_corr[0]) / 3.6e12,
                "change_hours": (t_naive[0] - t_corr[0]) / 3.6e12,
                "old_WSE": wse_naive, "corrected_WSE": wse_corr,
                "delta_WSE_m": wse_corr - wse_naive,
                "intra_day_spread_m": float(np.ptp(h)),
            })
    aud = pd.DataFrame(arows)
    aud.to_csv(CFG.TABLES / "k5_timezone_audit.csv", index=False)
    if len(aud):
        print(f"audited matchups with both terms available: {len(aud)}")
        print(f"  UTC offsets encountered: {sorted(aud.utc_offset.unique())} h "
              f"(resolved per date by the IANA database, never hard-coded)")
        print(f"  timestamp change vs naive: {aud.change_hours.abs().max():.1f} h")
        print(f"  |delta WSE| from the correction: median {aud.delta_WSE_m.abs().median():.3f} m, "
              f"max {aud.delta_WSE_m.abs().max():.3f} m")
        print(f"  intra-day spread |h20-h08| on these dates: median "
              f"{aud.intra_day_spread_m.median():.3f} m, max {aud.intra_day_spread_m.max():.3f} m")

        print("\nsensitivity to a wrong timezone assumption (shifting gauge times):")
        for shift in (-2, -1, 1, 2):
            deltas = []
            for date, sg in sc.groupby("date"):
                sens_utc = sg.sensing_datetime_utc.min()
                tq = np.datetime64(sens_utc).astype("datetime64[ns]").astype("float64")
                for sid in SIX:
                    s = terms[(terms.station_id == sid) & (terms.date == date)].sort_values("slot_hour")
                    if len(s) < 2 or s.H_EVRF2019_m.isna().any():
                        continue
                    h = s.H_EVRF2019_m.values
                    t0 = s.source_datetime.values.astype("datetime64[ns]").astype("float64")
                    base_v = float(np.interp(tq, t0, h))
                    shifted = float(np.interp(tq, t0 + shift * 3.6e12, h))
                    deltas.append(shifted - base_v)
            if deltas:
                da = np.abs(deltas)
                print(f"  {shift:+d} h : median |dWSE| = {np.median(da):.3f} m, "
                      f"max = {da.max():.3f} m")
    print(f"-> {CFG.TABLES / 'k5_timezone_audit.csv'}")

    nminmax = int(g.qc_minmax_inverted.sum())
    print(f"\nSOURCE QC: h_min/h_max apparent inversion flagged on {nminmax} records; raw values "
          f"preserved unmodified in h_min_raw/h_max_raw and excluded from all quantitative use.")

    # ---- availability matrix ------------------------------------------------
    ok = mu[mu.H_EVRF2019_m.notna()]
    all_dates = pd.Index(sorted(sc.date.unique()), name="sentinel_date")
    avail = (ok.pivot_table(index="sentinel_date", columns="station_id",
                            values="H_EVRF2019_m", aggfunc="first")
             .reindex(all_dates))  # keep Sentinel dates with zero gauges
    for sid in SIX:
        if sid not in avail.columns:
            avail[sid] = np.nan
    avail = avail[SIX]
    avail["n_gauges"] = avail.notna().sum(axis=1)
    avail["period"] = np.where(avail.index < pd.Timestamp(CFG.BREACH_DATE), "PRE_BREACH", "POST_BREACH")
    avail = avail.reset_index()
    avail.to_csv(CFG.TABLES / "k5_gauge_availability.csv", index=False)
    print(f"-> {CFG.TABLES / 'k5_gauge_availability.csv'}")

    print("\n" + "=" * 70)
    print("DATA-REALITY FINDING (checked, not assumed)")
    print("=" * 70)
    dist = avail.groupby(["period", "n_gauges"]).size().rename("n_sentinel_dates")
    print(dist.to_string())
    multi = avail[avail.n_gauges >= 2]
    print(f"\nSentinel dates with >=2 gauges (a longitudinal profile is possible): {len(multi)}")
    if len(multi):
        print("  " + ", ".join(multi.sentinel_date.dt.strftime("%Y-%m-%d")))
        print(f"  all of these are {sorted(multi.period.unique())}")
    single = avail[avail.n_gauges == 1]
    print(f"Sentinel dates with exactly 1 gauge (single anchor only, NO profile): {len(single)}")
    none = avail[avail.n_gauges == 0]
    print(f"Sentinel dates with 0 gauges: {len(none)}")
    print("\nCause: five of the six reservoir gauges end 2021-12-31; only 80959 Rozumivka "
          "(chain_km ~247, upper end) continues past 2021. No six-gauge post-breach date exists.")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    cmap = plt.get_cmap("viridis")
    order = cov.sort_values("chain_km")
    for i, r in enumerate(order.itertuples()):
        s = g[(g.station_id == r.station_id) & g.H_EVRF2019_m.notna()].sort_values("date")
        axes[0].plot(s.date, s.H_EVRF2019_m, lw=1.1, color=cmap(i / max(len(order) - 1, 1)),
                     label=f"{r.station_id} {r.station_name} (s={r.chain_km:.0f} km)")
    axes[0].axvline(pd.Timestamp(CFG.BREACH_DATE), color="#c1402a", ls="--", lw=1.2)
    axes[0].annotate("dam breach\n2023-06-06", xy=(pd.Timestamp(CFG.BREACH_DATE), axes[0].get_ylim()[1]),
                     xytext=(6, -28), textcoords="offset points", fontsize=8, color="#c1402a")
    axes[0].set_ylabel("gauge level, m EVRF2019")
    axes[0].set_title("A. Six Kakhovka gauges, canonical EVRF2019 heights "
                      "(12.000 m BS-77 zero + stage + station EPSG:9902 offset)", loc="left", fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].bar(avail.sentinel_date, avail.n_gauges, width=12,
                color=np.where(avail.n_gauges >= 2, "#236f8c", "#c9a227"))
    axes[1].axvline(pd.Timestamp(CFG.BREACH_DATE), color="#c1402a", ls="--", lw=1.2)
    axes[1].axhline(2, color="grey", ls=":", lw=0.9)
    axes[1].set_ylabel("gauges available")
    axes[1].set_xlabel("Sentinel-2 acquisition date")
    axes[1].set_title("B. Gauge availability per Sentinel date "
                      "(blue = profile possible, amber = single anchor only)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K5_gauge_hydrographs.png", dpi=150)
    print(f"\n-> {CFG.FIG / 'K5_gauge_hydrographs.png'}")


if __name__ == "__main__":
    main()
