#!/usr/bin/env python
"""HIST25B GATE 5 (normalisation) — S1 assets -> observation events -> state.

Nothing is classified here. This script only establishes the canonical record
that everything downstream depends on:

    asset / STAC item  ->  observation event  ->  hydraulic state + roles

WHY EVENT GROUPING MATTERS. 2,655 raw Sentinel-1 RTC items cover this
reservoir for 2019-2023, but they form only ~1,416 independent overpasses
(~1.9 spatial tiles per pass). Treating tiles as independent acquisitions
would pseudo-replicate every statistic computed later.

WHY GAUGE MATCHING NEEDS A METHOD FIELD. The gauge series is DAILY and
carries NO observation time -- every datetime in the source is 00:00:00, and
k5_gauge_canonicalization records obs_time_known=False with the intra-day
offset explicitly UNKNOWN. Hour-level bracketing of an overpass is therefore
impossible by construction. What CAN be stated honestly is: which daily
readings bracket the acquisition date, how far away they are in days, and how
fast the level was moving at the time. The last term dominates during
drawdown:

    pre-breach   median |dH/dt| = 0.014 m/day  -> intra-day term ~0.007 m
    06-06..06-20 median |dH/dt| = 0.570 m/day, max 1.410 m/day
                                             -> intra-day term 0.29-0.71 m

So a drawdown scene's GEOMETRY is sound while its ELEVATION LABEL is weak.
That is recorded per event, never used to discard the scene.

WHY THE REGIME BOUNDARY CANNOT COME FROM THE HYDROGRAPH. The record plateaus
at 13.79 m on 2023-06-10, then goes dark for 19 days -- every missing day in
the entire 2019-2025 series -- and resumes on 2023-07-01 at 12.20 m, 1.64 m
lower. A "first date where |dH/dt| < 0.05" rule reads that plateau as
stabilisation; it is a gap. From 2023-07-01 the value is pinned at exactly
12.204 m for many days, which looks like a station floor rather than
hydraulics. POSTBREACH_RIVER_DOMINATED therefore CANNOT be assigned from the
gauge and is left unassigned here, pending spatial evidence (water-extent
contraction, channel connectivity, longitudinal WSE) in a later gate.

KEEP EVERYTHING -- NEVER MIX HYDRAULIC STATES SILENTLY. No acquisition is
discarded for having the "wrong" stage. Each event observes the shoreline at
its own measured level, S_i = shoreline(x, y | H_i, t_i), so every reliable
shoreline constrains terrain as z(x_shore,i) ~ H_i with its own uncertainty.
H1/H2/H3 are selected SLICES of that series, not buckets.

Outputs
-------
outputs/tables/hist25b_s1_asset_inventory.csv
outputs/tables/hist25b_s1_observation_event_inventory.csv
outputs/tables/hist25b_regime_segmentation.csv
outputs/tables/hist25b_stage_matching.csv
outputs/figures/historical_bathymetry/png/hist25b_stage_time_inventory.png
"""
from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BBOX = [32.2, 46.5, 35.5, 48.0]
YEARS = (2019, 2020, 2021, 2022, 2023)
ASSETS_CSV = CFG.TABLES / "hist25b_s1_asset_inventory.csv"
EVENTS_CSV = CFG.TABLES / "hist25b_s1_observation_event_inventory.csv"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"

BREACH = pd.Timestamp("2023-06-06")
MARGIN_SLOPE = 0.03                 # hist24's diagnosed margin slope, m/m
TOL_TARGET, TOL_NEAR = 0.10, 0.30

# vertical uncertainty of a gauge-derived level, excluding the intra-day term
# (hist24: EPSG9902 grid 0.068, gauge reading 0.010, seiche 0.175, 1 sigma)
SIG_STATIC = float(np.sqrt(0.068 ** 2 + 0.010 ** 2 + 0.175 ** 2))
MAX_GAUGE_GAP_DAYS = 3              # beyond this, WSE is not resolvable
# drawdown detection: a fall far outside normal day-to-day variability
ANOMALY_RATE = 0.20                 # m/day; pre-breach p95 is 0.10
CUM_LOSS_MIN = 0.50                 # m, cumulative, to confirm ACTIVE_DRAWDOWN
FLAT_RUN_MIN = 4                    # identical values -> suspect station floor


def stac_range(d0, d1):
    out, token = [], None
    while True:
        q = {"collections": ["sentinel-1-rtc"], "bbox": BBOX, "limit": 500,
             "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z"}
        if token:
            q["token"] = token
        r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                   headers={"Content-Type": "application/json"})
        j = json.loads(urllib.request.urlopen(r, timeout=180).read())
        out += j["features"]
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        token = nxt[0].get("body", {}).get("token")
        if not token:
            break
    return out


def gauge_series():
    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    g = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = g.groupby("date").transformed_level_m.median().sort_index()
    lev.index = pd.to_datetime(lev.index)
    return lev.sort_index()


def match_gauge(ts, lev):
    """Attach a WSE to an acquisition DATE, with explicit method and support.

    The source is daily with no observation time, so no sub-daily bracketing
    is possible. Returns WSE_UNRESOLVED rather than inventing precision when
    temporal support is inadequate.
    """
    day = pd.Timestamp(ts.date())
    before = lev.index[lev.index <= day]
    after = lev.index[lev.index >= day]
    tb = (day - before[-1]).days if len(before) else np.nan
    ta = (after[0] - day).days if len(after) else np.nan
    out = dict(gauge_time_before_days=tb, gauge_time_after_days=ta,
               max_gap_hours=np.nan, wse_interpolated=False,
               gauge_match_method="WSE_UNRESOLVED", gauge_wse_m=np.nan,
               wse_uncertainty_m=np.nan, dHdt_m_per_day=np.nan)
    if not len(before) or not len(after):
        return out
    gap_days = (after[0] - before[-1]).days
    out["max_gap_hours"] = float(max(gap_days, 0) * 24)
    # local rate of change, from whichever neighbours exist
    lo, hi = before[-1], after[0]
    if hi == lo:
        nb = lev.index[lev.index < lo]
        rate = abs(lev[lo] - lev[nb[-1]]) / max((lo - nb[-1]).days, 1) if len(nb) else 0.0
    else:
        rate = abs(lev[hi] - lev[lo]) / max(gap_days, 1)
    out["dHdt_m_per_day"] = float(rate)
    # intra-day term: the level may move by up to half a day's change between
    # 00:00 (the nominal gauge stamp) and the unknown overpass time
    intraday = 0.5 * rate
    if day in lev.index:
        out.update(gauge_match_method="daily_observation_same_date",
                   gauge_wse_m=float(lev[day]), wse_interpolated=False)
    elif gap_days <= MAX_GAUGE_GAP_DAYS:
        f = (day - lo).days / gap_days
        out.update(gauge_match_method="interpolated_between_daily",
                   gauge_wse_m=float(lev[lo] + f * (lev[hi] - lev[lo])),
                   wse_interpolated=True)
        intraday = max(intraday, 0.5 * abs(lev[hi] - lev[lo]))
    else:
        return out          # gap too long: WSE_UNRESOLVED
    out["wse_uncertainty_m"] = float(np.sqrt(SIG_STATIC ** 2 + intraday ** 2))
    return out


def segment_regimes(lev):
    """Label each gauge DAY with a hydraulic regime.

    Deliberately conservative: ACTIVE_DRAWDOWN needs a sustained fall AND a
    material cumulative loss; anything the hydrograph cannot resolve becomes
    REGIME_UNRESOLVED; POSTBREACH_RIVER_DOMINATED is never assigned here
    because it is a morpho-hydraulic state requiring spatial evidence.
    """
    reg = pd.Series("REGIME_UNRESOLVED", index=lev.index, dtype=object)
    d = lev.diff()
    gaps = lev.index.to_series().diff().dt.days.fillna(1)
    reg[lev.index < BREACH] = "PREBREACH_IMPOUNDED"
    if BREACH in reg.index:
        reg[BREACH] = "BREACH_ONSET"
    post = lev.index[lev.index > BREACH]
    cum = 0.0
    active_done = False
    for t in post:
        if gaps[t] > MAX_GAUGE_GAP_DAYS:
            reg[t] = "REGIME_UNRESOLVED"       # no support across the blackout
            active_done = True                  # cannot claim continuity
            continue
        rate = d.get(t, np.nan)
        if not active_done and np.isfinite(rate) and rate < -ANOMALY_RATE:
            cum += -rate
            reg[t] = "ACTIVE_DRAWDOWN"
        elif not active_done and cum >= CUM_LOSS_MIN:
            # fall paused; only a SUSTAINED pause ends the drawdown, and a
            # pause immediately followed by an unobserved gap is not evidence
            fwd = lev.index[(lev.index > t)][:3]
            sustained = len(fwd) >= 2 and all(
                abs(lev[f] - lev[t]) < ANOMALY_RATE for f in fwd) and \
                all(gaps[f] <= MAX_GAUGE_GAP_DAYS for f in fwd)
            reg[t] = "POSTBREACH_TRANSITION" if sustained else "REGIME_UNRESOLVED"
        else:
            reg[t] = "POSTBREACH_TRANSITION" if cum >= CUM_LOSS_MIN else \
                     "REGIME_UNRESOLVED"
    # a long run of identical values is a suspected station floor, not a state
    same = (lev.diff() == 0).astype(int)
    run = same.groupby((same != same.shift()).cumsum()).transform("sum")
    reg[(same == 1) & (run >= FLAT_RUN_MIN)] = "REGIME_UNRESOLVED"
    return reg


def roles_for(r, targets):
    roles = []
    reg = r["regime"]
    if reg == "PREBREACH_IMPOUNDED":
        roles += ["PREBREACH_REFERENCE", "RADIOMETRIC_CALIBRATION"]
    elif reg == "BREACH_ONSET":
        roles += ["BREACH_ONSET_OBSERVATION", "HYDRAULIC_TRANSITION",
                  "TEMPORAL_CHANGE_REFERENCE"]
    elif reg == "ACTIVE_DRAWDOWN":
        roles += ["POSTBREACH_DRAWDOWN", "HYDRAULIC_TRANSITION",
                  "LOWER_STAGE_GEOMETRY", "CHANNEL_EXPOSURE_REFERENCE",
                  "TEMPORAL_CHANGE_REFERENCE", "FLOODED_VEGETATION_DIAGNOSTIC",
                  "FUTURE_HYDRAULIC_MODEL_VALIDATION"]
    elif reg == "POSTBREACH_TRANSITION":
        roles += ["POSTBREACH_TRANSITION_OBSERVATION", "LOWER_STAGE_GEOMETRY",
                  "CHANNEL_EXPOSURE_REFERENCE", "TEMPORAL_CHANGE_REFERENCE"]
    else:
        roles += ["REGIME_UNRESOLVED", "TEMPORAL_CHANGE_REFERENCE"]
    if r["gauge_match_method"] == "WSE_UNRESOLVED" or \
            not np.isfinite(r.get("gauge_wse_m", np.nan)):
        roles.append("WSE_UNRESOLVED")
        return "|".join(roles)
    adh, tgt = r["abs_dH_nearest_m"], r["nearest_target"]
    if reg == "PREBREACH_IMPOUNDED":
        if adh <= TOL_TARGET:
            roles.append(f"{tgt}_TARGET_STAGE_GEOMETRY")
        elif adh <= TOL_NEAR:
            roles.append(f"{tgt}_NEAR_TARGET_STAGE_GEOMETRY")
        else:
            roles.append("HIGHER_STAGE_GEOMETRY" if r["dH_nearest_m"] > 0
                         else "LOWER_STAGE_GEOMETRY")
    roles += ["STAGE_RESPONSE_OBSERVATION", "CLASSIFIER_DIAGNOSTIC"]
    return "|".join(roles)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    lev = gauge_series()
    inv3 = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv")
    inv3 = inv3.sort_values("H_evrf2019_m").reset_index(drop=True)
    targets = {r.contour_id: float(r.H_evrf2019_m) for r in inv3.itertuples()}
    print("target slices (measured stage order):")
    for c, t in targets.items():
        print(f"  {c} = {t:.3f} m EVRF2019")

    # ------------------------------------------------- regime segmentation
    print("\n" + "=" * 78)
    print("REGIME SEGMENTATION (never from a single derivative threshold)")
    print("=" * 78)
    reg = segment_regimes(lev)
    RS = pd.DataFrame(dict(date=reg.index, wse_m=lev.values,
                           dHdt_m_per_day=lev.diff().values, regime=reg.values))
    RS.to_csv(CFG.TABLES / "hist25b_regime_segmentation.csv", index=False)
    print(reg.value_counts().to_string())
    print("\n  around the breach:")
    for t in pd.date_range("2023-06-03", "2023-07-04"):
        if t in reg.index:
            print(f"    {t.date()}  {lev[t]:7.3f} m  {reg[t]}")
        else:
            print(f"    {t.date()}      ----     NO GAUGE RECORD")

    # ------------------------------------------------- assets -> events
    print("\n" + "=" * 78)
    print("ASSET INVENTORY -> OBSERVATION EVENTS")
    print("=" * 78)
    if ASSETS_CSV.exists():
        A = pd.read_csv(ASSETS_CSV)
        print(f"  reusing cached asset inventory: {len(A)} items")
    else:
        rows = []
        for year in YEARS:
            for q0, q1 in (("01-01", "03-31"), ("04-01", "06-30"),
                           ("07-01", "09-30"), ("10-01", "12-31")):
                feats = stac_range(f"{year}-{q0}", f"{year}-{q1}")
                for f in feats:
                    p = f["properties"]
                    rows.append(dict(
                        asset_id=f["id"], sensing_datetime=p["datetime"][:19],
                        date=p["datetime"][:10],
                        relative_orbit=p.get("sat:relative_orbit"),
                        orbit_state=p.get("sat:orbit_state", "UNKNOWN"),
                        platform=p.get("platform", ""),
                        polarisations="+".join(p.get("sar:polarizations", []))))
                print(f"  {year} {q0[:2]}-{q1[:2]}: {len(feats)} items", flush=True)
        A = pd.DataFrame(rows).drop_duplicates("asset_id")
        A = A[A.relative_orbit.notna()].copy()
        A["relative_orbit"] = A.relative_orbit.astype(int)
        A["event_id"] = (A.date + "_orb" + A.relative_orbit.astype(str) + "_"
                         + A.orbit_state.str[:3].str.upper())
        A.to_csv(ASSETS_CSV, index=False)
    print(f"\n  {len(A)} raw STAC items")

    E = (A.groupby(["event_id", "date", "relative_orbit", "orbit_state"])
           .agg(n_assets=("asset_id", "size"),
                sensing_start=("sensing_datetime", "min"),
                sensing_end=("sensing_datetime", "max"),
                platform=("platform", "first"),
                polarisations=("polarisations", "first"))
           .reset_index())
    print(f"  {len(E)} independent observation events "
          f"({len(A)/max(len(E),1):.2f} assets per overpass)")
    print("  tiles from one overpass are NOT independent acquisitions")

    # ------------------------------------------------- gauge + state
    ts = pd.to_datetime(E.sensing_start)
    gm = pd.DataFrame([match_gauge(t, lev) for t in ts])
    E = pd.concat([E.reset_index(drop=True), gm], axis=1)
    E["pre_post_breach"] = np.where(ts < BREACH, "PRE", "POST")
    day_idx = pd.to_datetime(E.date)
    E["regime"] = [reg.get(pd.Timestamp(d), "REGIME_UNRESOLVED") for d in day_idx]
    for c, t in targets.items():
        E[f"dH_{c}_m"] = E.gauge_wse_m - t
    arr = E[[f"dH_{c}_m" for c in targets]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        safe = np.where(np.isnan(arr), np.inf, np.abs(arr))
        k = np.argmin(safe, axis=1)
    allnan = ~np.isfinite(arr).any(axis=1)
    E["nearest_target"] = [list(targets)[i] for i in k]
    E.loc[allnan, "nearest_target"] = ""
    E["dH_nearest_m"] = arr[np.arange(len(arr)), k]
    E["abs_dH_nearest_m"] = np.abs(E.dH_nearest_m)
    E["stage_horizontal_error_m"] = E.abs_dH_nearest_m / MARGIN_SLOPE
    E["roles"] = [roles_for(r, targets) for _, r in E.iterrows()]
    E.to_csv(EVENTS_CSV, index=False)

    print("\n  gauge match methods:")
    print(E.gauge_match_method.value_counts().to_string())
    print("\n  regimes across events:")
    print(E.regime.value_counts().to_string())
    res = E[E.gauge_match_method != "WSE_UNRESOLVED"]
    if len(res):
        print(f"\n  WSE uncertainty: pre-breach median "
              f"{res[res.pre_post_breach=='PRE'].wse_uncertainty_m.median():.3f} m, "
              f"post-breach median "
              f"{res[res.pre_post_breach=='POST'].wse_uncertainty_m.median():.3f} m")

    # ------------------------------------------------- the 2023-06-08 event
    print("\n" + "=" * 78)
    print("THE 2023-06-08 EVENT — retained, multi-role, never relabelled H1")
    print("=" * 78)
    s = E[E.date == "2023-06-08"]
    for r in s.itertuples():
        print(f"  {r.event_id}  {r.n_assets} assets  WSE {r.gauge_wse_m:.3f} "
              f"+/- {r.wse_uncertainty_m:.3f} m  [{r.regime}]")
        print(f"    dH to nearest target ({r.nearest_target}): "
              f"{r.dH_nearest_m:+.3f} m  -> NOT an H1 observation")
        for role in r.roles.split("|"):
            print(f"      {role}")

    # ------------------------------------------------- stage availability
    print("\n" + "=" * 78)
    print("STAGE AVAILABILITY (events, pre-breach, resolved WSE only)")
    print("=" * 78)
    pre = E[(E.pre_post_breach == "PRE") &
            (E.gauge_match_method != "WSE_UNRESOLVED")]
    srows = []
    for c, t in targets.items():
        for tol in (0.10, 0.20, 0.30, 0.50):
            s = pre[(pre.gauge_wse_m - t).abs() <= tol]
            orbits = sorted(int(o) for o in s.relative_orbit.unique())
            srows.append(dict(contour_id=c, target_wse_m=t, tolerance_m=tol,
                              horizontal_error_m=tol / MARGIN_SLOPE,
                              events=len(s), dates=s.date.nunique(),
                              n_orbits=len(orbits),
                              orbits="|".join(map(str, orbits))))
            print(f"  {c} |dH|<={tol:.2f} m (<={tol/MARGIN_SLOPE:5.1f} m horiz): "
                  f"{len(s):3d} events, {len(orbits)} orbits {orbits}")
    pd.DataFrame(srows).to_csv(CFG.TABLES / "hist25b_stage_matching.csv",
                               index=False)

    # ------------------------------------------------- figure
    fig, ax = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    a = ax[0]
    a.plot(lev.index, lev.values, color=GREY, lw=0.9, zorder=1,
           label="gauge WSE (daily)")
    cols = {"PREBREACH_IMPOUNDED": BLUE, "BREACH_ONSET": PURPLE,
            "ACTIVE_DRAWDOWN": RED, "POSTBREACH_TRANSITION": AMBER,
            "REGIME_UNRESOLVED": GREY}
    for rg, c in cols.items():
        m = reg == rg
        if m.any():
            a.scatter(lev.index[m], lev.values[m], s=5, color=c, label=rg, zorder=2)
    for c, t in targets.items():
        a.axhline(t, color=GREEN, ls=":", lw=1.1)
        a.text(lev.index[5], t + 0.05, c, fontsize=8, color=GREEN)
    a.axvline(BREACH, color=RED, ls="--", lw=1.3)
    a.set_ylabel("WSE (m EVRF2019)")
    a.legend(fontsize=7.6, ncol=3, loc="lower left")
    a.set_title("a · gauge hydrograph with conservative regime segmentation\n"
                "POSTBREACH_RIVER_DOMINATED is deliberately never assigned "
                "from the hydrograph", fontsize=10.4, loc="left")
    a.grid(alpha=0.22)
    a = ax[1]
    ok = E[E.gauge_match_method != "WSE_UNRESOLVED"]
    bad = E[E.gauge_match_method == "WSE_UNRESOLVED"]
    a.scatter(pd.to_datetime(ok.date), ok.gauge_wse_m, s=9, color=BLUE,
              label=f"events with resolved WSE (n={len(ok)})")
    if len(bad):
        a.scatter(pd.to_datetime(bad.date),
                  np.full(len(bad), float(lev.min())), s=9, color=RED,
                  marker="x", label=f"WSE_UNRESOLVED (n={len(bad)})")
    a.axvline(BREACH, color=RED, ls="--", lw=1.3)
    a.set_ylabel("WSE at event (m)")
    a.set_xlabel("date")
    a.legend(fontsize=7.8); a.grid(alpha=0.22)
    a.set_title("b · S1 observation events placed on the stage-time plane",
                fontsize=10.4, loc="left")
    fig.suptitle("hist25b Gate 5 · canonical stage-time inventory of "
                 "Sentinel-1 observation events", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_stage_time_inventory.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> {ASSETS_CSV}")
    print(f"-> {EVENTS_CSV}")
    print(f"-> {CFG.TABLES/'hist25b_regime_segmentation.csv'}")
    print(f"-> {CFG.TABLES/'hist25b_stage_matching.csv'}")
    print(f"-> {FIGDIR/'hist25b_stage_time_inventory.png'}")
    print("\nSTOP. No classifier work until these inventories are reviewed.")


if __name__ == "__main__":
    main()
