#!/usr/bin/env python
"""P1 STEP 2-6 -- stratified selection for two validation datasets (2026-09-11
operator correction, second round). NOT "top-10 lowest cloud": Dataset A
(SA_2 multi-date/multi-year consensus) and Dataset B (satellite A(H),
stratified by Rozumivka gauge level, NOT by cloud alone) have different
selection objectives and are built separately, then unioned into one
download manifest. No download happens in this script -- metadata/ranking
only.

Outputs
-------
outputs/tables/P1_targeted_download_manifest.csv
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

from swot_dnipro import config as CFG

MIN_COVERAGE = 0.99
MIN_COVERAGE_B = 0.90   # relaxed for Dataset B only: extreme H (2023 drawdown/refill)
                        # events are optically rarer -- do not let a 99% coverage
                        # floor silently throw away the only low/high-H scenes
MAX_CLOUD_A = 30.0     # Dataset A: prioritise cleanliness, still not "perfect only"
MAX_CLOUD_B = 50.0     # Dataset B: a usable-but-cloudier unique H beats a 10th clean duplicate
GAUGE_START, CHECK_DATE = "2019-01-01", "2023-06-05"
N_LEVEL_CLASSES = 5
N_PER_CLASS = 2
TARGET_A = (8, 8)
MAIN_TILES = ("36TWS", "36TWT", "36TXT")     # cover 99.4% of the reservoir alone
NE_TILE = "36UXU"                             # only +0.6% area -- required only at CHECK_DATE
# est. GB per tile, from the 2023-06-05 cached set (0.85-1.1 GB observed)
EST_GB_PER_TILE = 1.05


def load_gauge_daily():
    g = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    roz = g[(g.station_or_domain == "station:80959") & (g.source == "gauge")].copy()
    roz["date"] = pd.to_datetime(roz.date)
    roz = roz.sort_values("date").drop_duplicates("date").set_index("date")["corrected_level_m"]
    return roz


def rate_flags(roz: pd.Series, dates: list[str]) -> pd.DataFrame:
    dH = roz.diff()  # per-day change, m/day (daily series so dH/dt == dH)
    thr = dH.abs().dropna()
    # data-driven bands from the whole pre-breach daily record, not asserted
    q50, q85 = thr.quantile([0.50, 0.85])
    rows = []
    for d in dates:
        ts = pd.Timestamp(d)
        h0 = roz.get(ts, np.nan)
        h_prev = roz.get(ts - pd.Timedelta(days=1), np.nan)
        h_next = roz.get(ts + pd.Timedelta(days=1), np.nan)
        rate = (h_next - h_prev) / 2.0 if np.isfinite(h_next) and np.isfinite(h_prev) else np.nan
        arate = abs(rate) if np.isfinite(rate) else np.nan
        if not np.isfinite(arate):
            flag = "UNKNOWN"
        elif arate <= q50:
            flag = "STABLE"
        elif arate <= q85:
            flag = "SLOW_CHANGE"
        else:
            flag = "RAPID_CHANGE" if arate > q85 * 2 else "RISING" if rate > 0 else "FALLING"
        rows.append({"date": d, "H_t": h0, "H_t_minus_1": h_prev, "H_t_plus_1": h_next,
                    "dH_dt_m_per_day": rate, "rate_flag": flag})
    print(f"  rate-of-change bands (data-driven from daily |dH|): "
          f"STABLE<={q50:.3f} m/day, SLOW_CHANGE<={q85:.3f} m/day, else RAPID/RISING/FALLING")
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    cand = pd.read_csv(CFG.TABLES / "prebreach_fullpool_scene_candidates.csv")
    roz = load_gauge_daily()
    good = cand[cand.native_coverage_fraction >= MIN_COVERAGE].copy()
    print(f"candidates with native_coverage_fraction >= {MIN_COVERAGE}: {len(good)} / {len(cand)}")

    # ================================================== Dataset A: SA_2 multi-year
    a_pool = good[good.est_cloud_over_reservoir_pct_weighted <= MAX_CLOUD_A].copy()
    a_pool["year"] = a_pool.date.str[:4]
    a_pool = a_pool.sort_values("est_cloud_over_reservoir_pct_weighted")
    picked_a = []
    for yr, g in a_pool.groupby("year"):
        picked_a.append(g.iloc[0])
        if len(g) > 1:
            # a 2nd pick from a different season if it improves coverage of the year
            others = g.iloc[1:]
            others = others.assign(month=others.date.str[5:7].astype(int))
            first_month = int(g.iloc[0].date[5:7])
            far = others.loc[(others.month - first_month).abs().idxmax()] if len(others) else None
            if far is not None and abs(int(far.date[5:7]) - first_month) >= 2:
                picked_a.append(far)
    dsA = pd.DataFrame(picked_a).drop_duplicates("date")
    lo, hi = TARGET_A
    if len(dsA) > hi:
        dsA = dsA.sort_values("est_cloud_over_reservoir_pct_weighted").head(hi)
    print(f"\nDataset A (SA_2 multi-year consensus): {len(dsA)} dates, years "
          f"{sorted(dsA.date.str[:4].unique())}")
    print(dsA[["date", "native_coverage_fraction", "est_cloud_over_reservoir_pct_weighted"]]
         .sort_values("date").to_string(index=False))

    # ================================================== Dataset B: A(H) stratified
    b_pool = cand[(cand.date >= GAUGE_START) & (cand.date <= CHECK_DATE)
                 & (cand.native_coverage_fraction >= MIN_COVERAGE_B)
                 & (cand.est_cloud_over_reservoir_pct_weighted <= MAX_CLOUD_B)
                 & cand.has_gauge].copy()
    print(f"\nDataset B candidate pool (coverage>=99%, gauge available, "
          f"cloud<={MAX_CLOUD_B}%): {len(b_pool)} dates")
    print("gauge level distribution over this pool:")
    print(b_pool.gauge_level_evrf2019_m.describe().to_string())

    edges = np.quantile(b_pool.gauge_level_evrf2019_m,
                        np.linspace(0, 1, N_LEVEL_CLASSES + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6  # inclusive bounds
    labels = ["LOW", "LOW-MEDIUM", "MEDIUM", "HIGH-MEDIUM", "HIGH"][:N_LEVEL_CLASSES]
    b_pool["level_class"] = pd.cut(b_pool.gauge_level_evrf2019_m, edges, labels=labels)
    print(f"\ndata-driven quantile edges (m EVRF2019): {np.round(edges, 3).tolist()}")

    rates = rate_flags(roz, b_pool.date.tolist())
    b_pool = b_pool.join(rates, on="date")

    picked_b = []
    for cls, g in b_pool.groupby("level_class", observed=True):
        g = g.sort_values(["rate_flag", "est_cloud_over_reservoir_pct_weighted"],
                          key=lambda s: s.map({"STABLE": 0, "SLOW_CHANGE": 1}).fillna(2)
                          if s.name == "rate_flag" else s)
        picked_b.append(g.head(N_PER_CLASS))
    dsB = pd.concat(picked_b).drop_duplicates("date") if picked_b else pd.DataFrame()
    # "maximum possible spread in H" is the explicit objective (operator, 2026-09-11)
    # -- do not let a STABLE-first sort inside a coarse quantile bin quietly drop
    # the pool's true extremes just because the extreme-H dates happen to be
    # drawdown/refill events (RAPID_CHANGE, not STABLE). Force them in, flagged.
    extreme_idx = [b_pool.gauge_level_evrf2019_m.idxmin(), b_pool.gauge_level_evrf2019_m.idxmax()]
    extremes = b_pool.loc[extreme_idx]
    added = extremes[~extremes.date.isin(dsB.date)]
    if len(added):
        print(f"  forcing in {len(added)} pool-extreme date(s) not picked by the "
              f"per-class STABLE-first sort: {added.date.tolist()} "
              f"(H={added.gauge_level_evrf2019_m.tolist()}, "
              f"rate_flag will be reported as-is, likely non-STABLE)")
        dsB = pd.concat([dsB, added]).drop_duplicates("date")
    print(f"\nDataset B (A(H) stratified by gauge level): {len(dsB)} dates")
    print(dsB[["date", "gauge_level_evrf2019_m", "level_class", "rate_flag",
              "dH_dt_m_per_day", "est_cloud_over_reservoir_pct_weighted"]]
         .sort_values("gauge_level_evrf2019_m").to_string(index=False))
    print(f"\nH range spanned by Dataset B: "
          f"{dsB.gauge_level_evrf2019_m.min():.2f} .. {dsB.gauge_level_evrf2019_m.max():.2f} m "
          f"EVRF2019 (spread {dsB.gauge_level_evrf2019_m.max()-dsB.gauge_level_evrf2019_m.min():.2f} m)")

    # ================================================== union -> manifest
    union_dates = sorted(set(dsA.date) | set(dsB.date) | {CHECK_DATE})
    print(f"\nunion of A + B + mandatory {CHECK_DATE}: {len(union_dates)} dates")

    disc = pd.read_csv(CFG.ROOT / "data/catalog/prebreach_sentinel_discovery_full.csv")
    disc = disc.sort_values("cloud_cover_metadata").drop_duplicates(["date", "tile_id"])
    manifest_rows = []
    for d in union_dates:
        reasons = []
        if d in set(dsA.date):
            reasons.append("SA2_validation")
        if d in set(dsB.date):
            reasons.append("AH_validation")
        if d == CHECK_DATE:
            reasons.append("2023-06-05_checkpoint")
        row_c = cand[cand.date == d]
        gauge_lvl = row_c.gauge_level_evrf2019_m.iloc[0] if len(row_c) else np.nan
        lvl_class = (dsB.set_index("date").level_class.get(d, "")
                    if len(dsB) else "")
        rate_flag = (dsB.set_index("date").rate_flag.get(d, "")
                    if len(dsB) else "")
        dHdt = (dsB.set_index("date").dH_dt_m_per_day.get(d, np.nan)
               if len(dsB) else np.nan)
        cov = row_c.native_coverage_fraction.iloc[0] if len(row_c) else np.nan
        cloud = row_c.est_cloud_over_reservoir_pct_weighted.iloc[0] if len(row_c) else np.nan
        tiles_for_date = list(MAIN_TILES) + ([NE_TILE] if d == CHECK_DATE else [])
        for t in tiles_for_date:
            r = disc[(disc.date == d) & (disc.tile_id == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            manifest_rows.append({
                "date": d, "product_id": r.product_id, "tile_id": t,
                "reason_for_selection": "|".join(reasons),
                "SA2_validation": d in set(dsA.date),
                "AH_validation": d in set(dsB.date),
                "H_Rozumivka_EVRF2019": gauge_lvl,
                "hydrological_level_class": lvl_class,
                "dH_dt": dHdt, "rate_flag": rate_flag,
                "coverage_fraction": cov, "cloud_fraction": cloud,
                "cached": bool(r.cached_locally),
                "download_required": not bool(r.cached_locally),
                "estimated_size_GB": EST_GB_PER_TILE,
            })
    man = pd.DataFrame(manifest_rows)
    out = CFG.TABLES / "P1_targeted_download_manifest.csv"
    man.to_csv(out, index=False)
    todo = man[man.download_required]
    print(f"\n-> {out}")
    print(f"total manifest rows (date x tile): {len(man)}")
    print(f"already cached: {int(man.cached.sum())}")
    print(f"TO DOWNLOAD: {len(todo)} granules, ~{len(todo) * EST_GB_PER_TILE:.1f} GB estimated")
    print(todo.groupby("tile_id").size().to_string())


if __name__ == "__main__":
    main()
