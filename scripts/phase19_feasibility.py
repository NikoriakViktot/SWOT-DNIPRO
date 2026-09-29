#!/usr/bin/env python
"""Phase 19 — STEP 1: post-breach channel-profile feasibility census (no imagery).

For every post-breach ATL13 date, quantify — from ATL13 geometry + SWORD alone,
before any Sentinel download — how likely that date is to yield a usable
MAIN_CHANNEL longitudinal profile.  The "channel-eligible" figures are an UPPER
BOUND: ~50 % of near-SWORD post-breach ATL13 returns turn out to be non-water
(drained bed) once Sentinel-2 is applied, so a date that looks marginal here will
not survive classification.

Output: outputs/tables/postbreach_profile_feasibility.csv
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
from swot_dnipro import sword as SW

PROC = CFG.ROOT / "data" / "processed"
CHANNEL_DIST_KM = 1.5          # same as phase19_classify.CHANNEL_DIST_KM
BREACH = "2023-06-06"
POST = "2023-09-01"

# HIGH needs enough margin to survive the ~50 % water loss + the fit gate
# (>=20 seg, >=20 km span, >=3 bins) in phase19_profiles.fit()
HIGH = dict(n=60, span=25.0, bins5=4)
MED = dict(n=25, span=20.0, bins5=3)


def occupied_bins(chain_km, width):
    if len(chain_km) == 0:
        return 0
    return int(np.unique(np.floor(np.asarray(chain_km) / width)).size)


def main() -> None:
    ch = pd.read_parquet(PROC / "profiles" / "sword_dnipro_channel.parquet")

    seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet")
    seg["dt"] = pd.to_datetime(seg["time"], utc=True, errors="coerce").dt.tz_localize(None)
    seg = seg.dropna(subset=["lat", "lon", "dt"])
    seg = seg[seg.dt >= BREACH].copy()
    seg["date"] = seg["dt"].dt.normalize()
    cch, cdist, _, _ = SW.assign_chainage(seg.lon, seg.lat, ch)
    seg["chain_km"], seg["dist_to_sword_km"] = cch, cdist
    seg = seg.dropna(subset=["chain_km"])
    seg["period"] = np.where(seg.dt < POST, "BREACH_DRAWDOWN", "POST_BREACH")

    # which dates already carry imagery / already produced a channel profile
    mm = pd.read_csv(CFG.TABLES / "atl13_image_matchups.csv", parse_dates=["date"]) \
        if (CFG.TABLES / "atl13_image_matchups.csv").exists() else pd.DataFrame(columns=["date"])
    have_img = set(pd.to_datetime(mm["date"]).dt.normalize()) if len(mm) else set()
    prof = pd.read_csv(CFG.TABLES / "channel_only_profiles.csv", parse_dates=["date"]) \
        if (CFG.TABLES / "channel_only_profiles.csv").exists() else pd.DataFrame()
    have_channel = set()
    if len(prof):
        pc = prof[prof.subset.isin(["MAIN_CHANNEL", "PRE_CHANNEL"])]
        have_channel = set(pd.to_datetime(pc["date"]).dt.normalize())

    rows = []
    for d, g in seg.groupby("date"):
        gc = g[g.dist_to_sword_km < CHANNEL_DIST_KM]
        x = gc.chain_km.to_numpy()
        span = float(x.max() - x.min()) if len(x) else 0.0
        b5, b10 = occupied_bins(x, 5.0), occupied_bins(x, 10.0)
        rec = {
            "date": d.date(), "period": g.period.iloc[0],
            "n_seg_total": len(g), "n_seg_channel": len(gc),
            "n_beams": int(g["gt"].nunique()), "n_rgts": int(g["rgt"].nunique()),
            "n_cycles": int(g.cycle.nunique()) if "cycle" in g else -1,
            "channel_span_km": round(span, 1),
            "bins_5km": b5, "bins_10km": b10,
            "chain_min_km": round(float(x.min()), 1) if len(x) else np.nan,
            "chain_max_km": round(float(x.max()), 1) if len(x) else np.nan,
            "lon_min": round(float(g.lon.min()), 3), "lon_max": round(float(g.lon.max()), 3),
            "frac_within_1p5km": round(len(gc) / max(len(g), 1), 3),
            "min_dist_to_sword_km": round(float(g.dist_to_sword_km.min()), 2),
            "has_imagery_now": d in have_img,
            "produced_channel_profile": d in have_channel,
        }
        if rec["n_seg_channel"] >= HIGH["n"] and span >= HIGH["span"] and b5 >= HIGH["bins5"]:
            rec["priority"] = "HIGH"
        elif rec["n_seg_channel"] >= MED["n"] and span >= MED["span"] and b5 >= MED["bins5"]:
            rec["priority"] = "MEDIUM"
        elif rec["n_seg_channel"] >= 10 and span >= 12.0:
            rec["priority"] = "LOW"
        else:
            rec["priority"] = "UNUSABLE"
        rows.append(rec)

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNUSABLE": 3}
    fz = pd.DataFrame(rows)
    fz["_o"] = fz.priority.map(order)
    fz = fz.sort_values(["_o", "date"]).drop(columns="_o")
    out = CFG.TABLES / "postbreach_profile_feasibility.csv"
    fz.to_csv(out, index=False)

    print(f"post-breach ATL13 dates: {len(fz)}  "
          f"(BREACH_DRAWDOWN {int((fz.period=='BREACH_DRAWDOWN').sum())}, "
          f"POST_BREACH {int((fz.period=='POST_BREACH').sum())})\n")
    piv = fz.groupby(["priority", "period"]).size().unstack(fill_value=0)
    print(piv.to_string(), "\n")

    for pr in ["HIGH", "MEDIUM"]:
        sub = fz[fz.priority == pr]
        no_img = sub[~sub.has_imagery_now]
        print(f"{pr}: {len(sub)} dates  |  with imagery now: {int(sub.has_imagery_now.sum())}  "
              f"|  produced a channel profile: {int(sub.produced_channel_profile.sum())}  "
              f"|  NEED imagery: {len(no_img)}")
        cols = ["date", "period", "n_seg_channel", "n_beams", "n_rgts", "channel_span_km",
                "bins_5km", "chain_min_km", "chain_max_km", "has_imagery_now",
                "produced_channel_profile"]
        print(sub[cols].to_string(index=False), "\n")

    print(f"-> {out}")


if __name__ == "__main__":
    main()
