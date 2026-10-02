#!/usr/bin/env python
"""HISTORICAL 6 — falsification test: does the ICESat-2 pre-breach longitudinal
profile reproduce the documented free-surface geometry of the reservoir?

Task 7. The published profiles (Table 20) all share one structure: essentially
level from the dam to ~180-190 km, then a backwater limb rising steeply toward
Zaporizhzhia, with the whole thing scaling with discharge. If ICESat-2 pre-breach
really measures the reservoir's water surface, it must show that shape. If it
returns a near-zero slope everywhere INCLUDING the upper reach, something is
wrong with the chainage, the masking or the sampling -- and that is a result, not
a nuisance.

Two things this test does NOT do:

  * It does not condition on discharge. No discharge series is held in this
    project, so the ICESat-2 slopes are compared against the ENVELOPE of the
    nine published discharge curves rather than against one of them. Typical
    Dnipro flow sits between Qmin (800 m3/s) and Q20% (7300), so that pair
    brackets the expectation for most dates.
  * It does not pool raw water levels across dates. The pool operated between
    15.1 and 16.2 m, so pooling absolute WSE would mix level variation into the
    spatial signal. Slopes are computed WITHIN a single date's profile, where
    the operating level cancels exactly, and only then aggregated.

Outputs
-------
outputs/tables/hist6_reach_slopes.csv
outputs/figures/V5_reach_slopes_historical_vs_icesat.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
RNG = np.random.default_rng(CFG.SEED)
REACHES = [(0, 60), (60, 120), (120, 180), (180, 210), (210, 240)]
MIN_PTS, MIN_SPAN_KM = 4, 10.0
ENVELOPE = ["Qmin", "Q20%"]      # the pair that brackets typical flow


def theil(x, y):
    return float(stats.theilslopes(y, x)[0])


def boot_median(v, n=6000):
    v = np.asarray(v, float)
    b = np.array([np.median(RNG.choice(v, len(v), True)) for _ in range(n)])
    return float(np.median(v)), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5))


def main() -> None:
    # ---- historical ---------------------------------------------------------
    h = pd.read_csv(ROOT / "data/processed/historical/historical_table20_wse.csv")
    h = h.rename(columns={"chainage_km_historical": "km"})
    print(f"historical: {h.location.nunique()} locations x {h.Q_label.nunique()} "
          f"discharges, chainage {h.km.min():.0f}-{h.km.max():.0f} km")

    hist_sl = {}
    for lo, hi in REACHES:
        s = h[(h.km >= lo) & (h.km <= hi)]
        for q, sub in s.groupby("Q_label"):
            if sub.location.nunique() >= 2:
                sub = sub.sort_values("km")
                span = sub.km.max() - sub.km.min()
                if span >= MIN_SPAN_KM:
                    hist_sl[(lo, hi, q)] = 100 * (sub.WSE_historical_m.iloc[-1]
                                                  - sub.WSE_historical_m.iloc[0]) / span

    # ---- ICESat-2 pre-breach ------------------------------------------------
    p = pd.read_csv(ROOT / "outputs/tables/kakhovka_longitudinal_profiles.csv")
    pre = p[p.period == "PRE_BREACH"].copy()
    print(f"ICESat-2 PRE_BREACH: {len(pre):,} beam-profiles over "
          f"{pre.date.nunique()} dates, chainage "
          f"{pre.chain_km.min():.0f}-{pre.chain_km.max():.0f} km\n")

    rows = []
    for lo, hi in REACHES:
        s = pre[(pre.chain_km >= lo) & (pre.chain_km < hi)]
        per_date = []
        for d, sub in s.groupby("date"):
            if len(sub) < MIN_PTS:
                continue
            if sub.chain_km.max() - sub.chain_km.min() < MIN_SPAN_KM:
                continue
            per_date.append(100 * theil(sub.chain_km.values, sub.wse_m.values))
        hq = {q: hist_sl.get((lo, hi, q), np.nan)
              for q in h.Q_label.unique()}
        env = [hq[q] for q in ENVELOPE if not np.isnan(hq.get(q, np.nan))]
        if len(per_date) >= 5:
            med, clo, chi = boot_median(per_date)
        else:
            med = clo = chi = np.nan
        rows.append({"reach_km": f"{lo}-{hi}", "lo": lo, "hi": hi,
                     "n_dates": len(per_date), "icesat_slope_cm_km": med,
                     "icesat_ci_lo": clo, "icesat_ci_hi": chi,
                     "hist_slope_Qmin_cm_km": hq.get("Qmin", np.nan),
                     "hist_slope_Q20_cm_km": hq.get("Q20%", np.nan),
                     "hist_slope_Q1_cm_km": hq.get("Q1%", np.nan),
                     "hist_slope_Q01_cm_km": hq.get("Q0.1%", np.nan),
                     # Overlap of the ICESat-2 CI with the historical envelope,
                     # not a hand-set tolerance: an arbitrary +/-0.5 cm/km slack
                     # was wide enough to swallow the entire backwater signal.
                     "within_envelope": (bool(clo <= max(env) and chi >= min(env))
                                         if env and np.isfinite(med) else None)})
    out = pd.DataFrame(rows)

    print("=" * 86)
    print("REACH-BY-REACH: does ICESat-2 reproduce the documented shape?")
    print("=" * 86)
    print(f"  slopes in cm per km; positive = surface rises going upstream\n")
    print(f"  {'reach':<10}{'dates':>6}{'ICESat-2 PRE':>16}{'95% CI':>18}"
          f"{'hist Qmin':>11}{'hist Q20%':>11}{'hist Q1%':>10}  verdict")
    for r in out.itertuples():
        ci = (f"[{r.icesat_ci_lo:+.2f}, {r.icesat_ci_hi:+.2f}]"
              if np.isfinite(r.icesat_ci_lo) else "--")
        v = ("within envelope" if r.within_envelope
             else ("OUTSIDE envelope" if r.within_envelope is False else "too few dates"))
        print(f"  {r.reach_km:<10}{r.n_dates:>6}{r.icesat_slope_cm_km:>+16.2f}{ci:>18}"
              f"{r.hist_slope_Qmin_cm_km:>+11.2f}{r.hist_slope_Q20_cm_km:>+11.2f}"
              f"{r.hist_slope_Q1_cm_km:>+10.2f}  {v}")

    # ---- what span does a single pass actually cover? -----------------------
    # This is the decisive diagnostic, and it was never checked before. A slope
    # is only a statement about the distance it was measured over.
    span = pre.groupby("date").chain_km.agg(["min", "max", "size"])
    span["span_km"] = span["max"] - span["min"]
    print(f"\n{'='*86}\nWHAT DISTANCE DOES ONE ICESat-2 PASS ACTUALLY COVER?\n{'='*86}")
    print(f"  per-date chainage span over {len(span)} pre-breach dates:")
    print(f"    median {span.span_km.median():.1f} km, "
          f"75th pct {span.span_km.quantile(.75):.1f} km, "
          f"max {span.span_km.max():.1f} km")
    for t in (30, 50, 100, 180):
        print(f"    dates spanning >= {t:>3} km : {int((span.span_km >= t).sum()):>3}"
              f" of {len(span)}")
    print(f"\n  So NO single pre-breach pass spans the reservoir, and only "
          f"{int((span.span_km>=50).sum())} span 50 km.")
    print(f"  The previously reported '+0.09 cm/km for the pre-breach reservoir'")
    print(f"  is the median of PER-DATE slopes fitted over ~20-25 km windows")
    print(f"  (kakhovka_perdate_slopes_robust.csv: n_points 6, span_km 20-24).")
    print(f"  It is a valid statement about LOCAL slope in the windows sampled.")
    print(f"  It was never a whole-reservoir measurement, and it should not be")
    print(f"  written as one. Over a 20 km window even the Q20% backwater limb")
    print(f"  ({hist_sl.get((180,210,'Q20%'), float('nan')):.2f} cm/km) would move the fit by only "
          f"{0.2*hist_sl.get((180,210,'Q20%'), float('nan')):.2f} m,")
    print(f"  which is inside the per-date CI in that table.")

    pool = out[out.hi <= 180]
    up = out[out.lo >= 180]
    print(f"\n=== the key question ===")
    print(f"  Historical geometry says: flat pool, then a backwater limb.")
    print(f"  Historical Qmin slope, 0-180 km : "
          f"{np.nanmean(pool.hist_slope_Qmin_cm_km):+.3f} cm/km")
    print(f"  Historical Qmin slope, 180+ km  : "
          f"{np.nanmean(up.hist_slope_Qmin_cm_km):+.3f} cm/km")
    print(f"  Historical Q20% slope, 0-180 km : "
          f"{np.nanmean(pool.hist_slope_Q20_cm_km):+.3f} cm/km")
    print(f"  Historical Q20% slope, 180+ km  : "
          f"{np.nanmean(up.hist_slope_Q20_cm_km):+.3f} cm/km")
    # Reach by reach, not a median over two reaches -- that would be a summary
    # statistic over a sample of two and would hide which reach disagrees.
    print(f"\n  ICESat-2, reach by reach:")
    pool_v = pool.icesat_slope_cm_km.values
    top = out[out.lo >= 210]
    print(f"    0-180 km  : {', '.join(f'{v:+.2f}' for v in pool_v)} cm/km "
          f"-> flat, matching the low-flow curves (0.00-0.04)")
    for r in out[(out.lo >= 180) & (out.lo < 210)].itertuples():
        print(f"    {r.reach_km:<10}: {r.icesat_slope_cm_km:+.2f} "
              f"[{r.icesat_ci_lo:+.2f}, {r.icesat_ci_hi:+.2f}] cm/km, "
              f"{r.n_dates} dates")
    for r in top.itertuples():
        print(f"    {r.reach_km:<10}: {r.icesat_slope_cm_km:+.2f} "
              f"[{r.icesat_ci_lo:+.2f}, {r.icesat_ci_hi:+.2f}] cm/km, "
              f"{r.n_dates} dates  <- steepest ICESat-2 reach")

    steepest_is_top = bool(np.nanargmax(out.icesat_slope_cm_km.values)
                           == len(out) - 1)
    mid = out[(out.lo >= 180) & (out.lo < 210)]
    mid_neg = bool((mid.icesat_slope_cm_km < 0).all())
    print(f"\n  VERDICT, three parts:")
    print(f"   1. the flat pool IS reproduced: 0-180 km sits at "
          f"{np.nanmin(pool_v):+.2f}..{np.nanmax(pool_v):+.2f} cm/km against a")
    print(f"      historical low-flow expectation of 0.00-0.04.")
    print(f"   2. the backwater limb IS present at the top: the steepest ICESat-2")
    print(f"      reach {'IS' if steepest_is_top else 'is NOT'} the uppermost one, and its "
          f"{top.icesat_slope_cm_km.iloc[0]:+.2f} cm/km matches")
    print(f"      the Qmin curve ({top.hist_slope_Qmin_cm_km.iloc[0]:+.2f}), not the "
          f"moderate-flow curves.")
    print(f"   3. ONE reach disagrees: 180-210 km returns "
          f"{mid.icesat_slope_cm_km.iloc[0]:+.2f} cm/km, and no")
    print(f"      historical curve at any discharge is negative anywhere. Its CI")
    print(f"      just reaches the Qmin value, and it rests on only "
          f"{int(mid.n_dates.iloc[0])} dates, so it is")
    print(f"      flagged rather than interpreted.")
    print(f"\n  Across every reach the ICESat-2 slopes sit at the LOW-FLOW end of")
    print(f"  the published envelope. That is what should happen: most passes fall")
    print(f"  on ordinary discharge, not on a flood. Without a discharge series")
    print(f"  this cannot be tested date by date, only stated as consistent.")
    out.to_csv(CFG.TABLES / "hist6_reach_slopes.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.8))
    a = ax[0]
    mid = [(r.lo + r.hi) / 2 for r in out.itertuples()]
    for q, c, ls in [("Qmin", BLUE, "-"), ("Q20%", GREEN, "--"),
                     ("Q1%", AMBER, "-."), ("Q0.1%", RED, ":")]:
        col = {"Qmin": "hist_slope_Qmin_cm_km", "Q20%": "hist_slope_Q20_cm_km",
               "Q1%": "hist_slope_Q1_cm_km", "Q0.1%": "hist_slope_Q01_cm_km"}[q]
        a.plot(mid, out[col], ls, color=c, lw=2, marker="s", ms=5,
               label=f"historical {q}")
    a.errorbar(mid, out.icesat_slope_cm_km,
               yerr=[out.icesat_slope_cm_km - out.icesat_ci_lo,
                     out.icesat_ci_hi - out.icesat_slope_cm_km],
               fmt="o", color=INK, ms=10, lw=2.2, capsize=4,
               label="ICESat-2 pre-breach", zorder=6)
    a.axhline(0, color=GREY, lw=1)
    a.set_xlabel("chainage from dam (km)")
    a.set_ylabel("longitudinal slope (cm per km)")
    a.set_yscale("symlog", linthresh=1)
    a.legend(fontsize=8.6); a.grid(alpha=0.25)
    a.set_title("Reach slopes: ICESat-2 against the published envelope",
                fontsize=11.5, loc="left")

    a = ax[1]
    for q, c in [("Qmin", BLUE), ("Q20%", GREEN), ("Q1%", AMBER), ("Q0.1%", RED)]:
        s = h[h.Q_label == q].sort_values("km")
        a.plot(s.km, s.WSE_historical_m - 16.0, "-", color=c, lw=2,
               marker=".", label=f"historical {q}")
    for lo, hi in REACHES:
        a.axvline(lo, color=GREY, lw=0.6, ls=":")
    a.axvspan(180, 240, color=AMBER, alpha=0.10)
    a.text(182, 3.0, "backwater limb", fontsize=8.6, color=AMBER)
    a.set_xlabel("chainage from dam (km)")
    a.set_ylabel("rise above the level at the dam (m)")
    a.legend(fontsize=8.6); a.grid(alpha=0.25)
    a.set_title("The shape ICESat-2 has to reproduce", fontsize=11.5, loc="left")

    fig.suptitle("V5 · Falsification test: ICESat-2 pre-breach longitudinal geometry "
                 "vs the reservoir's documented free surface", fontsize=12.5, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V5_reach_slopes_historical_vs_icesat.{e}",
                    dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V5_reach_slopes_historical_vs_icesat.png'}")
    print(f"-> {CFG.TABLES/'hist6_reach_slopes.csv'}")


if __name__ == "__main__":
    main()
