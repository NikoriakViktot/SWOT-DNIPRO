#!/usr/bin/env python
"""Phase 18/20 — robust statistics for the reservoir → river transition.

Adds to the earlier OLS-only result:
  * Theil-Sen slope per date (resistant to residual-pond outliers)
  * spatial-span sensitivity (does the minimum span drive the answer?)
  * bootstrap CI on the PRE vs POST median-slope difference
  * two-sided permutation test on the median-slope difference
  * sign test on the fraction of positive slopes

Independent unit: **one date/profile**. Segments and beams within a date are
correlated and are never treated as replicates.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import nmad

RNG = np.random.default_rng(CFG.SEED)
SPANS = [10.0, 20.0, 30.0, 40.0]
MIN_PTS, CHAIN_SEP = 3, 5.0


def n_distinct(v, sep=CHAIN_SEP):
    v = np.sort(np.asarray(v))
    keep = [v[0]]
    for x in v[1:]:
        if x - keep[-1] > sep:
            keep.append(x)
    return len(keep)


def fit(df, min_span):
    out = []
    for (date, period), g in df.groupby(["date", "period"]):
        span = g.chain_km.max() - g.chain_km.min()
        if len(g) < MIN_PTS or n_distinct(g.chain_km.values) < MIN_PTS or span < min_span:
            continue
        x, y = g.chain_km.to_numpy(float), g.wse_m.to_numpy(float)
        ols = float(np.polyfit(x, y, 1)[0]) * 100
        ts = stats.theilslopes(y, x, 0.95)
        out.append({"date": date, "period": period, "n_points": len(g), "span_km": span,
                    "slope_ols_cm_km": ols, "slope_theilsen_cm_km": float(ts[0]) * 100,
                    "ts_lo_cm_km": float(ts[2]) * 100, "ts_hi_cm_km": float(ts[3]) * 100,
                    "wse_range_m": float(y.max() - y.min()),
                    "wse_p95_p05_m": float(np.percentile(y, 95) - np.percentile(y, 5)),
                    "wse_nmad_m": nmad(y)})
    return pd.DataFrame(out)


def perm_test(a, b, n=20000):
    """Two-sided permutation test on the difference of medians."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = np.median(b) - np.median(a)
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(np.median(pool[na:]) - np.median(pool[:na])) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def boot_diff(a, b, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [np.median(RNG.choice(b, len(b), True)) - np.median(RNG.choice(a, len(a), True))
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    pts = pd.read_csv(CFG.FIGDATA / "FigD_kakhovka_profile_points.csv")
    pts = pts[["date", "period", "chain_km", "wse_m"]].dropna()

    # ---- span sensitivity --------------------------------------------------
    sens = []
    for ms in SPANS:
        p = fit(pts, ms)
        for per in ["PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"]:
            s = p[p.period == per]
            if s.empty:
                continue
            sens.append({"min_span_km": ms, "period": per, "n_dates": len(s),
                         "median_ols": float(np.median(s.slope_ols_cm_km)),
                         "median_theilsen": float(np.median(s.slope_theilsen_cm_km)),
                         "frac_pos_ols": float((s.slope_ols_cm_km > 0).mean()),
                         "frac_pos_ts": float((s.slope_theilsen_cm_km > 0).mean()),
                         "median_wse_p95_p05_m": float(np.median(s.wse_p95_p05_m))})
    sdf = pd.DataFrame(sens)
    sdf.to_csv(CFG.FIGDATA / "FigG_span_sensitivity.csv", index=False)
    print("=== span sensitivity (min chainage span required) ===")
    print(sdf.to_string(index=False))

    # ---- headline at the 20 km threshold -----------------------------------
    prof = fit(pts, 20.0)
    prof.to_csv(CFG.FIGDATA / "FigG_perdate_slopes_robust.csv", index=False)
    prof.to_csv(CFG.TABLES / "kakhovka_perdate_slopes_robust.csv", index=False)

    pre = prof[prof.period == "PRE_BREACH"]
    post = prof[prof.period == "POST_BREACH"]
    draw = prof[prof.period == "BREACH_DRAWDOWN"]

    rows = []
    for est in ["slope_ols_cm_km", "slope_theilsen_cm_km"]:
        obs, p_perm = perm_test(pre[est], post[est])
        lo, hi = boot_diff(pre[est], post[est])
        k_pre, k_post = int((pre[est] > 0).sum()), int((post[est] > 0).sum())
        sg_pre = stats.binomtest(k_pre, len(pre), 0.5).pvalue
        sg_post = stats.binomtest(k_post, len(post), 0.5).pvalue
        mw = stats.mannwhitneyu(pre[est], post[est], alternative="two-sided")
        rows.append({
            "estimator": "OLS" if "ols" in est else "Theil-Sen",
            "n_pre": len(pre), "n_post": len(post),
            "median_pre_cm_km": float(np.median(pre[est])),
            "median_post_cm_km": float(np.median(post[est])),
            "median_drawdown_cm_km": float(np.median(draw[est])) if len(draw) else np.nan,
            "nmad_pre": nmad(pre[est]), "nmad_post": nmad(post[est]),
            "diff_post_minus_pre_cm_km": obs,
            "diff_ci95_low": lo, "diff_ci95_high": hi,
            "permutation_p": p_perm,
            "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
            "pre_positive": f"{k_pre}/{len(pre)}", "pre_sign_test_p": float(sg_pre),
            "post_positive": f"{k_post}/{len(post)}", "post_sign_test_p": float(sg_post),
        })
    res = pd.DataFrame(rows)
    res.to_csv(CFG.TABLES / "kakhovka_transition_statistics.csv", index=False)
    res.to_csv(CFG.FIGDATA / "FigG_transition_statistics.csv", index=False)

    print("\n=== PRE vs POST transition statistics (unit = one date) ===")
    for r in res.itertuples():
        print(f"\n--- {r.estimator} ---")
        print(f"  median slope   PRE {r.median_pre_cm_km:+.3f}   "
              f"DRAWDOWN {r.median_drawdown_cm_km:+.3f}   POST {r.median_post_cm_km:+.3f} cm/km")
        print(f"  NMAD           PRE {r.nmad_pre:.3f}   POST {r.nmad_post:.3f}")
        print(f"  difference     {r.diff_post_minus_pre_cm_km:+.3f} cm/km  "
              f"95% CI [{r.diff_ci95_low:+.3f}, {r.diff_ci95_high:+.3f}]")
        print(f"  permutation p  {r.permutation_p:.5f}   Mann-Whitney p {r.mannwhitney_p:.5f}")
        print(f"  sign test      PRE {r.pre_positive} (p={r.pre_sign_test_p:.3f})   "
              f"POST {r.post_positive} (p={r.post_sign_test_p:.4f})")

    ro = res[res.estimator == "OLS"].iloc[0]
    rt = res[res.estimator == "Theil-Sen"].iloc[0]
    ratio = rt.median_post_cm_km / ro.median_post_cm_km if ro.median_post_cm_km else np.nan
    print(f"\nROBUSTNESS: post-breach median OLS {ro.median_post_cm_km:+.2f} vs "
          f"Theil-Sen {rt.median_post_cm_km:+.2f} cm/km  (ratio {ratio:.2f})")
    print("  -> the transition is" +
          (" ROBUST to the estimator." if 0.5 < ratio < 2.0 else
           " SENSITIVE to outliers/ponds — Theil-Sen must be preferred."))


if __name__ == "__main__":
    main()
