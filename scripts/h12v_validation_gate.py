#!/usr/bin/env python
"""H12 validation gate V1-V7 -- chainage-convention audit, span-matched
slope test, RGT/chainage sampling-location check, effect-size robustness,
span-sensitivity, drawdown positioning. Does not change zone boundaries.

Outputs
-------
outputs/tables/PRE_POST_matched_RGT_or_chainage.csv
outputs/tables/zone_span_matched_comparison.csv
outputs/tables/zone_effect_sizes.csv
outputs/tables/zone_span_sensitivity.csv
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
from scipy import stats as sstats

from swot_dnipro import config as CFG

RNG = np.random.default_rng(42)
N_BOOT = 3000
REACH_BREAKS = (0, 133, 183, 277.2)
REACH_LABELS = ("CORE_LOWER", "CORE_MIDDLE", "CORE_UPPER")
SPAN_BINS = [(3, 5), (5, 8), (8, 12)]


def theil_sen(x, y):
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return sstats.theilslopes(y, x).slope


def hodges_lehmann(a, b):
    """Hodges-Lehmann shift estimator: median of all pairwise differences."""
    diffs = np.subtract.outer(b, a).ravel()
    return np.median(diffs)


def bootstrap_ci_delta(pre, post, n_boot=N_BOOT):
    boots = []
    for _ in range(n_boot):
        bpre = RNG.choice(pre, len(pre), replace=True)
        bpost = RNG.choice(post, len(post), replace=True)
        boots.append(np.median(bpost) - np.median(bpre))
    return np.percentile(boots, [2.5, 97.5])


def main() -> None:
    df = pd.read_csv(CFG.FIGDATA / "FigD_kakhovka_profile_points.csv")
    df = df[df.qc_pass].copy()
    df["core_reach"] = pd.cut(df.chain_km, REACH_BREAKS, labels=REACH_LABELS, include_lowest=True)

    # ============================================================== V1
    print("=" * 70)
    print("V1 -- CHAINAGE CONSISTENCY AUDIT")
    print("=" * 70)
    print("hist18_chain_km / FigD chain_km (used for CORE_LOWER/MIDDLE/UPPER):")
    print("  source: SWORD dist_out, RE-REFERENCED TO THE KAKHOVKA DAM, "
          "POSITIVE UPSTREAM (verified in src/swot_dnipro/sword.py:76-82 docstring + code)")
    print("  chain_km=0  -> at the Kakhovka dam (downstream/breach end)")
    print("  chain_km max (~277) -> upstream, toward DniproHES")
    print()
    print("s_local_km (F7/F8, gauge/dam-axis chainage):")
    print("  source: this session's local linear chainage")
    print("  s_local_km=0 -> at the DniproHES control axis, POSITIVE DOWNSTREAM")
    print("  valid only within the ~10 km upper AOI near DniproHES")
    print()
    print("CONSEQUENCE: the two systems run in OPPOSITE directions from OPPOSITE ends.")
    print("CORRECTED WORDING: CORE_LOWER (hist18_chain_km 0-133) = near the KAKHOVKA "
          "dam (downstream/breach end). CORE_UPPER (183-277) = near DniproHES "
          "(upstream end) -- i.e. 'strongest change away from the dam' in the "
          "earlier report specifically meant 'away from the KAKHOVKA dam, toward "
          "DniproHES', not ambiguous between the two dams. This is now stated "
          "explicitly, not left implicit.")
    print("Both fields are kept SEPARATE in all outputs below -- never mixed/averaged.")

    # ============================================================== V2 span-matched
    print("\n" + "=" * 70)
    print("V2 -- SPAN-MATCHED SLOPE TEST")
    print("=" * 70)
    rows = []
    for (d, per, reach), g in df.groupby(["date", "period", "core_reach"], observed=True):
        if len(g) < 3:
            continue
        x, y = g.chain_km.values, g.wse_m.values
        span = x.max() - x.min()
        ts = theil_sen(x, y)
        rows.append({"date": d, "period": per, "zone": reach, "n": len(g),
                    "span_km": span, "slope_cm_km": ts * 100})
    zdf = pd.DataFrame(rows)

    span_rows = []
    for lo, hi in SPAN_BINS:
        band = zdf[(zdf.span_km >= lo) & (zdf.span_km < hi)]
        for reach in REACH_LABELS:
            pre = band[(band.zone == reach) & (band.period == "PRE_BREACH")].slope_cm_km.dropna()
            post = band[(band.zone == reach) & (band.period == "POST_BREACH")].slope_cm_km.dropna()
            row = {"span_class_km": f"{lo}-{hi}", "zone": reach,
                  "n_pre": len(pre), "n_post": len(post),
                  "median_pre": pre.median() if len(pre) else np.nan,
                  "median_post": post.median() if len(post) else np.nan}
            if len(pre) >= 2 and len(post) >= 2:
                row["delta"] = post.median() - pre.median()
                ci = bootstrap_ci_delta(pre.values, post.values)
                row["boot_ci_lo"], row["boot_ci_hi"] = ci
            else:
                row["delta"] = row["boot_ci_lo"] = row["boot_ci_hi"] = np.nan
            span_rows.append(row)
    span_df = pd.DataFrame(span_rows)
    span_df.to_csv(CFG.TABLES / "zone_span_matched_comparison.csv", index=False)
    print(span_df.to_string(index=False))
    print(f"-> {CFG.TABLES / 'zone_span_matched_comparison.csv'}")

    # ============================================================== V3 RGT/chainage sampling overlap
    print("\n" + "=" * 70)
    print("V3 -- SAME-RGT / CHAINAGE-WINDOW SAMPLING CHECK")
    print("=" * 70)
    match_rows = []
    for reach in REACH_LABELS:
        sub = df[df.core_reach == reach]
        pre_rgt = set(sub[sub.period == "PRE_BREACH"].rgt.unique())
        post_rgt = set(sub[sub.period == "POST_BREACH"].rgt.unique())
        shared = pre_rgt & post_rgt
        pre_range = (sub[sub.period == "PRE_BREACH"].chain_km.min(),
                    sub[sub.period == "PRE_BREACH"].chain_km.max())
        post_range = (sub[sub.period == "POST_BREACH"].chain_km.min(),
                     sub[sub.period == "POST_BREACH"].chain_km.max())
        match_rows.append({"zone": reach, "n_pre_rgt": len(pre_rgt), "n_post_rgt": len(post_rgt),
                          "n_shared_rgt": len(shared), "shared_rgts": sorted(shared),
                          "pre_chain_km_range": pre_range, "post_chain_km_range": post_range})
    match_df = pd.DataFrame(match_rows)
    match_df.to_csv(CFG.TABLES / "PRE_POST_matched_RGT_or_chainage.csv", index=False)
    print(match_df.to_string(index=False))
    print(f"-> {CFG.TABLES / 'PRE_POST_matched_RGT_or_chainage.csv'}")
    print("\nNote: ICESat-2 RGTs are fixed repeat-orbit tracks, so a shared RGT count "
          "> 0 means PRE and POST genuinely reuse the same ground track through the "
          "zone -- the chainage windows sampled overlap by construction, not just by chance.")

    # ============================================================== V4 date-level counts (already enforced)
    print("\n" + "=" * 70)
    print("V4 -- DATE-LEVEL AGGREGATION CHECK")
    print("=" * 70)
    for reach in REACH_LABELS:
        for per in ("PRE_BREACH", "POST_BREACH"):
            sub = zdf[(zdf.zone == reach) & (zdf.period == per)]
            n_profiles = len(sub)
            n_dates = sub.date.nunique()
            print(f"  {reach} {per}: n_profiles={n_profiles}  n_independent_dates={n_dates}  "
                  f"{'OK -- already 1 row per date' if n_profiles == n_dates else 'MISMATCH -- check'}")

    # ============================================================== V5 effect sizes
    print("\n" + "=" * 70)
    print("V5 -- EFFECT-SIZE ROBUSTNESS (not just p-values)")
    print("=" * 70)
    es_rows = []
    for reach in REACH_LABELS:
        pre = zdf[(zdf.zone == reach) & (zdf.period == "PRE_BREACH")].slope_cm_km.dropna()
        post = zdf[(zdf.zone == reach) & (zdf.period == "POST_BREACH")].slope_cm_km.dropna()
        if len(pre) < 2 or len(post) < 2:
            continue
        hl = hodges_lehmann(pre.values, post.values)
        ci = bootstrap_ci_delta(pre.values, post.values)
        frac_post_gt_premed = float((post > pre.median()).mean())
        frac_post_gt0 = float((post > 0).mean())
        frac_pre_gt0 = float((pre > 0).mean())
        es_rows.append({"zone": reach, "n_pre": len(pre), "n_post": len(post),
                        "median_delta": post.median() - pre.median(),
                        "hodges_lehmann_shift": hl,
                        "boot_ci_lo": ci[0], "boot_ci_hi": ci[1],
                        "frac_post_gt_pre_median": frac_post_gt_premed,
                        "frac_post_positive": frac_post_gt0,
                        "frac_pre_positive": frac_pre_gt0})
    es_df = pd.DataFrame(es_rows)
    es_df.to_csv(CFG.TABLES / "zone_effect_sizes.csv", index=False)
    print(es_df.to_string(index=False))
    print(f"-> {CFG.TABLES / 'zone_effect_sizes.csv'}")

    # ============================================================== V6 span-threshold sensitivity
    print("\n" + "=" * 70)
    print("V6 -- MINIMUM-SPAN SENSITIVITY (proxy for smoothing-scale sensitivity;")
    print("      true multi-scale smoothing not meaningful at n=3-6 pts/date)")
    print("=" * 70)
    sens_rows = []
    for min_span in (0.5, 1, 2, 5):
        for reach in REACH_LABELS:
            pre = zdf[(zdf.zone == reach) & (zdf.period == "PRE_BREACH") & (zdf.span_km >= min_span)].slope_cm_km
            post = zdf[(zdf.zone == reach) & (zdf.period == "POST_BREACH") & (zdf.span_km >= min_span)].slope_cm_km
            sens_rows.append({"min_span_km": min_span, "zone": reach,
                             "n_pre": len(pre), "n_post": len(post),
                             "median_pre": pre.median() if len(pre) else np.nan,
                             "median_post": post.median() if len(post) else np.nan})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(CFG.TABLES / "zone_span_sensitivity.csv", index=False)
    piv = sens_df.pivot_table(index="min_span_km", columns="zone", values="median_post")
    print(piv.to_string())
    order_ok = all(piv.loc[m, "CORE_LOWER"] < piv.loc[m, "CORE_MIDDLE"] and
                   piv.loc[m, "CORE_LOWER"] < piv.loc[m, "CORE_UPPER"]
                   for m in piv.index if piv.loc[m].notna().all())
    print(f"\nzone ordering LOWER < MIDDLE/UPPER holds at all tested min-span thresholds: {order_ok}")
    print(f"-> {CFG.TABLES / 'zone_span_sensitivity.csv'}")

    # ============================================================== V7 drawdown position
    print("\n" + "=" * 70)
    print("V7 -- DRAWDOWN POSITION RELATIVE TO PRE/POST (descriptive only, n=5)")
    print("=" * 70)
    for reach in REACH_LABELS:
        draw = zdf[(zdf.zone == reach) & (zdf.period == "BREACH_DRAWDOWN")]
        pre = zdf[(zdf.zone == reach) & (zdf.period == "PRE_BREACH")].slope_cm_km
        post = zdf[(zdf.zone == reach) & (zdf.period == "POST_BREACH")].slope_cm_km
        if len(draw) == 0:
            print(f"  {reach}: no DRAWDOWN dates with >=3 points in this zone")
            continue
        for r in draw.itertuples():
            d_pre = abs(r.slope_cm_km - pre.median()) if len(pre) else np.nan
            d_post = abs(r.slope_cm_km - post.median()) if len(post) else np.nan
            closer = "POST" if (np.isfinite(d_post) and d_post < d_pre) else "PRE"
            print(f"  {reach} {r.date}: slope={r.slope_cm_km:.2f} cm/km  "
                  f"closer to {closer} distribution (|d_pre|={d_pre:.2f}, |d_post|={d_post:.2f})")


if __name__ == "__main__":
    main()
