#!/usr/bin/env python
"""H12 -- longitudinal profile shape (H(s), dH/ds, d2H/ds2) and
zone-stratified slopes.

DATA REALITY CHECK (done before any modelling, not assumed): a spatial
join of all 1680 ATL13 profile points (FigD_kakhovka_profile_points.csv)
against hydraulic_longitudinal_domain.gpkg found ZERO points inside
DNIPROHES_CONTROL_ZONE or KAKHOVKA_UPPER_BACKWATER_CANDIDATE -- the
existing longitudinal-profile dataset never samples that far upstream
(max lat = 47.861, both upper zones sit beyond that). Zone-stratification
is therefore only meaningful WITHIN KAKHOVKA_RESERVOIR_CORE, split into
CORE_UPPER/CORE_MIDDLE/CORE_LOWER using this project's own existing reach
breakpoints (hist18_error_decomposition.csv: reach 1+2 0-133km, reach 3
133-183km, reach 4 183-250km) -- reused, not reinvented.

Independent unit: ONE OVERPASS = one (date, rgt, beam) transect row,
already true of the source table. Date-level aggregation used for
regime-level bootstrap/permutation tests per H12.7.

Outputs
-------
outputs/tables/profile_metrics_by_overpass.csv
outputs/tables/zone_slopes_by_overpass.csv
outputs/tables/zone_PRE_POST_comparison.csv
outputs/tables/chainage_binned_PRE_POST.csv
outputs/figures/H12_regime_median_Hs.png
outputs/figures/H12_regime_median_slope_Ss.png
outputs/figures/H12_regime_curvature_Cs.png
outputs/figures/H12_zone_slopes_PRE_POST.png
outputs/figures/H12_delta_slope_by_chainage.png
outputs/figures/H12_transition_sequence_2023.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from scipy import stats as sstats
from shapely.geometry import Point

from swot_dnipro import config as CFG

RNG = np.random.default_rng(42)
N_BOOT = 3000
BIN_KM = 10.0
REACH_BREAKS = (0, 133, 183, 277.2)  # reused from hist18_error_decomposition.csv
REACH_LABELS = ("CORE_LOWER", "CORE_MIDDLE", "CORE_UPPER")  # 0-133 nearest dam = LOWER
INK, BLUE, RED, AMBER, GREEN = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e"


def theil_sen(x, y):
    if len(x) < 2 or np.std(x) == 0:
        return np.nan, (np.nan, np.nan)
    res = sstats.theilslopes(y, x)
    return res.slope, (res.low_slope, res.high_slope)


def main() -> None:
    df = pd.read_csv(CFG.FIGDATA / "FigD_kakhovka_profile_points.csv")
    df = df[df.qc_pass].copy()  # H12.4 quality gate: use the project's own existing QC flag
    print(f"qc_pass points: {len(df)} / (pre-filter total known: 1680)")

    # ---- zone spatial join (H12.1 context) ----------------------------------
    hyd = gpd.read_file(CFG.ROOT / "data/processed/study_domain/hydraulic_longitudinal_domain.gpkg")
    TF = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    xs, ys = TF.transform(df.lon_mean.values, df.lat_mean.values)
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:32636")
    zone_hyd = pd.Series(["OUTSIDE_ALL"] * len(df), index=df.index)
    for _, row in hyd.iterrows():
        inside = pts.within(row.geometry).values
        zone_hyd[inside] = row.zone_name
    df["hydraulic_zone"] = zone_hyd.values
    print("hydraulic_zone membership (post-QC):")
    print(df.hydraulic_zone.value_counts().to_string())

    # ---- reach subdivision within CORE (H12.6 fallback, documented) --------
    df["core_reach"] = pd.cut(df.chain_km, REACH_BREAKS, labels=REACH_LABELS, include_lowest=True)

    df.to_csv(CFG.TABLES / "profile_metrics_by_overpass.csv", index=False)
    print(f"-> {CFG.TABLES / 'profile_metrics_by_overpass.csv'} (per-overpass point table)")

    # ================================================================ H12.13 per-date metrics
    date_rows = []
    for (d, per), g in df.groupby(["date", "period"]):
        if len(g) < 3:
            continue
        x, y = g.chain_km.values, g.wse_m.values
        ts, ts_ci = theil_sen(x, y)
        ols = np.polyfit(x, y, 1)[0] if len(x) >= 2 else np.nan
        date_rows.append({
            "date": d, "period": per, "n_points": len(g),
            "span_km": x.max() - x.min(), "wse_range_m": y.max() - y.min(),
            "wse_p95_p05_m": np.percentile(y, 95) - np.percentile(y, 5),
            "slope_theilsen_cm_km": ts * 100, "slope_ols_cm_km": ols * 100,
            "ts_ci_lo_cm_km": ts_ci[0] * 100, "ts_ci_hi_cm_km": ts_ci[1] * 100,
        })
    date_df = pd.DataFrame(date_rows)
    print(f"\nper-date profiles (n_points>=3): PRE={  (date_df.period=='PRE_BREACH').sum()}, "
          f"DRAWDOWN={(date_df.period=='BREACH_DRAWDOWN').sum()}, "
          f"POST={(date_df.period=='POST_BREACH').sum()}")

    # ================================================================ H12.6/H12.8 zone x date slopes
    zone_rows = []
    for (d, per, reach), g in df.groupby(["date", "period", "core_reach"], observed=True):
        if len(g) < 3:
            continue
        x, y = g.chain_km.values, g.wse_m.values
        ts, ts_ci = theil_sen(x, y)
        zone_rows.append({"date": d, "period": per, "zone": reach, "n": len(g),
                          "span_km": x.max() - x.min(),
                          "slope_theilsen_cm_km": ts * 100})
    zone_df = pd.DataFrame(zone_rows)
    zone_df.to_csv(CFG.TABLES / "zone_slopes_by_overpass.csv", index=False)
    print(f"-> {CFG.TABLES / 'zone_slopes_by_overpass.csv'} ({len(zone_df)} date x reach rows)")

    # PRE vs POST comparison per reach, bootstrap by DATE + permutation
    comp_rows = []
    for reach in REACH_LABELS:
        pre = zone_df[(zone_df.zone == reach) & (zone_df.period == "PRE_BREACH")].slope_theilsen_cm_km.dropna()
        post = zone_df[(zone_df.zone == reach) & (zone_df.period == "POST_BREACH")].slope_theilsen_cm_km.dropna()
        if len(pre) < 3 or len(post) < 3:
            comp_rows.append({"zone": reach, "n_pre": len(pre), "n_post": len(post),
                             "median_pre": pre.median() if len(pre) else np.nan,
                             "median_post": post.median() if len(post) else np.nan,
                             "delta_median": np.nan, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan,
                             "perm_p": np.nan, "mannwhitney_p": np.nan,
                             "frac_positive_pre": np.nan, "frac_positive_post": np.nan})
            continue
        delta_obs = post.median() - pre.median()
        boots = []
        for _ in range(N_BOOT):
            bpre = RNG.choice(pre.values, len(pre), replace=True)
            bpost = RNG.choice(post.values, len(post), replace=True)
            boots.append(np.median(bpost) - np.median(bpre))
        boot_ci = np.percentile(boots, [2.5, 97.5])
        combined = np.concatenate([pre.values, post.values])
        n1 = len(pre)
        perm_deltas = []
        for _ in range(2000):
            perm = RNG.permutation(combined)
            perm_deltas.append(np.median(perm[n1:]) - np.median(perm[:n1]))
        perm_p = float((np.abs(perm_deltas) >= abs(delta_obs)).mean())
        mw_p = sstats.mannwhitneyu(post, pre, alternative="two-sided").pvalue
        comp_rows.append({"zone": reach, "n_pre": len(pre), "n_post": len(post),
                         "median_pre": pre.median(), "median_post": post.median(),
                         "delta_median": delta_obs, "boot_ci_lo": boot_ci[0], "boot_ci_hi": boot_ci[1],
                         "perm_p": perm_p, "mannwhitney_p": mw_p,
                         "frac_positive_pre": (pre > 0).mean(), "frac_positive_post": (post > 0).mean()})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(CFG.TABLES / "zone_PRE_POST_comparison.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'zone_PRE_POST_comparison.csv'}")
    print(comp_df.to_string(index=False))

    # ================================================================ H12.14 chainage-binned PRE/POST
    bins = np.arange(0, df.chain_km.max() + BIN_KM, BIN_KM)
    df["chain_bin"] = pd.cut(df.chain_km, bins)
    bin_rows = []
    for b, g in df.groupby("chain_bin", observed=True):
        row = {"bin_lo": b.left, "bin_hi": b.right}
        for per in ("PRE_BREACH", "POST_BREACH"):
            sub = g[g.period == per]
            row[f"H_median_{per}"] = sub.wse_m.median() if len(sub) else np.nan
            row[f"n_{per}"] = len(sub)
        bin_rows.append(row)
    bin_df = pd.DataFrame(bin_rows).sort_values("bin_lo")
    bin_df["delta_H"] = bin_df.H_median_POST_BREACH - bin_df.H_median_PRE_BREACH
    # local slope from consecutive bin medians (finite difference, H12.3-style "binned smoothing")
    bin_df["s_mid"] = (bin_df.bin_lo + bin_df.bin_hi) / 2
    for per in ("PRE_BREACH", "POST_BREACH"):
        h = bin_df[f"H_median_{per}"].values
        s = bin_df.s_mid.values
        slope = np.gradient(h, s) * 100  # cm/km
        curv = np.gradient(slope, s)     # cm/km per km
        bin_df[f"S_{per}_cm_km"] = slope
        bin_df[f"C_{per}"] = curv
    bin_df.to_csv(CFG.TABLES / "chainage_binned_PRE_POST.csv", index=False)
    print(f"-> {CFG.TABLES / 'chainage_binned_PRE_POST.csv'} ({BIN_KM:.0f} km bins)")

    # ================================================================ figures
    fig, ax = plt.subplots(figsize=(11, 6))
    for per, color in (("PRE_BREACH", BLUE), ("POST_BREACH", RED), ("BREACH_DRAWDOWN", AMBER)):
        sub = bin_df if per != "BREACH_DRAWDOWN" else None
        if per == "BREACH_DRAWDOWN":
            continue
        ax.plot(bin_df.s_mid, bin_df[f"H_median_{per}"], color=color, marker="o", ms=4, label=per)
    ax.set_xlabel("chainage from dam, km"); ax.set_ylabel("median WSE, m EVRF2019")
    ax.set_title(f"H12 -- regime median H(s), {BIN_KM:.0f} km bins", loc="left", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CFG.FIG / "H12_regime_median_Hs.png", dpi=150)
    print(f"-> {CFG.FIG / 'H12_regime_median_Hs.png'}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for per, color in (("PRE_BREACH", BLUE), ("POST_BREACH", RED)):
        ax.plot(bin_df.s_mid, bin_df[f"S_{per}_cm_km"], color=color, marker="o", ms=4, label=per)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("chainage from dam, km"); ax.set_ylabel("local S(s) = dH/ds, cm/km")
    ax.set_title("H12 -- regime median S(s)", loc="left", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CFG.FIG / "H12_regime_median_slope_Ss.png", dpi=150)
    print(f"-> {CFG.FIG / 'H12_regime_median_slope_Ss.png'}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for per, color in (("PRE_BREACH", BLUE), ("POST_BREACH", RED)):
        ax.plot(bin_df.s_mid, bin_df[f"C_{per}"], color=color, marker="o", ms=4, label=per)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("chainage from dam, km"); ax.set_ylabel("C(s) = d2H/ds2, cm/km per km")
    ax.set_title("H12 -- regime median curvature C(s) -- SINGLE realisation, "
                "repeatability NOT yet tested across individual dates", loc="left", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CFG.FIG / "H12_regime_curvature_Cs.png", dpi=150)
    print(f"-> {CFG.FIG / 'H12_regime_curvature_Cs.png'}")

    fig, ax = plt.subplots(figsize=(8, 6))
    xpos = np.arange(len(REACH_LABELS))
    for i, per, color in ((0, "median_pre", BLUE), (1, "median_post", RED)):
        vals = [comp_df[comp_df.zone == z][per].iloc[0] if (comp_df.zone == z).any() else np.nan
               for z in REACH_LABELS]
        ax.bar(xpos + (i - 0.5) * 0.35, vals, width=0.35, color=color,
              label="PRE_BREACH" if per == "median_pre" else "POST_BREACH")
    ax.set_xticks(xpos); ax.set_xticklabels(REACH_LABELS, rotation=15)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("median Theil-Sen slope, cm/km")
    ax.set_title("H12 -- zone-stratified slope, PRE vs POST\n"
                "(DNIPROHES_CONTROL_ZONE / UPPER_BACKWATER_CANDIDATE: n=0, not shown -- see report)",
                loc="left", fontsize=9)
    ax.legend()
    fig.tight_layout(); fig.savefig(CFG.FIG / "H12_zone_slopes_PRE_POST.png", dpi=150)
    print(f"-> {CFG.FIG / 'H12_zone_slopes_PRE_POST.png'}")

    fig, ax = plt.subplots(figsize=(11, 6))
    dS = bin_df.S_POST_BREACH_cm_km - bin_df.S_PRE_BREACH_cm_km
    ax.bar(bin_df.s_mid, dS, width=BIN_KM * 0.9, color=INK, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("chainage from dam, km"); ax.set_ylabel("Delta S(s) = S_POST - S_PRE, cm/km")
    ax.set_title(f"H12 -- spatial location of slope change, {BIN_KM:.0f} km bins", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(CFG.FIG / "H12_delta_slope_by_chainage.png", dpi=150)
    print(f"-> {CFG.FIG / 'H12_delta_slope_by_chainage.png'}")

    # drawdown sequence
    draw = df[df.period == "BREACH_DRAWDOWN"].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.plasma
    dates = sorted(draw.date.unique())
    for i, d in enumerate(dates):
        g = draw[draw.date == d].sort_values("chain_km")
        ax.plot(g.chain_km, g.wse_m, marker="o", ms=5, color=cmap(i / max(len(dates) - 1, 1)),
               label=d)
    ax.set_xlabel("chainage from dam, km"); ax.set_ylabel("WSE, m EVRF2019")
    ax.set_title("H12.12 -- DRAWDOWN sequence H(s), 5 dates", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CFG.FIG / "H12_transition_sequence_2023.png", dpi=150)
    print(f"-> {CFG.FIG / 'H12_transition_sequence_2023.png'}")

    # ---- H12.10/11 quantified expectation checks ----------------------------
    print("\n" + "=" * 70)
    print("H12.10/11 -- PRE near-flat vs POST river-like, quantified")
    print("=" * 70)
    for per in ("PRE_BREACH", "POST_BREACH"):
        sub = date_df[date_df.period == per]
        print(f"{per}: median|slope|={sub.slope_theilsen_cm_km.abs().median():.3f} cm/km  "
              f"IQR of slope=[{sub.slope_theilsen_cm_km.quantile(.25):.3f},"
              f"{sub.slope_theilsen_cm_km.quantile(.75):.3f}]  "
              f"median wse_range={sub.wse_range_m.median():.3f} m  n={len(sub)}")


if __name__ == "__main__":
    main()
