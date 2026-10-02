#!/usr/bin/env python
"""H4-H6, H8-H11 -- dS/dQ hydraulic analysis, CORRECTLY SCOPED to the data
that actually exists.

CRITICAL, discovered before any modelling (not assumed): DniproHES Q
(data/processed/hydrology/dniprohes_releases.parquet) ends 2023-12-31. The
project's existing headline per-date slope table
(outputs/figure_data/FigG_perdate_slopes_robust.csv, 33 dates) has ALL 14
POST_BREACH dates in 2024-2025 -- ZERO overlap with Q. PRE_BREACH (14/14)
and BREACH_DRAWDOWN (5/5) dates ARE fully within Q coverage.

Consequence: the originally-specified PRE-vs-POST interaction test
(beta3 = Q x POST) is NOT FITTABLE with real post-breach data -- 0 points.
This script substitutes the scientifically next-best, honestly-labelled
comparison: PRE_BREACH (n=14) vs BREACH_DRAWDOWN (n=5) -- the drawdown IS a
genuine non-stationary transient, analysed separately per the operator's
own H3 instruction, and it is the ONLY period with real discharge-response
data available. This is NOT the PRE-vs-POST claim; it is reported as
PRE-vs-DRAWDOWN throughout, never silently relabelled.

Independent unit throughout: ONE ROW PER DATE (already true of the source
table -- no segment/node counting).

Outputs
-------
outputs/tables/slope_Q_by_overpass.csv
outputs/tables/slope_Q_lag_sensitivity.csv
outputs/tables/slope_Q_regression_results.csv
outputs/tables/drawdown_hydraulic_sequence.csv
outputs/figures/Q_vs_slope_PRE.png
outputs/figures/Q_vs_slope_DRAWDOWN.png
outputs/figures/dS_dQ_PRE_vs_DRAWDOWN.png
outputs/figures/lag_sensitivity.png
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
import statsmodels.api as sm
from scipy import stats as sstats

from swot_dnipro import config as CFG

LAGS = range(0, 6)
N_BOOT = 2000
RNG = np.random.default_rng(42)
INK, BLUE, RED, AMBER, GREEN = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e"


def theil_sen(x, y):
    if len(x) < 2:
        return np.nan, (np.nan, np.nan)
    res = sstats.theilslopes(y, x)
    return res.slope, (res.low_slope, res.high_slope)


def main() -> None:
    df = pd.read_csv(CFG.FIGDATA / "FigG_perdate_slopes_robust.csv")
    df["date"] = pd.to_datetime(df.date)
    q = pd.read_parquet(CFG.ROOT / "data/processed/hydrology/dniprohes_releases.parquet")
    q = q.set_index("date")["discharge_m3s"]

    for L in LAGS:
        df[f"Q_lag_{L}d"] = df.date.map(lambda d: q.get(d - pd.Timedelta(days=L), np.nan))

    df["has_Q"] = df[f"Q_lag_0d"].notna()
    n_pre = int((df.period == "PRE_BREACH").sum())
    n_post = int((df.period == "POST_BREACH").sum())
    n_draw = int((df.period == "BREACH_DRAWDOWN").sum())
    n_pre_q = int(((df.period == "PRE_BREACH") & df.has_Q).sum())
    n_post_q = int(((df.period == "POST_BREACH") & df.has_Q).sum())
    n_draw_q = int(((df.period == "BREACH_DRAWDOWN") & df.has_Q).sum())
    print(f"PRE_BREACH: {n_pre} dates total, {n_pre_q} with Q")
    print(f"BREACH_DRAWDOWN: {n_draw} dates total, {n_draw_q} with Q")
    print(f"POST_BREACH: {n_post} dates total, {n_post_q} with Q  <-- ZERO, as discovered")

    work = df[df.period.isin(["PRE_BREACH", "BREACH_DRAWDOWN"]) & df.has_Q].copy()
    work["is_drawdown"] = (work.period == "BREACH_DRAWDOWN").astype(int)
    work.to_csv(CFG.TABLES / "slope_Q_by_overpass.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'slope_Q_by_overpass.csv'} (n={len(work)})")

    # ================================================================ H4
    lag_rows = []
    for L in LAGS:
        col = f"Q_lag_{L}d"
        for period, sub in (("PRE_BREACH", work[work.period == "PRE_BREACH"]),
                            ("BREACH_DRAWDOWN", work[work.period == "BREACH_DRAWDOWN"]),
                            ("COMBINED", work)):
            x, y = sub[col].values, sub.slope_theilsen_cm_km.values
            if len(x) >= 3 and np.std(x) > 0:
                r, p_r = sstats.pearsonr(x, y)
                rho, p_rho = sstats.spearmanr(x, y)
            else:
                r = p_r = rho = p_rho = np.nan
            lag_rows.append({"lag_days": L, "period": period, "n": len(sub),
                            "pearson_r": r, "pearson_p": p_r,
                            "spearman_rho": rho, "spearman_p": p_rho})
    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(CFG.TABLES / "slope_Q_lag_sensitivity.csv", index=False)
    print(f"-> {CFG.TABLES / 'slope_Q_lag_sensitivity.csv'}")
    print(lag_df.to_string(index=False))

    # ================================================================ H5 figures
    for period, sub, fname, color in (
        ("PRE_BREACH", work[work.period == "PRE_BREACH"], "Q_vs_slope_PRE.png", BLUE),
        ("BREACH_DRAWDOWN", work[work.period == "BREACH_DRAWDOWN"], "Q_vs_slope_DRAWDOWN.png", RED)):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub.Q_lag_0d, sub.slope_theilsen_cm_km, color=color, s=60, zorder=5)
        for r in sub.itertuples():
            ax.annotate(r.date.strftime("%Y-%m"), (r.Q_lag_0d, r.slope_theilsen_cm_km),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
        if len(sub) >= 3:
            ts_slope, ts_ci = theil_sen(sub.Q_lag_0d.values, sub.slope_theilsen_cm_km.values)
            xx = np.linspace(sub.Q_lag_0d.min(), sub.Q_lag_0d.max(), 50)
            ax.plot(xx, ts_slope * (xx - sub.Q_lag_0d.median()) +
                    sub.slope_theilsen_cm_km.median(), color=color, ls="--", lw=1.5,
                    label=f"Theil-Sen dS/dQ={ts_slope:.5f} cm/km per m3/s")
        ax.set_xlabel("Q_DniproHES, lag 0d (m3/s)")
        ax.set_ylabel("longitudinal WSE slope (cm/km, Theil-Sen)")
        ax.set_title(f"{period} (n={len(sub)}) -- descriptive, not yet a claim", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(CFG.FIG / fname, dpi=150)
        print(f"-> {CFG.FIG / fname}")

    # ================================================================ H6 regression (OLS, lag_0d primary)
    X = sm.add_constant(work[["Q_lag_0d", "is_drawdown"]].assign(
        Q_x_drawdown=work.Q_lag_0d * work.is_drawdown))
    y = work.slope_theilsen_cm_km.values
    model = sm.OLS(y, X).fit(cov_type="HC3")  # heteroskedasticity-robust, n is tiny
    print("\n" + "=" * 70)
    print("H6 -- OLS: slope ~ Q_lag_0d + is_drawdown + Q_lag_0d:is_drawdown")
    print("(is_drawdown substitutes for is_POST -- POST_BREACH has ZERO Q overlap, see header)")
    print("=" * 70)
    print(model.summary())

    reg_rows = [{"term": t, "coef": model.params[t], "se": model.bse[t],
                "ci_lo": model.conf_int().loc[t, 0], "ci_hi": model.conf_int().loc[t, 1],
                "p": model.pvalues[t]} for t in model.params.index]
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(CFG.TABLES / "slope_Q_regression_results.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'slope_Q_regression_results.csv'}")

    # ================================================================ H9 robustness: bootstrap by DATE
    def fit_interaction(sub_df):
        Xb = sm.add_constant(sub_df[["Q_lag_0d", "is_drawdown"]].assign(
            Q_x_drawdown=sub_df.Q_lag_0d * sub_df.is_drawdown))
        try:
            m = sm.OLS(sub_df.slope_theilsen_cm_km.values, Xb).fit()
            return m.params.get("Q_x_drawdown", np.nan)
        except Exception:
            return np.nan

    boot_coefs = []
    idx = work.index.values
    for _ in range(N_BOOT):
        samp = RNG.choice(idx, size=len(idx), replace=True)
        sub = work.loc[samp]
        if sub.is_drawdown.nunique() < 2 or sub.Q_lag_0d.std() == 0:
            continue
        boot_coefs.append(fit_interaction(sub))
    boot_coefs = np.array([b for b in boot_coefs if np.isfinite(b)])
    boot_ci = np.percentile(boot_coefs, [2.5, 97.5]) if len(boot_coefs) > 50 else (np.nan, np.nan)
    print(f"\nH9 bootstrap (resampled by DATE, n_boot={len(boot_coefs)}): "
          f"Q_x_drawdown coef 95% CI = [{boot_ci[0]:.5f}, {boot_ci[1]:.5f}]")

    # separate Theil-Sen per regime (H9 robustness alternative to OLS)
    ts_pre, ts_pre_ci = theil_sen(work[work.period == "PRE_BREACH"].Q_lag_0d.values,
                                   work[work.period == "PRE_BREACH"].slope_theilsen_cm_km.values)
    ts_draw, ts_draw_ci = theil_sen(work[work.period == "BREACH_DRAWDOWN"].Q_lag_0d.values,
                                     work[work.period == "BREACH_DRAWDOWN"].slope_theilsen_cm_km.values)
    print(f"Theil-Sen dS/dQ, PRE_BREACH: {ts_pre:.5f} (90% CI {ts_pre_ci})")
    print(f"Theil-Sen dS/dQ, BREACH_DRAWDOWN: {ts_draw:.5f} (90% CI {ts_draw_ci})")

    # ================================================================ H10 summary
    print("\n" + "=" * 70)
    print("H10 -- PRE vs DRAWDOWN summary (median slope, IQR, range)")
    print("=" * 70)
    for period in ("PRE_BREACH", "BREACH_DRAWDOWN"):
        sub = work[work.period == period].slope_theilsen_cm_km
        print(f"{period}: median={sub.median():.3f}  IQR=[{sub.quantile(.25):.3f},"
              f"{sub.quantile(.75):.3f}]  range=[{sub.min():.3f},{sub.max():.3f}]  n={len(sub)}")

    # ================================================================ H11 drawdown sequence
    draw = df[df.period == "BREACH_DRAWDOWN"].sort_values("date").copy()
    draw["Q_DniproHES_m3s"] = draw.Q_lag_0d
    draw_out = draw[["date", "Q_DniproHES_m3s", "wse_range_m", "slope_theilsen_cm_km",
                     "slope_ols_cm_km", "span_km", "n_points"]]
    draw_out.to_csv(CFG.TABLES / "drawdown_hydraulic_sequence.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'drawdown_hydraulic_sequence.csv'}")
    print(draw_out.to_string(index=False))
    print("\nNote: operator's suggested window (2023-05-01..2023-07-15) does not match "
          "the actual DRAWDOWN date range in the existing project data "
          "(2023-07-07..2023-09-07, per this project's period-split convention, "
          "breach=2023-06-06, DRAWDOWN ends 2023-09-01). Reported as the data actually is.")

    # ---- lag sensitivity figure ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for period, color in (("PRE_BREACH", BLUE), ("BREACH_DRAWDOWN", RED), ("COMBINED", INK)):
        sub = lag_df[lag_df.period == period]
        ax.plot(sub.lag_days, sub.spearman_rho, marker="o", color=color, label=f"{period} Spearman rho")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Q lag (days)"); ax.set_ylabel("Spearman rho (slope vs Q)")
    ax.set_title("H4 -- lag sensitivity, ALL lags reported (no cherry-picking)", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "lag_sensitivity.png", dpi=150)
    print(f"-> {CFG.FIG / 'lag_sensitivity.png'}")

    # ---- dS/dQ PRE vs DRAWDOWN with CI figure ---------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ["PRE_BREACH", "BREACH_DRAWDOWN"]
    vals = [ts_pre, ts_draw]
    los = [ts_pre_ci[0], ts_draw_ci[0]]
    his = [ts_pre_ci[1], ts_draw_ci[1]]
    ax.errorbar([0, 1], vals, yerr=[[v - l for v, l in zip(vals, los)],
                                    [h - v for v, h in zip(vals, his)]],
               fmt="o", capsize=5, color=INK, ms=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("dS/dQ, Theil-Sen (cm/km per m3/s)")
    ax.set_title("H10 -- dS/dQ, PRE vs DRAWDOWN (90% CI)\n"
                "NOT PRE vs POST -- POST_BREACH has zero Q overlap", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "dS_dQ_PRE_vs_DRAWDOWN.png", dpi=150)
    print(f"-> {CFG.FIG / 'dS_dQ_PRE_vs_DRAWDOWN.png'}")


if __name__ == "__main__":
    main()
