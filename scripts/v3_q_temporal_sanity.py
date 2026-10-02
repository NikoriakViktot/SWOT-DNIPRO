#!/usr/bin/env python
"""V3 -- temporal sanity of the DniproHES Q series: annual/monthly stats,
full hydrograph, format-transition discontinuity check, duplicates/
negatives/spikes.

Outputs
-------
outputs/tables/dniprohes_q_annual_stats.csv
outputs/tables/dniprohes_q_monthly_stats.csv
outputs/figures/dniprohes_q_hydrograph_1980_2023.png
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

from swot_dnipro import config as CFG

PARQUET = CFG.ROOT / "data" / "processed" / "hydrology" / "dniprohes_releases.parquet"


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month

    # duplicates / negatives / non-numeric
    dup = df.date.duplicated().sum()
    neg = (df.discharge_m3s < 0).sum()
    nonnum = df.discharge_m3s.isna().sum()
    print(f"duplicate dates: {dup}")
    print(f"negative Q: {neg}")
    print(f"non-numeric/NaN Q: {nonnum}")

    q1, q3 = df.discharge_m3s.quantile([0.25, 0.75])
    iqr = q3 - q1
    spike_thr = q3 + 5 * iqr
    spikes = df[df.discharge_m3s > spike_thr]
    print(f"implausible spikes (> Q3+5*IQR = {spike_thr:.0f} m3/s): {len(spikes)}")
    if len(spikes):
        print(spikes[["date", "discharge_m3s"]].to_string(index=False))

    ann = df.groupby("year").agg(
        count=("discharge_m3s", "count"), mean=("discharge_m3s", "mean"),
        median=("discharge_m3s", "median"), min=("discharge_m3s", "min"),
        max=("discharge_m3s", "max"), p05=("discharge_m3s", lambda s: s.quantile(0.05)),
        p95=("discharge_m3s", lambda s: s.quantile(0.95)))
    ann.to_csv(CFG.TABLES / "dniprohes_q_annual_stats.csv")
    print(f"\n-> {CFG.TABLES / 'dniprohes_q_annual_stats.csv'}")

    mon = df.groupby(["year", "month"]).agg(
        count=("discharge_m3s", "count"), mean=("discharge_m3s", "mean"),
        median=("discharge_m3s", "median"), min=("discharge_m3s", "min"),
        max=("discharge_m3s", "max"))
    mon.to_csv(CFG.TABLES / "dniprohes_q_monthly_stats.csv")
    print(f"-> {CFG.TABLES / 'dniprohes_q_monthly_stats.csv'}")

    # format-transition discontinuity check: does the ANNUAL mean jump
    # unnaturally across 2018->2019 (Q-A) or 2020->2021 (Q-B)?
    print("\nyear-over-year mean discharge, transition years flagged:")
    ann2 = ann.reset_index()
    ann2["pct_change"] = ann2["mean"].pct_change() * 100
    for r in ann2.itertuples():
        flag = " <-- FORMAT BOUNDARY" if r.year in (2019, 2021) else ""
        print(f"  {r.year}: mean={r.mean:7.1f}  Δ%={r.pct_change:+6.1f}%{flag}")
    # a "discontinuity" would be a Δ% far outside the normal year-to-year
    # variability seen elsewhere in the 44-year record
    normal_spread = ann2["pct_change"].iloc[1:].abs().median()
    at_2019 = abs(ann2.loc[ann2.year == 2019, "pct_change"].iloc[0])
    at_2021 = abs(ann2.loc[ann2.year == 2021, "pct_change"].iloc[0])
    print(f"\nmedian |Δ%| across the whole record: {normal_spread:.1f}%")
    print(f"2019 (Q-A internal transition, clean->block format): {at_2019:.1f}%")
    print(f"2021 (Q-B format start): {at_2021:.1f}%")
    verdict = "NO ARTEFACT" if max(at_2019, at_2021) < 3 * normal_spread else "POSSIBLE ARTEFACT -- INSPECT"
    print(f"VERDICT: {verdict}")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    ax = axes[0]
    ax.plot(df.date, df.discharge_m3s, lw=0.3, color="#236f8c")
    ax.axvline(pd.Timestamp("2019-01-01"), color="#b07d27", ls="--", lw=1,
              label="format A internal transition (clean rows -> block)")
    ax.axvline(pd.Timestamp("2021-01-01"), color="#c1402a", ls="--", lw=1,
              label="format A -> format B (col mapping change)")
    ax.axvline(pd.Timestamp("2023-06-06"), color="black", ls=":", lw=1.2,
              label="Kakhovka dam breach")
    ax.set_ylabel("Q (m3/s)")
    ax.set_title("DniproHES (80039) daily release, 1980-2023 -- full hydrograph", loc="left")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(ann.index, ann["mean"], marker="o", ms=3, color="#3f7d4e", label="annual mean")
    ax.fill_between(ann.index, ann["p05"], ann["p95"], alpha=0.2, color="#3f7d4e",
                    label="p05-p95")
    ax.axvline(2019, color="#b07d27", ls="--", lw=1)
    ax.axvline(2021, color="#c1402a", ls="--", lw=1)
    ax.set_ylabel("Q (m3/s)"); ax.set_xlabel("year")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_fig = CFG.FIG / "dniprohes_q_hydrograph_1980_2023.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\n-> {out_fig}")


if __name__ == "__main__":
    main()
