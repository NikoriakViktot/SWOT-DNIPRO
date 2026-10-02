#!/usr/bin/env python
"""MS13 -- the breach fortnight from orbit (Section 4.2, Table 2, Figure 2), one source and one vertical tie.

Why this exists. Figure 2 (F15_swot_drawdown_and_wave.png) was carried from earlier work without a
script, and the absolute levels in Table 2 (17.61 m -> 5.71 m at the outlet, peak 10.31 m 15 km below
the dam) could not be reproduced: the tables they come from (p60 outlet, p59 wave) tie SWOT to EVRF2019
with the SUPERSEDED reservoir corrector c = -0.173 m (computed without the ATL13 permanent-tide term;
Supplementary S1.5), and the text carried a further unexplained offset of about +0.07 m. Differences
(the 11.90 m fall, the 9.10 m rise) do not depend on the tie; absolute levels do.

Here every absolute SWOT level is put in the gauge-anchored frame of the paper, H_S + c, with c the
reservoir closure residual of V1 (ms7_summary.csv, mean of the six station medians, -0.135 m), by
swapping the superseded constant for it: H = H_p60/p59 - c_superseded + c_paper. The Rozumivka gauge is
already EVRF2019 and is not touched.

Inputs   outputs/tables/p60_swot_outlet_drawdown.csv, outputs/tables/p59_swot_peak_wse_profile.csv,
         outputs/tables/egg2015_to_evrf2019_by_station.csv (via CFG.CORRECTOR_BY_STATION),
         outputs/paper/validation/ms7_summary.csv
Outputs  outputs/paper/validation/ms13_outlet_drawdown.csv, ms13_wave_profile.csv, ms13_summary.csv
         outputs/paper/figures/F15_swot_drawdown_and_wave.png / .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swot_dnipro import config as CFG  # noqa: E402

VAL = ROOT / "outputs/paper/validation"
FIG = ROOT / "outputs/paper/figures"
T = ROOT / "outputs/tables"
WINDOW = ("2023-05-28", "2023-06-30")
BLUE, ORANGE, GREY = "#1f6f8b", "#b5651d", "#8a8a8a"


def main() -> None:
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION)
    c_old = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean())          # what p59/p60 used
    S = pd.read_csv(VAL / "ms7_summary.csv")
    c_new = float(S[(S.claim_id == "V1_ATL13_GAUGE_CLOSURE") & (S.statistic == "mean_c_m")].value.iloc[0])
    shift = c_new - c_old

    o = pd.read_csv(T / "p60_swot_outlet_drawdown.csv", parse_dates=["date"])
    for col in ("H_outlet", "H_outlet_min", "H_outlet_max"):
        o[col] = o[col] + shift
    o = o[(o.date >= WINDOW[0]) & (o.date <= WINDOW[1])].reset_index(drop=True)
    o[["date", "swot_n", "H_outlet", "H_outlet_min", "H_outlet_max", "H_rozumivka"]].to_csv(
        VAL / "ms13_outlet_drawdown.csv", index=False)

    w = pd.read_csv(T / "p59_swot_peak_wse_profile.csv")
    for col in ("H_pre", "H_peak"):
        w[col] = w[col] + shift
    w.to_csv(VAL / "ms13_wave_profile.csv", index=False)

    a, b = o[o.date == "2023-05-31"].iloc[0], o[o.date == "2023-06-13"].iloc[0]
    k15 = w[w.bin_km == 15].iloc[0]
    summ = pd.DataFrame([
        dict(statistic="c_superseded_m", value=c_old), dict(statistic="c_paper_m", value=c_new),
        dict(statistic="outlet_2023-05-31_m", value=a.H_outlet, n=int(a.swot_n)),
        dict(statistic="outlet_2023-06-13_m", value=b.H_outlet, n=int(b.swot_n)),
        dict(statistic="outlet_fall_m", value=a.H_outlet - b.H_outlet),
        dict(statistic="rise_15km_m", value=k15.rise_m, n=int(k15.n_peak)),
        dict(statistic="peak_15km_m", value=k15.H_peak, note=str(k15.peak_date)),
        dict(statistic="rise_80km_m", value=float(w[w.bin_km == 80].rise_m.iloc[0]))])
    summ.to_csv(VAL / "ms13_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=200, gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax.fill_between(o.date, o.H_outlet_min, o.H_outlet_max, color=BLUE, alpha=0.18, lw=0)
    ax.plot(o.date, o.H_outlet, "-o", color=BLUE, ms=4, label="SWOT, outlet (node median; band = node range)")
    g = o.dropna(subset=["H_rozumivka"])
    ax.plot(g.date, g.H_rozumivka, "--s", color=ORANGE, ms=4, label="Rozumivka gauge (daily)")
    ax.axvline(pd.Timestamp(CFG.BREACH_DATE), color="0.5", ls=":", lw=1)
    ax.set_ylabel("water surface (m, gauge-anchored frame\ntied to EVRF2019)")
    ax.set_title("a · the pool emptying, outlet vs upstream gauge", loc="left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    x = w.bin_km + 2.5
    bx.plot(x, w.rise_m, "-o", color=BLUE, ms=4, label="peak rise above the pre-breach surface")
    bx.set_xlabel("distance below the dam (km)")
    bx.set_ylabel("rise (m)")
    bx.set_title("b · the wave, decaying downstream", loc="left", fontsize=10)
    bx.annotate(f"{k15.rise_m:.2f} m", (17.5, k15.rise_m), xytext=(6, 2), textcoords="offset points", fontsize=8, color=BLUE)
    bx.set_ylim(bottom=0)
    bx.legend(frameon=False, fontsize=7.5, loc="lower left")
    for z in (ax, bx):
        z.grid(axis="y", color="0.92")
        for s in ("top", "right"):
            z.spines[s].set_visible(False)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F15_swot_drawdown_and_wave.{ext}")
    print(f"c superseded {c_old:+.3f} -> paper {c_new:+.3f} (shift {shift:+.3f} m)")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
