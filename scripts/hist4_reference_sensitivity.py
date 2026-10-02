#!/usr/bin/env python
"""HISTORICAL 4 — formal sensitivity of the bathymetry comparison to the
historical reference level.

Four scenarios, and they are NOT four competing operational datums:

    A  16.00 m  NPG, normal impoundment level   -- the old assumption, REJECTED
    B  14.00 m  UNS, navigation drawdown level  -- the working reference
    C  13.71 m  capacity-curve estimate         -- independent, no ICESat-2
    D  14.11 m  ICESat-2 estimate               -- independent, circular here

B is the operational choice, because it is the only one of the four that is a
PUBLISHED LEVEL rather than an estimate. C and D are independent estimates of
the same quantity and are reported to show the spread; D is circular when used
against ICESat-2 and is flagged as such wherever it appears.

Inference is track-clustered throughout: photons along one ICESat-2 track are
autocorrelated, so 94 points are not 94 independent samples.

Outputs
-------
outputs/tables/historical_reference_sensitivity.csv
outputs/figures/V2_historical_reference_sensitivity.png
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

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
OLD = 16.00
RNG = np.random.default_rng(CFG.SEED)

SCEN = [
    ("A", "NPG 16.00 m\n(old assumption)", 16.00, "assumption", RED,
     "rejected by the data"),
    ("B", "UNS 14.00 m\n(working reference)", 14.00, "Table 19, published", GREEN,
     "preferred: the only PUBLISHED level of the four"),
    ("C", "capacity fit 13.71 m", 13.71, "Table 19 + soundings", BLUE,
     "independent estimate, no ICESat-2"),
    ("D", "ICESat-2 14.11 m", 14.11, "QA5", AMBER,
     "independent estimate, CIRCULAR against ICESat-2"),
]


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def boot(m, col, n=8000):
    u = m.track.unique()
    g = {t: m[col].values[m.track.values == t] for t in u}
    b = np.array([np.median(np.concatenate([g[t] for t in RNG.choice(u, len(u), True)]))
                  for _ in range(n)])
    return float(np.median(m[col])), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5)), len(u)


def main() -> None:
    cls = pd.read_csv(CFG.TABLES / "qa3_s7_classification.csv", parse_dates=["date"])
    s6 = cls[(cls.h_canopy == 0) & (cls.veg_ph_count == 0)
             & (cls.gnd_ph_count >= 50) & (cls.match_dist_m <= 50)]
    s7 = s6[(s6.dry_class == "DRY_EXPOSED_BED") & (s6.date > "2023-06-06")
            & (s6.chainage_km <= 200)].copy()

    print(f"S7, confirmed exposed bed, impounded reach")
    print(f"  {len(s7)} points, {s7.track.nunique()} tracks, "
          f"chainage {s7.chainage_km.min():.0f}-{s7.chainage_km.max():.0f} km, "
          f"{s7.date.min():%b %Y}-{s7.date.max():%b %Y}\n")
    print(f"  {'':2}{'reference level':<34}{'dH':>8}{'95% CI (track)':>20}"
          f"{'NMAD':>7}{'n':>6}{'trk':>5}  note")

    rows, dists = [], {}
    for sid, label, D, src, col, note in SCEN:
        s7["dHc"] = s7.dH + (OLD - D)
        med, lo, hi, ntr = boot(s7, "dHc")
        dists[sid] = s7.dHc.values.copy()
        zero_in = lo <= 0 <= hi
        print(f"  {sid} {label.replace(chr(10),' '):<34}{med:>+8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}{nmad(s7.dHc.values):>7.2f}"
              f"{len(s7):>6}{ntr:>5}  {note}")
        rows.append({"scenario": sid, "reference_level_m": D,
                     "reference_name": label.replace("\n", " "), "source": src,
                     "median_dH_m": med, "ci_lo_m": lo, "ci_hi_m": hi,
                     "nmad_m": nmad(s7.dHc.values), "n_points": len(s7),
                     "n_tracks": ntr, "zero_within_ci": zero_in,
                     "independent_of_icesat2": sid != "D",
                     "role": note})
    out = pd.DataFrame(rows)
    out.to_csv(CFG.TABLES / "historical_reference_sensitivity.csv", index=False)

    a = out[out.scenario == "A"].iloc[0]
    b = out[out.scenario == "B"].iloc[0]
    print(f"\n  A is rejected: {a.median_dH_m:+.2f} m, CI "
          f"[{a.ci_lo_m:+.2f}, {a.ci_hi_m:+.2f}] excludes zero by "
          f"{abs(a.ci_hi_m):.2f} m, i.e. {abs(a.median_dH_m)/a.nmad_m:.1f} NMAD.")
    print(f"  B is preferred: {b.median_dH_m:+.2f} m, zero "
          f"{'INSIDE' if b.zero_within_ci else 'OUTSIDE'} the CI.")
    print(f"  C and D bracket B by {out.reference_level_m[1:].max()-out.reference_level_m[1:].min():.2f} m,")
    print(f"  and that spread -- not the CI above -- is the honest uncertainty on")
    print(f"  the reference level. It is not propagated into any single CI here.")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))

    a_ = ax[0]
    bins = np.arange(-4.0, 3.01, 0.2)
    for sid, label, D, src, col, note in SCEN:
        a_.hist(dists[sid], bins=bins, histtype="step", lw=2.2, color=col,
                label=f"{sid} · {label.replace(chr(10), ' ')}")
    a_.axvline(0, color=INK, lw=1.6, ls="--")
    a_.text(0.02, 0.97, "0 = survey and ICESat-2 agree", transform=a_.transAxes,
            fontsize=8.6, color=INK, va="top")
    a_.set_xlabel("ICESat-2 − survey (m)")
    a_.set_ylabel(f"S7 observations (n={len(s7)})")
    a_.legend(fontsize=8.6, loc="upper right")
    a_.set_title("Residual distribution under each reference level",
                 fontsize=11.5, loc="left")

    a_ = ax[1]
    for i, r in out.iloc[::-1].reset_index(drop=True).iterrows():
        c = dict((s[0], s[4]) for s in SCEN)[r.scenario]
        a_.plot([r.ci_lo_m, r.ci_hi_m], [i, i], color=c, lw=7, alpha=0.45,
                solid_capstyle="butt")
        a_.plot(r.median_dH_m, i, "o", color=c, ms=12)
        a_.annotate(f"{r.median_dH_m:+.2f}", (r.median_dH_m, i),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=9.5, fontweight="bold", color=c)
    a_.axvline(0, color=INK, lw=1.8, ls="--")
    a_.set_yticks(range(len(out)))
    a_.set_yticklabels([f"{r.scenario} · {r.reference_level_m:.2f} m"
                        for _, r in out.iloc[::-1].iterrows()], fontsize=9.5)
    a_.set_xlim(-2.6, 1.2)
    a_.grid(axis="x", alpha=0.3)
    a_.set_xlabel("median ICESat-2 − survey (m), 95 % CI clustered by track")
    a_.set_title("A is rejected; B is the published level", fontsize=11.5, loc="left")

    fig.suptitle("V2 · Sensitivity of the bathymetry comparison to the historical "
                 "reference level   ·   S7 confirmed exposed bed, "
                 f"{len(s7)} points / {s7.track.nunique()} tracks",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V2_historical_reference_sensitivity.{e}",
                    dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V2_historical_reference_sensitivity.png'}")
    print(f"-> {CFG.TABLES/'historical_reference_sensitivity.csv'}")


if __name__ == "__main__":
    main()
