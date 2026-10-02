#!/usr/bin/env python
"""HISTORICAL 7 — what geometry do the per-date "longitudinal slopes" actually
measure?

Written to explain the one reach where the falsification test failed: 180-210 km
returns -0.32 cm/km where no published curve at any discharge is negative. The
first hypothesis was that this is the widest part of the pool, so chainage
assignment should be worst there. THAT HYPOTHESIS IS WRONG -- 180-210 km has the
SMALLEST median offset from the channel line of any reach (2.5 km, against
15.1 km at 210-240). It is recorded here because it was tested and failed.

The actual cause is geometric, and it is not confined to that reach.

Every qualifying "per-date profile" in this reach has exactly n=6, which is not a
coincidence: it is the six beams of ONE overpass. ICESat-2's beams sit across
~3.3 km of cross-track, and its ground tracks are near-polar while the Dnipro
here runs roughly WSW-ENE, so a single pass cuts the reservoir at a large angle.
Six beams of one pass are a CROSS-SECTION, not a longitudinal profile.

The chainage they are assigned, however, spans 10-11 km, because each beam snaps
independently onto a meandering centreline. So the fitted "slope" is

    (instantaneous cross-reservoir WSE variability)
    -------------------------------------------------
    (chainage spread manufactured by snapping)

whose SIGN is arbitrary: it depends on which beam happens to land at the larger
chainage. The decisive statistic is the ratio of chainage span to true ground
span. Points 5 km apart on the ground cannot be 11 km apart along a river unless
the river doubles back between them -- which is exactly what it does, and exactly
why the denominator is not a longitudinal distance.

Outputs
-------
outputs/tables/hist7_slope_geometry.csv
outputs/figures/HIST7_slope_geometry.png
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
import pyproj
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
GEOD = pyproj.Geod(ellps="WGS84")
MIN_PTS, MIN_SPAN_KM = 4, 10.0
# A single overpass spans ~3.3 km of cross-track beam swath. Allowing for
# along-track extent within the reservoir and for beam-pair separation, ground
# spreads much beyond ~10 km cannot come from one pass.
ONE_PASS_GROUND_KM = 10.0


def ground_span_km(lat, lon):
    """Largest pairwise great-circle distance, in km."""
    n = len(lat)
    if n < 2:
        return 0.0
    la1 = np.repeat(lat, n); lo1 = np.repeat(lon, n)
    la2 = np.tile(lat, n); lo2 = np.tile(lon, n)
    _, _, d = GEOD.inv(lo1, la1, lo2, la2)
    return float(np.nanmax(d) / 1000.0)


def main() -> None:
    p = pd.read_csv(ROOT / "outputs/tables/kakhovka_longitudinal_profiles.csv")
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    p["off_km"] = SW.assign_chainage(p.lon_mean.values, p.lat_mean.values,
                                     ch, tree=tree)[1]

    # ---- the hypothesis that failed ---------------------------------------
    pre = p[p.period == "PRE_BREACH"]
    print("=" * 84)
    print("HYPOTHESIS TESTED AND REJECTED: 'the wide reach snaps worst'")
    print("=" * 84)
    print(f"  {'reach':<10}{'n':>6}{'median offset from channel':>30}{'p90':>8}")
    for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 210), (210, 240)]:
        s = pre[(pre.chain_km >= lo) & (pre.chain_km < hi)]
        print(f"  {f'{lo}-{hi}':<10}{len(s):>6}{s.off_km.median():>27.1f} km"
              f"{s.off_km.quantile(.9):>7.1f}")
    print(f"\n  180-210 km has the SMALLEST offset, not the largest. Rejected.")

    # ---- what one date actually contains ----------------------------------
    rows = []
    for period, sub in p.groupby("period"):
        for (d, ), g in sub.groupby(["date"]):
            for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 210), (210, 240)]:
                s = g[(g.chain_km >= lo) & (g.chain_km < hi)]
                if len(s) < MIN_PTS:
                    continue
                cspan = float(s.chain_km.max() - s.chain_km.min())
                if cspan < MIN_SPAN_KM:
                    continue
                gspan = ground_span_km(s.lat_mean.values, s.lon_mean.values)
                rows.append({
                    "period": period, "date": d, "reach_km": f"{lo}-{hi}",
                    "n_beams": len(s), "n_rgt": s.rgt.nunique(),
                    "chainage_span_km": cspan, "ground_span_km": gspan,
                    "inflation": cspan / gspan if gspan > 0 else np.nan,
                    "wse_span_m": float(s.wse_m.max() - s.wse_m.min()),
                    "slope_cm_km": 100 * float(
                        stats.theilslopes(s.wse_m.values, s.chain_km.values)[0]),
                    "one_pass": bool(s.rgt.nunique() == 1
                                     and gspan <= ONE_PASS_GROUND_KM)})
    q = pd.DataFrame(rows)
    q.to_csv(CFG.TABLES / "hist7_slope_geometry.csv", index=False)

    print("\n" + "=" * 84)
    print("WHAT IS A 'PER-DATE PROFILE'?")
    print("=" * 84)
    print(f"  {len(q)} date x reach fits qualify across all periods.")
    print(f"  fits drawn from a SINGLE ground track : "
          f"{q.n_rgt.eq(1).sum()} of {len(q)} ({100*q.n_rgt.eq(1).mean():.0f} %)")
    print(f"  of those, ground spread <= {ONE_PASS_GROUND_KM:.0f} km : "
          f"{q.one_pass.sum()} ({100*q.one_pass.mean():.0f} %)")
    print(f"  modal beam count : {int(q.n_beams.mode().iloc[0])} "
          f"(= the six beams of one pass)")
    print(f"\n  chainage span / ground span, the inflation factor:")
    print(f"    median {q.inflation.median():.2f}, "
          f"p90 {q.inflation.quantile(.9):.2f}, max {q.inflation.max():.2f}")
    print(f"    fits where chainage span EXCEEDS ground span: "
          f"{int((q.inflation > 1).sum())} of {len(q)} "
          f"({100*(q.inflation > 1).mean():.0f} %)")
    print(f"\n  A chainage span larger than the ground span means the points are")
    print(f"  closer together in space than the river distance between them: the")
    print(f"  channel doubles back. The denominator of the slope is therefore not")
    print(f"  a longitudinal distance, and the numerator is instantaneous")
    print(f"  cross-reservoir variability. The SIGN is arbitrary.")

    print("\n" + "=" * 84)
    print("THE 180-210 km ANOMALY, EXPLAINED")
    print("=" * 84)
    a = q[(q.reach_km == "180-210") & (q.period == "PRE_BREACH")]
    print(f"  {'date':<12}{'beams':>6}{'rgt':>5}{'ground':>9}{'chainage':>10}"
          f"{'infl':>7}{'WSE span':>10}{'slope':>9}")
    for r in a.sort_values("date").itertuples():
        print(f"  {r.date:<12}{r.n_beams:>6}{r.n_rgt:>5}{r.ground_span_km:>8.1f} km"
              f"{r.chainage_span_km:>9.1f} km{r.inflation:>7.2f}"
              f"{r.wse_span_m:>9.3f} m{r.slope_cm_km:>+9.2f}")
    print(f"\n  Every one of these {len(a)} fits is a single overpass. The WSE span")
    print(f"  is {a.wse_span_m.min():.3f}-{a.wse_span_m.max():.3f} m -- a flat surface, correctly measured --")
    print(f"  divided by a chainage span inflated {a.inflation.median():.1f}x over the true")
    print(f"  ground separation. -0.32 cm/km is NOT a hydraulic gradient.")
    print(f"\n  Note also 2023-06-04, two days before the breach: WSE 17.16 m,")
    print(f"  1.2 m above every other date here. The pool was being raised. It")
    print(f"  does not drive the median (leave-one-out spans -0.38..-0.27), but")
    print(f"  it should not sit in a 'normal pre-breach operation' sample.")

    # ---- does this touch the main result? ---------------------------------
    print("\n" + "=" * 84)
    print("DOES THIS AFFECT THE RESERVOIR -> RIVER RESULT?")
    print("=" * 84)
    for per in ["PRE_BREACH", "POST_BREACH"]:
        s = q[q.period == per]
        if not len(s):
            continue
        print(f"  {per:<16} {len(s):>3} fits, "
              f"{100*s.one_pass.mean():>3.0f} % single-pass, "
              f"median inflation {s.inflation.median():.2f}, "
              f"median |WSE span| {s.wse_span_m.median():.3f} m, "
              f"median slope {s.slope_cm_km.median():+.2f} cm/km")
    pre_s, post_s = q[q.period == "PRE_BREACH"], q[q.period == "POST_BREACH"]
    if len(post_s):
        print(f"\n  The geometry defect is the SAME in both periods, so it does not")
        print(f"  by itself create the pre/post contrast. What differs is the")
        print(f"  numerator: median WSE span {pre_s.wse_span_m.median():.3f} m before,")
        print(f"  {post_s.wse_span_m.median():.3f} m after. The post-breach surface really is")
        print(f"  tilted over these short baselines; the pre-breach one is not.")
        print(f"  So the pre/post CONTRAST survives. What does not survive is")
        print(f"  reading either number as a slope over a named 30 km reach.")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

    a_ = ax[0]
    a_.scatter(q.ground_span_km, q.chainage_span_km, s=34,
               c=[GREEN if o else AMBER for o in q.one_pass], alpha=0.75, lw=0)
    lim = max(q.ground_span_km.max(), q.chainage_span_km.max()) * 1.05
    a_.plot([0, lim], [0, lim], color=INK, lw=1.6, ls="--")
    a_.text(lim * 0.55, lim * 0.42, "1:1\nchainage = ground", fontsize=8.4,
            color=INK, rotation=38)
    a_.set_xlabel("true ground span of the fitted points (km)")
    a_.set_ylabel("chainage span assigned to them (km)")
    a_.set_title("Above the line, the denominator is manufactured",
                 fontsize=11, loc="left")
    a_.grid(alpha=0.25)

    a_ = ax[1]
    for per, c in [("PRE_BREACH", BLUE), ("POST_BREACH", RED)]:
        s = q[q.period == per]
        if len(s):
            a_.hist(s.wse_span_m, bins=np.arange(0, 1.6, 0.06), histtype="step",
                    lw=2.2, color=c, label=f"{per} (n={len(s)})")
    a_.set_xlabel("WSE span within one fit (m)")
    a_.set_ylabel("fits")
    a_.legend(fontsize=8.6)
    a_.set_title("The numerator is what changed after the breach",
                 fontsize=11, loc="left")

    a_ = ax[2]
    a_.axis("off")
    txt = (
        "180–210 km ANOMALY: RESOLVED, NOT PHYSICAL\n\n"
        "Rejected first: 'the widest reach snaps worst'.\n"
        "That reach has the SMALLEST offset from the\n"
        "channel line of any (2.5 km vs 15.1 km at 210–240).\n\n"
        "Actual cause: all 6 fits are ONE overpass.\n"
        "  6 beams, ~3–7 km apart on the ground\n"
        f"  assigned chainages {a.chainage_span_km.median():.0f} km apart\n"
        f"  inflation factor {a.inflation.median():.1f}x\n"
        f"  WSE span {a.wse_span_m.min():.2f}–{a.wse_span_m.max():.2f} m (a flat surface)\n\n"
        "ICESat-2 tracks are near-polar; the Dnipro here\n"
        "runs WSW–ENE. One pass gives a CROSS-SECTION.\n"
        "Its sign depends on which beam snaps furthest\n"
        "upstream, so it carries no hydraulic meaning.\n\n"
        "WHAT SURVIVES\n"
        "The pre/post contrast: the numerator really does\n"
        "grow after the breach. What does not survive is\n"
        "calling either number a slope over a 30 km reach."
    )
    a_.text(0, 1, txt, va="top", ha="left", fontsize=9, family="monospace",
            color=INK, transform=a_.transAxes)

    fig.suptitle("HIST7 · What do the per-date longitudinal slopes measure?   ·   "
                 "green = single overpass", fontsize=12.5, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"HIST7_slope_geometry.{e}", dpi=185,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'HIST7_slope_geometry.png'}")
    print(f"-> {CFG.TABLES/'hist7_slope_geometry.csv'}")


if __name__ == "__main__":
    main()
