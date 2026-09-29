#!/usr/bin/env python
"""HISTORICAL 11 — figures for the dynamic pre-breach water surface.

Task 7. NOTE ON NUMBERING: V7 and V8 are already taken by
V7_atl08_lakebed.png and (in the earlier deck work) other V-series plates, so
these four carry descriptive filenames instead of bare V-numbers. Nothing is
overwritten.

    V7_historical_seiches.png       periods, nodal geometry, documented range
    V8_wind_setup_by_gauge.png      wind response against speed, seven gauges
    V9_scale_comparison.png         mean gradient vs seiche vs wind vs accuracy
    V10_conceptual_decomposition.png   H(x,t) = H_mean + H_wind + H_seiche + eps

V10 is explicitly a SCHEMATIC. Its wind and seiche components are drawn at the
documented amplitudes but with arbitrary phase, because no contemporaneous
forcing record exists for any ICESat-2 overpass. It illustrates a decomposition;
it does not estimate one.
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
PURPLE = "#6b4c8a"
H = ROOT / "data/processed/historical"


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"{name}.{e}", dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/(name + '.png')}")


def mode_shape(x, nodes, length=238.0):
    """Schematic standing-wave shape whose ZEROS sit at the documented nodes.

    A plain cos(n*pi*x/L) has its zeros wherever the algebra puts them -- for
    the uninodal mode that is mid-reservoir, whereas the source places the node
    at Kuchuhury, 150 km. Drawing the textbook shape next to a node line at
    150 km would show a zero crossing that is visibly not at the node. So the
    phase is warped through the documented node positions instead: the shape
    stays schematic, the node LOCATIONS are the source's.
    """
    knots = [0.0] + list(nodes) + [length]
    phase = [0.0] + [(i + 0.5) * np.pi for i in range(len(nodes))]
    phase += [phase[-1] + np.pi / 2]
    return np.cos(np.interp(x, knots, phase))


def v7(se):
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4),
                           gridspec_kw=dict(width_ratios=[1, 1.25]))
    a = ax[0]
    m = se[se.n_nodes.notna()].sort_values("period_h")
    y = np.arange(len(m))
    # Colour BY MODE, not by sort order: the right-hand panel draws the
    # uninodal in BLUE and the binodal in GREEN, and a bar chart keyed to the
    # sort order gave the same mode two different colours across the figure.
    MODE_C = {1: BLUE, 2: GREEN, 3: AMBER}
    a.barh(y, m.period_h, color=[MODE_C[int(n)] for n in m.n_nodes],
           alpha=0.8, height=0.5)
    for i, r in zip(y, m.itertuples()):
        lab = f"{r.period_h:.1f} h"
        if np.isfinite(r.peak_to_peak_cm):
            lab += f"   ·   {r.peak_to_peak_cm:.0f} cm peak-to-peak"
        a.text(r.period_h + 0.25, i, lab, va="center", fontsize=9.5)
    sm = se[se.n_nodes.isna()]
    for r in sm.itertuples():
        a.axvline(r.period_h, color=GREY, ls=":", lw=1.4)
        a.text(r.period_h, len(m) - 0.35, f" {r.period_h:.1f} h", fontsize=8,
               color=GREY, rotation=90, va="top")
    a.set_yticks(y)
    a.set_yticklabels([f"{r.mode}\n({int(r.n_nodes)} node"
                       f"{'s' if r.n_nodes > 1 else ''})" for r in m.itertuples()],
                      fontsize=9)
    a.set_xlim(0, 17); a.set_xlabel("period (hours)")
    a.grid(axis="x", alpha=0.25)
    a.set_title("Documented seiche modes", fontsize=11.5, loc="left")
    a.text(0.98, 0.04, "dotted: smooth oscillations found\nexperimentally, "
           "not identified as modes", transform=a.transAxes, ha="right",
           fontsize=7.6, color=GREY)

    a = ax[1]
    a.set_xlim(-6, 246); a.set_ylim(-1.55, 1.45)
    x = np.linspace(0, 240, 500)
    a.plot(x, mode_shape(x, [150.0]), color=BLUE, lw=2.2,
           label="uninodal (13.4 h) — node at 150 km")
    a.plot(x, mode_shape(x, [82.0, 175.0]), color=GREEN, lw=2.2, ls="--",
           label="binodal (7.3 h) — nodes at 82 and 175 km")
    a.axhline(0, color=INK, lw=1)
    for nm, km, what in [("Mykhailivka", 82, "binodal node"),
                         ("Kuchuhury", 150, "uninodal node"),
                         ("Blahovishchenka", 175, "binodal node")]:
        c = BLUE if "uninodal" in what else GREEN
        a.axvline(km, color=c, lw=1.6, ls=":")
    # Kuchuhury (150) and Blahovishchenka (175) are 25 km apart and their labels
    # overlapped, so they are staggered vertically.
    for (nm, km, what), yl in zip(
            [("Mykhailivka", 82, "binodal node"),
             ("Kuchuhury", 150, "uninodal node"),
             ("Blahovishchenka", 175, "binodal node")], [1.16, 1.16, 0.86]):
        c = BLUE if "uninodal" in what else GREEN
        a.text(km, yl, f"{nm}\n{km} km", fontsize=7.6, ha="center", color=c)
    for km, nm, ha in [(0, "dam", "left"), (133, "Nikopol (central)", "center"),
                       (238, "Dnipro HPP", "right")]:
        a.plot(km, 0, "v", color=RED, ms=9)
        a.text(km, 0.10, f"{nm}\nantinode", fontsize=7.6, ha=ha, color=RED)
    a.set_xlabel("chainage from the dam (km)")
    a.set_ylabel("relative displacement")
    a.set_yticks([])
    a.legend(fontsize=8.4, loc="lower center", framealpha=0.95)
    a.set_title("Nodal geometry, as located in the source text",
                fontsize=11.5, loc="left")

    fig.suptitle("V7 · Documented seiches of the Kakhovka reservoir   ·   "
                 "mode shapes are schematic; periods and node positions are "
                 "from the source", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "V7_historical_seiches")


def v8(wc, ws):
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
    a = ax[0]
    cols = [BLUE, GREEN, AMBER, PURPLE, GREY, RED, "#8a5a12"]
    for (st, g), c in zip(wc.groupby("station", sort=False), cols):
        g = g[g.setup_or_setdown == "setup"].sort_values("wind_speed_m_s")
        km = ws.loc[ws.station == st, "station_chainage_km"].iloc[0]
        a.errorbar(g.wind_speed_m_s, g.delta_level_cm,
                   yerr=g.read_uncertainty_cm, fmt="o-", color=c, lw=1.9, ms=6,
                   capsize=3, alpha=0.9, label=f"{st}  ({km:.0f} km)")
    a.set_xlabel("wind speed over the water surface (m/s)")
    a.set_ylabel("setup / setdown magnitude (cm)")
    a.legend(fontsize=8, title="gauge (chainage)", title_fontsize=8)
    a.grid(alpha=0.25)
    a.set_title("Read from Fig. 57 at the tick speeds, ±10 cm",
                fontsize=11.2, loc="left")

    a = ax[1]
    w = ws.sort_values("station_chainage_km")
    a.plot(w.station_chainage_km, w.max_response_cm, "o-", color=RED, lw=2.2,
           ms=11)
    for r in w.itertuples():
        a.annotate(f"{r.station}\n{r.max_response_cm:.0f} cm",
                   (r.station_chainage_km, r.max_response_cm),
                   textcoords="offset points", xytext=(0, 13), ha="center",
                   fontsize=8)
    a.set_xlabel("chainage from the dam (km)")
    a.set_ylabel("maximum response at 25 m/s (cm)")
    a.set_ylim(0, 190)
    a.grid(alpha=0.25)
    a.set_title("Both ENDS respond ~6× more than the centre — a tilt about a "
                "central pivot", fontsize=11.2, loc="left")

    fig.suptitle("V8 · Wind setup and setdown by gauge   ·   the source plots "
                 "both on one axis, so the magnitude is shared and the sign "
                 "follows the wind", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "V8_wind_setup_by_gauge")


def v9(cmp_):
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    d = cmp_.copy()
    kind = []
    for q in d.quantity:
        if "OBSERVED" in q:
            kind.append(RED)
        elif "mean hydraulic" in q:
            kind.append(BLUE)
        elif "seiche" in q:
            kind.append(GREEN)
        elif "wind" in q:
            kind.append(AMBER)
        else:
            kind.append(GREY)
    d = d.assign(c=kind).sort_values("value_cm")
    y = np.arange(len(d))
    ax.barh(y, d.value_cm.clip(lower=0.05), color=d.c, alpha=0.82, height=0.62)
    for i, r in zip(y, d.itertuples()):
        ax.text(max(r.value_cm, 0.05) * 1.12, i, f"{r.value_cm:.1f} cm",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([q.replace(", ", ",\n") for q in d.quantity], fontsize=8.6)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 500)
    ax.set_xlabel("magnitude (cm, log scale)")
    ax.grid(axis="x", alpha=0.28)
    ax.set_title("V9 · What moves the pre-breach water surface, to scale",
                 fontsize=13.5, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    # Below the axes, not above: at 8.8 pt this caption is three lines wide and
    # collided with the title when anchored to the top of the axes.
    fig.text(0.01, -0.02,
             "The observed within-overpass spread (red) EXCEEDS the entire mean "
             "hydraulic rise across the main pool (blue), and sits well inside "
             "the documented seiche range (green).\n"
             "Wind (amber) is larger again. The Qmin rise is 0.0 cm to the "
             "precision Table 20 is printed at, so it is drawn at the axis floor.",
             fontsize=9, color=GREY, va="top")
    save(fig, "V9_scale_comparison")


def v10(t20, se, ws):
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 8.2), sharex=True)
    x = np.linspace(0, 238, 400)
    pool = t20[t20.Q_label == "Q20%"].sort_values("km")
    Hm = np.interp(x, pool.km, pool.WSE_historical_m) - 16.0

    a = ax[0, 0]
    a.plot(x, 100 * Hm, color=BLUE, lw=2.4)
    a.set_ylabel("cm above the dam")
    a.set_title("H_mean(x) — long-term geometry (Table 20, Q20%)",
                fontsize=10.8, loc="left")
    a.grid(alpha=0.22)

    a = ax[0, 1]
    amp = ws.set_index("station").loc[:, "max_response_cm"]
    tilt = np.interp(x, [0, 133, 238], [-35, 0, +30])
    a.plot(x, tilt, color=AMBER, lw=2.4)
    a.axhline(0, color=INK, lw=0.9)
    a.set_title("H_wind(x,t) — a 15 m/s gale, arbitrary direction (Fig. 57)",
                fontsize=10.8, loc="left")
    a.grid(alpha=0.22)

    a = ax[1, 0]
    p2p = float(se.peak_to_peak_cm.dropna().iloc[0])
    seiche = (p2p / 2) * mode_shape(x, [82.0, 175.0])
    a.plot(x, seiche, color=GREEN, lw=2.4)
    a.axhline(0, color=INK, lw=0.9)
    for km in (82, 175):
        a.axvline(km, color=GREEN, ls=":", lw=1.4)
    a.set_ylabel("cm"); a.set_xlabel("chainage from the dam (km)")
    a.set_title(f"H_seiche(x,t) — binodal, {p2p:.0f} cm peak-to-peak, "
                f"ARBITRARY PHASE", fontsize=10.8, loc="left")
    a.grid(alpha=0.22)

    a = ax[1, 1]
    rng = np.random.default_rng(CFG.SEED)
    total = 100 * Hm + tilt + (p2p / 2) * mode_shape(x, [82.0, 175.0])
    a.plot(x, 100 * Hm, color=BLUE, lw=1.6, ls="--", label="H_mean only")
    a.plot(x, total, color=RED, lw=2.4, label="H_mean + wind + seiche")
    for x0 in (40, 105, 160, 205):
        seg = (x >= x0) & (x <= x0 + 11)
        a.plot(x[seg], total[seg] + rng.normal(0, 2, seg.sum()), color=INK,
               lw=3.4, alpha=0.75, solid_capstyle="butt")
    a.plot([], [], color=INK, lw=3.4, alpha=0.75,
           label="what one ICESat-2 overpass sees (~11 km)")
    a.axhline(0, color=GREY, lw=0.9)
    # Clip to the pool. Letting the backwater limb set the y-range (it reaches
    # +260 cm) compressed the pool into a flat line and hid the entire point of
    # the figure, which is that the dynamic terms dominate THERE.
    a.set_ylim(-45, 70)
    a.annotate("limb continues to +260 cm,\noff-scale", (232, 62),
               ha="right", va="top", fontsize=8, color=GREY)
    a.set_xlabel("chainage from the dam (km)")
    a.legend(fontsize=8.4, loc="lower left")
    a.grid(alpha=0.22)
    a.set_title("What a single overpass samples (pool detail; limb clipped)",
                fontsize=10.8, loc="left")

    fig.suptitle("V10 · SCHEMATIC decomposition   H(x,t) = H_mean(x) + "
                 "H_wind(x,t) + H_seiche(x,t) + ε   ·   amplitudes are "
                 "documented, phases are arbitrary", fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "V10_conceptual_decomposition")


def main() -> None:
    se = pd.read_csv(H / "historical_seiches.csv")
    wc = pd.read_csv(H / "historical_wind_setup_curves.csv")
    ws = pd.read_csv(H / "historical_wind_setup_summary.csv")
    cmp_ = pd.read_csv(CFG.TABLES / "hist10_dynamic_vs_mean.csv")
    t20 = pd.read_csv(H / "historical_table20_wse.csv").rename(
        columns={"chainage_km_historical": "km"})
    v7(se)
    v8(wc, ws)
    v9(cmp_)
    v10(t20, se, ws)
    print("\nV10 is a schematic: documented amplitudes, arbitrary phases. No")
    print("component is estimated for any specific ICESat-2 overpass, because")
    print("no contemporaneous wind or level record was used.")


if __name__ == "__main__":
    main()
