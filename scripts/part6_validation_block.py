#!/usr/bin/env python
"""PART 6 — the VALIDATION block that precedes the main Kakhovka results.

Answers the reviewer's first question ("why should I trust these heights?")
before any science claim is made. Nothing here is tuned to support a preferred
correction, and the two negative results (2023-04-05 anomaly; non-transferable
reservoir correction) are given the same weight as the positive ones.

Figures V1-V6 + figure data + a markdown section for the manuscript.
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
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

T, FIG, FD = CFG.TABLES, CFG.FIG, CFG.FIGDATA
GD = ROOT / "data/processed/gauges"
KHERSON = 80805
INK, BLUE, RED, AMBER, GREY = "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#8a94a3"
RNG = np.random.default_rng(CFG.SEED)


def nmad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    m = [np.median(RNG.choice(x, len(x), True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG/name}.png")


# ===================================================================== V1 ====
V1T = {
 "en": dict(t="The datum problem: four different reference surfaces",
  s="Gauges are normal heights on a 1977 Baltic datum; the two satellites are ellipsoidal with different geoid and tide conventions.\nNothing can be compared until all of it is on ONE surface in ONE tide system.",
  g="GAUGE  (in situ)", i="ICESat-2  ATL13", w="SWOT  L2_HR_PIXC",
  gg=["stage, cm above a\nlocal gauge zero","Baltic 1977 (BS-77)\nnormal heights","— (levelling network)","+ gauge zero\n(12.00 m reservoir,\n\u22125.00 m Kherson)","+ Δ EPSG:9902\n(0.172 … 0.216 m here)","EVRF2019\nEPSG:9389, zero-tide"],
  ii=["ht_water_surf,\nellipsoidal height","WGS84 / ITRF2020\nellipsoid","TIDE-FREE\n(ATL03 ATBD v007 p.8)","+ tide_earth_free2mean\n= 0.06029 − 0.180873 sin²φ","− ζ EGG2015\n(quasigeoid, zero-tide)","EGG2015 common frame\n(EVRF2007-consistent)"],
  ww=["height,\nellipsoidal (raw)","WGS84 ellipsoid;\n`geoid` var = EGM2008","MEAN-TIDE crust;\ntides reported, NOT applied","− solid_earth_tide\n− load_tide_fes\n− pole_tide","− ζ EGG2015\n(never − geoid: 23.9 m error)","EGG2015 common frame"],
  lab=["native","datum","tide","step 1","step 2","result"],
  f1="ONE COMMON FRAME  —  all comparisons in this study are made here",
  f2="EGG2015 quasigeoid (zero-tide) for the satellites  ·  EVRF2019 (zero-tide) for the gauges  ·  the two are tied per station by an empirical constant c, never by a global bias",
  f3="both ends of the EVRS branch are ZERO-TIDE → no tide-system mixing"),
 "uk": dict(t="Проблема датумів: чотири різні опорні поверхні",
  s="Пости — нормальні висоти в Балтійській системі 1977 р.; обидва супутники — еліпсоїдальні, з різними геоїдами й припливними конвенціями.\nНічого не можна порівнювати, доки все не зведено на ОДНУ поверхню в ОДНІЙ припливній системі.",
  g="ГІДРОПОСТ  (наземний)", i="ICESat-2  ATL13", w="SWOT  L2_HR_PIXC",
  gg=["стан, см над власним\nнулем поста","Балтійська 1977 (БС-77)\nнормальні висоти","— (нівелірна мережа)","+ нуль поста\n(12.00 м водосховище,\n\u22125.00 м Херсон)","+ Δ EPSG:9902\n(0.172 … 0.216 м тут)","EVRF2019\nEPSG:9389, нуль-приплив"],
  ii=["ht_water_surf,\nеліпсоїдальна висота","WGS84 / ITRF2020\nеліпсоїд","TIDE-FREE\n(ATL03 ATBD v007 с.8)","+ tide_earth_free2mean\n= 0.06029 − 0.180873 sin²φ","− ζ EGG2015\n(квазігеоїд, нуль-приплив)","спільна система EGG2015\n(узгоджена з EVRF2007)"],
  ww=["height,\nеліпсоїдальна (сира)","WGS84 еліпсоїд;\nполе `geoid` = EGM2008","MEAN-TIDE кора;\nприпливи подані, НЕ застосовані","− solid_earth_tide\n− load_tide_fes\n− pole_tide","− ζ EGG2015\n(ніколи − geoid: помилка 23.9 м)","спільна система EGG2015"],
  lab=["нативне","датум","приплив","крок 1","крок 2","результат"],
  f1="ОДНА СПІЛЬНА СИСТЕМА  —  усі порівняння в цій роботі зроблені тут",
  f2="квазігеоїд EGG2015 (нуль-приплив) для супутників  ·  EVRF2019 (нуль-приплив) для постів  ·  їх зв'язує station-specific константа c, ніколи не глобальне зміщення",
  f3="обидва кінці EVRS-гілки НУЛЬ-ПРИПЛИВНІ → змішування припливних систем немає"),
}


def v1_framework(lang="en"):
    """The datum problem: what each dataset is natively in, and the path to one frame."""
    L = V1T[lang]
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 3); ax.set_ylim(0, 10.4); ax.axis("off")

    ax.text(1.5, 10.15, L["t"],
            ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(1.5, 9.75, L["s"], ha="center", fontsize=9.5, color=GREY)

    cols = [(0.5, L["g"], BLUE, list(zip(L["lab"], L["gg"]))),
            (1.5, L["i"], RED, list(zip(L["lab"], L["ii"]))),
            (2.5, L["w"], AMBER, list(zip(L["lab"], L["ww"])))]
    ys = [8.55, 7.35, 6.35, 4.95, 3.45, 1.85]
    hs = [0.70, 0.55, 0.55, 0.95, 0.95, 0.80]
    for x, title, c, items in cols:
        ax.text(x, 9.15, title, ha="center", fontsize=11.5, fontweight="bold", color=c)
        for (lab, txt), y, h in zip(items, ys, hs):
            face = c if lab == L["lab"][5] else "white"
            tcol = "white" if lab == L["lab"][5] else INK
            ax.add_patch(Rectangle((x - 0.44, y - h / 2), 0.88, h, facecolor=face,
                                   edgecolor=c, linewidth=1.4, zorder=2,
                                   alpha=1.0 if lab == L["lab"][5] else 0.95))
            ax.text(x - 0.40, y + h / 2 - 0.13, lab.upper(), fontsize=6.6,
                    color=tcol if lab == L["lab"][5] else GREY, fontweight="bold")
            ax.text(x, y - 0.04, txt, ha="center", va="center", fontsize=8.2, color=tcol)
        for y0, h0, y1 in zip(ys[:-1], hs[:-1], ys[1:]):
            ax.add_patch(FancyArrowPatch((x, y0 - h0 / 2), (x, y1 + 0.30),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color=c, linewidth=1.1, zorder=1))

    ax.add_patch(Rectangle((0.02, 0.30), 2.96, 0.85, facecolor="#eef4f6",
                           edgecolor=BLUE, linewidth=1.6, zorder=1))
    ax.text(1.5, 0.90, L["f1"], ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(1.5, 0.55, L["f2"], ha="center", fontsize=8.6, color=INK)
    ax.text(1.5, 5.75, L["f3"], ha="center", fontsize=8, style="italic", color=GREY)
    fig.tight_layout()
    save(fig, f"V1_vertical_reference_framework_{lang}")


# ===================================================================== V2 ====
def v2_pixc_signs():
    d = pd.read_csv(FD / "Fig03_pixc_riversp_sign_validation.csv")
    d = d.set_index("variant").loc[["A", "B", "C", "D"]].reset_index()
    lab = ["height − (st+lt+pt)\nDOCUMENTED CHAIN", "height\nno tide correction",
           "height + (st+lt+pt)\nreversed sign", "height − geoid\ngeoid blunder"]
    col = [BLUE, AMBER, AMBER, RED]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                 gridspec_kw=dict(width_ratios=[2.4, 1]))
    y = np.arange(3)[::-1]
    for i in range(3):
        r = d.iloc[i]
        a1.errorbar(r.median_diff_m, y[i],
                    xerr=[[r.median_diff_m - r.ci95_low_m], [r.ci95_high_m - r.median_diff_m]],
                    fmt="o", ms=12, color=col[i], ecolor=col[i], elinewidth=2.4, capsize=5)
        a1.text(r.median_diff_m, y[i] + 0.22, f"{r.median_diff_m:+.4f} m",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    a1.axvline(0, color=INK, lw=1.2)
    a1.axvspan(-0.01, 0.01, color=BLUE, alpha=0.10)
    a1.set_yticks(y); a1.set_yticklabels(lab[:3], fontsize=9.5)
    a1.set_xlabel("PIXC − RiverSP, same cycle/pass  (m)")
    a1.set_xlim(-0.03, 0.17); a1.grid(axis="x", alpha=0.25)
    a1.set_title("Only the documented chain returns zero", fontsize=11, loc="left")

    r = d.iloc[3]
    a2.errorbar(r.median_diff_m, 0, xerr=[[r.median_diff_m - r.ci95_low_m],
                                          [r.ci95_high_m - r.median_diff_m]],
                fmt="o", ms=12, color=RED, ecolor=RED, elinewidth=2.4, capsize=5)
    a2.text(r.median_diff_m, 0.16, f"{r.median_diff_m:+.2f} m", ha="center",
            fontsize=11, fontweight="bold", color=RED)
    a2.axvline(0, color=INK, lw=1.2)
    a2.set_yticks([0]); a2.set_yticklabels([lab[3]], fontsize=9.5)
    a2.set_xlim(-25, 2); a2.set_ylim(-0.5, 0.5); a2.grid(axis="x", alpha=0.25)
    a2.set_xlabel("(m)"); a2.set_title("own axis", fontsize=9, loc="left", color=GREY)
    fig.suptitle(f"V2 · SWOT PIXC tide-correction sign, validated against RiverSP  "
                 f"(n = {int(d.n_nodes.iloc[0]):,} nodes, one cycle/pass)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    save(fig, "V2_pixc_sign_validation")
    d.to_csv(FD / "V2_pixc_sign_validation.csv", index=False)


# ===================================================================== V3 ====
def v3_colocated():
    c = pd.read_csv(T / "part4_colocated_swot_icesat.csv", parse_dates=["date"])
    c = c.sort_values("date")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(c))
    ok = c.within_overpass_nmad_m < 0.15
    for ax, xs, xlab in ((a1, x, None), (a2, c.dt_hours.abs().values,
                                         "|temporal separation| SWOT − ICESat-2  (h)")):
        ax.axhline(0, color=INK, ls="--", lw=1)
        ax.axhspan(-0.01, 0.01, color=BLUE, alpha=0.12)
        ax.errorbar(xs, c.d_swot_minus_icesat_m, yerr=c.within_overpass_nmad_m,
                    fmt="none", ecolor=GREY, elinewidth=1.6, capsize=4, zorder=2)
        ax.scatter(xs, c.d_swot_minus_icesat_m, s=130, zorder=3,
                   c=[BLUE if o else AMBER for o in ok], edgecolor="white", linewidth=1.2)
        for xi, r in zip(xs, c.itertuples()):
            ax.annotate(f"{r.d_swot_minus_icesat_m:+.3f}", (xi, r.d_swot_minus_icesat_m),
                        textcoords="offset points", xytext=(0, 14), ha="center",
                        fontsize=10, fontweight="bold")
        ax.set_ylim(-0.26, 0.14); ax.grid(alpha=0.25)
        if xlab:
            ax.set_xlabel(xlab); ax.set_xlim(0, 16)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d:%Y-%m-%d}\nΔt {t:+.1f} h" for d, t in
                                zip(c.date, c.dt_hours)], fontsize=9)
        ax.set_ylabel("$H_{SWOT} - H_{ICESat-2}$   (m)")
    a1.set_title("(a) one point per co-located overpass", fontsize=11, loc="left")
    a2.set_title("(b) the residual orders with Δt", fontsize=11, loc="left")
    a2.annotate("best-matched overpass\n2023-05-14, Δt = 2.3 h  →  +4 mm",
                xy=(2.28, 0.0036), xytext=(5.2, -0.14), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    fig.suptitle("V3 · SWOT ↔ ICESat-2 where the sensors actually overlap  "
                 "(19–57 km downstream; the gauge plays no part)", fontsize=12, y=1.02)
    fig.text(0.5, -0.04, "Independent unit = one overpass. Beams and segments within a date are "
             "pseudo-replicates and are used only for the error bar (within-overpass NMAD). "
             "Amber = ICESat-2 beam spread > 1 m.", ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout()
    save(fig, "V3_swot_icesat_colocated")
    c.to_csv(FD / "V3_swot_icesat_colocated.csv", index=False)


# ===================================================================== V4 ====
def v4_hydrograph():
    gv = pd.read_parquet(GD / "gauge_levels_evrf2019.parquet")
    kh = gv[(gv.station_id == KHERSON) & (gv.qc == "ok")].sort_values("date")
    k23 = kh[(kh.date >= "2023-03-01") & (kh.date <= "2023-06-20")]
    co = pd.read_csv(T / "part4_colocated_swot_icesat.csv", parse_dates=["date"])
    sw = pd.read_csv(T / "swot_validation_against_gauge_and_icesat.csv", parse_dates=["date"])
    sw = sw[sw.radius_km == 1.0]

    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(k23.date, k23.stage_m * 100, "-", color=INK, lw=1.4, zorder=3)
    for d in sw.date:
        ax.axvline(d, color=AMBER, lw=1.1, alpha=0.85)
    for d in co.date:
        ax.axvline(d, color=RED, ls="--", lw=1.3)
    ax.axvline(pd.Timestamp("2023-06-06"), color=RED, lw=2.4)
    ax.text(pd.Timestamp("2023-06-06"), 545, " dam breach", color=RED, fontsize=9,
            rotation=90, va="top")
    ax.plot([], [], color=AMBER, lw=1.1, label="SWOT PIXC overpass")
    ax.plot([], [], color=RED, ls="--", lw=1.3, label="same-day ICESat-2")
    ax.set_ylabel("Kherson stage  (cm above gauge zero)")
    ax.set_ylim(495, 560); ax.legend(fontsize=8.5, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.25)
    ax.set_title("V4 · Kherson gauge 80805, 2023, with satellite acquisition epochs",
                 fontsize=12, loc="left")

    rows = []
    for i, r in enumerate(co.itertuples()):
        a = fig.add_subplot(gs[1, i])
        w = kh[(kh.date >= r.date - pd.Timedelta(days=6)) &
               (kh.date <= r.date + pd.Timedelta(days=6))]
        a.plot(w.date, w.stage_m * 100, "o-", ms=4, color=INK, lw=1.2)
        a.axvline(r.date, color=RED, ls="--", lw=1.3)
        slope = r.gauge_dHdt_cm_day
        state = ("RISING" if slope > 3 else "FALLING" if slope < -3 else "STABLE")
        exp = r.expected_dH_cm
        obs = r.d_swot_minus_icesat_m * 100
        a.set_title(f"{r.date:%Y-%m-%d}  ·  {state}", fontsize=10,
                    color=BLUE if state == "STABLE" else INK)
        a.text(0.03, 0.04, f"dH/dt = {slope:+.1f} cm/day\n"
                           f"expected over Δt={r.dt_hours:+.1f} h: {exp:+.1f} cm\n"
                           f"observed SWOT−ICESat: {obs:+.1f} cm\n"
                           f"residual: {obs - exp:+.1f} cm",
               transform=a.transAxes, fontsize=8, va="bottom",
               bbox=dict(fc="white", ec=GREY, alpha=0.9))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        a.tick_params(axis="x", labelsize=8, rotation=30)
        a.set_ylabel("stage (cm)" if i == 0 else "")
        a.grid(alpha=0.25)
        rows.append({"date": r.date, "state": state, "dHdt_cm_day": slope,
                     "dt_hours": r.dt_hours, "expected_dH_cm": exp,
                     "observed_swot_minus_icesat_cm": obs, "residual_cm": obs - exp})
    pd.DataFrame(rows).to_csv(FD / "V4_hydrograph_context.csv", index=False)
    a0 = pd.DataFrame(rows).iloc[0]
    fig.text(0.5, -0.02, f"On {a0.date:%Y-%m-%d} the gauge was {a0.state}, so with SWOT "
             f"{a0.dt_hours:+.1f} h later the hydrograph predicts SWOT HIGHER by "
             f"{a0.expected_dH_cm:+.1f} cm; SWOT reads {a0.observed_swot_minus_icesat_cm:+.1f} cm. "
             f"The hydrological term has the wrong sign and makes the residual worse "
             f"({a0.residual_cm:+.1f} cm) — the anomaly is unresolved.",
             ha="center", fontsize=8.8, color=RED)
    save(fig, "V4_kherson_hydrograph_epochs")


# ===================================================================== V5 ====
def v5_forest():
    sw = pd.read_csv(T / "swot_validation_against_gauge_and_icesat.csv")
    mu = pd.read_csv(T / "icesat_station_matchups.csv")
    co = pd.read_csv(T / "part4_colocated_swot_icesat.csv")

    items = []
    s1 = sw[sw.radius_km == 1.0].R_swot_minus_gauge_m.dropna().values
    items.append(("SWOT − gauge  (1 km)", s1, "co-located <1 km · same day, not simultaneous · "
                                              "daily gauge, epoch unresolved", BLUE))
    k = mu[(mu.station_id == 80805) & (mu.radius_km == 2.0)].d_harm_m.values
    items.append(("ICESat-2 − gauge, Kherson (2 km)", k,
                  "NOT co-located: ICESat never closer than 13 km · daily gauge", AMBER))
    r = mu[(mu.domain == "reservoir") & (mu.radius_km == 2.0)].d_harm_m.values
    items.append(("ICESat-2 − gauge, reservoir (2 km)", r,
                  "co-located <2 km · pre-breach only", BLUE))
    d = co.d_swot_minus_icesat_m.values
    items.append(("SWOT − ICESat-2  (overpass)", d,
                  "sensor-to-sensor, 19–57 km downstream · gauge not involved · n = 3", RED))

    fig, ax = plt.subplots(figsize=(12, 5.2))
    y = np.arange(len(items))[::-1]
    out = []
    for yi, (lab, v, flag, c) in zip(y, items):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        med = np.median(v); lo, hi = boot_ci(v)
        ax.scatter(v, np.full(len(v), yi) + RNG.normal(0, 0.045, len(v)), s=16,
                   color=c, alpha=0.32, lw=0, zorder=2)
        if np.isfinite(lo):
            ax.plot([lo, hi], [yi, yi], color=c, lw=3.0, solid_capstyle="round", zorder=3)
        ax.scatter([med], [yi], s=150, marker="D", color=c, edgecolor="white",
                   linewidth=1.4, zorder=4)
        ax.text(med, yi + 0.235, f"{med:+.3f} m   (n={len(v)})", ha="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(0.322, yi - 0.245, flag, fontsize=7.6, color=GREY, style="italic")
        out.append({"comparison": lab, "n": len(v), "median_m": med,
                    "nmad_m": nmad(v), "ci95_low_m": lo, "ci95_high_m": hi,
                    "collocation": flag})
    ax.axvline(0, color=INK, lw=1.3)
    ax.axvspan(-0.05, 0.05, color=BLUE, alpha=0.08)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=10)
    ax.set_xlim(-0.35, 0.33); ax.set_ylim(-0.55, len(items) - 0.4)
    ax.set_xlabel("difference in water-surface height  (m)     ·     shaded band = ±5 cm")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("V5 · Every gauge/satellite residual, with its collocation quality stated",
                 fontsize=12, loc="left")
    fig.tight_layout()
    save(fig, "V5_residual_summary")
    pd.DataFrame(out).to_csv(FD / "V5_residual_summary.csv", index=False)
    return pd.DataFrame(out)


# ===================================================================== V6 ====
def v6_table(res):
    p = pd.read_csv(FD / "Fig03_pixc_riversp_sign_validation.csv").set_index("variant")
    co = pd.read_csv(T / "part4_colocated_swot_icesat.csv")
    best = co.loc[co.dt_hours.abs().idxmin()]
    h0 = pd.read_csv(FD / "V4_hydrograph_context.csv").iloc[0]
    sw1 = res[res.comparison.str.startswith("SWOT − gauge")].iloc[0]
    pool = pd.read_csv(T / "part4_pooling_test.csv")

    rows = [
        {"validation_question": "Is the SWOT PIXC tide-correction chain correct?",
         "test_performed": "PIXC vs RiverSP, same cycle/pass, n=1023 nodes",
         "numerical_result": f"{p.loc['A','median_diff_m']:+.4f} m "
                             f"(vs {p.loc['B','median_diff_m']:+.4f} uncorrected, "
                             f"{p.loc['C','median_diff_m']:+.4f} reversed, "
                             f"{p.loc['D','median_diff_m']:+.2f} geoid blunder)",
         "interpretation": "Only the documented chain returns zero",
         "status": "VALIDATED"},
        {"validation_question": "Can SWOT and ICESat-2 agree?",
         "test_performed": f"co-located overpass {best.date}, Δt = {best.dt_hours:+.2f} h, "
                           f"{int(best.n_segments)} segments",
         "numerical_result": f"{best.d_swot_minus_icesat_m:+.4f} m",
         "interpretation": "Excellent agreement under good temporal collocation; "
                           "residual orders with Δt",
         "status": "VALIDATED"},
        {"validation_question": "Does SWOT agree with the Kherson gauge?",
         "test_performed": "9 PIXC overpasses, open water within 1 km, gauge in EVRF2019",
         "numerical_result": f"{sw1.median_m:+.4f} m, NMAD {sw1.nmad_m:.4f}, "
                             f"CI [{sw1.ci95_low_m:+.4f}, {sw1.ci95_high_m:+.4f}]",
         "interpretation": "Good local agreement; CI brackets zero",
         "status": "VALIDATED"},
        {"validation_question": "Is the 2023-04-05 anomaly explained by the hydrograph?",
         "test_performed": "Kherson daily hydrograph slope over Δt = 14.2 h",
         "numerical_result": (f"gauge {h0.state} {h0.dHdt_cm_day:+.1f} cm/day → expected "
                              f"{h0.expected_dH_cm:+.1f} cm; observed "
                              f"{h0.observed_swot_minus_icesat_cm:+.1f} cm; residual "
                              f"{h0.residual_cm:+.1f} cm"),
         "interpretation": "Hydrology has the WRONG SIGN and makes it worse",
         "status": "UNRESOLVED"},
        {"validation_question": "Is the reservoir-derived correction transferable downstream?",
         "test_performed": "independent per-station constants, reservoir (n=5) vs Kherson",
         "numerical_result": f"reservoir {pool.iloc[0].median_m:+.3f} m "
                             f"[{pool.iloc[0].ci95_low_m:+.3f}, {pool.iloc[0].ci95_high_m:+.3f}] "
                             f"vs Kherson {pool.iloc[1].median_m:+.3f} m "
                             f"[{pool.iloc[1].ci95_low_m:+.3f}, {pool.iloc[1].ci95_high_m:+.3f}]",
         "interpretation": "Confidence intervals disjoint — a local/contextual term, "
                           "not a global sensor bias",
         "status": "NOT TRANSFERABLE"},
        {"validation_question": "Are all gauge levels in one geodetic system?",
         "test_performed": "BS-77 + EPSG:9902 → EVRF2019 for every station; grid sampler "
                           "cross-checked against an independent implementation",
         "numerical_result": "7 stations, 10,194 daily values; sampler agrees to 0.000 mm",
         "interpretation": "Geodetic transformation is complete and reproducible",
         "status": "VALIDATED"},
    ]
    t = pd.DataFrame(rows)
    t.to_csv(T / "validation_summary_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(15.5, 4.6)); ax.axis("off")
    colw = [0.235, 0.235, 0.245, 0.195, 0.09]
    hdr = ["Validation question", "Test performed", "Numerical result",
           "Interpretation", "Status"]
    scol = {"VALIDATED": BLUE, "UNRESOLVED": AMBER, "NOT TRANSFERABLE": RED}
    yy = 1.0
    x0 = np.concatenate([[0], np.cumsum(colw)])
    for j, h in enumerate(hdr):
        ax.text(x0[j] + 0.006, yy, h, fontsize=9.5, fontweight="bold", color=INK)
    ax.plot([0, 1], [yy - 0.035, yy - 0.035], color=INK, lw=1.4)
    yy -= 0.10
    for r in t.itertuples():
        vals = [r.validation_question, r.test_performed, r.numerical_result,
                r.interpretation, r.status]
        n = 0
        for j, v in enumerate(vals):
            wrap = int(colw[j] / 0.0052)
            txt = "\n".join([v[i:i + wrap] for i in range(0, len(v), wrap)])
            n = max(n, txt.count("\n") + 1)
            ax.text(x0[j] + 0.006, yy, txt, fontsize=7.6, va="top",
                    color=scol.get(v, INK), fontweight="bold" if j == 4 else "normal")
        yy -= 0.028 * n + 0.030
        ax.plot([0, 1], [yy + 0.014, yy + 0.014], color="#dddddd", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(yy, 1.06)
    ax.set_title("V6 · Validation summary", fontsize=12.5, loc="left", pad=12)
    save(fig, "V6_validation_summary_table")
    return t


# ===================================================================== md ====
def markdown(t, res):
    co = pd.read_csv(T / "part4_colocated_swot_icesat.csv")
    h0 = pd.read_csv(FD / "V4_hydrograph_context.csv").iloc[0]
    L = ["# 3. Validation of the observation framework\n",
         "This section establishes that the heights compared in Section 4 are "
         "commensurable. It precedes the scientific result deliberately: the first "
         "question these data raise is not what changed, but whether three "
         "instruments measured on three different reference surfaces can be "
         "compared at all.\n",
         "## 3.0 The datum problem\n",
         "The three observation systems are natively expressed on **four different "
         "reference surfaces**, in two different permanent-tide conventions "
         "(Figure V1):\n",
         "| system | native quantity | reference surface | tide system |",
         "|---|---|---|---|",
         "| Gauges | stage above a local zero | Baltic 1977 normal heights | levelling network |",
         "| ICESat-2 ATL13 | `ht_water_surf` | WGS84 / ITRF2020 ellipsoid | **tide-free** |",
         "| SWOT PIXC | `height` | WGS84 ellipsoid (its `geoid` field is EGM2008) | **mean-tide** |",
         "| target | normal height | EGG2015 quasigeoid / EVRF2019 | **zero-tide** |\n",
         "Gauge heights reach EVRF2019 by the official operation EPSG:9902 "
         "(0.172–0.216 m over this reach), applied at each station's own "
         "coordinates. ICESat-2 is moved from the tide-free system of ATL03 to the "
         "mean/zero-tide crust with `tide_earth_free2mean = 0.06029 − 0.180873 sin²φ` "
         "before the quasigeoid is removed. SWOT PIXC heights are already "
         "ellipsoidal and only the time-variable tides are removed. Because EGG2015 "
         "and EVRF2019 are both zero-tide, no tide-system mixing occurs anywhere in "
         "the chain. **The `geoid` field shipped with PIXC is never subtracted from "
         "PIXC `height`** — doing so is a 23.9 m error, quantified below.\n",
         "## 3.1 Validation of the SWOT PIXC vertical chain\n",
         "Four candidate chains were compared against the RiverSP node heights of the "
         "same cycle and pass (Figure V2, n = 1023 nodes). The documented chain "
         "`height − (solid_earth_tide + load_tide_fes + pole_tide)` returns "
         "**−0.0010 m**; omitting the correction returns +0.0720 m, reversing its "
         "sign +0.1451 m, and subtracting the geoid −23.89 m. The sign and content "
         "of the correction chain are therefore established empirically, not assumed "
         "from documentation.\n",
         "## 3.2 Cross-validation of SWOT and ICESat-2\n",
         "The two sensors were matched **to each other**, not through the gauge: "
         "ICESat-2 never passes closer than 13 km to the Kherson gauge, so any "
         "gauge-mediated sensor difference is spatially confounded. Each ATL13 "
         "segment was compared with the median of SWOT open-water pixels within "
         "500 m on the same day; the independent unit is one co-located overpass "
         "(Figure V3).\n",
         "| date | Δt (h) | segments | beams | distance from gauge | H_SWOT − H_ICESat-2 |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in co.itertuples():
        L.append(f"| {r.date} | {r.dt_hours:+.2f} | {int(r.n_segments)} | "
                 f"{int(r.n_beams)} | ~{r.median_dist_to_gauge_km:.0f} km | "
                 f"**{r.d_swot_minus_icesat_m:+.4f} m** |")
    L += ["", "The best temporally matched overpass (2023-05-14, Δt = 2.3 h, 3417 "
          "segments) gives **+0.004 m** — the two satellite surfaces agree to a few "
          "millimetres in the common frame **with no correction applied**. The "
          "residual orders with |Δt|, which is consistent with temporal/hydrodynamic "
          "mismatch rather than a vertical-datum offset; with n = 3 this is "
          "suggestive, not established, and no regression is fitted.\n",
          "## 3.3 Gauge-based consistency check at Kherson\n"]
    s = res[res.comparison.str.startswith("SWOT − gauge")].iloc[0]
    L += [f"Across nine PIXC overpasses, SWOT open water within 1 km of gauge 80805 "
          f"differs from the gauge in EVRF2019 by **{s.median_m:+.4f} m** "
          f"(NMAD {s.nmad_m:.3f}, 95 % CI [{s.ci95_low_m:+.3f}, {s.ci95_high_m:+.3f}]), "
          f"and the median moves by only ~8 cm across aggregation radii of 0.5–5 km. "
          f"The SWOT branch is locally consistent with an independent in-situ record.\n",
          "## 3.4 A validated negative result: the 2023-04-05 anomaly\n",
          "On 2023-04-05 the two sensors differ by −0.181 m. The Kherson hydrograph "
          f"does not explain it: the gauge was **rising** at {h0.dHdt_cm_day:+.1f} cm/day "
          f"(central difference), so with SWOT acquired {h0.dt_hours:+.1f} h later the "
          f"hydrograph predicts SWOT **higher** by {h0.expected_dH_cm:+.1f} cm, whereas "
          f"SWOT reads {abs(h0.observed_swot_minus_icesat_cm):.1f} cm **lower**. Applying "
          f"the hydrological term worsens the residual to {h0.residual_cm:+.1f} cm "
          f"(Figure V4). An 18 cm daily excursion is also a "
          "~99th-percentile event at this site (day-to-day |ΔH| median 3.0, p90 9.0, "
          "max 20.0 cm/day, n = 96). No wind or pressure record exists locally, so "
          "wind setup can be neither confirmed nor excluded. **The anomaly is reported "
          "as unresolved.**\n",
          "## 3.5 Limits of transferability of the gauge-derived correction\n"]
    pool = pd.read_csv(T / "part4_pooling_test.csv")
    L += [f"Empirical alignment constants were estimated independently at every "
          f"station. The five reservoir stations give a tight group "
          f"({pool.iloc[0].median_m:+.3f} m, station-to-station NMAD "
          f"{pool.iloc[0].nmad_m:.3f} m, CI [{pool.iloc[0].ci95_low_m:+.3f}, "
          f"{pool.iloc[0].ci95_high_m:+.3f}]); Kherson gives "
          f"{pool.iloc[1].median_m:+.3f} m, CI [{pool.iloc[1].ci95_low_m:+.3f}, "
          f"{pool.iloc[1].ci95_high_m:+.3f}] — **statistically indistinguishable from "
          f"zero**. The two confidence intervals are disjoint. Each constant is also "
          f"stable against the matchup radius (swing 1.8 cm at Rozumivka, 5.0 cm at "
          f"Kherson over 1–10 km), so neither is an artefact of the aggregation "
          f"window. A rank test is not quoted: with five stations against one it "
          f"cannot reach significance whatever the data say.\n",
          "**The reservoir-derived correction is therefore a local, contextual term, "
          "not a global satellite bias and not a datum fix, and it is never applied "
          "outside the domain in which it was estimated.**\n",
          "## 3.6 Validation summary\n",
          "| " + " | ".join(t.columns.str.replace('_', ' ')) + " |",
          "|" + "---|" * len(t.columns)]
    for r in t.itertuples(index=False):
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L += ["", "### Figures", "",
          "- **V1** Vertical-reference framework — the four native surfaces and the path to one frame",
          "- **V2** SWOT PIXC tide-correction sign, validated against RiverSP",
          "- **V3** SWOT ↔ ICESat-2 at true sensor overlap, by date and against Δt",
          "- **V4** Kherson hydrograph with satellite epochs and three zoom panels",
          "- **V5** All gauge/satellite residuals with collocation quality stated",
          "- **V6** Validation summary table"]
    (CFG.REPORTS / "validation_section.md").write_text("\n".join(L))
    print(f"  -> {CFG.REPORTS/'validation_section.md'}")


def main():
    print("building validation block V1-V6")
    v1_framework("en"); v1_framework("uk"); v2_pixc_signs(); v3_colocated(); v4_hydrograph()
    res = v5_forest(); t = v6_table(res); markdown(t, res)
    print(f"  -> {T/'validation_summary_table.csv'}")


if __name__ == "__main__":
    main()
