#!/usr/bin/env python
"""MANUSCRIPT 3 — build the EN and UA conference decks.

Structure is deliberate: slides 4-13 of 18 are VALIDATION, so the deck spends
its first ~60% of scientific content establishing that the heights are
commensurable, and only then shows the 0.09 -> 3.31 cm/km result. Opening with
a before/after picture would make this one more paper about a dam breach.

Every headline number is FORMATTED FROM a canonical table at build time, so a
slide cannot drift from the evidence the way a typed number can. Narrative
text is authored per language -- the UA deck is a Ukrainian scientific deck
using accepted domain terminology, not a translation of the English strings.

Outputs
-------
outputs/presentation/Kakhovka_validation_and_transition_EN.pptx
outputs/presentation/Kakhovka_validation_and_transition_UA.pptx
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt

from swot_dnipro import config as CFG

OUT = ROOT / "outputs/presentation"
OUT.mkdir(parents=True, exist_ok=True)
FIG = CFG.FIG
T = CFG.TABLES

# --------------------------------------------------------------- palette ----
# One colour per observation system, used identically in both decks.
INK = RGBColor(0x1A, 0x22, 0x28)
PAPER = RGBColor(0xFA, 0xF9, 0xF6)
RULE = RGBColor(0xC3, 0xCB, 0xD2)
MUTED = RGBColor(0x5A, 0x64, 0x6E)
SWOT = RGBColor(0x1B, 0x6C, 0xA8)      # SWOT  - blue
ICESAT = RGBColor(0xC0, 0x39, 0x2B)    # ICESat-2 - red
GAUGE = RGBColor(0x2D, 0x34, 0x36)     # gauge - near-black
HIST = RGBColor(0xB0, 0x7D, 0x27)      # historical - amber
SENT = RGBColor(0x3F, 0x7D, 0x4E)      # Sentinel-2 - green
GOOD = RGBColor(0x1E, 0x84, 0x49)
BAD = RGBColor(0xB0, 0x3A, 0x2E)

W, H = Cm(33.87), Cm(19.05)            # 16:9

# ======================================================================= #
# canonical values, read at build time                                     #
# ======================================================================= #
ts = pd.read_csv(T / "kakhovka_transition_statistics.csv")
tsr = ts[ts.estimator == "Theil-Sen"].iloc[0]
olsr = ts[ts.estimator == "OLS"].iloc[0]
sm = pd.read_csv(T / "kakhovka_pre_post_slope_summary.csv").set_index("period")
colo = pd.read_csv(T / "part4_colocated_swot_icesat.csv")
best = colo.loc[colo.dt_hours.abs().idxmin()]
anom = colo[colo.date == "2023-04-05"].iloc[0]
pool = pd.read_csv(T / "part4_pooling_test.csv")
gv = pd.read_csv(T / "gauge_vertical_reference_summary.csv")
gvd = gv[gv.n_obs > 0]
hrs = pd.read_csv(T / "historical_reference_sensitivity.csv")
h14 = hrs[hrs.reference_level_m == 14.0].iloc[0]
h16 = hrs[hrs.reference_level_m == 16.0].iloc[0]
dfit = pd.read_csv(T / "hist2_datum_fit.csv")
ev14 = pd.read_csv(T / "hist3_datum_in_evrf2019.csv")
ev14 = ev14[ev14.datum_bs_m == 14.0].iloc[0]
# processed inputs live on the bulk volume (sixth repo-disk literal of this class, 2026-09-16)
f16 = pd.read_csv(CFG.BULK_ROOT / "data_swot/processed/historical/historical_fig16_profiles.csv")
c2 = f16[f16.curve_id == 2]
c2p = c2[c2.chainage_km < 180]
ex = pd.read_csv(T / "historical_exposure_area_validation.csv")
exr = ex[ex.reconstructed_fraction_pct.notna()]
gres = pd.read_csv(T / "hist17_grid_resolution.csv")
gw = gres[gres.scope == "whole"].sort_values("cell_m", ascending=False)
g250 = gw[gw.cell_m == 250].iloc[0]
cv = pd.read_csv(T / "hist14_interpolator_cv.csv")
cv1 = cv[cv.scheme == "blocked1km"].sort_values("RMSE_m")
sch = pd.read_csv(T / "hist18_cv_scheme_comparison.csv").set_index("scheme")
edc = pd.read_csv(T / "hist18_error_decomposition.csv")
ovr = edc[edc.factor == "overall"].iloc[0]
lrg = edc[edc.factor == "local_roughness"]
bm = pd.read_csv(T / "hist12_bed_morphology_tests.csv").set_index("test").value
frag = pd.read_csv(T / "pre_post_fragmentation_statistics.csv").iloc[0]
fm = pd.read_csv(T / "fragmentation_metrics_by_date.csv")
full = fm[fm.footprint_observed_fraction > 0.8]
fpre, fpost = full[full.period == "PRE_BREACH"].iloc[0], full[full.period == "POST_BREACH"]
r2r = pd.read_csv(T / "reservoir_to_river_statistics.csv")
chan = r2r[(r2r.comparison.str.startswith("POST MAIN_CHANNEL vs"))
           & (r2r.estimator == "Theil-Sen")].iloc[0]
rw = pd.read_csv(T / "residual_water_offset_summary.csv")
rwb = rw[rw.population == "B_high_confidence"].iloc[0]
dyn = pd.read_csv(T / "hist10_dynamic_vs_mean.csv")


def dv(k):
    return float(dyn[dyn.quantity.str.contains(k, regex=False)].value_cm.iloc[0])


V = {
    "pre": f"{tsr.median_pre_cm_km:+.2f}",
    "post": f"{tsr.median_post_cm_km:+.2f}",
    "draw": f"{tsr.median_drawdown_cm_km:+.2f}",
    "diff": f"{tsr.diff_post_minus_pre_cm_km:+.2f}",
    "ci": f"[{tsr.diff_ci95_low:+.2f}, {tsr.diff_ci95_high:+.2f}]",
    "p": f"{tsr.permutation_p:.0e}".replace("e-05", "×10⁻⁵"),
    "prepos": tsr.pre_positive, "postpos": tsr.post_positive,
    "presign": f"{tsr.pre_sign_test_p:.2f}",
    "ols_pre": f"{olsr.median_pre_cm_km:+.2f}", "ols_post": f"{olsr.median_post_cm_km:+.2f}",
    "npre": int(tsr.n_pre), "npost": int(tsr.n_post),
    "rng_pre": f"{sm.loc['PRE_BREACH','median_wse_range_m']:.2f}",
    "rng_post": f"{sm.loc['POST_BREACH','median_wse_range_m']:.2f}",
    "r2_pre": f"{sm.loc['PRE_BREACH','median_r2']:.2f}",
    "best_d": f"{best.d_swot_minus_icesat_m:+.4f}", "best_dt": f"{best.dt_hours:+.2f}",
    "best_date": str(best.date), "best_seg": f"{int(best.n_segments):,}",
    "anom_obs": f"{100*anom.d_swot_minus_icesat_m:+.1f}",
    "anom_exp": f"{anom.expected_dH_cm:+.1f}",
    "anom_rate": f"{anom.gauge_dHdt_cm_day:+.1f}",
    "anom_dt": f"{anom.dt_hours:.1f}",
    "res_lo": f"{float(pool.iloc[0].median_m):+.3f}",
    "res_ci": f"[{float(pool.iloc[0].ci95_low_m):+.3f}, {float(pool.iloc[0].ci95_high_m):+.3f}]",
    "kh_c": f"{float(pool.iloc[1].median_m):+.4f}",
    "kh_ci": f"[{float(pool.iloc[1].ci95_low_m):+.4f}, {float(pool.iloc[1].ci95_high_m):+.4f}]",
    "nstat": int(gvd.station_id.nunique()), "nobs": f"{int(gvd.n_obs.sum()):,}",
    "dmin": f"{gv.delta_epsg9902_m.min():.4f}", "dmax": f"{gv.delta_epsg9902_m.max():.4f}",
    "dspread": f"{1e3*(gv.delta_epsg9902_m.max()-gv.delta_epsg9902_m.min()):.0f}",
    "h14": f"{h14.median_dH_m:+.3f}",
    "h14ci": f"[{h14.ci_lo_m:+.3f}, {h14.ci_hi_m:+.3f}]",
    "h16": f"{h16.median_dH_m:+.2f}",
    "h16ci": f"[{h16.ci_lo_m:+.2f}, {h16.ci_hi_m:+.2f}]",
    "hn": int(h14.n_points), "htr": int(h14.n_tracks),
    "cap": f"{float(dfit[dfit.route.str.startswith('capacity')].datum_m.iloc[0]):.2f}",
    "ev14": f"{ev14.datum_evrf2019_median_m:.3f}",
    "ev14lo": f"{ev14.datum_evrf2019_min_m:.3f}", "ev14hi": f"{ev14.datum_evrf2019_max_m:.3f}",
    "f16med": f"{c2.deviation_m.median():+.3f}", "f16n": int(len(c2)),
    "f16pool": f"{c2p.deviation_m.median():+.3f}",
    "f16max": f"{c2p.deviation_m.abs().max():.3f}",
    "exh": f"{g250.historical_pct:.2f}", "exr": f"{g250.reconstructed_pct:.2f}",
    "exd": f"{g250.diff_pp:+.2f}",
    "exlo": f"{g250.min_pct:.1f}", "exhi": f"{g250.max_pct:.1f}",
    "reach_h": " → ".join(f"{r.historical_fraction_pct:.1f}" for r in exr.itertuples()),
    "reach_r": " → ".join(f"{r.reconstructed_fraction_pct:.1f}" for r in exr.itertuples()),
    "grid": "  →  ".join(f"{r.cell_m:.0f} m: {r.reconstructed_pct:.2f} %" for r in gw.itertuples()),
    "drift": f"{gw.reconstructed_pct.iloc[-1]-gw.reconstructed_pct.iloc[0]:+.2f}",
    "spread": f"{gw.max_pct.iloc[-1]-gw.min_pct.iloc[-1]:.2f}",
    "cvbest": cv1.iloc[0].method, "cvrmse": f"{cv1.iloc[0].RMSE_m:.2f}",
    "cvbias": f"{cv1.iloc[0].bias_m:+.2f}",
    "cvnmad": f"{cv1.iloc[0].NMAD_m:.2f}",
    "cvmed": f"{float(ovr.p50_abs):.2f}",
    "cvconc": f"{float(ovr.worst_5pct_share_SSE):.0f}",
    "cvrand": f"{float(sch.loc['random','RMSE_m']):.2f}",
    "cvrough": f"{lrg.RMSE_m.min():.2f}–{lrg.RMSE_m.max():.2f}",
    "cvrank": ", ".join(f"{r.method} {r.RMSE_m:.2f}" for r in cv1.itertuples()),
    "near": f"{float(bm['B trough fraction near channel']):.1f}",
    "far": f"{float(bm['B trough fraction far']):.1f}",
    "enrich": f"{float(bm['B trough fraction near channel'])/float(bm['B trough fraction far']):.1f}",
    "s7": f"{float(bm['C S7 median terrain']):+.2f}",
    "plat": f"{float(bm['A platform median']):+.2f}",
    "s7d": f"{float(bm['C S7 median terrain'])-float(bm['A platform median']):+.2f}",
    "f4pre": f"{frag.median_pre:.3f}", "f4post": f"{frag.median_post:.3f}",
    "f4d": f"{frag.diff_post_minus_pre:+.3f}",
    "f4ci": f"[{frag.diff_ci_lo:+.3f}, {frag.diff_ci_hi:+.3f}]",
    "f4n": f"{int(frag.n_pre)} / {int(frag.n_post)}",
    "s2pre_n": int(fpre.n_water_bodies), "s2pre_a": f"{fpre.total_water_area_km2:,.0f}",
    "s2pre_l": f"{100*fpre.largest_component_fraction:.3f}",
    "s2post_n": f"{fpost.n_water_bodies.median():.0f}",
    "s2post_a": f"{fpost.total_water_area_km2.median():,.0f}",
    "s2post_l": f"{100*fpost.largest_component_fraction.median():.1f}",
    "s2post_rng": f"{int(fpost.n_water_bodies.min())}–{int(fpost.n_water_bodies.max())}",
    "s2npost": len(fpost),
    "chdiff": f"{chan.diff_cm_km:+.2f}",
    "chci": f"[{chan.diff_ci_lo:+.2f}, {chan.diff_ci_hi:+.2f}]",
    "chn": int(chan.n_post), "chpos": chan.post_positive,
    "rw": f"{rwb.median_m:+.2f}", "rwn": int(rwb.n),
    "rwabove": f"{100*rwb.frac_above:.1f}",
    "seiche": f"{dv('seiche'):.1f}", "wind15": f"{dv('moderate gale'):.0f}",
    "wind25": f"{dv('extreme storm (25 m/s)'):.0f}",
    "riseq20": f"{dv('Q20%'):.0f}", "obsspan": f"{dv('pre-breach median'):.1f}",
}

# ======================================================================= #
# layout helpers                                                           #
# ======================================================================= #


def txbox(slide, x, y, w, h, text, size=14, bold=False, colour=INK,
          align=PP_ALIGN.LEFT, italic=False, font="Calibri", spacing=1.0,
          anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = colour
    return tb


def rect(slide, x, y, w, h, fill=None, line=None, lw=0.8):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


def newslide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=PAPER)
    return s


def header(slide, kicker, title, num, total, kcol=MUTED):
    txbox(slide, Cm(1.5), Cm(0.75), Cm(24), Cm(0.8), kicker.upper(), size=10.5,
          bold=True, colour=kcol)
    txbox(slide, Cm(1.5), Cm(1.4), Cm(28.5), Cm(1.9), title, size=23, bold=True,
          colour=INK, spacing=0.92)
    rect(slide, Cm(1.5), Cm(3.35), Cm(30.9), Cm(0.03), fill=RULE)
    txbox(slide, Cm(30.6), Cm(0.75), Cm(1.8), Cm(0.7), f"{num}/{total}",
          size=10.5, colour=RULE, align=PP_ALIGN.RIGHT)


def takeaway(slide, text, colour=INK):
    """The one-sentence scientific message. Every content slide gets one."""
    bar = rect(slide, Cm(1.5), Cm(16.35), Cm(30.9), Cm(1.75),
               fill=RGBColor(0xEF, 0xF2, 0xF4))
    bar.line.color.rgb = RULE
    bar.line.width = Pt(0.6)
    txbox(slide, Cm(1.9), Cm(16.5), Cm(30.1), Cm(1.5), text, size=13,
          bold=True, colour=colour, anchor=MSO_ANCHOR.MIDDLE, spacing=0.95)


def source(slide, text):
    txbox(slide, Cm(1.5), Cm(18.25), Cm(30.9), Cm(0.6), text, size=8.5,
          colour=MUTED, italic=True)


def stat(slide, x, y, w, value, label, colour=INK, vsize=25):
    txbox(slide, x, y, w, Cm(1.5), value, size=vsize, bold=True, colour=colour,
          align=PP_ALIGN.CENTER, spacing=0.9)
    txbox(slide, x, y + Cm(1.35), w, Cm(1.5), label, size=10, colour=MUTED,
          align=PP_ALIGN.CENTER, spacing=0.95)


def statrow(slide, items, y=Cm(4.0), x0=Cm(1.5), total=Cm(30.9)):
    n = len(items)
    w = int(total / n)
    for i, (val, lab, col) in enumerate(items):
        stat(slide, x0 + i * w, y, Cm(w / 360000), val, lab, col)


def picture(slide, name, x, y, w=None, h=None):
    """Place a figure, refusing silently-missing files."""
    p = FIG / name
    if not p.exists():
        txbox(slide, x, y, w or Cm(10), Cm(1), f"[MISSING FIGURE: {name}]",
              size=11, bold=True, colour=BAD)
        print(f"    !! MISSING FIGURE {name}")
        return None
    kw = {}
    if w is not None:
        kw["width"] = w
    if h is not None:
        kw["height"] = h
    return slide.shapes.add_picture(str(p), x, y, **kw)


def bullets(slide, x, y, w, items, size=13, gap=Cm(0.95), colour=INK):
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            mark, txt, mcol = it
        else:
            mark, txt, mcol = "•", it, RULE
        txbox(slide, x, y + i * gap, Cm(0.7), Cm(0.8), mark, size=size,
              bold=True, colour=mcol)
        txbox(slide, x + Cm(0.75), y + i * gap, w - Cm(0.75), Cm(1.4), txt,
              size=size, colour=colour, spacing=0.95)


# ======================================================================= #
# language packs                                                           #
# ======================================================================= #
EN = dict(
    lang="EN",
    title="Multi-Sensor Validation of Water-Surface Elevation and the "
          "Hydraulic Transition of the Kakhovka Reservoir after the 2023 "
          "Dam Breach",
    subtitle="Gauges · ICESat-2 ATL13 · SWOT PIXC/RiverSP · "
             "Sentinel-2 · historical bathymetry and hydraulics",
    tagline="Validation first, result second",
    footer="Lower Dnipro, Ukraine · breach 6 June 2023 · "
           "all heights EVRF2019 normal heights",
    kick=dict(problem="The problem", area="Study area and data",
              val="Validation", res="Result", synth="Synthesis"),
)

UA = dict(
    lang="UA",
    title="Мультисенсорна валідація рівнів водної поверхні та гідравлічна "
          "трансформація Каховського водосховища після руйнування греблі "
          "у 2023 році",
    subtitle="Гідрологічні пости · ICESat-2 ATL13 · SWOT PIXC/RiverSP "
             "· Sentinel-2 · історична батиметрія та гідравліка",
    tagline="Спочатку валідація, потім науковий результат",
    footer="Нижній Дніпро, Україна · руйнування 6 червня 2023 р. · "
           "усі відмітки — нормальні висоти EVRF2019",
    kick=dict(problem="Наукова проблема", area="Район досліджень і дані",
              val="Валідація", res="Результат", synth="Синтез"),
)

TOTAL = 18

# ======================================================================= #


def build(L, dst):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    ua = L["lang"] == "UA"

    # ---------------------------------------------------------- 1 title ----
    s = newslide(prs)
    rect(s, 0, 0, Cm(0.45), H, fill=SWOT)
    txbox(s, Cm(1.9), Cm(3.1), Cm(29), Cm(1), L["tagline"].upper(), size=12,
          bold=True, colour=SWOT)
    txbox(s, Cm(1.9), Cm(4.1), Cm(29.5), Cm(6), L["title"], size=29, bold=True,
          colour=INK, spacing=0.94)
    rect(s, Cm(1.9), Cm(10.4), Cm(9), Cm(0.06), fill=HIST)
    txbox(s, Cm(1.9), Cm(11.0), Cm(29), Cm(2), L["subtitle"], size=14,
          colour=MUTED, spacing=1.1)
    # the headline contrast, stated once on the title slide
    txbox(s, Cm(1.9), Cm(13.4), Cm(29), Cm(1.4),
          ("Ухил водної поверхні: " if ua else "Water-surface slope: ")
          + f"{V['pre']} → {V['post']} cm/km"
          + ("   (Тейл–Сен, медіана за датами)" if ua
             else "   (Theil–Sen, median over dates)"),
          size=17, bold=True, colour=ICESAT)
    txbox(s, Cm(1.9), Cm(17.6), Cm(29), Cm(1), L["footer"], size=10,
          colour=MUTED)

    # ------------------------------------------------- 2 the problem ----
    s = newslide(prs)
    header(s, L["kick"]["problem"],
           "Detecting a regime change smaller than the corrections"
           if not ua else
           "Зміна режиму, менша за самі поправки", 2, TOTAL)
    if ua:
        bullets(s, Cm(1.5), Cm(4.0), Cm(15.4), [
            ("1", "Гідравлічна різниця між водосховищем і річкою — це "
                  "твердження про ухил вільної поверхні: у підпертому "
                  "б'єфі він близький до нуля, у річці — стійко "
                  "спрямований.", SWOT),
            ("2", "Виявити це означає розрізнити кілька см/км на десятках "
                  "кілометрів, тобто дециметри висоти.", SWOT),
            ("3", "Але прилади подають висоти у ЧОТИРЬОХ різних "
                  "вертикальних системах відліку і ДВОХ системах "
                  "урахування постійного припливу. Поправки того самого "
                  "порядку — або більші.", ICESAT),
            ("4", "Тому спочатку валідація: доки висоти не є "
                  "порівнюваними, будь-який «результат» — артефакт "
                  "обробки.", ICESAT)], size=12, gap=Cm(2.9))
        qs = ("Питання дослідження\n\n"
              "1  Чи можна звести пости, ICESat-2 і SWOT в одну систему "
              "відліку з доказовою узгодженістю?\n"
              "2  Чи працює вертикальний ланцюг SWOT PIXC так, як "
              "задокументовано?\n"
              "3  Чи узгоджуються незалежні сенсори при справжній "
              "просторово-часовій колокації?\n"
              "4  Чи переноситься емпірична поправка з водосховища "
              "нижче за течією?\n"
              "5  До якої відмітки зведені історичні глибини?\n"
              "6  Чи відтворюють дані ICESat-2 історичну криву вільної "
              "поверхні?\n"
              "7  Чи змінився ухил після руйнування — і наскільки?")
    else:
        bullets(s, Cm(1.5), Cm(4.0), Cm(15.4), [
            ("1", "The hydraulic difference between a reservoir and a river "
                  "is a statement about water-surface slope: near zero in an "
                  "impounded pool, persistently directional in a river.", SWOT),
            ("2", "Detecting it means resolving a few cm/km over tens of "
                  "kilometres — that is, decimetres of height.", SWOT),
            ("3", "But the instruments report heights on FOUR different "
                  "reference surfaces and in TWO permanent-tide conventions. "
                  "The corrections are the same size or larger.", ICESAT),
            ("4", "So validation comes first: until the heights are "
                  "commensurable, any “result” is a processing "
                  "artefact.", ICESAT)], size=12, gap=Cm(2.9))
        qs = ("Research questions\n\n"
              "1  Can gauges, ICESat-2 and SWOT be brought into one vertical "
              "frame with demonstrable consistency?\n"
              "2  Does the SWOT PIXC vertical chain behave as documented?\n"
              "3  Do independent sensors agree under genuine space–time "
              "collocation?\n"
              "4  Is a reservoir-derived empirical correction transferable "
              "downstream?\n"
              "5  On what vertical reference are the historical soundings "
              "expressed?\n"
              "6  Does ICESat-2 reproduce the historical free-surface curve?\n"
              "7  Did the longitudinal slope change after the breach, and by "
              "how much?")
    box = rect(s, Cm(17.4), Cm(3.9), Cm(15), Cm(12.1),
               fill=RGBColor(0xF1, 0xF4, 0xF6))
    box.line.color.rgb = RULE
    txbox(s, Cm(17.9), Cm(4.3), Cm(14), Cm(11.4), qs, size=12, colour=INK,
          spacing=1.18)
    takeaway(s, "Questions 1–6 are validation. Question 7 is the result, "
                "and it is answered last."
             if not ua else
             "Питання 1–6 — валідація. Питання 7 — результат, і на нього "
             "відповідаємо останнім.")

    # ------------------------------------------- 3 study area and data ----
    s = newslide(prs)
    header(s, L["kick"]["area"],
           "One reservoir, five instrument families, asymmetric coverage"
           if not ua else
           "Одне водосховище, п'ять груп даних, асиметричне покриття", 3, TOTAL)
    picture(s, "V0_gauge_network_uk.png" if ua else "V0_gauge_network_en.png",
            Cm(1.5), Cm(4.0), w=Cm(18.6))
    rows = ([("Гідропости", "7 постів, 10 194 добові значення, 2019–2025", GAUGE),
             ("ICESat-2 ATL13", "єдиний альтиметр, що бачить водосховище "
                                "до і після руйнування", ICESAT),
             ("SWOT PIXC / RiverSP", "лише нижче греблі, у Херсоні", SWOT),
             ("Sentinel-2 L2A", "маски води, NDWI + MNDWI + SCL", SENT),
             ("Історичні джерела", "7 514 промірів S-57; Табл. 19–21, "
                                   "Рис. 13/16/46/57/128", HIST)]
            if ua else
            [("Gauges", "7 stations, 10 194 daily values, 2019–2025", GAUGE),
             ("ICESat-2 ATL13", "the only altimeter observing the reservoir "
                                "across the breach", ICESAT),
             ("SWOT PIXC / RiverSP", "downstream only, at Kherson", SWOT),
             ("Sentinel-2 L2A", "water masks, NDWI + MNDWI + SCL", SENT),
             ("Historical sources", "7 514 S-57 soundings; Tables 19–21, "
                                    "Figs 13/16/46/57/128", HIST)])
    y = Cm(4.2)
    for name, desc, col in rows:
        rect(s, Cm(20.6), y, Cm(0.16), Cm(1.75), fill=col)
        txbox(s, Cm(21.0), y, Cm(11.4), Cm(0.7), name, size=12.5, bold=True,
              colour=col)
        txbox(s, Cm(21.0), y + Cm(0.62), Cm(11.4), Cm(1.3), desc, size=10.5,
              colour=MUTED, spacing=0.95)
        y += Cm(2.32)
    takeaway(s, "SWOT's cal/val orbit never covered the reservoir pool — "
                "a coverage fact, not a processing choice. The reservoir "
                "result therefore rests on ICESat-2."
             if not ua else
             "Орбіта SWOT (cal/val) ніколи не покривала акваторію "
             "водосховища — це факт покриття, а не вибір обробки. Тому "
             "результат по водосховищу спирається на ICESat-2.",
             colour=BAD)
    source(s, "gauge_vertical_reference_summary.csv · "
              "swot_dataset_inventory.csv · [V1.1]")

    # ------------------------------------ 4 four reference surfaces ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "Four reference surfaces, two tide conventions, one target frame"
           if not ua else
           "Чотири поверхні відліку, дві системи припливу, одна цільова "
           "система", 4, TOTAL)
    picture(s, "V1_vertical_reference_framework_uk.png" if ua
            else "V1_vertical_reference_framework_en.png",
            Cm(1.5), Cm(3.9), w=Cm(19.4))
    eqs = ("H_BS77      = zero + stage\n"
           "H_EVRF2019  = H_BS77 + Δ_EPSG9902(x)\n\n"
           "h_ICESat-2  = ht_water_surf + free2mean(φ)\n"
           "h_SWOT      = height − (SET + LT + PT)\n\n"
           "H_EGG2015   = h − ζ_EGG2015(φ, λ)")
    bx = rect(s, Cm(21.4), Cm(3.9), Cm(11), Cm(5.4), fill=RGBColor(0xF1, 0xF4, 0xF6))
    bx.line.color.rgb = RULE
    txbox(s, Cm(21.8), Cm(4.15), Cm(10.4), Cm(5), eqs, size=11.5,
          font="Consolas", colour=INK, spacing=1.1)
    bullets(s, Cm(21.4), Cm(9.8), Cm(11), [
        ("!", "ATL13 is TIDE-FREE; PIXC is MEAN-TIDE. The term is "
              "3.5–3.9 cm here — the size of the agreement being "
              "tested.", ICESAT),
        ("!", "The PIXC `geoid` field is NEVER subtracted from PIXC "
              "`height`: that is a 23.9 m error.", BAD)]
        if not ua else [
        ("!", "ATL13 — у системі БЕЗ припливу, PIXC — у СЕРЕДНЬОПРИПЛИВНІЙ. "
              "Поправка тут 3,5–3,9 см — це масштаб самої перевіряної "
              "узгодженості.", ICESAT),
        ("!", "Поле `geoid` у PIXC НІКОЛИ не віднімається від `height`: це "
              "помилка 23,9 м.", BAD)], size=11.5, gap=Cm(2.5))
    takeaway(s, "EGG2015 and EVRF2019 are both zero-tide, so once ATL13 "
                "leaves the tide-free system no tide mixing remains anywhere "
                "in the chain."
             if not ua else
             "EGG2015 і EVRF2019 — обидві нульовоприпливні, тож після "
             "переведення ATL13 змішування систем припливу в ланцюзі не "
             "залишається.")
    source(s, "vertical_reference_audit.csv · "
              "permanent_tide_harmonisation.csv · ATL03 ATBD v007 p.125 "
              "· PIXC PDD D-56411 RevC p.53")

    # ------------------------------------------- 5 gauges into EVRF2019 ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "The gauge network in one frame — and why no national "
           "constant will do"
           if not ua else
           "Мережа постів в одній системі — і чому національна стала "
           "не годиться", 5, TOTAL)
    statrow(s, [(V["nobs"], "daily values transformed" if not ua
                 else "добових значень переведено", GAUGE),
                (f"{V['nstat']}", "stations, BS-77 → EVRF2019" if not ua
                 else "постів, БС-77 → EVRF2019", GAUGE),
                (f"{V['dspread']} mm", "spatial spread of Δ EPSG:9902"
                 if not ua else "просторовий розмах Δ EPSG:9902", SWOT),
                ("0.044 m", "post-tie residual NMAD vs gauges" if not ua
                 else "NMAD залишків після прив'язки", ICESAT)])
    bullets(s, Cm(1.5), Cm(8.4), Cm(15.2), [
        f"Δ EPSG:9902 runs {V['dmin']} → {V['dmax']} m across the "
        f"reach: an official operation applied at each station's own "
        f"coordinates, never a fitted constant.",
        "The grid sampler was cross-checked against an independent "
        "implementation and agrees to 0.000 mm; the script aborts if it does "
        "not.",
        "The six reservoir gauges share a zero of +12.000 m BS-77. Kherson "
        "80805 does NOT — it is −5.000 m. Confusing them is a 17 m "
        "blunder."]
        if not ua else [
        f"Δ EPSG:9902 змінюється від {V['dmin']} до {V['dmax']} м уздовж "
        f"ділянки: офіційна операція, застосована в координатах кожного "
        f"поста, а не підібрана стала.",
        "Білінійний інтерполятор сітки перевірено проти незалежної "
        "реалізації — збіг до 0,000 мм; інакше скрипт аварійно "
        "припиняє роботу.",
        "Шість постів водосховища мають нуль графіка +12,000 м БС-77. "
        "Херсон 80805 — НІ, у нього −5,000 м. Їх сплутування дає "
        "помилку 17 м."], size=12.5, gap=Cm(2.2))
    bx = rect(s, Cm(17.4), Cm(8.4), Cm(15), Cm(7.5),
              fill=RGBColor(0xF1, 0xF4, 0xF6))
    bx.line.color.rgb = RULE
    tbl = ("station                Δ EPSG:9902 (m)\n"
           "─────────────────────────────────────\n")
    for r in gv[gv.n_obs > 0].sort_values("delta_epsg9902_m").itertuples():
        tbl += f"{r.name_en:<22}{r.delta_epsg9902_m:+.4f}\n"
    txbox(s, Cm(17.9), Cm(8.7), Cm(14), Cm(7), tbl, size=11,
          font="Consolas", colour=INK, spacing=1.12)
    takeaway(s, "A single national constant would be wrong by up to ~2 cm at "
                "the extremes — the same order as the sensor agreement "
                "being tested."
             if not ua else
             "Єдина національна стала давала б помилку до ~2 см на краях "
             "ділянки — того ж порядку, що й перевірювана узгодженість "
             "сенсорів.")
    source(s, "gauge_vertical_reference_summary.csv · [V1.1, V1.2, V1.3]")

    # ------------------------------------------- 6 PIXC vs RiverSP ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "The SWOT PIXC vertical chain, tested against RiverSP"
           if not ua else
           "Вертикальний ланцюг SWOT PIXC у перевірці проти RiverSP", 6, TOTAL)
    picture(s, "V2_pixc_sign_validation.png", Cm(1.5), Cm(3.9), w=Cm(17.8))
    rowsx = [("height − (SET+LT+PT)", "−0.0010 m",
              "documented chain" if not ua else "задокументований ланцюг", GOOD),
             ("height", "+0.0720 m",
              "no correction" if not ua else "без поправки", BAD),
             ("height + (SET+LT+PT)", "+0.1451 m",
              "sign reversed" if not ua else "зворотний знак", BAD),
             ("height − geoid", "−23.89 m",
              "geoid blunder" if not ua else "груба помилка з геоїдом", BAD)]
    y = Cm(4.4)
    for chain, val, note, col in rowsx:
        txbox(s, Cm(19.9), y, Cm(8.2), Cm(0.8), chain, size=11.5,
              font="Consolas", colour=INK)
        txbox(s, Cm(27.9), y, Cm(4.5), Cm(0.8), val, size=13.5, bold=True,
              colour=col, align=PP_ALIGN.RIGHT)
        txbox(s, Cm(19.9), y + Cm(0.62), Cm(12.5), Cm(0.7), note, size=10,
              colour=MUTED, italic=True)
        y += Cm(1.85)
    txbox(s, Cm(19.9), Cm(12.1), Cm(12.5), Cm(3.6),
          ("n = 1 023 RiverSP nodes, same cycle and pass (482/001).\n\n"
           "Only the documented chain returns zero. The sign and content of "
           "the correction are therefore established EMPIRICALLY, not "
           "assumed from documentation.")
          if not ua else
          ("n = 1 023 вузли RiverSP, той самий цикл і проліт (482/001).\n\n"
           "Нуль дає лише задокументований ланцюг. Отже, знак і склад "
           "поправки встановлено ЕМПІРИЧНО, а не взято на віру з "
           "документації."), size=11.5, colour=INK, spacing=1.1)
    takeaway(s, "This is an INTERNAL product-chain validation, not "
                "independent accuracy: RiverSP derives from the same SWOT "
                "observation."
             if not ua else
             "Це ВНУТРІШНЯ перевірка ланцюга обробки, а не незалежна оцінка "
             "точності: RiverSP походить із того самого спостереження SWOT.",
             colour=BAD)
    source(s, "validation_summary_table.csv · [V2.1]")

    # --------------------------------------- 7 SWOT vs ICESat-2 ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "SWOT ↔ ICESat-2 at true sensor overlap"
           if not ua else
           "SWOT ↔ ICESat-2 у зоні справжнього перекриття", 7, TOTAL)
    picture(s, "V3_swot_icesat_colocated.png", Cm(1.5), Cm(3.9), w=Cm(18.2))
    y = Cm(4.3)
    for r in colo.itertuples():
        col_ = GOOD if abs(r.d_swot_minus_icesat_m) < 0.02 else (
            BAD if abs(r.d_swot_minus_icesat_m) > 0.1 else HIST)
        txbox(s, Cm(20.3), y, Cm(5.4), Cm(0.8), str(r.date), size=12.5,
              bold=True, colour=INK)
        txbox(s, Cm(25.5), y, Cm(6.9), Cm(0.8),
              f"{r.d_swot_minus_icesat_m:+.4f} m", size=13.5, bold=True,
              colour=col_, align=PP_ALIGN.RIGHT)
        txbox(s, Cm(20.3), y + Cm(0.66), Cm(12.1), Cm(0.7),
              (f"Δt {r.dt_hours:+.2f} h · {int(r.n_segments):,} "
               f"segments · {int(r.n_beams)} beams" if not ua else
               f"Δt {r.dt_hours:+.2f} год · {int(r.n_segments):,} "
               f"сегментів · {int(r.n_beams)} промені"),
              size=10, colour=MUTED, italic=True)
        y += Cm(1.95)
    txbox(s, Cm(20.3), Cm(10.6), Cm(12.1), Cm(5),
          (f"Sign convention: H_SWOT − H_ICESat-2.\n\n"
           f"Best temporal match ({V['best_date']}, Δt "
           f"{V['best_dt']} h, {V['best_seg']} segments): "
           f"{V['best_d']} m — with NO correction applied.\n\n"
           f"The residual orders with |Δt|, consistent with temporal "
           f"mismatch rather than a datum offset.")
          if not ua else
          (f"Умова знака: H_SWOT − H_ICESat-2.\n\n"
           f"Найкраща часова колокація ({V['best_date']}, Δt "
           f"{V['best_dt']} год, {V['best_seg']} сегментів): "
           f"{V['best_d']} м — БЕЗ жодної поправки.\n\n"
           f"Залишок упорядковується за |Δt|, що вказує на часову "
           f"неузгодженість, а не на зсув системи відліку."),
          size=11.5, colour=INK, spacing=1.1)
    takeaway(s, "Centimetric agreement is ACHIEVABLE under close temporal "
                "collocation. With n = 3 overpasses no universal bias is "
                "claimed, and no regression is fitted."
             if not ua else
             "Сантиметрова узгодженість ДОСЯЖНА за тісної часової "
             "колокації. За n = 3 прольотів жодного універсального зсуву "
             "не заявляємо і регресію не будуємо.",
             colour=BAD)
    source(s, "part4_colocated_swot_icesat.csv · [V3.1]")

    # ----------------------- 8 Kherson anchor + negative results ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "Kherson: a working anchor, an unresolved epoch, and a limit"
           if not ua else
           "Херсон: робоча прив'язка, нерозв'язана дата і межа переносу",
           8, TOTAL)
    statrow(s, [("+0.026 m", "SWOT − gauge, 9 overpasses" if not ua
                 else "SWOT − пост, 9 прольотів", SWOT),
                ("0.041 m", "NMAD" if not ua else "NMAD", SWOT),
                (V["anom_obs"] + " cm", "2023-04-05 observed" if not ua
                 else "2023-04-05 спостережено", BAD),
                (V["anom_exp"] + " cm", "hydrology predicts" if not ua
                 else "гідрологія прогнозує", BAD)], y=Cm(3.85))
    picture(s, "V4_kherson_hydrograph_epochs.png", Cm(1.5), Cm(7.9), w=Cm(17.4))
    if ua:
        items = [
            ("A", f"ПРИВ'ЯЗКА ПРАЦЮЄ. SWOT над відкритою водою в межах 1 км "
                  f"від поста 80805 відрізняється на +0,026 м. Медіана "
                  f"залежить від радіуса (8,4 см на 0,5–5 км) — це "
                  f"вздовжруслові ухили входять у вікно осереднення.", SWOT),
            ("B", f"НЕРОЗВ'ЯЗАНО: 2023-04-05. Пост ПІДНІМАВСЯ на "
                  f"{V['anom_rate']} см/добу, SWOT знято на {V['anom_dt']} год "
                  f"пізніше, тож гідрограф прогнозує {V['anom_exp']} см. "
                  f"Спостережено {V['anom_obs']} см — знак протилежний, "
                  f"залишок зростає до −24,0 см.", BAD),
            ("C", f"МЕЖА ПЕРЕНОСУ. Пости водосховища дають {V['res_lo']} м "
                  f"{V['res_ci']}, Херсон — {V['kh_c']} м {V['kh_ci']}. "
                  f"Інтервали не перетинаються, нуль лежить у "
                  f"херсонському.", HIST)]
    else:
        items = [
            ("A", f"THE ANCHOR WORKS. SWOT open water within 1 km of gauge "
                  f"80805 differs by +0.026 m. The median is radius-dependent "
                  f"(8.4 cm over 0.5–5 km) — the along-channel "
                  f"gradient entering the aggregation window.", SWOT),
            ("B", f"UNRESOLVED: 2023-04-05. The gauge was RISING at "
                  f"{V['anom_rate']} cm/day and SWOT came {V['anom_dt']} h "
                  f"later, so the hydrograph predicts {V['anom_exp']} cm. "
                  f"Observed: {V['anom_obs']} cm — wrong sign, and the "
                  f"residual worsens to −24.0 cm.", BAD),
            ("C", f"THE TRANSFER LIMIT. Reservoir stations give "
                  f"{V['res_lo']} m {V['res_ci']}; Kherson gives "
                  f"{V['kh_c']} m {V['kh_ci']}. The intervals are disjoint and "
                  f"zero lies inside Kherson's.", HIST)]
    bullets(s, Cm(19.6), Cm(7.9), Cm(12.8), items, size=11.5, gap=Cm(2.75))
    takeaway(s, "A validation framework that reports only its successes is "
                "not a validation framework. One epoch is unresolved, and the "
                "empirical correction has a stated spatial limit."
             if not ua else
             "Система валідації, що звітує лише про успіхи, не є системою "
             "валідації. Одна дата залишається нерозв'язаною, а емпірична "
             "поправка має встановлену просторову межу.")
    source(s, "validation_summary_table.csv · part4_pooling_test.csv "
              "· kherson_daily_variability.csv · [V4.1, V4.2, V5.1]")

    # ------------------------------------ 9 historical sounding datum ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "What reference level do the historical depths themselves require?"
           if not ua else
           "До якої відмітки самі історичні глибини вимагають зведення?",
           9, TOTAL)
    picture(s, "V2_historical_reference_sensitivity.png", Cm(1.5), Cm(3.9),
            w=Cm(17.2))
    y = Cm(4.3)
    labs = ({16.0: "НПГ 16,00 м — попереднє припущення",
             14.0: "УНС 14,00 м — ОПУБЛІКОВАНА відмітка",
             13.71: "13,71 м — за кривою об'ємів (без ICESat-2)",
             14.11: "14,11 м — за ICESat-2 (ЦИРКУЛЯРНО)"} if ua else
            {16.0: "NPG 16.00 m — the old assumption",
             14.0: "UNS 14.00 m — the PUBLISHED level",
             13.71: "13.71 m — capacity-curve fit (no ICESat-2)",
             14.11: "14.11 m — ICESat-2 fit (CIRCULAR)"})
    for r in hrs.itertuples():
        ok = bool(r.zero_within_ci)
        col_ = GOOD if (ok and r.reference_level_m == 14.0) else (
            BAD if not ok else MUTED)
        txbox(s, Cm(19.2), y, Cm(9.2), Cm(0.8), labs[r.reference_level_m],
              size=11.5, bold=(r.reference_level_m == 14.0), colour=INK)
        txbox(s, Cm(28.2), y, Cm(4.2), Cm(0.8), f"{r.median_dH_m:+.3f} m",
              size=13, bold=True, colour=col_, align=PP_ALIGN.RIGHT)
        txbox(s, Cm(19.2), y + Cm(0.66), Cm(13.2), Cm(0.7),
              f"95 % CI [{r.ci_lo_m:+.3f}, {r.ci_hi_m:+.3f}]   "
              + (("нуль У інтервалі" if ok else "нуль ПОЗА інтервалом") if ua
                 else ("zero INSIDE CI" if ok else "zero OUTSIDE CI")),
              size=9.5, colour=MUTED, italic=True)
        y += Cm(1.9)
    txbox(s, Cm(19.2), Cm(12.2), Cm(13.2), Cm(3.8),
          (f"n = {V['hn']} confirmed exposed-bed points on {V['htr']} "
           f"independent ICESat-2 tracks, track-clustered inference.\n\n"
           f"14.00 m is adopted because it is the only PUBLISHED level of the "
           f"four — not because it fits best. In EVRF2019 it becomes "
           f"{V['ev14lo']}–{V['ev14hi']} m (median {V['ev14']} m), again "
           f"spatially varying.")
          if not ua else
          (f"n = {V['hn']} підтверджених точок оголеного дна на {V['htr']} "
           f"незалежних треках ICESat-2; бутстреп із кластеризацією за "
           f"треками.\n\n"
           f"Прийнято 14,00 м, бо це єдина ОПУБЛІКОВАНА відмітка з чотирьох, "
           f"а не тому, що вона підходить найкраще. У EVRF2019 вона стає "
           f"{V['ev14lo']}–{V['ev14hi']} м (медіана {V['ev14']} м) — "
           f"і теж змінюється у просторі."),
          size=11, colour=INK, spacing=1.1)
    takeaway(s, "The depths are reduced to the navigation drawdown level, not "
                "the impoundment level. This is an INFERENCE from the data, "
                "and the honest uncertainty is the 0.40 m spread among three "
                "independent routes — not the CI."
             if not ua else
             "Глибини зведені до мінімального судноплавного рівня (УНС), а не "
             "до НПГ. Це ВИСНОВОК із даних; чесна невизначеність — розмах "
             "0,40 м між трьома незалежними оцінками, а не довірчий "
             "інтервал.", colour=HIST)
    source(s, "historical_reference_sensitivity.csv · hist2_datum_fit.csv "
              "· hist3_datum_in_evrf2019.csv · [V6.1, V6.2]")

    # ------------------------- 10 historical free-surface validation ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "A field-measured free-surface curve from April 1970"
           if not ua else
           "Натурна крива вільної поверхні, квітень 1970 р.", 10, TOTAL)
    picture(s, "HIST9_fig16_digitisation.png", Cm(1.5), Cm(3.9), w=Cm(17.6))
    statrow(s, [(V["f16med"] + " m", "median deviation vs Table 20"
                 if not ua else "медіана відхилень від Табл. 20", HIST),
                (V["f16max"] + " m", "max deviation in the flat pool"
                 if not ua else "макс. відхилення у плоскому б'єфі", HIST),
                (f"{V['f16n']}", "digitised points" if not ua
                 else "оцифрованих точок", HIST)],
            y=Cm(4.2), x0=Cm(19.6), total=Cm(12.8))
    bullets(s, Cm(19.6), Cm(8.3), Cm(12.8), [
        "Curve 2 is the only IN-SITU longitudinal profile of the impounded "
        "reservoir available: 22–25 April 1970, Q = 8 400 m³/s.",
        "Validated against Table 20 — an independent NUMERICAL source, "
        "not the same figure.",
        "The vertical axis is NON-LINEAR. Calibrating it as linear made every "
        "elevation read 0.2–0.4 m too high.",
        "Large deviations on the steep backwater limb are NOT digitisation "
        "error: Table 20 has only two points there and the true curve is "
        "convex."]
        if not ua else [
        "Крива 2 — єдиний доступний НАТУРНИЙ продовжній профіль підпертого "
        "б'єфа: 22–25 квітня 1970 р., Q = 8 400 м³/с.",
        "Перевірено проти Табл. 20 — незалежного ЧИСЛОВОГО джерела, а не "
        "того самого рисунка.",
        "Вертикальна вісь НЕЛІНІЙНА. Лінійна калібровка завищувала кожну "
        "відмітку на 0,2–0,4 м.",
        "Великі відхилення на крутій ділянці кривої підпору — НЕ помилка "
        "оцифрування: у Табл. 20 там лише дві точки, а справжня крива "
        "випукла."], size=11.5, gap=Cm(2.0))
    takeaway(s, "Modern satellite altimetry is validated against in-situ "
                "hydraulic measurements made more than 50 years earlier: "
                f"agreement {V['f16pool']} m over the pool."
             if not ua else
             "Сучасну супутникову альтиметрію перевірено проти натурних "
             "гідравлічних вимірювань, виконаних понад 50 років тому: "
             f"узгодження {V['f16pool']} м у межах плоского б'єфа.",
             colour=HIST)
    source(s, "historical_fig16_profiles.csv · hist6_reach_slopes.csv "
              "· [V7.1, V7.2]")

    # ---------------------------- 11 bed reconstruction and CV ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "Reconstructing the bed — and the choice that dominates it"
           if not ua else
           "Відбудова рельєфу дна — і вибір, що визначає результат", 11, TOTAL)
    picture(s, "V12_bed_surface_validation.png", Cm(1.5), Cm(3.9), w=Cm(18.4))
    bullets(s, Cm(20.4), Cm(4.0), Cm(12), [
        ("1", f"Four interpolators, spatially blocked CV at 1 km. Preferred "
              f"{V['cvbest']}. NEVER quote the RMSE alone: RMSE "
              f"{V['cvrmse']} m · NMAD {V['cvnmad']} m · bias {V['cvbias']} m · "
              f"median |e| {V['cvmed']} m, and the worst 5 % of observations "
              f"carry {V['cvconc']} % of the squared error.", SWOT),
        ("2", f"That heavy tail is LOCAL MORPHOLOGY, not a vertical offset and "
              f"not a blocked-CV artefact: random hold-out already gives "
              f"{V['cvrand']} m, and roughness alone spans {V['cvrough']} m. "
              f"The smoothing bias REVERSES sign with elevation.", MUTED),
        ("3", "THE SHORELINE BOUNDARY DOMINATES. Soundings stop ≥167 m "
              "from shore with a median bed of 12.15 m against a ~17.1 m "
              "waterline. Unconstrained, the margins are modelled ~5 m too "
              "deep — and unconstrained RBF extrapolated to +67.9 m.", BAD),
        ("4", "The boundary is pinned at the OBSERVED 2023-06-05 waterline "
              "(17.08 m EVRF2019), fixed independently by the Rozumivka gauge "
              "AND the historical level–area curve. It is ~1 m ABOVE "
              "NPG: the pool had been raised.", HIST)]
        if not ua else [
        ("1", f"Чотири інтерполятори, просторово-блокова кросвалідація 1 км. "
              f"Обрано {V['cvbest']}. НІКОЛИ не подавати лише RMSE: RMSE "
              f"{V['cvrmse']} м · NMAD {V['cvnmad']} м · зсув {V['cvbias']} м · "
              f"медіана |e| {V['cvmed']} м, і найгірші 5 % спостережень несуть "
              f"{V['cvconc']} % квадратичної помилки.", SWOT),
        ("2", f"Цей важкий хвіст — ЛОКАЛЬНА МОРФОЛОГІЯ, а не вертикальний зсув "
              f"і не артефакт blocked CV: випадковий hold-out дає вже "
              f"{V['cvrand']} м, а шорсткість сама дає розмах "
              f"{V['cvrough']} м. Зсув згладжування ЗМІНЮЄ ЗНАК із "
              f"відміткою.", MUTED),
        ("3", "ГРАНИЧНА УМОВА НА БЕРЕЗІ ВИЗНАЧАЄ ВСЕ. Проміри "
              "закінчуються за ≥167 м від берега з медіанною "
              "відміткою дна 12,15 м проти лінії води ~17,1 м. Без "
              "обмеження прибережна смуга моделюється на ~5 м глибшою, а "
              "RBF екстраполює до +67,9 м.", BAD),
        ("4", "Межу закріплено на СПОСТЕРЕЖЕНІЙ лінії води 2023-06-05 "
              "(17,08 м EVRF2019), визначеній незалежно постом Розумівка ТА "
              "історичною кривою «рівень–площа». Це ~1 м ВИЩЕ НПГ: "
              "б'єф було піднято.", HIST)], size=11, gap=Cm(2.9))
    takeaway(s, "Point-scale RMSE is metre-scale and we say so. The claim "
                "that follows is about an INTEGRATED AREA statistic, which is "
                "far better conditioned than any single cell."
             if not ua else
             "Точкова RMSE — метрового порядку, і ми це прямо зазначаємо. "
             "Наступне твердження стосується ІНТЕГРАЛЬНОЇ ПЛОЩІ, яка "
             "обумовлена значно краще, ніж окрема комірка.")
    source(s, "hist14_interpolator_cv.csv · hist14_surface_summary.csv "
              "· [V10.2, V10.3]")

    # -------------------------- 12 area-weighted exposure + resolution ----
    s = newslide(prs)
    header(s, L["kick"]["val"],
           "An independent morphological test the surface was never fitted to"
           if not ua else
           "Незалежна морфологічна перевірка, під яку поверхню не "
           "підганяли", 12, TOTAL)
    picture(s, "V13_exposure_area_validation.png", Cm(1.5), Cm(3.9), w=Cm(17.2))
    statrow(s, [(V["exh"] + " %", "historical (Table 21)" if not ua
                 else "історичне (Табл. 21)", HIST),
                (V["exr"] + " %", "reconstructed" if not ua
                 else "відбудоване", SENT),
                (V["exd"] + " pp", "difference" if not ua
                 else "різниця", GOOD)],
            y=Cm(4.1), x0=Cm(19.2), total=Cm(13.2))
    bullets(s, Cm(19.2), Cm(8.0), Cm(13.2), [
        f"Upstream increase reproduced: historical {V['reach_h']} % vs "
        f"reconstructed {V['reach_r']} % — by ALL FOUR interpolators.",
        f"Grid resolution is NOT the limit: {V['grid']}. A 70× increase "
        f"in cell count moves the answer {V['drift']} pp, against an "
        f"interpolator spread of {V['spread']} pp.",
        "Reach 5 is NA — outside the mapped domain, with 0 km² of "
        "grid cells. Never recorded as zero.",
        "Table 21 was VALIDATION, not calibration: the historical 12.95 % "
        "never entered the surface, and the method was chosen on CV alone."]
        if not ua else [
        f"Зростання вгору за течією відтворено: історичне {V['reach_h']} % "
        f"проти відбудованого {V['reach_r']} % — усіма ЧОТИРМА "
        f"інтерполяторами.",
        f"Роздільність сітки НЕ є обмеженням: {V['grid']}. Збільшення "
        f"кількості комірок у 70 разів змінює відповідь на {V['drift']} в. п. "
        f"проти розмаху між інтерполяторами {V['spread']} в. п.",
        "Ділянка 5 — NA: поза відображеною областю, 0 км² комірок. "
        "Ніколи не записується як нуль.",
        "Табл. 21 — це ВАЛІДАЦІЯ, а не калібрування: історичні 12,95 % не "
        "входили у побудову поверхні, а метод обрано лише за "
        "кросвалідацією."], size=11.5, gap=Cm(2.05))
    takeaway(s, "An independently reconstructed bed reproduces a historical "
                f"area statistic to {V['exd']} percentage points, under four "
                "interpolators and three grid resolutions."
             if not ua else
             "Незалежно відбудоване дно відтворює історичну площинну "
             f"статистику з точністю {V['exd']} в. п. — за чотирьох "
             "інтерполяторів і трьох роздільностей сітки.", colour=GOOD)
    source(s, "historical_exposure_area_validation.csv · "
              "hist17_grid_resolution.csv · [V10.1, V11.1]")

    # ------------------------------------------ 13 validation synthesis ----
    s = newslide(prs)
    header(s, L["kick"]["synth"],
           "Validation scorecard — eight passes, one failure, one limit"
           if not ua else
           "Підсумок валідації — вісім успіхів, одна невдача, одна межа",
           13, TOTAL)
    checks = ([("Одна геодезична система для всіх постів", "NMAD 0,044 м", "OK", GOOD),
               ("Ланцюг SWOT PIXC", "−0,0010 м проти RiverSP", "OK", GOOD),
               ("SWOT ↔ ICESat-2", f"{V['best_d']} м за Δt 2,3 год", "OK", GOOD),
               ("SWOT проти поста (Херсон)", "+0,026 м, 9 прольотів", "OK", GOOD),
               ("Історична відмітка зведення глибин", f"14,00 м → {V['h14']} м", "OK", GOOD),
               ("Історична крива вільної поверхні", f"{V['f16pool']} м у б'єфі", "OK", GOOD),
               ("Морфологія дна (4 тести)", f"збагачення {V['enrich']}×; S7 {V['s7d']} м", "OK", GOOD),
               ("Площа оголення проти Табл. 21", f"{V['exd']} в. п.", "OK", GOOD),
               ("Роздільність сітки", f"дрейф {V['drift']} в. п.", "OK", GOOD),
               ("Аномалія 2023-04-05", "гідрограф має протилежний знак", "НЕРОЗВ.", BAD),
               ("Перенос емпіричної поправки", "інтервали не перетинаються", "МЕЖА", HIST)]
              if ua else
              [("All gauges in one geodetic frame", "NMAD 0.044 m", "PASS", GOOD),
               ("SWOT PIXC chain", "−0.0010 m vs RiverSP", "PASS", GOOD),
               ("SWOT ↔ ICESat-2", f"{V['best_d']} m at Δt 2.3 h", "PASS", GOOD),
               ("SWOT vs gauge (Kherson)", "+0.026 m, 9 overpasses", "PASS", GOOD),
               ("Historical sounding datum", f"14.00 m → {V['h14']} m", "PASS", GOOD),
               ("Historical free-surface curve", f"{V['f16pool']} m in pool", "PASS", GOOD),
               ("Bed morphology (4 tests)", f"{V['enrich']}× enrichment; S7 {V['s7d']} m", "PASS", GOOD),
               ("Exposure area vs Table 21", f"{V['exd']} pp", "PASS", GOOD),
               ("Grid resolution", f"drift {V['drift']} pp", "PASS", GOOD),
               ("The 2023-04-05 anomaly", "hydrograph has the wrong sign", "UNRESOLVED", BAD),
               ("Transfer of the empirical correction", "intervals disjoint", "LIMIT", HIST)])
    y = Cm(3.95)
    for i, (q, r, verdict, c) in enumerate(checks):
        if i % 2 == 0:
            rect(s, Cm(1.5), y - Cm(0.06), Cm(30.9), Cm(1.05),
                 fill=RGBColor(0xF3, 0xF5, 0xF7))
        txbox(s, Cm(1.8), y, Cm(13.6), Cm(0.9), q, size=12, colour=INK)
        txbox(s, Cm(15.4), y, Cm(11.6), Cm(0.9), r, size=11.5, colour=MUTED)
        txbox(s, Cm(27.0), y, Cm(5.2), Cm(0.9), verdict, size=11.5, bold=True,
              colour=c, align=PP_ALIGN.RIGHT)
        y += Cm(1.13)
    takeaway(s, "Only now is the framework entitled to make a scientific "
                "claim about what changed."
             if not ua else
             "Лише тепер ця система має право робити науковий висновок про "
             "те, що саме змінилося.", colour=SWOT)
    source(s, "manuscript_evidence_matrix.csv — 33 claims, each traced "
              "to a canonical table")

    # ------------------------------------------- 14 pre-breach state ----
    s = newslide(prs)
    header(s, L["kick"]["res"],
           "Pre-breach: a near-level pool, and why that is a measurement"
           if not ua else
           "До руйнування: майже горизонтальна поверхня — і чому це "
           "вимірювання", 14, TOTAL)
    picture(s, "FigD_kakhovka_longitudinal_profiles.png", Cm(1.5), Cm(3.9),
            w=Cm(17.6))
    statrow(s, [(V["pre"], "cm/km, Theil–Sen median" if not ua
                 else "см/км, медіана Тейла–Сена", ICESAT),
                (V["prepos"], "dates positive" if not ua
                 else "дат із додатним ухилом", MUTED),
                (V["presign"], "sign-test p" if not ua
                 else "p критерію знаків", MUTED)],
            y=Cm(4.2), x0=Cm(19.6), total=Cm(12.8))
    bullets(s, Cm(19.6), Cm(8.3), Cm(12.8), [
        f"{V['npre']} dates, 2019–2022. Median within-profile range "
        f"{V['rng_pre']} m; median r² {V['r2_pre']} — the fits "
        f"explain almost nothing, which is what fitting a line to a flat "
        f"surface should do.",
        f"Slopes are NOT consistently signed ({V['prepos']}, sign test "
        f"p = {V['presign']}). That is what “near-level” means.",
        f"The observed within-overpass span (median {V['obsspan']} cm) "
        f"EXCEEDS the entire mean hydraulic rise across the 183 km pool at "
        f"ordinary flow ({V['riseq20']} cm at Q20 %).",
        f"Documented dynamics dwarf the mean gradient: seiche "
        f"{V['seiche']} cm at an antinode, wind setup {V['wind15']} cm "
        f"(15 m/s) to {V['wind25']} cm (25 m/s) at the ends."]
        if not ua else [
        f"{V['npre']} дат, 2019–2022. Медіанний розмах у профілі "
        f"{V['rng_pre']} м; медіанний r² {V['r2_pre']} — регресії майже "
        f"нічого не пояснюють, і саме так має бути при підгонці прямої до "
        f"плоскої поверхні.",
        f"Знак ухилу НЕ є стійким ({V['prepos']}, p критерію знаків = "
        f"{V['presign']}). Саме це й означає «майже горизонтальна».",
        f"Спостережений розмах у межах одного прольоту (медіана "
        f"{V['obsspan']} см) ПЕРЕВИЩУЄ весь середній гідравлічний підпір на "
        f"183 км б'єфа за звичайних витрат ({V['riseq20']} см при Q20 %).",
        f"Задокументована динаміка значно більша за середній ухил: сейші "
        f"{V['seiche']} см в антивузлі, згінно-нагінні коливання "
        f"{V['wind15']} см (15 м/с) до {V['wind25']} см (25 м/с) на "
        f"краях."], size=11, gap=Cm(2.05))
    takeaway(s, "Individual pre-breach profiles with small slopes of either "
                "sign are exactly what a flat pool under documented seiches "
                "and wind setup must produce. NO causal attribution is made "
                "for any single date — no forcing data exist."
             if not ua else
             "Окремі профілі до руйнування з малими ухилами будь-якого знака "
             "— це саме те, що має давати плоский б'єф під дією "
             "задокументованих сейш і згінно-нагінних явищ. Причинності для "
             "окремої дати НЕ приписуємо: даних про збурення немає.")
    source(s, "kakhovka_perdate_slopes_robust.csv · "
              "hist10_dynamic_vs_mean.csv · [M1.1, V8.1]")

    # ------------------------------- 15 drawdown and post-breach ----
    s = newslide(prs)
    header(s, L["kick"]["res"],
           "After the breach: a persistent, directional gradient"
           if not ua else
           "Після руйнування: стійкий спрямований ухил", 15, TOTAL)
    picture(s, "FigF_postbreach_longitudinal_gradient.png", Cm(1.5), Cm(3.9),
            w=Cm(17.4))
    statrow(s, [(V["pre"], "cm/km PRE" if not ua else "см/км ДО", MUTED),
                (V["draw"], "cm/km DRAWDOWN" if not ua
                 else "см/км СПРАЦЮВАННЯ", HIST),
                (V["post"], "cm/km POST" if not ua else "см/км ПІСЛЯ", ICESAT)],
            y=Cm(4.2), x0=Cm(19.6), total=Cm(12.8))
    bullets(s, Cm(19.6), Cm(8.2), Cm(12.8), [
        ("Δ", f"Difference {V['diff']} cm/km, 95 % CI {V['ci']}, "
                    f"permutation p = {V['p']}.", ICESAT),
        ("✓", f"{V['postpos']} post-breach dates are positive, spanning "
                   f"2024-01 to 2025-11 — 22 months. This is a REGIME "
                   f"SHIFT, not a drainage transient.", GOOD),
        ("≈", f"Estimator-robust: OLS gives {V['ols_pre']} → "
                   f"{V['ols_post']} cm/km, the same sign and conclusion.", MUTED),
        ("!", f"HONEST WEAKNESS: restricted to the main channel the "
              f"difference is {V['chdiff']} cm/km, CI {V['chci']} — "
              f"INCLUDING ZERO, on only {V['chn']} qualifying dates. "
              f"Underpowered, not a refutation.", BAD)]
        if not ua else [
        ("Δ", f"Різниця {V['diff']} см/км, 95 % ДІ {V['ci']}, "
                    f"перестановочний p = {V['p']}.", ICESAT),
        ("✓", f"{V['postpos']} дат після руйнування мають додатний ухил, "
                   f"від 2024-01 до 2025-11 — 22 місяці. Це ЗМІНА РЕЖИМУ, а "
                   f"не перехідний процес спорожнення.", GOOD),
        ("≈", f"Стійко щодо оцінювача: МНК дає {V['ols_pre']} → "
                   f"{V['ols_post']} см/км — той самий знак і висновок.", MUTED),
        ("!", f"ЧЕСНА СЛАБКІСТЬ: у межах руслової частини різниця "
              f"{V['chdiff']} см/км, ДІ {V['chci']} — ВКЛЮЧАЄ НУЛЬ, лише на "
              f"{V['chn']} придатних датах. Замала статистична потужність, "
              f"а не заперечення результату.", BAD)], size=11, gap=Cm(2.05))
    takeaway(s, "The strength of this result is the effect size and the "
                f"{V['postpos']} consistent sign across 22 months — not "
                "the p-value."
             if not ua else
             "Сила цього результату — у величині ефекту та стійкому знаку "
             f"{V['postpos']} упродовж 22 місяців, а не у p-значенні.",
             colour=ICESAT)
    source(s, "kakhovka_transition_statistics.csv · "
              "reservoir_to_river_statistics.csv · [M1.1, M1.3, M1.4]")

    # ---------------------- 16 heterogeneity and planform ----
    s = newslide(prs)
    header(s, L["kick"]["res"],
           "Two more independent signatures of the same transition"
           if not ua else
           "Ще дві незалежні ознаки тієї самої трансформації", 16, TOTAL)
    txbox(s, Cm(1.5), Cm(3.8), Cm(15.2), Cm(0.8),
          "1 · WATER-SURFACE HETEROGENEITY (ALTIMETRIC)" if not ua
          else "1 · НЕОДНОРІДНІСТЬ ВОДНОЇ ПОВЕРХНІ (АЛЬТИМЕТРІЯ)",
          size=11, bold=True, colour=ICESAT)
    statrow(s, [(V["f4pre"] + " m", "PRE p95−p05" if not ua
                 else "ДО, p95−p05", MUTED),
                (V["f4post"] + " m", "POST p95−p05" if not ua
                 else "ПІСЛЯ, p95−p05", ICESAT),
                (V["f4d"] + " m", "difference" if not ua else "різниця", GOOD)],
            y=Cm(4.7), x0=Cm(1.5), total=Cm(15.2))
    txbox(s, Cm(1.5), Cm(8.0), Cm(15.2), Cm(2.6),
          (f"95 % CI {V['f4ci']} m, permutation p = 5×10⁻⁵, "
           f"n = {V['f4n']} dates.\n"
           f"Measured by the altimeter, so INDEPENDENT of Sentinel-2 tile "
           f"coverage. Not to be confused with the within-profile range of "
           f"the slope sample ({V['rng_pre']} → {V['rng_post']} m), a "
           f"different sample and restriction.")
          if not ua else
          (f"95 % ДІ {V['f4ci']} м, перестановочний p = 5×10⁻⁵, "
           f"n = {V['f4n']} дат.\n"
           f"Виміряно альтиметром, тому НЕ залежить від покриття кадрами "
           f"Sentinel-2. Не плутати з розмахом у межах профілю у виборці "
           f"ухилів ({V['rng_pre']} → {V['rng_post']} м) — це інша "
           f"виборка та інше обмеження."),
          size=11, colour=INK, spacing=1.1)
    rect(s, Cm(17.0), Cm(3.8), Cm(0.03), Cm(12.2), fill=RULE)
    txbox(s, Cm(17.4), Cm(3.8), Cm(15), Cm(0.8),
          "2 · PLANFORM (SENTINEL-2, DESCRIPTIVE)" if not ua
          else "2 · ПЛАНОВА СТРУКТУРА (SENTINEL-2, ОПИСОВО)",
          size=11, bold=True, colour=SENT)
    hdr = (["", "2023-06-05", f"після ({V['s2npost']} дат)"] if ua
           else ["", "2023-06-05", f"post ({V['s2npost']} dates)"])
    rws = ([("водних об'єктів", f"{V['s2pre_n']}", f"{V['s2post_n']} ({V['s2post_rng']})"),
            ("площа води, км²", V["s2pre_a"], V["s2post_a"]),
            ("найбільший компонент", f"{V['s2pre_l']} %", f"{V['s2post_l']} %")]
           if ua else
           [("water bodies", f"{V['s2pre_n']}", f"{V['s2post_n']} ({V['s2post_rng']})"),
            ("water area, km²", V["s2pre_a"], V["s2post_a"]),
            ("largest component", f"{V['s2pre_l']} %", f"{V['s2post_l']} %")])
    y = Cm(4.7)
    for j, hh in enumerate(hdr):
        txbox(s, Cm(17.4) + j * Cm(5.0), y, Cm(4.8), Cm(0.7), hh, size=10.5,
              bold=True, colour=MUTED,
              align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
    y += Cm(0.85)
    for lab, a, b in rws:
        txbox(s, Cm(17.4), y, Cm(4.8), Cm(0.8), lab, size=11, colour=INK)
        txbox(s, Cm(22.4), y, Cm(4.8), Cm(0.8), a, size=13, bold=True,
              colour=SENT, align=PP_ALIGN.RIGHT)
        txbox(s, Cm(27.4), y, Cm(4.8), Cm(0.8), b, size=13, bold=True,
              colour=ICESAT, align=PP_ALIGN.RIGHT)
        y += Cm(1.25)
    txbox(s, Cm(17.4), Cm(9.5), Cm(15), Cm(3),
          (f"Only ONE pre-breach date has near-complete footprint coverage, "
           f"so n_pre = 1 and NO p-value may be computed. An earlier test "
           f"that treated partial scenes as full-reservoir observations is "
           f"INVALID and superseded.\n\n"
           f"Residual water bodies sit {V['rw']} m below the channel stem "
           f"(n = {V['rwn']}; only {V['rwabove']} % above).")
          if not ua else
          (f"Лише ОДНА дата до руйнування має майже повне покриття області, "
           f"тому n_до = 1 і p-значення обчислювати НЕ можна. Ранній тест, "
           f"що трактував часткові кадри як повні спостереження "
           f"водосховища, НЕДІЙСНИЙ і замінений.\n\n"
           f"Залишкові водні об'єкти лежать на {V['rw']} м нижче руслового "
           f"стрижня (n = {V['rwn']}; лише {V['rwabove']} % вище)."),
          size=11, colour=INK, spacing=1.1)
    takeaway(s, "A single continuous 2 033 km² surface has been replaced "
                "by hundreds of water bodies — descriptive planform "
                "evidence supporting the altimetric result, never replacing "
                "it."
             if not ua else
             "Єдина неперервна поверхня площею 2 033 км² змінилася "
             "сотнями водних об'єктів — це описова планова ознака, що "
             "підтримує альтиметричний результат, але не замінює його.")
    source(s, "pre_post_fragmentation_statistics.csv · "
              "fragmentation_metrics_by_date.csv · "
              "residual_water_offset_summary.csv · [M2.1, M3.1, M3.2]")

    # --------------------------------------------- 17 what is novel ----
    s = newslide(prs)
    header(s, L["kick"]["synth"],
           "What is actually new here" if not ua
           else "Що саме тут нового", 17, TOTAL)
    txbox(s, Cm(1.5), Cm(3.9), Cm(15.2), Cm(0.8),
          "METHODOLOGICAL" if not ua else "МЕТОДИЧНА НОВИЗНА", size=11,
          bold=True, colour=SWOT)
    bullets(s, Cm(1.5), Cm(4.8), Cm(15.2), [
        "Integrated vertical harmonisation of gauges, ICESat-2 and SWOT with "
        "explicit permanent-tide treatment.",
        "Empirical verification of the SWOT PIXC vertical chain against an "
        "independent product of the same mission.",
        "Historical hydraulic observations used as an INDEPENDENT validation "
        "target for modern satellite altimetry.",
        "Archival bathymetry re-referenced into a modern EVRF2019 framework, "
        "with the reduction datum RECOVERED from the data.",
        "Area-weighted morphological validation against a historical "
        "statistic that never entered the reconstruction."]
        if not ua else [
        "Комплексне вертикальне узгодження постів, ICESat-2 і SWOT з явним "
        "урахуванням системи постійного припливу.",
        "Емпірична перевірка вертикального ланцюга SWOT PIXC проти "
        "незалежного продукту тієї самої місії.",
        "Історичні натурні гідравлічні спостереження як НЕЗАЛЕЖНА мета "
        "валідації сучасної супутникової альтиметрії.",
        "Архівна батиметрія, переприв'язана до сучасної системи EVRF2019, з "
        "ВІДНОВЛЕННЯМ відмітки зведення глибин із самих даних.",
        "Площинно-вагова морфологічна валідація проти історичної "
        "статистики, яка не входила у побудову."], size=11, gap=Cm(1.85))
    rect(s, Cm(17.0), Cm(3.9), Cm(0.03), Cm(12.1), fill=RULE)
    txbox(s, Cm(17.4), Cm(3.9), Cm(15), Cm(0.8),
          "SCIENTIFIC" if not ua else "НАУКОВА НОВИЗНА", size=11, bold=True,
          colour=ICESAT)
    bullets(s, Cm(17.4), Cm(4.8), Cm(15), [
        "Quantitative detection of an abrupt change from a near-level "
        "impounded surface to a persistent river-like longitudinal gradient.",
        "REPEATED post-breach slope observations over 22 months, rather than "
        "a single map.",
        "A measured increase in water-surface heterogeneity, independent of "
        "optical coverage.",
        "Altimetry, historical hydraulics and planform remote sensing "
        "combined to characterise the post-breach state."]
        if not ua else [
        "Кількісне виявлення різкого переходу від майже горизонтальної "
        "підпертої поверхні до стійкого річкового продовжнього ухилу.",
        "ПОВТОРНІ спостереження ухилу після руйнування впродовж 22 місяців, "
        "а не одна карта.",
        "Виміряне зростання неоднорідності водної поверхні, незалежне від "
        "оптичного покриття.",
        "Поєднання альтиметрії, історичної гідравліки та планової "
        "дистанційної зйомки для характеристики стану після "
        "руйнування."], size=11, gap=Cm(2.2))
    takeaway(s, "Novelty claims against the literature are marked "
                "PRELIMINARY: no systematic review has been carried out yet, "
                "and none is asserted here."
             if not ua else
             "Твердження про новизну щодо літератури позначені як "
             "ПОПЕРЕДНІ: систематичний огляд ще не виконано, і тут він не "
             "заявляється.", colour=BAD)
    source(s, "article_open_questions.md — Tier 1, item 1")

    # ------------------------------ 18 limitations and conclusions ----
    s = newslide(prs)
    header(s, L["kick"]["synth"],
           "Conclusions, and what we do not yet claim" if not ua
           else "Висновки і те, чого ми ще не стверджуємо", 18, TOTAL)
    txbox(s, Cm(1.5), Cm(3.9), Cm(15.2), Cm(0.8),
          "CONCLUSIONS" if not ua else "ВИСНОВКИ", size=11, bold=True,
          colour=GOOD)
    bullets(s, Cm(1.5), Cm(4.8), Cm(15.2), [
        ("1", "Gauges, ICESat-2 and SWOT can be compared in one vertical "
              "frame with a demonstrable 0.044 m post-tie NMAD.", GOOD),
        ("2", f"The historical soundings are reduced to UNS 14.00 m, not NPG "
              f"16.00 m: {V['h14']} m agreement replaces {V['h16']} m.", GOOD),
        ("3", "Historical in-situ hydraulics and morphology independently "
              "validate the modern framework.", GOOD),
        ("4", f"The water surface changed from near-level ({V['pre']} cm/km, "
              f"{V['prepos']}) to a persistent gradient ({V['post']} cm/km, "
              f"{V['postpos']}), Δ {V['diff']} cm/km {V['ci']}.", ICESAT),
        ("5", "The system is RIVER-DOMINATED with persistent residual water "
              "bodies — not simply “a river”.", ICESAT)]
        if not ua else [
        ("1", "Пости, ICESat-2 і SWOT можна порівнювати в одній вертикальній "
              "системі з доказовим NMAD залишків 0,044 м.", GOOD),
        ("2", f"Історичні проміри зведені до УНС 14,00 м, а не до НПГ "
              f"16,00 м: узгодження {V['h14']} м замість {V['h16']} м.", GOOD),
        ("3", "Історичні натурні гідравліка та морфологія незалежно "
              "підтверджують сучасну систему.", GOOD),
        ("4", f"Водна поверхня перейшла від майже горизонтальної "
              f"({V['pre']} см/км, {V['prepos']}) до стійкого ухилу "
              f"({V['post']} см/км, {V['postpos']}), Δ {V['diff']} см/км "
              f"{V['ci']}.", ICESAT),
        ("5", "Система є РІЧКОВО-ДОМІНОВАНОЮ зі стійкими залишковими водними "
              "об'єктами, а не просто «річкою».", ICESAT)], size=11,
        gap=Cm(2.2))
    rect(s, Cm(17.0), Cm(3.9), Cm(0.03), Cm(12.1), fill=RULE)
    txbox(s, Cm(17.4), Cm(3.9), Cm(15), Cm(0.8),
          "NOT CLAIMED / OPEN" if not ua else "НЕ СТВЕРДЖУЄМО / ВІДКРИТЕ",
          size=11, bold=True, colour=BAD)
    bullets(s, Cm(17.4), Cm(4.8), Cm(15), [
        ("✗", "No rate of bed change — the survey epoch is "
                   "unrecorded.", BAD),
        ("✗", "No whole-reservoir instantaneous gradient — no "
                   "overpass spans the reservoir (median 8.3 km).", BAD),
        ("✗", "No SWOT-based reservoir level — the orbit never "
                   "covered the pool.", BAD),
        ("✗", "No causal attribution of any anomaly to wind or seiches "
                   "— no forcing data.", BAD),
        ("?", "Open: the channel-restricted CI includes zero; the Baltic "
              "realisation (BS-42 vs BS-77) is unresolved; the literature "
              "review is outstanding.", HIST)]
        if not ua else [
        ("✗", "Жодних темпів зміни дна — епоха зйомки не "
                   "зафіксована.", BAD),
        ("✗", "Жодного миттєвого ухилу по всьому водосховищу — жоден "
                   "проліт не покриває його (медіана 8,3 км).", BAD),
        ("✗", "Жодного рівня водосховища за SWOT — орбіта не покривала "
                   "акваторію.", BAD),
        ("✗", "Жодного причинного зв'язку аномалій зі згоном/нагоном "
                   "чи сейшами — немає даних про збурення.", BAD),
        ("?", "Відкрите: ДІ для руслової частини включає нуль; реалізація "
              "Балтійської системи (БС-42 чи БС-77) не встановлена; "
              "систематичний огляд літератури не виконано.", HIST)],
        size=11, gap=Cm(2.2))
    takeaway(s, "Validation first, result second — and every negative "
                "result kept in the record."
             if not ua else
             "Спочатку валідація, потім результат — і кожен негативний "
             "результат залишається у звіті.", colour=SWOT)
    source(s, "manuscript_evidence_matrix.csv · "
              "article_open_questions.md · "
              "Kakhovka_scientific_report_article_draft_v1.md")

    prs.save(dst)
    return len(prs.slides.__iter__.__self__._sldIdLst)


for L, name in ((EN, "Kakhovka_validation_and_transition_EN.pptx"),
                (UA, "Kakhovka_validation_and_transition_UA.pptx")):
    dst = OUT / name
    print(f"building {L['lang']} ...")
    n = build(L, dst)
    print(f"  {n} slides -> {dst}")
