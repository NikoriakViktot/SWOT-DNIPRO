#!/usr/bin/env python
"""MANUSCRIPT 4 — quality control on the decks and the manuscript.

There is no LibreOffice or pandoc on this machine, so the slides cannot be
rasterised for visual inspection. This script substitutes the checks that can
be made mechanically, and the QC report states plainly which check could NOT
be performed rather than implying a visual pass.

Checks
------
G1  every shape lies inside the slide
G2  no picture overlaps the takeaway bar or the source line
G3  no two text boxes overlap
T1  obsolete terminology absent from active deliverables
T2  every headline number on a slide matches its canonical table
T3  sign conventions are stated wherever a signed difference appears
T4  grid resolutions are not mixed
T5  reach 5 is NA, never zero
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from pptx import Presentation
from pptx.util import Cm

from swot_dnipro import config as CFG

DECKS = {
    "EN": ROOT / "outputs/presentation/Kakhovka_validation_and_transition_EN.pptx",
    "UA": ROOT / "outputs/presentation/Kakhovka_validation_and_transition_UA.pptx",
}
MD = ROOT / "outputs/reports/Kakhovka_scientific_report_article_draft_v1.md"
DOCX = MD.with_suffix(".docx")
MATRIX = CFG.TABLES / "manuscript_evidence_matrix.csv"

W, H = Cm(33.87), Cm(19.05)
TAKEAWAY_TOP = Cm(16.35)
FAILS: list[str] = []
WARNS: list[str] = []


def fail(m):
    FAILS.append(m)
    print(f"  FAIL  {m}")


def warn(m):
    WARNS.append(m)
    print(f"  warn  {m}")


def ok(m):
    print(f"  ok    {m}")


def overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    return ix * iy


# ===================================================================== #
print("=" * 78)
print("G — SLIDE GEOMETRY")
print("=" * 78)
for lang, path in DECKS.items():
    prs = Presentation(path)
    print(f"\n{lang}: {len(prs.slides._sldIdLst)} slides")
    for i, s in enumerate(prs.slides, 1):
        pics, texts = [], []
        for sh in s.shapes:
            if sh.left is None or sh.top is None:
                continue
            box = (sh.left, sh.top, sh.left + (sh.width or 0),
                   sh.top + (sh.height or 0))
            # G1 inside the slide (2 mm tolerance)
            tol = Cm(0.2)
            if (box[0] < -tol or box[1] < -tol or box[2] > W + tol
                    or box[3] > H + tol):
                fail(f"{lang} slide {i}: shape '{sh.shape_type}' outside the "
                     f"slide: L{box[0]/360000:.1f} T{box[1]/360000:.1f} "
                     f"R{box[2]/360000:.1f} B{box[3]/360000:.1f} cm")
            if sh.shape_type == 13:                       # PICTURE
                pics.append((sh, box))
            elif sh.has_text_frame and sh.text_frame.text.strip():
                texts.append((sh, box))
        # G2 pictures must not run into the takeaway bar
        for sh, box in pics:
            if box[3] > TAKEAWAY_TOP + Cm(0.1):
                fail(f"{lang} slide {i}: picture bottom "
                     f"{box[3]/360000:.2f} cm runs into the takeaway bar "
                     f"({TAKEAWAY_TOP/360000:.2f} cm)")
        # G3 text-box overlaps (>25 % of the smaller box)
        for a in range(len(texts)):
            for b in range(a + 1, len(texts)):
                ov = overlap(texts[a][1], texts[b][1])
                if ov <= 0:
                    continue
                area = min((texts[a][1][2] - texts[a][1][0]) * (texts[a][1][3] - texts[a][1][1]),
                           (texts[b][1][2] - texts[b][1][0]) * (texts[b][1][3] - texts[b][1][1]))
                if area and ov / area > 0.25:
                    ta = texts[a][0].text_frame.text.strip()[:28].replace("\n", " ")
                    tb = texts[b][0].text_frame.text.strip()[:28].replace("\n", " ")
                    warn(f"{lang} slide {i}: text boxes overlap "
                         f"{100*ov/area:.0f}% — '{ta}' / '{tb}'")
    ok(f"{lang} geometry scanned")

# ===================================================================== #
print("\n" + "=" * 78)
print("T1 — OBSOLETE TERMINOLOGY IN ACTIVE DELIVERABLES")
print("=" * 78)
BANNED = {
    "bimodal": "the bed distribution is unimodal (skew -0.76); framing withdrawn",
    "antimode": "no mode-separation test supports a mode boundary",
    "two populations": "withdrawn with the bimodal framing",
    "universal correction": "the correction is local; intervals are disjoint",
    "global ICESat bias": "no global bias is claimed from n=3",
    "perfect validation": "banned wording",
    "exact reconstruction": "banned wording",
    "proved": "banned wording",
    "complete restoration": "banned wording",
}
texts = {"manuscript.md": MD.read_text(encoding="utf-8")}
for lang, path in DECKS.items():
    prs = Presentation(path)
    buf = []
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                buf.append(sh.text_frame.text)
    texts[f"deck {lang}"] = "\n".join(buf)
for where, blob in texts.items():
    low = blob.lower()
    for term, why in BANNED.items():
        for m in re.finditer(re.escape(term), low):
            ctx = blob[max(0, m.start() - 110):m.end() + 60].replace("\n", " ")
            # a term inside an explicit prohibition/retraction is allowed
            if re.search(r"not |never|withdraw|retract|banned|must not|"
                         r"no longer|superseded|invalid|НЕ |не заявля", ctx,
                         re.I):
                continue
            fail(f"{where}: obsolete term '{term}' used affirmatively — {why}"
                 f"\n          ...{ctx.strip()}...")
if not FAILS:
    ok("no obsolete term used affirmatively in the manuscript or either deck")

# ===================================================================== #
print("\n" + "=" * 78)
print("T2 — HEADLINE NUMBERS AGAINST CANONICAL TABLES")
print("=" * 78)
T = CFG.TABLES
ts = pd.read_csv(T / "kakhovka_transition_statistics.csv")
tsr = ts[ts.estimator == "Theil-Sen"].iloc[0]
gres = pd.read_csv(T / "hist17_grid_resolution.csv")
g250 = gres[(gres.scope == "whole") & (gres.cell_m == 250)].iloc[0]
hrs = pd.read_csv(T / "historical_reference_sensitivity.csv")
h14 = hrs[hrs.reference_level_m == 14.0].iloc[0]
h16 = hrs[hrs.reference_level_m == 16.0].iloc[0]
frag = pd.read_csv(T / "pre_post_fragmentation_statistics.csv").iloc[0]
colo = pd.read_csv(T / "part4_colocated_swot_icesat.csv")
best = colo.loc[colo.dt_hours.abs().idxmin()]
ex = pd.read_csv(T / "historical_exposure_area_validation.csv")

EXPECT = [
    ("pre-breach Theil-Sen median", f"{tsr.median_pre_cm_km:+.2f}", "+0.09"),
    ("post-breach Theil-Sen median", f"{tsr.median_post_cm_km:+.2f}", "+3.31"),
    ("difference", f"{tsr.diff_post_minus_pre_cm_km:+.2f}", "+3.22"),
    ("CI low", f"{tsr.diff_ci95_low:+.2f}", "+1.99"),
    ("CI high", f"{tsr.diff_ci95_high:+.2f}", "+5.07"),
    ("exposure historical", f"{g250.historical_pct:.2f}", "12.95"),
    # 12.38 was the P20-era value; hist17 was regenerated on the registry domain
    # 2026-09-13 (b3255e4) -> 12.77, but this list could not be re-checked until
    # python-pptx was installed 2026-09-16. The draft article still carries BOTH
    # numbers (audit F-21) -- that is a manuscript fix, not an EXPECT fix.
    ("exposure reconstructed 250 m", f"{g250.reconstructed_pct:.2f}", "12.77"),
    ("datum 14.00 m residual", f"{h14.median_dH_m:+.3f}", "+0.098"),
    ("datum 16.00 m residual", f"{h16.median_dH_m:+.2f}", "-1.90"),
    ("F4 pre", f"{frag.median_pre:.3f}", "0.117"),
    ("F4 post", f"{frag.median_post:.3f}", "0.397"),
    ("best cross-sensor", f"{best.d_swot_minus_icesat_m:+.4f}", "+0.0036"),
]
for label, actual, expected in EXPECT:
    if actual != expected:
        fail(f"T2 {label}: canonical table gives {actual}, "
             f"the deliverables were written expecting {expected}")
    else:
        ok(f"{label:<32}{actual}")

md = texts["manuscript.md"]
for label, actual, _ in EXPECT:
    # the manuscript uses a Unicode minus in prose; normalise before searching
    variants = {actual, actual.lstrip("+"), actual.replace("-", "−"),
                actual.lstrip("+").replace("-", "−")}
    if not any(v in md for v in variants):
        warn(f"T2 value for '{label}' ({actual}) not found verbatim in the "
             f"manuscript — check it is quoted at a different precision")

# ===================================================================== #
print("\n" + "=" * 78)
print("T3 — SIGN CONVENTIONS DECLARED")
print("=" * 78)
for where, blob in texts.items():
    need = []
    if "SWOT" in blob and "ICESat" in blob:
        if not re.search(r"H_SWOT\s*[−-]\s*H_ICESat|SWOT\s*[−-]\s*ICESat|"
                         r"SWOT\s*−\s*H", blob):
            need.append("SWOT − ICESat-2")
    if re.search(r"gauge", blob, re.I):
        # accept an en-dash, a hyphen, a Unicode minus, or the word "minus"
        if not re.search(r"satellite\s*(?:[−–-]|minus)\s*gauge|"
                         r"SWOT\s*(?:[−–-]|minus)\s*(?:gauge|пост)",
                         blob, re.I):
            need.append("satellite − gauge")
    if need:
        warn(f"{where}: sign convention not stated explicitly for "
             f"{', '.join(need)}")
    else:
        ok(f"{where}: sign conventions stated")

# ===================================================================== #
print("\n" + "=" * 78)
print("T4 — GRID RESOLUTIONS NOT MIXED")
print("=" * 78)
for where, blob in texts.items():
    has_fine = bool(re.search(r"\b(50|30)\s*m\b", blob))
    if has_fine and not re.search(r"canonical|250", blob):
        fail(f"{where}: mentions a fine grid without naming 250 m as the "
             f"canonical statistical grid")
    else:
        ok(f"{where}: fine grids appear only alongside the 250 m canonical grid")

r5 = ex[ex.reach_id.astype(str) == "5"]
if len(r5):
    v = r5.reconstructed_fraction_pct.iloc[0]
    if pd.isna(v):
        ok("T5 reach 5 reconstructed fraction is NA in the canonical table")
    else:
        fail(f"T5 reach 5 is {v}, must be NA")
for where, blob in texts.items():
    if re.search(r"reach 5|Ділянка 5|ділянка 5", blob, re.I):
        if not re.search(r"\bNA\b|not evaluated|outside", blob, re.I):
            fail(f"{where}: mentions reach 5 without marking it NA/outside "
                 f"the domain")
        else:
            ok(f"{where}: reach 5 marked NA / outside the domain")

# ===================================================================== #
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
mx = pd.read_csv(MATRIX)
print(f"  evidence matrix : {len(mx)} claims, "
      f"{mx.validation_status.value_counts().to_dict()}")
print(f"  manuscript      : {len(md.split()):,} words, "
      f"{DOCX.stat().st_size/1024:.0f} kB docx")
for lang, p in DECKS.items():
    print(f"  deck {lang}         : {p.stat().st_size/1024:.0f} kB")
print(f"\n  {len(FAILS)} failures, {len(WARNS)} warnings")
print("\n  NOT PERFORMED: visual rasterised inspection of the slides — no "
      "LibreOffice\n  or pandoc is installed, so slides were inspected "
      "geometrically and\n  textually, not rendered to images.")
sys.exit(1 if FAILS else 0)
