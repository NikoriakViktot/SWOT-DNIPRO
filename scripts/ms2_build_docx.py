#!/usr/bin/env python
"""MANUSCRIPT 2 — render the markdown draft to DOCX.

No pandoc on this machine, so this is a small purpose-built converter. It
handles exactly the constructs the draft uses -- headings, paragraphs, bold /
italic / inline code, pipe tables, fenced code, lists, rules and blockquotes --
and REFUSES to silently skip anything it does not understand, because a
converter that drops a table quietly is worse than one that crashes.

Outputs
-------
outputs/reports/Kakhovka_scientific_report_article_draft_v1.docx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

SRC = ROOT / "outputs/reports/Kakhovka_scientific_report_article_draft_v1.md"
DST = SRC.with_suffix(".docx")

INK = RGBColor(0x1A, 0x22, 0x28)
ACCENT = RGBColor(0x23, 0x6F, 0x8C)
MUTED = RGBColor(0x55, 0x5E, 0x68)

# ------------------------------------------------------------------ styling --
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.4)
sec.top_margin = sec.bottom_margin = Cm(2.2)

st = doc.styles["Normal"]
st.font.name = "Cambria"
st.font.size = Pt(10.5)
st.font.color.rgb = INK
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15
# East-Asian font name has to be set on the rPr or Word substitutes silently
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")

for name, size, bold, colour, before, after in (
    ("Heading 1", 17, True, ACCENT, 20, 8),
    ("Heading 2", 13.5, True, INK, 15, 6),
    ("Heading 3", 11.5, True, INK, 12, 4),
    ("Heading 4", 10.5, True, MUTED, 10, 3),
):
    s = doc.styles[name]
    s.font.name = "Cambria"
    s.font.size = Pt(size)
    s.font.bold = bold
    s.font.color.rgb = colour
    s.paragraph_format.space_before = Pt(before)
    s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True


def shade(cell, hexcolour):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolour)
    cell._tc.get_or_add_tcPr().append(el)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+?\*)")


def add_runs(par, text):
    """Bold / italic / inline-code, applied to one paragraph."""
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9)
            r.font.color.rgb = ACCENT
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


def emit_table(rows):
    """rows: list of lists of cell strings; row 0 is the header."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, cell.strip())
            for r in p.runs:
                r.font.size = Pt(8.8)
                if i == 0:
                    r.bold = True
            if i == 0:
                shade(c, "E8EEF2")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def is_sep(line):
    s = line.strip().strip("|")
    return bool(s) and set(s.replace(":", "").replace("-", "").replace("|", "").strip()) == set()


# ------------------------------------------------------------------- convert --
lines = SRC.read_text(encoding="utf-8").split("\n")
i, n_tables, n_head, n_para, n_code = 0, 0, 0, 0, 0

while i < len(lines):
    line = lines[i]
    s = line.strip()

    if not s:
        i += 1
        continue

    # fenced code
    if s.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            buf.append(lines[i])
            i += 1
        i += 1
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run("\n".join(buf))
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        n_code += 1
        continue

    # indented equation block (4 spaces, used for the vertical chains)
    if line.startswith("    ") and s and not s.startswith(("-", "*", "|", "#")):
        buf = []
        while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
            if lines[i].strip():
                buf.append(lines[i].strip())
            elif buf and i + 1 < len(lines) and lines[i + 1].startswith("    "):
                buf.append("")
            else:
                break
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run("\n".join(buf))
        r.font.name = "Consolas"
        r.font.size = Pt(9.5)
        n_code += 1
        continue

    # horizontal rule
    if s in ("---", "***", "___"):
        p = doc.add_paragraph()
        pr = p._p.get_or_add_pPr()
        bd = OxmlElement("w:pBdr")
        bt = OxmlElement("w:bottom")
        bt.set(qn("w:val"), "single")
        bt.set(qn("w:sz"), "6")
        bt.set(qn("w:color"), "C3CBD2")
        bd.append(bt)
        pr.append(bd)
        i += 1
        continue

    # heading
    if s.startswith("#"):
        lvl = len(s) - len(s.lstrip("#"))
        txt = s[lvl:].strip()
        if lvl == 1 and n_head > 0:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        p = doc.add_heading(level=min(lvl, 4))
        for r in list(p.runs):
            r.text = ""
        add_runs(p, txt)
        sty = doc.styles[f"Heading {min(lvl,4)}"]
        for r in p.runs:
            r.font.name = "Cambria"
            r.font.size = sty.font.size
            r.font.bold = sty.font.bold
            r.font.color.rgb = sty.font.color.rgb
        n_head += 1
        i += 1
        continue

    # table
    if s.startswith("|"):
        block = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip())
            i += 1
        rows = [[c for c in r.strip("|").split("|")] for r in block if not is_sep(r)]
        if rows:
            emit_table(rows)
            n_tables += 1
        continue

    # blockquote
    if s.startswith(">"):
        buf = []
        while i < len(lines) and lines[i].strip().startswith(">"):
            buf.append(lines[i].strip().lstrip(">").strip())
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        add_runs(p, " ".join(buf))
        for r in p.runs:
            r.italic = True
            r.font.color.rgb = MUTED
        continue

    # list item (may wrap onto continuation lines)
    m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
    if m:
        indent, marker, txt = m.group(1), m.group(2), m.group(3)
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
               and not lines[i].strip().startswith(("|", "#", ">", "```"))
               and lines[i].startswith((" ", "\t"))):
            txt += " " + lines[i].strip()
            i += 1
        style = "List Number" if marker[0].isdigit() else "List Bullet"
        p = doc.add_paragraph(style=style)
        p.paragraph_format.left_indent = Cm(0.7 + 0.5 * (len(indent) // 2))
        p.paragraph_format.space_after = Pt(3)
        add_runs(p, txt)
        for r in p.runs:
            r.font.size = Pt(10.5)
            r.font.name = "Cambria"
        continue

    # paragraph (join wrapped lines)
    buf = [s]
    i += 1
    while (i < len(lines) and lines[i].strip()
           and not lines[i].strip().startswith(("#", "|", ">", "-", "*", "```"))
           and not lines[i].startswith("    ")
           and not re.match(r"^\s*\d+\.\s+", lines[i])):
        buf.append(lines[i].strip())
        i += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_runs(p, " ".join(buf))
    n_para += 1

doc.save(DST)
print(f"  headings {n_head}   paragraphs {n_para}   tables {n_tables}   "
      f"code/equation blocks {n_code}")
print(f"-> {DST}")
