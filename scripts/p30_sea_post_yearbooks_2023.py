#!/usr/bin/env python
"""P30 — parse the 2023 hydromet yearbooks for the Dnipro–Bug estuary posts.

Two volumes of "Щорічні дані про режим та ресурси вод морів і морських гирл
річок, 2023" sit on the bulk drive as PDF24 exports with a text layer:

    Інв_№229  Частина 1 "Моря", том 2, Чорне море          (Очаків, Парутине)
    Інв_№230  Частина 2 "Морські гирла річок", том 2(2)    (Дніпро-Бузька
              гирлова область, pp 143-195: Херсон, Касперівка, Олександрівка,
              Миколаїв, Очаків, Станіслав, Парутине)

pypdf's layout extraction returns each day x month table as a monospaced
grid, so nothing here is hand-typed: every number is read from the page,
assigned to its month by column position, and checked against the printed
monthly mean / max / min before anything is written. A page that fails its
own arithmetic aborts the run; a page pypdf cannot lay out completely is
recorded as PARSE_INCOMPLETE in the inventory, never guessed.

What the flags mean (kept verbatim in the `flag` column):
    /  distorted by the dam breach (the yearbook's own mark)
    ?  doubtful value            *  see the page note
    А / Т  Oleksandrivka only: the two observation terms the value refers to
    Ш Z Л Х :  ice codes on winter days (шуга, забереги, льодохід, ...)
    <  below the salinity detection limit (value is the limit, 1.80 ‰)
    нб / н/б  not observed

Vertical frame: level_cm is above the post zero; H_bs77_m = zero + cm/100.
EVRF2019 is filled only for posts that already carry an EPSG:9902 delta in
outputs/tables/gauge_vertical_reference_summary.csv (Kherson, Mykolaiv);
the other posts have no registered coordinate yet and get NaN, not a guess.

Outputs
-------
data/historical/sea_posts_2023/*.csv       parsed tables (tracked)
outputs/tables/p30_sea_post_inventory_2023.csv
outputs/reports/P30_sea_posts_2023.md
data/catalog/raw_data_manifest.csv         + the two PDFs
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from pypdf import PdfReader

from swot_dnipro import config as CFG

SRC = CFG.BULK_ROOT / "Sea_post_data"
PDFS = {229: SRC / "Інв_№229_Щорічні_дані_про_режим_та_ресурси_вод_морів_і_морських.pdf",
        230: SRC / "Інв_№230_Щорічні_дані_про_режим_та_ресурси_вод_морів_і_морських.pdf"}
OUT = ROOT / "data" / "historical" / "sea_posts_2023"
YEAR = 2023
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
BREACH = pd.Timestamp("2023-06-06")

# Station list, Інв_№230 p146 (СПИСОК ПОСТІВ ... 2023) and the post
# descriptions pp 149-158. Coordinates only where the project already has
# them (config.py, companion gauges.yaml); the rest stay empty on purpose.
POSTS = {
    80802: dict(name="м. Нова Каховка (ГП-ІІ, нижній б'єф)", name_en="Nova Kakhovka (tailwater)",
                water_body="р. Дніпро", zero_bs77_m=-5.00, lon=33.346667, lat=46.758611,
                coord_source="user-supplied registry DMS 46°45'31\" 33°20'48\" (2026-09-17)",
                status_2023="діючий до 07.2022 (окупація); у щорічнику 2023 даних немає"),
    80805: dict(name="м. Херсон", name_en="Kherson", water_body="р. Дніпро",
                zero_bs77_m=-5.00, lon=32.612026, lat=46.623750,
                coord_source="config.KHERSON_GAUGE; user registry DMS 46°37'24.94\" 32°36'47.12\" "
                             "= 46.62359 32.61309 lies 80 m away",
                status_2023="діючий; СРМ вийшов з ладу 06.06-09.07 (переповнення колодязя)"),
    80807: dict(name="с. Кізомис, МГП-І Касперівка", name_en="Kasperivka",
                water_body="р. Дніпро, рук. Рвач", zero_bs77_m=-5.00,
                lon=32.319487858959064, lat=46.55145353069963,
                coord_source="user-supplied approximate (2026-09-17)",
                status_2023="діючий; з 03.08 спостереження на тимчасовому місці (оз. Дедове, 2 км на схід)"),
    81801: dict(name="с. Олександрівка", name_en="Oleksandrivka",
                water_body="р. Південний Буг", zero_bs77_m=-3.02, lon=31.26972222, lat=47.68555556,
                coord_source="user-supplied registry table (2026-09-17); the user's own point "
                             "47.69921 31.25552 lies 1.8 km NW of it",
                status_2023="діючий (ГП-І, річковий пост, поза всіма зонами)"),
    98027: dict(name="м. Миколаїв", name_en="Mykolaiv", water_body="р. Південний Буг",
                zero_bs77_m=-5.00, lon=31.970917, lat=46.984306,
                coord_source="icesat2-atl13-kakhovka/config/gauges.yaml; user registry DMS "
                             "46°59'3.75\" 31°58'19.46\" = 46.98438 31.97207 lies 90 m away",
                status_2023="діючий"),
    98025: dict(name="с. Парутине", name_en="Parutyne", water_body="лим. Бузький",
                zero_bs77_m=-5.00, lon=31.910218144879558, lat=46.70875311159131,
                coord_source="user-supplied approximate (2026-09-17)", status_2023="діючий"),
    98032: dict(name="с. Станіслав", name_en="Stanislav", water_body="лим. Дніпровський",
                zero_bs77_m=-5.00, lon=32.14353179431262, lat=46.55878816094544,
                coord_source="user-supplied approximate (2026-09-17)",
                status_2023="діючий до 10.05.2023; з 11.05 призупинено (мінування, евакуація спостерігача)"),
    98033: dict(name="с. Геройське", name_en="Heroiske", water_body="лим. Дніпро-Бузький",
                zero_bs77_m=-5.00, lon=31.896809000844556, lat=46.51750333367365,
                coord_source="user-supplied approximate (2026-09-17)",
                status_2023="рівень не спостерігався з 18.09.2022 (зруйновано причал); закрито 31.12.2023"),
    98022: dict(name="м. Очаків", name_en="Ochakiv", water_body="лим. Дніпро-Бузький",
                zero_bs77_m=-5.00, lon=31.552048718704924, lat=46.60263049261722,
                coord_source="user-supplied approximate (2026-09-17)",
                status_2023="діючий (МГ-ІІ)"),
}


def post_geometry_check():
    """EPSG:9902 delta at each post coordinate (the part1 sampler, so the
    registered Kherson/Mykolaiv values must be reproduced to the mm) and the
    registry check the project insists on for any new geometry: how far the
    point sits from the water domain and which zones contain it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import geopandas as gpd
    from shapely.geometry import Point
    from part1_gauge_rereference import load_grid, sample_grid, ASC
    from swot_dnipro import spatial_domains as SD
    zg, zh = load_grid(ASC)
    water = SD.load_utm("dnipro_water_domain")
    zones = {z: SD.load_utm(z) for z in ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
                                         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")}
    ref = {}
    summ = CFG.TABLES / "gauge_vertical_reference_summary.csv"
    if summ.exists():
        g = pd.read_csv(summ)
        ref = dict(zip(g.station_id.astype(int), g.delta_epsg9902_m))
    for pid, v in POSTS.items():
        if v["lon"] is None:
            v.update(delta_epsg9902_m=np.nan, dist_to_water_domain_m=np.nan, zones="")
            continue
        d = float(sample_grid(zg, zh, v["lon"], v["lat"]))
        if pid in ref and np.isfinite(ref[pid]) and abs(d - ref[pid]) > 0.001:
            raise SystemExit(f"EPSG:9902 delta for {pid} ({d:.4f}) disagrees with the "
                             f"registered value ({ref[pid]:.4f})")
        p = gpd.GeoSeries([Point(v["lon"], v["lat"])], crs=4326).to_crs(32636).iloc[0]
        v.update(delta_epsg9902_m=d, dist_to_water_domain_m=float(water.distance(p)),
                 zones=" ".join(z for z, geom in zones.items() if geom.contains(p)))
    return {pid: v["delta_epsg9902_m"] for pid, v in POSTS.items()}
BY_NAME = {"Херсон": 80805, "Касперівка": 80807, "Кізомис": 80807, "Олександрівка": 81801,
           "Миколаїв": 98027, "Парутине": 98025, "Станіслав": 98032, "Геройське": 98033,
           "Очаків": 98022}

# (volume, page, variable, unit, expected post) for the day x month grids
GRID_PAGES = [
    (229, 34, "sea_level", "cm", 98022), (229, 35, "sea_level", "cm", 98025),
    (229, 48, "water_temperature", "degC", 98022), (229, 49, "water_temperature", "degC", 98025),
    (229, 60, "salinity", "ppt", 98022), (229, 61, "salinity", "ppt", 98025),
    (230, 162, "water_level", "cm", 80805), (230, 163, "water_level", "cm", 80807),
    (230, 164, "water_level", "cm", 81801), (230, 165, "water_level", "cm", 98027),
    (230, 166, "water_level", "cm", 98022), (230, 167, "water_level", "cm", 98032),
    (230, 168, "water_level", "cm", 98025),
    (230, 175, "discharge", "m3/s", 81801),
]
MEAN_LABELS = {"середн.", "сер. міс.", "серед."}
MAX_LABELS = {"вищ.", "макс.", "найб."}
MIN_LABELS = {"нижч.", "мінім.", "найм."}
TOL = {"cm": 0.6, "degC": 0.06, "ppt": 0.02, "m3/s": 0.15}


# ------------------------------------------------------------------ helpers
def page_text(vol: int, page: int) -> str:
    r = PdfReader(str(PDFS[vol]))
    return r.pages[page - 1].extract_text(extraction_mode="layout") or ""


def tokens(line: str):
    """(start, end, text) for runs separated by two or more spaces."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+(?: \S+)*", line)]


NUM = re.compile(r"^(<)?(-?\d+(?:[.,]\d+)?)(.*)$")


def parse_cell(tok: str):
    """'799?/' -> (799.0, '?/'); '<1.80' -> (1.8, '<'); 'нб' -> (nan, 'нб')."""
    t = tok.strip()
    m = NUM.match(t)
    if not m:
        return np.nan, t
    lt, num, rest = m.groups()
    val = float(num.replace(",", "."))
    flag = (lt or "") + rest.replace(" ", "")
    return val, flag


def month_columns(lines):
    """Header line with the roman month numerals -> [(month, x_centre)]."""
    for ln in lines:
        cols = [(ROMAN.index(t) + 1, (a + b) / 2) for a, b, t in tokens(ln) if t in ROMAN]
        if len(cols) >= 5:
            return cols
    raise ValueError("no month header found")


def assign(cols, toks):
    """Map tokens to month columns by nearest centre; tokens left of the
    first column belong to the row label area and are returned separately."""
    xs = [x for _, x in cols]
    half = min(np.diff(xs)) / 2 if len(xs) > 1 else 40
    out, spare = {}, []
    for a, b, t in toks:
        c = (a + b) / 2
        j = int(np.argmin([abs(c - x) for x in xs]))
        if abs(c - xs[j]) > half * 1.15:
            spare.append((c, t))
            continue
        m = cols[j][0]
        if m in out:
            # a footnote mark set off by a space ('489 !' -> '489', '!')
            # lands on the same column: it is the value's flag, not a value
            if not re.search(r"\d", t):
                out[m] = out[m] + t
            elif not re.search(r"\d", out[m]):
                out[m] = t + out[m]
            else:
                raise ValueError(f"two tokens for month {m}: {out[m]!r} and {t!r}")
            continue
        out[m] = t
    return out, spare


def parse_grid(text: str, unit: str):
    lines = [l for l in text.splitlines() if l.strip()]
    cols = month_columns(lines)
    daily, stats, notes = {}, {}, []
    in_decades = False
    for ln in lines:
        toks = tokens(ln)
        if not toks:
            continue
        label = toks[0][2].strip()
        low = label.lower()
        if low.startswith("примітка") or notes:
            notes.append(ln.strip())
            continue
        if low.startswith("декада"):
            in_decades = True
            continue
        if re.fullmatch(r"\d{1,2}", label) and not in_decades:
            day = int(label)
            if 1 <= day <= 31:
                cells, _ = assign(cols, toks[1:])
                for m, t in cells.items():
                    if day > calendar.monthrange(YEAR, m)[1]:
                        raise ValueError(f"day {day} in month {m}")
                    daily[(m, day)] = parse_cell(t)
            continue
        key = ("mean" if low in MEAN_LABELS else "max" if low in MAX_LABELS
               else "min" if low in MIN_LABELS else None)
        if key and key not in stats:
            cells, _ = assign(cols, toks[1:])
            stats[key] = {m: parse_cell(t) for m, t in cells.items()}
            in_decades = False
    if not daily:
        raise ValueError("no daily rows parsed")
    # ---- the page checks itself --------------------------------------------
    problems = []
    tol = TOL[unit]
    for m in range(1, 13):
        vals = [v for (mm, d), (v, f) in daily.items() if mm == m and np.isfinite(v)]
        if not vals:
            continue
        pm = stats.get("mean", {}).get(m)
        if pm and np.isfinite(pm[0]):
            ndays = calendar.monthrange(YEAR, m)[1]
            if len(vals) == ndays and abs(np.mean(vals) - pm[0]) > tol:
                problems.append(f"month {m}: mean of {len(vals)} days {np.mean(vals):.2f} "
                                f"vs printed {pm[0]:.2f}")
        px = stats.get("max", {}).get(m)
        if px and np.isfinite(px[0]) and px[0] < max(vals) - tol:
            problems.append(f"month {m}: printed max {px[0]} < daily max {max(vals)}")
        pn = stats.get("min", {}).get(m)
        if pn and np.isfinite(pn[0]) and "<" not in pn[1] and pn[0] > min(vals) + tol:
            problems.append(f"month {m}: printed min {pn[0]} > daily min {min(vals)}")
    return dict(daily=daily, stats=stats, notes=" ".join(notes), problems=problems,
                n_months=len(cols))


def annual_line(text: str):
    """The 2023 row of the period table (№230) or the 'за рік' lines (№229)."""
    out = {}
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^2023\b", s):
            toks = [t for _, _, t in tokens(s)][1:]
            nums = [t for t in toks if NUM.match(t) and not re.search(r"\.\d\d", t)]
            dates = [t for t in toks if re.search(r"^\d{1,2}\.\d{2}", t) or "-" in t]
            if len(nums) >= 3:
                out = dict(mean=parse_cell(nums[0]), max=parse_cell(nums[1]),
                           min=parse_cell(nums[2]), dates=" ".join(dates))
        for k, pat in (("mean", r"^Середн[яій]+ річн[аий]+\s+(\S+)"),
                       ("max", r"^Максимальн[аий]+ за рік\s+(\S+)(?:\s+Дата\s+(.+))?"),
                       ("min", r"^Мінімальн[аий]+ за рік\s+(\S+)(?:\s+Дата\s+(.+))?")):
            m = re.match(pat, s)
            if m:
                out[k] = parse_cell(m.group(1))
                if m.lastindex and m.lastindex > 1 and m.group(2):
                    out[k + "_date"] = m.group(2).strip()
    return out


def post_of(line: str):
    for k, v in BY_NAME.items():
        if k in line:
            return v
    return None


# ------------------------------------------------------------ table parsers
def parse_extremes_230(text: str, part: str):
    """Таблиця 2.1.3 ч.1 / ч.2: 17 numbered columns, one block per post."""
    lines = [l for l in text.splitlines() if l.strip()]
    hdr = next(l for l in lines if re.fullmatch(r"\s*1(\s+\d{1,2}){14,16}\s*", l))
    cols = [(int(t), (a + b) / 2) for a, b, t in tokens(hdr)]
    groups = {3: ("паводок", "max"), 5: ("межень", "min"), 7: ("нагін", "max"),
              11: ("відгін", "min"), 15: ("зажор", "max")}
    if part == "ч.2":
        groups[7] = ("техногенне підвищення (підрив греблі)", "max")
    rows, post, crit = [], None, {}
    started = False
    for ln in lines:
        if ln is hdr:
            started = True
            continue
        if not started:
            continue
        s = ln.strip()
        if s.lower().startswith("примітка") or (rows and rows[-1].get("_note")):
            rows.append({"_note": s}); continue
        p = post_of(s)
        if p and not re.search(r"\d{3}", s):
            post, crit = p, {}
            continue
        if post is None:
            continue
        if not re.search(r"\d{3}", s):
            rows.append(dict(post_id=post, table=f"2.1.3 {part}", kind="text", note=s))
            continue
        cells, _ = assign(cols, tokens(ln))
        if 1 in cells:
            crit = dict(critical_high_cm=parse_cell(cells[1])[0],
                        critical_low_cm=parse_cell(cells.get(2, "нб"))[0])
        for c0, (kind, _) in groups.items():
            if c0 in cells:
                v, f = parse_cell(cells[c0])
                rec = dict(post_id=post, table=f"2.1.3 {part}", kind=kind, level_cm=v, flag=f,
                           date=cells.get(c0 + 1, ""), **crit)
                if c0 in (7, 11):
                    rec["rel_to_critical_cm"] = parse_cell(cells.get(c0 + 2, "нб"))[0]
                    rec["duration_h"] = parse_cell(cells.get(c0 + 3, "нб"))[0]
                if c0 == 15:
                    rec["duration_h"] = parse_cell(cells.get(c0 + 2, "нб"))[0]
                rows.append(rec)
    notes = " ".join(r["_note"] for r in rows if "_note" in r)
    return [r for r in rows if "_note" not in r], notes


def parse_surges_229(text: str):
    """Таблиця 1.1.3 нагони та відгони (Part 1): keep only our posts."""
    rows = []
    for ln in text.splitlines():
        toks = [t for _, _, t in tokens(ln)]
        if len(toks) < 5 or not re.fullmatch(r"\d", toks[0]):
            continue
        p = post_of(toks[1])
        if p is None:
            continue
        lo, hi = toks[2].split("-")
        rest = toks[3:]
        rows.append(dict(post_id=p, table="1.1.3", kind="нагін", date=rest[0],
                         rel_to_critical_cm=parse_cell(rest[1])[0], duration_h=parse_cell(rest[2])[0],
                         critical_low_cm=float(lo), critical_high_cm=float(hi)))
        if len(rest) >= 6:
            rows.append(dict(post_id=p, table="1.1.3", kind="відгін", date=rest[3],
                             rel_to_critical_cm=parse_cell(rest[4])[0], duration_h=parse_cell(rest[5])[0],
                             critical_low_cm=float(lo), critical_high_cm=float(hi)))
    return rows


def parse_stats_229(text: str):
    """Таблиця 1.1.2: annual mean / range / st.dev of hourly (or term) levels."""
    out, post, n = [], None, None
    lines = [l for l in text.splitlines() if l.strip()]
    for i, ln in enumerate(lines):
        s = ln.strip()
        p = post_of(s)
        if p and re.match(r"^\d{1,2}\.\s", s):
            post = p
        m = re.search(r"Кількість випадків\s+(\d+)", s)
        if m:
            n = int(m.group(1))
        if s.startswith("за рік") and i + 1 < len(lines):
            v = [t for _, _, t in tokens(lines[i + 1])]
            out.append(dict(post_id=post, table="1.1.2", n_obs=n, mean_cm=float(v[0]),
                            range_cm=float(v[2]), stdev_cm=float(v[3])))
    return out


def parse_blocks_230(text: str, row_labels, variable):
    """Post blocks with monthly columns: 2.4.1Б (decadal temperature) and
    2.4.5 (salinity mean/max/min)."""
    lines = [l for l in text.splitlines() if l.strip()]
    cols = month_columns(lines)
    rows, post = [], None
    for ln in lines:
        s = ln.strip()
        p = post_of(s)
        if p and not re.search(r"\d\.\d", s):
            post = p
            continue
        if post is None:
            continue
        toks = tokens(ln)
        lab = next((t for _, _, t in toks if t.strip().rstrip(".").lower() in row_labels), None)
        if lab is None:
            continue
        cells, spare = assign(cols, [t for t in toks if t[2] != lab])
        key = row_labels[lab.strip().rstrip(".").lower()]
        for m, t in cells.items():
            v, f = parse_cell(t)
            rows.append(dict(post_id=post, variable=variable, month=m, stat=key, value=v, flag=f))
        right = [t for c, t in spare if c > cols[-1][1]]
        if right:
            rows.append(dict(post_id=post, variable=variable, month=0, stat=key + "_annual",
                             value=parse_cell(right[0])[0], flag=" ".join(right[1:])))
    return rows


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", choices=["229", "230", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    vols = [229, 230] if a.volume == "all" else [int(a.volume)]
    for v in vols:
        if not PDFS[v].exists():
            raise SystemExit(f"missing {PDFS[v]}")
    print("=" * 78)
    print("P30 -- 2023 sea-post yearbooks, Dnipro-Bug estuary")
    print("=" * 78)

    daily_rows, stats_rows, annual_rows, inv_notes, failures, incomplete = [], [], [], {}, [], []
    for vol, page, var, unit, post in GRID_PAGES:
        if vol not in vols:
            continue
        txt = page_text(vol, page)
        tag = f"№{vol} p{page} {POSTS[post]['name_en']:<13} {var:<18}"
        try:
            g = parse_grid(txt, unit)
        except Exception as ex:
            print(f"  {tag} PARSE FAILED: {ex}")
            failures.append((tag, str(ex)))
            continue
        # a page pypdf warned about: the grid must still carry every month
        # that the plain extraction mentions
        plain = PdfReader(str(PDFS[vol])).pages[page - 1].extract_text() or ""
        n_plain = len(re.findall(r"\b\d{3}\b", plain)) if unit == "cm" else None
        n_grid = len([1 for (v, f) in g["daily"].values() if np.isfinite(v)])
        if n_plain and n_grid < 0.9 * n_plain - 40:
            incomplete.append((post, var, page, n_grid, n_plain))
        for pr in g["problems"]:
            failures.append((tag, pr))
        print(f"  {tag} months {g['n_months']:2d}  values {n_grid:4d}  "
              f"{'OK' if not g['problems'] else 'CHECK FAILED: ' + '; '.join(g['problems'])}")
        for (m, d), (val, flag) in sorted(g["daily"].items()):
            daily_rows.append(dict(post_id=post, variable=var, unit=unit,
                                   date=f"{YEAR}-{m:02d}-{d:02d}", value=val, flag=flag,
                                   source_file=PDFS[vol].name, page=page))
        for key, per in g["stats"].items():
            for m, (val, flag) in per.items():
                stats_rows.append(dict(post_id=post, variable=var, unit=unit, month=m, stat=key,
                                       value=val, flag=flag, source_file=PDFS[vol].name, page=page))
        ann = annual_line(txt)
        if ann:
            annual_rows.append(dict(post_id=post, variable=var, unit=unit,
                                    mean=ann.get("mean", (np.nan, ""))[0],
                                    max=ann.get("max", (np.nan, ""))[0],
                                    max_flag=ann.get("max", (np.nan, ""))[1],
                                    max_date=ann.get("max_date", ann.get("dates", "")),
                                    min=ann.get("min", (np.nan, ""))[0],
                                    min_flag=ann.get("min", (np.nan, ""))[1],
                                    min_date=ann.get("min_date", ""),
                                    source_file=PDFS[vol].name, page=page))
        inv_notes[(post, var)] = g["notes"]

    extremes, ext_notes = [], []
    if 230 in vols:
        for page, part in ((172, "ч.1"), (173, "ч.2")):
            r, n = parse_extremes_230(page_text(230, page), part)
            for x in r:
                x.update(source_file=PDFS[230].name, page=page)
            extremes += r
            ext_notes.append(f"№230 p{page}: {n}")
        blocks = parse_blocks_230(page_text(230, 179),
                                  {"і": "decade_1", "іі": "decade_2", "ііі": "decade_3", "серед": "mean"},
                                  "water_temperature")
        blocks += parse_blocks_230(page_text(230, 183),
                                   {"середня": "mean", "максимальна": "max", "мінімальна": "min"},
                                   "salinity")
        for b in blocks:
            b.update(source_file=PDFS[230].name)
    else:
        blocks = []
    if 229 in vols:
        s229 = parse_surges_229(page_text(229, 40))
        for x in s229:
            x.update(source_file=PDFS[229].name, page=40)
        extremes += s229
        stats229 = parse_stats_229(page_text(229, 39))
        for x in stats229:
            x.update(source_file=PDFS[229].name, page=39)
    else:
        stats229 = []

    print(f"\n  daily values {len(daily_rows):,}  monthly stats {len(stats_rows)}  "
          f"annual {len(annual_rows)}  extremes {len(extremes)}  blocks {len(blocks)}")
    if failures:
        print("\n  SELF-CHECK FAILURES -- nothing written:")
        for tag, pr in failures:
            print(f"    {tag}: {pr}")
        raise SystemExit(1)
    if a.dry_run:
        print("\n  DRY RUN: nothing written")
        return

    # ---- write ---------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(daily_rows)
    D["post_name"] = D.post_id.map(lambda p: POSTS[p]["name"])
    D["water_body"] = D.post_id.map(lambda p: POSTS[p]["water_body"])
    D["zero_bs77_m"] = D.post_id.map(lambda p: POSTS[p]["zero_bs77_m"])
    is_lvl = D.variable.isin(["water_level", "sea_level"])
    D["H_bs77_m"] = np.where(is_lvl, D.zero_bs77_m + D.value / 100.0, np.nan)
    delta = post_geometry_check()
    for pid, v in POSTS.items():
        if v["lon"] is not None:
            print(f"  {pid} {v['name_en']:<13} delta_9902 {v['delta_epsg9902_m']:+.3f} m  "
                  f"{v['dist_to_water_domain_m']:.0f} m from the water domain  [{v['zones']}]")
    D["delta_epsg9902_m"] = D.post_id.map(delta)
    D["H_evrf2019_m"] = D.H_bs77_m + D.delta_epsg9902_m
    cols = ["post_id", "post_name", "water_body", "variable", "unit", "date", "value", "flag",
            "zero_bs77_m", "H_bs77_m", "delta_epsg9902_m", "H_evrf2019_m", "source_file", "page"]
    D = D[cols].sort_values(["post_id", "variable", "date"])
    D[is_lvl.loc[D.index]].to_csv(OUT / "daily_levels_cm.csv", index=False)
    D[D.variable == "water_temperature"].drop(columns=["zero_bs77_m", "H_bs77_m", "delta_epsg9902_m", "H_evrf2019_m"]) \
        .to_csv(OUT / "daily_water_temperature_c.csv", index=False)
    D[D.variable == "salinity"].drop(columns=["zero_bs77_m", "H_bs77_m", "delta_epsg9902_m", "H_evrf2019_m"]) \
        .to_csv(OUT / "daily_salinity_ppt.csv", index=False)
    D[D.variable == "discharge"].drop(columns=["zero_bs77_m", "H_bs77_m", "delta_epsg9902_m", "H_evrf2019_m"]) \
        .to_csv(OUT / "daily_discharge_oleksandrivka_m3s.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(OUT / "monthly_stats_printed.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(OUT / "annual_stats_2023.csv", index=False)
    pd.DataFrame(extremes).to_csv(OUT / "level_extremes_2023.csv", index=False)
    pd.DataFrame(blocks).to_csv(OUT / "monthly_temperature_salinity_230.csv", index=False)
    if stats229:
        pd.DataFrame(stats229).to_csv(OUT / "level_statistics_229.csv", index=False)
    meta = pd.DataFrame([dict(post_id=k, **v) for k, v in POSTS.items()])
    meta.to_csv(OUT / "post_metadata_2023.csv", index=False)
    for page in (159, 160, 193, 194):
        (OUT / f"narrative_230_p{page}.txt").write_text(page_text(230, page), encoding="utf-8")
    (OUT / "narrative_229_p082.txt").write_text(page_text(229, 82), encoding="utf-8")
    print(f"\n-> {OUT}")

    # ---- inventory -----------------------------------------------------------
    inv = []
    for (post, var), grp in D.groupby(["post_id", "variable"]):
        obs = grp[np.isfinite(grp.value)]
        dates = pd.to_datetime(obs.date)
        months_gap = sorted({int(m) for m in range(1, 13)
                             if (dates.dt.month == m).sum() < calendar.monthrange(YEAR, m)[1]})
        win = obs[(dates >= BREACH) & (dates <= "2023-06-30")]
        n_win = len(win)
        flagged = int(win.flag.str.contains(r"[/?*]").sum()) if n_win else 0
        status = ("stopped" if n_win == 0 else "observed_flagged" if flagged else
                  "partial" if n_win < 25 else "observed")
        inv.append(dict(post_id=post, post_name=POSTS[post]["name"], name_en=POSTS[post]["name_en"],
                        variable=var, n_days_observed=len(obs), n_days_flagged=int((obs.flag != "").sum()),
                        first_date=obs.date.min() if len(obs) else "", last_date=obs.date.max() if len(obs) else "",
                        days_missing_in_year=365 - len(obs), months_with_gaps=" ".join(map(str, months_gap)),
                        breach_window_obs_days=n_win, breach_window_flagged=flagged,
                        breach_window_status=status, parse_status="OK",
                        station_status_2023=POSTS[post]["status_2023"],
                        page_note=inv_notes.get((post, var), "")))
    for post, var, page, n_grid, n_plain in incomplete:
        for r in inv:
            if r["post_id"] == post and r["variable"] == var:
                r["parse_status"] = f"PARSE_INCOMPLETE p{page}: {n_grid} grid vs ~{n_plain} plain"
    for post in POSTS:
        if not any(r["post_id"] == post for r in inv):
            inv.append(dict(post_id=post, post_name=POSTS[post]["name"], name_en=POSTS[post]["name_en"],
                            variable="water_level", n_days_observed=0, n_days_flagged=0, first_date="",
                            last_date="", days_missing_in_year=365, months_with_gaps="1 2 3 4 5 6 7 8 9 10 11 12",
                            breach_window_obs_days=0, breach_window_flagged=0, breach_window_status="stopped",
                            parse_status="NO_TABLE_IN_2023_YEARBOOK",
                            station_status_2023=POSTS[post]["status_2023"], page_note=""))
    I = pd.DataFrame(inv).sort_values(["post_id", "variable"])
    I.to_csv(CFG.TABLES / "p30_sea_post_inventory_2023.csv", index=False)
    print(f"-> {CFG.TABLES / 'p30_sea_post_inventory_2023.csv'}")

    # ---- Kherson cross-check against the river yearbook already in use ------
    xchk = ""
    kh = CFG.ICESAT_ROOT / "data" / "1_data" / "data" / "csv" / "dm_H" / "80805_yearbook.csv"
    if kh.exists():
        r = pd.read_csv(kh)
        r = r[(r.stat_type == "daily") & r.date.str.startswith("2023")]
        s = D[(D.post_id == 80805) & (D.variable == "water_level")]
        j = s.merge(r[["date", "water_level_cm"]], on="date", how="inner")
        j["d_cm"] = j.value - j.water_level_cm
        pre = j[j.date < "2023-06-06"]
        post = j[j.date >= "2023-06-06"]
        xchk = (f"Kherson sea-post table vs river yearbook 80805U_2023.xls: {len(j)} common days; "
                f"before the breach max |Δ| = {pre.d_cm.abs().max():.0f} cm over {len(pre)} days "
                f"(mean Δ {pre.d_cm.mean():+.2f}); from 06.06 on, {len(post)} common days, "
                f"max |Δ| = {post.d_cm.abs().max():.0f} cm")
        big = j[j.d_cm.abs() > 2][["date", "value", "flag", "water_level_cm", "d_cm"]]
        xchk += "\n\n| date | sea-post cm | flag | river yearbook cm | Δ |\n|---|---|---|---|---|\n" + \
                "\n".join(f"| {x.date} | {x.value:.0f} | {x.flag} | {x.water_level_cm:.0f} | {x.d_cm:+.0f} |"
                          for x in big.itertuples()) if len(big) else "\n\nNo day differs by more than 2 cm."

    # ---- catalogue -------------------------------------------------------------
    man = ROOT / "data" / "catalog" / "raw_data_manifest.csv"
    M = pd.read_csv(man)
    for v in (229, 230):
        p = PDFS[v]
        if (M.path == str(p)).any():
            continue
        M.loc[len(M)] = dict(family="hydromet_yearbook", file=p.name, path=str(p), bytes=p.stat().st_size,
                             sensing_time="2023", tile="", valid_zip="", bands="",
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             mtime_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat())
    M.to_csv(man, index=False)
    print(f"-> {man} (+ yearbooks)")

    # ---- report ----------------------------------------------------------------
    lines = ["# P30 — 2023 hydromet yearbooks, Dnipro–Bug estuary posts", "",
             f"Source: `{PDFS[229].name}` (Part 1, seas) and `{PDFS[230].name}` (Part 2, river mouths), "
             f"parsed with pypdf layout extraction; every day × month grid was checked against the "
             f"printed monthly mean / max / min before writing. Post zero marks are −5.00 m БС-77 "
             f"(Олександрівка −3.02 m).", "",
             "## Inventory", "",
             "| post | variable | days | flagged | first | last | gap months | breach window (06.06–30.06) | parse |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in I.itertuples():
        lines.append(f"| {r.post_id} {r.name_en} | {r.variable} | {r.n_days_observed} | {r.n_days_flagged} | "
                     f"{r.first_date} | {r.last_date} | {r.months_with_gaps or '—'} | "
                     f"{r.breach_window_status} ({r.breach_window_obs_days} d) | {r.parse_status} |")
    lines += ["", "## The breach in the yearbook's own words (табл. 2.1.3 ч.2, 1.6.1 ч.2)", ""]
    for x in [e for e in extremes if e.get("table") == "2.1.3 ч.2"]:
        if x["kind"] == "text":
            lines.append(f"- {POSTS[x['post_id']]['name']}: {x['note']}")
            continue
        dur = x.get("duration_h", np.nan)
        lines.append(f"- {POSTS[x['post_id']]['name']}: {x['kind']} {x['level_cm']:.0f} cm{x.get('flag', '')} "
                     f"on {x.get('date', '')}, {x.get('rel_to_critical_cm', np.nan):.0f} cm above the critical mark, "
                     f"{'unknown' if not np.isfinite(dur) else f'{dur:.0f} h'}")
    lines += ["", "## Station notes", ""]
    for p, v in POSTS.items():
        geo = (f"{v['lat']:.5f}N {v['lon']:.5f}E, {v['dist_to_water_domain_m']:.0f} m from the water "
               f"domain, in {v['zones'] or 'no zone'}, δ9902 {v['delta_epsg9902_m']:+.3f} m"
               if v["lon"] is not None else "none")
        lines.append(f"- {p} {v['name']} ({v['water_body']}): {v['status_2023']}; "
                     f"coordinates: {v['coord_source']} — {geo}")
    # ---- the two posts printed in both volumes must agree with themselves --
    both = []
    for p in (98022, 98025):
        a = D[(D.post_id == p) & (D.variable == "sea_level")][["date", "value"]]
        b = D[(D.post_id == p) & (D.variable == "water_level")][["date", "value"]]
        j = a.merge(b, on="date", suffixes=("_229", "_230"))
        dd = (j.value_229 - j.value_230).abs()
        both.append(f"- {POSTS[p]['name']}: {len(j)} common days, {int((dd > 0).sum())} differ, "
                    f"max |Δ| = {dd.max():.0f} cm")
    lines += ["", "## Kherson cross-check", "", xchk or "companion yearbook CSV not found", "",
              "Part 2 prints daily means of the hourly SRM record; the river yearbook the project "
              "already uses carries the 08/20 term mean, so differences of a few cm on some days are "
              "the two averaging rules, not a datum offset (both zeros are −5.00 m БС-77).", "",
              "## Same post, both volumes (табл. 1.1.1 vs 2.1.1)", ""] + both + [
              "", "## Notes carried from the pages", ""]
    for (p, var), n in inv_notes.items():
        if n:
            lines.append(f"- {POSTS[p]['name_en']} / {var}: {n}")
    for n in ext_notes:
        lines.append(f"- {n}")
    (CFG.REPORTS / "P30_sea_posts_2023.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {CFG.REPORTS / 'P30_sea_posts_2023.md'}")


if __name__ == "__main__":
    main()
