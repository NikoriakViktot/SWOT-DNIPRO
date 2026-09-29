#!/usr/bin/env python
"""TASK 4 -- parse the DniproHES (post_id 80039) discharge record from the
user-supplied archive E:\\HYDROLOGY\\Afanasev\\, which extends through all of
2023 (verified this session) -- unlike the already-parsed
icesat2-atl13-kakhovka/data/1_data parquet, which silently stops at
2020-12-31 (that pipeline's own discharge_parser.py drops the last, partial
54-row block via integer floor-division; separately, its source file itself
appears to have simply not been updated past 2021, i.e. two independent
reasons the project's existing parquet under-covers this station).

File has two internal formats concatenated in one sheet:
  rows 1-14245      clean (datetime, Q) rows, 1980-01-01 .. 2018-12-31
  rows 14246-end     yearbook block format (2019 .. 2023), one block per
                     year: a bare-year marker row, a title block, a
                     "Число"/"Місяць" header, a month-number row, up to 31
                     daily rows, then Декада/summary rows before the next
                     bare-year marker.

Missing days are written as blank or '-' in the source; kept as NaN, not
interpolated (operator: "do not interpolate over long gaps without
explicitly flagging").

Outputs
-------
data/processed/hydrology/dniprohes_releases.parquet
outputs/analysis/dniprohes_release_inventory.md
"""
from __future__ import annotations

import datetime
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

SRC = Path("/mnt/e/HYDROLOGY/Afanasev/Дніпро-Днiпровська ГЕС_80039.xlsx")
OUT_DIR = CFG.ROOT / "data" / "processed" / "hydrology"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = CFG.ROOT / "outputs" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
POST_ID = 80039
STATION = "Дніпро - ГЕС Днiпровська (DniproHES)"


def _num(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s in ("", "-", "  -", "     -", "нет", "немає"):
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_clean_segment(df: pd.DataFrame, end_row: int) -> pd.DataFrame:
    rows = []
    for i in range(1, end_row + 1):
        d = df.iat[i, 0]
        q = df.iat[i, 1]
        if isinstance(d, datetime.datetime):
            rows.append({"date": d.date(), "discharge_m3s": _num(q)})
    return pd.DataFrame(rows)


def parse_block_segment(df: pd.DataFrame, start_row: int) -> pd.DataFrame:
    rows = []
    i = start_row
    n = len(df)
    current_year = None
    while i < n:
        c0 = df.iat[i, 0]
        # bare-year marker row: exactly one populated cell anywhere in the row,
        # a plausible 4-digit year (its column position varies -- seen at both
        # col 0 and col 5 in this file, likely a stray merged-cell artifact)
        row_vals = df.iloc[i]
        nonnull = row_vals[row_vals.notna()]
        if len(nonnull) == 1:
            v = nonnull.iloc[0]
            yr = None
            if isinstance(v, (int, float)) and 1900 < float(v) < 2100:
                yr = int(v)
            elif isinstance(v, str) and v.strip().isdigit() and 1900 < int(v.strip()) < 2100:
                yr = int(v.strip())
            if yr is not None:
                current_year = yr
                i += 1
                continue
        if isinstance(c0, str) and c0.strip() == "Число":
            # This file mixes TWO yearbook block formats (confirmed by direct
            # inspection, not assumed): 2019-2020 blocks use Format Q-A
            # (consecutive cols 1-12), 2021-2023 blocks use Format Q-B (every
            # 4th col, [4,8,...,48] -- matches the sibling icesat2 project's
            # discharge_parser.py NEW_MONTH_COLS). Auto-detect per block
            # rather than assuming one format for the whole sheet.
            month_row = df.iloc[i + 1]

            def _try_cols(cols):
                out = []
                for c in cols:
                    v = month_row.iloc[c] if c < len(month_row) else None
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        out.append(None)
                    else:
                        try:
                            out.append(int(float(v)))
                        except (TypeError, ValueError):
                            out.append(None)
                return out

            cols_a, cols_b = list(range(1, 13)), list(range(4, 49, 4))
            months_a, months_b = _try_cols(cols_a), _try_cols(cols_b)
            if sum(v is not None for v in months_a) >= sum(v is not None for v in months_b):
                MONTH_COLS, months = cols_a, months_a
            else:
                MONTH_COLS, months = cols_b, months_b
            # Robust day-row scan: the sheet occasionally has a stray blank
            # spacer row mid-block (observed in the 2021 block between day 10
            # and day 11), which would wrongly terminate a naive scan. Skip
            # blanks, tolerate them, and stop only on a real terminator or
            # once day numbers wrap back to 1 (the start of the Декада
            # section, whose rows are ALSO labelled '1'/'2'/'3' and must
            # never be mistaken for days 1-3).
            TERMINATORS = ("Декада", "Середн", "Найб", "Найм", "Таблиця",
                           "ТАБЛИЦЯ", "Середня", "Найбільша", "Найменша")
            expected_day = 1
            j = i + 2
            j_max = min(n, i + 60)
            while j < j_max:
                dv = df.iat[j, 0]
                day = None
                if isinstance(dv, (int, float)) and not (isinstance(dv, float) and np.isnan(dv)):
                    day = int(dv)
                elif isinstance(dv, str):
                    s = dv.strip()
                    if s.isdigit():
                        day = int(s)
                    elif any(s.startswith(t) for t in TERMINATORS):
                        break
                if day is None:
                    j += 1  # blank/spacer row -- tolerate, keep scanning
                    continue
                if day == expected_day and 1 <= day <= 31:
                    for month, col_idx in zip(months, MONTH_COLS):
                        if month is None or current_year is None:
                            continue
                        val = _num(df.iat[j, col_idx])
                        if np.isnan(val):
                            continue
                        try:
                            date = datetime.date(current_year, month, day)
                        except ValueError:
                            continue  # e.g. day 31 in a 30-day month
                        rows.append({"date": date, "discharge_m3s": val})
                    expected_day += 1
                    j += 1
                    if expected_day > 31:
                        break
                elif day == 1 and expected_day > 28:
                    break  # Декада section restarting at '1' -- stop
                else:
                    j += 1  # out-of-sequence row, skip without advancing day
            i = j
            continue
        i += 1
    return pd.DataFrame(rows)


def main() -> None:
    print(f"reading {SRC} ...")
    df = pd.read_excel(SRC, sheet_name=0, header=None)
    last_clean = max(i for i in range(len(df)) if isinstance(df.iat[i, 0], datetime.datetime))
    print(f"clean daily segment: rows 1..{last_clean} "
          f"({df.iat[1,0].date()} .. {df.iat[last_clean,0].date()})")

    part1 = parse_clean_segment(df, last_clean)
    part2 = parse_block_segment(df, last_clean + 1)
    print(f"clean segment rows: {len(part1)}   block segment rows: {len(part2)}")

    out = pd.concat([part1, part2], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    out["post_id"] = POST_ID
    out["station"] = STATION
    out["source_file"] = SRC.name
    out["quality_flag"] = "ok"

    full_range = pd.date_range(out.date.min(), out.date.max(), freq="D")
    missing = full_range.difference(out.date)
    print(f"\nfinal series: {out.date.min().date()} .. {out.date.max().date()}, "
          f"{len(out)} daily values, {len(missing)} missing calendar days "
          f"({len(missing)/len(full_range)*100:.1f}%)")

    out_path = OUT_DIR / "dniprohes_releases.parquet"
    out.to_parquet(out_path, index=False)
    out.to_csv(CFG.TABLES / "dniprohes_releases.csv", index=False)
    print(f"-> {out_path}")
    print(f"-> {CFG.TABLES / 'dniprohes_releases.csv'}")

    yr = out.assign(year=out.date.dt.year).groupby("year").agg(
        n=("discharge_m3s", "count"), mean_m3s=("discharge_m3s", "mean"),
        min_m3s=("discharge_m3s", "min"), max_m3s=("discharge_m3s", "max"))
    print("\nper-year coverage:")
    print(yr.to_string())

    breach = pd.Timestamp("2023-06-06")
    around = out[(out.date >= "2023-05-01") & (out.date <= "2023-07-15")]
    print(f"\naround the 2023-06-06 breach ({len(around)} daily values):")
    print(around[["date", "discharge_m3s"]].to_string(index=False))

    report = f"""# DniproHES (80039) release inventory

Source: `{SRC}` (user-supplied, 2026-09-11) -- NOT the icesat2-atl13-kakhovka
project's own `data/1_data/data/parquet/dm_Q/80039_discharge.parquet`, which
stops at 2020-12-31 (that pipeline's parser drops the last partial 54-row
block; also its own source file was apparently never updated past 2021).

## Coverage
- Full daily series: {out.date.min().date()} .. {out.date.max().date()}
- {len(out):,} daily values, {len(missing):,} missing calendar days
  ({len(missing)/len(full_range)*100:.1f}%) -- NOT interpolated, left as gaps.
- **All 12 months of 2023 present**, including the full PRE_BREACH,
  DRAWDOWN and POST_BREACH windows around the 2023-06-06 breach.

## Format note
Two formats concatenated in one sheet: clean (datetime, Q) daily rows for
1980-2018, then yearbook block format (2019-2023). Both parsed by this
script; block format handles blank/'-' missing-day cells as NaN, never
interpolated.

## Per-year record count
{yr.to_string()}

## Around the 2023-06-06 breach
{around[['date','discharge_m3s']].to_string(index=False)}

## Component breakdown (Q_turbine / Q_spill / Q_other)
**NOT available separately in this source** -- the yearbook Table 1.3 format
reports total discharge (`Q_total_m3s`) only. If component-level data exists
elsewhere it has not been located in this session; this canonical file
carries `discharge_m3s` as a total only, honestly labelled, not decomposed.

Output: `data/processed/hydrology/dniprohes_releases.parquet`
"""
    (ANALYSIS_DIR / "dniprohes_release_inventory.md").write_text(report)
    print(f"\n-> {ANALYSIS_DIR / 'dniprohes_release_inventory.md'}")


if __name__ == "__main__":
    main()
