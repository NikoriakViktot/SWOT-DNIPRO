#!/usr/bin/env python
"""V2 -- raw-vs-parquet audit of dniprohes_releases.parquet against the
source Excel, cell by cell. Verification only -- does not touch the parser
or the parquet.

Outputs
-------
outputs/tables/dniprohes_q_raw_validation.csv
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
PARQUET = CFG.ROOT / "data" / "processed" / "hydrology" / "dniprohes_releases.parquet"

# 5 dates per sampled year, spread across months (incl. the format-transition
# boundary 2020-12 / 2021-01 / 2021-02, and a leap day)
SAMPLE_DATES = [
    "1980-01-01", "1980-04-15", "1980-07-04", "1980-09-30", "1980-12-31",
    "1990-02-14", "1990-05-20", "1990-06-30", "1990-08-11", "1990-11-01",
    "2000-01-15", "2000-02-29", "2000-06-01", "2000-09-09", "2000-12-25",
    "2010-03-03", "2010-05-05", "2010-07-07", "2010-08-30", "2010-10-10",
    "2018-01-01", "2018-04-04", "2018-06-15", "2018-09-01", "2018-12-31",
    "2019-01-01", "2019-05-05", "2019-06-05", "2019-08-25", "2019-12-31",
    "2020-01-01", "2020-06-10", "2020-09-19", "2020-11-30", "2020-12-31",
    "2021-01-01", "2021-01-15", "2021-02-01", "2021-06-01", "2021-12-31",
    "2022-01-01", "2022-04-26", "2022-08-09", "2022-10-18", "2022-12-31",
    "2023-01-01", "2023-05-06", "2023-06-05", "2023-06-06", "2023-12-31",
]


def _num(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s in ("", "-", "нет", "немає"):
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def find_raw_value_clean(df: pd.DataFrame, target: datetime.date):
    for i in range(1, 14246):
        v = df.iat[i, 0]
        if isinstance(v, datetime.datetime) and v.date() == target:
            return i, 0, 1, _num(df.iat[i, 1]), "clean_daily"
    return None


def find_raw_value_block(df: pd.DataFrame, target: datetime.date):
    n = len(df)
    current_year = None
    i = 14246
    while i < n:
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
        c0 = df.iat[i, 0]
        if isinstance(c0, str) and c0.strip() == "Число" and current_year == target.year:
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
            ma, mb = _try_cols(cols_a), _try_cols(cols_b)
            if sum(v is not None for v in ma) >= sum(v is not None for v in mb):
                month_cols, months, fmt = cols_a, ma, "A"
            else:
                month_cols, months, fmt = cols_b, mb, "B"
            if target.month in months:
                col_idx = month_cols[months.index(target.month)]
                j = i + 2
                for _ in range(35):
                    dv = df.iat[j, 0]
                    day = None
                    if isinstance(dv, (int, float)) and not (isinstance(dv, float) and np.isnan(dv)):
                        day = int(dv)
                    elif isinstance(dv, str) and dv.strip().isdigit():
                        day = int(dv.strip())
                    if day == target.day:
                        return j, col_idx, fmt, _num(df.iat[j, col_idx]), f"block_{fmt}"
                    j += 1
        elif isinstance(c0, str) and c0.strip() == "Число" and current_year is not None \
                and current_year > target.year:
            break
        i += 1
    return None


def main() -> None:
    print(f"reading {SRC} ...")
    raw = pd.read_excel(SRC, sheet_name=0, header=None)
    par = pd.read_parquet(PARQUET).set_index("date")["discharge_m3s"]

    rows = []
    for ds in SAMPLE_DATES:
        target = pd.Timestamp(ds).date()
        parsed_val = par.get(pd.Timestamp(ds), np.nan)
        if target.year <= 2018:
            hit = find_raw_value_clean(raw, target)
        else:
            hit = find_raw_value_block(raw, target)
        if hit is None:
            rows.append({"date": ds, "raw_row": None, "raw_col": None,
                        "raw_value": np.nan, "parsed_value": parsed_val,
                        "difference": np.nan, "format_type": "NOT_FOUND",
                        "PASS_FAIL": "FAIL_NOT_FOUND_IN_RAW"})
            continue
        row, col, fmt, raw_val, fmt_label = hit
        diff = (parsed_val - raw_val) if (np.isfinite(parsed_val) and np.isfinite(raw_val)) else np.nan
        both_nan = not np.isfinite(parsed_val) and not np.isfinite(raw_val)
        passed = both_nan or (np.isfinite(diff) and abs(diff) < 1e-6)
        rows.append({"date": ds, "raw_row": row, "raw_col": col,
                    "raw_value": raw_val, "parsed_value": parsed_val,
                    "difference": diff, "format_type": fmt_label,
                    "PASS_FAIL": "PASS" if passed else "FAIL_VALUE_MISMATCH"})

    out = pd.DataFrame(rows)
    out_path = CFG.TABLES / "dniprohes_q_raw_validation.csv"
    out.to_csv(out_path, index=False)
    print(f"-> {out_path}")
    print(out.PASS_FAIL.value_counts().to_string())
    fails = out[out.PASS_FAIL != "PASS"]
    if len(fails):
        print("\nFAILURES:")
        print(fails.to_string(index=False))
    else:
        print(f"\nALL {len(out)} SAMPLED DATES PASS (raw == parsed, incl. NaN==NaN).")


if __name__ == "__main__":
    main()
