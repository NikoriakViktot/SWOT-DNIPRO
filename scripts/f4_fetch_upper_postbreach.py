#!/usr/bin/env python
"""F4 (download step) -- fetch T36UXU (+ T36TXT if not already cached) for
the 15 selected post-breach upper-AOI dates. Reuses the proven serial
retry/backoff downloader from p1_targeted_fetch.py (5 concurrent connections
triggered CDSE 429s earlier this session; serial is what works).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from swot_dnipro import config as CFG
from p1_targeted_fetch import TokenBox, fetch_one, already_have

TILES = ("36TXT", "36UXU")


def main() -> None:
    sel = pd.read_csv(CFG.TABLES / "f3_selected_dates.csv")
    rows = []
    for d in sel.date:
        for t in TILES:
            rows.append({"date": d, "tile_id": t})
    todo_df = pd.DataFrame(rows)

    box = TokenBox()
    results = []
    for i, r in enumerate(todo_df.itertuples(), 1):
        res = fetch_one(r, box)
        results.append(res)
        print(f"[{i}/{len(todo_df)}] {res['status']}", flush=True)

    out = pd.DataFrame(results)
    out_path = CFG.ROOT / "data" / "catalog" / "F4_upper_postbreach_fetch_log.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print(out.status.value_counts().to_string())


if __name__ == "__main__":
    main()
