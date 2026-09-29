#!/usr/bin/env python
"""Phase 19 — expansion comparison: n=4 baseline vs the imagery-expanded rerun.

Reads the pre-expansion snapshot (data/.../_snapshot_pre_expanded_rerun or an
explicit --old dir) and the current Phase 19 tables, and prints / writes the
headline before/after table the operator asked for (2026-09-08).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import bootstrap_ci, nmad

OLD_DIR = Path("/mnt/e/data_swot/_snapshot_pre_expanded_rerun/tables")
NEW_DIR = CFG.TABLES


def load(dirp):
    p = dirp / "channel_only_profiles.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, parse_dates=["date"])
    return d[(d.period == "POST_BREACH") & (d.subset == "MAIN_CHANNEL")]


def summarise(post, label):
    s = post.slope_theilsen_cm_km.to_numpy(float) if len(post) else np.array([])
    row = {"dataset": label, "usable_channel_dates": len(s)}
    if len(s):
        row["median_theilsen_cm_km"] = float(np.median(s))
        lo, hi = bootstrap_ci(s, seed=CFG.SEED)
        row["boot_ci_lo"], row["boot_ci_hi"] = lo, hi
        row["nmad_cm_km"] = float(nmad(s))
        k = int((s > 0).sum())
        row["positive"] = f"{k}/{len(s)}"
        row["sign_test_p"] = float(stats.binomtest(k, len(s), 0.5).pvalue)
        row["median_span_km"] = float(np.median(post.span_km))
        if "temporal_conf" in post:
            vc = post.temporal_conf.value_counts()
            row["temporal_conf"] = " ".join(f"{k_}:{v_}" for k_, v_ in vc.sort_index().items())
            ab = post[post.temporal_conf.isin(["A", "B"])].slope_theilsen_cm_km
            if len(ab) >= 2:
                row["median_AB_only_cm_km"] = float(np.median(ab))
                row["n_AB"] = len(ab)
    return row


def main() -> None:
    old = load(OLD_DIR)
    new = load(NEW_DIR)
    rows = [summarise(old, f"baseline ({OLD_DIR.parent.name})"),
            summarise(new, "expanded (current)")]
    out = pd.DataFrame(rows)
    out.to_csv(NEW_DIR / "phase19_expansion_comparison.csv", index=False)

    pd.set_option("display.width", 240)
    print("=== Phase 19 channel-gradient: baseline vs imagery-expanded ===\n")
    print(out.to_string(index=False))

    if len(new):
        print("\n--- expanded per-date MAIN_CHANNEL profiles ---")
        cols = [c for c in ["date", "n_segments", "n_chain_bins", "span_km",
                            "slope_theilsen_cm_km", "slope_ols_cm_km", "ts_lo_cm_km",
                            "ts_hi_cm_km", "median_lag_days", "temporal_conf"] if c in new]
        print(new[cols].round(2).to_string(index=False))

    print(f"\n-> {NEW_DIR/'phase19_expansion_comparison.csv'}")


if __name__ == "__main__":
    main()
