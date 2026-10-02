#!/usr/bin/env python
"""K10e -- extend the water-mask cache with the 3 additional 2023 scenes that
close the October/November gap identified in k10e_point_attribution_2023.py.

The existing 20-scene cache (built by phase19_watermasks.py, which only reads
from data/raw/sentinel and a pre-verified inventory) ends at 2023-09-08. 3367
of 11,399 water-QC-accepted 2023 ICESat-2 segments fall in October and were
therefore UNATTRIBUTED (no scene within the 20-day window) -- not because of
vegetation, but because of this coverage gap. The 3 missing scenes already
exist on disk in data/raw/sentinel_p1_targeted/:

    2023-09-23  S2A ... T36TXT, T36UXU
    2023-10-03  S2A ... T36TXT, T36UXU
    2023-11-07  S2B ... T36TXT, T36UXU

This does NOT modify phase19_watermasks.py or its inventory-driven selection
logic -- it reuses the same WM.build_mask() / WM.label_water_bodies() calls
and appends to the SAME cache files (water_masks/*.npz, water_mask_summary.csv)
in the identical schema, so k10e_spectral_stack_2023.py picks them up with no
changes.

Outputs
-------
data/processed/water_masks/<scene>.npz            (3 new, or fewer if a tile
                                                     does not cover the domain)
outputs/tables/water_mask_summary.csv              (appended, existing rows kept)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import watermask as WM

MASKS = CFG.ROOT / "data/processed/water_masks"
P1 = CFG.ROOT / "data/raw/sentinel_p1_targeted"
OPERATING = (0.0, 0.0)

NEW_SCENES = [
    "S2A_MSIL2A_20230923T083641_N0510_R064_T36TXT_20241028T095503.SAFE",
    "S2A_MSIL2A_20230923T083641_N0510_R064_T36UXU_20241028T095503.SAFE",
    "S2A_MSIL2A_20231003T083801_N0510_R064_T36TXT_20241028T044108.SAFE",
    "S2A_MSIL2A_20231003T083801_N0510_R064_T36UXU_20241028T044108.SAFE",
    "S2B_MSIL2A_20231107T084049_N0510_R064_T36TXT_20241110T191746.SAFE",
    "S2B_MSIL2A_20231107T084049_N0510_R064_T36UXU_20241110T191746.SAFE",
]


def _save_npz(path, wm, lab):
    a = wm.transform
    np.savez_compressed(
        path, mask=wm.mask, labels=lab.astype("i4"),
        affine=np.array([a.a, a.b, a.c, a.d, a.e, a.f], float),
        crs=str(wm.crs), tile=wm.tile, sensing=wm.sensing_time,
        ndwi=wm.ndwi.astype("f2"), mndwi=wm.mndwi.astype("f2"), scl=wm.scl.astype("i1"))


def main() -> None:
    print("=" * 78)
    print("K10e -- extending water-mask cache: 2023-09-23 / 2023-10-03 / 2023-11-07")
    print("=" * 78)
    print("Closes the Oct/Nov 2023 attribution gap. phase19_watermasks.py and its")
    print("inventory are untouched -- this appends to the same cache in the same schema.\n")

    summary_path = CFG.TABLES / "water_mask_summary.csv"
    existing = pd.read_csv(summary_path)
    print(f"existing water_mask_summary.csv rows: {len(existing)}")

    rows = []
    for name in NEW_SCENES:
        if (existing.name == name).any():
            print(f"  {name[:50]:50s}  already in summary, skipped")
            continue
        zp = P1 / f"{name}.zip"
        if not zp.exists():
            print(f"  {name[:50]:50s}  MISSING zip")
            continue
        wm = WM.build_mask(zp, *OPERATING)
        lab, nlab = WM.label_water_bodies(wm)
        _save_npz(MASKS / f"{name}.npz", wm, lab)

        px_m = abs(wm.transform.a)
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        largest = int(sizes.max()) if sizes.size > 1 else 0
        valid = ~np.isin(wm.scl, WM.SCL_REJECT)
        rows.append({
            "name": name, "tile": wm.tile, "sensing_time": wm.sensing_time,
            "target_date": "", "regime": "", "cloud_cover_meta": np.nan,
            "px_m": px_m, "grid": f"{wm.mask.shape[1]}x{wm.mask.shape[0]}",
            "valid_frac": float(valid.mean()),
            "water_px": int(wm.mask.sum()), "water_km2": float(wm.mask.sum() * px_m**2 / 1e6),
            "n_components": int(nlab), "largest_component_px": largest,
            "largest_component_km2": float(largest * px_m**2 / 1e6),
        })
        print(f"  {name[:50]:50s}  OK  water {rows[-1]['water_km2']:.1f} km2, "
              f"valid_frac {rows[-1]['valid_frac']:.2f}", flush=True)

    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(summary_path, index=False)
        print(f"\n-> {summary_path}  ({len(existing)} -> {len(out)} rows)")
    else:
        print("\nno new rows added")
    print(f"-> {MASKS}  ({len(list(MASKS.glob('*.npz')))} npz total)")


if __name__ == "__main__":
    main()
