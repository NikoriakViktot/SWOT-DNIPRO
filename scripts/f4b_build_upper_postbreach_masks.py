#!/usr/bin/env python
"""F4 continued -- build water masks for the 15 post-breach upper-AOI dates
(T36TXT + T36UXU), reusing p1g_build_masks.py's helper functions.
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
from p1g_build_masks import find_zip, _geotiff, _npz, MASKS
from swot_dnipro import watermask as WM


def main() -> None:
    man = pd.read_csv(CFG.TABLES / "F4_upper_masks_manifest.csv")
    rows = []
    for i, r in enumerate(man.itertuples(), 1):
        zp = find_zip(r.date, r.tile_id)
        if zp is None:
            print(f"[{i}/{len(man)}] {r.date} {r.tile_id}: NOT ON DISK")
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": "NO_RAW_FILE"})
            continue
        stem = zp.name.replace(".zip", "")
        npz_path, tif_path = MASKS / f"{stem}.npz", MASKS / f"{stem}_water.tif"
        if npz_path.exists() and tif_path.exists():
            print(f"[{i}/{len(man)}] {r.date} {r.tile_id}: already masked")
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": "ALREADY_MASKED", "scene": stem})
            continue
        try:
            wm = WM.build_mask(zp, 0.0, 0.0)
            lab, nlab = WM.label_water_bodies(wm)
        except Exception as exc:
            print(f"[{i}/{len(man)}] {r.date} {r.tile_id}: FAILED {exc}")
            rows.append({"date": r.date, "tile_id": r.tile_id, "status": f"FAILED:{exc}"})
            continue
        _npz(npz_path, wm, lab)
        _geotiff(tif_path, wm)
        px_m = abs(wm.transform.a)
        water_km2 = float(wm.mask.sum() * px_m**2 / 1e6)
        print(f"[{i}/{len(man)}] {r.date} {r.tile_id}: water={water_km2:.1f} km2, {nlab} comps -> {stem}")
        rows.append({"date": r.date, "tile_id": r.tile_id, "status": "OK", "scene": stem,
                    "water_km2": water_km2})
    out = pd.DataFrame(rows)
    out_path = CFG.ROOT / "data" / "catalog" / "F4_upper_masks_log.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print(out.status.value_counts().to_string())


if __name__ == "__main__":
    main()
