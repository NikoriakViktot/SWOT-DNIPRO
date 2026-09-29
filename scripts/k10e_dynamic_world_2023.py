#!/usr/bin/env python
"""K10e.V4 -- Dynamic World probabilities for every 2023 ICESat-2 segment.

Requires the `geoid` conda env (earthengine-api with valid credentials at
~/.config/earthengine/credentials -- established this session via the
'notebook' OAuth flow after the earlier refresh_token had expired).

Runs AFTER k10e_point_attribution_2023.py: it reuses that script's per-segment
`sentinel_datetime` (the same Sentinel-2 scene date each segment was already
matched to for NDVI/NDWI/MNDWI/NDMI/BSI), so the Dynamic World read and the
spectral read are date-consistent by construction, not independently nearest-
matched (which could silently pick two different dates for the same point).

Retains full DW probabilities, not just the top-1 label (per spec V4):
    water_probability, trees_probability, grass_probability,
    flooded_vegetation_probability, crops_probability,
    shrub_scrub_probability, built_probability, bare_probability,
    snow_and_ice_probability, dw_label, dw_max_probability

Batched via Earth Engine reduceRegions per scene-date (not point-by-point),
sub-batched at BATCH_N points to stay well inside a single synchronous
getInfo() response.

Outputs
-------
data/processed/current_bed/k10e_post_breach_dynamic_world.parquet   (row_id, DW fields)
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

CUR = CFG.ROOT / "data" / "processed" / "current_bed"
DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
           "shrub_and_scrub", "built", "bare", "snow_and_ice"]
BATCH_N = 800


def main() -> None:
    import ee
    ee.Initialize(project="ee-nikoriakviktor")

    print("=" * 78)
    print("K10e.V4 -- Dynamic World probabilities for 2023 ICESat-2 segments")
    print("=" * 78)

    g = pd.read_parquet(CUR / "k10e_post_breach_vegetation_context.parquet")
    g["row_id"] = np.arange(len(g))
    have_scene = g.sentinel_datetime.notna()
    print(f"segments with a matched Sentinel-2 date (from point attribution): "
          f"{int(have_scene.sum()):,}/{len(g):,}")

    out_cols = {f"{b}_probability": np.full(len(g), np.nan) for b in DW_BANDS}
    out_cols["dw_label"] = np.full(len(g), np.nan)
    out_cols["dw_max_probability"] = np.full(len(g), np.nan)
    out_cols["dw_date_used"] = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")

    sub = g[have_scene]
    day_key = pd.to_datetime(sub.sentinel_datetime).dt.normalize()
    for day, idx in sub.groupby(day_key).groups.items():
        idx = np.asarray(idx)
        d1 = ee.Date(str(pd.Timestamp(day).date()))
        d2 = d1.advance(1, "day")
        n_avail = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(d1, d2).size().getInfo()
        if n_avail == 0:
            print(f"  {pd.Timestamp(day).date()}: NO Dynamic World image available, "
                  f"{len(idx):,} segments left unattributed for this date", flush=True)
            continue
        dw_img = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(d1, d2)
                 .mosaic().select(DW_BANDS + ["label"]))

        lon = g.loc[idx, "lon"].values
        lat = g.loc[idx, "lat"].values
        n_done = 0
        for a in range(0, len(idx), BATCH_N):
            b = idx[a:a + BATCH_N]
            feats = [ee.Feature(ee.Geometry.Point(float(lo), float(la)), {"row_id": int(i)})
                     for i, lo, la in zip(b, lon[a:a + BATCH_N], lat[a:a + BATCH_N])]
            fc = ee.FeatureCollection(feats)
            sampled = dw_img.reduceRegions(fc, ee.Reducer.first(), 10).getInfo()
            for f in sampled["features"]:
                p = f["properties"]
                rid = p["row_id"]
                # EE returns JSON null (-> Python None) for a masked/no-data
                # pixel, not NaN; np.array([None, 0.3, ...]) becomes dtype=object
                # and np.isfinite() raises TypeError on it, rather than treating
                # None as missing.
                probs = np.array([v if (v := p.get(bd)) is not None else np.nan
                                 for bd in DW_BANDS], dtype=float)
                for bd, val in zip(DW_BANDS, probs):
                    out_cols[f"{bd}_probability"][rid] = val
                if np.isfinite(probs).any():
                    out_cols["dw_max_probability"][rid] = np.nanmax(probs)
                out_cols["dw_label"][rid] = p.get("label", np.nan)
                out_cols["dw_date_used"][rid] = pd.Timestamp(day)
            n_done += len(b)
        print(f"  {pd.Timestamp(day).date()}: {n_done:,}/{len(idx):,} segments sampled",
              flush=True)

    dw = pd.DataFrame({"row_id": g.row_id.values, **out_cols})
    n_ok = dw.dw_max_probability.notna().sum()
    print(f"\nDynamic World attribution: {n_ok:,}/{len(g):,} segments "
          f"({100*n_ok/len(g):.1f}%)")

    LABELS = {0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation", 4: "crops",
             5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice"}
    dw["dw_label_name"] = dw.dw_label.map(lambda v: LABELS.get(int(v)) if np.isfinite(v) else None)
    print("\nDW top-1 label distribution (2023 accepted segments):")
    print(dw.dw_label_name.value_counts().to_string())

    op = CUR / "k10e_post_breach_dynamic_world.parquet"
    dw.to_parquet(op, index=False)
    print(f"\n-> {op}")


if __name__ == "__main__":
    main()
