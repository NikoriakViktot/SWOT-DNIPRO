#!/usr/bin/env python
"""MS10 -- how much of the Sentinel-2 fragmentation result depends on the mask parameters.

Section 4.5 reports the loss of the single connected pool from one coverage-qualified
pre-breach date (2023-06-05) and four post-breach dates: 2 -> median 503 water bodies,
largest component 99.995 % -> 58.2 %. Those counts come from one frozen rule (NDWI > 0 and
MNDWI > 0 with the SCL veto, 8-connectivity, minimum mapping unit 0.05 km2). A reviewer
may ask how the counts move with the thresholds, the minimum mapping unit and a
morphological cleaning step (Kirby et al., 2024). This script rebuilds the masks of the
five qualified dates from the stored per-scene index arrays and recomputes the
fragmentation metrics over a grid of variants:

    thresholds (NDWI, MNDWI): (-0.10, -0.10), (0.00, 0.00) = frozen, (0.10, 0.10),
                              (0.20, 0.00), (0.00, 0.20)
    minimum mapping unit:     0.02, 0.05 (frozen), 0.10 km2
    morphology:               none (frozen) / binary opening 3 x 3

Everything else is identical to scripts/phase20_water_objects.py: the same SCL rule
(watermask.SCL_REJECT, the SCL-water relaxation of 0.15 on NDWI), the same per-date
mosaic order, the same former-reservoir footprint (P20_reservoir_footprint.geojson),
8-connectivity, components below 9 pixels dropped, coverage measured against the
footprint's own area. The frozen variant must reproduce the published counts exactly;
the script aborts if it does not.

NOT OBSERVED IS NOT DRY: counts are conditional on the observed fraction of the footprint,
which the thresholds do not change (it depends on SCL and reflectance validity only).

Inputs (bulk volume; not in a release snapshot)
    <BULK_ROOT>/data_swot/processed/water_masks/<scene>.npz   ndwi, mndwi (float16), scl, valid, affine, crs
    outputs/tables/water_mask_summary.csv                      scene -> tile, sensing time
    outputs/tables/fragmentation_metrics_by_date.csv           the published counts (for the check)
    outputs/figure_data/P20_reservoir_footprint.geojson
Outputs
    outputs/paper/validation/ms10_s2_fragmentation_sensitivity.csv   one row per date x variant
    outputs/paper/validation/ms10_s2_fragmentation_summary.csv       one row per variant (pre value, post median / range)
    outputs/paper/validation/ms10_s2_fragmentation_sensitivity.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swot_dnipro import config as CFG  # noqa: E402
from swot_dnipro.watermask import SCL_REJECT  # noqa: E402

MASKS = CFG.BULK_ROOT / "data_swot/processed/water_masks"
SUMMARY = ROOT / "outputs/tables/water_mask_summary.csv"
PUBLISHED = ROOT / "outputs/tables/fragmentation_metrics_by_date.csv"
FOOTPRINT = ROOT / "outputs/figure_data/P20_reservoir_footprint.geojson"
OUT = ROOT / "outputs/paper/validation"
PIX_M, MIN_PIX, COVERAGE_HIGH = 20.0, 9, 0.80
THRESHOLDS = [(-0.10, -0.10), (0.00, 0.00), (0.10, 0.10), (0.20, 0.00), (0.00, 0.20)]
MMUS = (0.02, 0.05, 0.10)
MORPH = ("none", "opening3x3")
FROZEN = dict(ndwi_thr=0.0, mndwi_thr=0.0, mmu_km2=0.05, morphology="none")


def water_rule(ndwi, mndwi, scl, t_ndwi, t_mndwi):
    """Exactly watermask.build_mask's decision, on the stored arrays."""
    rej = np.isin(scl, SCL_REJECT)
    w = (ndwi > t_ndwi) & (mndwi > t_mndwi) & ~rej
    w |= (scl == 6) & (ndwi > t_ndwi - 0.15)
    return w & ~rej


def scene_arrays(name: str):
    z = np.load(MASKS / f"{name}.npz")
    return (z["ndwi"].astype("f4"), z["mndwi"].astype("f4"), z["scl"].astype("i2"),
            z["valid"].astype(bool), z["mask"].astype(bool), tuple(float(v) for v in z["affine"]),
            str(z["crs"]))


def mosaic(scenes, t_ndwi, t_mndwi):
    """Merge per-scene uint8 rasters (1 water, 0 observed non-water, 255 not observed)
    with rasterio.merge in the same scene order as phase20_water_objects."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.transform import Affine
    mems, srcs = [], []
    for name in scenes:
        ndwi, mndwi, scl, valid, _, aff, crs = scene_arrays(name)
        w = water_rule(ndwi, mndwi, scl, t_ndwi, t_mndwi)
        a = np.where(valid, w.astype("u1"), np.uint8(255))
        m = MemoryFile()
        with m.open(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="uint8",
                    crs=crs, transform=Affine(*aff), nodata=255) as ds:
            ds.write(a, 1)
        mems.append(m)
        srcs.append(m.open())
    arr, transform = merge(srcs, nodata=255)
    for s in srcs:
        s.close()
    for m in mems:
        m.close()
    return arr[0], transform


def metrics(mos, transform, fp_geom, mmu, morphology):
    from rasterio import features
    from scipy import ndimage
    H, W = mos.shape
    fp = features.rasterize([(fp_geom, 1)], out_shape=(H, W), transform=transform,
                            fill=0, dtype="uint8").astype(bool)
    valid = mos != 255
    frac = (valid & fp).sum() * PIX_M ** 2 / 1e6 / (fp_geom.area / 1e6)
    water = (mos == 1) & fp
    if morphology == "opening3x3":
        water = ndimage.binary_opening(water, structure=np.ones((3, 3)))
    lab, n = ndimage.label(water, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    areas = sizes[sizes >= MIN_PIX] * PIX_M ** 2 / 1e6
    real = areas[areas >= mmu]
    tot = float(real.sum()) if real.size else 0.0
    largest = float(real.max()) if real.size else 0.0
    return dict(footprint_observed_fraction=float(frac), n_water_bodies=int(real.size),
                total_water_area_km2=tot, largest_component_area_km2=largest,
                largest_component_fraction=largest / tot if tot else np.nan)


def main() -> None:
    if not MASKS.exists():
        sys.exit(f"mask store not available at {MASKS}; the tracked tables under {OUT} are the record")
    import geopandas as gpd
    pub = pd.read_csv(PUBLISHED)
    dates = pub[(pub.footprint_observed_fraction > COVERAGE_HIGH)
                & pub.period.isin(["PRE_BREACH", "POST_BREACH"])].date.tolist()
    wm = pd.read_csv(SUMMARY)
    wm["date"] = pd.to_datetime(wm.sensing_time, format="%Y%m%dT%H%M%S").dt.strftime("%Y-%m-%d")
    # the file carries EPSG:32636 coordinates under a 4326 tag (written by phase20 from the mosaic grid)
    fp_geom = gpd.read_file(FOOTPRINT).set_crs(CFG.CRS_METRIC, allow_override=True).geometry.iloc[0]
    if abs(fp_geom.area / 1e6 - pub.loc[pub.date == dates[0], "total_aoi_km2"].iloc[0]) > 0.1:
        sys.exit("footprint area does not match the published total_aoi_km2")
    # the frozen rule must reproduce the stored per-scene masks before anything else is trusted
    name0 = wm[wm.date == dates[0]].name.iloc[0]
    ndwi, mndwi, scl, valid, stored, _, _ = scene_arrays(name0)
    w0 = water_rule(ndwi, mndwi, scl, 0.0, 0.0)
    mism = int((w0 != stored)[valid].sum())
    print(f"frozen rule vs stored mask on {name0}: {mism} mismatching observed pixels of {int(valid.sum()):,}")

    rows, t0 = [], time.time()
    for d in dates:
        scenes = [n for n in wm[wm.date == d].name if (MASKS / f"{n}.npz").exists()]
        per = pub.loc[pub.date == d, "period"].iloc[0]
        for t_ndwi, t_mndwi in THRESHOLDS:
            mos, transform = mosaic(scenes, t_ndwi, t_mndwi)
            for morphology in MORPH:
                for mmu in MMUS:
                    m = metrics(mos, transform, fp_geom, mmu, morphology)
                    rows.append(dict(date=d, period=per, ndwi_thr=t_ndwi, mndwi_thr=t_mndwi, mmu_km2=mmu,
                                     morphology=morphology, n_scenes=len(scenes), **m))
            print(f"  {d} {per:11s} thr ({t_ndwi:+.2f}, {t_mndwi:+.2f}) done  [{time.time() - t0:5.0f} s]", flush=True)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "ms10_s2_fragmentation_sensitivity.csv", index=False)

    # the frozen variant must equal the published table
    fz = t[(t.ndwi_thr == FROZEN["ndwi_thr"]) & (t.mndwi_thr == FROZEN["mndwi_thr"])
           & (t.mmu_km2 == FROZEN["mmu_km2"]) & (t.morphology == FROZEN["morphology"])].set_index("date")
    ref = pub.set_index("date").loc[dates]
    # The stored masks were decided on float32 indices; the store keeps float16, so a few
    # pixels within ~1e-3 of the threshold flip and a component at the 9-pixel or MMU edge
    # can appear or vanish. Tolerance: 2 bodies and 0.1 km2 per date, reported explicitly.
    TOL_N, TOL_KM2 = 2, 0.2   # one body at the MMU edge is 0.05-0.2 km2
    diffs = [(d, int(fz.loc[d, "n_water_bodies"]) - int(ref.loc[d, "n_water_bodies"]),
              round(float(fz.loc[d, "total_water_area_km2"] - ref.loc[d, "total_water_area_km2"]), 3)) for d in dates]
    bad = [x for x in diffs if abs(x[1]) > TOL_N or abs(x[2]) > TOL_KM2]
    if bad:
        sys.exit(f"frozen variant does not reproduce the published counts within tolerance: {bad}")
    print(f"frozen variant vs published (bodies, km2) per date: {diffs}")

    # per-variant summary: pre value, post median and range
    pre, post = t[t.period == "PRE_BREACH"], t[t.period == "POST_BREACH"]
    keys = ["ndwi_thr", "mndwi_thr", "mmu_km2", "morphology"]
    s = post.groupby(keys).agg(post_n_median=("n_water_bodies", "median"), post_n_min=("n_water_bodies", "min"),
                               post_n_max=("n_water_bodies", "max"),
                               post_largest_fraction_median=("largest_component_fraction", "median"),
                               post_largest_fraction_min=("largest_component_fraction", "min"),
                               post_largest_fraction_max=("largest_component_fraction", "max"),
                               post_area_median_km2=("total_water_area_km2", "median")).reset_index()
    p = pre.set_index(keys)[["n_water_bodies", "largest_component_fraction", "total_water_area_km2"]]
    p.columns = ["pre_n", "pre_largest_fraction", "pre_area_km2"]
    s = s.merge(p.reset_index(), on=keys)
    s["frozen"] = (s.ndwi_thr == 0) & (s.mndwi_thr == 0) & (s.mmu_km2 == 0.05) & (s.morphology == "none")
    s.to_csv(OUT / "ms10_s2_fragmentation_summary.csv", index=False)

    f = s[s.frozen].iloc[0]
    md = ["# MS10 -- Sentinel-2 fragmentation: sensitivity to thresholds, minimum mapping unit and morphology", "",
          f"Dates: {dates} (footprint observed > {COVERAGE_HIGH:.0%}). Frozen rule rebuilt from the float16 index store: "
          f"pre {int(f.pre_n)} bodies, post median {f.post_n_median:.0f}; differences to the published table "
          f"(bodies, km²) per date {diffs} come from float16 rounding of pixels within ~1e-3 of the threshold.", "",
          "| NDWI > | MNDWI > | MMU km² | morphology | pre bodies | pre largest % | post bodies median (min–max) | post largest % median (min–max) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in s.sort_values(keys).itertuples():
        md.append(f"| {r.ndwi_thr:+.2f} | {r.mndwi_thr:+.2f} | {r.mmu_km2:.2f} | {r.morphology} | {int(r.pre_n)} | "
                  f"{100 * r.pre_largest_fraction:.3f} | {r.post_n_median:.0f} ({r.post_n_min}–{r.post_n_max}) | "
                  f"{100 * r.post_largest_fraction_median:.1f} ({100 * r.post_largest_fraction_min:.1f}–"
                  f"{100 * r.post_largest_fraction_max:.1f}) |")
    md += ["", f"Across all {len(s)} variants: post-breach median count {s.post_n_median.min():.0f}–{s.post_n_median.max():.0f}, "
               f"post largest-component fraction (median over dates) {100 * s.post_largest_fraction_median.min():.1f}–"
               f"{100 * s.post_largest_fraction_median.max():.1f} %; pre-breach largest component "
               f"{100 * s.pre_largest_fraction.min():.3f}–{100 * s.pre_largest_fraction.max():.3f} %, "
               f"pre count {int(s.pre_n.min())}–{int(s.pre_n.max())}.", ""]
    (OUT / "ms10_s2_fragmentation_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print(md[-2])


if __name__ == "__main__":
    main()
