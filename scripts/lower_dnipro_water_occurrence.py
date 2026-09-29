#!/usr/bin/env python
"""LOWER_DNIPRO.1 -- annual water-occurrence (flood-zone) maps, Kakhovka dam to
the Black Sea (lower Dnipro river reach, Kherson, and the Dnipro-Buh liman).

Extends the K10e effort to a NEW domain -- geographically disjoint from the
Kakhovka reservoir bbox used everywhere else in this project (Kakhovka:
33.3-35.4E). The first version of this script covered only the liman proper
(31.5-32.65E) and left a ~50 km GAP between it and the Kakhovka bbox's western
edge (33.3E), missing the river reach immediately below the dam and Kherson
itself. Fixed here: the AOI now runs continuously from the dam to the sea, and
its eastern edge (33.40E) deliberately overlaps the Kakhovka bbox's western
edge (33.30-33.35E) so there is no seam.

No local Sentinel-2 archive covers this area (checked: only T36TWS/WT/XS/XT/
UWU/UXU tiles exist locally, all Kakhovka-side); no historical bathymetry (D0)
or former-pool/breach-period framing exists for it either -- per the user's
explicit scoping, this is water-occurrence / ICESat-2 ground QC only, NOT a
D0/D1/channel-validation extension.

AOI is a first-pass bounding box from known regional geography (dam to
Ochakiv), NOT yet an exact hydrological boundary -- documented as such,
refine later if a precise river/liman polygon is needed.

Method: SAME classification logic as src/swot_dnipro/watermask.py, reimplemented
as a server-side Earth Engine reducer over COPERNICUS/S2_SR_HARMONIZED (no local
.SAFE.zip download needed -- that archive doesn't exist for this domain):

    NDWI  = (B3-B8)/(B3+B8) > 0
    MNDWI = (B3-B11)/(B3+B11) > 0
    SCL not in {0,1,3,8,9,10,11} (no-data/saturated/shadow/cloud/cirrus/snow)
    water = NDWI>0 AND MNDWI>0 AND SCL-permitted

For each year, water OCCURRENCE = fraction of clear (SCL-valid) observations
classified water at that pixel -- this is the actual flood-zone product: which
areas are permanently wet, seasonally flooded, or rarely wet, and how that
footprint shifts between the 2022 pre-breach baseline and 2023-2025.

Outputs
-------
data/processed (F:) /lower_dnipro_water_occurrence/lower_dnipro_water_occurrence_<year>.tif
    bands: water_occurrence (0-1), n_valid_obs
"""
from __future__ import annotations

from pathlib import Path

OUT = Path("/mnt/f/data_kakhovka_dem_swot/lower_dnipro_water_occurrence")
YEARS = [2022, 2023, 2024, 2025]
SCALE_M = 30   # matches the Kakhovka annual DW composites; the AOI is now ~2.6x
              # the original liman-only bbox, and 30 m keeps tile counts/runtime
              # reasonable without losing anything meaningful for flood-extent mapping
# first-pass bbox: Kakhovka dam (~33.35E) through Kherson (32.61E) to Ochakiv/
# the Black Sea mouth (31.5E); latitude spans the Kherson braided delta and
# the liman (46.30-47.00N). Eastern edge (33.40E) overlaps the Kakhovka bbox's
# western edge (33.30E) on purpose -- no gap. NOT an exact hydrological
# boundary.
AOI_BOUNDS = (31.50, 46.30, 33.40, 47.00)   # lon_min, lat_min, lon_max, lat_max
SCL_REJECT = [0, 1, 3, 8, 9, 10, 11]


def main() -> None:
    import time

    import ee
    import geemap
    import numpy as np
    import rasterio
    ee.Initialize(project="ee-nikoriakviktor")

    print("=" * 78)
    print("LOWER_DNIPRO.1 -- annual water-occurrence (flood-zone) maps, "
          "Kakhovka dam -> Kherson -> liman -> sea")
    print("=" * 78)
    print(f"AOI (first-pass bbox, NOT an exact hydrological boundary): {AOI_BOUNDS}")
    print("Method: NDWI>0 & MNDWI>0 & SCL-permitted, identical to "
          "src/swot_dnipro/watermask.py, run server-side over COPERNICUS/S2_SR_HARMONIZED.")
    print("No D0/D1/channel validation here -- no historical bathymetry exists for "
          "this domain; this is water-occurrence + (separately) ICESat-2 ground QC only.\n")

    OUT.mkdir(parents=True, exist_ok=True)
    lo, la, hi, ha = AOI_BOUNDS
    aoi = ee.Geometry.Rectangle([lo, la, hi, ha])

    def is_valid(path):
        try:
            with rasterio.open(path) as src:
                a = src.read(1)
                return bool(np.isfinite(a).any())
        except Exception:
            return False

    def add_water(img):
        ndwi = img.normalizedDifference(["B3", "B8"])
        mndwi = img.normalizedDifference(["B3", "B11"])
        scl = img.select("SCL")
        valid = scl.remap(SCL_REJECT, [1] * len(SCL_REJECT), 0).Not()
        water = ndwi.gt(0).And(mndwi.gt(0)).And(valid)
        return img.addBands(water.rename("water").toFloat()) \
                  .addBands(valid.rename("valid").toFloat())

    for year in YEARS:
        out_path = OUT / f"lower_dnipro_water_occurrence_{year}.tif"
        if out_path.exists() and is_valid(out_path):
            print(f"  {year}: CACHED -> {out_path}")
            continue
        # NOTE: the existing file (if any) is NOT deleted here anymore. A
        # 2026-09-15 run hit Earth Engine's Restricted Mode concurrency limit
        # mid-download, crashed past the old delete-then-refetch logic, and
        # left a corrupt (all -inf) file where a previously CACHED-valid 2022
        # composite used to be. Downloads now go to a .part sibling and only
        # replace out_path after passing is_valid(), so a failed attempt can
        # never destroy a good cached file.
        part_path = out_path.with_suffix(".tif.part")
        if part_path.exists():
            part_path.unlink()

        d1 = ee.Date(f"{year}-01-01")
        d2 = ee.Date(f"{year + 1}-01-01")
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(d1, d2).filterBounds(aoi)
              .map(add_water))
        n = col.size().getInfo()
        print(f"  {year}: {n} Sentinel-2 scenes intersecting the AOI", flush=True)
        if n == 0:
            print(f"    SKIP: no Sentinel-2 coverage for {year}")
            continue

        n_valid = col.select("valid").sum().rename("n_valid_obs")
        n_water = col.select("water").sum()
        occurrence = n_water.divide(n_valid.max(1)).rename("water_occurrence")
        composite = ee.Image.cat([occurrence, n_valid]).clip(aoi).toFloat()

        try:
            # max_requests=1: Restricted Mode enforces a low concurrency cap.
            # num_threads is a no-op in this geedim version (superseded by
            # max_requests/max_cpus) -- it silently changed nothing on the
            # first retry, which still hit "Too Many Requests" at the default
            # max_requests=32.
            geemap.download_ee_image(composite, filename=str(part_path), scale=SCALE_M,
                                     region=aoi, crs="EPSG:4326", dtype="float32",
                                     max_tile_size=30, max_requests=1, max_cpus=1)
        except Exception as e:
            print(f"    FAILED: {year} download raised {type(e).__name__}: {e} -- "
                  f"leaving prior cache (if any) untouched, will retry on next run",
                  flush=True)
            if part_path.exists():
                part_path.unlink()
            time.sleep(5)
            continue

        if is_valid(part_path):
            part_path.replace(out_path)
            print(f"    -> {out_path}", flush=True)
        else:
            print(f"    WARNING: {year} failed validation after download -- "
                  f"discarding, will retry on next run", flush=True)
            part_path.unlink()

    print(f"\n-> {OUT} ({len(list(OUT.glob('*.tif')))} annual water-occurrence maps)")


if __name__ == "__main__":
    main()
