#!/usr/bin/env python
"""K10e.V15 -- wall-to-wall annual Dynamic World composites, 2022-2025.

Requires the `geoid` conda env (earthengine-api + geemap, credentials
established this session).

Scope: 2022 (pre-breach baseline) through 2025 (latest complete/partial year),
over the FULL former-pool bounding box, not just where ICESat-2 has tracks --
this is a basin-scale product for the broader succession/roughness analysis
(V12/V15/V16), independent of the point-level attribution in
k10e_dynamic_world_2023.py.

RESOLUTION: 30 m, not DW's native 10 m. The former-pool bounding box is
142.6 x 122.0 km -- at 10 m that is 174 Mpx x 10 bands, well beyond a single
synchronous Earth Engine pixel request. 30 m (19.3 Mpx) is a deliberate
basin-scale choice for year-over-year succession context; the ICESat-2 point
attribution (k10e_dynamic_world_2023.py) already samples DW at native 10 m at
every accepted segment, so no point-level precision is lost by this choice.

Per-year composite, per Google's own guidance for Dynamic World time
compositing: the top-1 LABEL band is reduced by per-pixel MODE across every
DW image available that year (robust to individual cloudy/noisy scenes); each
class PROBABILITY band is reduced by MEAN across the same images.

Outputs
-------
data/processed/dynamic_world_annual/kakhovka_dw_<year>.tif
    bands: label_mode, water_mean, trees_mean, grass_mean, flooded_vegetation_mean,
           crops_mean, shrub_and_scrub_mean, built_mean, bare_mean,
           snow_and_ice_mean, n_images
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyproj
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

TO_LL = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform

OUT = Path("/mnt/f/data_kakhovka_dem_swot/dynamic_world_annual")
# Driven by the p0n manifest rather than hard-coded here, so the years, the
# zones and the ROIs all come from one registry-checked source.
MANIFEST = ROOT / "outputs/tables/p0n_dw_manifest.csv"
TARGETS_GEOJSON = ROOT / "data/processed/domains/p0n_dw_targets_utm.geojson"
SCALE_M = 30
DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
           "shrub_and_scrub", "built", "bare", "snow_and_ice"]


def main() -> None:
    import argparse
    import ee
    import geemap
    # The archive is wanted 2017-2026 eventually, but not equally: the years
    # that matter for the breach are 2023-2025, and Earth Engine's
    # noncommercial quota is throttling downloads to the point where a run may
    # not finish. Ordering by priority means an interrupted run leaves the
    # useful years on disk rather than the 2017 ones.
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="",
                    help="comma-separated, e.g. 2023,2024,2025")
    ap.add_argument("--zones", default="",
                    help="comma-separated substrings of the zone name")
    args = ap.parse_args()
    want_years = {int(y) for y in args.years.split(",") if y.strip()}
    want_zones = [z.strip().upper() for z in args.zones.split(",") if z.strip()]
    ee.Initialize(project="ee-nikoriakviktor")

    print("=" * 78)
    print("K10e.V15 -- wall-to-wall annual Dynamic World composites, 2022-2025")
    print("=" * 78)
    print(f"resolution: {SCALE_M} m (basin-scale succession context; point-level DW at "
          f"native 10 m is handled separately by k10e_dynamic_world_2023.py)\n")

    OUT.mkdir(parents=True, exist_ok=True)
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    import geopandas as gpd
    import pandas as pd
    MAN = pd.read_csv(MANIFEST)
    TGT = gpd.read_file(TARGETS_GEOJSON).set_index("analysis_zone")
    print(f"manifest: {len(MAN)} products over {MAN.analysis_zone.nunique()} zones, "
          f"years {MAN.year.min()}-{MAN.year.max()}")
    # The registry returns EPSG:32636 metres, so it still has to be projected
    # before ee.Geometry(). ee.Geometry() with no explicit CRS
    # assumes WGS84 degrees; handing it raw metre values (first vertex x =
    # 652840) built a "polygon" spanning hundreds of thousands of degrees,
    # which is why the very first .getInfo() call hung indefinitely. Also
    # simplified (37,421 vertices raw) since only the BOUNDS are used below --
    # exact shoreline detail buys nothing for a bounding-box export region.


    def is_valid(path):
        """A file that exists is not necessarily a complete download -- an
        interrupted run left an unreadable partial file, and one run that hit
        an internal geemap/geedim tiling error still wrote a readable-but-
        garbage GeoTIFF (band 1 entirely -inf). Check both before trusting a
        cached file, so a broken file is silently rebuilt rather than skipped."""
        import rasterio
        import numpy as np
        try:
            with rasterio.open(path) as src:
                a = src.read(1)
                return bool(np.isfinite(a).any())
        except Exception:
            return False

    done = failed = skipped = 0
    if want_years:
        MAN = MAN[MAN.year.astype(int).isin(want_years)]
    if want_zones:
        MAN = MAN[[any(w in z.upper() for w in want_zones)
                   for z in MAN.analysis_zone]]
    print(f"selected: {len(MAN)} products"
          f"{' for years ' + str(sorted(want_years)) if want_years else ''}"
          f"{' in ' + ','.join(want_zones) if want_zones else ''}\n")
    for row in MAN.sort_values(["analysis_zone", "year"]).itertuples():
        zone, year = row.analysis_zone, int(row.year)
        fp_m = TGT.loc[zone, "geometry"].simplify(100.0)
        fp_ll = shp_transform(TO_LL, fp_m)
        aoi = ee.Geometry(mapping(fp_ll))
        region = aoi.bounds()
        out_path = OUT / row.product
        if out_path.exists():
            if is_valid(out_path):
                print(f"  {zone} {year}: CACHED")
                skipped += 1
                continue
            print(f"  {zone} {year}: existing file failed validation, rebuilding")
            out_path.unlink()
        d1 = ee.Date(f"{year}-01-01")
        d2 = ee.Date(f"{year + 1}-01-01")
        col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(d1, d2).filterBounds(aoi)
        n = col.size().getInfo()
        print(f"  {zone} {year}: {n} DW images over {row.roi_km2:,.0f} km2", flush=True)
        if n == 0:
            print(f"    SKIP: no DW coverage")
            skipped += 1
            continue

        label_mode = col.select("label").reduce(ee.Reducer.mode()).rename("label_mode")
        prob_mean = col.select(DW_BANDS).reduce(ee.Reducer.mean()) \
            .rename([f"{b}_mean" for b in DW_BANDS])
        n_img = col.select("label").reduce(ee.Reducer.count()).rename("n_images")
        composite = ee.Image.cat([label_mode, prob_mean, n_img]).clip(aoi).toFloat()

        # ee_export_image is a single synchronous request, capped at 48 MB by
        # EE -- this composite is ~1.5 GB uncompressed (19.3 Mpx x 11 bands x
        # float32). download_ee_image tiles the region automatically and
        # stitches the result, no Drive export/polling round-trip needed.
        # RESTRICTED MODE. The project is on the Earth Engine noncommercial
        # tier, which caps concurrency: geedim's parallel tile fetch trips
        # "Too Many Requests: Exceeded Earth Engine concurrency limit" and the
        # whole product dies mid-download. Retry with backoff rather than lose
        # the run -- and on each retry, ask for fewer, larger tiles so there
        # are fewer concurrent requests.
        import time as _t
        for attempt in range(6):
            try:
                geemap.download_ee_image(
                    composite, filename=str(out_path), scale=SCALE_M,
                    region=region, crs="EPSG:4326", dtype="float32",
                    max_tile_size=30,          # EE hard limit is 32 MB
                    max_tile_dim=4000 if attempt == 0 else 2000)
                break
            except Exception as ex:
                if "concurrency" not in str(ex) and "Too Many Requests" not in str(ex):
                    raise
                wait = min(300, 20 * 2 ** attempt)
                print(f"    rate-limited (attempt {attempt+1}/6), "
                      f"waiting {wait}s", flush=True)
                out_path.unlink(missing_ok=True)
                _t.sleep(wait)
        else:
            print(f"    GIVING UP on {zone} {year} after 6 rate-limited "
                  f"attempts", flush=True)
            failed += 1
            continue
        # geedim's tiled downloader has silently produced an all-(-inf) file
        # before (2022, first attempt; 2024, this run) with NO exception raised
        # -- a transient per-tile failure that isn't surfaced as an error. Check
        # immediately rather than leaving a corrupt file for the next run's
        # is_valid() gate to catch hours later.
        if is_valid(out_path):
            print(f"    -> {out_path.name}", flush=True)
            done += 1
        else:
            print(f"    WARNING: {zone} {year} downloaded but failed validation "
                  f"(all non-finite) -- deleting, will retry on next run",
                  flush=True)
            out_path.unlink()
            failed += 1

    print(f"\n  {done} downloaded, {skipped} skipped/cached, {failed} failed")
    print(f"-> {OUT} ({len(list(OUT.glob('*.tif')))} composites)")


if __name__ == "__main__":
    main()
