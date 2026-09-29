#!/usr/bin/env python
"""Fetch FABDEM (bare-earth DEM, forest/building removed by ML) for the project AOI.

FABDEM = Copernicus GLO-30 with vegetation and building height biases removed
by a trained model (Hawker et al. 2022). It is LAND topography, not bathymetry
-- it says nothing about depth underwater. Its role here is the dry side of
what the liman/lower-Dnipro bathymetry work (`outputs/planning/13_..._ROADMAP.md`
Phase L2/L4) keeps needing and never had: floodplain/bank elevation to anchor
a domain's 0 m contour and to sanity-check `dnipro_water_domain`'s shoreline
against real bare-earth terrain, independent of the ESA WorldCover land-cover
classification that domain is built from.

AOI: the union of ZONE_1-4 (registry, EPSG:32636, reprojected to 4326 for the
package's own bounds convention) -- (west, south, east, north) =
(31.332, 46.110, 35.498, 47.961). 10 FABDEM tiles intersect this box, all in
one source ZIP; the `fabdem` package byte-range-extracts only those tiles, not
the whole archive.

CRS: raw pull stays EPSG:4326 (storage/interchange only, per repo policy) --
reproject to EPSG:32636 downstream, at the point of use, the same as every
other raw external product in this project (EMODnet, SWORD).

Storage: bulk data goes to drive F, never the repo disk.

Vertical datum: FABDEM elevations are EGM2008-referenced heights above the
geoid (inherited from Copernicus GLO-30) -- NOT EVRF2019, NOT BS-77. Treat
exactly like EMODnet: a prior/reference surface with its own datum, not
directly differenceable against this project's EVRF2019 products until a
reconciliation step is done. Not attempted here.

Outputs
-------
/mnt/f/data_kakhovka_dem_swot/fabdem/fabdem_project_aoi.tif   (EPSG:4326, merged)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
from shapely.ops import unary_union

from swot_dnipro import spatial_domains as SD

OUT_DIR = Path("/mnt/f/data_kakhovka_dem_swot/fabdem")
OUT_TIF = OUT_DIR / "fabdem_project_aoi.tif"
CACHE_DIR = OUT_DIR / "cache"
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")


def main() -> None:
    import fabdem

    print("=" * 78)
    print("FABDEM — bare-earth DEM for the project AOI (union of ZONE_1-4)")
    print("=" * 78)

    geoms = [SD.load_utm(z) for z in ZONES]
    union = unary_union(geoms)
    bounds = tuple(round(v, 5) for v in
                   gpd.GeoSeries([union], crs=32636).to_crs(4326).iloc[0].bounds)
    print(f"  AOI bounds (EPSG:4326, west/south/east/north): {bounds}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  cache: {CACHE_DIR}")
    print(f"  output: {OUT_TIF}\n")

    fabdem.download(bounds, output_path=str(OUT_TIF), cache=CACHE_DIR,
                    show_progress=True)

    import rasterio
    with rasterio.open(OUT_TIF) as src:
        print(f"\n-> {OUT_TIF}")
        print(f"   shape {src.shape}, crs {src.crs}, bounds {src.bounds}")
        print(f"   size on disk: {OUT_TIF.stat().st_size/1e6:.1f} MB")
        print(f"   NOTE: EGM2008-referenced heights, NOT EVRF2019/BS-77 -- "
              f"same caution as EMODnet, no reconciliation done here.")


if __name__ == "__main__":
    main()
