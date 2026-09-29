#!/usr/bin/env python
"""Download SWOT RiverSP + LakeSP for the lower_dnipro_kherson AOI.

discover_swot.py already catalogued this AOI (data/catalog/granules_discovered.
parquet), but its RiverSP/LakeSP rows are NOT spatially filtered -- its own
docstring says so ("continent-scale SP products... footprints cover all of
Europe and cannot be filtered spatially"), and its granule_id parsing uses an
offset that's wrong for these two product types (their filenames carry an
extra "_Reach_"/"_Obs_"/"_Prior_" token PIXC/Raster don't have), so the
catalog's "pass" column for RiverSP/LakeSP is not trustworthy for pre-filtering
either. Also, data/raw only has 9 PIXC files on disk, all from the unrelated
pixc_nova_kakhovka pull -- nothing for this AOI has actually been downloaded.

Given that, this re-queries CMR directly via swotdl.cmr.cmr_search_all() (which
injects the real download URL) with NO geometric pre-filter, matching how SWOT
river/lake hydrology data is normally consumed: RiverSP/LakeSP are organised
per continent-wide overpass ("_EU_" in the filename), not per basin, so a
single swath file covers all of Europe's reaches/lakes for that pass+cycle.
Confirmed counts: 3,366 RiverSP + 4,935 LakeSP granules across the configured
date range (2022-12-01..2026-03-05) -- ~1.7 MB each observed on a test
download, so of order 10-15 GB total, not the ~13 GB originally estimated from
the (wrong) 196+270 catalog counts. Downstream analysis filters each swath's
reach/lake vector layer down to the Dnipro corridor by geometry -- that
filtering is NOT done here.

Scope: RiverSP + LakeSP only, the direct water-LEVEL products (surface
elevation per river reach / lake extent), not PIXC (734 granules x ~385 MB =~
280 GB at THIS AOI's catalog count, though that count has the same caveat) or
Raster -- those are much larger and a separate decision, not implied by
"water levels".

Outputs
-------
data/raw/swot_l2_hr_riversp_2.0/<year>/<month>/  (zip auto-extracted, then deleted)
data/raw/swot_l2_hr_lakesp_2.0/<year>/<month>/

The directory is named after the CMR short_name, lower-cased, because that is
what ``swotdl.download._out_path`` builds -- NOT ``data/raw/riversp`` and
``data/raw/lakesp``, which this docstring claimed and which have never existed.
The distinction cost real time: with the documented paths empty, the RiverSP
pull looked like it had never run when in fact it had delivered 63 GB, while
LakeSP -- which genuinely had never run -- looked exactly the same.

RiverSP arrives as shapefiles (.shp/.dbf/.shx/.prj/.xml per granule), not .nc;
only PIXC-style products extract to NetCDF.

WHERE THE DATA ACTUALLY IS NOW (2026-09-16)
-------------------------------------------
This script still writes to the paths above -- ``pipeline.yaml`` sets
``paths.raw_dir: "data/raw"`` (T-09) -- but nothing is kept there any more. The
two directories held 483 GB of continent-wide granules and were **deleted** once
``scripts/p22b_verify_ukraine_clip.py`` proved the Ukraine clip a faithful subset
(completeness 8461/8461; all 4919 clipped files intact; 125 granules re-clipped
from the originals and compared attribute- and WKB-identical; 40 empty ones
confirmed empty). The surviving archive is::

    $BULK_ROOT/swot_ua/swot_l2_hr_{lakesp,riversp}_2.0/     (35 GB, 4919 granules)

So a re-run of this script will start filling the repo disk again at ~80 MB per
LakeSP granule. Re-clip with ``scripts/p22_clip_swot_to_ukraine.py`` and remove
the originals, or redirect ``paths.raw_dir`` first.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swotdl import cmr, download
from swotdl.settings import load_config

AOI_BBOX = (31.9, 46.35, 33.45, 47.05)   # matches discover_swot.py's lower_dnipro_kherson
PRODUCTS = ["SWOT_L2_HR_RiverSP_2.0", "SWOT_L2_HR_LakeSP_2.0"]


def _load_env_token():
    if os.environ.get("EARTHDATA_TOKEN"):
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.startswith("EARTHDATA_TOKEN="):
            os.environ["EARTHDATA_TOKEN"] = line.split("=", 1)[1].strip()


def main() -> None:
    _load_env_token()
    if not os.environ.get("EARTHDATA_TOKEN"):
        raise SystemExit("EARTHDATA_TOKEN not set (checked env and .env)")

    print("=" * 78)
    print("SWOT RiverSP + LakeSP -- lower_dnipro_kherson (Kakhovka dam to Kherson/liman)")
    print("=" * 78)
    print(f"AOI bbox: {AOI_BBOX}")
    print("PIXC and Raster NOT included (280 GB / 15 GB respectively) -- RiverSP/LakeSP "
          "are the direct water-level products and ~13 GB combined.\n")

    cfg = load_config(ROOT / "config" / "pipeline.yaml")

    for product in PRODUCTS:
        print(f"--- {product} ---")
        granules = cmr.cmr_search_all(
            cfg.cmr.base_url, product, AOI_BBOX,
            cfg.defaults.start, cfg.defaults.end, cfg.cmr.page_size)
        with_url = [g for g in granules if g.get("_download_url")]
        print(f"  CMR granules found: {len(granules)} ({len(with_url)} with a download URL)")
        if not with_url:
            continue

        # skip anything already on disk (same filename). _out_path() in
        # swotdl.download writes to data/raw/<product.lower()>/<year>/<month>/,
        # where product.lower() is the WHOLE product string (e.g.
        # "swot_l2_hr_riversp_2.0"), not just the short product name.
        # NOTE: _download_one() auto-extracts .zip -> .nc and deletes the zip,
        # so a resumed run comparing .zip URL basenames against on-disk .nc
        # filenames will under-match and re-download some already-fetched
        # granules. Safe (wasteful, not silently lossy) -- not fixed here.
        product_dir = ROOT / "data/raw" / product.lower()
        existing = {p.name for p in product_dir.rglob("*")} if product_dir.exists() else set()
        todo = [g for g in with_url
                if g["_download_url"].split("?")[0].rstrip("/").split("/")[-1] not in existing]
        print(f"  already on disk: {len(with_url) - len(todo)}, to download: {len(todo)}")
        if not todo:
            continue

        results = download.download_many(cfg, todo, product)
        ok = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]
        total_mb = sum(r.bytes_written or 0 for r in ok) / 1_048_576
        print(f"  downloaded: {len(ok)}/{len(todo)} ok, {total_mb:.0f} MB total")
        if failed:
            print(f"  FAILED ({len(failed)}):")
            for r in failed[:10]:
                print(f"    {r.title}: {r.error}")
        print()

    print("done.")


if __name__ == "__main__":
    main()
