"""Paths, constants and CRS policy for the SWOT-DNIPRO validation work.

CRS policy
----------
``CRS_GEOG``   EPSG:4326 — storage/interchange only (all source products are 4326).
``CRS_METRIC`` EPSG:32636 (WGS 84 / UTM zone 36N) — **every** distance, area,
               buffer and scale bar. The study area (32.3–35.4 °E) sits inside
               zone 36 (30–36 °E), so distortion is < 0.1 %.

Web Mercator (EPSG:3857) is never used for quantitative work.

External inputs
---------------
Some read-only inputs (the EGG2015 quasigeoid, the EPSG:9902 grid, the gauge
yearbooks) belong to the companion ICESat-2 project rather than to this
repository, and are far too large to vendor here. Its location is therefore
resolved, in order:

1. ``$SWOT_DNIPRO_ICESAT_ROOT``, if set;
2. otherwise a sibling checkout next to this repository.

The default assumes the two projects sit side by side::

    projects/
      SWOT-DNIPRO/                 <- this repository
      icesat2-atl13-kakhovka/      <- companion inputs

which is a layout anyone can reproduce, rather than one machine's absolute
path. Note that this is a real environment variable: ``.env`` is loaded only by
the ``swotdl`` CLI, not by the analysis scripts, so putting it there would have
no effect. Nothing in this module reads a credential; the only secret in the
project is ``EARTHDATA_TOKEN``, which stays in the untracked ``.env``.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Roots                                                                        #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
ICESAT_ROOT = Path(
    os.environ.get("SWOT_DNIPRO_ICESAT_ROOT", ROOT.parent / "icesat2-atl13-kakhovka")
).expanduser()

DATA_RAW = ROOT / "data" / "raw"
PIXC_DIR = DATA_RAW / "pixc_nova_kakhovka"

# Bulk storage. Every large download and every large derived cache (event
# mosaics, water masks, spectral stacks) is written here, NOT onto the repo
# disk: this volume has ~1.6 TB free against ~780 GB on the repo disk, it
# already holds the project's external archive, and the multi-zone Sentinel-1
# fetch runs to hundreds of GB. Small products that are part of the
# scientific record -- tables, figures, domain GeoJSON -- stay in the repo.
# Override with $SWOT_DNIPRO_BULK_ROOT; falls back to the repo if the volume
# is not mounted, so nothing silently writes into a missing path.
_BULK_DEFAULT = Path("/mnt/f/data_kakhovka_dem_swot")
BULK_ROOT = Path(
    os.environ.get("SWOT_DNIPRO_BULK_ROOT",
                   _BULK_DEFAULT if _BULK_DEFAULT.is_dir() else ROOT / "data")
).expanduser()
BULK_AVAILABLE = BULK_ROOT != ROOT / "data"
#: Sentinel-1 caches, per analysis zone
S1_CACHE = BULK_ROOT / "s1_zone_cache"

OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIGDATA = OUT / "figure_data"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

# Read-only inputs from the ICESat-2 project
EGG2015_TIF = ICESAT_ROOT / "data" / "1_data" / "egg_2015.tif"
UA2019Z_ASC = ICESAT_ROOT / "data" / "external" / "datum" / "ua_2019z.asc"
RESERVOIR_GEOJSON = ICESAT_ROOT / "data" / "Kakhovka_SA_2.geojson"
KHERSON_GAUGE_PARQUET = (
    ICESAT_ROOT / "data" / "1_data" / "data" / "parquet" / "dm_H" / "80805_yearbook.parquet"
)
KHERSON_ATL13 = ICESAT_ROOT / "data" / "processed" / "kherson_atl13_pass_levels.parquet"
KAKHOVKA_ATL13 = ICESAT_ROOT / "data" / "processed" / "kakhovka_atl13_pass_levels.parquet"
CORRECTOR_BY_STATION = ICESAT_ROOT / "outputs" / "tables" / "egg2015_to_evrf2019_by_station.csv"

# --------------------------------------------------------------------------- #
# CRS                                                                          #
# --------------------------------------------------------------------------- #
CRS_GEOG = "EPSG:4326"
CRS_METRIC = "EPSG:32636"  # WGS 84 / UTM 36N

# --------------------------------------------------------------------------- #
# Science constants                                                            #
# --------------------------------------------------------------------------- #
BREACH_DATE = "2023-06-06"
PREBREACH_START = "2023-04-05"
PREBREACH_END = "2023-06-05"
GAUGE_ZERO_BS77_RESERVOIR_M = 12.000
EPSG9902_ACCURACY_M = 0.068
SEED = 42

#: PIXC ``classification`` flag values (PIXC PDD D-56411 Rev C).
PIXC_CLASS = {
    1: "land",
    2: "land_near_water",
    3: "water_near_land",
    4: "open_water",
    5: "dark_water",
    6: "low_coh_water_near_land",
    7: "open_low_coh_water",
}
#: Accepted water classes for the primary aggregation.
PIXC_WATER_CLASSES_PRIMARY = (4,)

#: Kakhovka reservoir gauges. lon/lat from ``config/gauges.yaml`` of the ICESat-2 project.
RESERVOIR_GAUGES = [
    # id, name_en, lon, lat
    (80977, "Nova Kakhovka", 33.374615, 46.775432),
    (80971, "Velyka Lepetykha", 33.931094, 47.175331),
    (80964, "Nikopol", 34.375355, 47.554451),
    (80963, "Blahovishchenka", 34.821044, 47.463877),
    (80959, "Rozumivka", 35.148890, 47.771208),
    (80961, "Plavni", 35.326056, 47.567110),
]
KHERSON_GAUGE = (80805, "Kherson", 32.612026, 46.623750)
KAKHOVKA_DAM = (33.3667, 46.7783)  # Kakhovka HPP dam axis


def free2mean(lat_deg):
    """ATL03 ``tide_earth_free2mean`` (m).

    ATL03 ATBD Release 007, p. 125::

        tide_earth_free2mean = 0.06029 - 0.180873 sin^2(phi)

    "added to photon heights in the tide-free system, h_ph, to yield a mean tide
    system quantity". ATL13 ``ht_water_surf`` inherits ATL03's **tide-free**
    system, whereas EGG2015/EVRF2019 are zero-tide; for crustal ellipsoidal
    heights zero-tide and mean-tide coincide, so this term harmonises ATL13 to
    the convention EGG2015 expects.
    """
    import numpy as np

    return 0.06029 - 0.180873 * np.sin(np.radians(lat_deg)) ** 2
