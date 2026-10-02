"""Authoritative spatial domain registry.

Replaces every hand-written bbox/polygon literal that used to live inline in
individual scripts (``RES_BBOX_4326`` in ``phase20_water_objects.py``,
``RESERVOIR_WKT`` in ``phase19_discover_targeted.py``, the ``box(...)`` in
``discover_swot.py``) with named, versioned, provenance-carrying geometry
objects read from ``config/spatial_domains.yaml``.

Root-cause context (2026-09-11): those three literals independently encoded
three different, wrong extents for "the reservoir" -- one clipped ~6 km short
on the east, two clipped ~10-11.5 km short on the north -- which is what
produced the visible truncation of the exported bathymetric DEM near
Zaporizhzhia/DniproHES. See ``outputs/planning/00_REBUILD_EXECUTIVE_PLAN.md``.

Usage
-----
    from swot_dnipro import spatial_domains as SD
    geom = SD.load("reservoir_full_pool_prebreach")   # shapely, EPSG:4326
    bbox = SD.as_bbox("study_domain_full")            # (lon_min, lat_min, lon_max, lat_max)

No scientific script may declare its own bbox tuple or WKT literal. If an API
needs a flat bbox, derive it with ``as_bbox()``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyproj
import shapely
import yaml
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from swot_dnipro import config as CFG

YAML_PATH = CFG.ROOT / "config" / "spatial_domains.yaml"

_TF_4326_TO_M = pyproj.Transformer.from_crs(
    "EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform
_TF_M_TO_4326 = pyproj.Transformer.from_crs(
    CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform

#: names that are UNRESOLVED by design -- calling load() on these raises,
#: loudly, rather than silently falling back to a proxy geometry.
UNRESOLVED = {"reservoir_npu_domain", "below_dam_floodplain"}


class UnresolvedDomainError(LookupError):
    """Raised when a domain has no independent source geometry yet."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load(YAML_PATH.read_text())


def geom_hash(geom: "shapely.Geometry") -> str:
    """Short, stable content hash of a geometry's WKB -- the provenance tag
    every derived product should carry (``outputs/planning/02_..._impact_analysis.md`` sec 2.4)."""
    return hashlib.sha256(shapely.to_wkb(geom, output_dimension=2)).hexdigest()[:16]


def _read_geojson_union(path: Path) -> "shapely.Geometry":
    gj = json.loads(path.read_text())
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    return unary_union(geoms)


def _load_reservoir_full_pool_prebreach() -> "shapely.Geometry":
    # CFG.RESERVOIR_GEOJSON = ICESAT_ROOT/data/Kakhovka_SA_2.geojson.
    # Verified 2026-09 (part10_maps.py fig. M3): IoU vs. independent
    # data-driven footprint high; VolumeCorr=17455 (~17.5 km3, matches
    # reservoir full volume). Extended 2026-09-11 (P1C, this session) with a
    # multi-date consensus check -- see sa2_multidate_consensus.csv.
    return _read_geojson_union(CFG.RESERVOIR_GEOJSON)


def _load_study_domain_full(cfg: dict) -> "shapely.Geometry":
    base = _load_reservoir_full_pool_prebreach()
    buf_km = float(cfg["study_domain_full"]["buffer_km"])
    base_m = shp_transform(_TF_4326_TO_M, base)
    buffered_m = base_m.buffer(buf_km * 1000.0)
    return shp_transform(_TF_M_TO_4326, buffered_m)


def _load_dnipro_water_domain() -> "shapely.Geometry":
    # Derived from ESA WorldCover v200 (2021, 10 m) class 80 by
    # scripts/p0b_build_dnipro_water_domain.py, NOT hand-drawn. Covers the
    # connected Dnipro water system from the reservoir head at Khortytsia to
    # the Dnipro-Bug liman, with islands retained as interior rings.
    #
    # It exists because both earlier geometries are wrong in ways an area
    # check cannot see. P20_reservoir_footprint.geojson truncates 87.4 km2 of
    # real water at the eastern end (Sentinel-1 there: VV median -21.6 dB,
    # 99.0% below -15 dB) while adding ~123 km2 of land, so its total almost
    # matches. reservoir_full_pool_prebreach reaches further east but draws a
    # solid blob across Khortytsia, claiming 44.9 km2 of water in the northern
    # lobe where S1 measures 24.5 km2; the 25.8 km2 excess is the island, and
    # ESA WorldCover independently puts it at 24.1 km2.
    #
    # Stored in EPSG:32636 because all metric work in this repo is UTM 36N;
    # reprojected here to satisfy this module's EPSG:4326 contract.
    path = CFG.ROOT / "data/processed/domains/dnipro_water_domain_utm.geojson"
    if not path.exists():
        raise UnresolvedDomainError(
            f"dnipro_water_domain has not been built yet ({path} is missing). "
            f"Run scripts/p0b_build_dnipro_water_domain.py first."
        )
    geom_utm = _read_geojson_union(path)
    return shp_transform(_TF_M_TO_4326, geom_utm)


_ZONES_PATH = CFG.ROOT / "data/processed/domains/analysis_zones_utm.geojson"
_SUBZONES_PATH = CFG.ROOT / "data/processed/domains/analysis_subzones_utm.geojson"
_UA_CLIP_PATH = CFG.ROOT / "data/processed/domains/ukraine_swot_clip_domain.geojson"
_PAPER1_ZONES_PATH = CFG.ROOT / "outputs/paper/zones/paper1_zones_utm.geojson"


def _load_zone_layer(path: Path, key: str, name: str) -> "shapely.Geometry":
    """One analysis zone, reprojected to EPSG:4326 for this module's contract.

    Analysis zones are DOWNLOAD/ANALYSIS containers, never water masks. The
    water mask is derived separately, per observation event and per zone,
    because a SAR classifier calibrated on the calm reservoir does not
    transfer to the reed-choked delta or the wind-roughened estuary.
    """
    if not path.exists():
        raise UnresolvedDomainError(
            f"{name} has not been built yet ({path} is missing). "
            f"Run scripts/p0c_build_analysis_zones.py first."
        )
    gj = json.loads(path.read_text())
    geoms = [shape(f["geometry"]) for f in gj["features"]
             if f["properties"].get(key) == name]
    if not geoms:
        raise KeyError(f"{name!r} not present in {path}")
    return shp_transform(_TF_M_TO_4326, unary_union(geoms))


def _load_dnipro_system_master() -> "shapely.Geometry":
    # Catalogue/visualisation container only: the union of the three zones.
    # It is deliberately NOT a physical model -- nothing should calibrate or
    # validate against it, because it spans three different hydraulic systems.
    gj = json.loads(_ZONES_PATH.read_text())
    return shp_transform(_TF_M_TO_4326,
                         unary_union([shape(f["geometry"]) for f in gj["features"]]))


def load_subzone(name: str) -> "shapely.Geometry":
    """A ZONE_1 subzone (EPSG:4326). H1/H2/H3 reservoir-stage products are
    valid for KAKHOVKA_RESERVOIR_CORE only."""
    return _load_zone_layer(_SUBZONES_PATH, "analysis_subzone", name)


def load_subzone_utm(name: str) -> "shapely.Geometry":
    """A ZONE_1 subzone in EPSG:32636, for metric work."""
    return shp_transform(_TF_4326_TO_M, load_subzone(name))


def load_utm(name: str) -> "shapely.Geometry":
    """The named domain in EPSG:32636 (UTM 36N).

    Every distance, area and buffer computation in this repository is metric,
    so this is the accessor quantitative code should use; ``load()`` returns
    EPSG:4326 for storage and interchange only.
    """
    return shp_transform(_TF_4326_TO_M, load(name))


def _load_reservoir_npu_domain() -> "shapely.Geometry":
    raise UnresolvedDomainError(
        "reservoir_npu_domain is UNRESOLVED: no independent NPU polygon has "
        "been identified in either repository (checked data/historical/*, "
        "config/aoi/*). Historical exposure-fraction work must either fall "
        "back explicitly to reservoir_full_pool_prebreach with a stated "
        "caveat, or wait for the domain owner to supply a source geometry. "
        "See outputs/planning/01_domain_and_mask_redesign.md."
    )


def _load_below_dam_floodplain() -> "shapely.Geometry":
    # RESOLVED 2026-09-18 by scripts/p42_below_dam_floodplain.py (plan 15 WP2):
    # the observed June-2023 flood (cells wet in >= 2 of 11 Sentinel-1 events)
    # UNION HAND < 5 m from FABDEM relative to the pre-breach water surface,
    # within 10 km of it; ZONE_4 U ZONE_2 below the Kakhovka dam, pool excluded.
    # Provenance and the h0 sensitivity table: config/spatial_domains.yaml,
    # outputs/tables/p42_domain_provenance.csv, p42_hand_sensitivity.csv.
    # Stored in EPSG:32636; reprojected here to this module's EPSG:4326 contract.
    path = CFG.ROOT / "data/processed/domains/below_dam_floodplain_utm.geojson"
    if not path.exists():
        raise UnresolvedDomainError(
            f"below_dam_floodplain has not been built yet ({path} is missing). "
            f"Run scripts/p42_below_dam_floodplain.py first."
        )
    return shp_transform(_TF_M_TO_4326, _read_geojson_union(path))


def _load_ukraine_swot_clip_domain() -> "shapely.Geometry":
    """Archive-retention container for the continent-wide SWOT products.

    NOT a scientific domain: a national extent has no hydraulic meaning, and
    nothing should calibrate, validate or clip a product against it. Its only
    use is deciding which parts of a Europe-wide RiverSP/LakeSP granule are
    worth keeping on disk. See scripts/p22_build_ukraine_clip_domain.py for
    why a country polygon is the wrong instrument here.
    """
    gj = json.loads(_UA_CLIP_PATH.read_text())
    return unary_union([shape(f["geometry"]) for f in gj["features"]])


_STATIC_LOADERS = {
    "reservoir_full_pool_prebreach": lambda cfg: _load_reservoir_full_pool_prebreach(),
    "study_domain_full": _load_study_domain_full,
    "reservoir_npu_domain": lambda cfg: _load_reservoir_npu_domain(),
    "below_dam_floodplain": lambda cfg: _load_below_dam_floodplain(),
    "dnipro_water_domain": lambda cfg: _load_dnipro_water_domain(),
    "ZONE_1_KAKHOVKA_LOWER_DNIPRO":
        lambda cfg: _load_zone_layer(_ZONES_PATH, "analysis_zone",
                                     "ZONE_1_KAKHOVKA_LOWER_DNIPRO"),
    "ZONE_2_KHERSON_DELTA":
        lambda cfg: _load_zone_layer(_ZONES_PATH, "analysis_zone",
                                     "ZONE_2_KHERSON_DELTA"),
    # The dam-to-Kherson floodway, promoted from a ZONE_1 subzone once the
    # June 2023 envelope showed the flood occupied 3.0x the channel it was
    # filed under. The subzone KAKHOVKA_DAM_TO_KHERSON remains and still means
    # the CHANNEL; this zone means the floodway.
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY":
        lambda cfg: _load_zone_layer(_ZONES_PATH, "analysis_zone",
                                     "ZONE_4_DAM_TO_KHERSON_FLOODWAY"),
    "ZONE_3_DNIPRO_BUG_ESTUARY":
        lambda cfg: _load_zone_layer(_ZONES_PATH, "analysis_zone",
                                     "ZONE_3_DNIPRO_BUG_ESTUARY"),
    "DNIPRO_SYSTEM_MASTER": lambda cfg: _load_dnipro_system_master(),
    "ukraine_swot_clip_domain": lambda cfg: _load_ukraine_swot_clip_domain(),
    # Paper 1 zones R/F/D/E: disjoint, in flow order, frozen in a tracked file
    # by scripts/ms5_paper1_zones.py. The June 2023 flood envelope is an event
    # layer inside F, never the zone itself.
    **{name: (lambda cfg, n=name: _load_zone_layer(_PAPER1_ZONES_PATH, "domain", n))
       for name in ("R_FORMER_KAKHOVKA_RESERVOIR", "F_LOWER_DNIPRO_FLOODWAY",
                    "D_KHERSON_DELTA", "E_DNIPRO_BUG_ESTUARY",
                    "event_flood_2023_s1_envelope", "event_flood_2023_s1_envelope_1km")},
}


def load(name: str) -> "shapely.Geometry":
    """Return the named domain as a shapely geometry in EPSG:4326.

    Raises ``UnresolvedDomainError`` for domains that are deliberately not
    yet backed by an independent source geometry (see ``UNRESOLVED``).
    Raises ``KeyError`` for anything not in the registry at all -- this is
    intentional: a typo must fail loudly, never fall through to a silent
    default extent.
    """
    cfg = _cfg()
    if name not in cfg:
        raise KeyError(
            f"{name!r} is not a registered spatial domain. Known static "
            f"domains: {sorted(_STATIC_LOADERS)}. Per-date domains "
            f"(water_mask_date, dry_bed_mask_date, ...) are not loaded "
            f"through this function -- see outputs/planning/"
            f"01_domain_and_mask_redesign.md for their producers."
        )
    if name in _STATIC_LOADERS:
        return _STATIC_LOADERS[name](cfg)
    raise KeyError(
        f"{name!r} is registered in spatial_domains.yaml but has no static "
        f"loader -- it is a per-date/per-observation object, not a single "
        f"geometry. See the mask registry."
    )


def as_bbox(name: str) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max), derived programmatically --
    never hand-typed. This is the only sanctioned way to get a flat bbox for
    a discovery/download API query."""
    return tuple(round(v, 6) for v in load(name).bounds)  # type: ignore[return-value]


CANONICAL_CRS = "EPSG:32636"


def assert_projected_32636(obj, label: str = "geometry") -> None:
    """Fail fast before any distance/buffer/area/kriging/chainage/raster
    operation. Reprojection must happen explicitly at ingestion, not be
    silently patched up inside a scientific function (operator, 2026-09-11).

    Accepts a GeoDataFrame/GeoSeries (checks .crs) or a pyproj/rasterio CRS
    object directly.
    """
    crs = getattr(obj, "crs", obj)
    if crs is None:
        raise ValueError(f"{label}: CRS is missing -- must be {CANONICAL_CRS}")
    crs_str = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    if epsg == 4326 or "4326" in crs_str:
        raise ValueError(
            f"{label}: CRS is geographic (EPSG:4326, degrees) -- reproject to "
            f"{CANONICAL_CRS} BEFORE any distance/buffer/area/kriging/"
            f"chainage/rasterization operation, do not mix degrees and "
            f"metres.")
    if epsg != 32636:
        raise ValueError(
            f"{label}: CRS is {crs_str!r}, expected {CANONICAL_CRS}. "
            f"Reproject explicitly at ingestion.")


def registry_row(name: str) -> dict[str, Any]:
    """One row of outputs/tables/spatial_domain_registry_v2.csv for a static
    domain: geometry stats + provenance hash, computed fresh, never cached
    stale."""
    try:
        geom = load(name)
    except UnresolvedDomainError as e:
        return {"name": name, "status": "UNRESOLVED", "area_km2": None,
                "lon_min": None, "lat_min": None, "lon_max": None,
                "lat_max": None, "geometry_hash": None, "note": str(e)}
    geom_m = shp_transform(_TF_4326_TO_M, geom)
    lon_min, lat_min, lon_max, lat_max = geom.bounds
    return {"name": name, "status": "RESOLVED", "area_km2": geom_m.area / 1e6,
            "lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max,
            "lat_max": lat_max, "geometry_hash": geom_hash(geom), "note": ""}


# ---------------------------------------------------------------- grids ----
class DomainTruncationError(RuntimeError):
    """A grid or raster does not cover the authoritative domain.

    This exists because the same defect has now recurred three times. Every
    precomputed bed surface in this project was built on the superseded P20
    footprint, whose eastern edge is E 668,540 m, while the authoritative
    reservoir domain reaches E 678,000 m:

        kakhovka_bed_surface_250m.npz   gx max 668,670
        kakhovka_bed_surface_50m.npz    gx max 668,570
        kakhovka_bed_surface_30m.npz    gx max 668,540

    Any script that took its output grid from one of those files inherited a
    9.3-9.5 km cut through the eastern reservoir -- silently, because nothing
    ever checked. In hist24 that cut was not cosmetic: it corrupted the
    independent hypsometry, and removing it moved the Table 19 RMS area error
    from 154 -> 114 km2 to 95 -> 69 km2.

    A legacy surface may still be READ as a covariate. It may never define an
    output grid. build_grid() is the only sanctioned way to make one.
    """


def assert_covers(geom, gx, gy, *, what: str = "grid", cell: float | None = None
                  ) -> None:
    """Raise unless the grid spans the geometry's bounds.

    `cell` lets the last grid line sit one cell short of the bound, which is
    correct for cell-centre coordinates; without it the check is exact.
    """
    import numpy as _np
    gx = _np.asarray(gx, float)
    gy = _np.asarray(gy, float)
    x0, y0, x1, y1 = geom.bounds
    pad = float(cell) if cell else 0.0
    short = []
    if gx.min() > x0 + 1e-6:
        short.append(f"west by {gx.min() - x0:,.0f} m")
    if gx.max() + pad < x1 - 1e-6:
        short.append(f"EAST by {x1 - gx.max():,.0f} m")
    if gy.min() > y0 + 1e-6:
        short.append(f"south by {gy.min() - y0:,.0f} m")
    if gy.max() + pad < y1 - 1e-6:
        short.append(f"north by {y1 - gy.max():,.0f} m")
    if short:
        raise DomainTruncationError(
            f"{what} is truncated: {', '.join(short)}. The domain spans "
            f"E {x0:,.0f}..{x1:,.0f}, N {y0:,.0f}..{y1:,.0f}; the grid spans "
            f"E {gx.min():,.0f}..{gx.max():,.0f}, N {gy.min():,.0f}.."
            f"{gy.max():,.0f}. Build the grid from the registry with "
            f"SD.build_grid(), never from a precomputed surface file.")


def build_grid(geom, cell: float, *, what: str = "grid") -> dict:
    """The only sanctioned way to build an analysis grid.

    Takes its extent from the authoritative geometry and verifies the result,
    so a truncated grid cannot leave this function. Returns cell-centre
    coordinate vectors `gx`/`gy`, the flat indices of cells inside the
    geometry, and the grid shape.
    """
    import numpy as _np
    import shapely as _shp
    # NOT assert_projected_32636(): that helper expects a GeoDataFrame or a CRS
    # object and falls back to substring-matching "4326" in str(obj). On a bare
    # shapely geometry that is the WKT, and a coordinate such as 527746.4326
    # contains "4326" -- so it reports a metric geometry as geographic. A bare
    # geometry carries no CRS at all, so the only honest check is the magnitude
    # of its bounds.
    x0, y0, x1, y1 = geom.bounds
    if abs(x0) <= 180.0 and abs(y0) <= 90.0:
        raise ValueError(
            f"{what}: bounds {geom.bounds} look like degrees; reproject to "
            f"{CANONICAL_CRS} before building a metric grid")
    gx = _np.arange(_np.floor(x0 / cell) * cell,
                    _np.ceil(x1 / cell) * cell + cell, cell)
    gy = _np.arange(_np.floor(y0 / cell) * cell,
                    _np.ceil(y1 / cell) * cell + cell, cell)
    GX, GY = _np.meshgrid(gx, gy)
    ins = _np.where(_shp.contains_xy(geom, GX.ravel(), GY.ravel()))[0]
    assert_covers(geom, gx, gy, what=what, cell=cell)
    return {"gx": gx, "gy": gy, "nx": len(gx), "ny": len(gy),
            "ins_idx": ins, "cell_m": float(cell),
            "x": GX.ravel()[ins], "y": GY.ravel()[ins]}
