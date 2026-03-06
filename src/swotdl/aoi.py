"""AOI utilities: load GeoJSON and extract CMR-compatible bounding box."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]  # (west, south, east, north)


def load_aoi(path: Path) -> gpd.GeoDataFrame:
    """Read GeoJSON (Polygon / MultiPolygon) into a GeoDataFrame (EPSG:4326)."""
    gdf = gpd.read_file(path)
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError(f"AOI file is empty or has no valid geometry: {path}")
    if gdf.crs is None:
        logger.warning("AOI has no CRS — assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        logger.info("Re-projecting AOI from %s → EPSG:4326", gdf.crs)
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def aoi_to_bbox(gdf: gpd.GeoDataFrame) -> BBox:
    """Return (west, south, east, north) bounding box from GeoDataFrame."""
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = (float(minx), float(miny), float(maxx), float(maxy))
    logger.debug("AOI bbox: W=%.4f S=%.4f E=%.4f N=%.4f", *bbox)
    return bbox


def aoi_geometry(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Union of all geometries in the AOI GeoDataFrame."""
    return gdf.geometry.union_all()


def granule_intersects_aoi(granule_polygon_coords: list | None, aoi_geom: BaseGeometry) -> bool:
    """
    Check if CMR granule footprint intersects the AOI geometry.
    granule_polygon_coords: list of [lon, lat] from CMR entry['polygons'] or None.
    Returns True if no polygon available (bbox-only fallback).
    """
    if not granule_polygon_coords:
        return True  # no footprint → trust bbox filter

    try:
        from shapely.geometry import Polygon as SPolygon

        # CMR polygons come as flat list of "lat lon lat lon ..."
        if isinstance(granule_polygon_coords, list):
            flat = granule_polygon_coords[0] if isinstance(granule_polygon_coords[0], list) else granule_polygon_coords
            if isinstance(flat, str):
                pts = list(map(float, flat.split()))
                coords = [(pts[i + 1], pts[i]) for i in range(0, len(pts) - 1, 2)]
            else:
                coords = flat  # already [[lon, lat], ...]

            poly = SPolygon(coords)
            return bool(poly.intersects(aoi_geom))
    except Exception as exc:
        logger.debug("Could not parse granule polygon for AOI intersection check: %s", exc)
    return True
