"""Basic tests for AOI loading and bbox extraction."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# conftest.py adds src/ but keep explicit fallback for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swotdl.aoi import load_aoi, aoi_to_bbox, aoi_geometry


def _write_geojson(coords: list, tmp_path: Path) -> Path:
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }
    p = tmp_path / "test_aoi.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    return p


def test_bbox_simple(tmp_path):
    """A unit square should produce bbox (0,0,1,1)."""
    coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    path = _write_geojson(coords, tmp_path)
    gdf = load_aoi(path)
    bbox = aoi_to_bbox(gdf)
    assert bbox == (0.0, 0.0, 1.0, 1.0), f"Unexpected bbox: {bbox}"


def test_bbox_kakhovka(tmp_path):
    """Kakhovka reservoir bounding box should be within plausible Ukraine extent."""
    coords = [
        [33.35, 46.75], [35.34, 46.75], [35.34, 47.78], [33.35, 47.78], [33.35, 46.75]
    ]
    path = _write_geojson(coords, tmp_path)
    gdf = load_aoi(path)
    bbox = aoi_to_bbox(gdf)
    west, south, east, north = bbox
    assert 30 < west < 40, f"West lon out of range: {west}"
    assert 40 < south < 55, f"South lat out of range: {south}"
    assert west < east
    assert south < north


def test_empty_geojson_raises(tmp_path):
    """Empty FeatureCollection should raise ValueError."""
    gj = {"type": "FeatureCollection", "features": []}
    p = tmp_path / "empty.geojson"
    p.write_text(json.dumps(gj), encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_aoi(p)


def test_aoi_geometry_union(tmp_path):
    """aoi_geometry returns a non-empty shapely geometry."""
    coords = [[32, 46], [34, 46], [34, 48], [32, 48], [32, 46]]
    path = _write_geojson(coords, tmp_path)
    gdf = load_aoi(path)
    geom = aoi_geometry(gdf)
    assert not geom.is_empty
    assert geom.area > 0


def test_bbox_matches_config_kakhovka():
    """Config AOI file should have correct bbox for Kakhovka."""
    aoi_path = Path("config/aoi/kakhovka_reservoir.geojson")
    if not aoi_path.exists():
        pytest.skip("Config AOI file not present — run from project root")
    gdf = load_aoi(aoi_path)
    bbox = aoi_to_bbox(gdf)
    west, south, east, north = bbox
    # Kakhovka reservoir is in south-eastern Ukraine
    assert 30 < west < 36
    assert 45 < south < 49
    assert west < east
    assert south < north
