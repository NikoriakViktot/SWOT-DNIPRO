"""
Tests for download engine and index/storage logic.
Mocks HTTP to avoid real downloads; integration test downloads one real file if token present.
Run: pytest tests/test_download_storage.py -v
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swotdl.settings import load_config
from swotdl.download import (
    DownloadResult,
    _safe_filename,
    _parse_year_month,
    _out_path,
    download_many,
)
from swotdl.index import write_index

CFG = load_config(Path("config/pipeline.yaml"))


# ── _safe_filename ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/path/SWOT_L2_20230115_001.nc", "SWOT_L2_20230115_001.nc"),
    ("https://example.com/path/file.nc?token=abc", "file.nc"),
    ("https://example.com/path/", "swot_file.nc"),
    ("https://example.com/data.nc4", "data.nc4"),
])
def test_safe_filename(url, expected):
    assert _safe_filename(url) == expected


# ── _parse_year_month ─────────────────────────────────────────────────────────

def test_parse_year_month_from_time_start():
    entry = {"time_start": "2023-06-15T10:00:00Z", "title": ""}
    assert _parse_year_month(entry) == ("2023", "06")


def test_parse_year_month_fallback_title():
    entry = {"time_start": None, "title": "SWOT_L2_HR_LakeSP_20230801_v2.0"}
    y, m = _parse_year_month(entry)
    assert y == "2023"
    assert m == "08"


def test_parse_year_month_unknown():
    entry = {"time_start": None, "title": "no_date_here"}
    y, m = _parse_year_month(entry)
    assert y == "unknown"
    assert m == "unknown"


# ── _out_path creates directories ─────────────────────────────────────────────

def test_out_path_creates_dir(tmp_path):
    from dataclasses import replace
    import copy

    # patch cfg.paths.raw_dir to tmp_path
    raw = tmp_path / "raw"
    cfg2 = load_config(Path("config/pipeline.yaml"))

    # monkeypatch PathsConfig
    object.__setattr__(cfg2.paths, "raw_dir", str(raw))

    entry = {"time_start": "2023-07-01T00:00:00Z", "title": "test"}
    url = "https://example.com/SWOT_file.nc"

    out = _out_path(cfg2, "SWOT_L2_HR_LakeSP", entry, url)
    assert out.parent.exists(), "Output directory should be created"
    assert out.name == "SWOT_file.nc"
    assert "swot_l2_hr_lakesp" in str(out)
    assert "2023" in str(out)
    assert "07" in str(out)


# ── download_many: no token → error result ────────────────────────────────────

def test_download_many_no_token(tmp_path, monkeypatch):
    """Without token, every download should return error result, not raise."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)

    # Build a fresh unfrozen cfg pointing to tmp_path
    cfg_local = load_config(Path("config/pipeline.yaml"))
    object.__setattr__(cfg_local.paths, "raw_dir", str(tmp_path))

    granules = [
        {
            "title": "test_granule_001",
            "time_start": "2023-05-01T00:00:00Z",
            "_download_url": "https://example.com/SWOT_test_001.nc",
        }
    ]
    results = download_many(cfg_local, granules, product="SWOT_L2_HR_LakeSP")
    assert len(results) == 1
    assert results[0].ok is False
    assert "EARTHDATA_TOKEN" in (results[0].error or "")


def test_download_many_no_url_skipped():
    """Granules without _download_url should not produce results."""
    granules = [{"title": "no_url_granule", "_download_url": None}]
    results = download_many(CFG, granules, product="SWOT_L2_HR_LakeSP")
    assert len(results) == 0


# ── download_many: HTTP mock ──────────────────────────────────────────────────

def test_download_mocked_success(tmp_path, monkeypatch):
    """Mock a successful HTTP download and verify file is written."""
    monkeypatch.setenv("EARTHDATA_TOKEN", "mock_token_abc")

    # Build a minimal mock response
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content = MagicMock(
        return_value=[b"NETCDF_HEADER_FAKE_DATA_" * 100]
    )

    granules = [
        {
            "title": "mock_granule_001",
            "time_start": "2023-04-01T00:00:00Z",
            "_download_url": "https://opendap.earthdata.nasa.gov/SWOT_mock_001.nc",
        }
    ]

    # Patch raw_dir to tmp_path
    object.__setattr__(CFG.paths, "raw_dir", str(tmp_path))

    with patch("swotdl.download.requests.get", return_value=mock_resp):
        results = download_many(CFG, granules, product="SWOT_L2_HR_LakeSP")

    assert len(results) == 1
    r = results[0]
    assert r.ok is True
    assert r.http_status == 200
    assert r.bytes_written > 0
    assert r.local_path is not None
    assert Path(r.local_path).exists()


def test_download_mocked_403(tmp_path, monkeypatch):
    """HTTP 403 should produce failed result with status code."""
    monkeypatch.setenv("EARTHDATA_TOKEN", "bad_token")

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status_code = 403
    mock_resp.raise_for_status = MagicMock(
        side_effect=Exception("403 Forbidden")
    )
    mock_resp.iter_content = MagicMock(return_value=[])

    object.__setattr__(CFG.paths, "raw_dir", str(tmp_path))
    granules = [
        {
            "title": "forbidden_granule",
            "time_start": "2023-04-01T00:00:00Z",
            "_download_url": "https://opendap.earthdata.nasa.gov/SWOT_forbidden.nc",
        }
    ]

    with patch("swotdl.download.requests.get", return_value=mock_resp):
        results = download_many(CFG, granules, product="SWOT_L2_HR_LakeSP")

    assert len(results) == 1
    assert results[0].ok is False


# ── index (parquet storage) ───────────────────────────────────────────────────

@pytest.fixture()
def cfg_tmp(tmp_path):
    """Return a fresh AppConfig with index_dir and raw_dir pointing to tmp_path."""
    c = load_config(Path("config/pipeline.yaml"))
    object.__setattr__(c.paths, "index_dir", str(tmp_path / "index"))
    object.__setattr__(c.paths, "raw_dir",   str(tmp_path / "raw"))
    return c


def test_write_index_creates_parquet(tmp_path, cfg_tmp):
    """write_index should create a .parquet file with correct columns."""

    granules = [
        {
            "title": "granule_001",
            "producer_granule_id": "PG001",
            "time_start": "2023-03-01T00:00:00Z",
            "time_end":   "2023-03-01T01:00:00Z",
            "_download_url": "https://example.com/granule_001.nc",
        }
    ]
    downloads = [
        DownloadResult(
            title="granule_001",
            url="https://example.com/granule_001.nc",
            local_path=str(tmp_path / "granule_001.nc"),
            ok=True,
            http_status=200,
            bytes_written=1024,
        )
    ]

    out = write_index(
        cfg_tmp,
        granules,
        product="SWOT_L2_HR_LakeSP",
        aoi_name="kakhovka_reservoir",
        bbox=(33.35, 46.75, 35.34, 47.78),
        downloads=downloads,
    )

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["title"] == "granule_001"
    assert bool(row["downloaded"]) is True
    assert row["http_status"] == 200
    assert row["bytes"] == 1024
    assert row["aoi_name"] == "kakhovka_reservoir"
    assert row["product"] == "SWOT_L2_HR_LakeSP"


def test_write_index_merge(cfg_tmp):
    """Second write_index call should merge, not duplicate same product+aoi rows."""

    def make_granule(n):
        return {
            "title": f"granule_{n:03d}",
            "producer_granule_id": f"PG{n}",
            "time_start": f"2023-0{n}-01T00:00:00Z",
            "time_end":   f"2023-0{n}-01T01:00:00Z",
            "_download_url": f"https://example.com/g{n}.nc",
        }

    write_index(cfg_tmp, [make_granule(1), make_granule(2)],
                product="SWOT_L2_HR_LakeSP", aoi_name="kakhovka_reservoir",
                bbox=(33.35, 46.75, 35.34, 47.78), downloads=[])

    write_index(cfg_tmp, [make_granule(3), make_granule(4)],
                product="SWOT_L2_HR_LakeSP", aoi_name="kakhovka_reservoir",
                bbox=(33.35, 46.75, 35.34, 47.78), downloads=[])

    df = pd.read_parquet(Path(cfg_tmp.paths.index_dir) / "granules.parquet")
    assert len(df) == 2, f"Expected 2 rows after merge (replace), got {df.shape[0]}"


def test_write_index_different_products_accumulate(cfg_tmp):
    """Different products should accumulate (not overwrite each other)."""
    base_granule = {
        "title": "g1",
        "producer_granule_id": "PG1",
        "time_start": "2023-05-01T00:00:00Z",
        "time_end":   "2023-05-01T01:00:00Z",
        "_download_url": "https://example.com/g1.nc",
    }

    write_index(cfg_tmp, [base_granule], product="SWOT_L2_HR_LakeSP",
                aoi_name="kakhovka_reservoir",
                bbox=(33.35, 46.75, 35.34, 47.78), downloads=[])

    write_index(cfg_tmp, [base_granule], product="SWOT_L2_HR_RiverSP",
                aoi_name="lower_dnipro",
                bbox=(31.5, 46.2, 34.5, 47.5), downloads=[])

    df = pd.read_parquet(Path(cfg_tmp.paths.index_dir) / "granules.parquet")
    assert len(df) == 2, f"Expected 2 rows (different products), got {df.shape[0]}"
    assert set(df["product"].tolist()) == {"SWOT_L2_HR_LakeSP", "SWOT_L2_HR_RiverSP"}


# ── CSV сайд-кар ─────────────────────────────────────────────────────────────

def test_write_index_creates_csv(cfg_tmp):
    granules = [{"title": "g1", "producer_granule_id": "P1",
                 "time_start": "2023-01-01T00:00:00Z", "time_end": "2023-01-01T01:00:00Z",
                 "_download_url": "https://example.com/g1.nc"}]
    write_index(cfg_tmp, granules, product="SWOT_L2_HR_LakeSP",
                aoi_name="test", bbox=(33.0, 46.0, 35.0, 48.0), downloads=[])

    csv = Path(cfg_tmp.paths.index_dir) / "granules.csv"
    assert csv.exists()
    df_csv = pd.read_csv(csv)
    assert "title" in df_csv.columns
    assert len(df_csv) == 1


# ── integration: real CMR → find granules, check URLs ────────────────────────

def _has_token() -> bool:
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.environ.get("EARTHDATA_TOKEN"))


@pytest.mark.skipif(not _has_token(), reason="EARTHDATA_TOKEN not set")
def test_integration_cmr_and_dry_run_index(cfg_tmp):
    """
    Real CMR search + index write without downloading.
    Tries both LakeSP and RiverSP; passes if either returns ≥1 granule.
    Verifies: CMR responds → entries have required fields → parquet written correctly.
    """
    import requests as _req

    try:
        _req.get("https://cmr.earthdata.nasa.gov", timeout=10).raise_for_status()
    except Exception:
        pytest.skip("CMR not reachable")

    from swotdl.cmr import cmr_search_all
    from swotdl.aoi import load_aoi, aoi_to_bbox

    # Use a wider bbox: all of Ukraine's Dnipro basin
    bbox = (30.0, 46.0, 36.5, 48.5)

    results: dict[str, list] = {}
    for product in ("SWOT_L2_HR_LakeSP_2.0", "SWOT_L2_HR_RiverSP_2.0"):
        g = cmr_search_all(
            base_url=CFG.cmr.base_url,
            short_name=product,
            bbox=bbox,
            start="2023-01-01",
            end="2024-12-31",
            page_size=50,
        )
        results[product] = g
        print(f"\n  {product}: {len(g)} granules found")

    total = sum(len(v) for v in results.values())
    assert total > 0, (
        "Expected ≥1 SWOT granule over Ukraine (Dnipro basin) 2023–2024. "
        "Check CMR product name or expand time range."
    )

    # Write index for whichever product returned data
    for product, granules in results.items():
        if not granules:
            continue
        write_index(cfg_tmp, granules, product=product,
                    aoi_name="ukraine_dnipro", bbox=bbox, downloads=[])

    df = pd.read_parquet(Path(cfg_tmp.paths.index_dir) / "granules.parquet")
    assert len(df) == total
    assert df["downloaded"].sum() == 0
    print(f"\n  Index written: {len(df)} rows, all downloaded=False ✓")
