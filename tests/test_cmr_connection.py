"""
Integration tests for CMR connection and granule discovery.
These tests make real HTTP requests to NASA CMR (no auth needed for search).
Run from project root: pytest tests/test_cmr_connection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swotdl.cmr import cmr_search_all, _cmr_page, _extract_download_url
from swotdl.settings import load_config

CFG = load_config(Path("config/pipeline.yaml"))

# Small test AOI — just a few degrees around Kakhovka dam area
KAKHOVKA_BBOX = (33.35, 46.75, 35.34, 47.78)
# First LakeSP granule over this bbox appears April 2023; use wider window
START = "2023-04-01"
END   = "2023-09-30"


# ── helpers ──────────────────────────────────────────────────────────────────

def _cmr_reachable() -> bool:
    try:
        r = requests.get("https://cmr.earthdata.nasa.gov", timeout=10)
        return r.status_code < 500
    except Exception:
        return False


skip_if_no_internet = pytest.mark.skipif(
    not _cmr_reachable(), reason="CMR not reachable (no internet)"
)


# ── CMR search tests ─────────────────────────────────────────────────────────

@skip_if_no_internet
def test_cmr_search_returns_list():
    """CMR search must return a list (possibly empty, not an exception)."""
    result = cmr_search_all(
        base_url=CFG.cmr.base_url,
        short_name="SWOT_L2_HR_LakeSP_2.0",
        bbox=KAKHOVKA_BBOX,
        start=START,
        end=END,
        page_size=10,
    )
    assert isinstance(result, list), "Expected list from CMR search"


@skip_if_no_internet
def test_cmr_lakesp_finds_granules():
    """LakeSP product should have granules over Kakhovka Q1-2023."""
    granules = cmr_search_all(
        base_url=CFG.cmr.base_url,
        short_name="SWOT_L2_HR_LakeSP_2.0",
        bbox=KAKHOVKA_BBOX,
        start=START,
        end=END,
        page_size=50,
    )
    assert len(granules) > 0, (
        "Expected >0 LakeSP granules over Kakhovka Apr–Sep 2023. "
        "First granule in this bbox observed ~Apr 2023."
    )


@skip_if_no_internet
def test_cmr_granule_has_required_fields():
    """Each granule entry must contain standard CMR fields."""
    granules = cmr_search_all(
        base_url=CFG.cmr.base_url,
        short_name="SWOT_L2_HR_LakeSP_2.0",
        bbox=KAKHOVKA_BBOX,
        start=START,
        end=END,
        page_size=5,
    )
    if not granules:
        pytest.skip("No granules found — skipping field check")

    required = {"title", "time_start", "time_end"}
    for g in granules[:3]:
        missing = required - g.keys()
        assert not missing, f"Missing CMR fields: {missing} in {g.get('title')}"


@skip_if_no_internet
def test_cmr_riversp_search():
    """RiverSP search must succeed (may return 0 if river not in product)."""
    result = cmr_search_all(
        base_url=CFG.cmr.base_url,
        short_name="SWOT_L2_HR_RiverSP_2.0",
        bbox=(31.5, 46.2, 34.5, 47.5),   # lower Dnipro to sea
        start=START,
        end=END,
        page_size=20,
    )
    assert isinstance(result, list)
    print(f"\n  RiverSP granules found (lower Dnipro Q1-2023): {len(result)}")


@skip_if_no_internet
def test_cmr_pagination_consistency():
    """Two calls with different page_size must return same total count."""
    kw = dict(
        base_url=CFG.cmr.base_url,
        short_name="SWOT_L2_HR_LakeSP_2.0",
        bbox=KAKHOVKA_BBOX,
        start="2023-01-01",
        end="2024-12-31",
    )
    small_pages = cmr_search_all(**kw, page_size=10)
    large_pages = cmr_search_all(**kw, page_size=200)
    assert len(small_pages) == len(large_pages), (
        f"Pagination mismatch: page_size=10 → {len(small_pages)}, "
        f"page_size=200 → {len(large_pages)}"
    )


@skip_if_no_internet
def test_download_url_extracted():
    """At least some granules should have a parseable download URL."""
    granules = cmr_search_all(
        base_url=CFG.cmr.base_url,
        short_name="SWOT_L2_HR_LakeSP_2.0",
        bbox=KAKHOVKA_BBOX,
        start=START,
        end=END,
        page_size=20,
    )
    if not granules:
        pytest.skip("No granules")

    with_url = [g for g in granules if g.get("_download_url")]
    pct = len(with_url) / len(granules) * 100
    print(f"\n  {len(with_url)}/{len(granules)} granules have download URL ({pct:.0f}%)")
    assert len(with_url) > 0, "Expected at least one granule with a download URL"
