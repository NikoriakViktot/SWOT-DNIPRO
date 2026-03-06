"""NASA CMR search — paginated granule discovery."""

from __future__ import annotations

import logging
from typing import Optional

import requests

from swotdl.aoi import BBox, BaseGeometry, granule_intersects_aoi

logger = logging.getLogger(__name__)

# SWOT data is distributed as .zip archives containing .nc files.
# Links also contain .log, .met.json, .rc.xml — those must be skipped.
_SKIP_EXTENSIONS = (".log", ".json", ".xml", ".iso", ".md5", ".sha", ".cmr")
_SKIP_URL_KEYWORDS = ("opendap", "s3credentials", "virtual-directory",
                      "search.earthdata", "github.com", "podaac.jpl.nasa.gov",
                      "swot.jpl.nasa.gov", "aviso.altimetry", "nasa.gov/swot",
                      "jpl.nasa.gov/missions", "CitingPODAAC", "mission_do")
_WANT_EXTENSIONS = (".nc", ".nc4", ".zip")
_DATA_REL = "data#"


def _score_link(link: dict) -> int:
    """Score a link for download priority. Higher = better. -1 = skip."""
    href = (link.get("href") or "")
    href_lower = href.lower()
    # Must be https
    if not href_lower.startswith("https://"):
        return -1
    # Skip known non-data keywords
    if any(kw in href_lower for kw in _SKIP_URL_KEYWORDS):
        return -1
    # Get bare path without query
    path = href_lower.split("?")[0]
    # Skip metadata file extensions
    if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return -1
    rel = (link.get("rel") or "").lower()
    is_data_rel = _DATA_REL in rel
    # Score by preferred extension
    if path.endswith(".nc") or path.endswith(".nc4"):
        return 100 if is_data_rel else 50
    if path.endswith(".zip"):
        return 90 if is_data_rel else 40
    # Generic data# link without known extension — low score
    if is_data_rel:
        return 5
    return -1


def _extract_download_url(entry: dict) -> Optional[str]:
    """Return the best download URL from CMR links (prefer .nc > .zip > other data)."""
    links = entry.get("links") or []
    best_score = -1
    best_url: Optional[str] = None
    for link in links:
        score = _score_link(link)
        if score > best_score:
            best_score = score
            best_url = link["href"]
    return best_url if best_score > 0 else None


def _cmr_page(
    session: requests.Session,
    base_url: str,
    short_name: str,
    bbox: BBox,
    start: str,
    end: str,
    page_size: int,
    page_num: int,
) -> list[dict]:
    params = {
        "short_name": short_name,
        "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
        "bounding_box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "page_size": page_size,
        "page_num": page_num,
    }
    try:
        r = session.get(base_url, params=params, timeout=60)
        r.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("CMR HTTP error page=%d: %s", page_num, exc)
        raise
    data = r.json()
    entries = data.get("feed", {}).get("entry") or []
    logger.debug("CMR page %d: %d granules", page_num, len(entries))
    return entries


def cmr_search_all(
    base_url: str,
    short_name: str,
    bbox: BBox,
    start: str,
    end: str,
    page_size: int,
    aoi_geom: Optional[BaseGeometry] = None,
    granule_type: Optional[str] = None,
) -> list[dict]:
    """
    Paginate through CMR, optionally post-filter by AOI polygon and granule type.
    granule_type: e.g. 'Obs', 'Prior', 'Unassigned' — filters LakeSP/RiverSP sub-types.
    Returns enriched entry list with '_download_url' key injected.
    """
    session = requests.Session()
    all_entries: list[dict] = []
    page = 1

    while True:
        chunk = _cmr_page(session, base_url, short_name, bbox, start, end, page_size, page)
        if not chunk:
            break

        for entry in chunk:
            entry["_download_url"] = _extract_download_url(entry)
            title = entry.get("title") or ""
            # Filter by granule sub-type (Obs / Prior / Unassigned) in title
            if granule_type is not None:
                # e.g. title contains "_Obs_" or "_Prior_" or "_Unassigned_"
                if f"_{granule_type}_" not in title:
                    continue
            # Optional precise AOI filter using granule footprint
            if aoi_geom is not None:
                polys = entry.get("polygons")
                if not granule_intersects_aoi(polys, aoi_geom):
                    logger.debug("Skipping granule (no AOI intersection): %s", title)
                    continue
            all_entries.append(entry)

        page += 1
        # CMR returns fewer than page_size on last page
        if len(chunk) < page_size:
            break

    logger.info("CMR search done: %d granules (product=%s)", len(all_entries), short_name)
    return all_entries
