"""Granule index: build/update a Parquet table tracking every discovered granule."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from swotdl.aoi import BBox
from swotdl.download import DownloadResult
from swotdl.settings import AppConfig

logger = logging.getLogger(__name__)

INDEX_FILE = "granules.parquet"
INDEX_CSV = "granules.csv"


def _build_rows(
    granules: list[dict],
    product: str,
    aoi_name: str,
    bbox: BBox,
    dl_map: dict[str, DownloadResult],
) -> list[dict]:
    rows = []
    bbox_str = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    for g in granules:
        title = g.get("title") or ""
        res: Optional[DownloadResult] = dl_map.get(title)
        rows.append(
            {
                "product": product,
                "aoi_name": aoi_name,
                "bbox_used": bbox_str,
                "title": title,
                "producer_granule_id": g.get("producer_granule_id"),
                "time_start": g.get("time_start"),
                "time_end": g.get("time_end"),
                "download_url": g.get("_download_url"),
                "downloaded": bool(res and res.ok),
                "local_path": res.local_path if res else None,
                "bytes": res.bytes_written if res else None,
                "http_status": res.http_status if res else None,
                "error": res.error if res else None,
            }
        )
    return rows


def write_index(
    cfg: AppConfig,
    granules: list[dict],
    product: str,
    aoi_name: str,
    bbox: BBox,
    downloads: list[DownloadResult],
) -> Path:
    """
    Write / merge granule index.
    Merges with existing parquet (by title+product) so repeated runs accumulate.
    """
    index_dir = Path(cfg.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    out = index_dir / INDEX_FILE

    dl_map = {r.title: r for r in downloads}
    new_rows = _build_rows(granules, product, aoi_name, bbox, dl_map)
    new_df = pd.DataFrame(new_rows)

    if out.exists():
        existing = pd.read_parquet(out)
        # Remove stale rows for same product+aoi combination before merging
        mask = ~(
            (existing["product"] == product) & (existing["aoi_name"] == aoi_name)
        )
        merged = pd.concat([existing[mask], new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_parquet(out, index=False)
    # Also write human-readable CSV alongside
    merged.to_csv(index_dir / INDEX_CSV, index=False)
    logger.info("Index written: %s (%d total rows)", out, len(merged))
    return out
