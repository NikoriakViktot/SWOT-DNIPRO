"""Download engine: Bearer-token auth, retry/backoff, parallel workers, progress."""

from __future__ import annotations

import logging
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from swotdl.settings import AppConfig

logger = logging.getLogger(__name__)

# HTTP statuses that are transient and worth retrying
_RETRYABLE = {429, 500, 502, 503, 504}


@dataclass
class DownloadResult:
    title: str
    url: str
    local_path: Optional[str] = None
    ok: bool = False
    http_status: Optional[int] = None
    bytes_written: Optional[int] = None
    error: Optional[str] = None


def _safe_filename(url: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    # Only accept names that look like actual files (contain a dot)
    if not name or "." not in name:
        return "swot_file.nc"
    return name


def _parse_year_month(entry: dict) -> tuple[str, str]:
    """Extract YYYY/MM from time_start if available, else from title YYYYMMDD pattern."""
    ts = entry.get("time_start") or ""
    if len(ts) >= 7 and ts[4] == "-":
        return ts[:4], ts[5:7]
    # fallback: find first digit-run of length ≥8 that starts with "20"
    title = entry.get("title", "")
    for run in re.findall(r"[0-9]+", title):
        if len(run) >= 8 and run[:2] == "20":
            month = run[4:6]
            if "01" <= month <= "12":
                return run[:4], month
    return "unknown", "unknown"


def _out_path(cfg: AppConfig, product: str, entry: dict, url: str) -> Path:
    year, month = _parse_year_month(entry)
    dest_dir = Path(cfg.paths.raw_dir) / product.lower() / year / month
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / _safe_filename(url)


def _get_token() -> Optional[str]:
    tok = os.environ.get("EARTHDATA_TOKEN")
    # Strip Windows \r\n or any whitespace from .env file
    return tok.strip() if tok else None


def _download_one(cfg: AppConfig, product: str, entry: dict) -> DownloadResult:
    title = entry.get("title", "unknown")
    url = entry.get("_download_url")

    if not url:
        return DownloadResult(title=title, url="", error="No download URL in CMR metadata")

    token = _get_token()
    if not token:
        return DownloadResult(
            title=title, url=url, error="EARTHDATA_TOKEN env variable is not set"
        )

    out_path = _out_path(cfg, product, entry, url)
    headers = {"Authorization": f"Bearer {token}"}
    timeout = (cfg.download.connect_timeout_sec, cfg.download.timeout_sec)

    last_err: str = "unknown"
    last_status: Optional[int] = None

    for attempt in range(1, cfg.download.retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
                last_status = resp.status_code
                if resp.status_code in _RETRYABLE:
                    raise requests.HTTPError(
                        f"Retryable HTTP {resp.status_code}", response=resp
                    )
                resp.raise_for_status()

                tmp = out_path.with_suffix(out_path.suffix + ".part")
                total = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            total += len(chunk)
                tmp.replace(out_path)

                # Auto-extract .zip archives → keep only .nc/.nc4 files
                if out_path.suffix.lower() == ".zip":
                    try:
                        with zipfile.ZipFile(out_path, "r") as zf:
                            nc_names = [n for n in zf.namelist()
                                        if n.lower().endswith((".nc", ".nc4"))]
                            zf.extractall(out_path.parent,
                                          members=nc_names if nc_names else zf.namelist())
                        out_path.unlink()
                        if nc_names:
                            out_path = out_path.parent / Path(nc_names[0]).name
                    except Exception as unzip_exc:
                        logger.warning("Could not unzip %s: %s", out_path.name, unzip_exc)

            logger.info("OK  %s  (%.1f MB)", out_path.name, total / 1_048_576)
            return DownloadResult(
                title=title,
                url=url,
                local_path=str(out_path),
                ok=True,
                http_status=last_status,
                bytes_written=total,
            )

        except Exception as exc:
            last_err = str(exc)
            if attempt < cfg.download.retries:
                sleep_s = cfg.download.backoff_sec * attempt
                logger.warning(
                    "Attempt %d/%d failed for %s — retry in %.1fs: %s",
                    attempt, cfg.download.retries, title, sleep_s, last_err,
                )
                time.sleep(sleep_s)
            else:
                logger.error("Failed after %d attempts: %s — %s", attempt, title, last_err)

    return DownloadResult(
        title=title, url=url, http_status=last_status, error=last_err
    )


def download_many(
    cfg: AppConfig, granules: list[dict], product: str
) -> list[DownloadResult]:
    """Download all granules with a thread pool; returns list of results."""
    tasks = [g for g in granules if g.get("_download_url")]
    skipped = len(granules) - len(tasks)
    if skipped:
        logger.warning("%d granules skipped (no download URL)", skipped)

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=cfg.download.max_workers) as pool:
        futures = {pool.submit(_download_one, cfg, product, g): g for g in tasks}
        with tqdm(total=len(futures), unit="file", desc=f"Downloading {product}") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                pbar.update(1)
                pbar.set_postfix_str("OK" if res.ok else f"ERR: {(res.error or '')[:40]}")

    ok = sum(1 for r in results if r.ok)
    logger.info("Download complete: %d/%d OK", ok, len(results))
    return results
