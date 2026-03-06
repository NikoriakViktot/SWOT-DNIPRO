"""Set up file + console logging. Call setup_logging() once at startup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str, level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "run.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Suppress noisy third-party loggers
    for lib in ("urllib3", "requests", "fiona", "rasterio"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger("swotdl")
