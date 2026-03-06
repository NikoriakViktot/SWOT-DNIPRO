"""Configuration loader — reads config/pipeline.yaml into frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class CMRConfig:
    base_url: str
    page_size: int


@dataclass(frozen=True)
class DownloadConfig:
    timeout_sec: int
    connect_timeout_sec: int
    retries: int
    backoff_sec: float
    max_workers: int


@dataclass(frozen=True)
class DefaultsConfig:
    start: str
    end: str
    products: List[str]


@dataclass(frozen=True)
class PathsConfig:
    data_root: str
    raw_dir: str
    index_dir: str
    log_dir: str


@dataclass(frozen=True)
class AppConfig:
    cmr: CMRConfig
    download: DownloadConfig
    defaults: DefaultsConfig
    paths: PathsConfig


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig(
        cmr=CMRConfig(**raw["cmr"]),
        download=DownloadConfig(**raw["download"]),
        defaults=DefaultsConfig(**raw["defaults"]),
        paths=PathsConfig(**raw["paths"]),
    )
