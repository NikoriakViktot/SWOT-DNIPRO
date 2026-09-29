#!/usr/bin/env python
"""Phase 9 — ERA5 hourly wind and pressure over the lower Dnipro / Kherson.

Purpose: test whether wind setup can explain the 2023-04-05 SWOT-ICESat-2 anomaly,
which the daily gauge hydrograph could not (it has the wrong sign, and an 18 cm
excursion is a ~99th-percentile daily event there).

ERA5 is a **reanalysis**, not a station observation. Every product derived from it
is labelled as such.

Domain covers Kherson, the Dnipro-Buh liman and the former reservoir so that both
local wind and the along-liman fetch that drives setup are captured.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cdsapi

from swot_dnipro import config as CFG

OUT = CFG.ROOT / "data" / "raw" / "meteo"
OUT.mkdir(parents=True, exist_ok=True)

# north, west, south, east — Kherson 46.62N 32.61E; liman ~46.5N 31.5E; reservoir to 47.9N
AREA = [48.0, 31.0, 46.0, 35.5]
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "surface_pressure"]


def request(year: str, months: list[str], target: Path) -> None:
    if target.exists():
        print(f"  already have {target.name} ({target.stat().st_size/1e6:.1f} MB)")
        return
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": VARS,
            "year": [year],
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    print(f"  wrote {target.name} ({target.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    # critical window first: March-June 2023 brackets every SWOT x ICESat matchup
    request("2023", ["03", "04", "05", "06"], OUT / "era5_lower_dnipro_2023_03-06.nc")
