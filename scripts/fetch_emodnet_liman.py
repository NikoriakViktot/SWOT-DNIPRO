#!/usr/bin/env python
"""Direct EMODnet DTM 2024 pull for the Dnipro-Buh liman, via ERDDAP griddap.

No downloaded zip/script from an external source is used here -- the ERDDAP
endpoint and all six variables (elevation, value_count, cdi_index,
interpolation_flag, elevation_min, elevation_max, stdev) were verified to
exist against the live dataset (.das) before writing this.

interpolation_flag matters most for provenance: it distinguishes cells backed
by real soundings from cells that are purely interpolated -- critical before
trusting EMODnet as any kind of D0 for the lower Dnipro / liman, where no
project-native historical bathymetry exists (unlike Kakhovka's soundings).

AOI matches lower_dnipro_water_occurrence.py's domain (dam-to-sea corridor),
not just the liman proper, for consistency across this session's lower-Dnipro
layers.

Grid: 1/16 arc-minute (~115 m at this latitude), EPSG:4326, elevation relative
to Lowest Astronomical Tide (LAT) -- NOT EVRF2019/EGG2015. Vertical datum
conversion is a separate, later step, not done here.

Outputs
-------
data/processed/emodnet/emodnet_dtm2024_lower_dnipro.nc   (all 6 bands, native NetCDF)
"""
from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/emodnet"
BASE = "https://erddap.emodnet.eu/erddap/griddap/bathymetry_dtm_2024.nc"
# Widened 2026-09-15 per the P11 audit (outputs/reports/DATA_GAPS.md): the
# original 31.50 W edge missed 455 km2 of ZONE_3_DNIPRO_BUG_ESTUARY water,
# whose registry bbox (SD.as_bbox) reaches 31.331871 W. LAT_MAX/LON_MAX match
# the wider window already proven sufficient for the 2026-09-14 audited pull
# (data/bathymetry/dnieper_bug/EMODnet_DTM2024_DnieperBug_extended.nc).
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 31.30, 46.30, 33.45, 47.15
VARS = ["elevation", "value_count", "cdi_index", "interpolation_flag",
       "elevation_min", "elevation_max", "stdev"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "emodnet_dtm2024_lower_dnipro.nc"

    query = ",".join(f"{v}[({LAT_MIN}):({LAT_MAX})][({LON_MIN}):({LON_MAX})]" for v in VARS)
    url = f"{BASE}?{query}"

    print("=" * 78)
    print("EMODnet DTM 2024 -- lower Dnipro / Dnipro-Buh liman, via ERDDAP griddap")
    print("=" * 78)
    print(f"AOI: lon {LON_MIN}-{LON_MAX}, lat {LAT_MIN}-{LAT_MAX}")
    print(f"variables: {VARS}")
    print(f"vertical datum: Lowest Astronomical Tide (LAT) -- NOT EVRF2019/EGG2015 "
          f"(conversion is a separate step)")
    print(f"\nGET {url}\n")

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                total += len(chunk)
        print(f"-> {out_path}  ({total/1e6:.1f} MB)")

    import xarray as xr
    ds = xr.open_dataset(out_path)
    print(f"\nshape: {dict(ds.sizes)}")
    for v in VARS:
        if v in ds:
            print(f"  {v}: {ds[v].dtype}")
    if "interpolation_flag" in ds:
        import numpy as np
        flag = ds["interpolation_flag"].values
        vals, counts = np.unique(flag[np.isfinite(flag)], return_counts=True)
        print(f"\ninterpolation_flag distribution:")
        for v_, c in zip(vals, counts):
            print(f"  {int(v_)}: {c:,} cells ({100*c/flag.size:.1f}%)")
    ds.close()


if __name__ == "__main__":
    main()
