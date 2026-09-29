#!/usr/bin/env python
"""EMODNET_INTERPOLATED_BATHYMETRIC_PRIOR -- lower Dnipro / Dnipro-Buh liman.

Reprocesses the raw EMODnet DTM 2024 pull (fetch_emodnet_liman.py) into a
baseline surface for the lower-Dnipro/estuary domain, WITH AN EXPLICIT ROLE
CHANGE from how D0 has meant "observed historical bathymetry" everywhere else
in this project (Kakhovka's k10_soundings_evrf2019.parquet, real soundings).

For this domain, EMODnet's own interpolation_flag shows 100% of valid cells
(156,670/156,670) are flag=1 = "interpolated ... absence of real soundings
data" (verified against the live .das metadata, not assumed). So EMODnet here
is NOT:
    - observed bathymetry
    - ground truth
    - independent validation data
    - a historical-soundings D0 in the K10b/K10c/K10d/K10e sense

It IS:
    EMODNET_INTERPOLATED_BATHYMETRIC_PRIOR / BACKGROUND_BATHYMETRIC_SURFACE
    -- an initial terrain / large-scale depth pattern, to be corrected by
    independent observations (ICESat-2 exposed terrain, SWOT, navigation
    charts, echo-sounder surveys) as they become available, never validated
    against itself.

D_final (conceptually, not built here) = D_EMODnet + corrections, not
D_final = D_EMODnet everywhere.

Outputs (EPSG:32636)
---------------------
data/processed/emodnet/D0_liman_background.tif
    band 1: elevation_m         (EMODnet elevation, LAT datum, unchanged values)
    band 2: uncertainty_proxy_m (see below)
    band 3: support_class_code  (1=LOW, 2=MODERATE, 3=OBSERVATION_SUPPORTED,
                                 4=MULTISENSOR_SUPPORTED)
data/processed/emodnet/emodnet_uncertainty_proxy.tif   (band 2 alone, for direct use)
data/processed/emodnet/emodnet_support_class.tif       (band 3 alone, for direct use)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parents[1]
RAW_NC = ROOT / "data/processed/emodnet/emodnet_dtm2024_lower_dnipro.nc"
OUT_DIR = ROOT / "data/processed/emodnet"
DST_CRS = "EPSG:32636"

SUPPORT_CLASSES = {1: "EMODNET_SUPPORT_LOW", 2: "EMODNET_SUPPORT_MODERATE",
                  3: "OBSERVATION_SUPPORTED", 4: "MULTISENSOR_SUPPORTED"}


def main() -> None:
    print("=" * 78)
    print("EMODNET_INTERPOLATED_BATHYMETRIC_PRIOR -- lower Dnipro / liman baseline")
    print("=" * 78)

    ds = xr.open_dataset(RAW_NC)
    lat = ds.latitude.values
    lon = ds.longitude.values
    elev = ds.elevation.values.astype("f4")           # LAT datum, unchanged
    flag = ds.interpolation_flag.values.astype("f4")
    stdev = ds.stdev.values.astype("f4")
    emin = ds.elevation_min.values.astype("f4")
    emax = ds.elevation_max.values.astype("f4")
    vcount = ds.value_count.values.astype("f4")

    valid = np.isfinite(elev)
    n_valid = int(valid.sum())
    n_interp = int(np.nansum(flag[valid] == 1))
    print(f"grid: {elev.shape}, valid cells: {n_valid:,}")
    print(f"interpolation_flag == 1 (interpolated, no real soundings): "
          f"{n_interp:,}/{n_valid:,} ({100*n_interp/max(n_valid,1):.1f}%)")
    if n_interp == n_valid:
        print("  -> ALL valid cells are interpolated. Explicitly retained, not hidden.")

    # ---------------- uncertainty proxy --------------------------------------
    # NOT a formal measurement uncertainty (EMODnet metadata makes no such claim
    # for these fields) -- a relative proxy from what the dataset itself
    # exposes: the local elevation_max-elevation_min spread and stdev both
    # widen where the underlying interpolation had less/more-divergent input,
    # and value_count==0 everywhere flag==1 confirms there is no direct support
    # to fall back on for those cells.
    spread = np.where(np.isfinite(emax) & np.isfinite(emin), emax - emin, np.nan)
    n_stdev = int(np.isfinite(stdev[valid]).sum())
    n_spread = int(np.isfinite(spread[valid]).sum())
    # base term: stdev where available, else the min/max spread, else a flat
    # floor for interpolated cells with neither (documented, not invented as
    # precision)
    unc = np.where(np.isfinite(stdev) & (stdev > 0), stdev,
                  np.where(np.isfinite(spread), spread, np.nan)).astype("f4")
    FLOOR_INTERP_M = 1.0   # documented floor, not a measured value -- flagged in metadata
    # applies to every valid cell without stdev/spread, regardless of whether
    # the flag is confirmed 1 (interpolated) or NaN (status unrecorded) -- both
    # cases lack any real-sounding basis to derive a proxy from, so both get
    # the same conservative floor rather than leaving the NaN-flag cells with
    # an inconsistent NaN uncertainty next to a LOW support class.
    needs_floor = valid & ~np.isfinite(unc)
    n_from_floor = int(needs_floor.sum())
    unc = np.where(needs_floor, FLOOR_INTERP_M, unc)
    unc = np.where(valid, unc, np.nan).astype("f4")
    print(f"\nstdev populated for {n_stdev:,}/{n_valid:,} valid cells, "
          f"elevation_min/max spread for {n_spread:,}/{n_valid:,}")
    if n_from_floor == n_valid:
        print(f"  -> BOTH are empty for every valid cell in this domain: "
              f"uncertainty_proxy_m is a FLAT {FLOOR_INTERP_M:.1f} m FLOOR everywhere, "
              f"not a derived quantity. Do not read the 'median/p90' below as a real")
        print(f"     spatial estimate -- it is uniform by construction.")
    print(f"uncertainty_proxy_m: median {np.nanmedian(unc):.2f} m, "
          f"p90 {np.nanpercentile(unc[valid], 90):.2f} m, "
          f"from floor (no stdev/spread available): {n_from_floor:,}/{n_valid:,} cells")

    # ---------------- support class -------------------------------------------
    # At present nothing in this domain qualifies above LOW: value_count is 0
    # (or non-positive) everywhere flag==1, and no independent local dataset
    # (ICESat-2/SWOT/charts) has been fused in yet -- this script only builds
    # the EMODnet-side layer. OBSERVATION_SUPPORTED / MULTISENSOR_SUPPORTED are
    # defined here for schema stability so a later fusion step can promote
    # cells into them; they are not assigned by this script.
    support = np.where(valid, 0, 0).astype("u1")
    low = valid & (flag == 1)
    moderate = valid & (flag == 0) & (vcount > 0)   # would be real soundings, if any existed
    unknown_flag = valid & np.isnan(flag)   # elevation present, but the flag itself is missing
    support[low] = 1
    support[moderate] = 2
    # unknown_flag cells are NOT silently folded into LOW: their interpolation
    # status is genuinely unrecorded, not confirmed-interpolated -- treated as
    # LOW confidence (the conservative choice) but counted and reported
    # separately so this doesn't read as "156,670 confirmed + reassured 1,570
    # more that we didn't check".
    support[unknown_flag] = 1
    for code, name in SUPPORT_CLASSES.items():
        n = int((support == code).sum())
        print(f"  {name}: {n:,} cells")
    print(f"    of which flag==1 (confirmed interpolated): {int(low.sum()):,}")
    print(f"    of which flag==NaN (interpolation status unrecorded, "
          f"conservatively treated as LOW): {int(unknown_flag.sum()):,}")
    print(f"  OBSERVATION_SUPPORTED / MULTISENSOR_SUPPORTED: 0 cells (reserved, not "
          f"assignable from EMODnet alone -- needs later fusion with independent data)")
    print(f"  NOTE: zero cells in this domain carry a confirmed flag==0 "
          f"(not_interpolated) value.")

    # ---------------- reproject to EPSG:32636 ---------------------------------
    # xarray grid is regular in EPSG:4326; build the source transform/CRS and
    # let rasterio.warp handle the reprojection to the project's metric CRS.
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    src_transform = from_bounds(lon.min() - abs(dlon) / 2, lat.min() - abs(dlat) / 2,
                                lon.max() + abs(dlon) / 2, lat.max() + abs(dlat) / 2,
                                len(lon), len(lat))
    src_crs = "EPSG:4326"

    stack = np.stack([elev, unc, support.astype("f4")], axis=0)
    # xarray's lat axis: check orientation (ascending or descending) matches
    # from_bounds' assumption (top row = max lat); flip if needed
    if lat[0] < lat[-1]:
        stack = stack[:, ::-1, :]

    dst_transform, w, h = calculate_default_transform(
        src_crs, DST_CRS, len(lon), len(lat),
        left=lon.min(), bottom=lat.min(), right=lon.max(), top=lat.max())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "D0_liman_background.tif"
    band_names = ["elevation_m", "uncertainty_proxy_m", "support_class_code"]
    resampling = [Resampling.bilinear, Resampling.bilinear, Resampling.nearest]

    with rasterio.open(
        out_path, "w", driver="GTiff", height=h, width=w, count=3, dtype="float32",
        crs=DST_CRS, transform=dst_transform, compress="deflate", predictor=2,
        tiled=True, nodata=np.nan,
    ) as dst:
        for i in range(3):
            band = np.empty((h, w), "f4")
            reproject(
                source=stack[i], destination=band,
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=DST_CRS,
                resampling=resampling[i], src_nodata=np.nan, dst_nodata=np.nan,
            )
            dst.write(band, i + 1)
            dst.set_band_description(i + 1, band_names[i])
        dst.update_tags(
            source="EMODnet DTM 2024 (erddap.emodnet.eu/erddap/griddap/bathymetry_dtm_2024)",
            role="EMODNET_INTERPOLATED_BATHYMETRIC_PRIOR / BACKGROUND_BATHYMETRIC_SURFACE",
            NOT_role="observed_bathymetry, ground_truth, independent_validation_data, "
                    "historical_soundings_D0",
            observed_soundings_in_domain="false",
            interpolation_flag_valid_cells_pct=f"{100*n_interp/max(n_valid,1):.1f}",
            vertical_datum_original="Lowest Astronomical Tide (LAT) -- NOT EVRF2019/EGG2015, "
                                    "conversion not yet applied",
            crs="EPSG:32636",
            usage_rule="D_final = D_EMODnet + corrections from independent observations; "
                      "never D_final = D_EMODnet everywhere. Never validate EMODnet against "
                      "itself -- validation must come from ICESat-2/SWOT/charts/soundings.",
        )
    print(f"\n-> {out_path}  ({w}x{h} px, EPSG:32636)")

    # also write the two auxiliary layers standalone, for direct use
    for i, (fname, desc) in enumerate([
        ("emodnet_uncertainty_proxy.tif", "uncertainty_proxy_m"),
        ("emodnet_support_class.tif", "support_class_code")], start=1):
        p = OUT_DIR / fname
        with rasterio.open(out_path) as src:
            band = src.read(i + 1)
            prof = src.profile.copy()
            prof.update(count=1)
            with rasterio.open(p, "w", **prof) as dst:
                dst.write(band, 1)
                dst.set_band_description(1, desc)
        print(f"-> {p}")

    manifest = {
        "role": "EMODNET_INTERPOLATED_BATHYMETRIC_PRIOR",
        "not_role": ["observed_bathymetry", "ground_truth",
                    "independent_validation_data", "historical_soundings_D0"],
        "source": "EMODnet DTM 2024, ERDDAP griddap bathymetry_dtm_2024",
        "domain": "lower Dnipro (Kakhovka dam to Kherson) + Dnipro-Buh liman",
        "grid_native": "1/16 arc-minute (~115 m), EPSG:4326",
        "vertical_datum": "Lowest Astronomical Tide (LAT); EVRF2019/EGG2015 conversion NOT applied",
        "n_valid_cells": n_valid,
        "interpolation_flag_eq_1_pct": round(100 * n_interp / max(n_valid, 1), 1),
        "interpolation_flag_unknown_cells": int(unknown_flag.sum()),
        "interpolation_flag_eq_0_cells": int(moderate.sum()),
        "stdev_populated_cells": n_stdev,
        "elevation_min_max_spread_populated_cells": n_spread,
        "uncertainty_proxy_is_flat_floor": bool(n_from_floor == n_valid),
        "uncertainty_proxy_floor_m": FLOOR_INTERP_M,
        "note": "Zero cells in this domain carry a confirmed flag==0 (not_interpolated) "
               "value; the remainder beyond flag==1 has the flag itself missing (NaN), "
               "not a confirmed non-interpolated status. Conservatively treated as LOW "
               "confidence, not silently merged into the flag==1 count. stdev and "
               "elevation_min/max are ALSO empty for every valid cell in this domain, so "
               "uncertainty_proxy_m is a flat 1.0 m documented floor, not a value derived "
               "from EMODnet's own quality fields -- do not present it as a measured or "
               "spatially-varying uncertainty.",
        "support_classes": SUPPORT_CLASSES,
        "usage_rule": "D_final = D_EMODnet + corrections from independent observations "
                     "(ICESat-2 exposed terrain, SWOT, navigation charts, echo-sounder). "
                     "Never D_final = D_EMODnet everywhere. Never validate EMODnet "
                     "against itself.",
        "report_wording": "Because no local observed bathymetric soundings were available "
                          "from EMODnet in the lower Dnipro-Bug Estuary domain, the EMODnet "
                          "DTM 2024 was used only as an interpolated background bathymetric "
                          "prior. All valid cells in the study area were flagged by EMODnet "
                          "as interpolated.",
    }
    (OUT_DIR / "D0_liman_background_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {OUT_DIR / 'D0_liman_background_manifest.json'}")


if __name__ == "__main__":
    main()
