#!/usr/bin/env python
"""P0N — Dynamic World per analysis subzone: everything up to the export boundary.

The existing Dynamic World rasters are one bbox around the reservoir. Measured
against declared scope they are 98.2% of the reservoir core, 4.0% of the
dam-to-Kherson reach, and EXACTLY ZERO of ZONE_2 and ZONE_3. A cross-zone
analysis using them compares a measurement against an absence.

This rebuilds them as per-subzone products on registry geometry. It does
everything that needs no credentials and STOPS before the first ee.Initialize()
or Export, so that authorising Earth Engine is the only remaining step.

THE PRODUCTION UNIT IS THE ANALYSIS SUBZONE, NOT THE ZONE. A zone is the water
corridor plus a download buffer: ZONE_1 is 11,412 km2 of which 8,824 km2 is
buffer. Requiring wall-to-wall land cover over a download buffer would demand
data nobody needs and would fail a coverage gate for the wrong reason.

  ZONE_1  has four registry subzones, and their union is 2,588 km2 -- which is
          exactly the water domain clipped to ZONE_1, an independent check that
          the subzone partition is not drifting from the water geometry.
  ZONE_2  no subzones exist yet, so the target is the water domain clipped to
          the zone: 310 km2.
  ZONE_3  likewise: 2,342 km2.

THE ROI IS THE WHOLE ZONE, AND THE SUBZONES ARE METADATA. An earlier version
clipped the ROI to the water geometry plus a shoreline margin. That is wrong for
this covariate: land cover is also the input to a Manning roughness field, which
needs the floodplain the flood actually spreads over, not a belt around the
present waterline. Downloading the whole zone once and cropping later is cheaper
than discovering the margin was too tight and re-exporting sixty products.

The subzones are still carried, per product, so a later crop is a lookup rather
than a re-derivation, and so coverage can be reported against the analysis
target even though the download is wider.

Outputs (no credentials required)
---------------------------------
outputs/tables/p0n_dw_manifest.csv
data/processed/domains/p0n_dw_targets_utm.geojson
"""
from __future__ import annotations

import json
import subprocess
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

# ZONE_4 was created after this manifest was first written, and its geometry
# has since been rebuilt on the 10 km buffer. A covariate the floodway cannot
# use is a covariate the floodway does not have, so it is listed here.
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")
YEARS = tuple(range(2017, 2027))
# ee.Initialize() with no project raises "not signed up for Earth Engine"
# even on a working credential; the project is what registers the caller.
EE_PROJECT = os.environ.get("GEE_PROJECT", "ee-nikoriakviktor")
NATIVE_M = 10.0          # Dynamic World native resolution
ANALYSIS_M = 20.0        # the project analysis grid
MARGIN_KM = 0.5          # shoreline belt only: Gate 7C measured shoreline
                         # migration in tens of metres, and a 2 km margin blew
                         # FORMER_RESERVOIR_TRANSITION from 196 to 2,728 km2 --
                         # the buffer swallowing the target, which is exactly
                         # what p0c warned a download margin does
COVERAGE_GATE = 0.98     # a wall-to-wall annual product must be essentially whole
SUBZ = ROOT / "data/processed/domains/analysis_subzones_utm.geojson"
OUT_GEOJSON = ROOT / "data/processed/domains/p0n_dw_targets_utm.geojson"


def targets():
    """One ROI per ZONE, from the registry, in EPSG:32636.

    The subzones that fall inside are recorded so a later crop needs no
    re-derivation, and so coverage can be reported against the analysis target
    while the download stays wider."""
    sz = gpd.read_file(SUBZ)
    water = SD.load_utm("dnipro_water_domain")
    out = []
    for zn in ZONES:
        zone = SD.load_utm(zn)
        sub = sz[sz.analysis_zone == zn]
        names = list(sub.analysis_subzone) if len(sub) else []
        analysis = (unary_union(list(sub.geometry)) if len(sub)
                    else zone.intersection(water))
        out.append(dict(analysis_zone=zn,
                        analysis_subzones="|".join(names) if names
                        else f"{zn}_WATER",
                        n_subzones=len(names),
                        roi_km2=zone.area / 1e6,
                        analysis_target_km2=analysis.area / 1e6,
                        geometry=zone))
    return gpd.GeoDataFrame(out, crs=CFG.CRS_METRIC)


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P0N — Dynamic World per subzone, prepared to the export boundary")
    print("=" * 78)
    print(f"  git {commit}")
    print(f"  native {NATIVE_M:.0f} m -> analysis {ANALYSIS_M:.0f} m, "
          f"whole-zone ROI (Manning needs the floodplain), coverage gate "
          f"{100*COVERAGE_GATE:.0f}% of the analysis target")

    # ---- credentials, three states distinguished ------------------------
    state, detail = ee_state()
    print(f"\n  EARTH ENGINE: {state}")
    if detail:
        print(f"    {detail}")

    T = targets()
    SD.assert_projected_32636(T, "DW targets")
    print(f"\n  {len(T)} zone ROIs from the registry (download whole, crop later):")
    for r in T.itertuples():
        print(f"    {r.analysis_zone:30s} ROI {r.roi_km2:9,.0f} km2   "
              f"analysis target {r.analysis_target_km2:8,.0f} km2   "
              f"{r.n_subzones} subzone(s)")
    print(f"    {'TOTAL':30s} ROI {T.roi_km2.sum():9,.0f} km2   "
          f"analysis target {T.analysis_target_km2.sum():8,.0f} km2")

    # ---- geometry checks that need no credentials -----------------------
    print("\n  GEOMETRY CHECKS")
    ok = True
    for r in T.itertuples():
        zone = SD.load_utm(r.analysis_zone)
        inside = r.geometry.difference(zone).area / max(r.geometry.area, 1)
        if inside > 1e-6:
            print(f"    FAIL {r.analysis_subzone}: ROI leaves its zone")
            ok = False
    z1 = T[T.analysis_zone == ZONES[0]]
    water_z1 = SD.load_utm("dnipro_water_domain").intersection(SD.load_utm(ZONES[0]))
    d = abs(float(z1.analysis_target_km2.iloc[0]) - water_z1.area / 1e6)
    print(f"    ZONE_1 subzone union {float(z1.analysis_target_km2.iloc[0]):,.0f} km2 vs water "
          f"domain in zone {water_z1.area/1e6:,.0f} km2  "
          f"(delta {d:,.1f} km2)  {'OK' if d < 5 else 'DRIFT'}")
    if d >= 5:
        ok = False
    print(f"    all ROIs inside their zones: {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("geometry checks failed; not writing a manifest")

    # ---- the manifest, with the metadata each product must carry --------
    rows = []
    for r in T.itertuples():
        for y in YEARS:
            rows.append(dict(
                analysis_zone=r.analysis_zone,
                analysis_subzones=r.analysis_subzones,
                n_subzones=r.n_subzones,
                year=y,
                source="Dynamic World",
                native_resolution_m=NATIVE_M,
                analysis_resolution_m=ANALYSIS_M,
                roi_km2=r.roi_km2,
                analysis_target_km2=r.analysis_target_km2,
                registry_domain_version=SD.geom_hash(r.geometry),
                coverage_gate=COVERAGE_GATE,
                # filled by the exporter, never guessed here
                valid_observation_count=np.nan,
                coverage_fraction=np.nan,
                status="PENDING_EXPORT",
                product=f"dw_{r.analysis_zone.lower()}_{y}.tif"))
    M = pd.DataFrame(rows)
    M.to_csv(CFG.TABLES / "p0n_dw_manifest.csv", index=False)
    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    T.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"\n-> {CFG.TABLES / 'p0n_dw_manifest.csv'}  ({len(M)} products)")
    print(f"-> {OUT_GEOJSON}")

    # ---- dry run --------------------------------------------------------
    print("\n" + "=" * 78)
    print("DRY RUN — what would be exported")
    print("=" * 78)
    px = (ANALYSIS_M / 1e3) ** 2
    for zn, g in M.groupby("analysis_zone"):
        cells = g.roi_km2.sum() / px
        print(f"  {zn:30s} {len(g):3d} products, "
              f"{g.roi_km2.sum():9,.0f} km2-years, ~{cells/1e6:5.1f} M cells")
    print(f"  {'TOTAL':30s} {len(M):3d} products")
    print("\n  Each product must carry: analysis_zone, analysis_subzone, year,")
    print("  source, native_resolution, analysis_resolution,")
    print("  valid_observation_count, coverage_fraction, registry_domain_version")
    print(f"  and must pass coverage_fraction >= {COVERAGE_GATE:.2f} of its ROI,")
    print("  with NaN admitted only where there are genuinely no valid")
    print("  observations -- never because the extent was wrong.")

    print("\n" + "=" * 78)
    if state == "EE_AUTH=OK":
        print("Earth Engine is authorised: the exporter may run.")
    else:
        print("STOP AT THE EXPORT BOUNDARY")
        print("=" * 78)
        print("  Everything above needs no credentials and is done. The export")
        print("  itself does. Nothing is run automatically on your behalf.")
        print(auth_help())


def ee_state():
    try:
        import ee
    except ImportError as e:
        return "EE_PACKAGE=MISSING", str(e)
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception as e:
        return "EE_AUTH=FAILED", f"{type(e).__name__}: {str(e)[:160]}"
    # PROBE WITH WORK, NOT WITH A METADATA ACCESSOR. This reported
    # EE_AUTH=FAILED on a perfectly authorised credential for as long as it
    # called ee.data.getCloudApiUserProject(), which was removed from the ee
    # package (1.7.43 raises AttributeError). The check was failing, not the
    # thing being checked - and the message it printed sent the reader off to
    # create service accounts they already had.
    try:
        ee.Number(1).add(1).getInfo()
        return "EE_AUTH=OK", f"project {EE_PROJECT}, compute answering"
    except Exception as e:
        return "EE_AUTH=FAILED", f"{type(e).__name__}: {str(e)[:160]}"


def auth_help():
    return """
  TO AUTHORISE. A service account is the right choice here because the export
  is scripted and repeatable; the interactive flow stores a credential in your
  home directory that a rerun on another machine will not have.

    1  create or pick a Cloud project
       https://console.cloud.google.com/projectcreate
    2  register that project for Earth Engine
       https://code.earthengine.google.com/register
    3  enable the Earth Engine API on it
       https://console.cloud.google.com/apis/library/earthengine.googleapis.com
    4  create a service account and download a JSON key
       https://console.cloud.google.com/iam-admin/serviceaccounts
    5  register the service account for Earth Engine
       https://signup.earthengine.google.com/#!/service_accounts

  then put these in .env:

    GEE_PROJECT=your-project-id
    GEE_SERVICE_ACCOUNT=name@your-project.iam.gserviceaccount.com
    GEE_SERVICE_ACCOUNT_KEY=/absolute/path/to/key.json

  The interactive alternative, if you prefer it, is `earthengine authenticate`
  followed by GEE_PROJECT=your-project-id in .env.

  NOTE ON THE EXISTING .env. GEE_TOKEN currently holds a 62-character value
  beginning "4/1A". That is a one-time OAuth AUTHORIZATION CODE, not a token:
  it is exchanged for credentials within minutes of being issued and has long
  since expired. It authorises nothing and should be removed rather than left
  looking like a working secret."""


if __name__ == "__main__":
    main()
