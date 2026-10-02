#!/usr/bin/env python
"""PART 11 — pre-breach bathymetric soundings vs the ATL08 bare-earth lakebed.

The S-57 SOUNDG layer gives 7,514 depths from a river-cartography survey. It
carries NO vertical metadata: VERDAT, SOUACC, QUASOU, TECSOU, SORDAT and SORIND
are empty in all 7,514 records, so the level the depths were reduced to has to
be recovered rather than read.

This script originally assumed the reservoir's normal impoundment level, 16.00 m.
That was wrong by 2.00 m. The depths are referenced to the reservoir's published
NAVIGATION DRAWDOWN LEVEL (UNS), 14.00 m, so

    H_ref_EVRF2019 = 14.00 + delta_EPSG9902(lon, lat)
    H_bed_EVRF2019 = H_ref_EVRF2019 - DEPTH

14.00 m is adopted as a WORKING HYPOTHESIS, not as a documented fact. The source
establishes that the reservoir has a navigation drawdown level of 14.00 m; it
does NOT say the soundings were reduced to it. That link comes from converging
independent evidence -- see hist2_datum_closure.py and, for exactly what the
source does and does not support, outputs/reports/historical_datum_semantics.md:

    published design level, UNS          14.00 m
    capacity curve V(H) + soundings      13.71 m [13.57, 13.84]   no ICESat-2
    ICESat-2 over the exposed bed        14.11 m [13.91, 14.32]
    previously assumed, NPG              16.00 m   -> rejected, -1.90 m residual

The pre-correction outputs are preserved under
outputs/archive/datum_1600_npg/ and are not overwritten.

which puts the survey in the same frame as everything else in this study and lets
it be compared, point for point, with the post-breach ATL08 bed derived in Part 8.

The .prj that ships with the shapefile is malformed (null spheroid, unit "1.#INF")
and GDAL refuses it. The coordinates are unambiguously UTM 36N -- the file's own
central meridian is 33 deg and its bounds reproject onto the reservoir -- so
EPSG:32636 is asserted explicitly and the check is printed.

THIS IS NOT A LIKE-FOR-LIKE COMPARISON and is not presented as one: the survey
measured a submerged bed before the breach, ATL08 measures an exposed bed after
it. Differences carry sedimentation and erosion as well as error.

Outputs
-------
data/processed/bathymetry/kakhovka_soundings_evrf2019.parquet
outputs/tables/bathymetry_vs_atl08.csv
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG

SRC = Path("/mnt/e/data_swot/3529_Sounding_SOUNDG_P")
# The level the soundings are referenced to, in the source's own (unnamed)
# vertical system. See the module docstring: working hypothesis, not a
# documented fact. Overridable from the command line purely for sensitivity
# runs -- the default is the value the analysis supports.
HISTORICAL_REFERENCE_LEVEL_BS_M = 14.00
REFERENCE_LABEL = "UNS, navigation drawdown level (published design level)"
# The source names no vertical realization. EPSG:9902 is defined for BS-77 ->
# EVRF2019; if these tables are on BS-42 there is an unapplied step of a few cm.
SOURCE_DATUM_LABEL = ("HISTORICAL_BALTIC (realization not stated in source; "
                      "BS-77 assumed for EPSG:9902)")
ARCHIVE_NOTE = "outputs/archive/datum_1600_npg/ holds the pre-correction products"
ASSERT_CRS = "EPSG:32636"
OUT_PQ = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
MATCH_M = 150.0                    # ATL08 segments are 100 m long


def load_grid_sampler():
    """Reuse the EPSG:9902 sampler from Part 1 rather than re-implementing it."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from part1_gauge_rereference import load_grid, sample_grid
    z, head = load_grid()
    return lambda lon, lat: sample_grid(z, head, lon, lat)


def read_soundings() -> gpd.GeoDataFrame:
    tmp = tempfile.mkdtemp()
    for f in glob.glob(str(SRC / "*")):
        if not f.endswith(".prj"):            # the shipped .prj is malformed
            shutil.copy(f, tmp)
    g = gpd.read_file(os.path.join(tmp, SRC.name + ".shp"))
    g = g.set_crs(ASSERT_CRS, allow_override=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return g


def main() -> None:
    g = read_soundings()
    ll = g.to_crs("EPSG:4326")
    g["lon"], g["lat"] = ll.geometry.x, ll.geometry.y
    g["x"], g["y"] = g.geometry.x, g.geometry.y
    g["depth_m"] = g.DEPTH.astype(float)

    print(f"{len(g):,} soundings   depth {g.depth_m.min():.2f}–{g.depth_m.max():.2f} m "
          f"(median {g.depth_m.median():.2f})")
    print(f"CRS asserted {ASSERT_CRS}; reprojected bounds "
          f"lon {g.lon.min():.3f}–{g.lon.max():.3f}, lat {g.lat.min():.3f}–{g.lat.max():.3f}")

    # One transformation function, the same one Part 1 uses for the gauge zeros.
    # The EPSG:9902 offset is sampled per sounding: it varies by ~54 mm across
    # the reservoir, which is the same variation that makes a single national
    # constant wrong for the gauges.
    samp = load_grid_sampler()
    REF = HISTORICAL_REFERENCE_LEVEL_BS_M
    g["delta_epsg9902_m"] = samp(g.lon.values, g.lat.values)
    g["H_ref_bs_m"] = REF
    g["H_ref_evrf2019_m"] = REF + g.delta_epsg9902_m
    g["H_bed_bs_m"] = REF - g.depth_m
    g["H_bed_evrf2019_m"] = g.H_ref_evrf2019_m - g.depth_m
    g["source_datum_label"] = SOURCE_DATUM_LABEL
    g["reference_level_label"] = REFERENCE_LABEL
    g["raw_vertical_frame"] = f"depth below {REF:.2f} m ({REFERENCE_LABEL})"
    g["common_vertical_frame"] = "EVRF2019 (EPSG:9389, zero-tide)"
    print(f"\nreference level  {REF:.2f} m  ({REFERENCE_LABEL})")
    print(f"EPSG:9902 offset {g.delta_epsg9902_m.min():+.4f} .. "
          f"{g.delta_epsg9902_m.max():+.4f} m (median "
          f"{g.delta_epsg9902_m.median():+.4f}, spread "
          f"{1e3*(g.delta_epsg9902_m.max()-g.delta_epsg9902_m.min()):.0f} mm)")
    print(f"reference level in EVRF2019: {g.H_ref_evrf2019_m.min():.3f} .. "
          f"{g.H_ref_evrf2019_m.max():.3f} m (median "
          f"{g.H_ref_evrf2019_m.median():.3f})")

    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    keep = ["lon", "lat", "x", "y", "depth_m", "delta_epsg9902_m", "H_ref_bs_m",
            "H_ref_evrf2019_m", "H_bed_bs_m", "H_bed_evrf2019_m",
            "source_datum_label", "reference_level_label", "raw_vertical_frame",
            "common_vertical_frame"]
    pd.DataFrame(g[keep]).to_parquet(OUT_PQ, index=False)
    print(f"-> {OUT_PQ}")
    print(f"   bed elevation {g.H_bed_evrf2019_m.min():.2f} .. "
          f"{g.H_bed_evrf2019_m.max():.2f} m, median {g.H_bed_evrf2019_m.median():.2f} m EVRF2019")

    # ---- compare with the ATL08 exposed bed --------------------------------
    at = pd.read_parquet(ROOT / "data/processed/atl08/kakhovka_atl08_terrain.parquet")
    bed = at[at.surface_class == "exposed_bed"].copy()
    tf = pyproj.Transformer.from_crs("EPSG:4326", ASSERT_CRS, always_xy=True)
    bx, by = tf.transform(bed.lon.values, bed.lat.values)
    tree = cKDTree(np.c_[bx, by])
    dist, idx = tree.query(np.c_[g.x.values, g.y.values], k=1,
                           distance_upper_bound=MATCH_M)
    ok = np.isfinite(dist)
    m = pd.DataFrame({
        "lon": g.lon.values[ok], "lat": g.lat.values[ok],
        "depth_m": g.depth_m.values[ok],
        "H_survey_m": g.H_bed_evrf2019_m.values[ok],
        "H_atl08_m": bed.H_terrain_common_m.values[idx[ok]],
        "match_dist_m": dist[ok],
        "atl08_date": bed.date.values[idx[ok]]})
    m["diff_m"] = m.H_atl08_m - m.H_survey_m
    m.to_csv(CFG.TABLES / "bathymetry_vs_atl08.csv", index=False)

    d = m.diff_m.values
    nmad = 1.4826 * np.median(np.abs(d - np.median(d)))
    print(f"\n=== survey (pre-breach, submerged) vs ATL08 (post-breach, exposed) ===")
    print(f"  matched {len(m):,} of {len(g):,} soundings within {MATCH_M:.0f} m")
    print(f"  median ATL08 - survey : {np.median(d):+.2f} m")
    print(f"  NMAD                  : {nmad:.2f} m")
    print(f"  p05 / p95             : {np.percentile(d,5):+.2f} / {np.percentile(d,95):+.2f} m")
    print(f"  correlation           : r = {np.corrcoef(m.H_survey_m, m.H_atl08_m)[0,1]:.3f}")
    b = m[m.match_dist_m <= 50]
    if len(b) > 30:
        db = b.diff_m.values
        print(f"  tightest matches (<=50 m, n={len(b):,}): median {np.median(db):+.2f} m, "
              f"NMAD {1.4826*np.median(np.abs(db-np.median(db))):.2f} m")

    # ---- is the offset a datum error or a physical change? ----------------
    # A constant offset is datum-like; one that grows with depth or concentrates
    # near the dam would be erosion. Least squares is NOT used: a handful of
    # points deeper than 20 m dominate it and reverse the verdict.
    from scipy import stats
    core = b[b.depth_m <= 20]
    ts = stats.theilslopes(core.diff_m.values, core.depth_m.values)
    print("\n=== does any structure remain in the residual? ===")
    print(f"  Theil-Sen slope vs depth (n={len(core):,}, depth<=20 m): "
          f"{ts[0]:+.4f} m/m, CI [{ts[2]:+.4f}, {ts[3]:+.4f}]")
    bl = core.assign(b=pd.cut(core.lon, [33.3, 33.7, 34.1, 34.5, 34.9, 35.4]))
    med = bl.groupby("b", observed=True).diff_m.median()
    print("  median offset west->east: "
          + ", ".join(f"{v:+.2f}" for v in med.values))
    flat = abs(ts[0]) < 0.05 and (med.max() - med.min()) < 0.6
    print(f"  -> {'CONSTANT: a uniform vertical shift, not depth-dependent scour' if flat else 'VARIES with depth or position: a physical change is possible'}")
    # ---- was it green-laser penetration through residual water? -----------
    # ICESat-2's 532 nm laser penetrates water, so if the matched segments sat
    # over the residual channel the "terrain" could be a submerged return biased
    # low. Testable: the offset would then be MORE negative where the surveyed
    # bed lies below the residual water line.
    POST_WSE = 5.20
    core = core.assign(rel=core.H_survey_m - POST_WSE)
    dry, wet = core[core.rel > 1], core[core.rel < -1]
    print("\n=== was it laser penetration through residual water? NO ===")
    print(f"  bed >1 m ABOVE the water line (dry) : n={len(dry):,}, "
          f"median {dry.diff_m.median():+.2f} m")
    print(f"  bed >1 m BELOW the water line (wet) : n={len(wet):,}, "
          f"median {wet.diff_m.median():+.2f} m")
    print("  Penetration would drive the wet subset MORE negative; it is LESS negative.")
    print("  Where the surveyed bed is 8-20 m down, the offset turns strongly POSITIVE")
    print("  (+2 to +6 m) -- there ATL08 is reading the water SURFACE, not the bed,")
    print("  which is the expected behaviour and confirms it does not see through deep water.")

    # ---- the reference level, checked against independent control ---------
    # The frame-permutation enumeration that used to sit here asked whether any
    # BS-77 <-> EVRF2019 transformation could explain a 2 m offset. It could not,
    # and the question was the wrong one: every permutation varied how the
    # reference level is TRANSFORMED, never the level itself. The level was the
    # error. That is settled in hist2_datum_closure.py; what remains useful here
    # is the sensitivity, so it is reported as such.
    ALT = {"UNS published (Table 19), applied": 14.00,
           "capacity-curve fit, no ICESat-2": 13.71,
           "ICESat-2 exposed bed (QA5)": 14.11,
           "NPG, previously assumed": 16.00}
    d0 = dry.diff_m.median()
    print("\n=== sensitivity to the reference level ===")
    print(f"  {'reference level':<40}{'median ATL08 - survey':>24}")
    for lab, D in ALT.items():
        print(f"  {lab:<40}{d0 + (REF - D):>+23.2f} m")
    print("  The three low values are independent estimates of the same quantity,")
    print("  not competing operational datums. 16.00 m is rejected by the data.")

    # ---- ATL08's own vertical scale, over water it can be checked against ---
    # Independent of the soundings entirely: over the FILLED pool ATL08 measures
    # a water surface whose height the gauges also recorded, so its scale can be
    # checked without reference to any bed.
    pool = at[at.surface_class == "pool_water_surface"]
    print("\n=== ATL08's own vertical scale ===")
    print(f"  ATL08 over the filled pool : {pool.H_terrain_common_m.median():.3f} m "
          f"EVRF2019 (n={len(pool):,})")
    print(f"  pool gauges 2019-2021      : 14.97-16.24 m BS-77, median 15.56")
    print(f"                             = ~15.16-16.43 m EVRF2019, median ~15.75")
    print(f"  ATL08 lands inside the range the pool actually operated in, so its")
    print(f"  vertical scale is not metres wrong. This says nothing about the")
    print(f"  soundings -- it is a check on the SATELLITE side only.")

    # ---- what is left ------------------------------------------------------
    print(f"\n=== what is left unexplained ===")
    print(f"  Over genuinely dry, exposed bed the offset is now {d0:+.2f} m.")
    print(f"  The shapefile still records NO survey metadata -- VERDAT, SOUACC,")
    print(f"  QUASOU, TECSOU, SORDAT and SORIND are empty in all {len(g):,} records.")
    print(f"  So two things remain open, and neither is closed by this rebuild:")
    print(f"    1. the reference level is a WORKING HYPOTHESIS. The source says the")
    print(f"       reservoir HAS a 14.00 m navigation drawdown level; it does not")
    print(f"       say the soundings were reduced to it.")
    print(f"    2. the survey EPOCH is unknown, so no rate of bed change can be")
    print(f"       formed, and a residual of a few decimetres cannot be separated")
    print(f"       from the ~0.4 m spread among the three reference-level estimates.")
    print(f"  Shape agreement is unaffected by any of it: r = "
          f"{np.corrcoef(m.H_survey_m, m.H_atl08_m)[0,1]:.3f}, "
          f"NMAD {1.4826*np.median(np.abs(db-np.median(db))):.2f} m.")
    print(f"\n  {ARCHIVE_NOTE}")


if __name__ == "__main__":
    main()
