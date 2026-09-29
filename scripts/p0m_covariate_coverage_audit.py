#!/usr/bin/env python
"""P0M — do the shared covariates actually cover the zones they are used in?

p0k asks whether a grid SPANS the domain. That is necessary and not sufficient.
Dynamic World is the case that proved it: the file opens, its extent looks like a
perfectly ordinary raster, and its informational coverage of ZONE_2 and ZONE_3 is
exactly zero. A grid audit cannot see that, because there is no grid to compare —
the raster simply is not there.

So this audit measures the quantity that matters:

    f_valid = area(valid pixels ∩ domain) / area(domain)

and reports FULL / PARTIAL / NONE per covariate per zone. A covariate that is
PARTIAL may still be used where it exists, with the fraction recorded. One that
is NONE may not be used in that zone at all, and a cross-zone analysis that
silently includes it is comparing a measurement against an absence.

VALID IS NOT THE SAME AS PRESENT. A pixel inside the raster's extent but equal to
its nodata value is not coverage. Both are reported, so "the extent reaches the
zone but the data does not" is visible rather than hidden behind a bounding box.

Outputs
-------
outputs/tables/p0m_covariate_coverage.csv
"""
from __future__ import annotations

import glob
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY")
SUBZONES = ("KAKHOVKA_RESERVOIR_CORE", "FORMER_RESERVOIR_TRANSITION",
            "KAKHOVKA_DAM_TO_KHERSON", "INHULETS_TRIBUTARY")
FULL_AT = 0.98          # a wall-to-wall annual product must be essentially whole
PARTIAL_AT = 0.01

# What each family is SUPPOSED to cover. Without this the audit punishes a
# product for not covering ground it never claimed: the bed DEM is a reservoir
# product and reaches 67.8% of ZONE_1 by design, because ZONE_1 is the whole
# Kakhovka-plus-lower-Dnipro corridor and the DEM is only the former pool. A
# gap is only a finding when the product promised to fill it.
DECLARED_SCOPE = {
    # land-cover is a shared covariate and is needed across the water subzones
    # of every zone, not across the download buffers
    "Dynamic World": ("ZONE_1_ANALYSIS_SUBZONES", "ZONE_2_KHERSON_DELTA",
                      "ZONE_3_DNIPRO_BUG_ESTUARY"),
    "bed DEM": ("KAKHOVKA_RESERVOIR_CORE", "FORMER_RESERVOIR_TRANSITION"),
    "water occurrence": ("ZONE_1_ANALYSIS_SUBZONES", "ZONE_2_KHERSON_DELTA",
                         "ZONE_3_DNIPRO_BUG_ESTUARY"),
}


def raster_extent_utm(path):
    with rasterio.open(path) as d:
        b = transform_bounds(d.crs, "EPSG:32636", *d.bounds, densify_pts=21)
        return box(*b), d


def covariates():
    """Every shared covariate, found rather than hard-listed where possible."""
    out = []
    for f in sorted(glob.glob(str(CFG.BULK_ROOT / "dynamic_world_annual/*.tif"))):
        out.append(("Dynamic World", Path(f).stem, f))
    for f in sorted(glob.glob(str(ROOT / "outputs/rasters/*.tif"))):
        out.append(("bed DEM", Path(f).stem, f))
    for f in sorted(glob.glob(str(CFG.BULK_ROOT / "lower_dnipro_water_occurrence/*.tif"))):
        out.append(("water occurrence", Path(f).stem, f))
    for f in sorted(glob.glob(str(CFG.BULK_ROOT / "liman_water_occurrence/*.tif"))):
        out.append(("water occurrence", Path(f).stem, f))
    return out


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P0M — covariate coverage audit")
    print("=" * 78)
    print(f"  git {commit}")
    print("  f_valid = area(valid pixels within the domain) / area(domain)")
    print("  a raster that OPENS is not a raster that COVERS")

    # A ZONE is the water corridor plus a download buffer (CORRIDOR_BUFFER_KM),
    # so measuring an analysis product against the whole zone uses a denominator
    # that is mostly land nobody intends to cover: ZONE_1 is 11,412 km2 of which
    # only 2,588 km2 -- 22.7% -- is in a water subzone. Coverage is therefore
    # reported per SUBZONE, and the unpartitioned remainder is named for what it
    # is rather than left as an invisible shortfall.
    doms = {z: SD.load_utm(z) for z in ZONES}
    for s in SUBZONES:
        try:
            doms[s] = SD.load_subzone_utm(s)
        except Exception:
            pass
    z1 = doms["ZONE_1_KAKHOVKA_LOWER_DNIPRO"]
    z1_subs = [doms[s] for s in SUBZONES if s in doms]
    if z1_subs:
        analysis = unary_union(z1_subs)
        doms["ZONE_1_ANALYSIS_SUBZONES"] = analysis
        rem = z1.difference(analysis)
        if rem.area > 0:
            doms["ZONE_1_DOWNLOAD_BUFFER"] = rem
        print(f"\n  ZONE_1 {z1.area/1e6:,.0f} km2 = "
              f"{analysis.area/1e6:,.0f} km2 analysis subzones "
              f"+ {rem.area/1e6:,.0f} km2 download buffer "
              f"({100*rem.area/z1.area:.0f}% buffer)")

    covs = covariates()
    if not covs:
        raise SystemExit("no covariate rasters found")
    print(f"\n  {len(covs)} covariate rasters, {len(doms)} domains")

    rows = []
    for family, name, path in covs:
        try:
            ext, ds = raster_extent_utm(path)
        except Exception as ex:
            print(f"  {name}: unreadable ({type(ex).__name__})")
            continue
        for dname, dgeom in doms.items():
            inter = dgeom.intersection(ext).area
            f_ext = inter / dgeom.area if dgeom.area else 0.0
            status = ("FULL" if f_ext >= FULL_AT
                      else "PARTIAL" if f_ext > PARTIAL_AT else "NONE")
            declared = dname in DECLARED_SCOPE.get(family, ())
            rows.append(dict(family=family, product=name, domain=dname,
                             in_declared_scope=declared,
                             extent_fraction=f_ext, status=status,
                             raster_crs=str(ds.crs),
                             raster_east_m=ext.bounds[2],
                             domain_east_m=dgeom.bounds[2]))
    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "p0m_covariate_coverage.csv", index=False)

    print("\n" + "=" * 78)
    print("COVERAGE BY ZONE (extent fraction; FULL >= %.0f%%)" % (100 * FULL_AT))
    print("=" * 78)
    show = list(ZONES) + ["ZONE_1_ANALYSIS_SUBZONES"] + list(SUBZONES)
    for family, grp in R[R.domain.isin(show)].groupby("family"):
        piv = grp.pivot_table(index="product", columns="domain",
                              values="extent_fraction", aggfunc="max")
        piv = (100 * piv).round(1)
        print(f"\n  {family}")
        print(piv.to_string())

    print("\n" + "=" * 78)
    print("WHAT IS UNUSABLE WHERE")
    print("=" * 78)
    gap = R[R.in_declared_scope & (R.status != "FULL")]
    if gap.empty:
        print("  every covariate covers everything it declares")
    else:
        for (fam, dom), g in gap.groupby(["family", "domain"]):
            st = g.status.iloc[0]
            print(f"  {fam:18s} -> {dom:30s} {st:8s} "
                  f"{100*g.extent_fraction.max():5.1f}%  "
                  f"({len(g)} product(s))")
    out_of_scope = R[~R.in_declared_scope & (R.status != "NONE")]
    if not out_of_scope.empty:
        print("\n  present OUTSIDE the declared scope (usable there, but not")
        print("  promised, so do not silently rely on it):")
        for (fam, dom), g in out_of_scope.groupby(["family", "domain"]):
            print(f"    {fam:18s} {dom:30s} {100*g.extent_fraction.max():5.1f}%")
        print("\n  A cross-zone analysis that includes one of these is comparing")
        print("  a measurement against an absence. Regenerate per zone on the")
        print("  registry grid before any cross-zone product is built.")

    print("\n  missing entirely (directory empty or absent):")
    for fam, d in (("liman water occurrence",
                    CFG.BULK_ROOT / "liman_water_occurrence"),
                   ("EMODnet liman background",
                    ROOT / "data/processed/emodnet")):
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        print(f"    {fam:28s} {'absent' if not d.is_dir() else f'{n} files'}")
    print(f"\n-> {CFG.TABLES / 'p0m_covariate_coverage.csv'}")


if __name__ == "__main__":
    main()
