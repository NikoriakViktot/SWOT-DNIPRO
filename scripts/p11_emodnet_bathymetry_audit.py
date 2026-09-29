#!/usr/bin/env python
"""P11 — what the EMODnet Dnieper-Bug bathymetry actually contains.

The liman phases have been waiting on a depth source, and one has arrived:
EMODnet DTM 2024 over 31.5-33.45 E, 46.30-47.15 N, pulled from the ERDDAP
griddap endpoint on 2026-09-14 and also written as two GeoTIFFs (EPSG:4326 and
a 100 m EPSG:32636 resample).

BEFORE IT IS USED FOR ANYTHING, read its own flag. The file carries
`interpolation_flag`, and its attributes define it without ambiguity:

    flag_meaning : not_interpolated interpolated
    flag_values  : [0 1]
    long_name    : Indicator of cell processed as extrapolation of the
                   neighbouring cells (ABSENCE OF REAL SOUNDINGS DATA)

Counted over this window:

    finite elevation                158,240 cells
      flag == 1  interpolated       156,670
      flag == 0  not interpolated         0        <-- none. not one.
      flag absent, elevation 0.00     1,570        coastline fill

There is not a single measured sounding in this grid over our domain. Every
depth in it was extrapolated from neighbouring cells, `stdev` is empty in every
cell, `elevation_min` and `elevation_max` are empty in every cell, and the whole
156,670-cell field carries just TWO distinct `cdi_index` values - two source
datasets for the entire estuary, with no `cdi_reference` variable in the file to
say what they are.

THE CONSEQUENCE, stated plainly so nobody has to rediscover it: this product is
a PRIOR ON SHAPE, not a measurement. It may seed an interpolation, provide a
first guess, or be compared against. It may NOT be used as validation truth, it
may not be reported as "measured depth", and an RMSE computed against it is an
RMSE against somebody else's interpolation.

COVERAGE, with the TRUE domain area as the denominator - not the part that
happens to fall inside the raster, which is the mistake this pipeline keeps
catching:

    ZONE_3 water     2,342 km2 true,  1,887 in window,  1,424 with depth   60.8%
    ZONE_2 water       309 km2 true,    309 in window,    105 with depth   34.0%
    ZONE_4 water       592 km2 true,    350 in window,      0 with depth    0.0%

Two separate gaps, and they need different fixes. The WINDOW stops at 31.5 E
while ZONE_3 reaches 31.33 E, so 455 km2 of estuary water was never requested -
re-pull with a wider bounding box. Within the window, a further 24% of ZONE_3
water carries no value at all, which no re-pull will change.

THE VERTICAL DATUM IS LAT (SDN:P01::HGHTALAT, Lowest Astronomical Tide), which
is not the datum anything else in this project uses. Black Sea tides are small,
so the LAT-to-mean-sea offset is small too - but small is not zero and not
known, and the reconciliation chain in vertical.py exists precisely so that an
offset is transferred rather than assumed. Until that transfer is made, depths
from this file may not be differenced against EVRF2019 or BS-77 surfaces.

Outputs
-------
outputs/tables/p11_emodnet_bathymetry_audit.csv
outputs/tables/p11_emodnet_provenance.json
"""
from __future__ import annotations

import json
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
import xarray as xr
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import box

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

BATH = ROOT / "data/bathymetry/dnieper_bug"
NC = BATH / "EMODnet_DTM2024_DnieperBug_extended.nc"
TIF_UTM = BATH / "DBE_bathymetry_UTM36N_100m_LAT_extended.tif"
TIF_LL = BATH / "EMODnet_elevation_LAT_extended.tif"
TARGETS = ("ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY",
           "ZONE_4_DAM_TO_KHERSON_FLOODWAY")


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P11 — EMODnet Dnieper-Bug bathymetry: what is actually in it")
    print("=" * 78)
    print(f"  git {commit}")
    for p in (NC, TIF_UTM, TIF_LL):
        if not p.exists():
            raise SystemExit(f"missing {p}")
        print(f"  {p.name:48s} {p.stat().st_size/1e6:7.1f} MB")

    # ---------------------------------------------------------- provenance
    d = xr.open_dataset(NC)
    e = d.elevation.values
    f = d.interpolation_flag.values
    vc = d.value_count.values
    ci = d.cdi_index.values
    ok = np.isfinite(e)
    n_int = int((ok & (f == 1)).sum())
    n_meas = int((ok & (f == 0)).sum())
    n_noflag = int((ok & ~np.isfinite(f)).sum())

    print("\n" + "=" * 78)
    print("IS THIS A MEASUREMENT?  read the file's own flag")
    print("=" * 78)
    print(f"  interpolation_flag  {d.interpolation_flag.attrs.get('flag_meaning')}"
          f"  values {list(d.interpolation_flag.attrs.get('flag_values', []))}")
    print(f"  long_name: {d.interpolation_flag.attrs.get('long_name')}")
    print(f"\n  finite elevation            {int(ok.sum()):9,d} cells")
    print(f"    flag == 1  interpolated   {n_int:9,d}")
    print(f"    flag == 0  NOT interp.    {n_meas:9,d}")
    print(f"    flag absent               {n_noflag:9,d}  "
          f"(elevation exactly 0.00: coastline fill)")
    if n_meas == 0:
        print("\n  NO MEASURED SOUNDING EXISTS IN THIS GRID OVER OUR DOMAIN.")
        print("  Every depth was extrapolated from neighbouring cells. Usable")
        print("  as a shape prior; never as validation truth, never reported")
        print("  as measured depth, and an RMSE against it is an RMSE against")
        print("  another interpolation.")
    else:
        print(f"\n  {n_meas:,} cells carry real soundings -- those, and only")
        print("  those, may be treated as measurement.")

    sd = d.stdev.values[ok]
    mn, mx = d.elevation_min.values, d.elevation_max.values
    print(f"\n  stdev finite in        {int(np.isfinite(sd).sum()):,} of "
          f"{int(ok.sum()):,} cells")
    print(f"  elevation_min/max in   "
          f"{int((ok & np.isfinite(mn) & np.isfinite(mx)).sum()):,} cells")
    u = np.unique(ci[ok][np.isfinite(ci[ok])])
    print(f"  distinct cdi_index     {len(u)}  {u[:6]}")
    print(f"  cdi_reference present  {'cdi_reference' in d.variables}")
    print(f"  value_count            "
          f"{sorted({int(v) for v in np.unique(vc[ok][np.isfinite(vc[ok])])})[:8]}")

    print("\n  VERTICAL DATUM")
    print(f"    {d.elevation.attrs.get('sdn_parameter_name')}")
    print(f"    {d.elevation.attrs.get('sdn_parameter_urn')}")
    print("    LAT is not EVRF2019 and not BS-77. The offset is small in the")
    print("    Black Sea and it is not zero and not known; transfer it through")
    print("    vertical.py before differencing anything against this file.")

    # ---------------------------------------------------------- coverage
    print("\n" + "=" * 78)
    print("COVERAGE, against the TRUE domain area")
    print("=" * 78)
    with rasterio.open(TIF_UTM) as ds:
        a = ds.read(1, masked=True)
        tr, shp = ds.transform, (ds.height, ds.width)
        ext = box(*ds.bounds)
        res = ds.res[0]
    val = (~a.mask) & np.isfinite(a.filled(np.nan))
    px = res ** 2 / 1e6
    print(f"  raster {shp[1]}x{shp[0]} at {res:.0f} m, "
          f"E {ext.bounds[0]:,.0f}..{ext.bounds[2]:,.0f}  "
          f"N {ext.bounds[1]:,.0f}..{ext.bounds[3]:,.0f}")
    print(f"  cells with a depth {int(val.sum()):,} = {val.sum()*px:,.0f} km2, "
          f"range {a.min():.2f} .. {a.max():.2f} m LAT\n")

    wdom = SD.load_utm("dnipro_water_domain")
    rows = []
    print(f"  {'target':38s} {'true':>8s} {'in window':>10s} {'has depth':>10s} "
          f"{'% true':>8s}")
    for z in TARGETS:
        g0 = SD.load_utm(z)
        for lab, g in ((z, g0), (z + " water", g0.intersection(wdom))):
            true = g.area / 1e6
            inw = g.intersection(ext).area / 1e6
            if inw <= 0:
                hd = 0.0
            else:
                m = rio_rasterize([(g, 1)], out_shape=shp, transform=tr,
                                  fill=0, dtype="uint8").astype(bool)
                hd = float((val & m).sum()) * px
            rows.append(dict(target=lab, true_km2=true, in_window_km2=inw,
                             with_depth_km2=hd,
                             pct_of_true=100 * hd / true if true else np.nan,
                             pct_of_window=100 * hd / inw if inw else np.nan))
            print(f"  {lab:38s} {true:8,.0f} {inw:10,.0f} {hd:10,.0f} "
                  f"{100*hd/true if true else 0:7.1f}%")

    R = pd.DataFrame(rows)
    t = CFG.TABLES / "p11_emodnet_bathymetry_audit.csv"
    R.to_csv(t, index=False)

    z3 = R[R.target == "ZONE_3_DNIPRO_BUG_ESTUARY water"].iloc[0]
    outside = z3.true_km2 - z3.in_window_km2
    print("\n  TWO SEPARATE GAPS, needing different fixes:")
    print(f"    {outside:,.0f} km2 of ZONE_3 water lies OUTSIDE the requested "
          f"window")
    print(f"      the pull stopped at 31.50 E; ZONE_3 reaches "
          f"{SD.load_utm('ZONE_3_DNIPRO_BUG_ESTUARY').bounds[0]:,.0f} m E "
          f"(~31.33 E). Re-pull wider.")
    print(f"    {z3.in_window_km2 - z3.with_depth_km2:,.0f} km2 inside the "
          f"window carries no value at all")
    print(f"      ({100 - z3.pct_of_window:.0f}% of the in-window water). No "
          f"re-pull changes that.")

    prov = dict(
        product="EMODnet DTM 2024, Dnieper-Bug",
        files=[p.name for p in (NC, TIF_UTM, TIF_LL)],
        retrieved=str(d.attrs.get("history", ""))[:300],
        vertical_datum="LAT (SDN:P01::HGHTALAT)",
        datum_reconciled_to_project_frame=False,
        native_resolution_deg=float(abs(d.longitude[1] - d.longitude[0])),
        resampled_resolution_m=float(res),
        cells_finite=int(ok.sum()), cells_interpolated=n_int,
        cells_not_interpolated=n_meas, cells_flag_absent=n_noflag,
        distinct_cdi_index=int(len(u)),
        has_cdi_reference=bool("cdi_reference" in d.variables),
        has_stdev=bool(np.isfinite(sd).any()),
        role="SHAPE PRIOR ONLY -- not measurement, not validation truth",
        coverage=R.to_dict("records"), git=commit)
    j = CFG.TABLES / "p11_emodnet_provenance.json"
    j.write_text(json.dumps(prov, indent=2, default=float))
    print(f"\n-> {t}")
    print(f"-> {j}")
    print("\n  NEXT: register the file in the provenance system with role")
    print("  SHAPE_PRIOR, re-pull the window west to ~31.2 E, and establish")
    print("  the LAT -> project datum transfer before any differencing.")


if __name__ == "__main__":
    main()
