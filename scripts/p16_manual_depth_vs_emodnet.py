#!/usr/bin/env python
"""P16 — manual chart soundings vs EMODnet, a first look.

P11 established EMODnet's Dnieper-Bug pull is 99.2% `interpolation_flag==1`:
a shape prior, not a measurement, with nothing to validate it against inside
this project until now. `depth_manual/manual_depth_points_POINT.gpkg` (F:) is
the first real soundings this domain has -- values hand-clicked off 9
confirmed Soviet bathymetric chart sheets after OCR digitisation of the depth
labels ran at ~98% false positives (see depth_manual/README.md).

THIS IS A FIRST LOOK, NOT A VALIDATION RESULT. Two things are not resolved
here and must not be read past:

  DATUM. The chart's soundings are referenced to whatever datum that chart
  series used (a Soviet river chart typically states its own "проектный
  уровень" / gauge zero, printed on the sheet legend), NOT EMODnet's LAT.
  P11 already flagged that EMODnet depths may not be differenced against
  EVRF2019/BS-77 until that transfer is made; the same caution applies here,
  one datum stronger, because the chart's own datum is not yet read off the
  sheets either. Every difference below is BIAS + noise, and the bias
  includes an unknown datum offset -- only the bias-REMOVED residual speaks
  to spatial agreement.

  DATA QUALITY. sheet_id/raw_label per click is not required -- most points
  are depth_m + location only, and that is fine. Separately, 14 points carry
  depth_m in [21, 89] with NEITHER sheet_id NOR raw_label filled, no
  physically plausible reading of this system's soundings (the Dnipro-Buh
  liman and lower Dnipro do not carry 40-90 m anywhere), and they are exactly
  the largest outliers against EMODnet. They read as a distinct, separate
  batch -- not calibration noise -- and are flagged, not silently averaged in.

Outputs
-------
outputs/tables/p16_manual_depth_vs_emodnet.csv   (per-point, with flags)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import rasterio

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

POINTS = Path("/mnt/f/data_kakhovka_dem_swot/depth_manual/manual_depth_points_POINT.gpkg")
EMODNET = ROOT / "data/processed/emodnet/D0_liman_background.tif"
IMPLAUSIBLE_DEPTH_M = 20.0   # nothing in this system's charted extent is deeper
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")


def main() -> None:
    print("=" * 78)
    print("P16 — manual chart soundings vs EMODnet (first look, NOT a validation)")
    print("=" * 78)

    gdf = gpd.read_file(POINTS)
    n_total = len(gdf)
    n_null_geom = int(gdf.geometry.isna().sum())
    n_null_depth = int(gdf.depth_m.isna().sum())
    d = gdf[~gdf.geometry.isna() & ~gdf.depth_m.isna()].copy()
    print(f"  {n_total} rows: {n_null_geom} no geometry, {n_null_depth} no depth_m "
          f"-> {len(d)} usable")

    d["has_sheet_id"] = d.sheet_id.notna()
    print(f"  {d.has_sheet_id.sum()}/{len(d)} ({100*d.has_sheet_id.mean():.1f}%) "
          f"carry sheet_id (not required; most points are depth_m + location only)")

    d["implausible"] = (d.depth_m > IMPLAUSIBLE_DEPTH_M) & d.sheet_id.isna() & d.raw_label.isna()
    n_imp = int(d.implausible.sum())
    if n_imp:
        print(f"\n  {n_imp} points: depth_m > {IMPLAUSIBLE_DEPTH_M:.0f} m AND no "
              f"sheet_id/raw_label -- flagged IMPLAUSIBLE, not this system's range:")
        print("    " + ", ".join(f"{v:.0f}" for v in
                                 sorted(d.loc[d.implausible, "depth_m"])))

    for z in ZONES:
        inside = d.geometry.within(SD.load_utm(z))
        print(f"  in {z:32s} {int(inside.sum()):5d}")

    with rasterio.open(EMODNET) as src:
        coords = list(zip(d.geometry.x.values, d.geometry.y.values))
        d["emodnet_elev_m"] = [v[0] for v in src.sample(coords, indexes=[1])]
        d["emodnet_support_class"] = [v[0] for v in src.sample(coords, indexes=[3])]

    has_e = np.isfinite(d.emodnet_elev_m)
    print(f"\n  {int(has_e.sum())}/{len(d)} ({100*has_e.mean():.1f}%) points fall "
          f"where EMODnet has any value -- it is an estuary/liman-only "
          f"interpolation and most points sit upriver of its footprint")

    sub = d[has_e & ~d.implausible].copy()
    sub["chart_elev_m"] = -sub.depth_m       # water depth -> elevation sign only
    sub["diff_m"] = sub.emodnet_elev_m - sub.chart_elev_m   # + = EMODnet shallower
    bias = float(sub.diff_m.median())
    resid = sub.diff_m - bias
    r = float(sub.emodnet_elev_m.corr(sub.chart_elev_m))

    print("\n" + "=" * 78)
    print(f"COMPARISON  (n={len(sub)}, implausible points excluded, "
          f"DATUM NOT RECONCILED)")
    print("=" * 78)
    print(f"  Pearson r                    {r:.3f}")
    print(f"  raw diff   median {sub.diff_m.median():6.2f} m   "
          f"RMSE {np.sqrt((sub.diff_m**2).mean()):5.2f} m")
    print(f"  bias-removed residual        RMSE {np.sqrt((resid**2).mean()):5.2f} m"
          f"   MAD {resid.abs().median():5.2f} m")
    print(f"  ('raw diff' folds in an unknown chart-datum offset; only the "
          f"bias-removed line is a shape-agreement statement)")
    print(f"\n  EMODnet support_class at matched points: "
          f"{dict(sub.emodnet_support_class.value_counts())}  "
          f"(1 = EMODNET_SUPPORT_LOW = 100% interpolated, per P11)")

    out = d.drop(columns="geometry").copy()
    out["x"] = d.geometry.x
    out["y"] = d.geometry.y
    out_path = CFG.TABLES / "p16_manual_depth_vs_emodnet.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    print("\n  NEXT: read the chart datum off the 9 sheet legends before treating "
          f"any bias here as physical; check the {n_imp} implausible points in "
          "QGIS -- likely mis-clicks or a separate, unintended batch.")


if __name__ == "__main__":
    main()
