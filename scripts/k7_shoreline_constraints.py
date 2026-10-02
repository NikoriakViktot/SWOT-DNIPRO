#!/usr/bin/env python
"""K7 -- true per-date Sentinel validity + shoreline INEQUALITY constraints.

Uses the real Sen2Cor SCL layer stored alongside every built water mask. Cloud,
cloud shadow, cirrus, snow, saturated and no-data pixels stay UNKNOWN -- they
are never folded into DRY. The morphology prior's obs_count is NOT used as a
clear-sky proxy anywhere in this step.

Per date, on the 250 m analysis grid (co-registered with morphology_prior_v1):

    VALID_t             fraction of 20 m subpixels with usable SCL
    WET_t               fraction classified water
    DRY_t               valid and not water
    CONNECTED_WATER_t   water connected to the main along-channel network
    SHORELINE_t         valid wet cells adjacent to valid dry cells (and vice versa)

Constraints are emitted as one-sided BOUNDS, never as bed elevations:

    wet, hydraulically connected :  H_bed <= eta(chain,t) + tol
    valid dry adjacent to water  :  H_bed >= eta(chain,t) - tol

Only hydraulically connected water (class A) may use eta(chain,t) from the six
gauges. Disconnected residual bodies (class B) are emitted with
hydraulically_connected=False and eta=NaN -- they may sit at their own,
unknown level and must not inherit the gauge surface.

Outputs
-------
outputs/tables/k7_shoreline_constraints.csv
outputs/tables/k7_scene_validity_summary.csv
outputs/figures/K7_shoreline_constraints_examples.png
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW
from swot_dnipro.watermask import SCL_REJECT

MASK_DIR = CFG.ROOT / "data" / "processed" / "water_masks"
BATHY_DIR = CFG.ROOT / "data" / "processed" / "bathymetry"

#: a 250 m cell counts as unambiguously wet/dry only if the 20 m evidence agrees
F_VALID_MIN = 0.60
F_WET_HI = 0.80
F_WET_LO = 0.05
#: constraints are only emitted near the shoreline, where the bound is tight
SHORE_DILATE_CELLS = 2

#: tolerance budget (m). Mixed-pixel / classification term dominates on a flat
#: shore; it is a stated modelling choice, not a measurement.
TOL_CLASSIFICATION_M = 0.15
TOL_MIXED_PIXEL_M = 0.10

_TO_GEO = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True)


def scene_index() -> pd.DataFrame:
    rows = []
    for f in sorted(MASK_DIR.glob("*.npz")):
        m = re.search(r"_(\d{8})T(\d{6})_.*_T(\d{2}[A-Z]{3})_", f.stem)
        if m:
            rows.append({"npz": f, "date": pd.to_datetime(m.group(1)),
                         "tile": m.group(3), "scene": f.stem})
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("K7 -- SENTINEL VALID-OBSERVATION AND SHORELINE CONSTRAINTS")
    print("=" * 70)

    with rasterio.open(BATHY_DIR / "morphology_prior_v1.tif") as src:
        grid_tr, gH, gW, grid_crs = src.transform, src.height, src.width, src.crs
    print(f"analysis grid (shared with morphology_prior_v1): {gW}x{gH} @ 250 m, {grid_crs}")

    prof = pd.read_csv(CFG.TABLES / "k6_water_surface_profiles.csv", parse_dates=["date"])
    usable = prof[prof.profile_status == "OK"]
    dates_with_eta = sorted(usable.date.unique())
    print(f"dates with a usable eta(s) profile from K6: {len(dates_with_eta)}")

    sc = scene_index()
    sc = sc[sc.date.isin(dates_with_eta)]
    print(f"Sentinel scenes for those dates: {len(sc)} ({sc.tile.nunique()} tiles)")

    # chainage of every grid cell centre (computed once)
    cols, rows = np.meshgrid(np.arange(gW), np.arange(gH))
    xs, ys = rasterio.transform.xy(grid_tr, rows.ravel(), cols.ravel())
    lon, lat = _TO_GEO.transform(np.asarray(xs), np.asarray(ys))
    chan = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(chan)
    chain_km, off_km, _, _ = SW.assign_chainage(lon, lat, chan, tree=tree)
    chain_grid = chain_km.reshape(gH, gW)
    print("chain_km assigned to every grid cell (SWORD, Kakhovka-dam referenced, +upstream)")

    scene_rows, cons_rows, date_rows = [], [], []
    examples = {}
    for date, g in sc.groupby("date"):
        acc_valid = np.zeros((gH, gW), np.float32)
        acc_wet = np.zeros((gH, gW), np.float32)
        acc_cov = np.zeros((gH, gW), np.float32)
        for r in g.itertuples():
            z = np.load(r.npz, allow_pickle=True)
            scl = z["scl"]
            wet20 = z["mask"].astype(bool)
            valid20 = ~np.isin(scl, SCL_REJECT)
            aff = rasterio.Affine(*z["affine"][:6])
            src_crs = str(z["crs"])
            for arr, acc in ((valid20.astype(np.float32), acc_valid),
                             ((wet20 & valid20).astype(np.float32), acc_wet),
                             (np.ones_like(valid20, np.float32), acc_cov)):
                dst = np.zeros((gH, gW), np.float32)
                reproject(arr, dst, src_transform=aff, src_crs=src_crs,
                          dst_transform=grid_tr, dst_crs=grid_crs,
                          resampling=Resampling.average, src_nodata=None, dst_nodata=0.0)
                acc += dst
            n_valid_px = int(valid20.sum())
            scene_rows.append({
                "date": date, "tile": r.tile, "scene": r.scene,
                "n_pixels_20m": int(valid20.size),
                "frac_valid_scl": n_valid_px / valid20.size,
                "frac_water_of_valid": float(wet20[valid20].mean()) if n_valid_px else np.nan,
                "frac_cloud_or_invalid": float(np.isin(scl, SCL_REJECT).mean()),
            })
            del z, scl, wet20, valid20

        with np.errstate(invalid="ignore", divide="ignore"):
            f_valid = np.where(acc_cov > 0, acc_valid / np.maximum(acc_cov, 1e-6), 0.0)
            f_wet = np.where(acc_valid > 0, acc_wet / np.maximum(acc_valid, 1e-6), np.nan)

        observed = f_valid >= F_VALID_MIN
        wet = observed & (f_wet >= F_WET_HI)
        dry = observed & (f_wet <= F_WET_LO)
        unknown = ~observed

        # hydraulic connectivity: the component touching the along-channel corridor
        lab, n = ndimage.label(wet)
        corridor = (np.abs(off_km.reshape(gH, gW)) <= 3.0) & np.isfinite(chain_grid)
        main_ids = set(np.unique(lab[wet & corridor])) - {0}
        connected = np.isin(lab, list(main_ids)) if main_ids else np.zeros_like(wet)
        disconnected = wet & ~connected

        # shoreline: wet cells adjacent to dry, and dry cells adjacent to wet
        wet_d = ndimage.binary_dilation(wet, iterations=SHORE_DILATE_CELLS)
        dry_d = ndimage.binary_dilation(dry, iterations=SHORE_DILATE_CELLS)
        shore_wet = wet & dry_d
        shore_dry = dry & wet_d

        eta_of_s = usable[usable.date == date].sort_values("chain_km")
        eta_lut = np.interp(chain_grid.ravel(), eta_of_s.chain_km.values,
                            eta_of_s.eta_evrf2019_m.values).reshape(gH, gW)
        sig_row = eta_of_s.iloc[0]
        wse_sigma = float(np.sqrt(sig_row.sigma_gauge_m ** 2
                                  + sig_row.sigma_vertical_transform_m ** 2
                                  + (sig_row.sigma_temporal_matching_m if np.isfinite(sig_row.sigma_temporal_matching_m) else 0.0) ** 2
                                  + (sig_row.sigma_spatial_interpolation_m if np.isfinite(sig_row.sigma_spatial_interpolation_m) else 0.0) ** 2))
        tol = float(np.sqrt(wse_sigma ** 2 + TOL_CLASSIFICATION_M ** 2 + TOL_MIXED_PIXEL_M ** 2))

        for kind, sel in (("WET_UPPER_BOUND", shore_wet), ("DRY_LOWER_BOUND", shore_dry)):
            rr, cc = np.where(sel)
            if not len(rr):
                continue
            x, y = rasterio.transform.xy(grid_tr, rr, cc)
            conn = connected[rr, cc] if kind == "WET_UPPER_BOUND" else np.ones(len(rr), bool)
            eta_v = eta_lut[rr, cc]
            cons_rows.append(pd.DataFrame({
                "date": date, "x": x, "y": y,
                "chain_km": chain_grid[rr, cc],
                "constraint_type": kind,
                "lower_bound_m": (eta_v - tol) if kind == "DRY_LOWER_BOUND" else np.nan,
                "upper_bound_m": (eta_v + tol) if kind == "WET_UPPER_BOUND" else np.nan,
                "eta_evrf2019_m": np.where(conn, eta_v, np.nan),
                "wse_sigma_m": wse_sigma, "tolerance_m": tol,
                "sentinel_valid": True,
                "frac_valid": f_valid[rr, cc], "frac_wet": f_wet[rr, cc],
                "hydraulically_connected": conn,
                "source_scene": ",".join(sorted(g.tile)),
                "QC_flag": np.where(conn, "ok", "DISCONNECTED_water_body_level_unknown"),
            }))
            if kind == "WET_UPPER_BOUND":
                examples[date] = (wet, dry, unknown, connected, disconnected, shore_wet, shore_dry)

        n_cells = observed.size
        n_valid = int(observed.sum())
        date_rows.append({
            "date": date, "scope": "250m_analysis_grid", "n_cells": n_cells,
            "valid_fraction": n_valid / n_cells,
            "unknown_fraction": float(unknown.mean()),
            "wet_fraction_of_valid": wet.sum() / n_valid if n_valid else np.nan,
            "dry_fraction_of_valid": dry.sum() / n_valid if n_valid else np.nan,
            "ambiguous_fraction_of_valid": (n_valid - int(wet.sum()) - int(dry.sum())) / n_valid if n_valid else np.nan,
            "connected_water_fraction_of_wet": connected.sum() / wet.sum() if wet.sum() else np.nan,
            "disconnected_water_fraction_of_wet": disconnected.sum() / wet.sum() if wet.sum() else np.nan,
            "n_shore_wet": int(shore_wet.sum()), "n_shore_dry": int(shore_dry.sum()),
            "eta_median_m": float(np.nanmedian(eta_lut[wet])) if wet.any() else np.nan,
            "tolerance_m": tol, "wse_sigma_m": wse_sigma,
        })
        print(f"  {pd.Timestamp(date).date()}: observed {observed.mean()*100:5.1f}% of grid | "
              f"wet {wet.sum():6d} | dry {dry.sum():6d} | unknown {unknown.sum():6d} | "
              f"connected {connected.sum():6d} | disconnected {disconnected.sum():5d} | "
              f"shore wet/dry {shore_wet.sum():5d}/{shore_dry.sum():5d} | tol {tol:.3f} m")

    cons = pd.concat(cons_rows, ignore_index=True)
    cons.to_csv(CFG.TABLES / "k7_shoreline_constraints.csv", index=False)
    sv = pd.DataFrame(scene_rows)
    sv.to_csv(CFG.TABLES / "k7_scene_validity_summary.csv", index=False)
    dr = pd.DataFrame(date_rows)
    dr.to_csv(CFG.TABLES / "k7_per_date_validity_fractions.csv", index=False)
    print(f"-> {CFG.TABLES / 'k7_per_date_validity_fractions.csv'}")
    print("\nper-date fractions on the 250 m grid:")
    print(dr[["date", "valid_fraction", "unknown_fraction", "wet_fraction_of_valid",
              "dry_fraction_of_valid", "connected_water_fraction_of_wet",
              "disconnected_water_fraction_of_wet", "eta_median_m"]].to_string(
              index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {CFG.TABLES / 'k7_shoreline_constraints.csv'}  ({len(cons)} constraints)")
    print(f"-> {CFG.TABLES / 'k7_scene_validity_summary.csv'}")

    print("\nconstraint inventory:")
    print(cons.groupby(["constraint_type", "hydraulically_connected"]).size().to_string())
    print(f"\ndisconnected water constraints (eta deliberately NaN): "
          f"{int((~cons.hydraulically_connected).sum())}")
    print(f"scene validity: frac_valid_scl {sv.frac_valid_scl.min():.3f} .. {sv.frac_valid_scl.max():.3f} "
          f"(median {sv.frac_valid_scl.median():.3f})")

    # ---- figure ---------------------------------------------------------------
    show = sorted(examples)[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 6))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, show):
        wet, dry, unknown, connected, disconnected, shore_wet, shore_dry = examples[d]
        img = np.zeros(wet.shape + (3,), np.float32)
        img[dry] = (0.88, 0.85, 0.78)
        img[connected] = (0.14, 0.44, 0.55)
        img[disconnected] = (0.80, 0.55, 0.15)
        img[unknown] = (0.65, 0.65, 0.68)
        img[shore_dry] = (0.95, 0.75, 0.20)
        img[shore_wet] = (0.76, 0.25, 0.16)
        ax.imshow(img)
        ax.set_title(f"{pd.Timestamp(d).date()}\nred=wet shore (upper bound), "
                     f"amber=dry shore (lower bound)\ngrey=UNKNOWN (cloud/invalid), never DRY", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("K7 -- per-date validity, connectivity and shoreline inequality constraints", fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K7_shoreline_constraints_examples.png", dpi=150)
    print(f"-> {CFG.FIG / 'K7_shoreline_constraints_examples.png'}")


if __name__ == "__main__":
    main()
