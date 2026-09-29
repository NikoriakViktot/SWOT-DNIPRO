#!/usr/bin/env python
"""P68 -- are the source-1 near-shore residuals a DEM defect, or point-to-raster representativeness at an abrupt land-water edge?

The question p67 left open. p65 showed p53's residual against night ICESat-2 keys on ground elevation (Spearman -0.874) with a
100 % tail above 5 m of ground; p67 then showed the tail is NOT in the part of the old domain the new one excludes -- 90.7 % of
the tail sits inside the water polygon that both domains agree on. So "the domain let the bed onto land" describes why the old
domain was bad, but does not by itself account for the catastrophic residuals.

The remaining mechanism is representativeness, and it is well supported in the literature (registered SRC-39..44):
  * ICESat-2 ground elevation error rises sharply with slope, and part of that is horizontal geolocation converting into
    vertical error (Wang et al. 2019, 10.1364/OE.27.038168).
  * Post-calibration horizontal geolocation uncertainty is ~2.5-4.4 m per beam with a ~12 m footprint (Luthcke et al. 2021).
  * Intra-pixel relief and raster resolution amplify ICESat-2-to-DEM disagreement, which is exactly a 30 m bed cell spanning an
    abrupt land-water transition (Chen et al. 2022).
  * The land-water boundary is recognised as a special hard zone for ICESat-2 inland water (SRC-44).
A 30 m p53 cell may legitimately hold water at ~0 m while a 20 m ICESat-2 segment a few metres away legitimately holds the bank
at +10..30 m. Neither is wrong; they describe different parts of one sharp step, and the difference is then not a DEM error.

What this script measures, per source-1 night point:
    signed_distance_to_shore_m   + inside the pre-breach water polygon, - outside
    fabdem_slope_deg, fabdem_aspect_deg
    relief_r30_m, relief_r60_m   max-min of FABDEM within 30 m / 60 m
    water_frac_r30, water_frac_r60
    p53_nearest, p53_bilinear    the bed sampled both ways, and their difference
    dist_to_cell_centre_m        where in its 30 m cell the point falls
Stratification: distance to shore (0-30, 30-60, 60-90, 90-150, >150 m) CROSSED with slope (0-2, 2-5, 5-10, 10-20, >20 deg),
reporting n, bias, RMSE, LE90 and the share below -5 m in every cell of the table.
Nuth & Kaeaeb diagnostic (SRC-43): dh / tan(slope) against aspect. A systematic horizontal mismatch between the validation
geometry and the raster shows up as a cosine in aspect. This is a TEST ONLY -- nothing is co-registered or shifted here.
Outputs: outputs/tables/p68_shoreline_strata.csv, p68_point_attributes.parquet, p68_nuth_kaab.csv;
         outputs/figures/p68_shoreline_representativeness.png
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

spec = importlib.util.spec_from_file_location("p53", ROOT / "scripts/p53_below_dam_bed_dem.py")
P53 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P53)
spec2 = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(P63)
P57 = P63.P57
ZONES = {"ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
TAIL_M = -5.0
DIST_BINS = [-1e9, 0, 30, 60, 90, 150, 1e9]
DIST_LAB = ["outside water", "0-30 m", "30-60 m", "60-90 m", "90-150 m", ">150 m"]
SLOPE_BINS = [0, 2, 5, 10, 20, 1e9]
SLOPE_LAB = ["0-2 deg", "2-5 deg", "5-10 deg", "10-20 deg", ">20 deg"]


def stat(r, **extra):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) == 0:
        return dict(n=0, **extra)
    return dict(n=len(r), bias=round(float(r.mean()), 2), median=round(float(np.median(r)), 2),
                RMSE=round(float(np.sqrt((r ** 2).mean())), 2), LE90=round(float(np.percentile(np.abs(r), 90)), 2),
                share_below_5m=round(float((r < TAIL_M).mean()), 4), **extra)


def sample_at(path, xs, ys, resampling=Resampling.nearest):
    with rasterio.open(path) as ds:
        if resampling == Resampling.nearest:
            out = np.array([v[0] for v in ds.sample(np.c_[xs, ys], masked=False)], "f8")
            if ds.nodata is not None:
                out[out == ds.nodata] = np.nan
            return out
        # bilinear: read a small window per point is slow; do it via reproject onto the point grid using a 1-px dst
        a = ds.read(1).astype("f4")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        inv = ~ds.transform
        c, r = inv * (xs, ys)
        c = np.asarray(c) - 0.5; r = np.asarray(r) - 0.5
        c0 = np.floor(c).astype(int); r0 = np.floor(r).astype(int); fc = c - c0; fr = r - r0
        out = np.full(len(xs), np.nan)
        ok = (r0 >= 0) & (r0 < a.shape[0] - 1) & (c0 >= 0) & (c0 < a.shape[1] - 1)
        if ok.any():
            v00 = a[r0[ok], c0[ok]]; v01 = a[r0[ok], c0[ok] + 1]; v10 = a[r0[ok] + 1, c0[ok]]; v11 = a[r0[ok] + 1, c0[ok] + 1]
            out[ok] = ((1 - fr[ok]) * ((1 - fc[ok]) * v00 + fc[ok] * v01) + fr[ok] * ((1 - fc[ok]) * v10 + fc[ok] * v11))
        return out


def main():
    V = P63.build_set_C(); S1 = V[V.src == 1].copy(); S1["res"] = S1.dem - S1.H_ice
    print(f"source-1 night points: {len(S1):,}; tail (< {TAIL_M} m) {int((S1.res < TAIL_M).sum())} = {(S1.res < TAIL_M).mean():.1%}")
    parts = []
    for zone, n in ZONES.items():
        g = S1[S1.zone == zone].copy()
        if not len(g):
            continue
        fp = CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif"
        with rasterio.open(fp) as ds:
            fab = ds.read(1).astype("f4"); fab[fab == ds.nodata] = np.nan
            tr, crs, shape = ds.transform, ds.crs, ds.shape; cell = abs(ds.transform.a)
        # slope / aspect from FABDEM
        gy, gx = np.gradient(np.nan_to_num(fab, nan=np.nanmedian(fab)), cell, cell)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = (np.degrees(np.arctan2(-gx, gy)) + 360.0) % 360.0
        # intra-pixel relief at 30 m and 60 m radius (20 m cells -> sizes 3 and 7)
        f0 = np.nan_to_num(fab, nan=np.nanmedian(fab))
        rel30 = ndimage.maximum_filter(f0, size=3) - ndimage.minimum_filter(f0, size=3)
        rel60 = ndimage.maximum_filter(f0, size=7) - ndimage.minimum_filter(f0, size=7)
        # the pre-breach water polygon p53 actually used, on the FABDEM grid
        G = dict(nx=None)
        gg = SD.build_grid(SD.load_utm(zone), P53.CELL, what=f"p53 grid {zone}")
        gxs, gys = gg["gx"], gg["gy"]
        G = dict(nx=len(gxs), ny=len(gys), x0=float(gxs.min() - P53.CELL / 2), y1=float(gys.max() + P53.CELL / 2))
        G["transform"] = rasterio.transform.from_origin(G["x0"], G["y1"], P53.CELL, P53.CELL)
        wat30, _ = P53.water_polygon_mask(zone, n, G)
        water = np.zeros(shape, "u1")
        reproject(source=wat30.astype("u1"), src_transform=G["transform"], src_crs=CFG.CRS_METRIC,
                  destination=water, dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
        water = water.astype(bool)
        d_in = ndimage.distance_transform_edt(water) * cell
        d_out = ndimage.distance_transform_edt(~water) * cell
        signed = np.where(water, d_in, -d_out)
        wf30 = ndimage.uniform_filter(water.astype("f4"), size=3)
        wf60 = ndimage.uniform_filter(water.astype("f4"), size=7)
        inv = ~tr; c, r = inv * (g.x.values, g.y.values)
        c = np.floor(c).astype(int); r = np.floor(r).astype(int)
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        def pick(arr):
            v = np.full(len(g), np.nan); v[ok] = arr[r[ok], c[ok]]; return v
        g["signed_distance_to_shore_m"] = pick(signed); g["fabdem_slope_deg"] = pick(slope); g["fabdem_aspect_deg"] = pick(aspect)
        g["relief_r30_m"] = pick(rel30); g["relief_r60_m"] = pick(rel60)
        g["water_frac_r30"] = pick(wf30); g["water_frac_r60"] = pick(wf60)
        bp = ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_PRE_BREACH_30m.tif"
        if bp.exists():
            g["p53_nearest"] = sample_at(bp, g.x.values, g.y.values, Resampling.nearest)
            g["p53_bilinear"] = sample_at(bp, g.x.values, g.y.values, Resampling.bilinear)
            with rasterio.open(bp) as ds:
                ic, ir = (~ds.transform) * (g.x.values, g.y.values)
                cx = (np.floor(ic) + 0.5); cy = (np.floor(ir) + 0.5)
                px_, py_ = ds.transform * (cx, cy)
                g["dist_to_cell_centre_m"] = np.hypot(g.x.values - px_, g.y.values - py_)
        parts.append(g)
    A = pd.concat(parts, ignore_index=True)
    A["is_tail"] = A.res < TAIL_M
    A["dist_bin"] = pd.cut(A.signed_distance_to_shore_m, DIST_BINS, labels=DIST_LAB)
    A["slope_bin"] = pd.cut(A.fabdem_slope_deg, SLOPE_BINS, labels=SLOPE_LAB)
    A.to_parquet(CFG.TABLES / "p68_point_attributes.parquet", index=False)

    rows = [stat(A.res, stratum="ALL source-1 points", axis="reference")]
    for k, gg in A.groupby("dist_bin", observed=True):
        rows.append(stat(gg.res, stratum=str(k), axis="distance to shore"))
    for k, gg in A.groupby("slope_bin", observed=True):
        rows.append(stat(gg.res, stratum=str(k), axis="FABDEM slope"))
    for (d, s), gg in A.groupby(["dist_bin", "slope_bin"], observed=True):
        rows.append(stat(gg.res, stratum=f"{d} x {s}", axis="distance x slope"))
    for col, bins in (("relief_r30_m", [0, 1, 3, 6, 12, 1e9]), ("water_frac_r30", [-0.01, 0.2, 0.5, 0.8, 1.01])):
        A["_b"] = pd.cut(A[col], bins)
        for k, gg in A.groupby("_b", observed=True):
            rows.append(stat(gg.res, stratum=f"{col} {k}", axis=col))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p68_shoreline_strata.csv", index=False)
    print("\nDISTANCE TO SHORE"); print(T[T.axis == "distance to shore"][["stratum", "n", "bias", "RMSE", "LE90", "share_below_5m"]].to_string(index=False))
    print("\nFABDEM SLOPE"); print(T[T.axis == "FABDEM slope"][["stratum", "n", "bias", "RMSE", "LE90", "share_below_5m"]].to_string(index=False))
    piv = A.pivot_table(index="dist_bin", columns="slope_bin", values="res", aggfunc=lambda v: round(float((v < TAIL_M).mean()), 3), observed=True)
    pn = A.pivot_table(index="dist_bin", columns="slope_bin", values="res", aggfunc="size", observed=True)
    print("\nSHARE BELOW -5 m, distance x slope"); print(piv.to_string())
    print("\nn, distance x slope"); print(pn.to_string())

    # ---------------- Nuth & Kaeaeb diagnostic (TEST ONLY, nothing is shifted)
    m = (A.fabdem_slope_deg > 2) & np.isfinite(A.res) & np.isfinite(A.fabdem_aspect_deg)
    nk = A[m].copy(); nk["y"] = nk.res / np.tan(np.radians(nk.fabdem_slope_deg))
    nk = nk[np.abs(nk.y) < 200]
    rows = []
    if len(nk) > 50:
        th = np.radians(nk.fabdem_aspect_deg.values); yv = nk.y.values
        Amat = np.c_[np.cos(th), np.sin(th), np.ones(len(th))]
        coef, *_ = np.linalg.lstsq(Amat, yv, rcond=None)
        shift = float(np.hypot(coef[0], coef[1])); direction = float((np.degrees(np.arctan2(coef[1], coef[0]))) % 360)
        rows.append(dict(n=len(nk), horizontal_shift_m=round(shift, 2), shift_direction_deg=round(direction, 1),
                         vertical_term_m=round(float(coef[2]), 2),
                         note="Nuth & Kaeaeb (SRC-43) cosine fit of dh/tan(slope) vs aspect; TEST ONLY, no co-registration applied"))
        print(f"\nNUTH & KAAB: apparent horizontal shift {shift:.2f} m toward {direction:.0f} deg, vertical term {coef[2]:+.2f} m (n {len(nk):,})")
        print(f"  for scale: ICESat-2 post-calibration horizontal uncertainty is ~2.5-4.4 m with a ~12 m footprint (SRC-40)")
    pd.DataFrame(rows).to_csv(CFG.TABLES / "p68_nuth_kaab.csv", index=False)

    # ---------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(19, 12))
    ax = axes[0, 0]
    for lab, gg in A.groupby("dist_bin", observed=True):
        if len(gg) >= 20:
            ax.scatter(gg.fabdem_slope_deg, gg.res, s=5, alpha=0.4, label=f"{lab} (n {len(gg)})")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(TAIL_M, color="grey", ls=":", lw=1); ax.set_ylim(-45, 10)
    ax.set_xlabel("FABDEM slope, deg"); ax.set_ylabel("p53 − ICESat-2, m"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("(a) residual against slope, by distance to the pre-breach shore")
    ax = axes[0, 1]
    if len(piv):
        im = ax.imshow(piv.values.astype(float), cmap="Reds", vmin=0, vmax=max(0.01, np.nanmax(piv.values.astype(float))), aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, fontsize=8)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]; nn = pn.values[i, j] if pn.shape == piv.shape else 0
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}\nn{int(nn)}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="share of points below −5 m")
    ax.set_title("(b) the tail concentrates where? distance × slope")
    ax = axes[1, 0]
    ax.scatter(A.signed_distance_to_shore_m, A.res, s=5, alpha=0.35, c=np.clip(A.fabdem_slope_deg, 0, 25), cmap="viridis")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(TAIL_M, color="grey", ls=":", lw=1); ax.set_xlim(-200, 600); ax.set_ylim(-45, 10)
    ax.set_xlabel("signed distance to the pre-breach shore, m (+ inside water)"); ax.set_ylabel("p53 − ICESat-2, m"); ax.grid(alpha=0.3)
    ax.set_title("(c) residual against distance to the land–water edge (colour = slope)")
    ax = axes[1, 1]
    if len(nk) > 50:
        ax.scatter(nk.fabdem_aspect_deg, np.clip(nk.y, -60, 60), s=4, alpha=0.3)
        t = np.linspace(0, 360, 361); ax.plot(t, coef[0] * np.cos(np.radians(t)) + coef[1] * np.sin(np.radians(t)) + coef[2], "r-", lw=2,
                                              label=f"fit: shift {shift:.1f} m @ {direction:.0f}°")
        ax.legend(fontsize=8)
    ax.set_xlabel("FABDEM aspect, deg"); ax.set_ylabel("Δh / tan(slope), m"); ax.grid(alpha=0.3)
    ax.set_title("(d) Nuth & Kääb horizontal-mismatch diagnostic (test only)")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p68_shoreline_representativeness.png", dpi=110)
    print("-> outputs/figures/p68_shoreline_representativeness.png")


if __name__ == "__main__":
    main()
