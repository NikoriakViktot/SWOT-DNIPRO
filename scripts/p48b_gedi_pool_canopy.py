#!/usr/bin/env python
"""P48b -- GEDI L2A canopy on the drained Kakhovka bed, processed like the ICESat-2 product (p47) so the two can be compared.

Input: $BULK_ROOT/gedi/gedi02a_pool_<year>.parquet (p48: all shots in the pool bbox, 8 beams, rh25..rh100, quality/degrade/sensitivity).
Steps
  1. footprint on the ZONE_1 pool-window grid (p43 former_water_surface_mask, 20 m) -> keep shots on the former water surface only
  2. quality: quality_flag == 1, degrade_flag == 0, sensitivity >= 0.9 (L2A guidance), power beams only for heights (coverage beams kept in counts)
  3. dry at pass (as p47): 2023 not available (GEDI off Mar 2023 - Apr 2024); 2024+ -> cell's leaf-on water_share <= 50 %
  4. season: leaf-on Jun-Sep vs all; night shots flagged (solar_elevation < 0) -- L2A height quality is better at night
  5. woody threshold: rh98 >= 4 m (GEDI's ground-return width puts bare-ground rh98 at ~1-3 m; the ICESat-2 2 m threshold does NOT transfer)
Outputs
  outputs/tables/p48b_gedi_pool_canopy_by_year.csv     per year x season x recession zone: n shots, n quality, share rh98 >= 4 m / >= 2 m, rh98 p25/p50/p75/p95
  outputs/tables/p48b_gedi_vs_icesat2.csv              side-by-side with p47 (leaf-on, dry at pass): share >= 4 m and canopy p50 per year
  outputs/rasters/roughness/pool/zone1_poolwin_gedi_{rh98_mean,woody_share,n_shots}_<year>_250m.tif
  outputs/figures/p48b_gedi_pool.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG

GEDI = CFG.BULK_ROOT / "gedi"; RO = ROOT / "outputs/rasters/roughness/pool"; CELL = 250.0
WOODY_M = 4.0


def main():
    files = sorted(GEDI.glob("gedi02a_pool_*.parquet"))
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); D["time"] = pd.to_datetime(D.time); D["year"] = D.time.dt.year; D["month"] = D.time.dt.month
    tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); D["x"], D["y"] = tf.transform(D.lon.values, D.lat.values)
    with rasterio.open(RO / "zone1_poolwin_former_water_surface_mask.tif") as ds:
        fw = ds.read(1) == 1; tr = ds.transform; shape = fw.shape
    with rasterio.open(RO / "zone1_poolwin_bed_recession_zoning_2023.tif") as ds:
        zoning = ds.read(1)
    c, r = ~tr * (D.x.values, D.y.values); c = np.floor(c).astype(int); r = np.floor(r).astype(int)
    inb = (c >= 0) & (c < shape[1]) & (r >= 0) & (r < shape[0]); D = D[inb].copy(); c, r = c[inb], r[inb]
    D["in_former_water"] = fw[r, c]; D["recession_zone"] = zoning[r, c]
    print(f"shots in bbox {len(D):,}; on the former water surface {int(D.in_former_water.sum()):,}")
    D = D[D.in_former_water].copy()
    D["quality"] = (D.quality_flag == 1) & (D.degrade_flag == 0) & (D.sensitivity >= 0.9)
    D["night"] = D.solar_elevation < 0; D["leaf_on"] = D.month.between(6, 9)
    # dry at pass: year's leaf-on water share on the ZONE_1 annual grid
    D["dry"] = True
    for y in sorted(D.year.unique()):
        f = ROOT / "outputs/rasters/zone1/annual" / f"zone1_water_share_{y}_leafon_20m.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as ds:
            ws = ds.read(1); wtr = ds.transform
        m = (D.year == y).values; c2, r2 = ~wtr * (D.x.values[m], D.y.values[m]); c2 = np.floor(c2).astype(int); r2 = np.floor(r2).astype(int)
        ok = (c2 >= 0) & (c2 < ws.shape[1]) & (r2 >= 0) & (r2 < ws.shape[0]); v = np.zeros(m.sum(), "u1"); v[ok] = ws[r2[ok], c2[ok]]
        dry = ~((v != 255) & (v > 50)); idx = np.flatnonzero(m); D.loc[D.index[idx[~dry]], "dry"] = False
    print(f"dry-at-pass kept {int(D.dry.sum()):,} of {len(D):,}")
    Q = D[D.quality & D.dry].copy(); ZL = {1: "fast", 2: "mid", 3: "slow", 4: "never", 0: "n/a"}
    # ground-noise floor calibrated on the data: rh98 of quality power-beam shots that fall on cells classed bare/wet sediment in the
    # same year's p43 state map (codes 3 wet_sediment, 4 bare_sand_silt, 13-15 bed sediment). The woody threshold = p95 of that floor.
    global WOODY_M
    floor_rows = []
    for y in sorted(Q.year.unique()):
        st = {2024: "STATE_2024", 2025: "STATE_2025", 2026: "CURRENT_2026"}.get(int(y))
        f = RO / f"zone1_poolwin_manning_classes_{st}.tif" if st else None
        if f is None or not f.exists():
            continue
        with rasterio.open(f) as ds:
            cls = ds.read(1); ctr = ds.transform
        m = (Q.year == y).values & Q.power_beam.values; cc, rr = ~ctr * (Q.x.values[m], Q.y.values[m]); cc = np.floor(cc).astype(int); rr = np.floor(rr).astype(int)
        ok = (cc >= 0) & (cc < cls.shape[1]) & (rr >= 0) & (rr < cls.shape[0]); k = np.zeros(m.sum(), "u1"); k[ok] = cls[rr[ok], cc[ok]]
        bare = np.isin(k, (3, 4, 13, 14, 15)); v = Q.rh98.values[m][bare]
        if len(v) >= 200:
            floor_rows.append(dict(year=int(y), n_bare_shots=len(v), rh98_bare_p50=round(float(np.median(v)), 2), rh98_bare_p90=round(float(np.percentile(v, 90)), 2), rh98_bare_p95=round(float(np.percentile(v, 95)), 2), rh98_bare_p99=round(float(np.percentile(v, 99)), 2)))
    F = pd.DataFrame(floor_rows); F.to_csv(CFG.TABLES / "p48b_gedi_ground_floor.csv", index=False); print("ground floor (bare cells):"); print(F.to_string(index=False))
    if len(F):
        WOODY_M = float(np.ceil(F.rh98_bare_p95.max() * 2) / 2)          # rounded up to 0.5 m
        print(f"woody threshold set from the bare-ground floor: rh98 >= {WOODY_M} m")
    rows = []
    for (y, lo), s in Q.groupby(["year", "leaf_on"]):
        for zn, ss in list(s.groupby("recession_zone")) + [("ALL", s)]:
            pw = ss[ss.power_beam]
            rows.append(dict(year=int(y), season="leafon" if lo else "leafoff", recession_zone=ZL.get(zn, zn) if zn != "ALL" else "ALL", n_shots_all=int(len(D[(D.year == y) & (D.leaf_on == lo)])), n_quality_dry=len(ss), n_power=len(pw), night_share=round(float(ss.night.mean()), 3),
                             share_rh98_ge4m=round(float((pw.rh98 >= WOODY_M).mean()), 4) if len(pw) else np.nan, share_rh98_ge2m=round(float((pw.rh98 >= 2).mean()), 4) if len(pw) else np.nan,
                             rh98_p25=round(float(pw.rh98.quantile(.25)), 2) if len(pw) else np.nan, rh98_p50=round(float(pw.rh98.median()), 2) if len(pw) else np.nan, rh98_p75=round(float(pw.rh98.quantile(.75)), 2) if len(pw) else np.nan, rh98_p95=round(float(pw.rh98.quantile(.95)), 2) if len(pw) else np.nan,
                             rh98_p50_woody=round(float(pw.rh98[pw.rh98 >= WOODY_M].median()), 2) if (pw.rh98 >= WOODY_M).any() else np.nan))
    T = pd.DataFrame(rows).sort_values(["year", "season", "recession_zone"]); T.to_csv(CFG.TABLES / "p48b_gedi_pool_canopy_by_year.csv", index=False)
    print(T[T.recession_zone == "ALL"].to_string(index=False))
    # side-by-side with ICESat-2 (p47, leaf-on, dry at pass)
    I = pd.read_csv(CFG.TABLES / "p47_pool_canopy_by_year_icesat2_lowcnf.csv"); I = I[I.recession_zone == "ALL"].set_index("year")
    G = T[(T.recession_zone == "ALL") & (T.season == "leafon")].set_index("year")
    cmp = pd.DataFrame(dict(gedi_n_power=G.n_power, gedi_share_rh98_ge4m=G.share_rh98_ge4m, gedi_share_rh98_ge2m=G.share_rh98_ge2m, gedi_rh98_p50=G.rh98_p50, gedi_rh98_p50_woody=G.rh98_p50_woody,
                            icesat2_n_segments=I.n_segments, icesat2_share_canopy_ge4m=I.share_canopy_ge4m, icesat2_share_canopy_ge2m=I.share_canopy_ge2m, icesat2_h_canopy_p50=I.h_canopy_p50)).dropna(how="all")
    cmp.to_csv(CFG.TABLES / "p48b_gedi_vs_icesat2.csv"); print(cmp.to_string())
    # 250 m rasters
    x0, y1 = tr.c, tr.f; ny, nx = int(np.ceil(shape[0] * 20 / CELL)), int(np.ceil(shape[1] * 20 / CELL)); tr250 = from_origin(x0, y1, CELL, CELL)
    P = Q[Q.power_beam & Q.leaf_on]; cx = ((P.x.values - x0) // CELL).astype(int); cy = ((y1 - P.y.values) // CELL).astype(int)
    for y in sorted(P.year.unique()):
        m = (P.year.values == y) & (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        n = np.zeros((ny, nx), "i4"); hs = np.zeros((ny, nx), "f8"); wd = np.zeros((ny, nx), "i4")
        np.add.at(n, (cy[m], cx[m]), 1); np.add.at(hs, (cy[m], cx[m]), P.rh98.values[m]); np.add.at(wd, (cy[m], cx[m]), (P.rh98.values[m] >= WOODY_M).astype(int))
        for name, arr, dtype, nd, q in ((f"gedi_rh98_mean_{y}_250m", np.where(n > 0, hs / np.maximum(n, 1), -9999).astype("f4"), "float32", -9999, "mean rh98 [m] of quality power-beam shots, leaf-on, bed dry"),
                                        (f"gedi_woody_share_{y}_250m", np.where(n > 0, np.round(100 * wd / np.maximum(n, 1)), 255).astype("u1"), "uint8", 255, f"share of shots with rh98 >= {WOODY_M} m, % (255 no shot)"),
                                        (f"gedi_n_shots_{y}_250m", np.clip(n, 0, 65535).astype("u2"), "uint16", 0, "quality power-beam shots per cell")):
            with rasterio.open(RO / f"zone1_poolwin_{name}.tif", "w", driver="GTiff", height=ny, width=nx, count=1, dtype=dtype, crs=CFG.CRS_METRIC, transform=tr250, nodata=nd, compress="deflate") as ds:
                ds.write(arr, 1); ds.update_tags(quantity=q, year=str(y), source="GEDI L2A v002 (p48)", producer="p48b_gedi_pool_canopy.py")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]; years = sorted(P.year.unique()); data = [P[P.year == y].rh98.values for y in years]
    ax.boxplot(data, tick_labels=[f"{y}\nn={len(d):,}" for y, d in zip(years, data)], whis=(5, 95), showfliers=False); ax.axhline(WOODY_M, color="red", ls="--", lw=0.8); ax.set_ylabel("GEDI rh98, m (power beams, leaf-on, quality, bed dry)"); ax.set_title("GEDI L2A rh98 on the former pool"); ax.grid(alpha=0.3)
    ax = axes[1]; x = np.arange(len(cmp.index)); w = 0.35
    ax.bar(x - w / 2, 100 * cmp.gedi_share_rh98_ge4m, w, label="GEDI: rh98 ≥ 4 m"); ax.bar(x + w / 2, 100 * cmp.icesat2_share_canopy_ge4m, w, label="ICESat-2: h_canopy ≥ 4 m")
    ax.set_xticks(x); ax.set_xticklabels(cmp.index); ax.set_ylabel("share of footprints/segments, %"); ax.set_title("Woody share: GEDI vs ICESat-2 (same threshold 4 m)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax = axes[2]; y = years[-1]
    with rasterio.open(RO / f"zone1_poolwin_gedi_woody_share_{y}_250m.tif") as ds:
        a = ds.read(1).astype("f4"); a[a == 255] = np.nan; ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
    ax.imshow(np.where(fw[::5, ::5], 0.9, np.nan), extent=[tr.c, tr.c + shape[1] * 20, tr.f - shape[0] * 20, tr.f], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(a, extent=ext, cmap="YlGn", vmin=0, vmax=60, interpolation="nearest"); plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="shots with rh98 ≥ 4 m, %"); ax.set_title(f"{y} leaf-on: GEDI woody share per 250 m"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p48b_gedi_pool.png", dpi=100); print("-> outputs/figures/p48b_gedi_pool.png")


if __name__ == "__main__":
    main()
