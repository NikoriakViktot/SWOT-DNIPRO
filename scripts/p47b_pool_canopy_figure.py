#!/usr/bin/env python
"""P47b -- figure for the ICESat-2 canopy of the former pool (p47 products; dry-at-pass filtered, cnf low, 30 m).

Panels: (a) h_canopy per leaf-on season of the canopy_ok segments (box = IQR, whiskers 5-95 %), n per year;
(b) share of QC'd segments with canopy >= 2 m / >= 4 m, leaf-on vs all-season; (c,d) 250 m overgrowth-share maps
2025 / 2026 (share of segments with canopy >= 2 m per cell, on-track sample only).
Reads: outputs/tables/p47_pool_canopy_by_year_<tag>.csv, the GPKG of segments on F, the 250 m rasters.
Writes: outputs/figures/p47_pool_canopy_icesat2.png
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
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG

TAG = "icesat2_lowcnf"
ATL = CFG.BULK_ROOT / "data_swot/processed/atl08"
RO = ROOT / "outputs/rasters/roughness/pool"


def main():
    lo = pd.read_csv(CFG.TABLES / f"p47_pool_canopy_by_year_{TAG}.csv"); lo = lo[lo.recession_zone == "ALL"].set_index("year")
    al = pd.read_csv(CFG.TABLES / f"p47_pool_canopy_by_year_{TAG}_allseason.csv"); al = al[al.recession_zone == "ALL"].set_index("year")
    years = sorted(lo.index)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]; data = []
    for y in years:
        g = gpd.read_file(ATL / f"pool_canopy_segments_{TAG}.gpkg", layer=f"leafon_{y}", columns=["h_canopy", "canopy_ok"])
        data.append(g[g.canopy_ok == 1].h_canopy.values)
    ax.boxplot(data, tick_labels=[f"{y}\nn={len(d)}" for y, d in zip(years, data)], whis=(5, 95), showfliers=False)
    ax.set_ylabel("h_canopy, m above ground (canopy_ok, leaf-on, dry at pass)"); ax.set_title("ICESat-2 (PhoREAL, cnf low, 30 m) — former pool, bed dry at the pass"); ax.grid(alpha=0.3)
    ax = axes[0, 1]; x = np.arange(len(years)); w = 0.2
    ax.bar(x - 1.5 * w, 100 * lo.loc[years, "share_canopy_ge2m"], w, label="≥ 2 m, leaf-on"); ax.bar(x - 0.5 * w, 100 * lo.loc[years, "share_canopy_ge4m"], w, label="≥ 4 m, leaf-on")
    ax.bar(x + 0.5 * w, 100 * al.loc[years, "share_canopy_ge2m"], w, label="≥ 2 m, all seasons", alpha=0.5); ax.bar(x + 1.5 * w, 100 * al.loc[years, "share_canopy_ge4m"], w, label="≥ 4 m, all seasons", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{y}\npasses {int(lo.loc[y, 'n_passes_rgt_cycle'])}/{int(al.loc[y, 'n_passes_rgt_cycle'])}" for y in years]); ax.set_ylabel("share of QC'd segments, %"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("overgrowth: segments with canopy above threshold (2023 = only after the bed dried)")
    for ax, y in zip(axes[1], (2025, 2026)):
        f = RO / f"zone1_poolwin_{TAG}_overgrowth_share_{y}_250m.tif"
        if not f.exists():
            ax.set_visible(False); continue
        with rasterio.open(f) as ds:
            a = ds.read(1).astype("f4"); a[a == 255] = np.nan; ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
        with rasterio.open(RO / "zone1_poolwin_former_water_surface_mask.tif") as ds:
            fw = ds.read(1) == 1; ext2 = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
        ax.imshow(np.where(fw[::5, ::5], 0.9, np.nan), extent=ext2, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        im = ax.imshow(a, extent=ext, cmap="YlGn", vmin=0, vmax=60, interpolation="nearest"); plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="segments with canopy ≥ 2 m, %")
        ax.set_title(f"{y} leaf-on: overgrowth share per 250 m cell (on-track sample)"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); out = ROOT / "outputs/figures/p47_pool_canopy_icesat2.png"; fig.savefig(out, dpi=110); print("->", out)


if __name__ == "__main__":
    main()
