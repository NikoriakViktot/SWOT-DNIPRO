#!/usr/bin/env python
"""P46 -- index / class / SCL map atlas from the p40 annual composites (plan 15, user request 2026-09-19).

For every zone and year with a leaf-on composite: one PNG per index (NDVI NDWI MNDWI NDMI BSI AWEIsh NDTI),
one for the class mode, one for the SCL mode and one for the water share; plus one multi-year strip per index
(2017..2026, leaf-on) so the succession reads at a glance. Uses swot_dnipro.plotting.maps (project styling,
zone outline from the registry). Rasters are decimated for the figure (ZONE_1 is 96 Mpx) -- the PNG is a view,
the numbers live in the GeoTIFFs.
Outputs: outputs/figures/p40_maps/<zone>/<zone>_<var>_<year>_leafon.png and <zone>_<var>_years_leafon.png
Usage: python scripts/p46_index_maps.py --zones ZONE_1_KAKHOVKA_LOWER_DNIPRO ZONE_2_KHERSON_DELTA ZONE_4_DAM_TO_KHERSON_FLOODWAY [--window leafon]
"""
from __future__ import annotations

import argparse
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
import rasterio

from swot_dnipro import sentinel_preprocess as SP
from swot_dnipro.plotting import maps as M

ZN = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
IDX = ("NDVI", "NDWI", "MNDWI", "NDMI", "BSI", "AWEIsh", "NDTI")
SCL_COL = {0: "#ffffff", 1: "#ff0000", 2: "#404040", 3: "#8b4513", 4: "#2e8b57", 5: "#e6c35c", 6: "#1f78b4", 7: "#b0b0b0", 8: "#d0d0d0", 9: "#f0f0f0", 10: "#66ccff", 11: "#ff66ff"}
OUT = ROOT / "outputs/figures/p40_maps"


def decim(a, target=1500):
    s = max(1, int(np.ceil(max(a.shape) / target)))
    return a[::s, ::s]


def read(path, scale=None):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("f4"); nd = ds.nodata
        if nd is not None:
            a[a == nd] = np.nan
        ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
        tags = ds.tags()
    return decim(a) * (scale or 1.0), ext, tags


DOM = {"geom": None, "bounds": None}


def set_domain(name):
    from swot_dnipro import spatial_domains as SD
    g = SD.load_utm("reservoir_full_pool_prebreach" if name == "pool" else "below_dam_floodplain")
    DOM["geom"], DOM["bounds"] = g, g.bounds


def draw(ax, path, var, zone):
    if var in IDX:
        a, ext, tags = read(path, 1e-4); sc = M.INDEX_SCALE[var]
        im = ax.imshow(a, extent=ext, cmap=sc["cmap"], vmin=sc["vmin"], vmax=sc["vmax"], interpolation="nearest")
    elif var == "class_mode":
        a, ext, tags = read(path); a[a == 0] = np.nan
        cm_, nm_ = M.class_cmap(); im = ax.imshow(a, extent=ext, cmap=cm_, norm=nm_, interpolation="nearest")
    elif var == "scl_mode":
        from matplotlib.colors import ListedColormap
        a, ext, tags = read(path); a[a == 0] = np.nan
        im = ax.imshow(a, extent=ext, cmap=ListedColormap([SCL_COL[i] for i in range(12)]), vmin=-0.5, vmax=11.5, interpolation="nearest")
    else:  # water_share, n_valid
        a, ext, tags = read(path); a[a == 255] = np.nan
        im = ax.imshow(a, extent=ext, cmap="Blues" if var == "water_share" else "viridis", vmin=0, vmax=100 if var == "water_share" else None, interpolation="nearest")
    M.zone_outline(ax, zone); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if DOM["geom"] is not None:
        import geopandas as gpd
        gpd.GeoSeries([DOM["geom"]]).boundary.plot(ax=ax, color="black", lw=0.5)
        b = DOM["bounds"]; ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
    else:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    return im, tags


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zones", nargs="*", default=list(ZN)); ap.add_argument("--window", default="leafon"); ap.add_argument("--years", type=int, nargs="*", default=list(range(2017, 2027)))
    ap.add_argument("--domain", choices=("pool", "below_dam_floodplain"), help="crop the maps to a domain (pool = former water surface window; floodplain = p42 domain) and outline it")
    a = ap.parse_args()
    if a.domain:
        set_domain(a.domain)
    for zone in a.zones:
        n = ZN[zone]; src = ROOT / "outputs/rasters" / f"zone{n}" / "annual"; od = OUT / (f"{a.domain}_{zone}" if a.domain else zone); od.mkdir(parents=True, exist_ok=True)
        years = [y for y in a.years if (src / f"zone{n}_class_mode_{y}_{a.window}_20m.tif").exists()]
        if not years:
            print(f"{zone}: no {a.window} composites"); continue
        for var in IDX + ("class_mode", "scl_mode", "water_share"):
            files = [(y, src / f"zone{n}_{var}_{y}_{a.window}_20m.tif") for y in years if (src / f"zone{n}_{var}_{y}_{a.window}_20m.tif").exists()]
            if not files:
                continue
            for y, f in files:                                       # single-year maps
                fig, ax = plt.subplots(figsize=(9, 7))
                im, tags = draw(ax, f, var, zone)
                if var in IDX or var == "water_share":
                    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=var if var in IDX else "water share, % of observed dates")
                elif var == "class_mode":
                    M.class_legend(ax)
                ax.set_title(f"{zone} · {var} · {y} {a.window} · {tags.get('n_dates', '?')} dates")
                fig.savefig(od / f"{zone}_{var}_{y}_{a.window}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
            k = len(files); cols = min(5, k); rows = int(np.ceil(k / cols))   # multi-year strip
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, (y, f) in zip(axes.ravel(), files):
                ax.axis("on"); im, tags = draw(ax, f, var, zone); ax.set_title(f"{y} ({tags.get('n_dates', '?')} d)", fontsize=10)
            if var in IDX or var == "water_share":
                fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label=var)
            fig.suptitle(f"{zone} · {var} · {a.window} {years[0]}–{years[-1]}", fontsize=12)
            fig.savefig(od / f"{zone}_{var}_years_{a.window}.png", dpi=100, bbox_inches="tight"); plt.close(fig)
            print(f"  {zone} {var}: {k} years", flush=True)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
