#!/usr/bin/env python
"""U0 + U1 -- dedicated upper-reach AOI (NOT unioned with SA_2) and
pre-breach water-persistence masks, reusing the already-computed
water_count/obs_count rasters from p1c_v2_full_consensus.py (18 validated
PRE_BREACH dates) rather than rebuilding from scratch.

Outputs
-------
data/processed/study_domain/upper_dniprohes_transition_aoi.gpkg
outputs/figures/U1_prebreach_water_frequency.png
data/processed/study_domain/upper_prebreach_water_frequency.tif
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point, box

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

GAUGE_80957 = (35.08208, 47.87204)
AOI_BUFFER_KM = 10.0
STUDY_DIR = CFG.ROOT / "data" / "processed" / "study_domain"

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform
_TF_M_TO_4326 = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform


def main() -> None:
    sa2 = SD.load("reservoir_full_pool_prebreach")
    gauge_m = Point(*_TF_4326_TO_M(*GAUGE_80957))
    sa2_m = None
    import shapely.ops as ops
    sa2_m = ops.transform(_TF_4326_TO_M, sa2)

    # U0: dedicated AOI, centred on the gauge + a slice of SA_2's own
    # upper end, explicitly NOT unioned with SA_2 as a reservoir product
    aoi_m = gauge_m.buffer(AOI_BUFFER_KM * 1000)
    aoi_m = aoi_m.union(sa2_m.intersection(gauge_m.buffer(15000)))  # include the adjoining SA_2 slice for context only
    aoi = ops.transform(_TF_M_TO_4326, aoi_m)
    gdf = gpd.GeoDataFrame({"name": ["upper_dniprohes_transition_aoi"],
                            "note": ["ANALYSIS AOI ONLY -- not a reservoir product, "
                                    "never union into SA_2"]},
                           geometry=[aoi], crs="EPSG:4326")
    out_gpkg = STUDY_DIR / "upper_dniprohes_transition_aoi.gpkg"
    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"-> {out_gpkg}  area={aoi_m.area/1e6:.1f} km2")

    # U1: clip the already-computed water_count/obs_count rasters to this AOI
    wc_path = STUDY_DIR / "p1c_v2_water_count.tif"
    oc_path = STUDY_DIR / "p1c_v2_obs_count.tif"
    with rasterio.open(wc_path) as wds, rasterio.open(oc_path) as ods:
        minx, miny, maxx, maxy = aoi_m.bounds
        win = from_bounds(minx, miny, maxx, maxy, wds.transform)
        wc = wds.read(1, window=win)
        oc = ods.read(1, window=win)
        win_transform = wds.window_transform(win)
        crs = wds.crs

    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(oc > 0, wc / np.maximum(oc, 1), np.nan)

    out_tif = STUDY_DIR / "upper_prebreach_water_frequency.tif"
    with rasterio.open(out_tif, "w", driver="GTiff", height=freq.shape[0],
                       width=freq.shape[1], count=1, dtype="float32",
                       crs=crs, transform=win_transform, nodata=-1,
                       compress="deflate") as dst:
        dst.write(np.where(np.isnan(freq), -1, freq).astype("float32"), 1)
    print(f"-> {out_tif}")

    n_valid = int((oc > 0).sum())
    n_total = oc.size
    print(f"pixels with >=1 pre-breach observation: {n_valid}/{n_total} "
          f"({n_valid/n_total*100:.1f}%)")
    print(f"max obs_count in this window: {oc.max()} (of 18 possible dates)")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 9))
    ext = (minx, maxx, miny, maxy)
    im = ax.imshow(freq, extent=ext, origin="upper", cmap="Blues", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("pre-breach water frequency (18 VALID dates, 2017-2023)")

    def plot_poly(g, ax, **kw):
        gg = g.geoms if hasattr(g, "geoms") else [g]
        for p in gg:
            if p.is_empty:
                continue
            xs, ys = p.exterior.xy
            ax.plot(xs, ys, **kw)
    plot_poly(sa2_m, ax, color="black", lw=2, label="SA_2 (original, authoritative)")
    ax.scatter([gauge_m.x], [gauge_m.y], color="red", marker="^", s=150, zorder=5,
              label="gauge 80957 (Dniprovske pool, NOT Kakhovka)")
    ax.set_title("U1 -- pre-breach water-persistence frequency, upper DniproHES "
                "transition zone", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    fig.tight_layout()
    out_fig = CFG.FIG / "U1_prebreach_water_frequency.png"
    fig.savefig(out_fig, dpi=160)
    print(f"-> {out_fig}")


if __name__ == "__main__":
    main()
