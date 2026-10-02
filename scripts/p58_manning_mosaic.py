#!/usr/bin/env python
"""P58 -- one Manning-n raster per state for the whole breach system (pool + below-dam floodplain), for HEC-RAS (user 2026-09-19).

Sources (p43, 20 m): outputs/rasters/roughness/pool/zone1_poolwin_manning_{classes,n_low,n_base,n_high}_<STATE>.tif and
outputs/rasters/roughness/below_dam_floodplain/zone{4,2}_manning_*_<STATE>.tif. Priority where frames overlap: pool inside the registry
pool polygon; zone4 then zone2 outside it (zone4 first: the dam-Kherson reach, zone2 fills the delta). Output grid = the p55 seamless DEM
50 m union frame (so n and terrain share a grid) and a 20 m version on the same extent (HEC-RAS 2D land-cover layer).
Outputs: $BULK_ROOT/dem_seamless/manning_<layer>_<STATE>_{50,20}m.tif, outputs/tables/p58_manning_mosaic_summary.csv
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
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

RO = ROOT / "outputs/rasters/roughness"; OUT = CFG.BULK_ROOT / "dem_seamless"
STATES = ("BREACH_2023", "FIRST_EXPOSURE_2023", "STATE_2024", "STATE_2025", "CURRENT_2026")
LAYERS = {"classes": ("uint8", 0, Resampling.nearest), "n_base": ("float32", -9999.0, Resampling.nearest), "n_low": ("float32", -9999.0, Resampling.nearest), "n_high": ("float32", -9999.0, Resampling.nearest)}


def to_frame(path, shape, tr, dtype, nodata, rs):
    dst = np.full(shape, nodata, dtype)
    with rasterio.open(path) as ds:
        reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=tr, dst_crs=CFG.CRS_METRIC, resampling=rs, src_nodata=ds.nodata, dst_nodata=nodata)
    return dst


def main():
    with rasterio.open(OUT / "dem_seamless_evrf2019_50m.tif") as ds:
        b = ds.bounds; frames = {50: (ds.transform, (ds.height, ds.width))}
    tr20 = rasterio.transform.from_origin(b.left, b.top, 20.0, 20.0); frames[20] = (tr20, (int(round((b.top - b.bottom) / 20)), int(round((b.right - b.left) / 20))))
    pool_poly = SD.load_utm("reservoir_full_pool_prebreach"); rows = []
    for cell, (tr, shape) in frames.items():
        pool_in = features.rasterize([(pool_poly, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        for st in STATES:
            for layer, (dtype, nd, rs) in LAYERS.items():
                mos = np.full(shape, nd, dtype); src = np.zeros(shape, "u1")
                p = to_frame(RO / f"pool/zone1_poolwin_manning_{layer}_{st}.tif", shape, tr, dtype, nd, rs); m = (p != nd) & pool_in; mos[m] = p[m]; src[m] = 1
                for z, code in ((4, 2), (2, 3)):
                    f = RO / f"below_dam_floodplain/zone{z}_manning_{layer}_{st}.tif"
                    if f.exists():
                        a = to_frame(f, shape, tr, dtype, nd, rs); m = (a != nd) & ~pool_in & (src == 0); mos[m] = a[m]; src[m] = code
                prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype=dtype, crs=CFG.CRS_METRIC, transform=tr, nodata=nd, compress="deflate", tiled=True)
                with rasterio.open(OUT / f"manning_{layer}_{st}_{cell}m.tif", "w", **prof) as ds:
                    ds.write(mos, 1); ds.update_tags(state=st, layer=layer, sources="1 pool (p43 zone1_poolwin) | 2 zone4 | 3 zone2", classes="config/roughness_classes.yaml codes", producer="p58_manning_mosaic.py", grid="p55 seamless DEM frame")
                if layer == "n_base":
                    v = mos[mos != nd]; a_ = cell * cell / 1e6
                    rows.append(dict(cell_m=cell, state=st, km2_total=round(v.size * a_, 1), km2_pool=round(float((src == 1).sum()) * a_, 1), km2_zone4=round(float((src == 2).sum()) * a_, 1), km2_zone2=round(float((src == 3).sum()) * a_, 1), n_base_p10=round(float(np.percentile(v, 10)), 3), n_base_p50=round(float(np.median(v)), 3), n_base_p90=round(float(np.percentile(v, 90)), 3), n_base_area_weighted=round(float(v.mean()), 4)))
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(CFG.TABLES / "p58_manning_mosaic_summary.csv", index=False); print("->", OUT)


if __name__ == "__main__":
    main()
