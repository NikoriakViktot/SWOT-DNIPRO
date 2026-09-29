#!/usr/bin/env python
"""P41 -- static land-cover layers on the zone frames (plan 15, WP1): ESA WorldCover and Dynamic World annual clips.

Two sources, both Sentinel-2 based, both put on `SD.build_grid(zone, 20 m)` next to the p25/p40 products:

  WorldCover v100 (2020) and v200 (2021), 10 m, from Microsoft Planetary Computer (`esa-worldcover`,
  the same route p0b used for the water domain). Nearest onto 20 m (a 10 m class cannot be averaged).
  -> $BULK_ROOT/worldcover_frames/<ZONE>/wc_<year>_20m.tif   uint8 ESA classes (10 tree, 20 shrub, 30 grass,
     40 crop, 50 built, 60 bare, 80 water, 90 wetland, 95 mangrove, 100 moss); 0 = nodata.
  This is the PRE-BREACH static LULC (2020/2021) for BREACH_2023 states.

  Dynamic World annual clips (`$BULK_ROOT/dynamic_world_annual/dw_<zone>_<year>.tif`, EPSG:4326, 30 m,
  float32 bands label_mode + <class>_mean) -> `$BULK_ROOT/dynamic_world_frames/<ZONE>/dw_<year>_20m_from_annual.tif`
  in the p37a frame layout (band 1 label uint8, bands 2..10 probabilities x100 uint8, nodata 255), nearest for
  the label, bilinear for probabilities. Written ONLY for years without a p37a frame (2025, 2026 ...), never
  overwriting p37a's 2022-2024 frames. Clips are polygon-clipped, so the frame margins outside the zone polygon
  stay 255 -- that is recorded in the tags.

Usage: python scripts/p41_static_lulc_frames.py --zone ZONE_1_KAKHOVKA_LOWER_DNIPRO [--skip-worldcover] [--skip-dw]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP

ZONES = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": "zone_1_kakhovka_lower_dnipro", "ZONE_2_KHERSON_DELTA": "zone_2_kherson_delta",
         "ZONE_3_DNIPRO_BUG_ESTUARY": "zone_3_dnipro_bug_estuary", "ZONE_4_DAM_TO_KHERSON_FLOODWAY": "zone_4_dam_to_kherson_floodway"}
DW_CLASSES = ("water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice")
CELL = 20.0


def worldcover(zone: str, G: dict, out_dir: Path) -> list[str]:
    import planetary_computer as pc
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    bbox = transform_bounds(CFG.CRS_METRIC, "EPSG:4326", G["x0"], G["y0"], G["x1"], G["y1"])
    written = []
    for year, version in ((2020, "v100"), (2021, "v200")):
        items = list(cat.search(collections=["esa-worldcover"], bbox=bbox, query={"esa_worldcover:product_version": {"eq": version}}).items())
        if not items:
            items = [i for i in cat.search(collections=["esa-worldcover"], bbox=bbox).items() if str(year) in i.id]
        if not items:
            print(f"  WorldCover {year}: no items"); continue
        dst = np.zeros((G["ny"], G["nx"]), "u1")
        for it in items:
            href = it.assets["map"].href
            with rasterio.open(href) as src:
                from rasterio.windows import Window
                win = src.window(*transform_bounds(CFG.CRS_METRIC, src.crs, G["x0"], G["y0"], G["x1"], G["y1"]))
                win = win.round_offsets().round_lengths().intersection(Window(0, 0, src.width, src.height))
                if win.width <= 0 or win.height <= 0:
                    continue
                a = src.read(1, window=win); tr = src.window_transform(win)
                tmp = np.zeros_like(dst)
                reproject(source=a, destination=tmp, src_transform=tr, src_crs=src.crs, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                          resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
                dst[(dst == 0) & (tmp > 0)] = tmp[(dst == 0) & (tmp > 0)]
        p = out_dir / f"wc_{year}_20m.tif"
        SP.write_uint8(p, dst, G, dict(source=f"ESA WorldCover {version} ({year}) via Planetary Computer, tiles={'|'.join(i.id for i in items)}", classes="10 tree 20 shrub 30 grass 40 crop 50 built 60 bare 70 snow 80 water 90 wetland 95 mangrove 100 moss; 0 nodata",
                                        resampling="nearest 10 m -> 20 m", zone=zone, producer="p41_static_lulc_frames.py"), nodata=0)
        written.append(str(p)); print(f"  WorldCover {year}: {len(items)} tiles -> {p.name}, valid {float((dst > 0)[G['inside']].mean()):.3f} of zone")
    return written


def dw_annual(zone: str, G: dict, out_dir: Path) -> list[str]:
    src_dir = CFG.BULK_ROOT / "dynamic_world_annual"
    written = []
    for year in range(2017, 2027):
        if (out_dir / f"dw_{year}_20m.tif").exists():
            continue                                    # p37a frame exists -> keep it
        f = src_dir / f"dw_{ZONES[zone]}_{year}.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as src:
            names = [src.descriptions[i] or "" for i in range(src.count)]
            lab = src.read(1); probs = {n.replace("_mean", ""): src.read(i + 1) for i, n in enumerate(names) if n.endswith("_mean")}
            tr, crs = src.transform, src.crs
        out = np.full((10, G["ny"], G["nx"]), 255, "u1")
        d = np.full((G["ny"], G["nx"]), np.nan, "f4")
        reproject(source=lab.astype("f4"), destination=d, src_transform=tr, src_crs=crs, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC, resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        valid = np.isfinite(d)
        out[0][valid] = np.clip(d[valid], 0, 8).astype("u1")
        for bi, cls in enumerate(DW_CLASSES, 1):
            if cls not in probs:
                continue
            d = np.full((G["ny"], G["nx"]), np.nan, "f4")
            reproject(source=probs[cls].astype("f4"), destination=d, src_transform=tr, src_crs=crs, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            v = np.isfinite(d) & valid
            out[bi][v] = np.clip(np.round(d[v] * 100), 0, 100).astype("u1")
        p = out_dir / f"dw_{year}_20m_from_annual.tif"
        with rasterio.open(p, "w", **SP._profile(G, 10, "uint8", 255)) as dst:
            dst.write(out)
            for i, n in enumerate(["label"] + [f"{c}_prob_x100" for c in DW_CLASSES], 1):
                dst.set_band_description(i, n)
            dst.update_tags(source=f"{f.name} (GEE Dynamic World annual clip, 30 m, polygon-clipped)", year=str(year), zone=zone, layout="p37a frame layout: label + 9 probabilities x100; nodata 255",
                            resampling="nearest label, bilinear probabilities", note="margins outside the zone polygon are nodata (clip), unlike p37a frames", producer="p41_static_lulc_frames.py")
        written.append(str(p)); print(f"  DW annual {year}: -> {p.name}, valid {float(valid[G['inside']].mean()):.3f} of zone")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=list(ZONES))
    ap.add_argument("--skip-worldcover", action="store_true")
    ap.add_argument("--skip-dw", action="store_true")
    a = ap.parse_args()
    G = SP.zone_grid(a.zone, CELL)
    print(f"P41 {a.zone}: grid {G['nx']}x{G['ny']}")
    if not a.skip_worldcover:
        d = CFG.BULK_ROOT / "worldcover_frames" / a.zone; d.mkdir(parents=True, exist_ok=True)
        worldcover(a.zone, G, d)
    if not a.skip_dw:
        d = CFG.BULK_ROOT / "dynamic_world_frames" / a.zone; d.mkdir(parents=True, exist_ok=True)
        dw_annual(a.zone, G, d)


if __name__ == "__main__":
    main()
