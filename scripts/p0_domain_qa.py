#!/usr/bin/env python
"""P0 QA -- authoritative spatial domain implementation, verification map +
registry (2026-09-11 execution).

Outputs
-------
outputs/tables/spatial_domain_registry_v2.csv
outputs/figures/P0_spatial_domain_qa.png
"""
from __future__ import annotations

import json
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
from shapely.geometry import box as shbox, shape as shp_shape
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, PURP, GREY = ("#1a2228", "#236f8c", "#c1402a",
    "#b07d27", "#3f7d4e", "#7d5ba6", "#8a94a3")

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform
_TF_M_TO_4326 = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform

OLD_RES_BBOX = shbox(33.35, 46.65, 35.20, 47.95)                    # phase20_water_objects.py
OLD_DISCOVERY_WKT = shbox(33.35, 46.75, 35.35, 47.79)               # phase19_discover_targeted.py
OLD_DISCOVER_SWOT_BOX = shbox(33.3524, 46.7556, 35.3439, 47.7774)   # discover_swot.py


def area_km2(geom_4326):
    return shp_transform(_TF_4326_TO_M, geom_4326).area / 1e6


def main() -> None:
    rows = []
    for name in ("reservoir_full_pool_prebreach", "study_domain_full",
                "reservoir_npu_domain", "below_dam_floodplain"):
        r = SD.registry_row(name)
        r["kind"] = "authoritative (new)"
        rows.append(r)

    fp_old = json.load(open(CFG.FIGDATA / "P20_reservoir_footprint.geojson"))
    fp_geom_m = shp_shape(fp_old["geometry"])
    fp_geom = shp_transform(_TF_M_TO_4326, fp_geom_m)
    rows.append({"name": "P20_reservoir_footprint (OLD, pre-fix)", "status": "RETIRED",
                "area_km2": fp_geom_m.area / 1e6, "lon_min": fp_geom.bounds[0],
                "lat_min": fp_geom.bounds[1], "lon_max": fp_geom.bounds[2],
                "lat_max": fp_geom.bounds[3],
                "geometry_hash": SD.geom_hash(fp_geom), "note": "clipped by RES_BBOX_4326",
                "kind": "retired hardcode output"})

    for name, geom, src in (
        ("RES_BBOX_4326 (OLD, phase20_water_objects.py)", OLD_RES_BBOX, "phase20_water_objects.py:59"),
        ("RESERVOIR_WKT (OLD, phase19_discover_targeted.py)", OLD_DISCOVERY_WKT, "phase19_discover_targeted.py:35"),
        ("discover_swot.py box (OLD)", OLD_DISCOVER_SWOT_BOX, "discover_swot.py:36"),
    ):
        rows.append({"name": name, "status": "RETIRED", "area_km2": area_km2(geom),
                    "lon_min": geom.bounds[0], "lat_min": geom.bounds[1],
                    "lon_max": geom.bounds[2], "lat_max": geom.bounds[3],
                    "geometry_hash": SD.geom_hash(geom), "note": f"source: {src}",
                    "kind": "retired hardcode literal"})

    reg = pd.DataFrame(rows)
    out_csv = CFG.TABLES / "spatial_domain_registry_v2.csv"
    reg.to_csv(out_csv, index=False)
    print(f"-> {out_csv}")
    print(reg[["name", "status", "area_km2", "lon_min", "lat_min", "lon_max", "lat_max"]]
         .to_string(index=False))

    # ---- figure -----------------------------------------------------------
    reservoir = SD.load("reservoir_full_pool_prebreach")
    study = SD.load("study_domain_full")

    fig, ax = plt.subplots(figsize=(11, 9))

    def plot_poly(geom, ax, **kw):
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            if g.is_empty:
                continue
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, **kw)

    plot_poly(study, ax, color=GREEN, lw=1.6, ls="-.", zorder=2)
    ax.plot([], [], color=GREEN, lw=1.6, ls="-.",
            label=f"study_domain_full (NEW, discovery-only, "
                  f"{area_km2(study):,.0f} km2)")

    plot_poly(reservoir, ax, color=INK, lw=2.8, zorder=5)
    ax.plot([], [], color=INK, lw=2.8,
            label=f"reservoir_full_pool_prebreach = SA_2 (NEW authoritative, "
                  f"{area_km2(reservoir):,.0f} km2)")

    xs, ys = fp_geom.exterior.xy if hasattr(fp_geom, "exterior") else (list(fp_geom.geoms)[0].exterior.xy)
    ax.plot(xs, ys, color=PURP, lw=1.8, ls=":", zorder=4)
    ax.plot([], [], color=PURP, lw=1.8, ls=":",
            label=f"OLD P20_reservoir_footprint (pre-fix, {fp_geom_m.area/1e6:,.0f} km2)")

    for bx, name, col in ((OLD_RES_BBOX, "OLD RES_BBOX_4326", AMBER),
                          (OLD_DISCOVERY_WKT, "OLD RESERVOIR_WKT (discovery)", RED),
                          (OLD_DISCOVER_SWOT_BOX, "OLD discover_swot.py box", GREY)):
        xxs, yys = bx.exterior.xy
        ax.plot(xxs, yys, color=col, lw=1.3, ls="--", zorder=3, alpha=0.85)
        ax.plot([], [], color=col, lw=1.3, ls="--", alpha=0.85, label=name)

    dam_lon, dam_lat = CFG.KAKHOVKA_DAM
    ax.scatter([dam_lon], [dam_lat], color=INK, marker="*", s=240, zorder=8)
    ax.annotate("Kakhovka dam\n(breached 2023-06-06)", (dam_lon, dam_lat),
               textcoords="offset points", xytext=(8, 8), fontsize=8)
    roz_lon, roz_lat = 35.148890, 47.771208
    ax.scatter([roz_lon], [roz_lat], color=BLUE, marker="^", s=150, zorder=8)
    ax.annotate("Rozumivka gauge (80959)\n~ Zaporizhzhia / DniproHES end",
               (roz_lon, roz_lat), textcoords="offset points", xytext=(8, -14), fontsize=8)

    ax.set_xlabel("lon (deg)"); ax.set_ylabel("lat (deg)")
    ax.set_title("P0 QA -- authoritative spatial domains vs. the three retired "
                "hard-coded bboxes\n(the RES_BBOX_4326 line cutting through the "
                "NE reservoir is the visible root cause of the reported DEM truncation)",
                fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.2)
    ax.set_aspect(1 / np.cos(np.radians(47.3)))
    fig.tight_layout()
    out_fig = CFG.FIG / "P0_spatial_domain_qa.png"
    fig.savefig(out_fig, dpi=170)
    print(f"\n-> {out_fig}")

    # sanity check requested explicitly before P1A
    assert reservoir.within(study), "study_domain_full does NOT fully contain reservoir_full_pool_prebreach!"
    print("\nOK: study_domain_full fully contains reservoir_full_pool_prebreach.")


if __name__ == "__main__":
    main()
