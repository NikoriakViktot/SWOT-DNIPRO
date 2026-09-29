#!/usr/bin/env python
"""P1B (candidate ranking) + P1D (2023-06-05 mandatory checkpoint), 2026-09-11
operator correction. Reads P1A's catalogue, builds the TRUE geometric union
of native tile footprints per date (not a sum of per-tile areas, not an
assumed 3-tile list), and ranks/checks against reservoir_full_pool_prebreach.

Outputs
-------
outputs/tables/prebreach_fullpool_scene_candidates.csv   (P1B)
outputs/figures/P1_20230605_tile_union_checkpoint.png    (P1D)
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
from matplotlib.patches import Patch
from shapely import wkt as shwkt
from shapely.geometry import box as shbox
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY, PURP = ("#1a2228", "#236f8c", "#c1402a",
    "#b07d27", "#3f7d4e", "#8a94a3", "#7d5ba6")
CAT = CFG.ROOT / "data" / "catalog" / "prebreach_sentinel_discovery_full.csv"
# verified this session from P1A: only these 4 of the 6 candidate tiles ever
# have non-zero intersection with the reservoir (36TXS, 36UWU are 0.00 km2 at
# every date in the discovery table -- corroborates reselect_sentinel.py's
# independent ATL13-based finding of "0.0% useless" for the same two tiles)
REQUIRED_TILES = {"36TWS", "36TWT", "36TXT", "36UXU"}
ZERO_TILES = {"36TXS", "36UWU"}
CHECK_DATE = "2023-06-05"
OLD_RES_BBOX = shbox(33.35, 46.65, 35.20, 47.95)          # phase20_water_objects.py
OLD_DISCOVERY_WKT = shbox(33.35, 46.75, 35.35, 47.79)     # phase19_discover_targeted.py
OLD_DISCOVER_SWOT_BOX = shbox(33.3524, 46.7556, 35.3439, 47.7774)  # discover_swot.py

_TF_4326_TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform

RAW_DIRS = (CFG.ROOT / "data" / "raw" / "sentinel_p1_targeted",
           CFG.ROOT / "data" / "raw" / "sentinel", Path("/mnt/e/data_swot/sentinel"))


def is_cached_now(date: str, tile: str) -> bool:
    """Live filesystem check -- P1A's cached_locally column is a snapshot from
    before the targeted downloads (Step 6-7) landed, so it under-reports what
    is actually on disk now. Check all three raw locations directly."""
    tok = date.replace("-", "")
    for base in RAW_DIRS:
        if base.exists() and any(base.glob(f"*{tok}T*_T{tile}_*.SAFE.zip")):
            return True
    return False


def load_gauge():
    g = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    roz = g[(g.station_or_domain == "station:80959") & (g.source == "gauge")].copy()
    roz["date"] = pd.to_datetime(roz.date).dt.strftime("%Y-%m-%d")
    return roz.set_index("date")["corrected_level_m"]


def main() -> None:
    if not CAT.exists():
        raise SystemExit(f"missing {CAT} -- run scripts/p1a_prebreach_discovery.py first")
    df = pd.read_csv(CAT)
    df["geom"] = df.native_geometry_wkt.map(lambda s: shwkt.loads(s) if isinstance(s, str) and s else None)
    df = df.dropna(subset=["geom"])
    # dedupe (date, tile): reprocessing baselines can duplicate a real acquisition
    df = df.sort_values("cloud_cover_metadata").drop_duplicates(["date", "tile_id"], keep="first")

    reservoir = SD.load("reservoir_full_pool_prebreach")
    reservoir_m = shp_transform(_TF_4326_TO_M, reservoir)
    reservoir_area_km2 = reservoir_m.area / 1e6
    roz = load_gauge()

    # ================================================================ P1B
    rows = []
    for date, g in df.groupby("date"):
        g = g[g.tile_id.isin(REQUIRED_TILES)]
        if g.empty:
            continue
        geoms_m = [shp_transform(_TF_4326_TO_M, geo) for geo in g.geom]
        union_m = unary_union(geoms_m)
        inter_m = union_m.intersection(reservoir_m)
        cov_km2 = inter_m.area / 1e6
        frac = cov_km2 / reservoir_area_km2
        missing_km2 = reservoir_area_km2 - cov_km2
        present = set(g.tile_id)
        missing_tiles = REQUIRED_TILES - present
        # area-weighted cloud over the covering tiles (whole-scene cloudCover,
        # NOT a true reservoir-corridor SCL fraction -- that requires opening
        # the raster, only done for cached scenes; flagged explicitly)
        w = g.intersection_area_km2.clip(lower=0.01)
        cloud_w = float((g.cloud_cover_metadata * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        gauge_level = roz.get(date, np.nan)
        rows.append({
            "date": date, "n_tiles_present": len(present),
            "tiles_present": "|".join(sorted(present)),
            "missing_tiles": "|".join(sorted(missing_tiles)),
            "native_tile_union_area_km2": round(union_m.area / 1e6, 1),
            "reservoir_intersection_area_km2": round(cov_km2, 1),
            "native_coverage_fraction": round(frac, 4),
            "missing_area_km2": round(missing_km2, 1),
            "est_cloud_over_reservoir_pct_weighted": round(cloud_w, 1) if np.isfinite(cloud_w) else np.nan,
            "all_cached": bool(g.cached_locally.all()) if len(g) == len(REQUIRED_TILES) else False,
            "gauge_level_evrf2019_m": gauge_level,
            "has_gauge": bool(np.isfinite(gauge_level)) if gauge_level == gauge_level else False,
        })
    cand = pd.DataFrame(rows)
    # rank: full coverage first, then low cloud, then presence of gauge control
    cand["rank_score"] = (
        (1 - cand.native_coverage_fraction) * 1000
        + cand.est_cloud_over_reservoir_pct_weighted.fillna(50)
        + np.where(cand.has_gauge, 0, 5)
    )
    cand = cand.sort_values("rank_score").reset_index(drop=True)
    out_b = CFG.TABLES / "prebreach_fullpool_scene_candidates.csv"
    cand.to_csv(out_b, index=False)
    print(f"-> {out_b}  ({len(cand)} dates ranked)")
    print(f"\ndates with n_tiles_present == {len(REQUIRED_TILES)} (all required tiles present): "
          f"{int((cand.n_tiles_present == len(REQUIRED_TILES)).sum())} / {len(cand)}")
    print(f"dates with native_coverage_fraction >= 0.999: "
          f"{int((cand.native_coverage_fraction >= 0.999).sum())}")
    print(f"dates with native_coverage_fraction >= 0.95: "
          f"{int((cand.native_coverage_fraction >= 0.95).sum())}")
    print("\nTOP 10 CANDIDATES:")
    print(cand.head(10)[["date", "n_tiles_present", "native_coverage_fraction",
                         "est_cloud_over_reservoir_pct_weighted", "gauge_level_evrf2019_m",
                         "missing_tiles", "all_cached"]].to_string())

    # ================================================================ P1D
    d0 = df[(df.date == CHECK_DATE) & (df.tile_id.isin(REQUIRED_TILES))]
    tiles_geo = {r.tile_id: r.geom for r in d0.itertuples()}
    union_native = unary_union(list(tiles_geo.values()))
    union_m = shp_transform(_TF_4326_TO_M, union_native)
    inter_m = union_m.intersection(reservoir_m)
    native_frac = inter_m.area / reservoir_area_km2 / 1e6 * 1e6  # km2/km2
    native_frac = inter_m.area / 1e6 / reservoir_area_km2
    missing_km2 = reservoir_area_km2 - inter_m.area / 1e6
    uncovered = reservoir.difference(union_native)
    cached_tiles = sorted(t for t in tiles_geo if is_cached_now(CHECK_DATE, t))
    missing_tiles_060 = sorted(REQUIRED_TILES - set(d0.tile_id))
    print("\n" + "=" * 70)
    print("P1D -- 2023-06-05 TRUE NATIVE-TILE-UNION CHECKPOINT")
    print("=" * 70)
    print(f"required tiles found in catalogue for this date: {sorted(tiles_geo)}")
    print(f"cached locally on this date: {cached_tiles}")
    print(f"NOT cached (would need download): {sorted(set(tiles_geo) - set(cached_tiles))}")
    print(f"native geometric coverage fraction of reservoir_full_pool_prebreach: {native_frac:.4f}")
    print(f"missing_area_km2 (reservoir area not covered by ANY native tile): {missing_km2:.1f}")
    for t in sorted(tiles_geo):
        row = d0[d0.tile_id == t].iloc[0]
        print(f"  {t}: cached={t in cached_tiles}  cloud={row.cloud_cover_metadata:.1f}%  "
              f"reservoir_intersection={row.intersection_area_km2:.1f} km2")

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))

    def plot_poly(geom, **kw):
        if geom.is_empty:
            return
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        for gpart in geoms:
            if gpart.is_empty or gpart.area == 0:
                continue
            xs, ys = gpart.exterior.xy
            ax.fill(xs, ys, **kw)
            for interior in gpart.interiors:
                ix, iy = interior.xy
                ax.fill(ix, iy, color="white", zorder=kw.get("zorder", 1) + 0.1)

    tile_colors = {"36TWS": "#a8d5ba", "36TWT": "#8ecae6", "36TXT": "#ffd166",
                   "36UXU": "#f28482"}
    for t, geo in tiles_geo.items():
        xs, ys = geo.exterior.xy
        ax.plot(xs, ys, color=tile_colors.get(t, GREY), lw=2,
                linestyle="-" if t in cached_tiles else "--", zorder=3,
                label=f"{t} native footprint ({'cached' if t in cached_tiles else 'NOT cached'})")
        ax.fill(xs, ys, color=tile_colors.get(t, GREY), alpha=0.12, zorder=1)

    plot_poly(reservoir, facecolor="none", edgecolor=INK, lw=2.6, zorder=5)
    ax.plot([], [], color=INK, lw=2.6, label="reservoir_full_pool_prebreach (SA_2, authoritative)")

    plot_poly(uncovered, facecolor=RED, alpha=0.55, edgecolor=RED, lw=1.5, zorder=6)
    ax.plot([], [], color=RED, lw=6, alpha=0.55,
            label=f"UNCOVERED by any native tile ({missing_km2:.0f} km2, "
                  f"{(1-native_frac)*100:.1f}%)")

    # old clipped footprint + old bboxes, for direct visual comparison
    try:
        fp_old = json.load(open(CFG.FIGDATA / "P20_reservoir_footprint.geojson"))
        from shapely.geometry import shape as shp_shape
        fp_geom_m = shp_shape(fp_old["geometry"])
        tf_m_to_ll = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform
        fp_geom = shp_transform(tf_m_to_ll, fp_geom_m)
        plot_poly(fp_geom, facecolor="none", edgecolor=PURP, lw=1.8, linestyle=":", zorder=4)
        ax.plot([], [], color=PURP, lw=1.8, linestyle=":",
                label="OLD clipped P20_reservoir_footprint (pre-fix, for comparison)")
    except Exception as e:
        print(f"  (could not overlay old P20 footprint: {e})")

    for bx, name, col in ((OLD_RES_BBOX, "old RES_BBOX_4326", AMBER),
                          (OLD_DISCOVERY_WKT, "old RESERVOIR_WKT (discovery)", GREEN),
                          (OLD_DISCOVER_SWOT_BOX, "old discover_swot.py box", GREY)):
        xs, ys = bx.exterior.xy
        ax.plot(xs, ys, color=col, lw=1.2, linestyle="--", zorder=2, alpha=0.8)
        ax.plot([], [], color=col, lw=1.2, linestyle="--", alpha=0.8, label=name)

    # landmarks
    dam_lon, dam_lat = CFG.KAKHOVKA_DAM
    ax.scatter([dam_lon], [dam_lat], color=INK, marker="*", s=220, zorder=8,
              label="Kakhovka dam (breached 2023-06-06)")
    roz_lon, roz_lat = 35.148890, 47.771208
    ax.scatter([roz_lon], [roz_lat], color=BLUE, marker="^", s=140, zorder=8,
              label="Rozumivka gauge (80959) / near DniproHES-Zaporizhzhia end")

    ax.set_xlabel("lon (deg)"); ax.set_ylabel("lat (deg)")
    ax.set_title(f"P1D checkpoint -- {CHECK_DATE} native Sentinel-2 tile coverage "
                f"vs. authoritative reservoir\n"
                f"native coverage {native_frac*100:.1f}% -- missing {missing_km2:.0f} km2 "
                f"({(1-native_frac)*100:.1f}%), all in the NE/Zaporizhzhia corner",
                fontsize=12, loc="left")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95, ncol=1)
    ax.grid(alpha=0.2)
    ax.set_aspect(1 / np.cos(np.radians(47.3)))
    fig.tight_layout()
    out_fig = CFG.FIG / "P1_20230605_tile_union_checkpoint.png"
    fig.savefig(out_fig, dpi=170)
    print(f"\n-> {out_fig}")


if __name__ == "__main__":
    main()
