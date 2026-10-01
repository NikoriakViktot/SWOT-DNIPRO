#!/usr/bin/env python
"""Build the ICESat-2 pass layer of the companion app: ground tracks, not points.

One line per beam of one overpass (date × RGT × beam), for every ICESat-2 product
this project used:

    ATL13  water-surface heights, the observation of Paper 1, in its three disjoint
           samples: the former reservoir (R; `qc_pass` marks the transects that
           passed the slope-analysis QC), Kherson (V6) and the Dnipro–Buh estuary;
    ATL08  terrain segments over the drained bed, the independent check of the
           companion terrain work (k10: POST_BREACH, inside the former pool, night,
           snow-free land, ≥ 10 ground photons, canopy ≤ 3 m). Not part of Paper 1.

The source extractions are larger than the study system (the estuary table reaches
the open Black Sea), so every track is clipped to the union of the four frozen
zones R/F/D/E of the paper, taken from the geometry registry and buffered by
CLIP_BUFFER_M; nothing outside the study system is drawn.

Segments of one beam are ordered along track and joined into a LineString; a gap
above GAP_M (land between water bodies, dropped segments) starts a new part, so a
track never jumps across land. Lines are simplified in EPSG:32636 and written in
EPSG:4326 with five decimals.

    outputs/paper/app_data/icesat2_passes.geojson
    outputs/paper/app_data/icesat2_passes_manifest.json   sources, sha256, counts

    python app/prepare_app_icesat2_passes.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

OUT = ROOT / "outputs/paper/app_data"
ATL13 = {"reservoir": CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet",
         "kherson": CFG.ICESAT_ROOT / "data/processed/kherson_atl13_segments.parquet",
         "estuary": CFG.ICESAT_ROOT / "data/processed/dnipro_estuary_atl13_segments.parquet"}
PASS_LEVELS = {"reservoir": CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_pass_levels.parquet",
               "kherson": CFG.ICESAT_ROOT / "data/processed/kherson_atl13_pass_levels.parquet",
               "estuary": CFG.ICESAT_ROOT / "data/processed/dnipro_estuary_atl13_pass_levels.parquet"}
ATL08 = CFG.BULK_ROOT / "data_swot/processed/atl08/kakhovka_atl08_terrain.parquet"
GT = {10: "gt1l", 20: "gt1r", 30: "gt2l", 40: "gt2r", 50: "gt3l", 60: "gt3r"}
GAP_M, SIMPLIFY_M, CLIP_BUFFER_M = 3000.0, 100.0, 2000.0
ZONES = ("R_FORMER_KAKHOVKA_RESERVOIR", "F_LOWER_DNIPRO_FLOODWAY", "D_KHERSON_DELTA", "E_DNIPRO_BUG_ESTUARY")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def study_system():
    from shapely.ops import unary_union
    return unary_union([SD.load_utm(z) for z in ZONES]).buffer(CLIP_BUFFER_M)


DOMAIN = None


def tracks(df: pd.DataFrame, keys: list[str]) -> gpd.GeoDataFrame:
    """One (Multi)LineString per group of `keys`, split at along-track gaps, clipped
    to the study system (zones R/F/D/E + buffer, from the registry)."""
    global DOMAIN
    if DOMAIN is None:
        DOMAIN = study_system()
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(CFG.CRS_METRIC)
    g = g[g.within(DOMAIN)]
    g["x"], g["y"] = g.geometry.x, g.geometry.y
    rows = []
    for k, part in g.sort_values("time").groupby(keys, sort=False):
        if len(part) < 2:
            continue
        xy = part[["x", "y"]].to_numpy()
        step = np.hypot(*np.diff(xy, axis=0).T)
        cuts = np.flatnonzero(step > GAP_M) + 1
        pieces = [p for p in np.split(xy, cuts) if len(p) >= 2]
        if not pieces:
            continue
        geom = LineString(pieces[0]) if len(pieces) == 1 else MultiLineString([LineString(p) for p in pieces])
        rows.append(dict(zip(keys, k), n_segments=len(part), geometry=geom.simplify(SIMPLIFY_M)))
    out = gpd.GeoDataFrame(rows, crs=CFG.CRS_METRIC)
    out["geometry"] = out.intersection(DOMAIN)
    out = out[~out.is_empty & out.geometry.notna()].copy()
    out["length_km"] = (out.length / 1e3).round(1)
    return out.to_crs(4326)


def main() -> None:
    src, frames = {}, []
    # ATL13: the observation of Paper 1, three samples; period (and QC for R) from the pass-level tables
    for sample, p in ATL13.items():
        pl = pd.read_parquet(PASS_LEVELS[sample])
        pl["date"] = pd.to_datetime(pl.date).dt.strftime("%Y-%m-%d")
        if "qc_pass" not in pl:
            pl["qc_pass"] = np.nan
        qc = pl.groupby(["date", "rgt", "beam"]).agg(qc_pass=("qc_pass", "first"), period=("period", "first"))
        src[f"atl13_{sample}_pass_levels"] = f"{PASS_LEVELS[sample].name} ({sha(PASS_LEVELS[sample])})"
        s = pd.read_parquet(p, columns=["time", "rgt", "beam", "lat", "lon"])
        s["date"] = pd.to_datetime(s.time, utc=True).dt.strftime("%Y-%m-%d")
        t = tracks(s, ["date", "rgt", "beam"])
        t = t.merge(qc, left_on=["date", "rgt", "beam"], right_index=True, how="left")
        # beams the pass-level table dropped (too few points) keep their date's period, flagged
        dd = pl[pl.period == "BREACH_DRAWDOWN"].date
        lo, hi = (dd.min(), dd.max()) if len(dd) else (CFG.BREACH_DATE, CFG.BREACH_DATE)
        t["in_pass_table"] = t.period.notna()
        by_date = np.where(t.date < CFG.BREACH_DATE, "PRE_BREACH",
                           np.where(t.date <= hi, "BREACH_DRAWDOWN", "POST_BREACH"))
        t["period"] = t.period.fillna(pd.Series(by_date, index=t.index))
        t["qc_pass"] = t.qc_pass.map({True: "pass", False: "fail"}).fillna("not scored")
        t["product"], t["sample"] = "ATL13", sample
        frames.append(t)
        src[f"atl13_{sample}"] = f"{p.name} ({sha(p)})"
        print(f"  ATL13 {sample:<9} {len(t):5d} beam tracks, {s.shape[0]:,} segments; "
              f"{(~t.in_pass_table).sum()} beams not in the pass table (period by date)")
    # ATL08: terrain of the drained bed (companion work), k10 QC rules
    if ATL08.exists():
        a = pd.read_parquet(ATL08, columns=["time", "rgt", "gt", "lat", "lon", "in_former_pool", "period",
                                            "solar_elevation", "snowcover", "gnd_ph_count", "h_canopy"])
        a = a[a.in_former_pool & (a.period == "POST_BREACH") & (a.solar_elevation < 0) & (a.snowcover == 1)
              & (a.gnd_ph_count >= 10) & (a.h_canopy <= 3)].copy()
        a["beam"] = a["gt"].map(GT)
        a["time"] = pd.to_datetime(a.time, utc=True)
        a["date"] = a.time.dt.strftime("%Y-%m-%d")
        t = tracks(a, ["date", "rgt", "beam"])
        t["period"], t["qc_pass"], t["product"], t["sample"] = "POST_BREACH", "k10 QC", "ATL08", "drained bed (k10)"
        t["in_pass_table"] = True
        frames.append(t)
        src["atl08_terrain"] = f"{ATL08.relative_to(CFG.BULK_ROOT)} ({sha(ATL08)})"
        print(f"  ATL08 drained bed {len(t):5d} beam tracks, {a.shape[0]:,} segments after k10 QC")
    else:
        print(f"  ATL08 table not found at {ATL08}; layer built without it")

    g = pd.concat(frames, ignore_index=True)
    g = gpd.GeoDataFrame(g[["product", "sample", "date", "rgt", "beam", "period", "qc_pass", "in_pass_table",
                            "n_segments", "length_km", "geometry"]], crs=4326)
    g["rgt"] = g.rgt.astype(int)
    out = OUT / "icesat2_passes.geojson"
    g.to_file(out, driver="GeoJSON", COORDINATE_PRECISION=5)
    counts = g.groupby(["product", "sample", "period"]).size().reset_index(name="n_beam_tracks")
    passes = g.groupby(["product", "sample"]).apply(lambda x: x[["date", "rgt"]].drop_duplicates().shape[0]).to_dict()
    (OUT / "icesat2_passes_manifest.json").write_text(json.dumps(dict(
        built_by="app/prepare_app_icesat2_passes.py", gap_m=GAP_M, simplify_m=SIMPLIFY_M,
        clipped_to=f"union of {', '.join(ZONES)} (registry) + {CLIP_BUFFER_M:.0f} m",
        sources=src, n_beam_tracks=int(len(g)),
        n_passes_date_x_rgt={f"{k[0]} {k[1]}": int(v) for k, v in passes.items()},
        counts=counts.to_dict(orient="records"), bytes=out.stat().st_size,
        sha256=hashlib.sha256(out.read_bytes()).hexdigest()[:16]), indent=2) + "\n")
    print(f"  -> {out.name} {out.stat().st_size / 1e6:.2f} MB, {len(g)} beam tracks")


if __name__ == "__main__":
    main()
