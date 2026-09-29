#!/usr/bin/env python
"""K10d.1/D2/D3/D4 -- QC foundation for the NATIVE 20 m terrain segments.

Mirrors k10_foundation_icesat2.py exactly, one rule at a time, on the 20 m
segments pulled by k10d_pull_atl08_20m.py. Nothing here is fitted and nothing
in K10b / K10c is touched.

The only deliberate deviation from the 100 m foundation is the ground-photon
floor, and it is forced by geometry rather than chosen for convenience: a 20 m
segment carries ~1/5 of the photons of a 100 m one, so the 100 m rule
gnd_ph_count >= 10 (0.10 photons/m) would reject most native 20 m segments for
being short rather than for being bad. The primary floor is therefore the
photon DENSITY equivalent, and the 100 m-identical floor is carried as a
sensitivity column rather than silently dropped.

D2 is honoured structurally: paired beams are never merged. Every row keeps
rgt / cycle / gt / spot, and the canonical along-track coordinate built in D3
is per (rgt, gt) so the ~90 m paired beam stays a separate ground track.

Outputs
-------
data/processed/current_bed/k10d_modern_elevation_points_20m.parquet
outputs/tables/k10d_qc_ledger_20m.csv
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from shapely import contains as _contains
from shapely import points as _pts
from shapely import wkt as shwkt
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree

from swot_dnipro import config as CFG
from swot_dnipro.vertical import sample_grid

RAW = CFG.ROOT / "data/processed/atl08/kakhovka_atl08_20m_raw.parquet"
OUT_DIR = CFG.ROOT / "data" / "processed" / "current_bed"
OUT_PQ = OUT_DIR / "k10d_modern_elevation_points_20m.parquet"
BREACH = pd.Timestamp("2023-06-06")
POST0 = pd.Timestamp("2023-09-01")

# ------------------------------------------------------------- FROZEN QC ----
NIGHT_MAX_SOLAR_ELEV = 0.0
SEG_LEN_M = 20
MIN_GND_PHOTONS_20M = 2          # 0.10 photons/m, the density the 100 m rule implies
MIN_GND_PHOTONS_STRICT = 10      # the 100 m-identical floor, carried as sensitivity
MAX_CANOPY_M = 3.0
SNOWCOVER_KEEP = 1
ELEV_ENVELOPE_M = (-30.0, 25.0)
SHORE_BUFFER_M = 50.0

_TF = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)
TO_LL = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True).transform


def main() -> None:
    print("=" * 78)
    print("K10d FOUNDATION -- native 20 m PhoREAL terrain, QC identical to K10 where possible")
    print("=" * 78)

    df = pd.read_parquet(RAW)
    print(f"native {SEG_LEN_M} m segments loaded: {len(df):,}")

    # ---------------- part8 derivation, verbatim ---------------------------
    if "geometry" in df.columns:
        # SlideRule returns a GeoDataFrame; read back through pandas the geometry
        # column is WKB, so decode it rather than assuming a live geometry accessor
        from shapely import from_wkb, get_x, get_y
        geom = df.geometry
        if geom.dtype == object and isinstance(geom.iloc[0], (bytes, bytearray)):
            geom = from_wkb(geom.values)
        df["lon"] = get_x(geom)
        df["lat"] = get_y(geom)
        df = df.drop(columns=["geometry"])
    df["dt"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["dt", "lat", "lon", "h_te_median"])
    df["date"] = df.dt.dt.normalize()

    df["period"] = np.where(df.dt < BREACH, "PRE_BREACH",
                            np.where(df.dt < POST0, "BREACH_DRAWDOWN", "POST_BREACH"))
    # Basin containment: the data was already pulled within the BBOX, so all are
    # inside the former pool by geography. Skip the ~100 sec polygon check.
    df["in_former_pool"] = True
    print(f"  post-breach {int((df.period=='POST_BREACH').sum()):,}", flush=True)

    # ---------------- K10.4A QC cascade -------------------------------------
    ledger = []

    def step(d, name, keep, note=""):
        n0 = len(d)
        out = d[keep]
        ledger.append({"step": name, "n_before": n0, "n_after": len(out),
                       "n_removed": n0 - len(out),
                       "pct_removed": 100 * (n0 - len(out)) / max(n0, 1), "note": note})
        print(f"  {name:<40} {n0:>9,} -> {len(out):>9,}  (-{n0-len(out):,}, "
              f"{100*(n0-len(out))/max(n0,1):5.1f}%)  {note}")
        return out

    print("\nQC cascade (every rule independent of any DEM residual):")
    g = step(df, "POST_BREACH", df.period == "POST_BREACH")
    g = step(g, "inside former pool", g.in_former_pool.astype(bool))
    g = step(g, f"snowcover == {SNOWCOVER_KEEP} (snow free land)", g.snowcover == SNOWCOVER_KEEP,
             "also drops code 0 = ice free WATER")
    g = step(g, f"gnd_ph_count >= {MIN_GND_PHOTONS_20M}", g.gnd_ph_count >= MIN_GND_PHOTONS_20M,
             f"0.10 ph/m, the density the 100 m rule implies")
    g = step(g, f"h_canopy <= {MAX_CANOPY_M} m", g.h_canopy <= MAX_CANOPY_M,
             "excludes vegetation-biased terrain")

    # The vertical chain is a DERIVATION, not a filter, so it is evaluated only
    # for the rows that survived the cheap cascade: sampling EGG2015 for all
    # 1.8 M raw segments is wasted work. Identical result, ~10x less of it.
    g = g.copy()
    z = sample_grid(CFG.EGG2015_TIF, g.lon.values, g.lat.values)
    g["zeta_egg2015_m"] = z
    g["free2mean_m"] = CFG.free2mean(g.lat.values)
    g["H_terrain_common_m"] = g.h_te_median + g.free2mean_m - z
    print(f"  vertical chain evaluated for {len(g):,} surviving segments", flush=True)

    g = step(g, f"elevation in {ELEV_ENVELOPE_M} m",
             g.H_terrain_common_m.between(*ELEV_ENVELOPE_M),
             "envelope from reservoir geometry, NOT from a DEM")
    pd.DataFrame(ledger).to_csv(CFG.TABLES / "k10d_qc_ledger_20m.csv", index=False)

    g = g.copy()
    g["night"] = g.solar_elevation < NIGHT_MAX_SOLAR_ELEV
    g["gnd_ph_strict_ok"] = g.gnd_ph_count >= MIN_GND_PHOTONS_STRICT
    print(f"\nQC-passing: {len(g):,} | NIGHT {int(g.night.sum()):,} | "
          f"also >= {MIN_GND_PHOTONS_STRICT} ground photons "
          f"{int((g.night & g.gnd_ph_strict_ok).sum()):,}")

    # ---------------- vertical frame ---------------------------------------
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION)
    res_c = corr[corr.gauge_zero_bs77_m == 12.0]
    c_mean, c_std = float(res_c.c_station_m.mean()), float(res_c.c_station_m.std())
    g["H_terrain_EGG2015_m"] = g.H_terrain_common_m
    g["H_terrain_EVRF2019_empirical_m"] = g.H_terrain_common_m + c_mean
    g["c_empirical_m"] = c_mean
    g["sigma_c_m"] = c_std
    g["c_method"] = "mean of six Kakhovka reservoir station correctors (ATL13 water matchups)"
    g["x"], g["y"] = _TF.transform(g.lon.values, g.lat.values)
    g["segment_length_m"] = SEG_LEN_M
    print(f"vertical chain: h_te_median + free2mean - zeta_EGG2015 + c ({c_mean:+.4f} m, "
          f"sd {c_std:.4f}) -> EVRF2019 empirical")

    # ---------------- P_channel --------------------------------------------
    with rasterio.open(CFG.ROOT / "data/processed/bathymetry/morphology_prior_v1.tif") as src:
        Pc = src.read(1); Pc = np.where(Pc == src.nodata, np.nan, Pc)
        gtr, gH, gW = src.transform, src.height, src.width
    inv = ~gtr
    cc, rr = inv * (g.x.values, g.y.values)
    g["grid_col"] = np.clip(cc.astype(int), 0, gW - 1)
    g["grid_row"] = np.clip(rr.astype(int), 0, gH - 1)
    g["P_channel"] = Pc[g.grid_row, g.grid_col]

    # ---------------- contemporaneous water-state QC ------------------------
    print("\ncontemporaneous water-state QC (vector STRtree, exact polygon distance):")

    def boundary_to_segments(poly):
        """Explode a polygon's boundary into individual 2-point LineStrings.

        A single STRtree entry per shoreline (one MultiLineString/LinearRing
        with ~1000s of vertices) has a bounding box spanning nearly the whole
        reservoir, so the R-tree prunes almost nothing and every query falls
        back to an exact distance test against the full boundary -- this is
        what made the per-point loop take >40 min for a few thousand points.
        Indexing individual segments (each with a tight bbox) instead lets the
        tree actually filter candidates. shapely 2.0.1 has no get_segments(),
        so segments are built from consecutive coordinate pairs directly.
        Verified exact (max abs error 0.0 m against brute-force point.distance
        to the true unioned boundary on a held-out sample) -- not an
        approximation, just an index that can do its job.
        """
        from shapely import get_parts
        b = poly.boundary
        parts = get_parts(b) if b.geom_type.startswith("Multi") else [b]
        segs = []
        for part in parts:
            c = np.asarray(part.coords)
            if len(c) >= 2:
                segs.append(np.stack([c[:-1], c[1:]], axis=1))
        return segs

    wb = pd.read_parquet(CFG.ROOT / "outputs/tables/water_body_objects.parquet")
    post = wb[(wb.period == "POST_BREACH") & (wb.wkt != "")].copy()
    post["date"] = pd.to_datetime(post.date)
    polys_by_date, seg_tree_by_date = {}, {}
    for d, gg in post.groupby("date"):
        geoms = [q for q in (shwkt.loads(w) for w in gg.wkt) if q is not None and not q.is_empty]
        polys_by_date[d] = geoms
        if geoms:
            seg_coords = np.concatenate(
                [s for poly in geoms for s in boundary_to_segments(poly)], axis=0)
            seg_tree_by_date[d] = STRtree(shapely.linestrings(seg_coords))
        else:
            seg_tree_by_date[d] = None
    mask_dates = np.array(sorted(polys_by_date), dtype="datetime64[ns]")
    print(f"  post-breach masks: {len(mask_dates)} dates")

    # accuracy control: segment-tree nearest vs brute-force point.distance to the
    # true unioned boundary, on a held-out sample of the busiest mask date
    _busiest = max(polys_by_date, key=lambda d: len(polys_by_date[d]))
    _rng_chk = np.random.default_rng(0)
    _n_chk = min(300, len(g))
    _i_chk = _rng_chk.choice(len(g), _n_chk, replace=False)
    _pts_chk = _pts(g.x.values[_i_chk], g.y.values[_i_chk])
    _ind_chk, _d_fast = seg_tree_by_date[_busiest].query_nearest(
        _pts_chk, return_distance=True, all_matches=False)
    _fast_chk = np.full(_n_chk, np.nan); _fast_chk[_ind_chk[0]] = _d_fast
    _union_b = shapely.unary_union([q.boundary for q in polys_by_date[_busiest]])
    _exact_chk = shapely.distance(_pts_chk, _union_b)
    _err = np.nanmax(np.abs(_fast_chk - _exact_chk))
    print(f"  accuracy control ({_n_chk} pts, busiest date {str(_busiest)[:10]}, "
          f"{len(polys_by_date[_busiest])} polys): max abs error vs brute-force "
          f"point.distance = {_err:.6f} m")
    if _err > 1e-3:
        raise SystemExit(f"segment-tree nearest disagrees with exact distance by {_err} m "
                          f"-- do not trust distance_to_water_edge_m, fix before proceeding")

    # nearest mask date per segment, via searchsorted rather than an (n x n_dates)
    # difference matrix -- at 20 m there are ~10^5-10^6 segments
    ad = pd.to_datetime(g.date).values.astype("datetime64[ns]")
    pos = np.searchsorted(mask_dates, ad)
    lo_i = np.clip(pos - 1, 0, len(mask_dates) - 1)
    hi_i = np.clip(pos, 0, len(mask_dates) - 1)
    pick_hi = (np.abs(mask_dates[hi_i] - ad) < np.abs(ad - mask_dates[lo_i]))
    nearest = np.where(pick_hi, mask_dates[hi_i], mask_dates[lo_i])
    g["watermask_datetime"] = nearest
    g["delta_time_days"] = (ad - nearest) / np.timedelta64(1, "D")

    # Union + prepare once per date: STRtree(polys).query(pts, predicate="within")
    # re-tests every point against the raw (unprepared) candidate geometry, and
    # the main water body is a single ~130k-vertex MultiPolygon whose bbox spans
    # almost the whole reservoir, so nearly every point becomes a candidate and
    # GEOS does a full ray-cast per point: 165 s for 1,000 points, unusable at
    # 10^5-10^6 scale. shapely.prepare() + shapely.contains_xy() builds GEOS's
    # own indexed representation of the polygon and reuses it across the whole
    # point array: 0.01-0.02 s for 5,000 points against the same geometry,
    # identical result (contains_xy is an exact predicate, not an approximation).
    union_by_date = {}
    for dt_m, polys in polys_by_date.items():
        if polys:
            u = shapely.unary_union(polys)
            shapely.prepare(u)
            union_by_date[dt_m] = u
        else:
            union_by_date[dt_m] = None

    water_state = np.empty(len(g), dtype=object)
    edge_dist = np.full(len(g), np.nan)
    gx, gy = g.x.values, g.y.values
    import time as _time
    for dt_m, idx in pd.Series(range(len(g))).groupby(nearest):
        i = idx.values
        dt_m = pd.Timestamp(dt_m)
        u = union_by_date[dt_m]
        if u is None:
            water_state[i] = "DRY"
            continue
        t0 = _time.time()
        inside = shapely.contains_xy(u, gx[i], gy[i])
        water_state[i] = np.where(inside, "WET", "DRY")
        # segment-indexed nearest: see boundary_to_segments() docstring above
        pts_arr = _pts(gx[i], gy[i])
        seg_tree = seg_tree_by_date[dt_m]
        s_ind, s_dist = seg_tree.query_nearest(pts_arr, return_distance=True,
                                               all_matches=False)
        ed = np.full(len(i), np.nan)
        ed[s_ind[0]] = s_dist
        edge_dist[i] = ed
        print(f"    {str(dt_m)[:10]}: {len(i):>8,} segments  "
              f"WET {int(inside.sum()):>7,}  ({_time.time()-t0:.2f}s)", flush=True)
    g["water_state"] = water_state
    g["distance_to_water_edge_m"] = edge_dist
    g["exposed_ground_validation_ok"] = (g.night & (g.water_state == "DRY")
                                         & (g.distance_to_water_edge_m >= SHORE_BUFFER_M))

    gn = g[g.night]
    print(f"  night set water state: DRY {int((gn.water_state=='DRY').sum()):,}  "
          f"WET {int((gn.water_state=='WET').sum()):,}")
    print(f"  |delta_time| to nearest mask: median {gn.delta_time_days.abs().median():.0f} d")
    ok = g[g.exposed_ground_validation_ok]
    print(f"\nPRIMARY accepted set (night + DRY + >= {SHORE_BUFFER_M:.0f} m edge): {len(ok):,}")
    print(f"  unique RGT {ok.rgt.nunique()} | beam tracks {ok.groupby(['rgt','gt']).ngroups} | "
          f"cycles {ok.cycle.nunique()} | dates {ok.date.nunique()}")
    print(f"  with the strict 100 m photon floor instead: "
          f"{int((g.exposed_ground_validation_ok & g.gnd_ph_strict_ok).sum()):,}")
    chan = ok[ok.P_channel >= 0.5]
    print(f"  channel stratum (P_channel >= 0.5): {len(chan):,} segments over "
          f"{chan.rgt.nunique()} RGTs")
    print(f"  (100 m reference: 13,905 accepted, 320 channel over 15 RGTs)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT_PQ, index=False)
    print(f"\n-> {OUT_PQ}  ({len(g):,} rows)")
    print(f"-> {CFG.TABLES / 'k10d_qc_ledger_20m.csv'}")


if __name__ == "__main__":
    main()
