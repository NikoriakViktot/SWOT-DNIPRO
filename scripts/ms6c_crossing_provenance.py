#!/usr/bin/env python
"""MS6C — where did the v5 S3.3 crossing numbers come from? (diagnostic, report only)

Paper 1 v5 Supplement S3.3 reports SWOT - ICESat-2 crossings whose code and
evidence table no longer exist anywhere. ms6b recomputes them under the rules
S3.1 states and does not reproduce them (30 / 327 against 32 / 343; 3 RiverSP
crossings inside the reservoir outline against "21 inside the former reservoir
footprint"). This script asks one question:

    does ONE coherent rule set reproduce, simultaneously, the v5 counts
    32 / 70 / 241 (= 343), the period split 9 / 6 / 17, the footprint subset
    21, the 8 small-lake pairs, and the published statistics?

It is not a search for a variant that hits 21. Variants are scored through
hierarchical gates and dropped at the first one they fail:

    G1  <=1 d / 1-3 d / 3-10 d = 32 / 70 / 241, RiverSP only. 343 is the
        crossing count of Table S1's EGG-vs-EGM row, so these three bins come
        from one RiverSP table; LakeSP cannot be inside it.
    G2  PRE / TRANSITION / POST = 9 / 6 / 17 (32 - 9 - 17 = 6).
    G3  "former reservoir footprint" = 21 with the SAME source type as G1.
        If 21 needs LakeSP or a different geometry, that is recorded as a
        second provenance path, which is the finding, not a near miss.
    G4  small LakeSP pairs = 8.
    G5  only then medians, NMAD, share within +-10 cm, CI.

Everything is built from node-level pairs (ms6b.node_pairs), matched ONCE with
every node quality and two radii, so each crossing rule is a regrouping, not a
re-match. Inputs are cached on the bulk volume.

Factors: time window (24 h | calendar |ddate| <= 1 | same calendar day),
crossing unit (ovp x pass | ovp x pass x reach | ovp x date), zone assembly
(filter crossings by the union | filter nodes by the union | per-zone concat
without dedup | per-zone concat, exact-row dedup | per-zone, dedup by
ovp x pass), node quality, radius, universe, POST cut-off, footprint geometry,
LakeSP construction (segments in polygon | 500 m cells in polygon).
Per-zone assembly regroups nodes by zone but keeps every ATL13 segment of the
overpass, so a crossing on a zone seam may differ slightly from true per-zone
matching; recorded, not hidden.

Outputs
-------
outputs/paper/tables/ms6c_variants.csv          gate results for every variant
outputs/paper/tables/ms6c_lake_variants.csv
outputs/paper/tables/ms6c_fingerprints.parquet  crossing rows of the best variants
outputs/paper/tables/ms6c_crossing_provenance.md
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import ms6b_swot_icesat_crossings as XS

UTM = CFG.CRS_METRIC
TAB = ROOT / "outputs/paper/tables"
CACHE = CFG.BULK_ROOT / "ms6c_cache"

V5 = dict(n1=32, n13=70, n310=241, pre=9, trans=6, post=17, foot=21, lake_small=8,
          med=2.1, lo=-1.3, hi=5.5, nmad=8.1, w10=66.0,
          pre_med=2.5, pre_nmad=4.4, post_med=3.3, post_nmad=9.5,
          foot_med=2.5, foot_nmad=8.1, lake_med=3.9, lake_nmad=8.0)
BREACH = pd.Timestamp(CFG.BREACH_DATE, tz="UTC")
POST_CUTS = ["2023-07-01", "2023-07-07", "2023-09-01", "2023-09-08"]
RADII = (200.0, 500.0)
OLD = {"Z1": "ZONE_1_KAKHOVKA_LOWER_DNIPRO", "Z2": "ZONE_2_KHERSON_DELTA",
       "Z3": "ZONE_3_DNIPRO_BUG_ESTUARY", "Z4": "ZONE_4_DAM_TO_KHERSON_FLOODWAY"}
UNIVERSES = {"Z1234": ["Z1", "Z2", "Z3", "Z4"], "Z123_master": ["Z1", "Z2", "Z3"],
             "Z14": ["Z1", "Z4"]}
QUALITY = {"q1_dark": lambda p: (p.node_q <= 1) & (p.dark_frac < 0.5),
           "q2_dark": lambda p: (p.node_q <= 2) & (p.dark_frac < 0.5),
           "q1_nodark": lambda p: p.node_q <= 1}
UNITS = {"ovp_pass": ("ovp", "pass_id"), "ovp_pass_reach": ("ovp", "pass_id", "reach_id"),
         "ovp_date": ("ovp", "swot_date")}
ASSEMBLY = ["union_at_crossing", "union_at_node", "perzone_concat",
            "perzone_dedup_exact", "perzone_dedup_ovp_pass"]
WINDOWS = ["h24", "cal1", "sameday"]
VERSIONS = ["all", "latest"]   # every processing version on disk | one per granule
FOOTPRINTS = ["SA2", "Rcore", "Z1", "pool_20230605"]


# ------------------------------------------------------------------ cache --
def geoms() -> dict:
    g = {k: SD.load_utm(v) for k, v in OLD.items()}
    g["SA2"] = SD.load_utm("reservoir_full_pool_prebreach")
    g["Rcore"] = SD.load_utm("R_FORMER_KAKHOVKA_RESERVOIR")
    for L, n in XS.PAPER_ZONES.items():
        g[f"P{L}"] = SD.load_utm(n)
    return g


def pool_20230605():
    """The 5 June 2023 Sentinel-2 pool: largest connected water component of
    the four reservoir tiles, mosaicked at 100 m (mode). Returns a membership
    function and the area, which must land near v5's 2 299 km2."""
    import rasterio
    from rasterio.merge import merge
    from scipy import ndimage
    fs = sorted(XS.S2_MASKS.glob("S2*_20230605T*_water.tif"))
    srcs = [rasterio.open(f) for f in fs]
    arr, tr = merge(srcs, res=100, nodata=255,
                    resampling=rasterio.enums.Resampling.mode)
    for s in srcs:
        s.close()
    w = arr[0] == 1
    lab, n = ndimage.label(w)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    pool = lab == big
    area = pool.sum() * 0.01

    def inside(x, y):
        c = ((np.asarray(x) - tr.c) / tr.a).astype(int)
        r = ((np.asarray(y) - tr.f) / tr.e).astype(int)
        ok = (r >= 0) & (r < pool.shape[0]) & (c >= 0) & (c < pool.shape[1])
        out = np.zeros(len(c), bool)
        out[ok] = pool[r[ok], c[ok]]
        return out
    return inside, area


def load_cached(G) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    fa, fn, fl, fp = (CACHE / f for f in ("atl13.parquet", "nodes_all.parquet",
                                          "lakes.parquet", "pairs.parquet"))
    union = unary_union([G[k] for k in OLD] + [G[f"P{L}"] for L in XS.PAPER_ZONES])
    if not fa.exists():
        print("caching ATL13 ...")
        XS.load_atl13(union).to_parquet(fa)
    if not fn.exists():
        print("caching RiverSP nodes, all qualities ...")
        XS.load_nodes(union, all_quality=True).to_parquet(fn)
    if not fl.exists():
        print("caching LakeSP ...")
        XS.load_lakes(union).to_parquet(fl)
    atl, nodes, lakes = pd.read_parquet(fa), pd.read_parquet(fn), gpd.read_parquet(fl)
    if not fp.exists():
        print("matching node pairs (radii 200/500 m, |dt| <= 11 d) ...")
        pr = XS.node_pairs(atl, nodes, radii=RADII, max_dt_days=11.0)
        pr.to_parquet(fp)
    pairs = pd.read_parquet(fp)
    return atl, nodes, lakes, pairs


def annotate(pairs: pd.DataFrame, nodes: pd.DataFrame, G: dict, pool_in) -> pd.DataFrame:
    n = nodes[["node_q", "dark_frac", "reach_id", "node_id", "crid", "granule"]]
    p = pairs.join(n, on="node_row")
    p["is_latest"] = p.node_row.isin(XS.latest_version(nodes).index)
    p["t"] = pd.to_datetime(p.t, utc=True)
    p["icesat_time"] = pd.to_datetime(p.icesat_time, utc=True)
    p["swot_date"] = p.t.dt.strftime("%Y%m%d")
    for k in list(OLD) + ["SA2", "Rcore"] + [f"P{L}" for L in XS.PAPER_ZONES]:
        p[f"in_{k}"] = shapely.contains_xy(G[k], p.x.values, p.y.values)
    p["in_pool_20230605"] = pool_in(p.x.values, p.y.values)
    return p


# --------------------------------------------------------------- crossings --
def _dday(icesat_time, swot_time):
    return (swot_time.dt.normalize() - icesat_time.dt.normalize()).dt.days.abs()


def window_bins(x: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "h24":
        a = x.dt_days.abs()
        return pd.cut(a, [-1e-9, 1, 3, 10], labels=["<=1", "1-3", "3-10"]).astype(str)
    d = _dday(x.icesat_time, x.swot_time)
    edges = [-1, 1, 3, 10] if rule == "cal1" else [-1, 0, 3, 10]
    return pd.cut(d, edges, labels=["<=1", "1-3", "3-10"]).astype(str)


def crossings(p: pd.DataFrame, G: dict, universe: str, assembly: str, unit: str,
              radius: float, window: str) -> pd.DataFrame:
    zs = UNIVERSES[universe]
    if window == "h24":
        p = p[p.dt_d.abs() <= 10]
    keys = UNITS[unit]
    in_u = p[[f"in_{z}" for z in zs]].any(axis=1)
    if assembly == "union_at_crossing":
        x = XS.group_crossings(p, radius, keys)
        u = unary_union([G[z] for z in zs])
        x = x[shapely.contains_xy(u, x.x.values, x.y.values)]
    elif assembly == "union_at_node":
        x = XS.group_crossings(p[in_u], radius, keys)
    else:
        parts = []
        for z in zs:
            xz = XS.group_crossings(p[p[f"in_{z}"]], radius, keys)
            parts.append(xz.assign(asm_zone=z))
        x = pd.concat(parts, ignore_index=True)
        kc = [c for c in x.columns if c not in ("asm_zone",)]
        if assembly == "perzone_dedup_exact":
            x = x.drop_duplicates(subset=kc)
        elif assembly == "perzone_dedup_ovp_pass":
            x = x.drop_duplicates(subset=["ovp", "swot_pass"] if "swot_pass" in x else ["ovp"])
    x = x.copy()
    x["bin"] = window_bins(x, window)
    return x[x.bin != "nan"]


def stats(d) -> dict:
    d = np.asarray(d, float)
    if not len(d):
        return dict(med=np.nan, nmad=np.nan, w10=np.nan)
    return dict(med=100 * np.median(d), nmad=100 * XS.nmad(d),
                w10=100 * np.mean(np.abs(d) <= 0.10))


def evaluate(x: pd.DataFrame, G: dict, pool_in) -> dict:
    n = x.bin.value_counts()
    r = dict(n1=int(n.get("<=1", 0)), n13=int(n.get("1-3", 0)), n310=int(n.get("3-10", 0)))
    x1 = x[x.bin == "<=1"]
    r.update({f"all_{k}": v for k, v in stats(x1.d_m).items()})
    it = pd.to_datetime(x1.icesat_time, utc=True)
    for cut in POST_CUTS:
        c = pd.Timestamp(cut, tz="UTC")
        tag = cut[5:].replace("-", "")
        pre, post = x1[it < BREACH], x1[it >= c]
        r[f"pre"] = len(pre)
        r[f"post_{tag}"] = len(post)
        r[f"trans_{tag}"] = len(x1) - len(pre) - len(post)
        r[f"post_{tag}_med"] = stats(post.d_m)["med"]
        r[f"post_{tag}_nmad"] = stats(post.d_m)["nmad"]
    r.update({f"pre_{k}": v for k, v in stats(x1[it < BREACH].d_m).items()})
    for fk in FOOTPRINTS:
        m = (pool_in(x1.x.values, x1.y.values) if fk == "pool_20230605"
             else shapely.contains_xy(G[fk], x1.x.values, x1.y.values))
        r[f"foot_{fk}"] = int(m.sum())
        r[f"foot_{fk}_med"] = stats(x1.d_m[m])["med"]
    return r


def gates(r: dict) -> dict:
    g1 = (r["n1"], r["n13"], r["n310"]) == (V5["n1"], V5["n13"], V5["n310"])
    g2_cuts = [c for c in POST_CUTS
               if (r["pre"], r[f"trans_{c[5:].replace('-', '')}"],
                   r[f"post_{c[5:].replace('-', '')}"]) == (V5["pre"], V5["trans"], V5["post"])]
    g3_feet = [f for f in FOOTPRINTS if r[f"foot_{f}"] == V5["foot"]]
    dist1 = (abs(r["n1"] - V5["n1"]) + abs(r["n13"] - V5["n13"])
             + abs(r["n310"] - V5["n310"]))
    return dict(G1=g1, G2_post_cuts=";".join(g2_cuts), G3_footprints=";".join(g3_feet),
                dist_G1=dist1, dist_n1=abs(r["n1"] - V5["n1"]))


# ------------------------------------------------------------------- lakes --
def lake_variants(atl: pd.DataFrame, lakes: gpd.GeoDataFrame, G: dict) -> pd.DataFrame:
    rows = []
    pts = gpd.GeoDataFrame(atl[["ovp", "time", "H", "x", "y"]],
                           geometry=gpd.points_from_xy(atl.x, atl.y), crs=UTM)
    lk = lakes.reset_index(drop=True).reset_index(names="lake_row")
    lk["is_latest"] = lk.index.isin(XS.latest_version(lk).index)
    lk["t"] = pd.to_datetime(lk.t, utc=True)
    j = gpd.sjoin(pts, lk[["lake_row", "obs_id", "t", "H", "area_total", "quality_f",
                           "is_latest", "geometry"]], predicate="within", lsuffix="a", rsuffix="s")
    j["time"] = pd.to_datetime(j.time, utc=True)
    j["dt"] = (j.t - j.time).dt.total_seconds() / 86400.0
    j["dday"] = (j.t.dt.normalize() - j.time.dt.normalize()).dt.days.abs()
    j["cell"] = (j.x // 500).astype(int).astype(str) + "_" + (j.y // 500).astype(int).astype(str)
    cent = lk.set_index("lake_row").geometry.representative_point()
    for universe, zs in UNIVERSES.items():
        u = unary_union([G[z] for z in zs])
        inu = set(cent.index[shapely.contains_xy(u, cent.x.values, cent.y.values)])
        ju = j[j.lake_row.isin(inu)]
        for ver, window, qual, constr in itertools.product(
                VERSIONS, WINDOWS, ("all", "q<=1"), ("segments", "cells500")):
            s = ju if ver == "all" else ju[ju.is_latest]
            s = s[s.dt.abs() <= 1] if window == "h24" else (
                s[s.dday <= 1] if window == "cal1" else s[s.dday == 0])
            if qual == "q<=1":
                s = s[s.quality_f <= 1]
            if constr == "cells500":
                s = s.groupby(["ovp", "lake_row", "cell"]).agg(
                    H_a=("H_a", "median"), H_s=("H_s", "first"),
                    area_total=("area_total", "first")).reset_index()
            g = s.groupby(["ovp", "lake_row"]).agg(
                H_a=("H_a", "median"), H_s=("H_s", "first"),
                area_total=("area_total", "first"))
            d = g.H_s - g.H_a
            small = g.area_total <= XS.LAKE_SMALL_KM2
            rows.append(dict(universe=universe, versions=ver, window=window, quality=qual,
                             construction=constr, n_small=int(small.sum()),
                             n_large=int((~small).sum()),
                             small_med=stats(d[small])["med"], small_nmad=stats(d[small])["nmad"],
                             large_med=stats(d[~small])["med"],
                             large_nmad=stats(d[~small])["nmad"]))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main --
def verify_baseline(p: pd.DataFrame, G: dict) -> str:
    """ms6b's published-to-disk crossings must be reproduced row by row from the
    cached pairs under ms6b's own rule (guards cache + refactor)."""
    ref = pd.read_csv(TAB / "ms6b_swot_icesat_crossings.csv")
    out = []
    q = p[QUALITY["q1_dark"](p)]
    for universe, zs in (("legacy_ZONE_1to4", ["Z1", "Z2", "Z3", "Z4"]),
                         ("paper_RFDE", [f"P{L}" for L in XS.PAPER_ZONES])):
        x = XS.group_crossings(q[q.dt_d.abs() <= 10], 200.0)
        u = unary_union([G[z] for z in zs])
        x = x[shapely.contains_xy(u, x.x.values, x.y.values)]
        r = ref[ref.universe == universe]
        a = x.sort_values(["ovp", "swot_pass"])[["ovp", "swot_pass", "n_nodes", "d_m"]]
        b = r.sort_values(["ovp", "swot_pass"])[["ovp", "swot_pass", "n_nodes", "d_m"]]
        same = (len(a) == len(b)
                and (a.ovp.values == b.ovp.values).all()
                and (a.swot_pass.values == b.swot_pass.values).all()
                and (a.n_nodes.values == b.n_nodes.values).all()
                and np.allclose(a.d_m.values, b.d_m.values, atol=1e-9))
        out.append(f"{universe}: {len(a)} vs {len(b)} rows, identical={same}")
        assert same, f"baseline differs from ms6b for {universe}"
    return "; ".join(out)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    G = geoms()
    pool_in, pool_area = pool_20230605()
    print(f"5 Jun 2023 S2 pool: {pool_area:,.0f} km2 (v5 figure legend: 2 299 km2)")
    atl, nodes, lakes, pairs = load_cached(G)
    p = annotate(pairs, nodes, G, pool_in)
    print(f"node pairs: {len(p):,}")
    base = verify_baseline(p, G)
    print("baseline:", base)

    rows, keep = [], {}
    for (ver, qn, radius, universe, asm, unit, window) in itertools.product(
            VERSIONS, QUALITY, RADII, UNIVERSES, ASSEMBLY, UNITS, WINDOWS):
        pv = p if ver == "all" else p[p.is_latest]
        x = crossings(pv[QUALITY[qn](pv)], G, universe, asm, unit, radius, window)
        r = evaluate(x, G, pool_in)
        vid = f"{ver}|{qn}|r{int(radius)}|{universe}|{asm}|{unit}|{window}"
        rows.append(dict(variant=vid, versions=ver, quality=qn, radius=radius, universe=universe,
                         assembly=asm, unit=unit, window=window, **r, **gates(r)))
        keep[vid] = x
    V = pd.DataFrame(rows).sort_values(["G1", "dist_G1", "dist_n1"],
                                       ascending=[False, True, True])
    V.to_csv(TAB / "ms6c_variants.csv", index=False)
    L = lake_variants(atl, lakes, G)
    L.to_csv(TAB / "ms6c_lake_variants.csv", index=False)

    # fingerprints of every G1 pass, else the 10 closest to G1
    best = V[V.G1].variant.tolist() or V.head(10).variant.tolist()
    fp = []
    for vid in best:
        x = keep[vid]
        x = x.assign(variant=vid)
        for k in ["SA2", "Rcore", "Z1", "Z2", "Z3", "Z4"]:
            x[f"in_{k}"] = shapely.contains_xy(G[k], x.x.values, x.y.values)
        x["in_pool_20230605"] = pool_in(x.x.values, x.y.values)
        fp.append(x)
    pd.concat(fp, ignore_index=True).to_parquet(TAB / "ms6c_fingerprints.parquet")

    report(V, L, base, pool_area)


def report(V: pd.DataFrame, L: pd.DataFrame, base: str, pool_area: float) -> None:
    g1 = V[V.G1]
    lines = ["# MS6C — provenance of the v5 S3.3 crossing numbers", "",
             f"Baseline guard (cached pairs regrouped under the ms6b rule == ms6b tables, row by row): {base}.",
             f"5 June 2023 S2 pool at 100 m: {pool_area:,.0f} km² (v5: 2 299 km²).",
             f"Variants evaluated: {len(V)}; lake variants: {len(L)}.", ""]
    lines += [f"## G1 (32 / 70 / 241 ⇒ 343, RiverSP): {len(g1)} variant(s) pass", ""]
    show = g1 if len(g1) else V.head(10)
    cols = ["variant", "n1", "n13", "n310", "pre", "trans_0901", "post_0901", "trans_0908",
            "post_0908", "foot_SA2", "foot_Rcore", "foot_Z1", "foot_pool_20230605",
            "all_med", "all_nmad", "G2_post_cuts", "G3_footprints"]
    lines += ["" if len(g1) else "No variant passes G1; the ten closest:", "",
              show[cols].round(1).to_markdown(index=False), ""]
    for gate, col in (("G2 (9 / 6 / 17)", "G2_post_cuts"), ("G3 (footprint 21)", "G3_footprints")):
        hit = V[V[col].astype(str).str.len() > 0]
        lines += [f"## {gate}: {len(hit)} variant(s) reproduce it on their own; "
                  f"{int((hit.G1).sum())} of them also pass G1", ""]
        if len(hit):
            lines += [hit[["variant", "n1", "n13", "n310", col]].head(15).to_markdown(index=False), ""]
    lk = L[L.n_small == 8]
    lines += [f"## G4 (8 small-lake pairs): {len(lk)} lake variant(s)", "",
              (lk.round(1).to_markdown(index=False) if len(lk)
               else L.assign(d=(L.n_small - 8).abs()).sort_values("d").head(8)
               .drop(columns="d").round(1).to_markdown(index=False)), ""]
    (TAB / "ms6c_crossing_provenance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
