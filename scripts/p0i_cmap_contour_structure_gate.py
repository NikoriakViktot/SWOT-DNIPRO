#!/usr/bin/env python
"""Reconstruct CMAP2020 back into the isobaths it actually is.

CMAP2020 carries 95,132 points but only 259 distinct depth values, with the
top 20 covering 92.1% of points, and on a map the points trace lines rather
than filling area. That is a digitised-contour signature, so the 95,132 are
NOT 95,132 observations: they are vertices sampled along a much smaller number
of isobaths, and errors along one isobath are correlated.

This gate turns the point cloud back into contour components and measures how
much independent information it actually carries. It does NOT build a model
and does NOT touch the production DEM.

PROVENANCE WORDING. The evidence shows CMAP2020 and the 7,514 primary
soundings share a bathymetric source: 4,584 points coincide within 2 m and the
depth values are identical wherever they coincide (median residual 0.0000 m
and NMAD 0.0000 m at every matching radius from 2 m to 250 m). That proves a
common source or cartographic base -- one survey, or one chart built from it --
not that CMAP2020 is an independent field campaign. It is therefore classed
SAME_SOURCE_BATHYMETRY / DIGITISED_CONTOURS, never RAW_INDEPENDENT_SOUNDINGS.

THE TWO RMSE FIGURES ARE THE SAME TEST. 0.031 m is the <=2 m matched subset
(4,584 points); 0.303 m is the <=100 m subset (7,409). The datum hypothesis is
identical; the spread grows with the matching radius because a 100 m "match"
compares different places on the bed. NMAD stays 0.0000 m throughout.

VERTICAL DATUM, established empirically in p0h:
    H_bed_EVRF2019 = 14.00 (BS-77) + EPSG:9902 correction + value

EFFECTIVE INFORMATION is reported in two senses, and NEITHER is a kriging
weight:
    structural   the number of connected contour components
    spatial      N_eff(Lc) = sum_i max(1, L_i / Lc), swept over correlation
                 lengths 0.5/1/2/5 km so the sensitivity to that assumption
                 is visible rather than hidden in one number

Outputs
-------
outputs/bathymetry_provenance/12_cmap_contour_components.csv
outputs/bathymetry_provenance/13_cmap_effective_information.csv
outputs/bathymetry_provenance/14_cmap_primary_overlap.csv
data/processed/bathymetry/cmap2020_contours_utm.gpkg
outputs/figures/historical_bathymetry/png/p0i_cmap_contours.png
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
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CMAP = Path("/mnt/d/data_RAS_Dnipro/GIS_dnipro/KahovkaRes_CMAP2020_UTM36/"
            "KahovkaRes_CMAP2020_UTM36.shp")
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
OUT = CFG.OUT / "bathymetry_provenance"
GPKG = ROOT / "data/processed/bathymetry/cmap2020_contours_utm.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
REF_BS77 = 14.0
LINK_FACTOR = 3.0          # a gap larger than this x local spacing breaks a line
LINK_MIN, LINK_MAX = 40.0, 400.0
CORR_LENGTHS_KM = (0.5, 1.0, 2.0, 5.0)
OVERLAP_BANDS = (100, 250, 500, 1000, 2000)
MIN_COMPONENT_PTS = 3


def mst_path(xy):
    """Order a component's vertices along its own minimum spanning tree, then
    take the tree diameter. A contour is a line, so its vertices must be
    ordered before a length or a LineString means anything."""
    n = len(xy)
    if n < 2:
        return xy, 0.0
    t = cKDTree(xy)
    k = min(8, n)
    d, idx = t.query(xy, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    w = d[:, 1:].ravel()
    g = coo_matrix((w, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(g).tocoo()
    length = float(mst.sum())
    # tree diameter by two greedy walks
    adj = {}
    for a, b, wt in zip(mst.row, mst.col, mst.data):
        adj.setdefault(a, []).append((b, wt))
        adj.setdefault(b, []).append((a, wt))
    if not adj:
        return xy, length

    def far(src):
        seen, order, stack = {src: 0.0}, [src], [src]
        while stack:
            u = stack.pop()
            for v, wt in adj.get(u, []):
                if v not in seen:
                    seen[v] = seen[u] + wt
                    order.append(v)
                    stack.append(v)
        e = max(seen, key=seen.get)
        return e, seen, order

    a, _, _ = far(next(iter(adj)))
    b, dist_b, order = far(a)
    # walk back from b toward a through decreasing distance
    path = [p for p in order if p in dist_b]
    path.sort(key=lambda p: dist_b[p])
    return xy[path], length


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    GPKG.parent.mkdir(parents=True, exist_ok=True)

    C = gpd.read_file(CMAP)
    cx, cy = C.geometry.x.to_numpy(), C.geometry.y.to_numpy()
    cv = C["field_3"].astype(float).to_numpy()
    print(f"CMAP2020: {len(C):,} points, {len(np.unique(cv)):,} distinct depths")

    P = pd.read_parquet(PRIMARY)
    ptree = cKDTree(np.c_[P.x.to_numpy(), P.y.to_numpy()])
    # EPSG:9902 correction interpolated from the primary points that carry it
    _, pi = ptree.query(np.c_[cx, cy], k=1)
    d9902 = P.delta_epsg9902_m.to_numpy()[pi]
    H_bed = REF_BS77 + d9902 + cv
    print(f"  H_bed_EVRF2019 = 14.00 + d9902 + value -> "
          f"{H_bed.min():.2f} .. {H_bed.max():.2f} m")

    # ------------------------------------------------ contour reconstruction
    print("\n" + "=" * 78)
    print("RECONSTRUCTING CONTOUR COMPONENTS (one isobath can exist in several")
    print("separate places, so components are found per depth value)")
    print("=" * 78)
    rows, geoms = [], []
    for val in np.unique(cv):
        m = cv == val
        if m.sum() < MIN_COMPONENT_PTS:
            continue
        xy = np.c_[cx[m], cy[m]]
        t = cKDTree(xy)
        dd, _ = t.query(xy, k=min(2, len(xy)))
        nn = dd[:, 1] if dd.ndim > 1 and dd.shape[1] > 1 else np.array([LINK_MIN])
        link = float(np.clip(np.median(nn) * LINK_FACTOR, LINK_MIN, LINK_MAX))
        pairs = t.query_pairs(link, output_type="ndarray")
        if len(pairs) == 0:
            continue
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(xy), len(xy)))
        ncomp, lab = connected_components(g, directed=False)
        for c in range(ncomp):
            sel = lab == c
            if sel.sum() < MIN_COMPONENT_PTS:
                continue
            pts, length = mst_path(xy[sel])
            if length <= 0:
                continue
            hb = REF_BS77 + float(np.median(d9902[m][sel])) + val
            closed = (np.hypot(*(pts[0] - pts[-1])) < link)
            rows.append(dict(depth_value=float(val), H_bed_EVRF2019_m=hb,
                             component_id=f"{val:+.2f}_{c:04d}",
                             length_m=length, n_points=int(sel.sum()),
                             mean_spacing_m=length / max(sel.sum() - 1, 1),
                             link_threshold_m=link,
                             closed=bool(closed),
                             east=float(pts[:, 0].mean()),
                             north=float(pts[:, 1].mean())))
            if len(pts) >= 2:
                geoms.append(LineString(pts))
            else:
                geoms.append(MultiPoint(pts).centroid.buffer(1))
    Cm = pd.DataFrame(rows)
    print(f"  {len(Cm):,} contour components from "
          f"{Cm.depth_value.nunique():,} depth values")
    print(f"  total contour length {Cm.length_m.sum()/1000:,.0f} km")
    print(f"  component length: median {Cm.length_m.median():,.0f} m, "
          f"p90 {Cm.length_m.quantile(.9):,.0f} m, "
          f"max {Cm.length_m.max()/1000:,.1f} km")
    print(f"  closed loops: {int(Cm.closed.sum()):,} of {len(Cm):,}")
    Cm.to_csv(OUT / "12_cmap_contour_components.csv", index=False)

    G = gpd.GeoDataFrame(Cm, geometry=geoms, crs=CFG.CRS_METRIC)
    G.to_file(GPKG, layer="cmap2020_contours", driver="GPKG")
    print(f"  -> {GPKG}")

    print("\n  length by depth band:")
    Cm["band"] = pd.cut(Cm.H_bed_EVRF2019_m,
                        [-25, -10, -5, 0, 5, 10, 16],
                        labels=["<-10", "-10..-5", "-5..0", "0..5", "5..10",
                                "10..16"])
    print(Cm.groupby("band", observed=True).agg(
        components=("component_id", "size"),
        km=("length_m", lambda s: s.sum() / 1000)).to_string())

    # ------------------------------------------------ effective information
    print("\n" + "=" * 78)
    print("EFFECTIVE INFORMATION (descriptive only -- never a kriging weight)")
    print("=" * 78)
    eff = [dict(metric="raw_points", value=float(len(C))),
           dict(metric="distinct_depth_values", value=float(Cm.depth_value.nunique())),
           dict(metric="contour_components", value=float(len(Cm))),
           dict(metric="total_length_km", value=float(Cm.length_m.sum() / 1000))]
    for lc in CORR_LENGTHS_KM:
        n_eff = float(np.maximum(1.0, Cm.length_m / (lc * 1000)).sum())
        eff.append(dict(metric=f"N_eff_Lc_{lc}km", value=n_eff))
        print(f"  Lc = {lc:4.1f} km -> N_eff = {n_eff:9,.0f}   "
              f"({n_eff/len(C):.4f} x the raw point count)")
    pd.DataFrame(eff).to_csv(OUT / "13_cmap_effective_information.csv", index=False)
    print(f"\n  raw points {len(C):,} | components {len(Cm):,} | "
          f"depth values {Cm.depth_value.nunique():,}")
    print("  Treating the points as independent would overstate the "
          "information by two to three orders of magnitude.")

    # ------------------------------------------------ overlap with primary
    print("\n" + "=" * 78)
    print("WHERE DOES CMAP ADD GEOMETRY THE 7,514 SOUNDINGS DO NOT HAVE?")
    print("=" * 78)
    ov = []
    for r, geom in zip(Cm.itertuples(), geoms):
        try:
            pts = np.asarray(geom.coords)
        except Exception:
            continue
        step = max(1, len(pts) // 40)
        s = pts[::step]
        dmin, _ = ptree.query(s, k=1)
        ov.append(dict(component_id=r.component_id,
                       H_bed_EVRF2019_m=r.H_bed_EVRF2019_m,
                       length_m=r.length_m,
                       median_dist_to_primary_m=float(np.median(dmin)),
                       min_dist_to_primary_m=float(dmin.min())))
    O = pd.DataFrame(ov)
    O.to_csv(OUT / "14_cmap_primary_overlap.csv", index=False)
    tot = O.length_m.sum() / 1000
    print(f"  total reconstructed contour length: {tot:,.0f} km")
    for b in OVERLAP_BANDS:
        far = O[O.median_dist_to_primary_m > b]
        print(f"    beyond {b:5d} m from any primary sounding: "
              f"{far.length_m.sum()/1000:7,.0f} km "
              f"({100*far.length_m.sum()/O.length_m.sum():5.1f}%), "
              f"{len(far):5,} components")
    deep = O[O.H_bed_EVRF2019_m < -5]
    deep_far = deep[deep.median_dist_to_primary_m > 500]
    print(f"\n  DEEP contours (bed < -5 m): {deep.length_m.sum()/1000:,.0f} km, "
          f"of which {deep_far.length_m.sum()/1000:,.0f} km sit >500 m from any "
          f"primary sounding")
    if deep_far.length_m.sum() / max(deep.length_m.sum(), 1) < 0.05:
        print("  VERDICT: CMAP2020 does NOT add thalweg geometry. Its deep "
              "contours are already covered by the primary soundings, and its "
              "length is concentrated in the shallow margin, not the channel.")
    else:
        print("  CMAP2020 carries deep-channel geometry the sparse primary "
              "set does not reach.")

    # ------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.4))
    a = ax[0]
    vmin, vmax = np.percentile(Cm.H_bed_EVRF2019_m, [2, 98])
    for r, geom in zip(Cm.itertuples(), geoms):
        try:
            c = np.asarray(geom.coords)
        except Exception:
            continue
        col = plt.cm.viridis((r.H_bed_EVRF2019_m - vmin) / max(vmax - vmin, 1e-9))
        a.plot(c[:, 0] / 1000, c[:, 1] / 1000, color=col, lw=0.35)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=a, label="contour bed elevation (m EVRF2019)")
    a.set_title(f"a · {len(Cm):,} reconstructed contour components "
                f"({Cm.length_m.sum()/1000:,.0f} km)\n"
                "from 95,132 points and 259 depth values",
                fontsize=10.2, loc="left")
    a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    a.grid(alpha=0.25)
    a = ax[1]
    far = O[O.median_dist_to_primary_m > 500].set_index("component_id")
    for r, geom in zip(Cm.itertuples(), geoms):
        try:
            c = np.asarray(geom.coords)
        except Exception:
            continue
        isfar = r.component_id in far.index
        a.plot(c[:, 0] / 1000, c[:, 1] / 1000,
               color=RED if isfar else "#c9d4dc", lw=0.5 if isfar else 0.3,
               zorder=3 if isfar else 1)
    a.plot(P.x / 1000, P.y / 1000, ".", ms=0.5, color=INK, alpha=0.55, zorder=2)
    a.set_title("b · red = contour length >500 m from any primary sounding\n"
                "black = the 7,514 primary soundings", fontsize=10.2, loc="left")
    a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    a.grid(alpha=0.25)
    fig.suptitle("p0i · CMAP2020 is isobaths, not 95,132 observations",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "p0i_cmap_contours.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}")
    print("\nSTOP before any soft-contour model. Primary soundings remain the "
          "only HARD observations.")


if __name__ == "__main__":
    main()
