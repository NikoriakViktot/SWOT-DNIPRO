#!/usr/bin/env python
"""HIST 26 — rebuild H1/H2/H3 as CONTINUOUS-FIELD contours, off the staircase.

Roadmap step 2. hist23 built these three contours from a BINARY water mask and
vectorised it with rasterio.features.shapes, so every vertex of
prebreach_contours.gpkg sits on a multiple of 20 m. Gate 7C2's null test found
that, and 12_GATE_7C_ROLE_FREEZE.md forbids using such a polygon as positional
truth. This file replaces the geometry extraction, and ONLY the geometry
extraction.

THE CLASSIFICATION RULE IS NOT TOUCHED. hist23 says in as many words that tuning
the thresholds to improve contour nesting would be fitting the answer, and that
still holds. The frozen rule from swot_dnipro.watermask is

    water = [ (NDWI > t) AND (MNDWI > t) ]  OR  [ SCL == 6 AND NDWI > t - 0.15 ]
            AND NOT SCL in {0,1,3,8,9,10,11}

and the same rule is reproduced here exactly. What changes is that instead of
thresholding it into a binary mask and tracing pixel edges, it is evaluated as a
CONTINUOUS decision field whose zero level IS that boundary:

    W = max( min(NDWI - t, MNDWI - t),  NDWI - t + 0.15  where SCL == 6 )
    F = -W                                    F < 0 is water

Because min() and max() of continuous functions are continuous, the zero level
of F is exactly the set the binary rule would have produced -- located to
sub-pixel precision by Marching Squares rather than snapped to cell edges.

ONE HONEST DISCONTINUITY. The SCL == 6 branch is a discrete switch, so F jumps
across an SCL class edge. Where the shoreline runs along such an edge the
contour is only as good as the 20 m SCL raster there; everywhere else it is
genuinely sub-pixel. That is reported, not hidden.

COMPOSITING IS UNCHANGED. Single dates observe as little as 62% of the
footprint, so hist23 composites near-dates at a stable level with "first valid
observation wins, best cloud first" -- never an average, which would blend
different stages. The same accumulation is used here, on the continuous field.

Outputs
-------
data/processed/bathymetry/prebreach_contours_continuous.gpkg
outputs/tables/hist26_continuous_contours.csv
outputs/figures/historical_bathymetry/png/hist26_contour_comparison.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage.measure import find_contours

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import watermask as WM
from hist25b_gate7_anchored_classifier import CELL, build_grid, clean_water
from hist25b_gate7c_v2_same_date_controls import http_json

INK, BLUE, RED, AMBER, GREEN, GREY = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
GPKG_OLD = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
GPKG_NEW = ROOT / "data/processed/bathymetry/prebreach_contours_continuous.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
CACHE = CFG.BULK_ROOT / "hist26_continuous_fields"

# Date groups copied verbatim from hist23; changing them would change WHICH
# water is being contoured, not how its boundary is located.
CONTOURS = {
    "H1": dict(dates=["2023-06-05"], role="constraint already in use"),
    "H2": dict(dates=["2019-02-19", "2019-02-24", "2019-03-03", "2019-03-18",
                      "2019-03-16", "2019-03-21", "2019-03-06", "2019-03-13"],
               role="strongest flatness control (6 gauges, spread 0.08 m)"),
    "H3": dict(dates=["2023-02-23", "2023-02-20", "2023-02-10"],
               role="largest vertical separation, near UNS"),
}
MIN_PART_KM2 = 0.05


def fetch_field(date, G, tok):
    """The frozen water rule as a CONTINUOUS field, on the analysis grid.

    Returns (W, valid). W > 0 is water by the frozen rule; its zero level is
    that rule's own boundary."""
    q = {"collections": ["sentinel-2-l2a"], "bbox": G["bbox_ll"], "limit": 60,
         "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z"}
    feats = http_json(STAC, q)["features"]
    if not feats:
        return None, None
    shp = (G["ny"], G["nx"])
    acc = {b: np.full(shp, np.nan, np.float32) for b in ("B03", "B08", "B11")}
    scl = np.zeros(shp, np.uint8)
    for f in sorted(feats, key=lambda x: x["properties"].get("eo:cloud_cover", 100)):
        for band in ("B03", "B08", "B11", "SCL"):
            if band not in f["assets"]:
                continue
            try:
                with rasterio.open(f["assets"][band]["href"] + "?" + tok) as ds:
                    a = ds.read(1, window=from_bounds(G["x0"], G["y0"], G["x1"],
                                                      G["y1"], ds.transform),
                                out_shape=shp,
                                resampling=(Resampling.nearest if band == "SCL"
                                            else Resampling.bilinear),
                                boundless=True, fill_value=0)
            except Exception as ex:
                print(f"      {band} read failed: {type(ex).__name__}")
                continue
            m = a > 0
            if band == "SCL":
                scl[m] = a[m].astype(np.uint8)
            else:
                acc[band][m] = a[m].astype(np.float32)
    g, n, s = acc["B03"], acc["B08"], acc["B11"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (g - n) / (g + n)
        mndwi = (g - s) / (g + s)
    t = float(WM.DEFAULT_NDWI)
    tm = float(WM.DEFAULT_MNDWI)
    both = np.minimum(ndwi - t, mndwi - tm)
    relaxed = np.where(scl == 6, ndwi - (t - 0.15), -np.inf)
    W = np.maximum(both, relaxed).astype(np.float32)
    valid = (np.isfinite(ndwi) & np.isfinite(mndwi)
             & ~np.isin(scl, WM.SCL_REJECT))
    W[~valid] = np.nan
    return W, valid


def rings_to_polygon(rings, F, G):
    """Assemble Marching-Squares rings into a polygon, deciding by FIELD SIGN.

    An earlier version classified each ring as shell or hole by counting how
    many larger rings contained it. That is unsound for these rings, and
    measurably so: a Marching-Squares ring around a convoluted shoreline
    self-touches, and the shoelace area then CANCELS sub-loops of opposite
    orientation while buffer(0) resolves them into real coverage. Ranking by
    shoelace area therefore ordered the rings wrongly, and the assembled H1
    came out at 3,112 km2 -- larger than the 2,340 km2 analysis domain that
    contains it, which is the impossibility that exposed the bug.

    Polygonising the linework and then asking the FIELD whether each resulting
    face is water needs no orientation, no area ranking and no containment
    tests: a face is kept when F < 0 at its representative point. Islands,
    lakes on islands and every deeper nesting fall out correctly."""
    from shapely.ops import polygonize
    lines = []
    for r in rings:
        if len(r) < 4:
            continue
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[0]])      # polygonize needs closed linework
        lines.append(LineString(r))
    faces = list(polygonize(unary_union(lines)))
    if not faces:
        return None
    keep = []
    for f in faces:
        p = f.representative_point()
        j = int((p.x - G["x0"]) / CELL)
        i = int((G["y1"] - p.y) / CELL)
        if 0 <= i < F.shape[0] and 0 <= j < F.shape[1] and F[i, j] < 0:
            keep.append(f)
    if not keep:
        return None
    return unary_union(keep)


def contour_geometry(F, valid, G, region, frac=0.8):
    """Sub-pixel contour of F = 0, keeping only rings that belong to the body.

    Whole rings are selected, never trimmed: a ring is kept when at least
    `frac` of its vertices fall inside `region` (the dilated, hole-filled main
    water body). That keeps the outer shoreline and every island inside it,
    and drops distant ponds and the contour that traces the domain edge --
    without cutting any ring open, which would break polygon assembly."""
    rings, lines = [], []
    for c in find_contours(np.nan_to_num(F, nan=1e6), 0.0):
        if len(c) < 4:
            continue
        x = G["x0"] + (c[:, 1] + 0.5) * CELL
        y = G["y1"] - (c[:, 0] + 0.5) * CELL
        ri = np.clip(np.round(c[:, 0]).astype(int), 0, F.shape[0] - 1)
        ci = np.clip(np.round(c[:, 1]).astype(int), 0, F.shape[1] - 1)
        if valid[ri, ci].mean() < frac or region[ri, ci].mean() < frac:
            continue
        rings.append(np.c_[x, y])
        lines.append(LineString(np.c_[x, y]))
    return rings, lines


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("HIST 26 — continuous-field contours for H1/H2/H3")
    print("=" * 78)
    print(f"  git {commit}")
    print("  the water RULE is unchanged; only the boundary extraction is")

    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    G["bbox_ll"] = [round(v, 5) for v in
                    gpd.GeoSeries([fp], crs=32636).to_crs(4326).iloc[0].bounds]
    inside = G["inside"]
    tok = http_json(SAS, timeout=90)["token"]

    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    gg = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = gg.groupby("date").transformed_level_m.median()

    old = gpd.read_file(GPKG_OLD, layer="contour_polygons").set_index("contour_id")
    rows, polys, shorelines = [], {}, {}
    for cid, spec in CONTOURS.items():
        print(f"\n  {cid}: {spec['role']}")
        cf = CACHE / f"{cid}_field.npz"
        if cf.exists():
            z = np.load(cf, allow_pickle=True)
            Wacc, vacc, used = z["W"], z["valid"], [str(u) for u in z["used"]]
            print(f"    cached composite: {len(used)} dates")
        else:
            Wacc = np.full(inside.shape, np.nan, np.float32)
            vacc = np.zeros(inside.shape, bool)
            used = []
            for d in spec["dates"]:
                Wd, vd = fetch_field(d, G, tok)
                if Wd is None:
                    print(f"    {d}: no S2 items")
                    continue
                new = vd & ~vacc & inside
                gain = float(new.sum()) / inside.sum()
                # first valid observation wins -- never an average, which would
                # blend two different stages into one contour
                Wacc[new] = Wd[new]
                vacc |= vd
                if gain > 0.001:
                    used.append(d)
                cov = (vacc & inside).sum() / inside.sum()
                print(f"    {d}: +{100*gain:5.1f}% newly observed "
                      f"(cumulative {100*cov:5.1f}%)")
                if cov > 0.995:
                    print("    -> coverage complete")
                    break
            np.savez_compressed(cf, W=Wacc, valid=vacc,
                                used=np.array(used, dtype="U10"))
        obs = float((vacc & inside).sum()) / inside.sum()
        hs = [float(lev[d]) for d in used if d in lev.index]
        H = float(np.mean(hs)) if hs else np.nan
        spread = float(np.max(hs) - np.min(hs)) if len(hs) > 1 else 0.0

        F = -Wacc
        F[~(vacc & inside)] = np.nan
        wet = clean_water((F < 0) & vacc & inside)
        lab, n = ndimage.label(wet, structure=np.ones((3, 3), int))
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        conn = lab == (int(np.argmax(sizes)) + 1)
        # contour only the main body's neighbourhood, so distant ponds do not
        # become part of the reservoir polygon
        # Contour the FULL field and select whole rings afterwards. Masking F
        # to a band first was wrong: the field becomes nodata inside the band's
        # inner edge, which reads as land, so the band's own inner edge came
        # back as a contour. That inflated H1 from 2,130 to 3,112 km2 while the
        # perimeter fell correctly from 1,543 to 1,168 km -- the boundary was
        # right and the ring assembly was not.
        k = np.ones((3, 3), bool)
        region = ndimage.binary_dilation(ndimage.binary_fill_holes(conn), k, 3)
        rings, lines = contour_geometry(F, vacc & inside, G, region)
        poly = rings_to_polygon(rings, F, G)
        if poly is None:
            print("    no closed contour -- skipped")
            continue
        polys[cid] = poly
        shorelines[cid] = unary_union(lines)

        allxy = np.vstack(rings)
        onmul = float(np.mean((np.mod(allxy[:, 0], CELL) < 1e-6)
                              & (np.mod(allxy[:, 1], CELL) < 1e-6)))
        oldpoly = old.loc[cid, "geometry"]
        print(f"    level used   {H:.4f} m (spread {spread:.3f} m over "
              f"{len(used)} dates), observed {100*obs:.1f}%")
        print(f"    polygon      {poly.area/1e6:,.1f} km2, boundary "
              f"{poly.boundary.length/1e3:,.0f} km, {len(rings):,} rings")
        print(f"    was (stairs) {oldpoly.area/1e6:,.1f} km2, boundary "
              f"{oldpoly.boundary.length/1e3:,.0f} km")
        print(f"    vertices on a 20 m multiple: {100*onmul:.2f}% "
              f"(the staircase version was 100%)")
        rows.append(dict(contour_id=cid, role=spec["role"],
                         dates="|".join(used), n_dates=len(used),
                         H_evrf2019_m=H, level_spread_m=spread,
                         observed_fraction=obs,
                         area_km2=poly.area / 1e6,
                         boundary_km=poly.boundary.length / 1e3,
                         n_rings=len(rings),
                         frac_vertices_on_grid_multiple=onmul,
                         area_km2_staircase=oldpoly.area / 1e6,
                         boundary_km_staircase=oldpoly.boundary.length / 1e3,
                         grid_cell_m=CELL))

    if not rows:
        raise SystemExit("no contour produced")
    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "hist26_continuous_contours.csv", index=False)

    GPKG_NEW.parent.mkdir(parents=True, exist_ok=True)
    if GPKG_NEW.exists():
        GPKG_NEW.unlink()
    gpd.GeoDataFrame(R[["contour_id", "H_evrf2019_m", "role", "n_dates"]],
                     geometry=[polys[c] for c in R.contour_id],
                     crs=CFG.CRS_METRIC).to_file(
        GPKG_NEW, layer="contour_polygons", driver="GPKG")
    gpd.GeoDataFrame(R[["contour_id", "H_evrf2019_m"]],
                     geometry=[shorelines[c] for c in R.contour_id],
                     crs=CFG.CRS_METRIC).to_file(
        GPKG_NEW, layer="shorelines", driver="GPKG")
    print(f"\n-> {GPKG_NEW}")

    verify(R)
    figure(R, polys, old)
    print("\nSTOP. Next: re-measure the migration rate as a local k(x).")


def verify(R):
    """The staircase test, run in reverse on the new geometry."""
    print("\n  VERIFY — the defect this rebuild exists to remove:")
    bad = R[R.frac_vertices_on_grid_multiple > 0.02]
    for r in R.itertuples():
        print(f"    {r.contour_id}: {100*r.frac_vertices_on_grid_multiple:.2f}% "
              f"of vertices on a 20 m multiple  "
              f"{'STILL A STAIRCASE' if r.frac_vertices_on_grid_multiple > 0.02 else 'OK'}")
    if len(bad):
        raise SystemExit("the rebuilt contour is still snapped to the grid")
    print("    nesting (each contour must contain the one below it):")
    for a, b in (("H3", "H2"), ("H2", "H1")):
        if a in set(R.contour_id) and b in set(R.contour_id):
            ra = R[R.contour_id == a].iloc[0]
            rb = R[R.contour_id == b].iloc[0]
            ok = rb.area_km2 > ra.area_km2
            print(f"      area({b}) {rb.area_km2:,.1f} > area({a}) "
                  f"{ra.area_km2:,.1f} km2  {'OK' if ok else 'VIOLATED'}")


def figure(R, polys, old):
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    cols = {"H1": AMBER, "H2": GREEN, "H3": BLUE}
    for cid, c in cols.items():
        if cid not in polys:
            continue
        gpd.GeoSeries([polys[cid]]).boundary.plot(ax=ax[0], color=c, lw=.6)
    ax[0].set_title("rebuilt continuous-field contours", color=INK)
    ax[0].set_xlabel("easting (m)"); ax[0].set_ylabel("northing (m)")
    ax[0].set_aspect("equal")
    h = [plt.Line2D([], [], color=v, label=k) for k, v in cols.items()]
    ax[0].legend(handles=h, fontsize=9)

    x = np.arange(len(R))
    ax[1].bar(x - .2, R.area_km2_staircase, .4, color=GREY, label="staircase")
    ax[1].bar(x + .2, R.area_km2, .4, color=BLUE, label="continuous")
    for i, r in enumerate(R.itertuples()):
        d = r.area_km2 - r.area_km2_staircase
        ax[1].text(i, max(r.area_km2, r.area_km2_staircase) + 12,
                   f"{d:+.1f} km²", ha="center", fontsize=9, color=INK)
    ax[1].set_xticks(x); ax[1].set_xticklabels(R.contour_id)
    ax[1].set_ylabel("polygon area (km²)")
    ax[1].set_title("area change from removing the staircase", color=INK)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("hist26 · same water rule, sub-pixel boundary", color=INK,
                 fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "hist26_contour_comparison.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
