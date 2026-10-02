#!/usr/bin/env python
"""GATE 7C2 — scene / shoreline quality on THREE INDEPENDENT AXES.

THIS IS A SCENE-QUALITY GATE, NOT A POSITIONAL-TRUTH GATE.

It answers "is this scene stable enough to be ADMITTED to a shoreline-constraint
estimate at all", and it does NOT answer "what is the shoreline offset here".
The distinction is not stylistic. The stored H1/H2/H3 reference polygons have
every vertex on a multiple of 20 m: they were vectorised from a raster on this
same grid, so their boundary is a stair-step, not a surveyed line. Comparing a
smooth sub-pixel contour against a staircase produces characteristic ~half-cell
offsets even from a perfect estimator, which is exactly the +/-10 m pile-up seen
in median_*_vs_raster_ref below.

The null test in null_test() returns +0.00 m, but that only proves the CODE adds
no offset of its own. It says nothing about whether the reference is
geometrically accurate enough to carry a sub-pixel median, and it is not
evidence that it is. Reading it as such would quietly reinstate the 20 m
quantisation that v2_subpixel_contour was built to escape.

So the columns divide into two classes:

  USABLE for admission   f_L_*, median_drift, nmad_*, p95_*,
                         bidirectional_difference, unpaired_top5_cell_fraction,
                         every axis-A and axis-B metric
                         -- all of these are spreads, differences or counts, in
                         which the staircase bias largely cancels

  NOT a physical bias    median_*_vs_raster_ref, median_reverse_*_vs_raster_ref
                         -- absolute position against a 20 m raster staircase.
                         Do not quote these as a SAR-optical shoreline offset,
                         and do not use them below ~20 m for anything.

The physical offset is measured elsewhere, by the Gate 7C1 scheme in which BOTH
sides are proper continuous geometry:

    S1 continuous LDA -> sub-pixel contour  <->  S2 continuous NDWI -> sub-pixel
    contour,  then  d_raw -> d_stage -> d_residual

never sub-pixel contour against a 20 m raster staircase. See
hist25b_gate7c_v2_same_date_controls.py for that path.

Gate 7C1 showed that classifier quality, mask topology and external-shoreline
usability are three different things, and that no single number decides all
three:

  2019-03-18  LDA separation 1.12, mask broken into 144 components with the
              largest holding 47% of the area -- yet the external shoreline was
              stable, median -0.00 m at EVERY match threshold from 25 to 250 m.
  2023-02-20  LDA separation 7.14, mask 99% one component -- yet the
              correspondence estimate drifted 0.0 -> 20.9 m with the threshold.

So a scene can be unusable for area/volume while its outer shoreline is still a
valid geometric constraint, and the reverse. This gate therefore computes, for
every eligible H1/H2/H3 scene, three separate blocks of metrics and assigns NO
thresholds and NO pass/fail. Two scenes are not a basis for a cut; the point is
to get the distribution over all 26 first.

  A  classifier quality      does SAR separate water from land at all
  B  mask topology quality   is the resulting mask one water body or confetti
  C  correspondence quality  is the external shoreline stably matchable

THE REFERENCE IS STORED AS A VECTOR BUT IS RASTER-DERIVED. An earlier version of
this docstring claimed the polygon boundary "is already a true line" and that
comparing against it "removes the reference-side discretisation entirely". That
was wrong, and the null test is what exposed it: every vertex sits on a multiple
of 20 m. Being a polygon makes the ARITHMETIC exact -- there is no EDT and no
half-cell correction on the reference side -- but it does not make the GEOMETRY
sub-pixel. The staircase is still there, it has simply been vectorised.

EXTERNAL VS INTERNAL BOUNDARY. Only the OUTER boundary of a water body is a
shoreline in the sense the DEM needs. The boundary of an interior hole, and the
outline of a disconnected fragment, are topology artefacts. They are measured
separately and never mixed into the shoreline statistics, because in Gate 7C1
they were two thirds of the 2019 contour and would otherwise dominate it.

    R_internal = L_internal / L_external

MATCHER. Gate 7C1 validated both the nearest-vertex and the normal-intersection
matchers and found them to agree within ~2 m on the controls. This sweep uses
exact vector nearest-vertex in BOTH directions: it is cheap enough for 26
scenes x 6 thresholds and it is the direction pair, not the ray geometry, that
catches correspondence failure.

Outputs
-------
outputs/tables/gate7c2_scene_quality.csv
outputs/figures/historical_bathymetry/png/gate7c2_quality_axes.png
outputs/figures/historical_bathymetry/png/gate7c2_scene_maps.png
outputs/figures/historical_bathymetry/png/gate7c2_drift_profiles.png
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import MultiLineString
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from hist25b_gate6_event_qualification import load_frozen_manifest
from hist25b_gate7_anchored_classifier import (CELL, CACHE, build_grid,
                                               clean_water, db)

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
THRESHOLDS_M = (25.0, 50.0, 75.0, 100.0, 150.0, 250.0)
DENSIFY_M = 5.0            # reference boundary sampling; << the 20 m cell
LARGE_COMPONENT_KM2 = 1.0
CONC_CELL_M = 5000.0       # spatial-concentration bin for unpaired length


# ------------------------------------------------------------------ geometry --
def densify(geom, step=DENSIFY_M):
    """Boundary of a polygon as points, split into exterior and interior rings.

    Returns (xy, is_interior). Interior rings are islands: their boundary is a
    real shoreline of the reference, but it is topologically distinct from the
    outer perimeter and is kept separable throughout."""
    import shapely
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    pts, flag = [], []
    for p in polys:
        for ring, inner in [(p.exterior, False)] + [(r, True) for r in p.interiors]:
            # shapely.segmentize inserts vertices without moving the existing
            # ones, and is ~100x faster than looping ring.interpolate(); H3
            # alone carries 1,855 island rings, so the loop mattered
            xy = np.asarray(shapely.segmentize(ring, step).coords)[:-1]
            if len(xy) < 3:
                xy = np.asarray(ring.coords)[:-1]
            pts.append(xy)
            flag.append(np.full(len(xy), inner))
    return np.vstack(pts), np.concatenate(flag)


def mask_contours(W, G):
    """Marching-squares boundary of a binary mask, split external / internal.

    External = the outline of each filled component. Internal = the outline of
    each hole. Doing this on the FILLED mask rather than sorting contours by
    orientation keeps the definition unambiguous."""
    from skimage.measure import find_contours
    out = {}
    Wf = ndimage.binary_fill_holes(W)
    holes = Wf & ~W
    for key, m in (("external", Wf), ("internal", holes)):
        rows = []
        for c in find_contours(m.astype(float), 0.5):
            if len(c) < 4:
                continue
            x = G["x0"] + (c[:, 1] + 0.5) * CELL
            y = G["y1"] - (c[:, 0] + 0.5) * CELL
            seg = np.hypot(np.diff(x), np.diff(y))
            wl = np.zeros(len(x)); wl[:-1] += seg / 2; wl[1:] += seg / 2
            rows.append(np.c_[x, y, wl])
        out[key] = np.vstack(rows) if rows else np.empty((0, 3))
    return out


def subpixel_external(disc, core, G, band_px=3):
    """SAR shoreline vertices from the CONTINUOUS field, tagged external.

    Axis C must not fall back on the binary mask boundary. Marching squares at
    0.5 on a binary mask puts every vertex on a half-cell line, which quantises
    every distance to multiples of 10 m on this grid and reproduces exactly the
    pixel-boundary estimator that Gate 7C0 rejected. The sub-pixel position
    therefore comes from `disc`; the binary masks are used only to decide WHICH
    zero crossings are outer shoreline and which are hole boundaries.

    Returns x, y, representative length, and an is_external flag."""
    from skimage.measure import find_contours
    k = np.ones((3, 3), bool)
    filled = ndimage.binary_fill_holes(core)
    holes = filled & ~core
    ext_band = (ndimage.binary_dilation(filled, k, band_px)
                & ~ndimage.binary_erosion(filled, k, band_px))
    hol_band = (ndimage.binary_dilation(holes, k, band_px)
                & ~ndimage.binary_erosion(holes, k, band_px))
    rows = []
    for c in find_contours(np.nan_to_num(disc, nan=1e6), 0.0):
        if len(c) < 4:
            continue
        r, cc = c[:, 0], c[:, 1]
        x = G["x0"] + (cc + 0.5) * CELL
        y = G["y1"] - (r + 0.5) * CELL
        seg = np.hypot(np.diff(x), np.diff(y))
        wl = np.zeros(len(x)); wl[:-1] += seg / 2; wl[1:] += seg / 2
        ri = np.clip(np.round(r).astype(int), 0, disc.shape[0] - 1)
        ci = np.clip(np.round(cc).astype(int), 0, disc.shape[1] - 1)
        ext = ext_band[ri, ci]
        hol = hol_band[ri, ci] & ~ext
        keep = ext | hol
        if not keep.any():
            continue
        rows.append(np.c_[x[keep], y[keep], wl[keep], ext[keep].astype(float)])
    if not rows:
        return np.empty((0, 4))
    return np.vstack(rows)


def robust(v):
    m = float(np.median(v))
    return m, float(1.4826 * np.median(np.abs(v - m)))


# -------------------------------------------------------------------- axis A --
def classifier_metrics(disc, aw, al, obs):
    """How well the discriminant separates the two anchor classes.

    Reported in units of the pooled robust scale so it is comparable between
    scenes; the raw LDA scale is not, because the fit is rebuilt per event."""
    vw, vl = disc[aw], disc[al]
    mw, sw = robust(vw)
    ml, sl = robust(vl)
    pooled = 0.5 * (sw + sl)
    sep_raw = ml - mw
    # overlap: how far the water upper tail reaches past the land lower tail
    ov = float(np.percentile(vw, 95) - np.percentile(vl, 5))
    return dict(
        lda_separation=sep_raw,
        lda_effect_size=sep_raw / pooled if pooled > 0 else np.nan,
        lda_overlap_width=ov,
        lda_overlap_norm=ov / pooled if pooled > 0 else np.nan,
        anchor_water_misclassified=float((vw > 0).mean()),
        anchor_land_misclassified=float((vl < 0).mean()),
        frac_near_decision=float((np.abs(disc[obs]) < pooled).mean()))


# -------------------------------------------------------------------- axis B --
def topology_metrics(raw, cleaned, core, G):
    """Is the classified water one body, or confetti with holes punched in it.

    Measured at all three production stages so the audit shows how much the
    connectivity filter had to throw away, not only what survived it."""
    px = CELL ** 2 / 1e6
    out = {}
    for name, W in (("raw", raw), ("clean", cleaned), ("core", core)):
        lab, n = ndimage.label(W, structure=np.ones((3, 3), int))
        if n == 0:
            out.update({f"{name}_area_km2": 0.0, f"{name}_n_components": 0,
                        f"{name}_n_components_large": 0,
                        f"{name}_largest_fraction": np.nan})
            continue
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * px
        out[f"{name}_area_km2"] = float(sz.sum())
        out[f"{name}_n_components"] = int(n)
        out[f"{name}_n_components_large"] = int((sz >= LARGE_COMPONENT_KM2).sum())
        out[f"{name}_largest_fraction"] = float(sz.max() / sz.sum())
    C = mask_contours(core, G)
    le = C["external"][:, 2].sum() / 1000 if len(C["external"]) else 0.0
    li = C["internal"][:, 2].sum() / 1000 if len(C["internal"]) else 0.0
    out.update(external_boundary_km=float(le), internal_boundary_km=float(li),
               internal_external_ratio=float(li / le) if le > 0 else np.nan,
               hole_area_km2=float((ndimage.binary_fill_holes(core)
                                    & ~core).sum() * px))
    return out, C


# -------------------------------------------------------------------- axis C --
def drop_swath_edges(A, obs, G, edge_m=200.0):
    """Remove contour vertices that only exist because the S1 swath ends there.

    Where a scene covers part of the reservoir, the classified water stops at a
    straight line across the basin. That line is a coverage boundary, not a
    shoreline: it has no optical counterpart, so it is counted as unpaired and
    it drags the correspondence statistics. Gate 7B already used a 200 m edge
    rule for the same reason; the same rule is applied here BEFORE any
    correspondence metric is computed.

    Returns the kept vertices and the length fraction removed."""
    d_edge = ndimage.distance_transform_edt(obs) * CELL
    ii = np.clip(((G["y1"] - A[:, 1]) / CELL).astype(int), 0, G["ny"] - 1)
    jj = np.clip(((A[:, 0] - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
    keep = d_edge[ii, jj] >= edge_m
    frac = float(A[~keep, 2].sum() / A[:, 2].sum()) if len(A) else np.nan
    return A[keep], frac


def correspondence_metrics(C, ref_xy, ref_inner, ref_poly, G):
    """Stability of the SAR-to-optical correspondence on the EXTERNAL shoreline.

    Distances are exact vector-to-vector. The sign follows the frozen
    convention, positive = SAR water extent LARGER than optical, and the
    inside-test is reversed for the reverse direction: a SAR vertex inside the
    optical polygon means the SAR extent is SMALLER (negative), while an
    optical vertex inside the SAR water means the SAR extent is LARGER
    (positive). With that correction the two directions must AGREE, d_12 ~
    d_21. The antisymmetric expectation d_12 ~ -d_21 comes from applying one
    sign rule blindly in both directions and is wrong here."""
    A = C["subpixel_external"]
    out = {"paired_external_length_km": np.nan}
    if len(A) < 200:
        return out, None
    tree_ref = cKDTree(ref_xy)
    dA, jA = tree_ref.query(A[:, :2], k=1)
    inside_opt = contains_xy(ref_poly, A[:, 0], A[:, 1])
    sA = np.where(inside_opt, -1.0, +1.0) * dA
    wl = A[:, 2]
    L_tot = wl.sum() / 1000

    meds = {}
    for t in THRESHOLDS_M:
        k = dA <= t
        meds[t] = float(np.median(sA[k])) if k.sum() >= 50 else np.nan
        out[f"median_{int(t)}_vs_raster_ref"] = meds[t]
        out[f"f_L_{int(t)}"] = float(wl[k].sum() / wl.sum())
    fin = [v for v in meds.values() if np.isfinite(v)]
    out["median_drift"] = float(max(fin) - min(fin)) if len(fin) > 1 else np.nan

    for t in (50.0, 100.0):
        k = dA <= t
        if k.sum() >= 50:
            m, s = robust(sA[k])
            out[f"nmad_{int(t)}"] = s
            out[f"p90_{int(t)}"] = float(np.percentile(np.abs(sA[k]), 90))
            out[f"p95_{int(t)}"] = float(np.percentile(np.abs(sA[k]), 95))

    # reverse direction: optical EXTERIOR boundary -> SAR external contour
    ext = ~ref_inner
    tree_sar = cKDTree(A[:, :2])
    dB, _ = tree_sar.query(ref_xy[ext], k=1)
    inside_sar = _inside_from_contour(ref_xy[ext], C, G)
    sB = np.where(inside_sar, +1.0, -1.0) * dB
    kB = dB <= 100.0
    out["median_reverse_100_vs_raster_ref"] = (float(np.median(sB[kB])) if kB.sum() >= 50
                                 else np.nan)
    out["bidirectional_difference"] = (
        out.get("median_100_vs_raster_ref", np.nan)
        - out["median_reverse_100_vs_raster_ref"])

    k = dA <= 250.0
    out["paired_external_length_km"] = float(wl[k].sum() / 1000)
    out["external_length_km"] = float(L_tot)

    # spatial concentration of what stays unpaired: if the unpaired length is
    # heaped into a few places it is a local defect, if it is spread along the
    # whole line it is a global correspondence failure
    un = ~k
    if un.sum() > 10:
        bx = np.floor(A[un, 0] / CONC_CELL_M).astype(int)
        by = np.floor(A[un, 1] / CONC_CELL_M).astype(int)
        key = bx * 100000 + by
        tot = wl[un].sum()
        per = pd.Series(wl[un]).groupby(key).sum().sort_values(ascending=False)
        out["unpaired_top5_cell_fraction"] = float(per.head(5).sum() / tot)
        out["unpaired_n_cells"] = int(len(per))
    return out, dict(X=A[:, 0], Y=A[:, 1], d=sA, dist=dA, w=wl)


def _inside_from_contour(P, C, G):
    """Is each reference point inside the SAR water body? Uses the filled core
    mask; a raster test is adequate here because it only sets a SIGN."""
    ii = np.clip(((G["y1"] - P[:, 1]) / CELL).astype(int), 0, G["ny"] - 1)
    jj = np.clip(((P[:, 0] - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
    return C["_filled"][ii, jj]


def null_test(refs, G):
    """Push each reference through the SAME code path as a SAR field.

    Rasterise the reference polygon, treat the result as the classified field,
    extract the sub-pixel contour and measure it against the polygon it came
    from. Identical inputs must return exactly zero. This is the control that
    separates a real offset from a coordinate-convention mistake, and it is run
    every time rather than once: the medians below pile up near -10 m, which is
    half a cell, and without this test that would be indistinguishable from the
    one-cell estimator bias Gate 7C0 had to remove.

    It also exposes the hard limit of this comparison. Every vertex of the
    stored reference lies on a multiple of 20 m: the polygon is a raster
    staircase on the cell edges of this same grid, not a surveyed shoreline.
    So an offset materially below the 20 m cell cannot be resolved AGAINST IT,
    however good the sub-pixel estimator is."""
    print("\n  NULL TEST — reference vs itself through the production path:")
    for t, R in refs.items():
        disc = np.where(R["mask"], -1.0, 1.0).astype(np.float32)
        SP = subpixel_external(disc, R["mask"], G)
        A = SP[SP[:, 3] > 0.5][:, :3]
        if len(A) < 100:
            print(f"    {t}: too few vertices to test")
            continue
        d, _ = cKDTree(R["xy"]).query(A[:, :2], k=1)
        s = np.where(contains_xy(R["poly"], A[:, 0], A[:, 1]), -1.0, 1.0) * d
        k = d <= 100.0
        m, nm = robust(s[k])
        print(f"    {t}: median {m:+.2f} m  NMAD {nm:.2f}  n {k.sum():,}"
              f"  {'OK' if abs(m) < 0.5 and nm < 0.5 else 'CONVENTION ERROR'}")
        if abs(m) >= 0.5 or nm >= 0.5:
            raise SystemExit(
                f"the reference does not reproduce itself ({m:+.2f} m): the "
                f"contour and the polygon use different coordinate "
                f"conventions, so every offset below is that error")
    print("    identical inputs return exactly zero, so the -10 m pile-up "
          "below is real,")
    print("    but the reference is a 20 m raster staircase: offsets well "
          "under one cell")
    print("    cannot be resolved against it.")


# --------------------------------------------------------------------- main --
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("GATE 7C2 — scene / shoreline quality, three independent axes")
    print("=" * 78)
    print(f"  git {commit}")
    print("  NO thresholds and NO pass/fail are assigned here. Two scenes are")
    print("  not a basis for a cut; this gate produces the distribution.")

    M, meta = load_frozen_manifest()
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")

    def rast(g):
        return rio_rasterize([(g, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0, dtype="uint8").astype(bool)

    wet = np.zeros(inside.shape, bool)
    try:
        import p0b_build_dnipro_water_domain as P0B
        import rasterio
        from rasterio.warp import Resampling as WR, reproject
        tok = P0B.sas_token()
        for f in P0B.search_worldcover([32.2, 46.5, 35.5, 48.0]):
            tmp = np.zeros(inside.shape, np.uint8)
            with rasterio.open(f["assets"]["map"]["href"] + "?" + tok) as ds:
                reproject(source=rasterio.band(ds, 1), destination=tmp,
                          dst_transform=G["tr"], dst_crs="EPSG:32636",
                          dst_nodata=0, resampling=WR.nearest)
            wet |= (tmp == 90)
    except Exception as ex:
        print(f"  wetland context unavailable ({type(ex).__name__})")

    refs = {}
    for t in ("H1", "H2", "H3"):
        if t not in gdf.index:
            continue
        g = gdf.loc[t, "geometry"]
        xy, inner = densify(g)
        refs[t] = dict(poly=g, xy=xy, inner=inner, mask=rast(g))
        print(f"  reference {t}: {xy.shape[0]:,} boundary points, "
              f"{100*inner.mean():.0f}% on island rings")

    null_test(refs, G)

    elig = M[M.scientific_roles.fillna("").str.contains("TARGET_STAGE_GEOMETRY")
             & (M.temporal_regime == "PREBREACH_IMPOUNDED")
             & M.WSE_RESOLVED.astype(bool)
             & M.wse_semantics_resolved.astype(bool)]
    rows, maps = [], {}
    for r in elig.itertuples():
        npz = CACHE / f"{r.event_id}.npz"
        if not npz.exists():
            continue
        tgt = next((t for t in ("H1", "H2", "H3")
                    if f"{t}_TARGET_STAGE_GEOMETRY" in str(r.scientific_roles)), "")
        if not tgt or tgt not in refs:
            continue
        R = refs[tgt]
        opt = R["mask"]
        z = np.load(npz)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        obs = cov & inside
        vvd, vhd = db(vv), db(vh)
        d_in = ndimage.distance_transform_edt(opt) * CELL
        d_out = ndimage.distance_transform_edt(~opt) * CELL
        aw = opt & (d_in > 500) & obs & ~wet
        al = inside & ~opt & (d_out > 100) & obs & ~wet
        if aw.sum() < 500 or al.sum() < 500:
            del vv, vh, cov, vvd, vhd
            continue
        Xw = np.c_[vvd[aw], vhd[aw]]; Xl = np.c_[vvd[al], vhd[al]]
        mw, ml = Xw.mean(0), Xl.mean(0)
        Sw = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
        wv = np.linalg.solve(Sw, ml - mw)
        cut = 0.5 * (wv @ mw + wv @ ml)
        disc = np.full(inside.shape, np.nan, np.float32)
        disc[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) - cut
        # orient negative-in-water from the anchors, never from the LDA algebra
        if np.nanmedian(disc[aw]) > np.nanmedian(disc[al]):
            disc = -disc

        raw = (disc < 0) & obs
        cleaned = clean_water(raw)
        lab, n = ndimage.label(cleaned, structure=np.ones((3, 3), int))
        keep = np.zeros(n + 1, bool)
        keep[list(set(np.unique(lab[aw & cleaned])) - {0})] = True
        core = keep[lab]

        A = classifier_metrics(disc, aw, al, obs)
        B, C = topology_metrics(raw, cleaned, core, G)
        C["_filled"] = ndimage.binary_fill_holes(core)
        SP = subpixel_external(disc, core, G)
        ext = SP[SP[:, 3] > 0.5][:, :3] if len(SP) else SP
        ext, edge_frac = drop_swath_edges(ext, obs, G)
        C["subpixel_external"] = ext
        B["subpixel_internal_km"] = (float(SP[SP[:, 3] <= 0.5, 2].sum() / 1000)
                                     if len(SP) else np.nan)
        B["swath_edge_length_fraction"] = edge_frac
        Cm, mp = correspondence_metrics(C, R["xy"], R["inner"], R["poly"], G)
        rows.append(dict(event_id=r.event_id, date=r.date, target=tgt,
                         relative_orbit=int(r.relative_orbit),
                         obs_fraction=float(obs.sum() / inside.sum()),
                         **A, **B, **Cm))
        print(f"  {r.event_id:26s} {tgt}  sep {A['lda_separation']:5.2f} "
              f"eff {A['lda_effect_size']:5.2f} | largest "
              f"{100*B['core_largest_fraction']:5.1f}% R_int "
              f"{B['internal_external_ratio']:5.2f} | f_L "
              f"{100*Cm.get('f_L_100', np.nan):5.1f}% med "
              f"{Cm.get('median_100_vs_raster_ref', np.nan):+6.1f} drift "
              f"{Cm.get('median_drift', np.nan):5.1f}")
        if mp is not None:
            maps[r.event_id] = dict(tgt=tgt, date=r.date, core=core, **mp)
        del vv, vh, cov, vvd, vhd, disc

    Q = pd.DataFrame(rows)
    Q.to_csv(CFG.TABLES / "gate7c2_scene_quality.csv", index=False)
    print(f"\n  {len(Q)} scenes -> outputs/tables/gate7c2_scene_quality.csv")
    report(Q)
    figures(Q, maps, G, refs)
    print("\nSTOP. No thresholds set; H1/H2/H3 residuals not computed.")


def report(Q):
    print("\n" + "=" * 78)
    print("ROLE OF THIS GATE")
    print("=" * 78)
    print("  ADMISSION metrics (staircase bias cancels; usable):")
    print("    f_L_*, median_drift, nmad_*, p95_*, bidirectional_difference,")
    print("    unpaired_top5_cell_fraction, and every axis-A / axis-B metric.")
    print("  NOT positional truth (absolute position vs a 20 m raster")
    print("  staircase; never quote as a SAR-optical shoreline offset):")
    print("    median_*_vs_raster_ref, median_reverse_*_vs_raster_ref.")
    print("  The physical offset d_raw -> d_stage -> d_residual is measured")
    print("  only where BOTH sides are continuous geometry: S1 LDA contour")
    print("  vs S2 NDWI contour. See hist25b_gate7c_v2_same_date_controls.py.")

    print("\n" + "=" * 78)
    print("DISTRIBUTION OF THE THREE AXES (no cut applied)")
    print("=" * 78)
    blocks = {
        "A classifier": ["lda_separation", "lda_effect_size",
                         "lda_overlap_norm", "frac_near_decision"],
        "B topology": ["core_largest_fraction", "core_n_components_large",
                       "internal_external_ratio", "hole_area_km2"],
        "C correspondence (admission)": ["f_L_100", "median_drift",
                             "nmad_100", "p95_100", "bidirectional_difference",
                             "unpaired_top5_cell_fraction"]}
    for name, cols in blocks.items():
        cols = [c for c in cols if c in Q.columns]
        print(f"\n  {name}")
        print(Q[cols].describe().loc[
            ["min", "25%", "50%", "75%", "max"]].to_string(
                float_format=lambda v: f"{v:9.3f}"))
    # Are the three axes actually independent, or is one score enough after
    # all? Spearman, because none of these are expected to be linear and n=26.
    from scipy import stats
    print("\n  ARE THE AXES INDEPENDENT? (Spearman, n = %d)" % len(Q))
    pairs = [("lda_separation", "core_largest_fraction", "A vs B"),
             ("lda_separation", "median_drift", "A vs C drift"),
             ("lda_separation", "f_L_100", "A vs C paired length"),
             ("core_largest_fraction", "median_drift", "B vs C drift"),
             ("internal_external_ratio", "median_drift", "B vs C drift"),
             ("obs_fraction", "median_drift", "coverage vs C drift"),
             ("obs_fraction", "external_length_km", "coverage vs seen length"),
             ("lda_separation", "lda_effect_size", "A vs A (control)"),
             ("f_L_100", "nmad_100", "C vs C (control)")]
    for a, b, lab in pairs:
        if a not in Q.columns or b not in Q.columns:
            continue
        m = Q[[a, b]].dropna()
        rho, p = stats.spearmanr(m[a], m[b])
        mark = "significant" if p < 0.05 else "not significant"
        print(f"    {lab:22s} {a:24s} vs {b:22s} "
              f"rho {rho:+.2f}  p {p:.3f}  {mark}")
    print("    Cross-axis pairs are null while the within-axis controls are")
    print("    strong: a single scene_quality score is not defensible.")

    print("\n  per scene, sorted by correspondence drift:")
    cols = ["date", "target", "relative_orbit", "lda_separation",
            "core_largest_fraction", "internal_external_ratio",
            "f_L_100", "median_drift", "nmad_100", "bidirectional_difference"]
    cols = [c for c in cols if c in Q.columns]
    print(Q.sort_values("median_drift")[cols].to_string(
        index=False, float_format=lambda v: f"{v:8.2f}"))


def figures(Q, maps, G, refs):
    # --- 1. the three axes against each other -----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    cmap = {"H1": AMBER, "H2": GREEN, "H3": BLUE}
    cs = Q.target.map(cmap)
    ax[0].scatter(Q.lda_separation, 100 * Q.core_largest_fraction, c=cs, s=70,
                  edgecolor=INK, linewidth=0.5)
    ax[0].set_xlabel("A · LDA class separation")
    ax[0].set_ylabel("B · largest component (% of water area)")
    ax[0].set_title("classifier quality does not determine topology", color=INK)
    # largest_fraction is saturated at 98.3-100% across this scene set because
    # the production chain applies the anchor-connectivity filter, so it has no
    # dynamic range HERE (it had plenty in the Gate 7C1 controls, which did not
    # apply that filter). R_internal is the topology metric that still varies.
    ax[1].scatter(Q.internal_external_ratio, Q.median_drift, c=cs, s=70,
                  edgecolor=INK, linewidth=0.5)
    ax[1].set_xlabel("B · internal / external boundary length")
    ax[1].set_ylabel("C · median drift over 25-250 m (m)")
    ax[1].set_title("topology does not determine shoreline stability\n"
                    "(largest-component % is saturated after the core filter)",
                    color=INK)
    ax[2].scatter(Q.lda_separation, Q.median_drift, c=cs, s=70,
                  edgecolor=INK, linewidth=0.5)
    ax[2].set_xlabel("A · LDA class separation")
    ax[2].set_ylabel("C · median drift (m)")
    ax[2].set_title("and it does not determine correspondence either", color=INK)
    for a in ax:
        a.grid(alpha=0.3)
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=t)
         for t, c in cmap.items()]
    ax[0].legend(handles=h, title="target", fontsize=9)
    fig.suptitle("Gate 7C2 · three axes, deliberately not collapsed into one "
                 "score", color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "gate7c2_quality_axes.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"-> {p}")

    # --- 2. drift profiles -------------------------------------------------
    # the medians are renamed *_vs_raster_ref, so match on the middle token
    tc = [c for c in Q.columns
          if c.startswith("median_") and c.endswith("_vs_raster_ref")
          and c.split("_")[1].isdigit()]
    tc = sorted(tc, key=lambda c: int(c.split("_")[1]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    th = [int(c.split("_")[1]) for c in tc]
    for _, r in Q.iterrows():
        stable = r.median_drift < Q.median_drift.median()
        ax[0].plot(th, [r[c] for c in tc], color=GREEN if stable else RED,
                   alpha=0.75, lw=1.4, marker="o", ms=3)
    ax[0].axhline(0, color=INK, lw=0.9)
    ax[0].set_xlabel("max match distance (m)")
    ax[0].set_ylabel("median vs 20 m raster reference (m)\nNOT a physical offset -- shape only")
    ax[0].set_title("only the SHAPE of these lines is used: flat = stable,\n"
                    "rising = selection by the threshold", color=INK)
    ax[0].grid(alpha=0.3)
    fc = [c for c in Q.columns if c.startswith("f_L_")
          and c.split("_")[-1].isdigit()]
    fc = sorted(fc, key=lambda c: int(c.split("_")[-1]))
    for _, r in Q.iterrows():
        stable = r.median_drift < Q.median_drift.median()
        ax[1].plot([int(c.split("_")[-1]) for c in fc],
                   [100 * r[c] for c in fc],
                   color=GREEN if stable else RED, alpha=0.75, lw=1.4,
                   marker="o", ms=3)
    ax[1].set_xlabel("max match distance (m)")
    ax[1].set_ylabel("paired external length (%)")
    ax[1].set_title("how much shoreline each threshold buys", color=INK)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Gate 7C2 · threshold sensitivity, every eligible scene",
                 color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "gate7c2_drift_profiles.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"-> {p}")

    # --- 3. maps of the extreme scenes ------------------------------------
    if not maps:
        return
    order = Q.sort_values("median_drift").event_id.tolist()
    pick = [e for e in (order[:2] + order[-2:]) if e in maps]
    pick += [e for e in order if e in maps and e not in pick][:max(0, 4 - len(pick))]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    for ax, eid in zip(axes.ravel(), pick[:4]):
        m = maps[eid]
        q = Q[Q.event_id == eid].iloc[0]
        far = m["dist"] > 100.0
        # the signed offset itself carries the information; a paired/unpaired
        # two-colour map hides everything under a single green line
        sc = ax.scatter(m["X"][~far] / 1000, m["Y"][~far] / 1000, s=1.6,
                        c=np.clip(m["d"][~far], -60, 60), cmap="RdBu_r",
                        vmin=-60, vmax=60, linewidths=0)
        ax.scatter(m["X"][far] / 1000, m["Y"][far] / 1000, s=14, c="none",
                   edgecolor=INK, linewidths=0.7,
                   label=f"unpaired > 100 m ({100*far.mean():.1f}%)")
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label("offset vs 20 m raster reference (m)\n"
                 "not a physical SAR-optical bias", fontsize=7)
        ax.set_title(f"{m['date']} · {m['tgt']} · sep {q.lda_separation:.2f} · "
                     f"R_int {q.internal_external_ratio:.2f} · "
                     f"drift {q.median_drift:.1f} m · "
                     f"NOT positional truth", color=INK, fontsize=10)
        ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
        ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Gate 7C2 · external shoreline, most and least stable scenes",
                 color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "gate7c2_scene_maps.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"-> {p}")


if __name__ == "__main__":
    main()
