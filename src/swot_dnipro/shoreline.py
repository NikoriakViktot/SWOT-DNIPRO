"""Dated shorelines from the optical water rule, as continuous contours.

The ZONE_1 method, lifted from `scripts/hist26_continuous_contours.py` so that
zones 2/3/4 can use it unchanged. The idea there (Gate 7C2) is that a shoreline
traced along 20 m pixel edges is a staircase whose position error is the pixel
itself, and `12_GATE_7C_ROLE_FREEZE.md` forbids treating such a staircase as
positional truth. Instead the water RULE is evaluated as a continuous field and
its zero level is taken by Marching Squares, giving sub-pixel vertices.

The field, reconstructed from the products p25 writes
-----------------------------------------------------
hist26 builds ``W = max(min(NDWI - t, MNDWI - t), NDWI - (t - 0.15) where SCL == 6)``
from bands + SCL. p25 persists the indices and the three-valued mask but not
SCL, so the field is rebuilt from those two exactly:

    W = min(NDWI, MNDWI)                          (t = 0 in the frozen rule)
    W = +eps  where water3 == 1 and W <= 0        (the SCL==6-relaxed pixels)
    W = NaN   where water3 == 255                 (not observed)

Where the mask says water, W > 0; where it says land, W <= 0 -- so the zero
level of W IS the mask boundary, but interpolated. Nothing about the rule is
re-decided here.

Multi-date compositing is "first valid observation wins" in the order the
caller supplies (hist23_three_contour_masks.py:260-265); water is never OR-ed
across dates, which would inflate the pool.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize

EPS = 1e-3


def decision_field(ndwi: np.ndarray, mndwi: np.ndarray, water3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(W, valid) on the zone grid from p25's indices + three-valued mask."""
    valid = water3 != 255
    W = np.minimum(ndwi, mndwi).astype("f4")
    relaxed = valid & (water3 == 1) & ~(W > 0)
    W[relaxed] = EPS
    W[~valid] = np.nan
    return W, valid


def composite_first_valid(fields: list[np.ndarray], valids: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First valid observation wins. Returns (W, valid, n_dates_used_per_pixel)."""
    W = np.full(fields[0].shape, np.nan, "f4")
    V = np.zeros(fields[0].shape, bool)
    src = np.zeros(fields[0].shape, "u1")
    for i, (f, v) in enumerate(zip(fields, valids), 1):
        new = v & ~V
        W[new] = f[new]
        V |= v
        src[new] = i
    return W, V, src


def main_water_region(W: np.ndarray, valid: np.ndarray, min_part_km2: float,
                      cell: float, dilate_px: int = 10) -> np.ndarray:
    """The dilated, hole-filled main water body: where contour rings must lie.

    Largest connected component of W > 0 (plus every component >= min_part_km2
    -- the delta is many arms, not one blob), holes filled, dilated by
    `dilate_px` so the outer shoreline ring falls inside the region test."""
    from scipy import ndimage
    water = valid & (W > 0)
    lab, n = ndimage.label(water, structure=np.ones((3, 3), int))
    if n == 0:
        return np.zeros_like(water)
    px_km2 = cell ** 2 / 1e6
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * px_km2
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_part_km2
    keep[int(np.argmax(sizes)) + 1] = True
    body = keep[lab]
    body = ndimage.binary_fill_holes(body)
    return ndimage.binary_dilation(body, iterations=dilate_px)


def contour_geometry(F: np.ndarray, valid: np.ndarray, G: dict, region: np.ndarray,
                     cell: float, frac: float = 0.8):
    """Sub-pixel contour of F = 0 (hist26.contour_geometry, grid-adapted).

    Whole rings are kept, never trimmed: a ring is kept when >= `frac` of its
    vertices are valid AND inside `region`. F must be the NEGATED water field
    (water < 0) so that `rings_to_polygon` can test faces by sign, exactly as
    hist26 does. `G` needs `x0` (left edge) and `y1` (top edge); row 0 = top."""
    from skimage.measure import find_contours
    rings, lines = [], []
    for c in find_contours(np.nan_to_num(F, nan=1e6), 0.0):
        if len(c) < 4:
            continue
        x = G["x0"] + (c[:, 1] + 0.5) * cell
        y = G["y1"] - (c[:, 0] + 0.5) * cell
        ri = np.clip(np.round(c[:, 0]).astype(int), 0, F.shape[0] - 1)
        ci = np.clip(np.round(c[:, 1]).astype(int), 0, F.shape[1] - 1)
        if valid[ri, ci].mean() < frac or region[ri, ci].mean() < frac:
            continue
        rings.append(np.c_[x, y])
        lines.append(LineString(np.c_[x, y]))
    return rings, lines


def rings_to_polygon(rings, F: np.ndarray, G: dict, cell: float):
    """Assemble rings into a polygon by FIELD SIGN (hist26.rings_to_polygon).

    Polygonise the linework, keep every face whose representative point has
    F < 0 (water). No orientation, no area ranking, no containment tests --
    hist26's docstring records the 3,112 km2 > 2,340 km2 impossibility that
    the earlier ranking approach produced."""
    lines = []
    for r in rings:
        if len(r) < 4:
            continue
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[0]])
        lines.append(LineString(r))
    if not lines:
        return None
    faces = list(polygonize(unary_union(lines)))
    keep = []
    for f in faces:
        p = f.representative_point()
        j = int((p.x - G["x0"]) / cell)
        i = int((G["y1"] - p.y) / cell)
        if 0 <= i < F.shape[0] and 0 <= j < F.shape[1] and np.isfinite(F[i, j]) and F[i, j] < 0:
            keep.append(f)
    return unary_union(keep) if keep else None


def close_field(W: np.ndarray, valid: np.ndarray, inside: np.ndarray, land_value: float = 1e6) -> np.ndarray:
    """Negated field F = -W with NOT-observed and OUTSIDE-zone cells set to land.

    Without this the main water body's outer ring runs off wherever water
    crosses the zone boundary (ZONE_2's boundary cuts the liman) and is then
    dropped by the ring-validity test: on ZONE_2 that lost 104 of 187 km2 of
    water. Setting those cells to land closes every ring at the zone edge and
    at cloud edges. The closing segments are NOT shorelines -- `edge_safe`
    removes the vertices that lie on them before they can become constraints
    (protocol 9.3: real shore vs frame edge, cloud edge, AOI edge)."""
    F = -W.astype("f4")
    F[~valid] = land_value
    F[~inside] = land_value
    return F


def edge_safe(valid: np.ndarray, inside: np.ndarray, cell: float, margin_px: float = 1.5) -> np.ndarray:
    """True where a vertex is at least `margin_px` from any not-observed or
    outside-zone cell -- i.e. where a contour vertex is a WATER edge, not a data
    or domain edge."""
    from scipy import ndimage
    ok = valid & inside
    dist = ndimage.distance_transform_edt(ok)
    return dist >= margin_px


def densify(geom, spacing_m: float) -> np.ndarray:
    """Boundary vertices at fixed spacing (hist24.densify)."""
    pts = []
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            L = ring.length
            n = max(int(L // spacing_m), 4)
            for k in range(n):
                p = ring.interpolate(k / n, normalized=True)
                pts.append((p.x, p.y))
    return np.array(pts)


def slope_from_soundings(pts: np.ndarray, snd_xy: np.ndarray, snd_z: np.ndarray,
                         radius_m: float = 1000.0, min_pts: int = 8) -> np.ndarray:
    """Local bed slope (m/m) at each vertex from a planar LSQ over soundings within
    `radius_m` -- soundings only, never a DEM (anti-circular; gate7b:199-212)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(snd_xy)
    out = np.full(len(pts), np.nan)
    for i, nb in enumerate(tree.query_ball_point(pts, r=radius_m)):
        if len(nb) < min_pts:
            continue
        X = np.c_[snd_xy[nb] - pts[i], np.ones(len(nb))]
        coef, *_ = np.linalg.lstsq(X, snd_z[nb], rcond=None)
        out[i] = float(np.hypot(coef[0], coef[1]))
    return out
