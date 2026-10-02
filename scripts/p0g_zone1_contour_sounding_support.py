#!/usr/bin/env python
"""Is the newly recovered contour area supported by soundings, or invented?

The corrected contours add ~96-98 km2 at every level (about 4.8%) by removing
the P20 truncation. Before rebuilding the DEM on them, this asks whether that
new ground is measured, interpolated, or extrapolated -- because a DEM will
happily produce a smooth surface over territory no echo sounder ever crossed,
and the resulting A(H) and V(H) would then be model artefacts.

The question is sharper than it looks. hist24 records that the highest
sounding in the survey is 15.3153 m against a 17.08 m waterline, so a 1.77 m
elevation band already had no sounding control at all. The recovered area sits
in exactly that shallow margin, so the expectation is extrapolation -- but the
point is to measure it, not assume it.

Support classes, by distance to the nearest real sounding:

    SURVEY_SUPPORTED   within SUPPORT_M          -- a sounding is right there
    INTERPOLATION      within INTERP_M           -- between soundings
    EXTRAPOLATION      beyond INTERP_M           -- the surface is invented

SUPPORT_M is derived from the survey's own nearest-neighbour spacing rather
than chosen, so the classes describe this dataset instead of a convention.

Table 19 is deliberately NOT used anywhere here. It is the independent control
for the rebuilt A(H) later, and using it now would spend that independence.

Outputs
-------
outputs/tables/p0g_contour_sounding_support.csv
outputs/tables/p0g_recovered_area_breakdown.csv
outputs/figures/historical_bathymetry/png/p0g_sounding_support_map.png
outputs/figures/historical_bathymetry/png/p0g_recovered_area_detail.png
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
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import box

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
CELL = 50.0                      # support mapping does not need 20 m
SOUNDINGS = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
NEW_GPKG = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
OLD_GPKG = ROOT / "data/processed/bathymetry/legacy_p20/prebreach_contours_LEGACY_P20.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
Z_MAX_SOUNDING = 15.3153         # highest sounding in the survey (hist24)
DIST_BANDS = (500.0, 1000.0, 2000.0)

# named sub-regions of the recovery, so the eastern strip and the Khortytsia
# arms can be judged separately from ordinary edge slivers
SUBREGIONS = {
    "eastern_strip": (668540.0, 5240000.0, 1e7, 1e7),
    "khortytsia_arms": (645000.0, 5285000.0, 668540.0, 1e7),
}


def _draw_km(ax, geom, color, lw, ls):
    """Draw a polygon boundary in kilometres. geopandas .plot() works in the
    geometry's own units (metres here) and would rescale axes that are in km."""
    for g in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        for ring in [g.exterior, *g.interiors]:
            xy = np.asarray(ring.coords)
            ax.plot(xy[:, 0] / 1000, xy[:, 1] / 1000, color=color, lw=lw,
                    linestyle=ls)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if not SOUNDINGS.exists():
        raise SystemExit(f"soundings not found: {SOUNDINGS}")
    S = pd.read_parquet(SOUNDINGS)
    xcol, ycol, zcol = "x", "y", "H_bed_evrf2019_m"
    print(f"soundings: {len(S):,} points, columns x={xcol} y={ycol} z={zcol}")
    print(f"  z range {S[zcol].min():.2f} .. {S[zcol].max():.2f} m EVRF2019")
    xy = np.c_[S[xcol].to_numpy(), S[ycol].to_numpy()]
    tree = cKDTree(xy)

    # survey's own spacing sets the support radius
    d1, _ = tree.query(xy, k=2)
    nn = d1[:, 1]
    support_m = float(np.percentile(nn, 90))
    interp_m = float(np.percentile(nn, 99)) * 2
    print(f"  nearest-neighbour spacing: median {np.median(nn):.0f} m, "
          f"p90 {support_m:.0f} m, p99 {np.percentile(nn,99):.0f} m")
    print(f"  -> SURVEY_SUPPORTED <= {support_m:.0f} m, "
          f"INTERPOLATION <= {interp_m:.0f} m, beyond that EXTRAPOLATION")

    gn = gpd.read_file(NEW_GPKG, layer="contour_polygons").set_index("contour_id")
    go = gpd.read_file(OLD_GPKG, layer="contour_polygons").set_index("contour_id")

    # common grid over the corrected H1 (the largest contour)
    fp = gn.loc["H1", "geometry"]
    b = fp.bounds
    PAD = 4000.0
    x0, y0 = np.floor((b[0] - PAD) / CELL) * CELL, np.floor((b[1] - PAD) / CELL) * CELL
    x1, y1 = np.ceil((b[2] + PAD) / CELL) * CELL, np.ceil((b[3] + PAD) / CELL) * CELL
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    tr = from_origin(x0, y1, CELL, CELL)
    print(f"\nsupport grid {nx} x {ny} at {CELL:.0f} m")

    # Distance to the nearest sounding is necessary but NOT sufficient: this
    # survey runs in profiles, so a cell can sit 300 m from a sounding and
    # still lie between two parallel tracks, where the across-track bed
    # gradient is unconstrained. Anisotropy is measured from the k nearest
    # soundings: if they are collinear the cell is flanked by one track only.
    K_ANISO = 8

    xs = x0 + (np.arange(nx) + 0.5) * CELL
    ys = y1 - (np.arange(ny) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    cells = np.c_[gx.ravel(), gy.ravel()]
    dist, _ = tree.query(cells, k=1)
    dist = dist.reshape(ny, nx)

    dk, ik = tree.query(cells, k=K_ANISO)
    nb = xy[ik]                                   # (ncell, K, 2)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / K_ANISO
    tr_ = cov[:, 0, 0] + cov[:, 1, 1]
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    disc = np.sqrt(np.maximum(tr_ ** 2 / 4 - det, 0))
    l1, l2 = tr_ / 2 + disc, tr_ / 2 - disc
    aniso = np.sqrt(np.maximum(l2, 0) / np.maximum(l1, 1e-9)).reshape(ny, nx)
    # 1 = soundings spread in all directions, 0 = a single straight track
    print(f"  local sounding anisotropy (k={K_ANISO}): "
          f"median {np.median(aniso):.3f}; "
          f"{100*float((aniso < 0.25).mean()):.1f}% of cells are track-like")

    def rast(g):
        return rio_rasterize([(g, 1)], out_shape=(ny, nx), transform=tr,
                             fill=0, dtype="uint8").astype(bool)

    # point-in-polygon once, vectorised: rebuilding the difference geometry
    # per point would be 7,514 boolean ops per contour
    pts = gpd.GeoSeries(gpd.points_from_xy(S[xcol], S[ycol]), crs=CFG.CRS_METRIC)
    rows, brk = [], []
    for cid in ("H3", "H2", "H1"):
        new = rast(gn.loc[cid, "geometry"])
        old = rast(go.loc[cid, "geometry"])
        rec = new & ~old                      # recovered by the domain fix
        px = CELL ** 2 / 1e6
        in_new = pts.within(gn.loc[cid, "geometry"]).to_numpy()
        in_old = pts.within(go.loc[cid, "geometry"]).to_numpy()
        counts = {"corrected_total": int(in_new.sum()),
                  "legacy_P20": int(in_old.sum()),
                  "recovered": int((in_new & ~in_old).sum())}
        for nm, m in (("corrected_total", new), ("legacy_P20", old),
                      ("recovered", rec)):
            if not m.any():
                continue
            d = dist[m]
            inside_pts = counts[nm]
            rows.append(dict(
                contour_id=cid, region=nm, area_km2=m.sum() * px,
                n_soundings_inside=inside_pts,
                density_per_km2=(inside_pts / (m.sum() * px)
                                 if m.sum() and np.isfinite(inside_pts) else np.nan),
                dist_median_m=float(np.median(d)),
                dist_p90_m=float(np.percentile(d, 90)),
                **{f"frac_beyond_{int(bm)}m": float((d > bm).mean())
                   for bm in DIST_BANDS},
                aniso_median=float(np.median(aniso[m])),
                frac_track_like=float((aniso[m] < 0.25).mean()),
                frac_survey_supported=float((d <= support_m).mean()),
                frac_interpolation=float(((d > support_m) & (d <= interp_m)).mean()),
                frac_extrapolation=float((d > interp_m).mean())))
        # named sub-regions of the recovered area
        for sn, bb in SUBREGIONS.items():
            sub = rec & rast(box(*bb))
            if sub.sum() == 0:
                continue
            d = dist[sub]
            brk.append(dict(contour_id=cid, subregion=sn,
                            area_km2=sub.sum() * px,
                            dist_median_m=float(np.median(d)),
                            dist_p90_m=float(np.percentile(d, 90)),
                            frac_extrapolation=float((d > interp_m).mean())))
        if cid == "H1":
            keep = (new, old, rec)

    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "p0g_contour_sounding_support.csv", index=False)
    B = pd.DataFrame(brk)
    B.to_csv(CFG.TABLES / "p0g_recovered_area_breakdown.csv", index=False)

    print("\n" + "=" * 78)
    print("SUPPORT OF THE RECOVERED AREA")
    print("=" * 78)
    show = ["contour_id", "region", "area_km2", "n_soundings_inside",
            "dist_median_m", "dist_p90_m", "frac_survey_supported",
            "frac_extrapolation"]
    print(R[show].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nrecovered-area sub-regions:")
    print(B.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    rec_h1 = R[(R.contour_id == "H1") & (R.region == "recovered")].iloc[0]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  recovered area at H1: {rec_h1.area_km2:,.1f} km2")
    print(f"  soundings inside it : {int(rec_h1.n_soundings_inside)}")
    print(f"  median distance to the nearest sounding: "
          f"{rec_h1.dist_median_m:,.0f} m (p90 {rec_h1.dist_p90_m:,.0f} m)")
    print(f"  classified EXTRAPOLATION: {100*rec_h1.frac_extrapolation:.1f}%")
    print(f"\n  the survey's highest sounding is {Z_MAX_SOUNDING:.2f} m against a "
          f"17.08 m waterline, so the shallow margin had no control even before "
          f"the domain fix")

    # ------------------------------------------------------------- maps
    new, old, rec = keep
    ext = [x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000]
    cls = np.zeros((ny, nx), np.uint8)
    cls[new & (dist <= support_m)] = 1
    cls[new & (dist > support_m) & (dist <= interp_m)] = 2
    cls[new & (dist > interp_m)] = 3
    cmap = matplotlib.colors.ListedColormap(
        ["#ffffff", GREEN, AMBER, RED])

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.6))
    a = ax[0]
    a.imshow(cls, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=3,
             interpolation="nearest")
    a.plot(S[xcol] / 1000, S[ycol] / 1000, ".", ms=0.35, color=INK, alpha=0.55)
    _draw_km(a, go.loc["H1", "geometry"], PURPLE, 1.0, "--")
    a.set_title("a · H1 bed-support classes with soundings (black) and the "
                "old P20 boundary (dashed)\n"
                f"green = survey-supported (<= {support_m:.0f} m), "
                f"amber = interpolation, red = extrapolation",
                fontsize=9.8, loc="left")
    a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    a = ax[1]
    show_rec = np.where(rec, 1, 0) + np.where(old & new, 2, 0)
    a.imshow(show_rec, extent=ext, origin="upper", vmin=0, vmax=2,
             cmap=matplotlib.colors.ListedColormap(["#ffffff", RED, "#cfe0ec"]),
             interpolation="nearest")
    a.plot(S[xcol] / 1000, S[ycol] / 1000, ".", ms=0.35, color=INK, alpha=0.55)
    a.set_title("b · red = area recovered by the domain fix, blue = already in "
                "P20\nsoundings overlaid to show where the new ground is "
                "measured", fontsize=9.8, loc="left")
    a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    fig.suptitle("p0g · is the recovered contour area supported by soundings?",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out1 = FIGDIR / "p0g_sounding_support_map.png"
    fig.savefig(out1, dpi=165, bbox_inches="tight")
    plt.close(fig)

    # detail panels on the named sub-regions
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    for a, (sn, bb) in zip(ax, SUBREGIONS.items()):
        a.imshow(cls, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=3,
                 interpolation="nearest")
        a.plot(S[xcol] / 1000, S[ycol] / 1000, ".", ms=1.6, color=INK)
        _draw_km(a, go.loc["H1", "geometry"], PURPLE, 1.4, "--")
        a.set_xlim(bb[0] / 1000 - 6, min(bb[2], x1) / 1000 + 3)
        a.set_ylim(max(bb[1], y0) / 1000 - 4, min(bb[3], y1) / 1000 + 3)
        sub = B[(B.subregion == sn) & (B.contour_id == "H1")]
        txt = (f"{sub.area_km2.iloc[0]:,.1f} km2, "
               f"{100*sub.frac_extrapolation.iloc[0]:.0f}% extrapolation"
               if len(sub) else "not present at H1")
        a.set_title(f"{sn.replace('_',' ')}\n{txt}", fontsize=10, loc="left")
        a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    fig.suptitle("p0g · recovered sub-regions in detail (dashed = old P20 limit)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out2 = FIGDIR / "p0g_recovered_area_detail.png"
    fig.savefig(out2, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.TABLES/'p0g_contour_sounding_support.csv'}")
    print(f"-> {CFG.TABLES/'p0g_recovered_area_breakdown.csv'}")
    print(f"-> {out1}")
    print(f"-> {out2}")
    print("\nSTOP before hist24. Table 19 deliberately unused: it is the "
          "independent control for the rebuilt A(H).")


if __name__ == "__main__":
    main()
