#!/usr/bin/env python
"""P55g -- map atlas and local kriging-support diagnosis for the residual source-2 errors. DIAGNOSIS ONLY.

p53 is frozen and the p55 admissibility gate is not touched here. After p55e/p55f the remaining source-2 tail is inside the
registry pool polygon, near its inner edge, and p55f established that distance to the boundary is the dominant axis while
distance to the nearest sounding separates the tail IN REVERSE (tail points are closer to soundings). This script answers the
question that follows -- "would more/better kriging points help?" -- with maps and counts rather than assertion.

Three map levels:
  A overview      the whole source-2 area, and the edge belt on its own, with every water-mask CONTOUR drawn as a line so the
                  masks can be told apart, plus soundings, ICESat-2 ground points and the residual extremes of BOTH signs.
  B clusters      one large panel per tail cluster: registry boundary, mask contours, hist20 bed, FABDEM hillshade, soundings,
                  ICESat-2 points labelled with bed / ground / residual / distances.
  C support       for the same clusters: sounding density, distance-to-sounding, counts in 100/250/500/1000 m rings, and the
                  actual k = 32 Euclidean neighbourhood hist14 uses, with the neighbours that lie OUTSIDE the pool drawn in a
                  separate colour -- that is where a Euclidean neighbourhood crosses the shore.

WHAT THE INTERPOLATOR ACTUALLY DOES (read from the code, not assumed):
  hist14._ok is local ordinary kriging on the k = 32 nearest neighbours by PLAIN EUCLIDEAN distance (`OK_K = 32`,
  `cKDTree(...).query(k=K)`). There is no search radius, no anisotropy, no barrier and no sector search. The exported
  `kakhovka_bed_OK_epoch_*.tif` is the `epoch` shoreline variant: 7 514 soundings PLUS shoreline pseudo-points placed along
  the registry boundary every `SHORE_STEP_M = 250 m` and held at 17.08 m. So the shore IS already constrained; the question
  is whether those pseudo-points survive into the 32 nearest neighbours of a near-shore target, or are outvoted by the much
  denser soundings a little further away. That is counted here, per problem point.

The sensitivity test refits ONE variogram and then changes ONLY the neighbourhood (k, a radius cap, denser shore points), so
any difference is attributable to neighbourhood configuration and not to a different model.
Outputs: outputs/figures/p55g/*.png (+ .pdf for the two overviews), outputs/tables/p55g_*.csv
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from shapely.geometry import MultiLineString, Point

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

spec = importlib.util.spec_from_file_location("hist14", ROOT / "scripts/hist14_bed_surface.py")
H14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(H14)

FIG = ROOT / "outputs/figures/p55g"; FIG.mkdir(parents=True, exist_ok=True)
PTS = ROOT / "outputs/tables/p55f_source2_points_with_axes.csv"
SND = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
BED30 = ROOT / "outputs/rasters/kakhovka_bed_OK_epoch_30m.tif"
DIST_SND = ROOT / "outputs/rasters/kakhovka_bed_dist_to_sounding_250m.tif"
ISL = ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif"
WFRAC = ROOT / "outputs/rasters/zone1/zone1_water_frac_PRE_BREACH_20m.tif"
CONT_GPKG = ROOT / "data/processed/bathymetry/prebreach_contours_continuous.gpkg"
FAB = {z: CFG.BULK_ROOT / "terrain" / z / "fabdem_evrf2019_20m.tif"
       for z in ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")}
POOL = SD.load_utm("reservoir_full_pool_prebreach")
TAIL_M = 5.0                  # |residual| threshold for "extreme", both signs (p55f used the negative side only)
CLUSTER_M = 1000.0            # single-linkage distance that defines a cluster
OK_K = H14.OK_K               # 32
SHORE_STEP_M = H14.SHORE_STEP_M      # 250 m
SHORE_LEVEL = H14.SHORE_VARIANTS["epoch"]   # 17.08 m
PAD_M = 900.0                 # half-width of a cluster panel

# one colour scheme for every map in the atlas
C = dict(pool="#111111", island="#1b7837", optical="#0570b0", contour="#8c510a",
         snd="#d95f02", ice="#4d4d4d", neg="#b2182b", pos="#2166ac", worst="#000000",
         nn_in="#1a9850", nn_out="#d73027")


# ------------------------------------------------------------------ helpers

def read_window(path, x0, x1, y0, y1, resampling=Resampling.bilinear):
    """Read the sub-window of a raster covering a metric box; returns (array, extent) or (None, None)."""
    p = Path(path)
    if not p.exists():
        return None, None
    with rasterio.open(p) as ds:
        try:
            win = rasterio.windows.from_bounds(x0, y0, x1, y1, ds.transform)
            a = ds.read(1, window=win, boundless=True, fill_value=ds.nodata if ds.nodata is not None else 0,
                        resampling=resampling).astype("f4")
        except Exception:
            return None, None
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
    return a, (x0, x1, y0, y1)


def poly_lines(geom):
    b = geom.boundary
    return list(b.geoms) if isinstance(b, MultiLineString) else [b]


def draw_poly(ax, geom, color, lw, label=None, ls="-", zorder=6):
    first = True
    for ln in poly_lines(geom):
        x, y = ln.xy
        ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=zorder, label=label if first else None)
        first = False


def raster_contour(ax, path, x0, x1, y0, y1, levels, color, lw=1.1, label=None, nearest=False):
    """Draw a raster's iso-line as a CONTOUR, so overlapping masks stay distinguishable."""
    a, ext = read_window(path, x0, x1, y0, y1, Resampling.nearest if nearest else Resampling.bilinear)
    if a is None or not np.isfinite(a).any():
        return False
    ny, nx = a.shape
    xs = np.linspace(x0, x1, nx); ys = np.linspace(y1, y0, ny)
    ax.contour(xs, ys, np.nan_to_num(a, nan=-999), levels=levels, colors=[color], linewidths=lw, zorder=7)
    if label:
        ax.plot([], [], color=color, lw=lw, label=label)
    return True


def scale_bar(ax, x0, x1, y0, y1, frac=0.25):
    """Manual scale bar (matplotlib-scalebar is not installed in this environment)."""
    span = x1 - x0
    nice = np.array([50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000], float)
    L = nice[np.argmin(np.abs(nice - span * frac))]
    xa = x0 + 0.06 * span; ya = y0 + 0.07 * (y1 - y0)
    ax.add_patch(Rectangle((xa, ya), L, 0.012 * (y1 - y0), fc="k", ec="k", zorder=12))
    ax.add_patch(Rectangle((xa + L / 2, ya), L / 2, 0.012 * (y1 - y0), fc="w", ec="k", zorder=12))
    ax.text(xa + L / 2, ya + 0.022 * (y1 - y0), f"{L/1000:g} km" if L >= 1000 else f"{L:g} m",
            ha="center", va="bottom", fontsize=8, zorder=12,
            bbox=dict(fc="w", ec="none", alpha=0.75, pad=1))


def north_arrow(ax, x0, x1, y0, y1):
    xa = x1 - 0.06 * (x1 - x0); ya = y0 + 0.07 * (y1 - y0); h = 0.07 * (y1 - y0)
    ax.annotate("", xy=(xa, ya + h), xytext=(xa, ya), zorder=12,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax.text(xa, ya + h + 0.012 * (y1 - y0), "N", ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=12)


def frame(ax, x0, x1, y0, y1, title):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5, ls=":")
    ax.set_xlabel("easting, m (EPSG:32636)", fontsize=8); ax.set_ylabel("northing, m", fontsize=8)
    ax.tick_params(labelsize=7)
    scale_bar(ax, x0, x1, y0, y1); north_arrow(ax, x0, x1, y0, y1)


def hillshade(ax, zone, x0, x1, y0, y1, alpha=1.0):
    a, _ = read_window(FAB.get(zone, ""), x0, x1, y0, y1)
    if a is None or not np.isfinite(a).any():
        return None
    ls = LightSource(315, 45)
    rgb = ls.shade(np.nan_to_num(a, nan=float(np.nanmedian(a))), cmap=plt.get_cmap("Greys_r"),
                   blend_mode="soft", vert_exag=6, dx=20, dy=20)
    ax.imshow(rgb, extent=(x0, x1, y0, y1), zorder=1, alpha=alpha, interpolation="bilinear")
    return a


# ------------------------------------------------------------------ data

def load():
    V = pd.read_csv(PTS)
    V = V[np.isfinite(V.res)].copy()
    S = pd.read_parquet(SND, columns=["x", "y", "depth_m", "H_bed_evrf2019_m"])
    bnd = POOL.boundary
    n_sh = max(int(bnd.length / SHORE_STEP_M), 100)
    sh = np.array([bnd.interpolate(t, normalized=True).coords[0]
                   for t in np.linspace(0, 1, n_sh, endpoint=False)])
    print(f"validation points {len(V):,}; soundings {len(S):,}; "
          f"hist14 shoreline pseudo-points {len(sh):,} at {SHORE_STEP_M:.0f} m, held at {SHORE_LEVEL:.2f} m", flush=True)
    return V, S, sh


def clusters(V):
    T = V[np.abs(V.res) > TAIL_M].copy()
    if len(T) < 2:
        T["cluster"] = np.arange(len(T)); return T
    Z = linkage(np.c_[T.x, T.y], "single")
    T["cluster"] = fcluster(Z, CLUSTER_M, "distance")
    return T


def neighbourhood_stats(px, py, S_xy, S_z, sh):
    """Exactly the neighbourhood hist14 uses: k = 32 nearest by Euclidean distance over soundings + shore pseudo-points."""
    tr = np.vstack([S_xy, sh])
    is_shore = np.r_[np.zeros(len(S_xy), bool), np.ones(len(sh), bool)]
    z_tr = np.r_[S_z, np.full(len(sh), SHORE_LEVEL)]
    tree = cKDTree(tr)
    d, idx = tree.query(np.c_[px, py], k=min(OK_K, len(tr)))
    snd_tree = cKDTree(S_xy)
    out = []
    for i in range(len(px)):
        ii = idx[i]; dd = d[i]
        nsh = int(is_shore[ii].sum())
        b = np.degrees(np.arctan2(tr[ii, 1] - py[i], tr[ii, 0] - px[i])) % 360.0
        bs = np.sort(b); gaps = np.diff(np.r_[bs, bs[0] + 360.0])
        R = np.hypot(np.cos(np.radians(b)).mean(), np.sin(np.radians(b)).mean())   # 0 = isotropic, 1 = all one side
        inside_nb = np.array([POOL.contains(Point(*tr[j])) for j in ii])
        rings = {f"n_snd_{r}m": int(len(snd_tree.query_ball_point([px[i], py[i]], r=r))) for r in (100, 250, 500, 1000, 2000)}
        out.append(dict(d_nearest_any=float(dd[0]), d_nearest_sounding=float(snd_tree.query([px[i], py[i]], k=1)[0]),
                        k_used=len(ii), n_shore_in_k=nsh, n_sounding_in_k=len(ii) - nsh,
                        d_k_max=float(dd[-1]), anisotropy_R=float(R), max_bearing_gap_deg=float(gaps.max()),
                        n_neighbours_outside_pool=int((~inside_nb).sum()),
                        neighbour_bed_p50=float(np.median(z_tr[ii])), neighbour_bed_min=float(z_tr[ii].min()),
                        neighbour_bed_max=float(z_tr[ii].max()), **rings))
    return pd.DataFrame(out), tr, is_shore, z_tr, tree


# ------------------------------------------------------------------ A. overview

def overview(V, S, sh, T):
    x0, y0, x1, y1 = POOL.bounds
    px = 0.03 * (x1 - x0); x0 -= px; x1 += px; y0 -= px; y1 += px
    fig, ax = plt.subplots(figsize=(19, 11))
    hillshade(ax, "ZONE_1_KAKHOVKA_LOWER_DNIPRO", x0, x1, y0, y1, alpha=0.9)
    draw_poly(ax, POOL, C["pool"], 1.8, "registry pool polygon (full pool, trusted basin)")
    raster_contour(ax, ISL, x0, x1, y0, y1, [0.5], C["island"], 1.0, "p64 island mask (contour)", nearest=True)
    raster_contour(ax, WFRAC, x0, x1, y0, y1, [50.0], C["optical"], 0.9, "pre-breach optical water 50 % (contour)")
    if CONT_GPKG.exists():
        try:
            g = gpd.read_file(CONT_GPKG).to_crs(CFG.CRS_METRIC)
            g.boundary.plot(ax=ax, color=C["contour"], lw=0.6, zorder=6)
            ax.plot([], [], color=C["contour"], lw=0.6, label="pre-breach continuous contours")
        except Exception as e:
            print(f"  contours gpkg skipped: {type(e).__name__}", flush=True)
    ax.scatter(S.x, S.y, s=1.2, c=C["snd"], alpha=0.55, lw=0, zorder=4, label=f"soundings ({len(S):,})")
    sub = V.sample(min(len(V), 30000), random_state=0)
    ax.scatter(sub.x, sub.y, s=0.5, c=C["ice"], alpha=0.25, lw=0, zorder=3,
               label=f"ICESat-2 ground on source 2 ({len(V):,}, {len(sub):,} drawn)")
    neg = T[T.res < 0]; pos = T[T.res > 0]
    ax.scatter(pos.x, pos.y, s=46, facecolor="none", edgecolor=C["pos"], lw=1.5, zorder=9,
               label=f"residual > +5 m ({len(pos)})")
    ax.scatter(neg.x, neg.y, s=52, facecolor="none", edgecolor=C["neg"], lw=1.8, zorder=10,
               label=f"residual < −5 m ({len(neg)})")
    w = V.loc[V.res.idxmin()]
    ax.scatter([w.x], [w.y], s=150, marker="*", c=C["worst"], zorder=11, label=f"worst residual {w.res:.1f} m")
    frame(ax, x0, x1, y0, y1, "A1. Source 2 (hist20 reservoir bed) — every mask as a contour, soundings, "
                              "ICESat-2 ground and the residual extremes of both signs")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.95)
    fig.tight_layout(); fig.savefig(FIG / "p55g_A1_overview.png", dpi=150); fig.savefig(FIG / "p55g_A1_overview.pdf")
    plt.close(fig); print("-> p55g_A1_overview.png/.pdf", flush=True)


def overview_edge(V, S, sh, T):
    """A2: the edge belt only. Buffers are drawn INWARD from the registry boundary."""
    x0, y0, x1, y1 = POOL.bounds
    px = 0.03 * (x1 - x0); x0 -= px; x1 += px; y0 -= px; y1 += px
    fig, ax = plt.subplots(figsize=(19, 11))
    for r, col, a_ in ((250, "#fee0d2", 0.95), (100, "#fc9272", 0.95), (50, "#de2d26", 0.95)):
        ring = POOL.difference(POOL.buffer(-r))
        gpd.GeoSeries([ring], crs=CFG.CRS_METRIC).plot(ax=ax, color=col, alpha=a_, zorder=2, lw=0)
    draw_poly(ax, POOL, C["pool"], 1.6, "registry pool polygon")
    raster_contour(ax, ISL, x0, x1, y0, y1, [0.5], C["island"], 1.0, "p64 island mask", nearest=True)
    raster_contour(ax, WFRAC, x0, x1, y0, y1, [50.0], C["optical"], 0.9, "pre-breach optical water 50 %")
    edge = V[V.dist_to_boundary_m <= 250]
    ax.scatter(S.x, S.y, s=1.0, c=C["snd"], alpha=0.45, lw=0, zorder=4, label=f"soundings ({len(S):,})")
    ax.scatter(sh[:, 0], sh[:, 1], s=7, marker="s", c="#6a51a3", zorder=5,
               label=f"hist14 shore pseudo-points, {SHORE_STEP_M:.0f} m @ {SHORE_LEVEL:.2f} m ({len(sh):,})")
    ax.scatter(edge.x, edge.y, s=1.6, c=C["ice"], alpha=0.5, lw=0, zorder=6,
               label=f"ICESat-2 ground within 250 m of the boundary ({len(edge):,})")
    neg = T[T.res < 0]; pos = T[T.res > 0]
    ax.scatter(pos.x, pos.y, s=46, facecolor="none", edgecolor=C["pos"], lw=1.5, zorder=9, label="residual > +5 m")
    ax.scatter(neg.x, neg.y, s=52, facecolor="none", edgecolor=C["neg"], lw=1.8, zorder=10, label="residual < −5 m")
    h = [Patch(fc="#de2d26", label="0–50 m from the boundary"), Patch(fc="#fc9272", label="50–100 m"),
         Patch(fc="#fee0d2", label="100–250 m")]
    frame(ax, x0, x1, y0, y1, "A2. Edge belt: inward buffers from the registry boundary, with the mask contours, "
                              "the soundings and the shoreline pseudo-points the interpolator actually sees")
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.95)
    fig.tight_layout(); fig.savefig(FIG / "p55g_A2_edge_belt.png", dpi=150); fig.savefig(FIG / "p55g_A2_edge_belt.pdf")
    plt.close(fig); print("-> p55g_A2_edge_belt.png/.pdf", flush=True)


# ------------------------------------------------------------------ B. cluster panels

def cluster_panel(cid, G, V, S, sh, tag):
    cx, cy = float(G.x.mean()), float(G.y.mean())
    x0, x1, y0, y1 = cx - PAD_M, cx + PAD_M, cy - PAD_M, cy + PAD_M
    zone = G.zone.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(19, 9.2))

    ax = axes[0]
    bed, _ = read_window(BED30, x0, x1, y0, y1)
    if bed is not None and np.isfinite(bed).any():
        im = ax.imshow(bed, extent=(x0, x1, y0, y1), cmap="viridis", zorder=2, interpolation="nearest",
                       vmin=float(np.nanpercentile(bed, 2)), vmax=float(np.nanpercentile(bed, 98)))
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label="hist20 bed, m EVRF2019")
    draw_poly(ax, POOL, C["pool"], 2.2, "registry pool boundary")
    raster_contour(ax, ISL, x0, x1, y0, y1, [0.5], C["island"], 1.4, "p64 island mask", nearest=True)
    raster_contour(ax, WFRAC, x0, x1, y0, y1, [50.0], C["optical"], 1.2, "optical water 50 %")
    loc = S[(S.x > x0) & (S.x < x1) & (S.y > y0) & (S.y < y1)]
    ax.scatter(loc.x, loc.y, s=22, c=C["snd"], edgecolor="k", lw=0.3, zorder=8, label=f"soundings in frame ({len(loc)})")
    m = (sh[:, 0] > x0) & (sh[:, 0] < x1) & (sh[:, 1] > y0) & (sh[:, 1] < y1)
    ax.scatter(sh[m, 0], sh[m, 1], s=42, marker="s", c="#6a51a3", edgecolor="k", lw=0.4, zorder=8,
               label=f"shore pseudo-points @ {SHORE_LEVEL:.2f} m ({int(m.sum())})")
    near = V[(V.x > x0) & (V.x < x1) & (V.y > y0) & (V.y < y1)]
    ok = near[np.abs(near.res) <= TAIL_M]
    ax.scatter(ok.x, ok.y, s=10, c=C["ice"], alpha=0.65, lw=0, zorder=7, label=f"ICESat-2, |res| <= 5 m ({len(ok)})")
    # Numbered markers plus ONE side table: per-point annotation boxes overlapped illegibly once a cluster held
    # more than about four points.
    Gs = G.sort_values("y", ascending=False).reset_index(drop=True)
    for i, p in Gs.iterrows():
        col = C["neg"] if p.res < 0 else C["pos"]
        ax.scatter([p.x], [p.y], s=115, facecolor="none", edgecolor=col, lw=2.2, zorder=11)
        ax.annotate(str(i + 1), (p.x, p.y), ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color=col, zorder=12)
    lines = [f"{'#':>2} {'bed':>6} {'ICE':>6} {'res':>6} {'d_bnd':>6} {'d_snd':>6}"]
    for i, p in Gs.iterrows():
        lines.append(f"{i+1:>2} {p.dem:6.1f} {p.H_ice:6.1f} {p.res:+6.1f} "
                     f"{p.dist_to_boundary_m:6.0f} {p.dist_to_sounding_m:6.0f}")
    ax.text(0.985, 0.015, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
            family="monospace", fontsize=6.6, zorder=13,
            bbox=dict(fc="w", ec="#555555", lw=0.7, alpha=0.93, pad=3))
    frame(ax, x0, x1, y0, y1, f"B. cluster {cid} ({tag}) — hist20 bed, mask contours, soundings, ICESat-2\n"
                              f"bed / ICE = ICESat-2 ground / res / distances in m")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92)

    ax = axes[1]
    fab = hillshade(ax, zone, x0, x1, y0, y1)
    if fab is not None:
        cs = ax.contour(np.linspace(x0, x1, fab.shape[1]), np.linspace(y1, y0, fab.shape[0]),
                        np.nan_to_num(fab, nan=0), levels=np.arange(-10, 60, 2.5), colors=["#333333"],
                        linewidths=0.45, zorder=4)
        ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.0f")
    draw_poly(ax, POOL, C["pool"], 2.2, "registry pool boundary")
    raster_contour(ax, ISL, x0, x1, y0, y1, [0.5], C["island"], 1.4, "p64 island mask", nearest=True)
    ax.scatter(loc.x, loc.y, s=18, c=C["snd"], edgecolor="k", lw=0.3, zorder=8, label="soundings")
    for i, p in Gs.iterrows():
        ax.scatter([p.x], [p.y], s=115, facecolor="none", edgecolor=C["neg"] if p.res < 0 else C["pos"], lw=2.2, zorder=11)
        ax.annotate(str(i + 1), (p.x, p.y), ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=12)
    ax.text(0.985, 0.015,
            f"FABDEM p50 {Gs.fab.median():.2f} m\nslope p50 {Gs.fabdem_slope_deg.median():.1f}°\n"
            f"relief 200 m p50 {Gs.fabdem_relief_200m_m.median():.1f} m\n"
            f"ICESat-2 ground p50 {Gs.H_ice.median():.2f} m\n"
            f"shore constraint {SHORE_LEVEL:.2f} m  ({SHORE_LEVEL - Gs.H_ice.median():+.2f} m vs ground)",
            transform=ax.transAxes, ha="right", va="bottom", family="monospace", fontsize=7.2, zorder=13,
            bbox=dict(fc="w", ec="#555555", lw=0.7, alpha=0.93, pad=3))
    frame(ax, x0, x1, y0, y1, f"B. cluster {cid} — FABDEM hillshade + 2.5 m contours (terrain context)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
    fig.tight_layout(); fig.savefig(FIG / f"p55g_B_cluster_{cid:02d}.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------ C. kriging-support panels

def support_panel(cid, G, S, sh, tr, is_shore, tree):
    cx, cy = float(G.x.mean()), float(G.y.mean())
    x0, x1, y0, y1 = cx - 2 * PAD_M, cx + 2 * PAD_M, cy - 2 * PAD_M, cy + 2 * PAD_M
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.6))

    # C1 sounding density on a 100 m grid
    ax = axes[0]
    step = 100.0
    gx = np.arange(x0, x1, step); gy = np.arange(y0, y1, step)
    XX, YY = np.meshgrid(gx + step / 2, gy + step / 2)
    snd_tree = cKDTree(np.c_[S.x, S.y])
    dens = np.array(snd_tree.query_ball_point(np.c_[XX.ravel(), YY.ravel()], r=250.0, return_length=True),
                    float).reshape(XX.shape)
    im = ax.pcolormesh(gx, gy, dens, cmap="magma", shading="auto", zorder=2)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label="soundings within 250 m of the cell")
    draw_poly(ax, POOL, "#ffffff", 2.0, "registry pool boundary")
    ax.scatter(G.x, G.y, s=90, facecolor="none", edgecolor="#39ff14", lw=2.0, zorder=10, label="problem points")
    frame(ax, x0, x1, y0, y1, f"C1. cluster {cid} — sounding density (count within 250 m)")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # C2 distance to the nearest sounding, as contours over the same window
    ax = axes[1]
    dist = snd_tree.query(np.c_[XX.ravel(), YY.ravel()], k=1)[0].reshape(XX.shape)
    im = ax.pcolormesh(gx, gy, dist, cmap="cividis_r", shading="auto", zorder=2)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label="distance to the nearest sounding, m")
    cs = ax.contour(gx + step / 2, gy + step / 2, dist, levels=[100, 250, 500, 1000], colors=["w"], linewidths=1.0, zorder=5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f m")
    draw_poly(ax, POOL, "#ff0000", 2.0, "registry pool boundary")
    ax.scatter(G.x, G.y, s=90, facecolor="none", edgecolor="#39ff14", lw=2.0, zorder=10, label="problem points")
    frame(ax, x0, x1, y0, y1, f"C2. cluster {cid} — distance to the nearest sounding")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # C3 the actual k = 32 Euclidean neighbourhood, with neighbours outside the pool in red
    ax = axes[2]
    draw_poly(ax, POOL, C["pool"], 2.0, "registry pool boundary")
    raster_contour(ax, ISL, x0, x1, y0, y1, [0.5], C["island"], 1.2, "p64 island mask", nearest=True)
    d, idx = tree.query(np.c_[G.x.values, G.y.values], k=min(OK_K, len(tr)))
    n_out_tot = 0
    for i in range(len(G)):
        for j, jj in enumerate(idx[i]):
            inside = POOL.contains(Point(*tr[jj]))
            n_out_tot += (not inside)
            ax.plot([G.x.values[i], tr[jj, 0]], [G.y.values[i], tr[jj, 1]],
                    color=C["nn_in"] if inside else C["nn_out"], lw=0.55, alpha=0.75, zorder=6)
        ax.scatter([G.x.values[i]], [G.y.values[i]], s=110, facecolor="none", edgecolor="#39ff14", lw=2.2, zorder=11)
    nb = np.unique(idx.ravel())
    ax.scatter(tr[nb[~is_shore[nb]], 0], tr[nb[~is_shore[nb]], 1], s=24, c=C["snd"], edgecolor="k", lw=0.3, zorder=9,
               label="sounding in the k = 32 neighbourhood")
    ax.scatter(tr[nb[is_shore[nb]], 0], tr[nb[is_shore[nb]], 1], s=44, marker="s", c="#6a51a3", edgecolor="k", lw=0.4,
               zorder=9, label="shore pseudo-point in the neighbourhood")
    ax.plot([], [], color=C["nn_in"], lw=1.2, label="neighbour inside the pool")
    ax.plot([], [], color=C["nn_out"], lw=1.2, label="neighbour OUTSIDE the pool (across the shore)")
    frame(ax, x0, x1, y0, y1, f"C3. cluster {cid} — the k = {OK_K} EUCLIDEAN neighbourhood hist14 actually uses\n"
                              f"{n_out_tot} of {len(G)*OK_K} neighbour links cross the pool boundary")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG / f"p55g_C_support_{cid:02d}.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------ sensitivity

def fit_variogram(S, rng, n=4000, max_km=8.0, nbin=24, knn=40):
    """One spherical variogram, refitted here and then held FIXED across every neighbourhood variant.

    Pairs come from each sampled point's `knn` nearest neighbours, not from random pairs: the reservoir is ~200 km long, so
    random pairs are almost all beyond the lag range of interest and the short-lag bins come out empty.
    """
    x = S.x.values; y = S.y.values; z = S.H_bed_evrf2019_m.values
    xy = np.c_[x, y]
    i0 = rng.choice(len(S), min(n, len(S)), replace=False)
    _, nb = cKDTree(xy).query(xy[i0], k=min(knn + 1, len(S)))
    i = np.repeat(i0, nb.shape[1] - 1); j = nb[:, 1:].ravel()
    h = np.hypot(x[i] - x[j], y[i] - y[j]); g = 0.5 * (z[i] - z[j]) ** 2
    keep = (h > 0) & (h <= max_km * 1000)
    h, g = h[keep], g[keep]
    if len(h) < 500:
        raise SystemExit("variogram: too few short-lag pairs; widen knn or max_km")
    edges = np.linspace(0, max_km * 1000, nbin + 1); w = np.digitize(h, edges) - 1
    hh, gg = [], []
    for k in range(nbin):
        m = w == k
        if m.sum() >= 50:
            hh.append(h[m].mean()); gg.append(g[m].mean())
    hh = np.array(hh); gg = np.array(gg)
    sill = float(np.mean(gg[-max(len(gg) // 3, 1):]))
    best, err = None, np.inf
    for rg in np.linspace(300, max_km * 1000, 60):
        e = float(np.mean((H14.spherical(hh, rg, sill, 0.0) - gg) ** 2))
        if e < err:
            err, best = e, rg
    return (float(best), sill, 0.0), hh, gg


def sensitivity(T, S, sh, vg, top_ids):
    """Change ONLY the neighbourhood; the variogram and the data are identical across variants."""
    S_xy = np.c_[S.x, S.y]; S_z = S.H_bed_evrf2019_m.values
    bnd = POOL.boundary
    dense = np.array([bnd.interpolate(t, normalized=True).coords[0]
                      for t in np.linspace(0, 1, max(int(bnd.length / 50.0), 100), endpoint=False)])
    rows = []
    for cid in top_ids:
        G = T[T.cluster == cid]
        te = np.c_[G.x.values, G.y.values]
        variants = {
            "as built: soundings + shore @250 m, k=32": (np.vstack([S_xy, sh]), np.r_[S_z, np.full(len(sh), SHORE_LEVEL)], 32),
            "k=16": (np.vstack([S_xy, sh]), np.r_[S_z, np.full(len(sh), SHORE_LEVEL)], 16),
            "k=64": (np.vstack([S_xy, sh]), np.r_[S_z, np.full(len(sh), SHORE_LEVEL)], 64),
            "k=128": (np.vstack([S_xy, sh]), np.r_[S_z, np.full(len(sh), SHORE_LEVEL)], 128),
            "soundings only, no shore constraint, k=32": (S_xy, S_z, 32),
            "shore densified 250 -> 50 m, k=32": (np.vstack([S_xy, dense]), np.r_[S_z, np.full(len(dense), SHORE_LEVEL)], 32),
            "shore densified 50 m + k=64": (np.vstack([S_xy, dense]), np.r_[S_z, np.full(len(dense), SHORE_LEVEL)], 64),
        }
        for name, (xy, z, k) in variants.items():
            old_k = H14.OK_K
            H14.OK_K = k
            try:
                pred = H14._ok(xy, z, te, vg)
            finally:
                H14.OK_K = old_k
            res = pred - G.H_ice.values
            rows.append(dict(cluster=cid, variant=name, k=k, n_train=len(xy), n_points=len(G),
                             pred_p50=round(float(np.median(pred)), 2),
                             icesat2_p50=round(float(np.median(G.H_ice.values)), 2),
                             as_built_p50=round(float(np.median(G.dem.values)), 2),
                             residual_p50=round(float(np.median(res)), 2),
                             residual_worst=round(float(res.min()), 2),
                             abs_residual_mean=round(float(np.mean(np.abs(res))), 2)))
            print(f"  cluster {cid} | {name:45s} -> median residual {np.median(res):+6.2f} m, worst {res.min():+6.2f}", flush=True)
    return pd.DataFrame(rows)


def shore_figure(V, T, CL, SENS):
    """D. The boundary CONDITION, not the point density: where the shore constraint sits against the measured ground."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    ax = axes[0]
    b = V[V.dist_to_boundary_m <= 1000]
    bins = [(0, 25), (25, 50), (50, 100), (100, 250), (250, 500), (500, 1000)]
    lab, med, p10, p90 = [], [], [], []
    for lo, hi in bins:
        s = b[(b.dist_to_boundary_m >= lo) & (b.dist_to_boundary_m < hi)]
        lab.append(f"{lo}-{hi}"); med.append(s.H_ice.median())
        p10.append(s.H_ice.quantile(.10)); p90.append(s.H_ice.quantile(.90))
    xp = np.arange(len(bins))
    ax.fill_between(xp, p10, p90, color=C["ice"], alpha=0.22, label="ICESat-2 ground, p10-p90")
    ax.plot(xp, med, "o-", color=C["ice"], lw=1.8, label="ICESat-2 ground, median")
    ax.axhline(SHORE_LEVEL, color=C["neg"], lw=2.0, ls="--", label=f"hist14 shore constraint, {SHORE_LEVEL:.2f} m (epoch)")
    ax.axhline(H14.SHORE_VARIANTS["npg"], color="#e08214", lw=1.4, ls=":", label=f"'npg' variant, {H14.SHORE_VARIANTS['npg']:.2f} m")
    ax.set_xticks(xp); ax.set_xticklabels(lab, fontsize=8); ax.set_xlabel("distance to the registry boundary, m")
    ax.set_ylabel("m EVRF2019"); ax.grid(alpha=0.3)
    ax.set_title("D1. The shoreline pseudo-points are held ABOVE the measured ground\n"
                 "within 50 m of the boundary 100 % of the ground is below 17.08 m")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    P = pd.read_csv(CFG.TABLES / "p55g_point_diagnostics.csv")
    for grp, col in (("tail", C["neg"]), ("edge control |res| <= 1 m", C["ice"])):
        s = P[P.group == grp]
        ax.hist(s.n_shore_in_k, bins=np.arange(0, 34, 2), alpha=0.55, color=col, label=f"{grp} (n={len(s)})")
    ax.axvline(OK_K / 2, color="k", ls="--", lw=1.0, label="half the neighbourhood")
    ax.set_xlabel(f"shore pseudo-points among the k = {OK_K} nearest neighbours")
    ax.set_ylabel("points"); ax.grid(alpha=0.3)
    ax.set_title("D2. Half the kriging neighbourhood near the shore IS the constraint\n"
                 "median 15-16 of 32 for tail and control alike")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    for cid, col in zip(SENS.cluster.unique(), (C["neg"], C["pos"])):
        s = SENS[SENS.cluster == cid]
        ax.barh(np.arange(len(s)) + (0.4 if col == C["pos"] else 0.0), s.residual_p50, height=0.38, color=col,
                label=f"cluster {cid} (n={int(s.n_points.iloc[0])}, ICESat-2 {s.icesat2_p50.iloc[0]:.1f} m)")
        ax.set_yticks(np.arange(len(s)) + 0.2); ax.set_yticklabels(s.variant, fontsize=7.5)
    ax.axvline(0, color="k", lw=1.0)
    ax.invert_yaxis(); ax.set_xlabel("median residual of the local re-solve, m (0 = matches ICESat-2 ground)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("D3. Neighbourhood sensitivity: the two largest clusters move in\nOPPOSITE directions, "
                 "so one global shore level cannot serve both")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "p55g_D_shore_constraint.png", dpi=150)
    fig.savefig(FIG / "p55g_D_shore_constraint.pdf"); plt.close(fig)
    print("-> p55g_D_shore_constraint.png/.pdf", flush=True)


# ------------------------------------------------------------------ main

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-maps", action="store_true")
    A = ap.parse_args()
    pd.set_option("display.width", 250)
    rng = np.random.default_rng(42)
    V, S, sh = load()
    T = clusters(V)
    S_xy = np.c_[S.x, S.y]; S_z = S.H_bed_evrf2019_m.values

    # ---- per-point neighbourhood diagnosis, for the tail AND for a matched non-tail edge control
    ctrl = V[(np.abs(V.res) <= 1.0) & (V.dist_to_boundary_m <= 250)].sample(
        min(300, int(((np.abs(V.res) <= 1.0) & (V.dist_to_boundary_m <= 250)).sum())), random_state=0)
    P = pd.concat([T.assign(group="tail"), ctrl.assign(group="edge control |res| <= 1 m")], ignore_index=True)
    N, tr, is_shore, z_tr, tree = neighbourhood_stats(P.x.values, P.y.values, S_xy, S_z, sh)
    P = pd.concat([P.reset_index(drop=True), N], axis=1)
    P.to_csv(CFG.TABLES / "p55g_point_diagnostics.csv", index=False)

    cols = ["d_nearest_sounding", "n_snd_100m", "n_snd_250m", "n_snd_500m", "n_snd_1000m", "n_snd_2000m",
            "n_shore_in_k", "d_k_max", "anisotropy_R", "max_bearing_gap_deg", "n_neighbours_outside_pool",
            "dist_to_boundary_m", "fabdem_slope_deg", "fabdem_relief_200m_m"]
    CMP = P.groupby("group")[cols].median().round(2).T
    CMP.to_csv(CFG.TABLES / "p55g_tail_vs_control.csv")
    print("\nTAIL vs MATCHED EDGE CONTROL (medians)"); print(CMP.to_string())

    # ---- cluster table
    rows = []
    for cid, G in T.groupby("cluster"):
        g = P[(P.group == "tail") & (P.cluster == cid)]
        rows.append(dict(cluster_id=int(cid), n_points=len(G), zone=G.zone.iloc[0],
                         x=round(float(G.x.mean()), 1), y=round(float(G.y.mean()), 1),
                         worst_residual=round(float(G.res.min()), 2), median_residual=round(float(G.res.median()), 2),
                         n_negative=int((G.res < 0).sum()), n_positive=int((G.res > 0).sum()),
                         dist_to_boundary_p50=round(float(G.dist_to_boundary_m.median()), 1),
                         nearest_sounding_p50=round(float(g.d_nearest_sounding.median()), 1),
                         n_snd_250m_p50=float(g.n_snd_250m.median()), n_snd_500m_p50=float(g.n_snd_500m.median()),
                         n_shore_in_k_p50=float(g.n_shore_in_k.median()),
                         k_radius_p50=round(float(g.d_k_max.median()), 1),
                         anisotropy_R_p50=round(float(g.anisotropy_R.median()), 3),
                         nb_outside_pool_p50=float(g.n_neighbours_outside_pool.median()),
                         slope_p50=round(float(G.fabdem_slope_deg.median()), 2),
                         relief200_p50=round(float(G.fabdem_relief_200m_m.median()), 2)))
    CL = pd.DataFrame(rows).sort_values("n_points", ascending=False)

    def diagnose(r):
        if r.n_snd_250m_p50 >= 5 and r.nearest_sounding_p50 < 300:
            more = "unlikely"
            dx = "dense local support; boundary/geometry effect"
        elif r.nearest_sounding_p50 > 600 or r.n_snd_500m_p50 < 3:
            more = "yes"
            dx = "genuinely thin local support"
        else:
            more = "maybe"
            dx = "moderate support; check anisotropy"
        if r.nb_outside_pool_p50 > 0:
            dx += "; Euclidean neighbours cross the shore"
        if r.anisotropy_R_p50 > 0.5:
            dx += "; neighbours one-sided"
        return pd.Series(dict(tentative_diagnosis=dx, would_more_points_help=more))

    CL = pd.concat([CL, CL.apply(diagnose, axis=1)], axis=1)
    CL.to_csv(CFG.TABLES / "p55g_cluster_diagnostics.csv", index=False)
    print("\nCLUSTERS"); print(CL.to_string(index=False))

    # ---- edge belt vs interior summary
    rows = []
    for lab, m in (("edge belt <= 250 m", V.dist_to_boundary_m <= 250),
                   ("0-50 m", V.dist_to_boundary_m <= 50),
                   ("50-100 m", (V.dist_to_boundary_m > 50) & (V.dist_to_boundary_m <= 100)),
                   ("100-250 m", (V.dist_to_boundary_m > 100) & (V.dist_to_boundary_m <= 250)),
                   ("interior > 250 m", V.dist_to_boundary_m > 250)):
        g = V[m.values]
        rows.append(dict(stratum=lab, n=len(g), RMSE=round(float(np.sqrt((g.res ** 2).mean())), 3),
                         MAE=round(float(np.abs(g.res).mean()), 3), bias=round(float(g.res.mean()), 3),
                         P_res_below_m5=round(float((g.res < -5).mean()), 5),
                         P_res_above_p5=round(float((g.res > 5).mean()), 5),
                         fabdem_RMSE=round(float(np.sqrt((g.fab_res ** 2).mean())), 3)))
    SUM = pd.DataFrame(rows); SUM.to_csv(CFG.TABLES / "p55g_edge_vs_interior.csv", index=False)
    print("\nEDGE BELT vs INTERIOR"); print(SUM.to_string(index=False))

    # ---- maps
    if not A.skip_maps:
        overview(V, S, sh, T); overview_edge(V, S, sh, T)
    top = CL[CL.n_points >= 2].cluster_id.tolist()
    worst_cid = int(T.loc[T.res.idxmin(), "cluster"])
    todo = list(dict.fromkeys(top + [worst_cid]))
    for cid in (todo if not A.skip_maps else []):
        G = T[T.cluster == cid]
        tag = "largest" if cid == CL.cluster_id.iloc[0] else ("worst residual" if cid == worst_cid else "2+ extremes")
        cluster_panel(cid, G, V, S, sh, tag)
        support_panel(cid, G, S, sh, tr, is_shore, tree)
        print(f"-> cluster {cid}: B and C panels ({len(G)} points)", flush=True)

    # ---- sensitivity on the two largest clusters
    vg, hh, gg = fit_variogram(S, rng)
    print(f"\nrefitted spherical variogram: range {vg[0]/1000:.2f} km, sill {vg[1]:.1f} m2, nugget {vg[2]:.1f}", flush=True)
    SENS = sensitivity(T, S, sh, vg, CL.cluster_id.tolist()[:2])
    SENS.to_csv(CFG.TABLES / "p55g_neighbourhood_sensitivity.csv", index=False)
    print("\nNEIGHBOURHOOD SENSITIVITY"); print(SENS.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hh / 1000, gg, "o", color=C["snd"], label="empirical semivariance (soundings)")
    hs = np.linspace(0, hh.max(), 300)
    ax.plot(hs / 1000, H14.spherical(hs, *vg), color=C["pool"], lw=1.6,
            label=f"spherical fit: range {vg[0]/1000:.2f} km, sill {vg[1]:.1f} m²")
    ax.axvline(0.25, color=C["neg"], ls="--", lw=1.0, label="shore pseudo-point spacing 250 m")
    ax.set_xlabel("lag, km"); ax.set_ylabel("semivariance, m²"); ax.grid(alpha=0.3)
    ax.set_title("Variogram used for the neighbourhood sensitivity test (held fixed across variants)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "p55g_variogram.png", dpi=140); plt.close(fig)

    shore_figure(V, T, CL, SENS)
    print("\n-> outputs/figures/p55g/*.png|pdf")
    print("-> outputs/tables/p55g_{point_diagnostics,tail_vs_control,cluster_diagnostics,edge_vs_interior,neighbourhood_sensitivity}.csv")


if __name__ == "__main__":
    main()
