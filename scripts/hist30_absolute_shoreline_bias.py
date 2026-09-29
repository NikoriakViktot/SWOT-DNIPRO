#!/usr/bin/env python
"""HIST 30 — the ABSOLUTE shoreline bias of S2 and S1, against the soundings.

Gate 7C3 could only ever measure S1 against S2. It found -3.7 m, which is
consistent with both being right and equally consistent with both being wrong
together: S1 - true = +27 m and S2 - true = +31 m would produce the same -4 m.
Only an independent reference separates those, and hist29 established that the
soundings provide one at H3 and nowhere else.

    bias_S2,true = S2_H3 contour  -  sounding isobath at 14.189 m
    bias_S1,true = S1_H3 contour  -  sounding isobath at 14.189 m

THE REFERENCE IS SOUNDINGS ONLY. TIN-interpolated crossings of z = 14.189 m,
restricted to triangles that actually bracket the level and whose longest edge
is under 1 km. No S1, no S2, no shoreline boundary condition, no P20 footprint.

WHAT THIS CANNOT DO, stated before the numbers. The supported isobath is 55.1 km
against an H3 shoreline of 1,846 km -- about 3%. This measures the bias where
the survey reached the margin, which is not a random sample of the shoreline:
soundings exist where a vessel could go. A result here is evidence about
absolute placement, not a basin-wide correction, and it is NOT transferable to
H1 or H2, which have no bracketing soundings at all.

UNCERTAINTY OF THE REFERENCE ITSELF. A comparison is only as good as what it is
compared against, so the horizontal uncertainty of the isobath is carried:

    sigma_x,bathy = sqrt(sigma_z^2 + sigma_interp^2 + sigma_WSE^2) / |s_local|

with s_local the slope of the TIN plane AT the isobath -- the shallow margin,
median 0.0206 m/m. That is an order of magnitude steeper than the 0.0053 m/m
deep-bed slope whose misuse Gate 7C3 had to retire, and it is the correct
quantity here precisely because it is local to the waterline.

sigma_interp is not assumed: it is measured by leave-one-out cross-validation of
the TIN on the soundings that support this isobath.

Outputs
-------
outputs/tables/hist30_absolute_bias.csv
outputs/figures/historical_bathymetry/png/hist30_absolute_bias.png
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
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from hist25b_gate6_event_qualification import load_frozen_manifest
from hist25b_gate7_anchored_classifier import CELL, CACHE, build_grid, clean_water, db
from hist25b_gate7c1_pairing_audit import contour_points, match_normal

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
ISOBATHS = ROOT / "data/processed/bathymetry/sounding_isobaths.gpkg"
CONTOURS = ROOT / "data/processed/bathymetry/prebreach_contours_continuous.gpkg"
FIELDS = CFG.BULK_ROOT / "hist26_continuous_fields"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
STAGE = 14.188588
MAX_EDGE_M = 1000.0
MATCH_MAX_M = 250.0
DENSIFY_M = 5.0
SIGMA_Z_M = 0.15        # single-beam survey + datum transfer; ASSUMED, see note
SIGMA_WSE_M = 0.155     # half the H3 group's own 0.310 m level spread


def densify(geom, step=DENSIFY_M):
    import shapely
    gs = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    out = [np.asarray(shapely.segmentize(g, step).coords) for g in gs
           if g.length > 0]
    return np.vstack(out) if out else np.empty((0, 2))


def loo_sigma_interp(xy, z, near):
    """Leave-one-out TIN error on the soundings that support the isobath.

    Rebuilding the whole triangulation 248 times is unnecessary: dropping one
    point and interpolating it from its neighbours by an inverse-distance plane
    fit reproduces what the TIN does locally, and is what the isobath's vertical
    error actually depends on."""
    tree = cKDTree(xy)
    idx = np.where(near)[0]
    err = []
    for i in idx:
        d, j = tree.query(xy[i], k=9)
        j = j[1:]                      # drop the point itself
        if len(j) < 5:
            continue
        A = np.c_[xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1], np.ones(len(j))]
        try:
            c, *_ = np.linalg.lstsq(A, z[j], rcond=None)
        except Exception:
            continue
        err.append(z[i] - c[2])
    err = np.asarray(err)
    if len(err) < 20:
        return np.nan, 0
    return float(1.4826 * np.median(np.abs(err - np.median(err)))), len(err)


def sar_contour(event_id, G, gdf_h3):
    """The M3 external shoreline for one S1 event, as vertices with tangents."""
    npz = CACHE / f"{event_id}.npz"
    if not npz.exists():
        return None
    from rasterio.features import rasterize as rio_rasterize
    opt = rio_rasterize([(gdf_h3, 1)], out_shape=(G["ny"], G["nx"]),
                        transform=G["tr"], fill=0, dtype="uint8").astype(bool)
    z = np.load(npz)
    vv, vh, cov = z["vv"], z["vh"], z["cov"]
    obs = cov & G["inside"]
    vvd, vhd = db(vv), db(vh)
    d_in = ndimage.distance_transform_edt(opt) * CELL
    d_out = ndimage.distance_transform_edt(~opt) * CELL
    aw = opt & (d_in > 500) & obs
    al = G["inside"] & ~opt & (d_out > 100) & obs
    if aw.sum() < 500 or al.sum() < 500:
        return None
    Xw = np.c_[vvd[aw], vhd[aw]]; Xl = np.c_[vvd[al], vhd[al]]
    mw, ml = Xw.mean(0), Xl.mean(0)
    Sw = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
    wv = np.linalg.solve(Sw, ml - mw)
    cut = 0.5 * (wv @ mw + wv @ ml)
    disc = np.full(G["inside"].shape, np.nan, np.float32)
    disc[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) - cut
    if np.nanmedian(disc[aw]) > np.nanmedian(disc[al]):
        disc = -disc
    core = clean_water((disc < 0) & obs)
    lab, n = ndimage.label(core, structure=np.ones((3, 3), int))
    keep = np.zeros(n + 1, bool)
    keep[list(set(np.unique(lab[aw & core])) - {0})] = True
    core = keep[lab]
    k = np.ones((3, 3), bool)
    filled = ndimage.binary_fill_holes(core)
    band = (ndimage.binary_dilation(filled, k, 3)
            & ~ndimage.binary_erosion(filled, k, 3) & obs)
    return contour_points(disc, band, G), disc, obs


def measure(A, ref_xy, inside_water):
    """Signed distance from EACH REFERENCE POINT to the nearest contour vertex.

    Anchored on the reference, not on the contour, and this is not a detail. An
    earlier version queried the isobath from each shoreline vertex, so S1 and S2
    were each measured on whatever part of the isobath their own contour
    happened to reach - different subsets. Differencing two medians taken over
    different subsets is meaningless, and it showed: S1 - S2 came out at +52.3 m
    against Gate 7C3's independent -3.7 m for the same sensors. Anchoring on the
    reference gives one common support, so the two sensors are comparable to
    each other and to Gate 7C3.

    Sign follows the frozen convention: positive = the detected water extent is
    LARGER than the bathymetric reference."""
    if A is None or len(A) < 30:
        return None
    d, _ = cKDTree(A[:, :2]).query(ref_xy, k=1)
    ok = d <= MATCH_MAX_M
    if ok.sum() < 30:
        return None
    # is the REFERENCE point inside the detected water? if so the detected
    # extent reaches beyond it -> positive
    sign = np.where(inside_water(ref_xy[ok, 0], ref_xy[ok, 1]), +1.0, -1.0)
    out = np.full(len(ref_xy), np.nan)
    out[ok] = sign * d[ok]
    return out


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("HIST 30 — absolute shoreline bias of S2 and S1 at H3")
    print("=" * 78)
    print(f"  git {commit}")
    print("  reference: sounding-only TIN isobath at 14.189 m, 55.1 km")
    print("  this is ~3% of H3's shoreline, where the survey reached the")
    print("  margin -- evidence on absolute placement, not a basin correction")

    iso = gpd.read_file(ISOBATHS, layer="sounding_isobaths")
    iso = iso[iso.contour_id == "H3"]
    if iso.empty:
        raise SystemExit("no H3 sounding isobath; run hist29")
    ref_xy = densify(unary_union(iso.geometry.tolist()))
    ref_tree = cKDTree(ref_xy)
    print(f"  reference points: {len(ref_xy):,} at {DENSIFY_M:.0f} m spacing")

    # ---- uncertainty of the reference -----------------------------------
    P = pd.read_parquet(PRIMARY)
    xy = np.c_[P.x.to_numpy(), P.y.to_numpy()]
    z = P.H_bed_evrf2019_m.to_numpy()
    near = cKDTree(xy).query_ball_point  # noqa: F841
    dref, _ = ref_tree.query(xy, k=1)
    support = dref <= 2000.0
    s_interp, n_loo = loo_sigma_interp(xy, z, support)
    T = Delaunay(xy)
    tri = T.simplices
    edge = np.max([np.linalg.norm(xy[tri[:, a]] - xy[tri[:, b]], axis=1)
                   for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    zt = z[tri]
    keep = (zt.min(1) < STAGE) & (zt.max(1) > STAGE) & (edge <= MAX_EDGE_M)
    grads = []
    for t in tri[keep]:
        p, v = xy[t], z[t]
        A = np.c_[p[:, 0] - p[0, 0], p[:, 1] - p[0, 1], np.ones(3)]
        try:
            c, *_ = np.linalg.lstsq(A, v, rcond=None)
            grads.append(float(np.hypot(c[0], c[1])))
        except Exception:
            pass
    s_local = float(np.nanmedian(grads))
    sig_z_total = float(np.sqrt(SIGMA_Z_M ** 2 + (s_interp if np.isfinite(s_interp) else 0) ** 2
                                + SIGMA_WSE_M ** 2))
    sigma_x_bathy = sig_z_total / s_local
    print(f"\n  REFERENCE UNCERTAINTY")
    print(f"    sigma_z      {SIGMA_Z_M:.3f} m   ASSUMED survey + datum; "
          f"replace when survey metadata is available")
    print(f"    sigma_interp {s_interp:.3f} m   measured, leave-one-out on "
          f"{n_loo:,} supporting soundings")
    print(f"    sigma_WSE    {SIGMA_WSE_M:.3f} m   half the H3 group's own "
          f"0.310 m level spread")
    print(f"    combined     {sig_z_total:.3f} m vertical")
    print(f"    local slope  {s_local:.4f} m/m AT the isobath "
          f"(the deep-bed slope is 0.0053)")
    print(f"    -> sigma_x,bathy = {sigma_x_bathy:.1f} m horizontal")

    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)

    rows = []
    # ---- S2 --------------------------------------------------------------
    cg = gpd.read_file(CONTOURS, layer="contour_polygons").set_index("contour_id")
    h3poly = cg.loc["H3", "geometry"]
    zf = np.load(FIELDS / "H3_field.npz", allow_pickle=True)
    F2 = -zf["W"]; F2[~zf["valid"]] = np.nan
    wet2 = clean_water((F2 < 0) & zf["valid"] & G["inside"])
    lab, n = ndimage.label(wet2, structure=np.ones((3, 3), int))
    sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    conn2 = ndimage.binary_fill_holes(lab == (int(np.argmax(sz)) + 1))
    k = np.ones((3, 3), bool)
    band2 = (ndimage.binary_dilation(conn2, k, 3)
             & ~ndimage.binary_erosion(conn2, k, 3) & zf["valid"] & G["inside"])
    A2 = contour_points(F2, band2, G)

    def in_water_field(Farr):
        def f(X, Y):
            i = np.clip(((G["y1"] - Y) / CELL).astype(int), 0, G["ny"] - 1)
            j = np.clip(((X - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
            return np.nan_to_num(Farr[i, j], nan=1.0) < 0
        return f

    d2 = measure(A2, ref_xy, in_water_field(F2))
    m2 = None
    if d2 is not None:
        v = d2[np.isfinite(d2)]
        med = float(np.median(v))
        nm = float(1.4826 * np.median(np.abs(v - med)))
        m2 = np.c_[ref_xy[np.isfinite(d2)], v]
        print(f"\n  S2 (hist26 H3 continuous contour) vs the isobath")
        print(f"    reference points matched {len(v):,} of {len(ref_xy):,}")
        print(f"    bias_S2,true  median {med:+7.1f} m   NMAD {nm:6.1f}   "
              f"P90 {np.percentile(np.abs(v), 90):6.1f}")
        print(f"    against sigma_x,bathy {sigma_x_bathy:.1f} m -> "
              f"{'WITHIN' if abs(med) <= sigma_x_bathy else 'EXCEEDS'} "
              f"the reference's own uncertainty")
        rows.append(dict(sensor="S2", event_id="hist26_H3_composite",
                         n=len(v), length_km=float(len(v) * DENSIFY_M / 1000),
                         bias_median_m=med, nmad_m=nm,
                         p90_m=float(np.percentile(np.abs(v), 90)),
                         sigma_x_bathy_m=sigma_x_bathy))
        keepS2 = m2
        S2d = d2
    else:
        keepS2 = None
        S2d = None
        print("\n  S2: too few matches to the isobath")

    # ---- S1, per admitted H3 scene --------------------------------------
    adm = pd.read_csv(CFG.TABLES / "gate7c2b_admission.csv")
    h3 = adm[(adm.target == "H3") & adm.shoreline_constraint_usable.astype(bool)]
    print(f"\n  S1, per admitted H3 scene ({len(h3)} scenes)")
    keepS1 = []
    for r in h3.itertuples():
        out = sar_contour(r.event_id, G, h3poly)
        if out is None:
            print(f"    {r.event_id:26s} unusable")
            continue
        A1, disc, obs = out
        d1 = measure(A1, ref_xy, in_water_field(disc))
        if d1 is None:
            print(f"    {r.event_id:26s} too few matches")
            continue
        # COMMON SUPPORT: only reference points both sensors reached
        both = np.isfinite(d1) & (np.isfinite(S2d) if S2d is not None
                                  else np.isfinite(d1))
        v = d1[both]
        med = float(np.median(v))
        nm = float(1.4826 * np.median(np.abs(v - med)))
        paired_s2 = float(np.median(S2d[both])) if S2d is not None else np.nan
        print(f"    {r.event_id:26s} n {both.sum():6,d}  "
              f"bias_S1 {med:+7.1f}  bias_S2(same pts) {paired_s2:+7.1f}  "
              f"S1-S2 {med-paired_s2:+7.1f} m  NMAD {nm:6.1f}")
        rows.append(dict(sensor="S1", event_id=r.event_id, n=int(both.sum()),
                         length_km=float(both.sum() * DENSIFY_M / 1000),
                         bias_median_m=med, nmad_m=nm,
                         p90_m=float(np.percentile(np.abs(v), 90)),
                         paired_s2_bias_m=paired_s2,
                         s1_minus_s2_m=med - paired_s2,
                         sigma_x_bathy_m=sigma_x_bathy))
        keepS1.append(np.c_[ref_xy[both], v])

    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "hist30_absolute_bias.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'hist30_absolute_bias.csv'}")

    s1 = R[R.sensor == "S1"]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not s1.empty and keepS2 is not None:
        b1 = float(s1.bias_median_m.median())
        b2 = float(R[R.sensor == "S2"].bias_median_m.iloc[0])
        print(f"  bias_S2,true = {b2:+.1f} m")
        print(f"  bias_S1,true = {b1:+.1f} m   (median of {len(s1)} scenes, "
              f"range {s1.bias_median_m.min():+.1f} .. "
              f"{s1.bias_median_m.max():+.1f})")
        if "s1_minus_s2_m" in s1.columns:
            dd = float(s1.s1_minus_s2_m.median())
            print(f"  S1 - S2 on COMMON reference points: {dd:+.1f} m, against")
            print(f"    Gate 7C3's independent S1-S2 estimate of -3.7 m")
        big = max(abs(b1), abs(b2))
        print(f"\n  sigma_x,bathy = {sigma_x_bathy:.1f} m")
        if big <= sigma_x_bathy:
            print("  Neither sensor's absolute offset exceeds the reference's")
            print("  own horizontal uncertainty. The scenario Gate 7C3 could")
            print("  not exclude -- both sensors displaced together by tens of")
            print("  metres -- is ruled out at H3, on the 3% of the shoreline")
            print("  the survey reached.")
        else:
            print("  At least one sensor is displaced beyond the reference's")
            print("  own uncertainty. This is the common-mode offset Gate 7C3")
            print("  was structurally unable to see.")
    print("\n  NOT TRANSFERABLE. H1 and H2 have zero bracketing soundings, so")
    print("  this result does not license a correction at their stages. They")
    print("  stay absolute_validation = UNAVAILABLE and must be carried as")
    print("  soft constraints with conservative uncertainty.")
    figure(R, ref_xy, keepS2, keepS1, sigma_x_bathy)


def figure(R, ref_xy, m2, m1list, sig):
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    ax[0].plot(ref_xy[:, 0] / 1000, ref_xy[:, 1] / 1000, ".", ms=.6,
               color=GREY, label="sounding isobath 14.189 m")
    if m2 is not None:
        sc = ax[0].scatter(m2[:, 0] / 1000, m2[:, 1] / 1000, s=5,
                           c=np.clip(m2[:, 2], -60, 60), cmap="RdBu_r",
                           vmin=-60, vmax=60, linewidths=0)
        cb = fig.colorbar(sc, ax=ax[0], fraction=.03, pad=.01)
        cb.set_label("S2 offset from the isobath (m)", fontsize=8)
    ax[0].set_title("where the survey reached the margin", color=INK)
    ax[0].set_xlabel("easting (km)"); ax[0].set_ylabel("northing (km)")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8, markerscale=8)

    if m2 is not None:
        ax[1].hist(np.clip(m2[:, 2], -150, 150), bins=60, color=BLUE,
                   alpha=.75, label=f"S2  median {np.median(m2[:,2]):+.1f} m")
    if m1list:
        allv = np.concatenate([m[:, 2] for m in m1list])
        ax[1].hist(np.clip(allv, -150, 150), bins=60, color=AMBER, alpha=.6,
                   label=f"S1  median {np.median(allv):+.1f} m")
    ax[1].axvline(0, color=INK, lw=1.2)
    ax[1].axvspan(-sig, sig, color=GREEN, alpha=.15,
                  label=f"±sigma_x,bathy = {sig:.0f} m")
    ax[1].set_xlabel("offset from the sounding isobath (m)\n+ = detected water "
                     "extends beyond the reference")
    ax[1].set_ylabel("vertices"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("absolute bias, which S1-vs-S2 could never show", color=INK)

    s1 = R[R.sensor == "S1"]
    if not s1.empty:
        y = np.arange(len(s1))
        ax[2].errorbar(s1.bias_median_m, y, xerr=s1.nmad_m, fmt="o",
                       color=AMBER, capsize=3, label="S1 scenes")
        ax[2].set_yticks(y)
        ax[2].set_yticklabels([e.replace("_", " ")[:16] for e in s1.event_id],
                             fontsize=7)
    s2 = R[R.sensor == "S2"]
    if not s2.empty:
        ax[2].axvline(float(s2.bias_median_m.iloc[0]), color=BLUE, lw=1.4,
                      label="S2 composite")
    ax[2].axvline(0, color=INK, lw=1.2)
    ax[2].axvspan(-sig, sig, color=GREEN, alpha=.15)
    ax[2].set_xlabel("bias vs the sounding isobath (m)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="x")
    ax[2].set_title("per scene", color=INK)
    fig.suptitle("hist30 · absolute bias at H3, the only stage the soundings "
                 "bracket", color=INK, fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "hist30_absolute_bias.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
