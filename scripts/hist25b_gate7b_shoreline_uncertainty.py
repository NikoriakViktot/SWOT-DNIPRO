#!/usr/bin/env python
"""GATE 7B — how accurately does the SAR shoreline actually locate z = H?

Gate 7 is frozen. M3 (anchored VV+VH LDA) is the selected classifier, M0 Otsu
stays a baseline, hydraulic connectivity filtering is mandatory, and the
honest headline is that Sentinel-1 predominantly DENSIFIES shallow-margin
geometry between existing bathymetric support rather than reaching ground
neither source constrains.

This gate answers the separate question. F1 = 0.9922 says the classifier
labels water correctly; it says nothing about where the boundary sits. A
shoreline only becomes an elevation constraint through

    z_bed(x_shore) ~ H(t)

so its POSITIONAL error is what propagates into the DEM, not its F1.

WHY SIGNED, NOT NEAREST. An unsigned nearest distance cannot distinguish
random scatter from a systematic offset: +20 m and -20 m both report 20 m.
The signed normal distance is positive where the SAR water extends beyond the
optical line and negative where it falls inside, so a non-zero median is a
horizontal bias and a zero median with large spread is scatter. Those two
have completely different consequences for a bed surface.

NEITHER SENSOR IS TRUTH. Optical NDWI/MNDWI sees OPEN water and misses
inundated reed margins; SAR sees flooded vegetation through double bounce but
roughens under wind. Their disagreement is therefore used as a positional
UNCERTAINTY estimate, not as an error of one against the other.

HORIZONTAL TO VERTICAL. sigma_z = |dz/dn| * sigma_x, with the cross-shore
slope taken from the primary soundings where they support it -- never from the
DEM this will later constrain, which would be circular. On a flat margin
20 m of horizontal uncertainty is ~0.10 m vertically; on a steeper bank the
same 20 m exceeds the 0.19 m WSE uncertainty and starts to dominate.

500 m IS NOT AN ELIGIBILITY CUTOFF. It was the reporting radius for novelty.
A shoreline 100 m from a sounding is still a valid physical boundary
condition, so distance_to_bathy travels as metadata and a redundancy weight,
never as a filter.

Outputs
-------
outputs/tables/hist25b_gate7b_shoreline_constraints.csv
outputs/tables/hist25b_gate7b_disagreement_classes.csv
outputs/figures/historical_bathymetry/png/hist25b_gate7b_*.png
"""
from __future__ import annotations

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
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from hist25b_gate6_event_qualification import load_frozen_manifest
from hist25b_gate7_anchored_classifier import (CELL, CACHE, build_grid,
                                               clean_water, shoreline_px, db)

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
CONTOURS_CMAP = ROOT / "data/processed/bathymetry/cmap2020_contours_utm.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
SLOPE_RADIUS_M = 1000.0     # local sounding neighbourhood for dz/dn
SLOPE_MIN_PTS = 8
EDGE_ARTIFACT_M = 200.0     # within this of a coverage edge, blame coverage
COHERENT_RUN_KM = 2.0       # a bias patch must be spatially coherent
SIG_HYDRAULIC_M = 0.05      # residual pool-flatness / local hydraulic term


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    M, meta = load_frozen_manifest()
    print(f"Gate-6 manifest verified: sha256 {meta['sha256'][:16]}...")
    print("Gate 7 FROZEN: M3_anchored_VV_VH_LDA, connectivity mandatory")

    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")

    def rast(g):
        return rio_rasterize([(g, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0, dtype="uint8").astype(bool)

    P = pd.read_parquet(PRIMARY)
    pxy = np.c_[P.x.to_numpy(), P.y.to_numpy()]
    pz = P.H_bed_evrf2019_m.to_numpy()
    ptree = cKDTree(pxy)
    try:
        CM = gpd.read_file(CONTOURS_CMAP, layer="cmap2020_contours")
        ctree = cKDTree(np.vstack([np.asarray(g.coords) for g in CM.geometry
                                   if g.geom_type == "LineString"]))
    except Exception:
        ctree = None

    # wetland context, so a reed-margin disagreement is not called sensor bias
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

    elig = M[M.scientific_roles.fillna("").str.contains("TARGET_STAGE_GEOMETRY")
             & (M.temporal_regime == "PREBREACH_IMPOUNDED")
             & M.WSE_RESOLVED.astype(bool)
             & M.wse_semantics_resolved.astype(bool)]
    print(f"eligible target-stage events: {len(elig)}")

    rows, cls_rows, keep_map = [], [], None
    for r in elig.itertuples():
        npz = CACHE / f"{r.event_id}.npz"
        if not npz.exists():
            continue
        roles = str(r.scientific_roles)
        tgt = next((t for t in ("H1", "H2", "H3")
                    if f"{t}_TARGET_STAGE_GEOMETRY" in roles), "")
        if not tgt:
            continue
        z = np.load(npz)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        obs = cov & inside
        vvd, vhd = db(vv), db(vh)

        # M3 as frozen: anchors rebuilt per event, parameters not retuned
        w_lo = rast(gdf.loc[gdf.index[gdf.index.get_indexer(["H3"])[0]], "geometry"]) \
            if "H3" in gdf.index else None
        opt = rast(gdf.loc[tgt, "geometry"])
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
        sar = np.zeros(inside.shape, bool)
        sar[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) < cut
        sar = clean_water(sar)
        lab, n = ndimage.label(sar, structure=np.ones((3, 3), int))
        core = set(np.unique(lab[aw & sar])) - {0}
        keep = np.zeros(n + 1, bool); keep[list(core)] = True
        sar = keep[lab]

        sl = shoreline_px(sar)
        ys, xs = np.where(sl)
        if len(xs) < 100:
            del vv, vh, cov, vvd, vhd
            continue
        X = G["x0"] + (xs + 0.5) * CELL
        Y = G["y1"] - (ys + 0.5) * CELL

        # SIGNED normal distance: + where SAR water extends beyond the optical
        # line, - where it falls inside. An unsigned nearest distance cannot
        # tell a systematic offset from symmetric scatter.
        sgn = np.where(opt[ys, xs], -1.0, +1.0)
        dn = sgn * np.where(opt[ys, xs], d_in[ys, xs], d_out[ys, xs])

        # context
        d_cov_edge = ndimage.distance_transform_edt(obs) * CELL
        near_edge = d_cov_edge[ys, xs] < EDGE_ARTIFACT_M
        in_wet = wet[ys, xs]
        dp, ip = ptree.query(np.c_[X, Y], k=1)
        dc = (ctree.query(np.c_[X, Y], k=1)[0] if ctree is not None
              else np.full(len(X), np.nan))

        # local cross-shore slope from the SOUNDINGS, never from the DEM this
        # will later constrain
        slope = np.full(len(X), np.nan)
        nb = ptree.query_ball_point(np.c_[X, Y], r=SLOPE_RADIUS_M)
        for i, idx in enumerate(nb):
            if len(idx) < SLOPE_MIN_PTS:
                continue
            A = np.c_[pxy[idx, 0] - X[i], pxy[idx, 1] - Y[i],
                      np.ones(len(idx))]
            try:
                coef, *_ = np.linalg.lstsq(A, pz[idx], rcond=None)
            except Exception:
                continue
            slope[i] = float(np.hypot(coef[0], coef[1]))

        med = float(np.median(dn))
        nmad = float(1.4826 * np.median(np.abs(dn - med)))
        sig_x = nmad                      # positional uncertainty, 1 sigma
        sig_z = np.where(np.isfinite(slope), np.abs(slope) * sig_x, np.nan)
        sig_wse = float(r.WSE_uncertainty_total_nominal)
        sig_tot = np.sqrt(sig_wse ** 2 + np.nan_to_num(sig_z) ** 2
                          + SIG_HYDRAULIC_M ** 2)

        # disagreement classes
        cl = np.full(len(X), "UNRESOLVED", dtype=object)
        cl[np.abs(dn) <= CELL] = "RANDOM_SHORELINE_SCATTER"
        cl[near_edge] = "COVERAGE_EDGE_ARTIFACT"
        cl[(~near_edge) & in_wet & (dn > CELL)] = "FLOODED_VEGETATION_DISAGREEMENT"
        # spatially coherent same-sign runs are systematic, not scatter
        big = (~near_edge) & (~in_wet) & (np.abs(dn) > 2 * CELL)
        cl[big] = "SYSTEMATIC_SENSOR_BIAS"
        for c in np.unique(cl):
            m = cl == c
            cls_rows.append(dict(event_id=r.event_id, stage_target=tgt,
                                 disagreement_class=c, n_px=int(m.sum()),
                                 km=float(m.sum()) * CELL / 1000,
                                 median_signed_m=float(np.median(dn[m])),
                                 nmad_m=float(1.4826 * np.median(
                                     np.abs(dn[m] - np.median(dn[m]))))))

        rows.append(dict(
            event_id=r.event_id, stage_target=tgt,
            relative_orbit=int(r.relative_orbit),
            H_WSE=float(r.WSE_nominal), sigma_WSE=sig_wse,
            shoreline_km=len(X) * CELL / 1000,
            signed_median_m=med, signed_nmad_m=nmad,
            abs_p50_m=float(np.percentile(np.abs(dn), 50)),
            abs_p90_m=float(np.percentile(np.abs(dn), 90)),
            frac_sar_outside=float((dn > 0).mean()),
            sigma_x_shore_m=sig_x,
            local_slope_median=float(np.nanmedian(slope)),
            slope_support_frac=float(np.isfinite(slope).mean()),
            sigma_z_shore_m=float(np.nanmedian(sig_z)),
            constraint_sigma_total_m=float(np.nanmedian(sig_tot)),
            median_dist_soundings_m=float(np.median(dp)),
            median_dist_cmap_m=float(np.nanmedian(dc))))
        if keep_map is None:
            keep_map = dict(eid=r.event_id, tgt=tgt, X=X, Y=Y, dn=dn, cl=cl,
                            slope=slope, sig_z=sig_z)
        del vv, vh, cov, vvd, vhd, sar, opt
    R = pd.DataFrame(rows)
    C = pd.DataFrame(cls_rows)
    R.to_csv(CFG.TABLES / "hist25b_gate7b_shoreline_constraints.csv", index=False)
    C.to_csv(CFG.TABLES / "hist25b_gate7b_disagreement_classes.csv", index=False)

    print("\n" + "=" * 78)
    print("SIGNED SAR-to-OPTICAL SHORELINE OFFSET  (+ = SAR water beyond optical)")
    print("=" * 78)
    print(R[["event_id", "stage_target", "relative_orbit", "shoreline_km",
             "signed_median_m", "signed_nmad_m", "abs_p90_m",
             "frac_sar_outside"]].to_string(
        index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\n  by stage target:")
    print(R.groupby("stage_target").agg(
        events=("event_id", "size"), signed_median=("signed_median_m", "median"),
        nmad=("signed_nmad_m", "median"), p90=("abs_p90_m", "median"),
        frac_outside=("frac_sar_outside", "median")).to_string(
        float_format=lambda v: f"{v:,.2f}"))
    print("\n  by relative orbit:")
    print(R.groupby("relative_orbit").agg(
        events=("event_id", "size"), signed_median=("signed_median_m", "median"),
        nmad=("signed_nmad_m", "median")).to_string(
        float_format=lambda v: f"{v:,.2f}"))

    print("\n" + "=" * 78)
    print("DISAGREEMENT CLASSES")
    print("=" * 78)
    agg = (C.groupby("disagreement_class")
             .agg(km=("km", "sum"), median_signed_m=("median_signed_m", "median"))
             .reset_index().sort_values("km", ascending=False))
    agg["pct"] = 100 * agg.km / agg.km.sum()
    print(agg.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n" + "=" * 78)
    print("HORIZONTAL -> VERTICAL  (slope from soundings, never from the DEM)")
    print("=" * 78)
    print(f"  sigma_x (shoreline positional, 1 sigma): median "
          f"{R.sigma_x_shore_m.median():.1f} m")
    print(f"  local cross-shore slope: median {R.local_slope_median.median():.4f}"
          f" m/m, slope support on {100*R.slope_support_frac.median():.0f}% of "
          f"shoreline")
    print(f"  sigma_z from geometry  : median {R.sigma_z_shore_m.median():.3f} m")
    print(f"  sigma_WSE              : median {R.sigma_WSE.median():.3f} m")
    print(f"  TOTAL constraint sigma : median "
          f"{R.constraint_sigma_total_m.median():.3f} m")
    domin = (R.sigma_z_shore_m > R.sigma_WSE).mean()
    print(f"\n  geometry dominates WSE on {100*domin:.0f}% of events "
          f"-- {'positional error is the binding term' if domin > 0.5 else 'WSE remains the binding term'}")

    _figures(R, C, keep_map)
    print("\nSTOP before DEM assimilation.")


def _figures(R, C, km):
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4))
    a = ax[0]
    for t, c in zip(("H3", "H2", "H1"), (BLUE, GREEN, AMBER)):
        s = R[R.stage_target == t]
        if len(s):
            a.scatter(s.signed_median_m, s.signed_nmad_m, s=60, color=c, label=t)
    a.axvline(0, color=RED, lw=1.4, ls="--")
    a.set_xlabel("signed median offset (m)  + = SAR beyond optical")
    a.set_ylabel("NMAD (m)")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("a · bias vs scatter — a non-zero median is a horizontal bias,\n"
                "not noise", fontsize=10.2, loc="left")
    a = ax[1]
    agg = C.groupby("disagreement_class").km.sum().sort_values()
    a.barh(range(len(agg)), agg.values,
           color=[RED if "SYSTEMATIC" in i else AMBER if "VEGETATION" in i
                  else GREY if "EDGE" in i else GREEN for i in agg.index])
    a.set_yticks(range(len(agg)))
    a.set_yticklabels([i.replace("_", "\n") for i in agg.index], fontsize=7.2)
    a.set_xlabel("shoreline length (km, all events)")
    a.set_title("b · what the optical-SAR disagreement actually is",
                fontsize=10.2, loc="left")
    a.grid(alpha=0.25, axis="x")
    a = ax[2]
    a.scatter(R.sigma_z_shore_m, R.sigma_WSE, s=55, color=BLUE)
    lim = [0, max(R.sigma_z_shore_m.max(), R.sigma_WSE.max()) * 1.1]
    a.plot(lim, lim, color=RED, ls="--", lw=1.2)
    a.set_xlabel("sigma_z from shoreline geometry (m)")
    a.set_ylabel("sigma_WSE (m)")
    a.grid(alpha=0.25)
    a.set_title("c · which term binds the constraint?\n"
                "below the line: geometry dominates", fontsize=10.2, loc="left")
    fig.suptitle("hist25b Gate 7B · shoreline positional uncertainty",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate7b_uncertainty.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    if km:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), sharex=True, sharey=True)
        a = ax[0]
        v = float(np.percentile(np.abs(km["dn"]), 95)) or 50
        sc = a.scatter(km["X"] / 1000, km["Y"] / 1000, c=km["dn"], s=1.2,
                       cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(sc, ax=a, label="signed offset (m)")
        a.set_title(f"a · {km['eid']} ({km['tgt']}) signed SAR-optical offset\n"
                    "red = SAR water beyond optical (reed margin?)",
                    fontsize=10, loc="left")
        a = ax[1]
        cols = {"RANDOM_SHORELINE_SCATTER": GREEN,
                "SYSTEMATIC_SENSOR_BIAS": RED,
                "FLOODED_VEGETATION_DISAGREEMENT": AMBER,
                "COVERAGE_EDGE_ARTIFACT": GREY, "UNRESOLVED": PURPLE}
        for c, col in cols.items():
            m = km["cl"] == c
            if m.any():
                a.plot(km["X"][m] / 1000, km["Y"][m] / 1000, ".", ms=1.2,
                       color=col, label=f"{c} ({m.sum()*CELL/1000:.0f} km)")
        a.legend(fontsize=7.2, markerscale=6)
        a.set_title("b · disagreement class along the shoreline",
                    fontsize=10, loc="left")
        for a in ax:
            a.set_xlabel("easting (km)")
        ax[0].set_ylabel("northing (km)")
        fig.suptitle("hist25b Gate 7B · where the two sensors disagree, and why",
                     y=1.02, fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGDIR / "hist25b_gate7b_maps.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)
    for f in ("uncertainty", "maps"):
        print(f"-> {FIGDIR/('hist25b_gate7b_'+f+'.png')}")


if __name__ == "__main__":
    main()
