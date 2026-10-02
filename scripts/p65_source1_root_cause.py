#!/usr/bin/env python
"""P65 -- root cause of the source-1 defect (zone bed v2, p53, 3 744 km2 below the dam). DIAGNOSIS ONLY, nothing is repaired.

Why this is a gate (user, 2026-09-20). p63's FABDEM control settled the ambiguity that had protected this surface: on the SAME
night ICESat-2 ground segments the product is at bias -2.76 m, LE90 7.30 m, with 11.8 % of points more than 5 m below the
reference and a worst case of -45.3 m, while FABDEM at those very points sits at bias -0.065 m, LE90 1.32 m, tail 0.05 %. The
"validation is just contaminated by water" explanation is therefore excluded: where ICESat-2 sees ground, FABDEM agrees with it
and the derived surface does not. This is not a rim artefact either -- it is 3 744 km2 of the downstream terrain that HEC-RAS
would run on. Nothing is fixed here and no gate is invented: the job is to find what generates the bias and the heavy negative
tail, so that the repair is aimed rather than guessed.

The surface is decomposed along axes that separate the candidate mechanisms, all on the same night points:
    product - ICESat-2, FABDEM - ICESat-2, product - FABDEM          three differences, so the reference is never assumed
    zone, source provenance                                          which of p53's zone runs carries it
    distance to the nearest real sounding                            interpolation vs extrapolation
    distance to the shoreline / the 2019-2022 water polygon edge      the taper-to-MAL band
    distance to the channel centreline (SWORD)                        along/across the channel
    elevation, slope                                                  terrain dependence
    chainage (upstream/downstream position)                           a datum or level error would trend with it
    WorldCover class, water history                                   surface type
Four hypotheses are tested explicitly, and each row of the table says which one it supports:
    H1 height/depth semantics  -- a Z = WL - depth error, a wrong datum, a sign, or a correction applied twice: a near-constant
                                  offset, or one that tracks the assumed water level along the chainage.
    H2 extrapolation           -- the surface continues mathematically far from any real sounding: the residual grows with the
                                  distance to support and the tail lives there.
    H3 pseudo/boundary points  -- one family of constraints pulls the surface down: the residual concentrates in the taper band
                                  near the shoreline, at a magnitude set by the MAL line.
    H4 reservoir logic applied downstream -- assumptions built for the pool (a ~16 m water surface, its shoreline) leaking below
                                  the dam, where the water surface is near 0 m: the residual then keys on the pool-derived layers.
Outputs: outputs/tables/p65_source1_decomposition.csv (residuals by every stratum, with the hypothesis each one bears on),
         outputs/tables/p65_source1_hypotheses.csv   (one verdict row per hypothesis, with the number that decides it),
         outputs/figures/p65_source1_root_cause.png  (map of product - FABDEM over the zone beds + the diagnostic panels)
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

spec = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P63)
P57 = P63.P57
ZONES = {"ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
BED = {n: ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_PRE_BREACH_30m.tif" for n in (2, 3, 4)}
SRC_SUFFIX = "_source"        # p53 writes a companion provenance raster
TAIL_M = -5.0


def q(s, p):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    return round(float(np.percentile(s, p)), 3) if len(s) else np.nan


def stat(g, col, label, hypothesis, **extra):
    r = g[col].values; r = r[np.isfinite(r)]
    if len(r) < 30:
        return dict(stratum=label, bears_on=hypothesis, n=len(r), **extra)
    return dict(stratum=label, bears_on=hypothesis, n=len(r), bias=round(float(r.mean()), 3), median=round(float(np.median(r)), 3),
                RMSE=round(float(np.sqrt((r ** 2).mean())), 3), LE90=round(float(np.percentile(np.abs(r), 90)), 3),
                worst=round(float(r.min()), 2), share_below_5m=round(float((r < TAIL_M).mean()), 4), **extra)


def main():
    V = P63.build_set_C()
    S1 = V[V.src == 1].copy()
    print(f"source-1 night ground segments: {len(S1):,} over {S1.zone.nunique()} zones")
    S1["res"] = S1.dem - S1.H_ice                 # product - ICESat-2
    S1["fab_res"] = S1.fab - S1.H_ice             # FABDEM - ICESat-2  (the control)
    S1["dem_fab"] = S1.dem - S1.fab               # product - FABDEM   (what p53 changed)
    S1["is_tail"] = S1.res < TAIL_M   # not "tail": that shadows DataFrame.tail
    print(f"  product-ICESat2 bias {S1.res.mean():+.3f}, FABDEM-ICESat2 bias {S1.fab_res.mean():+.3f}, "
          f"product-FABDEM bias {S1.dem_fab.mean():+.3f}; tail {S1.is_tail.mean():.1%}")

    # ---------------- support geometry: distance to a real sounding, to the water-polygon edge, to the channel
    rows = []
    # (a) distance to the nearest real sounding used by p53
    snd = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
    if snd.exists():
        sd = pd.read_parquet(snd, columns=["x", "y"])
        from scipy.spatial import cKDTree
        d_snd, _ = cKDTree(np.c_[sd.x.values, sd.y.values]).query(np.c_[S1.x.values, S1.y.values], k=1)
        S1["d_sounding_km"] = d_snd / 1e3
    else:
        S1["d_sounding_km"] = np.nan
        print("  soundings parquet not found -- H2 cannot be tested on distance to support")
    # (b) distance to the edge of the bathymetric domain, i.e. into the taper band (H3)
    S1["d_edge_m"] = np.nan
    for zn, n in ZONES.items():
        if not BED[n].exists():
            continue
        with rasterio.open(BED[n]) as ds:
            a = ds.read(1); has = np.isfinite(a) & (a != ds.nodata); cell = abs(ds.transform.a)
            dist = ndimage.distance_transform_edt(has) * cell           # distance INTO the bathymetric surface from its edge
            m = (S1.zone == zn).values
            if m.any():
                rc = rasterio.transform.rowcol(ds.transform, S1.x.values[m], S1.y.values[m])
                r_, c_ = np.asarray(rc[0]), np.asarray(rc[1])
                ok = (r_ >= 0) & (r_ < a.shape[0]) & (c_ >= 0) & (c_ < a.shape[1])
                v = np.full(m.sum(), np.nan); v[ok] = dist[r_[ok], c_[ok]]
                S1.loc[m, "d_edge_m"] = v
    # (c) chainage and distance to the channel (H1 trend, H4)
    try:
        ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
        if "chain_km" not in ch:
            ch = SW.chainage_from_dam(ch, *CFG.KAKHOVKA_DAM)
        from pyproj import Transformer
        tf = Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(S1.x.values, S1.y.values)
        S1["chain_km"], S1["d_channel_km"], _, _ = SW.assign_chainage(lon, lat, ch, max_dist_km=60)
    except Exception as e:
        S1["chain_km"] = np.nan; S1["d_channel_km"] = np.nan
        print(f"  chainage unavailable: {type(e).__name__}: {e}")

    # ---------------- decomposition
    rows.append(stat(S1, "res", "ALL source-1 points (product - ICESat-2)", "reference"))
    rows.append(stat(S1, "fab_res", "ALL source-1 points (FABDEM - ICESat-2) [CONTROL]", "reference"))
    rows.append(stat(S1, "dem_fab", "ALL source-1 points (product - FABDEM)", "reference"))
    for zn, g in S1.groupby("zone"):
        rows.append(stat(g, "res", f"zone {zn}", "H4 reservoir logic leaking downstream"))
        rows.append(stat(g, "fab_res", f"zone {zn} [FABDEM control]", "H4"))
    for col, hyp, bins, unit in (("d_sounding_km", "H2 extrapolation", [0, 0.5, 1, 2, 5, 10, 1e9], "km to nearest sounding"),
                                 ("d_edge_m", "H3 pseudo/boundary constraints", [0, 30, 60, 150, 300, 1e9], "m inside the bathymetric edge"),
                                 ("d_channel_km", "H2/H3 across-channel position", [0, 0.25, 0.5, 1, 2, 1e9], "km to the SWORD channel"),
                                 ("H_ice", "terrain dependence", [-10, 0, 2, 5, 10, 1e9], "m EVRF2019 ground"),
                                 ("chain_km", "H1 datum/level trend", [-10, 20, 40, 60, 80, 1e9], "km below the dam")):
        if not np.isfinite(S1[col]).any():
            continue
        S1["_b"] = pd.cut(S1[col], bins)
        for k, g in S1.groupby("_b", observed=True):
            rows.append(stat(g, "res", f"{unit} in {k}", hyp))
        rows.append(dict(stratum=f"--- correlation of residual with {unit}", bears_on=hyp, n=int(np.isfinite(S1[col]).sum()),
                         bias=round(float(pd.Series(S1.res).corr(pd.Series(S1[col]), method="spearman")), 3)))
    if "wc" in S1:
        for k, g in S1.groupby("wc"):
            rows.append(stat(g, "res", f"WorldCover class {int(k) if np.isfinite(k) else k}", "surface type"))
    T = pd.DataFrame(rows); T.to_csv(CFG.TABLES / "p65_source1_decomposition.csv", index=False)
    print("\n" + T.to_string(index=False))

    # ---------------- hypothesis verdicts
    def corr(col):
        return float(pd.Series(S1.res).corr(pd.Series(S1[col]), method="spearman")) if np.isfinite(S1[col]).any() else np.nan
    tail = S1[S1.is_tail]
    ver = []
    ver.append(dict(hypothesis="H1 height/depth semantics (datum, sign, double correction)",
                    key_number=f"median product-FABDEM {S1.dem_fab.median():+.2f} m, spearman(res, chainage) {corr('chain_km'):+.3f}",
                    verdict="supported if the offset is near-constant or trends with chainage"))
    ver.append(dict(hypothesis="H2 extrapolation far from real support",
                    key_number=f"spearman(res, distance to sounding) {corr('d_sounding_km'):+.3f}; tail points median distance "
                               f"{tail.d_sounding_km.median() if len(tail) else np.nan:.2f} km vs {S1.d_sounding_km.median():.2f} km overall",
                    verdict="supported if the tail sits far from support"))
    ver.append(dict(hypothesis="H3 pseudo/boundary constraints pull the surface down",
                    key_number=f"spearman(res, distance inside the edge) {corr('d_edge_m'):+.3f}; tail points median "
                               f"{tail.d_edge_m.median() if len(tail) else np.nan:.0f} m inside vs {S1.d_edge_m.median():.0f} m overall",
                    verdict="supported if the tail hugs the edge/taper band"))
    ver.append(dict(hypothesis="H4 reservoir logic applied below the dam",
                    key_number="; ".join(f"{z}: bias {g.res.mean():+.2f} / tail {g.is_tail.mean():.1%}" for z, g in S1.groupby("zone")),
                    verdict="supported if one zone carries it and FABDEM there is clean"))
    H = pd.DataFrame(ver); H.to_csv(CFG.TABLES / "p65_source1_hypotheses.csv", index=False)
    print("\n" + H.to_string(index=False))

    # ---------------- figure: the map is the point
    fig, axes = plt.subplots(2, 2, figsize=(19, 12))
    ax = axes[0, 0]
    sc = ax.scatter(S1.x, S1.y, c=np.clip(S1.res, -10, 5), s=3, cmap="RdBu_r", vmin=-10, vmax=5)
    plt.colorbar(sc, ax=ax, label="product − ICESat-2, m"); ax.set_aspect("equal")
    ax.set_title(f"(a) source-1 residual in space (n {len(S1):,}; tail {S1.is_tail.mean():.1%} below −5 m)")
    ax = axes[0, 1]
    ax.scatter(S1.d_sounding_km, S1.res, s=3, alpha=0.3, color="#2166ac", label="product")
    ax.scatter(S1.d_sounding_km, S1.fab_res, s=3, alpha=0.3, color="#b2182b", label="FABDEM (control)")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(TAIL_M, color="grey", ls=":", lw=1)
    ax.set_xlabel("distance to the nearest real sounding, km"); ax.set_ylabel("residual, m"); ax.set_ylim(-25, 10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("(b) H2: does the error grow away from support?")
    ax = axes[1, 0]
    ax.scatter(S1.d_edge_m, S1.res, s=3, alpha=0.3, color="#2166ac")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(TAIL_M, color="grey", ls=":", lw=1)
    ax.set_xlabel("distance inside the bathymetric edge, m"); ax.set_ylabel("product − ICESat-2, m"); ax.set_ylim(-25, 10)
    ax.set_xlim(0, 1500); ax.grid(alpha=0.3); ax.set_title("(c) H3: is the error in the taper band at the edge?")
    ax = axes[1, 1]
    for lab, col, c in (("product − ICESat-2", "res", "#2166ac"), ("FABDEM − ICESat-2", "fab_res", "#b2182b"), ("product − FABDEM", "dem_fab", "#238b45")):
        v = S1[col].values; v = v[np.isfinite(v)]
        ax.hist(np.clip(v, -20, 10), bins=120, histtype="step", lw=1.4, density=True, label=f"{lab} (median {np.median(v):+.2f})")
    ax.axvline(0, color="k", lw=0.8); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel("m"); ax.set_title("(d) three differences: the control separates a bad surface from a bad reference")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p65_source1_root_cause.png", dpi=110)
    print("-> outputs/figures/p65_source1_root_cause.png")


if __name__ == "__main__":
    main()
