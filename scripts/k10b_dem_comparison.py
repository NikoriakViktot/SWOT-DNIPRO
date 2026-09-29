#!/usr/bin/env python
"""K10b -- D0 (historical M2) vs D1 (ICESat-2-only kriging) on the SAME frozen
independent ICESat-2 validation RGTs.

Primary accuracy statement comes from the grouped leave-one-RGT-out design, NOT
from any gridded error map. The 2/5/10 km block maps are spatial diagnostics:
  5 km  = PRIMARY local error map
  2 km  = fine-scale diagnostic
 10 km  = coarse robustness / regional sensitivity

Residual sign convention, fixed once:

    residual = H_DEM - H_ICESat2      (positive = DEM too high)

The kriging numerics are imported from k9_predictive_validation rather than
re-implemented, so the solver guards and the skgstat variogram fitter that were
debugged in K9 are reused exactly.

Outputs
-------
outputs/tables/k10b_dem_comparison_global.csv
outputs/tables/k10b_validation_by_rgt.csv
outputs/tables/k10b_accuracy_by_distance.csv
outputs/tables/k10b_vertical_sensitivity.csv
outputs/tables/k10b_night_vs_day.csv
outputs/tables/k10b_block_error_maps.csv
outputs/tables/k10b_kriging_diagnostics.csv
outputs/figures/K10_validation_by_track.png
outputs/figures/K10_accuracy_by_distance.png
outputs/figures/K10_validation_error_maps.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
# reuse the K9 numerics verbatim (guards + skgstat fitter, already debugged there)
from k9_predictive_validation import fit_variogram, nmad, ok_predict, _FALLBACK

BATHY = CFG.ROOT / "data" / "processed" / "bathymetry"
CUR = CFG.ROOT / "data" / "processed" / "current_bed"
RNG = np.random.default_rng(CFG.SEED)
DIST_BINS = [0, 500, 1000, 2000, 5000, np.inf]
DIST_LABELS = ["0-0.5 km", "0.5-1 km", "1-2 km", "2-5 km", ">5 km"]
BLOCK_SCALES = {2000: "fine diagnostic", 5000: "PRIMARY", 10000: "coarse robustness"}
N_BOOT = 2000
PREDICTORS = ["P_former_channel", "distance_to_channel_m"]


def metrics(res, label, n_rgt=np.nan):
    res = np.asarray(res, float); res = res[np.isfinite(res)]
    if not len(res):
        return {"set": label, "N": 0}
    return {"set": label, "N": len(res), "N_RGT": n_rgt,
            "RMSE": float(np.sqrt((res ** 2).mean())),
            "MAE": float(np.abs(res).mean()),
            "mean_bias": float(res.mean()), "median_bias": float(np.median(res)),
            "NMAD": nmad(res), "std": float(res.std(ddof=1)) if len(res) > 1 else np.nan,
            "P05": float(np.percentile(res, 5)), "P50": float(np.percentile(res, 50)),
            "P95": float(np.percentile(res, 95)),
            "P95_abs_error": float(np.percentile(np.abs(res), 95))}


def main() -> None:
    print("=" * 70)
    print("K10b -- D0 (historical M2) vs D1 (ICESat-2 kriging), frozen holdout")
    print("=" * 70)
    print("residual = H_DEM - H_ICESat2   (positive = DEM too high)\n")

    g = pd.read_parquet(CUR / "k10_modern_elevation_points.parquet")
    prim = g[g.exposed_ground_validation_ok]
    build = prim[prim.split == "MODEL_BUILD"]
    valid = prim[prim.split == "INDEPENDENT_VALIDATION"]
    print(f"primary exposed-ground set (night + DRY + >=50 m edge): {len(prim):,}")
    print(f"  MODEL_BUILD {len(build):,} ({build.rgt.nunique()} RGTs) | "
          f"INDEPENDENT_VALIDATION {len(valid):,} ({valid.rgt.nunique()} RGTs)")

    zv = valid.H_terrain_EVRF2019_empirical_m.values
    xy_v = np.c_[valid.x.values, valid.y.values]

    # ---------------- D0: historical M2, trained on ALL legacy soundings ----
    print("\n" + "-" * 70)
    print("D0 = HISTORICAL_M2 (regression on frozen prior bands + residual kriging)")
    print("-" * 70)
    sd = pd.read_parquet(BATHY / "kakhovka_soundings_evrf2019.parquet").copy()
    with rasterio.open(BATHY / "morphology_prior_v1.tif") as src:
        P = src.read(1); P = np.where(P == src.nodata, np.nan, P)
        D = src.read(2); tr = src.transform; gH, gW = src.shape
    inv = ~tr
    c_, r_ = inv * (sd.x.values, sd.y.values)
    sd["P_former_channel"] = P[np.clip(r_.astype(int), 0, gH - 1), np.clip(c_.astype(int), 0, gW - 1)]
    sd["distance_to_channel_m"] = D[np.clip(r_.astype(int), 0, gH - 1), np.clip(c_.astype(int), 0, gW - 1)]
    sd = sd[sd.P_former_channel.notna()].copy()
    sd = (sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0))
          .groupby(["_kx", "_ky"], as_index=False)
          .agg(x=("x", "mean"), y=("y", "mean"), H=("H_bed_evrf2019_m", "mean"),
               P_former_channel=("P_former_channel", "mean"),
               distance_to_channel_m=("distance_to_channel_m", "mean")))
    xy_s = np.c_[sd.x.values, sd.y.values]
    X = np.c_[np.ones(len(sd)), sd[PREDICTORS].values]
    beta, *_ = np.linalg.lstsq(X, sd.H.values, rcond=None)
    res_tr = sd.H.values - X @ beta
    vg0 = fit_variogram(xy_s, res_tr)
    print(f"  soundings used: {len(sd):,}; residual variogram range {vg0[0]/1000:.2f} km, "
          f"sill {vg0[1]:.2f}, nugget {vg0[2]:.2f}")

    cv, rv = inv * (valid.x.values, valid.y.values)
    Pv = P[np.clip(rv.astype(int), 0, gH - 1), np.clip(cv.astype(int), 0, gW - 1)]
    Dv = D[np.clip(rv.astype(int), 0, gH - 1), np.clip(cv.astype(int), 0, gW - 1)]
    Xv = np.c_[np.ones(len(valid)), Pv, Dv]
    fb0 = dict(_FALLBACK)
    rk0, var0 = ok_predict(xy_s, res_tr, xy_v, vg0)
    d0 = Xv @ beta + rk0
    print(f"  D0 predictions: {np.nanmin(d0):.2f} .. {np.nanmax(d0):.2f} m "
          f"(ICESat-2 reference {zv.min():.2f} .. {zv.max():.2f})")

    # ---------------- D1: ICESat-2-only kriging, BUILD RGTs only -----------
    print("\n" + "-" * 70)
    print("D1 = ICESAT2_OK (ordinary kriging, BUILD RGTs only)")
    print("-" * 70)
    xy_b = np.c_[build.x.values, build.y.values]
    zb = build.H_terrain_EVRF2019_empirical_m.values
    vg1 = fit_variogram(xy_b, zb)
    print(f"  training segments: {len(build):,}; variogram range {vg1[0]/1000:.2f} km, "
          f"sill {vg1[1]:.2f}, nugget {vg1[2]:.2f}")
    fb1 = dict(_FALLBACK)
    d1, var1 = ok_predict(xy_b, zb, xy_v, vg1)
    print(f"  D1 predictions: {np.nanmin(d1):.2f} .. {np.nanmax(d1):.2f} m")

    diag = pd.DataFrame([
        {"model": "D0_HISTORICAL_M2", "vg_range_m": vg0[0], "vg_sill": vg0[1], "vg_nugget": vg0[2],
         "n_train": len(sd), "weight_guard_fallbacks": _FALLBACK["n_bad"] - fb0["n_bad"],
         "envelope_fallbacks": _FALLBACK["n_env"] - fb0["n_env"],
         "pred_min": float(np.nanmin(d0)), "pred_max": float(np.nanmax(d0))},
        {"model": "D1_ICESAT2_OK", "vg_range_m": vg1[0], "vg_sill": vg1[1], "vg_nugget": vg1[2],
         "n_train": len(build), "weight_guard_fallbacks": _FALLBACK["n_bad"] - fb1["n_bad"],
         "envelope_fallbacks": _FALLBACK["n_env"] - fb1["n_env"],
         "pred_min": float(np.nanmin(d1)), "pred_max": float(np.nanmax(d1))}])
    diag.to_csv(CFG.TABLES / "k10b_kriging_diagnostics.csv", index=False)
    print("\nkriging integrity (no silent clipping anywhere):")
    print(diag.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- support geometry -------------------------------------
    tree_b = cKDTree(xy_b)
    dist_nn, _ = tree_b.query(xy_v, k=1)
    v = valid.copy()
    v["d0"] = d0; v["d1"] = d1
    v["res_d0"] = d0 - zv
    v["res_d1"] = d1 - zv
    v["dist_to_IS2_training_m"] = dist_nn
    v["dist_class"] = pd.cut(dist_nn, DIST_BINS, labels=DIST_LABELS, right=False)
    for rad in (1000, 2000, 5000):
        v[f"n_IS2_training_within_{rad//1000}km"] = [len(i) for i in tree_b.query_ball_point(xy_v, rad)]
    print(f"\ndistance from validation segments to nearest BUILD segment: "
          f"median {np.median(dist_nn):.0f} m, p90 {np.percentile(dist_nn,90):.0f} m, "
          f"max {dist_nn.max():.0f} m")

    # ---------------- global metrics ---------------------------------------
    rows = [metrics(v.res_d0, "D0_HISTORICAL_M2", v.rgt.nunique()),
            metrics(v.res_d1, "D1_ICESAT2_OK", v.rgt.nunique())]
    glob = pd.DataFrame(rows)
    glob.to_csv(CFG.TABLES / "k10b_dem_comparison_global.csv", index=False)
    print("\n" + "=" * 70)
    print("GLOBAL INDEPENDENT VALIDATION (frozen holdout RGTs)")
    print("=" * 70)
    print(glob.to_string(index=False, float_format=lambda v_: f"{v_:.3f}"))

    # paired delta, bootstrapped by RGT (never by 100 m segment)
    rgts = v.rgt.unique()
    dr = np.sqrt((v.res_d1 ** 2).mean()) - np.sqrt((v.res_d0 ** 2).mean())
    boots = []
    for _ in range(N_BOOT):
        pick = RNG.choice(rgts, len(rgts), replace=True)
        sel = np.concatenate([np.where(v.rgt.values == p)[0] for p in pick])
        boots.append(np.sqrt((v.res_d1.values[sel] ** 2).mean())
                     - np.sqrt((v.res_d0.values[sel] ** 2).mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\ndelta_RMSE (D1 - D0) = {dr:+.3f} m, 95% CI by RGT [{lo:+.3f}, {hi:+.3f}] "
          f"-> {'D1 better' if hi < 0 else 'D0 better' if lo > 0 else 'not separable'}")

    # ---------------- by RGT ------------------------------------------------
    brows = []
    for rgt, gg in v.groupby("rgt"):
        for m, col in (("D0_HISTORICAL_M2", "res_d0"), ("D1_ICESAT2_OK", "res_d1")):
            r = metrics(gg[col], m)
            r.update({"rgt": int(rgt), "date": str(pd.to_datetime(gg.date).min().date()),
                      "n_segments": len(gg)})
            brows.append(r)
    byrgt = pd.DataFrame(brows)
    byrgt.to_csv(CFG.TABLES / "k10b_validation_by_rgt.csv", index=False)
    print("\nBY INDEPENDENT RGT (prevents one dense track dominating):")
    print(byrgt.pivot_table(index="rgt", columns="set", values=["RMSE", "mean_bias", "NMAD"])
          .to_string(float_format=lambda v_: f"{v_:.3f}"))

    # ---------------- by distance to IS2 training --------------------------
    drows = []
    for dc, gg in v.groupby("dist_class", observed=True):
        for m, col in (("D0_HISTORICAL_M2", "res_d0"), ("D1_ICESAT2_OK", "res_d1")):
            r = metrics(gg[col], m, gg.rgt.nunique())
            r.update({"dist_class": str(dc), "n_segments": len(gg)})
            drows.append(r)
    bydist = pd.DataFrame(drows)
    bydist.to_csv(CFG.TABLES / "k10b_accuracy_by_distance.csv", index=False)
    print("\nACCURACY BY DISTANCE TO NEAREST ICESat-2 TRAINING SEGMENT:")
    piv = bydist.pivot_table(index="dist_class", columns="set", values="RMSE").reindex(DIST_LABELS)
    piv["n"] = bydist.groupby("dist_class").n_segments.first().reindex(DIST_LABELS)
    piv["N_RGT"] = bydist.groupby("dist_class").N_RGT.first().reindex(DIST_LABELS)
    print(piv.to_string(float_format=lambda v_: f"{v_:.3f}"))

    cross = None
    for lab in DIST_LABELS:
        if lab in piv.index and np.isfinite(piv.loc[lab]).all():
            if piv.loc[lab, "D1_ICESAT2_OK"] > piv.loc[lab, "D0_HISTORICAL_M2"] and cross is None:
                cross = lab
    print(f"\ncrossover (first bin where D1 becomes worse than D0): "
          f"{cross if cross else 'none within the sampled distance range'}")

    # ---------------- vertical-reference sensitivity -----------------------
    c_sd = float(g.sigma_c_m.iloc[0])
    vrows = []
    for dc_ in (-c_sd, 0.0, +c_sd):
        # perturbing c shifts the ICESat-2 reference AND D1 (which is built from it)
        # together, so D1's bias is insensitive by construction; D0 is not.
        zshift = dc_
        vrows.append({"delta_c_m": dc_,
                      "D0_bias": float((v.res_d0 - zshift).mean()),
                      "D0_RMSE": float(np.sqrt(((v.res_d0 - zshift) ** 2).mean())),
                      "D0_NMAD": nmad(v.res_d0 - zshift),
                      "D1_bias": float(v.res_d1.mean()), "D1_RMSE": float(np.sqrt((v.res_d1 ** 2).mean())),
                      "D1_NMAD": nmad(v.res_d1)})
    vs = pd.DataFrame(vrows)
    vs.to_csv(CFG.TABLES / "k10b_vertical_sensitivity.csv", index=False)
    print(f"\nVERTICAL-BRIDGE SENSITIVITY (c = {g.c_empirical_m.iloc[0]:+.3f} +/- {c_sd:.3f} m):")
    print(vs.to_string(index=False, float_format=lambda v_: f"{v_:.3f}"))
    print("  STRUCTURAL NOTE: perturbing c moves the ICESat-2 reference and D1 together, so")
    print("  D1's bias is insensitive to the vertical bridge BY CONSTRUCTION. Only D0's bias")
    print("  carries the bridge uncertainty. A D1-vs-D0 bias comparison is therefore not a")
    print("  clean test of either -- this is stated rather than hidden.")

    # ---------------- night vs day sensitivity -----------------------------
    day = g[(~g.night) & (g.water_state == "DRY") & (g.distance_to_water_edge_m >= 50.0)
            & (g.split == "INDEPENDENT_VALIDATION")]
    nrows = [metrics(v.res_d0, "NIGHT D0", v.rgt.nunique()),
             metrics(v.res_d1, "NIGHT D1", v.rgt.nunique())]
    if len(day) > 50:
        xy_d = np.c_[day.x.values, day.y.values]
        zd = day.H_terrain_EVRF2019_empirical_m.values
        cdd, rdd = inv * (day.x.values, day.y.values)
        Xd = np.c_[np.ones(len(day)), P[np.clip(rdd.astype(int), 0, gH - 1), np.clip(cdd.astype(int), 0, gW - 1)],
                   D[np.clip(rdd.astype(int), 0, gH - 1), np.clip(cdd.astype(int), 0, gW - 1)]]
        rk0d, _ = ok_predict(xy_s, res_tr, xy_d, vg0)
        d1d, _ = ok_predict(xy_b, zb, xy_d, vg1)
        nrows.append(metrics(Xd @ beta + rk0d - zd, "DAY D0", day.rgt.nunique()))
        nrows.append(metrics(d1d - zd, "DAY D1", day.rgt.nunique()))
    nd = pd.DataFrame(nrows)
    nd.to_csv(CFG.TABLES / "k10b_night_vs_day.csv", index=False)
    print(f"\nNIGHT (canonical) vs DAY sensitivity  [day validation segments: {len(day):,}]:")
    print(nd.to_string(index=False, float_format=lambda v_: f"{v_:.3f}"))

    # ---------------- block error maps -------------------------------------
    brows = []
    for B, role in BLOCK_SCALES.items():
        bx = (v.x // B).astype(int); by = (v.y // B).astype(int)
        for (ix, iy), gg in v.groupby([bx, by]):
            n, nr = len(gg), gg.rgt.nunique()
            support = (4 if n >= 50 and nr >= 3 else 3 if n >= 20 and nr >= 2
                       else 2 if n >= 20 else 1)
            brows.append({"block_m": B, "role": role, "bx": ix, "by": iy,
                          "x_center": (ix + 0.5) * B, "y_center": (iy + 0.5) * B,
                          "n_segments": n, "n_unique_RGT": nr,
                          "validation_support_class": support,
                          "robust": bool(n >= 20 and nr >= 2),
                          "exploratory_single_track": bool(n >= 20 and nr < 2),
                          "D0_RMSE": float(np.sqrt((gg.res_d0 ** 2).mean())),
                          "D0_NMAD": nmad(gg.res_d0), "D0_bias": float(gg.res_d0.mean()),
                          "D0_median_bias": float(gg.res_d0.median()),
                          "D1_RMSE": float(np.sqrt((gg.res_d1 ** 2).mean())),
                          "D1_NMAD": nmad(gg.res_d1), "D1_bias": float(gg.res_d1.mean()),
                          "D1_median_bias": float(gg.res_d1.median()),
                          "median_date": str(pd.to_datetime(gg.date).median().date()),
                          "median_dist_to_training_m": float(gg.dist_to_IS2_training_m.median())})
    bm = pd.DataFrame(brows)
    bm.to_csv(CFG.TABLES / "k10b_block_error_maps.csv", index=False)
    print("\nBLOCK ERROR MAPS (no interpolation into unsupported blocks):")
    for B, role in BLOCK_SCALES.items():
        s_ = bm[bm.block_m == B]
        print(f"  {B//1000:2d} km ({role:20s}): {len(s_):4d} blocks with data | "
              f"robust (n>=20 & >=2 RGT) {int(s_.robust.sum()):3d} | "
              f"EXPLORATORY_SINGLE_TRACK {int(s_.exploratory_single_track.sum()):3d} | "
              f"sparse {int((s_.validation_support_class==1).sum()):3d}")

    # ---------------- figures ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    w = 0.38
    idx = np.arange(byrgt.rgt.nunique())
    p0 = byrgt[byrgt.set == "D0_HISTORICAL_M2"].sort_values("rgt")
    p1 = byrgt[byrgt.set == "D1_ICESAT2_OK"].sort_values("rgt")
    axes[0].bar(idx - w/2, p0.RMSE, w, label="D0 historical M2", color="#8a94a3")
    axes[0].bar(idx + w/2, p1.RMSE, w, label="D1 ICESat-2 OK", color="#236f8c")
    axes[0].set_xticks(idx); axes[0].set_xticklabels(p0.rgt.astype(int), fontsize=8)
    axes[0].set_xlabel("independent RGT"); axes[0].set_ylabel("RMSE, m")
    axes[0].set_title("A. accuracy by held-out track", loc="left", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].bar(idx - w/2, p0.mean_bias, w, color="#8a94a3")
    axes[1].bar(idx + w/2, p1.mean_bias, w, color="#236f8c")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xticks(idx); axes[1].set_xticklabels(p0.rgt.astype(int), fontsize=8)
    axes[1].set_xlabel("independent RGT"); axes[1].set_ylabel("mean bias, m")
    axes[1].set_title("B. bias by held-out track (positive = DEM too high)", loc="left", fontsize=10)
    fig.suptitle("K10 -- independent ICESat-2 validation by track", fontsize=11)
    fig.tight_layout(); fig.savefig(CFG.FIG / "K10_validation_by_track.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(DIST_LABELS))
    ax.plot(xs, piv.D0_HISTORICAL_M2.values, "o-", color="#8a94a3", label="D0 historical M2")
    ax.plot(xs, piv.D1_ICESAT2_OK.values, "s-", color="#236f8c", label="D1 ICESat-2 OK")
    for i, lab in enumerate(DIST_LABELS):
        if lab in piv.index and np.isfinite(piv.loc[lab, "n"]):
            ax.annotate(f"n={int(piv.loc[lab,'n'])}", (i, 0), xytext=(0, -28),
                        textcoords="offset points", ha="center", fontsize=7, color="#1a2228")
    ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS)
    ax.set_xlabel("distance to nearest ICESat-2 training segment")
    ax.set_ylabel("RMSE, m")
    ax.set_title("K10 -- radius of influence of ICESat-2 kriging", loc="left", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(CFG.FIG / "K10_accuracy_by_distance.png", dpi=150)

    pri = bm[(bm.block_m == 5000)]
    rb = pri[pri.robust]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, col, ttl in zip(axes, ["D1_RMSE", "D1_NMAD", "D1_bias"],
                            ["RMSE", "NMAD", "bias (positive = DEM too high)"]):
        cmap = "viridis" if col != "D1_bias" else "coolwarm"
        vlim = (-np.nanmax(np.abs(rb.D1_bias)), np.nanmax(np.abs(rb.D1_bias))) if col == "D1_bias" else (None, None)
        sc = ax.scatter(rb.x_center, rb.y_center, c=rb[col], s=260, marker="s",
                        cmap=cmap, vmin=vlim[0], vmax=vlim[1], edgecolor="k", lw=0.4)
        ax.scatter(pri[~pri.robust].x_center, pri[~pri.robust].y_center, s=260, marker="s",
                   facecolor="none", edgecolor="#c9c9c9", lw=0.6)
        ax.scatter(v.x, v.y, s=0.5, c="#c1402a", alpha=0.25, zorder=0)
        fig.colorbar(sc, ax=ax, label=f"{ttl}, m")
        ax.set_aspect("equal"); ax.set_title(f"{ttl} (5 km blocks, robust only)", fontsize=9)
        ax.set_xlabel("Easting, m")
    axes[0].set_ylabel("Northing, m")
    fig.suptitle("K10 -- local validation error, D1 ICESat-2 kriging. Empty squares = support "
                 "too weak (NoData), red = independent ATL08 tracks. No interpolation.", fontsize=10)
    fig.tight_layout(); fig.savefig(CFG.FIG / "K10_validation_error_maps.png", dpi=150)
    print(f"\n-> figures written to {CFG.FIG}")


if __name__ == "__main__":
    main()
