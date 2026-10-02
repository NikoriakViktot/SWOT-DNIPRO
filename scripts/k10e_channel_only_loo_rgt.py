#!/usr/bin/env python
"""K10e -- CHANNEL_ONLY_LEAVE_ONE_RGT_OUT on the vegetation-QC'd 20 m PRIMARY set.

Repeats k10d_channel_only_loo_rgt.py's design exactly (same D0, same fold
structure: remove every accepted observation of one RGT, refit D1 on the
remainder, predict the held-out channel observations, compare against D0 at
the identical locations) but on a materially different, and cleaner, input:

    k10d (100 m):  13,905 accepted (night+DRY+50m), NO vegetation QC
                    320 channel segments, 7 RGTs
    k10e (20 m):   PRIMARY_GROUND_VALIDATION_SET only -- night+DRY+50m AND
                    low vegetation/woody-canopy/mixed-pixel risk (spectral
                    indices + PhoREAL ATL03 structure + Dynamic World
                    trees/shrub probability) AND DRY_BARE_SEDIMENT or
                    SPARSE_HERBACEOUS surface class

The first pass, at the LOW_RISK=0.20 threshold used everywhere else in K10e,
gave only 41 channel segments and just 2 usable RGT folds (>=5 points): RGT
1112 (26 pts, D0 better) and RGT 266 (6 pts, D1 better) -- 1/2, pooled D0
better. Too few folds to conclude anything. This run uses a relaxed 0.30
threshold on all three risk scores (k10e_split_relaxed_0_30, built alongside
the primary 0.20 split, not replacing it) to check whether the direction is
sensitive to exactly where that line is drawn: 68 channel segments, 4 usable
folds (RGT 1112: 34, 1211: 14, 266: 7, 769: 7).

The channel count under vegetation QC is smaller than the unfiltered 20 m
count (1,613) mostly because half of the water-QC-accepted channel population
(816/1613) is classified OPEN_WATER by Sentinel-2 despite passing the exact
vector water QC as DRY -- a real, separately worth investigating discrepancy,
not something this run resolves.

Outputs
-------
outputs/tables/channel_only_LOO_RGT_veg_qc_relaxed030.csv
outputs/figures/channel_LOO_RGT_veg_qc_relaxed030_results.png
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
from k9_predictive_validation import fit_variogram, nmad, ok_predict

BATHY = CFG.ROOT / "data" / "processed" / "bathymetry"
CUR = CFG.ROOT / "data" / "processed" / "current_bed"
RNG = np.random.default_rng(CFG.SEED)
PREDICTORS = ["P_former_channel", "distance_to_channel_m"]
CHANNEL_P = 0.5
MIN_CHANNEL_POINTS = 5      # identical threshold to k10d_channel_only_loo_rgt.py
N_BOOT = 2000


def rmse(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.sqrt((v ** 2).mean())) if len(v) else np.nan


def main() -> None:
    print("=" * 78)
    print("K10e -- CHANNEL_ONLY_LEAVE_ONE_RGT_OUT, vegetation-QC'd 20 m PRIMARY set")
    print("=" * 78)
    print("Does the channel-stratum D1 advantage survive full-RGT holdout on a ground")
    print("reference cleaned of vegetation/canopy contamination?")
    print("residual = H_DEM - H_ICESat2\n")

    SPLIT_COL = "k10e_split_relaxed_0_30"   # sensitivity: relaxed risk threshold
                                            # (0.20 gave only 2 usable channel folds;
                                            # see k10e_merge_and_rescore_2023.py)
    g = pd.read_parquet(CUR / "k10e_post_breach_full_context.parquet")
    prim = g[g[SPLIT_COL] == "PRIMARY_GROUND_VALIDATION_SET"].copy().reset_index(drop=True)
    xy = np.c_[prim.x.values, prim.y.values]
    H = prim.H_terrain_EVRF2019_empirical_m.values
    is_chan = (prim.P_channel >= CHANNEL_P).values

    print(f"PRIMARY_GROUND_VALIDATION_SET: {len(prim):,} segments, {prim.rgt.nunique()} RGTs")
    print(f"channel stratum (P_channel >= {CHANNEL_P}): {int(is_chan.sum()):,} segments")
    per_rgt_chan = pd.Series(is_chan).groupby(prim.rgt.values).sum()
    per_rgt_chan = per_rgt_chan[per_rgt_chan > 0].sort_values(ascending=False)
    print("channel segments per RGT (ALL, before the >=5 usability threshold):")
    for r, n in per_rgt_chan.items():
        flag = "" if n >= MIN_CHANNEL_POINTS else "  <- below MIN_CHANNEL_POINTS, excluded"
        print(f"    RGT {int(r):5d}: {int(n):3d}{flag}")

    # ------------------------------------------------ D0, fitted once ------
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
    cv, rv = inv * (prim.x.values, prim.y.values)
    Xv = np.c_[np.ones(len(prim)),
               P[np.clip(rv.astype(int), 0, gH - 1), np.clip(cv.astype(int), 0, gW - 1)],
               D[np.clip(rv.astype(int), 0, gH - 1), np.clip(cv.astype(int), 0, gW - 1)]]
    rk0, _ = ok_predict(xy_s, res_tr, xy, vg0)
    d0_all = Xv @ beta + rk0

    # ------------------------------------------------ folds -----------------
    sup = per_rgt_chan[per_rgt_chan >= MIN_CHANNEL_POINTS]
    fold_rgts = sorted(int(r) for r in sup.index)
    print(f"\nusable channel-supporting RGTs (>= {MIN_CHANNEL_POINTS} channel segments): "
          f"{len(fold_rgts)} -> {fold_rgts}")
    if len(fold_rgts) < 3:
        print(f"WARNING: only {len(fold_rgts)} usable RGT fold(s). A leave-one-RGT-out test")
        print("with this few independent folds cannot establish reproducibility across")
        print("tracks -- it is reported below, but the D15 decision downgrades accordingly.")

    rows, pooled = [], []
    for r in fold_rgts:
        held = (prim.rgt.values == r)
        train = ~held
        xy_tr, z_tr = xy[train], H[train]
        vg1 = fit_variogram(xy_tr, z_tr)
        d1 = ok_predict(xy_tr, z_tr, xy[held], vg1, want_var=False)
        dist_tr, _ = cKDTree(xy_tr).query(xy[held], k=1)

        z_h = H[held]
        d0_h = d0_all[held]
        chan_h = is_chan[held]
        ok = np.isfinite(d0_h) & np.isfinite(d1)

        for stratum, m in (("channel", chan_h & ok), ("non_channel", (~chan_h) & ok)):
            if m.sum() < 3:
                continue
            e0, e1 = d0_h[m] - z_h[m], d1[m] - z_h[m]
            rows.append({
                "rgt": r, "stratum": stratum, "n_validation_points": int(m.sum()),
                "RMSE_D0": rmse(e0), "RMSE_D1_RAW": rmse(e1),
                "delta_RMSE_RAW": rmse(e1) - rmse(e0),
                "MAE_D0": float(np.abs(e0).mean()), "MAE_D1_RAW": float(np.abs(e1).mean()),
                "NMAD_D0": nmad(e0), "NMAD_D1_RAW": nmad(e1),
                "bias_D0": float(e0.mean()), "bias_D1_RAW": float(e1.mean()),
                "median_distance_to_training_m": float(np.median(dist_tr[m])),
                "min_distance_to_training_m": float(np.min(dist_tr[m])),
                "D1_better": bool(rmse(e1) < rmse(e0)),
                "vg_range_m": vg1[0], "n_train_segments": int(train.sum())})
            if stratum == "channel":
                pooled.append(pd.DataFrame({"rgt": r, "e0": e0, "e1": e1}))
        cr = [x for x in rows if x["rgt"] == r and x["stratum"] == "channel"]
        if cr:
            k = cr[0]
            print(f"  RGT {r:4d}: channel n={k['n_validation_points']:4d}  "
                  f"D0 {k['RMSE_D0']:6.3f}  D1 {k['RMSE_D1_RAW']:6.3f}  "
                  f"delta {k['delta_RMSE_RAW']:+6.3f}  "
                  f"nearest training {k['median_distance_to_training_m']:7.0f} m  "
                  f"-> {'D1' if k['D1_better'] else 'D0'}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(CFG.TABLES / "channel_only_LOO_RGT_veg_qc_relaxed030.csv", index=False)

    if not pooled:
        print("\nNo usable folds -- cannot compute a pooled result.")
        return

    # ------------------------------------------------ pooled verdict --------
    pool = pd.concat(pooled, ignore_index=True)
    ch = out[out.stratum == "channel"]
    nc = out[out.stratum == "non_channel"]
    n_fav = int(ch.D1_better.sum())
    r0, r1 = rmse(pool.e0), rmse(pool.e1)

    rgts = pool.rgt.unique()
    idx = {q: np.where(pool.rgt.values == q)[0] for q in rgts}
    if len(rgts) >= 2:
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = RNG.choice(rgts, len(rgts), replace=True)
            sel = np.concatenate([idx[p] for p in pick])
            boots[b] = rmse(pool.e1.values[sel]) - rmse(pool.e0.values[sel])
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        lo = hi = np.nan

    print("\n" + "=" * 78)
    print("POOLED CHANNEL RESULT (vegetation-QC'd PRIMARY set, every fold's held-out "
          "RGT absent from training)")
    print("=" * 78)
    print(f"channel validation points {len(pool):,} over {len(rgts)} usable RGT fold(s)")
    print(f"  RMSE D0 {r0:.3f} m | RMSE D1_RAW {r1:.3f} m | delta {r1-r0:+.3f} m, "
          f"95% CI by RGT [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo)
          else f"  RMSE D0 {r0:.3f} m | RMSE D1_RAW {r1:.3f} m | delta {r1-r0:+.3f} m, "
               f"CI not defined (< 2 RGT folds)")
    print(f"  RGTs where D1 beats D0 in the channel stratum: {n_fav}/{len(ch)}")
    if len(nc):
        print(f"  internal control, non-channel stratum of the SAME folds: "
              f"median fold RMSE D0 {nc.RMSE_D0.median():.3f} m vs D1 "
              f"{nc.RMSE_D1_RAW.median():.3f} m | RGTs favouring D1: "
              f"{int(nc.D1_better.sum())}/{len(nc)}")

    print("\n" + "=" * 78)
    print("D15 -- CHANNEL EVIDENCE DECISION (vegetation-QC'd)")
    print("=" * 78)
    if len(rgts) < 3:
        verdict = "INSUFFICIENT_FOLDS"
        decision = ("N/A", f"only {len(rgts)} usable RGT fold(s) after vegetation QC + the "
                           f">= {MIN_CHANNEL_POINTS}-point usability threshold -- cannot "
                           f"establish or refute cross-RGT reproducibility from this many folds")
    else:
        separable = np.isfinite(hi) and hi < 0
        majority = n_fav >= max(3, int(np.ceil(0.6 * len(ch))))
        if separable and majority:
            verdict, decision = "PASS", ("A", "D1 beats D0 on most channel RGTs under "
                                              "full-RGT holdout")
        elif n_fav >= 1:
            verdict, decision = "PARTIAL", ("A/B", "direction reproduces on some independent "
                                                   "RGTs but not decisively")
        else:
            verdict, decision = "FAIL", ("B", "no channel RGT reproduces the advantage")
    print(f"outcome {decision[0]}: {decision[1]}")
    print(f"ICESat-2 channel added value (vegetation-QC'd): {verdict}")
    print("\nNO HYBRID DEM IS BUILT. No weighting function is defined.")

    # ------------------------------------------------ figure ----------------
    if len(ch):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ch_s = ch.sort_values("rgt")
        idxs = np.arange(len(ch_s))
        w = 0.38
        axes[0].bar(idxs - w / 2, ch_s.RMSE_D0, w, label="D0 historical M2", color="#8a94a3")
        axes[0].bar(idxs + w / 2, ch_s.RMSE_D1_RAW, w, label="D1 ICESat-2 OK", color="#236f8c")
        axes[0].set_xticks(idxs); axes[0].set_xticklabels(ch_s.rgt.astype(int), fontsize=8)
        axes[0].set_xlabel("held-out RGT (entire track removed)")
        axes[0].set_ylabel("RMSE, m")
        axes[0].set_title("A. channel stratum, vegetation-QC'd PRIMARY", loc="left", fontsize=10)
        axes[0].legend(fontsize=8)

        axes[1].axhline(0, color="grey", lw=0.8)
        cols = ["#236f8c" if b else "#c1402a" for b in ch_s.D1_better]
        axes[1].bar(idxs, ch_s.delta_RMSE_RAW, 0.6, color=cols)
        for i, (_, rr) in enumerate(ch_s.iterrows()):
            axes[1].annotate(f"n={int(rr.n_validation_points)}", (i, 0), xytext=(0, -26),
                             textcoords="offset points", ha="center", fontsize=7)
        axes[1].set_xticks(idxs); axes[1].set_xticklabels(ch_s.rgt.astype(int), fontsize=8)
        axes[1].set_ylabel("delta RMSE (D1 - D0), m")
        axes[1].set_title("B. blue = D1 better, red = D0 better", loc="left", fontsize=10)
        fig.suptitle("K10e -- channel LOO-RGT on the vegetation-QC'd 20 m PRIMARY set",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(CFG.FIG / "channel_LOO_RGT_veg_qc_relaxed030_results.png", dpi=150)
        print(f"\n-> {CFG.FIG / 'channel_LOO_RGT_veg_qc_relaxed030_results.png'}")
    print(f"-> {CFG.TABLES / 'channel_only_LOO_RGT_veg_qc_relaxed030.csv'}")


if __name__ == "__main__":
    main()
