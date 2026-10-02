#!/usr/bin/env python
"""K10d.14/15 -- CHANNEL_ONLY_LEAVE_ONE_RGT_OUT.

K10c concluded (outcome B) that D1's local added value is CONFINED to the
channel stratum: on P_channel >= 0.5 D1 beat D0 by -4.60 / -3.46 / -1.58 m in
the 0-0.5 / 0.5-1 / 1-2 km bins, while on P_channel < 0.5 it never separated
from D0 at any distance. That evidence rests on ~5-6 independent RGTs, so
before any morphology-dependent hybridisation is authorised it must be shown
that the channel advantage is not carried by one or two particular tracks.

This is the test. For every channel-supporting RGT:

    remove EVERY accepted observation of that RGT (all beams, all cycles)
    refit the variogram and the kriging system on the remaining RGTs only
    predict the held-out RGT's observations
    compare against D0 at the identical locations

It is deliberately the SAME design as K10b's leave-one-RGT-out -- the hardest
spatial-transfer test, which D1 failed globally -- restricted to the channel
stratum. The non-channel observations of the same held-out RGT are predicted in
the same fold and reported alongside as the internal control.

D0 is trained on legacy soundings only, so it is unaffected by the ICESat-2
folds by construction and is evaluated at the identical held-out points.

The D1_MEDIAN20 / D1_KALMAN20 columns required by D14 stay NaN here: the native
20 m canonical profiles do not exist yet (see k10d_pull_atl08_20m.py -- the
project holds no ATL08 granule, so 20 m has to be re-aggregated from ATL03
photons). This run answers D14 for D1_RAW, which is the variant K10c's verdict
was actually built on.

Outputs
-------
outputs/tables/channel_only_LOO_RGT.csv
outputs/figures/channel_LOO_RGT_results.png
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
MIN_CHANNEL_POINTS = 5      # an RGT below this cannot support a channel statement
N_BOOT = 2000


def rmse(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.sqrt((v ** 2).mean())) if len(v) else np.nan


def main() -> None:
    print("=" * 78)
    print("K10d.14 -- CHANNEL_ONLY_LEAVE_ONE_RGT_OUT")
    print("=" * 78)
    print("Does the channel-stratum D1 advantage survive when an ENTIRE channel-supporting")
    print("RGT is absent from training?  residual = H_DEM - H_ICESat2\n")

    g = pd.read_parquet(CUR / "k10_modern_elevation_points.parquet")
    prim = g[g.exposed_ground_validation_ok].copy().reset_index(drop=True)
    xy = np.c_[prim.x.values, prim.y.values]
    H = prim.H_terrain_EVRF2019_empirical_m.values
    is_chan = (prim.P_channel >= CHANNEL_P).values

    print(f"accepted set: {len(prim):,} segments, {prim.rgt.nunique()} RGTs")
    print(f"channel stratum (P_channel >= {CHANNEL_P}): {int(is_chan.sum()):,} segments")

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
    chan_per_rgt = pd.Series(is_chan).groupby(prim.rgt.values).sum()
    sup = chan_per_rgt[chan_per_rgt >= MIN_CHANNEL_POINTS]
    fold_rgts = sorted(int(r) for r in sup.index)
    print(f"channel-supporting RGTs (>= {MIN_CHANNEL_POINTS} channel segments): "
          f"{len(fold_rgts)} -> {fold_rgts}")
    print(f"channel segments per RGT: "
          f"{dict((int(k), int(v)) for k, v in sup.items())}\n")

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
                "RMSE_D1_MEDIAN20": np.nan, "RMSE_D1_KALMAN20": np.nan,
                "delta_RMSE_RAW": rmse(e1) - rmse(e0),
                "MAE_D0": float(np.abs(e0).mean()), "MAE_D1_RAW": float(np.abs(e1).mean()),
                "NMAD_D0": nmad(e0), "NMAD_D1_RAW": nmad(e1),
                "bias_D0": float(e0.mean()), "bias_D1_RAW": float(e1.mean()),
                "median_distance_to_training_m": float(np.median(dist_tr[m])),
                "min_distance_to_training_m": float(np.min(dist_tr[m])),
                "D1_better": bool(rmse(e1) < rmse(e0)),
                "vg_range_m": vg1[0], "n_train_segments": int(train.sum()),
                "d1_variant_note": "D1_RAW only; native 20 m canonical profiles not yet built",
            })
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
    out.to_csv(CFG.TABLES / "channel_only_LOO_RGT.csv", index=False)

    # ------------------------------------------------ pooled verdict --------
    pool = pd.concat(pooled, ignore_index=True)
    ch = out[out.stratum == "channel"]
    nc = out[out.stratum == "non_channel"]
    n_fav = int(ch.D1_better.sum())
    r0, r1 = rmse(pool.e0), rmse(pool.e1)

    rgts = pool.rgt.unique()
    idx = {q: np.where(pool.rgt.values == q)[0] for q in rgts}
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.choice(rgts, len(rgts), replace=True)
        sel = np.concatenate([idx[p] for p in pick])
        boots[b] = rmse(pool.e1.values[sel]) - rmse(pool.e0.values[sel])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    print("\n" + "=" * 78)
    print("POOLED CHANNEL RESULT (every fold's held-out RGT absent from training)")
    print("=" * 78)
    print(f"channel validation points {len(pool):,} over {len(rgts)} independent RGTs")
    print(f"  RMSE D0 {r0:.3f} m | RMSE D1_RAW {r1:.3f} m | delta {r1-r0:+.3f} m, "
          f"95% CI by RGT [{lo:+.3f}, {hi:+.3f}]")
    print(f"  RGTs where D1 beats D0 in the channel stratum: {n_fav}/{len(ch)}")
    if len(nc):
        print(f"  internal control, non-channel stratum of the SAME folds: "
              f"median fold RMSE D0 {nc.RMSE_D0.median():.3f} m vs D1 "
              f"{nc.RMSE_D1_RAW.median():.3f} m | RGTs favouring D1: "
              f"{int(nc.D1_better.sum())}/{len(nc)}")

    separable = hi < 0
    majority = n_fav >= max(3, int(np.ceil(0.6 * len(ch))))
    if separable and majority:
        verdict, decision = "PASS", ("A", "D1 beats D0 on most channel RGTs under full-RGT "
                                          "holdout -> strong support for a local channel "
                                          "correction")
    elif n_fav >= 3 and (separable or majority):
        verdict, decision = "PARTIAL", ("A/B", "the channel advantage reproduces on several "
                                               "independent RGTs but not decisively -> "
                                               "morphology-dependent hybridisation may be "
                                               "prepared, not yet applied")
    elif n_fav >= 1:
        verdict, decision = "FAIL", ("B", "the channel advantage rests on only 1-2 RGTs -> "
                                          "insufficient evidence for general hybridisation")
    else:
        verdict, decision = "FAIL", ("B", "no channel RGT reproduces the advantage under "
                                          "full-RGT holdout -> do not hybridise")

    print("\n" + "=" * 78)
    print("D15 -- CHANNEL EVIDENCE DECISION")
    print("=" * 78)
    print(f"outcome {decision[0]}: {decision[1]}")
    print(f"ICESat-2 channel added value: {verdict}")
    print("Kalman canonicalisation: NOT ASSESSED (native 20 m profiles do not exist yet)")
    print("morphology-dependent hybridisation: still NO-GO at this stage -- D20 forbids "
          "defining w(d, P_channel) before the 20 m branch is resolved.")
    print("\nNO HYBRID DEM IS BUILT. No weighting function is defined.")

    # ------------------------------------------------ figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ch_s = ch.sort_values("rgt")
    idxs = np.arange(len(ch_s))
    w = 0.38
    axes[0].bar(idxs - w / 2, ch_s.RMSE_D0, w, label="D0 historical M2", color="#8a94a3")
    axes[0].bar(idxs + w / 2, ch_s.RMSE_D1_RAW, w, label="D1 ICESat-2 OK", color="#236f8c")
    axes[0].set_xticks(idxs); axes[0].set_xticklabels(ch_s.rgt.astype(int), fontsize=8)
    axes[0].set_xlabel("held-out RGT (entire track removed)")
    axes[0].set_ylabel("RMSE, m")
    axes[0].set_title("A. channel stratum, P_channel >= 0.5", loc="left", fontsize=10)
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

    if len(nc):
        nc_s = nc.sort_values("rgt")
        axes[2].axhline(0, color="grey", lw=0.8)
        axes[2].bar(np.arange(len(nc_s)), nc_s.delta_RMSE_RAW, 0.6, color="#b8a24a")
        axes[2].set_xticks(np.arange(len(nc_s)))
        axes[2].set_xticklabels(nc_s.rgt.astype(int), fontsize=8)
        axes[2].set_ylabel("delta RMSE (D1 - D0), m")
        axes[2].set_title("C. internal control: non-channel, same folds", loc="left", fontsize=10)
    fig.suptitle("K10d.14 -- does the channel advantage survive removing a whole RGT?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "channel_LOO_RGT_results.png", dpi=150)
    print(f"\n-> {CFG.TABLES / 'channel_only_LOO_RGT.csv'}")
    print(f"-> {CFG.FIG / 'channel_LOO_RGT_results.png'}")


if __name__ == "__main__":
    main()
