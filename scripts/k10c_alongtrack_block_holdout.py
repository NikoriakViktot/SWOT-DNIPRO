#!/usr/bin/env python
"""K10c -- VALIDATION_B_LOCAL_INFLUENCE: along-track blocked holdout for the
ICESat-2 radius of influence.

This does NOT replace and does NOT modify K10b. K10b stays frozen as

    VALIDATION_A_GLOBAL_TRANSFER
    -> D0 historical M2 beats D1 ICESat-2 kriging when prediction requires
       inter-track extrapolation at ~10-20 km scales.

K10c answers a different question, and only that question:

    does D1 add predictive value NEAR an ICESat-2 ground track, and is there a
    defensible radius of influence?

Design
------
Unit of holdout: a CONTIGUOUS run of accepted ATL08 segments along ONE beam
ground track (rgt, gt), ordered by the native along-track coordinate x_atc.
Random segment holdout is never used -- along-track autocorrelation would make
it trivially optimistic.

Guard/embargo: applied ALONG TRACK:

    TRAIN | GUARD | TEST | GUARD | TRAIN

The guard is deliberately NOT a global geographic exclusion radius. A global
radius would force distance_to_nearest_training >= guard for every validation
point and would erase exactly the near-field bins the experiment exists to
measure. Support outside the embargoed window enters the analysis at its true
geometric distance, which is the independent variable:

    distance_to_nearest_training_segment_m

EMBARGO SCOPE -- this matters more than the guard width and is therefore run
BOTH ways, because ICESat-2 beams come in pairs ~90 m apart:

  BEAM      embargo the held-out beam track only. The paired beam of the SAME
            overpass (~90 m away, same date, same atmospheric state) stays in
            training. This is the OPERATIONAL question: in production a hybrid
            DEM would have every accepted beam available, so "distance to the
            nearest accepted ICESat-2 segment" is what a radius of influence
            would actually be keyed on.
  RGT_PASS  embargo EVERY beam of the same RGT inside the along-track window.
            The nearest surviving support is then a genuinely different overpass.
            This is the CONSERVATIVE question and it is the one that decides
            whether a near-field D1 advantage is real information transfer or
            merely same-overpass redundancy.

RGT_PASS is the HEADLINE scope. Under BEAM the nearest surviving support is the
paired beam ~90 m away for almost every held-out segment, so the distance axis
collapses onto one value and the bins beyond 0.5 km are nearly empty: that
configuration cannot answer a radius-of-influence question at all. A D1
advantage that appears under BEAM but disappears under RGT_PASS is NOT a radius
of influence -- it is the paired beam re-measuring the same ground, and the
decision logic below treats it as such.

Leakage: for every fold the variogram and the kriging system are re-fitted on
that fold's training data only. No full-data D1 surface is ever evaluated.
D0 is unaffected by the folds by construction -- it is trained on legacy
soundings, never on ICESat-2 -- so it is fitted once and predicted at every
validation segment, giving both models the same frozen reference points.

Primary set (B9): NIGHT + contemporaneously DRY + >= 50 m vector shoreline
buffer = exposed_ground_validation_ok, 13,905 segments before fold assignment.
The DAY sensitivity is not repeated here; it belongs to K10b.

Residual sign convention, unchanged:

    residual = H_DEM - H_ICESat2      (positive = DEM too high)

Outputs
-------
outputs/tables/validation_manifest_v2_alongtrack_blocks.csv   (per-segment)
outputs/tables/k10c_fold_registry.csv
outputs/tables/k10c_distance_bins.csv
outputs/tables/k10c_rgt_support.csv
outputs/tables/k10c_morphology_stratified.csv
outputs/tables/k10c_guard_sensitivity.csv
outputs/tables/k10c_vertical_sensitivity.csv
outputs/reports/K10c_alongtrack_block_holdout.md
outputs/figures/K10c_radius_of_influence.png
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
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
from k9_predictive_validation import fit_variogram, nmad, ok_predict, _FALLBACK

BATHY = CFG.ROOT / "data" / "processed" / "bathymetry"
CUR = CFG.ROOT / "data" / "processed" / "current_bed"
RNG = np.random.default_rng(CFG.SEED)

# ----------------------------------------------------------------- FROZEN ---
# block lengths: one representative value inside each requested range
BLOCK_LENGTHS_M = {1500: "1-2 km", 3500: "2-5 km", 7500: "5-10 km"}
MAX_BLOCKS_PER_TRACK = 2          # bounds cost, and stops one long track dominating
MIN_BLOCK_SEGMENTS = 5
MIN_TRAIN_SEGMENTS = 200

PRIMARY_GUARD_M = 1000            # headline analysis
GUARD_LEVELS_M = [250, 500, 1000, 2000, 5000]
GUARD_SWEEP_BLOCK_M = 3500        # guard sweep runs on the middle block scale only
GUARD_SWEEP_BLOCKS_PER_TRACK = 1
EMBARGO_SCOPES = ["BEAM", "RGT_PASS"]
# RGT_PASS is the HEADLINE scope. Under BEAM the paired beam sits ~90 m away for
# almost every held-out segment, so the distance axis collapses and the near-field
# bins measure same-overpass redundancy rather than transfer distance.
PRIMARY_SCOPE = "RGT_PASS"

DIST_BINS = [0, 500, 1000, 2000, 5000, 10000, np.inf]
DIST_LABELS = ["0-0.5 km", "0.5-1 km", "1-2 km", "2-5 km", "5-10 km", ">10 km"]
N_BOOT = 2000
PREDICTORS = ["P_former_channel", "distance_to_channel_m"]
SHORELINE_PROXIMAL_M = 150.0      # >= 50 m already enforced by QC; this splits near/far


def metrics(res, label, n_rgt=np.nan, n_blocks=np.nan):
    res = np.asarray(res, float)
    res = res[np.isfinite(res)]
    if not len(res):
        return {"set": label, "N": 0, "N_RGT": n_rgt, "N_blocks": n_blocks}
    return {"set": label, "N": len(res), "N_RGT": n_rgt, "N_blocks": n_blocks,
            "RMSE": float(np.sqrt((res ** 2).mean())),
            "MAE": float(np.abs(res).mean()),
            "mean_bias": float(res.mean()), "median_bias": float(np.median(res)),
            "NMAD": nmad(res),
            "P95_abs_error": float(np.percentile(np.abs(res), 95))}


def support_class(n_rgt, n_blocks, n_points):
    if n_points < 10 or n_rgt < 1:
        return "UNSUPPORTED"
    if n_rgt >= 3 and n_blocks >= 5:
        return "ROBUST_MULTI_RGT"
    if n_rgt >= 2:
        return "LIMITED_MULTI_RGT"
    return "EXPLORATORY_SINGLE_RGT"


def boot_delta_rmse(df, n_boot=N_BOOT):
    """delta_RMSE = RMSE(D1) - RMSE(D0), bootstrapped over whole RGTs.

    RGT is the resampling unit. 100 m ATL08 segments are never treated as
    independent draws, and blocks are carried with their parent RGT.
    """
    rgts = df.rgt.unique()
    if len(rgts) < 2:
        return np.nan, np.nan
    e0 = df.D0_error.values
    e1 = df.D1_error.values
    by_rgt = {r: np.where(df.rgt.values == r)[0] for r in rgts}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(rgts, len(rgts), replace=True)
        sel = np.concatenate([by_rgt[p] for p in pick])
        out[b] = np.sqrt((e1[sel] ** 2).mean()) - np.sqrt((e0[sel] ** 2).mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def build_blocks(prim, block_len, max_blocks):
    """Contiguous along-track blocks, walking each beam track from its start."""
    blocks = []
    for (rgt, gt), gg in prim.groupby(["rgt", "gt"], sort=True):
        pos = gg.index.values                      # positional index into prim
        xa = prim.x_atc.values[pos]
        order = np.argsort(xa)
        pos, xa = pos[order], xa[order]
        x0 = xa.min()
        n_made = 0
        k = 0
        while n_made < max_blocks and x0 + (k + 1) * block_len <= xa.max():
            s = x0 + k * block_len
            e = s + block_len
            sel = pos[(xa >= s) & (xa < e)]
            k += 1
            if len(sel) < MIN_BLOCK_SEGMENTS:
                continue
            blocks.append({"rgt": int(rgt), "gt": int(gt),
                           "block_id": f"R{int(rgt)}_G{int(gt)}_L{int(block_len)}_{n_made}",
                           "block_length_m": int(block_len),
                           "block_scale": BLOCK_LENGTHS_M[block_len],
                           "x_atc_start": float(s), "x_atc_end": float(e),
                           "rows": sel, "track_rows": pos, "track_xatc": xa})
            n_made += 1
    return blocks


def run_fold(block, guard_m, scope, prim, xy, H, d0_all, keys):
    """Hold out one contiguous along-track block + its along-track guard zone,
    refit the variogram and the kriging system on the remainder, predict."""
    s, e = block["x_atc_start"], block["x_atc_end"]
    rgt_a, gt_a, xatc_a = keys
    in_window = (xatc_a >= s - guard_m) & (xatc_a < e + guard_m)
    same = rgt_a == block["rgt"]
    if scope == "BEAM":
        same &= gt_a == block["gt"]
    embargo = in_window & same
    test = block["rows"]

    train_mask = ~embargo
    train_mask[test] = False
    if train_mask.sum() < MIN_TRAIN_SEGMENTS:
        return None

    xy_tr, z_tr = xy[train_mask], H[train_mask]
    xy_te, z_te = xy[test], H[test]

    vg = fit_variogram(xy_tr, z_tr)
    d1, _ = ok_predict(xy_tr, z_tr, xy_te, vg)
    dist_tr, _ = cKDTree(xy_tr).query(xy_te, k=1)

    sub = prim.iloc[test]
    return pd.DataFrame({
        "rgt": block["rgt"], "gt": block["gt"], "block_id": block["block_id"],
        "block_length_m": block["block_length_m"], "block_scale": block["block_scale"],
        "guard_m": guard_m, "embargo_scope": scope,
        "validation_date": sub.date.dt.date.astype(str).values,
        "x": sub.x.values, "y": sub.y.values,
        "reference_height": z_te,
        "D0_prediction": d0_all[test], "D1_prediction": d1,
        "D0_error": d0_all[test] - z_te, "D1_error": d1 - z_te,
        "distance_to_training": dist_tr,
        "P_channel": sub.P_channel.values,
        "morphology_class": sub.morphology_class.values,
        "distance_to_shoreline": sub.distance_to_water_edge_m.values,
        "n_train_segments": int(train_mask.sum()),
        "vg_range_m": vg[0], "vg_sill": vg[1], "vg_nugget": vg[2],
    })


def main() -> None:
    t_start = time.time()
    print("=" * 78)
    print("K10c -- VALIDATION_B_LOCAL_INFLUENCE (along-track blocked holdout)")
    print("=" * 78)
    print("K10b / VALIDATION_A_GLOBAL_TRANSFER is NOT modified. Its conclusion stands:")
    print("  D0 historical M2 beats D1 ICESat-2 kriging under inter-track extrapolation.")
    print("residual = H_DEM - H_ICESat2   (positive = DEM too high)\n")

    # ------------------------------------------------ primary set (B9, B10) --
    g = pd.read_parquet(CUR / "k10_modern_elevation_points.parquet")
    prim = g[g.exposed_ground_validation_ok].copy().reset_index(drop=True)
    prim["date"] = pd.to_datetime(prim.date)
    prim["morphology_class"] = np.where(
        prim.P_channel >= 0.5, "probable_channel",
        np.where(prim.distance_to_water_edge_m < SHORELINE_PROXIMAL_M,
                 "shoreline_proximal", "platform_floodplain"))
    print(f"primary set (NIGHT + DRY + >=50 m shoreline buffer): {len(prim):,} segments, "
          f"{prim.rgt.nunique()} RGTs, {prim.groupby(['rgt','gt']).ngroups} beam tracks")
    print("  water-QC traceability retained per segment: water_state, watermask date, "
          "delta_time_days, distance_to_water_edge_m")
    print("  morphology split: " + ", ".join(
        f"{k} {v:,}" for k, v in prim.morphology_class.value_counts().items()))

    xy = np.c_[prim.x.values, prim.y.values]
    H = prim.H_terrain_EVRF2019_empirical_m.values

    # ------------------------------------------------ D0, fitted once (B5) --
    print("\n" + "-" * 78)
    print("D0 = HISTORICAL_M2 (regression on frozen prior bands + residual kriging)")
    print("-" * 78)
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
    print(f"  soundings {len(sd):,}; residual variogram range {vg0[0]/1000:.2f} km, "
          f"sill {vg0[1]:.2f}, nugget {vg0[2]:.2f}")
    print(f"  D0 is independent of every ICESat-2 fold by construction (legacy soundings only),")
    print(f"  so it is fitted once and evaluated at the identical held-out reference points.")

    # ------------------------------------------------ fold construction (B2) -
    print("\n" + "-" * 78)
    print("B2/B3 -- CONTIGUOUS ALONG-TRACK BLOCKS + ALONG-TRACK GUARD")
    print("-" * 78)
    keys = (prim.rgt.values.astype(int), prim["gt"].values.astype(int), prim.x_atc.values)
    folds = []
    for L in BLOCK_LENGTHS_M:
        bl = build_blocks(prim, L, MAX_BLOCKS_PER_TRACK)
        for sc in EMBARGO_SCOPES:
            for b in bl:
                folds.append((b, PRIMARY_GUARD_M, sc, "PRIMARY"))
        print(f"  block length {L:5d} m ({BLOCK_LENGTHS_M[L]:7s}): {len(bl):4d} blocks, "
              f"{len(set(b['rgt'] for b in bl))} RGTs, "
              f"{sum(len(b['rows']) for b in bl):,} held-out segments")
    sweep = build_blocks(prim, GUARD_SWEEP_BLOCK_M, GUARD_SWEEP_BLOCKS_PER_TRACK)
    for gm in GUARD_LEVELS_M:
        for sc in EMBARGO_SCOPES:
            for b in sweep:
                folds.append((b, gm, sc, "GUARD_SWEEP"))
    print(f"  embargo scopes: {EMBARGO_SCOPES} (primary = {PRIMARY_SCOPE}; RGT_PASS also "
          f"removes the ~90 m paired beam of the same overpass)")
    print(f"  guard sweep: {len(sweep)} blocks x {len(GUARD_LEVELS_M)} guard levels "
          f"{GUARD_LEVELS_M} m (block scale {GUARD_SWEEP_BLOCK_M} m)")
    print(f"  total folds to fit: {len(folds):,} "
          f"(each refits variogram + kriging on its own training data)")

    # ------------------------------------------------ run folds (B4) --------
    print("\nfitting folds (variogram + kriging re-estimated inside every fold) ...")
    out, registry, n_skip = [], [], 0
    for i, (blk, gm, sc, exp) in enumerate(folds, 1):
        r = run_fold(blk, gm, sc, prim, xy, H, d0_all, keys)
        if r is None:
            n_skip += 1
            continue
        r["experiment"] = exp
        out.append(r)
        registry.append({"experiment": exp, "embargo_scope": sc, "block_id": blk["block_id"],
                         "rgt": blk["rgt"],
                         "gt": blk["gt"], "block_length_m": blk["block_length_m"],
                         "block_scale": blk["block_scale"], "guard_m": gm,
                         "n_validation_segments": len(r),
                         "n_train_segments": int(r.n_train_segments.iloc[0]),
                         "median_distance_to_training_m": float(r.distance_to_training.median()),
                         "vg_range_m": float(r.vg_range_m.iloc[0])})
        if i % 50 == 0 or i == len(folds):
            print(f"  {i:5d}/{len(folds)}  ({time.time()-t_start:6.0f} s elapsed)", flush=True)
    res = pd.concat(out, ignore_index=True)
    reg = pd.DataFrame(registry)
    reg.to_csv(CFG.TABLES / "k10c_fold_registry.csv", index=False)
    print(f"folds fitted {len(reg):,}, skipped for insufficient training data {n_skip}")
    print(f"kriging integrity: weight-guard fallbacks {_FALLBACK['n_bad']:,} of "
          f"{_FALLBACK['n_total']:,} solved systems, envelope fallbacks {_FALLBACK['n_env']:,}")

    res["dist_class"] = pd.cut(res.distance_to_training, DIST_BINS,
                               labels=DIST_LABELS, right=False)
    # A validation segment outside the morphology_prior_v1 footprint has no D0
    # prediction at all, so it cannot enter a PAIRED D0-vs-D1 comparison. Such rows
    # are flagged and excluded from every metric, but kept in the manifest.
    res["paired_comparison_ok"] = np.isfinite(res.D0_error) & np.isfinite(res.D1_error)
    n_unpaired = int((~res.paired_comparison_ok).sum())
    if n_unpaired:
        print(f"\n{n_unpaired} held-out segments have no D0 prediction (outside the "
              f"morphology_prior_v1 footprint) and are excluded from all paired metrics; "
              f"they stay in the manifest flagged paired_comparison_ok = False.")
    res_ok = res[res.paired_comparison_ok]
    pri_all = res_ok[res_ok.experiment == "PRIMARY"].copy()
    pri = pri_all[pri_all.embargo_scope == PRIMARY_SCOPE].copy()
    alt = pri_all[pri_all.embargo_scope == "BEAM"].copy()

    print(f"\nheld-out validation segments: {len(res):,} total, {len(pri):,} in the PRIMARY "
          f"experiment / {PRIMARY_SCOPE} embargo "
          f"({pri.rgt.nunique()} RGTs, {pri.block_id.nunique()} blocks)")
    print("distance from held-out segment to nearest surviving training segment:")
    for name, dd in (("BEAM    ", alt), ("RGT_PASS", pri)):
        if len(dd):
            print(f"  {name}: median {dd.distance_to_training.median():7.0f} m, "
                  f"p10 {dd.distance_to_training.quantile(.10):7.0f} m, "
                  f"p90 {dd.distance_to_training.quantile(.90):8.0f} m, "
                  f"max {dd.distance_to_training.max():8.0f} m")

    # ------------------------------------------------ B6/B7 distance bins ---
    def distance_table(df, n_boot=N_BOOT):
        rows = []
        for dc in DIST_LABELS:
            gg = df[df.dist_class == dc]
            if not len(gg):
                rows.append({"dist_class": dc, "n_points": 0, "n_blocks": 0, "n_RGT": 0,
                             "support": "UNSUPPORTED"})
                continue
            nr, nb = gg.rgt.nunique(), gg.block_id.nunique()
            m0 = metrics(gg.D0_error, "D0", nr, nb)
            m1 = metrics(gg.D1_error, "D1", nr, nb)
            lo, hi = boot_delta_rmse(gg, n_boot=n_boot)
            rows.append({
                "dist_class": dc, "n_points": len(gg), "n_blocks": nb, "n_RGT": nr,
                "support": support_class(nr, nb, len(gg)),
                "RMSE_D0": m0["RMSE"], "RMSE_D1": m1["RMSE"],
                "delta_RMSE": m1["RMSE"] - m0["RMSE"],
                "delta_RMSE_ci_lo": lo, "delta_RMSE_ci_hi": hi,
                "MAE_D0": m0["MAE"], "MAE_D1": m1["MAE"],
                "NMAD_D0": m0["NMAD"], "NMAD_D1": m1["NMAD"],
                "bias_D0": m0["mean_bias"], "bias_D1": m1["mean_bias"],
                "D1_better": bool(np.isfinite(hi) and hi < 0),
                "D0_better": bool(np.isfinite(lo) and lo > 0)})
        return pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("B6/B7 -- ACCURACY BY DISTANCE TO NEAREST ICESat-2 TRAINING SEGMENT")
    print("=" * 78)
    bydist = distance_table(pri)
    bydist_alt = distance_table(alt)
    allbins = pd.concat([bydist.assign(embargo_scope=PRIMARY_SCOPE),
                         bydist_alt.assign(embargo_scope="BEAM")], ignore_index=True)
    allbins.to_csv(CFG.TABLES / "k10c_distance_bins.csv", index=False)
    show = ["dist_class", "n_points", "n_blocks", "n_RGT", "support", "RMSE_D0",
            "RMSE_D1", "delta_RMSE", "delta_RMSE_ci_lo", "delta_RMSE_ci_hi"]
    print("\n[HEADLINE -- RGT_PASS embargo: the whole overpass is removed, so the nearest "
          "surviving support is a genuinely different pass]")
    print(bydist[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[BEAM embargo: the paired beam of the SAME overpass (~90 m) stays in training. "
          "The distance axis collapses, so this is a redundancy check, not a radius measurement]")
    print(bydist_alt[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\ndelta_RMSE = RMSE(D1) - RMSE(D0); negative = D1 better. CI is a 2000x bootstrap")
    print("over whole RGTs -- never over individual 100 m segments.")

    # per-RGT support inside each bin
    srows = []
    for dc in DIST_LABELS:
        gg = pri[pri.dist_class == dc]
        for rgt, hh in gg.groupby("rgt"):
            srows.append({"dist_class": dc, "rgt": int(rgt), "n_points": len(hh),
                          "n_blocks": hh.block_id.nunique(),
                          "RMSE_D0": float(np.sqrt((hh.D0_error ** 2).mean())),
                          "RMSE_D1": float(np.sqrt((hh.D1_error ** 2).mean()))})
    sup = pd.DataFrame(srows)
    if len(sup):
        sup["D1_better"] = sup.RMSE_D1 < sup.RMSE_D0
        sup.to_csv(CFG.TABLES / "k10c_rgt_support.csv", index=False)
        print("\nper-RGT reproducibility inside each distance bin "
              "(how many independent tracks agree):")
        agr = sup.groupby("dist_class", observed=True).agg(
            n_RGT=("rgt", "nunique"), n_RGT_favouring_D1=("D1_better", "sum"))
        print(agr.reindex(DIST_LABELS).to_string())

    # ------------------------------------------------ B8 morphology ---------
    print("\n" + "=" * 78)
    print("B8 -- MORPHOLOGY-STRATIFIED RADIUS OF INFLUENCE")
    print("=" * 78)
    mrows = []
    strata = [("P_channel>=0.5", pri[pri.P_channel >= 0.5]),
              ("P_channel<0.5", pri[pri.P_channel < 0.5])]
    strata += [(f"class:{c}", pri[pri.morphology_class == c])
               for c in sorted(pri.morphology_class.unique())]
    for name, ss in strata:
        for dc in DIST_LABELS:
            gg = ss[ss.dist_class == dc]
            if len(gg) < 10:
                continue
            nr, nb = gg.rgt.nunique(), gg.block_id.nunique()
            lo, hi = boot_delta_rmse(gg, n_boot=500)
            r0 = float(np.sqrt((gg.D0_error ** 2).mean()))
            r1 = float(np.sqrt((gg.D1_error ** 2).mean()))
            mrows.append({"stratum": name, "dist_class": dc, "n_points": len(gg),
                          "n_blocks": nb, "n_RGT": nr,
                          "support": support_class(nr, nb, len(gg)),
                          "RMSE_D0": r0, "RMSE_D1": r1, "delta_RMSE": r1 - r0,
                          "delta_RMSE_ci_lo": lo, "delta_RMSE_ci_hi": hi,
                          "D1_better": bool(np.isfinite(hi) and hi < 0)})
    morph = pd.DataFrame(mrows)
    morph.to_csv(CFG.TABLES / "k10c_morphology_stratified.csv", index=False)
    for name, _ in strata:
        mm = morph[morph.stratum == name]
        if not len(mm):
            continue
        print(f"\n{name}:")
        print(mm[["dist_class", "n_points", "n_RGT", "support", "RMSE_D0", "RMSE_D1",
                  "delta_RMSE", "delta_RMSE_ci_lo", "delta_RMSE_ci_hi"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------ B3 guard sensitivity --
    print("\n" + "=" * 78)
    print("B3 -- GUARD / EMBARGO SENSITIVITY (does the near field survive a wider embargo?)")
    print("=" * 78)
    sw = res_ok[res_ok.experiment == "GUARD_SWEEP"]
    grows = []
    for (gm, sc), gg in sw.groupby(["guard_m", "embargo_scope"]):
        for dc in DIST_LABELS + ["ALL"]:
            hh = gg if dc == "ALL" else gg[gg.dist_class == dc]
            if len(hh) < 10:
                continue
            nr, nb = hh.rgt.nunique(), hh.block_id.nunique()
            lo, hi = boot_delta_rmse(hh, n_boot=500)
            r0 = float(np.sqrt((hh.D0_error ** 2).mean()))
            r1 = float(np.sqrt((hh.D1_error ** 2).mean()))
            grows.append({"guard_m": int(gm), "embargo_scope": sc,
                          "dist_class": dc, "n_points": len(hh),
                          "n_blocks": nb, "n_RGT": nr,
                          "support": support_class(nr, nb, len(hh)),
                          "median_distance_to_training_m": float(hh.distance_to_training.median()),
                          "RMSE_D0": r0, "RMSE_D1": r1, "delta_RMSE": r1 - r0,
                          "delta_RMSE_ci_lo": lo, "delta_RMSE_ci_hi": hi})
    guard = pd.DataFrame(grows)
    guard.to_csv(CFG.TABLES / "k10c_guard_sensitivity.csv", index=False)
    print(guard.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------ B11 vertical bridge ---
    c_sd = float(g.sigma_c_m.iloc[0])
    c_emp = float(g.c_empirical_m.iloc[0])
    vrows = []
    for dc_ in (-c_sd, 0.0, +c_sd):
        vrows.append({"delta_c_m": dc_,
                      "D0_bias": float((pri.D0_error - dc_).mean()),
                      "D0_RMSE": float(np.sqrt(((pri.D0_error - dc_) ** 2).mean())),
                      "D0_NMAD": nmad(pri.D0_error - dc_),
                      "D1_bias": float(pri.D1_error.mean()),
                      "D1_RMSE": float(np.sqrt((pri.D1_error ** 2).mean())),
                      "D1_NMAD": nmad(pri.D1_error)})
    vs = pd.DataFrame(vrows)
    vs.to_csv(CFG.TABLES / "k10c_vertical_sensitivity.csv", index=False)
    print("\n" + "=" * 78)
    print(f"B11 -- VERTICAL-BRIDGE SENSITIVITY (c = {c_emp:+.3f} +/- {c_sd:.3f} m)")
    print("=" * 78)
    print(vs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  D1 and its reference share the same empirical bridge, so a D1 bias is NOT an")
    print("  independent test of the vertical datum. Only D0 carries the bridge uncertainty.")

    # ------------------------------------------------ B12 crossover ---------
    ordered = bydist[bydist.n_points > 0].set_index("dist_class").reindex(DIST_LABELS).dropna(subset=["RMSE_D0"])
    robust = ordered[ordered.support.isin(["ROBUST_MULTI_RGT", "LIMITED_MULTI_RGT"])]
    d1_wins = robust[robust.D1_better]
    alt_ord = bydist_alt[bydist_alt.n_points > 0].set_index("dist_class")
    alt_wins = set(alt_ord[alt_ord.D1_better
                           & alt_ord.support.isin(["ROBUST_MULTI_RGT",
                                                   "LIMITED_MULTI_RGT"])].index)
    # Is the aggregate near-field gain actually confined to one morphology stratum?
    # If channel cells win robustly and non-channel cells never do, a single global
    # radius would be the wrong object to fit, whatever the pooled bins say.
    def _wins(stratum):
        w = morph[(morph.stratum == stratum) & morph.D1_better
                  & morph.support.isin(["ROBUST_MULTI_RGT", "LIMITED_MULTI_RGT"])]
        return [d for d in DIST_LABELS if d in set(w.dist_class)]
    chan_wins = _wins("P_channel>=0.5") if len(morph) else []
    nonchan_wins = _wins("P_channel<0.5") if len(morph) else []
    morph_confined = bool(chan_wins) and not nonchan_wins

    crossover, outcome = None, None
    if morph_confined:
        last_c = chan_wins[-1]
        i_c = DIST_LABELS.index(last_c)
        nxt_c = DIST_LABELS[i_c + 1] if i_c + 1 < len(DIST_LABELS) else None
        crossover = (f"P_channel>=0.5: {last_c} -> {nxt_c}" if nxt_c else f"beyond {last_c}")
        outcome = ("B", f"the added value is CONFINED to P_channel>=0.5, where D1 wins out to "
                        f"{last_c}, while P_channel<0.5 never separates from D0 at any distance "
                        f"-> use morphology- and distance-dependent confidence weighting, NOT one "
                        f"global radius")
    elif len(d1_wins):
        last = d1_wins.index[-1]
        nxt = DIST_LABELS[DIST_LABELS.index(last) + 1] if DIST_LABELS.index(last) + 1 < len(DIST_LABELS) else None
        crossover = f"{last} -> {nxt}" if nxt else f"beyond {last}"
        near_only = all(DIST_LABELS.index(i) <= 1 for i in d1_wins.index)
        rgt_consistent = True
        if len(sup):
            for dc in d1_wins.index:
                ss = sup[sup.dist_class == dc]
                if len(ss) and ss.D1_better.mean() < 0.6:
                    rgt_consistent = False
        if not rgt_consistent:
            outcome = ("B", "the crossover varies strongly between RGTs -> use spatially varying "
                            "confidence weighting, NOT one radius")
        elif near_only:
            outcome = ("D", "D1 helps only inside a narrow track corridor -> keep D1 as a local "
                            "correction / validation support, not a basin-wide bathymetric source")
        else:
            outcome = ("A", "a stable crossover exists across independent overpasses -> a radius "
                            "of influence may be used for hybridisation")
    elif alt_wins:
        outcome = ("D", "D1 beats D0 only while the ~90 m paired beam of the SAME overpass stays "
                        "in training, and never once the whole overpass is embargoed -> "
                        "same-overpass redundancy, not a transferable radius of influence; keep "
                        "D1 as local correction / validation support only")
    else:
        outcome = ("C", "D1 never robustly beats D0 at any supported distance -> do NOT build a "
                        "D0/D1 hybrid DEM")

    verdict_local = "PASS" if outcome[0] == "A" else "PARTIAL" if outcome[0] in ("B", "D") else "FAIL"
    go = "GO" if outcome[0] == "A" else "CONDITIONAL" if outcome[0] in ("B", "D") else "NO-GO"

    print("\n" + "=" * 78)
    print("B12/B13 -- RADIUS OF INFLUENCE AND HYBRIDISATION DECISION")
    print("=" * 78)
    print(f"outcome {outcome[0]}: {outcome[1]}")
    print(f"crossover (last supported bin where D1 beats D0, CI excluding zero): "
          f"{crossover if crossover else 'none'}")
    print(f"bins where D1 wins under the headline RGT_PASS embargo: "
          f"{sorted(d1_wins.index) if len(d1_wins) else 'none'}")
    print(f"bins where D1 wins under the optimistic BEAM embargo (paired beam retained): "
          f"{sorted(alt_wins) if alt_wins else 'none'}")
    print(f"morphology: D1 wins on P_channel>=0.5 in {chan_wins if chan_wins else 'no bin'}; "
          f"on P_channel<0.5 in {nonchan_wins if nonchan_wins else 'no bin'}")
    print(f"D1 local added value: {verdict_local}")
    print(f"hybridisation: {go}")
    print("NO HYBRID DEM IS BUILT HERE (B13). Nothing is blended.")

    # ------------------------------------------------ per-segment manifest --
    keep = ["experiment", "embargo_scope", "rgt", "gt", "block_id", "block_length_m",
            "block_scale", "guard_m",
            "validation_date", "x", "y", "reference_height", "D0_prediction", "D1_prediction",
            "D0_error", "D1_error", "distance_to_training", "dist_class", "P_channel",
            "morphology_class", "distance_to_shoreline", "paired_comparison_ok",
            "n_train_segments", "vg_range_m", "vg_sill", "vg_nugget"]
    man = res[keep].copy()
    sup_map = allbins.set_index(["embargo_scope", "dist_class"]).support.to_dict()
    man["validation_support_class"] = [
        sup_map.get((s, str(d)), "UNSUPPORTED")
        for s, d in zip(man.embargo_scope, man.dist_class)]
    mp = CFG.TABLES / "validation_manifest_v2_alongtrack_blocks.csv"
    man.to_csv(mp, index=False)
    print(f"\n-> {mp}  ({len(man):,} held-out segments; K10b outputs untouched)")

    # ------------------------------------------------ figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ok = bydist[bydist.n_points > 0]
    xs = np.arange(len(ok))
    axes[0].plot(xs, ok.RMSE_D0, "o-", color="#8a94a3", label="D0 historical M2")
    axes[0].plot(xs, ok.RMSE_D1, "s-", color="#236f8c", label="D1 ICESat-2 OK")
    for i, (_, r) in enumerate(ok.iterrows()):
        axes[0].annotate(f"n={int(r.n_points)}\n{int(r.n_RGT)} RGT", (i, r.RMSE_D0),
                         xytext=(0, -32), textcoords="offset points", ha="center", fontsize=7)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(ok.dist_class, fontsize=8)
    axes[0].set_xlabel("distance to nearest surviving training segment")
    axes[0].set_ylabel("RMSE, m")
    axes[0].set_title("A. along-track blocked holdout", loc="left", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].errorbar(xs, ok.delta_RMSE,
                     yerr=[np.abs(ok.delta_RMSE - ok.delta_RMSE_ci_lo),
                           np.abs(ok.delta_RMSE_ci_hi - ok.delta_RMSE)],
                     fmt="o", color="#c1402a", capsize=4)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(ok.dist_class, fontsize=8)
    axes[1].set_ylabel("delta RMSE (D1 - D0), m")
    axes[1].set_title("B. negative = D1 better (95% CI bootstrapped by RGT)",
                      loc="left", fontsize=10)

    gsw = guard[(guard.dist_class == "ALL") & (guard.embargo_scope == PRIMARY_SCOPE)]
    if len(gsw):
        axes[2].plot(gsw.guard_m, gsw.RMSE_D0, "o-", color="#8a94a3", label="D0")
        axes[2].plot(gsw.guard_m, gsw.RMSE_D1, "s-", color="#236f8c", label="D1")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("along-track guard / embargo, m")
        axes[2].set_ylabel("RMSE, m")
        axes[2].set_title("C. guard sensitivity (2-5 km blocks)", loc="left", fontsize=10)
        axes[2].legend(fontsize=8)
    fig.suptitle("K10c -- VALIDATION_B_LOCAL_INFLUENCE: does ICESat-2 kriging add value "
                 "near its own ground tracks?", fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K10c_radius_of_influence.png", dpi=150)

    # ------------------------------------------------ report ----------------
    def md(df_, cols=None):
        d_ = df_[cols] if cols else df_
        return "```\n" + d_.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```"

    rep = [
        "# K10c -- VALIDATION_B_LOCAL_INFLUENCE (along-track blocked holdout)",
        "",
        f"**D1 local added value: {verdict_local}. Hybridisation: {go}. "
        f"Outcome {outcome[0]} -- {outcome[1]}.**",
        "",
        "K10b / `VALIDATION_A_GLOBAL_TRANSFER` is untouched and still stands: D0 historical M2",
        "outperforms D1 ICESat-2 kriging when prediction requires inter-track extrapolation at",
        "~10-20 km scales. K10c asks only whether D1 adds value *near* its own ground tracks.",
        "",
        "## Design",
        "",
        f"- Holdout unit: contiguous along-track run of accepted ATL08 segments on one beam "
        f"track `(rgt, gt)`, ordered by `x_atc`. Block scales {list(BLOCK_LENGTHS_M.values())}.",
        f"- Guard/embargo applied **along track** (primary {PRIMARY_GUARD_M} m; swept over "
        f"{GUARD_LEVELS_M} m). A global geographic exclusion radius was deliberately rejected: "
        f"it would force `distance_to_nearest_training >= guard` everywhere and erase the "
        f"near-field bins.",
        "- Embargo scope is run **both ways**, because ICESat-2 beams come in pairs ~90 m apart. "
        "`RGT_PASS` (headline) removes every beam of the overpass, so the nearest surviving "
        "support is a different pass. `BEAM` keeps the paired beam, which pins almost every "
        "held-out segment at ~90 m from training: its distance axis collapses, so it is reported "
        "as a redundancy check, not as a radius measurement.",
        "- Per fold the variogram and the kriging system are refitted on training data only; no "
        "full-data D1 surface is ever evaluated (B4).",
        "- D0 is trained on legacy soundings only, so it is independent of every fold by "
        "construction and is evaluated at the identical held-out reference points (B5).",
        f"- Primary set: NIGHT + contemporaneously DRY + >=50 m vector shoreline buffer, "
        f"{len(prim):,} segments before fold assignment (B9).",
        "",
        "## Returned quantities (B13)",
        "",
        f"1. RGTs represented: **{pri.rgt.nunique()}** of {prim.rgt.nunique()} accepted "
        f"({pri['gt'].nunique()} beam ids, {reg[reg.experiment=='PRIMARY'].block_id.nunique()} "
        f"blocks in the primary experiment)",
        f"2. Held-out blocks: **{reg.block_id.nunique()}** distinct blocks, "
        f"{len(reg):,} fitted folds ({len(res):,} held-out segment predictions)",
        f"3. Validation-to-training distance: median "
        f"{pri.distance_to_training.median():.0f} m, p10 "
        f"{pri.distance_to_training.quantile(.10):.0f} m, p90 "
        f"{pri.distance_to_training.quantile(.90):.0f} m, max "
        f"{pri.distance_to_training.max():.0f} m",
        "",
        "### 4-6. D0/D1 accuracy, delta CI and RGT support by distance",
        "",
        "**RGT_PASS embargo -- HEADLINE** (whole overpass removed; nearest support is a "
        "genuinely different pass):",
        "",
        md(bydist, ["dist_class", "n_points", "n_blocks", "n_RGT", "support", "RMSE_D0",
                    "RMSE_D1", "delta_RMSE", "delta_RMSE_ci_lo", "delta_RMSE_ci_hi",
                    "NMAD_D0", "NMAD_D1", "bias_D0", "bias_D1"]),
        "",
        "**BEAM embargo -- redundancy check only** (paired beam of the same overpass stays in "
        "training, ~90 m away, so the distance axis collapses):",
        "",
        md(bydist_alt, ["dist_class", "n_points", "n_blocks", "n_RGT", "support", "RMSE_D0",
                        "RMSE_D1", "delta_RMSE", "delta_RMSE_ci_lo", "delta_RMSE_ci_hi",
                        "NMAD_D0", "NMAD_D1", "bias_D0", "bias_D1"]),
        "",
        "delta_RMSE = RMSE(D1) - RMSE(D0); negative means D1 better. The 95% CI is a "
        f"{N_BOOT}x bootstrap resampling **whole RGTs**, never individual 100 m segments.",
        "",
        f"### 7. Morphology-stratified ({PRIMARY_SCOPE} embargo)",
        "",
        md(morph, ["stratum", "dist_class", "n_points", "n_RGT", "support", "RMSE_D0",
                   "RMSE_D1", "delta_RMSE", "delta_RMSE_ci_lo", "delta_RMSE_ci_hi"]) if len(morph)
        else "_no stratum reached the minimum support._",
        "",
        "### Guard / embargo sensitivity (B3)",
        "",
        md(guard, ["guard_m", "embargo_scope", "dist_class", "n_points", "n_RGT", "support",
                   "median_distance_to_training_m", "RMSE_D0", "RMSE_D1", "delta_RMSE",
                   "delta_RMSE_ci_lo", "delta_RMSE_ci_hi"]),
        "",
        "### 8. Does the exploratory 1-2 km crossover survive?",
        "",
        f"Last supported bin in which D1 beats D0 with a CI excluding zero: "
        f"**{crossover if crossover else 'none'}**.",
        "",
        f"- bins where D1 wins under the headline `RGT_PASS` embargo: "
        f"`{sorted(d1_wins.index) if len(d1_wins) else 'none'}`",
        f"- bins where D1 wins under the optimistic `BEAM` embargo (paired beam retained): "
        f"`{sorted(alt_wins) if alt_wins else 'none'}`",
        f"- morphology: D1 wins on `P_channel>=0.5` in `{chan_wins if chan_wins else 'no bin'}`, "
        f"on `P_channel<0.5` in `{nonchan_wins if nonchan_wins else 'no bin'}`",
        "",
        "### 9-11. Decision",
        "",
        f"- proposed radius of influence: "
        f"**{crossover if outcome[0] == 'A' else 'not defensible from this evidence'}**",
        f"- D1 local added value: **{verdict_local}**",
        f"- hybridisation: **{go}** -- {outcome[1]}",
        "",
        "No hybrid DEM is built (B13). D0 and D1 are not blended anywhere in this script.",
        "",
        "### Vertical bridge (B11)",
        "",
        md(vs),
        "",
        "Perturbing `c` moves the ICESat-2 reference and D1 together, so a D1 bias is not an "
        "independent check of the vertical datum. Only D0 carries the bridge uncertainty.",
        "",
        "### Kriging integrity",
        "",
        f"weight-guard fallbacks {_FALLBACK['n_bad']:,} of {_FALLBACK['n_total']:,} solved "
        f"systems; envelope fallbacks {_FALLBACK['n_env']:,}. Folds skipped for insufficient "
        f"training data: {n_skip}.",
        "",
        f"_written {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
    ]
    rp = CFG.REPORTS / "K10c_alongtrack_block_holdout.md"
    rp.write_text("\n".join(rep))
    print(f"-> {rp}")
    print(f"-> {CFG.FIG / 'K10c_radius_of_influence.png'}")
    print(f"\ntotal runtime {time.time()-t_start:.0f} s")


if __name__ == "__main__":
    main()
