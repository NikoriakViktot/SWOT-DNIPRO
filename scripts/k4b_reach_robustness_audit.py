#!/usr/bin/env python
"""K4b -- reach-stratified / longitudinal-detrend / buffer-geometry robustness
audit, requested after K4's first (GO) verdict. The operator's point: a global
Spearman rho with p~0 on ~7300 spatially autocorrelated points is not, by
itself, strong evidence -- and a positive rho(H, P_channel) alongside a
monotonically DECREASING median H across morphology classes is surprising
enough that it needs to be explained, not just re-reported.

Does NOT touch the frozen morphology_prior_v1.tif (K4's freeze stands). This
is a read-only audit of it, plus one fresh rebuild of the K3 per-date loop
using a true Euclidean buffer (distance_transform_edt) instead of iterated
binary_dilation, to check whether the diamond-vs-circle buffer geometry
changes the result.

Checks implemented (see report for what is explicitly NOT covered -- this is
not the full physics-informed v2 prior or the K9 nested-spatial-CV gate,
both reserved for later per the operator's own staging):
  1. rho(H, P), rho(H, d), rho(P, d) -- restated for reference
  2. reach-stratified rho(H, P | reach) -- Simpson's-paradox check
  3. longitudinal detrend: r = H - f(chain_km) (lowess baseline),
     then rho(r, P) and rho(r, d) -- does P explain variance BEYOND the
     along-reservoir gradient?
  4. partial OLS: H ~ chain_km + distance_to_channel + P_channel (HC3),
     does the P_channel coefficient survive controlling for chainage+distance?
  5. buffer-geometry robustness: binary_dilation (diamond, connectivity=1)
     vs distance_transform_edt with sampling=(250,250) (true Euclidean) for
     the 2 km "observed vicinity" buffer -- Jaccard of P>=0.5 masks, and
     whether the correlation / verdict changes under the EDT variant.

Explicitly NOT done here (flagged, not silently skipped):
  - the physics-informed hydraulic-connectivity/cost-distance prior (v2)
  - true per-date cloud/valid-pixel masks for the obs_count denominator
    (obs_count here means "near detected water this date", a proxy -- not
    "this cell was actually cloud-free and inspected this date")
  - spatially blocked / variogram-informed CV (reserved for K9)

Outputs
-------
outputs/tables/k4b_reach_stratified_correlation.csv
outputs/tables/k4b_detrended_partial_correlation.csv
outputs/tables/k4b_dilation_vs_edt_robustness.csv
outputs/figures/K4b_reach_stratified_and_detrended.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from rasterio import features
from scipy import stats as sstats
from scipy.ndimage import binary_dilation, distance_transform_edt
from shapely import wkt as shwkt
from statsmodels.nonparametric.smoothers_lowess import lowess

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

BATHY_DIR = CFG.ROOT / "data" / "processed" / "bathymetry"
CELL_M = 250.0
BUFFER_M = 2000.0
REACH_BREAKS = (-60, 0, 60, 133, 183, 230, 277.2, 330)  # finer than H12's 3-zone split


def main() -> None:
    # ---- 1. load frozen prior + soundings + chainage ----------------------
    print("=" * 70)
    print("K4b -- reach-stratified / detrend / buffer-geometry audit")
    print("(does not modify the frozen morphology_prior_v1.tif)")
    print("=" * 70)

    with rasterio.open(BATHY_DIR / "morphology_prior_v1.tif") as src:
        P = src.read(1)
        D = src.read(2)
        Nobs = src.read(3)
        transform = src.transform
        H, W = P.shape
        P = np.where(P == src.nodata, np.nan, P)

    sd = pd.read_parquet(BATHY_DIR / "kakhovka_soundings_evrf2019.parquet").copy()
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    chain_km, sword_dist_km, _, _ = SW.assign_chainage(sd.lon.values, sd.lat.values, ch, tree=tree)
    sd["chain_km"] = chain_km

    inv = ~transform
    cols, rows = inv * (sd.x.values, sd.y.values)
    cols, rows = np.clip(cols.astype(int), 0, W - 1), np.clip(rows.astype(int), 0, H - 1)
    sd["P_former_channel"] = P[rows, cols]
    sd["distance_to_channel_m"] = D[rows, cols]
    sd["n_dates_observed"] = Nobs[rows, cols]
    valid = sd.P_former_channel.notna() & (sd.n_dates_observed > 0) & sd.chain_km.notna()
    sv = sd[valid].copy()
    print(f"soundings with valid prior + chainage: {len(sv)} / {len(sd)}")
    print(f"chain_km range in sounding footprint: [{sv.chain_km.min():.1f}, {sv.chain_km.max():.1f}] km "
          f"(SWORD dist_out re-referenced to Kakhovka dam, positive upstream)")

    # ---- 2. global correlations (restated) ---------------------------------
    r_hp, p_hp = sstats.spearmanr(sv.H_bed_evrf2019_m, sv.P_former_channel)
    r_hd, p_hd = sstats.spearmanr(sv.H_bed_evrf2019_m, sv.distance_to_channel_m)
    r_pd, p_pd = sstats.spearmanr(sv.P_former_channel, sv.distance_to_channel_m)
    print(f"\nglobal: rho(H,P)={r_hp:+.3f} (p={p_hp:.1e}), rho(H,d)={r_hd:+.3f} (p={p_hd:.1e}), "
          f"rho(P,d)={r_pd:+.3f} (p={p_pd:.1e})")
    print("  note the sign convention here differs from K4's print statement: K4 correlated P "
          "against -H (so higher P -> more negative H there read as positive rho). Here rho(H,P) "
          "is against RAW H, so the expected physical sign is NEGATIVE (higher P -> lower/deeper H).")

    # ---- 3. reach-stratified rho(H,P) --------------------------------------
    sv["reach_bin"] = pd.cut(sv.chain_km, REACH_BREAKS, include_lowest=True)
    reach_rows = []
    for rb, g in sv.groupby("reach_bin", observed=True):
        if len(g) < 20:
            reach_rows.append({"reach_bin": str(rb), "n": len(g), "rho_H_P": np.nan, "p_H_P": np.nan,
                              "median_H": g.H_bed_evrf2019_m.median() if len(g) else np.nan,
                              "note": "n<20, not tested"})
            continue
        rr, pp = sstats.spearmanr(g.H_bed_evrf2019_m, g.P_former_channel)
        reach_rows.append({"reach_bin": str(rb), "n": len(g), "rho_H_P": rr, "p_H_P": pp,
                          "median_H": g.H_bed_evrf2019_m.median(), "note": ""})
    reach_df = pd.DataFrame(reach_rows)
    reach_df.to_csv(CFG.TABLES / "k4b_reach_stratified_correlation.csv", index=False)
    print(f"\nreach-stratified rho(H,P) (Simpson's-paradox check):")
    print(reach_df.to_string(index=False))

    # ---- 4. longitudinal detrend + partial correlation --------------------
    order = np.argsort(sv.chain_km.values)
    lo = lowess(sv.H_bed_evrf2019_m.values[order], sv.chain_km.values[order], frac=0.15, return_sorted=True)
    baseline = np.interp(sv.chain_km.values, lo[:, 0], lo[:, 1])
    sv["H_detrended"] = sv.H_bed_evrf2019_m.values - baseline

    r_rp, p_rp = sstats.spearmanr(sv.H_detrended, sv.P_former_channel)
    r_rd, p_rd = sstats.spearmanr(sv.H_detrended, sv.distance_to_channel_m)
    print(f"\nafter removing longitudinal baseline f(chain_km) (lowess, frac=0.15):")
    print(f"  rho(H_detrended, P) = {r_rp:+.3f} (p={p_rp:.1e})  [expect NEGATIVE if P carries "
          f"morphological info beyond the along-reservoir gradient]")
    print(f"  rho(H_detrended, d) = {r_rd:+.3f} (p={p_rd:.1e})  [expect POSITIVE]")

    Xp = sm.add_constant(sv[["chain_km", "distance_to_channel_m", "P_former_channel"]])
    ols = sm.OLS(sv.H_bed_evrf2019_m, Xp).fit(cov_type="HC3")
    print("\npartial OLS  H_bed ~ chain_km + distance_to_channel_m + P_former_channel  (HC3 robust SE):")
    print(ols.summary().tables[1])
    pchan_coef = ols.params["P_former_channel"]
    pchan_p = ols.pvalues["P_former_channel"]
    print(f"\nP_former_channel coefficient after controlling for chain_km + distance_to_channel: "
          f"{pchan_coef:+.3f} m/unit-P (p={pchan_p:.1e})")
    print(f"  {'survives with expected (negative) sign -- P carries independent morphological info' if pchan_coef < 0 and pchan_p < 0.05 else 'DOES NOT survive as expected -- effect may be redundant with chain_km/distance'}")

    detrend_df = pd.DataFrame([{
        "rho_H_P_global": r_hp, "p_H_P_global": p_hp,
        "rho_H_d_global": r_hd, "p_H_d_global": p_hd,
        "rho_P_d_global": r_pd, "p_P_d_global": p_pd,
        "rho_Hdetrended_P": r_rp, "p_Hdetrended_P": p_rp,
        "rho_Hdetrended_d": r_rd, "p_Hdetrended_d": p_rd,
        "ols_Pchannel_coef": pchan_coef, "ols_Pchannel_p": pchan_p,
        "ols_chainkm_coef": ols.params["chain_km"], "ols_chainkm_p": ols.pvalues["chain_km"],
        "ols_dist_coef": ols.params["distance_to_channel_m"], "ols_dist_p": ols.pvalues["distance_to_channel_m"],
        "ols_r2": ols.rsquared, "n": len(sv),
    }])
    detrend_df.to_csv(CFG.TABLES / "k4b_detrended_partial_correlation.csv", index=False)
    print(f"-> {CFG.TABLES / 'k4b_detrended_partial_correlation.csv'}")

    # ---- 5. buffer-geometry robustness: dilation vs EDT --------------------
    print("\n" + "=" * 70)
    print("buffer-geometry robustness: diamond binary_dilation vs true Euclidean EDT")
    print("=" * 70)
    wb = pd.read_parquet(CFG.ROOT / "outputs/tables/water_body_objects.parquet")
    post = wb[(wb.period == "POST_BREACH") & (wb.wkt != "")].copy()
    SIMPLIFY_M = CELL_M / 2
    BUFFER_CELLS = int(round(BUFFER_M / CELL_M))

    channel_count_edt = np.zeros((H, W), dtype=np.int16)
    obs_count_edt = np.zeros((H, W), dtype=np.int16)
    for date, g in post.groupby("date"):
        all_geoms = [shwkt.loads(w).simplify(SIMPLIFY_M) for w in g.wkt]
        all_geoms = [gm for gm in all_geoms if gm is not None and not gm.is_empty]
        if all_geoms:
            water_mask = features.rasterize([(gm, 1) for gm in all_geoms], out_shape=(H, W),
                                            transform=transform, fill=0, dtype="uint8").astype(bool)
        else:
            water_mask = np.zeros((H, W), bool)
        if water_mask.any():
            dist_edt = distance_transform_edt(~water_mask, sampling=(CELL_M, CELL_M))
            obs_mask = dist_edt <= BUFFER_M
        else:
            obs_mask = water_mask
        obs_count_edt += obs_mask.astype(np.int16)

        main = g[g.is_main_component]
        main_geoms = [shwkt.loads(w).simplify(SIMPLIFY_M) for w in main.wkt]
        main_geoms = [gm for gm in main_geoms if gm is not None and not gm.is_empty]
        if main_geoms:
            chan_mask = features.rasterize([(gm, 1) for gm in main_geoms], out_shape=(H, W),
                                           transform=transform, fill=0, dtype="uint8").astype(bool)
            channel_count_edt += (chan_mask & obs_mask).astype(np.int16)

    with np.errstate(divide="ignore", invalid="ignore"):
        P_edt = np.where(obs_count_edt > 0, channel_count_edt / np.maximum(obs_count_edt, 1), np.nan)

    mask_orig = np.nan_to_num(P) >= 0.5
    mask_edt = np.nan_to_num(P_edt) >= 0.5
    inter = (mask_orig & mask_edt).sum()
    union = (mask_orig | mask_edt).sum()
    jaccard = inter / union if union else np.nan
    print(f"P>=0.5 mask: dilation variant {mask_orig.sum()} cells, EDT variant {mask_edt.sum()} cells, "
          f"Jaccard(dilation, EDT) = {jaccard:.3f}")

    sv["P_former_channel_edt"] = P_edt[rows[valid.values], cols[valid.values]]
    r_hp_edt, p_hp_edt = sstats.spearmanr(sv.H_bed_evrf2019_m, sv.P_former_channel_edt)
    print(f"rho(H, P_edt) = {r_hp_edt:+.3f} (p={p_hp_edt:.1e})  vs  original dilation rho(H,P) = {r_hp:+.3f} (p={p_hp:.1e})")
    verdict_orig = "GO-like (negative, strong)" if r_hp < -0.15 and p_hp < 0.01 else "not GO-like"
    verdict_edt = "GO-like (negative, strong)" if r_hp_edt < -0.15 and p_hp_edt < 0.01 else "not GO-like"
    print(f"verdict class under dilation buffer: {verdict_orig}; under EDT buffer: {verdict_edt} "
          f"-> {'ROBUST to buffer geometry' if verdict_orig == verdict_edt else 'NOT robust -- buffer geometry matters'}")

    robust_df = pd.DataFrame([{
        "jaccard_dilation_vs_edt_P50": jaccard,
        "n_cells_dilation_P50": int(mask_orig.sum()), "n_cells_edt_P50": int(mask_edt.sum()),
        "rho_H_P_dilation": r_hp, "p_H_P_dilation": p_hp,
        "rho_H_P_edt": r_hp_edt, "p_H_P_edt": p_hp_edt,
        "robust": verdict_orig == verdict_edt,
    }])
    robust_df.to_csv(CFG.TABLES / "k4b_dilation_vs_edt_robustness.csv", index=False)
    print(f"-> {CFG.TABLES / 'k4b_dilation_vs_edt_robustness.csv'}")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    rc = reach_df.dropna(subset=["rho_H_P"])
    axes[0].bar(range(len(rc)), rc.rho_H_P, tick_label=[s.replace(", ", "\n") for s in rc.reach_bin],
               color=["#236f8c" if v < 0 else "#c1402a" for v in rc.rho_H_P])
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_ylabel("rho(H_bed, P_former_channel) within reach bin")
    axes[0].set_title("A. reach-stratified rho(H,P)\n(expected negative in every bin)", fontsize=9)
    axes[0].tick_params(axis="x", labelsize=6, rotation=45)

    axes[1].scatter(sv.chain_km, sv.H_bed_evrf2019_m, s=3, alpha=0.15, color="grey", label="soundings")
    axes[1].plot(lo[:, 0], lo[:, 1], color="black", lw=2, label="lowess baseline f(s)")
    axes[1].set_xlabel("chain_km (from Kakhovka dam, +upstream)")
    axes[1].set_ylabel("H_bed, m EVRF2019")
    axes[1].set_title("B. longitudinal baseline used for detrending", fontsize=9)
    axes[1].legend(fontsize=7)

    axes[2].scatter(sv.P_former_channel, sv.H_detrended, s=4, alpha=0.25, color="#3f7d4e")
    axes[2].set_xlabel("P_former_channel")
    axes[2].set_ylabel("H_detrended = H_bed - f(chain_km), m")
    axes[2].set_title(f"C. detrended residual vs P\nrho={r_rp:+.3f} (p={p_rp:.1e})", fontsize=9)
    fig.suptitle("K4b -- reach-stratified / longitudinal-detrend robustness audit of morphology_prior_v1")
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K4b_reach_stratified_and_detrended.png", dpi=150)
    print(f"\n-> {CFG.FIG / 'K4b_reach_stratified_and_detrended.png'}")


if __name__ == "__main__":
    main()
