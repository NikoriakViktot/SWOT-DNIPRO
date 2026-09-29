#!/usr/bin/env python
"""Phase 19 steps 10-15 — channel-only profiles, regime statistics, residual-water offsets.

Consumes outputs/tables/atl13_water_classification.parquet.

The decisive test: does the post-breach longitudinal gradient survive when only
MAIN_CHANNEL observations are used?  The all-water benchmark (~+3.31 cm/km Theil-Sen)
is deliberately NOT used to tune anything here.

A robustness ladder of channel definitions is carried through:
    ALL_WATER            every Sentinel-confirmed water segment
    MAIN_CHANNEL_GEOM    geometry + SWORD topology + connectivity only
    MAIN_CHANNEL         + WSE physically consistent with the local water surface
If the slope is ~2-4 cm/km under all three it is a channel gradient; if it collapses
under MAIN_CHANNEL_GEOM it was bed topography.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import bootstrap_ci, nmad

PROC = CFG.ROOT / "data" / "processed"
RNG = np.random.default_rng(CFG.SEED)
MIN_SEG, MIN_SPAN, MIN_CHAIN_PTS, CHAIN_SEP = 20, 20.0, 3, 5.0
CHANNEL_CLASSES = ("MAIN_CHANNEL",)
RESIDUAL_CLASSES = ("RESIDUAL_POND", "FLOODED_DEPRESSION")
PHYS_OUTLIER_M = 3.0            # a FLAG on |H_residual - H_channel_hat|, never a removal reason


def temporal_conf(lag_days):
    """Sentinel↔ICESat time-separation confidence (operator spec 2026-09-08)."""
    if not np.isfinite(lag_days):
        return "D"
    if lag_days <= 2:
        return "A"
    if lag_days <= 5:
        return "B"
    if lag_days <= 10:
        return "C"
    return "D"


def n_distinct(v, sep=CHAIN_SEP):
    v = np.sort(np.asarray(v))
    keep = [v[0]]
    for x in v[1:]:
        if x - keep[-1] > sep:
            keep.append(x)
    return len(keep)


def fit(g, subset):
    x, y = g.chain_km.to_numpy(float), g.wse_m.to_numpy(float)
    if len(g) < MIN_SEG:
        return None
    span = x.max() - x.min()
    if span < MIN_SPAN or n_distinct(x) < MIN_CHAIN_PTS:
        return None
    ols_c = np.polyfit(x, y, 1)
    ols = float(ols_c[0]) * 100
    ts = stats.theilslopes(y, x, 0.95)
    resid_ols = y - np.polyval(ols_c, x)
    resid_ts = y - (ts[0] * x + ts[1])
    return {"subset": subset, "n_segments": len(g),
            "n_beams": int(g.beam.nunique()) if "beam" in g else -1,
            "n_tracks": int(g.rgt.nunique()) if "rgt" in g else -1,
            "n_chain_bins": n_distinct(x),
            "span_km": span, "chain_min_km": x.min(), "chain_max_km": x.max(),
            "slope_ols_cm_km": ols, "slope_theilsen_cm_km": float(ts[0]) * 100,
            "ts_lo_cm_km": float(ts[2]) * 100, "ts_hi_cm_km": float(ts[3]) * 100,
            "median_wse_m": float(np.median(y)), "nmad_m": nmad(y),
            "p05_m": float(np.percentile(y, 5)), "p95_m": float(np.percentile(y, 95)),
            "p95_p05_m": float(np.percentile(y, 95) - np.percentile(y, 5)),
            "resid_rms_ols_m": float(np.sqrt(np.mean(resid_ols ** 2))),
            "resid_nmad_ts_m": nmad(resid_ts)}


def perm_test(a, b, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = np.median(b) - np.median(a)
    pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(np.median(pool[na:]) - np.median(pool[:na])) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def boot_diff(a, b, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [np.median(RNG.choice(b, len(b), True)) - np.median(RNG.choice(a, len(a), True))
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def regime_stats(pre, post, draw, label):
    rows = []
    for est in ["slope_ols_cm_km", "slope_theilsen_cm_km"]:
        if len(pre) < 3 or len(post) < 3:
            continue
        obs, p = perm_test(pre[est], post[est])
        lo, hi = boot_diff(pre[est], post[est])
        kpre, kpost = int((pre[est] > 0).sum()), int((post[est] > 0).sum())
        mw = stats.mannwhitneyu(pre[est], post[est], alternative="two-sided")
        blo, bhi = bootstrap_ci(post[est], seed=CFG.SEED)
        plo, phi = bootstrap_ci(pre[est], seed=CFG.SEED)
        rows.append({
            "comparison": label, "estimator": "OLS" if "ols" in est else "Theil-Sen",
            "n_pre": len(pre), "n_draw": len(draw), "n_post": len(post),
            "median_pre_cm_km": float(np.median(pre[est])), "pre_ci_lo": plo, "pre_ci_hi": phi,
            "median_draw_cm_km": float(np.median(draw[est])) if len(draw) else np.nan,
            "median_post_cm_km": float(np.median(post[est])), "post_ci_lo": blo, "post_ci_hi": bhi,
            "nmad_pre": nmad(pre[est]), "nmad_post": nmad(post[est]),
            "diff_cm_km": obs, "diff_ci_lo": lo, "diff_ci_hi": hi,
            "permutation_p": p, "mannwhitney_p": float(mw.pvalue),
            "pre_positive": f"{kpre}/{len(pre)}",
            "pre_sign_p": float(stats.binomtest(kpre, len(pre), 0.5).pvalue),
            "post_positive": f"{kpost}/{len(post)}",
            "post_sign_p": float(stats.binomtest(kpost, len(post), 0.5).pvalue),
            "cliffs_delta": float(2 * mw.statistic / (len(pre) * len(post)) - 1),
        })
    return rows


def main() -> None:
    cl = pd.read_parquet(CFG.TABLES / "atl13_water_classification.parquet")
    cl["date"] = pd.to_datetime(cl["date"])

    # ---- pre-breach: whole AOI is one pool, no imagery needed --------------
    from swot_dnipro.vertical import sample_grid
    from swot_dnipro import sword as SW
    ch = pd.read_parquet(PROC / "profiles" / "sword_dnipro_channel.parquet")
    pre = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet")
    pre["dt"] = pd.to_datetime(pre["time"], utc=True, errors="coerce").dt.tz_localize(None)
    pre = pre[pre.dt < CFG.BREACH_DATE].dropna(subset=["lat", "lon", "h_wgs84_m", "dt"])
    z = sample_grid(CFG.EGG2015_TIF, pre.lon.values, pre.lat.values)
    pre["wse_m"] = pre.h_wgs84_m.values + CFG.free2mean(pre.lat.values) - z
    c, d, _, _ = SW.assign_chainage(pre.lon, pre.lat, ch)
    pre["chain_km"], pre["dist_to_sword_km"] = c, d
    pre["date"] = pre.dt.dt.normalize()
    pre = pre.dropna(subset=["chain_km", "wse_m"])
    pre_ch = pre[pre.dist_to_sword_km < 1.5]

    # ---- per-date profiles ------------------------------------------------
    rows = []
    for d_, g in pre.groupby("date"):
        r = fit(g, "PRE_ALL_WATER")
        if r: rows.append({"date": d_, "period": "PRE_BREACH", **r})
    for d_, g in pre_ch.groupby("date"):
        r = fit(g, "PRE_CHANNEL")
        if r: rows.append({"date": d_, "period": "PRE_BREACH", **r})
    for (d_, per), g in cl.groupby(["date", "period"]):
        gw = g[g.is_water == 1.0]
        for sub, sel in [("ALL_WATER", gw),
                         ("MAIN_CHANNEL_GEOM", g[g.water_class_geom.isin(CHANNEL_CLASSES)]),
                         ("MAIN_CHANNEL", g[g.water_class.isin(CHANNEL_CLASSES)])]:
            r = fit(sel, sub)
            if r:
                lag = float(sel.abs_lag_days.median()) if len(sel) else np.nan
                r["median_lag_days"] = lag
                r["temporal_conf"] = temporal_conf(lag)
                rows.append({"date": d_, "period": per, **r})
    prof = pd.DataFrame(rows)
    prof.to_csv(CFG.TABLES / "channel_only_profiles.csv", index=False)
    prof.to_csv(CFG.FIGDATA / "P19_profiles.csv", index=False)

    # ---- all-water vs channel-only per date ------------------------------
    piv = prof.pivot_table(index=["date", "period"], columns="subset",
                           values="slope_theilsen_cm_km")
    span = prof.pivot_table(index=["date", "period"], columns="subset", values="span_km")
    nseg = prof.pivot_table(index=["date", "period"], columns="subset", values="n_segments")
    cmp_ = pd.DataFrame({"date": [i[0] for i in piv.index],
                         "period": [i[1] for i in piv.index]})
    for s in ("ALL_WATER", "MAIN_CHANNEL_GEOM", "MAIN_CHANNEL"):
        cmp_[f"slope_{s.lower()}_cm_km"] = piv[s].values if s in piv else np.nan
        cmp_[f"n_{s.lower()}"] = nseg[s].values if s in nseg else np.nan
        cmp_[f"span_{s.lower()}_km"] = span[s].values if s in span else np.nan
    cmp_["slope_all_water_cm_km"] = cmp_.get("slope_all_water_cm_km")
    cmp_["slope_channel_cm_km"] = cmp_["slope_main_channel_cm_km"]
    cmp_["delta_slope_cm_km"] = cmp_["slope_channel_cm_km"] - cmp_["slope_all_water_cm_km"]
    cmp_ = cmp_.dropna(subset=["slope_all_water_cm_km", "slope_channel_cm_km"], how="all")
    cmp_.to_csv(CFG.TABLES / "allwater_vs_channel_slopes.csv", index=False)
    cmp_.to_csv(CFG.FIGDATA / "P19_allwater_vs_channel.csv", index=False)

    # ---- regime statistics (unit = one date) ----------------------------
    def sub(df, per, s):
        return df[(df.period == per) & (df.subset == s)]
    draw = sub(prof, "BREACH_DRAWDOWN", "MAIN_CHANNEL")
    stat_rows = []
    stat_rows += regime_stats(sub(prof, "PRE_BREACH", "PRE_CHANNEL"),
                              sub(prof, "POST_BREACH", "MAIN_CHANNEL"), draw,
                              "POST MAIN_CHANNEL vs PRE channel-restricted")
    stat_rows += regime_stats(sub(prof, "PRE_BREACH", "PRE_CHANNEL"),
                              sub(prof, "POST_BREACH", "MAIN_CHANNEL_GEOM"), draw,
                              "POST MAIN_CHANNEL_GEOM vs PRE channel-restricted")
    stat_rows += regime_stats(sub(prof, "PRE_BREACH", "PRE_ALL_WATER"),
                              sub(prof, "POST_BREACH", "ALL_WATER"), draw,
                              "POST ALL_WATER vs PRE all-water")
    st = pd.DataFrame(stat_rows)
    st.to_csv(CFG.TABLES / "reservoir_to_river_statistics.csv", index=False)
    st.to_csv(CFG.FIGDATA / "P19_statistics.csv", index=False)

    # ---- residual water vs reconstructed channel (steps 14-15) ----------
    # Operator spec 2026-09-08: NEVER exclude an observation on |delta_H| alone.
    #   * SUSPECT_NONWATER  — independent geometric / temporal evidence it is not a
    #     genuine, channel-relevant water body (distance, component size, single
    #     beam, no persistence, edge sliver, WSE gate, stale imagery).
    #   * PHYSICAL_OUTLIER  — |H_residual - H_channel_hat| > 3 m.  A FLAG ONLY.
    # Every flagged row is kept in residual_water_offsets.csv; three headline
    # statistics are reported (A all / B high-confidence / C B minus outliers).

    # per (date, water_body) component context
    rb = cl[cl.water_class.isin(RESIDUAL_CLASSES)].copy()
    body = (rb.groupby(["date", "water_body_id"])
              .agg(n_seg_body=("lat", "size"), n_beam_body=("gt", "nunique")).reset_index())
    persist_dates = rb.groupby("persist_id").date.nunique().rename("persist_n_dates")

    res_rows = []
    for d_, g in cl.groupby("date"):
        gc = g[g.water_class.isin(CHANNEL_CLASSES)]
        gr = g[g.water_class.isin(RESIDUAL_CLASSES)]
        if len(gc) < MIN_SEG or len(gr) == 0:
            continue
        x, y = gc.chain_km.to_numpy(float), gc.wse_m.to_numpy(float)
        if x.max() - x.min() < MIN_SPAN:
            continue
        ts = stats.theilslopes(y, x, 0.95)
        slope, icpt = float(ts[0]), float(ts[1])
        for r in gr.itertuples():
            if not (x.min() - 20 <= r.chain_km <= x.max() + 20):
                continue
            h_hat = slope * r.chain_km + icpt
            dH = r.wse_m - h_hat
            nb = body[(body.date == d_) & (body.water_body_id == r.water_body_id)]
            n_seg_body = int(nb.n_seg_body.iloc[0]) if len(nb) else 1
            n_beam_body = int(nb.n_beam_body.iloc[0]) if len(nb) else 1
            n_dates = int(persist_dates.get(r.persist_id, 1))
            reasons = []
            if r.dist_to_sword_km > 4.0:
                reasons.append("far_from_channel")
            if n_seg_body < 8:
                reasons.append("tiny_component")
            if n_beam_body <= 1:
                reasons.append("single_beam")
            if n_dates <= 1:
                reasons.append("not_persistent")
            if np.isfinite(r.dist_to_water_edge_km) and r.dist_to_water_edge_km < 0.06:
                reasons.append("edge_sliver")
            if not bool(r.wse_consistent):
                reasons.append("wse_gate_fail")
            if np.isfinite(r.abs_lag_days) and r.abs_lag_days > 5:
                reasons.append("stale_imagery")
            # SUSPECT if the body is off the channel AND either its height is
            # physically inconsistent with the local water surface (WSE gate), or
            # it is a small / single-beam / non-persistent detection; or >=4 signs.
            far = "far_from_channel" in reasons
            suspect = ((far and "wse_gate_fail" in reasons)
                       or (far and ("single_beam" in reasons or "tiny_component" in reasons)
                           and "not_persistent" in reasons)
                       or len(reasons) >= 4)
            res_rows.append({
                "date": d_, "water_class": r.water_class,
                "persist_id": r.persist_id, "water_body_id": r.water_body_id,
                "chain_km": r.chain_km, "dist_to_sword_km": r.dist_to_sword_km,
                "dist_to_water_edge_km": r.dist_to_water_edge_km,
                "n_seg_body": n_seg_body, "n_beam_body": n_beam_body,
                "persist_n_dates": n_dates, "lat": r.lat, "lon": r.lon,
                "H_residual_m": r.wse_m, "H_channel_hat_m": h_hat,
                "delta_H_residual_m": dH, "channel_slope_cm_km": slope * 100,
                "abs_lag_days": r.abs_lag_days,
                "temporal_conf": temporal_conf(r.abs_lag_days),
                "wse_consistent": bool(r.wse_consistent),
                "suspect_nonwater": bool(suspect),
                "suspect_reasons": "|".join(reasons),
                "physical_outlier": bool(abs(dH) > PHYS_OUTLIER_M),
                "image_name": r.image_name, "delta_time_days": r.delta_time_days})
    res = pd.DataFrame(res_rows)
    per_body = pd.DataFrame()
    if len(res):
        res.to_csv(CFG.TABLES / "residual_water_offsets.csv", index=False)
        res.to_csv(CFG.FIGDATA / "P19_residual_offsets.csv", index=False)

        def _stat(v):
            v = np.asarray(v, float)
            return dict(n=len(v), median_m=float(np.median(v)) if len(v) else np.nan,
                        nmad_m=float(nmad(v)) if len(v) else np.nan,
                        frac_above=float((v > 0).mean()) if len(v) else np.nan,
                        p05=float(np.percentile(v, 5)) if len(v) else np.nan,
                        p95=float(np.percentile(v, 95)) if len(v) else np.nan,
                        min=float(v.min()) if len(v) else np.nan,
                        max=float(v.max()) if len(v) else np.nan)
        hc = res[~res.suspect_nonwater]
        hc_no = hc[~hc.physical_outlier]
        summ = pd.DataFrame([
            {"population": "A_all_classified_residual", **_stat(res.delta_H_residual_m)},
            {"population": "B_high_confidence", **_stat(hc.delta_H_residual_m)},
            {"population": "C_high_confidence_no_physical_outlier",
             **_stat(hc_no.delta_H_residual_m)}])
        summ.to_csv(CFG.TABLES / "residual_water_offset_summary.csv", index=False)
        summ.to_csv(CFG.FIGDATA / "P19_residual_offset_summary.csv", index=False)

        per_body = (res.groupby(["persist_id", "date"])
                    .agg(n=("delta_H_residual_m", "size"),
                         median_delta_m=("delta_H_residual_m", "median"),
                         nmad_m=("delta_H_residual_m", nmad),
                         suspect=("suspect_nonwater", "max"),
                         chain_km=("chain_km", "median"),
                         lat=("lat", "median"), lon=("lon", "median")).reset_index())
        per_body.to_csv(CFG.FIGDATA / "P19_residual_per_body.csv", index=False)
        # cross-date persistence: same location, >1 date, sign preserved
        pers = (per_body.groupby("persist_id")
                .agg(n_dates=("date", "nunique"),
                     median_delta_m=("median_delta_m", "median"),
                     min_delta_m=("median_delta_m", "min"),
                     max_delta_m=("median_delta_m", "max"),
                     any_suspect=("suspect", "max"),
                     chain_km=("chain_km", "median")).reset_index())
        pers["sign_stable"] = (np.sign(pers.min_delta_m) == np.sign(pers.max_delta_m))
        pers.to_csv(CFG.TABLES / "residual_water_persistence.csv", index=False)

    # ---- reservoir_to_river_statistics headline print -------------------
    pd.set_option("display.width", 220)
    print("=== per-date profiles (unit = one date) ===")
    print(prof.groupby(["period", "subset"]).agg(
        n_dates=("date", "nunique"),
        median_ts=("slope_theilsen_cm_km", "median"),
        median_ols=("slope_ols_cm_km", "median"),
        frac_pos=("slope_theilsen_cm_km", lambda x: round((x > 0).mean(), 3)),
        median_span=("span_km", "median")).to_string())

    if len(cmp_):
        print(f"\n=== ALL_WATER vs MAIN_CHANNEL per date (n={len(cmp_)}) ===")
        show = cmp_[cmp_.period != "PRE_BREACH"]
        print(show[["date", "period", "slope_all_water_cm_km", "slope_main_channel_geom_cm_km",
                    "slope_channel_cm_km", "delta_slope_cm_km"]].round(3).to_string(index=False))
        print(f"\n  median all-water   {cmp_.slope_all_water_cm_km.median():+.3f}")
        print(f"  median geom-only   {cmp_.slope_main_channel_geom_cm_km.median():+.3f}")
        print(f"  median channel     {cmp_.slope_channel_cm_km.median():+.3f} cm/km "
              f"(Δ {cmp_.delta_slope_cm_km.median():+.3f})")

    if len(st):
        print("\n=== regime statistics ===")
        for r in st.itertuples():
            print(f"\n--- {r.comparison} [{r.estimator}] ---")
            print(f"  PRE {r.median_pre_cm_km:+.3f} [{r.pre_ci_lo:+.3f},{r.pre_ci_hi:+.3f}] n={r.n_pre}"
                  f"   DRAW {r.median_draw_cm_km:+.3f}"
                  f"   POST {r.median_post_cm_km:+.3f} [{r.post_ci_lo:+.3f},{r.post_ci_hi:+.3f}] n={r.n_post}")
            print(f"  diff {r.diff_cm_km:+.3f} CI [{r.diff_ci_lo:+.3f},{r.diff_ci_hi:+.3f}]  "
                  f"perm p={r.permutation_p:.5f}  MW p={r.mannwhitney_p:.5f}  d={r.cliffs_delta:+.2f}")
            print(f"  sign PRE {r.pre_positive} (p={r.pre_sign_p:.3f})  "
                  f"POST {r.post_positive} (p={r.post_sign_p:.4f})")

    if len(res):
        print(f"\n=== residual water vs reconstructed channel "
              f"(n={len(res)} segs, {res.date.nunique()} dates, "
              f"{res.persist_id.nunique()} locations) ===")
        sm = pd.read_csv(CFG.TABLES / "residual_water_offset_summary.csv")
        for r in sm.itertuples():
            print(f"  {r.population:38s} n={r.n:5d}  median {r.median_m:+.3f} m  "
                  f"NMAD {r.nmad_m:.3f}  above {r.frac_above*100:4.1f}%  "
                  f"p05 {r.p05:+.2f}  p95 {r.p95:+.2f}  range [{r.min:+.2f},{r.max:+.2f}]")
        nsus = int(res.suspect_nonwater.sum())
        nout = int(res.physical_outlier.sum())
        print(f"  flagged: SUSPECT_NONWATER {nsus}/{len(res)}  "
              f"PHYSICAL_OUTLIER {nout}/{len(res)}  (kept in the table, not deleted)")
        if nsus:
            print("  top SUSPECT_NONWATER reasons: "
                  + ", ".join(f"{k}×{v}" for k, v in
                              res[res.suspect_nonwater].suspect_reasons.str.split("|")
                              .explode().value_counts().head(5).items()))
        v = res.delta_H_residual_m
        print(res.groupby("water_class").agg(n=("delta_H_residual_m", "size"),
              median_m=("delta_H_residual_m", "median"),
              nmad_m=("delta_H_residual_m", nmad)).to_string())
        if len(per_body):
            pers = pd.read_csv(CFG.TABLES / "residual_water_persistence.csv")
            multi = pers[pers.n_dates > 1]
            print(f"\n  water-body locations seen on >1 date: {len(multi)}/{len(pers)}; "
                  f"sign-stable across dates: {int(multi.sign_stable.sum())}/{len(multi)}")


if __name__ == "__main__":
    main()
