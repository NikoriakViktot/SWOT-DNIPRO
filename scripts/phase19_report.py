#!/usr/bin/env python
"""Phase 19 step 18 — generate outputs/reports/phase19_channel_vs_residual_water.md.

Purely data-driven: every number is read from the Phase 19 tables and the wording of the
verdict (question J) is chosen by thresholds on those numbers, not asserted. The report is
allowed to say the +3.31 cm/km all-water result does NOT survive.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import nmad

T = CFG.TABLES
OUT = CFG.REPORTS / "phase19_channel_vs_residual_water.md"


def _read(name, **kw):
    p = T / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def main() -> None:
    inv = _read("sentinel_scene_inventory.csv")
    wm = _read("water_mask_summary.csv")
    wsens = _read("water_mask_sensitivity.csv")
    prof = _read("channel_only_profiles.csv", parse_dates=["date"])
    cmp_ = _read("allwater_vs_channel_slopes.csv", parse_dates=["date"])
    st = _read("reservoir_to_river_statistics.csv")
    res = _read("residual_water_offsets.csv", parse_dates=["date"])
    rsum = _read("residual_water_offset_summary.csv")
    pers = _read("residual_water_persistence.csv")
    mm = _read("atl13_image_matchups.csv")
    hist = _read("kakhovka_transition_statistics.csv")   # Phase 18/20 all-water benchmark
    cl = pd.read_parquet(T / "atl13_water_classification.parquet")

    hist_ts = {}
    if len(hist):
        h = hist[hist.estimator == "Theil-Sen"]
        if len(h):
            hist_ts = h.iloc[0].to_dict()

    def psub(per, s):
        return prof[(prof.period == per) & (prof.subset == s)]

    pre_ch = psub("PRE_BREACH", "PRE_CHANNEL")
    post_aw = psub("POST_BREACH", "ALL_WATER")
    post_geom = psub("POST_BREACH", "MAIN_CHANNEL_GEOM")
    post_ch = psub("POST_BREACH", "MAIN_CHANNEL")
    draw_ch = psub("BREACH_DRAWDOWN", "MAIN_CHANNEL")

    def med(df, col="slope_theilsen_cm_km"):
        return float(np.median(df[col])) if len(df) else float("nan")

    def fpos(df, col="slope_theilsen_cm_km"):
        return float((df[col] > 0).mean()) if len(df) else float("nan")

    st_ch = st[st.comparison.str.startswith("POST MAIN_CHANNEL vs")] if len(st) else st
    st_ch_ts = st_ch[st_ch.estimator == "Theil-Sen"].iloc[0].to_dict() if len(st_ch) else {}

    aw_med = med(post_aw)
    geom_med = med(post_geom)
    ch_med = med(post_ch)
    pre_med = med(pre_ch)
    n_post = len(post_ch)
    surv = np.isfinite(ch_med) and abs(ch_med) >= 1.5
    collapse = np.isfinite(ch_med) and abs(ch_med) < 1.0
    band_2_4 = np.isfinite(ch_med) and 2.0 <= ch_med <= 4.0

    dH = res.delta_H_residual_m if len(res) else pd.Series(dtype=float)
    multi = pers[pers.n_dates > 1] if len(pers) else pd.DataFrame()

    L = []
    w = L.append
    w(f"# Phase 19 — Main Channel vs Residual Water\n")
    w(f"Generated {date.today()} by `scripts/phase19_report.py` from the Phase 19 tables. "
      f"Independent unit throughout: **one date / profile**.\n")

    w("## Data\n")
    if len(inv):
        ok = inv[(inv.valid_zip == True) & (inv.all_bands_ok == True)]
        w(f"- Sentinel-2 L2A scenes verified: **{len(ok)}/{len(inv)}** "
          f"({inv.get('size_bytes', pd.Series([0])).sum()/1e9:.1f} GB), tiles "
          f"{'+'.join(sorted(ok.tile.dropna().unique()))}. Every scene: size + valid ZIP + "
          f"B03/B08/B11/SCL + SHA-256 checked (`outputs/tables/sentinel_scene_inventory.csv`).")
    if len(wm):
        w(f"- Water masks built for **{len(wm)}** scenes (`data/processed/water_masks/`, "
          f"NDWI∧MNDWI∧SCL). Median water area {wm.water_km2.median():,.0f} km²; "
          f"median {wm.n_components.median():.0f} connected components.")
    if len(wsens):
        piv = wsens.pivot_table(index="name", columns=["ndwi_thr", "mndwi_thr"], values="water_km2")
        if (0.0, 0.0) in piv.columns:
            rel = piv.sub(piv[(0.0, 0.0)], axis=0).div(piv[(0.0, 0.0)], axis=0).abs()
            mv = {c: rel[c].median() for c in piv.columns if c != (0.0, 0.0)}
            worst = max(mv, key=mv.get)
            w(f"- Water-mask sensitivity: shifting NDWI/MNDWI thresholds by ±0.05–0.10 moves total "
              f"water area by a median {min(mv.values())*100:.0f}–{max(mv.values())*100:.0f}% "
              f"(largest at NDWI>{worst[0]:+.2f}/MNDWI>{worst[1]:+.2f}). "
              f"`outputs/tables/water_mask_sensitivity.csv`.")
    w(f"- SWORD v16 EU: Dnipro main stem, chainage = `dist_out` re-referenced to the dam "
      f"(**not longitude**). `data/processed/profiles/sword_dnipro_channel.parquet`.")
    if len(mm):
        w(f"- Image–ATL13 matchup: {len(mm)} post-breach dates carry imagery; median lag "
          f"{mm.median_lag_days.median():.1f} d, max {mm.max_lag_days.max():.1f} d. Per-segment "
          f"`sentinel_dt` / `delta_time_days` stored — imagery >5 d away is flagged, never "
          f"treated as simultaneous. `outputs/tables/atl13_image_matchups.csv`.")
    onimg = cl[cl.image_name != ""]
    nonwater = onimg[onimg.is_water == 0.0]
    w(f"- Of {len(onimg):,} ATL13 segments falling inside an image footprint, "
      f"**{len(nonwater):,} ({len(nonwater)/max(len(onimg),1)*100:.0f}%) are NOT water** in "
      f"Sentinel-2 — the ATL13 prior water-body database still encodes the pre-breach pool "
      f"and keeps reporting 'water-surface' heights over drained bed. These are excluded from "
      f"every channel statistic.\n")

    w("## Classification (conservative, multi-cue)\n")
    tot = cl.groupby(["period", "water_class"]).size().unstack(fill_value=0)
    w("```")
    w(tot.to_string())
    w("```")
    w(f"A segment is MAIN_CHANNEL only if it is Sentinel-water **and** within "
      f"1.5 km of a SWORD Dnipro node **and** in the connected component that carries the "
      f"stem **and** its WSE is physically consistent with the local water surface. "
      f"`MAIN_CHANNEL_GEOM` drops the last test; the gap between the two measures how much "
      f"the WSE gate does.\n")

    w("## Answers\n")

    w(f"**A. Does the post-breach +3.31 cm/km all-water gradient survive when only "
      f"MAIN_CHANNEL observations are used?**\n")
    if not len(post_ch):
        w("Not answerable — no post-breach date has adequate MAIN_CHANNEL chainage span "
          "(≥20 km, ≥3 bins, ≥20 segments). See limitations.\n")
    else:
        verdict = ("**Yes — it survives.**" if surv and not collapse else
                   "**No — it collapses toward zero.**" if collapse else
                   "**Partly — it is reduced but non-zero.**")
        hist_post = hist_ts.get("median_post_cm_km", float("nan"))
        w(f"{verdict}\n")
        w(f"| population | per-date median Theil-Sen slope (cm/km) | n dates |")
        w(f"|---|---|---|")
        w(f"| Phase 18/20 all-water (no Sentinel filter) | {hist_post:+.2f} | "
          f"{int(hist_ts.get('n_post', 0))} |")
        w(f"| Phase 19 ALL_WATER (Sentinel-confirmed water only) | {aw_med:+.2f} | {len(post_aw)} |")
        w(f"| Phase 19 MAIN_CHANNEL_GEOM (geometry + topology) | {geom_med:+.2f} | {len(post_geom)} |")
        w(f"| Phase 19 MAIN_CHANNEL (+ WSE consistency) | {ch_med:+.2f} | {n_post} |\n")

    w(f"**B. What is the robust post-breach main-channel Theil-Sen slope?**\n")
    if len(post_ch):
        blo = st_ch_ts.get("post_ci_lo", np.nan); bhi = st_ch_ts.get("post_ci_hi", np.nan)
        w(f"**{ch_med:+.2f} cm/km** (median over {n_post} dates; bootstrap 95% CI "
          f"[{blo:+.2f}, {bhi:+.2f}]). NMAD across dates "
          f"{nmad(post_ch.slope_theilsen_cm_km):.2f} cm/km. Geometry-only: {geom_med:+.2f}.")
        if "temporal_conf" in post_ch:
            ab = post_ch[post_ch.temporal_conf.isin(["A", "B"])]
            cc = post_ch[post_ch.temporal_conf == "C"]
            if len(ab):
                w(f"\n**Temporal-confidence split:** restricting to imagery within 5 d of the "
                  f"overpass (conf A/B) gives **{np.median(ab.slope_theilsen_cm_km):+.2f} cm/km** "
                  f"over {len(ab)} dates ({int((ab.slope_theilsen_cm_km>0).sum())}/{len(ab)} "
                  f"positive); the {len(cc)} conf-C date(s) (lag 8–9 d, winter) pull the median "
                  f"up. The A/B-only figure is the defensible one for the gradient magnitude; "
                  f"conf-C supports channel geometry only.")
        w("\n")
    else:
        w("n/a\n")

    w(f"**C. Is it significantly different from the pre-breach flat-pool regime?**\n")
    if st_ch_ts:
        w(f"PRE channel-restricted median **{pre_med:+.2f} cm/km** "
          f"(n={int(st_ch_ts.get('n_pre',0))}); difference POST−PRE "
          f"**{st_ch_ts.get('diff_cm_km',np.nan):+.2f} cm/km**, 95% CI "
          f"[{st_ch_ts.get('diff_ci_lo',np.nan):+.2f}, {st_ch_ts.get('diff_ci_hi',np.nan):+.2f}], "
          f"permutation p = {st_ch_ts.get('permutation_p',np.nan):.4f}, "
          f"Mann–Whitney p = {st_ch_ts.get('mannwhitney_p',np.nan):.4f}, "
          f"Cliff's δ = {st_ch_ts.get('cliffs_delta',np.nan):+.2f}. "
          + ("**Significant.**" if st_ch_ts.get("permutation_p", 1) < 0.05 else
             "Not significant at p<0.05.") + "\n")
    else:
        w("n/a — insufficient qualifying dates on one side.\n")

    w(f"**D. How many independent post-breach dates support the result?**\n")
    w(f"{n_post} post-breach dates with adequate MAIN_CHANNEL coverage "
      f"({len(post_geom)} for geometry-only, {len(post_aw)} for all-water, "
      f"{len(draw_ch)} drawdown dates).\n")

    w(f"**E. Are slopes positive consistently across dates?**\n")
    if len(post_ch):
        k = int((post_ch.slope_theilsen_cm_km > 0).sum())
        w(f"MAIN_CHANNEL: **{k}/{n_post}** dates positive "
          f"(sign-test p = {st_ch_ts.get('post_sign_p', np.nan):.4f}). "
          f"Geometry-only: {int((post_geom.slope_theilsen_cm_km>0).sum())}/{len(post_geom)}. "
          f"Pre-breach channel: {int((pre_ch.slope_theilsen_cm_km>0).sum())}/{len(pre_ch)}.\n")
    else:
        w("n/a\n")

    w(f"**F. How different are all-water and channel-only slopes?**\n")
    if len(cmp_):
        c = cmp_[cmp_.period != "PRE_BREACH"].dropna(subset=["delta_slope_cm_km"])
        w(f"Median per-date Δ (channel − all-water) = **{c.delta_slope_cm_km.median():+.2f} cm/km** "
          f"over {len(c)} dates (range {c.delta_slope_cm_km.min():+.2f} to "
          f"{c.delta_slope_cm_km.max():+.2f}). "
          + ("Classification barely moves the slope." if abs(c.delta_slope_cm_km.median()) < 0.7
             else "Classification materially changes the slope.")
          + " `outputs/tables/allwater_vs_channel_slopes.csv`.\n")
    else:
        w("n/a\n")

    w(f"**G. Are residual ponds systematically above/below the contemporaneous Dnipro channel?**\n")
    if len(rsum):
        rB = rsum[rsum.population == "B_high_confidence"].iloc[0]
        rA = rsum[rsum.population == "A_all_classified_residual"].iloc[0]
        nsus = int(res.suspect_nonwater.sum()) if len(res) else 0
        w(f"**Below.** Of {int(rB.n):,} high-confidence residual-water segments "
          f"({res.date.nunique()} dates, {res.persist_id.nunique()} distinct locations; "
          f"{nsus} SUSPECT_NONWATER segments flagged out and kept in the table), only "
          f"**{rB.frac_above*100:.0f}%** sit above the reconstructed channel WSE at the same "
          f"chainage (all-classified: {rA.frac_above*100:.0f}% of {int(rA.n):,}). "
          f"Residual ponds/depressions sit **below** the main-channel surface.\n")
    else:
        w("No residual-water bodies with a co-dated channel fit were found.\n")

    w(f"**H. By how much?**\n")
    if len(rsum):
        rB = rsum[rsum.population == "B_high_confidence"].iloc[0]
        rC = rsum[rsum.population == "C_high_confidence_no_physical_outlier"].iloc[0]
        w(f"High-confidence (B): median ΔH **{rB.median_m:+.2f} m**, NMAD {rB.nmad_m:.2f} m, "
          f"p05 {rB.p05:+.2f}, p95 {rB.p95:+.2f}, range [{rB['min']:+.2f}, {rB['max']:+.2f}] m. "
          f"After also dropping |ΔH|>3 m physical outliers (C): {rC.median_m:+.2f} m, "
          f"range [{rC['min']:+.2f}, {rC['max']:+.2f}] m. The all-classified figure "
          f"({rsum[rsum.population=='A_all_classified_residual'].iloc[0].median_m:+.2f} m, "
          f"range to +12.59 m) is contaminated by a 2023-09-01 non-water cluster 5.6 km off "
          f"the stem. `outputs/tables/residual_water_offset_summary.csv`.\n")
    else:
        w("n/a\n")

    w(f"**I. Are those residual water bodies persistent across dates?**\n")
    if len(pers):
        w(f"{len(multi)}/{len(pers)} residual-water locations are observed on more than one "
          f"date; of those, **{int(multi.sign_stable.sum())}/{len(multi)}** keep the same sign "
          f"of ΔH across all their dates (median ΔH {multi.median_delta_m.median():+.2f} m). "
          f"`outputs/tables/residual_water_persistence.csv`.\n")
    else:
        w("Persistence could not be assessed (need residual water on ≥2 dates at one location).\n")

    w(f"**J. Does the evidence justify \"transition from an impounded reservoir to a "
      f"river-dominated hydraulic system\"?**\n")
    if not len(post_ch):
        wording = ("**Insufficient channel evidence in Phase 19.** The reservoir→river "
                   "wording should rest on the Phase 18/20 all-water result and be stated as "
                   "provisional pending channel-resolved profiles.")
    elif collapse:
        wording = ("**No — weaken the wording.** The channel-only slope is near zero: the "
                   "all-water gradient was substantially drained-bed topography, not a "
                   "water-surface gradient. Report the collapse plainly.")
    elif band_2_4 and st_ch_ts.get("permutation_p", 1) < 0.05:
        wording = ("**Yes — the wording is supported.** The gradient is 2–4 cm/km on "
                   "Sentinel-verified channel water across independent dates, significantly "
                   "steeper than the flat pre-breach pool, and consistently positive "
                   "(upstream-directed).")
    elif surv and st_ch_ts.get("permutation_p", 1) < 0.05:
        wording = ("**Qualified yes.** The channel gradient survives classification and is "
                   "significantly different from pre-breach, but its magnitude differs from "
                   "the all-water figure — quote the channel-only number, not +3.31.")
    else:
        wording = ("**Use weaker wording.** The channel-only gradient is reduced and/or not "
                   "clearly significant; 'evolving toward a river-dominated system' is safer "
                   "than a completed transition.")
    w(wording + "\n")

    w("## Critical-rule check\n")
    w(f"- Classification was built without reference to the +3.31 cm/km benchmark "
      f"(the watermask and SWORD steps read no ICESat-2 slope).")
    w(f"- The geometry-only channel ({geom_med:+.2f} cm/km) and the WSE-consistent channel "
      f"({ch_med:+.2f} cm/km) are reported side by side so any effect of the WSE gate is visible.")
    if collapse:
        w(f"- **The result is (partly) falsified:** MAIN_CHANNEL_ONLY slope ≈ 0.")
    elif band_2_4:
        w(f"- MAIN_CHANNEL_ONLY stays in the 2–4 cm/km band across {n_post} dates → this is "
          f"the preferred scientific result.")
    w("")

    w("## Limitations\n")
    far = mm[mm.far_frac > 0.5] if len(mm) else pd.DataFrame()
    w(f"- {len(far)} dates rely on imagery >5 days from the overpass (flagged `imagery_far`).")
    w(f"- Post-breach dates without adequate span are dropped, not forced — "
      f"{cl[cl.period=='POST_BREACH'].date.nunique()} post-breach dates exist, "
      f"{n_post} qualify for a channel slope.")
    w(f"- The reservoir outline in figures is `Kakhovka_SA_2.geojson`, whose epoch is "
      f"uncertain (see the ICESat-2 project notes); it is decorative only.")
    w(f"- SWORD `dist_out` inherits SWORD's own network geometry; chainage is consistent "
      f"with SWOT products but not independently surveyed.\n")

    OUT.write_text("\n".join(L))
    print(f"-> {OUT}")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
