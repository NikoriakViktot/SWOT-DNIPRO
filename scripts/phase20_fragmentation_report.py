#!/usr/bin/env python
"""Phase 20 STEP 4-10 — pre/post fragmentation statistics, ATL13 linkage, indicators,
figures, report. Consumes the Phase 20 STEP 1-3 tables.

Primary unit throughout: one date / mask.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date as _date
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro.plotting.style import nmad, save, use_style

from phase20_water_objects import date_mosaic, sword_nodes

T = CFG.TABLES
FD = CFG.FIGDATA
FIG = CFG.FIG
BREACH = pd.Timestamp("2023-06-06")
RNG = np.random.default_rng(CFG.SEED)


def perm_test(a, b, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    obs = np.median(b) - np.median(a)
    pool = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(np.median(pool[na:]) - np.median(pool[:na])) >= abs(obs) - 1e-12:
            cnt += 1
    return float(obs), (cnt + 1) / (n + 1)


def boot_diff(a, b, n=20000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    d = [np.median(RNG.choice(b, len(b), True)) - np.median(RNG.choice(a, len(a), True))
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def regime_row(pre, post, metric, label):
    obs, p = perm_test(pre, post)
    lo, hi = boot_diff(pre, post)
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    pre, post = pre[np.isfinite(pre)], post[np.isfinite(post)]
    mw = (stats.mannwhitneyu(pre, post, alternative="two-sided").pvalue
          if len(pre) >= 2 and len(post) >= 2 else np.nan)
    return {"metric": metric, "label": label,
            "n_pre": len(pre), "n_post": len(post),
            "median_pre": float(np.median(pre)) if len(pre) else np.nan,
            "median_post": float(np.median(post)) if len(post) else np.nan,
            "diff_post_minus_pre": obs, "diff_ci_lo": lo, "diff_ci_hi": hi,
            "permutation_p": p, "mannwhitney_p": float(mw) if mw == mw else np.nan}


# --------------------------------------------------------------------------- #
# STEP 5 — ATL13 water-surface elevation heterogeneity per date               #
# --------------------------------------------------------------------------- #
def atl13_wse_by_date():
    cl = pd.read_parquet(T / "atl13_water_classification.parquet")
    cl["date"] = pd.to_datetime(cl["date"])
    cls_map = {"MAIN_CHANNEL": "main_channel",
               "CONNECTED_SIDE_CHANNEL": "connected_side", "TRIBUTARY": "connected_side",
               "RESIDUAL_POND": "isolated_residual", "FLOODED_DEPRESSION": "isolated_residual"}
    cl["plan_class"] = cl.water_class.map(cls_map)
    cl.loc[cl.plan_class.isna() & (cl.is_water == 0.0), "plan_class"] = "exposed_bottom"
    cl["plan_class"] = cl.plan_class.fillna("unknown")

    # pre-breach ATL13 (whole pool) — reload segments, EGG2015 frame, one WSE per date
    from swot_dnipro.vertical import sample_grid
    pre = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet")
    pre["dt"] = pd.to_datetime(pre["time"], utc=True, errors="coerce").dt.tz_localize(None)
    pre = pre[pre.dt < BREACH].dropna(subset=["lat", "lon", "h_wgs84_m", "dt"])
    z = sample_grid(CFG.EGG2015_TIF, pre.lon.values, pre.lat.values)
    pre["wse_m"] = pre.h_wgs84_m.values + CFG.free2mean(pre.lat.values) - z
    pre["date"] = pre.dt.dt.normalize()

    rows = []
    for d, g in pre.groupby("date"):
        if len(g) < 30:
            continue
        v = g.wse_m.values
        rows.append({"date": d, "period": "PRE_BREACH", "n": len(g),
                     "wse_p05_p95_range_m": float(np.percentile(v, 95) - np.percentile(v, 5)),
                     "wse_nmad_m": nmad(v)})
    for (d, per), g in cl.groupby(["date", "period"]):
        w = g[g.plan_class.isin(["main_channel", "connected_side"])]
        if len(w) < 30:
            continue
        v = w.wse_m.values
        rows.append({"date": d, "period": per, "n": len(w),
                     "wse_p05_p95_range_m": float(np.percentile(v, 95) - np.percentile(v, 5)),
                     "wse_nmad_m": nmad(v)})
    return pd.DataFrame(rows), cl


# --------------------------------------------------------------------------- #
def representative_dates(fr):
    def pick(per):
        s = fr[fr.period == per]
        return s.sort_values("total_water_area_km2", ascending=False).iloc[0].date if len(s) else None
    return pick("PRE_BREACH"), pick("DRAWDOWN"), pick("POST_BREACH")


def panel_mask(ax, d, wm, nodes):
    # masks live on the bulk volume (the fourth repo-disk literal of this kind
    # found 2026-09-16; the empty list it produced crashed rasterio.merge)
    mdir = CFG.BULK_ROOT / "data_swot/processed/water_masks"
    files = [mdir / f"{n}_water.tif" for n in wm[wm.date == d].name
             if (mdir / f"{n}_water.tif").exists()]
    if not files:
        ax.set_title(f"{d}: no mask files found"); return
    mos, transform = date_mosaic(files)
    water = (mos == 1)
    h, w = water.shape
    ext = [transform.c, transform.c + w * transform.a,
           transform.f + h * transform.e, transform.f]
    ax.imshow(np.where(mos == 255, np.nan, water), extent=ext, origin="upper",
              cmap="Blues", vmin=0, vmax=1.3, interpolation="nearest")
    ax.scatter(nodes.x, nodes.y, s=1, c="#b03a2e", alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def main() -> None:
    use_style()
    fr = pd.read_csv(T / "fragmentation_metrics_by_date.csv", parse_dates=["date"])
    cn = pd.read_csv(T / "channel_connectivity_by_date.csv", parse_dates=["date"])
    df = fr.merge(cn, on="date", how="left")
    wse, cl = atl13_wse_by_date()
    df = df.merge(wse[["date", "wse_p05_p95_range_m", "wse_nmad_m"]], on="date", how="left")
    # carry pre-breach wse rows that have no S2 mask
    pre_wse = wse[wse.period == "PRE_BREACH"]

    # ---- STEP 6: indicators ------------------------------------------------
    df = df.rename(columns={
        "largest_component_fraction": "F1_largest_fraction",
        "fragmentation_index": "F2_fragmentation_index",
        "connected_channel_len_km": "F3_connected_channel_len_km",
        "wse_p05_p95_range_m": "F4_wse_p05_p95_range_m",
        "n_water_bodies": "F5_n_water_bodies"})
    df.to_csv(FD / "P20_indicators.csv", index=False)

    # ---- STEP 7: statistics, SPLIT BY SENSITIVITY TO SCENE COVERAGE -------
    # Sentinel-2 planform metrics are conditional on how much of the former pool a
    # date actually observes. Only ONE pre-breach date (2023-06-05) is spatially
    # complete, so PRE-vs-POST *inferential* testing of the F-metrics is not done:
    # a median over three PRE dates is dominated by two partial scenes that see 8 %
    # and 49 % of the pool and would report the full reservoir as the more
    # fragmented state. The planform result is reported as a descriptive
    # before/after contrast; the inferential evidence comes from ATL13, which has a
    # real repeated sample on both sides and is coverage-independent.
    def col(per, c, src=df):
        return src[src.period == per][c].dropna().values

    GATES = [("ALL", 0.0), ("cov>=0.50", 0.50), ("cov>=0.80", 0.80)]
    # `inferential` used to be the literal False on every row. That happened to
    # be the right answer, but for a reason the table could not show: with one
    # spatially complete PRE_BREACH date, no PRE-vs-POST contrast of a planform
    # metric is a test. Make the rule explicit and carry the n it rests on, so
    # a future cohort that DOES clear it is recognised without editing code.
    MIN_INFERENTIAL_N = 3          # dates on EACH side of the breach
    sens_rows = []
    for gate, lo in GATES:
        s = df[df.footprint_observed_fraction >= lo]
        for c, lab in [("F1_largest_fraction", "largest connected-body fraction"),
                       ("F2_fragmentation_index", "fragmentation index (1 - F1)"),
                       ("F5_n_water_bodies", "number of water bodies >= MMU"),
                       ("total_water_area_km2", "total water area (km2)"),
                       ("connected_channel_fraction", "connected / observable channel")]:
            if c not in s.columns:
                continue
            per_v = {per: s[s.period == per][c].dropna().values
                     for per in ("PRE_BREACH", "DRAWDOWN", "POST_BREACH")}
            n_pre, n_post = len(per_v["PRE_BREACH"]), len(per_v["POST_BREACH"])
            # An uncontrolled-coverage cohort ("ALL") is never inferential,
            # whatever its n: the audit's central finding (F-02) is that the
            # PRE cohort's F2 spread is ordered by coverage, not by state. Only
            # a coverage-gated cohort with >= MIN_INFERENTIAL_N dates on both
            # sides qualifies. Today that is no row; the rule makes it visible
            # when a future cohort does qualify.
            inferential = bool(gate != "ALL" and min(n_pre, n_post) >= MIN_INFERENTIAL_N)
            for per, v in per_v.items():
                sens_rows.append({
                    "gate": gate, "metric": c, "label": lab, "period": per,
                    "n_dates": len(v),
                    "n_dates_pre": n_pre, "n_dates_post": n_post,
                    "median": float(np.median(v)) if len(v) else np.nan,
                    "nmad": nmad(v) if len(v) > 1 else np.nan,
                    "p05": float(np.percentile(v, 5)) if len(v) else np.nan,
                    "p95": float(np.percentile(v, 95)) if len(v) else np.nan,
                    "inferential": inferential,
                    "inferential_rule": f"gate!=ALL and min(n_pre,n_post)>={MIN_INFERENTIAL_N}"})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(T / "phase20_coverage_sensitivity.csv", index=False)

    # The ONLY inferential PRE-vs-POST test kept here: ATL13 water-surface
    # heterogeneity. It is measured from the altimeter, not from the image
    # footprint, so it is unaffected by Sentinel-2 tile coverage, and it has
    # n = 192 / 23 dates.
    st = pd.DataFrame([regime_row(col("PRE_BREACH", "wse_p05_p95_range_m", wse),
                                  col("POST_BREACH", "wse_p05_p95_range_m", wse),
                                  "F4_wse_p05_p95_range_m",
                                  "ATL13 water-surface p95-p05 range (m)")])
    st.to_csv(T / "pre_post_fragmentation_statistics.csv", index=False)

    # the single spatially complete pre-breach planform observation
    comp = df[(df.period == "PRE_BREACH") & (df.footprint_observed_fraction >= 0.80)]
    comp = comp.sort_values("footprint_observed_fraction").iloc[-1] if len(comp) else None

    # ---- STEP 8: 4-panel main figure -----------------------------------
    wm = pd.read_csv(T / "water_mask_summary.csv")
    wm["dt"] = pd.to_datetime(wm["sensing_time"], format="%Y%m%dT%H%M%S")
    wm["date"] = wm["dt"].dt.strftime("%Y-%m-%d")
    nodes = sword_nodes()
    dpre, ddraw, dpost = representative_dates(fr)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3)
    for i, (dd, ttl) in enumerate([(dpre, "PRE-breach"), (ddraw, "drawdown"),
                                   (dpost, "POST-breach")]):
        ax = fig.add_subplot(gs[0, i])
        if dd is not None:
            panel_mask(ax, str(dd.date()) if hasattr(dd, "date") else str(dd), wm, nodes)
        ax.set_title(f"{'ABC'[i]}  {ttl}  {dd}", fontsize=9)
    axd = fig.add_subplot(gs[1, :])
    d2 = df.sort_values("date")
    axd.plot(d2.date, d2.F2_fragmentation_index, "o-", ms=3, color="#b03a2e",
             label="fragmentation index (1 - largest fraction)")
    axd.plot(d2.date, d2.F1_largest_fraction, "s-", ms=3, color="#1b6ca8",
             label="largest connected-body fraction")
    axd.axvline(BREACH, color="k", ls="--", lw=1)
    axd.text(BREACH, 0.02, " 2023-06-06 breach", fontsize=8, rotation=90, va="bottom")
    axd.set_ylim(-0.03, 1.03); axd.set_ylabel("fraction / index")
    axd.legend(fontsize=8, loc="center left"); axd.set_title("D  planform fragmentation over time", fontsize=9)
    fig.tight_layout()
    save(fig, "P20_Fig1_fragmentation", FIG)
    plt.close(fig)

    # ---- STEP 9: ATL13 chainage vs elevation, by class -----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    col_c = {"main_channel": "#1b6ca8", "connected_side": "#48c9b0",
             "isolated_residual": "#b03a2e", "exposed_bottom": "#c8b878",
             "unknown": "#c4ccd1"}
    reps = {}
    for per, ax in zip(["PRE_BREACH", "POST_BREACH"], axes):
        sub = cl[cl.period == per]
        if per == "PRE_BREACH":
            dsel = sub.date.value_counts().idxmax() if len(sub) else None
        else:
            # a post date with a real channel profile
            prof = pd.read_csv(T / "channel_only_profiles.csv", parse_dates=["date"])
            cand = prof[(prof.period == "POST_BREACH") & (prof.subset == "MAIN_CHANNEL")]
            dsel = cand.sort_values("n_segments").iloc[-1].date if len(cand) else None
        reps[per] = dsel
        gg = sub[sub.date == dsel] if dsel is not None else sub.iloc[:0]
        for cls_, c in col_c.items():
            p = gg[gg.plan_class == cls_]
            if len(p):
                ax.scatter(p.chain_km, p.wse_m, s=6, c=c, alpha=0.5, lw=0, label=cls_)
        ax.set_title(f"{per}  {pd.Timestamp(dsel).date() if dsel is not None else '-'}", fontsize=9)
        ax.set_xlabel("chainage from dam (km)"); ax.set_ylabel("ATL13 WSE (m, EGG2015)")
        ax.legend(fontsize=7)
    fig.suptitle("ATL13 elevation vs chainage by planform class — why the all-water slope was contaminated",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "P20_Fig2_atl13_by_class", FIG)
    plt.close(fig)

    # ---- STEP 10: wording -------------------------------------------------
    def gmed(gate_lo, per, c, src=df):
        v = src[(src.period == per) & (src.footprint_observed_fraction >= gate_lo)][c].dropna()
        return float(v.median()) if len(v) else np.nan

    L = []
    w = L.append
    w("# Phase 20 — Loss of the continuous Kakhovka reservoir water surface\n")
    w(f"Generated {_date.today()}. Primary unit: one Sentinel-2 date / water mask. "
      f"Water bodies are connected components of the S2 NDWI∧MNDWI∧SCL mask inside the "
      f"data-driven former-pool footprint (largest connected component of the 2023-06-05 "
      f"mosaic, {df.total_aoi_km2.iloc[0]:,.0f} km²); MMU 0.05 km².\n")

    w("## Coverage is a first-class variable\n")
    w("Every planform metric below is conditional on `footprint_observed_fraction` — the "
      "share of the former pool a date actually observes. A single-tile scene can see as "
      "little as 8 % of the pool; its 'largest connected body' is small for that reason "
      "alone. Dates are kept in all tables and classed, never deleted:\n")
    w("| period | PARTIAL <0.50 | ACCEPTABLE 0.50–0.80 | HIGH ≥0.80 |")
    w("|---|---:|---:|---:|")
    ct = df.pivot_table(index="period", columns="coverage_class", values="date",
                        aggfunc="count").fillna(0).astype(int)
    for per in ("PRE_BREACH", "DRAWDOWN", "POST_BREACH"):
        if per in ct.index:
            r = ct.loc[per]
            w(f"| {per} | {r.get('PARTIAL', 0)} | {r.get('ACCEPTABLE', 0)} | {r.get('HIGH', 0)} |")
    w("")

    w("## Planform: a descriptive before/after contrast, NOT a significance test\n")
    if comp is not None:
        w(f"**The only spatially complete pre-breach Sentinel-2 observation** "
          f"({comp.date if not hasattr(comp.date, 'date') else comp.date.date()}, "
          f"coverage {comp.footprint_observed_fraction:.2f}) shows a nearly continuous "
          f"reservoir surface: **{comp.total_water_area_km2:,.0f} km² of water with "
          f"{comp.F1_largest_fraction*100:.3f} % held in the largest connected component** "
          f"({int(comp.F5_n_water_bodies)} bodies ≥ MMU).\n")
        w("This is **n = 1**. It is not a sample of a pre-breach regime and no p-value is "
          "computed from it — it is one unambiguous observation of a known physical state "
          "(a filled reservoir). The two other pre-breach dates see 8 % and ~49 % of the "
          "pool and are excluded from the contrast for that reason, not for their values.\n")
    w("The post-breach side does have a real sample:\n")
    w("| gate | n POST dates | largest-body fraction | fragmentation index | bodies ≥ MMU |")
    w("|---|---:|---:|---:|---:|")
    for gate, lo in (("ALL", 0.0), ("cov ≥ 0.50", 0.50), ("cov ≥ 0.80", 0.80)):
        n = int(((df.period == "POST_BREACH") & (df.footprint_observed_fraction >= lo)).sum())
        w(f"| {gate} | {n} | {gmed(lo,'POST_BREACH','F1_largest_fraction'):.3f} | "
          f"{gmed(lo,'POST_BREACH','F2_fragmentation_index'):.3f} | "
          f"{gmed(lo,'POST_BREACH','F5_n_water_bodies'):.0f} |")
    w("")
    w("The areal metrics are **stable across the coverage gates**, so the post-breach "
      "fragmentation is not an artefact of scene footprint. The **body count is not** "
      "stable and must never be compared across dates without a gate.\n")

    w("## Channel connectivity — use the fraction, not the kilometres\n")
    w("`connected_channel_len_km` is capped by the SWORD node set itself: the node "
      "sequence has a >5 km hole at chainage 32–44 km, so the walk from the dam can "
      "never report more than 32 km whatever the water does. Numerator and denominator "
      "share that ceiling, so **`connected_channel_fraction` = connected / observable is "
      "the metric to analyse**; it is valid over the first ~32 km of the stem and is "
      "*not* a statement about the whole 240 km reservoir.\n")
    w("| gate | PRE (complete obs.) | POST median | n POST |")
    w("|---|---:|---:|---:|")
    for gate, lo in (("ALL", 0.0), ("cov ≥ 0.50", 0.50), ("cov ≥ 0.80", 0.80)):
        n = int(((df.period == "POST_BREACH") & (df.footprint_observed_fraction >= lo)
                 & df.connected_channel_fraction.notna()).sum())
        w(f"| {gate} | {gmed(0.80,'PRE_BREACH','connected_channel_fraction'):.2f} | "
          f"{gmed(lo,'POST_BREACH','connected_channel_fraction'):.3f} | {n} |")
    w("")

    w("## The inferential evidence is ICESat-2, and it is separate\n")
    w("The two sensors are deliberately not made to prove the same thing with the same "
      "test. Sentinel-2 gives structural/planform evidence with one complete pre-breach "
      "observation; ICESat-2 gives repeated quantitative evidence on both sides:\n")
    r = st.iloc[0]
    w(f"- **water-surface heterogeneity** (p95−p05 range within a date): "
      f"**{r.median_pre:.3f} → {r.median_post:.3f} m**, Δ **{r.diff_post_minus_pre:+.3f} m** "
      f"95 % CI [{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}], permutation p = "
      f"{r.permutation_p:.5f}, Mann–Whitney p = {r.mannwhitney_p:.2e} "
      f"(n = {r.n_pre} / {r.n_post} dates). Measured by the altimeter, so it is "
      f"**independent of Sentinel-2 tile coverage**.")
    w("- **longitudinal slope**, PRE n = 14 vs POST n = 14 (`make_transition_stats.py`, "
      "FigG): +0.09 → +3.31 cm/km, Δ +3.22 [+1.99, +5.07] cm/km, permutation p = 1e-4.")
    w("")

    w("## Files\n")
    w("- `outputs/tables/water_body_objects.parquet` — every (date, component), with the "
      "date's coverage fraction and class carried on each row")
    w("- `outputs/tables/fragmentation_metrics_by_date.csv` — incl. "
      "`footprint_observed_fraction`, `observed_aoi_km2`, `valid_pixels`, "
      "`total_aoi_pixels`, `coverage_class`")
    w("- `outputs/tables/channel_connectivity_by_date.csv` — incl. "
      "`observable_channel_len_km`, `connected_channel_fraction`")
    w("- `outputs/tables/phase20_coverage_sensitivity.csv` — every metric × gate × period")
    w("- `outputs/tables/pre_post_fragmentation_statistics.csv` — ATL13 WSE test only")
    w("- `outputs/figures/P20_Fig1_fragmentation.*`, `P20_Fig2_atl13_by_class.*`")
    w("")
    w("## Caveats\n")
    w("- Only one spatially complete pre-breach Sentinel-2 date exists; the planform "
      "contrast is descriptive by construction and is reported as such.")
    w("- Body counts depend on scene footprint (single-tile dates see fewer bodies simply "
      "by seeing less pool) — always report them with a coverage gate.")
    w("- Connected-channel kilometres are capped at 32 km by a hole in the SWORD node "
      "sequence, not by hydrology; use `connected_channel_fraction`.")
    w("- `gdalwarp -tps` on the S1 side was geocoding, not terrain correction; the S1 "
      "branch is closed (backscatter thresholding did not separate water from the "
      "exposed bed in the reservoir interior — this is a data limitation, not evidence).")
    w("- Connectivity uses SWORD topology + connected-component identity, not Euclidean "
      "proximity; gaps up to 5 km are bridged.")
    (CFG.REPORTS / "kakhovka_fragmentation_analysis.md").write_text("\n".join(L))

    print("\n".join(L))
    print(f"\n-> {CFG.REPORTS/'kakhovka_fragmentation_analysis.md'}")
    print(f"-> {T/'pre_post_fragmentation_statistics.csv'}")
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
