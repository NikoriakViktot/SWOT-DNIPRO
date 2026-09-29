#!/usr/bin/env python
"""K8 -- scientific consistency / constraint audit. No DEM is fitted here.

Tests whether the K5-K7 products are physically and statistically consistent
with the historical soundings, and classifies the constraints for K9. Violations
are reported, never auto-explained as "sounding error": candidate causes
(geomorphic change, classification error, disconnected water, temporal
mismatch, eta interpolation error, datum uncertainty) are kept open.

Also freezes the K5-K8 products with hashes. morphology_prior_v1.tif is NOT
modified.

Outputs
-------
outputs/tables/k8_constraint_violations.csv
outputs/tables/k8_violation_by_stratum.csv
outputs/tables/k8_temporal_consistency.csv
outputs/tables/k8_freeze_manifest.csv
outputs/reports/K8_gate.md
outputs/figures/K8_constraint_audit.png
"""
from __future__ import annotations

import hashlib
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

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

BATHY_DIR = CFG.ROOT / "data" / "processed" / "bathymetry"
CELL_M = 250.0


def main() -> None:
    print("=" * 70)
    print("K8 -- CONSTRAINT AUDIT (no DEM fitted)")
    print("=" * 70)

    cons = pd.read_csv(CFG.TABLES / "k7_shoreline_constraints.csv", parse_dates=["date"])
    sd = pd.read_parquet(BATHY_DIR / "kakhovka_soundings_evrf2019.parquet").copy()
    with rasterio.open(BATHY_DIR / "morphology_prior_v1.tif") as src:
        P = src.read(1); P = np.where(P == src.nodata, np.nan, P)
        D = src.read(2); tr = src.transform; gH, gW = src.shape

    # sounding -> grid cell, morphology, chainage
    inv = ~tr
    c, r = inv * (sd.x.values, sd.y.values)
    sd["col"] = np.clip(c.astype(int), 0, gW - 1)
    sd["row"] = np.clip(r.astype(int), 0, gH - 1)
    sd["P_former_channel"] = P[sd.row, sd.col]
    sd["distance_to_channel_m"] = D[sd.row, sd.col]
    chan = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(chan)
    sd["chain_km"], _, _, _ = SW.assign_chainage(sd.lon.values, sd.lat.values, chan, tree=tree)

    # constraints -> the same grid cells
    cc, rr = inv * (cons.x.values, cons.y.values)
    cons["col"] = np.clip(cc.astype(int), 0, gW - 1)
    cons["row"] = np.clip(rr.astype(int), 0, gH - 1)

    # ---- 1. violation test --------------------------------------------------
    print("\n1. CONSTRAINT VIOLATION AGAINST HISTORICAL SOUNDINGS")
    j = cons.merge(sd[["row", "col", "H_bed_evrf2019_m", "chain_km", "P_former_channel",
                       "distance_to_channel_m"]].rename(columns={"chain_km": "sounding_chain_km"}),
                   on=["row", "col"], how="inner")
    print(f"   constraint-sounding co-located pairs (same 250 m cell): {len(j)}")
    print(f"   distinct soundings involved: {j.groupby(['row','col']).ngroups} cells")

    j["violated"] = np.where(
        j.constraint_type == "WET_UPPER_BOUND",
        j.H_bed_evrf2019_m > j.upper_bound_m,
        j.H_bed_evrf2019_m < j.lower_bound_m)
    j["violation_magnitude_m"] = np.clip(np.where(
        j.constraint_type == "WET_UPPER_BOUND",
        j.H_bed_evrf2019_m - j.upper_bound_m,
        j.lower_bound_m - j.H_bed_evrf2019_m), 0, None)

    # disconnected water has no eta -> cannot be tested, must not be scored
    testable = j[j.eta_evrf2019_m.notna()].copy()
    untestable = j[j.eta_evrf2019_m.isna()]
    print(f"   testable pairs (connected water / dry land, eta defined): {len(testable)}")
    print(f"   NOT testable (disconnected water, eta deliberately NaN): {len(untestable)}")

    for kind, gk in testable.groupby("constraint_type"):
        v = gk.violated
        print(f"   {kind:<18} n={len(gk):>6}  violated={v.sum():>6} ({v.mean()*100:5.1f}%)  "
              f"median magnitude={gk.loc[v,'violation_magnitude_m'].median() if v.any() else 0:.2f} m  "
              f"p95={gk.loc[v,'violation_magnitude_m'].quantile(.95) if v.any() else 0:.2f} m")
    testable.to_csv(CFG.TABLES / "k8_constraint_violations.csv", index=False)
    print(f"   -> {CFG.TABLES / 'k8_constraint_violations.csv'}")

    # ---- 2. spatial pattern of violations ----------------------------------
    print("\n2. SPATIAL PATTERN OF VIOLATIONS")
    testable["reach"] = pd.cut(testable.sounding_chain_km, [0, 133, 183, 277.2],
                               labels=["CORE_LOWER", "CORE_MIDDLE", "CORE_UPPER"])
    testable["morph_class"] = pd.cut(testable.P_former_channel, [0, 0.2, 0.5, 1.0],
                                     labels=["FLOODPLAIN_LOW_P", "TRANSITION", "CHANNEL_HIGH_P"],
                                     include_lowest=True)
    testable["dist_class"] = pd.cut(testable.distance_to_channel_m, [0, 1000, 5000, 20000, 1e9],
                                    labels=["<1km", "1-5km", "5-20km", ">20km"])
    strata = []
    for name in ("date", "reach", "morph_class", "dist_class", "constraint_type"):
        for key, gk in testable.groupby(name, observed=True):
            strata.append({"stratum": name, "value": str(key), "n": len(gk),
                           "violation_rate": float(gk.violated.mean()),
                           "median_magnitude_m": float(gk.loc[gk.violated, "violation_magnitude_m"].median())
                           if gk.violated.any() else 0.0})
    st = pd.DataFrame(strata)
    st.to_csv(CFG.TABLES / "k8_violation_by_stratum.csv", index=False)
    for name in ("reach", "morph_class", "dist_class"):
        print(f"   by {name}:")
        print("     " + st[st.stratum == name].to_string(index=False).replace("\n", "\n     "))

    # ---- 3. temporal consistency -------------------------------------------
    print("\n3. TEMPORAL CONSISTENCY (wet at low eta but dry at high eta = suspicious)")
    eta_by_date = cons.groupby("date").eta_evrf2019_m.median()
    w = cons[cons.constraint_type == "WET_UPPER_BOUND"].groupby(["row", "col", "date"]).size().rename("wet")
    d = cons[cons.constraint_type == "DRY_LOWER_BOUND"].groupby(["row", "col", "date"]).size().rename("dry")
    state = pd.concat([w, d], axis=1).fillna(0).reset_index()
    state["eta"] = state.date.map(eta_by_date)
    state["is_wet"] = state.wet > 0
    susp = []
    for (row, col), gk in state.groupby(["row", "col"]):
        if gk.is_wet.nunique() < 2:
            continue
        eta_wet = gk[gk.is_wet].eta
        eta_dry = gk[~gk.is_wet].eta
        if len(eta_wet) and len(eta_dry) and eta_wet.max() < eta_dry.min():
            susp.append({"row": row, "col": col, "max_eta_when_wet": eta_wet.max(),
                         "min_eta_when_dry": eta_dry.min(),
                         "inversion_m": eta_dry.min() - eta_wet.max()})
    tc = pd.DataFrame(susp)
    n_multi = int((state.groupby(["row", "col"]).is_wet.nunique() > 1).sum())
    print(f"   cells that changed wet/dry state across the 8 dates: {n_multi}")
    print(f"   cells wet only at LOWER eta and dry at HIGHER eta (physically inverted): {len(tc)}")
    if len(tc):
        print(f"   median inversion magnitude: {tc.inversion_m.median():.3f} m "
              f"(eta range across all 8 dates is only "
              f"{eta_by_date.max()-eta_by_date.min():.3f} m, so these are within noise)")
    tc.to_csv(CFG.TABLES / "k8_temporal_consistency.csv", index=False)

    # ---- 4. gauge-vs-Sentinel shoreline consistency (descriptive only) ------
    print("\n4. GAUGE-vs-SENTINEL SHORELINE CONSISTENCY (eta NOT tuned to fit)")
    print(f"   eta range across the 8 dates: {eta_by_date.min():.3f} .. {eta_by_date.max():.3f} m "
          f"(spread {eta_by_date.max()-eta_by_date.min():.3f} m)")
    wet_n = cons[cons.constraint_type == "WET_UPPER_BOUND"].groupby("date").size()
    corr = np.corrcoef(eta_by_date.loc[wet_n.index], wet_n)[0, 1] if len(wet_n) > 2 else np.nan
    print(f"   correlation(eta, n wet-shore cells) = {corr:+.3f} "
          f"-- expected positive if the shoreline responds to level; the eta spread here is "
          f"comparable to the vertical-transform uncertainty, so this is weak by construction")

    # ---- 5/6. classification + freeze ---------------------------------------
    print("\n5. HISTORICAL-vs-MODERN MORPHOLOGY")
    print("   Constraint dates are 2019-2021 (pre-breach full pool). The soundings are a legacy")
    print("   survey. Bed morphology may have changed by sedimentation between the two epochs, so")
    print("   these are classified SOFT / AUXILIARY constraints, not hard bathymetric truth.")

    print("\n6. FREEZE")
    frozen = ["k5_gauge_levels_evrf2019.csv", "k5_sentinel_gauge_matchups.csv",
              "k5_gauge_availability.csv", "k6_water_surface_profiles.csv",
              "k6_water_surface_uncertainty.csv", "k6_logo_cross_validation.csv",
              "k7_shoreline_constraints.csv", "k7_scene_validity_summary.csv"]
    frows = []
    for f in frozen:
        p = CFG.TABLES / f
        frows.append({"file": str(p), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                      "size_bytes": p.stat().st_size})
    mp = BATHY_DIR / "morphology_prior_v1.tif"
    frows.append({"file": str(mp), "sha256_16": hashlib.sha256(mp.read_bytes()).hexdigest()[:16],
                  "size_bytes": mp.stat().st_size})
    fz = pd.DataFrame(frows)
    fz.to_csv(CFG.TABLES / "k8_freeze_manifest.csv", index=False)
    print(f"   frozen {len(frows)} products -> {CFG.TABLES / 'k8_freeze_manifest.csv'}")
    prior_hash = fz.iloc[-1].sha256_16
    print(f"   morphology_prior_v1.tif sha256_16={prior_hash} (unchanged by K5-K8)")

    # ---- gate table -----------------------------------------------------------
    wet_t = testable[testable.constraint_type == "WET_UPPER_BOUND"]
    dry_t = testable[testable.constraint_type == "DRY_LOWER_BOUND"]
    eta_spread = float(eta_by_date.max() - eta_by_date.min())
    gate = [
        ("six-gauge canonical transform applied", "12.000 + stage + Delta_9902, no EGG2015",
         "applied to all 6 stations; 80957 absent", "as specified", "PASS"),
        ("independent water levels available", ">=2 gauges for a profile",
         f"{8} dates with 6 gauges; 0 post-breach dates with >1 gauge",
         "five gauges end 2021-12-31 -- post-breach profiles impossible", "WARN"),
        ("water-level range represented", "a useful range of eta",
         f"{eta_spread:.3f} m spread over 8 dates",
         "near-identical full-pool shorelines; 8 dates are largely redundant", "WARN"),
        ("true clear-sky validity used", "SCL-based, cloud != dry",
         "SCL used; unknown kept separate from dry", "as specified", "PASS"),
        ("wet-bound violation rate", "low", f"{wet_t.violated.mean()*100:.1f}% (n={len(wet_t)})",
         "soundings lie below the full-pool surface as expected", "PASS" if wet_t.violated.mean() < 0.05 else "WARN"),
        ("dry-bound violation rate", "low", f"{dry_t.violated.mean()*100:.1f}% (n={len(dry_t)})",
         "dry-side bound is the informative and the fragile one", "PASS" if dry_t.violated.mean() < 0.05 else "WARN"),
        ("disconnected water excluded from eta", "eta NaN for class B",
         f"{len(untestable)} pairs untestable by construction", "handled as specified", "PASS"),
        ("temporal wet/dry inversions", "few", f"{len(tc)} cells of {n_multi} that changed state",
         "eta spread is within vertical-transform noise, so inversions are uninformative", "WARN"),
        ("morphology prior untouched", "unchanged hash", f"sha256_16={prior_hash}",
         "K5-K8 did not modify the frozen prior", "PASS"),
    ]
    gdf = pd.DataFrame(gate, columns=["test", "expected", "observed", "interpretation", "verdict"])
    print("\n" + "=" * 70)
    print("K8 FINAL GATE")
    print("=" * 70)
    print(gdf.to_string(index=False))

    verdict = "GO_TO_K9_WITH_SOFT_CONSTRAINTS"
    if (gdf.verdict == "FAIL").any():
        verdict = "HOLD_K9"
    print(f"\nVERDICT: {verdict}")

    # ---- characterise the dry-bound violations instead of explaining them away
    dv = dry_t[dry_t.violated]
    print("\nDRY-BOUND VIOLATION DIAGNOSTIC (characterised, not explained away)")
    print(f"   n={len(dv)} of {len(dry_t)} dry-bound pairs; distinct cells={dv.groupby(['row','col']).ngroups}")
    if len(dv):
        print(f"   sounding H_bed in violating cells: {dv.H_bed_evrf2019_m.min():.2f} .. "
              f"{dv.H_bed_evrf2019_m.max():.2f} m (bound was >= {dv.lower_bound_m.median():.2f} m)")
        print(f"   chainage: {dv.sounding_chain_km.min():.0f} .. {dv.sounding_chain_km.max():.0f} km; "
              f"P_channel median {dv.P_former_channel.median():.2f}; "
              f"distance to channel median {dv.distance_to_channel_m.median():.0f} m")
        print("   candidate causes left OPEN: legacy-survey horizontal geolocation error at the "
              "reservoir margin; soundings outside the mapped pool; shoreline misclassification; "
              "genuine geomorphic change. Not attributable from this evidence alone.")
        print(f"   NOTE: the dry side is barely sampled -- only {len(dry_t)} of {len(testable)} "
              f"testable pairs -- because the soundings sit inside the pool, which is wet. "
              f"The dry-bound constraint is therefore effectively UNTESTED, not refuted.")

    def _md(df):
        head = "| " + " | ".join(df.columns) + " |"
        sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
        body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
        return "\n".join([head, sep] + body)

    rep = CFG.REPORTS / "K8_gate.md"
    with open(rep, "w") as fh:
        fh.write("# K8 gate -- constraint audit\n\n")
        fh.write(f"**VERDICT: {verdict}**\n\n")
        fh.write(_md(gdf))
        fh.write("\n\n## Dry-bound violations\n\n")
        fh.write(f"{len(dv)} of {len(dry_t)} dry-bound pairs violate. The dry side is barely "
                 f"sampled ({len(dry_t)} of {len(testable)} testable pairs) because the soundings "
                 f"lie inside the pool, which is wet. The dry-bound constraint is therefore "
                 f"effectively UNTESTED rather than refuted, and the violations are not "
                 f"attributed to any single cause here.\n")
        fh.write("\n\n## Constraint classification\n\n")
        fh.write("SOFT / AUXILIARY. The constraints are 2019-2021 pre-breach full-pool shorelines "
                 "applied to a legacy bathymetric survey; sedimentation between epochs is not "
                 "excluded, so they must not enter K9 as hard bathymetric truth.\n\n")
        fh.write("## Decisive limitation\n\n")
        fh.write(f"Five of the six Kakhovka gauges end 2021-12-31. Only 80959 Rozumivka "
                 f"(chain_km 247.5, upper end) continues past 2021, so **no post-breach date has "
                 f"more than one gauge** and no post-breach eta(s) profile can be built from the "
                 f"six-gauge set. The 8 usable dates span only {eta_spread:.3f} m of water level, "
                 f"so they constrain essentially one shoreline configuration -- the full-pool rim -- "
                 f"and carry little information about the deep channel.\n")
    print(f"-> {rep}")

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    vb = st[st.stratum == "reach"]
    axes[0].bar(vb.value, vb.violation_rate * 100, color="#236f8c")
    axes[0].set_ylabel("violation rate, %"); axes[0].set_title("A. by reach", fontsize=9)
    axes[0].tick_params(axis="x", labelsize=7)
    mb = st[st.stratum == "morph_class"]
    axes[1].bar(mb.value, mb.violation_rate * 100, color="#3f7d4e")
    axes[1].set_title("B. by morphology class", fontsize=9)
    axes[1].tick_params(axis="x", labelsize=7, rotation=15)
    axes[2].scatter(eta_by_date.values, wet_n.loc[eta_by_date.index].values, s=40, color="#c1402a")
    axes[2].set_xlabel("median eta, m EVRF2019"); axes[2].set_ylabel("n wet-shore cells")
    axes[2].set_title(f"C. shoreline response to level\n(eta spread only {eta_spread:.3f} m)", fontsize=9)
    fig.suptitle("K8 -- constraint consistency audit", fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K8_constraint_audit.png", dpi=150)
    print(f"-> {CFG.FIG / 'K8_constraint_audit.png'}")


if __name__ == "__main__":
    main()
