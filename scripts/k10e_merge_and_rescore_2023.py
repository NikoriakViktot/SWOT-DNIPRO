#!/usr/bin/env python
"""K10e.V4+V9(revised) -- merge Dynamic World into the 2023 context and rebuild
risk scores with an independent structural signal, not spectral indices alone.

CAUTION on DW label reliability here: top-1 DW labels on this freshly-exposed
lakebed include crops=31.3% and built=8.7% of all 2023 accepted segments --
implausible for a drained reservoir bed with no cropland or buildings on it.
Dynamic World is trained on a global land-cover distribution and its top-1
label is not trusted on this surface type; bright dry sediment/salt deposits
here plausibly get confused with bare cropland or built surfaces spectrally.

trees_probability and shrub_and_scrub_probability are treated as more
reliable structural cues than the top-1 label or the crops/built/grass
probabilities, because woody canopy has a distinctive NIR/structure signal
less easily confused with bare ground. water_probability is cross-checked
against the existing exact vector water QC (V8) as a sanity control, not
used to override it.

Outputs
-------
data/processed/current_bed/k10e_post_breach_full_context.parquet   (merged, V9 rescored)
outputs/tables/k10e_primary_ground_validation_post_breach_v2.csv
outputs/tables/k10e_vegetation_diagnostic_post_breach_v2.csv
outputs/tables/k10e_dw_reliability_check.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

CUR = CFG.ROOT / "data" / "processed" / "current_bed"
LOW_RISK = 0.20


def clip01(v):
    return np.clip(np.nan_to_num(v, nan=0.0), 0.0, 1.0)


def main() -> None:
    print("=" * 78)
    print("K10e -- merge Dynamic World, rebuild risk scores with structural evidence")
    print("=" * 78)

    g = pd.read_parquet(CUR / "k10e_post_breach_vegetation_context.parquet")
    g["row_id"] = np.arange(len(g))
    dw = pd.read_parquet(CUR / "k10e_post_breach_dynamic_world.parquet")
    g = g.merge(dw, on="row_id", how="left")
    print(f"merged: {len(g):,} rows, DW attribution "
          f"{int(g.dw_max_probability.notna().sum()):,}/{len(g):,}")

    # ---------------- DW reliability sanity check ---------------------------
    print("\n" + "-" * 78)
    print("DW reliability check: top-1 label vs the ALREADY-VALIDATED exact-vector water QC")
    print("-" * 78)
    check = g[g.dw_label_name.notna()].copy()
    check["exact_water"] = check.water_state == "WET"
    xt = pd.crosstab(check.dw_label_name, check.exact_water)
    xt.columns = ["exact_DRY", "exact_WET"]
    xt["dw_water_precision"] = np.where(check.groupby("dw_label_name").size() > 0,
                                        xt.get("exact_WET", 0) / xt.sum(1), np.nan)
    print(xt.to_string())
    xt.to_csv(CFG.TABLES / "k10e_dw_reliability_check.csv")
    n_crops = int((check.dw_label_name == "crops").sum())
    n_built = int((check.dw_label_name == "built").sum())
    print(f"\nWARNING: DW top-1 labels 'crops' ({n_crops:,}, "
          f"{100*n_crops/len(check):.1f}%) and 'built' ({n_built:,}, "
          f"{100*n_built/len(check):.1f}%) on a drained reservoir bed with no "
          f"cropland or structures on it are NOT trusted as land-cover truth here.")
    print("Only trees_probability / shrub_and_scrub_probability are used below, as an")
    print("independent STRUCTURAL cue (woody canopy), not as a general land-cover label.")

    # ---------------- V9 revised: structural risk now uses DW + PhoREAL -----
    print("\n" + "-" * 78)
    print("V9 (revised) -- woody_canopy_risk_score now combines PhoREAL structure "
          "(ATL03 photons) AND Dynamic World trees/shrub probability (independent source)")
    print("-" * 78)
    hc = g.h_canopy.values
    veg_ph_frac = np.where(g.ph_count.values > 0,
                           g.veg_ph_count.values / np.maximum(g.ph_count.values, 1), 0.0)
    woody_dw = clip01(g.trees_probability.fillna(0).values
                      + g.shrub_and_scrub_probability.fillna(0).values)
    phoreal_component = 0.6 * clip01(hc / 3.0) + 0.4 * clip01(veg_ph_frac / 0.3)
    # two INDEPENDENT sources (ATL03 photon structure vs Sentinel-2-trained DW
    # classifier); take the max rather than average, so either source alone can
    # flag a risk -- averaging would let one weak source dilute a strong flag
    # from the other.
    g["woody_canopy_risk_score"] = np.maximum(phoreal_component, woody_dw)
    g["woody_canopy_risk_dw_component"] = woody_dw
    g["woody_canopy_risk_phoreal_component"] = phoreal_component
    g["woody_canopy_risk_note"] = "max(PhoREAL ATL03 structure, DW trees+shrub_scrub probability)"

    print(f"  woody_canopy_risk_score: median {np.nanmedian(g.woody_canopy_risk_score):.3f}, "
          f"p90 {np.nanpercentile(g.woody_canopy_risk_score,90):.3f}")
    print(f"  segments flagged (>=0.20) by PhoREAL alone: "
          f"{int((phoreal_component>=0.20).sum()):,}")
    print(f"  segments flagged (>=0.20) by DW alone: {int((woody_dw>=0.20).sum()):,}")
    print(f"  segments flagged by BOTH: "
          f"{int(((phoreal_component>=0.20)&(woody_dw>=0.20)).sum()):,}")
    print(f"  segments flagged by EITHER (union, what the risk score now uses): "
          f"{int((g.woody_canopy_risk_score>=0.20).sum()):,}")

    # PhoREAL-only PRIMARY, computed fresh here (not a hardcoded number from the
    # earlier 2023-only run) so the "what did adding DW change" comparison below
    # is always against the actual current population.
    water_qc_ok0 = g.exposed_ground_validation_ok
    good_class0 = g.surface_class.isin(["DRY_BARE_SEDIMENT", "SPARSE_HERBACEOUS"])
    attributed0 = g.surface_class != "UNATTRIBUTED"
    low_risk_phoreal_only = ((g.vegetation_risk_score < LOW_RISK) & (phoreal_component < LOW_RISK)
                            & (g.mixed_pixel_risk_score < LOW_RISK))
    primary_phoreal_only = water_qc_ok0 & low_risk_phoreal_only & good_class0 & attributed0
    n_primary_phoreal_only = int(primary_phoreal_only.sum())
    n_chan_phoreal_only = int((primary_phoreal_only & (g.P_channel >= 0.5)).sum())
    rgt_chan_phoreal_only = g[primary_phoreal_only & (g.P_channel >= 0.5)].rgt.nunique()

    # ---------------- V10 revised: PRIMARY / DIAGNOSTIC split ----------------
    print("\n" + "-" * 78)
    print("V10 (revised) -- PRIMARY_GROUND_VALIDATION_SET with DW-informed woody risk")
    print("-" * 78)
    water_qc_ok = g.exposed_ground_validation_ok
    low_risk = ((g.vegetation_risk_score < LOW_RISK) & (g.woody_canopy_risk_score < LOW_RISK)
                & (g.mixed_pixel_risk_score < LOW_RISK))
    good_class = g.surface_class.isin(["DRY_BARE_SEDIMENT", "SPARSE_HERBACEOUS"])
    attributed = g.surface_class != "UNATTRIBUTED"
    primary = water_qc_ok & low_risk & good_class & attributed
    diagnostic = water_qc_ok & ~primary
    g["k10e_split"] = np.select(
        [primary, diagnostic, ~water_qc_ok],
        ["PRIMARY_GROUND_VALIDATION_SET", "VEGETATION_DIAGNOSTIC_SET", "REJECTED_BY_WATER_QC"],
        default="REJECTED_BY_WATER_QC")

    print(f"  PRIMARY_GROUND_VALIDATION_SET (DW-informed): {int(primary.sum()):,} "
          f"(PhoREAL-only woody risk, same population: {n_primary_phoreal_only:,})")
    print(f"  VEGETATION_DIAGNOSTIC_SET: {int(diagnostic.sum()):,}")
    newly_flagged = int((water_qc_ok & good_class & attributed
                         & (phoreal_component < LOW_RISK) & (woody_dw >= LOW_RISK)
                         & (g.vegetation_risk_score < LOW_RISK)
                         & (g.mixed_pixel_risk_score < LOW_RISK)).sum())
    print(f"  segments PhoREAL called low-risk but DW trees/shrub flagged as woody "
          f"(newly excluded from PRIMARY by adding DW): {newly_flagged:,}")

    chan_primary = primary & (g.P_channel >= 0.5)
    print(f"\n  channel stratum (P_channel>=0.5), PhoREAL-only PRIMARY: "
          f"{n_chan_phoreal_only:,} segments, {rgt_chan_phoreal_only} RGTs")
    print(f"  channel stratum (P_channel>=0.5) inside DW-informed PRIMARY: "
          f"{int(chan_primary.sum()):,} segments, {g[chan_primary].rgt.nunique()} RGTs")

    # Sensitivity: a second, relaxed risk threshold, stored ALONGSIDE the
    # primary 0.20 split rather than replacing it -- LOW_RISK was picked as a
    # round number, not fit to anything, and the channel stratum specifically
    # is thin enough (41 segments at 0.20) that the choice of threshold
    # visibly changes which RGTs have usable fold support for D14. Reporting
    # both lets that sensitivity be seen instead of silently picking one.
    RELAXED_RISK = 0.30
    low_risk_relaxed = ((g.vegetation_risk_score < RELAXED_RISK)
                        & (g.woody_canopy_risk_score < RELAXED_RISK)
                        & (g.mixed_pixel_risk_score < RELAXED_RISK))
    primary_relaxed = water_qc_ok & low_risk_relaxed & good_class & attributed
    g["k10e_split_relaxed_0_30"] = np.where(primary_relaxed, "PRIMARY_GROUND_VALIDATION_SET",
                                            g.k10e_split)
    chan_primary_relaxed = primary_relaxed & (g.P_channel >= 0.5)
    print(f"\n  SENSITIVITY -- relaxed risk threshold {RELAXED_RISK} (all three risk scores):")
    print(f"    PRIMARY_GROUND_VALIDATION_SET: {int(primary_relaxed.sum()):,} "
          f"(vs {int(primary.sum()):,} at 0.20)")
    print(f"    channel stratum: {int(chan_primary_relaxed.sum()):,} segments, "
          f"{g[chan_primary_relaxed].rgt.nunique()} RGTs (vs {int(chan_primary.sum()):,} "
          f"segments, {g[chan_primary].rgt.nunique()} RGTs at 0.20)")
    print(f"    per-RGT channel counts at {RELAXED_RISK}:")
    print(g[chan_primary_relaxed].groupby("rgt").size().sort_values(ascending=False).to_string())

    gp = CUR / "k10e_post_breach_full_context.parquet"
    g.to_parquet(gp, index=False)
    g[primary].to_csv(CFG.TABLES / "k10e_primary_ground_validation_post_breach_v2.csv", index=False)
    g[diagnostic].to_csv(CFG.TABLES / "k10e_vegetation_diagnostic_post_breach_v2.csv", index=False)
    print(f"\n-> {gp}")
    print(f"-> {CFG.TABLES / 'k10e_primary_ground_validation_post_breach_v2.csv'}")
    print(f"-> {CFG.TABLES / 'k10e_vegetation_diagnostic_post_breach_v2.csv'}")


if __name__ == "__main__":
    main()
