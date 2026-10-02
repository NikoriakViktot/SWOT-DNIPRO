#!/usr/bin/env python
"""HIST 31 — the DEM constraint table: what each shoreline is licensed to do.

This is what hist24's gate consumes instead of a global offset constant. The
semantics are frozen here:

    DIRECT validation is evidence where available,
    not a prerequisite for every stage.

PER-STAGE CLASSES, from hist29's TIN bracketing:

    H3  DIRECT       512 bracketing triangles, 55.1 km of supported isobath
    H2  UNAVAILABLE  zero bracketing triangles; the highest sounding in the
                     survey is 15.32 m, below H2's 15.444 m
    H1  UNAVAILABLE  zero bracketing triangles

WHAT DIRECT DOES NOT MEAN. hist30 measured the absolute bias at H3 as +19.6 m
(S2) and +17.6 m (S1), against a reference whose own horizontal uncertainty is
87.2 m. That rules out a gross common-mode displacement; it is not a fine metric
calibration of the shoreline. DIRECT therefore buys an externally estimated
absolute-validation term, not precision.

WHAT IS EXPLICITLY FORBIDDEN, and enforced by this file having no such column:

  * no offset correction of +19.6, +17.6 or any average of them
  * no transfer of the H3 result to H2 or H1
  * no OFFSET_FROM_TRUE_WATERLINE_M, at 181, at 29-36, or any other value
  * no cross-stage transfer: d_stage stays NaN and unlicensed (hist27 showed
    k(x) is not identifiable at all)

Every shoreline constrains one thing only:  z(x_i) ~ H_WSE(t_i).

sigma_x_reference IS NaN FOR H1 AND H2, NOT ZERO. Zero would claim the absolute
placement is perfectly known there, which is the opposite of the truth. NaN
propagates into sigma_total as a stated lower bound and into quality_flag.

Outputs
-------
outputs/tables/hist31_dem_constraints.csv
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

# sigma_x of one sensor, from the S1-vs-S2 scatter. If the two sensors are
# independent and comparably precise, NMAD(S1-S2)^2 = sigma_S1^2 + sigma_S2^2,
# so each is NMAD/sqrt(2). That is an estimate, not a measurement of either
# sensor alone, and it is labelled as such in the table.
SENSOR_SPLIT = np.sqrt(2.0)


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("HIST 31 — DEM constraint table")
    print("=" * 78)
    print(f"  git {commit}")
    print("  DIRECT validation is evidence where available, not a")
    print("  prerequisite for every stage.")

    sup = pd.read_csv(CFG.TABLES / "hist29_sounding_support.csv").set_index(
        "contour_id")
    con = pd.read_csv(CFG.TABLES / "hist26_continuous_contours.csv").set_index(
        "contour_id")
    adm = pd.read_csv(CFG.TABLES / "gate7c2b_admission.csv")
    bias = pd.read_csv(CFG.TABLES / "hist30_absolute_bias.csv")

    sig_ref_h3 = float(bias.sigma_x_bathy_m.dropna().iloc[0])

    rows = []
    for cid in ("H1", "H2", "H3"):
        s = sup.loc[cid]
        c = con.loc[cid]
        a = adm[(adm.target == cid)
                & adm.shoreline_constraint_usable.astype(bool)]
        # per-stage sensor scatter, from the admitted scenes' S1-vs-S2 NMAD
        nmad = float(a.sigma_x_shoreline_m.median()) if len(a) else np.nan
        sig_sensor = nmad / SENSOR_SPLIT if np.isfinite(nmad) else np.nan
        direct = s.absolute_validation == "DIRECT"
        sig_ref = sig_ref_h3 if direct else np.nan
        if direct:
            sig_tot = float(np.hypot(sig_sensor, sig_ref))
            flag = "ABSOLUTE_COMMON_MODE_EXCLUDED"
            note = ("gross common-mode displacement ruled out at "
                    f"sigma_x_reference = {sig_ref:.0f} m; not a fine "
                    "calibration")
        else:
            # sigma_total is a LOWER BOUND: the absolute component is unknown,
            # not zero, so it cannot be added and must not be omitted silently
            sig_tot = float(sig_sensor)
            flag = "ABSOLUTE_UNVALIDATED"
            note = ("no bracketing soundings at this stage; sigma_total is a "
                    "LOWER BOUND, the absolute term is unknown")
        rows.append(dict(
            constraint_source="S2_continuous_contour_hist26",
            stage_target=cid,
            stage_m=float(c.H_evrf2019_m),
            stage_group_spread_m=float(c.level_spread_m),
            n_admitted_s1_scenes=int(len(a)),
            absolute_validation_class=str(s.absolute_validation),
            absolute_validation_supported_length_km=float(
                s.supported_isobath_length_km),
            n_bracketing_triangles=int(s.n_bracketing_triangles),
            sigma_x_sensor_m=sig_sensor,
            sigma_x_reference_m=sig_ref,
            sigma_total_m=sig_tot,
            sigma_total_is_lower_bound=not direct,
            quality_flag=flag,
            cross_stage_transfer_licensed=False,
            offset_correction_applied_m=0.0,
            note=note))

    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "hist31_dem_constraints.csv", index=False)

    show = ["stage_target", "stage_m", "absolute_validation_class",
            "absolute_validation_supported_length_km", "sigma_x_sensor_m",
            "sigma_x_reference_m", "sigma_total_m",
            "sigma_total_is_lower_bound", "quality_flag"]
    print()
    print(R[show].to_string(index=False, float_format=lambda v: f"{v:9.2f}",
                            na_rep="NaN"))
    print(f"\n-> {CFG.TABLES / 'hist31_dem_constraints.csv'}")

    # self-checks, because the whole point is that these cannot drift
    assert (R.offset_correction_applied_m == 0).all(), \
        "an offset correction has been introduced"
    assert not R.cross_stage_transfer_licensed.any(), \
        "cross-stage transfer has been licensed"
    bad = R[(R.absolute_validation_class == "UNAVAILABLE")
            & R.sigma_x_reference_m.notna()]
    assert bad.empty, "sigma_x_reference must be NaN where validation is UNAVAILABLE"
    print("\n  checks: no offset correction, no cross-stage transfer,")
    print("          sigma_x_reference is NaN (not 0) where UNAVAILABLE")
    print("\n  Each shoreline constrains z(x_i) ~ H_WSE(t_i), nothing more.")
    print("\nSTOP. hist24's gate now reads this table.")


if __name__ == "__main__":
    main()
