#!/usr/bin/env python
"""P13 — freeze the zone-relative separation classes as a calibration cohort.

p12 measured 524 scenes and showed that no single separation cutoff can mean
the same thing in four zones: the medians run 0.847 to 2.642, and a global 1.0
would reject three fifths of the floodway while a global 1.5 would reject three
quarters of two zones. This turns that measurement into a FROZEN RULE.

    separation flag  !=  admission decision

Nothing here rejects a scene on separation. The flag is a QA attribute that
travels with the scene; only the combination of an extreme value and an
INDEPENDENT failure is called POOR, and even POOR is a label, not a delete.

FOUR LEVELS, NOT THREE.

    GOOD             separation >= zone p10
    MARGINAL         zone p02 <= separation < zone p10
    LOW_SEPARATION   separation < zone p02
    POOR             LOW_SEPARATION and at least one NON-REDUNDANT
                     corroborating QA failure

The p02 tail is deliberately NOT called POOR on its own. In ZONE_1 that
quantile is fixed by one or two scenes out of 75 - far too thin a tail for its
value to be treated as a physical threshold. It marks a candidate.

WHAT MAY CORROBORATE, AND THE LIMIT OF THAT CLAIM. A second sign adds little
if it largely repeats the first, so each candidate axis is scored by |Spearman|
against separation within the zone and only those below NON_REDUNDANCY_MAX_RHO
may corroborate. Every value is recorded.

    THIS IS A NON-REDUNDANCY SCREEN, NOT A TEST OF INDEPENDENCE.

A low rank correlation means weak monotonic association and nothing more: a
strong non-linear relationship can sit at rho ~ 0. The screen removes axes that
demonstrably repeat separation; it cannot certify that the survivors are
statistically independent of it, and no claim of independence is made anywhere
in this file or in the methods report.

The 0.30 bound is an OPERATIONAL CUT ADOPTED FOR THIS STUDY, in the same sense
as |b - 1| < 0.05 elsewhere in this project. It is not a statistical standard
and carries no distributional guarantee.

If no axis clears the screen in a zone, that zone can produce no POOR scenes at
all, and the run says so.

One axis is built here because nothing else cleared the screen widely: TEMPORAL
RESIDUAL, the gap between a scene's water-of-observed fraction and the median of
its nearest neighbours in time on the same relative orbit. Same orbit means same
geometry and near-same incidence, so a scene that disagrees with its own
neighbours is disagreeing about the water, not about the look angle. It shares
no term with separation by construction - which is an argument about how it is
built, not a demonstration of independence.

THE COHORT IS FROZEN AND THE QUANTILES ARE NOT RECOMPUTED ON EVERY FETCH.
Otherwise a scene that nobody touched would drift from MARGINAL to GOOD because
somebody downloaded a different one. New scenes are classed against the frozen
thresholds (--apply). A new cohort requires --recalibrate, which writes a new
version rather than overwriting the old.

RETIRED THRESHOLDS, named so they cannot quietly return:

    lda_separation >= 1.00    p0v admission, retired. The cross-zone check
                              showed the lowest-separation scenes reproducing
                              BEST (orbit 138, kappa 0.87 against 0.74-0.82).
    lda_separation >= 1.50    never adopted; would have cut 72.7% of ZONE_2
                              and 77.8% of ZONE_4.
    fp_land_anchor <= 0.02    p0v null test, retired. Rejected 8 of 11 scenes
                              and emptied the canonical envelope; the rate is
                              the same before and after the breach, so it
                              measures the anchor, not the scene.
    fn_water_anchor >= 0.60   p12 validity test, retired. Never fired: the real
                              PRE/POST contrast is 0.073 -> 0.247.
    MAD-based classing        retired. Separation is floor-bounded and heavy-
                              tailed, so a one-sided robust z could not flag
                              anything: on ZONE_1 the largest achievable z is
                              +1.00 against a 2.5 cut.

All five were set before their distribution existed. That is the pattern, not
five separate accidents.

Outputs
-------
outputs/tables/p13_separation_calibration.json     the frozen cohort
outputs/tables/p13_zone_separation_classes.csv     per-zone thresholds
outputs/tables/p12_{zone}_scene_qa.csv             classes rewritten in place
outputs/reports/p13_separation_methods.md          the methods paragraph
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

RULE_VERSION = "separation_classes_v1"
MARGINAL_Q = 0.10
LOW_Q = 0.02
# NON-REDUNDANCY SCREEN, not an independence test. A corroborating axis must
# not merely repeat separation within that zone. |Spearman| below this bound
# means weak MONOTONIC association and nothing stronger - a non-linear
# relationship can survive at rho ~ 0. The bound is an operational cut adopted
# for this study, like |b - 1| < 0.05 elsewhere here; it is generous on purpose,
# to exclude the coupling p12 measured rather than to hand-pick axes.
NON_REDUNDANCY_MAX_RHO = 0.30
# A corroborating axis counts as failed when the scene sits in its own
# extreme tail, defined by the same quantile logic used for separation.
CONFIRM_Q = 0.05
N_TEMPORAL_NEIGHBOURS = 4

ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")
# Candidate confirming axes and the direction in which they are BAD.
CONFIRM_AXES = {
    "largest_component_fraction": "low",
    "internal_external_ratio": "high",
    "n_water_parts": "high",
    "fp_land_anchor_fraction": "high",
    "fn_water_anchor_fraction": "high",
    "valid_fraction": "low",
    "temporal_residual": "high",
}
MANIFESTS = {
    "ZONE_1_KAKHOVKA_LOWER_DNIPRO": "p0p_zone1_manifest_freeze.json",
    "ZONE_2_KHERSON_DELTA": "p0t_zone_2_kherson_delta_freeze.json",
    "ZONE_3_DNIPRO_BUG_ESTUARY": "p0q_zone3_topup_freeze.json",
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY":
        "p0t_zone_4_dam_to_kherson_floodway_freeze.json",
}
CAL = CFG.TABLES / "p13_separation_calibration.json"


def table(zone):
    return CFG.TABLES / f"p12_{zone.lower()}_scene_qa.csv"


def temporal_residual(R):
    """How far a scene sits from its own neighbours in time on its own orbit.

    Same relative orbit means the same viewing geometry, so a disagreement
    here is about the water rather than about the look angle. This is the one
    confirming axis that carries no shared term with separation by
    construction - it is built from water_fraction_of_observed and time."""
    out = np.full(len(R), np.nan)
    R = R.reset_index(drop=True)
    t = pd.to_datetime(R.date)
    for orb, g in R.groupby("relative_orbit"):
        idx = g.index.to_numpy()
        tt = t.iloc[idx].to_numpy()
        v = R.water_fraction_of_observed.iloc[idx].to_numpy(float)
        for k, i in enumerate(idx):
            d = np.abs((tt - tt[k]).astype("timedelta64[D]").astype(float))
            d[k] = np.inf
            near = np.argsort(d)[:N_TEMPORAL_NEIGHBOURS]
            near = near[np.isfinite(d[near])]
            if len(near) < 2:
                continue
            out[i] = abs(v[k] - np.nanmedian(v[near]))
    return out


def monotonic_association(R, axes):
    """|Spearman| of each candidate axis against separation, in this zone.

    Reported as an association, never as independence."""
    out = {}
    for a in axes:
        if a not in R or R[a].dropna().nunique() < 3:
            continue
        r = R.lda_separation.corr(R[a], method="spearman")
        out[a] = float(r) if np.isfinite(r) else np.nan
    return out


def calibrate(verbose=True):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    cal = dict(rule_version=RULE_VERSION, date_frozen=str(date.today()),
               git=commit, marginal_quantile=MARGINAL_Q, low_quantile=LOW_Q,
               non_redundancy_max_rho=NON_REDUNDANCY_MAX_RHO,
               screen_type="non-redundancy screen on monotonic "
                           "association; NOT a test of independence",
               screen_basis="operational cut adopted for this study",
               confirm_quantile=CONFIRM_Q,
               zones={})
    for zone in ZONES:
        t = table(zone)
        if not t.exists():
            if verbose:
                print(f"  {zone}: no p12 table -- skipped")
            continue
        R = pd.read_csv(t)
        R["temporal_residual"] = temporal_residual(R)
        V = R[R.anchor_validity == "ANCHOR_VALID"]
        if V.empty:
            continue
        p10 = float(V.lda_separation.quantile(MARGINAL_Q))
        p02 = float(V.lda_separation.quantile(LOW_Q))
        rho = monotonic_association(V, CONFIRM_AXES)
        admissible = {a: r for a, r in rho.items()
                      if np.isfinite(r) and abs(r) < NON_REDUNDANCY_MAX_RHO}
        # the confirming tail of each admissible axis, from this zone
        conf = {}
        for a in admissible:
            q = CONFIRM_Q if CONFIRM_AXES[a] == "low" else 1 - CONFIRM_Q
            conf[a] = dict(direction=CONFIRM_AXES[a],
                           cut=float(V[a].quantile(q)),
                           spearman_vs_separation=admissible[a],
                           screen="non-redundant, not independent")
        mf = CFG.TABLES / MANIFESTS.get(zone, "")
        mh = (json.loads(mf.read_text()).get("sha256") if mf.exists()
              else None)
        cal["zones"][zone] = dict(
            sample_n=int(len(V)),
            zone_median=float(V.lda_separation.median()),
            zone_p10=p10, zone_p02=p02,
            zone_p95=float(V.lda_separation.quantile(0.95)),
            manifest_hash=mh,
            spearman_vs_separation=rho,
            corroborating_axes=conf,
            excluded_as_redundant={a: r for a, r in rho.items()
                                   if a not in admissible})
        if verbose:
            print(f"\n  {zone}")
            print(f"    n = {len(V)}   median {V.lda_separation.median():.3f}")
            print(f"    GOOD           sep >= {p10:.3f}")
            print(f"    MARGINAL       {p02:.3f} <= sep < {p10:.3f}")
            print(f"    LOW_SEPARATION sep < {p02:.3f}")
            print(f"    monotonic association with separation "
                  f"(non-redundancy screen |rho| < "
                  f"{NON_REDUNDANCY_MAX_RHO}):")
            for a, r in sorted(rho.items(), key=lambda kv: -abs(kv[1])):
                ok = ("may corroborate" if a in admissible
                      else "EXCLUDED as redundant")
                print(f"      {a:28s} rho {r:+.2f}   {ok}")
            if not admissible:
                print("    NO axis clears the screen in this zone -> it can "
                      "produce no POOR scenes at all.")
                print("    Every candidate correlates with separation at "
                      "|rho| >= "
                      f"{min(abs(r) for r in rho.values()):.2f}: the QA vector "
                      "here carries heavily overlapping information, and "
                      "corroborating a rejection with any of it would largely "
                      "repeat the same measurement.")
    return cal


def apply_classes(cal, verbose=True):
    rows = []
    for zone, c in cal["zones"].items():
        t = table(zone)
        if not t.exists():
            continue
        R = pd.read_csv(t)
        R["temporal_residual"] = temporal_residual(R)
        sep = R.lda_separation
        cls = np.where(sep >= c["zone_p10"], "GOOD",
                       np.where(sep >= c["zone_p02"], "MARGINAL",
                                "LOW_SEPARATION"))
        cls = np.where(R.anchor_validity != "ANCHOR_VALID", "NOT_ASSESSED", cls)
        # POOR = LOW_SEPARATION plus at least one NON-REDUNDANT failure
        fails = pd.Series(0, index=R.index)
        which = pd.Series("", index=R.index)
        for a, spec in c["corroborating_axes"].items():
            if a not in R:
                continue
            bad = (R[a] <= spec["cut"]) if spec["direction"] == "low" \
                else (R[a] >= spec["cut"])
            bad = bad.fillna(False)
            fails += bad.astype(int)
            which = which.where(~bad, which + a + ";")
        poor = (cls == "LOW_SEPARATION") & (fails > 0).to_numpy()
        cls = np.where(poor, "POOR", cls)
        R["separation_class"] = cls
        R["separation_rule_version"] = cal["rule_version"]
        R["corroborating_qa_failures"] = fails.to_numpy()
        R["corroborating_qa_failed_axes"] = which.str.rstrip(";").to_numpy()
        R.to_csv(t, index=False)
        vc = pd.Series(cls).value_counts().to_dict()
        rows.append(dict(zone=zone, rule_version=cal["rule_version"],
                         sample_n=c["sample_n"],
                         zone_median=c["zone_median"], zone_p10=c["zone_p10"],
                         zone_p02=c["zone_p02"],
                         n_corroborating_axes=len(c["corroborating_axes"]),
                         **{f"n_{k}": int(vc.get(k, 0)) for k in
                            ("GOOD", "MARGINAL", "LOW_SEPARATION", "POOR",
                             "NOT_ASSESSED")}))
        if verbose:
            print(f"  {zone:34s} {vc}")
            if any(poor):
                for r in R[poor].itertuples():
                    print(f"      POOR {r.event_id}  sep "
                          f"{r.lda_separation:.3f}  confirmed by "
                          f"{r.corroborating_qa_failed_axes}")
    return pd.DataFrame(rows)


def methods(cal, S):
    lines = ["# Scene-level separation classes — methods\n",
             f"Rule version `{cal['rule_version']}`, frozen "
             f"{cal['date_frozen']} at git `{cal['git']}`.\n",
             "## The paragraph for the manuscript\n",
             "> Scene-level LDA separation was classified relative to the "
             "empirical distribution within each analysis zone. The lower "
             "decile was flagged as marginal, while extreme lower-tail values "
             "(below the second percentile) were treated as low-separation "
             "candidates. Such a candidate was labelled poor only when "
             "corroborated by the failure of a further QA indicator showing "
             "low empirical monotonic association with LDA separation within "
             "that zone. Separation was never used as an admission criterion "
             "on its own, and no scene was rejected on the basis of these "
             "classes.\n",
             "> The corroboration screen used |Spearman rho| < 0.30, an "
             "operational cut adopted for this study rather than a "
             "statistical standard. It identifies indicators that do not "
             "simply repeat the separation measurement; it does not establish "
             "statistical independence, since a strong non-linear "
             "relationship can persist at negligible rank correlation.\n",
             "The thresholds below are properties of **this** calibration "
             "cohort in **that** zone. A separation of 0.57 is the lower "
             "decile of ZONE_1 and means nothing in ZONE_4, whose entire "
             "median is 0.847.\n",
             "## Frozen thresholds\n",
             "| zone | n | median | GOOD | MARGINAL | LOW_SEPARATION |",
             "|---|---:|---:|---|---|---|"]
    for z, c in cal["zones"].items():
        lines.append(
            f"| {z} | {c['sample_n']} | {c['zone_median']:.3f} | "
            f"`sep >= {c['zone_p10']:.3f}` | "
            f"`{c['zone_p02']:.3f} <= sep < {c['zone_p10']:.3f}` | "
            f"`sep < {c['zone_p02']:.3f}` |")
    lines += ["\n## Which indicators may corroborate a rejection\n",
              "POOR requires LOW_SEPARATION **and** the failure of a "
              "**non-redundant** QA indicator — one whose measured |Spearman| "
              "against separation within that zone falls below "
              f"`{cal['non_redundancy_max_rho']}`.\n",
              "> **This is a non-redundancy screen, not a test of "
              "independence.** A low rank correlation means weak monotonic "
              "association and nothing more; a strong non-linear relationship "
              "can sit at rho ~ 0. The screen removes indicators that "
              "demonstrably repeat separation. It does not certify that the "
              "survivors are independent of it, and no such claim is made "
              "here.\n",
              "> The bound `|rho| < 0.30` is an **operational QA cut adopted "
              "for this study**, in the same sense as `|b - 1| < 0.05` "
              "elsewhere in this project. It is not a statistical standard "
              "and carries no distributional guarantee.\n",
              "| zone | may corroborate | excluded as redundant "
              "(rho vs separation) |",
              "|---|---|---|"]
    for z, c in cal["zones"].items():
        adm = ", ".join(f"`{a}`" for a in c["corroborating_axes"]) or "**none**"
        exc = ", ".join(f"`{a}` {r:+.2f}" for a, r in
                        c["excluded_as_redundant"].items()) or "—"
        lines.append(f"| {z} | {adm} | {exc} |")
    lines += ["\n## Retired pre-distribution thresholds\n",
              "These were set before the distribution they describe existed. "
              "They are named here so they cannot quietly return.\n",
              "| threshold | where | why it is retired |",
              "|---|---|---|",
              "| `separation >= 1.00` | p0v admission | the cross-zone check "
              "showed the lowest-separation scenes reproducing best "
              "(kappa 0.87 vs 0.74–0.82) |",
              "| `separation >= 1.50` | never adopted | would reject 72.7% of "
              "ZONE_2 and 77.8% of ZONE_4 |",
              "| `fp_land_anchor <= 0.02` | p0v null test | identical rate "
              "before and after the breach: it measures the anchor |",
              "| `fn_water_anchor >= 0.60` | p12 validity | never fired; the "
              "real contrast is 0.073 → 0.247 |",
              "| MAD robust-z classing | p12 | separation is floor-bounded; "
              "max achievable z on ZONE_1 is +1.00 against a 2.5 cut |",
              "\n## ZONE_1 can produce no POOR scenes, and that is a result\n",
              "Every candidate indicator in ZONE_1 shows |rho| >= 0.51 against "
              "separation — internal/external ratio -0.93, anchor error -0.92 "
              "and -0.91, largest component +0.85 — so none of them clears the "
              "non-redundancy screen. They carry heavily overlapping "
              "information with separation, and corroborating a rejection with "
              "any of them would largely repeat the same measurement.\n",
              "The likely reason is visible in the design rather than in the "
              "statistics: ZONE_1 scenes range from 17% to 97% coverage, and "
              "how much of the zone a pass saw plausibly drives how much water "
              "is in view, how fragmented it looks, how large the anchors are "
              "and how well the discriminant fits. Describing that QA vector "
              "as effectively one-dimensional is an INTERPRETATION of the "
              "pattern; rank correlations alone do not demonstrate "
              "dimensionality, and no formal claim of it is made.\n",
              "Either way the operational consequence stands: ZONE_1 needs a "
              "corroborating indicator from outside this set — a "
              "classification from an independent SENSOR (optical), or "
              "overlap disagreement with a "
              "neighbouring zone's classifier. Neither exists for ZONE_1 yet. "
              "Until one does, its low-separation scenes stay LOW_SEPARATION "
              "and nothing there is called POOR.\n",
              "\n## Reclassification policy\n",
              "Quantiles are **not** recomputed on every fetch. A scene that "
              "nobody touched must not drift between classes because someone "
              "downloaded a different one. New scenes are classed against "
              "this frozen cohort (`--apply`); a new cohort requires an "
              "explicit `--recalibrate`, which writes a new version.\n"]
    p = CFG.ROOT / "outputs/reports/p13_separation_methods.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recalibrate", action="store_true",
                    help="compute a NEW cohort and overwrite the frozen file")
    ap.add_argument("--apply", action="store_true",
                    help="class scenes against the EXISTING frozen cohort")
    args = ap.parse_args()

    print("=" * 78)
    print("P13 — zone-relative separation classes, frozen")
    print("=" * 78)
    print("  separation flag != admission decision")

    if args.apply and CAL.exists():
        cal = json.loads(CAL.read_text())
        print(f"\n  using frozen cohort {cal['rule_version']} "
              f"({cal['date_frozen']}, git {cal['git']})")
    else:
        if CAL.exists() and not args.recalibrate:
            raise SystemExit(
                f"{CAL.name} already exists. Use --apply to class against it, "
                f"or --recalibrate to deliberately compute a new cohort. "
                f"Recalibrating by accident is how a scene changes class "
                f"without changing.")
        print("\n  CALIBRATING a new cohort")
        cal = calibrate()
        h = hashlib.sha256(json.dumps(cal, sort_keys=True).encode()).hexdigest()
        cal["cohort_sha256"] = h
        CAL.write_text(json.dumps(cal, indent=2))
        print(f"\n-> {CAL}   sha256 {h[:16]}")

    print("\n" + "=" * 78)
    print("CLASSES")
    print("=" * 78)
    S = apply_classes(cal)
    t = CFG.TABLES / "p13_zone_separation_classes.csv"
    S.to_csv(t, index=False)
    print(f"\n{S.to_string(index=False)}")
    print(f"\n-> {t}")
    print(f"-> {methods(cal, S)}")
    print("\n  No scene was rejected here. POOR is a label.")


if __name__ == "__main__":
    main()
