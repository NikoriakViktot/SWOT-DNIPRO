#!/usr/bin/env python
"""P12 — the scene-QA distribution for every zone, on its own correct grid.

This is the measurement that has to exist before any admission rule is frozen.
p0v learned the hard way what happens otherwise: a separation cutoff of 1.00,
chosen from a gap that looked clean in a sorted list of eleven numbers, would
have discarded the only acquisition of the breach day - and the cross-zone check
then showed those very scenes reproducing BEST. A land-anchor cutoff of 0.02,
set before the distribution existed, rejected 8 of 11 scenes and emptied the
canonical envelope.

So: measure first, per zone, and only then class.

ONE CUTOFF FOR FOUR ZONES WOULD BE AN ARTEFACT. The reservoir, the delta, the
estuary and the floodway have different SAR physics - a wind-roughened liman and
a reed-choked delta do not produce the same class separation as an impounded
pool, and a number that is unremarkable in one is an outlier in another. Every
class here is therefore relative to the zone's own distribution: a one-sided
robust z against that zone's median, with the MAD as the scale.

WHAT IS MEASURED, per scene:

    coverage_fraction            observed area / domain area
    valid_fraction               finite pixels / grid
    water_fraction_of_observed   the only water number comparable across scenes
    lda_separation               class contrast along the discriminant
    largest_component_fraction   is the water one body or ten thousand specks
    internal_external_ratio      hole area / water area: how sponge-like
    fp_land_anchor_fraction      water called inside the dry anchor
    fn_water_anchor_fraction     dry called inside the wet anchor

THE WATER ANCHOR DEGRADES AFTER THE BREACH, AND THAT IS REPORTED, NOT GATED.
The anchors come from the registry water domain, which in ZONE_1 includes the
Kakhovka pool, and after 2023-06-06 that pool is dry ground. Measured:

    ZONE_1 fn_water_anchor    PRE  median 0.073     POST  median 0.247

A real 3.4x degradation - and not one scene of 75 crosses the 0.60 cut this
file originally carried, so that cut never fired and the claim it justified was
false. It is gone. What remains is a validity test for the genuinely undefined
case only: too few anchor pixels to fit a discriminant at all.

AND SEPARATION IS ASSOCIATED WITH COVERAGE. On ZONE_1, Spearman between
lda_separation and coverage_fraction is +0.52, and against anchor size +0.52:
orbits that see 17-29% of the zone score 0.86-1.07, orbits that see 92-97%
score 3.07-4.57. So part of what separation measures is how much of the zone
the scene saw - which is one more reason it cannot carry an admission decision
by itself. Stated as an association: a rank correlation bounds monotonic
association only, and says nothing about independence in either direction.

Nothing is admitted or rejected here. p12 writes distributions and classes; the
zone-specific admission rules are frozen from these tables afterwards.

Outputs
-------
outputs/tables/p12_{zone}_scene_qa.csv
outputs/tables/p12_zone_qa_summary.csv
outputs/figures/phase19_20/png/p12_zone_scene_qa.png
"""
from __future__ import annotations

import argparse
import subprocess
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
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin
from scipy import ndimage

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from p0r_zone2_flood_envelope import water_mask, CELL

INK, BLUE, RED, SAND = "#1a2228", "#236f8c", "#c1402a", "#b07d27"
FIGDIR = CFG.FIG / "phase19_20" / "png"
BREACH = pd.Timestamp("2023-06-06")

ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")

# CLASSES ARE QUANTILES OF THE ZONE'S OWN DISTRIBUTION, and the first version
# of this file got that wrong in a way worth keeping on the record.
#
# It used a one-sided robust z (MAD) and classed everything GOOD in ZONE_1 -
# not because every scene was good, but because a MAD rule CANNOT bite on that
# distribution. ZONE_1 separation runs p05 = 0.29 to p95 = 10.59 with a median
# of 2.64, so the MAD scale is 2.63 - as wide as the centre itself - and the
# variable is bounded below by zero. The largest z any scene can reach is
# +1.00, against a MARGINAL cut at 2.5. Nothing could ever be flagged.
#
# Quantiles have no such failure mode, and they are what "lower tail" means
# anyway: MARGINAL below the zone's 10th percentile, POOR below its 2nd AND
# with an independent sign of degradation. The cut VALUE is reported per zone
# so the reader sees what the rule actually did.
MARGINAL_Q = 0.10
POOR_Q = 0.02

# ANCHOR VALIDITY IS TESTED ONLY WHERE IT IS GENUINELY UNDEFINED. The first
# version also carried ANCHOR_INVALID_FN = 0.60 - a wet anchor coming back
# mostly dry - on the reasoning that the drained Kakhovka pool stops being
# water after 2023-06-06. Measured, the effect is real and much smaller than
# the number I picked before looking:
#
#     ZONE_1 fn_water_anchor   PRE  median 0.073      POST  median 0.247
#
# A 3.4x rise, and not one scene of 75 reaches 0.60. So the threshold never
# fired and the claim that post-breach scenes are labelled ANCHOR_INVALID was
# false as written. That is the third threshold this campaign that was set
# before its distribution existed, so it is gone: fn_water_anchor_fraction is
# reported, the PRE/POST contrast is printed, and the only hard validity test
# left is the one that is genuinely undefined - too few anchor pixels to fit a
# discriminant at all.
ANCHOR_MIN_PX = 2000


def robust_z(x):
    """Deviation BELOW the median in MAD units. One-sided on purpose: a scene
    that separates unusually well is not a problem."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros_like(x)
    return (med - x) / mad


def topology(m):
    """One body or ten thousand specks, and how full of holes it is."""
    px = CELL ** 2 / 1e6
    lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
    if not n:
        return 0, 0.0, np.nan
    sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * px
    filled = ndimage.binary_fill_holes(m)
    holes = float((filled & ~m).sum()) * px
    water = float(m.sum()) * px
    return int(n), float(sz.max() / sz.sum()), holes / water if water else np.nan


def zone_setup(zone):
    g = SD.load_utm(zone)
    gr = SD.build_grid(g, CELL, what=f"{zone} QA grid")
    tr = from_origin(float(gr["gx"][0]), float(gr["gy"][-1]), CELL, CELL)
    shp = (gr["ny"], gr["nx"])
    inside = rio_rasterize([(g, 1)], out_shape=shp, transform=tr, fill=0,
                           dtype="uint8").astype(bool)
    w = SD.load_utm("dnipro_water_domain").intersection(g)
    aw = rio_rasterize([(w.buffer(-200.0), 1)], out_shape=shp, transform=tr,
                       fill=0, dtype="uint8").astype(bool)
    al = rio_rasterize([(g.difference(w.buffer(6000.0)), 1)], out_shape=shp,
                       transform=tr, fill=0, dtype="uint8").astype(bool)
    return g, inside, aw, al, shp


def classify(R, quiet=False):
    """Zone-relative classes, from this zone's own quantiles."""
    R = R.copy()
    invalid = ((R.n_water_anchor_px < ANCHOR_MIN_PX) |
               (R.n_land_anchor_px < ANCHOR_MIN_PX) |
               R.lda_separation.isna())
    ok = ~invalid
    cuts = {}
    for col, out in (("lda_separation", "separation"),
                     ("coverage_fraction", "coverage"),
                     ("largest_component_fraction", "topology")):
        v = R[col].where(ok)
        qm = float(v.quantile(MARGINAL_Q)) if ok.any() else np.nan
        qp = float(v.quantile(POOR_Q)) if ok.any() else np.nan
        cuts[out] = (qm, qp)
        R[out + "_pctile"] = v.rank(pct=True)
        R[out + "_class"] = np.where(
            invalid, "NOT_ASSESSED",
            np.where(v <= qp, "POOR",
                     np.where(v <= qm, "MARGINAL", "GOOD")))
    # POOR needs a second, independent sign - a scene is not rejected for being
    # in the lower tail of one variable alone.
    lone = ((R.separation_class == "POOR") &
            (R.topology_class == "GOOD") & (R.coverage_class == "GOOD"))
    R.loc[lone, "separation_class"] = "MARGINAL"
    R["anchor_validity"] = np.where(invalid, "ANCHOR_UNDEFINED", "ANCHOR_VALID")
    R.attrs["cuts"] = cuts
    if not quiet:
        for k, (qm, qp) in cuts.items():
            print(f"    {k:12s} MARGINAL <= {qm:7.3f} (p{100*MARGINAL_Q:.0f})   "
                  f"POOR <= {qp:7.3f} (p{100*POOR_Q:.0f})")
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reclass", action="store_true",
                    help="re-apply the classes to the saved tables, without "
                         "recomputing a single mask")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P12 — scene QA distributions, per zone, on the registry grid")
    print("=" * 78)
    print(f"  git {commit}")
    print(f"  classes are ZONE-RELATIVE QUANTILES: MARGINAL at or below "
          f"p{100*MARGINAL_Q:.0f}, POOR at or below p{100*POOR_Q:.0f} and only "
          f"with a second sign")

    summary = []
    if args.reclass:
        # The metrics cost an hour of mask computation; the classes cost
        # nothing. Separating them means a rule can be revised without paying
        # for the measurement again - which matters, because the first rule
        # here had to be thrown away.
        for zone in ([args.zone] if args.zone else ZONES):
            t = CFG.TABLES / f"p12_{zone.lower()}_scene_qa.csv"
            if not t.exists():
                continue
            R = pd.read_csv(t)
            drop = [c for c in R.columns
                    if c.endswith(("_class", "_robust_z", "_pctile"))
                    or c == "anchor_validity"]
            print(f"\n{'=' * 78}\n{zone}\n{'=' * 78}")
            R = classify(R.drop(columns=drop))
            R.to_csv(t, index=False)
            V = R[R.anchor_validity == "ANCHOR_VALID"]
            print(f"  {len(R)} scenes  separation median "
                  f"{V.lda_separation.median():.3f}")
            print(f"  classes: separation "
                  f"{dict(V.separation_class.value_counts())}")
            print(f"           coverage   "
                  f"{dict(V.coverage_class.value_counts())}")
            print(f"           topology   "
                  f"{dict(V.topology_class.value_counts())}")
            summary.append(dict(
                zone=zone, n_scenes=len(R),
                n_anchor_undefined=int((R.anchor_validity ==
                                        "ANCHOR_UNDEFINED").sum()),
                sep_median=float(V.lda_separation.median()),
                sep_p10_cut=float(V.lda_separation.quantile(MARGINAL_Q)),
                sep_p02_cut=float(V.lda_separation.quantile(POOR_Q)),
                sep_p95=float(V.lda_separation.quantile(.95)),
                sep_vs_coverage_spearman=float(R.lda_separation.corr(
                    R.coverage_fraction, method="spearman")),
                coverage_median=float(V.coverage_fraction.median()),
                n_sep_marginal=int((V.separation_class == "MARGINAL").sum()),
                n_sep_poor=int((V.separation_class == "POOR").sum())))
        if summary:
            S = pd.DataFrame(summary)
            S.to_csv(CFG.TABLES / "p12_zone_qa_summary.csv", index=False)
            print("\n" + "=" * 78)
            print("ONE CUTOFF WOULD NOT HAVE FITTED ALL FOUR")
            print("=" * 78)
            print(S.to_string(index=False,
                              float_format=lambda v: f"{v:8.3f}"))
            print(f"\n  -> {CFG.TABLES / 'p12_zone_qa_summary.csv'}")
            figure()
        return

    for zone in ([args.zone] if args.zone else ZONES):
        cache = CFG.S1_CACHE / zone
        files = sorted(cache.glob("*.npz"))
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"\n  {zone}: no cached events"); continue
        print(f"\n{'=' * 78}\n{zone}\n{'=' * 78}")
        g, inside, aw, al, shp = zone_setup(zone)
        px = CELL ** 2 / 1e6
        dom = float(inside.sum()) * px
        print(f"  {dom:,.0f} km2 on grid {shp[1]}x{shp[0]}, {len(files)} events")
        print(f"  anchors: water {aw.sum():,} px, land {al.sum():,} px")

        rows = []
        for i, f in enumerate(files, 1):
            eid = f.stem
            try:
                z = np.load(f)
                vv, vh, cov = z["vv"], z["vh"], z["cov"]
            except Exception as ex:
                print(f"    {eid}: unreadable ({type(ex).__name__})"); continue
            valid = cov & inside
            obs = float(valid.sum()) * px
            if obs <= 0:
                del z, vv, vh, cov, valid
                continue
            m, sep, km2, naw = water_mask(vv, vh, valid, aw, al)
            nal = int((al & valid).sum())
            if m is None:
                rows.append(dict(event_id=eid, date=eid[:10],
                                 relative_orbit=int(eid.split("_orb")[1]
                                                    .split("_")[0]),
                                 coverage_fraction=obs / dom,
                                 valid_fraction=float(cov.mean()),
                                 n_water_anchor_px=int(naw),
                                 n_land_anchor_px=nal,
                                 lda_separation=np.nan))
                del z, vv, vh, cov, valid
                continue
            nparts, largest, ier = topology(m)
            wa, la = aw & valid, al & valid
            rows.append(dict(
                event_id=eid, date=eid[:10],
                relative_orbit=int(eid.split("_orb")[1].split("_")[0]),
                orbit_state=eid.split("_")[-1],
                phase="PRE" if pd.Timestamp(eid[:10]) < BREACH else "POST",
                observed_area_km2=obs, coverage_fraction=obs / dom,
                valid_fraction=float(cov.mean()),
                water_area_km2=km2,
                water_fraction_of_observed=km2 / obs if obs else np.nan,
                lda_separation=sep,
                largest_component_fraction=largest,
                internal_external_ratio=ier, n_water_parts=nparts,
                fp_land_anchor_fraction=float((m & la).sum() /
                                              max(la.sum(), 1)),
                fn_water_anchor_fraction=float((~m & wa).sum() /
                                               max(wa.sum(), 1)),
                n_water_anchor_px=int(naw), n_land_anchor_px=nal))
            if i % 25 == 0 or i == len(files):
                print(f"    {i}/{len(files)}", flush=True)
            del z, vv, vh, cov, valid, m

        if not rows:
            print("    nothing measurable"); continue
        R = classify(pd.DataFrame(rows))
        t = CFG.TABLES / f"p12_{zone.lower()}_scene_qa.csv"
        R.to_csv(t, index=False)

        V = R[R.anchor_validity == "ANCHOR_VALID"]
        n_inv = int((R.anchor_validity == "ANCHOR_UNDEFINED").sum())
        print(f"\n  {len(R)} scenes, {n_inv} with no fittable anchor")
        if "phase" in R and R.phase.nunique() > 1:
            g = R.groupby("phase")
            print("  anchor degradation, PRE vs POST (reported, never a gate):")
            for ph, s_ in g:
                print(f"    {ph:5s} n={len(s_):3d}  fn_water_anchor med "
                      f"{s_.fn_water_anchor_fraction.median():.3f}  "
                      f"fp_land med {s_.fp_land_anchor_fraction.median():.3f}")
        if V.empty:
            print("    no scene has a valid anchor; nothing can be classed")
            continue
        print(f"\n  distributions over the {len(V)} anchor-valid scenes:")
        for col in ("coverage_fraction", "valid_fraction",
                    "water_fraction_of_observed", "lda_separation",
                    "largest_component_fraction", "internal_external_ratio",
                    "fp_land_anchor_fraction", "fn_water_anchor_fraction"):
            if col not in V or V[col].isna().all():
                continue
            q = V[col].quantile([.05, .25, .5, .75, .95])
            print(f"    {col:28s} p05 {q[.05]:7.3f}  p25 {q[.25]:7.3f}  "
                  f"med {q[.5]:7.3f}  p75 {q[.75]:7.3f}  p95 {q[.95]:7.3f}")
        sc = R.lda_separation.corr(R.coverage_fraction, method="spearman")
        sa = R.lda_separation.corr(R.n_water_anchor_px, method="spearman")
        print(f"\n  separation is associated with how much was seen: "
              f"Spearman vs coverage {sc:+.2f}, vs anchor size {sa:+.2f} "
              f"(association, not a claim about independence)")
        print(f"\n  classes: separation {dict(V.separation_class.value_counts())}")
        print(f"           coverage   {dict(V.coverage_class.value_counts())}")
        print(f"           topology   {dict(V.topology_class.value_counts())}")
        print(f"\n  by orbit (water normalised on OBSERVED area):")
        for orb, s in V.groupby("relative_orbit"):
            print(f"    orbit {orb:4d}  n={len(s):3d}  coverage med "
                  f"{100*s.coverage_fraction.median():5.1f}%  "
                  f"water-of-obs {100*s.water_fraction_of_observed.median():5.1f}%"
                  f"  sep med {s.lda_separation.median():5.2f}")
        print(f"  -> {t}")

        summary.append(dict(
            zone=zone, n_scenes=len(R), n_anchor_invalid=n_inv,
            sep_median=float(V.lda_separation.median()),
            sep_p10_cut=float(V.lda_separation.quantile(MARGINAL_Q)),
            sep_p02_cut=float(V.lda_separation.quantile(POOR_Q)),
            sep_p95=float(V.lda_separation.quantile(.95)),
            sep_vs_coverage_spearman=float(
                R.lda_separation.corr(R.coverage_fraction, method="spearman")),
            coverage_median=float(V.coverage_fraction.median()),
            n_sep_marginal=int((V.separation_class == "MARGINAL").sum()),
            n_sep_poor=int((V.separation_class == "POOR").sum())))

    if not summary:
        raise SystemExit("no zone produced a table")
    S = pd.DataFrame(summary)
    S.to_csv(CFG.TABLES / "p12_zone_qa_summary.csv", index=False)
    print("\n" + "=" * 78)
    print("ONE CUTOFF WOULD NOT HAVE FITTED ALL FOUR")
    print("=" * 78)
    print(S.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\n  -> {CFG.TABLES / 'p12_zone_qa_summary.csv'}")
    figure()


def figure():
    fs = sorted(CFG.TABLES.glob("p12_zone_*_scene_qa.csv"))
    if not fs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    cols = [("lda_separation", "LDA separation"),
            ("coverage_fraction", "coverage of the zone"),
            ("water_fraction_of_observed", "water / observed"),
            ("largest_component_fraction", "largest component / water")]
    for ax, (col, lab) in zip(axes, cols):
        data, names = [], []
        for f in fs:
            R = pd.read_csv(f)
            V = R[R.anchor_validity == "ANCHOR_VALID"]
            if col not in V or V[col].dropna().empty:
                continue
            data.append(V[col].dropna().values)
            names.append(f.stem.replace("p12_zone_", "Z")
                          .replace("_scene_qa", "").split("_")[0])
        if not data:
            continue
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        for b in bp["boxes"]:
            b.set_facecolor(BLUE); b.set_alpha(.45)
        ax.set_title(lab, color=INK, fontsize=11)
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("p12 · scene QA per zone — every distribution is different, "
                 "which is why no cutoff is global", color=INK, fontsize=12)
    fig.tight_layout()
    p = FIGDIR / "p12_zone_scene_qa.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
