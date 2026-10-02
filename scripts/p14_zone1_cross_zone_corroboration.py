#!/usr/bin/env python
"""P14 — can ZONE_4's classifier corroborate a ZONE_1 low-separation scene?

p13 left ZONE_1 with no usable corroborating indicator: every candidate axis
there sits at |Spearman| >= 0.51 against separation, so none clears the
non-redundancy screen and the zone can produce no POOR scenes. The cheapest
remaining candidate is the one the zone overlaps were built for - 4,167 km2
shared with ZONE_4, already fetched on both sides.

WHAT THIS IS, AND WHAT IT IS NOT. It is not independent truth. It is a SECOND
ZONE-SPECIFIC CLASSIFIER READING THE SAME SAR SCENE: same granule, same
radiometry, different anchors, different domain, different fitted discriminant,
different grid. Everything except the backscatter differs. That makes it
external to ZONE_1's configuration, which is exactly what p13 needs - and it
makes any agreement a statement about configuration, not about accuracy.

ONLY IDENTICAL OBSERVATIONS ARE COMPARED:

    same date, same relative orbit, same ASC/DES   - one event_id, both caches
    inside ZONE_1 n ZONE_4                          - the shared geometry
    inside valid_Z1 n valid_Z4                      - both actually saw it

The last is not a detail. A pixel one side did not observe must never enter as
the other side's "dry": that is the error this whole campaign has been about,
and counting it would manufacture disagreement out of a footprint edge.

    D = A(M_Z1 != M_Z4) / A(valid_Z1 n valid_Z4)

TWO WEAK CLASSIFIERS DISAGREEING CORROBORATES NOTHING. A ZONE_4 scene may only
corroborate a ZONE_1 candidate if the ZONE_4 side is itself sound: not
LOW_SEPARATION, not POOR, and with enough valid support in the overlap to mean
anything. Scenes failing that are measured and reported, never used as evidence.

NO DISAGREEMENT THRESHOLD IS SET IN ADVANCE. The baseline distribution of D is
measured over matched scenes that are GOOD or MARGINAL on both sides, and only
then can "unusual disagreement" be defined - the same discipline that separation
itself went through, after five thresholds set before their distributions
existed had to be retired.

AND THE AXIS ONLY COUNTS IF IT CLEARS THE SAME SCREEN. Once D exists, it is
scored against ZONE_1 separation with the frozen operational cut
|rho| < 0.30 from p13. If D largely repeats separation it cannot promote
LOW_SEPARATION to POOR either, and that is a result rather than a failure.

Outputs
-------
outputs/tables/p14_zone1_cross_zone_corroboration.csv
outputs/tables/p14_zone1_corroboration_verdict.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import hist25b_gate6_event_qualification as G6
from p0r_zone2_flood_envelope import water_mask, CELL

Z1 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
Z4 = "ZONE_4_DAM_TO_KHERSON_FLOODWAY"
# Scenes fetched only to answer this question live in their OWN cache. Putting
# them in the ZONE_4 cache would create products no ZONE_4 manifest asked for,
# and p0k's event-key audit would rightly call them orphans.
XCACHE = "ZONE_4_grid_crosscheck_for_zone1"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
MIN_COMMON_KM2 = 200.0
SCREEN_MAX_RHO = 0.30          # frozen in p13; not re-derived here
CAL = CFG.TABLES / "p13_separation_calibration.json"
UNSOUND = ("LOW_SEPARATION", "POOR", "NOT_ASSESSED")


def http_json(url, payload, tries=6, timeout=180):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as ex:
            last = ex
            time.sleep(min(60, 4 * 2 ** a))
    raise RuntimeError(f"{type(last).__name__}: {last}")


def setup(zone):
    g = SD.load_utm(zone)
    gr = SD.build_grid(g, CELL, what=f"{zone} cross-check grid")
    x0, y1 = float(gr["gx"][0]), float(gr["gy"][-1])
    tr = from_origin(x0, y1, CELL, CELL)
    shp = (gr["ny"], gr["nx"])
    inside = rio_rasterize([(g, 1)], out_shape=shp, transform=tr, fill=0,
                           dtype="uint8").astype(bool)
    w = SD.load_utm("dnipro_water_domain").intersection(g)
    aw = rio_rasterize([(w.buffer(-200.0), 1)], out_shape=shp, transform=tr,
                       fill=0, dtype="uint8").astype(bool)
    al = rio_rasterize([(g.difference(w.buffer(6000.0)), 1)], out_shape=shp,
                       transform=tr, fill=0, dtype="uint8").astype(bool)
    return dict(geom=g, x0=x0, y0=float(gr["gy"][0]),
                x1=float(gr["gx"][-1]), y1=y1, nx=gr["nx"], ny=gr["ny"],
                tr=tr, shp=shp, inside=inside, aw=aw, al=al)


def window(A, B):
    """Integer index windows onto the shared rectangle, asserted aligned."""
    for g in (A, B):
        assert abs(g["x0"] / CELL - round(g["x0"] / CELL)) < 1e-6
        assert abs(g["y1"] / CELL - round(g["y1"] / CELL)) < 1e-6
    x0 = max(A["x0"], B["x0"])
    x1 = min(A["x0"] + A["shp"][1] * CELL, B["x0"] + B["shp"][1] * CELL)
    y1 = min(A["y1"], B["y1"])
    y0 = max(A["y1"] - A["shp"][0] * CELL, B["y1"] - B["shp"][0] * CELL)
    if x1 <= x0 or y1 <= y0:
        return None
    out = []
    for g in (A, B):
        c0 = int(round((x0 - g["x0"]) / CELL))
        r0 = int(round((g["y1"] - y1) / CELL))
        out.append((slice(r0, r0 + int(round((y1 - y0) / CELL))),
                    slice(c0, c0 + int(round((x1 - x0) / CELL)))))
    return out


def kappa(a, b):
    if not a.size:
        return np.nan
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def fetch_on_z4_grid(event_ids, G, cache):
    """Build the ZONE_4-grid product for scenes the ZONE_4 design never asked
    for, into their own cache."""
    cache.mkdir(parents=True, exist_ok=True)
    G6.CACHE = cache
    tok = json.loads(urllib.request.urlopen(G6.SAS, timeout=90).read())["token"]
    t0 = time.time()
    bbox = [round(v, 5) for v in
            gpd.GeoSeries([G["geom"]], crs=32636).to_crs(4326).iloc[0].bounds]
    done = 0
    for i, eid in enumerate(event_ids, 1):
        if (cache / f"{eid}.npz").exists():
            done += 1
            continue
        if time.time() - t0 > 1800:
            tok = json.loads(urllib.request.urlopen(
                G6.SAS, timeout=90).read())["token"]
            t0 = time.time()
        date, orb = eid[:10], int(eid.split("_orb")[1].split("_")[0])
        q = {"collections": ["sentinel-1-rtc"], "bbox": bbox, "limit": 100,
             "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z"}
        try:
            feats = http_json(STAC, q)["features"]
        except Exception as ex:
            print(f"    [{i}/{len(event_ids)}] {eid}: STAC "
                  f"{type(ex).__name__}"); continue
        items = [f for f in feats
                 if f["properties"].get("sat:relative_orbit") == orb]
        if not items:
            print(f"    [{i}/{len(event_ids)}] {eid}: no granule on that orbit")
            continue
        print(f"    [{i}/{len(event_ids)}] {eid}: {len(items)} granule(s)",
              flush=True)
        try:
            vv, vh, cov, seam = G6.build_event(eid, items, tok, G)
        except Exception as ex:
            print(f"      BUILD FAILED {type(ex).__name__}"); continue
        frac = float((cov & G["inside"]).mean())
        if frac < 0.01:
            (cache / f"{eid}.npz").unlink(missing_ok=True)
            print(f"      EMPTY ({100*frac:.1f}%) -- deleted"); continue
        done += 1
        del vv, vh, cov
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-missing", action="store_true",
                    help="build ZONE_4-grid products for the ZONE_1 candidate "
                         "scenes the ZONE_4 design never sampled")
    args = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P14 — ZONE_1 corroboration from the ZONE_4 classifier")
    print("=" * 78)
    print(f"  git {commit}")
    print("  external to ZONE_1's configuration, NOT independent truth")

    R1 = pd.read_csv(CFG.TABLES / f"p12_{Z1.lower()}_scene_qa.csv")
    R4 = pd.read_csv(CFG.TABLES / f"p12_{Z4.lower()}_scene_qa.csv")
    cand = R1[R1.separation_class.isin(("LOW_SEPARATION", "MARGINAL"))]
    print(f"\n  ZONE_1 classes {dict(R1.separation_class.value_counts())}")
    print(f"  candidates needing corroboration: {len(cand)} "
          f"({int((cand.separation_class == 'LOW_SEPARATION').sum())} "
          f"LOW_SEPARATION)")

    A, B = setup(Z1), setup(Z4)
    ov = A["geom"].intersection(B["geom"])
    print(f"  ZONE_1 n ZONE_4 = {ov.area/1e6:,.0f} km2")

    cache1 = CFG.S1_CACHE / Z1
    cache4 = CFG.S1_CACHE / Z4
    cx = CFG.S1_CACHE / XCACHE
    have1 = {p.stem for p in cache1.glob("*.npz")}
    have4 = {p.stem for p in cache4.glob("*.npz")}
    havex = {p.stem for p in cx.glob("*.npz")} if cx.exists() else set()
    matched = sorted(have1 & (have4 | havex))
    print(f"\n  ZONE_1 cache {len(have1)}, ZONE_4 cache {len(have4)}, "
          f"cross-check cache {len(havex)}")
    print(f"  matched event_ids (same date, orbit and pass direction): "
          f"{len(matched)}")

    reach = sorted(set(cand.event_id) & set(matched))
    print(f"\n  of the {len(cand)} ZONE_1 candidates, {len(reach)} are matched")
    if not reach:
        print("  -> AS SAMPLED, THIS TEST CANNOT REACH A SINGLE CANDIDATE.")
        print("     The two manifests were stratified independently (ZONE_1 by")
        print("     stage x regime x orbit, ZONE_4 by time-since-breach x")
        print("     orbit) and intersect almost only in June 2023, where both")
        print("     designs happen to sample. Every matched scene is GOOD on")
        print("     both sides, so the matched set measures a baseline and")
        print("     nothing else.")
        missing = sorted(set(cand.event_id) - set(matched))
        print(f"     {len(missing)} candidate scene(s) would have to be built "
              f"on the ZONE_4 grid:")
        for e in missing:
            r = cand[cand.event_id == e].iloc[0]
            print(f"       {e:26s} {r.separation_class:15s} "
                  f"sep {r.lda_separation:.3f}")
        if args.fetch_missing:
            print(f"\n  --fetch-missing: building {len(missing)} product(s) "
                  f"into {cx}")
            n = fetch_on_z4_grid(missing, B, cx)
            print(f"  {n} of {len(missing)} available; rerun to use them")
            havex = {p.stem for p in cx.glob("*.npz")}
            matched = sorted(have1 & (have4 | havex))
            reach = sorted(set(cand.event_id) & set(matched))
            print(f"  candidates now matched: {len(reach)}")
        else:
            print("\n  rerun with --fetch-missing to build them "
                  "(~8 events, about 1 GB)")

    if not matched:
        raise SystemExit("no matched scene at all")

    win = window(A, B)
    wa, wb = win
    both = A["inside"][wa] & B["inside"][wb]
    px = CELL ** 2 / 1e6
    print(f"\n  shared rectangle {both.shape}, both domains "
          f"{both.sum()*px:,.0f} km2")

    cls1 = dict(zip(R1.event_id, R1.separation_class))
    cls4 = dict(zip(R4.event_id, R4.separation_class))
    # A cross-check scene has no p13 class because no ZONE_4 manifest asked
    # for it. Leaving it unlabelled silently admitted it to the "sound on both
    # sides" baseline - which is the thing this test exists to prevent. The
    # FROZEN ZONE_4 thresholds are exactly what to apply: that is what a frozen
    # cohort is for, and it is why p13 refuses to recalibrate by accident.
    zc = json.loads(CAL.read_text())["zones"][Z4]
    z4_p10, z4_p02 = zc["zone_p10"], zc["zone_p02"]
    print(f"\n  cross-check scenes are classed against the FROZEN ZONE_4 "
          f"thresholds: GOOD >= {z4_p10:.3f}, LOW < {z4_p02:.3f}")

    def class_z4(eid, sep):
        if eid in cls4:
            return cls4[eid], "manifest"
        c = ("GOOD" if sep >= z4_p10
             else "MARGINAL" if sep >= z4_p02 else "LOW_SEPARATION")
        return c, "crosscheck_frozen_thresholds"
    rows = []
    for eid in matched:
        f4 = (cache4 / f"{eid}.npz") if (cache4 / f"{eid}.npz").exists() \
            else (cx / f"{eid}.npz")
        z1, z4 = np.load(cache1 / f"{eid}.npz"), np.load(f4)
        m1, s1, _, _ = water_mask(z1["vv"], z1["vh"],
                                  z1["cov"] & A["inside"], A["aw"], A["al"])
        m4, s4, _, _ = water_mask(z4["vv"], z4["vh"],
                                  z4["cov"] & B["inside"], B["aw"], B["al"])
        if m1 is None or m4 is None:
            print(f"    {eid}: one side could not classify -- skipped")
            continue
        v1 = (z1["cov"] & A["inside"])[wa]
        v4 = (z4["cov"] & B["inside"])[wb]
        ok = both & v1 & v4
        area = float(ok.sum()) * px
        if area < MIN_COMMON_KM2:
            print(f"    {eid}: only {area:,.0f} km2 seen by both -- skipped")
            continue
        a, b = m1[wa][ok], m4[wb][ok]
        cls_z4, how4 = class_z4(eid, float(s4))
        rows.append(dict(
            event_id=eid, date=eid[:10],
            relative_orbit=int(eid.split("_orb")[1].split("_")[0]),
            common_valid_area_km2=area,
            water_fraction_Z1=float(a.mean()),
            water_fraction_Z4=float(b.mean()),
            disagreement_fraction=float((a != b).mean()),
            water_jaccard=float((a & b).sum() / max((a | b).sum(), 1)),
            cohen_kappa=kappa(a, b),
            Z1_separation=float(s1), Z4_separation=float(s4),
            Z1_separation_class=cls1.get(eid, "?"),
            Z4_separation_class=cls_z4, Z4_class_source=how4,
            source_Z4="manifest" if (CFG.S1_CACHE / Z4 / f"{eid}.npz").exists()
            else "crosscheck"))
        r = rows[-1]
        print(f"    {eid:26s} {area:7,.0f} km2  D {100*r['disagreement_fraction']:5.2f}%"
              f"  J {r['water_jaccard']:.2f}  kappa {r['cohen_kappa']:5.2f}"
              f"  Z1 {r['Z1_separation_class']:14s} "
              f"Z4 {r['Z4_separation_class']:14s}"
              f"{'*' if r['Z4_class_source'].startswith('crosscheck') else ''}",
              flush=True)
        del z1, z4, m1, m4

    if not rows:
        raise SystemExit("no comparable scene")
    D = pd.DataFrame(rows)
    t = CFG.TABLES / "p14_zone1_cross_zone_corroboration.csv"
    D.to_csv(t, index=False)

    # -------------------------------------------------- baseline, then screen
    print("\n" + "=" * 78)
    print("BASELINE DISAGREEMENT — measured, not assumed")
    print("=" * 78)
    sound = D[~D.Z1_separation_class.isin(UNSOUND) &
              ~D.Z4_separation_class.isin(UNSOUND)]
    print(f"  {len(sound)} matched scenes sound on BOTH sides "
          f"(a weak classifier disagreeing with another weak one corroborates "
          f"nothing)")
    if len(sound) >= 3:
        q = sound.disagreement_fraction.quantile([.05, .25, .5, .75, .95])
        print(f"    D  p05 {q[.05]:.4f}  p25 {q[.25]:.4f}  med {q[.5]:.4f}  "
              f"p75 {q[.75]:.4f}  p95 {q[.95]:.4f}")
        print(f"    kappa median {sound.cohen_kappa.median():.2f}, "
              f"Jaccard median {sound.water_jaccard.median():.2f}")
    else:
        print("    too few to define a baseline")

    print("\n" + "=" * 78)
    print("NON-REDUNDANCY SCREEN, frozen operational cut |rho| < "
          f"{SCREEN_MAX_RHO}")
    print("=" * 78)
    rho = np.nan
    if len(D) >= 5:
        rho = float(D.Z1_separation.corr(D.disagreement_fraction,
                                         method="spearman"))
    admissible = bool(np.isfinite(rho) and abs(rho) < SCREEN_MAX_RHO)
    print(f"  rho(D, ZONE_1 separation) = {rho:+.2f} over n = {len(D)}")
    print(f"  ZONE_1 separation range in the matched set: "
          f"{D.Z1_separation.min():.2f} .. {D.Z1_separation.max():.2f}")
    verdict = dict(
        git=commit, n_matched=int(len(D)), n_sound=int(len(sound)),
        n_zone1_candidates=int(len(cand)),
        n_candidates_matched=int(len(reach)),
        spearman_D_vs_Z1_separation=rho if np.isfinite(rho) else None,
        screen_max_rho=SCREEN_MAX_RHO,
        screen_note="non-redundancy screen on monotonic association; "
                    "NOT a test of independence",
        clears_screen=admissible,
        baseline_D_median=float(sound.disagreement_fraction.median())
        if len(sound) else None,
        disagreement_threshold="NOT SET — baseline measured only",
        usable_as_corroborator=False, reason=None)

    if not len(reach):
        verdict["reason"] = (
            "no ZONE_1 LOW_SEPARATION or MARGINAL scene is matched, so the "
            "axis cannot be exercised on any candidate whatever its rho")
        print(f"\n  VERDICT: the axis cannot be used yet.")
        print(f"    {verdict['reason']}.")
    elif not admissible:
        verdict["reason"] = (
            f"|rho| = {abs(rho):.2f} >= {SCREEN_MAX_RHO}: disagreement largely "
            f"repeats separation and cannot corroborate it")
        print(f"\n  VERDICT: EXCLUDED as redundant. {verdict['reason']}.")
    else:
        verdict["usable_as_corroborator"] = True
        verdict["reason"] = (
            "clears the screen and reaches candidates; a disagreement "
            "threshold must still be set from the baseline distribution")
        print(f"\n  VERDICT: clears the screen and reaches candidates.")
        print(f"    A disagreement threshold is still NOT set: define it from "
              f"the baseline above, then add cross_zone_disagreement to "
              f"ZONE_1's corroborating axes with an explicit p13 "
              f"--recalibrate.")
    if len(D) < 20:
        tail = int((D.Z1_separation < R1.lda_separation.quantile(0.10)).sum())
        print(f"\n  SAMPLE CAVEAT: n = {len(D)}, of which {tail} sit below "
              f"ZONE_1's own p10. A rank correlation on this many points is "
              f"poorly determined; treat the rho above as indicative"
              f"{', and note the low tail is barely represented' if tail < 3 else ''}.")

    # WHAT THE TWO CANDIDATES LOOK LIKE, reported because the numbers are
    # striking and withheld from the decision because the axis did not clear
    # the screen. Neither is promoted to POOR.
    low = D[D.Z1_separation_class == "LOW_SEPARATION"]
    if len(low) and len(sound) >= 3:
        p95 = float(sound.disagreement_fraction.quantile(.95))
        print(f"\n  the ZONE_1 LOW_SEPARATION scenes, against a sound-pair "
              f"baseline of median {sound.disagreement_fraction.median():.4f} "
              f"and p95 {p95:.4f}:")
        for r in low.itertuples():
            where = ("INSIDE the baseline" if r.disagreement_fraction <= p95
                     else "FAR outside it")
            print(f"    {r.event_id:26s} D {100*r.disagreement_fraction:5.2f}% "
                  f"kappa {r.cohen_kappa:.2f}  Z4 side {r.Z4_separation_class}"
                  f"  -> {where}")
        print("    These do NOT become POOR: the axis was excluded, and using "
              "it anyway would be picking the evidence after seeing it.")
        verdict["low_separation_scenes"] = [
            dict(event_id=r.event_id,
                 disagreement_fraction=float(r.disagreement_fraction),
                 cohen_kappa=float(r.cohen_kappa),
                 z4_class=r.Z4_separation_class,
                 outside_baseline_p95=bool(r.disagreement_fraction > p95))
            for r in low.itertuples()]

    # THE STRUCTURAL POINT, which matters more than this one verdict.
    if not admissible:
        note = ("Every axis tried so far for ZONE_1 is derived from the SAR "
                "water mask, and a mask inherits the discriminant's behaviour: "
                "when separation is poor the mask changes, so any mask-derived "
                "quantity moves with separation almost by construction. p13 "
                "measured |rho| >= 0.51 for seven such axes; p14 measures "
                "-0.83 for cross-zone disagreement. A corroborator that clears "
                "the screen will most likely have to come from OUTSIDE the SAR "
                "mask entirely - a near-coincident optical (Sentinel-2) water "
                "extent, or an altimetric one - rather than from another way "
                "of reading the same mask.")
        verdict["structural_note"] = note
        print(f"\n  NOTE. {note}")

    j = CFG.TABLES / "p14_zone1_corroboration_verdict.json"
    j.write_text(json.dumps(verdict, indent=2))
    print(f"\n-> {t}")
    print(f"-> {j}")


if __name__ == "__main__":
    main()
