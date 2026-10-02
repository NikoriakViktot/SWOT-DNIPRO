#!/usr/bin/env python
"""P15 — a minimal optical cohort to test whether Sentinel-2 can corroborate
ZONE_1 low-separation scenes.

p13 found no usable corroborating indicator in ZONE_1: seven candidate axes all
sat at |Spearman| >= 0.51 against separation. p14 added cross-zone disagreement
and got -0.83. Both results have the same cause, and it is structural: every
axis tried was derived from the SAR water mask, and a mask inherits the
discriminant's behaviour, so any mask-derived quantity moves with separation
almost by construction.

So the next candidate has to come from another sensor. This builds the targeted
cohort for that test - NOT a ZONE_1 optical archive, which is a much larger and
separate job.

    Z1 SAR scene -> same/near-date S2 -> common valid footprint -> D_opt

ANOTHER SENSOR IS NOT AUTOMATICALLY NON-REDUNDANT, and the architecture must
not assume it is. If SAR and optical both degrade on flooded vegetation, or
both degrade when a pass sees only part of the zone, their disagreement will
track separation anyway. That is why the same operational screen applies at the
end: |rho| < 0.30 against ZONE_1 separation, measured, not declared. Optical
failing the screen is a valid outcome.

THE OPTICAL RULE IS FROZEN HERE, BEFORE ANY CANDIDATE IS LOOKED AT, and its
sha256 is written with the manifest. Choosing an index, a threshold or a
cleaning step after seeing how the candidates score would be fitting the
optical evidence to the SAR question - which is exactly the failure this test
exists to avoid. The rule is deliberately ordinary:

    valid      SCL in {4, 5, 6, 7, 11}      the same clear classes as p10
    MNDWI      (B03 - B11) / (B03 + B11)
    NDWI       (B03 - B08) / (B03 + B08)
    water      MNDWI > 0 AND NDWI > 0       both, to suppress dark soil and
                                            terrain shadow that MNDWI alone
                                            accepts
    cleaning   drop parts < 0.05 km2, fill holes < 0.05 km2 - identical to the
               SAR side, so cleaning cannot be the source of disagreement
    grid       read at the 20 m SAR grid, nearest neighbour

THREE COHORTS, and the controls are sampled before the candidates are scored:

    LOW_SEPARATION   all of them (2)
    MARGINAL         all of them (6)
    GOOD             N_CONTROL drawn stratified by orbit and season, seeded

The baseline for "unusual disagreement" will be measured on the GOOD and
MARGINAL controls only. The LOW candidates are never used to set the cut that
judges them.

DATE MATCHING, in strict preference order:

    dt = 0     same day
    |dt| <= 1
    |dt| <= 3  only when the reservoir stage moved less than STAGE_TOL_M
               between the two dates, or when no gauge covers the pair and the
               match is recorded as stage_unknown
    |dt| > 3   never. Seven days is long enough for the water to move on its
               own, and a disagreement would then be about time, not quality.

Nothing is downloaded by --manifest. It writes the cohort, the frozen rule and
their hashes; --compare does the work afterwards.

Outputs
-------
outputs/tables/p15_z1_optical_cohort.csv
outputs/tables/p15_z1_optical_freeze.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

import geopandas as gpd
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from p0r_zone2_flood_envelope import water_mask, CELL
import hist25b_gate6_event_qualification as G6

S2_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
S2_CACHE = "ZONE_1_s2_crosscheck"

Z1 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
N_CONTROL = 24
MAX_DT_DAYS = 3
STAGE_TOL_M = 0.10
MIN_SCOPE_COVERAGE = 0.30       # the S2 date must see a real part of ZONE_1
GRANULE_CLOUD_PREFILTER = 80    # generous; the real number is measured later

# ----------------------------------------------------------- the frozen rule
# Written down before a single candidate is scored. Its sha256 goes in the
# freeze, so a later change is visible as a different rule rather than as a
# better result.
OPTICAL_RULE = dict(
    rule_id="z1_optical_water_v1",
    valid_scl_classes=[4, 5, 6, 7, 11],
    indices=dict(MNDWI="(B03 - B11) / (B03 + B11)",
                 NDWI="(B03 - B08) / (B03 + B08)"),
    water="MNDWI > 0 AND NDWI > 0",
    min_part_km2=0.05,
    max_hole_km2=0.05,
    grid="SAR 20 m grid, nearest neighbour",
    note="both indices required; MNDWI alone accepts dark soil and terrain "
         "shadow. Cleaning is identical to the SAR side so that cleaning "
         "cannot itself generate disagreement.")


# ------------------------------------------------- the frozen decision layer
# Frozen BEFORE the download, for the same reason the optical rule is: once the
# two LOW candidates have a number, any choice about what counts as
# "disagreement" is a choice made in their light.
DECISION_RULE = dict(
    rule_id="p15_z1_optical_corroboration_v1",
    primary_metric="D_opt = area(SAR_water XOR S2_water) / "
                   "area(common_valid_footprint)",
    baseline_cohorts=["CONTROL_GOOD", "CANDIDATE_MARGINAL"],
    low_candidates_in_baseline=False,
    failure_cut="empirical p95(D_opt) of the baseline",
    failure_cut_status="operational QA cut adopted for this study; NOT a "
                       "statistical significance threshold",
    corroboration_requires=[
        "D_opt > frozen baseline p95",
        "usable common-valid support in the optical comparison",
        "D_opt clears the frozen non-redundancy screen "
        "|Spearman rho(D_opt, separation_Z1)| < 0.30"],
    otherwise="LOW_SEPARATION remains LOW_SEPARATION",
    min_common_valid_fraction=0.10,
    uncertainty="the full baseline distribution is stored, with a bootstrap "
                "CI for p95. At n ~ 26 the p95 is effectively set by the top "
                "one or two observations and must not be read as a limit",
    clustering="comparisons are resampled by UNIQUE S2 ACQUISITION DATE, not "
               "by SAR-S2 pair: four optical scenes serve two SAR scenes each "
               "and those pairs share the whole optical error. Duplicated "
               "pairs are KEPT - they are physically distinct SAR "
               "observations - but their optical error is not treated as "
               "independent",
    scope_note="f_common_valid = A(valid_SAR n valid_S2) / A(target scope) is "
               "what decides usability, never the granule cloud percentage. A "
               "pair surviving on 35% of ZONE_1 is local corroboration on "
               "those 35%, not a statement about the zone")
N_BOOTSTRAP = 2000


def decision_hash():
    return hashlib.sha256(
        json.dumps(DECISION_RULE, sort_keys=True).encode()).hexdigest()


def rule_hash():
    return hashlib.sha256(
        json.dumps(OPTICAL_RULE, sort_keys=True).encode()).hexdigest()


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


def kappa(a, b):
    if not a.size:
        return np.nan
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def season_of(m):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[m]


def gauge():
    """Reservoir stage, for the |dt| <= 3 admissibility test only."""
    try:
        w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
        g = w[(w.source == "gauge") & (w.domain == "reservoir")]
        s = g.groupby("date").transformed_level_m.median()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as ex:
        print(f"  gauge unavailable ({type(ex).__name__}); every near-date "
              f"match will be recorded as stage_unknown")
        return pd.Series(dtype=float)


def s2_dates(bbox, d0, d1):
    q = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 200,
         "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z",
         "query": {"eo:cloud_cover": {"lt": GRANULE_CLOUD_PREFILTER}}}
    out = {}
    for f in http_json(STAC, q).get("features", []):
        p = f["properties"]
        d = p["datetime"][:10]
        out.setdefault(d, []).append(f)
    return out


def optical_mask(items, G, tok):
    """The FROZEN optical rule, applied on the SAR grid.

    Bands are mosaicked in acquisition order like the SAR side: the first
    granule to supply a finite pixel keeps it. Nothing here is tunable - the
    rule lives in OPTICAL_RULE and its hash is in the freeze."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from scipy import ndimage
    ny, nx = G["ny"], G["nx"]
    acc = {b: np.full((ny, nx), np.nan, np.float32)
           for b in ("B03", "B08", "B11")}
    scl = np.zeros((ny, nx), np.uint8)
    for it in sorted(items, key=lambda f: f["properties"]["datetime"]):
        got = {}
        for b in ("B03", "B08", "B11", "SCL"):
            href = it["assets"].get(b, {}).get("href")
            if not href:
                return None, None
            try:
                with rasterio.open(href + "?" + tok) as ds:
                    from rasterio.warp import transform_bounds
                    bb = transform_bounds(CFG.CRS_METRIC, ds.crs,
                                          G["x0"], G["y0"], G["x1"], G["y1"],
                                          densify_pts=21)
                    got[b] = ds.read(
                        1, window=from_bounds(*bb, ds.transform),
                        out_shape=(ny, nx),
                        resampling=Resampling.nearest,
                        boundless=True, fill_value=0)
            except Exception:
                return None, None
        ok = got["SCL"] > 0
        new = ok & (scl == 0)
        scl[new] = got["SCL"][new]
        for b in ("B03", "B08", "B11"):
            acc[b][new] = got[b][new].astype(np.float32)
        del got
    valid = np.isin(scl, OPTICAL_RULE["valid_scl_classes"])
    g, n, w = acc["B03"], acc["B08"], acc["B11"]
    with np.errstate(invalid="ignore", divide="ignore"):
        mndwi = (g - w) / (g + w)
        ndwi = (g - n) / (g + n)
    m = valid & (mndwi > 0) & (ndwi > 0)
    m = np.nan_to_num(m, nan=False).astype(bool)
    # the same cleaning as the SAR side, so cleaning cannot create disagreement
    px = CELL ** 2 / 1e6
    lab, k = ndimage.label(m, structure=np.ones((3, 3), int))
    if k:
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1)) * px
        keep = np.zeros(k + 1, bool)
        keep[1:] = sz >= OPTICAL_RULE["min_part_km2"]
        m = keep[lab]
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes, structure=np.ones((3, 3), int))
    if hn:
        hs = ndimage.sum(np.ones_like(hl), hl, range(1, hn + 1)) * px
        fill = np.zeros(hn + 1, bool)
        fill[1:] = hs < OPTICAL_RULE["max_hole_km2"]
        m |= fill[hl]
    del acc, scl, mndwi, ndwi
    return m, valid


def block_bootstrap(D, stat, clusters, n=None, seed=None):
    """Resample UNIQUE S2 DATES, not pairs.

    Four optical scenes serve two SAR scenes each, and those pairs share the
    whole optical error of that scene. The pairs are kept - they are physically
    distinct SAR observations - but a bootstrap that resampled pairs would
    treat that shared error as independent."""
    rng = np.random.default_rng(seed if seed is not None else CFG.SEED)
    keys = np.array(sorted(set(clusters)))
    out = []
    for _ in range(n or N_BOOTSTRAP):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([np.where(clusters == k)[0] for k in pick])
        v = stat(D.iloc[idx])
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true",
                    help="build and freeze the cohort; downloads nothing")
    ap.add_argument("--compare", action="store_true",
                    help="download the cohort's S2 scenes and score every pair")
    ap.add_argument("--freeze-decision", action="store_true",
                    help="write the decision layer into the freeze, before "
                         "any optical data exists")
    args = ap.parse_args()
    if args.freeze_decision:
        fz = CFG.TABLES / "p15_z1_optical_freeze.json"
        d = json.loads(fz.read_text())
        if "decision_rule" in d:
            raise SystemExit(
                "the decision layer is already frozen "
                f"({d['decision_rule']['rule_id']}, sha256 "
                f"{d['decision_rule_sha256'][:16]}). Changing it now would be "
                "changing the rule in the light of the data it judges.")
        d["decision_rule"] = DECISION_RULE
        d["decision_rule_sha256"] = decision_hash()
        d["decision_frozen_before_download"] = True
        d["n_bootstrap"] = N_BOOTSTRAP
        fz.write_text(json.dumps(d, indent=2))
        print("=" * 78)
        print("P15 — decision layer frozen")
        print("=" * 78)
        print(f"  {DECISION_RULE['rule_id']}  sha256 {decision_hash()[:16]}")
        for k in ("primary_metric", "failure_cut", "failure_cut_status",
                  "otherwise"):
            print(f"    {k:22s} {DECISION_RULE[k]}")
        print(f"    baseline from        "
              f"{' + '.join(DECISION_RULE['baseline_cohorts'])}")
        for c in DECISION_RULE["corroboration_requires"]:
            print(f"      requires           {c}")
        print(f"    clustering           by unique S2 acquisition date")
        print(f"\n-> {fz}")
        print("  Frozen before a single optical pixel was downloaded.")
        return
    if args.compare:
        return compare()
    if not args.manifest:
        raise SystemExit("run --freeze-decision, then --manifest, then "
                         "--compare")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P15 — minimal ZONE_1 optical corroboration cohort")
    print("=" * 78)
    print(f"  git {commit}")
    print(f"  optical rule {OPTICAL_RULE['rule_id']} frozen at "
          f"sha256 {rule_hash()[:16]}")
    print(f"    valid  SCL in {OPTICAL_RULE['valid_scl_classes']}")
    print(f"    water  {OPTICAL_RULE['water']}")
    print("    frozen BEFORE any candidate is scored")

    R = pd.read_csv(CFG.TABLES / f"p12_{Z1.lower()}_scene_qa.csv")
    R["dt"] = pd.to_datetime(R.date)
    R["season"] = [season_of(t.month) for t in R.dt]
    low = R[R.separation_class == "LOW_SEPARATION"]
    marg = R[R.separation_class == "MARGINAL"]
    good = R[R.separation_class == "GOOD"]
    print(f"\n  ZONE_1 cohort: {len(low)} LOW_SEPARATION, {len(marg)} "
          f"MARGINAL, {len(good)} GOOD available as controls")

    # Controls stratified by orbit and season, drawn with the project seed.
    # Drawn now, before anything optical is measured, so the control set cannot
    # be influenced by how the candidates turn out.
    rng = np.random.default_rng(CFG.SEED)
    cells = list(good.groupby(["relative_orbit", "season"], observed=True))
    picks = []
    while len(picks) < min(N_CONTROL, len(good)) and cells:
        for _, g in cells:
            avail = [i for i in g.index if i not in picks]
            if not avail:
                continue
            picks.append(int(rng.choice(avail)))
            if len(picks) >= min(N_CONTROL, len(good)):
                break
        else:
            continue
        break
    ctrl = good.loc[sorted(set(picks))]
    print(f"  controls drawn: {len(ctrl)} GOOD over "
          f"{ctrl.relative_orbit.nunique()} orbits, "
          f"{ctrl.season.nunique()} seasons (seed {CFG.SEED})")

    targets = pd.concat([low, marg, ctrl]).sort_values("date")
    targets["cohort"] = np.where(
        targets.separation_class == "LOW_SEPARATION", "CANDIDATE_LOW",
        np.where(targets.separation_class == "MARGINAL", "CANDIDATE_MARGINAL",
                 "CONTROL_GOOD"))
    print(f"  {len(targets)} SAR scenes to match against Sentinel-2")

    g1 = SD.load_utm(Z1)
    bbox = [round(v, 5) for v in
            gpd.GeoSeries([g1], crs=32636).to_crs(4326).iloc[0].bounds]
    lev = gauge()

    rows = []
    for i, r in enumerate(targets.itertuples(), 1):
        d = pd.Timestamp(r.date)
        cat = s2_dates(bbox,
                       (d - pd.Timedelta(days=MAX_DT_DAYS)).date().isoformat(),
                       (d + pd.Timedelta(days=MAX_DT_DAYS)).date().isoformat())
        best = None
        for sd, items in sorted(cat.items()):
            dt = (pd.Timestamp(sd) - d).days
            if abs(dt) > MAX_DT_DAYS:
                continue
            fp = gpd.GeoSeries(
                [__import__("shapely.geometry", fromlist=["shape"]).shape(
                    it["geometry"]) for it in items],
                crs=4326).to_crs(CFG.CRS_METRIC).union_all()
            cov = fp.intersection(g1).area / g1.area
            if cov < MIN_SCOPE_COVERAGE:
                continue
            # the |dt| <= 3 admissibility test
            if abs(dt) <= 1:
                stage = "within_1_day"
                ok = True
            elif len(lev):
                a = lev.reindex([d]).ffill().iloc[0] if len(lev) else np.nan
                b = lev.reindex([pd.Timestamp(sd)]).ffill().iloc[0]
                try:
                    a = float(lev.asof(d)); b = float(lev.asof(pd.Timestamp(sd)))
                    ok = np.isfinite(a) and np.isfinite(b) and \
                        abs(a - b) <= STAGE_TOL_M
                    stage = (f"dstage {abs(a-b):.3f} m" if np.isfinite(a) and
                             np.isfinite(b) else "stage_unknown")
                    if "unknown" in stage:
                        ok = True
                except Exception:
                    stage, ok = "stage_unknown", True
            else:
                stage, ok = "stage_unknown", True
            if not ok:
                continue
            cand = dict(
                s2_date=sd, dt_days=int(dt), n_granules=len(items),
                s2_scope_coverage=float(cov), stage_test=stage,
                granule_cloud_mean=float(np.mean(
                    [it["properties"].get("eo:cloud_cover", np.nan)
                     for it in items])),
                tiles="|".join(sorted({it["properties"].get("s2:mgrs_tile", "?")
                                       for it in items})))
            key = (abs(dt), cand["granule_cloud_mean"])
            if best is None or key < best[0]:
                best = (key, cand)
        row = dict(event_id=r.event_id, sar_date=r.date, cohort=r.cohort,
                   separation_class=r.separation_class,
                   lda_separation=r.lda_separation,
                   relative_orbit=r.relative_orbit, season=r.season,
                   sar_coverage_fraction=r.coverage_fraction)
        row.update(best[1] if best else dict(
            s2_date=None, dt_days=None, n_granules=0, s2_scope_coverage=0.0,
            stage_test="no admissible S2", granule_cloud_mean=np.nan,
            tiles=""))
        rows.append(row)
        if i % 5 == 0 or i == len(targets):
            print(f"    {i}/{len(targets)} matched", flush=True)

    M = pd.DataFrame(rows)
    got = M[M.s2_date.notna()]
    print("\n" + "=" * 78)
    print("COHORT")
    print("=" * 78)
    for c, g in M.groupby("cohort"):
        gg = g[g.s2_date.notna()]
        print(f"  {c:20s} {len(g):3d} SAR scenes, {len(gg):3d} with an "
              f"admissible S2 match")
        if len(gg):
            print(f"      |dt| = 0: {int((gg.dt_days == 0).sum())}   "
                  f"|dt| <= 1: {int((gg.dt_days.abs() <= 1).sum())}   "
                  f"|dt| <= 3: {len(gg)}")
    print(f"\n  {len(got)} of {len(M)} SAR scenes have an optical partner")
    if len(got):
        print(f"  median |dt| {got.dt_days.abs().median():.0f} d, "
              f"median S2 coverage of ZONE_1 "
              f"{100*got.s2_scope_coverage.median():.0f}%, "
              f"median granule cloud {got.granule_cloud_mean.median():.0f}%")
    n_low = int((M.cohort == "CANDIDATE_LOW").sum())
    n_low_ok = int(((M.cohort == "CANDIDATE_LOW") & M.s2_date.notna()).sum())
    print(f"\n  the decisive number: {n_low_ok} of {n_low} LOW_SEPARATION "
          f"candidates have an optical partner")
    if n_low_ok < n_low:
        for r in M[(M.cohort == "CANDIDATE_LOW") & M.s2_date.isna()].itertuples():
            print(f"    no partner for {r.event_id} ({r.sar_date})")

    # SCENES SHARING AN OPTICAL PARTNER ARE NOT INDEPENDENT OF EACH OTHER.
    # Two SAR scenes a day or two apart can legitimately match the same S2
    # date, and then their two disagreement values share the whole optical
    # error of that scene. It does not invalidate either comparison, but they
    # must not be counted as two independent pieces of evidence.
    dup = got.groupby("s2_date").event_id.apply(list)
    dup = {k: v for k, v in dup.items() if len(v) > 1}
    if dup:
        print(f"\n  {len(dup)} S2 date(s) serve more than one SAR scene; "
              f"those comparisons share the optical error and are not "
              f"independent of each other:")
        for k, v in dup.items():
            print(f"    {k}  ->  {', '.join(v)}")

    csv = CFG.TABLES / "p15_z1_optical_cohort.csv"
    M.to_csv(csv, index=False)
    h = hashlib.sha256(csv.read_bytes()).hexdigest()
    (CFG.TABLES / "p15_z1_optical_freeze.json").write_text(json.dumps(dict(
        zone=Z1, git=commit, cohort=str(csv.relative_to(ROOT)), sha256=h,
        optical_rule=OPTICAL_RULE, optical_rule_sha256=rule_hash(),
        rule_frozen_before_results=True,
        n_candidate_low=n_low, n_candidate_low_matched=n_low_ok,
        n_candidate_marginal=int((M.cohort == "CANDIDATE_MARGINAL").sum()),
        n_control_good=int((M.cohort == "CONTROL_GOOD").sum()),
        control_seed=int(CFG.SEED), max_dt_days=MAX_DT_DAYS,
        stage_tol_m=STAGE_TOL_M, min_scope_coverage=MIN_SCOPE_COVERAGE,
        baseline_policy="the disagreement baseline is measured on CONTROL_GOOD "
                        "and CANDIDATE_MARGINAL only; CANDIDATE_LOW never "
                        "informs the cut that judges it",
        screen="the same operational cut |rho| < 0.30 against ZONE_1 "
               "separation; another sensor is not assumed non-redundant",
        shared_s2_dates={k: v for k, v in dup.items()},
        shared_s2_note="comparisons sharing an S2 date share its optical "
                       "error and are not independent of each other",
        bbox_4326=bbox), indent=2))
    print(f"\n-> {csv}")
    print(f"-> freeze sha256 {h[:16]}, optical rule {rule_hash()[:16]}")
    print("\n  Nothing downloaded. ZONE_1 stays GOOD / MARGINAL / "
          "LOW_SEPARATION with POOR unavailable until this test runs.")


def compare():
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    fz = CFG.TABLES / "p15_z1_optical_freeze.json"
    F = json.loads(fz.read_text())
    if "decision_rule" not in F:
        raise SystemExit("freeze the decision layer first")
    if F["optical_rule_sha256"] != rule_hash():
        raise SystemExit(
            f"the optical rule in this file no longer matches the frozen one "
            f"({rule_hash()[:16]} vs {F['optical_rule_sha256'][:16]}). That is "
            f"a different rule, not a better result.")
    DEC = F["decision_rule"]
    print("=" * 78)
    print("P15 — optical comparison")
    print("=" * 78)
    print(f"  git {commit}")
    print(f"  optical rule   {F['optical_rule_sha256'][:16]}  verified")
    print(f"  decision rule  {F['decision_rule_sha256'][:16]}  frozen "
          f"before download")

    M = pd.read_csv(CFG.TABLES / "p15_z1_optical_cohort.csv")
    M = M[M.s2_date.notna()].copy()
    print(f"  {len(M)} SAR-S2 pairs over {M.s2_date.nunique()} optical dates")

    g1 = SD.load_utm(Z1)
    gr = SD.build_grid(g1, CELL, what=f"{Z1} optical grid")
    x0, y1 = float(gr["gx"][0]), float(gr["gy"][-1])
    G = dict(x0=x0, y0=float(gr["gy"][0]), x1=float(gr["gx"][-1]), y1=y1,
             nx=gr["nx"], ny=gr["ny"])
    tr = from_origin(x0, y1, CELL, CELL)
    shp = (gr["ny"], gr["nx"])
    inside = rio_rasterize([(g1, 1)], out_shape=shp, transform=tr, fill=0,
                           dtype="uint8").astype(bool)
    w = SD.load_utm("dnipro_water_domain").intersection(g1)
    aw = rio_rasterize([(w.buffer(-200.0), 1)], out_shape=shp, transform=tr,
                       fill=0, dtype="uint8").astype(bool)
    al = rio_rasterize([(g1.difference(w.buffer(6000.0)), 1)], out_shape=shp,
                       transform=tr, fill=0, dtype="uint8").astype(bool)
    px = CELL ** 2 / 1e6
    scope = float(inside.sum()) * px
    print(f"  ZONE_1 {scope:,.0f} km2 on {shp[1]}x{shp[0]}")

    bbox = [round(v, 5) for v in
            gpd.GeoSeries([g1], crs=32636).to_crs(4326).iloc[0].bounds]
    cache = CFG.S1_CACHE.parent / S2_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    tok = json.loads(urllib.request.urlopen(S2_SAS, timeout=90).read())["token"]
    t0 = time.time()

    opt = {}
    for i, sd in enumerate(sorted(M.s2_date.unique()), 1):
        f = cache / f"{sd}.npz"
        if f.exists():
            opt[sd] = f
            continue
        if time.time() - t0 > 1800:
            tok = json.loads(urllib.request.urlopen(
                S2_SAS, timeout=90).read())["token"]
            t0 = time.time()
        cat = s2_dates(bbox, sd, sd)
        items = cat.get(sd, [])
        print(f"  [{i}/{M.s2_date.nunique()}] {sd}: {len(items)} granule(s)",
              flush=True)
        if not items:
            continue
        m, v = optical_mask(items, G, tok)
        if m is None:
            print("      asset read failed -- skipped"); continue
        np.savez_compressed(f, water=np.packbits(m), valid=np.packbits(v),
                            shape=np.array(shp))
        print(f"      valid {100*float(v[inside].mean()):5.1f}% of ZONE_1, "
              f"water {float(m.sum())*px:7,.0f} km2", flush=True)
        opt[sd] = f
        del m, v

    rows = []
    for r in M.itertuples():
        if r.s2_date not in opt:
            continue
        sar = CFG.S1_CACHE / Z1 / f"{r.event_id}.npz"
        if not sar.exists():
            continue
        z = np.load(sar)
        ms, sep, _, _ = water_mask(z["vv"], z["vh"], z["cov"] & inside, aw, al)
        if ms is None:
            del z; continue
        vs = z["cov"] & inside
        o = np.load(opt[r.s2_date])
        mo = np.unpackbits(o["water"], count=shp[0]*shp[1]).astype(bool).reshape(shp)
        vo = np.unpackbits(o["valid"], count=shp[0]*shp[1]).astype(bool).reshape(shp)
        ok = inside & vs & vo
        area = float(ok.sum()) * px
        frac = area / scope
        a, b = ms[ok], mo[ok]
        rows.append(dict(
            sar_event_id=r.event_id, s2_date=r.s2_date, dt_days=r.dt_days,
            delta_stage=r.stage_test, sar_separation=r.lda_separation,
            sar_separation_class=r.separation_class, cohort=r.cohort,
            relative_orbit=r.relative_orbit,
            common_valid_area_km2=area, common_valid_fraction=frac,
            D_opt=float((a != b).mean()) if a.size else np.nan,
            kappa=kappa(a, b) if a.size else np.nan,
            jaccard=float((a & b).sum() / max((a | b).sum(), 1)),
            water_fraction_SAR=float(a.mean()) if a.size else np.nan,
            water_fraction_S2=float(b.mean()) if b.size else np.nan,
            delta_water_fraction=float(a.mean() - b.mean()) if a.size else np.nan,
            shared_s2_cluster_id=r.s2_date,
            optical_rule_sha256=F["optical_rule_sha256"],
            cohort_sha256=F["sha256"]))
        q = rows[-1]
        print(f"    {r.event_id:26s} {r.s2_date}  dt {int(r.dt_days):+d}  "
              f"common {area:6,.0f} km2 ({100*frac:4.1f}%)  "
              f"D_opt {100*q['D_opt']:5.2f}%  kappa {q['kappa']:5.2f}"
              f"   [{r.cohort}]", flush=True)
        del z, ms, vs, o, mo, vo

    if not rows:
        raise SystemExit("no pair could be compared")
    D = pd.DataFrame(rows)
    t = CFG.TABLES / "p15_z1_optical_pairs.csv"
    D.to_csv(t, index=False)
    print(f"\n-> {t}")
    return evaluate(D, F, DEC, commit)


def evaluate(D, F, DEC, commit):
    usable = D[D.common_valid_fraction >= DEC["min_common_valid_fraction"]]
    print("\n" + "=" * 78)
    print("BASELINE — controls only, candidates excluded by the frozen rule")
    print("=" * 78)
    thin = D[D.common_valid_fraction < DEC["min_common_valid_fraction"]]
    if len(thin):
        print(f"  {len(thin)} pair(s) below the "
              f"{100*DEC['min_common_valid_fraction']:.0f}% common-valid floor."
              f"  The granule cloud percentage said nothing about this; the "
              f"measured common-valid fraction is what decides usability:")
        for r in thin.itertuples():
            print(f"    {r.sar_event_id:26s} {r.s2_date}  "
                  f"{100*r.common_valid_fraction:5.1f}%  "
                  f"SAR water {100*r.water_fraction_SAR:5.1f}% / "
                  f"S2 {100*r.water_fraction_S2:5.1f}%"
                  f"{'   <- CANDIDATE, untestable' if r.cohort == 'CANDIDATE_LOW' else ''}")
    base = usable[usable.cohort.isin(DEC["baseline_cohorts"])]
    print(f"  baseline from {len(base)} pairs "
          f"({' + '.join(DEC['baseline_cohorts'])}), "
          f"{base.shared_s2_cluster_id.nunique()} optical dates")
    if len(base) < 5:
        raise SystemExit("baseline too thin to define a cut")
    p95 = float(base.D_opt.quantile(0.95))
    q = base.D_opt.quantile([.05, .25, .5, .75, .95])
    print(f"  D_opt  p05 {q[.05]:.4f}  p25 {q[.25]:.4f}  med {q[.5]:.4f}  "
          f"p75 {q[.75]:.4f}  p95 {q[.95]:.4f}")
    bs = block_bootstrap(base, lambda d: float(d.D_opt.quantile(0.95)),
                         base.shared_s2_cluster_id.to_numpy())
    lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) \
        if len(bs) else (np.nan, np.nan)
    print(f"  p95 = {p95:.4f}, block-bootstrap 95% CI [{lo:.4f}, {hi:.4f}] "
          f"over {base.shared_s2_cluster_id.nunique()} optical dates")
    print(f"  At n = {len(base)} this quantile is set by the top one or two "
          f"observations. It is an operational QA cut, not a limit.")

    print("\n" + "=" * 78)
    print(f"NON-REDUNDANCY SCREEN, frozen |rho| < 0.30")
    print("=" * 78)
    rho = float(usable.sar_separation.corr(usable.D_opt, method="spearman"))
    bsr = block_bootstrap(
        usable, lambda d: float(d.sar_separation.corr(d.D_opt,
                                                      method="spearman")),
        usable.shared_s2_cluster_id.to_numpy())
    rlo, rhi = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))) \
        if len(bsr) else (np.nan, np.nan)
    clears = bool(abs(rho) < 0.30)
    print(f"  rho(D_opt, ZONE_1 separation) = {rho:+.2f} over n = {len(usable)}")
    print(f"  block-bootstrap 95% CI [{rlo:+.2f}, {rhi:+.2f}], clustered by "
          f"optical date")
    print(f"  -> {'CLEARS' if clears else 'EXCLUDED as redundant'}")

    print("\n" + "=" * 78)
    print("THE LOW_SEPARATION CANDIDATES — looked at last")
    print("=" * 78)
    low = usable[usable.cohort == "CANDIDATE_LOW"]
    verdicts = []
    for r in low.itertuples():
        over = r.D_opt > p95
        ok = over and clears
        print(f"  {r.sar_event_id}  sep {r.sar_separation:.3f}")
        print(f"    D_opt {100*r.D_opt:.2f}% vs baseline p95 "
              f"{100*p95:.2f}%  -> {'ABOVE' if over else 'within'}")
        print(f"    common valid {r.common_valid_area_km2:,.0f} km2 "
              f"({100*r.common_valid_fraction:.1f}% of ZONE_1) — any "
              f"corroboration is local to that area")
        print(f"    kappa {r.kappa:.2f}, Jaccard {r.jaccard:.2f}, "
              f"dwater {100*r.delta_water_fraction:+.2f} pp")
        print(f"    -> {'POOR' if ok else 'stays LOW_SEPARATION'}"
              f"{'' if clears else ' (the axis did not clear the screen)'}")
        verdicts.append(dict(sar_event_id=r.sar_event_id,
                             D_opt=float(r.D_opt), above_p95=bool(over),
                             common_valid_fraction=float(r.common_valid_fraction),
                             verdict="POOR" if ok else "LOW_SEPARATION"))
    out = dict(git=commit, n_pairs=int(len(D)), n_usable=int(len(usable)),
               n_baseline=int(len(base)),
               n_optical_dates=int(D.shared_s2_cluster_id.nunique()),
               baseline_p95=p95, baseline_p95_ci95=[lo, hi],
               baseline_quantiles={str(k): float(v) for k, v in q.items()},
               spearman_D_opt_vs_separation=rho,
               spearman_ci95=[rlo, rhi], clears_screen=clears,
               decision_rule_sha256=F["decision_rule_sha256"],
               optical_rule_sha256=F["optical_rule_sha256"],
               candidates=verdicts)
    j = CFG.TABLES / "p15_z1_optical_verdict.json"
    j.write_text(json.dumps(out, indent=2))
    print(f"\n-> {j}")
    if not clears:
        print("\n  ZONE_1 keeps POOR = 0. Another sensor was not enough: "
              "optical disagreement also tracks separation.")


if __name__ == "__main__":
    main()
