#!/usr/bin/env python
"""P23 -- SWOT LakeSP across the breach, inside ONE orbit configuration.

DESIGN FROZEN BEFORE ANY RESULT WAS LOOKED AT (audit protocol section 7: a
hypothesis formed after seeing the data is exploratory and must say so). This
docstring is the pre-registration; the code below implements it and nothing
else. Status of every number this script produces: EXPLORATORY.

Why this exists
---------------
The manuscript's planform-fragmentation thesis rests on Sentinel-2 with a
PRE_BREACH cohort of THREE dates, only one of them spatially complete
(audit 20260916T093000Z, F-02/F-03). Coverage-matched recomputation is
impossible on optics: PRE_BREACH at coverage_class == HIGH is n = 1.

SWOT LakeSP offers what optics cannot here. Over ZONE_1 there are 48 dates
before the breach and 32 after it in the SAME cal/val 1-day repeat orbit
(cycles 478-543 PRE, 544-578 POST): same swaths, same viewing geometry, same
instrument, same processing, near-daily, radar (cloud-independent), and a
mission whose processing chain shares nothing with Sentinel-1/2.

The question (not "does LakeSP reproduce F2")
---------------------------------------------
    Did the remotely-sensed water-object structure over ZONE_1 undergo a
    persistent discontinuity across 2023-06-06 under a FIXED observational
    geometry?

LakeSP counts lake OBJECTS matched (Obs) or unmatched (Unassigned) to a prior
database; that is a different estimator with a different support from a raster
fragmentation index. It corroborates or fails to corroborate the PHENOMENON.
It does not and cannot reproduce the METRIC. Nothing here is compared to F2.

Design, fixed
-------------
* Unit of analysis: the DATE (one overpass). Objects within a date are
  pseudo-replicates and are only aggregated, never counted as independent.
* Population: cal/val cycles 478-578 ONLY. Post-breach science-orbit cycles
  001-025 are EXCLUDED: they share no cycle with PRE and have different swath
  geometry; mixing them would re-create exactly the coverage confound this
  design exists to avoid.
* Objects: Obs UNION Unassigned. On 2023-04-05 ZONE_1 has 119 Obs against
  1,468 Unassigned -- Unassigned dominates 12:1 BEFORE the breach, so it is
  not "post-breach ponds", it is most of the water. Using Obs alone would
  measure the prior-matching algorithm, not the water.
* Spatial filter: object geometry intersects
  SD.load("ZONE_1_KAKHOVKA_LOWER_DNIPRO").
* Per-date metrics: n_objects, sum_area_detct_km2, sum_area_total_km2,
  median_wse_m, wse_p90_p10_m, frac_partial (partial_f > 0), frac_dark
  (mean dark_frac), n_granules.
* QC: objects with quality_f != 0 are dropped. ASSUMPTION, stated: 0 = good
  per the usual SWOT flag convention; the PDD is not available locally and
  this has not been verified against it. The unfiltered count is kept in a
  column so the effect of the assumption is visible.
* Coverage guard (SWOT is not immune): frac_partial is reported per date and
  every contrast is repeated on the subset of objects with partial_f == 0.
  "Not observed is not dry" applies to swath edges too.
* Test: two-sided permutation test on the difference of PRE/POST medians of
  each per-date metric, 20,000 permutations of the date labels, seed
  CFG.SEED. Effect reported as difference of medians with a bootstrap 95 % CI.
  No p-value is read as proof of absence.

REVISION v2 -- POST-HOC, made after the v1 results were seen (2026-09-16)
-------------------------------------------------------------------------
Three things changed, and because they were decided after looking at v1 the
outputs carry a `_v2` suffix and v1 stays on disk. Two are implementation
errors; one is a population change that v1's numbers themselves exposed.

1. UNITS. LakeSP `area_total` / `area_detct` are already km2 (PDD); v1 divided
   by 1e6 and reported ~0. Bug, fixed. v1 area rows are void.
2. QC. v1 KEPT ONLY quality_f == 0. In the 482_001 granule every one of the
   119 Obs and 1,468 Unassigned objects has quality_f == 1, so v1 counted a
   small remnant and called it the population. The flag's semantics are not
   verifiable locally (no PDD), so QC is now a SENSITIVITY arm: primary =
   all objects; `_q0` = quality_f == 0; `_nopartial` = partial_f == 0.
3. POPULATION. v1 median WSE over ZONE_1 was 74 m (Obs) / 55 m (Unassigned).
   The Dnipro sits at 0-16 m; 74 m is steppe ponds on the terraces inside the
   ZONE_1 polygon. The question is about the river-reservoir system, so the
   default population is now objects intersecting the registry's
   `dnipro_water_domain` (the connected Dnipro water system), with
   `--domain ZONE_1_KAKHOVKA_LOWER_DNIPRO` reproducing v1's population.
   This is the change that is genuinely post-hoc, and it is labelled so.

Outputs
-------
outputs/tables/p23_lakesp_<domain>_by_date[_v2].csv
outputs/tables/p23_lakesp_pre_post_calval_<domain>[_v2].csv
outputs/figures/P23_lakesp_calval_transition_<domain>[_v2].png
"""
from __future__ import annotations

import glob
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

ZONE = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
LAKESP = CFG.BULK_ROOT / "swot_ua" / "swot_l2_hr_lakesp_2.0"
BREACH = "20230606"
CALVAL_CYCLES = (478, 578)          # inclusive; PRE = 478..543, POST = 544..578
N_PERM = 20_000
N_BOOT = 5_000
METRICS = ["n_objects", "sum_area_detct_km2", "sum_area_total_km2",
           "median_wse_m", "wse_p90_p10_m", "frac_partial", "frac_dark"]
_NAME = re.compile(r"LakeSP_(Obs|Unassigned)_(\d{3})_(\d{3})_EU_(\d{8})T")


def scan() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(LAKESP / "**" / "*.shp"), recursive=True):
        m = _NAME.search(os.path.basename(p))
        if not m:
            continue
        rows.append(dict(path=p, kind=m.group(1), cycle=int(m.group(2)),
                         pas=int(m.group(3)), date=m.group(4)))
    F = pd.DataFrame(rows)
    F = F[(F.cycle >= CALVAL_CYCLES[0]) & (F.cycle <= CALVAL_CYCLES[1])]
    F["period"] = np.where(F.date < BREACH, "PRE", "POST")
    return F.sort_values(["date", "cycle", "pas"]).reset_index(drop=True)


def per_date(F: pd.DataFrame, zone, bbox) -> pd.DataFrame:
    out = []
    for d, g in F.groupby("date"):
        parts = []
        for r in g.itertuples():
            try:
                x = gpd.read_file(r.path, bbox=bbox)
            except Exception:
                continue
            if len(x):
                x = x[x.intersects(zone)]
            if len(x):
                x = x.assign(kind=r.kind)
                parts.append(x)
        if not parts:
            continue
        X = pd.concat(parts, ignore_index=True)
        n_raw = len(X)
        for c in ("area_detct", "area_total", "wse", "partial_f", "dark_frac", "quality_f"):
            if c not in X:
                X[c] = np.nan
        def block(Y, tag=""):
            if len(Y) == 0:
                return {f"n_objects{tag}": 0}
            w = Y.wse.astype(float).dropna()
            return {
                f"n_objects{tag}": int(len(Y)),
                # area_* are km2 in the product already (v1 divided by 1e6 -- wrong)
                f"sum_area_detct_km2{tag}": float(Y.area_detct.astype(float).sum()),
                f"sum_area_total_km2{tag}": float(Y.area_total.astype(float).sum()),
                f"median_wse_m{tag}": float(w.median()) if len(w) else np.nan,
                f"wse_p90_p10_m{tag}": float(w.quantile(.9) - w.quantile(.1)) if len(w) > 2 else np.nan,
                f"frac_partial{tag}": float((Y.partial_f.astype(float) > 0).mean()),
                f"frac_dark{tag}": float(Y.dark_frac.astype(float).mean()),
            }
        qf = X.quality_f.astype(float).fillna(-1)
        row = dict(date=d, period=g.period.iloc[0], cycle_min=int(g.cycle.min()),
                   cycle_max=int(g.cycle.max()), n_granules=len(g),
                   n_objects_unfiltered=n_raw, n_obs=int((X.kind == "Obs").sum()),
                   n_unassigned=int((X.kind == "Unassigned").sum()),
                   frac_quality_f_nonzero=float((qf != 0).mean()))
        row.update(block(X))                                   # PRIMARY: all objects
        row.update(block(X[qf == 0], "_q0"))                   # sensitivity: quality_f == 0
        row.update(block(X[X.partial_f.astype(float).fillna(0) == 0], "_nopartial"))
        out.append(row)
        print(f"  {d} {row['period']:4s} cyc {row['cycle_min']}-{row['cycle_max']}  "
              f"obj {row['n_objects']:5d} (obs {row['n_obs']}, unass {row['n_unassigned']})  "
              f"area_det {row.get('sum_area_detct_km2', np.nan):8.1f} km2  "
              f"partial {row.get('frac_partial', np.nan):.2f}", flush=True)
    return pd.DataFrame(out)


def perm_test(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float, float]:
    """diff of medians (b - a), permutation p (two-sided), bootstrap 95% CI."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan, np.nan
    obs = np.median(b) - np.median(a)
    pool = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if abs(np.median(pool[na:]) - np.median(pool[:na])) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (N_PERM + 1)
    boots = np.array([np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a)))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(obs), float(p), float(lo), float(hi)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="dnipro_water_domain",
                    choices=["dnipro_water_domain", ZONE],
                    help="object population: the connected Dnipro water system (v2 default) "
                         "or the whole ZONE_1 polygon (reproduces v1's population)")
    ap.add_argument("--tag", default="_v2", help="output suffix; '' overwrites v1 names")
    a = ap.parse_args()
    dom_short = "dwd" if a.domain == "dnipro_water_domain" else "zone1"
    suffix = f"_{dom_short}{a.tag}"

    print("=" * 78)
    print(f"P23 -- SWOT LakeSP across the breach, cal/val orbit only  [EXPLORATORY, {suffix}]")
    print("=" * 78)
    zone = SD.load(a.domain)
    bbox = SD.as_bbox(a.domain)
    F = scan()
    print(f"  granules in cycles {CALVAL_CYCLES[0]}-{CALVAL_CYCLES[1]}: {len(F)} "
          f"({F.kind.value_counts().to_dict()}); dates PRE={F[F.period=='PRE'].date.nunique()} "
          f"POST={F[F.period=='POST'].date.nunique()}")
    assert not (set(F[F.period == "PRE"].cycle) & set(F[F.period == "POST"].cycle)), \
        "PRE and POST share a cycle -- the period cut is wrong"

    D = per_date(F, zone, bbox)
    by_date_path = CFG.TABLES / f"p23_lakesp_by_date{suffix}.csv"
    D.to_csv(by_date_path, index=False)

    rng = np.random.default_rng(CFG.SEED)
    rows = []
    SUBSETS = {"": "all objects (PRIMARY)", "_q0": "quality_f==0 (sensitivity)",
               "_nopartial": "partial_f==0 (sensitivity)"}
    for tag, label in SUBSETS.items():
        for m in METRICS:
            c = m + tag
            if c not in D:
                continue
            a_ = D[D.period == "PRE"][c].values.astype(float)
            b_ = D[D.period == "POST"][c].values.astype(float)
            obs, p, lo, hi = perm_test(a_, b_, rng)
            rows.append(dict(metric=m, subset=label, domain=a.domain,
                             n_pre=int(np.isfinite(a_).sum()), n_post=int(np.isfinite(b_).sum()),
                             median_pre=float(np.nanmedian(a_)) if np.isfinite(a_).any() else np.nan,
                             median_post=float(np.nanmedian(b_)) if np.isfinite(b_).any() else np.nan,
                             diff_post_minus_pre=obs, perm_p_two_sided=p,
                             boot_ci95_lo=lo, boot_ci95_hi=hi,
                             cycles="478-578 cal/val only",
                             status="EXPLORATORY; v2 post-hoc revision, see docstring"))
    R = pd.DataFrame(rows)
    R_path = CFG.TABLES / f"p23_lakesp_pre_post_calval{suffix}.csv"
    R.to_csv(R_path, index=False)
    print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D["dt"] = pd.to_datetime(D.date)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, c, lab in zip(axes, ("n_objects", "sum_area_detct_km2", "median_wse_m"),
                          ("objects (Obs ∪ Unassigned)", "detected water area, km²",
                           "median WSE, m")):
        for per, col in (("PRE", "#1b6ca8"), ("POST", "#c0392b")):
            s = D[D.period == per]
            ax.plot(s.dt, s[c], "o-", ms=3, lw=0.8, color=col, label=per)
            if f"{c}_nopartial" in D:
                ax.plot(s.dt, s[f"{c}_nopartial"], "x", ms=3, color=col, alpha=0.5)
        ax.axvline(pd.Timestamp("2023-06-06"), color="k", lw=1, ls="--")
        ax.set_ylabel(lab); ax.grid(alpha=.3)
    axes[0].legend(title="x = partial_f==0 subset", fontsize=8)
    axes[0].set_title(f"SWOT LakeSP over {a.domain}, cal/val orbit cycles 478–578 — EXPLORATORY{suffix}")
    fig.tight_layout()
    out = CFG.FIG / f"P23_lakesp_calval_transition{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"\n-> {by_date_path}")
    print(f"-> {R_path}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
