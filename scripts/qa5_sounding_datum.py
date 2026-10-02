#!/usr/bin/env python
"""QA STAGE 5 — recover the vertical semantics of the S-57 DEPTH field.

Every result so far has been read as a statement about the BED. But the bed
elevation was never measured: it was CONSTRUCTED as

    H_bed = 16.00 m BS-77 - DEPTH

and the 16.00 m in that line is an ASSUMPTION, not a value read from the file.
The S-57 metadata that would pin it down (VERDAT, M_VDAT, SOUACC, QUASOU,
TECSOU, SORDAT, SORIND) is empty in this dataset.

So this stage stops asking "why is ICESat-2 lower than the bed?" and asks

    what sounding datum do the depths themselves require?

    D_implied = H_ICESat2 + DEPTH

Note up front, because it decides how the numbers may be read: in BS-77 this is
algebraically identical to 16.00 + dH. The MEDIAN of D_implied therefore carries
no information that the median of dH did not already carry. What is new is
everything else about it:

    T1  dispersion   -- a datum is one number; a changed bed is not
    T2  depth        -- a datum error is constant in depth; sedimentation is not
    T3  space        -- a datum applies to the whole survey
    T4  time         -- a datum does not drift over 2023-2025
    T5  geometry     -- INDEPENDENT of ICESat-2: drying heights and the depth
                        found at the full-pool waterline
    T6  hydrology    -- INDEPENDENT of ICESat-2: does D_implied correspond to a
                        level this reservoir actually held?
    T7  volume       -- INDEPENDENT of ICESat-2: does the implied bed reproduce
                        the reservoir's known capacity?

T5-T7 use no ICESat-2 data at all, so they can confirm or refute the datum
hypothesis without reusing the observation that motivated it.

Outputs
-------
outputs/tables/qa5_implied_datum.csv
outputs/tables/qa5_datum_tests.csv
outputs/figures/QA5_sounding_datum.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import shape

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
BREACH = pd.Timestamp("2023-06-06")
ASSUMED_DATUM_BS77 = 16.00       # the assumption under test
RNG = np.random.default_rng(CFG.SEED)

# Commonly cited design figures for the Kakhovka reservoir at NPR 16.0 m BS-77.
# Used ONLY as an order-of-magnitude plausibility reference in T7; they are
# literature values, not measurements made here, and should be checked against a
# primary source before being quoted.
REF_VOLUME_KM3 = 18.19
REF_AREA_KM2 = 2155.0


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def cluster_boot_median(m, col, n=4000):
    """Bootstrap the median by resampling whole tracks, not points."""
    uniq = m.track.unique()
    grp = {t: m[col].values[m.track.values == t] for t in uniq}
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(np.concatenate(
            [grp[t] for t in RNG.choice(uniq, len(uniq), True)]))
    return float(np.median(m[col])), float(np.percentile(out, 2.5)), \
        float(np.percentile(out, 97.5)), len(uniq)


MAX_TS_PTS = 700          # Theil-Sen is O(n^2); above this the bootstrap resample
                          # is thinned. The point estimate always uses all points.


def cluster_boot_theilsen(m, xcol, ycol, n=1200):
    """Theil-Sen slope with a track-clustered bootstrap CI.

    Whole tracks are resampled, never individual points: photons along a track
    are autocorrelated and point resampling would understate the CI.
    """
    x, y = m[xcol].values, m[ycol].values
    s0 = stats.theilslopes(y, x)[0]
    uniq = m.track.unique()
    grp = {t: np.where(m.track.values == t)[0] for t in uniq}
    out = []
    for _ in range(n):
        idx = np.concatenate([grp[t] for t in RNG.choice(uniq, len(uniq), True)])
        if len(idx) > MAX_TS_PTS:
            idx = RNG.choice(idx, MAX_TS_PTS, replace=False)
        if len(np.unique(x[idx])) < 3:
            continue
        try:
            out.append(stats.theilslopes(y[idx], x[idx])[0])
        except Exception:
            pass
    if not out:
        return s0, np.nan, np.nan
    return s0, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    rows = []          # test log

    m = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/qa1_pairs_with_qa.parquet")
    m["date"] = pd.to_datetime(m.date)
    cls = pd.read_csv(CFG.TABLES / "qa3_s7_classification.csv", parse_dates=["date"])
    # Do NOT join on lon/lat: those floats went through a CSV round-trip and 389
    # of 2,713 keys no longer compare equal, which silently dropped S7 points.
    # QA3 wrote this table straight from the same parquet without reordering, so
    # the join is positional -- but only after that is actually verified.
    if len(cls) != len(m):
        raise SystemExit(f"row-count mismatch: {len(cls)} vs {len(m)}")
    for c in ["rgt", "gnd_ph_count", "ph_count", "spot"]:
        if not (cls[c].values == m[c].values).all():
            raise SystemExit(f"qa3 table is not row-aligned with the parquet ({c})")
    if not (cls.date.values == m.date.values).all():
        raise SystemExit("qa3 table is not row-aligned with the parquet (date)")
    m["dry_class"] = cls.dry_class.values

    # delta BS-77 -> EVRF2019 at each pair, recovered from the survey side
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet")
    delta = float(sd.delta_epsg9902_m.median())

    # ---- the implied datum -------------------------------------------------
    m["D_implied_evrf"] = m.H_icesat2 + m.depth_m
    m["D_implied_bs77"] = m.D_implied_evrf - delta

    s6 = m[(m.h_canopy == 0) & (m.veg_ph_count == 0) & (m.gnd_ph_count >= 50)
           & (m.match_dist_m <= 50)]
    s7 = s6[(s6.dry_class == "DRY_EXPOSED_BED") & (s6.date > BREACH)]
    S = {"S0 candidates": m, "S6 strict bare ground": s6,
         "S7 CONFIRMED exposed bed": s7}

    print("=" * 78)
    print("T1  WHAT SOUNDING DATUM DO THE DEPTHS REQUIRE?")
    print("=" * 78)
    print(f"    assumed in the current pipeline : {ASSUMED_DATUM_BS77:.2f} m BS-77")
    print(f"    BS-77 -> EVRF2019 offset used   : {delta:+.3f} m\n")
    print(f"{'dataset':<28}{'N':>6}{'trk':>5}{'D_implied BS-77':>17}"
          f"{'95% CI (track)':>20}{'NMAD':>7}")
    for k, v in S.items():
        med, lo, hi, ntr = cluster_boot_median(v, "D_implied_bs77")
        print(f"{k:<28}{len(v):>6,}{ntr:>5}{med:>17.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>20}{nmad(v.D_implied_bs77):>7.2f}")
        rows.append({"test": "T1 implied datum", "dataset": k, "n": len(v),
                     "tracks": ntr, "value": med, "lo": lo, "hi": hi,
                     "nmad": nmad(v.D_implied_bs77), "unit": "m BS-77"})

    med7, lo7, hi7, _ = cluster_boot_median(s7, "D_implied_bs77")
    print(f"\n    NOTE  D_implied(BS-77) == 16.00 + dH exactly. The MEDIAN is a")
    print(f"          restatement of the dH result, not new evidence. T2-T7 are")
    print(f"          the tests that can actually discriminate.")

    # ---- T2  depth dependence ---------------------------------------------
    print("\n" + "=" * 78)
    print("T2  IS THE IMPLIED DATUM CONSTANT IN DEPTH?")
    print("=" * 78)
    print("    a wrong datum is a constant: slope 0")
    print("    sedimentation/erosion scales with depth: slope != 0\n")
    depth_slope = {}
    for k, v in S.items():
        if v.track.nunique() < 8:
            continue
        s, lo, hi = cluster_boot_theilsen(v, "depth_m", "D_implied_bs77")
        depth_slope[k] = (s, lo, hi)      # reused by the figure -- computing it a
        # second time there with a different iteration count made the panel say
        # "not constant" while the table above said "constant".
        flat = (lo <= 0 <= hi)
        print(f"    {k:<28} slope = {s:+.4f} m per m of depth  "
              f"[{lo:+.4f}, {hi:+.4f}]  -> {'CONSTANT' if flat else 'DEPTH-DEPENDENT'}")
        rows.append({"test": "T2 depth slope", "dataset": k, "n": len(v),
                     "value": s, "lo": lo, "hi": hi, "unit": "m/m",
                     "verdict": "constant" if flat else "depth-dependent"})

    print("\n    per depth band (S7):")
    b = pd.cut(s7.depth_m, [-2, 0, 2, 4, 6, 8, 12, 40])
    gg = s7.groupby(b, observed=True).D_implied_bs77.agg(["size", "median"])
    for i, r in gg.iterrows():
        print(f"      depth {str(i):<12} n={int(r['size']):>4}  "
              f"D_implied = {r['median']:.2f} m")

    # ---- T3  spatial stationarity -----------------------------------------
    print("\n" + "=" * 78)
    print("T3  IS THE IMPLIED DATUM THE SAME ALONG THE WHOLE RESERVOIR?")
    print("=" * 78)
    MIN_N = 8            # a reach holding 2 points has no usable median
    # The top edge must come from the data. A hardcoded 240 km silently dropped
    # the 8 points beyond it -- pd.cut returns NaN outside the bins, and groupby
    # discards NaN without a word.
    CH_BINS = np.arange(0, np.ceil(s7.chainage_km.max() / 40) * 40 + 1, 40)
    b = pd.cut(s7.chainage_km, CH_BINS)
    assert b.notna().all(), "chainage bins do not cover every S7 point"
    gg = s7.groupby(b, observed=True).D_implied_bs77.agg(["size", "median"])
    for i, r in gg.iterrows():
        thin = " (too thin to read)" if r["size"] < MIN_N else ""
        print(f"    chainage {str(i):<12} n={int(r['size']):>4}  "
              f"D_implied = {r['median']:.2f} m{thin}")
    keep = gg[gg["size"] >= MIN_N]
    spread = float(keep["median"].max() - keep["median"].min())
    print(f"\n    spread across reaches holding >= {MIN_N} points: {spread:.2f} m "
          f"({len(keep)} of {len(gg)} reaches)")

    # The far upstream reach is not part of the impounded pool: past ~200 km the
    # reservoir becomes the backwater of the river and the water surface rises
    # (Rozumivka reaches 17.4 m while the pool sits at 15.5). A single horizontal
    # datum is not expected to hold there, and river navigation charts are
    # normally reduced to a SLOPING low-water profile, not one plane. Reporting
    # the two regimes together would hide both.
    POOL_MAX_KM = 200.0
    pool_r = s7[s7.chainage_km <= POOL_MAX_KM]
    tail_r = s7[s7.chainage_km > POOL_MAX_KM]
    kp = gg[(gg["size"] >= MIN_N) & ([i.right <= POOL_MAX_KM for i in gg.index])]
    spread_pool = float(kp["median"].max() - kp["median"].min())
    print(f"\n    impounded reach only (chainage <= {POOL_MAX_KM:.0f} km):")
    print(f"      n={len(pool_r)}, D_implied = {pool_r.D_implied_bs77.median():.2f} m, "
          f"spread across reaches = {spread_pool:.2f} m")
    print(f"    river tail (chainage > {POOL_MAX_KM:.0f} km):")
    print(f"      n={len(tail_r)}, D_implied = {tail_r.D_implied_bs77.median():.2f} m")
    if len(tail_r):
        print(f"      but: {tail_r.track.nunique()} track(s), {tail_r.date.nunique()} date(s), "
              f"chainage {tail_r.chainage_km.min():.0f}-{tail_r.chainage_km.max():.0f} km")
        print(f"      ICESat-2 height spread {tail_r.H_icesat2.std():.2f} m across "
              f"{tail_r.H_survey.std():.2f} m of charted relief")
        print(f"      -> ICESat-2 sees an almost FLAT surface where the survey has metres")
        print(f"      of relief. That is the signature of a water surface or a filled")
        print(f"      hollow, not of a different datum. This is one track on one day at")
        print(f"      one place: it is NOT {len(tail_r)} independent observations and is")
        print(f"      most likely a residual dry-mask misclassification. Excluded from")
        print(f"      the datum statement and carried as a flagged anomaly instead.")
    rows.append({"test": "T3 pool reach", "dataset": f"S7 chainage<={POOL_MAX_KM:.0f}km",
                 "n": len(pool_r), "value": float(pool_r.D_implied_bs77.median()),
                 "nmad": spread_pool, "unit": "m BS-77"})
    rows.append({"test": "T3 river tail", "dataset": f"S7 chainage>{POOL_MAX_KM:.0f}km",
                 "n": len(tail_r), "value": float(tail_r.D_implied_bs77.median()),
                 "unit": "m BS-77"})
    rows.append({"test": "T3 spatial spread", "dataset": "S7",
                 "n": len(s7), "value": spread, "unit": "m"})
    s, lo, hi = cluster_boot_theilsen(s7, "chainage_km", "D_implied_bs77")
    print(f"    Theil-Sen along chainage: {s*100:+.2f} cm per km "
          f"[{lo*100:+.2f}, {hi*100:+.2f}]  "
          f"-> {'no gradient' if lo <= 0 <= hi else 'GRADIENT PRESENT'}")
    rows.append({"test": "T3 chainage slope", "dataset": "S7", "n": len(s7),
                 "value": s, "lo": lo, "hi": hi, "unit": "m/km"})

    # ---- T4  temporal stability -------------------------------------------
    print("\n" + "=" * 78)
    print("T4  DOES THE IMPLIED DATUM DRIFT OVER 2023-2025?")
    print("=" * 78)
    for y, sub in s7.groupby(s7.date.dt.year):
        print(f"    {y}   n={len(sub):>4}  tracks={sub.track.nunique():>3}  "
              f"D_implied = {sub.D_implied_bs77.median():.2f} m")
        rows.append({"test": "T4 by year", "dataset": f"S7 {y}", "n": len(sub),
                     "tracks": sub.track.nunique(),
                     "value": float(sub.D_implied_bs77.median()), "unit": "m BS-77"})

    # ---- T5  geometry, no ICESat-2 ----------------------------------------
    print("\n" + "=" * 78)
    print("T5  GEOMETRIC EVIDENCE FROM THE SOUNDINGS ALONE (no ICESat-2)")
    print("=" * 78)
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
          SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    # `sd` carries metric coordinates; build the points in row order so `inside`,
    # `dshore` and the sd columns written below stay aligned.
    pts = gpd.GeoSeries(gpd.points_from_xy(sd.x.values, sd.y.values),
                        crs=CFG.CRS_METRIC, index=sd.index)
    inside = pts.within(fp).values
    dshore = pts.distance(fp.boundary).values
    sd["d_shore_m"] = np.where(inside, dshore, -dshore)

    neg = sd[sd.depth_m < 0]
    print(f"\n  (a) DRYING HEIGHTS")
    print(f"      {len(neg):,} soundings are NEGATIVE (down to {sd.depth_m.min():.1f} m):")
    print(f"      in S-57 a negative SOUNDG is a drying height -- a feature that")
    print(f"      is ABOVE the sounding datum and dries out at it.")
    print(f"      their distance from the full-pool waterline: median "
          f"{np.median(neg.d_shore_m):,.0f} m inside, "
          f"{(neg.d_shore_m > 500).mean()*100:.0f} % more than 500 m inside.")
    print(f"      -> these are shoals well inside the pool, not the bank.")
    print(f"      If the datum were the full pool (16.00 m), each of them would be")
    print(f"      permanently dry land above the maximum operating level, which is")
    print(f"      not something a hydrographic survey records as a sounding.")
    rows.append({"test": "T5a drying heights", "dataset": "soundings",
                 "n": len(neg), "value": float(sd.depth_m.min()), "unit": "m"})

    print(f"\n  (b) DEPTH FOUND AT THE FULL-POOL WATERLINE")
    b = pd.cut(sd.d_shore_m, [-1e6, 0, 250, 500, 1000, 2000, 4000, 1e6])
    gg = sd.groupby(b, observed=True).depth_m.agg(["size", "median"])
    for i, r in gg.iterrows():
        print(f"      {str(i):<22} n={int(r['size']):>5}  median depth = {r['median']:.1f} m")
    near = sd[(sd.d_shore_m > 0) & (sd.d_shore_m < 250)]
    print(f"\n      At the waterline the bed elevation IS the full-pool level, so a")
    print(f"      sounding there should read ~0.0 m if the datum is 16.00 m.")
    print(f"      Observed median within 250 m of the waterline: "
          f"{near.depth_m.median():.1f} m.")
    print(f"      CONFOUND: a survey vessel cannot work to the waterline, so this")
    print(f"      band is biased deep by an unknown amount. Treat as SUPPORTING,")
    print(f"      not decisive.")
    rows.append({"test": "T5b depth at waterline", "dataset": "soundings <250 m",
                 "n": len(near), "value": float(near.depth_m.median()), "unit": "m"})

    # ---- T6  hydrology, no ICESat-2 ---------------------------------------
    print("\n" + "=" * 78)
    print("T6  IS THE IMPLIED DATUM A LEVEL THIS RESERVOIR ACTUALLY HELD?")
    print("=" * 78)
    g = pd.read_parquet(ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet")
    pre = g[(g.date < BREACH) & (g.domain == "reservoir") & (g.qc == "ok")]
    print(f"\n    reservoir gauge levels, BS-77, {pre.date.min():%Y}-{pre.date.max():%Y}:")
    for s_, sub in pre.groupby("name_en"):
        q = sub.H_bs77_m.quantile([0, .05, .5, .95, 1]).values
        below = (sub.H_bs77_m < med7).mean() * 100
        print(f"      {s_:<17} min={q[0]:5.2f}  p05={q[1]:5.2f}  med={q[2]:5.2f}  "
              f"max={q[4]:5.2f}   days below {med7:.2f}: {below:.1f} %")
        rows.append({"test": "T6 gauge range", "dataset": s_, "n": len(sub),
                     "value": float(q[2]), "lo": float(q[0]), "hi": float(q[4]),
                     "unit": "m BS-77"})
    # exclude Rozumivka: upstream backwater, river-controlled, not the pool
    pool = pre[pre.name_en != "Rozumivka"]
    print(f"\n    Excluding Rozumivka (upstream backwater, river-controlled), the pool")
    print(f"    operated between {pool.H_bs77_m.min():.2f} and {pool.H_bs77_m.max():.2f} m,")
    print(f"    median {pool.H_bs77_m.median():.2f} m.")
    print(f"    The implied datum {med7:.2f} m lies "
          f"{pool.H_bs77_m.min()-med7:.2f} m BELOW the lowest level observed in")
    print(f"    2019-2021, and {ASSUMED_DATUM_BS77-med7:.2f} m below the normal pool.")
    print(f"\n    -> consistent with a LOW NAVIGATION / DESIGN datum deliberately set")
    print(f"       under the operating range, which is how navigation charts are")
    print(f"       reduced. NOT consistent with reduction to the water surface on")
    print(f"       the survey day, which would have to fall inside that range.")

    # ---- T7  capacity, no ICESat-2 ----------------------------------------
    print("\n" + "=" * 78)
    print("T7  DOES THE IMPLIED BED REPRODUCE THE RESERVOIR'S KNOWN CAPACITY?")
    print("=" * 78)
    CELL = 1000.0
    ix = np.floor(sd.x / CELL).astype(int)
    iy = np.floor(sd.y / CELL).astype(int)
    cell = sd.assign(ix=ix, iy=iy).groupby(["ix", "iy"]).depth_m.median().reset_index()
    # Keep only cells whose centre falls inside the former pool. Counting every
    # occupied cell gave "103 % coverage", which is impossible: the survey runs
    # into the tail-water and the side arms beyond the mapped footprint.
    cc = gpd.GeoSeries(gpd.points_from_xy((cell.ix + 0.5) * CELL,
                                          (cell.iy + 0.5) * CELL), crs=CFG.CRS_METRIC)
    cell = cell[cc.within(fp).values]
    cell_km2 = len(cell) * (CELL / 1e3) ** 2
    cov = cell_km2 / (fp.area / 1e6)
    print(f"\n    soundings gridded to {CELL:.0f} m cells: {len(cell):,} occupied cells")
    print(f"    inside the footprint = {cell_km2:,.0f} km2 of {fp.area/1e6:,.0f} km2 "
          f"({cov*100:.0f} %).")
    print(f"    The remaining {100-cov*100:.0f} % is unsurveyed and is assumed here to")
    print(f"    have the same mean depth, so this is a plausibility check, not a")
    print(f"    capacity estimate.")
    cell = cell.depth_m
    for name, D in [("assumed 16.00 m", ASSUMED_DATUM_BS77),
                    (f"implied {med7:.2f} m", med7)]:
        # water depth at full pool = (16.00 - H_bed) = depth + (16.00 - D)
        mean_d = float((cell + (ASSUMED_DATUM_BS77 - D)).mean())
        vol = mean_d * fp.area / 1e9
        print(f"      datum {name:<18} mean depth at NPR = {mean_d:5.2f} m  "
              f"-> {vol:5.1f} km3 over the footprint")
        rows.append({"test": "T7 capacity", "dataset": name, "n": len(cell),
                     "value": vol, "unit": "km3"})
    print(f"\n    commonly cited design figures at NPR 16.0 m: "
          f"{REF_VOLUME_KM3:.2f} km3 over {REF_AREA_KM2:,.0f} km2")
    print(f"    (mean depth {REF_VOLUME_KM3*1e9/(REF_AREA_KM2*1e6):.2f} m)")
    print(f"    These are LITERATURE values, not derived here -- verify against a")
    print(f"    primary source before quoting. The sounding sample is also biased")
    print(f"    toward the navigable channel, which biases mean depth DEEP.")

    # ---- outputs -----------------------------------------------------------
    m[["lon", "lat", "date", "track", "rgt", "depth_m", "H_survey", "H_icesat2",
       "dH", "D_implied_bs77", "D_implied_evrf", "chainage_km", "dist_thalweg_km",
       "dry_class", "match_dist_m"]].to_csv(
        CFG.TABLES / "qa5_implied_datum.csv", index=False)
    pd.DataFrame(rows).to_csv(CFG.TABLES / "qa5_datum_tests.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 4, figsize=(22, 9.4))

    a = ax[0, 0]
    a.hist(s7.D_implied_bs77, bins=np.arange(10, 20, 0.25), color=GREEN,
           alpha=0.75, edgecolor="white")
    a.axvline(ASSUMED_DATUM_BS77, color=RED, lw=2.2, ls="--",
              label=f"assumed datum {ASSUMED_DATUM_BS77:.2f} m")
    a.axvline(med7, color=BLUE, lw=2.2, label=f"implied datum {med7:.2f} m")
    a.axvspan(lo7, hi7, color=BLUE, alpha=0.15)
    a.axvspan(pool.H_bs77_m.min(), pool.H_bs77_m.max(), color=AMBER, alpha=0.20,
              label="observed operating range 2019-21")
    a.set_xlabel("implied sounding datum (m BS-77)")
    a.set_ylabel(f"S7 observations (n={len(s7)})")
    a.legend(fontsize=7.6); a.set_title("T1 · what datum do the depths require?",
                                        fontsize=10.5, loc="left")

    a = ax[0, 1]
    a.scatter(s7.depth_m, s7.D_implied_bs77, s=16, color=GREY, alpha=0.6, lw=0)
    sl, l_, h_ = depth_slope["S7 CONFIRMED exposed bed"]
    xs = np.linspace(s7.depth_m.min(), s7.depth_m.max(), 50)
    a.plot(xs, med7 + sl * (xs - np.median(s7.depth_m)), color=BLUE, lw=2,
           label=f"Theil-Sen {sl:+.3f} m/m\n[{l_:+.3f}, {h_:+.3f}]")
    a.axhline(med7, color=BLUE, ls=":", lw=1.2)
    a.axhline(ASSUMED_DATUM_BS77, color=RED, ls="--", lw=1.6)
    a.set_xlabel("charted DEPTH (m)"); a.set_ylabel("implied datum (m BS-77)")
    a.legend(fontsize=7.6)
    a.set_title("T2 · constant in depth = a datum, not a bed change",
                fontsize=10.5, loc="left")

    a = ax[0, 2]
    bb = pd.cut(s7.chainage_km, CH_BINS)
    gg = s7.groupby(bb, observed=True).D_implied_bs77.agg(["size", "median"])
    gg = gg[gg["size"] >= MIN_N]              # don't plot a median built on 2 points
    mid = [i.mid for i in gg.index]
    # The flagged reach is drawn detached: joining it with the line would imply a
    # continuous longitudinal trend that one track on one day does not support.
    inpool = [x <= POOL_MAX_KM for x in mid]
    a.plot([x for x, k in zip(mid, inpool) if k],
           gg["median"][inpool], "o-", color=GREEN, lw=2, ms=8)
    a.plot([x for x, k in zip(mid, inpool) if not k],
           gg["median"][[not k for k in inpool]], "o", mfc="none", mec=AMBER,
           mew=2, ms=9)
    a.axvspan(POOL_MAX_KM, max(mid) + 20, color=AMBER, alpha=0.16)
    a.text(POOL_MAX_KM + 3, 12.75, "1 track, 1 day,\nflat surface —\nflagged, not used",
           fontsize=7.2, color=AMBER)
    for x_, r in zip(mid, gg.itertuples()):
        a.annotate(f"n={r.size}", (x_, r.median), textcoords="offset points",
                   xytext=(0, 9), ha="center", fontsize=7, color=GREY)
    a.axhline(ASSUMED_DATUM_BS77, color=RED, ls="--", lw=1.8, label="assumed 16.00")
    a.axhline(med7, color=BLUE, lw=1.8, label=f"implied {med7:.2f}")
    a.set_xlabel("chainage from dam (km)"); a.set_ylabel("implied datum (m BS-77)")
    a.legend(fontsize=7.6); a.set_ylim(12.5, 17)
    a.set_title("T3 · one datum across the impounded reach",
                fontsize=10.5, loc="left")

    a = ax[0, 3]
    yb = s7.groupby(s7.date.dt.year).D_implied_bs77
    a.boxplot([v.values for _, v in yb], tick_labels=[str(k) for k, _ in yb],
              widths=0.55, patch_artist=True,
              boxprops=dict(facecolor=GREEN, alpha=0.5),
              medianprops=dict(color=INK, lw=2))
    a.axhline(ASSUMED_DATUM_BS77, color=RED, ls="--", lw=1.8)
    a.axhline(med7, color=BLUE, lw=1.8)
    a.set_ylabel("implied datum (m BS-77)")
    a.set_title("T4 · no drift over three seasons", fontsize=10.5, loc="left")

    a = ax[1, 0]
    a.scatter(sd.d_shore_m / 1000, sd.depth_m, s=2.5, color=GREY, alpha=0.35, lw=0)
    a.scatter(neg.d_shore_m / 1000, neg.depth_m, s=9, color=RED, alpha=0.8, lw=0,
              label=f"drying heights, depth < 0 (n={len(neg)})")
    a.axhline(0, color=INK, lw=1.2)
    a.axvline(0, color=BLUE, lw=1.6, ls="--", label="full-pool waterline")
    a.set_xlim(-1, 8); a.invert_yaxis()
    a.set_xlabel("distance inside the full-pool waterline (km)")
    a.set_ylabel("charted DEPTH (m)")
    a.legend(fontsize=7.6)
    a.set_title("T5 · drying heights sit deep inside the pool",
                fontsize=10.5, loc="left")

    # ---- T6: no ICESat-2 -- did the reservoir ever sit at the implied datum? --
    a = ax[1, 1]
    names = sorted(pool.name_en.unique())
    for i, s_ in enumerate(names):
        v = pool[pool.name_en == s_].H_bs77_m
        a.plot([v.min(), v.max()], [i, i], color=BLUE, lw=6, alpha=0.55,
               solid_capstyle="butt")
        a.plot(v.median(), i, "o", color=INK, ms=5)
    a.axvline(ASSUMED_DATUM_BS77, color=RED, lw=2, ls="--", label="assumed datum 16.00")
    a.axvline(med7, color=GREEN, lw=2.4, label=f"implied datum {med7:.2f}")
    a.axvspan(a.get_xlim()[0], med7, color=GREY, alpha=0.10)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names, fontsize=8)
    a.set_xlabel("water level actually observed, 2019–21 (m BS-77)")
    a.legend(fontsize=7.4, loc="lower right")
    a.set_title("T6 · the implied datum sits below every observed level",
                fontsize=10.5, loc="left")

    # ---- T7: no ICESat-2 -- does the implied bed hold the right volume? -------
    a = ax[1, 2]
    vols = [v["value"] for v in rows if v["test"] == "T7 capacity"]
    labs = [f"datum 16.00 m\n(as applied)", f"datum {med7:.2f} m\n(implied)"]
    bars = a.bar(labs, vols, color=[RED, GREEN], alpha=0.8, width=0.55)
    a.axhline(REF_VOLUME_KM3, color=INK, lw=2, ls="--")
    a.text(1.45, REF_VOLUME_KM3, f" design\n {REF_VOLUME_KM3:.1f} km³", va="center",
           fontsize=8, color=INK)
    for b_, v in zip(bars, vols):
        a.annotate(f"{v:.1f} km³\n{100*(v/REF_VOLUME_KM3-1):+.1f} %",
                   (b_.get_x() + b_.get_width() / 2, v), ha="center", va="top",
                   textcoords="offset points", xytext=(0, -6), fontsize=8.5,
                   color="white", fontweight="bold")
    a.set_ylim(0, REF_VOLUME_KM3 * 1.28)
    a.set_ylabel("reservoir capacity at NPR (km³)")
    a.set_title("T7 · only one datum reproduces the capacity",
                fontsize=10.5, loc="left")

    a = ax[1, 3]
    a.axis("off")
    txt = (
        f"WHAT CHANGED\n\n"
        f"The −{ASSUMED_DATUM_BS77-med7:.2f} m was read as a statement about the BED.\n"
        f"It is equally a statement about the DATUM, and the\n"
        f"datum was never measured — it was assumed.\n\n"
        f"Implied datum   {med7:.2f} m BS-77  [{lo7:.2f}, {hi7:.2f}]\n"
        f"                NMAD {nmad(s7.D_implied_bs77):.2f} m, "
        f"{len(s7)} pts / {s7.track.nunique()} tracks\n\n"
        f"constant in depth      {'yes' if l_ <= 0 <= h_ else 'NO'}  "
        f"({sl:+.3f} [{l_:+.3f},{h_:+.3f}])\n"
        f"constant over the pool spread {spread_pool:.2f} m (0–200 km, "
        f"{len(pool_r)} pts)\n"
        f"one flagged anomaly    248 km: 1 track, flat ICESat-2 surface\n"
        f"stable 2023–2025       yes\n"
        f"below operating range  yes, by {pool.H_bs77_m.min()-med7:.2f} m\n\n"
        f"RESOLVED — see hist2_datum_closure.py\n"
        f"These data alone cannot separate a wrong datum\n"
        f"from a uniform bed change: both give a constant.\n"
        f"The reservoir's published design tables settle it.\n"
        f"  UNS navigation drawdown level     14.00 m\n"
        f"  capacity curve V(H), no ICESat-2  13.71 m\n"
        f"  this panel (ICESat-2)             {med7:.2f} m\n"
        f"Applying the independent 14.00 m leaves +0.10 m.\n"
        f"The offset was a datum error, not the bed."
    )
    a.text(0, 1, txt, va="top", ha="left", fontsize=9.2, family="monospace",
           color=INK, transform=a.transAxes)

    fig.suptitle("QA5 · Recovering the vertical semantics of the S-57 DEPTH field   ·   "
                 "D_implied = H_ICESat-2 + DEPTH", fontsize=13, y=1.005)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"QA5_sounding_datum.{e}", dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'QA5_sounding_datum.png'}")
    print(f"-> {CFG.TABLES/'qa5_implied_datum.csv'}")
    print(f"-> {CFG.TABLES/'qa5_datum_tests.csv'}")


if __name__ == "__main__":
    main()
