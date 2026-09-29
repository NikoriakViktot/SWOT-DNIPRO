#!/usr/bin/env python
"""HISTORICAL 22 — find an independent pre-breach LOW-WATER shoreline.

hist21 established that the highest-value missing dataset is a second
pre-breach reservoir-wide water extent at a substantially lower level than
2023-06-05 (17.08 m EVRF2019), because:

  * the near-shore zone (<500 m) carries 54 % of the DEM's squared error;
  * the historical A(H) curve cannot be used as a constraint without making
    the Table 21 validation circular;
  * post-breach extents fail on level meaning, geometry, H(x) and epoch.

This searches for one. Tasks 1-6 of the protocol: gauge history, then the
Sentinel-2 and Landsat archives, then level control, separation and -- the
question that decides whether it is worth downloading anything -- how much of
the high-error zone a second contour would actually cross.

Nothing is downloaded and no DEM is modified here. This is a search.

The archive queries hit the Planetary Computer STAC API, which is open and
needs no credentials. Cloud cover is the STAC scene-level figure, which is for
the whole tile and not for the reservoir; it is a screening variable only.

Outputs
-------
outputs/tables/prebreach_low_water_candidates.csv
outputs/tables/sentinel_low_water_candidates.csv
outputs/tables/landsat_low_water_candidates.csv
outputs/tables/second_contour_value.csv
outputs/figures/V18_low_water_search.png
"""
from __future__ import annotations

import json
import sys
import urllib.request
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
import pyproj
from shapely.geometry import shape

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
H_REF = 17.083588            # the constraint currently in use (2023-06-05)
BREACH = "2023-06-06"
UNS_BS, NPG_BS = 14.00, 16.00
BS_TO_EVRF = 0.185
CLOUD_MAX = 40               # screening only
# a candidate window must be pre-breach and below this
LOW_THRESHOLD = 15.5
MGRS_NEEDED = {"36TWS", "36TWT", "36TXS", "36TXT", "36UWU", "36UXU"}


def stac(coll, start, end, bbox, cloud=CLOUD_MAX, limit=500):
    q = {"collections": [coll], "bbox": bbox, "limit": limit,
         "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z"}
    if cloud is not None:
        q["query"] = {"eo:cloud_cover": {"lt": cloud}}
    req = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read()
                      ).get("features", [])


def main() -> None:
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
          SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    tf = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326",
                                     always_xy=True)
    lo1, la1 = tf.transform(fp.bounds[0], fp.bounds[1])
    lo2, la2 = tf.transform(fp.bounds[2], fp.bounds[3])
    bbox = [round(lo1, 3), round(la1, 3), round(lo2, 3), round(la2, 3)]

    # ================================================= TASK 1 — gauge history
    print("=" * 78)
    print("TASK 1 — GAUGE HISTORY: when was the pre-breach pool low?")
    print("=" * 78)
    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    g = w[(w.source == "gauge") & (w.domain == "reservoir")].copy()
    g["date"] = pd.to_datetime(g.date)
    pre = g[g.date < BREACH]
    print(f"  reservoir gauge record: {pre.date.min().date()} .. "
          f"{pre.date.max().date()}  ({pre.station_or_domain.nunique()} "
          f"stations, {len(pre):,} daily values)")
    print(f"  NOTE: the record starts in 2019, so the search is limited to "
          f"2019-2023.\n  Landsat-era low-water events (1990s-2010s) cannot be "
          f"level-controlled\n  from this repository and are addressed "
          f"separately in the report.")

    daily = (pre.groupby("date")
             .agg(n_gauges=("station_or_domain", "nunique"),
                  H_med=("transformed_level_m", "median"),
                  H_min=("transformed_level_m", "min"),
                  H_max=("transformed_level_m", "max"))
             .reset_index())
    daily["interstation_spread_m"] = daily.H_max - daily.H_min
    daily["dH_from_2023_06_05_m"] = daily.H_med - H_REF
    low = daily[daily.H_med < LOW_THRESHOLD].copy()
    # group into contiguous windows
    brk = (low.date.diff().dt.days.fillna(1) > 5).cumsum()
    print(f"\n  contiguous pre-breach windows with median level < "
          f"{LOW_THRESHOLD} m:")
    print(f"  {'window':<26}{'days':>6}{'H_med':>9}{'H_min':>8}"
          f"{'dH':>8}{'gauges':>8}{'spread':>8}")
    wins = []
    for k, s in low.groupby(brk):
        wins.append(dict(
            window=f"{s.date.min().date()}..{s.date.max().date()}",
            start=s.date.min(), end=s.date.max(), days=len(s),
            H_med=s.H_med.median(), H_min=s.H_med.min(),
            dH=s.H_med.median() - H_REF,
            n_gauges=int(s.n_gauges.max()),
            spread=float(s.interstation_spread_m.max())))
        r = wins[-1]
        print(f"  {r['window']:<26}{r['days']:>6}{r['H_med']:>9.2f}"
              f"{r['H_min']:>8.2f}{r['dH']:>+8.2f}{r['n_gauges']:>8}"
              f"{r['spread']:>8.2f}")
    daily.to_csv(CFG.TABLES / "prebreach_low_water_candidates.csv", index=False)

    # the two windows worth searching: deepest drawdown, and best gauge control
    W = pd.DataFrame(wins)
    w_low = W.loc[W.H_min.idxmin()]
    w_ctrl = W.loc[W.n_gauges.idxmax()]
    print(f"\n  DEEPEST DRAWDOWN : {w_low.window}  H_min {w_low.H_min:.2f} m, "
          f"dH {w_low.dH:+.2f} m, {w_low.n_gauges} gauge(s)")
    print(f"  BEST LEVEL CONTROL: {w_ctrl.window}  H_med {w_ctrl.H_med:.2f} m, "
          f"dH {w_ctrl.dH:+.2f} m, {w_ctrl.n_gauges} gauges, "
          f"inter-station spread <= {w_ctrl.spread:.2f} m")
    print(f"\n  The {w_ctrl.n_gauges}-gauge spread of {w_ctrl.spread:.2f} m is "
          f"itself independent evidence that the\n  pre-breach pool was level, "
          f"which is what makes a single contour elevation valid.")

    searchwins = []
    for r, tag in ((w_low, "deepest drawdown"), (w_ctrl, "best level control")):
        searchwins.append((str(r.start.date()), str(r.end.date()), tag,
                           float(r.H_med), float(r.n_gauges)))
    # de-duplicate if the same window wins both
    seen, sw = set(), []
    for s in searchwins:
        if (s[0], s[1]) not in seen:
            seen.add((s[0], s[1])); sw.append(s)

    # ============================================ TASKS 2-3 — archive search
    scenes = {"sentinel-2-l2a": [], "landsat-c2-l2": []}
    for coll, label in (("sentinel-2-l2a", "Sentinel-2 L2A"),
                        ("landsat-c2-l2", "Landsat C2 L2")):
        print("\n" + "=" * 78)
        print(f"TASK {'2' if 'sentinel' in coll else '3'} — {label} ARCHIVE")
        print("=" * 78)
        for start, end, tag, hmed, ng in sw:
            try:
                feats = stac(coll, start, end, bbox)
            except Exception as e:
                print(f"  {tag}: query failed -- {type(e).__name__}: {e}")
                continue
            rows = []
            for f in feats:
                p = f["properties"]
                tile = (p.get("s2:mgrs_tile")
                        or f"{p.get('landsat:wrs_path','')}/"
                           f"{p.get('landsat:wrs_row','')}")
                rows.append(dict(date=p["datetime"][:10], tile=tile,
                                 cloud=round(float(p.get("eo:cloud_cover", -1)), 1),
                                 platform=p.get("platform", ""),
                                 collection=coll, window=tag, id=f["id"]))
            df = pd.DataFrame(rows)
            if df.empty:
                print(f"  {tag} ({start}..{end}): no scenes under "
                      f"{CLOUD_MAX}% cloud")
                continue
            byd = (df.groupby("date")
                     .agg(n_scenes=("tile", "size"),
                          tiles=("tile", lambda v: sorted(set(v))),
                          cloud_min=("cloud", "min"),
                          cloud_max=("cloud", "max"),
                          cloud_med=("cloud", "median"),
                          platform=("platform", "first"))
                     .reset_index())
            byd["n_tiles"] = byd.tiles.map(len)
            if "sentinel" in coll:
                byd["mgrs_complete"] = byd.tiles.map(
                    lambda t: MGRS_NEEDED.issubset(set(t)))
            else:
                byd["mgrs_complete"] = byd.n_tiles >= 4
            byd["window"] = tag
            byd["reservoir_level_evrf2019_m"] = hmed
            byd["n_gauges"] = int(ng)
            byd["dH_from_2023_06_05_m"] = hmed - H_REF
            scenes[coll].append(byd)
            print(f"\n  {tag} ({start}..{end}), reservoir level "
                  f"{hmed:.2f} m, dH {hmed-H_REF:+.2f} m")
            print(f"  {'date':<12}{'tiles':>6}{'complete':>10}"
                  f"{'cloud min-max':>15}{'platform':>12}")
            for r in byd.sort_values(["mgrs_complete", "cloud_med"],
                                     ascending=[False, True]).itertuples():
                flag = "YES" if r.mgrs_complete else "no"
                print(f"  {r.date:<12}{r.n_tiles:>6}{flag:>10}"
                      f"{f'{r.cloud_min:.0f}-{r.cloud_max:.0f}%':>15}"
                      f"{str(r.platform)[:11]:>12}")

    s2 = pd.concat(scenes["sentinel-2-l2a"]) if scenes["sentinel-2-l2a"] else pd.DataFrame()
    ls = pd.concat(scenes["landsat-c2-l2"]) if scenes["landsat-c2-l2"] else pd.DataFrame()
    for df, name in ((s2, "sentinel_low_water_candidates"),
                     (ls, "landsat_low_water_candidates")):
        if not df.empty:
            df.assign(tiles=df.tiles.map(lambda t: "|".join(t))).to_csv(
                CFG.TABLES / f"{name}.csv", index=False)

    # ==================================== TASKS 4-5 — level control & dH
    print("\n" + "=" * 78)
    print("TASKS 4-5 — LEVEL CONTROL AND SEPARATION")
    print("=" * 78)
    best = []
    for df, lab in ((s2, "Sentinel-2"), (ls, "Landsat")):
        if df.empty:
            continue
        cand = df[df.mgrs_complete & (df.cloud_max <= 25)]
        for r in cand.itertuples():
            d = pd.Timestamp(r.date)
            gd = pre[pre.date == d]
            if len(gd) == 0:
                hh, ngg, spr = np.nan, 0, np.nan
            else:
                hh = float(gd.transformed_level_m.median())
                ngg = int(gd.station_or_domain.nunique())
                spr = float(gd.transformed_level_m.max()
                            - gd.transformed_level_m.min())
            best.append(dict(sensor=lab, date=r.date, n_tiles=r.n_tiles,
                             cloud_max=r.cloud_max,
                             H_evrf2019_m=hh, n_gauges=ngg,
                             interstation_spread_m=spr,
                             dH_m=hh - H_REF if np.isfinite(hh) else np.nan,
                             H_bs77_m=hh - BS_TO_EVRF if np.isfinite(hh) else np.nan))
    B = pd.DataFrame(best).sort_values("dH_m")
    print(f"  complete-coverage, low-cloud candidates with same-day gauge "
          f"control:\n")
    print(f"  {'sensor':<11}{'date':<12}{'tiles':>6}{'cloud':>7}"
          f"{'H EVRF':>9}{'H BS-77':>9}{'dH':>8}{'gauges':>8}{'spread':>8}")
    for r in B.itertuples():
        print(f"  {r.sensor:<11}{r.date:<12}{r.n_tiles:>6}"
              f"{r.cloud_max:>6.0f}%{r.H_evrf2019_m:>9.2f}{r.H_bs77_m:>9.2f}"
              f"{r.dH_m:>+8.2f}{r.n_gauges:>8}"
              + (f"{r.interstation_spread_m:>8.2f}" if np.isfinite(r.interstation_spread_m) else f"{'n/a':>8}"))
    strong = B[B.dH_m.abs() >= 2.0]
    adequate = B[(B.dH_m.abs() >= 1.5) & (B.dH_m.abs() < 2.0)]
    print(f"\n  |dH| >= 2.0 m (strong): {len(strong)} dates")
    print(f"  1.5 <= |dH| < 2.0 m (adequate): {len(adequate)} dates")

    # ============================ TASK 6 — would it cross the problem zone?
    print("\n" + "=" * 78)
    print("TASK 6 — WOULD A SECOND CONTOUR CROSS THE HIGH-ERROR ZONE?")
    print("=" * 78)
    print("  Answered from the soundings themselves: a shoreline at H2 pins the")
    print("  surface where the bed is at H2, so the informative belt is the set")
    print("  of locations with bed between H2 and the current 17.08 m contour.")
    dec = pd.read_csv(CFG.TABLES / "hist18_error_decomposition.csv")
    ovr = dec[dec.factor == "overall"].iloc[0]
    # rebuild the per-sounding frame the same way hist18 does
    import shapely
    from scipy.spatial import cKDTree
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                "kakhovka_soundings_evrf2019.parquet")
    sd = sd.assign(_kx=sd.x.round(0), _ky=sd.y.round(0)).groupby(
        ["_kx", "_ky"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"),
        H_bed_evrf2019_m=("H_bed_evrf2019_m", "mean"))
    fa = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                "hist14_cv_fold_assignments.parquet")
    assert np.allclose(fa.x.values, sd.x.values), "fold file misaligned"
    dshore = shapely.distance(shapely.points(sd.x.values, sd.y.values),
                              shapely.boundary(fp))
    S = pd.DataFrame(dict(bed=sd.H_bed_evrf2019_m.values, d_shore=dshore))

    rows = []
    print(f"\n  {'candidate H2':<28}{'belt n':>8}{'% of all':>10}"
          f"{'% within 500 m':>16}{'% of 0-500m pts':>17}")
    cands = [("current constraint only 17.08", H_REF, H_REF)]
    for lab, h2 in (("UNS 14.00 m BS-77 -> 14.19", UNS_BS + BS_TO_EVRF),
                    ("Feb 2023 low 13.93", 13.933588),
                    ("Feb 2019 low 15.39", 15.392436)):
        cands.append((lab, h2, H_REF))
    near = S[S.d_shore < 500]
    for lab, h2, h1 in cands[1:]:
        belt = S[(S.bed >= h2) & (S.bed <= h1)]
        bnear = belt[belt.d_shore < 500]
        rows.append(dict(candidate=lab, H2_evrf2019_m=h2,
                         belt_n=len(belt),
                         belt_pct_of_all=100 * len(belt) / len(S),
                         belt_pct_within_500m=100 * len(bnear) / max(1, len(belt)),
                         pct_of_near_shore_covered=100 * len(bnear) / max(1, len(near))))
        r = rows[-1]
        print(f"  {lab:<28}{r['belt_n']:>8}{r['belt_pct_of_all']:>9.1f}%"
              f"{r['belt_pct_within_500m']:>15.0f}%"
              f"{r['pct_of_near_shore_covered']:>16.0f}%")
    V = pd.DataFrame(rows)
    V.to_csv(CFG.TABLES / "second_contour_value.csv", index=False)
    print(f"\n  reference: {len(near):,} soundings lie within 500 m of the "
          f"shore ({100*len(near)/len(S):.0f}% of the survey)")
    print(f"  and hist18 measured RMSE {float(dec[(dec.factor=='dist_to_shoreline')&(dec.bin=='0-500 m')].RMSE_m.iloc[0]):.2f} m "
          f"there, carrying "
          f"{float(dec[(dec.factor=='dist_to_shoreline')&(dec.bin=='0-500 m')].share_SSE_pct.iloc[0]):.0f}% "
          f"of the squared error.")
    bestrow = V.loc[V.pct_of_near_shore_covered.idxmax()]
    print(f"\n  BEST: a contour at {bestrow.candidate} would lie in the "
          f"elevation band of")
    print(f"  {bestrow.belt_n:,} soundings, of which "
          f"{bestrow.belt_pct_within_500m:.0f}% sit within 500 m of the "
          f"shore, covering")
    print(f"  {bestrow.pct_of_near_shore_covered:.0f}% of the high-error "
          f"near-shore population.")

    # ---------------- the finding that changes the validation design --------
    zmax = float(S.bed.max())
    gap = H_REF - zmax
    print("\n  " + "-" * 74)
    print("  BUT NOTE — AND THIS CHANGES HOW THE EXPERIMENT MUST BE VALIDATED")
    print("  " + "-" * 74)
    print(f"    The highest sounding in the whole survey is {zmax:.2f} m, "
          f"against a waterline")
    print(f"    at {H_REF:.2f} m. Above that single highest sounding there is "
          f"a {gap:.2f} m band")
    print(f"    with NOTHING in it:")
    for lo, hi in ((13.90, 14.20), (14.20, 15.00), (15.00, zmax)):
        n = int(((S.bed >= lo) & (S.bed < hi)).sum())
        print(f"      {lo:5.2f}-{hi:5.2f} m : {n:5d} soundings")
    n_top = int((S.bed > zmax).sum())
    print(f"      {zmax:5.2f}-{H_REF:5.2f} m : {n_top:5d} soundings"
          f"   <-- the unsurveyed margin, {gap:.2f} m thick")
    print(f"\n    A contour at the Feb 2019 level (15.42 m) therefore sits "
          f"ABOVE EVERY")
    print(f"    SOUNDING. That is exactly why it is valuable -- it adds "
          f"elevation data")
    print(f"    where none exists -- but it is also why the frozen-fold CV "
          f"CANNOT")
    print(f"    demonstrate the improvement: cross-validation scores only at "
          f"sounding")
    print(f"    locations, and there are none in that band.")
    print(f"\n    So Task 8 as specified would report 'no improvement' even if "
          f"the margin")
    print(f"    got dramatically better. The measurable design is instead:")
    print(f"      LEAVE-ONE-CONTOUR-OUT. With three separated levels "
          f"({H_REF:.2f}, ~15.42, ~14.38 m)")
    print(f"      fit with two and predict the third. That is independent of "
          f"the soundings,")
    print(f"      independent of Tables 19/21, and it measures accuracy "
          f"exactly in the")
    print(f"      unsurveyed margin where the DEM is weakest.")
    print(f"    The frozen-fold sounding CV stays as the guard that the "
          f"interior did not")
    print(f"    degrade -- which is what it is good for.")

    # ================================================================ figure
    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    a = fig.add_subplot(gs[0, :])
    roz = pre[pre.station_or_domain == "station:80959"].sort_values("date")
    a.plot(roz.date, roz.transformed_level_m, color=BLUE, lw=1.1,
           label="Rozumivka 80959")
    multi = daily[daily.n_gauges >= 4]
    a.fill_between(multi.date, multi.H_min, multi.H_max, color=GREEN,
                   alpha=0.35, label="range across 4–6 gauges")
    a.axhline(H_REF, color=RED, lw=1.8, ls="--",
              label=f"current constraint {H_REF:.2f} m (2023-06-05)")
    a.axhline(UNS_BS + BS_TO_EVRF, color=AMBER, lw=1.6, ls=":",
              label=f"УНС {UNS_BS:.2f} m BS-77 = {UNS_BS+BS_TO_EVRF:.2f} m")
    a.axhline(NPG_BS + BS_TO_EVRF, color=GREY, lw=1.3, ls=":",
              label=f"НПГ {NPG_BS:.2f} m BS-77")
    for r in B.drop_duplicates("date").itertuples():
        a.plot([pd.Timestamp(r.date)], [r.H_evrf2019_m], "v", ms=9,
               color=INK, zorder=6)
        a.annotate(r.date[2:], (pd.Timestamp(r.date), r.H_evrf2019_m),
                   textcoords="offset points", xytext=(0, -16), ha="center",
                   fontsize=7.4, rotation=45)
    a.set_ylabel("reservoir level (m EVRF2019)")
    a.legend(fontsize=8.4, ncol=3, loc="lower left")
    a.grid(alpha=0.25)
    a.set_title("a · Pre-breach reservoir level, with complete-coverage "
                "low-cloud satellite dates marked (▼)", fontsize=11,
                loc="left")

    a = fig.add_subplot(gs[1, 0])
    for df, c, m, lab in ((s2, BLUE, "o", "Sentinel-2 L2A"),
                          (ls, AMBER, "s", "Landsat C2 L2")):
        if df.empty:
            continue
        d = df.drop_duplicates("date")
        a.scatter(pd.to_datetime(d.date), d.cloud_med,
                  s=np.where(d.mgrs_complete, 95, 26), c=c, marker=m,
                  alpha=0.85, label=lab, edgecolor="white", linewidth=0.6)
    a.axhline(25, color=RED, ls="--", lw=1.4, label="25 % cloud screen")
    a.set_ylabel("scene cloud cover (median over tiles, %)")
    a.legend(fontsize=8.4)
    a.grid(alpha=0.25)
    a.tick_params(axis="x", rotation=30, labelsize=8)
    a.set_title("b · Archive availability in the low-water windows\n"
                "large marker = complete reservoir coverage",
                fontsize=11, loc="left")

    a = fig.add_subplot(gs[1, 1])
    edges = [0, 250, 500, 1000, 2000, 4000, 1e9]
    labs = ["0–250", "250–500", "500–1k", "1–2k", "2–4k", ">4k"]
    S["_b"] = pd.cut(S.d_shore, edges, labels=labs, include_lowest=True)
    tot = S.groupby("_b", observed=True).size()
    x = np.arange(len(tot))
    a.bar(x, tot.values, color=GREY, alpha=0.45, label="all soundings")
    for (lab, h2), c in zip([("14.19 m (УНС)", UNS_BS + BS_TO_EVRF),
                             ("13.93 m (Feb 2023)", 13.933588)],
                            [GREEN, BLUE]):
        belt = S[(S.bed >= h2) & (S.bed <= H_REF)]
        cnt = belt.groupby("_b", observed=True).size().reindex(tot.index,
                                                              fill_value=0)
        a.plot(x, cnt.values, "o-", color=c, lw=2.1, ms=6,
               label=f"in the {h2:.2f}–17.08 m belt")
    a.set_xticks(x)
    a.set_xticklabels(labs, fontsize=8.6)
    a.set_xlabel("distance to shoreline (m)")
    a.set_ylabel("soundings")
    a.legend(fontsize=8.4)
    a.grid(alpha=0.25, axis="y")
    a.set_title("c · Does the second contour reach the problem zone?\n"
                "the 0–500 m bins carry 54 % of the squared error",
                fontsize=11, loc="left")

    fig.suptitle("V18 · Search for an independent pre-breach low-water "
                 "shoreline   ·   gauge history + Sentinel-2 + Landsat, "
                 "nothing downloaded yet", fontsize=12.6, y=0.985)
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V18_low_water_search.{e}", dpi=175,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V18_low_water_search.png'}")
    for n in ("prebreach_low_water_candidates", "sentinel_low_water_candidates",
              "landsat_low_water_candidates", "second_contour_value"):
        print(f"-> {CFG.TABLES/(n + '.csv')}")


if __name__ == "__main__":
    main()
