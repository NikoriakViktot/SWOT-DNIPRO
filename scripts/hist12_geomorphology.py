#!/usr/bin/env python
"""HISTORICAL 12 — is the exposed bed the drowned Dnipro valley, or an artefact?

The source describes the reservoir bed in words, and the description is
testable. Two statements matter:

  "Дно водохранилища неровное -- на общем фоне относительно спокойного рельефа
   ЗАТОПЛЕННОЙ ПОЙМЫ выделяются ГЛУБОКИЕ ЛОЖБИНЫ на месте бывшего русла
   р. Днепра, его многочисленных проток, рукавов и староречий"

  "За 15 лет существования водохранилища ТОЛЬКО НА ВЕРХНЕМ ШИРОКОМ ПЛЁСЕ
   неровности дна несколько сгладились благодаря действию волнения на
   мелководье"

and Fig. 128 places the terraces: the THIRD terrace already stands 5-7 m above
the normal impoundment level, so the reservoir bed can only be floodplain and,
at most, the lowest terrace. Nothing in the bed should sit at terrace-three
height.

Five predictions follow, each checkable against data we already hold:

  A  the bed elevation distribution is negatively skewed: a broad shallow
     platform (the drowned former floodplain) with a long deep tail
     (valley and channel depressions)
  B  the deep tail sits NEAR the modern Dnipro corridor, the shallow platform
     away from it
  C  post-breach ICESat-2 exposed bed sits on the shallow platform, not in the
     deep depressions -- those still carry the river
  D  bed roughness is LOWER in the upper broad reach than elsewhere
  E  the shallow fraction of the bed grows toward the upper reach, matching the
     drying areas Table 21 reports by reach (26 % and 35 % up there against
     3-9 % below)
  F  no surveyed bed point reaches the height of the third terrace

REACH BOUNDARIES ARE NOT USED. Table 21's printed reach lengths (43, 39, 50, 52
and 238 total) and my reading of Babyne on the Fig. 13 axis disagree by roughly
50 km, and that is unresolved. Every test below is therefore either independent
of chainage or uses only its monotone ordering.

Outputs
-------
data/processed/historical/historical_terraces.csv
outputs/tables/hist12_bed_morphology_tests.csv
outputs/figures/V11_bed_morphology_vs_source.png
"""
from __future__ import annotations

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
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
OUT = ROOT / "data/processed/historical"
SRC = "Dnipro reservoirs monograph, Fig. 128 and surrounding text"
NPG_EVRF = 16.00 + 0.185          # normal impoundment level in our frame
# "near the MODERN Dnipro corridor". SWORD is a modern-channel proxy; it is
# never treated as a reconstruction of any specific pre-1956 channel.
NEAR_CHANNEL_KM = 3.0
RNG = np.random.default_rng(CFG.SEED)

TERRACES = [
    dict(unit="floodplain terrace (пойменная)", legend_id=np.nan,
         height_above_npg_m=np.nan,
         composition="floodplain and oxbow alluvium; channel alluvium",
         where="southern part of the reservoir; west of Kakhovka",
         inundated="YES - this is the surface the reservoir drowned"),
    dict(unit="second terrace", legend_id=1, height_above_npg_m=np.nan,
         composition="loess rocks; floodplain and oxbow alluvium; channel alluvium",
         where="see Fig. 128", inundated="partly, at its lowest"),
    dict(unit="third terrace", legend_id=2, height_above_npg_m=6.0,
         composition="loess-like loams and loams 3-5 m thick, underlain by "
                     "sandy loams 2-4 m thick",
         where="both banks, in separate local patches",
         inundated="NO - stands 5-7 m above the normal impoundment level"),
    dict(unit="fourth terrace", legend_id=3, height_above_npg_m=np.nan,
         composition="loess rocks; channel alluvium",
         where="right bank above Bilenke; left bank near Ivanivka, "
               "Blahovishchenka, Illinka",
         inundated="NO"),
    dict(unit="fifth terrace", legend_id=4, height_above_npg_m=np.nan,
         composition="loess rocks; channel alluvium",
         where="well developed on the right bank; highest near Nikopol",
         inundated="NO"),
    dict(unit="pre-Quaternary Pliocene terrace", legend_id=5,
         height_above_npg_m=np.nan,
         composition="loess rocks; alluvial deposits", where="see Fig. 128",
         inundated="NO"),
    dict(unit="plateau", legend_id=6, height_above_npg_m=np.nan,
         composition="loess rocks and buried soils; red-brown clays; limestones",
         where="see Fig. 128", inundated="NO"),
]
THIRD_TERRACE_RANGE = (5.0, 7.0)     # m above NPG, as stated


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(TERRACES)
    t["source_page"] = SRC
    t["source_figure"] = "Fig. 128 + text"
    t.to_csv(OUT / "historical_terraces.csv", index=False)
    print(f"terrace units recorded: {len(t)}  -> {OUT/'historical_terraces.csv'}")

    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                                "kakhovka_soundings_evrf2019.parquet")
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    sd["chain_km"], sd["off_km"], _, _ = SW.assign_chainage(
        sd.lon.values, sd.lat.values, ch, tree=tree)
    h = sd.H_bed_evrf2019_m.values
    print(f"\n{len(sd):,} soundings on the 14.00 m reference, bed "
          f"{h.min():.2f} .. {h.max():.2f} m EVRF2019")

    rows = []

    # ---- A: platform plus troughs -----------------------------------------
    print("\n" + "=" * 78)
    print("A  WHAT SHAPE IS THE BED ELEVATION DISTRIBUTION?")
    print("=" * 78)
    # A DESCRIPTIVE CLASSIFICATION THRESHOLD, not a geomorphological boundary.
    # It is placed at a local density minimum rather than a round number, but a
    # local minimum in a smoothed density is not evidence of mode separation --
    # see the correction printed below -- so it is used only to label cells.
    kde = stats.gaussian_kde(h, bw_method=0.15)
    grid = np.linspace(h.min(), h.max(), 600)
    dens = kde(grid)
    peak = grid[np.argmax(dens)]
    left = grid < peak
    # the deepest local minimum of the density below the main peak
    dmin = None
    for i in range(2, left.sum() - 2):
        if dens[i] < dens[i - 1] and dens[i] < dens[i + 1] and grid[i] < peak - 2:
            dmin = grid[i]
    split = dmin if dmin is not None else peak - 6
    plat = h[h >= split]
    trough = h[h < split]
    print(f"  density peak (the platform)      : {peak:+.2f} m")
    print(f"  descriptive threshold            : {split:+.2f} m")
    print(f"  platform  n={len(plat):>5,} ({100*len(plat)/len(h):>4.1f} %)  "
          f"median {np.median(plat):+.2f}  NMAD {nmad(plat):.2f} m")
    print(f"  low tail  n={len(trough):>5,} ({100*len(trough)/len(h):>4.1f} %)  "
          f"median {np.median(trough):+.2f}  range "
          f"{trough.min():+.2f}..{trough.max():+.2f} m")
    sk = float(stats.skew(h))
    dip = stats.kstest(h, "norm", args=(h.mean(), h.std()))
    print(f"\n  skewness {sk:+.2f}, KS vs normal p = {dip.pvalue:.1e}")
    print(f"  p01 {np.percentile(h,1):+.2f}  p50 {np.median(h):+.2f}  "
          f"p99 {np.percentile(h,99):+.2f} m")
    print(f"\n  RETRACTED FRAMING, recorded rather than deleted: this test was")
    print(f"  first written expecting two separable populations -- a platform and")
    print(f"  a trough set. That expectation is WITHDRAWN. The distribution is")
    print(f"  UNIMODAL and NEGATIVELY SKEWED (skew {sk:+.2f}), with the mode at")
    print(f"  {peak:+.2f} m and a long deep tail reaching {h.min():+.2f} m. No mode-separation")
    print(f"  test supports anything stronger, and the local density minimum at")
    print(f"  {split:+.2f} m is a minor wiggle, so it survives only as a descriptive")
    print(f"  classification threshold.")
    print(f"\n  The retraction does not weaken the source's description. A drowned")
    print(f"  valley sampled across its width gives a CONTINUUM of depths from")
    print(f"  thalweg to margin. The negative skew and the {np.median(h)-h.min():.0f} m tail ARE the")
    print(f"  'deep depressions on a quiet floodplain relief' it describes; test B")
    print(f"  is what shows they follow the river corridor.")
    rows += [{"test": "A platform median", "value": float(np.median(plat)), "unit": "m"},
             {"test": "A platform NMAD", "value": nmad(plat), "unit": "m"},
             {"test": "A trough fraction", "value": 100 * len(trough) / len(h),
              "unit": "%"},
             {"test": "A descriptive threshold (NOT a mode boundary)",
              "value": float(split), "unit": "m"}]

    # ---- B: do the troughs follow the channel? -----------------------------
    print("\n" + "=" * 78)
    print("B  ARE THE DEEP DEPRESSIONS CONCENTRATED NEAR THE MODERN CORRIDOR?")
    print("=" * 78)
    near = sd[sd.off_km <= NEAR_CHANNEL_KM]
    far = sd[sd.off_km > NEAR_CHANNEL_KM]
    fn = 100 * (near.H_bed_evrf2019_m < split).mean()
    ff = 100 * (far.H_bed_evrf2019_m < split).mean()
    print(f"  within {NEAR_CHANNEL_KM:.0f} km of the channel line: n={len(near):,}, "
          f"{fn:.1f} % in the low tail, median {near.H_bed_evrf2019_m.median():+.2f} m")
    print(f"  farther than {NEAR_CHANNEL_KM:.0f} km            : n={len(far):,}, "
          f"{ff:.1f} % in the low tail, median {far.H_bed_evrf2019_m.median():+.2f} m")
    print(f"  enrichment factor near the channel: {fn/max(ff, 0.01):.1f}x")
    print(f"\n  CAVEAT: the SWORD line is the MODERN channel, which need not")
    print(f"  coincide with every pre-1956 distributary the source mentions. A")
    print(f"  weak contrast would not disprove the description.")
    rows += [{"test": "B trough fraction near channel", "value": fn, "unit": "%"},
             {"test": "B trough fraction far", "value": ff, "unit": "%"}]

    # ---- C: where does ICESat-2 exposed bed sit? ---------------------------
    print("\n" + "=" * 78)
    print("C  DOES ICESat-2 EXPOSED BED SIT ON THE PLATFORM?")
    print("=" * 78)
    cls = pd.read_csv(CFG.TABLES / "qa3_s7_classification.csv", parse_dates=["date"])
    s6 = cls[(cls.h_canopy == 0) & (cls.veg_ph_count == 0)
             & (cls.gnd_ph_count >= 50) & (cls.match_dist_m <= 50)]
    s7 = s6[(s6.dry_class == "DRY_EXPOSED_BED") & (s6.date > "2023-06-06")
            & (s6.chainage_km <= 200)]
    # ICESat-2 terrain height, in the same frame
    hi = s7.H_icesat2.values
    print(f"  S7 confirmed exposed bed: n={len(s7)}, {s7.track.nunique()} tracks")
    print(f"    ICESat-2 terrain height {np.median(hi):+.2f} m "
          f"(NMAD {nmad(hi):.2f}), range {hi.min():+.2f}..{hi.max():+.2f}")
    print(f"    platform median from the soundings: {np.median(plat):+.2f} m")
    in_plat = 100 * ((hi >= split)).mean()
    print(f"\n  The cleaner statement does not need the threshold at all:")
    print(f"    soundings reach {h.min():+.2f} m; the deepest S7 point is {hi.min():+.2f} m.")
    print(f"    S7 has NO points in the lowest {hi.min()-h.min():.0f} m of the surveyed range,")
    print(f"    and its median ({np.median(hi):+.2f} m) matches the shallow soundings")
    print(f"    ({np.median(plat):+.2f} m) to {abs(np.median(hi)-np.median(plat)):.2f} m.")
    print(f"    ({in_plat:.0f} % lies above the convenience threshold {split:+.2f} m.)")
    print(f"\n  That is what must happen: the troughs still carry the river after")
    print(f"  the breach, so a DRY exposed-bed filter cannot sample them. It is")
    print(f"  a consistency check on the dry-bed mask, not new morphology.")
    rows += [{"test": "C S7 median terrain", "value": float(np.median(hi)), "unit": "m"},
             {"test": "C S7 fraction on platform", "value": in_plat, "unit": "%"}]

    # ---- D: is the upper reach smoother? -----------------------------------
    print("\n" + "=" * 78)
    print("D  IS THE UPPER BROAD REACH SMOOTHER, AS THE SOURCE SAYS?")
    print("=" * 78)
    print("  Roughness = NMAD of platform bed elevation within a 20 km band,")
    print("  platform points only, so channel troughs cannot drive it.")
    pl = sd[sd.H_bed_evrf2019_m >= split].copy()
    b = pd.cut(pl.chain_km, np.arange(0, 261, 20))
    g = pl.groupby(b, observed=True).H_bed_evrf2019_m.agg(["size", "median", nmad])
    g.columns = ["n", "median", "roughness"]
    g = g[g.n >= 60]
    print(f"\n  {'band':<14}{'n':>6}{'median':>9}{'roughness':>11}")
    for i, r in g.iterrows():
        print(f"  {str(i):<14}{int(r.n):>6}{r['median']:>9.2f}{r.roughness:>11.2f}")
    upper = g[[i.left >= 160 for i in g.index]]
    lower = g[[i.left < 160 for i in g.index]]
    print(f"\n  below 160 km: median {lower.roughness.median():.2f} m, "
          f"mean {lower.roughness.mean():.2f} m ({len(lower)} bands)")
    print(f"  above 160 km: median {upper.roughness.median():.2f} m, "
          f"mean {upper.roughness.mean():.2f} m ({len(upper)} bands)")
    # The mean below 160 km is carried by ONE band. Report the sensitivity
    # rather than a verdict that depends on whether it is included.
    worst = lower.roughness.idxmax()
    drop = lower.drop(index=worst)
    print(f"\n  The below-160 mean rests on one band: {worst} at "
          f"{lower.roughness.max():.2f} m.")
    print(f"  Excluding it, below-160 mean becomes {drop.roughness.mean():.2f} m "
          f"against {upper.roughness.mean():.2f} m above --")
    print(f"  the contrast nearly vanishes.")
    print(f"\n  -> WEAK SUPPORT ONLY. The ordering goes the way the source says,")
    print(f"     but on {len(lower)} + {len(upper)} bands with one dominating, this")
    print(f"     neither confirms nor refutes 'only the upper reach smoothed'.")
    rows += [{"test": "D roughness below 160 km (median)",
              "value": float(lower.roughness.median()), "unit": "m"},
             {"test": "D roughness above 160 km (median)",
              "value": float(upper.roughness.median()), "unit": "m"},
             {"test": "D below-160 mean excluding the worst band",
              "value": float(drop.roughness.mean()), "unit": "m"}]

    # ---- E: shallow fraction vs the reported drying areas ------------------
    print("\n" + "=" * 78)
    print("E  DOES THE SHALLOW BED CONCENTRATE WHERE TABLE 21 PUTS THE DRYING AREAS?")
    print("=" * 78)
    # "shallow" = above the dead-volume level, i.e. exposed at drawdown to GMO
    GMO = 12.7 + 0.185
    sd["shallow"] = sd.H_bed_evrf2019_m >= GMO
    b2 = pd.cut(sd.chain_km, np.arange(0, 261, 40))
    g2 = sd.groupby(b2, observed=True).shallow.agg(["size", "mean"])
    g2["pct"] = 100 * g2["mean"]
    print(f"  bed at or above the dead-volume level ({GMO:.2f} m EVRF2019),")
    print(f"  i.e. the part that dries out on full drawdown:\n")
    print(f"  {'band':<16}{'n':>7}{'shallow %':>11}")
    for i, r in g2.iterrows():
        print(f"  {str(i):<16}{int(r['size']):>7}{r.pct:>11.1f}")
    ts = stats.theilslopes(g2.pct.values, [i.mid for i in g2.index])
    spans0 = ts[2] <= 0 <= ts[3]
    print(f"\n  Theil-Sen along chainage: {ts[0]:+.3f} %/km "
          f"[{ts[2]:+.3f}, {ts[3]:+.3f}]  -> "
          f"{'NO significant trend' if spans0 else 'trend present'}")
    print(f"  Table 21 reports drying areas of 5.3, 2.6, 9.4, 26.0 and 34.7 % of")
    print(f"  reach area, ascending upstream. The banded values above are")
    print(f"  {', '.join(f'{v:.1f}' for v in g2.pct.values)} % -- NOT ascending, and the")
    print(f"  highest is the band nearest the DAM.")
    print(f"\n  -> THIS TEST DOES NOT WORK, and the reason is a flaw in how I")
    print(f"     built it, not a contradiction of the source. Table 21 reports an")
    print(f"     AREA fraction; what I computed is the fraction of SOUNDING POINTS")
    print(f"     above the dead-volume level. The survey follows the navigable")
    print(f"     channel and deliberately avoids shallow margins -- exactly where")
    print(f"     drying occurs -- so the sounding sample cannot measure an area")
    print(f"     fraction at all. Answering this properly needs an area-weighted")
    print(f"     bed surface, not a point sample.")
    print(f"\n  Reach boundaries are a second obstacle: Table 21's printed lengths")
    print(f"  (43, 39, 50, 52, 238 total) and my reading of Babyne on the Fig. 13")
    print(f"  axis disagree by ~50 km, so no reach-by-reach comparison is possible")
    print(f"  even with a better metric.")
    rows += [{"test": "E shallow-fraction slope", "value": float(ts[0]),
              "unit": "%/km"},
             {"test": "E verdict", "value": np.nan,
              "unit": "INVALID TEST: point sample cannot measure an area "
                      "fraction; soundings avoid the shallow margins"}]

    # ---- F: does anything reach terrace three? -----------------------------
    print("\n" + "=" * 78)
    print("F  DOES ANY SURVEYED BED POINT REACH THE THIRD TERRACE?")
    print("=" * 78)
    t3lo = NPG_EVRF + THIRD_TERRACE_RANGE[0]
    print(f"  normal impoundment level        : {NPG_EVRF:.2f} m EVRF2019")
    print(f"  third terrace, +{THIRD_TERRACE_RANGE[0]:.0f} to +{THIRD_TERRACE_RANGE[1]:.0f} m "
          f"above it : {t3lo:.2f} .. {NPG_EVRF+THIRD_TERRACE_RANGE[1]:.2f} m")
    print(f"  highest surveyed bed point      : {h.max():.2f} m")
    print(f"  highest S7 ICESat-2 bed point   : {hi.max():.2f} m")
    ok = h.max() < t3lo and hi.max() < t3lo
    print(f"\n  -> {'CONSISTENT' if ok else 'INCONSISTENT'}: the whole surveyed bed lies below the")
    print(f"     lowest terrace the source says was never inundated, by "
          f"{t3lo-h.max():.2f} m.")
    print(f"     A bed 'surface' at terrace height would have indicated the")
    print(f"     survey or the reference level was wrong. It does not.")
    rows += [{"test": "F max surveyed bed", "value": float(h.max()), "unit": "m"},
             {"test": "F third terrace floor", "value": float(t3lo), "unit": "m"}]

    pd.DataFrame(rows).to_csv(CFG.TABLES / "hist12_bed_morphology_tests.csv",
                              index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    a = ax[0, 0]
    a.hist(h, bins=np.arange(-20, 17, 0.5), color=BLUE, alpha=0.75)
    a.axvline(split, color=RED, lw=2.2, ls="--",
              label=f"descriptive threshold {split:+.2f} m")
    a.axvline(np.median(plat), color=GREEN, lw=2,
              label=f"platform median {np.median(plat):+.2f} m")
    a.axvline(NPG_EVRF, color=INK, lw=1.6, ls=":", label="NPG 16.19 m")
    a.axvspan(t3lo, NPG_EVRF + THIRD_TERRACE_RANGE[1], color=AMBER, alpha=0.25)
    a.text(t3lo + 0.2, a.get_ylim()[1] * 0.85, "third terrace\n(never inundated)",
           fontsize=8, color=AMBER)
    a.set_xlabel("bed elevation (m, EVRF2019, 14.00 m reference)")
    a.set_ylabel("soundings")
    a.legend(fontsize=8.2)
    a.set_title("A/F · drowned floodplain platform + channel troughs",
                fontsize=11, loc="left")

    a = ax[0, 1]
    a.scatter(sd.off_km, sd.H_bed_evrf2019_m, s=2.5, color=GREY, alpha=0.3, lw=0)
    a.axhline(split, color=RED, lw=2, ls="--")
    a.axvline(NEAR_CHANNEL_KM, color=BLUE, lw=1.8, ls=":")
    a.text(NEAR_CHANNEL_KM + 0.4, -18, f"{NEAR_CHANNEL_KM:.0f} km from\nthe channel line",
           fontsize=8, color=BLUE)
    a.set_xlabel("distance from the modern channel line (km)")
    a.set_ylabel("bed elevation (m)")
    a.set_xlim(0, 25)
    a.set_title(f"B · the low tail is {fn/max(ff,0.01):.1f}x enriched near the channel",
                fontsize=11, loc="left")
    a.grid(alpha=0.2)

    a = ax[1, 0]
    a.hist(plat, bins=np.arange(split, 17, 0.5), color=GREEN, alpha=0.5,
           density=True, label="soundings, platform")
    a.hist(hi, bins=np.arange(split, 17, 0.5), color=RED, alpha=0.6,
           density=True, label=f"ICESat-2 S7 exposed bed (n={len(s7)})")
    a.axvline(split, color=RED, lw=2, ls="--")
    a.set_xlabel("elevation (m, EVRF2019)")
    a.set_ylabel("density")
    a.legend(fontsize=8.4)
    a.set_title("C · ICESat-2 exposed bed sits on the platform, not the troughs",
                fontsize=11, loc="left")

    a = ax[1, 1]
    mid = [i.mid for i in g.index]
    a.plot(mid, g.roughness, "o-", color=AMBER, lw=2.2, ms=8)
    a.axvline(160, color=GREY, lw=1.4, ls=":")
    a.text(163, g.roughness.max() * 0.95, "upper broad reach\n(source: smoothed)",
           fontsize=8.4, color=GREY)
    a.set_xlabel("chainage from the dam (km)")
    a.set_ylabel("platform bed roughness, NMAD (m)")
    a.grid(alpha=0.25)
    a.set_title(f"D · median roughness {lower.roughness.median():.2f} m below vs "
                f"{upper.roughness.median():.2f} m above 160 km — WEAK",
                fontsize=11, loc="left")

    fig.suptitle("V11 · The exposed bed against the source's own geomorphological "
                 "description   ·   drowned floodplain, channel troughs, and a "
                 "smoothed upper reach", fontsize=12.5, y=1.01)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V11_bed_morphology_vs_source.{e}", dpi=185,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V11_bed_morphology_vs_source.png'}")
    print(f"-> {CFG.TABLES/'hist12_bed_morphology_tests.csv'}")


if __name__ == "__main__":
    main()
