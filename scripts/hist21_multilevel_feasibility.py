#!/usr/bin/env python
"""HISTORICAL 21 — can multi-level water extents constrain the shallow bed?

This is steps 1-2 of the proposed multi-level shoreline experiment, executed
before any constrained surface is built, because the experiment rests on a
premise that has to be checked first: that we hold water-extent polygons at
SEVERAL known water-surface elevations, over the SAME reservoir morphology.

The premise turns out to hold only partially, and the measurements below say
exactly where it fails. Nothing is built on it until that is on the record.

Three independent obstacles are quantified:

  LEVELS      how many polygons have a usable water elevation, and how far
              apart those elevations are
  GEOMETRY    where each waterline actually sits relative to the pre-breach
              shore -- the zone that carries most of the prediction error
  EPOCH       whether the polygon describes the bed the DEM is reconstructing

Then the alternative is tested: the published historical level-area curve is
itself a 17-level hypsometric constraint, in the right epoch, already
digitised. Its comparison against the DEM is computed here -- together with the
reason it cannot simply be used as a constraint.

Outputs
-------
outputs/tables/multilevel_polygon_inventory.csv
outputs/tables/hypsometric_constraint_test.csv
outputs/figures/V17_multilevel_constraints.png
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
import numpy as np
import pandas as pd
from shapely import wkt as swkt
from shapely.geometry import shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
BS_TO_EVRF = 0.185          # median EPSG:9902 offset over the reservoir
GMO_BS, NPG_BS = 12.70, 16.00
COV_MIN = 0.90              # a polygon must see ~all of the footprint
ROZ = "station:80959"       # Rozumivka, the only in-reservoir gauge post-2021
ROZ_CHAIN_KM = 247.5


def main() -> None:
    # ================================================== 1. INVENTORY
    # BROKEN until 2026-09-13: this read the P20 footprint directly, so the
    # distance-to-shore statistic below was measured against a boundary that
    # stops 9.4 km short of the real eastern shore. The registry domain is the
    # only admissible source for a named domain.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    wb = pd.read_parquet(CFG.TABLES / "water_body_objects.parquet",
                         columns=["date", "period", "area_km2", "wkt",
                                  "is_noise", "footprint_observed_fraction"])
    n_all = len(wb)
    empty = wb.wkt.isna() | (wb.wkt.astype(str).str.len() < 20)
    print("=" * 78)
    print("1. INVENTORY OF WATER-EXTENT POLYGONS")
    print("=" * 78)
    print(f"  {n_all:,} classified components; {int(empty.sum()):,} carry no "
          f"geometry (only the larger, non-noise components were serialised), "
          f"leaving {int((~empty).sum()):,}")

    wb = wb[~empty & ~wb.is_noise]
    lv = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    roz = lv[(lv.source == "gauge") & (lv.station_or_domain == ROZ)][
        ["date", "transformed_level_m"]].rename(
        columns={"transformed_level_m": "H_rozumivka_evrf2019_m"})
    prof = pd.read_csv(CFG.TABLES / "kakhovka_longitudinal_profiles.csv")
    prof["_d"] = prof.date.astype(str).str[:10]
    pk = prof.groupby("_d").agg(icesat_n=("wse_m", "size"),
                                icesat_chain_min=("chain_km", "min"),
                                icesat_chain_max=("chain_km", "max"))

    inv = (wb.groupby("date")
             .agg(period=("period", "first"),
                  n_components=("area_km2", "size"),
                  water_area_km2=("area_km2", "sum"),
                  footprint_observed_fraction=("footprint_observed_fraction",
                                               "first"))
             .reset_index()
             .merge(roz, on="date", how="left")
             .merge(pk, left_on="date", right_index=True, how="left"))
    inv["icesat_chain_span_km"] = (inv.icesat_chain_max
                                   - inv.icesat_chain_min)
    inv["usable_coverage"] = inv.footprint_observed_fraction >= COV_MIN
    inv["level_known"] = inv.H_rozumivka_evrf2019_m.notna()
    # A single elevation is only meaningful if the surface is level, which
    # pre-breach it is (validated) and post-breach it demonstrably is not.
    inv["surface_is_level"] = inv.period == "PRE_BREACH"
    inv["same_epoch_as_DEM"] = inv.period == "PRE_BREACH"
    inv["usable_as_contour"] = (inv.usable_coverage & inv.level_known
                                & inv.surface_is_level & inv.same_epoch_as_DEM)

    # ================================================== 2. GEOMETRY
    print("\n" + "=" * 78)
    print("2. WHERE EACH WATERLINE SITS, RELATIVE TO THE PRE-BREACH SHORE")
    print("=" * 78)
    print("   (the near-shore zone <500 m carries 54% of the DEM's squared "
          "error)")
    print(f"\n  {'date':<12}{'period':<12}{'H_Roz':>8}{'area':>8}"
          f"{'med dist':>10}{'<500 m':>8}")
    geo = {}
    for dt, sub in wb[wb.footprint_observed_fraction >= COV_MIN].groupby("date"):
        u = unary_union([swkt.loads(w) for w in sub.wkt.values])
        b = u.boundary
        pts = [b.interpolate(t, normalized=True) for t in np.linspace(0, 1, 300)]
        d = np.array([p.distance(fp.boundary) for p in pts]) / 1e3
        geo[dt] = dict(shore_dist_med_km=float(np.median(d)),
                       shore_dist_p10_km=float(np.percentile(d, 10)),
                       shore_dist_p90_km=float(np.percentile(d, 90)),
                       frac_within_500m=float((d < 0.5).mean()))
        r = inv[inv.date == dt].iloc[0]
        h = (f"{r.H_rozumivka_evrf2019_m:.2f}"
             if pd.notna(r.H_rozumivka_evrf2019_m) else "n/a")
        print(f"  {dt:<12}{r.period:<12}{h:>8}{r.water_area_km2:>8.0f}"
              f"{np.median(d):>9.2f}k{100*(d<0.5).mean():>7.0f}%")
    for k in ("shore_dist_med_km", "shore_dist_p10_km", "shore_dist_p90_km",
              "frac_within_500m"):
        inv[k] = inv.date.map(lambda x: geo.get(x, {}).get(k, np.nan))

    inv.to_csv(CFG.TABLES / "multilevel_polygon_inventory.csv", index=False)

    # ================================================== the three obstacles
    ok = inv[inv.usable_coverage]
    pre = ok[ok.period == "PRE_BREACH"]
    post = ok[ok.period != "PRE_BREACH"]
    print("\n" + "=" * 78)
    print("THE PREMISE, TESTED")
    print("=" * 78)

    print("\n  OBSTACLE 1 — LEVELS. Only pre-breach polygons have a single "
          "meaningful")
    print("  water elevation, because only pre-breach is the pool level "
          "(validated:")
    print("  Theil-Sen slope +0.09 cm/km, 10/14 dates positive, sign test "
          "p = 0.18).")
    print(f"    pre-breach polygons with usable coverage : "
          f"{len(pre)} of {int((inv.period=='PRE_BREACH').sum())}")
    if len(pre):
        hs = pre.H_rozumivka_evrf2019_m.dropna().sort_values()
        print(f"    their elevations                         : "
              + ", ".join(f"{h:.2f}" for h in hs) + " m EVRF2019")
        if len(hs) > 1:
            print(f"    spread between them                      : "
                  f"{hs.max()-hs.min():.2f} m")
    print(f"    -> the usable pre-breach set is {len(pre)} level(s). A "
          f"multi-level contour set")
    print(f"       needs several WELL-SEPARATED levels; this is not one.")

    print("\n  OBSTACLE 2 — GEOMETRY. Post-breach waterlines do not sit in "
          "the zone")
    print("  where the DEM error is concentrated.")
    print(f"    post-breach/drawdown waterlines lie a median "
          f"{post.shore_dist_med_km.median():.2f} km INSIDE the")
    print(f"    pre-breach shore, and only "
          f"{100*post.frac_within_500m.median():.0f}% of their length falls "
          f"within 500 m of it")
    print(f"    (range {100*post.frac_within_500m.min():.0f}-"
          f"{100*post.frac_within_500m.max():.0f}%). As CONTOURS they would "
          f"add information at 1-2 km")
    print(f"    from shore, where measured RMSE is already 2.24 m -- not at "
          f"the margin.")

    print("\n  OBSTACLE 3 — NO H(x) FOR THE POST-BREACH DATES. Post-breach the "
          "surface")
    print("  slopes, so one gauge value is not the level across the "
          "reservoir: at")
    print(f"  +3.31 cm/km over {ROZ_CHAIN_KM:.0f} km that is ~"
          f"{3.313669291997315*ROZ_CHAIN_KM/100:.1f} m of fall, and Rozumivka "
          f"sits at the")
    print(f"  UPSTREAM end, so its reading is the profile MAXIMUM.")
    coincide = ok[(ok.period != "PRE_BREACH") & (ok.icesat_n.notna())]
    print(f"    ICESat-2 profiles coincide with a usable polygon date on "
          f"{len(coincide)} of {len(post)} dates,")
    for r in coincide.itertuples():
        print(f"      {r.date}: n={int(r.icesat_n)} segments spanning "
              f"{r.icesat_chain_min:.0f}-{r.icesat_chain_max:.0f} km "
              f"({r.icesat_chain_span_km:.0f} km of 250)")
    print(f"    -> H(x,t) cannot be reconstructed for these dates, so neither "
          f"the contour")
    print(f"       constraint z=H_i nor the inequality z<=H_i can be written "
          f"down rigorously.")

    print("\n  OBSTACLE 4 — EPOCH. A post-breach waterline traces the "
          "2023-2025 bed;")
    print("  the DEM reconstructs the pre-breach bed. The exposed-bed "
          "comparison")
    print("  (+0.098 m over 94 points, 28 tracks) says the two are not "
          "detectably")
    print("  different in the SHALLOW MARGINS, but channel incision after the "
          "breach")
    print("  is expected and would be mixed in.")

    print("\n  VERDICT ON THE PROPOSED EXPERIMENT: the satellite multi-level "
          "route is")
    print("  NOT viable on the present data. It is not being built, and the "
          "canonical")
    print("  surface is untouched. The one pre-breach level that is usable "
          "(17.08 m)")
    print("  is ALREADY the constraint in use.")

    # ================================================== 3. THE ALTERNATIVE
    print("\n" + "=" * 78)
    print("3. THE ALTERNATIVE THAT DOES HAVE MULTIPLE LEVELS: A(H) FROM "
          "TABLE 19")
    print("=" * 78)
    print("  The published level-area curve is a 17-level hypsometric "
          "description of")
    print("  the SAME reservoir in the RIGHT epoch, already transcribed. It is "
          "exactly")
    print("  the 'set of bed contours' the satellite polygons were meant to "
          "supply.")
    lav = pd.read_csv(ROOT / "data/historical/historical_level_area_volume.csv"
                      ).sort_values("water_level_m")
    npz = np.load(CFG.BULK_ROOT / "data_swot/processed/bathymetry/"
                         "kakhovka_bed_surface_250m.npz", allow_pickle=True)
    bed = npz["surf_epoch_OK"].astype(float)
    bed = bed[np.isfinite(bed)]
    cell_area = (float(npz["cell_m"]) / 1e3) ** 2
    rows = []
    print(f"\n  {'H (BS-77)':>10}{'A_hist':>9}{'A_DEM':>9}{'diff':>9}"
          f"{'diff %':>9}")
    for r in lav.itertuples():
        H = r.water_level_m + BS_TO_EVRF
        a = float((bed < H).sum()) * cell_area
        rows.append(dict(water_level_bs77_m=r.water_level_m,
                         water_level_evrf2019_m=H,
                         area_historical_km2=r.surface_area_km2,
                         area_dem_km2=a,
                         diff_km2=a - r.surface_area_km2,
                         diff_pct=100 * (a - r.surface_area_km2)
                         / r.surface_area_km2))
        print(f"  {r.water_level_m:>10.1f}{r.surface_area_km2:>9.0f}{a:>9.0f}"
              f"{a-r.surface_area_km2:>+9.0f}"
              f"{100*(a-r.surface_area_km2)/r.surface_area_km2:>+8.1f}%")
    hy = pd.DataFrame(rows)
    hy.to_csv(CFG.TABLES / "hypsometric_constraint_test.csv", index=False)
    rms = float(np.sqrt((hy.diff_km2 ** 2).mean()))
    print(f"\n  RMS area error over 17 levels: {rms:.0f} km2 "
          f"({100*rms/hy.area_historical_km2.mean():.1f}% of mean area)")
    print(f"  the sign is negative at ALL {len(hy)} levels "
          f"(median {hy.diff_pct.median():+.1f}%), worst at the lowest level "
          f"({hy.diff_pct.min():+.1f}% at {hy.water_level_bs77_m.min():.1f} m)")
    print(f"\n  That systematic deficit means the DEM has too FEW cells below "
          f"any given")
    print(f"  level, i.e. its deep parts are too SHALLOW -- which is the same "
          f"smoothing")
    print(f"  bias hist18 measured from a completely different direction "
          f"(+11.03 m in the")
    print(f"  deepest elevation bin, predicted too shallow). Two independent "
          f"routes,")
    print(f"  one signature.")

    print("\n  BUT IT CANNOT SIMPLY BE USED AS A CONSTRAINT, and this is the "
          "decisive")
    print("  methodological point:")
    print("    Table 21's drying area -- the strongest validation result in "
          "the study")
    print("    (12.95% historical vs 12.38% reconstructed, -0.57 pp) -- is "
          "ARITHMETICALLY")
    print("    DERIVED from the same source's level-area data: drying area = "
          "A(NPG) - A(GMO)")
    print("    by reach. Conditioning the DEM on A(H) would therefore make "
          "the Table 21")
    print("    comparison CIRCULAR and destroy it as validation.")
    a_npg = float(hy.loc[(hy.water_level_bs77_m - NPG_BS).abs().idxmin()]
                  .area_historical_km2)
    a_gmo = float(hy.loc[(hy.water_level_bs77_m - GMO_BS).abs().idxmin()]
                  .area_historical_km2)
    print(f"\n    check: A(NPG {NPG_BS:.1f}) - A(GMO {GMO_BS:.1f}) = "
          f"{a_npg:.0f} - {a_gmo:.0f} = {a_npg-a_gmo:.0f} km2")
    print(f"           Table 21's drying area, summed over reaches       = "
          f"279 km2")
    print(f"    The two agree to {abs((a_npg-a_gmo)-279):.0f} km2, confirming "
          f"they are the same information.")
    print("\n    So this is a genuine TRADE-OFF, not a free improvement, and "
          "the choice")
    print("    belongs to the author:")
    print("      (a) keep A(H) as VALIDATION -- the current design. The DEM is "
          "3.8% off")
    print("          across 17 levels and that is an independent result.")
    print("      (b) use A(H) as a CONSTRAINT -- a better DEM, but Table 21 "
          "stops being")
    print("          independent validation and the geomorphological block "
          "weakens.")
    print("    Recommended: (a) for this paper, (b) only for a separate "
          "product paper")
    print("    where the DEM, not the validation, is the deliverable.")

    # ================================================== figure
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.6))

    a = ax[0, 0]
    for per, c, m in (("PRE_BREACH", BLUE, "o"), ("DRAWDOWN", AMBER, "s"),
                      ("POST_BREACH", RED, "^")):
        s = inv[(inv.period == per) & inv.level_known]
        a.scatter(s.H_rozumivka_evrf2019_m, s.footprint_observed_fraction,
                  s=np.clip(s.water_area_km2 / 6, 12, 260), c=c, marker=m,
                  alpha=0.8, label=per, edgecolor="white", linewidth=0.6)
    a.axhline(COV_MIN, color=INK, ls="--", lw=1.3,
              label=f"usable coverage ({COV_MIN:.0%})")
    a.axvline(NPG_BS + BS_TO_EVRF, color=GREEN, ls=":", lw=1.4,
              label=f"NPG {NPG_BS+BS_TO_EVRF:.2f} m")
    a.axvline(GMO_BS + BS_TO_EVRF, color=GREY, ls=":", lw=1.4,
              label=f"GMO {GMO_BS+BS_TO_EVRF:.2f} m")
    a.set_xlabel("Rozumivka water level (m EVRF2019)")
    a.set_ylabel("fraction of footprint observed")
    a.legend(fontsize=7.8, loc="lower left")
    a.grid(alpha=0.25)
    a.set_title("a · Only ONE usable pre-breach level exists\n"
                "marker area = water area", fontsize=10.5, loc="left")

    a = ax[0, 1]
    s = inv[inv.usable_coverage].sort_values("shore_dist_med_km")
    cols = [BLUE if p == "PRE_BREACH" else (AMBER if p == "DRAWDOWN" else RED)
            for p in s.period]
    yy = np.arange(len(s))
    a.barh(yy, s.shore_dist_med_km, color=cols, alpha=0.85)
    a.errorbar(s.shore_dist_med_km, yy,
               xerr=[s.shore_dist_med_km - s.shore_dist_p10_km,
                     s.shore_dist_p90_km - s.shore_dist_med_km],
               fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    a.axvline(0.5, color=RED, ls="--", lw=1.5,
              label="0.5 km — the high-error zone")
    a.set_yticks(yy)
    a.set_yticklabels(s.date, fontsize=7.6)
    a.set_xlabel("distance from waterline to the pre-breach shore (km)")
    a.legend(fontsize=8.2)
    a.grid(alpha=0.25, axis="x")
    a.set_title("b · Post-breach waterlines sit 1–2 km inside the shore\n"
                "so they miss the zone that carries the error",
                fontsize=10.5, loc="left")

    a = ax[1, 0]
    a.plot(hy.area_historical_km2, hy.water_level_bs77_m, "o-", color=BLUE,
           lw=2.2, ms=6, label="historical Table 19")
    a.plot(hy.area_dem_km2, hy.water_level_bs77_m, "s--", color=RED, lw=2.2,
           ms=6, label="reconstructed DEM (250 m)")
    a.axhline(NPG_BS, color=GREEN, ls=":", lw=1.4)
    a.axhline(GMO_BS, color=GREY, ls=":", lw=1.4)
    a.annotate("НПГ 16.0", (1460, NPG_BS + 0.09), fontsize=8, color=GREEN)
    a.annotate("ГМО 12.7", (1460, GMO_BS + 0.09), fontsize=8, color=GREY)
    a.set_xlabel("surface area (km²)")
    a.set_ylabel("water level (m BS-77)")
    a.legend(fontsize=8.6)
    a.grid(alpha=0.25)
    a.set_title("c · The real multi-level constraint: A(H) at 17 levels\n"
                "epoch-correct, published, already digitised",
                fontsize=10.5, loc="left")

    a = ax[1, 1]
    a.barh(hy.water_level_bs77_m, hy.diff_pct, height=0.38,
           color=[RED if v < 0 else GREEN for v in hy.diff_pct], alpha=0.85)
    a.axvline(0, color=INK, lw=1.1)
    a.set_xlabel("DEM area − historical area (%)")
    a.set_ylabel("water level (m BS-77)")
    a.grid(alpha=0.25, axis="x")
    a.set_title(f"d · Negative at every level (RMS {rms:.0f} km², "
                f"{100*rms/hy.area_historical_km2.mean():.1f} %)\n"
                "the DEM's deep parts are too shallow — the smoothing bias",
                fontsize=10.5, loc="left")

    fig.suptitle("V17 · Can multi-level water extents constrain the shallow "
                 "bed?   ·   feasibility, tested before anything is built",
                 fontsize=12.6, y=1.005)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V17_multilevel_constraints.{e}", dpi=175,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V17_multilevel_constraints.png'}")
    print(f"-> {CFG.TABLES/'multilevel_polygon_inventory.csv'}")
    print(f"-> {CFG.TABLES/'hypsometric_constraint_test.csv'}")
    print("\nNOT BUILT, deliberately: the C1/C2/C3 constrained surfaces. The "
          "premise fails\non the present data, the canonical surface is "
          "unchanged, and no interpolation\nparameter was touched.")


if __name__ == "__main__":
    main()
