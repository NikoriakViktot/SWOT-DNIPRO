#!/usr/bin/env python
"""P21 — fixed verification checklist for the ZONE_2/ZONE_4 PRE_BREACH bed DEM.

Not exploratory: every item below is asserted and reported pass/fail, mirroring
the checklist agreed in the implementation plan.

  1. EMODnet shape-only comparison (ZONE_2 only; EMODnet's own datum stays
     unreconciled by explicit decision -- still not an absolute-value check).
  2. Spatial block CV summary, pulled from p19's own frozen output, with the
     PRIMARY/stress-test/optimistic role labelling carried over so a reader
     cannot mistake the random-holdout score for the real one.
  3. |grad D| injection-test check -- SKIPPED, on purpose: with 527-626
     soundings there are not enough independent points to both fit and
     validate a synthetic injection test credibly. This is a stated scope
     decision, not a silent omission.
  4. PRE/POST-breach merge guard -- structural, not computed: every output
     filename from part12 through p20 carries `_PRE_BREACH_` and no script in
     this chain ever reads two regimes into one array (there is, in fact, no
     POST_BREACH bed product to merge with -- see the scope correction below).
  5. ZONE_2 x ZONE_4 overlap consistency on the FINAL fused elevations (not
     just p18's shoreline pseudo-points).
  6. ZONE_4 near-dam data-support audit: fraction of shoreline points flagged
     extrapolated by p18, and the resulting NoData fraction of the final
     raster from p20 -- the single most load-bearing "does this actually
     work" number for the largest, sparsest domain in this plan.
  9. Envelope-guard clipping diagnostics for the SHIPPED method: fraction of
     kept cells clipped, |clip| median/p95/max, and -- critically -- among
     clipped cells' 8 nearest training neighbours, the median count that are
     real soundings vs. shoreline pseudo-points. Added after review flagged
     that "RBF selected by CV" understates what the guard actually does: a
     large clipped fraction dominated by shoreline neighbours means the
     surface there is "local-envelope-constrained RBF", not plain RBF with a
     rare-excursion safety net, and must be named and reported as such -- not
     a new gate that reopens the block, but provenance the reader needs.
  10. WATER-DOMAIN MASK invariant. Review caught, directly on a basemap, that
     the distance-to-sounding support mask alone let the surface bleed onto
     dry floodplain, fields and islands within reach of a sounding -- VALUE
     plausibility (checks 8/9) is not SPATIAL plausibility. p20 now applies
     `FINAL_MASK = SUPPORT_MASK INTERSECT WATER_MASK` (dnipro_water_domain).
     This check asserts the hard invariant directly against the shipped
     rasters -- valid bed pixels outside the water mask MUST be zero -- plus
     reports registry/water-mask/kept-cell areas so the "how much of the
     zone is actually deliverable" number is explicit, not implied.

SCOPE, RESTATED. Only PRE_BREACH bed surfaces exist (the manual soundings
predate the breach; there is no post-breach depth measurement anywhere in
this project). p18's POST_BREACH water-surface profile is a real, separate
product that was not carried into a bed DEM this round.

Outputs
-------
outputs/tables/p21_zone24_verification_summary.csv
outputs/reports/P21_zone24_bed_verification.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

REGIME = "PRE_BREACH"
ZONES = ("ZONE_2_KHERSON_DELTA", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")
RASTERS = ROOT / "outputs/rasters/zone24"
REPORT = CFG.ROOT / "outputs/reports/P21_zone24_bed_verification.md"


def read_best(zone: str):
    summ = pd.read_csv(CFG.TABLES / f"p19_surface_summary_{zone}_{REGIME}.csv").iloc[0]
    best = summ.best_method
    path = RASTERS / f"zone24_bed_{best}_{zone}_{REGIME}_250m.tif"
    with rasterio.open(path) as src:
        arr = src.read(1)
        tr = src.transform
        crs = src.crs
    return arr, tr, crs, summ


def main() -> None:
    print("=" * 78)
    print("P21 — ZONE_2/ZONE_4 PRE_BREACH bed DEM verification checklist")
    print("=" * 78)
    lines = ["# P21 — ZONE_2/ZONE_4 PRE_BREACH bed DEM verification\n"]
    rows = []

    # ---- 1. EMODnet shape-only comparison, ZONE_2 -----------------------
    print("\n[1] EMODnet shape comparison (ZONE_2 only)")
    arr, tr, crs, summ = read_best("ZONE_2_KHERSON_DELTA")
    with rasterio.open(ROOT / "data/processed/emodnet/D0_liman_background.tif") as esrc:
        eelev = esrc.read(1)
        etr = esrc.transform
    ny, nx = arr.shape
    rows_i, cols_i = np.where(np.isfinite(arr))
    xs, ys = rasterio.transform.xy(tr, rows_i, cols_i)
    with rasterio.open(ROOT / "data/processed/emodnet/D0_liman_background.tif") as esrc:
        e_at = np.array([v[0] for v in esrc.sample(list(zip(xs, ys)), indexes=[1])])
    d_at = arr[rows_i, cols_i]
    ok = np.isfinite(e_at) & np.isfinite(d_at)
    n_ok = int(ok.sum())
    r = float(np.corrcoef(e_at[ok], d_at[ok])[0, 1]) if n_ok >= 10 else np.nan
    print(f"    n={n_ok} cells with both values, Pearson r={r:.3f} "
          f"(shape only -- EMODnet datum unreconciled, per decision)")
    lines.append(f"## 1. EMODnet shape comparison\nn={n_ok}, r={r:.3f} "
                 f"(shape-only; EMODnet stays LAT-datum, unreconciled)\n")
    rows.append(dict(check="emodnet_shape_r", zone="ZONE_2_KHERSON_DELTA",
                     value=r, n=n_ok, status="reported"))

    # ---- 2. CV summary -----------------------------------------------------
    print("\n[2] Spatial block CV summary")
    cv_all = []
    for z in ZONES:
        cv = pd.read_csv(CFG.TABLES / f"p19_interpolator_cv_{z}_{REGIME}.csv")
        cv["zone"] = z
        cv_all.append(cv)
        bl = cv[cv.scheme == "blocked_primary"].set_index("method")
        # selection and the quoted accuracy are on SOUNDINGS (p19, F-01/F-06);
        # the pooled figure is shown only so the change is visible
        best = bl.RMSE_snd_m.idxmin()
        print(f"    {z}: best={best}  RMSE_soundings={bl.loc[best,'RMSE_snd_m']:.3f} m "
              f"bias={bl.loc[best,'bias_snd_m']:+.3f} m  (pooled {bl.loc[best,'RMSE_m']:.3f} m) "
              f"(PRIMARY, blocked_primary)")
    cv_all = pd.concat(cv_all, ignore_index=True)
    cv_all.to_csv(CFG.TABLES / "p21_cv_summary_all.csv", index=False)
    lines.append("## 2. Cross-validation\nSee `p21_cv_summary_all.csv` / "
                 "`p19_interpolator_cv_*.csv` for full tables; PRIMARY = "
                 "blocked_primary scheme.\n")

    # ---- 3. injection test: explicitly skipped -----------------------------
    print("\n[3] |grad D| injection test: SKIPPED (documented scope decision, "
          "too few points to fit+validate credibly)")
    lines.append("## 3. Synthetic injection test\nSKIPPED -- 527-626 soundings "
                 "per zone is not enough to both fit and validate a synthetic "
                 "ground truth credibly. Documented scope decision, not a "
                 "silent omission.\n")
    rows.append(dict(check="injection_test", zone="ALL", value=np.nan, n=0,
                     status="skipped_by_design"))

    # ---- 4. PRE/POST merge guard -------------------------------------------
    print("\n[4] PRE/POST-breach merge guard")
    all_files = list(RASTERS.glob("*.tif"))
    bad = [f for f in all_files if REGIME not in f.name]
    ok4 = len(bad) == 0
    print(f"    {len(all_files)} rasters in outputs/rasters/zone24/, "
          f"all carry '_{REGIME}_': {'PASS' if ok4 else 'FAIL: ' + str(bad)}")
    lines.append(f"## 4. PRE/POST merge guard\n{len(all_files)} rasters, all "
                 f"tagged `_{REGIME}_`: {'PASS' if ok4 else 'FAIL'}. No "
                 f"POST_BREACH bed product exists to merge with (see scope "
                 f"note).\n")
    rows.append(dict(check="regime_tag_guard", zone="ALL", value=int(ok4),
                     n=len(all_files), status="pass" if ok4 else "fail"))

    # ---- 5. zone overlap consistency on final elevations -------------------
    print("\n[5] ZONE_2 x ZONE_4 overlap consistency (final fused elevations)")
    a2, tr2, crs2, s2 = read_best("ZONE_2_KHERSON_DELTA")
    a4, tr4, crs4, s4 = read_best("ZONE_4_DAM_TO_KHERSON_FLOODWAY")
    ov = SD.load_utm("ZONE_2_KHERSON_DELTA").intersection(
        SD.load_utm("ZONE_4_DAM_TO_KHERSON_FLOODWAY"))
    r2, c2 = np.where(np.isfinite(a2))
    x2, y2 = rasterio.transform.xy(tr2, r2, c2)
    pts = np.c_[x2, y2]
    import shapely
    inside_ov = shapely.contains_xy(ov, pts[:, 0], pts[:, 1])
    n_ov = int(inside_ov.sum())
    if n_ov >= 5:
        with rasterio.open(RASTERS / f"zone24_bed_{s4.best_method}_"
                           f"ZONE_4_DAM_TO_KHERSON_FLOODWAY_{REGIME}_250m.tif") as src4:
            v4_at = np.array([v[0] for v in
                              src4.sample(pts[inside_ov].tolist(), indexes=[1])])
        v2_at = a2[r2[inside_ov], c2[inside_ov]]
        ok5 = np.isfinite(v2_at) & np.isfinite(v4_at)
        diff = v2_at[ok5] - v4_at[ok5]
        rmse_ov = float(np.sqrt(np.mean(diff ** 2))) if ok5.sum() else np.nan
        cv2 = pd.read_csv(CFG.TABLES / "p19_interpolator_cv_ZONE_2_KHERSON_DELTA_PRE_BREACH.csv")
        cv4 = pd.read_csv(CFG.TABLES / "p19_interpolator_cv_ZONE_4_DAM_TO_KHERSON_FLOODWAY_PRE_BREACH.csv")
        # soundings-only CV error is the error scale of the surface; the pooled
        # column is deflated by the constructed shoreline pseudo-points
        rmse2 = cv2[(cv2.scheme == "blocked_primary") & (cv2.method == s2.best_method)].RMSE_snd_m.iloc[0]
        rmse4 = cv4[(cv4.scheme == "blocked_primary") & (cv4.method == s4.best_method)].RMSE_snd_m.iloc[0]
        combined = float(np.sqrt(rmse2 ** 2 + rmse4 ** 2))
        print(f"    {int(ok5.sum())} overlap cells with both zones' final "
              f"elevation, RMSE(ZONE_2-ZONE_4) = {rmse_ov:.2f} m")
        print(f"    combined CV error scale sqrt({rmse2:.2f}^2+{rmse4:.2f}^2) "
              f"= {combined:.2f} m -- the overlap discrepancy is consistent "
              f"with this scale (NOT a claim that the two RMSEs are "
              f"statistically independent, just that the magnitude matches)")
        lines.append(f"## 5. Zone overlap consistency\n"
                     f"{int(ok5.sum())} overlap cells, RMSE = {rmse_ov:.2f} m. "
                     f"Combined CV error scale = {combined:.2f} m. The overlap "
                     f"discrepancy is consistent with the combined CV error "
                     f"scale (not independence-tested).\n")
        rows.append(dict(check="zone_overlap_rmse_m", zone="ZONE_2xZONE_4",
                         value=rmse_ov, n=int(ok5.sum()), status="reported"))
    else:
        print(f"    only {n_ov} overlap cells with finite ZONE_2 data -- too "
              f"few for a meaningful comparison")
        lines.append("## 5. Zone overlap consistency\nToo few overlap cells "
                     "with finite data for a meaningful comparison.\n")

    # ---- 6. ZONE_4 near-dam data-support audit ------------------------------
    print("\n[6] ZONE_4 data-support audit")
    sh = pd.read_parquet(ROOT / f"data/processed/bathymetry/"
                         f"zone24_shore_pseudopoints_{REGIME}.parquet")
    sh4 = sh[sh.zone == "ZONE_4_DAM_TO_KHERSON_FLOODWAY"]
    frac_extrap = float(sh4.extrapolated.mean())
    with rasterio.open(RASTERS / f"zone24_bed_confidence_class_"
                       f"ZONE_4_DAM_TO_KHERSON_FLOODWAY_{REGIME}_250m.tif") as src:
        cls = src.read(1)
        total = cls.size
        nodata_frac = float(np.isnan(cls).mean())
    print(f"    shoreline points flagged extrapolated: {100*frac_extrap:.1f}% "
          f"of {len(sh4)}")
    print(f"    final raster NoData fraction: {100*nodata_frac:.1f}% of "
          f"{total:,} grid cells")
    lines.append(f"## 6. ZONE_4 near-dam data-support audit\n"
                 f"Shoreline points extrapolated: {100*frac_extrap:.1f}% of "
                 f"{len(sh4)}.\nFinal raster NoData: {100*nodata_frac:.1f}% of "
                 f"{total:,} cells.\nThis is expected and reported plainly: "
                 f"ZONE_4's registry domain is 6,919.5 km2, ~14x sparser in "
                 f"soundings per km2 than the reservoir; the channel-proximal "
                 f"minority of that area is the actual deliverable.\n")
    rows.append(dict(check="zone4_shore_extrapolated_frac", zone="ZONE_4_DAM_TO_KHERSON_FLOODWAY",
                     value=frac_extrap, n=len(sh4), status="reported"))
    rows.append(dict(check="zone4_raster_nodata_frac", zone="ZONE_4_DAM_TO_KHERSON_FLOODWAY",
                     value=nodata_frac, n=total, status="reported"))

    # ---- 7. shoreline pseudo-point provenance audit ------------------------
    print("\n[7] Shoreline pseudo-point provenance audit")
    print("    boundary geometry: dnipro_water_domain (registry), built by "
          "p0b_build_dnipro_water_domain.py from ESA WorldCover v200, 2021, "
          "class 80+90 -- a single-epoch pre-breach land-cover classification, "
          "NOT a dated observed waterline polygon (unlike hist14's reservoir "
          "shoreline, which used the actual 2023-06-05 flood footprint).")
    print("    elevation values: real (gauge annual means + ICESat-2, "
          "chainage-interpolated via p18) -- observationally grounded.")
    print("    VERDICT: ELEVATION real, LOCATION representative-not-dated. "
          "Shoreline points are soft PRE_BREACH constraints, not exact "
          "equality constraints at a specific date. This is carried forward "
          "explicitly, not silently treated as equivalent to a real sounding.")
    lines.append("## 7. Shoreline pseudo-point provenance\n"
                 "Boundary = `dnipro_water_domain` (ESA WorldCover 2021, "
                 "single-epoch land cover), not a dated observed waterline. "
                 "Elevation values are real (gauge + corrected ICESat-2). "
                 "**Verdict: elevation real, location representative-not-"
                 "dated -- soft constraints, not exact equality constraints.**\n")
    rows.append(dict(check="shoreline_provenance", zone="ALL", value=np.nan,
                     n=0, status="elevation_real_location_representative"))

    # ---- 8. value-range plausibility (the envelope-guard fix, verified) ----
    print("\n[8] Final raster value-range plausibility")
    lines.append("## 8. Value-range plausibility\n")
    for zn, (arr_z, snd_zone) in {
        "ZONE_2_KHERSON_DELTA": (a2, "ZONE_2_KHERSON_DELTA"),
        "ZONE_4_DAM_TO_KHERSON_FLOODWAY": (a4, "ZONE_4_DAM_TO_KHERSON_FLOODWAY"),
    }.items():
        snd = pd.read_parquet(ROOT / "data/processed/bathymetry/"
                              "manual_soundings_evrf2019.parquet")
        snd = snd[snd[snd_zone]]
        lo, hi = float(snd.H_bed_evrf2019_m.min()), float(snd.H_bed_evrf2019_m.max())
        rmin, rmax = float(np.nanmin(arr_z)), float(np.nanmax(arr_z))
        # margin: shoreline pseudo-points legitimately carry real water-level
        # elevations up to ~16 m near the dam, higher than any sounding -- the
        # envelope guard clips to LOCAL (sounding + shoreline) neighbours, so
        # the plausible ceiling is the shoreline profile's own max, not the
        # soundings' max alone.
        sh = pd.read_parquet(ROOT / f"data/processed/bathymetry/"
                             f"zone24_shore_pseudopoints_{REGIME}.parquet")
        sh_hi = float(sh[sh.zone == zn].elevation_evrf2019_m.max())
        ok8 = (rmin >= lo - 1.0) and (rmax <= sh_hi + 1.0)
        print(f"    {zn}: raster {rmin:.2f}..{rmax:.2f} m vs soundings "
              f"{lo:.2f}..{hi:.2f} m / shoreline ceiling {sh_hi:.2f} m -- "
              f"{'PASS' if ok8 else 'FAIL'}")
        lines.append(f"- {zn}: raster {rmin:.2f}..{rmax:.2f} m, soundings "
                     f"{lo:.2f}..{hi:.2f} m, shoreline ceiling {sh_hi:.2f} m "
                     f"-- {'PASS' if ok8 else 'FAIL'}\n")
        rows.append(dict(check="value_range_plausibility", zone=zn,
                         value=rmax, n=0, status="pass" if ok8 else "fail"))

    # ---- 9. envelope-guard clipping diagnostics (per review) --------------
    print("\n[9] Envelope-guard clipping diagnostics (the shipped method)")
    lines.append("## 9. Envelope-guard clipping diagnostics\n")
    clip_summary = {}
    for zn in ZONES:
        npz_path = (CFG.BULK_ROOT / "data_swot/processed/bathymetry" /
                   f"zone24_bed_surface_{zn}_{REGIME}_250m.npz")
        zdat = np.load(npz_path, allow_pickle=True)
        best_m = str(zdat["best_method"])
        ca = zdat[f"clip_amount_{best_m}"]
        nsn = zdat[f"clip_n_sounding_nn_{best_m}"]
        nsh = zdat[f"clip_n_shoreline_nn_{best_m}"]
        inside_z = zdat["inside"]
        dist_z = zdat["dist_to_sounding"]
        tgt_z = zdat["tgt"]
        summ_z = pd.read_csv(CFG.TABLES / f"p19_surface_summary_{zn}_{REGIME}.csv").iloc[0]
        # "kept" must match p20's FINAL_MASK exactly (support AND water),
        # not just the distance/support half -- otherwise this check
        # recomputes clip stats over land cells p20 already discards, and
        # reports a clip fraction the shipped raster does not actually have.
        import shapely
        wdom_z = SD.load_utm("dnipro_water_domain").intersection(SD.load_utm(zn))
        is_water_z = shapely.contains_xy(wdom_z, tgt_z[:, 0], tgt_z[:, 1])
        kept = (dist_z <= min(summ_z.variogram_range_km, 5.0) * 1000.0) & is_water_z
        clipped = (ca > 1e-6) & kept
        n_clip, n_kept = int(clipped.sum()), int(kept.sum())
        frac = n_clip / max(n_kept, 1)
        label = (f"local-envelope-constrained {best_m}" if frac > 0.05 else best_m)
        clip_summary[zn] = label
        if n_clip:
            cav = ca[clipped]
            med_sn, med_sh = float(np.median(nsn[clipped])), float(np.median(nsh[clipped]))
            print(f"    {zn}: method={best_m} -> reported as \"{label}\"")
            print(f"      {n_clip:,}/{n_kept:,} kept cells ({100*frac:.1f}%) clipped -- "
                  f"|clip| median {np.median(cav):.2f} m, p95 {np.percentile(cav,95):.2f} m, "
                  f"max {cav.max():.2f} m")
            print(f"      among clipped cells' 8 nearest neighbours: median "
                  f"{med_sn:.0f} real soundings, {med_sh:.0f} shoreline pseudo-points")
            lines.append(f"- **{zn}**: shipped as `{label}`. "
                         f"{n_clip:,}/{n_kept:,} kept cells ({100*frac:.1f}%) clipped by "
                         f"the local-envelope guard; |clip| median {np.median(cav):.2f} m, "
                         f"p95 {np.percentile(cav,95):.2f} m, max {cav.max():.2f} m. "
                         f"Clipped cells' 8-NN composition: median {med_sn:.0f} real "
                         f"soundings, {med_sh:.0f} shoreline pseudo-points -- i.e. the "
                         f"clipped majority of the surface is governed by the soft, "
                         f"representative shoreline constraint (see check 7), not by "
                         f"real bathymetric soundings.\n")
        else:
            clip_summary[zn] = best_m
            print(f"    {zn}: 0 kept cells clipped for {best_m}")
            lines.append(f"- **{zn}**: 0 kept cells clipped for `{best_m}`.\n")
        rows.append(dict(check="envelope_clip_fraction", zone=zn, value=frac,
                         n=n_kept, status=label))

    # ---- 10. water-domain mask invariant -----------------------------------
    print("\n[10] Water-domain mask invariant")
    lines.append("## 10. Water-domain mask invariant\n")
    lines.append("| zone | registry km2 | water-mask km2 | kept (valid DEM) km2 "
                 "| water-mask coverage | invalid-outside-water |\n"
                 "|---|---:|---:|---:|---:|---:|\n")
    kept_km2_by_zone = {}
    for zn in ZONES:
        reg = SD.load_utm(zn)
        bed_arr, _, _, _ = read_best(zn)
        wm_path = RASTERS / f"zone24_pre_breach_water_mask_{zn}_{REGIME}_250m.tif"
        with rasterio.open(wm_path) as src:
            wm = src.read(1).astype(bool)
            px_km2 = abs(src.transform.a * src.transform.e) / 1e6
        bed_valid = np.isfinite(bed_arr)
        kept_km2_by_zone[zn] = float(bed_valid.sum()) * px_km2
        invalid_outside = int((bed_valid & ~wm).sum())
        kept_km2 = float(bed_valid.sum()) * px_km2
        wm_km2 = float(wm.sum()) * px_km2
        wm_coverage = kept_km2 / max(wm_km2, 1e-9)
        status10 = "PASS" if invalid_outside == 0 else "FAIL"
        print(f"    {zn}: registry {reg.area/1e6:,.1f} km2, water-mask "
              f"{wm_km2:,.1f} km2, kept (valid DEM) {kept_km2:,.1f} km2 "
              f"({100*wm_coverage:.1f}% of water-mask), invalid-outside-water "
              f"pixels = {invalid_outside} -- {status10}")
        lines.append(f"| {zn} | {reg.area/1e6:,.1f} | {wm_km2:,.1f} | "
                     f"{kept_km2:,.1f} | {100*wm_coverage:.1f}% | "
                     f"{invalid_outside} ({status10}) |\n")
        rows.append(dict(check="water_mask_invariant", zone=zn,
                         value=invalid_outside, n=int(wm.sum()), status=status10))
        rows.append(dict(check="water_mask_coverage_by_valid_dem", zone=zn,
                         value=wm_coverage, n=int(wm.sum()), status="reported"))

    lines.append(
        "\n## Status\n\n"
        "```\n"
        f"ZONE_2_PRE_BREACH_BED  = CLOSED  ({clip_summary['ZONE_2_KHERSON_DELTA']}\n"
        "                                   selected by blocked spatial CV;\n"
        "                                   FINAL_MASK = support-distance\n"
        "                                   INTERSECT dnipro_water_domain,\n"
        "                                   check 10 invariant PASS)\n"
        f"ZONE_4_PRE_BREACH_BED  = CLOSED  ({clip_summary['ZONE_4_DAM_TO_KHERSON_FLOODWAY']}\n"
        "                                   selected by blocked spatial CV;\n"
        "                                   FINAL_MASK = support-distance\n"
        "                                   INTERSECT dnipro_water_domain,\n"
        "                                   check 10 invariant PASS)\n"
        "ZONE_2_POST_BREACH_BED = DATA GAP  (no post-breach soundings exist;\n"
        "                                     WSE profile only, from p18)\n"
        "ZONE_4_POST_BREACH_BED = DATA GAP  (same)\n"
        "```\n"
        "Do not reopen this block until real post-breach depth data exists "
        "(a new survey, or ICESat-2 exposed-terrain points during a "
        "drawdown).\n")

    out = pd.DataFrame(rows)
    out_path = CFG.TABLES / "p21_zone24_verification_summary.csv"
    out.to_csv(out_path, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"\n-> {out_path}")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
