#!/usr/bin/env python
"""MS7D — does the manuscript say what the tables say?

Every validation number the paper-1 text states is regenerated here from
outputs/paper/validation/ms7_summary.csv (and the ms7b slope table) in the
manuscript's own formatting, and must appear verbatim in the markdown. A
second pass scans for numbers superseded by ms7 (the irreproducible v5 values),
and a third checks that the figures drawn from those tables are newer than
the tables. The audit exits non-zero on any failure, so it can gate the docx
build.

    python scripts/ms7d_manuscript_audit.py [path/to/manuscript.md]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "outputs/paper/validation"
FIG = ROOT / "outputs/paper/figures"
MD = ROOT / "outputs/paper/paper1_manuscript_en-v6.md"

# superseded v5 values; (pattern, allowed-context substrings)
SUPERSEDED = [
    (r"\b32 crossings\b", ()), (r"n = 32\b", ()), (r"32 same-day", ()),
    (r"\b21 crossings\b", ()), (r"n = 21\)", ()),
    (r"\b107\b", ()), (r"\b55 overpasses\b", ()),
    (r"\b68 (SWOT )?passes\b", ()), (r"\(68 ", ()),
    (r"15\.4 cm", ("71 passes",)), (r"14\.9 cm", ()), (r"9\.8 cm", ()),
    (r"\b343 crossings\b", ()), (r"number 32\b", ()), (r"none fall in the estuary", ()), (r"\+2\.1 cm", ()),
    (r"same sign and similar size", ()),
    (r"0\.83 \[0\.57", ()), (r"\b23 overpasses\b", ()), (r"0\.94 \(9", ()),
    # wording withdrawn before submission (reviewer points)
    (r"(?i)for the first time", ()), (r"order of magnitude below the observed change", ()),
    (r"largest reservoir for which", ()), (r"established within weeks", ()),
    (r"we are not aware of a published quantification", ()), (r"perched remnants", ()),
    (r"0\.0094", ()), (r"51 times", ()), (r"−0\.009 cm/km", ()), (r"1\.02 cm/km", ()),
    (r"\+3\.276", ()), (r"5628", ()), (r"\+16\.7 cm", ()), (r"0\.93 \[0\.54", ()),
    (r"0\.74 \[0\.12", ()), (r"at most 0\.44", ()), (r"527 day pairs", ()),
    (r"\+2\.2 cm", ()), (r"−1\.6 cm", ()), (r"−4\.8 cm", ()), (r"about −5 cm", ()),
    (r"reservoir gauge records end on 31 December 2021", ("five of the six",)),
    (r"reservoir series end on 31 December 2021", ("Five of the six",)),
]
FIGS_FROM_TABLES = ["F16_closure_offsets.png", "F24_station_panels.png", "F27_covariability_main.png", "F28_water_levels_main.png",
                    "FS_V4_swot_icesat_agreement.png", "F18_swot_icesat_whole_zone_map.png",
                    "F17_reference_surfaces.png", "F19_swot_icesat_egg2015_along_system.png",
                    "F20_vertical_chains.png", "F21_covariability_scatter.png",
                    "F22_gauge_network_correlation.png", "F23_slope_frame_invariance.png",
                    "F25_breach_fortnight_posts.png", "F26_downstream_posts_2023.png"]


def cm(v, sign=True, nd=1):
    s = f"{100 * v:+.{nd}f}" if sign else f"{100 * v:.{nd}f}"
    return s.replace("-", "−")


def ci(lo, hi):
    return f"[{cm(lo)}, {cm(hi)}]"


def f2(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")


def claims() -> list[tuple[str, str]]:
    S = pd.read_csv(VAL / "ms7_summary.csv")
    B = pd.read_csv(VAL / "ms7b_slope_geoid_sampling_summary.csv").set_index("statistic").value

    def g(**kw):
        s = S
        for k, v in kw.items():
            s = s[s[k].isna()] if v is None else s[s[k] == v]
        assert len(s) == 1, (kw, len(s))
        return s.iloc[0]

    V1 = "V1_ATL13_GAUGE_CLOSURE"
    V2 = "V2_ATL13_GAUGE_COVARIABILITY"
    V3 = "V3_ROZUMIVKA_TRANSFER"
    V4 = "V4_SWOT_ICESAT_DIRECT"
    V5 = "V5_LAKESP_PRODUCT"
    V6 = "V6_KHERSON_CLOSURE"
    r3 = dict(claim_id=V3, radius_km=3.0, variant="raw_median")
    out = []
    # V1
    out.append(("V1 n", f"{int(g(claim_id=V1, statistic='n_matchups').value)} pre-breach matchups on "
                        f"{int(g(claim_id=V1, statistic='n_dates').value)} dates"))
    out.append(("V1 mean c", f"−{abs(g(claim_id=V1, statistic='mean_c_m').value):.3f} m"))
    ms = [g(claim_id=V1, statistic=k) for k in ("matchup_set_pearson_r", "matchup_set_theil_sen")]
    out.append(("V1 matchup-set r", f"r = {ms[0].value:.2f} [{ms[0].ci_lo:.2f}, {ms[0].ci_hi:.2f}]"))
    # V2
    r = g(claim_id=V2, statistic="pearson_r")
    out.append(("V2 r", f"r = {r.value:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"))
    out.append(("V2 n", f"{int(g(claim_id=V2, statistic='n_overpasses').value)} independent"))
    out.append(("V2 TS", f"Theil–Sen slope {g(claim_id=V2, statistic='theil_sen').value:.2f}"))
    # V3
    is2 = g(claim_id=V3, statistic="median_m", sensor="ICESat-2")
    out.append(("V3 c_IS2", f"{cm(is2.value)} cm {ci(is2.ci_lo, is2.ci_hi)}"))
    sw = g(statistic="median_m", sensor="SWOT", **r3)
    out.append(("V3 c_SWOT", f"{cm(sw.value)} cm {ci(sw.ci_lo, sw.ci_hi)}"))
    out.append(("V3 n", f"{int(g(statistic='n', sensor='SWOT', **r3).value)} passes"))
    out.append(("V3 nmad", f"NMAD {cm(g(statistic='nmad_m', sensor='SWOT', **r3).value, False)} cm"))
    dd = g(claim_id=V3, statistic="median_diff_m", radius_km=3.0,
           variant="raw_median: median c_SWOT - median c_IS2, both resampled")
    out.append(("V3 diff", f"{cm(dd.value)} cm {ci(dd.ci_lo, dd.ci_hi)}"))
    fr = g(claim_id=V3, statistic="median_m", radius_km=3.0,
           variant="raw_median: c_SWOT - c_IS2 (frozen pre-breach transfer)")
    out.append(("V3 frozen", f"{cm(fr.value)} cm {ci(fr.ci_lo, fr.ci_hi)}"))
    out.append(("V3 5km", f"{cm(g(claim_id=V3, statistic='median_m', radius_km=5.0, variant='raw_median').value)} cm "
                          f"({int(g(claim_id=V3, statistic='n', radius_km=5.0, variant='raw_median').value)} passes)"))
    sc = dict(claim_id=V3, radius_km=3.0, variant="slope_corrected")
    out.append(("V3 slope-corr", f"{cm(g(statistic='median_m', sensor='SWOT', **sc).value)} cm "
                                 f"({int(g(statistic='n', sensor='SWOT', **sc).value)} passes)"))
    out.append(("V3 covar r", f"r = {g(statistic='covar_pearson_r', **r3).value:.2f}, "
                              f"ρ = {g(statistic='covar_spearman_rho', **r3).value:.2f}"))
    # V4
    p = dict(claim_id=V4, variant=None)
    m = g(statistic="median_m", **p)
    out.append(("V4 median", f"{cm(m.value)} cm {ci(m.ci_lo, m.ci_hi)}"))
    out.append(("V4 nmad", f"NMAD {cm(g(statistic='nmad_m', **p).value, False)} cm"))
    out.append(("V4 rmse", f"{cm(g(statistic='rmse_m', **p).value, False)} cm"))
    ex = dict(claim_id=V4, variant="excluding 2023-06-06..06-20 (breach fortnight)")
    out.append(("V4 excl rmse", f"RMSE {cm(g(statistic='rmse_m', **ex).value, False)} cm"))
    out.append(("V4 excl nmad", f"NMAD {cm(g(statistic='nmad_m', **ex).value, False)} cm"))
    for w in ("1-3", "3-10"):
        out.append((f"V4 {w} d", f"NMAD {cm(g(claim_id=V4, statistic='nmad_m', variant=f'timing sensitivity {w} d').value, False)} cm"))
    fm = dict(claim_id=V4, variant="SWOT geoid at the node for both sensors (ellipsoidal difference)")
    out.append(("V4 frame", f"median {1000 * g(statistic='frame_change_median_abs_m', **fm).value:.1f} mm"))
    out.append(("V4 anomaly r", f"r = {g(statistic='pearson_r_anomaly', **p).value:.2f}"))
    out.append(("V4 lever r", f"r = {g(claim_id=V4, statistic='pearson_r_raw', variant='excluding the single R crossing (leverage)').value:.2f}"))
    cf = g(claim_id=V4, statistic="atl13_coef_controlling_zone_chainage")
    out.append(("V4 coef", f"{cf.value:.2f} [{cf.ci_lo:.2f}, {cf.ci_hi:.2f}]"))
    for nm, key in (("production: EGG2015, ATL13 + mean-tide term", "chain production"),
                    ("ATL13 permanent-tide term omitted", "chain tide omitted"),
                    ("ATL13 permanent-tide term with the opposite sign", "chain opposite sign"),
                    ("naive product difference wse - ht_ortho", "chain naive")):
        v = g(claim_id=V4, statistic="median_m", variant=f"chain: {nm}")
        out.append((key, f"{cm(v.value)} cm {ci(v.ci_lo, v.ci_hi)}"))
    out.append(("geoid diff", f"{cm(g(claim_id=V4, statistic='geoid_diff_median_m').value)} cm (SWOT geoid_hght − ATL13 geoid"))
    out.append(("geoid n", f"{int(g(claim_id=V4, statistic='geoid_diff_n_nodes').value)} nodes"))
    out.append(("geoid expected", f"{cm(g(claim_id=V4, statistic='geoid_diff_expected_if_atl13_tide_free_m').value)} cm a tide-free"))
    # V5
    sm = g(claim_id=V5, statistic="median_m", variant="small lakes")
    out.append(("V5 small", f"{cm(sm.value)} cm {ci(sm.ci_lo, sm.ci_hi)}"))
    # V6
    for series, key in (("ICESat-2 pre_breach", "V6 ICESat pre"), ("SWOT RiverSP pre_breach", "V6 RiverSP pre"),
                        ("SWOT RiverSP post_breach", "V6 RiverSP post"), ("SWOT PIXC pre_breach", "V6 PIXC")):
        v = g(claim_id=V6, statistic="median_m", series=series)
        out.append((key, f"{cm(v.value)} cm {ci(v.ci_lo, v.ci_hi)}"))
    out.append(("V6 PIXC radii", f"{cm(g(claim_id=V6, statistic='pixc_median_c_range_over_radii_m').value, False)} cm across aggregation radii"))
    # V1 transfer / drift / gradient
    out.append(("V1 LOO mean", f"RMSE {cm(g(claim_id=V1, statistic='loo_mean_rmse_m').value, False)} cm (max "
                               f"{cm(g(claim_id=V1, statistic='loo_mean_max_abs_m').value, False)} cm)"))
    out.append(("V1 LOO nearest", f"nearest other gauge along the reach with {cm(g(claim_id=V1, statistic='loo_nearest_rmse_m').value, False)} cm"))
    dr = g(claim_id=V1, statistic="drift_per_degree_lon_m")
    out.append(("V1 lon drift", f"{f2(dr.value, 3)} ± {dr.ci_lo:.3f} m per degree of longitude (p {g(claim_id=V1, statistic='drift_per_degree_lon_p').value:.3f})"))
    gr = g(claim_id=V1, statistic="gradient_along_reach_cm_per_km")
    out.append(("V1 gradient", f"{f2(gr.value, 3)} cm/km [{f2(gr.ci_lo, 3)}, +{gr.ci_hi:.3f}]"))
    out.append(("V1 gradient ratio", f"{int(round(3.2232 / max(abs(gr.ci_lo), abs(gr.ci_hi))))} times the largest value"))
    # V8 reference surfaces
    V8 = "V8_REFERENCE_SURFACES"
    out.append(("V8 sep", f"+{100 * g(claim_id=V8, statistic='egm2008_minus_egg2015_median_m').value:.1f} cm (NMAD "
                          f"{100 * g(claim_id=V8, statistic='egm2008_minus_egg2015_nmad_m').value:.1f} cm; range "
                          f"+{100 * g(claim_id=V8, statistic='egm2008_minus_egg2015_min_m').value:.1f} to "
                          f"+{100 * g(claim_id=V8, statistic='egm2008_minus_egg2015_max_m').value:.1f} cm; planar gradient "
                          f"{g(claim_id=V8, statistic='egm2008_minus_egg2015_planar_gradient_mm_per_km').value:.2f} mm/km"))
    out.append(("V8 n", f"{int(g(claim_id=V8, statistic='n_nodes').value)} good SWOT node locations"))
    out.append(("V8 9902", f"{g(claim_id=V8, statistic='epsg9902_min_m').value:.3f} to {g(claim_id=V8, statistic='epsg9902_max_m').value:.3f} m "
                           f"(median {g(claim_id=V8, statistic='epsg9902_median_m').value:.3f} m)"))
    out.append(("V8 NGA", f"SWOT geoid_hght − NGA EGM2008 {cm(g(claim_id=V8, statistic='swot_geoid_minus_nga_egm2008_median_m').value)} cm, "
                          f"NMAD {cm(g(claim_id=V8, statistic='swot_geoid_minus_nga_egm2008_nmad_m').value, False)} cm"))
    out.append(("V4G ATL13-SWOT geoid", f"ATL13 geoid − SWOT geoid_hght {cm(-g(claim_id=V4, statistic='geoid_diff_median_m').value)} cm, "
                                        f"NMAD {cm(g(claim_id=V4, statistic='geoid_diff_nmad_m').value, False)} cm"))
    # V7 downstream posts
    V7 = "V7_DOWNSTREAM_POSTS_2023"
    for st_, key in (("Parutyne", "V7 Parutyne"), ("Mykolaiv", "V7 Mykolaiv"), ("Ochakiv", "V7 Ochakiv")):
        kw = dict(claim_id=V7, station=st_, sensor="SWOT RiverSP", period="breach_fortnight")
        out.append((key, f"r = {g(statistic='pearson_r', **kw).value:.3f} over"))
        out.append((key + " c", f"c {cm(g(statistic='c_median_m', **kw).value)} cm, NMAD {cm(g(statistic='c_nmad_m', **kw).value, False)} cm"))
    rv = dict(claim_id=V7, station="Kherson", variant="river yearbook only, 2023-06-13..07-08")
    out.append(("V7 Kherson river-only", f"r = {g(statistic='pearson_r', **rv).value:.3f} over a {g(statistic='gauge_range_m', **rv).value:.2f} m fall; "
                                         f"median {cm(g(statistic='residual_after_c_pre_median_m', **rv).value)}"))
    pk = dict(claim_id=V7, station="Kherson", variant="peak, river yearbook, 2023-06-06..06-12")
    out.append(("V7 Kherson peak", f"r = {g(statistic='pearson_r', **pk).value:.3f} over {g(statistic='gauge_range_m', **pk).value:.2f} m; "
                                   f"median {cm(g(statistic='residual_after_c_pre_median_m', **pk).value)}"))
    out.append(("V7 yearbooks", f"SD {cm(g(claim_id=V7, statistic='kherson_river_minus_sea_sd_m').value, False)} cm (NMAD "
                                f"{cm(g(claim_id=V7, statistic='kherson_river_minus_sea_nmad_m').value, False)} cm, max "
                                f"{100 * g(claim_id=V7, statistic='kherson_river_minus_sea_max_abs_m').value:.0f} cm) over "
                                f"{int(g(claim_id=V7, statistic='kherson_yearbooks_common_days').value)} common days"))
    rc = g(claim_id=V7, statistic="daily_change_pearson_r", variant="recession 2023-06-13..07-08")
    out.append(("V7 recession", f"r = {rc.value:.2f} [{rc.ci_lo:.2f}, {rc.ci_hi:.2f}]"))
    lag = {v: g(claim_id=V7, statistic="daily_change_lag_pearson_r", variant=v).value
           for v in ("lag +0 d, all 2023", "lag -1 d, all 2023", "lag +1 d, all 2023")}
    out.append(("V7 lag", f"(r = {lag['lag +0 d, all 2023']:.2f}, against {lag['lag -1 d, all 2023']:.2f} at −1 day and "
                          f"{lag['lag +1 d, all 2023']:.2f} at +1 day)"))
    # gauge network
    GN = "GAUGE_NETWORK"
    out.append(("network levels", f"levels r {g(claim_id=GN, statistic='reservoir_level_r_min').value:.2f}–"
                                   f"{g(claim_id=GN, statistic='reservoir_level_r_max').value:.2f} (effective n "
                                   f"{g(claim_id=GN, statistic='reservoir_n_eff_min').value:.0f}–{g(claim_id=GN, statistic='reservoir_n_eff_max').value:.0f}"))
    out.append(("network post", f"{int(g(claim_id=GN, statistic='post_rozumivka_kherson_daily_change_r', variant='from 2023-07-01').n)} day pairs"))
    # slope frames
    F = pd.read_csv(VAL / "ms7b_slope_frames_summary.csv")
    fv = lambda st, var: F[(F.statistic == st) & (F.variant == var)].value.iloc[0]
    out.append(("frames nearest", f"max |ΔS| {fv('max_abs_dS_cm_per_km', 'plus_c_nearest'):.2f} cm/km, "
                                  f"{int(fv('n_overpasses_changed_over_0.01', 'plus_c_nearest'))} of 33"))
    out.append(("frames nearest contrast", f"contrast +{fv('contrast_cm_per_km', 'plus_c_nearest'):.3f} cm/km"))
    out.append(("frames no tide", f"max |ΔS| {fv('max_abs_dS_cm_per_km', 'egg2015_no_tide'):.4f} cm/km; contrast +{fv('contrast_cm_per_km', 'egg2015_no_tide'):.3f}"))
    out.append(("frames linear", f"ΔS = {abs(fv('max_abs_dS_cm_per_km', 'plus_c_linear')):.3f} cm/km on every overpass"))
    # slope geoid sampling
    out.append(("slope contrast nearest", f"{float(B['contrast_nearest']):+.3f} cm/km".replace("-", "−")))
    out.append(("slope median dS", f"{float(B['post_median_abs_dS_cm_km']):.3f} cm/km"))
    return out


def main() -> int:
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MD
    md = md_path.read_text()
    fails = 0
    print(f"== claims in {md_path.name}")
    for key, frag in claims():
        ok = frag in md
        fails += not ok
        print(f"  {'ok ' if ok else 'MISSING'}  {key:<24} {frag}")
    print("== superseded values")
    for pat, allow in SUPERSEDED:
        for i, line in enumerate(md.splitlines(), 1):
            for mt in re.finditer(pat, line):
                ctx = line[max(0, mt.start() - 60): mt.end() + 60]
                if any(a in ctx for a in allow):
                    continue
                fails += 1
                print(f"  FOUND  line {i}: /{pat}/ ...{ctx}...")
    print("== figure numbering follows order of appearance")
    for kind in ("", "S"):
        nums = [int(n) for n in re.findall(rf"\*\*Figure {kind}(\d+)\.\*\*", md)]
        ok = nums == list(range(1, len(nums) + 1))
        fails += not ok
        print(f"  {'ok ' if ok else 'OUT OF ORDER'}  Figure {kind or 'main'}: {nums}")
    print("== figures newer than their tables")
    t = max((VAL / f).stat().st_mtime for f in ("ms7_evidence.csv", "ms7_summary.csv"))
    for f in FIGS_FROM_TABLES:
        p = FIG / f
        ok = p.exists() and (p.stat().st_mtime >= t or f.startswith("F18"))
        fails += not ok
        print(f"  {'ok ' if ok else 'STALE'}  {f}")
    print(f"== {fails} failure(s)")
    if not fails:
        write_build_info(md_path, len(claims()))
    return 1 if fails else 0


PIPELINE = ["scripts/ms5_paper1_zones.py", "scripts/ms6_paper1_figures.py",
            "scripts/ms6b_swot_icesat_crossings.py", "scripts/ms7_validation_paths.py",
            "scripts/ms7b_slope_geoid_sampling.py", "scripts/ms7c_validation_figures.py",
            "scripts/ms7d_manuscript_audit.py", "scripts/ms8_wind_setup_20230405.py",
            "app/prepare_app_data.py", "app/prepare_app_s1_layers.py", "app/prepare_app_icesat2_passes.py"]
MANUSCRIPT_VERSION = "v6.2-rc1"


def write_build_info(md_path: Path, n_claims: int) -> None:
    """A release stamp the companion app shows: which manuscript, which
    evidence build, which pipeline (content hash of its scripts), audit result."""
    import hashlib
    import json
    import subprocess
    from datetime import datetime, timezone
    h = hashlib.sha256()
    for f in PIPELINE:
        h.update((ROOT / f).read_bytes())
    ev = VAL / "ms7_evidence.csv"

    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    info = dict(manuscript_version=MANUSCRIPT_VERSION, manuscript=str(md_path.relative_to(ROOT)),
                manuscript_sha256=hashlib.sha256(md_path.read_bytes()).hexdigest()[:16],
                evidence_built_utc=datetime.fromtimestamp(ev.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                evidence_sha256=hashlib.sha256(ev.read_bytes()).hexdigest()[:16],
                pipeline_sha256=h.hexdigest()[:16], pipeline=PIPELINE,
                audit="passed", audit_claims=n_claims,
                audited_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                git_commit_at_audit=git("rev-parse", "--short", "HEAD"),
                git_dirty_at_audit=bool(git("status", "--porcelain")))
    (VAL / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"== build info -> {(VAL / 'build_info.json').relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
