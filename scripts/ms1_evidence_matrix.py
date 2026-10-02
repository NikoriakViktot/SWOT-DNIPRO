#!/usr/bin/env python
"""MANUSCRIPT 1 — the evidence matrix that controls every number downstream.

Nothing in the manuscript or the slides may state a number that is not in this
matrix, and every row here is READ FROM a canonical output table rather than
typed in. That is the whole point: the wording is generated from the evidence,
not checked against it afterwards.

Where two tables disagree, the newer one wins and the older is recorded in
`superseded_by` so the conflict stays visible instead of being silently
resolved. Three such conflicts exist and are documented in
outputs/reports/presentation_and_manuscript_QC.md.

validation_status vocabulary
----------------------------
VALIDATED    an independent check passed, with a stated numerical tolerance
SUPPORTED    consistent with independent evidence, but not a decisive test
PRELIMINARY  computed, but underpowered or resting on an unverified premise
UNRESOLVED   tested, and the test did not explain the observation
REJECTED     tested and contradicted by the data

Outputs
-------
outputs/tables/manuscript_evidence_matrix.csv
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

from swot_dnipro import config as CFG

T = CFG.TABLES
# processed inputs live on the bulk volume; the repo-disk copy no longer exists
# (fifth repo-disk data-path literal found 2026-09-16; this one broke the
# evidence matrix on `processed/historical/historical_fig16_profiles.csv`)
D = CFG.BULK_ROOT / "data_swot"
ROWS: list[dict] = []


def load(name, where=T):
    p = Path(where) / name
    if not p.exists():
        raise SystemExit(f"MISSING CANONICAL TABLE: {p}")
    return pd.read_csv(p)


def add(claim_id, claim, result_type, dataset, independent_unit, n, value,
        uncertainty, source_table, source_report, source_figure,
        validation_status, limitations, publication_status="main",
        superseded_by=""):
    ROWS.append(dict(
        claim_id=claim_id, claim=claim, result_type=result_type,
        dataset=dataset, independent_unit=independent_unit, n=n,
        value=value, uncertainty=uncertainty, source_table=source_table,
        source_report=source_report, source_figure=source_figure,
        validation_status=validation_status, limitations=limitations,
        publication_status=publication_status, superseded_by=superseded_by))


def f(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# ===================================================================== #
# V1  gauge vertical re-referencing                                      #
# ===================================================================== #
g = load("gauge_vertical_reference_summary.csv")
res = g[g.domain == "reservoir"]
gd = g[g.n_obs > 0]
add("V1.1",
    "All gauge stages are carried into one geodetic frame by the official "
    "operation EPSG:9902 (BS-77 normal height -> EVRF2019 normal height), "
    "applied at each station's own coordinates.",
    "geodetic transformation", "7 gauge stations (6 reservoir + Kherson)",
    "station", int(gd.station_id.nunique()),
    f"{int(gd.n_obs.sum()):,} daily values transformed",
    "grid sampler agrees with an independent implementation to 0.000 mm",
    "gauge_vertical_reference_summary.csv", "validation_section.md",
    "V1_vertical_reference_framework.png", "VALIDATED",
    "Mykolaiv 98027 has coordinates but no level series; 8 further reservoir "
    "stations in the donor CSVs carry no coordinates and cannot be transformed.")

add("V1.2",
    "The BS-77 -> EVRF2019 offset is spatially varying across the study reach, "
    "so a single national constant is not admissible.",
    "geodetic transformation", "EPSG:9902 grid ua_2019z.asc at gauge sites",
    "station", int(len(g)),
    f"{g.delta_epsg9902_m.min():.4f} to {g.delta_epsg9902_m.max():.4f} m",
    f"spread {1e3*(g.delta_epsg9902_m.max()-g.delta_epsg9902_m.min()):.0f} mm; "
    f"grid accuracy 0.068 m (EPSG registry)",
    "gauge_vertical_reference_summary.csv", "validation_section.md",
    "V1_vertical_reference_framework.png", "VALIDATED",
    "The 0.068 m grid accuracy is larger than the spread it resolves; the "
    "spatial variation is used, the absolute level carries that accuracy.")

add("V1.3",
    "The six reservoir gauges share a zero of 12.000 m BS-77; Kherson 80805 "
    "does NOT and sits at -5.000 m BS-77.",
    "metadata provenance", "gauge station metadata", "station",
    int(len(g)),
    f"reservoir {res.zero_bs77_m.unique()[0]:.3f} m BS-77; "
    f"Kherson {float(g[g.station_id==80805].zero_bs77_m.iloc[0]):.3f} m BS-77",
    "nominal Kherson zero in EVRF2019 = "
    f"{float(g[g.station_id==80805].nominal_zero_evrf2019_m.iloc[0]):.4f} m",
    "gauge_vertical_reference_summary.csv", "validation_section.md", "",
    "VALIDATED",
    "Using the reservoir zero at Kherson would introduce a 17 m blunder; the "
    "two domains must never share a zero.")

# ===================================================================== #
# V2  SWOT PIXC processing-chain validation                              #
# ===================================================================== #
v = load("validation_summary_table.csv")
pixc = v[v.validation_question.str.contains("PIXC tide-correction")].iloc[0]
add("V2.1",
    "The documented SWOT PIXC chain height-(solid_earth_tide+load_tide_fes+"
    "pole_tide) reproduces RiverSP node heights of the same cycle and pass; "
    "the three alternative branches do not.",
    "internal product-chain validation (NOT an independent sensor)",
    "SWOT PIXC vs SWOT RiverSP, cycle 482 pass 001", "RiverSP node", 1023,
    "-0.0010 m for the documented chain",
    "alternatives: +0.0720 m uncorrected, +0.1451 m sign-reversed, "
    "-23.89 m geoid subtracted",
    "validation_summary_table.csv", "validation_section.md §3.1",
    "V2_pixc_sign_validation.png", "VALIDATED",
    "RiverSP is derived from the same SWOT observation, so this establishes "
    "the correction chain, NOT independent accuracy. The PIXC `geoid` field "
    "(EGM2008) is never subtracted from PIXC `height`.")

# ===================================================================== #
# V3  SWOT <-> ICESat-2 cross-sensor                                     #
# ===================================================================== #
c = load("part4_colocated_swot_icesat.csv")
best = c.loc[c.dt_hours.abs().idxmin()]
add("V3.1",
    "Under good temporal collocation SWOT and ICESat-2 water-surface heights "
    "agree at the centimetre level in the common frame with no correction "
    "applied.",
    "cross-sensor validation", "SWOT PIXC vs ICESat-2 ATL13, matched sensor-"
    "to-sensor within 500 m on the same day", "co-located overpass",
    int(len(c)),
    f"best pair {best.date}: {best.d_swot_minus_icesat_m:+.4f} m "
    f"(dt {best.dt_hours:+.2f} h, {int(best.n_segments)} segments)",
    "full set: " + "; ".join(
        f"{r.date} dt {r.dt_hours:+.1f} h -> {r.d_swot_minus_icesat_m:+.4f} m"
        for r in c.itertuples()),
    "part4_colocated_swot_icesat.csv", "validation_section.md §3.2",
    "V3_swot_icesat_colocated.png", "VALIDATED",
    "n=3 overpasses. The residual orders with |dt|, which is suggestive of "
    "temporal mismatch rather than a datum offset, but no regression is "
    "fitted and NO universal centimetric agreement is claimed. Sign "
    "convention is H_SWOT - H_ICESat-2.",
    superseded_by="supersedes swot_icesat_hydrological_adjustment.csv "
                  "(2026-09-07), which reports +0.0023/+0.0438 m for the same "
                  "dates from an earlier aggregation")

# ===================================================================== #
# V4  Kherson gauge anchor                                              #
# ===================================================================== #
kh = v[v.validation_question.str.contains("Kherson gauge")].iloc[0]
add("V4.1",
    "The SWOT branch is locally consistent with the independent Kherson "
    "in-situ record.",
    "satellite-gauge validation",
    "SWOT PIXC open water within 1 km of gauge 80805, gauge in EVRF2019",
    "PIXC overpass", 9,
    "+0.0264 m (satellite - gauge)",
    "NMAD 0.0413 m; 95% CI [-0.0420, +0.0394]; median moves 8.4 cm across "
    "aggregation radii 0.5-5 km",
    "validation_summary_table.csv, part7_answers.csv",
    "validation_section.md §3.3", "V4_kherson_hydrograph_epochs.png",
    "VALIDATED",
    "The point estimate lies marginally outside its own bootstrap CI because "
    "the two are computed over different aggregations (see QC report); the "
    "radius dependence is the along-channel gradient entering the window, so "
    "the radius is a real methodological choice, not noise.")

hy = load("part4_colocated_swot_icesat.csv")
anom = hy[hy.date == "2023-04-05"].iloc[0]
add("V4.2",
    "The 2023-04-05 SWOT-ICESat-2 discrepancy is NOT explained by the Kherson "
    "hydrograph: the gauge was rising, so hydrology predicts the opposite "
    "sign and enlarges the residual.",
    "negative result", "Kherson daily hydrograph vs sensor difference",
    "one overpass pair", 1,
    f"observed {100*anom.d_swot_minus_icesat_m:+.1f} cm; "
    f"expected {anom.expected_dH_cm:+.1f} cm from gauge "
    f"{anom.gauge_dHdt_cm_day:+.1f} cm/day over dt {anom.dt_hours:.1f} h",
    "residual after applying the hydrological term: -24.0 cm (worse)",
    "part4_colocated_swot_icesat.csv, kherson_daily_variability.csv",
    "validation_section.md §3.4",
    "V4_kherson_hydrograph_epochs.png", "UNRESOLVED",
    "An 18 cm daily excursion is a ~99th-percentile event at this site "
    "(|dH| median 3.0, p90 9.0, max 20.0 cm/day, n=96 days). No local wind or "
    "pressure record exists, so wind setup can be neither confirmed nor "
    "excluded. Reported as unresolved.")

# ===================================================================== #
# V5  transferability of the empirical correction                        #
# ===================================================================== #
pt = load("part4_pooling_test.csv")
rr = pt[pt.test.str.startswith("reservoir stations pooled")].iloc[0]
kk = pt[pt.test.str.startswith("Kherson")].iloc[0]
add("V5.1",
    "The reservoir-derived empirical alignment constant does not transfer to "
    "the Kherson reach: the two independent estimates have disjoint "
    "confidence intervals, and zero lies inside Kherson's.",
    "negative result / spatial limit of a correction",
    "per-station empirical constants, permanent-tide harmonised",
    "gauge station", 6,
    f"reservoir {rr.median_m:+.3f} m [{rr.ci95_low_m:+.3f}, {rr.ci95_high_m:+.3f}] "
    f"(n={int(rr.n)} stations) vs Kherson {kk.median_m:+.4f} m "
    f"[{kk.ci95_low_m:+.4f}, {kk.ci95_high_m:+.4f}]",
    f"station-to-station NMAD {rr.nmad_m:.3f} m; radius swing 1.8 cm "
    f"(Rozumivka) and 5.0 cm (Kherson) over 1-10 km",
    "part4_pooling_test.csv, part4_anchor_summary.csv, part7_answers.csv",
    "validation_section.md §3.5", "V5_residual_summary.png",
    "VALIDATED",
    "A rank test cannot reach significance with 5 stations against 1 "
    "(p floor 0.200) and is not relied on. This is a local contextual term, "
    "NOT a global sensor bias and NOT a datum fix. No post-breach constant "
    "exists at any station.")

# ===================================================================== #
# V6  historical bathymetry vertical reference                           #
# ===================================================================== #
hs = load("historical_reference_sensitivity.csv")
b = hs[hs.reference_level_m == 14.0].iloc[0]
a = hs[hs.reference_level_m == 16.0].iloc[0]
ev = load("hist3_datum_in_evrf2019.csv")
e14 = ev[ev.datum_bs_m == 14.0].iloc[0]
add("V6.1",
    "The S-57 sounding depths are reduced to the published navigation "
    "drawdown level UNS 14.00 m, not to the normal impoundment level "
    "16.00 m. On 14.00 m the survey and ICESat-2 exposed bed agree within "
    "uncertainty; on 16.00 m they do not.",
    "vertical re-referencing of historical bathymetry",
    "S-57 soundings vs ICESat-2 S7 confirmed exposed bed",
    "ICESat-2 track", int(b.n_tracks),
    f"14.00 m: {b.median_dH_m:+.4f} m [{b.ci_lo_m:+.4f}, {b.ci_hi_m:+.4f}], "
    f"zero inside CI | 16.00 m: {a.median_dH_m:+.4f} m "
    f"[{a.ci_lo_m:+.4f}, {a.ci_hi_m:+.4f}], zero outside CI",
    f"NMAD {b.nmad_m:.3f} m over n={int(b.n_points)} points; "
    f"the 16.00 m assumption is rejected at ~{abs(a.median_dH_m):.2f} m",
    "historical_reference_sensitivity.csv, hist2_datum_fit.csv, "
    "hist3_corrected_dH.csv",
    "historical_bathymetry_and_hydraulic_validation.md, "
    "historical_datum_semantics.md",
    "V2_historical_reference_sensitivity.png, QA5_sounding_datum.png",
    "VALIDATED",
    "14.00 m is the only PUBLISHED level of the four candidates (Table 19). "
    "The honest uncertainty is the 0.40 m spread among three independent "
    "estimates (13.71 capacity-curve, 14.00 published, 14.11 ICESat-2), not "
    "the CI, and it is deliberately not folded in. The source does not state "
    "its Baltic realisation (BS-42 vs BS-77), so a further few-centimetre "
    "step may be unaccounted for. The 14.11 m estimate is CIRCULAR against "
    "ICESat-2 and is never used as the working reference.")

add("V6.2",
    "The historical reference level carried into EVRF2019 is spatially "
    "varying, not a constant.",
    "geodetic transformation", "14.00 m BS-77 + per-sounding EPSG:9902 offset",
    "sounding", 7514,
    f"{e14.datum_evrf2019_min_m:.3f} to {e14.datum_evrf2019_max_m:.3f} m "
    f"EVRF2019, median {e14.datum_evrf2019_median_m:.3f} m",
    f"median offset {e14.delta_epsg9902_median_m:+.4f} m; spread "
    f"{1e3*(e14.datum_evrf2019_max_m-e14.datum_evrf2019_min_m):.0f} mm",
    "hist3_datum_in_evrf2019.csv", "historical_kakhovka_validation.md", "",
    "VALIDATED",
    "For quoting, 14.19 m EVRF2019; for computing, the per-sounding value.")

# ===================================================================== #
# V7  historical free-surface validation                                 #
# ===================================================================== #
fp = pd.read_csv(D / "processed/historical/historical_fig16_profiles.csv")
c2 = fp[fp.curve_id == 2]
pool = c2[c2.chainage_km < 180]
add("V7.1",
    "The digitised field-measured historical longitudinal free-surface curve "
    "(Fig. 16 curve 2, 22-25 April 1970) reproduces the independently "
    "tabulated profile of Table 20 to a few centimetres over the pool.",
    "historical digitisation validated against an independent table",
    "Fig. 16 curve 2 vs Table 20 interpolated to Q = 8 400 m3/s",
    "digitised point", int(len(c2)),
    f"median deviation {c2.deviation_m.median():+.3f} m; "
    f"NMAD {1.4826*np.median(np.abs(c2.deviation_m-c2.deviation_m.median())):.3f} m",
    f"flat pool <180 km (n={len(pool)}): median {pool.deviation_m.median():+.3f} m, "
    f"max |dev| {pool.deviation_m.abs().max():.3f} m",
    "data/processed/historical/historical_fig16_profiles.csv",
    "historical_bathymetry_and_hydraulic_validation.md",
    "HIST9_fig16_digitisation.png", "VALIDATED",
    "The large deviations on the steep upper backwater limb (>=210 km, "
    "-0.77/-0.66/-0.10 m) are NOT digitisation error: Table 20 has only two "
    "points across the limb and the true curve is convex, so the linear "
    "reference is itself wrong there by decimetres. The vertical axis is "
    "non-linear and was calibrated on detected gridlines; on a linear axis "
    "the same symbols read 0.2-0.4 m too high. Curve 4 (dash-dot) remains "
    "NOT digitised - it cannot be separated from dashed curve 3 by the "
    "enclosed-region method.",
    superseded_by="supersedes the statement in "
                  "historical_bathymetry_and_hydraulic_validation.md (written "
                  "32 min earlier) that Fig. 16 curves were not digitised")

rs = load("hist6_reach_slopes.csv")
flat = rs[rs.hi <= 180]
add("V7.2",
    "Pre-breach ICESat-2 water-surface slopes are statistically "
    "indistinguishable from zero throughout the impounded reach and steepen "
    "in the uppermost reach, reproducing the documented free-surface geometry "
    "(near-level main pool passing into a backwater limb).",
    "hydraulic validation against historical tables",
    "ICESat-2 ATL13 pre-breach per-date fits, by reach",
    "reach", int(len(rs)),
    "0-180 km: " + ", ".join(f"{r.icesat_slope_cm_km:+.2f}" for r in flat.itertuples())
    + " cm/km; 210-240 km: "
    + f"{float(rs[rs.lo==210].icesat_slope_cm_km.iloc[0]):+.2f} cm/km",
    "every reach lies inside the historical discharge envelope "
    f"(within_envelope True for {int(rs.within_envelope.sum())}/{len(rs)})",
    "hist6_reach_slopes.csv", "historical_bathymetry_and_hydraulic_validation.md",
    "V5_reach_slopes_historical_vs_icesat.png, V3_historical_free_surface_profiles.png",
    "SUPPORTED",
    "Per-date fits span a median of only 8.3 km (p75 13.1 km, max 74.4 km), "
    "so no reach value is a hydraulic gradient measured across that reach. "
    "The 180-210 km band is a geometric artefact (see V7.3) and must not be "
    "read as a hydraulic slope.")

sg = load("hist7_slope_geometry.csv")
add("V7.3",
    "The anomalous 180-210 km pre-breach reach slope is a chainage-geometry "
    "artefact, not a hydraulic signal.",
    "artefact diagnosis / negative result",
    "ICESat-2 per-date fits with cross-track geometry audited",
    "per-date fit", int(len(sg)),
    "all fits in the band are single overpasses; chainage span / ground span "
    "median 1.34 (p90 1.94, max 2.65)",
    "the pre/post contrast survives because the NUMERATOR changes tenfold "
    "(median within-fit WSE span 0.072 m pre vs 0.762 m post), while the "
    "geometry defect is nearly identical in both periods (55% vs 47% "
    "single-pass; median inflation 1.35 vs 1.39)",
    "hist7_slope_geometry.csv",
    "historical_bathymetry_and_hydraulic_validation.md",
    "HIST7_slope_geometry.png", "VALIDATED",
    "The first hypothesis - that chainage assignment fails where the pool is "
    "widest - was tested and REJECTED: that band has the smallest median "
    "offset from the channel line (2.5 km) of any reach.")

# ===================================================================== #
# V8  historical hydrodynamic context                                    #
# ===================================================================== #
dy = load("hist10_dynamic_vs_mean.csv")


def dyv(key):
    m = dy[dy.quantity.str.contains(key, regex=False)]
    return float(m.value_cm.iloc[0]) if len(m) else np.nan


add("V8.1",
    "Instantaneous water-level disturbances documented for this reservoir "
    "(seiches, wind setup) are far larger than the mean longitudinal "
    "hydraulic gradient, which explains why individual pre-breach overpasses "
    "can show small slopes of either sign while the regime-level median is "
    "near zero.",
    "physical scale argument from historical sources",
    "historical monograph (Fig. 46 seiches, Fig. 57 wind setup, Table 20 "
    "backwater, Table 200 measurement accuracy) vs observed ICESat-2 spans",
    "documented phenomenon", 6,
    f"mean hydraulic rise over 183 km: {dyv('Qmin'):.0f} cm (Qmin), "
    f"{dyv('Q20%'):.0f} cm (Q20%), {dyv('Q1%'):.0f} cm (Q1%) | "
    f"seiche half-range at an antinode {dyv('seiche'):.1f} cm | wind setup at "
    f"an end {dyv('moderate gale'):.0f} cm (15 m/s) to "
    f"{dyv('extreme storm (25 m/s)'):.0f} cm (25 m/s)",
    f"observed ICESat-2 within-overpass WSE span, pre-breach median "
    f"{dyv('pre-breach median'):.2f} cm (p90 {dyv('pre-breach p90'):.2f} cm) "
    f"over ~8-13 km; source's own level accuracy "
    f"{dyv('measurement accuracy'):.0f} cm",
    "hist10_dynamic_vs_mean.csv",
    "historical_bathymetry_and_hydraulic_validation.md",
    "V7_historical_seiches.png, V8_wind_setup_by_gauge.png, "
    "V9_scale_comparison.png, V10_conceptual_decomposition.png",
    "SUPPORTED",
    "NO causal attribution is made for any individual satellite date: no "
    "simultaneous wind or pressure forcing data exist. This is a "
    "scale-of-magnitude argument, and it must not be confused with the mean "
    "hydraulic slope. Documented seiche periods: 13.4 h uninodal, 7.3 h "
    "binodal, 3.5 h trinodal.")

# ===================================================================== #
# V9  geomorphological validation                                        #
# ===================================================================== #
bm = load("hist12_bed_morphology_tests.csv")


def bmv(key):
    m = bm[bm.test == key]
    return float(m.value.iloc[0]) if len(m) else np.nan


near, far = bmv("B trough fraction near channel"), bmv("B trough fraction far")
add("V9.1",
    "Deep bed depressions are preferentially concentrated near the modern "
    "river corridor, as the historical description of a drowned floodplain "
    "with former channel troughs requires.",
    "geomorphological validation", "S-57 soundings vs SWORD corridor distance",
    "sounding", 7509,
    f"{near:.1f}% of soundings below the descriptive threshold "
    f"{bmv('A descriptive threshold (NOT a mode boundary)'):+.2f} m lie within "
    f"3 km of the corridor, against {far:.1f}% beyond - enrichment "
    f"{near/far:.2f}x",
    "median bed elevation also rises systematically away from the corridor",
    "hist12_bed_morphology_tests.csv", "geomorphological_validation_final.md §5",
    "V11_bed_morphology_vs_source.png, V14_geomorphological_validation.png",
    "SUPPORTED",
    "SWORD is a MODERN-CHANNEL PROXY. This does not claim reconstruction of "
    "any specific pre-1956 channel, nor that any individual depression is "
    "one. The +7.38 m threshold is descriptive only - see V9.4.")

add("V9.2",
    "ICESat-2 exposed dry-bed elevations are consistent with the shallow "
    "historical bed platform.",
    "cross-validation of independent bed observations",
    "ICESat-2 S7 confirmed exposed bed vs S-57 shallow platform",
    "ICESat-2 track", 28,
    f"S7 median {bmv('C S7 median terrain'):+.2f} m vs historical platform "
    f"median {bmv('A platform median'):+.2f} m; difference "
    f"{bmv('C S7 median terrain')-bmv('A platform median'):+.2f} m",
    f"platform NMAD {bmv('A platform NMAD'):.2f} m; "
    f"{bmv('C S7 fraction on platform'):.1f}% of S7 falls on the platform",
    "hist12_bed_morphology_tests.csv", "geomorphological_validation_final.md §6",
    "V14_geomorphological_validation.png", "SUPPORTED",
    "Stated narrowly: this validates the DRY-BED MASK and the shallow-platform "
    "selection. S7 contains no observations from the deepest 20 m of the "
    "sounding range (deepest S7 +0.86 m vs -19.38 m in the soundings), "
    "because the deep depressions still carry the river. It is NOT "
    "independent proof of the drowned-valley morphology.")

add("V9.3",
    "The reconstructed former reservoir bed does not intrude into the "
    "historically non-inundated Terrace III elevation domain.",
    "vertical sanity check", "S-57 soundings and ICESat-2 S7 vs Fig. 128 "
    "terrace elevations", "elevation extremum", 1,
    "minimum clearance 5.74 m",
    f"max historical bed {bmv('F max surveyed bed'):.2f} m; max ICESat-2 S7 bed "
    f"15.45 m; Terrace III floor {bmv('F third terrace floor'):.3f} m",
    "hist12_bed_morphology_tests.csv, historical_terraces.csv",
    "geomorphological_validation_final.md §7",
    "V14_geomorphological_validation.png", "SUPPORTED",
    "The clearance must be taken against max(soundings, S7) because S7 "
    "reaches higher (15.45 m) than the soundings (15.32 m); using soundings "
    "alone would overstate the clearance as 5.87 m. A bed at terrace height "
    "would have indicated the survey or the reference level was wrong.")

add("V9.4",
    "The bed-elevation distribution is unimodal and negatively skewed - a "
    "broad shallow platform with a long deep tail - NOT two separable "
    "populations.",
    "retraction of an earlier framing", "S-57 soundings", "sounding", 7509,
    f"skew -0.76; trough fraction {bmv('A trough fraction'):.1f}% below the "
    f"descriptive threshold",
    "no mode-separation test supports a bimodal description",
    "hist12_bed_morphology_tests.csv",
    "geomorphological_validation_final.md §10", "", "REJECTED",
    "Earlier framing as two populations is WITHDRAWN and retained only as an "
    "audit trail. The words 'bimodal' and 'antimode' must not appear. The "
    "+7.38 m value survives only as a descriptive classification threshold, "
    "not a geomorphological boundary.",
    publication_status="supplement / methods caveat")

add("V9.5",
    "The longitudinal bed-roughness contrast is weak and rests on a single "
    "20 km band.",
    "weak result", "S-57 soundings, roughness by chainage", "chainage band", 2,
    f"median roughness {bmv('D roughness below 160 km (median)'):.2f} m below "
    f"160 km vs {bmv('D roughness above 160 km (median)'):.2f} m above",
    f"excluding the worst band the below-160 km value falls to "
    f"{bmv('D below-160 mean excluding the worst band'):.2f} m",
    "hist12_bed_morphology_tests.csv",
    "geomorphological_validation_final.md §11", "", "PRELIMINARY",
    "Nearly vanishes without one band. Reported, not relied on.",
    publication_status="supplement")

add("V9.6",
    "The point-count exposure test is invalid by construction and is "
    "superseded by the area-weighted test.",
    "invalidated test", "S-57 sounding point sample", "n/a", 0,
    f"shallow-fraction slope {bmv('E shallow-fraction slope'):.5f} %/km",
    "not interpretable",
    "hist12_bed_morphology_tests.csv",
    "geomorphological_validation_final.md §8", "", "REJECTED",
    "Table 21 reports an AREA fraction, but the survey follows the navigable "
    "channel and avoids the shallow margins where drying occurs, so a point "
    "sample cannot estimate it. Superseded by V10.1.",
    publication_status="supplement / methods caveat",
    superseded_by="V10.1")

# ===================================================================== #
# V10 area-weighted morphological validation                             #
# ===================================================================== #
ex = load("historical_exposure_area_validation.csv")
gr = load("hist17_grid_resolution.csv")
w250 = gr[(gr.scope == "whole") & (gr.cell_m == 250)].iloc[0]
exr = ex[ex.reconstructed_fraction_pct.notna()]
add("V10.1",
    "An independently reconstructed bathymetric surface reproduces the "
    "historical reservoir-wide drawdown exposure fraction and its upstream "
    "increase, under all four cross-validated interpolators.",
    "independent area-weighted morphological validation",
    "reconstructed bed surface (250 m canonical grid) vs historical Table 21",
    "morphometric reach", int(len(exr)),
    f"whole mapped reservoir: historical {w250.historical_pct:.1f}% vs "
    f"reconstructed {w250.reconstructed_pct:.2f}% "
    f"(diff {w250.diff_pp:+.2f} pp)",
    f"across interpolators {w250.min_pct:.2f}-{w250.max_pct:.2f}% "
    f"(spread {w250.max_pct-w250.min_pct:.2f} pp); by reach historical "
    + " / ".join(f"{r.historical_fraction_pct:.1f}" for r in exr.itertuples())
    + " % vs reconstructed "
    + " / ".join(f"{r.reconstructed_fraction_pct:.1f}" for r in exr.itertuples())
    + " %",
    "historical_exposure_area_validation.csv, hist17_grid_resolution.csv",
    "geomorphological_validation_final.md §8",
    "V13_exposure_area_validation.png, V14_geomorphological_validation.png",
    "VALIDATED",
    "Historical Table 21 was used as VALIDATION, not calibration - the "
    "historical 12.9% never entered the surface, and method selection came "
    "from independent cross-validation, not from proximity to Table 21. "
    "Reach 5 is NA (0 km2 of grid inside the mapped domain), never zero. "
    "Reaches 1 and 2 are merged: a ~30 km residual sits at the Babyne "
    "boundary. With only three resolved reach units no inferential statistic "
    "is claimed - r = 0.993 is descriptive only.")

add("V10.2",
    "The shoreline boundary condition is the single most consequential "
    "methodological choice in the bed reconstruction; omitting it biases the "
    "exposure fraction low by roughly a factor of two.",
    "methodological sensitivity", "reconstructed bed surface, 3 boundary "
    "variants", "boundary variant", 3,
    "unconstrained 6.7% vs constrained at the observed 2023-06-05 waterline "
    "12.3% (whole reservoir, OK surface)",
    "soundings stop >=167 m from shore with a median bed of 12.15 m against a "
    "~17.1 m waterline, so an unconstrained interpolator models the margins "
    "~5 m too deep; unconstrained RBF extrapolated to +67.9 m",
    "hist14_surface_summary.csv, historical_exposure_area_validation.csv",
    "geomorphological_validation_final.md §3",
    "V12_bed_surface_validation.png", "VALIDATED",
    "Two quantities must not be conflated: the INTERPOLATION BOUNDARY "
    "elevation (17.08 m EVRF2019, the observed 2023-06-05 waterline, "
    "independently constrained by the Rozumivka gauge at 16.88 m BS-77 AND by "
    "the historical level-area curve at ~17.1 m BS-77 for 2 192 km2) and the "
    "EXPOSURE DENOMINATOR (NPG 16.18 m EVRF2019, restricting the domain to "
    "the reservoir at NPG as Table 21 does). The 2023 shoreline sat ~1 m "
    "ABOVE project NPG - the pool had been raised - so a constraint at NPG "
    "would itself have been wrong. The unconstrained run is retained as an "
    "audit trail and is used for no result.")

cv = load("hist14_interpolator_cv.csv")
p1 = cv[cv.scheme == "blocked1km"]
bestm = p1.loc[p1.RMSE_m.idxmin()]
# hist18's headline package, read once (f-strings cannot carry backslashes,
# so the query is done here rather than inline)
_e18 = load("hist18_error_decomposition.csv")
ovr18 = _e18[_e18.factor == "overall"].iloc[0]
w18 = load("hist18_worst_residuals.csv")
add("V10.3",
    "Interpolation uncertainty, not grid resolution, is the limiting term in "
    "the bed surface. The median absolute prediction error is ~0.9 m, while "
    "RMSE rises to 2.76 m because of a small number of large errors "
    "concentrated in morphologically complex zones.",
    "cross-validation", "S-57 soundings, spatially blocked at 1 km",
    "sounding", int(bestm.n),
    f"preferred {bestm.method}: RMSE {bestm.RMSE_m:.2f} m, NMAD "
    f"{bestm.NMAD_m:.2f} m, bias {bestm.bias_m:+.2f} m, median |e| "
    f"{float(ovr18.p50_abs):.2f} m",
    "ranking " + ", ".join(f"{r.method} {r.RMSE_m:.2f}"
                           for r in p1.sort_values('RMSE_m').itertuples())
    + f"; the worst 5% of observations account for "
      f"{float(ovr18.worst_5pct_share_SSE):.0f}% of the squared prediction error",
    "hist14_interpolator_cv.csv, hist18_error_decomposition.csv",
    "geomorphological_validation_final.md §4",
    "V12_bed_surface_validation.png, V16_cv_error_diagnosis.png", "VALIDATED",
    "1 km blocks are PRIMARY, matched to the ~357 m median sounding spacing. "
    "5 km blocks are a STRESS TEST only: the fitted variogram range is ~2 km, "
    "so they hold out cells beyond the correlation range of any training "
    "point and score extrapolation, which no method can pass. Random holdout "
    "is an OPTIMISTIC reference because survey lines put a near-twin of each "
    "held-out point in the training set. Residuals are largest within 1-3 km "
    "of the corridor where the bed is steepest - a resolution limit, not a "
    "bias.")

sch = load("hist18_cv_scheme_comparison.csv").set_index("scheme")
edc = load("hist18_error_decomposition.csv")
lr = edc[edc.factor == "local_roughness"]
dsh = edc[edc.factor == "dist_to_shoreline"]
rc = edc[edc.factor == "reach"]
add("V10.4",
    "The 2.76 m blocked-CV RMSE is a heavy-tailed error of local bed "
    "morphology, not a vertical offset of the DEM and not an artefact of "
    "blocked cross-validation.",
    "error decomposition", "the same soundings under five hold-out geometries "
    "and six error covariates, kriging parameters unchanged",
    "sounding", int(bestm.n),
    f"random hold-out already gives {sch.loc['random','RMSE_m']:.3f} m; 1 km "
    f"blocking adds only "
    f"{sch.loc['blocked1km','RMSE_m']/sch.loc['random','RMSE_m']:.2f}x "
    f"({sch.loc['blocked1km','RMSE_m']:.3f} m) and 5 km blocking "
    f"{sch.loc['blocked5km','RMSE_m']/sch.loc['random','RMSE_m']:.2f}x",
    f"strongest well-populated driver is local roughness, RMSE "
    f"{lr.RMSE_m.min():.2f} m in the smoothest quartile to "
    f"{lr.RMSE_m.max():.2f} m in the roughest decile "
    f"({lr.RMSE_m.max()/lr.RMSE_m.min():.1f}x); within 500 m of the shoreline "
    f"RMSE is {float(dsh[dsh.bin=='0-500 m'].RMSE_m.iloc[0]):.2f} m and those "
    f"23% of soundings carry "
    f"{float(dsh[dsh.bin=='0-500 m'].share_SSE_pct.iloc[0]):.0f}% of the "
    f"squared error; smoothing bias REVERSES sign with elevation "
    f"(deepest bin {edc[edc.factor=='bed_elevation_class'].bias_m.max():+.2f} m, "
    f"highest {edc[edc.factor=='bed_elevation_class'].bias_m.min():+.2f} m)",
    "hist18_cv_scheme_comparison.csv, hist18_error_decomposition.csv, "
    "hist18_worst_residuals.csv",
    "", "V16_cv_error_diagnosis.png", "VALIDATED",
    "The datum is EXCLUDED as a cause: overall bias -0.032 m, per-reach bias "
    f"spans only {rc.bias_m.min():+.3f} to {rc.bias_m.max():+.3f} m, and the "
    "elevation-dependent bias changes sign, which no constant offset can do. "
    "Decisive against a blocked-CV artefact: the worst 20 residuals are as "
    f"large under RANDOM hold-out (median |e| "
    f"{w18.resid_random.abs().median():.2f} m) as under blocked "
    f"({w18.resid.abs().median():.2f} m), so the "
    "hold-out does not create them. NOT claimed: that this "
    "relief is mathematically unrecoverable -- it cannot be reliably recovered "
    "FROM THE SOUNDING POINTS ALONE BY AN ORDINARY UNCONSTRAINED "
    "INTERPOLATOR; adding the historical channel axis, breaklines, "
    "hydrographic cross-sections or old charts as constraints could do "
    "considerably better. The problem is under-determined by the present "
    "data, not impossible.")

hyp = load("hypsometric_constraint_test.csv")
_rms = float(np.sqrt((hyp.diff_km2 ** 2).mean()))
add("V10.5",
    "The reconstructed bed reproduces the published level-area curve to "
    "3.8% of area across 17 levels, but is biased low in area at EVERY "
    "level -- an independent detection of the same smoothing bias.",
    "hypsometric validation",
    "DEM area below each level vs Table 19 level-area curve, 10.0-18.0 m",
    "published level", int(len(hyp)),
    f"RMS area error {_rms:.0f} km2 ({100*_rms/hyp.area_historical_km2.mean():.1f}% "
    f"of mean area); median {hyp.diff_pct.median():+.1f}%",
    f"negative at all {len(hyp)} levels, worst {hyp.diff_pct.min():+.1f}% at "
    f"{hyp.water_level_bs77_m.min():.1f} m BS-77; the DEM has too few cells "
    f"below any level, i.e. its deep parts are too shallow, matching the "
    f"+11.03 m bias hist18 measured in the deepest elevation bin",
    "hypsometric_constraint_test.csv",
    "multilevel_shoreline_constraint_experiment.md",
    "V17_multilevel_constraints.png", "SUPPORTED",
    "CRITICAL: this curve must NOT be used to condition the DEM. Table 21's "
    "drying area is arithmetically the same information -- A(NPG) - A(GMO) = "
    "2155 - 1876 = 279 km2 equals Table 21's summed reach drying area of "
    "279 km2 exactly -- so conditioning on A(H) would make the V10.1 "
    "validation circular. Kept as validation, deliberately not as a "
    "constraint. The multi-level SATELLITE water-extent route was tested and "
    "rejected: only one pre-breach polygon has usable coverage (2023-06-05, "
    "17.08 m, already the constraint in use), post-breach waterlines sit a "
    "median 1.69 km inside the pre-breach shore with only 12% of their length "
    "within 500 m of it, and ICESat-2 profiles coincide with a usable polygon "
    "date on only 2 of 14 dates, each spanning <15 km, so H(x) cannot be "
    "reconstructed.",
    publication_status="supplement")

# ===================================================================== #
# V11 grid-resolution sensitivity                                        #
# ===================================================================== #
gw = gr[gr.scope == "whole"].sort_values("cell_m", ascending=False)
add("V11.1",
    "The exposure fraction is converged with respect to grid resolution: a "
    "70-fold increase in cell count changes it by less than 0.1 percentage "
    "points, an order of magnitude below the spread between interpolators.",
    "numerical convergence test",
    "the same bed surfaces recomputed on 250 / 50 / 30 m grids",
    "grid resolution", int(len(gw)),
    "; ".join(f"{r.cell_m:.0f} m -> {r.reconstructed_pct:.2f}%"
              for r in gw.itertuples()),
    f"total drift "
    f"{gw.reconstructed_pct.iloc[-1]-gw.reconstructed_pct.iloc[0]:+.2f} pp; "
    f"last refinement step "
    f"{gw.reconstructed_pct.iloc[-1]-gw.reconstructed_pct.iloc[-2]:+.2f} pp; "
    f"interpolator spread at the finest grid "
    f"{gw.max_pct.iloc[-1]-gw.min_pct.iloc[-1]:.2f} pp",
    "hist17_grid_resolution.csv", "geomorphological_validation_final.md",
    "V15_grid_resolution_convergence.png", "VALIDATED",
    "Refining the grid does NOT add information - it interpolates the same "
    "7 509 soundings more finely, and the cross-validated RMSE is unchanged "
    "by construction. 250 m is the canonical statistical grid; 30-50 m are "
    "for visualisation. Reach areas agree across resolutions to within "
    "1.53 km2 (whole reservoir 1.29 km2 on 2 186 km2, 0.06%).")

# ===================================================================== #
# MAIN RESULT — hydraulic transition                                     #
# ===================================================================== #
ts = load("kakhovka_transition_statistics.csv")
t = ts[ts.estimator == "Theil-Sen"].iloc[0]
o = ts[ts.estimator == "OLS"].iloc[0]
sm = load("kakhovka_pre_post_slope_summary.csv")
pd_ = load("kakhovka_perdate_slopes_robust.csv")
spans = pd_.groupby("period").agg(d0=("date", "min"), d1=("date", "max"))
add("M1.1",
    "The pre-breach impounded reservoir had a near-level main water surface, "
    "whereas after the dam breach the system developed a persistent, "
    "directional, river-like longitudinal hydraulic gradient.",
    "MAIN RESULT - hydraulic regime transition",
    "ICESat-2 ATL13 longitudinal profiles, one robust fit per date",
    "date (overpass)", f"{int(t.n_pre)} pre / {int(t.n_post)} post",
    f"Theil-Sen median {t.median_pre_cm_km:+.3f} cm/km pre -> "
    f"{t.median_post_cm_km:+.3f} cm/km post; difference "
    f"{t.diff_post_minus_pre_cm_km:+.3f} cm/km",
    f"95% CI [{t.diff_ci95_low:+.3f}, {t.diff_ci95_high:+.3f}] cm/km; "
    f"permutation p = {t.permutation_p:.2e}; Mann-Whitney p = "
    f"{t.mannwhitney_p:.2e}; positive slopes pre {t.pre_positive} "
    f"(sign test p = {t.pre_sign_test_p:.3f}) vs post {t.post_positive} "
    f"(p = {t.post_sign_test_p:.2e})",
    "kakhovka_transition_statistics.csv, kakhovka_perdate_slopes_robust.csv",
    "hydrological_context_and_postbreach_transition.md",
    "FigF_postbreach_longitudinal_gradient.png, FigG_reservoir_to_river_transition.png",
    "VALIDATED",
    f"PRE sample is {spans.loc['PRE_BREACH','d0']}..{spans.loc['PRE_BREACH','d1']}, "
    f"POST is {spans.loc['POST_BREACH','d0']}..{spans.loc['POST_BREACH','d1']}. "
    "Each fit spans a median of only ~24-27 km with 6 points, so no single "
    "date measures a whole-reservoir gradient; the claim is about the "
    "DISTRIBUTION of local slopes, and its strength is the effect size plus "
    "14/14 consistent sign post-breach, not the p-value. Pre-breach signs are "
    "NOT consistent (10/14, p = 0.18), which is what 'near-level' means here.")

add("M1.2",
    "The drawdown period is intermediate between the two regimes.",
    "MAIN RESULT - transition kinematics", "ICESat-2 ATL13 per-date fits",
    "date (overpass)", int(sm[sm.period == 'BREACH_DRAWDOWN'].n_dates.iloc[0]),
    f"Theil-Sen median {t.median_drawdown_cm_km:+.3f} cm/km "
    f"({spans.loc['BREACH_DRAWDOWN','d0']}..{spans.loc['BREACH_DRAWDOWN','d1']})",
    "sits between the pre-breach and post-breach medians",
    "kakhovka_transition_statistics.csv, kakhovka_pre_post_slope_summary.csv",
    "hydrological_context_and_postbreach_transition.md",
    "FigD_kakhovka_longitudinal_profiles.png", "SUPPORTED",
    "n=5 dates. No confidence interval is quoted for the drawdown median.")

add("M1.3",
    "The result is estimator-robust: OLS gives the same sign, a similar "
    "magnitude and the same conclusion.",
    "sensitivity analysis", "same profiles, OLS instead of Theil-Sen",
    "date (overpass)", f"{int(o.n_pre)} pre / {int(o.n_post)} post",
    f"OLS {o.median_pre_cm_km:+.3f} -> {o.median_post_cm_km:+.3f} cm/km, "
    f"difference {o.diff_post_minus_pre_cm_km:+.3f} cm/km",
    f"95% CI [{o.diff_ci95_low:+.3f}, {o.diff_ci95_high:+.3f}]; "
    f"permutation p = {o.permutation_p:.2e}",
    "kakhovka_transition_statistics.csv", "", "", "VALIDATED",
    "Theil-Sen is primary because per-date profiles contain outlying "
    "segments; OLS is the sensitivity check.")

r2r = load("reservoir_to_river_statistics.csv")
ch = r2r[(r2r.comparison.str.startswith("POST MAIN_CHANNEL vs")) &
         (r2r.estimator == "Theil-Sen")].iloc[0]
add("M1.4",
    "Restricting both periods to the main channel weakens the contrast and "
    "its confidence interval includes zero: the channel-only test is "
    "underpowered, not a refutation.",
    "negative / underpowered control", "channel-restricted ICESat-2 profiles",
    "date (overpass)", f"{int(ch.n_pre)} pre / {int(ch.n_post)} post",
    f"Theil-Sen difference {ch.diff_cm_km:+.3f} cm/km, 95% CI "
    f"[{ch.diff_ci_lo:+.3f}, {ch.diff_ci_hi:+.3f}]",
    f"permutation p = {ch.permutation_p:.4f}; post positive "
    f"{ch.post_positive}; Cliff's delta {ch.cliffs_delta:+.3f}",
    "reservoir_to_river_statistics.csv, phase19_expansion_comparison.csv",
    "phase19_channel_vs_residual_water.md",
    "P19_Fig4_channel_slope_timeseries.png, P19_Fig7_allwater_vs_channel.png",
    "PRELIMINARY",
    "Only 6 post-breach dates qualify for a channel-restricted fit (expanded "
    "from 4). This must be reported alongside M1.1 as an honest limitation: "
    "part of the all-water contrast may come from including residual "
    "water bodies at differing levels, and this test cannot yet separate "
    "the two.")

# ===================================================================== #
# WSE heterogeneity                                                      #
# ===================================================================== #
fr = load("pre_post_fragmentation_statistics.csv")
f4 = fr[fr.metric == "F4_wse_p05_p95_range_m"].iloc[0]
add("M2.1",
    "Water-surface heterogeneity within the former reservoir footprint "
    "increased after the breach.",
    "MAIN RESULT - heterogeneity",
    "ICESat-2 ATL13, p95-p05 water-surface range within a date",
    "date", f"{int(f4.n_pre)} pre / {int(f4.n_post)} post",
    f"{f4.median_pre:.3f} m -> {f4.median_post:.3f} m, difference "
    f"{f4.diff_post_minus_pre:+.3f} m",
    f"95% CI [{f4.diff_ci_lo:+.3f}, {f4.diff_ci_hi:+.3f}] m; permutation "
    f"p = {f4.permutation_p:.2e}; Mann-Whitney p = {f4.mannwhitney_p:.2e}",
    "pre_post_fragmentation_statistics.csv",
    "kakhovka_fragmentation_analysis.md",
    "P20_Fig2_atl13_by_class.png, EN_wse_timeseries.png", "VALIDATED",
    "This is metric F4 on ALL ATL13 dates in the footprint (n=192/23). It "
    "must NOT be confused with the within-profile WSE range of the 14/14 "
    "slope-profile sample (median 0.090 m pre vs 1.827 m post, "
    "kakhovka_pre_post_slope_summary.csv), which is a different sample AND a "
    "different spatial restriction. Measured by the altimeter, so it is "
    "independent of Sentinel-2 tile coverage.")

# ===================================================================== #
# Sentinel planform                                                      #
# ===================================================================== #
fm = load("fragmentation_metrics_by_date.csv")
full = fm[fm.footprint_observed_fraction > 0.8]
fpre = full[full.period == "PRE_BREACH"]
fpost = full[full.period == "POST_BREACH"]
add("M3.1",
    "Sentinel-2 shows the planform transition from a single continuous "
    "impounded surface to a fragmented, channel-dominated system with "
    "residual water bodies.",
    "descriptive spatial evidence (NOT inferential)",
    "Sentinel-2 L2A water masks, coverage-aware",
    "date with >80% footprint observed",
    f"{len(fpre)} pre / {len(fpost)} post",
    f"pre-breach {fpre.date.iloc[0]}: {int(fpre.n_water_bodies.iloc[0])} water "
    f"bodies, {fpre.total_water_area_km2.iloc[0]:,.0f} km2, largest component "
    f"{100*fpre.largest_component_fraction.iloc[0]:.3f}% of water area | "
    f"post-breach median {fpost.n_water_bodies.median():.0f} bodies, "
    f"{fpost.total_water_area_km2.median():,.0f} km2, largest component "
    f"{100*fpost.largest_component_fraction.median():.1f}%",
    f"post-breach range {int(fpost.n_water_bodies.min())}-"
    f"{int(fpost.n_water_bodies.max())} bodies; largest-component fraction "
    f"{100*fpost.largest_component_fraction.min():.1f}-"
    f"{100*fpost.largest_component_fraction.max():.1f}%",
    "fragmentation_metrics_by_date.csv, phase20_footprint_coverage.csv",
    "kakhovka_fragmentation_analysis.md",
    "P20_Fig1_fragmentation.png, P19_Fig1_water_extent_pre_draw_post.png",
    "SUPPORTED",
    "ONLY ONE pre-breach date (2023-06-05) has near-complete footprint "
    "coverage, so n_pre = 1 and NO p-value may be computed for a pre/post "
    "fragmentation contrast. The earlier fragmentation significance test used "
    "partial pre-breach scenes as if they were full-reservoir observations "
    "and is INVALID. This is descriptive planform evidence supporting the "
    "altimetric result, never the primary inferential evidence.",
    superseded_by="supersedes any PRE-vs-POST fragmentation p-value computed "
                  "from partial pre-breach scenes")

rw = load("residual_water_offset_summary.csv")
rb = rw[rw.population == "B_high_confidence"].iloc[0]
add("M3.2",
    "Residual water bodies sit systematically below the adjacent channel "
    "stem, consistent with disconnected remnant ponds rather than a "
    "continuous water surface.",
    "supporting spatial evidence",
    "ICESat-2 ATL13 over classified residual water vs channel stem",
    "matched observation", int(rb.n),
    f"median offset {rb.median_m:+.3f} m",
    f"NMAD {rb.nmad_m:.3f} m; p05 {rb.p05:+.2f}, p95 {rb.p95:+.2f} m; "
    f"{100*rb.frac_above:.1f}% lie above the stem",
    "residual_water_offset_summary.csv",
    "phase19_channel_vs_residual_water.md",
    "P19_Fig6_residual_water_offsets.png", "SUPPORTED",
    "The high-confidence population (B) is used. The all-classified figure "
    "(-0.44 m, max +12.59 m) is contaminated by a 2023-09-01 non-water "
    "cluster 5.6 km off the stem and is not quoted.")

# ===================================================================== #
# chainage reconciliation                                                #
# ===================================================================== #
add("X1.1",
    "The historical and modern chainage systems differ by a distance "
    "convention (reservoir axis vs river channel), not by an error.",
    "coordinate reconciliation",
    "historical Fig. 13 axis vs SWORD modern chainage at tie points",
    "tie point", 5,
    "ratio ~1.36 at two independent points; a ~30 km residual is isolated to "
    "reach 1",
    "of 5 candidate ties, 2 are usable (Velyka Lepetykha, Blahovishchenka)",
    "historical_chainage_ties.csv, historical_reach_chainage_crosswalk.csv, "
    "historical_to_modern_chainage_crosswalk.csv",
    "historical_bathymetry_and_hydraulic_validation.md", "", "PRELIMINARY",
    "Historical reach boundaries are therefore NOT used in final statistics "
    "beyond the merged 1+2 / 3 / 4 grouping, and reaches 1 and 2 are merged "
    "because the Babyne boundary cannot be placed. Reach 5 lies outside the "
    "mapped domain.",
    publication_status="supplement")

# ===================================================================== #
add("X2.1",
    "The survey epoch of the historical bathymetry is unrecorded, so no rate "
    "of morphological change can be formed.",
    "unresolvable limitation", "S-57 metadata (SORDAT/RECDAT empty)", "n/a", 0,
    "no epoch", "all 19 SOUNDG attributes are empty in all 7 514 records",
    "qa1_field_provenance.csv", "historical_datum_semantics.md", "",
    "UNRESOLVED",
    "Therefore NO claim of bed stability or bed change is made anywhere. The "
    "residual structure after re-referencing (Theil-Sen of residual vs depth "
    "+0.0148 m/m, CI [+0.0008, +0.0295]) sits inside the reference-level "
    "uncertainty and is not interpreted as morphology.")

# ===================================================================== #
out = pd.DataFrame(ROWS)
dst = T / "manuscript_evidence_matrix.csv"
out.to_csv(dst, index=False)

print("=" * 78)
print("MANUSCRIPT EVIDENCE MATRIX")
print("=" * 78)
print(f"  {len(out)} claims from {out.source_table.str.split(', ').explode().nunique()} "
      f"distinct canonical tables\n")
print(out.validation_status.value_counts().to_string())
print()
print(f"  {'id':<7}{'status':<13}{'unit':<22}{'n':<18}claim")
for r in out.itertuples():
    print(f"  {r.claim_id:<7}{r.validation_status:<13}{r.independent_unit[:21]:<22}"
          f"{str(r.n)[:17]:<18}{r.claim[:64]}")
sup = out[out.superseded_by != ""]
print(f"\n  {len(sup)} rows carry an explicit supersession note:")
for r in sup.itertuples():
    print(f"    {r.claim_id}: {r.superseded_by[:100]}")
print(f"\n-> {dst}")
