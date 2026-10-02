#!/usr/bin/env python
"""MS7 — the validation architecture of paper 1, rebuilt from reproducible data.

v5 compressed its vertical validation into one block whose headline numbers
(32 / 21 crossings) could not be reproduced by any rule set (ms6c). This script
rebuilds it as four separate paths, each with its own question, its own
independent unit and a row-level evidence table, so that every number the
manuscript states can be traced to literal rows:

  V1  GAUGE <-> ATL13, reservoir R, ABSOLUTE CLOSURE
      "does the height sit right?"  c = H_gauge(EVRF2019) - H_ATL13(EGG2015,
      mean-tide crust), per beam transect at each of the six reservoir gauges
      at the station's reported radius (the companion repository's matchups),
      pre-breach. Expected from v5: 53 matchups / 30 dates, c = -0.135 m.
  V2  GAUGE <-> ATL13, reservoir R, TEMPORAL CO-VARIABILITY
      "does ATL13 follow the level through time?" Within-station anomalies of
      QC-passed ATL13 overpass levels against the same-date gauge, nearest
      reservoir gauge <= 20 km from the overpass centroid, pre-breach; and the
      changes between successive visits. v5 states 107 overpasses, r = 0.96,
      rho = 0.94, Theil-Sen 1.01; its exact selection is not recoverable, so the
      rule is stated here and both are reported. Kept apart from V1: a closure
      and a co-variability test answer different questions.
  V3  ROZUMIVKA — CROSS-EPOCH, CROSS-SENSOR TRANSFER THROUGH ONE GAUGE
      Two closures computed INDEPENDENTLY against the same gauge:
          c_IS2  = H_gauge - H_ICESat-2   before the breach (from V1)
          c_SWOT = H_gauge - H_SWOT       after the breach (RiverSP nodes)
      and only then compared, c_SWOT - c_IS2. Nothing corrects SWOT by ATL13
      before the comparison. The frozen-transfer sensitivity applies the
      pre-breach c_IS2 to post-breach SWOT without recalibration and reports
      the residual left against the gauge. Independent unit: the SWOT pass.
      Support radius 1 / 3 (primary) / 5 km; raw median of the nodes, and a
      slope-corrected level at the gauge (Theil-Sen of node WSE on SWORD
      distance within the pass), because after the breach this is a sloping
      river and distance from the post alone moves the level.
  V4  SWOT <-> ATL13 DIRECT CROSSINGS over R/F/D/E
      A cross-sensor spatial consistency check of the whole system, NOT the
      reservoir's validation (R holds one crossing). RiverSP only; LakeSP is a
      separate product-specific test. Primary agreement metrics: median
      SWOT - ATL13, NMAD, MAE, RMSE, share within +-10 cm. Supplementary:
      Pearson r, Spearman rho, Theil-Sen, Deming and Passing-Bablok on the
      absolute heights, and the same correlations on SPATIAL ANOMALIES (height
      minus the zone median of the pair means, zones with n >= 3), because a
      raw r over a system spanning ~15 m of relief measures that both sensors
      see the Dnipro fall, not that they agree. Plus a regression controlling
      for zone and SWORD distance. Zone-specific correlation only for n >= 10;
      smaller zones are listed as descriptive pairs.

Every row of every path is written to one evidence table with a claim_id; the
summary carries the claim_ids V1_ATL13_GAUGE_CLOSURE, V2_ATL13_GAUGE_COVARIABILITY,
V3_ROZUMIVKA_TRANSFER, V4_SWOT_ICESAT_DIRECT.

Outputs (outputs/paper/validation/)
-------
ms7_evidence.csv            one row per matchup / overpass / pass / crossing
ms7_summary.csv             one row per claim and statistic, with the v5 value
ms7_station_inventory.csv   gauge record extents (for Section 2.2)
../figures/FS_V4_swot_icesat_agreement.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import stats
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro.vertical import haversine_km, read_pixc, sample_grid_bilinear
import ms6b_swot_icesat_crossings as XS
import ms6c_crossing_provenance as XC

OUT = ROOT / "outputs/paper/validation"
FIG = ROOT / "outputs/paper/figures"
IS2_TAB = CFG.ICESAT_ROOT / "outputs/tables"
GAUGES = ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet"
BREACH = pd.Timestamp(CFG.BREACH_DATE)
ROZ = 80959
RNG = np.random.default_rng(CFG.SEED)
N_BOOT = 10_000
V2_MAX_KM = 20.0
V3_RADII_KM = (1.0, 3.0, 5.0)
V3_PRIMARY_KM = 3.0
V4_MIN_N_ZONE_R = 10
V4_MIN_N_ANOM = 3

V5 = {  # what v5 printed, for the side-by-side column only
    ("V1_ATL13_GAUGE_CLOSURE", "n_matchups"): 53,
    ("V1_ATL13_GAUGE_CLOSURE", "n_dates"): 30,
    ("V1_ATL13_GAUGE_CLOSURE", "mean_c_m"): -0.135,
    ("V1_ATL13_GAUGE_CLOSURE", "median_c_m"): -0.132,
    ("V1_ATL13_GAUGE_CLOSURE", "sd_between_station_m"): 0.036,
    ("V1_ATL13_GAUGE_CLOSURE", "range_between_station_m"): 0.087,
    ("V2_ATL13_GAUGE_COVARIABILITY", "n_overpasses"): 107,
    ("V2_ATL13_GAUGE_COVARIABILITY", "pearson_r"): 0.96,
    ("V2_ATL13_GAUGE_COVARIABILITY", "spearman_rho"): 0.94,
    ("V2_ATL13_GAUGE_COVARIABILITY", "theil_sen"): 1.01,
    ("V2_ATL13_GAUGE_COVARIABILITY", "diff_n"): 55,
    ("V2_ATL13_GAUGE_COVARIABILITY", "diff_pearson_r"): 0.92,
    ("V2_ATL13_GAUGE_COVARIABILITY", "diff_theil_sen"): 1.05,
    ("V3_ROZUMIVKA_TRANSFER", "is2_n"): 9,
    ("V3_ROZUMIVKA_TRANSFER", "is2_median_c_m"): -0.107,
    ("V3_ROZUMIVKA_TRANSFER", "swot_n_passes"): 68,
    ("V3_ROZUMIVKA_TRANSFER", "swot_median_c_m"): -0.154,
    ("V3_ROZUMIVKA_TRANSFER", "swot_nmad_m"): 0.168,
    ("V4_SWOT_ICESAT_DIRECT", "n"): 32,
    ("V4_SWOT_ICESAT_DIRECT", "median_d_m"): 0.021,
    ("V4_SWOT_ICESAT_DIRECT", "nmad_m"): 0.081,
}
V5_KHERSON = {  # v5 Table S2 / Figure 5: (median c, n)
    "ICESat-2 pre_breach": (-0.031, 23), "ICESat-2 post_breach": (0.002, 10),
    "SWOT RiverSP pre_breach": (0.009, 50), "SWOT RiverSP post_breach": (-0.025, 37),
    "SWOT PIXC pre_breach": (-0.026, 9)}


# ----------------------------------------------------------------- helpers --
def nmad(x):
    return XS.nmad(x)


def _rng(*arrays):
    """A generator seeded by the data itself (SEED + CRC32 of the sample), so an
    interval does not depend on how many other bootstraps ran before it."""
    import zlib
    h = 0
    for a in arrays:
        h = zlib.crc32(np.ascontiguousarray(np.asarray(a, float)).tobytes(), h)
    return np.random.default_rng([CFG.SEED, h])


def boot_median(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    m = np.median(x[_rng(x).integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def boot_r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5:
        return (np.nan, np.nan)
    idx = _rng(a, b).integers(0, len(a), (N_BOOT, len(a)))
    r = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in idx])
    return tuple(np.nanpercentile(r, [2.5, 97.5]))


def deming(x, y, lam=1.0):
    """Deming regression, error-variance ratio lam (1 = orthogonal)."""
    mx, my = x.mean(), y.mean()
    sxx, syy, sxy = np.var(x, ddof=1), np.var(y, ddof=1), np.cov(x, y)[0, 1]
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return b, my - b * mx


def passing_bablok(x, y):
    """Passing-Bablok slope and intercept (shifted median of pairwise slopes)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i, j = np.triu_indices(len(x), 1)
    dx, dy = x[j] - x[i], y[j] - y[i]
    ok = dx != 0
    s = dy[ok] / dx[ok]
    s = s[s != -1]
    k = int(np.sum(s < -1))
    s = np.sort(s)
    n = len(s)
    b = (s[(n - 1) // 2 + k] if n % 2 else 0.5 * (s[n // 2 - 1 + k] + s[n // 2 + k]))
    return b, np.median(y - b * x)


def rows_stats(claim, d, unit, **extra) -> list[dict]:
    """The agreement block of one residual sample."""
    d = np.asarray(d, float)
    lo, hi = boot_median(d)
    base = dict(claim_id=claim, unit=unit, **extra)
    out = [dict(base, statistic="n", value=len(d)),
           dict(base, statistic="median_m", value=np.median(d), ci_lo=lo, ci_hi=hi),
           dict(base, statistic="nmad_m", value=nmad(d)),
           dict(base, statistic="mae_m", value=np.mean(np.abs(d))),
           dict(base, statistic="rmse_m", value=np.sqrt(np.mean(d ** 2))),
           dict(base, statistic="within_10cm_pct", value=100 * np.mean(np.abs(d) <= 0.10))]
    return out


def gauge_levels() -> pd.DataFrame:
    g = pd.read_parquet(GAUGES)
    g["date"] = pd.to_datetime(g.date).dt.normalize()
    return g


# ---------------------------------------------------------------------- V1 --
def v1(evidence: list, summary: list) -> pd.DataFrame:
    C = "V1_ATL13_GAUGE_CLOSURE"
    m = pd.read_csv(IS2_TAB / "gauge_icesat_egg2015_matchups.csv")
    st = pd.read_csv(IS2_TAB / "egg2015_to_evrf2019_by_station.csv")
    rows = []
    for _, s in st.iterrows():
        x = m[(m.station_id == s.station_id) & (m.radius_km == s.reported_radius_km)].copy()
        # the companion chain is tide-free; production ATL13 adds free2mean, so
        # c = gauge - satellite moves by -free2mean (station latitude)
        x["c_m"] = x.c_station_A_m - CFG.free2mean(x.station_lat)
        assert np.isclose(x.c_station_A_m.median(), s.c_station_m, atol=1e-9), s.name_en
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)
    x = x[pd.to_datetime(x.date) < BREACH]
    for _, r in x.iterrows():
        evidence.append(dict(claim_id=C, validation_path="V1", station=r.name_en,
                             sensor="ICESat-2 ATL13", period="PRE_BREACH",
                             overpass_id=f"{r.rgt}_{r.date}", beam=r.beam, date=r.date,
                             distance_m=1000 * r.mean_distance_km, n_nodes=r.n_segments,
                             gauge_level_m=r.h_gauge_evrf2019_A_m,
                             satellite_level_m=r.h_icesat_egg2015_m + CFG.free2mean(r.station_lat),
                             closure_residual_m=r.c_m, radius_km=r.radius_km,
                             included_primary=True, exclusion_reason=""))
    per = x.groupby("name_en").c_m.median()
    lo_hi = {k: boot_median(g.c_m) for k, g in x.groupby("name_en")}
    for k, v in per.items():
        summary.append(dict(claim_id=C, statistic="station_median_c_m", station=k, value=v,
                            ci_lo=lo_hi[k][0], ci_hi=lo_hi[k][1],
                            unit="beam transect", n=int((x.name_en == k).sum())))
    for k, g in x.groupby("name_en"):
        summary.append(dict(claim_id=C, statistic="station_pearson_r", station=k, n=len(g),
                            value=stats.pearsonr(g.h_gauge_evrf2019_A_m, g.h_icesat_egg2015_m)[0],
                            unit="beam transect", note="V1 closure matchups"))
    # co-variability of the closure matchup set itself: one point per station-date,
    # within-station anomalies (so no closure residual can create it)
    sd = x.groupby(["name_en", "date"]).agg(G=("h_gauge_evrf2019_A_m", "median"),
                                            H=("h_icesat_egg2015_m", "median")).reset_index()
    sd["Ga"] = sd.G - sd.groupby("name_en").G.transform("mean")
    sd["Ha"] = sd.H - sd.groupby("name_en").H.transform("mean")
    lo, hi = boot_r(sd.Ga, sd.Ha)
    for k, v in (("matchup_set_n_station_dates", len(sd)),
                 ("matchup_set_pearson_r", stats.pearsonr(sd.Ga, sd.Ha)[0]),
                 ("matchup_set_spearman_rho", stats.spearmanr(sd.Ga, sd.Ha)[0]),
                 ("matchup_set_theil_sen", stats.theilslopes(sd.Ha, sd.Ga)[0])):
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="station-date",
                            ci_lo=lo if k.endswith("pearson_r") else np.nan,
                            ci_hi=hi if k.endswith("pearson_r") else np.nan))
    # how well the tie transfers between gauges, and whether c drifts along the reach
    al = pd.read_csv(IS2_TAB / "kakhovka_alignment_by_station.csv")[["name_en", "distance_from_dam_km", "lon"]]
    med = x.groupby("name_en").c_m.median().to_frame("c").join(al.set_index("name_en"))
    e_mean = np.array([med.c[k] - med.c.drop(k).mean() for k in med.index])
    e_near = np.array([med.c[k] - med.c.drop(k)[(med.distance_from_dam_km.drop(k)
                                                  - med.distance_from_dam_km[k]).abs().idxmin()]
                       for k in med.index])
    lr_lon = stats.linregress(med.lon, med.c)
    lr_km = stats.linregress(med.distance_from_dam_km, med.c)
    tq = stats.t.ppf(0.975, len(med) - 2)
    for k, v, extra in (
            ("loo_mean_rmse_m", np.sqrt(np.mean(e_mean ** 2)), dict(note="withheld gauge predicted by the mean of the other five station medians")),
            ("loo_mean_max_abs_m", np.abs(e_mean).max(), {}),
            ("loo_nearest_rmse_m", np.sqrt(np.mean(e_near ** 2)), dict(note="predicted by the nearest other gauge along the reach (chainage)")),
            ("drift_per_degree_lon_m", lr_lon.slope, dict(ci_lo=lr_lon.stderr, note=f"OLS of the six station medians on longitude; ci_lo holds the standard error; p = {lr_lon.pvalue:.3f}")),
            ("drift_per_degree_lon_p", lr_lon.pvalue, {}),
            ("gradient_along_reach_cm_per_km", 100 * lr_km.slope,
             dict(ci_lo=100 * (lr_km.slope - tq * lr_km.stderr), ci_hi=100 * (lr_km.slope + tq * lr_km.stderr),
                  note=f"OLS of the six station medians on chainage from the dam, t interval (4 df); p = {lr_km.pvalue:.3f}")),
            ("gradient_along_reach_p", lr_km.pvalue, {})):
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="station", **extra))
    agg = dict(n_matchups=len(x), n_dates=x.groupby("name_en").date.nunique().sum(),
               n_stations=x.name_en.nunique(), mean_c_m=per.mean(), median_c_m=per.median(),
               sd_between_station_m=per.std(), range_between_station_m=per.max() - per.min())
    for k, v in agg.items():
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="station / beam transect"))
    return x


# ---------------------------------------------------------------------- V2 --
def v2(evidence: list, summary: list) -> None:
    C = "V2_ATL13_GAUGE_COVARIABILITY"
    pl = pd.read_csv(IS2_TAB / "atl13_pass_levels.csv")
    pl = pl[(pl.period == "PRE_BREACH") & pl.qc_pass]
    ov = pl.groupby(["rgt", "date"]).agg(H=("median_wse_evrs_m", "median"),
                                         lon=("lon_mean", "median"),
                                         lat=("lat_mean", "median")).reset_index()
    gx = {i: (lo, la) for i, _, lo, la in CFG.RESERVOIR_GAUGES}
    names = {i: n for i, n, *_ in CFG.RESERVOIR_GAUGES}
    d = np.stack([haversine_km(ov.lon.values, ov.lat.values, *gx[i]) for i in gx], axis=1)
    ids = np.array(list(gx))
    ov["station_id"] = ids[d.argmin(1)]
    ov["dist_km"] = d.min(1)
    g = gauge_levels()
    ov["date_ts"] = pd.to_datetime(ov.date)
    ov = ov.merge(g[["station_id", "date", "H_evrf2019_m"]],
                  left_on=["station_id", "date_ts"], right_on=["station_id", "date"],
                  how="left", suffixes=("", "_g"))
    keep = (ov.dist_km <= V2_MAX_KM) & ov.H_evrf2019_m.notna()
    for _, r in ov.iterrows():
        k = (r.dist_km <= V2_MAX_KM) and pd.notna(r.H_evrf2019_m)
        evidence.append(dict(claim_id=C, validation_path="V2", station=names[r.station_id],
                             sensor="ICESat-2 ATL13", period="PRE_BREACH",
                             overpass_id=f"{r.rgt}_{r.date}", date=r.date,
                             distance_m=1000 * r.dist_km, gauge_level_m=r.H_evrf2019_m,
                             # companion pass levels are tide-free; + free2mean puts
                             # them in the mean-tide crust of every other closure
                             satellite_level_m=r.H + CFG.free2mean(r.lat), included_primary=bool(k),
                             exclusion_reason="" if k else (
                                 "no same-date gauge value" if pd.isna(r.H_evrf2019_m)
                                 else f"> {V2_MAX_KM:g} km from every gauge")))
    x = ov[keep].copy()
    x["Ha"] = x.H - x.groupby("station_id").H.transform("mean")
    x["Ga"] = x.H_evrf2019_m - x.groupby("station_id").H_evrf2019_m.transform("mean")
    lo, hi = boot_r(x.Ga, x.Ha)
    res = dict(n_overpasses=len(x), pearson_r=stats.pearsonr(x.Ga, x.Ha)[0],
               spearman_rho=stats.spearmanr(x.Ga, x.Ha)[0],
               theil_sen=stats.theilslopes(x.Ha, x.Ga)[0])
    for k, v in res.items():
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="overpass",
                            ci_lo=lo if k == "pearson_r" else np.nan,
                            ci_hi=hi if k == "pearson_r" else np.nan))
    for sid, gs in x.groupby("station_id"):
        if len(gs) >= 3:
            summary.append(dict(claim_id=C, statistic="station_pearson_r", station=names[sid],
                                value=stats.pearsonr(gs.H_evrf2019_m, gs.H)[0], n=len(gs),
                                unit="overpass", note="absolute levels, V2 sample"))
    # changes between successive visits at the same station
    x = x.sort_values(["station_id", "date_ts"])
    x["dH"] = x.groupby("station_id").H.diff()
    x["dG"] = x.groupby("station_id").H_evrf2019_m.diff()
    y = x.dropna(subset=["dH", "dG"])
    lo, hi = boot_r(y.dG, y.dH)
    for k, v in dict(diff_n=len(y), diff_pearson_r=stats.pearsonr(y.dG, y.dH)[0],
                     diff_spearman_rho=stats.spearmanr(y.dG, y.dH)[0],
                     diff_theil_sen=stats.theilslopes(y.dH, y.dG)[0]).items():
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="successive-visit change",
                            ci_lo=lo if k == "diff_pearson_r" else np.nan,
                            ci_hi=hi if k == "diff_pearson_r" else np.nan))


# ---------------------------------------------------------------------- V3 --
def v3(v1x: pd.DataFrame, evidence: list, summary: list) -> None:
    C = "V3_ROZUMIVKA_TRANSFER"
    is2 = v1x[v1x.station_id == ROZ]
    c_is2 = float(is2.c_m.median())
    summary += rows_stats(C, is2.c_m, "beam transect", sensor="ICESat-2", period="PRE_BREACH")

    lon0, lat0 = next((lo, la) for i, _, lo, la in CFG.RESERVOIR_GAUGES if i == ROZ)
    nodes = pd.read_parquet(XC.CACHE / "nodes_all.parquet")
    nodes = XS.latest_version(nodes)
    nodes["dist_km"] = haversine_km(nodes.lon.values, nodes.lat.values, lon0, lat0)
    near = nodes[nodes.dist_km <= max(V3_RADII_KM)].copy()
    ref = near.loc[near.dist_km.idxmin()]
    s0 = float(ref.p_dist_out)            # SWORD distance of the node nearest the post
    good = (near.node_q <= XS.NODE_Q_MAX) & (near.dark_frac < XS.DARK_MAX)
    near["t"] = pd.to_datetime(near.t, utc=True)
    near["date"] = near.t.dt.tz_convert(None).dt.normalize()
    near = near[near.date >= BREACH]
    g = gauge_levels()
    g = g[g.station_id == ROZ][["date", "H_evrf2019_m"]]
    for radius in V3_RADII_KM:
        sel = near[(near.dist_km <= radius)]
        per = []
        for pid, p in sel.groupby("pass_id"):
            pg = p[good.reindex(p.index).values]
            date = p.date.iloc[0]
            gl = g.loc[g.date == date, "H_evrf2019_m"]
            reason = ""
            if pg.empty:
                reason = "no node with node_q <= 1 and dark_frac < 0.5"
            elif gl.empty:
                reason = "no same-date gauge value"
            H_raw = float(pg.H.median()) if len(pg) else np.nan
            H_slope = np.nan
            if len(pg) >= 3 and pg.p_dist_out.nunique() >= 3:
                b, a, *_ = stats.theilslopes(pg.H.values, pg.p_dist_out.values)
                H_slope = a + b * s0
            G = float(gl.iloc[0]) if len(gl) else np.nan
            row = dict(pass_id=pid, date=date, n_nodes=len(pg), H_raw=H_raw,
                       H_slope=H_slope, G=G, ok=reason == "")
            per.append(row)
            evidence.append(dict(claim_id=C, validation_path="V3", station="Rozumivka",
                                 sensor="SWOT RiverSP", period="POST_BREACH",
                                 swot_pass_id=pid, date=date.date().isoformat(),
                                 radius_km=radius, n_nodes=len(pg),
                                 distance_m=1000 * float(pg.dist_km.median()) if len(pg) else np.nan,
                                 gauge_level_m=G, satellite_level_m=H_raw,
                                 satellite_level_slope_corrected_m=H_slope,
                                 closure_residual_m=G - H_raw,
                                 closure_residual_slope_corrected_m=G - H_slope,
                                 included_primary=bool(reason == "" and radius == V3_PRIMARY_KM),
                                 exclusion_reason=reason or (
                                     "" if radius == V3_PRIMARY_KM else "radius sensitivity")))
        P = pd.DataFrame(per)
        P = P[P.ok]
        for variant, col in (("raw_median", "H_raw"), ("slope_corrected", "H_slope")):
            q = P.dropna(subset=[col])
            c = q.G - q[col]
            tag = dict(sensor="SWOT", period="POST_BREACH", radius_km=radius, variant=variant,
                       date_range=f"{q.date.min().date()}..{q.date.max().date()}" if len(q) else "")
            summary += rows_stats(C, c, "SWOT pass", **tag)
            if len(q) >= 5:
                # does SWOT follow the gauge through the new river's swings?
                # (sensitivity to variability, not absolute agreement)
                for k, v in (("covar_pearson_r", stats.pearsonr(q.G, q[col])[0]),
                             ("covar_spearman_rho", stats.spearmanr(q.G, q[col])[0]),
                             ("covar_theil_sen", stats.theilslopes(q[col], q.G)[0]),
                             ("covar_gauge_range_m", q.G.max() - q.G.min())):
                    summary.append(dict(claim_id=C, statistic=k, value=v, unit="SWOT pass", **tag))
            # the comparison, only after both closures exist independently
            summary += rows_stats(C, c - c_is2, "SWOT pass", **dict(
                tag, variant=f"{variant}: c_SWOT - c_IS2 (frozen pre-breach transfer)"))
            # comparison of two independent estimates: resample both samples
            if len(c) >= 3:
                a_, b_ = c.values, is2.c_m.values
                rg = _rng(a_, b_)
                dd = (np.median(a_[rg.integers(0, len(a_), (N_BOOT, len(a_)))], axis=1)
                      - np.median(b_[rg.integers(0, len(b_), (N_BOOT, len(b_)))], axis=1))
                summary.append(dict(claim_id=C, statistic="median_diff_m", unit="SWOT pass / beam transect",
                                    value=np.median(a_) - np.median(b_),
                                    ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5),
                                    **dict(tag, variant=f"{variant}: median c_SWOT - median c_IS2, "
                                                        f"both resampled")))


# ---------------------------------------------------------------------- V4 --
def v4(evidence: list, summary: list) -> pd.DataFrame:
    C = "V4_SWOT_ICESAT_DIRECT"
    G = XC.geoms()
    atl, nodes, lakes, pairs = XC.load_cached(G)
    p = XC.annotate(pairs, nodes, G, lambda x, y: np.zeros(len(x), bool))
    p = p.join(nodes[["p_dist_out"]], on="node_row")
    q = p[XC.QUALITY["q1_dark"](p) & (p.dt_d.abs() <= 10) & (p.n_seg_200 > 0)]
    q = q.assign(d=q.H_swot - q.H_atl_200)
    gb = q.groupby(["ovp", "pass_id"])
    x = pd.DataFrame(dict(icesat_time=gb.icesat_time.first(), dt_days=gb.dt_d.median(),
                          n_nodes=gb.d.size(), d_m=gb.d.median(), H_swot=gb.H_swot.median(),
                          H_atl=gb.H_atl_200.median(), dist_out=gb.p_dist_out.median(),
                          x=gb.x.median(), y=gb.y.median(),
                          node_q_max=gb.node_q.max(), dark_max=gb.dark_frac.max())).reset_index()
    zones = {L: SD.load_utm(n) for L, n in XS.PAPER_ZONES.items()}
    u = unary_union(list(zones.values()))
    x = x[shapely.contains_xy(u, x.x.values, x.y.values)].copy()
    x["zone"] = "none"
    for L, gz in zones.items():
        x.loc[shapely.contains_xy(gz, x.x.values, x.y.values), "zone"] = L
    # guard: the same crossings as ms6b (row by row)
    ref = pd.read_csv(XS.TAB / "ms6b_swot_icesat_crossings.csv").query("universe == 'paper_RFDE'")
    a = x.sort_values(["ovp", "pass_id"])
    b = ref.sort_values(["ovp", "swot_pass"])
    assert (a.ovp.values == b.ovp.values).all() and np.allclose(a.d_m.values, b.d_m.values)
    x["primary"] = x.dt_days.abs() <= 1
    ll = gpd.GeoSeries(gpd.points_from_xy(x.x, x.y), crs=CFG.CRS_METRIC).to_crs(4326)
    x["lon"], x["lat"] = ll.x.values, ll.y.values
    for _, r in x.iterrows():
        evidence.append(dict(claim_id=C, validation_path="V4", sensor="SWOT RiverSP - ATL13",
                             zone=r.zone, overpass_id=r.ovp, swot_pass_id=r.pass_id,
                             date=pd.Timestamp(r.icesat_time).date().isoformat(),
                             dt_hours=24 * r.dt_days, n_nodes=r.n_nodes,
                             gauge_level_m=np.nan, satellite_level_m=r.H_swot,
                             icesat_level_m=r.H_atl, closure_residual_m=r.d_m,
                             sword_dist_out_m=r.dist_out, lon=r.lon, lat=r.lat,
                             included_primary=bool(r.primary),
                             exclusion_reason="" if r.primary else "|dt| > 24 h (timing sensitivity)"))
    y = x[x.primary].copy()
    summary += rows_stats(C, y.d_m, "crossing (ovp x SWOT pass)")
    for lo_d, hi_d in ((1, 3), (3, 10)):
        w = x[(x.dt_days.abs() > lo_d) & (x.dt_days.abs() <= hi_d)]
        summary += rows_stats(C, w.d_m, "crossing (ovp x SWOT pass)",
                              variant=f"timing sensitivity {lo_d}-{hi_d} d")
    frame_check(x, atl, nodes, summary)
    chain_variants(x, atl, nodes, summary, evidence)
    # sensitivity: the breach fortnight, when the level moved by decimetres per
    # hour, so a crossing hours apart compares two different water surfaces
    it = pd.to_datetime(y.icesat_time, utc=True).dt.tz_convert(None)
    calm = ~it.between(BREACH, BREACH + pd.Timedelta(days=14))
    summary += rows_stats(C, y.d_m[calm], "crossing (ovp x SWOT pass)",
                          variant="excluding 2023-06-06..06-20 (breach fortnight)")
    X, Y = y.H_atl.values, y.H_swot.values
    lo, hi = boot_r(X, Y)
    ts = stats.theilslopes(Y, X)
    dm, pb = deming(X, Y), passing_bablok(X, Y)
    for k, v, extra in (("pearson_r_raw", stats.pearsonr(X, Y)[0], dict(ci_lo=lo, ci_hi=hi)),
                        ("spearman_rho_raw", stats.spearmanr(X, Y)[0], {}),
                        ("theil_sen_slope", ts[0], dict(ci_lo=ts[2], ci_hi=ts[3])),
                        ("theil_sen_intercept_m", ts[1], {}),
                        ("deming_slope", dm[0], {}), ("deming_intercept_m", dm[1], {}),
                        ("passing_bablok_slope", pb[0], {}),
                        ("passing_bablok_intercept_m", pb[1], {}),
                        ("height_range_m", X.max() - X.min(), {})):
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="crossing", **extra))
    # spatial anomalies: remove the zone median of the pair means (zones n >= 3)
    y["Hm"] = 0.5 * (y.H_swot + y.H_atl)
    nz = y.zone.map(y.zone.value_counts())
    a = y[nz >= V4_MIN_N_ANOM].copy()
    med = a.groupby("zone").Hm.transform("median")
    a["As"], a["Aa"] = a.H_swot - med, a.H_atl - med
    lo, hi = boot_r(a.Aa, a.As)
    for k, v, extra in (("n_anomaly", len(a), dict(note=f"zones with n >= {V4_MIN_N_ANOM}: "
                                                  + ",".join(sorted(a.zone.unique())))),
                        ("pearson_r_anomaly", stats.pearsonr(a.Aa, a.As)[0], dict(ci_lo=lo, ci_hi=hi)),
                        ("spearman_rho_anomaly", stats.spearmanr(a.Aa, a.As)[0], {}),
                        ("theil_sen_slope_anomaly", stats.theilslopes(a.As, a.Aa)[0], {}),
                        ("anomaly_range_m", a.Aa.max() - a.Aa.min(), {})):
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="crossing", **extra))
    # the two things that can carry a correlation: one leverage point, one event
    ita = pd.to_datetime(a.icesat_time, utc=True).dt.tz_convert(None)
    ac = a[~ita.between(BREACH, BREACH + pd.Timedelta(days=14)).values]
    yr = y[y.zone != "R"]
    summary.append(dict(claim_id=C, statistic="pearson_r_anomaly", value=stats.pearsonr(ac.Aa, ac.As)[0],
                        unit="crossing", variant="excluding 2023-06-06..06-20 (breach fortnight)",
                        note=f"n = {len(ac)}"))
    summary.append(dict(claim_id=C, statistic="spearman_rho_anomaly", value=stats.spearmanr(ac.Aa, ac.As)[0],
                        unit="crossing", variant="excluding 2023-06-06..06-20 (breach fortnight)",
                        note=f"n = {len(ac)}"))
    summary.append(dict(claim_id=C, statistic="pearson_r_raw", value=stats.pearsonr(yr.H_atl, yr.H_swot)[0],
                        unit="crossing", variant="excluding the single R crossing (leverage)",
                        note=f"n = {len(yr)}"))
    # regression controlling for zone and SWORD distance
    Z = pd.get_dummies(y.zone, drop_first=True).astype(float)
    M = np.column_stack([np.ones(len(y)), y.H_atl, y.dist_out / 1e3, Z.values])
    beta = np.linalg.lstsq(M, y.H_swot.values, rcond=None)[0]
    bb = []
    rg = _rng(y.H_atl, y.H_swot)
    for _ in range(2000):
        i = rg.integers(0, len(y), len(y))
        bb.append(np.linalg.lstsq(M[i], y.H_swot.values[i], rcond=None)[0][1])
    summary.append(dict(claim_id=C, statistic="atl13_coef_controlling_zone_chainage",
                        value=beta[1], ci_lo=np.nanpercentile(bb, 2.5),
                        ci_hi=np.nanpercentile(bb, 97.5), unit="crossing",
                        note="SWOT ~ ATL13 + SWORD dist_out + zone"))
    # zone-specific: correlation only with enough pairs
    for zn, gz in y.groupby("zone"):
        summary.append(dict(claim_id=C, statistic="zone_n", zone=zn, value=len(gz), unit="crossing"))
        summary.append(dict(claim_id=C, statistic="zone_median_d_m", zone=zn,
                            value=gz.d_m.median(), unit="crossing"))
        if len(gz) >= V4_MIN_N_ZONE_R:
            summary.append(dict(claim_id=C, statistic="zone_pearson_r_raw", zone=zn,
                                value=stats.pearsonr(gz.H_atl, gz.H_swot)[0], unit="crossing"))
        else:
            summary.append(dict(claim_id=C, statistic="zone_correlation", zone=zn, value=np.nan,
                                unit="crossing", note=f"n < {V4_MIN_N_ZONE_R}: descriptive pairs only"))
    return y, a


# ---------------------------------------------------------------------- V6 --
KHERSON = 80805
V6_ICESAT_KM, V6_RIVERSP_KM, V6_PIXC_KM = 10.0, 3.0, 1.0
V6_PIXC_RADII = (0.5, 1.0, 2.0, 3.0, 5.0)
V6_POST_FROM = pd.Timestamp("2023-07-01")   # after the breach month (flood + recession)
PIXC_DIR = ROOT / "data/raw/pixc_nova_kakhovka"


def _closure_series(C, label, sensor, period, support, units: pd.DataFrame, evidence, summary):
    """units: one row per independent unit with columns unit_id, date, H (sat,
    EGG2015 mean-tide), G (gauge EVRF2019), n, dist_m."""
    u = units.dropna(subset=["H", "G"]).copy()
    u["c"] = u.G - u.H
    for _, r in units.iterrows():
        ok = pd.notna(r.H) and pd.notna(r.G)
        evidence.append(dict(claim_id=C, validation_path="V6", station="Kherson", sensor=sensor,
                             period=period, series=label, radius_km=support,
                             overpass_id=r.unit_id if "ICESat" in sensor else "",
                             swot_pass_id=r.unit_id if "SWOT" in sensor else "",
                             date=pd.Timestamp(r.date).date().isoformat(), n_nodes=r.n,
                             distance_m=r.dist_m, gauge_level_m=r.G, satellite_level_m=r.H,
                             closure_residual_m=(r.G - r.H) if ok else np.nan,
                             included_primary=bool(ok),
                             exclusion_reason="" if ok else "no same-date gauge value"))
    summary.extend(rows_stats(C, u.c, "pass / overpass", sensor=sensor, period=period,
                              series=label, radius_km=support,
                              date_range=f"{u.date.min().date()}..{u.date.max().date()}" if len(u) else ""))
    if len(u) >= 5:
        lo_r, hi_r = boot_r(u.G, u.H)
        summary.append(dict(claim_id=C, statistic="covar_pearson_r_ci", value=np.nan, ci_lo=lo_r,
                            ci_hi=hi_r, unit="pass / overpass", sensor=sensor, period=period,
                            series=label, radius_km=support))
        for k, v in (("covar_pearson_r", stats.pearsonr(u.G, u.H)[0]),
                     ("covar_spearman_rho", stats.spearmanr(u.G, u.H)[0]),
                     ("covar_theil_sen", stats.theilslopes(u.H, u.G)[0]),
                     ("covar_gauge_range_m", u.G.max() - u.G.min())):
            summary.append(dict(claim_id=C, statistic=k, value=v, unit="pass / overpass",
                                sensor=sensor, period=period, series=label, radius_km=support))
    return u


def v6(evidence: list, summary: list) -> dict:
    """Kherson as a local anchor: gauge - satellite closure per sensor/product
    and period, each series with its own support, pass IDs in the evidence."""
    C = "V6_KHERSON_CLOSURE"
    lon0, lat0 = CFG.KHERSON_GAUGE[2:]
    g = gauge_levels()
    g = g[g.station_id == KHERSON].set_index("date").H_evrf2019_m
    gate = lambda d: g.get(pd.Timestamp(d).normalize(), np.nan)
    out = {}

    # ICESat-2 ATL13, overpass = (rgt, date), segments within 10 km
    cols = ["time", "rgt", "beam", "segment_id", "lat", "lon", "h_wgs84_m"]
    a = pd.concat([pd.read_parquet(f, columns=cols) for f in XS.ATL13], ignore_index=True)
    a = a.drop_duplicates(["time", "beam", "segment_id"])
    a["dist_km"] = haversine_km(a.lon.values, a.lat.values, lon0, lat0)
    a = a[a.dist_km <= V6_ICESAT_KM].copy()
    a["H"] = (a.h_wgs84_m + CFG.free2mean(a.lat.values)
              - sample_grid_bilinear(CFG.EGG2015_TIF, a.lon.values, a.lat.values))
    a["date"] = a.time.dt.tz_convert(None).dt.normalize()
    ov = a.groupby(["rgt", "date"]).agg(H=("H", "median"), n=("H", "size"),
                                        dist_m=("dist_km", lambda v: 1000 * np.median(v))).reset_index()
    ov["unit_id"] = ov.rgt.astype(str) + "_" + ov.date.dt.strftime("%Y%m%d")
    ov["G"] = ov.date.map(gate)
    for per, sel in (("PRE_BREACH", ov.date < BREACH), ("POST_BREACH", ov.date >= V6_POST_FROM)):
        out[("ICESat-2", per)] = _closure_series(C, f"ICESat-2 {per.lower()}", "ICESat-2 ATL13",
                                                 per, V6_ICESAT_KM, ov[sel], evidence, summary)

    # SWOT RiverSP, pass = (cycle_pass, date), good nodes within 3 km
    n = XS.latest_version(pd.read_parquet(XC.CACHE / "nodes_all.parquet"))
    n = n[(n.node_q <= XS.NODE_Q_MAX) & (n.dark_frac < XS.DARK_MAX)].copy()
    n["dist_km"] = haversine_km(n.lon.values, n.lat.values, lon0, lat0)
    n = n[n.dist_km <= V6_RIVERSP_KM].copy()
    n["date"] = pd.to_datetime(n.t, utc=True).dt.tz_convert(None).dt.normalize()
    ps = n.groupby("pass_id").agg(H=("H", "median"), n=("H", "size"), date=("date", "first"),
                                  dist_m=("dist_km", lambda v: 1000 * np.median(v))).reset_index()
    ps = ps.rename(columns={"pass_id": "unit_id"})
    ps["G"] = ps.date.map(gate)
    for per, sel in (("PRE_BREACH", ps.date < BREACH), ("POST_BREACH", ps.date >= V6_POST_FROM)):
        out[("RiverSP", per)] = _closure_series(C, f"SWOT RiverSP {per.lower()}", "SWOT RiverSP",
                                                per, V6_RIVERSP_KM, ps[sel], evidence, summary)

    # SWOT PIXC, open water (class 4) within 1 km, tide-corrected ellipsoidal height
    rows, sens = [], []
    d = 0.06
    for f in sorted(PIXC_DIR.glob("SWOT_L2_HR_PIXC_*.nc")):
        px = read_pixc(
            f, bbox=(lon0 - d, lat0 - d, lon0 + d, lat0 + d))
        dist = haversine_km(px["lon"], px["lat"], lon0, lat0)
        m = dist <= V6_PIXC_KM
        date = pd.Timestamp(px["time_granule_start"][:10])
        zeta_all = sample_grid_bilinear(CFG.EGG2015_TIF, px["lon"], px["lat"]) if len(dist) else []
        for rad in V6_PIXC_RADII:
            mm = dist <= rad
            if mm.any():
                sens.append(dict(date=date, radius_km=rad,
                                 H=float(np.median(px["h_ell"][mm] - zeta_all[mm]))))
        if m.sum() == 0:
            rows.append(dict(unit_id=f.name[:-3], date=date, H=np.nan, n=0, dist_m=np.nan))
            continue
        H = px["h_ell"][m] - sample_grid_bilinear(CFG.EGG2015_TIF, px["lon"][m], px["lat"][m])
        rows.append(dict(unit_id=f"{px['cycle']}_{px['pass']:03d}_{px['tile']}", date=date,
                         H=float(np.median(H)), n=int(m.sum()),
                         dist_m=1000 * float(np.median(dist[m]))))
    pp = pd.DataFrame(rows)
    pp["G"] = pp.date.map(gate)
    pp = pp[pp.date < BREACH]
    # aggregation-radius sensitivity of the PIXC closure
    sv = pd.DataFrame(sens)
    sv = sv[sv.date < BREACH]
    sv["c"] = sv.date.map(gate) - sv.H
    med = sv.groupby("radius_km").c.median()
    for rad, v in med.items():
        summary.append(dict(claim_id=C, statistic="pixc_median_c_by_radius_m", radius_km=rad,
                            value=v, series="SWOT PIXC pre_breach", unit="pass"))
    summary.append(dict(claim_id=C, statistic="pixc_median_c_range_over_radii_m",
                        value=med.max() - med.min(), series="SWOT PIXC pre_breach", unit="pass",
                        note=f"radii {min(V6_PIXC_RADII):g}-{max(V6_PIXC_RADII):g} km"))
    out[("PIXC", "PRE_BREACH")] = _closure_series(C, "SWOT PIXC pre_breach", "SWOT PIXC",
                                                  "PRE_BREACH", V6_PIXC_KM, pp, evidence, summary)
    return out


def chain_variants(x: pd.DataFrame, atl: pd.DataFrame, nodes: pd.DataFrame, summary: list,
                   evidence: list) -> None:
    """Which vertical chain closes the gap (Figure S3b, Section S1.5), on the
    primary crossings (<= 24 h, R/F/D/E): the production chain, the ATL13
    permanent-tide term omitted, the term applied with the opposite sign, and
    the naive product-to-product difference wse - ht_ortho. Plus the geoid
    each product carries at the same nodes (Figure S3a)."""
    C = "V4_SWOT_ICESAT_DIRECT"
    prim = x[x.dt_days.abs() <= 1]
    a0 = atl[atl.ovp.isin(prim.ovp.unique())]
    cols = ["time", "beam", "segment_id", "H_egm2008_m"]
    egm = pd.concat([pd.read_parquet(f, columns=cols) for f in XS.ATL13], ignore_index=True)
    a0 = a0.merge(egm.drop_duplicates(["time", "beam", "segment_id"]),
                  on=["time", "beam", "segment_id"], how="left")
    n = nodes[(nodes.node_q <= XS.NODE_Q_MAX) & (nodes.dark_frac < XS.DARK_MAX)].copy()
    zeta_a = a0.h_wgs84_m - a0.H + CFG.free2mean(a0.lat.values)       # production zeta
    f2m = CFG.free2mean(a0.lat.values)
    variants = {
        "production: EGG2015, ATL13 + mean-tide term": (a0.H, n.H),
        "ATL13 permanent-tide term omitted": (a0.h_wgs84_m - zeta_a, n.H),
        "ATL13 permanent-tide term with the opposite sign": (a0.h_wgs84_m - f2m - zeta_a, n.H),
        "naive product difference wse - ht_ortho": (a0.H_egm2008_m, n.wse),
    }
    for name, (ha, hn) in variants.items():
        aa, nn = a0.assign(H=ha.values), n.assign(H=hn.values)
        e = XS.group_crossings(XS.node_pairs(aa.dropna(subset=["H"]), nn))
        m = prim.merge(e[["ovp", "swot_pass", "d_m"]], left_on=["ovp", "pass_id"],
                       right_on=["ovp", "swot_pass"], suffixes=("_prod", ""))
        summary.extend(rows_stats(C, m.d_m, "crossing (ovp x SWOT pass)", variant=f"chain: {name}"))
    # geoid carried by each product at the same nodes
    aa = a0.assign(H=(a0.h_wgs84_m - a0.H_egm2008_m).values)
    nn = n.assign(H=n.geoid_hght.values)
    pr = XS.node_pairs(aa.dropna(subset=["H"]), nn)
    pr = pr[pr.n_seg_200 > 0]
    g = (pr.H_swot - pr.H_atl_200).groupby(pr.node_row).median()
    phi = np.deg2rad(n.loc[g.index, "lat"].median())
    k_love = 0.3
    expected = (1 + k_love) * (0.099 - 0.296 * np.sin(phi) ** 2)   # N_mean - N_free, m (IERS)
    for k, v, note in (("geoid_diff_n_nodes", len(g), "SWOT geoid_hght - ATL13 geoid, all good nodes within 200 m of the primary overpasses' segments (the geoid is static)"),
                       ("geoid_diff_median_m", g.median(), ""),
                       ("geoid_diff_nmad_m", nmad(g), ""),
                       ("geoid_diff_expected_if_atl13_tide_free_m", expected,
                        f"(1+k)(0.099-0.296 sin^2 phi), k={k_love}, phi={np.rad2deg(phi):.2f}")):
        summary.append(dict(claim_id=C, statistic=k, value=v, unit="node", note=note))
    for nr, v in g.items():
        evidence.append(dict(claim_id="V4G_PRODUCT_GEOIDS_AT_NODES", validation_path="V4-geoid",
                             sensor="geoid SWOT - ATL13",
                             swot_pass_id=f"node_row {nr}", closure_residual_m=v,
                             included_primary=True, exclusion_reason=""))


# ---------------------------------------------------------------------- V8 --
def v8_reference_surfaces(summary: list, evidence: list) -> None:
    """The reference surfaces between the sensors, measured at the good SWOT
    nodes inside R/F/D/E (one value per node location): SWOT geoid_hght
    (EGM2008) - zeta_EGG2015, the EPSG:9902 grid, and SWOT geoid_hght against
    the NGA EGM2008 grid (PROJ us_nga_egm08_25, local file, no network)."""
    import pyproj
    from pyproj import Transformer
    C = "V8_REFERENCE_SURFACES"
    n = XS.latest_version(pd.read_parquet(XC.CACHE / "nodes_all.parquet"))
    n = n[(n.node_q <= XS.NODE_Q_MAX) & (n.dark_frac < XS.DARK_MAX)]
    u = unary_union([SD.load_utm(v) for v in XS.PAPER_ZONES.values()])
    n = n[shapely.contains_xy(u, n.x.values, n.y.values)]
    g = n.groupby("node_id").agg(lon=("lon", "median"), lat=("lat", "median"), x=("x", "median"),
                                 y=("y", "median"), N=("geoid_hght", "median")).reset_index()
    g["zeta"] = sample_grid_bilinear(CFG.EGG2015_TIF, g.lon.values, g.lat.values)
    g["sep"] = g.N - g.zeta
    A = np.column_stack([np.ones(len(g)), (g.x - g.x.mean()) / 1e3, (g.y - g.y.mean()) / 1e3])
    coef = np.linalg.lstsq(A, g.sep.values, rcond=None)[0]
    grad_mm_km = 1000 * np.hypot(coef[1], coef[2])
    from swot_dnipro.vertical import sample_grid
    g["d9902"] = sample_grid(CFG.UA2019Z_ASC, g.lon.values, g.lat.values)
    pyproj.network.set_network_enabled(False)
    t = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
    _, _, H = t.transform(g.lon.values, g.lat.values, np.zeros(len(g)))
    g["N_nga"] = -np.asarray(H)
    rows = (("n_nodes", len(g), "good RiverSP node locations in R/F/D/E"),
            ("egm2008_minus_egg2015_median_m", g.sep.median(), "SWOT geoid_hght - zeta_EGG2015 (bilinear)"),
            ("egm2008_minus_egg2015_nmad_m", nmad(g.sep), ""),
            ("egm2008_minus_egg2015_min_m", g.sep.min(), ""),
            ("egm2008_minus_egg2015_max_m", g.sep.max(), ""),
            ("egm2008_minus_egg2015_planar_gradient_mm_per_km", grad_mm_km, "plane fitted in EPSG:32636"),
            ("epsg9902_min_m", np.nanmin(g.d9902), "ua_2019z.asc at the nodes"),
            ("epsg9902_max_m", np.nanmax(g.d9902), ""),
            ("epsg9902_median_m", np.nanmedian(g.d9902), ""),
            ("swot_geoid_minus_nga_egm2008_median_m", (g.N - g.N_nga).median(), "NGA EGM2008 via PROJ us_nga_egm08_25"),
            ("swot_geoid_minus_nga_egm2008_nmad_m", nmad(g.N - g.N_nga), ""))
    for k, v, note in rows:
        summary.append(dict(claim_id=C, statistic=k, value=float(v), unit="node", note=note))
    for _, r in g.iterrows():
        evidence.append(dict(claim_id=C, validation_path="V8", sensor="reference surfaces",
                             swot_pass_id=f"node {r.node_id}", lon=r.lon, lat=r.lat,
                             closure_residual_m=r.sep, gauge_level_m=r.d9902, satellite_level_m=r.N,
                             icesat_level_m=r.zeta, included_primary=True, exclusion_reason=""))


# ---------------------------------------------------------------------- V7 --
SEA_POSTS = ROOT / "data/historical/sea_posts_2023/daily_levels_cm.csv"
V7_POSTS = {80805: "Kherson", 80807: "Kasperivka", 98032: "Stanislav", 98025: "Parutyne",
            98027: "Mykolaiv", 98022: "Ochakiv"}
V7_BREACH_END = pd.Timestamp("2023-06-30")          # the shaded breach fortnight(s) of Figure S8
KHERSON_RIVER_ONLY = (pd.Timestamp("2023-06-13"), pd.Timestamp("2023-07-08"))
KHERSON_PEAK = (pd.Timestamp("2023-06-06"), pd.Timestamp("2023-06-12"))


def _period(d):
    return np.where(d < BREACH, "pre_breach", np.where(d <= V7_BREACH_END, "breach_fortnight", "post_breach"))


def v7_downstream_posts(summary: list, evidence: list) -> None:
    """The posts below the dam through 2023 (Figures S8, S9; Section S3.9):
    sea-yearbook daily levels (EVRF2019) against SWOT RiverSP (good nodes
    <= 3 km, one value per pass) and ICESat-2 ATL13 (<= 10 km, one per
    overpass). c_pre, the median gauge - SWOT closure over the pre-breach
    passes of 2023, is each post's shift in the figures. At Kherson the river
    yearbook is compared with the sea yearbook and fills the days the sea
    series lost to a recorder failure."""
    C = "V7_DOWNSTREAM_POSTS_2023"
    meta = pd.read_csv(SEA_POSTS.parent / "post_metadata_2023.csv").set_index("post_id")
    lv = pd.read_csv(SEA_POSTS)
    lv = lv[(lv.variable == "water_level") & lv.post_id.isin(V7_POSTS)].copy()
    lv["date"] = pd.to_datetime(lv.date)
    sea = lv.set_index(["post_id", "date"]).H_evrf2019_m
    riv = gauge_levels()
    riv = riv[(riv.station_id == KHERSON) & (riv.date.dt.year == 2023)].set_index("date").H_evrf2019_m

    n = XS.latest_version(pd.read_parquet(XC.CACHE / "nodes_all.parquet"))
    n = n[(n.node_q <= XS.NODE_Q_MAX) & (n.dark_frac < XS.DARK_MAX)].copy()
    n["date"] = pd.to_datetime(n.t, utc=True).dt.tz_convert(None).dt.normalize()
    n = n[n.date.dt.year == 2023]
    cols = ["time", "rgt", "beam", "segment_id", "lat", "lon", "h_wgs84_m"]
    a = pd.concat([pd.read_parquet(f, columns=cols) for f in XS.ATL13], ignore_index=True)
    a = a[a.time.dt.year == 2023].drop_duplicates(["time", "beam", "segment_id"])
    a["date"] = a.time.dt.tz_convert(None).dt.normalize()

    series = {}
    for pid, name in V7_POSTS.items():
        lo, la = meta.loc[pid, "lon"], meta.loc[pid, "lat"]
        G = sea.loc[pid] if pid in sea.index.get_level_values(0) else pd.Series(dtype=float)
        dn = haversine_km(n.lon.values, n.lat.values, lo, la)
        sw = n[dn <= V6_RIVERSP_KM].groupby("pass_id").agg(H=("H", "median"), date=("date", "first"),
                                                           n=("H", "size")).reset_index()
        sw["sensor"], sw["unit_id"] = "SWOT RiverSP", sw.pass_id
        da = haversine_km(a.lon.values, a.lat.values, lo, la)
        ai = a[da <= V6_ICESAT_KM].copy()
        ai["H"] = (ai.h_wgs84_m + CFG.free2mean(ai.lat.values)
                   - sample_grid_bilinear(CFG.EGG2015_TIF, ai.lon.values, ai.lat.values))
        ic = ai.groupby(["rgt", "date"]).agg(H=("H", "median"), n=("H", "size")).reset_index()
        ic["sensor"], ic["unit_id"] = "ICESat-2 ATL13", ic.rgt.astype(str) + "_" + ic.date.dt.strftime("%Y%m%d")
        u = pd.concat([sw[["unit_id", "date", "H", "n", "sensor"]], ic[["unit_id", "date", "H", "n", "sensor"]]])
        u["G"] = u.date.map(G)
        u["period"] = _period(u.date)
        pre = u[(u.sensor == "SWOT RiverSP") & (u.period == "pre_breach")].dropna(subset=["G"])
        c_pre = float((pre.G - pre.H).median()) if len(pre) else np.nan
        summary.append(dict(claim_id=C, statistic="c_pre_swot_m", station=name, value=c_pre, n=len(pre),
                            unit="pass", note="median gauge - SWOT over the 2023 pre-breach passes; the figure shift"))
        for (sen, per), g in u.dropna(subset=["G"]).groupby(["sensor", "period"]):
            c = g.G - g.H
            summary.append(dict(claim_id=C, statistic="c_median_m", station=name, sensor=sen, period=per,
                                value=c.median(), n=len(g), unit="pass / overpass"))
            summary.append(dict(claim_id=C, statistic="c_nmad_m", station=name, sensor=sen, period=per,
                                value=nmad(c), n=len(g), unit="pass / overpass"))
            if len(g) >= 3:
                summary.append(dict(claim_id=C, statistic="pearson_r", station=name, sensor=sen, period=per,
                                    value=stats.pearsonr(g.G, g.H)[0], n=len(g), unit="pass / overpass"))
                summary.append(dict(claim_id=C, statistic="gauge_range_m", station=name, sensor=sen,
                                    period=per, value=g.G.max() - g.G.min(), n=len(g), unit="pass / overpass"))
        for _, r in u.iterrows():
            ok = pd.notna(r.G)
            evidence.append(dict(claim_id=C, validation_path="V7", station=name, sensor=r.sensor,
                                 period=r.period, overpass_id=r.unit_id if "ICESat" in r.sensor else "",
                                 swot_pass_id=r.unit_id if "SWOT" in r.sensor else "",
                                 date=r.date.date().isoformat(), n_nodes=r.n, gauge_level_m=r.G,
                                 satellite_level_m=r.H, closure_residual_m=(r.G - r.H) if ok else np.nan,
                                 included_primary=bool(ok),
                                 exclusion_reason="" if ok else "no same-date sea-yearbook value"))
        series[name] = (u, c_pre)

    # Kherson: the two yearbooks, and the days only the river yearbook reports
    gs = sea.loc[KHERSON]
    both = pd.concat([gs.rename("sea"), riv.rename("river")], axis=1).dropna()
    d = both.river - both.sea
    only = riv.index.difference(gs.dropna().index)
    for k, v in (("kherson_river_minus_sea_median_m", d.median()), ("kherson_river_minus_sea_sd_m", d.std()),
                 ("kherson_river_minus_sea_nmad_m", nmad(d)), ("kherson_river_minus_sea_max_abs_m", d.abs().max()),
                 ("kherson_yearbooks_common_days", len(d)), ("kherson_river_only_days", len(only))):
        summary.append(dict(claim_id=C, statistic=k, value=float(v), station="Kherson", unit="day",
                            note=f"river-only days {only.min().date()}..{only.max().date()}" if k == "kherson_river_only_days" else ""))
    u, c_pre = series["Kherson"]
    sw = u[u.sensor == "SWOT RiverSP"].copy()
    sw["G_river"] = sw.date.map(riv)
    sw["G_sea"] = sw.date.map(gs)

    # the river-yearbook closure (as V6): the shift for comparisons against the river yearbook
    pre_r = sw[(sw.date < BREACH)].dropna(subset=["G_river"])
    c_river = float((pre_r.G_river - pre_r.H).median())
    summary.append(dict(claim_id=C, statistic="c_pre_swot_river_yearbook_m", station="Kherson", value=c_river,
                        n=len(pre_r), unit="pass", note="median river-yearbook gauge - SWOT, 2023 pre-breach passes"))

    def block(tag, g, gcol):
        g = g.dropna(subset=[gcol])
        if len(g) < 3:
            return
        res = (g.H + c_river) - g[gcol]         # satellite + pre-breach closure - gauge
        for k, v in (("n", len(g)), ("pearson_r", stats.pearsonr(g[gcol], g.H)[0]),
                     ("gauge_range_m", g[gcol].max() - g[gcol].min()),
                     ("residual_after_c_pre_median_m", res.median()), ("residual_after_c_pre_nmad_m", nmad(res))):
            summary.append(dict(claim_id=C, statistic=k, station="Kherson", variant=tag, value=float(v),
                                unit="SWOT pass", note=f"{g.date.min().date()}..{g.date.max().date()}; "
                                                       "residual = SWOT + c_river - river-yearbook gauge"))

    block("river yearbook only, 2023-06-13..07-08",
          sw[sw.date.between(*KHERSON_RIVER_ONLY) & sw.date.isin(only)], "G_river")
    pk = sw[sw.date.between(*KHERSON_PEAK)]
    block("peak, river yearbook, 2023-06-06..06-12", pk, "G_river")

    # day-to-day changes of the daily (fast-sampling) series at Kherson
    g = sw.dropna(subset=["G_river"]).groupby("date").agg(H=("H", "median"), G=("G_river", "first")).sort_index()
    dd = g.diff()
    step = pd.Series(g.index, index=g.index).diff().dt.days
    dd = dd[step == 1]
    for tag, sel in (("recession 2023-06-13..07-08", dd.index.to_series().between(*KHERSON_RIVER_ONLY)),
                     ("before the breach", dd.index < BREACH)):
        x = dd[np.asarray(sel)]
        if len(x) >= 3:
            lo_r, hi_r = boot_r(x.G, x.H)
            summary.append(dict(claim_id=C, statistic="daily_change_pearson_r", station="Kherson", variant=tag,
                                value=stats.pearsonr(x.G, x.H)[0], ci_lo=lo_r, ci_hi=hi_r, n=len(x),
                                unit="consecutive-day pair"))
            summary.append(dict(claim_id=C, statistic="daily_change_spearman_rho", station="Kherson", variant=tag,
                                value=stats.spearmanr(x.G, x.H)[0], n=len(x), unit="consecutive-day pair"))
    # lag structure: where the level moved (breach + recession) and over all of 2023
    base = pd.concat([g.H.asfreq("D"), riv.asfreq("D").rename("G")], axis=1)
    for win, (w0, w1) in (("2023-06-06..07-08", ("2023-06-06", "2023-07-08")), ("all 2023", ("2023-01-01", "2023-12-31"))):
        full = base.loc[w0:w1]
        dH, dG = full.H.diff(), full.G.diff()
        for lag in (-1, 0, 1):
            m = pd.concat([dH, dG.shift(lag)], axis=1).dropna()
            if len(m) >= 3:
                summary.append(dict(claim_id=C, statistic="daily_change_lag_pearson_r", station="Kherson",
                                    variant=f"lag {lag:+d} d, {win}",
                                    value=stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])[0],
                                    n=len(m), unit="consecutive-day pair"))


# ------------------------------------------------------------ gauge network --
NETWORK_ORDER = ["Plavni", "Rozumivka", "Blahovishchenka", "Nikopol", "Velyka Lepetykha",
                 "Nova Kakhovka", "Kherson"]


def gauge_network(summary: list, evidence: list) -> None:
    """Gauge-to-gauge co-variation, 2019-2021 (Figure S7, Section S3.7):
    pairwise Pearson r of daily levels, of linearly detrended levels and of
    daily changes, on pairwise-complete dates, with the Bretherton effective
    sample size n (1 - r1 r2) / (1 + r1 r2) from the lag-1 autocorrelations."""
    import itertools
    C = "GAUGE_NETWORK"
    g = gauge_levels()
    w = g[g.date < "2022-01-01"].pivot_table(index="date", columns="name_en", values="H_evrf2019_m")
    w = w[[c for c in NETWORK_ORDER if c in w]]
    t = np.arange(len(w))
    det = w.apply(lambda c: c - np.polyval(np.polyfit(t[c.notna()], c.dropna(), 1), t))
    d = w.diff()
    rows = []
    for a, b in itertools.combinations(w.columns, 2):
        m = w[[a, b]].dropna()
        r1, r2 = m[a].autocorr(1), m[b].autocorr(1)
        rows.append(dict(a=a, b=b, n=len(m), r_level=m[a].corr(m[b]), r_detrended=det[a].corr(det[b]),
                         r_daily_change=d[a].corr(d[b]), n_eff=len(m) * (1 - r1 * r2) / (1 + r1 * r2),
                         n_change=int(d[[a, b]].dropna().shape[0])))
    R = pd.DataFrame(rows)
    for _, r in R.iterrows():
        evidence.append(dict(claim_id=C, validation_path="network", station=f"{r.a} | {r.b}",
                             sensor="gauge-gauge", n_nodes=r.n, closure_residual_m=r.r_level,
                             gauge_level_m=r.r_detrended, satellite_level_m=r.r_daily_change,
                             icesat_level_m=r.n_eff, included_primary=True,
                             exclusion_reason="columns: closure_residual_m=r level, gauge_level_m=r detrended, "
                                              "satellite_level_m=r daily change, icesat_level_m=n_eff"))
    res = R[(R.a != "Kherson") & (R.b != "Kherson")]
    ends = R[R.a.isin(["Plavni", "Rozumivka"]) & R.b.isin(["Velyka Lepetykha", "Nova Kakhovka"])]
    mid = R[(R.a == "Blahovishchenka") & (R.b == "Nikopol")]
    for k, v in (("reservoir_level_r_min", res.r_level.min()), ("reservoir_level_r_max", res.r_level.max()),
                 ("reservoir_n_eff_min", res.n_eff.min()), ("reservoir_n_eff_max", res.n_eff.max()),
                 ("reservoir_n_days_max", res.n.max()),
                 ("ends_daily_change_r_min", ends.r_daily_change.min()),
                 ("ends_daily_change_r_max", ends.r_daily_change.max()),
                 ("mid_pair_daily_change_r", mid.r_daily_change.iloc[0])):
        summary.append(dict(claim_id=C, statistic=k, value=float(v), unit="gauge pair", note="2019-2021"))
    p = g.pivot_table(index="date", columns="name_en", values="H_evrf2019_m")[["Rozumivka", "Kherson"]]
    for tag, t0 in (("from 2023-06-06", BREACH), ("from 2023-07-01", pd.Timestamp("2023-07-01"))):
        q = p[p.index >= t0].diff().dropna()
        summary.append(dict(claim_id=C, statistic="post_rozumivka_kherson_daily_change_r", variant=tag,
                            value=float(q.Rozumivka.corr(q.Kherson)), n=len(q), unit="day pair"))
    R.to_csv(OUT / "ms7_gauge_network_pairs.csv", index=False)


def frame_check(x: pd.DataFrame, atl: pd.DataFrame, nodes: pd.DataFrame, summary: list) -> None:
    """Does the crossing difference depend on the reference surface?

    Primary (the algebra of Section S1): put ATL13 on SWOT's own geoid at the
    node. Since the same N_node is then subtracted from both sensors, that is
    the ellipsoidal difference (SWOT wse + geoid_hght vs ATL13 h + free2mean),
    and it differs from the EGG2015 difference only by zeta_node - zeta_segment
    over <= 200 m: millimetres.
    Caution variant: ATL13's OWN EGM2008 height (H_egm2008). ATL03/ATL13 carry
    the geoid in the tide-free system, SWOT's geoid_hght is not, so mixing the
    two products' geoid heights moves the difference by centimetres. That is a
    reason never to difference geoid-referenced heights across products."""
    C = "V4_SWOT_ICESAT_DIRECT"
    a0 = atl[atl.ovp.isin(x.ovp.unique())]
    n = nodes[(nodes.node_q <= XS.NODE_Q_MAX) & (nodes.dark_frac < XS.DARK_MAX)].copy()
    variants = {}
    a = a0.copy()
    a["H"] = a.h_wgs84_m + CFG.free2mean(a.lat.values)
    ne = n.copy()
    ne["H"] = ne.wse + ne.geoid_hght
    variants["SWOT geoid at the node for both sensors (ellipsoidal difference)"] = (a, ne)
    cols = ["time", "beam", "segment_id", "H_egm2008_m"]
    egm = pd.concat([pd.read_parquet(f, columns=cols) for f in XS.ATL13], ignore_index=True)
    egm = egm.drop_duplicates(["time", "beam", "segment_id"])
    b = a0.merge(egm, on=["time", "beam", "segment_id"], how="left")
    b["H"] = b.H_egm2008_m + CFG.free2mean(b.lat.values)
    nw = n.copy()
    nw["H"] = nw.wse
    variants["each product's own EGM2008 height (caution: mixed tide systems)"] = (
        b.dropna(subset=["H"]), nw)
    for name, (aa, nn) in variants.items():
        e = XS.group_crossings(XS.node_pairs(aa, nn))
        m = x.merge(e[["ovp", "swot_pass", "d_m"]], left_on=["ovp", "pass_id"],
                    right_on=["ovp", "swot_pass"], suffixes=("", "_alt"))
        dd = (m.d_m_alt - m.d_m).abs()
        for k, v in (("frame_change_n", len(m)), ("frame_change_median_abs_m", dd.median()),
                     ("frame_change_max_abs_m", dd.max())):
            summary.append(dict(claim_id=C, statistic=k, value=v, unit="crossing, all windows",
                                variant=name, note="vs EGG2015 for both sensors"))


def v5_lakes(evidence: list, summary: list) -> None:
    """LakeSP as a separate, product-specific test (never pooled with RiverSP):
    lake wse against the median ATL13 height inside the observed polygon, same
    day (|dt| <= 24 h), one processing version per granule."""
    C = "V5_LAKESP_PRODUCT"
    G = XC.geoms()
    atl, _, lakes, _ = XC.load_cached(G)
    u = unary_union([SD.load_utm(n) for n in XS.PAPER_ZONES.values()])
    lk = XS.latest_version(lakes).reset_index(drop=True)
    lk = lk[lk.intersects(u)].reset_index(names="lake_row")
    lk["t"] = pd.to_datetime(lk.t, utc=True)
    pts = gpd.GeoDataFrame(atl[["ovp", "time", "H"]],
                           geometry=gpd.points_from_xy(atl.x, atl.y), crs=CFG.CRS_METRIC)
    j = gpd.sjoin(pts, lk[["lake_row", "obs_id", "lake_id", "t", "H", "area_total", "geometry"]],
                  predicate="within", lsuffix="a", rsuffix="s")
    j["dt"] = (j.t - pd.to_datetime(j.time, utc=True)).dt.total_seconds() / 86400.0
    j = j[j.dt.abs() <= 1]
    g = j.groupby(["ovp", "lake_row"]).agg(H_a=("H_a", "median"), H_s=("H_s", "first"),
                                          rng=("H_a", lambda v: np.percentile(v, 95) - np.percentile(v, 5)),
                                          n_seg=("H_a", "size"), area=("area_total", "first"),
                                          lake_id=("lake_id", "first"), dt_d=("dt", "median"),
                                          time=("time", "first")).reset_index()
    g["d"] = g.H_s - g.H_a
    g["size"] = np.where(g.area <= XS.LAKE_SMALL_KM2, "small", "large")
    for _, r in g.iterrows():
        evidence.append(dict(claim_id=C, validation_path="V5", sensor="SWOT LakeSP - ATL13",
                             overpass_id=r.ovp, swot_pass_id=f"lake {r.lake_id}",
                             date=pd.Timestamp(r.time).date().isoformat(), dt_hours=24 * r.dt_d,
                             n_nodes=r.n_seg, satellite_level_m=r.H_s, icesat_level_m=r.H_a,
                             closure_residual_m=r.d, zone=r["size"],
                             included_primary=r["size"] == "small",
                             exclusion_reason="" if r["size"] == "small" else
                             f"lake > {XS.LAKE_SMALL_KM2:g} km2: a lake-averaged level of a sloping body is not a test"))
    for sz in ("small", "large"):
        h = g[g["size"] == sz]
        summary += rows_stats(C, h.d, "lake x overpass", variant=f"{sz} lakes")
        summary.append(dict(claim_id=C, statistic="icesat_p95_p05_inside_polygon_median_m",
                            value=h.rng.median(), unit="lake x overpass", variant=f"{sz} lakes"))


def figure_v4(y: pd.DataFrame, a: pd.DataFrame, S: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"R": "#5b8fa8", "F": "#c1402a", "D": "#3f7d4e", "E": "#7a4f9e"}
    s4 = S[S.claim_id == "V4_SWOT_ICESAT_DIRECT"]
    v = s4[s4.variant.isna()].set_index("statistic").value
    w = s4[s4.variant.fillna("").str.startswith("excluding 2023")].set_index("statistic").value
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    it = pd.to_datetime(y.icesat_time, utc=True).dt.tz_convert(None)
    y = y.assign(breach=it.between(BREACH, BREACH + pd.Timedelta(days=14)).values)
    for zn, g in y.groupby("zone"):
        c0 = g[~g.breach]
        ax[0].scatter(c0.H_atl, c0.H_swot, s=40, c=col.get(zn, "k"), ec="k", lw=0.5, label=zn)
        c1 = g[g.breach]
        ax[0].scatter(c1.H_atl, c1.H_swot, s=55, fc="none", ec=col.get(zn, "k"), lw=1.6)
    ax[0].scatter([], [], s=55, fc="none", ec="k", lw=1.6, label="6–20 Jun 2023")
    lim = [min(y.H_atl.min(), y.H_swot.min()) - 0.5, max(y.H_atl.max(), y.H_swot.max()) + 0.5]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlim=lim, ylim=lim, xlabel="ICESat-2 ATL13, EGG2015 (m)",
              ylabel="SWOT RiverSP, EGG2015 (m)", title="(a) absolute heights")
    ax[0].text(0.03, 0.97,
               f"n = {int(v['n'])}\nmedian Δ = {100 * v['median_m']:+.1f} cm\n"
               f"NMAD = {100 * v['nmad_m']:.1f} cm\nRMSE = {100 * v['rmse_m']:.1f} cm\n"
               f"r = {v['pearson_r_raw']:.3f}, ρ = {v['spearman_rho_raw']:.3f}\n"
               f"Theil–Sen β = {v['theil_sen_slope']:.3f}\n"
               f"without breach fortnight (n = {int(w['n'])}):\n"
               f"  NMAD {100 * w['nmad_m']:.1f} cm, RMSE {100 * w['rmse_m']:.1f} cm",
               transform=ax[0].transAxes, va="top", fontsize=8.5)
    ax[0].legend(title="zone", fontsize=8, loc="lower right")
    ita = pd.to_datetime(a.icesat_time, utc=True).dt.tz_convert(None)
    a = a.assign(breach=ita.between(BREACH, BREACH + pd.Timedelta(days=14)).values)
    for zn, g in a.groupby("zone"):
        c0, c1 = g[~g.breach], g[g.breach]
        ax[1].scatter(c0.Aa, c0.As, s=40, c=col.get(zn, "k"), ec="k", lw=0.5, label=zn)
        ax[1].scatter(c1.Aa, c1.As, s=55, fc="none", ec=col.get(zn, "k"), lw=1.6)
    m = max(abs(a.Aa).max(), abs(a.As).max()) * 1.1
    ax[1].plot([-m, m], [-m, m], "k--", lw=0.8)
    ax[1].set(xlim=(-m, m), ylim=(-m, m), xlabel="ICESat-2 anomaly (m)",
              ylabel="SWOT anomaly (m)",
              title="(b) anomalies from the zone median")
    ax[1].text(0.03, 0.97, f"n = {int(v['n_anomaly'])} (zones n ≥ {V4_MIN_N_ANOM})\n"
               f"r = {v['pearson_r_anomaly']:.3f}, ρ = {v['spearman_rho_anomaly']:.3f}",
               transform=ax[1].transAxes, va="top", fontsize=8.5)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"FS_V4_swot_icesat_agreement.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def station_inventory() -> pd.DataFrame:
    g = gauge_levels()
    inv = g.groupby(["station_id", "name_en"]).agg(first=("date", "min"), last=("date", "max"),
                                                   n_days=("date", "size")).reset_index()
    inv["n_days_post_breach"] = inv.station_id.map(
        g[g.date >= BREACH].groupby("station_id").size()).fillna(0).astype(int)
    inv["spans_breach"] = inv["last"] >= BREACH
    return inv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev, sm = [], []
    v1x = v1(ev, sm)
    v2(ev, sm)
    v3(v1x, ev, sm)
    y, a = v4(ev, sm)
    v5_lakes(ev, sm)
    v6(ev, sm)
    v8_reference_surfaces(sm, ev)
    v7_downstream_posts(sm, ev)
    gauge_network(sm, ev)
    E = pd.DataFrame(ev)
    S = pd.DataFrame(sm)
    S["v5_value"] = [V5.get((c, s)) for c, s in zip(S.claim_id, S.statistic)]
    if "series" in S:
        for lab, (med, n) in V5_KHERSON.items():
            k = (S.claim_id == "V6_KHERSON_CLOSURE") & (S.series == lab)
            S.loc[k & (S.statistic == "median_m"), "v5_value"] = med
            S.loc[k & (S.statistic == "n"), "v5_value"] = n
    E.to_csv(OUT / "ms7_evidence.csv", index=False)
    S.to_csv(OUT / "ms7_summary.csv", index=False)
    inv = station_inventory()
    inv.to_csv(OUT / "ms7_station_inventory.csv", index=False)
    figure_v4(y, a, S)
    with pd.option_context("display.width", 220, "display.max_rows", 300,
                           "display.max_colwidth", 60):
        print(S.drop(columns=[c for c in ("unit",) if c in S]).round(4).to_string(index=False))
        print(inv.to_string(index=False))


if __name__ == "__main__":
    main()
