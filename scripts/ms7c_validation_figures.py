#!/usr/bin/env python
"""MS7C — Figure 5 (F16) and Figure S6 (F24), drawn only from ms7's tables.

The v5 versions of both figures printed numbers that no longer match the
reproducible validation (Rozumivka SWOT n = 68, c = -15.4 / -14.9 cm; an
ICESat-2 c of -9.8 cm on a panel whose text said -10.7). Here every point,
interval and annotation is read from outputs/paper/validation/ms7_evidence.csv
and ms7_summary.csv; nothing is typed in. The only other input is the raw
daily Rozumivka gauge series for the bottom panels of F24.

F16  closure c = gauge - satellite at the control points, by sensor and period,
     median with the 95 % bootstrap interval over independent units (V1 per
     reservoir gauge, V3 SWOT at Rozumivka, V6 at Kherson).
F24  satellite (EGG2015, mean-tide crust) against the same-date gauge
     (EVRF2019) per station (V2 overpasses, V3 SWOT passes, V6 Kherson), and
     Rozumivka 2019-2025 twice: raw EGG2015-referenced satellite heights, and
     the same heights shifted by the PRE-BREACH ICESat-2 closure c_IS2 only.
     Post-breach SWOT plays no part in estimating c_IS2: the lower panel is an
     out-of-period, cross-sensor transfer test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VAL = ROOT / "outputs/paper/validation"
FIG = ROOT / "outputs/paper/figures"
GAUGES = ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet"
COL = {"ICESat-2": "#2a73c9", "SWOT RiverSP": "#e8643a", "SWOT PIXC": "#1aa37a"}
RES_ORDER = ["Plavni", "Rozumivka", "Blahovishchenka", "Nikopol", "Velyka Lepetykha",
             "Nova Kakhovka"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load():
    return (pd.read_csv(VAL / "ms7_evidence.csv", low_memory=False),
            pd.read_csv(VAL / "ms7_summary.csv"))


def _stat(S, **kw):
    s = S
    for k, v in kw.items():
        s = s[s[k].isna()] if v is None else s[s[k] == v]
    assert len(s) == 1, (kw, len(s))
    return s.iloc[0]


# ------------------------------------------------------------------- F16 --
def f16(E, S):
    rows = []
    for st in RES_ORDER:
        r = _stat(S, claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="station_median_c_m", station=st)
        rows.append((f"{st} · ICESat-2 · pre-breach · reported radius", "ICESat-2",
                     r.value, r.ci_lo, r.ci_hi, int(r.n)))
        if st == "Rozumivka":
            kw = dict(claim_id="V3_ROZUMIVKA_TRANSFER", sensor="SWOT", radius_km=3.0,
                      variant="raw_median")
            m, n = _stat(S, statistic="median_m", **kw), _stat(S, statistic="n", **kw)
            rows.append(("Rozumivka · SWOT RiverSP · post-breach · ≤3 km", "SWOT RiverSP",
                         m.value, m.ci_lo, m.ci_hi, int(n.value)))
    for series, sensor, lab in (
            ("ICESat-2 pre_breach", "ICESat-2", "Kherson · ICESat-2 · pre-breach · ≤10 km"),
            ("ICESat-2 post_breach", "ICESat-2", "Kherson · ICESat-2 · post-breach · ≤10 km"),
            ("SWOT RiverSP pre_breach", "SWOT RiverSP", "Kherson · SWOT RiverSP · pre-breach · ≤3 km"),
            ("SWOT RiverSP post_breach", "SWOT RiverSP", "Kherson · SWOT RiverSP · post-breach · ≤3 km"),
            ("SWOT PIXC pre_breach", "SWOT PIXC", "Kherson · SWOT PIXC · pre-breach · 1 km")):
        kw = dict(claim_id="V6_KHERSON_CLOSURE", series=series)
        m, n = _stat(S, statistic="median_m", **kw), _stat(S, statistic="n", **kw)
        rows.append((lab, sensor, m.value, m.ci_lo, m.ci_hi, int(n.value)))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, sen, v, lo, hi, n) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], c=COL[sen], lw=2.2)
        ax.scatter(v, yi, s=45, c=COL[sen], ec="w", zorder=3)
        ax.text(1.005, yi, f"n={n}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_xlabel("empirical local vertical closure residual c = H_gauge(EVRF2019) − "
                  "H_sat(EGG2015, mean-tide crust), m (median, 95 % bootstrap CI)")
    for sen, c in COL.items():
        ax.plot([], [], c=c, lw=2.2, marker="o", label=sen)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F16_closure_offsets.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F24 --
def _panel(ax, pts, title):
    """Co-variability only: satellite against gauge, r / rho / n per series and
    its Theil-Sen line. Closures are Figure 5's subject, not this one's."""
    from scipy import stats
    txt = []
    for sensor, lab, g in pts:
        ax.scatter(g.gauge_level_m, g.satellite_level_m, s=18, c=COL[sensor], ec="w", lw=0.4,
                   label=f"{lab} (n={len(g)})", marker="D" if "SWOT" in sensor else "o")
        if len(g) >= 3:
            x, y = g.gauge_level_m.values, g.satellite_level_m.values
            b, a, *_ = stats.theilslopes(y, x)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, a + b * xs, c=COL[sensor], lw=0.9)
            txt.append(f"{lab}: r {stats.pearsonr(x, y)[0]:.2f}, "
                       f"ρ {stats.spearmanr(x, y)[0]:.2f}, n {len(g)}")
        else:
            txt.append(f"{lab}: n {len(g)} (too few for r)")
    ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=6.5,
            bbox=dict(fc="w", ec="0.8", lw=0.5))
    ax.set_title(title, loc="left", fontsize=9, weight="bold")
    ax.set_xlabel("gauge, m EVRF2019", fontsize=8)
    ax.set_ylabel("satellite, m EGG2015", fontsize=8)
    ax.legend(fontsize=6, loc="lower right", frameon=False)


def f24(E, S):
    inc = E[E.included_primary.astype(bool)]
    v2 = inc[inc.claim_id == "V2_ATL13_GAUGE_COVARIABILITY"]
    v3 = inc[inc.claim_id == "V3_ROZUMIVKA_TRANSFER"]
    v6 = inc[inc.claim_id == "V6_KHERSON_CLOSURE"]
    v1r = inc[(inc.claim_id == "V1_ATL13_GAUGE_CLOSURE") & (inc.station == "Rozumivka")]

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.8, 0.8], hspace=0.45, wspace=0.28)
    order = RES_ORDER + ["Kherson"]
    for i, st in enumerate(order):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if st == "Kherson":
            pts = [("ICESat-2" if "ICESat" in s else ("SWOT PIXC" if "PIXC" in s else "SWOT RiverSP"),
                    s.replace("_breach", "").replace("pre", "pre").replace("post", "post"), g)
                   for s, g in v6.groupby("series", sort=False)]
        else:
            pts = [("ICESat-2", "ICESat-2 pre", v2[v2.station == st])]
            if st == "Rozumivka":
                pts.append(("SWOT RiverSP", "SWOT post ≤3 km", v3))
        _panel(ax, pts, st)
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    c_is2 = _stat(S, claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_m", sensor="ICESat-2").value
    ax.text(0, 1, "Each panel: satellite (EGG2015, mean-tide crust) against the same-date\n"
                  "gauge (EVRF2019), with r, ρ, n and the Theil-Sen line per series.\n"
                  "Reservoir: ICESat-2 overpasses ≤20 km, nearest gauge (V2).\n"
                  "Rozumivka SWOT: RiverSP passes, nodes ≤3 km, node_q ≤1 (V3).\n"
                  "Kherson: V6 series, each with its own support.\n"
                  "Closure residuals c: Figure 5. All values from ms7_evidence.csv.", va="top", fontsize=8)

    g = pd.read_parquet(GAUGES)
    g = g[g.station_id == 80959]
    g["date"] = pd.to_datetime(g.date)
    for k, (shift, title) in enumerate((
            (0.0, "Rozumivka 2019–2025, raw satellite heights (EGG2015, mean-tide crust)"),
            (c_is2, f"Rozumivka 2019–2025, satellite heights + pre-breach ICESat-2 closure "
                    f"c_IS2 = {100 * c_is2:+.1f} cm (post-breach SWOT not used to estimate it)"))):
        ax = fig.add_subplot(gs[2 + k, :])
        ax.plot(g.date, g.H_evrf2019_m, c="0.25", lw=0.8, label="Rozumivka gauge 80959, daily (EVRF2019)")
        d1 = pd.to_datetime(v1r.date)
        ax.scatter(d1, v1r.satellite_level_m + shift, s=18, c=COL["ICESat-2"], zorder=3,
                   label=f"ICESat-2, pre-breach (n={len(v1r)})")
        d3 = pd.to_datetime(v3.date)
        ax.scatter(d3, v3.satellite_level_m + shift, s=18, marker="D", c=COL["SWOT RiverSP"],
                   zorder=3, label=f"SWOT RiverSP, post-breach, ≤3 km (n={len(v3)})")
        ax.axvline(pd.Timestamp("2023-06-06"), c="grey", ls=":", lw=0.8)
        ax.set_title(title, loc="left", fontsize=9, weight="bold")
        ax.set_ylabel("m")
        ax.legend(fontsize=7, loc="lower left", ncol=3, frameon=False)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F24_station_panels.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F19 --
def f19(E, S):
    """SWOT - ICESat-2 against distance from the dam along SWORD, primary crossings."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import ms6c_crossing_provenance as XC
    from swot_dnipro import config as CFG
    from swot_dnipro.vertical import haversine_km
    v4 = E[(E.claim_id == "V4_SWOT_ICESAT_DIRECT") & (E.validation_path == "V4")
           & E.included_primary.astype(bool)].copy()
    n = pd.read_parquet(XC.CACHE / "nodes_all.parquet", columns=["lon", "lat", "p_dist_out"])
    dam = n.iloc[np.argmin(haversine_km(n.lon.values, n.lat.values, *CFG.KAKHOVKA_DAM))]
    v4["s_km"] = (v4.sword_dist_out_m - dam.p_dist_out) / -1000.0
    it = pd.to_datetime(v4.date)
    per = np.where(it < pd.Timestamp("2023-06-06"), "PRE_BREACH",
                   np.where(it < pd.Timestamp("2023-09-01"), "BREACH_DRAWDOWN", "POST_BREACH"))
    cols = {"PRE_BREACH": "#1aa37a", "BREACH_DRAWDOWN": "#e8643a", "POST_BREACH": "#2a73c9"}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    out = int((v4.closure_residual_m.abs() > 0.5).sum())
    for p_ in cols:
        g = v4[per == p_]
        ax.scatter(g.s_km, g.closure_residual_m.clip(-0.5, 0.5), s=28, c=cols[p_], ec="w",
                   lw=0.4, label=f"{p_} (n={len(g)})")
    ax.axhline(0, c="grey", lw=0.8)
    ax.axvline(0, c="grey", ls=":", lw=1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("SWORD distance from the dam, km (negative = upstream, former reservoir)")
    ax.set_ylabel("SWOT − ICESat-2, m")
    ax.set_title(f"SWOT RiverSP − ICESat-2, both EGG2015, |Δt| ≤ 24 h, R/F/D/E"
                 + (f"; {out} point(s) beyond ±0.5 m drawn at the edge" if out else ""),
                 loc="left", fontsize=10, weight="bold")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F19_swot_icesat_egg2015_along_system.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F20 --
def f20(E, S):
    g = E[E.validation_path == "V4-geoid"].closure_residual_m * 100
    exp_ = 100 * _stat(S, claim_id="V4_SWOT_ICESAT_DIRECT",
                       statistic="geoid_diff_expected_if_atl13_tide_free_m").value
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(wspace=0.75))
    ax[0].hist(g.clip(-10, 10), bins=80, color="#2a73c9")
    ax[0].axvline(exp_, c="#e8643a", lw=2, label=f"expected if the ATL13 geoid were tide-free: {exp_:+.1f} cm")
    ax[0].axvline(g.median(), c="k", ls="--", label=f"observed median {g.median():+.1f} cm")
    ax[0].set(xlabel="SWOT geoid_hght − ATL13 geoid (ht_water_surf − ht_ortho), cm", ylabel="nodes",
              title=f"(a) the two products' geoids at the same nodes (n = {len(g)})")
    ax[0].legend(fontsize=8, frameon=False)
    names = ["production: EGG2015, ATL13 + mean-tide term", "ATL13 permanent-tide term omitted",
             "ATL13 permanent-tide term with the opposite sign", "naive product difference wse - ht_ortho"]
    for i, nm in enumerate(names):
        r = _stat(S, claim_id="V4_SWOT_ICESAT_DIRECT", statistic="median_m", variant=f"chain: {nm}")
        q = _stat(S, claim_id="V4_SWOT_ICESAT_DIRECT", statistic="nmad_m", variant=f"chain: {nm}")
        y = len(names) - i
        ax[1].plot([100 * r.ci_lo, 100 * r.ci_hi], [y, y], c="#2a73c9", lw=2)
        ax[1].scatter(100 * r.value, y, s=60, c="#2a73c9", zorder=3)
        ax[1].text(100 * r.value, y + 0.18, f"{100 * r.value:+.1f} cm (NMAD {100 * q.value:.1f})",
                   ha="center", fontsize=8)
    ax[1].set_yticks(range(len(names), 0, -1), [n.replace("naive product difference ", "naive: ")
                                                for n in names], fontsize=8)
    ax[1].axvline(0, c="grey", lw=0.8)
    ax[1].set_ylim(0.5, len(names) + 0.6)
    ax[1].set(xlabel="median SWOT − ICESat-2, cm (95 % bootstrap CI over the 28 crossings)",
              title="(b) which vertical chain closes the gap (|Δt| ≤ 24 h)")
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F20_vertical_chains.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F21 --
def f21(E, S):
    from scipy import stats
    inc = E[E.included_primary.astype(bool)]
    v1 = inc[inc.claim_id == "V1_ATL13_GAUGE_CLOSURE"]
    sd = v1.groupby(["station", "date"]).agg(G=("gauge_level_m", "median"),
                                             H=("satellite_level_m", "median")).reset_index()
    sd["Ga"] = sd.G - sd.groupby("station").G.transform("mean")
    sd["Ha"] = sd.H - sd.groupby("station").H.transform("mean")
    v2 = inc[inc.claim_id == "V2_ATL13_GAUGE_COVARIABILITY"].copy()
    v2["Ga"] = v2.gauge_level_m - v2.groupby("station").gauge_level_m.transform("mean")
    v2["Ha"] = v2.satellite_level_m - v2.groupby("station").satellite_level_m.transform("mean")
    v6 = inc[inc.claim_id == "V6_KHERSON_CLOSURE"]
    V1, V2, V6 = "V1_ATL13_GAUGE_CLOSURE", "V2_ATL13_GAUGE_COVARIABILITY", "V6_KHERSON_CLOSURE"

    def ci_of(**kw):
        r = _stat(S, **kw)
        return r.ci_lo, r.ci_hi

    panels = [
        ("(a) closure matchups, one point per station-date — absolute", sd.G, sd.H, None,
         "station-dates", "ATL13"),
        ("(b) the same, within-station anomalies (V1 matchup set)", sd.Ga, sd.Ha,
         ci_of(claim_id=V1, statistic="matchup_set_pearson_r"), "station-dates", "ATL13"),
        ("(c) ≤20 km, nearest gauge, within-station anomalies (V2)", v2.Ga, v2.Ha,
         ci_of(claim_id=V2, statistic="pearson_r"), "overpasses", "ATL13"),
    ]
    for series, title, unit, sen in (("ICESat-2 pre_breach", "(d) ATL13 vs Kherson, pre-breach ≤10 km", "overpasses", "ATL13"),
                                     ("ICESat-2 post_breach", "(e) ATL13 vs Kherson, post-breach ≤10 km", "overpasses", "ATL13"),
                                     ("SWOT PIXC pre_breach", "(f) SWOT PIXC vs Kherson, 1 km", "passes", "SWOT")):
        g = v6[v6.series == series]
        panels.append((title, g.gauge_level_m, g.satellite_level_m,
                       ci_of(claim_id=V6, statistic="covar_pearson_r_ci", series=series), unit, sen))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (title, x, y, ci, unit, sen) in zip(axs.ravel(), panels):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ax.scatter(x, y, s=18, c="#e8643a" if sen == "SWOT" else "#2a73c9", ec="w", lw=0.4)
        b, a, *_ = stats.theilslopes(y, x)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, a + b * xs, c="0.2", lw=1, label=f"Theil–Sen slope {b:.2f}")
        r = stats.pearsonr(x, y)[0]
        rho = stats.spearmanr(x, y)[0]
        cis = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci and np.isfinite(ci[0]) else ""
        ax.text(0.03, 0.97, f"r = {r:.2f}{cis}\nρ = {rho:.2f}\nn = {len(x)} {unit}",
                transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="w", ec="0.8", lw=0.5))
        ax.set_title(title, loc="left", fontsize=9, weight="bold")
        ax.set_xlabel("gauge, m EVRF2019" if "anomal" not in title else "gauge anomaly, m", fontsize=8)
        ax.set_ylabel(f"{sen}, m EGG2015" if "anomal" not in title else f"{sen} anomaly, m", fontsize=8)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("Satellite against gauge; r with 95 % bootstrap interval where computed in ms7",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F21_covariability_scatter.{e}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F17 --
def f17(E, S):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import rasterio
    from swot_dnipro import config as CFG
    from swot_dnipro import spatial_domains as SD
    from swot_dnipro.vertical import sample_grid
    import geopandas as gpd
    v8 = E[E.claim_id == "V8_REFERENCE_SURFACES"]
    pts = gpd.GeoSeries(gpd.points_from_xy(v8.lon, v8.lat), crs=4326).to_crs(CFG.CRS_METRIC)
    zones = {L: SD.load_utm(n) for L, n in (("R", "R_FORMER_KAKHOVKA_RESERVOIR"), ("F", "F_LOWER_DNIPRO_FLOODWAY"),
                                             ("D", "D_KHERSON_DELTA"), ("E", "E_DNIPRO_BUG_ESTUARY"))}
    ZC = {"R": "#5b8fa8", "F": "#c1402a", "D": "#3f7d4e", "E": "#7a4f9e"}
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.2))
    sc = ax[0].scatter(pts.x, pts.y, c=100 * v8.closure_residual_m, s=5, cmap="Blues")
    fig.colorbar(sc, ax=ax[0], label="N_EGM2008 (SWOT geoid_hght) − ζ_EGG2015, cm")
    with rasterio.open(CFG.UA2019Z_ASC) as src:
        arr = src.read(1, masked=True)
        b = src.bounds
    b4 = gpd.GeoSeries([__import__("shapely").geometry.box(b.left, b.bottom, b.right, b.top)], crs=4326)
    im = ax[1].imshow(arr, extent=(b.left, b.right, b.bottom, b.top), cmap="Blues", origin="upper")
    fig.colorbar(im, ax=ax[1], label="Δ9902 (BS-77 → EVRF2019), m")
    st = [(n, lo, la) for _, n, lo, la in CFG.RESERVOIR_GAUGES] + [CFG.KHERSON_GAUGE[1:]]
    val = sample_grid(CFG.UA2019Z_ASC, [x[1] for x in st], [x[2] for x in st])
    for (n, lo, la), v in zip(st, val):
        ax[1].scatter(lo, la, s=30, marker="s", c="k")
        ax[1].annotate(f"{n} {v:.3f}", (lo, la), xytext=(4, 4), textcoords="offset points", fontsize=7)
    zb = gpd.GeoSeries(list(zones.values()), crs=CFG.CRS_METRIC)
    for (L, g), a in ((z, None) for z in zones.items()):
        gpd.GeoSeries([g], crs=CFG.CRS_METRIC).boundary.plot(ax=ax[0], color=ZC[L], lw=0.9)
        gpd.GeoSeries([g], crs=CFG.CRS_METRIC).to_crs(4326).boundary.plot(ax=ax[1], color=ZC[L], lw=0.9)
    x0, y0, x1, y1 = zb.total_bounds
    ax[0].set(xlim=(x0 - 5e3, x1 + 5e3), ylim=(y0 - 5e3, y1 + 5e3), xlabel="UTM 36N easting, m",
              ylabel="northing, m", title=f"(a) EGM2008 − EGG2015 at {len(v8)} good SWOT node locations, R/F/D/E")
    lo0, la0, lo1, la1 = zb.to_crs(4326).total_bounds
    ax[1].set(xlim=(lo0 - 0.1, lo1 + 0.9), ylim=(la0 - 0.1, la1 + 0.1), xlabel="°E", ylabel="°N",
              title="(b) the EPSG:9902 grid (ua_2019z.asc) and its value at each gauge")
    ax[0].ticklabel_format(style="plain")
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F17_reference_surfaces.{e}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F22 --
def f22(E, S):
    R = pd.read_csv(VAL / "ms7_gauge_network_pairs.csv")
    order = ["Plavni", "Rozumivka", "Blahovishchenka", "Nikopol", "Velyka Lepetykha", "Nova Kakhovka", "Kherson"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for a, (col, title, cmap, vmin) in zip(ax, (("r_level", "raw daily levels", "Blues", 0),
                                               ("r_detrended", "linearly detrended", "Blues", 0),
                                               ("r_daily_change", "first differences (ΔH/day)", "RdBu_r", -1))):
        M = pd.DataFrame(np.nan, index=order, columns=order)
        N = M.copy()
        for _, r in R.iterrows():
            M.loc[r.a, r.b] = M.loc[r.b, r.a] = r[col]
            N.loc[r.a, r.b] = N.loc[r.b, r.a] = r.n if col != "r_daily_change" else r.n_change
        im = a.imshow(M.values, cmap=cmap, vmin=vmin, vmax=1)
        for i in range(len(order)):
            for j in range(len(order)):
                if i != j:
                    a.text(j, i, f"{M.values[i, j]:.2f}\nn{int(N.values[i, j])}", ha="center", va="center", fontsize=6.5,
                           color="w" if abs(M.values[i, j]) > 0.7 else "k")
        a.set_xticks(range(len(order)), [o.replace("Velyka ", "V. ").replace("Blahovishchenka", "Blahovish.") for o in order],
                     rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(len(order)), [o.replace("Velyka ", "V. ").replace("Blahovishchenka", "Blahovish.") for o in order],
                     fontsize=8)
        a.set_title(title, fontsize=10, weight="bold")
        fig.colorbar(im, ax=a, fraction=0.046, label="Pearson r")
    fig.suptitle("Gauge–gauge correlation, 2019–2021, pairwise-complete common dates; upstream → downstream",
                 x=0.01, ha="left", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F22_gauge_network_correlation.{e}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- F23 --
def f23(E, S):
    T = pd.read_csv(VAL / "ms7b_slope_frames.csv")
    F = pd.read_csv(VAL / "ms7b_slope_frames_summary.csv")
    g = _stat(S, claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="gradient_along_reach_cm_per_km")
    frames = ["production", "egg2015_no_tide", "plus_const_c", "plus_c_nearest", "plus_c_linear"]
    col = {"egg2015_no_tide": "#1aa37a", "plus_const_c": "#2a73c9", "plus_c_nearest": "#e8643a",
           "plus_c_linear": "#7a4f9e"}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for k in frames[1:]:
        m = F[(F.variant == k) & (F.statistic == "max_abs_dS_cm_per_km")].value.iloc[0]
        ax[0].scatter(T.S_production, T[f"S_{k}"] - T.S_production, s=22, c=col[k],
                      label=f"{k}: max |ΔS| {m:.3g} cm/km")
    ax[0].axhline(0, c="grey", lw=0.8)
    ax[0].set(xlabel="production per-overpass Theil–Sen slope, cm/km", ylabel="S_frame − S_production, cm/km",
              title=f"(a) per-overpass slope change by vertical frame ({len(T)} overpasses)")
    ax[0].legend(fontsize=8, frameon=False)
    for i, k in enumerate(frames):
        r = F[(F.variant == k) & (F.statistic == "contrast_cm_per_km")].iloc[0]
        y = len(frames) - i
        ax[1].plot([r.ci_lo, r.ci_hi], [y, y], c="#2a73c9", lw=2)
        ax[1].scatter(r.value, y, s=50, c="#2a73c9", zorder=3)
        ax[1].text(r.value, y + 0.2, f"{r.value:+.3f}", ha="center", fontsize=8)
    ax[1].axvspan(g.ci_lo, g.ci_hi, color="#e8643a", alpha=0.25,
                  label=f"95 % interval of the c gradient along the reach [{g.ci_lo:+.3f}, {g.ci_hi:+.3f}] cm/km")
    ax[1].axvline(0, c="grey", lw=0.8)
    ax[1].set_ylim(0.4, len(frames) + 0.7)
    ax[1].set_yticks(range(len(frames), 0, -1), frames)
    ax[1].set(xlabel="post − pre median slope, cm/km (95 % bootstrap CI)",
              title="(b) the headline contrast in every vertical frame")
    ax[1].legend(fontsize=8, loc="lower right", frameon=False)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F23_slope_frame_invariance.{e}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- F25, F26 --
def _gauge_2023():
    lv = pd.read_csv(ROOT / "data/historical/sea_posts_2023/daily_levels_cm.csv")
    lv = lv[lv.variable == "water_level"].copy()
    lv["date"] = pd.to_datetime(lv.date)
    g = pd.read_parquet(GAUGES)
    g["date"] = pd.to_datetime(g.date)
    riv = g[(g.station_id == 80805) & (g.date.dt.year == 2023)].set_index("date").H_evrf2019_m
    return lv, riv


def _post_panel(ax, E, S, name, pid, lv, riv, xlim, shift_note=True):
    v7 = E[(E.claim_id == "V7_DOWNSTREAM_POSTS_2023") & (E.station == name)].copy()
    v7["date"] = pd.to_datetime(v7.date)
    c = _stat(S, claim_id="V7_DOWNSTREAM_POSTS_2023", statistic="c_pre_swot_m", station=name).value
    gs = lv[lv.post_id == pid].set_index("date").H_evrf2019_m.sort_index()
    gs = gs.reindex(pd.date_range("2023-01-01", "2023-12-31"))       # gaps break the line
    ax.plot(gs.index, gs.values, c="0.25", lw=0.9, marker="." if xlim else None, ms=3, label="gauge, sea yearbook")
    if name == "Kherson":
        rv = riv.reindex(pd.date_range("2023-01-01", "2023-12-31"))
        ax.plot(rv.index, rv.values, c="0.45", lw=0.9, ls="--", label="gauge, river yearbook")
    for sen, mk, col in (("ICESat-2 ATL13", "o", "#2a73c9"), ("SWOT RiverSP", "D", "#1aa37a")):
        q = v7[v7.sensor == sen]
        if xlim:                                     # count what the panel shows
            q = q[(q.date >= xlim[0]) & (q.date <= xlim[1])]
        ax.scatter(q.date, q.satellite_level_m + c, s=18, marker=mk, c=col, ec="w", lw=0.4, zorder=3,
                   label=f"{sen.split()[0]} {'RiverSP ' if 'SWOT' in sen else ''}+ c_pre (n={len(q)})")
    ax.axvspan(pd.Timestamp("2023-06-06"), pd.Timestamp("2023-06-30"), color="#f3d9cc", alpha=0.45, lw=0)
    ax.set_ylabel("m EVRF2019")
    ax.set_title(f"{name} — satellites + c_pre = {100 * c:+.1f} cm", loc="left", fontsize=9, weight="bold")
    if xlim:
        ax.set_xlim(*xlim)
        w = gs[xlim[0]:xlim[1]].dropna()
        if name == "Kherson":
            w = pd.concat([w, riv[xlim[0]:xlim[1]]])
        lo, hi = w.min() - 0.25 * (w.max() - w.min()), w.max() + 0.15 * (w.max() - w.min())
        ax.set_ylim(lo, hi)
        vv = v7[(v7.date >= xlim[0]) & (v7.date <= xlim[1])].satellite_level_m + c
        out = int(((vv < lo) | (vv > hi)).sum())
        if out:
            ax.text(0.99, 0.02, f"{out} satellite point(s) outside the axis", transform=ax.transAxes,
                    ha="right", fontsize=7, color="0.4")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    return v7


def f25(E, S):
    lv, riv = _gauge_2023()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for a, (name, pid) in zip(ax, (("Kherson", 80805), ("Parutyne", 98025), ("Mykolaiv", 98027))):
        _post_panel(a, E, S, name, pid, lv, riv, (pd.Timestamp("2023-05-18"), pd.Timestamp("2023-07-22")))
        a.tick_params(axis="x", rotation=30)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F25_breach_fortnight_posts.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f26(E, S):
    lv, riv = _gauge_2023()
    posts = (("Kherson", 80805), ("Kasperivka", 80807), ("Stanislav", 98032), ("Parutyne", 98025),
             ("Mykolaiv", 98027), ("Ochakiv", 98022))
    fig, ax = plt.subplots(len(posts), 1, figsize=(15, 3.1 * len(posts)), sharex=True)
    V7 = S[S.claim_id == "V7_DOWNSTREAM_POSTS_2023"]
    for a, (name, pid) in zip(ax, posts):
        v7 = _post_panel(a, E, S, name, pid, lv, riv, None)
        txt = []
        for (sen, per), g in V7[(V7.station == name) & V7.statistic.isin(["c_median_m"])].groupby(["sensor", "period"]):
            n_ = int(g.n.iloc[0])
            r_ = V7[(V7.station == name) & (V7.sensor == sen) & (V7.period == per) & (V7.statistic == "pearson_r")]
            nm = V7[(V7.station == name) & (V7.sensor == sen) & (V7.period == per) & (V7.statistic == "c_nmad_m")]
            txt.append(f"{sen.split()[0]} {per.replace('_', ' ')}: n {n_}, "
                       f"r {r_.value.iloc[0]:.2f}" if len(r_) else f"{sen.split()[0]} {per.replace('_', ' ')}: n {n_}, r –")
            txt[-1] += f", c {100 * g.value.iloc[0]:+.1f} cm, NMAD {100 * nm.value.iloc[0]:.1f} cm"
        a.text(1.01, 1, "\n".join(txt), transform=a.transAxes, va="top", fontsize=7)
        lim = v7.satellite_level_m.dropna()
        gs = lv[lv.post_id == pid].H_evrf2019_m
        lo, hi = np.nanpercentile(gs, 0.5) - 0.3, max(np.nanmax(gs), 0.5) + 0.2
        out = int(((v7.satellite_level_m + _stat(S, claim_id="V7_DOWNSTREAM_POSTS_2023", statistic="c_pre_swot_m",
                                                   station=name).value < lo) |
                   (v7.satellite_level_m + _stat(S, claim_id="V7_DOWNSTREAM_POSTS_2023", statistic="c_pre_swot_m",
                                                   station=name).value > hi)).sum())
        a.set_ylim(lo, hi)
        if out:
            a.text(0.99, 0.02, f"{out} satellite point(s) outside the axis", transform=a.transAxes,
                   ha="right", fontsize=7, color="0.4")
    ax[-1].set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
    fig.suptitle("Posts below the dam and on the limans, 2023: gauge, SWOT RiverSP (≤3 km) and ICESat-2 (≤10 km), "
                 "satellites + each post's pre-breach SWOT closure c_pre (sea yearbook); shaded: 6–30 June",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F26_downstream_posts_2023.{e}", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ F27 (main) --
def f27(E, S):
    """Main-text Figure 6: the satellites follow the water level through time.
    (a) V2 within-station anomalies, (b) V3 SWOT at Rozumivka after the breach,
    (c) V7 daily SWOT at the posts below the dam through the breach."""
    from scipy import stats
    plt.rcParams.update({"font.size": 12})
    inc = E[E.included_primary.astype(bool)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))

    v2 = inc[inc.claim_id == "V2_ATL13_GAUGE_COVARIABILITY"].copy()
    v2["Ga"] = v2.gauge_level_m - v2.groupby("station").gauge_level_m.transform("mean")
    v2["Ha"] = v2.satellite_level_m - v2.groupby("station").satellite_level_m.transform("mean")
    for st, g in v2.groupby("station"):
        ax[0].scatter(g.Ga, g.Ha, s=16, label=st, ec="w", lw=0.3)
    r = _stat(S, claim_id="V2_ATL13_GAUGE_COVARIABILITY", statistic="pearson_r")
    ts = _stat(S, claim_id="V2_ATL13_GAUGE_COVARIABILITY", statistic="theil_sen").value
    rho = _stat(S, claim_id="V2_ATL13_GAUGE_COVARIABILITY", statistic="spearman_rho").value
    m = float(np.abs(v2[["Ga", "Ha"]].values).max()) * 1.08
    ax[0].plot([-m, m], [-m, m], c="0.6", ls="--", lw=0.8)
    b, a, *_ = stats.theilslopes(v2.Ha, v2.Ga)
    ax[0].plot([-m, m], [a - b * m, a + b * m], c="k", lw=1)
    ax[0].text(0.03, 0.97, f"r = {r.value:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]\nρ = {rho:.2f}\n"
               f"Theil–Sen {ts:.2f}\nn = {len(v2)} overpasses", transform=ax[0].transAxes, va="top", fontsize=8.5,
               bbox=dict(fc="w", ec="0.8", lw=0.5))
    ax[0].set(xlim=(-m, m), ylim=(-m, m), xlabel="gauge anomaly, m", ylabel="ICESat-2 ATL13 anomaly, m",
              title="(a) ATL13 at the six reservoir gauges, before the breach")
    ax[0].legend(fontsize=6.5, loc="lower right", frameon=False)

    v3 = inc[inc.claim_id == "V3_ROZUMIVKA_TRANSFER"]
    kw = dict(claim_id="V3_ROZUMIVKA_TRANSFER", sensor="SWOT", radius_km=3.0, variant="raw_median")
    ax[1].scatter(v3.gauge_level_m, v3.satellite_level_m, s=18, c="#e8643a", marker="D", ec="w", lw=0.3)
    lo, hi = v3.gauge_level_m.min() - 0.2, v3.gauge_level_m.max() + 0.2
    ax[1].plot([lo, hi], [lo, hi], c="0.6", ls="--", lw=0.8, label="1:1")
    b, a, *_ = stats.theilslopes(v3.satellite_level_m, v3.gauge_level_m)
    ax[1].plot([lo, hi], [a + b * lo, a + b * hi], c="k", lw=1, label="Theil–Sen")
    ax[1].text(0.03, 0.97, f"r = {_stat(S, statistic='covar_pearson_r', **kw).value:.2f}\n"
               f"ρ = {_stat(S, statistic='covar_spearman_rho', **kw).value:.2f}\n"
               f"Theil–Sen {_stat(S, statistic='covar_theil_sen', **kw).value:.2f}\n"
               f"range {_stat(S, statistic='covar_gauge_range_m', **kw).value:.2f} m\n"
               f"n = {int(_stat(S, statistic='n', **kw).value)} passes", transform=ax[1].transAxes, va="top",
               fontsize=8.5, bbox=dict(fc="w", ec="0.8", lw=0.5))
    ax[1].set(xlabel="Rozumivka gauge, m EVRF2019", ylabel="SWOT RiverSP (≤3 km), m EGG2015",
              title="(b) SWOT at Rozumivka after the breach, Aug 2023 – Apr 2025")
    ax[1].legend(fontsize=7, loc="lower right", frameon=False)

    v7 = inc[inc.claim_id == "V7_DOWNSTREAM_POSTS_2023"].copy()
    v7["date"] = pd.to_datetime(v7.date)
    g = pd.read_parquet(GAUGES)
    g["date"] = pd.to_datetime(g.date)
    riv = g[g.station_id == 80805].set_index("date").H_evrf2019_m
    kh = E[(E.claim_id == "V7_DOWNSTREAM_POSTS_2023") & (E.station == "Kherson") & (E.sensor == "SWOT RiverSP")].copy()
    kh["date"] = pd.to_datetime(kh.date)
    kh = kh[kh.date.between("2023-06-13", "2023-07-08")].copy()
    kh["G"] = kh.date.map(riv)
    kh = kh[kh.gauge_level_m.isna()].dropna(subset=["G"])          # the river-yearbook-only days
    V7 = S[S.claim_id == "V7_DOWNSTREAM_POSTS_2023"]
    series = [("Kherson", kh.G, kh.satellite_level_m, "#2a73c9",
               _stat(S, claim_id="V7_DOWNSTREAM_POSTS_2023", station="Kherson", statistic="pearson_r",
                     variant="river yearbook only, 2023-06-13..07-08").value, len(kh), "13 Jun – 8 Jul, river yearbook")]
    for st, col in (("Parutyne", "#1aa37a"), ("Mykolaiv", "#7a4f9e")):
        q = v7[(v7.station == st) & (v7.sensor == "SWOT RiverSP") & (v7.period == "breach_fortnight")]
        rr = V7[(V7.station == st) & (V7.sensor == "SWOT RiverSP") & (V7.period == "breach_fortnight")
                & (V7.statistic == "pearson_r")].value.iloc[0]
        series.append((st, q.gauge_level_m, q.satellite_level_m, col, rr, len(q), "6–30 June"))
    for st, G, H, col, rr, n, win in series:
        Ga, Ha = G - G.mean(), H - H.mean()
        ax[2].scatter(Ga, Ha, s=18, c=col, ec="w", lw=0.3, label=f"{st} ({win}): r = {rr:.3f}, n = {n}")
    ax[2].plot([-1, 2.6], [-1, 2.6], c="0.6", ls="--", lw=0.8)
    ax[2].set(xlabel="gauge anomaly, m", ylabel="SWOT RiverSP anomaly, m",
              title="(c) daily SWOT at posts below the dam through the breach")
    ax[2].legend(fontsize=7, loc="upper left", frameon=False)
    ax[2].set_xlim(-0.8, 2.5)
    ax[2].set_ylim(-0.8, 2.5)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F27_covariability_main.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": 9})


# ------------------------------------------------------------ F28 (main) --
def f28(E, S):
    """Main-text Figure 7: water level through time, gauges with SWOT and
    ICESat-2. (a) Rozumivka 2019-2025, both satellites shifted by the
    pre-breach ICESat-2 closure c_IS2 only (the V3 transfer test in time);
    (b, c) Kherson and Parutyne through the breach, daily SWOT + c_pre."""
    plt.rcParams.update({"font.size": 12})
    inc = E[E.included_primary.astype(bool)]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    g = pd.read_parquet(GAUGES)
    g["date"] = pd.to_datetime(g.date)
    roz = g[g.station_id == 80959].set_index("date").H_evrf2019_m.reindex(pd.date_range("2019-01-01", "2025-12-31"))
    c_is2 = _stat(S, claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_m", sensor="ICESat-2").value
    v1r = inc[(inc.claim_id == "V1_ATL13_GAUGE_CLOSURE") & (inc.station == "Rozumivka")]
    v3 = inc[inc.claim_id == "V3_ROZUMIVKA_TRANSFER"]
    ax.plot(roz.index, roz.values, c="0.25", lw=0.9, label="Rozumivka gauge, daily (EVRF2019)")
    ax.scatter(pd.to_datetime(v1r.date), v1r.satellite_level_m + c_is2, s=30, c="#2a73c9", zorder=3,
               label=f"ICESat-2 ATL13, before the breach (n={len(v1r)})")
    ax.scatter(pd.to_datetime(v3.date), v3.satellite_level_m + c_is2, s=26, marker="D", c="#e8643a", ec="w",
               lw=0.3, zorder=3, label=f"SWOT RiverSP, after the breach (n={len(v3)})")
    ax.axvline(pd.Timestamp("2023-06-06"), c="grey", ls=":", lw=1)
    ax.set_ylabel("m EVRF2019")
    ax.set_title(f"(a) Rozumivka 2019–2025: both satellites + the pre-breach ICESat-2 closure "
                 f"c_IS2 = {100 * c_is2:+.1f} cm (SWOT not used to estimate it)", loc="left", fontsize=12)
    ax.legend(fontsize=10, loc="lower left", frameon=False)
    lv, riv = _gauge_2023()
    for k, (name, pid) in enumerate((("Kherson", 80805), ("Parutyne", 98025))):
        a = fig.add_subplot(gs[1, k])
        _post_panel(a, E, S, name, pid, lv, riv, (pd.Timestamp("2023-05-18"), pd.Timestamp("2023-07-22")))
        a.set_title(f"({'bc'[k]}) {a.get_title(loc='left')}", loc="left", fontsize=12)
        a.tick_params(axis="x", rotation=30)
        a.legend(fontsize=9, loc="upper right", frameon=False)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"F28_water_levels_main.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update({"font.size": 9})


def main():
    E, S = load()
    f16(E, S)
    f24(E, S)
    f19(E, S)
    f20(E, S)
    f21(E, S)
    f17(E, S)
    f22(E, S)
    f23(E, S)
    f25(E, S)
    f26(E, S)
    f27(E, S)
    f28(E, S)
    print("F16, F17, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28 ->", FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
