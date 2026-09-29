import streamlit as st

from lib import charts as K
from lib import config as C
from lib import data as D

st.title("Validation paths V1–V6")
st.markdown(
    "Each path answers its own question with its own independent unit, and every number comes from "
    f"[ms7_summary.csv]({C.gh(C.SUMMARY)}); every plotted point is a row of "
    f"[ms7_evidence.csv]({C.gh(C.EVIDENCE)}).")


def table(claim, **extra):
    s = D.summary()
    s = s[s.claim_id == claim].dropna(axis=1, how="all")
    for k, v in extra.items():
        s = s[s[k] == v]
    with st.expander("Summary rows"):
        st.dataframe(s.drop(columns=["claim_id"]), hide_index=True, width="stretch")


tabs = st.tabs(list(C.CLAIMS.values()))

with tabs[0]:
    st.markdown("**Question:** does the ATL13 height sit right? Absolute closure of ATL13 against the six "
                "reservoir gauges before the breach; one point per beam transect at the station's radius.")
    c = st.columns(4)
    c[0].metric("matchups", int(D.val(claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="n_matchups")))
    c[1].metric("dates", int(D.val(claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="n_dates")))
    c[2].metric("mean of station medians c", D.cm(D.val(claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="mean_c_m")))
    c[3].metric("between-station SD", D.cm(D.val(claim_id="V1_ATL13_GAUGE_CLOSURE",
                                                  statistic="sd_between_station_m"), False))
    st.plotly_chart(K.v1_closure_by_station(), width="stretch")
    table("V1_ATL13_GAUGE_CLOSURE")

with tabs[1]:
    st.markdown("**Question:** does ATL13 follow the level through time? Within-station anomalies, so no "
                "closure residual can create the association. Rule: QC-passed overpasses, nearest reservoir "
                "gauge within 20 km of the overpass centroid, same-date gauge value.")
    r = D.stat(claim_id="V2_ATL13_GAUGE_COVARIABILITY", statistic="pearson_r")
    c = st.columns(4)
    c[0].metric("overpasses", int(D.val(claim_id="V2_ATL13_GAUGE_COVARIABILITY", statistic="n_overpasses")))
    c[1].metric("Pearson r", f"{r.value:.3f}", f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", delta_color="off")
    c[2].metric("Spearman ρ", f"{D.val(claim_id='V2_ATL13_GAUGE_COVARIABILITY', statistic='spearman_rho'):.3f}")
    c[3].metric("Theil–Sen slope", f"{D.val(claim_id='V2_ATL13_GAUGE_COVARIABILITY', statistic='theil_sen'):.3f}")
    st.plotly_chart(K.v2_anomaly_scatter(), width="stretch")
    table("V2_ATL13_GAUGE_COVARIABILITY")

with tabs[2]:
    st.markdown(
        "**Question:** does the pre-breach ICESat-2 frame carry over to post-breach SWOT? Two closures are "
        "computed **independently against the same gauge** — ICESat-2 before the breach (c_IS2) and SWOT "
        "after it (c_SWOT) — and only then compared. Nothing corrects SWOT by ICESat-2 first, so the test "
        "is not circular. The SWOT pass is the independent unit.")
    is2 = D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_m", sensor="ICESat-2")
    sw = D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_m", sensor="SWOT", radius_km=3.0,
                variant="raw_median")
    dd = D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_diff_m", radius_km=3.0,
                variant="raw_median: median c_SWOT - median c_IS2, both resampled")
    c = st.columns(3)
    c[0].metric("c_IS2, pre-breach", D.cm(is2.value), D.ci_cm(is2), delta_color="off")
    c[1].metric("c_SWOT, post-breach (≤3 km)", D.cm(sw.value), D.ci_cm(sw), delta_color="off")
    c[2].metric("c_SWOT − c_IS2", D.cm(dd.value), D.ci_cm(dd), delta_color="off")
    st.caption("No statistically resolved difference was detected. An interval that contains zero is not a "
               "formal equivalence test.")
    st.plotly_chart(K.v3_closures(), width="stretch")
    st.markdown("**Support and slope sensitivity** (post-breach Rozumivka is a sloping river):")
    st.dataframe(K.v3_radius_table(), hide_index=True, width="stretch")
    table("V3_ROZUMIVKA_TRANSFER")

with tabs[3]:
    st.markdown("**Question:** do the two altimeters agree directly? A system-wide consistency check over "
                "R/F/D/E — **not** the reservoir's validation (R holds one crossing). RiverSP only; one "
                "crossing = one ICESat-2 overpass × one SWOT pass within 24 h.")
    p = dict(claim_id="V4_SWOT_ICESAT_DIRECT", variant=None)
    m = D.stat(statistic="median_m", **p)
    ex = dict(claim_id="V4_SWOT_ICESAT_DIRECT", variant="excluding 2023-06-06..06-20 (breach fortnight)")
    c = st.columns(5)
    c[0].metric("crossings", int(D.val(statistic="n", **p)))
    c[1].metric("median Δ", D.cm(m.value), D.ci_cm(m), delta_color="off")
    c[2].metric("NMAD", D.cm(D.val(statistic="nmad_m", **p), False))
    c[3].metric("RMSE", D.cm(D.val(statistic="rmse_m", **p), False),
                f"{D.cm(D.val(statistic='rmse_m', **ex), False)} without 6–20 Jun 2023", delta_color="off")
    c[4].metric("anomaly r", f"{D.val(statistic='pearson_r_anomaly', **p):.2f}",
                f"ρ {D.val(statistic='spearman_rho_anomaly', **p):.2f}", delta_color="off")
    st.warning(
        f"Raw r on absolute heights is {D.val(statistic='pearson_r_raw', **p):.3f}, but it spans an "
        f"{D.val(statistic='height_range_m', **p):.1f} m range and is carried by one high crossing in R: "
        f"without it r = {D.val(claim_id='V4_SWOT_ICESAT_DIRECT', statistic='pearson_r_raw', variant='excluding the single R crossing (leverage)'):.2f}. "
        "Bias, NMAD and RMSE are the agreement metrics; correlation is supplementary.")
    st.plotly_chart(K.v4_scatter(), width="stretch")
    st.markdown("**Temporal-collocation sensitivity.** The full-sample RMSE is dominated by a crossing on "
                "6 June 2023, when a few hours of separation meant a metre-scale change in level; the "
                "primary sample keeps it.")
    st.plotly_chart(K.v4_timegap(), width="stretch")
    table("V4_SWOT_ICESAT_DIRECT")

with tabs[4]:
    st.markdown("**Product-specific supplementary test, never pooled with RiverSP.** Lake-averaged LakeSP "
                "level against the median ATL13 height inside the observed polygon, same day.")
    for sz in ("small lakes", "large lakes"):
        r = D.stat(claim_id="V5_LAKESP_PRODUCT", statistic="median_m", variant=sz)
        n = int(D.val(claim_id="V5_LAKESP_PRODUCT", statistic="n", variant=sz))
        nm = D.val(claim_id="V5_LAKESP_PRODUCT", statistic="nmad_m", variant=sz)
        st.markdown(f"- **{sz}** (n = {n} lake-overpass pairs): {D.cm(r.value)} {D.ci_cm(r)}, NMAD {D.cm(nm, False)}")
    st.caption("A lake-averaged level of a large, sloping polygon is not a valid test; it is shown, not used.")
    table("V5_LAKESP_PRODUCT")

with tabs[5]:
    st.markdown("**Kherson as a local anchor.** Closure c = gauge − satellite per sensor, product and period, "
                "each with its own support: ICESat-2 segments ≤ 10 km, RiverSP nodes ≤ 3 km, PIXC open water "
                "≤ 1 km. Post-breach series start on 1 July 2023.")
    st.plotly_chart(K.v6_closures(), width="stretch")
    st.plotly_chart(K.v6_scatter(), width="stretch")
    table("V6_KHERSON_CLOSURE")

with tabs[6]:
    st.markdown("**V7 — posts below the dam, 2023.** Sea-yearbook daily levels against SWOT RiverSP (≤ 3 km) and "
                "ICESat-2 (≤ 10 km) by period; c_pre is each post's pre-breach SWOT closure, the shift in "
                "Figures S8–S9. **V8 — reference surfaces** at the good SWOT node locations in R/F/D/E. "
                "**Gauge network** — gauge-to-gauge co-variation, 2019–2021.")
    s7 = D.summary()
    s7 = s7[s7.claim_id == "V7_DOWNSTREAM_POSTS_2023"]
    t = s7[s7.statistic.isin(["c_median_m", "c_nmad_m", "pearson_r"])].pivot_table(
        index=["station", "sensor", "period"], columns="statistic", values="value", aggfunc="first")
    t["n"] = s7[s7.statistic == "c_median_m"].set_index(["station", "sensor", "period"]).n
    t[["c_median_m", "c_nmad_m"]] = (100 * t[["c_median_m", "c_nmad_m"]]).round(1)
    st.dataframe(t.rename(columns={"c_median_m": "c, cm", "c_nmad_m": "NMAD, cm", "pearson_r": "r"}).round(3),
                 width="stretch")
    st.markdown("**Kherson: yearbooks, peak, recession and lag**")
    st.dataframe(s7[s7.station.eq("Kherson") & (s7.variant.notna() | s7.statistic.str.startswith("kherson"))]
                 .dropna(axis=1, how="all").drop(columns=["claim_id"]), hide_index=True, width="stretch")
    for claim in ("V8_REFERENCE_SURFACES", "GAUGE_NETWORK"):
        st.markdown(f"**{claim.replace('_', ' ').title()}**")
        st.dataframe(D.summary()[D.summary().claim_id == claim].dropna(axis=1, how="all").drop(columns=["claim_id"]),
                     hide_index=True, width="stretch")
