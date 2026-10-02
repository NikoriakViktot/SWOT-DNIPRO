import re

import streamlit as st

from lib import config as C
from lib import data as D

md = D.manuscript()
title = re.search(r"^# (.+)$", md, flags=re.M)
st.title(title.group(1) if title else "From impounded pool to river")
st.caption("Companion app to the manuscript. Every number on these pages is read from the "
           "reproducible validation tables; nothing is typed in.")

c = st.columns(5)
for col, (lab, url) in zip(c, [("🔗 Repository", C.REPO_URL), ("📄 Manuscript", C.gh(C.MANUSCRIPT_MD)),
                               ("📓 Notebooks", C.gh(C.NOTEBOOKS, "tree")),
                               ("🗂 Validation outputs", C.gh(C.VALIDATION, "tree")),
                               ("🖼 Figures", C.gh(C.FIGURES, "tree"))]):
    col.link_button(lab, url, width="stretch")

abstract = D.section("ABSTRACT")
if abstract:
    with st.expander("Abstract", expanded=True):
        st.markdown(abstract)

st.subheader("How the evidence fits together")
st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", fontname="Helvetica", fontsize=11];
  gauges [label="In-situ gauges\\n(BS-77 → EVRF2019)"];
  is2 [label="ICESat-2 ATL13\\npre-breach, R", fillcolor="#dbe9f6"];
  swot [label="SWOT RiverSP\\npost-breach", fillcolor="#fde3d6"];
  slope [label="Headline result:\\nper-overpass slope\\npre vs post", fillcolor="#e3eedc"];
  gauges -> is2 [label="V1 closure\\nV2 co-variability"];
  gauges -> swot [label="V3 Rozumivka\\nV6 Kherson"];
  is2 -> swot [label="V4 direct crossings\\nR/F/D/E", dir=both, style=dashed];
  is2 -> slope [label="EGG2015, bilinear"];
}
""")

st.subheader("Key validation results")
k = st.columns(4)
v1 = D.stat(claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="mean_c_m")
k[0].metric("V1 closure, six reservoir gauges", D.cm(v1.value),
            f"{int(D.val(claim_id='V1_ATL13_GAUGE_CLOSURE', statistic='n_matchups'))} matchups / "
            f"{int(D.val(claim_id='V1_ATL13_GAUGE_CLOSURE', statistic='n_dates'))} dates", delta_color="off")
v2 = D.stat(claim_id="V2_ATL13_GAUGE_COVARIABILITY", statistic="pearson_r")
k[1].metric("V2 co-variability r", f"{v2.value:.3f}",
            f"n = {int(D.val(claim_id='V2_ATL13_GAUGE_COVARIABILITY', statistic='n_overpasses'))} overpasses",
            delta_color="off")
v3 = D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_diff_m", radius_km=3.0,
            variant="raw_median: median c_SWOT - median c_IS2, both resampled")
k[2].metric("V3 c_SWOT − c_IS2 (Rozumivka)", D.cm(v3.value), D.ci_cm(v3) + " — no resolved difference",
            delta_color="off")
v4 = D.stat(claim_id="V4_SWOT_ICESAT_DIRECT", statistic="median_m", variant=None)
k[3].metric("V4 SWOT − ICESat-2, crossings ≤ 24 h", D.cm(v4.value),
            f"NMAD {D.cm(D.val(claim_id='V4_SWOT_ICESAT_DIRECT', statistic='nmad_m', variant=None), False)}, "
            f"n = {int(D.val(claim_id='V4_SWOT_ICESAT_DIRECT', statistic='n', variant=None))}", delta_color="off")

sl = D.slope_sampling().set_index("statistic").value
st.info(f"Headline slope contrast (production, bilinear EGG2015): **{float(sl['contrast_bilinear']):+.3f} cm/km**, "
        f"{int(float(sl['post_positive_bilinear']))}/{int(float(sl['n_post']))} post-breach slopes positive. "
        f"With nearest-cell EGG2015 sampling: {float(sl['contrast_nearest']):+.3f} cm/km, "
        f"{int(float(sl['post_positive_nearest']))}/{int(float(sl['n_post']))} positive.")
st.caption("Source: " + ", ".join(f"[{p.name}]({C.gh(p)})" for p in (C.SUMMARY, C.EVIDENCE, C.SLOPE_SAMPLING)))
