import pandas as pd
import streamlit as st

from lib import config as C
from lib import data as D

st.title("Figures & tables")
st.markdown("The figures of the manuscript in its order, with captions taken from the text. Status: "
            "**recomputed** — drawn from the reproducible ms7/ms6 outputs; **carried from v5 — pending "
            "final rebuild** — the v5 image, for sections the validation rebuild did not touch.")

figs = D.manuscript_figures()
if figs.empty:
    st.error(f"Manuscript not found at {C.MANUSCRIPT_MD.relative_to(C.ROOT)}.")
for _, f in figs.iterrows():
    status, script = C.FIGURE_STATUS.get(f.file, C.DEFAULT_STATUS)
    with st.container(border=True):
        badge = "🟢" if status == "recomputed" else "🟡"
        st.markdown(f"#### {f.label} {badge} `{status}`")
        p = C.FIGURES / f.file
        if p.exists():
            st.image(str(p), width="stretch")
        else:
            st.warning(f"{f.file} is missing.")
        st.markdown(f.caption)
        links = [f"[image]({C.gh(p)})"]
        if script:
            links.append(f"[script]({C.gh(C.ROOT / script)})")
        st.caption(" · ".join(links))

st.header("Tables")
for p, what in ((C.SUMMARY, "one row per claim and statistic, with the v5 value where one existed"),
                (C.EVIDENCE, "one row per matchup / overpass / pass / crossing, with pass IDs"),
                (C.INVENTORY, "gauge record extents"),
                (C.SLOPE_SAMPLING, "headline slope vs EGG2015 sampling")):
    with st.expander(f"{p.name} — {what}"):
        df = pd.read_csv(p, low_memory=False)
        st.dataframe(df, hide_index=True, width="stretch", height=320)
        st.download_button("Download CSV", p.read_bytes(), file_name=p.name, key=p.name)
        st.markdown(f"[On GitHub]({C.gh(p)})")
