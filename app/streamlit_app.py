"""Companion app for paper 1 — From impounded pool to river.

    streamlit run app/streamlit_app.py

Reads only reproducible outputs (outputs/paper/validation, outputs/paper/zones,
outputs/paper/app_data, outputs/paper/figures and the manuscript); see
app/lib/config.py for every path and link.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as C  # noqa: E402
from lib import state as S  # noqa: E402

st.set_page_config(page_title="Kakhovka: pool to river", page_icon="🌊", layout="wide")

pages = [
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/maps.py", title="Study area & maps", icon="🗺️"),
    st.Page("views/validation.py", title="Validation paths V1–V6", icon="📐"),
    st.Page("views/figures.py", title="Figures & tables", icon="🖼️"),
    st.Page("views/reproducibility.py", title="Notebooks & reproducibility", icon="📓"),
    st.Page("views/methods.py", title="Data & methods", icon="🧭"),
]

with st.sidebar:
    st.markdown("### Links")
    st.link_button("Repository", C.REPO_URL, width="stretch")
    st.link_button("Manuscript (Markdown)", C.gh(C.MANUSCRIPT_MD), width="stretch")
    st.link_button("Notebooks", C.gh(C.NOTEBOOKS, "tree"), width="stretch")
    st.link_button("Validation outputs", C.gh(C.VALIDATION, "tree"), width="stretch")
    st.link_button("Figures", C.gh(C.FIGURES, "tree"), width="stretch")
    st.link_button("Data & code availability",
                   C.gh(C.MANUSCRIPT_MD) + "#data-and-code-availability", width="stretch")
    st.link_button(f"Manuscript .docx ({C.RELEASE_TAG})", C.RELEASE_URL, width="stretch")
    st.caption(f"Links resolve on branch `{C.BRANCH}`.")
    st.caption(f"Session `{S.session_id()}`: your filters are kept in this link and survive a reload.")
    from lib import data as D
    b = D.build_info()
    st.markdown("---")
    st.caption(
        f"**Release** `{C.RELEASE_TAG}` · manuscript **{b.get('manuscript_version', '?')}** "
        f"(`{b.get('manuscript_sha256', '?')}`)  \n"
        f"**Git commit** `{b.get('git_commit_running', '?')}`  \n"
        f"**Evidence built** {b.get('evidence_built_utc', '?')} (`{b.get('evidence_sha256', '?')}`)  \n"
        f"**Validation pipeline** `{b.get('pipeline_sha256', '?')}` · audit **{b.get('audit', 'not run')}** "
        f"({b.get('audit_claims', '?')} claims)")

S.restore()                     # filters back after a reload (?sid=… in the URL)
st.navigation(pages).run()
S.save()
