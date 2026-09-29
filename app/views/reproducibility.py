import ast
import json

import streamlit as st

from lib import config as C

st.title("Notebooks & reproducibility")

st.header("The pipeline behind the paper's validation")
st.markdown("Run in this order from the repository root; each script's docstring records what it "
            "checks and why. The audit exits non-zero if the manuscript, tables and figures disagree.")
for path, what in C.PIPELINE:
    p = C.ROOT / path
    doc = ""
    if p.exists():
        try:
            doc = (ast.get_docstring(ast.parse(p.read_text())) or "").split("\n")[0]
        except SyntaxError:
            pass
    st.markdown(f"- [`{path}`]({C.gh(p)}) — {what}" + (f"  \n  <small>{doc}</small>" if doc else ""),
                unsafe_allow_html=True)
st.code("""python scripts/ms5_paper1_zones.py
python scripts/ms6_paper1_figures.py          # F00, F18 (and the ms6b crossings)
python scripts/ms7_validation_paths.py        # V1–V6 evidence + summary
python scripts/ms7b_slope_geoid_sampling.py
python scripts/ms7c_validation_figures.py
python scripts/ms7d_manuscript_audit.py       # must report 0 failures
python app/prepare_app_data.py                # refresh this app's data layer
pytest tests/test_paper1_zones.py tests/test_ms7_validation.py""", language="bash")
st.caption("ms6b/ms7 need the full research environment (the companion ICESat-2 repository and the "
           "SWOT archive on the bulk volume). This app does not: it reads only the tracked outputs.")

st.header("Notebooks")
nbs = sorted(C.NOTEBOOKS.glob("*.ipynb"))
for p in nbs:
    try:
        nb = json.loads(p.read_text())
        md = next(("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"), "")
    except Exception:
        md = ""
    head = md.strip().split("\n")[0].lstrip("# ").strip() if md else p.stem
    rest = " ".join(l.strip() for l in md.strip().split("\n")[1:4] if l.strip())[:260]
    rel = p.relative_to(C.ROOT).as_posix()
    nbv = f"https://nbviewer.org/github/{C.REPO_URL.split('github.com/')[1]}/blob/{C.BRANCH}/{rel}"
    with st.container(border=True):
        st.markdown(f"**{head}**  \n{rest}")
        st.caption(f"[GitHub]({C.gh(p)}) · [nbviewer]({nbv})")
st.caption("Notebooks 10–14 are the vertical-chain and Kherson pilot analyses behind Supplementary "
           "Sections S1–S3; 15 assembles publication figures. The validation numbers of the paper are "
           "now produced by the ms7 scripts above.")

st.header("Outputs")
for p, what in ((C.VALIDATION, "validation evidence and summary tables"), (C.FIGURES, "figures"),
                (C.ZONES, "frozen analysis zones and their manifest"), (C.APP_DATA, "this app's data layer")):
    st.markdown(f"- [`{p.relative_to(C.ROOT).as_posix()}/`]({C.gh(p, 'tree')}) — {what}")
