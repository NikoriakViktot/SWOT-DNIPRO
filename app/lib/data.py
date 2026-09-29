"""Cached loaders. The app reads only reproducible outputs: the ms7 tables,
the frozen zone manifest, the app_data layer and the manuscript itself."""
from __future__ import annotations

import json
import re

import geopandas as gpd
import pandas as pd
import streamlit as st

from . import config as C


@st.cache_data
def summary() -> pd.DataFrame:
    return pd.read_csv(C.SUMMARY)


@st.cache_data
def evidence() -> pd.DataFrame:
    e = pd.read_csv(C.EVIDENCE, low_memory=False)
    e["included_primary"] = e.included_primary.astype(str).str.lower().eq("true")
    return e


@st.cache_data
def inventory() -> pd.DataFrame:
    return pd.read_csv(C.INVENTORY)


@st.cache_data
def slope_sampling() -> pd.DataFrame:
    return pd.read_csv(C.SLOPE_SAMPLING)


@st.cache_data
def zone_manifest() -> dict:
    return json.loads(C.ZONE_MANIFEST.read_text())


@st.cache_data
def build_info() -> dict:
    """The release stamp written by ms7d_manuscript_audit.py, plus the commit
    the app itself is running from (Streamlit Cloud clones with .git)."""
    import subprocess
    info = json.loads(C.BUILD_INFO.read_text()) if C.BUILD_INFO.exists() else {}
    try:
        info["git_commit_running"] = subprocess.check_output(
            ["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["git_commit_running"] = "unknown"
    return info


@st.cache_data
def app_manifest() -> dict:
    return json.loads((C.APP_DATA / "manifest.json").read_text())


@st.cache_data
def layer(name: str) -> gpd.GeoDataFrame | pd.DataFrame:
    p = C.APP_DATA / name
    return gpd.read_file(p) if p.suffix == ".geojson" else pd.read_csv(p)


def stat(**kw) -> pd.Series:
    """One row of the ms7 summary; keyword None means 'column is empty'.
    Raises if the selection is not exactly one row, so a changed table can
    never silently feed a different number into the app."""
    s = summary()
    for k, v in kw.items():
        s = s[s[k].isna()] if v is None else s[s[k] == v]
    if len(s) != 1:
        raise KeyError(f"summary selection {kw} matched {len(s)} rows")
    return s.iloc[0]


def val(**kw) -> float:
    return float(stat(**kw).value)


def cm(v: float, sign: bool = True, nd: int = 1) -> str:
    s = f"{100 * v:+.{nd}f}" if sign else f"{100 * v:.{nd}f}"
    return s.replace("-", "−") + " cm"


def ci_cm(r: pd.Series) -> str:
    return f"[{cm(r.ci_lo).replace(' cm', '')}, {cm(r.ci_hi).replace(' cm', '')}] cm"


@st.cache_data
def manuscript() -> str:
    return C.MANUSCRIPT_MD.read_text() if C.MANUSCRIPT_MD.exists() else ""


@st.cache_data
def manuscript_figures() -> pd.DataFrame:
    """Figures in manuscript order with their captions, parsed from the text."""
    md = manuscript()
    rows = []
    for m in re.finditer(r"!\[\]\(figures/([^)]+)\)\s*\n\s*\n\*\*(Figure [^.]+)\.\*\*\s*([^\n]*)", md):
        rows.append(dict(file=m.group(1), label=m.group(2), caption=m.group(3).strip()))
    return pd.DataFrame(rows)


def section(title: str) -> str:
    """Text of one manuscript section, up to the next heading of any level."""
    md = manuscript()
    m = re.search(rf"^#+ {re.escape(title)}\s*$(.*?)(?=^#+ )", md, flags=re.M | re.S)
    return m.group(1).strip() if m else ""
