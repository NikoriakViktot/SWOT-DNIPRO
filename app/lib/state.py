"""Filters that survive a page reload.

Streamlit forgets `st.session_state` when the browser reloads the page: a reload is a new session.
Every visitor therefore gets a short session id, kept in the URL as `?sid=…`, and the values of
the persisted widgets (keys starting with `f_`) are written to a small JSON file per session id.
On the next run the values come back from that file, so a reload, a bookmark of the URL or a
switch to another page and back keeps the chosen date, basemap, opacity and toggles.

Limits, stated: the store lives in the app container's temporary directory, so a redeploy or a
container restart of Streamlit Community Cloud starts it empty; a URL shared with someone else
shares the filters too (that is the point of a bookmark). Only plain values are stored.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

import streamlit as st

PREFIX = "f_"
_DIR = Path(os.environ.get("APP_STATE_DIR", Path(tempfile.gettempdir()) / "swot_dnipro_app_state"))
_SID = re.compile(r"^[0-9a-f]{12}$")
_PLAIN = (str, int, float, bool)


def session_id() -> str:
    """The visitor's session id: from the URL if it is valid, else a new one; always written back."""
    if "_sid" not in st.session_state:
        q = st.query_params.get("sid")
        st.session_state["_sid"] = q if isinstance(q, str) and _SID.match(q) else uuid.uuid4().hex[:12]
    sid = st.session_state["_sid"]
    if st.query_params.get("sid") != sid:          # page switches drop query params; put it back
        st.query_params["sid"] = sid
    return sid


def _file(sid: str) -> Path:
    return _DIR / f"{sid}.json"


def _load(sid: str) -> dict:
    try:
        return json.loads(_file(sid).read_text())
    except (OSError, ValueError):
        return {}


def restore() -> None:
    """Put stored values back for every persisted key the session does not hold.
    Runs on every rerun, because Streamlit drops the state of widgets that a page did not draw."""
    for k, v in _load(session_id()).items():
        if k.startswith(PREFIX) and k not in st.session_state:
            st.session_state[k] = v


def save() -> None:
    """Merge the session's current persisted values into the store (other pages' values are kept)."""
    sid = session_id()
    cur = {k: v for k, v in st.session_state.items()
           if isinstance(k, str) and k.startswith(PREFIX) and isinstance(v, _PLAIN)}
    if not cur:
        return
    data = _load(sid)
    if all(data.get(k) == v for k, v in cur.items()):
        return
    data.update(cur)
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        tmp = _file(sid).with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(_file(sid))
    except OSError:
        pass                                          # never break the page over a lost preference


def init(key: str, default, options=None):
    """Seed a persisted widget once; drop a stored value that is no longer a valid option.
    Widgets created after this must NOT pass value=/index=, so Streamlit reads the key."""
    if key in st.session_state and options is not None and st.session_state[key] not in list(options):
        del st.session_state[key]
    st.session_state.setdefault(key, default)
    return st.session_state[key]
