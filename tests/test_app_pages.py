"""Smoke test: every page of the companion app runs without an exception on
the tracked outputs alone."""
import sys
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
PAGES = ["home", "maps", "validation", "figures", "reproducibility", "methods"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    at = st_testing.AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
