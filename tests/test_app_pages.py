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


def test_s1_layer_manifest_matches_files():
    """The per-date Sentinel-1 layer: every date's PNG is present, its sha256 matches
    the manifest and the p94 table has the same dates."""
    import hashlib
    import json

    import pandas as pd

    man = json.loads((ROOT / "outputs/paper/app_data/s1/manifest.json").read_text())
    assert len(man["layers"]) == 11
    for lay in man["layers"]:
        p = ROOT / "outputs/paper/app_data" / lay["file"]
        assert p.exists(), p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == lay["sha256"], p
        assert 0 <= lay["corridor_coverage"] <= 1
    t = pd.read_csv(ROOT / "outputs/paper/app_data/s1_flood_dynamics.csv")
    assert set(t.date) == {lay["date"] for lay in man["layers"]}
    corr = t[t.region == "DNIPRO_CORRIDOR"].set_index("date").new_water_km2
    assert corr["2023-06-01"] == 0.0 and corr["2023-06-09"] > 250


def test_icesat2_pass_layer_matches_manifest():
    """The ICESat-2 pass layer: hash matches, every track has a product, period and
    date, and the ATL13 samples of the paper are all present."""
    import hashlib
    import json

    import geopandas as gpd

    app_data = ROOT / "outputs/paper/app_data"
    man = json.loads((app_data / "icesat2_passes_manifest.json").read_text())
    p = app_data / "icesat2_passes.geojson"
    assert hashlib.sha256(p.read_bytes()).hexdigest()[:16] == man["sha256"]
    g = gpd.read_file(p)
    assert len(g) == man["n_beam_tracks"]
    assert set(g["product"]) >= {"ATL13"}
    assert set(g[g["product"] == "ATL13"]["sample"]) == {"reservoir", "kherson", "estuary"}
    assert set(g.period) <= {"PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"}
    assert g.geometry.notna().all() and g.is_valid.all()
