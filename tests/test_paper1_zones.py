"""Paper 1 zones R/F/D/E: the frozen file must still be the geometry the
manuscript describes - disjoint, with the areas and hashes in the manifest."""
import itertools
import json

import pytest
from shapely.geometry import shape

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

MANIFEST = CFG.ROOT / "outputs/paper/zones/paper1_zones_manifest.json"
ZONES = ("R_FORMER_KAKHOVKA_RESERVOIR", "F_LOWER_DNIPRO_FLOODWAY",
         "D_KHERSON_DELTA", "E_DNIPRO_BUG_ESTUARY")

pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason="run scripts/ms5_paper1_zones.py first")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def zones():
    return {k: SD.load_utm(k) for k in ZONES}


def test_zones_are_disjoint(zones, manifest):
    tol = manifest["overlap_tolerance_km2"]
    for a, b in itertools.combinations(ZONES, 2):
        assert zones[a].intersection(zones[b]).area / 1e6 < tol, (a, b)


@pytest.mark.parametrize("name", ZONES)
def test_zone_matches_manifest(zones, manifest, name):
    m = manifest["zones"][name]
    # hash the geometry exactly as frozen in the file (EPSG:32636); a hash
    # taken after reprojection would depend on the PROJ version
    gj = json.loads((CFG.ROOT / manifest["file"]).read_text())
    raw = next(shape(f["geometry"]) for f in gj["features"]
               if f["properties"]["domain"] == name)
    assert SD.geom_hash(raw) == m["geometry_hash_utm"]
    assert zones[name].area / 1e6 == pytest.approx(m["area_km2"], abs=0.1)


def test_flood_envelope_is_an_event_layer_not_a_zone(zones, manifest):
    env = SD.load_utm("event_flood_2023_s1_envelope")
    assert env.area / 1e6 == pytest.approx(
        manifest["event_layers"]["event_flood_2023_s1_envelope"]["area_km2"], abs=0.1)
    # F is hydrographic: it must not equal, or be bounded by, the envelope
    assert zones["F_LOWER_DNIPRO_FLOODWAY"].area > 2 * env.area
