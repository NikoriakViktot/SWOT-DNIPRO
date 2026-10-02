"""ms7 validation paths: every primary count in the summary is backed by the
same number of evidence rows, and the reproduced v5 closure stays reproduced."""
import pandas as pd
import pytest

from swot_dnipro import config as CFG

D = CFG.ROOT / "outputs/paper/validation"
pytestmark = pytest.mark.skipif(not (D / "ms7_summary.csv").exists(),
                                reason="run scripts/ms7_validation_paths.py first")


@pytest.fixture(scope="module")
def tables():
    return pd.read_csv(D / "ms7_summary.csv"), pd.read_csv(D / "ms7_evidence.csv")


def _v(S, claim, stat):
    s = S[(S.claim_id == claim) & (S.statistic == stat)]
    if "variant" in s:
        s = s[s.variant.isna()]
    return float(s.value.iloc[0])


def test_v1_reproduces_v5_closure(tables):
    S, E = tables
    assert _v(S, "V1_ATL13_GAUGE_CLOSURE", "n_matchups") == 53
    assert _v(S, "V1_ATL13_GAUGE_CLOSURE", "n_dates") == 30
    assert _v(S, "V1_ATL13_GAUGE_CLOSURE", "mean_c_m") == pytest.approx(-0.135, abs=5e-4)
    assert (E.claim_id == "V1_ATL13_GAUGE_CLOSURE").sum() == 53


@pytest.mark.parametrize("claim,stat", [
    ("V2_ATL13_GAUGE_COVARIABILITY", "n_overpasses"),
    ("V4_SWOT_ICESAT_DIRECT", "n"),
])
def test_primary_counts_have_evidence_rows(tables, claim, stat):
    S, E = tables
    assert _v(S, claim, stat) == E[(E.claim_id == claim) & E.included_primary].shape[0]


def test_v3_primary_passes_have_evidence_rows(tables):
    S, E = tables
    s = S[(S.claim_id == "V3_ROZUMIVKA_TRANSFER") & (S.statistic == "n")
          & (S.radius_km == 3.0) & (S.variant == "raw_median")]
    e = E[(E.claim_id == "V3_ROZUMIVKA_TRANSFER") & E.included_primary]
    assert float(s.value.iloc[0]) == len(e)
