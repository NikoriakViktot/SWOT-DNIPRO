"""The 08:00 / 20:00 gauge term readings are Kyiv civil time. Every conversion to the
UTC satellite timeline must follow the Europe/Kyiv rules of the individual date, so a
seasonal offset can never hide an hour. Three layers are checked: the zoneinfo rules
themselves, the k5 term table this repository writes, and the V1 ICESat-2 matchups
of the companion repository (time difference to the nearest term)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swot_dnipro import config as CFG  # noqa: E402

KYIV, UTC = ZoneInfo("Europe/Kyiv"), ZoneInfo("UTC")
TERMS = (8, 20)


def term_utc(day: pd.Timestamp, hour: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(day.year, day.month, day.day, hour, tzinfo=KYIV).astimezone(UTC))


@pytest.mark.parametrize("day,offset", [("2020-01-15", 2), ("2020-07-15", 3), ("2023-03-26", 3),
                                        ("2023-10-29", 2), ("2021-03-28", 3), ("2021-10-31", 2)])
def test_kyiv_terms_resolve_to_the_seasonal_offset(day, offset):
    """08:00 and 20:00 Kyiv are never inside a DST gap or fold, so both resolve
    unambiguously, with +2 h in winter and +3 h in summer (transition days included)."""
    d = pd.Timestamp(day)
    for h in TERMS:
        loc = datetime(d.year, d.month, d.day, h, tzinfo=KYIV)
        assert loc.utcoffset() == timedelta(hours=offset), (day, h, loc.utcoffset())
        assert term_utc(d, h).hour == h - offset


def test_k5_term_table_offsets_follow_zoneinfo():
    p = ROOT / "outputs/tables/k5_gauge_levels_evrf2019.csv"
    if not p.exists():
        pytest.skip("k5 term table not built")
    t = pd.read_csv(p, usecols=["datetime_local_kyiv", "datetime_utc", "utc_offset_hours", "observation_slot"])
    t = t[t.observation_slot.isin(["h_08", "h_20"])]
    # the local strings carry their own offset (+02:00 / +03:00): parse through UTC, then back to Kyiv
    loc = pd.to_datetime(t.datetime_local_kyiv, utc=True).dt.tz_convert(KYIV)
    utc = pd.to_datetime(t.datetime_utc, utc=True)
    assert set(t.utc_offset_hours.unique()) <= {2.0, 3.0}
    naive = loc.dt.tz_localize(None)
    expect = np.array([datetime(x.year, x.month, x.day, x.hour, tzinfo=KYIV).utcoffset().total_seconds() / 3600
                       for x in naive])
    assert np.array_equal(expect, t.utc_offset_hours.to_numpy()), "k5 offsets disagree with Europe/Kyiv"
    assert (utc == loc).all(), "datetime_utc is not the same instant as the local term"
    assert ((utc.dt.tz_convert(KYIV).dt.hour) == naive.dt.hour).all()
    assert set(naive.dt.hour.unique()) == set(TERMS)


def test_v1_matchups_sit_at_the_nearest_kyiv_term():
    """time_diff_A_h of every V1 matchup must equal the distance from the ICESat-2 UTC
    epoch to the nearest 08:00/20:00 Kyiv term of that date or its neighbours. A fixed
    UTC+2 or UTC+3 would fail this on half of the year."""
    p = CFG.ICESAT_ROOT / "outputs/tables/gauge_icesat_egg2015_matchups.csv"
    if not p.exists():
        pytest.skip("companion matchup table not available")
    m = pd.read_csv(p, usecols=["datetime", "time_diff_A_h", "gauge_interpolated_A"])
    when = pd.to_datetime(m.datetime, utc=True)
    diff = []
    for w in when:
        day = w.tz_convert(KYIV).normalize().tz_localize(None)
        cands = [term_utc(day + pd.Timedelta(days=k), h) for k in (-1, 0, 1) for h in TERMS]
        diff.append(min(abs((c - w).total_seconds()) for c in cands) / 3600)
    diff = np.array(diff)
    bad = np.abs(diff - m.time_diff_A_h.to_numpy()) > 2 / 60
    assert not bad.any(), f"{bad.sum()} of {len(m)} matchups are not at the nearest Kyiv term; e.g. {m[bad].head(3)}"
    assert (diff <= 6 + 1e-9).all()
    assert m.gauge_interpolated_A.astype(str).str.lower().eq("true").all()
