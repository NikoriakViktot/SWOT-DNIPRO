"""Invariants of swot_dnipro.composites (plan 15 WP0.4; audit F-30)."""
import numpy as np

from swot_dnipro import composites as C


def test_mode_ignores_invalid_votes():
    # cell (0,0): observed twice as class 6, three dates unobserved -> 6, never 0
    st = np.zeros((5, 1, 2), "u1")
    st[0, 0, 0] = 6; st[1, 0, 0] = 6
    # cell (0,1): never observed -> 0
    m = C.class_mode(st)
    assert m[0, 0] == 6 and m[0, 1] == 0
    assert C.n_valid(st, 0)[0, 0] == 2 and C.n_valid(st, 0)[0, 1] == 0


def test_mode_tie_goes_to_lowest_code():
    st = np.array([[[3]], [[7]]], "u1")
    assert C.class_mode(st)[0, 0] == 3


def test_water_share_normalises_by_observed_only():
    w = np.array([[[1]], [[0]], [[255]], [[255]]], "u1")
    assert C.water_share(w)[0, 0] == 50            # 1 of 2 observed, not 1 of 4
    assert C.water_share(np.full((3, 1, 1), 255, "u1"))[0, 0] == 255


def test_median_index_nodata_propagation():
    st = np.array([[[1000]], [[3000]], [[C.INDEX_NODATA]]], "i2")
    assert C.median_index(st)[0, 0] == 2000
    assert C.median_index(np.full((2, 1, 1), C.INDEX_NODATA, "i2"))[0, 0] == C.INDEX_NODATA


def test_scl_shares_groups():
    scl = np.array([[[4]], [[5]], [[6]], [[0]]], "u1")
    s = C.scl_shares(scl)
    assert s["scl_share_veg"][0, 0] == 33 and s["scl_share_water"][0, 0] == 33
    assert s["scl_n_valid"][0, 0] == 3 and s["scl_mode"][0, 0] == 4
