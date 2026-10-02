"""Composite operators shared by p25 / p40 / p43 (plan 15, WP0.4).

Every temporal composite in this project must (a) carry its own observation count and
(b) never let "not observed" vote. The audit run 20260918T175108Z found both mistakes in
early pilot code (F-30: INVALID class 0 in the mode vote collapsed valid area to the
footprint of single-tile dates), so the operators live here once, with tests.

Conventions
-----------
* class stacks are uint8 (t, ny, nx), 0 = INVALID (not observed); classes 1..9.
* water3 stacks are uint8, 0 land / 1 water / 255 not observed.
* index stacks are int16 (t, ny, nx) with INDEX_NODATA = -32768.
* SCL stacks are uint8, 0 = no data; ESA classes 1..11.
"""
from __future__ import annotations

import numpy as np

INDEX_NODATA = -32768
MASK_NODATA = 255


def n_valid(stack: np.ndarray, nodata) -> np.ndarray:
    """Number of observed dates per cell (uint8)."""
    return (stack != nodata).sum(axis=0).astype("u1")


def class_mode(stack: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Per-cell mode over OBSERVED classes only; ties -> lowest class code; 0 where never observed.

    Class 0 never votes: a cell seen twice as class 6 and three times unobserved is class 6."""
    counts = np.zeros((n_classes,) + stack.shape[1:], "u2")
    for k in range(1, n_classes):
        counts[k] = (stack == k).sum(axis=0)
    mode = counts.argmax(axis=0).astype("u1")
    mode[counts.max(axis=0) == 0] = 0
    return mode


def water_share(water3: np.ndarray) -> np.ndarray:
    """Share of observed dates classed water, in percent (uint8); 255 where never observed."""
    obs = water3 != MASK_NODATA
    cnt = obs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(cnt > 0, (water3 == 1).sum(axis=0) / np.maximum(cnt, 1), np.nan)
    return np.where(cnt > 0, np.round(s * 100), MASK_NODATA).astype("u1")


def median_index(stack: np.ndarray, nodata: int = INDEX_NODATA) -> np.ndarray:
    """Per-cell median over observed dates (int16), nodata where never observed."""
    a = stack.astype("f4")
    a[stack == nodata] = np.nan
    with np.errstate(all="ignore"):
        m = np.nanmedian(a, axis=0)
    return np.where(np.isfinite(m), np.round(m), nodata).astype("i2")


def scl_shares(scl: np.ndarray) -> dict:
    """Shares (percent, uint8) of the S2 Scene Classification groups over observed dates.

    groups: veg = {4}, notveg = {5}, water = {6}, cloud_shadow_snow = {3, 8, 9, 10, 11};
    plus the SCL mode over observed dates (0 where never observed)."""
    obs = scl != 0
    cnt = obs.sum(axis=0)
    out = {}
    for name, codes in (("veg", (4,)), ("notveg", (5,)), ("water", (6,)), ("cloud_shadow_snow", (3, 8, 9, 10, 11))):
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(cnt > 0, np.isin(scl, codes).sum(axis=0) / np.maximum(cnt, 1), np.nan)
        out[f"scl_share_{name}"] = np.where(cnt > 0, np.round(s * 100), MASK_NODATA).astype("u1")
    out["scl_mode"] = class_mode(scl, n_classes=12)
    out["scl_n_valid"] = cnt.astype("u1")
    return out
