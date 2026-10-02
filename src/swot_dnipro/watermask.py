"""Sentinel-2 water masking for the post-breach classification.

Method
------
Water is never decided from a single index. A pixel is water only if it passes a
spectral test **and** the L2A scene classification permits it:

    NDWI  = (B03 - B08) / (B03 + B08)      green / NIR      10 m
    MNDWI = (B03 - B11) / (B03 + B11)      green / SWIR     20 m -> resampled to 10 m

    water = (NDWI > t_ndwi) AND (MNDWI > t_mndwi) AND SCL in {water, ...allowed}
            AND NOT SCL in {cloud, cirrus, cloud shadow, snow, saturated, no-data}

SCL classes (Sen2Cor L2A):
    0 no-data          1 saturated/defective  2 dark-area/topo shadow
    3 cloud shadow     4 vegetation           5 bare soil
    6 WATER            7 unclassified         8 cloud medium prob
    9 cloud high prob 10 thin cirrus         11 snow/ice

Thresholds are configurable and their sensitivity is tested; the defaults
(NDWI > 0.0, MNDWI > 0.0) are the conventional McFeeters/Xu values.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: SCL values that veto a water classification regardless of the indices.
SCL_REJECT = (0, 1, 3, 8, 9, 10, 11)
#: SCL values compatible with water.
SCL_WATER_OK = (2, 6, 7, 4, 5)

DEFAULT_NDWI = 0.0
DEFAULT_MNDWI = 0.0


@dataclass
class WaterMask:
    mask: np.ndarray          # bool, True = water
    ndwi: np.ndarray
    mndwi: np.ndarray
    scl: np.ndarray
    transform: object
    crs: object
    tile: str
    sensing_time: str
    #: bool, True where the pixel was actually OBSERVED (SCL not in SCL_REJECT
    #: and a real reflectance). A pixel that is False here is cloud, shadow,
    #: snow or no-data -- it is NOT land, and must never be written as 0.
    #: Optional so positional constructors elsewhere keep working; build_mask
    #: always sets it.
    valid: np.ndarray | None = None

    @property
    def n_water(self) -> int:
        return int(self.mask.sum())


def _find(names, band: str, res: str) -> str | None:
    hits = [n for n in names if n.endswith(".jp2") and f"_{band}_{res}.jp2" in n]
    return hits[0] if hits else None


def build_mask(zip_path: Path, ndwi_thr: float = DEFAULT_NDWI,
               mndwi_thr: float = DEFAULT_MNDWI, subsample: int = 2) -> WaterMask:
    """Build a water mask from an S2 L2A .SAFE zip.

    ``subsample`` decimates the 10 m grid (2 -> 20 m) to keep memory sane over a
    110x110 km tile; water bodies of interest here are hundreds of metres wide.
    """
    import rasterio
    from rasterio.enums import Resampling

    zp = str(zip_path)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    b03 = _find(names, "B03", "10m")
    b08 = _find(names, "B08", "10m")
    b11 = _find(names, "B11", "20m")
    scl = _find(names, "SCL", "20m")
    if not all([b03, b08, b11, scl]):
        raise RuntimeError(f"missing bands in {zip_path.name}")

    def read(member, out_shape=None):
        with rasterio.open(f"zip+file://{zp}!/{member}") as src:
            if out_shape is None:
                a = src.read(1, out_shape=(src.height // subsample, src.width // subsample),
                             resampling=Resampling.average).astype("f4")
                return a, src.transform * src.transform.scale(subsample, subsample), src.crs
            a = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype("f4")
            return a, None, None

    g, tr, crs = read(b03)
    n, _, _ = read(b08)
    shape = g.shape
    s11, _, _ = read(b11, out_shape=shape)
    sc, _, _ = read(scl, out_shape=shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (g - n) / (g + n)
        mndwi = (g - s11) / (g + s11)
    ndwi = np.nan_to_num(ndwi, nan=-9.0)
    mndwi = np.nan_to_num(mndwi, nan=-9.0)
    sc = sc.astype("i2")

    water = (ndwi > ndwi_thr) & (mndwi > mndwi_thr)
    water &= ~np.isin(sc, SCL_REJECT)
    water |= (sc == 6) & (ndwi > ndwi_thr - 0.15)   # trust SCL water with a relaxed index
    water &= ~np.isin(sc, SCL_REJECT)
    water &= (g > 0)

    # The observed/unobserved distinction, carried separately from water/land.
    # Until 2026-09-16 the rejected SCL classes were folded into water=False and
    # written as 0, indistinguishable from dry bed, so a cloud-covered tile
    # reported footprint_observed_fraction ~ 1.0 with zero water (Phase 20 dates
    # 2023-05-18 and 2023-08-06 did exactly that). Coverage must be measured
    # from THIS array, never inferred from the absence of water.
    valid = ~np.isin(sc, SCL_REJECT) & (g > 0)

    name = zip_path.name
    parts = name.split("_")
    return WaterMask(mask=water, ndwi=ndwi, mndwi=mndwi, scl=sc, transform=tr, crs=crs,
                     tile=parts[5] if len(parts) > 5 else "",
                     sensing_time=parts[2] if len(parts) > 2 else "",
                     valid=valid)


def sample_mask(wm: WaterMask, lon, lat) -> np.ndarray:
    """Return the water flag at geographic points (bool array, False off-grid)."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", wm.crs, always_xy=True)
    x, y = tf.transform(np.asarray(lon), np.asarray(lat))
    inv = ~wm.transform
    col, row = inv * (x, y)
    col = np.floor(col).astype(int)
    row = np.floor(row).astype(int)
    ok = (row >= 0) & (row < wm.mask.shape[0]) & (col >= 0) & (col < wm.mask.shape[1])
    out = np.zeros(len(col), bool)
    out[ok] = wm.mask[row[ok], col[ok]]
    return out


def distance_to_water_edge_km(wm: WaterMask, lon, lat) -> np.ndarray:
    """Distance from each point to the nearest non-water pixel (km).

    Large values mean the point sits well inside a broad water body; small values
    mean it is near a shoreline. Used as a morphology cue, never on its own.
    """
    from pyproj import Transformer
    from scipy import ndimage

    dist_px = ndimage.distance_transform_edt(wm.mask)
    px_m = abs(wm.transform.a)
    tf = Transformer.from_crs("EPSG:4326", wm.crs, always_xy=True)
    x, y = tf.transform(np.asarray(lon), np.asarray(lat))
    inv = ~wm.transform
    col, row = inv * (x, y)
    col = np.floor(col).astype(int)
    row = np.floor(row).astype(int)
    ok = (row >= 0) & (row < wm.mask.shape[0]) & (col >= 0) & (col < wm.mask.shape[1])
    out = np.full(len(col), np.nan)
    out[ok] = dist_px[row[ok], col[ok]] * px_m / 1000.0
    return out


def label_water_bodies(wm: WaterMask):
    """Connected-component labelling of the water mask (8-connectivity)."""
    from scipy import ndimage

    lab, n = ndimage.label(wm.mask, structure=np.ones((3, 3), int))
    return lab, n


def sample_labels(wm: WaterMask, lab: np.ndarray, lon, lat) -> np.ndarray:
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", wm.crs, always_xy=True)
    x, y = tf.transform(np.asarray(lon), np.asarray(lat))
    inv = ~wm.transform
    col = np.floor((inv * (x, y))[0]).astype(int)
    row = np.floor((inv * (x, y))[1]).astype(int)
    ok = (row >= 0) & (row < lab.shape[0]) & (col >= 0) & (col < lab.shape[1])
    out = np.zeros(len(col), int)
    out[ok] = lab[row[ok], col[ok]]
    return out
