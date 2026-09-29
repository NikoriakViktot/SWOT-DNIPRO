"""Canonical Sentinel-2 L2A preprocessing: bands -> indices -> masks -> classes.

This is the module `outputs/planning/03_sentinel_reprocessing_plan.md` §3.3
asked for and that never existed: until 2026-09-16 the logic lived inline in
`phase19_watermasks.py` (water rule), `k10e_spectral_stack_2023.py` (NDVI /
NDMI / BSI and the 10-class surface scheme) and `p15_z1_optical_corroboration.py`
(mosaicking onto a registry grid). Nothing scientific is changed here; the
three are gathered, made pure, and given the two things they all lacked.

The two things they lacked
--------------------------
1. **The BOA offset.** Every L2A scene in this project's archive -- both the
   2023+ acquisitions (PB 05.10) and the 2017-2021 ones, which are Collection-1
   reprocessings (PB 05.00) -- carries ``BOA_ADD_OFFSET = -1000`` on all 13
   bands with ``BOA_QUANTIFICATION_VALUE = 10000`` (verified 2026-09-16 on
   `MTD_MSIL2A.xml` of one scene from each end). Reflectance is therefore
   ``(DN - 1000) / 10000``, and none of the existing code applied it.

   What that did and did not break, precisely:
   * The frozen water rule tests ``(B03-B08)/(B03+B08) > 0``, i.e. ``B03 > B08``.
     A common additive offset cancels in that inequality, so **every existing
     water mask is unaffected** and this module reproduces them.
   * Any NON-zero threshold is affected: NDVI at 0.15/0.30, NDMI at 0.10, BSI
     at 0.10, the SCL==6 relaxation at -0.15. With the offset left in, the
     denominator is inflated by 2000 DN and every index is compressed toward
     zero -- the k10e class fractions were computed on compressed indices.
   * A non-normalised index cannot cancel the offset at all. AWEIsh is one;
     it would be ~-1000 x (1 + 2.5 - 1.5 x 2 - 0.25) = off by hundreds of DN.
   The offset is read from the scene's own metadata, never assumed.

2. **A third state.** ``valid`` (observed) is carried separately from
   ``water`` from the first line to the last, and written as 255 in every
   mask. A cloud pixel is not a land pixel. See `p24_reemit_masks_3value.py`
   for what folding them together did to Phase 20's coverage QC.

Grid convention (inherited, documented, not "fixed")
----------------------------------------------------
`spatial_domains.build_grid` returns cell CENTRES on multiples of ``cell``; the
pixel EDGES are therefore at ``k*cell + cell/2``, half a pixel off the MGRS tile
edges (which sit on multiples of 20 m). Every Sentinel-1 zone grid in this
project has that same registration, so the optical products are put on the
identical grid with nearest-neighbour resampling and the half-pixel offset is
accepted as the project's grid convention. Position uncertainty of ±10 m is
already inside the ``sigma_shoreline_position = slope * cell / 2`` term of the
constraint budget (`hist24_three_contour_dem.py:338`).

Index registry
--------------
NDWI, MNDWI, NDVI were registered in `outputs/planning/index_registry.csv`.
NDMI and BSI were in production (k10e) but unregistered; AWEIsh and NDTI are
new here. All seven are registered in the same commit as this module.

    NDVI  = (B08 - B04) / (B08 + B04)
    NDWI  = (B03 - B08) / (B03 + B08)                        McFeeters 1996
    MNDWI = (B03 - B11) / (B03 + B11)                        Xu 2006
    NDMI  = (B08 - B11) / (B08 + B11)                        Gao 1996
    BSI   = ((B11+B04) - (B08+B02)) / ((B11+B04) + (B08+B02))
    AWEIsh = B02 + 2.5*B03 - 1.5*(B08 + B11) - 0.25*B12       Feyisa et al. 2014
    NDTI  = (B04 - B03) / (B04 + B03)                        Lacaux et al. 2007

AWEIsh is on reflectance (0-1); the published water threshold is ~0 but this
module does NOT threshold it -- the water rule stays the frozen NDWI/MNDWI/SCL
one (`12_GATE_7C_ROLE_FREEZE.md`). AWEIsh and NDTI are persisted as indices for
the maps and for later, separately-gated, use.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as CFG
from . import watermask as WM

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
INDEX_NAMES = ("NDVI", "NDWI", "MNDWI", "NDMI", "BSI", "AWEIsh", "NDTI")
INDEX_SCALE = 10000                 # int16 storage: value * 10000
INDEX_NODATA = -32768
MASK_NODATA = 255                   # 0 = land, 1 = water, 255 = not observed

#: Sen2Cor band_id -> band name, for reading BOA_ADD_OFFSET per band.
_BAND_ID = {0: "B01", 1: "B02", 2: "B03", 3: "B04", 4: "B05", 5: "B06", 6: "B07",
            7: "B08", 8: "B8A", 9: "B09", 10: "B10", 11: "B11", 12: "B12"}
_BANDS_10M = ("B02", "B03", "B04", "B08")
_BANDS_20M = ("B11", "B12", "SCL")

#: Physical surface classes -- verbatim k10e_spectral_stack_2023.CLASSES.
CLASSES = {
    0: "INVALID", 1: "OPEN_WATER", 2: "SHALLOW_OR_MIXED_WATER", 3: "WET_SEDIMENT",
    4: "DRY_BARE_SEDIMENT", 5: "SPARSE_HERBACEOUS", 6: "DENSE_HERBACEOUS",
    7: "REED_OR_FLOODED_VEGETATION", 8: "BUILT_HARD_SURFACE", 9: "AMBIGUOUS",
}
CLASSES_INV = {v: k for k, v in CLASSES.items()}
#: k10e thresholds, unchanged. Now applied to offset-corrected reflectance.
NDVI_VEG, NDVI_SPARSE = 0.30, 0.15
NDMI_WET = 0.10
BSI_BARE = 0.10


# --------------------------------------------------------------------------- #
# Scene metadata                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class SceneMeta:
    name: str
    tile: str
    sensing_time: str
    processing_baseline: str
    quantification: float
    offsets: dict = field(default_factory=dict)      # band name -> additive DN offset

    def reflectance(self, band: str, dn: np.ndarray) -> np.ndarray:
        off = float(self.offsets.get(band, 0.0))
        return (dn.astype("f4") + off) / self.quantification


def read_metadata(zip_path: Path) -> SceneMeta:
    """PB, quantification and per-band BOA_ADD_OFFSET from MTD_MSIL2A.xml.

    Raises if the metadata is missing -- an offset that cannot be read must not
    be silently assumed to be zero, because that is exactly the failure this
    module exists to end."""
    with zipfile.ZipFile(str(zip_path)) as z:
        mtd = [n for n in z.namelist() if n.endswith("MTD_MSIL2A.xml")]
        if not mtd:
            raise RuntimeError(f"{zip_path.name}: no MTD_MSIL2A.xml")
        xml = z.read(mtd[0]).decode("utf-8", "ignore")
    pb = re.search(r"<PROCESSING_BASELINE>([^<]+)", xml)
    q = re.search(r"<BOA_QUANTIFICATION_VALUE[^>]*>(\d+)", xml)
    offs = {_BAND_ID[int(b)]: float(v)
            for b, v in re.findall(r'<BOA_ADD_OFFSET band_id="(\d+)">(-?\d+)', xml)}
    if not q:
        raise RuntimeError(f"{zip_path.name}: no BOA_QUANTIFICATION_VALUE")
    parts = zip_path.name.split("_")
    return SceneMeta(name=zip_path.name.replace(".zip", ""),
                     tile=parts[5] if len(parts) > 5 else "",
                     sensing_time=parts[2] if len(parts) > 2 else "",
                     processing_baseline=pb.group(1) if pb else "UNKNOWN",
                     quantification=float(q.group(1)), offsets=offs)


# --------------------------------------------------------------------------- #
# Band reading                                                                 #
# --------------------------------------------------------------------------- #
def read_scene_bands(zip_path: Path, cell: float = 20.0, upsample=None) -> dict:
    """Read B02/B03/B04/B08/B11/B12/SCL from a SAFE zip onto one grid at `cell`.

    Same mechanics as `watermask.build_mask`: 10 m bands are decimated by
    averaging to `cell`, 20 m bands and SCL are resampled to that shape.

    `upsample` sets the interpolation used for the CONTINUOUS 20 m reflectance
    bands (B11, B12) when the target is finer than 20 m. It defaults to nearest,
    which is what every existing 20 m caller got and where it is a no-op because
    the shapes already match. The 10 m canonical build passes bilinear: a 20 m
    band must be brought to the lattice ONCE, before any index is computed, and
    never resampled again afterwards. SCL is categorical and always moves by
    nearest neighbour whatever `upsample` says.
    Reflectance bands come back as float32 WITH the BOA offset applied; SCL
    stays integer. Returns the tile grid too (`transform`, `crs`).
    """
    import rasterio
    from rasterio.enums import Resampling

    meta = read_metadata(zip_path)
    zp = str(zip_path)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    sub = int(round(cell / 10.0))
    out: dict = {"meta": meta}

    def member(band, res):
        m = WM._find(names, band, res)
        if m is None:
            raise RuntimeError(f"{zip_path.name}: missing {band}_{res}")
        return m

    shape = transform = crs = None
    for b in _BANDS_10M:
        with rasterio.open(f"zip+file://{zp}!/{member(b, '10m')}") as src:
            if shape is None:
                shape = (src.height // sub, src.width // sub)
                transform = src.transform * src.transform.scale(sub, sub)
                crs = src.crs
            dn = src.read(1, out_shape=shape, resampling=Resampling.average)
        out[b] = meta.reflectance(b, dn)
    up = upsample if upsample is not None else Resampling.nearest
    for b in _BANDS_20M:
        rs = Resampling.nearest if b == "SCL" else up          # SCL is categorical, never interpolated
        with rasterio.open(f"zip+file://{zp}!/{member(b, '20m')}") as src:
            dn = src.read(1, out_shape=shape, resampling=rs)
        out[b] = dn.astype("i2") if b == "SCL" else meta.reflectance(b, dn)
    out["transform"], out["crs"], out["shape"] = transform, crs, shape
    return out


# --------------------------------------------------------------------------- #
# Indices, masks, classes -- pure array functions                              #
# --------------------------------------------------------------------------- #
def _nd(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((a - b) / (a + b)).astype("f4")


def compute_indices(r: dict) -> dict:
    """All seven indices from offset-corrected reflectance. NaN where undefined."""
    B02, B03, B04, B08, B11, B12 = (r[k] for k in ("B02", "B03", "B04", "B08", "B11", "B12"))
    with np.errstate(invalid="ignore", divide="ignore"):
        bsi = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
    return {
        "NDVI": _nd(B08, B04),
        "NDWI": _nd(B03, B08),
        "MNDWI": _nd(B03, B11),
        "NDMI": _nd(B08, B11),
        "BSI": bsi.astype("f4"),
        "AWEIsh": (B02 + 2.5 * B03 - 1.5 * (B08 + B11) - 0.25 * B12).astype("f4"),
        "NDTI": _nd(B04, B03),
    }


def valid_mask(scl: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Observed pixels: SCL not in SCL_REJECT and a real reflectance."""
    return ~np.isin(scl, WM.SCL_REJECT) & np.isfinite(green) & (green > 0)


def water_rule(ndwi, mndwi, scl, valid,
               ndwi_thr: float = WM.DEFAULT_NDWI,
               mndwi_thr: float = WM.DEFAULT_MNDWI) -> np.ndarray:
    """The FROZEN water rule (watermask.build_mask:103-107), unchanged.

    Because the threshold is 0 the BOA offset does not move it; the SCL==6
    relaxation at -0.15 is the one part that now sees corrected indices."""
    ndwi = np.nan_to_num(ndwi, nan=-9.0)
    mndwi = np.nan_to_num(mndwi, nan=-9.0)
    water = (ndwi > ndwi_thr) & (mndwi > mndwi_thr)
    water &= ~np.isin(scl, WM.SCL_REJECT)
    water |= (scl == 6) & (ndwi > ndwi_thr - 0.15)
    water &= ~np.isin(scl, WM.SCL_REJECT)
    water &= valid
    return water


def classify(ndvi, ndwi, mndwi, ndmi, bsi, scl, water_mask, valid) -> np.ndarray:
    """10-class physical surface scheme -- verbatim k10e_spectral_stack_2023.classify.

    Decision order matters: later assignments overwrite earlier ones. SCL==5
    ("not vegetated") with low NDVI/NDMI is DRY_BARE_SEDIMENT, not built --
    k10e's docstring records why (0.39 "built" on freshly exposed lakebed).
    BUILT_HARD_SURFACE is kept in the legend and never assigned."""
    from scipy import ndimage
    cls = np.zeros(ndvi.shape, "u1")
    cls[valid] = CLASSES_INV["AMBIGUOUS"]
    cls[valid & water_mask] = CLASSES_INV["OPEN_WATER"]
    wm_dil = ndimage.binary_dilation(water_mask, iterations=1)
    wm_ero = ndimage.binary_erosion(water_mask, iterations=1)
    cls[valid & (wm_dil & ~wm_ero)] = CLASSES_INV["SHALLOW_OR_MIXED_WATER"]
    nonwater = valid & ~water_mask
    ndvi = np.nan_to_num(ndvi, nan=-9.0); ndmi = np.nan_to_num(ndmi, nan=-9.0)
    bsi = np.nan_to_num(bsi, nan=-9.0)
    cls[nonwater & (scl == 5) & (ndvi < NDVI_SPARSE) & (ndmi < NDMI_WET)] = CLASSES_INV["DRY_BARE_SEDIMENT"]
    cls[nonwater & (bsi >= BSI_BARE) & (ndvi < NDVI_SPARSE) & (ndmi < NDMI_WET)] = CLASSES_INV["DRY_BARE_SEDIMENT"]
    cls[nonwater & (ndmi >= NDMI_WET) & (ndvi < NDVI_VEG) & (bsi < BSI_BARE)] = CLASSES_INV["WET_SEDIMENT"]
    cls[nonwater & (ndvi >= NDVI_SPARSE) & (ndvi < NDVI_VEG) & (ndmi < NDMI_WET)] = CLASSES_INV["SPARSE_HERBACEOUS"]
    cls[nonwater & (ndvi >= NDVI_VEG) & (ndmi < NDMI_WET)] = CLASSES_INV["DENSE_HERBACEOUS"]
    cls[nonwater & (ndvi >= NDVI_VEG) & (ndmi >= NDMI_WET)] = CLASSES_INV["REED_OR_FLOODED_VEGETATION"]
    return cls


# --------------------------------------------------------------------------- #
# Zone grid and mosaicking                                                     #
# --------------------------------------------------------------------------- #
def zone_grid(zone: str, cell: float = 20.0) -> dict:
    """The registry grid for a zone, with the raster conventions attached.

    Adds to `build_grid`'s dict: `x0, y0, x1, y1` (pixel EDGES), `transform`
    (north-up, row 0 = top), `inside` (ny, nx bool). Note `gy` is ascending
    while rows run top-down, so row i corresponds to gy[ny-1-i]."""
    from rasterio.transform import from_origin
    from . import spatial_domains as SD
    G = SD.build_grid(SD.load_utm(zone), cell, what=f"{zone} optical grid")
    gx, gy = G["gx"], G["gy"]
    G["x0"], G["x1"] = float(gx[0] - cell / 2), float(gx[-1] + cell / 2)
    G["y0"], G["y1"] = float(gy[0] - cell / 2), float(gy[-1] + cell / 2)
    G["transform"] = from_origin(G["x0"], G["y1"], cell, cell)
    inside = np.zeros(G["ny"] * G["nx"], bool)
    inside[G["ins_idx"]] = True
    # build_grid's flat index is row-major over (gy ascending, gx); flip to top-down
    G["inside"] = inside.reshape(G["ny"], G["nx"])[::-1, :]
    G["zone"] = zone
    return G


def mosaic_to_grid(zip_paths: list[Path], G: dict, cell: float = 20.0, upsample=None) -> dict | None:
    """Read each scene, resample nearest onto the zone grid, first valid wins.

    "First valid" is decided by SCL > 0, as in p15's optical_mask; scenes are
    taken in the order given (callers sort by cloud or by name). Returns the
    reflectance bands + SCL on the zone grid, plus the list of scene metas and
    a per-pixel `n_scenes` count."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    ny, nx = G["ny"], G["nx"]
    acc = {b: np.full((ny, nx), np.nan, "f4") for b in _BANDS_10M + ("B11", "B12")}
    scl = np.zeros((ny, nx), "i2")
    n_scenes = np.zeros((ny, nx), "u1")
    metas = []
    for zp in zip_paths:
        try:
            r = read_scene_bands(zp, cell, upsample=upsample)
        except Exception as ex:
            print(f"      skip {Path(zp).name[:45]}: {type(ex).__name__}: {str(ex)[:60]}")
            continue
        got = {}
        for b in _BANDS_10M + _BANDS_20M:
            src = r[b]
            dst = np.full((ny, nx), 0 if b == "SCL" else np.nan,
                          "i2" if b == "SCL" else "f4")
            reproject(source=src, destination=dst,
                      src_transform=r["transform"], src_crs=r["crs"],
                      dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                      resampling=Resampling.nearest,
                      src_nodata=0 if b == "SCL" else None,
                      dst_nodata=0 if b == "SCL" else np.nan)
            got[b] = dst
        new = (got["SCL"] > 0) & (scl == 0)
        scl[new] = got["SCL"][new]
        for b in acc:
            acc[b][new] = got[b][new]
        n_scenes[got["SCL"] > 0] += 1
        metas.append(r["meta"])
    if not metas:
        return None
    return {**acc, "SCL": scl, "n_scenes": n_scenes, "metas": metas}


def scl_to_grid(zip_paths: list[Path], G: dict, cell: float = 20.0) -> np.ndarray | None:
    """SCL only, first valid wins -- the cheap backfill for per-date stacks written before
    p25 persisted SCL (plan 15 WP0.2). Reads one 20 m JP2 per scene instead of seven bands;
    the mosaic order and 'first valid' rule are identical to `mosaic_to_grid`, so the SCL
    written this way is the SCL that produced the existing indices/class/water3 files."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    ny, nx = G["ny"], G["nx"]
    scl = np.zeros((ny, nx), "i2")
    n = 0
    for zp in zip_paths:
        zp = Path(zp)
        try:
            with zipfile.ZipFile(str(zp)) as z:
                names = z.namelist()
            m = WM._find(names, "SCL", "20m")
            if m is None:
                raise RuntimeError("missing SCL_20m")
            with rasterio.open(f"zip+file://{zp}!/{m}") as src:
                sub = int(round(cell / 20.0))
                shape = (src.height // sub, src.width // sub)
                tr = src.transform * src.transform.scale(sub, sub)
                dn = src.read(1, out_shape=shape, resampling=Resampling.nearest).astype("i2")
                crs = src.crs
        except Exception as ex:
            print(f"      skip {zp.name[:45]}: {type(ex).__name__}: {str(ex)[:60]}")
            continue
        dst = np.zeros((ny, nx), "i2")
        reproject(source=dn, destination=dst, src_transform=tr, src_crs=crs,
                  dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                  resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        new = (dst > 0) & (scl == 0)
        scl[new] = dst[new]
        n += 1
    return scl if n else None


def process_on_grid(bands: dict) -> dict:
    """indices + valid + water + class from a band dict (tile or zone grid)."""
    idx = compute_indices(bands)
    valid = valid_mask(bands["SCL"], bands["B03"])
    water = water_rule(idx["NDWI"], idx["MNDWI"], bands["SCL"], valid)
    cls = classify(idx["NDVI"], idx["NDWI"], idx["MNDWI"], idx["NDMI"], idx["BSI"],
                   bands["SCL"], water, valid)
    return {"indices": idx, "valid": valid, "water": water, "class": cls}


# --------------------------------------------------------------------------- #
# Writers -- indices / class / valid kept SEPARATE (plan 03 §3.3)              #
# --------------------------------------------------------------------------- #
def _profile(G: dict, count: int, dtype: str, nodata) -> dict:
    return dict(driver="GTiff", height=G["ny"], width=G["nx"], count=count, dtype=dtype,
                crs=CFG.CRS_METRIC, transform=G["transform"], compress="deflate",
                predictor=2, tiled=True, nodata=nodata)


def write_indices(path: Path, idx: dict, valid: np.ndarray, G: dict, tags: dict) -> None:
    import rasterio
    with rasterio.open(path, "w", **_profile(G, len(INDEX_NAMES), "int16", INDEX_NODATA)) as dst:
        for i, name in enumerate(INDEX_NAMES, 1):
            a = idx[name]
            q = np.where(valid & np.isfinite(a),
                         np.clip(np.round(a * INDEX_SCALE), -32767, 32767),
                         INDEX_NODATA).astype("i2")
            dst.write(q, i)
            dst.set_band_description(i, name)
        dst.update_tags(scale=f"value/{INDEX_SCALE}", nodata=str(INDEX_NODATA), **tags)


def write_uint8(path: Path, arr: np.ndarray, G: dict, tags: dict, nodata=MASK_NODATA) -> None:
    import rasterio
    with rasterio.open(path, "w", **_profile(G, 1, "uint8", nodata)) as dst:
        dst.write(arr.astype("u1"), 1)
        dst.update_tags(**tags)


def three_valued(water: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """0 = observed land, 1 = observed water, 255 = not observed."""
    return np.where(valid, water.astype("u1"), np.uint8(MASK_NODATA)).astype("u1")
