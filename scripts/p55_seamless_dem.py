#!/usr/bin/env python
"""P55 -- one seamless DEM in EVRF2019 (plan 16, WP-D; user 2026-09-19): bathymetric bed where we have it, FABDEM (converted, p56) elsewhere.

Priority per cell (highest first), all in m EVRF2019 on the zone's 20 m frame (and a 50 m mosaic over the union of the four frames):
  1  zone bed DEM v2 (p53: zone{2,3,4}_bed_v2_PRE_BREACH_30m, kriged inside the 2019-2022 water polygon, bank taper to MAL)
  2  reservoir bed DEM (hist20 kakhovka_bed_OK_epoch_50m; to be replaced by the WP-C v2 when it exists -- same file name pattern)
  3  FABDEM in EVRF2019 (p56 terrain/<ZONE>/fabdem_evrf2019_20m)
FABDEM is never used where a VALID bathymetric surface exists, because FABDEM on water is the water surface, not terrain. The
converse also holds and is enforced per cell, and the rule is POSITIVE (p55e, 2026-09-20): a reservoir bed is written only where
the registry polygon confirms the cell is inside the water body, and the p64 island class and FABDEM's height then act as vetoes
inside that domain. Neither veto grants permission by itself. The earlier rule admitted every cell that was not a confirmed
island, inside a 60 m buffer around the polygon, which put 14.0 km2 of interpolated bed on the bank and produced the worst
source-2 residuals in the product (-20.2 m). The registry polygon is the water body's outline and still encloses islands and
valley slopes; writing interpolated bathymetry over them is what produced the artefacts p62/p63 measured (2026-09-19).

Seams: outside the bathymetry, within FEATHER_M of its edge, FABDEM is shifted by the product-level STEP measured at the seam
(`fab + w * (bed_edge - FABDEM_edge)`), and only where that step is at most MAX_SEAM_STEP_M. The earlier form blended the two
VALUES, `(1-w)*fab + w*bed`, which at the first cell outside the bathymetry (w ~ 0.8) replaced 34 m of bank with 18 m and dug a
trench along every bathymetric boundary. A step larger than the gate is a real bank and is left as a step, not smoothed away.

Every classification decision reads FABDEM with NEAREST resampling, while the elevation written to the product is bilinear:
FABDEM's native sample is 21 x 31 m at this latitude, and a bilinear mix of 14.5 m water and 40 m bank lands at 20-30 m and
silently flips an elevation gate.
Outputs
  $BULK_ROOT/dem_seamless/<ZONE>_dem_evrf2019_20m.tif, <ZONE>_dem_source_20m.tif (1/2/3, 0 nodata)
  $BULK_ROOT/dem_seamless/dem_seamless_evrf2019_50m.tif, dem_seamless_source_50m.tif  (union frame)
  outputs/tables/p55_seam_check.csv   per zone: km2 per source, seam statistics (|bathy - FABDEM| at the seam ring before blending)
  outputs/figures/p55_seamless_dem.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from rasterio import features

POOL_POLY = SD.load_utm("reservoir_full_pool_prebreach")      # inside it FABDEM is the 16 m water surface, never terrain

OUT = CFG.BULK_ROOT / "dem_seamless"; OUT.mkdir(exist_ok=True)
ZN = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
POOL = ROOT / "outputs/rasters/kakhovka_bed_OK_epoch_50m.tif"
FEATHER_M = 100.0
POOL_WS_MAX_M = 17.0      # Inside the registry pool FABDEM is the PRE-BREACH WATER SURFACE: measured p1 14.46, p95 14.57, p99 22.65 m.
                          # Above this it is real terrain -- island, valley slope or bank caught inside the coarse polygon -- and no
                          # bathymetric surface may be written over it. Used ONLY inside the pool: below the dam the water surface is
                          # near 0 m, so an absolute elevation gate would be meaningless there.
MAX_SEAM_STEP_M = 1.0     # The feather closes only a PRODUCT-LEVEL mismatch, of the order of the night-ICESat-2 LE90 of FABDEM itself
                          # (p57: FABDEM LE90 1.010 m). A larger step at the seam is a real bank and is left alone.
GAP_MAX_M = 150.0         # The pool gap fill is used only where BOTH the nearest bed cell and the shoreline lie within this radius,
                          # which is inside the tile halo, so both distance transforms are exact and the result is tiling-independent.
POOL_POLY_BUF = POOL_POLY.buffer(60.0)   # hoisted: it was rebuilt on every tile
ISLAND_MASK = ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif"   # p64: 0 bed/water, 1 land, 2 UNKNOWN
BED_ACCEPTED_STATUS = (1, 2)   # p53 prediction_status a consumer may use: 1 supported, 2 true-shore conditioned
P66_CLASS = "outputs/rasters/zone{n}/prebreach/prebreach_class.tif"   # below-dam pre-breach classes; 1 = CORE water
CORE_WATER_CLASS = 1
# Inside CORE water FABDEM is the WATER SURFACE, not terrain -- the same fact this script already relies on inside the
# reservoir polygon (measured: 0.00 m below the dam, 14.5 m in the pool, both very tight). So where p53 has no admissible
# bathymetry, the seamless DEM must declare a GAP rather than hand the cell FABDEM: putting the water surface there
# creates a vertical wall of several metres against the neighbouring bed and, worse, presents a water level as ground to
# a hydraulic model. No MAL and no FABDEM water surface is used as a boundary condition at the support edge.
GATE = "basin"            # patch p55e: "basin" = the registry polygon confirms membership and p64/FABDEM only veto inside it;
                          # "corrected" = the p55b rule, p64 class inside the 60 m buffer; "height" = the original FABDEM-height
                          # arbiter inside the 60 m buffer. See the block in compose_tile().


def to_frame(path: Path, shape, transform, resampling=Resampling.bilinear) -> np.ndarray:
    dst = np.full(shape, np.nan, "f4")
    with rasterio.open(path) as ds:
        reproject(source=rasterio.band(ds, 1), destination=dst, dst_transform=transform, dst_crs=CFG.CRS_METRIC, resampling=resampling, src_nodata=ds.nodata, dst_nodata=np.nan)
    return dst


def compose_tile(shape, transform, cell, core=None):
    """(dem, source, seam residuals, aux counts) for ONE tile (with halo) -- called per window so a 96 Mpx frame never sits in memory at once.

    `core` is the slice of this array that will actually be written; the seam sample and the aux counters are restricted to it so
    nothing inside the halo is counted two to four times.
    """
    # FABDEM is read TWICE. The elevation that goes into the product is bilinear and stays smooth; every classification decision
    # (is this the water surface? how big is the seam step?) is taken on the NEAREST read, because FABDEM's native sample is
    # 21 x 31 m and a bilinear mix of 14.5 m water and 40 m bank lands at 20-30 m and silently flips an elevation gate.
    core_water = np.zeros(shape, bool)
    for n in (4, 2, 3):
        cp = ROOT / P66_CLASS.format(n=n)
        if cp.exists():
            cv = to_frame(cp, shape, transform, Resampling.nearest)
            core_water |= np.nan_to_num(cv, nan=0).astype("u1") == CORE_WATER_CLASS
    fab = np.full(shape, np.nan, "f4"); fab_gate = np.full(shape, np.nan, "f4")
    for z in ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY"):
        p = CFG.BULK_ROOT / "terrain" / z / "fabdem_evrf2019_20m.tif"
        f = to_frame(p, shape, transform); fab = np.where(np.isnan(fab), f, fab)
        g = to_frame(p, shape, transform, Resampling.nearest); fab_gate = np.where(np.isnan(fab_gate), g, fab_gate)
    bathy = np.full(shape, np.nan, "f4"); src = np.zeros(shape, "u1")
    pool_in0 = features.rasterize([(POOL_POLY_BUF, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    pool_in = features.rasterize([(POOL_POLY, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    # WHERE MAY A BATHYMETRIC SURFACE BE WRITTEN? (patch p55e, 2026-09-20)
    # ADMISSION IS POSITIVE. The earlier rule asked only "is this cell NOT an island?" and treated every answer other than a
    # confirmed island -- including UNKNOWN, including everything outside p64's window -- as permission to write reservoir bed.
    # Absence of an island veto is not evidence of basin membership. Measured consequence (p55d/p55e): the 60 m buffer around
    # the registry polygon carried 14.0 km2 of bed, of which 43 % (ZONE_1) to 67 % (ZONE_4) sits on cells whose FABDEM stands
    # above the full-pool water surface, and the night-ICESat-2 tail rises monotonically into the shoreline -- 0.04 % more than
    # 5 m below ground beyond 1 km, 12.7 % within 50 m of the boundary, worst -20.16 m. FABDEM at the same points is 1.3 %, so
    # this is the product, not the validation target.
    # "basin" (default): membership is asserted by the registry polygon itself, WITHOUT the 60 m buffer -- that buffer is a
    #   tolerance for how the outline was drawn, never a statement that the cell is inside the water body. Two vetoes then act
    #   inside the confirmed domain: p64's island class, and FABDEM standing above the maximum operating level, since a cell
    #   whose bare earth is already above full pool was never submerged bed. Neither veto can grant permission on its own.
    # "corrected" (patch p55b) and "height" (the original) are kept runnable so the three can be compared on one frame.
    fab_ws_qc = (~np.isfinite(fab_gate)) | (fab_gate <= POOL_WS_MAX_M)
    island = to_frame(ISLAND_MASK, shape, transform, Resampling.nearest) if ISLAND_MASK.exists() else np.full(shape, np.nan, "f4")
    if GATE == "basin":
        bed_allowed = pool_in & ~(island == 1) & fab_ws_qc
    elif GATE == "corrected" and ISLAND_MASK.exists():
        bed_allowed = pool_in0 & ~(island == 1)          # NaN (outside the p64 window) -> allowed: the defect under repair
    else:
        bed_allowed = pool_in0 & fab_ws_qc
    pool = to_frame(POOL, shape, transform)
    m = np.isfinite(pool) & bed_allowed; bathy[m] = pool[m]; src[m] = 2   # hist20 bed inside the confirmed basin only
    for n in (4, 2, 3):
        b = ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_PRE_BREACH_30m.tif"
        if b.exists():
            zb = to_frame(b, shape, transform)
            # A FINITE ELEVATION IS NOT A VALID BATHYMETRY (repair 2026-09-20). p53 used to write a surface over every
            # cell of a union water polygon, so land inside that polygon carried a bed pinned near MAL ~ 0 m and this
            # merge accepted it purely because it was a number. The status raster is now the contract: only cells p53
            # itself declares supported (1) or true-shore conditioned (2) may replace terrain. Missing status raster
            # means a pre-repair product -> refuse to consume it rather than silently trust the elevation.
            st = ROOT / f"outputs/rasters/zone{n}/zone{n}_bed_v2_status_30m.tif"
            if st.exists():
                sv = to_frame(st, shape, transform, Resampling.nearest)
                ok_bed = np.isfinite(zb) & np.isin(np.nan_to_num(sv, nan=0).astype("u1"), BED_ACCEPTED_STATUS)
            else:
                raise SystemExit(f"{b.name} has no companion {st.name}: rerun p53 for zone{n} -- p55 no longer treats a finite elevation as valid bathymetry")
            m = ok_bed & ~pool_in0; bathy[m] = zb[m]; src[m] = 1      # never inside the reservoir polygon
    has_b = np.isfinite(bathy); has_f = np.isfinite(fab)
    # Former reservoir without a bed DEM (the unsounded belt below the 16 m shoreline), restricted to cells where FABDEM is the water
    # surface. Islands and valley slopes inside the coarse polygon keep FABDEM instead of being dragged down to the shoreline.
    # The gap fill INVENTS a value where nothing was surveyed, so unlike the bed it keeps FABDEM as a QC BOUND as well as the
    # spatial class: a cell whose FABDEM stands well above the pre-breach water surface must never be handed an invented bed,
    # whatever the optical record failed to corroborate. Without this, dropping the height arbiter re-opened a 0.4 km2 tail at
    # -31 m below FABDEM (measured on GEDI, 2026-09-20).
    gap = pool_in & ~has_b & bed_allowed & fab_ws_qc
    if gap.any() and has_b.any() and (~pool_in).any():
        d_b, ib = ndimage.distance_transform_edt(~has_b, return_indices=True); d_s, is_ = ndimage.distance_transform_edt(pool_in, return_indices=True)
        vb = bathy[ib[0], ib[1]]
        vs = np.minimum(fab_gate[is_[0], is_[1]], POOL_WS_MAX_M)   # the shoreline value must BE a shoreline: where the coarse polygon
        del ib, is_                                                # boundary cuts a valley slope, FABDEM just outside it is 30-60 m
        fill = (d_s * vb + d_b * vs) / np.maximum(d_s + d_b, 1e-6)
        ok = (gap & np.isfinite(vb) & np.isfinite(vs)
              & (d_b * cell <= GAP_MAX_M) & (d_s * cell <= GAP_MAX_M))   # both ends inside the halo -> tiling-independent
        bathy[ok] = fill[ok]; src[ok] = 5; has_b = np.isfinite(bathy)
    # border_value=1 so the tile's own array edge is not eroded into a spurious ring, and restricted to the core so the halo
    # overlap is not counted several times. Both biased the seam percentiles the feather gate is calibrated against.
    ring = has_b & ~ndimage.binary_erosion(has_b, iterations=1, border_value=1) & has_f & ~pool_in
    if core is not None:
        keep = np.zeros(shape, bool); keep[core] = True; ring &= keep
    seam = (bathy - fab)[ring]
    if has_b.any() and (~has_b).any():
        dist_out, idx = ndimage.distance_transform_edt(~has_b, return_indices=True); dist_out = dist_out.astype("f4") * cell
        near_b = bathy[idx[0], idx[1]]; near_f = fab_gate[idx[0], idx[1]]; del idx     # BOTH sampled at the nearest bathymetric cell
    else:
        dist_out = np.full(shape, np.inf, "f4"); near_b = bathy; near_f = fab_gate     # no bathymetry (or nothing but) -> no feather
    # OFFSET taper, not a value blend. The old form pulled FABDEM towards the BED VALUE, which replaced 34 m of land with 18 m at the
    # first cell outside the bathymetry and dug a trench along every bathymetric boundary. This shifts FABDEM by the product-level
    # step measured AT the seam, so FABDEM's own relief is preserved, and only where that step is small enough to be a datum
    # mismatch rather than a real bank.
    step = (near_b - near_f).astype("f4")
    blendable = np.isfinite(step) & (np.abs(step) <= MAX_SEAM_STEP_M)
    w = np.where(blendable, np.clip(1.0 - dist_out / FEATHER_M, 0.0, 1.0), 0.0).astype("f4")
    off = np.where(blendable, step, 0.0).astype("f4")          # keeps 0 * NaN out of the result beyond the band
    # CORE water without an admissible bed is a declared gap, never FABDEM (see CORE_WATER_CLASS above).
    gap_core = core_water & ~has_b & ~pool_in0
    usable_f = has_f & ~gap_core
    dem = np.where(has_b, bathy, np.where(usable_f, fab + w * off, np.nan)).astype("f4")
    src = np.where(has_b, src, np.where(usable_f, 3, 0)).astype("u1"); src[(~has_b) & usable_f & (w > 0)] = 4
    sl = core if core is not None else (slice(None), slice(None))
    aux = np.array([int((pool_in & ~bed_allowed & (src >= 3))[sl].sum()),      # pool cells where FABDEM is terrain and was kept
                    int((pool_in & bed_allowed & (src == 3))[sl].sum()),                # unsounded belt left at the FABDEM water surface
                    int(gap_core[sl].sum())], "i8")                                     # CORE water declared a gap rather than given FABDEM
    return dem, src, seam, aux


def compose_to_files(dem_path: Path, src_path: Path, shape, transform, cell, tags, tile=2048, halo_m=300.0):
    """Tiled composition with a halo so the feather/distance transform is exact at tile edges; returns per-source cell counts and seam residuals."""
    assert GAP_MAX_M < halo_m, "GAP_MAX_M must stay inside the halo or the gap fill stops being tiling-independent"
    halo = int(np.ceil(halo_m / cell)); counts = np.zeros(6, "i8"); aux = np.zeros(3, "i8"); seams = []
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, crs=CFG.CRS_METRIC, transform=transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(dem_path, "w", dtype="float32", nodata=-9999.0, **prof) as dd, rasterio.open(src_path, "w", dtype="uint8", nodata=0, **prof) as ss:
        for r0 in range(0, shape[0], tile):
            for c0 in range(0, shape[1], tile):
                r1, c1 = min(r0 + tile, shape[0]), min(c0 + tile, shape[1]); hr0, hc0 = max(r0 - halo, 0), max(c0 - halo, 0); hr1, hc1 = min(r1 + halo, shape[0]), min(c1 + halo, shape[1])
                htr = transform * rasterio.Affine.translation(hc0, hr0)
                core = (slice(r0 - hr0, r1 - hr0), slice(c0 - hc0, c1 - hc0))
                dem, src, seam, ax = compose_tile((hr1 - hr0, hc1 - hc0), htr, cell, core=core); d = dem[core]; sc = src[core]
                win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0); dd.write(np.nan_to_num(d, nan=-9999.0), 1, window=win); ss.write(sc, 1, window=win)
                counts += np.bincount(sc.ravel(), minlength=6)[:6]; aux += ax; seams.append(seam)
        dd.update_tags(**tags); ss.update_tags(**tags)
    return counts, np.concatenate(seams) if seams else np.array([]), aux


def write(path: Path, arr, transform, dtype, nodata, tags):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype, crs=CFG.CRS_METRIC, transform=transform, nodata=nodata, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(np.nan_to_num(arr, nan=nodata) if dtype == "float32" else arr, 1); ds.update_tags(**tags)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--zones", nargs="*", default=list(ZN)); ap.add_argument("--no-union", action="store_true")
    ap.add_argument("--gate", choices=("basin", "corrected", "height"), default="basin",
                    help="p55e: positive basin membership (default); 'corrected' = the p55b p64 rule inside the 60 m buffer; 'height' = the original FABDEM-height arbiter")
    a = ap.parse_args()
    global GATE; GATE = a.gate
    print(f"bed gate: {GATE}" + ("" if GATE == "height" else f"  ({ISLAND_MASK.name}, {'present' if ISLAND_MASK.exists() else 'MISSING -> island veto inactive'})"), flush=True)
    rows = []; tags = dict(vertical_datum="EVRF2019", sources="1 zone bed DEM v2 (p53) | 2 reservoir bed DEM (hist20), inside the registry pool polygon only | 3 FABDEM->EVRF2019 (p56) | 4 FABDEM offset-tapered within 100 m of a bathymetric edge, |step| <= 1 m | 5 former-pool gap fill, water-surface cells only", gate=f"bed admitted only inside the registry pool polygon (no buffer), vetoed by the p64 island class and by FABDEM (nearest) > {POOL_WS_MAX_M} m; seam step <= {MAX_SEAM_STEP_M} m; gap fill radius <= {GAP_MAX_M} m", producer="p55_seamless_dem.py", bed_gate=a.gate)
    def row(name, cell, counts, seam, aux):
        a_ = cell * cell / 1e6
        return dict(zone=name, cell_m=cell, km2_zone_bed=round(counts[1] * a_, 1), km2_pool_bed=round(counts[2] * a_, 1), km2_pool_gap_fill=round(counts[5] * a_, 1), km2_fabdem=round((counts[3] + counts[4]) * a_, 1), km2_feather=round(counts[4] * a_, 1),
                    km2_pool_terrain_kept=round(aux[0] * a_, 1), km2_pool_ws_as_fabdem=round(aux[1] * a_, 1), km2_core_gap=round(aux[2] * a_, 1), seam_n=len(seam),
                    seam_median_bathy_minus_fabdem=round(float(np.median(seam)), 2) if len(seam) else np.nan, seam_p10=round(float(np.percentile(seam, 10)), 2) if len(seam) else np.nan, seam_p90=round(float(np.percentile(seam, 90)), 2) if len(seam) else np.nan)
    for zone in a.zones:
        with rasterio.open(CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif") as ds:
            shape, tr, cell = (ds.height, ds.width), ds.transform, ds.res[0]
        counts, seam, aux = compose_to_files(OUT / f"{zone}_dem_evrf2019_20m.tif", OUT / f"{zone}_dem_source_20m.tif", shape, tr, cell, tags)
        rows.append(row(zone, cell, counts, seam, aux)); print(rows[-1], flush=True)
    if not a.no_union:
        bounds = []
        for zone in ZN:
            with rasterio.open(CFG.BULK_ROOT / "terrain" / zone / "fabdem_evrf2019_20m.tif") as ds:
                bounds.append(ds.bounds)
        x0 = min(b.left for b in bounds); x1 = max(b.right for b in bounds); y0 = min(b.bottom for b in bounds); y1 = max(b.top for b in bounds); cell = 50.0
        tr = from_origin(np.floor(x0 / cell) * cell, np.ceil(y1 / cell) * cell, cell, cell); shape = (int(np.ceil((y1 - y0) / cell)) + 1, int(np.ceil((x1 - x0) / cell)) + 1)
        counts, seam, aux = compose_to_files(OUT / "dem_seamless_evrf2019_50m.tif", OUT / "dem_seamless_source_50m.tif", shape, tr, cell, tags)
        rows.append(row("UNION_50m", cell, counts, seam, aux))
    sp = CFG.TABLES / "p55_seam_check.csv"; S = pd.concat([pd.read_csv(sp), pd.DataFrame(rows)], ignore_index=True).drop_duplicates("zone", keep="last") if sp.exists() else pd.DataFrame(rows); S.to_csv(sp, index=False); print(S.to_string(index=False))
    if a.no_union:
        return
    with rasterio.open(OUT / "dem_seamless_evrf2019_50m.tif") as ds:
        dem = ds.read(1, out_shape=(ds.height // 4, ds.width // 4)).astype("f4"); dem[dem == ds.nodata] = np.nan; ext = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top); cell = ds.res[0]
    with rasterio.open(OUT / "dem_seamless_source_50m.tif") as ds:
        src = ds.read(1, out_shape=(ds.height // 4, ds.width // 4))
    k = 4; d = dem; s = src; tr = None
    # figure: union mosaic (shaded) + source map
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    ls = LightSource(315, 45); rgb = ls.shade(np.where(np.isfinite(d), np.clip(d, -20, 60), 0), cmap=plt.get_cmap("terrain"), vmin=-20, vmax=60, blend_mode="soft", vert_exag=3, dx=cell * k, dy=cell * k); rgb[..., 3] = np.isfinite(d)
    axes[0].imshow(rgb, extent=ext); axes[0].set_title("Seamless DEM, m EVRF2019 (−20..60 m colour range; bathymetry where surveyed, FABDEM elsewhere)"); axes[0].set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(-20, 60)); plt.colorbar(sm, ax=axes[0], fraction=0.025, label="m EVRF2019")
    cm = ListedColormap(["#ffffff", "#08519c", "#6baed6", "#d9d9d9", "#fdae61", "#9e9ac8"]); axes[1].imshow(np.where(s > 0, s, np.nan), extent=ext, cmap=cm, vmin=-0.5, vmax=5.5, interpolation="nearest"); axes[1].set_aspect("equal")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(fc="#08519c", label="1 zone bed DEM v2 (p53)"), Patch(fc="#6baed6", label="2 reservoir bed DEM"), Patch(fc="#d9d9d9", label="3 FABDEM → EVRF2019"), Patch(fc="#fdae61", label="4 FABDEM feathered (100 m seam)"), Patch(fc="#9e9ac8", label="5 former-pool gap fill (bed ↔ 16 m shoreline)")], loc="lower right"); axes[1].set_title("source of each cell")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p55_seamless_dem.png", dpi=90); print("-> outputs/figures/p55_seamless_dem.png")


if __name__ == "__main__":
    main()
