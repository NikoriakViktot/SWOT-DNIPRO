#!/usr/bin/env python
"""P42 -- the floodplain below the Kakhovka dam: domain, the June-2023 flood as rasters, HAND (plan 15, WP2).

Resolves the registry domain `below_dam_floodplain` (UNRESOLVED since the registry was created) from
two documented sources, and writes the observed June-2023 flood as per-event rasters:

  (a) OBSERVED envelope: `ZONE4_FLOOD_ENVELOPE_CANONICAL` + `ZONE2_FLOOD_ENVELOPE_CANONICAL`
      (data/processed/domains/zone{4,2}_flood_composite_utm.geojson; Sentinel-1 RTC, M3, June 2023) and the
      per-event water masks of the three June-2023 S1 caches (`per_scene_water.npz`, 11 events 06-01..06-30).
  (b) HYDRAULIC extent: HAND from FABDEM (terrain/<ZONE>/fabdem_20m.tif) with WhiteboxTools
      (breach_depressions -> d8 -> flow accumulation; "streams" = the PRE-BREACH water surface
      (`water_prebreach` layers) UNION accumulation >= STREAM_CELLS), then HAND < h0 where h0 is the smallest of
      H0_CANDIDATES that covers >= COVER_TARGET of the observed envelope cells (sensitivity table written).
  Domain = ((a) U HAND<h0) restricted to ZONE_4 U ZONE_2, minus the reservoir pool polygon (buffered 200 m) and
  minus everything east of the Kakhovka dam (x > dam_x + 1 km; CFG.KAKHOVKA_DAM), i.e. below the dam only.
  The DniproHES axis file (`damb_DniproGES`) is the UPSTREAM end of the reservoir and is not used here.

Outputs (bulk: $BULK_ROOT/floodplain/<ZONE>/, vector + tables in the repo):
  <ZONE>_hand_m.tif, <ZONE>_streams.tif, <ZONE>_flood_extent_2023-06-dd.tif (0/1/255), <ZONE>_flood_n_events.tif,
  <ZONE>_flood_max_envelope.tif, <ZONE>_flood_first_seen_idx.tif / _last_seen_idx.tif, <ZONE>_domain_mask.tif
  data/processed/domains/below_dam_floodplain_utm.geojson (EPSG:32636)
  outputs/tables/p42_hand_sensitivity.csv, p42_domain_provenance.csv, outputs/figures/p42_below_dam_floodplain_check.png
The registry entry is NOT written by this script: check the PNG on a basemap first, then add the YAML block it prints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import shape, box
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP
from swot_dnipro import spatial_domains as SD

ZONES = {"ZONE_4_DAM_TO_KHERSON_FLOODWAY": dict(n=4, caches=("ZONE_4_FLOODWAY_june2023_s20", "KAKHOVKA_DAM_TO_KHERSON_flood_june2023"), comp="zone4_flood_composite_utm.geojson", env="ZONE4_FLOOD_ENVELOPE_CANONICAL"),
         "ZONE_2_KHERSON_DELTA": dict(n=2, caches=("ZONE_2_KHERSON_DELTA_flood_june2023",), comp="zone2_flood_composite_utm.geojson", env="ZONE2_FLOOD_ENVELOPE_CANONICAL")}
OUT = CFG.BULK_ROOT / "floodplain"
STREAM_CELLS = None            # streams = PRE-BREACH WATER ONLY (Dnipro, delta channels); accumulation-based gullies made HAND<5 m cover 5 100 km2 of terrace (v3)
DIST_MAX_M = 10000.0           # hydraulic extent capped at 10 km from pre-breach water (terrain/<ZONE>/dist_ref_water_m.tif)
MIN_EVENTS = 2                 # a cell counts as OBSERVED flood only if wet in >= 2 S1 events (M3 single-scene false water, P31)
INHULETS_BUFFER_M = 1500.0     # v7 (user 2026-09-19): the Inhulets valley (subzone INHULETS_TRIBUTARY) is NOT the Dnipro floodplain - cut from streams and domain
USE_CHRONIC = False          # v9 (2026-09-19): OFF. zone*_p_water is built from the June-2023 flood scenes, so P>=0.5 & PRE water<10 % is the real flood (checked on the p42b map: it overlaps S2 water). A true chronic-dark-land layer needs a PRE-breach S1 stack (p31/p38 caches) -- open task.
CHRONIC_P = 0.50               # v7: chronic radar-dark land = S1 P_water >= 0.5 (p0v/p0w zone*_p_water) while S2 PRE water share < 10 % and WorldCover != water
S2_JUNE = ("2023-06-08", "2023-06-20", "2023-06-30")   # optical June-2023 water (p25 water3), the trustworthy open-water record
H0_CANDIDATES = (2.0, 3.0, 5.0, 8.0, 10.0)
COVER_TARGET = 0.95            # v8: per-reach h0 = smallest h with HAND<h covering >= 95 % of the OPTICAL June water of that reach
H0_FIXED = 4.5                 # v18: ZONE_2 fallback = Kherson peak 5.56 m - 1 m river floor. v5-v7 (superseded by the v8 profile; kept for the sensitivity table). User 2026-09-19: the stage rise
                               # was ~5.2 m at Kherson (gauge 80805: 0.32 -> 5.56 m abs on 06-08) but far larger near the dam, so one
                               # fixed h0 over 90 km of river has no physical meaning -> h0(x) by distance from the dam.
H0_BIN_M = 10000.0             # v8: reach bins by straight-line distance from the Kakhovka dam (SWORD chainage not needed at 10 km resolution)
H0_MIN, H0_MAX = 2.0, 10.0     # v18: bounds relative to the 1 m river floor -> peak WSE 3-11 m (Kherson gauge 5.56 m; near the dam ~9-10 m)
H0_MIN_CELLS = 5000            # v9: bins from < 5000 optical cells (S2 June covers only part of the reach) are interpolated
INH_WIDE_BUFFER_M = 12000.0    # v11 (user 2026-09-19): the thin green strips WEST of the Inhulets (x 465-480 km E, y 5172-5221 km N) are the Inhulets valley, 0-15 km from its axis
INH_WIDE_FROM_Y = 5185000.0    # ... applied only north of the confluence with the Dnipro (a 12 km buffer lower down would cut the Dnipro channel itself)
CUT_RECTS = ((462000.0, 5175000.0, 481000.0, 5300000.0),   # v13: west of the Inhulets mouth, north of the user's line
             (481000.0, 5180000.0, 503000.0, 5300000.0),   # v13: the Inhulets valley itself (east side), north of the Dnipro floodplain
             (503000.0, 5187000.0, 530000.0, 5300000.0))   # v14: isolated terrace fragments (48 + 9 + 8 km2) NE of the Inhulets mouth, not connected to the Dnipro floodplain
APPENDIX_NORTH_Y = 5175000.0   # v12 (user drew the cut line, 2026-09-19): north of this latitude and west of the Inhulets mouth ...
APPENDIX_EAST_X = 481000.0     # ... (x < 481 km E) nothing is Dnipro floodplain -- applied to BOTH zones   # v10 (user 2026-09-19): the northern tongue of ZONE_2 (strip x<476 km, y>Kherson latitude+~9 km, along the Inhulets) is not the Dnipro delta/estuary floodplain -> cut
SWOT_PEAK_CSV = Path(__file__).resolve().parents[1] / "outputs/tables/p59_swot_node_peak.csv"   # v20: observed peak WSE per SWORD node (SWOT KaRIn, 1-day orbit, 06-07..06-11.2023)
SWOT_MARGIN_M = 0.5            # v20: added to the node peak: SWOT 11:00 UTC daily sample under-reads the gauge peak (Kherson 4.98 vs 5.78 m) and the median node sits -0.21 m below the gauge; sensitivity 0.3/0.8 in the journal
OBSERVED_VETO = "ceiling_only"   # set from --observed-veto; see the veto block in main()
SWOT_MAX_DIST_M = 15000.0      # v20: cells farther than this from any SWOT Dnipro node fall back to the gauge cap
H0_MAX_BELOW_GAUGE = 4.5       # v19: downstream of the Kherson gauge the peak WSE cannot exceed the gauge peak (5.56 m) -> h0 <= 5.56 - 1 m river floor
RIVER_LEVEL_M = 1.0            # v17: absolute ceiling FABDEM < h0(x) + RIVER_LEVEL_M (lower Dnipro water level 0.3-1 m BS77/EGM2008 along the reach)
PRE_WATER_MAX_ELEV_M = 3.5     # v16: pre-breach water bodies above this FABDEM elevation (m, EGM2008) are terrace canals/ponds, not floodplain streams
CELL = 20.0


def h0_profile(h: np.ndarray, opt: np.ndarray, ddam: np.ndarray, zone_mask: np.ndarray, h0_max: float = H0_MAX) -> tuple[np.ndarray, list[dict]]:
    """h0 per distance-from-dam bin: smallest h (0.5 m steps) with HAND<h covering >= COVER_TARGET of the optical June water cells
    of that bin. Returns the per-cell h0 limit and the profile rows (for p42_h0_profile.csv)."""
    ok = np.isfinite(h) & opt & zone_mask & np.isfinite(ddam)
    b = np.floor(ddam / H0_BIN_M); nb = int(np.nanmax(b[zone_mask & np.isfinite(ddam)])) + 1
    grid = np.arange(H0_MIN, h0_max + 0.01, 0.5); rows = []; h0s = np.full(nb, np.nan)
    for i in range(nb):
        sel = ok & (b == i); n = int(sel.sum())
        if n >= H0_MIN_CELLS:
            hv = h[sel]; cov = [(hv < g).mean() for g in grid]; j = next((k for k, c in enumerate(cov) if c >= COVER_TARGET), len(grid) - 1)
            h0s[i] = grid[j]; rows.append(dict(bin_km=i * H0_BIN_M / 1000, n_optical_cells=n, h0_m=float(grid[j]), coverage=round(float(cov[j]), 4), cov_at_5m=round(float((hv < 5).mean()), 4), interpolated=False))
        else:
            rows.append(dict(bin_km=i * H0_BIN_M / 1000, n_optical_cells=n, h0_m=np.nan, coverage=np.nan, cov_at_5m=np.nan, interpolated=True))
    good = np.isfinite(h0s)
    if good.sum() == 0:
        h0s[:] = min(H0_FIXED, h0_max)
    else:
        h0s = np.interp(np.arange(nb), np.flatnonzero(good), h0s[good])
    for r, v in zip(rows, h0s):
        r["h0_m"] = float(v)
    lim = np.full(h.shape, np.nan, "f4"); m = zone_mask & np.isfinite(ddam)
    lim[m] = h0s[np.clip(b[m].astype(int), 0, nb - 1)]
    return lim, rows


def cache_events(zone: str, G: dict) -> dict:
    """Per-event water masks (0/1/255) on the zone grid from every listed cache (first cache wins per event)."""
    ev: dict[str, np.ndarray] = {}
    for c in ZONES[zone]["caches"]:
        f = CFG.S1_CACHE / c / "per_scene_water.npz"
        if not f.exists():
            print(f"  cache {c}: no per_scene_water.npz"); continue
        z = np.load(f, allow_pickle=True)
        shp = tuple(int(v) for v in z["shape"]); cell = float(z["cell"]); x0 = float(z["x0"]); y1 = float(z["y1"])
        tr = from_origin(x0, y1, cell, cell)
        keys = [k for k in z.keys() if k[:4] == "2023" and not k.startswith("valid_")]

        def unpack(key):                      # p0v/p0w store np.packbits(bool mask) row-major
            return np.unpackbits(z[key], count=shp[0] * shp[1]).astype(bool).reshape(shp)

        for k in keys:
            if k in ev:
                continue
            w = unpack(k).astype("u1")
            valid = unpack(f"valid_{k}") if f"valid_{k}" in z else np.ones(shp, bool)
            src = np.where(valid, w, 255).astype("u1")
            dst = np.full((G["ny"], G["nx"]), 255, "u1")
            reproject(source=src, destination=dst, src_transform=tr, src_crs=CFG.CRS_METRIC, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC,
                      resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
            ev[k] = dst
        print(f"  cache {c}: {len(keys)} events on grid {shp} cell {cell}")
    return dict(sorted(ev.items()))


def to_grid(path: Path, G: dict, nodata=np.nan, resampling=Resampling.nearest) -> np.ndarray:
    """Any raster (older frame, other size) -> the zone grid."""
    with rasterio.open(path) as src:
        a = src.read(1).astype("f4")
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
        dst = np.full((G["ny"], G["nx"]), np.nan, "f4")
        reproject(source=a, destination=dst, src_transform=src.transform, src_crs=src.crs, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def hand(zone: str, G: dict, out_dir: Path, prebreach_water: np.ndarray) -> np.ndarray:
    from whitebox import WhiteboxTools
    tmp = out_dir / "_wbt"; tmp.mkdir(parents=True, exist_ok=True)
    dem_in = CFG.BULK_ROOT / "terrain" / zone / "fabdem_20m.tif"
    with rasterio.open(dem_in) as src:
        dem = src.read(1).astype("f4"); prof = src.profile.copy()
        assert (src.height, src.width) == (G["ny"], G["nx"]), "FABDEM frame must be the zone grid"
    dem[dem == prof.get("nodata", -9999)] = np.nan
    dem = np.maximum(dem, RIVER_LEVEL_M)                     # v18: FABDEM on water is a -2..-5 m artefact; HAND was inflated ~5 m (optical June water: DEM p50 0.8 m but HAND 6.8 m at Kherson)
    dem = np.where(np.isfinite(dem), dem, -9999).astype("f4")
    prof.update(dtype="float32", nodata=-9999, compress="deflate")
    with rasterio.open(tmp / "dem.tif", "w", **prof) as d:
        d.write(dem, 1)
    w = WhiteboxTools(); w.set_verbose_mode(False); w.set_working_dir(str(tmp))
    assert w.breach_depressions("dem.tif", "breached.tif", fill_pits=True) == 0
    assert w.d8_pointer("breached.tif", "d8.tif") == 0
    assert w.d8_flow_accumulation("breached.tif", "acc.tif", out_type="cells") == 0
    with rasterio.open(tmp / "acc.tif") as s:
        acc = s.read(1)
    streams = (((acc >= STREAM_CELLS) if STREAM_CELLS else np.zeros(acc.shape, bool)) | (prebreach_water == 1)).astype("i4")
    p2 = prof.copy(); p2.update(dtype="int32", nodata=-1)
    with rasterio.open(tmp / "streams.tif", "w", **p2) as d:
        d.write(streams, 1)
    assert w.elevation_above_stream("breached.tif", "streams.tif", "hand.tif") == 0
    with rasterio.open(tmp / "hand.tif") as s:
        h = s.read(1).astype("f4"); nd = s.nodata
    h[(h == nd) | ~np.isfinite(h)] = np.nan
    with rasterio.open(out_dir / f"{zone}_hand_m.tif", "w", **SP._profile(G, 1, "float32", -9999)) as d:
        d.write(np.nan_to_num(h, nan=-9999), 1)
        d.update_tags(quantity="HAND [m] above the nearest stream cell (FABDEM breached, D8)", streams=f"pre-breach S1 water" + (f" UNION D8 accumulation >= {STREAM_CELLS} cells" if STREAM_CELLS else " only"), producer="p42_below_dam_floodplain.py", tool="WhiteboxTools 2.4.0")
    SP.write_uint8(out_dir / f"{zone}_streams.tif", streams.astype("u1"), G, dict(values="1 stream/pre-breach water cell", producer="p42_below_dam_floodplain.py"), nodata=255)
    return h


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--zones", nargs="*", default=list(ZONES))
    ap.add_argument("--observed-veto", choices=("ceiling_only", "hand_and_ceiling"), default="ceiling_only",
                    help="what may delete an OBSERVED S1 flood cell. ceiling_only (v21, default): the FABDEM ceiling, chronic, Inhulets and pre-breach water. hand_and_ceiling (v20): additionally HAND < h0(x), which removed 41.6 km2 of observed delta water because HAND is unreliable where the mapped drainage is incomplete.")
    a = ap.parse_args()
    global OBSERVED_VETO; OBSERVED_VETO = a.observed_veto
    print(f"observed-flood veto: {OBSERVED_VETO}", flush=True)
    pool = SD.load_utm("reservoir_full_pool_prebreach")
    dam = gpd.GeoSeries([__import__("shapely").geometry.Point(*CFG.KAKHOVKA_DAM)], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]
    east_cut = dam.x + 1000.0
    parts, sens, prov, h0rows = [], [], [], []
    t0 = time.time()
    for zone in a.zones:
        z = ZONES[zone]; G = SP.zone_grid(zone, CELL); out_dir = OUT / zone; out_dir.mkdir(parents=True, exist_ok=True)
        comp = gpd.read_file(ROOT / "data/processed/domains" / z["comp"]).to_crs(CFG.CRS_METRIC)
        env_poly = unary_union(comp[comp.layer == z["env"]].geometry)
        pre_poly = unary_union(comp[comp.layer == "water_prebreach"].geometry)
        env = features.rasterize([(env_poly, 1)], out_shape=(G["ny"], G["nx"]), transform=G["transform"], fill=0, dtype="uint8")
        pre = features.rasterize([(pre_poly, 1)], out_shape=(G["ny"], G["nx"]), transform=G["transform"], fill=0, dtype="uint8")
        zone_mask = G["inside"]
        inh = features.rasterize([(SD.load_subzone_utm("INHULETS_TRIBUTARY").buffer(INHULETS_BUFFER_M), 1)], out_shape=(G["ny"], G["nx"]), transform=G["transform"], fill=0, dtype="uint8").astype(bool)
        pre[inh] = 0                                   # Inhulets is not a Dnipro stream for HAND
        # v16 (user 2026-09-19, DEM check on fig08): pre-breach water on the TERRACE (irrigation canals, ponds on the sand arena at 5-15 m)
        # seeded HAND and let 10-15 m terrain into the domain. Keep only water bodies at floodplain level: component median FABDEM
        # < PRE_WATER_MAX_ELEV_M, plus the largest component (the Dnipro network) whatever its median.
        from scipy import ndimage
        with rasterio.open(CFG.BULK_ROOT / "terrain" / zone / "fabdem_20m.tif") as ds_:
            dem_ = ds_.read(1).astype("f4"); dem_[dem_ == ds_.nodata] = np.nan
        lab_, n_lab = ndimage.label(pre, structure=np.ones((3, 3))); idx_ = np.arange(1, n_lab + 1)
        med_ = np.array(ndimage.median(np.nan_to_num(dem_, nan=99.0), lab_, idx_)); size_ = np.bincount(lab_.ravel())[1:]
        keep_ = (med_ < PRE_WATER_MAX_ELEV_M) | (idx_ == idx_[np.argmax(size_)])
        drop_mask = np.isin(lab_, idx_[~keep_]); print(f"  pre-breach water: {n_lab} components, dropped {int((~keep_).sum())} terrace components ({drop_mask.sum()*0.0004:.0f} km2, median FABDEM >= {PRE_WATER_MAX_ELEV_M} m) from the HAND streams")
        pre[drop_mask] = 0
        n_ = z["n"]
        pw = to_grid(ROOT / f"outputs/rasters/zone{n_}/zone{n_}_p_water.tif", G)
        wf_pre = to_grid(ROOT / f"outputs/rasters/zone{n_}/zone{n_}_water_frac_PRE_BREACH_20m.tif", G)
        wf_pre[wf_pre == 255] = np.nan
        wc_p = CFG.BULK_ROOT / "worldcover_frames" / zone / "wc_2021_20m.tif"
        wc = to_grid(wc_p, G) if wc_p.exists() else np.full((G["ny"], G["nx"]), np.nan, "f4")
        chronic = (np.nan_to_num(pw, nan=0) >= CHRONIC_P) & (np.nan_to_num(wf_pre, nan=0) < 10) & (np.nan_to_num(wc, nan=0) != 80)
        chronic_diag = chronic.copy()
        if not USE_CHRONIC:
            chronic = np.zeros_like(chronic)
        SP.write_uint8(out_dir / f"{zone}_chronic_radar_dark_land.tif", chronic_diag.astype("u1"), G, dict(values="1 = S1 P_water>=0.5 while S2 PRE water<10 % and WorldCover!=80 (radar-dark sand/fields, P31)", producer="p42_below_dam_floodplain.py"), nodata=255)
        print(f"{zone}: envelope {env.sum()*0.0004:.0f} km2, pre-breach water {pre.sum()*0.0004:.0f} km2, chronic radar-dark {chronic.sum()*0.0004:.0f} km2, Inhulets cut {inh.sum()*0.0004:.0f} km2  ({time.time()-t0:.0f}s)")
        # optical June-2023 water (S2 water3), outside pre-breach water
        s2w = np.zeros((G["ny"], G["nx"]), bool); s2d = []
        for d in S2_JUNE:
            f = CFG.BULK_ROOT / "zone_spectral" / zone / f"{d}_water3.tif"
            if f.exists():
                with rasterio.open(f) as ds:
                    s2w |= ds.read(1) == 1
                s2d.append(d)
        # per-event flood rasters, CLEANED: S1 water AND HAND<h0 AND not chronic AND not Inhulets (HAND computed below; apply after)
        h = hand(zone, G, out_dir, pre)
        with rasterio.open(CFG.BULK_ROOT / "terrain" / zone / "dist_ref_water_m.tif") as ds:
            dist = ds.read(1).astype("f4"); dist[dist == ds.nodata] = np.nan
        ev = cache_events(zone, G)
        xs_c = G["x0"] + (np.arange(G["nx"]) + 0.5) * CELL; ys_c = G["y1"] - (np.arange(G["ny"]) + 0.5) * CELL
        ddam = np.sqrt((xs_c[None, :] - dam.x) ** 2 + (ys_c[:, None] - dam.y) ** 2).astype("f4")
        flood_opt_raw = s2w & (pre == 0) & ~inh & ~chronic
        raw_stack = np.stack([ev[k] for k in ev if k > "2023-06-06"]) if ev else None
        nwet_raw = ((raw_stack == 1) & (raw_stack != 255)).sum(0) if raw_stack is not None else np.zeros(h.shape, "u1")
        opt_cal = flood_opt_raw & (nwet_raw >= 1)                       # v19: calibrate h0 on TWO-SENSOR agreement (S2 June water also S1-wet in >= 1 event) -> optical false water on dark fields/terrace cannot inflate the profile
        print(f"  h0 calibration cells: optical {flood_opt_raw.sum()*0.0004:.0f} km2 -> also S1-wet {opt_cal.sum()*0.0004:.0f} km2")
        h0cap = H0_MAX_BELOW_GAUGE if zone == "ZONE_2_KHERSON_DELTA" else H0_MAX
        hlim_2s, prof_rows = h0_profile(h, opt_cal, ddam, zone_mask, h0_max=h0cap)          # kept as a diagnostic (two-sensor calibration, v19)
        # v20 (p59): the flood ceiling is the OBSERVED peak water surface -- SWOT KaRIn daily WSE of the Dnipro nodes during the event,
        # smoothed along the channel; every cell takes the peak of its nearest node (+ margin). HAND limit = peak WSE - river floor.
        from scipy.spatial import cKDTree
        nod = pd.read_csv(SWOT_PEAK_CSV).dropna(subset=["H_peak_smooth"]); tree_n = cKDTree(np.c_[nod.x.values, nod.y.values])
        YY, XX = np.meshgrid(ys_c, xs_c, indexing="ij"); dn, ii = tree_n.query(np.c_[XX.ravel(), YY.ravel()]); dn = dn.reshape(h.shape); ii = ii.reshape(h.shape)
        wse_peak = (nod.H_peak_smooth.values[ii] + SWOT_MARGIN_M).astype("f4"); far = dn > SWOT_MAX_DIST_M
        wse_peak[far] = np.minimum(wse_peak[far], H0_MAX_BELOW_GAUGE + RIVER_LEVEL_M)      # beyond SWOT coverage (outer delta): never above the Kherson peak
        hlim = np.where(zone_mask, wse_peak - RIVER_LEVEL_M, np.nan).astype("f4")
        print(f"  v20 SWOT ceiling: peak WSE {np.nanpercentile(wse_peak[zone_mask], 5):.1f}..{np.nanpercentile(wse_peak[zone_mask], 95):.1f} m (p5-p95), {far[zone_mask].mean()*100:.0f} % of cells > {SWOT_MAX_DIST_M/1e3:.0f} km from a SWOT node; two-sensor h0 profile kept only as diagnostic")
        for r in prof_rows:
            r["hlim_swot_median_in_bin"] = float(np.nanmedian(hlim[zone_mask & (np.floor(ddam / H0_BIN_M) == r["bin_km"] * 1000 / H0_BIN_M)])) if np.isfinite(hlim).any() else np.nan
        ceiling = np.isfinite(dem_) & (dem_ < hlim + RIVER_LEVEL_M)          # v17/v20: FABDEM below the observed peak water surface (fig08 DEM histogram): HAND alone still let 10-15 m terrace in where a low-lying stream cell was reachable
        # hoisted above the optical gate (v21): both observed terms now share one elevation criterion.
        # v21: the same rule as for the S1 term. v15 added a HAND gate here because optical water on the 20-45 m terrace
        # (dark fields and hill shadow) was entering the domain; that concern is real, but the criterion that answers it
        # is the ELEVATION ceiling -- is this ground below the observed peak water surface -- not HAND, which asks how
        # far the cell sits above the nearest MAPPED channel and is therefore unreliable exactly where the delta's
        # channel network is incompletely mapped. Under the v20 gate Sentinel-2 saw 81 km2 of June water in ZONE_2 and
        # 20 km2 survived; a terrain model was deleting three quarters of an observation.
        if OBSERVED_VETO == "hand_and_ceiling":
            flood_opt = flood_opt_raw & np.isfinite(h) & (h < hlim)
            gate_name = "HAND veto"
        else:
            flood_opt = flood_opt_raw & ceiling
            gate_name = "elevation ceiling"
        print(f"  optical June water: raw {flood_opt_raw.sum()*0.0004:.0f} km2 -> after the {gate_name} {flood_opt.sum()*0.0004:.0f} km2")
        for r in prof_rows:
            r["zone"] = zone
        h0rows.extend(prof_rows)
        print(f"  h0 profile (km from dam -> m): " + ", ".join(f"{r['bin_km']:.0f}:{r['h0_m']:.1f}{'*' if r['interpolated'] else ''}" for r in prof_rows))
        with rasterio.open(out_dir / f"{zone}_hand_limit_m.tif", "w", **SP._profile(G, 1, "float32", -9999)) as d:
            d.write(np.nan_to_num(hlim, nan=-9999), 1); d.update_tags(quantity="HAND limit [m] = SWOT peak WSE of the nearest Dnipro node (p59, +0.3 m) - 1 m river floor (v20); two-sensor h0 profile only in p42_h0_profile.csv", producer="p42_below_dam_floodplain.py")
        # v21 (user 2026-09-21: "розширювати на дельту обов'язково"). The veto used to gate the OBSERVED flood on
        # HAND < h0 as well as on the FABDEM ceiling. Measured consequence in the delta: of the 41.6 km2 of S1 new water
        # that the domain dropped in ZONE_2, only 1.6 % had HAND <= h0(x), while 69.2 % was BELOW the ceiling -- so HAND,
        # not elevation, was doing the cutting, and p42's own `flood_n_events` came out 0 at the median on cells where
        # three peak scenes saw water. HAND is height above the NEAREST MAPPED drainage, and in a delta the mapped
        # drainage is incomplete, so HAND is measured to a distant channel and comes out too high; that dependence on
        # flowpath representation is exactly the structural limit Johnson et al. 2019 (SRC-55) describe.
        # THE RULE: a terrain model may EXTEND the domain beyond what was observed, it may not delete an observation.
        # The FABDEM ceiling stays on the observed term -- it is what keeps M3's false water off the 10-15 m terrace
        # (it still removes 4.8 km2 of >10 m terrace here) -- and so do the chronic, Inhulets and pre-breach-water vetoes.
        # `--observed-veto hand_and_ceiling` restores the v20 behaviour for comparison.
        if OBSERVED_VETO == "hand_and_ceiling":
            veto = ~(np.isfinite(h) & (h < hlim) & ceiling) | chronic | inh | (pre == 1)
        else:
            veto = ~ceiling | chronic | inh | (pre == 1)      # v9: pre-breach water (river, pool, channels) is not flood
        for k in ev:
            m = ev[k]; m = np.where((m == 1) & veto, 0, m).astype("u1"); ev[k] = m
        if ev:
            stack = np.stack(list(ev.values()))
            obs = stack != 255; nwet = ((stack == 1) & obs).sum(0).astype("u1"); nobs = obs.sum(0).astype("u1")
            maxenv = np.where(nobs > 0, (nwet > 0).astype("u1"), 255).astype("u1")
            first = np.full(nwet.shape, 255, "u1"); last = np.full(nwet.shape, 255, "u1")
            for i in range(len(ev)):
                wet = stack[i] == 1
                first[(first == 255) & wet] = i; last[wet] = i
            for k, m in ev.items():
                SP.write_uint8(out_dir / f"{zone}_flood_extent_{k}.tif", m, G, dict(values="0 dry 1 water 255 not observed", source="S1 RTC M3 (p0r/p0x caches)", event=k, producer="p42_below_dam_floodplain.py"))
            SP.write_uint8(out_dir / f"{zone}_flood_n_events.tif", nwet, G, dict(values="events classed water (of n observed)", events="|".join(ev), producer="p42_below_dam_floodplain.py"), nodata=255)
            SP.write_uint8(out_dir / f"{zone}_flood_n_observed.tif", nobs, G, dict(values="events observed", producer="p42_below_dam_floodplain.py"), nodata=255)
            SP.write_uint8(out_dir / f"{zone}_flood_max_envelope.tif", maxenv, G, dict(values="1 water in >=1 event; 0 never; 255 never observed", producer="p42_below_dam_floodplain.py"))
            SP.write_uint8(out_dir / f"{zone}_flood_first_seen_idx.tif", first, G, dict(values="index into events tag (255 never wet)", events="|".join(ev), producer="p42_below_dam_floodplain.py"))
            SP.write_uint8(out_dir / f"{zone}_flood_last_seen_idx.tif", last, G, dict(values="index into events tag", events="|".join(ev), producer="p42_below_dam_floodplain.py"))
            print(f"  {len(ev)} events (cleaned) -> rasters; max envelope {(maxenv==1).sum()*0.0004:.0f} km2 vs canonical {env.sum()*0.0004:.0f} km2")
        persistent = (((nwet >= MIN_EVENTS) if ev else (env == 1)) | flood_opt) & ~inh
        SP.write_uint8(out_dir / f"{zone}_flood_envelope_clean.tif", np.where(zone_mask, persistent.astype("u1"), 255).astype("u1"), G,
                       dict(values="1 = flooded June 2023: S1 water in >=2 events after terrain/chronic veto, OR Sentinel-2 water on " + "|".join(s2d) + " outside pre-breach water", producer="p42_below_dam_floodplain.py"))
        SP.write_uint8(out_dir / f"{zone}_flood_optical_june2023.tif", np.where(zone_mask, flood_opt.astype("u1"), 255).astype("u1"), G, dict(values="1 = S2 water on " + "|".join(s2d) + " outside pre-breach water", producer="p42_below_dam_floodplain.py"))
        print(f"  persistent S1 {(nwet >= MIN_EVENTS).sum()*0.0004:.0f} km2, optical S2 June {flood_opt.sum()*0.0004:.0f} km2, union {persistent.sum()*0.0004:.0f} km2")
        ok = np.isfinite(h) & persistent & zone_mask
        chosen = None
        for h0 in H0_CANDIDATES:
            cov = float((h[ok] < h0).mean()) if ok.any() else np.nan
            area = float(((h < h0) & zone_mask & np.isfinite(h)).sum() * 0.0004)
            area_d = float(((h < h0) & (dist <= DIST_MAX_M) & zone_mask & np.isfinite(h)).sum() * 0.0004)
            sens.append(dict(zone=zone, h0_m=h0, persistent_flood_km2=round(float(persistent.sum()) * 0.0004, 1), coverage_of_persistent=round(cov, 4), hand_lt_h0_km2=round(area, 1), hand_lt_h0_within_10km_km2=round(area_d, 1)))
            if chosen is None and cov >= COVER_TARGET:
                chosen = h0
        chosen = f"profile {min(r['h0_m'] for r in prof_rows):.1f}-{max(r['h0_m'] for r in prof_rows):.1f}"
        print(f"  HAND: h0 = {chosen} m (fixed-h0 coverage of flood cells for reference: {[s['coverage_of_persistent'] for s in sens if s['zone']==zone]})")
        dom = (persistent | (np.isfinite(h) & (h < hlim) & (dist <= DIST_MAX_M))) & zone_mask & ceiling
        print(f"  elevation ceiling h0(x)+{RIVER_LEVEL_M} m removes {float((~ceiling & zone_mask & np.isfinite(h) & (h < hlim)).sum())*0.0004:.0f} km2 of HAND-eligible cells")
        # below the dam only: drop the pool (+200 m) and everything east of dam_x + 1 km
        poolm = features.rasterize([(pool.buffer(200), 1)], out_shape=dom.shape, transform=G["transform"], fill=0, dtype="uint8").astype(bool)
        xs = G["x0"] + (np.arange(G["nx"]) + 0.5) * CELL
        ys_ = G["y1"] - (np.arange(G["ny"]) + 0.5) * CELL
        cut = np.zeros(dom.shape, bool)
        for (x0r, y0r, x1r, y1r) in CUT_RECTS:
            cut |= (ys_[:, None] >= y0r) & (ys_[:, None] < y1r) & (xs[None, :] >= x0r) & (xs[None, :] < x1r)
        print(f"  rectangle cuts {[(int(r[0]/1e3), int(r[1]/1e3), int(r[2]/1e3)) for r in CUT_RECTS]} km: -{float((dom & cut).sum() * 0.0004):.0f} km2")
        dom &= ~poolm & ~cut; dom[:, xs > east_cut] = False          # v13: no buffer polygons in the domain shape (inh only feeds the HAND streams)
        SP.write_uint8(out_dir / f"{zone}_domain_mask.tif", dom.astype("u1"), G, dict(values="1 below_dam_floodplain candidate cells", h0=str(chosen), producer="p42_below_dam_floodplain.py"), nodata=0)
        polys = [shape(g) for g, v in features.shapes(dom.astype("u1"), mask=dom, transform=G["transform"]) if v == 1]
        u = unary_union(polys)
        parts.append(u)
        prov.append(dict(zone=zone, envelope_layer=z["env"], envelope_km2=round(env.sum()*0.0004, 1), persistent_ge2_km2=round(float(persistent.sum())*0.0004, 1), optical_s2_km2=round(float(flood_opt.sum())*0.0004, 1), s1_events=len(ev), hand_h0_m=chosen, hand_streams="pre-breach water only", dist_cap_m=DIST_MAX_M,
                         domain_km2=round(dom.sum()*0.0004, 1), east_cut_m=round(east_cut), pool_excluded="reservoir_full_pool_prebreach buffer 200 m"))
    dom_geom = unary_union(parts)
    # drop slivers < 0.05 km2 (the p0r sieve), keep holes
    dom_geom = unary_union([g for g in (dom_geom.geoms if dom_geom.geom_type == "MultiPolygon" else [dom_geom]) if g.area >= 5e4])
    gdf = gpd.GeoDataFrame(dict(name=["below_dam_floodplain"], area_km2=[round(dom_geom.area / 1e6, 1)], source=["p42_below_dam_floodplain.py: observed June-2023 S1 flood envelope (canonical) U (HAND<h0 within 10 km of pre-breach water; FABDEM, WBT, streams = pre-breach water), persistent = wet in >=2 S1 events; ZONE_4 U ZONE_2, below the Kakhovka dam, pool excluded"],
                                built_utc=[time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())]), geometry=[dom_geom], crs=CFG.CRS_METRIC)
    outv = ROOT / "data/processed/domains/below_dam_floodplain_utm.geojson"
    gdf.to_file(outv, driver="GeoJSON")
    pd.DataFrame(sens).to_csv(CFG.TABLES / "p42_hand_sensitivity.csv", index=False)
    pd.DataFrame(prov).to_csv(CFG.TABLES / "p42_domain_provenance.csv", index=False)
    h0df = pd.DataFrame(h0rows); h0df.to_csv(CFG.TABLES / "p42_h0_profile.csv", index=False)
    # Kherson gauge (80805) stage rise for the profile check: daily 2023-06-05 vs max 2023-06-08
    try:
        gk = pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET); gk = gk[gk.stat_type == "daily"]; gk["date"] = pd.to_datetime(gk.date)
        pre_l = float(gk.loc[gk.date == "2023-06-05", "water_level_m_abs"].iloc[0]); pk = float(gk.loc[gk.date.between("2023-06-06", "2023-06-12"), "water_level_m_abs"].max())
        kh_rise = pk - pre_l
    except Exception as e:
        kh_rise = np.nan; print("  Kherson gauge not read:", e)
    # check figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw=dict(height_ratios=(3, 1)))
    kh = gpd.GeoSeries([__import__("shapely").geometry.Point(CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3])], crs="EPSG:4326").to_crs(CFG.CRS_METRIC).iloc[0]   # (id, name, lon, lat)
    for zn, s in h0df.groupby("zone"):
        ax2.step(s.bin_km + H0_BIN_M / 2000, s.h0_m, where="mid", label=f"h0(x) {zn[:6]}"); ax2.scatter(s.bin_km[s.interpolated] + H0_BIN_M / 2000, s.h0_m[s.interpolated], marker="x", color="k", zorder=3)
    dk = float(np.hypot(kh.x - dam.x, kh.y - dam.y) / 1000)
    ax2.axvline(dk, color="k", ls=":"); ax2.annotate(f"Kherson gauge: rise {kh_rise:.2f} m (06-05 -> peak)", (dk, H0_MAX - 1), fontsize=9)
    ax2.axhline(kh_rise, color="k", lw=0.6, ls="--"); ax2.set_xlabel("straight-line distance from the Kakhovka dam, km"); ax2.set_ylabel("HAND limit h0, m"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    for zn, col in (("ZONE_4_DAM_TO_KHERSON_FLOODWAY", "0.6"), ("ZONE_2_KHERSON_DELTA", "0.3")):
        gpd.GeoSeries([SD.load_utm(zn)], crs=CFG.CRS_METRIC).boundary.plot(ax=ax, color=col, lw=0.8)
    gpd.GeoSeries([pool], crs=CFG.CRS_METRIC).plot(ax=ax, color="#9ecae1", alpha=0.5, label="pool")
    gdf.plot(ax=ax, color="#31a354", alpha=0.6, label="below_dam_floodplain")
    for zone in a.zones:
        comp = gpd.read_file(ROOT / "data/processed/domains" / ZONES[zone]["comp"]).to_crs(CFG.CRS_METRIC)
        with rasterio.open(OUT / zone / f"{zone}_flood_envelope_clean.tif") as ds:
            fe = ds.read(1); ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
        ax.imshow(np.where(fe == 1, 1, np.nan), extent=ext, cmap="Reds", vmin=0, vmax=1.3, alpha=0.9, interpolation="nearest")
    ax.plot(dam.x, dam.y, "k^", ms=8); ax.annotate("Kakhovka dam", (dam.x, dam.y))
    ax.plot(kh.x, kh.y, "ks", ms=6); ax.annotate("Kherson", (kh.x, kh.y))
    ax.set_title(f"below_dam_floodplain {gdf.area_km2.iloc[0]} km2 (green); red = June-2023 flood (cleaned S1 >=2 events U S2 optical); blue = pool\nHAND limit = h0(x) profile calibrated on S2 June water (below)", fontsize=10); ax.set_aspect("equal")
    (ROOT / "outputs/figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "outputs/figures/p42_below_dam_floodplain_check.png", dpi=150, bbox_inches="tight")
    print(f"\n-> {outv} ({gdf.area_km2.iloc[0]} km2); figure outputs/figures/p42_below_dam_floodplain_check.png")
    print("Registry block to add to config/spatial_domains.yaml AFTER the basemap check:")
    print(json.dumps(dict(below_dam_floodplain=dict(source="data/processed/domains/below_dam_floodplain_utm.geojson", crs="EPSG:32636", status="RESOLVED 2026-09-18 by p42 (see p42_domain_provenance.csv)")), indent=2))


if __name__ == "__main__":
    main()
