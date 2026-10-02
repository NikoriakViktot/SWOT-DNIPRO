#!/usr/bin/env python
"""Phase 19 steps 5-9 — classify post-breach ATL13 water observations.

Pipeline
--------
1. SWORD v16  -> Dnipro main stem + authoritative chainage from the dam (not longitude)
2. ATL13 segments -> WSE (EGG2015 frame, permanent-tide harmonised), period, SWORD chainage
3. cached Sentinel-2 water masks (scripts/phase19_watermasks.py) -> per date, MULTI-TILE:
   every verified scene within +/-MAX_LAG days is a candidate; each segment is sampled
   from whichever covering scene is closest in time.
4. conservative classification from FOUR cues, never one:
       Sentinel water geometry  +  SWORD distance  +  connected-component connectivity
       +  water-body morphology  (+ WSE consistency as a physical sanity gate)
   classes: MAIN_CHANNEL / CONNECTED_SIDE_CHANNEL / RESIDUAL_POND / FLOODED_DEPRESSION
            / TRIBUTARY / UNKNOWN.  Anything ambiguous stays UNKNOWN.
5. two channel definitions are kept side by side:
       water_class_geom  — geometry + topology only
       water_class       — geom AND WSE physically consistent with the local water surface
   so the downstream slope test can be shown to survive (or not) either way, without the
   +3.31 cm/km benchmark ever being used to tune the classification.

Outputs
-------
outputs/tables/atl13_water_classification.parquet
outputs/tables/atl13_image_matchups.csv
data/processed/postbreach_classification/atl13_classified.parquet
data/processed/profiles/sword_dnipro_channel.parquet
"""
from __future__ import annotations

import sys
import warnings
from collections import OrderedDict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW
from swot_dnipro import watermask as WM      # the single definition of which SCL classes count as an observation
from swot_dnipro.plotting.style import nmad

MASKS = CFG.ROOT / "data" / "processed" / "water_masks"
PROC = CFG.ROOT / "data" / "processed"
(PROC / "postbreach_classification").mkdir(parents=True, exist_ok=True)
(PROC / "profiles").mkdir(parents=True, exist_ok=True)

MAX_LAG_DAYS = 10.0          # candidates; the real lag is stored per segment, never hidden
FAR_LAG_DAYS = 5.0           # flag: imagery this far away is not "simultaneous"
CHANNEL_DIST_KM = 1.5        # SWORD proximity for "on the main stem"
SIDE_DIST_KM = 6.0
POND_EDGE_KM = 0.25          # >250 m from any shore -> broad pond, else shallow/narrow
WSE_HARD_M = 1.25            # a connected river surface cannot deviate this far locally
PERSIST_LON, PERSIST_LAT = 0.004, 0.003   # ~300 m cells for cross-date pond identity

#: full-tile 20 m masks are ~0.4 GB each once the distance transform is built; on a
#: 16 GB box the old "load every scene up front" approach OOM-killed the process
#: mid-run (no traceback, just gone). Hold at most this many resident, LRU-evicted.
_MASK_CACHE: "OrderedDict[str, Mask]" = OrderedDict()
_MASK_CACHE_MAX = 12


def get_mask(path: Path) -> "Mask":
    key = path.stem
    m = _MASK_CACHE.get(key)
    if m is None:
        m = Mask(path)
        _MASK_CACHE[key] = m
        while len(_MASK_CACHE) > _MASK_CACHE_MAX:
            _MASK_CACHE.popitem(last=False)
    else:
        _MASK_CACHE.move_to_end(key)
    return m


class Mask:
    """A cached Sentinel-2 water mask (20 m) with point sampling."""

    def __init__(self, npz: Path):
        from affine import Affine
        z = np.load(npz, allow_pickle=True)
        self.mask = z["mask"]
        # OBSERVEDNESS (repair 2026-09-20, finding F01). This used to read `mask` and `labels` only, so a cloud inside the
        # raster came back as a genuine observation of "not water": `classify_date` then let a nearby CLOUDY scene outbid a
        # slightly older CLEAR one, and the segment was recorded as dry with the reason "not water in Sentinel mask". The
        # cache already carries the observedness plane the rest of the project uses -- the same SCL_REJECT rule as
        # watermask.py and composites.py -- so it is loaded here and kept separate from geometric coverage. An unobserved
        # pixel is never a dry vote.
        if "valid" in z.files:
            self.valid = np.asarray(z["valid"], bool)
        elif "scl" in z.files:
            self.valid = ~np.isin(np.asarray(z["scl"]), WM.SCL_REJECT)
        else:
            raise SystemExit(f"{npz.name}: no `valid` and no `scl` plane -- this cache predates the observedness "
                             f"convention and cannot distinguish dry from unobserved. Rebuild it with phase19_watermasks.py")
        if self.valid.shape != self.mask.shape:
            raise SystemExit(f"{npz.name}: valid {self.valid.shape} does not match mask {self.mask.shape}")
        self.labels = np.asarray(z["labels"], dtype=np.int32)
        a = z["affine"]
        self.transform = Affine(*a)
        self.crs = str(z["crs"])
        self.tile = str(z["tile"])
        self.sensing = str(z["sensing"])
        self.name = npz.stem
        self._edt = None
        self._tf = None

    def _xy(self, lon, lat):
        from pyproj import Transformer
        if self._tf is None:
            self._tf = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x, y = self._tf.transform(np.asarray(lon, float), np.asarray(lat, float))
        inv = ~self.transform
        col, row = inv * (x, y)
        return np.floor(col).astype(int), np.floor(row).astype(int)

    def sample(self, lon, lat):
        """Return (on_grid, observed, is_water, label, edge_km) for each point.

        `on_grid` is geometric coverage alone -- the point falls inside this scene's raster. `observed` additionally
        requires that the pixel was actually seen (SCL not in SCL_REJECT). The two are returned separately because
        conflating them is what turned cloud into dry land.
        """
        from scipy import ndimage
        col, row = self._xy(lon, lat)
        h, w = self.mask.shape
        ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        obs = np.zeros(len(col), bool)
        isw = np.zeros(len(col), bool)
        lab = np.zeros(len(col), int)
        edge = np.full(len(col), np.nan)
        if ok.any():
            obs[ok] = self.valid[row[ok], col[ok]]
            isw[ok] = self.mask[row[ok], col[ok]]
            lab[ok] = self.labels[row[ok], col[ok]]
            if self._edt is None:
                self._edt = (ndimage.distance_transform_edt(self.mask)
                             * abs(self.transform.a) / 1000.0).astype(np.float32)
            edge[ok] = self._edt[row[ok], col[ok]]
        return ok, obs & ok, isw & obs, lab, edge


def build_channel():
    nc = sorted((CFG.ROOT / "data/reference/river_network/sword_v16").rglob("*.nc"))
    if not nc:
        raise SystemExit("SWORD netCDF not found under data/reference/river_network/sword_v16")
    nc = nc[0]
    nodes = SW.load_nodes(nc)
    dn = SW.dnipro_nodes(nodes)
    ch = SW.chainage_from_dam(dn, *CFG.KAKHOVKA_DAM)
    ch = ch[(ch.chain_km > -60) & (ch.chain_km < 320)].reset_index(drop=True)
    ch.to_parquet(PROC / "profiles" / "sword_dnipro_channel.parquet", index=False)
    print(f"SWORD {nc.name}: {len(ch):,} Dnipro nodes, chainage "
          f"{ch.chain_km.min():.0f}..{ch.chain_km.max():.0f} km "
          f"(corr with longitude {np.corrcoef(ch.chain_km, ch.lon)[0,1]:.3f} -> not longitude)")
    return ch


def load_atl13(ch):
    from swot_dnipro.vertical import sample_grid
    seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet")
    seg["dt"] = pd.to_datetime(seg["time"], utc=True, errors="coerce").dt.tz_localize(None)
    seg = seg.dropna(subset=["lat", "lon", "h_wgs84_m", "dt"])
    seg["date"] = seg["dt"].dt.normalize()
    z = sample_grid(CFG.EGG2015_TIF, seg.lon.values, seg.lat.values)
    seg["wse_m"] = seg.h_wgs84_m.values + CFG.free2mean(seg.lat.values) - z
    seg["period"] = np.where(seg.dt < CFG.BREACH_DATE, "PRE_BREACH",
                             np.where(seg.dt < "2023-09-01", "BREACH_DRAWDOWN", "POST_BREACH"))
    cch, cdist, nid, rid = SW.assign_chainage(seg.lon, seg.lat, ch)
    seg["chain_km"], seg["dist_to_sword_km"] = cch, cdist
    seg["sword_node_id"], seg["sword_reach_id"] = nid, rid
    seg = seg.dropna(subset=["chain_km", "wse_m"]).reset_index(drop=True)
    print(f"ATL13 segments with SWORD chainage: {len(seg):,}")
    print(seg.groupby("period").size().to_string())
    return seg


def classify_date(g, masks, inv_row_by_name):
    """Classify all ATL13 segments of one date. Returns the augmented frame."""
    lon, lat = g.lon.values, g.lat.values
    n = len(g)
    atl_dt = g.dt.iloc[0]

    # ---- multi-tile sampling: best (closest-in-time) covering scene per segment ----
    best_lag = np.full(n, np.inf)
    isw = np.zeros(n, bool)
    lab = np.zeros(n, int)
    edge = np.full(n, np.nan)
    img = np.array([""] * n, dtype=object)
    sdt = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    on_any = np.zeros(n, bool)
    on_obs = np.zeros(n, bool)          # seen by at least one scene, not merely covered by one
    # per-scene "which components carry the main stem" cache
    main_ids_by_scene = {}
    dsw = g.dist_to_sword_km.values

    for m in masks:
        s_dt = pd.to_datetime(m.sensing, format="%Y%m%dT%H%M%S")
        lag_d = abs((s_dt - atl_dt).total_seconds()) / 86400.0
        if lag_d > MAX_LAG_DAYS:
            continue
        ok, obs, w, lb, ed = m.sample(lon, lat)
        on_any |= ok
        on_obs |= obs
        # main-stem components in this scene
        near = obs & w & (dsw < CHANNEL_DIST_KM) & (lb > 0)
        mids = set(pd.Series(lb[near]).value_counts().head(3).index) if near.any() else set()
        main_ids_by_scene[m.name] = mids
        # The closest scene that actually SAW the point wins. Selecting on coverage alone let a nearer cloudy
        # scene displace a slightly older clear one and then report its cloud as dry ground (finding F01).
        take = obs & (lag_d < best_lag)
        best_lag[take] = lag_d
        isw[take] = w[take]
        lab[take] = lb[take]
        edge[take] = ed[take]
        img[take] = m.name
        sdt[take] = np.datetime64(s_dt)

    in_main = np.array([lab[i] in main_ids_by_scene.get(img[i], set()) and lab[i] > 0
                        for i in range(n)])

    # ---- geometry + topology + morphology classification -------------------
    geom = np.full(n, "UNKNOWN", dtype=object)
    why = np.full(n, "", dtype=object)
    offgrid = ~on_any
    geom[offgrid] = "UNKNOWN"; why[offgrid] = "outside every image footprint"
    # Covered by imagery but never seen through it. This used to be indistinguishable from a dry observation.
    unobs = on_any & (~on_obs)
    geom[unobs] = "UNKNOWN"; why[unobs] = "inside an image footprint but NOT OBSERVED in any scene within the lag window (cloud, shadow, snow or no-data)"
    dry = on_obs & (~isw)
    geom[dry] = "UNKNOWN"; why[dry] = "not water in Sentinel mask (ATL13 prior DB is pre-breach)"
    w = on_obs & isw

    def setc(mask, cls, reason):
        geom[mask] = cls
        why[mask] = reason

    setc(w & (dsw < CHANNEL_DIST_KM) & in_main, "MAIN_CHANNEL",
         "water, <1.5 km of SWORD Dnipro, in the main connected water body")
    setc(w & (dsw < CHANNEL_DIST_KM) & (~in_main), "UNKNOWN",
         "water near SWORD but hydraulically disconnected component - ambiguous")
    setc(w & (dsw >= CHANNEL_DIST_KM) & (dsw < SIDE_DIST_KM) & in_main,
         "CONNECTED_SIDE_CHANNEL", "water, connected to main body, 1.5-6 km off the stem")
    setc(w & (dsw >= SIDE_DIST_KM) & in_main, "TRIBUTARY",
         "water, connected but >6 km off the main stem")
    setc(w & (dsw >= CHANNEL_DIST_KM) & (~in_main) & (edge > POND_EDGE_KM), "RESIDUAL_POND",
         "water, disconnected component, broad (>250 m to a shore)")
    setc(w & (dsw >= CHANNEL_DIST_KM) & (~in_main) & (edge <= POND_EDGE_KM), "FLOODED_DEPRESSION",
         "water, disconnected component, shallow/narrow (<250 m to a shore)")

    # ---- WSE consistency gate (physical, not fitted to a slope) ------------
    wse = g.wse_m.values
    chain = g.chain_km.values
    prov_main = geom == "MAIN_CHANNEL"
    wse_resid = np.full(n, np.nan)
    wse_ok = np.ones(n, bool)
    if prov_main.sum() >= 8 and (chain[prov_main].max() - chain[prov_main].min()) > 10:
        ts = stats.theilslopes(wse[prov_main], chain[prov_main])
        fit = ts[0] * chain + ts[1]
        wse_resid = wse - fit
        thr = max(WSE_HARD_M, 4 * nmad(wse_resid[prov_main]))
        wse_ok = np.abs(wse_resid) <= thr
    elif prov_main.any():
        med = np.median(wse[prov_main])
        wse_resid = wse - med
        wse_ok = np.abs(wse_resid) <= WSE_HARD_M

    cls = geom.copy()
    demote = prov_main & (~wse_ok)
    cls[demote] = "UNKNOWN"
    why[demote] = "near/connected channel but WSE inconsistent with the local water surface"

    # ---- cross-date persistence key for residual water --------------------
    pid = np.array([""] * n, dtype=object)
    resid = np.isin(cls, ["RESIDUAL_POND", "FLOODED_DEPRESSION"])
    pid[resid] = [f"{round(x / PERSIST_LON)}_{round(y / PERSIST_LAT)}"
                  for x, y in zip(lon[resid], lat[resid])]

    return g.assign(
        water_class=cls, water_class_geom=geom, class_reason=why,
        is_water=np.where(on_obs, isw, np.nan), in_main_body=np.where(on_obs, in_main, np.nan),
        observed=on_obs, covered_by_imagery=on_any,
        water_body_id=[f"{img[i]}:{lab[i]}" if lab[i] > 0 else "" for i in range(n)],
        dist_to_water_edge_km=edge,
        image_name=img, sentinel_dt=pd.to_datetime(sdt),
        delta_time_days=(pd.to_datetime(sdt) - atl_dt).total_seconds() / 86400.0,
        abs_lag_days=np.where(np.isfinite(best_lag), best_lag, np.nan),
        imagery_far=np.where(np.isfinite(best_lag), best_lag > FAR_LAG_DAYS, True),
        wse_resid_from_channel_m=wse_resid, wse_consistent=wse_ok,
        persist_id=pid)


def main() -> None:
    ch = build_channel()
    seg = load_atl13(ch)

    inv = pd.read_csv(CFG.TABLES / "sentinel_scene_inventory.csv").drop_duplicates("name")
    inv = inv[(inv.get("valid_zip", False)) & (inv.get("all_bands_ok", False))]
    names = set(inv.name)
    mask_index = [(p, pd.to_datetime(p.stem.split("_")[2], format="%Y%m%dT%H%M%S"))
                  for p in sorted(MASKS.glob("*.npz")) if p.stem in names]
    if not mask_index:
        raise SystemExit("no cached masks matching verified scenes — run phase19_watermasks.py")
    tiles = sorted({p.stem.split("_")[5] for p, _ in mask_index})
    print(f"\ncached water masks: {len(mask_index)}  tiles {tiles}  "
          f"(loaded on demand, LRU cap {_MASK_CACHE_MAX})")

    target = seg[seg.period.isin(["BREACH_DRAWDOWN", "POST_BREACH"])].copy()
    out = []
    for d, g in target.groupby("date"):
        atl_dt = g.dt.iloc[0]
        cand = [p for p, s in mask_index
                if abs((s - atl_dt).total_seconds()) / 86400.0 <= MAX_LAG_DAYS]
        day_masks = [get_mask(p) for p in cand]
        gg = classify_date(g.reset_index(drop=True), day_masks, inv)
        out.append(gg)
        vc = gg.water_class.value_counts()
        used = sorted(set(gg.loc[gg.image_name != "", "image_name"]))
        lag = gg.abs_lag_days.dropna()
        tag = "" if used else "  [NO IMAGERY]"
        print(f"  {d.date()}  n={len(gg):5d}  scenes={len(used)}  "
              f"lag={lag.median():.1f}d " +
              "  ".join(f"{k}={v}" for k, v in vc.head(4).items()) + tag)

    cl = pd.concat(out, ignore_index=True)
    keep = ["date", "dt", "period", "rgt", "beam", "gt", "lat", "lon", "h_wgs84_m", "wse_m",
            "chain_km", "dist_to_sword_km", "sword_node_id", "sword_reach_id",
            "water_class", "water_class_geom", "class_reason", "is_water", "in_main_body",
            "water_body_id", "persist_id", "dist_to_water_edge_km",
            "image_name", "sentinel_dt", "delta_time_days", "abs_lag_days", "imagery_far",
            "wse_resid_from_channel_m", "wse_consistent"]
    cl = cl[[c for c in keep if c in cl]]
    cl.to_parquet(CFG.TABLES / "atl13_water_classification.parquet", index=False)
    cl.to_parquet(PROC / "postbreach_classification" / "atl13_classified.parquet", index=False)

    # per-date image matchup ledger (step 7)
    mm = (cl[cl.image_name != ""].groupby(["date", "period"])
          .agg(n_segments=("lat", "size"),
               n_scenes=("image_name", "nunique"),
               tiles=("image_name", lambda s: "+".join(sorted({x.split("_")[5] for x in s}))),
               median_lag_days=("abs_lag_days", "median"),
               max_lag_days=("abs_lag_days", "max"),
               sentinel_dates=("sentinel_dt", lambda s: "+".join(
                   sorted({str(pd.Timestamp(x).date()) for x in s.dropna()}))),
               far_frac=("imagery_far", "mean")).reset_index())
    mm.to_csv(CFG.TABLES / "atl13_image_matchups.csv", index=False)

    print("\n=== classification totals (unit = ATL13 segment) ===")
    print(cl.groupby(["period", "water_class"]).size().to_string())
    nonwater = cl[cl.is_water == 0.0]
    onimg = cl[cl.image_name != ""]
    print(f"\non-image segments: {len(onimg):,}   of which NOT water: "
          f"{len(nonwater):,} ({len(nonwater)/max(len(onimg),1)*100:.1f}%)  "
          f"[ATL13 pre-breach prior DB over drained bed]")
    print(f"MAIN_CHANNEL (geom): {(cl.water_class_geom=='MAIN_CHANNEL').sum():,}   "
          f"after WSE gate: {(cl.water_class=='MAIN_CHANNEL').sum():,}")
    print(f"\n-> {CFG.TABLES/'atl13_water_classification.parquet'}")
    print(f"-> {CFG.TABLES/'atl13_image_matchups.csv'}")


if __name__ == "__main__":
    main()
