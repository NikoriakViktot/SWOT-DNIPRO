#!/usr/bin/env python
"""P43 -- roughness classes and Manning's n (low/base/high) per STATE and DOMAIN, from project data only (plan 15, WP3).

Canonical successor of audit_runs/20260918T175108Z/pilots/roughness (R-020) with the two fixes of that run kept
(INVALID never votes; pre-existing woody cover flagged everywhere) and the additions the plan asks for:
  * domains   pool                  former water surface of the reservoir (pool polygon AND pre-breach mode water),
                                    on the ZONE_1 grid window
              below_dam_floodplain  p42 domain (data/processed/domains/below_dam_floodplain_utm.geojson) on the ZONE_4
                                    and ZONE_2 grids (the two frames that cover it)
  * states    BREACH_2023_bed       pool only: bed zoning by the 2023 recession (fast / mid / slow / never exposed)
                                    -> channel_sand / sand_silt_flat / wet_silt_depression (roughness of the emptying bed)
              BREACH_2023           pre-breach cover: p40 leafon <PRE_YEAR> composite + WorldCover 2021 + DW 2022
              FIRST_EXPOSURE_2023   p40 first_exposure_2023 window, DW 2023
              STATE_2024 / STATE_2025 / CURRENT_2026   p40 leafon composites; DW frame of the year (p37a or p41 from-annual),
                                    falling back to the latest available frame (recorded in tags)
  * class rule (config/roughness_classes.yaml codes 1..15; n from the same file, never from an index):
      water share >= 80 % ........................ open_water            water share < 80 % & S2 water ...... intermittent_water
      WET_SEDIMENT ........ wet_sediment          DRY_BARE_SEDIMENT ..... bare_sand_silt
      SPARSE_HERBACEOUS ... sparse_herbaceous     DENSE_HERBACEOUS (woody < 0.3) ... dense_herbaceous
      REED (woody < 0.3) .. reed_tall_herb        green & DW woody label (trees/shrub) & not legacy .. young_woody_dense
      green & woody prob >= 0.3 & not label & not legacy .. young_woody_sparse
      legacy woody (WorldCover 2021 tree/shrub or DW 2022 trees/shrub) & green ... mature_woody_legacy
      DW built >= 0.5 or WorldCover 50 ........... built
      floodplain only: (DW crops label or WorldCover 40) & herbaceous S2 class ... cropland   (never inside the pool)
      BREACH_2023: young_woody classes are not allowed (nothing had grown yet).
  * winter variant: the woody classes get n x leaf_off_factor_woody (yaml) in *_winter rasters (scenario, not a state).
  * ATL08 canopy density (audit pilots/icesat2/out/canopy_segments_20m.parquet, canopy_ok, leaf-on) is written as an
    independent woody indicator raster (segments with h_canopy >= 2 m per 250 m cell) and used as an OR-condition for
    young_woody_dense where DW is missing for the year (2025/2026 without a frame).
Outputs: outputs/rasters/roughness/<domain>/<grid>_manning_classes_<state>.tif, _manning_n_{low,base,high}_<state>.tif,
         _manning_uncertainty_<state>.tif, _dn_base_<a>_<b>.tif; outputs/tables/p43_class_areas_<domain>.csv,
         p43_transition_<domain>_<a>_<b>.csv, p43_roughness_transition_summary.csv, p43_bed_recession_zoning.csv
Usage: python scripts/p43_roughness_states.py --domain pool | below_dam_floodplain [--states ...]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio import features
from rasterio.windows import from_bounds

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP
from swot_dnipro import spatial_domains as SD

Y = yaml.safe_load((ROOT / "config/roughness_classes.yaml").read_text())
CODES = {v: int(k) for k, v in Y["codes"].items()}
NAMES = {int(k): v for k, v in Y["codes"].items()}
NT = {k: (v["n_low"], v["n_base"], v["n_high"]) for k, v in Y["classes"].items()}
ZN = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
STATES = ["BREACH_2023_bed", "BREACH_2023", "FIRST_EXPOSURE_2023", "STATE_2024", "STATE_2025", "CURRENT_2026"]
STATE_WINDOW = {"BREACH_2023": None, "FIRST_EXPOSURE_2023": (2023, "first_exposure_2023"), "STATE_2024": (2024, "leafon"), "STATE_2025": (2025, "leafon"), "CURRENT_2026": (2026, "leafon")}
DW_YEAR = {"BREACH_2023": 2022, "FIRST_EXPOSURE_2023": 2023, "STATE_2024": 2024, "STATE_2025": 2025, "CURRENT_2026": 2026}
PRE_YEARS = (2022, 2021, 2020, 2019)         # last complete pre-breach leaf-on composite with >= 2 dates
RECESSION_DATES = dict(fast="2023-06-30", mid="2023-08-06", slow="2023-09-08")   # fixed before running (plan 15 R5)
PX_KM2 = 0.0004
OUT = ROOT / "outputs" / "rasters" / "roughness"


# ------------------------------------------------------------------ readers on a (zone grid, window) pair
class Frame:
    def __init__(self, zone: str, bounds=None):
        self.zone, self.n = zone, ZN[zone]
        self.G = SP.zone_grid(zone, 20.0)
        self.bounds = bounds
        with rasterio.open(self._annual("class_mode", 2025, "leafon") or self._any_stack()) as ds:
            self.win = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths() if bounds else None
            self.tr = ds.window_transform(self.win) if self.win else ds.transform
            self.shape = (int(self.win.height), int(self.win.width)) if self.win else (ds.height, ds.width)
        self.tag = f"zone{self.n}" + ("_poolwin" if bounds else "")

    def _any_stack(self):
        return sorted((CFG.BULK_ROOT / "zone_spectral" / self.zone).glob("*_class.tif"))[0]

    def _annual(self, var, year, window):
        p = ROOT / "outputs/rasters" / f"zone{self.n}" / "annual" / f"zone{self.n}_{var}_{year}_{window}_20m.tif"
        return p if p.exists() else None

    def read(self, path, band=1, dtype="f4", nodata=None):
        with rasterio.open(path) as ds:
            a = ds.read(band, window=self.win).astype(dtype)
            nd = ds.nodata if nodata is None else nodata
            if nd is not None and dtype == "f4":
                a[a == nd] = np.nan
            return a

    def annual(self, var, year, window, dtype="u1"):
        p = self._annual(var, year, window)
        return (self.read(p, dtype=dtype, nodata=-1) if p else None), (p.name if p else None)

    def dw(self, year):
        """label (u1, 255 nodata), woody prob (f4), built prob, crops label bool; searches p37a frame then p41 from-annual then nearest earlier year."""
        d = CFG.BULK_ROOT / "dynamic_world_frames" / self.zone
        for y in [year] + [y2 for y2 in range(year - 1, 2016, -1)]:
            for name in (f"dw_{y}_20m.tif", f"dw_{y}_20m_from_annual.tif"):
                p = d / name
                if p.exists():
                    lab = self.read(p, 1, "u1", 255)
                    trees, shrub, built = (self.read(p, b, "f4", 255) / 100.0 for b in (3, 7, 8))
                    valid = lab != 255
                    return dict(label=lab, woody=np.where(valid, trees + shrub, np.nan).astype("f4"), built=np.where(valid, built, np.nan).astype("f4"),
                                woody_label=valid & np.isin(lab, (1, 5)), crops_label=valid & (lab == 4), source=f"{name} (asked {year})")
        return None

    def worldcover(self, year=2021):
        p = CFG.BULK_ROOT / "worldcover_frames" / self.zone / f"wc_{year}_20m.tif"
        return self.read(p, 1, "u1", 0) if p.exists() else None

    def rasterize(self, geom):
        return features.rasterize([(geom, 1)], out_shape=self.shape, transform=self.tr, fill=0, dtype="uint8").astype(bool)

    def write(self, name, arr, dtype, nodata, tags):
        d = OUT / tags.get("domain", "x"); d.mkdir(parents=True, exist_ok=True)
        with rasterio.open(d / f"{self.tag}_{name}.tif", "w", driver="GTiff", height=self.shape[0], width=self.shape[1], count=1, dtype=dtype, crs=CFG.CRS_METRIC, transform=self.tr, nodata=nodata, compress="deflate", tiled=True) as ds:
            ds.write(arr.astype(dtype), 1); ds.update_tags(**{k: str(v) for k, v in tags.items()}, producer="p43_roughness_states.py", n_rule="class -> n from config/roughness_classes.yaml; never from an index")


def canopy_density(fr: Frame) -> np.ndarray | None:
    """Independent woody indicator: ATL08 canopy_ok, leaf-on, h_canopy >= 2 m, counted per 250 m cell, spread to 20 m (u1 count; 255 no track within 250 m)."""
    p = ROOT / "audit_runs/20260918T175108Z/pilots/icesat2/out/canopy_segments_20m.parquet"
    if not p.exists():
        return None
    import geopandas as gpd
    g = gpd.read_parquet(p, columns=["geometry", "year", "month", "canopy_ok", "h_canopy"])
    g = g[g.canopy_ok & g.month.between(6, 9) & (g.year >= 2024)]
    x0, y1 = fr.tr.c, fr.tr.f
    cx = ((g.geometry.x.values - x0) // 250).astype(int); cy = ((y1 - g.geometry.y.values) // 250).astype(int)
    ny, nx = -(-fr.shape[0] * 20 // 250), -(-fr.shape[1] * 20 // 250)
    inb = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    cnt = np.zeros((ny, nx), "i4"); any_ = np.zeros((ny, nx), bool)
    np.add.at(cnt, (cy[inb], cx[inb]), (g.h_canopy.values[inb] >= 2.0).astype(int)); any_[cy[inb], cx[inb]] = True
    up = np.repeat(np.repeat(cnt, 13, 0), 13, 1)[:fr.shape[0], :fr.shape[1]]   # 250/20 = 12.5 -> 13 then crop (indicator only)
    anyu = np.repeat(np.repeat(any_, 13, 0), 13, 1)[:fr.shape[0], :fr.shape[1]]
    return np.where(anyu, np.clip(up, 0, 254), 255).astype("u1")


def classify(mode, wshare, dw, wc, legacy, pool_mask, allow_young: bool, canopy_cnt=None):
    rc = np.zeros(mode.shape, "u1")
    ws = np.where(wshare == 255, 0, wshare).astype("f4")
    woody = np.nan_to_num(dw["woody"], nan=0.0) if dw else np.zeros(mode.shape, "f4")
    wl = dw["woody_label"] if dw else np.zeros(mode.shape, bool)
    if canopy_cnt is not None:
        wl = wl | ((canopy_cnt != 255) & (canopy_cnt >= 3))
    built = (np.nan_to_num(dw["built"], nan=0.0) >= 0.5) if dw else np.zeros(mode.shape, bool)
    crops = dw["crops_label"] if dw else np.zeros(mode.shape, bool)
    if wc is not None:
        built |= wc == 50; crops |= wc == 40
    water = np.isin(mode, (1, 2)); green = np.isin(mode, (5, 6, 7))
    rc[water & (ws >= 80)] = CODES["open_water"]; rc[water & (ws < 80)] = CODES["intermittent_water"]
    rc[mode == 3] = CODES["wet_sediment"]; rc[mode == 4] = CODES["bare_sand_silt"]
    rc[mode == 5] = CODES["sparse_herbaceous"]; rc[mode == 6] = CODES["dense_herbaceous"]; rc[mode == 7] = CODES["reed_tall_herb"]
    if allow_young:
        rc[green & (woody >= 0.30) & ~wl & ~legacy] = CODES["young_woody_sparse"]
        rc[green & wl & ~legacy] = CODES["young_woody_dense"]
    rc[green & legacy] = CODES["mature_woody_legacy"]
    rc[crops & np.isin(mode, (5, 6)) & ~pool_mask] = CODES["cropland"]
    rc[built & (mode != 0) & ~water] = CODES["built"]
    rc[mode == 0] = 0
    return rc


def emit(fr: Frame, domain: str, state: str, rc, mask, extra: dict):
    rc = np.where(mask, rc, 0).astype("u1")
    low = np.full(rc.shape, np.nan, "f4"); base = low.copy(); high = low.copy()
    for k, nm in NAMES.items():
        if nm in NT:
            m = rc == k; low[m], base[m], high[m] = NT[nm]
    tags = dict(domain=domain, state=state, legend=NAMES, **extra)
    fr.write(f"manning_classes_{state}", rc, "uint8", 0, tags)
    for nm, a in (("low", low), ("base", base), ("high", high)):
        fr.write(f"manning_n_{nm}_{state}", np.nan_to_num(a, nan=-9999), "float32", -9999, dict(tags, quantity=f"Manning n {nm} [s m^-1/3]"))
    fr.write(f"manning_uncertainty_{state}", np.nan_to_num(np.where(np.isfinite(base), (high - low) / base, np.nan), nan=-9999), "float32", -9999, dict(tags, quantity="(n_high-n_low)/n_base"))
    if state != "BREACH_2023_bed":
        f = float(Y["season"]["leaf_off_factor_woody"])
        wm = np.isin(rc, [CODES[c] for c in ("young_woody_sparse", "young_woody_dense", "mature_woody_legacy", "reed_tall_herb")])
        fr.write(f"manning_n_base_{state}_winter", np.nan_to_num(np.where(wm, base * f, base), nan=-9999), "float32", -9999, dict(tags, quantity=f"n_base leaf-off scenario: woody/reed x {f}"))
    rows = dict(domain=domain, grid=fr.tag, state=state, valid_km2=round(float((rc > 0).sum()) * PX_KM2, 1), n_base_area_weighted=round(float(np.nanmean(base[rc > 0])), 4) if (rc > 0).any() else np.nan)
    for k, nm in NAMES.items():
        if k:
            rows[f"{nm}_km2"] = round(float((rc == k).sum()) * PX_KM2, 1)
    return rc, base, rows


def bed_recession(fr: Frame, former_water: np.ndarray):
    """BREACH_2023_bed zoning from the first non-water observation in the 2023 ZONE_1 stacks."""
    st = CFG.BULK_ROOT / "zone_spectral" / fr.zone
    dates = sorted(p.name[:10] for p in st.glob("2023-*_class.tif") if p.name[:10] >= CFG.BREACH_DATE)
    first = np.full(fr.shape, "", dtype="U10"); seen = np.zeros(fr.shape, bool); ever_obs = np.zeros(fr.shape, bool)
    substrate = np.zeros(fr.shape, "u1")
    for d in dates:
        c = fr.read(st / f"{d}_class.tif", 1, "u1", 0)
        v = c != 0; nonwater = v & ~np.isin(c, (1, 2)) & ~seen
        first[nonwater] = d; substrate[nonwater] = c[nonwater]; seen |= nonwater; ever_obs |= v
    rc = np.zeros(fr.shape, "u1")
    fw = former_water & ever_obs
    rc[fw & ~seen] = CODES["channel_sand"]                                             # never exposed in 2023
    rc[fw & seen & (first > RECESSION_DATES["mid"])] = CODES["wet_silt_depression"]   # slow: exposed after early August
    rc[fw & seen & (first <= RECESSION_DATES["mid"])] = CODES["sand_silt_flat"]       # fast/mid
    zoning = np.where(~fw, 0, np.where(~seen, 4, np.where(first <= RECESSION_DATES["fast"], 1, np.where(first <= RECESSION_DATES["mid"], 2, 3)))).astype("u1")
    fr.write("bed_recession_zoning_2023", zoning, "uint8", 0, dict(domain="pool", values="1 fast (exposed by 06-30) 2 mid (by 08-06) 3 slow (after 08-06) 4 never exposed in 2023", dates="|".join(dates)))
    fr.write("bed_first_exposure_substrate_2023", substrate, "uint8", 0, dict(domain="pool", values="S2 class at the first non-water observation", legend=SP.CLASSES))
    rows = [dict(zone_code=k, label=l, km2=round(float((zoning == k).sum()) * PX_KM2, 1)) for k, l in ((1, "fast"), (2, "mid"), (3, "slow"), (4, "never"))]
    for k, l in ((3, "WET_SEDIMENT"), (4, "DRY_BARE_SEDIMENT"), (5, "SPARSE_HERBACEOUS"), (6, "DENSE_HERBACEOUS"), (7, "REED")):
        rows.append(dict(zone_code=f"substrate_{k}", label=l, km2=round(float((substrate == k)[fw].sum()) * PX_KM2, 1)))
    pd.DataFrame(rows).to_csv(CFG.TABLES / "p43_bed_recession_zoning.csv", index=False)
    return rc, dates


def run(domain: str, fr: Frame, mask: np.ndarray, pool_mask: np.ndarray, legacy_base: np.ndarray, states: list[str], former_water=None):
    areas, classes, bases, trans = [], {}, {}, []
    wc = fr.worldcover(2021)
    cd = canopy_density(fr)
    if cd is not None:
        fr.write("atl08_canopy_ge2m_count_250m", cd, "uint8", 255, dict(domain=domain, quantity="ATL08 canopy_ok segments with h_canopy>=2 m per 250 m cell (leaf-on 2024-2025)"))
    for state in states:
        if state == "BREACH_2023_bed":
            if former_water is None:
                continue
            rc, dates = bed_recession(fr, former_water)
            rc, b, rows = emit(fr, domain, state, rc, mask, dict(dates="|".join(dates), recession_thresholds=RECESSION_DATES))
        else:
            if state == "BREACH_2023":
                mode = wsh = None
                for y in PRE_YEARS:
                    mode, src = fr.annual("class_mode", y, "leafon"); wsh, _ = fr.annual("water_share", y, "leafon")
                    nv, _ = fr.annual("n_valid", y, "leafon")
                    if mode is not None and nv is not None and (nv[mask] >= 2).mean() > 0.5:
                        break
                if mode is None:
                    print(f"  {state}: no pre-breach leaf-on composite"); continue
                dw = fr.dw(2022); allow = False
            else:
                y, w = STATE_WINDOW[state]
                mode, src = fr.annual("class_mode", y, w); wsh, _ = fr.annual("water_share", y, w)
                if mode is None:
                    print(f"  {state}: composite {y} {w} missing"); continue
                dw = fr.dw(DW_YEAR[state]); allow = True
            legacy = legacy_base.copy()
            if wc is not None:
                legacy |= np.isin(wc, (10, 20))
            dw22 = fr.dw(2022)
            if dw22 is not None:
                legacy |= dw22["woody_label"]
            rc = classify(mode, wsh, dw, wc, legacy, pool_mask, allow, cd if state in ("STATE_2025", "CURRENT_2026") else None)
            rc, b, rows = emit(fr, domain, state, rc, mask, dict(s2_source=src, dw_source=dw["source"] if dw else "none", worldcover="wc_2021" if wc is not None else "none"))
        classes[state], bases[state] = rc, b; areas.append(rows)
        print(f"  {domain}/{fr.tag} {state}: valid {rows['valid_km2']} km2, n_base {rows['n_base_area_weighted']}", flush=True)
    order = [s for s in states if s in classes and s != "BREACH_2023_bed"]
    for a, b in list(zip(order[:-1], order[1:])) + ([(order[0], order[-1])] if len(order) > 2 else []):
        ok = (classes[a] > 0) & (classes[b] > 0)
        m = np.zeros((16, 16), "i8"); np.add.at(m, (classes[a][ok], classes[b][ok]), 1)
        pd.DataFrame(m[1:, 1:] * PX_KM2, index=[NAMES[i] for i in range(1, 16)], columns=[NAMES[i] for i in range(1, 16)]).round(1).to_csv(CFG.TABLES / f"p43_transition_{domain}_{fr.tag}_{a}_{b}.csv")
        dn = np.where(ok, bases[b] - bases[a], np.nan).astype("f4")
        fr.write(f"dn_base_{a}_{b}", np.nan_to_num(dn, nan=-9999), "float32", -9999, dict(domain=domain, quantity="n_base(b)-n_base(a)"))
        d = dn[ok & np.isfinite(dn)]
        trans.append(dict(domain=domain, grid=fr.tag, from_state=a, to_state=b, area_km2=round(float(ok.sum()) * PX_KM2, 1), median_dn=float(np.median(d)) if d.size else np.nan,
                          share_dn_gt_0p02=float((d > 0.02).mean()) if d.size else np.nan, share_dn_lt_m0p02=float((d < -0.02).mean()) if d.size else np.nan,
                          n_a=float(np.nanmean(bases[a][ok])), n_b=float(np.nanmean(bases[b][ok]))))
    return areas, trans


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", required=True, choices=("pool", "below_dam_floodplain")); ap.add_argument("--states", nargs="*", default=STATES); a = ap.parse_args()
    t0 = time.time(); areas, trans = [], []
    if a.domain == "pool":
        pool = SD.load_utm("reservoir_full_pool_prebreach"); fr = Frame("ZONE_1_KAKHOVKA_LOWER_DNIPRO", pool.bounds)
        pre = fr.read(ROOT / "outputs/rasters/zone1/zone1_class_PRE_BREACH_mode_20m.tif", 1, "u1", 0)
        pm = fr.rasterize(pool); former_water = pm & np.isin(pre, (1, 2)); legacy = pm & ~np.isin(pre, (1, 2)) & (pre != 0)
        fr.write("former_water_surface_mask", former_water.astype("u1"), "uint8", 255, dict(domain="pool", values="1 pool polygon AND pre-breach mode water"))
        ar, tr = run("pool", fr, pm, pm, legacy, a.states, former_water); areas += ar; trans += tr
    else:
        import geopandas as gpd
        dom = gpd.read_file(ROOT / "data/processed/domains/below_dam_floodplain_utm.geojson").to_crs(CFG.CRS_METRIC).geometry.unary_union
        for zone in ("ZONE_4_DAM_TO_KHERSON_FLOODWAY", "ZONE_2_KHERSON_DELTA"):
            fr = Frame(zone); m = fr.rasterize(dom)
            if not m.any():
                continue
            pm = fr.rasterize(SD.load_utm("reservoir_full_pool_prebreach"))
            ar, tr = run("below_dam_floodplain", fr, m, pm, np.zeros(fr.shape, bool), [s for s in a.states if s != "BREACH_2023_bed"]); areas += ar; trans += tr
    pd.DataFrame(areas).to_csv(CFG.TABLES / f"p43_class_areas_{a.domain}.csv", index=False)
    p = CFG.TABLES / "p43_roughness_transition_summary.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    new = pd.DataFrame(trans)
    pd.concat([old[old.domain != a.domain] if len(old) else old, new]).to_csv(p, index=False)
    print(f"done {a.domain} in {time.time()-t0:.0f}s -> {OUT / a.domain}")


if __name__ == "__main__":
    main()
