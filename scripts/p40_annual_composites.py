#!/usr/bin/env python
"""P40 -- per-year, per-window composites of the p25 stacks for every zone (plan 15, WP1).

Windows (fixed):
    leafon    Jun-Sep          spring   Apr-May        autumn   Oct-Nov
    winter    Dec(Y-1)-Feb(Y)  (leaf-off roughness scenario)
    drawdown_2023        2023-06-06 .. 2023-07-15   (reservoir emptying)
    first_exposure_2023  2023-07-16 .. 2023-09-30   (first post-drainage state)
Per zone x year x window, on the registry grid (SD.build_grid, 20 m):
    zone<N>_<INDEX>_<year>_<window>_20m.tif   int16 median x1e4 (7 indices)
    zone<N>_class_mode_<year>_<window>_20m.tif uint8 (INVALID never votes: swot_dnipro.composites)
    zone<N>_water_share_<year>_<window>_20m.tif uint8 % of observed dates (255 never observed)
    zone<N>_n_valid_<year>_<window>_20m.tif    uint8 observed dates
    zone<N>_scl_mode_<year>_<window>_20m.tif   uint8 ESA SCL mode; zone<N>_scl_share_{veg,notveg,water}_..  uint8 %
Tables:
    outputs/tables/p40_class_shares_<zone>.csv    per date: class shares of the OBSERVED area inside the zone
    outputs/tables/p40_window_summary_<zone>.csv  per year x window: n_dates, dates, tiles_complete count,
                                                  median/min/max of observed share and class shares
Composites are computed in row blocks (256 rows x all dates of the window) so ZONE_1 (96 Mpx) never holds
more than one block; a date enters a window only if its class file exists; SCL products are written only
if every date of the window has a <date>_scl.tif (p25 backfill) -- otherwise the SCL layers are skipped and
the summary says so.

Usage:  python scripts/p40_annual_composites.py --zone ZONE_4_DAM_TO_KHERSON_FLOODWAY [--years 2023 2024] [--windows leafon]
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
from rasterio.windows import Window

from swot_dnipro import composites as CO
from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP

ZONES = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2,
         "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
STACKS = CFG.BULK_ROOT / "zone_spectral"
CELL, BLOCK = 20.0, 256
WINDOWS = {"leafon": (6, 9), "spring": (4, 5), "autumn": (10, 11), "winter": (12, 2)}
EVENTS = {"drawdown_2023": ("2023-06-06", "2023-07-15"), "first_exposure_2023": ("2023-07-16", "2023-09-30")}


def dates_in(zone_dir: Path, year: int, window: str) -> list[str]:
    ds = sorted(p.name[:10] for p in zone_dir.glob("*_class.tif"))
    if window in EVENTS:
        a, b = EVENTS[window]
        return [d for d in ds if a <= d <= b]
    m0, m1 = WINDOWS[window]
    if window == "winter":
        return [d for d in ds if (d[:4] == str(year - 1) and int(d[5:7]) == 12) or (d[:4] == str(year) and int(d[5:7]) <= 2)]
    return [d for d in ds if d[:4] == str(year) and m0 <= int(d[5:7]) <= m1]


def per_date_shares(zone_dir: Path, dates: list[str], inside: np.ndarray) -> list[dict]:
    rows = []
    for d in dates:
        with rasterio.open(zone_dir / f"{d}_class.tif") as ds:
            c = ds.read(1); tc = ds.tags().get("tiles_complete", "")
        if tc == "" and (zone_dir / f"{d}_scl.tif").exists():     # dates written before p25 tagged tiles_complete: the SCL backfill carries it
            with rasterio.open(zone_dir / f"{d}_scl.tif") as ds:
                tc = ds.tags().get("tiles_complete", "")
        obs = inside & (c != 0)
        n = int(obs.sum())
        r = dict(date=d, tiles_complete=tc, observed_share=round(n / max(int(inside.sum()), 1), 4), observed_km2=round(n * 0.0004, 1))
        for k, nm in SP.CLASSES.items():
            if k:
                r[f"share_{nm}"] = round(float((obs & (c == k)).sum()) / max(n, 1), 4)
        r["share_water_1_2"] = round(float((obs & np.isin(c, (1, 2))).sum()) / max(n, 1), 4)
        r["share_sediment_3_4"] = round(float((obs & np.isin(c, (3, 4))).sum()) / max(n, 1), 4)
        r["share_veg_5_6_7"] = round(float((obs & np.isin(c, (5, 6, 7))).sum()) / max(n, 1), 4)
        rows.append(r)
    return rows


def composite(zone: str, G: dict, zone_dir: Path, out_dir: Path, year: int, window: str, dates: list[str]) -> dict:
    n, ny, nx = ZONES[zone], G["ny"], G["nx"]
    tag = f"{year}_{window}"
    scl_dates = [d for d in dates if (zone_dir / f"{d}_scl.tif").exists()]   # SCL layers from the dates that have it (older stacks whose zips are gone cannot be backfilled)
    have_scl = len(scl_dates) > 0
    med = {k: np.full((ny, nx), CO.INDEX_NODATA, "i2") for k in SP.INDEX_NAMES}
    mode = np.zeros((ny, nx), "u1"); wsh = np.full((ny, nx), 255, "u1"); nval = np.zeros((ny, nx), "u1")
    scl_out = {k: np.full((ny, nx), 255 if "share" in k else 0, "u1") for k in ("scl_mode", "scl_share_veg", "scl_share_notveg", "scl_share_water")} if have_scl else {}
    si = [rasterio.open(zone_dir / f"{d}_indices.tif") for d in dates]
    sc = [rasterio.open(zone_dir / f"{d}_class.tif") for d in dates]
    sw = [rasterio.open(zone_dir / f"{d}_water3.tif") for d in dates]
    ss = [rasterio.open(zone_dir / f"{d}_scl.tif") for d in scl_dates]
    try:
        for r0 in range(0, ny, BLOCK):
            h = min(BLOCK, ny - r0); win = Window(0, r0, nx, h)
            C = np.stack([s.read(1, window=win) for s in sc])
            W = np.stack([s.read(1, window=win) for s in sw])
            mode[r0:r0 + h] = CO.class_mode(C); wsh[r0:r0 + h] = CO.water_share(W); nval[r0:r0 + h] = CO.n_valid(C, 0)
            for bi, k in enumerate(SP.INDEX_NAMES, 1):
                A = np.stack([s.read(bi, window=win) for s in si])
                med[k][r0:r0 + h] = CO.median_index(A)
            if have_scl:
                S = np.stack([s.read(1, window=win) for s in ss])
                sh = CO.scl_shares(S)
                for k in scl_out:
                    scl_out[k][r0:r0 + h] = sh[k]
            del C, W
    finally:
        for s in si + sc + sw + ss:
            s.close()
    tags = dict(zone=zone, year=str(year), window=window, n_dates=str(len(dates)), dates="|".join(dates), cell_m=str(CELL),
                producer="p40_annual_composites.py", vote="INVALID never votes (swot_dnipro.composites)")
    for k in SP.INDEX_NAMES:
        p = out_dir / f"zone{n}_{k}_{tag}_20m.tif"
        with rasterio.open(p, "w", **SP._profile(G, 1, "int16", CO.INDEX_NODATA)) as dst:
            dst.write(med[k], 1); dst.set_band_description(1, f"{k} median x{SP.INDEX_SCALE}"); dst.update_tags(index=k, scale=f"value/{SP.INDEX_SCALE}", **tags)
    SP.write_uint8(out_dir / f"zone{n}_class_mode_{tag}_20m.tif", mode, G, {**tags, "legend": str(SP.CLASSES)}, nodata=0)
    SP.write_uint8(out_dir / f"zone{n}_water_share_{tag}_20m.tif", wsh, G, {**tags, "values": "percent of observed dates with water; 255 never observed"})
    SP.write_uint8(out_dir / f"zone{n}_n_valid_{tag}_20m.tif", nval, G, {**tags, "values": "observed dates"}, nodata=0)
    for k, a in scl_out.items():
        SP.write_uint8(out_dir / f"zone{n}_{k}_{tag}_20m.tif", a, G, {**tags, "scl_dates": "|".join(scl_dates), "n_scl_dates": str(len(scl_dates)),
                                                                       "values": "ESA SCL mode (0 none)" if k == "scl_mode" else "percent of observed SCL dates; 255 never observed"},
                       nodata=0 if k == "scl_mode" else 255)
    return dict(have_scl=have_scl, n_scl_dates=len(scl_dates), inside_observed_share=round(float((nval[G["inside"]] > 0).mean()), 4), inside_ge2_share=round(float((nval[G["inside"]] >= 2).mean()), 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=list(ZONES))
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2017, 2027)))
    ap.add_argument("--windows", nargs="*", default=list(WINDOWS) + list(EVENTS))
    a = ap.parse_args()
    G = SP.zone_grid(a.zone, CELL)
    zone_dir = STACKS / a.zone
    out_dir = ROOT / "outputs" / "rasters" / f"zone{ZONES[a.zone]}" / "annual"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"P40 {a.zone}: grid {G['nx']}x{G['ny']}, stacks {len(list(zone_dir.glob('*_class.tif')))} dates")
    summ, shares = [], {}
    t0 = time.time()
    for w in a.windows:
        years = [2023] if w in EVENTS else a.years
        for y in years:
            dates = dates_in(zone_dir, y, w)
            if not dates:
                continue
            rows = per_date_shares(zone_dir, dates, G["inside"])
            for r in rows:
                shares[r["date"]] = r
            res = composite(a.zone, G, zone_dir, out_dir, y, w, dates)
            df = pd.DataFrame(rows)
            s = dict(year=y, window=w, n_dates=len(dates), dates="|".join(dates), n_tiles_complete=int((df.tiles_complete == "True").sum()), **res,
                     observed_share_median=df.observed_share.median(), observed_share_min=df.observed_share.min())
            for k in ("share_water_1_2", "share_sediment_3_4", "share_veg_5_6_7", "share_REED_OR_FLOODED_VEGETATION", "share_DENSE_HERBACEOUS", "share_DRY_BARE_SEDIMENT"):
                s[f"{k}_median"] = round(df[k].median(), 4); s[f"{k}_min"] = round(df[k].min(), 4); s[f"{k}_max"] = round(df[k].max(), 4)
            summ.append(s)
            print(f"  {y} {w:20s} {len(dates):3d} dates  obs>=1 {res['inside_observed_share']:.2f}  veg median {s['share_veg_5_6_7_median']:.2f}  scl={res['have_scl']}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(sorted(shares.values(), key=lambda r: r["date"])).to_csv(CFG.TABLES / f"p40_class_shares_{a.zone}.csv", index=False)
    pd.DataFrame(summ).to_csv(CFG.TABLES / f"p40_window_summary_{a.zone}.csv", index=False)
    print(f"-> {out_dir}  ({len(summ)} year-windows)")


if __name__ == "__main__":
    main()
