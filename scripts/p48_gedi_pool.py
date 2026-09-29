#!/usr/bin/env python
"""P48 -- GEDI L2A canopy heights over the former Kakhovka pool (plan 15, user request 2026-09-19).

GEDI was never used in this project. CMR (earthaccess) lists ~100 GEDI02_A v002 granules per year over the pool bbox
(33.30-35.40 E, 46.70-47.90 N); each granule is a whole orbit (~1 GB), of which only a few km fall in the bbox. This
script therefore streams every granule (earthaccess.open -> fsspec/h5py, Earthdata login from EARTHDATA_TOKEN or netrc)
and reads ONLY the per-beam lat/lon arrays first, then the shot fields for the in-bbox indices:
  shot_number, delta_time, lat_lowestmode, lon_lowestmode, elev_lowestmode, rh (rh25, rh50, rh75, rh95, rh98, rh100),
  quality_flag, degrade_flag, sensitivity, solar_elevation, beam (power/coverage), land_cover_data/leaf_off_flag when present
Filters written as flags, not applied: quality_flag == 1, degrade_flag == 0, sensitivity >= 0.9 (GEDI L2A user guide
recommendations); night shots (solar_elevation < 0); power beams (BEAM0101, 0110, 1000, 1011).
Output: $BULK_ROOT/gedi/gedi02a_pool_<year>.parquet (one row per shot), outputs/tables/p48_gedi_pool_summary.csv
(per year: shots, quality shots, rh98 quantiles inside the former water surface, comparison with ICESat-2 p47).
Usage: python scripts/p48_gedi_pool.py --years 2024 2025 [--max-granules N]
Cost: ~30-80 MB transferred per granule (lat/lon + subset), i.e. a few GB for two years; hours, not days.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

BBOX = (33.30, 46.70, 35.40, 47.90)
OUT = CFG.BULK_ROOT / "gedi"
BEAMS = ["BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011", "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"]
POWER = {"BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"}
BLOCK = 32 * 2 ** 20
RH = {"rh25": 25, "rh50": 50, "rh75": 75, "rh95": 95, "rh98": 98, "rh100": 100}


def login():
    import earthaccess, os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if os.environ.get("EARTHDATA_TOKEN"):
        return earthaccess.login(strategy="environment")
    return earthaccess.login()


def read_granule(f, year: int) -> pd.DataFrame:
    import h5py
    rows = []
    with h5py.File(f, "r") as h:
        for b in BEAMS:
            if b not in h:
                continue
            g = h[b]
            tb = time.time()
            lat = g["lat_lowestmode"][:]; lon = g["lon_lowestmode"][:]
            m = (lon >= BBOX[0]) & (lon <= BBOX[2]) & (lat >= BBOX[1]) & (lat <= BBOX[3])
            if not m.any():
                print(f"    {b}: {len(lat)} shots, none in bbox ({time.time()-tb:.0f}s)", flush=True); continue
            idx = np.flatnonzero(m); i0, i1 = int(idx.min()), int(idx.max()) + 1     # contiguous slice reads (fancy indexing over HTTP is element-wise)
            sl = slice(i0, i1); keep = m[sl]

            def rd(name):
                return g[name][sl][keep]
            rh = g["rh"][sl, :][keep]
            d = dict(shot_number=rd("shot_number"), delta_time=rd("delta_time"), lat=lat[sl][keep], lon=lon[sl][keep], elev_lowestmode=rd("elev_lowestmode"),
                     quality_flag=rd("quality_flag"), degrade_flag=rd("degrade_flag"), sensitivity=rd("sensitivity"), solar_elevation=rd("solar_elevation"),
                     beam=b, power_beam=b in POWER)
            for k, p in RH.items():
                d[k] = rh[:, p]
            if "land_cover_data" in g and "leaf_off_flag" in g["land_cover_data"]:
                d["leaf_off_flag"] = g["land_cover_data"]["leaf_off_flag"][sl][keep]
            print(f"    {b}: {len(lat)} shots, {int(m.sum())} in bbox ({time.time()-tb:.0f}s)", flush=True)
            rows.append(pd.DataFrame(d))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.Timestamp("2018-01-01") + pd.to_timedelta(df.delta_time, unit="s")
    df["year"] = year
    return df


def main() -> None:
    import earthaccess
    ap = argparse.ArgumentParser(); ap.add_argument("--years", type=int, nargs="*", default=[2024, 2025])
    ap.add_argument("--max-granules", type=int, default=0,
                    help="probe run: read only the first N granules. A probe NEVER publishes to the canonical path.")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="publish to the canonical path even though granules failed to read or the existing product would shrink")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    login()
    summ, ledgers = [], []
    for y in a.years:
        res = earthaccess.search_data(short_name="GEDI02_A", version="002", bounding_box=BBOX, temporal=(f"{y}-01-01", f"{y}-12-31"), count=2000)
        if a.max_granules:
            res = res[: a.max_granules]
        print(f"{y}: {len(res)} granules", flush=True)
        # streaming (fsspec + h5py) took 133 s per beam because the chunked datasets are scattered through the file:
        # download the granule to a scratch dir on F, read it locally, delete it (2026-09-19)
        tmp = OUT / "_tmp"; tmp.mkdir(parents=True, exist_ok=True)
        parts, t0, ledger = [], time.time(), []
        for i, r in enumerate(res, 1):
            gid = r["meta"]["native-id"]
            try:
                td = time.time(); local = earthaccess.download([r], local_path=str(tmp), threads=1)[0]
                print(f"  downloaded {Path(local).stat().st_size/2**20:.0f} MB in {time.time()-td:.0f}s", flush=True)
                try:
                    df = read_granule(str(local), y)
                finally:
                    Path(local).unlink(missing_ok=True)
                if len(df):
                    df["granule"] = gid; parts.append(df)
                ledger.append(dict(year=y, granule=gid, status="ok" if len(df) else "empty", shots=len(df), error=""))
                print(f"  [{i}/{len(res)}] {gid[:40]} shots {len(df)} {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                ledger.append(dict(year=y, granule=gid, status="failed", shots=0, error=f"{type(e).__name__}: {str(e)[:120]}"))
                print(f"  [{i}/{len(res)}] FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
        L = pd.DataFrame(ledger); ledgers.append(L)
        n_ok = int((L.status == "ok").sum()); n_empty = int((L.status == "empty").sum()); n_failed = int((L.status == "failed").sum())
        if not parts:
            print(f"{y}: nothing read ({n_failed} failures) -- the existing product is left untouched", flush=True)
            continue
        D = pd.concat(parts, ignore_index=True)
        # PUBLICATION (repair 2026-09-20, finding F09). A truncated or partly failed run used to overwrite the canonical
        # year product unconditionally, so `--max-granules 1` or one network failure could replace a complete year with a
        # subset and nothing in the file would say so. Publication to the canonical name now requires: not a probe run,
        # no failed granule, and no granule that the existing product has and this run does not. Anything else is written
        # beside the canonical product under an explicit name, and the canonical one is left alone.
        canonical = OUT / f"gedi02a_pool_{y}.parquet"
        dest, why = canonical, ""
        if a.max_granules:
            dest, why = OUT / f"gedi02a_pool_{y}_probe.parquet", f"probe run (--max-granules {a.max_granules})"
        elif n_failed and not a.allow_incomplete:
            dest, why = OUT / f"gedi02a_pool_{y}_incomplete.parquet", f"{n_failed} granule(s) failed to read"
        elif canonical.exists():
            try:
                old = set(pd.read_parquet(canonical, columns=["granule"]).granule.unique())
            except Exception:
                old = set()
            lost = old - set(D.granule.unique())
            if lost and not a.allow_incomplete:
                dest, why = OUT / f"gedi02a_pool_{y}_incomplete.parquet", f"{len(lost)} granule(s) present in the existing product are missing from this run"
        if why:
            print(f"  REFUSING to publish {y} to {canonical.name}: {why}. Writing {dest.name} instead "
                  f"(pass --allow-incomplete to override).", flush=True)
        part = dest.with_suffix(".part")                       # atomic: a crash mid-write cannot leave a half product in place
        D.to_parquet(part, index=False); os.replace(part, dest)
        q = D[(D.quality_flag == 1) & (D.degrade_flag == 0) & (D.sensitivity >= 0.9)]
        summ.append(dict(year=y, granules_selected=len(res), granules_read=n_ok, granules_empty=n_empty, granules_failed=n_failed,
                         published_to=dest.name, complete=not bool(why),
                         shots=len(D), quality_shots=len(q), rh98_p50=q.rh98.median(), rh98_p75=q.rh98.quantile(.75), rh98_p95=q.rh98.quantile(.95),
                         share_rh98_ge2m=(q.rh98 >= 2).mean(), night_share=(q.solar_elevation < 0).mean()))
        print(summ[-1], flush=True)
    pd.DataFrame(summ).to_csv(CFG.TABLES / "p48_gedi_pool_summary.csv", index=False)
    if ledgers:
        pd.concat(ledgers, ignore_index=True).to_csv(CFG.TABLES / "p48_gedi_granule_ledger.csv", index=False)
        print("-> outputs/tables/p48_gedi_granule_ledger.csv")


if __name__ == "__main__":
    main()
