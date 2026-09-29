#!/usr/bin/env python
"""P22b -- prove the Ukraine-clipped SWOT archive is a faithful subset BEFORE the
483 GB of continent-wide originals are deleted.

"Faithful" is made concrete as four checks, each with a hard pass criterion.
Any failure -> exit 1 and nothing may be deleted.

  A. COMPLETENESS   every *.shp under data/raw/swot_l2_hr_{lakesp,riversp}_2.0
                    has exactly one manifest row, and no row is a failure.
  B. INTEGRITY      every clipped file opens, is EPSG:4326, and its feature count
                    equals the manifest's n_kept.
  C. FIDELITY       stratified random sample (seed CFG.SEED) of CLIPPED granules,
                    N per stratum (Obs / Prior / Unassigned / Reach / Node):
                    re-clip the ORIGINAL with the same registry domain and require
                    identical row count, identical column set and dtypes, identical
                    attribute values, and identical geometries (WKB equality).
  D. NO-LOSS        random sample of NO_FEATURES granules: the original has ZERO
                    features intersecting the domain (exact test, not bbox).

Output: outputs/tables/p22b_clip_verification.csv (one row per check/granule)
        and a PASS/FAIL summary. The deletion step is NOT in this script.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

SRC = ROOT / "data" / "raw"
DST = CFG.BULK_ROOT / "swot_ua"
PRODUCTS = ("swot_l2_hr_lakesp_2.0", "swot_l2_hr_riversp_2.0")
STRATA = ("LakeSP_Obs", "LakeSP_Prior", "LakeSP_Unassigned", "RiverSP_Reach", "RiverSP_Node")


def stratum(name: str) -> str:
    for s in STRATA:
        if s in name:
            return s
    return "other"


def reclip(shp: Path, dom, bbox) -> gpd.GeoDataFrame:
    g = gpd.read_file(shp, bbox=bbox)
    if len(g):
        g = g[g.intersects(dom)]
    return g


def same_frame(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> tuple[bool, str]:
    if len(a) != len(b):
        return False, f"rows {len(a)} vs {len(b)}"
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {set(a.columns) ^ set(b.columns)}"
    if len(a) == 0:
        return True, "both empty"
    # order-independent: sort both by WKB of geometry then all attribute columns as strings
    def key(df):
        k = df.geometry.apply(lambda g: g.wkb_hex if g is not None else "")
        return np.lexsort([df[c].astype(str).values for c in df.columns if c != "geometry"][::-1] + [k.values])
    a2 = a.iloc[key(a)].reset_index(drop=True)
    b2 = b.iloc[key(b)].reset_index(drop=True)
    for c in a.columns:
        if c == "geometry":
            ga = a2.geometry.apply(lambda g: g.wkb_hex if g is not None else "")
            gb = b2.geometry.apply(lambda g: g.wkb_hex if g is not None else "")
            if not (ga.values == gb.values).all():
                return False, f"geometry WKB differs in {(ga.values != gb.values).sum()} rows"
            continue
        va, vb = a2[c], b2[c]
        if str(va.dtype) != str(vb.dtype):
            return False, f"dtype {c}: {va.dtype} vs {vb.dtype}"
        if va.dtype.kind == "f":
            eq = np.isclose(va.astype(float).values, vb.astype(float).values, rtol=0, atol=0, equal_nan=True)
        else:
            eq = (va.astype(str).values == vb.astype(str).values)
        if not eq.all():
            return False, f"values differ in column {c} ({(~eq).sum()} rows)"
    return True, "identical"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-stratum", type=int, default=25)
    ap.add_argument("--n-nofeat", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(CFG.SEED)
    dom = SD.load("ukraine_swot_clip_domain")
    bbox = SD.as_bbox("ukraine_swot_clip_domain")
    M = pd.read_csv(CFG.TABLES / "p22_ukraine_clip_manifest.csv")
    rows = []
    fails = 0
    print("=" * 78)
    print("P22b -- verify the Ukraine clip before deleting originals")
    print("=" * 78)

    # ---- A. completeness --------------------------------------------------
    disk = {}
    for prod in PRODUCTS:
        for p in (SRC / prod).rglob("*.shp"):
            disk[p.stem] = (prod, p)
    man = set(M.granule)
    missing_rows = sorted(set(disk) - man)
    extra_rows = sorted(man - set(disk))
    dup = M.granule.duplicated().sum()
    bad = M[M.stage.isin(["INFO_FAILED", "READ_FAILED"])]
    okA = not missing_rows and dup == 0 and len(bad) == 0
    print(f"\nA. completeness: {len(disk):,} granules on disk, {len(man):,} manifest rows, "
          f"{len(missing_rows)} without a row, {len(extra_rows)} rows without a file, "
          f"{dup} duplicates, {len(bad)} failed -> {'PASS' if okA else 'FAIL'}")
    if missing_rows[:5]:
        print("   e.g. missing:", missing_rows[:5])
    rows.append(dict(check="A_completeness", granule="ALL", result="PASS" if okA else "FAIL",
                     detail=f"disk {len(disk)} rows {len(man)} missing {len(missing_rows)} extra {len(extra_rows)} dup {dup} failed {len(bad)}"))
    fails += 0 if okA else 1

    # ---- B. integrity of every clipped file ---------------------------------
    C = M[M.stage == "CLIPPED"]
    n_bad_open = n_bad_crs = n_bad_count = 0
    for r in C.itertuples():
        prod, src = disk.get(r.granule, (None, None))
        if src is None:
            n_bad_open += 1; continue
        dst = DST / src.relative_to(SRC)
        try:
            info = pyogrio.read_info(dst)
        except Exception:
            n_bad_open += 1; rows.append(dict(check="B_integrity", granule=r.granule, result="FAIL", detail="cannot open")); continue
        crs_ok = "4326" in str(info.get("crs", ""))
        cnt_ok = int(info.get("features", -1)) == int(r.n_kept)
        n_bad_crs += 0 if crs_ok else 1
        n_bad_count += 0 if cnt_ok else 1
        if not (crs_ok and cnt_ok):
            rows.append(dict(check="B_integrity", granule=r.granule, result="FAIL",
                             detail=f"crs {info.get('crs')} features {info.get('features')} vs n_kept {r.n_kept}"))
    okB = n_bad_open == 0 and n_bad_crs == 0 and n_bad_count == 0
    print(f"B. integrity: {len(C):,} clipped files -- unopenable {n_bad_open}, wrong CRS {n_bad_crs}, "
          f"count != manifest {n_bad_count} -> {'PASS' if okB else 'FAIL'}  ({time.time()-t0:.0f}s)")
    rows.append(dict(check="B_integrity", granule="ALL", result="PASS" if okB else "FAIL",
                     detail=f"{len(C)} files; unopenable {n_bad_open} crs {n_bad_crs} count {n_bad_count}"))
    fails += 0 if okB else 1

    # ---- C. fidelity on a stratified sample ----------------------------------
    C = C.assign(stratum=C.granule.map(stratum))
    n_id = n_diff = 0
    for s in STRATA:
        pool = C[C.stratum == s]
        take = pool.iloc[rng.choice(len(pool), min(a.n_per_stratum, len(pool)), replace=False)] if len(pool) else pool
        for r in take.itertuples():
            prod, src = disk[r.granule]
            dst = DST / src.relative_to(SRC)
            try:
                orig = reclip(src, dom, bbox)
                clip = gpd.read_file(dst)
                ok, why = same_frame(orig, clip)
            except Exception as ex:
                ok, why = False, f"{type(ex).__name__}: {str(ex)[:60]}"
            n_id += ok; n_diff += (not ok)
            rows.append(dict(check="C_fidelity", granule=r.granule, result="PASS" if ok else "FAIL", detail=f"{s}: {why}"))
            if not ok:
                print(f"   FIDELITY FAIL {r.granule}: {why}")
    okC = n_diff == 0
    print(f"C. fidelity: {n_id + n_diff} sampled granules re-clipped from the originals -- identical {n_id}, "
          f"different {n_diff} -> {'PASS' if okC else 'FAIL'}  ({time.time()-t0:.0f}s)")
    rows.append(dict(check="C_fidelity", granule="SAMPLE", result="PASS" if okC else "FAIL",
                     detail=f"{n_id} identical / {n_diff} different across {len(STRATA)} strata x {a.n_per_stratum}"))
    fails += 0 if okC else 1

    # ---- D. no-loss on NO_FEATURES sample -----------------------------------
    NF = M[M.stage == "NO_FEATURES"]
    take = NF.iloc[rng.choice(len(NF), min(a.n_nofeat, len(NF)), replace=False)]
    n_zero = n_lost = 0
    for r in take.itertuples():
        prod, src = disk[r.granule]
        try:
            n = len(reclip(src, dom, bbox))
        except Exception as ex:
            n = -1
        if n == 0:
            n_zero += 1
        else:
            n_lost += 1
            print(f"   NO-LOSS FAIL {r.granule}: original has {n} feature(s) in the domain")
        rows.append(dict(check="D_noloss", granule=r.granule, result="PASS" if n == 0 else "FAIL", detail=f"domain features in original: {n}"))
    okD = n_lost == 0
    print(f"D. no-loss: {len(take)} NO_FEATURES granules re-tested against the domain -- zero features {n_zero}, "
          f"features found {n_lost} -> {'PASS' if okD else 'FAIL'}  ({time.time()-t0:.0f}s)")
    rows.append(dict(check="D_noloss", granule="SAMPLE", result="PASS" if okD else "FAIL", detail=f"{n_zero} zero / {n_lost} with features"))
    fails += 0 if okD else 1

    out = CFG.TABLES / "p22b_clip_verification.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS -- the clipped archive is a faithful subset' if fails == 0 else f'FAIL ({fails} check(s)) -- DO NOT DELETE'}")
    print(f"-> {out}   ({time.time()-t0:.0f}s)")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
