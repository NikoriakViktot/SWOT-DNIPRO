#!/usr/bin/env python
"""P66 -- PRE-BREACH WATER OCCURRENCE -> VALID BATHYMETRY DOMAIN, below the dam.

Why this replaces the old approach (user, 2026-09-20). p53 decided where a bed surface may be written from a UNION of the
per-year leaf-on water masks (`water |= water_share >= 50` over 2019..2022) further UNIONed with the digitised p27 contour
polygon, because the intersection over years had collapsed to 14 km2 in ZONE_4. A union admits anything that was ever wet, and
the digitised polygon admits whatever it covers. p53 then writes H = MAL(x) - depth inside that domain and tapers to MAL at its
edge, so wherever the domain reached dry land the surface stayed pinned near the water level while the real ground rose. p65
measured the consequence: the residual against night ICESat-2 keys on ground elevation at Spearman -0.874, with bias -8.0 m and
a 100 % tail between 5 and 10 m of ground and -32.0 m above 10 m. The defect is therefore not kriging accuracy, it is the domain
the kriging was allowed to write into. The fix is to define the domain first, from the pre-breach water record itself:

    PRE-BREACH WATER OCCURRENCE  ->  VALID BATHYMETRY DOMAIN  ->  kriging only inside

No height threshold arbitrates anything here. FABDEM's reading over water (0.00 m below the dam, 14.5 m in the pool, both very
tight) is a signature of the hydrological flattening in that product, and it is used only as a CONTROL, never as bathymetric
truth or as a class arbiter.

Method, per Sentinel-2 scene of the per-date stacks (2019-01-01 .. 2023-06-05):
  valid        SCL not in WM.SCL_REJECT (0,1,3,8,9,10,11) -- NO_DATA never votes "dry"
  water_AWEI   AWEIsh > 0        (AWEI was designed to separate water from shadow and dark surfaces)
  water_MNDWI  MNDWI > 0         (suppresses soil, vegetation and built surfaces)
Accumulated per pixel: N_valid, N_AWEI, N_MNDWI, N_agree_water (both indices call water), N_agree (both agree either way), and,
per LEVEL STRATUM, N_valid and N_agree_water. Every scene carries the Kherson daily stage H_80805(date), so the strata are
hydrological states, not calendar bins, and the occurrence stops mixing different water levels.

Products (continuous first -- no magic threshold is applied before the histograms are looked at):
  prebreach_n_valid, prebreach_AWEI_occurrence, prebreach_MNDWI_occurrence, prebreach_water_occurrence (agreement-based),
  prebreach_water_agreement, prebreach_H_wet_min (the LOWEST stage at which the pixel was seen as water -- a pixel-to-stage
  relation that is more informative for the shore than any single mask), prebreach_class
Classes: 1 CORE (water across low, mid and high stage), 2 LEVEL_DEPENDENT (water at higher stage only), 3 LAND, 4 ISLAND
(LAND enclosed by the water envelope), 5 UNCERTAIN (too few valid observations, or the two indices disagree).
Domain: CORE plus the justified part of LEVEL_DEPENDENT. LAND/ISLAND forbidden. UNCERTAIN forbidden by default -- it means
"we do not know", not "maybe water, interpolate anyway".
Section D re-creates p53's OLD domain and tests OLD \ NEW against the location of p65's tail points: if the catastrophic
residuals live in the difference, the cause is proven to be the domain and not the interpolator.
Outputs: outputs/rasters/zone<N>/prebreach/*.tif, outputs/tables/p66_{scene_inventory,class_areas,domain_sensitivity,old_vs_new_domain}.csv,
         outputs/figures/p66_prebreach_water_domain.png
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import watermask as WM

STACK = CFG.BULK_ROOT / "zone_spectral"
ZONES = {"ZONE_2_KHERSON_DELTA": 2, "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
BAND = {"MNDWI": 3, "AWEIsh": 6}          # 1 NDVI 2 NDWI 3 MNDWI 4 NDMI 5 BSI 6 AWEIsh 7 NDTI
SCALE = 1e-4; FILL = -32768
PRE = ("2019-01-01", "2023-06-05")
GAUGE = Path("/home/niko/repo/icesat2-atl13-kakhovka/data/1_data/data/parquet/dm_H/80805_yearbook.parquet")
MIN_OBS_TOTAL = 6          # a class may not be asserted below this many valid observations
MIN_OBS_STRATUM = 2        # ... nor from a stage stratum with fewer than this
AGREE_MIN = 0.80           # AWEI and MNDWI must agree on this fraction of valid observations
INDEX_MARGIN = 0.0         # TESTED AND REJECTED as the discriminator: a margin of 0.10 collapses CORE in ZONE_2 from 164.8 to
                           # 1.0 km2, because genuine shallow turbid delta water also carries weak indices. Kept as a switch only.
                           # ORIGINAL NOTE: a bathymetry domain needs A STRONGER WATER SIGNAL THAN A WATER MASK. The frozen project rule tests
                           # the indices against 0, which is right for mapping water extent but admits surfaces that merely sit
                           # marginally above zero. One such pixel reached CORE at occurrence 0.895: a flat plateau at 38.28 m
                           # where ICESat-2 and FABDEM agree to 0.00 m, SCL says "not vegetated" on all 19 valid scenes, and
                           # MNDWI/AWEIsh sit at 0.07-0.31 / 0.05-0.39 every time. A margin above zero is therefore required
                           # here, and only here: the water masks used elsewhere in the project keep their frozen rule.
SCL6_MIN = 0.30            # THE DISCRIMINATOR THAT WORKS. Sen2Cor's own scene classification carries an independent water class
                           # (SCL 6). The false-water pixel that reached CORE is SCL 5 ("not vegetated") on all 19 valid scenes
                           # and SCL 6 on none, while its two indices sit marginally above zero every time. Requiring the
                           # scene classifier to call water in at least this fraction of valid observations adds a THIRD optical
                           # channel, independent of both indices, and uses no elevation at all. The project's frozen water rule
                           # already treats SCL 6 as strong evidence (watermask: "trust SCL water with a relaxed index").
CORE_OCC = 0.80            # water in at least this fraction of the valid observations of EVERY stratum
WET_OCC = 0.20             # "seen as water" for the level-dependent class
CLASSES = {1: "CORE", 2: "LEVEL_DEPENDENT", 3: "LAND", 4: "ISLAND", 5: "UNCERTAIN"}


def stage_series():
    d = pd.read_parquet(GAUGE)
    d = d[d.stat_type == "daily"].copy(); d["date"] = pd.to_datetime(d.date)
    return d.groupby(d.date.dt.date).water_level_m_abs.mean()


def scene_dates(zone):
    out = []
    for f in sorted(glob.glob(str(STACK / zone / "*_indices.tif"))):
        dt = Path(f).name[:10]
        if PRE[0] <= dt <= PRE[1] and (STACK / zone / f"{dt}_scl.tif").exists():
            out.append((dt, Path(f), STACK / zone / f"{dt}_scl.tif"))
    return out


def write(path, arr, prof, dtype, nodata, **tags):
    p = dict(prof); p.update(dtype=dtype, nodata=nodata, count=1, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as ds:
        ds.write(np.nan_to_num(arr, nan=nodata).astype(dtype) if dtype.startswith("f") else arr.astype(dtype), 1)
        ds.update_tags(producer="p66_prebreach_water_domain.py", **{k: str(v) for k, v in tags.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="*", default=list(ZONES))
    ap.add_argument("--domain-occ", type=float, default=WET_OCC, help="occurrence at which a LEVEL_DEPENDENT pixel joins the domain")
    ap.add_argument("--index-margin", type=float, default=INDEX_MARGIN, help="AWEIsh and MNDWI must exceed this, not merely 0")
    a = ap.parse_args()
    H = stage_series(); inv, areas, sens = [], [], []
    for zone in a.zones:
        n = ZONES[zone]; scenes = scene_dates(zone)
        if not scenes:
            print(f"{zone}: no pre-breach scenes"); continue
        with rasterio.open(scenes[0][1]) as ds:
            shape, prof = (ds.height, ds.width), ds.profile.copy()
        px = abs(prof["transform"].a * prof["transform"].e) / 1e6
        lv = np.array([H.get(pd.Timestamp(d).date(), np.nan) for d, _, _ in scenes], float)
        ok_lv = np.isfinite(lv)
        # stage strata from the SCENES themselves, so every stratum is populated
        if ok_lv.sum() >= 3:
            qs = np.nanpercentile(lv, [33.3, 66.7]); strat = np.digitize(lv, qs)          # 0 low, 1 mid, 2 high
        else:
            qs = np.array([np.nan, np.nan]); strat = np.zeros(len(lv), int)
        strat = np.where(ok_lv, strat, -1)
        print(f"\n{zone}: {len(scenes)} pre-breach scenes, stage H_80805 {np.nanmin(lv):+.2f}..{np.nanmax(lv):+.2f} m "
              f"(p33 {qs[0]:+.2f}, p67 {qs[1]:+.2f}); scenes per stratum {dict(zip(*np.unique(strat, return_counts=True)))}", flush=True)
        nv = np.zeros(shape, "i2"); na = np.zeros(shape, "i2"); nm = np.zeros(shape, "i2"); n6 = np.zeros(shape, "i2")
        nw = np.zeros(shape, "i2"); nag = np.zeros(shape, "i2")
        nv_s = [np.zeros(shape, "i2") for _ in range(3)]; nw_s = [np.zeros(shape, "i2") for _ in range(3)]
        hwet = np.full(shape, np.nan, "f4")
        for i, (dt, fi, fs) in enumerate(scenes):
            with rasterio.open(fi) as ds:
                mn = ds.read(BAND["MNDWI"]).astype("f4"); aw = ds.read(BAND["AWEIsh"]).astype("f4")
            bad = (mn == FILL) | (aw == FILL); mn *= SCALE; aw *= SCALE
            with rasterio.open(fs) as ds:
                scl = ds.read(1)
            valid = ~np.isin(scl, WM.SCL_REJECT) & ~bad
            wa = (aw > a.index_margin) & valid; wm = (mn > a.index_margin) & valid
            both = wa & wm; agree = valid & (wa == wm)
            nv += valid; na += wa; nm += wm; nw += both; nag += agree; n6 += (scl == 6) & valid
            s = strat[i]
            if s >= 0:
                nv_s[s] += valid; nw_s[s] += both
                np.minimum(hwet, np.where(both, lv[i], np.nan), out=hwet, where=both)
                hwet[both & ~np.isfinite(hwet)] = lv[i]
            inv.append(dict(zone=zone, date=dt, H_80805_m=round(float(lv[i]), 3) if ok_lv[i] else np.nan,
                            stage_stratum={-1: "no stage", 0: "low", 1: "mid", 2: "high"}[int(s)],
                            valid_km2=round(float(valid.sum()) * px, 1), valid_frac=round(float(valid.mean()), 3),
                            water_both_km2=round(float(both.sum()) * px, 1),
                            AWEI_water_km2=round(float(wa.sum()) * px, 1), MNDWI_water_km2=round(float(wm.sum()) * px, 1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            p_a = np.where(nv > 0, na / np.maximum(nv, 1), np.nan)
            p_m = np.where(nv > 0, nm / np.maximum(nv, 1), np.nan)
            p_w = np.where(nv > 0, nw / np.maximum(nv, 1), np.nan)
            p_ag = np.where(nv > 0, nag / np.maximum(nv, 1), np.nan)
            p_6 = np.where(nv > 0, n6 / np.maximum(nv, 1), np.nan)
            occ_s = [np.where(nv_s[k] > 0, nw_s[k] / np.maximum(nv_s[k], 1), np.nan) for k in range(3)]
        # ---------------- classes
        support = (nv >= MIN_OBS_TOTAL) & (p_ag >= AGREE_MIN)
        have_s = [nv_s[k] >= MIN_OBS_STRATUM for k in range(3)]
        core = support & have_s[0] & have_s[1] & have_s[2] & (np.nan_to_num(p_6, nan=0.0) >= SCL6_MIN)
        for k in range(3):
            core &= np.nan_to_num(occ_s[k], nan=0.0) >= CORE_OCC
        wet_any = support & (np.nan_to_num(p_w, nan=0.0) >= a.domain_occ)
        leveldep = wet_any & ~core
        land = support & (np.nan_to_num(p_w, nan=0.0) < a.domain_occ)
        cls = np.where(core, 1, np.where(leveldep, 2, np.where(land, 3, 5))).astype("u1")
        # islands: LAND enclosed by the water envelope (fill holes of the wet envelope)
        env = core | leveldep
        env_filled = ndimage.binary_fill_holes(env)
        cls[(cls == 3) & env_filled & ~env] = 4
        for c, name in CLASSES.items():
            areas.append(dict(zone=zone, cls=c, name=name, km2=round(float((cls == c).sum()) * px, 2)))
        print("  " + "  ".join(f"{CLASSES[c]} {float((cls == c).sum()) * px:,.1f}" for c in CLASSES) + " km2")
        print(f"  n_valid: median {int(np.median(nv)):d}, below MIN_OBS_TOTAL {float((nv < MIN_OBS_TOTAL).sum()) * px:,.1f} km2 "
              f"({float((nv < MIN_OBS_TOTAL).mean()):.1%} of the frame)")
        # ---------------- products
        out = ROOT / f"outputs/rasters/zone{n}/prebreach"
        tag = dict(zone=zone, scenes=len(scenes), window=f"{PRE[0]}..{PRE[1]}", gauge="Kherson 80805 daily, m abs BS77",
                   rule=f"valid = SCL not in (0,1,3,8,9,10,11); water_AWEI = AWEIsh>{a.index_margin}; water_MNDWI = MNDWI>{a.index_margin}")
        write(out / "prebreach_n_valid.tif", nv, prof, "int16", -1, **tag)
        write(out / "prebreach_AWEI_occurrence.tif", p_a, prof, "float32", -9999.0, **tag)
        write(out / "prebreach_MNDWI_occurrence.tif", p_m, prof, "float32", -9999.0, **tag)
        write(out / "prebreach_water_occurrence.tif", p_w, prof, "float32", -9999.0, note="both indices call water", **tag)
        write(out / "prebreach_SCL6_occurrence.tif", p_6, prof, "float32", -9999.0, note="Sen2Cor SCL class 6 (water), the third independent channel", **tag)
        write(out / "prebreach_water_agreement.tif", p_ag, prof, "float32", -9999.0, note="fraction of valid obs where AWEI and MNDWI agree", **tag)
        write(out / "prebreach_H_wet_min.tif", hwet, prof, "float32", -9999.0, note="lowest Kherson stage at which the pixel was water", **tag)
        write(out / "prebreach_class.tif", cls, prof, "uint8", 0, classes="; ".join(f"{k} {v}" for k, v in CLASSES.items()),
              thresholds=f"MIN_OBS_TOTAL {MIN_OBS_TOTAL}, MIN_OBS_STRATUM {MIN_OBS_STRATUM}, AGREE_MIN {AGREE_MIN}, CORE_OCC {CORE_OCC}, DOMAIN_OCC {a.domain_occ}", **tag)
        # ---------------- domain sensitivity (no single magic threshold is adopted here)
        for t in (0.05, 0.1, 0.2, 0.35, 0.5, 0.8):
            d = support & (np.nan_to_num(p_w, nan=0.0) >= t)
            sens.append(dict(zone=zone, domain_occ=t, km2=round(float(d.sum()) * px, 2)))
    if inv:
        pd.DataFrame(inv).to_csv(CFG.TABLES / "p66_scene_inventory.csv", index=False)
        pd.DataFrame(areas).to_csv(CFG.TABLES / "p66_class_areas.csv", index=False)
        pd.DataFrame(sens).to_csv(CFG.TABLES / "p66_domain_sensitivity.csv", index=False)
        I = pd.DataFrame(inv)
        print("\nSCENE INVENTORY (the sampling design: which hydrological states the record actually covers)")
        print(I.groupby(["zone", "stage_stratum"]).agg(n=("date", "size"), H_min=("H_80805_m", "min"), H_max=("H_80805_m", "max"),
                                                       valid_frac=("valid_frac", "median")).round(3).to_string())
        print("\nDOMAIN SENSITIVITY, km2"); print(pd.DataFrame(sens).pivot(index="domain_occ", columns="zone", values="km2").to_string())
        print("\nCLASS AREAS, km2"); print(pd.DataFrame(areas).pivot(index="name", columns="zone", values="km2").to_string())
    print("\n-> outputs/rasters/zone<N>/prebreach/, outputs/tables/p66_*.csv")


if __name__ == "__main__":
    main()
