#!/usr/bin/env python
"""What data exists, and where the gaps are, for each analysis zone.

Answers one question before any downloading starts: for ZONE_1 / ZONE_2 /
ZONE_3, what is already on disk, what is only catalogued, and what is simply
missing. Covers the local repository, the external archive on drive F, and
the Planetary Computer catalogue.

The important distinction throughout is between

    CATALOGUED  we know the scene exists (STAC), nothing is on disk
    ON DISK     bytes are local, but possibly on a superseded grid
    USABLE      on disk AND valid for the current domain

Anything built on the old P20 footprint counts as ON DISK but NOT usable,
because P20 truncates 87 km2 of real water and swallows Khortytsia; products
clipped to it have to be rebuilt on dnipro_water_domain.

Outputs
-------
outputs/tables/p0d_data_inventory_by_zone.csv
outputs/tables/p0d_s1_catalogue_by_zone.csv
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape as shp_shape

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
#: The bulk archive. Resolved through the config, never spelled out here: a
#: literal path reports "nothing downloaded" instead of failing when the volume
#: is not mounted, which is the same wrong answer this script exists to prevent.
FDRIVE = CFG.BULK_ROOT
ZONES = ["ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY"]
#: Kept in step with p10_s2_zone_fetch.WESTERN — the delta/liman tiles that the
#: eastern Kakhovka-side archive does not contain.
WESTERN_S2_TILES = {"36TUS", "36TUT", "36TVS", "36TVT"}
YEARS = (2019, 2020, 2021, 2022, 2023)


def stac_range(collection, bbox, d0, d1):
    out, token = [], None
    while True:
        q = {"collections": [collection], "bbox": bbox, "limit": 500,
             "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z"}
        if token:
            q["token"] = token
        r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                   headers={"Content-Type": "application/json"})
        j = json.loads(urllib.request.urlopen(r, timeout=180).read())
        out += j["features"]
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        token = nxt[0].get("body", {}).get("token")
        if not token:
            break
    return out


def du(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1e9
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def nfiles(path: Path, pat="*") -> int:
    return len([f for f in path.rglob(pat) if f.is_file()]) if path.exists() else 0


def main() -> None:
    zg = {z: SD.load(z) for z in ZONES}
    zu = {z: SD.load_utm(z) for z in ZONES}
    print("analysis zones:")
    for z in ZONES:
        print(f"  {z:<30} {zu[z].area/1e6:8,.0f} km2")

    rows = []

    # ---------------------------------------------------------- Sentinel-1
    print("\n" + "=" * 78)
    print("SENTINEL-1 RTC — catalogue vs disk, per zone")
    print("=" * 78)
    s1_rows = []
    for z in ZONES:
        bbox = [round(v, 5) for v in zg[z].bounds]
        feats = []
        for y in YEARS:
            for q0, q1 in (("01-01", "03-31"), ("04-01", "06-30"),
                           ("07-01", "09-30"), ("10-01", "12-31")):
                feats += stac_range("sentinel-1-rtc", bbox,
                                    f"{y}-{q0}", f"{y}-{q1}")
        # de-duplicate to observation events, and keep only those whose
        # footprint actually intersects the zone (the bbox is much larger)
        seen, ev = set(), []
        for f in feats:
            p = f["properties"]
            orb = p.get("sat:relative_orbit")
            if orb is None or not f.get("geometry"):
                continue
            key = (p["datetime"][:10], int(orb))
            g = shp_shape(f["geometry"])
            if not g.intersects(zg[z]):
                continue
            if key in seen:
                continue
            seen.add(key)
            ev.append(dict(date=key[0], relative_orbit=key[1],
                           orbit_state=p.get("sat:orbit_state", "")))
        E = pd.DataFrame(ev)
        s1_rows.append(E.assign(analysis_zone=z))
        print(f"  {z:<30} {len(feats):5d} assets -> {len(E):5d} events "
              f"over {E.date.nunique() if len(E) else 0} dates")
    S1 = pd.concat(s1_rows, ignore_index=True)
    S1.to_csv(CFG.TABLES / "p0d_s1_catalogue_by_zone.csv", index=False)

    # What is on disk. The per-zone event caches under CFG.S1_CACHE are the
    # live products, built on the registry grid by p0o_zone_s1_fetch.py.
    #
    # This block used to count only the three hist25/Gate-5/Gate-6 directories
    # under data_swot/processed/bathymetry/ and hard-code "ZONE_1 only, all on
    # the P20 grid". Those directories now hold 5, 0 and 0 .npz; the real cache
    # holds several hundred. The script therefore reported "nothing downloaded"
    # for ZONE_2 and ZONE_3 while 150 and 182 events sat on the volume, and
    # that false line was copied onward into DATA_GAPS.md. The legacy
    # directories are still listed, as legacy, so the P20-era leftovers stay
    # visible instead of being silently folded into the live count.
    legacy_dirs = {
        "s1_cache (hist25, P20 grid)": CFG.BULK_ROOT / "data_swot/processed/bathymetry/s1_cache",
        "s1b_cache (Gate 5, P20 grid)": CFG.BULK_ROOT / "data_swot/processed/bathymetry/s1b_cache",
        "s1b_events (Gate 6, P20 grid)": CFG.BULK_ROOT / "data_swot/processed/bathymetry/s1b_events",
    }
    print(f"\n  live per-zone event caches ({CFG.S1_CACHE}):")
    for z in ZONES:
        p = CFG.S1_CACHE / z
        print(f"    {z:<30} {nfiles(p,'*.npz'):4d} events  {du(p):6.1f} GB")
    print("\n  legacy P20-grid caches (superseded, not used by current scripts):")
    for nm, p in legacy_dirs.items():
        print(f"    {nm:<34} {nfiles(p,'*.npz'):4d} files  {du(p):6.1f} GB")

    for z in ZONES:
        cat = int((S1.analysis_zone == z).sum())
        p = CFG.S1_CACHE / z
        disk = nfiles(p, "*.npz")
        rows.append(dict(
            analysis_zone=z, dataset="Sentinel-1 RTC (events)",
            catalogued=cat, on_disk=disk,
            # products are built on the registry grid by p0o, so they are
            # valid for the current domain; what is NOT implied is that any
            # water classification has been run on them
            usable_for_current_domain=disk,
            gb_on_disk=round(du(p), 1),
            note=(f"p0o event cache on the registry grid; {disk} events"
                  if disk else "nothing downloaded")))

    # ---------------------------------------------------------- Sentinel-2
    print("\n" + "=" * 78)
    print("SENTINEL-2 — local raw, drive F raw, drive F water masks")
    print("=" * 78)
    local_raw = ROOT / "data/raw/sentinel_p1_targeted"
    f_raw = FDRIVE / "data_swot/sentinel"
    f_wm = FDRIVE / "data_swot/processed/water_masks"
    f_si = FDRIVE / "data_swot/processed/spectral_indices"

    def s2_dates(path, pat="*.zip"):
        out = []
        for f in (path.glob(pat) if path.exists() else []):
            m = re.search(r"_(\d{8})T\d{6}_", f.name)
            if m:
                out.append(m.group(1))
        return sorted(set(out))

    lr = s2_dates(local_raw)
    fr = s2_dates(f_raw)
    fw = sorted({m.group(1) for f in (f_wm.glob("*_water.tif") if f_wm.exists() else [])
                 if (m := re.search(r"_(\d{8})T\d{6}_", f.name))})
    print(f"  local data/raw/sentinel_p1_targeted : {nfiles(local_raw,'*.zip'):4d} zips, "
          f"{du(local_raw):6.1f} GB, {len(lr)} dates "
          f"({lr[0] if lr else '-'}..{lr[-1] if lr else '-'})")
    print(f"  F  data_swot/sentinel               : {nfiles(f_raw,'*.zip'):4d} zips, "
          f"{du(f_raw):6.1f} GB, {len(fr)} dates "
          f"({fr[0] if fr else '-'}..{fr[-1] if fr else '-'})")
    print(f"  F  processed/water_masks            : {nfiles(f_wm,'*_water.tif'):4d} masks, "
          f"{du(f_wm):6.1f} GB, {len(fw)} dates "
          f"({fw[0] if fw else '-'}..{fw[-1] if fw else '-'})")
    print(f"  F  processed/spectral_indices       : {nfiles(f_si,'*.tif'):4d} stacks, "
          f"{du(f_si):6.1f} GB")

    BREACH = "20230606"
    pre_lr = [d for d in lr if d < BREACH]
    pre_fr = [d for d in fr if d < BREACH]
    pre_fw = [d for d in fw if d < BREACH]
    print(f"\n  PRE-BREACH dates (< {BREACH}):")
    print(f"    local raw {len(pre_lr):3d} | F raw {len(pre_fr):3d} | "
          f"F water masks {len(pre_fw):3d}")
    print(f"    F water-mask pre-breach dates: {' '.join(pre_fw)}")

    cc = CFG.BULK_ROOT / "data_swot/processed/bathymetry/contour_cache"
    cc_dates = sorted({m.group(1) for f in (cc.glob("wm_*.npz") if cc.exists() else [])
                       if (m := re.search(r"wm_(\d{4}-\d{2}-\d{2})_", f.name))})
    print(f"\n  hist23 contour_cache (20 m, P20 grid): {len(cc_dates)} dates")
    print(f"    {' '.join(cc_dates)}")

    rows.append(dict(analysis_zone=ZONES[0], dataset="Sentinel-2 L2A raw (zip)",
                     catalogued=len(set(lr) | set(fr)),
                     on_disk=nfiles(local_raw, "*.zip") + nfiles(f_raw, "*.zip"),
                     usable_for_current_domain=nfiles(local_raw, "*.zip") + nfiles(f_raw, "*.zip"),
                     gb_on_disk=round(du(local_raw) + du(f_raw), 1),
                     note="raw scenes are grid-independent, so still usable"))
    rows.append(dict(analysis_zone=ZONES[0], dataset="Sentinel-2 water masks (F)",
                     catalogued=len(fw), on_disk=len(fw),
                     usable_for_current_domain=0, gb_on_disk=round(du(f_wm), 1),
                     note="105 dates 2017-2026 but clipped to the old reservoir "
                          "grid; no Feb/Mar 2019, so H2 is not covered"))
    rows.append(dict(analysis_zone=ZONES[0],
                     dataset="hist23 contour cache (H1/H2/H3 optical)",
                     catalogued=len(cc_dates), on_disk=len(cc_dates),
                     usable_for_current_domain=0,
                     gb_on_disk=round(du(cc), 3),
                     note="20 m on the P20 grid; must be rebuilt"))

    # ---------------------------------------------------------- zones 2 / 3
    liman = FDRIVE / "liman_water_occurrence"
    lower = FDRIVE / "lower_dnipro_water_occurrence"
    print("\n" + "=" * 78)
    print("ZONE 2 / ZONE 3 dedicated products on drive F")
    print("=" * 78)
    print(f"  lower_dnipro_water_occurrence : {nfiles(lower):3d} files, "
          f"{du(lower):.3f} GB  {[f.name for f in lower.glob('*')] if lower.exists() else []}")
    print(f"  liman_water_occurrence        : {nfiles(liman):3d} files, "
          f"{du(liman):.3f} GB  <-- EMPTY")
    rows.append(dict(analysis_zone=ZONES[1],
                     dataset="lower Dnipro water occurrence (F)",
                     catalogued=1, on_disk=nfiles(lower),
                     usable_for_current_domain=0, gb_on_disk=round(du(lower), 3),
                     note="single 2022 raster only; no S1, no per-event masks"))
    rows.append(dict(analysis_zone=ZONES[2],
                     dataset="liman water occurrence (F)",
                     catalogued=0, on_disk=nfiles(liman),
                     usable_for_current_domain=0, gb_on_disk=0.0,
                     note="directory is EMPTY; the run never produced output"))
    # Western-tile optical, fetched per zone by p10_s2_zone_fetch.py into a
    # single flat directory. Counted against each zone's own frozen manifest
    # rather than assumed absent: this block used to hard-code on_disk=0 and
    # "no optical fetched for this zone", which stopped being true once the
    # ZONE_2 pull completed (120/120).
    zf = FDRIVE / "s2_zone_fetch"
    disk_pairs = set()
    for f in (zf.glob("*.zip") if zf.exists() else []):
        m = re.search(r"_(\d{4})(\d{2})(\d{2})T\d{6}_.*_T(36[A-Z]{3})_", f.name)
        if m:
            disk_pairs.add((f"{m.group(1)}-{m.group(2)}-{m.group(3)}", m.group(4)))
    print(f"\n  F s2_zone_fetch (WESTERN tiles) : {nfiles(zf,'*.zip'):4d} zips, "
          f"{du(zf):6.1f} GB, {len(disk_pairs)} (date, tile) pairs")
    for z, man in (
        (ZONES[1], "p10_zone_2_kherson_delta_s2_manifest.csv"),
        (ZONES[2], "p10_zone_3_dnipro_bug_estuary_s2_manifest.csv"),
    ):
        path = CFG.TABLES / man
        if not path.exists():
            continue
        M = pd.read_csv(path)
        want = {(str(r.date)[:10], t.strip())
                for _, r in M.iterrows() for t in str(r.tiles).split("|")
                if t.strip() in WESTERN_S2_TILES}
        have = want & disk_pairs
        missing = sorted({t for _, t in (want - have)})
        print(f"    {z:<30} {len(have):4d} / {len(want):4d} pairs"
              + (f"   missing tiles: {' '.join(missing)}" if missing else "   COMPLETE"))
        rows.append(dict(
            analysis_zone=z, dataset="Sentinel-2 L2A raw, western tiles (F)",
            catalogued=len(want), on_disk=len(have),
            usable_for_current_domain=len(have),
            gb_on_disk=round(du(zf) * len(have) / max(len(disk_pairs), 1), 1),
            note=("manifest complete" if not missing
                  else f"{len(want)-len(have)} pairs missing "
                       f"({','.join(missing)})")))

    # ---------------------------------------------------------- other
    print("\n" + "=" * 78)
    print("OTHER LOCAL / F DATASETS")
    print("=" * 78)
    others = {
        # originals deleted 2026-09-16 after p22b verified the clip; see DATA_GAPS.md
        "SWOT RiverSP (UA-clipped, F)": FDRIVE / "swot_ua/swot_l2_hr_riversp_2.0",
        "SWOT LakeSP (UA-clipped, F)": FDRIVE / "swot_ua/swot_l2_hr_lakesp_2.0",
        "SWOT PIXC (local raw)": ROOT / "data/raw/pixc_nova_kakhovka",
        "meteo (local raw)": ROOT / "data/raw/meteo",
        "F dynamic_world_annual": FDRIVE / "dynamic_world_annual",
        "F bathymetry": FDRIVE / "data_swot/processed/bathymetry",
        "F atl08": FDRIVE / "data_swot/processed/atl08",
        "F emodnet": FDRIVE / "data_swot/processed/emodnet",
        "F soundings (SOUNDG_P)": FDRIVE / "data_swot/3529_Sounding_SOUNDG_P",
    }
    for nm, p in others.items():
        print(f"  {nm:<32} {nfiles(p):5d} files  {du(p):7.1f} GB"
              f"{'' if p.exists() else '   (absent)'}")

    D = pd.DataFrame(rows)
    D.to_csv(CFG.TABLES / "p0d_data_inventory_by_zone.csv", index=False)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(D.to_string(index=False, max_colwidth=52))
    print(f"\n-> {CFG.TABLES/'p0d_data_inventory_by_zone.csv'}")
    print(f"-> {CFG.TABLES/'p0d_s1_catalogue_by_zone.csv'}")


if __name__ == "__main__":
    main()
