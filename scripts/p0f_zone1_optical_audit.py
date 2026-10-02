#!/usr/bin/env python
"""ZONE 1 rebuild, step 2 — what optical do we already have, per contour?

Before re-downloading anything, establish which of the 168 Sentinel-2 SAFE
archives already on disk (local 79 + drive F 89, ~156 GB) actually cover each
target contour date and the reservoir core, and where the genuine gaps are.
Raw SAFE archives are grid-independent, so they survive the domain correction
and must not be fetched again.

The contour dates come from the frozen hist23 inventory. Coverage is judged
per date against KAKHOVKA_RESERVOIR_CORE using the MGRS tile footprints, not
by assuming a tile list.

Nothing is downloaded here.

Outputs
-------
outputs/tables/p0f_zone1_optical_audit.csv
outputs/tables/p0f_zone1_optical_gaps.csv
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
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
LOCAL_RAW = ROOT / "data/raw/sentinel_p1_targeted"
F_RAW = CFG.BULK_ROOT / "data_swot/sentinel"
CORE = "KAKHOVKA_RESERVOIR_CORE"
COVER_OK = 0.995       # fraction of the core a date's tiles must cover


def scan(path: Path, origin: str):
    rows = []
    for f in (path.glob("*.zip") if path.exists() else []):
        m = re.match(r"(S2[ABC])_MSIL2A_(\d{8})T(\d{6})_.*?_(T\d{2}[A-Z]{3})_", f.name)
        if not m:
            continue
        rows.append(dict(platform=m.group(1), date=m.group(2),
                         time=m.group(3), tile=m.group(4),
                         origin=origin, path=str(f),
                         gb=round(f.stat().st_size / 1e9, 2)))
    return rows


def tile_geoms(tiles, core_ll):
    """MGRS tile footprints from STAC, so coverage is measured rather than
    assumed from a hard-coded tile list."""
    out = {}
    bbox = [round(v, 5) for v in core_ll.bounds]
    q = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 200,
         "datetime": "2023-06-01T00:00:00Z/2023-06-30T23:59:59Z"}
    r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                               headers={"Content-Type": "application/json"})
    for f in json.loads(urllib.request.urlopen(r, timeout=180).read())["features"]:
        t = "T" + f["properties"].get("s2:mgrs_tile", "")
        if t in tiles and t not in out:
            out[t] = shp_shape(f["geometry"])
    return out


def main() -> None:
    core_ll = SD.load_subzone(CORE)
    core_utm = SD.load_subzone_utm(CORE)
    z1_utm = SD.load_utm("ZONE_1_KAKHOVKA_LOWER_DNIPRO")
    print(f"{CORE}: {core_utm.area/1e6:,.0f} km2")
    print(f"ZONE_1 analysis AOI: {z1_utm.area/1e6:,.0f} km2")

    inv = pd.read_csv(CFG.TABLES / "prebreach_contour_inventory.csv")
    inv = inv.sort_values("H_evrf2019_m").reset_index(drop=True)
    targets = {}
    for r in inv.itertuples():
        for d in str(r.dates).split("|"):
            targets.setdefault(r.contour_id, []).append(d)
    print("\ntarget contour dates (frozen hist23 inventory):")
    for c, ds in targets.items():
        wse = float(inv.set_index("contour_id").loc[c, "H_evrf2019_m"])
        print(f"  {c} ({wse:.3f} m): {', '.join(ds)}")

    A = pd.DataFrame(scan(LOCAL_RAW, "local") + scan(F_RAW, "driveF"))
    print(f"\nraw Sentinel-2 SAFE archives on disk: {len(A)} "
          f"({A.gb.sum():.0f} GB)")
    print(A.groupby("origin").agg(n=("path", "size"), gb=("gb", "sum")).to_string())
    print(f"  distinct dates {A.date.nunique()}, tiles "
          f"{sorted(A.tile.unique())}")

    tg = tile_geoms(set(A.tile.unique()), core_ll)
    print(f"\nMGRS tile footprints resolved from STAC: {sorted(tg)}")
    cov = {}
    for t, g in tg.items():
        cov[t] = g.intersection(core_ll).area / core_ll.area
        print(f"  {t}: covers {100*cov[t]:5.1f}% of the reservoir core")

    # ------------------------------------------------ per-date coverage
    rows = []
    for d, grp in A.groupby("date"):
        ts = sorted(set(grp.tile))
        geoms = [tg[t] for t in ts if t in tg]
        frac = (unary_union(geoms).intersection(core_ll).area / core_ll.area
                if geoms else 0.0)
        rows.append(dict(date=d, tiles="|".join(ts), n_files=len(grp),
                         gb=round(grp.gb.sum(), 2),
                         origins="|".join(sorted(set(grp.origin))),
                         core_coverage=frac,
                         full_core=frac >= COVER_OK))
    C = pd.DataFrame(rows).sort_values("date")
    C.to_csv(CFG.TABLES / "p0f_zone1_optical_audit.csv", index=False)
    print(f"\ndates on disk with FULL reservoir-core coverage "
          f"(>= {100*COVER_OK:.1f}%): {int(C.full_core.sum())} of {len(C)}")

    # ------------------------------------------------ contour gaps
    print("\n" + "=" * 78)
    print("PER-CONTOUR GAP ANALYSIS")
    print("=" * 78)
    gaps = []
    for cid, ds in targets.items():
        for d in ds:
            key = d.replace("-", "")
            hit = C[C.date == key]
            have = len(hit) > 0
            frac = float(hit.core_coverage.iloc[0]) if have else 0.0
            status = ("FULL" if frac >= COVER_OK else
                      "PARTIAL" if frac > 0 else "MISSING")
            gaps.append(dict(contour_id=cid, target_date=d, on_disk=have,
                             core_coverage=frac, status=status,
                             tiles=hit.tiles.iloc[0] if have else "",
                             origins=hit.origins.iloc[0] if have else ""))
            print(f"  {cid} {d}: {status:<8} coverage {100*frac:5.1f}%"
                  f"{'  tiles ' + hit.tiles.iloc[0] if have else ''}")
    G = pd.DataFrame(gaps)
    G.to_csv(CFG.TABLES / "p0f_zone1_optical_gaps.csv", index=False)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for cid in targets:
        s = G[G.contour_id == cid]
        full = int((s.status == "FULL").sum())
        print(f"  {cid}: {full}/{len(s)} target dates fully covered on disk")
    need = G[G.status != "FULL"]
    print(f"\n  dates needing an optical fetch: {len(need)} of {len(G)}")
    if len(need):
        print(need[["contour_id", "target_date", "status",
                    "core_coverage"]].to_string(index=False))
    print(f"\n  raw archives already on disk are grid-independent and must "
          f"NOT be re-downloaded ({A.gb.sum():.0f} GB preserved)")
    print(f"-> {CFG.TABLES/'p0f_zone1_optical_audit.csv'}")
    print(f"-> {CFG.TABLES/'p0f_zone1_optical_gaps.csv'}")


if __name__ == "__main__":
    main()
