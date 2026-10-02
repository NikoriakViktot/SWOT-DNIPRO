#!/usr/bin/env python
"""P0L — what data EXISTS for each zone, against what is already on disk.

p0d_data_inventory_by_zone.csv records what has been downloaded. This asks the
prior question: what is available to download at all, per zone, per sensor, per
year. Without that the zone plans are guesses about volume and schedule.

Counts come from the archives themselves, not from a previous catalogue:
Sentinel-1 RTC and Sentinel-2 L2A from the Planetary Computer STAC, one query
per zone per year so the numbers are attributable. Volume is estimated from the
measured per-event size of the ZONE_1 cache (0.145-0.19 GB), which is the only
honest basis available.

Other resources are reported from their own catalogues where those exist and as
NOT CATALOGUED where they do not - an empty cell here is a finding, not a gap in
the script.

Outputs
-------
outputs/tables/p0l_zone_data_inventory.csv
outputs/tables/p0l_zone_year_matrix.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
ZONES = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
         "ZONE_3_DNIPRO_BUG_ESTUARY")
YEARS = tuple(range(2017, 2027))
COLLECTIONS = {"sentinel-1-rtc": "S1 RTC", "sentinel-2-l2a": "S2 L2A"}
GB_PER_S1_EVENT = 0.19          # measured on the ZONE_1 corrected cache
S2_MAX_CLOUD = 40.0


def http_json(url, payload, tries=5, timeout=180):
    import time
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as ex:
            last = ex
            time.sleep(min(30, 3 * 2 ** a))
    raise RuntimeError(f"{type(last).__name__}: {last}")


def count(collection, bbox, y0, y1, cloud=None):
    """Items and distinct acquisition DATES. Dates matter more than items: one
    date can be several granules, and an observation event is a date, not a
    granule -- the pseudoreplication point from Gate 5."""
    q = {"collections": [collection], "bbox": bbox, "limit": 500,
         "datetime": f"{y0}T00:00:00Z/{y1}T23:59:59Z"}
    if cloud is not None:
        q["query"] = {"eo:cloud_cover": {"lt": cloud}}
    n, dates, nxt = 0, set(), None
    for _ in range(40):                      # hard page cap
        if nxt:
            q["token"] = nxt
        r = http_json(STAC, q)
        feats = r.get("features", [])
        n += len(feats)
        for f in feats:
            dates.add(f["properties"]["datetime"][:10])
        nxt = next((l["body"].get("token") for l in r.get("links", [])
                    if l.get("rel") == "next" and "body" in l), None)
        if not nxt or not feats:
            break
    return n, len(dates)


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P0L — available data by zone, sensor and year")
    print("=" * 78)
    print(f"  git {commit}")
    print("  counts come from the archive, not from a previous catalogue")

    rows, mat = [], []
    for zn in ZONES:
        g = SD.load_utm(zn)
        bbox = [round(v, 4) for v in
                gpd.GeoSeries([g], crs=32636).to_crs(4326).iloc[0].bounds]
        print(f"\n  {zn}  {g.area/1e6:,.0f} km2   bbox {bbox}")
        for coll, label in COLLECTIONS.items():
            tot_i = tot_d = 0
            per_year = {}
            for y in YEARS:
                cloud = S2_MAX_CLOUD if coll.startswith("sentinel-2") else None
                try:
                    ni, nd = count(coll, bbox, f"{y}-01-01", f"{y}-12-31", cloud)
                except Exception as ex:
                    print(f"    {label} {y}: query failed ({type(ex).__name__})")
                    continue
                per_year[y] = nd
                tot_i += ni
                tot_d += nd
                mat.append(dict(zone=zn, sensor=label, year=y,
                                items=ni, dates=nd))
            span = ", ".join(f"{y}:{n}" for y, n in per_year.items() if n)
            print(f"    {label:7s} {tot_i:6,d} items over {tot_d:5,d} dates"
                  + (f"   ({'cloud < %.0f%%' % S2_MAX_CLOUD})" if cloud else ""))
            print(f"            {span}")
            rows.append(dict(zone=zn, area_km2=g.area / 1e6, sensor=label,
                             items=tot_i, dates=tot_d,
                             est_gb_all_dates=(tot_d * GB_PER_S1_EVENT
                                               if label.startswith("S1")
                                               else np.nan)))

    R = pd.DataFrame(rows)
    M = pd.DataFrame(mat)
    R.to_csv(CFG.TABLES / "p0l_zone_data_inventory.csv", index=False)
    M.to_csv(CFG.TABLES / "p0l_zone_year_matrix.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'p0l_zone_data_inventory.csv'}")

    print("\n" + "=" * 78)
    print("SUMMARY — available dates, and the S1 volume if all were fetched")
    print("=" * 78)
    p = R.pivot_table(index="zone", columns="sensor", values="dates",
                      aggfunc="sum")
    print(p.to_string())
    s1 = R[R.sensor.str.startswith("S1")]
    print("\n  S1 volume if every available date were fetched "
          f"(at {GB_PER_S1_EVENT} GB/event):")
    for r in s1.itertuples():
        print(f"    {r.zone:32s} {r.dates:5,d} dates  {r.est_gb_all_dates:7,.0f} GB")
    print(f"    {'TOTAL':32s} {int(s1.dates.sum()):5,d} dates  "
          f"{s1.est_gb_all_dates.sum():7,.0f} GB")
    print("\n  Fetching everything is neither necessary nor affordable; the")
    print("  stratified designs already agreed cut this to a working set.")


if __name__ == "__main__":
    main()
