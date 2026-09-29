#!/usr/bin/env python
"""Build the small, tracked data layer the companion app reads.

The app must run from a plain clone (and on Streamlit Community Cloud), so it
never touches the companion ICESat-2 repository, the bulk volume or data/.
This script, run once in the full research environment, writes light
EPSG:4326 layers to outputs/paper/app_data/ from the reproducible sources:

    zones.geojson          paper zones R/F/D/E (outputs/paper/zones, frozen)
    flood_event.geojson    June 2023 S1 flood envelope inside F (event layer)
    water.geojson          pre-breach water below the dam (dnipro_water_domain)
    stations.csv           gauges and 2023 yearbook posts, zone, record span
    atl13_transects.csv    ICESat-2 transects used by the slope analysis
    crossings.csv          V4 SWOT-ICESat-2 crossings (ms7 evidence)
    supports.geojson       support circles of the V3/V6 closures
    swot_calorbit.geojson  the SWOT calibration-orbit reach, dam -> Kherson
    manifest.json          source files and their content hashes

Geometries are simplified in EPSG:32636 (metres) before reprojection; the
tolerances are recorded in the manifest. Areas shown in the app come from the
zone manifest, never from the simplified shapes.

    python app/prepare_app_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import p30_sea_post_yearbooks_2023 as P30

OUT = ROOT / "outputs/paper/app_data"
ZONES = ROOT / "outputs/paper/zones"
VAL = ROOT / "outputs/paper/validation"
FIGDATA = ROOT / "outputs/figure_data/FigD_kakhovka_profile_points.csv"
SIMPLIFY_M = {"zones": 50.0, "flood": 100.0, "water": 60.0}
SUPPORTS_KM = {("Rozumivka", "SWOT RiverSP (V3)"): 3.0, ("Kherson", "SWOT RiverSP (V6)"): 3.0,
               ("Kherson", "ICESat-2 (V6)"): 10.0, ("Kherson", "SWOT PIXC (V6)"): 1.0}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _to4326(g, name, tol):
    s = gpd.GeoSeries([g], crs=CFG.CRS_METRIC).simplify(tol, preserve_topology=True)
    return s.to_crs(4326).iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((ZONES / "paper1_zones_manifest.json").read_text())
    src = {}

    # zones
    rows = []
    for name, m in man["zones"].items():
        rows.append(dict(domain=name, letter=m["letter"], label=m["label"], area_km2=m["area_km2"],
                         geometry=_to4326(SD.load_utm(name), name, SIMPLIFY_M["zones"])))
    gpd.GeoDataFrame(rows, crs=4326).to_file(OUT / "zones.geojson", driver="GeoJSON")
    src["zones"] = man["file"]

    # flood event inside F
    F = SD.load_utm("F_LOWER_DNIPRO_FLOODWAY")
    env = SD.load_utm("event_flood_2023_s1_envelope").intersection(F)
    gpd.GeoDataFrame([dict(layer="June 2023 Sentinel-1 flood envelope within F (event layer, not a zone)",
                           area_km2=round(env.area / 1e6, 1),
                           geometry=_to4326(env, "flood", SIMPLIFY_M["flood"]))],
                     crs=4326).to_file(OUT / "flood_event.geojson", driver="GeoJSON")

    # pre-breach water below the dam (R is itself the reservoir footprint)
    water = SD.load_utm("dnipro_water_domain").difference(SD.load_utm("R_FORMER_KAKHOVKA_RESERVOIR"))
    gpd.GeoDataFrame([dict(layer="pre-breach water below the dam (ESA WorldCover 2021)",
                           geometry=_to4326(water, "water", SIMPLIFY_M["water"]))],
                     crs=4326).to_file(OUT / "water.geojson", driver="GeoJSON")
    src["water"] = "registry: dnipro_water_domain"

    # stations
    inv = pd.read_csv(VAL / "ms7_station_inventory.csv")
    st = [dict(id=i, name=n, lon=lo, lat=la, kind="reservoir gauge")
          for i, n, lo, la in CFG.RESERVOIR_GAUGES]
    st.append(dict(id=CFG.KHERSON_GAUGE[0], name=CFG.KHERSON_GAUGE[1], lon=CFG.KHERSON_GAUGE[2],
                   lat=CFG.KHERSON_GAUGE[3], kind="downstream gauge"))
    for pid, m in P30.POSTS.items():
        if pid in (80805,):
            continue
        st.append(dict(id=pid, name=m["name_en"], lon=m["lon"], lat=m["lat"],
                       kind="2023 yearbook post", status_2023=m.get("status_2023", "")))
    st = pd.DataFrame(st)
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(CFG.CRS_METRIC)
    st["zone"] = "none"
    for name, m in man["zones"].items():
        st.loc[SD.load_utm(name).covers(pts.geometry).values, "zone"] = m["letter"]
    st = st.merge(inv[["station_id", "first", "last", "n_days", "n_days_post_breach", "spans_breach"]],
                  left_on="id", right_on="station_id", how="left").drop(columns="station_id")
    st.to_csv(OUT / "stations.csv", index=False)

    # ATL13 transects used by the slope analysis
    t = pd.read_csv(FIGDATA, usecols=["date", "period", "rgt", "beam", "lon_mean", "lat_mean",
                                      "chain_km", "wse_m", "n_points"])
    t.to_csv(OUT / "atl13_transects.csv", index=False)
    src["atl13_transects"] = f"{FIGDATA.relative_to(ROOT)} ({sha(FIGDATA)})"

    # V4 crossings
    E = pd.read_csv(VAL / "ms7_evidence.csv", low_memory=False)
    c = E[(E.claim_id == "V4_SWOT_ICESAT_DIRECT") & (E.validation_path == "V4")]
    c = c[["overpass_id", "swot_pass_id", "date", "dt_hours", "zone", "n_nodes", "satellite_level_m",
           "icesat_level_m", "closure_residual_m", "included_primary", "exclusion_reason", "lon", "lat"]]
    c.to_csv(OUT / "crossings.csv", index=False)
    src["evidence"] = f"outputs/paper/validation/ms7_evidence.csv ({sha(VAL / 'ms7_evidence.csv')})"
    src["summary"] = f"outputs/paper/validation/ms7_summary.csv ({sha(VAL / 'ms7_summary.csv')})"

    # support circles and the calibration-orbit reach
    circ = []
    for (stname, what), km in SUPPORTS_KM.items():
        r = st[st.name == stname].iloc[0]
        p = gpd.GeoSeries([Point(r.lon, r.lat)], crs=4326).to_crs(CFG.CRS_METRIC).buffer(km * 1000)
        circ.append(dict(station=stname, support=what, radius_km=km, geometry=p.to_crs(4326).iloc[0]))
    gpd.GeoDataFrame(circ, crs=4326).to_file(OUT / "supports.geojson", driver="GeoJSON")
    line = LineString([CFG.KAKHOVKA_DAM, CFG.KHERSON_GAUGE[2:]])
    gpd.GeoDataFrame([dict(layer="SWOT calibration-orbit repeat, outlet to Kherson (daily, June 2023)",
                           geometry=line)], crs=4326).to_file(OUT / "swot_calorbit.geojson", driver="GeoJSON")

    (OUT / "manifest.json").write_text(json.dumps(dict(
        built_by="app/prepare_app_data.py", simplify_m=SIMPLIFY_M, sources=src,
        dam=CFG.KAKHOVKA_DAM, zones_manifest=f"outputs/paper/zones/paper1_zones_manifest.json"),
        indent=2) + "\n")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<24} {f.stat().st_size / 1e3:8.1f} kB")


if __name__ == "__main__":
    main()
