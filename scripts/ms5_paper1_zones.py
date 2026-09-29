#!/usr/bin/env python
"""MS5 — the four analysis zones of paper 1 (R, F, D, E), frozen for submission.

WHY A NEW ZONE SET
------------------
Paper 1 v5 described its zones with the registry's download containers:

    ZONE_1 11,412 km2   reservoir + lower Dnipro + a 10 km land corridor
    ZONE_2  1,397.7     delta (the manuscript still said 1,677, pre-p0x)
    ZONE_3  7,621.3     estuary
    ZONE_4  6,919.5     canonical S1 flood envelope + 10 km (the manuscript
                        said 695, the unbuffered envelope of the first build
                        on 2026-09-14, which no longer exists on disk)

and called them "nested": F inside R. Measured on the registered geometry that
is false - 2,753 km2 (40 %) of ZONE_4 lies outside ZONE_1, and even the
envelope + 1 km puts 178 km2 outside it. The zones overlap by design (10 km
seams), because they were built to download Sentinel-1 without losing
coverage at a seam, not to describe the system.

The paper needs the opposite property: four hydro-geomorphically distinct
systems, in flow order, that do not overlap -

    R  former Kakhovka Reservoir        impounded -> drained
    F  lower Dnipro floodway            dam -> head of the Kherson delta
    D  Kherson delta (plavni)           distributaries, islands, reed wetland
    E  Dnipro-Buh estuary               liman to the sea

and F must NOT be defined by the 2023 event. The June 2023 Sentinel-1 flood
envelope is an observation made inside F, so it is carried as a separate event
layer, never as a zone boundary. The ZONE_1..4 containers stay untouched: other
branches (floodstate-eo, the S1 caches) were fetched on their grids.

CONSTRUCTION (no hand-drawn geometry; every input is a registered domain)
------------------------------------------------------------------------
R  = KAKHOVKA_RESERVOIR_CORE (2,144 km2): dnipro_water_domain (WorldCover
     2021, islands kept as holes) inside the pre-breach full-pool polygon SA_2,
     east of the dam. The reservoir is the unit, so R is its footprint, not a
     corridor around it.
F, D, E = the water BELOW THE DAM (dnipro_water_domain west of the dam cut)
     buffered by the same 10 km corridor p0c uses, cut into easting bands at
     the p0c landmarks:
         R | F   Kakhovka dam axis (dam easting - 500 m, as the R subzone)
         F | D   Kherson (head of the delta)
         D | E   CUT_2_3_EASTING, where the delta opens into the liman
     The corridor is grown from below-dam water only, so F does not inherit a
     shoreline strip of the reservoir north of the dam. F = its band minus R.
     D keeps only band components that reach the F | D cut (the p0x rule: a
     zone is one corridor); the Southern Bug fragment the band slices goes to
     E, whose system it belongs to.

The zones are pairwise disjoint (asserted to < 0.001 km2) and therefore
additive. They do not have to cover the old containers: land more than 10 km
from below-dam water, and land around the reservoir, belong to no zone.

EVENT LAYERS (not zones)
------------------------
event_flood_2023_s1_envelope        canonical June 2023 S1 envelope, as built by
                                    p0v (ZONE4_FLOOD_ENVELOPE_CANONICAL)
event_flood_2023_s1_envelope_1km    the same + 1 km
Their split across the zones is written to the manifest: the canonical
envelope also covers the lower reservoir near the dam, so "envelope" and "F"
are different objects and neither may stand in for the other.

Outputs
-------
outputs/paper/zones/paper1_zones_utm.geojson     (EPSG:32636, tracked)
outputs/paper/zones/paper1_zones_manifest.json   areas, hashes, overlaps, commit
outputs/paper/zones/paper1_zones_summary.csv
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import p0c_build_analysis_zones as P0C
import p30_sea_post_yearbooks_2023 as P30

UTM = CFG.CRS_METRIC
OUTDIR = ROOT / "outputs/paper/zones"
ZONES_FILE = OUTDIR / "paper1_zones_utm.geojson"
MANIFEST = OUTDIR / "paper1_zones_manifest.json"
SUMMARY = OUTDIR / "paper1_zones_summary.csv"
ENVELOPE_SRC = ROOT / "data/processed/domains/zone4_flood_composite_utm.geojson"
ENVELOPE_LAYER = "ZONE4_FLOOD_ENVELOPE_CANONICAL"

CORRIDOR_BUFFER_KM = P0C.CORRIDOR_BUFFER_KM   # the same 10 km as the containers
DAM_CUT_OFFSET_M = 500.0                      # as p0c's reservoir-core split
EVENT_BUFFER_KM = 1.0
OVERLAP_TOL_KM2 = 1e-3

ZONES = {
    "R_FORMER_KAKHOVKA_RESERVOIR": dict(
        letter="R", label="Reservoir", legacy="ZONE_1 subzone KAKHOVKA_RESERVOIR_CORE",
        description="Former Kakhovka Reservoir footprint, DniproHES to the dam."),
    "F_LOWER_DNIPRO_FLOODWAY": dict(
        letter="F", label="Floodway", legacy="part of ZONE_1 below the dam; "
        "replaces ZONE_4 (event-defined) as the zone",
        description="Lower Dnipro from the Kakhovka dam to the head of the Kherson "
                    "delta, with the Inhulets confluence; 10 km corridor."),
    "D_KHERSON_DELTA": dict(
        letter="D", label="Delta", legacy="ZONE_2 without its 5 km seam overlaps",
        description="Kherson delta: distributaries, islands, plavni; 10 km corridor."),
    "E_DNIPRO_BUG_ESTUARY": dict(
        letter="E", label="Estuary", legacy="ZONE_3 without its 5 km seam overlap",
        description="Dnipro-Buh estuary, lower Southern Bug and Inhul mouth, to the sea."),
}
EVENTS = {
    "event_flood_2023_s1_envelope": 0.0,
    "event_flood_2023_s1_envelope_1km": EVENT_BUFFER_KM,
}


def _parts(g):
    return list(getattr(g, "geoms", [g]))


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build() -> tuple[dict, dict, dict]:
    L = {k: P0C.utm_point(*v) for k, v in P0C.LANDMARKS.items()}
    cut_dam = L["Kakhovka_dam_Nova_Kakhovka"].x - DAM_CUT_OFFSET_M
    cut_fd = L["Kherson"].x
    cut_de = P0C.CUT_2_3_EASTING
    BIG = 1e7

    water = SD.load_utm("dnipro_water_domain")
    R = SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE")
    below = water.intersection(box(-BIG, -BIG, cut_dam, BIG))
    corridor = below.buffer(CORRIDOR_BUFFER_KM * 1000.0)

    def band(e0, e1):
        return corridor.intersection(box(e0, -BIG, e1, BIG))

    F = band(cut_fd, cut_dam).difference(R)
    d_band = band(cut_de, cut_fd)
    d_keep = [p for p in _parts(d_band) if p.bounds[2] >= cut_fd - 1.0]
    D = unary_union(d_keep)
    d_rest = d_band.difference(D)
    E = unary_union([band(-BIG, cut_de), d_rest])
    zones = dict(zip(ZONES, (R, F, D, E)))

    env_src = gpd.read_file(ENVELOPE_SRC).set_index("layer")
    SD.assert_projected_32636(env_src, "flood envelope")
    env = env_src.geometry[ENVELOPE_LAYER]
    events = {k: (env if b == 0 else env.buffer(b * 1000.0)) for k, b in EVENTS.items()}

    cuts = dict(R_F_dam_easting_m=cut_dam, F_D_kherson_easting_m=cut_fd,
                D_E_delta_liman_easting_m=cut_de,
                d_band_fragments_to_E_km2=round(d_rest.area / 1e6, 3))
    return zones, events, cuts


def write(zones: dict, events: dict) -> gpd.GeoDataFrame:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, g in zones.items():
        m = ZONES[name]
        rows.append(dict(domain=name, kind="analysis_zone", letter=m["letter"],
                         label=m["label"], legacy=m["legacy"],
                         description=m["description"], geometry=g))
    for name, g in events.items():
        rows.append(dict(domain=name, kind="event_layer", letter="", label="",
                         legacy=f"{ENVELOPE_SRC.name}:{ENVELOPE_LAYER}",
                         description=f"June 2023 Sentinel-1 flood envelope + "
                                     f"{EVENTS[name]:g} km; an observation, not a zone",
                         geometry=g))
    gdf = gpd.GeoDataFrame(rows, crs=UTM)
    gdf.to_file(ZONES_FILE, driver="GeoJSON", COORDINATE_PRECISION=2)
    # Everything downstream is computed from the file as written, so the hashes
    # in the manifest are the hashes any reader of the file will reproduce.
    return gpd.read_file(ZONES_FILE).set_index("domain")


def main() -> None:
    zones, events, cuts = build()
    gdf = write(zones, events)
    Z = {k: gdf.geometry[k] for k in ZONES}
    V = {k: gdf.geometry[k] for k in EVENTS}

    # ---- QA: disjoint, non-empty, one corridor each
    overlaps = {f"{a}|{b}": Z[a].intersection(Z[b]).area / 1e6
                for a, b in itertools.combinations(Z, 2)}
    for k, v in overlaps.items():
        assert v < OVERLAP_TOL_KM2, f"zones overlap: {k} {v:.4f} km2"
    for k, g in Z.items():
        assert g.area > 0, f"{k} is empty"

    # ---- where the event lies, per zone
    union = unary_union(list(Z.values()))
    ev_split = {}
    for ek, eg in V.items():
        d = {zk: round(eg.intersection(zg).area / 1e6, 2) for zk, zg in Z.items()}
        d["outside_all_zones"] = round(eg.difference(union).area / 1e6, 2)
        d["total"] = round(eg.area / 1e6, 2)
        ev_split[ek] = d

    # ---- which zone every station falls in (for Table 1 and the figures)
    st = [(i, n, lon, lat, "gauge") for i, n, lon, lat in CFG.RESERVOIR_GAUGES]
    st.append((*CFG.KHERSON_GAUGE, "gauge"))
    for sid, m in P30.POSTS.items():
        st.append((sid, m["name_en"], m["lon"], m["lat"], "yearbook_post"))
    sgdf = gpd.GeoDataFrame(st, columns=["id", "name", "lon", "lat", "kind"],
                            geometry=gpd.points_from_xy([s[2] for s in st],
                                                        [s[3] for s in st]),
                            crs=4326).to_crs(UTM)
    station_zone = {}
    for _, r in sgdf.iterrows():
        hit = [ZONES[k]["letter"] for k, g in Z.items() if g.covers(r.geometry)]
        station_zone[f'{r["id"]} {r["name"]}'] = hit[0] if hit else "none"

    # ---- ATL13 support: the slope analysis lives in R
    atl = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet",
                          columns=["lon", "lat"])
    ap = gpd.GeoSeries(gpd.points_from_xy(atl.lon, atl.lat), crs=4326).to_crs(UTM)
    atl_in_r = float(shapely.contains_xy(Z["R_FORMER_KAKHOVKA_RESERVOIR"],
                                         ap.x.values, ap.y.values).mean())

    legacy = {k: SD.load_utm(k) for k in (
        "ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
        "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")}
    l1, l4 = legacy["ZONE_1_KAKHOVKA_LOWER_DNIPRO"], legacy["ZONE_4_DAM_TO_KHERSON_FLOODWAY"]

    manifest = dict(
        purpose="Analysis zones of paper 1 (R/F/D/E), frozen for submission.",
        built_by="scripts/ms5_paper1_zones.py",
        built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--", "scripts", "src", "config")),
        crs=UTM,
        file=str(ZONES_FILE.relative_to(ROOT)),
        construction=dict(corridor_buffer_km=CORRIDOR_BUFFER_KM,
                          dam_cut_offset_m=DAM_CUT_OFFSET_M, **cuts,
                          inputs={k: SD.geom_hash(SD.load(k)) for k in (
                              "dnipro_water_domain",)} | {
                              "KAKHOVKA_RESERVOIR_CORE":
                                  SD.geom_hash(SD.load_subzone("KAKHOVKA_RESERVOIR_CORE")),
                              ENVELOPE_LAYER: SD.geom_hash(
                                  gpd.read_file(ENVELOPE_SRC).set_index("layer")
                                  .geometry[ENVELOPE_LAYER])}),
        zones={k: dict(letter=ZONES[k]["letter"], label=ZONES[k]["label"],
                       area_km2=round(Z[k].area / 1e6, 1),
                       geometry_hash_utm=SD.geom_hash(Z[k]),
                       geometry_hash_4326=SD.registry_row(k)["geometry_hash"]
                       if k in SD._STATIC_LOADERS else None,
                       n_parts=len(_parts(Z[k])), legacy=ZONES[k]["legacy"])
               for k in Z},
        zones_total_km2=round(sum(g.area for g in Z.values()) / 1e6, 1),
        pairwise_overlap_km2={k: round(v, 6) for k, v in overlaps.items()},
        overlap_tolerance_km2=OVERLAP_TOL_KM2,
        event_layers={k: dict(area_km2=round(V[k].area / 1e6, 1),
                              geometry_hash_utm=SD.geom_hash(V[k]),
                              buffer_km=EVENTS[k]) for k in V},
        event_split_km2=ev_split,
        station_zone=station_zone,
        atl13_segments_in_R_fraction=round(atl_in_r, 4),
        legacy_containers=dict(
            {k: dict(area_km2=round(g.area / 1e6, 1), geometry_hash=SD.geom_hash(g))
             for k, g in legacy.items()},
            ZONE_4_outside_ZONE_1_km2=round(l4.difference(l1).area / 1e6, 1),
            ZONE_4_inside_ZONE_1_fraction=round(l4.intersection(l1).area / l4.area, 4),
            note="Download/processing containers with 10 km seam overlaps. Not the "
                 "paper's zones; retained unchanged for the S1 caches built on them."),
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    pd.DataFrame([dict(domain=k, **{kk: v for kk, v in m.items() if kk != "legacy"})
                  for k, m in manifest["zones"].items()]
                 + [dict(domain=k, **m) for k, m in manifest["event_layers"].items()]
                 ).to_csv(SUMMARY, index=False)

    print(f"zones -> {ZONES_FILE.relative_to(ROOT)}")
    for k, m in manifest["zones"].items():
        print(f"  {m['letter']}  {k:<30} {m['area_km2']:>9,.1f} km2  {m['geometry_hash_utm']}")
    print(f"  total {manifest['zones_total_km2']:,.1f} km2; max overlap "
          f"{max(overlaps.values()):.6f} km2")
    for k, d in ev_split.items():
        print(f"  {k}: {d}")
    print(f"  stations: {station_zone}")
    print(f"  ATL13 in R: {atl_in_r:.4f}")


if __name__ == "__main__":
    main()
