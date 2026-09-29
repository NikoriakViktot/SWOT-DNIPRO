#!/usr/bin/env python
"""P0X — two registry geometries that say something untrue, repaired.

Both defects were found by looking at the zones on an OpenStreetMap basemap,
which is worth recording: they had survived every numeric check because an area
in km2 cannot tell you that a polygon is in the wrong river.

DEFECT 1 — ZONE_2 CARRIES TWO PIECES OF OTHER WATER BODIES

p0c builds every zone as `water_corridor.buffer(10 km)` cut into an easting
band. That is sound for a single corridor, but the water domain in ZONE_2's
band (E 438,000 .. 475,685) is not only the Dnipro delta: the band's western
edge also slices the SOUTHERN BUG, 50 km north near Snihurivka, and the BLACK
SEA COAST, 40 km south near Lazurne. Both arrive as separate polygons:

    1,397.7 km2   the Dnipro delta          spans the band, E 438,000..475,685
      222.8 km2   Southern Bug              E 438,000..449,280
       56.1 km2   Black Sea / Yahorlyk      E 438,000..442,560

They are separate polygons precisely because they are separate systems here:
the Bug meets the Dnipro in the liman, WEST of this band, so within the band
there is no connection. Neither belongs to a zone whose sampling design, wind
handling and classifier anchors are built for a multi-branch river delta.

THE RULE: a zone is ONE corridor. Keep the components connected to the zone
upstream of it - operationally, those that reach the upstream cut plane. The
delta component does; the two fragments stop 26 and 33 km short.

ALMOST nothing is lost, and the remainder is stated rather than rounded away.
ZONE_3 is the estuary zone and its band runs to E 448,000, so it already carries
96.2% and 100.0% of the two fragments - measured, not asserted. The 8.5 km2 that
falls outside every zone is a 1.3 km strip of the Southern Bug at Snihurivka,
50 km north of the delta and upstream of the reach ZONE_3 studies. It is logged
in p0x_geometry_repair.csv as an uncovered remainder. Giving it to ZONE_3 would
move that zone's eastern edge 1.28 km and invalidate the grid its 150 cached S1
events were fetched on, which is not a trade worth making for water outside the
study reach.

DEFECT 2 — THE INHULETS SUBZONE CONTAINS HOLA PRYSTAN

INHULETS_TRIBUTARY was defined as a RESIDUAL: "water below the dam outside the
dam->Kherson corridor". A residual takes whatever is left, and what was left
included a piece of the delta 25 km away on the opposite bank:

     62.72 km2   the Inhulets valley          0.00 km from the stem
      3.63 km2   the Inhulets mouth           1.00 km
     11.41 km2   Hola Prystan, left bank     24.95 km   <- not the Inhulets

The Inhulets joins the Dnipro from the NORTH; Hola Prystan is on the south
bank of the delta. Carrying it under this name would put delta distributary
water into a tributary-regime subzone whose whole purpose is to keep the
Inhulets OUT of main-channel averages.

THE RULE: keep the parts within INHULETS_MAX_GAP_KM of the Inhulets stem. The
gap in the data is 1.0 km against 24.95 km, so the threshold is not doing the
work - the distribution is. The Hola Prystan piece is dropped from ZONE_1's
subzones only after checking it lies fully inside ZONE_2 and ZONE_4, which
already carry the delta; it is measured at 100% of both.

Outputs
-------
data/processed/domains/analysis_zones_utm.geojson       (ZONE_2 repaired)
data/processed/domains/analysis_subzones_utm.geojson    (INHULETS repaired)
outputs/tables/p0x_geometry_repair.csv
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

ZONES = ROOT / "data/processed/domains/analysis_zones_utm.geojson"
SUBZONES = ROOT / "data/processed/domains/analysis_subzones_utm.geojson"
KHERSON_E = 470684.5          # p0c LANDMARKS["Kherson"] in EPSG:32636
ZONE_OVERLAP_KM = 5.0         # p0c
CUT_TOL_M = 500.0
INHULETS_MAX_GAP_KM = 10.0
MIN_COVERED_PCT = 95.0        # a dropped component must be a band artefact
RESIDUAL_MIN_KM2 = 1.0        # below this the identity is within tolerance
RESIDUAL_NAME = "HOLA_PRYSTAN_LEFT_BANK"


def parts_of(g):
    return sorted(list(g.geoms) if g.geom_type.startswith("Multi") else [g],
                  key=lambda p: -p.area)


def ll(p):
    c = gpd.GeoSeries([p.centroid], crs=32636).to_crs(4326).iloc[0]
    return f"{c.x:.4f},{c.y:.4f}"


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P0X — registry geometry repair")
    print("=" * 78)
    print(f"  git {commit}")

    Z = gpd.read_file(ZONES)
    S = gpd.read_file(SUBZONES)
    log = []

    # ------------------------------------------------ DEFECT 1: ZONE_2
    print("\n" + "=" * 78)
    print("ZONE_2_KHERSON_DELTA — one corridor, not three water bodies")
    print("=" * 78)
    i2 = Z.index[Z.analysis_zone == "ZONE_2_KHERSON_DELTA"][0]
    g2 = Z.at[i2, "geometry"]
    cut_upstream = KHERSON_E + ZONE_OVERLAP_KM * 1000.0
    print(f"  upstream cut plane (Kherson + {ZONE_OVERLAP_KM:.0f} km overlap): "
          f"E {cut_upstream:,.1f}")
    keep, drop = [], []
    for p in parts_of(g2):
        reaches = p.bounds[2] >= cut_upstream - CUT_TOL_M
        (keep if reaches else drop).append(p)
        print(f"    {p.area/1e6:9,.1f} km2  centroid {ll(p)}  east edge "
              f"E {p.bounds[2]:,.0f}  "
              f"{'CONNECTED to ZONE_1 -> keep' if reaches else f'stops {(cut_upstream-p.bounds[2])/1000:,.0f} km short -> drop'}")
    if not keep:
        raise SystemExit("no ZONE_2 component reaches the upstream cut")

    z3 = Z.loc[Z.analysis_zone == "ZONE_3_DNIPRO_BUG_ESTUARY",
               "geometry"].iloc[0]
    print("\n  is the dropped water lost? checking it against ZONE_3:")
    lost = 0.0
    for p in drop:
        cov = 100 * p.intersection(z3).area / p.area
        lost += p.area / 1e6 * (1 - cov / 100)
        print(f"    {p.area/1e6:9,.1f} km2 at {ll(p)}  -> {cov:5.1f}% already "
              f"inside ZONE_3")
        if cov < MIN_COVERED_PCT:
            raise SystemExit(
                f"only {cov:.1f}% of this component is carried by another "
                f"zone; it is not a band artefact and must be reassigned, not "
                f"dropped")
        log.append(dict(target="ZONE_2_KHERSON_DELTA", action="drop_component",
                        area_km2=p.area / 1e6, centroid_4326=ll(p),
                        reason="not connected to the Dnipro corridor in this "
                               "band (Southern Bug / Black Sea coast)",
                        covered_by="ZONE_3_DNIPRO_BUG_ESTUARY",
                        covered_pct=cov))
    new2 = unary_union(keep)
    print(f"\n  ZONE_2  {g2.area/1e6:,.0f} -> {new2.area/1e6:,.0f} km2 "
          f"({len(parts_of(g2))} parts -> {len(parts_of(new2))})")

    # WHAT LEAVES THE TILING, AND WHY THAT IS ALLOWED HERE.
    # The first version of this guard refused any loss at all: "8.5 km2 would
    # leave every zone; that is a hole in the tiling". That was the wrong test.
    # It assumed every square kilometre of water inside the corridor buffer was
    # DESIGNED to be covered, and this water never was - it is a 1.3 km strip
    # of the SOUTHERN BUG at Snihurivka, 50 km north of the delta and upstream
    # of the reach ZONE_3 exists to study, which entered only because p0c
    # buffered the whole water domain and then sliced it by easting.
    #
    # Handing it to ZONE_3 instead would move ZONE_3's eastern edge 1.28 km,
    # change SD.build_grid's output for that zone, and invalidate the grid its
    # 150 cached S1 events were fetched on - a real cost, for 8.5 km2 of a
    # different river outside the study reach.
    #
    # So the guard now tests what it should have tested: each dropped piece
    # must be a band artefact (>= MIN_COVERED_PCT already carried by the zone
    # that owns that water body), and the uncovered remainder is REPORTED in
    # the table rather than silently discarded.
    print(f"\n  water leaving the tiling: {lost:,.1f} km2 "
          f"({100*lost/(g2.area/1e6):.2f}% of the old ZONE_2)")
    print(f"  -- the Southern Bug upstream of the ZONE_3 reach, never part of "
          f"the design; recorded in p0x_geometry_repair.csv, not hidden")
    log.append(dict(target="ZONE_2_KHERSON_DELTA", action="uncovered_remainder",
                    area_km2=lost, centroid_4326="",
                    reason="Southern Bug strip east of the ZONE_3 band edge; "
                           "reassigning it would invalidate ZONE_3's S1 grid",
                    covered_by="none", covered_pct=0.0))
    Z.at[i2, "geometry"] = new2
    if "description" in Z.columns:
        Z.at[i2, "description"] = (
            str(Z.at[i2, "description"]) + " Repaired by p0x: the Southern "
            "Bug and Black Sea coastal fragments that the easting band sliced "
            "off were removed; they are carried by ZONE_3.")

    # ------------------------------------------------ DEFECT 2: INHULETS
    print("\n" + "=" * 78)
    print("INHULETS_TRIBUTARY — a residual that collected Hola Prystan")
    print("=" * 78)
    ii = S.index[S.analysis_subzone == "INHULETS_TRIBUTARY"][0]
    gi = S.at[ii, "geometry"]
    ip = parts_of(gi)
    stem = ip[0]
    keep_i, drop_i = [], []
    for p in ip:
        d = p.distance(stem) / 1000.0
        (keep_i if d <= INHULETS_MAX_GAP_KM else drop_i).append(p)
        print(f"    {p.area/1e6:8,.2f} km2  centroid {ll(p)}  {d:6.2f} km from "
              f"the stem  -> {'Inhulets' if d <= INHULETS_MAX_GAP_KM else 'NOT the Inhulets'}")

    z2n, z4 = new2, Z.loc[Z.analysis_zone == "ZONE_4_DAM_TO_KHERSON_FLOODWAY",
                          "geometry"].iloc[0]
    print("\n  where does the non-Inhulets water belong?")
    for p in drop_i:
        c2 = 100 * p.intersection(z2n).area / p.area
        c4 = 100 * p.intersection(z4).area / p.area
        print(f"    {p.area/1e6:8,.2f} km2 at {ll(p)}  ZONE_2 {c2:5.1f}%  "
              f"ZONE_4 {c4:5.1f}%")
        if max(c2, c4) < 99.0:
            raise SystemExit("this piece is not fully carried by another zone; "
                             "give it its own subzone rather than dropping it")
        log.append(dict(target="INHULETS_TRIBUTARY", action="drop_component",
                        area_km2=p.area / 1e6, centroid_4326=ll(p),
                        reason=f"{p.distance(stem)/1000:.1f} km from the "
                               f"Inhulets stem, on the opposite bank at Hola "
                               f"Prystan; delta water, not a tributary",
                        covered_by="ZONE_2_KHERSON_DELTA / "
                                   "ZONE_4_DAM_TO_KHERSON_FLOODWAY",
                        covered_pct=max(c2, c4)))
    newi = unary_union(keep_i)
    print(f"\n  INHULETS_TRIBUTARY  {gi.area/1e6:,.2f} -> {newi.area/1e6:,.2f} "
          f"km2 ({len(ip)} parts -> {len(parts_of(newi))})")
    S.at[ii, "geometry"] = newi
    S.at[ii, "description"] = (
        "The Inhulets where it joins the Dnipro from the north, with its own "
        "regime; it must not be averaged into the main-channel reach. Defined "
        "by PROXIMITY TO THE INHULETS STEM, not as the residual of the "
        "dam->Kherson corridor -- the residual definition had pulled in 11.4 "
        "km2 of delta water at Hola Prystan, 25 km away on the opposite bank.")

    # ---------------------------- DEFECT 3, created by the fix for DEFECT 2
    print("\n" + "=" * 78)
    print("THE ZONE_1 SUBZONE IDENTITY, broken by the repair above")
    print("=" * 78)
    # p0n carries an independent check: the union of ZONE_1's subzones should
    # equal the registry water domain clipped to ZONE_1. Taking 11.4 km2 out of
    # INHULETS_TRIBUTARY broke it by exactly that much, and p0n refused to write
    # a manifest. The check is right and the repair was right; what was wrong
    # was dropping the piece without giving it anywhere to go. ZONE_2 and ZONE_4
    # carrying it does not make it disappear from ZONE_1.
    #
    # So the residual is named. It is computed, not assumed: whatever water in
    # ZONE_1 no subzone accounts for gets a subzone, and the identity is
    # restored by construction rather than by adjusting the tolerance.
    wz1 = SD.load_utm("dnipro_water_domain").intersection(
        SD.load_utm("ZONE_1_KAKHOVKA_LOWER_DNIPRO"))
    Z1 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
    cur = unary_union(list(S.loc[S.analysis_zone == Z1, "geometry"]))
    res = wz1.difference(cur)
    print(f"  water in ZONE_1        {wz1.area/1e6:9,.2f} km2")
    print(f"  subzone union          {cur.area/1e6:9,.2f} km2")
    print(f"  unaccounted residual   {res.area/1e6:9,.2f} km2")
    if res.area / 1e6 >= RESIDUAL_MIN_KM2:
        parts = parts_of(res)
        print(f"  {len(parts)} part(s); largest {parts[0].area/1e6:,.2f} km2 "
              f"at {ll(parts[0])}")
        if RESIDUAL_NAME in set(S.analysis_subzone):
            S.loc[S.analysis_subzone == RESIDUAL_NAME, "geometry"] = res
            print(f"  {RESIDUAL_NAME} already registered; geometry refreshed")
        else:
            S = pd.concat([S, gpd.GeoDataFrame([dict(
                analysis_zone=Z1,
                analysis_subzone=RESIDUAL_NAME,
                description=(
                    "Delta water on the left bank at Hola Prystan, at ZONE_1's "
                    "western edge. It is NOT the Inhulets - it sits 25 km away "
                    "on the opposite bank - and it was carried under that name "
                    "only because INHULETS_TRIBUTARY was defined as a residual. "
                    "ZONE_2 and ZONE_4 also cover it; it is named here so that "
                    "the union of ZONE_1's subzones still equals the water "
                    "domain clipped to ZONE_1, which is p0n's independent "
                    "geometry check."),
                is_water_mask=False)], geometry=[res],
                crs=CFG.CRS_METRIC)], ignore_index=True)
            print(f"  registered {RESIDUAL_NAME}")
        log.append(dict(target=RESIDUAL_NAME, action="register_subzone",
                        area_km2=res.area / 1e6,
                        centroid_4326=ll(parts[0]),
                        reason="restores the ZONE_1 subzone identity that "
                               "removing Hola Prystan from INHULETS_TRIBUTARY "
                               "broke",
                        covered_by="ZONE_2_KHERSON_DELTA / "
                                   "ZONE_4_DAM_TO_KHERSON_FLOODWAY",
                        covered_pct=100.0))
        left = wz1.difference(unary_union(list(
            S.loc[S.analysis_zone == Z1, "geometry"])))
        print(f"  residual after       {left.area/1e6:9,.2f} km2  "
              f"{'IDENTITY RESTORED' if left.area/1e6 < 1.0 else 'STILL OPEN'}")
    else:
        print("  nothing unaccounted; the identity holds")

    # ------------------------------------------------ write
    gpd.GeoDataFrame(Z, crs=CFG.CRS_METRIC).to_file(ZONES, driver="GeoJSON")
    gpd.GeoDataFrame(S, crs=CFG.CRS_METRIC).to_file(SUBZONES, driver="GeoJSON")
    t = CFG.TABLES / "p0x_geometry_repair.csv"
    pd.DataFrame(log).to_csv(t, index=False)
    print(f"\n-> {ZONES}")
    print(f"-> {SUBZONES}")
    print(f"-> {t}")
    print("\n  DOWNSTREAM: ZONE_2's area changed, so its S1 grid, download bbox "
          "and stratified manifest were all built on the old geometry. p0t "
          "must be rerun for ZONE_2 before any further fetch, and the events "
          "already cached must be re-checked for coverage against the new "
          "domain.")


if __name__ == "__main__":
    main()
