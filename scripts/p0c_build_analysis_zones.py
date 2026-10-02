#!/usr/bin/env python
"""Define the three Dnipro analysis zones (download/analysis AOIs).

THESE ARE NOT WATER MASKS. They are stable spatial containers for downloading
Sentinel-1 and for tagging every derived record with where it came from. The
water mask is built later, independently, per observation event and per zone.
ESA WorldCover is used here only as bootstrap evidence for where the water
system runs -- never as a final shoreline.

WHY THREE ZONES RATHER THAN ONE
-------------------------------
A single domain from Zaporizhzhia to Ochakiv would merge three hydraulically
different systems, and a SAR classifier calibrated on one of them does not
transfer to the others:

  ZONE_1  impounded reservoir -> drawdown -> river channel. Calm water, a
          clean dark-SAR signature, and the only zone where a reservoir stage
          H1/H2/H3 means anything.
  ZONE_2  multi-branch delta: distributaries, islands, floodplain, reed beds
          and very narrow channels. "dark SAR = water" degrades here because
          flooded vegetation double-bounces bright and channels fall below
          the resolution cell.
  ZONE_3  estuary: wind waves, wind setup, seiche, salinity and Black Sea
          forcing roughen the surface, so open water is often NOT dark.

H1/H2/H3 reservoir-stage products apply ONLY to KAKHOVKA_RESERVOIR_CORE, not
to all of ZONE_1, which is why Zone 1 carries explicit subzones.

BOUNDARIES
----------
Zones are cut at named landmarks along the system and given a deliberate
overlap, so no Sentinel-1 coverage is lost at an artificial seam and the
exact scientific boundary can be set later without re-downloading:

    ZONE_1 | ZONE_2   at Kherson
    ZONE_2 | ZONE_3   where the delta opens into the liman

Each zone is the water corridor buffered by CORRIDOR_BUFFER_KM, clipped to
its easting band. Following the water rather than using a lat/lon rectangle
keeps the AOIs tight enough to download while guaranteeing the channel,
floodplain and islands are inside.

Outputs
-------
data/processed/domains/analysis_zones_utm.geojson      (EPSG:32636)
data/processed/domains/analysis_subzones_utm.geojson   (EPSG:32636)
outputs/tables/p0c_analysis_zones.csv
outputs/figures/phase19_20/png/p0c_analysis_zones.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
UTM = "EPSG:32636"
OUTDIR = ROOT / "data/processed/domains"
ZONES_GEOJSON = OUTDIR / "analysis_zones_utm.geojson"
SUBZONES_GEOJSON = OUTDIR / "analysis_subzones_utm.geojson"
FIGDIR = CFG.FIG / "phase19_20" / "png"

CORRIDOR_BUFFER_KM = 10.0     # download margin around the water corridor
ZONE_OVERLAP_KM = 5.0         # each side of a cut, so zones overlap by 10 km
MAIN_CHANNEL_CORRIDOR_KM = 8.0  # half-width around the dam -> Kherson axis

# Named landmarks, in EPSG:4326, used to place the cuts. Recorded here so the
# boundaries are traceable to places rather than to bare coordinates.
LANDMARKS = {
    "DniproHES_Zaporizhzhia": (35.087, 47.868),
    "Khortytsia": (35.060, 47.840),
    "Kakhovka_dam_Nova_Kakhovka": (33.370, 46.778),
    "Kherson": (32.617, 46.635),
    "Hola_Prystan_delta": (32.520, 46.520),
    "liman_mid": (31.900, 46.620),
    "Ochakiv_liman_mouth": (31.540, 46.612),
    "Mykolaiv_Southern_Bug": (31.995, 46.975),
    "Inhul_mouth": (31.990, 46.960),
}
# where the delta opens into the liman; between Hola Prystan and liman_mid
CUT_2_3_EASTING = 443000.0


def utm_point(lon, lat):
    return gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(UTM).iloc[0]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    water = SD.load_utm("dnipro_water_domain")
    L = {k: utm_point(*v) for k, v in LANDMARKS.items()}
    print("landmarks (EPSG:32636):")
    for k, p in L.items():
        print(f"  {k:<28} E {p.x:7.0f}  N {p.y:8.0f}")

    cut12 = L["Kherson"].x
    cut23 = CUT_2_3_EASTING
    ov = ZONE_OVERLAP_KM * 1000.0
    print(f"\ncuts: ZONE_1|ZONE_2 at Kherson E {cut12:.0f}; "
          f"ZONE_2|ZONE_3 at E {cut23:.0f}")
    print(f"overlap {2*ZONE_OVERLAP_KM:.0f} km at each cut, "
          f"corridor buffer {CORRIDOR_BUFFER_KM:.0f} km")

    corridor = water.buffer(CORRIDOR_BUFFER_KM * 1000.0)
    b = corridor.bounds
    BIG = 1e7

    def band(emin, emax):
        return corridor.intersection(box(emin, -BIG, emax, BIG))

    zones = {
        "ZONE_1_KAKHOVKA_LOWER_DNIPRO": band(cut12 - ov, b[2] + 1000),
        "ZONE_2_KHERSON_DELTA": band(cut23 - ov, cut12 + ov),
        "ZONE_3_DNIPRO_BUG_ESTUARY": band(b[0] - 1000, cut23 + ov),
    }
    descr = {
        "ZONE_1_KAKHOVKA_LOWER_DNIPRO":
            "Kakhovka reservoir / former reservoir to DniproHES, plus the "
            "Dnipro below the former dam and the river corridor to Kherson. "
            "Impounded -> drawdown -> river.",
        "ZONE_2_KHERSON_DELTA":
            "Kherson, the downstream distributaries, islands, floodplain and "
            "reed wetlands, and the transition toward the estuary. "
            "Multi-branch river; dark-SAR=water is unreliable here.",
        "ZONE_3_DNIPRO_BUG_ESTUARY":
            "Dnipro-Bug estuary, Mykolaiv, the lower Southern Bug and the "
            "Inhul mouth, with the outlet toward the Black Sea. "
            "Wind setup, waves and coastal forcing roughen the surface.",
    }

    # ------------------------------------------------ ZONE_1 subzones
    dam_e = L["Kakhovka_dam_Nova_Kakhovka"].x
    core_nominal = SD.load_utm("reservoir_full_pool_prebreach")
    z1 = zones["ZONE_1_KAKHOVKA_LOWER_DNIPRO"]
    res_water = water.intersection(box(dam_e - 500, -BIG, BIG, BIG))
    riv_water = water.intersection(box(-BIG, -BIG, dam_e - 500, BIG)).intersection(z1)
    # The reach below the dam is NOT one thing: the Inhulets joins the Dnipro
    # from the north and carries its own regime, so it is separated from the
    # main channel. The main channel is the water inside a corridor along the
    # dam -> Kherson axis; whatever the corridor does not contain is
    # tributary. Using an along-axis corridor rather than a bare northing cut
    # keeps the split tied to the river's actual course.
    axis = LineString([(L["Kakhovka_dam_Nova_Kakhovka"].x,
                        L["Kakhovka_dam_Nova_Kakhovka"].y),
                       (L["Kherson"].x, L["Kherson"].y)])
    main_corridor = axis.buffer(MAIN_CHANNEL_CORRIDOR_KM * 1000.0)
    dam_to_kherson = riv_water.intersection(main_corridor)
    tributaries = riv_water.difference(main_corridor)

    # Split against the SA_2 envelope itself, NOT a buffered copy: the
    # corridor buffer is a download margin and would swallow the whole
    # reservoir, leaving the transition subzone empty.
    subzones = {
        "KAKHOVKA_RESERVOIR_CORE": res_water.intersection(core_nominal),
        "FORMER_RESERVOIR_TRANSITION": res_water.difference(core_nominal),
        "KAKHOVKA_DAM_TO_KHERSON": dam_to_kherson,
        "INHULETS_TRIBUTARY": tributaries,
    }
    sub_descr = {
        "KAKHOVKA_RESERVOIR_CORE":
            "The reservoir proper, up to DniproHES. Extent follows "
            "reservoir_full_pool_prebreach (SA_2) but takes its geometry from "
            "dnipro_water_domain, so Khortytsia stays an island rather than "
            "being swallowed. H1/H2/H3 stage products apply HERE ONLY.",
        "FORMER_RESERVOIR_TRANSITION":
            "Water inside ZONE_1 east of the dam but outside the nominal "
            "SA_2 envelope: upper reach and arms exposed by the corrected "
            "domain. Not valid for reservoir-stage products.",
        "KAKHOVKA_DAM_TO_KHERSON":
            "The Dnipro main channel from the Kakhovka dam down to Kherson: "
            "the reach that carried the breach flood wave. River channel, "
            "never a reservoir stage.",
        "INHULETS_TRIBUTARY":
            "Water below the dam outside the dam->Kherson corridor -- chiefly "
            "the Inhulets, which joins from the north with its own regime and "
            "must not be averaged into the main-channel reach.",
    }

    # ------------------------------------------------ checks
    print("\ncoverage checks (landmark -> zone):")
    rows = []
    for zn, g in zones.items():
        inside = [k for k, p in L.items() if g.contains(p)]
        print(f"  {zn:<30} {g.area/1e6:8,.0f} km2  contains: "
              f"{', '.join(inside) if inside else '(none)'}")
        bb = g.bounds
        rows.append(dict(zone=zn, area_km2=g.area / 1e6,
                         east_min_m=bb[0], east_max_m=bb[2],
                         north_min_m=bb[1], north_max_m=bb[3],
                         water_area_km2=water.intersection(g).area / 1e6,
                         landmarks="|".join(inside),
                         description=descr[zn]))
    Z = pd.DataFrame(rows)
    Z.to_csv(CFG.TABLES / "p0c_analysis_zones.csv", index=False)

    must = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO":
            ["DniproHES_Zaporizhzhia", "Kakhovka_dam_Nova_Kakhovka", "Kherson"],
            "ZONE_2_KHERSON_DELTA": ["Kherson", "Hola_Prystan_delta"],
            "ZONE_3_DNIPRO_BUG_ESTUARY":
            ["liman_mid", "Ochakiv_liman_mouth", "Mykolaiv_Southern_Bug",
             "Inhul_mouth"]}
    ok = True
    for zn, names in must.items():
        miss = [n for n in names if not zones[zn].contains(L[n])]
        if miss:
            ok = False
            print(f"  *** {zn} is MISSING {miss}")
    print(f"  required-landmark check: {'PASS' if ok else 'FAIL'}")

    ov12 = zones["ZONE_1_KAKHOVKA_LOWER_DNIPRO"].intersection(
        zones["ZONE_2_KHERSON_DELTA"])
    ov23 = zones["ZONE_2_KHERSON_DELTA"].intersection(
        zones["ZONE_3_DNIPRO_BUG_ESTUARY"])
    print(f"\noverlaps (deliberate, so no coverage is lost at a seam):")
    print(f"  ZONE_1 n ZONE_2 {ov12.area/1e6:7,.0f} km2")
    print(f"  ZONE_2 n ZONE_3 {ov23.area/1e6:7,.0f} km2")
    union = unary_union(list(zones.values()))
    print(f"\nDNIPRO_SYSTEM_MASTER (union, catalogue/visualisation only): "
          f"{union.area/1e6:,.0f} km2")
    print(f"  water inside the three zones: "
          f"{water.intersection(union).area/1e6:,.0f} km2 of "
          f"{water.area/1e6:,.0f} km2 total "
          f"({100*water.intersection(union).area/water.area:.2f}%)")

    print("\nZONE_1 subzones:")
    for sn, g in subzones.items():
        print(f"  {sn:<30} {g.area/1e6:8,.1f} km2")

    # ------------------------------------------------ write
    gpd.GeoDataFrame(
        dict(analysis_zone=list(zones), description=[descr[z] for z in zones],
             corridor_buffer_km=CORRIDOR_BUFFER_KM,
             zone_overlap_km=2 * ZONE_OVERLAP_KM,
             is_water_mask=False),
        geometry=list(zones.values()), crs=UTM).to_file(
            ZONES_GEOJSON, driver="GeoJSON")
    gpd.GeoDataFrame(
        dict(analysis_zone="ZONE_1_KAKHOVKA_LOWER_DNIPRO",
             analysis_subzone=list(subzones),
             description=[sub_descr[s] for s in subzones],
             is_water_mask=False),
        geometry=list(subzones.values()), crs=UTM).to_file(
            SUBZONES_GEOJSON, driver="GeoJSON")
    print(f"\n-> {ZONES_GEOJSON}")
    print(f"-> {SUBZONES_GEOJSON}")
    print(f"-> {CFG.TABLES/'p0c_analysis_zones.csv'}")

    # ------------------------------------------------ figure
    fig, ax = plt.subplots(2, 1, figsize=(15, 11))
    a = ax[0]
    cols = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": BLUE,
            "ZONE_2_KHERSON_DELTA": GREEN,
            "ZONE_3_DNIPRO_BUG_ESTUARY": PURPLE}
    for zn, g in zones.items():
        gpd.GeoSeries([g], crs=UTM).plot(ax=a, facecolor=cols[zn], alpha=0.26,
                                         edgecolor=cols[zn], lw=1.4,
                                         label=zn.replace("_", " "))
    gpd.GeoSeries([water], crs=UTM).plot(ax=a, facecolor="#1b4f66",
                                         edgecolor="none")
    for k, p in L.items():
        a.plot(p.x, p.y, "o", ms=4.5, color=RED)
        a.annotate(k.replace("_", " "), (p.x, p.y), xytext=(5, 4),
                   textcoords="offset points", fontsize=6.8, color=INK)
    a.legend(fontsize=8.2, loc="upper left")
    a.set_title("a · three analysis zones (download/analysis AOIs, NOT water "
                "masks); dark = water corridor", fontsize=10.6, loc="left")
    a.set_xlabel("easting (m, EPSG:32636)"); a.set_ylabel("northing (m)")
    a.grid(alpha=0.25)
    a = ax[1]
    sc = {"KAKHOVKA_RESERVOIR_CORE": BLUE,
          "FORMER_RESERVOIR_TRANSITION": AMBER,
          "KAKHOVKA_DAM_TO_KHERSON": GREEN,
          "INHULETS_TRIBUTARY": PURPLE}
    for sn, g in subzones.items():
        if g.is_empty:
            continue
        gpd.GeoSeries([g], crs=UTM).plot(ax=a, facecolor=sc[sn], alpha=0.75,
                                         edgecolor="none",
                                         label=sn.replace("_", " "))
    a.axvline(dam_e, color=RED, ls="--", lw=1.3)
    a.annotate("Kakhovka dam", (dam_e, 5185000), xytext=(6, 0),
               textcoords="offset points", fontsize=8, color=RED)
    a.legend(fontsize=8.2, loc="upper left")
    a.set_title("b · ZONE_1 subzones — H1/H2/H3 stage products apply ONLY to "
                "KAKHOVKA_RESERVOIR_CORE", fontsize=10.6, loc="left")
    a.set_xlabel("easting (m, EPSG:32636)"); a.set_ylabel("northing (m)")
    a.grid(alpha=0.25)
    fig.suptitle("p0c · Dnipro analysis zones: Kakhovka/Lower Dnipro · "
                 "Kherson/Delta · Dnipro-Bug Estuary", y=1.0, fontsize=12.5)
    fig.tight_layout()
    out = FIGDIR / "p0c_analysis_zones.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
