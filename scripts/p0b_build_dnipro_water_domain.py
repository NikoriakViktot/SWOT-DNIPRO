#!/usr/bin/env python
"""Build the Dnipro analysis water domain: reservoir head -> Dnipro-Bug liman.

WHY THIS EXISTS
---------------
Every hist* product so far was clipped to outputs/figure_data/
P20_reservoir_footprint.geojson, a derived artefact rather than a registry
domain. Measured against the registry's own authoritative polygon it is
wrong in both directions:

    * it TRUNCATES 87.4 km2 of real water at the eastern (upstream) end.
      Sentinel-1 over that strip at WSE 15.447 m gives VV median -21.6 dB
      with 99.0% of cells below -15 dB -- unambiguous open water.
    * it ADDS ~123 km2 of land (VV median -11.5 dB, only 15.8% water-like).

The totals nearly cancel (2191.8 vs 2174.7 km2), which is why the error
survived: an area check cannot see it, only a shape check can.

The registry polygon (data/Kakhovka_SA_2.geojson) is better but ALSO wrong at
the head. Khortytsia island splits the Dnipro into two arms there, and the
polygon draws one solid blob across both. Over the northern lobe it claims
44.9 km2 of water where Sentinel-1 measures 24.5 km2; the excess of 25.8 km2
is the island itself, independently confirmed by ESA WorldCover at 24.1 km2.
Its own provenance says "Promoted to authoritative; not yet hand-edited" --
it was validated by IoU and by volume (VolumeCorr 17.5 km3), and neither
integral metric can detect an island swallowed inside the polygon.

So this domain is derived from DATA rather than from any hand polygon.

METHOD
------
ESA WorldCover v200 (2021, 10 m) class 80 "permanent water bodies" is the
base. 2021 is pre-breach, so the reservoir is at normal pool. Class 90
(herbaceous wetland) is unioned in where it touches water, because the
shoreline margin must be inside the analysis domain.

The domain is then the single connected water body seeded in the reservoir,
which follows the Dnipro downstream through Kherson and the delta into the
Dnipro-Bug liman. Connectivity does the work a hand polygon cannot: islands
(Khortytsia and the delta islands) fall out as interior rings automatically,
and disconnected lakes and ponds are excluded.

Seaward the domain is cut at the study limit 31.5 E (the liman mouth at
Ochakiv), inherited from config/aoi/lower_dnipro_to_sea.geojson, so the open
Black Sea is not swallowed by the connectivity step.

CRS: everything is computed and stored in EPSG:32636 (UTM 36N), per the
repository CRS policy that all distance/area work is metric. The whole domain
from 31.5 E to 35.35 E lies inside zone 36N.

Outputs
-------
data/processed/domains/dnipro_water_domain_utm.geojson   (EPSG:32636)
outputs/tables/p0b_domain_comparison.csv
outputs/figures/phase19_20/png/p0b_dnipro_water_domain.png
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("VSI_CACHE", "TRUE")

import urllib.request

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling as WR
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import Point, box, shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/esa-worldcover"
CELL = 20.0
UTM = "EPSG:32636"

# study extent: reservoir head (Khortytsia, ~47.90 N) down to the liman mouth
WGS_BBOX = [31.50, 46.20, 35.35, 47.90]
WC_WATER, WC_WETLAND = 80, 90
MARGIN_BUFFER_M = 200.0      # shoreline may move outside the 2021 water line
MIN_ISLAND_KM2 = 0.05        # keep islands above this; smaller rings are speckle
MIN_PART_KM2 = 0.50

OUTDIR = ROOT / "data/processed/domains"
DOMAIN_GEOJSON = OUTDIR / "dnipro_water_domain_utm.geojson"
FIGDIR = CFG.FIG / "phase19_20" / "png"


def sas_token():
    return json.loads(urllib.request.urlopen(SAS, timeout=90).read())["token"]


def search_worldcover(bbox):
    q = {"collections": ["esa-worldcover"], "bbox": bbox, "limit": 100}
    r = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                               headers={"Content-Type": "application/json"})
    feats = json.loads(urllib.request.urlopen(r, timeout=180).read())["features"]
    # keep only the newest product version; mixing 2.0.0 with 1.0.0 lets the
    # older tiles overwrite the newer ones during mosaicking
    vers = sorted({f["properties"].get("esa_worldcover:product_version", "")
                   for f in feats})
    newest = vers[-1]
    return [f for f in feats
            if f["properties"].get("esa_worldcover:product_version") == newest]


def utm_grid(wgs_bbox):
    g = gpd.GeoSeries([box(*wgs_bbox)], crs=4326).to_crs(UTM).iloc[0]
    x0 = np.floor(g.bounds[0] / CELL) * CELL
    y0 = np.floor(g.bounds[1] / CELL) * CELL
    x1 = np.ceil(g.bounds[2] / CELL) * CELL
    y1 = np.ceil(g.bounds[3] / CELL) * CELL
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, nx=nx, ny=ny,
                tr=from_origin(x0, y1, CELL, CELL))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    G = utm_grid(WGS_BBOX)
    print(f"analysis grid {G['nx']} x {G['ny']} at {CELL:.0f} m in {UTM}")
    print(f"  E {G['x0']:.0f}..{G['x1']:.0f}   N {G['y0']:.0f}..{G['y1']:.0f}")
    print(f"  {(G['x1']-G['x0'])/1000:.0f} km x {(G['y1']-G['y0'])/1000:.0f} km")

    # ---------------------------------------------------- WorldCover mosaic
    tok = sas_token()
    feats = search_worldcover(WGS_BBOX)
    print(f"\nESA WorldCover v200 tiles: {len(feats)}")
    lc = np.zeros((G["ny"], G["nx"]), np.uint8)
    for f in feats:
        tmp = np.zeros((G["ny"], G["nx"]), np.uint8)
        try:
            with rasterio.open(f["assets"]["map"]["href"] + "?" + tok) as ds:
                reproject(source=rasterio.band(ds, 1), destination=tmp,
                          dst_transform=G["tr"], dst_crs=UTM,
                          dst_nodata=0, resampling=WR.nearest)
        except Exception as ex:
            print(f"  skip {f['id'][:44]}: {type(ex).__name__}")
            continue
        lc = np.where(tmp > 0, tmp, lc)
        print(f"  {f['id'][:48]}  cumulative cover "
              f"{100*float((lc>0).mean()):.1f}%")

    water = lc == WC_WATER
    wetland = lc == WC_WETLAND
    print(f"\n  permanent water   {water.sum()*CELL**2/1e6:9,.0f} km2")
    print(f"  herbaceous wetland{wetland.sum()*CELL**2/1e6:9,.0f} km2")

    # wetland only where it actually touches water: that is shoreline margin,
    # not an unrelated marsh somewhere else in the scene
    touch = ndimage.binary_dilation(water, np.ones((3, 3), bool), iterations=2)
    base = water | (wetland & touch)
    print(f"  water + adjacent wetland {base.sum()*CELL**2/1e6:,.0f} km2")

    # ---------------------------------------------------- connectivity
    # The component is chosen by overlap with the reservoir rather than by a
    # hand-picked seed point, so the result cannot depend on guessing a
    # coordinate that happens to fall on water.
    auth0 = gpd.GeoSeries([SD.load("reservoir_full_pool_prebreach")],
                          crs=4326).to_crs(UTM).iloc[0]
    res_mask = rio_rasterize([(auth0, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0,
                             dtype="uint8").astype(bool)
    lab, n = ndimage.label(base, structure=np.ones((3, 3), int))
    print(f"\n  {n:,} connected water components")

    def pick(mask, label_txt):
        ids = lab[mask & base]
        ids = ids[ids > 0]
        if ids.size == 0:
            raise SystemExit(f"no water component overlaps {label_txt}")
        vals, counts = np.unique(ids, return_counts=True)
        cid = int(vals[np.argmax(counts)])
        print(f"  {label_txt}: component {cid}, "
              f"{counts.max()*CELL**2/1e6:,.0f} km2 of overlap")
        return cid

    # In WorldCover 2021 the Kakhovka dam is still intact, so the reservoir
    # and the river below it are SEPARATE water bodies. That is physically
    # correct for a pre-breach domain, so both are selected explicitly rather
    # than bridged. Each is tied to a registry/AOI geometry, never to a
    # hand-picked component id.
    lower_aoi = gpd.read_file(ROOT / "config/aoi/lower_dnipro_to_sea.geojson"
                              ).to_crs(UTM).geometry.iloc[0]
    low_mask = rio_rasterize([(lower_aoi, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0, dtype="uint8").astype(bool)
    cid_res = pick(res_mask, "reservoir (Kakhovka_SA_2)")
    cid_low = pick(low_mask, "lower Dnipro + liman (lower_dnipro_to_sea AOI)")
    if cid_res == cid_low:
        print("  NOTE: reservoir and lower reach resolve to one component")
    body = (lab == cid_res) | (lab == cid_low)
    print(f"  reservoir part {float((lab==cid_res).sum())*CELL**2/1e6:,.0f} km2, "
          f"lower part {float((lab==cid_low).sum())*CELL**2/1e6:,.0f} km2")
    print(f"  connected Dnipro body: {body.sum()*CELL**2/1e6:,.0f} km2")
    del lab

    # allow the shoreline to move outside the 2021 water line
    dist = ndimage.distance_transform_edt(~body) * CELL
    domain = body | (dist <= MARGIN_BUFFER_M)
    # islands must survive the buffer: re-open anything that is a real island
    holes = ndimage.binary_fill_holes(domain) & ~domain
    hl, hn = ndimage.label(holes, structure=np.ones((3, 3), int))
    if hn:
        sizes = ndimage.sum(np.ones_like(hl), hl, range(1, hn + 1))
        keep = np.flatnonzero(sizes * CELL ** 2 / 1e6 >= MIN_ISLAND_KM2) + 1
        print(f"  interior islands: {hn:,} total, "
              f"{len(keep):,} above {MIN_ISLAND_KM2} km2 "
              f"({sizes[keep-1].sum()*CELL**2/1e6:,.1f} km2)")
    print(f"  domain with {MARGIN_BUFFER_M:.0f} m margin: "
          f"{domain.sum()*CELL**2/1e6:,.0f} km2")

    # ---------------------------------------------------- polygonise
    print("\npolygonising ...")
    polys = [shp_shape(g) for g, v in
             rio_shapes(domain.astype(np.uint8), mask=domain,
                        transform=G["tr"]) if v == 1]
    polys = [p for p in polys if p.area / 1e6 >= MIN_PART_KM2]
    dom = unary_union(polys)
    parts = list(dom.geoms) if hasattr(dom, "geoms") else [dom]
    parts = sorted(parts, key=lambda p: -p.area)
    inner = sum(len(p.interiors) for p in parts)
    print(f"  {len(parts)} part(s), {inner} interior ring(s), "
          f"{dom.area/1e6:,.1f} km2")
    for p in parts[:3]:
        b = p.bounds
        print(f"    part {p.area/1e6:9,.1f} km2  E {b[0]:.0f}..{b[2]:.0f}"
              f"  N {b[1]:.0f}..{b[3]:.0f}  holes={len(p.interiors)}")

    gdf = gpd.GeoDataFrame(
        dict(name=["dnipro_water_domain"],
             description=["Connected Dnipro water body, reservoir head "
                          "(Khortytsia) to Dnipro-Bug liman, from ESA "
                          "WorldCover v200 2021 class 80 (+adjacent 90), "
                          f"{MARGIN_BUFFER_M:.0f} m shoreline margin"],
             source=["ESA WorldCover v200 (2021), 10 m"],
             cell_m=[CELL], margin_m=[MARGIN_BUFFER_M]),
        geometry=[dom], crs=UTM)
    gdf.to_file(DOMAIN_GEOJSON, driver="GeoJSON")
    print(f"\n-> {DOMAIN_GEOJSON}  (stored in {UTM})")

    # ---------------------------------------------------- comparison
    auth = gpd.GeoSeries([SD.load("reservoir_full_pool_prebreach")],
                         crs=4326).to_crs(UTM).iloc[0]
    p20 = shp_shape(json.load(
        open(CFG.FIGDATA / "P20_reservoir_footprint.geojson"))["geometry"])
    res_part = max(parts, key=lambda p: p.intersection(auth).area)
    rows = []
    for nm, g in (("new_domain_reservoir_part", res_part),
                  ("authoritative_Kakhovka_SA_2", auth),
                  ("P20_reservoir_footprint", p20)):
        rows.append(dict(geometry_name=nm, area_km2=g.area / 1e6,
                         east_bound_m=g.bounds[2], north_bound_m=g.bounds[3],
                         n_interior_rings=(len(g.interiors)
                                           if g.geom_type == "Polygon"
                                           else sum(len(q.interiors)
                                                    for q in g.geoms))))
    C = pd.DataFrame(rows)
    C.to_csv(CFG.TABLES / "p0b_domain_comparison.csv", index=False)
    print("\n" + C.to_string(index=False))
    print(f"\nfull domain incl. lower Dnipro and liman: {dom.area/1e6:,.1f} km2")

    # ---------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.4))
    a = ax[0]
    gpd.GeoSeries([dom], crs=UTM).plot(ax=a, facecolor="#cfe0ec",
                                       edgecolor=BLUE, lw=0.5)
    gpd.GeoSeries([p20], crs=UTM).boundary.plot(ax=a, color=AMBER, lw=1.1,
                                                ls="--")
    a.set_title("a · new connected domain: reservoir head -> Dnipro-Bug liman\n"
                "amber dashed = old P20 footprint (reservoir only, truncated)",
                fontsize=10.4, loc="left")
    a.set_xlabel("easting (m, EPSG:32636)"); a.set_ylabel("northing (m)")
    a.grid(alpha=0.25)
    a = ax[1]
    kb = (645000, 5290000, 672000, 5308000)
    gpd.GeoSeries([dom], crs=UTM).plot(ax=a, facecolor="#cfe0ec",
                                       edgecolor=BLUE, lw=0.8)
    gpd.GeoSeries([auth], crs=UTM).boundary.plot(ax=a, color=RED, lw=1.6)
    a.set_xlim(kb[0], kb[2]); a.set_ylim(kb[1], kb[3])
    a.set_title("b · Khortytsia: the new domain keeps both arms and the island\n"
                "red = authoritative polygon (island swallowed)",
                fontsize=10.4, loc="left")
    a.set_xlabel("easting (m, EPSG:32636)"); a.set_ylabel("northing (m)")
    a.grid(alpha=0.25)
    fig.suptitle("p0b · Dnipro analysis water domain, derived from ESA "
                 "WorldCover rather than a hand polygon", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "p0b_dnipro_water_domain.png"
    fig.savefig(out, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
