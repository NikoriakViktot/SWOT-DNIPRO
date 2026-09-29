#!/usr/bin/env python
"""P0R — the observed June-2023 flood envelope for ZONE_2, as a real domain.

`below_dam_floodplain` is UNRESOLVED in the registry, which is why ZONE_2 could
not be designed: without knowing what a flooded event should cover there is no
coverage gate and no stratification, and the orbit-14 lesson is that "the fetch
succeeded" says nothing about whether the zone was observed.

The registry note asks for a HAND-derived or administrative floodplain. This
builds something better for the purpose: the extent the water ACTUALLY reached,
from Sentinel-1 over the breach itself.

    2023-06-01, 06-02   pre-breach baseline
    2023-06-06          the breach
    06-09 ... 06-30     the wave and its recession

OBSERVED, NOT MODELLED. A HAND surface would give a plausible floodplain from
terrain; this gives the one the flood used. The difference matters for a delta,
where the water goes where the channels and levees let it, not where the
elevation alone suggests.

THE MASK RULE IS M3, NOT OTSU, AND THE FIRST VERSION OF THIS FILE GOT IT WRONG.
I reached for Otsu on VV because there is no same-date optical anchor here, and
Gate 7 had already measured what that costs:

    M0_otsu_baseline        precision 0.997  recall 0.745  IoU 0.744
    M3_anchored_VV_VH_LDA   precision 0.996  recall 0.991  IoU 0.984

Otsu under-detects by a quarter, and worse for a multi-date envelope, its
threshold is fitted per scene, so it drifts with incidence angle, soil moisture
and wind. The run proved it: the Otsu threshold and the "water" area correlated
at +0.87, a PRE-breach date (06-02) returned 518 km2 against 323 km2 on the day
of the breach, and within a single orbit the series ran in opposite directions -
orbit 87 fell 518 -> 365 -> 206 while orbit 138 rose 323 -> 544 -> 784, peaking
three weeks AFTER the breach. That is a measurement of the threshold, not of the
flood.

The anchors M3 needs do not have to come from optical. They come from the
registry: the eroded core of the water domain is water on every date, and land
far outside any plausible flood is land on every date. Both are stable by
construction, which is exactly what a multi-date envelope requires.

WHAT IS WRITTEN. Three geometries, kept separate because they answer different
questions:

    water_prebreach     the channel before the wave
    flood_envelope      union of all post-breach observed water = the domain
    flood_increment     envelope minus pre-breach = what the breach added

and the envelope buffered by BUFFER_KM, which is what a download and
stratification domain needs: the margin where a mask boundary can move between
events without leaving the domain.

Outputs
-------
data/processed/domains/zone_2_kherson_delta_flood_envelope_utm.geojson
outputs/tables/p0r_zone2_flood_events.csv
outputs/figures/phase19_20/png/p0r_zone2_flood_envelope.png
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
import hist25b_gate6_event_qualification as G6

# --domain selects the target. Both were flooded and both need an envelope;
# the dam-to-Kherson reach is a narrow 170 km2 channel polygon, so the SEARCH
# area there is the reach buffered out and clipped to its parent zone -- an
# envelope confined to the channel it started from could only ever return that
# channel.
TARGETS = {
    "ZONE_2_KHERSON_DELTA": ("zone", None),
    "KAKHOVKA_DAM_TO_KHERSON": ("subzone", "ZONE_1_KAKHOVKA_LOWER_DNIPRO"),
}
ZONE = "ZONE_2_KHERSON_DELTA"
CELL = 20.0
BREACH = "2023-06-06"
WINDOW = ("2023-06-01", "2023-06-30")
BUFFER_KM = 1.0
BBOX_MARGIN_KM = 5.0       # acquisition margin on the download bbox
SUBZONE_SEARCH_KM = 8.0    # how far out of a narrow reach to look
UNION_MIN_PART_KM2 = 0.10  # sieve the union before vectorising
MIN_PART_KM2 = 0.05
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
OUT = ROOT / "data/processed/domains/zone_2_kherson_delta_flood_envelope_utm.geojson"
FIGDIR = CFG.FIG / "phase19_20" / "png"


def water_mask(vv, vh, cov, anchor_w, anchor_l):
    """Gate 7's M3: anchored VV/VH linear discriminant, then its cleaning.

    The anchors are fixed across all dates, so the decision boundary is fitted
    to the same two classes every time. That is the property Otsu lacked: its
    per-scene threshold moved with the scene, and the envelope then measured
    the threshold."""
    d_vv = 10 * np.log10(np.maximum(vv, 1e-6))
    d_vh = 10 * np.log10(np.maximum(vh, 1e-6))
    aw = anchor_w & cov & np.isfinite(d_vv) & np.isfinite(d_vh)
    al = anchor_l & cov & np.isfinite(d_vv) & np.isfinite(d_vh)
    if aw.sum() < 2000 or al.sum() < 2000:
        return None, np.nan, np.nan, np.nan
    Xw = np.c_[d_vv[aw], d_vh[aw]]
    Xl = np.c_[d_vv[al], d_vh[al]]
    mw, ml = Xw.mean(0), Xl.mean(0)
    S = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
    w = np.linalg.solve(S, ml - mw)
    cut = 0.5 * (w @ mw + w @ ml)
    disc = np.full(vv.shape, np.nan, np.float32)
    ok = cov & np.isfinite(d_vv) & np.isfinite(d_vh)
    disc[ok] = (np.c_[d_vv[ok], d_vh[ok]] @ w) - cut
    if np.nanmedian(disc[aw]) > np.nanmedian(disc[al]):
        disc = -disc                      # orient negative-in-water
    m = (disc < 0) & ok
    px = CELL ** 2 / 1e6
    lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
    if n:
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) * px
        keep = np.zeros(n + 1, bool)
        keep[1:] = sz >= MIN_PART_KM2
        m = keep[lab]
    sep = float(np.nanmedian(disc[al]) - np.nanmedian(disc[aw]))
    return m, sep, float(m.sum() * px), float(aw.sum())


def sieve(mask, min_km2=UNION_MIN_PART_KM2):
    """Drop specks before vectorising.

    A 20 m union mask over a delta carries tens of thousands of one- and
    two-pixel parts, and rasterio.features.shapes then spends longer building
    polygons for speckle than the entire fetch took -- two runs hung here. These
    parts are far below any scale the envelope is used at."""
    px = CELL ** 2 / 1e6
    lab, k = ndimage.label(mask, structure=np.ones((3, 3), int))
    if not k:
        return mask
    sz = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1)) * px
    keep = np.zeros(k + 1, bool)
    keep[1:] = sz >= min_km2
    out = keep[lab]
    holes = ndimage.binary_fill_holes(out) & ~out
    hl, hn = ndimage.label(holes, structure=np.ones((3, 3), int))
    if hn:
        hs = ndimage.sum(np.ones_like(hl), hl, range(1, hn + 1)) * px
        fill = np.zeros(hn + 1, bool)
        fill[1:] = hs < min_km2
        out |= fill[hl]
    print(f"      sieve {k:,} parts -> {int(keep.sum()):,} kept", flush=True)
    return out


ENVELOPE_CELL = 100.0     # the envelope is a DOMAIN, not a shoreline


def polygonise(mask, G, cell=ENVELOPE_CELL):
    """Vectorise at ENVELOPE_CELL, not at the analysis cell.

    Two OOM kills came from insisting on 20 m here. The output is an
    acquisition and stratification domain, buffered by a kilometre before use,
    so a 20 m boundary was precision nobody consumes -- and it cost millions of
    vertices. Block-reducing to 100 m first cuts the vertex count ~25x.

    rio_shapes also uses 4-connectivity by default while ndimage.label above
    uses 8, which is why 335 sieved components arrived here as 6,327 polygons;
    connectivity=8 makes the two agree."""
    from rasterio.features import shapes as rio_shapes
    from rasterio.transform import from_origin
    k = max(1, int(round(cell / CELL)))
    ny, nx = mask.shape
    ny2, nx2 = ny // k, nx // k
    red = mask[:ny2 * k, :nx2 * k].reshape(ny2, k, nx2, k).any(axis=(1, 3))
    tr = from_origin(G["x0"], G["y1"], CELL * k, CELL * k)
    G = dict(G, tr=tr)
    mask = red
    # Simplify each part BEFORE the union. A raster boundary traced at 20 m
    # carries a vertex every cell, so a 300 km2 region arrives with millions of
    # them and unary_union was killed by the OOM reaper twice. Simplifying at
    # half a cell removes the staircase and nothing else -- the geometry is a
    # raster mask, so sub-cell detail was never real.
    geoms = [shp_shape(g).simplify(cell / 2) for g, v in rio_shapes(
        mask.astype(np.uint8), mask=mask, transform=G["tr"],
        connectivity=8) if v == 1]
    geoms = [g for g in geoms if not g.is_empty]
    if not geoms:
        return None
    print(f"      union of {len(geoms):,} simplified parts", flush=True)
    return unary_union(geoms).buffer(0)


def main() -> None:
    global ZONE, OUT
    FIGDIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P0R — ZONE_2 observed flood envelope, June 2023")
    print("=" * 78)
    print(f"  git {commit}; breach {BREACH}; window {WINDOW[0]}..{WINDOW[1]}")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=ZONE, choices=list(TARGETS))
    ZONE = ap.parse_args().domain
    kind, parent = TARGETS[ZONE]
    OUT = ROOT / f"data/processed/domains/{ZONE.lower()}_flood_envelope_utm.geojson"
    print(f"  target {ZONE} ({kind})")
    if kind == "zone":
        z = SD.load_utm(ZONE)
    else:
        core = SD.load_subzone_utm(ZONE)
        z = core.buffer(SUBZONE_SEARCH_KM * 1000.0).intersection(
            SD.load_utm(parent))
        print(f"  reach {core.area/1e6:,.0f} km2 -> search area "
              f"{z.area/1e6:,.0f} km2 (+{SUBZONE_SEARCH_KM:.0f} km, "
              f"clipped to {parent})")
    gr = SD.build_grid(z, CELL, what=f"{ZONE} flood grid")
    from rasterio.transform import from_origin
    from rasterio.features import rasterize as rio_rasterize
    tr = from_origin(float(gr["gx"][0]), float(gr["gy"][-1]), CELL, CELL)
    inside = rio_rasterize([(z, 1)], out_shape=(gr["ny"], gr["nx"]),
                           transform=tr, fill=0, dtype="uint8").astype(bool)
    bbox = [round(v, 5) for v in
            gpd.GeoSeries([z], crs=32636).to_crs(4326).iloc[0].bounds]
    G = dict(x0=float(gr["gx"][0]), y0=float(gr["gy"][0]),
             x1=float(gr["gx"][-1]), y1=float(gr["gy"][-1]),
             nx=gr["nx"], ny=gr["ny"], tr=tr, inside=inside)
    print(f"  domain {z.area/1e6:,.0f} km2, grid {G['nx']}x{G['ny']} at {CELL:.0f} m")

    # FIXED anchors from the registry, identical on every date. Water: the
    # eroded core of the water domain, which is channel on any date. Land:
    # outside a generous buffer of it, which no plausible flood reached.
    wdom = SD.load_utm("dnipro_water_domain").intersection(z)
    aw_geom = wdom.buffer(-200.0)
    al_geom = z.difference(wdom.buffer(6000.0))
    anchor_w = rio_rasterize([(aw_geom, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=tr, fill=0, dtype="uint8").astype(bool)
    anchor_l = rio_rasterize([(al_geom, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=tr, fill=0, dtype="uint8").astype(bool)
    print(f"  anchors: water {aw_geom.area/1e6:,.0f} km2 "
          f"({anchor_w.sum():,} px), land {al_geom.area/1e6:,.0f} km2 "
          f"({anchor_l.sum():,} px)")
    if anchor_w.sum() < 2000 or anchor_l.sum() < 2000:
        raise SystemExit("anchors too small for a stable discriminant")

    q = {"collections": ["sentinel-1-rtc"], "bbox": bbox, "limit": 200,
         "datetime": f"{WINDOW[0]}T00:00:00Z/{WINDOW[1]}T23:59:59Z"}
    req = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    feats = json.loads(urllib.request.urlopen(req, timeout=180).read())["features"]
    ev = {}
    for f in feats:
        p = f["properties"]
        k = (p["datetime"][:10], p.get("sat:relative_orbit"),
             "ASC" if str(p.get("sat:orbit_state", "")).startswith("asc") else "DES")
        ev.setdefault(k, []).append(f)
    print(f"  {len(feats)} granules -> {len(ev)} events")

    cache = CFG.S1_CACHE / f"{ZONE}_flood_june2023"
    cache.mkdir(parents=True, exist_ok=True)
    G6.CACHE = cache
    tok = json.loads(urllib.request.urlopen(G6.SAS, timeout=90).read())["token"]
    t_tok = time.time()

    # accumulate in RASTER space and vectorise once: per-event polygonisation
    # of a delta mask produces tens of thousands of parts and the union then
    # dominates the runtime
    rows = []
    acc_pre = np.zeros(inside.shape, bool)
    acc_post = np.zeros(inside.shape, bool)
    n_pre = n_post = 0
    for (date, orb, st), items in sorted(ev.items()):
        eid = f"{date}_orb{orb}_{st}"
        if time.time() - t_tok > 1800:
            tok = json.loads(urllib.request.urlopen(
                G6.SAS, timeout=90).read())["token"]
            t_tok = time.time()
        print(f"\n  {eid}: {len(items)} granule(s)", flush=True)
        try:
            vv, vh, cov, seam = G6.build_event(eid, items, tok, G)
        except Exception as ex:
            print(f"    BUILD FAILED {type(ex).__name__}")
            continue
        obs = float(cov[inside].mean())
        if obs < 0.01:
            (cache / f"{eid}.npz").unlink(missing_ok=True)
            print(f"    EMPTY ({100*obs:.1f}%) -- deleted")
            continue
        m, sep, km2, naw = water_mask(vv, vh, cov & inside, anchor_w, anchor_l)
        if m is None:
            print(f"    anchors not observed on this date -- skipped")
            continue
        phase = "PRE" if date < BREACH else "POST"
        if phase == "PRE":
            acc_pre |= m; n_pre += 1
        else:
            acc_post |= m; n_post += 1
        rows.append(dict(event_id=eid, date=date, relative_orbit=orb,
                         orbit_state=st, phase=phase,
                         observed_fraction=obs, lda_separation=sep,
                         n_water_anchor_px=int(naw), water_km2=km2,
                         water_pct_of_zone=100 * km2 / (z.area / 1e6)))
        print(f"    obs {100*obs:.0f}%  LDA sep {sep:.2f}  "
              f"water {km2:,.0f} km2 ({100*km2/(z.area/1e6):.0f}% of zone)  "
              f"[{phase}]", flush=True)
        del vv, vh, cov, m

    if not rows:
        raise SystemExit("no usable event")
    R = pd.DataFrame(rows)
    R.to_csv(CFG.TABLES / "p0r_zone2_flood_events.csv", index=False)

    if not acc_post.any():
        raise SystemExit("no post-breach event produced a mask")
    print(f"\n  vectorising the unions ({n_pre} PRE, {n_post} POST events)...",
          flush=True)
    pre_u = polygonise(sieve(acc_pre), G) if acc_pre.any() else None
    env = polygonise(sieve(acc_post), G)
    inc = polygonise(sieve(acc_post & ~acc_pre), G) if acc_pre.any() else None
    buf = env.buffer(BUFFER_KM * 1000.0).intersection(z)

    print("\n" + "=" * 78)
    print("ENVELOPE")
    print("=" * 78)
    print(f"  pre-breach water      {pre_u.area/1e6 if pre_u else 0:8,.0f} km2  "
          f"({n_pre} event(s))")
    print(f"  flood envelope        {env.area/1e6:8,.0f} km2  "
          f"({n_post} post-breach event(s))")
    if inc is not None:
        print(f"  increment over pre    {inc.area/1e6:8,.0f} km2  "
              f"= what the breach added")
    print(f"  envelope + {BUFFER_KM:.0f} km buffer {buf.area/1e6:8,.0f} km2  "
          f"({100*buf.area/z.area:.0f}% of the zone)")
    w = SD.load_utm("dnipro_water_domain").intersection(z)
    print(f"  for comparison, the registry water domain in this zone is "
          f"{w.area/1e6:,.0f} km2")
    print(f"  -> the envelope is {env.area/w.area:.1f}x the normal water extent")

    # A DOWNLOAD BBOX WITH MARGIN. The envelope is the analysis target; an
    # acquisition domain wants a rectangle with room, so a granule that shifts
    # between orbits still lands inside it.
    from shapely.geometry import box as shp_box
    b0, b1, b2, b3 = buf.bounds
    mg = BBOX_MARGIN_KM * 1000.0
    dl = shp_box(b0 - mg, b1 - mg, b2 + mg, b3 + mg)
    ll = gpd.GeoSeries([dl], crs=32636).to_crs(4326).iloc[0].bounds
    print(f"  download bbox (+{BBOX_MARGIN_KM:.0f} km) {dl.area/1e6:,.0f} km2")
    print(f"    EPSG:32636  {b0-mg:,.0f} {b1-mg:,.0f} {b2+mg:,.0f} {b3+mg:,.0f}")
    print(f"    EPSG:4326   {[round(v, 5) for v in ll]}")

    layers = [("water_prebreach", pre_u), ("flood_envelope", env),
              ("flood_increment", inc), ("flood_envelope_buffered", buf),
              ("download_bbox", dl)]
    gdf = gpd.GeoDataFrame(
        dict(layer=[n for n, g in layers if g is not None],
             area_km2=[g.area / 1e6 for n, g in layers if g is not None],
             buffer_km=[BUFFER_KM if n.endswith("buffered") else 0.0
                        for n, g in layers if g is not None],
             provenance=["observed Sentinel-1, June 2023"] *
                        len([1 for n, g in layers if g is not None])),
        geometry=[g for n, g in layers if g is not None],
        crs=CFG.CRS_METRIC)
    gdf.to_file(OUT, driver="GeoJSON")
    print(f"\n-> {OUT}")
    print(f"-> {CFG.TABLES / 'p0r_zone2_flood_events.csv'}")
    figure(z, w, pre_u, env, buf, R)


def figure(z, w, pre_u, env, buf, R):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.6))
    gpd.GeoSeries([z]).boundary.plot(ax=ax[0], color="#8a94a3", lw=.8,
                                     label="ZONE_2")
    gpd.GeoSeries([buf]).plot(ax=ax[0], color="#b07d27", alpha=.25)
    gpd.GeoSeries([env]).plot(ax=ax[0], color="#236f8c", alpha=.65)
    if pre_u is not None:
        gpd.GeoSeries([pre_u]).plot(ax=ax[0], color="#1a2228", alpha=.9)
    ax[0].set_title("observed flood envelope (blue) over pre-breach water "
                    "(dark)\nwith the 1 km buffer (amber)", color="#1a2228",
                    fontsize=11)
    ax[0].set_xlabel("easting (m)"); ax[0].set_ylabel("northing (m)")
    ax[0].set_aspect("equal")

    R2 = R.sort_values("date")
    c = np.where(R2.phase == "PRE", "#3f7d4e", "#c1402a")
    ax[1].bar(range(len(R2)), R2.water_km2, color=c)
    ax[1].axhline(w.area / 1e6, color="#1a2228", ls="--", lw=1.2,
                  label=f"registry water domain {w.area/1e6:,.0f} km²")
    ax[1].set_xticks(range(len(R2)))
    ax[1].set_xticklabels(R2.date, rotation=60, ha="right", fontsize=8)
    ax[1].set_ylabel("classified water (km²)")
    ax[1].set_title("per event: green = pre-breach, red = post",
                    color="#1a2228", fontsize=11)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("p0r · ZONE_2: the extent the water actually reached",
                 color="#1a2228", fontsize=13)
    fig.tight_layout()
    p = FIGDIR / "p0r_zone2_flood_envelope.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
