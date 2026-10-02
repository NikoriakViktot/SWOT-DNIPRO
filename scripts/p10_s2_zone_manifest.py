#!/usr/bin/env python
"""P10 — a frozen, stratified Sentinel-2 manifest per zone, with MEASURED cloud.

THE OPTICAL ARCHIVE STOPS AT THE RESERVOIR. Everything on disk - 158 granules,
79 GB, plus 28 GB of index stacks - is on four tiles east of E 499,980, because
it was all acquired for the reservoir bathymetry. Measured against the zones:

    KAKHOVKA_RESERVOIR_CORE   100.0%
    ZONE_1                     81.3%
    ZONE_4                     53.8%   eastern half only, near the dam
    ZONE_2                      0.0%
    ZONE_3                      0.0%

The delta and the liman have no Sentinel-2 at all. Not sparse coverage - none.

TWO FROZEN TILE SETS close that, and naming them is the point: provenance and
the coverage audit both become statements about a named set rather than about
whichever tiles happened to be fetched.

    EASTERN_S2_TILE_SET   36TWS 36TWT 36TXT 36UXU    held, the reservoir
    WESTERN_S2_TILE_SET   36TUS 36TUT 36TVS 36TVT    missing, delta + liman

A tile outside both raises. A new tile means the domains moved, and that should
stop the run, not quietly enlarge the set.

THE SELECTION UNIT IS A DATE, NOT A GRANULE, for the same reason the S1 unit is
an event: one overpass split across four tiles is one observation, and counting
it four times is the pseudoreplication Gate 5 removed.

CLOUD IS MEASURED OVER THE ZONE, NOT READ OFF THE GRANULE. `eo:cloud_cover` is
a whole-tile number and a tile is 110 km wide, so it need not describe the zone.
Every candidate date therefore has its SCL band read at SCL_SCALE over the zone
geometry and the clear fraction computed there, and that is what the strata are
built on.

Having measured it: the granule number turns out to be a GOOD proxy here,
r = -0.88 / -0.90 / -0.89 on ZONE_2 / ZONE_3 / ZONE_4. That is the opposite of
what this paragraph asserted before the measurement existed, and it is recorded
rather than quietly dropped. Measuring is still the right call - the proxy is
strong on average and says nothing about an individual date, and the spread is
what a manifest selects on: ZONE_2 runs p10 = 0.24 to p90 = 1.00 around a
median of 0.80.

NO CLOUD CUTOFF IS ASSERTED. The distribution is printed and the design takes
the best PER_CELL dates in each stratum; MIN_CLEAR_REPORT below is a reporting
line, not a gate. Setting a threshold before seeing the distribution is what
emptied the ZONE_4 canonical envelope earlier in this campaign.

AND THE MEASUREMENT BUDGET IS SPENT PER STRATUM, AT RANDOM. The first run of
this file sorted candidates by eo:cloud_cover and measured the clearest head.
Every measured date then came back at clear ~ 1.00 - a property of the sort,
not of the sky - and the correlation it printed to argue that the granule
number is a poor proxy had been computed on a sample selected BY that number.
Measuring a random draw within each design cell instead gives, on ZONE_2,
p10 = 0.26 and a median of 0.82, and the granule number turns out to track the
measured one closely (r = -0.89). Worth stating plainly: the proxy is better
than I assumed. The strata are still built on the measured value, because the
zone sits in a corner of its tiles and there is no reason that correlation
holds everywhere.

STRATA, one design per zone, because the physics differs:

    ZONE_2   season x breach phase x tile coverage
             The delta has no gauge and no reservoir stage, so there is no
             hydrological-state axis to use; phase separates two different
             systems either side of 2023-06-06 and season carries the
             vegetation and discharge cycle. Inventing a state axis without a
             gauge would be inventing the state.

    ZONE_3   season x wind class x tile coverage
             Wind is what sets the estuary up, so it comes first here exactly
             as it does in the S1 design (p0e). ERA5 hourly via Open-Meteo,
             interpolated to the acquisition hour, never a daily mean.

    ZONE_4   flood phase (time since breach) x tile support (western/eastern)
             The floodway is ordered by the wave, and half of it sits on the
             western tiles it has never had, so which SET a date comes from is
             itself a stratum.

Nothing is downloaded. p10 writes the manifest and its sha256; the fetch reads
the freeze and verifies it first, as p0o does for S1.

Outputs
-------
outputs/tables/p10_{zone}_s2_manifest.csv
outputs/tables/p10_{zone}_s2_freeze.json
outputs/tables/p10_s2_tile_sets.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
BREACH = pd.Timestamp("2023-06-06")
WINDOW = ("2017-01-01", "2026-12-31")

EASTERN_S2_TILE_SET = ("36TWS", "36TWT", "36TXT", "36UXU")
WESTERN_S2_TILE_SET = ("36TUS", "36TUT", "36TVS", "36TVT")
KNOWN_TILES = set(EASTERN_S2_TILE_SET) | set(WESTERN_S2_TILE_SET)

SCL_SCALE = 200.0        # m, for the cloud measurement only
# SCL classes that are a usable surface observation. 4 vegetation, 5 not
# vegetated, 6 water, 7 unclassified, 11 snow. Everything else is nodata,
# saturated, dark area, cloud shadow, cloud or cirrus.
SCL_CLEAR = (4, 5, 6, 7, 11)
GRANULE_CLOUD_PREFILTER = 70     # generous; the real number is measured
MIN_SCOPE_COVERAGE = 0.60        # a date must actually see most of the zone
MIN_CLEAR_REPORT = 0.70          # reported, never applied as a gate
PER_CELL = 2

WIND_BINS = [(-0.01, 3.0, "CALM"), (3.0, 5.0, "LOW"),
             (5.0, 8.0, "MODERATE"), (8.0, 99.0, "HIGH")]
SINCE_BINS = [(-10**6, 0, "PRE"), (0, 7, "D0_7"), (7, 30, "D7_30"),
              (30, 90, "D30_90"), (90, 365, "D90_365"), (365, 10**6, "Y1_PLUS")]

ZONES = {
    "ZONE_2_KHERSON_DELTA": dict(axes=("season", "phase", "tile_set"),
                                 target=60),
    "ZONE_3_DNIPRO_BUG_ESTUARY": dict(axes=("season", "wind_class", "tile_set"),
                                      target=80),
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY": dict(axes=("since_breach", "tile_set"),
                                           target=60),
}


def http_json(url, payload=None, tries=6, timeout=180):
    last = None
    for a in range(tries):
        try:
            if payload is None:
                return json.loads(urllib.request.urlopen(url, timeout=timeout).read())
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as ex:
            last = ex
            time.sleep(min(60, 4 * 2 ** a))
    raise RuntimeError(f"{type(last).__name__}: {last}")


def tile_set_of(tile):
    if tile in EASTERN_S2_TILE_SET:
        return "EASTERN"
    if tile in WESTERN_S2_TILE_SET:
        return "WESTERN"
    raise SystemExit(
        f"tile {tile} is in neither frozen set. A new tile means the domains "
        f"moved; update EASTERN_S2_TILE_SET / WESTERN_S2_TILE_SET deliberately "
        f"rather than letting the set grow by accident.")


def season_of(ts):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON",
            11: "SON"}[ts.month]


def wind_class(u):
    for lo, hi, nm in WIND_BINS:
        if lo < u <= hi:
            return nm
    return "HIGH"


def catalogue(bbox):
    """Every L2A granule over the bbox, paged."""
    out, token = [], None
    q = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 500,
         "datetime": f"{WINDOW[0]}T00:00:00Z/{WINDOW[1]}T23:59:59Z",
         "query": {"eo:cloud_cover": {"lt": GRANULE_CLOUD_PREFILTER}}}
    for _ in range(40):
        if token:
            q["token"] = token
        r = http_json(STAC, q)
        fs = r.get("features", [])
        out += fs
        token = next((l["body"].get("token") for l in r.get("links", [])
                      if l.get("rel") == "next" and "body" in l), None)
        if not token or not fs:
            break
    return out


def by_date(feats, zone_geom):
    """Collapse granules into dates, with the footprint union per date."""
    g = {}
    for f in feats:
        p = f["properties"]
        d = p["datetime"][:10]
        g.setdefault(d, []).append(f)
    rows = []
    for d, items in sorted(g.items()):
        tiles = sorted({i["properties"].get("s2:mgrs_tile", "?") for i in items})
        sets = {tile_set_of(t) for t in tiles}
        fp = unary_union([shp_shape(i["geometry"]) for i in items])
        fp_m = gpd.GeoSeries([fp], crs=4326).to_crs(CFG.CRS_METRIC).iloc[0]
        cov = fp_m.intersection(zone_geom).area / zone_geom.area
        rows.append(dict(
            date=d, n_granules=len(items), tiles="|".join(tiles),
            tile_set="+".join(sorted(sets)),
            relative_orbit=items[0]["properties"].get("sat:relative_orbit"),
            granule_cloud_mean=float(np.mean(
                [i["properties"].get("eo:cloud_cover", np.nan) for i in items])),
            scope_coverage=float(cov), items=items))
    return pd.DataFrame(rows)


def assign_axes(D, spec, zone_geom):
    """Stratification axes, assigned before any cloud is measured."""
    D = D.copy()
    D["dt"] = pd.to_datetime(D.date)
    D["season"] = [season_of(t) for t in D.dt]
    D["phase"] = np.where(D.dt < BREACH, "PRE", "POST")
    D["since_breach_days"] = (D.dt - BREACH).dt.days
    D["since_breach"] = pd.cut(
        D.since_breach_days,
        bins=[b[0] for b in SINCE_BINS] + [SINCE_BINS[-1][1]],
        labels=[b[2] for b in SINCE_BINS], right=False)
    if "wind_class" in spec["axes"]:
        w = wind_for(sorted(D.date), zone_geom)
        D["wind_speed_ms"] = [w.get(d, np.nan) for d in D.date]
        D["wind_class"] = [wind_class(u) if np.isfinite(u) else "UNKNOWN"
                           for u in D.wind_speed_ms]
    return D


def clear_fraction(items, zone_geom, tok):
    """Clear fraction of the ZONE, from SCL, at SCL_SCALE.

    This is the number `eo:cloud_cover` cannot give: the tile is 110 km wide
    and the zone occupies a corner of it."""
    tot = clear = 0
    for it in items:
        href = it["assets"].get("SCL", {}).get("href")
        if not href:
            continue
        try:
            with rasterio.open(href + "?" + tok) as ds:
                b = transform_bounds(CFG.CRS_METRIC, ds.crs,
                                     *zone_geom.bounds, densify_pts=21)
                from rasterio.windows import from_bounds
                win = from_bounds(*b, ds.transform)
                k = max(1, int(round(SCL_SCALE / abs(ds.transform.a))))
                h = max(1, int(win.height // k))
                w = max(1, int(win.width // k))
                a = ds.read(1, window=win, out_shape=(h, w),
                            resampling=Resampling.nearest,
                            boundless=True, fill_value=0)
        except Exception:
            continue
        obs = a != 0
        tot += int(obs.sum())
        clear += int(np.isin(a, SCL_CLEAR).sum())
    return (clear / tot) if tot else np.nan


def wind_for(dates, zone_geom):
    """ERA5 hourly wind at the zone centroid, interpolated to ~10:30 local."""
    c = gpd.GeoSeries([zone_geom.centroid], crs=CFG.CRS_METRIC) \
           .to_crs(4326).iloc[0]
    q = urllib.parse.urlencode({
        "latitude": round(c.y, 3), "longitude": round(c.x, 3),
        "start_date": min(dates), "end_date": max(dates),
        "hourly": "wind_speed_10m", "windspeed_unit": "ms",
        "timezone": "UTC"})
    j = http_json(f"{ARCHIVE}?{q}")
    h = pd.Series(j["hourly"]["wind_speed_10m"],
                  index=pd.to_datetime(j["hourly"]["time"])).astype(float)
    # Sentinel-2 crosses at about 10:30 local; 08:00 UTC is the nearest hour
    # for this longitude. Taking a daily mean would average a calm morning
    # with a windy afternoon and call the scene MODERATE.
    return {d: float(h.get(pd.Timestamp(d) + pd.Timedelta(hours=8), np.nan))
            for d in dates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="", help="one zone, or all of them")
    ap.add_argument("--max-measure", type=int, default=400,
                    help="cap on dates whose SCL is actually read")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P10 — stratified Sentinel-2 manifests, cloud measured over the zone")
    print("=" * 78)
    print(f"  git {commit}")
    print(f"  EASTERN_S2_TILE_SET  {' '.join(EASTERN_S2_TILE_SET)}   (held)")
    print(f"  WESTERN_S2_TILE_SET  {' '.join(WESTERN_S2_TILE_SET)}   (missing)")
    (CFG.TABLES / "p10_s2_tile_sets.json").write_text(json.dumps(dict(
        eastern=list(EASTERN_S2_TILE_SET), western=list(WESTERN_S2_TILE_SET),
        scl_scale_m=SCL_SCALE, scl_clear_classes=list(SCL_CLEAR),
        git=commit), indent=2))

    tok = http_json(SAS)["token"]
    t_tok = time.time()
    todo = [args.zone] if args.zone else list(ZONES)

    for zone in todo:
        spec = ZONES[zone]
        print(f"\n{'=' * 78}\n{zone}\n{'=' * 78}")
        g = SD.load_utm(zone)
        bbox = [round(v, 5) for v in
                gpd.GeoSeries([g], crs=32636).to_crs(4326).iloc[0].bounds]
        feats = catalogue(bbox)
        D = by_date(feats, g)
        print(f"  {len(feats):,} granules -> {len(D):,} dates "
              f"(granule cloud < {GRANULE_CLOUD_PREFILTER}%)")
        print(f"  tile sets present: "
              f"{dict(D.tile_set.value_counts())}")

        D = D[D.scope_coverage >= MIN_SCOPE_COVERAGE].copy()
        print(f"  {len(D):,} dates cover >= {100*MIN_SCOPE_COVERAGE:.0f}% of "
              f"the zone")
        if D.empty:
            print("  nothing to stratify"); continue

        # THE AXES ARE ASSIGNED BEFORE THE MEASUREMENT, so the measurement
        # budget can be spent per stratum. The first version of this sorted by
        # granule cloud and measured the clearest head, which put the
        # eo:cloud_cover filter back in through the side door: every measured
        # date then came back at clear ~ 1.00, the distribution was an artefact
        # of the sort, and the correlation it printed to argue that granule
        # cloud is a poor proxy was computed on a sample chosen BY granule
        # cloud. Candidates are now drawn at random within each design cell.
        D = assign_axes(D, spec, g)
        axes = list(spec["axes"])
        cells = list(D.groupby(axes, observed=True))
        per = max(1, args.max_measure // max(len(cells), 1))
        rng = np.random.default_rng(CFG.SEED)
        pick = []
        for _, sub in cells:
            take = min(per, len(sub))
            pick += list(rng.choice(sub.index, size=take, replace=False))
        D = D.loc[sorted(set(pick))].copy()
        print(f"  design {' x '.join(axes)}: {len(cells)} cells; measuring "
              f"up to {per} random candidates per cell = {len(D)} dates "
              f"at {SCL_SCALE:.0f} m ...", flush=True)
        cf = []
        for i, r in enumerate(D.itertuples(), 1):
            if time.time() - t_tok > 1800:
                tok = http_json(SAS)["token"]; t_tok = time.time()
            cf.append(clear_fraction(r.items, g, tok))
            if i % 25 == 0:
                print(f"    {i}/{len(D)}", flush=True)
        D["clear_fraction_over_zone"] = cf
        D = D[np.isfinite(D.clear_fraction_over_zone)]
        print(f"  measured {len(D)} dates")
        if D.empty:
            print("  no date could be measured"); continue

        q = D.clear_fraction_over_zone.quantile([.1, .25, .5, .75, .9])
        print(f"  clear fraction over the zone: "
              f"p10 {q[.1]:.2f}  p25 {q[.25]:.2f}  median {q[.5]:.2f}  "
              f"p75 {q[.75]:.2f}  p90 {q[.9]:.2f}")
        print(f"  (>= {MIN_CLEAR_REPORT:.2f} on "
              f"{int((D.clear_fraction_over_zone >= MIN_CLEAR_REPORT).sum())} "
              f"of {len(D)} dates -- reported, not applied)")
        # Report the proxy quality, do not assert it. On ZONE_2 the granule
        # number turns out to track the measured clear fraction closely
        # (r = -0.89), which is the opposite of what I expected when writing
        # this; on a zone that sits in a tile corner it need not. Either way
        # the strata are built on the measured number, so the proxy never has
        # to be trusted - but claiming it is poor when it is not would be as
        # wrong as the reverse.
        d0 = D.granule_cloud_mean.corr(D.clear_fraction_over_zone)
        q = ("tracks it closely" if d0 <= -0.8 else
             "tracks it loosely" if d0 <= -0.5 else "is a poor proxy")
        print(f"  granule eo:cloud_cover vs measured clear fraction: "
              f"r = {d0:+.2f} -- the granule number {q} here. The strata use "
              f"the measured number regardless.")

        keep = []
        for _, sub in D.groupby(axes, observed=True):
            keep += list(sub.sort_values("clear_fraction_over_zone",
                                         ascending=False)
                            .head(PER_CELL).index)
        extra = spec["target"] - len(keep)
        if extra > 0:
            pool = D.drop(index=keep).sort_values(
                "clear_fraction_over_zone", ascending=False)
            keep += list(pool.head(extra).index)
            print(f"  design fills {len(cells)*PER_CELL} cell-slots; "
                  f"+{min(extra, len(pool))} clearest dates added to reach "
                  f"the target")
        S = D.loc[sorted(set(keep))].drop(columns=["items", "dt"])
        S["analysis_zone"] = zone
        S["select_reason"] = "_x_".join(axes)
        S["selection_unit"] = "date"

        print(f"\n  selected {len(S)} dates")
        for ax in axes:
            print(f"    {ax:16s} {dict(S[ax].value_counts())}")
        print(f"    clear fraction median {S.clear_fraction_over_zone.median():.2f}"
              f", min {S.clear_fraction_over_zone.min():.2f}")
        print(f"    ~{0.6*S.n_granules.sum():,.0f} GB at ~0.6 GB per granule "
              f"({S.n_granules.sum()} granules)")

        csv = CFG.TABLES / f"p10_{zone.lower()}_s2_manifest.csv"
        S.sort_values("date").to_csv(csv, index=False)
        h = hashlib.sha256(csv.read_bytes()).hexdigest()
        (CFG.TABLES / f"p10_{zone.lower()}_s2_freeze.json").write_text(
            json.dumps(dict(
                analysis_zone=zone, manifest=str(csv.relative_to(ROOT)),
                sha256=h, selection_unit="date",
                n_dates_selected=int(len(S)), n_dates_measured=int(len(D)),
                design=axes, per_cell=PER_CELL, target=spec["target"],
                cloud="measured over the zone from SCL, not eo:cloud_cover",
                scl_scale_m=SCL_SCALE, scl_clear_classes=list(SCL_CLEAR),
                min_scope_coverage=MIN_SCOPE_COVERAGE,
                min_clear_reported=MIN_CLEAR_REPORT,
                tile_sets=dict(eastern=list(EASTERN_S2_TILE_SET),
                               western=list(WESTERN_S2_TILE_SET)),
                bbox_4326=bbox, git=commit), indent=2))
        print(f"  -> {csv}")
        print(f"  -> freeze sha256 {h[:16]}")

    print("\n  Nothing downloaded. The fetch verifies these hashes first.")


if __name__ == "__main__":
    main()
