#!/usr/bin/env python
"""ZONE 3 — ERA5 wind forcing, and a wind-stratified Sentinel-1 selection.

NOTHING IS DOWNLOADED HERE. This produces the catalogue and the proposed
first-pass manifest only; the Sentinel-1 fetch waits for review.

WHY WIND COMES FIRST IN THE ESTUARY
-----------------------------------
In ZONE_1 open water is reliably dark in C-band because the impounded surface
is smooth. In the Dnipro-Bug estuary it often is not: wind waves roughen the
surface and raise backscatter until water and land overlap. Wind is therefore
not an auxiliary attribute here, it is one of the primary variables setting
the SAR response, and selecting scenes before knowing it would produce an
archive that silently confounds wind with geometry.

So the stratification differs by zone:

    ZONE_1   stage x regime x orbit
    ZONE_2   season x hydrological state x orbit
    ZONE_3   wind x season x orbit x estuarine state

DIRECTION MATTERS, NOT ONLY SPEED. The liman is a long, shallow, roughly
zonal basin, so an along-axis wind drives set-up/set-down and fetch-limited
waves quite differently from a cross-axis wind of the same strength. The
estuary axis is derived from the geometry of the water in ZONE_3 rather than
assumed, and the wind vector is resolved into along- and cross-axis
components against it.

A SINGLE POINT WOULD NOT DO. ERA5 is ~0.25 deg (~28 km) and ZONE_3 spans
several cells, so the field is sampled on a grid over the water and
summarised per event (mean, median, p90) instead of being reduced to one
station.

TIME, NOT DATE. Wind changes within a day, so every event is matched to its
actual SAR acquisition timestamp by interpolating the hourly series, never by
taking a daily mean.

SOURCE. ERA5 hourly via the Open-Meteo archive API (no credentials needed).
The Planetary Computer's era5-pds collection stops in 2020 and so cannot
cover the breach period.

WIND CLASSES ARE PROVISIONAL. CALM/LOW/MODERATE/HIGH below are operational
bin edges for balancing the sample, NOT physically established thresholds.
They must be revisited against measured event-level VV/VH behaviour over
stable open water once the first subset exists.

Outputs
-------
outputs/tables/p0e_zone3_era5_hourly.csv
outputs/tables/p0e_zone3_s1_wind_joined.csv
outputs/tables/p0e_zone3_coverage_matrix.csv
outputs/tables/p0e_zone3_firstpass_manifest.csv
outputs/figures/phase19_20/png/p0e_zone3_wind.png
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
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
from shapely.geometry import Point

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ZONE = "ZONE_3_DNIPRO_BUG_ESTUARY"
START, END = "2019-01-01", "2023-12-31"
ERA5_SPACING_DEG = 0.25          # match the ERA5 grid; finer adds no information
FIGDIR = CFG.FIG / "phase19_20" / "png"
HOURLY_CSV = CFG.TABLES / "p0e_zone3_era5_hourly.csv"

# provisional operational bins, to be validated against VV/VH behaviour
WIND_BINS = [(-0.01, 3.0, "CALM"), (3.0, 5.0, "LOW"),
             (5.0, 8.0, "MODERATE"), (8.0, 99.0, "HIGH")]
TARGET_PER_CELL = 2              # events per wind x season x orbit cell
GB_PER_EVENT = 0.145             # measured from the ZONE_1 event cache
# The liman basin, bounded by landmarks (Ochakiv mouth -> the delta cut) so
# the axis PCA sees the estuary and neither the Southern Bug arm nor the
# bbox-clipped Black Sea block.
LIMAN_BOX = (383000.0, 5145000.0, 448000.0, 5180000.0)


def _stac(bbox, d0, d1):
    out, token = [], None
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    while True:
        q = {"collections": ["sentinel-1-rtc"], "bbox": bbox, "limit": 500,
             "datetime": f"{d0}T00:00:00Z/{d1}T23:59:59Z"}
        if token:
            q["token"] = token
        r = urllib.request.Request(url, data=json.dumps(q).encode(),
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


def wind_class(u):
    for lo, hi, nm in WIND_BINS:
        if lo < u <= hi:
            return nm
    return "HIGH"


def season_of(ts):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON",
            11: "SON"}[ts.month]


def fetch_point(lat, lon, start, end):
    q = urllib.parse.urlencode({
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": start, "end_date": end,
        "hourly": "wind_speed_10m,wind_direction_10m,pressure_msl",
        "wind_speed_unit": "ms", "timezone": "UTC"})
    for attempt in range(5):
        try:
            j = json.loads(urllib.request.urlopen(f"{ARCHIVE}?{q}", timeout=180).read())
            h = j["hourly"]
            return pd.DataFrame(dict(
                time=pd.to_datetime(h["time"]),
                wind_speed=h["wind_speed_10m"],
                wind_dir=h["wind_direction_10m"],
                mslp=h["pressure_msl"], lat=lat, lon=lon))
        except Exception as ex:
            if attempt == 4:
                raise
            time.sleep(4 * (attempt + 1))


def estuary_axis(water_utm):
    """Principal axis of the LIMAN, derived from geometry.

    Restricted to LIMAN_BOX. Two things would otherwise corrupt the axis:
    ZONE_3 water contains the Southern Bug arm, ~143 km north to Mykolaiv and
    longer than the liman's ~59 km east-west extent, which alone drags a
    whole-zone PCA to 174 deg; and the bbox-clipped Black Sea block in the
    south-west drags it to 127 deg. Resolving wind on either would invert the
    physics, since for a long shallow zonal basin it is the along-basin
    easterly/westerly component that drives set-up and fetch-limited waves.

    Within LIMAN_BOX the PCA returns 92.3 deg, which agrees with the
    independent landmark estimate (Ochakiv mouth -> delta cut = 90.0 deg) to
    2.3 deg -- a check that the derived axis is the real one.
    """
    from shapely.geometry import box as _box
    liman = water_utm.intersection(_box(*LIMAN_BOX))
    parts = list(liman.geoms) if hasattr(liman, "geoms") else [liman]
    parts = [p for p in parts if p.area > 1e6]
    pts = np.vstack([np.asarray(p.exterior.coords) for p in parts])
    pts = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    ax = vt[0] / np.linalg.norm(vt[0])
    if ax[0] < 0:
        ax = -ax                      # point it eastward for interpretability
    return ax, liman


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    z_ll = SD.load(ZONE)
    z_utm = SD.load_utm(ZONE)
    water = SD.load_utm("dnipro_water_domain").intersection(z_utm)
    print(f"{ZONE}: {z_utm.area/1e6:,.0f} km2, water {water.area/1e6:,.0f} km2")

    ax, liman = estuary_axis(water)
    bearing = (np.degrees(np.arctan2(ax[0], ax[1]))) % 360
    print(f"liman basin (landmark-bounded): "
          f"{liman.area/1e6:,.0f} km2 of {water.area/1e6:,.0f} km2 ZONE_3 water")
    print(f"estuary principal axis: ({ax[0]:+.3f}, {ax[1]:+.3f}) "
          f"= {bearing:.1f} deg from north (derived from the liman basin; a "
          f"PCA over all ZONE_3 water gives 174 deg - the Southern Bug - and "
          f"south-of-5180000 gives 127 deg - the clipped sea block)")
    print(f"  cross-check: Ochakiv mouth -> delta cut implies 90.0 deg; "
          f"agreement {abs(bearing-90.0):.1f} deg")

    # ------------------------------------------------ ERA5 sample grid
    b = z_ll.bounds
    lons = np.arange(np.floor(b[0] / ERA5_SPACING_DEG) * ERA5_SPACING_DEG,
                     b[2] + ERA5_SPACING_DEG, ERA5_SPACING_DEG)
    lats = np.arange(np.floor(b[1] / ERA5_SPACING_DEG) * ERA5_SPACING_DEG,
                     b[3] + ERA5_SPACING_DEG, ERA5_SPACING_DEG)
    wl = gpd.GeoSeries([water], crs=32636).to_crs(4326).iloc[0]
    pts = [(la, lo) for la in lats for lo in lons
           if wl.buffer(0.05).contains(Point(lo, la))]
    print(f"\nERA5 sample points over ZONE_3 water: {len(pts)} "
          f"at {ERA5_SPACING_DEG} deg spacing")
    for la, lo in pts:
        print(f"    {la:.2f} N  {lo:.2f} E")

    if HOURLY_CSV.exists():
        H = pd.read_csv(HOURLY_CSV, parse_dates=["time"])
        print(f"\nreusing cached hourly ERA5: {len(H):,} rows")
    else:
        frames = []
        for i, (la, lo) in enumerate(pts, 1):
            df = fetch_point(la, lo, START, END)
            frames.append(df)
            print(f"  [{i}/{len(pts)}] {la:.2f},{lo:.2f}: {len(df):,} hours",
                  flush=True)
        H = pd.concat(frames, ignore_index=True)
        H.to_csv(HOURLY_CSV, index=False)
    print(f"hourly ERA5: {H.time.min()} .. {H.time.max()}, "
          f"{H.time.nunique():,} timestamps x {len(pts)} points")

    # meteorological direction is the direction wind comes FROM
    th = np.radians(H.wind_dir.to_numpy())
    H["u10"] = -H.wind_speed.to_numpy() * np.sin(th)
    H["v10"] = -H.wind_speed.to_numpy() * np.cos(th)

    # field summary per timestamp
    F = (H.groupby("time")
           .agg(wind_speed_mean=("wind_speed", "mean"),
                wind_speed_median=("wind_speed", "median"),
                wind_speed_p90=("wind_speed", lambda s: float(np.percentile(s, 90))),
                u10_mean=("u10", "mean"), v10_mean=("v10", "mean"),
                mslp_mean=("mslp", "mean"))
           .reset_index().sort_values("time"))
    # Direction is a VECTOR (circular) mean -- arctan2 of the averaged
    # components -- never an arithmetic mean of degrees, which would be
    # meaningless across the 0/360 wrap.
    F["wind_dir_mean"] = (np.degrees(np.arctan2(-F.u10_mean, -F.v10_mean))) % 360
    # Components are resolved from u,v directly against the axis unit vector
    # ax = (sin(theta), cos(theta)), i.e. along = u*sin(theta) + v*cos(theta),
    # not reconstructed from the mean angle.
    #
    # SIGN CONVENTION (the axis has a 180 deg ambiguity; this pins it):
    #   positive along  = WEST_TO_EAST      (ax points east, ax[0] > 0)
    #   positive cross  = SOUTH_TO_NORTH    (along, cross) is right-handed
    # Without this, the sign of along_estuary_wind is uninterpretable and
    # set-up cannot be told from set-down.
    F["along_estuary_wind"] = F.u10_mean * ax[0] + F.v10_mean * ax[1]
    F["cross_estuary_wind"] = -F.u10_mean * ax[1] + F.v10_mean * ax[0]
    n_ph = len(H)
    print(f"\nfield summary: {len(F):,} hourly timestamps x {len(pts)} spatial "
          f"samples = {n_ph:,} point-hour records")
    print(f"  U10 mean {F.wind_speed_mean.mean():.2f} m/s, "
          f"p90 {np.percentile(F.wind_speed_mean,90):.2f}, "
          f"max {F.wind_speed_mean.max():.2f}")

    # ------------------------------------------------ join to S1 events
    # Real acquisition timestamps must come from a ZONE_3 STAC query: the
    # ZONE_1 asset inventory does not contain zone-3-only overpasses, and a
    # daily mean would defeat the point of matching wind to the instant.
    cache = CFG.TABLES / "p0e_zone3_s1_events.csv"
    if cache.exists():
        S = pd.read_csv(cache, parse_dates=["acquisition_datetime"])
        print(f"\nreusing cached ZONE_3 S1 event list: {len(S)} events")
    else:
        print("\nquerying STAC for ZONE_3 Sentinel-1 acquisition timestamps ...")
        bbox = [round(v, 5) for v in z_ll.bounds]
        rows, seen = [], set()
        for y in range(int(START[:4]), int(END[:4]) + 1):
            for q0, q1 in (("01-01", "03-31"), ("04-01", "06-30"),
                           ("07-01", "09-30"), ("10-01", "12-31")):
                feats = _stac(bbox, f"{y}-{q0}", f"{y}-{q1}")
                for f in feats:
                    p = f["properties"]
                    orb = p.get("sat:relative_orbit")
                    if orb is None or not f.get("geometry"):
                        continue
                    from shapely.geometry import shape as _sh
                    if not _sh(f["geometry"]).intersects(z_ll):
                        continue
                    key = (p["datetime"][:10], int(orb))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(dict(date=key[0], relative_orbit=key[1],
                                     orbit_state=p.get("sat:orbit_state", ""),
                                     acquisition_datetime=p["datetime"][:19]))
                print(f"  {y} {q0[:2]}-{q1[:2]}: {len(feats)} assets, "
                      f"{len(rows)} events so far", flush=True)
        S = pd.DataFrame(rows)
        S["acquisition_datetime"] = pd.to_datetime(S.acquisition_datetime)
        S.to_csv(cache, index=False)
    S["analysis_zone"] = ZONE
    print(f"ZONE_3 events with a true acquisition timestamp: {len(S)}")
    hh = S.acquisition_datetime.dt.hour
    print(f"  overpass hours (UTC): {sorted(hh.unique())}")

    Fi = F.set_index("time")
    cols = ["wind_speed_mean", "wind_speed_median", "wind_speed_p90",
            "wind_dir_mean", "along_estuary_wind", "cross_estuary_wind",
            "mslp_mean"]
    # interpolate the hourly field to each acquisition instant
    tgt = pd.DatetimeIndex(S.acquisition_datetime)
    Fr = Fi[cols].reindex(Fi.index.union(tgt)).sort_index().interpolate(
        method="time", limit_direction="both")
    for c in cols:
        S[c] = Fr.loc[tgt, c].to_numpy()
    S["wind_class"] = [wind_class(u) for u in S.wind_speed_mean]
    S["season"] = [season_of(t) for t in S.acquisition_datetime]
    S.to_csv(CFG.TABLES / "p0e_zone3_s1_wind_joined.csv", index=False)

    print("\nwind class distribution over ZONE_3 events:")
    print(S.wind_class.value_counts().reindex(
        [n for _, _, n in WIND_BINS]).fillna(0).astype(int).to_string())
    print("\nseason distribution:")
    print(S.season.value_counts().to_string())

    # ------------------------------------------------ coverage matrix
    M = (S.groupby(["wind_class", "season", "relative_orbit"])
           .size().reset_index(name="available_events"))
    M.to_csv(CFG.TABLES / "p0e_zone3_coverage_matrix.csv", index=False)
    print(f"\ncoverage matrix wind x season x orbit: {len(M)} populated cells "
          f"of {4*4*S.relative_orbit.nunique()} possible")
    piv = S.pivot_table(index="wind_class", columns="season",
                        values="date", aggfunc="count").reindex(
        [n for _, _, n in WIND_BINS])
    print(piv.fillna(0).astype(int).to_string())

    # ------------------------------------------------ first-pass manifest
    # balance across wind x season x orbit; the same orbit must appear under
    # several wind states so wind cannot be confounded with geometry
    S["_r"] = S.groupby(["wind_class", "season", "relative_orbit"]
                        ).cumcount()
    pick = S[S._r < TARGET_PER_CELL].copy()
    pick = pick.drop(columns="_r")
    pick["select_reason"] = ("zone3_wind_stratified_firstpass")
    pick.to_csv(CFG.TABLES / "p0e_zone3_firstpass_manifest.csv", index=False)
    print(f"\nFIRST-PASS MANIFEST: {len(pick)} events "
          f"(target {TARGET_PER_CELL} per wind x season x orbit cell)")
    print(pick.pivot_table(index="wind_class", columns="season", values="date",
                           aggfunc="count").reindex(
        [n for _, _, n in WIND_BINS]).fillna(0).astype(int).to_string())
    print("\n  orbits represented per wind class (wind must not be confounded "
          "with geometry):")
    for wc in [n for _, _, n in WIND_BINS]:
        o = sorted(pick[pick.wind_class == wc].relative_orbit.unique())
        print(f"    {wc:<9} orbits {o}")
    print(f"\nESTIMATED DOWNLOAD: {len(pick)} events x {GB_PER_EVENT:.3f} GB "
          f"= {len(pick)*GB_PER_EVENT:.1f} GB")
    print(f"  (vs {len(S)} catalogued events = "
          f"{len(S)*GB_PER_EVENT:.0f} GB for a blind full fetch)")

    # ------------------------------------------------ figure
    fig, ax2 = plt.subplots(2, 2, figsize=(14, 9))
    a = ax2[0, 0]
    a.plot(F.time, F.wind_speed_mean, lw=0.35, color=BLUE)
    for lo, hi, nm in WIND_BINS[:-1]:
        a.axhline(hi, color=GREY, ls=":", lw=1)
    a.set_ylabel("ZONE_3 mean U10 (m/s)")
    a.set_title("a · ERA5 hourly wind over the estuary, 2019-2023\n"
                "dotted = provisional class edges", fontsize=10.2, loc="left")
    a.grid(alpha=0.25)
    a = ax2[0, 1]
    a.hist(F.wind_speed_mean, bins=70, color=BLUE, alpha=0.85)
    for lo, hi, nm in WIND_BINS[:-1]:
        a.axvline(hi, color=RED, ls="--", lw=1.2)
    a.set_xlabel("mean U10 (m/s)"); a.set_ylabel("hours")
    a.set_title("b · wind-speed distribution (class edges are operational, "
                "not physical)", fontsize=10.2, loc="left")
    a.grid(alpha=0.25)
    a = ax2[1, 0]
    sc = a.scatter(S.along_estuary_wind, S.cross_estuary_wind, s=18,
                   c=S.wind_speed_mean, cmap="viridis")
    fig.colorbar(sc, ax=a, label="U10 (m/s)")
    a.axhline(0, color=GREY, lw=0.8); a.axvline(0, color=GREY, lw=0.8)
    a.set_xlabel(f"along-estuary wind (m/s), axis {bearing:.0f} deg")
    a.set_ylabel("cross-estuary wind (m/s)")
    a.set_title("c · wind resolved on the derived estuary axis, at each S1 "
                "overpass", fontsize=10.2, loc="left")
    a.grid(alpha=0.25)
    a = ax2[1, 1]
    order = [n for _, _, n in WIND_BINS]
    tab = S.pivot_table(index="wind_class", columns="season", values="date",
                        aggfunc="count").reindex(order).fillna(0)
    bot = np.zeros(len(tab))
    for s, c in zip(["DJF", "MAM", "JJA", "SON"], [BLUE, GREEN, AMBER, PURPLE]):
        if s in tab.columns:
            a.bar(range(len(tab)), tab[s], bottom=bot, color=c, label=s)
            bot += tab[s].to_numpy()
    a.set_xticks(range(len(tab))); a.set_xticklabels(tab.index, fontsize=9)
    a.set_ylabel("catalogued S1 events"); a.legend(fontsize=8.5)
    a.set_title("d · ZONE_3 event availability by wind class and season",
                fontsize=10.2, loc="left")
    a.grid(alpha=0.25, axis="y")
    fig.suptitle("p0e · ZONE_3 estuary: wind must be known BEFORE Sentinel-1 "
                 "is selected", y=1.0, fontsize=12.4)
    fig.tight_layout()
    out = FIGDIR / "p0e_zone3_wind.png"
    fig.savefig(out, dpi=165, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ freeze
    import hashlib
    man = CFG.TABLES / "p0e_zone3_firstpass_manifest.csv"
    meta = dict(
        analysis_zone=ZONE,
        manifest=str(man),
        sha256=hashlib.sha256(man.read_bytes()).hexdigest(),
        n_events_selected=int(len(pick)),
        n_events_catalogued=int(len(S)),
        estimated_download_gb=round(len(pick) * GB_PER_EVENT, 1),
        download_target=str(CFG.S1_CACHE / ZONE),
        era5_source="ERA5 hourly via Open-Meteo archive API",
        era5_timestamps=int(len(F)),
        era5_spatial_samples=int(len(pts)),
        era5_point_hour_records=int(n_ph),
        era5_period=[START, END],
        estuary_axis_azimuth_deg=round(float(bearing), 2),
        estuary_axis_unit_east_north=[round(float(ax[0]), 4),
                                      round(float(ax[1]), 4)],
        positive_along_estuary="WEST_TO_EAST",
        positive_cross_estuary="SOUTH_TO_NORTH",
        axis_cross_check_landmark_deg=90.0,
        axis_derivation="PCA over the landmark-bounded liman basin; a PCA over "
                        "all ZONE_3 water gives 174 deg (Southern Bug arm) and "
                        "south-of-5180000 gives 127 deg (clipped sea block)",
        wind_direction_averaging="vector (circular) mean of u,v components",
        along_cross_derivation="resolved from u,v against the axis unit "
                               "vector, not from the mean angle",
        wind_classes_status="PROVISIONAL operational bin edges; must be "
                            "revalidated against event-level VV/VH over "
                            "stable open water",
        wind_class_edges_m_s={n: [lo, hi] for lo, hi, n in WIND_BINS},
        frozen_utc=pd.Timestamp.utcnow().isoformat())
    (CFG.TABLES / "p0e_zone3_manifest_freeze.json").write_text(
        json.dumps(meta, indent=2))
    print(f"\nFROZEN: sha256 {meta['sha256'][:16]}...  "
          f"axis {meta['estuary_axis_azimuth_deg']} deg, "
          f"+along = {meta['positive_along_estuary']}, "
          f"+cross = {meta['positive_cross_estuary']}")
    print(f"  download target: {meta['download_target']}")

    for p in ("p0e_zone3_era5_hourly", "p0e_zone3_s1_wind_joined",
              "p0e_zone3_coverage_matrix", "p0e_zone3_firstpass_manifest"):
        print(f"-> {CFG.TABLES/(p+'.csv')}")
    print(f"-> {CFG.TABLES/'p0e_zone3_manifest_freeze.json'}")
    print(f"-> {out}")
    print("\nFROZEN, NOT DOWNLOADED. ZONE_1 rebuild takes priority.")


if __name__ == "__main__":
    main()
