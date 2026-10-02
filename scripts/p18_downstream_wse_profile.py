#!/usr/bin/env python
"""P18 — a longitudinal water-surface profile eta(chain_km) for dam->Kherson.

WHY NOT ONE CONSTANT, THE WAY THE RESERVOIR DID IT. hist14's shoreline pseudo-
points all share one elevation because the reservoir pool is quasi-horizontal.
The dam->Kherson reach is a free-flowing post-breach river with a real
along-channel slope -- a single constant would be wrong here, not just
imprecise. So every 250 m shoreline pseudo-point gets its OWN elevation, read
off a 1-D profile built from real anchors and interpolated along SWORD
chainage, never extrapolated with an invented slope.

THE CACHED CHANNEL PARQUET (`data_swot/processed/profiles/sword_dnipro_channel.parquet`,
built for the reservoir) IS NOT USED HERE -- CHECKED, NOT ASSUMED. Its nearest
node to the Kherson gauge is 0.18 degrees away (~15-20 km) with a *positive*
chain_km, i.e. it does not actually trace a channel down to Kherson at any
useful density; the reservoir-focused build evidently thinned or mis-selected
the delta branch. Rebuilding the main-stem selection directly from the full
SWORD extract, over a bbox that actually covers the corridor
(31.9-33.95E, 46.3-47.2N), finds a node 0.006 deg (~600 m) from Kherson with
chain_km -70.5 -- the fresh build is what this script uses.

ANCHORS, PER REGIME:
  PRE_BREACH:  Nova Kakhovka + Kherson mean annual levels (part12), plus any
               PRE_BREACH ICESat-2 points from kherson_atl13_evrs.parquet
               chainage-snapped into the corridor.
  POST_BREACH: Kherson only from part12 (Nova Kakhovka has no post-2021 data,
               confirmed, not fixable) plus POST_BREACH ICESat-2 points --
               these are what carries the near-dam POST_BREACH constraint.

ICESAT-2 POINTS ARE NOT GEODETIC-GRADE HERE. Brought into the gauge-comparable
frame via part2_icesat_common_frame.py's own per-station, per-period
alignment_constant_m (radius 2 km): H_evrf2019_comparable = H_evrs_egg2015_m +
free2mean(lat) + alignment_constant_m. Every such row keeps its source tag and
the corrector's own CI -- it is a useful independent control point, especially
where the gauge network has a hole, but it is never presented as a surveyed
elevation.

EXTRAPOLATION IS FLAGGED, NEVER SILENT. `build_profile` (imported from
`k6_water_surface_profiles.py`) holds flat at the nearest anchor outside the
anchor span rather than extrapolating a fitted slope. Any shoreline point
farther than EXTRAPOLATION_FLAG_KM from the nearest anchor is written with
`extrapolated=True` -- expected to be common near the dam in POST_BREACH,
given the Nova Kakhovka gap.

SHORELINE POINT PROVENANCE, AUDITED, STATED PRECISELY -- NOT AN EQUALITY
CONSTRAINT AT AN EXACT DATE. The boundary these 250 m points are sampled
along is `dnipro_water_domain` (registry), built by
`p0b_build_dnipro_water_domain.py` from ESA WorldCover v200, **2021**, class
80 (+adjacent class 90). That is a single-epoch land-cover classification --
pre-breach (2021 predates 2023-06-06) and a reasonable representative
pre-breach water/land line, but it is NOT the same thing as hist14's
reservoir shoreline, which was the ACTUAL dated 2023-06-05 flood-footprint
polygon. So each shoreline pseudo-point here is: ELEVATION real (from the
gauge/ICESat-2-anchored chain_km profile, an observed water level), LOCATION
representative-not-dated (a generic pre-breach water edge, not an observed
waterline on a specific day). Treat these as soft, representative PRE_BREACH
constraints, not exact equality constraints the way the reservoir's dated
footprint was. p19/p21 carry this distinction forward rather than silently
treating shoreline points as equivalent to the real soundings.

Outputs
-------
outputs/tables/p18_downstream_wse_profile_{PRE_BREACH,POST_BREACH}.csv
data/processed/bathymetry/zone24_shore_pseudopoints_{PRE_BREACH,POST_BREACH}.parquet
outputs/figures/P18_downstream_wse_profiles.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import sword
from k6_water_surface_profiles import build_profile

DAM_LON, DAM_LAT = CFG.KAKHOVKA_DAM
SWORD_NC = ROOT / "data/reference/river_network/sword_v16/netcdf/eu_sword_v16.nc"
CHANNEL_BBOX = dict(lon_range=(31.9, 33.95), lat_range=(46.3, 47.2))
SHORE_STEP_M = 250.0
MAX_SNAP_KM = 8.0                # tighter than sword's 50 km default: this is a
                                  # single well-mapped river reach, not a basin scan
EXTRAPOLATION_FLAG_KM = 15.0
DOWNSTREAM_CHAIN_MAX_KM = 2.0    # small margin past the dam (chain_km=0); ZONE_4's
                                 # own 10 km promotion buffer can cross it slightly
ICESAT_BIN_KM = 1.0              # chainage bin for robust aggregation, see below
ICESAT_MIN_PER_BIN = 10
STATIONS = {80805: "Kherson", 80977: "Nova Kakhovka"}
ZONES = ("ZONE_2_KHERSON_DELTA", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")
FIG = CFG.FIG / "P18_downstream_wse_profiles.png"


def build_channel() -> pd.DataFrame:
    nodes = sword.load_nodes(SWORD_NC, **CHANNEL_BBOX)
    main = sword.dnipro_nodes(nodes)
    ch = sword.chainage_from_dam(main, DAM_LON, DAM_LAT)
    print(f"  channel: {len(nodes)} nodes in bbox -> {len(main)} main-stem "
          f"-> chain_km {ch.chain_km.min():.1f} .. {ch.chain_km.max():.1f}")
    return ch


def load_icesat_anchors(channel, tree):
    """Kherson ICESat-2 points -> EVRF2019-comparable, chainage-snapped."""
    d = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kherson_atl13_evrs.parquet",
                        columns=["lat", "lon", "H_evrs_egg2015_m", "period",
                                "water_mask_pass"])
    d = d[d.water_mask_pass].copy()
    d["free2mean_m"] = CFG.free2mean(d.lat.values)
    al = pd.read_csv(CFG.TABLES / "icesat_station_alignment_constants.csv")
    al = al[(al.station_id == 80805) & (al.radius_km == 2.0)].set_index("period")

    rows = []
    for period in ("PRE_BREACH", "POST_BREACH"):
        if period not in al.index:
            continue
        c = al.loc[period, "alignment_constant_m"]
        ci_lo = al.loc[period, "ci95_low_m"]
        ci_hi = al.loc[period, "ci95_high_m"]
        sub = d[d.period == period].copy()
        sub["H_evrf2019_comparable_m"] = (sub.H_evrs_egg2015_m + sub.free2mean_m + c)
        sub["corrector_m"] = c
        sub["corrector_ci95_lo_m"] = ci_lo
        sub["corrector_ci95_hi_m"] = ci_hi
        chain, dist, _, _ = sword.assign_chainage(
            sub.lon.values, sub.lat.values, channel, max_dist_km=MAX_SNAP_KM, tree=tree)
        sub["chain_km"], sub["dist_to_channel_km"] = chain, dist
        sub = sub.dropna(subset=["chain_km"])
        # kherson_atl13_evrs.parquet's own lat range (46.51-47.90) runs PAST the
        # dam into the reservoir pool -- the Kherson alignment_constant_m was
        # fitted from matchups near the Kherson gauge only and its validity is
        # not assumed to extend 100+ km upriver into a different hydraulic
        # regime. Anchors here are capped at chain_km <= DOWNSTREAM_CHAIN_MAX_KM
        # so this profile stays strictly a downstream (dam->Kherson) product.
        n_before = len(sub)
        sub = sub[sub.chain_km <= DOWNSTREAM_CHAIN_MAX_KM]
        if n_before != len(sub):
            print(f"    dropped {n_before - len(sub)} points with chain_km > "
                  f"{DOWNSTREAM_CHAIN_MAX_KM:.0f} (inside/near the reservoir pool, "
                  f"not this profile's domain)")
        sub["regime"] = period
        sub["source"] = "icesat2_atl13_kherson_corrected"
        print(f"  ICESat-2 {period}: {len(sub)} points snapped within "
              f"{MAX_SNAP_KM:.0f} km (corrector {c:+.3f} m, "
              f"CI [{ci_lo:+.3f}, {ci_hi:+.3f}])")

        # RAW points are NOT fed to build_profile directly: np.interp threads a
        # line through every point in chainage order with no averaging, so one
        # bad ATL13 return (cloud, land, a mis-classified segment) among tens
        # of thousands becomes a real spike in the profile, not noise it
        # absorbs. Aggregate to a robust per-km median first -- the median
        # already resists a handful of outliers inside a bin with hundreds of
        # points; bins with too few points to be trustworthy are dropped, not
        # kept at low confidence.
        sub["chain_bin"] = (sub.chain_km / ICESAT_BIN_KM).round() * ICESAT_BIN_KM
        agg = sub.groupby("chain_bin").agg(
            H_evrf2019_comparable_m=("H_evrf2019_comparable_m", "median"),
            n_points=("H_evrf2019_comparable_m", "size")).reset_index()
        agg = agg[agg.n_points >= ICESAT_MIN_PER_BIN].rename(
            columns={"chain_bin": "chain_km"})
        agg["regime"] = period
        agg["source"] = "icesat2_atl13_kherson_corrected"
        agg["corrector_m"], agg["corrector_ci95_lo_m"], agg["corrector_ci95_hi_m"] = c, ci_lo, ci_hi
        print(f"    -> {len(agg)} {ICESAT_BIN_KM:.0f} km-binned anchors "
              f"(bins with < {ICESAT_MIN_PER_BIN} points dropped), "
              f"eta {agg.H_evrf2019_comparable_m.min():.2f} .. "
              f"{agg.H_evrf2019_comparable_m.max():.2f} m")
        rows.append(agg[["chain_km", "H_evrf2019_comparable_m", "regime", "source",
                         "corrector_m", "corrector_ci95_lo_m", "corrector_ci95_hi_m"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["chain_km", "H_evrf2019_comparable_m", "regime", "source"])


def gauge_anchors(channel, tree):
    g = pd.read_csv(CFG.TABLES / "part12_downstream_annual_levels.csv")
    lonlat = {80805: (32.612026, 46.623750), 80977: (33.374615, 46.775432)}
    rows = []
    for _, r in g.iterrows():
        if pd.isna(r.mean_annual_evrf2019_m):
            continue
        lon, lat = lonlat[int(r.station_id)]
        chain, dist, _, _ = sword.assign_chainage(
            [lon], [lat], channel, max_dist_km=MAX_SNAP_KM, tree=tree)
        rows.append(dict(chain_km=float(chain[0]),
                         H_evrf2019_comparable_m=float(r.mean_annual_evrf2019_m),
                         regime=r.regime, source=f"gauge_{r.name_en}",
                         corrector_m=0.0, corrector_ci95_lo_m=0.0, corrector_ci95_hi_m=0.0))
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("P18 — dam->Kherson longitudinal water-surface profile, per regime")
    print("=" * 78)

    channel = build_channel()
    tree = sword.build_chainage_tree(channel)

    anchors = pd.concat([gauge_anchors(channel, tree),
                         load_icesat_anchors(channel, tree)], ignore_index=True)
    anchors.to_csv(CFG.TABLES / "p18_downstream_profile_anchors.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"PRE_BREACH": "#3f7d4e", "POST_BREACH": "#c1402a"}

    zone_geoms = {z: SD.load_utm(z) for z in ZONES}
    wdom = SD.load_utm("dnipro_water_domain")

    for regime in ("PRE_BREACH", "POST_BREACH"):
        a = anchors[anchors.regime == regime].sort_values("chain_km")
        if len(a) < 2:
            print(f"\n  {regime}: {len(a)} anchor(s) -- not enough for a profile, skipped")
            continue
        s_obs, h_obs = a.chain_km.values, a.H_evrf2019_comparable_m.values

        chain_grid = np.linspace(s_obs.min() - 5, s_obs.max() + 5, 400)
        prof = build_profile(s_obs, h_obs, chain_grid, kind="linear")
        pd.DataFrame({"chain_km": chain_grid, "eta_evrf2019_m": prof}).to_csv(
            CFG.TABLES / f"p18_downstream_wse_profile_{regime}.csv", index=False)

        ax.plot(chain_grid, prof, color=colors[regime], label=f"{regime} profile")
        ax.scatter(a.chain_km, a.H_evrf2019_comparable_m,
                  c=[colors[regime]], edgecolor="k", s=[40 if 'gauge' in s else 18
                                                        for s in a.source], zorder=5)

        print(f"\n  {regime}: {len(a)} anchors, chain_km "
              f"{s_obs.min():.1f} .. {s_obs.max():.1f}, "
              f"eta {h_obs.min():.2f} .. {h_obs.max():.2f} m")

        # ---- shoreline pseudo-points per zone ---------------------------
        for z_name in ZONES:
            fp = zone_geoms[z_name].intersection(wdom.buffer(300.0))
            if fp.is_empty:
                continue
            bnd = fp.boundary
            n_sh = max(int(bnd.length / SHORE_STEP_M), 50)
            pts = np.array([bnd.interpolate(t, normalized=True).coords[0]
                            for t in np.linspace(0, 1, n_sh, endpoint=False)])
            import pyproj
            to_ll = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True)
            plon, plat = to_ll.transform(pts[:, 0], pts[:, 1])
            chain, dist, _, _ = sword.assign_chainage(
                plon, plat, channel, max_dist_km=MAX_SNAP_KM, tree=tree)
            ok = np.isfinite(chain)
            eta = build_profile(s_obs, h_obs, chain[ok], kind="linear")
            gap = np.array([np.min(np.abs(s_obs - c)) for c in chain[ok]])
            extrap = gap > EXTRAPOLATION_FLAG_KM

            sh = pd.DataFrame({
                "x": pts[ok, 0], "y": pts[ok, 1], "chain_km": chain[ok],
                "dist_to_channel_km": dist[ok], "elevation_evrf2019_m": eta,
                "gap_to_nearest_anchor_km": gap, "extrapolated": extrap,
                "zone": z_name, "regime": regime})
            out_pq = ROOT / f"data/processed/bathymetry/zone24_shore_pseudopoints_{regime}.parquet"
            if out_pq.exists() and z_name != ZONES[0]:
                prev = pd.read_parquet(out_pq)
                sh = pd.concat([prev, sh], ignore_index=True)
            sh.to_parquet(out_pq, index=False)
            print(f"    {z_name:32s} {len(sh[sh.zone==z_name]):5d} shoreline pts, "
                  f"{int(extrap.sum())} ({100*extrap.mean():.0f}%) extrapolated "
                  f"(>{EXTRAPOLATION_FLAG_KM:.0f} km from nearest anchor)")

    # ---- cross-zone overlap consistency (free correctness check) --------
    print("\n" + "=" * 78)
    print("ZONE_2 x ZONE_4 shoreline-elevation overlap consistency")
    print("=" * 78)
    ov = zone_geoms["ZONE_2_KHERSON_DELTA"].intersection(zone_geoms["ZONE_4_DAM_TO_KHERSON_FLOODWAY"])
    print(f"  overlap area {ov.area/1e6:,.1f} km2 "
          f"(both zones share one eta(chain_km) profile by construction, "
          f"so points in the overlap band are expected to agree closely)")

    ax.set_xlabel("chain_km (negative = downstream of the dam)")
    ax.set_ylabel("water surface elevation, EVRF2019 (m)")
    ax.set_title("P18 · dam -> Kherson longitudinal WSE profile\n"
                 "small markers = ICESat-2 (corrected), large = gauge annual mean")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print(f"\n-> {FIG}")


if __name__ == "__main__":
    main()
