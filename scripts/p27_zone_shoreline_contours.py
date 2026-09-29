#!/usr/bin/env python
"""P27 -- dated, mask-derived shoreline constraints for ZONE_2 / ZONE_3 / ZONE_4.

What this replaces
------------------
The zone 2/4 bed DEM (p19) is constrained by p18's pseudo-points: the boundary
of the zone ∩ WorldCover-2021 water (buffered 300 m), carrying a MEAN-ANNUAL
water level read off a chainage profile -- "ELEVATION real, LOCATION
representative-not-dated" in p18's own words. Neither the position nor the
date is an observation. ZONE_1's DEM does it differently and better: hist23
takes dated Sentinel-2 water masks whose water level is KNOWN from the gauge
that day, hist26 extracts the boundary as a continuous contour, hist24 feeds it
to kriging as a SOFT constraint with a per-point sigma budget.

This script is hist23 + hist26 for the three downstream zones, on the p25 zone
stacks. The kriging step is p28.

Dates and levels
----------------
Dates: p25 manifest rows with regime PRE_BREACH and valid_frac_inside >= 0.80,
that also have a Kherson (80805) gauge reading in
`all_water_levels_common_frame.csv` (source gauge, domain downstream, EVRF2019).
Grouping: as hist23, one stage group if the level spread across the chosen
dates is <= LEVEL_SPREAD_MAX (0.35 m); otherwise the dates are split at the
level median into two groups. For the lower Dnipro the pre-breach stage range
is ~0.5 m, so expect one or two groups -- not the reservoir's three.

Level along the zone: H(x, date) = H_Kherson(date) + G * s(x), with
G = +0.0014 m/km (the clean-track pre-breach water-surface gradient,
SCIENTIFIC_FINDINGS Finding 6) and s(x) = chainage upstream of Kherson (km),
sampled from the nearest p18 pseudo-point's `chain_km` (used ONLY for the
along-channel coordinate, never for elevation). ZONE_2/3 sit within a few km
of the gauge so the term is centimetres; ZONE_4 reaches ~60 km, ~8 cm.

Sigma budget per group (quadrature, hist24:331-371 adapted)
-----------------------------------------------------------
    sigma_epsg9902            0.068   BS-77 -> EVRF2019 grid (config)
    sigma_gauge_read          0.010
    sigma_level_spread        measured across the group's dates
    sigma_day_variability     0.044   Kherson NMAD |dH|/day (kherson_daily_variability.csv)
    sigma_gradient            |G| * s_max * 0.5   (half the along-zone range, as a bound)
    sigma_wind_setup          ZONE_2/4: 0.09 (Kherson p90 |dH|/day, used as a proxy bound)
                              ZONE_3 : scenario band 0.10 .. 0.20 -> sigma_total_lo / _hi
    sigma_shoreline_position  slope * cell / 2
    sigma_mixed_pixel         slope * 20
    sigma_emergent_vegetation slope * 20 * f_reed  (f_reed = share of vertices with a
                              REED_OR_FLOODED_VEGETATION pixel in their 3x3)
Slope is fitted from SOUNDINGS within 1 km (never from a DEM). No tide model
exists for the liman; the wind band is carried as a band, not collapsed.

Outputs
-------
data/processed/bathymetry/zone_shore_contours_<ZONE>_PRE_BREACH.gpkg
    layer contours: polygon per group + H_evrf2019_m, sigma_total_m (+lo/hi), dates, n_dates
    layer vertices: densified (150 m) points with x, y, H, sigma, slope, chain_km
outputs/tables/p27_zone_shoreline_uncertainty.csv   (one row per zone x group)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP
from swot_dnipro import shoreline as SH
from swot_dnipro import spatial_domains as SD

ZONES = ("ZONE_2_KHERSON_DELTA", "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")
CELL = 20.0
MIN_VALID = 0.10          # a date needs DATA in the zone, not full coverage --
                          # coverage is a property of the multi-date composite
MIN_COMPOSITE_COV = 0.60  # ...and THIS is the gate that matters. ZONE_4's dates are
                          # 0.11-0.61 each (western tiles only) yet composite to 100 %
LEVEL_SPREAD_MAX = 0.35
MIN_PART_KM2 = 0.05
SPACING_M = 150.0
GRAD_M_PER_KM = 0.0014
SIG_EPSG9902, SIG_GAUGE_READ, SIG_DAY = 0.068, 0.010, 0.044
SIG_WIND = {"ZONE_2_KHERSON_DELTA": (0.09, 0.09), "ZONE_4_DAM_TO_KHERSON_FLOODWAY": (0.09, 0.09),
            "ZONE_3_DNIPRO_BUG_ESTUARY": (0.10, 0.20)}
STACKS = CFG.BULK_ROOT / "zone_spectral"
OUT_GPKG = ROOT / "data" / "processed" / "bathymetry"


def kherson_levels() -> pd.Series:
    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    g = w[(w.source == "gauge") & (w.domain == "downstream")]
    return g.groupby("date").transformed_level_m.median()


def load_dates(zone: str, lev: pd.Series) -> pd.DataFrame:
    man = pd.read_csv(CFG.TABLES / f"p25_zone_spectral_manifest_{zone}.csv")
    ok = man[(man.regime == "PRE_BREACH") & man.status.isin(["WRITTEN", "CACHED"])
             & (man.valid_frac_inside >= MIN_VALID)].copy()
    ok["H"] = ok.date.map(lev)
    ok = ok.dropna(subset=["H"]).sort_values("valid_frac_inside", ascending=False)
    return ok


def group_dates(D: pd.DataFrame) -> list[pd.DataFrame]:
    if D.H.max() - D.H.min() <= LEVEL_SPREAD_MAX:
        return [D]
    m = D.H.median()
    return [D[D.H <= m], D[D.H > m]]


def read_products(zone: str, date: str):
    with rasterio.open(STACKS / zone / f"{date}_indices.tif") as s:
        ndwi = s.read(2).astype("f4"); mndwi = s.read(3).astype("f4")
        nd = s.nodata
    ndwi[ndwi == nd] = np.nan; mndwi[mndwi == nd] = np.nan
    ndwi /= SP.INDEX_SCALE; mndwi /= SP.INDEX_SCALE
    with rasterio.open(STACKS / zone / f"{date}_water3.tif") as s:
        w3 = s.read(1)
    with rasterio.open(STACKS / zone / f"{date}_class.tif") as s:
        cls = s.read(1)
    return ndwi, mndwi, w3, cls


def chainage_lookup(zone: str):
    """Nearest p18 pseudo-point chain_km, for the along-channel coordinate only."""
    p = ROOT / "data/processed/bathymetry/zone24_shore_pseudopoints_PRE_BREACH.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d = d[d.zone == zone] if (d.zone == zone).any() else d
    if len(d) == 0:
        return None
    tree = cKDTree(np.c_[d.x.values, d.y.values])
    ck = d.chain_km.values
    return lambda xy: ck[tree.query(xy, k=1)[1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=ZONES)
    a = ap.parse_args()
    zone = a.zone
    print("=" * 78)
    print(f"P27 -- dated S2 shoreline constraints for {zone}")
    print("=" * 78)

    G = SP.zone_grid(zone, CELL)
    lev = kherson_levels()
    D = load_dates(zone, lev)
    print(f"  PRE_BREACH dates with data in the zone and a Kherson level: {len(D)}  "
          f"H range {D.H.min():.2f}..{D.H.max():.2f} m EVRF2019")
    if len(D) < 3:
        raise SystemExit("fewer than 3 usable dates -- not a constraint, stop")

    snd = pd.read_parquet(ROOT / "data/processed/bathymetry/manual_soundings_evrf2019.parquet")
    snd = snd[snd[zone]]
    snd_xy, snd_z = np.c_[snd.x.values, snd.y.values], snd.H_bed_evrf2019_m.values
    chain = chainage_lookup(zone)
    wind_lo, wind_hi = SIG_WIND[zone]

    groups = group_dates(D)
    rows, polys, verts = [], [], []
    for gi, Dg in enumerate(groups, 1):
        gid = f"S{gi}"
        print(f"\n  group {gid}: {len(Dg)} dates, H {Dg.H.min():.3f}..{Dg.H.max():.3f} m")
        fields, valids, reed_any = [], [], np.zeros((G["ny"], G["nx"]), bool)
        for r in Dg.itertuples():
            ndwi, mndwi, w3, cls = read_products(zone, r.date)
            W, V = SH.decision_field(ndwi, mndwi, w3)
            fields.append(W); valids.append(V)
            reed_any |= cls == SP.CLASSES_INV["REED_OR_FLOODED_VEGETATION"]
        W, V, src = SH.composite_first_valid(fields, valids)
        inside = G["inside"]
        cov = float((V & inside).sum() / inside.sum())
        if cov < MIN_COMPOSITE_COV:
            print(f"    composite coverage {100*cov:.1f} % < {100*MIN_COMPOSITE_COV:.0f} % -- group skipped"); continue
        water_km2 = float(((W > 0) & V & inside).sum() * CELL ** 2 / 1e6)
        region = SH.main_water_region(W, V & inside, MIN_PART_KM2, CELL) & inside
        # water < 0; not-observed and outside-zone cells closed as land so the
        # main body's ring survives where water crosses the zone boundary
        F = SH.close_field(W, V, inside)
        rings, _ = SH.contour_geometry(F, np.ones_like(V), G, region, CELL)
        poly = SH.rings_to_polygon(rings, F, G, CELL)
        if poly is None:
            print("    no polygon -- skipping group"); continue
        area = poly.area / 1e6; length = poly.length / 1e3
        pts_all = SH.densify(poly, SPACING_M)
        # vertices on the zone edge or a cloud edge are data boundaries, not shore
        safe = SH.edge_safe(V, inside, CELL)
        ci_a = np.clip(((pts_all[:, 0] - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
        ri_a = np.clip(((G["y1"] - pts_all[:, 1]) / CELL).astype(int), 0, G["ny"] - 1)
        keep_v = safe[ri_a, ci_a]
        # The registry ZONE_4 polygon reaches ~41 km UPSTREAM of the dam, so its
        # PRE_BREACH water composite includes the reservoir tail at ~16 m. A
        # Kherson-based level is wrong there by 16 m -- exactly the defect p18
        # carried (F-16) and that this script would otherwise repeat. Vertices
        # inside the registry's reservoir polygon are dropped: the reservoir
        # shore is ZONE_1's business (hist23/24), the floodway's is this one's.
        import shapely
        res = SD.load_utm("reservoir_full_pool_prebreach")
        in_res = shapely.contains_xy(res, pts_all[:, 0], pts_all[:, 1])
        n_res_dropped = int((keep_v & in_res).sum())
        keep_v &= ~in_res
        pts = pts_all[keep_v]
        n_edge_dropped = int((~safe[ri_a, ci_a]).sum())
        print(f"    composite water {water_km2:,.1f} km2 -> polygon {area:,.1f} km2 "
              f"({100*area/max(water_km2,1e-9):.0f} %); vertices {len(pts_all):,} -> "
              f"{len(pts):,} after dropping {n_edge_dropped:,} on zone/cloud edges and "
              f"{n_res_dropped:,} inside the reservoir polygon")
        slope = SH.slope_from_soundings(pts, snd_xy, snd_z, 1000.0)
        sl_med = float(np.nanmedian(slope)) if np.isfinite(slope).any() else np.nan
        slope = np.where(np.isfinite(slope), slope, sl_med)
        # along-channel coordinate and dated level
        s_km = chain(pts) if chain is not None else np.zeros(len(pts))
        H0 = float(Dg.H.mean())
        H = H0 + GRAD_M_PER_KM * s_km
        # reed adjacency at vertices
        ci = np.clip(((pts[:, 0] - G["x0"]) / CELL).astype(int), 0, G["nx"] - 1)
        ri = np.clip(((G["y1"] - pts[:, 1]) / CELL).astype(int), 0, G["ny"] - 1)
        reed_nb = ndimage.binary_dilation(reed_any, iterations=1)[ri, ci]
        f_reed = float(reed_nb.mean())
        comp = dict(sigma_epsg9902_m=SIG_EPSG9902, sigma_gauge_read_m=SIG_GAUGE_READ,
                    sigma_level_spread_m=float(Dg.H.max() - Dg.H.min()),
                    sigma_day_variability_m=SIG_DAY,
                    sigma_gradient_m=abs(GRAD_M_PER_KM) * float(np.nanmax(np.abs(s_km)) if len(s_km) else 0) * 0.5,
                    sigma_shoreline_position_m=sl_med * CELL / 2,
                    sigma_mixed_pixel_m=sl_med * 20.0,
                    sigma_emergent_vegetation_m=sl_med * 20.0 * f_reed)
        base = sum(v ** 2 for v in comp.values())
        tot_lo = float(np.sqrt(base + wind_lo ** 2)); tot_hi = float(np.sqrt(base + wind_hi ** 2))
        tot = tot_hi if zone == "ZONE_3_DNIPRO_BUG_ESTUARY" else tot_lo   # conservative for the liman
        print(f"    composite coverage {100*cov:.1f} %, {len(rings)} rings, polygon {area:,.1f} km2, "
              f"shoreline {length:,.0f} km, {len(pts):,} vertices at {SPACING_M:.0f} m")
        print(f"    median slope {sl_med:.4f} m/m, f_reed {f_reed:.2f}, H0 {H0:.3f} m, "
              f"sigma_total {tot_lo:.3f}..{tot_hi:.3f} m")
        for k, v in comp.items():
            print(f"      {k:<30}{v:.3f}")
        rows.append(dict(zone=zone, group=gid, n_dates=len(Dg), dates="|".join(Dg.date),
                         H0_kherson_evrf2019_m=H0, gradient_m_per_km=GRAD_M_PER_KM,
                         composite_coverage=cov, composite_water_km2=water_km2,
                         n_rings=len(rings), polygon_area_km2=area,
                         polygon_to_water_ratio=area / max(water_km2, 1e-9),
                         shoreline_km=length, n_vertices_all=len(pts_all),
                         n_vertices_edge_dropped=n_edge_dropped, n_vertices_reservoir_dropped=n_res_dropped,
                         n_vertices=len(pts), spacing_m=SPACING_M,
                         median_slope_m_per_m=sl_med, f_reed_adjacent=f_reed,
                         sigma_wind_lo_m=wind_lo, sigma_wind_hi_m=wind_hi,
                         sigma_total_lo_m=tot_lo, sigma_total_hi_m=tot_hi, sigma_total_m=tot, **comp))
        polys.append(dict(zone=zone, group=gid, H_evrf2019_m=H0, sigma_total_m=tot,
                          sigma_total_lo_m=tot_lo, sigma_total_hi_m=tot_hi,
                          n_dates=len(Dg), dates="|".join(Dg.date), geometry=poly))
        verts.append(pd.DataFrame(dict(zone=zone, group=gid, x=pts[:, 0], y=pts[:, 1],
                                       chain_km=s_km, H_evrf2019_m=H, sigma_m=tot,
                                       slope_m_per_m=slope, reed_adjacent=reed_nb)))

    if not rows:
        raise SystemExit("no shoreline polygon produced")
    OUT_GPKG.mkdir(parents=True, exist_ok=True)
    gp = OUT_GPKG / f"zone_shore_contours_{zone}_PRE_BREACH.gpkg"
    gpd.GeoDataFrame(polys, crs=CFG.CRS_METRIC).to_file(gp, layer="contours", driver="GPKG")
    Vt = pd.concat(verts, ignore_index=True)
    gpd.GeoDataFrame(Vt, geometry=[Point(x, y) for x, y in zip(Vt.x, Vt.y)],
                     crs=CFG.CRS_METRIC).to_file(gp, layer="vertices", driver="GPKG")
    tab = CFG.TABLES / "p27_zone_shoreline_uncertainty.csv"
    T = pd.DataFrame(rows)
    if tab.exists():
        old = pd.read_csv(tab); old = old[old.zone != zone]
        T = pd.concat([old, T], ignore_index=True)
    T.to_csv(tab, index=False)
    print(f"\n-> {gp}\n-> {tab}")


if __name__ == "__main__":
    main()
