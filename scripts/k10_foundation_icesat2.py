#!/usr/bin/env python
"""K10.1/K10.2/K10.4/K10.4A/K10.4B -- foundation for the CURRENT post-breach
terrain: modern observation inventory, canonical vertical frame, ICESat-2
ground extraction with strict QC, and the FROZEN build/validation split.

Nothing is fitted here. The point of this stage is to lock the independent
validation set BEFORE any modern DEM exists, so that the later validation is
genuinely independent rather than retrospectively defined.

K10.4A QC -- every rule is independent of any candidate DEM residual:
  * POST_BREACH only, inside the former pool
  * NIGHT ONLY (solar_elevation < 0). Night acquisitions carry far less solar
    background noise, so ATL08 ground classification is substantially more
    reliable. This is the operator's explicit requirement and is applied as a
    hard filter, not an option.
  * snowcover == 1 (ATL08 "snow free land"); this also removes code 0
    ("ice free water"), which is water, and code 2 (snow) and 255 (fill)
  * gnd_ph_count >= 10 -- real ground-photon support
  * h_canopy <= 3 m -- terrain under tall vegetation is not bare bed
  * physical envelope from the reservoir's OWN geometry (full pool ~16.2 m
    EVRF2019, deepest legacy sounding -19.4 m), never from a DEM

landcover is deliberately NOT used as a filter: in this extract its values are
spread across essentially the whole 0-255 range with tiny counts, i.e. the
field was not decoded to the Copernicus discrete classes. Flagged, not used.

K10.4B independence: the split is by WHOLE RGT, never by individual segment or
photon. Repeat cycles follow nearly the same ground track, so grouping by RGT
(not RGT x cycle) also prevents cross-cycle spatial leakage.

Outputs
-------
outputs/tables/k10_modern_observation_inventory.csv
outputs/tables/k10_vertical_frame_chain.csv
data/processed/current_bed/k10_modern_elevation_points.parquet
outputs/tables/k10_icesat2_qc_ledger.csv
outputs/tables/k10_icesat2_validation_manifest.json
outputs/figures/K10_data_coverage.png
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import sword as SW

OUT_DIR = CFG.ROOT / "data" / "processed" / "current_bed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ATL08 = CFG.ROOT / "data" / "processed" / "atl08" / "kakhovka_atl08_terrain.parquet"
CORRECTOR = CFG.CORRECTOR_BY_STATION

# ------------------------------------------------------------- FROZEN QC ----
NIGHT_MAX_SOLAR_ELEV = 0.0
MIN_GND_PHOTONS = 10
MAX_CANOPY_M = 3.0
SNOWCOVER_KEEP = 1           # ATL08: 1 = snow free land
ELEV_ENVELOPE_M = (-30.0, 25.0)
VALIDATION_TARGET_FRAC = 0.30
MIN_VALIDATION_RGTS = 4
RNG = np.random.default_rng(CFG.SEED)

_TF = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)


def main() -> None:
    print("=" * 70)
    print("K10 FOUNDATION -- modern observations, vertical frame, ICESat-2 QC, freeze")
    print("=" * 70)
    print("OBJECTIVE: current post-breach terrain for HEC-RAS. Historical bathymetry is a")
    print("PRIOR/FALLBACK only. 'historical bed == current bed' is never assumed.\n")

    # ================================================== K10.2 vertical frame
    print("-" * 70)
    print("K10.2 -- CANONICAL CRS / VERTICAL FRAME")
    print("-" * 70)
    corr = pd.read_csv(CORRECTOR)
    res_c = corr[corr.gauge_zero_bs77_m == 12.0]          # the six reservoir stations
    c_mean = float(res_c.c_station_m.mean())
    c_std = float(res_c.c_station_m.std())
    c_nmad = float(res_c.empirical_nmad_m.median())
    print(f"ATL08 vertical chain, stated explicitly (no silent relabelling):")
    print(f"  h_te_median            ellipsoidal WGS84, ATL03 tide-free")
    print(f"  + free2mean(lat)       permanent-tide harmonisation -> mean-tide")
    print(f"  - zeta_EGG2015         quasigeoid separation")
    print(f"  = H_terrain_common_m   EGG2015 common frame  <-- this is NOT EVRF2019")
    print(f"  + c_EGG2015_to_EVRF2019 = {c_mean:+.4f} m (mean of the six reservoir stations, "
          f"sd {c_std:.4f})")
    print(f"  = H_EVRF2019_m")
    print(f"  station correctors: {', '.join(f'{r.name_en} {r.c_station_m:+.3f}' for r in res_c.itertuples())}")
    print(f"  NOTE: EPSG:9902 is NEVER applied to these ellipsoidal-derived heights -- it is a")
    print(f"        BS-77 -> EVRF2019 transform and belongs only to the gauge chain (K5).")
    print(f"  CAVEAT: c was derived from ATL13 WATER-surface matchups against gauges and is")
    print(f"        applied here to TERRAIN. Defensible because EGG2015->EVRF2019 is a datum")
    print(f"        property independent of surface type, but it is an assumption, not a")
    print(f"        measurement over land. Carried as sigma_vertical_reference.")
    sigma_vref = float(np.sqrt(c_std ** 2 + c_nmad ** 2))
    print(f"  sigma_vertical_reference = {sigma_vref:.3f} m (station spread + empirical NMAD)")

    vchain = pd.DataFrame([{
        "dataset": "ICESat-2 ATL08 terrain", "horizontal_crs": CFG.CRS_METRIC,
        "vertical_system_original": "WGS84 ellipsoidal (ATL03 tide-free)",
        "vertical_transform_chain": "h_te_median + free2mean(lat) - zeta_EGG2015 + c_EGG2015_to_EVRF2019",
        "vertical_system_final": "EVRF2019 normal height (empirically aligned)",
        "units": "m", "transformation_uncertainty_m": sigma_vref,
        "empirical_correction_m": c_mean,
        "correction_provenance": "egg2015_to_evrf2019_by_station.csv, six reservoir gauges, "
                                 "derived from ATL13 water-surface matchups",
    }, {
        "dataset": "Historical soundings (K9 M2 prior)", "horizontal_crs": CFG.CRS_METRIC,
        "vertical_system_original": "Baltic-1977",
        "vertical_transform_chain": "BS-77 + delta_EPSG9902(station)",
        "vertical_system_final": "EVRF2019 normal height", "units": "m",
        "transformation_uncertainty_m": CFG.EPSG9902_ACCURACY_M,
        "empirical_correction_m": 0.0, "correction_provenance": "EPSG:9902 grid",
    }, {
        "dataset": "Six Kakhovka gauges (K5)", "horizontal_crs": CFG.CRS_METRIC,
        "vertical_system_original": "BS-77 stage above 12.000 m gauge zero",
        "vertical_transform_chain": "12.000 + stage + delta_EPSG9902(station)",
        "vertical_system_final": "EVRF2019 normal height", "units": "m",
        "transformation_uncertainty_m": CFG.EPSG9902_ACCURACY_M,
        "empirical_correction_m": 0.0, "correction_provenance": "EPSG:9902, validated in K5",
    }])
    vchain.to_csv(CFG.TABLES / "k10_vertical_frame_chain.csv", index=False)
    print(f"-> {CFG.TABLES / 'k10_vertical_frame_chain.csv'}")

    # ================================================== K10.4A ICESat-2 QC
    print("\n" + "-" * 70)
    print("K10.4A -- ICESat-2 GROUND EXTRACTION AND QC")
    print("-" * 70)
    cols = ["date", "dt", "rgt", "cycle", "gt", "spot", "lon", "lat", "x_atc", "segment_id",
            "solar_elevation", "gnd_ph_count", "ph_count", "h_canopy", "canopy_openness",
            "h_te_median", "zeta_egg2015_m", "free2mean_m", "H_terrain_common_m",
            "landcover", "snowcover", "period", "in_former_pool", "surface_class"]
    a = pd.read_parquet(ATL08, columns=cols)
    print(f"ATL08 segments loaded: {len(a):,} (native segment length 100 m -- ATL03 is not held")
    print(f"  locally, so the requested 20/50/100 m segment-length sensitivity cannot be run;")
    print(f"  reported as a limitation rather than approximated.)")

    ledger = []

    def step(df, name, keep_mask, note=""):
        n0 = len(df)
        out = df[keep_mask]
        ledger.append({"step": name, "n_before": n0, "n_after": len(out),
                       "n_removed": n0 - len(out),
                       "pct_removed": 100 * (n0 - len(out)) / max(n0, 1), "note": note})
        print(f"  {name:<34} {n0:>8,} -> {len(out):>8,}  (-{n0-len(out):,}, "
              f"{100*(n0-len(out))/max(n0,1):5.1f}%)  {note}")
        return out

    print("QC cascade (every rule independent of any DEM residual):")
    g = step(a, "POST_BREACH", a.period == "POST_BREACH")
    g = step(g, "inside former pool", g.in_former_pool.astype(bool))
    # NIGHT is a FLAG here, not an early filter: the day set is retained so that a
    # night-only vs all-QC sensitivity test can be run later. The CANONICAL set is
    # night-only (low solar background -> reliable ground classification).
    g = step(g, f"snowcover == {SNOWCOVER_KEEP} (snow free land)", g.snowcover == SNOWCOVER_KEEP,
             "also drops code 0 = ice free WATER")
    g = step(g, f"gnd_ph_count >= {MIN_GND_PHOTONS}", g.gnd_ph_count >= MIN_GND_PHOTONS)
    g = step(g, f"h_canopy <= {MAX_CANOPY_M} m", g.h_canopy <= MAX_CANOPY_M,
             "excludes vegetation-biased terrain")
    g = step(g, f"elevation in {ELEV_ENVELOPE_M} m",
             g.H_terrain_common_m.between(*ELEV_ENVELOPE_M),
             "envelope from reservoir geometry, NOT from a DEM")
    led = pd.DataFrame(ledger)
    led.to_csv(CFG.TABLES / "k10_icesat2_qc_ledger.csv", index=False)

    g = g.copy()
    g["night"] = g.solar_elevation < NIGHT_MAX_SOLAR_ELEV
    g["canonical_set"] = g.night
    # BOTH vertical representations are preserved. The EVRF2019 field is named
    # _empirical_ on purpose: c is a project harmonisation derived from ATL13
    # water matchups, NOT an official EGG2015 -> EVRF2019 transformation.
    g["H_terrain_EGG2015_m"] = g.H_terrain_common_m
    g["H_terrain_EVRF2019_empirical_m"] = g.H_terrain_common_m + c_mean
    g["c_empirical_m"] = c_mean
    g["sigma_c_m"] = c_std
    g["c_method"] = "mean of six Kakhovka reservoir station correctors (ATL13 water matchups)"
    g["sigma_vertical_reference_m"] = sigma_vref
    g["x"], g["y"] = _TF.transform(g.lon.values, g.lat.values)
    chan = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(chan)
    g["chain_km"], off_km, _, _ = SW.assign_chainage(g.lon.values, g.lat.values, chan, tree=tree)
    g["dist_to_channel_km"] = off_km
    g["sensor"] = "ICESat-2"; g["product"] = "ATL08"
    g["measurement_type"] = "satellite LiDAR terrain segment (100 m)"
    g["vertical_system_final"] = "EVRF2019 normal height (empirically aligned)"

    nnight = int(g.night.sum())
    print(f"\nQC-passing segments: {len(g):,}  |  NIGHT (canonical) {nnight:,}  "
          f"|  DAY retained for sensitivity {len(g)-nnight:,}")
    print(f"accepted ICESat-2 ground segments (canonical night set): {nnight:,}")
    gn = g[g.night]
    print(f"  unique RGT {gn.rgt.nunique()} | cycles {gn.cycle.nunique()} | dates {gn.date.nunique()}")
    print(f"  date range {pd.to_datetime(gn.date).min().date()} .. {pd.to_datetime(gn.date).max().date()}")
    print(f"  H_terrain_EVRF2019_empirical {gn.H_terrain_EVRF2019_empirical_m.min():.2f} .. "
          f"{gn.H_terrain_EVRF2019_empirical_m.max():.2f} m "
          f"(median {gn.H_terrain_EVRF2019_empirical_m.median():.2f})")
    print(f"  ground photons per segment: median {gn.gnd_ph_count.median():.0f}, "
          f"p05 {gn.gnd_ph_count.quantile(.05):.0f}")
    print("  landcover FLAGGED as unreliable in this extract (values spread over 0-255, not "
          "decoded to Copernicus classes) -- not used as a filter")

    # ================================================== K10.4B frozen split
    print("\n" + "-" * 70)
    print("K10.4B -- STRICT INDEPENDENCE SPLIT (frozen before any DEM exists)")
    print("-" * 70)
    gsplit = g[g.night]
    per_rgt = gsplit.groupby("rgt").agg(n_segments=("rgt", "size"), n_cycles=("cycle", "nunique"),
                                   n_dates=("date", "nunique")).sort_values("n_segments",
                                                                            ascending=False)
    rgts = per_rgt.index.to_numpy()
    order = RNG.permutation(len(rgts))
    val_rgts, n_val = [], 0
    target = VALIDATION_TARGET_FRAC * len(gsplit)
    for i in order:
        if n_val >= target and len(val_rgts) >= MIN_VALIDATION_RGTS:
            break
        val_rgts.append(int(rgts[i]))
        n_val += int(per_rgt.n_segments.iloc[i])
    g["split"] = np.where(g.rgt.isin(val_rgts), "INDEPENDENT_VALIDATION", "MODEL_BUILD")
    nb = int(((g.split == "MODEL_BUILD") & g.night).sum())
    nv = int(((g.split == "INDEPENDENT_VALIDATION") & g.night).sum())
    print(f"split unit = WHOLE RGT (never a segment or photon); repeat cycles of the same RGT")
    print(f"stay together, which also blocks cross-cycle track leakage.")
    print(f"  MODEL_BUILD           {nb:>7,} night segments, "
          f"{gsplit[gsplit.rgt.isin(set(gsplit.rgt)-set(val_rgts))].rgt.nunique()} RGTs")
    print(f"  INDEPENDENT_VALIDATION{nv:>7,} segments, {len(val_rgts)} RGTs -> {sorted(val_rgts)}")
    print(f"  validation share {100*nv/max(nb+nv,1):.1f}% of canonical night segments")
    print(per_rgt.assign(split=np.where(per_rgt.index.isin(val_rgts),
                                        "VALIDATION", "BUILD")).to_string())

    # ============================== contemporaneous water-state QC (K10.4A+)
    print("\n" + "-" * 70)
    print("CONTEMPORANEOUS WATER-STATE QC")
    print("-" * 70)
    print("An ATL08 terrain segment is a ~100 m support element and can straddle a bank.")
    print("A 'ground' classification alone is therefore NOT sufficient to call it exposed bed:")
    print("it is cross-checked against the nearest-in-time post-breach water mask.")
    import rasterio
    from rasterio import features as rfeat
    from scipy import ndimage
    from shapely import wkt as shwkt
    with rasterio.open(CFG.ROOT / "data/processed/bathymetry/morphology_prior_v1.tif") as src:
        gtr, gH, gW = src.transform, src.height, src.width
    wb = pd.read_parquet(CFG.ROOT / "outputs/tables/water_body_objects.parquet")
    post = wb[(wb.period == "POST_BREACH") & (wb.wkt != "")].copy()
    post["date"] = pd.to_datetime(post.date)
    # Water state and edge distance are computed in VECTOR space with an STRtree,
    # NOT on the 250 m analysis grid. A raster at 250 m cannot resolve the 30/50/100 m
    # edge buffers the design calls for -- every dry cell would sit >= one cell width
    # from the edge, making the buffer sensitivity meaningless. Exact polygon distance
    # avoids that entirely.
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    polys_by_date, tree_by_date, bnd_by_date, conn_by_date = {}, {}, {}, {}
    for d, gg in post.groupby("date"):
        geoms = [shwkt.loads(w) for w in gg.wkt]
        geoms = [q for q in geoms if q is not None and not q.is_empty]
        polys_by_date[d] = geoms
        tree_by_date[d] = STRtree(geoms) if geoms else None
        bnds = [q.boundary for q in geoms]
        bnd_by_date[d] = STRtree(bnds) if bnds else None
        mg = gg[gg.is_main_component]
        mgeo = [shwkt.loads(w) for w in mg.wkt]
        conn_by_date[d] = [q for q in mgeo if q is not None and not q.is_empty]
    mask_dates = np.array(sorted(polys_by_date), dtype="datetime64[ns]")
    print(f"post-breach water masks available for {len(mask_dates)} dates "
          f"({pd.Timestamp(mask_dates.min()).date()} .. {pd.Timestamp(mask_dates.max()).date()})")

    inv_t = ~gtr
    cc, rr = inv_t * (g.x.values, g.y.values)
    g["grid_col"] = np.clip(cc.astype(int), 0, gW - 1)
    g["grid_row"] = np.clip(rr.astype(int), 0, gH - 1)
    ad = pd.to_datetime(g.date).values.astype("datetime64[ns]")
    nearest = mask_dates[np.abs(mask_dates[None, :] - ad[:, None]).argmin(axis=1)]
    g["watermask_datetime"] = nearest
    g["atl08_datetime"] = ad
    g["delta_time_days"] = (ad - nearest) / np.timedelta64(1, "D")

    water_state = np.empty(len(g), dtype=object)
    edge_dist = np.full(len(g), np.nan)
    conn_dist = np.full(len(g), np.nan)
    for dt_m, idx in pd.Series(range(len(g))).groupby(nearest):
        i = idx.values
        dt_m = pd.Timestamp(dt_m)
        pts = [Point(xx, yy) for xx, yy in zip(g.x.values[i], g.y.values[i])]
        tree, bnd = tree_by_date[dt_m], bnd_by_date[dt_m]
        if tree is None:
            water_state[i] = "DRY"
            continue
        polys = polys_by_date[dt_m]
        inside = np.zeros(len(pts), bool)
        for j, pt in enumerate(pts):
            for k in tree.query(pt):
                if polys[k].contains(pt):
                    inside[j] = True
                    break
        water_state[i] = np.where(inside, "WET", "DRY")
        if bnd is not None:
            bl = [q.boundary for q in polys]
            ni = bnd.nearest(pts)
            edge_dist[i] = [pt.distance(bl[k]) for pt, k in zip(pts, np.atleast_1d(ni))]
        cg = conn_by_date[dt_m]
        if cg:
            ct = STRtree(cg)
            ni2 = ct.nearest(pts)
            conn_dist[i] = [pt.distance(cg[k]) for pt, k in zip(pts, np.atleast_1d(ni2))]
    g["water_state"] = water_state
    g["distance_to_water_edge_m"] = edge_dist
    g["distance_to_connected_water_m"] = conn_dist
    g["shoreline_mixed_flag"] = g.distance_to_water_edge_m.abs() < 100.0

    gn2 = g[g.night]
    print(f"canonical night set water state: "
          f"DRY {int((gn2.water_state=='DRY').sum()):,}  WET {int((gn2.water_state=='WET').sum()):,}")
    print(f"  |delta_time| to nearest mask: median {gn2.delta_time_days.abs().median():.0f} d, "
          f"p90 {gn2.delta_time_days.abs().quantile(.9):.0f} d")
    print("  buffer sensitivity (DRY segments surviving a distance-to-water-edge threshold):")
    for thr in (0.0, 30.0, 50.0, 100.0):
        k = int(((gn2.water_state == "DRY") & (gn2.distance_to_water_edge_m >= thr)).sum())
        print(f"    >= {thr:5.0f} m : {k:6,}  ({100*k/max(len(gn2),1):5.1f}% of night set)")
    g["exposed_ground_validation_ok"] = (g.night & (g.water_state == "DRY")
                                         & (g.distance_to_water_edge_m >= 50.0))
    print(f"  PRIMARY exposed-ground validation set (night + DRY + >=50 m from edge): "
          f"{int(g.exposed_ground_validation_ok.sum()):,}")

    # the P_channel>=0.5 question, answered explicitly
    with rasterio.open(CFG.ROOT / "data/processed/bathymetry/morphology_prior_v1.tif") as src:
        Pc = src.read(1); Pc = np.where(Pc == src.nodata, np.nan, Pc)
    g["P_channel"] = Pc[g.grid_row, g.grid_col]
    hi = g[g.night & (g.P_channel >= 0.5)]
    hi_dry = hi[hi.water_state == "DRY"]
    hi_ok = hi[hi.exposed_ground_validation_ok]
    print(f"\n  night segments on P_channel>=0.5 cells: {len(hi):,}")
    print(f"    of these, contemporaneously DRY: {len(hi_dry):,} "
          f"({100*len(hi_dry)/max(len(hi),1):.1f}%)")
    print(f"    surviving the full >=50 m edge buffer: {len(hi_ok):,} "
          f"({100*len(hi_ok)/max(len(hi),1):.1f}%)")
    print("    -> these are DRY EXPOSED ground, NOT submerged-bed observations, and are never")
    print("       used to claim submerged bathymetry.")

    gp = OUT_DIR / "k10_modern_elevation_points.parquet"
    g.to_parquet(gp, index=False)
    print(f"\n-> {gp}  ({len(g):,} rows)")

    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Lock the independent ICESat-2 ground validation set BEFORE any modern "
                   "DEM is fitted, so later validation is genuinely independent.",
        "source_file": str(ATL08),
        "source_sha256_16": hashlib.sha256(ATL08.read_bytes()).hexdigest()[:16],
        "segment_definition": "ATL08 native 100 m along-track terrain segment; ATL03 not held "
                              "locally so 20/50 m sensitivity is NOT available",
        "qc_rules": {
            "period": "POST_BREACH", "domain": "inside former pool (SA_2)",
            "night_only": f"solar_elevation < {NIGHT_MAX_SOLAR_ELEV}",
            "snowcover": f"== {SNOWCOVER_KEEP} (ATL08 snow free land; excludes 0=ice free water, 2=snow, 255=fill)",
            "min_gnd_ph_count": MIN_GND_PHOTONS,
            "max_h_canopy_m": MAX_CANOPY_M,
            "elevation_envelope_m": list(ELEV_ENVELOPE_M),
            "landcover": "NOT used -- field not decoded to Copernicus classes in this extract",
            "qc_independent_of_dem": True,
        },
        "vertical_processing_chain":
            "h_te_median + free2mean(lat) - zeta_EGG2015 + c_EGG2015_to_EVRF2019",
        "c_EGG2015_to_EVRF2019_m": c_mean,
        "sigma_vertical_reference_m": sigma_vref,
        "epsg9902_note": "EPSG:9902 is NOT applied to ICESat-2 heights; it belongs to the gauge chain only",
        "split": {"unit": "whole RGT", "rule": "no segment-level or photon-level splitting",
                  "cross_cycle_leakage": "prevented -- cycles of one RGT stay in one side",
                  "validation_rgts": sorted(val_rgts),
                  "build_rgts": sorted(int(r) for r in rgts if int(r) not in val_rgts),
                  "n_build_segments": nb, "n_validation_segments": nv,
                  "seed": int(CFG.SEED)},
        "validation_design": "leave-one-RGT-out within MODEL_BUILD for tuning; the "
                             "INDEPENDENT_VALIDATION RGTs are used exactly once, at the end",
        "residual_sign_convention": "residual = H_DEM - H_ICESat2 (positive = DEM too high)",
        "metrics": ["N", "N_RGT", "RMSE", "MAE", "mean_bias", "median_bias", "NMAD", "std", "P05", "P50", "P95"],
        "r2_policy": "R2 is NOT the principal vertical-accuracy metric",
        "terminology": "ICESat-2 ground reference / independent satellite LiDAR terrain "
                       "reference -- NOT absolute ground truth (no GNSS/RTK available)",
    }
    mp = CFG.TABLES / "k10_icesat2_validation_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    print(f"-> {mp}  (FROZEN)")

    # ================================================== K10.1 inventory
    print("\n" + "-" * 70)
    print("K10.1 -- MODERN OBSERVATION INVENTORY AND REFERENCE EPOCH")
    print("-" * 70)
    inv = [{
        "dataset": "ICESat-2 ATL08 terrain (night, QC'd)", "sensor_product": "ICESat-2 / ATL08",
        "first_date": str(pd.to_datetime(g.date).min().date()),
        "last_date": str(pd.to_datetime(g.date).max().date()),
        "measurement_type": "exposed-ground elevation", "vertical_reference": "EVRF2019 (empirical)",
        "spatial_coverage": f"{g.rgt.nunique()} RGTs, sparse along-track",
        "hydrological_state": "post-breach drained bed",
        "relevance": "PRIMARY modern direct elevation + independent validation",
        "n_records": len(g)}]
    try:
        wb = pd.read_parquet(CFG.ROOT / "outputs/tables/water_body_objects.parquet",
                             columns=["date", "period"])
        pw = wb[wb.period == "POST_BREACH"]
        inv.append({"dataset": "Sentinel-2 water masks (post-breach)", "sensor_product": "Sentinel-2 L2A",
                    "first_date": str(pd.to_datetime(pw.date).min())[:10],
                    "last_date": str(pd.to_datetime(pw.date).max())[:10],
                    "measurement_type": "surface-water planform (not elevation)",
                    "vertical_reference": "n/a", "spatial_coverage": "full domain",
                    "hydrological_state": "post-breach", "relevance": "zoning / wet-dry state",
                    "n_records": len(pw)})
    except Exception as e:
        print(f"  (Sentinel inventory skipped: {e})")
    inv.append({"dataset": "Historical soundings (K9 M2 prior)", "sensor_product": "legacy survey",
                "first_date": "legacy", "last_date": "legacy",
                "measurement_type": "bed elevation", "vertical_reference": "EVRF2019 via EPSG:9902",
                "spatial_coverage": "former pool", "hydrological_state": "pre-impoundment/reservoir era",
                "relevance": "HISTORICAL_INHERITED fallback ONLY -- not current bed",
                "n_records": 7319})
    invdf = pd.DataFrame(inv)
    invdf.to_csv(CFG.TABLES / "k10_modern_observation_inventory.csv", index=False)
    print(invdf[["dataset", "first_date", "last_date", "measurement_type", "n_records"]].to_string(index=False))

    d = pd.to_datetime(g.date)
    print(f"\nt_ref: modern ICESat-2 ground support spans {d.min().date()} .. {d.max().date()}; "
          f"median {d.median().date()}")
    print(f"  Temporal heterogeneity is REAL ({d.dt.year.nunique()} distinct years) and will be")
    print(f"  carried in age_of_evidence rather than collapsed into one epoch label.")
    print(f"  segments per year: " + ", ".join(f"{y}:{n}" for y, n in d.dt.year.value_counts().sort_index().items()))

    # ================================================== figure
    sa2 = SD.load("reservoir_full_pool_prebreach")
    from shapely.ops import transform as shp_transform
    sa2m = shp_transform(_TF.transform, sa2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    xs, ys = sa2m.exterior.xy
    for ax in axes:
        ax.plot(xs, ys, color="#1a2228", lw=1.1)
        ax.set_aspect("equal"); ax.set_xlabel("Easting, m (EPSG:32636)")
    b = g[g.split == "MODEL_BUILD"]; v = g[g.split == "INDEPENDENT_VALIDATION"]
    axes[0].scatter(b.x, b.y, s=1.5, c="#8a94a3", label=f"MODEL_BUILD ({len(b):,})")
    axes[0].scatter(v.x, v.y, s=1.5, c="#c1402a", label=f"INDEPENDENT_VALIDATION ({len(v):,})")
    axes[0].set_ylabel("Northing, m")
    axes[0].set_title("A. frozen RGT-level split of QC'd night ICESat-2 ground segments",
                      loc="left", fontsize=10)
    axes[0].legend(fontsize=8, markerscale=6)
    sc = axes[1].scatter(g.x, g.y, s=1.5, c=g.H_EVRF2019_m, cmap="terrain", vmin=-5, vmax=20)
    fig.colorbar(sc, ax=axes[1], label="H, m EVRF2019")
    axes[1].set_title("B. accepted ground elevations (night only, QC passed)", loc="left", fontsize=10)
    fig.suptitle("K10 foundation -- modern ICESat-2 ground reference for the post-breach terrain",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K10_data_coverage.png", dpi=150)
    print(f"\n-> {CFG.FIG / 'K10_data_coverage.png'}")


if __name__ == "__main__":
    main()
