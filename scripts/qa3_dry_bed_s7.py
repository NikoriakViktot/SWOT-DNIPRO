#!/usr/bin/env python
"""QA STAGE 3 — S7: confirmed post-breach EXPOSED FORMER RESERVOIR BED.

S6 was strict BARE GROUND by photon evidence. That is not the same thing as
strict FORMER RESERVOIR BED: a clean canopy-free ground return can equally be a
bank, an island, or a spot that was still under water on the day ICESat-2 flew
over. `date > 2023-06-06` does not fix this either, because the pool drained
gradually over months.

So every candidate observation is checked against a DYNAMIC water mask taken
from the Sentinel-2 scene nearest in time to that ICESat-2 pass, and classified:

    DRY_EXPOSED_BED        not water, and >= SHORE_BUF_M from any water pixel
    WATER                  water on that date
    SHORELINE_AMBIGUOUS    dry but within SHORE_BUF_M of water
    OUTSIDE_FORMER_POOL    outside the data-driven former-pool footprint
    UNKNOWN                no Sentinel-2 mask within MAX_LAG_D days

    S7 = S6 AND post-breach AND DRY_EXPOSED_BED

Outputs
-------
outputs/tables/qa3_s7_classification.csv
outputs/tables/qa3_coverage.csv
outputs/tables/qa3_model_s0_s6_s7.csv
data/processed/bathymetry/qa3_analysis_zone.gpkg
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from affine import Affine
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

MASKS = ROOT / "data/processed/water_masks"
BREACH = pd.Timestamp("2023-06-06")
SHORE_BUF_M = 100.0        # confidently dry must be this far from any water pixel
MAX_LAG_D = 20             # a water mask further than this in time proves nothing
PIX = 20.0
RNG = np.random.default_rng(CFG.SEED)
TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def cluster_boot(m, cols, n=3000):
    if m.track.nunique() < 8 or len(m) < 40:
        return None
    ctr = {c: float(m[c].median()) for c in cols}
    X = np.c_[np.ones(len(m)), (m[cols] - pd.Series(ctr)).values]
    y = m.dH.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    uniq = m.track.unique()
    grp = {t: np.where(m.track.values == t)[0] for t in uniq}
    out = []
    for _ in range(n):
        idx = np.concatenate([grp[t] for t in RNG.choice(uniq, len(uniq), True)])
        try:
            out.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        except Exception:
            pass
    B = np.array(out)
    return beta, np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0), len(uniq)


def main() -> None:
    m = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/qa1_pairs_with_qa.parquet")
    m["date"] = pd.to_datetime(m.date)
    m["x"], m["y"] = TO_M.transform(m.lon.values, m.lat.values)
    print(f"{len(m):,} candidate post-breach ICESat-2 terrain observations "
          f"(NOT yet 'exposed bed')")

    # ---- dynamic water mask, nearest Sentinel-2 scene in time ---------------
    wm = pd.read_csv(CFG.TABLES / "water_mask_summary.csv")
    wm["dt"] = pd.to_datetime(wm.sensing_time, format="%Y%m%dT%H%M%S")
    wm = wm[wm.dt >= BREACH]
    scenes = {}
    for r in wm.itertuples():
        p = MASKS / f"{r.name}.npz"
        if p.exists():
            scenes.setdefault(r.dt.normalize(), []).append(p)
    sdates = pd.DatetimeIndex(sorted(scenes))
    print(f"{len(sdates)} post-breach Sentinel-2 water-mask dates available")

    cls = np.full(len(m), "UNKNOWN", dtype=object)
    lag = np.full(len(m), np.nan)
    cache = {}
    for d in sorted(m.date.unique()):
        d = pd.Timestamp(d)
        sel = np.where(m.date.values == np.datetime64(d))[0]
        if not len(sdates):
            continue
        k = int(np.argmin(np.abs((sdates - d).days)))
        lg = abs((sdates[k] - d).days)
        lag[sel] = lg
        if lg > MAX_LAG_D:
            continue
        key = sdates[k]
        if key not in cache:
            layers = []
            for p in scenes[key]:
                z = np.load(p)
                a = z["affine"]
                tr = Affine(a[0], a[1], a[2], a[3], a[4], a[5])
                water = z["mask"]
                # distance (in pixels) from every cell to the nearest water cell
                dist = ndimage.distance_transform_edt(~water) * PIX
                layers.append((tr, str(z["crs"]), water, dist))
            cache[key] = layers
        for i in sel:
            got = False
            for tr, crs, water, dist in cache[key]:
                tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                px, py = tf.transform(m.lon.values[i], m.lat.values[i])
                c, r_ = ~tr * (px, py)
                c, r_ = int(c), int(r_)
                if 0 <= r_ < water.shape[0] and 0 <= c < water.shape[1]:
                    if water[r_, c]:
                        cls[i] = "WATER"
                    elif dist[r_, c] >= SHORE_BUF_M:
                        cls[i] = "DRY_EXPOSED_BED"
                    else:
                        cls[i] = "SHORELINE_AMBIGUOUS"
                    got = True
                    break
            if not got:
                cls[i] = "UNKNOWN"
    m["dry_class"] = cls
    m["watermask_lag_d"] = lag

    print("\n=== dynamic water-mask classification of the candidates ===")
    print(m.dry_class.value_counts().to_string())
    print(f"\nby year:")
    print(pd.crosstab(m.date.dt.year, m.dry_class).to_string())

    # ---- S6 and S7 ---------------------------------------------------------
    s6mask = ((m.h_canopy == 0) & (m.veg_ph_count == 0) & (m.gnd_ph_count >= 50)
              & (m.match_dist_m <= 50))
    S = {"S0 candidates": m,
         "S6 strict bare ground": m[s6mask],
         "S7 CONFIRMED exposed bed": m[s6mask & (m.dry_class == "DRY_EXPOSED_BED")
                                       & (m.date > BREACH)]}
    m.to_csv(CFG.TABLES / "qa3_s7_classification.csv", index=False)

    cols = ["dist_thalweg_km", "chainage_km", "H_survey"]
    rows = []
    print(f"\n=== the headline result on each dataset ===")
    print(f"{'dataset':<28}{'N':>6}{'trk':>5}{'median dH':>11}{'NMAD':>7}"
          f"{'C':>8}{'CI(C)':>18}")
    for k, v in S.items():
        r = cluster_boot(v, cols)
        C, lo, hi, ntr = (r[0][0], r[1][0], r[2][0], r[3]) if r else (np.nan,) * 4
        print(f"{k:<28}{len(v):>6,}{v.track.nunique():>5}{v.dH.median():>+11.2f}"
              f"{nmad(v.dH.values):>7.2f}{C:>+8.2f}  [{lo:+.2f},{hi:+.2f}]")
        rows.append({"dataset": k, "n": len(v), "tracks": v.track.nunique(),
                     "median_dH": v.dH.median(), "nmad": nmad(v.dH.values),
                     "C": C, "C_lo": lo, "C_hi": hi,
                     "date_min": str(v.date.min().date()), "date_max": str(v.date.max().date()),
                     "chainage_min": v.chainage_km.min(), "chainage_max": v.chainage_km.max()})
    pd.DataFrame(rows).to_csv(CFG.TABLES / "qa3_model_s0_s6_s7.csv", index=False)

    s7 = S["S7 CONFIRMED exposed bed"]
    print(f"\n=== what does S7 actually represent? ===")
    print(f"  {len(s7):,} confirmed observations, {s7.track.nunique()} independent tracks")
    print(f"  dates {s7.date.min():%Y-%m-%d} .. {s7.date.max():%Y-%m-%d}")
    print(f"  by year: {dict(s7.date.dt.year.value_counts().sort_index())}")
    print(f"  chainage {s7.chainage_km.min():.0f}-{s7.chainage_km.max():.0f} km "
          f"(reservoir spans 0-240 km)")
    print(f"  bed elevation {s7.H_survey.min():.1f}..{s7.H_survey.max():.1f} m")

    # ---- analysis zone + coverage -----------------------------------------
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp_m = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
            SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    # The support zone is built from S7 ONLY -- the confirmed exposed-bed subset --
    # never from all of `m`, which would claim support wherever any observation fell.
    if s7.empty:
        raise SystemExit("S7 CONFIRMED exposed bed is empty: no analysis zone to build")
    pts = gpd.GeoSeries(gpd.points_from_xy(s7.x.values, s7.y.values), crs=CFG.CRS_METRIC)
    zone = unary_union(pts.buffer(1000).values).intersection(fp_m)
    gpd.GeoDataFrame({"name": ["S7 analysis zone"]}, geometry=[zone],
                     crs=CFG.CRS_METRIC).to_file(
        ROOT / "data/processed/bathymetry/qa3_analysis_zone.gpkg", driver="GPKG")

    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet")
    bath = unary_union(gpd.GeoSeries(gpd.points_from_xy(sd.x, sd.y),
                                     crs=CFG.CRS_METRIC).buffer(500).values)
    cov = [
        ("former reservoir footprint", fp_m.area / 1e6),
        ("area covered by the old bathymetry", bath.area / 1e6),
        ("bathymetry INSIDE the former pool", bath.intersection(fp_m).area / 1e6),
        ("S7 analysis zone (1 km around confirmed points)", zone.area / 1e6),
    ]
    print(f"\n=== spatial coverage ===")
    for k, v in cov:
        print(f"  {k:<50}{v:>10,.0f} km2")
    frac = zone.area / fp_m.area
    print(f"\n  coverage = analysis zone / former reservoir = {frac*100:.1f} %")
    print(f"  -> the result describes THIS zone, not the whole reservoir, unless "
          f"the zone spans it.")
    pd.DataFrame(cov, columns=["parameter", "km2"]).assign(
        coverage_fraction=[np.nan, np.nan, np.nan, frac]).to_csv(
        CFG.TABLES / "qa3_coverage.csv", index=False)


if __name__ == "__main__":
    main()
