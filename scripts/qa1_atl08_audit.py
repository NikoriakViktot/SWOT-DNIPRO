#!/usr/bin/env python
"""QA STAGE 1 — audit of what the ICESat-2 heights in dH actually are.

Assumes nothing. Establishes the provenance of the height, inventories the
quality fields that genuinely exist, builds a subset ladder S0-S6 of increasing
strictness, and re-fits the model on every rung with the bootstrap clustered by
track.

SIGN, everywhere:
    dH = H_ICESat2 - H_survey
    dH < 0  ->  the ICESat-2 surface is LOWER than the old bathymetry
    dH > 0  ->  the ICESat-2 surface is HIGHER
No causal word (erosion / sedimentation) is used until the QA concludes.

Outputs
-------
outputs/tables/qa1_field_provenance.csv
outputs/tables/qa1_subset_ladder.csv
outputs/tables/qa1_model_by_subset.csv
outputs/tables/qa1_temporal.csv
outputs/reports/qa1_atl08_audit.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

RNG = np.random.default_rng(CFG.SEED)
SIGN = "dH = H_ICESat2 - H_survey;  dH<0 = ICESat-2 surface LOWER than old bathymetry"

# What the pipeline actually calls. Established by reading the code, not assumed.
PROVENANCE = [
    dict(field="h_te_median", source="SlideRule atl08p (PhoREAL)",
         meaning="median TERRAIN (ground) height of the 100 m segment",
         vertical_reference="WGS84 ellipsoid (ATL03 photon heights, tide-free)",
         used="YES - this is H_ATL08 in dH"),
    dict(field="h_canopy", source="SlideRule atl08p (PhoREAL)",
         meaning="canopy height ABOVE the fitted terrain (relative, not absolute)",
         vertical_reference="relative to h_te_median", used="no - QA only"),
    dict(field="h_max_canopy / h_mean_canopy / h_min_canopy", source="SlideRule atl08p",
         meaning="canopy height statistics above terrain",
         vertical_reference="relative to h_te_median", used="no - QA only"),
    dict(field="gnd_ph_count", source="SlideRule atl08p",
         meaning="photons classified GROUND in the segment",
         vertical_reference="-", used="no - QA only (terrain support)"),
    dict(field="veg_ph_count", source="SlideRule atl08p",
         meaning="photons classified VEGETATION in the segment",
         vertical_reference="-", used="no - QA only (contamination)"),
    dict(field="ph_count", source="SlideRule atl08p", meaning="total photons",
         vertical_reference="-", used="no - QA only"),
    dict(field="canopy_openness", source="SlideRule atl08p",
         meaning="stdev of canopy photon heights", vertical_reference="-",
         used="no - QA only"),
    dict(field="landcover / snowcover", source="SlideRule atl08p (ATL08 ancillary)",
         meaning="Copernicus land cover / snow flag", vertical_reference="-",
         used="no - QA only"),
    dict(field="solar_elevation", source="SlideRule atl08p",
         meaning="solar elevation -> day/night", vertical_reference="-",
         used="no - QA only"),
    dict(field="spot / gt / rgt / cycle / segment_id", source="SlideRule atl08p",
         meaning="beam and track identity", vertical_reference="-",
         used="clustering only"),
]
NOT_AVAILABLE = [
    "terrain uncertainty (h_te_uncertainty)", "terrain/canopy quality scores",
    "ATL08 v6 subset/fit quality flags", "urban flag",
    "cloud / atmospheric quality flags", "h_te_best_fit / h_te_interp / h_te_max",
]


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def cluster_boot(m, cols, n=3000):
    if m.track.nunique() < 8 or len(m) < 40:
        return None
    X = np.c_[np.ones(len(m)), m[cols].values]
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
    pd.DataFrame(PROVENANCE).to_csv(CFG.TABLES / "qa1_field_provenance.csv", index=False)

    # ---- rebuild pairs carrying every QA field -----------------------------
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet")
    at = pd.read_parquet(ROOT / "data/processed/atl08/kakhovka_atl08_terrain.parquet")
    raw = pd.read_parquet(ROOT / "data/processed/atl08/kakhovka_atl08_raw.parquet")
    # QA fields live in the raw return; join on the identity keys
    keys = ["rgt", "cycle", "gt", "segment_id"]
    qa = raw[keys + ["gnd_ph_count", "veg_ph_count", "ph_count", "h_canopy",
                     "canopy_openness", "landcover", "snowcover", "solar_elevation",
                     "spot"]].drop_duplicates(keys)
    at = at.drop(columns=[c for c in qa.columns if c in at.columns and c not in keys])
    at = at.merge(qa, on=keys, how="left")
    bed = at[at.surface_class == "exposed_bed"].copy()
    print(f"exposed-bed segments with QA fields: {len(bed):,} "
          f"({bed.gnd_ph_count.notna().mean()*100:.1f} % matched)")

    import pyproj
    from scipy.spatial import cKDTree
    TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)
    bx, by = TO_M.transform(bed.lon.values, bed.lat.values)
    tree = cKDTree(np.c_[bx, by])
    dist, idx = tree.query(np.c_[sd.x.values, sd.y.values], k=1,
                           distance_upper_bound=250.0)
    ok = np.isfinite(dist)
    b = bed.iloc[idx[ok]].reset_index(drop=True)
    m = pd.DataFrame({
        "lon": sd.lon.values[ok], "lat": sd.lat.values[ok],
        "x": sd.x.values[ok], "y": sd.y.values[ok],
        "H_survey": sd.H_bed_evrf2019_m.values[ok], "depth_m": sd.depth_m.values[ok],
        "H_icesat2": b.H_terrain_common_m.values, "match_dist_m": dist[ok]})
    for c in ["gnd_ph_count", "veg_ph_count", "ph_count", "h_canopy",
              "canopy_openness", "landcover", "snowcover", "solar_elevation",
              "spot", "rgt", "date"]:
        m[c] = b[c].values
    m["date"] = pd.to_datetime(m.date)
    m["year"] = m.date.dt.year
    m["track"] = m.rgt.astype(str) + "_" + m.date.dt.strftime("%Y%m%d")
    m["dH"] = m.H_icesat2 - m.H_survey
    m["veg_frac"] = m.veg_ph_count / m.ph_count.replace(0, np.nan)

    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    cx, cy = TO_M.transform(ch.lon.values, ch.lat.values)
    d2, i2 = cKDTree(np.c_[cx, cy]).query(np.c_[m.x.values, m.y.values], k=1)
    m["dist_thalweg_km"] = d2 / 1000.0
    m["chainage_km"] = ch.chain_km.values[i2]
    m = m[(m.H_survey - 5.20) > 1.0].dropna(subset=["dH", "gnd_ph_count"])

    print(f"\n=== QA field distributions over {len(m):,} pairs ===")
    for c in ["gnd_ph_count", "veg_ph_count", "veg_frac", "h_canopy", "canopy_openness"]:
        q = m[c].quantile([0, .25, .5, .75, .95, 1]).round(3).tolist()
        print(f"  {c:16s} min/q25/med/q75/p95/max = {q}")
    print(f"  landcover classes present: {sorted(m.landcover.dropna().unique().tolist())}")
    print(f"  h_canopy == 0 : {100*(m.h_canopy==0).mean():.1f} %   "
          f"veg_ph_count == 0 : {100*(m.veg_ph_count==0).mean():.1f} %")

    # ---- subset ladder ------------------------------------------------------
    S = {}
    S["S0 all pairs"] = m
    S["S1 terrain solution exists"] = m[m.gnd_ph_count > 0]
    S["S2 + >=20 ground photons"] = S["S1 terrain solution exists"][
        S["S1 terrain solution exists"].gnd_ph_count >= 20]
    S["S3 + canopy < 1 m"] = S["S2 + >=20 ground photons"][
        S["S2 + >=20 ground photons"].h_canopy < 1.0]
    S["S4 + veg photon frac < 5%"] = S["S3 + canopy < 1 m"][
        S["S3 + canopy < 1 m"].veg_frac.fillna(0) < 0.05]
    S["S5 + sounding within 50 m"] = S["S4 + veg photon frac < 5%"][
        S["S4 + veg photon frac < 5%"].match_dist_m <= 50]
    S["S6 STRICT bare ground"] = S["S5 + sounding within 50 m"][
        (S["S5 + sounding within 50 m"].h_canopy == 0)
        & (S["S5 + sounding within 50 m"].veg_ph_count == 0)
        & (S["S5 + sounding within 50 m"].gnd_ph_count >= 50)]

    rows = []
    print(f"\n=== subset ladder   ({SIGN}) ===")
    print(f"{'subset':<30}{'N':>7}{'tracks':>8}{'median':>9}{'NMAD':>8}"
          f"{'q05':>8}{'q95':>8}")
    for k, v in S.items():
        if not len(v):
            continue
        rows.append({"subset": k, "n": len(v), "tracks": v.track.nunique(),
                     "median_dH": v.dH.median(), "mean_dH": v.dH.mean(),
                     "nmad": nmad(v.dH.values), "rmse": float(np.sqrt((v.dH**2).mean())),
                     "q05": v.dH.quantile(.05), "q25": v.dH.quantile(.25),
                     "q75": v.dH.quantile(.75), "q95": v.dH.quantile(.95),
                     "chainage_min": v.chainage_km.min(), "chainage_max": v.chainage_km.max()})
        print(f"{k:<30}{len(v):>7,}{v.track.nunique():>8}{v.dH.median():>+9.2f}"
              f"{nmad(v.dH.values):>8.2f}{v.dH.quantile(.05):>+8.2f}{v.dH.quantile(.95):>+8.2f}")
    pd.DataFrame(rows).to_csv(CFG.TABLES / "qa1_subset_ladder.csv", index=False)

    # ---- model on every rung ------------------------------------------------
    # Centre the predictors on the S0 medians so the intercept is the fitted dH
    # AT A TYPICAL POINT, not an extrapolation to H_survey = 0 (which lies far
    # outside the data and made the earlier intercept CI meaningless).
    cols = ["dist_thalweg_km", "chainage_km", "H_survey"]
    ctr = {c: float(m[c].median()) for c in cols}
    print(f"\n  predictors centred at: " +
          ", ".join(f"{c}={v:.2f}" for c, v in ctr.items()))
    # Centre on COPIES. Doing this in place mutated the shared frames -- including
    # `m` itself, which was then written to parquet with a shifted H_survey and
    # silently corrupted every downstream user of that file.
    Sc = {}
    for k, v in S.items():
        v = v.copy()
        for c in cols:
            v[c] = v[c] - ctr[c]
        Sc[k] = v
    mres = []
    print(f"\n=== model per subset, bootstrap by TRACK, predictors CENTRED "
          f"(C = fitted dH at a typical point) ===")
    print(f"{'subset':<30}{'N':>6}{'trk':>5}{'C':>8}{'CI(C)':>20}"
          f"{'H_survey':>10}{'CI':>18}")
    for k, v in Sc.items():
        r = cluster_boot(v, cols)
        if r is None:
            print(f"{k:<30}{len(v):>6,}{v.track.nunique():>5}   too few tracks/points")
            continue
        beta, lo, hi, ntr = r
        j = cols.index("H_survey") + 1
        print(f"{k:<30}{len(v):>6,}{ntr:>5}{beta[0]:>+8.2f}"
              f"  [{lo[0]:+.2f},{hi[0]:+.2f}]{beta[j]:>+10.3f}  [{lo[j]:+.3f},{hi[j]:+.3f}]")
        d = {"subset": k, "n": len(v), "tracks": ntr, "C": beta[0],
             "C_lo": lo[0], "C_hi": hi[0]}
        for c, b_, l_, h_ in zip(cols, beta[1:], lo[1:], hi[1:]):
            d[c] = b_; d[c + "_lo"] = l_; d[c + "_hi"] = h_
            d[c + "_excl0"] = bool(l_ * h_ > 0)
        mres.append(d)
    pd.DataFrame(mres).to_csv(CFG.TABLES / "qa1_model_by_subset.csv", index=False)

    # ---- temporal -----------------------------------------------------------
    strict = S["S6 STRICT bare ground"] if len(S["S6 STRICT bare ground"]) > 40 \
        else S["S4 + veg photon frac < 5%"]   # uncentred originals
    t = m.groupby("year").agg(n=("dH", "size"), tracks=("track", "nunique"),
                              median=("dH", "median"), nmad=("dH", nmad)).round(2)
    t2 = strict.groupby("year").agg(n=("dH", "size"), median=("dH", "median")).round(2)
    print(f"\n=== temporal: does dH drift with time since drainage? ===")
    print("  all pairs:"); print(t.to_string())
    print("  strictest usable subset:"); print(t2.to_string())
    t.to_csv(CFG.TABLES / "qa1_temporal.csv")

    # DATUM LABEL, added after the reference level was corrected. H_survey and

    # dH in this file are on the OLD 16.00 m (NPG) reference, which is 2.00 m

    # above the resolved UNS level. The file is kept unchanged as the QA audit

    # trail; consumers must shift by (16.00 - reference) or use H_icesat2 only.

    m['reference_level_bs_m'] = 16.00

    m['reference_level_note'] = ('OLD NPG reference, superseded by UNS 14.00 m '

                                 '-- see hist2/hist3. dH here is 2.00 m low.')


    m.to_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/qa1_pairs_with_qa.parquet", index=False)
    print(f"\n-> {CFG.TABLES/'qa1_subset_ladder.csv'}")
    print(f"-> {CFG.TABLES/'qa1_model_by_subset.csv'}")
    print("\nFields NOT available from SlideRule PhoREAL (so NOT used):")
    for f in NOT_AVAILABLE:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
