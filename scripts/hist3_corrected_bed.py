#!/usr/bin/env python
"""HISTORICAL 3 — rebuild the bed elevations on the RESOLVED sounding datum,
and carry that datum into EVRF2019.

hist2 established that the S-57 depths are reduced to the reservoir's navigation
drawdown level (UNS), not to the normal impoundment level:

    published design level      UNS   14.00 m   (Table 19 notes)
    capacity curve + soundings        13.71 m   [13.57, 13.84], no ICESat-2
    ICESat-2 exposed bed (QA5)        14.11 m   [13.91, 14.32]
    previously assumed          NPG   16.00 m

The pipeline still carries 16.00 m, so every bed elevation derived from it is
2.00 m too high. This script rebuilds them on 14.00 m -- the published value,
chosen over the two fitted ones precisely because it is documented rather than
estimated -- and reports the sensitivity across all three.

THE DATUM ITSELF NEEDS CONVERTING. 14.00 m is a level in the historical Baltic
system; everything else in this study is EVRF2019. It goes through the same
EPSG:9902 grid as the gauges and is NOT a single constant: the offset varies
across the reservoir, so it is applied per sounding, exactly as Part 1 does for
gauge zeros.

    H_bed_EVRF2019(x) = (14.00 + delta_EPSG9902(x)) - DEPTH

CAVEAT, stated because it is the weakest link: the source does not name its
vertical system. EPSG:9902 is defined for BS-77 -> EVRF2019. If these design
tables are on BS-42 instead, there is a further step of a few centimetres that
is not applied here. That is small against the 2.00 m being corrected, and
against the 0.50 m scatter, but it is not zero.

SUPERSEDED, and left in place only as a record of the transition. When this was
written part11 still assumed 16.00 m, so writing a separate corrected file was
the only way to avoid destroying the pre-correction audit trail. part11 has
since been rebuilt on 14.00 m (see its header), so
`kakhovka_soundings_evrf2019.parquet` is now itself on the UNS reference and is
the AUTHORITATIVE file -- it carries the fuller metadata (H_ref_bs_m,
H_ref_evrf2019_m, reference_level_label, datum_provenance).

Do not read `kakhovka_soundings_evrf2019_uns14.parquet` for new work: it is a
duplicate of the authoritative file's bed elevations under a different label.
Nothing outside this script reads it.

Outputs
-------
data/processed/bathymetry/kakhovka_soundings_evrf2019_uns14.parquet
outputs/tables/hist3_datum_in_evrf2019.csv
outputs/tables/hist3_corrected_dH.csv
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

OLD_DATUM_BS = 16.00                       # NPG, what the pipeline assumed
NEW_DATUM_BS = 14.00                       # UNS, published navigation drawdown
ALT = {"UNS published (Table 19)": 14.00,
       "capacity-curve fit (hist2)": 13.71,
       "ICESat-2 fit (QA5 S7 pool)": 14.11,
       "previously assumed NPG": 16.00}
RNG = np.random.default_rng(CFG.SEED)
SRC = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
DST = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019_uns14.parquet"


def nmad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def boot_median(m, col, n=6000):
    u = m.track.unique()
    g = {t: m[col].values[m.track.values == t] for t in u}
    b = np.array([np.median(np.concatenate([g[t] for t in RNG.choice(u, len(u), True)]))
                  for _ in range(n)])
    return float(np.median(m[col])), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5)), len(u)


def main() -> None:
    sd = pd.read_parquet(SRC)

    # SUPERSESSION GUARD. This script exists to shift a 16.00 m surface down to
    # 14.00 m. part11 now produces 14.00 m directly, so running it again would
    # either crash on the renamed columns or, worse, apply a second -2.00 m
    # shift. Refuse, and say why, rather than fail obscurely.
    ref = sd["H_ref_bs_m"].iloc[0] if "H_ref_bs_m" in sd else None
    if ref is not None and abs(float(ref) - NEW_DATUM_BS) < 0.01:
        print("=" * 72)
        print("SUPERSEDED -- NOTHING TO DO")
        print("=" * 72)
        print(f"  {SRC.name} already carries the {NEW_DATUM_BS:.2f} m UNS reference")
        print(f"  (H_ref_bs_m = {float(ref):.2f}), because part11 has been rebuilt on it.")
        print(f"  That file is now AUTHORITATIVE and carries the fuller metadata.")
        print(f"\n  This script is kept as the record of the transition. Its output,")
        print(f"  {DST.name},")
        print(f"  is a redundant duplicate; nothing outside this script reads it.")
        print(f"\n  Re-running the shift would subtract another "
              f"{NEW_DATUM_BS-OLD_DATUM_BS:+.2f} m and be wrong.")
        print(f"  For the datum evidence see hist2_datum_closure.py; for the")
        print(f"  reference in EVRF2019 see outputs/tables/hist3_datum_in_evrf2019.csv.")
        return

    # ---- the datum, carried into EVRF2019 ---------------------------------
    # delta_EPSG9902 was already sampled per sounding when the file was built,
    # so the same grid and the same interpolation are reused -- no second
    # implementation to disagree with the first.
    d = sd.delta_epsg9902_m.values
    print("=" * 72)
    print("THE SOUNDING DATUM IN EVRF2019")
    print("=" * 72)
    print(f"  {NEW_DATUM_BS:.2f} m historical Baltic (UNS, navigation drawdown)")
    print(f"  + EPSG:9902 offset, sampled per sounding, not a constant:")
    print(f"      min {d.min():+.4f}  median {np.median(d):+.4f}  max {d.max():+.4f} m"
          f"   (range {1e3*(d.max()-d.min()):.0f} mm across the reservoir)")
    ev = NEW_DATUM_BS + d
    print(f"\n  -> sounding datum in EVRF2019: {ev.min():.3f} .. {ev.max():.3f} m, "
          f"median {np.median(ev):.3f} m")
    print(f"\n  For quoting a single number: {np.median(ev):.2f} m EVRF2019.")
    print(f"  For computing: use the per-point value. The {1e3*(d.max()-d.min()):.0f} mm spread is small")
    print(f"  but it is the same spread that makes a single national constant")
    print(f"  wrong for the gauges, so it is not dropped here either.")

    rows = []
    for name, D in ALT.items():
        e = D + d
        rows.append({"datum_name": name, "datum_bs_m": D,
                     "datum_evrf2019_median_m": float(np.median(e)),
                     "datum_evrf2019_min_m": float(e.min()),
                     "datum_evrf2019_max_m": float(e.max()),
                     "delta_epsg9902_median_m": float(np.median(d))})
    pd.DataFrame(rows).to_csv(CFG.TABLES / "hist3_datum_in_evrf2019.csv", index=False)
    print(f"\n  {'datum':<30}{'BS':>8}{'EVRF2019 (median)':>20}")
    for r in rows:
        print(f"  {r['datum_name']:<30}{r['datum_bs_m']:>8.2f}"
              f"{r['datum_evrf2019_median_m']:>20.3f}")

    # ---- rebuild the bed ---------------------------------------------------
    out = sd.copy()
    out["sounding_datum_bs_m"] = NEW_DATUM_BS
    out["sounding_datum_evrf2019_m"] = ev
    out["H_bed_bs77_m"] = NEW_DATUM_BS - out.depth_m
    out["H_bed_evrf2019_m"] = out.sounding_datum_evrf2019_m - out.depth_m
    out["raw_vertical_frame"] = (f"depth below {NEW_DATUM_BS:.2f} m UNS "
                                 f"(navigation drawdown), historical Baltic")
    out["datum_provenance"] = ("published design level, Table 19; corroborated by "
                               "capacity-curve fit 13.71 m and ICESat-2 14.11 m")
    out = out.drop(columns=["normal_pool_bs77_m"])
    shift = out.H_bed_evrf2019_m - sd.H_bed_evrf2019_m
    assert np.allclose(shift, NEW_DATUM_BS - OLD_DATUM_BS), "bed shift is not uniform"
    out.to_parquet(DST, index=False)
    print(f"\n  every bed elevation moves {NEW_DATUM_BS-OLD_DATUM_BS:+.2f} m")
    print(f"  -> {DST.name}   ({len(out):,} soundings)")
    print(f"  {SRC.name} is left unchanged.")

    # ---- what it does to the comparison ------------------------------------
    print("\n" + "=" * 72)
    print("THE BATHYMETRY-ICESat-2 COMPARISON ON THE CORRECTED DATUM")
    print("=" * 72)
    cls = pd.read_csv(CFG.TABLES / "qa3_s7_classification.csv", parse_dates=["date"])
    s6 = cls[(cls.h_canopy == 0) & (cls.veg_ph_count == 0)
             & (cls.gnd_ph_count >= 50) & (cls.match_dist_m <= 50)]
    s7 = s6[(s6.dry_class == "DRY_EXPOSED_BED") & (s6.date > "2023-06-06")
            & (s6.chainage_km <= 200)].copy()

    res = []
    print(f"\n  S7, impounded reach: {len(s7)} points, {s7.track.nunique()} tracks, "
          f"NMAD {nmad(s7.dH.values):.2f} m\n")
    print(f"  {'datum applied':<30}{'dH':>8}{'95% CI (track)':>20}{'note':>14}")
    for name, D in ALT.items():
        s7["dHc"] = s7.dH + (OLD_DATUM_BS - D)
        med, lo, hi, ntr = boot_median(s7, "dHc")
        circ = "circular" if "ICESat-2" in name else ""
        print(f"  {name:<30}{med:>+8.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{circ:>14}")
        res.append({"datum_name": name, "datum_bs_m": D, "n": len(s7),
                    "tracks": ntr, "median_dH_m": med, "ci_lo": lo, "ci_hi": hi,
                    "nmad_m": nmad(s7.dHc.values),
                    "independent_of_icesat2": "ICESat-2" not in name})
    pd.DataFrame(res).to_csv(CFG.TABLES / "hist3_corrected_dH.csv", index=False)

    r14 = [r for r in res if r["datum_bs_m"] == NEW_DATUM_BS][0]
    print(f"\n  On the published UNS datum the 1960s survey and ICESat-2 differ by")
    print(f"  {r14['median_dH_m']:+.2f} m [{r14['ci_lo']:+.2f}, {r14['ci_hi']:+.2f}], "
          f"against a point scatter of {r14['nmad_m']:.2f} m.")
    print(f"  Zero is {'INSIDE' if r14['ci_lo'] <= 0 <= r14['ci_hi'] else 'OUTSIDE'} "
          f"that interval.")
    if not (r14["ci_lo"] <= 0 <= r14["ci_hi"]):
        print(f"  A residual of this size is not evidence of bed change: the datum")
        print(f"  itself is only known to ~0.2-0.4 m (three estimates spanning 0.40 m),")
        print(f"  and that uncertainty is not propagated into the CI above.")
    print(f"\n  NOT CLAIMED: that the bed is unchanged since the survey. The survey")
    print(f"  epoch is still unrecorded, so no rate can be formed, and a residual")
    print(f"  of a few decimetres cannot be separated from datum uncertainty.")

    print("\n" + "=" * 72)
    print("DOWNSTREAM: NOT rerun here")
    print("=" * 72)
    print("  part11/part11b/part11c and every table and figure derived from them")
    print("  still carry the 16.00 m bed and are now known to be 2.00 m high.")
    print("  Regenerating them is a separate, reviewable step, not a side effect")
    print("  of this script -- they are the audit trail for what was published")
    print("  before the datum was resolved.")


if __name__ == "__main__":
    main()
