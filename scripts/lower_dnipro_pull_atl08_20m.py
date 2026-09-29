#!/usr/bin/env python
"""LOWER_DNIPRO.2 -- native 20 m PhoREAL terrain segments, Kakhovka dam to the
Black Sea (lower Dnipro reach, Kherson, and the Dnipro-Buh liman).

Same SlideRule atl08p call as k10d_pull_atl08_20m.py (len=20, res=20, ats=5,
cnt=5 -- the same fix for PhoREAL's unsatisfiable default ats=20 at len=20),
pointed at the lower-Dnipro AOI instead of Kakhovka. Requires a sliderule
client >= 5.x (the 4.15.1 client bundled in the `geoid` conda env is
incompatible with the current server; this session uses an isolated venv at
$SCRATCHPAD/sr5venv with sliderule 5.6.0 -- see k10d_pull_atl08_20m.py's
history for the diagnosis).

BBOX matches lower_dnipro_water_occurrence.py exactly: dam (~33.35E) through
Kherson (32.61E) to Ochakiv/the sea (31.5E), overlapping the Kakhovka bbox's
western edge (33.30E) on purpose so there is no gap between the two domains.
An earlier version of this script only covered the liman proper (31.5-32.65E)
and silently missed the dam-to-Kherson river reach.

Period 2022-01-01 .. 2026-01-01: unlike the Kakhovka reservoir (whose useful
window starts at the 2023-06-06 breach), this reach has no breach/drainage
event of its own -- pre-2023 ground exposure is just as valid a QC target as
post-2023, and both are wanted so the 2022 water-occurrence baseline
(LOWER_DNIPRO.1) has a matching pre/post ICESat-2 ground comparison if one
turns out to be useful.

Outputs
-------
data/processed/atl08/lower_dnipro_atl08_20m_raw.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sliderule
from sliderule import icesat2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/atl08/lower_dnipro_atl08_20m_raw.parquet"
# same bbox as lower_dnipro_water_occurrence.py -- keep them in lockstep
BBOX = (31.50, 46.30, 33.40, 47.00)


def poly_from_bbox(b):
    lo, la, hi, ha = b
    return [{"lon": lo, "lat": la}, {"lon": hi, "lat": la}, {"lon": hi, "lat": ha},
            {"lon": lo, "lat": ha}, {"lon": lo, "lat": la}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-len", type=int, default=20)
    ap.add_argument("--ats", type=float, default=5.0)
    ap.add_argument("--cnt", type=int, default=5)
    ap.add_argument("--t0", default="2022-01-01")
    ap.add_argument("--t1", default="2026-01-01")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sliderule.init("slideruleearth.io", verbose=False)
    poly = poly_from_bbox(BBOX)

    edges = pd.date_range(a.t0, a.t1, freq="QS").tolist()
    if pd.Timestamp(a.t0) not in edges:
        edges.insert(0, pd.Timestamp(a.t0))
    if pd.Timestamp(a.t1) not in edges:
        edges.append(pd.Timestamp(a.t1))
    windows = list(zip(edges[:-1], edges[1:]))

    print(f"SlideRule atl08p (PhoREAL) NATIVE {a.seg_len} m over lower-Dnipro bbox {BBOX}")
    print(f"period {a.t0} .. {a.t1}  ({len(windows)} quarterly requests)")
    print(f"ats={a.ats}, cnt={a.cnt} (PhoREAL defaults ats=20/cnt=10 are unsatisfiable "
          f"at len={a.seg_len}; see k10d_pull_atl08_20m.py for the diagnosis)\n")

    frames, failed = [], []
    for w0, w1 in windows:
        parms = {
            "poly": poly, "t0": str(w0.date()), "t1": str(w1.date()),
            "srt": icesat2.SRT_LAND, "cnf": icesat2.CNF_SURFACE_HIGH,
            "atl08_class": ["atl08_ground", "atl08_canopy", "atl08_top_of_canopy"],
            "len": a.seg_len, "res": a.seg_len, "ats": a.ats, "cnt": a.cnt,
            "phoreal": {"binsize": 1.0, "geoloc": "center",
                        "use_abs_h": False, "send_waveform": False},
        }
        import time
        t = time.time()
        try:
            g = icesat2.atl08p(parms)
        except Exception as e:
            print(f"  {w0.date()} .. {w1.date()}: FAILED {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            failed.append((str(w0.date()), str(w1.date())))
            continue
        if len(g):
            frames.append(g.reset_index())
        print(f"  {w0.date()} .. {w1.date()}: {len(g):>8,} segments in "
              f"{time.time()-t:>5.0f}s", flush=True)

    if not frames:
        raise SystemExit("no data returned")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"\n-> {out}  ({len(df):,} native {a.seg_len} m segments)")
    if failed:
        print(f"FAILED windows (rerun before trusting completeness): {failed}")
    print(f"RGTs {df.rgt.nunique()}, cycles {df.cycle.nunique()}, "
          f"ground photons/segment median {df.gnd_ph_count.median():.0f}")


if __name__ == "__main__":
    main()
