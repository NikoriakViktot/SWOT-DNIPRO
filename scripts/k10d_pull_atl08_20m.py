#!/usr/bin/env python
"""K10d.0 -- pull NATIVE 20 m PhoREAL terrain segments from ATL03 photons.

The project has never held an NSIDC ATL08 granule: data/processed/atl08 was
produced by SlideRule `icesat2.atl08p`, i.e. PhoREAL run server-side over ATL03
photons with len=100, res=100 (scripts/part8_atl08_terrain.py). The 100 m
support is therefore a REQUEST PARAMETER, not a product property, and the ASAS
fields the K10d spec asks for (h_te_best_fit_20m, latitude_20, longitude_20) do
not exist anywhere in this project.

Native 20 m is consequently obtained the only defensible way: by re-aggregating
the SAME ATL03 signal photons at len=20, res=20. This is NOT interpolation of
the 100 m product, and it is arguably stronger than ATL08's h_te_best_fit_20m,
which is a polynomial fitted across the 100 m segment and merely sampled at
20 m sub-segments.

Scope: POST_BREACH only (2023-09-01 .. 2026-01-01), matching the window the
100 m canonical set actually covers, so that D1_RAW / D1_MEDIAN20 / D1_KALMAN20
are later compared on like-for-like observations. Extending to 2026 data is a
separate decision and is deliberately NOT taken here.

Requires the `geoid` conda env (sliderule 4.15.1); the analysis env geohydroai
does not have sliderule installed.

Writes data/processed/atl08/kakhovka_atl08_20m_raw.parquet and touches nothing
that K10b / K10c depend on.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import sliderule
from sliderule import icesat2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/atl08/kakhovka_atl08_20m_raw.parquet"
BBOX = (33.30, 46.70, 35.40, 47.90)      # identical AOI to part8 / the ATL13 branch


def poly_from_bbox(b):
    lo, la, hi, ha = b
    return [{"lon": lo, "lat": la}, {"lon": hi, "lat": la}, {"lon": hi, "lat": ha},
            {"lon": lo, "lat": ha}, {"lon": lo, "lat": la}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-len", type=int, default=20)
    ap.add_argument("--ats", type=float, default=5.0)
    ap.add_argument("--cnt", type=int, default=5)
    ap.add_argument("--t0", default="2023-09-01")
    ap.add_argument("--t1", default="2026-01-01")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cnf", choices=("high", "medium", "low"), default="high",
                    help="ATL03 signal confidence floor; HIGH (k10d terrain default) drops most canopy photons -> use low for canopy pulls (plan 15, p47)")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sliderule.init("slideruleearth.io", verbose=False)
    poly = poly_from_bbox(BBOX)

    # quarterly windows: one 20 m request over the whole period is large enough
    # to time out, and a failed quarter can be retried without redoing the rest
    edges = pd.date_range(a.t0, a.t1, freq="QS").tolist()
    if pd.Timestamp(a.t0) not in edges:
        edges.insert(0, pd.Timestamp(a.t0))
    if pd.Timestamp(a.t1) not in edges:
        edges.append(pd.Timestamp(a.t1))
    windows = list(zip(edges[:-1], edges[1:]))

    print(f"SlideRule atl08p (PhoREAL) NATIVE {a.seg_len} m over {BBOX}")
    print(f"period {a.t0} .. {a.t1}  ({len(windows)} quarterly requests)")
    print("source = ATL03 signal photons, re-aggregated at the requested segment "
          "length. No interpolation of the 100 m product.\n")

    frames, failed = [], []
    t_all = time.time()
    for w0, w1 in windows:
        parms = {
            "poly": poly, "t0": str(w0.date()), "t1": str(w1.date()),
            "srt": icesat2.SRT_LAND, "cnf": {"high": icesat2.CNF_SURFACE_HIGH, "medium": icesat2.CNF_SURFACE_MEDIUM, "low": icesat2.CNF_SURFACE_LOW}[a.cnf],
            "atl08_class": ["atl08_ground", "atl08_canopy", "atl08_top_of_canopy"],
            "len": a.seg_len, "res": a.seg_len,
            # PhoREAL defaults (ats=20 m, cnt=10) are unsatisfiable at len=20: a
            # 20 m segment cannot contain photons spread over >=20 m, so the
            # default request returns ~0.1% of the expected segments (257 rows
            # for a quarter where len=100 returns 44,335). Lowered to the
            # smallest values that keep real photon support: at ats=5/cnt=5 the
            # same quarter yields 189,103 segments with a median of 17 ground
            # photons each (p10 = 6).
            "ats": a.ats, "cnt": a.cnt,
            "phoreal": {"binsize": 1.0, "geoloc": "center",
                        "use_abs_h": False, "send_waveform": False},
        }
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
    print(f"\n-> {out}  ({len(df):,} native {a.seg_len} m segments, "
          f"{time.time()-t_all:.0f} s total)")
    if failed:
        print(f"FAILED windows (rerun these before using the file): {failed}")
    print(f"columns: {list(df.columns)}")
    print(f"RGTs {df.rgt.nunique()}, cycles {df.cycle.nunique()}, "
          f"ground photons/segment median {df.gnd_ph_count.median():.0f}, "
          f"p10 {df.gnd_ph_count.quantile(.10):.0f}")


if __name__ == "__main__":
    main()
