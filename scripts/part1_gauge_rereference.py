#!/usr/bin/env python
"""PART 1 — re-reference every gauge series into EVRF2019.

    H_BS77     = zero_BS77 + stage
    H_EVRF2019 = H_BS77 + delta_EPSG9902(lon, lat)

This is the OFFICIAL GEODETIC TRANSFORMATION only. No empirical satellite/gauge
alignment is applied here and none is implied; those live in Part 2 and are kept
in separate columns so the provenance of every metre is traceable.

Outputs
-------
outputs/tables/gauge_vertical_reference_summary.csv
data/processed/gauges/gauge_levels_evrf2019.parquet
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
import yaml

from swot_dnipro import config as CFG

ICE = CFG.ICESAT_ROOT
ASC = ICE / "data/external/datum/ua_2019z.asc"
CSV_DIR = ICE / "data/1_data/data/csv/dm_H"
PQ_DIR = ICE / "data/1_data/data/parquet"
OUT_PQ = ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet"
FILL_RUN = 5          # >= this many identical consecutive values -> fill_suspect


# --------------------------------------------------------------------------- #
# EPSG:9902  BS-77 -> EVRF2019 (zero-tide) grid
# --------------------------------------------------------------------------- #
def load_grid(path=ASC):
    head, vals = {}, []
    with open(path) as fh:
        for _ in range(6):
            k, v = fh.readline().split()
            head[k.lower()] = float(v)
        for line in fh:
            vals.extend(float(x) for x in line.split())
    nc, nr = int(head["ncols"]), int(head["nrows"])
    z = np.array(vals, float).reshape(nr, nc)
    z[z == head["nodata_value"]] = np.nan
    return z, head


def sample_grid(z, head, lon, lat):
    """Bilinear sample of an ESRI ASCII grid (row 0 = north)."""
    cs, nc, nr = head["cellsize"], int(head["ncols"]), int(head["nrows"])
    x0 = head["xllcorner"] + cs / 2.0                 # centre of column 0
    ytop = head["yllcorner"] + cs * (nr - 0.5)        # centre of row 0
    fc = (np.asarray(lon, float) - x0) / cs
    fr = (ytop - np.asarray(lat, float)) / cs
    c0 = np.clip(np.floor(fc).astype(int), 0, nc - 2)
    r0 = np.clip(np.floor(fr).astype(int), 0, nr - 2)
    dc, dr = fc - c0, fr - r0
    q = (z[r0, c0] * (1 - dc) * (1 - dr) + z[r0, c0 + 1] * dc * (1 - dr) +
         z[r0 + 1, c0] * (1 - dc) * dr + z[r0 + 1, c0 + 1] * dc * dr)
    return q


# --------------------------------------------------------------------------- #
def stations():
    cfg = yaml.safe_load((ICE / "config/gauges.yaml").read_text())
    out = []
    for s in cfg.get("stations", []):
        out.append({"station_id": int(s["id"]), "name": s.get("name"),
                    "name_en": s.get("name_en"), "slug": s.get("slug"),
                    "lat": s["lat"], "lon": s["lon"],
                    "zero_bs77_m": s.get("gauge_zero_baltic_m"),
                    "domain": "reservoir"})
    for s in cfg.get("downstream", []):
        if s.get("gauge_zero_baltic_m") is None:
            continue
        out.append({"station_id": int(s["id"]), "name": s.get("name"),
                    "name_en": s.get("name_en"), "slug": s.get("slug"),
                    "lat": s["lat"], "lon": s["lon"],
                    "zero_bs77_m": s.get("gauge_zero_baltic_m"),
                    "domain": "downstream"})
    return pd.DataFrame(out)


def flag_fill(v: pd.Series) -> pd.Series:
    """True where a value repeats identically for >= FILL_RUN consecutive rows.

    Catches donor fill blocks such as Rozumivka 2023-07-01..20, which sit at
    exactly 0.0 cm on both terms for 20 days (unique in 2019-2025).
    """
    g = (v != v.shift()).cumsum()
    n = v.groupby(g).transform("size")
    return (n >= FILL_RUN) & v.notna()


def read_daily_csv(sid) -> pd.DataFrame:
    for name in (f"{sid}_reservoir.csv", f"{sid}_yearbook.csv"):
        p = CSV_DIR / name
        if p.exists():
            d = pd.read_csv(p)
            d = d[d.stat_type == "daily"].copy()
            d["date"] = pd.to_datetime(d["date"])
            d["stage_m"] = d["water_level_cm"] / 100.0
            d["source_file"] = name
            d["series"] = "daily_yearbook"
            return d[["date", "stage_m", "series", "source_file"]]
    return pd.DataFrame()


def read_term_parquet(sid) -> pd.DataFrame:
    fs = sorted((PQ_DIR / f"post_id={sid}").glob("**/*.parquet"))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["h_mean"])
    d["stage_m"] = d["h_mean"] / 100.0
    d["series"] = "term_08_20_mean"
    d["source_file"] = f"parquet/post_id={sid}"
    return d[["date", "stage_m", "series", "source_file"]]


def main() -> None:
    z, head = load_grid()
    st = stations()
    st["delta_epsg9902_m"] = sample_grid(z, head, st.lon.values, st.lat.values)
    st["nominal_zero_evrf2019_m"] = st.zero_bs77_m + st.delta_epsg9902_m

    # self-check against the values the ICESat-2 project computed independently
    ref = pd.read_csv(ICE / "outputs/tables/egg2015_to_evrf2019_by_station.csv")
    chk = st.merge(ref[["station_id", "delta_epsg9902_m"]], on="station_id",
                   suffixes=("", "_ref"))
    dmax = float((chk.delta_epsg9902_m - chk.delta_epsg9902_m_ref).abs().max())
    print(f"grid sampler check vs ICESat-2 project: max |diff| = {dmax*1000:.3f} mm "
          f"over {len(chk)} stations")
    if dmax > 1e-3:
        raise SystemExit("EPSG:9902 sampler disagrees with the reference values")

    rows, summary = [], []
    for _, s in st.iterrows():
        sid = s.station_id
        parts = [p for p in (read_daily_csv(sid), read_term_parquet(sid)) if len(p)]
        if not parts:
            summary.append({**s.to_dict(), "n_obs": 0, "date_min": pd.NaT,
                            "date_max": pd.NaT, "series_used": "",
                            "n_fill_suspect": 0, "spans_breach": False,
                            "notes": "no level series found in the donor layout"})
            continue
        d = pd.concat(parts, ignore_index=True).sort_values(["date", "series"])
        # prefer the term/parquet value where a date exists in both
        d = d.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
        d["fill_suspect"] = flag_fill(d.stage_m)
        d["station_id"] = sid
        d["name_en"] = s.name_en
        d["domain"] = s.domain
        d["lat"], d["lon"] = s.lat, s.lon
        d["zero_bs77_m"] = s.zero_bs77_m
        d["delta_epsg9902_m"] = s.delta_epsg9902_m
        d["H_bs77_m"] = d.zero_bs77_m + d.stage_m
        d["H_evrf2019_m"] = d.H_bs77_m + d.delta_epsg9902_m
        d["raw_vertical_frame"] = "BS-77 stage above gauge zero"
        d["common_vertical_frame"] = "EVRF2019 (EPSG:9389, zero-tide)"
        d["qc"] = np.where(d.fill_suspect, "fill_suspect", "ok")
        rows.append(d)

        spans = bool((d.date.min() < pd.Timestamp("2023-06-06")) and
                     (d.date.max() > pd.Timestamp("2023-06-06")))
        summary.append({**s.to_dict(), "n_obs": len(d),
                        "date_min": d.date.min(), "date_max": d.date.max(),
                        "series_used": "+".join(sorted(d.series.unique())),
                        "n_fill_suspect": int(d.fill_suspect.sum()),
                        "spans_breach": spans,
                        "notes": ""})

    lv = pd.concat(rows, ignore_index=True)
    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    lv.to_parquet(OUT_PQ, index=False)

    sm = pd.DataFrame(summary).sort_values(["domain", "station_id"])
    sm.to_csv(CFG.TABLES / "gauge_vertical_reference_summary.csv", index=False)

    print(f"\n-> {OUT_PQ}  ({len(lv):,} rows, {lv.station_id.nunique()} stations)")
    print(f"-> {CFG.TABLES/'gauge_vertical_reference_summary.csv'}\n")
    cols = ["station_id", "name_en", "domain", "zero_bs77_m", "delta_epsg9902_m",
            "nominal_zero_evrf2019_m", "n_obs", "date_min", "date_max",
            "n_fill_suspect", "spans_breach"]
    with pd.option_context("display.width", 200):
        print(sm[cols].to_string(index=False))


if __name__ == "__main__":
    main()
