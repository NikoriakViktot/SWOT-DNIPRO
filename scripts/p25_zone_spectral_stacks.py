#!/usr/bin/env python
"""P25 -- per-date spectral stacks and per-regime composites on every zone grid.

The first time any spectral index is computed over ZONE_2, ZONE_3 or ZONE_4.
All 89 existing k10e stacks and 83 part9 NDVI tiles are eastern MGRS tiles on
their own tile grids; nothing was ever put on a registry zone grid, and the
delta / estuary / floodway had zero optical index coverage.

Everything scientific lives in `swot_dnipro.sentinel_preprocess` (frozen water
rule, k10e classes, BOA offset, seven indices). This script only decides WHICH
scenes go on WHICH grid, writes the three separate products plan 03 §3.3 asks
for, and builds composites.

Scene selection
---------------
ZONE_2/3/4  the frozen p10 manifests (`outputs/tables/p10_<zone>_s2_manifest.csv`),
            one row per DATE with `tiles` pipe-separated; every listed tile is
            looked for in the local SAFE archives. Dates whose tiles are not
            all on disk are still processed (partial mosaic) and recorded as
            such -- ZONE_3's 36TUS/36TUT are still being fetched.
ZONE_1      `outputs/tables/water_mask_summary.csv` grouped by date (the
            reservoir-side scenes phase19/p1/f4 verified) -- 53 dates 2023-05..2025-11 only.
            `--dates all` (plan 15 WP0.2) instead enumerates EVERY eastern-tile SAFE zip in
            the local stores (36TWS/36TWT/36TXS/36TXT, 2017-2026) grouped by date, so the
            pre-breach years and 2026 get stacks too.

SCL (added 2026-09-18, plan 15 WP0.2)
------------------------------------
    <date>_scl.tif       uint8 ESA Scene Classification on the zone grid (0 = no data) --
                         the 'land use Sentinel-2 itself provides'. Dates written before this
                         change are backfilled cheaply with `sentinel_preprocess.scl_to_grid`
                         (SCL band only, same first-valid mosaic order). Every per-date file
                         now carries the tag `tiles_complete` (all expected tiles found).

Grids and products
------------------
Grid: `sentinel_preprocess.zone_grid(zone, 20 m)` -> `SD.build_grid`.
Per date, three files on the bulk volume, `BULK_ROOT/zone_spectral/<ZONE>/`:
    <date>_indices.tif   7 x int16 (value*10000, nodata -32768): NDVI NDWI MNDWI NDMI BSI AWEIsh NDTI
    <date>_class.tif     uint8 k10e physical class (0 = invalid)
    <date>_water3.tif    uint8 0 land / 1 water / 255 not observed
Per regime, small composites in `outputs/rasters/zone<N>/`:
    zone<N>_<index>_<regime>_median_20m.tif, zone<N>_class_<regime>_mode_20m.tif,
    zone<N>_water_frac_<regime>_20m.tif, zone<N>_n_valid_<regime>_20m.tif
Composites are built in row blocks from the per-date int16 files, so ZONE_1
(96 Mpx) never holds more than one block of dates in memory.

Regime labels
-------------
PRE_BREACH  < 2023-06-06 ; BREACH_DRAWDOWN 2023-06-06 .. 2023-08-31 ;
POST_BREACH >= 2023-09-01 (config.BREACH_DATE + the POST0 cut used by
part9/k10d). part9 called the middle one `DRAWDOWN`; this script uses
`BREACH_DRAWDOWN` everywhere and the manifest records the mapping.

Usage
-----
python scripts/p25_zone_spectral_stacks.py --zone ZONE_2_KHERSON_DELTA
python scripts/p25_zone_spectral_stacks.py --zone ZONE_1_KAKHOVKA_LOWER_DNIPRO --limit 5
python scripts/p25_zone_spectral_stacks.py --zone ZONE_2_KHERSON_DELTA --composites-only
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP

ZONES = {
    "ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2,
    "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4,
}
MANIFESTS = {
    "ZONE_2_KHERSON_DELTA": "p10_zone_2_kherson_delta_s2_manifest.csv",
    "ZONE_3_DNIPRO_BUG_ESTUARY": "p10_zone_3_dnipro_bug_estuary_s2_manifest.csv",
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY": "p10_zone_4_dam_to_kherson_floodway_s2_manifest.csv",
}
SAFE_DIRS = (CFG.BULK_ROOT / "s2_zone_fetch", CFG.BULK_ROOT / "data_swot" / "sentinel",
             ROOT / "data" / "raw" / "sentinel_p1_targeted")
OUT_ROOT = CFG.BULK_ROOT / "zone_spectral"
POST0 = "2023-09-01"
CELL = 20.0
BLOCK_ROWS = 256
REGIMES = ("PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH")


def regime_of(date: str) -> str:
    if date < CFG.BREACH_DATE:
        return "PRE_BREACH"
    return "BREACH_DRAWDOWN" if date < POST0 else "POST_BREACH"


def find_zip(date: str, tile: str) -> Path | None:
    tok = date.replace("-", "")
    for d in SAFE_DIRS:
        if not d.exists():
            continue
        hits = sorted(d.glob(f"*{tok}T*_T{tile}_*.SAFE.zip"))
        hits = [h for h in hits if h.stat().st_size > 200_000_000]
        if hits:
            return hits[0]
    return None


EASTERN = ("36TWS", "36TWT", "36TXT")          # 36TXS is an orbit-edge sliver (0.04 GB) and was never archived
EASTERN_OPTIONAL = ("36TXS", "36UWU", "36UXU")


def scene_plan(zone: str, dates: str = "frag") -> pd.DataFrame:
    """One row per date: tiles expected, zips found, regime, manifest extras."""
    rows = []
    if zone not in MANIFESTS and dates == "all":
        by_date: dict[str, dict] = {}
        for d in SAFE_DIRS:
            if not d.exists():
                continue
            for z in d.glob("*_MSIL2A_*.SAFE.zip"):
                parts = z.name.split("_")
                if len(parts) < 6 or parts[5][1:] not in EASTERN + EASTERN_OPTIONAL or z.stat().st_size < 50_000_000:   # 50 MB: R021 edge slivers of 36TXT were stacked by the old plan and must stay backfillable
                    continue
                date = f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"
                by_date.setdefault(date, {})
                by_date[date].setdefault(parts[5][1:], str(z))      # first store wins
        for date, tz in sorted(by_date.items()):
            rows.append(dict(date=date, tiles_expected="|".join(EASTERN), tiles_found="|".join(sorted(tz)),
                             zips=[tz[t] for t in sorted(tz)], clear_fraction_manifest=np.nan, since_breach="", season=""))
    elif zone in MANIFESTS:
        m = pd.read_csv(CFG.TABLES / MANIFESTS[zone])
        for r in m.itertuples():
            date = str(r.date)[:10]
            tiles = [t.strip() for t in str(r.tiles).split("|") if t.strip()]
            zips = {t: find_zip(date, t) for t in tiles}
            rows.append(dict(date=date, tiles_expected="|".join(tiles),
                             tiles_found="|".join(t for t, z in zips.items() if z),
                             zips=[str(z) for z in zips.values() if z],
                             clear_fraction_manifest=getattr(r, "clear_fraction_over_zone", np.nan),
                             since_breach=getattr(r, "since_breach", ""),
                             season=getattr(r, "season", "")))
    else:
        wm = pd.read_csv(CFG.TABLES / "water_mask_summary.csv")
        wm["date"] = pd.to_datetime(wm.sensing_time, format="%Y%m%dT%H%M%S").dt.strftime("%Y-%m-%d")
        for date, g in wm.groupby("date"):
            zips, tiles = [], []
            for n, t in zip(g.name, g.tile):
                tiles.append(str(t).replace("T36", "36"))
                for d in SAFE_DIRS:
                    p = d / f"{n}.zip"
                    if p.exists():
                        zips.append(str(p)); break
            rows.append(dict(date=date, tiles_expected="|".join(tiles),
                             tiles_found="|".join(tiles[:len(zips)]), zips=zips,
                             clear_fraction_manifest=np.nan, since_breach="", season=""))
    P = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    P["regime"] = P.date.map(regime_of)
    P["n_zips"] = P.zips.map(len)
    return P


def process_date(row, G, out_dir: Path) -> dict:
    date = row.date
    paths = {k: out_dir / f"{date}_{k}.tif" for k in ("indices", "class", "water3", "scl")}
    tiles_complete = set(str(row.tiles_expected).split("|")) <= set(str(row.tiles_found).split("|"))
    rec = dict(date=date, regime=row.regime, tiles_expected=row.tiles_expected,
               tiles_found=row.tiles_found, n_scenes=row.n_zips, tiles_complete=tiles_complete)
    core = all(paths[k].exists() for k in ("indices", "class", "water3"))
    if core and paths["scl"].exists():
        rec["status"] = "CACHED"
        return rec
    if not row.zips:
        rec["status"] = "NO_SCENES"
        return rec
    t0 = time.time()
    if core:                                   # SCL backfill only (plan 15 WP0.2)
        scl = SP.scl_to_grid([Path(z) for z in row.zips], G, CELL)
        if scl is None:
            rec["status"] = "SCL_READ_FAILED"
            return rec
        SP.write_uint8(paths["scl"], np.clip(scl, 0, 255), G,
                       dict(zone=G["zone"], date=date, regime=row.regime, cell_m=str(CELL),
                            values="ESA SCL 0=nodata 1..11; first-valid mosaic order as the indices",
                            tiles_complete=str(tiles_complete), producer="p25_zone_spectral_stacks.py (scl backfill)"),
                       nodata=0)
        rec["status"] = "SCL_BACKFILLED"
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    B = SP.mosaic_to_grid([Path(z) for z in row.zips], G, CELL)
    if B is None:
        rec["status"] = "READ_FAILED"
        return rec
    P = SP.process_on_grid(B)
    ins = G["inside"]
    v = P["valid"] & ins
    rec.update(valid_frac_inside=float(v.sum() / max(ins.sum(), 1)),
               water_frac_of_valid=float(P["water"][v].mean()) if v.any() else np.nan,
               scenes="|".join(m.name for m in B["metas"]),
               processing_baselines="|".join(sorted({m.processing_baseline for m in B["metas"]})),
               boa_offset_B03="|".join(sorted({str(m.offsets.get("B03")) for m in B["metas"]})))
    cls = P["class"][v]
    for k, name in SP.CLASSES.items():
        if k in (0, 8):
            continue
        rec[f"frac_{name}"] = float(np.mean(cls == k)) if len(cls) else np.nan
    tags = dict(zone=G["zone"], date=date, regime=row.regime, cell_m=str(CELL),
                scenes=rec["scenes"], processing_baselines=rec["processing_baselines"],
                boa_offset_applied="yes (from MTD_MSIL2A.xml)",
                water_rule="NDWI>0 & MNDWI>0 & SCL-permitted (frozen)",
                class_thresholds=f"NDVI {SP.NDVI_SPARSE}/{SP.NDVI_VEG} NDMI {SP.NDMI_WET} BSI {SP.BSI_BARE}",
                grid="SD.build_grid cell-centre registration", producer="p25_zone_spectral_stacks.py",
                tiles_complete=str(tiles_complete))
    out_dir.mkdir(parents=True, exist_ok=True)
    SP.write_indices(paths["indices"], P["indices"], P["valid"], G, tags)
    SP.write_uint8(paths["class"], P["class"], G, {**tags, "legend": str(SP.CLASSES)}, nodata=0)
    SP.write_uint8(paths["water3"], SP.three_valued(P["water"], P["valid"]), G,
                   {**tags, "values": "0=land 1=water 255=not_observed"})
    SP.write_uint8(paths["scl"], np.clip(B["SCL"], 0, 255), G,
                   {**tags, "values": "ESA SCL 0=nodata 1..11 (first-valid mosaic)"}, nodata=0)
    rec["status"] = "WRITTEN"
    rec["seconds"] = round(time.time() - t0, 1)
    del B, P
    return rec


def composites(zone: str, G: dict, man: pd.DataFrame) -> list[str]:
    """Per-regime median index, class mode, water frequency, n_valid -- blockwise."""
    n = ZONES[zone]
    out_dir = ROOT / "outputs" / "rasters" / f"zone{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = OUT_ROOT / zone
    written = []
    ny, nx = G["ny"], G["nx"]
    for regime in REGIMES:
        dates = man[(man.regime == regime) & man.status.isin(["WRITTEN", "CACHED", "SCL_BACKFILLED"])].date.tolist()
        if not dates:
            continue
        idx_files = [src_dir / f"{d}_indices.tif" for d in dates]
        cls_files = [src_dir / f"{d}_class.tif" for d in dates]
        w3_files = [src_dir / f"{d}_water3.tif" for d in dates]
        med = {k: np.full((ny, nx), SP.INDEX_NODATA, "i2") for k in SP.INDEX_NAMES}
        mode = np.zeros((ny, nx), "u1")
        wfrac = np.full((ny, nx), 255, "u1")
        nval = np.zeros((ny, nx), "u1")
        srcs_i = [rasterio.open(f) for f in idx_files]
        srcs_c = [rasterio.open(f) for f in cls_files]
        srcs_w = [rasterio.open(f) for f in w3_files]
        try:
            for r0 in range(0, ny, BLOCK_ROWS):
                h = min(BLOCK_ROWS, ny - r0)
                win = Window(0, r0, nx, h)
                W = np.stack([s.read(1, window=win) for s in srcs_w])          # (t,h,nx) u8
                obs = W != 255
                cnt = obs.sum(0).astype("u1")
                nval[r0:r0 + h] = cnt
                with np.errstate(invalid="ignore"):
                    wf = np.where(cnt > 0, (W == 1).sum(0) / np.maximum(cnt, 1), np.nan)
                wfrac[r0:r0 + h] = np.where(cnt > 0, np.round(wf * 100), 255).astype("u1")
                C = np.stack([s.read(1, window=win) for s in srcs_c])
                # mode over valid classes (0 = invalid excluded)
                counts = np.zeros((10, h, nx), "u1")
                for k in range(1, 10):
                    counts[k] = (C == k).sum(0)
                mo = counts.argmax(0).astype("u1")
                mo[counts.max(0) == 0] = 0
                mode[r0:r0 + h] = mo
                for bi, k in enumerate(SP.INDEX_NAMES, 1):
                    A = np.stack([s.read(bi, window=win) for s in srcs_i]).astype("f4")
                    A[A == SP.INDEX_NODATA] = np.nan
                    m = np.nanmedian(A, axis=0)
                    med[k][r0:r0 + h] = np.where(np.isfinite(m), np.round(m), SP.INDEX_NODATA).astype("i2")
                del W, C, A
        finally:
            for s in srcs_i + srcs_c + srcs_w:
                s.close()
        tags = dict(zone=zone, regime=regime, n_dates=str(len(dates)), dates="|".join(dates),
                    producer="p25_zone_spectral_stacks.py", cell_m=str(CELL))
        for k in SP.INDEX_NAMES:
            p = out_dir / f"zone{n}_{k}_{regime}_median_20m.tif"
            with rasterio.open(p, "w", **SP._profile(G, 1, "int16", SP.INDEX_NODATA)) as dst:
                dst.write(med[k], 1); dst.set_band_description(1, f"{k} median x{SP.INDEX_SCALE}")
                dst.update_tags(index=k, scale=f"value/{SP.INDEX_SCALE}", **tags)
            written.append(str(p))
        SP.write_uint8(out_dir / f"zone{n}_class_{regime}_mode_20m.tif", mode, G,
                       {**tags, "legend": str(SP.CLASSES)}, nodata=0)
        SP.write_uint8(out_dir / f"zone{n}_water_frac_{regime}_20m.tif", wfrac, G,
                       {**tags, "values": "percent of observed dates with water; 255 = never observed"})
        SP.write_uint8(out_dir / f"zone{n}_n_valid_{regime}_20m.tif", nval, G,
                       {**tags, "values": "number of dates observed"}, nodata=0)
        written += [str(out_dir / f"zone{n}_{s}_{regime}_20m.tif") for s in ("class_mode", "water_frac", "n_valid")]
        print(f"  composites {regime}: {len(dates)} dates -> {out_dir}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=list(ZONES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--composites-only", action="store_true")
    ap.add_argument("--dates", choices=("frag", "all"), default="frag",
                    help="ZONE_1 only: 'frag' = the 53 water_mask_summary dates; 'all' = every eastern-tile scene on disk")
    ap.add_argument("--extra-dates", nargs="*", default=[], help="manifest zones: add these YYYY-MM-DD dates with the manifest's tile set (2026-09-19: June-2023 flood dates for ZONE_2)")
    a = ap.parse_args()

    print("=" * 78)
    print(f"P25 -- zone spectral stacks: {a.zone}")
    print("=" * 78)
    G = SP.zone_grid(a.zone, CELL)
    print(f"  grid {G['nx']}x{G['ny']} at {CELL:.0f} m, {int(G['inside'].sum()):,} cells inside")
    plan = scene_plan(a.zone, a.dates)
    if a.extra_dates:
        tiles = sorted(set(t for ts in plan.tiles_expected for t in str(ts).split("|") if t))
        extra = []
        for date in a.extra_dates:
            if date in set(plan.date):
                continue
            zips = {t: find_zip(date, t) for t in tiles}
            if any(zips.values()):
                extra.append(dict(date=date, tiles_expected="|".join(tiles), tiles_found="|".join(t for t, z in zips.items() if z), zips=[str(z) for z in zips.values() if z], clear_fraction_manifest=np.nan, since_breach="", season=""))
        if extra:
            ex = pd.DataFrame(extra); ex["regime"] = ex.date.map(regime_of); ex["n_zips"] = ex.zips.map(len)
            plan = pd.concat([plan, ex], ignore_index=True).sort_values("date").reset_index(drop=True)
        print(f"  extra dates added: {[e['date'] for e in extra]}")
    print(f"  dates in plan: {len(plan)}  with zips: {int((plan.n_zips>0).sum())}  "
          f"regimes: {plan.regime.value_counts().to_dict()}")
    if a.limit:
        plan = plan.head(a.limit)

    out_dir = OUT_ROOT / a.zone
    man_path = CFG.TABLES / f"p25_zone_spectral_manifest_{a.zone}.csv"
    recs = []
    if not a.composites_only:
        t0 = time.time()
        for i, row in enumerate(plan.itertuples(), 1):
            rec = process_date(row, G, out_dir)
            recs.append(rec)
            print(f"  [{i}/{len(plan)}] {row.date} {row.regime:15s} {rec['status']:11s} "
                  f"scenes {rec.get('n_scenes',0)} valid {rec.get('valid_frac_inside',np.nan):.2f} "
                  f"water {rec.get('water_frac_of_valid',np.nan):.3f} {time.time()-t0:.0f}s", flush=True)
        man = pd.DataFrame(recs)
        man.to_csv(man_path, index=False)
        print(f"\n-> {man_path}")
        print(man.status.value_counts().to_string())
    else:
        man = pd.read_csv(man_path)

    written = composites(a.zone, G, man)
    print(f"\n  {len(written)} composite rasters written")


if __name__ == "__main__":
    main()
