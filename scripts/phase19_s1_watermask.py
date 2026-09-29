#!/usr/bin/env python
"""Phase 19 — Sentinel-1 SAR water mask (SNAP-free), for the imagery-poor dates.

Why SNAP-free: the reference pipeline (github.com/NikoriakViktot/flood_snap) is an
ESA-SNAP-12 + esa_snappy + WhiteboxTools + Docker microservice stack; SNAP is not
installed here and will not fit (C: has ~9 GB free). The Kakhovka reservoir / lower
Dnipro is FLAT, so full radiometric terrain correction is unnecessary — GDAL's SAFE
driver gives calibrated sigma0, GCP geocoding is enough, and the water rule is a
simple backscatter threshold (dark = smooth water).

Reused verbatim from flood_snap: the Otsu-on-log10(sigma0) threshold engine
(worker/engines/otsu.py) — clamp [0.005, 0.05], fallback 0.017, min 1000 valid px —
and the "sigma0_VV < threshold => water" rule with the 1-10 % water sanity check.

Output matches scripts/phase19_watermasks.py so scripts/phase19_classify.py can sample
an S1 mask exactly like an S2 mask:
    data/processed/water_masks_s1/<scene>.npz   mask(bool) + labels(i4) + affine + crs + tile + sensing
    data/processed/water_masks_s1/<scene>_water.tif
    outputs/tables/s1_scene_inventory.csv

Usage:  phase19_s1_watermask.py 2024-11-05 [2024-02-07 ...]   (defaults to the
imagery-poor HIGH feasibility dates)
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import rasterio
import rasterio.control
import requests
from scipy import ndimage

from swot_dnipro import config as CFG

RAW = Path("/mnt/e/data_swot/sentinel1")
MASKS = CFG.ROOT / "data" / "processed" / "water_masks_s1"
INV = CFG.TABLES / "s1_scene_inventory.csv"
RAW.mkdir(parents=True, exist_ok=True)
MASKS.mkdir(parents=True, exist_ok=True)

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")

PIX_M = 20.0                    # match the 20 m S2 water masks
MAX_LAG_DAYS = 6               # S1 candidates this far from the ATL13 date
# Otsu clamps — flood_snap/worker/engines/otsu.py
OTSU_FALLBACK, OTSU_MIN, OTSU_MAX, OTSU_MIN_VALID = 0.017, 0.005, 0.05, 1000

# imagery-poor HIGH feasibility dates (S2 cloud / lag failed them)
DEFAULT_DATES = ["2024-11-05", "2024-02-07", "2025-08-05"]


# --------------------------------------------------------------------------- #
# Otsu — reused verbatim from flood_snap worker/engines/otsu.py               #
# --------------------------------------------------------------------------- #
def otsu_threshold(sigma0: np.ndarray) -> float:
    valid = sigma0[np.isfinite(sigma0) & (sigma0 > 0)]
    if valid.size < OTSU_MIN_VALID:
        return OTSU_FALLBACK
    data = np.log10(valid)
    hist, edges = np.histogram(data, bins=256)
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * edges[:-1])
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)
    thr = 10 ** edges[int(np.argmax(sigma_b))]
    return float(np.clip(thr, OTSU_MIN, OTSU_MAX))


# --------------------------------------------------------------------------- #
# CDSE                                                                         #
# --------------------------------------------------------------------------- #
def token() -> str:
    c = json.loads((pathlib.Path.home() / ".config/cdse/credentials.json").read_text())
    r = requests.post(TOKEN_URL, timeout=60, data={
        "client_id": "cdse-public", "username": c["username"],
        "password": c["password"], "grant_type": "password"})
    r.raise_for_status()
    return r.json()["access_token"]


class KeepAuth(requests.Session):
    def rebuild_auth(self, prepared, response):
        return


def discover(atl_date: pd.Timestamp, bbox) -> pd.DataFrame:
    lo, la, hi, ha = bbox
    wkt = f"POLYGON(({lo} {la},{hi} {la},{hi} {ha},{lo} {ha},{lo} {la}))"
    a = (atl_date - pd.Timedelta(days=MAX_LAG_DAYS)).strftime("%Y-%m-%d")
    b = (atl_date + pd.Timedelta(days=MAX_LAG_DAYS)).strftime("%Y-%m-%d")
    f = (f"Collection/Name eq 'SENTINEL-1' "
         f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
         f"and ContentDate/Start gt {a}T00:00:00.000Z "
         f"and ContentDate/Start lt {b}T23:59:59.999Z "
         f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
         f"and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S')")
    tok = token()
    r = requests.get(ODATA, params={"$filter": f, "$top": 100, "$expand": "Attributes",
                                    "$orderby": "ContentDate/Start asc"},
                     headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    r.raise_for_status()
    rows = []
    for p in r.json().get("value", []):
        name = p.get("Name", "")
        if "_COG" in name or "GRDH" not in name:
            continue
        st = p.get("ContentDate", {}).get("Start")
        rows.append({"name": name, "product_id": p.get("Id"), "sensing_start": st,
                     "size_bytes": int(p.get("ContentLength") or 0),
                     "lag_days": abs((pd.to_datetime(st).tz_localize(None) - atl_date)
                                     .total_seconds()) / 86400.0})
    return pd.DataFrame(rows).sort_values("lag_days")


def download(row, sess) -> Path:
    zp = RAW / f"{row['name']}.zip"
    if zp.exists():
        return zp
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({row['product_id']})/$value"
    t0 = time.time()
    with sess.get(url, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        tmp = zp.with_suffix(".zip.part")
        n = 0
        with open(tmp, "wb") as fh:
            for ch in resp.iter_content(1 << 20):
                fh.write(ch); n += len(ch)
        tmp.replace(zp)
    print(f"    downloaded {n/1e6:.0f} MB in {time.time()-t0:.0f}s -> {zp.name}", flush=True)
    return zp


# --------------------------------------------------------------------------- #
# SAR -> sigma0 -> water mask                                                  #
# --------------------------------------------------------------------------- #
def safe_manifest(zp: Path) -> str:
    """Extract the .SAFE and return the path GDAL's SAFE driver wants."""
    out = RAW / zp.stem
    if not out.exists():
        with zipfile.ZipFile(zp) as z:
            z.extractall(RAW)
    safe = next(out.glob("*.SAFE")) if not out.name.endswith(".SAFE") else out
    return str(safe)


"""GDAL's SAFE driver does not expose the SIGMA0 subdataset for this product
('Measurement bands not found'), so sigma0 = DN**2 / sigmaNought(line,pixel)**2 is
applied from the calibration LUT — see _calib_full()."""


def _calib_full(safe: str, pol: str, H: int, W: int) -> np.ndarray:
    """sigmaNought LUT -> full radar-grid array, row-blocked to bound memory."""
    from scipy.interpolate import RectBivariateSpline
    xml = next(Path(safe).glob(f"annotation/calibration/calibration-*-{pol}-*.xml"))
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    lines, pax, rows = [], None, []
    for cv in root.findall(".//calibrationVector"):
        lines.append(int(cv.findtext("line")))
        px = np.array(cv.findtext("pixel").split(), float)
        rows.append(np.array(cv.findtext("sigmaNought").split(), float))
        if pax is None:
            pax = px
    lines = np.array(lines, float)
    spl = RectBivariateSpline(lines, pax, np.array(rows),
                              kx=min(3, len(lines) - 1), ky=min(3, len(pax) - 1))
    cc = np.arange(W, dtype=float)
    sn = np.empty((H, W), dtype="float32")
    for r0 in range(0, H, 2048):
        r1 = min(r0 + 2048, H)
        sn[r0:r1] = spl(np.arange(r0, r1, dtype=float), cc).astype("float32")
    return np.clip(sn, 1e-3, None)


def read_sigma0(safe: str, bbox, utm_epsg: str, pol: str = "vv"):
    """UNCALIB DN -> calibration LUT -> sigma0 (full frame) -> write with the product
    GCPs -> gdalwarp -tps clip+geocode to the AOI (UTM, linear scale). Flat terrain,
    no RTC. Returns (sigma0_utm, transform, crs)."""
    lo, la, hi, ha = bbox
    sub = f"SENTINEL1_CALIB:UNCALIB:{safe}/manifest.safe:IW_{pol.upper()}:AMPLITUDE"
    with rasterio.open(sub) as ds:
        gcps, gcrs = ds.gcps
        H, W = ds.height, ds.width
        dn = ds.read(1).astype("float32")
    sn = _calib_full(safe, pol, H, W)
    s0 = (dn * dn) / (sn * sn)
    del sn
    s0[~np.isfinite(s0)] = 0.0
    s0[dn <= 0] = 0.0
    del dn

    with tempfile.TemporaryDirectory() as td:
        radar = Path(td) / "s0_radar.tif"
        geo = Path(td) / "s0_utm.tif"
        with rasterio.open(radar, "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="float32", gcps=gcps, crs=gcrs or "EPSG:4326",
                           nodata=0.0, compress="lzw") as dst:
            dst.write(s0, 1)
        del s0
        cmd = ["gdalwarp", "-overwrite", "-q", "-t_srs", utm_epsg, "-tps",
               "-r", "bilinear", "-tr", str(PIX_M), str(PIX_M),
               "-te_srs", "EPSG:4326", "-te", str(lo), str(la), str(hi), str(ha),
               "-dstnodata", "0", str(radar), str(geo)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with rasterio.open(geo) as ds:
            out = ds.read(1).astype("float32")
            transform, crs = ds.transform, ds.crs
    out[~np.isfinite(out)] = 0.0
    return out, transform, crs


def enhanced_lee(img: np.ndarray, size: int = 7, looks: float = 4.0) -> np.ndarray:
    """Enhanced-Lee speckle filter (Lopes 1990) — replaces SNAP 'Lee Sigma'."""
    img = img.astype("float32")
    valid = img > 0
    m = ndimage.uniform_filter(np.where(valid, img, 0.0), size)
    c = ndimage.uniform_filter(valid.astype("float32"), size)
    mean = np.divide(m, c, out=np.zeros_like(m), where=c > 0)
    sq = ndimage.uniform_filter(np.where(valid, img * img, 0.0), size)
    meansq = np.divide(sq, c, out=np.zeros_like(sq), where=c > 0)
    var = np.clip(meansq - mean * mean, 0, None)
    ci = np.sqrt(var) / np.maximum(mean, 1e-6)
    cu = 1.0 / np.sqrt(looks)
    cmax = np.sqrt(1.0 + 2.0 / looks)
    w = np.exp(-(ci - cu) / np.maximum(cmax - ci, 1e-6))
    w = np.clip(w, 0, 1)
    out = np.where(ci <= cu, mean, np.where(ci >= cmax, img, mean + w * (img - mean)))
    return np.where(valid, out, 0.0).astype("float32")


VH_MIN, VH_MAX = 0.0020, 0.0130       # Otsu clamp for cross-pol (~ -27 .. -19 dB)


def water_mask(vv: np.ndarray, vh: np.ndarray | None):
    """Dual-pol dark-water rule: (sigma0_VV < Otsu_VV) AND (sigma0_VH < Otsu_VH).

    Cross-pol (VH) is added because over the drained Kakhovka bed and bare autumn
    fields VV alone is near-unimodal — smooth dry ground is as dark as calm water.
    VH is lower over genuine water (closer to the noise floor) than over soil.
    """
    fvv = enhanced_lee(vv)
    valid = fvv > 0
    thr_vv = otsu_threshold(fvv)
    water = valid & (fvv < thr_vv)
    thr_vh = np.nan
    if vh is not None:
        fvh = enhanced_lee(vh)
        valid &= fvh > 0
        thr_vh = float(np.clip(otsu_threshold(fvh), VH_MIN, VH_MAX))
        water &= (fvh < thr_vh)
    water &= valid
    water = ndimage.binary_opening(water, np.ones((3, 3)))
    water = ndimage.binary_closing(water, np.ones((3, 3)))
    labels, n = ndimage.label(water, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        water = np.isin(labels, np.where(sizes >= 25)[0])       # drop < 1 ha specks
        labels, n = ndimage.label(water, structure=np.ones((3, 3)))
    with np.errstate(divide="ignore"):
        s0_db = np.where(valid, 10 * np.log10(np.maximum(fvv, 1e-6)), np.nan)
    return water, labels.astype("i4"), thr_vv, thr_vh, s0_db, valid


def process(atl_date_str: str, sess) -> dict | None:
    d = pd.Timestamp(atl_date_str)
    seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet")
    seg["dt"] = pd.to_datetime(seg["time"], utc=True, errors="coerce").dt.tz_localize(None)
    g = seg[(seg.dt >= d) & (seg.dt < d + pd.Timedelta(days=1))]
    if g.empty:
        print(f"  {atl_date_str}: no ATL13 segments"); return None
    pad = 0.15
    bbox = tuple(float(x) for x in (g.lon.min() - pad, g.lat.min() - pad,
                                    g.lon.max() + pad, g.lat.max() + pad))
    zone = int((bbox[0] + bbox[2]) / 2 // 6) + 31
    utm = f"EPSG:326{zone:02d}"
    print(f"  {atl_date_str}: ATL13 bbox {tuple(round(x,2) for x in bbox)}  UTM {utm}", flush=True)

    cand = discover(d, bbox)
    if cand.empty:
        print("    no S1 IW_GRDH candidates"); return None
    print(cand[["name", "sensing_start", "lag_days", "size_bytes"]].head(6).to_string(index=False))

    # ATL13-track pixel coverage test: the frame must actually contain the track,
    # not just intersect the padded bbox (adjacent along-track frames do not).
    tlo, tla, thi, tha = (float(g.lon.min()), float(g.lat.min()),
                          float(g.lon.max()), float(g.lat.max()))

    for _, row in cand.iterrows():
        try:
            zp = download(row, sess)
            safe = safe_manifest(zp)
            vv, transform, crs = read_sigma0(safe, bbox, utm, "vv")
            try:
                vh, _, _ = read_sigma0(safe, bbox, utm, "vh")
            except Exception:
                vh = None
        except subprocess.CalledProcessError as e:
            print(f"    {row['name']}: gdalwarp failed ({e.stderr[-300:]}), next", flush=True)
            continue
        except Exception as e:
            import traceback
            print(f"    {row['name']}: {type(e).__name__} {e}\n{traceback.format_exc()}", flush=True)
            continue

        # does the frame cover the ATL13 track itself?
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        gx, gy = tf.transform(g.lon.values, g.lat.values)
        inv = ~transform
        gc, gr = inv * (gx, gy)
        gc = np.floor(gc).astype(int); gr = np.floor(gr).astype(int)
        hgt, wid = vv.shape
        on = (gr >= 0) & (gr < hgt) & (gc >= 0) & (gc < wid)
        cov = float((on & (vv[np.clip(gr, 0, hgt - 1), np.clip(gc, 0, wid - 1)] > 0)).mean())
        if cov < 0.5:
            print(f"    {row['name']}: covers only {cov*100:.0f}% of the ATL13 track, next",
                  flush=True)
            continue

        water, labels, thr, thr_vh, s0_db, valid = water_mask(vv, vh)
        wk = float(water.sum() * PIX_M ** 2 / 1e6)
        wpct = 100 * water.sum() / max(valid.sum(), 1)
        sensing = pd.to_datetime(row["sensing_start"]).strftime("%Y%m%dT%H%M%S")
        stem = row["name"].replace(".SAFE", "")
        a = transform
        np.savez_compressed(
            MASKS / f"{stem}.npz",
            mask=water, labels=labels,
            affine=np.array([a.a, a.b, a.c, a.d, a.e, a.f], float),
            crs=str(crs), tile=f"S1_{row['name'].split('_')[4][:8]}", sensing=sensing,
            sigma0_db=s0_db.astype("f2"))
        h, w = water.shape
        with rasterio.open(MASKS / f"{stem}_water.tif", "w", driver="GTiff",
                           height=h, width=w, count=1, dtype="uint8",
                           crs=crs, transform=transform, compress="deflate",
                           nodata=255) as dst:
            dst.write(np.where(valid, water.astype("uint8"), 255), 1)
            dst.update_tags(1, sensing_time=sensing,
                            sensor="Sentinel-1 IW GRDH VV+VH",
                            method=f"enhanced-Lee 7x7 + Otsu VV={thr:.4f} VH={thr_vh:.4f}")
        # how much of the ATL13 track sits on S1 water (diagnostic, printed)
        trk_water = float(water[np.clip(gr, 0, hgt - 1), np.clip(gc, 0, wid - 1)][on].mean())
        print(f"    OK {stem}: otsu VV={thr:.4f} VH={thr_vh:.4f}  water {wk:.1f} km2 "
              f"({wpct:.1f}%)  comps {labels.max()}  track-on-water {trk_water*100:.0f}%  "
              f"lag {row['lag_days']:.2f}d", flush=True)
        return {"atl_date": atl_date_str, "name": stem, "sensing_start": row["sensing_start"],
                "lag_days": row["lag_days"], "otsu_vv": thr, "otsu_vh": thr_vh,
                "water_km2": wk, "water_pct": wpct, "track_on_water_pct": trk_water * 100,
                "n_components": int(labels.max()), "track_cov_pct": cov * 100,
                "valid_px": int(valid.sum()), "crs": str(crs),
                "npz": str(MASKS / f"{stem}.npz")}
    return None


def main() -> None:
    dates = sys.argv[1:] or DEFAULT_DATES
    sess = KeepAuth()
    sess.headers.update({"Authorization": f"Bearer {token()}"})
    rows = []
    for ds in dates:
        r = process(ds, sess)
        if r:
            rows.append(r)
    if rows:
        add = pd.DataFrame(rows)
        old = pd.read_csv(INV) if INV.exists() else pd.DataFrame()
        merged = pd.concat([old[~old.name.isin(add.name)] if len(old) else old, add],
                           ignore_index=True)
        merged.to_csv(INV, index=False)
        print(f"\n-> {INV}  ({len(merged)} S1 masks)")
        print(add.to_string(index=False))
    else:
        print("\nno S1 masks produced")


if __name__ == "__main__":
    main()
