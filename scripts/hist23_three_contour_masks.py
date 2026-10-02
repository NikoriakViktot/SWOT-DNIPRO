#!/usr/bin/env python
"""HISTORICAL 23 — build the three pre-breach shoreline contours.

Steps 1-3 of the three-contour experiment: acquire the imagery, build water
masks with the already-validated method, extract shorelines, and test whether
the three polygons are hydraulically consistent BEFORE any of them is used as
a bathymetric constraint.

THE THREE CONTOURS. Each carries different information and neither of the new
two dominates the other:

  H1  17.08 m  2023-06-05   the constraint already in use
  H2  15.43 m  2019-02-19   SIX gauges agreeing to 0.08 m -- the strongest
                            independent control on pool flatness, which is
                            what licenses treating a shoreline as one
                            elevation at all
  H3  14.38 m  2023-02-23   the largest vertical separation (-2.70 m), close
                            to the published UNS level

H2 is not "worse" than H3 for having a higher level. H3 buys vertical reach;
H2 buys verifiable horizontality. The experiment needs both.

ACQUISITION. Full SAFE archives would be ~18 GB for 18 tiles. Only four bands
are needed (B03, B08, B11, SCL), the Planetary Computer serves them as
individual COGs, and the reservoir occupies a small part of each tile -- so
each band is read in a window at 20 m directly into a common grid. That is the
same 20 m working resolution the existing pipeline uses (`subsample=2`), and it
is far finer than needed for a reservoir-scale shoreline.

Water-mask thresholds are IMPORTED from swot_dnipro.watermask and are NOT
adjusted here. Tuning them to improve contour nesting would be fitting the
answer.

Outputs
-------
data/processed/bathymetry/prebreach_contours.gpkg        (3 polygons + shorelines)
outputs/tables/prebreach_contour_inventory.csv
outputs/tables/prebreach_contour_nesting.csv
outputs/figures/V19_three_prebreach_contours.png
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import watermask as WM

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SASURL = ("https://planetarycomputer.microsoft.com/api/sas/v1/token/"
          "sentinel-2-l2a")
CELL = 20.0
BANDS = ("B03", "B08", "B11", "SCL")
GPKG = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
# Windowed optical reads are bulk data and live on drive F; the corrected
# grid differs from the P20 one, so the old cache cannot be reused.
CACHE = CFG.BULK_ROOT / "contour_cache_corrected"

# Each contour is a GROUP of dates whose reservoir level agrees closely.
# A single date is not enough: 2019-02-19 and 2023-02-23 each observe only
# ~62% of the footprint, because Sentinel-2 granules of those orbits are
# partial over this reservoir. Coverage is therefore composited across
# near-dates at a stable level ("first valid observation wins", best cloud
# first), and the residual level spread within a group is carried into the
# uncertainty budget rather than ignored.
#
# H2's group spans 0.070 m over eight dates -- and that group is exactly the
# six-gauge window, so its flatness is measured, not assumed.
CONTOURS = {
    "H1": dict(dates=["2023-06-05"],
               role="constraint already in use"),
    "H2": dict(dates=["2019-02-19", "2019-02-24", "2019-03-03", "2019-03-18",
                      "2019-03-16", "2019-03-21", "2019-03-06", "2019-03-13"],
               role="strongest flatness control (6 gauges, spread 0.08 m)"),
    "H3": dict(dates=["2023-02-23", "2023-02-20", "2023-02-10"],
               role="largest vertical separation, near UNS"),
}
LEVEL_SPREAD_MAX = 0.35     # refuse a group whose levels disagree by more


def sas_token():
    return json.loads(urllib.request.urlopen(SASURL, timeout=90).read())["token"]


def stac_scenes(date, bbox):
    q = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 50,
         "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z"}
    req = urllib.request.Request(STAC, data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read()
                      )["features"]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    # The analysis extent comes from the spatial-domain registry, never from
    # a derived geojson. P20_reservoir_footprint.geojson -- used here until
    # 2026-09 -- truncates 87.4 km2 of real water at the eastern end and adds
    # ~123 km2 of land, errors that nearly cancel in total area and so
    # survived review. H1/H2/H3 stay reservoir-only products, so the extent is
    # the reservoir water: the SA_2-bounded core plus the corrected water
    # just outside that envelope (the Khortytsia arms), and nothing below the
    # dam.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])

    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    g = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = g.groupby("date").transformed_level_m.median()
    ngau = g.groupby("date").station_or_domain.nunique()
    spr_ = g.groupby("date").transformed_level_m.agg(lambda v: v.max()-v.min())

    print("=" * 78)
    print("LEVEL VERIFICATION — every date checked against the gauge table")
    print("=" * 78)
    for cid, c in CONTOURS.items():
        hs = []
        for d in c["dates"]:
            if d not in lev.index:
                raise SystemExit(f"{cid}: no reservoir gauge reading for {d}")
            hs.append(float(lev[d]))
        c["date_levels"] = hs
        c["H"] = float(np.mean(hs))
        c["level_spread"] = float(np.max(hs) - np.min(hs))
        c["n_gauges"] = int(max(ngau[d] for d in c["dates"]))
        c["gauge_spread"] = float(max(spr_[d] for d in c["dates"]))
        print(f"\n  {cid}: {len(c['dates'])} date(s), mean H = {c['H']:.4f} m "
              f"EVRF2019, level spread {c['level_spread']:.3f} m")
        print(f"      up to {c['n_gauges']} gauge(s), worst same-day "
              f"inter-station spread {c['gauge_spread']:.3f} m — {c['role']}")
        for d, h in zip(c["dates"], hs):
            print(f"        {d}  {h:.4f} m  ({int(ngau[d])} gauges)")
        if c["level_spread"] > LEVEL_SPREAD_MAX:
            raise SystemExit(f"{cid}: level spread {c['level_spread']:.3f} m "
                             f"exceeds {LEVEL_SPREAD_MAX} m -- these dates are "
                             f"not one contour")
    hh = sorted(c["H"] for c in CONTOURS.values())
    print(f"\n  separations: {hh[2]-hh[1]:.2f} m and {hh[1]-hh[0]:.2f} m; "
          f"total span {hh[2]-hh[0]:.2f} m")

    # ---- common 20 m grid --------------------------------------------------
    x0 = np.floor(fp.bounds[0] / CELL) * CELL
    y0 = np.floor(fp.bounds[1] / CELL) * CELL
    x1 = np.ceil(fp.bounds[2] / CELL) * CELL
    y1 = np.ceil(fp.bounds[3] / CELL) * CELL
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    tr = from_origin(x0, y1, CELL, CELL)
    from rasterio.features import rasterize as rio_rasterize
    inside = rio_rasterize([(fp, 1)], out_shape=(ny, nx), transform=tr,
                           fill=0, dtype="uint8").astype(bool)
    print(f"\n  grid {nx} x {ny} at {CELL:.0f} m; footprint "
          f"{inside.sum()*CELL**2/1e6:,.0f} km2 in {inside.sum()/1e6:.2f} M cells")

    bbox4326 = [round(v, 5) for v in unary_union([
        SD.load_subzone("KAKHOVKA_RESERVOIR_CORE"),
        SD.load_subzone("FORMER_RESERVOIR_TRANSITION")]).bounds]
    tok = sas_token()

    def read_date(date):
        """Composite one date onto the common grid. Cached."""
        npz = CACHE / f"wm_{date}_{CELL:.0f}m.npz"
        if npz.exists():
            z = np.load(npz)
            return z["water"], z["valid"]
        feats = stac_scenes(date, bbox4326)
        byt = {}
        for f in feats:
            t = f["properties"]["s2:mgrs_tile"]
            cc = float(f["properties"].get("eo:cloud_cover", 100))
            if t not in byt or cc < byt[t][0]:
                byt[t] = (cc, f)
        water = np.zeros((ny, nx), bool)
        valid = np.zeros((ny, nx), bool)
        for t, (cc, f) in sorted(byt.items()):
            try:
                arr = {}
                for b in BANDS:
                    with rasterio.open(f["assets"][b]["href"] + "?" + tok) as ds:
                        arr[b] = ds.read(
                            1, window=from_bounds(x0, y0, x1, y1, ds.transform),
                            out_shape=(ny, nx),
                            resampling=Resampling.nearest,
                            boundless=True, fill_value=0)
            except Exception as e:
                print(f"      {t}: READ FAILED {type(e).__name__} -- skipped")
                continue
            scl = arr["SCL"].astype(np.int16)
            cov = scl > 0
            if not cov.any():
                continue
            g3 = arr["B03"].astype(np.float32)
            n8 = arr["B08"].astype(np.float32)
            s11 = arr["B11"].astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                ndwi = np.nan_to_num((g3 - n8) / (g3 + n8), nan=-9.0)
                mndwi = np.nan_to_num((g3 - s11) / (g3 + s11), nan=-9.0)
            wt = (ndwi > WM.DEFAULT_NDWI) & (mndwi > WM.DEFAULT_MNDWI)
            wt &= ~np.isin(scl, WM.SCL_REJECT)
            wt |= (scl == 6) & (ndwi > WM.DEFAULT_NDWI - 0.15)
            wt &= ~np.isin(scl, WM.SCL_REJECT)
            wt &= g3 > 0
            usable = cov & ~np.isin(scl, WM.SCL_REJECT)
            water |= wt & usable
            valid |= usable
        np.savez_compressed(npz, water=water, valid=valid)
        return water, valid

    inv_rows, polys, valids = [], {}, {}
    for cid, c in CONTOURS.items():
        print("\n" + "=" * 78)
        print(f"{cid} — {len(c['dates'])} date(s) — H = {c['H']:.3f} m EVRF2019")
        print("=" * 78)
        water = np.zeros((ny, nx), bool)
        valid = np.zeros((ny, nx), bool)
        used = []
        for d in c["dates"]:
            wt, vd = read_date(d)
            newpx = vd & ~valid & inside
            gain = float(newpx.sum()) / max(1, inside.sum())
            # FIRST VALID OBSERVATION WINS -- never OR water across dates,
            # which would inflate the extent with any single date's false
            # positives.
            water |= wt & newpx
            valid |= vd
            used.append((d, gain))
            print(f"    {d}: +{100*gain:5.1f}% of footprint newly observed "
                  f"(cumulative {100*(valid&inside).sum()/inside.sum():5.1f}%)")
            if (valid & inside).sum() / inside.sum() > 0.995:
                print(f"    -> coverage complete, remaining dates not needed")
                break
        obs = float((valid & inside).sum()) / inside.sum()
        contributed = [d for d, gn in used if gn > 0.001]
        hs = [float(lev[d]) for d in contributed]
        c["H_used"] = float(np.mean(hs))
        c["spread_used"] = float(np.max(hs) - np.min(hs)) if len(hs) > 1 else 0.0
        wf = water & inside
        lab, nlab = ndimage.label(wf, structure=np.ones((3, 3), int))
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nlab + 1))
        conn = lab == (int(np.argmax(sizes)) + 1)
        geoms = [shp_shape(sh) for sh, v in rio_shapes(
            conn.astype(np.uint8), mask=conn, transform=tr) if v == 1]
        poly = unary_union(geoms).buffer(0)
        if poly.geom_type == "MultiPolygon":
            keep = [q for q in poly.geoms if q.area > 0.5e6]
            poly = unary_union(keep) if keep else poly
        polys[cid] = poly
        valids[cid] = valid & inside
        print(f"  dates that contributed : {len(contributed)} "
              f"({', '.join(contributed)})")
        print(f"  level actually used    : {c['H_used']:.4f} m "
              f"(spread {c['spread_used']:.3f} m)")
        print(f"  observed fraction      : {obs:.3f}")
        print(f"  water inside footprint : {wf.sum()*CELL**2/1e6:,.0f} km2 in "
              f"{nlab:,} components")
        print(f"  polygon                : {poly.area/1e6:,.0f} km2, "
              f"boundary {poly.boundary.length/1e3:,.0f} km")
        inv_rows.append(dict(
            contour_id=cid, dates="|".join(contributed),
            n_dates=len(contributed), H_evrf2019_m=c["H_used"],
            H_bs77_m=c["H_used"] - 0.185,
            level_spread_within_group_m=c["spread_used"],
            n_gauges=c["n_gauges"], interstation_spread_m=c["gauge_spread"],
            role=c["role"], observed_fraction=obs,
            water_area_inside_footprint_km2=wf.sum()*CELL**2/1e6,
            n_components=int(nlab), polygon_area_km2=poly.area/1e6,
            shoreline_length_km=poly.boundary.length/1e3, grid_cell_m=CELL,
            ndwi_thr=WM.DEFAULT_NDWI, mndwi_thr=WM.DEFAULT_MNDWI))

    inv = pd.DataFrame(inv_rows).sort_values("H_evrf2019_m").reset_index(drop=True)
    inv.to_csv(CFG.TABLES / "prebreach_contour_inventory.csv", index=False)

    # ============================================ HYDRAULIC CONSISTENCY
    print("\n" + "=" * 78)
    print("3. HYDRAULIC CONSISTENCY — on the COMMON observed domain")
    print("=" * 78)
    common = valids["H1"] & valids["H2"] & valids["H3"]
    print(f"  common observed domain: {common.sum()*CELL**2/1e6:,.0f} km2 "
          f"({100*common.sum()/inside.sum():.1f}% of the footprint)")
    print(f"  Comparing raw polygon areas would be invalid: each contour sees")
    print(f"  a different part of the reservoir. Areas below are restricted to")
    print(f"  the SAME cells for all three.")
    areas = {}
    for cid in inv.contour_id:
        m = rio_rasterize([(polys[cid], 1)], out_shape=(ny, nx), transform=tr,
                          fill=0, dtype="uint8").astype(bool)
        areas[cid] = dict(full=polys[cid].area/1e6,
                          common=float((m & common).sum())*CELL**2/1e6,
                          mask=m)
    print(f"\n  {'contour':>8}{'H':>9}{'full km2':>11}{'on common km2':>15}")
    for cid in inv.contour_id:
        H = float(inv[inv.contour_id == cid].H_evrf2019_m.iloc[0])
        print(f"  {cid:>8}{H:>9.3f}{areas[cid]['full']:>11,.0f}"
              f"{areas[cid]['common']:>15,.0f}")
    seq = list(inv.contour_id)
    ok_area = all(areas[seq[i]]["common"] < areas[seq[i+1]]["common"]
                  for i in range(len(seq)-1))
    print(f"\n  monotonic on the common domain? "
          f"{'PASS' if ok_area else 'FAIL'}")

    rows = []
    print(f"\n  {'lower':>6}{'higher':>8}{'inside %':>10}{'non-nested km2':>16}"
          f"{'verdict':>9}")
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            lo, hi = seq[i], seq[j]
            a = areas[lo]["mask"] & common
            b = areas[hi]["mask"] & common
            inter = float((a & b).sum())
            pct = 100*inter/max(1, a.sum())
            nn = float((a & ~b).sum())*CELL**2/1e6
            rows.append(dict(lower=lo, higher=hi,
                             lower_area_common_km2=a.sum()*CELL**2/1e6,
                             higher_area_common_km2=b.sum()*CELL**2/1e6,
                             pct_of_lower_inside_higher=pct,
                             non_nested_km2=nn))
            print(f"  {lo:>6}{hi:>8}{pct:>9.2f}%{nn:>16.1f}"
                  f"{'ok' if pct > 97 else 'CHECK':>9}")
    pd.DataFrame(rows).to_csv(
        CFG.TABLES / "prebreach_contour_nesting.csv", index=False)
    print("\n  Nesting was NOT enforced and no water-mask threshold was")
    print("  adjusted to obtain it. Residual non-nested area is expected from")
    print("  disconnected water at low level, 20 m mixed pixels, and February")
    print("  shadow/ice misclassification. It is reported, not removed.")

    # ---- an independent check the contours were never fitted to -----------
    lav = pd.read_csv(ROOT / "data/historical/historical_level_area_volume.csv")
    print("\n  CROSS-CHECK against Table 19 A(H) — the contours were built")
    print("  without any reference to it:")
    print(f"  {'contour':>8}{'H BS-77':>10}{'A observed':>12}"
          f"{'A Table 19':>12}{'diff':>9}")
    for r in inv.itertuples():
        ah = float(np.interp(r.H_bs77_m, lav.water_level_m.values[::-1],
                             lav.surface_area_km2.values[::-1]))
        print(f"  {r.contour_id:>8}{r.H_bs77_m:>10.3f}"
              f"{r.polygon_area_km2:>12,.0f}{ah:>12,.0f}"
              f"{100*(r.polygon_area_km2-ah)/ah:>+8.1f}%")
    print("  A consistent NEGATIVE bias is expected: NDWI/MNDWI sees OPEN")
    print("  water, and the reedy margins of this reservoir are inundated but")
    print("  not open. That offset is carried into the contour uncertainty in")
    print("  hist24 rather than corrected away here.")

    try:
        import geopandas as gpd
        gpd.GeoDataFrame(inv.assign(geometry=[polys[c] for c in inv.contour_id]),
                         crs=CFG.CRS_METRIC).to_file(
            GPKG, layer="contour_polygons", driver="GPKG")
        gpd.GeoDataFrame(
            inv[["contour_id", "dates", "H_evrf2019_m"]].assign(
                geometry=[polys[c].boundary for c in inv.contour_id]),
            crs=CFG.CRS_METRIC).to_file(GPKG, layer="shorelines", driver="GPKG")
        print(f"\n-> {GPKG}")
    except Exception as e:
        print(f"\n  geopackage write skipped: {type(e).__name__}: {e}")

    # ================================================================ figure
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.6))
    a = ax[0]
    cols = {"H1": RED, "H2": GREEN, "H3": BLUE}
    for r in inv.iloc[::-1].itertuples():          # highest first
        p = polys[r.contour_id]
        gs = p.geoms if p.geom_type == "MultiPolygon" else [p]
        for k, gg in enumerate(gs):
            xs, ys = gg.exterior.xy
            a.fill(np.array(xs) / 1e3, np.array(ys) / 1e3,
                   color=cols[r.contour_id], alpha=0.42,
                   label=(f"{r.contour_id}  {r.H_evrf2019_m:.2f} m  "
                          f"{r.polygon_area_km2:,.0f} km²  "
                          f"({r.n_dates}d, obs {r.observed_fraction:.2f})")
                   if k == 0 else None)
    xs, ys = fp.exterior.xy
    a.plot(np.array(xs) / 1e3, np.array(ys) / 1e3, color=INK, lw=0.9,
           label="pre-breach footprint")
    a.set_aspect("equal")
    a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
    a.legend(fontsize=8.4, loc="lower left")
    a.set_title("a · Three pre-breach water extents\nnested by construction "
                "of the reservoir, not by processing", fontsize=11, loc="left")

    a = ax[1]
    a.plot(inv.polygon_area_km2, inv.H_evrf2019_m, "o-", color=INK, lw=2,
           ms=9, zorder=4, label="Sentinel-2 contours")
    for r in inv.itertuples():
        a.annotate(f"  {r.contour_id}",
                   (r.polygon_area_km2, r.H_evrf2019_m), fontsize=9,
                   va="center")
        if r.n_gauges > 1:
            a.errorbar([r.polygon_area_km2], [r.H_evrf2019_m],
                       yerr=[r.interstation_spread_m], fmt="none",
                       ecolor=GREEN, elinewidth=2.4, capsize=5, zorder=5)
    lav = pd.read_csv(ROOT / "data/historical/historical_level_area_volume.csv")
    a.plot(lav.surface_area_km2, lav.water_level_m + 0.185, "s--",
           color=AMBER, lw=1.6, ms=4.5, alpha=0.9,
           label="historical Table 19 (validation only)")
    a.set_xlabel("water-surface area (km²)")
    a.set_ylabel("level (m EVRF2019)")
    a.legend(fontsize=8.6); a.grid(alpha=0.25)
    a.set_title("b · Independent A(H) points\nTable 19 shown for reference — "
                "NOT used to build them", fontsize=11, loc="left")

    fig.suptitle("V19 · Three independent pre-breach shoreline contours   ·   "
                 "Sentinel-2 L2A, NDWI+MNDWI+SCL, thresholds unchanged",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V19_three_prebreach_contours.{e}", dpi=175,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/'V19_three_prebreach_contours.png'}")
    print(f"-> {CFG.TABLES/'prebreach_contour_inventory.csv'}")
    print(f"-> {CFG.TABLES/'prebreach_contour_nesting.csv'}")


if __name__ == "__main__":
    main()
