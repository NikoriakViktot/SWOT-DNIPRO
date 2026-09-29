#!/usr/bin/env python
"""P62 -- GEDI L2A analysis on the drained Kakhovka bed: ground elevations in EVRF2019, canopy structure per roughness class,
and an orthogonal cross-check against ICESat-2 (user 2026-09-19: "тепер по геді треба зробити аналіз і канопі це готово").

Input: $BULK_ROOT/gedi/gedi02a_pool_<year>.parquet (p48; 2024 1.57 M shots 06.05-31.12, 2025 1.19 M shots to 09.07; GEDI L2A v002.
2026: CMR returns 0 granules -- the ISS instrument's 2026 record is not in the archive yet, recorded as a gap, NOT as "no vegetation").

Vertical chain (rule of 2026-09-19, memory feedback-vertical-chain-via-ellipsoid): elev_lowestmode is ALREADY ellipsoidal (WGS84),
so no geoid is removed; only the frame step is applied: H_EVRF2019 = elev_lowestmode + free2mean(lat) - zeta_EGG2015(lon,lat) + c,
c = mean EGG2015->EVRF2019 corrector of the six pool stations (CFG.CORRECTOR_BY_STATION, -0.173 m). free2mean is applied on the
ICESat-2 convention (tide-free -> mean tide, -0.037 m here); the `--no-free2mean` arm reports the same statistics without it, so the
GEDI tide convention is tested empirically rather than assumed.

A  GROUND, orthogonal to p57's ICESat-2: residual = seamless DEM (p55) - GEDI ground, m EVRF2019, after dropping |res| > 50 m blunders
   (0.75 %). "Bare" CANNOT be defined by rh98 (GEDI's ground-return width puts bare-ground rh98 at ~2.7 m, p48b), so the bare set is
   taken from the surface class (bare_sand_silt / wet_sediment / sparse_herbaceous) with a second arm rh98 < 3 m. Strata: DEM source
   (1 zone bed v2, 2 reservoir bed hist20, 3 FABDEM, 4 feather band, 5 pool gap fill), night/day, power/coverage, sensitivity, year,
   class. FABDEM is carried as a DEM-INDEPENDENT CONTROL: where p55 keeps FABDEM (source 3) the two rasters are identical, so
   "FABDEM - GEDI" isolates GEDI's own error while "seamless - FABDEM" isolates what p55 changed.
B  CANOPY per Manning class (the measured heights the roughness table lacked): the p43 class raster of the matching state
   (2024 -> STATE_2024, 2025 -> STATE_2025) sampled at each footprint -> rh98/rh75/rh50 quantiles, woody share (rh98 >= 4 m, the
   GEDI-specific threshold from p48b's bare-ground floor), structure ratios rh50/rh98 and rh75/rh98.
B2 MATCHED WINDOW: GEDI 2025 stops on 09.07, so the raw 2024-vs-2025 contrast mixes growth with season and sampling. Both years are
   therefore also compared on 01.06-09.07 only, which leaves the bare and water classes as the internal null control.
C  GEDI vs ICESat-2 on the SAME 250 m cells. Their HEIGHTS are not comparable (ATL08 h_canopy is 0 in 94 % of cells where no canopy is
   detected; GEDI rh98 cannot go below ~2.7 m), so the comparable quantity is each sensor's WOODY SHARE of footprints per cell, each
   with its own threshold (GEDI rh98 >= 4 m, ICESat-2 h_canopy >= 2 m).
D  GROWTH 2024 -> 2025 on cells seen in both years: GEDI's ISS orbit has no exact repeat tracks, so this is reported as
   NOT_IDENTIFIABLE (3 common cells) and the growth signal is read from B2 instead.
Outputs: outputs/tables/p62_gedi_ground_accuracy.csv, p62_gedi_canopy_by_class.csv, p62_gedi_canopy_matched_window.csv,
         p62_gedi_vs_icesat2_cells.csv, p62_gedi_growth_paired.csv; outputs/figures/p62_gedi_analysis.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import vertical as VT

GEDI = CFG.BULK_ROOT / "gedi"; SEAM = CFG.BULK_ROOT / "dem_seamless"; RGH = ROOT / "outputs/rasters/roughness/pool"
ZONE1 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"; WOODY_M = 4.0; LOW_RH98_M = 3.0; BLUNDER_M = 50.0; SENS_MIN = 0.9
MIN_SHOTS, MIN_SEGS, WOODY_CELL_PCT, MIN_CELLS_CLASS = 3, 5, 20.0, 15
MATCH_WINDOW = ("06-01", "07-09")  # GEDI 2025 ends 09.07 -> the only window both years share
def src_area_km2():
    """ZONE_1 area per p55 source class, READ from the raster: hard-coding it made the area column report pre-fix areas next to post-fix statistics."""
    p = SEAM / f"{ZONE1}_dem_source_20m.tif"
    if not p.exists():
        return {}
    with rasterio.open(p) as ds:
        a = ds.read(1); px = abs(ds.transform.a * ds.transform.e) / 1e6
        u, n = np.unique(a[a > 0], return_counts=True)
    return {int(k): round(float(c) * px, 1) for k, c in zip(u, n)}
STATE_OF_YEAR = {2024: "STATE_2024", 2025: "STATE_2025"}
SRC_NAME = {1: "1 zone bed v2", 2: "2 reservoir bed (hist20)", 3: "3 FABDEM", 4: "4 FABDEM feathered", 5: "5 pool gap fill"}


def sample(path, xs, ys):
    """Raster values at UTM points; NaN outside / at nodata."""
    path = Path(path)
    if not path.exists():
        return np.full(len(xs), np.nan)
    with rasterio.open(path) as ds:
        out = np.array([v[0] for v in ds.sample(np.c_[xs, ys], masked=False)], dtype="f8")
        if ds.nodata is not None:
            out[out == ds.nodata] = np.nan
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys)
        outside = (np.asarray(r) < 0) | (np.asarray(r) >= ds.height) | (np.asarray(c) < 0) | (np.asarray(c) >= ds.width)
        out[outside] = np.nan
    return out


def metrics(res, label, **extra):
    r = np.asarray(res, float); r = r[np.isfinite(r)]
    if len(r) < 30:
        return dict(stratum=label, n=len(r), **extra)
    a = np.abs(r)
    return dict(stratum=label, n=len(r), bias=round(float(np.mean(r)), 3), median=round(float(np.median(r)), 3), MAE=round(float(np.mean(a)), 3),
                RMSE=round(float(np.sqrt(np.mean(r ** 2))), 3), LE90=round(float(np.percentile(a, 90)), 3), LE95=round(float(np.percentile(a, 95)), 3),
                NMAD=round(float(1.4826 * np.median(np.abs(r - np.median(r)))), 3), **extra)


def q(s, p):
    return round(float(np.nanpercentile(s, p)), 2) if np.isfinite(s).sum() >= 10 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-free2mean", action="store_true", help="vertical-convention arm: skip the tide-free->mean-tide term")
    args = ap.parse_args()
    SRC_AREA = src_area_km2()
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr.c_station_m.mean())
    codes = yaml.safe_load((ROOT / "config/roughness_classes.yaml").read_text())["codes"]
    tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)
    # ---------------- load + clean
    parts = []
    for f in sorted(GEDI.glob("gedi02a_pool_*.parquet")):
        d = pd.read_parquet(f); d["year"] = int(f.stem.split("_")[-1]); parts.append(d)
    D = pd.concat(parts, ignore_index=True); n_raw = len(D)
    D = D[(D.quality_flag == 1) & (D.degrade_flag == 0) & D.sensitivity.between(SENS_MIN, 1.0) & D.rh98.between(-2, 60) & D.elev_lowestmode.between(-60, 300)].copy()
    D["x"], D["y"] = tf.transform(D.lon.values, D.lat.values); D["night"] = D.solar_elevation < 0; D["power"] = D.power_beam.astype(bool) if "power_beam" in D else D.beam.isin({"BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011"})
    D["zeta"] = VT.sample_grid(CFG.EGG2015_TIF, D.lon.values, D.lat.values); f2m = 0.0 if args.no_free2mean else CFG.free2mean(D.lat.values)
    D["H_gedi"] = D.elev_lowestmode + f2m - D.zeta + c
    D["leafon"] = pd.to_datetime(D.time).dt.month.between(6, 9)
    print(f"GEDI shots: {n_raw:,} raw -> {len(D):,} quality (quality_flag 1, degrade 0, sensitivity >= {SENS_MIN}); night {D.night.mean():.0%}, power {D.power.mean():.0%}, leaf-on {D.leafon.mean():.0%}")
    print(f"chain: elev_lowestmode (WGS84 ellipsoid) {'+ free2mean ' + f'{np.median(f2m):+.3f}' if not args.no_free2mean else '(NO free2mean arm)'} - zeta_EGG2015 {np.median(D.zeta):.3f} + c {c:+.3f} -> EVRF2019")
    # ---------------- surface class per shot (needed by A and B): the p43 Manning classes of the matching state
    cls = np.full(len(D), np.nan)
    for yr, state in STATE_OF_YEAR.items():
        m = (D.year == yr).values
        if m.any():
            cls[m] = sample(RGH / f"zone1_poolwin_manning_classes_{state}.tif", D.x.values[m], D.y.values[m])
    D["cls"] = cls; D["cls_name"] = pd.Series(cls).map(lambda v: codes.get(int(v), "outside") if np.isfinite(v) else "outside").values
    # ---------------- A. ground vs the seamless DEM
    D["dem"] = sample(SEAM / f"{ZONE1}_dem_evrf2019_20m.tif", D.x.values, D.y.values); D["dem_src"] = sample(SEAM / f"{ZONE1}_dem_source_20m.tif", D.x.values, D.y.values)
    D["res"] = D.dem - D.H_gedi
    G0 = D[np.isfinite(D.res)].copy(); blunder = np.abs(G0.res) > BLUNDER_M; G = G0[~blunder].copy()
    # "bare" cannot be defined by rh98: GEDI's ground-return width puts bare-ground rh98 at ~2.7 m (p48b), so rh98 < 2 m never happens.
    # Two independent definitions are used instead: the surface class (bare/wet sediment) and a low-canopy arm rh98 < 3 m.
    G["bare_cls"] = G.cls_name.isin(["bare_sand_silt", "wet_sediment", "sparse_herbaceous"]); G["low_rh98"] = G.rh98 < LOW_RH98_M
    print(f"ground set: {len(G0):,} shots on the DEM, {int(blunder.sum()):,} blunders |res| > {BLUNDER_M} m dropped ({blunder.mean():.2%})")
    rows = [metrics(G.res, "all quality shots", arm="free2mean" if not args.no_free2mean else "no_free2mean")]
    base = G[G.bare_cls & G.night & G.power]
    rows.append(metrics(base.res, "bare/sparse class + night + power  [comparable to p57 ICESat-2 night ground]"))
    rows.append(metrics(G[G.bare_cls].res, "bare/sparse class, all beams"))
    rows.append(metrics(G[G.low_rh98].res, f"low canopy rh98 < {LOW_RH98_M} m"))
    rows.append(metrics(G[G.rh98 >= WOODY_M].res, f"woody rh98 >= {WOODY_M} m"))
    for k, g in G.groupby("night"):
        rows.append(metrics(g[g.bare_cls].res, f"bare/sparse, {'night' if k else 'day'}"))
    for k, g in G.groupby("power"):
        rows.append(metrics(g[g.bare_cls].res, f"bare/sparse, {'power beam' if k else 'coverage beam'}"))
    for k, g in G.groupby("year"):
        rows.append(metrics(g[g.bare_cls & g.night & g.power].res, f"bare/sparse+night+power, {k}"))
    G["sens_bin"] = pd.cut(G.sensitivity, [0.9, 0.95, 0.98, 1.0])
    for k, g in G[G.bare_cls & G.night & G.power].groupby("sens_bin", observed=True):
        rows.append(metrics(g.res, f"bare/sparse+night+power, sensitivity {k}"))
    for k, g in G[G.night & G.power].groupby("dem_src"):
        rows.append(metrics(g[g.bare_cls].res, f"bare/sparse+night+power, DEM source {SRC_NAME.get(int(k), k)}"))
    for k, g in G[G.night & G.power].groupby("cls_name"):
        if len(g) >= 200:
            rows.append(metrics(g.res, f"night+power, class {k}"))
    # FABDEM as the DEM-independent control: where the seamless DEM keeps FABDEM (source 3) the two must agree exactly, so
    # FABDEM - GEDI isolates GEDI's own error, and (seamless - FABDEM) isolates what p55 changed. Source 5 (gap fill) and 4
    # (feather band) are the only places where the seamless DEM departs from FABDEM outside the bed.
    G["fab"] = sample(CFG.BULK_ROOT / "terrain" / ZONE1 / "fabdem_evrf2019_20m.tif", G.x.values, G.y.values)
    G["fab_res"] = G.fab - G.H_gedi; G["dem_minus_fab"] = G.dem - G.fab
    for k, g in G[np.isfinite(G.fab)].groupby("dem_src"):
        rows.append(dict(stratum=f"CONTROL FABDEM - GEDI, DEM source {SRC_NAME.get(int(k), k)}", n=len(g),
                         bias=round(float(g.fab_res.mean()), 3), median=round(float(g.fab_res.median()), 3),
                         MAE=round(float(g.fab_res.abs().mean()), 3), RMSE=round(float(np.sqrt(np.mean(g.fab_res ** 2))), 3),
                         LE90=round(float(np.percentile(g.fab_res.abs(), 90)), 3), LE95=round(float(np.percentile(g.fab_res.abs(), 95)), 3),
                         NMAD=round(float(1.4826 * np.median(np.abs(g.fab_res - g.fab_res.median()))), 3),
                         seamless_minus_FABDEM_median=round(float(g.dem_minus_fab.median()), 2), area_km2=SRC_AREA.get(int(k), np.nan)))
    A = pd.DataFrame(rows); A.to_csv(CFG.TABLES / "p62_gedi_ground_accuracy.csv", index=False)
    print("\nA. GEDI ground vs seamless DEM (residual = DEM - GEDI, m EVRF2019):"); print(A.to_string(index=False))
    # ---------------- B. canopy per Manning class
    C = D[D.power & np.isfinite(D.cls)].copy()
    rows = []
    for (yr, season), g0 in C.groupby(["year", C.leafon.map({True: "leafon", False: "other"})]):
        for name, g in g0.groupby("cls_name"):
            if len(g) < 30:
                continue
            rows.append(dict(year=yr, season=season, manning_class=name, code=int(g.cls.iloc[0]), n_shots=len(g), night_share=round(float(g.night.mean()), 2),
                             rh98_p25=q(g.rh98, 25), rh98_p50=q(g.rh98, 50), rh98_p75=q(g.rh98, 75), rh98_p90=q(g.rh98, 90), rh98_p95=q(g.rh98, 95),
                             share_rh98_ge4m=round(float((g.rh98 >= WOODY_M).mean()), 3), rh98_p50_woody=q(g.rh98[g.rh98 >= WOODY_M], 50),
                             rh75_p50=q(g.rh75, 50), rh50_p50=q(g.rh50, 50), ratio_rh50_rh98=round(float(np.nanmedian(g.rh50 / g.rh98.where(g.rh98 > 1))), 2),
                             ratio_rh75_rh98=round(float(np.nanmedian(g.rh75 / g.rh98.where(g.rh98 > 1))), 2)))
    B = pd.DataFrame(rows).sort_values(["year", "season", "code"]); B.to_csv(CFG.TABLES / "p62_gedi_canopy_by_class.csv", index=False)
    # matched acquisition window: GEDI 2025 stops on 09.07, so a raw 2024-vs-2025 comparison mixes growth with season/sampling.
    # Both years are therefore also compared on 01.06-09.07 only.
    W = C[pd.to_datetime(C.time).dt.strftime("%m-%d").between(MATCH_WINDOW[0], MATCH_WINDOW[1])]
    mw = []
    for (yr, name), g in W.groupby(["year", "cls_name"]):
        if len(g) >= 100:
            mw.append(dict(year=yr, manning_class=name, n_shots=len(g), rh98_p50=q(g.rh98, 50), rh98_p90=q(g.rh98, 90), share_rh98_ge4m=round(float((g.rh98 >= WOODY_M).mean()), 3)))
    MW = pd.DataFrame(mw)
    if len(MW):
        piv = MW.pivot(index="manning_class", columns="year", values=["rh98_p50", "share_rh98_ge4m", "n_shots"])
        MW.to_csv(CFG.TABLES / "p62_gedi_canopy_matched_window.csv", index=False)
        print(f"\nB2. matched window {MATCH_WINDOW[0]}..{MATCH_WINDOW[1]} in both years (removes the season/sampling confound):"); print(piv.round(2).to_string())
    print("\nB. GEDI canopy per Manning class (power beams; leaf-on rows):")
    print(B[B.season == "leafon"][["year", "manning_class", "n_shots", "rh98_p50", "rh98_p90", "share_rh98_ge4m", "rh98_p50_woody", "ratio_rh50_rh98"]].to_string(index=False))
    # ---------------- C. GEDI vs ICESat-2 on the same 250 m cells
    # The two sensors' HEIGHTS are not directly comparable: ATL08 h_canopy is 0 where no canopy is detected (94 % of cells), while GEDI rh98
    # cannot go below its ~2.7 m ground-return floor. The comparable quantity is the WOODY SHARE of footprints per cell, each with its own
    # sensor-appropriate threshold (GEDI rh98 >= 4 m, ICESat-2 h_canopy >= 2 m), which is what the p48b/p47 share rasters already carry (%).
    def rd(name):
        p = RGH / f"zone1_poolwin_{name}_250m.tif"
        if not p.exists():
            return None, None
        with rasterio.open(p) as ds:
            return ds.read(1, masked=True).astype("f8").filled(np.nan), ds.transform
    rows, cells = [], {}
    for yr in (2024, 2025):
        g_rh, tr = rd(f"gedi_rh98_mean_{yr}"); g_w, _ = rd(f"gedi_woody_share_{yr}"); g_n, _ = rd(f"gedi_n_shots_{yr}")
        i_s, _ = rd(f"icesat2_lowcnf_allseason_overgrowth_share_{yr}"); i_n, _ = rd(f"icesat2_lowcnf_allseason_n_segments_{yr}"); i_h, _ = rd(f"icesat2_lowcnf_allseason_canopy_h_mean_{yr}")
        if g_rh is None or i_s is None:
            continue
        ok = np.isfinite(g_rh) & np.isfinite(i_s) & (g_n >= MIN_SHOTS) & (i_n >= MIN_SEGS)
        if ok.sum() < 20:
            rows.append(dict(year=yr, n_cells=int(ok.sum()), note="too few co-located cells")); continue
        rr, cc = np.where(ok)
        P = pd.DataFrame(dict(year=yr, row=rr, col=cc, x=tr.c + (cc + 0.5) * tr.a, y=tr.f + (rr + 0.5) * tr.e,
                              gedi_rh98=g_rh[ok], gedi_woody=g_w[ok], gedi_n=g_n[ok], icesat2_woody=i_s[ok], icesat2_h=i_h[ok], icesat2_n=i_n[ok]))
        cells[yr] = P
        rho = spearmanr(P.gedi_woody, P.icesat2_woody).statistic
        both = (P.gedi_woody >= WOODY_CELL_PCT) & (P.icesat2_woody >= WOODY_CELL_PCT); neither = (P.gedi_woody < WOODY_CELL_PCT) & (P.icesat2_woody < WOODY_CELL_PCT)
        rows.append(dict(year=yr, n_cells=len(P), gedi_n_p50=q(P.gedi_n, 50), icesat2_n_p50=q(P.icesat2_n, 50),
                         gedi_woody_pct_p50=q(P.gedi_woody, 50), icesat2_woody_pct_p50=q(P.icesat2_woody, 50),
                         gedi_cells_woody=round(float((P.gedi_woody >= WOODY_CELL_PCT).mean()), 3), icesat2_cells_woody=round(float((P.icesat2_woody >= WOODY_CELL_PCT).mean()), 3),
                         spearman_share=round(float(rho), 3), agreement=round(float((both | neither).mean()), 3),
                         gedi_rh98_p50=q(P.gedi_rh98, 50), icesat2_h_p50_woody_cells=q(P.icesat2_h[P.icesat2_woody >= WOODY_CELL_PCT], 50)))
    Cc = pd.DataFrame(rows); Cc.to_csv(CFG.TABLES / "p62_gedi_vs_icesat2_cells.csv", index=False)
    print(f"\nC. GEDI vs ICESat-2 woody share, same 250 m cells (GEDI >= {MIN_SHOTS} shots, ICESat-2 >= {MIN_SEGS} segments; a cell counts as woody at >= {WOODY_CELL_PCT} %):")
    print(Cc.to_string(index=False))
    pair_frames = list(cells.values())
    # ---------------- D. growth 2024 -> 2025, paired cells (same grid indices)
    growth = pd.DataFrame()
    if len(cells) == 2:
        M = cells[2024].merge(cells[2025], on=["row", "col"], suffixes=("_2024", "_2025"))
        print(f"\nD. cells observed in both years: {len(M)}")
        if len(M) < 20:
            pd.DataFrame([dict(stratum="paired-cell growth", n=len(M), verdict="NOT_IDENTIFIABLE: GEDI has no exact repeat tracks, so the 2024 and 2025 footprints do not revisit the same 250 m cells; use the matched-window distributions (p62_gedi_canopy_matched_window.csv) instead")]).to_csv(CFG.TABLES / "p62_gedi_growth_paired.csv", index=False)
            print("   verdict: NOT_IDENTIFIABLE (no repeat tracks) -> growth is read from the matched-window distributions, not from paired cells")
        if len(M) >= 20:
            M["d_rh98"] = M.gedi_rh98_2025 - M.gedi_rh98_2024; M["d_woody"] = M.gedi_woody_2025 - M.gedi_woody_2024; M["d_icesat2_woody"] = M.icesat2_woody_2025 - M.icesat2_woody_2024
            M["cls"] = sample(RGH / "zone1_poolwin_manning_classes_STATE_2025.tif", M.x_2024.values, M.y_2024.values)
            M["cls_name"] = pd.Series(M.cls).map(lambda v: codes.get(int(v), "outside") if np.isfinite(v) else "outside").values
            g = [dict(stratum="all paired cells", n=len(M), rh98_2024_p50=q(M.gedi_rh98_2024, 50), rh98_2025_p50=q(M.gedi_rh98_2025, 50),
                      d_rh98_median=round(float(M.d_rh98.median()), 2), d_rh98_p25=q(M.d_rh98, 25), d_rh98_p75=q(M.d_rh98, 75),
                      woody_pct_2024=q(M.gedi_woody_2024, 50), woody_pct_2025=q(M.gedi_woody_2025, 50), d_woody_median=round(float(M.d_woody.median()), 1),
                      d_icesat2_woody_median=round(float(M.d_icesat2_woody.median()), 1))]
            for k, gg in M.groupby("cls_name"):
                if len(gg) >= MIN_CELLS_CLASS:
                    g.append(dict(stratum=k, n=len(gg), rh98_2024_p50=q(gg.gedi_rh98_2024, 50), rh98_2025_p50=q(gg.gedi_rh98_2025, 50),
                                  d_rh98_median=round(float(gg.d_rh98.median()), 2), d_rh98_p25=q(gg.d_rh98, 25), d_rh98_p75=q(gg.d_rh98, 75),
                                  woody_pct_2024=q(gg.gedi_woody_2024, 50), woody_pct_2025=q(gg.gedi_woody_2025, 50), d_woody_median=round(float(gg.d_woody.median()), 1),
                                  d_icesat2_woody_median=round(float(gg.d_icesat2_woody.median()), 1)))
            growth = pd.DataFrame(g); growth.to_csv(CFG.TABLES / "p62_gedi_growth_paired.csv", index=False)
            print("Paired 250 m cells 2024 -> 2025 (GEDI rh98 mean per cell, m; woody share, %):"); print(growth.to_string(index=False))
    # ---------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax = axes[0, 0]; b = G[G.bare_cls & G.night & G.power]
    for k, g in b.groupby("dem_src"):
        if len(g) >= 200:
            ax.hist(np.clip(g.res, -6, 6), bins=80, histtype="step", lw=1.4, density=True, label=f"{SRC_NAME.get(int(k), k)} (n {len(g):,}, RMSE {np.sqrt(np.mean(g.res ** 2)):.2f})")
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("seamless DEM − GEDI ground, m EVRF2019"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"(a) GEDI ground vs the seamless DEM, bare + night + power (n {len(b):,}; bias {b.res.mean():+.2f}, RMSE {np.sqrt(np.mean(b.res ** 2)):.2f}, LE90 {np.percentile(np.abs(b.res), 90):.2f} m)")
    ax = axes[0, 1]; lo = B[(B.season == "leafon")]
    if len(lo):
        piv = lo.pivot_table(index="manning_class", columns="year", values="rh98_p50"); piv = piv.reindex(piv.mean(axis=1).sort_values().index)
        piv.plot.barh(ax=ax, width=0.8); ax.axvline(2.72, color="grey", ls=":", lw=1.2); ax.text(2.76, len(piv) - 0.4, "bare-ground\nfloor 2.72 m", fontsize=7, color="grey", va="top"); ax.axvline(WOODY_M, color="#b2182b", ls="--", lw=1.2); ax.text(WOODY_M + 0.08, len(piv) - 0.4, "woody\nthreshold 4 m", fontsize=7, color="#b2182b", va="top")
    ax.set_xlabel("GEDI rh98 median, m"); ax.set_ylabel(""); ax.set_title("(b) canopy height per Manning class (leaf-on, power beams)"); ax.grid(alpha=0.3, axis="x")
    ax = axes[1, 0]
    if pair_frames:
        P = pd.concat(pair_frames)
        for yr, g in P.groupby("year"):
            ax.scatter(g.icesat2_woody + np.random.default_rng(CFG.SEED).uniform(-2, 2, len(g)), g.gedi_woody + np.random.default_rng(CFG.SEED + 1).uniform(-2, 2, len(g)),
                       s=9, alpha=0.45, label=f"{yr} (n {len(g)}, ρ {spearmanr(g.gedi_woody, g.icesat2_woody).statistic:.2f})")
        ax.plot([0, 100], [0, 100], "k--", lw=0.8, label="1:1"); ax.axhline(WOODY_CELL_PCT, color="#b2182b", ls=":", lw=1); ax.axvline(WOODY_CELL_PCT, color="#1a9850", ls=":", lw=1)
        ax.set_xlim(-4, 104); ax.set_ylim(-4, 104)
    ax.set_xlabel("ICESat-2 share of segments with h_canopy ≥ 2 m, %"); ax.set_ylabel("GEDI share of shots with rh98 ≥ 4 m, %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("(c) two lidars on the same 250 m cells: woody share (jittered ±2 %)")
    ax = axes[1, 1]
    if len(MW):
        w = MW.pivot(index="manning_class", columns="year", values="share_rh98_ge4m").dropna()
        if len(w):
            w = w.reindex(w[2025].sort_values().index) if 2025 in w else w
            yy = np.arange(len(w)); ax.barh(yy - 0.2, w[2024] * 100, height=0.4, color="#4292c6", label="2024")
            ax.barh(yy + 0.2, w[2025] * 100, height=0.4, color="#e6550d", label="2025")
            ax.set_yticks(yy); ax.set_yticklabels(w.index, fontsize=9); ax.legend(fontsize=8)
    ax.set_xlabel("share of GEDI shots with rh98 ≥ 4 m, %"); ax.grid(alpha=0.3, axis="x")
    ax.set_title(f"(d) same acquisition window {MATCH_WINDOW[0]}–{MATCH_WINDOW[1]} in both years\n(bare and water classes are the null control; paired-cell growth is not identifiable)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p62_gedi_analysis.png", dpi=110); print("-> outputs/figures/p62_gedi_analysis.png")


if __name__ == "__main__":
    main()
