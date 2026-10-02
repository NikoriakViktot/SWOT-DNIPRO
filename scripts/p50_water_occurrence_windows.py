#!/usr/bin/env python
"""P50 -- water-occurrence (SWOM-type) classes of the reservoir bed on several windows, INCLUDING the 2024 spring flood (user 2026-09-19).

Why: p43's `never` (163.3 km2) is 'never exposed during the 2023 drawdown' -- a window that ends before the spring-2024 flood, which re-wetted ~50 % of the bed
(Lischenko, Kozlova, Andreiev 2025, UJRS 12(4)). Their 6-class water-occurrence map covers 2023-06-20 .. 2024-09-26 with a binary NDWI>0 rule. This script
recomputes the same product from our per-date ZONE_1 stacks so that the comparison is like-for-like, and adds a `reflooded_spring2024` layer.

Domain: registry `reservoir_full_pool_prebreach` (SD.load_utm) rasterised on the ZONE_1 20 m grid.
Per date (all $BULK_ROOT/zone_spectral/ZONE_1*/<date>_water3.tif in the window; 255 = not observed and NEVER a dry vote):
  rule "water3" : our frozen water rule (sentinel_preprocess), 0 land / 1 water / 255 not observed
  rule "ndwi0"  : NDWI (band 2 of <date>_indices.tif, int16 x1e4) > 0 -> water, else land; nodata -> not observed   (Lischenko's rule)
Frequency f = n_water / n_valid (n_valid >= MIN_VALID, else 'unclassified'); classes exactly as theirs:
  1: 0 %   2: (0,25]   3: (25,50]   4: (50,75]   5: (75,100)   6: 100 %
Windows: LISCH 2023-06-20..2024-09-26 (theirs) | DRAWDOWN 2023-06-06..2023-09-30 (p43) | POST 2023-06-20..last date
Outputs:
  outputs/rasters/roughness/pool/zone1_water_occurrence_class_<WINDOW>_<rule>_20m.tif (uint8, 0 = unclassified)
  outputs/rasters/roughness/pool/zone1_reflooded_spring2024_20m.tif  (1 = wet in Mar-May 2024 AFTER being observed land in 2023-08..2023-10; 0 = not; 255 = never observed in both)
  outputs/tables/p50_water_occurrence_class_shares.csv   class shares (% of classified pool area) per window/rule + the published Lischenko values
  outputs/tables/p50_water_share_by_date.csv             water % of the OBSERVED pool area per date (+ observed share of the pool): the comparator of '37 % May 2024 -> 8.2 % Sep 2024'
  outputs/tables/p50_never_vs_recession_zones.csv        cross-tab of the occurrence class (LISCH, water3) with the p43 bed-recession zoning
  outputs/figures/p50_water_occurrence.png
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
import rasterio
from rasterio import features
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP
from swot_dnipro import spatial_domains as SD

ZONE = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
STK = CFG.BULK_ROOT / "zone_spectral" / ZONE
RO = ROOT / "outputs/rasters/roughness/pool"
MIN_VALID = 3
WINDOWS = {"LISCH": ("2023-06-20", "2024-09-26"), "DRAWDOWN": ("2023-06-06", "2023-09-30"), "POST": ("2023-06-20", "2026-12-31")}
RULES = ("water3", "ndwi0")
# published (Lischenko et al. 2025, UJRS 12(4), read from the PDF): share of the bed area, %
LISCH = {1: 41.62, 2: 20.62, 3: 23.71, 4: 5.76, 5: 2.5, 6: 6.0}
LABEL = {1: "never water (0 %)", 2: "1-25 %", 3: "25-50 %", 4: "50-75 %", 5: "75-99 %", 6: "always water (100 %)"}


def read_date(d: str, rule: str, shape) -> np.ndarray:
    """uint8: 0 land, 1 water, 255 not observed."""
    if rule == "water3":
        with rasterio.open(STK / f"{d}_water3.tif") as ds:
            return ds.read(1)
    with rasterio.open(STK / f"{d}_indices.tif") as ds:
        a = ds.read(2)                                     # NDWI
    out = np.where(a > 0, 1, 0).astype("u1"); out[a == -32768] = 255
    return out


def classes(nw: np.ndarray, nv: np.ndarray) -> np.ndarray:
    f = np.where(nv > 0, 100.0 * nw / np.maximum(nv, 1), np.nan)
    c = np.zeros(nw.shape, "u1")
    ok = nv >= MIN_VALID
    c[ok & (f == 0)] = 1; c[ok & (f > 0) & (f <= 25)] = 2; c[ok & (f > 25) & (f <= 50)] = 3; c[ok & (f > 50) & (f <= 75)] = 4; c[ok & (f > 75) & (f < 100)] = 5; c[ok & (f == 100)] = 6
    return c


def main():
    G = SP.zone_grid(ZONE, 20.0); shape = (G["ny"], G["nx"])
    pool = features.rasterize([(SD.load_utm("reservoir_full_pool_prebreach"), 1)], out_shape=shape, transform=G["transform"], fill=0, dtype="uint8").astype(bool)
    dates = sorted(p.name[:10] for p in STK.glob("*_water3.tif") if p.name[:4].isdigit() and "2023-06-05" <= p.name[:10] <= "2026-12-31")
    print(f"{len(dates)} ZONE_1 dates {dates[0]} .. {dates[-1]}; pool {pool.sum()*0.0004:.0f} km2", flush=True)
    NW = {(w, r): np.zeros(shape, "u1") for w in WINDOWS for r in RULES}; NV = {(w, r): np.zeros(shape, "u1") for w in WINDOWS for r in RULES}
    wet_spring = np.zeros(shape, bool); obs_spring = np.zeros(shape, bool); land_2023 = np.zeros(shape, bool); obs_2023 = np.zeros(shape, bool)
    rows = []
    for i, d in enumerate(dates, 1):
        for r in RULES:
            a = read_date(d, r, shape)
            obs = (a != 255) & pool; wat = (a == 1) & pool
            rows.append(dict(date=d, rule=r, pool_km2=round(pool.sum() * 0.0004, 1), observed_km2=round(obs.sum() * 0.0004, 1), observed_share=round(float(obs.sum() / pool.sum()), 3), water_km2=round(wat.sum() * 0.0004, 1), water_pct_of_observed=round(100 * float(wat.sum() / max(obs.sum(), 1)), 2)))
            for w, (t0, t1) in WINDOWS.items():
                if t0 <= d <= t1:
                    NV[(w, r)] += obs.astype("u1"); NW[(w, r)] += wat.astype("u1")
            if r == "water3":
                if "2024-03-01" <= d <= "2024-05-31":
                    obs_spring |= obs; wet_spring |= wat
                if "2023-08-01" <= d <= "2023-10-31":
                    obs_2023 |= obs; land_2023 |= ((a == 0) & pool)
        if i % 8 == 0:
            print(f"  {i}/{len(dates)} {d}", flush=True)
    ts = pd.DataFrame(rows); ts.to_csv(CFG.TABLES / "p50_water_share_by_date.csv", index=False)
    prof = SP._profile(G, 1, "uint8", 0)
    shares = []; cls_lisch = None
    for w in WINDOWS:
        for r in RULES:
            c = classes(NW[(w, r)], NV[(w, r)]); c[~pool] = 0
            with rasterio.open(RO / f"zone1_water_occurrence_class_{w}_{r}_20m.tif", "w", **prof) as ds:
                ds.write(c, 1); ds.update_tags(values="0 unclassified (n_valid<3 or outside pool); 1 never 0%; 2 (0,25]; 3 (25,50]; 4 (50,75]; 5 (75,100); 6 always 100%", window="..".join(WINDOWS[w]), rule=r, producer="p50_water_occurrence_windows.py")
            tot = int((c > 0).sum()); n_dates = int(NV[(w, r)][pool].max())
            row = dict(window=w, rule=r, dates_in_window=sum(WINDOWS[w][0] <= d <= WINDOWS[w][1] for d in dates), classified_km2=round(tot * 0.0004, 1), unclassified_km2=round(float((pool & (c == 0)).sum()) * 0.0004, 1))
            for k in range(1, 7):
                row[f"class{k}_pct"] = round(100 * float((c == k).sum()) / max(tot, 1), 2)
            shares.append(row)
            if (w, r) == ("LISCH", "water3"):
                cls_lisch = c
    lisch_row = dict(window="LISCH_PUBLISHED", rule="NDWI>0 (12 scenes, GEE)", dates_in_window=12, classified_km2=np.nan, unclassified_km2=np.nan, **{f"class{k}_pct": v for k, v in LISCH.items()})
    S = pd.DataFrame(shares + [lisch_row]); S.to_csv(CFG.TABLES / "p50_water_occurrence_class_shares.csv", index=False); print(S.to_string(index=False))
    refl = np.full(shape, 255, "u1"); ok = obs_spring & obs_2023; refl[ok] = (wet_spring & land_2023)[ok].astype("u1"); refl[~pool] = 255
    with rasterio.open(RO / "zone1_reflooded_spring2024_20m.tif", "w", **SP._profile(G, 1, "uint8", 255)) as ds:
        ds.write(refl, 1); ds.update_tags(values="1 wet in Mar-May 2024 after being observed land in Aug-Oct 2023; 0 not; 255 unobserved in one of the periods", producer="p50_water_occurrence_windows.py")
    print(f"re-flooded in spring 2024: {(refl == 1).sum()*0.0004:.0f} km2 of {(refl != 255).sum()*0.0004:.0f} km2 observed in both periods (pool {pool.sum()*0.0004:.0f} km2)")
    zone = RO / "zone1_poolwin_bed_recession_zoning_2023.tif"
    with rasterio.open(zone) as ds:
        zr = ds.read(1); ztr = ds.transform
    ys, xs = np.nonzero(cls_lisch > 0); cx, cy = G["transform"] * (xs + 0.5, ys + 0.5)
    r_, c_ = rasterio.transform.rowcol(ztr, cx, cy); r_ = np.asarray(r_); c_ = np.asarray(c_); okz = (r_ >= 0) & (r_ < zr.shape[0]) & (c_ >= 0) & (c_ < zr.shape[1])
    zz = np.zeros(len(xs), "u1"); zz[okz] = zr[r_[okz], c_[okz]]
    xt = pd.crosstab(pd.Series(cls_lisch[ys, xs], name="occurrence_class_LISCH_water3"), pd.Series(zz, name="p43_recession_zone(1 fast,2 mid,3 slow,4 never,0 outside)")) * 0.0004
    xt.round(1).to_csv(CFG.TABLES / "p50_never_vs_recession_zones.csv"); print(xt.round(1).to_string())
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    ax = axes[0]; k = np.arange(1, 7); w = 0.27
    a1 = S[(S.window == "LISCH") & (S.rule == "water3")].iloc[0]; a2 = S[(S.window == "LISCH") & (S.rule == "ndwi0")].iloc[0]; a3 = S[(S.window == "DRAWDOWN") & (S.rule == "water3")].iloc[0]
    ax.bar(k - w, [LISCH[i] for i in k], w, color="#888", label="Lischenko 2025 (published, 12 scenes)"); ax.bar(k, [a2[f"class{i}_pct"] for i in k], w, color="#2171b5", label=f"ours, same window & rule NDWI>0 ({int(a2.dates_in_window)} dates)")
    ax.bar(k + w, [a3[f"class{i}_pct"] for i in k], w, color="#d94801", label="ours, 2023 drawdown only (p43 window; excludes the 2024 flood)")
    ax.set_xticks(k); ax.set_xticklabels([LABEL[i] for i in k], rotation=25, ha="right"); ax.set_ylabel("% of classified bed area"); ax.set_title("(a) water-occurrence classes: window matters"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]; t = ts[(ts.rule == "water3") & (ts.observed_share >= 0.85)].copy(); t["date"] = pd.to_datetime(t.date)
    ax.plot(t.date, t.water_pct_of_observed, "-o", ms=3, color="#2171b5", label="water3 (our rule)"); t2 = ts[(ts.rule == "ndwi0") & (ts.observed_share >= 0.85)].copy(); t2["date"] = pd.to_datetime(t2.date); ax.plot(t2.date, t2.water_pct_of_observed, "-s", ms=3, color="#e6550d", label="NDWI>0 (their rule)")
    ax.axvline(pd.Timestamp("2023-06-06"), color="red", ls="--", lw=0.8); ax.set_ylabel("water % of the OBSERVED pool area (dates with ≥ 85 % observed)"); ax.set_title("(b) water share of the bed by date (published: 37 % May 2024 → 8.2 % Sep 2024)"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[2]; ex = [G["x0"], G["x0"] + G["nx"] * 20, G["y1"] - G["ny"] * 20, G["y1"]]; k8 = 8
    c8 = cls_lisch[::k8, ::k8].astype("f4"); c8[c8 == 0] = np.nan
    ax.imshow(c8, extent=ex, cmap=ListedColormap(["#f7f7f7", "#c6dbef", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]), vmin=0.5, vmax=6.5, interpolation="nearest")
    ax.set_xlim(500e3, 690e3); ax.set_ylim(5140e3, 5315e3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(c) occurrence class, window 2023-06-20 … 2024-09-26, rule water3 (1 never … 6 always)")
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p50_water_occurrence.png", dpi=95); print("-> outputs/figures/p50_water_occurrence.png")


if __name__ == "__main__":
    main()
