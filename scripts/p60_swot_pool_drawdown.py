#!/usr/bin/env python
"""P60 -- SWOT KaRIn over the RESERVOIR itself in 2023: what exists during the June drawdown, what exists afterwards, all tied to EVRF2019
(user 2026-09-19: "плюс якщо є на самому водосховищі ... якщо є озерні їх потрібно виловити ... увязати до EVRF2019").

Coverage fact established first (scan of every LakeSP Obs/Prior/Unassigned and RiverSP Node file, May-Dec 2023, against the registry pool):
  * the 1-day cal/val orbit (pass 001, daily to 10.07.2023) crosses the Dnipro at the DAM and runs down the lower river; east of the dam it does
    NOT cover the pool -> ZERO LakeSP objects intersect the pool from 25.05 to 28.07.2023, and only 4 RiverSP nodes (reach 22511300110, the
    dam reach, chainage -0.6..0 km) are inside the pool polygon. The PIXC frame of part3 (pass 001, 04.04-05.06.2023) is 32.53-32.69E, i.e.
    the lower river near Kherson, not the pool (17_HECRAS row 'initial level: 559 k points over the pool' was wrong -> corrected here).
  * the science orbit reaches the pool from 29.07.2023 (passes 027, 206, 234, 305, 512, 540); LakeSP splits the emptied reservoir into
    several 'KAKHOVKA RESERVOIR' objects (50-720 km2).
So the June drawdown of the reservoir is seen by SWOT at ONE place -- the outlet -- as a daily series; the pool interior in June has no SWOT.
Vertical chain (same as p59): H_EVRF2019 = wse + geoid_hght (-> ellipsoid) + free2mean(lat) - zeta_EGG2015 + c (c = mean of the six pool
stations, CFG.CORRECTOR_BY_STATION).
A  RiverSP nodes inside the pool (+1 km), node_q <= 1, dark_frac < 0.5, 25.05-31.12.2023. Chainage s_up = dist_out - dist_out(dam node)
   (km, positive upstream). 'outlet' = -1 <= s_up <= 1 km; 'pool' = s_up > 1 km (science orbit only).
   -> p60_swot_outlet_drawdown.csv: daily outlet H (median of the outlet nodes, pass UTC time) with Rozumivka gauge (k5, upstream end of
      the pool, the only station reporting after 2021) and the p59 0-5 km bin below the dam -> the head drop through the breach.
   -> p60_swot_pool_profiles.csv: 10-km bin medians of H along the former pool per science-orbit date (residual river profile).
B  LakeSP Obs objects intersecting the pool: one wse per object -> H_EVRF2019; chainage of the centroid via the RiverSP node channel
   (sword.assign_chainage). Objects > LARGE_KM2 are flagged LARGE_MIXED: one number averaged over a sloping river tens of km long is not
   a level and is excluded from profiles (kept in the table).
   -> p60_swot_pool_lakes.csv
C  Validation: SWOT (nodes and small lakes) within 5 km of Rozumivka vs the gauge, by date -> p60_swot_pool_vs_gauges.csv.
D  Coverage register -> p60_swot_pool_coverage.csv (product x date x pass x n objects/nodes in the pool).
Figure: outputs/figures/p60_swot_pool_drawdown.png
"""
from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro import vertical as VT
from swot_dnipro import sword as SW

RIV = CFG.BULK_ROOT / "swot_ua/swot_l2_hr_riversp_2.0/2023"; LAK = CFG.BULK_ROOT / "swot_ua/swot_l2_hr_lakesp_2.0/2023"
DATES = ("20230525", "20231231"); MONTHS = [f"{m:02d}" for m in range(5, 13)]; BIN_KM = 10.0; LARGE_KM2 = 100.0; OUTLET_KM = 1.0; GAUGE_R_M = 5000.0
FILL = -1e11


def fdate(f): return Path(f).name.split("_")[8][:8]
def fpass(f): return Path(f).name.split("_")[6]


def chain(df, c):
    df["zeta"] = VT.sample_grid(CFG.EGG2015_TIF, df.lon.values, df.lat.values); df["H_evrf"] = df.wse + df.geoid_hght + CFG.free2mean(df.lat.values) - df.zeta + c; return df


def files(root, pattern):
    out = sorted(f for m in MONTHS for f in glob.glob(str(root / m / pattern))); return [f for f in out if DATES[0] <= fdate(f) <= DATES[1]]


def main():
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean()); tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)
    pool = SD.load_utm("reservoir_full_pool_prebreach"); pool_ll = gpd.GeoSeries([pool], crs=CFG.CRS_METRIC).to_crs("EPSG:4326").iloc[0]; pool_buf = pool_ll.buffer(0.012); pb = pool_buf.bounds
    dam = np.array(tf.transform(*CFG.KAKHOVKA_DAM)); cov = []
    # ---------------- A. RiverSP nodes in the pool (+1 km), all passes May-Dec 2023
    parts, chan = [], []
    for f in files(RIV, "*RiverSP_Node*.shp"):
        g = gpd.read_file(f, bbox=pb)
        if not len(g):
            continue
        g = g[shapely.contains_xy(pool_buf, g.geometry.x.values, g.geometry.y.values)]
        n_in = len(g)
        # channel geometry for chainage: EVERY main-stem node of the swath (SWORD prior), valid or not -- validity is irrelevant for position
        m = g[g.river_name.astype(str).str.contains("Dnipro|Dnieper|no_data", regex=True) & (g.p_dist_out > 0)]
        chan.append(pd.DataFrame(dict(node_id=m.node_id.values, reach_id=m.reach_id.values, lon=m.geometry.x.values, lat=m.geometry.y.values, dist_out=m.p_dist_out.values)))
        g = g[(g.wse > FILL) & (g.node_q <= 1) & (g.dark_frac < 0.5) & (g.geoid_hght > FILL)]
        if n_in:
            cov.append(dict(product="RiverSP_Node", date=fdate(f), pas=fpass(f), n_in_pool=n_in, n_valid=len(g)))
        if len(g):
            g = g.assign(date=pd.Timestamp(fdate(f)), pas=fpass(f), lon=g.geometry.x, lat=g.geometry.y); parts.append(pd.DataFrame(g.drop(columns="geometry")))
    N = chain(pd.concat(parts, ignore_index=True), c); N["x"], N["y"] = tf.transform(N.lon.values, N.lat.values); N["river_name"] = N.river_name.astype(str)
    d_dam = np.hypot(N.x - dam[0], N.y - dam[1]); dist_out_dam = float(N.loc[d_dam.idxmin(), "p_dist_out"]); N["s_up_km"] = (N.p_dist_out - dist_out_dam) / 1e3
    N = N[N.s_up_km > -OUTLET_KM].copy(); N["where"] = np.where(N.s_up_km <= OUTLET_KM, "outlet", "pool"); N["utc"] = N.time_str.where(N.time_str != "no_data")
    print(f"pool RiverSP: {len(N):,} node obs, {N.date.nunique()} dates, {N.node_id.nunique()} nodes; outlet obs {int((N['where']=='outlet').sum())}, pool obs {int((N['where']=='pool').sum())} "
          f"(first pool date {N[N['where']=='pool'].date.min().date() if (N['where']=='pool').any() else None}); passes {sorted(N.pas.unique())}")
    N[["date", "pas", "utc", "reach_id", "node_id", "river_name", "where", "lon", "lat", "x", "y", "s_up_km", "p_dist_out", "wse", "wse_u", "geoid_hght", "zeta", "H_evrf", "width", "node_q", "dark_frac"]].to_parquet(CFG.TABLES / "p60_swot_pool_nodes.parquet", index=False)
    # channel for lake chainage: every main-stem SWORD node of every swath that touched the pool, deduplicated, chainage relative to the dam node
    ch = pd.concat(chan, ignore_index=True).groupby("node_id").agg(lon=("lon", "median"), lat=("lat", "median"), dist_out=("dist_out", "median"), reach_id=("reach_id", "first")).reset_index()
    ch["chain_km"] = (ch.dist_out - dist_out_dam) / 1e3; ch = ch[ch.chain_km > -OUTLET_KM]; tree = SW.build_chainage_tree(ch)
    print(f"channel for chainage: {len(ch)} SWORD nodes, {ch.chain_km.min():.1f}..{ch.chain_km.max():.1f} km upstream of the dam")
    # outlet drawdown series vs Rozumivka and vs p59 0-5 km bin below the dam
    out = N[N["where"] == "outlet"].groupby("date").agg(swot_n=("H_evrf", "size"), H_outlet=("H_evrf", "median"), H_outlet_min=("H_evrf", "min"), H_outlet_max=("H_evrf", "max"), utc=("utc", "first"), pas=("pas", "first")).reset_index()
    k5 = pd.read_csv(CFG.TABLES / "k5_gauge_levels_evrf2019.csv", usecols=["station_name", "x_utm", "y_utm", "date", "H_EVRF2019_m"]); k5["date"] = pd.to_datetime(k5.date); k5 = k5[(k5.date >= DATES[0]) & (k5.date <= DATES[1])]
    roz = k5[k5.station_name == "Rozumivka"].groupby("date").H_EVRF2019_m.agg(H_rozumivka="mean", roz_terms="size").reset_index()  # 08h/20h terms -> daily mean
    p59 = pd.read_csv(CFG.TABLES / "p59_swot_flood_profiles.csv", parse_dates=["date"]); p59 = p59[p59.bin_km == 0][["date", "p50"]].rename(columns={"p50": "H_below_dam_0_5km_p59"})
    out = out.merge(roz, on="date", how="left").merge(p59, on="date", how="left"); out["head_pool_gradient"] = out.H_rozumivka - out.H_outlet; out["head_drop_breach"] = out.H_outlet - out.H_below_dam_0_5km_p59
    out.to_csv(CFG.TABLES / "p60_swot_outlet_drawdown.csv", index=False)
    print("outlet drawdown (m EVRF2019):"); print(out[(out.date >= "2023-05-30") & (out.date <= "2023-07-01")][["date", "utc", "swot_n", "H_outlet", "H_rozumivka", "H_below_dam_0_5km_p59", "head_pool_gradient", "head_drop_breach"]].round(2).to_string(index=False))
    # pool profiles (science orbit)
    P = N[N["where"] == "pool"].copy(); P["bin_km"] = (np.floor(P.s_up_km / BIN_KM) * BIN_KM).astype(int)
    prof = P.groupby(["date", "pas", "bin_km"]).H_evrf.agg(n="size", p25=lambda v: v.quantile(.25), p50="median", p75=lambda v: v.quantile(.75)).reset_index(); prof = prof[prof.n >= 3]; prof.to_csv(CFG.TABLES / "p60_swot_pool_profiles.csv", index=False)
    if len(prof):
        print("pool profiles, science orbit (10-km bin medians, m EVRF2019):"); print(prof.pivot_table(index="date", columns="bin_km", values="p50").round(1).iloc[:, :14].to_string())
    # ---------------- B. LakeSP Obs objects intersecting the pool
    lp = []
    for f in files(LAK, "*LakeSP_Obs*.shp"):
        g = gpd.read_file(f, bbox=pb)
        if not len(g):
            continue
        g = g[g.intersects(pool_ll)]
        if not len(g):
            continue
        cov.append(dict(product="LakeSP_Obs", date=fdate(f), pas=fpass(f), n_in_pool=len(g), n_valid=int(((g.wse > FILL) & (g.geoid_hght > FILL)).sum())))
        g = g[(g.wse > FILL) & (g.geoid_hght > FILL)].copy(); g["date"] = pd.Timestamp(fdate(f)); g["pas"] = fpass(f); cen = g.geometry.representative_point(); g["lon"], g["lat"] = cen.x, cen.y
        g["area_in_pool_km2"] = gpd.GeoSeries(g.geometry.intersection(pool_ll), crs="EPSG:4326").to_crs(CFG.CRS_METRIC).area.values / 1e6
        lp.append(pd.DataFrame(g.drop(columns="geometry")))
    L = chain(pd.concat(lp, ignore_index=True), c); L["lake_name"] = L.lake_name.astype(str); L["x"], L["y"] = tf.transform(L.lon.values, L.lat.values)
    L["s_up_km"], L["dist_to_channel_km"], _, _ = SW.assign_chainage(L.lon.values, L.lat.values, ch, max_dist_km=50.0, tree=tree); L["size_class"] = np.where(L.area_total > LARGE_KM2, "LARGE_MIXED", "small")
    L["in_pool_frac"] = (L.area_in_pool_km2 / L.area_total).clip(0, 1)
    # usable as a LEVEL of the residual water in the former pool: small, mostly inside the pool, good quality, not a wet-mud sliver
    L["level_ok"] = (L.size_class == "small") & (L.in_pool_frac >= 0.5) & (L.quality_f == 0) & (L.dark_frac < 0.5) & (L.area_detct >= 0.5)
    L = L[["date", "pas", "time_str", "lake_id", "lake_name", "size_class", "level_ok", "area_total", "area_detct", "area_in_pool_km2", "in_pool_frac", "lon", "lat", "x", "y", "s_up_km", "dist_to_channel_km", "wse", "wse_u", "wse_std", "quality_f", "partial_f", "dark_frac", "geoid_hght", "zeta", "H_evrf"]].sort_values(["date", "area_total"], ascending=[True, False])
    L.to_csv(CFG.TABLES / "p60_swot_pool_lakes.csv", index=False)
    print(f"LakeSP Obs in pool: {len(L):,} objects on {L.date.nunique()} dates (first {L.date.min().date()}), passes {sorted(L.pas.unique())}; LARGE_MIXED {int((L.size_class=='LARGE_MIXED').sum())}, small {int((L.size_class=='small').sum())}, level_ok {int(L.level_ok.sum())}")
    ok = L[L.level_ok]; print("level_ok objects by date:"); print(ok.groupby("date").agg(pas=("pas", "first"), n=("lake_id", "size"), area_sum=("area_total", "sum"), H_p10=("H_evrf", lambda v: v.quantile(.1)), H_p50=("H_evrf", "median"), H_p90=("H_evrf", lambda v: v.quantile(.9)), s_min=("s_up_km", "min"), s_max=("s_up_km", "max")).round(1).to_string())
    big = L[L.size_class == "LARGE_MIXED"]; print("LARGE_MIXED objects (one wse over a sloping residual river -- not a level):"); print(big.groupby("date").agg(pas=("pas", "first"), n=("lake_id", "size"), area_max=("area_total", "max"), H_min=("H_evrf", "min"), H_max=("H_evrf", "max"), s_min=("s_up_km", "min"), s_max=("s_up_km", "max")).round(1).to_string())
    # ---------------- C. validation near Rozumivka (nodes + small lakes within 5 km)
    sx, sy = k5[k5.station_name == "Rozumivka"][["x_utm", "y_utm"]].iloc[0]
    nn = N[np.hypot(N.x - sx, N.y - sy) < GAUGE_R_M].groupby("date").H_evrf.agg(swot_node_n="size", swot_node_p50="median").reset_index()
    ll = L[L.level_ok & (np.hypot(L.x - sx, L.y - sy) < GAUGE_R_M)].groupby("date").H_evrf.agg(swot_lake_n="size", swot_lake_p50="median").reset_index()
    G = roz.merge(nn, on="date", how="outer").merge(ll, on="date", how="outer").sort_values("date"); G["diff_node"] = G.swot_node_p50 - G.H_rozumivka; G["diff_lake"] = G.swot_lake_p50 - G.H_rozumivka; G.to_csv(CFG.TABLES / "p60_swot_pool_vs_gauges.csv", index=False)
    v = G.dropna(subset=["diff_node"]); w = G.dropna(subset=["diff_lake"])
    print(f"SWOT vs Rozumivka: nodes n={len(v)} median diff {v.diff_node.median():+.2f} MAD {(v.diff_node - v.diff_node.median()).abs().median():.2f} | small lakes n={len(w)} median diff {w.diff_lake.median():+.2f}" if len(v) or len(w) else "SWOT vs Rozumivka: no co-located observations")
    # ---------------- D. coverage register
    C = pd.DataFrame(cov).sort_values(["product", "date", "pas"]); C.to_csv(CFG.TABLES / "p60_swot_pool_coverage.csv", index=False)
    jun = C[(C.date < "20230729")]; print(f"coverage before 29.07.2023: RiverSP files touching pool {int((jun['product']=='RiverSP_Node').sum())} (all pass 001, outlet nodes only), LakeSP objects in pool {int(jun[jun['product']=='LakeSP_Obs'].n_in_pool.sum())}")
    # ---------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)); ax = axes[0]; o = out[(out.date >= "2023-05-27") & (out.date <= "2023-07-10")]
    ax.plot(o.date, o.H_outlet, "-o", color="#b2182b", ms=4, label="SWOT at the outlet (dam reach, chainage −0.6…0 km)"); ax.fill_between(o.date, o.H_outlet_min, o.H_outlet_max, color="#b2182b", alpha=0.2)
    ax.plot(o.date, o.H_rozumivka, "k-s", ms=3, label="Rozumivka gauge (upstream end of the pool, k5)"); ax.plot(o.date, o.H_below_dam_0_5km_p59, "-^", color="#2166ac", ms=3, label="SWOT 0–5 km below the dam (p59)")
    ax.axvline(pd.Timestamp(CFG.BREACH_DATE), color="grey", ls="--", lw=1); ax.set_ylabel("m EVRF2019"); ax.set_title("June 2023: the reservoir drawdown at the only place SWOT sees it — the outlet"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)
    ax = axes[1]; ev = sorted(prof.date.unique()); cmap = plt.get_cmap("viridis")
    for i, d in enumerate(ev):
        s = prof[prof.date == d].sort_values("bin_km"); ax.plot(s.bin_km + BIN_KM / 2, s.p50, "-o", ms=2.5, color=cmap(i / max(len(ev) - 1, 1)), lw=1.1, label=pd.Timestamp(d).strftime("%m-%d"))
    sm = L[L.level_ok].dropna(subset=["s_up_km"]); ax.scatter(sm.s_up_km, sm.H_evrf, s=np.clip(sm.area_total * 2, 6, 80), facecolors="none", edgecolors="#d95f02", lw=0.8, label="LakeSP objects, level-quality (< 100 km², ≥ 50 % in pool, quality_f 0)")
    ax.set_ylim(-2, 20)
    ax.set_xlim(0, 255); ax.axhline(17.4, color="grey", ls=":", lw=1); ax.text(2, 17.55, "pre-breach pool 17.4 m", fontsize=7, color="grey")
    ax.set_xlabel("chainage upstream of the dam along SWORD, km"); ax.set_ylabel("m EVRF2019"); ax.set_title("Residual river in the former pool, science orbit from 29.07.2023 (RiverSP 10-km bins; LakeSP small objects)"); ax.legend(fontsize=6.5, ncol=3); ax.grid(alpha=0.3)
    ax = axes[2]; ax.plot(G.date, G.H_rozumivka, "k-", lw=1, label="Rozumivka gauge"); ax.plot(G.date, G.swot_node_p50, "o", color="#1b6ca8", ms=4, label="SWOT RiverSP nodes ≤ 5 km"); ax.plot(G.date, G.swot_lake_p50, "s", color="#d95f02", ms=4, label="SWOT LakeSP small objects ≤ 5 km")
    ax.set_ylabel("m EVRF2019"); ax.set_title("Chain check at the upstream end: SWOT vs Rozumivka, 2023"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p60_swot_pool_drawdown.png", dpi=100); print("-> outputs/figures/p60_swot_pool_drawdown.png")


if __name__ == "__main__":
    main()
