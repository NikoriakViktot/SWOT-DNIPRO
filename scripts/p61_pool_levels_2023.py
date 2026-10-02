#!/usr/bin/env python
"""P61 -- Kakhovka POOL water levels in 2023 from every source we have, in ONE vertical frame (EVRF2019) and ONE longitudinal coordinate
(SWORD chainage upstream of the dam, km). User 2026-09-19: "потрібно рівні самого каховського водосховища це важливо"; "може є дані в
інтернеті пошук по свот лаке"; "там є пост на дельті Дніпра" (Kasperivka 80807 -> SWOT check appended here).

Sources and their conversion to EVRF2019 (c = mean EGG2015->EVRF2019 corrector of the six pool stations, CFG.CORRECTOR_BY_STATION):
  ROZUMIVKA_GAUGE   k5 daily mean of the 08h/20h terms, already EVRF2019 (BS77 + EPSG:9902). Constant runs >= 5 d after 12.06 flagged FILLED.
  NIKOPOL_UHE       Ukrhydroenergo press values at the Nikopol post (BS77 absolute, m): 06.06 08:00 16.44 (11.74 + 4.7 m 'since the morning of
                    06.06', uhe.gov.ua 09.06), 07.06 10:00 14.41 (interfax 915384), 08.06 08:00 13.05 (podrobnosti), 09.06 08:00 11.74 (uhe.gov.ua),
                    10.06 ~09:00 10.20 (unn.ua, published 09:34), 11.06 08:00 9.35 (ukrinform 3721216); 13.06 09:00 '< 9 m' (Energoatom) and the post
                    silted -> right-censored, not a value. BS77 -> EVRF2019 with the Nikopol delta_epsg9902 (+0.172).
  KHERSON_UHE       same press series for the Dnipro at Kherson (11.06 4.18, 13.06 08:00 2.96) -> fills the yearbook gap (SRM broken 06.06-09.07).
  SWOT_OUTLET       p60 outlet nodes (dam reach, chainage -0.6..0 km), daily, EVRF2019.
  SWOT_LAKESP_OK    p60 LakeSP Obs objects with level_ok (science orbit, from 29.07).
  SWOT_RIVERSP_POOL p60 RiverSP 10-km bin medians inside the pool (science orbit).
  ICESAT2_ATL13     sibling repo pass levels (kakhovka_atl13_pass_levels.parquet): per (date, rgt) median over beams of median_wse_evrs_m
                    (EGG2015 quasi-geoid, mean tide) + c -> EVRF2019; qc_pass share and beam spread kept; chainage from the beam mid-points.
  GREALM_S6A        USDA/NASA G-REALM lake 000873 'Kakhovskoye' (10-day, Sentinel-6A pass 42, target 47.502N 34.182E, downloaded 2026-09-19 from
                    https://earth.gsfc.nasa.gov/gwm/zip/lake.10.tar.gz): column 15 = height in EGM2008 -> + N_EGM2008(PROJ us_nga_egg08_25) ->
                    ellipsoid -> + free2mean - zeta_EGG2015 + c -> EVRF2019 (the same chain as FABDEM in p56). Ice flag kept.
  YI2025_*          Yi (2025, WRR, doi 10.1029/2024WR038314) Zenodo 14639520 observations.mat: obs.L Nikopol levels (days after breach, m, frame
                    assumed BS77), obs.L_altimetry (Hydroweb/USDA, 'orthometric' -> assumed EGM2008, converted at the G-REALM point), obs.A S1 area.
                    Day 0 assumed = 2023-06-06 00:00 UTC (not stated in the code) -> quality ASSUMED_TIME_AND_FRAME.
Not obtainable without registration (recorded as gaps): DAHITI id 11364 (S3A/S3B/S6A), Hydroweb/hydroweb.next Kakhovka (2016-2023).
Outputs: outputs/tables/p61_pool_levels_2023.csv, p61_pool_level_snapshots.csv, p61_swot_vs_kasperivka.csv, p61_yi2025_reservoir_area.csv,
         outputs/figures/p61_pool_levels_2023.png
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
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import vertical as VT
from swot_dnipro import sword as SW

SIB = Path(CFG.ICESAT_ROOT) if hasattr(CFG, "ICESAT_ROOT") else ROOT.parent / "icesat2-atl13-kakhovka"
GREALM = CFG.BULK_ROOT / "literature/grealm/lake000873.10d.2.txt"; YI = CFG.BULK_ROOT / "literature/yi2025_zenodo/code_share_v2"
BREACH_T0 = pd.Timestamp("2023-06-06 00:00")  # assumption for Yi's day axis
T_EGM = Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
COLS = ["source", "domain", "datetime_utc", "date", "chain_km", "lon", "lat", "H_evrf2019", "H_native", "native_frame", "conversion", "quality", "note"]


def egm2008_to_evrf(H, lon, lat, c):
    _, _, N = T_EGM.transform(np.atleast_1d(lon), np.atleast_1d(lat), np.zeros(np.size(lon))); N = np.asarray(N)
    zeta = np.asarray(VT.sample_grid(CFG.EGG2015_TIF, np.atleast_1d(lon), np.atleast_1d(lat)))
    return np.asarray(H) + N + CFG.free2mean(np.atleast_1d(lat)) - zeta + c, N, zeta


def main():
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr.c_station_m.mean()); st = corr.set_index("name_en")
    tf_ll = Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326", always_xy=True)
    # channel for chainage: the p60 node set (SWORD nodes of every swath that touched the pool) -- identical coordinate to p59/p60
    nodes = pd.read_parquet(CFG.TABLES / "p60_swot_pool_nodes.parquet"); ch60 = nodes.groupby("node_id").agg(lon=("lon", "median"), lat=("lat", "median"), chain_km=("s_up_km", "median"), reach_id=("reach_id", "first")).reset_index()
    sw = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")  # SWORD main stem, dam..~191 km (k5/k6 coordinate)
    if "chain_km" not in sw:
        sw = SW.chainage_from_dam(sw, *CFG.KAKHOVKA_DAM)
    # ONE chainage for everything: the SWORD main stem of k5/k6 (dam .. 320 km). The RiverSP p_dist_out of p60 differs by ~13 km near Zaporizhzhia
    # (nodes of a side arm at Khortytsia), so SWOT positions are re-snapped here from lon/lat rather than taken from p60's s_up_km.
    ch = sw[sw.chain_km > -1][["node_id", "reach_id", "lon", "lat", "chain_km"]].reset_index(drop=True); del ch60
    tree = SW.build_chainage_tree(ch); print(f"chainage channel (SWORD k5/k6): {len(ch)} nodes, {ch.chain_km.min():.1f}..{ch.chain_km.max():.1f} km")
    def chain_of(lon, lat): return SW.assign_chainage(np.atleast_1d(lon), np.atleast_1d(lat), ch, max_dist_km=40, tree=tree)[0]
    rows = []
    def add(source, dt, H, lon, lat, H_native, frame, conv, quality, note="", domain="pool", chain=None):
        dt = pd.Timestamp(dt); dt = dt.tz_convert(None) if dt.tzinfo is not None else dt; ck = float(chain_of(lon, lat)[0]) if chain is None else chain
        rows.append(dict(source=source, domain=domain, datetime_utc=dt, date=dt.normalize(), chain_km=ck, lon=lon, lat=lat, H_evrf2019=float(H), H_native=float(H_native) if H_native is not None else np.nan, native_frame=frame, conversion=conv, quality=quality, note=note))
    # 1. Rozumivka (k5)
    k5 = pd.read_csv(CFG.TABLES / "k5_gauge_levels_evrf2019.csv", usecols=["station_name", "x_utm", "y_utm", "date", "H_EVRF2019_m"]); k5["date"] = pd.to_datetime(k5.date)
    r = k5[(k5.station_name == "Rozumivka") & (k5.date >= "2023-05-20") & (k5.date <= "2023-12-31")].dropna(subset=["H_EVRF2019_m"]).groupby("date").agg(H=("H_EVRF2019_m", "mean"), n=("H_EVRF2019_m", "size"), x=("x_utm", "first"), y=("y_utm", "first")).reset_index()
    same = (r.H.diff().abs() < 1e-9); grp = (~same).cumsum(); run = same.groupby(grp).transform("size") + 1; filled = ((run >= 5) & (r.date >= "2023-06-12")); filled = filled | filled.shift(-1, fill_value=False) & (r.H == r.H.shift(-1))
    rl, rb = tf_ll.transform(r.x.iloc[0], r.y.iloc[0])
    for _, q in r.iterrows():
        add("ROZUMIVKA_GAUGE", q.date + pd.Timedelta(hours=11), q.H, rl, rb, None, "EVRF2019 (k5: BS77 + EPSG:9902)", "none", "FILLED_SUSPECT" if filled.loc[_] else "OK", f"daily mean of {int(q.n)} terms")
    # 2. Nikopol & Kherson press values (Ukrhydroenergo)
    nik = st.loc["Nikopol"]; dN = float(nik.delta_epsg9902_m); dK = float(pd.read_csv(ROOT / "data/historical/sea_posts_2023/post_metadata_2023.csv").set_index("post_id").loc[80805, "delta_epsg9902_m"])
    for dt, H, q, src in [("2023-06-06 05:00", 16.44, "DERIVED", "uhe.gov.ua 09.06: 11.74 + 4.7 m since the morning of 06.06"), ("2023-06-07 07:00", 14.41, "PRESS", "interfax 915384, 10:00 Kyiv"), ("2023-06-08 05:00", 13.05, "PRESS", "podrobnosti 2474570, 08:00 Kyiv"),
                         ("2023-06-09 05:00", 11.74, "PRESS", "uhe.gov.ua, glavcom 933295, 08:00 Kyiv"), ("2023-06-10 06:00", 10.20, "PRESS_TIME_UNCERTAIN", "unn.ua, published 09:34 Kyiv, text says 12:00"), ("2023-06-11 05:00", 9.35, "PRESS", "ukrinform 3721216, 08:00 Kyiv")]:
        add("NIKOPOL_UHE", dt, H + dN, float(nik.lon), float(nik.lat), H, "BS77 absolute (press)", f"+delta_epsg9902 Nikopol {dN:+.3f}", q, src)
    add("NIKOPOL_UHE", "2023-06-13 06:00", 9.0 + dN, float(nik.lon), float(nik.lat), 9.0, "BS77 absolute (press)", f"+{dN:+.3f}", "CENSORED_UPPER_BOUND", "Energoatom: '< 9 m'; post silted, Ukrhydrometcenter stopped measuring")
    for dt, H, src in [("2023-06-11 05:00", 4.18, "ukrinform 3721216"), ("2023-06-13 05:00", 2.96, "ua-energy 13.06, 08:00 Kyiv")]:
        add("KHERSON_UHE", dt, H + dK, CFG.KHERSON_GAUGE[0], CFG.KHERSON_GAUGE[1], H, "BS77 absolute (press)", f"+delta_epsg9902 Kherson {dK:+.3f}", "PRESS", src, domain="below_dam", chain=-np.nan)
    # 3. SWOT p60
    o = pd.read_csv(CFG.TABLES / "p60_swot_outlet_drawdown.csv", parse_dates=["date"])
    for _, q in o.iterrows():
        add("SWOT_OUTLET", q.utc if isinstance(q.utc, str) else q.date, q.H_outlet, 33.365, 46.77, np.nan, "SWOT wse+geoid_hght (ellipsoid)", "+free2mean - zeta_EGG2015 + c", "OK" if q.swot_n >= 2 else "SINGLE_NODE", f"{int(q.swot_n)} nodes, min {q.H_outlet_min:.2f} max {q.H_outlet_max:.2f}", chain=0.0)
    L = pd.read_csv(CFG.TABLES / "p60_swot_pool_lakes.csv", parse_dates=["date"]); L = L[L.level_ok]
    for _, q in L.iterrows():
        add("SWOT_LAKESP_OK", q.time_str if isinstance(q.time_str, str) and q.time_str != "no_data" else q.date, q.H_evrf, q.lon, q.lat, q.wse, "SWOT LakeSP wse (geoid)", "+geoid_hght +free2mean - zeta + c", "OK", f"lake {q.lake_id} {q.area_total:.1f} km2")
    # RiverSP nodes inside the pool (science orbit): re-snapped to the k5/k6 chainage, 10-km bins per date
    pn = nodes[nodes["where"] == "pool"].copy(); pn["chain_km"] = chain_of(pn.lon.values, pn.lat.values); pn = pn.dropna(subset=["chain_km"]); pn["bin"] = (np.floor(pn.chain_km / 10) * 10).astype(int)
    P = pn.groupby(["date", "pas", "bin"]).agg(H=("H_evrf", "median"), n=("H_evrf", "size"), lon=("lon", "median"), lat=("lat", "median")).reset_index(); P = P[P.n >= 3]
    for _, q in P.iterrows():
        add("SWOT_RIVERSP_POOL", pd.Timestamp(q.date) + pd.Timedelta(hours=12), q.H, q.lon, q.lat, np.nan, "SWOT RiverSP wse (geoid)", "+geoid_hght +free2mean - zeta + c", "OK", f"bin {int(q.bin)}-{int(q.bin) + 10} km, n {int(q.n)}, pass {q.pas}", chain=q.bin + 5.0)
    # 4. ICESat-2 ATL13 (sibling)
    a = pd.read_parquet(SIB / "data/processed/kakhovka_atl13_pass_levels.parquet"); a["t"] = pd.to_datetime(a.datetime).dt.tz_localize(None); a = a[(a.t >= "2023-05-20") & (a.t <= "2023-12-31")]
    g = a.groupby(["date", "rgt"]).agg(t=("t", "min"), H=("median_wse_evrs_m", "median"), spread=("median_wse_evrs_m", lambda v: v.max() - v.min()), n=("beam", "size"), nqc=("qc_pass", "sum"), lon=("lon_mean", "median"), lat=("lat_mean", "median")).reset_index()
    for _, q in g.iterrows():
        qual = "OK" if q.nqc >= max(2, q.n // 2) and q.spread < 0.5 else ("QC_PARTIAL" if q.nqc > 0 else "QC_FAIL_slope_or_spread")
        add("ICESAT2_ATL13", q.t, q.H + c, q.lon, q.lat, q.H, "EVRS/EGG2015 mean tide (sibling chain)", f"+c {c:+.3f}", qual, f"RGT {q.rgt}, {int(q.nqc)}/{int(q.n)} beams qc_pass, beam spread {q.spread:.2f} m")
    # 5. G-REALM
    rec = []
    for line in GREALM.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 16 and p[2].isdigit() and len(p[2]) == 8 and p[2] != "99999999":
            rec.append(dict(mission=p[0], cycle=int(p[1]), date=pd.Timestamp(p[2]) + pd.Timedelta(hours=int(p[3]), minutes=int(p[4])), dh=float(p[5]), err=float(p[6]), ice=int(p[13]), H_egm08=float(p[14])))
    G = pd.DataFrame(rec); G = G[(G.H_egm08 < 9000) & (G.date >= "2023-01-01")]
    Hv, N, zeta = egm2008_to_evrf(G.H_egm08.values, np.full(len(G), 34.182), np.full(len(G), 47.502), c)
    for (_, q), h in zip(G.iterrows(), Hv):
        add("GREALM_S6A", q.date, h, 34.182, 47.502, q.H_egm08, "EGM2008 orthometric (G-REALM col 15)", f"+N_EGM2008 {N[0]:.3f} +free2mean - zeta_EGG2015 {zeta[0]:.3f} + c {c:+.3f}", "ICE_FLAG" if q.ice else "OK", f"{q.mission} cycle {q.cycle}, err {q.err:.3f} m, dh vs J2 ref {q.dh:+.2f}")
    print(f"G-REALM point: N_EGM2008 {N[0]:.3f}, zeta_EGG2015 {zeta[0]:.3f}, free2mean {CFG.free2mean(47.502):+.4f}, c {c:+.3f} -> EGM2008 -> EVRF2019 shift {Hv[0] - G.H_egm08.iloc[0]:+.3f} m")
    # 6. Yi 2025
    yl = pd.read_csv(YI / "obs_L.csv"); ya = pd.read_csv(YI / "obs_L_altimetry.csv"); yA = pd.read_csv(YI / "obs_A.csv")
    for _, q in yl.iterrows():
        add("YI2025_NIKOPOL", BREACH_T0 + pd.Timedelta(days=float(q.iloc[0])), q.iloc[1] + dN, float(nik.lon), float(nik.lat), q.iloc[1], "assumed BS77 (obs.L 'water level changes at Nikopol')", f"+{dN:+.3f}", "ASSUMED_TIME_AND_FRAME", f"day {q.iloc[0]:.3f} after 2023-06-06 00:00 (assumed t0)")
    Ha, _, _ = egm2008_to_evrf(ya.iloc[:, 1].values, np.full(len(ya), 34.182), np.full(len(ya), 47.502), c)
    for (_, q), h in zip(ya.iterrows(), Ha):
        add("YI2025_ALTIMETRY", BREACH_T0 + pd.Timedelta(days=float(q.iloc[0])), h, 34.182, 47.502, q.iloc[1], "assumed EGM2008 orthometric (Hydroweb/USDA)", "as G-REALM", "ASSUMED_TIME_AND_FRAME", f"day {q.iloc[0]:.3f}")
    yA = yA.rename(columns={yA.columns[0]: "day_after_breach", yA.columns[1]: "area_km2_S1"}); yA["datetime_assumed"] = BREACH_T0 + pd.to_timedelta(yA.day_after_breach, unit="D"); yA.to_csv(CFG.TABLES / "p61_yi2025_reservoir_area.csv", index=False)
    # ---------------- register
    R = pd.DataFrame(rows)[COLS].sort_values(["datetime_utc", "source"]); R.to_csv(CFG.TABLES / "p61_pool_levels_2023.csv", index=False)
    print(R.groupby("source").agg(n=("H_evrf2019", "size"), first=("date", "min"), last=("date", "max"), chain=("chain_km", "median")).to_string())
    jun = R[(R.domain == "pool") & (R.date >= "2023-05-30") & (R.date <= "2023-07-01") & ~R.source.str.startswith("YI2025")]
    print("\nJune 2023, pool (m EVRF2019; columns = source):"); print(jun.pivot_table(index="date", columns="source", values="H_evrf2019", aggfunc="median").round(2).to_string())
    # snapshots along the pool: dates with >= 2 pool sources within the same day
    snap = jun[jun.quality.isin(["OK", "PRESS", "DERIVED", "PRESS_TIME_UNCERTAIN", "SINGLE_NODE", "QC_PARTIAL"])].copy(); cnt = snap.groupby("date").source.nunique(); snap = snap[snap.date.isin(cnt[cnt >= 2].index)]
    snap[["date", "source", "chain_km", "H_evrf2019", "quality"]].sort_values(["date", "chain_km"]).to_csv(CFG.TABLES / "p61_pool_level_snapshots.csv", index=False)
    # ---------------- Kasperivka (delta post) vs SWOT p59 nodes within 10 km
    N59 = pd.read_parquet(CFG.TABLES / "p59_swot_flood_nodes.parquet"); tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); meta = pd.read_csv(ROOT / "data/historical/sea_posts_2023/post_metadata_2023.csv").set_index("post_id")
    kx, ky = tf.transform(meta.loc[80807, "lon"], meta.loc[80807, "lat"]); near = N59[np.hypot(N59.x - kx, N59.y - ky) < 10000]
    s = near.groupby("date").H_evrf.agg(swot_n="size", swot_p50="median").reset_index(); gk = pd.read_csv(ROOT / "data/historical/sea_posts_2023/daily_levels_cm.csv"); gk = gk[gk.post_id == 80807][["date", "H_evrf2019_m", "flag"]]; gk["date"] = pd.to_datetime(gk.date)
    K = gk.merge(s, on="date", how="outer").sort_values("date"); K = K[(K.date >= "2023-05-25") & (K.date <= "2023-07-10")]; K["diff"] = K.swot_p50 - K.H_evrf2019_m; K.to_csv(CFG.TABLES / "p61_swot_vs_kasperivka.csv", index=False)
    v = K.dropna(subset=["diff"]); print(f"\nKasperivka 80807 (delta, Rvach arm) vs SWOT nodes <= 10 km (chainage {near.s_km.min():.0f}-{near.s_km.max():.0f} km below dam): n {len(v)}, median {v['diff'].median():+.2f}, MAD {(v['diff'] - v['diff'].median()).abs().median():.2f}")
    print("SWOT dates near Kasperivka 01-20.06:", ", ".join(f"{d.strftime('%m-%d')} {h:.2f}(n{int(n)})" for d, n, h in s[(s.date >= '2023-06-01') & (s.date <= '2023-06-20')][["date", "swot_n", "swot_p50"]].itertuples(index=False)) or "none (nodes flagged during the peak)")
    # ---------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(26, 7.5)); pool = R[R.domain == "pool"]
    style = {"ROZUMIVKA_GAUGE": ("k", "-", None, 3), "NIKOPOL_UHE": ("#7b3294", "", "D", 6), "SWOT_OUTLET": ("#b2182b", "-", "o", 4), "ICESAT2_ATL13": ("#1a9850", "", "^", 8), "GREALM_S6A": ("#2166ac", "-", "s", 5),
             "SWOT_LAKESP_OK": ("#d95f02", "", "o", 4), "SWOT_RIVERSP_POOL": ("#e7298a", "", "x", 4), "YI2025_NIKOPOL": ("#7b3294", "", "+", 7), "YI2025_ALTIMETRY": ("#2166ac", "", "+", 7)}
    for ax, (t0, t1) in zip(axes[:2], (("2023-05-25", "2023-12-31"), ("2023-06-03", "2023-07-02"))):
        w = pool[(pool.datetime_utc >= t0) & (pool.datetime_utc <= t1)]
        for srcname, (col, ls, mk, ms) in style.items():
            s_ = w[w.source == srcname].sort_values("datetime_utc")
            if not len(s_):
                continue
            ok = ~s_.quality.str.contains("FILLED|CENSORED|QC_FAIL")
            ax.plot(s_.datetime_utc[ok], s_.H_evrf2019[ok], ls=ls or "none", marker=mk, ms=ms, color=col, lw=1.2, label=f"{srcname} (chainage {s_.chain_km.median():.0f} km)" if s_.chain_km.notna().any() else srcname)
            if (~ok).any():
                ax.plot(s_.datetime_utc[~ok], s_.H_evrf2019[~ok], ls="none", marker=mk, ms=ms, mfc="none", color=col, alpha=0.5)
        ax.axvline(pd.Timestamp(CFG.BREACH_DATE), color="grey", ls="--", lw=1); ax.grid(alpha=0.3); ax.set_ylabel("m EVRF2019"); ax.tick_params(axis="x", rotation=30)
    axes[0].set_title("Kakhovka pool level 2023, all sources in EVRF2019 (hollow = flagged)"); axes[0].legend(fontsize=7); axes[1].set_title("June 2023 drawdown: outlet (SWOT), Nikopol (Ukrhydroenergo), 105 km (G-REALM, ICESat-2), upstream end (Rozumivka)")
    ax = axes[2]; cmap = plt.get_cmap("plasma"); days = sorted(snap.date.unique())
    for i, d in enumerate(days):
        s_ = snap[snap.date == d].sort_values("chain_km"); ax.plot(s_.chain_km, s_.H_evrf2019, "-o", ms=4, lw=1.2, color=cmap(i / max(len(days) - 1, 1)), label=pd.Timestamp(d).strftime("%m-%d"))
    ax.axhline(17.4, color="grey", ls=":", lw=1); ax.set_xlabel("chainage upstream of the dam, km"); ax.set_ylabel("m EVRF2019"); ax.set_title("Longitudinal water-surface snapshots during the drawdown (days with ≥ 2 sources)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p61_pool_levels_2023.png", dpi=100); print("-> outputs/figures/p61_pool_levels_2023.png")


if __name__ == "__main__":
    main()
