#!/usr/bin/env python
"""P59 -- SWOT KaRIn water-surface elevation of the lower Dnipro during the June-2023 breach flood (user 2026-09-19: "приступай").

The SWOT clip ($BULK_ROOT/swot_ua/swot_l2_hr_riversp_2.0/2023/{05,06,07}) holds the 1-day cal/val orbit, pass 001, which crossed the
lower Dnipro EVERY day 2023-06-01..06-29 (28 Node files, ~1 100 SWORD nodes dam -> estuary). Nothing in the project had used the
post-breach part of it: `swot_levels_common_frame.parquet` stops on 2023-06-05 and covers the reservoir only.

Per node: keep node_q <= NODE_Q_MAX, wse valid, dark_frac < 0.5. Vertical chain to the project frame (same form as ICESat-2 / FABDEM):
    wse (EGM2008 geoid, tides removed) + geoid_hght -> WGS84 ellipsoid; + free2mean(lat); - zeta_EGG2015; + c_EGG2015_to_EVRF2019 = H_EVRF2019
Along-channel coordinate: SWORD p_dist_out -> distance from the dam s = dist_out(dam) - dist_out (km, positive downstream).
Outputs
  outputs/tables/p59_swot_flood_nodes.parquet          every accepted node observation (date, reach, node, s_km, H_evrf, wse_u, width, q)
  outputs/tables/p59_swot_flood_profiles.csv           per date x 5-km bin: median / p25 / p75 H, n nodes
  outputs/tables/p59_swot_peak_wse_profile.csv         per 5-km bin: peak H over the event, date of peak, pre-breach H (05-25..06-05), rise
  outputs/tables/p59_swot_vs_kherson.csv               SWOT nodes within 3 km of the Kherson gauge vs the daily gauge (BS77 -> EVRF2019)
  outputs/figures/p59_swot_flood_wse.png
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
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swot_dnipro import config as CFG
from swot_dnipro import vertical as VT

SWOT = CFG.BULK_ROOT / "swot_ua/swot_l2_hr_riversp_2.0/2023"
BBOX = (32.3, 46.3, 33.42, 46.95)                       # lower Dnipro, dam -> estuary
DATES = ("2023-05-25", "2023-07-10")
NODE_Q_MAX = 1; DARK_MAX = 0.5; BIN_KM = 5.0
KH_DELTA_9902 = 0.22                                    # Kherson BS77 -> EVRF2019 (k5 order of magnitude, same as p52/p53)


def main():
    corr = pd.read_csv(CFG.CORRECTOR_BY_STATION); c = float(corr[corr.gauge_zero_bs77_m == 12.0].c_station_m.mean())
    files = sorted(f for m in ("05", "06", "07") for f in glob.glob(str(SWOT / m / "*RiverSP_Node*.shp")))
    files = [f for f in files if DATES[0].replace("-", "") <= Path(f).name.split("_")[8][:8] <= DATES[1].replace("-", "")]
    print(f"{len(files)} Node files {DATES[0]}..{DATES[1]}", flush=True)
    parts = []
    for f in files:
        g = gpd.read_file(f, bbox=BBOX)
        if not len(g):
            continue
        g = g[(g.wse > -1e11) & (g.node_q <= NODE_Q_MAX) & (g.dark_frac < DARK_MAX) & (g.geoid_hght > -1e11)].copy()
        if not len(g):
            continue
        g["date"] = pd.Timestamp(Path(f).name.split("_")[8][:8]); g["pass_id"] = int(Path(f).name.split("_")[6]); g["lon"] = g.geometry.x; g["lat"] = g.geometry.y
        parts.append(pd.DataFrame(g.drop(columns="geometry")))
    N = pd.concat(parts, ignore_index=True)
    N["zeta"] = VT.sample_grid(CFG.EGG2015_TIF, N.lon.values, N.lat.values); N["H_evrf"] = N.wse + N.geoid_hght + CFG.free2mean(N.lat.values) - N.zeta + c
    tf = Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True); N["x"], N["y"] = tf.transform(N.lon.values, N.lat.values)
    dam = np.array(tf.transform(*CFG.KAKHOVKA_DAM)); kh = np.array(tf.transform(CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3]))
    d_dam = np.hypot(N.x - dam[0], N.y - dam[1]); dist_out_dam = float(N.loc[d_dam.idxmin(), "p_dist_out"]); N["s_km"] = (dist_out_dam - N.p_dist_out) / 1e3
    N = N[N.s_km > -2].copy()                                    # downstream of the dam (keep 2 km upstream for the tailwater node)
    N["river_name"] = N.river_name.astype(str)
    print(f"accepted node observations: {len(N):,} on {N.date.nunique()} dates, {N.node_id.nunique()} nodes, s 0..{N.s_km.max():.0f} km; rivers: {N.river_name.value_counts().head(4).to_dict()}")
    N[["date", "pass_id", "reach_id", "node_id", "river_name", "lon", "lat", "x", "y", "s_km", "p_dist_out", "wse", "wse_u", "wse_r_u", "geoid_hght", "zeta", "H_evrf", "width", "node_q", "dark_frac", "xovr_cal_q"]].to_parquet(CFG.TABLES / "p59_swot_flood_nodes.parquet", index=False)
    N = N[N.river_name == "Dnipro"].copy()                  # profiles on the Dnipro main stem only: tributary nodes (Inhulets, Kokan') carry their own dist_out
    N["bin_km"] = (np.floor(N.s_km / BIN_KM) * BIN_KM).astype(int); print(f"Dnipro main-stem observations: {len(N):,}, {N.node_id.nunique()} nodes")
    prof = N.groupby(["date", "bin_km"]).H_evrf.agg(n="size", p25=lambda v: v.quantile(.25), p50="median", p75=lambda v: v.quantile(.75)).reset_index(); prof = prof[prof.n >= 3]
    prof.to_csv(CFG.TABLES / "p59_swot_flood_profiles.csv", index=False)
    pre = prof[(prof.date >= "2023-05-25") & (prof.date <= "2023-06-05")].groupby("bin_km").p50.median().rename("H_pre")
    pk = prof[prof.date >= "2023-06-06"].sort_values("p50", ascending=False).drop_duplicates("bin_km").set_index("bin_km")[["date", "p50", "n"]].rename(columns={"date": "peak_date", "p50": "H_peak", "n": "n_peak"})
    P = pd.concat([pre, pk], axis=1).sort_index(); P["rise_m"] = P.H_peak - P.H_pre; P.to_csv(CFG.TABLES / "p59_swot_peak_wse_profile.csv"); print(P.round(2).to_string())
    # per-node peak table for p42 (flood ceiling from observation): pre-breach median, event maximum, smoothed along s (rolling median of 7 nodes ~1.4 km)
    nod = N.groupby("node_id").agg(x=("x", "first"), y=("y", "first"), s_km=("s_km", "first"), n_obs=("H_evrf", "size")).reset_index()
    pre_n = N[(N.date >= "2023-05-25") & (N.date <= "2023-06-05")].groupby("node_id").H_evrf.median().rename("H_pre")
    ev_n = N[N.date >= "2023-06-06"].sort_values("H_evrf", ascending=False).drop_duplicates("node_id").set_index("node_id")[["H_evrf", "date"]].rename(columns={"H_evrf": "H_peak", "date": "peak_date"})
    nod = nod.set_index("node_id").join(pre_n).join(ev_n).reset_index().sort_values("s_km"); nod["H_peak_smooth"] = nod.H_peak.rolling(7, center=True, min_periods=3).median()
    nod.to_csv(CFG.TABLES / "p59_swot_node_peak.csv", index=False); print(f"node peak table: {len(nod)} nodes, H_peak_smooth {nod.H_peak_smooth.min():.1f}..{nod.H_peak_smooth.max():.1f} m")
    # Kherson gauge comparison
    g = pd.read_parquet(CFG.KHERSON_GAUGE_PARQUET); g = g[g.stat_type == "daily"].copy(); g["date"] = pd.to_datetime(g.date); g = g[(g.date >= DATES[0]) & (g.date <= DATES[1])]; g["H_gauge_evrf"] = g.water_level_m_abs + KH_DELTA_9902
    near = N[np.hypot(N.x - kh[0], N.y - kh[1]) < 3000].groupby("date").H_evrf.agg(swot_n="size", swot_p50="median", swot_p25=lambda v: v.quantile(.25), swot_p75=lambda v: v.quantile(.75)).reset_index()
    K = g[["date", "water_level_m_abs", "H_gauge_evrf"]].merge(near, on="date", how="left"); K["swot_minus_gauge"] = K.swot_p50 - K.H_gauge_evrf; K.to_csv(CFG.TABLES / "p59_swot_vs_kherson.csv", index=False)
    v = K.dropna(subset=["swot_p50"]); print(f"SWOT vs Kherson gauge ({len(v)} days): median diff {v.swot_minus_gauge.median():+.2f} m, MAD {1.4826*np.median(np.abs(v.swot_minus_gauge - v.swot_minus_gauge.median())):.2f}; peak gauge {g.H_gauge_evrf.max():.2f} on {g.loc[g.H_gauge_evrf.idxmax(), 'date'].date()}, peak SWOT near gauge {v.swot_p50.max():.2f} on {v.loc[v.swot_p50.idxmax(), 'date'].date()}")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    ax = axes[0]; dates = sorted(prof.date.unique()); cmap = plt.get_cmap("viridis"); ev = [d for d in dates if d >= pd.Timestamp("2023-06-05") and d <= pd.Timestamp("2023-06-25")]
    for i, d in enumerate(ev):
        s = prof[prof.date == d].sort_values("bin_km"); ax.plot(s.bin_km + BIN_KM / 2, s.p50, "-", color=cmap(i / max(len(ev) - 1, 1)), lw=1.4, label=pd.Timestamp(d).strftime("%m-%d"))
    ax.plot(P.index + BIN_KM / 2, P.H_pre, "k--", lw=1, label="pre-breach (05-25..06-05)"); ax.axvline(np.hypot(kh[0] - dam[0], kh[1] - dam[1]) / 1e3, color="grey", ls=":"); ax.text(np.hypot(kh[0] - dam[0], kh[1] - dam[1]) / 1e3 + 1, ax.get_ylim()[1] * 0.95, "Kherson (straight-line)", fontsize=8, color="grey")
    ax.set_xlabel("distance downstream of the dam along SWORD, km"); ax.set_ylabel("SWOT water-surface elevation, m EVRF2019"); ax.set_title("SWOT KaRIn daily WSE profiles, lower Dnipro, June 2023 (node_q ≤ 1, 5-km bin medians)"); ax.legend(fontsize=6.5, ncol=3); ax.grid(alpha=0.3)
    ax = axes[1]; ax.plot(K.date, K.H_gauge_evrf, "k-o", ms=3, label="Kherson gauge 80805 (daily, BS77 + 0.22)"); ax.errorbar(K.date, K.swot_p50, yerr=[K.swot_p50 - K.swot_p25, K.swot_p75 - K.swot_p50], fmt="s", color="#1b6ca8", ms=4, capsize=2, label="SWOT nodes within 3 km of the gauge (p50, IQR)")
    ax.set_ylabel("m EVRF2019"); ax.set_title(f"SWOT vs gauge at Kherson: median difference {v.swot_minus_gauge.median():+.2f} m"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=30)
    ax = axes[2]; ax.plot(P.index + BIN_KM / 2, P.H_peak, "r-o", ms=3, label="peak WSE (max of daily bin medians)"); ax.plot(P.index + BIN_KM / 2, P.H_pre, "k--", label="pre-breach WSE"); ax.fill_between(P.index + BIN_KM / 2, P.H_pre, P.H_peak, color="red", alpha=0.15, label="rise")
    try:
        h0 = pd.read_csv(CFG.TABLES / "p42_h0_profile.csv"); h0 = h0[h0.zone.str.startswith("ZONE_4")]; ax.step(h0.bin_km, h0.h0_m + 1.0, where="post", color="#7a3b00", lw=1.2, label="p42 h0(x)+1 m (S1∧S2 calibration; straight-line km)")
    except Exception:
        pass
    ax.set_xlabel("distance downstream of the dam, km"); ax.set_ylabel("m EVRF2019"); ax.set_title("peak water surface from SWOT vs the terrain-based flood ceiling used in p42"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/p59_swot_flood_wse.png", dpi=100); print("-> outputs/figures/p59_swot_flood_wse.png")


if __name__ == "__main__":
    main()
