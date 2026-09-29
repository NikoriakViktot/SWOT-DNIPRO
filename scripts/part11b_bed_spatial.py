#!/usr/bin/env python
"""PART 11b — is the bed-vs-bed offset uniform, or does it carry a sedimentation signal?

A far stronger discriminator than enumerating datum permutations. Reservoir
sedimentation is not uniform: it is least in the scoured thalweg and greatest
on the low-energy floodplain shelf and in the backwater reaches. So:

  offset flat against distance-to-channel and against chainage
      -> a systematic property of the survey (datum realisation, reduction,
         draft/sound-velocity, digitisation)
  offset small near the channel and large on the shelf
      -> real accumulation since the survey

Compares the post-breach ATL08 EXPOSED BED against the survey bed -- bed to bed,
not water surface to datum.

Outputs
-------
outputs/tables/bed_offset_spatial.csv
outputs/figures/M4_bed_offset_spatial.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from scipy import stats
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
POST_WSE = 5.20
TO_M = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True)


def nmad(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def main() -> None:
    m = pd.read_csv(CFG.TABLES / "bathymetry_vs_atl08.csv")
    m = m[m.match_dist_m <= 50].copy()
    # bed to bed only: drop anything where ATL08 is reading residual water
    m = m[(m.H_survey_m - POST_WSE) > 1.0].copy()
    print(f"{len(m):,} bed-to-bed pairs (survey bed >1 m above the residual water line)")

    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    cx, cy = TO_M.transform(ch.lon.values, ch.lat.values)
    tree = cKDTree(np.c_[cx, cy])
    px, py = TO_M.transform(m.lon.values, m.lat.values)
    dist, idx = tree.query(np.c_[px, py], k=1)
    m["dist_to_channel_km"] = dist / 1000.0
    m["chain_km"] = ch.chain_km.values[idx]

    # ---- the discriminating tests -----------------------------------------
    print("\n=== offset vs distance to the Dnipro thalweg ===")
    m["dbin"] = pd.cut(m.dist_to_channel_km, [0, 0.5, 1, 2, 4, 8, 30])
    t1 = m.groupby("dbin", observed=True).agg(
        n=("diff_m", "size"), median=("diff_m", "median"),
        nmad=("diff_m", nmad)).round(2)
    print(t1.to_string())
    ts_d = stats.theilslopes(m.diff_m.values, m.dist_to_channel_km.values)
    print(f"  Theil-Sen: {ts_d[0]:+.4f} m per km from the channel, "
          f"CI [{ts_d[2]:+.4f}, {ts_d[3]:+.4f}]")

    print("\n=== offset vs chainage from the dam (backwater gradient) ===")
    m["cbin"] = pd.cut(m.chain_km, [0, 30, 60, 90, 120, 160, 240])
    t2 = m.groupby("cbin", observed=True).agg(
        n=("diff_m", "size"), median=("diff_m", "median"),
        nmad=("diff_m", nmad)).round(2)
    print(t2.to_string())
    ts_c = stats.theilslopes(m.diff_m.values, m.chain_km.values)
    print(f"  Theil-Sen: {ts_c[0]:+.5f} m per km along the reservoir, "
          f"CI [{ts_c[2]:+.5f}, {ts_c[3]:+.5f}]")

    # ---- verdict, computed --------------------------------------------------
    rng_d = t1["median"].max() - t1["median"].min()
    rng_c = t2["median"].max() - t2["median"].min()
    # Sedimentation predicts a DIRECTION, not just a spread: deposition is least
    # in the scoured thalweg and greatest off-channel and up-reservoir, so the
    # modern bed should sit relatively higher there, i.e. diff becomes LESS
    # negative with both distance-from-channel and chainage. Test the sign.
    dir_d = ts_d[0] > 0 and ts_d[2] > 0
    dir_c = ts_c[0] > 0 and ts_c[2] > 0
    floor = float(t1["median"].min())          # offset where deposition is least
    print("\n=== verdict ===")
    print(f"  overall median            : {m.diff_m.median():+.2f} m, "
          f"NMAD {nmad(m.diff_m.values):.2f} m")
    print(f"  spread, distance-to-channel: {rng_d:.2f} m  "
          f"(slope {ts_d[0]:+.4f} m/km, CI excludes 0: {ts_d[2]*ts_d[3] > 0})")
    print(f"  spread, chainage           : {rng_c:.2f} m  "
          f"(slope {ts_c[0]:+.5f} m/km, CI excludes 0: {ts_c[2]*ts_c[3] > 0})")
    print(f"  in-thalweg offset          : {floor:+.2f} m  <- where deposition is least")
    print()
    if dir_d and dir_c:
        print("  TWO COMPONENTS, and they must not be conflated:")
        print(f"   1. a FLOOR of about {abs(floor):.1f} m present even in the scoured thalweg,")
        print("      where sedimentation should be near zero. Morphology cannot produce")
        print("      this; it is a systematic property of the survey or its reduction.")
        print(f"   2. on top of it, {rng_d:.1f}-{rng_c:.1f} m of spatially structured excess")
        print("      whose SIGN matches sedimentation in both axes -- the modern bed sits")
        print("      relatively higher off-channel and up-reservoir, exactly where a")
        print("      reservoir deposits. Both slopes are small but their CIs exclude zero.")
        print("   So: a systematic survey offset PLUS a plausible real accumulation signal.")
    else:
        print("  The spatial structure does not follow the sedimentation direction in")
        print("  both axes, so a uniform systematic offset is the simpler reading.")

    m.drop(columns=["dbin", "cbin"]).to_csv(CFG.TABLES / "bed_offset_spatial.csv",
                                            index=False)

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (x, lab, t, ts) in zip(ax, [
            (m.dist_to_channel_km, "distance to the Dnipro thalweg (km)", t1, ts_d),
            (m.chain_km, "chainage from the dam (km)", t2, ts_c)]):
        a.scatter(x, m.diff_m, s=8, color=GREY, alpha=0.35, lw=0)
        c = t.reset_index()
        mid = [iv.mid for iv in c.iloc[:, 0]]
        a.plot(mid, c["median"], "o-", ms=9, lw=2.4, color=RED)
        a.axhline(m.diff_m.median(), color=BLUE, ls="--", lw=1.6)
        a.axhline(0, color=INK, lw=1)
        a.set_xlabel(lab); a.set_ylabel("ATL08 bed − survey bed  (m)")
        a.set_ylim(-6, 4); a.grid(alpha=0.22)
        a.set_title(f"Theil-Sen {ts[0]:+.4f} m/km", fontsize=10, loc="left")
    ax[2].hist(m.diff_m, bins=np.arange(-6, 4.1, 0.25), color=AMBER, alpha=0.85)
    ax[2].axvline(m.diff_m.median(), color=RED, lw=2)
    ax[2].axvline(0, color=INK, lw=1.2)
    ax[2].set_xlabel("ATL08 bed − survey bed  (m)"); ax[2].set_ylabel("pairs")
    ax[2].grid(alpha=0.22)
    ax[2].set_title(f"median {m.diff_m.median():+.2f} m, NMAD {nmad(m.diff_m.values):.2f} m",
                    fontsize=10, loc="left")
    fig.suptitle("M4 · Is the bed-to-bed offset uniform, or does it carry a sedimentation "
                 f"signal?   (n = {len(m):,} pairs, dry exposed bed only)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"M4_bed_offset_spatial.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'M4_bed_offset_spatial.png'}")
    print(f"-> {CFG.TABLES/'bed_offset_spatial.csv'}")


if __name__ == "__main__":
    main()
