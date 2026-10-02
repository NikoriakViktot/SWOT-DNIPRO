#!/usr/bin/env python
"""QA STAGE 2 — the decisive check: does the terrain fit sit on the LOWER surface?

Pulls raw ATL03 photons with ATL08 classification labels for a stratified sample
of the segments used in dH, and plots each one against the fitted terrain height
and the old bathymetric bed. If the terrain solution is riding on vegetation, it
will be visible here and nowhere else.

Sample is stratified deliberately: most-negative dH, median dH, positive dH,
near thalweg, far from thalweg, with and without vegetation flag.

dH = H_ICESat2 - H_survey.  dH<0 = ICESat-2 surface LOWER than old bathymetry.

Outputs
-------
outputs/tables/qa2_photon_check.csv
outputs/figures/QA2_photon_profiles.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sliderule
from sliderule import icesat2

from swot_dnipro import config as CFG
from swot_dnipro.vertical import sample_grid

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
CLS = {0: ("noise", "#dddddd"), 1: ("ground", "#8a5a12"),
       2: ("canopy", "#5aa06a"), 3: ("top of canopy", "#2f6b3f"),
       4: ("unclassified", "#b8c4cc")}
BOX_M = 0.0045          # ~500 m half-box for the PULL
SEG_M = 60.0            # photons are then restricted to the actual
                        # 100 m segment footprint before any statistic:
                        # a 500 m box straddles banks and margins and
                        # its median is not comparable to the fit.


def pick_sample(m, k=2):
    """Stratified: extremes of dH, median, near/far thalweg, veg/no-veg."""
    s = []
    tag = []
    for name, sub in [
            ("most negative dH", m.nsmallest(60, "dH")),
            ("median dH", m.iloc[(m.dH - m.dH.median()).abs().argsort()[:60]]),
            ("positive dH", m.nlargest(60, "dH")),
            ("near thalweg", m[m.dist_thalweg_km < 1].nsmallest(60, "match_dist_m")),
            ("far from thalweg", m[m.dist_thalweg_km > 10].nsmallest(60, "match_dist_m")),
            ("vegetation flagged", m[m.h_canopy > 2].nsmallest(60, "match_dist_m")),
            ("strict bare ground", m[(m.h_canopy == 0) & (m.veg_ph_count == 0)
                                     & (m.gnd_ph_count >= 50)].nsmallest(60, "match_dist_m")),
    ]:
        if not len(sub):
            continue
        take = sub.sample(min(k, len(sub)), random_state=CFG.SEED)
        s.append(take); tag += [name] * len(take)
    out = pd.concat(s, ignore_index=True)
    out["stratum"] = tag
    return out


def main() -> None:
    m = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/qa1_pairs_with_qa.parquet")
    sel = pick_sample(m)
    print(f"{len(sel)} segments selected across {sel.stratum.nunique()} strata")

    sliderule.init("slideruleearth.io", verbose=False)
    rows, panels = [], []
    for i, r in enumerate(sel.itertuples()):
        poly = [{"lon": r.lon - BOX_M, "lat": r.lat - BOX_M},
                {"lon": r.lon + BOX_M, "lat": r.lat - BOX_M},
                {"lon": r.lon + BOX_M, "lat": r.lat + BOX_M},
                {"lon": r.lon - BOX_M, "lat": r.lat + BOX_M},
                {"lon": r.lon - BOX_M, "lat": r.lat - BOX_M}]
        d0 = pd.Timestamp(r.date)
        parms = {"poly": poly, "t0": str(d0.date()), "t1": str((d0 + pd.Timedelta(days=1)).date()),
                 "srt": icesat2.SRT_LAND, "cnf": -1, "pass_invalid": True,
                 "atl08_class": ["atl08_noise", "atl08_ground", "atl08_canopy",
                                 "atl08_top_of_canopy", "atl08_unclassified"]}
        try:
            g = icesat2.atl03sp(parms)
        except Exception as e:
            print(f"  {i:2d} {r.stratum:<20} pull failed: {type(e).__name__}")
            continue
        if not len(g):
            print(f"  {i:2d} {r.stratum:<20} no photons returned")
            continue
        g = g[g["rgt"] == r.rgt] if "rgt" in g else g
        if not len(g):
            continue
        lat = g.geometry.y.values
        lon = g.geometry.x.values
        z = sample_grid(CFG.EGG2015_TIF, lon, lat)
        h = g["height"].values + CFG.free2mean(lat) - z
        cl = g["atl08_class"].values if "atl08_class" in g else np.zeros(len(g), int)

        # restrict to the real segment footprint before comparing to the fit
        import pyproj as _pj
        _g = _pj.Geod(ellps="WGS84")
        _, _, dseg = _g.inv(np.full(len(lat), r.lon), np.full(len(lat), r.lat), lon, lat)
        near = dseg <= SEG_M
        lat, lon, h, cl = lat[near], lon[near], h[near], cl[near]
        gnd = h[cl == 1]
        if len(gnd) < 5:
            print(f"  {i:2d} {r.stratum:<20} <5 ground photons in the box")
            continue
        rows.append({"stratum": r.stratum, "lon": r.lon, "lat": r.lat,
                     "date": d0.date(), "rgt": int(r.rgt),
                     "dH": r.dH, "H_icesat2_fit": r.H_icesat2, "H_survey": r.H_survey,
                     "n_photons": len(h), "n_ground_ph": int((cl == 1).sum()),
                     "n_canopy_ph": int(np.isin(cl, [2, 3]).sum()),
                     "photon_ground_median": float(np.median(gnd)),
                     "photon_ground_p10": float(np.percentile(gnd, 10)),
                     "fit_minus_photon_ground": float(r.H_icesat2 - np.median(gnd)),
                     "photon_ground_minus_survey": float(np.median(gnd) - r.H_survey)})
        panels.append((r, lat, h, cl, np.median(gnd)))
        print(f"  {i:2d} {r.stratum:<20} ph={len(h):5d} gnd={int((cl==1).sum()):4d} "
              f"canopy={int(np.isin(cl,[2,3]).sum()):4d}  "
              f"fit-photonGround={r.H_icesat2-np.median(gnd):+.2f} m")

    q = pd.DataFrame(rows)
    q.to_csv(CFG.TABLES / "qa2_photon_check.csv", index=False)
    if not len(q):
        print("no usable segments"); return

    print(f"\n=== does the fitted terrain sit on the photon ground surface? ===")
    print(f"  n = {len(q)} segments")
    print(f"  fit - photon ground median : {q.fit_minus_photon_ground.median():+.2f} m "
          f"(NMAD {1.4826*np.median(np.abs(q.fit_minus_photon_ground-q.fit_minus_photon_ground.median())):.2f})")
    print(f"  -> {'fit sits ON the photon ground' if abs(q.fit_minus_photon_ground.median())<0.5 else 'FIT IS DISPLACED from the photon ground'}")
    # dH computed straight from the photons, bypassing the ATL08 fit entirely
    print(f"\n=== dH from RAW PHOTONS, bypassing the ATL08 terrain fit ===")
    print(f"  photon ground - survey : {q.photon_ground_minus_survey.median():+.2f} m "
          f"(NMAD {1.4826*np.median(np.abs(q.photon_ground_minus_survey-q.photon_ground_minus_survey.median())):.2f})")
    print(f"  same pairs via ATL08   : {q.dH.median():+.2f} m")
    print(f"  difference             : "
          f"{q.photon_ground_minus_survey.median()-q.dH.median():+.2f} m")

    n = len(panels)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.0 * nrow))
    for ax, (r, lat, h, cl, gmed) in zip(np.ravel(axes), panels):
        for k, (nm, col) in CLS.items():
            s = cl == k
            if s.any():
                ax.scatter(lat[s], h[s], s=2.2, color=col, lw=0,
                           label=nm if ax is np.ravel(axes)[0] else None)
        ax.axhline(r.H_icesat2, color=BLUE, lw=1.8, label="ATL08 terrain fit"
                   if ax is np.ravel(axes)[0] else None)
        ax.axhline(r.H_survey, color=RED, lw=1.8, ls="--", label="old bathymetry"
                   if ax is np.ravel(axes)[0] else None)
        ax.set_title(f"{r.stratum}\ndH={r.dH:+.2f} m", fontsize=8)
        ax.set_ylim(min(r.H_survey, gmed) - 6, max(r.H_survey, gmed) + 12)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=6.5, markerscale=3, loc="upper left")
    fig.suptitle("QA2 · ATL03 photons with ATL08 classes, the fitted terrain, and the "
                 "old bathymetry   ·   dH = ICESat-2 − survey", fontsize=12, y=1.0)
    fig.supylabel("height (m, common frame)", fontsize=9)
    fig.supxlabel("latitude", fontsize=9)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"QA2_photon_profiles.{e}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'QA2_photon_profiles.png'}")


if __name__ == "__main__":
    main()
