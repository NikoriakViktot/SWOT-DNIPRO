#!/usr/bin/env python
"""MS7B — is the headline slope sensitive to how EGG2015 is sampled?

``vertical.sample_grid`` takes the nearest cell of the 1' EGG2015 grid; the
validation chain (ms6b/ms7) now interpolates bilinearly, because nearest-cell
sampling puts centimetre steps between points ~200 m apart. This script asks
whether the per-overpass slopes of the headline result depend on that choice.

The production transect levels (``FigD_kakhovka_profile_points.csv``, the slope points,
``median_wse_evrs_m``) are rebuilt here from the ATL13 segments of each
transect. The segment sets match the production n_points exactly, and
median(h - zeta_bilinear) reproduces the production value to 0.0 mm on every
transect: the companion chain already interpolates. The nearest-cell variant is
therefore the sensitivity:

    S_nearest  : production transect level + [median(h - zeta_nearest)
                                             - median(h - zeta_bilinear)]
    S_bilinear : production (Theil-Sen of wse_m on chain_km, six beam medians)

Outputs: outputs/paper/validation/ms7b_slope_geoid_sampling.csv (per date)
         outputs/paper/validation/ms7b_slope_geoid_sampling_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from swot_dnipro import config as CFG
from swot_dnipro.vertical import sample_grid, sample_grid_bilinear

OUT = ROOT / "outputs/paper/validation"
SLOPES = ROOT / "outputs/tables/kakhovka_perdate_slopes_robust.csv"
PROFILE = ROOT / "outputs/figure_data/FigD_kakhovka_profile_points.csv"  # points make_transition_stats fits
SEGMENTS = CFG.ICESAT_ROOT / "data/processed/kakhovka_atl13_segments.parquet"


def main() -> None:
    sl = pd.read_csv(SLOPES)
    pm = pd.read_csv(PROFILE)
    pm = pm[pm.date.isin(sl.date)]
    seg = pd.read_parquet(SEGMENTS, columns=["time", "rgt", "beam", "lat", "lon", "h_wgs84_m"])
    seg["date"] = seg.time.dt.strftime("%Y-%m-%d")
    s = seg.merge(pm[["date", "rgt", "beam"]].drop_duplicates(), on=["date", "rgt", "beam"])
    s["Hb"] = s.h_wgs84_m - sample_grid_bilinear(CFG.EGG2015_TIF, s.lon.values, s.lat.values)
    s["Hn"] = s.h_wgs84_m - sample_grid(CFG.EGG2015_TIF, s.lon.values, s.lat.values)
    g = s.groupby(["date", "rgt", "beam"]).agg(Hb=("Hb", "median"), Hn=("Hn", "median"),
                                               n=("Hb", "size")).reset_index()
    m = pm.merge(g, on=["date", "rgt", "beam"], how="left")
    assert (m.n == m.n_points).all(), "segment sets differ from production"
    rep = float(np.abs(m.Hb - m.median_wse_evrs_m).max())
    assert rep < 1e-4, f"bilinear does not reproduce production ({rep:.4f} m)"
    m["wse_near"] = m.wse_m + (m.Hn - m.Hb)

    rows = []
    for d, x in m.groupby("date"):
        sb = 100 * stats.theilslopes(x.wse_m.values, x.chain_km.values)[0]
        sn = 100 * stats.theilslopes(x.wse_near.values, x.chain_km.values)[0]
        prod = float(sl.loc[sl.date == d, "slope_theilsen_cm_km"].iloc[0])
        rows.append(dict(date=d, period=sl.loc[sl.date == d, "period"].iloc[0],
                         n_transects=len(x), S_production_cm_km=prod, S_bilinear_cm_km=sb,
                         S_nearest_cm_km=sn, dS_cm_km=sn - sb,
                         max_transect_shift_m=float(np.abs(x.Hn - x.Hb).max())))
    D = pd.DataFrame(rows)
    assert np.allclose(D.S_bilinear_cm_km, D.S_production_cm_km, atol=1e-6), \
        "bilinear slopes do not reproduce the production slopes"
    D.to_csv(OUT / "ms7b_slope_geoid_sampling.csv", index=False)

    pre, post = D[D.period == "PRE_BREACH"], D[D.period == "POST_BREACH"]
    summ = dict(
        n_dates=len(D), n_pre=len(pre), n_post=len(post),
        max_abs_dS_cm_km=D.dS_cm_km.abs().max(), median_dS_cm_km=D.dS_cm_km.median(),
        pre_median_bilinear=pre.S_bilinear_cm_km.median(), pre_median_nearest=pre.S_nearest_cm_km.median(),
        post_median_bilinear=post.S_bilinear_cm_km.median(), post_median_nearest=post.S_nearest_cm_km.median(),
        contrast_bilinear=post.S_bilinear_cm_km.median() - pre.S_bilinear_cm_km.median(),
        contrast_nearest=post.S_nearest_cm_km.median() - pre.S_nearest_cm_km.median(),
        post_positive_bilinear=int((post.S_bilinear_cm_km > 0).sum()),
        post_positive_nearest=int((post.S_nearest_cm_km > 0).sum()),
        max_transect_shift_m=D.max_transect_shift_m.max(),
        production_reproduced_by_bilinear_m=rep,
        pre_median_abs_dS_cm_km=pre.dS_cm_km.abs().median(),
        post_median_abs_dS_cm_km=post.dS_cm_km.abs().median())
    worst = D.loc[D.dS_cm_km.abs().idxmax()]
    ci = sl.set_index("date").loc[worst.date, ["ts_lo_cm_km", "ts_hi_cm_km"]]
    summ["note_max"] = (f"max |dS| on {worst.date} ({worst.period}): transect shifts <= "
                        f"{worst.max_transect_shift_m * 1000:.1f} mm move a six-point Theil-Sen "
                        f"median across pairwise slopes; that fit's own 95 % CI is "
                        f"[{ci.ts_lo_cm_km:.1f}, {ci.ts_hi_cm_km:.1f}] cm/km")
    S = pd.DataFrame([dict(claim_id="SLOPE_GEOID_SAMPLING", statistic=k, value=v)
                      for k, v in summ.items()])
    S.to_csv(OUT / "ms7b_slope_geoid_sampling_summary.csv", index=False)
    print(S.to_string(index=False))


def frames() -> None:
    """Figure S10 and Sections 5.5 / S5.2: the per-overpass slope in alternative
    vertical frames, on the same points and overpasses as the production fit.

        production        wse_m (EGG2015, ATL13 + mean-tide term)
        egg2015_no_tide   median_wse_evrs_m (the tide-free companion value)
        plus_const_c      production + the mean reservoir closure c (V1)
        plus_c_nearest    production + the V1 closure of the nearest gauge
                          along the reach (piecewise constant)
        plus_c_linear     production + the V1 gradient of c along the reach

    The contrast (post - pre median) carries a percentile bootstrap interval
    (20 000 resamples of each period, seeded as make_transition_stats.py)."""
    sl = pd.read_csv(SLOPES)
    pm = pd.read_csv(PROFILE)
    pm = pm[pm.date.isin(sl.date)].copy()
    S = pd.read_csv(OUT / "ms7_summary.csv")
    E = pd.read_csv(OUT / "ms7_evidence.csv", low_memory=False)
    v1 = E[(E.claim_id == "V1_ATL13_GAUGE_CLOSURE") & E.included_primary.astype(str).str.lower().eq("true")]
    al = pd.read_csv(CFG.ICESAT_ROOT / "outputs/tables/kakhovka_alignment_by_station.csv")
    med = v1.groupby("station").closure_residual_m.median().rename("c").to_frame().join(
        al.set_index("name_en").distance_from_dam_km)
    c_mean = float(med.c.mean())
    grad = float(S.loc[(S.claim_id == "V1_ATL13_GAUGE_CLOSURE")
                       & (S.statistic == "gradient_along_reach_cm_per_km"), "value"].iloc[0]) / 100  # m/km
    km = med.distance_from_dam_km.values
    near = med.c.values[np.abs(pm.chain_km.values[:, None] - km[None, :]).argmin(1)]
    F = {"production": pm.wse_m, "egg2015_no_tide": pm.median_wse_evrs_m,
         "plus_const_c": pm.wse_m + c_mean, "plus_c_nearest": pm.wse_m + near,
         "plus_c_linear": pm.wse_m + grad * pm.chain_km}
    rows = []
    for d, idx in pm.groupby("date").groups.items():
        x = pm.loc[idx, "chain_km"].values
        per = sl.loc[sl.date == d, "period"].iloc[0]
        r = dict(date=d, period=per)
        for k, y in F.items():
            r[f"S_{k}"] = 100 * stats.theilslopes(y.loc[idx].values, x)[0]
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "ms7b_slope_frames.csv", index=False)
    rng = np.random.default_rng(CFG.SEED)
    out = []
    pre, post = T[T.period == "PRE_BREACH"], T[T.period == "POST_BREACH"]
    for k in F:
        a, b = pre[f"S_{k}"].values, post[f"S_{k}"].values
        dd = [np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a))) for _ in range(20000)]
        dS = (T[f"S_{k}"] - T["S_production"]).abs()
        out += [dict(claim_id="SLOPE_FRAMES", statistic="contrast_cm_per_km", variant=k,
                     value=np.median(b) - np.median(a), ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5)),
                dict(claim_id="SLOPE_FRAMES", statistic="max_abs_dS_cm_per_km", variant=k, value=dS.max()),
                dict(claim_id="SLOPE_FRAMES", statistic="n_overpasses_changed_over_0.01", variant=k,
                     value=int((dS > 0.01).sum())),
                dict(claim_id="SLOPE_FRAMES", statistic="post_positive", variant=k, value=int((b > 0).sum()))]
    out.append(dict(claim_id="SLOPE_FRAMES", statistic="c_gradient_cm_per_km", variant="plus_c_linear",
                    value=100 * grad, note="V1 gradient_along_reach_cm_per_km"))
    pd.DataFrame(out).to_csv(OUT / "ms7b_slope_frames_summary.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
    frames()
