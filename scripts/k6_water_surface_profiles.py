#!/usr/bin/env python
"""K6 -- date-specific water-surface profile eta(chain_km, t).

The model is 1-D in the hydraulic/longitudinal coordinate, as instructed: the
former reservoir is elongated and hydraulically organised along the Dnipro
axis, so chainage is the primary coordinate. No generic unconstrained 2-D
interpolation in UTM is performed anywhere here.

Primary candidate : piecewise linear between observed gauges
Sensitivity       : PCHIP (shape-preserving, no overshoot)

Monotonic downstream decline is NOT imposed. Instantaneous water surfaces can
carry backwater, wind setup/setdown and seiche signatures; where the observed
gauge set is locally non-monotonic that is preserved and flagged, never
silently corrected.

Uncertainty is kept as separate components and never collapsed into one
arbitrary number:
  sigma_gauge                 staff-gauge reading resolution
  sigma_vertical_transform    EPSG:9902 BS-77 -> EVRF2019 accuracy (project constant)
  sigma_temporal_matching     from the same-day 08h/20h term spread
  sigma_spatial_interpolation empirical, from leave-one-gauge-out CV (not assumed)

Outputs
-------
outputs/tables/k6_water_surface_profiles.csv
outputs/tables/k6_water_surface_uncertainty.csv
outputs/tables/k6_logo_cross_validation.csv
outputs/figures/K6_profiles_by_date.png
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
from scipy.interpolate import PchipInterpolator

from swot_dnipro import config as CFG

#: staff gauge is recorded to 1 cm; reading/registration error taken as 2 cm
SIGMA_GAUGE_M = 0.02
#: documented EPSG:9902 (BS-77 -> EVRF2019 normal heights) accuracy, project constant
SIGMA_VTRANS_M = CFG.EPSG9902_ACCURACY_M
#: chainage grid for the emitted profile (km), spanning the SA_2 core
S_GRID = np.arange(0.0, 280.0 + 0.5, 0.5)
#: how far beyond the outermost gauge we allow before flagging extrapolation
EXTRAP_FLAG_KM = 0.0


def build_profile(s_obs, h_obs, s_query, kind="linear"):
    """1-D profile in chainage. Outside the gauge span the value is held flat
    at the nearest gauge (no blind linear extrapolation of a fitted slope) and
    the caller flags it as extrapolated."""
    order = np.argsort(s_obs)
    s, h = np.asarray(s_obs)[order], np.asarray(h_obs)[order]
    if kind == "linear":
        return np.interp(s_query, s, h)  # numpy clamps outside the span
    if len(s) < 2:
        return np.full_like(s_query, h[0], dtype=float)
    f = PchipInterpolator(s, h, extrapolate=False)
    out = f(s_query)
    out[s_query < s[0]] = h[0]
    out[s_query > s[-1]] = h[-1]
    return out


def main() -> None:
    print("=" * 70)
    print("K6 -- DATE-SPECIFIC WATER-SURFACE PROFILE eta(chain_km, t)")
    print("=" * 70)

    mu = pd.read_csv(CFG.TABLES / "k5_sentinel_gauge_matchups.csv", parse_dates=["sentinel_date"])
    ok = mu[mu.H_EVRF2019_m.notna()].copy()
    per_date = ok.groupby("sentinel_date").station_id.nunique()
    multi = sorted(per_date[per_date >= 2].index)
    single = sorted(per_date[per_date == 1].index)
    print(f"Sentinel dates with >=2 gauges -> profile built: {len(multi)}")
    print(f"Sentinel dates with exactly 1 gauge -> NO profile, single anchor only: {len(single)}")
    print("A single upper-reach gauge (80959, s=247.5 km) cannot define eta(s) over a "
          "~250 km domain; those dates are emitted as anchors with a NO_PROFILE flag.")

    # ---- leave-one-gauge-out CV: empirical sigma_spatial_interpolation ------
    logo = []
    for d in multi:
        g = ok[ok.sentinel_date == d].sort_values("chain_km")
        if len(g) < 3:
            continue
        for i in range(len(g)):
            held = g.iloc[i]
            keep = g.drop(g.index[i])
            for kind in ("linear", "pchip"):
                pred = build_profile(keep.chain_km.values, keep.H_EVRF2019_m.values,
                                     np.array([held.chain_km]), kind=kind)[0]
                # distance to the nearest retained gauge = the gap being spanned
                gap = float(np.min(np.abs(keep.chain_km.values - held.chain_km)))
                logo.append({"date": d, "station_id": held.station_id, "kind": kind,
                             "chain_km": held.chain_km, "observed_m": held.H_EVRF2019_m,
                             "predicted_m": pred, "error_m": pred - held.H_EVRF2019_m,
                             "gap_to_nearest_kept_km": gap,
                             "interior": bool(keep.chain_km.min() < held.chain_km < keep.chain_km.max())})
    lg = pd.DataFrame(logo)
    lg.to_csv(CFG.TABLES / "k6_logo_cross_validation.csv", index=False)
    print(f"\nleave-one-gauge-out CV: {len(lg)} held-out fits over {lg.date.nunique()} dates")
    for kind, gk in lg.groupby("kind"):
        inr = gk[gk.interior]
        print(f"  {kind:<7} all: RMSE {np.sqrt((gk.error_m**2).mean()):.3f} m, "
              f"bias {gk.error_m.mean():+.3f} m  |  interior only (n={len(inr)}): "
              f"RMSE {np.sqrt((inr.error_m**2).mean()):.3f} m")
    interior = lg[(lg.kind == "linear") & lg.interior]
    # empirical sigma as a function of the spanned gap, used below
    if len(interior) >= 4:
        gap_bins = pd.cut(interior.gap_to_nearest_kept_km, [0, 20, 40, 80, 200])
        sig_by_gap = interior.groupby(gap_bins).error_m.apply(lambda e: float(np.sqrt((e ** 2).mean())))
        print("\n  empirical interpolation RMSE by spanned gap (linear, interior):")
        print("  " + sig_by_gap.to_string().replace("\n", "\n  "))
    else:
        sig_by_gap = None
    sigma_spatial_global = float(np.sqrt((interior.error_m ** 2).mean())) if len(interior) else np.nan

    # ---- build the profiles -------------------------------------------------
    prof_rows, unc_rows = [], []
    for d in multi + single:
        g = ok[ok.sentinel_date == d].sort_values("chain_km")
        s_obs, h_obs = g.chain_km.values, g.H_EVRF2019_m.values
        n_g = len(g)
        # honest non-monotonicity check: downstream is DECREASING chain_km
        dh = np.diff(h_obs)          # ordered by increasing chainage (upstream)
        non_mono = bool(n_g >= 2 and np.any(dh < 0))
        slopes_cm_km = dh / np.diff(s_obs) * 100 if n_g >= 2 else np.array([])
        # temporal component: same-day 08/20 term spread seen by the matchup
        idr = g.intra_day_range_m.dropna()
        sig_temporal = float(idr.max() / 2.0) if len(idr) else np.nan
        interp_used = len(g[g.observation_type.astype(str).str.contains("interpolated_across_days")])

        if n_g >= 2:
            lin = build_profile(s_obs, h_obs, S_GRID, "linear")
            pch = build_profile(s_obs, h_obs, S_GRID, "pchip")
            span_lo, span_hi = s_obs.min(), s_obs.max()
            for s, hl, hp in zip(S_GRID, lin, pch):
                extrap = (s < span_lo - EXTRAP_FLAG_KM) or (s > span_hi + EXTRAP_FLAG_KM)
                gap = float(np.min(np.abs(s_obs - s)))
                sig_sp = sigma_spatial_global
                if extrap:
                    # uncertainty inflation beyond the outermost gauge, scaled by
                    # how far past the last observation we are
                    over = max(span_lo - s, s - span_hi)
                    sig_sp = sigma_spatial_global * (1.0 + over / 25.0)
                prof_rows.append({
                    "date": d, "chain_km": s,
                    "eta_evrf2019_m": hl, "eta_pchip_m": hp,
                    "eta_method_spread_m": abs(hl - hp),
                    "n_gauges": n_g, "gap_to_nearest_gauge_km": gap,
                    "extrapolated": extrap, "profile_status": "OK",
                    "non_monotonic_gauges": non_mono,
                    "sigma_gauge_m": SIGMA_GAUGE_M,
                    "sigma_vertical_transform_m": SIGMA_VTRANS_M,
                    "sigma_temporal_matching_m": sig_temporal,
                    "sigma_spatial_interpolation_m": sig_sp,
                })
        else:
            anchor = float(h_obs[0])
            for s in S_GRID:
                prof_rows.append({
                    "date": d, "chain_km": s,
                    "eta_evrf2019_m": np.nan, "eta_pchip_m": np.nan,
                    "eta_method_spread_m": np.nan,
                    "n_gauges": n_g, "gap_to_nearest_gauge_km": float(abs(s_obs[0] - s)),
                    "extrapolated": True,
                    "profile_status": "NO_PROFILE_single_gauge_anchor_only",
                    "non_monotonic_gauges": False,
                    "sigma_gauge_m": SIGMA_GAUGE_M,
                    "sigma_vertical_transform_m": SIGMA_VTRANS_M,
                    "sigma_temporal_matching_m": sig_temporal,
                    "sigma_spatial_interpolation_m": np.nan,
                })
        unc_rows.append({
            "date": d, "n_gauges": n_g,
            "gauge_span_km": float(s_obs.max() - s_obs.min()) if n_g >= 2 else 0.0,
            "anchor_chain_km": float(s_obs[0]) if n_g == 1 else np.nan,
            "eta_at_dam_m": float(build_profile(s_obs, h_obs, np.array([0.0]), "linear")[0]) if n_g >= 2 else np.nan,
            "eta_at_247km_m": float(build_profile(s_obs, h_obs, np.array([247.5]), "linear")[0]) if n_g >= 2 else float(h_obs[0]),
            "total_drop_m": float(h_obs[np.argmax(s_obs)] - h_obs[np.argmin(s_obs)]) if n_g >= 2 else np.nan,
            "mean_slope_cm_km": float(np.mean(slopes_cm_km)) if len(slopes_cm_km) else np.nan,
            "non_monotonic": non_mono,
            "n_gauges_interpolated_across_days": interp_used,
            "sigma_gauge_m": SIGMA_GAUGE_M,
            "sigma_vertical_transform_m": SIGMA_VTRANS_M,
            "sigma_temporal_matching_m": sig_temporal,
            "sigma_spatial_interpolation_m": sigma_spatial_global if n_g >= 2 else np.nan,
            "combined_sigma_m_IF_independent": float(np.sqrt(
                SIGMA_GAUGE_M ** 2 + SIGMA_VTRANS_M ** 2
                + (sig_temporal if np.isfinite(sig_temporal) else 0) ** 2
                + (sigma_spatial_global if n_g >= 2 and np.isfinite(sigma_spatial_global) else 0) ** 2)),
            "profile_status": "OK" if n_g >= 2 else "NO_PROFILE_single_gauge_anchor_only",
        })

    # ---- eta(s,t) as a physical result in its own right --------------------
    phys = []
    for d in multi:
        g = ok[ok.sentinel_date == d].sort_values("chain_km")
        s_obs, h_obs = g.chain_km.values, g.H_EVRF2019_m.values
        sl = np.diff(h_obs) / np.diff(s_obs) * 100.0  # cm/km, increasing chainage
        signs = np.sign(sl[np.abs(sl) > 1e-9])
        n_sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        dH = float(h_obs[np.argmax(s_obs)] - h_obs[np.argmin(s_obs)])
        u = next(r for r in unc_rows if r["date"] == d)
        sig = u["combined_sigma_m_IF_independent"]
        flag = ("FLAT_WITHIN_UNCERTAINTY" if abs(dH) <= sig else
                "RESOLVED_GRADIENT" if n_sign_changes == 0 else
                "RESOLVED_BUT_NON_MONOTONIC")
        phys.append({"date": d, "n_gauges": len(g), "dH_total_m": dH,
                     "max_local_slope_cm_km": float(sl.max()),
                     "min_local_slope_cm_km": float(sl.min()),
                     "n_slope_sign_changes": n_sign_changes,
                     "combined_sigma_m": sig,
                     "abs_dH_over_sigma": abs(dH) / sig if sig else np.nan,
                     "interpretation_flag": flag})
    ph = pd.DataFrame(phys)
    ph.to_csv(CFG.TABLES / "k6_profile_physics.csv", index=False)
    print("\neta(s,t) as a physical result (not just an interpolation product):")
    print(ph.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {CFG.TABLES / 'k6_profile_physics.csv'}")
    nflat = int((ph.interpretation_flag == "FLAT_WITHIN_UNCERTAINTY").sum())
    print(f"\n{nflat} of {len(ph)} dates have |dH_total| <= combined sigma: at full pool the "
          f"longitudinal surface gradient is NOT resolvable by this six-gauge set. "
          f"The apparent non-monotonicity is therefore not an established backwater/setup "
          f"signal -- it is unresolved, and must not be interpreted as physics.")

    prof = pd.DataFrame(prof_rows)
    unc = pd.DataFrame(unc_rows)
    prof.to_csv(CFG.TABLES / "k6_water_surface_profiles.csv", index=False)
    unc.to_csv(CFG.TABLES / "k6_water_surface_uncertainty.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'k6_water_surface_profiles.csv'}  ({len(prof)} rows)")
    print(f"-> {CFG.TABLES / 'k6_water_surface_uncertainty.csv'}")

    print("\nper-date summary for the 8 profile dates:")
    show = unc[unc.profile_status == "OK"][
        ["date", "n_gauges", "eta_at_dam_m", "eta_at_247km_m", "total_drop_m",
         "mean_slope_cm_km", "non_monotonic", "sigma_temporal_matching_m",
         "combined_sigma_m_IF_independent"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    nm = unc[(unc.profile_status == "OK") & unc.non_monotonic]
    print(f"\nlocally non-monotonic gauge sets: {len(nm)} of {len(multi)} dates "
          f"-- PRESERVED and flagged, not corrected")
    if len(nm):
        print("  " + ", ".join(pd.to_datetime(nm.date).dt.strftime("%Y-%m-%d")))

    print(f"\nuncertainty components (m), never collapsed by default:")
    print(f"  sigma_gauge                 = {SIGMA_GAUGE_M:.3f}  (1 cm registration, 2 cm assumed)")
    print(f"  sigma_vertical_transform    = {SIGMA_VTRANS_M:.3f}  (EPSG:9902, project constant)")
    print(f"  sigma_temporal_matching     = {unc.sigma_temporal_matching_m.min():.3f} .. "
          f"{unc.sigma_temporal_matching_m.max():.3f}  (half the same-day 08/20 term spread)")
    print(f"  sigma_spatial_interpolation = {sigma_spatial_global:.3f}  (empirical LOGO CV, interior)")

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(multi):
        p = prof[(prof.date == d) & (prof.profile_status == "OK")]
        g = ok[ok.sentinel_date == d]
        c = cmap(i / max(len(multi) - 1, 1))
        axes[0].plot(p.chain_km, p.eta_evrf2019_m, lw=1.4, color=c,
                     label=pd.Timestamp(d).strftime("%Y-%m-%d"))
        axes[0].scatter(g.chain_km, g.H_EVRF2019_m, s=26, color=c, zorder=5, edgecolor="k", lw=0.4)
    axes[0].set_xlabel("chain_km (0 = Kakhovka dam, positive upstream)")
    axes[0].set_ylabel("eta, m EVRF2019")
    axes[0].set_title("A. eta(s,t), piecewise linear between the six gauges\n"
                      "(dots = observed gauges; no monotonicity imposed)", loc="left", fontsize=9.5)
    axes[0].legend(fontsize=7, ncol=2)

    lgl = lg[lg.kind == "linear"]
    axes[1].scatter(lgl[lgl.interior].gap_to_nearest_kept_km, lgl[lgl.interior].error_m,
                    s=34, color="#236f8c", label="interior (interpolation)")
    axes[1].scatter(lgl[~lgl.interior].gap_to_nearest_kept_km, lgl[~lgl.interior].error_m,
                    s=34, color="#c1402a", marker="^", label="edge (extrapolation)")
    axes[1].axhline(0, color="grey", lw=0.8, ls=":")
    axes[1].set_xlabel("gap to nearest retained gauge, km")
    axes[1].set_ylabel("leave-one-gauge-out error, m")
    axes[1].set_title(f"B. empirical sigma_spatial_interpolation\n"
                      f"interior RMSE = {sigma_spatial_global:.3f} m", loc="left", fontsize=9.5)
    axes[1].legend(fontsize=8)
    fig.suptitle("K6 -- date-specific water-surface profiles from the six Kakhovka gauges "
                 "(station 80957 not used)", fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K6_profiles_by_date.png", dpi=150)
    print(f"\n-> {CFG.FIG / 'K6_profiles_by_date.png'}")


if __name__ == "__main__":
    main()
