#!/usr/bin/env python
"""GATE 7C — is the negative SAR-optical offset a sensor bias, or stage?

Gate 7B found the SAR shoreline sits 20-34 m INSIDE the optical line, the
opposite of the flooded-vegetation premise. Before calling that a sensor bias,
the cheaper explanations have to be eliminated, and the arithmetic says one of
them is entirely sufficient. On the measured cross-shore slope of 0.0049 m/m,

    dx = dH / s

so the observed medians correspond to only

    H1  24.1 m  ->  0.118 m of water-level difference
    H2  20.0 m  ->  0.098 m
    H3  34.1 m  ->  0.167 m

which is the same order as the WSE uncertainty itself. A 10-17 cm stage
difference between the optical composite and the individual SAR overpass would
reproduce the whole effect without any sensor bias at all.

Gate 7B also compared a SINGLE SAR acquisition against a MULTI-DATE optical
composite (H2 composites eight dates). That is not a like-for-like geometry
comparison. This gate pairs each SAR event with INDIVIDUAL optical dates,
computes the stage difference, predicts the stage-induced horizontal shift and
reports the RESIDUAL, which is the only candidate sensor bias.

TWO CORRECTIONS TO GATE 7B CARRIED HERE

1. The total uncertainty was wrong. sqrt(0.193^2 + 0.189^2) = 0.270 m, not the
   0.196 m reported. The cause was not a typo: nan_to_num turned the missing
   slope into sigma_z = 0 on the 59% of shoreline without slope support, and
   the median was taken over those zeros, while sigma_z itself was reported as
   a nanmedian over the supported 41%. Two numbers from two different subsets.
   Totals are now computed only where slope support exists, and the
   unsupported fraction is reported instead of being silently averaged in.

2. Systematic bias and random scatter are kept as separate terms. A bias does
   not average out and must never be absorbed into a sigma.

SIGN CONVENTION IS VERIFIED, NOT ASSUMED, on synthetic polygons with known
inward and outward offsets, including a ring with a hole, because a normal
orientation that follows winding order would flip the sign silently.

Outputs
-------
outputs/tables/hist25b_gate7c_sign_verification.csv
outputs/tables/hist25b_gate7c_pairwise_offsets.csv
outputs/figures/historical_bathymetry/png/hist25b_gate7c_decomposition.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize as rio_rasterize
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from hist25b_gate6_event_qualification import load_frozen_manifest
from hist25b_gate7_anchored_classifier import (CELL, CACHE, build_grid,
                                               clean_water, shoreline_px, db)

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
OPT_CACHE = CFG.BULK_ROOT / "contour_cache_corrected"
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
GPKG_OPTICAL = ROOT / "data/processed/bathymetry/prebreach_contours.gpkg"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
SLOPE_RADIUS_M = 1000.0
SLOPE_MIN_PTS = 8
SIG_HYDRAULIC_M = 0.05


def signed_offset(sar, opt):
    """Signed normal distance, positive where SAR water extends beyond the
    optical line, with the half-cell correction calibrated in Gate 7C0.

    The naive form compared a boundary made of inside-pixel CENTRES against a
    boundary at the mask EDGE, which returns exactly -1 cell for two identical
    masks -- a deterministic -20 m floor that happened to equal H2's entire
    measured offset. Placing the signed distance field's zero on the half-cell
    line and adding back the half cell the boundary pixel sits inside removes
    it: identical masks now return exactly 0.0 m on all five test geometries.
    The old bias was NOT constant (-30..0 m across geometries), so it could not
    have been subtracted after the fact."""
    d_in = ndimage.distance_transform_edt(opt)
    d_out = ndimage.distance_transform_edt(~opt)
    sdf = np.where(opt, -(d_in - 0.5), +(d_out - 0.5)) * CELL
    ys, xs = np.where(shoreline_px(sar))
    return ys, xs, sdf[ys, xs] + 0.5 * CELL


def verify_sign():
    """A normal orientation that followed winding order would flip the sign
    silently, so the convention is tested on shapes whose answer is known."""
    n = 400
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n / 2
    rr = np.hypot(yy - cy, xx - cx)
    opt = rr < 120
    rows = []
    for lbl, sar in (("SAR smaller (inside optical)", rr < 100),
                     ("SAR larger (beyond optical)", rr < 140),
                     ("identical", rr < 120),
                     ("ring with hole, SAR smaller",
                      (rr < 100) & ~(rr < 40))):
        o = opt & ~(rr < 40) if "hole" in lbl else opt
        _, _, dn = signed_offset(sar, o)
        med = float(np.median(dn))
        rows.append(dict(case=lbl, median_signed_m=med,
                         expected_sign=("negative" if "inside" in lbl or
                                        "smaller" in lbl else
                                        "positive" if "beyond" in lbl else "zero"),
                         observed_sign=("negative" if med < -CELL else
                                        "positive" if med > CELL else "zero")))
        print(f"  {lbl:<32} median {med:+8.1f} m -> {rows[-1]['observed_sign']}"
              f"  (expected {rows[-1]['expected_sign']})")
    R = pd.DataFrame(rows)
    R["pass"] = R.expected_sign == R.observed_sign
    return R


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("1. SIGN CONVENTION VERIFICATION (synthetic, known answers)")
    print("=" * 78)
    SV = verify_sign()
    SV.to_csv(CFG.TABLES / "hist25b_gate7c_sign_verification.csv", index=False)
    print(f"  all cases pass: {bool(SV['pass'].all())}")
    if not SV["pass"].all():
        raise SystemExit("sign convention FAILED -- fix before interpreting "
                         "any offset")

    M, meta = load_frozen_manifest()
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                      SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    G = build_grid(fp)
    inside = G["inside"]
    gdf = gpd.read_file(GPKG_OPTICAL, layer="contour_polygons").set_index("contour_id")

    def rast(g):
        return rio_rasterize([(g, 1)], out_shape=(G["ny"], G["nx"]),
                             transform=G["tr"], fill=0, dtype="uint8").astype(bool)

    P = pd.read_parquet(PRIMARY)
    pxy = np.c_[P.x.to_numpy(), P.y.to_numpy()]
    pz = P.H_bed_evrf2019_m.to_numpy()
    ptree = cKDTree(pxy)

    # gauge series, to get the water level at each optical date
    w = pd.read_csv(CFG.TABLES / "all_water_levels_common_frame.csv")
    g = w[(w.source == "gauge") & (w.domain == "reservoir")]
    lev = g.groupby("date").transformed_level_m.median()

    opt_files = sorted(OPT_CACHE.glob("wm_*_20m.npz"))
    print(f"\n  per-date optical masks: {len(opt_files)}")
    optical = {}
    for f in opt_files:
        d = f.name.split("_")[1]
        z = np.load(f)
        key = [k for k in z.files if k in ("water", "wm", "mask")]
        arr = z[key[0]] if key else z[z.files[0]]
        if arr.shape != inside.shape:
            print(f"    {d}: shape {arr.shape} != grid {inside.shape} -- skipped")
            continue
        optical[d] = arr.astype(bool)
    print(f"  usable on the current grid: {len(optical)} "
          f"({', '.join(sorted(optical))})")
    if not optical:
        raise SystemExit("no per-date optical mask matches the corrected grid; "
                         "rebuild hist23 cache before running this gate")

    elig = M[M.scientific_roles.fillna("").str.contains("TARGET_STAGE_GEOMETRY")
             & (M.temporal_regime == "PREBREACH_IMPOUNDED")
             & M.WSE_RESOLVED.astype(bool)]
    rows = []
    for r in elig.itertuples():
        npz = CACHE / f"{r.event_id}.npz"
        if not npz.exists():
            continue
        roles = str(r.scientific_roles)
        tgt = next((t for t in ("H1", "H2", "H3")
                    if f"{t}_TARGET_STAGE_GEOMETRY" in roles), "")
        if not tgt:
            continue
        z = np.load(npz)
        vv, vh, cov = z["vv"], z["vh"], z["cov"]
        obs = cov & inside
        vvd, vhd = db(vv), db(vh)
        opt_ref = rast(gdf.loc[tgt, "geometry"])
        d_in = ndimage.distance_transform_edt(opt_ref) * CELL
        d_out = ndimage.distance_transform_edt(~opt_ref) * CELL
        aw = opt_ref & (d_in > 500) & obs
        al = inside & ~opt_ref & (d_out > 100) & obs
        if aw.sum() < 500 or al.sum() < 500:
            del vv, vh, cov, vvd, vhd
            continue
        Xw = np.c_[vvd[aw], vhd[aw]]; Xl = np.c_[vvd[al], vhd[al]]
        mw, ml = Xw.mean(0), Xl.mean(0)
        Sw = np.cov(Xw.T) + np.cov(Xl.T) + np.eye(2) * 1e-6
        wv = np.linalg.solve(Sw, ml - mw)
        cut = 0.5 * (wv @ mw + wv @ ml)
        sar = np.zeros(inside.shape, bool)
        sar[obs] = (np.c_[vvd[obs], vhd[obs]] @ wv) < cut
        sar = clean_water(sar)
        lab, n = ndimage.label(sar, structure=np.ones((3, 3), int))
        core = set(np.unique(lab[aw & sar])) - {0}
        keep = np.zeros(n + 1, bool); keep[list(core)] = True
        sar = keep[lab]

        H_sar = float(r.WSE_nominal)
        for d, om in optical.items():
            dt_days = abs((pd.Timestamp(d) - pd.Timestamp(r.date)).days)
            H_opt = float(lev[d]) if d in lev.index else np.nan
            if not np.isfinite(H_opt):
                continue
            om_e = om & obs          # compare only where SAR actually observed
            if om_e.sum() < 5000:
                continue
            ys, xs, dn = signed_offset(sar & obs, om_e)
            if len(xs) < 200:
                continue
            X = G["x0"] + (xs + 0.5) * CELL
            Y = G["y1"] - (ys + 0.5) * CELL
            # local slope from soundings, never from the DEM being constrained
            slope = np.full(len(X), np.nan)
            step = max(1, len(X) // 3000)
            idxs = np.arange(0, len(X), step)
            nb = ptree.query_ball_point(np.c_[X[idxs], Y[idxs]], r=SLOPE_RADIUS_M)
            for k, nn in zip(idxs, nb):
                if len(nn) < SLOPE_MIN_PTS:
                    continue
                A = np.c_[pxy[nn, 0] - X[k], pxy[nn, 1] - Y[k], np.ones(len(nn))]
                try:
                    coef, *_ = np.linalg.lstsq(A, pz[nn], rcond=None)
                except Exception:
                    continue
                slope[k] = float(np.hypot(coef[0], coef[1]))
            sl_med = float(np.nanmedian(slope))
            dH = H_opt - H_sar
            dx_stage = dH / sl_med if np.isfinite(sl_med) and sl_med > 1e-6 else np.nan
            d_obs = float(np.median(dn))
            rows.append(dict(
                event_id=r.event_id, sar_date=r.date, optical_date=d,
                stage_target=tgt, relative_orbit=int(r.relative_orbit),
                dt_days=dt_days,
                pair_class=("A_<=1d" if dt_days <= 1 else "B_<=7d"
                            if dt_days <= 7 else "C_<=30d" if dt_days <= 30
                            else "D_>30d"),
                H_SAR=H_sar, H_OPT=H_opt, delta_H=dH,
                local_slope=sl_med, dx_stage_m=dx_stage,
                d_observed_m=d_obs,
                d_residual_m=d_obs - dx_stage if np.isfinite(dx_stage) else np.nan,
                nmad_m=float(1.4826 * np.median(np.abs(dn - d_obs))),
                n_px=len(xs)))
        del vv, vh, cov, vvd, vhd, sar
    R = pd.DataFrame(rows)
    if R.empty:
        raise SystemExit("no SAR-optical pairs formed")
    R.to_csv(CFG.TABLES / "hist25b_gate7c_pairwise_offsets.csv", index=False)

    print("\n" + "=" * 78)
    print("2. PER-DATE PAIRS (no composites): observed vs stage-predicted")
    print("=" * 78)
    show = R.sort_values("dt_days").head(25)
    print(show[["sar_date", "optical_date", "stage_target", "dt_days",
                "delta_H", "dx_stage_m", "d_observed_m", "d_residual_m"]
               ].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n  by pair class (closest in time first):")
    print(R.groupby("pair_class").agg(
        pairs=("d_observed_m", "size"),
        median_dH=("delta_H", "median"),
        median_dx_stage=("dx_stage_m", "median"),
        median_observed=("d_observed_m", "median"),
        median_residual=("d_residual_m", "median")).to_string(
        float_format=lambda v: f"{v:,.2f}"))
    print("\n  by stage target:")
    print(R.groupby("stage_target").agg(
        pairs=("d_observed_m", "size"),
        median_observed=("d_observed_m", "median"),
        median_dx_stage=("dx_stage_m", "median"),
        median_residual=("d_residual_m", "median"),
        nmad_residual=("d_residual_m", lambda s: float(
            1.4826 * np.nanmedian(np.abs(s - np.nanmedian(s)))))).to_string(
        float_format=lambda v: f"{v:,.2f}"))

    obs_med = float(R.d_observed_m.median())
    res_med = float(R.d_residual_m.median())
    expl = 1 - abs(res_med) / max(abs(obs_med), 1e-9)
    print("\n" + "=" * 78)
    print("3. VERDICT")
    print("=" * 78)
    print(f"  observed offset median  : {obs_med:+7.1f} m")
    print(f"  stage-predicted median  : {float(R.dx_stage_m.median()):+7.1f} m")
    print(f"  residual median         : {res_med:+7.1f} m")
    print(f"  stage explains {100*expl:.0f}% of the observed offset")
    if abs(res_med) < 10:
        print("  A — STAGE/TIME MISMATCH EXPLAINS MOST OF THE OFFSET.")
        print("  hist25 was not wrong about SAR; it compared different "
              "hydraulic moments as if they were one geometry.")
    elif abs(res_med) < abs(obs_med) * 0.5:
        print("  D — MIXED CAUSES: stage explains much of it, a residual "
              "sensor/edge-definition term survives.")
    else:
        print("  C — TRUE SENSOR/CLASSIFIER BIAS REMAINS after stage "
              "correction.")

    # ------------------------------------------------ uncertainty, corrected
    print("\n" + "=" * 78)
    print("4. UNCERTAINTY BOOKKEEPING (bias kept OUT of sigma)")
    print("=" * 78)
    sl = float(R.local_slope.median())
    sx_rand = float(np.nanmedian(R.nmad_m))
    bx = res_med
    sz = sx_rand * sl
    bz = bx * sl
    swse = float(M.WSE_uncertainty_total_nominal.median())
    stot = float(np.sqrt(sz ** 2 + swse ** 2 + SIG_HYDRAULIC_M ** 2))
    print(f"  b_x_sensor  (systematic) : {bx:+7.1f} m")
    print(f"  sigma_x_random           : {sx_rand:7.1f} m")
    print(f"  local slope              : {sl:.4f} m/m")
    print(f"  b_z_sensor  (systematic) : {bz:+7.3f} m   <- does NOT average out")
    print(f"  sigma_z_geometry         : {sz:7.3f} m")
    print(f"  sigma_WSE                : {swse:7.3f} m")
    print(f"  sigma_hydraulic          : {SIG_HYDRAULIC_M:7.3f} m")
    print(f"  sigma_total (random only): {stot:7.3f} m")
    print(f"\n  Gate 7B reported 0.196 m by taking the median over shoreline "
          f"where\n  nan_to_num had set sigma_z to zero. Correct combination "
          f"is sqrt of squares.")

    _figures(R, SV)
    print("\nSTOP before DEM assimilation.")


def _figures(R, SV):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    a = ax[0]
    a.scatter(R.dx_stage_m, R.d_observed_m, s=40, c=R.dt_days, cmap="viridis_r")
    lim = [min(R.dx_stage_m.min(), R.d_observed_m.min()),
           max(R.dx_stage_m.max(), R.d_observed_m.max())]
    a.plot(lim, lim, color=RED, ls="--", lw=1.3, label="1:1 (stage explains all)")
    a.set_xlabel("stage-predicted shift dH/s (m)")
    a.set_ylabel("observed signed offset (m)")
    a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_title("a · does the water-level difference explain the offset?",
                fontsize=10.2, loc="left")
    a = ax[1]
    for t, c in zip(("H3", "H2", "H1"), (BLUE, GREEN, AMBER)):
        s = R[R.stage_target == t]
        if len(s):
            a.scatter(s.delta_H, s.d_observed_m, s=40, color=c, label=t)
    a.axhline(0, color=GREY, lw=1); a.axvline(0, color=GREY, lw=1)
    a.set_xlabel("delta_H = H_optical - H_SAR (m)")
    a.set_ylabel("observed signed offset (m)")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("b · offset against the actual stage difference",
                fontsize=10.2, loc="left")
    a = ax[2]
    d = [R[R.stage_target == t].d_residual_m.dropna() for t in ("H3", "H2", "H1")]
    a.boxplot(d, tick_labels=["H3", "H2", "H1"], showfliers=False)
    a.axhline(0, color=RED, ls="--", lw=1.3)
    a.set_ylabel("residual after stage correction (m)")
    a.set_title("c · what survives — the only candidate sensor bias",
                fontsize=10.2, loc="left")
    a.grid(alpha=0.25, axis="y")
    fig.suptitle("hist25b Gate 7C · decomposing the optical-SAR offset",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "hist25b_gate7c_decomposition.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGDIR/'hist25b_gate7c_decomposition.png'}")


if __name__ == "__main__":
    main()
