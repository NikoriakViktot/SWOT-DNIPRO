#!/usr/bin/env python
"""K0 (freeze/confirm inputs) + K3 (build post-breach morphological prior,
ZERO soundings used) + K4 (freeze the prior, then falsification-test it
against historical sounding depths -- the prior must never have seen these
depths before this test).

Data source for K3: outputs/tables/water_body_objects.parquet (already
built by phase20_water_objects.py from 32 POST_BREACH Sentinel-2 dates,
main-channel/connectivity classification) -- reused, not rebuilt from raw
imagery, per this session's own "don't redo what already exists" practice.

Outputs
-------
data/processed/bathymetry/morphology_prior_v1.tif   (multi-band: P_former_channel, distance_to_channel_m, n_dates_observed)
outputs/tables/morphology_prior_freeze_manifest.csv
outputs/tables/morphology_prior_validation.csv
outputs/figures/K3_morphology_prior.png
outputs/figures/K4_prior_vs_soundings.png
"""
from __future__ import annotations

import hashlib
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
import pyproj
import rasterio
from rasterio import features
from scipy import stats as sstats
from scipy.ndimage import distance_transform_edt, binary_dilation
from shapely import wkt as shwkt
from shapely.ops import transform as shp_transform

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

CELL_M = 250.0  # canonical project resolution (hist20_export_dem.py); 100m timed
                # out / was killed on the full 180x130km SA_2 extent
BATHY_DIR = CFG.ROOT / "data" / "processed" / "bathymetry"
_TF = pyproj.Transformer.from_crs("EPSG:4326", CFG.CRS_METRIC, always_xy=True).transform


def main() -> None:
    # ================================================================ K0
    print("=" * 70)
    print("K0 -- FREEZE / CONFIRM INPUTS")
    print("=" * 70)
    sa2 = SD.load("reservoir_full_pool_prebreach")
    sa2_m = shp_transform(_TF, sa2)
    print(f"KAKHOVKA_RESERVOIR_CORE (=SA_2): area={sa2_m.area/1e6:.1f} km2, "
          f"UNCHANGED from earlier freeze (spatial_domains.yaml never edited)")
    sd = pd.read_parquet(BATHY_DIR / "kakhovka_soundings_evrf2019.parquet")
    print(f"historical soundings: {len(sd)} points, common_vertical_frame="
          f"{sd.common_vertical_frame.iloc[0]!r}")
    print(f"horizontal CRS for this script: EPSG:32636 throughout "
          f"(sounding x/y already in this system per project convention)")
    print("K1: primary domain = KAKHOVKA_RESERVOIR_CORE only. "
          "DNIPROVSKE_RESERVOIR_UPSTREAM explicitly excluded.")
    print("K2: soundings are the only direct bed-elevation observations used as hard data "
          "(not built here -- reused as-is from hist14_bed_surface.py's input).")

    # ================================================================ common grid over SA_2
    minx, miny, maxx, maxy = sa2_m.buffer(1000).bounds
    minx, miny = np.floor(minx / CELL_M) * CELL_M, np.floor(miny / CELL_M) * CELL_M
    maxx, maxy = np.ceil(maxx / CELL_M) * CELL_M, np.ceil(maxy / CELL_M) * CELL_M
    W, H = int((maxx - minx) / CELL_M), int((maxy - miny) / CELL_M)
    transform = rasterio.transform.from_origin(minx, maxy, CELL_M, CELL_M)
    print(f"\ncommon grid: {W}x{H} @ {CELL_M:.0f} m over SA_2 + 1km buffer")

    # ================================================================ K3
    print("\n" + "=" * 70)
    print("K3 -- POST-BREACH MORPHOLOGICAL PRIOR (zero soundings used)")
    print("=" * 70)
    wb = pd.read_parquet(CFG.ROOT / "outputs/tables/water_body_objects.parquet")
    post = wb[(wb.period == "POST_BREACH") & (wb.wkt != "")].copy()
    print(f"POST_BREACH components with geometry: {len(post)}, "
          f"{post.date.nunique()} unique dates")

    # NOTE: the first attempt at this loop used shapely unary_union() on the
    # full per-date polygon set followed by .buffer(2000) on that union.
    # These per-date water-body polygons come from pixel-level vectorization
    # of Sentinel-2 masks and are extremely complex (one single date/polygon
    # reached >500k vertices, 8+ MB of WKT) -- unioning then buffering such
    # geometries in vector space exploded RSS to ~15GB and was OOM-killed by
    # the kernel. Fixed by rasterizing the (simplified) polygons directly --
    # rasterize does not require a prior union -- and doing the 2km "observed
    # vicinity" buffer as a raster binary_dilation instead of a vector buffer.
    SIMPLIFY_M = CELL_M / 2  # well below grid resolution; does not change the mask
    BUFFER_CELLS = int(round(2000.0 / CELL_M))

    channel_count = np.zeros((H, W), dtype=np.int16)
    obs_count = np.zeros((H, W), dtype=np.int16)
    for date, g in post.groupby("date"):
        all_geoms = [shwkt.loads(w).simplify(SIMPLIFY_M) for w in g.wkt]
        all_geoms = [gm for gm in all_geoms if gm is not None and not gm.is_empty]
        if all_geoms:
            water_mask = features.rasterize([(gm, 1) for gm in all_geoms], out_shape=(H, W),
                                            transform=transform, fill=0, dtype="uint8").astype(bool)
        else:
            water_mask = np.zeros((H, W), bool)
        obs_mask = binary_dilation(water_mask, iterations=BUFFER_CELLS) if water_mask.any() else water_mask
        obs_count += obs_mask.astype(np.int16)

        main = g[g.is_main_component]
        main_geoms = [shwkt.loads(w).simplify(SIMPLIFY_M) for w in main.wkt]
        main_geoms = [gm for gm in main_geoms if gm is not None and not gm.is_empty]
        if main_geoms:
            chan_mask = features.rasterize([(gm, 1) for gm in main_geoms], out_shape=(H, W),
                                           transform=transform, fill=0, dtype="uint8").astype(bool)
            channel_count += (chan_mask & obs_mask).astype(np.int16)

    with np.errstate(divide="ignore", invalid="ignore"):
        P_former_channel = np.where(obs_count > 0, channel_count / np.maximum(obs_count, 1), np.nan)

    # distance to the >=50%-persistence channel core (in metres, via EDT)
    persistent_channel = P_former_channel >= 0.5
    dist_to_channel_m = distance_transform_edt(~persistent_channel) * CELL_M

    print(f"P_former_channel: {np.nanmean(P_former_channel):.3f} mean, "
          f"{(np.nan_to_num(P_former_channel) >= 0.5).sum()} cells >=0.5 persistence "
          f"({(np.nan_to_num(P_former_channel) >= 0.5).sum() * CELL_M**2 / 1e6:.1f} km2)")

    # ---- freeze -------------------------------------------------------------
    out_tif = BATHY_DIR / "morphology_prior_v1.tif"
    with rasterio.open(out_tif, "w", driver="GTiff", height=H, width=W, count=3,
                       dtype="float32", crs="EPSG:32636", transform=transform,
                       nodata=-9999, compress="deflate") as dst:
        dst.write(np.where(np.isnan(P_former_channel), -9999, P_former_channel).astype("float32"), 1)
        dst.write(dist_to_channel_m.astype("float32"), 2)
        dst.write(obs_count.astype("float32"), 3)
        dst.set_band_description(1, "P_former_channel")
        dst.set_band_description(2, "distance_to_channel_m")
        dst.set_band_description(3, "n_dates_observed")
    sha = hashlib.sha256(out_tif.read_bytes()).hexdigest()[:16]
    print(f"\n-> {out_tif}  (K4 FREEZE, sha256_16={sha})")
    pd.DataFrame([{"file": str(out_tif), "sha256_16": sha, "n_post_breach_dates_used": post.date.nunique(),
                  "cell_m": CELL_M, "crs": "EPSG:32636"}]).to_csv(
        CFG.TABLES / "morphology_prior_freeze_manifest.csv", index=False)

    # ---- K3 figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(P_former_channel, extent=(minx, maxx, miny, maxy), origin="upper",
                   cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="P_former_channel (post-breach persistence, 0-1)")
    xs, ys = sa2_m.exterior.xy if hasattr(sa2_m, "exterior") else (list(sa2_m.geoms)[0].exterior.xy)
    ax.plot(xs, ys, color="black", lw=1)
    ax.set_title(f"K3 -- post-breach morphological prior, {post.date.nunique()} dates, "
                "ZERO soundings used", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K3_morphology_prior.png", dpi=150)
    print(f"-> {CFG.FIG / 'K3_morphology_prior.png'}")

    # ================================================================ K4 falsification test
    print("\n" + "=" * 70)
    print("K4 -- FALSIFICATION TEST: does the (sounding-blind) prior predict")
    print("       where historical soundings are actually deeper?")
    print("=" * 70)
    inv = ~transform
    cols, rows = inv * (sd.x.values, sd.y.values)
    cols, rows = np.clip(cols.astype(int), 0, W - 1), np.clip(rows.astype(int), 0, H - 1)
    sd = sd.copy()
    sd["P_former_channel"] = P_former_channel[rows, cols]
    sd["distance_to_channel_m"] = dist_to_channel_m[rows, cols]
    sd["n_dates_observed"] = obs_count[rows, cols]
    valid = sd.P_former_channel.notna() & (sd.n_dates_observed > 0)
    sv = sd[valid]
    print(f"soundings with valid prior coverage: {len(sv)} / {len(sd)}")

    r_p, p_p = sstats.pearsonr(sv.P_former_channel, -sv.H_bed_evrf2019_m)  # deeper = more negative-ish bed; use -H as "depth-like"
    rho_p, prho_p = sstats.spearmanr(sv.P_former_channel, -sv.H_bed_evrf2019_m)
    r_d, p_d = sstats.pearsonr(sv.distance_to_channel_m, sv.H_bed_evrf2019_m)
    rho_d, prho_d = sstats.spearmanr(sv.distance_to_channel_m, sv.H_bed_evrf2019_m)
    print(f"bed elevation vs P_former_channel: Pearson r={r_p:+.3f} (p={p_p:.1e}), "
          f"Spearman rho={rho_p:+.3f} (p={prho_p:.1e})  [higher P_channel -> LOWER bed elevation expected]")
    print(f"bed elevation vs distance_to_channel_m: Pearson r={r_d:+.3f} (p={p_d:.1e}), "
          f"Spearman rho={rho_d:+.3f} (p={prho_d:.1e})  [further from channel -> HIGHER bed elevation expected]")

    deepest_decile_thr = sv.H_bed_evrf2019_m.quantile(0.10)
    deep = sv.H_bed_evrf2019_m <= deepest_decile_thr
    print(f"\ndeepest decile (H_bed <= {deepest_decile_thr:.2f} m): "
          f"mean P_former_channel = {sv.loc[deep,'P_former_channel'].mean():.3f} "
          f"vs shallower 90%: {sv.loc[~deep,'P_former_channel'].mean():.3f}")
    print(f"deepest decile: mean distance_to_channel_m = {sv.loc[deep,'distance_to_channel_m'].mean():.0f} m "
          f"vs shallower 90%: {sv.loc[~deep,'distance_to_channel_m'].mean():.0f} m")

    # roughness by morphology class (quick 3-class split)
    sv = sv.assign(morph_class=pd.cut(sv.P_former_channel, [0, 0.2, 0.5, 1.0],
                                      labels=["FLOODPLAIN_LOW_P", "TRANSITION", "CHANNEL_HIGH_P"],
                                      include_lowest=True))
    print("\nbed elevation by morphology class (median, n):")
    print(sv.groupby("morph_class", observed=True).H_bed_evrf2019_m.agg(["median", "count"]).to_string())

    val_df = pd.DataFrame([{
        "n_soundings_valid": len(sv),
        "pearson_r_P_channel_vs_neg_elevation": r_p, "pearson_p": p_p,
        "spearman_rho_P_channel_vs_neg_elevation": rho_p, "spearman_p": prho_p,
        "pearson_r_distance_vs_elevation": r_d, "pearson_p_dist": p_d,
        "spearman_rho_distance_vs_elevation": rho_d, "spearman_p_dist": prho_d,
        "deepest_decile_mean_Pchannel": sv.loc[deep, "P_former_channel"].mean(),
        "shallow90_mean_Pchannel": sv.loc[~deep, "P_former_channel"].mean(),
        "deepest_decile_mean_dist_channel_m": sv.loc[deep, "distance_to_channel_m"].mean(),
        "shallow90_mean_dist_channel_m": sv.loc[~deep, "distance_to_channel_m"].mean(),
    }])
    val_df.to_csv(CFG.TABLES / "morphology_prior_validation.csv", index=False)
    print(f"\n-> {CFG.TABLES / 'morphology_prior_validation.csv'}")

    # NOTE (sign convention): rho_p is Spearman(P_former_channel, -H_bed) --
    # already negated, so higher-is-better ("GO" wants rho_p > threshold).
    # rho_d is Spearman(distance_to_channel_m, H_bed) computed against the
    # RAW (non-negated) elevation -- further from the channel is physically
    # expected to mean HIGHER (shallower) bed elevation, i.e. a POSITIVE
    # rho_d, not negative. The first version of this gate checked
    # `rho_d < -0.10`, which silently required the wrong sign and produced a
    # spurious MARGINAL verdict despite rho_d=+0.60 (p=0.0) -- fixed here.
    verdict = "GO" if (rho_p > 0.15 and prho_p < 0.01 and rho_d > 0.10 and prho_d < 0.01) else \
             "MARGINAL" if (prho_p < 0.05 or prho_d < 0.05) else "NO-GO"
    print(f"\nVERDICT (K4 falsification gate): {verdict} -- prior may "
          f"{'be used as a kriging covariate (K9 M2/M3)' if verdict=='GO' else 'need review before use' if verdict=='MARGINAL' else 'NOT be used as a kriging covariate'}")

    # ---- K4 figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(sv.P_former_channel, sv.H_bed_evrf2019_m, s=4, alpha=0.3, color="#236f8c")
    axes[0].set_xlabel("P_former_channel (prior, no soundings used)")
    axes[0].set_ylabel("historical bed elevation, m EVRF2019")
    axes[0].set_title(f"rho={rho_p:+.3f} (p={prho_p:.1e})", fontsize=9)
    axes[1].scatter(sv.distance_to_channel_m, sv.H_bed_evrf2019_m, s=4, alpha=0.3, color="#c1402a")
    axes[1].set_xlabel("distance to persistent channel, m")
    axes[1].set_ylabel("historical bed elevation, m EVRF2019")
    axes[1].set_title(f"rho={rho_d:+.3f} (p={prho_d:.1e})", fontsize=9)
    fig.suptitle("K4 -- falsification test: sounding-blind morphology prior vs. historical soundings")
    fig.tight_layout()
    fig.savefig(CFG.FIG / "K4_prior_vs_soundings.png", dpi=150)
    print(f"-> {CFG.FIG / 'K4_prior_vs_soundings.png'}")


if __name__ == "__main__":
    main()
