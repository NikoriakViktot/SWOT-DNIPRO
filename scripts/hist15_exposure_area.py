#!/usr/bin/env python
"""HISTORICAL 15 — the exposure test, redone as an AREA fraction.

Tasks 2 and 3. This replaces hist12/E, which was invalid: it counted SOUNDING
POINTS above the dead-volume level, and the survey follows the navigable channel
and avoids the shallow margins, which is exactly where drying happens. A point
sample cannot estimate an area fraction.

The quantity Table 21 reports is

    area of drying (осушек) / area at the normal impoundment horizon

i.e. the fraction of each reach that emerges when the reservoir is drawn down
from NPG (16.0 m) to the dead-volume horizon GMO (12.7 m). On an interpolated
bed surface that is a straightforward area count: a cell dries if its bed sits
at or above GMO, since every cell inside the pre-breach footprint is below NPG
by construction.

    historical, by reach:  5.3, 2.6, 9.4, 26.0, 34.7 %   (ascending upstream)
    total:                 279 / 2155 km2 = 12.9 %

TWO RESTRICTIONS CARRIED FROM hist13, both real:

  * the Kakhovka HPP - Babyne boundary is not placeable (a ~30 km residual sits
    there), so reaches 1 and 2 are MERGED and compared against their combined
    historical fraction (26+14)/(495+532) = 3.9 %;
  * reach 5 lies mostly above the upper limit of the mapped pre-breach
    footprint, so it is reported as not resolvable rather than compared.

Task 3 is not a separate calculation: the fraction is computed on ALL FOUR
interpolated surfaces and the spread across them IS the interpolation
uncertainty.

Outputs
-------
outputs/tables/historical_exposure_area_validation.csv
outputs/figures/V13_exposure_area_validation.png
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
import pyproj

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD
from swot_dnipro import sword as SW

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
GMO_EVRF = 12.70 + 0.185          # dead-volume horizon in our frame
NPG_EVRF = 16.00 + 0.185
SINUOSITY = 1.36                  # axis -> channel, from hist13

# Reach limits on the CHANNEL axis. Nikopol and Verkhnia Tarasivka are placed to
# ~1 km by hist13; the upper limits are axis positions scaled by the sinuosity.
REACHES = [
    dict(reach_id="1+2 merged", lo=0.0, hi=133.0,
         historical_fraction=(26 + 14) / (495 + 532) * 100,
         note="reaches 1 and 2 merged: the Babyne boundary is not placeable"),
    dict(reach_id="3", lo=133.0, hi=183.0, historical_fraction=34 / 363 * 100,
         note="Nikopol to Verkhnia Tarasivka, both placed to ~1 km"),
    dict(reach_id="4", lo=183.0, hi=186.0 * SINUOSITY,
         historical_fraction=180 / 693 * 100,
         note="upper broad lake reach; upper limit is an axis position scaled "
              "by the sinuosity, so it is softer"),
    dict(reach_id="5", lo=186.0 * SINUOSITY, hi=1e9,
         historical_fraction=25 / 72 * 100,
         note="upper channel reach, mostly ABOVE the mapped footprint"),
]
HIST_TOTAL_PCT = 279 / 2155 * 100


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=250.0,
                    help="which hist14 surface resolution to read")
    args = ap.parse_args()
    npz = np.load(CFG.BULK_ROOT / "data_swot/processed/bathymetry" / f"kakhovka_bed_surface_{args.cell:.0f}m.npz",
                  allow_pickle=True)
    gx, gy = npz["gx"], npz["gy"]
    # Refuse a truncated surface rather than publish from it. The bed-surface
    # npz files were written on the P20 footprint (E max 668,540-668,670) while
    # the registry domain reaches 678,000 m; consuming one silently truncated
    # every product below. Regenerate with hist14 after its domain fix.
    _fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                       SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    SD.assert_covers(_fp, gx, gy, what=str("bed surface npz (hist15)"),
                     cell=float(npz["cell_m"]) if "cell_m" in npz else 0.0)
    # Surfaces are stored INSIDE-ONLY as flat float32 (see hist14): every
    # consumer indexed by the mask anyway, and full grids would be 1.86 GB at
    # 30 m. `ins_idx` maps them back onto the raster when a map is needed.
    ins_idx = npz["ins_idx"]
    cell_km2 = (float(npz["cell_m"]) / 1e3) ** 2
    preferred = str(npz["preferred"])
    variants = [str(v) for v in npz["shore_variants"]]
    methods = sorted({k.split("_", 2)[2] for k in npz.files
                      if k.startswith("surf_")})
    print(f"shoreline-constraint variants: {', '.join(variants)}")
    print(f"surface grid {len(gx)}x{len(gy)} at {float(npz['cell_m']):.0f} m, "
          f"{len(ins_idx):,} cells inside ({len(ins_idx)*cell_km2:,.0f} km2)")
    print(f"methods: {', '.join(methods)}   preferred: {preferred}")
    print(f"drying threshold GMO = {GMO_EVRF:.3f} m EVRF2019")

    # ---- chainage per cell -------------------------------------------------
    to_ll = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326",
                                        always_xy=True)
    lon, lat = to_ll.transform(gx[ins_idx % len(gx)], gy[ins_idx // len(gx)])
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    km, _, _, _ = SW.assign_chainage(lon, lat, ch, tree=tree)
    print(f"cell chainage {np.nanmin(km):.0f}..{np.nanmax(km):.0f} km")

    # ---- the fraction, per reach, per method -------------------------------
    rows = []
    for R in REACHES:
        sel = (km >= R["lo"]) & (km < R["hi"]) & np.isfinite(km)
        area = sel.sum() * cell_km2
        fr = {}
        for m in methods:
            v = npz[f"surf_epoch_{m}"][sel]
            # DENOMINATOR = the reservoir at NPG, i.e. cells whose modelled bed
            # lies BELOW the normal impoundment level. The footprint is the
            # 2023-06-05 waterline, ~1 m above NPG, so it encloses a marginal
            # band that is dry land at NPG and must not be counted as "drying".
            # Without this the fraction is inflated: Table 21's denominator is
            # the area AT NPG.
            res = v[np.isfinite(v) & (v < NPG_EVRF)]
            fr[m] = 100 * float((res >= GMO_EVRF).mean()) if len(res) else np.nan
        vals = np.array([fr[m] for m in methods], float)
        rows.append({"reach_id": R["reach_id"], "lo_km": R["lo"],
                     "hi_km": min(R["hi"], np.nanmax(km)),
                     "reach_area_km2": area,
                     "historical_fraction_pct": R["historical_fraction"],
                     "reconstructed_fraction_pct": float(np.nanmedian(vals)),
                     "reconstructed_min_pct": float(np.nanmin(vals)),
                     "reconstructed_max_pct": float(np.nanmax(vals)),
                     "difference_pct": float(np.nanmedian(vals))
                     - R["historical_fraction"],
                     "surface_method": f"median of {len(methods)} interpolators "
                                       f"(preferred {preferred})",
                     "uncertainty_pct": float(np.nanmax(vals) - np.nanmin(vals)),
                     "note": R["note"],
                     **{f"pct_{m}": fr[m] for m in methods}})
    out = pd.DataFrame(rows)

    print("\n" + "=" * 82)
    print("AREA-WEIGHTED EXPOSURE FRACTION vs TABLE 21")
    print("=" * 82)
    print(f"  {'reach':<12}{'km range':>14}{'area':>9}{'historical':>12}"
          f"{'reconstructed':>15}{'spread':>9}{'diff':>8}")
    for r in out.itertuples():
        print(f"  {r.reach_id:<12}{f'{r.lo_km:.0f}-{r.hi_km:.0f}':>14}"
              f"{r.reach_area_km2:>8.0f}k{r.historical_fraction_pct:>11.1f}%"
              f"{r.reconstructed_fraction_pct:>14.1f}%"
              f"{r.uncertainty_pct:>8.1f}%{r.difference_pct:>+8.1f}")

    # total, all cells
    tot = {}
    for m in methods:
        v = npz[f"surf_epoch_{m}"]
        res = v[np.isfinite(v) & (v < NPG_EVRF)]
        tot[m] = 100 * float((res >= GMO_EVRF).mean())
    tv = np.array(list(tot.values()))
    print("\n  sensitivity to the SHORELINE CONSTRAINT, whole reservoir, "
          f"{preferred}:")
    for var in variants:
        v = npz[f"surf_{var}_{preferred}"]
        res = v[np.isfinite(v) & (v < NPG_EVRF)]
        print(f"    constraint '{var:<6}' -> {100*(res >= GMO_EVRF).mean():>5.1f} % "
              f"(historical {HIST_TOTAL_PCT:.1f} %)")
    print("  The unconstrained surface is wrong by a factor of ~2. This is the")
    print("  single largest methodological choice in the whole test.")
    print(f"\n  WHOLE RESERVOIR   historical {HIST_TOTAL_PCT:.1f} %   "
          f"reconstructed {np.median(tv):.1f} % "
          f"(range {tv.min():.1f}-{tv.max():.1f} across methods)")
    print(f"  difference {np.median(tv)-HIST_TOTAL_PCT:+.1f} percentage points")

    # ---- does the upward increase reproduce? -------------------------------
    print("\n" + "=" * 82)
    print("DOES THE HISTORICAL UPWARD INCREASE REPRODUCE? (Task 3)")
    print("=" * 82)
    use = out[out.reach_id != "5"]
    hi_ = use.historical_fraction_pct.values
    re_ = use.reconstructed_fraction_pct.values
    print(f"  reaches used: {', '.join(use.reach_id)}  (reach 5 excluded: mostly")
    print(f"  above the mapped footprint, {out[out.reach_id=='5'].reach_area_km2.iloc[0]:.0f} km2 of grid)")
    print(f"\n  historical    {', '.join(f'{v:.1f}' for v in hi_)} %")
    print(f"  reconstructed {', '.join(f'{v:.1f}' for v in re_)} %")
    mono_h = bool(np.all(np.diff(hi_) > 0))
    mono_r = bool(np.all(np.diff(re_) > 0))
    print(f"\n  monotone increasing upstream?   historical {mono_h}, "
          f"reconstructed {mono_r}")
    per_method = {m: use[f"pct_{m}"].values for m in methods}
    agree = {m: bool(np.all(np.diff(v) > 0)) for m, v in per_method.items()}
    print(f"  per interpolator: "
          + ", ".join(f"{m} {'yes' if a else 'NO'}" for m, a in agree.items()))
    n_ok = sum(agree.values())
    if n_ok == len(methods):
        print(f"\n  -> the increase reproduces across ALL {len(methods)} interpolators, so it")
        print(f"     is a property of the data and not of the interpolation.")
    elif n_ok == 0:
        print(f"\n  -> the increase does NOT reproduce under any interpolator.")
    else:
        print(f"\n  -> DISAGREEMENT: {n_ok} of {len(methods)} interpolators reproduce the")
        print(f"     increase. Reported as such; not resolved by choosing one.")
    if np.isfinite(re_).all():
        rho = float(np.corrcoef(hi_, re_)[0, 1]) if len(hi_) > 2 else np.nan
        print(f"\n  correlation across the {len(hi_)} usable reaches: r = {rho:.3f}")
        print(f"  (three points -- an ordering check, not a fit)")

    out["cell_m"] = float(npz["cell_m"])
    tag = "" if abs(float(npz["cell_m"]) - 250) < 1 else f"_{float(npz['cell_m']):.0f}m"
    out.to_csv(CFG.TABLES / f"historical_exposure_area_validation{tag}.csv",
               index=False)
    print(f"\n-> {CFG.TABLES/('historical_exposure_area_validation'+tag+'.csv')}")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6))
    a = ax[0]
    w = 0.36
    y = np.arange(len(use))
    a.barh(y - w / 2, use.historical_fraction_pct, height=w, color=BLUE,
           alpha=0.85, label="Table 21 (reported)")
    a.barh(y + w / 2, use.reconstructed_fraction_pct, height=w, color=GREEN,
           alpha=0.85, label="reconstructed (median of 4 surfaces)")
    a.errorbar(use.reconstructed_fraction_pct, y + w / 2,
               xerr=[use.reconstructed_fraction_pct - use.reconstructed_min_pct,
                     use.reconstructed_max_pct - use.reconstructed_fraction_pct],
               fmt="none", ecolor=INK, capsize=4, lw=1.4)
    a.set_yticks(y)
    a.set_yticklabels([f"reach {r}" for r in use.reach_id], fontsize=10)
    a.set_xlabel("area drying between NPG and GMO (% of reach area)")
    a.legend(fontsize=8.6)
    a.grid(axis="x", alpha=0.25)
    a.set_title("Area fraction, not point count — the valid replacement for "
                "test E", fontsize=10.8, loc="left")

    a = ax[1]
    for m, c in zip(methods, [BLUE, GREEN, AMBER, RED]):
        a.plot(use.lo_km, use[f"pct_{m}"], "o-", color=c, lw=1.8, ms=7, label=m)
    a.plot(use.lo_km, use.historical_fraction_pct, "s--", color=INK, lw=2.4,
           ms=9, label="Table 21")
    a.set_xlabel("reach start, channel chainage (km)")
    a.set_ylabel("drying area fraction (%)")
    a.legend(fontsize=8.6)
    a.grid(alpha=0.25)
    a.set_title("Does the upstream increase survive the choice of interpolator?",
                fontsize=10.8, loc="left")

    fig.suptitle("V13 · Historical drying-area fractions against an "
                 "area-weighted reconstruction", fontsize=12.5, y=1.01)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V13_exposure_area_validation{tag}.{e}", dpi=185,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/('V13_exposure_area_validation'+tag+'.png')}")


if __name__ == "__main__":
    main()
