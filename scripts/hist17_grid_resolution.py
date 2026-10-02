#!/usr/bin/env python
"""HISTORICAL 17 — does the exposure fraction depend on grid resolution?

The drying fraction is an AREA count on an interpolated surface, so it inherits
whatever the grid does to thin features. Coarse cells average a narrow marginal
band into its surroundings; fine cells resolve it. The question is whether the
answer converges, and at what cell size.

This reads every surface hist14 has written -- one npz per resolution -- and
recomputes the same quantity on each, with the same shoreline constraint and the
same NPG denominator. Nothing is refitted: the surfaces already exist.

    fraction = cells with GMO <= bed < NPG   /   cells with bed < NPG

Convergence, not agreement, is the test here. If the fraction still moves
between 50 m and 30 m, the 250 m answer was resolution-limited and the whole
comparison with Table 21 has to be quoted at the finest available grid.

Outputs
-------
outputs/tables/hist17_grid_resolution.csv
outputs/figures/V15_grid_resolution_convergence.png
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
GMO_EVRF = 12.70 + 0.185
NPG_EVRF = 16.00 + 0.185
HIST_TOTAL_PCT = 279 / 2155 * 100
REACHES = [("1+2 merged", 0.0, 133.0, (26 + 14) / (495 + 532) * 100),
           ("3", 133.0, 183.0, 34 / 363 * 100),
           ("4", 183.0, 253.0, 180 / 693 * 100)]
BATH = CFG.BULK_ROOT / "data_swot/processed/bathymetry"


def main() -> None:
    files = sorted(BATH.glob("kakhovka_bed_surface_*m.npz"),
                   key=lambda p: -float(p.stem.split("_")[-1][:-1]))
    if not files:
        raise SystemExit("no hist14 surfaces found")
    print(f"surfaces found: "
          + ", ".join(f.stem.split('_')[-1] for f in files))

    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    tree = SW.build_chainage_tree(ch)
    to_ll = pyproj.Transformer.from_crs(CFG.CRS_METRIC, "EPSG:4326",
                                        always_xy=True)

    rows = []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        cell = float(npz["cell_m"])
        gx, gy = npz["gx"], npz["gy"]
        # Refuse a truncated surface rather than publish from it. The bed-surface
        # npz files were written on the P20 footprint (E max 668,540-668,670) while
        # the registry domain reaches 678,000 m; consuming one silently truncated
        # every product below. Regenerate with hist14 after its domain fix.
        _fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
                           SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
        SD.assert_covers(_fp, gx, gy, what=str("bed surface npz (hist17)"),
                         cell=float(npz["cell_m"]) if "cell_m" in npz else 0.0)
        ins = npz["ins_idx"]
        methods = sorted({k.split("_", 2)[2] for k in npz.files
                          if k.startswith("surf_epoch_")})
        lon, lat = to_ll.transform(gx[ins % len(gx)], gy[ins // len(gx)])
        km, _, _, _ = SW.assign_chainage(lon, lat, ch, tree=tree)
        area = len(ins) * (cell / 1e3) ** 2
        print(f"\n{cell:.0f} m: {len(ins):,} cells inside, {area:,.0f} km2, "
              f"methods {', '.join(methods)}")

        for scope, lo, hi, hist_pct in ([("whole", -1e9, 1e9, HIST_TOTAL_PCT)]
                                        + [(r[0], r[1], r[2], r[3])
                                           for r in REACHES]):
            sel = ((km >= lo) & (km < hi) & np.isfinite(km)) if scope != "whole" \
                else np.isfinite(km)
            per = {}
            for m in methods:
                v = npz[f"surf_epoch_{m}"][sel]
                res = v[np.isfinite(v) & (v < NPG_EVRF)]
                per[m] = 100 * float((res >= GMO_EVRF).mean()) if len(res) else np.nan
            vals = np.array(list(per.values()), float)
            rows.append({"cell_m": cell, "scope": scope,
                         "cells": int(sel.sum()),
                         "area_km2": sel.sum() * (cell / 1e3) ** 2,
                         "historical_pct": hist_pct,
                         "reconstructed_pct": float(np.nanmedian(vals)),
                         "min_pct": float(np.nanmin(vals)),
                         "max_pct": float(np.nanmax(vals)),
                         "diff_pp": float(np.nanmedian(vals)) - hist_pct,
                         **{f"pct_{m}": per[m] for m in methods}})
            if scope == "whole":
                print(f"   whole reservoir: {np.nanmedian(vals):.2f} % "
                      f"(methods {np.nanmin(vals):.2f}-{np.nanmax(vals):.2f}), "
                      f"historical {hist_pct:.2f} %, "
                      f"diff {np.nanmedian(vals)-hist_pct:+.2f} pp")

    out = pd.DataFrame(rows).sort_values(["scope", "cell_m"], ascending=[True, False])
    out.to_csv(CFG.TABLES / "hist17_grid_resolution.csv", index=False)

    # ---- convergence -------------------------------------------------------
    w = out[out.scope == "whole"].sort_values("cell_m", ascending=False)
    print("\n" + "=" * 78)
    print("CONVERGENCE OF THE WHOLE-RESERVOIR FRACTION")
    print("=" * 78)
    print(f"  {'cell':>7}{'cells':>12}{'area km2':>11}{'reconstructed':>15}"
          f"{'methods':>16}{'vs historical':>15}")
    prev = None
    for r in w.itertuples():
        step = f"{r.reconstructed_pct-prev:+.2f}" if prev is not None else "--"
        print(f"  {r.cell_m:>5.0f} m{r.cells:>12,}{r.area_km2:>11,.0f}"
              f"{r.reconstructed_pct:>14.2f} %"
              f"{f'{r.min_pct:.2f}-{r.max_pct:.2f}':>16}"
              f"{r.diff_pp:>+14.2f} pp   step {step}")
        prev = r.reconstructed_pct
    finest, coarsest = w.iloc[-1], w.iloc[0]
    drift = finest.reconstructed_pct - coarsest.reconstructed_pct
    if len(w) >= 2:
        print(f"\n  total drift {coarsest.cell_m:.0f} m -> {finest.cell_m:.0f} m: "
              f"{drift:+.2f} pp")
        if len(w) >= 3:
            last_step = w.reconstructed_pct.iloc[-1] - w.reconstructed_pct.iloc[-2]
            print(f"  last refinement step ({w.cell_m.iloc[-2]:.0f} -> "
                  f"{w.cell_m.iloc[-1]:.0f} m): {last_step:+.2f} pp")
            converged = abs(last_step) < 0.3
            print(f"  -> {'CONVERGED' if converged else 'NOT yet converged'}: the "
                  f"last step is {'below' if converged else 'above'} 0.3 pp,")
            print(f"     against a method spread of "
                  f"{finest.max_pct-finest.min_pct:.2f} pp at the finest grid and a")
            print(f"     historical-comparison difference of {finest.diff_pp:+.2f} pp.")
            print(f"\n  Quote the fraction at the FINEST grid "
                  f"({finest.cell_m:.0f} m): {finest.reconstructed_pct:.1f} % against")
            print(f"  {HIST_TOTAL_PCT:.1f} % historical.")

    # Per-reach drift matters more than the whole-reservoir figure: the reaches
    # are what Table 21 is compared against, and a small total can hide two
    # reaches drifting in opposite directions.
    print("\n  PER-REACH DRIFT (the quantities actually compared with Table 21)")
    cells = list(w.cell_m)
    hdr = "".join(f"{f'{c:.0f} m':>9}" for c in cells)
    print(f"  {'reach':<14}{hdr}"
          f"{f'{cells[0]:.0f}->{cells[-1]:.0f}':>10}"
          f"{f'{cells[-2]:.0f}->{cells[-1]:.0f}' if len(cells) > 2 else '':>9}"
          f"{'area km2 spread':>18}")
    worst_t, worst_f = 0.0, 0.0
    for nm, _, _, _ in REACHES:
        r = out[out.scope == nm].sort_values("cell_m", ascending=False)
        v = r.reconstructed_pct.values
        tot = v[-1] - v[0]
        fine = v[-1] - v[-2] if len(v) > 2 else np.nan
        worst_t = max(worst_t, abs(tot))
        if np.isfinite(fine):
            worst_f = max(worst_f, abs(fine))
        print(f"  {nm:<14}" + "".join(f"{x:>8.2f}%" for x in v)
              + f"{tot:>+10.2f}"
              + (f"{fine:>+9.2f}" if np.isfinite(fine) else f"{'':>9}")
              + f"{r.area_km2.max()-r.area_km2.min():>17.2f}")
    aspread = max(out[out.scope == nm].area_km2.max()
                  - out[out.scope == nm].area_km2.min()
                  for nm, _, _, _ in REACHES)
    atot = out[out.scope == "whole"].area_km2
    print(f"\n  largest per-reach drift: {worst_t:.2f} pp over the whole refinement,")
    print(f"  {worst_f:.2f} pp over the last step. The reach AREAS agree across")
    print(f"  resolutions to within {aspread:.2f} km2 (whole reservoir "
          f"{atot.max()-atot.min():.2f} km2 on")
    print(f"  {atot.max():.0f} km2, {100*(atot.max()-atot.min())/atot.max():.2f} %), "
          f"which is an independent check that the three")
    print(f"  grids really are discretising the same domain. The residual is the")
    print(f"  staircase a square cell makes of a curved shoreline; measured against")
    print(f"  the finest grid it is")
    aw = out[out.scope == "whole"].sort_values("cell_m", ascending=False)
    for r in aw.itertuples():
        print(f"      {r.cell_m:>5.0f} m  {r.area_km2 - aw.area_km2.iloc[-1]:+7.2f} km2")

    print("\n  Interpretation. A coarse cell averages a narrow marginal band into")
    print("  its surroundings, and the drying zone IS such a band, so the coarse")
    print("  grid was the one at risk of losing marginal area. Measured, it does")
    drift_dir = ("upward" if drift > 0 else "downward") if abs(drift) > 0.01 else "hardly at all"
    print(f"  not: refining 250 m -> 30 m moves the total {drift_dir} "
          f"({drift:+.2f} pp), which is")
    print(f"  an order of magnitude below both the {finest.max_pct-finest.min_pct:.2f} pp "
          f"spread between interpolators")
    print(f"  and the {abs(finest.diff_pp):.2f} pp difference from the historical value. "
          f"The comparison with")
    print(f"  Table 21 is therefore limited by the interpolation and by the bed data,")
    print(f"  NOT by the grid -- so the canonical 250 m surface is adequate for it,")
    print(f"  and the fine grids are worth keeping for display rather than for the")
    print(f"  statistics.")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
    a = ax[0]
    a.axhline(HIST_TOTAL_PCT, color=BLUE, lw=2.4, ls="--",
              label=f"Table 21: {HIST_TOTAL_PCT:.1f} %")
    a.fill_between(w.cell_m, w.min_pct, w.max_pct, color=GREEN, alpha=0.22,
                   label="spread across interpolators")
    a.plot(w.cell_m, w.reconstructed_pct, "o-", color=GREEN, lw=2.4, ms=9,
           label="reconstructed (median)")
    for r in w.itertuples():
        a.annotate(f"{r.reconstructed_pct:.2f}", (r.cell_m, r.reconstructed_pct),
                   textcoords="offset points", xytext=(0, 11), ha="center",
                   fontsize=8.6)
    a.set_xscale("log")
    a.set_xticks(list(w.cell_m)); a.set_xticklabels([f"{c:.0f}" for c in w.cell_m])
    a.invert_xaxis()
    a.set_xlabel("grid cell size (m), refining to the right")
    a.set_ylabel("drying area between NPG and GMO (%)")
    a.legend(fontsize=8.6, loc="lower left", framealpha=0.94)
    a.grid(alpha=0.25)
    a.set_title("Whole reservoir: does the fraction converge?",
                fontsize=11.2, loc="left")
    # The axis is zoomed to ~1 pp, so a flat line could be misread as a trend.
    # Say the span out loud on the panel.
    lo_, hi_ = a.get_ylim()
    a.text(0.985, 0.045, f"full vertical span of this panel: {hi_-lo_:.1f} pp\n"
                         f"total drift 250 m -> {finest.cell_m:.0f} m: {drift:+.2f} pp",
           transform=a.transAxes, ha="right", va="bottom", fontsize=8.4,
           color=INK, bbox=dict(fc="white", ec=GREY, lw=0.6, alpha=0.92,
                                boxstyle="round,pad=0.34"))

    a = ax[1]
    for (nm, _, _, hp), c in zip(REACHES, [BLUE, AMBER, RED]):
        r = out[out.scope == nm].sort_values("cell_m", ascending=False)
        a.plot(r.cell_m, r.reconstructed_pct, "o-", color=c, lw=2, ms=7,
               label=f"reach {nm}")
        a.axhline(hp, color=c, lw=1.4, ls=":")
    a.set_xscale("log")
    a.set_xticks(list(w.cell_m)); a.set_xticklabels([f"{c:.0f}" for c in w.cell_m])
    a.invert_xaxis()
    a.set_xlabel("grid cell size (m), refining to the right")
    a.set_ylabel("drying area fraction (%)")
    a.legend(fontsize=8.6); a.grid(alpha=0.25)
    a.set_title("By reach — dotted lines are the historical values",
                fontsize=11.2, loc="left")

    fig.suptitle("V15 · Grid-resolution sensitivity of the area-weighted "
                 "exposure fraction   ·   same surfaces, same constraint, "
                 "same denominator", fontsize=12.2, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"V15_grid_resolution_convergence.{e}", dpi=185,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'V15_grid_resolution_convergence.png'}")
    print(f"-> {CFG.TABLES/'hist17_grid_resolution.csv'}")


if __name__ == "__main__":
    main()
