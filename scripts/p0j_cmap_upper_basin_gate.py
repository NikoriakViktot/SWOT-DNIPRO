#!/usr/bin/env python
"""Does CMAP2020 bridge the unsounded upper-basin band? Verdict: no.

The remaining hope for CMAP2020 was that, having added no thalweg, it might
constrain the vertical gap the primary survey leaves open: the highest primary
sounding is 15.3153 m against a 17.0836 m shoreline, so the interpolator
crosses 1.77 m of upper slope with no internal support. If CMAP carried
15-16 m isobaths there, the chain would become

    soundings -> CMAP 15-16 m -> observed shoreline at 17.08 m

which would be materially stronger than what exists now.

It does not. CMAP's highest contour is 15.2158 m -- 0.0995 m BELOW where the
primary soundings already stop -- and the 15.32-17.08 m band contains exactly
zero contour length. Its vertical envelope is strictly inside the primary
survey's at BOTH ends (CMAP -15.78..15.22 against primary -19.38..15.32), so
it neither reaches deeper nor shallower.

That answers the gate at step 2, so the later sections (transect ordering,
reach-level support scores, A(H) prospective testing) are not run: they all
presuppose contour length inside the band, and there is none. Reporting a
reach-by-reach breakdown of zero would be ceremony, not evidence.

VERDICT: C — REDUNDANT. No soft-contour experiment is justified.

Combined with the p0h/p0i findings this is consistent rather than surprising:
CMAP shares a source with the primary bathymetry (identical depth values
where co-located, NMAD 0.0000 m at every matching radius), 97.8% of its length
lies within 500 m of a primary sounding, and its effective information is
~4,859 contour components -- fewer than the 7,514 soundings themselves.

Outputs
-------
outputs/bathymetry_provenance/15_cmap_upper_basin_support.csv
outputs/figures/historical_bathymetry/png/p0j_cmap_upper_basin.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
CONTOURS = ROOT / "data/processed/bathymetry/cmap2020_contours_utm.gpkg"
OUT = CFG.OUT / "bathymetry_provenance"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
H_SHORE = 17.0836          # H1 epoch shoreline, independently established
BANDS = [(-99, -10, "< -10 m"), (-10, -5, "-10..-5 m"), (-5, 0, "-5..0 m"),
         (0, 5, "0..5 m"), (5, 10, "5..10 m"), (10, 15, "10..15 m"),
         (15, None, "15..15.32 m"), (None, 16, "15.32..16 m  [GAP]"),
         (16, 99, "> 16 m  [GAP]")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(PRIMARY)
    G = gpd.read_file(CONTOURS, layer="cmap2020_contours")
    h_max = float(P.H_bed_evrf2019_m.max())
    h_min = float(P.H_bed_evrf2019_m.min())
    c_max = float(G.H_bed_EVRF2019_m.max())
    c_min = float(G.H_bed_EVRF2019_m.min())

    print("=" * 78)
    print("VERTICAL ENVELOPES")
    print("=" * 78)
    print(f"  primary soundings : {h_min:8.4f} .. {h_max:8.4f} m EVRF2019")
    print(f"  CMAP2020 contours : {c_min:8.4f} .. {c_max:8.4f} m EVRF2019")
    print(f"  CMAP max is {c_max-h_max:+.4f} m relative to the primary max")
    print(f"  CMAP min is {c_min-h_min:+.4f} m relative to the primary min")
    print(f"\n  TARGET GAP: {h_max:.4f} -> {H_SHORE:.4f} m "
          f"({H_SHORE-h_max:.2f} m of unsupported upper slope)")

    rows, tot = [], G.length_m.sum() / 1000
    for lo, hi, lbl in BANDS:
        lo = h_max if lo is None else lo
        hi = h_max if hi is None else hi
        s = G[(G.H_bed_EVRF2019_m > lo) & (G.H_bed_EVRF2019_m <= hi)]
        km = s.length_m.sum() / 1000
        rows.append(dict(band=lbl, low_m=lo, high_m=hi, length_km=km,
                         pct_of_total=100 * km / tot if tot else 0.0,
                         n_components=len(s),
                         in_unsounded_gap=lo >= h_max))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "15_cmap_upper_basin_support.csv", index=False)
    print("\n" + "=" * 78)
    print("CONTOUR LENGTH BY ELEVATION BAND")
    print("=" * 78)
    print(B.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    gap_km = float(B[B.in_unsounded_gap].length_km.sum())
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  contour length inside the {h_max:.2f}-{H_SHORE:.2f} m band: "
          f"{gap_km:.1f} km")
    if gap_km < 1.0:
        print("  C — REDUNDANT. CMAP2020 stops BELOW where the soundings stop,")
        print("  so it cannot bridge the unsounded upper slope. Its vertical")
        print("  envelope is strictly inside the primary survey's at both ends.")
        print("  No soft-contour experiment is justified; the later sections of")
        print("  the gate presuppose length inside the band and are not run.")
    else:
        print("  Band is populated; proceed to transect ordering diagnostics.")

    # ------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(16.5, 6.2),
                           gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    a.hist(P.H_bed_evrf2019_m, bins=90, color=GREY, alpha=0.75,
           label=f"primary soundings (n={len(P):,})")
    w = np.repeat(G.H_bed_EVRF2019_m.to_numpy(),
                  np.maximum(1, (G.length_m / 100).astype(int)))
    a.hist(w, bins=90, color=BLUE, alpha=0.6,
           label="CMAP contour length (per 100 m)")
    a.axvline(h_max, color=RED, lw=1.8,
              label=f"highest primary sounding {h_max:.2f} m")
    a.axvline(c_max, color=PURPLE, lw=1.8, ls="-.",
              label=f"highest CMAP contour {c_max:.2f} m")
    a.axvline(H_SHORE, color=GREEN, lw=1.8, ls="--",
              label=f"epoch shoreline {H_SHORE:.2f} m")
    a.axvspan(h_max, H_SHORE, color=RED, alpha=0.10)
    a.text((h_max + H_SHORE) / 2, a.get_ylim()[1] * 0.75,
           "unsounded\nband\n(EMPTY)", ha="center", fontsize=9.5, color=RED,
           weight="bold")
    a.set_xlabel("bed elevation (m EVRF2019)"); a.set_ylabel("count")
    a.legend(fontsize=8.2, loc="upper left")
    a.set_title("a · CMAP stops 0.10 m BELOW the highest primary sounding,\n"
                "so the 1.77 m upper slope stays unsupported",
                fontsize=10.4, loc="left")
    a.grid(alpha=0.25)

    a = ax[1]
    hi = G[G.H_bed_EVRF2019_m > 10]
    lo = G[G.H_bed_EVRF2019_m <= 10]
    lo.plot(ax=a, color="#c9d4dc", lw=0.3)
    hi.plot(ax=a, color=BLUE, lw=0.6)
    a.plot(P.x, P.y, ".", ms=0.5, color=RED, alpha=0.5)
    a.set_title("b · blue = CMAP contours above 10 m, red = primary soundings\n"
                "nothing anywhere sits above 15.22 m", fontsize=10.4, loc="left")
    a.set_xlabel("easting (m)"); a.set_ylabel("northing (m)")
    a.grid(alpha=0.25)
    fig.suptitle("p0j · CMAP2020 does not bridge the unsounded upper basin — "
                 "verdict C, REDUNDANT", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "p0j_cmap_upper_basin.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {OUT/'15_cmap_upper_basin_support.csv'}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
