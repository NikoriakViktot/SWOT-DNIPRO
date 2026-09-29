#!/usr/bin/env python
"""P26 -- the spectral atlas: index, class and coverage maps for every zone.

Slots 7-12 of `outputs/planning/07_cartographic_atlas_plan.md` §7.2, drawn from
the p25 composites in `outputs/rasters/zone<N>/`. Every map carries the §7.1
furniture (registry outline, scale bar, north arrow, legend / colour bar, CRS,
acquisition window, provenance), and every multi-panel comparison of one index
uses ONE fixed colour scale (`plotting.maps.INDEX_SCALE`) so PRE and POST are
comparable by eye. No science is computed here.

Per zone:
    zone<N>_<index>_regimes.png       3 panels PRE / BREACH_DRAWDOWN / POST, shared scale
    zone<N>_class_<regime>.png        k10e physical class mode
    zone<N>_water_frac_<regime>.png   % of observed dates with water
    zone<N>_n_valid_<regime>.png      number of dates observed  (the coverage map --
                                      what "not observed" looks like, per pixel)

Usage
-----
python scripts/p26_zone_spectral_maps.py --zone ZONE_2_KHERSON_DELTA
"""
from __future__ import annotations

import argparse
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

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP
from swot_dnipro.plotting import maps as M
from swot_dnipro.plotting import style as ST

ZONES = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": 1, "ZONE_2_KHERSON_DELTA": 2,
         "ZONE_3_DNIPRO_BUG_ESTUARY": 3, "ZONE_4_DAM_TO_KHERSON_FLOODWAY": 4}
REGIMES = ("PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True, choices=list(ZONES))
    a = ap.parse_args()
    zone, n = a.zone, ZONES[a.zone]
    ST.use_style()
    rdir = ROOT / "outputs" / "rasters" / f"zone{n}"
    fdir = ROOT / "outputs" / "figures" / "atlas" / f"zone{n}"
    fdir.mkdir(parents=True, exist_ok=True)
    have = {r: (rdir / f"zone{n}_NDVI_{r}_median_20m.tif").exists() for r in REGIMES}
    regs = [r for r in REGIMES if have[r]]
    print("=" * 78)
    print(f"P26 -- spectral atlas {zone}: regimes with composites: {regs}")
    print("=" * 78)
    if not regs:
        raise SystemExit("no composites -- run p25 first")
    written = []

    w1, h1 = M.fig_size(zone, width_in=4.6)          # one panel, zone aspect
    # ---- one figure per index, regimes side by side, shared scale ------------
    for idx in SP.INDEX_NAMES:
        fig, axes = plt.subplots(1, len(regs), figsize=(w1 * len(regs) + 1.2, h1), squeeze=False)
        last_tags = {}
        for ax, r in zip(axes[0], regs):
            p = rdir / f"zone{n}_{idx}_{r}_median_20m.tif"
            _, tags = M.draw_index(ax, p, idx, zone, add_cbar=(r == regs[-1]))
            M.finish(ax, f"{idx} median · {r} · n={tags.get('n_dates','?')}", M.caption_from_tags(tags))
            last_tags = tags
        fig.suptitle(f"{zone} — {idx} (Sentinel-2 L2A, 20 m, BOA offset applied, fixed scale)", fontsize=10)
        fig.tight_layout()
        written += ST.save(fig, f"zone{n}_{idx}_regimes", fdir)
        plt.close(fig)
        print(f"  {idx}: {len(regs)} panels")

    # ---- class mode, water frequency, coverage -- one per regime -------------
    wc, hc = M.fig_size(zone, width_in=6.0, extra_w=3.2)   # room for the outside legend
    ws, hs = M.fig_size(zone, width_in=6.0, extra_w=1.2)   # room for a colour bar
    for r in regs:
        fig, ax = plt.subplots(figsize=(wc, hc))
        tags = M.draw_class(ax, rdir / f"zone{n}_class_{r}_mode_20m.tif", zone)
        M.finish(ax, f"physical surface class (mode over dates) · {r}",
                 M.caption_from_tags(tags, "k10e scheme; thresholds on offset-corrected reflectance; classes not yet validated against labelled samples"))
        fig.tight_layout(); written += ST.save(fig, f"zone{n}_class_{r}", fdir); plt.close(fig)

        fig, ax = plt.subplots(figsize=(ws, hs))
        tags = M.draw_uint8(ax, rdir / f"zone{n}_water_frac_{r}_20m.tif", zone, "% of observed dates with water", cmap="Blues", vmax=100)
        M.finish(ax, f"water frequency · {r}", M.caption_from_tags(tags, "255 = never observed, shown blank"))
        fig.tight_layout(); written += ST.save(fig, f"zone{n}_water_frac_{r}", fdir); plt.close(fig)

        fig, ax = plt.subplots(figsize=(ws, hs))
        tags = M.draw_uint8(ax, rdir / f"zone{n}_n_valid_{r}_20m.tif", zone, "dates observed", cmap="magma")
        M.finish(ax, f"observation count · {r}", M.caption_from_tags(tags, "not observed is not dry: this is the denominator of every map above"))
        fig.tight_layout(); written += ST.save(fig, f"zone{n}_n_valid_{r}", fdir); plt.close(fig)
        print(f"  {r}: class / water_frac / n_valid")

    print(f"\n  {len(written)} files -> {fdir}")


if __name__ == "__main__":
    main()
