#!/usr/bin/env python
"""P42b -- why the Sentinel-1 flood envelope was wrong: one June-2023 event, raw M3 vs vetoed vs optical.

Panels for ZONE_4 (frame decimated): (a) raw M3 water of the peak event 2023-06-09 (p0v cache, before any veto);
(b) chronic radar-dark land (S1 P_water >= 0.5 while S2 pre-breach water share < 10 %) and HAND >= 5 m veto;
(c) cleaned event water (p42 v7); (d) Sentinel-2 optical water 2023-06-08 outside pre-breach water; (e) final
June-2023 envelope (S1 >= 2 events U S2). Numbers in the title: km2 per layer inside the zone. The figure documents
the M3 anchor-design defect (audit P31/P38: radar-dark bare sand / smooth fields inside the water class) and the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from swot_dnipro import config as CFG
from swot_dnipro import sentinel_preprocess as SP

ZONE = "ZONE_4_DAM_TO_KHERSON_FLOODWAY"
FP = CFG.BULK_ROOT / "floodplain" / ZONE
EVENT = "2023-06-09_orb14_ASC"


def raw_event(G):
    z = np.load(CFG.S1_CACHE / "ZONE_4_FLOODWAY_june2023_s20" / "per_scene_water.npz", allow_pickle=True)
    shp = tuple(int(v) for v in z["shape"]); tr = from_origin(float(z["x0"]), float(z["y1"]), float(z["cell"]), float(z["cell"]))
    w = np.unpackbits(z[EVENT], count=shp[0] * shp[1]).astype("u1").reshape(shp)
    v = np.unpackbits(z["valid_" + EVENT], count=shp[0] * shp[1]).astype(bool).reshape(shp) if "valid_" + EVENT in z else np.ones(shp, bool)
    src = np.where(v, w, 255).astype("u1"); dst = np.full((G["ny"], G["nx"]), 255, "u1")
    reproject(source=src, destination=dst, src_transform=tr, src_crs=CFG.CRS_METRIC, dst_transform=G["transform"], dst_crs=CFG.CRS_METRIC, resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
    return dst


def rd(name):
    with rasterio.open(FP / f"{ZONE}_{name}.tif") as ds:
        a = ds.read(1); return a, [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]


def main():
    G = SP.zone_grid(ZONE, 20.0); inside = G["inside"]
    raw = raw_event(G)
    chronic, ext = rd("chronic_radar_dark_land"); hand, _ = rd("hand_m"); clean, _ = rd(f"flood_extent_{EVENT}"); opt, _ = rd("flood_optical_june2023"); env, _ = rd("flood_envelope_clean")
    with rasterio.open(CFG.BULK_ROOT / "zone_spectral" / ZONE / "2023-06-08_water3.tif") as ds:
        s2 = ds.read(1)
    hv = ~(np.isfinite(hand) & (hand != -9999) & (hand < 5))
    layers = [("(a) raw M3 water, S1 2023-06-09 (peak)", (raw == 1) & inside, "Reds"),
              ("(b) DIAGNOSTIC (not applied): P_water>=0.5 & PRE water<10 % = real June flood, NOT dark land (yellow); HAND-limit veto (grey)", None, None),
              ("(c) cleaned S1 water 2023-06-09 (HAND<h0(x), not pre-breach water, not Inhulets)", (clean == 1) & inside, "Reds"),
              ("(d) Sentinel-2 water 2023-06-08 (frozen NDWI/MNDWI rule)", (s2 == 1) & inside, "Blues"),
              ("(e) final June-2023 flood: S1 >= 2 events U S2, outside pre-breach water", (env == 1) & inside, "Purples"),
              ("(f) S2 optical flood outside pre-breach water", (opt == 1) & inside, "Blues")]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for ax, (title, m, cmap) in zip(axes.ravel(), layers):
        ax.imshow(np.where(inside[::4, ::4], 0.12, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        if m is None:
            ax.imshow(np.where((chronic == 1)[::4, ::4], 1, np.nan), extent=ext, cmap="spring", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(np.where((hv & inside)[::4, ::4], 0.6, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1, alpha=0.5, interpolation="nearest")
            ax.set_title(f"{title}: chronic {((chronic == 1) & inside).sum()*4e-4:.0f} km2")
        else:
            ax.imshow(np.where(m[::4, ::4], 1, np.nan), extent=ext, cmap=cmap, vmin=0, vmax=1.3, interpolation="nearest")
            ax.set_title(f"{title}: {m.sum()*4e-4:.0f} km2")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Why the S1 M3 envelope over-detects: radar-dark sand/fields sit inside M3's water class (P31/P38); the fix is terrain + exclusion + persistence + optics (p42 v7)", fontsize=12)
    fig.tight_layout(); out = ROOT / "outputs/figures/p42b_s1_flood_diagnostic_ZONE_4.png"; fig.savefig(out, dpi=100); print("->", out)


if __name__ == "__main__":
    main()
