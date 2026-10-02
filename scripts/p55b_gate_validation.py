#!/usr/bin/env python
"""P55b -- controlled-change validation for the CORRECTED BOTTOM GATE in p55_seamless_dem.py.

The change. p55 used to decide "may a bathymetric surface be written in this cell?" from FABDEM's absolute height alone
(`fab <= 17 m`). That rule fixed the 59 km2 of artificial holes at the pool rim, but it also carried a physically understood
failure: FABDEM has its own metre-level errors, so wherever FABDEM overestimates on the drained bed the rule silently threw
away good bathymetry. Measured on p57's night sample: 25 of 11,198 dry-bed points, where FABDEM sits 4.8 m above the ICESat-2
ground, and all 25 are classified bed/water by the corroborated p64 mask. The patch therefore replaces the height arbiter with
the SPATIAL class (p64: never observed as water in 2019-2022, with optical support, and above the water surface in FABDEM).
FABDEM's height stays as one corroborating criterion inside that mask and as a QC bound on the gap-fill shoreline value, but it
is no longer the sole arbiter of the semantic class of a surface (user, 2026-09-20).

This script re-runs the SAME battery used for the first fix, over three versions of the ZONE_1 rasters:
  ORIGINAL   before any fix          (scratchpad/before_p55fix)
  HEIGHT     height-gate fix         (scratchpad/after_height_gate)
  CORRECTED  this patch              (the live $BULK_ROOT/dem_seamless)
  A  frame identity and the six bitwise invariants, CORRECTED vs ORIGINAL and CORRECTED vs HEIGHT
  B  the 25 displaced dry-bed points: residual under each version, the explicit before/after evidence that the change is
     physically motivated and not metric tuning
  C  new-tail check: over every cell whose class differs between HEIGHT and CORRECTED, the night-ICESat-2 residual distribution,
     so that returning bathymetry cannot hide a new below-ground tail somewhere else
Outputs: outputs/tables/p55b_gate_invariants.csv, p55b_displaced_points.csv, p55b_gate_deltas.csv
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import rasterio

from swot_dnipro import config as CFG

spec = importlib.util.spec_from_file_location("p63", ROOT / "scripts/p63_dem_accuracy_by_source.py")
P63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P63)
P57 = P63.P57
SCRATCH = Path("/tmp/claude-1000/-home-niko-repo-SWOT-DNIPRO/9177877c-b212-4adf-b40a-33f677174025/scratchpad")
ZONE1 = "ZONE_1_KAKHOVKA_LOWER_DNIPRO"
VERSIONS = {"ORIGINAL": SCRATCH / "before_p55fix", "HEIGHT": SCRATCH / "after_height_gate", "CORRECTED": CFG.BULK_ROOT / "dem_seamless"}
MAX_STEP_M = 1.0
PX = 0.0004   # 20 m cell, km2


def read(dirpath, what):
    p = Path(dirpath) / f"{ZONE1}_dem_{what}_20m.tif"
    with rasterio.open(p) as ds:
        return ds.read(1), ds.transform, ds.shape, ds.crs


def invariants(tag, A, a, B, b, fab, bathy_may_grow=False):
    """The six proof-level checks; A/a = reference version source+dem, B/b = new version.

    `bathy_may_grow` is set for the CORRECTED-vs-HEIGHT comparison only: this patch deliberately RETURNS bathymetry that the
    height gate had removed, so the monotonicity that held for the first fix is expected to reverse there and is reported as
    the intended effect, with its area, rather than as a failure.
    """
    out = []
    def add(name, ok, detail="", expected=False):
        out.append(dict(comparison=tag, invariant=name, result=("EXPECTED" if expected else ("PASS" if ok else "FAIL")), detail=detail))
    add("{src==0} unchanged (no new empty cells)", bool(((A == 0) == (B == 0)).all()),
        f"{int(((A == 0) != (B == 0)).sum())} cells differ")
    bo, bn = np.isin(A, [1, 2, 5]), np.isin(B, [1, 2, 5])
    add("bathymetric classes {1,2,5} only shrink" + (" (patch returns some)" if bathy_may_grow else ""),
        bool((bn & ~bo).sum() == 0), f"{(bn & ~bo).sum() * PX:.2f} km2 returned to bathymetry", expected=bathy_may_grow)
    fo, fn = np.isin(A, [3, 4]), np.isin(B, [3, 4])
    add("FABDEM-derived classes {3,4} only grow" + (" (patch returns some)" if bathy_may_grow else ""),
        bool((fo & ~fn).sum() == 0), f"{(fo & ~fn).sum() * PX:.2f} km2 handed back to bathymetry", expected=bathy_may_grow)
    m3 = (A == 3) & (B == 3); add("src==3 bitwise identical", bool(np.array_equal(a[m3], b[m3])), f"n {int(m3.sum()):,}")
    m1 = (A == 1) & (B == 1); add("src==1 bitwise identical (gate never leaks below the dam)", bool(np.array_equal(a[m1], b[m1])), f"n {int(m1.sum()):,}")
    m4 = B == 4; dev = np.abs(b[m4] - fab[m4]); dev = dev[np.isfinite(dev)]
    mx = float(dev.max()) if len(dev) else 0.0
    add(f"|dem - FABDEM| <= {MAX_STEP_M} m wherever src==4", bool(mx <= MAX_STEP_M + 1e-4), f"max {mx:.4f} m over n {int(m4.sum()):,}")
    return out


def main():
    ref, tr, shape, crs = read(VERSIONS["CORRECTED"], "source")
    fab = np.full(shape, np.nan, "f4")
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    with rasterio.open(CFG.BULK_ROOT / "terrain" / ZONE1 / "fabdem_evrf2019_20m.tif") as f:
        reproject(source=rasterio.band(f, 1), destination=fab, dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.bilinear, src_nodata=f.nodata, dst_nodata=np.nan)
    S, D = {}, {}
    for k, d in VERSIONS.items():
        S[k], t, sh, c = read(d, "source")
        D[k] = read(d, "evrf2019")[0].astype("f8")
        assert (t, sh, c) == (tr, shape, crs), f"{k}: frame differs -- p58 depends on the p55 frame"
    print("FRAME identical across all three versions: True")
    # ---------------- A. invariants
    rows = invariants("CORRECTED vs ORIGINAL", S["ORIGINAL"], D["ORIGINAL"], S["CORRECTED"], D["CORRECTED"], fab)
    rows += invariants("CORRECTED vs HEIGHT", S["HEIGHT"], D["HEIGHT"], S["CORRECTED"], D["CORRECTED"], fab, bathy_may_grow=True)
    I = pd.DataFrame(rows); I.to_csv(CFG.TABLES / "p55b_gate_invariants.csv", index=False)
    print(I.to_string(index=False))
    print("\nsource-class areas, km2:")
    A = pd.DataFrame({k: {c: round(float((S[k] == c).sum()) * PX, 2) for c in range(6)} for k in VERSIONS})
    print(A.to_string())
    # ---------------- B/C. night ICESat-2
    V = P63.build_set_C(); B = V[V.zone == ZONE1].copy()
    for k, d in VERSIONS.items():
        B[f"src_{k}"] = P57.sample(Path(d) / f"{ZONE1}_dem_source_20m.tif", B.x.values, B.y.values)
        B[f"dem_{k}"] = P57.sample(Path(d) / f"{ZONE1}_dem_evrf2019_20m.tif", B.x.values, B.y.values)
        B[f"res_{k}"] = B[f"dem_{k}"] - B.H_ice
    B["island"] = P57.sample(ROOT / "outputs/rasters/zone1/zone1_pool_islands_30m.tif", B.x.values, B.y.values)
    disp = B[B.src_ORIGINAL != B.src_HEIGHT].copy()
    print(f"\nB. dry-bed points the HEIGHT gate displaced: {len(disp)} of {len(B):,}")
    if len(disp):
        T = pd.DataFrame([dict(version=k, n=len(disp), median_abs_res=round(float(disp[f"res_{k}"].abs().median()), 3),
                               RMSE=round(float(np.sqrt((disp[f"res_{k}"] ** 2).mean())), 3),
                               bias=round(float(disp[f"res_{k}"].mean()), 3),
                               n_src2=int((disp[f"src_{k}"] == 2).sum()), n_src3_or_4=int(disp[f"src_{k}"].isin([3, 4]).sum()))
                          for k in VERSIONS])
        T["FABDEM_minus_ICESat2_median"] = round(float((disp.fab - disp.H_ice).median()), 3)
        T["p64_class_0_bed_water"] = int((disp.island == 0).sum()); T["p64_class_1_land"] = int((disp.island == 1).sum())
        T["p64_class_2_unknown"] = int((disp.island == 2).sum())
        T.to_csv(CFG.TABLES / "p55b_displaced_points.csv", index=False)
        print(T.to_string(index=False))
    # C. every point whose class moved between HEIGHT and CORRECTED -> did a new tail appear anywhere?
    ch = B[B.src_HEIGHT != B.src_CORRECTED].copy()
    print(f"\nC. dry-bed points whose class the PATCH changed: {len(ch)}")
    rows = []
    for name, g in (("points the patch moved", ch), ("all dry-bed points", B)):
        for k in VERSIONS:
            r = g[f"res_{k}"].dropna()
            if len(r):
                rows.append(dict(subset=name, version=k, n=len(r), bias=round(float(r.mean()), 3), median=round(float(r.median()), 3),
                                 RMSE=round(float(np.sqrt((r ** 2).mean())), 3), LE90=round(float(np.percentile(np.abs(r), 90)), 3),
                                 worst=round(float(r.min()), 2), share_below_5m=round(float((r < -5).mean()), 5)))
    C = pd.DataFrame(rows); C.to_csv(CFG.TABLES / "p55b_gate_deltas.csv", index=False)
    print(C.to_string(index=False))
    print("\n-> outputs/tables/p55b_{gate_invariants,displaced_points,gate_deltas}.csv")


if __name__ == "__main__":
    main()
