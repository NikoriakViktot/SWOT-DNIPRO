#!/usr/bin/env python
"""Provenance gate for the newly found bathymetry layers.

Two datasets turned up on drive D that dwarf the 7,514 historical soundings
currently driving the DEM: reliefUTM36N (994,609 points) and
KahovkaRes_CMAP2020_UTM36 (95,132 points). Adding them would be a mistake
until their origin is established, because if either is a raster or an
interpolation of the SAME soundings, feeding it to kriging hands one old
surface a million votes and lets the DEM be validated by its own derivative.

NOTHING IS ADDED TO THE DEM HERE. This returns numerical evidence only.

The decisive tests:

  GATE B  does reliefUTM36N sit on a regular lattice? A raster-to-point
          conversion betrays itself by exact modular spacing, not by looking
          dense.
  GATE E  how closely does each layer reproduce the 7,514 soundings? Very
          small RMSE combined with a lattice means DERIVED, not independent.
  GATE G  sign convention is TESTED, never inferred from a field name.
          For a depth referenced to a ~16 m pool, both H = ref + d and
          H = ref - d are evaluated against the primary soundings and the
          winner is chosen by agreement, not by appearance.

Outputs land in outputs/bathymetry_provenance/.
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
from scipy.spatial import cKDTree

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY, PURPLE = (
    "#1a2228", "#236f8c", "#c1402a", "#b07d27", "#3f7d4e", "#8a94a3", "#7a4f9e")
D = Path("/mnt/d/data_RAS_Dnipro/GIS_dnipro")
RELIEF = D / "reliefUTM36N/reliefUTM36N.shp"
CMAP = D / "KahovkaRes_CMAP2020_UTM36/KahovkaRes_CMAP2020_UTM36.shp"
PRIMARY = CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet"
OUT = CFG.OUT / "bathymetry_provenance"
FIGDIR = CFG.FIG / "historical_bathymetry" / "png"
CAND_CELLS = (5.0, 10.0, 20.0, 25.0, 30.0, 50.0, 100.0)
# Reference level for the depth layers, established by test rather than
# assumed: 14.00 m BS-77 ("UNS, navigation drawdown level"), the same datum
# the primary soundings carry, converted per point with the EPSG:9902 grid.
# With this reference CMAP2020 matches the primary soundings at median
# residual 0.0000 m and NMAD 0.0000 m; a 16 m reference leaves a +1.81 m bias.
REF_BS77 = 14.0
DIST_CLASSES = (0.01, 1, 5, 10, 25, 50, 100, 250, 500)


def robust(d):
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0)
    med = float(np.median(d))
    return dict(n=int(d.size), bias=float(np.mean(d)), median=med,
                mae=float(np.mean(np.abs(d))),
                rmse=float(np.sqrt(np.mean(d ** 2))),
                nmad=float(1.4826 * np.median(np.abs(d - med))),
                p90=float(np.percentile(np.abs(d), 90)),
                p95=float(np.percentile(np.abs(d), 95)))


def lattice_test(x, y, name):
    """A raster converted to points lies on a lattice with an ARBITRARY
    origin, so testing x mod cell against zero fails even for a perfect grid
    (it reported 0% for all three sources). Test spacing instead: subtract the
    modal offset first, then ask how many points fall on the lattice."""
    rows = []
    for c in CAND_CELLS:
        fx = (x / c) - np.floor(x / c)
        fy = (y / c) - np.floor(y / c)
        ox = float(np.median(fx))          # the grid's own origin offset
        oy = float(np.median(fy))
        dx = np.minimum(np.abs(fx - ox), 1 - np.abs(fx - ox))
        dy = np.minimum(np.abs(fy - oy), 1 - np.abs(fy - oy))
        on = (dx < 0.02) & (dy < 0.02)
        rows.append(dict(dataset=name, cell_m=c,
                         frac_on_lattice=float(on.mean()),
                         frac_x_on=float((dx < 0.02).mean()),
                         frac_y_on=float((dy < 0.02).mean()),
                         origin_offset_x=ox * c, origin_offset_y=oy * c))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- load
    P = pd.read_parquet(PRIMARY)
    px, py, pz = (P.x.to_numpy(), P.y.to_numpy(),
                  P.H_bed_evrf2019_m.to_numpy())
    ptree = cKDTree(np.c_[px, py])
    print(f"PRIMARY soundings: {len(P):,}  z {pz.min():.2f}..{pz.max():.2f} m "
          f"EVRF2019")

    R = gpd.read_file(RELIEF)
    rx = R.geometry.x.to_numpy(); ry = R.geometry.y.to_numpy()
    print(f"reliefUTM36N: {len(R):,} points, crs {R.crs}")
    C = gpd.read_file(CMAP)
    cx = C.geometry.x.to_numpy(); cy = C.geometry.y.to_numpy()
    cz_raw = C["field_3"].astype(float).to_numpy()
    print(f"CMAP2020: {len(C):,} points, crs {C.crs}, "
          f"value {cz_raw.min():.2f}..{cz_raw.max():.2f}")

    inv = []
    for nm, gx, gy, n in (("primary_soundings", px, py, len(P)),
                          ("reliefUTM36N", rx, ry, len(R)),
                          ("CMAP2020", cx, cy, len(C))):
        t = cKDTree(np.c_[gx, gy])
        d, _ = t.query(np.c_[gx, gy], k=2)
        nn = d[:, 1]
        inv.append(dict(dataset=nm, n_points=n,
                        east_min=gx.min(), east_max=gx.max(),
                        north_min=gy.min(), north_max=gy.max(),
                        nn_median_m=float(np.median(nn)),
                        nn_p05_m=float(np.percentile(nn, 5)),
                        nn_p95_m=float(np.percentile(nn, 95)),
                        nn_modal_m=float(np.round(np.median(nn), 2)),
                        dup_coords=int(n - len(np.unique(
                            np.c_[gx, gy], axis=0)))))
    I = pd.DataFrame(inv)
    I.to_csv(OUT / "01_source_inventory.csv", index=False)
    print("\n" + "=" * 78)
    print("GATE A — INVENTORY")
    print("=" * 78)
    print(I.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ------------------------------------------- GATE A: field relationships
    print("\n" + "=" * 78)
    print("GATE A — reliefUTM36N FIELD RELATIONSHIPS (signs are tested)")
    print("=" * 78)
    H = R["Height"].astype(float).to_numpy()
    Dp = R["Depth"].astype(float).to_numpy()
    Rl = R["Relief"].astype(float).to_numpy()
    rel_rows = []
    for lbl, resid in (("Relief = Height + Depth", Rl - (H + Dp)),
                       ("Relief = Height - Depth", Rl - (H - Dp)),
                       ("Depth  = Relief - Height", Dp - (Rl - H)),
                       ("Depth  = Height - Relief", Dp - (H - Rl))):
        a = np.abs(resid)
        rel_rows.append(dict(equation=lbl, median_resid=float(np.median(resid)),
                             nmad=float(1.4826 * np.median(
                                 np.abs(resid - np.median(resid)))),
                             max_abs=float(a.max()),
                             frac_1e6=float((a < 1e-6).mean()),
                             frac_1mm=float((a < 1e-3).mean()),
                             frac_1cm=float((a < 1e-2).mean())))
        print(f"  {lbl:<26} median {np.median(resid):+8.4f}  max|r| "
              f"{a.max():8.3f}  exact {100*(a<1e-6).mean():5.1f}%  "
              f"<1cm {100*(a<1e-2).mean():5.1f}%")
    pd.DataFrame(rel_rows).to_csv(OUT / "02_relief_field_relationships.csv",
                                  index=False)
    print(f"\n  Height: {100*float((H == 16.0).mean()):.1f}% exactly 16.00 m, "
          f"{100*float((H == REF_BS77).mean()):.1f}% exactly {REF_BS77:.2f} m; "
          f"unique values {len(np.unique(H)):,}")
    print(f"  Depth : {100*float((Dp == 0).mean()):.1f}% exactly 0; "
          f"unique {len(np.unique(Dp)):,}")

    # ------------------------------------------------- GATE B: lattice
    print("\n" + "=" * 78)
    print("GATE B/C — LATTICE STRUCTURE (a raster-to-point conversion shows")
    print("           exact modular spacing)")
    print("=" * 78)
    gr = pd.DataFrame(lattice_test(rx, ry, "reliefUTM36N")
                      + lattice_test(cx, cy, "CMAP2020")
                      + lattice_test(px, py, "primary_soundings"))
    gr.to_csv(OUT / "03_grid_structure_tests.csv", index=False)
    for ds in ("reliefUTM36N", "CMAP2020", "primary_soundings"):
        s = gr[gr.dataset == ds].sort_values("frac_on_lattice", ascending=False)
        best = s.iloc[0]
        print(f"  {ds:<20} best cell {best.cell_m:6.1f} m -> "
              f"{100*best.frac_on_lattice:5.1f}% of points on the lattice")

    # ------------------------------------------------- GATE C: CMAP structure
    ucz, cnt = np.unique(cz_raw, return_counts=True)
    dec = np.array([len(str(v).split(".")[-1]) if "." in str(v) else 0
                    for v in ucz[:5000]])
    cm_rows = [dict(metric="n_points", value=len(C)),
               dict(metric="n_unique_depths", value=int(len(ucz))),
               dict(metric="max_repeat_count", value=int(cnt.max())),
               dict(metric="frac_in_top20_values",
                    value=float(np.sort(cnt)[::-1][:20].sum() / len(C))),
               dict(metric="median_decimal_places", value=float(np.median(dec)))]
    pd.DataFrame(cm_rows).to_csv(OUT / "04_cmap_structure_tests.csv", index=False)
    print(f"\n  CMAP2020: {len(ucz):,} unique depths, most repeated value "
          f"occurs {cnt.max():,} times, top-20 values cover "
          f"{100*np.sort(cnt)[::-1][:20].sum()/len(C):.1f}% of points")

    # ------------------------------------------------- GATE D/E: vs primary
    print("\n" + "=" * 78)
    print("GATE D/E — AGREEMENT WITH THE 7,514 PRIMARY SOUNDINGS")
    print("=" * 78)
    rtree = cKDTree(np.c_[rx, ry])
    ctree = cKDTree(np.c_[cx, cy])
    dr, ir = rtree.query(np.c_[px, py], k=1)
    dc, ic = ctree.query(np.c_[px, py], k=1)
    nn_rows = []
    for nm, dd in (("reliefUTM36N", dr), ("CMAP2020", dc)):
        row = dict(dataset=nm, median_dist_m=float(np.median(dd)))
        for t in DIST_CLASSES:
            row[f"frac_within_{t}m"] = float((dd <= t).mean())
        nn_rows.append(row)
        print(f"  primary -> {nm}: median {np.median(dd):7.1f} m; "
              f"within 1 m {100*(dd<=1).mean():5.1f}%, "
              f"10 m {100*(dd<=10).mean():5.1f}%, "
              f"50 m {100*(dd<=50).mean():5.1f}%")
    pd.DataFrame(nn_rows).to_csv(
        OUT / "05_primary_nearest_neighbor_comparison.csv", index=False)

    # ------------------------------------------------- GATE G: datum
    print("\n" + "=" * 78)
    print("GATE G — VERTICAL DATUM HYPOTHESES (chosen by agreement, not name)")
    print("=" * 78)
    d9902 = P.delta_epsg9902_m.to_numpy()
    near = dr <= 100.0
    nearc = dc <= 100.0
    hyp = []
    for nm, mask, idx, z_candidates in (
            ("reliefUTM36N", near, ir,
             {"Relief as-is": Rl, "Height+Depth": H + Dp,
              "16.0+Depth": 16.0 + Dp,
              "14.0+Depth (BS77)": REF_BS77 + Dp,
              "14.0+d9902+Depth (EVRF)": REF_BS77 + Dp}),
            ("CMAP2020", nearc, ic,
             {"value as-is": cz_raw, "16.0+value": 16.0 + cz_raw,
              "14.0+value (BS77)": REF_BS77 + cz_raw,
              "14.0+d9902+value (EVRF)": REF_BS77 + cz_raw,
              "14.0-value": REF_BS77 - cz_raw})):
        for lbl, zc in z_candidates.items():
            off = d9902[mask] if "d9902" in lbl else 0.0
            d = zc[idx[mask]] + off - pz[mask]
            st = robust(d)
            hyp.append(dict(dataset=nm, hypothesis=lbl, **st))
            print(f"  {nm:<14} {lbl:<16} n={st['n']:6,}  bias {st['bias']:+7.2f}  "
                  f"RMSE {st['rmse']:7.2f}  NMAD {st['nmad']:6.2f} m")
    Hy = pd.DataFrame(hyp)
    Hy.to_csv(OUT / "08_vertical_datum_hypotheses.csv", index=False)
    best_r = Hy[Hy.dataset == "reliefUTM36N"].nsmallest(1, "rmse").iloc[0]
    best_c = Hy[Hy.dataset == "CMAP2020"].nsmallest(1, "rmse").iloc[0]
    print(f"\n  BEST reliefUTM36N: {best_r.hypothesis} "
          f"(RMSE {best_r.rmse:.3f} m, bias {best_r.bias:+.3f})")
    print(f"  BEST CMAP2020    : {best_c.hypothesis} "
          f"(RMSE {best_c.rmse:.3f} m, bias {best_c.bias:+.3f})")
    md = float(np.median(d9902))
    zr_best = {"Relief as-is": Rl, "Height+Depth": H + Dp,
               "16.0+Depth": 16.0 + Dp, "14.0+Depth (BS77)": REF_BS77 + Dp,
               "14.0+d9902+Depth (EVRF)": REF_BS77 + md + Dp}[best_r.hypothesis]
    zc_best = {"value as-is": cz_raw, "16.0+value": 16.0 + cz_raw,
               "14.0+value (BS77)": REF_BS77 + cz_raw,
               "14.0+d9902+value (EVRF)": REF_BS77 + md + cz_raw,
               "14.0-value": REF_BS77 - cz_raw}[best_c.hypothesis]
    pd.DataFrame([dict(dataset="reliefUTM36N", **robust(zr_best[ir[near]] - pz[near])),
                  dict(dataset="CMAP2020", **robust(zc_best[ic[nearc]] - pz[nearc]))]
                 ).assign(hypothesis=[best_r.hypothesis, best_c.hypothesis]
                          ).to_csv(OUT / "06_primary_vertical_comparison.csv",
                                   index=False)

    # ------------------------------------------------- GATE J: classification
    lat_r = gr[gr.dataset == "reliefUTM36N"].frac_on_lattice.max()
    lat_c = gr[gr.dataset == "CMAP2020"].frac_on_lattice.max()
    def classify(nn_cv, rmse, nmad, nm):
        """Regularity is judged by the SPREAD of nearest-neighbour spacing,
        not by modular arithmetic: a raster exported to points keeps constant
        spacing but has an arbitrary, possibly rotated origin, so the modular
        test reports ~0% even for a perfect grid."""
        if nn_cv < 0.05:
            return ("GRID_DERIVED_FROM_BATHYMETRY", "MODEL_COMPARISON_ONLY",
                    f"nearest-neighbour spacing is constant (CV {nn_cv:.4f}): "
                    "a raster exported to points. Its cells are not "
                    "independent observations and must never be kriged.")
        if nmad < 0.05 and rmse < 0.5:
            return ("DIGITISED_NAVIGATION_SOUNDINGS", "HARD_OBSERVATION",
                    f"irregular spacing (CV {nn_cv:.2f}) and reproduces the "
                    "primary soundings exactly where they coincide "
                    f"(NMAD {nmad:.3f} m): the same survey, sampled more "
                    "densely -- not a derivative.")
        if rmse < 0.5:
            return ("UNKNOWN_PROVENANCE", "DO_NOT_USE_UNTIL_PROVENANCE_RESOLVED",
                    "irregular but reproduces the primary set too closely to "
                    "be independent")
        return ("UNKNOWN_PROVENANCE", "DO_NOT_USE_UNTIL_PROVENANCE_RESOLVED",
                "irregular and materially different from the primary set")
    _cv0 = {r.dataset: (r.nn_p95_m - r.nn_p05_m) / max(r.nn_median_m, 1e-9)
            for r in I.itertuples()}
    dec_rows = [dict(source="primary_soundings_7514", n_points=len(P),
                     structure="irregular",
                     nn_spacing_cv=float(_cv0["primary_soundings"]),
                     nn_median_m=float(I[I.dataset=="primary_soundings"].nn_median_m.iloc[0]),
                     rmse_vs_primary=0.0, nmad_vs_primary=0.0,
                     provenance="RAW_PRIMARY", allowed_role="HARD_OBSERVATION",
                     note="the reference set")]
    cv = {r.dataset: (r.nn_p95_m - r.nn_p05_m) / max(r.nn_median_m, 1e-9)
          for r in I.itertuples()}
    for nm, n, rm, nmd in (("reliefUTM36N", len(R), best_r.rmse, best_r.nmad),
                           ("CMAP2020", len(C), best_c.rmse, best_c.nmad)):
        prov, role, note = classify(cv[nm], rm, nmd, nm)
        dec_rows.append(dict(source=nm, n_points=n,
                             structure=("regular grid" if cv[nm] < 0.05
                                        else "irregular"),
                             nn_spacing_cv=float(cv[nm]),
                             nn_median_m=float(I[I.dataset==nm].nn_median_m.iloc[0]),
                             rmse_vs_primary=float(rm), nmad_vs_primary=float(nmd),
                             provenance=prov, allowed_role=role, note=note))
    dec_rows.append(dict(source="HEC-RAS DniproGES1D geometry", n_points=np.nan,
                         structure="model geometry", nn_spacing_cv=np.nan,
                         nn_median_m=np.nan,
                         rmse_vs_primary=np.nan, nmad_vs_primary=np.nan,
                         provenance="MODEL_DERIVED_GEOMETRY",
                         allowed_role="MODEL_COMPARISON_ONLY",
                         note="not read here; its terrain source must be "
                              "identified before any use"))
    DEC = pd.DataFrame(dec_rows)
    DEC.to_csv(OUT / "11_source_provenance_decision.csv", index=False)
    print("\n" + "=" * 78)
    print("GATE J — PROVENANCE DECISION")
    print("=" * 78)
    print(DEC[["source", "n_points", "structure", "nn_spacing_cv",
               "nn_median_m", "rmse_vs_primary", "nmad_vs_primary",
               "provenance", "allowed_role"]].to_string(
        index=False, float_format=lambda v: f"{v:,.3f}"))

    _figures(P, px, py, pz, rx, ry, zr_best, cx, cy, zc_best,
             ir, ic, near, nearc, dr, dc, gr, best_r, best_c)
    print(f"\n-> {OUT}/  (11 tables)")
    print("\nSTOP. No point from either layer enters the production DEM "
          "until these numbers are reviewed.")


def _figures(P, px, py, pz, rx, ry, zr, cx, cy, zc,
             ir, ic, near, nearc, dr, dc, gr, best_r, best_c):
    ext_kw = dict(s=0.12, alpha=0.5, linewidths=0)
    # --- source overlap map
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.8), sharex=True, sharey=True)
    for a, (nm, X, Y, c) in zip(ax, (
            ("primary soundings (7,514)", px, py, RED),
            ("reliefUTM36N (994,609)", rx, ry, BLUE),
            ("CMAP2020 (95,132)", cx, cy, GREEN))):
        a.scatter(X / 1000, Y / 1000, color=c, **ext_kw)
        a.set_title(nm, fontsize=10.4, loc="left")
        a.set_xlabel("easting (km)")
        a.grid(alpha=0.25)
    ax[0].set_ylabel("northing (km)")
    fig.suptitle("p0h · spatial coverage of the three bathymetry sources",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "p0h_source_overlap_map.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # --- scatter vs primary + residual maps
    fig, ax = plt.subplots(2, 2, figsize=(13.6, 10.4))
    for j, (nm, idx, msk, zz, best) in enumerate((
            ("reliefUTM36N", ir, near, zr, best_r),
            ("CMAP2020", ic, nearc, zc, best_c))):
        a = ax[0, j]
        zs = zz[idx[msk]]
        a.scatter(pz[msk], zs, s=2.5, alpha=0.25, color=BLUE, linewidths=0)
        lim = [min(pz[msk].min(), zs.min()), max(pz[msk].max(), zs.max())]
        a.plot(lim, lim, color=RED, lw=1.2, ls="--")
        a.set_xlabel("primary sounding bed (m EVRF2019)")
        a.set_ylabel(f"{nm} ({best.hypothesis})")
        a.set_title(f"{'ab'[j]} · {nm} vs primary\nRMSE {best.rmse:.2f} m, "
                    f"bias {best.bias:+.2f} m, NMAD {best.nmad:.2f} m",
                    fontsize=10.2, loc="left")
        a.grid(alpha=0.25)
        a = ax[1, j]
        d = zs - pz[msk]
        v = float(np.percentile(np.abs(d), 95)) or 1.0
        sc = a.scatter(px[msk] / 1000, py[msk] / 1000, c=d, s=4,
                       cmap="RdBu_r", vmin=-v, vmax=v, linewidths=0)
        fig.colorbar(sc, ax=a, label="dz (m)")
        a.set_xlabel("easting (km)"); a.set_ylabel("northing (km)")
        a.set_title(f"{'cd'[j]} · residual map, {nm}", fontsize=10.2, loc="left")
    fig.suptitle("p0h · do the new layers reproduce the primary soundings?",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "p0h_primary_vs_sources.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # --- lattice structure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    a = ax[0]
    for ds, c in (("reliefUTM36N", BLUE), ("CMAP2020", GREEN),
                  ("primary_soundings", RED)):
        s = gr[gr.dataset == ds].sort_values("cell_m")
        a.plot(s.cell_m, 100 * s.frac_on_lattice, "o-", color=c, label=ds)
    a.set_xlabel("candidate cell size (m)")
    a.set_ylabel("% of points exactly on the lattice")
    a.legend(fontsize=8.5); a.grid(alpha=0.25)
    a.set_title("a · lattice test — a raster-to-point conversion sits on an "
                "exact grid", fontsize=10.2, loc="left")
    a = ax[1]
    n = min(4000, len(rx))
    sel = np.random.default_rng(0).choice(len(rx), n, replace=False)
    a.scatter(rx[sel] % 100, ry[sel] % 100, s=3, alpha=0.35, color=BLUE,
              linewidths=0)
    a.set_xlabel("easting mod 100 m"); a.set_ylabel("northing mod 100 m")
    a.set_title("b · reliefUTM36N coordinates modulo 100 m\n"
                "clustering at discrete values = grid origin",
                fontsize=10.2, loc="left")
    a.grid(alpha=0.25)
    fig.suptitle("p0h · structure tests", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "p0h_structure_tests.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # --- longitudinal profile
    fig, a = plt.subplots(figsize=(13.2, 5.6))
    for nm, X, Z, c in (("primary soundings", px, pz, RED),
                        ("reliefUTM36N", rx, zr, BLUE),
                        ("CMAP2020", cx, zc, GREEN)):
        bins = np.arange(525000, 680000, 1000)
        idx = np.digitize(X, bins)
        med = [np.median(Z[idx == i]) if (idx == i).sum() > 3 else np.nan
               for i in range(1, len(bins))]
        p05 = [np.percentile(Z[idx == i], 5) if (idx == i).sum() > 3 else np.nan
               for i in range(1, len(bins))]
        a.plot(bins[:-1] / 1000, med, color=c, lw=1.5, label=f"{nm} median")
        a.plot(bins[:-1] / 1000, p05, color=c, lw=0.9, ls=":",
               label=f"{nm} p05 (thalweg)")
    a.set_xlabel("easting (km)"); a.set_ylabel("bed elevation (m EVRF2019)")
    a.legend(fontsize=8, ncol=3); a.grid(alpha=0.25)
    a.set_title("p0h · longitudinal comparison in 1 km bins — the p05 line "
                "shows whether a source carries a deeper thalweg",
                fontsize=10.6, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "p0h_longitudinal_profile.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    for f in ("source_overlap_map", "primary_vs_sources", "structure_tests",
              "longitudinal_profile"):
        print(f"-> {FIGDIR/('p0h_'+f+'.png')}")


if __name__ == "__main__":
    main()
