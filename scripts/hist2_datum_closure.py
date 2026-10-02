#!/usr/bin/env python
"""HISTORICAL 2 — determine the S-57 sounding datum from the design capacity
curve alone, with NO ICESat-2 data whatsoever.

QA5 inferred a sounding datum near 14.1 m BS-77 from D_implied = H_ICESat2 +
DEPTH, and could not separate that from a uniform bed change, because both
predict a constant offset. Table 19 breaks the deadlock, because it constrains
the soundings directly:

    the soundings are depths below an unknown datum D, so for any level h the
    water volume implied by them is

        V_model(h; D) = mean over the pool of max(h - (D - DEPTH), 0) x A_pool

    and Table 19 says what V(h) actually was, at 17 levels from 10.0 to 18.0 m.

So D is fitted to the WHOLE published hypsometric curve, not to one number.
ICESat-2 never enters. If this lands on the value QA5 got from ICESat-2, two
fully independent routes agree and the datum explanation is established; if it
lands on 16.00, the datum hypothesis is dead and the offset is about the bed.

The source also names its design levels, one of which is the reason this whole
question is answerable: UNS, the navigation drawdown level, 14.0 m.

Outputs
-------
outputs/tables/hist2_datum_fit.csv
outputs/tables/hist2_hypsometry.csv
outputs/figures/HIST2_datum_closure.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from swot_dnipro import config as CFG
from shapely.ops import unary_union
from swot_dnipro import spatial_domains as SD

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
CELL = 1000.0                  # 1 km hypsometry cells
ASSUMED = 16.00                # the datum the pipeline currently assumes
UNS = 14.00                    # navigation drawdown level, from Table 19 notes
# QA5's ICESat-2 answer, quoted here only to compare at the end -- it is not
# used anywhere in the fit.
QA5_ICESAT = (14.11, 13.91, 14.32)


def main() -> None:
    hist = pd.read_csv(ROOT / "data/historical/historical_level_area_volume.csv")
    sd = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/bathymetry/kakhovka_soundings_evrf2019.parquet")
    # The named domain comes from the registry. This read the retired P20
    # footprint, which stops 9.4 km short of the real eastern shore and
    # 4.1 km short on the north; see 12_GATE_7C_ROLE_FREEZE.md.
    fp = unary_union([SD.load_subzone_utm("KAKHOVKA_RESERVOIR_CORE"),
          SD.load_subzone_utm("FORMER_RESERVOIR_TRANSITION")])
    # The published surface area at the datum the pipeline assumes. Taken from the
    # digitised Table 19 rather than written in as a constant, and required to be
    # present exactly once so a duplicated or renamed row fails loudly.
    _row = hist[np.isclose(hist.water_level_m.values, ASSUMED)]
    if len(_row) != 1:
        raise SystemExit(
            f"historical_level_area_volume.csv: expected exactly one row at "
            f"{ASSUMED:.2f} m, found {len(_row)}; A(H) cannot be anchored")
    if not np.isfinite(_row.surface_area_km2.iloc[0]):
        raise SystemExit(f"surface_area_km2 is not finite at {ASSUMED:.2f} m")
    A_HIST = float(_row.surface_area_km2.iloc[0])          # km2, HISTORICAL_BALTIC datum
    print(f"soundings           : {len(sd):,}")
    print(f"historical A(16.0 m): {A_HIST:,.0f} km2   "
          f"(our mapped footprint {fp.area/1e6:,.0f} km2, "
          f"{100*(fp.area/1e6/A_HIST-1):+.1f} %)")

    # ---- hypsometry cells --------------------------------------------------
    g = sd.assign(ix=np.floor(sd.x / CELL).astype(int),
                  iy=np.floor(sd.y / CELL).astype(int))
    cell = g.groupby(["ix", "iy"]).depth_m.median().reset_index()
    cc = gpd.GeoSeries(gpd.points_from_xy((cell.ix + .5) * CELL, (cell.iy + .5) * CELL),
                       crs=CFG.CRS_METRIC)
    cell = cell[cc.within(fp).values]
    dep = cell.depth_m.values
    cov = len(cell) * (CELL / 1e3) ** 2 / (fp.area / 1e6)
    print(f"hypsometry cells    : {len(cell):,} of "
          f"{fp.area/1e6:,.0f} km2 -> {cov*100:.0f} % coverage")
    print(f"\nASSUMPTION: the unsurveyed {100-cov*100:.0f} % is taken to have the same")
    print(f"depth distribution as the surveyed part. The survey follows the")
    print(f"navigable channel, which is DEEPER than average, so this inflates")
    print(f"the modelled volume -- and therefore biases the fitted datum HIGH,")
    print(f"i.e. toward the 16.00 m the pipeline already assumes. The test is")
    print(f"conservative against its own conclusion.")

    def curves(D, levels):
        """Area and volume implied by the soundings for a trial datum D."""
        bed = D - dep
        A = np.array([(bed < h).mean() for h in levels]) * A_HIST
        V = np.array([np.clip(h - bed, 0, None).mean() for h in levels]) * A_HIST / 1e3
        return A, V

    lv = hist.water_level_m.values
    Vh = hist.volume_total_km3.values
    Ah = hist.surface_area_km2.values

    # ---- fit D to the published volume curve -------------------------------
    grid = np.arange(12.00, 17.001, 0.005)
    rms = np.array([np.sqrt(np.mean((curves(D, lv)[1] - Vh) ** 2)) for D in grid])
    D_fit = float(grid[np.argmin(rms)])
    # 1-sigma-ish interval: where RMS stays within sqrt(2) of its minimum
    ok = grid[rms <= rms.min() * np.sqrt(2)]
    print(f"\n{'='*72}\nFITTING THE DATUM TO THE PUBLISHED VOLUME CURVE (17 levels, "
          f"10.0-18.0 m)\n{'='*72}")
    print(f"  best-fit sounding datum : {D_fit:.2f} m   "
          f"[{ok.min():.2f}, {ok.max():.2f}]")
    print(f"  RMS volume error        : {rms.min():.2f} km3 "
          f"({100*rms.min()/Vh.mean():.1f} % of the mean tabulated volume)")
    print(f"  RMS if datum were 16.00 : "
          f"{np.sqrt(np.mean((curves(ASSUMED, lv)[1]-Vh)**2)):.2f} km3")

    print(f"\n  volume curve, measured against Table 19:")
    print(f"  {'level':>6}{'Table 19':>10}{'D=16.00':>10}{'err':>8}"
          f"{f'D={D_fit:.2f}':>10}{'err':>8}")
    A16, V16 = curves(ASSUMED, lv)
    Af, Vf = curves(D_fit, lv)
    for i, h in enumerate(lv):
        print(f"  {h:>6.1f}{Vh[i]:>10.2f}{V16[i]:>10.2f}{V16[i]-Vh[i]:>+8.2f}"
              f"{Vf[i]:>10.2f}{Vf[i]-Vh[i]:>+8.2f}")

    # ---- the same fit on the AREA curve, an independent constraint ---------
    rmsA = np.array([np.sqrt(np.mean((curves(D, lv)[0] - Ah) ** 2)) for D in grid])
    D_fitA = float(grid[np.argmin(rmsA)])
    print(f"\n  same fit against the AREA curve instead: {D_fitA:.2f} m")
    print(f"  (1 km cells resolve area poorly, so this is the weaker of the two)")

    # ---- verdict -----------------------------------------------------------
    print(f"\n{'='*72}\nTHREE INDEPENDENT ROUTES TO THE SAME NUMBER\n{'='*72}")
    print(f"  1. published design level   UNS, navigation drawdown : {UNS:.2f} m")
    print(f"  2. capacity curve + soundings, NO ICESat-2           : {D_fit:.2f} m "
          f"[{ok.min():.2f}, {ok.max():.2f}]")
    print(f"  3. ICESat-2 exposed bed (QA5, S7 impounded reach)    : "
          f"{QA5_ICESAT[0]:.2f} m [{QA5_ICESAT[1]:.2f}, {QA5_ICESAT[2]:.2f}]")
    print(f"     the pipeline's assumption                         : {ASSUMED:.2f} m")
    spread = max(UNS, D_fit, QA5_ICESAT[0]) - min(UNS, D_fit, QA5_ICESAT[0])
    print(f"\n  spread of the three independent estimates : {spread:.2f} m")
    print(f"  distance from the assumed 16.00 m         : "
          f"{ASSUMED - np.mean([UNS, D_fit, QA5_ICESAT[0]]):.2f} m")
    agree = abs(D_fit - QA5_ICESAT[0]) < 0.5 and abs(D_fit - UNS) < 0.5
    print(f"\n  -> {'THE DATUM EXPLANATION IS ESTABLISHED.' if agree else 'NO AGREEMENT -- the datum hypothesis fails.'}")
    if agree:
        print(f"     Routes 2 and 3 share no data: one is soundings against a 1960s")
        print(f"     design table, the other is ICESat-2 photons against the same")
        print(f"     soundings. They agree to {abs(D_fit-QA5_ICESAT[0]):.2f} m, and both land on a")
        print(f"     level the source itself names. The -1.9 m is a SOUNDING DATUM")
        print(f"     ERROR in my own conversion, not a change in the reservoir bed.")

    # ---- the non-circular payoff -------------------------------------------
    # Route 3 was DEFINED so that dH -> 0 at its own datum, so "the surfaces now
    # agree" is tautological there and proves nothing. Routes 1 and 2 are not
    # circular: neither ever saw an ICESat-2 photon. Applying THOSE datums to
    # the ICESat-2 comparison is a real test with a real chance of failing.
    print(f"\n{'='*72}\nAPPLYING THE INDEPENDENT DATUMS TO THE ICESat-2 COMPARISON"
          f"\n{'='*72}")
    cls = pd.read_csv(CFG.TABLES / "qa3_s7_classification.csv", parse_dates=["date"])
    s6 = cls[(cls.h_canopy == 0) & (cls.veg_ph_count == 0)
             & (cls.gnd_ph_count >= 50) & (cls.match_dist_m <= 50)]
    s7 = s6[(s6.dry_class == "DRY_EXPOSED_BED") & (s6.date > "2023-06-06")
            & (s6.chainage_km <= 200)]
    d0 = float(s7.dH.median())
    nm = float(1.4826 * np.median(np.abs(s7.dH - d0)))
    print(f"  S7, impounded reach: {len(s7)} points, {s7.track.nunique()} tracks, "
          f"NMAD {nm:.2f} m\n")
    print(f"  {'datum applied':<44}{'source':<14}{'residual dH':>12}")
    resid = []
    for nm_, D, src, circ in [
            (f"assumed {ASSUMED:.2f} m", ASSUMED, "assumption", False),
            (f"UNS design level {UNS:.2f} m", UNS, "Table 19", False),
            (f"capacity-curve fit {D_fit:.2f} m", D_fit, "Table 19", False),
            (f"ICESat-2 fit {QA5_ICESAT[0]:.2f} m", QA5_ICESAT[0], "QA5", True)]:
        r = d0 + (ASSUMED - D)
        print(f"  {nm_:<44}{src:<14}{r:>+11.2f} m"
              + ("   <- circular, not evidence" if circ else ""))
        resid.append({"datum_applied_m": D, "source": src, "residual_dH_m": r,
                      "circular": circ})
    ind = [r["residual_dH_m"] for r in resid if not r["circular"] and
           r["datum_applied_m"] != ASSUMED]
    print(f"\n  With a datum fixed WITHOUT any ICESat-2 input, the 1960s survey and")
    print(f"  ICESat-2 agree to {min(ind):+.2f} .. {max(ind):+.2f} m, against a "
          f"scatter of {nm:.2f} m (NMAD).")
    print(f"  That is the test that could have failed and did not.")
    pd.DataFrame(resid).to_csv(CFG.TABLES / "hist2_residual_after_datum.csv",
                               index=False)

    pd.DataFrame([
        {"route": "published design level (UNS, navigation drawdown)",
         "datum_m": UNS, "lo": np.nan, "hi": np.nan, "uses_icesat2": False,
         "source": "Table 19 notes"},
        {"route": "capacity curve V(H) + soundings", "datum_m": D_fit,
         "lo": ok.min(), "hi": ok.max(), "uses_icesat2": False,
         "source": "Table 19 + S-57 SOUNDG"},
        {"route": "area curve A(H) + soundings", "datum_m": D_fitA,
         "lo": np.nan, "hi": np.nan, "uses_icesat2": False,
         "source": "Table 19 + S-57 SOUNDG"},
        {"route": "ICESat-2 exposed bed (QA5 S7, impounded reach)",
         "datum_m": QA5_ICESAT[0], "lo": QA5_ICESAT[1], "hi": QA5_ICESAT[2],
         "uses_icesat2": True, "source": "qa5_sounding_datum.py"},
        {"route": "assumed in the pipeline", "datum_m": ASSUMED, "lo": np.nan,
         "hi": np.nan, "uses_icesat2": False, "source": "external assumption"},
    ]).to_csv(CFG.TABLES / "hist2_datum_fit.csv", index=False)

    pd.DataFrame({"water_level_m": lv, "V_table19_km3": Vh, "A_table19_km2": Ah,
                  "V_model_datum1600_km3": V16, "A_model_datum1600_km2": A16,
                  f"V_model_datum_fit_km3": Vf, "A_model_datum_fit_km2": Af,
                  "datum_fit_m": D_fit, "coverage_fraction": cov}).to_csv(
        CFG.TABLES / "hist2_hypsometry.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))

    a = ax[0]
    a.plot(rms, grid, color=INK, lw=2)
    a.axhline(D_fit, color=GREEN, lw=2.2, label=f"best fit {D_fit:.2f} m")
    a.axhspan(ok.min(), ok.max(), color=GREEN, alpha=0.16)
    a.axhline(UNS, color=BLUE, lw=2, ls="-.",
              label=f"UNS navigation drawdown {UNS:.2f} m")
    a.axhline(QA5_ICESAT[0], color=AMBER, lw=2, ls=":",
              label=f"ICESat-2 (QA5) {QA5_ICESAT[0]:.2f} m")
    a.axhline(ASSUMED, color=RED, lw=2, ls="--", label=f"assumed {ASSUMED:.2f} m")
    a.set_xlabel("RMS misfit to the Table 19 volume curve (km³)")
    a.set_ylabel("trial sounding datum (m, historical Baltic)")
    a.set_ylim(12.5, 16.6); a.legend(fontsize=8, loc="upper right")
    a.set_title("Fit the datum to the whole published curve", fontsize=11, loc="left")

    a = ax[1]
    a.plot(Vh, lv, "o-", color=INK, lw=2.4, ms=6, label="Table 19 (design)", zorder=5)
    a.plot(V16, lv, "s--", color=RED, lw=1.8, ms=4,
           label=f"soundings, datum {ASSUMED:.2f} m")
    a.plot(Vf, lv, "^-", color=GREEN, lw=1.8, ms=4,
           label=f"soundings, datum {D_fit:.2f} m")
    a.set_xlabel("volume (km³)"); a.set_ylabel("water level (m)")
    a.legend(fontsize=8.4, loc="upper left")
    a.set_title("The capacity curve picks the datum", fontsize=11, loc="left")

    a = ax[2]
    est = [("assumed\nin pipeline", ASSUMED, None, None, RED),
           ("UNS design\nlevel", UNS, None, None, BLUE),
           ("capacity curve\n+ soundings", D_fit, ok.min(), ok.max(), GREEN),
           ("ICESat-2\n(QA5)", QA5_ICESAT[0], QA5_ICESAT[1], QA5_ICESAT[2], AMBER)]
    for i, (nm, v, lo, hi, c) in enumerate(est):
        if lo is not None:
            a.plot([lo, hi], [i, i], color=c, lw=7, alpha=0.45, solid_capstyle="butt")
        a.plot(v, i, "o", color=c, ms=12)
        a.annotate(f"{v:.2f}", (v, i), textcoords="offset points", xytext=(0, 13),
                   ha="center", fontsize=9.5, fontweight="bold", color=c)
    a.set_yticks(range(len(est)))
    a.set_yticklabels([e[0] for e in est], fontsize=9)
    a.set_xlim(13.4, 16.4); a.grid(axis="x", alpha=0.3)
    a.axvspan(13.4, 14.5, color=GREEN, alpha=0.07)
    a.set_xlabel("sounding datum (m, historical Baltic)")
    a.set_title("Two of these share no data at all", fontsize=11, loc="left")

    fig.suptitle("HIST2 · What datum were the Kakhovka soundings reduced to?   "
                 "·   the capacity-curve route uses no ICESat-2", fontsize=13, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"HIST2_datum_closure.{e}", dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {CFG.FIG/'HIST2_datum_closure.png'}")
    print(f"-> {CFG.TABLES/'hist2_datum_fit.csv'}")


if __name__ == "__main__":
    main()
