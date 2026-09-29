#!/usr/bin/env python
"""HISTORICAL 5 — put the historical tables on the modern chainage axis.

Tasks 5 and 6. The historical source measures distance along the reservoir axis
from the dam; this study measures SWORD flow-distance along the channel. They
are NOT the same number: the channel meanders, so the modern axis is longer.
Before any historical profile can be compared with an ICESat-2 profile, the two
have to be tied together, and the tie has to carry its uncertainty.

Longitude is NOT used as a proxy for river distance anywhere: the reservoir runs
WSW-ENE but with large excursions, and chainage correlates only loosely with
longitude.

Anchors available:
  - the dam, 0 km in both systems, by construction
  - "79-й км" and "117-й км" -- Table 20 rows that ARE historical chainages
  - gauges whose names appear in Table 20 (Nikopol, Rozumivka) and whose
    coordinates we hold, so their modern chainage is computable
  - the reservoir's total historical length, 238 km (Table 21)

Everything else in Table 20 is placed by interpolation and is flagged as such.

Outputs
-------
data/processed/historical/historical_table20_wse.csv
data/processed/historical/historical_reaches.csv
data/processed/historical/historical_fig16_profiles.csv
outputs/tables/historical_to_modern_chainage_crosswalk.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from swot_dnipro import config as CFG
from swot_dnipro import sword as SW

OUT = ROOT / "data/processed/historical"
HIST = ROOT / "data/historical"
SRC_PAGE = "Dnipro reservoirs monograph, photographed pages"
DATUM = "HISTORICAL_BALTIC (realization not stated in source)"

# Table 20 locations that are also gauges we hold coordinates for. Name matching
# is done explicitly rather than fuzzily -- there are only a handful and a wrong
# tie would propagate into every reach comparison.
GAUGE_TIE = {"Nikopol": "Nikopol", "Rozumivka": "Rozumivka",
             "Kakhovka HPP": None}          # dam handled separately

# Historical chainages read off the km axis of Fig. 13, which plots the named
# settlements against distance from the dam. Read from a rotated photograph, so
# +/- 5 km at best -- but the HORIZONTAL axis is far less affected by the
# perspective distortion that makes the vertical axis of Fig. 16 unusable.
# NOT included: Fig. 13's "Каховка" at ~12 km is the TOWN of Kakhovka, upstream
# of the dam; our "Nova Kakhovka" gauge sits at the dam itself, chainage 0.2 km.
# Tying them would have forced a ratio of 0.02 into the fit off a name collision.
FIG13_KM = {"Velyka Lepetykha": 72, "Nikopol": 133,
            "Blahovishchenka": 180, "Plavni": 197, "Rozumivka": 218}
# The same axis, for the places that are Table 20 ROWS. Kept separate from the
# gauge dict above so the two uses cannot be confused.
FIG13_T20_KM = {"Hornostaivka": 48, "Kachkarivka": 58, "Nikopol": 133,
                "Verkhnia Tarasivka": 183, "Bilenke": 200, "Lysa Hora": 207,
                "Rozumivka": 218, "Zaporizhzhia": 222,
                # Table 21 gives 238 km for the whole reservoir, Kakhovka HPP to
                # Dnipro HPP, which fixes the last row. Without it, row-order
                # interpolation cannot extrapolate and silently pinned Dnipro HPP
                # to Zaporizhzhia's 222 km, emptying the 210-240 km reach.
                "Dnipro HPP": 238}
FIG13_UNC_KM = 5.0
# A gauge far from the channel line snaps to the wrong node, so its modern
# chainage is unreliable however good the historical reading is.
GOOD_TIE_MAX_OFFSET_KM = 6.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t20 = pd.read_csv(HIST / "historical_longitudinal_wse.csv")
    reaches = pd.read_csv(HIST / "historical_reservoir_reaches.csv")

    # ---- modern chainage for the gauges ------------------------------------
    ch = pd.read_parquet(CFG.BULK_ROOT / "data_swot/processed/profiles/sword_dnipro_channel.parquet")
    print(f"SWORD channel: {len(ch):,} nodes, chainage "
          f"{ch.chain_km.min():.0f}..{ch.chain_km.max():.0f} km")
    print(f"  correlation of chainage with longitude: "
          f"{np.corrcoef(ch.chain_km, ch.lon)[0,1]:.3f}  <- not a substitute")

    g = pd.read_parquet(ROOT / "data/processed/gauges/gauge_levels_evrf2019.parquet")
    gp = g.groupby("name_en")[["lat", "lon"]].first().reset_index()
    tree = SW.build_chainage_tree(ch)
    gp["chain_km_modern"], gp["dist_to_channel_km"], _, _ = SW.assign_chainage(
        gp.lon.values, gp.lat.values, ch, tree=tree)
    print(f"\ngauge chainages on the modern axis:")
    for r in gp.sort_values("chain_km_modern").itertuples():
        print(f"  {r.name_en:<18}{r.chain_km_modern:>8.1f} km   "
              f"({r.dist_to_channel_km*1000:.0f} m from the channel line)")

    # ---- build the crosswalk ----------------------------------------------
    rows = []
    order = t20[["location", "order_from_dam", "chainage_km"]].drop_duplicates(
        "location").sort_values("order_from_dam")
    for r in order.itertuples():
        hist_km, how, unc, mod = r.chainage_km, "", np.nan, np.nan
        if r.location == "Kakhovka HPP":
            hist_km, mod, how, unc = 0.0, 0.0, "anchor: the dam, 0 in both", 0.0
        elif not np.isnan(r.chainage_km):
            how, unc = "anchor: the row IS a historical chainage marker", 2.0
        tie = GAUGE_TIE.get(r.location)
        if tie is not None and tie in set(gp.name_en):
            mod = float(gp.loc[gp.name_en == tie, "chain_km_modern"].iloc[0])
            how = (how + " | " if how else "") + f"gauge tie: {tie}"
            unc = 5.0
        rows.append({"location": r.location, "order_from_dam": r.order_from_dam,
                     "chainage_km_historical": hist_km,
                     "chainage_km_modern_sword": mod, "tie_method": how,
                     "uncertainty_km": unc})
    cw = pd.DataFrame(rows)

    # scale between the two axes, from the pairs that are tied on BOTH sides
    both = cw.dropna(subset=["chainage_km_historical", "chainage_km_modern_sword"])
    both = both[both.chainage_km_historical > 0]
    print(f"\npairs tied on BOTH axes: {len(both)}")
    for r in both.itertuples():
        print(f"  {r.location:<18} historical {r.chainage_km_historical:>6.1f} km  "
              f"modern {r.chainage_km_modern_sword:>6.1f} km  "
              f"ratio {r.chainage_km_modern_sword/r.chainage_km_historical:.3f}")

    # ---- tie the two axes through the gauges ------------------------------
    # Fig. 13 supplies the historical chainage, SWORD the modern one, for the
    # same named place. The tie is only as good as the gauge's distance from the
    # channel line: a gauge 15 km off snaps to the wrong node.
    print(f"\nties between the two axes (Fig. 13 vs SWORD):")
    print(f"  {'place':<20}{'hist km':>9}{'modern km':>11}{'ratio':>8}"
          f"{'offset':>9}  quality")
    ties = []
    for nm, hkm in FIG13_KM.items():
        row = gp[gp.name_en == nm]
        if not len(row):
            continue
        mkm = float(row.chain_km_modern.iloc[0])
        off = float(row.dist_to_channel_km.iloc[0])
        good = off <= GOOD_TIE_MAX_OFFSET_KM
        print(f"  {nm:<20}{hkm:>9.0f}{mkm:>11.1f}{mkm/hkm:>8.2f}{off:>8.1f} km"
              f"  {'USABLE' if good else 'rejected: too far from the channel'}")
        ties.append({"place": nm, "hist_km": hkm, "modern_km": mkm,
                     "offset_km": off, "usable": good})
    ties = pd.DataFrame(ties)
    ok = ties[ties.usable]
    ratio = float((ok.modern_km / ok.hist_km).median()) if len(ok) else np.nan
    resid = (ok.modern_km - ok.hist_km).abs().max() if len(ok) else np.nan
    print(f"\n  usable ties: {len(ok)} of {len(ties)}")
    print(f"  ratio from usable ties only : {ratio:.3f}")
    print(f"  largest |modern - historical| among them : {resid:.1f} km")
    print(f"\n  The two axes are the SAME SCALE to within the reading error. The")
    print(f"  rejected ties all give ratios of 1.14-1.20, and all of them are")
    print(f"  gauges 12-15 km from the channel line -- that is a snapping artefact,")
    print(f"  not sinuosity. An identity mapping is adopted:")
    print(f"      chainage_modern = chainage_historical,  +/- 10 km")
    print(f"  The +/-10 km combines the +/-{FIG13_UNC_KM:.0f} km reading error on")
    print(f"  Fig. 13's axis with the residual scatter of the usable ties. It is")
    print(f"  a PROVISIONAL mapping: two good ties cannot establish a reach-varying")
    print(f"  relation, and the historical axis is a centreline through a 2-25 km")
    print(f"  wide pool, not a thalweg.")
    RATIO_USED, UNC_KM = 1.0, 10.0

    # place the Table 20 rows that Fig. 13's axis names directly
    for nm, hkm in FIG13_T20_KM.items():
        hit = cw.location.eq(nm) & cw.chainage_km_historical.isna()
        cw.loc[hit, "chainage_km_historical"] = hkm
        cw.loc[hit, "tie_method"] = "historical chainage read from the Fig. 13 axis"
        cw.loc[hit, "uncertainty_km"] = FIG13_UNC_KM

    # the rest are interpolated on ROW ORDER, which the table itself fixes:
    # Table 20 lists its locations from the dam upstream, so order is monotone in
    # chainage even where the distance is unknown.
    anc = cw.dropna(subset=["chainage_km_historical"]).sort_values("order_from_dam")
    need = cw.chainage_km_historical.isna()
    cw.loc[need, "chainage_km_historical"] = np.interp(
        cw.loc[need, "order_from_dam"], anc.order_from_dam, anc.chainage_km_historical)
    cw.loc[need, "tie_method"] = "interpolated on Table 20 row order"
    cw.loc[need, "uncertainty_km"] = 15.0
    print(f"\n  Table 20 locations placed on the historical axis: "
          f"{len(anc)} from figure/marker anchors, {int(need.sum())} interpolated")

    fill = cw.chainage_km_historical.isna()
    cw.loc[~fill & cw.chainage_km_modern_sword.isna(), "chainage_km_modern_sword"] = (
        cw.loc[~fill & cw.chainage_km_modern_sword.isna(),
               "chainage_km_historical"] * RATIO_USED)
    cw.loc[~fill & cw.tie_method.eq(""), "tie_method"] = "identity mapping"
    cw.loc[~fill & cw.uncertainty_km.isna(), "uncertainty_km"] = UNC_KM
    cw["sinuosity_ratio_used"] = RATIO_USED
    ties.to_csv(CFG.TABLES / "historical_chainage_ties.csv", index=False)
    cw.to_csv(CFG.TABLES / "historical_to_modern_chainage_crosswalk.csv", index=False)
    n_anchor = int((cw.uncertainty_km <= 5).sum())
    print(f"\n  crosswalk: {len(cw)} locations, {n_anchor} anchored, "
          f"{int(cw.chainage_km_historical.isna().sum())} still unplaced")
    print(f"  -> {CFG.TABLES/'historical_to_modern_chainage_crosswalk.csv'}")

    # ---- processed historical tables --------------------------------------
    t20 = t20.merge(cw[["location", "chainage_km_historical",
                        "chainage_km_modern_sword", "tie_method",
                        "uncertainty_km"]], on="location", how="left")
    t20["historical_datum_label"] = DATUM
    t20["source_page"] = SRC_PAGE
    t20["source_figure"] = "Table 20"
    t20.to_csv(OUT / "historical_table20_wse.csv", index=False)
    reaches.to_csv(OUT / "historical_reaches.csv", index=False)
    print(f"  -> {OUT/'historical_table20_wse.csv'}")
    print(f"  -> {OUT/'historical_reaches.csv'}")

    # ---- Fig. 16: what can honestly be extracted --------------------------
    print("\n" + "=" * 72)
    print("FIG. 16 — WHY IT IS NOT DIGITISED TO NUMBERS HERE")
    print("=" * 72)
    print("  The photograph is taken at an angle: the horizontal gridlines")
    print("  converge toward the top of the plot, so pixel position is not linear")
    print("  in elevation. Reading curve 3 by eye and checking it against Table 20")
    print("  -- which contains exactly that curve as its Q0.1% column -- shows a")
    print("  systematic error of about +0.2 m in the flat reach. That is larger")
    print("  than the entire signal being tested in the pool (Table 20 spans")
    print("  0.00-0.06 m from the dam to Verkhnia Tarasivka at low flow), so a")
    print("  by-eye digitisation would be worse than useless there.")
    print("\n  What IS reliable from Fig. 16, and needs no digitisation:")
    print("   - curves 1 and 3 are already numeric: they are Table 20 columns.")
    print("   - curve 2 (observed, 22-25 Apr 1970) lies ON curve 1 over the whole")
    print("     length -> the design profile was confirmed in the field.")
    print("   - the observed profile is flat to ~185-190 km, then rises steeply,")
    print("     reaching ~19 m near 235 km.")
    print("   - curve 4 (observed, Hп=16.16 m, Q=9040 m3/s) starts ABOVE 16.00 at")
    print("     the dam, as its caption states, and sits above curve 1 through the")
    print("     mid-reservoir.")
    print("\n  To get curves 2 and 4 as numbers: a flatbed scan, or a photograph")
    print("  taken square-on, would make the rectification reliable. Recorded as")
    print("  a data request, not guessed.")
    pd.DataFrame([{
        "curve_id": 1, "observed_or_calculated": "calculated",
        "Q_m3_s": 8400, "H_dam_m": 16.00, "date": "1966 design",
        "status": "NOT digitised - available numerically in Table 20",
        "historical_datum_label": DATUM, "source_page": SRC_PAGE,
        "source_figure": "Fig. 16"},
        {"curve_id": 2, "observed_or_calculated": "OBSERVED",
         "Q_m3_s": 8400, "H_dam_m": 16.00, "date": "1970-04-22/25",
         "status": "NOT digitised - photograph has perspective distortion; "
                   "qualitatively lies on curve 1, flat to ~185-190 km then "
                   "rising to ~19 m near 235 km",
         "historical_datum_label": DATUM, "source_page": SRC_PAGE,
         "source_figure": "Fig. 16"},
        {"curve_id": 3, "observed_or_calculated": "calculated",
         "Q_m3_s": 23800, "H_dam_m": 16.00, "date": "1966 design",
         "status": "NOT digitised - IS the Q0.1% column of Table 20",
         "historical_datum_label": DATUM, "source_page": SRC_PAGE,
         "source_figure": "Fig. 16"},
        {"curve_id": 4, "observed_or_calculated": "OBSERVED",
         "Q_m3_s": 9040, "H_dam_m": 16.16, "date": "1970-04-28/30",
         "status": "NOT digitised - photograph has perspective distortion; "
                   "starts at 16.16 m at the dam per the caption",
         "historical_datum_label": DATUM, "source_page": SRC_PAGE,
         "source_figure": "Fig. 16"}]).to_csv(
        OUT / "historical_fig16_profiles.csv", index=False)
    print(f"\n  -> {OUT/'historical_fig16_profiles.csv'} (metadata + status, no "
          f"fabricated coordinates)")


if __name__ == "__main__":
    main()
