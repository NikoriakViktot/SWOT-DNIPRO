#!/usr/bin/env python
"""P44 -- HEC-RAS import package from the p43 roughness states (plan 15, WP4). Design only; HEC-RAS is not run here.

For every p43 state raster (outputs/rasters/roughness/<domain>/<grid>_manning_classes_<state>.tif) this writes
  outputs/hecras/<domain>/<grid>_<state>/
      landcover_<state>.tif            uint8 class code raster (HEC-RAS 2D "Land Cover layer": import in RAS Mapper)
      landcover_<state>.tif.vat.dbf    (not written; the class table below is the attribute table)
      manning_table_<state>.csv        Name,ID,n_low,n_base,n_high,percent_impervious  (paste into the Land Cover
                                       Manning's n table; one column per scenario N_LOW / N_BASE / N_HIGH)
      landcover_<state>_winter.csv     same with the leaf-off factor on woody/reed classes
      README.md                        what the layer is, its provenance and the steps: create Land Cover layer ->
                                       assign n -> associate geometry -> (Calibration Regions) -> RECOMPUTE 2D
                                       hydraulic property tables (HEC-RAS 2D UM 6.4, audit SRC-02)
  outputs/hecras/scenarios.csv         one row per (domain, grid, state, variant) incl. the dam-break bounding runs
                                       for BREACH_2023 (x0.8, x1.2, near-dam x2 within 5 km; audit SRC-01) and the
                                       paired roughness-only runs (BREACH_2023 -> STATE_2025/CURRENT_2026)
  outputs/hecras/sensitivity_design.csv  T1..T9 (audit 10_MANNING §5) with the outputs to compare
The 1D legacy geometry (DniproGES1D) is NOT touched (MODEL_COMPARISON_ONLY). For a 1D geometry the class raster
would still have to be averaged along cross-section LOB/channel/ROB -- that step is out of scope here.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml

Y = yaml.safe_load((ROOT / "config/roughness_classes.yaml").read_text())
NAMES = {int(k): v for k, v in Y["codes"].items()}
SRC = ROOT / "outputs/rasters/roughness"
OUT = ROOT / "outputs/hecras"
METRICS = "WSE; depth; velocity; arrival time; inundation extent; conveyance per profile line; discharge split channel/floodplain; pond connectivity"
PURPOSE = {"BREACH_2023_bed": "roughness of the reservoir BED for the emptying / wave-over-bed hydraulics (pool only)",
           "BREACH_2023": "pre-breach cover: dam-break reconstruction (June 2023)", "FIRST_EXPOSURE_2023": "first post-drainage state (late summer 2023)",
           "STATE_2024": "second season", "STATE_2025": "third season", "CURRENT_2026": "current state (2026)"}


def table(path: Path, factor_woody: float = 1.0):
    woody = ("young_woody_sparse", "young_woody_dense", "mature_woody_legacy", "reed_tall_herb")
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Name", "ID", "n_low", "n_base", "n_high", "percent_impervious", "calibration_status", "sources"])
        for k, nm in NAMES.items():
            if nm not in Y["classes"]:
                continue
            c = Y["classes"][nm]; fac = factor_woody if nm in woody else 1.0
            w.writerow([nm, k, round(c["n_low"] * fac, 4), round(c["n_base"] * fac, 4), round(c["n_high"] * fac, 4), 90 if nm == "built" else 0, c["status"], "|".join(c["sources"])])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scen, n = [], 0
    for cls in sorted(SRC.rglob("*_manning_classes_*.tif")):
        domain = cls.parent.name
        grid, state = cls.name.split("_manning_classes_")[0], cls.name.split("_manning_classes_")[1].replace(".tif", "")
        d = OUT / domain / f"{grid}_{state}"; d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cls, d / f"landcover_{state}.tif")
        table(d / f"manning_table_{state}.csv"); table(d / f"manning_table_{state}_winter.csv", float(Y["season"]["leaf_off_factor_woody"]))
        (d / "README.md").write_text(f"""# HEC-RAS land-cover package: {domain} / {grid} / {state}

Purpose: {PURPOSE.get(state, state)}
Layer: `landcover_{state}.tif` -- uint8 class codes (0 = no data), EPSG:32636, 20 m, produced by
`scripts/p43_roughness_states.py` from project data (Sentinel-2 classes, hydroperiod, WorldCover 2021, Dynamic World,
ATL08 canopy). n values: `manning_table_{state}.csv` (`config/roughness_classes.yaml`), literature priors -- NOT calibrated.
Winter (leaf-off) variant: `manning_table_{state}_winter.csv` (woody/reed x {Y['season']['leaf_off_factor_woody']}).

Steps (HEC-RAS 6.x, RAS Mapper): Map Layers -> Land Cover -> Create a new Land Cover layer from this GeoTIFF ->
Data Table Editor: paste the ID/Name/n column of the chosen scenario (N_LOW / N_BASE / N_HIGH) -> Manage Geometry
Associations: attach the layer to the NEW 2D geometry (project terrain, not DniproGES1D) -> (optional) Calibration
Regions with a traceable reason -> RECOMPUTE the 2D flow-area hydraulic property tables before running.
Dam-break (BREACH_2023 only): channel n bounds 0.025-0.075, overbank 0.05-0.15; bounding runs x0.8 / x1.2 on the whole
field; x2 within 5 km downstream of the dam (HEC-RAS 6.4 dam-break guidance).
""")
        for var in ("N_LOW", "N_BASE", "N_HIGH"):
            n += 1
            scen.append(dict(scenario_id=f"HS-{n:03d}", domain=domain, grid=grid, state=state, variant=var, landcover=str(d / f"landcover_{state}.tif"), table=str(d / f"manning_table_{state}.csv"),
                             column=var.split("_")[1].lower(), purpose=PURPOSE.get(state, ""), metrics=METRICS, status="READY_TO_IMPORT"))
        if state == "BREACH_2023":
            for var, note in (("N_BASE_x0.8", "official -20 % bounding run"), ("N_BASE_x1.2", "official +20 % bounding run"), ("N_DAMBREAK_NEAR_DAM_x2", "x2 within 5 km downstream of the dam")):
                n += 1
                scen.append(dict(scenario_id=f"HS-{n:03d}", domain=domain, grid=grid, state=state, variant=var, landcover=str(d / f"landcover_{state}.tif"), table=str(d / f"manning_table_{state}.csv"),
                                 column="base (apply factor at import)", purpose=note, metrics=METRICS, status="READY_TO_IMPORT"))
    for a, b in (("BREACH_2023", "STATE_2025"), ("BREACH_2023", "CURRENT_2026"), ("STATE_2024", "STATE_2025")):
        n += 1
        scen.append(dict(scenario_id=f"HS-{n:03d}", domain="pool+below_dam_floodplain", grid="all", state=b, variant=f"PAIRED {a} -> {b} (roughness only)", landcover="", table="",
                         column="base", purpose="identical terrain/mesh/BC; only the land-cover layer changes; then a second pair with post-breach terrain (ATL08 ground)", metrics=METRICS, status="DESIGN"))
    with (OUT / "scenarios.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scen[0])); w.writeheader(); w.writerows(scen)
    design = [
        ("T1 roughness bounds", "n field N_LOW/N_BASE/N_HIGH per state", "breach hydrograph, terrain, mesh, BC", "arrival later & depth higher with n"),
        ("T2 official +/-20 %", "all n x0.8 / x1.2 (BREACH_2023)", "as T1", "bracket of T1"),
        ("T3 near-dam factor", "x2 within 5 km", "as T1", "local stage rise"),
        ("T4 breach parameters", "breach width/time in the literature range", "n = N_BASE", "equifinality check vs T1"),
        ("T5 terrain", "pre-breach bed DEM +/- method spread; post-breach ATL08 ground", "n = N_BASE", "separates morphology from roughness"),
        ("T6 paired succession", "land cover BREACH_2023 -> STATE_2025 only", "everything else", "pool: conveyance loss where n doubled"),
        ("T7 depth dependence", "n(h): emergent (n_high) vs submerged (n_low) by canopy height 1-5 m", "as T6", "shallow runs more sensitive"),
        ("T8 leaf-on/off", "winter tables (x0.7 woody/reed)", "as T6", "~30 % lower n on woody cells"),
        ("T9 holdout validation", "calibrate on Kherson WSE; validate on S1 extents / arrival (p42 rasters) not used for tuning", "-", "identifiability statement"),
    ]
    with (OUT / "sensitivity_design.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["test", "vary", "hold", "expected", "outputs"]); [w.writerow(list(d) + [METRICS]) for d in design]
    print(f"{n} scenarios, {len(design)} tests -> {OUT}")


if __name__ == "__main__":
    main()
