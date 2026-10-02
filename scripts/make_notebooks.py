#!/usr/bin/env python
"""Generate the reproducible analysis notebooks.

Notebooks are deliberately thin: they call ``src/swot_dnipro`` and the two
pipeline scripts. No scientific algorithm is defined inside a notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(exist_ok=True)

PROV = '''\
# --- provenance: environment, git SHA and input hashes -----------------------
import hashlib, json, subprocess, sys, platform
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np, pandas as pd, geopandas as gpd, pyproj, rasterio, matplotlib
from swot_dnipro import config as CFG

np.random.seed(CFG.SEED)

def sha256(p, cap=64 * 1024 * 1024):
    p = Path(p)
    if not p.exists() or p.stat().st_size > cap:
        return f"(skipped: {'missing' if not p.exists() else 'too large'})"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=CFG.ROOT, text=True).strip()
    except Exception:
        return "(not a git checkout)"

print("python      ", platform.python_version())
print("numpy       ", np.__version__)
print("pandas      ", pd.__version__)
print("geopandas   ", gpd.__version__)
print("matplotlib  ", matplotlib.__version__)
print("pyproj/PROJ ", pyproj.__version__, "/", pyproj.proj_version_str)
print("rasterio/GDAL", rasterio.__version__, "/", rasterio.__gdal_version__)
print("git SHA     ", git_sha())
print("seed        ", CFG.SEED)
print()
for p in [CFG.EGG2015_TIF, CFG.UA2019Z_ASC, CFG.CORRECTOR_BY_STATION,
          CFG.KHERSON_GAUGE_PARQUET, CFG.KHERSON_ATL13]:
    print(f"  {sha256(p)}  {Path(p).name}")
'''


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


SPECS = {
"10_swot_vertical_chain_validation": [
    md("""# 10 — SWOT vertical chain validation

**Scientific question.** What exactly is `PIXC height`, and which geophysical
corrections must be applied to place SWOT and ICESat-2 in one vertical frame?

**Inputs** PIXC granules (`data/raw/pixc_nova_kakhovka/`), one RiverSP Node granule
on the same cycle/pass, EGG2015, ua_2019z.asc.
**Outputs** `outputs/figure_data/Fig03_*`, `outputs/tables/vertical_reference_audit.csv`.
**Statistical unit** one RiverSP node. **Validation** independent-product sign test.
"""),
    code(PROV),
    md("""## 1. Product definitions — read from the granules, not assumed

PIXC PDD **D-56411 Rev C p. 53**: `geoid`, `solid_earth_tide`, `load_tide_fes` and
`pole_tide` are each *"reported for reference but is not applied to the reported
height"*; `height` is *"Height of the pixel above the reference ellipsoid"*.

ATL03 ATBD **v007 p. 8**: *"ATL03 heights are in the tide-free system"*, and the
solid-Earth/load/pole tides are **already applied** upstream (pp. 20, 26)."""),
    code("""import netCDF4 as nc
f = sorted(CFG.PIXC_DIR.glob("*.nc"))[0]
ds = nc.Dataset(str(f)); pc = ds.groups["pixel_cloud"]
for k in ["height", "geoid", "solid_earth_tide", "load_tide_fes", "pole_tide"]:
    v = pc.variables[k]
    print(f"### {k}")
    for a in ("long_name", "source", "comment"):
        if a in v.ncattrs():
            print("   ", a, ":", str(v.getncattr(a))[:220])
ds.close()"""),
    md("""## 2. Sign test against RiverSP (same cycle and pass)

The sign is established by agreement with an independently documented product,
never by numerical plausibility."""),
    code("""d = pd.read_csv(CFG.FIGDATA / "Fig03_pixc_riversp_sign_validation.csv")
display(d[["variant", "chain", "interpretation", "median_diff_m", "nmad_m", "n_nodes"]])
print("\\nAccepted chain: h_SWOT = height - solid_earth_tide - load_tide_fes - pole_tide")"""),
    md("""## 3. Permanent-tide systems

| quantity | system |
|---|---|
| ATL13 `ht_water_surf` | tide-free |
| SWOT `h` after §2 (`solid_earth_tide` excludes the permanent tide) | mean-tide crust |
| EGG2015, EVRF2019 (EPSG:9389) | zero-tide |

For crustal ellipsoidal heights zero-tide ≡ mean-tide, so **only ICESat-2 needs
harmonising**, via `tide_earth_free2mean = 0.06029 − 0.180873 sin²φ` (ATL03 ATBD p. 125)."""),
    code("""print(pd.read_csv(CFG.TABLES / "vertical_reference_audit.csv").to_string(index=False)[:2600])
for name, lat in [("Kherson", 46.6238), ("Nova Kakhovka", 46.7754), ("Rozumivka", 47.7712)]:
    print(f"  free2mean({name:15s} phi={lat:.4f}) = {CFG.free2mean(lat):+.4f} m")"""),
],

"11_swot_kherson_pilot": [
    md("""# 11 — Kherson pilot: SWOT, ICESat-2 and the gauge

**Scientific question.** How do SWOT and ICESat-2 compare with the Kherson gauge
over the six downloaded pre-breach dates?

**Inputs** 6 PIXC granules, gauge 80805 daily series, ATL13 pass levels.
**Outputs** `Fig05_*`, `Fig07_*`, `Fig08_*` figure data.
**Statistical unit** SWOT: one overpass. ICESat-2: date × RGT. Bootstrap seed 42.

> **Caveat established in notebook 13:** the ICESat-2 pass-levels used here are an
> AOI-scale aggregate a median 29.8 km from the gauge, while SWOT is within 1 km.
> The gauge-mediated cross-sensor difference is therefore **confounded by the
> channel gradient**. Use notebook 13 for the like-for-like number."""),
    code(PROV),
    code("""import subprocess
subprocess.run([sys.executable, "../scripts/make_figure_data.py"], check=True)"""),
    code("""sw = pd.read_csv(CFG.FIGDATA / "Fig05_kherson_timeseries_swot.csv", parse_dates=["date", "swot_utc"])
ic = pd.read_csv(CFG.FIGDATA / "Fig05_kherson_timeseries_icesat.csv", parse_dates=["date", "icesat_utc"])
display(sw[["date", "swot_utc", "n_px", "H_SWOT_EGG2015_m", "nmad_m", "stage_bs77_m",
            "H_gauge_EVRF2019_m", "c_SWOT_m"]])
display(ic[["date", "icesat_utc", "rgt", "n_points", "H_tidefree_m", "H_meantide_m",
            "dist_to_gauge_km", "qc_flag"]])"""),
    md("## Residual statistics and harmonisation variants"),
    code("""display(pd.read_csv(CFG.FIGDATA / "Fig07_sensor_residuals.csv"))
display(pd.read_csv(CFG.FIGDATA / "Fig08_harmonisation_variants.csv"))"""),
],

"12_swot_spatial_diagnostics": [
    md("""# 12 — Spatial diagnostics: water mask, radius, pixel maps

**Scientific question.** How sensitive is the SWOT estimate to the water-mask and
aggregation-radius choices, and where do the accepted pixels lie?

**Outputs** `Fig09_*`, `Fig10_*`, `Fig11_*`, `Fig14_*`."""),
    code(PROV),
    code("""wm = pd.read_csv(CFG.FIGDATA / "Fig09_water_mask_sensitivity.csv")
rad = pd.read_csv(CFG.FIGDATA / "Fig10_radius_sensitivity.csv")
display(wm); display(rad)
print("Accepted: classification == 4 (open_water), radius 1 km.")
print("Broader classes move the median by up to "
      f"{(wm.median_residual_m.max() - wm.median_residual_m.min()):.3f} m;")
print("radius 0.5 -> 10 km moves it by "
      f"{(rad.median_residual_m.iloc[-1] - rad.median_residual_m.iloc[0]):+.3f} m "
      "(real channel gradient).")"""),
],

"13_swot_three_way_comparison": [
    md("""# 13 — Three-way comparison, 2023-04-05, and the co-located cross-sensor test

**Scientific question.** On the one date when all three systems observed, what do
they say — and what is the *like-for-like* SWOT ↔ ICESat-2 difference?

**Terminology.** The triple is **SAME-DAY, NOT SIMULTANEOUS**: Δt(SWOT − ICESat-2)
= 14.22 h and the gauge is a daily value with no epoch.

**Bug found by the figure work.** ICESat-2 has **zero** ATL13 segments within 10 km
of the Kherson gauge in the 2023 pre-breach window. Comparing gauge-collocated SWOT
against AOI-scale ICESat-2 mixes in ~30 km of channel gradient. The fix is to match
the two sensors *to each other* (500 m) and leave the gauge out."""),
    code(PROV),
    code("""seg = pd.read_parquet(CFG.ICESAT_ROOT / "data/processed/kherson_atl13_segments.parquet")
seg["dt"] = pd.to_datetime(seg["time"], utc=True, errors="coerce").dt.tz_localize(None)
from swot_dnipro.vertical import haversine_km
seg["dist_km"] = haversine_km(seg.lon, seg.lat, CFG.KHERSON_GAUGE[2], CFG.KHERSON_GAUGE[3])
pre = seg[(seg.dt >= "2023-01-01") & (seg.dt < CFG.BREACH_DATE)]
for r in (0.5, 1, 2, 5, 10, 20):
    print(f"  ATL13 segments within {r:4.1f} km of the gauge, 2023 pre-breach: "
          f"{int((pre.dist_km < r).sum()):5d}")"""),
    code("""subprocess.run([sys.executable, "../scripts/make_colocated.py"], check=True)"""),
    code("""summ = pd.read_csv(CFG.FIGDATA / "Fig16_colocated_summary.csv")
beam = pd.read_csv(CFG.FIGDATA / "Fig16_colocated_pairs_perbeam.csv")
display(summ.T); display(beam)"""),
    md("""**Result.** Spatially co-located, same-day, gauge-free:
`H_SWOT − H_ICESat-2 = −0.179 m`, NMAD 0.011 m over 6 beams,
95 % CI [−0.207, −0.146] — the CI **excludes zero**.

**But** all six beams come from a single RGT on a single date, so they are
pseudo-replicates of one overpass, and Δt is 14.22 h. This is one clean
measurement, not six independent ones."""),
],

"14_reservoir_corrector_harmonised": [
    md("""# 14 — Reservoir correctors before and after permanent-tide harmonisation

**Scientific question.** How much does harmonising ATL13's tide-free heights to the
zero/mean-tide crust convention change the six empirical station correctors?

**Input** `icesat2-atl13-kakhovka/outputs/tables/egg2015_to_evrf2019_by_station.csv`
(READ ONLY — the ICESat-2 project is not modified).
**Output** `Fig04_*`, `Fig15_*`."""),
    code(PROV),
    code("""d = pd.read_csv(CFG.FIGDATA / "Fig04_permanent_tide_correctors.csv")
s = pd.read_csv(CFG.FIGDATA / "Fig04_regional_median_summary.csv")
display(d[["name_en", "lat", "free2mean_m", "c_original_m", "c_harmonised_m", "delta_c_m",
           "empirical_nmad_m", "n_matchups"]])
display(s)"""),
    md("""**Recommendation, not applied.** The regional median moves
`c = −0.1695 m → −0.1321 m`. The existing chain `H = h_ATL13 − ζ_EGG2015` mixes a
tide-free ellipsoidal height with a zero-tide quasigeoid; adding
`tide_earth_free2mean` removes that inconsistency. Do **not** mix the old and
harmonised values in one table."""),
],

"15_publication_figures": [
    md("""# 15 — Publication figures

Assembly only. Every number is read from `outputs/figure_data/`; no science here.
Run notebooks 10–14 (or the two pipeline scripts) first."""),
    code(PROV),
    code("""subprocess.run([sys.executable, "../scripts/make_figure_data.py"], check=True)
subprocess.run([sys.executable, "../scripts/make_colocated.py"], check=True)
subprocess.run([sys.executable, "../scripts/make_figures.py"], check=True)"""),
    code("""from IPython.display import Image, display as disp
for p in sorted(CFG.FIG.glob("Fig*.png")):
    print(p.name); disp(Image(str(p), width=880))"""),
],
}

for name, cells in SPECS.items():
    (NB / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1))
    print("wrote", name + ".ipynb")
