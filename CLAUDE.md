# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research pipeline studying the Kakhovka reservoir/dam breach and the lower Dnipro using SWOT satellite
altimetry, ICESat-2, Sentinel-1/2, EMODnet bathymetry, and digitised historical gauge/bathymetry records.
It is a scientific investigation, not a maintained application: most of `scripts/` is a numbered lab
notebook (one script per analysis step, in the order it was run), while `src/` holds the two pieces of
code that are treated as an actual stable library.

The Kakhovka flood-state EO classification branch (Sentinel-1/2 water/flood-state mapping on the canonical
B1/B2/B3 frames, `p0o`-`p83`) has been split out into its own repository, `floodstate-eo`; the numbering gap
in `scripts/` where those files used to be is expected. This repository keeps the SWOT/ICESat-2/bathymetry
work, the below-dam Manning/HEC-RAS hydraulic chain, and the phase19/20 fragmentation study.

The repo depends on a **sibling checkout** of a companion project, `icesat2-atl13-kakhovka`, expected at
`../icesat2-atl13-kakhovka` next to this repository. It supplies read-only inputs this repo does not vendor
(EGG2015 geoid, EPSG:9902 grid, gauge yearbooks, ATL13 parquet outputs). Location resolution order (see
`src/swot_dnipro/config.py`): `$SWOT_DNIPRO_ICESAT_ROOT` env var, else `<this repo>/../icesat2-atl13-kakhovka`.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # pytest
pip install -e ".[notebook]"     # jupyter, matplotlib, xarray, netCDF4, etc.

# Tests
pytest tests/ -v
pytest tests/test_aoi.py -v -k some_test   # single test

# SWOT download CLI (the swotdl package)
python -m swotdl download --aoi <path/to/aoi.geojson> --product <CMR_SHORT_NAME> \
    --start YYYY-MM-DD --end YYYY-MM-DD [--dry-run]
bash scripts/run_download.sh      # runs all three AOIs/products from config/pipeline.yaml defaults

# Analysis scripts (no single entrypoint — run individually, in phase order)
python scripts/phase19_manifest.py
bash scripts/run_phase19.sh       # phase19 end-to-end chain, in order
```

`EARTHDATA_TOKEN` must be set (via `.env`, copied from `.env.example`, or exported) before any download.
It is loaded via `python-dotenv` **only** inside `swotdl.cli` — analysis scripts under `scripts/` do not
load `.env` themselves.

## Architecture

### Two separate Python packages under `src/`

- **`swotdl`** — the installable CLI (`swotdl` console script / `python -m swotdl`). Downloads SWOT L2 HR
  granules (LakeSP, RiverSP, PIXC) from NASA CMR using an Earthdata Bearer Token. Config is loaded from
  `config/pipeline.yaml` into frozen dataclasses (`swotdl/settings.py`); AOI handling in `swotdl/aoi.py`,
  CMR search in `swotdl/cmr.py`, download/retry logic in `swotdl/download.py`, a granule index in
  `swotdl/index.py`. This package has real unit tests in `tests/`.
- **`swot_dnipro`** — the analysis library imported by everything in `scripts/`:
  - `config.py` — the single source of truth for paths, the CRS policy, and science constants (breach
    date, datum offsets, gauge coordinates, PIXC classification codes). Read this file first when working
    on any analysis script.
  - `spatial_domains.py` — the **authoritative geometry registry**, backed by `config/spatial_domains.yaml`.
    A past incident (documented in `outputs/planning/00_REBUILD_EXECUTIVE_PLAN.md`) traced a truncated
    bathymetric DEM to three independent hand-written bbox literals in different scripts that each encoded
    a slightly different (wrong) reservoir extent. **No script may declare its own bbox/WKT literal for a
    named domain** — load it via `SD.load(name)` (shapely geometry) or `SD.as_bbox(name)` instead. Some
    domains are intentionally `UNRESOLVED` and raise `UnresolvedDomainError` until a real source geometry
    is confirmed — do not add a fallback/proxy geometry to silence that.
  - `sword.py`, `vertical.py`, `watermask.py` — SWORD river-chainage lookup, the SWOT/ICESat-2/gauge
    vertical-datum reconciliation chain, and the Sentinel-2 water-masking rules, respectively. Each has a
    substantial module docstring explaining the method and the empirical checks behind it — read it before
    changing the logic.
  - `plotting/` — shared figure styling.

### CRS policy (applies everywhere in `swot_dnipro` and `scripts/`)

- `EPSG:4326` — storage/interchange only (all source products arrive in this CRS).
- `EPSG:32636` (WGS 84 / UTM 36N) — **every** distance, area, buffer, or scale computation.
- `EPSG:3857` (Web Mercator) is never used for quantitative work.

### `scripts/` naming convention

Scripts are prefixed by campaign/phase and are meant to be read in numeric/lettered order within a prefix
(each depends on outputs of the previous one in its own sequence): `hist*` (historical bathymetry
digitisation and reconstruction), `k*` (ICESat-2/gauge/kriging validation chain, e.g. `k9`→`k10`→`k10b`…),
`part*` (vertical-datum and master-level reconciliation), `f*` (post-breach hydraulic domain), `p1*`
(stratified SWOT/Sentinel discovery and fetch), `phase19_*`/`phase20_*` (the current active phase — water
masks, classification, fragmentation), `qa*`/`v2`/`v3` (audits), `ms*` (manuscript/report assembly). Treat
a script's comments/docstring as the record of what was verified and what is still an open assumption —
this pipeline has a documented history of scripts asserting things that later turned out false (e.g. the
bbox incident above), so prefer trusting `outputs/planning/*.md` and a script's own audit trail over
re-deriving conclusions from scratch.

### `outputs/planning/00`–`11` (`*.md`)

These are the authoritative design/rebuild documents for the current architecture rework (spatial domains,
masks, bathymetry DEM, RMSE validation, cartographic atlas, recompute DAG). Consult them before restructuring
anything they cover — they carry root-cause analysis and explicit before/after rationale, not just plans.

### Data layout

- `data/raw/` — large, gitignored downloaded products (SWOT NetCDF, Sentinel scenes, PIXC). Populated by
  `swotdl` and the `fetch_*`/`phase19_fetch_*`/`p1_targeted_fetch.py` scripts.
- `data/historical/` — small, tracked CSVs of hand-digitised historical level/area/volume and reach data.
- `data/logs/` — run logs (gitignored, `.gitkeep` only).
- `outputs/{figures,figure_data,tables,reports,rasters,planning,presentation,archive,audit}/` — all derived
  products; regenerate from `scripts/`, never hand-edit.
