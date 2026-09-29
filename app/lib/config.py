"""Paths and links for the companion app — the only place either is spelled out.

Every path is relative to the repository root, so the app runs from a plain
clone and on Streamlit Community Cloud. Links point at GitHub; set the three
PRODUCTION values below after the app is deployed and the branch is merged.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- PRODUCTION: replace after deployment -------------------------------------
REPO_URL = "https://github.com/NikoriakViktot/SWOT-DNIPRO"
BRANCH = "paper1-v6"                 # links resolve on the release tag (immutable)
APP_URL = "https://swot-dnipro.streamlit.app"   # placeholder until deployed
RELEASE_TAG = "paper1-v6"
RELEASE_URL = f"{REPO_URL}/releases/tag/{RELEASE_TAG}"   # v6.docx is a release asset, not in git
# ------------------------------------------------------------------------------

PAPER = ROOT / "outputs/paper"
MANUSCRIPT_MD = PAPER / "paper1_manuscript_en-v6.md"
MANUSCRIPT_DOCX = PAPER / "paper1_manuscript_en-v6.docx"
FIGURES = PAPER / "figures"
VALIDATION = PAPER / "validation"
ZONES = PAPER / "zones"
APP_DATA = PAPER / "app_data"
NOTEBOOKS = ROOT / "notebooks"

SUMMARY = VALIDATION / "ms7_summary.csv"
EVIDENCE = VALIDATION / "ms7_evidence.csv"
INVENTORY = VALIDATION / "ms7_station_inventory.csv"
SLOPE_SAMPLING = VALIDATION / "ms7b_slope_geoid_sampling_summary.csv"
SLOPE_FRAMES = VALIDATION / "ms7b_slope_frames_summary.csv"
BUILD_INFO = VALIDATION / "build_info.json"
ZONE_MANIFEST = ZONES / "paper1_zones_manifest.json"

# the scripts that make every number and figure of the paper's validation
PIPELINE = [
    ("scripts/ms5_paper1_zones.py", "Analysis zones R/F/D/E, frozen with hashes"),
    ("scripts/ms6_paper1_figures.py", "Figure 1 (study area) and Figure S1 (crossing map)"),
    ("scripts/ms6b_swot_icesat_crossings.py", "SWOT–ICESat-2 crossings from the primary products"),
    ("scripts/ms6c_crossing_provenance.py", "Why the v5 crossing numbers cannot be reproduced"),
    ("scripts/ms7_validation_paths.py", "Validation paths V1–V6: evidence and summary tables"),
    ("scripts/ms7b_slope_geoid_sampling.py", "Headline slope vs EGG2015 sampling (nearest / bilinear)"),
    ("scripts/ms7c_validation_figures.py", "Figures 5, S2, S3, S5, S6 from the ms7 tables"),
    ("scripts/ms7d_manuscript_audit.py", "Manuscript ↔ tables ↔ figures consistency audit"),
    ("app/prepare_app_data.py", "The light data layer this app reads"),
]

# figure status: how each figure of the manuscript was produced
FIGURE_STATUS = {
    "F00_study_area.png": ("recomputed", "scripts/ms6_paper1_figures.py"),
    "F18_swot_icesat_whole_zone_map.png": ("recomputed", "scripts/ms6_paper1_figures.py"),
    "FS_V4_swot_icesat_agreement.png": ("recomputed", "scripts/ms7_validation_paths.py"),
    **{f: ("recomputed", "scripts/ms7c_validation_figures.py") for f in (
        "F16_closure_offsets.png", "F17_reference_surfaces.png", "F19_swot_icesat_egg2015_along_system.png",
        "F20_vertical_chains.png", "F21_covariability_scatter.png", "F22_gauge_network_correlation.png",
        "F23_slope_frame_invariance.png", "F24_station_panels.png", "F25_breach_fortnight_posts.png",
        "F26_downstream_posts_2023.png")},
    # slope / heterogeneity / drawdown figures: checked against the tables, unchanged
    **{f: ("verified — v5 image consistent with the tables", None) for f in (
        "F02_slope_per_overpass.png", "F04_heterogeneity.png", "F15_swot_drawdown_and_wave.png")},
}
DEFAULT_STATUS = ("carried from v5 — pending final rebuild", None)

CLAIMS = {
    "V1_ATL13_GAUGE_CLOSURE": "V1 — Gauge ↔ ATL13 closure",
    "V2_ATL13_GAUGE_COVARIABILITY": "V2 — Gauge ↔ ATL13 co-variability",
    "V3_ROZUMIVKA_TRANSFER": "V3 — Rozumivka cross-epoch transfer",
    "V4_SWOT_ICESAT_DIRECT": "V4 — Direct SWOT ↔ ICESat-2 crossings",
    "V5_LAKESP_PRODUCT": "V5 — LakeSP (supplementary)",
    "V6_KHERSON_CLOSURE": "V6 — Kherson local closure",
    "V7_DOWNSTREAM_POSTS_2023": "V7 · V8 · network — posts, surfaces, gauges",
}

ZONE_COLOURS = {"R": "#5b8fa8", "F": "#c1402a", "D": "#3f7d4e", "E": "#7a4f9e"}


def gh(path: str | Path, kind: str = "blob") -> str:
    """GitHub URL of a repository path (blob for files, tree for folders)."""
    p = Path(path)
    rel = p.relative_to(ROOT).as_posix() if p.is_absolute() else p.as_posix()
    return f"{REPO_URL}/{kind}/{BRANCH}/{rel}"
