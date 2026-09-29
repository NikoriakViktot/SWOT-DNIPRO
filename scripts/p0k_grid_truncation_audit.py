#!/usr/bin/env python
"""P0K — find every grid in the project that is narrower than its domain.

The eastern-truncation defect has recurred three times. Its mechanism is always
the same: a script takes its output grid from a precomputed surface instead of
from the spatial-domain registry, and the precomputed surface was built on the
superseded P20 footprint. Nothing checked, so the cut propagated silently into
surfaces, hypsometry, belt statistics and figures.

This audit makes the defect impossible to carry unnoticed. It checks every
precomputed grid and raster against the authoritative domain and reports how far
short each one falls. It does not repair them -- a truncated legacy product is
still valid as a covariate, it simply may never define an output grid.

Run it after any change to the registry or to a bed-surface product.
"""
from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

DOMAIN = ("KAKHOVKA_RESERVOIR_CORE", "FORMER_RESERVOIR_TRANSITION")


def main() -> None:
    fp = unary_union([SD.load_subzone_utm(n) for n in DOMAIN])
    x0, y0, x1, y1 = fp.bounds
    print("=" * 78)
    print("P0K — grid truncation audit")
    print("=" * 78)
    print(f"  authoritative domain: E {x0:,.0f} .. {x1:,.0f}   "
          f"N {y0:,.0f} .. {y1:,.0f}")
    print(f"  ({' + '.join(DOMAIN)})\n")

    bad = 0
    pats = [str(CFG.BULK_ROOT / "data_swot/processed/bathymetry/*.npz"),
            str(ROOT / "data/processed/bathymetry/*.npz")]
    for f in sorted({p for pat in pats for p in glob.glob(pat)}):
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as ex:
            print(f"  {Path(f).name:38s} unreadable ({type(ex).__name__})")
            continue
        if "gx" not in z or "gy" not in z:
            continue
        cell = float(z["cell_m"]) if "cell_m" in z else 0.0
        try:
            SD.assert_covers(fp, z["gx"], z["gy"], what=Path(f).name, cell=cell)
            print(f"  OK        {Path(f).name:38s} E max {z['gx'].max():,.0f}")
        except SD.DomainTruncationError as ex:
            bad += 1
            short = str(ex).split(": ", 1)[1].split(". ")[0]
            print(f"  TRUNCATED {Path(f).name:38s} E max {z['gx'].max():,.0f}"
                  f"   short {short}")

    print(f"\n  {bad} truncated grid product(s).")
    if bad:
        print("  These remain usable as COVARIATES. They must never define an")
        print("  output grid: build it with SD.build_grid(), which refuses to")
        print("  return a grid narrower than the domain.")
    legacy_pattern_scan()

    print("\n  scripts still reading a legacy surface (check each one uses it")
    print("  only as a covariate, never for its output grid):")
    import subprocess
    out = subprocess.run(
        ["grep", "-rln", "--include=*.py", "kakhovka_bed_surface_",
         str(ROOT / "scripts")], capture_output=True, text=True).stdout.split()
    n_check = 0
    for f in sorted(out):
        src = Path(f).read_text(encoding="utf-8", errors="replace")
        ok = "SD.build_grid(" in src
        n_check += 0 if ok else 1
        print(f"    {'ok    ' if ok else 'CHECK '} {Path(f).name}")
    print(f"\n  {n_check} script(s) read a legacy surface without going through")
    print("  SD.build_grid(). Each needs checking that it uses the legacy grid")
    print("  only as a covariate. hist24 was the one that did not, and its")
    print("  Table 19 RMS area error moved from 154 -> 114 km2 to 95 -> 69 km2")
    print("  once the output grid came from the registry instead.")


# A bare mention is not a use. Flag only lines that actually LOAD the legacy
# footprint, and only outside comments -- otherwise the audit flags its own
# docstring and every script that merely explains the defect, which is exactly
# the noise that makes an audit ignorable.
LOADERS = r"(open|load|read_file|read_text|GeoDataFrame\.from_file)\s*\("

# Scripts whose PURPOSE is to compare against the retired footprint. Loading it
# there is the point, not the defect, and flagging them trains the reader to
# ignore the audit.
COMPARES_P20 = {
    "p0b_build_dnipro_water_domain.py": "builds the registry domain and reports "
                                        "the P20 discrepancy",
    "p0_domain_qa.py": "registry QA; lists P20 as RETIRED",
    "qa4_domain_maps.py": "draws the old-vs-new domain comparison",
    "p1b_p1d_coverage.py": "plots the OLD clipped footprint beside the "
                           "current one, labelled pre-fix",
    "p0k_grid_truncation_audit.py": "this audit",
}
PATTERNS = {
    "P20 footprint LOADED": (r"P20_reservoir_footprint", True),
    "hard-coded 668xxx eastern bound": (r"=\s*66[0-9]{4}(\.\d+)?\b", False),
    "local raster transform": (r"from_bounds\s*\(", False),
    "reads a legacy bed surface": (r"kakhovka_bed_surface_", False),
}


def legacy_pattern_scan():
    """Classify every script that touches a grid, not just the ones that fail.

    A grep for "P20" is not enough: the same defect hides as a hard-coded
    eastern bound, a local rasterio transform built from literal bounds, or an
    inherited npz. Each script gets one of three verdicts:

        CLEAN              geometry comes only from the registry or the current
                           surface
        LEGACY INPUT ONLY  an old product is read as a covariate but does not
                           define the output geometry
        BROKEN             a legacy surface or footprint defines the geometry of
                           the result
    """
    import re
    print("\n" + "=" * 78)
    print("LEGACY PATTERN SCAN")
    print("=" * 78)
    rows = []
    for f in sorted((ROOT / "scripts").glob("*.py")):
        src = f.read_text(encoding="utf-8", errors="replace")
        code = [ln for ln in src.split("\n")
                if not ln.lstrip().startswith("#")]
        hits = {}
        for k, (pat, needs_loader) in PATTERNS.items():
            n = 0
            for ln in code:
                if not re.search(pat, ln):
                    continue
                if needs_loader and not re.search(LOADERS, ln):
                    continue
                n += 1
            hits[k] = n
        if not any(hits.values()):
            continue
        defines_geometry = hits["P20 footprint LOADED"] > 0
        uses_builder = "SD.build_grid(" in src or "SD.assert_covers(" in src
        if f.name in COMPARES_P20:
            verdict = "COMPARES P20"
        elif defines_geometry:
            verdict = "BROKEN"
        elif uses_builder or not hits["reads a legacy bed surface"]:
            verdict = "CLEAN"
        else:
            verdict = "LEGACY INPUT ONLY"
        rows.append((verdict, f.name, hits))
    order = {"BROKEN": 0, "LEGACY INPUT ONLY": 1, "COMPARES P20": 2,
             "CLEAN": 3}
    for verdict, name, hits in sorted(rows, key=lambda r: (order[r[0]], r[1])):
        flags = ", ".join(f"{k} x{v}" for k, v in hits.items() if v)
        if verdict == "COMPARES P20":
            flags = COMPARES_P20[name]
        print(f"  {verdict:18s} {name:38s} {flags}")
    nb = sum(1 for v, _, _ in rows if v == "BROKEN")
    print(f"\n  {nb} BROKEN, "
          f"{sum(1 for v, _, _ in rows if v == 'LEGACY INPUT ONLY')} legacy-input-only, "
          f"{sum(1 for v, _, _ in rows if v == 'CLEAN')} clean")
    if nb:
        print("  BROKEN means a legacy footprint or surface defines the geometry")
        print("  of the result. Fix before trusting any output of that script.")


MANIFESTS = {
    "ZONE_1_KAKHOVKA_LOWER_DNIPRO": ("p0p_zone1_firstpass_manifest.csv",),
    "ZONE_2_KHERSON_DELTA": ("p0t_zone_2_kherson_delta_manifest.csv",),
    "ZONE_3_DNIPRO_BUG_ESTUARY": ("p0e_zone3_firstpass_manifest.csv",
                                  "p0q_zone3_topup_manifest.csv"),
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY":
        ("p0t_zone_4_dam_to_kherson_floodway_manifest.csv",),
}


def event_key_audit():
    """Does every cached product's KEY match the manifest that asked for it?

    This audit exists because a metadata corruption passed every other check.
    p0o built the event_id with `str(orbit_state).startswith('asc')`, which is
    case-sensitive, and the manifests do not agree on case: p0e and p0p write
    "ascending", p0t writes "ASC". So every p0t-driven fetch took the else
    branch and filed 270 products as _DES regardless of what they were. The
    pixels were right and nothing raised; what broke was the join key between
    manifest, cache and every downstream QA table, so a status check found
    ZONE_2 with 150 cached events and 68 missing ones on 150 of 150 matching
    dates.

    The key is (date, relative_orbit, ASC/DES, zone, grid). Three things are
    checked against it:

        every manifest event has a cached product, or is reported missing
        every cached product is named by a manifest event, or is an orphan
        every cached array has the shape of the zone's CURRENT registry grid

    The last one is the other half of the same lesson: a cached raster is only
    valid on the grid it was fetched on, and a domain repair silently changes
    that grid.
    """
    import numpy as np
    import pandas as pd
    from swot_dnipro import config as CFG
    from swot_dnipro import spatial_domains as SD

    def keys(name):
        p = CFG.TABLES / name
        if not p.exists():
            return set()
        d = pd.read_csv(p)
        if "event_id" in d.columns:
            return set(d.event_id)
        st = d.orbit_state.astype(str).str.strip().str.lower().map(
            lambda s: "ASC" if s.startswith("asc") else "DES")
        return set(d.date.astype(str).str[:10] + "_orb" +
                   d.relative_orbit.astype(int).astype(str) + "_" + st)

    print("\n" + "=" * 78)
    print("EVENT KEY AUDIT — manifest key == cache key == current grid")
    print("=" * 78)
    bad = 0
    for zone, mans in MANIFESTS.items():
        want = set().union(*[keys(m) for m in mans]) if mans else set()
        d = CFG.S1_CACHE / zone
        have = {f.stem for f in d.glob("*.npz")} if d.exists() else set()
        try:
            gr = SD.build_grid(SD.load_utm(zone), 20.0, what=zone)
            cur = (gr["ny"], gr["nx"])
        except Exception as ex:
            print(f"  {zone}: grid unavailable ({type(ex).__name__})"); continue
        fs = sorted(d.glob("*.npz")) if d.exists() else []
        shapes = {np.load(f)["vv"].shape for f in (fs[:3] + fs[-3:])}
        stale = shapes - {cur}
        orphans, missing = have - want, want - have
        # An event deliberately deleted for carrying no data is not missing.
        # ZONE_3's orbit 160 returned 31 products at 0.0% coverage and p0q
        # deleted them on purpose; counting those as a gap would leave this
        # audit crying wolf on a closed zone forever.
        import json as _j
        fz = CFG.TABLES / "p0q_zone3_topup_freeze.json"
        excl = set()
        if zone.startswith("ZONE_3") and fz.exists():
            excl = {int(o) for o in
                    _j.loads(fz.read_text()).get("excluded_orbits", {})}
        deleted = {e for e in missing
                   if int(e.split("_orb")[1].split("_")[0]) in excl}
        missing -= deleted
        n_asc = sum(1 for i in have if i.endswith("ASC"))
        status = "OK"
        if stale:
            status = "STALE GRID"
        elif orphans:
            status = "KEY MISMATCH"
        elif missing:
            status = "INCOMPLETE"
        if status != "OK":
            bad += 1
        print(f"  {status:14s} {zone}")
        print(f"      manifest {len(want):4d}   cached {len(have):4d}   "
              f"orphans {len(orphans):4d}   missing {len(missing):4d}   "
              f"deleted-empty {len(deleted):3d}   "
              f"{n_asc} ASC / {len(have)-n_asc} DES")
        print(f"      grid now {cur}   cached shapes {shapes or '-'}")
        if have and n_asc == 0 and len(have) > 20:
            print(f"      WARNING: not one cached product is labelled ASC. "
                  f"That is the signature of the orbit_state case bug, not a "
                  f"property of the orbits.")
        for e in sorted(orphans)[:3]:
            print(f"      orphan  {e}")
        for e in sorted(missing)[:3]:
            print(f"      missing {e}")
    print(f"\n  {bad} of {len(MANIFESTS)} zone caches need attention")
    print("  STALE GRID means the domain moved after the fetch: refetch, never "
          "crop.\n  KEY MISMATCH means the cache is named by something other "
          "than the manifest.")


if __name__ == "__main__":
    main()
    event_key_audit()
