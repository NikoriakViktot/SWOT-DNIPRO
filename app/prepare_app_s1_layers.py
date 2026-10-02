#!/usr/bin/env python
"""Build the Sentinel-1 flood-dynamics layer the companion app shows per date.

The per-date S1 dark-water masks of June 2023 (p0v/p0w M3 per-scene masks,
20 m zone caches) live in the sibling repository `floodstate-eo`, which renders
them as classed PNGs for its own dashboard (workflow p98) and summarises them
per date and region in table p94 (`p94_flood_dynamics_s1.csv`). This script
copies that product into this repository's light app layer, so the app runs
from a plain clone:

    outputs/paper/app_data/s1/<date>.png      one classed PNG per date
                                              1 = new dark water (not water on 06-01/02)
                                              2 = dark water on pre-breach water
                                              3 = observable domain, not observed on this date
                                              0 = transparent (observed and not dark, or
                                                  outside the S1 observable domain)
    outputs/paper/app_data/s1/manifest.json   bounds, legend, palette, per-date orbit and
                                              p94 numbers, source files with sha256 and the
                                              floodstate-eo commit
    outputs/paper/app_data/s1_flood_dynamics.csv   p94 S1 rows (corridor, Inhulets valley,
                                              p42 floodplain, estuary)

The source PNGs are on a regular EPSG:4326 grid (0.001° × 0.00072°). Leaflet
stretches an image overlay linearly in Web Mercator, so the rows are resampled
here (nearest) to equal Mercator spacing; otherwise the overlay drifts by a
few hundred metres at mid-image. Column spacing is already uniform in Mercator.

NOT OBSERVED IS NOT DRY: class 3 is where the sensor did not look on that
date (orbit 138 covers only the eastern half). Dark water is low VV/VH
backscatter: smooth sand, bare fields and wet soil can be dark (false water),
and water under forest and reed is invisible to the sensor.

Source location: $FLOODSTATE_EO_ROOT, else <this repo>/../floodstate-eo.

    python app/prepare_app_s1_layers.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FE = Path(os.environ.get("FLOODSTATE_EO_ROOT", ROOT.parent / "floodstate-eo")).resolve()
FE_DATA = FE / "apps/dashboard/data"
FE_TABLES = FE / "case_studies/kakhovka_2023/tables"
OUT = ROOT / "outputs/paper/app_data"
OUT_S1 = OUT / "s1"

PALETTE = {1: "#1f5fa8", 2: "#b9c7d6", 3: "#dcd9d2"}
LEGEND = {1: "new dark water (not water on 1–2 June)",
          2: "dark water on pre-breach water",
          3: "not observed on this date (observable domain)"}
REGIONS = {"DNIPRO_CORRIDOR": "Dnipro corridor",
           "P42_FLOODPLAIN_DOMAIN": "terrain-eligible floodplain (p42)",
           "INHULETS_VALLEY_rect": "Inhulets valley",
           "ESTUARY_ZONE3_west_of_B2": "estuary (own grid, not on the map)"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _merc_y(lat_deg: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2))


def mercator_rows(a: np.ndarray, lat_s: float, lat_n: float) -> np.ndarray:
    """Resample rows (row 0 = north) to equal Web-Mercator spacing, nearest neighbour."""
    ny = a.shape[0]
    y_n, y_s = _merc_y(lat_n), _merc_y(lat_s)
    # centre of each output row in Mercator y, then its latitude, then the source row
    yc = y_n - (np.arange(ny) + 0.5) * (y_n - y_s) / ny
    lat = np.degrees(2 * np.arctan(np.exp(yc)) - np.pi / 2)
    src = np.clip(np.floor((lat_n - lat) / (lat_n - lat_s) * ny).astype(int), 0, ny - 1)
    return a[src]


def _hexrgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def write_png(arr: np.ndarray, path: Path) -> None:
    img = Image.fromarray(arr.astype("u1"), "P")
    pal = [0, 0, 0] * 256
    for k, h in PALETTE.items():
        pal[3 * k:3 * k + 3] = list(_hexrgb(h))
    img.putpalette(pal)
    img.save(path, optimize=True, transparency=0)


def main() -> None:
    if not FE_DATA.exists():
        sys.exit(f"floodstate-eo dashboard data not found at {FE_DATA} (set FLOODSTATE_EO_ROOT)")
    fe_man = json.loads((FE_DATA / "manifest.json").read_text())
    by_id = {l["id"]: l for l in fe_man["layers"]}
    dates = sorted(l["id"].replace("s1_new_", "") for l in fe_man["layers"] if l["group"] == "s1_daily")
    if not dates:
        sys.exit("no s1_daily layers in the floodstate-eo manifest")
    bounds = by_id[f"s1_new_{dates[0]}"]["bounds"]            # [[lat_s, lon_w], [lat_n, lon_e]]
    try:
        commit = subprocess.check_output(["git", "-C", str(FE), "rev-parse", "--short", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(FE), "status", "--porcelain",
                                              "apps/dashboard/data/s1"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", False

    # the S1 observable domain: union of every date's valid footprint
    foot, new, src = {}, {}, {}
    for d in dates:
        for kind, store in (("new", new), ("footprint", foot)):
            l = by_id[f"s1_{kind}_{d}"]
            p = FE_DATA / l["file"]
            h = sha(p)
            if h != l["sha256"]:
                sys.exit(f"{p} does not match the floodstate-eo manifest sha256")
            assert l["bounds"] == bounds, (d, kind, l["bounds"])
            store[d] = np.array(Image.open(p).convert("P"), dtype="u1")
            src[f"{kind}/{d}"] = dict(file=str(p.relative_to(FE)), sha256=h)
    observable = np.zeros_like(next(iter(foot.values())), dtype=bool)
    for a in foot.values():
        observable |= a == 1

    p94 = pd.read_csv(FE_TABLES / "p94_flood_dynamics_s1.csv")
    est = pd.read_csv(FE_TABLES / "p94_flood_dynamics_estuary_s1.csv")
    est["orbit"] = est.date.map(p94.drop_duplicates("date").set_index("date").orbit)
    dyn = pd.concat([p94, est], ignore_index=True, sort=False)
    dyn = dyn[dyn.sensor == "S1"].copy()
    dyn["region_label"] = dyn.region.map(REGIONS).fillna(dyn.region)
    dyn.to_csv(OUT / "s1_flood_dynamics.csv", index=False)
    src["p94_table"] = dict(file="case_studies/kakhovka_2023/tables/p94_flood_dynamics_s1.csv",
                            sha256=sha(FE_TABLES / "p94_flood_dynamics_s1.csv"))
    src["p94_table_estuary"] = dict(file="case_studies/kakhovka_2023/tables/p94_flood_dynamics_estuary_s1.csv",
                                    sha256=sha(FE_TABLES / "p94_flood_dynamics_estuary_s1.csv"))
    corridor = p94[p94.region == "DNIPRO_CORRIDOR"].set_index("date")

    OUT_S1.mkdir(parents=True, exist_ok=True)
    for old in OUT_S1.glob("*.png"):
        old.unlink()
    lat_s, lat_n = bounds[0][0], bounds[1][0]
    layers = []
    for d in dates:
        cls = np.zeros(observable.shape, "u1")
        cls[observable & (foot[d] != 1)] = 3
        cls[new[d] == 2] = 2
        cls[new[d] == 1] = 1
        p = OUT_S1 / f"{d}.png"
        write_png(mercator_rows(cls, lat_s, lat_n), p)
        r = corridor.loc[d]
        layers.append(dict(date=d, file=f"s1/{d}.png", orbit=r.orbit, bytes=p.stat().st_size, sha256=sha(p),
                           corridor_new_water_km2=float(r.new_water_km2),
                           corridor_new_water_common_footprint_km2=float(r.new_water_common_footprint_km2),
                           corridor_coverage=float(r.coverage), corridor_water_km2=float(r.water_km2)))
        print(f"  {d} {r.orbit:<11} new {r.new_water_km2:6.1f} km²  coverage {r.coverage:.0%}  {p.stat().st_size / 1e3:5.1f} kB")

    man = dict(
        built_by="app/prepare_app_s1_layers.py",
        source_repository="floodstate-eo", source_commit=commit, source_dirty=dirty,
        source_workflows=["case_studies/kakhovka_2023/workflows/paper/p98_dashboard_layers.py (classed PNGs)",
                          "case_studies/kakhovka_2023/workflows/m6/p94_flood_dynamics.py (per-date table)"],
        product="p0v/p0w M3 per-scene Sentinel-1 dark-water masks (20 m zone caches), B1 ∪ B2 frames; "
                "new = dark water that was not water on 1–2 June 2023",
        caveats=["not observed is not dry: class 3 is where the sensor did not look on that date",
                 "dark water = low VV/VH backscatter: smooth sand, bare fields and wet soil can be dark (false water)",
                 "water under forest and reed is invisible to the sensor",
                 "the estuary (zone E) is on its own grid in p94 and is not drawn"],
        grid=dict(crs="EPSG:4326", bounds=bounds, nx=int(observable.shape[1]), ny=int(observable.shape[0]),
                  rows="resampled to equal Web-Mercator spacing (nearest) for an exact Leaflet overlay"),
        legend={str(k): v for k, v in LEGEND.items()}, palette={str(k): v for k, v in PALETTE.items()},
        regions=REGIONS, layers=layers, sources=src)
    (OUT_S1 / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    tot = sum(l["bytes"] for l in layers)
    print(f"  {len(layers)} dates, {tot / 1e3:.0f} kB, floodstate-eo {commit}{' (dirty)' if dirty else ''}")


if __name__ == "__main__":
    main()
