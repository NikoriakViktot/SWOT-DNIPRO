#!/usr/bin/env python
"""P0Y — the four analysis zones on an OpenStreetMap basemap, from the registry.

The first version of this map was written by hand as a one-off HTML file. That
was a mistake of the kind this repository has a documented history of: a
derived artefact with no script behind it cannot be regenerated when the
geometry changes, and the geometry changed twice in one session (ZONE_2 lost
two polygons that were in the wrong river, ZONE_4 was rebuilt with a 10 km
buffer). So this is a generator, and every layer is read from the registry.

IT IS DELIBERATELY A BASEMAP VIEW. The defects that this map exposed - a
"Kherson delta" polygon 50 km up the Southern Bug, an "Inhulets" subzone
containing Hola Prystan on the opposite bank - were invisible in every numeric
check, because an area in km2 cannot tell you which river a polygon is in. Named
towns and rivers underneath the geometry can.

Everything is inlined: the CSP that serves these pages allows no external tiles
from a published artefact, so the file is written to disk and opened locally,
where OSM tiles load normally.

Outputs
-------
outputs/maps/zones_openstreetmap.html
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD

OUT = ROOT / "outputs/maps/zones_openstreetmap.html"
SIMPLIFY_M = 60.0

ZONE_STYLE = {
    "ZONE_1_KAKHOVKA_LOWER_DNIPRO": ("#b07d27", "Зона 1 · Каховка + нижній Дніпро"),
    "ZONE_2_KHERSON_DELTA":         ("#3f7d4e", "Зона 2 · Херсонська дельта"),
    "ZONE_3_DNIPRO_BUG_ESTUARY":    ("#1d6e8c", "Зона 3 · Дніпро-Бузький лиман"),
    "ZONE_4_DAM_TO_KHERSON_FLOODWAY": ("#a8306a", "Зона 4 · заплава ГЕС→Херсон"),
}
SUBZONE_LABEL = {
    "KAKHOVKA_RESERVOIR_CORE": "підзона · ядро водосховища",
    "FORMER_RESERVOIR_TRANSITION": "підзона · перехідна смуга",
    "KAKHOVKA_DAM_TO_KHERSON": "підзона · русло ГЕС→Херсон",
    "INHULETS_TRIBUTARY": "підзона · Інгулець",
}


def feat(geom, props):
    g = gpd.GeoSeries([geom.simplify(SIMPLIFY_M)],
                      crs=CFG.CRS_METRIC).to_crs(4326).iloc[0]
    return dict(type="Feature", properties=props,
                geometry=json.loads(gpd.GeoSeries([g]).to_json())
                         ["features"][0]["geometry"])


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    print("=" * 78)
    print("P0Y — zones on OpenStreetMap")
    print("=" * 78)
    print(f"  git {commit}; simplify {SIMPLIFY_M:.0f} m")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    groups = {"ЗОНИ": [], "ПІДЗОНИ": [], "ВОДА": [], "ЗАТОПЛЕННЯ": []}

    for name, (colour, label) in ZONE_STYLE.items():
        try:
            g = SD.load_utm(name)
        except Exception as ex:
            print(f"  {name}: {type(ex).__name__} -- skipped"); continue
        km2 = g.area / 1e6
        print(f"  {label:34s} {km2:9,.0f} km2")
        groups["ЗОНИ"].append(feat(g, dict(label=label, colour=colour,
                                           km2=round(km2), fill=0.10, on=True)))

    for name, label in SUBZONE_LABEL.items():
        try:
            g = SD.load_subzone_utm(name)
        except Exception as ex:
            print(f"  {name}: {type(ex).__name__} -- skipped"); continue
        km2 = g.area / 1e6
        print(f"  {label:34s} {km2:9,.0f} km2")
        groups["ПІДЗОНИ"].append(feat(g, dict(label=label, colour="#7a4fa3",
                                              km2=round(km2), fill=0.18,
                                              on=False)))

    w = SD.load_utm("dnipro_water_domain")
    print(f"  {'водна область (реєстр)':34s} {w.area/1e6:9,.0f} km2")
    groups["ВОДА"].append(feat(w, dict(label="водна область (реєстр)",
                                       colour="#1f6fb0",
                                       km2=round(w.area / 1e6), fill=0.45,
                                       on=False)))

    env = [("data/processed/domains/zone_2_kherson_delta_flood_envelope_utm.geojson",
            "flood_envelope", "обвідна затоплення · дельта", "#c1402a"),
           ("data/processed/domains/zone4_flood_composite_utm.geojson",
            "ZONE4_FLOOD_ENVELOPE_CANONICAL",
            "обвідна затоплення · заплава (канонічна)", "#c1402a"),
           ("data/processed/domains/zone4_flood_composite_utm.geojson",
            "ZONE4_FLOOD_ENVELOPE_ALL_OBSERVATIONS",
            "обвідна затоплення · заплава (всі спостереження)", "#e08a3c")]
    for rel, layer, label, colour in env:
        p = ROOT / rel
        if not p.exists():
            print(f"  {label}: {rel} missing -- skipped"); continue
        E = gpd.read_file(p).set_index("layer")
        if layer not in E.index:
            print(f"  {label}: layer {layer} absent -- skipped"); continue
        g = E.loc[layer, "geometry"]
        print(f"  {label:34s} {g.area/1e6:9,.0f} km2")
        groups["ЗАТОПЛЕННЯ"].append(feat(g, dict(label=label, colour=colour,
                                                 km2=round(g.area / 1e6),
                                                 fill=0.30, on=False)))

    data = {k: dict(type="FeatureCollection", features=v)
            for k, v in groups.items() if v}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)) \
                   .replace("__COMMIT__", commit)
    OUT.write_text(html, encoding="utf-8")
    print(f"\n-> {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print("  open it locally; a published artefact cannot load OSM tiles.")


TEMPLATE = """<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Зони аналізу</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body,#map{height:100%;margin:0}
 #panel{position:absolute;top:12px;right:12px;z-index:1000;background:#fff;
   border-radius:8px;box-shadow:0 2px 14px rgba(0,0,0,.28);padding:14px 16px;
   font:13px/1.45 system-ui,sans-serif;max-width:340px;max-height:88vh;
   overflow:auto}
 #panel h3{margin:0 0 10px;font-size:15px}
 .grp{margin:12px 0 4px;font-size:11px;letter-spacing:.08em;color:#8a8378}
 .row{display:flex;align-items:center;gap:8px;padding:3px 0}
 .row label{flex:1;cursor:pointer}
 .sw{width:26px;height:12px;border-radius:3px;flex:none}
 .km{color:#7b746a;font-variant-numeric:tabular-nums}
 .note{margin-top:14px;padding-top:10px;border-top:1px solid #e7e2d8;
   color:#6d675e;font-size:12px}
</style></head><body>
<div id="map"></div><div id="panel"><h3>Зони аналізу</h3><div id="legend"></div>
<div class="note">Обвідні затоплення — спостережені Sentinel-1, червень 2023.
Зона 4 перекриває Зону 1 і Зону 2: спільна межа — єдина незалежна перевірка
класифікаторів. Не спостережене ≠ сухе: див.
<code>zone4_observation_support.tif</code>.<br>git __COMMIT__</div></div>
<script>
const DATA = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:18, attribution:'© OpenStreetMap'}).addTo(map);
const legend = document.getElementById('legend');
let bounds = null;
for (const [grp, fc] of Object.entries(DATA)) {
  const h = document.createElement('div');
  h.className = 'grp'; h.textContent = grp; legend.appendChild(h);
  for (const f of fc.features) {
    const p = f.properties;
    const lyr = L.geoJSON(f, {style:{color:p.colour, weight:1.6,
      fillColor:p.colour, fillOpacity:p.fill}});
    lyr.bindTooltip(p.label + ' — ' + p.km2.toLocaleString('uk') + ' км²');
    const row = document.createElement('div'); row.className = 'row';
    const cb = document.createElement('input'); cb.type = 'checkbox';
    cb.checked = !!p.on; cb.id = 'c' + Math.random().toString(36).slice(2);
    const sw = document.createElement('span'); sw.className = 'sw';
    sw.style.background = p.colour;
    const lb = document.createElement('label'); lb.htmlFor = cb.id;
    lb.textContent = p.label;
    const km = document.createElement('span'); km.className = 'km';
    km.textContent = p.km2.toLocaleString('uk');
    row.append(cb, sw, lb, km); legend.appendChild(row);
    cb.onchange = () => cb.checked ? lyr.addTo(map) : map.removeLayer(lyr);
    if (p.on) lyr.addTo(map);
    bounds = bounds ? bounds.extend(lyr.getBounds()) : lyr.getBounds();
  }
}
map.fitBounds(bounds, {padding:[20,20]});
</script></body></html>
"""


if __name__ == "__main__":
    main()
