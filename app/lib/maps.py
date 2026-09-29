"""The study-area map: every layer from outputs/paper/app_data, switchable."""
from __future__ import annotations

import folium
import numpy as np
from branca.colormap import LinearColormap

from . import config as C
from . import data as D

PERIOD_COLOURS = {"PRE_BREACH": "#1f6f8b", "BREACH_DRAWDOWN": "#e8a33a", "POST_BREACH": "#2e7d32"}


def study_map(show_crossings: bool = True) -> folium.Map:
    z = D.layer("zones.geojson")
    m = folium.Map(location=[46.95, 33.4], zoom_start=8, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite (Esri)", show=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)

    fz = folium.FeatureGroup(name="Analysis zones R / F / D / E", show=True)
    for _, r in z.iterrows():
        c = C.ZONE_COLOURS[r.letter]
        folium.GeoJson(r.geometry.__geo_interface__,
                       style_function=lambda _, c=c: dict(color=c, weight=2, fillColor=c, fillOpacity=0.18),
                       tooltip=f"<b>{r.letter} — {r.label}</b><br>{r.domain}<br>{r.area_km2:,.1f} km²").add_to(fz)
    fz.add_to(m)

    w = D.layer("water.geojson")
    folium.GeoJson(w.geometry.iloc[0].__geo_interface__, name="Pre-breach water below the dam",
                   style_function=lambda _: dict(color="#3b7fb0", weight=0, fillColor="#3b7fb0",
                                                 fillOpacity=0.55)).add_to(m)
    f = D.layer("flood_event.geojson")
    folium.GeoJson(f.geometry.iloc[0].__geo_interface__, name="June 2023 flood envelope in F (event layer)",
                   show=False, tooltip=f"{f.layer.iloc[0]}<br>{f.area_km2.iloc[0]:,.0f} km²",
                   style_function=lambda _: dict(color="#c1402a", weight=0, fillColor="#c1402a",
                                                 fillOpacity=0.35)).add_to(m)

    t = D.layer("atl13_transects.csv")
    for per, g in t.groupby("period"):
        fg = folium.FeatureGroup(name=f"ICESat-2 ATL13 transects — {per.lower().replace('_', ' ')} "
                                      f"({len(g)})", show=per != "BREACH_DRAWDOWN")
        for _, r in g.iterrows():
            folium.CircleMarker([r.lat_mean, r.lon_mean], radius=2, weight=0, fill=True,
                                fill_opacity=0.7, fill_color=PERIOD_COLOURS.get(per, "grey"),
                                tooltip=f"{r.date} · rgt {r.rgt} {r.beam} · {r.chain_km:.0f} km").add_to(fg)
        fg.add_to(m)

    folium.GeoJson(D.layer("swot_calorbit.geojson").geometry.iloc[0].__geo_interface__,
                   name="SWOT calibration-orbit reach (daily, June 2023)",
                   style_function=lambda _: dict(color="#c0392b", weight=3, dashArray="6 6")).add_to(m)
    sp = D.layer("supports.geojson")
    fs = folium.FeatureGroup(name="Closure supports (Rozumivka V3, Kherson V6)", show=True)
    for _, r in sp.iterrows():
        folium.GeoJson(r.geometry.__geo_interface__, tooltip=f"{r.station}: {r.support}, {r.radius_km:g} km",
                       style_function=lambda _: dict(color="#333", weight=1, dashArray="3 3",
                                                     fillOpacity=0)).add_to(fs)
    fs.add_to(m)

    st_ = D.layer("stations.csv")
    fg = folium.FeatureGroup(name="Gauges and 2023 yearbook posts", show=True)
    for _, r in st_.iterrows():
        key = r["name"] in ("Rozumivka", "Kherson")
        span = (f"<br>record {r['first']} → {r['last']}" if isinstance(r.get("first"), str) else "")
        folium.CircleMarker([r.lat, r.lon], radius=9 if key else 6, weight=2,
                            color="#000" if key else "#7a4a14",
                            fill=True, fill_color="#b5651d" if "gauge" in r.kind else "#ffffff",
                            fill_opacity=0.95,
                            tooltip=f"<b>{r['name']}</b> ({r.kind})<br>zone {r.zone}{span}").add_to(fg)
    fg.add_to(m)
    lon, lat = D.app_manifest()["dam"]          # from swot_dnipro.config.KAKHOVKA_DAM
    folium.Marker([lat, lon], tooltip="Kakhovka dam (breached 6 June 2023)",
                  icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)

    if show_crossings:
        cr = D.layer("crossings.csv")
        cr = cr[cr.included_primary.astype(str).str.lower().eq("true")]
        cmap = LinearColormap(["#2166ac", "#f7f7f7", "#b2182b"], vmin=-0.3, vmax=0.3,
                              caption="SWOT − ICESat-2, m (crossings ≤ 24 h, clipped ±0.3)")
        fc = folium.FeatureGroup(name=f"SWOT–ICESat-2 crossings ≤ 24 h ({len(cr)})", show=True)
        for _, r in cr.iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=7, weight=1, color="#222", fill=True,
                                fill_color=cmap(float(np.clip(r.closure_residual_m, -0.3, 0.3))),
                                fill_opacity=0.95,
                                tooltip=f"{r.date} · zone {r.zone}<br>Δ = {100 * r.closure_residual_m:+.1f} cm"
                                        f"<br>{r.n_nodes} nodes · Δt {r.dt_hours:+.1f} h").add_to(fc)
        fc.add_to(m)
        cmap.add_to(m)
    b = z.total_bounds                      # frame all four zones, R included
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    return m
