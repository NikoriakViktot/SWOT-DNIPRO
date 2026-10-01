"""The study-area map: every layer from outputs/paper/app_data, switchable."""
from __future__ import annotations

import folium
import numpy as np
from branca.colormap import LinearColormap

from . import config as C
from . import data as D

PERIOD_COLOURS = {"PRE_BREACH": "#f5c518", "BREACH_DRAWDOWN": "#f97306", "POST_BREACH": "#d7191c"}   # as Figure 1: yellow before, red after


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
    folium.GeoJson(f.geometry.iloc[0].__geo_interface__,
                   name="Union of the per-scene June 2023 S1 masks within F (display/reference layer, not a zone; "
                        "per-date event observations below)",
                   show=False, tooltip=f"{f.layer.iloc[0]}<br>{f.area_km2.iloc[0]:,.0f} km²",
                   style_function=lambda _: dict(color="#c1402a", weight=0, fillColor="#c1402a",
                                                 fillOpacity=0.35)).add_to(m)

    # ICESat-2 passes as ground tracks (one line per beam of one overpass), not points
    ps = D.layer("icesat2_passes.geojson")
    ps["date"] = ps.date.astype(str)
    fields = ["product", "sample", "date", "rgt", "beam", "period", "qc_pass", "n_segments"]
    aliases = ["product", "sample", "date", "RGT", "beam", "period", "QC", "segments"]
    for per in ("PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH"):
        g = ps[(ps["product"] == "ATL13") & (ps.period == per)]
        if g.empty:
            continue
        n_pass = g[["date", "rgt"]].drop_duplicates().shape[0]
        folium.GeoJson(
            g.to_json(), name=f"ICESat-2 ATL13 passes — {per.lower().replace('_', ' ')} "
                              f"({n_pass} passes, {len(g)} beams)", show=per != "BREACH_DRAWDOWN",
            style_function=lambda _, c=PERIOD_COLOURS[per]: dict(color=c, weight=1.6, opacity=0.85),
            highlight_function=lambda _: dict(weight=4, opacity=1),
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases)).add_to(m)
    g = ps[ps["product"] == "ATL08"]
    if not g.empty:
        n_pass = g[["date", "rgt"]].drop_duplicates().shape[0]
        folium.GeoJson(
            g.to_json(), name=f"ICESat-2 ATL08 passes — drained-bed terrain, k10 QC, companion work "
                              f"({n_pass} passes, {len(g)} beams)", show=False,
            style_function=lambda _: dict(color="#8e44ad", weight=1.6, opacity=0.85, dashArray="4 3"),
            highlight_function=lambda _: dict(weight=4, opacity=1),
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases)).add_to(m)

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


BASEMAPS = {
    "Satellite (Esri World Imagery)": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
    # CARTO basemaps answer without an API key with a 2 kB "API KEY REQUIRED" stub since 2026 (checked
    # 2026-10-01: light_all and dark_all tiles both return the same placeholder), so the grey and dark
    # canvases come from Esri, which needs no key.
    "Gray (Esri Light Gray Canvas)": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS User Community"),
    "Dark (Esri Dark Gray Canvas)": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS User Community"),
    "OpenStreetMap": dict(tiles="OpenStreetMap", attr=None),
}


def s1_flood_map(date: str, show_unobserved: bool = True, opacity: float = 0.8,
                 basemap: str = "Satellite (Esri World Imagery)") -> folium.Map:
    """One Sentinel-1 acquisition date of June 2023: new dark water, dark water on
    pre-breach water and the part of the observable domain the sensor did not see
    that day, as a classed image overlay over the zones below the dam."""
    man = D.s1_manifest()
    lay = {l["date"]: l for l in man["layers"]}[date]
    b = man["grid"]["bounds"]
    m = folium.Map(location=[(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2], zoom_start=9,
                   tiles=None, control_scale=True)
    bm = BASEMAPS[basemap]
    folium.TileLayer(tiles=bm["tiles"], attr=bm["attr"], name=basemap, control=False, max_zoom=18).add_to(m)

    png = C.S1_DIR / f"{date}.png"
    if show_unobserved:
        folium.raster_layers.ImageOverlay(str(png), bounds=b, opacity=opacity, interactive=False,
                                          cross_origin=False, zindex=5,
                                          name=f"Sentinel-1 dark water {date} — event observation").add_to(m)
    else:                                   # drop class 3 at render time: no second data file
        import numpy as np
        from PIL import Image
        a = np.array(Image.open(png).convert("P"))
        rgba = np.zeros(a.shape + (4,), "u1")
        for k in (1, 2):
            r, g, bb = (int(man["palette"][str(k)][i:i + 2], 16) for i in (1, 3, 5))
            rgba[a == k] = (r, g, bb, 255)
        folium.raster_layers.ImageOverlay(rgba, bounds=b, opacity=opacity, interactive=False,
                                          cross_origin=False, zindex=5, mercator_project=False,
                                          pixelated=False, name=f"Sentinel-1 dark water {date} — event observation").add_to(m)

    z = D.layer("zones.geojson")
    fz = folium.FeatureGroup(name="Analysis zones R / F / D / E — permanent domains (outline)", show=True)
    for _, r in z.iterrows():
        c = C.ZONE_COLOURS[r.letter]
        folium.GeoJson(r.geometry.__geo_interface__,
                       style_function=lambda _, c=c: dict(color=c, weight=2, fill=False),
                       tooltip=f"<b>{r.letter} — {r.label}</b>").add_to(fz)
    fz.add_to(m)
    st_ = D.layer("stations.csv")
    fg = folium.FeatureGroup(name="Gauges and 2023 yearbook posts", show=True)
    for _, r in st_[st_.zone.isin(["F", "D", "E"])].iterrows():
        folium.CircleMarker([r.lat, r.lon], radius=6, weight=2, color="#222", fill=True,
                            fill_color="#b5651d" if "gauge" in r.kind else "#ffffff", fill_opacity=0.95,
                            tooltip=f"<b>{r['name']}</b> ({r.kind})").add_to(fg)
    fg.add_to(m)
    lon, lat = D.app_manifest()["dam"]
    folium.Marker([lat, lon], tooltip="Kakhovka dam (breached 6 June 2023)",
                  icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)
    m.fit_bounds(b)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    return m
