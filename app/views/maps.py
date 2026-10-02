import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from lib import charts as K
from lib import config as C
from lib import data as D
from lib import maps as M
from lib import state as S

st.title("Study area & maps")
st.markdown(
    "Four analysis zones in flow order that **do not overlap**: **R** — the former Kakhovka Reservoir "
    "footprint; **F** — the lower Dnipro floodway from the dam to Kherson; **D** — the Kherson delta; "
    "**E** — the Dnipro–Buh estuary. The June 2023 Sentinel-1 flood envelope is an *event layer* inside "
    "F, not a zone boundary. ICESat-2 is drawn as **passes** (one line per beam of one overpass, split "
    "where the track leaves the water), not as points: ATL13 water-surface heights in the three samples of "
    "this paper (reservoir, Kherson, estuary), coloured by period, and the ATL08 terrain passes over the "
    "drained bed that the companion terrain work uses (not part of this paper; off by default). Hover a "
    "track for product, sample, date, RGT, beam and QC. Use the layer control (top right) to switch layers.")

S.init("f_maps_crossings", True)
show_x = st.toggle("Show SWOT–ICESat-2 crossings (V4)", key="f_maps_crossings")
st_folium(M.study_map(show_crossings=show_x), height=640, use_container_width=True, returned_objects=[])

# ---------------------------------------------------------------- Sentinel-1 flood by date --
st.subheader("Sentinel-1 flood dynamics below the dam, June 2023")
sm = D.s1_manifest()
dates = [l["date"] for l in sm["layers"]]
by_date = {l["date"]: l for l in sm["layers"]}
st.markdown(
    "Three kinds of layer, kept apart on purpose: **F** (and R, D, E) are the *permanent analysis domains* of the "
    "paper, fixed by the hydrography before any analysis; the **June 2023 Sentinel-1 masks** below are *event "
    "observations*, one map per acquisition date; the **union of the per-scene masks** on the study map above is "
    "a *display/reference layer* that inherits every false-water pixel of its scenes and is not a zone boundary. "
    "Each date is shown on its own, not as a union: **blue** is dark water that was *not* water on 1–2 June "
    "(new water), **grey-blue** is dark water on pre-breach water, and **light grey** is the part of the "
    "observable domain the sensor did not see on that date. Orbit 138 (6, 18 and 30 June) covers only the eastern "
    "half, so its numbers are not comparable with the full-coverage dates. **Not observed is not dry.** Dark "
    "water is low VV/VH backscatter: smooth sand, bare fields and wet soil can be dark (false water, visible as "
    "speckle on the terrace north of the river), and water under forest and reed is invisible to the sensor.")
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    S.init("f_s1_date", "2023-06-09" if "2023-06-09" in dates else dates[0], dates)
    s1_date = st.select_slider("Acquisition date", options=dates, key="f_s1_date",
                               format_func=lambda d: f"{d[8:]}.{d[5:7]} · {by_date[d]['orbit']}")
with c2:
    S.init("f_s1_basemap", list(M.BASEMAPS)[0], list(M.BASEMAPS))
    s1_base = st.selectbox("Basemap", list(M.BASEMAPS), key="f_s1_basemap")
with c3:
    S.init("f_s1_unobserved", True)
    s1_unobs = st.toggle("Show not-observed area", key="f_s1_unobserved")
with c4:
    S.init("f_s1_opacity", 0.8)
    s1_op = st.slider("Overlay opacity", 0.3, 1.0, step=0.05, key="f_s1_opacity")
lay = by_date[s1_date]
k1, k2, k3, k4 = st.columns(4)
k1.metric("New dark water, Dnipro corridor", f"{lay['corridor_new_water_km2']:,.0f} km²",
          help="p94: water that was not water on 1–2 June, inside the corridor and the date's footprint")
k2.metric("Same, common footprint of all 11 dates", f"{lay['corridor_new_water_common_footprint_km2']:,.0f} km²",
          help="comparable across dates: only pixels observed on every one of the 11 dates")
k3.metric("Corridor observed on this date", f"{100 * lay['corridor_coverage']:.0f} %",
          help="fraction of the S1 observable domain (union of the 11 footprints) seen on this date")
k4.metric("Orbit", lay["orbit"])
st_folium(M.s1_flood_map(s1_date, show_unobserved=s1_unobs, opacity=s1_op, basemap=s1_base),
          height=560, use_container_width=True, returned_objects=[])
st.markdown(" · ".join(
    f"<span style='display:inline-block;width:14px;height:14px;background:{sm['palette'][k]};"
    f"border:1px solid #999;vertical-align:middle'></span> {v}" for k, v in sm["legend"].items()),
    unsafe_allow_html=True)
st.plotly_chart(K.s1_flood_dynamics(selected=s1_date), width="stretch")
st.caption(
    f"Per-date Sentinel-1 dark-water masks (p0v/p0w M3, 20 m zone caches) and the p94 per-date table from the "
    f"companion repository *floodstate-eo* (commit {sm['source_commit']}"
    f"{', working tree' if sm.get('source_dirty') else ''}), rendered on a 0.001° grid; "
    f"areas are from the p94 table, not from the display image. The estuary (zone E) is summarised in p94 on "
    f"its own grid and is not drawn. Build: `app/prepare_app_s1_layers.py` · "
    f"[layer manifest]({C.gh(C.S1_MANIFEST)}) · [table]({C.gh(C.S1_DYNAMICS)})")
with st.expander("All dates and regions (p94 table)"):
    t = D.s1_dynamics()
    st.dataframe(t[["date", "orbit", "region_label", "region_km2", "valid_km2", "coverage", "water_km2",
                    "new_water_km2", "new_water_common_footprint_km2"]]
                 .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")), hide_index=True, width="stretch")

pm = D.icesat2_passes_manifest()
st.caption("ICESat-2 passes (date × RGT) in the layer: " + " · ".join(
    f"{k}: {v}" for k, v in pm["n_passes_date_x_rgt"].items()) +
    f". Built by `app/prepare_app_icesat2_passes.py` from the ATL13 segment tables of the companion "
    f"repository and the ATL08 terrain table; sources and hashes in "
    f"[icesat2_passes_manifest.json]({C.gh(C.APP_DATA / 'icesat2_passes_manifest.json')}).")

m = D.zone_manifest()
zt = pd.DataFrame([dict(zone=z["letter"], label=z["label"], registry_id=k, area_km2=z["area_km2"],
                        hash=z["geometry_hash_utm"]) for k, z in m["zones"].items()])
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Zones (frozen geometry)")
    st.dataframe(zt, hide_index=True, width="stretch")
    st.caption(f"Pairwise overlaps < {m['overlap_tolerance_km2']} km², so areas are additive "
               f"({m['zones_total_km2']:,.1f} km² in total). "
               f"[Manifest]({C.gh(C.ZONE_MANIFEST)}) · [geometry]({C.gh(C.ROOT / m['file'])})")
with c2:
    st.subheader("Observation geometry by zone")
    st.plotly_chart(K.zone_counts(), width="stretch")

st.subheader("Gauge records")
st.plotly_chart(K.station_inventory(), width="stretch")
st.caption("Five of the six reservoir series end on 31 December 2021; Rozumivka and Kherson continue "
           "through 2025. All records are daily values: one yearbook value per day in 2019, the daily mean of the "
           "08:00/20:00 term readings from 2020, and daily means of the hourly record at the 2023 sea-yearbook posts "
           f"(Oleksandrivka: term values). Source: [{C.INVENTORY.name}]({C.gh(C.INVENTORY)})")
st.dataframe(D.layer("stations.csv"), hide_index=True, width="stretch")
