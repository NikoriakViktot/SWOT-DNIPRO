import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from lib import charts as K
from lib import config as C
from lib import data as D
from lib import maps as M

st.title("Study area & maps")
st.markdown(
    "Four analysis zones in flow order that **do not overlap**: **R** — the former Kakhovka Reservoir "
    "footprint; **F** — the lower Dnipro floodway from the dam to Kherson; **D** — the Kherson delta; "
    "**E** — the Dnipro–Buh estuary. The June 2023 Sentinel-1 flood envelope is an *event layer* inside "
    "F, not a zone boundary. Use the layer control (top right) to switch layers.")

show_x = st.toggle("Show SWOT–ICESat-2 crossings (V4)", value=True)
st_folium(M.study_map(show_crossings=show_x), height=640, use_container_width=True, returned_objects=[])

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
           f"through 2025. Source: [{C.INVENTORY.name}]({C.gh(C.INVENTORY)})")
st.dataframe(D.layer("stations.csv"), hide_index=True, width="stretch")
