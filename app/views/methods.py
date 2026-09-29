import streamlit as st

from lib import config as C
from lib import data as D

st.title("Data & methods")

st.header("Data")
st.markdown("""
- **Gauges** — six Kakhovka reservoir gauges (zero 12.000 m BS-77) and Kherson 80805, daily; 2023 UkrHMI
  yearbooks for the posts below the dam. Carried to **EVRF2019** by the official EPSG:9902 operation at each
  station's own coordinates.
- **ICESat-2 ATL13** Release 007, `ht_water_surf` (ellipsoidal, tide-free) + the ATL03 permanent-tide term
  → mean-tide crust.
- **SWOT** L2 HR RiverSP / LakeSP (v2.0) and PIXC; RiverSP `wse` + `geoid_hght` → ellipsoidal height.
- **EGG2015** quasigeoid (1′ × 1′) for both satellite branches.
- **Zones** R/F/D/E from the frozen registry geometry; Sentinel-1/2 masks for the event layer and backgrounds.
""")

st.header("Matching rules")
v4 = dict(claim_id="V4_SWOT_ICESAT_DIRECT")
rows = []
for lab, var in (("≤ 24 h (primary)", None), ("1–3 d", "timing sensitivity 1-3 d"),
                 ("3–10 d", "timing sensitivity 3-10 d")):
    rows.append(f"| {lab} | {int(D.val(statistic='n', variant=var, **v4))} | "
                f"{D.cm(D.val(statistic='median_m', variant=var, **v4))} | "
                f"{D.cm(D.val(statistic='nmad_m', variant=var, **v4), False)} |")
st.markdown("""
- **Crossing (V4):** an ATL13 segment within 200 m of a good RiverSP node (`node_q ≤ 1`, dark fraction < 0.5);
  node differences reduced by the median inside **one ICESat-2 overpass × one SWOT pass**, the independent unit.
- **Closures (V1, V3, V6):** c = H_gauge(EVRF2019) − H_satellite(EGG2015, mean-tide crust), same calendar
  date; the independent unit is the beam transect (V1), the SWOT pass (V3, V6) or the overpass (V6 ICESat-2).
- **Co-variability (V2):** within-station anomalies, so no closure constant can create the association.
- **Time windows** — the spread grows with separation while the median stays near zero:

| window | crossings | median | NMAD |
|---|---|---|---|
""" + "\n".join(rows))

st.header("EGG2015: bilinear, not nearest cell")
sl = D.slope_sampling().set_index("statistic").value
fr = D.stat(claim_id="V4_SWOT_ICESAT_DIRECT", statistic="frame_change_median_abs_m",
            variant="SWOT geoid at the node for both sensors (ellipsoidal difference)")
st.markdown(f"""
On the 1′ EGG2015 grid neighbouring cells differ by about 1.6 cm (median over the area). Sampling the nearest
cell puts artificial centimetre steps between two points ~200 m apart — exactly the geometry of a crossing.
The validation chain therefore interpolates bilinearly. With it, reducing both sensors to EGG2015 instead of
SWOT's EGM2008 changes the crossing differences by a median of **{1000 * fr.value:.1f} mm**, as the algebra requires.

The headline slopes were already built on bilinear heights (reproduced to 0.0 mm). Sampling the nearest cell
instead changes the per-overpass slopes by a median of {float(sl['pre_median_abs_dS_cm_km']):.3f} cm/km before
and {float(sl['post_median_abs_dS_cm_km']):.3f} cm/km after the breach; the contrast moves from
{float(sl['contrast_bilinear']):+.3f} to {float(sl['contrast_nearest']):+.3f} cm/km and
{int(float(sl['post_positive_nearest']))}/{int(float(sl['n_post']))} post-breach slopes stay positive.
""")

st.header("Why the validation figures were recomputed")
st.markdown(f"""
The v5 supplement stated SWOT–ICESat-2 statistics (32 crossings; 21 "inside the reservoir footprint";
68 Rozumivka passes) whose code and evidence table no longer existed. A search over 1 620 rule variants
(`scripts/ms6c_crossing_provenance.py`) found no rule set that reproduces them together; the 21 appears only
under rules that break the 32 and only inside the wider ZONE_1 corridor. Those numbers are therefore not
used. The validation was rebuilt as six separate paths from the primary products, and every figure that
carried a validation number was redrawn from the resulting tables. The manuscript is checked against them by
[`ms7d_manuscript_audit.py`]({C.gh(C.ROOT / 'scripts/ms7d_manuscript_audit.py')}).
""")
