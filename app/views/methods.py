import streamlit as st

from lib import config as C
from lib import data as D

st.title("Data & methods")

st.header("Data")
st.markdown("""
- **Gauges** — six Kakhovka reservoir gauges (zero 12.000 m BS-77) and Kherson 80805; 2023 UkrHMI
  yearbooks for the posts below the dam. Carried to **EVRF2019** by the official EPSG:9902 operation at each
  station's own coordinates. **Every gauge value is a daily value, not an instantaneous reading at the
  satellite epoch.** The averaging rule differs by source: the 2019 river yearbooks give one daily value;
  from 2020 the reservoir gauges and Kherson report the two term readings at 08:00 and 20:00 Kyiv time, and
  the series used here is their **daily mean** (V1 alone interpolates between the two bracketing term
  readings when both exist); the 2023 sea-and-estuary yearbook prints **daily means of the hourly record**
  (Oleksandrivka: term values), so the sea and river yearbooks differ at Kherson by a few centimetres on some
  days through the averaging rule, not the datum.
- **Wind** — no station wind record covers the study period. The repository holds ERA5 hourly 10 m wind and
  pressure over the Dnipro–Buh liman (Open-Meteo archive, 20 cells, 2019–2023; catalogue p0e). On
  5 April 2023 the along-liman wind turned from −4 m/s (away from Kherson) to +2 m/s (toward Kherson) between
  the ICESat-2 and SWOT epochs, so wind setup predicts a rise of order +5 cm, the same sign as the gauge trend
  and opposite to the observed −18 cm; the anomaly stays unexplained (`scripts/ms8_wind_setup_20230405.py`).
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
  date against the daily gauge value (daily mean of the 08:00/20:00 terms, or the yearbook daily value); the
  independent unit is the beam transect (V1), the SWOT pass (V3, V6) or the overpass (V6 ICESat-2).
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
