"""Interactive figures. Every mark is a row of ms7_evidence.csv or ms7_summary.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import config as C
from . import data as D

BREACH = pd.Timestamp("2023-06-06")
BLUE, ORANGE, GREEN, GREY = "#2a73c9", "#e8643a", "#1aa37a", "#8a94a3"


def _theil_line(fig, x, y, row=None, col=None, name="Theil–Sen", color="black"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return
    b, a, *_ = stats.theilslopes(y, x)
    xs = np.array([x.min(), x.max()])
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=color, width=1.5),
                             name=f"{name} slope {b:.2f}"), **kw)


# ------------------------------------------------------------------- V1 --
def v1_closure_by_station() -> go.Figure:
    e = D.evidence()
    e = e[(e.claim_id == "V1_ATL13_GAUGE_CLOSURE") & e.included_primary]
    s = D.summary()
    s = s[(s.claim_id == "V1_ATL13_GAUGE_CLOSURE") & (s.statistic == "station_median_c_m")]
    order = s.sort_values("value").station.tolist()
    fig = go.Figure()
    fig.add_trace(go.Box(x=100 * e.closure_residual_m, y=e.station, orientation="h", boxpoints="all",
                         jitter=0.4, pointpos=0, marker=dict(color=BLUE, size=6), line=dict(color=GREY),
                         fillcolor="rgba(0,0,0,0)", name="beam transects",
                         customdata=e[["date", "overpass_id", "radius_km"]],
                         hovertemplate="c = %{x:.1f} cm<br>%{customdata[0]}<br>ovp %{customdata[1]}"
                                       "<br>radius %{customdata[2]} km<extra></extra>"))
    fig.add_trace(go.Scatter(x=100 * s.value, y=s.station, mode="markers", name="station median, 95 % CI",
                             marker=dict(color="black", size=11, symbol="diamond"),
                             error_x=dict(type="data", symmetric=False, array=100 * (s.ci_hi - s.value),
                                          arrayminus=100 * (s.value - s.ci_lo))))
    fig.add_vline(x=100 * D.val(claim_id="V1_ATL13_GAUGE_CLOSURE", statistic="mean_c_m"),
                  line_dash="dash", annotation_text="mean of station medians")
    fig.update_layout(xaxis_title="c = H_gauge(EVRF2019) − H_ATL13(EGG2015, mean-tide), cm",
                      yaxis=dict(categoryorder="array", categoryarray=order), height=420,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    return fig


# ------------------------------------------------------------------- V2 --
def v2_anomaly_scatter() -> go.Figure:
    e = D.evidence()
    e = e[(e.claim_id == "V2_ATL13_GAUGE_COVARIABILITY") & e.included_primary].copy()
    e["gauge anomaly, m"] = e.gauge_level_m - e.groupby("station").gauge_level_m.transform("mean")
    e["ATL13 anomaly, m"] = e.satellite_level_m - e.groupby("station").satellite_level_m.transform("mean")
    fig = px.scatter(e, x="gauge anomaly, m", y="ATL13 anomaly, m", color="station",
                     hover_data=["date", "overpass_id", "distance_m"], height=480)
    lim = float(np.nanmax(np.abs(e[["gauge anomaly, m", "ATL13 anomaly, m"]].values))) * 1.05
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="1:1",
                             line=dict(color=GREY, dash="dash")))
    _theil_line(fig, e["gauge anomaly, m"], e["ATL13 anomaly, m"])
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------------- V3 --
def v3_closures() -> go.Figure:
    e = D.evidence()
    ic = e[(e.claim_id == "V1_ATL13_GAUGE_CLOSURE") & (e.station == "Rozumivka") & e.included_primary]
    sw = e[(e.claim_id == "V3_ROZUMIVKA_TRANSFER") & e.included_primary]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=("closure residual c through time, the same gauge",
                                        "the two closures, estimated independently"))
    fig.add_trace(go.Scatter(x=pd.to_datetime(ic.date), y=100 * ic.closure_residual_m, mode="markers",
                             marker=dict(color=BLUE, size=8), name="ICESat-2, pre-breach (beam transects)"),
                  1, 1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(sw.date), y=100 * sw.closure_residual_m, mode="markers",
                             marker=dict(color=ORANGE, size=7, symbol="diamond"),
                             name="SWOT RiverSP, post-breach (passes, ≤3 km)",
                             customdata=sw[["swot_pass_id", "n_nodes"]],
                             hovertemplate="%{x}<br>c = %{y:.1f} cm<br>%{customdata[0]}"
                                           "<br>%{customdata[1]} nodes<extra></extra>"), 1, 1)
    fig.add_vline(x=BREACH, line_dash="dot", row=1, col=1)
    rows = [("ICESat-2 pre", D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_m",
                                    sensor="ICESat-2"), BLUE),
            ("SWOT post", D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_m", sensor="SWOT",
                                 radius_km=3.0, variant="raw_median"), ORANGE),
            ("difference", D.stat(claim_id="V3_ROZUMIVKA_TRANSFER", statistic="median_diff_m",
                                  radius_km=3.0,
                                  variant="raw_median: median c_SWOT - median c_IS2, both resampled"),
             "black")]
    for lab, r, c in rows:
        fig.add_trace(go.Scatter(x=[lab], y=[100 * r.value], mode="markers", marker=dict(size=12, color=c),
                                 error_y=dict(type="data", symmetric=False, array=[100 * (r.ci_hi - r.value)],
                                              arrayminus=[100 * (r.value - r.ci_lo)]),
                                 showlegend=False), 1, 2)
    fig.add_hline(y=0, line_color=GREY, row=1, col=2)
    fig.update_yaxes(title_text="c = gauge − satellite, cm", row=1, col=1)
    fig.update_yaxes(title_text="median, 95 % CI, cm", row=1, col=2)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


def v3_radius_table() -> pd.DataFrame:
    s = D.summary()
    s = s[(s.claim_id == "V3_ROZUMIVKA_TRANSFER") & (s.sensor == "SWOT")
          & s.variant.isin(["raw_median", "slope_corrected"]) & s.statistic.isin(["n", "median_m", "nmad_m"])]
    t = s.pivot_table(index=["radius_km", "variant"], columns="statistic", values="value").reset_index()
    t["median, cm"] = (100 * t.median_m).round(1)
    t["NMAD, cm"] = (100 * t.nmad_m).round(1)
    return t[["radius_km", "variant", "n", "median, cm", "NMAD, cm"]].rename(
        columns={"radius_km": "support, km", "n": "passes"})


# ------------------------------------------------------------------- V4 --
def _v4_primary() -> pd.DataFrame:
    e = D.evidence()
    y = e[(e.claim_id == "V4_SWOT_ICESAT_DIRECT") & (e.validation_path == "V4") & e.included_primary].copy()
    d = pd.to_datetime(y.date)
    y["breach fortnight"] = d.between(BREACH, BREACH + pd.Timedelta(days=14))
    return y


def v4_scatter() -> go.Figure:
    y = _v4_primary()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("absolute heights (EGG2015)",
                                                        "anomalies from the zone median (zones n ≥ 3)"))
    for z, g in y.groupby("zone"):
        for bf, sym in ((False, "circle"), (True, "circle-open")):
            h = g[g["breach fortnight"] == bf]
            fig.add_trace(go.Scatter(x=h.icesat_level_m, y=h.satellite_level_m, mode="markers",
                                     marker=dict(color=C.ZONE_COLOURS.get(z, "black"), size=9, symbol=sym,
                                                 line=dict(width=1.5)),
                                     name=f"{z}{' · 6–20 Jun 2023' if bf else ''}",
                                     customdata=h[["date", "overpass_id", "swot_pass_id", "closure_residual_m"]],
                                     hovertemplate="%{customdata[0]}<br>ovp %{customdata[1]}<br>"
                                                   "%{customdata[2]}<br>Δ = %{customdata[3]:.3f} m"
                                                   "<extra></extra>"), 1, 1)
    lim = [y[["icesat_level_m", "satellite_level_m"]].min().min() - 0.5,
           y[["icesat_level_m", "satellite_level_m"]].max().max() + 0.5]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dash"),
                             name="1:1"), 1, 1)
    nz = y.zone.map(y.zone.value_counts())
    a = y[nz >= 3].copy()
    hm = 0.5 * (a.satellite_level_m + a.icesat_level_m)
    med = hm.groupby(a.zone).transform("median")
    a["As"], a["Aa"] = a.satellite_level_m - med, a.icesat_level_m - med
    for z, g in a.groupby("zone"):
        fig.add_trace(go.Scatter(x=g.Aa, y=g.As, mode="markers", showlegend=False,
                                 marker=dict(color=C.ZONE_COLOURS.get(z), size=9,
                                             symbol=np.where(g["breach fortnight"], "circle-open", "circle"),
                                             line=dict(width=1.5))), 1, 2)
    m = float(np.abs(a[["Aa", "As"]].values).max()) * 1.1
    fig.add_trace(go.Scatter(x=[-m, m], y=[-m, m], mode="lines", line=dict(color=GREY, dash="dash"),
                             showlegend=False), 1, 2)
    fig.update_xaxes(title_text="ICESat-2 ATL13, m", row=1, col=1)
    fig.update_yaxes(title_text="SWOT RiverSP, m", row=1, col=1)
    fig.update_xaxes(title_text="ICESat-2 anomaly, m", row=1, col=2)
    fig.update_yaxes(title_text="SWOT anomaly, m", row=1, col=2)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def v4_timegap() -> go.Figure:
    rows = []
    for lab, var in (("≤ 24 h (primary)", None), ("≤ 24 h, without 6–20 Jun 2023",
                                                  "excluding 2023-06-06..06-20 (breach fortnight)"),
                     ("1–3 d", "timing sensitivity 1-3 d"), ("3–10 d", "timing sensitivity 3-10 d")):
        kw = dict(claim_id="V4_SWOT_ICESAT_DIRECT", variant=var)
        rows.append(dict(window=lab, n=int(D.val(statistic="n", **kw)),
                         median=100 * D.val(statistic="median_m", **kw),
                         NMAD=100 * D.val(statistic="nmad_m", **kw),
                         RMSE=100 * D.val(statistic="rmse_m", **kw)))
    t = pd.DataFrame(rows)
    fig = go.Figure()
    for k, c in (("median", "black"), ("NMAD", BLUE), ("RMSE", ORANGE)):
        fig.add_trace(go.Bar(x=t.window, y=t[k], name=k, marker_color=c,
                             text=[f"{v:.1f}" for v in t[k]], textposition="outside"))
    fig.update_layout(barmode="group", yaxis_title="cm", height=380,
                      xaxis=dict(ticktext=[f"{w}<br>n = {n}" for w, n in zip(t.window, t.n)],
                                 tickvals=t.window), margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------------- V6 --
def v6_closures() -> go.Figure:
    s = D.summary()
    s = s[(s.claim_id == "V6_KHERSON_CLOSURE") & (s.statistic == "median_m")]
    n = D.summary()
    n = n[(n.claim_id == "V6_KHERSON_CLOSURE") & (n.statistic == "n")].set_index("series").value
    col = {"ICESat-2": BLUE, "SWOT RiverSP": ORANGE, "SWOT PIXC": GREEN}
    fig = go.Figure()
    for _, r in s.iterrows():
        sen = next(k for k in col if r.series.startswith(k))
        fig.add_trace(go.Scatter(x=[100 * r.value], y=[f"{r.series.replace('_', '-')} (n={int(n[r.series])})"],
                                 mode="markers", marker=dict(size=11, color=col[sen]), showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=[100 * (r.ci_hi - r.value)],
                                              arrayminus=[100 * (r.value - r.ci_lo)])))
    fig.add_vline(x=0, line_color=GREY)
    fig.update_layout(xaxis_title="c = gauge − satellite, cm (median, 95 % CI)", height=330,
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def v6_scatter() -> go.Figure:
    e = D.evidence()
    e = e[(e.claim_id == "V6_KHERSON_CLOSURE") & e.included_primary]
    fig = px.scatter(e, x="gauge_level_m", y="satellite_level_m", color="series",
                     hover_data=["date", "overpass_id", "swot_pass_id", "n_nodes"], height=430,
                     labels=dict(gauge_level_m="gauge, m EVRF2019", satellite_level_m="satellite, m EGG2015"))
    lo, hi = e.gauge_level_m.min(), e.gauge_level_m.max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                             line=dict(color=GREY, dash="dash")))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------ inventory --
def station_inventory() -> go.Figure:
    inv = D.inventory()
    fig = px.timeline(inv, x_start="first", x_end="last", y="name_en", color="spans_breach",
                      color_discrete_map={True: ORANGE, False: BLUE},
                      hover_data=["n_days", "n_days_post_breach"], height=320,
                      labels=dict(spans_breach="record spans the breach", name_en=""))
    fig.add_vline(x=BREACH.timestamp() * 1000, line_dash="dot")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def zone_counts() -> go.Figure:
    m = D.zone_manifest()
    st_ = D.layer("stations.csv")
    cr = D.layer("crossings.csv")
    rows = []
    for name, z in m["zones"].items():
        L = z["letter"]
        rows.append(dict(zone=L, what="area, 100 km²", value=z["area_km2"] / 100))
        rows.append(dict(zone=L, what="stations", value=int((st_.zone == L).sum())))
        rows.append(dict(zone=L, what="crossings ≤ 24 h",
                         value=int(((cr.zone == L) & cr.included_primary.astype(str).str.lower().eq("true")).sum())))
    t = pd.DataFrame(rows)
    fig = px.bar(t, x="zone", y="value", color="what", barmode="group", height=340, text="value",
                 category_orders=dict(zone=list("RFDE")))
    fig.update_traces(texttemplate="%{text:.0f}")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), yaxis_title="")
    return fig
