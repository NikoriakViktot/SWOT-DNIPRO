#!/usr/bin/env python
"""MS6B — SWOT against ICESat-2 at every crossing over the paper zones (S3.3, Figure S1).

The v5 numbers of Supplementary S3.3 (+2.1 cm [-1.3, +5.5], NMAD 8.1 cm,
n = 32 crossings within a day) and Figure S1 were produced by code that no
repository kept. This module recomputes them from the primary products under
the rules S3.1 states, so that the figure and its numbers have a source:

  * a SWOT RiverSP node of good quality (node_q <= 1, dark_frac < 0.5) inside
    the analysis zones, and ATL13 segments within 200 m of it;
  * the independent unit is the CROSSING - one ICESat-2 overpass (rgt, date)
    against one SWOT pass - with node-level differences reduced by the median
    inside it; |dt| <= 1 d primary, 1-3 d and 3-10 d as timing sensitivity;
  * both sensors reduced to EGG2015 at their own points:
        SWOT     h = wse + geoid_hght            (RiverSP wse is tide-corrected,
                                                  mean-tide crust)
        ICESat-2 h = h_wgs84 + free2mean(lat)    (tide-free -> mean-tide crust)
        H_EGG2015 = h - zeta_EGG2015   (zeta interpolated bilinearly: crossings
                                        are differences over <= 200 m, and a
                                        nearest-cell sample adds cm steps)
  * LakeSP: small (<= 20 km2) observed lakes with ATL13 segments inside the
    polygon, lake wse against the median ATL13 height inside it; large lakes
    are shown, never tested (a lake-averaged level of a sloping body is not a
    level).

The universe is run twice: on the paper zones R/F/D/E and on the union of the
old download containers ZONE_1..4, so the effect of the zone change on the
published numbers is measured rather than assumed.

Outputs
-------
outputs/paper/tables/ms6b_swot_icesat_crossings.csv     one row per crossing
outputs/paper/tables/ms6b_swot_icesat_lake_pairs.csv
outputs/paper/tables/ms6b_swot_icesat_summary.csv       per universe/window/period/zone
"""
from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from swot_dnipro import config as CFG
from swot_dnipro import spatial_domains as SD
from swot_dnipro.vertical import sample_grid_bilinear

UTM = CFG.CRS_METRIC
TAB = ROOT / "outputs/paper/tables"
SWOT_UA = CFG.BULK_ROOT / "swot_ua"
ATL13 = [CFG.ICESAT_ROOT / f"data/processed/{n}_atl13_segments.parquet"
         for n in ("kakhovka", "kherson", "dnipro_estuary")]
S2_MASKS = CFG.BULK_ROOT / "data_swot/processed/water_masks"
MASK_DATES = {"pre": "20230605", "post": "20230908"}

PAPER_ZONES = {"R": "R_FORMER_KAKHOVKA_RESERVOIR", "F": "F_LOWER_DNIPRO_FLOODWAY",
               "D": "D_KHERSON_DELTA", "E": "E_DNIPRO_BUG_ESTUARY"}
LEGACY = ("ZONE_1_KAKHOVKA_LOWER_DNIPRO", "ZONE_2_KHERSON_DELTA",
          "ZONE_3_DNIPRO_BUG_ESTUARY", "ZONE_4_DAM_TO_KHERSON_FLOODWAY")

START = pd.Timestamp("2023-03-29", tz="UTC")      # SWOT science data begin
POST_START = pd.Timestamp("2023-09-01", tz="UTC")  # the POST_BREACH period
RADIUS_M = 200.0
NODE_Q_MAX, DARK_MAX = 1, 0.5
LAKE_SMALL_KM2 = 20.0
WINDOWS = {"<=1d": (0, 1), "1-3d": (1, 3), "3-10d": (3, 10)}
N_BOOT = 10_000
FILL = -1e11

_TS = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def _file_time(p: Path) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(_TS.search(p.name).group(1), "%Y%m%dT%H%M%S"),
                        tz="UTC")


_CRID = re.compile(r"_(P[A-Z]{2}\d)_(\d\d)\.shp$")
#: newer processing first; PIC2 > PIC0 (reprocessed) > PGC0 (forward stream)
CRID_RANK = {"PIC2": 3, "PIC0": 2, "PGC0": 1}


def latest_version(g: pd.DataFrame) -> pd.DataFrame:
    """Keep one processing version per granule: the highest CRID rank, then the
    highest product counter."""
    r = g.crid.map(CRID_RANK).fillna(0) * 100 + g.product_counter
    best = r.groupby(g.granule).transform("max")
    return g[r == best]


def _cycle_pass(p: Path) -> str:
    parts = p.name.split("_")
    i = parts.index("Node") if "Node" in parts else parts.index("Obs")
    return f"{parts[i + 1]}_{parts[i + 2]}"


def nmad(x) -> float:
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def boot_ci(x, rng) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    m = np.median(x[rng.integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


# ------------------------------------------------------------------ inputs --
def universes() -> dict[str, dict]:
    paper = {L: SD.load_utm(n) for L, n in PAPER_ZONES.items()}
    legacy = unary_union([SD.load_utm(n) for n in LEGACY])
    return {"paper_RFDE": dict(union=unary_union(list(paper.values())), zones=paper),
            "legacy_ZONE_1to4": dict(union=legacy, zones=paper)}


def load_atl13(union) -> pd.DataFrame:
    cols = ["time", "rgt", "beam", "segment_id", "lat", "lon", "h_wgs84_m"]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in ATL13], ignore_index=True)
    d = d[d.time >= START].drop_duplicates(["time", "beam", "segment_id"])
    xy = gpd.GeoSeries(gpd.points_from_xy(d.lon, d.lat), crs=4326).to_crs(UTM)
    d["x"], d["y"] = xy.x.values, xy.y.values
    d = d[shapely.contains_xy(union, d.x.values, d.y.values)].copy()
    d["date"] = d.time.dt.floor("D")
    d["ovp"] = d.rgt.astype(str) + "_" + d.date.dt.strftime("%Y%m%d")
    zeta = sample_grid_bilinear(CFG.EGG2015_TIF, d.lon.values, d.lat.values)
    d["H"] = d.h_wgs84_m + CFG.free2mean(d.lat.values) - zeta
    return d.reset_index(drop=True)


def _read_swot(kind: str, union) -> pd.DataFrame | gpd.GeoDataFrame:
    sub = "swot_l2_hr_riversp_2.0" if kind == "node" else "swot_l2_hr_lakesp_2.0"
    pat = "*RiverSP_Node*.shp" if kind == "node" else "*LakeSP_Obs*.shp"
    bbox = gpd.GeoSeries([union], crs=UTM).to_crs(4326).total_bounds
    frames = []
    for p in sorted((SWOT_UA / sub).rglob(pat)):
        if _file_time(p) < START - pd.Timedelta(days=11):
            continue
        g = pyogrio.read_dataframe(p, bbox=tuple(bbox))
        if g.empty:
            continue
        g["cycle_pass"] = _cycle_pass(p)
        # The archive holds some granules in more than one processing version
        # (CRID PGC0 / PIC0 / PIC2, product counter _01.._04): record which, so
        # the same observation is never counted twice (see latest_version()).
        crid, counter = _CRID.search(p.name).groups()
        g["crid"], g["product_counter"] = crid, int(counter)
        g["granule"] = _CRID.sub("", p.name)
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    g = g[(g.wse > FILL) & (g.geoid_hght > FILL)]
    g["t"] = pd.to_datetime(g.time_str.str.replace("Z", ""), errors="coerce", utc=True)
    return g[g.t.notna()]


def load_nodes(union, all_quality: bool = False) -> pd.DataFrame:
    """Good RiverSP nodes (S3.1 rule), or every node when ``all_quality`` so a
    caller can apply its own quality filter later (ms6c)."""
    g = _read_swot("node", union)
    if not all_quality:
        g = g[(g.node_q <= NODE_Q_MAX) & (g.dark_frac < DARK_MAX)]
    g = g.copy()
    xy = gpd.GeoSeries(gpd.points_from_xy(g.lon, g.lat), crs=4326).to_crs(UTM)
    g["x"], g["y"] = xy.x.values, xy.y.values
    g = g[shapely.contains_xy(union, g.x.values, g.y.values)].copy()
    zeta = sample_grid_bilinear(CFG.EGG2015_TIF, g.lon.values, g.lat.values)
    g["H"] = g.wse + g.geoid_hght - zeta
    g["pass_id"] = g.cycle_pass + "_" + g.t.dt.strftime("%Y%m%d")
    return pd.DataFrame(g.drop(columns="geometry")).reset_index(drop=True)


def load_lakes(union) -> gpd.GeoDataFrame:
    g = _read_swot("lake", union).to_crs(UTM)
    g = g[g.intersects(union)].copy()
    c = g.geometry.representative_point().to_crs(4326)
    zeta = sample_grid_bilinear(CFG.EGG2015_TIF, c.x.values, c.y.values)
    g["H"] = g.wse + g.geoid_hght - zeta
    g["size"] = np.where(g.area_total <= LAKE_SMALL_KM2, "small", "large")
    return g


# --------------------------------------------------------------- matching --
def node_pairs(atl: pd.DataFrame, nodes: pd.DataFrame,
               radii=(RADIUS_M,), max_dt_days: float = 10.0) -> pd.DataFrame:
    """One row per (ICESat-2 overpass, SWOT node observation) with at least one
    ATL13 segment of that overpass within max(radii) of the node: the median
    ATL13 height within each radius (NaN where none), the nearest distance, and
    the node-level dt. Grouping into crossings is a separate step, so the
    crossing rule can vary without re-matching (ms6c)."""
    rmax = max(radii)
    rows = []
    ntree = cKDTree(nodes[["x", "y"]].values)
    for ovp, a in atl.groupby("ovp"):
        t_a = a.time.median()
        near = ntree.query_ball_point(a[["x", "y"]].values, rmax)
        idx = np.unique(np.concatenate([np.asarray(n, int) for n in near]))
        if not len(idx):
            continue
        cand = nodes.iloc[idx]
        dt = (cand.t - t_a).dt.total_seconds() / 86400.0
        keep = dt.abs().values <= max_dt_days
        cand, dt = cand[keep], dt[keep]
        if cand.empty:
            continue
        axy = a[["x", "y"]].values
        atree = cKDTree(axy)
        dist, _ = atree.query(cand[["x", "y"]].values)
        rec = dict(ovp=ovp, icesat_time=t_a, node_row=cand.index.values,
                   pass_id=cand.pass_id.values, t=cand.t.values, dt_d=dt.values,
                   x=cand.x.values, y=cand.y.values, H_swot=cand.H.values,
                   dist_m=dist)
        for r in radii:
            ks = atree.query_ball_point(cand[["x", "y"]].values, r)
            rec[f"H_atl_{int(r)}"] = np.array(
                [np.median(a.H.values[k]) if k else np.nan for k in ks])
            rec[f"n_seg_{int(r)}"] = np.array([len(k) for k in ks])
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def group_crossings(pairs: pd.DataFrame, radius: float = RADIUS_M,
                    keys=("ovp", "pass_id")) -> pd.DataFrame:
    """Crossings = node-level differences reduced by the median inside each
    group (S3.1: one ICESat-2 overpass against one SWOT pass)."""
    p = pairs[pairs[f"n_seg_{int(radius)}"] > 0]
    p = p.assign(d=p.H_swot - p[f"H_atl_{int(radius)}"])
    g = p.groupby(list(keys), sort=True)
    out = pd.DataFrame(dict(
        icesat_time=g.icesat_time.first(), swot_time=g.t.median(),
        dt_days=g.dt_d.median(), n_nodes=g.d.size(), d_m=g.d.median(),
        x=g.x.median(), y=g.y.median())).reset_index()
    out["swot_time"] = pd.to_datetime(out.swot_time, utc=True)
    return out.rename(columns={"pass_id": "swot_pass"})


def match_nodes(atl: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    return group_crossings(node_pairs(atl, nodes))


def match_lakes(atl: pd.DataFrame, lakes: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    pts = gpd.GeoDataFrame(atl[["ovp", "time", "H"]],
                           geometry=gpd.points_from_xy(atl.x, atl.y), crs=UTM)
    j = gpd.sjoin(pts, lakes[["obs_id", "cycle_pass", "t", "H", "size", "area_total",
                              "geometry"]].reset_index(names="lake_row"),
                  predicate="within", lsuffix="a", rsuffix="s")
    j["dt_days"] = (j.t - j.time).dt.total_seconds() / 86400.0
    j = j[j.dt_days.abs() <= 1]
    for (ovp, lr), g in j.groupby(["ovp", "lake_row"]):
        c = lakes.loc[lr].geometry.representative_point()
        rows.append(dict(ovp=ovp, lake_obs=g.obs_id.iloc[0], size=g["size"].iloc[0],
                         area_km2=float(g.area_total.iloc[0]), n_seg=len(g),
                         icesat_range_m=float(np.ptp(g.H_a)), dt_days=float(g.dt_days.median()),
                         icesat_time=g.time.median(),
                         d_m=float(g.H_s.iloc[0] - np.median(g.H_a)), x=c.x, y=c.y))
    return pd.DataFrame(rows)


def label(df: pd.DataFrame, zones: dict) -> pd.DataFrame:
    df = df.copy()
    df["period"] = np.where(df.icesat_time >= POST_START, "post", "pre_drawdown")
    df["zone"] = "none"
    for L, g in zones.items():
        df.loc[shapely.contains_xy(g, df.x.values, df.y.values), "zone"] = L
    df["window"] = pd.cut(df.dt_days.abs(), [-1e-9, 1, 3, 10],
                          labels=list(WINDOWS)).astype(str)
    return df


def summarise(df: pd.DataFrame, universe: str, rng) -> list[dict]:
    out = []

    def row(sel, what):
        x = sel.d_m.values
        lo, hi = boot_ci(x, rng)
        out.append(dict(universe=universe, subset=what, n=len(x),
                        median_cm=100 * np.median(x) if len(x) else np.nan,
                        ci_lo_cm=100 * lo, ci_hi_cm=100 * hi, nmad_cm=100 * nmad(x),
                        within_10cm_pct=100 * np.mean(np.abs(x) <= 0.10) if len(x) else np.nan))

    for w in WINDOWS:
        s = df[df.window == w]
        row(s, f"window {w}")
    s1 = df[df.window == "<=1d"]
    for p in ("pre_drawdown", "post"):
        row(s1[s1.period == p], f"<=1d period {p}")
    for z in sorted(s1.zone.unique()):
        row(s1[s1.zone == z], f"<=1d zone {z}")
    return out


def run() -> dict:
    TAB.mkdir(parents=True, exist_ok=True)
    U = universes()
    rng = np.random.default_rng(CFG.SEED)
    big = U["legacy_ZONE_1to4"]["union"].union(U["paper_RFDE"]["union"])
    print("reading ATL13 ...")
    atl = load_atl13(big)
    print(f"  {len(atl):,} segments, {atl.ovp.nunique()} overpasses")
    print("reading RiverSP nodes ...")
    nodes = load_nodes(big)
    print(f"  {len(nodes):,} good nodes, {nodes.pass_id.nunique()} passes")
    print("reading LakeSP ...")
    lakes = load_lakes(big)
    print(f"  {len(lakes):,} lake observations")

    xs = match_nodes(atl, nodes)
    lk = match_lakes(atl, lakes)
    res, summ = {}, []
    for name, u in U.items():
        inside = lambda d: d[shapely.contains_xy(u["union"], d.x.values, d.y.values)]
        x = label(inside(xs), u["zones"]).assign(universe=name)
        l = label(inside(lk), u["zones"]).assign(universe=name) if len(lk) else lk
        summ += summarise(x, name, rng)
        for sz in ("small", "large"):
            ls = l[l["size"] == sz] if len(l) else l
            if len(ls):
                summ.append(dict(universe=name, subset=f"LakeSP {sz} <=1d", n=len(ls),
                                 median_cm=100 * ls.d_m.median(),
                                 nmad_cm=100 * nmad(ls.d_m)))
        res[name] = dict(crossings=x, lakes=l)
    pd.concat([r["crossings"] for r in res.values()]).to_csv(
        TAB / "ms6b_swot_icesat_crossings.csv", index=False)
    if len(lk):
        pd.concat([r["lakes"] for r in res.values()]).to_csv(
            TAB / "ms6b_swot_icesat_lake_pairs.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(TAB / "ms6b_swot_icesat_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(S.round(2).to_string(index=False))
    return res


# ----------------------------------------------------------------- figure --
def _mask_panel(ax, date):
    import rasterio
    from rasterio.merge import merge
    fs = sorted(S2_MASKS.glob(f"S2*_{date}T*_water.tif"))
    srcs = [rasterio.open(f) for f in fs]
    arr, tr = merge(srcs, res=100, nodata=255, resampling=rasterio.enums.Resampling.mode)
    for s in srcs:
        s.close()
    a = arr[0].astype(float)
    a[a == 255] = np.nan
    h, w = a.shape
    ext = (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)
    ax.imshow(np.where(np.isnan(a), np.nan, 0.0), extent=ext, cmap="Greys",
              vmin=0, vmax=12, zorder=0)          # observed footprint, pale
    ax.imshow(np.where(a == 1, 1.0, np.nan), extent=ext, cmap="Blues",
              vmin=0, vmax=2.5, zorder=1)


def figure(zones_4326: dict, zstyle: dict, save) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    xf, lf = TAB / "ms6b_swot_icesat_crossings.csv", TAB / "ms6b_swot_icesat_lake_pairs.csv"
    if not xf.exists():
        run()
    x = pd.read_csv(xf).query("universe == 'paper_RFDE'")
    l = pd.read_csv(lf).query("universe == 'paper_RFDE'") if lf.exists() else pd.DataFrame()
    x1 = x[x.window == "<=1d"]
    l1 = l[l.window == "<=1d"] if len(l) else l
    Z = {L: g.to_crs(UTM) for L, g in zones_4326.items()}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharex=True, sharey=True)
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=0.3)
    cmap = plt.get_cmap("RdBu_r")
    for ax, per, title, md in zip(
            axes, ("pre_drawdown", "post"),
            ("(a) Pre-breach + drawdown, |Δt| ≤ 1 d", "(b) Post-breach, |Δt| ≤ 1 d"),
            (MASK_DATES["pre"], MASK_DATES["post"])):
        _mask_panel(ax, md)
        for L, g in Z.items():
            g.boundary.plot(ax=ax, color=zstyle[L][1], lw=1.0, zorder=2)
            # letters sit clear of the crossings: F at the top of its northern
            # lobe, R just south of the reservoir's widest part
            b = g.total_bounds
            p = {"F": (b[0] + 0.55 * (b[2] - b[0]), b[3] - 9e3),
                 "R": (620e3, 5.228e6)}.get(L)
            if p is None:
                q = g.iloc[0].representative_point()
                p = (q.x, q.y)
            ax.text(*p, L, fontsize=15, weight="bold", color=zstyle[L][1],
                    ha="center", va="center", zorder=7,
                    path_effects=[pe.withStroke(linewidth=3, foreground="w")])
        s = x1[x1.period == per]
        ax.scatter(s.x, s.y, c=s.d_m.clip(-0.3, 0.3), cmap=cmap, norm=norm, s=55,
                   ec="k", lw=0.6, zorder=5)
        n_small = n_large = 0
        if len(l1):
            ls = l1[l1.period == per]
            sm, lg = ls[ls["size"] == "small"], ls[ls["size"] == "large"]
            n_small, n_large = len(sm), len(lg)
            ax.scatter(sm.x, sm.y, c=sm.d_m.clip(-0.3, 0.3), cmap=cmap, norm=norm,
                       marker="D", s=65, ec="k", lw=0.6, zorder=5)
            ax.scatter(lg.x, lg.y, marker="s", s=80, fc="none", lw=1.8, zorder=4,
                       ec=cmap(norm(lg.d_m.clip(-0.3, 0.3))))
        dd = f"{md[:4]}-{md[4:6]}-{md[6:]}"
        ax.set_title(f"{title}\nS2 water {dd} (reservoir tiles only)", loc="left",
                     fontsize=10, weight="bold")
        ax.legend(handles=[
            Line2D([], [], ls="", marker="o", mfc="#f0c8c0", mec="k", ms=7,
                   label=f"RiverSP node crossings (n={len(s)})"),
            Line2D([], [], ls="", marker="D", mfc="#f0c8c0", mec="k", ms=7,
                   label=f"LakeSP water body ≤{LAKE_SMALL_KM2:g} km² (n={n_small})"),
            Line2D([], [], ls="", marker="s", mfc="none", mec="#b03a2e", mew=1.8, ms=8,
                   label=f"LakeSP >{LAKE_SMALL_KM2:g} km², not level: not a valid test (n={n_large})"),
            Line2D([], [], c="k", lw=1.0, label="analysis zones R / F / D / E"),
        ], loc="lower right", fontsize=7.5, framealpha=0.9)
        ax.set_aspect("equal")
        ax.set_xlabel("UTM 36N easting, m")
        ax.ticklabel_format(style="plain")
    axes[0].set_ylabel("northing, m")
    b = unary_union([g.iloc[0] for g in Z.values()]).bounds
    axes[0].set_xlim(b[0] - 5e3, b[2] + 5e3)
    axes[0].set_ylim(b[1] - 5e3, b[3] + 5e3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01,
                 label="SWOT − ICESat-2, both in EGG2015, m (clipped ±0.3)")
    save(fig, "F18_swot_icesat_whole_zone_map")


if __name__ == "__main__":
    run()
