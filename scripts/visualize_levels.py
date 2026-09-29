"""
visualize_levels.py — Multi-period SWOT water level visualization.

Generates static figures from downloaded SWOT data + gauge xlsx files.
Run from project root: python scripts/visualize_levels.py

Produces:
  data/figures/01_swot_timeseries_kakhovka.png
  data/figures/02_swot_timeseries_dniprovske.png
  data/figures/03_gauge_timeseries.png
  data/figures/04_period_comparison_boxplot.png
  data/figures/05_swot_vs_gauge.png
  data/figures/06_multi_aoi_overlay.png
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = ROOT / "data" / "index" / "granules.parquet"
# The continent-wide originals under data/raw were deleted 2026-09-16 after
# scripts/p22b_verify_ukraine_clip.py proved the Ukraine clip a faithful subset
# (483 GB -> 35 GB). Both products now live clipped on the bulk volume.
_SWOT_UA = Path(os.environ.get("SWOT_DNIPRO_BULK_ROOT", "/mnt/f/data_kakhovka_dem_swot")) / "swot_ua"
LAKESP_DIR = _SWOT_UA / "swot_l2_hr_lakesp_2.0"
RIVERSP_DIR = _SWOT_UA / "swot_l2_hr_riversp_2.0"
GAUGE_KAKH  = ROOT / "data" / "гідропости_каховка.xlsx"
GAUGE_DNIPRO = ROOT / "data" / "гідропости_дніпровське.xlsx"

DAM_BREACH = pd.Timestamp("2023-06-06")

PERIODS = {
    "До руйнування\n(до 06.06.2023)":   (pd.Timestamp("2022-12-01"), pd.Timestamp("2023-06-05")),
    "Руйнування\n(червень 2023)":        (pd.Timestamp("2023-06-06"), pd.Timestamp("2023-07-31")),
    "Після\n(серпень 2023+)":            (pd.Timestamp("2023-08-01"), pd.Timestamp("2026-03-05")),
}
PERIOD_COLORS = ["steelblue", "crimson", "darkorange"]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_nc_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.nc"))


def load_lakesp_wse(nc_files: list[Path], aoi_bbox: tuple | None = None) -> pd.DataFrame:
    """Load WSE from LakeSP NetCDF files. Returns daily-aggregated DataFrame."""
    try:
        import xarray as xr
        import netCDF4  # noqa: F401
    except ImportError:
        print("  xarray/netCDF4 not installed — skipping NetCDF loading")
        return pd.DataFrame()

    rows = []
    SWOT_EPOCH = pd.Timestamp("2000-01-01")

    for f in nc_files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
            # Find WSE variable
            wse_var = next((v for v in ["wse", "WSE", "water_surface_elevation"] if v in ds), None)
            if wse_var is None:
                continue
            wse = ds[wse_var].values.flatten().astype(float)

            # Time
            time_var = next((v for v in ["time", "time_tai"] if v in ds), None)
            if time_var:
                t_raw = ds[time_var].values.flatten().astype(float)
                # SWOT time: seconds since 2000-01-01
                times = [SWOT_EPOCH + pd.Timedelta(seconds=float(s))
                         if not np.isnan(s) else pd.NaT for s in t_raw]
            else:
                times = [pd.NaT] * len(wse)

            # Quality filter
            quality = None
            for qv in ["quality_f", "quality_flag", "qual_f"]:
                if qv in ds:
                    quality = ds[qv].values.flatten().astype(float)
                    break

            for i, (w, t) in enumerate(zip(wse, times)):
                if np.isnan(w) or w < -9000 or t is pd.NaT:
                    continue
                if quality is not None and quality[i] != 0:
                    continue
                if w < 0 or w > 150:  # sanity range (m above geoid)
                    continue
                rows.append({"date": t.date(), "wse": w})
            ds.close()
        except Exception as exc:
            print(f"  Warning: could not read {f.name}: {exc}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["wse"].agg(["mean", "std", "count"]).reset_index()
    daily.columns = ["date", "wse_mean", "wse_std", "n_obs"]
    daily = daily[daily["n_obs"] >= 1]
    daily["wse_std"] = daily["wse_std"].fillna(0)
    return daily.sort_values("date")


def detect_level_col(df: pd.DataFrame) -> str | None:
    """Auto-detect water level column in gauge DataFrame."""
    keywords = ["рівень", "level", "wl", "wse", "h_", "відмітка", "позначка"]
    for col in df.columns:
        cl = col.lower().strip()
        if any(kw in cl for kw in keywords):
            return col
    # fallback: first numeric column after date
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric[0] if numeric else None


def detect_date_col(df: pd.DataFrame) -> str | None:
    keywords = ["дата", "date", "час", "time", "datetime"]
    for col in df.columns:
        cl = col.lower().strip()
        if any(kw in cl for kw in keywords):
            return col
    return None


def load_gauge_stations(path: Path) -> pd.DataFrame:
    """Load gauge station metadata (hydropost, id, lon, lat, period)."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(path)
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        df["lon"] = pd.to_numeric(df.get("lon", pd.Series(dtype=float)), errors="coerce")
        df["lat"] = pd.to_numeric(df.get("lat", pd.Series(dtype=float)), errors="coerce")
        return df
    except Exception as exc:
        print(f"  Error loading {path.name}: {exc}")
        return pd.DataFrame()


def load_gauge(path: Path, name: str) -> pd.DataFrame:
    """Load gauge TIME SERIES if available (multi-sheet or second sheet)."""
    if not path.exists():
        return pd.DataFrame()
    try:
        # Try reading sheet named 'data' or index 1
        xl = pd.ExcelFile(path)
        sheets_with_data = [s for s in xl.sheet_names if s.lower() not in ("stations", "info", "metadata")]
        if len(sheets_with_data) > 1:
            df = pd.read_excel(path, sheet_name=sheets_with_data[1])
        else:
            return pd.DataFrame()  # only metadata sheet

        date_col = detect_date_col(df)
        level_col = detect_level_col(df)
        if date_col is None or level_col is None:
            return pd.DataFrame()

        df = df[[date_col, level_col]].copy()
        df.columns = ["date", "level"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["level"] = pd.to_numeric(df["level"], errors="coerce")
        df = df.dropna()
        df["station"] = name
        if df["level"].median() > 10:
            df["level"] = df["level"] / 100.0
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


def label_period(date: pd.Timestamp) -> str:
    for label, (s, e) in PERIODS.items():
        if s <= date <= e:
            return label
    return "Невідомо"


# ── figure 1: SWOT time series Kakhovka ──────────────────────────────────────

def fig_swot_timeseries(daily: pd.DataFrame, aoi_name: str, out: Path) -> None:
    if daily.empty:
        print(f"  No SWOT data for {aoi_name} — skipping figure")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily["date"], daily["wse_mean"], color="steelblue", lw=1.5,
            label="WSE (SWOT LakeSP)")
    ax.fill_between(daily["date"],
                    daily["wse_mean"] - daily["wse_std"],
                    daily["wse_mean"] + daily["wse_std"],
                    alpha=0.25, color="steelblue", label="±1σ")

    ax.axvline(DAM_BREACH, color="crimson", ls="--", lw=1.5, label="Руйнування дамби 06.06.2023")

    # Shade periods
    for (label, (s, e)), color in zip(PERIODS.items(), PERIOD_COLORS):
        ax.axvspan(s, e, alpha=0.06, color=color)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Рівень поверхні води (WSE), м")
    ax.set_title(f"SWOT LakeSP — часовий ряд WSE: {aoi_name}")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}  ({len(daily)} daily obs)")


# ── figure 2: gauge time series ───────────────────────────────────────────────

def fig_gauge_timeseries(gauges: list[tuple[str, pd.DataFrame]], out: Path) -> None:
    valid = [(name, df) for name, df in gauges if not df.empty]
    if not valid:
        print("  No gauge data — skipping figure")
        return

    fig, axes = plt.subplots(len(valid), 1, figsize=(13, 4 * len(valid)), sharex=True)
    if len(valid) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, valid):
        ax.plot(df["date"], df["level"], color="seagreen", lw=1.3, alpha=0.85)
        ax.axvline(DAM_BREACH, color="crimson", ls="--", lw=1.5, label="Руйнування дамби")
        ax.set_ylabel("Рівень, м")
        ax.set_title(f"Гідропост: {name}")
        ax.legend(fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    axes[-1].set_xlabel("Дата")
    fig.suptitle("Рівні води: дані гідропостів", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── figure 3: period comparison box plot ─────────────────────────────────────

def fig_period_boxplot(daily: pd.DataFrame, aoi_name: str, out: Path) -> None:
    if daily.empty:
        return

    df = daily.copy()
    df["period"] = df["date"].apply(label_period)
    df = df[df["period"] != "Невідомо"]

    if df.empty:
        print(f"  No period data for {aoi_name}")
        return

    period_order = list(PERIODS.keys())
    present = [p for p in period_order if p in df["period"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {p: c for p, c in zip(period_order, PERIOD_COLORS)}
    sns.boxplot(
        data=df[df["period"].isin(present)],
        x="period", y="wse_mean",
        order=present,
        palette=palette, ax=ax,
        width=0.5, linewidth=1.2,
    )
    ax.set_xlabel("Період")
    ax.set_ylabel("Рівень поверхні (WSE), м")
    ax.set_title(f"Порівняння WSE по періодах — {aoi_name}")

    # annotate medians
    for i, period in enumerate(present):
        sub = df[df["period"] == period]["wse_mean"]
        med = sub.median()
        ax.text(i, med + 0.05, f"{med:.2f} м", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")


# ── figure 4: SWOT vs gauge scatter ──────────────────────────────────────────

def fig_swot_vs_gauge(swot: pd.DataFrame, gauge: pd.DataFrame,
                      swot_name: str, gauge_name: str, out: Path) -> None:
    if swot.empty or gauge.empty:
        print(f"  Insufficient data for SWOT vs gauge comparison")
        return

    # Merge on date with ±2 day tolerance
    swot_m = swot[["date", "wse_mean"]].copy()
    swot_m["date"] = pd.to_datetime(swot_m["date"])
    gauge_m = gauge[["date", "level"]].copy()
    gauge_m["date"] = pd.to_datetime(gauge_m["date"])

    merged = pd.merge_asof(
        swot_m.sort_values("date"),
        gauge_m.sort_values("date"),
        on="date", tolerance=pd.Timedelta("2D"), direction="nearest"
    ).dropna()

    if len(merged) < 3:
        print(f"  Only {len(merged)} matching obs — skipping scatter")
        return

    x, y = merged["level"].values, merged["wse_mean"].values
    rmse = np.sqrt(np.mean((y - x) ** 2))
    bias = np.mean(y - x)
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=merged["date"].astype("int64"),
                    cmap="viridis", alpha=0.75, s=40, edgecolors="k", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Дата").ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: pd.Timestamp(int(v)).strftime("%m/%Y")
        )
    )

    lim = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Гідропост: {gauge_name}, м")
    ax.set_ylabel(f"SWOT WSE: {swot_name}, м")
    ax.set_title(f"SWOT vs гідропост\nRMSE={rmse:.3f} м  |  bias={bias:+.3f} м  |  r={corr:.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}  n={len(merged)}")


# ── figure 5: multi-AOI granule coverage timeline ────────────────────────────

def fig_granule_coverage(out: Path) -> None:
    if not INDEX_FILE.exists():
        print("  No granule index yet — skipping coverage figure")
        return

    df = pd.read_parquet(INDEX_FILE)
    if df.empty:
        return

    df["time_start"] = pd.to_datetime(df["time_start"], errors="coerce")
    df = df.dropna(subset=["time_start"])

    fig, ax = plt.subplots(figsize=(13, 4))
    products = df["product"].unique()
    cmap = plt.cm.get_cmap("tab10", len(products))
    color_map = {p: cmap(i) for i, p in enumerate(products)}

    for aoi_name, grp in df.groupby("aoi_name"):
        y_val = list(df["aoi_name"].unique()).index(aoi_name)
        for _, row in grp.iterrows():
            color = color_map.get(row["product"], "gray")
            marker = "o" if row.get("downloaded") else "x"
            ax.scatter(row["time_start"], y_val, color=color,
                       marker=marker, s=30, alpha=0.7)

    ax.axvline(DAM_BREACH, color="crimson", ls="--", lw=1.5, label="Руйнування дамби")
    ax.set_yticks(range(len(df["aoi_name"].unique())))
    ax.set_yticklabels(df["aoi_name"].unique())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Дата гранули")
    ax.set_title("Покриття гранулів SWOT по AOI та продукту")

    patches = [mpatches.Patch(color=color_map[p], label=p) for p in products]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}  ({len(df)} granules)")


# ── figure 6: station location map ───────────────────────────────────────────

def fig_station_map(stations_kakh: pd.DataFrame, stations_dnipro: pd.DataFrame,
                    out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Ukraine rough outline using key points
    ukraine_lons = [22, 40, 40, 38, 36, 34, 31, 22, 22]
    ukraine_lats = [44, 44, 52, 52, 48, 46, 46, 46, 44]
    ax.plot(ukraine_lons, ukraine_lats, "k-", lw=0.5, alpha=0.3)

    # AOI bounding boxes
    aoi_boxes = [
        ("Каховське\nвдх", 33.35, 46.75, 35.34, 47.78, "steelblue"),
        ("Дніпровське\nвдх", 34.40, 47.50, 35.60, 48.55, "darkorange"),
        ("Нижній Дніпро\nдо моря", 31.50, 46.20, 34.50, 47.50, "seagreen"),
    ]
    for label, w, s, e, n, color in aoi_boxes:
        rect = mpatches.FancyBboxPatch(
            (w, s), e - w, n - s,
            boxstyle="square,pad=0", linewidth=1.5,
            edgecolor=color, facecolor=color, alpha=0.08
        )
        ax.add_patch(rect)
        ax.text((w + e) / 2, n + 0.05, label, ha="center", fontsize=7.5,
                color=color, fontweight="bold")

    # Key landmarks
    landmarks = [
        ("Херсон",    32.682, 46.657),
        ("Каховка\n(дамба)", 33.37, 46.75),
        ("Запоріжжя", 35.13, 47.84),
        ("Дніпро",    35.04, 48.45),
        ("Чорне море\n(гирло)", 32.26, 46.53),
    ]
    for lname, llon, llat in landmarks:
        ax.plot(llon, llat, "k^", ms=7, zorder=5)
        ax.annotate(lname, (llon, llat), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5)

    # Gauge stations
    colors = {"Каховка": "steelblue", "Дніпровське": "darkorange"}
    for df, repo_name in [(stations_kakh, "Каховка"), (stations_dnipro, "Дніпровське")]:
        if df.empty:
            continue
        valid = df.dropna(subset=["lon", "lat"])
        if valid.empty:
            continue
        ax.scatter(valid["lon"], valid["lat"],
                   c=colors[repo_name], s=80, marker="o", zorder=6,
                   label=f"Гідропости ({repo_name})", edgecolors="white", lw=0.5)
        for _, row in valid.iterrows():
            ax.annotate(str(row.get("hydropost", "")),
                        (row["lon"], row["lat"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlim(30.5, 37)
    ax.set_ylim(45.8, 49.5)
    ax.set_xlabel("Довгота, °E")
    ax.set_ylabel("Широта, °N")
    ax.set_title("Карта: AOI зони SWOT та гідропости")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")


def fig_demo_swot_periods(out: Path) -> None:
    """Demo figure: synthetic water level curves showing 3 characteristic periods."""
    np.random.seed(42)
    dates_pre = pd.date_range("2023-04-01", "2023-06-05", freq="5D")
    dates_breach = pd.date_range("2023-06-06", "2023-07-31", freq="5D")
    dates_post = pd.date_range("2023-08-01", "2025-12-01", freq="10D")

    # Synthetic WSE: pre~14m, drops to ~0m, post~0.5m (post-breach Kakhovka)
    wse_pre = 13.5 + 0.3 * np.sin(np.linspace(0, np.pi, len(dates_pre))) \
              + np.random.normal(0, 0.1, len(dates_pre))
    wse_breach = np.linspace(13.5, 1.0, len(dates_breach)) \
                 + np.random.normal(0, 0.2, len(dates_breach))
    wse_post = 1.0 + 0.15 * np.sin(np.linspace(0, 6 * np.pi, len(dates_post))) \
               + np.random.normal(0, 0.08, len(dates_post))

    all_dates = list(dates_pre) + list(dates_breach) + list(dates_post)
    all_wse   = np.concatenate([wse_pre, wse_breach, wse_post])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates_pre, wse_pre, color="steelblue", lw=2, label="До руйнування (~13.5 м)")
    ax.plot(dates_breach, wse_breach, color="crimson", lw=2, label="Руйнування дамби (падіння)")
    ax.plot(dates_post, wse_post, color="darkorange", lw=1.5, label="Після (~1 м — заплава)")
    ax.axvline(DAM_BREACH, color="crimson", ls="--", lw=2)
    ax.annotate("06.06.2023\nРуйнування\nКаховської дамби",
                xy=(DAM_BREACH, 8), xytext=(DAM_BREACH + pd.Timedelta("30D"), 10),
                arrowprops=dict(arrowstyle="->", color="crimson"),
                fontsize=9, color="crimson")

    ax.set_xlabel("Дата")
    ax.set_ylabel("Рівень поверхні води (WSE), м")
    ax.set_title("DEMO — Каховське водосховище: SWOT WSE по трьох періодах\n"
                 "(синтетичні дані — буде замінено реальними після завантаження)")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_ylim(-0.5, 16)
    ax.text(0.98, 0.97, "⚠ DEMO DATA", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="gray",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="orange", alpha=0.8))
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== SWOT Water Level Visualization ===\n")

    # Load gauge station metadata
    print("[1/6] Loading gauge station metadata...")
    stations_kakh  = load_gauge_stations(GAUGE_KAKH)
    stations_dnipro = load_gauge_stations(GAUGE_DNIPRO)
    print(f"  Каховка: {len(stations_kakh)} stations, Дніпровське: {len(stations_dnipro)} stations")

    # Load gauge time series (if available)
    gauge_kakh  = load_gauge(GAUGE_KAKH,  "Каховка")
    gauge_dnipro = load_gauge(GAUGE_DNIPRO, "Дніпровське")

    # Try to load SWOT LakeSP data
    print("[2/6] Loading SWOT LakeSP data (Kakhovka)...")
    nc_files_kakh = find_nc_files(LAKESP_DIR)
    print(f"  Found {len(nc_files_kakh)} LakeSP NetCDF files")
    daily_kakh = load_lakesp_wse(nc_files_kakh)

    print("[3/6] Loading SWOT LakeSP data (Dniprovske)...")
    daily_dnipro = pd.DataFrame()  # separate AOI would need separate filter

    # Figure 1: SWOT time series Kakhovka
    print("[4/6] Plotting SWOT time series...")
    if not daily_kakh.empty:
        fig_swot_timeseries(daily_kakh, "Каховка",
                            FIG_DIR / "01_swot_timeseries_kakhovka.png")
    else:
        print("  No SWOT data yet — run: python -m swotdl download ...")
        # Create placeholder figure
        _placeholder("01_swot_timeseries_kakhovka.png",
                     "Дані ще не завантажені\nВиконайте: python -m swotdl download --aoi config/aoi/kakhovka_reservoir.geojson --product SWOT_L2_HR_LakeSP_2.0")

    # Figure 2: Gauge time series
    print("[5/6] Plotting gauge time series...")
    fig_gauge_timeseries(
        [("Каховка", gauge_kakh), ("Дніпровське", gauge_dnipro)],
        FIG_DIR / "03_gauge_timeseries.png"
    )

    # Figure 3: Period comparison
    if not daily_kakh.empty:
        fig_period_boxplot(daily_kakh, "Каховка",
                           FIG_DIR / "04_period_comparison_boxplot.png")

    # Figure 4: SWOT vs gauge
    if not daily_kakh.empty and not gauge_kakh.empty:
        fig_swot_vs_gauge(daily_kakh, gauge_kakh,
                          "Каховка (SWOT)", "Каховка (гідропост)",
                          FIG_DIR / "05_swot_vs_gauge.png")

    # Figure 5: Granule coverage
    print("[6/6] Plotting granule coverage...")
    fig_granule_coverage(FIG_DIR / "06_granule_coverage.png")

    # Station map
    print("[+] Plotting station map...")
    fig_station_map(stations_kakh, stations_dnipro,
                    FIG_DIR / "00_station_map.png")

    # Demo SWOT periods
    print("[+] Plotting demo SWOT period figure...")
    if daily_kakh.empty:
        fig_demo_swot_periods(FIG_DIR / "01_swot_demo_periods.png")

    print(f"\nDone. Figures saved to: {FIG_DIR}")


def _placeholder(filename: str, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107"))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=100)
    plt.close()
    print(f"  Placeholder saved: {filename}")


if __name__ == "__main__":
    main()
