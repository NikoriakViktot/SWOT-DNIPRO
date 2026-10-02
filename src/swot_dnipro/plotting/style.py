"""Shared figure style, colour policy and map furniture.

One style for the whole project so figures read as a single system.
Every figure is saved as both vector PDF and 400 dpi PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Colour policy — one colour per observation system, used everywhere           #
# --------------------------------------------------------------------------- #
C = {
    "swot": "#1b6ca8",        # SWOT — blue
    "icesat": "#c0392b",      # ICESat-2 — red
    "gauge": "#2d3436",       # gauge — near-black
    "reservoir": "#a8cfe0",   # former reservoir polygon fill
    "river": "#4a90c2",       # Dnipro channel
    "dam": "#7d3c98",         # dam
    "accent": "#e08214",      # highlight / preferred variant
    "muted": "#95a5a6",       # rejected / context
    "grid": "#d8dbdd",
    "bad": "#b03a2e",
    "ok": "#1e8449",
}

MARKERS = {"swot": "o", "icesat": "^", "gauge": "s", "dam": "*"}


def use_style() -> None:
    """Apply the project-wide matplotlib style. Call once per figure script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#33393d",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": C["grid"],
            "grid.linewidth": 0.5,
            "legend.fontsize": 7.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#c8cccf",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,   # embed as TrueType -> editable/selectable text
            "ps.fonttype": 42,
        }
    )


def save(fig, stem: str, outdir: Path) -> list[Path]:
    """Save *fig* as ``<stem>.pdf`` (vector) and ``<stem>.png`` (400 dpi)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def panel_label(ax, letter: str, dx: float = 0.012, dy: float = 0.985) -> None:
    """Bold panel label (a), (b), ... in the top-left of *ax*."""
    ax.text(
        dx, dy, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82),
    )


# --------------------------------------------------------------------------- #
# Map furniture — metric CRS assumed (axes in metres)                          #
# --------------------------------------------------------------------------- #
def scale_bar(ax, length_km: float | None = None, loc: str = "lower left",
              pad_frac: float = 0.05) -> None:
    """Draw a metric scale bar. Requires axes in projected metres."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = x1 - x0
    if length_km is None:
        raw = span * 0.22 / 1000.0
        nice = np.array([1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500])
        length_km = float(nice[np.argmin(np.abs(nice - raw))])
    L = length_km * 1000.0
    px, py = pad_frac * span, pad_frac * (y1 - y0)
    bx = x0 + px if "left" in loc else x1 - px - L
    by = y0 + py if "lower" in loc else y1 - py
    h = (y1 - y0) * 0.008
    # two-tone bar
    ax.add_patch(mpl.patches.Rectangle((bx, by), L / 2, h, fc="black", ec="black", lw=0.5, zorder=9))
    ax.add_patch(mpl.patches.Rectangle((bx + L / 2, by), L / 2, h, fc="white", ec="black", lw=0.5, zorder=9))
    ax.text(bx, by + h * 1.5, "0", ha="center", va="bottom", fontsize=6.5, zorder=9)
    ax.text(bx + L, by + h * 1.5, f"{length_km:g} km", ha="center", va="bottom", fontsize=6.5, zorder=9)


def north_arrow(ax, loc=(0.955, 0.93), size: float = 0.055) -> None:
    """Simple north arrow. Valid because UTM grid north ≈ true north here."""
    x, y = loc
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction",
        textcoords="axes fraction", ha="center", va="center",
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1),
        zorder=10,
    )


def graticule(ax, gdf_crs_metric, step: float = 0.5) -> None:
    """Label a projected axis with lon/lat ticks (degrees) instead of metres."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", gdf_crs_metric, always_xy=True)
    inv = Transformer.from_crs(gdf_crs_metric, "EPSG:4326", always_xy=True)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lon0, lat0 = inv.transform(x0, y0)
    lon1, lat1 = inv.transform(x1, y1)
    lons = np.arange(np.floor(lon0 / step) * step, lon1 + step, step)
    lats = np.arange(np.floor(lat0 / step) * step, lat1 + step, step)
    xt = [tf.transform(lo, (lat0 + lat1) / 2)[0] for lo in lons]
    yt = [tf.transform((lon0 + lon1) / 2, la)[1] for la in lats]
    keep_x = [(t, lo) for t, lo in zip(xt, lons) if x0 < t < x1]
    keep_y = [(t, la) for t, la in zip(yt, lats) if y0 < t < y1]
    ax.set_xticks([t for t, _ in keep_x])
    ax.set_xticklabels([f"{lo:.1f}°E" for _, lo in keep_x])
    ax.set_yticks([t for t, _ in keep_y])
    ax.set_yticklabels([f"{la:.1f}°N" for _, la in keep_y])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def zero_line(ax, y: float = 0.0, **kw) -> None:
    kw = {"color": "k", "lw": 0.8, "ls": "--", "alpha": 0.7, "zorder": 1, **kw}
    ax.axhline(y, **kw)


def nmad(x) -> float:
    """Normalised median absolute deviation — robust 1σ equivalent."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def bootstrap_ci(x, n: int = 10000, seed: int = 42, q=(2.5, 97.5)):
    """Percentile bootstrap CI of the median over independent samples."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    med = [np.median(rng.choice(x, x.size, replace=True)) for _ in range(n)]
    return tuple(float(v) for v in np.percentile(med, q))
