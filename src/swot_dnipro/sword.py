"""SWORD v16 access — authoritative Dnipro reach topology and river chainage.

SWORD (SWOT River Database) is the prior river network SWOT itself uses, so using it
for chainage keeps our longitudinal coordinate consistent with the SWOT products.

Chainage here is SWORD's own ``dist_out`` (distance from the river outlet, m),
converted to km and re-referenced to the Kakhovka dam. That is a genuine
along-network distance, not a projection of longitude.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_nodes(nc_path: Path, lon_range=(32.0, 36.0), lat_range=(46.0, 48.5)) -> pd.DataFrame:
    """Load SWORD nodes within a bounding box.

    SWORD netCDF groups: ``/nodes`` and ``/reaches``. Node variables used:
    ``x``, ``y``, ``node_id``, ``reach_id``, ``dist_out``, ``width``, ``river_name``.
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path))
    g = ds.groups["nodes"]

    def arr(name):
        if name not in g.variables:
            return None
        v = np.ma.filled(g.variables[name][:], np.nan)
        return v

    lon, lat = arr("x"), arr("y")
    m = ((lon >= lon_range[0]) & (lon <= lon_range[1]) &
         (lat >= lat_range[0]) & (lat <= lat_range[1]))
    out = {"lon": lon[m], "lat": lat[m]}
    for name in ("node_id", "reach_id", "dist_out", "width", "wse", "n_chan_max",
                 "facc", "type"):
        v = arr(name)
        if v is not None and v.ndim == 1 and len(v) == len(lon):
            out[name] = v[m]
    rn = g.variables.get("river_name")
    if rn is not None:
        try:
            names = nc.chartostring(rn[:]) if rn.ndim > 1 else rn[:]
            out["river_name"] = np.asarray(names, dtype=str)[m]
        except Exception:
            pass
    ds.close()
    df = pd.DataFrame(out)
    for c in ("node_id", "reach_id"):
        if c in df:
            df[c] = df[c].astype("int64", errors="ignore")
    return df


def dnipro_nodes(nodes: pd.DataFrame, name_hint=("Dnipro", "Dnieper", "Дніпро")) -> pd.DataFrame:
    """Select the Dnipro main stem.

    Prefer the SWORD ``river_name`` attribute; fall back to flow accumulation, which
    is far larger on the main stem than on any tributary in this basin.
    """
    if "river_name" in nodes:
        m = nodes.river_name.astype(str).str.contains("|".join(name_hint), case=False, na=False)
        if m.sum() > 50:
            return nodes[m].copy()
    if "facc" in nodes:
        thr = np.nanpercentile(nodes.facc, 97)
        return nodes[nodes.facc >= thr].copy()
    return nodes.copy()


def chainage_from_dam(nodes: pd.DataFrame, dam_lon: float, dam_lat: float) -> pd.DataFrame:
    """Add ``chain_km`` = SWORD dist_out re-referenced to the dam (positive upstream)."""
    if "dist_out" not in nodes:
        raise RuntimeError("SWORD nodes lack dist_out")
    d = nodes.dropna(subset=["dist_out"]).copy()
    j = np.argmin((d.lon - dam_lon) ** 2 + (d.lat - dam_lat) ** 2)
    dam_dist = d.dist_out.iloc[j]
    d["chain_km"] = (d.dist_out - dam_dist) / 1000.0
    return d


def build_chainage_tree(channel: pd.DataFrame):
    """Build the KD-tree used by :func:`assign_chainage` once, for reuse.

    Returns ``(tree, kx)``, which can be handed back via the *tree* argument of
    :func:`assign_chainage`. Callers that snap many point sets against the same
    channel (e.g. one per connected component) should build it once — the tree
    depends only on *channel*.
    """
    from scipy.spatial import cKDTree

    lat0 = float(np.mean(channel.lat))
    kx = 111.32 * np.cos(np.radians(lat0))
    return cKDTree(np.c_[channel.lon * kx, channel.lat * 110.57]), kx


def assign_chainage(points_lon, points_lat, channel: pd.DataFrame,
                    max_dist_km: float = 50.0, tree=None):
    """Snap arbitrary points to the nearest channel node; return chainage + distance.

    Returns ``(chain_km, dist_to_channel_km, node_id, reach_id)``; entries beyond
    *max_dist_km* are NaN so nothing is silently attributed to the main stem.

    *tree* is an optional ``(tree, kx)`` pair from :func:`build_chainage_tree`.
    Passing it skips rebuilding the KD-tree and is numerically identical.
    """
    if tree is None:
        tree, kx = build_chainage_tree(channel)
    else:
        tree, kx = tree
    q = np.c_[np.asarray(points_lon) * kx, np.asarray(points_lat) * 110.57]
    dist, idx = tree.query(q, k=1)
    chain = channel.chain_km.to_numpy()[idx]
    nid = channel.node_id.to_numpy()[idx] if "node_id" in channel else np.full(len(idx), -1)
    rid = channel.reach_id.to_numpy()[idx] if "reach_id" in channel else np.full(len(idx), -1)
    bad = dist > max_dist_km
    chain = np.where(bad, np.nan, chain)
    dist = np.where(bad, np.nan, dist)
    return chain, dist, nid, rid
