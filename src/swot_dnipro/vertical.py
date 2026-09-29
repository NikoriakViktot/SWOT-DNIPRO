"""The SWOT / ICESat-2 / gauge vertical chains, in one place.

Established by the audit in ``outputs/reports/prebreach_swot_validation_pilot.md``:

SWOT PIXC
    ``height`` is a **raw ellipsoidal** height (WGS 84). ``geoid``,
    ``solid_earth_tide``, ``load_tide_fes`` and ``pole_tide`` are each
    *"reported for reference but not applied to the reported height"*
    (PIXC PDD D-56411 Rev C, p. 53). Therefore::

        h_SWOT = height - solid_earth_tide - load_tide_fes - pole_tide

    Sign verified empirically against RiverSP on the same cycle/pass
    (482/001, n = 1023 nodes): median difference **-0.0008 m**, versus
    +0.0721 m (no correction), +0.1450 m (wrong sign), -23.89 m (geoid
    subtracted). ``geoid`` must NOT be subtracted from PIXC ``height``.

    Because ``solid_earth_tide`` excludes the zero-frequency permanent tide,
    ``h_SWOT`` retains the permanent crustal deformation -> **mean-tide crust**,
    which is the convention a zero-tide quasigeoid expects.

ICESat-2 ATL13
    ``ht_water_surf`` is ellipsoidal with the solid-Earth, load and pole tides
    already applied upstream in ATL03, realised in the **tide-free** system
    (ATL03 ATBD v007, pp. 8, 20, 26). To reach the EGG2015 convention add
    ``tide_earth_free2mean`` (ATBD p. 125).

Both branches then use the same quasigeoid::

    H_EGG2015 = h - zeta_EGG2015(lat, lon)

EPSG:9902 maps BS-77 *normal* height -> EVRF2019 *normal* height. It is not a
quasigeoid and is never applied to an ellipsoidal or geoid-referenced height.
"""
from __future__ import annotations

import numpy as np

from .config import PIXC_WATER_CLASSES_PRIMARY, free2mean

PIXC_TIDE_FIELDS = ("solid_earth_tide", "load_tide_fes", "pole_tide")


def pixc_ellipsoidal_height(height, solid_earth_tide, load_tide_fes, pole_tide):
    """Tide-corrected ellipsoidal height from PIXC (mean-tide crust)."""
    t = (
        np.nan_to_num(solid_earth_tide)
        + np.nan_to_num(load_tide_fes)
        + np.nan_to_num(pole_tide)
    )
    return np.asarray(height, float) - t


def atl13_to_mean_tide(h_tide_free, lat_deg):
    """ATL13 tide-free ellipsoidal height -> mean/zero-tide crust convention."""
    return np.asarray(h_tide_free, float) + free2mean(np.asarray(lat_deg, float))


def to_egg2015(h_ellipsoidal, zeta_egg2015):
    """Normal-height estimate in the EGG2015 frame."""
    return np.asarray(h_ellipsoidal, float) - np.asarray(zeta_egg2015, float)


def gauge_evrf2019(stage_m, gauge_zero_bs77_m, delta_epsg9902_m):
    """Gauge branch: BS-77 stage -> EVRF2019 normal height (EPSG:9389)."""
    return (
        np.asarray(stage_m, float)
        + float(gauge_zero_bs77_m)
        + np.asarray(delta_epsg9902_m, float)
    )


# --------------------------------------------------------------------------- #
# PIXC reading                                                                 #
# --------------------------------------------------------------------------- #
def read_pixc(path, bbox=None, water_classes=PIXC_WATER_CLASSES_PRIMARY):
    """Read one PIXC granule into a dict of arrays, optionally clipped.

    Returns ``lat, lon, height, geoid, classification, tides, h_ell`` plus the
    granule's ``cycle``, ``pass``, ``tile`` and ``time_granule_start``.
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(path))
    pc = ds.groups["pixel_cloud"]

    def g(name):
        return np.ma.filled(pc.variables[name][:].astype("f8"), np.nan)

    lat, lon = g("latitude"), g("longitude")
    height = g("height")
    cls = pc.variables["classification"][:].astype(int)
    st, lt, pt = (g(k) for k in PIXC_TIDE_FIELDS)

    m = np.isfinite(height)
    if bbox is not None:
        w, s, e, n = bbox
        m &= (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    if water_classes is not None:
        m &= np.isin(cls, list(water_classes))

    out = {
        "lat": lat[m], "lon": lon[m], "height": height[m],
        "geoid": g("geoid")[m], "classification": cls[m],
        "solid_earth_tide": st[m], "load_tide_fes": lt[m], "pole_tide": pt[m],
        "water_frac": g("water_frac")[m] if "water_frac" in pc.variables else None,
        "h_ell": pixc_ellipsoidal_height(height, st, lt, pt)[m],
        "cycle": int(ds.getncattr("cycle_number")),
        "pass": int(ds.getncattr("pass_number")),
        "tile": str(ds.getncattr("tile_name")),
        "time_granule_start": str(ds.getncattr("time_granule_start")),
        "n_total": int(m.size), "n_kept": int(m.sum()),
    }
    ds.close()
    return out


def sample_grid(tif_path, lon, lat):
    """Sample a raster (EGG2015 quasigeoid or the EPSG:9902 grid) at points."""
    import rasterio

    with rasterio.open(str(tif_path)) as src:
        return np.array([v[0] for v in src.sample(list(zip(np.asarray(lon), np.asarray(lat))))],
                        dtype=float)


def sample_grid_bilinear(tif_path, lon, lat):
    """Bilinear sample of a geographic grid at points (cell values at centres).

    ``sample_grid`` takes the nearest cell. On the 1' EGG2015 grid neighbouring
    cells differ by 1.6 cm at the median (p99 11.5 cm) over the study area, so
    two points 200 m apart that straddle a cell edge get a centimetre step that
    the surface does not have. Differences between nearby points (satellite
    crossings) must therefore interpolate; this does, and leaves
    ``sample_grid`` unchanged for the code that already uses it.
    """
    import rasterio
    from scipy.ndimage import map_coordinates

    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    with rasterio.open(str(tif_path)) as src:
        t = src.transform
        c0, r1 = ~t * (lon.min(), lat.min())
        c1, r0 = ~t * (lon.max(), lat.max())
        win = rasterio.windows.Window(int(np.floor(min(c0, c1))) - 2, int(np.floor(min(r0, r1))) - 2,
                                      int(abs(c1 - c0)) + 5, int(abs(r1 - r0)) + 5)
        a = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
        wt = src.window_transform(win)
    cols, rows = ~wt * (lon, lat)
    return map_coordinates(a, [np.asarray(rows) - 0.5, np.asarray(cols) - 0.5],
                           order=1, mode="nearest")


def haversine_km(lon, lat, lon0, lat0):
    """Great-circle distance in km (adequate for <100 km neighbourhood work)."""
    R = 6371.0088
    p1, p2 = np.radians(lat), np.radians(lat0)
    dp = p2 - p1
    dl = np.radians(lon0 - np.asarray(lon))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
