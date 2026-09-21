"""DEM sources: ArcticDEM (PGC STAC + COG) with a Copernicus GLO-30 fallback.

Purpose
-------
Fetch an elevation model covering a glacier's buffered analysis window, in
EPSG:3413, from one of two sources:

1. **ArcticDEM v4.1 10 m mosaics**, queried through the PGC STAC API and read
   as windowed COG reads, rather than through a bundled tile index: the STAC
   API stays current as ArcticDEM releases change, and a windowed read avoids
   pulling a full mosaic tile just to crop it down to one glacier's window
   (see ``docs/decisions/dem_backend.md``).
2. **Copernicus DEM GLO-30**, read from the public AWS COG bucket, used when
   ArcticDEM has no coverage or returns a near-empty window (e.g. south of the
   ArcticDEM footprint, which matters for RGI region 02).

Inputs
------
- A bounding box in EPSG:3413 (the buffered, squared glacier window).

Outputs
-------
- An ``xarray.DataArray`` (float32, dims ``y``/``x``, CRS EPSG:3413, NaN
  nodata), or ``None`` when the source has no usable coverage.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import numpy as np
import rasterio
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

#: Working CRS for all DEM products (NSIDC Sea Ice Polar Stereographic North).
TARGET_EPSG = 3413

#: PGC STAC API root. A static S3 ``catalog.json`` endpoint was checked and
#: found dead (the key returns NoSuchKey); the maintained entry point is the
#: STAC API below.
PGC_STAC_URL = "https://stac.pgc.umn.edu/api/v1"

#: ArcticDEM release. Verified current against the live PGC catalog: the
#: available mosaic collections are v3.0 and v4.1, so v4.1 is newest.
ARCTICDEM_VERSION = "v4.1"
ARCTICDEM_RESOLUTION = "10m"
ARCTICDEM_COLLECTION = f"arcticdem-mosaics-{ARCTICDEM_VERSION}-{ARCTICDEM_RESOLUTION}"
ARCTICDEM_BASE_RES_M = 10

COPDEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
COPDEM_RESOLUTION_M = 30

#: Minimum fraction of finite pixels in the fetched window for a source to be
#: considered usable. Deliberately low: this only screens out windows that are
#: essentially empty (e.g. entirely outside a DEM's footprint); the >50%
#: zero-pixel rejection in ``dem.processing`` does the real quality gating
#: after masking.
MIN_VALID_PCT = 1.0


def configure_gdal_for_public_cogs() -> None:
    """Set the GDAL environment for anonymous, range-request COG access.

    Both the PGC and Copernicus buckets are public: signing must be disabled,
    and directory listing suppressed so GDAL does not issue a LIST per open.

    ``GDAL_HTTP_TIMEOUT``/``GDAL_HTTP_CONNECTTIMEOUT`` are the critical ones:
    without them, GDAL's underlying curl handle has **no timeout at all** on
    a stalled or slow-to-respond HTTP range request, so a single flaky
    request against the public S3/AWS buckets hangs the whole process
    indefinitely -- confirmed live in this session (a Copernicus DEM fetch
    stalled with 0% CPU for several minutes, invisible to Python-level
    signal handling, and not reproducible as a normal slow-but-finite
    request). ``GDAL_HTTP_MAX_RETRY``/``GDAL_HTTP_RETRY_DELAY`` already
    existed but cannot help if the initial connection itself never times
    out to trigger a retry.
    """
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", "50000000")


def _array_from_window(
    src: rasterio.DatasetReader,
    bounds: tuple[float, float, float, float],
) -> Optional[xr.DataArray]:
    """Read a windowed slice of an open dataset into a georeferenced DataArray."""
    window = window_from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return None

    data = src.read(1, window=window, masked=True, boundless=True, fill_value=np.nan)
    arr = np.asarray(data.filled(np.nan), dtype=np.float32)
    if arr.size == 0:
        return None

    # Some tiles declare a sentinel nodata (e.g. -9999) that survives masking
    # when boundless reads pad outside the tile.
    if src.nodata is not None and np.isfinite(src.nodata):
        arr[arr == np.float32(src.nodata)] = np.nan

    transform = src.window_transform(window)
    return _wrap_array(arr, transform, src.crs.to_epsg() or TARGET_EPSG)


def _wrap_array(arr: np.ndarray, transform, epsg: int) -> xr.DataArray:
    """Wrap a numpy array + affine transform into a CRS-aware DataArray."""
    height, width = arr.shape
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    da = xr.DataArray(arr, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da.rio.write_crs(epsg, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    return da


def valid_fraction_pct(da: xr.DataArray) -> float:
    """Percentage of finite pixels in ``da``."""
    values = np.asarray(da.values)
    return 100.0 * float(np.isfinite(values).mean()) if values.size else 0.0


#: (connect, read) timeout in seconds for the PGC STAC API's HTTP requests.
#: pystac_client/requests otherwise have no default timeout, so a stalled
#: connection to the STAC API hangs the whole process indefinitely -- the
#: same class of bug fixed for GDAL's COG reads in
#: :func:`configure_gdal_for_public_cogs`, confirmed live in this session.
STAC_HTTP_TIMEOUT = (10.0, 30.0)


def search_arcticdem_items(
    bounds_3413: tuple[float, float, float, float],
    stac_url: str = PGC_STAC_URL,
    collection: str = ARCTICDEM_COLLECTION,
) -> list:
    """Return the STAC items of the ArcticDEM mosaic tiles covering ``bounds_3413``.

    The PGC STAC API indexes items in WGS84, so the EPSG:3413 window is
    densified and transformed to a lon/lat bbox for the search.
    """
    from pystac_client import Client

    bbox_4326 = transform_bounds(
        f"EPSG:{TARGET_EPSG}", "EPSG:4326", *bounds_3413, densify_pts=21
    )
    client = Client.open(stac_url, timeout=STAC_HTTP_TIMEOUT)
    search = client.search(collections=[collection], bbox=list(bbox_4326))
    return list(search.items())


def load_arcticdem(
    bounds_3413: tuple[float, float, float, float],
    stac_url: str = PGC_STAC_URL,
    collection: str = ARCTICDEM_COLLECTION,
    asset_key: str = "dem",
) -> Optional[xr.DataArray]:
    """Load the ArcticDEM 10 m mosaic over ``bounds_3413`` via STAC + windowed COGs.

    The ArcticDEM mosaics are natively in EPSG:3413 at 10 m, which is exactly
    the target grid, so no reprojection is needed here -- only a windowed read
    and, when the window straddles a tile boundary, a mosaic of the per-tile
    windows.

    Returns ``None`` if no tile covers the window or nothing valid was read.
    """
    configure_gdal_for_public_cogs()

    items = search_arcticdem_items(bounds_3413, stac_url=stac_url, collection=collection)
    if not items:
        return None

    hrefs = [
        item.assets[asset_key].href
        for item in items
        if asset_key in item.assets and item.assets[asset_key].href
    ]
    if not hrefs:
        return None

    if len(hrefs) == 1:
        with rasterio.open(hrefs[0]) as src:
            return _array_from_window(src, bounds_3413)

    return _merge_cog_tiles(hrefs, bounds_3413, resolution=ARCTICDEM_BASE_RES_M)


def _merge_cog_tiles(
    hrefs: Sequence[str],
    bounds: tuple[float, float, float, float],
    resolution: Optional[float] = None,
    fallback_epsg: int = TARGET_EPSG,
) -> Optional[xr.DataArray]:
    """Mosaic several remote COG tiles over a common bbox into one ``DataArray``.

    Shared by :func:`load_arcticdem`'s multi-tile path (``resolution`` given,
    stays in the tiles' native CRS -- ArcticDEM's EPSG:3413) and
    :func:`load_copernicus_dem`'s geographic mosaic (``resolution=None``, lets
    ``rasterio.merge`` infer it from the tiles -- Copernicus GLO-30's WGS84
    grid). Both need the same open/merge/wrap/close-defensively shape; only
    the resolution argument and the CRS actually differ between them.

    Tiles that fail to open (e.g. an ocean tile absent from the bucket) are
    skipped rather than aborting the whole mosaic.
    """
    datasets = []
    try:
        for href in hrefs:
            try:
                datasets.append(rasterio.open(href))
            except Exception:
                continue
        if not datasets:
            return None

        merge_kwargs = {"bounds": bounds, "nodata": np.nan}
        if resolution is not None:
            merge_kwargs["res"] = resolution
        merged, transform = rio_merge(datasets, **merge_kwargs)
        if merged is None or merged.size == 0:
            return None

        arr = np.asarray(merged[0], dtype=np.float32)
        epsg = datasets[0].crs.to_epsg() or fallback_epsg
        return _wrap_array(arr, transform, epsg)
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def _copdem_tile_name(lat_deg: int, lon_deg: int) -> str:
    lat_tag = f"{'N' if lat_deg >= 0 else 'S'}{abs(lat_deg):02d}_00"
    lon_tag = f"{'E' if lon_deg >= 0 else 'W'}{abs(lon_deg):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat_tag}_{lon_tag}_DEM"


def copdem_tile_url(lat_deg: int, lon_deg: int, base_url: str = COPDEM_BASE_URL) -> str:
    """Return the public AWS COG URL of one Copernicus GLO-30 1-degree tile."""
    name = _copdem_tile_name(lat_deg, lon_deg)
    return f"{base_url}/{name}/{name}.tif"


def copdem_tile_indices(
    bounds_4326: tuple[float, float, float, float],
) -> tuple[range, range]:
    """Return the (latitude, longitude) integer tile ranges covering a lon/lat bbox."""
    minx, miny, maxx, maxy = bounds_4326
    lon0 = max(-180, int(math.floor(minx)))
    lon1 = min(179, int(math.ceil(maxx)) - 1)
    lat0 = max(-90, int(math.floor(miny)))
    lat1 = min(89, int(math.ceil(maxy)) - 1)
    if lon1 < lon0 or lat1 < lat0:
        return range(0), range(0)
    return range(lat0, lat1 + 1), range(lon0, lon1 + 1)


def load_copernicus_dem(
    bounds_3413: tuple[float, float, float, float],
    base_url: str = COPDEM_BASE_URL,
) -> Optional[xr.DataArray]:
    """Load Copernicus DEM GLO-30 over ``bounds_3413``, reprojected to EPSG:3413.

    The tiles are 1-degree COGs on a geographic grid, so the window must be
    expressed in lon/lat, mosaicked there, then reprojected to the
    polar-stereographic target grid -- unlike ArcticDEM, which is already
    natively on that grid.
    """
    configure_gdal_for_public_cogs()

    bounds_4326 = transform_bounds(
        f"EPSG:{TARGET_EPSG}", "EPSG:4326", *bounds_3413, densify_pts=21
    )
    lat_range, lon_range = copdem_tile_indices(bounds_4326)
    if len(lat_range) == 0 or len(lon_range) == 0:
        return None

    hrefs = [
        copdem_tile_url(lat, lon, base_url) for lat in lat_range for lon in lon_range
    ]
    dem_wgs84 = _merge_cog_tiles(hrefs, bounds_4326, fallback_epsg=4326)
    if dem_wgs84 is None:
        return None

    dem_3413 = dem_wgs84.rio.reproject(
        dst_crs=f"EPSG:{TARGET_EPSG}",
        resolution=COPDEM_RESOLUTION_M,
        resampling=Resampling.bilinear,
        nodata=np.nan,
    )
    dem_3413.rio.write_nodata(np.nan, inplace=True)
    return dem_3413.astype("float32")


def load_best_dem(
    bounds_3413: tuple[float, float, float, float],
    use_copernicus_fallback: bool = True,
    stac_url: str = PGC_STAC_URL,
    collection: str = ARCTICDEM_COLLECTION,
    force_copernicus: bool = False,
) -> tuple[Optional[xr.DataArray], str, int, float]:
    """Try ArcticDEM first, then Copernicus GLO-30.

    Returns ``(dem, source_label, base_resolution_m, valid_pct)``. ``dem`` is
    ``None`` when neither source yields a window with at least
    ``MIN_VALID_PCT`` finite pixels.

    ``force_copernicus`` skips ArcticDEM entirely and goes straight to
    Copernicus GLO-30, even where ArcticDEM would succeed. This exists so the
    fallback path can be exercised and verified on demand (e.g. by the
    integration fixture), rather than only running on glaciers that happen to
    fall outside ArcticDEM's footprint.
    """
    if not force_copernicus:
        try:
            dem = load_arcticdem(bounds_3413, stac_url=stac_url, collection=collection)
        except Exception:
            dem = None

        if dem is not None:
            pct = valid_fraction_pct(dem)
            if pct >= MIN_VALID_PCT:
                label = f"ArcticDEM {ARCTICDEM_VERSION} {ARCTICDEM_RESOLUTION} (PGC STAC/COG)"
                return dem, label, ARCTICDEM_BASE_RES_M, pct

    if not use_copernicus_fallback and not force_copernicus:
        return None, "", 0, 0.0

    try:
        dem = load_copernicus_dem(bounds_3413)
    except Exception:
        dem = None

    if dem is None:
        return None, "", 0, 0.0

    pct = valid_fraction_pct(dem)
    if pct < MIN_VALID_PCT:
        return None, "", 0, pct
    return dem, "Copernicus DEM GLO-30 (AWS COG)", COPDEM_RESOLUTION_M, pct
