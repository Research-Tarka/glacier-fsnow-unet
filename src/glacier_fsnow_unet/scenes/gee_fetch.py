"""Fetch one scene's pixels from Earth Engine and write it into the glacier zarr store.

Purpose
-------
The missing link between scene *discovery* (``scenes.download.search_scenes``/
``filter_scenes``, which only query GEE metadata) and the zarr *write* API
(``scenes.zarr_store.append_scene``, which only accepts already-in-memory
arrays): actually pull the TOA band pixels for one selected scene over one
glacier's analysis window, convert them to the pipeline's band/format
conventions, and append them to the glacier's store.

Inputs
------
- An initialized ``ee`` module, a :class:`~.download.SceneRecord`, the
  sensor's :class:`~.sensors.SensorSpec`, and the glacier's window bounds.

Outputs
-------
- One call to :func:`~.zarr_store.append_scene`.

How pixels are pulled
----------------------
``ee.Image.getDownloadURL`` on the clipped, band-selected image returns a
GeoTIFF (or a zip of per-band GeoTIFFs) over HTTP; the response is read
directly from memory via GDAL's ``/vsimem/`` virtual filesystem, with no
GeoTIFF ever touching local disk -- this fetch stage produces zarr, like every
other raster product in this pipeline (see ``docs/decisions/dem_backend.md``).
``LANDSAT/.../T1_TOA`` and ``COPERNICUS/S2_HARMONIZED`` already deliver
calibrated top-of-atmosphere reflectance (Landsat) / scaled reflectance
(Sentinel-2, ``DN / 10000``) band values, so no manual gain/offset or solar-
angle correction is applied here; :func:`~.spectral_indices.sentinel2_toa_reflectance`
is used for the S2 DN scaling, and Landsat bands are used as GEE returns them.

The request region is given in **EPSG:3413**, not EPSG:4326, and this matters:
verified live, a lon/lat rectangle (``ee.Geometry.Rectangle(bounds, 'EPSG:4326')``)
at ~60 degN reprojects into EPSG:3413 as a visibly non-rectangular quadrilateral
(polar stereographic distorts a geographic bounding box significantly at high
latitude), which shifted the fetched grid's origin by roughly 1.5 km relative
to the DEM's window for the same glacier -- not a rounding error, a real
misalignment. Requesting the region directly in EPSG:3413 (the same CRS and
the same window ``dem.engine.window_polygon_from_registry`` already builds)
avoids the distortion entirely and was confirmed, live, to land within one
pixel of the DEM's window corner.

RGB rendering: an honest simplification
----------------------------------------
``rgb_raw`` is a straightforward per-band 2nd-98th percentile stretch of the
visible RGB bands. ``rgb_shadow`` applies the same stretch after boosting
shadow/low-reflectance pixels with a fixed gamma curve (``value ** 0.6``),
which brightens dark terrain (cast shadow, crevasses) without needing a DEM-
aware hillshade correction. This is **not** a port of any pre-existing
shadow-enhancement algorithm -- it is a new, simple, from-scratch rendering
chosen because a full radiometric/illumination-corrected renderer was out of
scope for this session. It exists so ``rgb_raw``/``rgb_shadow`` are genuinely
populated and visually inspectable, not because it reproduces a specific
published method. Replacing it with a more sophisticated renderer later needs
no change to the zarr schema.
"""

from __future__ import annotations

import numpy as np

from .sensors import SensorSpec

#: Percentile stretch bounds for RGB rendering.
_STRETCH_LOW_PCT = 2.0
_STRETCH_HIGH_PCT = 98.0
#: Gamma applied to brighten shadows in the "shadow-enhanced" composite.
_SHADOW_GAMMA = 0.6


#: The common working CRS every scene is requested in (matches the DEM's).
TARGET_EPSG = "EPSG:3413"


def reflectance_sanity_check(
    ee_module,
    collection: str,
    scene_id: str,
    bands: list[str],
    window_bounds_3413: tuple[float, float, float, float],
    scale_m: float,
    min_max_reflectance: float = 0.15,
) -> tuple[bool, dict]:
    """Reject a scene whose AOI reflectance is implausible, before downloading it.

    The scene-level cloud-cover property from ``search_scenes`` is a whole-
    footprint average and can hide a genuinely bad acquisition over one
    glacier's small AOI -- e.g. a WRS-2 path/row edge overlap where the AOI
    sits mostly outside the actual swath, which reads back as near-zero or
    negative "reflectance" once GEE fills the gap. This does one small
    ``reduceRegion(minMax())`` server-side call (no pixels downloaded) and
    rejects the scene if the brightest pixel over the AOI is not at least
    ``min_max_reflectance`` -- real glacier ice/snow is bright; a scene that
    fails this is not worth spending a download on.

    ``window_bounds_3413`` should be the same EPSG:3413 analysis-window bounds
    passed to :func:`fetch_and_store_scene`, for the same reason given there:
    a lon/lat rectangle distorts significantly on reprojection at high
    latitude.

    Returns ``(ok, stats)`` where ``stats`` is the raw min/max per band, for
    logging.
    """
    region = ee_module.Geometry.Rectangle(list(window_bounds_3413), TARGET_EPSG, False)
    image = ee_module.Image(f"{collection}/{scene_id}")
    stats = image.select(bands).reduceRegion(
        reducer=ee_module.Reducer.minMax(), geometry=region, scale=scale_m, maxPixels=1e9
    ).getInfo()

    max_values = [v for k, v in stats.items() if k.endswith("_max") and v is not None]
    if not max_values:
        return False, stats
    return max(max_values) >= min_max_reflectance, stats


def _download_geotiff_bytes(
    ee_image, region, scale_m: float, bands: list[str], crs: str = TARGET_EPSG
) -> bytes:
    """Fetch one Earth Engine image as raw GeoTIFF bytes via ``getDownloadURL``.

    ``crs`` is passed explicitly (alongside ``region`` and ``scale``) so
    every scene of a glacier is reprojected onto the *same* target CRS at
    request time, rather than each landing in its own source
    image's native UTM zone (which differs between orbit passes / WRS-2
    path-rows and would otherwise violate ``zarr_store.append_scene``'s "every
    scene of a glacier must share the same grid" invariant). Verified live:
    two different Landsat 8/9 acquisitions of the same glacier window, fetched
    this way, come back with byte-identical shape and affine transform.

    An earlier version of this function tried to pin the grid with
    ``crsTransform``/``dimensions`` instead of ``region``/``scale``/``crs``;
    Earth Engine silently substituted its own transform in that combination
    (verified live, not assumed), which is why this simpler, standard
    parameter set is used instead.

    Raises on any HTTP or Earth Engine error; the caller is responsible for
    recording the failure (e.g. in the scene cache) and moving on.
    """
    import requests

    url = ee_image.select(bands).getDownloadURL(
        {
            "region": region,
            "scale": scale_m,
            "crs": crs,
            "filePerBand": False,
            "format": "GEO_TIFF",
            "maxPixels": 1_000_000_000,
        }
    )
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content


def _read_geotiff_bands(data: bytes, n_bands: int) -> tuple[np.ndarray, object, str]:
    """Read a GeoTIFF held in memory into a ``(n_bands, H, W)`` float32 array.

    Uses GDAL's ``/vsimem/`` in-memory filesystem so the bytes never touch
    local disk. Returns ``(bands, transform, crs_wkt)``.
    """
    import rasterio
    from rasterio.io import MemoryFile

    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            if src.count < n_bands:
                raise ValueError(
                    f"Expected >= {n_bands} bands in the downloaded GeoTIFF, got {src.count}"
                )
            arr = src.read(list(range(1, n_bands + 1))).astype(np.float32)
            if src.nodata is not None:
                arr[arr == np.float32(src.nodata)] = np.nan
            transform = src.transform
            crs_wkt = src.crs.to_wkt() if src.crs is not None else ""
    return arr, transform, crs_wkt


def _percentile_stretch(band: np.ndarray) -> np.ndarray:
    """Stretch one band to [0, 1] using its own 2nd-98th percentile range."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    low, high = np.percentile(finite, [_STRETCH_LOW_PCT, _STRETCH_HIGH_PCT])
    if high <= low:
        return np.zeros_like(band, dtype=np.float32)
    stretched = (band - low) / (high - low)
    return np.clip(np.nan_to_num(stretched, nan=0.0), 0.0, 1.0).astype(np.float32)


def render_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Render ``(rgb_raw, rgb_shadow)`` as ``(H, W, 3)`` float32 in [0, 1].

    See the module docstring for what "shadow-enhanced" means here.
    """
    stacked = np.stack(
        [_percentile_stretch(red), _percentile_stretch(green), _percentile_stretch(blue)],
        axis=-1,
    )
    shadow = np.power(np.clip(stacked, 0.0, 1.0), _SHADOW_GAMMA).astype(np.float32)
    return stacked, shadow


def fetch_and_store_scene(
    ee_module,
    zarr_path,
    group: str,
    sensor_name: str,
    scene_id: str,
    year: int,
    date: str,
    window_bounds_3413: tuple[float, float, float, float],
    spec: SensorSpec,
) -> int:
    """Fetch one scene's TOA bands and RGB composites, and append them to the store.

    Parameters
    ----------
    ee_module
        The initialized ``ee`` module (see ``scenes.gee_auth.initialize_ee``).
    zarr_path
        ``<glacier_dir>/<glims_id>.zarr``.
    group
        The zarr sensor group (``s2``/``l89``/``l7``/``l5``), see
        ``scenes.zarr_store.group_for_sensor``.
    window_bounds_3413
        ``(minx, miny, maxx, maxy)`` of the glacier's **analysis window in
        EPSG:3413** -- the same window ``dem.engine.window_polygon_from_registry``
        builds from the registry's ``window_minx``/``window_miny``/
        ``window_maxx``/``window_maxy`` (reprojected from ``metric_crs``).
        Must be given in EPSG:3413, not EPSG:4326 -- see the module
        docstring for why a lon/lat rectangle reprojects into a
        non-rectangular, offset region at this latitude. Also deliberately
        **not** the registry's raw ``min_lon``/``max_lon`` columns (the
        unbuffered glacier polygon bbox, used for scene *search* filtering in
        ``scenes.download``), which is a differently-shaped window around the
        same glacier by construction.

    Returns
    -------
    int
        The index the scene occupies (see :func:`~.zarr_store.append_scene`);
        ``-1`` if it was already present.
    """
    from .zarr_store import append_scene

    image = ee_module.Image(f"{spec.collection}/{scene_id}")
    region = ee_module.Geometry.Rectangle(list(window_bounds_3413), TARGET_EPSG, False)

    toa_data = _download_geotiff_bytes(image, region, spec.native_resolution_m, list(spec.bands))
    toa_raw, transform_toa, crs_wkt_toa = _read_geotiff_bands(toa_data, len(spec.bands))

    if spec.is_sentinel:
        from ..features.spectral_indices import sentinel2_toa_reflectance

        toa = sentinel2_toa_reflectance(toa_raw, clip=True)
    else:
        toa = np.clip(toa_raw, 0.0, 1.0).astype(np.float32)

    # Canonical band order is [Blue, Green, Red, NIR, SWIR1, SWIR2]; sensors.py
    # already lists each sensor's bands in that order (see SENSORS in sensors.py).
    blue, green, red = toa[0], toa[1], toa[2]

    if spec.has_pan:
        pan_data = _download_geotiff_bytes(
            image, region, spec.pan_resolution_m, [spec.pan_band]
        )
        pan_raw, transform_rgb, crs_wkt_rgb = _read_geotiff_bands(pan_data, 1)
        rgb_shape = pan_raw.shape[1:]
        rgb_red = _resample_nearest(red, rgb_shape)
        rgb_green = _resample_nearest(green, rgb_shape)
        rgb_blue = _resample_nearest(blue, rgb_shape)
    else:
        transform_rgb, crs_wkt_rgb = transform_toa, crs_wkt_toa
        rgb_red, rgb_green, rgb_blue = red, green, blue

    rgb_raw_f, rgb_shadow_f = render_rgb(rgb_red, rgb_green, rgb_blue)

    from .zarr_store import rgb_to_uint8

    valid_mask = np.isfinite(rgb_red) & np.isfinite(rgb_green) & np.isfinite(rgb_blue)
    rgb_raw_u8 = rgb_to_uint8(rgb_raw_f, valid_mask=valid_mask)
    rgb_shadow_u8 = rgb_to_uint8(rgb_shadow_f, valid_mask=valid_mask)

    return append_scene(
        zarr_path,
        group=group,
        sensor_name=sensor_name,
        toa_bands=[toa[i] for i in range(toa.shape[0])],
        band_names=list(spec.bands),
        rgb_raw=rgb_raw_u8,
        rgb_shadow=rgb_shadow_u8,
        crs_wkt_toa=crs_wkt_toa,
        transform_toa=transform_toa,
        crs_wkt_rgb=crs_wkt_rgb,
        transform_rgb=transform_rgb,
        scene_id=scene_id,
        year=year,
        date=date,
    )


def _resample_nearest(band: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample of a 2D array to ``target_shape``.

    Used only to bring a 30 m visible band onto the panchromatic band's finer
    grid before Brovey-independent RGB rendering; the model-facing spectral
    indices are computed from the native-resolution bands, never this
    upsampled copy.
    """
    src_h, src_w = band.shape
    dst_h, dst_w = target_shape
    row_idx = np.clip((np.arange(dst_h) * src_h / dst_h).astype(int), 0, src_h - 1)
    col_idx = np.clip((np.arange(dst_w) * src_w / dst_w).astype(int), 0, src_w - 1)
    return band[row_idx][:, col_idx]
