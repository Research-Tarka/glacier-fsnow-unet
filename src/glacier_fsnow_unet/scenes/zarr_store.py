"""Per-glacier zarr store: the scene (TOA/RGB) write and read path.

Purpose
-------
Persist downloaded satellite scenes into **one zarr store per glacier**, with
one group per sensor family (see Layout below), so a glacier's full time
series across all five sensors lives in a single chunked, compressed store
rather than one file per scene. Stage 4 writes here; stages 7-12 read back
through this module and :mod:`glacier_fsnow_unet.inference.zarr_store`.

Layout
------
``<glims_id>.zarr`` lives beside the glacier's DEM, i.e.
``<split_root>/<glims_id>/<glims_id>.zarr``, and holds one group per sensor
family::

    <glims_id>.zarr/
        s2/                                   # Sentinel-2
            toa         (N, 6, H30, W30)  uint16  -- reflectance x 10000
            rgb_raw     (N, 3, Hrgb, Wrgb) uint8
            rgb_shadow  (N, 3, Hrgb, Wrgb) uint8
            scene_id    (N,)  str      year (N,) int16
            date        (N,)  str      sensor (N,) str
            .attrs: crs_wkt_toa, transform_toa, crs_wkt_rgb, transform_rgb,
                    band_names, toa_scale_factor, toa_nodata_uint16
        l89/  7 TOA bands (6 spectral + PAN), rgb at 15 m
        l7/   6 TOA bands, rgb at 15 m
        l5/   6 TOA bands, rgb at 30 m

Inference results (``inference``/``inference_done``) and the annual
``year_data`` group are added to the *same* store by
:mod:`glacier_fsnow_unet.inference.zarr_store`.

Sensor keys vs zarr groups
--------------------------
The download stage speaks in :mod:`~glacier_fsnow_unet.scenes.sensors` keys
(``L5``/``L7``/``L8``/``L9``/``S2``); the store speaks in group names. Landsat 8
and 9 share the ``l89`` group because they share a band layout and are
interchangeable for the model. :data:`SENSOR_TO_GROUP` is the single mapping.

Concurrency
-----------
One :class:`threading.Lock` per zarr path (keyed by ``str(path)``), held across
the whole read-modify-write of an append, so parallel download workers on the
same glacier cannot interleave a resize with a write.

Vectorization
-------------
The float32 -> uint16 TOA quantisation and the float -> uint8 RGB rendering are
whole-array numpy expressions; the only Python loops are over bands (at most 7)
and over scenes.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: TOA reflectance is stored as ``uint16 = round(reflectance * 10000)``.
TOA_SCALE_FACTOR: int = 10_000
#: Sentinel value written where reflectance is not finite.
TOA_NODATA_UINT16: int = 65_535

#: Sensor key (see ``scenes.sensors``) -> zarr group name.
SENSOR_TO_GROUP: dict[str, str] = {
    "L5": "l5",
    "L7": "l7",
    "L8": "l89",
    "L9": "l89",
    "S2": "s2",
}

#: Every zarr group name the pipeline knows about, in reference-grid priority
#: order (the first group present defines the reference grid downstream).
GROUP_ORDER: tuple[str, ...] = ("s2", "l89", "l7", "l5")

#: Native resolution, in metres, of the RGB composite stored per group.
GROUP_RGB_RESOLUTION_M: dict[str, float] = {
    "s2": 10.0,
    "l89": 15.0,
    "l7": 15.0,
    "l5": 30.0,
}

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

_SCENE_ID_CACHE: dict[tuple[str, str], set[str]] = {}
_SCENE_ID_CACHE_GUARD = threading.Lock()


def group_for_sensor(sensor_key: str) -> str:
    """Map a sensor key onto its zarr group name."""
    try:
        return SENSOR_TO_GROUP[str(sensor_key).upper()]
    except KeyError:
        raise KeyError(
            f"Unknown sensor '{sensor_key}'. Valid sensors: "
            f"{', '.join(sorted(SENSOR_TO_GROUP))}"
        ) from None


def zarr_path_for_glacier(glacier_dir: str | Path) -> Path:
    """Return ``<glacier_dir>/<glims_id>.zarr`` for a glacier directory.

    The store is named after the directory, which is the GLIMS id, so the
    downstream scanner can recover a glacier's id from its store path alone,
    without opening the store or reading a separate metadata file.
    """
    glacier_dir = Path(glacier_dir)
    return glacier_dir / f"{glacier_dir.name}.zarr"


def _get_lock(zarr_path: Path) -> threading.Lock:
    """One lock per store path, created on first use."""
    key = str(zarr_path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _compressors():
    """Build the Blosc compressors, importing numcodecs lazily.

    TOA and the small metadata arrays use zstd/BITSHUFFLE (best ratio on the
    16-bit reflectance planes); the already-quantised RGB uses lz4/NOSHUFFLE,
    which is much faster and barely worse on 8-bit imagery.
    """
    from numcodecs import Blosc, VLenUTF8

    return {
        "toa": Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE),
        "rgb": Blosc(cname="lz4", clevel=3, shuffle=Blosc.NOSHUFFLE),
        "text": VLenUTF8(),
    }


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def toa_to_uint16(reflectance: np.ndarray) -> np.ndarray:
    """Quantise float32 reflectance in [0, 1] to uint16, NaN -> nodata.

    Values are clipped to [0, :data:`TOA_SCALE_FACTOR`], so an out-of-range
    reflectance saturates rather than wrapping around.
    """
    array = np.asarray(reflectance, dtype=np.float32)
    scaled = np.where(
        np.isfinite(array),
        np.clip(np.round(array * TOA_SCALE_FACTOR), 0, TOA_SCALE_FACTOR),
        TOA_NODATA_UINT16,
    )
    return scaled.astype(np.uint16)


def uint16_to_toa(
    stored: np.ndarray,
    scale: int = TOA_SCALE_FACTOR,
    nodata: int = TOA_NODATA_UINT16,
) -> np.ndarray:
    """Inverse of :func:`toa_to_uint16`: uint16 -> float32 reflectance, nodata -> NaN."""
    stored = np.asarray(stored)
    out = stored.astype(np.float32) / np.float32(scale)
    out[stored == nodata] = np.nan
    return out


def rgb_to_uint8(
    rendered: np.ndarray, valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convert an ``(H, W, 3)`` float composite in [0, 1] to ``(3, H, W)`` uint8.

    Non-finite pixels, and pixels outside ``valid_mask``, become 0 (black),
    which is what the downstream viewers treat as "no imagery here".
    """
    safe = np.nan_to_num(
        np.asarray(rendered, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    out = np.round(np.clip(safe, 0.0, 1.0) * 255.0).astype(np.uint8)
    if valid_mask is not None:
        out[~np.asarray(valid_mask, dtype=bool)] = 0
    return np.transpose(out, (2, 0, 1))


def transform_to_list(transform) -> list[float]:
    """Flatten a ``rasterio.Affine`` (or any 6-sequence) to ``[a, b, c, d, e, f]``.

    Zarr attributes must be JSON-serialisable, so the affine cannot be stored
    directly; :func:`list_to_transform` rebuilds it on read.
    """
    try:
        return [
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
        ]
    except AttributeError:
        return [float(v) for v in list(transform)[:6]]


def list_to_transform(values: Sequence[float]):
    """Rebuild a ``rasterio.Affine`` from the 6 stored coefficients."""
    from rasterio.transform import Affine

    return Affine(*(float(v) for v in list(values)[:6]))


# ---------------------------------------------------------------------------
# Scene-id membership (download-time dedup)
# ---------------------------------------------------------------------------


def _load_scene_ids(zarr_path: Path, group: str) -> set[str]:
    """Read every ``scene_id`` of one group; empty set if absent or unreadable."""
    import zarr

    path = Path(zarr_path)
    if not path.exists():
        return set()
    try:
        store = zarr.open_group(str(path), mode="r")
        if group in store and "scene_id" in store[group]:
            return {str(x) for x in store[group]["scene_id"][:]}
    except Exception:  # noqa: BLE001 -- a partially written store must not crash discovery
        pass
    return set()


def scene_ids(zarr_path: str | Path, group: str) -> set[str]:
    """Scene ids stored in one group, cached in memory per (path, group).

    The store is read once per glacier/sensor; subsequent
    :func:`scene_exists` calls are O(1) dictionary lookups, which matters
    because the download loop calls it for every candidate scene.
    """
    path = Path(zarr_path)
    key = (str(path), group)

    cached = _SCENE_ID_CACHE.get(key)
    if cached is not None:
        return cached

    with _SCENE_ID_CACHE_GUARD:
        cached = _SCENE_ID_CACHE.get(key)
        if cached is not None:
            return cached
        ids = _load_scene_ids(path, group)
        _SCENE_ID_CACHE[key] = ids
        return ids


def scene_exists(zarr_path: str | Path, group: str, scene_id: str) -> bool:
    """Whether this scene is already stored in the glacier's zarr."""
    return str(scene_id) in scene_ids(zarr_path, group)


def _remember_scene_id(zarr_path: Path, group: str, scene_id: str) -> None:
    """Keep the in-memory set consistent after a successful append."""
    key = (str(zarr_path), group)
    with _SCENE_ID_CACHE_GUARD:
        if key in _SCENE_ID_CACHE:
            _SCENE_ID_CACHE[key].add(str(scene_id))


def clear_scene_id_cache() -> None:
    """Drop the in-memory scene-id cache (used by tests and long-lived processes)."""
    with _SCENE_ID_CACHE_GUARD:
        _SCENE_ID_CACHE.clear()


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def append_scene(
    zarr_path: str | Path,
    group: str,
    sensor_name: str,
    toa_bands: Sequence[np.ndarray],
    band_names: Sequence[str],
    rgb_raw: np.ndarray,
    rgb_shadow: np.ndarray,
    crs_wkt_toa: str,
    transform_toa,
    crs_wkt_rgb: str,
    transform_rgb,
    scene_id: str,
    year: int,
    date: str,
    skip_existing: bool = True,
) -> int:
    """Append one scene to a glacier's zarr store, creating the group if needed.

    Parameters
    ----------
    toa_bands
        Sequence of ``(H30, W30)`` float32 reflectance planes, in the order of
        ``band_names``; quantised to uint16 on write.
    rgb_raw, rgb_shadow
        ``(3, Hrgb, Wrgb)`` uint8 composites (plain and shadow-enhanced).
    skip_existing
        When True (the default) a scene id already present is a no-op, so a
        re-run is idempotent.

    Returns
    -------
    int
        The index the scene occupies along the scene axis; ``-1`` when the
        scene was already present and ``skip_existing`` is True.

    Thread-safe: the whole read-modify-write is held under this store's lock.
    """
    zarr_path = Path(zarr_path)
    lock = _get_lock(zarr_path)
    with lock:
        if skip_existing and str(scene_id) in _load_scene_ids(zarr_path, group):
            _remember_scene_id(zarr_path, group, scene_id)
            return -1
        index = _append_locked(
            zarr_path, group, sensor_name, toa_bands, band_names,
            rgb_raw, rgb_shadow, crs_wkt_toa, transform_toa,
            crs_wkt_rgb, transform_rgb, scene_id, year, date,
        )
    _remember_scene_id(zarr_path, group, scene_id)
    return index


def _append_locked(
    zarr_path: Path,
    group: str,
    sensor_name: str,
    toa_bands: Sequence[np.ndarray],
    band_names: Sequence[str],
    rgb_raw: np.ndarray,
    rgb_shadow: np.ndarray,
    crs_wkt_toa: str,
    transform_toa,
    crs_wkt_rgb: str,
    transform_rgb,
    scene_id: str,
    year: int,
    date: str,
) -> int:
    """Do the append; the caller must already hold the store's lock."""
    import zarr

    toa_bands = list(toa_bands)
    band_names = [str(n) for n in band_names]
    if not toa_bands:
        raise ValueError("append_scene requires at least one TOA band.")
    if len(toa_bands) != len(band_names):
        raise ValueError(
            f"{len(toa_bands)} TOA band(s) but {len(band_names)} band name(s); "
            f"they must correspond one to one."
        )

    rgb_raw = np.asarray(rgb_raw, dtype=np.uint8)
    rgb_shadow = np.asarray(rgb_shadow, dtype=np.uint8)
    for name, array in (("rgb_raw", rgb_raw), ("rgb_shadow", rgb_shadow)):
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError(f"{name} must be (3, H, W) uint8, got {array.shape}")
    if rgb_raw.shape != rgb_shadow.shape:
        raise ValueError(
            f"rgb_raw {rgb_raw.shape} and rgb_shadow {rgb_shadow.shape} must match."
        )

    n_bands = len(toa_bands)
    height, width = np.asarray(toa_bands[0]).shape
    _, rgb_height, rgb_width = rgb_raw.shape

    # One vectorized quantisation per band, stacked into (B, H, W).
    toa_stack = np.stack([toa_to_uint16(band) for band in toa_bands], axis=0)

    compressors = _compressors()
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(zarr_path), mode="a")

    if group not in store:
        grp = store.require_group(group)
        grp.create_dataset(
            "toa",
            shape=(0, n_bands, height, width),
            chunks=(1, n_bands, height, width),
            dtype="uint16",
            compressor=compressors["toa"],
        )
        for name in ("rgb_raw", "rgb_shadow"):
            grp.create_dataset(
                name,
                shape=(0, 3, rgb_height, rgb_width),
                chunks=(1, 3, rgb_height, rgb_width),
                dtype="uint8",
                compressor=compressors["rgb"],
            )
        for name in ("scene_id", "date", "sensor"):
            grp.create_dataset(
                name, shape=(0,), chunks=(1000,),
                dtype=object, object_codec=compressors["text"],
            )
        grp.create_dataset(
            "year", shape=(0,), chunks=(1000,),
            dtype="int16", compressor=compressors["toa"],
        )
        grp.attrs.update(
            {
                "crs_wkt_toa": str(crs_wkt_toa),
                "transform_toa": transform_to_list(transform_toa),
                "crs_wkt_rgb": str(crs_wkt_rgb),
                "transform_rgb": transform_to_list(transform_rgb),
                "band_names": band_names,
                "toa_scale_factor": TOA_SCALE_FACTOR,
                "toa_nodata_uint16": TOA_NODATA_UINT16,
            }
        )
    else:
        grp = store[group]
        stored_shape = grp["toa"].shape
        if (stored_shape[1], stored_shape[2], stored_shape[3]) != (n_bands, height, width):
            raise ValueError(
                f"Scene shape (bands={n_bands}, {height}x{width}) does not match "
                f"the existing '{group}' group "
                f"(bands={stored_shape[1]}, {stored_shape[2]}x{stored_shape[3]}). "
                f"Every scene of a glacier must share the DEM's grid."
            )

    index = int(grp["toa"].shape[0])

    grp["toa"].resize((index + 1, n_bands, height, width))
    grp["toa"][index] = toa_stack

    grp["rgb_raw"].resize((index + 1, 3, rgb_height, rgb_width))
    grp["rgb_raw"][index] = rgb_raw

    grp["rgb_shadow"].resize((index + 1, 3, rgb_height, rgb_width))
    grp["rgb_shadow"][index] = rgb_shadow

    grp["scene_id"].resize((index + 1,))
    grp["scene_id"][index] = str(scene_id)

    grp["year"].resize((index + 1,))
    grp["year"][index] = np.int16(year)

    grp["date"].resize((index + 1,))
    grp["date"][index] = str(date)

    grp["sensor"].resize((index + 1,))
    grp["sensor"][index] = str(sensor_name)

    return index


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def open_group(zarr_path: str | Path, group: str, mode: str = "r"):
    """Open one sensor group of a glacier store, with a clear error if absent."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode=mode)
    if group not in store:
        raise KeyError(f"Group '{group}' is absent from {zarr_path}")
    return store[group]


def group_names(zarr_path: str | Path) -> list[str]:
    """Sensor groups present in a store, in :data:`GROUP_ORDER`."""
    import zarr

    path = Path(zarr_path)
    if not path.exists():
        return []
    try:
        store = zarr.open_group(str(path), mode="r")
    except Exception:  # noqa: BLE001
        return []
    return [g for g in GROUP_ORDER if g in store]


def read_scene_metadata(zarr_path: str | Path, group: str) -> dict:
    """Read the per-scene metadata arrays of one group in a single open.

    Returns ``{"scene_id": [...], "year": ndarray, "date": [...],
    "sensor": [...], "n_scenes": int}``.
    """
    grp = open_group(zarr_path, group)
    years = np.asarray(grp["year"][:], dtype=np.int16)
    return {
        "scene_id": [str(s) for s in grp["scene_id"][:]],
        "year": years,
        "date": [str(d) for d in grp["date"][:]],
        "sensor": [str(s) for s in grp["sensor"][:]],
        "n_scenes": int(years.shape[0]),
    }


def read_toa(
    zarr_path: str | Path,
    group: str,
    index: int,
    n_bands: Optional[int] = 6,
) -> tuple[np.ndarray, object, str]:
    """Read one scene's TOA bands as float32 reflectance.

    ``n_bands`` truncates the band axis, which is how the ``l89`` group's
    7 bands (6 spectral + panchromatic) are reduced to the 6 the index stack
    needs. Pass ``None`` to read every stored band.

    Returns ``(bands, transform, crs_wkt)`` with ``bands`` shaped
    ``(n_bands, H, W)`` and nodata as NaN.
    """
    grp = open_group(zarr_path, group)
    attrs = dict(grp.attrs)

    scale = int(attrs.get("toa_scale_factor", TOA_SCALE_FACTOR))
    nodata = int(attrs.get("toa_nodata_uint16", TOA_NODATA_UINT16))
    transform_values = attrs.get("transform_toa")
    if transform_values is None:
        raise RuntimeError(f"'transform_toa' is missing from {zarr_path}/{group}.attrs")

    stored = np.asarray(grp["toa"][int(index)])
    if n_bands is not None and stored.shape[0] > n_bands:
        stored = stored[:n_bands]

    return (
        uint16_to_toa(stored, scale=scale, nodata=nodata),
        list_to_transform(transform_values),
        str(attrs.get("crs_wkt_toa", "")),
    )


def read_rgb(
    zarr_path: str | Path, group: str, index: int, shadow: bool = True
) -> np.ndarray:
    """Read one scene's ``(3, H, W)`` uint8 RGB composite (shadow or raw)."""
    grp = open_group(zarr_path, group)
    key = "rgb_shadow" if shadow else "rgb_raw"
    if key not in grp:
        key = "rgb_raw"
    return np.asarray(grp[key][int(index)], dtype=np.uint8)


def read_group_transform_crs(zarr_path: str | Path, group: str) -> tuple[object, str]:
    """Read ``(transform_toa, crs_wkt_toa)`` of one group."""
    return transform_crs_from_group(open_group(zarr_path, group))


def transform_crs_from_group(grp) -> tuple[object, str]:
    """Read ``(transform_toa, crs_wkt_toa)`` from an already-open group.

    Avoids reopening the store when a caller is walking many scenes of the same
    group, which is the hot path in the inference and compositing stages.
    """
    attrs = dict(grp.attrs)
    return list_to_transform(attrs["transform_toa"]), str(attrs.get("crs_wkt_toa", ""))


def read_group_shape(zarr_path: str | Path, group: str) -> tuple[int, int]:
    """Read the ``(H, W)`` raster shape of one group's TOA array."""
    shape = open_group(zarr_path, group)["toa"].shape  # (N, B, H, W)
    return int(shape[2]), int(shape[3])
