"""Per-glacier DEM storage in the glacier's shared zarr store.

Purpose
-------
Persist the three DEM output resolutions (10, 15, 30 m) and the
interior-hole-filled 30 m product into the *same* ``<glims_id>.zarr`` store
that :mod:`glacier_fsnow_unet.scenes.zarr_store` uses for a glacier's TOA/RGB
scene time series, under a ``dem`` group. See ``docs/decisions/dem_backend.md``
for why one shared store per glacier was chosen over a second store.

Layout
------
::

    <glims_id>.zarr/
        dem/
            elev_10m          (H10, W10)  float32   NaN nodata
            elev_15m          (H15, W15)  float32   NaN nodata
            elev_30m          (H30, W30)  float32   NaN nodata
            elev_30m_interp   (H30, W30)  float32   NaN nodata, interior holes filled
            .attrs: crs_wkt, transform_10m, transform_15m, transform_30m
                    (each a 6-element affine list, the same convention
                    ``scenes.zarr_store.transform_to_list``/``list_to_transform``
                    use for ``transform_toa``), source_dem, source_base_resolution_m,
                    arcticdem_version, margin_m, resample_method,
                    source_window_valid_pct, polygon_stats (JSON string),
                    dem_30_interp_status, generated_at

A DEM has no scene axis (one elevation model per glacier, not one per
acquisition), so each resolution is a plain fixed-shape array -- unlike
``scenes.zarr_store``'s ``(N, B, H, W)`` TOA arrays, there is no append/resize
here, only a create-or-overwrite write.

Concurrency
-----------
Reuses :func:`glacier_fsnow_unet.scenes.zarr_store._get_lock`'s one-lock-per-path
convention so a DEM write and a concurrent scene-append on the same glacier
cannot interleave.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..scenes.zarr_store import (
    _get_lock,
    list_to_transform,
    transform_to_list,
    zarr_path_for_glacier,
)

#: Resolution (m) -> array name inside the ``dem`` group.
_RESOLUTION_ARRAY_NAMES: dict[int, str] = {10: "elev_10m", 15: "elev_15m", 30: "elev_30m"}
INTERP_30M_ARRAY_NAME = "elev_30m_interp"


def _compressor():
    """Blosc/zstd, matching the ratio-favouring choice used for scene TOA planes."""
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def write_dem(
    glacier_dir: str | Path,
    resolutions: dict[int, np.ndarray],
    transforms: dict[int, object],
    crs_wkt: str,
    attrs: Optional[dict] = None,
) -> Path:
    """Write the per-resolution DEM arrays into ``<glims_id>.zarr/dem``.

    Parameters
    ----------
    glacier_dir
        The glacier's directory; the store is ``<glacier_dir>/<glims_id>.zarr``.
    resolutions
        ``{10: array_10m, 15: array_15m, 30: array_30m}``, float32 with NaN
        nodata. Any subset of the three keys may be given.
    transforms
        ``{resolution: affine_transform}``, one per key present in ``resolutions``.
    crs_wkt
        The common CRS (EPSG:3413) as WKT, shared by every resolution.
    attrs
        Extra JSON-serialisable metadata (provenance, coverage stats) merged
        into the group's attrs; ``polygon_stats`` is stored as a JSON string if
        it is a dict, since zarr attrs must round-trip through JSON.

    Returns
    -------
    Path
        The zarr store path written to.
    """
    import zarr

    glacier_dir = Path(glacier_dir)
    zarr_path = zarr_path_for_glacier(glacier_dir)
    lock = _get_lock(zarr_path)

    with lock:
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        store = zarr.open_group(str(zarr_path), mode="a")
        grp = store.require_group("dem")

        compressor = _compressor()
        group_attrs: dict = dict(grp.attrs)
        group_attrs["crs_wkt"] = str(crs_wkt)

        for res, array in resolutions.items():
            name = _RESOLUTION_ARRAY_NAMES.get(int(res))
            if name is None:
                raise ValueError(
                    f"Unsupported DEM resolution {res} m; expected one of "
                    f"{sorted(_RESOLUTION_ARRAY_NAMES)}"
                )
            data = np.asarray(array, dtype=np.float32)
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=data, chunks=data.shape, compressor=compressor)

            transform = transforms.get(res)
            if transform is not None:
                group_attrs[f"transform_{int(res)}m"] = transform_to_list(transform)

        if attrs:
            for key, value in attrs.items():
                if key == "polygon_stats" and isinstance(value, dict):
                    group_attrs[key] = json.dumps(value)
                else:
                    group_attrs[key] = value

        grp.attrs.update(group_attrs)

    return zarr_path


def write_interpolated_30m(
    glacier_dir: str | Path,
    filled_array: np.ndarray,
    status: str = "ok",
) -> Path:
    """Write the interior-hole-filled 30 m product into the same ``dem`` group.

    Stored separately from ``elev_30m`` (rather than overwriting it) so the
    unfilled product -- with its holes still visible -- remains available.
    """
    import zarr

    glacier_dir = Path(glacier_dir)
    zarr_path = zarr_path_for_glacier(glacier_dir)
    lock = _get_lock(zarr_path)

    with lock:
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        store = zarr.open_group(str(zarr_path), mode="a")
        grp = store.require_group("dem")

        data = np.asarray(filled_array, dtype=np.float32)
        if INTERP_30M_ARRAY_NAME in grp:
            del grp[INTERP_30M_ARRAY_NAME]
        grp.create_dataset(
            INTERP_30M_ARRAY_NAME, data=data, chunks=data.shape, compressor=_compressor()
        )
        grp.attrs["dem_30_interp_status"] = status

    return zarr_path


def read_dem(
    glacier_dir: str | Path, resolution: int, interpolated: bool = False
) -> tuple[np.ndarray, object, str]:
    """Read one resolution's DEM array back as ``(array, transform, crs_wkt)``.

    ``interpolated=True`` reads ``elev_30m_interp`` (only valid for ``resolution=30``).
    """
    import zarr

    glacier_dir = Path(glacier_dir)
    zarr_path = zarr_path_for_glacier(glacier_dir)
    store = zarr.open_group(str(zarr_path), mode="r")
    if "dem" not in store:
        raise KeyError(f"No 'dem' group in {zarr_path}")
    grp = store["dem"]

    if interpolated:
        if int(resolution) != 30:
            raise ValueError("The interior-hole-filled product only exists at 30 m.")
        name = INTERP_30M_ARRAY_NAME
    else:
        name = _RESOLUTION_ARRAY_NAMES.get(int(resolution))
        if name is None:
            raise ValueError(f"Unsupported DEM resolution {resolution} m")

    if name not in grp:
        raise KeyError(f"'{name}' is absent from {zarr_path}/dem")

    attrs = dict(grp.attrs)
    transform_key = "transform_30m" if interpolated else f"transform_{int(resolution)}m"
    transform_values = attrs.get(transform_key)
    transform = list_to_transform(transform_values) if transform_values else None
    return np.asarray(grp[name][:], dtype=np.float32), transform, str(attrs.get("crs_wkt", ""))


def read_dem_attrs(glacier_dir: str | Path) -> dict:
    """Read the ``dem`` group's full attrs dict (provenance/coverage metadata)."""
    import zarr

    glacier_dir = Path(glacier_dir)
    zarr_path = zarr_path_for_glacier(glacier_dir)
    store = zarr.open_group(str(zarr_path), mode="r")
    if "dem" not in store:
        raise KeyError(f"No 'dem' group in {zarr_path}")
    attrs = dict(store["dem"].attrs)
    if isinstance(attrs.get("polygon_stats"), str):
        try:
            attrs["polygon_stats"] = json.loads(attrs["polygon_stats"])
        except (TypeError, ValueError):
            pass
    return attrs


def dem_group_exists(glacier_dir: str | Path) -> bool:
    """Whether ``<glims_id>.zarr/dem`` already exists (the resume/skip flag)."""
    import zarr

    glacier_dir = Path(glacier_dir)
    zarr_path = zarr_path_for_glacier(glacier_dir)
    if not zarr_path.exists():
        return False
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception:  # noqa: BLE001 -- a partially written store must not crash discovery
        return False
    return "dem" in store
