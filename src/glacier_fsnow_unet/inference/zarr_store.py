"""Per-scene inference output and the annual composite, in the glacier's zarr store.

Purpose
-------
Persist the two gridded products the inference chain computes but does not
otherwise write anywhere: each scene's per-pixel class map (stage 7) and the
annual worst-state composite (stage 8), plus the glacier's single VGS
reference mask (stage 9). Everything lands in the *same*
``<glims_id>.zarr`` store :mod:`glacier_fsnow_unet.scenes.zarr_store` already
uses for TOA/RGB -- this is the module that store's own docstring names as
where ``inference``/``inference_done`` and ``year_data`` are added.

Layout
------
::

    <glims_id>.zarr/
        s2/ l89/ l7/ l5/                  # existing sensor groups
            inference       (N, H, W)  uint8   -- class map, 1..4 / 0=nodata
            inference_done  (N,)       uint8   -- 0=pending 1=done 2=unprocessable

        year_data/
            years      (Y,)          int16   -- sorted years with a composite
            map_etat   (Y, H30, W30) uint8    -- annual worst-state composite
            vgs_ref    (H30, W30)    uint8    -- the glacier's single VGS mask
            .attrs: crs_wkt, transform (6-float list, same convention as
                    scenes.zarr_store's transform_toa), vgs_ref_year

``inference``/``inference_done`` mirror the scene axis of the sensor group
they belong to (one entry per scene, same index as ``toa``/``scene_id``), so a
class map is always addressed the same way its source scene is. ``year_data``
has no sensor axis: the annual composite already merges every sensor's scenes
for that glacier-year, and the VGS reference mask is one mask for the whole
time series (paper Section 4.7: chosen once, from the earliest qualifying
year, then applied to every year).

Concurrency
-----------
Reuses :mod:`glacier_fsnow_unet.scenes.zarr_store`'s one-lock-per-path table,
so an inference write on one sensor group cannot interleave with a concurrent
scene append or DEM write on the same glacier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..scenes.zarr_store import (
    _get_lock,
    list_to_transform,
    transform_to_list,
    zarr_path_for_glacier,
)

#: inference_done values.
DONE_PENDING = 0
DONE_COMPLETE = 1
DONE_UNPROCESSABLE = 2

#: Fill value of freshly created ``inference`` slots (never a valid class code).
_INFERENCE_UNWRITTEN_FILL = 255


def _compressor():
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


# ---------------------------------------------------------------------------
# Per-scene inference
# ---------------------------------------------------------------------------


def _ensure_inference_arrays(grp, n_scenes: int, height: int, width: int):
    """Create (or resize) ``inference`` and ``inference_done`` for one group."""
    if "inference" not in grp:
        grp.create_dataset(
            "inference",
            shape=(0, height, width),
            chunks=(1, height, width),
            dtype="uint8",
            compressor=_compressor(),
            fill_value=_INFERENCE_UNWRITTEN_FILL,
        )
    inference = grp["inference"]
    if inference.shape[0] < n_scenes:
        inference.resize((n_scenes, height, width))

    if "inference_done" not in grp:
        grp.create_dataset(
            "inference_done",
            shape=(0,),
            chunks=(512,),
            dtype="uint8",
            compressor=_compressor(),
            fill_value=DONE_PENDING,
        )
    done = grp["inference_done"]
    if done.shape[0] < n_scenes:
        done.resize((n_scenes,))
    return inference, done


def write_inference(zarr_path: str | Path, group: str, index: int, class_map: np.ndarray) -> None:
    """Write one scene's ``(H, W)`` class map into ``<group>/inference[index]``.

    Raster codes: 1 Cloud, 2 Snow, 3 Ice, 4 Other, 0 NoData (see
    ``inference.classes``). Marks ``inference_done[index] = 1``.
    """
    import zarr

    class_map = np.asarray(class_map, dtype=np.uint8)
    height, width = class_map.shape
    zarr_path = Path(zarr_path)
    lock = _get_lock(zarr_path)

    with lock:
        store = zarr.open_group(str(zarr_path), mode="a")
        if group not in store:
            raise KeyError(f"Group '{group}' is absent from {zarr_path}")
        grp = store[group]

        n_scenes = int(grp["toa"].shape[0])
        inference, done = _ensure_inference_arrays(
            grp, max(n_scenes, index + 1), height, width
        )
        inference[index] = class_map
        done[index] = DONE_COMPLETE


def mark_scene_unprocessable(zarr_path: str | Path, group: str, index: int) -> None:
    """Mark a scene as unprocessable (e.g. failed the precheck) without a class map.

    Best-effort: swallows errors, since this is a progress annotation, not a
    result -- a failure here must never abort the inference run.
    """
    import zarr

    zarr_path = Path(zarr_path)
    lock = _get_lock(zarr_path)
    try:
        with lock:
            store = zarr.open_group(str(zarr_path), mode="a")
            if group not in store:
                return
            grp = store[group]
            n_scenes = int(grp["toa"].shape[0])
            _, done = _ensure_inference_arrays(
                grp, max(n_scenes, index + 1), *_toa_hw(grp)
            )
            done[index] = DONE_UNPROCESSABLE
    except Exception:  # noqa: BLE001 -- best-effort status write
        pass


def _toa_hw(grp) -> tuple[int, int]:
    shape = grp["toa"].shape
    return int(shape[2]), int(shape[3])


def read_inference(zarr_path: str | Path, group: str, index: int) -> Optional[np.ndarray]:
    """Read one scene's ``(H, W)`` class map.

    Returns ``None`` when the scene has no real inference output yet -- either
    because ``inference_done`` is not 1 (complete), or because the array slot
    was never written (still the ``255`` creation fill value).
    """
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if group not in store or "inference" not in store[group]:
        return None
    grp = store[group]
    if index >= grp["inference"].shape[0]:
        return None
    if inference_status(zarr_path, group, index) != DONE_COMPLETE:
        return None
    return np.asarray(grp["inference"][index], dtype=np.uint8)


def inference_status(zarr_path: str | Path, group: str, index: int) -> int:
    """Return this scene's ``inference_done`` status (0/1/2), or 0 if unwritten."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if group not in store or "inference_done" not in store[group]:
        return DONE_PENDING
    done = store[group]["inference_done"]
    if index >= done.shape[0]:
        return DONE_PENDING
    return int(done[index])


# ---------------------------------------------------------------------------
# Annual composite (year_data)
# ---------------------------------------------------------------------------


def _get_or_create_year_data(store, height: int, width: int, crs_wkt: str, transform):
    if "year_data" not in store:
        yd = store.require_group("year_data")
        yd.create_dataset(
            "years", shape=(0,), chunks=(1000,), dtype="int16", compressor=_compressor(),
        )
        yd.create_dataset(
            "map_etat",
            shape=(0, height, width),
            chunks=(1, height, width),
            dtype="uint8",
            compressor=_compressor(),
            fill_value=0,
        )
        yd.attrs["crs_wkt"] = str(crs_wkt)
        yd.attrs["transform"] = transform_to_list(transform)
    else:
        yd = store["year_data"]
    return yd


def _find_or_append_year(yd, year: int, height: int, width: int) -> int:
    years_arr = yd["years"]
    years = years_arr[:].tolist()
    if year in years:
        return years.index(year)

    new_index = len(years)
    years_arr.resize((new_index + 1,))
    years_arr[new_index] = np.int16(year)
    yd["map_etat"].resize((new_index + 1, height, width))
    return new_index


def write_annual_composite(
    zarr_path: str | Path,
    year: int,
    composite: np.ndarray,
    crs_wkt: str,
    transform,
) -> None:
    """Write one glacier-year's worst-state composite into ``year_data/map_etat``.

    Creates the ``year_data`` group on first use, carrying ``crs_wkt`` and
    ``transform`` as attrs (the same 6-float affine convention
    ``scenes.zarr_store`` uses for ``transform_toa``).
    """
    import zarr

    composite = np.asarray(composite, dtype=np.uint8)
    height, width = composite.shape
    zarr_path = Path(zarr_path)
    lock = _get_lock(zarr_path)

    with lock:
        store = zarr.open_group(str(zarr_path), mode="a")
        yd = _get_or_create_year_data(store, height, width, crs_wkt, transform)
        index = _find_or_append_year(yd, int(year), height, width)
        yd["map_etat"][index] = composite


def write_annual_composites_batch(
    zarr_path: str | Path,
    composites: dict[int, np.ndarray],
    crs_wkt: str,
    transform,
) -> None:
    """Write many glacier-years in one zarr open (avoids N lock/open round-trips)."""
    import zarr

    if not composites:
        return

    first = np.asarray(next(iter(composites.values())), dtype=np.uint8)
    height, width = first.shape
    zarr_path = Path(zarr_path)
    lock = _get_lock(zarr_path)

    with lock:
        store = zarr.open_group(str(zarr_path), mode="a")
        yd = _get_or_create_year_data(store, height, width, crs_wkt, transform)

        years_arr = yd["years"]
        current_years = years_arr[:].tolist()
        year_to_index = {int(y): i for i, y in enumerate(current_years)}

        new_years = sorted(y for y in composites if int(y) not in year_to_index)
        if new_years:
            n_existing = len(current_years)
            new_total = n_existing + len(new_years)
            years_arr.resize((new_total,))
            yd["map_etat"].resize((new_total, height, width))
            for offset, year in enumerate(new_years):
                index = n_existing + offset
                years_arr[index] = np.int16(year)
                year_to_index[int(year)] = index

        for year, composite in sorted(composites.items()):
            yd["map_etat"][year_to_index[int(year)]] = np.asarray(composite, dtype=np.uint8)


def read_annual_composite(zarr_path: str | Path, year: int) -> Optional[np.ndarray]:
    """Read one year's annual composite, or ``None`` if that year is absent."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if "year_data" not in store:
        return None
    yd = store["year_data"]
    years = yd["years"][:].tolist()
    if int(year) not in years:
        return None
    index = years.index(int(year))
    return np.asarray(yd["map_etat"][index], dtype=np.uint8)


def read_all_annual_composites(zarr_path: str | Path) -> dict[int, np.ndarray]:
    """Read every year's composite in one open. Returns ``{year: (H, W) array}``."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if "year_data" not in store:
        return {}
    yd = store["year_data"]
    years = [int(y) for y in yd["years"][:].tolist()]
    if not years:
        return {}
    all_data = np.asarray(yd["map_etat"][:], dtype=np.uint8)
    return {year: all_data[i] for i, year in enumerate(years)}


def read_available_years(zarr_path: str | Path) -> list[int]:
    """Sorted list of years with an annual composite."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if "year_data" not in store:
        return []
    return sorted(int(y) for y in store["year_data"]["years"][:].tolist())


# ---------------------------------------------------------------------------
# VGS reference mask
# ---------------------------------------------------------------------------


def write_vgs_reference(
    zarr_path: str | Path,
    vgs_mask: np.ndarray,
    reference_year: Optional[int],
    crs_wkt: str = "",
    transform=None,
) -> None:
    """Write the glacier's single VGS reference mask into ``year_data/vgs_ref``.

    Requires ``year_data`` to already exist (via a prior
    :func:`write_annual_composite` call) unless ``transform`` is given to
    create it fresh.
    """
    import zarr

    mask = np.asarray(vgs_mask, dtype=np.uint8)
    height, width = mask.shape
    zarr_path = Path(zarr_path)
    lock = _get_lock(zarr_path)

    with lock:
        store = zarr.open_group(str(zarr_path), mode="a")
        if "year_data" not in store:
            if transform is None:
                raise RuntimeError(
                    "year_data does not exist yet; pass transform to create it, "
                    "or write an annual composite first."
                )
            yd = _get_or_create_year_data(store, height, width, crs_wkt, transform)
        else:
            yd = store["year_data"]

        if "vgs_ref" not in yd:
            yd.create_dataset(
                "vgs_ref",
                shape=(height, width),
                chunks=(height, width),
                dtype="uint8",
                compressor=_compressor(),
                fill_value=0,
            )
        yd["vgs_ref"][:] = mask
        yd.attrs["vgs_ref_year"] = None if reference_year is None else int(reference_year)


def read_vgs_reference(zarr_path: str | Path) -> Optional[np.ndarray]:
    """Read the VGS reference mask as a boolean array, or ``None`` if absent."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if "year_data" not in store or "vgs_ref" not in store["year_data"]:
        return None
    return np.asarray(store["year_data"]["vgs_ref"][:], dtype=np.uint8).astype(bool)


def read_year_data_attrs(zarr_path: str | Path) -> dict:
    """Read ``year_data``'s attrs (``crs_wkt``, ``transform``, ``vgs_ref_year``)."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if "year_data" not in store:
        return {}
    return dict(store["year_data"].attrs)


def read_year_data_transform_crs(zarr_path: str | Path) -> tuple[Optional[object], str]:
    """Return ``(transform, crs_wkt)`` from ``year_data``'s attrs."""
    attrs = read_year_data_attrs(zarr_path)
    crs_wkt = str(attrs.get("crs_wkt", ""))
    transform_values = attrs.get("transform")
    transform = list_to_transform(transform_values) if transform_values else None
    return transform, crs_wkt
