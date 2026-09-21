"""Sliding-window U-Net inference over full glacier scenes.

Purpose
-------
Apply the trained four-class U-Net to a scene's 11-channel spectral-index
stack, tiling the scene into overlapping patches and averaging the softmax
probabilities where patches overlap: 48x48 patches, stride 16 -- the same
patch/stride pair the paper's Section 4.2 specifies for training, reused
unchanged at inference. See ``docs/decisions/architecture_conformance.md``
for the inference-tiling item: "50% overlap at inference" does not appear
anywhere in the paper text, and this module's own former default (stride 24,
a literal 50% of 48) was not traceable to any cited source. Stride 16 with a
48 px patch is roughly 67% overlap, not 50%.

Inputs
------
- A ``(11, H, W)`` float32 stack of spectral indices.
- A loaded model (see ``model_loader``).

Outputs
-------
- ``(H, W)`` uint8 class map using the pipeline's raster codes
  (1 Cloud, 2 Snow, 3 Ice, 4 Other; 0 NoData), and optionally the
  ``(4, H, W)`` probability volume.
- :func:`run_inference_for_glacier` drives the whole per-scene loop for one
  glacier's zarr store -- reading TOA, computing the 11 spectral indices,
  running the model, and writing each class map back through
  :mod:`glacier_fsnow_unet.inference.zarr_store` -- so ``scripts/07_run_inference.py``
  is a thin CLI wrapper around it rather than owning this logic itself.
- :func:`build_annual_composites_for_glacier` drives stage 8: gathering every
  sensor group's per-scene class maps for one glacier, grouping by year, and
  writing the worst-state composite (``inference.classes.worst_state_composite``)
  through the same zarr store.

Vectorization
-------------
Patches are extracted as a single strided batch and pushed through the model in
mini-batches; accumulation into the full-scene probability volume uses
whole-array slice additions. There is no per-pixel Python anywhere. Patch
*positions* are enumerated in Python, but that loop is over tiles (thousands at
most), not pixels, and each iteration does whole-array work.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .classes import CLOUD, ICE, NODATA, OTHER, SNOW

#: Paper Section 4.2: 48x48 patches.
DEFAULT_PATCH_SIZE = 48
#: Same stride as training (paper Section 4.2); ~67% overlap at inference,
#: not 50% -- see the module docstring and docs/decisions/architecture_conformance.md.
DEFAULT_STRIDE = 16
#: Patches per forward pass.
DEFAULT_BATCH_SIZE = 384

#: Model output channel order (paper training labels) -> raster class codes.
#: Cloud 0 -> 1, Snow 1 -> 2, Ice 2 -> 3, Other 3 -> 4.
TRAINING_LABEL_TO_RASTER = np.array([CLOUD, SNOW, ICE, OTHER], dtype=np.uint8)


def patch_origins(size: int, patch: int, stride: int) -> list[int]:
    """Top-left offsets tiling one axis, always including the final edge patch.

    The last origin is snapped to ``size - patch`` so the trailing strip is
    covered even when ``(size - patch)`` is not a multiple of ``stride``.
    """
    if size <= patch:
        return [0]
    origins = list(range(0, size - patch + 1, stride))
    if origins[-1] != size - patch:
        origins.append(size - patch)
    return origins


def _hann_weights(patch: int) -> np.ndarray:
    """Separable raised-cosine weights, tapering each patch towards its edges.

    Weighting overlapping predictions by distance from the patch centre avoids
    the visible seams that plain averaging leaves at tile boundaries, because
    predictions near a patch edge see less context.
    """
    window = np.hanning(patch + 2)[1:-1]  # drop the zero endpoints
    weights = np.outer(window, window).astype(np.float32)
    return np.maximum(weights, 1e-6)


def predict_scene(
    model,
    features: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
    valid_mask: Optional[np.ndarray] = None,
    return_probabilities: bool = False,
    use_hann_weighting: bool = True,
    ambiguity_threshold: float = 0.0,
):
    """Run sliding-window inference over a full scene.

    Parameters
    ----------
    model
        A loaded U-Net (see :func:`~.model_loader.load_model`).
    features
        ``(11, H, W)`` float32 spectral-index stack.
    patch_size, stride
        Tiling geometry; the default stride (16) matches the training
        stride, giving ~67% overlap at inference (not 50%).
    valid_mask
        Optional ``(H, W)`` boolean mask; pixels outside it are set to NoData.
    return_probabilities
        Also return the ``(4, H, W)`` averaged probability volume.
    ambiguity_threshold
        Confidence gap below which the class-priority rule replaces the argmax.
        ``0`` is plain argmax; see :func:`probabilities_to_classes`.

    Returns
    -------
    np.ndarray | tuple
        The ``(H, W)`` uint8 class map, or ``(class_map, probabilities)``.
    """
    import torch

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"Expected a (C, H, W) feature stack, got {features.shape}")

    n_channels, height, width = features.shape
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        features = np.pad(
            features, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )
    _, padded_h, padded_w = features.shape

    # Non-finite features (nodata) would poison the convolutions.
    finite = np.all(np.isfinite(features), axis=0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    rows = patch_origins(padded_h, patch_size, stride)
    cols = patch_origins(padded_w, patch_size, stride)
    positions = [(r, c) for r in rows for c in cols]

    n_classes = 4
    accumulator = np.zeros((n_classes, padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weights = _hann_weights(patch_size) if use_hann_weighting else np.ones(
        (patch_size, patch_size), dtype=np.float32
    )

    model_device = next(model.parameters()).device if hasattr(model, "parameters") else device

    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            batch = np.stack(
                [features[:, r:r + patch_size, c:c + patch_size] for r, c in chunk],
                axis=0,
            )
            tensor = torch.from_numpy(batch).to(model_device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for (r, c), prob in zip(chunk, probabilities):
                accumulator[:, r:r + patch_size, c:c + patch_size] += prob * weights
                weight_sum[r:r + patch_size, c:c + patch_size] += weights

    np.divide(
        accumulator, np.maximum(weight_sum, 1e-6), out=accumulator, where=weight_sum > 0
    )

    accumulator = accumulator[:, :height, :width]
    finite = finite[:height, :width]

    class_map = probabilities_to_classes(
        accumulator, ambiguity_threshold=ambiguity_threshold
    )
    class_map[~finite] = NODATA
    if valid_mask is not None:
        class_map[~np.asarray(valid_mask, dtype=bool)] = NODATA

    if return_probabilities:
        return class_map, accumulator
    return class_map


def probabilities_to_classes(
    probabilities: np.ndarray, ambiguity_threshold: float = 0.0
) -> np.ndarray:
    """Reduce a ``(4, H, W)`` probability volume to raster class codes.

    Maps the model's training-label order (Cloud 0, Snow 1, Ice 2, Other 3)
    onto the pipeline's 1-based raster codes.

    Parameters
    ----------
    ambiguity_threshold
        When positive, pixels whose top-two probability gap falls below it are
        assigned by the Cloud > Snow > Ice > Other priority order instead of by
        argmax -- the model is not really choosing between two classes that
        close, and for this task an arbitrary tie-break is not neutral (see
        :mod:`glacier_fsnow_unet.interpretation.ambiguity`). ``0`` -- the
        default, and what applies whenever no calibration file is present -- is
        exactly plain argmax.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape[0] != 4:
        raise ValueError(
            f"Expected 4 class channels, got {probabilities.shape[0]}"
        )

    if ambiguity_threshold and ambiguity_threshold > 0:
        from ..interpretation.ambiguity import apply_priority_rule

        labels = apply_priority_rule(probabilities, float(ambiguity_threshold))
    else:
        labels = np.argmax(probabilities, axis=0)
    return TRAINING_LABEL_TO_RASTER[labels]


def resolve_ambiguity_threshold(model_path) -> float:
    """Read the calibrated ambiguity threshold sitting beside a checkpoint.

    Returns ``0.0`` -- plain argmax -- when no calibration has been produced
    yet, or when the file present cannot be read. Inference must never fail
    because an optional analysis output is missing or malformed; the untouched
    argmax it falls back to is always a valid result.
    """
    from ..interpretation.ambiguity import read_calibration

    if model_path is None:
        return 0.0
    from pathlib import Path as _Path

    path = _Path(str(model_path))
    directory = path.parent if path.is_file() else path
    calibration = read_calibration(directory)
    return float(calibration["threshold"]) if calibration else 0.0


def valid_ratio(class_map: np.ndarray) -> float:
    """Share of pixels carrying a real class (not NoData)."""
    array = np.asarray(class_map, dtype=np.uint8)
    return float(((array >= CLOUD) & (array <= OTHER)).mean()) if array.size else 0.0


def scene_passes_precheck(
    features: np.ndarray, min_valid_ratio: float = 0.30
) -> bool:
    """Whether a scene has enough finite pixels to be worth running inference on.

    A scene that is mostly nodata (edge-of-swath, heavy fill) is skipped
    before the model runs, since its inference output would be dominated by
    the no-data mask rather than a meaningful classification.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return False
    finite = np.all(np.isfinite(features), axis=0)
    return float(finite.mean()) >= min_valid_ratio


def run_inference_for_glacier(
    zarr_path,
    model,
    device: str = "cuda",
    groups: Optional[list[str]] = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_valid_ratio: float = 0.30,
    overwrite: bool = False,
    ambiguity_threshold: float = 0.0,
) -> dict[str, int]:
    """Run inference over every scene of one glacier's zarr store, writing results back.

    For each sensor group present in the store, and each scene in that group:
    read the TOA bands, compute the 11-channel spectral-index stack, precheck
    for enough valid pixels, run the sliding-window model, and write the class
    map (and progress status) through :mod:`glacier_fsnow_unet.inference.zarr_store`.

    ``ambiguity_threshold`` is passed through to every scene; callers get it
    from :func:`resolve_ambiguity_threshold`, which reads whatever calibration
    sits beside the checkpoint and returns ``0`` (plain argmax) when there is
    none.

    Returns
    -------
    dict
        Counters: ``{"processed": n, "skipped_precheck": n, "already_done": n}``.
    """
    from ..features.spectral_indices import compute_indices
    from ..scenes.zarr_store import GROUP_ORDER, group_names, read_scene_metadata, read_toa
    from . import zarr_store as izs

    counters = {"processed": 0, "skipped_precheck": 0, "already_done": 0}
    available_groups = groups if groups is not None else group_names(zarr_path)

    for group in available_groups:
        if group not in GROUP_ORDER:
            continue
        try:
            metadata = read_scene_metadata(zarr_path, group)
        except KeyError:
            continue

        for index in range(metadata["n_scenes"]):
            if not overwrite and izs.inference_status(zarr_path, group, index) != izs.DONE_PENDING:
                counters["already_done"] += 1
                continue

            bands, _, _ = read_toa(zarr_path, group, index, n_bands=6)
            features = compute_indices(bands)

            if not scene_passes_precheck(features, min_valid_ratio=min_valid_ratio):
                izs.mark_scene_unprocessable(zarr_path, group, index)
                counters["skipped_precheck"] += 1
                continue

            valid_mask = np.all(np.isfinite(bands), axis=0)
            class_map = predict_scene(
                model,
                features,
                patch_size=patch_size,
                stride=stride,
                batch_size=batch_size,
                device=device,
                valid_mask=valid_mask,
                ambiguity_threshold=ambiguity_threshold,
            )
            izs.write_inference(zarr_path, group, index, class_map)
            counters["processed"] += 1

    return counters


def _align_class_map_to_reference(
    class_map: np.ndarray,
    src_transform,
    src_crs_wkt: str,
    ref_shape: tuple[int, int],
    ref_transform,
    ref_crs_wkt: str,
) -> np.ndarray:
    """Reproject one sensor's class map onto the reference grid.

    Each sensor group keeps its own native pixel grid in the zarr store (30 m
    for Landsat, ~10-20 m for Sentinel-2 -- there is no shared glacier-wide
    grid across groups), so a class map must be resampled onto one common
    grid before different sensors' scenes can be combined into a single
    annual composite. Nearest-neighbor is correct here: the source is
    already a categorical class map (Cloud/Snow/Ice/Other/NoData), not a
    continuous field, so any other resampling method would invent
    intermediate class values that do not exist.
    """
    if class_map.shape == ref_shape and src_transform == ref_transform:
        return class_map

    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject

    destination = np.zeros(ref_shape, dtype=np.uint8)
    src_crs = CRS.from_user_input(src_crs_wkt) if src_crs_wkt else None
    dst_crs = CRS.from_user_input(ref_crs_wkt) if ref_crs_wkt else src_crs
    reproject(
        class_map,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return destination


def build_annual_composites_for_glacier(
    zarr_path,
    min_scene_agreement: int = 2,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build and write the annual worst-state composite for every year with
    at least one scene carrying an inference result, across every sensor group.

    Each sensor group has its own native pixel grid (Landsat at 30 m,
    Sentinel-2 at its own finer resolution) -- there is no glacier-wide grid
    shared across groups. The reference grid for the composite is the finest
    resolution available for this glacier, in priority order
    ``s2 > l89 > l7 > l5`` (:data:`GROUP_ORDER`); every other group's class
    maps are reprojected onto it (:func:`_align_class_map_to_reference`)
    before compositing. Compositing onto the finest available grid rather than
    the coarsest keeps every sensor's full spatial detail: downsampling to the
    coarsest common grid would discard Sentinel-2 resolution in every year that
    happens to also contain a Landsat scene.

    Returns
    -------
    dict
        ``{"years_written": n, "years_skipped_existing": n}``.
    """
    from ..scenes.zarr_store import (
        GROUP_ORDER,
        group_names,
        open_group,
        read_group_shape,
        read_scene_metadata,
        transform_crs_from_group,
    )
    from . import zarr_store as izs
    from .classes import worst_state_composite

    zarr_path = str(zarr_path)
    existing_years = set(izs.read_available_years(zarr_path)) if not overwrite else set()

    groups = group_names(zarr_path)
    if not groups:
        return {"years_written": 0, "years_skipped_existing": 0}

    # GROUP_ORDER is already finest-resolution-first (s2 > l89 > l7 > l5);
    # group_names() preserves that order, so the first available group is
    # the reference grid for this glacier.
    ref_group = groups[0]
    ref_shape = read_group_shape(zarr_path, ref_group)
    ref_transform, ref_crs_wkt = transform_crs_from_group(open_group(zarr_path, ref_group))

    scenes_by_year: dict[int, list[np.ndarray]] = {}
    for group in groups:
        if group not in GROUP_ORDER:
            continue
        metadata = read_scene_metadata(zarr_path, group)
        src_transform, src_crs_wkt = transform_crs_from_group(open_group(zarr_path, group))
        for index, year in enumerate(metadata["year"]):
            year = int(year)
            class_map = izs.read_inference(zarr_path, group, index)
            if class_map is None:
                continue
            aligned = _align_class_map_to_reference(
                class_map, src_transform, src_crs_wkt, ref_shape, ref_transform, ref_crs_wkt
            )
            scenes_by_year.setdefault(year, []).append(aligned)

    counters = {"years_written": 0, "years_skipped_existing": 0}
    to_write: dict[int, np.ndarray] = {}
    for year, class_maps in scenes_by_year.items():
        if year in existing_years:
            counters["years_skipped_existing"] += 1
            continue
        to_write[year] = worst_state_composite(class_maps, min_scene_agreement=min_scene_agreement)

    if to_write:
        izs.write_annual_composites_batch(zarr_path, to_write, ref_crs_wkt, ref_transform)
        counters["years_written"] = len(to_write)

    return counters
