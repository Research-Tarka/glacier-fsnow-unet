"""Class codes and the worst-state annual compositing rule.

Purpose
-------
Define the four surface classes and implement the pixel-wise "worst state"
composite that reduces all retained scenes of one ablation season to a single
annual state map (paper, Sections 3.2 and 4.7).

Class codes
-----------
The paper numbers the *training* labels 0-based (Cloud 0, Snow 1, Ice 2,
Other 3). The rasters written by the pipeline use 1-based codes, with 0 and 255
reserved for nodata:

=====  ==========  =====================
Code   Class       Training label (paper)
=====  ==========  =====================
0      NoData      --
1      Cloud       0
2      Snow        1
3      Ice         2
4      Other       3
=====  ==========  =====================

Worst-state priority
--------------------
"For each pixel, the most ablated observed state is retained with the strict
priority order **Other > Ice > Snow > Cloud**" -- i.e. codes ``4 > 3 > 2 > 1``,
so :data:`PRIORITY_ORDER` is evaluated highest-first. Evaluating Other first
also prevents a single very cloudy scene from contaminating the whole year.

Agreement rule
--------------
"When the same worst state is observed in at least two scenes, that state is
retained as a robust annual assignment; when fewer than two scenes are
available, the single worst observation is used."
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

NODATA = 0
CLOUD = 1
SNOW = 2
ICE = 3
OTHER = 4

#: Human-readable names, keyed by raster code.
CLASS_NAMES = {
    NODATA: "NoData",
    CLOUD: "Cloud",
    SNOW: "Snow",
    ICE: "Ice",
    OTHER: "Other",
}

#: Mapping from raster code to the paper's 0-based training label.
RASTER_TO_TRAINING_LABEL = {CLOUD: 0, SNOW: 1, ICE: 2, OTHER: 3}

#: Worst-state priority, most-ablated first: Other > Ice > Snow > Cloud.
PRIORITY_ORDER: tuple[int, ...] = (OTHER, ICE, SNOW, CLOUD)

#: The valid (non-nodata) class codes.
VALID_CLASSES: tuple[int, ...] = (CLOUD, SNOW, ICE, OTHER)

#: Default minimum number of agreeing scenes for a robust annual assignment.
DEFAULT_MIN_SCENE_AGREEMENT = 2


def sanitize(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only codes 1..4; everything else (0, 255, ...) becomes NoData.

    Returns ``(clean, valid_mask)``.
    """
    pred = np.asarray(prediction, dtype=np.uint8)
    valid = (pred >= CLOUD) & (pred <= OTHER)
    return np.where(valid, pred, np.uint8(NODATA)).astype(np.uint8), valid


def class_counts(stack: np.ndarray) -> dict[int, np.ndarray]:
    """Per-class observation counts across the scene axis of a ``(N, ...)`` stack.

    Fully vectorized: one ``sum`` over the scene axis per class, no Python loop
    over pixels or scenes.
    """
    stack = np.asarray(stack, dtype=np.uint8)
    return {
        cls: (stack == cls).sum(axis=0, dtype=np.uint16) for cls in PRIORITY_ORDER
    }


def worst_state_composite(
    scenes: Sequence[np.ndarray] | np.ndarray,
    min_scene_agreement: int = DEFAULT_MIN_SCENE_AGREEMENT,
) -> np.ndarray:
    """Reduce a stack of per-scene class maps to one annual worst-state map.

    Rules, in order of application, per pixel:

    1. **Single scene** -- the sanitized scene is the answer.
    2. **Agreement pass** -- walking the priority order Other, Ice, Snow, Cloud,
       the first class observed in at least ``threshold`` scenes wins. The
       threshold is ``min_scene_agreement`` capped at the number of available
       scenes (so a 1-scene year is never blocked), and drops to 1 when fewer
       scenes than the requested agreement are available.
    3. **Fallback pass** -- for pixels where no class reached the threshold but
       something *was* observed, the highest-priority class observed at least
       once wins. This ensures an observed pixel is never left NoData.
    4. Pixels never validly observed in any scene stay :data:`NODATA`.

    Parameters
    ----------
    scenes
        Sequence of ``(H, W)`` class maps, or a pre-stacked ``(N, H, W)`` array.
    min_scene_agreement
        Scenes that must agree for a robust assignment (paper: 2).

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 annual composite.
    """
    if isinstance(scenes, np.ndarray) and scenes.ndim >= 3:
        stack = np.asarray(scenes, dtype=np.uint8)
    else:
        scenes = list(scenes)
        if not scenes:
            raise ValueError("No scenes provided for the annual composite.")
        stack = np.stack([np.asarray(s, dtype=np.uint8) for s in scenes], axis=0)

    n_scenes = stack.shape[0]
    if n_scenes == 0:
        raise ValueError("No scenes provided for the annual composite.")
    if n_scenes == 1:
        clean, _ = sanitize(stack[0])
        return clean

    stack, _ = sanitize(stack)
    counts = class_counts(stack)

    threshold = min(min_scene_agreement, n_scenes) if n_scenes >= min_scene_agreement else 1
    threshold = max(1, threshold)

    out = np.zeros(stack.shape[1:], dtype=np.uint8)

    # Pass 1: robust assignment, highest priority first.
    for cls in PRIORITY_ORDER:
        np.copyto(out, np.uint8(cls), where=(counts[cls] >= threshold) & (out == NODATA))

    # Pass 2: fallback for observed-but-below-threshold pixels.
    observed = np.zeros(out.shape, dtype=bool)
    for cls in PRIORITY_ORDER:
        observed |= counts[cls] > 0

    for cls in PRIORITY_ORDER:
        remaining = (out == NODATA) & observed
        if not remaining.any():
            break
        np.copyto(out, np.uint8(cls), where=remaining & (counts[cls] >= 1))

    return out


def cloud_fraction(state_map: np.ndarray, within: np.ndarray | None = None) -> float:
    """Fraction of validly observed pixels classified Cloud.

    ``within`` optionally restricts the computation to a region of interest
    (e.g. the RGI polygon or the VGS mask).
    """
    state = np.asarray(state_map, dtype=np.uint8)
    valid = (state >= CLOUD) & (state <= OTHER)
    if within is not None:
        valid &= np.asarray(within, dtype=bool)

    total = int(valid.sum())
    if total == 0:
        return 0.0
    return float(((state == CLOUD) & valid).sum()) / total
