"""Calibrating the confidence gap below which a class priority rule applies.

The problem
-----------
A softmax argmax treats a pixel where the top two classes score 0.51 and 0.49
exactly like one where they score 0.99 and 0.01. In the first case the model is
not really choosing; it is breaking a tie, and which side it lands on is
arbitrary. For this task the arbitrary choice is not neutral: the four classes
sit in a melt-severity order (Cloud, Snow, Ice, Other), and a near-tie resolved
towards the less-ablated class systematically biases the derived snow fraction.

The rule
--------
When the gap ``P(top1) - P(top2)`` falls below a threshold ``t``, the pixel is
assigned by the fixed priority order Cloud > Snow > Ice > Other, restricted to
the two contending classes, rather than by argmax. Above ``t`` the argmax
stands untouched. ``t = 0`` is exactly plain argmax, so the rule can only ever
be as aggressive as the calibration allows.

Choosing the threshold
----------------------
A larger ``t`` applies the rule to more pixels, which is the point, but it also
overrides more decisions the model got right. :func:`calibrate_threshold`
sweeps ``t`` and keeps the **largest** value whose macro-mIoU is still within
``tolerance`` of the untouched-argmax baseline. Picking the largest rather than
the best-scoring value is deliberate: the aim is the widest ambiguity band the
evidence tolerates, not a micro-optimised mIoU, which on a validation partition
would be fitting noise.

Why the sweep is incremental
----------------------------
The naive sweep re-scores every pixel at every candidate threshold: `O(N x T)`
for N pixels and T thresholds, and N here is tens of millions. Instead, pixels
are sorted once by their confidence gap and the thresholds are walked in
increasing order. Raising ``t`` can only ever *add* pixels to the overridden
set -- never remove one -- so each step needs only the pixels whose gap falls in
the newly-covered band, and their effect is applied to the running confusion
matrix as a pair of `+1`/`-1` bincount updates. That makes the whole sweep
`O(N log N)` for the sort plus `O(N)` for the walk, independent of how finely
the thresholds are spaced.

Determinism
-----------
Everything downstream of the collected gaps is pure array arithmetic on sorted
data, so two calibrations over the same predictions give byte-identical
thresholds. The only randomness is in which tiles the loader yields, which the
caller seeds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "AmbiguityCalibration",
    "AMBIGUITY_FILENAME",
    "PRIORITY_BY_TRAINING_LABEL",
    "collect_confidence_gaps",
    "calibrate_threshold",
    "apply_priority_rule",
    "write_calibration",
    "read_calibration",
]

#: Written next to the model checkpoint; inference reads it if present.
AMBIGUITY_FILENAME = "ambiguity_threshold.json"

#: Priority rank per *training* label (Cloud 0, Snow 1, Ice 2, Other 3), lower
#: wins. The order is Cloud > Snow > Ice > Other, so the rank is the label
#: itself -- stated explicitly rather than relying on that coincidence, since
#: the raster codes elsewhere in the pipeline use a different (1-based) scheme.
PRIORITY_BY_TRAINING_LABEL: tuple[int, ...] = (0, 1, 2, 3)


@dataclass(frozen=True)
class AmbiguityCalibration:
    """The chosen threshold and the evidence for it."""

    threshold: float
    macro_miou_baseline: float
    macro_miou_with_rule: float
    gain: float
    tolerance: float
    n_scenes_analyzed: int
    n_pixels_analyzed: int
    model_path: str
    split: str
    generated_at: str
    #: One record per swept threshold: `{threshold, macro_miou, delta}`.
    sweep: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_priority_rule(
    probabilities: np.ndarray,
    threshold: float,
    priority: Sequence[int] = PRIORITY_BY_TRAINING_LABEL,
) -> np.ndarray:
    """Resolve near-ties by class priority, leaving confident pixels alone.

    Parameters
    ----------
    probabilities
        ``(C, ...)`` softmax probabilities over training labels.
    threshold
        Pixels whose top-two gap is strictly below this are decided by
        ``priority``; the rest keep their argmax. ``0`` is a no-op.

    Returns
    -------
    np.ndarray
        Training labels, same shape as ``probabilities`` minus the class axis.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a leading class axis")

    # Partial sort is enough: only the top two entries are ever read.
    order = np.argsort(-probabilities, axis=0, kind="stable")
    top1, top2 = order[0], order[1]
    flat = probabilities.reshape(probabilities.shape[0], -1)
    columns = np.arange(flat.shape[1])
    gap = (
        flat[top1.reshape(-1), columns] - flat[top2.reshape(-1), columns]
    ).reshape(top1.shape)

    if threshold <= 0:
        return top1.astype(np.uint8)

    rank = np.asarray(priority, dtype=np.int16)
    # Among the two contenders, keep whichever ranks higher (lower number).
    preferred = np.where(rank[top1] <= rank[top2], top1, top2)
    return np.where(gap < float(threshold), preferred, top1).astype(np.uint8)


def collect_confidence_gaps(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ignore_index: int = 255,
    priority: Sequence[int] = PRIORITY_BY_TRAINING_LABEL,
) -> dict[str, np.ndarray]:
    """Reduce one batch of predictions to the four vectors the sweep needs.

    Only annotated pixels are kept, and only their top-two structure: the
    ground-truth label, the argmax label, the label the priority rule *would*
    assign, and the gap at which the rule would start applying. Everything the
    sweep does is a function of these four, so the full probability volume is
    never carried past this point.

    Parameters
    ----------
    probabilities
        ``(B, C, H, W)`` or ``(C, H, W)`` softmax probabilities.
    labels
        Matching ground-truth training labels, ``ignore_index`` where absent.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim == 3:
        probabilities = probabilities[None]
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = labels[None]

    n_classes = probabilities.shape[1]
    flat = probabilities.transpose(0, 2, 3, 1).reshape(-1, n_classes)
    truth = labels.reshape(-1)

    keep = truth != ignore_index
    if not np.any(keep):
        empty_i = np.empty(0, dtype=np.int16)
        return {
            "truth": empty_i,
            "argmax": empty_i,
            "preferred": empty_i,
            "gap": np.empty(0, dtype=np.float32),
        }

    flat = flat[keep]
    truth = truth[keep].astype(np.int16)

    order = np.argsort(-flat, axis=1, kind="stable")
    top1 = order[:, 0].astype(np.int16)
    top2 = order[:, 1].astype(np.int16)
    rows = np.arange(flat.shape[0])
    gap = (flat[rows, top1] - flat[rows, top2]).astype(np.float32)

    rank = np.asarray(priority, dtype=np.int16)
    preferred = np.where(rank[top1] <= rank[top2], top1, top2).astype(np.int16)

    return {"truth": truth, "argmax": top1, "preferred": preferred, "gap": gap}


def _macro_miou(confusion: np.ndarray) -> float:
    """Unweighted mean IoU over the classes actually present in the truth."""
    conf = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denominator = tp + fp + fn
    present = conf.sum(axis=1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denominator > 0, tp / np.maximum(denominator, 1e-12), np.nan)
    usable = present & np.isfinite(iou)
    return float(iou[usable].mean()) if np.any(usable) else float("nan")


def _confusion(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """`(C, C)` counts indexed `[true, predicted]`, via one bincount."""
    flat = np.bincount(
        truth.astype(np.int64) * n_classes + predicted.astype(np.int64),
        minlength=n_classes * n_classes,
    )
    return flat.reshape(n_classes, n_classes).astype(np.int64)


def calibrate_threshold(
    collected: dict[str, np.ndarray],
    tolerance: float = 0.01,
    sweep_max: float = 0.50,
    sweep_steps: int = 201,
    n_classes: int = 4,
) -> tuple[float, float, float, list[dict[str, float]]]:
    """Sweep thresholds and pick the widest one within the mIoU budget.

    Walks the candidate thresholds in increasing order over gap-sorted pixels,
    updating the confusion matrix incrementally as each pixel crosses into the
    overridden band (see the module docstring for why this is `O(N log N)`
    rather than `O(N x T)`).

    Returns
    -------
    tuple
        ``(threshold, baseline_miou, miou_at_threshold, sweep_records)``.
    """
    truth = np.asarray(collected["truth"], dtype=np.int64)
    argmax = np.asarray(collected["argmax"], dtype=np.int64)
    preferred = np.asarray(collected["preferred"], dtype=np.int64)
    gap = np.asarray(collected["gap"], dtype=np.float32)

    if truth.size == 0:
        return 0.0, float("nan"), float("nan"), []

    baseline_confusion = _confusion(truth, argmax, n_classes)
    baseline_miou = _macro_miou(baseline_confusion)

    order = np.argsort(gap, kind="stable")
    truth, argmax, preferred, gap = (
        truth[order],
        argmax[order],
        preferred[order],
        gap[order],
    )
    # Pixels whose priority choice equals their argmax cannot change the
    # confusion matrix at any threshold; dropping them shortens the walk
    # without altering a single count.
    changes = preferred != argmax

    thresholds = np.linspace(0.0, float(sweep_max), max(2, int(sweep_steps)))
    # For each threshold, how many sorted pixels have gap < t.
    boundaries = np.searchsorted(gap, thresholds, side="left")

    confusion = baseline_confusion.copy()
    records: list[dict[str, float]] = []
    cursor = 0

    for threshold, boundary in zip(thresholds, boundaries):
        if boundary > cursor:
            window = slice(cursor, int(boundary))
            moved = changes[window]
            if np.any(moved):
                t_moved = truth[window][moved]
                from_moved = argmax[window][moved]
                to_moved = preferred[window][moved]
                confusion -= _confusion(t_moved, from_moved, n_classes)
                confusion += _confusion(t_moved, to_moved, n_classes)
            cursor = int(boundary)

        miou = _macro_miou(confusion)
        records.append(
            {
                "threshold": float(threshold),
                "macro_miou": float(miou),
                "delta": float(miou - baseline_miou),
            }
        )

    # Widest threshold still inside the budget. Record 0 is plain argmax, whose
    # delta is exactly 0, so an acceptable candidate always exists.
    acceptable = [r for r in records if r["delta"] >= -abs(float(tolerance))]
    chosen = max(acceptable, key=lambda r: r["threshold"])
    return (
        float(chosen["threshold"]),
        float(baseline_miou),
        float(chosen["macro_miou"]),
        records,
    )


def write_calibration(
    calibration: AmbiguityCalibration, destination: str | Path
) -> Path:
    """Write the calibration next to a checkpoint, as ``ambiguity_threshold.json``.

    ``destination`` may be the directory or the file itself.
    """
    path = Path(destination)
    if path.is_dir() or not path.suffix:
        path = path / AMBIGUITY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


def read_calibration(source: str | Path) -> Optional[dict[str, Any]]:
    """Read a calibration from a directory or file, or None if absent.

    Returns None rather than raising for every recoverable case -- missing
    file, unreadable JSON, no usable ``threshold`` key. The caller's fallback
    is plain argmax, which is always correct, so a malformed file must degrade
    to that rather than stop a production inference run.
    """
    path = Path(source)
    if path.is_dir():
        path = path / AMBIGUITY_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        threshold = float(payload["threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(threshold) or threshold < 0:
        return None
    payload["threshold"] = threshold
    return payload


def build_calibration(
    threshold: float,
    baseline_miou: float,
    miou_with_rule: float,
    tolerance: float,
    n_scenes: int,
    n_pixels: int,
    model_path: str,
    split: str,
    sweep: list[dict[str, float]],
) -> AmbiguityCalibration:
    """Assemble the record written to ``ambiguity_threshold.json``."""
    return AmbiguityCalibration(
        threshold=float(threshold),
        macro_miou_baseline=float(baseline_miou),
        macro_miou_with_rule=float(miou_with_rule),
        gain=float(miou_with_rule - baseline_miou),
        tolerance=float(tolerance),
        n_scenes_analyzed=int(n_scenes),
        n_pixels_analyzed=int(n_pixels),
        model_path=str(model_path),
        split=str(split),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sweep=list(sweep),
    )
