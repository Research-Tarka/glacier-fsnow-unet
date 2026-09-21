"""Segmentation metrics: confusion matrices, IoU variants, boundary metrics.

Everything reported for a partition derives from one pooled `C x C` confusion
matrix accumulated over that partition's tiles. Pooling first and deriving
second matters: averaging per-tile IoU instead would let a tile containing
three Ice pixels weigh as heavily as one containing three thousand.

Confusion accumulation stays on the GPU (a `bincount` over the flattened
`true * C + pred` index) so each batch transfers one small `C x C` matrix to
the host rather than a full prediction volume.

One rounding behaviour is deliberate and load-bearing: the macro metrics are
rounded to three decimals before being returned, and `miou` is what drives
early stopping. An epoch therefore has to gain at least 0.001 mIoU on an
already-rounded value to count as an improvement; smaller gains are invisible
and consume patience. Removing the rounding makes runs train materially
longer and converge somewhere else, so it is preserved on purpose. See
`docs/decisions/training_reproducibility_audit.md`.

`CalibrationTally` sits alongside the confusion matrix and answers the question
the confusion matrix cannot: not how often the model is right, but whether its
reported probability tracks how often it is right. It accumulates in the same
bounded, on-device way — a fixed-length vector per partition regardless of how
many pixels pass through it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES

__all__ = [
    "confusion_from_predictions",
    "update_confusion",
    "multiclass_iou",
    "compute_confusion_metrics",
    "canonical_main_iou_metric",
    "select_main_iou_metric",
    "binary_glacier_iou",
    "boundary_band",
    "compute_boundary_metrics",
    "dilated_confusion",
    "CalibrationTally",
    "DEFAULT_CALIBRATION_BINS",
]

# Metric names accepted for `main_iou_metric`, mapped to a canonical spelling.
_MAIN_METRIC_ALIASES: Mapping[str, str] = {
    "miou": "miou_macro",
    "miou_macro": "miou_macro",
    "miou_weighted": "miou_w_glacier",
    "miou_w_glacier": "miou_w_glacier",
    "miou_inv_freq": "miou_invfq_w_glacier",
    "miou_invfq_w_glacier": "miou_invfq_w_glacier",
}


def confusion_from_predictions(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Confusion matrix for one batch, computed on the tensors' own device.

    Returns a `(num_classes, num_classes)` int64 tensor indexed
    `[true, predicted]`.
    """
    mask = target != ignore_index
    if not torch.any(mask):
        return torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=pred.device
        )

    true_flat = target[mask].reshape(-1).long()
    pred_flat = pred[mask].reshape(-1).long()
    flat_index = true_flat * num_classes + pred_flat
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.int64)


def update_confusion(
    confusion: np.ndarray,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Accumulate one batch into a NumPy confusion matrix, in place."""
    batch = confusion_from_predictions(pred, target, num_classes, ignore_index)
    confusion += batch.cpu().numpy()
    return confusion


def multiclass_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Mean IoU over the classes present, from a single batch.

    Classes with an empty union (absent from both prediction and truth in this
    batch) are skipped rather than scored zero — a class that cannot appear
    should not drag the mean down.

    Derived from the batch confusion matrix rather than from per-class boolean
    masks, which is the same arithmetic in one pass instead of `num_classes`.
    """
    conf = confusion_from_predictions(pred, target, num_classes, ignore_index)
    if int(conf.sum()) == 0:
        return float("nan")

    conf = conf.to(torch.float64)
    tp = torch.diagonal(conf)
    union = conf.sum(dim=0) + conf.sum(dim=1) - tp

    present = union > 0
    if not torch.any(present):
        return float("nan")
    return float((tp[present] / union[present]).mean().item())


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _round3(value: float) -> float:
    """Round to three decimals, passing NaN through.

    Applied to every macro metric. See the module docstring: `miou` drives
    early stopping, so this rounding is part of the stopping behaviour.
    """
    return (
        round(float(value), 3)
        if isinstance(value, (float, int, np.floating)) and math.isfinite(float(value))
        else float("nan")
    )


def compute_confusion_metrics(
    confusion: np.ndarray,
    num_classes: int = NUM_CLASSES,
    beta: float = 1.0,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, object]:
    """Derive per-class and macro metrics from a pooled confusion matrix.

    Args:
        confusion: `(C, C)` counts indexed `[true, predicted]`.
        beta: for F-beta. 1.0 gives F1.

    Returns a dict with `per_class` (a list of one dict per class) and `macro`.
    Macro includes three mIoU flavours, Cohen's kappa, and Gorodkin's
    multiclass MCC.
    """
    conf = np.asarray(confusion, dtype=np.int64)
    total = int(conf.sum())

    tp = np.diag(conf).astype(np.float64)
    predicted_totals = conf.sum(axis=0).astype(np.float64)
    true_totals = conf.sum(axis=1).astype(np.float64)
    fp = predicted_totals - tp
    fn = true_totals - tp
    tn = float(total) - (tp + fp + fn)

    def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(
            num, den, out=np.full_like(num, np.nan, dtype=np.float64), where=den != 0
        )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    beta_sq = beta * beta
    f_beta = _safe_div((1 + beta_sq) * precision * recall, beta_sq * precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    specificity = _safe_div(tn, tn + fp)

    # Three mIoU flavours: unweighted, weighted by class support, and weighted
    # by inverse class frequency (which favours the rare classes).
    valid = np.isfinite(iou) & (true_totals > 0)
    miou = _nanmean(iou)
    if np.any(valid):
        miou_weighted = float(np.average(iou[valid], weights=true_totals[valid]))
        frequencies = true_totals / float(total) if total > 0 else true_totals
        inv_freq = np.divide(
            1.0, frequencies, out=np.zeros_like(frequencies), where=frequencies > 0
        )
        miou_inv_freq = float(np.average(iou[valid], weights=inv_freq[valid]))
    else:
        miou_weighted = miou_inv_freq = float("nan")

    # Cohen's kappa: observed agreement corrected for chance agreement.
    if total > 0:
        p_observed = tp.sum() / total
        p_expected = float(np.dot(predicted_totals, true_totals)) / float(total * total)
        kappa = (
            (p_observed - p_expected) / (1 - p_expected)
            if (1 - p_expected) != 0
            else float("nan")
        )
    else:
        kappa = float("nan")

    # Gorodkin's multiclass MCC. Equals kappa's zero point under random
    # prediction but is less sensitive to marginal imbalance.
    if total > 0:
        total_f = float(total)
        numerator = float(tp.sum()) * total_f - float(
            np.dot(predicted_totals, true_totals)
        )
        denom_sq = (total_f**2 - float(np.dot(predicted_totals, predicted_totals))) * (
            total_f**2 - float(np.dot(true_totals, true_totals))
        )
        mcc = (
            numerator / math.sqrt(denom_sq)
            if denom_sq > 0 and math.isfinite(denom_sq)
            else float("nan")
        )
    else:
        mcc = float("nan")

    per_class = [
        {
            "class": index,
            "class_name": class_names[index] if index < len(class_names) else str(index),
            "support": int(true_totals[index]),
            "precision": _round3(precision[index]),
            "recall": _round3(recall[index]),
            "f1": _round3(f1[index]),
            "f_beta": _round3(f_beta[index]),
            "iou": _round3(iou[index]),
            "specificity": _round3(specificity[index]),
        }
        for index in range(num_classes)
    ]

    return {
        "per_class": per_class,
        "macro": {
            "beta": float(beta),
            "miou": _round3(miou),
            "miou_weighted": _round3(miou_weighted),
            "miou_inv_freq": _round3(miou_inv_freq),
            "kappa": _round3(kappa),
            "mcc": _round3(mcc),
            "specificity": _round3(_nanmean(specificity)),
            "total": total,
        },
    }


def binary_glacier_iou(confusion: np.ndarray) -> float:
    """IoU of Snow+Ice against Cloud+Other, collapsed to two classes.

    Separates *finding* the glacier from *discriminating within* it: this
    measures only the former, so comparing it against the four-class mIoU
    shows how much residual error sits at the internal Snow/Ice partition
    rather than at the glacier boundary.
    """
    conf = np.asarray(confusion, dtype=np.int64)
    glacier = [1, 2]  # Snow, Ice
    non_glacier = [0, 3]  # Cloud, Other

    tp = int(conf[np.ix_(glacier, glacier)].sum())
    fn = int(conf[np.ix_(glacier, non_glacier)].sum())
    fp = int(conf[np.ix_(non_glacier, glacier)].sum())

    denominator = tp + fp + fn
    return float(tp / denominator) if denominator > 0 else float("nan")


def canonical_main_iou_metric(metric: Optional[str]) -> str:
    """Normalise a `main_iou_metric` name, defaulting to the glacier-weighted one."""
    return _MAIN_METRIC_ALIASES.get(str(metric or "").strip().lower(), "miou_w_glacier")


def select_main_iou_metric(
    metric: Optional[str],
    *,
    miou_macro: float,
    miou_w_glacier: float,
    miou_invfq_w_glacier: float,
) -> tuple[str, float]:
    """Resolve the metric name that drives early stopping, and its value."""
    name = canonical_main_iou_metric(metric)
    value = {
        "miou_macro": miou_macro,
        "miou_w_glacier": miou_w_glacier,
        "miou_invfq_w_glacier": miou_invfq_w_glacier,
    }[name]
    return name, value


def boundary_band(labels: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Split labels into an interior map and the class boundaries removed from it.

    A pixel is on a boundary when any of its eight neighbours carries a
    different valid class. At 30 m a boundary pixel physically mixes several
    surface types, so its single label is partly arbitrary; excluding a band of
    them isolates how much residual error is boundary ambiguity rather than
    interior misclassification.

    Args:
        labels: `(H, W)` uint8, `IGNORE_INDEX` for no-data.
        width: band thickness in pixels. 0 is a no-op.

    Returns:
        `(interior, boundary)`. `interior` has the band set to `IGNORE_INDEX`;
        `boundary` keeps the original labels on the band and `IGNORE_INDEX`
        elsewhere. The two partition the valid pixels.
    """
    width = int(width)
    if width <= 0:
        return labels, np.full_like(labels, IGNORE_INDEX)

    valid = labels != IGNORE_INDEX
    padded_labels = np.pad(labels, 1, constant_values=IGNORE_INDEX)
    padded_valid = np.pad(valid, 1, constant_values=False)
    height, cols = labels.shape

    # Compare against all eight shifts at once instead of looping.
    is_boundary = np.zeros((height, cols), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour = padded_labels[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + cols]
            neighbour_valid = padded_valid[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + cols]
            is_boundary |= (labels != neighbour) & valid & neighbour_valid

    if width > 1:
        is_boundary = _dilate(is_boundary, width - 1)

    boundary = np.full_like(labels, IGNORE_INDEX)
    boundary[is_boundary] = labels[is_boundary]
    interior = labels.copy()
    interior[is_boundary] = IGNORE_INDEX
    return interior, boundary


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-structuring-element binary dilation.

    Separable: dilating rows then columns is O(2r) shifts instead of the
    O((2r+1)^2) of the naive two-dimensional sweep, and gives the same result
    for a square element.
    """
    if radius <= 0:
        return mask

    out = mask
    for axis in (0, 1):
        padded = np.pad(
            out,
            [(radius, radius) if a == axis else (0, 0) for a in range(2)],
            constant_values=False,
        )
        length = out.shape[axis]
        accumulated = np.zeros_like(out)
        for offset in range(2 * radius + 1):
            window = (
                padded[offset : offset + length, :]
                if axis == 0
                else padded[:, offset : offset + length]
            )
            accumulated |= window
        out = accumulated
    return out


def dilated_confusion(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance: int = 2,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Confusion matrix that forgives an error within `tolerance` pixels.

    A predicted class counts as correct if the true class appears anywhere in
    its neighbourhood. This separates "the network chose the wrong class" from
    "the network drew the right boundary one pixel off", which at 30 m is
    frequently within annotation precision.

    Implemented as a max-pool over each class's one-hot plane, so the whole
    batch is handled in one pass per class rather than per pixel.
    """
    if tolerance <= 0:
        return confusion_from_predictions(pred, target, num_classes, ignore_index)

    valid = target != ignore_index
    kernel = 2 * int(tolerance) + 1

    safe_target = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target.long(), num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()
    nearby = F.max_pool2d(one_hot, kernel_size=kernel, stride=1, padding=tolerance) > 0

    pred_index = pred.long().unsqueeze(1)
    pred_is_nearby = torch.gather(nearby, 1, pred_index).squeeze(1)

    # Where the prediction is acceptable within tolerance, score it on the
    # diagonal; otherwise fall back to the strict pairing.
    effective_target = torch.where(pred_is_nearby & valid, pred.long(), safe_target.long())
    effective_target = torch.where(
        valid, effective_target, torch.full_like(effective_target, ignore_index)
    )
    return confusion_from_predictions(pred, effective_target, num_classes, ignore_index)


def compute_boundary_metrics(
    confusion: Optional[np.ndarray],
    class_names: Sequence[str] = CLASS_NAMES,
) -> Optional[dict[str, object]]:
    """Per-class F1 restricted to boundary pixels, plus its macro mean.

    Reported alongside the interior metrics so the two can be compared: a large
    gap indicates residual error is concentrated at class transitions.
    """
    if confusion is None:
        return None

    metrics = compute_confusion_metrics(confusion, class_names=class_names)
    per_class = [
        {
            "class_name": entry["class_name"],
            "boundary_f1": entry["f1"],
            "boundary_iou": entry["iou"],
            "support": entry["support"],
        }
        for entry in metrics["per_class"]
    ]
    f1_values = np.array(
        [entry["boundary_f1"] for entry in per_class], dtype=np.float64
    )
    return {
        "per_class": per_class,
        "macro": {
            "boundary_f1": _round3(_nanmean(f1_values)),
            "total": metrics["macro"]["total"],
        },
    }


# -- calibration -------------------------------------------------------------

#: Confidence bins for the reliability table. Ten equal-width bins over [0, 1]
#: is the conventional choice and enough resolution to see the shape of the
#: curve; the low bins are usually near-empty for a four-class problem, where
#: the argmax probability cannot fall below 0.25.
DEFAULT_CALIBRATION_BINS: int = 10


class CalibrationTally:
    """A running count of (correct, total) per softmax-confidence bin.

    A model's mIoU says how often it is right. It says nothing about whether the
    model *knows* when it is right, which is the question that decides whether a
    per-pixel probability can be thresholded, used to flag scenes for review, or
    propagated into an uncertainty on a published snow fraction. A model that is
    70% accurate while reporting 0.99 confidence everywhere is unusable for all
    three, and no aggregate score distinguishes it from one that is honestly
    uncertain on the hard pixels.

    **Bounded memory is the design constraint.** The obvious implementation
    keeps every pixel's probability and reduces at the end; on this corpus that
    is tens of millions of floats per partition. Instead each batch is reduced
    immediately into three small vectors — per-bin pixel counts, per-bin correct
    counts, per-bin summed confidence — of length `n_bins`. The accumulator is
    the same size after one batch as after ten thousand, and the result is
    identical to the batch-free computation, since counts and sums are exactly
    associative.

    Accumulation runs on whatever device the tensors are already on, so a CUDA
    evaluation pass moves a handful of numbers to the host once per partition
    rather than a prediction volume per batch.

    Bin `b` covers confidence in `[b/n_bins, (b+1)/n_bins)`, with the top bin
    closed at 1.0 so a confidence of exactly 1.0 lands in the last bin rather
    than out of range.
    """

    def __init__(
        self,
        n_bins: int = DEFAULT_CALIBRATION_BINS,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        self.n_bins = int(n_bins)
        self._count = torch.zeros(self.n_bins, dtype=torch.int64, device=device)
        self._correct = torch.zeros(self.n_bins, dtype=torch.float64, device=device)
        self._confidence_sum = torch.zeros(self.n_bins, dtype=torch.float64, device=device)
        # Mean confidence split by whether the prediction was right. The gap
        # between the two says whether the probabilities carry usable
        # information at all, which the per-bin table shows only indirectly.
        self._correct_confidence_sum = 0.0
        self._incorrect_confidence_sum = 0.0
        self._n_correct = 0
        self._n_incorrect = 0

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = IGNORE_INDEX,
    ) -> "CalibrationTally":
        """Fold one batch of per-pixel softmax probabilities into the tally.

        Args:
            probabilities: `(N, C, H, W)` softmax output. The confidence scored
                is the maximum over classes and the prediction is its argmax —
                both taken from the probabilities themselves rather than one of
                them passed in separately, so a confidence can never end up
                describing a different prediction than the one it was measured
                against.
            target: `(N, H, W)` class indices, `ignore_index` for no-data.
        """
        valid = target != ignore_index
        if not bool(valid.any()):
            return self

        confidence, predicted = probabilities.max(dim=1)
        confidence = confidence[valid].to(torch.float64)
        is_correct = predicted[valid] == target[valid]

        # clamp keeps a confidence of exactly 1.0 (and float noise just above
        # it) inside the last bin instead of indexing off the end.
        index = torch.clamp(
            (confidence * self.n_bins).floor().to(torch.int64), 0, self.n_bins - 1
        )

        device = self._count.device
        index = index.to(device)
        confidence = confidence.to(device)
        hit = is_correct.to(device=device, dtype=torch.float64)

        self._count += torch.bincount(index, minlength=self.n_bins)
        self._correct += torch.bincount(index, weights=hit, minlength=self.n_bins)
        self._confidence_sum += torch.bincount(
            index, weights=confidence, minlength=self.n_bins
        )

        correct_mask = is_correct.to(device)
        self._n_correct += int(correct_mask.sum())
        self._n_incorrect += int((~correct_mask).sum())
        self._correct_confidence_sum += float(confidence[correct_mask].sum())
        self._incorrect_confidence_sum += float(confidence[~correct_mask].sum())
        return self

    # -- readouts ------------------------------------------------------------

    @property
    def total(self) -> int:
        """Pixels tallied."""
        return int(self._count.sum())

    def rows(self) -> list[dict[str, Any]]:
        """One row per confidence bin, then an `overall` summary row.

        Each bin row carries its interval, how many pixels fell in it, the mean
        confidence the model reported there, the fraction actually correct, and
        the signed gap between the two. A positive gap is overconfidence — the
        model claimed more certainty than it delivered — which is the failure
        mode worth naming, because it is the one that makes a probability
        unusable as a filter.

        The summary row carries the expected calibration error (the
        count-weighted mean absolute gap over non-empty bins) and the mean
        confidence on correct versus incorrect predictions, whose difference is
        reported as `confidence_separation`.

        Empty bins are written with empty metric cells rather than zeros: no
        pixel landed there, which is not the same as being wrong there.
        """
        count = self._count.cpu().numpy().astype(np.int64)
        correct = np.rint(self._correct.cpu().numpy()).astype(np.int64)
        confidence_sum = self._confidence_sum.cpu().numpy().astype(np.float64)
        total = self.total

        rows: list[dict[str, Any]] = []
        weighted_gap = 0.0
        for index in range(self.n_bins):
            n = int(count[index])
            row: dict[str, Any] = {
                "bin": index,
                "confidence_lower": round(index / self.n_bins, 6),
                "confidence_upper": round((index + 1) / self.n_bins, 6),
                "n_pixels": n,
                "pixel_share_pct": round(100.0 * n / total, 4) if total else 0.0,
                "n_correct": int(correct[index]),
            }
            if n > 0:
                accuracy = float(correct[index]) / n
                mean_confidence = float(confidence_sum[index]) / n
                row["accuracy"] = round(accuracy, 6)
                row["mean_confidence"] = round(mean_confidence, 6)
                row["gap_confidence_minus_accuracy"] = round(
                    mean_confidence - accuracy, 6
                )
                weighted_gap += n * abs(mean_confidence - accuracy)
            else:
                row["accuracy"] = float("nan")
                row["mean_confidence"] = float("nan")
                row["gap_confidence_minus_accuracy"] = float("nan")
            rows.append(row)

        n_correct = self._n_correct
        n_incorrect = self._n_incorrect
        overall_confidence = float(confidence_sum.sum()) / total if total else float("nan")
        overall_accuracy = n_correct / total if total else float("nan")
        rows.append(
            {
                "bin": "overall",
                "confidence_lower": 0.0,
                "confidence_upper": 1.0,
                "n_pixels": total,
                "pixel_share_pct": 100.0 if total else 0.0,
                "n_correct": n_correct,
                "accuracy": round(overall_accuracy, 6) if total else float("nan"),
                "mean_confidence": (
                    round(overall_confidence, 6) if total else float("nan")
                ),
                "gap_confidence_minus_accuracy": (
                    round(overall_confidence - overall_accuracy, 6)
                    if total
                    else float("nan")
                ),
                "expected_calibration_error": (
                    round(weighted_gap / total, 6) if total else float("nan")
                ),
                "mean_confidence_when_correct": (
                    round(self._correct_confidence_sum / n_correct, 6)
                    if n_correct
                    else float("nan")
                ),
                "mean_confidence_when_incorrect": (
                    round(self._incorrect_confidence_sum / n_incorrect, 6)
                    if n_incorrect
                    else float("nan")
                ),
                "confidence_separation": (
                    round(
                        self._correct_confidence_sum / n_correct
                        - self._incorrect_confidence_sum / n_incorrect,
                        6,
                    )
                    if n_correct and n_incorrect
                    else float("nan")
                ),
            }
        )
        return rows

    def summary(self) -> dict[str, float]:
        """The scalar calibration numbers, without the per-bin table."""
        overall = self.rows()[-1]
        return {
            key: float(overall[key])
            for key in (
                "accuracy",
                "mean_confidence",
                "expected_calibration_error",
                "mean_confidence_when_correct",
                "mean_confidence_when_incorrect",
                "confidence_separation",
            )
        }
