"""Loss terms for four-class glacier surface segmentation.

The training objective is class-weighted cross-entropy plus a set of targeted
penalties on specific confusion pairs. The penalties exist because not all
errors cost the same downstream: snow-fraction is a ratio of Snow to Snow+Ice
pixels, so a Snow/Ice mix-up moves the published quantity directly, while a
Cloud/Other mix-up decides whether a whole scene is usable at all. Plain
per-pixel cross-entropy treats all four classes as interchangeable and has no
way to express that.

Every penalty has the same shape: given ground truth class `s`, penalise the
softmax probability assigned to class `t`, averaged over the pixels whose true
class is `s`. That single primitive is `directional_penalty`; the nine named
functions are thin, self-documenting wrappers over it. Writing them out
longhand nine times, as the original did, made it easy for the wrong class
index to go unnoticed in one of them.

Class weighting follows `w_k = N_total / (num_classes * N_k)` over the training
corpus, so a class occupying a tenth of the pixels gets ten times the weight.

`deep_supervision_loss` is a separate, optional term added only when the model
carries auxiliary decoder heads. It reuses the same weighted cross-entropy, so
a rare class is weighted identically at every depth, and it reduces the labels
to each head's resolution by majority vote — see `downsample_labels` for why
neither averaging nor nearest-neighbour sampling is correct for categorical
labels.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from .config import (
    CLOUD,
    ICE,
    IGNORE_INDEX,
    NUM_CLASSES,
    OTHER,
    SNOW,
    LossPenalties,
)

__all__ = [
    "directional_penalty",
    "anti_other_loss",
    "ice_from_snow_loss",
    "snow_from_ice_loss",
    "other_from_ice_loss",
    "other_from_cloud_loss",
    "cloud_from_other_loss",
    "cloud_from_snow_loss",
    "cloud_from_ice_loss",
    "pessimistic_state_loss",
    "cloud_safety_loss",
    "weighted_cross_entropy",
    "compute_class_weights",
    "total_penalty",
    "downsample_labels",
    "deep_supervision_loss",
    "DEEP_SUPERVISION_WEIGHTS",
]

#: Loss weight per auxiliary head, coarsest level first. The heads sit at 1/4
#: and 1/2 of the input resolution; the coarser a head is, the less its
#: prediction constrains the output that is actually scored, so it carries the
#: smaller weight.
#:
#: Chosen to be small in total. The two terms sum to 0.5 against the primary
#: loss's 1.0, so the objective's magnitude rises by at most half and the
#: already-tuned learning rate stays in range — a weighting that doubled the
#: loss would effectively be a learning-rate change wearing an architecture
#: change's name, and the comparison against the reference would no longer be
#: about deep supervision.
DEEP_SUPERVISION_WEIGHTS: tuple[float, ...] = (0.2, 0.3)


def directional_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    true_class: Optional[int],
    predicted_class: int,
    beta: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalise probability mass on `predicted_class` where truth is `true_class`.

    This is the single primitive behind every named confusion penalty:

        beta * mean over {pixels with target == true_class} of P(predicted_class)

    Passing `true_class=None` selects every valid pixel whose class is *not*
    `predicted_class`, which is what the "don't over-predict Other" term wants.

    Args:
        logits: (N, C, H, W) raw scores.
        targets: (N, H, W) class indices, `ignore_index` for no-data.
        true_class: the ground-truth class to restrict to, or None for
            "any class other than `predicted_class`".
        predicted_class: the class whose probability is penalised.
        beta: penalty weight. Zero returns zero without computing a softmax.
        probs: precomputed `softmax(logits, dim=1)`, to share one softmax
            across several penalties in the same step.

    Returns:
        Scalar tensor. Zero when the weight is zero or no pixel qualifies.
    """
    if beta <= 0.0:
        return logits.new_zeros(())

    valid = targets != ignore_index
    mask = valid & (targets != predicted_class) if true_class is None else valid & (targets == true_class)
    if not torch.any(mask):
        return logits.new_zeros(())

    if probs is None:
        probs = torch.softmax(logits, dim=1)
    return beta * probs[:, predicted_class][mask].mean()


# -- the nine named penalties -------------------------------------------------
# Each names one confusion direction. The wrapper exists so call sites and
# tests read as glaciology rather than as class-index arithmetic.


def anti_other_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Other anywhere the truth is not Other.

    Other is a residual category defined by exclusion, so it is the easy
    answer for any pixel the network finds ambiguous. Left unchecked it absorbs
    dark ice and shadowed snow.
    """
    return directional_penalty(logits, targets, None, OTHER, beta, ignore_index, probs)


def ice_from_snow_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Ice where the truth is Snow.

    Depresses the snow-fraction numerator, biasing the published quantity low.
    """
    return directional_penalty(logits, targets, SNOW, ICE, beta, ignore_index, probs)


def snow_from_ice_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Snow where the truth is Ice.

    The dominant error mode by volume, and it inflates snow fraction.
    """
    return directional_penalty(logits, targets, ICE, SNOW, beta, ignore_index, probs)


def other_from_ice_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Other where the truth is Ice.

    Debris-laden or fragmented ice reflects much like bare rock at 30 m. This
    is the one penalty active in the reference configuration, at weight 10:
    losing ice pixels to Other shrinks the snow-fraction denominator and so
    also biases the result high.
    """
    return directional_penalty(logits, targets, ICE, OTHER, beta, ignore_index, probs)


def other_from_cloud_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Other where the truth is Cloud."""
    return directional_penalty(logits, targets, CLOUD, OTHER, beta, ignore_index, probs)


def cloud_from_other_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Cloud where the truth is Other.

    False cloud discards otherwise usable scenes at the compositing stage.
    """
    return directional_penalty(logits, targets, OTHER, CLOUD, beta, ignore_index, probs)


def cloud_from_snow_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Cloud where the truth is Snow.

    Bright snow and thin cirrus are the classic confusion that threshold-based
    external cloud masks fail on, which is why cloud is learned here at all.
    """
    return directional_penalty(logits, targets, SNOW, CLOUD, beta, ignore_index, probs)


def cloud_from_ice_loss(logits, targets, beta=1.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise predicting Cloud where the truth is Ice."""
    return directional_penalty(logits, targets, ICE, CLOUD, beta, ignore_index, probs)


def cloud_safety_loss(logits, targets, beta=2.0, ignore_index=IGNORE_INDEX, probs=None):
    """Penalise combined Ice+Other mass on pixels whose truth is Cloud.

    A coarser relative of the two cloud-source penalties: rather than naming a
    single wrong class it pushes down the whole "definitely not cloud" mass,
    steering residual uncertainty on cloud pixels toward Snow (spectrally the
    nearest neighbour) instead of toward a glacier surface class.
    """
    if beta <= 0.0:
        return logits.new_zeros(())

    mask = (targets != ignore_index) & (targets == CLOUD)
    if not torch.any(mask):
        return logits.new_zeros(())

    if probs is None:
        probs = torch.softmax(logits, dim=1)
    return beta * (probs[:, ICE] + probs[:, OTHER])[mask].mean()


def pessimistic_state_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 3.0,
    ignore_index: int = IGNORE_INDEX,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Asymmetric penalty on the softmax-expected melt-severity state.

    Treats the class index as an ordinal melt-severity scale (Cloud < Snow <
    Ice < Other), takes the probability-weighted expected state, and
    penalises predicting a *less* severe state than the truth `alpha` times
    as hard as predicting a more severe one. The asymmetry follows from what
    an error costs: overstating remaining snow makes a retreating glacier
    look healthier than it is.

    Cloud pixels are excluded — cloud is not a point on the melt-severity
    scale, it is the absence of an observation.

    Not part of the reference objective; retained as a configurable term.
    """
    valid = targets != ignore_index
    mask = valid & (targets != CLOUD)
    if not torch.any(mask):
        return logits.new_zeros(())

    if probs is None:
        probs = torch.softmax(logits, dim=1)

    states = torch.arange(
        logits.size(1), device=logits.device, dtype=probs.dtype
    ).view(1, -1, 1, 1)
    expected = torch.sum(probs * states, dim=1)

    diff = expected[mask] - targets.to(expected.dtype)[mask]
    return torch.where(diff < 0, alpha * diff.abs(), diff.abs()).mean()


def compute_class_weights(
    class_counts: Mapping[int, int],
    num_classes: int = NUM_CLASSES,
) -> list[float]:
    """Inverse-frequency class weights, `w_k = N_total / (num_classes * N_k)`.

    A class with an exactly average share gets weight 1. Absent classes are
    floored at a count of 1 rather than producing an infinite weight.
    """
    total = sum(int(class_counts.get(k, 0)) for k in range(num_classes))
    if total <= 0:
        return [1.0] * num_classes
    return [
        total / max(1, num_classes * int(class_counts.get(k, 0)))
        for k in range(num_classes)
    ]


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Class-weighted cross-entropy, averaged over valid pixels only.

    `reduction="mean"` with a `weight` normalises by the sum of weights rather
    than the pixel count, which makes the loss magnitude depend on the class
    mix of each batch. Reducing manually over the valid mask keeps the scale
    comparable from batch to batch, and returns a clean zero for an
    all-no-data batch instead of a NaN.
    """
    valid = targets != ignore_index
    if not torch.any(valid):
        return logits.new_zeros(())

    per_pixel = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    )
    return per_pixel[valid].mean()


def total_penalty(
    logits: torch.Tensor,
    targets: torch.Tensor,
    penalties: LossPenalties,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Sum of every enabled confusion penalty.

    Returns zero without touching the logits when all weights are zero. When
    any are active, one softmax is computed and shared across all of them —
    the original evaluated a separate softmax per term, which on a
    (96, 4, 48, 48) batch is seven redundant passes.
    """
    if not penalties.any_active():
        return logits.new_zeros(())

    probs = torch.softmax(logits, dim=1)
    terms = (
        (None, OTHER, penalties.other_when_not_other),
        (SNOW, ICE, penalties.ice_instead_of_snow),
        (ICE, SNOW, penalties.snow_instead_of_ice),
        (ICE, OTHER, penalties.other_instead_of_ice),
        (CLOUD, OTHER, penalties.other_instead_of_cloud),
        (OTHER, CLOUD, penalties.cloud_instead_of_other),
        (SNOW, CLOUD, penalties.cloud_instead_of_snow),
        (ICE, CLOUD, penalties.cloud_instead_of_ice),
    )

    total = logits.new_zeros(())
    for true_class, predicted_class, beta in terms:
        if beta > 0.0:
            total = total + directional_penalty(
                logits, targets, true_class, predicted_class, beta, ignore_index, probs
            )
    return total


# -- deep supervision ---------------------------------------------------------


def downsample_labels(
    targets: torch.Tensor,
    size: tuple[int, int],
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Reduce a label map to `size` by majority vote within each cell.

    Class indices are categorical, so the two obvious reductions are both
    wrong. Averaging them is meaningless — the mean of Cloud (0) and Ice (2) is
    Snow (1), a class neither pixel belonged to. Nearest-neighbour sampling is
    at least type-correct but throws away every pixel it does not land on, which
    at 1/4 resolution is fifteen of every sixteen, and makes the auxiliary
    target depend on an arbitrary alignment choice rather than on the region.

    Majority vote uses all of them: each output cell takes the class holding the
    most pixels in the region that maps to it. Implemented as adaptive average
    pooling over a one-hot encoding, which is exactly a per-class count within
    each cell, followed by an argmax.

    `ignore_index` is handled by counting ignored pixels as their own additional
    channel and competing on equal terms. A cell that is mostly no-data becomes
    `ignore_index` and is skipped by the loss; a cell with a real majority class
    keeps it even if some of its pixels were ignored. Neither direction leaks:
    an ignored pixel can never be promoted into a real class it did not hold,
    and a real class is never suppressed by a minority of no-data.

    Ties go to the lowest index among the tied classes, with `ignore_index`
    ranked last so a cell that is exactly half annotated stays annotated.
    """
    if targets.shape[-2:] == torch.Size(size):
        return targets

    valid = targets != ignore_index
    safe = torch.where(valid, targets, torch.zeros_like(targets))

    # (B, num_classes + 1, H, W): one channel per class, plus one for no-data.
    one_hot = F.one_hot(safe.long(), num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()
    counts = torch.cat([one_hot, (~valid).unsqueeze(1).float()], dim=1)

    # Average pooling over a one-hot map is the per-class share within each
    # cell, which ranks identically to the count and needs no cell-size term.
    pooled = F.adaptive_avg_pool2d(counts, size)
    winner = pooled.argmax(dim=1)

    return torch.where(
        winner == num_classes,
        torch.full_like(winner, ignore_index),
        winner,
    ).to(targets.dtype)


def deep_supervision_loss(
    aux_logits: "list[torch.Tensor]",
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    weights: "tuple[float, ...]" = DEEP_SUPERVISION_WEIGHTS,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Weighted cross-entropy over the auxiliary decoder heads.

    Each head is scored against the ground truth reduced to that head's own
    resolution, rather than the head's logits being upsampled to full
    resolution. Upsampling would score the head on detail it never had the
    resolution to represent, which is the opposite of what deep supervision is
    for: the point is to ask each level for a correct prediction *at its own
    scale*, so a coarse level learns coarse structure instead of being
    penalised for missing fine boundaries.

    Uses the same class weighting as the primary loss, so a rare class is not
    weighted one way at the output and another two levels up.

    Args:
        aux_logits: one `(N, C, h, w)` tensor per auxiliary head, in the order
            `weights` describes.
        weights: per-head loss weight. Extra heads beyond its length are
            skipped rather than defaulting to a weight, which would put an
            unstated number into the objective.
    """
    if not aux_logits:
        return targets.new_zeros((), dtype=torch.float32)

    total = aux_logits[0].new_zeros(())
    for logits, head_weight in zip(aux_logits, weights):
        if head_weight <= 0.0:
            continue
        reduced = downsample_labels(
            targets,
            (int(logits.shape[-2]), int(logits.shape[-1])),
            num_classes=int(logits.shape[1]),
            ignore_index=ignore_index,
        )
        total = total + head_weight * weighted_cross_entropy(
            logits, reduced, weight, ignore_index
        )
    return total
