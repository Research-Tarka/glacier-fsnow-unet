"""Feature attribution: permutation importance and expected-gradient attribution.

Two complementary answers to "how much does this spectral index actually
contribute?", reported per feature *and per class* so a channel that only
matters for Ice is not hidden behind a macro average.

**Permutation importance** shuffles one input channel across the batch and
measures how far IoU falls. A channel the model relies on causes a large drop; a
redundant one causes almost none. Permutation rather than zeroing: setting a
channel to zero moves it outside the distribution the network was trained on, so
the drop conflates "this feature mattered" with "this input is now nonsense".
Shuffling keeps each channel's marginal distribution intact and destroys only its
correspondence with the label, which is the thing being measured.

**Expected-gradient attribution** integrates the input gradient along straight
paths from randomly drawn baselines to the actual input, which is the Shapley
value of each channel under the model's own local linearity. It answers a
different question than permutation does: permutation measures how much *the
score degrades* without a feature, integrated gradients measure how much *the
logit is built* from it. A feature can be redundant with another (low permutation
importance, because the twin covers for it) while still carrying real signal
(non-zero attribution). Reporting both is what makes that distinguishable.

**Why expected gradients rather than a library.** The published SHAP explainers
(`shap.GradientExplainer`, `captum.GradientShap`) both assume one scalar output
per sample. This model emits a dense `(B, C, H, W)` logit volume, so using them
means reshaping the segmentation output into a scalar per class and per sample
anyway — which is precisely the spatial mean this module takes. Doing that
reduction explicitly is a few dozen lines of plain autograd, keeps the class-wise
reduction under our own control, composes with the AMP and context plumbing the
rest of the training code uses, and adds no dependency to a training environment
that otherwise needs none. The estimator implemented here is the standard
expected-gradients formulation: the Shapley-value approximation obtained by
averaging path gradients over baselines drawn from the data distribution.

Both functions are exact about their determinism: every random draw comes from an
explicitly seeded generator, so two runs at the same seed produce identical
tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from .metrics import compute_confusion_metrics, confusion_from_predictions

__all__ = [
    "permutation_importance",
    "permutation_importance_table",
    "expected_gradient_attribution",
    "FeatureImportance",
    "FeatureAttribution",
]


class FeatureImportance(dict):
    """Per-feature macro importance scores, with convenience ordering."""

    @property
    def ranked(self) -> list[tuple[str, float]]:
        """Features from most to least important."""
        return sorted(self.items(), key=lambda kv: kv[1], reverse=True)

    def normalised(self) -> dict[str, float]:
        """Scores rescaled to percentages summing to 100.

        Only positive drops contribute: a negative drop means the model did
        marginally better without the feature, which is noise, not evidence
        that the feature is harmful.
        """
        positive = {name: max(0.0, value) for name, value in self.items()}
        total = sum(positive.values())
        if total <= 0:
            return {name: 0.0 for name in self}
        return {name: 100.0 * value / total for name, value in positive.items()}


@dataclass
class FeatureAttribution:
    """A per-feature x per-class attribution table.

    `values` is `(n_features, n_classes)`, already reduced over pixels and
    samples. `feature_names` and `class_names` label its axes.
    """

    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    values: np.ndarray
    method: str
    n_samples: int
    n_baselines: int

    def rows(self) -> list[dict[str, Any]]:
        """Flatten to one record per (feature, class), for CSV writing."""
        out: list[dict[str, Any]] = []
        totals = np.abs(self.values).sum()
        for f_index, feature in enumerate(self.feature_names):
            feature_total = float(np.abs(self.values[f_index]).sum())
            for c_index, class_name in enumerate(self.class_names):
                value = float(self.values[f_index, c_index])
                out.append(
                    {
                        "feature": feature,
                        "class": c_index,
                        "class_name": class_name,
                        "attribution": value,
                        "abs_attribution": abs(value),
                        "share_pct": (
                            100.0 * abs(value) / float(totals) if totals > 0 else 0.0
                        ),
                        "feature_abs_total": feature_total,
                        "method": self.method,
                    }
                )
        return out

    def macro(self) -> dict[str, float]:
        """Mean absolute attribution per feature, across classes."""
        return {
            name: float(np.abs(self.values[index]).mean())
            for index, name in enumerate(self.feature_names)
        }


# -- permutation importance --------------------------------------------------


@torch.inference_mode()
def _confusion_over_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    permute_channel: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    use_amp: bool = False,
    use_context: bool = False,
    max_batches: int = 0,
) -> np.ndarray:
    """Pooled confusion matrix, optionally with one channel shuffled.

    The shuffle permutes the channel across the batch dimension, so each tile
    receives another tile's values for that index while every other channel
    stays put.
    """
    model.eval()
    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device
    )

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        context = (
            batch[2].to(device, non_blocking=True)
            if len(batch) > 2 and torch.is_tensor(batch[2])
            else None
        )

        if permute_channel is not None and x.size(0) > 1:
            order = torch.randperm(x.size(0), generator=generator, device="cpu").to(x.device)
            x = x.clone()
            x[:, permute_channel] = x[order, permute_channel]

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(x, context=context) if use_context else model(x)

        confusion += confusion_from_predictions(
            logits.argmax(dim=1), y, NUM_CLASSES, IGNORE_INDEX
        )

    return confusion.cpu().numpy()


def _per_class_iou(confusion: np.ndarray) -> dict[str, float]:
    """Map class name to IoU, from a pooled confusion matrix."""
    metrics = compute_confusion_metrics(confusion)
    return {
        str(entry["class_name"]): float(entry["iou"])
        for entry in metrics["per_class"]  # type: ignore[union-attr]
    }


def permutation_importance(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: Sequence[str],
    device: torch.device,
    seed: int = 0,
    use_amp: bool = False,
    use_context: bool = False,
    max_batches: int = 0,
) -> FeatureImportance:
    """Drop in macro mIoU when each feature channel is shuffled.

    Args:
        loader: a validation loader. Reused for every channel, so it must be
            re-iterable.
        feature_names: channel names, in channel order.
        seed: each channel is permuted under the same seed, so the comparison
            between channels is not confounded by different shuffles.
        max_batches: cap the batches evaluated per channel. 0 uses all.

    Returns:
        `FeatureImportance` mapping name to `baseline_mIoU - permuted_mIoU`.
        Larger means more important; near-zero means redundant.
    """
    table = permutation_importance_table(
        model,
        loader,
        feature_names,
        device,
        seed=seed,
        use_amp=use_amp,
        use_context=use_context,
        max_batches=max_batches,
    )
    scores = FeatureImportance()
    for row in table:
        if row["class_name"] == "macro":
            scores[str(row["feature"])] = float(row["importance_delta_iou"])
    return scores


def permutation_importance_table(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: Sequence[str],
    device: torch.device,
    seed: int = 0,
    use_amp: bool = False,
    use_context: bool = False,
    max_batches: int = 0,
    class_names: Sequence[str] = CLASS_NAMES,
) -> list[dict[str, Any]]:
    """Per-feature, per-class IoU drop when a channel is shuffled.

    The per-class breakdown is the point: a channel can be indispensable for one
    class and irrelevant to the other three, which a single macro number hides
    entirely. `Index_NDSI` separating Snow from Ice is exactly that shape.

    One evaluation pass per channel plus one baseline pass, and every class's
    IoU is read off the same pooled confusion matrix, so the per-class detail
    costs nothing beyond the macro number.

    Returns one row per (feature, class) plus one `macro` row per feature, each
    with `feature`, `class`, `class_name`, `importance_delta_iou`,
    `baseline_iou` and `permuted_iou`.
    """
    baseline_confusion = _confusion_over_loader(
        model, loader, device, None, None, use_amp, use_context, max_batches
    )
    baseline_metrics = compute_confusion_metrics(baseline_confusion)
    baseline_per_class = _per_class_iou(baseline_confusion)
    baseline_macro = float(baseline_metrics["macro"]["miou"])  # type: ignore[index]

    rows: list[dict[str, Any]] = []
    for channel, name in enumerate(feature_names):
        generator = torch.Generator()
        generator.manual_seed(seed + channel)
        permuted_confusion = _confusion_over_loader(
            model, loader, device, channel, generator, use_amp, use_context, max_batches
        )
        permuted_metrics = compute_confusion_metrics(permuted_confusion)
        permuted_per_class = _per_class_iou(permuted_confusion)
        permuted_macro = float(permuted_metrics["macro"]["miou"])  # type: ignore[index]

        for class_index, class_name in enumerate(class_names):
            base = baseline_per_class.get(str(class_name), float("nan"))
            perm = permuted_per_class.get(str(class_name), float("nan"))
            rows.append(
                {
                    "feature": str(name),
                    "class": class_index,
                    "class_name": str(class_name),
                    "importance_delta_iou": float(base - perm),
                    "baseline_iou": float(base),
                    "permuted_iou": float(perm),
                }
            )

        rows.append(
            {
                "feature": str(name),
                "class": -1,
                "class_name": "macro",
                "importance_delta_iou": float(baseline_macro - permuted_macro),
                "baseline_iou": baseline_macro,
                "permuted_iou": permuted_macro,
            }
        )

    return rows


# -- expected-gradient attribution -------------------------------------------


def _collect_samples(
    loader: Iterable[Any],
    limit: int,
    device: torch.device,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stack up to `limit` tiles (and their context vectors) from a loader."""
    inputs: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    seen = 0

    for batch in loader:
        x = batch[0]
        context = batch[2] if len(batch) > 2 and torch.is_tensor(batch[2]) else None
        take = min(int(x.size(0)), max(0, limit - seen))
        if take <= 0:
            break
        inputs.append(x[:take].detach())
        if context is not None:
            contexts.append(context[:take].detach())
        seen += take
        if seen >= limit:
            break

    if not inputs:
        raise ValueError("the loader yielded no samples for attribution")

    stacked = torch.cat(inputs, dim=0).to(device)
    stacked_context = (
        torch.cat(contexts, dim=0).to(device)
        if contexts and len(contexts) == len(inputs)
        else None
    )
    return stacked, stacked_context


def expected_gradient_attribution(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_names: Sequence[str],
    device: torch.device,
    n_samples: int = 64,
    n_baselines: int = 16,
    batch_size: int = 8,
    seed: int = 0,
    use_context: bool = False,
    class_names: Sequence[str] = CLASS_NAMES,
) -> FeatureAttribution:
    """Expected-gradient (Shapley-value) attribution per feature and class.

    For each sample `x`, a baseline `b` drawn from the same data distribution,
    and an interpolation coefficient `a ~ U(0, 1)`, the estimator accumulates

        (x - b) * d/dz mean_pixels(logit_c(z))   at   z = b + a * (x - b)

    and averages over draws. That is the expected-gradients formulation of the
    Shapley value: the path integral of the gradient from baseline to input,
    with the baseline marginalised over the data rather than fixed at zero. A
    fixed zero baseline would sit outside the distribution of normalised
    spectral indices, and would attribute to a channel merely for being
    non-zero.

    The model's output is reduced to one scalar per (sample, class) by averaging
    the class logit over the tile's pixels, so the gradient answers "how much of
    this tile's average evidence for class c came from this channel".

    Deliberately runs in float32 with autocast off: the accumulated gradient is
    a small quantity summed over many draws, and float16 rounding at that scale
    changes the ranking, not just the digits.

    Args:
        n_samples: tiles to explain. Cost is linear in this.
        n_baselines: path draws per sample. Cost is linear in this too; the
            estimator's variance falls as `1 / sqrt(n_baselines)`.
        batch_size: tiles per backward pass, to bound memory.
        seed: seeds the baseline pairing and the interpolation coefficients.

    Returns:
        `FeatureAttribution` with a `(n_features, n_classes)` value array. Each
        entry is the mean signed attribution over samples and pixels.
    """
    model.eval()
    n_features = len(feature_names)
    n_classes = len(class_names)

    samples, contexts = _collect_samples(loader, int(n_samples), device)
    n_actual = int(samples.size(0))
    # Baselines are drawn from the same pool. With one sample there is no other
    # tile to interpolate from, so the attribution is defined as zero rather
    # than silently interpolating a sample against itself.
    if n_actual < 2:
        return FeatureAttribution(
            feature_names=tuple(str(f) for f in feature_names),
            class_names=tuple(str(c) for c in class_names),
            values=np.zeros((n_features, n_classes), dtype=np.float64),
            method="expected_gradients",
            n_samples=n_actual,
            n_baselines=0,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    totals = torch.zeros((n_features, n_classes), dtype=torch.float64, device=device)
    draws = 0

    for draw in range(max(1, int(n_baselines))):
        # Pair each sample with a different sample as its baseline. A rolled
        # permutation guarantees no sample is its own baseline, which would
        # make the path length zero and contribute nothing but noise.
        order = torch.randperm(n_actual, generator=generator)
        baseline_index = torch.roll(order, shifts=1 + draw % max(1, n_actual - 1))
        baseline_index = baseline_index.to(device)

        for start in range(0, n_actual, max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), n_actual)
            x = samples[start:stop].float()
            baseline = samples.index_select(0, baseline_index[start:stop]).float()

            alpha = torch.rand(
                (x.size(0), 1, 1, 1), generator=generator, dtype=torch.float32
            ).to(device)
            delta = x - baseline
            point = (baseline + alpha * delta).requires_grad_(True)

            context = (
                contexts[start:stop] if (use_context and contexts is not None) else None
            )

            # Autocast off on purpose - see the docstring.
            with autocast(device_type=device.type, enabled=False):
                logits = model(point, context=context) if use_context else model(point)

            for class_index in range(min(n_classes, int(logits.size(1)))):
                scalar = logits[:, class_index].mean(dim=(1, 2)).sum()
                (grad,) = torch.autograd.grad(
                    scalar, point, retain_graph=class_index < n_classes - 1
                )
                # (B, C, H, W) -> per-channel contribution, averaged over pixels
                # and summed over the batch.
                contribution = (grad.detach() * delta).mean(dim=(2, 3)).sum(dim=0)
                totals[: contribution.numel(), class_index] += contribution.to(
                    torch.float64
                )

            draws += x.size(0)

    values = (totals / max(1, draws)).detach().cpu().numpy()
    return FeatureAttribution(
        feature_names=tuple(str(f) for f in feature_names),
        class_names=tuple(str(c) for c in class_names),
        values=values,
        method="expected_gradients",
        n_samples=n_actual,
        n_baselines=int(n_baselines),
    )
