"""Running the feature-attribution analyses against a finished run.

`shap_importance` implements the two analyses; this module decides when to run
them, builds the loader they need, and turns their output into export-ready
rows. Keeping that separate means `bootstrap`, `cv` and the single-run path all
request attribution the same way, and the cost policy lives in one place.

**Cost, and why both default to off.** Permutation importance is one full
evaluation pass *per input channel* — eleven channels means eleven extra passes
over the validation partition, so it roughly matches the cost of eleven
training epochs' worth of validation. Expected-gradient attribution is
`n_baselines` backward passes over `n_samples` tiles, which at the defaults here
is small in absolute terms but is still a backward pass, and unlike permutation
it cannot reuse `inference_mode`.

Neither number is large next to a 500-epoch run. Both are large next to the
three-epoch smoke runs and the HPO trials that dominate the number of times
training is invoked, and neither feeds back into training or model selection —
they describe a finished model. So they are opt-in, and the flags that enable
them are the same flags whether one seed or five are being trained.

Attribution is measured on the validation partition by default: test is held
for the final score and should be read once, and training would report how much
the model memorised rather than what it relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .config import CLASS_NAMES, TrainingConfig
from .dataset import PreparedData
from .shap_importance import (
    expected_gradient_attribution,
    permutation_importance_table,
)
from .torch_dataset import TileDataset
from .train import TrainingResult, resolve_device

__all__ = ["AttributionOptions", "compute_attributions"]


@dataclass(frozen=True)
class AttributionOptions:
    """What to compute, on which partition, and how hard to work at it."""

    permutation: bool = False
    gradient: bool = False
    split: str = "valid"  # "train" | "valid" | "test"
    max_batches: int = 0  # 0 uses the whole partition
    seed: int = 0
    shap_samples: int = 64
    shap_baselines: int = 16
    shap_batch_size: int = 8

    @property
    def any_enabled(self) -> bool:
        return self.permutation or self.gradient


def _tiles_for(data: PreparedData, split: str) -> list[tuple[int, int, int, int]]:
    return {
        "train": data.tiles_train,
        "valid": data.tiles_val,
        "test": data.tiles_test,
    }.get(split, data.tiles_val)


def _loader(
    data: PreparedData,
    config: TrainingConfig,
    split: str,
    device: torch.device,
) -> Optional[DataLoader]:
    tiles = _tiles_for(data, split)
    if not tiles:
        return None

    dataset = TileDataset(
        tiles,
        features=data.features,
        labels=data.labels,
        mean=data.mean,
        std=data.std,
        patch_size=config.patch_size,
        sensor_norm_stats=data.sensor_norm_stats,
        scene_sensors=data.scene_sensors,
        scene_context=data.scene_context,
    )
    # Deliberately single-process: permutation importance re-iterates this
    # loader once per channel, and respawning a worker pool per channel costs
    # more than the loading it parallelises for a partition of this size.
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def compute_attributions(
    result: TrainingResult,
    config: TrainingConfig,
    data: PreparedData,
    options: Optional[AttributionOptions],
) -> tuple[Optional[list[dict[str, Any]]], Optional[list[dict[str, Any]]]]:
    """Run the enabled analyses against `result.model`.

    Returns `(permutation_rows, attribution_rows)`, either of which is None
    when that analysis was not requested or could not run. A failure in either
    is reported and swallowed: attribution describes a model that has already
    been trained and saved, and losing the description is not a reason to lose
    the run.
    """
    if options is None or not options.any_enabled:
        return None, None

    device = resolve_device(config.device)
    loader = _loader(data, config, options.split, device)
    if loader is None:
        print(f"[attribution] partition {options.split!r} has no tiles; skipping")
        return None, None

    model = result.model.to(device)
    features = list(result.features)

    permutation_rows: Optional[list[dict[str, Any]]] = None
    if options.permutation:
        try:
            print(
                f"[attribution] permutation importance over {len(features)} channels "
                f"on {options.split}"
            )
            permutation_rows = permutation_importance_table(
                model,
                loader,
                features,
                device,
                seed=options.seed,
                use_amp=config.use_amp and device.type == "cuda",
                use_context=config.use_spatial_context,
                max_batches=options.max_batches,
                class_names=CLASS_NAMES,
            )
        except Exception as exc:  # noqa: BLE001 - the trained model is unaffected
            print(f"[attribution] permutation importance failed: {exc}")

    attribution_rows: Optional[list[dict[str, Any]]] = None
    if options.gradient:
        try:
            print(
                f"[attribution] expected-gradient attribution on {options.split} "
                f"({options.shap_samples} tiles x {options.shap_baselines} baselines)"
            )
            attribution = expected_gradient_attribution(
                model,
                loader,
                features,
                device,
                n_samples=options.shap_samples,
                n_baselines=options.shap_baselines,
                batch_size=options.shap_batch_size,
                seed=options.seed,
                use_context=config.use_spatial_context,
                class_names=CLASS_NAMES,
            )
            attribution_rows = attribution.rows()
        except Exception as exc:  # noqa: BLE001 - as above
            print(f"[attribution] gradient attribution failed: {exc}")

    return permutation_rows, attribution_rows
