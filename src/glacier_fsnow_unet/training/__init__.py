"""U-Net training for four-class glacier surface segmentation.

Entry points, in the order a run uses them:

- `config`: `TrainingConfig`, built from the repo's central `config.yaml`.
- `dataset`: scene discovery, tiling, and the glacier-level stratified split.
- `torch_dataset`: the `Dataset` and the balanced samplers.
- `model_architectures`: the attention U-Net and its factory.
- `losses`: weighted cross-entropy plus the confusion-pair penalties.
- `metrics`: confusion matrices, IoU variants, boundary metrics.
- `train`: `train_unified`, one training run.
- `breakdown`: per-scene, per-glacier and per-sensor metric disaggregation.
- `hpo` / `cv` / `bootstrap`: search, k-fold, and multi-seed orchestration.
- `export`: checkpoint writing and the full set of CSV/JSON run exports.
- `shap_importance`: permutation importance and expected-gradient attribution.
- `attribution_runner`: when to run those, and on which partition.

Everything is imported normally; there is no dynamic loading anywhere.
"""

from __future__ import annotations

from .config import (
    CLASS_NAMES,
    DEFAULT_FEATURES,
    IGNORE_INDEX,
    NUM_CLASSES,
    NUM_SENSORS,
    SENSOR_TO_IDX,
    LossPenalties,
    SearchSpaceEntry,
    TrainingConfig,
    training_config_from_pipeline,
)
from .breakdown import SceneConfusions, SceneIdentity, build_breakdown
from .dataset import PreparedData, SceneRecord, prepare_data, scan_scenes, split_scenes
from .model_architectures import UNet, build_model
from .train import TrainingResult, seed_everything, train_unified

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_FEATURES",
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "NUM_SENSORS",
    "SENSOR_TO_IDX",
    "LossPenalties",
    "PreparedData",
    "SceneConfusions",
    "SceneIdentity",
    "SceneRecord",
    "SearchSpaceEntry",
    "TrainingConfig",
    "TrainingResult",
    "UNet",
    "build_breakdown",
    "build_model",
    "prepare_data",
    "scan_scenes",
    "seed_everything",
    "split_scenes",
    "train_unified",
    "training_config_from_pipeline",
]
