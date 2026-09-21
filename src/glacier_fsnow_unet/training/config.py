"""Training run configuration.

`TrainingConfig` is the single immutable object that a training run is
parameterised by. It is built from the repo's central `configs/config.yaml`
(via `glacier_fsnow_unet.config.PipelineConfig`) rather than parsed ad hoc, so
there is exactly one place where a hyperparameter can enter the pipeline and no
opportunity for a code-level default to silently disagree with the YAML.

Two design choices are deliberate and worth stating, because getting either
wrong has historically produced a model that trains fine but scores worse:

*No silent fallbacks.* Every hyperparameter that affects the result is a
required field with a literal default that matches the reference configuration.
There is no "if the caller did not set it, derive something plausible" branch —
that pattern is how a tile-selection threshold can end up 368x larger than
intended without raising anything.

*One name per concept, end to end.* Each loss penalty has a single field name
from YAML through to the loss function. No renaming layer sits in between, so
there is nothing to bypass and no path on which a penalty quietly reverts to a
hard-coded default.

See `docs/decisions/training_active_path.md` for the reference values and
`docs/decisions/training_reproducibility_audit.md` for why both rules exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Training-label encoding. The ordering is a melt-severity ordering
# (Cloud < Snow < Ice < Other) that the state-error metrics rely on: they read
# the signed distance `predicted - true` in exactly this order.
CLASS_NAMES: tuple[str, ...] = ("Cloud", "Snow", "Ice", "Other")
NUM_CLASSES: int = len(CLASS_NAMES)
IGNORE_INDEX: int = 255

CLOUD, SNOW, ICE, OTHER = 0, 1, 2, 3

# Raw annotation values are 1..4 with 0 meaning "not annotated".
RAW_TO_TRAIN_LABEL: Mapping[int, int] = {1: CLOUD, 2: SNOW, 3: ICE, 4: OTHER}

# The 11 normalised spectral indices, in channel order. This order is
# load-bearing: it is baked into the first convolution of every trained
# checkpoint and into the SHAP attribution table.
DEFAULT_FEATURES: tuple[str, ...] = (
    "Index_NDVI",
    "Index_NDSI",
    "Index_NDWI_GAO",
    "Index_NBR",
    "Index_ND_SWIR1_SWIR2",
    "Index_ND_BLUE_RED",
    "Index_ND_BLUE_NIR",
    "Index_ND_GREEN_RED",
    "Index_r",
    "Index_g",
    "Index_b",
)

# Sensor label -> index, used to route per-sensor BatchNorm and FiLM embeddings.
# Only consulted when per-sensor adaptation is enabled; the reference
# configuration collapses every sensor to index 0.
SENSOR_TO_IDX: Mapping[str, int] = {
    "landsat": 0,  # Landsat 8/9
    "landsat5": 1,
    "landsat7": 2,
    "sentinel": 3,
    "map": 4,
}
NUM_SENSORS: int = max(SENSOR_TO_IDX.values()) + 1

# Sensors that the split balancer tries to represent in every partition.
BALANCED_SENSORS: tuple[str, ...] = ("landsat", "landsat5", "landsat7", "sentinel")


@dataclass(frozen=True)
class LossPenalties:
    """Weights for the nine confusion-aware auxiliary loss terms.

    Each term penalises probability mass placed on one class when the ground
    truth is another. A weight of 0 disables that term entirely (the term is
    skipped, not multiplied by zero, so it costs nothing).

    The reference configuration activates exactly one of these:
    `other_instead_of_ice = 10`. Every other weight is 0.
    """

    other_when_not_other: float = 0.0
    ice_instead_of_snow: float = 0.0
    snow_instead_of_ice: float = 0.0
    other_instead_of_ice: float = 10.0
    other_instead_of_cloud: float = 0.0
    cloud_instead_of_other: float = 0.0
    cloud_instead_of_snow: float = 0.0
    cloud_instead_of_ice: float = 0.0
    cloud_dem_independence: float = 0.0

    def any_active(self) -> bool:
        return any(
            w > 0.0
            for w in (
                self.other_when_not_other,
                self.ice_instead_of_snow,
                self.snow_instead_of_ice,
                self.other_instead_of_ice,
                self.other_instead_of_cloud,
                self.cloud_instead_of_other,
                self.cloud_instead_of_snow,
                self.cloud_instead_of_ice,
            )
        )

    def zeroed(self) -> "LossPenalties":
        """All weights off. Used by the cloud/glacier split strategy, which
        trains two specialised models for which the cross-class penalties are
        meaningless."""
        return LossPenalties(
            other_when_not_other=0.0,
            ice_instead_of_snow=0.0,
            snow_instead_of_ice=0.0,
            other_instead_of_ice=0.0,
            other_instead_of_cloud=0.0,
            cloud_instead_of_other=0.0,
            cloud_instead_of_snow=0.0,
            cloud_instead_of_ice=0.0,
            cloud_dem_independence=0.0,
        )


@dataclass(frozen=True)
class SearchSpaceEntry:
    """One hyperparameter's search range for Optuna."""

    type: str  # "log_uniform" | "uniform" | "choice"
    bounds: Optional[tuple[float, float]] = None
    values: Optional[tuple[Any, ...]] = None


@dataclass(frozen=True)
class TrainingConfig:
    """Everything one training run needs. Immutable; use `with_overrides`."""

    # --- data ---
    train_root: Path = Path(".")
    model_root: Path = Path(".")
    features: tuple[str, ...] = DEFAULT_FEATURES
    skip_landsat7_2003_2012: bool = True
    force_rebuild_cache: bool = False

    # --- model ---
    model_type: str = "unet"
    base_channels: int = 48
    dropout_p: float = 0.08507918967907807
    strategy: str = "mono"  # "mono" | "per_sensor"
    mono_common_mode: bool = True

    # --- optional architecture variants ---
    # All three defaults below are the reference: they reproduce the published
    # checkpoint's parameter set exactly, and the real-checkpoint compatibility
    # test asserts that. Each is a post-publication experiment, off unless
    # deliberately switched on.
    #
    # "batch" (reference) | "group". GroupNorm keeps no running statistics, so
    # it is unsupported with per-sensor normalisation banks (num_sensors > 1).
    norm_type: str = "batch"
    # Multi-head self-attention over the bottleneck grid (6x6 at patch_size 48).
    bottleneck_attention: bool = False
    bottleneck_attention_heads: int = 8
    # Auxiliary logit heads on the two intermediate decoder levels, each scored
    # against the labels reduced to its own resolution.
    deep_supervision: bool = False
    # Additive attention gate on every skip connection (Oktay et al. 2018).
    # True is the reference: the published checkpoint has these gates. False
    # is the ablation reported in the paper -- skip connections carry the raw
    # encoder feature map straight into the decoder, and the gates' parameters
    # are absent rather than merely bypassed.
    use_attention_gates: bool = True

    # --- optimisation ---
    lr: float = 0.00042607242544092326
    weight_decay: float = 5.3010972865943284e-05
    batch_size: int = 96
    epochs: int = 500
    patience: int = 15
    scheduler_min_lr: float = 1e-05

    # --- tiling ---
    patch_size: int = 48
    stride: int = 16
    min_valid: int = 5
    rotations: bool = False
    ignore_boundary: int = 1

    # --- split ---
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    split_glacier: bool = True
    split_by_sensor: bool = True
    glacier_size_stratify: bool = True
    class_density_balance: bool = False
    class_density_min_expected_ratio: float = 0.6
    class_density_max_moves: int = 20
    split_seed: int = 42

    # --- sensor adaptation (all forced off by mono_common_mode) ---
    per_sensor_normalization: bool = True
    sensor_balanced_sampling: bool = True
    use_sensor_film: bool = True
    glacier_balanced_sampling: bool = False
    glacier_coverage_sampling: bool = False
    sampling_seed: Optional[int] = None

    # --- context conditioning ---
    use_spatial_features: bool = False
    use_temporal_features: bool = False
    use_area_feature: bool = False

    # --- loss ---
    penalties: LossPenalties = field(default_factory=LossPenalties)

    # --- metrics ---
    main_iou_metric: str = "miou_macro"
    compute_boundary_metrics: bool = False

    # --- post-run diagnostics ---
    # These change what a finished run writes out, never how it trains, so they
    # are safe to alter without invalidating a comparison against the published
    # model. Both are computed in the single final evaluation pass.
    #
    # Rows kept per split in worst_scenes.csv / worst_glaciers.csv. 0 skips both.
    worst_case_count: int = 20
    # Equal-width softmax-confidence bins in calibration_<split>.csv.
    calibration_bins: int = 10

    # --- runtime ---
    device: Optional[str] = "cuda"
    num_workers: int = 8
    seed: int = 0
    use_amp: bool = True
    torch_compile: bool = False
    deterministic: bool = False
    # Reproduce the original best-epoch selection, in which the "best" snapshot
    # aliased the live parameters and was therefore overwritten by later
    # epochs. Only useful for bit-comparing against an existing checkpoint.
    keep_last_epoch_weights: bool = False
    max_scenes: Optional[int] = None  # cap the corpus, for smoke runs

    # --- HPO ---
    trials: int = 0
    trial_epochs: int = 25
    optuna_sampler: str = "tpe"
    optuna_pruner: str = "none"
    optuna_storage: Optional[str] = None
    search_space: Mapping[str, SearchSpaceEntry] = field(default_factory=dict)

    # --- cross-validation ---
    cv_enabled: bool = False
    cv_folds: int = 2
    cv_by_glacier: bool = True
    cv_seed: int = 42
    cv_fold_index: Optional[int] = None

    # --- bootstrap ---
    bootstrap_enabled: bool = True
    bootstrap_n_seeds: int = 1
    bootstrap_seed_base: int = 0
    bootstrap_create_ensemble: bool = True

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if not 0 < self.stride <= self.patch_size:
            raise ValueError(
                f"stride must be in (0, patch_size]; got stride={self.stride} "
                f"patch_size={self.patch_size}"
            )
        if self.min_valid < 0:
            raise ValueError(f"min_valid must be >= 0, got {self.min_valid}")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if not 0.0 <= self.test_ratio < 1.0:
            raise ValueError(f"test_ratio must be in [0, 1), got {self.test_ratio}")
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError(
                "val_ratio + test_ratio must leave room for a training split; got "
                f"{self.val_ratio} + {self.test_ratio}"
            )
        if not self.features:
            raise ValueError("features must not be empty")
        if self.model_type != "unet":
            raise ValueError(f"unknown model_type {self.model_type!r} (only 'unet')")
        if self.norm_type.strip().lower() not in ("batch", "group"):
            raise ValueError(
                f"unknown norm_type {self.norm_type!r} (expected 'batch' or 'group')"
            )
        if self.bottleneck_attention_heads <= 0:
            raise ValueError(
                "bottleneck_attention_heads must be positive, got "
                f"{self.bottleneck_attention_heads}"
            )
        if self.calibration_bins <= 0:
            raise ValueError(
                f"calibration_bins must be positive, got {self.calibration_bins}"
            )
        if self.worst_case_count < 0:
            raise ValueError(
                f"worst_case_count must be >= 0, got {self.worst_case_count}"
            )

    # -- derived -----------------------------------------------------------

    @property
    def mono_common_mode_active(self) -> bool:
        """True when the single-shared-model mode is in force.

        This mode disables per-sensor normalisation, sensor-balanced batching,
        sensor FiLM conditioning, and per-sensor BatchNorm all at once. It is
        what the reference configuration runs, so the shipped model uses none
        of the per-sensor machinery despite it all being implemented.
        """
        return self.mono_common_mode and self.strategy.strip().lower() == "mono"

    @property
    def effective_num_sensors(self) -> int:
        """Size of the per-sensor normalisation/FiLM banks. 1 collapses them to
        a single shared layer, which is what the reference model has.

        Raises when GroupNorm is combined with a real bank: GroupNorm holds no
        running statistics, so per-sensor banks of it would be a per-sensor
        rescaling and not per-sensor normalisation. Raised here rather than
        deep in the model so the run fails at configuration time, before
        anything is trained.
        """
        n_sensors = 1 if self.mono_common_mode_active else NUM_SENSORS
        if n_sensors > 1 and self.norm_type.strip().lower() == "group":
            raise ValueError(
                "norm_type='group' requires the single shared normalisation "
                "layer: GroupNorm keeps no running statistics for a per-sensor "
                "bank to accumulate. Set mono_common_mode=true (the reference "
                "setting) or use norm_type='batch'."
            )
        return n_sensors

    @property
    def sensor_adaptation(self) -> tuple[bool, bool, bool]:
        """(per_sensor_normalization, sensor_balanced_sampling, use_sensor_film),
        after applying the mono-common-mode override."""
        if self.mono_common_mode_active:
            return (False, False, False)
        return (
            self.per_sensor_normalization,
            self.sensor_balanced_sampling,
            self.use_sensor_film,
        )

    @property
    def use_spatial_context(self) -> bool:
        """Whether to build the FiLM context-gating branch at all."""
        return bool(
            self.use_spatial_features
            or self.use_temporal_features
            or self.use_area_feature
        )

    @property
    def in_channels(self) -> int:
        return len(self.features)

    def with_overrides(self, **kwargs: Any) -> "TrainingConfig":
        """A copy with fields replaced. Validation re-runs."""
        return replace(self, **kwargs)


def _search_space_from(raw: Mapping[str, Any]) -> dict[str, SearchSpaceEntry]:
    out: dict[str, SearchSpaceEntry] = {}
    for name, spec in (raw or {}).items():
        if isinstance(spec, SearchSpaceEntry):
            out[str(name)] = spec
            continue
        if hasattr(spec, "model_dump"):  # a pydantic model from the central config
            spec = spec.model_dump()
        if isinstance(spec, Mapping):
            bounds = spec.get("bounds")
            values = spec.get("values")
            out[str(name)] = SearchSpaceEntry(
                type=str(spec.get("type", "uniform")),
                bounds=(float(bounds[0]), float(bounds[1])) if bounds else None,
                values=tuple(values) if values else None,
            )
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            out[str(name)] = SearchSpaceEntry(type="choice", values=tuple(spec))
    return out


def training_config_from_pipeline(pipeline_cfg: Any) -> TrainingConfig:
    """Project the repo-wide `PipelineConfig` onto a `TrainingConfig`.

    Only the sections that affect training are read. Field-by-field rather
    than by reflection, so that adding a field to either side is a visible
    change here rather than a silent no-op.
    """
    paths = pipeline_cfg.paths
    model = pipeline_cfg.model
    train = pipeline_cfg.train
    split = pipeline_cfg.split
    metrics = pipeline_cfg.metrics_loss
    context = pipeline_cfg.context
    hpo = pipeline_cfg.hpo
    cv = pipeline_cfg.cross_validation
    boot = pipeline_cfg.bootstrap

    return TrainingConfig(
        train_root=Path(paths.train_root),
        model_root=Path(paths.model_train_root),
        features=tuple(pipeline_cfg.features.spectral_indices) or DEFAULT_FEATURES,
        skip_landsat7_2003_2012=pipeline_cfg.scene_download.skip_landsat7_2003_2012,
        force_rebuild_cache=split.force_rebuild_npz,
        model_type=model.type,
        base_channels=model.base_channels,
        dropout_p=train.dropout_p,
        strategy=model.strategy,
        mono_common_mode=model.mono_common_mode,
        norm_type=model.norm_type,
        bottleneck_attention=model.bottleneck_attention,
        bottleneck_attention_heads=model.bottleneck_attention_heads,
        deep_supervision=model.deep_supervision,
        use_attention_gates=model.use_attention_gates,
        lr=train.lr,
        weight_decay=train.weight_decay,
        batch_size=train.batch_size,
        epochs=train.epochs,
        patience=train.patience,
        scheduler_min_lr=train.scheduler_min_lr,
        patch_size=train.patch_size,
        stride=train.stride,
        min_valid=train.min_valid,
        rotations=train.rotations,
        ignore_boundary=metrics.ignore_boundary,
        val_ratio=split.val_ratio,
        test_ratio=split.test_ratio,
        split_glacier=split.split_glacier,
        split_by_sensor=split.split_by_sensor,
        glacier_size_stratify=split.glacier_size_stratify,
        class_density_balance=split.class_density_balance,
        class_density_min_expected_ratio=split.class_density_min_expected_ratio,
        class_density_max_moves=split.class_density_max_moves,
        use_spatial_features=context.use_spatial_features,
        use_temporal_features=context.use_temporal_features,
        use_area_feature=context.use_area_feature,
        penalties=LossPenalties(
            other_when_not_other=metrics.penalty_other_when_not_other,
            ice_instead_of_snow=metrics.penalty_ice_instead_of_snow,
            snow_instead_of_ice=metrics.penalty_snow_instead_of_ice,
            other_instead_of_ice=metrics.penalty_other_instead_of_ice,
            other_instead_of_cloud=metrics.penalty_other_instead_of_cloud,
            cloud_instead_of_other=metrics.penalty_cloud_instead_of_other,
            cloud_instead_of_snow=metrics.penalty_cloud_instead_of_snow,
            cloud_instead_of_ice=metrics.penalty_cloud_instead_of_ice,
            cloud_dem_independence=metrics.cloud_dem_independence_lambda,
        ),
        main_iou_metric=metrics.main_iou_metric,
        compute_boundary_metrics=metrics.compute_boundary_metrics,
        worst_case_count=metrics.worst_case_count,
        calibration_bins=metrics.calibration_bins,
        device=model.device,
        num_workers=train.num_workers,
        trials=hpo.trials,
        trial_epochs=hpo.trial_epochs,
        optuna_sampler=hpo.sampler,
        optuna_pruner=hpo.pruner,
        optuna_storage=hpo.storage,
        search_space=_search_space_from(hpo.search_space),
        cv_enabled=cv.enabled,
        cv_folds=cv.folds,
        cv_by_glacier=cv.by_glacier,
        cv_seed=cv.seed,
        bootstrap_enabled=boot.enabled,
        bootstrap_n_seeds=boot.n_seeds,
        bootstrap_seed_base=boot.seed_base,
        bootstrap_create_ensemble=boot.create_ensemble,
    )
