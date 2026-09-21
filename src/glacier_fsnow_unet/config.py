"""Configuration loading and validation for glacier-fsnow-unet.

Decision (Phase 0, ROADMAP key decision #5): use pydantic (v2 BaseModel) rather
than omegaconf or a hand-written PyYAML validation pass. Rationale: the config
has ~15 nested sections with mixed types (paths, floats, ints, bools, nested
lists of dicts for scene-download splits and HPO search spaces). Pydantic gives
per-field type coercion and a single ValidationError listing every offending
field (path + expected type + given value) with no extra boilerplate beyond the
model definitions themselves. Omegaconf's variable interpolation is not needed
here because ${VAR} substitution is handled explicitly against the environment/
.env before YAML parsing (see `_substitute_env_vars`), so a plain "the fields
raise cleanly" story from pydantic is simpler than adopting an extra dependency
whose main selling point (interpolation, merging) we do not otherwise need.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

CONFIG_ENV_VAR = "GLACIER_FSNOW_CONFIG"


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or invalid."""


class PathsConfig(BaseModel):
    data_root: str
    output_root: str
    model_root: str
    train_root: str
    apply_root: str
    model_train_root: str
    model_inference_root: str
    published_root: str
    zenodo_train_data_root: Optional[str] = None


class IsolatedGlacierConfig(BaseModel):
    rgi_shapefiles: list[str]
    output_json: str
    output_named_json: str
    output_root: str
    min_area_km2: float = 0.05


class DemConfig(BaseModel):
    input_root: str
    glacier_dem_root: str
    glacier_input_root: str
    arctic_dem_stac_url: str
    copernicus_fallback: bool = True
    max_workers: int = 8


class SceneSplit(BaseModel):
    root: str
    ee_project: str


class DriveExportConfig(BaseModel):
    enabled: bool = False
    folder: str = "GEE_Glacier_Exports"
    max_tasks: int = 2000


class SceneDownloadConfig(BaseModel):
    splits: list[SceneSplit]
    ee_project_default: str
    max_cloud_fraction: float = 0.30
    min_aoi_coverage: float = 0.70
    download_workers: int = 4
    glacier_workers: int = 1
    max_scenes_per_year: int = 9999
    skip_landsat7_2003_2012: bool = True
    drive_export: DriveExportConfig = Field(default_factory=DriveExportConfig)


class FeaturesConfig(BaseModel):
    sources: list[str]
    spectral_indices: list[str]
    max_workers: int = 1


class SplitConfig(BaseModel):
    split_glacier: bool = True
    split_by_sensor: bool = True
    glacier_size_stratify: bool = True
    class_density_balance: bool = False
    class_density_min_expected_ratio: float = 0.6
    class_density_max_moves: int = 20
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    force_rebuild_npz: bool = True


class ModelConfig(BaseModel):
    strategy: str = "mono"
    mono_common_mode: bool = True
    type: str = "unet"
    device: str = "cuda"
    base_channels: int = 48
    # Optional architecture variants. Every default here is the reference: it
    # is what the published checkpoint's weights load into.
    norm_type: str = "batch"
    bottleneck_attention: bool = False
    bottleneck_attention_heads: int = 8
    deep_supervision: bool = False
    use_attention_gates: bool = True


class TrainConfig(BaseModel):
    lr: float
    patch_size: int = 48
    stride: int = 16
    batch_size: int = 96
    epochs: int = 500
    patience: int = 15
    min_valid: int = 5
    num_workers: int = 8
    weight_decay: float
    dropout_p: float
    scheduler_min_lr: float = 0.00001
    rotations: bool = False


class CrossValidationConfig(BaseModel):
    enabled: bool = False
    folds: int = 2
    by_glacier: bool = True
    seed: int = 42


class BootstrapConfig(BaseModel):
    enabled: bool = True
    n_seeds: int = 5
    seed_base: int = 0
    create_ensemble: bool = True


class SearchSpaceEntry(BaseModel):
    type: str
    bounds: list[float]


class HpoConfig(BaseModel):
    trials: int = 0
    trial_epochs: int = 25
    sampler: str = "tpe"
    pruner: str = "none"
    storage: Optional[str] = None
    search_space: dict[str, SearchSpaceEntry] = Field(default_factory=dict)


class MetricsLossConfig(BaseModel):
    main_iou_metric: str = "miou_macro"
    compute_boundary_metrics: bool = False
    ignore_boundary: int = 1
    penalty_other_when_not_other: float = 0
    penalty_ice_instead_of_snow: float = 0
    penalty_snow_instead_of_ice: float = 0
    penalty_other_instead_of_ice: float = 10
    penalty_other_instead_of_cloud: float = 0
    penalty_cloud_instead_of_other: float = 0
    penalty_cloud_instead_of_snow: float = 0
    penalty_cloud_instead_of_ice: float = 0
    cloud_dem_independence_lambda: float = 0
    # Post-run diagnostics. Affect what a run exports, never how it trains.
    worst_case_count: int = 20
    calibration_bins: int = 10


class ContextConfig(BaseModel):
    use_spatial_features: bool = False
    use_temporal_features: bool = False
    use_area_feature: bool = False


class InferenceConfig(BaseModel):
    patch_size: int = 48
    stride: int = 16
    batch_size: int = 384
    cpu_threads: int = 12
    max_workers: int = 8
    max_workers_apply: int = 8
    torch_compile: bool = True
    torch_compile_mode: str = "max-autotune"
    overwrite: bool = False
    scene_precheck_min_valid_ratio: float = 0.30


class BoundaryPostprocessConfig(BaseModel):
    max_chain_length: int = 0
    search_radius: int = 4
    min_chain_length: int = 0
    propagate: bool = False
    propagate_max_iterations: int = 20


class MapSgvStatsConfig(BaseModel):
    map_etat_overwrite: bool = True
    sgv_overwrite: bool = True
    map_etat_max_scene_cloud_frac: float = 1.0
    map_etat_min_scene_threshold_from_2013: int = 2
    map_etat_min_scene_threshold_before_2013: int = 2
    map_etat_max_workers: int = 20
    sgv_rgi_shapefiles: list[str]
    sgv_ref_min_glacier_coverage_pct: float = 70.0
    sgv_ref_max_overshoot_pct: float = 50.0
    sgv_ref_connectivity_tolerance: int = 0
    sgv_ref_max_cloud_pct: float = 10.0
    emprise_generate_etat: bool = True
    emprise_generate_3d_html: bool = False
    emprise_generate_3d_package: bool = True
    emprise_generate_3d_rgb: bool = True
    stats_process_etat: bool = True
    stats_process_physic: bool = False


class FeatureImportanceConfig(BaseModel):
    enabled: bool = True
    metric: str = "main"
    split: str = "valid"
    max_batches: int = 0
    context: bool = True
    num_workers: int = 0


class AmbiguityInterpretationConfig(BaseModel):
    """Calibration of the confidence-gap threshold used to arbitrate near-ties.

    ``tolerance`` is the macro-mIoU degradation budget: the sweep keeps the
    largest threshold whose priority-override rule costs no more than this.
    """

    enabled: bool = True
    split: str = "valid"
    #: Maximum acceptable macro-mIoU loss, as a fraction (0.01 = one point).
    tolerance: float = 0.01
    #: Scenes to sample; 0 uses every scene of the partition.
    max_scenes: int = 0
    #: Highest gap threshold considered. Gaps above this are never overridden.
    threshold_sweep_max: float = 0.50
    threshold_sweep_steps: int = 201
    batch_size: int = 8


class FeatureImportanceInterpretationConfig(BaseModel):
    """Permutation importance, run against a finished checkpoint.

    Deliberately separate from the top-level ``feature_importance`` section:
    that one parameterises the attribution pass a *training run* can be asked
    to append to its own exports, and its cost budget is set relative to a
    training run. This one parameterises an analyst re-running the same
    measurement standalone, where a different partition or batch cap is the
    normal thing to want. Sharing one block would mean changing a training
    export's contents as a side effect of an interpretation run.
    """

    enabled: bool = True
    split: str = "valid"
    #: Cap batches evaluated per channel; 0 uses the whole partition.
    max_batches: int = 0
    seed: int = 0


class ShapInterpretationConfig(BaseModel):
    enabled: bool = True
    method: str = "shap_gradient"
    split: str = "valid"
    #: Baseline path draws per explained tile.
    background_samples: int = 128
    #: Tiles explained.
    test_samples: int = 512
    batch_size: int = 8
    class_conditioned: bool = True
    min_class_pixels: int = 256
    seed: int = 0


class GradCamInterpretationConfig(BaseModel):
    enabled: bool = True
    split: str = "valid"
    samples: int = 48
    batch_size: int = 8
    class_conditioned: bool = True
    min_class_pixels: int = 256
    #: Module whose activations and gradients the map is built from. The
    #: deepest encoder stage is the standard choice: coarsest spatial grid,
    #: richest semantics, and upstream of every decoder skip connection.
    target_layer: str = "enc4"
    #: Per-pixel maps are written into this zarr store under the output
    #: directory, never as loose .npz or image files.
    zarr_dirname: str = "gradcam.zarr"


class UnetInterpretationConfig(BaseModel):
    """The four interpretation analyses, each independently switchable.

    Every analysis carries its own ``enabled`` flag and its own settings, so
    turning one off cannot silently change another's parameters, and a config
    reader can see at a glance which analyses a given run performed.
    """

    output_dirname: str = "interpretation"
    ambiguity: AmbiguityInterpretationConfig = Field(
        default_factory=AmbiguityInterpretationConfig
    )
    feature_importance: FeatureImportanceInterpretationConfig = Field(
        default_factory=FeatureImportanceInterpretationConfig
    )
    shap: ShapInterpretationConfig = Field(default_factory=ShapInterpretationConfig)
    grad_cam: GradCamInterpretationConfig = Field(
        default_factory=GradCamInterpretationConfig
    )


class DebugConfig(BaseModel):
    feature_diagnostic: bool = True
    mode_execution: str = "prod"


# ----------------------------------------------------------------------
# Non-U-Net baseline comparisons (scripts/baselines/14_..18_).
#
# These are a separate, optional comparison suite (NDSI threshold, Random
# Forest, DeepLabv3, SegFormer) run against the exact same glacier-level
# split/features/metrics as the published U-Net, never a variant of the
# U-Net training pipeline itself. They get their own top-level section
# rather than reusing `train`/`model`/`hpo` so that changing a baseline's
# hyperparameter can never be confused with changing the published model's.
# `scripts/05_train_model.py` does not read this section at all.
# ----------------------------------------------------------------------


class NdsiBaselineConfig(BaseModel):
    """scripts/baselines/14_ndsi_baseline.py -- 3-threshold NDSI/NDVI tree."""

    out_dir: str = "_work/baselines/ndsi"
    #: Grid-search step for the coarse sweep, then the fine sweep around it.
    coarse_step: float = 0.05
    fine_step: float = 0.01


class RfBaselineConfig(BaseModel):
    """scripts/baselines/15_rf_baseline.py -- Random Forest on the same 11
    spectral indices. `mode: grid` reproduces the original 24-combination
    manual grid search (the published test mIoU = 0.570); `mode: optuna` is
    the recommended method going forward (TPE search over a wider version
    of the same hyperparameter families).
    """

    out_dir: str = "_work/baselines/rf"
    mode: str = "optuna"  # "grid" | "optuna"
    #: Stratified-by-class pixel subsample size used for the HP search
    #: (the final model is always refit on the full train partition).
    subsample_size: int = 400_000
    subsample_seed: int = 0
    primary_seed: int = 0
    #: Full-data refits for the seed-sensitivity check (in addition to the
    #: primary seed's refit already produced by the HP-search stage).
    n_seeds: int = 5
    seed_start: int = 0
    #: Capped, not -1: a full ~2.24M-row refit at n_jobs=-1 previously
    #: crashed with a MemoryError partway through a seed loop.
    full_data_n_jobs: int = 8
    #: Safe at -1: HP-search fits are on the (much smaller) subsample only.
    search_n_jobs: int = -1
    n_trials: int = 20
    optuna_seed: int = 0
    #: null uses "<out_dir>/optuna_rf.db"; set for a shared/persistent study.
    optuna_storage: Optional[str] = None


class TorchBaselineConfig(BaseModel):
    """Shared schema for the DeepLabv3 and SegFormer baselines
    (scripts/baselines/16_deeplab_baseline.py, 17_segformer_baseline.py).

    Each script dispatches on its own subcommand (`hpo` | `confirm`), so one
    config section covers both the short-epoch Optuna search and the
    full-protocol multi-seed confirmation of its winner.
    """

    out_dir: str = "_work/baselines/deeplab"
    batch_size: int = 64
    num_workers: int = 4
    #: Only used directly by `hpo` when the search space omits dropout_p;
    #: `confirm` always takes dropout_p from --dropout-p or this field.
    dropout_p: float = 0.1

    # --- hpo subcommand: short-epoch Optuna search over lr/weight_decay/dropout_p ---
    n_trials: int = 20
    trial_epochs: int = 15
    trial_patience: int = 5
    trial_seed: int = 0
    #: null uses "<out_dir>/optuna_<name>.db".
    optuna_storage: Optional[str] = None

    # --- confirm subcommand: full-protocol multi-seed confirmation ---
    n_seeds: int = 5
    seed_start: int = 0
    final_epochs: int = 500
    final_patience: int = 15
    #: The HPO winner's hyperparameters. Set these (via this file or
    #: --lr/--weight-decay/--dropout-p) once hpo has produced a winner.
    lr: Optional[float] = None
    weight_decay: Optional[float] = None


class BaselinesConfig(BaseModel):
    """Root section for the non-U-Net baseline comparison scripts."""

    #: Corpus root, read-only, same corpus the U-Net trains on. Defaults to
    #: `paths.train_root` (see `training_config_from_pipeline`-adjacent
    #: loading in scripts/baselines/_common.py) when left null.
    train_root: Optional[str] = None
    #: A GlacierSplit_WithBoundaries checkpoint used only to cross-check the
    #: reproduced split scene-by-scene before any baseline number is trusted.
    check_checkpoint: Optional[str] = (
        "_work/rse_sigtest/checkpoints/glacier_split/seed_0_model.pt"
    )
    ndsi: NdsiBaselineConfig = Field(default_factory=NdsiBaselineConfig)
    rf: RfBaselineConfig = Field(default_factory=RfBaselineConfig)
    deeplab: TorchBaselineConfig = Field(
        default_factory=lambda: TorchBaselineConfig(out_dir="_work/baselines/deeplab")
    )
    segformer: TorchBaselineConfig = Field(
        default_factory=lambda: TorchBaselineConfig(out_dir="_work/baselines/segformer")
    )


class PipelineConfig(BaseModel):
    """Root configuration model for the full glacier-fsnow-unet pipeline."""

    paths: PathsConfig
    isolated_glacier: IsolatedGlacierConfig
    dem: DemConfig
    scene_download: SceneDownloadConfig
    features: FeaturesConfig
    split: SplitConfig
    model: ModelConfig
    train: TrainConfig
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    hpo: HpoConfig = Field(default_factory=HpoConfig)
    metrics_loss: MetricsLossConfig = Field(default_factory=MetricsLossConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    boundary_postprocess: BoundaryPostprocessConfig = Field(default_factory=BoundaryPostprocessConfig)
    map_sgv_stats: MapSgvStatsConfig
    feature_importance: FeatureImportanceConfig = Field(default_factory=FeatureImportanceConfig)
    unet_interpretation: UnetInterpretationConfig = Field(default_factory=UnetInterpretationConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)


def _substitute_env_vars(raw_text: str) -> str:
    """Replace ${VAR} placeholders in raw_text with values from the environment.

    Raises ConfigError naming the missing variable if a referenced ${VAR} is not
    set in the environment (or in a .env file already loaded via load_dotenv).
    """

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ConfigError(
                f"Config references ${{{var_name}}} but no such environment "
                f"variable is set (check your .env file or environment)."
            )
        return value

    return _ENV_VAR_PATTERN.sub(_replace, raw_text)


def load_config(config_path: Optional[str] = None, env_file: Optional[str] = None) -> PipelineConfig:
    """Load, substitute, and validate the pipeline configuration.

    Resolution order for the config path: explicit `config_path` argument,
    then the GLACIER_FSNOW_CONFIG environment variable, then
    "configs/config.yaml" relative to the current working directory.

    A .env file (default: ".env" in the current working directory) is loaded
    first via python-dotenv so its values are available for ${VAR} substitution.
    """
    load_dotenv(dotenv_path=env_file)  # no-op if the file does not exist

    resolved_path = config_path or os.environ.get(CONFIG_ENV_VAR) or "configs/config.yaml"
    path = Path(resolved_path)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found at '{path}'. Pass --config, set "
            f"{CONFIG_ENV_VAR}, or create configs/config.yaml (see "
            f"configs/config.example.yaml)."
        )

    raw_text = path.read_text(encoding="utf-8")
    substituted_text = _substitute_env_vars(raw_text)

    try:
        raw_data = yaml.safe_load(substituted_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in '{path}': {exc}") from exc

    try:
        return PipelineConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in '{path}':\n{exc}") from exc
