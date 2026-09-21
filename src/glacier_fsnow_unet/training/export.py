"""Checkpoint writing and the full set of per-run evaluation exports.

A checkpoint has to carry everything inference needs to reconstruct the model
and preprocess inputs the same way training did. Weights alone are not enough:
without the normalisation statistics and the exact feature list *in order*, the
same weights applied to the same imagery give different answers. Both travel
inside the checkpoint rather than alongside it, so they cannot be separated
from the weights they belong to.

**Everything is CSV and JSON, never a spreadsheet.** A multi-sheet workbook is
one opaque binary blob: it does not diff, a one-cell change rewrites the whole
file in version control, and reading it needs an optional dependency that a
training environment otherwise has no use for. Where the natural shape is a
workbook with several sheets, this module writes one CSV per sheet instead —
strictly more inspectable, `grep`-able and diffable, at the cost of a few more
files in a directory that is read by tooling anyway. The one thing a workbook
buys that separate CSVs do not is a single-file download, which matters for
distribution but not for a run directory; packaging for distribution is a
separate step and belongs there, not here.

**What a run directory contains**, all of it written by `save_model`:

    model.pt                       weights + self-describing metadata
    model_config.json              every hyperparameter, feature and dimension
    run_summary.json               headline metrics, split sizes, timing
    epoch_metrics.csv              per-epoch loss and validation metrics
    metrics_<split>.csv            per-class rows plus a `global` macro row
    confusion_matrix_<split>.csv   the pooled confusion matrix, raw counts
    confusion_<split>_normalized.csv  the same, each row divided by its total
    confusion_boundary_<split>.csv the same, restricted to class-transition pixels
    calibration_<split>.csv        accuracy and mean confidence per confidence bin
    scene_performance.csv          one row per evaluated scene
    glacier_performance.csv        one row per (glacier, sensor, split)
    sensor_performance.csv         one row per (sensor, split)
    sensor_robustness.csv          per split, the spread across sensors
    class_iou_by_sensor.csv        per (split, sensor), the four class IoUs
    worst_scenes.csv               the lowest-mIoU scenes per split
    worst_glaciers.csv             the lowest-mIoU (glacier, sensor) pairs per split
    feature_importance.csv         per feature x class, permutation importance
    shap_importance.csv            per feature x class, gradient attribution

Files whose source data is absent are simply not written, rather than written
empty — an absent file is unambiguous, whereas a header-only CSV looks like a
run that produced nothing.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from .config import CLASS_NAMES, TrainingConfig
from .train import TrainingResult

__all__ = [
    "save_model",
    "write_metrics_summary",
    "write_history_csv",
    "write_confusion_csv",
    "write_normalized_confusion_csv",
    "write_calibration_csv",
    "write_rows_csv",
    "write_split_metrics_csv",
    "write_feature_importance_csv",
    "write_attribution_csv",
    "checkpoint_metadata",
    "model_config_document",
    "run_summary_document",
]


# -- generic writers ---------------------------------------------------------


def write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Write a list of dicts as a CSV, or nothing at all if there are no rows.

    Columns default to the union of every row's keys in first-seen order, so a
    row carrying an extra field widens the table rather than losing the field.
    """
    if not rows:
        return None

    if columns is None:
        ordered: list[str] = []
        for row in rows:
            ordered.extend(key for key in row if key not in ordered)
        columns = ordered

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in columns})
    return path


def _cell(value: Any) -> Any:
    """Render one value for CSV.

    NaN becomes an empty cell rather than the string "nan": a reader loading
    the column gets a missing value, which is what it is, instead of a token
    that silently turns the column into text.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return "" if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    return value


def write_history_csv(path: Path, history: Mapping[str, Sequence[float]]) -> Optional[Path]:
    """Write the per-epoch history as one row per epoch."""
    if not history:
        return None

    columns = sorted(history)
    n_epochs = max((len(history[c]) for c in columns), default=0)
    if n_epochs == 0:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *columns])
        for index in range(n_epochs):
            writer.writerow(
                [index + 1]
                + [
                    _cell(history[column][index]) if index < len(history[column]) else ""
                    for column in columns
                ]
            )
    return path


def write_confusion_csv(
    path: Path,
    confusion: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Optional[Path]:
    """Write a confusion matrix with labelled rows (true) and columns (predicted).

    A `row_total` column and a `predicted_total` row are appended so the
    marginals are readable directly rather than needing to be summed by the
    reader — they are what every per-class support and precision denominator is
    built from.
    """
    if confusion is None:
        return None

    matrix = np.asarray(confusion, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true \\ predicted", *class_names, "row_total"])
        for index, row in enumerate(matrix):
            label = class_names[index] if index < len(class_names) else str(index)
            writer.writerow([label, *(int(v) for v in row), int(row.sum())])
        writer.writerow(
            [
                "predicted_total",
                *(int(v) for v in matrix.sum(axis=0)),
                int(matrix.sum()),
            ]
        )
    return path


def write_normalized_confusion_csv(
    path: Path,
    confusion: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Optional[Path]:
    """Write the same matrix with each row divided by its own total.

    Cell `(i, j)` reads as `P(predicted = j | true = i)`: the fraction of class
    `i`'s pixels the model assigned to class `j`. The diagonal is per-class
    recall, and the off-diagonal cells are directly the quantity "how often is
    Ice called Snow".

    This is written alongside the raw-count matrix rather than instead of it,
    because they answer different questions and neither substitutes for the
    other. Raw counts carry the support — a 40% error rate on eight hundred
    pixels and on eight million are not the same finding — but comparing two
    error rates across classes whose totals differ by an order of magnitude
    means dividing in your head every time. The normalised view does that
    division once, and `row_total` is kept as a column so the support is never
    more than one glance away.

    A class with no true pixels gets an empty row rather than a division by
    zero: it was not measured, which is not the same as being predicted
    correctly 0% of the time.
    """
    if confusion is None:
        return None

    matrix = np.asarray(confusion, dtype=np.float64)
    totals = matrix.sum(axis=1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true \\ predicted", *class_names, "row_total"])
        for index, row in enumerate(matrix):
            label = class_names[index] if index < len(class_names) else str(index)
            total = float(totals[index])
            if total <= 0.0:
                writer.writerow([label, *([""] * len(row)), 0])
                continue
            writer.writerow(
                [label, *(round(float(v) / total, 6) for v in row), int(total)]
            )
    return path


def write_split_metrics_csv(
    path: Path,
    metrics: Mapping[str, Any],
    class_names: Sequence[str] = CLASS_NAMES,
) -> Optional[Path]:
    """Write one partition's metrics: a row per class, then a `global` row.

    The two are in one file rather than two because they are read together —
    a per-class IoU is only interpretable next to the macro it contributes to.
    The macro row carries `class_name = "global"` and an empty `class`, so the
    file can be filtered either way without a second lookup.
    """
    if not metrics:
        return None

    per_class = metrics.get("per_class") or []
    macro = dict(metrics.get("macro") or {})
    if not per_class and not macro:
        return None

    columns = [
        "class",
        "class_name",
        "support",
        "precision",
        "recall",
        "f1",
        "f_beta",
        "iou",
        "specificity",
    ]
    macro_columns = [
        "miou_macro",
        "miou_weighted",
        "miou_invfq_weighted",
        "kappa",
        "mcc",
        "glacier_iou_binary",
        "n_pixels",
    ]

    rows: list[dict[str, Any]] = [dict(entry) for entry in per_class]
    rows.append(
        {
            "class": "",
            "class_name": "global",
            "support": macro.get("total", ""),
            "precision": "",
            "recall": "",
            "f1": "",
            "f_beta": "",
            "iou": macro.get("miou", ""),
            "specificity": macro.get("specificity", ""),
            "miou_macro": macro.get("miou", ""),
            "miou_weighted": macro.get("miou_weighted", ""),
            "miou_invfq_weighted": macro.get("miou_inv_freq", ""),
            "kappa": macro.get("kappa", ""),
            "mcc": macro.get("mcc", ""),
            "glacier_iou_binary": metrics.get("glacier_iou_binary", ""),
            "n_pixels": macro.get("total", ""),
        }
    )
    return write_rows_csv(path, rows, columns=[*columns, *macro_columns])


def write_calibration_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Optional[Path]:
    """Write one partition's reliability table.

    One row per confidence bin plus a final `overall` row, from
    `metrics.CalibrationTally.rows()`. Columns are fixed rather than derived
    from the rows, so the per-bin rows and the wider `overall` row line up in
    one table instead of the summary-only columns being dropped or reordered.
    """
    return write_rows_csv(
        path,
        rows,
        columns=[
            "bin",
            "confidence_lower",
            "confidence_upper",
            "n_pixels",
            "pixel_share_pct",
            "n_correct",
            "accuracy",
            "mean_confidence",
            "gap_confidence_minus_accuracy",
            "expected_calibration_error",
            "mean_confidence_when_correct",
            "mean_confidence_when_incorrect",
            "confidence_separation",
        ],
    )


def write_feature_importance_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Optional[Path]:
    """Write the per-feature, per-class permutation importance table."""
    return write_rows_csv(
        path,
        rows,
        columns=[
            "feature",
            "class",
            "class_name",
            "importance_delta_iou",
            "baseline_iou",
            "permuted_iou",
        ],
    )


def write_attribution_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Optional[Path]:
    """Write the per-feature, per-class gradient attribution table."""
    return write_rows_csv(
        path,
        rows,
        columns=[
            "feature",
            "class",
            "class_name",
            "attribution",
            "abs_attribution",
            "share_pct",
            "feature_abs_total",
            "method",
        ],
    )


def write_metrics_summary(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Optional[Path]:
    """Write a list of per-run metric dicts as a CSV.

    Columns are the union of every row's keys, so a run that failed and
    recorded only an error still appears rather than being dropped.
    """
    return write_rows_csv(path, rows)


# -- run description ---------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert a config value into something `json.dumps` accepts."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def model_config_document(
    result: TrainingResult,
    config: TrainingConfig,
) -> dict[str, Any]:
    """The full run configuration, as a nested document.

    Grouped by what each field governs — data, architecture, optimisation,
    tiling, split, loss, metrics, runtime — rather than flattened, because a
    reader reproducing a run needs to know which knobs belong together. Every
    field of `TrainingConfig` appears somewhere in here: the `_remaining` block
    catches anything not explicitly placed above, so adding a config field can
    never silently drop it from the exported record.
    """
    placed: set[str] = set()

    def take(*names: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in names:
            if hasattr(config, name):
                out[name] = _jsonable(getattr(config, name))
                placed.add(name)
        return out

    document: dict[str, Any] = {
        "schema": "glacier-fsnow-unet/model_config/1",
        "created_at": datetime.now().isoformat(),
        "data": {
            **take(
                "train_root",
                "model_root",
                "skip_landsat7_2003_2012",
                "force_rebuild_cache",
                "max_scenes",
            ),
            "features": list(result.features),
            "n_features": len(result.features),
            "class_names": list(CLASS_NAMES),
            "num_classes": len(CLASS_NAMES),
        },
        "architecture": {
            **take(
                "model_type",
                "base_channels",
                "dropout_p",
                "strategy",
                "mono_common_mode",
                "norm_type",
                "bottleneck_attention",
                "bottleneck_attention_heads",
                "deep_supervision",
                "use_attention_gates",
            ),
            "in_channels": config.in_channels,
            "num_classes": len(CLASS_NAMES),
            "num_sensors": config.effective_num_sensors,
            "use_spatial_context": config.use_spatial_context,
            "use_sensor_film": config.sensor_adaptation[2],
        },
        "optimisation": take(
            "lr", "weight_decay", "batch_size", "epochs", "patience", "scheduler_min_lr"
        ),
        "tiling": take("patch_size", "stride", "min_valid", "rotations", "ignore_boundary"),
        "split": take(
            "val_ratio",
            "test_ratio",
            "split_glacier",
            "split_by_sensor",
            "glacier_size_stratify",
            "class_density_balance",
            "class_density_min_expected_ratio",
            "class_density_max_moves",
            "split_seed",
        ),
        "sensor_adaptation": {
            **take(
                "per_sensor_normalization",
                "sensor_balanced_sampling",
                "use_sensor_film",
                "glacier_balanced_sampling",
                "glacier_coverage_sampling",
                "sampling_seed",
            ),
            # What the mono-common-mode override actually left active, which is
            # not the same as what was requested above.
            "effective": {
                "per_sensor_normalization": config.sensor_adaptation[0],
                "sensor_balanced_sampling": config.sensor_adaptation[1],
                "use_sensor_film": config.sensor_adaptation[2],
            },
        },
        "context": take(
            "use_spatial_features", "use_temporal_features", "use_area_feature"
        ),
        "loss": {
            "penalties": _jsonable(config.penalties),
            "class_counts": {
                CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else str(k): int(v)
                for k, v in result.class_counts.items()
            },
        },
        "metrics": take("main_iou_metric", "compute_boundary_metrics"),
        "runtime": take(
            "device",
            "num_workers",
            "seed",
            "use_amp",
            "torch_compile",
            "deterministic",
            "keep_last_epoch_weights",
        ),
        "hpo": take(
            "trials",
            "trial_epochs",
            "optuna_sampler",
            "optuna_pruner",
            "optuna_storage",
            "search_space",
        ),
        "cross_validation": take(
            "cv_enabled", "cv_folds", "cv_by_glacier", "cv_seed", "cv_fold_index"
        ),
        "bootstrap": take(
            "bootstrap_enabled",
            "bootstrap_n_seeds",
            "bootstrap_seed_base",
            "bootstrap_create_ensemble",
        ),
        "normalisation": {
            "mean": np.asarray(result.mean, dtype=np.float32).tolist(),
            "std": np.asarray(result.std, dtype=np.float32).tolist(),
        },
    }
    # `penalties` is exported inside `loss`; mark it placed so it does not also
    # appear under `_remaining`.
    placed.add("penalties")

    remaining = {
        name: _jsonable(getattr(config, name))
        for name in sorted(vars(config))
        if name not in placed
    }
    if remaining:
        document["_remaining"] = remaining
    return document


def run_summary_document(
    result: TrainingResult,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Headline numbers for one run: what a reader wants before opening a CSV."""
    best = {
        key: _jsonable(value)
        for key, value in result.best_state.items()
        if key not in ("model", "confusion")
    }

    per_split: dict[str, Any] = {}
    for split, metrics in result.metrics.items():
        macro = metrics.get("macro") if isinstance(metrics, Mapping) else None
        if not isinstance(macro, Mapping):
            continue
        per_split[split] = {
            **{k: _jsonable(v) for k, v in macro.items()},
            "glacier_iou_binary": _jsonable(
                metrics.get("glacier_iou_binary")  # type: ignore[union-attr]
            ),
        }

    return {
        "schema": "glacier-fsnow-unet/run_summary/1",
        "created_at": datetime.now().isoformat(),
        "seed": config.seed,
        "split_seed": config.split_seed,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "epochs_configured": config.epochs,
        "patience": config.patience,
        "main_iou_metric": config.main_iou_metric,
        "best_state": best,
        "splits": _jsonable(result.split_counts),
        "metrics": per_split,
        "features": list(result.features),
        "class_names": list(CLASS_NAMES),
    }


def checkpoint_metadata(
    result: TrainingResult,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Assemble the metadata block stored beside the weights.

    Everything here is needed either to rebuild the architecture (channel
    counts, base width, sensor-bank size) or to reproduce preprocessing
    (feature order, normalisation statistics). Metrics and history ride along
    so a checkpoint is self-describing.
    """
    return {
        "features": list(result.features),
        "mean": np.asarray(result.mean, dtype=np.float32).tolist(),
        "std": np.asarray(result.std, dtype=np.float32).tolist(),
        "class_names": list(CLASS_NAMES),
        "class_counts": dict(result.class_counts),
        "architecture": {
            "model_type": config.model_type,
            "in_channels": config.in_channels,
            "num_classes": len(CLASS_NAMES),
            "base_channels": config.base_channels,
            "dropout_p": config.dropout_p,
            "num_sensors": config.effective_num_sensors,
            "use_spatial_context": config.use_spatial_context,
            "use_sensor_film": config.sensor_adaptation[2],
            # Needed to rebuild the architecture these weights fit: a
            # GroupNorm or attention-carrying checkpoint will not load into a
            # model built from the defaults, and the failure would otherwise
            # surface as a key mismatch with nothing explaining it.
            "norm_type": config.norm_type,
            "bottleneck_attention": config.bottleneck_attention,
            "bottleneck_attention_heads": config.bottleneck_attention_heads,
            "deep_supervision": config.deep_supervision,
            "use_attention_gates": config.use_attention_gates,
        },
        "training": {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "patch_size": config.patch_size,
            "stride": config.stride,
            "epochs": config.epochs,
            "patience": config.patience,
            "seed": config.seed,
            "split_seed": config.split_seed,
            "min_valid": config.min_valid,
            "ignore_boundary": config.ignore_boundary,
            "main_iou_metric": config.main_iou_metric,
        },
        "best_state": {
            key: value
            for key, value in result.best_state.items()
            if key not in ("model", "confusion")
        },
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "metrics": result.metrics,
        "history": result.history,
        "created_at": datetime.now().isoformat(),
    }


# -- the top-level export ----------------------------------------------------


def save_model(
    result: TrainingResult,
    config: TrainingConfig,
    out_dir: Path,
    feature_importance: Optional[Sequence[Mapping[str, Any]]] = None,
    attribution: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Path:
    """Write `model.pt` plus every CSV/JSON side export for this run.

    The weights written are the best-epoch snapshot when one exists. That
    snapshot is a genuine copy taken at the time — see `train._snapshot` — so
    what lands on disk is the epoch the reported metrics describe, not
    whichever epoch training happened to stop on.

    Args:
        feature_importance: rows from `permutation_importance_table`.
        attribution: rows from `FeatureAttribution.rows()`.

    Both are optional because they cost extra evaluation passes and are driven
    by the caller's flags; when absent, their files are simply not written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = result.best_state.get("model") or result.model.state_dict()
    model_path = out_dir / "model.pt"
    torch.save(
        {"model": state_dict, "meta": checkpoint_metadata(result, config)},
        model_path,
    )

    _write_json(out_dir / "model_config.json", model_config_document(result, config))
    _write_json(out_dir / "run_summary.json", run_summary_document(result, config))

    write_history_csv(out_dir / "epoch_metrics.csv", result.history)

    for split, matrix in result.confusions.items():
        write_confusion_csv(out_dir / f"confusion_matrix_{split}.csv", matrix)
        write_normalized_confusion_csv(
            out_dir / f"confusion_{split}_normalized.csv", matrix
        )
    for split, matrix in result.boundary_confusions.items():
        write_confusion_csv(out_dir / f"confusion_boundary_{split}.csv", matrix)

    for split, rows in (result.calibration or {}).items():
        write_calibration_csv(out_dir / f"calibration_{split}.csv", rows)

    for split, metrics in result.metrics.items():
        if isinstance(metrics, Mapping) and metrics.get("per_class"):
            write_split_metrics_csv(out_dir / f"metrics_{split}.csv", metrics)

    for name, rows in (result.breakdown or {}).items():
        write_rows_csv(out_dir / f"{name}.csv", rows)

    if feature_importance:
        write_feature_importance_csv(out_dir / "feature_importance.csv", feature_importance)
    if attribution:
        write_attribution_csv(out_dir / "shap_importance.csv", attribution)

    # Retained for readers that already expect it; the same content lives in
    # the per-split CSVs above in a form that loads directly into a dataframe.
    _write_json(out_dir / "metrics.json", result.metrics)
    return model_path


def _write_json(path: Path, document: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(document), indent=2, default=str), encoding="utf-8"
    )
    return path
