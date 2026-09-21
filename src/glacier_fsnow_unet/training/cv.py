"""K-fold cross-validation over glaciers.

Each fold holds out one group of glaciers for validation and trains on the
rest. Folds are built at the *glacier* level, never the scene level, for the
same reason the main split is: scenes of one glacier in different years are
near-duplicates, so splitting them across a fold boundary measures memorisation
rather than transfer.

Folds are balanced by annotated pixel count rather than by glacier count.
Glaciers differ in size by orders of magnitude, so five folds of equal
membership can be very unequal in supervision. Assignment is greedy — take
glaciers largest-first, put each in the currently-smallest fold — which gets
close to equal totals in one pass.

There is no test split during cross-validation: every glacier serves as
validation in exactly one fold, so the union of folds already covers the
corpus. `test_ratio` is forced to zero.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .config import IGNORE_INDEX, TrainingConfig
from .dataset import (
    PreparedData,
    SceneRecord,
    collect_tiles,
    compute_class_counts,
    compute_mean_std,
    compute_mean_std_by_sensor,
    group_scenes_by_glacier,
    prepare_data,
)
from .export import save_model, write_metrics_summary
from .train import train_unified

__all__ = ["assign_glacier_folds", "run_cv"]


def assign_glacier_folds(
    glacier_groups: dict[str, list[int]],
    labels: Sequence[np.ndarray],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign each glacier to a fold, balancing annotated pixels per fold.

    Glaciers are shuffled with the given seed, then sorted by descending pixel
    count and placed greedily into whichever fold is currently smallest. The
    largest-first order matters: placing big glaciers last leaves no room to
    compensate for them.

    Deterministic given the same inputs and seed — sorted iteration throughout,
    with the glacier ID as a final tie-break.
    """
    import random

    pixel_counts = {
        gid: sum(int(np.count_nonzero(labels[i] != IGNORE_INDEX)) for i in indices)
        for gid, indices in glacier_groups.items()
    }

    ordered = sorted(glacier_groups)
    random.Random(seed).shuffle(ordered)
    # Descending size, ties broken by ID so the shuffle only decides between
    # glaciers of genuinely equal size.
    ordered.sort(key=lambda gid: (-pixel_counts[gid], gid))

    fold_totals = [0] * n_folds
    assignment: dict[str, int] = {}
    for gid in ordered:
        target = min(range(n_folds), key=lambda f: (fold_totals[f], f))
        assignment[gid] = target
        fold_totals[target] += pixel_counts[gid]

    return assignment


def _fold_data(
    data: PreparedData,
    config: TrainingConfig,
    validation_scenes: set[int],
) -> PreparedData:
    """Re-derive a PreparedData with this fold's scenes held out.

    Tiles are recomputed rather than re-read, and normalisation statistics and
    class weights are recomputed from this fold's training scenes only —
    reusing the full-corpus statistics would leak the held-out fold into
    training.
    """
    tiles = collect_tiles(
        data.labels,
        patch_size=config.patch_size,
        stride=config.stride,
        min_valid=config.min_valid,
        include_rotations=config.rotations,
    )
    train_indices = [
        i for i in range(len(data.scene_records)) if i not in validation_scenes
    ]

    mean, std = compute_mean_std(data.features, train_indices)
    per_sensor_norm, _, _ = config.sensor_adaptation

    return PreparedData(
        features=data.features,
        labels=data.labels,
        boundary_labels=data.boundary_labels,
        scene_records=data.scene_records,
        mean=mean,
        std=std,
        sensor_norm_stats=(
            compute_mean_std_by_sensor(data.features, data.scene_sensors, train_indices)
            or None
            if per_sensor_norm
            else None
        ),
        tiles_train=[t for t in tiles if t[0] not in validation_scenes],
        tiles_val=[t for t in tiles if t[0] in validation_scenes],
        tiles_test=[],
        class_counts=compute_class_counts(data.labels, train_indices),
        patch_size=config.patch_size,
        stride=config.stride,
        scene_context=data.scene_context,
    )


def run_cv(
    config: TrainingConfig,
    data: Optional[PreparedData] = None,
    run_root: Optional[Path] = None,
) -> Path:
    """Run k-fold cross-validation and summarise across folds.

    Returns the run directory holding `fold_*/` and the summary files.
    """
    if data is None:
        data = prepare_data(config)

    run_root = Path(
        run_root or config.model_root / f"cv_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    glacier_groups = group_scenes_by_glacier(data.scene_records)
    n_folds = max(2, min(config.cv_folds, len(glacier_groups)))
    if n_folds != config.cv_folds:
        print(
            f"[cv] {len(glacier_groups)} glaciers available; using {n_folds} folds "
            f"instead of {config.cv_folds}"
        )

    assignment = assign_glacier_folds(
        glacier_groups, data.labels, n_folds, config.cv_seed
    )
    print(f"[cv] {n_folds} folds over {len(glacier_groups)} glaciers -> {run_root}")

    summaries: list[dict[str, Any]] = []
    for fold in range(n_folds):
        held_out = {
            index
            for gid, target in assignment.items()
            if target == fold
            for index in glacier_groups[gid]
        }
        fold_dir = run_root / f"fold_{fold}"
        print(f"[cv] fold {fold + 1}/{n_folds}: {len(held_out)} validation scenes")

        fold_config = config.with_overrides(
            test_ratio=0.0, cv_fold_index=fold, model_root=fold_dir
        )

        try:
            result = train_unified(fold_config, data=_fold_data(data, fold_config, held_out))
        except Exception as exc:  # noqa: BLE001 - one bad fold must not end the sweep
            print(f"[cv] fold {fold} failed: {exc}")
            summaries.append({"fold": fold, "error": str(exc)})
            continue

        save_model(result, fold_config, fold_dir)
        summaries.append(
            {
                "fold": fold,
                "best_epoch": result.best_epoch,
                "n_val_scenes": len(held_out),
                **{
                    key: result.best_state.get(key)
                    for key in (
                        "val_miou_macro",
                        "val_kappa",
                        "val_mcc",
                        "train_loss",
                        "val_loss",
                    )
                },
            }
        )

    scores = [
        float(row["val_miou_macro"])
        for row in summaries
        if isinstance(row.get("val_miou_macro"), (int, float))
        and row["val_miou_macro"] == row["val_miou_macro"]
    ]
    aggregate = (
        {
            "val_miou_macro_mean": float(np.mean(scores)),
            "val_miou_macro_std": float(np.std(scores)),
            "n_folds_completed": len(scores),
        }
        if scores
        else {}
    )

    (run_root / "cv_summary.json").write_text(
        json.dumps(
            {
                "n_folds": n_folds,
                "cv_seed": config.cv_seed,
                "n_glaciers": len(glacier_groups),
                "per_fold": summaries,
                "aggregate": aggregate,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_metrics_summary(run_root / "cv_summary.csv", summaries)

    if aggregate:
        print(
            f"[cv] val mIoU {aggregate['val_miou_macro_mean']:.4f} "
            f"+/- {aggregate['val_miou_macro_std']:.4f} "
            f"over {aggregate['n_folds_completed']} folds"
        )
    return run_root
