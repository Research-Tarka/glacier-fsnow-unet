"""Multi-seed bootstrap: quantifying initialisation variance.

Trains the same configuration N times, varying only the random seed, and
reports the spread. This separates two things that a single run conflates: how
good the configuration is, and how much of a given run's score was luck in the
weight initialisation and batch ordering.

The protocol's key property is that **the split is identical across seeds**.
Only initialisation and mini-batch order vary. If the split changed too, the
resulting spread would mix initialisation variance with split variance and
could not be reported as either. Seeding is therefore split-independent by
construction: `TrainingConfig.split_seed` is fixed while `seed` varies.

Reporting the mean and standard deviation across seeds, rather than the best
seed, is what makes the number honest — picking the best of five and reporting
it is a silent one-in-five selection on the validation set.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from .attribution_runner import AttributionOptions, compute_attributions
from .config import TrainingConfig
from .dataset import PreparedData, prepare_data
from .export import save_model, write_metrics_summary, write_rows_csv
from .hpo import run_trials
from .train import TrainingResult

__all__ = ["run_bootstrap", "build_ensemble", "aggregate_rows", "AGGREGATE_METRICS"]


def build_ensemble(state_dicts: list[dict[str, Any]], out_dir: Path) -> Optional[Path]:
    """Write an ensemble checkpoint holding every seed's weights.

    Stores the state dicts side by side rather than averaging them. Averaging
    weights across independently initialised networks is meaningless — their
    hidden units are not in correspondence. The ensemble is applied at
    inference by averaging softmax outputs, which is well defined.
    """
    if not state_dicts:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.pt"
    torch.save(
        {
            "ensemble_state_dicts": state_dicts,
            "n_models": len(state_dicts),
            "ensemble": True,
        },
        path,
    )
    return path


def _seed_summary(seed: int, result: TrainingResult) -> dict[str, Any]:
    """One row describing a finished seed.

    Pulls the held-out numbers from `result.metrics` rather than only from
    `best_state`, so the across-seed spread can be reported for test as well as
    validation — the paper's quoted seed variance is a validation number, and
    without the test columns there is nothing to check it against.
    """
    row: dict[str, Any] = {
        "seed": seed,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
    }
    for key in (
        "val_miou_macro",
        "val_miou_weighted",
        "val_miou_inv_freq",
        "val_kappa",
        "val_mcc",
        "val_glacier_iou",
        "train_loss",
        "val_loss",
    ):
        row[key] = result.best_state.get(key)

    for split, prefix in (("valid", "val"), ("test", "test")):
        metrics = result.metrics.get(split)
        macro = metrics.get("macro") if isinstance(metrics, Mapping) else None
        if not isinstance(macro, Mapping):
            continue
        if prefix == "test":
            row["test_miou_macro"] = macro.get("miou")
            row["test_miou_weighted"] = macro.get("miou_weighted")
            row["test_kappa"] = macro.get("kappa")
            row["test_mcc"] = macro.get("mcc")

    return row


def run_bootstrap(
    config: TrainingConfig,
    data: Optional[PreparedData] = None,
    run_root: Optional[Path] = None,
    attribution_options: Optional[AttributionOptions] = None,
) -> Path:
    """Train `bootstrap_n_seeds` models and summarise the spread.

    The corpus is prepared once and reused: it depends on `split_seed`, which
    does not vary across bootstrap seeds, so re-preparing it per seed would
    both waste time and risk the split drifting between seeds.

    Returns the run directory holding `seed_*/`, `ensemble/`, and the summary.
    """
    if data is None:
        data = prepare_data(config)

    run_root = Path(
        run_root
        or config.model_root / f"bootstrap_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    n_seeds = max(1, config.bootstrap_n_seeds)
    print(f"[bootstrap] {n_seeds} seed(s) from base {config.bootstrap_seed_base} -> {run_root}")

    summaries: list[dict[str, Any]] = []
    state_dicts: list[dict[str, Any]] = []

    for offset in range(n_seeds):
        seed = config.bootstrap_seed_base + offset
        seed_dir = run_root / f"seed_{offset}"
        print(f"[bootstrap] seed {seed}")

        # Only `seed` changes. `split_seed` is untouched, so every seed trains
        # and validates on exactly the same glaciers.
        seed_config = config.with_overrides(seed=seed, model_root=seed_dir)

        try:
            result, winning_config, _trials = run_trials(seed_config, data=data)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"[bootstrap] seed {seed} failed: {exc}")
            summaries.append({"seed": seed, "error": str(exc)})
            continue

        importance, attribution = compute_attributions(
            result, winning_config, data, attribution_options
        )
        save_model(
            result,
            winning_config,
            seed_dir,
            feature_importance=importance,
            attribution=attribution,
        )
        summaries.append(_seed_summary(seed, result))
        if result.best_state.get("model") is not None:
            state_dicts.append(result.best_state["model"])

    ensemble_path = (
        build_ensemble(state_dicts, run_root / "ensemble")
        if config.bootstrap_create_ensemble
        else None
    )

    aggregate = _aggregate(summaries)
    (run_root / "bootstrap_summary.json").write_text(
        json.dumps(
            {
                "n_seeds": n_seeds,
                "seed_base": config.bootstrap_seed_base,
                "ensemble_path": str(ensemble_path) if ensemble_path else None,
                "per_seed": summaries,
                "aggregate": aggregate,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_metrics_summary(run_root / "bootstrap_summary.csv", summaries)
    write_rows_csv(
        run_root / "bootstrap_aggregate.csv",
        aggregate_rows(aggregate),
        columns=["metric", "n_seeds", "mean", "std", "min", "max"],
    )

    for metric, stats in sorted(aggregate.items()):
        print(f"[bootstrap] {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    return run_root


# Every numeric field a per-seed summary can carry, aggregated across seeds.
# Listed explicitly rather than discovered, so a field appearing in only some
# seeds' summaries cannot silently change what the aggregate covers.
AGGREGATE_METRICS: tuple[str, ...] = (
    "best_epoch",
    "stopped_epoch",
    "val_miou_macro",
    "val_miou_weighted",
    "val_miou_inv_freq",
    "val_kappa",
    "val_mcc",
    "val_glacier_iou",
    "test_miou_macro",
    "test_miou_weighted",
    "test_kappa",
    "test_mcc",
    "train_loss",
    "val_loss",
)


def _aggregate(
    summaries: list[dict[str, Any]],
    metrics: Sequence[str] = AGGREGATE_METRICS,
) -> dict[str, dict[str, float]]:
    """Mean, std, min and max of each metric across the seeds that succeeded.

    Population std (`ddof=0`), matching how the seed-to-seed spread is
    conventionally reported for a fixed, small number of replications. This is
    the number the reported seed-to-seed uncertainty comes from, so it is
    computed over every metric a seed records rather than a chosen few — a
    spread quoted for mIoU says nothing about whether kappa was equally stable.
    """
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [
            float(row[metric])
            for row in summaries
            if isinstance(row.get(metric), (int, float))
            and not isinstance(row.get(metric), bool)
            and row[metric] == row[metric]
        ]
        if values:
            array = np.asarray(values, dtype=np.float64)
            out[metric] = {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
                "n": len(values),
            }
    return out


def aggregate_rows(aggregate: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    """The aggregate as one row per metric, for a CSV alongside the per-seed rows.

    Kept as a separate file rather than extra columns on the per-seed table:
    the two have different row semantics (one row per seed against one row per
    metric) and merging them produces a table where half the cells are blank.
    """
    return [
        {
            "metric": metric,
            "n_seeds": int(stats.get("n", 0)),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        for metric, stats in sorted(aggregate.items())
    ]
