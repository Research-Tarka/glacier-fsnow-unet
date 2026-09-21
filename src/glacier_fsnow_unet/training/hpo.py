"""Hyperparameter optimisation with Optuna.

`run_trials` is the entry point for a training job: with `trials == 0` it runs
once with the configured hyperparameters (the reference behaviour, since those
values are already tuned), and otherwise runs an Optuna search followed by a
full-length run at the best point found.

Two aspects of the protocol are deliberate.

*Trials are short, the final run is long.* Each trial trains for
`trial_epochs` (25 by default) rather than the full 500, because ranking
hyperparameters does not need convergence — it needs enough signal to tell
good from bad. The winner is then retrained at full length.

*Trials run on the same seed as each other and as the final model.* Searching
across seeds would confound hyperparameter quality with initialisation luck
and amount to picking the best seed, not the best hyperparameters.
Initialisation variance is measured separately by the bootstrap.

Pruned and failed trials are recorded rather than discarded, so a sweep where
everything got pruned still yields a best-effort answer instead of an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import TrainingConfig
from .dataset import PreparedData, prepare_data
from .metrics import canonical_main_iou_metric
from .search_space import sanitise_overrides, suggest_with_optuna
from .train import TrainingResult, train_unified

__all__ = ["TrialRecord", "run_trials", "build_sampler", "build_pruner"]


@dataclass
class TrialRecord:
    """One trial's outcome, including the pruned and failed cases."""

    trial_id: int
    params: dict[str, Any]
    score: float
    pruned: bool = False
    failed: bool = False
    error: Optional[str] = None
    pruned_epoch: Optional[int] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Whether this trial's score can be trusted for selection."""
        return not self.failed and math.isfinite(self.score) and self.score > -1e11


def build_sampler(name: str) -> Any:
    """Build an Optuna sampler.

    TPE is the default: it models the density of good and bad configurations
    separately, which suits a small budget over a handful of continuous
    hyperparameters better than random search.
    """
    import optuna

    key = str(name or "tpe").strip().lower()
    if key == "cmaes":
        return optuna.samplers.CmaEsSampler()
    if key == "random":
        return optuna.samplers.RandomSampler()
    return optuna.samplers.TPESampler()


def build_pruner(name: str, max_resource: Optional[int] = None) -> Any:
    """Build an Optuna pruner.

    Hyperband stops unpromising trials early and reallocates their budget,
    which roughly triples the number of configurations a fixed budget can
    cover. `none` is the reference setting: with only 50 trials over three
    hyperparameters, pruning risks discarding a configuration that starts slow
    and finishes well.
    """
    import optuna

    key = str(name or "none").strip().lower()
    if key == "median":
        return optuna.pruners.MedianPruner()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, int(max_resource)) if max_resource else "auto",
        )
    return optuna.pruners.NopPruner()


def run_trials(
    config: TrainingConfig,
    data: Optional[PreparedData] = None,
) -> tuple[TrainingResult, TrainingConfig, list[TrialRecord]]:
    """Run the search (if any) and return the final model.

    Returns `(result, winning_config, trial_records)`. With `trials == 0` the
    records list is empty and the winning config is the one passed in.

    The corpus is prepared once and shared across every trial: tiling and
    normalisation do not depend on the searched hyperparameters, so repeating
    them per trial would be pure overhead.
    """
    if data is None:
        data = prepare_data(config)

    if config.trials <= 0 or not config.search_space:
        return train_unified(config, data=data), config, []

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric_name = canonical_main_iou_metric(config.main_iou_metric)
    records: list[TrialRecord] = []

    trial_epochs = config.trial_epochs if config.trial_epochs > 0 else config.epochs

    def objective(trial: Any) -> float:
        overrides = suggest_with_optuna(trial, config.search_space)
        trial_config = config.with_overrides(
            **overrides,
            epochs=trial_epochs,
            patience=min(config.patience, trial_epochs),
        )
        print(f"[hpo] trial {trial.number + 1}/{config.trials} {overrides}")

        try:
            result = train_unified(trial_config, data=data, optuna_trial=trial, progress=False)
        except optuna.TrialPruned:
            records.append(
                TrialRecord(
                    trial_id=trial.number + 1,
                    params=overrides,
                    score=float(trial.user_attrs.get("last_score", -1e12)),
                    pruned=True,
                    pruned_epoch=trial.last_step,
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001 - one bad trial must not end the sweep
            records.append(
                TrialRecord(
                    trial_id=trial.number + 1,
                    params=overrides,
                    score=-1e12,
                    failed=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"[hpo] trial {trial.number + 1} failed: {exc}")
            return -1e12

        score = float(result.best_state.get("val_miou_macro", float("nan")))
        if not math.isfinite(score):
            score = -1e12
        records.append(
            TrialRecord(
                trial_id=trial.number + 1,
                params=overrides,
                score=score,
                metrics=dict(result.best_state),
            )
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=build_sampler(config.optuna_sampler),
        pruner=build_pruner(config.optuna_pruner, max_resource=trial_epochs),
        storage=config.optuna_storage,
        load_if_exists=config.optuna_storage is not None,
    )
    study.optimize(objective, n_trials=config.trials)

    completed = [r for r in records if r.usable and not r.pruned]
    if completed:
        best_params = sanitise_overrides(study.best_params)
    else:
        # Every trial was pruned. Fall back to the best partial result rather
        # than failing: a pruned trial's score is still a real measurement, and
        # a sweep that pruned everything usually means the pruner was too
        # aggressive, not that no configuration works.
        salvageable = [r for r in records if r.usable]
        if not salvageable:
            raise RuntimeError(
                "no Optuna trial produced a usable score; check the search space, "
                "the data, and available GPU memory"
            )
        best = max(salvageable, key=lambda r: r.score)
        best_params = sanitise_overrides(best.params)
        print(f"[hpo] every trial was pruned; falling back to trial {best.trial_id}")

    print(f"[hpo] best {metric_name} at {best_params}; retraining at full length")
    final_config = config.with_overrides(**best_params)
    return train_unified(final_config, data=data), final_config, records
