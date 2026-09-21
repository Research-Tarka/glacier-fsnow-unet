#!/usr/bin/env python3
"""Random Forest baseline: the middle rung of the value staircase between
the NDSI/NDVI threshold classifier (test mIoU 0.402) and the U-Net
(test mIoU 0.701 +/- 0.008).

Trains a scikit-learn RandomForestClassifier on the same 11 normalized
spectral indices the U-Net consumes, using the exact glacier-level split
(cross-checked scene-by-scene against a real trained checkpoint). Two
hyperparameter-search modes, selected via `--mode` or `baselines.rf.mode` in
configs/config.yaml:

    grid    The original 24-combination manual grid search (test mIoU =
            0.570, the number cited in the paper). Kept for exact
            reproducibility of that result.
    optuna  TPE-sampled search over a wider version of the same
            hyperparameter families (the recommended default).

Both modes search on a stratified 400,000-pixel subsample of the ~2.24M-
pixel train partition (selecting on VALID macro mIoU, never test), then
refit the winner on the FULL train partition. The final model is then
evaluated once on TEST, plus `--n-seeds` additional full-data refits at the
same hyperparameters (different random_state) to report a seed-sensitivity
range -- RF's own bagging already averages over many trees, so this
spread is expected to be far tighter than the U-Net's bootstrap spread.

Examples:

    # Reads baselines.rf from configs/config.yaml (mode=optuna by default).
    python scripts/baselines/15_rf_baseline.py

    # Reproduce the published grid-search result exactly.
    python scripts/baselines/15_rf_baseline.py --mode grid

    # Vary trial count without touching the YAML.
    python scripts/baselines/15_rf_baseline.py --mode optuna --n-trials 30

`n_jobs` is deliberately capped (not -1) for the full-2.24M-pixel refits:
each worker holds intermediate per-tree buffers sized off the full array,
and building hundreds of trees across every core at once can spike
working-set memory past available RAM. The grid/Optuna search stage
(subsample-sized fits) remains safe at n_jobs=-1.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import (  # noqa: E402
    BaselinesConfigError,
    build_split,
    compare_to_unet,
    load_baselines_config,
    scene_indices_for_split,
)
from glacier_fsnow_unet.training.config import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_FEATURES,
    IGNORE_INDEX,
    NUM_CLASSES,
)
from glacier_fsnow_unet.training.metrics import compute_confusion_metrics  # noqa: E402

N_FEATURES = len(DEFAULT_FEATURES)


def _extract_one_scene(data, si: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    label = data.labels[si]
    valid = label != IGNORE_INDEX
    if not np.any(valid):
        return None
    feat = data.features[si].astype(np.float32)
    return feat[:, valid].T, label[valid].astype(np.int64)


def extract_pixels(
    data, scene_indices: list[int], n_jobs: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-scene (C, H, W) feature/label arrays into (N, C) / (N,)
    pixel arrays. Per-scene extraction is independent, so it is parallelised
    across scenes with threads: numpy's masking/transpose releases the GIL,
    and this stage is otherwise the dominant wall-clock cost before fitting.
    """
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_one_scene)(data, si) for si in scene_indices
    )
    feats_all = [r[0] for r in results if r is not None]
    labels_all = [r[1] for r in results if r is not None]
    X = np.concatenate(feats_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    return X, y


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_target: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    if n_target >= n_total:
        return X, y
    classes, counts = np.unique(y, return_counts=True)
    fractions = counts / n_total
    take_per_class = np.maximum(1, np.round(fractions * n_target).astype(int))
    selected_idx = []
    for cls, take in zip(classes, take_per_class):
        cls_idx = np.flatnonzero(y == cls)
        take = min(take, cls_idx.size)
        selected_idx.append(rng.choice(cls_idx, size=take, replace=False))
    selected_idx = np.concatenate(selected_idx)
    rng.shuffle(selected_idx)
    return X[selected_idx], y[selected_idx]


def confusion_matrix_from_preds(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401  (import check)

    pred = model.predict(X)
    confusion = confusion_matrix_from_preds(y, pred)
    metrics = compute_confusion_metrics(confusion)
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class": metrics["per_class"], "macro": metrics["macro"],
    }


def manual_grid_search(
    X_train_sub, y_train_sub, X_valid, y_valid, n_jobs: int, primary_seed: int,
) -> tuple[dict, float, list[dict]]:
    """The original 24-combination manual grid (historical, for exact
    reproducibility of the published test mIoU = 0.570)."""
    from sklearn.ensemble import RandomForestClassifier

    grid = {
        "n_estimators": [200, 500], "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", None],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"[hp_search] {len(combos)} combinations over {keys}")

    history, best_score, best_params = [], -1.0, None
    t0 = time.time()
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        clf = RandomForestClassifier(
            n_jobs=n_jobs, random_state=primary_seed, class_weight=None, **params,
        )
        t_fit0 = time.time()
        clf.fit(X_train_sub, y_train_sub)
        fit_s = time.time() - t_fit0

        pred_valid = clf.predict(X_valid)
        confusion = confusion_matrix_from_preds(y_valid, pred_valid)
        score = float(compute_confusion_metrics(confusion)["macro"]["miou"])
        history.append({**params, "valid_miou_macro": score, "fit_seconds": round(fit_s, 2)})
        print(f"[hp_search] {i + 1}/{len(combos)} {params} -> valid_miou={score:.4f} ({fit_s:.1f}s)")
        if score > best_score:
            best_score, best_params = score, params
    print(f"[hp_search] done in {time.time() - t0:.1f}s; best={best_params} valid_miou={best_score:.4f}")
    return best_params, best_score, history


def optuna_search(
    X_train_sub, y_train_sub, X_valid, y_valid, n_trials: int, n_jobs: int,
    primary_seed: int, optuna_seed: int, storage_path: Path,
) -> tuple[dict, float, list[dict]]:
    import optuna
    from sklearn.ensemble import RandomForestClassifier

    def objective(trial: "optuna.Trial") -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_categorical("max_depth", [None, 10, 15, 20, 25, 30, 40, 50]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        )
        clf = RandomForestClassifier(n_jobs=n_jobs, random_state=primary_seed, class_weight=None, **params)
        clf.fit(X_train_sub, y_train_sub)
        pred_valid = clf.predict(X_valid)
        confusion = confusion_matrix_from_preds(y_valid, pred_valid)
        score = float(compute_confusion_metrics(confusion)["macro"]["miou"])
        trial.set_user_attr("valid_miou", score)
        del clf
        gc.collect()
        return score

    study = optuna.create_study(
        study_name="rf_hpo", storage=f"sqlite:///{storage_path}", direction="maximize",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=optuna_seed),
        pruner=optuna.pruners.NopPruner(),
    )
    n_done = len(study.trials)
    n_remaining = max(0, n_trials - n_done)
    print(f"[rf_optuna] {n_done} trials already done, {n_remaining} remaining (target {n_trials}).")
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining)

    best = study.best_trial
    print(f"[rf_optuna] BEST TRIAL #{best.number} valid_miou={best.value:.4f} params={best.params}")
    history = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    return best.params, float(best.value), history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument("--out", default=None, help="output directory (overrides config)")
    parser.add_argument("--train-root", default=None, help="corpus root (overrides config)")
    parser.add_argument("--mode", default=None, choices=("grid", "optuna"))

    run = parser.add_argument_group("run overrides")
    run.add_argument("--n-trials", type=int, default=None, help="Optuna trials (mode=optuna)")
    run.add_argument("--n-seeds", type=int, default=None, help="confirmation seeds")
    run.add_argument("--seed-start", type=int, default=None)
    run.add_argument("--subsample-size", type=int, default=None)
    run.add_argument("--full-data-n-jobs", type=int, default=None)
    run.add_argument("--search-n-jobs", type=int, default=None)
    run.add_argument(
        "--no-require-cross-check", action="store_true",
        help="proceed even if no reference checkpoint is found for the split cross-check",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        cfg, default_train_root = load_baselines_config(args.config)
    except BaselinesConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    rf_cfg = cfg.rf
    mode = args.mode or rf_cfg.mode
    out_dir = Path(args.out) if args.out else REPO_ROOT / rf_cfg.out_dir
    train_root = Path(args.train_root) if args.train_root else default_train_root
    subsample_size = args.subsample_size or rf_cfg.subsample_size
    n_trials = args.n_trials or rf_cfg.n_trials
    n_seeds = args.n_seeds or rf_cfg.n_seeds
    seed_start = args.seed_start if args.seed_start is not None else rf_cfg.seed_start
    full_data_n_jobs = args.full_data_n_jobs or rf_cfg.full_data_n_jobs
    search_n_jobs = args.search_n_jobs if args.search_n_jobs is not None else rf_cfg.search_n_jobs

    print(f"[config] {args.config or 'configs/config.yaml'}")
    print(f"[config] mode={mode} corpus={train_root} output={out_dir}")
    print(f"[config] subsample_size={subsample_size:,} full_data_n_jobs={full_data_n_jobs}")

    out_dir.mkdir(parents=True, exist_ok=True)

    check_checkpoint = Path(cfg.check_checkpoint) if cfg.check_checkpoint else None
    if check_checkpoint and not check_checkpoint.is_absolute():
        check_checkpoint = REPO_ROOT / check_checkpoint
    data, check = build_split(
        train_root=train_root, check_checkpoint=check_checkpoint,
        require_cross_check=not args.no_require_cross_check,
    )

    train_idx = scene_indices_for_split(data, "train")
    valid_idx = scene_indices_for_split(data, "valid")
    test_idx = scene_indices_for_split(data, "test")

    print("[data] extracting pixel arrays ...")
    t0 = time.time()
    X_train, y_train = extract_pixels(data, train_idx)
    X_valid, y_valid = extract_pixels(data, valid_idx)
    X_test, y_test = extract_pixels(data, test_idx)
    print(
        f"[data] extracted train={X_train.shape} valid={X_valid.shape} "
        f"test={X_test.shape} in {time.time() - t0:.1f}s"
    )

    train_class_counts = {CLASS_NAMES[c]: int((y_train == c).sum()) for c in range(NUM_CLASSES)}
    print(f"[data] full train class counts: {train_class_counts}")

    X_train_sub, y_train_sub = stratified_subsample(
        X_train, y_train, subsample_size, rf_cfg.subsample_seed
    )
    print(f"[data] HPO subsample size={X_train_sub.shape[0]:,} (seed={rf_cfg.subsample_seed})")

    if mode == "grid":
        best_params, best_valid_score, history = manual_grid_search(
            X_train_sub, y_train_sub, X_valid, y_valid,
            n_jobs=search_n_jobs, primary_seed=rf_cfg.primary_seed,
        )
    else:
        storage_path = (
            Path(rf_cfg.optuna_storage) if rf_cfg.optuna_storage else out_dir / "optuna_rf.db"
        )
        best_params, best_valid_score, history = optuna_search(
            X_train_sub, y_train_sub, X_valid, y_valid,
            n_trials=n_trials, n_jobs=search_n_jobs,
            primary_seed=rf_cfg.primary_seed, optuna_seed=rf_cfg.optuna_seed,
            storage_path=storage_path,
        )

    with open(out_dir / "hp_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"mode": mode, "best_params": best_params, "best_valid_miou_macro": best_valid_score,
             "history": history, "split_check": check},
            f, indent=2, default=str,
        )
    gc.collect()

    print(f"[main] refitting FINAL model on FULL train partition "
          f"({X_train.shape[0]:,} pixels) with best_params={best_params} ...")
    from sklearn.ensemble import RandomForestClassifier

    def make_rf(seed: int) -> "RandomForestClassifier":
        return RandomForestClassifier(
            n_jobs=full_data_n_jobs, random_state=seed, class_weight=None, **best_params,
        )

    t0 = time.time()
    final_model = make_rf(rf_cfg.primary_seed)
    final_model.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    print(f"[main] final fit done in {fit_seconds:.1f}s")
    gc.collect()

    results = {}
    for split_name, X, y in (("train", X_train, y_train), ("valid", X_valid, y_valid), ("test", X_test, y_test)):
        report = evaluate(final_model, X, y)
        report["split"] = split_name
        report["n_pixels"] = int(X.shape[0])
        results[split_name] = report
        print(f"[result] {split_name}: miou_macro={report['macro']['miou']} "
              f"kappa={report['macro']['kappa']} mcc={report['macro']['mcc']}")

    feature_importances = {
        DEFAULT_FEATURES[i]: float(final_model.feature_importances_[i]) for i in range(N_FEATURES)
    }

    import joblib

    joblib.dump(final_model, out_dir / "rf_model_primary_seed.joblib")

    print(f"[main] seed-sensitivity check over {n_seeds} seeds starting at {seed_start} ...")
    seed_results = {}
    test_mious = []
    for seed in range(seed_start, seed_start + n_seeds):
        if seed == rf_cfg.primary_seed and seed == seed_start:
            model_s = final_model
        else:
            model_s = make_rf(seed)
            model_s.fit(X_train, y_train)
        report_s = evaluate(model_s, X_test, y_test)
        seed_results[seed] = {
            "test_miou_macro": report_s["macro"]["miou"],
            "test_kappa": report_s["macro"]["kappa"],
            "test_mcc": report_s["macro"]["mcc"],
        }
        test_mious.append(report_s["macro"]["miou"])
        print(f"[main] seed={seed} test_miou={report_s['macro']['miou']:.4f}")
        if model_s is not final_model:
            del model_s
        gc.collect()

    seed_mious = np.array(test_mious, dtype=np.float64)
    seed_summary = {
        "n_seeds": len(seed_results), "seeds": sorted(seed_results.keys()),
        "per_seed": seed_results,
        "test_miou_mean": float(seed_mious.mean()),
        "test_miou_std": float(seed_mious.std(ddof=1)) if seed_mious.size > 1 else float("nan"),
    }
    print(f"[result] seed-sensitivity: test_miou = {seed_summary['test_miou_mean']:.4f} "
          f"+/- {seed_summary['test_miou_std']:.4f} over {seed_summary['n_seeds']} seeds")

    comparison = compare_to_unet(test_mious)
    print(f"[result] comparison vs U-Net: {json.dumps(comparison, indent=2)}")

    output = {
        "mode": mode,
        "hp_search": {"best_params": best_params, "best_valid_miou_macro": best_valid_score},
        "final_fit_seconds": fit_seconds,
        "train_full_class_counts": train_class_counts,
        "feature_importances": feature_importances,
        "split_cross_check": check,
        "results_primary_seed": results,
        "seed_sensitivity": seed_summary,
        "comparison_vs_unet": comparison,
    }
    with open(out_dir / "rf_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[done] {out_dir / 'rf_baseline_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
