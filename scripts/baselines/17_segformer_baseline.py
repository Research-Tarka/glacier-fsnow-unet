#!/usr/bin/env python3
"""SegFormer baseline (HuggingFace transformers, MiT encoder truncated to 3
stages for the 48x48 input -- see glacier_fsnow_unet.training.segformer_model
for the scale-check rationale).

Run with no arguments to do everything: Optuna search over
lr/weight_decay/dropout_p, then multi-seed confirmation of the best trial
found, evaluated on test and compared against the published U-Net. Reads
`baselines.segformer` from configs/config.yaml by default.

Two subcommands are also available to run either stage on its own:

    hpo      Optuna search only. Writes hpo_segformer_summary.json.
    confirm  Multi-seed confirmation only, of --lr/--weight-decay/--dropout-p
             (or baselines.segformer.lr/weight_decay/dropout_p if not
             passed). Writes segformer_confirmation_summary.json plus one
             checkpoint per seed.

Examples:

    python scripts/baselines/17_segformer_baseline.py
    python scripts/baselines/17_segformer_baseline.py hpo
    python scripts/baselines/17_segformer_baseline.py confirm --lr 0.0009 --weight-decay 0.0002 --dropout-p 0.15

Mixed precision (AMP) and cuDNN autotuning are enabled by default on CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import (  # noqa: E402
    BaselinesConfigError,
    build_split,
    check_no_duplicate_training_process,
    compare_to_unet,
    load_baselines_config,
    torch_baseline_checkpoint_meta,
    train_one_config,
)
from glacier_fsnow_unet.training.segformer_model import build_segformer_glacier  # noqa: E402


def _forward(model, x):
    return model(x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: configs/config.yaml)")
    parser.add_argument("--out", default=None, help="output directory (overrides config)")
    parser.add_argument("--train-root", default=None, help="corpus root (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--no-require-cross-check", action="store_true",
        help="proceed even if no reference checkpoint is found for the split cross-check",
    )
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")

    def add_hpo_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--n-trials", type=int, default=None)
        p.add_argument("--trial-epochs", type=int, default=None)
        p.add_argument("--trial-patience", type=int, default=None)
        p.add_argument("--trial-seed", type=int, default=None)

    def add_confirm_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--weight-decay", type=float, default=None)
        p.add_argument("--dropout-p", type=float, default=None)
        p.add_argument("--n-seeds", type=int, default=None)
        p.add_argument("--seed-start", type=int, default=None)
        p.add_argument("--final-epochs", type=int, default=None)
        p.add_argument("--final-patience", type=int, default=None)

    # No subcommand (the default): run hpo, then confirm on its best trial.
    add_hpo_args(parser)
    add_confirm_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=False)

    hpo = subparsers.add_parser("hpo", help="Optuna search over lr/weight_decay/dropout_p only")
    add_hpo_args(hpo)

    confirm = subparsers.add_parser("confirm", help="multi-seed confirmation of one hyperparameter set only")
    add_confirm_args(confirm)

    return parser


def run_hpo(cfg, data, out_dir: Path, args, num_workers: int, batch_size: int, use_amp: bool) -> dict:
    n_trials = args.n_trials or cfg.n_trials
    trial_epochs = args.trial_epochs or cfg.trial_epochs
    trial_patience = args.trial_patience or cfg.trial_patience
    trial_seed = args.trial_seed if args.trial_seed is not None else cfg.trial_seed
    storage_path = Path(cfg.optuna_storage) if cfg.optuna_storage else out_dir / "optuna_segformer.db"

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout_p = trial.suggest_float("dropout_p", 0.0, 0.5)

        def build_model_fn():
            return build_segformer_glacier(in_channels=11, num_classes=4, dropout_p=dropout_p)

        result = train_one_config(
            data, build_model_fn=build_model_fn, forward_fn=_forward,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            max_epochs=trial_epochs, patience=trial_patience,
            seed=trial_seed, log_prefix=f"segformer-trial{trial.number}",
            num_workers=num_workers, dropout_p=dropout_p, use_amp=use_amp,
        )
        result.pop("model_state", None)
        trial.set_user_attr("val_miou", result["best_val_miou"])
        trial.set_user_attr("n_epochs_run", result["n_epochs_run"])
        trial.set_user_attr("total_time_s", result["total_time_s"])

        with open(out_dir / "hpo_trials_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "trial": trial.number, "lr": lr, "weight_decay": weight_decay,
                "dropout_p": dropout_p, "val_miou": result["best_val_miou"],
                "n_epochs_run": result["n_epochs_run"], "total_time_s": result["total_time_s"],
            }) + "\n")
        return result["best_val_miou"]

    study = optuna.create_study(
        study_name="segformer_hpo", storage=f"sqlite:///{storage_path}",
        direction="maximize", load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
    )
    n_done = len(study.trials)
    n_remaining = max(0, n_trials - n_done)
    print(f"[hpo] {n_done} trials already done, {n_remaining} remaining (target {n_trials}).")
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining)

    best = study.best_trial
    print(f"[result] BEST TRIAL #{best.number} val_miou={best.value:.4f} params={best.params}")

    summary = {
        "n_trials": len(study.trials), "best_trial_number": best.number,
        "best_val_miou": best.value, "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(out_dir / "hpo_segformer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[done] {out_dir / 'hpo_segformer_summary.json'}")
    print(f"[next] confirm --lr {best.params['lr']} --weight-decay {best.params['weight_decay']} "
          f"--dropout-p {best.params['dropout_p']}")
    return best.params


def run_confirm(
    cfg, data, out_dir: Path, args, num_workers: int, batch_size: int, use_amp: bool,
    best_params: dict | None = None,
) -> int:
    import numpy as np
    import torch

    best_params = best_params or {}
    lr = args.lr if args.lr is not None else best_params.get("lr", cfg.lr)
    weight_decay = args.weight_decay if args.weight_decay is not None else best_params.get("weight_decay", cfg.weight_decay)
    dropout_p = args.dropout_p if args.dropout_p is not None else best_params.get("dropout_p", cfg.dropout_p)
    if lr is None or weight_decay is None:
        print(
            "[error] no (lr, weight_decay) available -- run the hpo subcommand first, "
            "set baselines.segformer.lr/weight_decay in configs/config.yaml, or pass "
            "--lr/--weight-decay explicitly.",
            file=sys.stderr,
        )
        return 2

    n_seeds = args.n_seeds or cfg.n_seeds
    seed_start = args.seed_start if args.seed_start is not None else cfg.seed_start
    final_epochs = args.final_epochs or cfg.final_epochs
    final_patience = args.final_patience or cfg.final_patience

    def build_model_fn():
        return build_segformer_glacier(in_channels=11, num_classes=4, dropout_p=dropout_p)

    per_seed_results = []
    test_mious = []
    for seed in range(seed_start, seed_start + n_seeds):
        print(f"[confirm] ==== SEED {seed} ====")
        result = train_one_config(
            data, build_model_fn=build_model_fn, forward_fn=_forward,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            max_epochs=final_epochs, patience=final_patience,
            seed=seed, log_prefix=f"segformer-confirm-seed{seed}",
            num_workers=num_workers, dropout_p=dropout_p, use_amp=use_amp,
        )
        model_state = result.pop("model_state", None)
        if model_state is not None:
            meta = torch_baseline_checkpoint_meta(data)
            meta.update({
                "architecture": "segformer_3stage",
                "lr": lr, "weight_decay": weight_decay, "dropout_p": dropout_p, "seed": seed,
            })
            torch.save(
                {"model": model_state, "meta": meta},
                out_dir / f"segformer_seed{seed}_checkpoint.pt",
            )
        test_miou = result["metrics"]["test"]["macro"]["miou"]
        test_mious.append(test_miou)
        per_seed_results.append(result)
        print(f"[result] seed {seed} -> test_miou={test_miou:.4f}")
        with open(out_dir / f"segformer_seed{seed}_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

    comparison = compare_to_unet(test_mious)
    print(f"[result] comparison vs U-Net: {json.dumps(comparison, indent=2)}")

    summary = {
        "architecture": "SegFormer (3-stage MiT encoder, all-MLP decoder)",
        "hyperparameters": {"lr": lr, "weight_decay": weight_decay, "dropout_p": dropout_p},
        "n_seeds": n_seeds,
        "test_miou_per_seed": test_mious,
        "test_miou_mean": float(np.mean(test_mious)),
        "test_miou_std": float(np.std(test_mious, ddof=1)) if len(test_mious) > 1 else float("nan"),
        "comparison_vs_unet": comparison,
    }
    with open(out_dir / "segformer_confirmation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[done] {out_dir / 'segformer_confirmation_summary.json'}")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    try:
        pipeline_cfg, default_train_root = load_baselines_config(args.config)
    except BaselinesConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    cfg = pipeline_cfg.segformer
    out_dir = Path(args.out) if args.out else REPO_ROOT / cfg.out_dir
    train_root = Path(args.train_root) if args.train_root else default_train_root
    num_workers = args.num_workers if args.num_workers is not None else cfg.num_workers
    batch_size = args.batch_size or cfg.batch_size
    use_amp = not args.no_amp

    print(f"[config] {args.config or 'configs/config.yaml'}")
    print(f"[config] command={args.command or 'all (hpo+confirm)'} corpus={train_root} output={out_dir}")
    print(f"[config] batch_size={batch_size} num_workers={num_workers} amp={use_amp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    check_no_duplicate_training_process()

    check_checkpoint = (
        Path(pipeline_cfg.check_checkpoint) if pipeline_cfg.check_checkpoint else None
    )
    if check_checkpoint and not check_checkpoint.is_absolute():
        check_checkpoint = REPO_ROOT / check_checkpoint
    data, check = build_split(
        train_root=train_root, check_checkpoint=check_checkpoint,
        require_cross_check=not args.no_require_cross_check,
    )
    with open(out_dir / "split_check.json", "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2, default=str)

    if args.command == "hpo":
        run_hpo(cfg, data, out_dir, args, num_workers, batch_size, use_amp)
        return 0
    if args.command == "confirm":
        return run_confirm(cfg, data, out_dir, args, num_workers, batch_size, use_amp)

    print("[main] no subcommand given -- running hpo then confirm on its best trial")
    best_params = run_hpo(cfg, data, out_dir, args, num_workers, batch_size, use_amp)
    return run_confirm(cfg, data, out_dir, args, num_workers, batch_size, use_amp, best_params=best_params)


if __name__ == "__main__":
    raise SystemExit(main())
