#!/usr/bin/env python3
"""Train the four-class glacier surface segmentation U-Net.

Dispatches to one of four strategies based on the configuration:

    hpo + bootstrap -> search hyperparameters, then bootstrap the winner
    cross-validation -> k-fold over glaciers
    bootstrap        -> N seeds at fixed hyperparameters
    single           -> one run

The reference configuration takes the bootstrap path with `n_seeds: 1` and
`trials: 0`, which is a single run at already-tuned hyperparameters.

Examples:

    # Train exactly as configured.
    python scripts/05_train_model.py --config configs/config.yaml

    # Short run on a slice of the corpus, sized for a laptop GPU.
    python scripts/05_train_model.py --epochs 3 --batch-size 32 \\
        --num-workers 2 --max-scenes 120 --out _work/smoke

    # Bit-reproducible run (slower: disables the cuDNN autotuner).
    python scripts/05_train_model.py --deterministic

    # Also write both feature-attribution tables.
    python scripts/05_train_model.py --feature-importance --shap

Every run writes the model, its full configuration, per-epoch history,
per-class metrics and confusion matrices per partition, and the per-scene,
per-glacier and per-sensor breakdowns. The two feature-attribution tables are
behind flags because each costs extra evaluation passes over a finished model
and neither feeds back into training; see `training/attribution_runner.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.training.attribution_runner import (  # noqa: E402
    AttributionOptions,
    compute_attributions,
)
from glacier_fsnow_unet.training.bootstrap import run_bootstrap  # noqa: E402
from glacier_fsnow_unet.training.config import training_config_from_pipeline  # noqa: E402
from glacier_fsnow_unet.training.cv import run_cv  # noqa: E402
from glacier_fsnow_unet.training.dataset import prepare_data  # noqa: E402
from glacier_fsnow_unet.training.export import save_model  # noqa: E402
from glacier_fsnow_unet.training.hpo import run_trials  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--out", default=None, help="output directory (overrides config)")
    parser.add_argument(
        "--train-root", default=None, help="corpus root (overrides config)"
    )

    run = parser.add_argument_group("run overrides")
    run.add_argument("--epochs", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--num-workers", type=int, default=None)
    run.add_argument("--patience", type=int, default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="seed for the train/val/test partition (independent of --seed, "
        "which only varies model initialisation across bootstrap runs)",
    )
    run.add_argument("--device", default=None, help="cuda | cpu")
    run.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="use only the first N scenes, for quick checks",
    )
    run.add_argument(
        "--n-seeds", type=int, default=None, help="bootstrap seed count"
    )
    run.add_argument("--trials", type=int, default=None, help="Optuna trials; 0 disables")

    architecture = parser.add_argument_group(
        "architecture variants",
        "All off by default. The defaults reproduce the published checkpoint's "
        "architecture exactly; any of these makes the model a different one.",
    )
    architecture.add_argument(
        "--norm-type",
        default=None,
        choices=("batch", "group"),
        help=(
            "normalisation layer: batch (reference, matches the published "
            "checkpoint) or group (batch-independent, no running statistics)"
        ),
    )
    architecture.add_argument(
        "--bottleneck-attention",
        action="store_true",
        help="add multi-head self-attention over the bottleneck grid",
    )
    architecture.add_argument(
        "--bottleneck-attention-heads",
        type=int,
        default=None,
        help="attention heads at the bottleneck (default 8)",
    )
    architecture.add_argument(
        "--deep-supervision",
        action="store_true",
        help=(
            "add auxiliary logit heads to the intermediate decoder levels, "
            "each scored against labels reduced to its own resolution"
        ),
    )

    attribution = parser.add_argument_group("feature attribution")
    attribution.add_argument(
        "--feature-importance",
        action="store_true",
        help=(
            "write feature_importance.csv: per feature and per class, the IoU "
            "lost when that input channel is shuffled. Costs one extra "
            "evaluation pass per channel, so it is off by default"
        ),
    )
    attribution.add_argument(
        "--shap",
        action="store_true",
        help=(
            "write shap_importance.csv: per feature and per class, the "
            "expected-gradient (Shapley) attribution. Costs a backward pass "
            "per baseline draw, so it is off by default"
        ),
    )
    attribution.add_argument(
        "--attribution-split",
        default="valid",
        choices=("train", "valid", "test"),
        help="partition to measure attribution on (default: valid)",
    )
    attribution.add_argument(
        "--attribution-max-batches",
        type=int,
        default=0,
        help="cap batches per channel for permutation importance; 0 uses all",
    )
    attribution.add_argument(
        "--shap-samples", type=int, default=64, help="tiles to explain (default 64)"
    )
    attribution.add_argument(
        "--shap-baselines",
        type=int,
        default=16,
        help="path draws per tile (default 16); variance falls as 1/sqrt(n)",
    )

    flags = parser.add_argument_group("behaviour")
    flags.add_argument(
        "--deterministic",
        action="store_true",
        help="pin cuDNN to deterministic kernels (slower, bit-reproducible)",
    )
    flags.add_argument(
        "--no-amp", action="store_true", help="disable mixed precision"
    )
    flags.add_argument(
        "--torch-compile", action="store_true", help="compile the model before training"
    )
    flags.add_argument(
        "--single",
        action="store_true",
        help="force one run, ignoring the bootstrap and CV settings",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        pipeline_config = load_config(args.config)
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    config = training_config_from_pipeline(pipeline_config)

    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "patience": args.patience,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "device": args.device,
            "max_scenes": args.max_scenes,
            "bootstrap_n_seeds": args.n_seeds,
            "trials": args.trials,
            "norm_type": args.norm_type,
            "bottleneck_attention_heads": args.bottleneck_attention_heads,
            "train_root": Path(args.train_root) if args.train_root else None,
            "model_root": Path(args.out) if args.out else None,
        }.items()
        if value is not None
    }
    if args.deterministic:
        overrides["deterministic"] = True
    if args.no_amp:
        overrides["use_amp"] = False
    if args.torch_compile:
        overrides["torch_compile"] = True
    # Store-true flags can only turn a variant on, never off: absence means
    # "leave the configured value alone", not "disable".
    if args.bottleneck_attention:
        overrides["bottleneck_attention"] = True
    if args.deep_supervision:
        overrides["deep_supervision"] = True
    if args.single:
        overrides["bootstrap_enabled"] = False
        overrides["cv_enabled"] = False

    config = config.with_overrides(**overrides)

    print(f"[config] {args.config or 'configs/config.yaml'}")
    print(f"[config] corpus={config.train_root}")
    print(f"[config] output={config.model_root}")
    print(
        f"[config] strategy={config.strategy} "
        f"mono_common_mode={config.mono_common_mode_active}"
    )
    print(
        f"[config] patch={config.patch_size} stride={config.stride} "
        f"batch={config.batch_size} base_ch={config.base_channels}"
    )
    print(
        f"[config] lr={config.lr:.6g} weight_decay={config.weight_decay:.6g} "
        f"dropout={config.dropout_p:.4g}"
    )
    print(f"[config] epochs={config.epochs} patience={config.patience}")

    variants = [
        name
        for name, on in (
            (f"norm_type={config.norm_type}", config.norm_type != "batch"),
            (
                f"bottleneck_attention({config.bottleneck_attention_heads} heads)",
                config.bottleneck_attention,
            ),
            ("deep_supervision", config.deep_supervision),
        )
        if on
    ]
    print(
        "[config] architecture: "
        + (", ".join(variants) if variants else "reference (matches the published model)")
    )

    active = [
        f"{name}={weight:g}"
        for name, weight in vars(config.penalties).items()
        if weight > 0
    ]
    print(f"[config] active loss penalties: {', '.join(active) if active else 'none'}")

    try:
        data = prepare_data(config)
    except (RuntimeError, ValueError) as exc:
        print(f"[error] could not prepare the corpus: {exc}", file=sys.stderr)
        return 3

    print(
        f"[data] {len(data.scene_records)} scenes, "
        f"{len(set(data.glacier_ids))} glaciers, "
        f"tiles train/val/test = "
        f"{len(data.tiles_train)}/{len(data.tiles_val)}/{len(data.tiles_test)}"
    )
    print(f"[data] class pixel counts: {data.class_counts}")

    attribution_options = AttributionOptions(
        permutation=args.feature_importance,
        gradient=args.shap,
        split=args.attribution_split,
        max_batches=args.attribution_max_batches,
        seed=config.seed,
        shap_samples=args.shap_samples,
        shap_baselines=args.shap_baselines,
    )
    if attribution_options.any_enabled:
        enabled = ", ".join(
            name
            for name, on in (
                ("permutation", attribution_options.permutation),
                ("gradient", attribution_options.gradient),
            )
            if on
        )
        print(f"[config] attribution: {enabled} on {attribution_options.split}")

    if config.cv_enabled:
        run_root = run_cv(config, data=data)
    elif config.bootstrap_enabled:
        run_root = run_bootstrap(
            config, data=data, attribution_options=attribution_options
        )
    else:
        result, winning_config, _trials = run_trials(config, data=data)
        run_root = config.model_root
        importance, attribution = compute_attributions(
            result, winning_config, data, attribution_options
        )
        save_model(
            result,
            winning_config,
            run_root,
            feature_importance=importance,
            attribution=attribution,
        )
        print(
            f"[result] best epoch {result.best_epoch}: "
            f"val mIoU={result.best_state.get('val_miou_macro')} "
            f"kappa={result.best_state.get('val_kappa')}"
        )

    print(f"[done] {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
