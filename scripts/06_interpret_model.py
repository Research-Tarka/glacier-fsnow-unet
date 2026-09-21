#!/usr/bin/env python3
"""Stage 6 -- interpret a trained U-Net: ambiguity, importance, SHAP, Grad-CAM.

Purpose
-------
Run four independent analyses against a finished checkpoint in a single pass
over the corpus:

1. **Ambiguity calibration** -- the confidence gap below which the
   Cloud > Snow > Ice > Other priority rule should override a plain argmax,
   calibrated so macro mIoU degrades by no more than a configured tolerance.
   Writes ``ambiguity_threshold.json``. ``scripts/07_run_inference.py`` reads
   that file when it sits beside the checkpoint; put it there with
   ``--install-ambiguity-threshold``, which is opt-in so that a calibration run
   does not silently change every subsequent production inference.
2. **Permutation feature importance** -- IoU lost, per band and per class, when
   one input channel is shuffled.
3. **SHAP band attribution** -- expected-gradient (Shapley) attribution per band
   and per class.
4. **Grad-CAM** -- per-class spatial localisation maps.

This stage is analyst-run. Nothing later in the pipeline consumes its outputs
except the ambiguity threshold, so the orchestrator does not include it in a
plain ``--from``/``--to`` range; select it with ``--only 6``.

Switching analyses on and off
-----------------------------
Each analysis has its own ``enabled`` flag and its own settings under
``unet_interpretation`` in the config, and is skipped with an explicit log line
when disabled. ``--only`` narrows further at the command line but cannot enable
what the config has switched off, so the config stays the single record of what
a run was permitted to do.

Inputs
------
- A trained checkpoint under ``config.paths.model_inference_root``.
- The annotated corpus under ``config.paths.train_root`` (override with
  ``--train-root``), the same corpus training reads.

Outputs
-------
Under ``<output>/<unet_interpretation.output_dirname>/``:
``ambiguity_threshold.json``, ``ambiguity_threshold.html``,
``permutation_importance.csv``,
``shap_band_attribution.csv``, ``shap_global_bands.html``,
``shap_per_class_bands.html``, ``gradcam.zarr``, ``gradcam_heatmaps.html``,
and ``interpretation_summary.json`` combining every analysis.

Exit codes
----------
- 0: every enabled analysis ran (or every analysis was disabled).
- 1: an analysis failed, or the corpus could not be prepared.
- 2: bad configuration or invocation.
- 4: no usable trained model.

Usage
-----
    python scripts/06_interpret_model.py --config configs/config.yaml
    python scripts/06_interpret_model.py --only ambiguity,shap
    python scripts/06_interpret_model.py --skip-gradcam --max-scenes 40
    python scripts/06_interpret_model.py --train-root D:/corpus --out _work/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.inference.model_loader import (  # noqa: E402
    ModelNotAvailableError,
    load_model,
)
from glacier_fsnow_unet.interpretation.runner import (  # noqa: E402
    ANALYSIS_NAMES,
    run_interpretation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--env-file", default=None, help="path to a .env file")
    parser.add_argument(
        "--model",
        default=None,
        help="checkpoint file or directory (overrides paths.model_inference_root)",
    )
    parser.add_argument(
        "--train-root",
        default=None,
        help="annotated corpus root (overrides paths.train_root)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output root; the interpretation directory is created inside it "
        "(default: paths.output_root)",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="use only the first N corpus scenes, for a quick check",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="seed for the train/valid/test partition; must match the seed the "
        "checkpoint was trained under for the partition to mean anything",
    )

    selection = parser.add_argument_group(
        "analysis selection",
        "The config's per-analysis `enabled` flags decide what may run. These "
        "flags can narrow that further, never widen it.",
    )
    selection.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of: " + ", ".join(ANALYSIS_NAMES),
    )
    for name in ANALYSIS_NAMES:
        selection.add_argument(
            f"--skip-{name.replace('_', '-')}",
            dest=f"skip_{name}",
            action="store_true",
            help=f"skip the {name.replace('_', ' ')} analysis",
        )
    selection.add_argument(
        "--no-html",
        action="store_true",
        help="write the data outputs (JSON, CSV, zarr) but no HTML reports",
    )
    parser.add_argument(
        "--install-ambiguity-threshold",
        action="store_true",
        help="also copy ambiguity_threshold.json beside the checkpoint, which "
             "is what makes inference start using it. Off by default: the "
             "checkpoint directory is often a published artifact, and a "
             "calibration run should not silently change every subsequent "
             "production inference.",
    )
    return parser


def _resolve_selection(args: argparse.Namespace) -> set[str] | None:
    """Turn --only / --skip-* into a set of analysis names, or None for 'all'."""
    if args.only:
        requested = {part.strip() for part in args.only.split(",") if part.strip()}
        unknown = requested - set(ANALYSIS_NAMES)
        if unknown:
            raise SystemExit(
                f"[error] --only: unknown analysis {sorted(unknown)}; "
                f"choose from {list(ANALYSIS_NAMES)}"
            )
    else:
        requested = set(ANALYSIS_NAMES)

    skipped = {name for name in ANALYSIS_NAMES if getattr(args, f"skip_{name}")}
    selected = requested - skipped
    if selected == set(ANALYSIS_NAMES):
        return None
    return selected


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection = _resolve_selection(args)

    try:
        pipeline_config = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    from glacier_fsnow_unet.training.config import training_config_from_pipeline
    from glacier_fsnow_unet.training.dataset import prepare_data
    from glacier_fsnow_unet.training.train import resolve_device

    training_config = training_config_from_pipeline(pipeline_config)
    overrides = {
        key: value
        for key, value in {
            "device": args.device,
            "max_scenes": args.max_scenes,
            "split_seed": args.split_seed,
            "train_root": Path(args.train_root) if args.train_root else None,
        }.items()
        if value is not None
    }
    if overrides:
        training_config = training_config.with_overrides(**overrides)

    interpretation = pipeline_config.unet_interpretation
    output_root = Path(args.out or pipeline_config.paths.output_root)
    output_dir = output_root / interpretation.output_dirname
    model_root = args.model or pipeline_config.paths.model_inference_root
    device = resolve_device(training_config.device)

    print("=" * 78)
    print("Stage 6: U-Net interpretation")
    print("=" * 78)
    print(f"  model:  {model_root}")
    print(f"  corpus: {training_config.train_root}")
    print(f"  output: {output_dir}")
    print(f"  device: {device}")
    enabled = [
        name
        for name in ANALYSIS_NAMES
        if getattr(interpretation, name).enabled
        and (selection is None or name in selection)
    ]
    print(f"  analyses: {', '.join(enabled) if enabled else 'none'}")

    if not enabled:
        print(
            "\n[done] Every analysis is disabled; nothing to do. Enable one "
            "under `unet_interpretation` in the config."
        )
        return 0

    try:
        if Path(str(model_root)).is_file():
            model = load_model(
                checkpoint_path=model_root,
                device=str(device),
                base_channels=pipeline_config.model.base_channels,
            )
        else:
            model = load_model(
                model_root=model_root,
                device=str(device),
                base_channels=pipeline_config.model.base_channels,
            )
    except ModelNotAvailableError as exc:
        print(f"\n[BLOCKED] {exc}\n", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        print(f"\n[error] could not load the model: {exc}\n", file=sys.stderr)
        return 1

    try:
        data = prepare_data(training_config)
    except (RuntimeError, ValueError) as exc:
        print(f"\n[error] could not prepare the corpus: {exc}\n", file=sys.stderr)
        return 1

    print(
        f"  corpus: {len(data.scene_records)} scenes, "
        f"tiles train/valid/test = "
        f"{len(data.tiles_train)}/{len(data.tiles_val)}/{len(data.tiles_test)}"
    )

    outcome = run_interpretation(
        model,
        data,
        training_config,
        interpretation,
        device,
        output_dir,
        model_path=str(model_root),
        only=selection,
        write_html=not args.no_html,
        install_ambiguity_threshold=args.install_ambiguity_threshold,
    )

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    for name in ANALYSIS_NAMES:
        if name in outcome.summary["analyses"]:
            status = f"  OK   ({outcome.summary['analyses'][name]['duration_s']:.1f}s)"
        elif name in outcome.failed:
            status = f"FAILED  {outcome.failed[name]}"
        else:
            status = " SKIP "
        print(f"  [{status}] {name}")

    print(f"\n  {len(outcome.written)} file(s) written under {output_dir}")
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
