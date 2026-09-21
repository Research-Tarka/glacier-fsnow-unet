#!/usr/bin/env python3
"""Stage 7 -- run U-Net inference over downloaded scenes.

Purpose
-------
Apply the trained four-class U-Net to each scene's 11-channel spectral-index
stack, producing a per-scene class map (Cloud/Snow/Ice/Other), and write it
into the glacier's zarr store.

Inputs
------
- Per-glacier scene data from stage 4 (``<glacier_dir>/<glims_id>.zarr``).
- A trained model under ``config.paths.model_inference_root``.
- Optionally ``ambiguity_threshold.json`` beside that checkpoint, written by
  stage 6. When present, pixels whose top-two probability gap falls below the
  calibrated threshold are assigned by the Cloud > Snow > Ice > Other priority
  order instead of by argmax; when absent, the argmax stands untouched.

Outputs
-------
- ``inference``/``inference_done`` arrays per sensor group, in the same
  ``<glims_id>.zarr`` store (see ``glacier_fsnow_unet.inference.zarr_store``).

Usage
-----
    python scripts/07_run_inference.py --config configs/config.yaml
    python scripts/07_run_inference.py --check-model
    python scripts/07_run_inference.py --split 1 --only-id G214501E60996N
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.inference.engine import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_PATCH_SIZE,
    DEFAULT_STRIDE,
    resolve_ambiguity_threshold,
    run_inference_for_glacier,
)
from glacier_fsnow_unet.inference.model_loader import (  # noqa: E402
    ModelNotAvailableError,
    load_model,
)
from glacier_fsnow_unet.scenes.zarr_store import zarr_path_for_glacier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run U-Net inference over downloaded scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--model", default=None, help="Path to model.pt")
    parser.add_argument("--split", default="all", help="Split number, or 'all' (default)")
    parser.add_argument(
        "--registry", default=None, help="Path to glacier_registry.parquet"
    )
    parser.add_argument(
        "--glacier-root",
        default=None,
        help="Root of the per-glacier directory tree (default: the split's root)",
    )
    parser.add_argument("--only-id", default=None, help="Process a single glacier id")
    parser.add_argument("--limit", type=int, default=0, help="Max glaciers (0 = all)")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="Sliding-window stride (default: same as training stride)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument(
        "--ambiguity-threshold",
        type=float,
        default=None,
        help="Confidence gap below which the Cloud > Snow > Ice > Other "
             "priority rule replaces the argmax. Default: read from "
             "ambiguity_threshold.json beside the checkpoint if present, else "
             "0 (plain argmax). Pass 0 to force plain argmax.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-model",
        action="store_true",
        help="Report whether a usable trained model is available, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    device = args.device or cfg.model.device
    model_root = args.model or cfg.paths.model_inference_root

    print("=" * 78)
    print("Stage 7: U-Net inference")
    print("=" * 78)
    print(f"  model root: {model_root}")
    print(f"  patch={args.patch_size} stride={args.stride} "
          f"batch={args.batch_size} device={device}")

    if args.ambiguity_threshold is not None:
        ambiguity_threshold = max(0.0, float(args.ambiguity_threshold))
        source = "--ambiguity-threshold"
    else:
        ambiguity_threshold = resolve_ambiguity_threshold(model_root)
        source = "ambiguity_threshold.json" if ambiguity_threshold else "no calibration"
    if ambiguity_threshold > 0:
        print(f"  ambiguity rule: gap < {ambiguity_threshold:.4f} resolved by "
              f"Cloud > Snow > Ice > Other  (from {source})")
    else:
        print(f"  ambiguity rule: off, plain argmax  ({source}); run "
              f"scripts/06_interpret_model.py to calibrate one")

    try:
        if Path(str(model_root)).is_file():
            model = load_model(checkpoint_path=model_root, device=device,
                               base_channels=cfg.model.base_channels)
        else:
            model = load_model(model_root=model_root, device=device,
                               base_channels=cfg.model.base_channels)
    except ModelNotAvailableError as exc:
        print(f"\n[BLOCKED] {exc}\n", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Could not load the model: {exc}\n", file=sys.stderr)
        return 1

    if args.check_model:
        print("  [OK] A usable trained model was loaded.")
        return 0

    registry_path = Path(
        args.registry or (Path(cfg.isolated_glacier.output_json).parent / REGISTRY_FILENAME)
    )
    try:
        registry = read_registry(registry_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    sd = cfg.scene_download
    split_indices = (
        list(range(1, len(sd.splits) + 1))
        if str(args.split).strip().lower() == "all"
        else [int(args.split)]
    )

    assignment_path = registry_path.parent / "split_assignment.parquet"
    assignment = None
    if assignment_path.is_file():
        import pandas as pd

        assignment = pd.read_parquet(assignment_path)

    totals = {"processed": 0, "skipped_precheck": 0, "already_done": 0}
    n_glaciers = 0

    for split_num in split_indices:
        if args.glacier_root:
            glacier_root = Path(args.glacier_root)
        else:
            glacier_root = Path(sd.splits[split_num - 1].root)

        if assignment is not None:
            ids = set(
                assignment.loc[
                    assignment["split_index"] == (split_num - 1), "glims_id"
                ].astype(str)
            )
            subset = registry[registry["glims_id"].astype(str).isin(ids)]
        else:
            subset = registry

        if args.only_id:
            subset = subset[subset["glims_id"].astype(str) == args.only_id]
        if args.limit:
            subset = subset.head(max(0, args.limit - n_glaciers))

        print(f"\n-- Split {split_num}: {len(subset)} glacier(s) -> {glacier_root}")

        for glims_id in subset["glims_id"].astype(str):
            glacier_dir = glacier_root / glims_id
            zarr_path = zarr_path_for_glacier(glacier_dir)
            if not zarr_path.exists():
                print(f"   [skip] {glims_id}: no zarr store at {zarr_path}")
                continue

            counters = run_inference_for_glacier(
                zarr_path,
                model,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                min_valid_ratio=cfg.inference.scene_precheck_min_valid_ratio,
                overwrite=args.overwrite,
                ambiguity_threshold=ambiguity_threshold,
            )
            for key in totals:
                totals[key] += counters[key]
            n_glaciers += 1
            print(f"   [{glims_id}] processed={counters['processed']} "
                  f"skipped={counters['skipped_precheck']} "
                  f"already_done={counters['already_done']}")

            if args.limit and n_glaciers >= args.limit:
                break
        if args.limit and n_glaciers >= args.limit:
            break

    print(f"\n[DONE] glaciers={n_glaciers} processed={totals['processed']} "
          f"skipped={totals['skipped_precheck']} already_done={totals['already_done']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
