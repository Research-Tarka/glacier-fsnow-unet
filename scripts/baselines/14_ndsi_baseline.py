#!/usr/bin/env python3
"""NDSI/NDVI-threshold baseline: the simplest possible 4-class classifier.

A strict depth-2 decision tree using only 2 of the 11 spectral indices
(NDSI, NDVI), each compared against one threshold per branch -- 3 free
scalar parameters total. Thresholds are chosen by a coarse-then-fine grid
search on the TRAIN partition (maximizing macro mIoU, pooled confusion
matrix), then reported on TRAIN/VALID/TEST using the exact glacier-level
split the U-Net was trained on (cross-checked scene-by-scene against a real
trained checkpoint before anything is reported).

Decision hierarchy:
  1. NDSI >= t_snowice  -> candidate {Snow, Ice}; else candidate {Cloud, Other}.
  2. Within {Snow, Ice}: NDVI < t_ice -> Ice, else Snow.
  3. Within {Cloud, Other}: NDVI > t_other -> Other, else Cloud.

Run with no arguments to use configs/config.yaml's `baselines.ndsi` section
(test mIoU 0.402 at the default grid resolution); see that file for every
adjustable parameter (grid step sizes, output directory).

Examples:

    python scripts/baselines/14_ndsi_baseline.py

    # Finer grid, different output directory, without touching the YAML.
    python scripts/baselines/14_ndsi_baseline.py --fine-step 0.005 --out _work/ndsi_v2

No brightness/reflectance-magnitude band is available in this corpus's
feature cache (only normalized-difference indices and relative-chromaticity
fractions), so classic brightness-based cloud detection is not reproducible
baseline-only from this feature set -- a deliberate methodological
limitation, not an oversight.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numba
import numpy as np
from numba import prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import (  # noqa: E402
    BaselinesConfigError,
    build_split,
    load_baselines_config,
    scene_indices_for_split,
)
from glacier_fsnow_unet.training.config import IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from glacier_fsnow_unet.training.metrics import compute_confusion_metrics  # noqa: E402

# Class indices, matching glacier_fsnow_unet.training.config.
CLOUD, SNOW, ICE, OTHER = 0, 1, 2, 3
NDVI_CHANNEL = 0  # Index_NDVI
NDSI_CHANNEL = 1  # Index_NDSI


@dataclass(frozen=True)
class ThresholdParams:
    t_snowice: float  # NDSI split: >= t_snowice -> {Snow, Ice} branch
    t_ice: float  # NDVI split within {Snow, Ice}: < t_ice -> Ice, else Snow
    t_other: float  # NDVI split within {Cloud, Other}: > t_other -> Other, else Cloud


def classify(features: np.ndarray, params: ThresholdParams) -> np.ndarray:
    """Classify a `(11, H, W)` or `(11, N)` feature array into 0..3 labels."""
    ndvi = features[NDVI_CHANNEL].astype(np.float32)
    ndsi = features[NDSI_CHANNEL].astype(np.float32)

    snowice_branch = ndsi >= params.t_snowice
    out = np.empty(ndvi.shape, dtype=np.uint8)
    out[snowice_branch & (ndvi < params.t_ice)] = ICE
    out[snowice_branch & (ndvi >= params.t_ice)] = SNOW
    out[(~snowice_branch) & (ndvi > params.t_other)] = OTHER
    out[(~snowice_branch) & (ndvi <= params.t_other)] = CLOUD
    return out


def confusion_for_params(data, scene_indices: list[int], params: ThresholdParams) -> np.ndarray:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for si in scene_indices:
        label = data.labels[si]
        valid = label != IGNORE_INDEX
        if not np.any(valid):
            continue
        pred = classify(data.features[si], params)
        true_flat = label[valid].astype(np.int64)
        pred_flat = pred[valid].astype(np.int64)
        idx = true_flat * NUM_CLASSES + pred_flat
        confusion += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
    return confusion


@numba.njit(parallel=True, cache=True)
def _confusions_for_grid(
    ndvi: np.ndarray,
    ndsi: np.ndarray,
    label: np.ndarray,
    t_snowice_grid: np.ndarray,
    t_ice_grid: np.ndarray,
    t_other_grid: np.ndarray,
) -> np.ndarray:
    """Confusion matrix for every (t_snowice, t_ice, t_other) combo, in one
    parallel pass over pixels per combo -- avoids materialising a boolean
    mask array per combo, which is what made the pure-numpy grid search
    memory-bandwidth bound at a few thousand combos."""
    n_a, n_b, n_c = t_snowice_grid.size, t_ice_grid.size, t_other_grid.size
    n_pixels = ndvi.size
    out = np.zeros((n_a, n_b, n_c, NUM_CLASSES * NUM_CLASSES), dtype=np.int64)

    for ia in prange(n_a):
        t_snowice = t_snowice_grid[ia]
        local = np.zeros((n_b, n_c, NUM_CLASSES * NUM_CLASSES), dtype=np.int64)
        for p in range(n_pixels):
            snowice = ndsi[p] >= t_snowice
            v = ndvi[p]
            lab = label[p]
            for ib in range(n_b):
                if snowice:
                    pred = ICE if v < t_ice_grid[ib] else SNOW
                    for ic in range(n_c):
                        local[ib, ic, lab * NUM_CLASSES + pred] += 1
                else:
                    for ic in range(n_c):
                        pred = OTHER if v > t_other_grid[ic] else CLOUD
                        local[ib, ic, lab * NUM_CLASSES + pred] += 1
        out[ia] = local
    return out


def grid_search_thresholds(
    data,
    train_scene_indices: list[int],
    t_snowice_grid: np.ndarray,
    t_ice_grid: np.ndarray,
    t_other_grid: np.ndarray,
) -> tuple[ThresholdParams, float, list[dict]]:
    """Exhaustive grid search over the 3 thresholds, maximizing TRAIN macro mIoU.

    Pre-extracts (NDVI, NDSI, label) for every valid train pixel once, then
    the numba-compiled kernel above sweeps every threshold combo in one
    pass, parallelised over the outer (t_snowice) axis.
    """
    ndvi_all, ndsi_all, label_all = [], [], []
    for si in train_scene_indices:
        label = data.labels[si]
        valid = label != IGNORE_INDEX
        if not np.any(valid):
            continue
        feat = data.features[si].astype(np.float32)
        ndvi_all.append(feat[0][valid])
        ndsi_all.append(feat[1][valid])
        label_all.append(label[valid].astype(np.int64))

    ndvi = np.concatenate(ndvi_all)
    ndsi = np.concatenate(ndsi_all)
    label = np.concatenate(label_all)
    print(f"[grid_search] {ndvi.size:,} labeled train pixels")

    t_snowice_grid = t_snowice_grid.astype(np.float32)
    t_ice_grid = t_ice_grid.astype(np.float32)
    t_other_grid = t_other_grid.astype(np.float32)

    t0 = time.time()
    confusions = _confusions_for_grid(ndvi, ndsi, label, t_snowice_grid, t_ice_grid, t_other_grid)
    total = t_snowice_grid.size * t_ice_grid.size * t_other_grid.size
    elapsed = time.time() - t0
    print(f"[grid_search] evaluated {total} combos in {elapsed:.1f}s")

    best_score = -1.0
    best_params = None
    history = []
    for ia, t_snowice in enumerate(t_snowice_grid):
        for ib, t_ice in enumerate(t_ice_grid):
            for ic, t_other in enumerate(t_other_grid):
                confusion = confusions[ia, ib, ic].reshape(NUM_CLASSES, NUM_CLASSES)
                score = float(compute_confusion_metrics(confusion)["macro"]["miou"])
                if score > best_score:
                    best_score = score
                    best_params = ThresholdParams(float(t_snowice), float(t_ice), float(t_other))
                history.append(
                    {
                        "t_snowice": float(t_snowice), "t_ice": float(t_ice),
                        "t_other": float(t_other), "train_miou_macro": score,
                    }
                )
    return best_params, best_score, history


def full_report(data, split_name: str, params: ThresholdParams) -> dict:
    scene_idx = scene_indices_for_split(data, split_name)
    confusion = confusion_for_params(data, scene_idx, params)
    metrics = compute_confusion_metrics(confusion)
    return {
        "split": split_name, "n_scenes": len(scene_idx),
        "confusion_matrix": confusion.tolist(),
        "per_class": metrics["per_class"], "macro": metrics["macro"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="path to config.yaml (default: configs/config.yaml, matching "
        "05_train_model.py's resolution order)",
    )
    parser.add_argument("--out", default=None, help="output directory (overrides config)")
    parser.add_argument("--train-root", default=None, help="corpus root (overrides config)")
    parser.add_argument("--coarse-step", type=float, default=None)
    parser.add_argument("--fine-step", type=float, default=None)
    parser.add_argument(
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

    out_dir = Path(args.out) if args.out else REPO_ROOT / cfg.ndsi.out_dir
    train_root = Path(args.train_root) if args.train_root else default_train_root
    coarse_step = args.coarse_step if args.coarse_step is not None else cfg.ndsi.coarse_step
    fine_step = args.fine_step if args.fine_step is not None else cfg.ndsi.fine_step

    print(f"[config] {args.config or 'configs/config.yaml'}")
    print(f"[config] corpus={train_root}")
    print(f"[config] output={out_dir}")
    print(f"[config] coarse_step={coarse_step} fine_step={fine_step}")

    out_dir.mkdir(parents=True, exist_ok=True)

    check_checkpoint = Path(cfg.check_checkpoint) if cfg.check_checkpoint else None
    if check_checkpoint and not check_checkpoint.is_absolute():
        check_checkpoint = REPO_ROOT / check_checkpoint
    data, check = build_split(
        train_root=train_root,
        check_checkpoint=check_checkpoint,
        require_cross_check=not args.no_require_cross_check,
    )

    train_idx = scene_indices_for_split(data, "train")
    print(f"[data] train/valid/test scenes = "
          f"{len(train_idx)}/{len(scene_indices_for_split(data, 'valid'))}/"
          f"{len(scene_indices_for_split(data, 'test'))}")

    print("[main] coarse grid search on TRAIN ...")
    coarse_snowice = np.arange(-0.2, 0.81, coarse_step)
    coarse_ice = np.arange(-0.3, 0.31, coarse_step)
    coarse_other = np.arange(-0.2, 0.41, coarse_step)
    best_params, best_score, coarse_history = grid_search_thresholds(
        data, train_idx, coarse_snowice, coarse_ice, coarse_other
    )
    print(f"[main] coarse best: {best_params} train_miou_macro={best_score:.4f}")

    print("[main] fine grid search around coarse optimum ...")
    fine_snowice = np.arange(best_params.t_snowice - 0.05, best_params.t_snowice + 0.051, fine_step)
    fine_ice = np.arange(best_params.t_ice - 0.05, best_params.t_ice + 0.051, fine_step)
    fine_other = np.arange(best_params.t_other - 0.05, best_params.t_other + 0.051, fine_step)
    best_params_fine, best_score_fine, fine_history = grid_search_thresholds(
        data, train_idx, fine_snowice, fine_ice, fine_other
    )
    if best_score_fine > best_score:
        best_params, best_score = best_params_fine, best_score_fine
    print(f"[main] fine best: {best_params} train_miou_macro={best_score:.4f}")

    results = {}
    for split_name in ("train", "valid", "test"):
        report = full_report(data, split_name, best_params)
        results[split_name] = report
        print(
            f"[result] {split_name}: miou_macro={report['macro']['miou']} "
            f"kappa={report['macro']['kappa']} mcc={report['macro']['mcc']}"
        )
        for pc in report["per_class"]:
            print(f"    {pc['class_name']:6s} IoU={pc['iou']}")

    output = {
        "thresholds": {
            "t_snowice_NDSI": best_params.t_snowice,
            "t_ice_NDVI": best_params.t_ice,
            "t_other_NDVI": best_params.t_other,
        },
        "train_miou_macro_at_selection": best_score,
        "split_cross_check": check,
        "results": results,
    }
    with open(out_dir / "ndsi_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_dir / "grid_search_history.json", "w", encoding="utf-8") as f:
        json.dump({"coarse": coarse_history, "fine": fine_history}, f)

    print(f"[done] {out_dir / 'ndsi_baseline_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
