#!/usr/bin/env python3
"""Predictive-entropy uncertainty analysis for any trained checkpoint.

Pure post-processing: loads a checkpoint's weights, runs a forward pass
over the test partition (reproduced via the same glacier-level split as
every other baseline script, not re-derived ad hoc), and reports the AUROC
of predictive entropy against pixel-level correctness. No retraining, no
gradient steps.

For every valid test-set pixel: softmax over the 4 classes, Shannon entropy
of that distribution, and whether the argmax prediction is correct. Pooled:
AUROC(score=entropy, label=is_error). This works for the published U-Net
(the reference checkpoint family) and equally for a confirmed DeepLab or
SegFormer checkpoint written by 16_/17_ -- anything whose checkpoint stores
`meta["architecture"]`, `meta["mean"]`, `meta["std"]`, `meta["features"]` in
the schema `glacier_fsnow_unet.training.export.checkpoint_metadata` writes.

Examples:

    python scripts/baselines/19_uncertainty_analysis.py --checkpoint _work/models/train/seed_0/model.pt --model-type unet
    python scripts/baselines/19_uncertainty_analysis.py --checkpoint _work/baselines/deeplab/deeplab_seed0_checkpoint.pt --model-type deeplab

`--model-type` picks which architecture builder reads the checkpoint's
state dict; `unet` uses `glacier_fsnow_unet.training.model_architectures
.build_model`, `deeplab`/`segformer` use this directory's own model
builders. The checkpoint's own recorded (mean, std) is always used for
normalisation, never recomputed, so a result is faithful to what the
weights were actually trained under.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """probs: (..., C) or (C, ...) -- pass axis explicitly at the call site."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=0)


def build_model_for_checkpoint(model_type: str, meta: dict):
    import torch.nn as nn

    if model_type == "unet":
        from glacier_fsnow_unet.training.model_architectures import build_model

        arch = meta["architecture"]
        return build_model(
            model_type="unet",
            in_channels=arch["in_channels"],
            num_classes=arch.get("num_classes", NUM_CLASSES),
            base_ch=arch["base_channels"],
            dropout_p=0.0,
            use_spatial_context=arch.get("use_spatial_context", False),
            num_sensors=arch.get("num_sensors", 1),
            use_sensor_film=arch.get("use_sensor_film", False),
            norm_type=arch.get("norm_type", "batch"),
            bottleneck_attention=arch.get("bottleneck_attention", False),
            bottleneck_attention_heads=arch.get("bottleneck_attention_heads", 8),
            deep_supervision=arch.get("deep_supervision", False),
            use_attention_gates=arch.get("use_attention_gates", True),
        )
    # DeepLab's dropout, when enabled, is spliced in as an extra Sequential
    # layer around the final conv -- it changes the state_dict's key names,
    # not just a probability, so the checkpoint's own dropout_p must be used
    # to rebuild a matching architecture (eval() disables the dropout itself
    # regardless of the value).
    if model_type == "deeplab":
        from glacier_fsnow_unet.training.deeplab_model import build_deeplabv3_glacier

        return build_deeplabv3_glacier(
            in_channels=len(meta["features"]), num_classes=NUM_CLASSES,
            dropout_p=meta.get("dropout_p", 0.1),
        )
    if model_type == "segformer":
        from glacier_fsnow_unet.training.segformer_model import build_segformer_glacier

        return build_segformer_glacier(
            in_channels=len(meta["features"]), num_classes=NUM_CLASSES,
            dropout_p=meta.get("dropout_p", 0.1),
        )
    raise ValueError(f"unknown --model-type {model_type!r} (expected unet, deeplab, or segformer)")


def forward_logits(model_type: str, model, x):
    if model_type == "deeplab":
        return model(x)["out"]
    return model(x)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml (default: configs/config.yaml)")
    parser.add_argument("--checkpoint", required=True, help="path to a model.pt checkpoint")
    parser.add_argument(
        "--model-type", default="unet", choices=("unet", "deeplab", "segformer"),
        help="architecture to rebuild the checkpoint's state dict into",
    )
    parser.add_argument("--train-root", default=None, help="corpus root (overrides config)")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument(
        "--no-require-cross-check", action="store_true",
        help="proceed even if no reference checkpoint is found for the split cross-check",
    )
    args = parser.parse_args()

    try:
        pipeline_cfg, default_train_root = load_baselines_config(args.config)
    except BaselinesConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"[error] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else REPO_ROOT / "_work" / "baselines" / "uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_root = Path(args.train_root) if args.train_root else default_train_root
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[config] checkpoint={checkpoint_path} model_type={args.model_type} device={device}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("model_state")
    if state_dict is None:
        print("[error] checkpoint has neither 'model' nor 'model_state'", file=sys.stderr)
        return 2

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    inv_std = 1.0 / (std + 1e-6)

    model = build_model_for_checkpoint(args.model_type, meta)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("[model] checkpoint loaded, strict state_dict match")

    check_checkpoint = (
        Path(pipeline_cfg.check_checkpoint) if pipeline_cfg.check_checkpoint else None
    )
    if check_checkpoint and not check_checkpoint.is_absolute():
        check_checkpoint = REPO_ROOT / check_checkpoint
    data, check = build_split(
        train_root=train_root, check_checkpoint=check_checkpoint,
        require_cross_check=not args.no_require_cross_check,
    )
    test_scene_idx = scene_indices_for_split(data, "test")
    print(f"[data] {len(test_scene_idx)} test scenes")

    patch = data.patch_size
    all_entropy, all_is_error, all_true = [], [], []
    per_scene_rows = []

    with torch.inference_mode():
        for si in test_scene_idx:
            label = data.labels[si]
            feat = data.features[si].astype(np.float32)
            h, w = label.shape
            normed = (feat - mean[:, None, None]) * inv_std[:, None, None]
            np.nan_to_num(normed, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            pad_h, pad_w = (-h) % patch, (-w) % patch
            if pad_h or pad_w:
                normed_p = np.zeros((normed.shape[0], h + pad_h, w + pad_w), dtype=np.float32)
                normed_p[:, :h, :w] = normed
                labels_p = np.full((h + pad_h, w + pad_w), IGNORE_INDEX, dtype=np.uint8)
                labels_p[:h, :w] = label
            else:
                normed_p, labels_p = normed, label

            hp, wp = normed_p.shape[1], normed_p.shape[2]
            scene_entropy, scene_is_error = [], []
            for ty in range(hp // patch):
                for tx in range(wp // patch):
                    y0, x0 = ty * patch, tx * patch
                    label_tile = labels_p[y0 : y0 + patch, x0 : x0 + patch]
                    valid = label_tile != IGNORE_INDEX
                    if not valid.any():
                        continue
                    feat_tile = normed_p[:, y0 : y0 + patch, x0 : x0 + patch]
                    x = torch.from_numpy(feat_tile).unsqueeze(0).to(device)
                    logits = forward_logits(args.model_type, model, x)
                    probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
                    pred = probs.argmax(axis=0).astype(np.uint8)
                    entropy = shannon_entropy(probs)

                    v_entropy = entropy[valid]
                    v_pred = pred[valid]
                    v_true = label_tile[valid]
                    v_is_error = (v_pred != v_true).astype(np.uint8)

                    all_entropy.append(v_entropy)
                    all_is_error.append(v_is_error)
                    all_true.append(v_true)
                    scene_entropy.append(v_entropy)
                    scene_is_error.append(v_is_error)

            if scene_entropy:
                ent = np.concatenate(scene_entropy)
                err = np.concatenate(scene_is_error)
                row = {"scene_index": si, "n_pixels": int(err.size), "error_rate": float(err.mean())}
                if err.size >= 200 and 0 < err.sum() < err.size:
                    row["auroc"] = float(roc_auc_score(err, ent))
                else:
                    row["auroc"] = ""
                per_scene_rows.append(row)

    entropy_all = np.concatenate(all_entropy)
    is_error_all = np.concatenate(all_is_error)
    true_all = np.concatenate(all_true)

    error_rate = float(is_error_all.mean())
    auroc = float(roc_auc_score(is_error_all, entropy_all))
    print(f"[result] pooled test-pixel error rate: {error_rate:.4f}")
    print(f"[result] AUROC(entropy, is_error) = {auroc:.4f}")

    per_class_auroc = {}
    class_names = meta.get("class_names", ["Cloud", "Snow", "Ice", "Other"])
    for c in range(NUM_CLASSES):
        mask = true_all == c
        if mask.sum() < 20 or is_error_all[mask].sum() == 0 or is_error_all[mask].sum() == mask.sum():
            per_class_auroc[class_names[c]] = None
            continue
        per_class_auroc[class_names[c]] = float(roc_auc_score(is_error_all[mask], entropy_all[mask]))
    print(f"[result] per-class AUROC: {per_class_auroc}")

    mean_entropy_correct = float(entropy_all[is_error_all == 0].mean())
    mean_entropy_incorrect = float(entropy_all[is_error_all == 1].mean())
    print(f"[result] mean entropy | correct={mean_entropy_correct:.4f} | incorrect={mean_entropy_incorrect:.4f}")

    results = {
        "checkpoint": str(checkpoint_path),
        "model_type": args.model_type,
        "n_test_scenes": len(test_scene_idx),
        "n_pixels_evaluated": int(is_error_all.size),
        "pooled_error_rate": error_rate,
        "auroc_entropy_vs_error": auroc,
        "per_class_auroc": per_class_auroc,
        "mean_entropy_correct": mean_entropy_correct,
        "mean_entropy_incorrect": mean_entropy_incorrect,
        "max_possible_entropy_4class_nats": float(np.log(NUM_CLASSES)),
        "split_cross_check": check,
    }
    with open(out_dir / "uncertainty_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    import csv

    with open(out_dir / "per_scene_auroc.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_index", "n_pixels", "error_rate", "auroc"])
        writer.writeheader()
        writer.writerows(per_scene_rows)

    print(f"[done] {out_dir / 'uncertainty_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
