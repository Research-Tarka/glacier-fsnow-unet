"""The training loop.

`train_unified` runs one training job end to end: build the model, iterate
epochs of mixed-precision optimisation, evaluate on validation each epoch,
early-stop on the configured mIoU, and return the best model with its metrics.

Three things in here are worth reading before changing anything.

**Best-epoch snapshots are deep copies.** `Module.state_dict()` returns
references to the live parameter tensors, not copies. Snapshotting it and
continuing to train silently overwrites the snapshot in place, so the "best"
weights become whatever the last epoch produced and the final restore is a
no-op. This loop copies to CPU at snapshot time, which fixes that and keeps a
second copy of the weights out of VRAM. `keep_last_epoch_weights` reproduces
the original behaviour for checkpoint comparison.

**The early-stopping comparison is on a value rounded to three decimals.**
That rounding happens inside the metric computation and is preserved
deliberately — it makes patience considerably tighter than it looks, and
removing it changes where runs converge. See
`docs/decisions/training_reproducibility_audit.md`.

**Memory.** Mixed precision is on by default on CUDA, which is what lets
batch_size 96 at 48x48 fit in about 6 GB. Confusion matrices accumulate on the
GPU and transfer one small `C x C` matrix per batch instead of whole
prediction volumes. Validation runs under `inference_mode`.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .breakdown import (
    SceneConfusions,
    SceneIdentity,
    accumulate_scene_confusions,
    build_breakdown,
    scene_confusion_stack,
)
from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, TrainingConfig
from .dataset import PreparedData, prepare_data
from .losses import (
    compute_class_weights,
    deep_supervision_loss,
    total_penalty,
    weighted_cross_entropy,
)
from .metrics import (
    CalibrationTally,
    binary_glacier_iou,
    compute_boundary_metrics,
    compute_confusion_metrics,
    confusion_from_predictions,
    select_main_iou_metric,
)
from .model_architectures import build_model
from .torch_dataset import (
    TileDataset,
    GlacierBalancedSampler,
    SensorBalancedBatchSampler,
    build_glacier_tile_index,
    build_sensor_tile_index,
    worker_init_fn,
)

__all__ = ["TrainingResult", "seed_everything", "resolve_device", "train_unified"]


@dataclass
class TrainingResult:
    """A finished training run: the model plus everything worth recording."""

    model: nn.Module
    best_state: dict[str, Any]
    history: dict[str, list[float]]
    class_counts: dict[int, int]
    mean: np.ndarray
    std: np.ndarray
    features: tuple[str, ...]
    best_epoch: int
    stopped_epoch: int
    confusions: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trial_params: dict[str, Any] = field(default_factory=dict)
    # Per-scene confusion matrices for every partition, and the tables derived
    # from them. Populated by the final evaluation pass, not per epoch.
    scene_confusions: Optional[SceneConfusions] = None
    breakdown: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    boundary_confusions: dict[str, np.ndarray] = field(default_factory=dict)
    split_counts: dict[str, Any] = field(default_factory=dict)
    # Per-split reliability tables, from the same final pass as the confusion
    # matrices, so a calibration number and a score always describe the same
    # weights on the same pixels.
    calibration: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that can affect a run, before anything consumes one.

    Covers Python's `random`, NumPy, torch (CPU and all CUDA devices), and sets
    `PYTHONHASHSEED` for child processes.

    A caveat on `PYTHONHASHSEED`: setting it here does not change the *current*
    interpreter's hash seed, which was fixed at startup. Nothing in this
    pipeline depends on it — split determinism comes from sorting, not from
    hash stability — so this only makes spawned workers consistent. Relying on
    hash order was the original bug; see the audit document.

    With `deterministic`, also pins cuDNN to deterministic algorithms and turns
    off its autotuner. That costs throughput, so it is opt-in: benchmarking
    picks different kernels per run, which changes floating-point reduction
    order and makes runs non-bit-reproducible. TF32 matmul/convolution is
    subject to the same trade-off (it trades mantissa bits for throughput on
    Ampere-and-later GPUs) and is tied to the same flag.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve the training device, falling back to CPU with a warning.

    Silently running on CPU when CUDA was asked for turns a 20-minute run into
    an overnight one, so the fallback is announced.
    """
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[train] CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of the model's parameters and buffers.

    `copy=True` is the whole point: without it this aliases the live tensors
    and the snapshot is destroyed by the next optimiser step. Copying to CPU
    rather than cloning on-device keeps a full second set of weights out of
    VRAM, which matters on an 8 GB card.
    """
    return {
        key: value.detach().to("cpu", copy=True)
        for key, value in model.state_dict().items()
    }


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the original module behind a `torch.compile` wrapper.

    Compiled models prefix every `state_dict` key with `_orig_mod.`, which
    makes their checkpoints unloadable into an uncompiled model. Unwrapping
    before saving keeps checkpoints portable.
    """
    return getattr(model, "_orig_mod", model)


def _build_loaders(
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, TileDataset, TileDataset]:
    """Construct the training and validation loaders.

    `persistent_workers` is only valid with workers, and `prefetch_factor` only
    applies to them, so both are conditional. Pinned memory is enabled only on
    CUDA, where it actually buys an async host-to-device copy.
    """
    use_boundary = config.ignore_boundary > 0 and config.compute_boundary_metrics
    _, sensor_balanced, _ = config.sensor_adaptation

    common = dict(
        features=data.features,
        labels=data.labels,
        mean=data.mean,
        std=data.std,
        patch_size=config.patch_size,
        sensor_norm_stats=data.sensor_norm_stats,
        scene_sensors=data.scene_sensors,
        scene_context=data.scene_context,
    )

    train_dataset = TileDataset(
        data.tiles_train, return_tile_meta=use_boundary, **common
    )
    val_dataset = TileDataset(
        data.tiles_val,
        return_tile_meta=use_boundary,
        return_scene_index=not use_boundary,
        **common,
    )

    loader_kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["worker_init_fn"] = worker_init_fn

    batch_sampler = None
    if sensor_balanced:
        sensor_index = build_sensor_tile_index(data.tiles_train, data.scene_sensors)
        if len(sensor_index) > 1:
            batch_sampler = SensorBalancedBatchSampler(
                sensor_index,
                batch_size=config.batch_size,
                drop_last=False,
                seed=config.sampling_seed if config.sampling_seed is not None else config.seed,
            )
        else:
            print("[train] sensor-balanced batching requested but only one sensor present")

    if batch_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **loader_kwargs)
    elif config.glacier_balanced_sampling:
        sampler = GlacierBalancedSampler(
            build_glacier_tile_index(data.tiles_train, data.glacier_ids),
            num_samples=len(data.tiles_train),
            seed=config.sampling_seed if config.sampling_seed is not None else config.seed,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler, **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, train_dataset, val_dataset


def _unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Normalise the several batch shapes the datasets can produce."""
    if len(batch) >= 7:
        x, y, context, scene_index = batch[0], batch[1], batch[2], batch[3]
    elif len(batch) == 4:
        x, y, context, scene_index = batch
    elif len(batch) == 3:
        x, y, context = batch
        scene_index = None
    else:
        x, y = batch
        context = scene_index = None
    return x, y, context, scene_index


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    use_amp: bool,
    use_context: bool,
    num_classes: int = NUM_CLASSES,
    scene_stack: Optional[torch.Tensor] = None,
    calibration: Optional[CalibrationTally] = None,
) -> tuple[float, np.ndarray]:
    """Validation pass: mean loss and the pooled confusion matrix.

    `inference_mode` rather than `no_grad` — it additionally skips autograd's
    version-counter bookkeeping.

    The confusion matrix accumulates on-device and comes back once at the end,
    rather than moving every batch's predictions to host memory.

    With `scene_stack` supplied — a `(n_scenes, C, C)` accumulator — each
    batch is *also* attributed to the scene each tile came from, which is what
    the per-glacier and per-scene breakdowns are derived from. That costs one
    extra `bincount` per batch over the same predictions, and requires the
    loader to yield a scene index (`TileDataset(return_scene_index=True)`).
    The pooled matrix is computed independently rather than summed from the
    stack, so the two are a genuine cross-check of each other.

    With `calibration` supplied, each batch's softmax is also folded into a
    fixed-size confidence tally. That costs one softmax per batch — the loop
    would otherwise only take an argmax, which does not need one — so it is
    passed in only by the final evaluation pass, never by the per-epoch loop
    where the cost would be paid `epochs` times for a number nothing reads
    until the run ends.
    """
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch in loader:
        x, y, context, scene_index = _unpack(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if context is not None:
            context = context.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(x, context=context) if use_context else model(x)
            loss = weighted_cross_entropy(logits.float(), y, class_weights)

        total_loss += float(loss.item()) * x.size(0)
        total_items += x.size(0)
        predicted = logits.argmax(dim=1)
        confusion += confusion_from_predictions(predicted, y, num_classes, IGNORE_INDEX)

        if scene_stack is not None and scene_index is not None:
            accumulate_scene_confusions(
                scene_stack, predicted, y, scene_index, num_classes, IGNORE_INDEX
            )

        if calibration is not None:
            # float() first: a float16 softmax under AMP quantises the
            # confidence coarsely enough to shift pixels between bins, which
            # would be a measurement artefact rather than model behaviour.
            calibration.update(
                torch.softmax(logits.float(), dim=1), y, IGNORE_INDEX
            )

    mean_loss = total_loss / max(total_items, 1)
    return mean_loss, confusion.cpu().numpy()


def train_unified(
    config: TrainingConfig,
    data: Optional[PreparedData] = None,
    optuna_trial: Any = None,
    progress: bool = True,
) -> TrainingResult:
    """Train one model and return it with its best-epoch metrics.

    Args:
        config: the run's hyperparameters.
        data: an already-prepared corpus, to avoid re-tiling across the runs of
            a bootstrap or CV sweep. Prepared from `config` when omitted.
        optuna_trial: when running under HPO, receives per-epoch reports and
            may prune the trial.
        progress: print a line per epoch.
    """
    device = resolve_device(config.device)
    seed_everything(config.seed, deterministic=config.deterministic)

    if data is None:
        data = prepare_data(config)

    if not data.tiles_train:
        raise RuntimeError("the training split contains no tiles")
    if not data.tiles_val:
        raise RuntimeError("the validation split contains no tiles")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader, val_loader, train_dataset, _ = _build_loaders(
        data, config, device, generator
    )

    _, _, use_sensor_film = config.sensor_adaptation
    model = build_model(
        config.model_type,
        in_channels=config.in_channels,
        num_classes=NUM_CLASSES,
        base_ch=config.base_channels,
        dropout_p=config.dropout_p,
        use_spatial_context=config.use_spatial_context,
        num_sensors=config.effective_num_sensors,
        use_sensor_film=use_sensor_film,
        norm_type=config.norm_type,
        bottleneck_attention=config.bottleneck_attention,
        bottleneck_attention_heads=config.bottleneck_attention_heads,
        deep_supervision=config.deep_supervision,
        use_attention_gates=config.use_attention_gates,
    ).to(device)

    if config.torch_compile:
        try:
            model = torch.compile(model, mode="default")
        except Exception as exc:  # noqa: BLE001 - fall back, but say so
            # A silent fallback here changes a run's numerics without changing
            # its logs, which makes two "identical" runs disagree for no
            # visible reason.
            print(f"[train] torch.compile failed, continuing eagerly: {exc}")

    class_weights = torch.tensor(
        compute_class_weights(data.class_counts), dtype=torch.float32, device=device
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, config.epochs), eta_min=max(0.0, config.scheduler_min_lr)
    )
    use_amp = config.use_amp and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_miou_macro": [],
        "val_kappa": [],
        "val_mcc": [],
        "lr": [],
    }

    best_score = -math.inf
    best_state: dict[str, Any] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    last_epoch = 0

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            x, y, context, _ = _unpack(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if context is not None:
                context = context.to(device, non_blocking=True)

            # set_to_none frees the gradient buffers rather than zeroing them.
            optimiser.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                forward_kwargs: dict[str, Any] = {}
                if config.use_spatial_context:
                    forward_kwargs["context"] = context
                if config.deep_supervision:
                    forward_kwargs["return_aux"] = True

                output = model(x, **forward_kwargs)
                logits, aux_logits = (
                    output if config.deep_supervision else (output, [])
                )

                # Losses run in float32: softmax over four classes in float16
                # loses precision exactly where the penalties read it.
                logits_f32 = logits.float()
                loss = weighted_cross_entropy(
                    logits_f32, y, class_weights
                ) + total_penalty(logits_f32, y, config.penalties)

                if aux_logits:
                    # Only the primary head's output is ever evaluated or
                    # exported, so this term shapes training and nothing else.
                    loss = loss + deep_supervision_loss(
                        [tensor.float() for tensor in aux_logits], y, class_weights
                    )

            scaler.scale(loss).backward()
            # Unscale before clipping, or the clip threshold is applied to
            # gradients still multiplied by the AMP loss scale.
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(seen, 1)

        val_loss, confusion = evaluate(
            model, val_loader, device, class_weights, use_amp, config.use_spatial_context
        )
        val_metrics = compute_confusion_metrics(confusion)["macro"]
        miou_macro = val_metrics["miou"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou_macro"].append(miou_macro)
        history["val_kappa"].append(val_metrics["kappa"])
        history["val_mcc"].append(val_metrics["mcc"])
        history["lr"].append(float(optimiser.param_groups[0]["lr"]))

        # The glacier-weighted variants need per-glacier confusion matrices,
        # which the reference metric (miou_macro) does not, so they are not
        # computed here. Selection still routes through the same helper so a
        # different main_iou_metric picks up the right value.
        metric_name, score = select_main_iou_metric(
            config.main_iou_metric,
            miou_macro=miou_macro,
            miou_w_glacier=miou_macro,
            miou_invfq_w_glacier=val_metrics["miou_inv_freq"],
        )

        last_epoch = epoch
        if progress:
            print(
                f"[epoch {epoch}/{config.epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mIoU={miou_macro:.4f} kappa={val_metrics['kappa']:.4f} "
                f"({time.time() - started:.1f}s)"
            )

        # `score` arrives already rounded to three decimals. Preserved: it
        # tightens patience relative to a full-precision comparison, and the
        # reference model was trained under it.
        improved = math.isfinite(score) and score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                "model": (
                    None if config.keep_last_epoch_weights else _snapshot(_unwrap(model))
                ),
                "epoch": epoch,
                "val_miou_macro": miou_macro,
                "val_miou_weighted": val_metrics["miou_weighted"],
                "val_miou_inv_freq": val_metrics["miou_inv_freq"],
                "val_kappa": val_metrics["kappa"],
                "val_mcc": val_metrics["mcc"],
                "val_glacier_iou": binary_glacier_iou(confusion),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "confusion": confusion.copy(),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if optuna_trial is not None:
            optuna_trial.report(float(score) if math.isfinite(score) else -1e12, step=epoch)
            if optuna_trial.should_prune():
                import optuna

                raise optuna.TrialPruned(f"{metric_name}={score:.6f} at epoch {epoch}")

        if epochs_without_improvement >= config.patience:
            if progress:
                print(f"[train] early stop at epoch {epoch} (patience {config.patience})")
            break

    if not best_state:
        # Never improved: keep the final weights so the run still yields a model.
        best_state = {
            "model": _snapshot(_unwrap(model)),
            "epoch": last_epoch,
            "val_miou_macro": history["val_miou_macro"][-1] if history["val_miou_macro"] else float("nan"),
            "train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
            "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
            "confusion": None,
        }
        best_epoch = last_epoch

    # Restore the best weights. This is a real restore because the snapshot was
    # a copy; snapshotting the live state_dict would make it a no-op.
    if best_state.get("model") is not None:
        _unwrap(model).load_state_dict(best_state["model"])

    # One final pass over every partition on the restored best weights. This
    # re-evaluates validation rather than reusing the best epoch's matrix: the
    # per-scene attribution has to come from the same forward pass as the
    # pooled numbers it is compared against, and only this pass carries scene
    # identity. The cost is one extra inference pass per partition, once per
    # run, against `epochs` training passes.
    scenes, confusions, boundary_confusions, calibration = _final_evaluation(
        model, data, config, device, class_weights, use_amp
    )

    if not confusions and best_state.get("confusion") is not None:
        confusions["valid"] = best_state["confusion"]

    metrics = {
        split: {
            **compute_confusion_metrics(matrix),
            "glacier_iou_binary": binary_glacier_iou(matrix),
        }
        for split, matrix in confusions.items()
    }
    for split, matrix in boundary_confusions.items():
        boundary = compute_boundary_metrics(matrix)
        if boundary is not None:
            metrics[f"{split}_boundary"] = boundary

    return TrainingResult(
        model=_unwrap(model),
        best_state=best_state,
        history=history,
        class_counts=data.class_counts,
        mean=data.mean,
        std=data.std,
        features=tuple(config.features),
        best_epoch=best_epoch,
        stopped_epoch=last_epoch,
        confusions=confusions,
        metrics=metrics,
        scene_confusions=scenes,
        breakdown=(
            build_breakdown(scenes, worst_count=config.worst_case_count)
            if scenes is not None
            else {}
        ),
        boundary_confusions=boundary_confusions,
        split_counts=_split_counts(data),
        calibration=calibration,
    )


def _split_counts(data: PreparedData) -> dict[str, Any]:
    """Scene, glacier and tile counts per partition.

    Recorded because a score is not interpretable without knowing what it was
    measured over — a test mIoU from 20 glaciers and one from 200 are not
    comparable numbers even under identical training.
    """
    tiles = {
        "train": data.tiles_train,
        "valid": data.tiles_val,
        "test": data.tiles_test,
    }
    counts: dict[str, Any] = {"total_scenes": len(data.scene_records)}
    glacier_ids = data.glacier_ids

    for split, tile_list in tiles.items():
        scene_indices = sorted({int(t[0]) for t in tile_list})
        counts[f"tiles_{split}"] = len(tile_list)
        counts[f"scenes_{split}"] = len(scene_indices)
        counts[f"glaciers_{split}"] = len(
            {glacier_ids[i] for i in scene_indices if i < len(glacier_ids)}
        )

    sensors: dict[str, int] = {}
    for record in data.scene_records:
        sensors[record.sensor] = sensors.get(record.sensor, 0) + 1
    counts["scenes_by_sensor"] = dict(sorted(sensors.items()))
    return counts


def _final_evaluation(
    model: nn.Module,
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    use_amp: bool,
) -> tuple[
    Optional[SceneConfusions],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, list[dict[str, Any]]],
]:
    """Evaluate every partition, accumulating per-scene confusion matrices.

    Returns `(scene_confusions, pooled_per_split, boundary_per_split,
    calibration_per_split)`.

    Train is evaluated too. Its score is not a generalisation estimate and must
    never be read as one, but the gap between it and validation is the clearest
    single indicator of overfitting, and reporting only the two held-out
    partitions leaves that gap unmeasurable.

    The boundary matrices come from a second pass over labels restricted to the
    class-transition band, which `prepare_data` already separated out. They are
    only computed when configured, since they double the evaluation cost.

    The calibration tallies ride along on this same pass rather than costing
    their own: the softmax they need is computed from logits the pass already
    has, and the accumulator is a fixed-length vector, so the only added cost
    is one softmax and one bincount per batch on an evaluation that runs once
    per run.
    """
    n_scenes = len(data.scene_records)
    stack = scene_confusion_stack(n_scenes, NUM_CLASSES, device)
    scene_split: dict[int, str] = {}

    partitions = (
        ("train", data.tiles_train),
        ("valid", data.tiles_val),
        ("test", data.tiles_test),
    )

    common = dict(
        features=data.features,
        labels=data.labels,
        mean=data.mean,
        std=data.std,
        patch_size=config.patch_size,
        sensor_norm_stats=data.sensor_norm_stats,
        scene_sensors=data.scene_sensors,
        scene_context=data.scene_context,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = False

    confusions: dict[str, np.ndarray] = {}
    calibration: dict[str, list[dict[str, Any]]] = {}
    for split, tile_list in partitions:
        if not tile_list:
            continue
        for tile in tile_list:
            scene_split[int(tile[0])] = split

        loader = DataLoader(
            TileDataset(tile_list, return_scene_index=True, **common), **loader_kwargs
        )
        tally = CalibrationTally(config.calibration_bins, device=device)
        _, confusions[split] = evaluate(
            model,
            loader,
            device,
            class_weights,
            use_amp,
            config.use_spatial_context,
            scene_stack=stack,
            calibration=tally,
        )
        if tally.total > 0:
            calibration[split] = tally.rows()

    boundary_confusions: dict[str, np.ndarray] = {}
    if config.compute_boundary_metrics and config.ignore_boundary > 0:
        boundary_common = dict(common)
        boundary_common["labels"] = data.boundary_labels
        for split, tile_list in partitions:
            if not tile_list:
                continue
            loader = DataLoader(
                TileDataset(tile_list, **boundary_common), **loader_kwargs
            )
            _, boundary_confusions[split] = evaluate(
                model, loader, device, class_weights, use_amp, config.use_spatial_context
            )

    matrices = stack.cpu().numpy()
    identities = [
        SceneIdentity(
            scene_index=index,
            glacier_id=record.glacier_id,
            year=record.year,
            scene_id=record.scene_id,
            sensor=record.sensor,
            split=scene_split.get(index, "unused"),
        )
        for index, record in enumerate(data.scene_records)
    ]
    scenes = SceneConfusions(identities=identities, matrices=matrices)
    return scenes, confusions, boundary_confusions, calibration
