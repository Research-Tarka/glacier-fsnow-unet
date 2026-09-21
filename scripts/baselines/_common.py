"""Shared plumbing for the `scripts/baselines/*.py` model-comparison scripts
(NDSI, Random Forest, DeepLabv3, SegFormer, and the statistical comparator).

Centralises what would otherwise be duplicated across those scripts:
reproducing the exact glacier-level split, loading `baselines:` out of the
same `configs/config.yaml` the U-Net reads, the generic PyTorch train/eval
loop shared by DeepLab and SegFormer, the duplicate-training-process guard,
and the Welch t-test + bootstrap CI comparison against the published U-Net.

`build_split()` reproduces the split via
`glacier_fsnow_unet.training.dataset.prepare_data` and cross-checks it
scene-by-scene against a real trained checkpoint's recorded split
(`meta["scene_identities"]` / `meta["scene_splits"]`), so a baseline number
is never reported against a split that silently drifted from the published
one. `load_baselines_config()` loads the full `PipelineConfig` via
`glacier_fsnow_unet.config.load_config` and returns just its `.baselines`
sub-model plus the resolved corpus root -- one config file, one loader,
shared with `05_train_model.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.config import BaselinesConfig  # noqa: E402,F401 (re-exported)
from glacier_fsnow_unet.training.config import (  # noqa: E402
    DEFAULT_FEATURES,
    IGNORE_INDEX,
    NUM_CLASSES,
    TrainingConfig,
)
from glacier_fsnow_unet.training import dataset as ds  # noqa: E402
from glacier_fsnow_unet.training.metrics import (  # noqa: E402
    compute_confusion_metrics,
    update_confusion,
)

# Default corpus root -- the real feature cache, read-only. Matches
# `paths.train_root` in configs/config.yaml; used only if a caller invokes
# `_common` functions directly without going through `load_baselines_config`
# (which always resolves `paths.train_root` from the loaded config instead).
DEFAULT_TRAIN_ROOT = Path("D:/Recherche/Maxime/Data_V4/Train")

# A reference checkpoint used only to cross-check the reproduced split. Any
# checkpoint from the GlacierSplit_WithBoundaries family works. Not required
# to exist on disk -- see build_split()'s docstring.
DEFAULT_CHECK_CHECKPOINT = (
    REPO_ROOT / "_work" / "sigtest" / "checkpoints" / "glacier_split" / "seed_0_model.pt"
)

# Published U-Net reference numbers (read-only), for statistical comparison.
# Source: the published Zenodo deposit's UNet_Models/GlacierSplit_WithBoundaries/
# bootstrap_summary.csv (5 seeds). Hardcoded here deliberately: the published
# deposit is read-only and must never be a runtime dependency of these scripts
# (it may not even be mounted on every machine).
UNET_TEST_MIOU_5SEEDS: tuple[float, ...] = (0.705, 0.713, 0.696, 0.699, 0.692)

# Re-exported so callers can catch a single error type without importing
# from glacier_fsnow_unet.config directly.
BaselinesConfigError = ConfigError


def load_baselines_config(config_path: Optional[str] = None) -> tuple[BaselinesConfig, Path]:
    """Load `configs/config.yaml` (or `config_path`) and return its
    `baselines:` section, alongside the resolved corpus root.

    Uses the exact same `glacier_fsnow_unet.config.load_config` mechanism as
    `scripts/05_train_model.py` -- one config file, one loader, for both the
    published U-Net and every baseline script. `paths.train_root` is used as
    the corpus root unless `baselines.train_root` overrides it.
    """
    pipeline_cfg = load_config(config_path)
    baselines_cfg = pipeline_cfg.baselines
    train_root = Path(baselines_cfg.train_root) if baselines_cfg.train_root else Path(
        pipeline_cfg.paths.train_root
    )
    return baselines_cfg, train_root


def apply_overrides(model: Any, **overrides: Any) -> Any:
    """Return a copy of a pydantic sub-config with only the non-None
    overrides applied -- the same "override only if explicitly passed"
    contract `05_train_model.py::main()` uses for the U-Net's CLI flags."""
    changes = {k: v for k, v in overrides.items() if v is not None}
    return model.model_copy(update=changes) if changes else model


# ---------------------------------------------------------------------------
# Split reproduction
# ---------------------------------------------------------------------------


def reference_config(train_root: Optional[Path] = None, **overrides: Any) -> TrainingConfig:
    """The `TrainingConfig` fields that determine the split and the features,
    matching `configs/config.yaml`'s reference (production) values exactly.

    Only fields that affect scan_scenes / split_scenes / load_scene_arrays
    matter here; patch_size/stride/batch_size etc. are also fixed at the
    values that produced the published checkpoints so every baseline sees
    the identical tiling, but a caller may override any of them (e.g. a
    different patch_size for a smoke test) via **overrides.
    """
    base: dict[str, Any] = dict(
        train_root=train_root or DEFAULT_TRAIN_ROOT,
        features=DEFAULT_FEATURES,
        skip_landsat7_2003_2012=True,
        val_ratio=0.20,
        test_ratio=0.10,
        split_glacier=True,
        split_by_sensor=True,
        glacier_size_stratify=True,
        class_density_balance=False,
        split_seed=42,
        ignore_boundary=0,  # reference: boundary band kept
        patch_size=48,
        stride=16,
        min_valid=5,
        rotations=False,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def prepare(config: Optional[TrainingConfig] = None) -> "ds.PreparedData":
    config = config or reference_config()
    return ds.prepare_data(config)


def _scene_key(glacier_id: str, year: Any, entity_id: str) -> tuple[str, int, str]:
    return (str(glacier_id), int(year), str(entity_id))


def cross_check_against_checkpoint(
    data: "ds.PreparedData",
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """Compare the reproduced split against a real trained checkpoint's
    recorded split. Returns a report dict; does not raise itself (callers
    decide whether a mismatch is fatal -- see `build_split()` below, which
    is fatal by default, matching every prior baseline script's behaviour).

    If no checkpoint is available at `checkpoint_path` (or the default), the
    cross-check is skipped with an explicit `"skipped"` reason rather than
    silently reporting 0 mismatches -- a skipped check must never be
    mistaken for a passed one.
    """
    checkpoint_path = checkpoint_path or DEFAULT_CHECK_CHECKPOINT
    if not Path(checkpoint_path).is_file():
        return {
            "skipped": True,
            "reason": f"no reference checkpoint at {checkpoint_path}",
            "n_scenes_ours": len(data.scene_records),
            "mismatches": 0,
        }

    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    ref_identities = meta["scene_identities"]
    ref_splits = meta["scene_splits"]
    assert len(ref_identities) == len(ref_splits)

    train_idx = {t[0] for t in data.tiles_train}
    val_idx = {t[0] for t in data.tiles_val}
    test_idx = {t[0] for t in data.tiles_test}
    scene_index_to_split: dict[int, str] = {}
    for i in range(len(data.scene_records)):
        if i in train_idx:
            scene_index_to_split[i] = "train"
        elif i in val_idx:
            scene_index_to_split[i] = "valid"
        elif i in test_idx:
            scene_index_to_split[i] = "test"

    key_to_index = {
        _scene_key(rec.glacier_id, rec.year, rec.scene_id): i
        for i, rec in enumerate(data.scene_records)
    }

    mismatches = []
    matched = 0
    missing_in_ours = 0
    for identity, ref_split in zip(ref_identities, ref_splits):
        key = _scene_key(identity["id_glims"], identity["year"], identity["entityid"])
        our_index = key_to_index.get(key)
        if our_index is None:
            missing_in_ours += 1
            continue
        our_split = scene_index_to_split.get(our_index, "dropped_no_tiles")
        if our_split != ref_split:
            mismatches.append({"key": list(key), "reference": ref_split, "ours": our_split})
        else:
            matched += 1

    return {
        "skipped": False,
        "n_scenes_ours": len(data.scene_records),
        "n_scenes_reference": len(ref_identities),
        "matched": matched,
        "mismatches": len(mismatches),
        "missing_in_ours": missing_in_ours,
        "mismatch_examples": mismatches[:20],
    }


def _split_cache_path(train_root: Path) -> Path:
    import hashlib

    key = hashlib.sha1(str(train_root).encode("utf-8")).hexdigest()[:16]
    cache_dir = REPO_ROOT / "_work" / "baselines" / "_split_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"prepared_data_{key}.pkl"


def build_split(
    train_root: Optional[Path] = None,
    check_checkpoint: Optional[Path] = None,
    require_cross_check: bool = True,
    use_cache: bool = True,
) -> tuple["ds.PreparedData", dict]:
    """Reproduce the exact glacier-level split and cross-check it.

    Raises `AssertionError` on a genuine mismatch (never safe to compare
    against the published 0.701 in that case). A *skipped* cross-check
    (no reference checkpoint found) only raises if `require_cross_check` is
    True (the default) -- set it False for environments that intentionally
    have no copy of the reference checkpoint (e.g. CI, a fresh clone) and
    are willing to trust the split reproduction unverified.

    Loading and tiling the ~540-scene corpus from disk dominates the
    wall-clock cost of every baseline script's startup. `use_cache=True`
    (the default) pickles the assembled `PreparedData` to
    `_work/baselines/_split_cache/` on first use and reloads it on
    subsequent invocations against the same `train_root`, so a HPO search
    followed by a multi-seed confirmation only pays that cost once.
    """
    train_root = Path(train_root) if train_root else DEFAULT_TRAIN_ROOT
    cache_path = _split_cache_path(train_root) if use_cache else None

    if cache_path and cache_path.is_file():
        import pickle

        print(f"[baselines] loading cached prepared split from {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    else:
        print("[baselines] reproducing the exact glacier-level split ...")
        config = reference_config(train_root=train_root)
        data = prepare(config)
        if cache_path:
            import pickle

            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[baselines] cached prepared split to {cache_path}")

    check = cross_check_against_checkpoint(data, checkpoint_path=check_checkpoint)
    if check.get("mismatches", 0) > 0:
        raise AssertionError(
            f"Split reproduction MISMATCHES the reference checkpoint's split on "
            f"{check['mismatches']} scenes -- refusing to report numbers as "
            f"comparable to the published U-Net until this is resolved."
        )
    if check.get("skipped") and require_cross_check:
        raise AssertionError(
            f"Split cross-check was skipped ({check['reason']}) and "
            f"require_cross_check=True -- pass check_checkpoint= explicitly, "
            f"or require_cross_check=False to proceed unverified."
        )
    print(f"[baselines] split cross-check: {check}")
    print(
        f"[baselines] tiles train/val/test = "
        f"{len(data.tiles_train)}/{len(data.tiles_val)}/{len(data.tiles_test)}"
    )
    return data, check


def scene_indices_for_split(data: "ds.PreparedData", split: str) -> list[int]:
    train_idx = {t[0] for t in data.tiles_train}
    val_idx = {t[0] for t in data.tiles_val}
    test_idx = {t[0] for t in data.tiles_test}
    mapping = {"train": train_idx, "valid": val_idx, "test": test_idx}
    return sorted(mapping[split])


# ---------------------------------------------------------------------------
# Duplicate-training-process guard
# ---------------------------------------------------------------------------


def check_no_duplicate_training_process(extra_pattern: str = "baselines") -> None:
    """Refuse to start if another training-like process is already running.

    Two trainings at once (the U-Net trainer, or two baseline scripts) can
    saturate system RAM through duplicated DataLoader workers, so this is a
    hard stop rather than a warning. `extra_pattern` is matched against each
    process's command line in addition to `05_train_model`, so both the
    published U-Net trainer and any baseline script are covered by one
    guard. Best-effort: warns instead of raising if the check itself cannot
    run (e.g. no PowerShell on the host).
    """
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                f"Where-Object {{ $_.CommandLine -match '05_train_model|{extra_pattern}' }} | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True, text=True, timeout=15,
        )
        pids = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
        own_pid = str(os.getpid())
        other_pids = [p for p in pids if p != own_pid]
        if other_pids:
            raise RuntimeError(
                f"Another training-like process appears to be running already "
                f"(PID(s) {other_pids}). Refusing to start a second one -- stop it "
                f"first; never run two trainings in parallel."
            )
    except FileNotFoundError:
        print("[baselines] WARNING: could not run the duplicate-process check (powershell not found).")


# ---------------------------------------------------------------------------
# Generic PyTorch train/eval loop (shared by DeepLab and SegFormer)
# ---------------------------------------------------------------------------


def build_datasets(data: "ds.PreparedData"):
    from glacier_fsnow_unet.training.torch_dataset import TileDataset

    def _make(tiles):
        return TileDataset(
            tiles=tiles, features=data.features, labels=data.labels,
            mean=data.mean, std=data.std, patch_size=data.patch_size,
            sensor_norm_stats=None, scene_sensors=data.scene_sensors, scene_context=None,
        )

    return _make(data.tiles_train), _make(data.tiles_val), _make(data.tiles_test)


def _collate(batch):
    import torch

    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def evaluate_torch_model(
    model, loader, forward_fn: Callable[[Any, Any], Any], use_amp: bool = True
):
    """forward_fn(model, x) -> logits (N, C, H, W). Lets DeepLab (dict output
    under `["out"]`) and SegFormer (plain tensor) share one eval loop."""
    import torch
    import torch.nn as nn

    device = next(model.parameters()).device
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = forward_fn(model, x)
                loss = criterion(logits, y)
            total_loss += float(loss.item())
            n_batches += 1
            pred = logits.argmax(dim=1)
            update_confusion(confusion, pred, y)
    return confusion, total_loss / max(1, n_batches)


def train_one_config(
    data: "ds.PreparedData",
    build_model_fn: Callable[[], Any],
    forward_fn: Callable[[Any, Any], Any],
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    log_prefix: str,
    num_workers: int = 4,
    dropout_p: Optional[float] = None,
    device: Optional[str] = None,
    use_amp: bool = True,
) -> dict:
    """Generic train/eval loop shared by the DeepLab and SegFormer baselines.

    Deliberately not `glacier_fsnow_unet.training.train`: that loop is
    coupled to the U-Net's architecture-variant machinery (attention gates,
    deep supervision, sensor FiLM, torch.compile), none of which these
    baselines use. Mixed precision (`use_amp`) and cuDNN autotuning are on
    by default on CUDA -- both are inference/training speedups only, with
    no effect on the split, the model, or the reported metrics.
    """
    import torch
    from torch.utils.data import DataLoader

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = use_amp and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)

    pin_memory = dev.type == "cuda"
    train_ds, val_ds, test_ds = build_datasets(data)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_collate, drop_last=True, persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2),
        collate_fn=_collate, persistent_workers=num_workers > 0, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2),
        collate_fn=_collate, persistent_workers=num_workers > 0, pin_memory=pin_memory,
    )

    model = build_model_fn().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    best_val_miou = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    t_start = time.time()
    for epoch in range(max_epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp_enabled):
                logits = forward_fn(model, x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_confusion, val_loss = evaluate_torch_model(
            model, val_loader, forward_fn, use_amp=amp_enabled
        )
        val_metrics = compute_confusion_metrics(val_confusion)
        val_miou = val_metrics["macro"]["miou"]
        scheduler.step(val_miou)
        epoch_time = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_miou": val_miou, "epoch_time_s": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[{log_prefix}] epoch {epoch:3d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_miou={val_miou:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} ({epoch_time:.1f}s)"
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{log_prefix}] early stopping at epoch {epoch} (patience={patience})")
                break

    total_time = time.time() - t_start
    print(f"[{log_prefix}] training done in {total_time:.1f}s, best_val_miou={best_val_miou:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2),
        collate_fn=_collate, pin_memory=pin_memory,
    )
    train_confusion, _ = evaluate_torch_model(model, train_eval_loader, forward_fn, use_amp=amp_enabled)
    val_confusion, _ = evaluate_torch_model(model, val_loader, forward_fn, use_amp=amp_enabled)
    test_confusion, _ = evaluate_torch_model(model, test_loader, forward_fn, use_amp=amp_enabled)

    return {
        "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "dropout_p": dropout_p, "seed": seed,
        "best_val_miou": best_val_miou, "n_epochs_run": len(history),
        "total_time_s": total_time, "history": history,
        "confusions": {
            "train": train_confusion.tolist(),
            "valid": val_confusion.tolist(),
            "test": test_confusion.tolist(),
        },
        "metrics": {
            "train": compute_confusion_metrics(train_confusion),
            "valid": compute_confusion_metrics(val_confusion),
            "test": compute_confusion_metrics(test_confusion),
        },
        "model_state": best_state,
    }


def torch_baseline_checkpoint_meta(data: "ds.PreparedData") -> dict:
    """`meta` fields shared by every 16_/17_ checkpoint, in the same shape
    `19_uncertainty_analysis.py` expects (`mean`, `std`, `features`,
    `class_names`) -- a slimmed-down version of
    `glacier_fsnow_unet.training.export.checkpoint_metadata`'s schema,
    covering only what a post-hoc analysis over these baselines needs.
    """
    return {
        "mean": np.asarray(data.mean, dtype=np.float32).tolist(),
        "std": np.asarray(data.std, dtype=np.float32).tolist(),
        "features": list(DEFAULT_FEATURES),
        "class_names": ["Cloud", "Snow", "Ice", "Other"],
    }


# ---------------------------------------------------------------------------
# Statistical comparison (generic; also used by 18_compare_models.py)
# ---------------------------------------------------------------------------


def compare_series(
    candidate: Any,
    reference: Any,
    paired: bool = False,
    n_boot: int = 10_000,
    boot_seed: int = 0,
) -> dict:
    """Welch's t-test (or paired t-test) + a bootstrap CI on the mean
    difference between two series of per-seed test scores.

    `significant_at_0.05` is explicitly `None` (never a silently-wrong
    `False`) whenever either group has fewer than 2 seeds: a t-test is
    undefined there (no within-group variance to estimate), and `NaN < 0.05`
    evaluates `False` in Python, which would otherwise misreport a
    single-seed run as "not significant".
    """
    from scipy import stats

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    if paired:
        if cand.size != ref.size:
            raise ValueError(
                f"paired comparison requires equal-length series, got "
                f"{cand.size} vs {ref.size}"
            )
        if cand.size < 2:
            t_stat, p_value = float("nan"), float("nan")
        else:
            t_stat, p_value = stats.ttest_rel(cand, ref)
    else:
        if cand.size < 2 or ref.size < 2:
            t_stat, p_value = float("nan"), float("nan")
        else:
            t_stat, p_value = stats.ttest_ind(cand, ref, equal_var=False)

    rng = np.random.default_rng(boot_seed)
    if paired:
        diffs = cand - ref
        boot_means = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sample = rng.choice(diffs, size=diffs.size, replace=True)
            boot_means[i] = sample.mean()
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        mean_diff = float(diffs.mean())
    else:
        boot = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            boot_cand = rng.choice(cand, size=cand.size, replace=True)
            boot_ref = rng.choice(ref, size=ref.size, replace=True)
            boot[i] = boot_cand.mean() - boot_ref.mean()
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        mean_diff = float(cand.mean() - ref.mean())

    n_min = min(cand.size, ref.size) if not paired else cand.size
    return {
        "candidate_mean": float(cand.mean()) if cand.size else float("nan"),
        "candidate_std": float(cand.std(ddof=1)) if cand.size > 1 else float("nan"),
        "candidate_n_seeds": int(cand.size),
        "reference_mean": float(ref.mean()) if ref.size else float("nan"),
        "reference_std": float(ref.std(ddof=1)) if ref.size > 1 else float("nan"),
        "reference_n_seeds": int(ref.size),
        "paired": paired,
        "mean_diff_candidate_minus_reference": mean_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "bootstrap_ci_95_diff": [float(ci_low), float(ci_high)],
        "significant_at_0.05": (bool(p_value < 0.05) if n_min >= 2 else None),
    }


def compare_to_unet(candidate_test_mious: Any) -> dict:
    """Convenience wrapper: compare a candidate's per-seed test mIoU series
    against the published U-Net's 5-seed test mIoU (unpaired Welch test,
    since the two seed sets are independent runs, not a matched design)."""
    return compare_series(candidate_test_mious, UNET_TEST_MIOU_5SEEDS, paired=False)
