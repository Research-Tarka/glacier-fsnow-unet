"""Orchestrating the four interpretation analyses against a finished model.

One entry point, :func:`run_interpretation`, drives four independent analyses:
ambiguity calibration, permutation feature importance, expected-gradient (SHAP)
band attribution, and Grad-CAM class localisation. Each is switched on or off
by its own `enabled` flag in the config, and each writes its own outputs plus a
section of a single combined JSON summary.

Why one entry point rather than four scripts
--------------------------------------------
All four need the same three expensive things: a loaded checkpoint, a prepared
corpus, and a DataLoader over one partition of it. Preparing the corpus
dominates the runtime of any single analysis, so four separate invocations
would pay for it four times over to produce four reports about the same model.
Running them together also guarantees they describe the *same* partition of the
*same* corpus, which four independently-invoked scripts cannot promise.

The analyses stay strictly separate inside this module -- one function each,
none reading another's results -- so disabling any of them removes exactly its
own work and nothing else.

Failure policy
--------------
An analysis that raises is reported and skipped, and the run continues. These
describe a model that is already trained and saved; losing one description is
not a reason to lose the other three, and the summary records which analyses
failed rather than quietly emitting a shorter report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = ["InterpretationOutcome", "run_interpretation", "ANALYSIS_NAMES"]

#: The four analyses, in the order they run.
ANALYSIS_NAMES: tuple[str, ...] = ("ambiguity", "feature_importance", "shap", "grad_cam")


@dataclass
class InterpretationOutcome:
    """What ran, what it produced, and what it cost."""

    summary: dict[str, Any] = field(default_factory=dict)
    written: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ran_anything(self) -> bool:
        return bool(self.summary.get("analyses"))


def _log(message: str) -> None:
    print(f"[interpret] {message}", flush=True)


def _loader_for(data, training_config, split: str, device, batch_size: Optional[int] = None):
    """A single-process DataLoader over one partition's tiles.

    Single-process on purpose: permutation importance re-iterates this loader
    once per input channel, and respawning a worker pool per channel costs more
    than the loading it would parallelise at this corpus size.
    """
    from torch.utils.data import DataLoader

    from ..training.torch_dataset import TileDataset

    tiles = {
        "train": data.tiles_train,
        "valid": data.tiles_val,
        "test": data.tiles_test,
    }.get(split, data.tiles_val)
    if not tiles:
        return None

    dataset = TileDataset(
        tiles,
        features=data.features,
        labels=data.labels,
        mean=data.mean,
        std=data.std,
        patch_size=training_config.patch_size,
        sensor_norm_stats=data.sensor_norm_stats,
        scene_sensors=data.scene_sensors,
        scene_context=data.scene_context,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size or training_config.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


# -- ambiguity ---------------------------------------------------------------


def run_ambiguity(
    model,
    data,
    training_config,
    settings,
    device,
    output_dir: Path,
    model_path: str,
    write_html: bool = True,
    install_beside_checkpoint: bool = False,
) -> tuple[dict[str, Any], list[Path]]:
    """Calibrate the confidence-gap threshold and write the calibration file.

    The threshold always lands in ``ambiguity_threshold.json`` inside
    ``output_dir``. Inference reads the copy sitting beside the *checkpoint*,
    and installing it there is opt-in (``install_beside_checkpoint``): the
    checkpoint directory is frequently a published, immutable artifact, and a
    calibration run -- especially a short one over a slice of the corpus --
    must not silently change how every subsequent production inference behaves.
    The log line names the file to copy when the flag is not set.
    """
    import torch

    from .ambiguity import (
        AMBIGUITY_FILENAME,
        build_calibration,
        calibrate_threshold,
        collect_confidence_gaps,
        write_calibration,
    )

    loader = _loader_for(
        data, training_config, settings.split, device, settings.batch_size
    )
    if loader is None:
        raise ValueError(f"partition {settings.split!r} has no tiles")

    max_scenes = int(settings.max_scenes)
    parts: list[dict[str, np.ndarray]] = []
    n_tiles = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True).float()
            y = batch[1]
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            parts.append(
                collect_confidence_gaps(probabilities, y.cpu().numpy())
            )
            n_tiles += int(x.size(0))
            if max_scenes and n_tiles >= max_scenes:
                break

    collected = {
        key: np.concatenate([part[key] for part in parts]) if parts else np.empty(0)
        for key in ("truth", "argmax", "preferred", "gap")
    }
    n_pixels = int(collected["truth"].size)
    _log(f"ambiguity: {n_tiles} tiles, {n_pixels} annotated pixels")

    threshold, baseline, with_rule, sweep = calibrate_threshold(
        collected,
        tolerance=settings.tolerance,
        sweep_max=settings.threshold_sweep_max,
        sweep_steps=settings.threshold_sweep_steps,
    )
    calibration = build_calibration(
        threshold=threshold,
        baseline_miou=baseline,
        miou_with_rule=with_rule,
        tolerance=settings.tolerance,
        n_scenes=n_tiles,
        n_pixels=n_pixels,
        model_path=str(model_path),
        split=settings.split,
        sweep=sweep,
    )

    written = [write_calibration(calibration, output_dir)]
    _log(
        f"ambiguity: threshold={threshold:.4f} "
        f"baseline mIoU={baseline:.5f} with rule={with_rule:.5f} "
        f"(delta {with_rule - baseline:+.5f}, budget {settings.tolerance:g})"
    )

    # The copy inference reads has to sit beside the checkpoint.
    checkpoint_dir = Path(model_path)
    checkpoint_dir = checkpoint_dir.parent if checkpoint_dir.is_file() else checkpoint_dir
    if install_beside_checkpoint:
        try:
            beside = write_calibration(calibration, checkpoint_dir)
            if beside not in written:
                written.append(beside)
            _log(f"ambiguity: installed at {beside}; inference will use it")
        except OSError as exc:
            _log(
                f"ambiguity: could not install {AMBIGUITY_FILENAME} beside the "
                f"checkpoint ({exc}); inference keeps plain argmax"
            )
    else:
        _log(
            f"ambiguity: not installed. Inference keeps plain argmax until "
            f"{written[0]} is copied to {checkpoint_dir / AMBIGUITY_FILENAME} "
            f"(or re-run with --install-ambiguity-threshold)"
        )

    if write_html:
        from .reports import write_ambiguity_report

        written.append(
            write_ambiguity_report(
                sweep,
                threshold,
                settings.tolerance,
                baseline,
                output_dir / "ambiguity_threshold.html",
            )
        )

    payload = calibration.to_dict()
    # The full sweep is already in ambiguity_threshold.json; repeating a few
    # hundred records inside the combined summary makes it unreadable.
    payload.pop("sweep", None)
    payload["sweep_points"] = len(sweep)
    return payload, written


# -- permutation feature importance ------------------------------------------


def run_feature_importance(
    model, data, training_config, settings, device, output_dir: Path
) -> tuple[dict[str, Any], list[Path]]:
    """Permutation importance per band and per class, as a CSV plus summary.

    The measurement itself is
    :func:`glacier_fsnow_unet.training.shap_importance.permutation_importance_table`;
    this function only sources the loader and writes the outputs, so the
    numbers here and the ones a training run exports come from one
    implementation.
    """
    import pandas as pd

    from ..training.config import CLASS_NAMES
    from ..training.shap_importance import permutation_importance_table

    loader = _loader_for(data, training_config, settings.split, device)
    if loader is None:
        raise ValueError(f"partition {settings.split!r} has no tiles")

    features = list(training_config.features)
    _log(
        f"feature importance: {len(features)} bands on {settings.split} "
        f"({len(features) + 1} evaluation passes)"
    )
    rows = permutation_importance_table(
        model,
        loader,
        features,
        device,
        seed=int(settings.seed),
        use_amp=training_config.use_amp and device.type == "cuda",
        use_context=training_config.use_spatial_context,
        max_batches=int(settings.max_batches),
        class_names=CLASS_NAMES,
    )

    table = pd.DataFrame(rows)
    destination = output_dir / "permutation_importance.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)

    macro = table[table["class_name"] == "macro"]
    ranked = macro.sort_values("importance_delta_iou", ascending=False)
    for row in ranked.head(3).itertuples():
        _log(f"feature importance: {row.feature} -> {row.importance_delta_iou:+.5f} mIoU")

    return (
        {
            "split": settings.split,
            "n_bands": len(features),
            "max_batches": int(settings.max_batches),
            "ranked_macro": [
                {"feature": str(r.feature), "delta_miou": float(r.importance_delta_iou)}
                for r in ranked.itertuples()
            ],
            "table": str(destination),
        },
        [destination],
    )


# -- expected-gradient (SHAP) attribution ------------------------------------


def run_shap(
    model, data, training_config, settings, device, output_dir: Path, write_html: bool = True
) -> tuple[dict[str, Any], list[Path]]:
    """Expected-gradient band attribution per class, plus both HTML views.

    The estimator is
    :func:`glacier_fsnow_unet.training.shap_importance.expected_gradient_attribution`,
    whose own docstring records why it is a hand-rolled expected-gradients
    Shapley estimator rather than a call into `shap` or `captum`.
    """
    import pandas as pd

    from ..training.config import CLASS_NAMES
    from ..training.shap_importance import expected_gradient_attribution

    loader = _loader_for(
        data, training_config, settings.split, device, settings.batch_size
    )
    if loader is None:
        raise ValueError(f"partition {settings.split!r} has no tiles")

    features = list(training_config.features)
    _log(
        f"shap: {settings.test_samples} tiles x {settings.background_samples} "
        f"baselines on {settings.split}"
    )
    attribution = expected_gradient_attribution(
        model,
        loader,
        features,
        device,
        n_samples=int(settings.test_samples),
        n_baselines=int(settings.background_samples),
        batch_size=int(settings.batch_size),
        seed=int(settings.seed),
        use_context=training_config.use_spatial_context,
        class_names=CLASS_NAMES,
    )

    written: list[Path] = []
    table = pd.DataFrame(attribution.rows())
    destination = output_dir / "shap_band_attribution.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    written.append(destination)

    macro = attribution.macro()
    if write_html:
        from .reports import write_shap_global_report, write_shap_per_class_report

        written.append(
            write_shap_global_report(
                list(macro.keys()),
                list(macro.values()),
                output_dir / "shap_global_bands.html",
                method=attribution.method,
            )
        )
        written.append(
            write_shap_per_class_report(
                attribution.feature_names,
                attribution.class_names,
                attribution.values,
                output_dir / "shap_per_class_bands.html",
                method=attribution.method,
            )
        )

    ranked = sorted(macro.items(), key=lambda kv: kv[1], reverse=True)
    for name, value in ranked[:3]:
        _log(f"shap: {name} -> mean |attribution| {value:.6g}")

    return (
        {
            "method": settings.method,
            "split": settings.split,
            "n_samples": attribution.n_samples,
            "n_baselines": attribution.n_baselines,
            "macro_ranked": [
                {"feature": name, "mean_abs_attribution": float(value)}
                for name, value in ranked
            ],
            "table": str(destination),
        },
        written,
    )


# -- Grad-CAM ----------------------------------------------------------------


def run_grad_cam(
    model, data, training_config, settings, device, output_dir: Path, write_html: bool = True
) -> tuple[dict[str, Any], list[Path]]:
    """Per-class localisation maps, persisted in zarr and rendered to HTML."""
    from ..training.config import CLASS_NAMES, IGNORE_INDEX
    from .grad_cam import compute_grad_cam, write_maps_to_zarr

    loader = _loader_for(
        data, training_config, settings.split, device, settings.batch_size
    )
    if loader is None:
        raise ValueError(f"partition {settings.split!r} has no tiles")

    _log(
        f"grad-cam: {settings.samples} tiles on {settings.split}, "
        f"layer {settings.target_layer}"
    )
    result = compute_grad_cam(
        model,
        loader,
        device,
        CLASS_NAMES,
        target_layer=settings.target_layer,
        n_samples=int(settings.samples),
        batch_size=int(settings.batch_size),
        class_conditioned=bool(settings.class_conditioned),
        min_class_pixels=int(settings.min_class_pixels),
        ignore_index=IGNORE_INDEX,
    )

    written = [write_maps_to_zarr(result, output_dir / settings.zarr_dirname)]
    _log(f"grad-cam: per-pixel maps written to {written[0]}")

    summary = result.summary()
    if write_html:
        from .reports import write_grad_cam_report

        written.append(
            write_grad_cam_report(
                result.class_names,
                [
                    result.class_mean_map(index)
                    for index in range(len(result.class_names))
                ],
                [record["n_tiles_explained"] for record in summary],
                output_dir / "gradcam_heatmaps.html",
                target_layer=result.target_layer,
            )
        )

    for record in summary:
        _log(
            f"grad-cam: {record['class_name']} explained on "
            f"{record['n_tiles_explained']}/{result.n_tiles} tiles"
        )

    return (
        {
            "split": settings.split,
            "target_layer": result.target_layer,
            "n_tiles": result.n_tiles,
            "class_conditioned": bool(settings.class_conditioned),
            "min_class_pixels": int(settings.min_class_pixels),
            "per_class": summary,
            "maps_zarr": str(written[0]),
        },
        written,
    )


# -- orchestration -----------------------------------------------------------


def run_interpretation(
    model,
    data,
    training_config,
    interpretation_config,
    device,
    output_dir: str | Path,
    model_path: str,
    only: Optional[set[str]] = None,
    write_html: bool = True,
    install_ambiguity_threshold: bool = False,
) -> InterpretationOutcome:
    """Run every enabled analysis and write the combined JSON summary.

    Parameters
    ----------
    interpretation_config
        A :class:`~glacier_fsnow_unet.config.UnetInterpretationConfig`. Each of
        its four sub-sections carries its own ``enabled`` flag; a disabled one
        is logged and skipped, never silently omitted.
    only
        Optional CLI-level narrowing on top of the config flags. An analysis
        runs when it is enabled in the config *and* (if ``only`` is given)
        named in it -- so a CLI flag can subtract from what the config allows
        but never add to it, and the config stays the single source of truth
        for what a run is permitted to do.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcome = InterpretationOutcome()
    analyses: dict[str, Any] = {}

    sections = {
        "ambiguity": (
            interpretation_config.ambiguity,
            lambda s: run_ambiguity(
                model, data, training_config, s, device, output_dir, model_path,
                write_html=write_html,
                install_beside_checkpoint=install_ambiguity_threshold,
            ),
        ),
        "feature_importance": (
            interpretation_config.feature_importance,
            lambda s: run_feature_importance(
                model, data, training_config, s, device, output_dir
            ),
        ),
        "shap": (
            interpretation_config.shap,
            lambda s: run_shap(
                model, data, training_config, s, device, output_dir,
                write_html=write_html,
            ),
        ),
        "grad_cam": (
            interpretation_config.grad_cam,
            lambda s: run_grad_cam(
                model, data, training_config, s, device, output_dir,
                write_html=write_html,
            ),
        ),
    }

    for name in ANALYSIS_NAMES:
        settings, run = sections[name]
        if not settings.enabled:
            _log(f"{name}: SKIPPED (unet_interpretation.{name}.enabled is false)")
            outcome.skipped.append(name)
            continue
        if only is not None and name not in only:
            _log(f"{name}: SKIPPED (not selected on the command line)")
            outcome.skipped.append(name)
            continue

        started = time.perf_counter()
        try:
            payload, written = run(settings)
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            _log(f"{name}: FAILED ({type(exc).__name__}: {exc})")
            outcome.failed[name] = f"{type(exc).__name__}: {exc}"
            continue

        payload["duration_s"] = round(time.perf_counter() - started, 2)
        analyses[name] = payload
        outcome.written.extend(written)
        _log(f"{name}: done in {payload['duration_s']:.1f}s")

    outcome.summary = {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "analyses": analyses,
        "skipped": outcome.skipped,
        "failed": outcome.failed,
    }

    summary_path = output_dir / "interpretation_summary.json"
    summary_path.write_text(
        json.dumps(outcome.summary, indent=2, default=str), encoding="utf-8"
    )
    outcome.written.append(summary_path)
    return outcome
