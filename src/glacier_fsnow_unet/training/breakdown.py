"""Per-scene, per-glacier and per-sensor metric breakdowns.

A single pooled confusion matrix per partition answers "how good is the model",
and nothing else. It cannot say whether a mediocre score is uniform weakness or
one badly-handled sensor dragging down three good ones, nor which glaciers the
model fails on. Those questions need the confusion matrix *disaggregated* by the
scene each prediction came from, which is what this module produces.

**One accumulator, every table.** Evaluation builds a single `(n_scenes, C, C)`
stack — one confusion matrix per scene — and every table here is a different
grouping of that same stack. A per-glacier row is the element-wise sum of its
scenes' matrices; a per-sensor row sums a different subset; the pooled matrix is
the sum of all of them. Nothing is recomputed from predictions, so the
breakdowns are guaranteed consistent with each other and with the pooled score
by construction rather than by convention.

**Summing matrices, not averaging scores.** A glacier's IoU is derived from its
summed confusion matrix, not from the mean of its scenes' IoUs. Averaging scores
would let a scene with three Ice pixels weigh as heavily as one with three
thousand — the same pooling argument the module-level metrics rest on, applied
one level down.

**Vectorised accumulation.** The per-scene stack is filled by one `bincount`
over the flattened `(scene, true, predicted)` index per batch, so a batch of 96
tiles spanning many scenes costs one kernel rather than one pass per scene. The
whole stack for this corpus is 540 x 4 x 4 int64, well under a megabyte, so it
stays resident on the GPU for the duration of a pass and transfers once.

Two derived quantities need a definition stated up front:

*Glacier-weighted mIoU* is the mean of per-glacier mIoUs — every glacier counts
once regardless of how many scenes or pixels it contributed. It answers "how
well does this work on a typical glacier", which is the question a user applying
the model to a new glacier is actually asking, and it is not the same as the
pixel-pooled score when one glacier carries a hundred scenes and the median
carries two.

*Sensor robustness* is the spread of a metric across sensors within a partition
— min, max, gap, standard deviation, and which sensor is worst. A model that
scores well on average while failing on one sensor is not usable across the
archive, and the pooled number cannot show that.

*Worst-case shortlists* re-sort the per-scene and per-glacier tables by mIoU and
keep the bottom few per partition. They add no new measurement — every number in
them is already in the tables above — but they turn "the model averages 0.71"
into a list of specific scenes to open, which is what improving a model actually
starts from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from .config import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from .metrics import (
    binary_glacier_iou,
    compute_confusion_metrics,
)

__all__ = [
    "SceneIdentity",
    "SceneConfusions",
    "accumulate_scene_confusions",
    "scene_confusion_stack",
    "scene_performance_table",
    "glacier_performance_table",
    "sensor_performance_table",
    "sensor_robustness_table",
    "class_iou_by_sensor_table",
    "worst_scenes_table",
    "worst_glaciers_table",
    "build_breakdown",
    "DEFAULT_WORST_COUNT",
]

# Sensor labels as they appear in the corpus, mapped to display names. Keeps the
# published tables readable without changing the identifiers the pipeline keys on.
SENSOR_DISPLAY: Mapping[str, str] = {
    "landsat": "Landsat8/9",
    "landsat5": "Landsat5",
    "landsat7": "Landsat7",
    "sentinel": "Sentinel2",
    "map": "Map",
}


def sensor_display_name(sensor: str) -> str:
    """Readable sensor name, falling back to the raw label."""
    key = str(sensor or "unknown").strip().lower()
    return SENSOR_DISPLAY.get(key, key)


@dataclass(frozen=True)
class SceneIdentity:
    """Everything needed to attribute one scene's predictions."""

    scene_index: int
    glacier_id: str
    year: int
    scene_id: str
    sensor: str
    split: str


@dataclass
class SceneConfusions:
    """Per-scene confusion matrices for one or more partitions.

    `matrices` is `(n_scenes, C, C)` indexed by the scene's position in
    `identities`. Scenes with no evaluated pixels keep an all-zero matrix and
    are dropped from the tables rather than scored NaN.
    """

    identities: list[SceneIdentity]
    matrices: np.ndarray

    def __post_init__(self) -> None:
        if self.matrices.shape[0] != len(self.identities):
            raise ValueError(
                f"{self.matrices.shape[0]} matrices for {len(self.identities)} scenes"
            )

    @property
    def pooled(self) -> np.ndarray:
        """The sum over every scene — identical to a pooled evaluation pass."""
        return self.matrices.sum(axis=0)

    def pooled_for(self, split: str) -> np.ndarray:
        """The sum over one partition's scenes."""
        mask = np.array(
            [identity.split == split for identity in self.identities], dtype=bool
        )
        if not mask.any():
            return np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        return self.matrices[mask].sum(axis=0)

    def splits(self) -> list[str]:
        """Partition names present, in a stable order."""
        return sorted({identity.split for identity in self.identities})


def scene_confusion_stack(
    n_scenes: int,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """An empty `(n_scenes, C, C)` accumulator on the given device."""
    return torch.zeros(
        (int(n_scenes), int(num_classes), int(num_classes)),
        dtype=torch.int64,
        device=device,
    )


def accumulate_scene_confusions(
    stack: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    scene_index: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Add one batch to the per-scene confusion stack, in place.

    `scene_index` is one scene id per tile in the batch; it is broadcast over
    that tile's pixels so every pixel is attributed to the scene it came from.

    The whole batch is one `bincount` over the flattened
    `scene * C * C + true * C + pred` index. Looping per scene instead would
    launch one kernel per distinct scene in the batch, which for a shuffled
    loader is close to one per tile.
    """
    n_scenes = int(stack.shape[0])
    if n_scenes == 0:
        return stack

    valid = target != ignore_index
    if not torch.any(valid):
        return stack

    # Broadcast the per-tile scene id across that tile's pixels.
    scenes = scene_index.to(target.device).long().reshape(-1, *([1] * (target.dim() - 1)))
    scenes = scenes.expand_as(target)

    true_flat = target[valid].reshape(-1).long()
    pred_flat = pred[valid].reshape(-1).long()
    scene_flat = scenes[valid].reshape(-1).long()

    # Guard against an out-of-range scene id rather than corrupting a
    # neighbouring scene's counts through index wraparound.
    in_range = (scene_flat >= 0) & (scene_flat < n_scenes)
    if not torch.all(in_range):
        true_flat = true_flat[in_range]
        pred_flat = pred_flat[in_range]
        scene_flat = scene_flat[in_range]
        if scene_flat.numel() == 0:
            return stack

    cells = num_classes * num_classes
    flat = scene_flat * cells + true_flat * num_classes + pred_flat
    counts = torch.bincount(flat, minlength=n_scenes * cells)
    stack += counts[: n_scenes * cells].reshape(n_scenes, num_classes, num_classes)
    return stack


# -- derived rows ------------------------------------------------------------


def _class_pixel_shares(confusion: np.ndarray) -> dict[str, float]:
    """Each class's share of the true pixels, as a percentage."""
    totals = np.asarray(confusion, dtype=np.float64).sum(axis=1)
    total = float(totals.sum())
    return {
        CLASS_NAMES[index]: (100.0 * float(totals[index]) / total if total > 0 else 0.0)
        for index in range(min(len(CLASS_NAMES), totals.size))
    }


def _metrics_row(confusion: np.ndarray) -> dict[str, Any]:
    """The metric columns shared by every breakdown table."""
    metrics = compute_confusion_metrics(confusion)
    macro = metrics["macro"]
    per_class = metrics["per_class"]

    row: dict[str, Any] = {
        "n_pixels": int(macro["total"]),  # type: ignore[index]
        "miou_macro": macro["miou"],  # type: ignore[index]
        "miou_weighted": macro["miou_weighted"],  # type: ignore[index]
        "miou_invfq_weighted": macro["miou_inv_freq"],  # type: ignore[index]
        "kappa": macro["kappa"],  # type: ignore[index]
        "mcc": macro["mcc"],  # type: ignore[index]
        "glacier_iou_binary": binary_glacier_iou(confusion),
    }

    f1_values = [
        entry["f1"] for entry in per_class if isinstance(entry["f1"], float)  # type: ignore[union-attr,index]
    ]
    finite_f1 = [v for v in f1_values if np.isfinite(v)]
    row["macro_f1"] = float(np.mean(finite_f1)) if finite_f1 else float("nan")

    shares = _class_pixel_shares(confusion)
    worst_name, worst_value = None, np.inf
    for entry in per_class:  # type: ignore[union-attr]
        name = str(entry["class_name"])  # type: ignore[index]
        iou = entry["iou"]  # type: ignore[index]
        row[f"iou_{name}"] = iou
        row[f"pixel_share_pct_{name}"] = round(shares.get(name, 0.0), 3)
        # A class absent from this scene is not evidence of a weakness, so only
        # classes that are actually present compete for "worst".
        if (
            isinstance(iou, float)
            and np.isfinite(iou)
            and shares.get(name, 0.0) > 0.0
            and iou < worst_value
        ):
            worst_name, worst_value = name, iou
    row["worst_class"] = worst_name if worst_name is not None else ""

    return row


def scene_performance_table(scenes: SceneConfusions) -> list[dict[str, Any]]:
    """One row per evaluated scene, with its own metrics and class mix.

    This is the finest grain available: the scene is the unit the corpus is
    annotated in, and a per-tile table would be both enormous and dominated by
    tiles too small to carry all four classes.

    Rows are ordered by `(split, glacier_id, year, scene_id)` so the table is
    stable across runs and diffable.
    """
    rows: list[dict[str, Any]] = []
    for position, identity in enumerate(scenes.identities):
        confusion = scenes.matrices[position]
        if int(confusion.sum()) == 0:
            continue  # never evaluated: no tiles, or every pixel was no-data
        rows.append(
            {
                "split": identity.split,
                "glacier_id": identity.glacier_id,
                "year": identity.year,
                "scene_id": identity.scene_id,
                "sensor": sensor_display_name(identity.sensor),
                "sensor_key": identity.sensor,
                **_metrics_row(confusion),
            }
        )

    rows.sort(key=lambda r: (r["split"], r["glacier_id"], r["year"], r["scene_id"]))
    return rows


def _group(
    scenes: SceneConfusions,
    key,
) -> dict[Any, tuple[np.ndarray, list[int]]]:
    """Sum confusion matrices into groups defined by `key(identity)`.

    Groups carry *positions* into the stack rather than identity objects, so a
    later regrouping (per-glacier within a sensor, say) can go back to the
    matrices without matching on object identity.
    """
    grouped: dict[Any, tuple[np.ndarray, list[int]]] = {}
    for position, identity in enumerate(scenes.identities):
        confusion = scenes.matrices[position]
        if int(confusion.sum()) == 0:
            continue
        group_key = key(identity)
        if group_key in grouped:
            total, members = grouped[group_key]
            grouped[group_key] = (total + confusion, members + [position])
        else:
            grouped[group_key] = (confusion.copy(), [position])
    return grouped


def glacier_performance_table(scenes: SceneConfusions) -> list[dict[str, Any]]:
    """One row per (glacier, sensor, split).

    Split by sensor as well as glacier because the same glacier imaged by
    Landsat 5 and by Sentinel-2 is two different measurement problems; merging
    them would hide a sensor-specific failure inside a glacier-level average.
    """
    grouped = _group(
        scenes, lambda i: (i.split, i.glacier_id, sensor_display_name(i.sensor))
    )

    rows: list[dict[str, Any]] = []
    for (split, glacier_id, sensor), (confusion, positions) in grouped.items():
        years = sorted({scenes.identities[p].year for p in positions})
        rows.append(
            {
                "split": split,
                "glacier_id": glacier_id,
                "sensor": sensor,
                "n_scenes": len(positions),
                "years": ",".join(str(year) for year in years),
                **_metrics_row(confusion),
            }
        )

    rows.sort(key=lambda r: (r["split"], r["glacier_id"], r["sensor"]))
    return rows


def _glacier_weighted_miou(
    scenes: SceneConfusions,
    positions: Sequence[int],
) -> float:
    """Mean of per-glacier mIoUs over the given scene positions.

    Every glacier counts once. See the module docstring for why this differs
    from, and complements, the pixel-pooled score.
    """
    by_glacier: dict[str, np.ndarray] = {}
    for position in positions:
        confusion = scenes.matrices[position]
        if int(confusion.sum()) == 0:
            continue
        glacier_id = scenes.identities[position].glacier_id
        if glacier_id in by_glacier:
            by_glacier[glacier_id] = by_glacier[glacier_id] + confusion
        else:
            by_glacier[glacier_id] = confusion.copy()

    values = []
    for confusion in by_glacier.values():
        miou = compute_confusion_metrics(confusion)["macro"]["miou"]  # type: ignore[index]
        if isinstance(miou, float) and np.isfinite(miou):
            values.append(miou)
    return float(np.mean(values)) if values else float("nan")


def sensor_performance_table(scenes: SceneConfusions) -> list[dict[str, Any]]:
    """One row per (sensor, split), rolled up from the per-glacier rows.

    Carries both the pixel-pooled mIoU and the glacier-weighted one, because a
    sensor that images one huge glacier well and twenty small ones badly scores
    very differently under the two.
    """
    grouped = _group(scenes, lambda i: (i.split, sensor_display_name(i.sensor)))

    rows: list[dict[str, Any]] = []
    for (split, sensor), (confusion, positions) in grouped.items():
        glaciers = {scenes.identities[p].glacier_id for p in positions}
        rows.append(
            {
                "split": split,
                "sensor": sensor,
                "n_glaciers": len(glaciers),
                "n_scenes": len(positions),
                "miou_glacier_weighted": round(
                    _glacier_weighted_miou(scenes, positions), 3
                ),
                **_metrics_row(confusion),
            }
        )

    rows.sort(key=lambda r: (r["split"], r["sensor"]))
    return rows


# Metrics whose spread across sensors is worth reporting. Everything here is
# "higher is better", which is what makes `worst_sensor` mean the minimum.
ROBUSTNESS_METRICS: tuple[str, ...] = (
    "miou_macro",
    "miou_weighted",
    "miou_invfq_weighted",
    "miou_glacier_weighted",
    "kappa",
    "mcc",
    "macro_f1",
    "glacier_iou_binary",
    *(f"iou_{name}" for name in CLASS_NAMES),
)


def sensor_robustness_table(
    sensor_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = ROBUSTNESS_METRICS,
) -> list[dict[str, Any]]:
    """Per split, the spread of each metric across sensors.

    A model is only usable across the archive if it works on every sensor, and
    an average over sensors cannot show that one of them fails. This reports
    min, max, gap (max - min), population standard deviation, and the name of
    the sensor holding the minimum.

    Args:
        sensor_rows: the output of `sensor_performance_table`.
    """
    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for row in sensor_rows:
        by_split.setdefault(str(row["split"]), []).append(row)

    out: list[dict[str, Any]] = []
    for split in sorted(by_split):
        rows = by_split[split]
        sensors = sorted({str(r["sensor"]) for r in rows})
        summary: dict[str, Any] = {
            "split": split,
            "n_sensors": len(sensors),
            "sensors": ", ".join(sensors),
        }

        for metric in metrics:
            pairs = [
                (str(r["sensor"]), float(r[metric]))
                for r in rows
                if isinstance(r.get(metric), (int, float))
                and np.isfinite(float(r[metric]))
            ]
            if not pairs:
                summary[f"{metric}_min"] = float("nan")
                summary[f"{metric}_max"] = float("nan")
                summary[f"{metric}_gap"] = float("nan")
                summary[f"{metric}_std"] = float("nan")
                summary[f"{metric}_worst_sensor"] = ""
                continue

            values = np.array([v for _, v in pairs], dtype=np.float64)
            worst_sensor = min(pairs, key=lambda kv: (kv[1], kv[0]))[0]
            summary[f"{metric}_min"] = round(float(values.min()), 4)
            summary[f"{metric}_max"] = round(float(values.max()), 4)
            summary[f"{metric}_gap"] = round(float(values.max() - values.min()), 4)
            summary[f"{metric}_std"] = round(float(values.std()), 4)
            summary[f"{metric}_worst_sensor"] = worst_sensor

        out.append(summary)

    return out


def class_iou_by_sensor_table(
    sensor_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """A narrow (split, sensor) x per-class-IoU view.

    The same numbers the per-sensor table already carries, reduced to just the
    class IoUs. Kept as its own table because "which class does each sensor
    struggle with" is a question asked on its own, and answering it from the
    wide table means ignoring twenty other columns.
    """
    return [
        {
            "split": row["split"],
            "sensor": row["sensor"],
            "n_glaciers": row.get("n_glaciers", 0),
            "n_scenes": row.get("n_scenes", 0),
            **{f"iou_{name}": row.get(f"iou_{name}") for name in CLASS_NAMES},
        }
        for row in sensor_rows
    ]


# -- worst-case shortlists ---------------------------------------------------
#
# The per-scene and per-glacier tables already hold every number these need.
# What they do not do is answer "where should I go look first", because they are
# sorted for diffability (by identity) rather than by how badly the model did.
# A shortlist sorted by mIoU, truncated, and carrying the identity columns
# needed to pull the actual imagery is a different question asked of the same
# rows, so it is derived from them rather than recomputed.

#: How many rows each worst-case shortlist keeps. Twenty is enough to see
#: whether the failures share a sensor, a region or a year, and short enough to
#: open every one of them.
DEFAULT_WORST_COUNT: int = 20

# The identity and diagnostic columns a worst-case row carries, in the order a
# reader wants them: what the scene is, then how the model did, then what the
# scene was made of. `pixel_share_pct_Cloud` is the cloud fraction *as the
# annotation sees it* — the corpus carries no separate scene-level cloud
# metadata, and the annotated Cloud share is the quantity that actually bears on
# whether a low score is the model's fault or the scene's.
_WORST_SCENE_COLUMNS: tuple[str, ...] = (
    "split",
    "glacier_id",
    "year",
    "scene_id",
    "sensor",
    "sensor_key",
    "miou_macro",
    "worst_class",
    *(f"iou_{name}" for name in CLASS_NAMES),
    *(f"pixel_share_pct_{name}" for name in CLASS_NAMES),
    "n_pixels",
    "kappa",
)

_WORST_GLACIER_COLUMNS: tuple[str, ...] = (
    "split",
    "glacier_id",
    "sensor",
    "n_scenes",
    "years",
    "miou_macro",
    "worst_class",
    *(f"iou_{name}" for name in CLASS_NAMES),
    *(f"pixel_share_pct_{name}" for name in CLASS_NAMES),
    "n_pixels",
    "kappa",
)


def _worst_rows(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    count: int,
    splits: Optional[Sequence[str]],
    tie_break: Sequence[str],
) -> list[dict[str, Any]]:
    """The `count` lowest-mIoU rows per split, projected onto `columns`.

    Ranked within each split rather than globally: train and test mIoUs are not
    comparable, so a global sort would fill the list with whichever partition
    happens to score lower overall and hide the worst cases in the others.

    Rows whose mIoU is not finite are dropped. A non-finite mIoU means no class
    was both present and predicted — an empty or degenerate matrix — which is a
    data condition, not a model failure, and putting it at the top of a
    "worst performing" list is actively misleading.
    """
    count = max(0, int(count))
    if count == 0:
        return []

    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get("miou_macro")
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            continue
        split = str(row.get("split", ""))
        if splits is not None and split not in splits:
            continue
        by_split.setdefault(split, []).append(row)

    out: list[dict[str, Any]] = []
    for split in sorted(by_split):
        ranked = sorted(
            by_split[split],
            key=lambda r: (
                float(r["miou_macro"]),
                *(str(r.get(field, "")) for field in tie_break),
            ),
        )
        for rank, row in enumerate(ranked[:count], start=1):
            out.append(
                {"rank": rank, **{name: row.get(name) for name in columns}}
            )
    return out


def worst_scenes_table(
    scenes: SceneConfusions,
    count: int = DEFAULT_WORST_COUNT,
    splits: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """The lowest-scoring scenes per split, with enough identity to find them.

    Every column here already exists in `scene_performance_table`; this is that
    table sorted by mIoU, cut to `count` rows per split, and narrowed to the
    fields needed to go and look at the imagery. Keeping it as a separate export
    rather than expecting a reader to sort the full table is the whole point —
    the shortlist is what gets acted on.

    Args:
        count: rows kept per split.
        splits: partitions to include, or None for all of them.
    """
    return _worst_rows(
        scene_performance_table(scenes),
        _WORST_SCENE_COLUMNS,
        count,
        splits,
        tie_break=("glacier_id", "year", "scene_id"),
    )


def worst_glaciers_table(
    scenes: SceneConfusions,
    count: int = DEFAULT_WORST_COUNT,
    splits: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """The lowest-scoring (glacier, sensor) pairs per split.

    Grouped exactly as `glacier_performance_table` groups — by sensor as well as
    glacier — because a glacier that one sensor handles well and another fails
    on is the case worth finding, and merging the two would average it away.
    """
    return _worst_rows(
        glacier_performance_table(scenes),
        _WORST_GLACIER_COLUMNS,
        count,
        splits,
        tie_break=("glacier_id", "sensor"),
    )


def build_breakdown(
    scenes: SceneConfusions,
    worst_count: int = DEFAULT_WORST_COUNT,
) -> dict[str, list[dict[str, Any]]]:
    """Every breakdown table, from one per-scene confusion stack.

    Returned as a dict of table name to rows so the export layer writes them
    without needing to know how any of them is derived.

    Args:
        worst_count: rows per split in the two worst-case shortlists. Zero
            omits them entirely.
    """
    sensor_rows = sensor_performance_table(scenes)
    tables = {
        "scene_performance": scene_performance_table(scenes),
        "glacier_performance": glacier_performance_table(scenes),
        "sensor_performance": sensor_rows,
        "sensor_robustness": sensor_robustness_table(sensor_rows),
        "class_iou_by_sensor": class_iou_by_sensor_table(sensor_rows),
    }
    if worst_count > 0:
        tables["worst_scenes"] = worst_scenes_table(scenes, worst_count)
        tables["worst_glaciers"] = worst_glaciers_table(scenes, worst_count)
    return tables
