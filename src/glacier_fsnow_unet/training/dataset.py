"""Scene discovery, per-scene tensor caching, tiling, and the stratified split.

This module turns a directory tree of annotated glacier scenes into the three
tile lists a training run consumes. Four things about it are load-bearing.

**Determinism is enforced by sorting, not by seeding alone.** Seeding a
shuffle only helps if the sequence being shuffled is itself in a fixed order.
Scene discovery therefore sorts by `(glacier_id, year, scene_id)`, and — more
importantly — every later decision that picks "a glacier" out of a collection
iterates a sorted sequence, never a `set`. Python randomises string hashing per
process, so iterating a set of glacier IDs and taking the first match yields a
different glacier on every run even with the RNG seeded. That is not a
theoretical concern: it made the split irreproducible across processes. Sets
are used for membership tests only; ordering always comes from a sorted list.
See `docs/decisions/training_reproducibility_audit.md`.

**The split is glacier-level.** Every scene from a glacier lands in exactly one
partition. Scenes from the same glacier taken in different years look far more
alike than scenes from different glaciers, so a scene-level split lets the
network recognise glaciers it has already seen and reports that as
generalisation.

**Four constraints are enforced in sequence** after the initial size-stratified
assignment: each partition should contain every sensor, every class, and every
sensor-class combination, and training keeps priority over validation and test
when they conflict. Each repair pass moves whole glaciers, never scenes, so
glacier-level exclusivity survives.

**Cached tensors carry a signature.** A per-scene cache is reused only when the
hash of its inputs — the input rasters with their sizes and mtimes, the requested
feature list, and a schema version — matches. A cache whose signature does not
match is rebuilt, never silently reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .config import (
    BALANCED_SENSORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    RAW_TO_TRAIN_LABEL,
    TrainingConfig,
)
from .metrics import boundary_band

__all__ = [
    "SceneRecord",
    "PreparedData",
    "scan_scenes",
    "load_scene_arrays",
    "scene_cache_signature",
    "collect_tiles",
    "split_scenes",
    "compute_mean_std",
    "compute_class_counts",
    "prepare_data",
]

SCENE_CACHE_DIRNAME = "ia_cache_all"
SCENE_CACHE_FILENAME = "scene_features.npz"
SCENE_CACHE_META = "scene_features.json"

# Bump when the stored array layout changes, so old caches are rebuilt rather
# than misread.
SCENE_CACHE_SCHEMA_VERSION = 1

LANDSAT7_GAP_START, LANDSAT7_GAP_END = 2003, 2012


@dataclass(frozen=True)
class SceneRecord:
    """One annotated scene and the path to its cached tensors."""

    scene_dir: Path
    glacier_id: str
    year: int
    scene_id: str
    sensor: str
    cache_path: Path

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.glacier_id, self.year, self.scene_id)


@dataclass
class PreparedData:
    """Everything a training run needs from the data pipeline.

    `features` holds one `(C, H, W)` float16 array per scene. float16 is what
    the cache stores and what the corpus fits in; conversion to float32 happens
    per tile, on the small array, in the Dataset.
    """

    features: list[np.ndarray]
    labels: list[np.ndarray]
    boundary_labels: list[np.ndarray]
    scene_records: list[SceneRecord]
    mean: np.ndarray
    std: np.ndarray
    sensor_norm_stats: Optional[dict[str, dict[str, np.ndarray]]]
    tiles_train: list[tuple[int, int, int, int]]
    tiles_val: list[tuple[int, int, int, int]]
    tiles_test: list[tuple[int, int, int, int]]
    class_counts: dict[int, int]
    patch_size: int
    stride: int
    scene_context: list[Optional[Mapping[str, float]]] = field(default_factory=list)

    @property
    def scene_sensors(self) -> list[str]:
        return [record.sensor for record in self.scene_records]

    @property
    def glacier_ids(self) -> list[str]:
        return [record.glacier_id for record in self.scene_records]


# -- discovery ---------------------------------------------------------------


def _parse_identity(scene_dir: Path, root: Path) -> tuple[str, int, str]:
    """Recover `(glacier_id, year, scene_id)` from `<root>/<glacier>/<year>/<scene>`."""
    try:
        parts = scene_dir.resolve().relative_to(root.resolve()).parts
        if len(parts) >= 3 and parts[1].isdigit() and len(parts[1]) == 4:
            return parts[0], int(parts[1]), parts[2]
    except (ValueError, OSError):
        pass

    year = next(
        (int(p) for p in scene_dir.parts if p.isdigit() and len(p) == 4),
        0,
    )
    return scene_dir.parent.parent.name, year, scene_dir.name


def scan_scenes(
    train_root: Path,
    skip_landsat7_gap: bool = True,
    cache_dirname: str = SCENE_CACHE_DIRNAME,
    cache_filename: str = SCENE_CACHE_FILENAME,
) -> list[SceneRecord]:
    """Find every cached scene under `train_root`, in a stable order.

    `os.walk` does not guarantee an order, so the result is sorted by
    `(glacier_id, year, scene_id)` before returning. Every downstream scene
    index derives from this ordering, so sorting here is what makes the whole
    pipeline reproducible.

    Landsat 7 scenes from 2003-2012 are dropped when `skip_landsat7_gap` is
    set: the Scan Line Corrector failure leaves wedge-shaped voids that are not
    representative of the sensor.
    """
    train_root = Path(train_root)
    records: list[SceneRecord] = []

    for current, _dirs, files in os.walk(train_root):
        if cache_filename not in files:
            continue
        cache_path = Path(current) / cache_filename
        scene_dir = (
            cache_path.parent.parent
            if cache_path.parent.name == cache_dirname
            else cache_path.parent
        )

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                sensor = str(data["sensor"].item()).strip().lower()
                glacier_id = str(data["id_glims"].item())
                year = int(data["year"].item())
                scene_id = str(data["scene_id"].item())
        except (OSError, KeyError, ValueError):
            glacier_id, year, scene_id = _parse_identity(scene_dir, train_root)
            sensor = "unknown"

        if not glacier_id:
            glacier_id, year, scene_id = _parse_identity(scene_dir, train_root)

        if (
            skip_landsat7_gap
            and sensor == "landsat7"
            and LANDSAT7_GAP_START <= year <= LANDSAT7_GAP_END
        ):
            continue

        records.append(
            SceneRecord(
                scene_dir=scene_dir,
                glacier_id=glacier_id,
                year=year,
                scene_id=scene_id,
                sensor=sensor,
                cache_path=cache_path,
            )
        )

    records.sort(key=lambda record: record.sort_key)
    return records


def scene_cache_signature(
    scene_dir: Path,
    source_paths: Sequence[Path],
    feature_names: Sequence[str],
    schema_version: int = SCENE_CACHE_SCHEMA_VERSION,
) -> str:
    """Hash of everything that determines a cached scene tensor's contents.

    Covers the scene path, the exact feature list *in order*, the schema
    version, and each input raster's size and mtime. Anything that changes the
    stored arrays changes this hash; anything that does not — patch size,
    split ratios, learning rate — is deliberately absent, because it does not
    affect the per-scene tensors and including it would force needless rebuilds.
    """
    payload = {
        "schema_version": int(schema_version),
        "scene": str(scene_dir),
        "features": list(feature_names),
        "sources": [],
    }
    for path in sorted(source_paths, key=str):
        try:
            stat = Path(path).stat()
            payload["sources"].append([str(path), int(stat.st_size), int(stat.st_mtime)])
        except OSError:
            payload["sources"].append([str(path), -1, -1])

    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def cache_is_current(
    cache_path: Path,
    meta_path: Path,
    expected_signature: str,
) -> bool:
    """True only when a cache exists and its recorded signature matches.

    A mismatch means rebuild. There is no "close enough" branch: silently
    reusing a cache built under a different configuration is how a run ends up
    training on the wrong features or the wrong split.
    """
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("signature") == expected_signature


def load_scene_arrays(
    record: SceneRecord,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Read one scene's features and labels, selecting the requested channels.

    Features stay float16 — the dtype they are stored in. Converting the whole
    corpus to float32 up front would quadruple its resident size for no gain,
    since normalisation happens per tile.

    Arrays are copied out of the `NpzFile` and the handle closed immediately,
    so decompression buffers do not accumulate across hundreds of scenes.
    """
    with np.load(record.cache_path, allow_pickle=False) as data:
        features = np.asarray(data["features"])
        labels = np.asarray(data["labels"], dtype=np.uint8)
        stored_names = (
            [str(name) for name in np.asarray(data["feature_names"]).tolist()]
            if "feature_names" in data
            else None
        )

    if stored_names is None:
        if features.shape[0] != len(feature_names):
            raise ValueError(
                f"{record.cache_path}: cache has {features.shape[0]} channels but "
                f"{len(feature_names)} features were requested, and the cache "
                f"records no channel names to match them by."
            )
        selected = features
    else:
        index_of = {name: i for i, name in enumerate(stored_names)}
        missing = [name for name in feature_names if name not in index_of]
        if missing:
            raise ValueError(f"{record.cache_path}: cache is missing features {missing}")
        selected = features[[index_of[name] for name in feature_names], ...]

    if selected.dtype != np.float16:
        selected = selected.astype(np.float16)

    # Any pixel with a non-finite value in any channel is unusable. Zero the
    # channels so downstream arithmetic stays finite, and mark the label
    # no-data so it never contributes to a loss or a metric.
    invalid = ~np.all(np.isfinite(selected.astype(np.float32)), axis=0)
    if np.any(invalid):
        selected = selected.copy()
        labels = labels.copy()
        selected[:, invalid] = 0
        labels[invalid] = IGNORE_INDEX

    return np.ascontiguousarray(selected), labels


def remap_raw_labels(raw: np.ndarray) -> np.ndarray:
    """Map raw annotation values 1..4 to training labels 0..3, 0 to no-data."""
    out = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw_value, target in RAW_TO_TRAIN_LABEL.items():
        out[raw == raw_value] = target
    return out


# -- tiling ------------------------------------------------------------------


def collect_tiles(
    labels: Sequence[np.ndarray],
    patch_size: int,
    stride: int,
    min_valid: int,
    include_rotations: bool = False,
) -> list[tuple[int, int, int, int]]:
    """Enumerate `(scene_index, y, x, rotation)` for every usable tile.

    A tile is kept when it holds at least `min_valid` annotated pixels; a
    window that is almost all no-data teaches nothing and would distort the
    class weights.

    The valid-pixel count per window comes from a summed-area table, so the
    whole scene is one vectorised pass rather than a Python loop per window.
    Rotation is recorded as part of the tile identity, not applied here, so the
    enumeration stays deterministic and the rotated view is materialised only
    when the tile is actually fetched.
    """
    tiles: list[tuple[int, int, int, int]] = []
    rotations = (0, 1, 2, 3) if include_rotations else (0,)
    patch = int(patch_size)

    for scene_index, label in enumerate(labels):
        height, width = label.shape
        if height < patch or width < patch:
            continue

        integral = np.pad((label != IGNORE_INDEX).astype(np.int32), ((1, 0), (1, 0)))
        np.cumsum(integral, axis=0, out=integral)
        np.cumsum(integral, axis=1, out=integral)
        window_counts = (
            integral[patch:, patch:]
            - integral[:-patch, patch:]
            - integral[patch:, :-patch]
            + integral[:-patch, :-patch]
        )

        ys = range(0, height - patch + 1, stride)
        xs = range(0, width - patch + 1, stride)
        for y0 in ys:
            row = window_counts[y0]
            for x0 in xs:
                if row[x0] < min_valid:
                    continue
                tiles.extend((scene_index, y0, x0, rot) for rot in rotations)

    return tiles


# -- normalisation and counts ------------------------------------------------


def compute_mean_std(
    features: Sequence[np.ndarray],
    scene_indices: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation over the given scenes.

    Computed from the training scenes only — deriving normalisation statistics
    from validation or test data leaks information about them into training.

    Accumulates channel sums and sums of squares per scene, then combines. The
    original walked every pixel in Python via Welford's algorithm, which for
    this corpus is on the order of a hundred million interpreted iterations;
    this is the same quantity in a few vectorised reductions. Accumulators are
    float64 so the sum of squares does not lose precision across scenes.
    """
    indices = (
        list(range(len(features))) if scene_indices is None else list(scene_indices)
    )
    if not indices:
        raise ValueError("cannot compute normalisation statistics from zero scenes")

    channels = features[indices[0]].shape[0]
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    count = np.zeros(channels, dtype=np.int64)

    for index in indices:
        flat = features[index].reshape(channels, -1).astype(np.float64)
        finite = np.isfinite(flat)
        safe = np.where(finite, flat, 0.0)
        total += safe.sum(axis=1)
        total_sq += (safe * safe).sum(axis=1)
        count += finite.sum(axis=1)

    if count.sum() == 0:
        raise ValueError("no finite feature values available for normalisation")

    safe_count = np.maximum(count, 1)
    mean = total / safe_count
    variance = np.maximum(total_sq / safe_count - mean * mean, 0.0)
    # Match the sample (n-1) convention, guarding single-value channels.
    correction = safe_count / np.maximum(safe_count - 1, 1)
    std = np.sqrt(np.maximum(variance * correction, 1e-8))

    return mean.astype(np.float32), std.astype(np.float32)


def compute_mean_std_by_sensor(
    features: Sequence[np.ndarray],
    scene_sensors: Sequence[str],
    scene_indices: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    """Per-sensor normalisation statistics, over the training scenes only.

    Used when per-sensor normalisation is enabled. The reference configuration
    disables it in favour of one global set of statistics.
    """
    by_sensor: dict[str, list[int]] = {}
    for index in scene_indices:
        if index < len(scene_sensors):
            by_sensor.setdefault(str(scene_sensors[index]).lower(), []).append(index)

    stats: dict[str, dict[str, np.ndarray]] = {}
    for sensor in sorted(by_sensor):
        try:
            mean, std = compute_mean_std(features, by_sensor[sensor])
            stats[sensor] = {"mean": mean, "std": std}
        except ValueError:
            continue
    return stats


def compute_class_counts(
    labels: Sequence[np.ndarray],
    scene_indices: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> dict[int, int]:
    """Pixel count per class over the given scenes, for class weighting."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for index in scene_indices:
        label = labels[index]
        valid = label[label != IGNORE_INDEX]
        if valid.size:
            counts += np.bincount(valid, minlength=num_classes)[:num_classes]
    return {k: int(counts[k]) for k in range(num_classes)}


def scene_class_counts(
    label: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    valid = label[label != IGNORE_INDEX]
    if valid.size == 0:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(valid, minlength=num_classes)[:num_classes].astype(np.int64)


# -- the split ---------------------------------------------------------------
#
# Every function below takes and returns *sorted lists* of glacier IDs, never
# sets, for the reason given in the module docstring. Membership tests may use
# a set built locally; iteration order never comes from one.


def group_scenes_by_glacier(records: Sequence[SceneRecord]) -> dict[str, list[int]]:
    """Map glacier ID to its scene indices.

    Insertion order follows the sorted scene list, and the keys are sorted
    again wherever they drive a decision, so the grouping is stable.
    """
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.glacier_id, []).append(index)
    return groups


def _stratify_by_size(
    glacier_groups: Mapping[str, list[int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Optional[tuple[list[str], list[str], list[str]]]:
    """Split glaciers into train/val/test, stratified by scene count.

    Three strata — 1 scene, 2-4 scenes, 5+ scenes — each split independently,
    so a partition cannot end up holding all the heavily-imaged glaciers. With
    206 glaciers averaging 2.6 scenes each, an unstratified draw can easily put
    most of the large ones on one side.

    Each stratum's glacier list is sorted before shuffling, so a given seed
    always produces the same assignment.
    """
    if not glacier_groups:
        return None

    import random

    sizes = {gid: len(indices) for gid, indices in glacier_groups.items()}
    strata = {
        "small": sorted(g for g, n in sizes.items() if n <= 1),
        "medium": sorted(g for g, n in sizes.items() if 2 <= n <= 4),
        "large": sorted(g for g, n in sizes.items() if n >= 5),
    }

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    rng = random.Random(seed)

    for name in ("small", "medium", "large"):
        members = list(strata[name])
        if not members:
            continue
        rng.shuffle(members)
        n = len(members)

        n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = int(round(n * val_ratio)) if val_ratio > 0 else 0

        # Never empty a stratum's training share.
        if n_test + n_val >= n:
            n_val = max(0, n - 1 - n_test)
        if n_test + n_val >= n:
            n_test = max(0, n - 1 - n_val)
        if val_ratio > 0 and n_val == 0 and n > n_test + 1:
            n_val = 1
        if test_ratio > 0 and n_test == 0 and n > n_val + 1:
            n_test = 1

        test.extend(members[:n_test])
        val.extend(members[n_test : n_test + n_val])
        train.extend(members[n_test + n_val :])

    if not train:
        return None
    return sorted(train), sorted(val), sorted(test)


def _greedy_class_balanced_split(
    glacier_groups: Mapping[str, list[int]],
    labels: Sequence[np.ndarray],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Optional[tuple[list[str], list[str], list[str]]]:
    """Fallback split: greedily pick glaciers to hit target class-pixel counts.

    Used when size stratification is disabled or infeasible. Repeatedly takes
    the glacier that brings the partition's per-class pixel totals closest to
    their targets. Candidates are iterated in sorted order and ties broken by
    glacier ID, so the greedy choice is deterministic.
    """
    import random

    glacier_ids = sorted(glacier_groups)
    if len(glacier_ids) <= 1:
        return None

    per_glacier = {
        gid: sum(
            (scene_class_counts(labels[i]) for i in glacier_groups[gid]),
            start=np.zeros(NUM_CLASSES, dtype=np.int64),
        )
        for gid in glacier_ids
    }
    corpus_total = sum(per_glacier.values(), start=np.zeros(NUM_CLASSES, dtype=np.int64))

    count = len(glacier_ids)
    n_val = int(round(count * val_ratio)) if val_ratio > 0 else 0
    n_test = int(round(count * test_ratio)) if test_ratio > 0 else 0
    if n_val + n_test > count - 1:
        n_test = max(0, min(n_test, count - 1 - n_val))
        n_val = max(0, min(n_val, count - 1 - n_test))

    rng = random.Random(seed)
    remaining = list(glacier_ids)
    rng.shuffle(remaining)

    if int(corpus_total.sum()) == 0:
        return (
            sorted(remaining[n_val + n_test :]),
            sorted(remaining[:n_val]),
            sorted(remaining[n_val : n_val + n_test]),
        )

    def _distance(counts: np.ndarray, target: np.ndarray) -> float:
        active = target > 0
        if not np.any(active):
            return 0.0
        return float(np.sum(np.abs(counts - target) / np.maximum(target, 1.0)))

    def _take(target: np.ndarray, how_many: int) -> list[str]:
        chosen: list[str] = []
        running = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _ in range(how_many):
            if not remaining:
                break
            # Sorted iteration plus an explicit ID tie-break: no reliance on
            # container order anywhere in the choice.
            best = min(
                sorted(remaining),
                key=lambda gid: (_distance(running + per_glacier[gid], target), gid),
            )
            chosen.append(best)
            running += per_glacier[best]
            remaining.remove(best)
        return chosen

    val = _take(corpus_total.astype(np.float64) * val_ratio, n_val)
    test = _take(corpus_total.astype(np.float64) * test_ratio, n_test)
    return sorted(remaining), sorted(val), sorted(test)


def _enforce_sensor_coverage(
    glacier_groups: Mapping[str, list[int]],
    scene_sensors: Sequence[str],
    train: list[str],
    val: list[str],
    test: list[str],
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Move glaciers so each partition sees every available sensor.

    Modifies the three lists in place, keeping them sorted. Training has
    priority: if a sensor is missing there it is recovered from validation or
    test, because a sensor absent from training is one the model can never
    learn, whereas a sensor absent from validation only costs a measurement.

    Donors are chosen from a sorted list, so the same glacier moves on every
    run given the same inputs.
    """
    sensors_per_glacier = {
        gid: {scene_sensors[i] for i in indices}
        for gid, indices in glacier_groups.items()
    }
    available = sorted(
        {
            sensor
            for sensors in sensors_per_glacier.values()
            for sensor in sensors
            if sensor in BALANCED_SENSORS
        }
    )

    def _has(partition: list[str], sensor: str) -> bool:
        return any(sensor in sensors_per_glacier.get(gid, ()) for gid in partition)

    def _move(sensor: str, donor: list[str], destination: list[str]) -> bool:
        if len(donor) <= 1:
            return False
        for gid in sorted(donor):  # sorted: never hash order
            if sensor in sensors_per_glacier.get(gid, ()):
                donor.remove(gid)
                destination.append(gid)
                destination.sort()
                return True
        return False

    if val_ratio > 0:
        for sensor in available:
            if not _has(val, sensor):
                _move(sensor, train, val)
    if test_ratio > 0:
        for sensor in available:
            if not _has(test, sensor):
                _move(sensor, train, test)

    for sensor in available:
        if not _has(train, sensor):
            _move(sensor, val, train) or _move(sensor, test, train)


def _enforce_class_coverage(
    glacier_groups: Mapping[str, list[int]],
    labels: Sequence[np.ndarray],
    train: list[str],
    val: list[str],
    test: list[str],
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Move glaciers so each partition contains all four classes.

    When a class is missing, the donor is the glacier holding the most pixels
    of it — giving the receiving partition enough of the class to measure,
    rather than a token handful. Ties break on glacier ID.
    """
    counts_per_glacier = {
        gid: sum(
            (scene_class_counts(labels[i]) for i in indices),
            start=np.zeros(NUM_CLASSES, dtype=np.int64),
        )
        for gid, indices in glacier_groups.items()
    }

    def _has(partition: list[str], class_id: int) -> bool:
        return any(counts_per_glacier[gid][class_id] > 0 for gid in partition)

    def _move(class_id: int, donor: list[str], destination: list[str]) -> bool:
        if len(donor) <= 1:
            return False
        candidates = [gid for gid in sorted(donor) if counts_per_glacier[gid][class_id] > 0]
        if not candidates:
            return False
        best = max(candidates, key=lambda gid: (int(counts_per_glacier[gid][class_id]), gid))
        donor.remove(best)
        destination.append(best)
        destination.sort()
        return True

    if val_ratio > 0:
        for class_id in range(NUM_CLASSES):
            if not _has(val, class_id):
                _move(class_id, train, val)
    if test_ratio > 0:
        for class_id in range(NUM_CLASSES):
            if not _has(test, class_id):
                _move(class_id, train, test)

    for class_id in range(NUM_CLASSES):
        if not _has(train, class_id):
            _move(class_id, val, train) or _move(class_id, test, train)


def _enforce_sensor_class_coverage(
    glacier_groups: Mapping[str, list[int]],
    labels: Sequence[np.ndarray],
    scene_sensors: Sequence[str],
    train: list[str],
    val: list[str],
    test: list[str],
    val_ratio: float,
    test_ratio: float,
) -> list[str]:
    """Try to give each partition every sensor-class combination that exists.

    Finer than the two passes above: a partition can hold Landsat 5 scenes and
    Ice pixels while containing no Landsat 5 *Ice* pixel. Donor selection
    prefers a glacier whose removal costs the donor partition the fewest
    sensor-class combinations, so repairing one gap does not open another.

    Some combinations are structurally unsatisfiable — the corpus contains
    very few Landsat 7 Ice pixels in total, so they cannot appear in all three
    partitions at once. Returns a list of human-readable warnings rather than
    failing.
    """
    glacier_pairs: dict[str, set[tuple[str, int]]] = {}
    for gid, indices in glacier_groups.items():
        pairs: set[tuple[str, int]] = set()
        for i in indices:
            sensor = scene_sensors[i]
            present = np.nonzero(scene_class_counts(labels[i]))[0]
            pairs.update((sensor, int(c)) for c in present)
        glacier_pairs[gid] = pairs

    all_pairs = sorted(
        {
            pair
            for pairs in glacier_pairs.values()
            for pair in pairs
            if pair[0] in BALANCED_SENSORS
        }
    )

    def _has(partition: list[str], pair: tuple[str, int]) -> bool:
        return any(pair in glacier_pairs[gid] for gid in partition)

    def _move(pair: tuple[str, int], donor: list[str], destination: list[str]) -> bool:
        if len(donor) <= 1:
            return False
        candidates = [gid for gid in sorted(donor) if pair in glacier_pairs[gid]]
        if not candidates:
            return False

        def _cost(gid: str) -> tuple[int, int, str]:
            retained: set[tuple[str, int]] = set()
            for other in donor:
                if other != gid:
                    retained |= glacier_pairs[other]
            lost = len({p for g in donor for p in glacier_pairs[g]} - retained)
            return (lost, len(glacier_pairs[gid]), gid)

        best = min(candidates, key=_cost)
        if _cost(best)[0] > 0 and len(candidates) == len(donor):
            return False
        donor.remove(best)
        destination.append(best)
        destination.sort()
        return True

    if val_ratio > 0:
        for pair in all_pairs:
            if not _has(val, pair):
                _move(pair, train, val)
    if test_ratio > 0:
        for pair in all_pairs:
            if not _has(test, pair):
                _move(pair, train, test)

    for pair in all_pairs:
        if not _has(train, pair):
            _move(pair, val, train) or _move(pair, test, train)

    warnings: list[str] = []
    for name, partition, enabled in (
        ("train", train, True),
        ("val", val, val_ratio > 0),
        ("test", test, test_ratio > 0),
    ):
        if not enabled:
            continue
        missing = [pair for pair in all_pairs if not _has(partition, pair)]
        warnings.extend(
            f"{name} has no {sensor} pixels of class {CLASS_NAMES[class_id]}"
            for sensor, class_id in missing
        )
    return warnings


def split_scenes(
    records: Sequence[SceneRecord],
    labels: Sequence[np.ndarray],
    config: TrainingConfig,
) -> tuple[list[int], list[int], list[int], list[str]]:
    """Partition scenes into train/val/test indices.

    With `split_glacier` set (the reference behaviour) the split is by glacier
    and every scene from a glacier stays together. Otherwise scenes are split
    directly, which is faster to satisfy but lets the model see the same
    glacier on both sides — the leakage the glacier-level protocol exists to
    prevent.

    Deterministic given `(records, labels, config)`: sorted inputs, a seeded
    RNG, and no reliance on set iteration order anywhere.

    Returns `(train, val, test, warnings)` with the index lists sorted.
    """
    scene_sensors = [record.sensor for record in records]
    warnings: list[str] = []

    if config.split_glacier:
        glacier_groups = group_scenes_by_glacier(records)

        assignment = None
        if config.glacier_size_stratify:
            assignment = _stratify_by_size(
                glacier_groups, config.val_ratio, config.test_ratio, config.split_seed
            )
        if assignment is None:
            assignment = _greedy_class_balanced_split(
                glacier_groups,
                labels,
                config.val_ratio,
                config.test_ratio,
                config.split_seed,
            )

        if assignment is not None:
            train_g, val_g, test_g = (list(part) for part in assignment)

            _enforce_sensor_coverage(
                glacier_groups, scene_sensors, train_g, val_g, test_g,
                config.val_ratio, config.test_ratio,
            )
            _enforce_class_coverage(
                glacier_groups, labels, train_g, val_g, test_g,
                config.val_ratio, config.test_ratio,
            )
            warnings = _enforce_sensor_class_coverage(
                glacier_groups, labels, scene_sensors, train_g, val_g, test_g,
                config.val_ratio, config.test_ratio,
            )

            overlap = (set(train_g) & set(val_g)) | (set(train_g) & set(test_g)) | (
                set(val_g) & set(test_g)
            )
            if overlap:
                raise RuntimeError(
                    f"glacier-level split is not exclusive: {sorted(overlap)} appear "
                    f"in more than one partition"
                )

            expand = lambda gids: sorted(i for g in gids for i in glacier_groups[g])
            train_i, val_i, test_i = expand(train_g), expand(val_g), expand(test_g)
            warnings.extend(
                _proportion_warnings(train_i, val_i, test_i, records, config)
            )
            return train_i, val_i, test_i, warnings

        warnings.append(
            "glacier-level split was not feasible (too few glaciers); "
            "fell back to a scene-level split, which permits within-glacier leakage"
        )

    return (*_split_scene_level(records, labels, config), warnings)


def _proportion_warnings(
    train: Sequence[int],
    val: Sequence[int],
    test: Sequence[int],
    records: Sequence[SceneRecord],
    config: TrainingConfig,
) -> list[str]:
    """Flag a split whose scene proportions drift far from the configured ones.

    The split targets glacier *counts*, but what a partition actually measures
    is its *scenes*. Glaciers differ enormously in scene count -- one glacier
    in this corpus carries 120 scenes against a median of 2 -- so a
    size-stratified draw that looks correct by glacier count can still put a
    third of all scenes into a test partition meant to hold a tenth.

    That is not leakage and the partition stays exclusive, but it makes the
    resulting score hard to compare against a run with different proportions.
    Worth saying out loud rather than leaving to be noticed downstream.
    """
    total = len(train) + len(val) + len(test)
    if total == 0:
        return []

    messages: list[str] = []
    for name, indices, target in (
        ("val", val, config.val_ratio),
        ("test", test, config.test_ratio),
    ):
        if target <= 0:
            continue
        actual = len(indices) / total
        # Trigger only on a substantial drift, so ordinary rounding is quiet.
        if actual > 2 * target or actual < 0.4 * target:
            messages.append(
                f"{name} holds {actual:.1%} of scenes but was configured for "
                f"{target:.0%}; one or a few glaciers with many scenes dominate "
                f"this partition, so its score is not comparable to a run with "
                f"different proportions (try another split_seed)"
            )

    # Name a single glacier that dominates a held-out partition.
    for name, indices in (("val", val), ("test", test)):
        if len(indices) < 5:
            continue
        counts: dict[str, int] = {}
        for i in indices:
            counts[records[i].glacier_id] = counts.get(records[i].glacier_id, 0) + 1
        glacier, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        share = count / len(indices)
        if share > 0.5:
            messages.append(
                f"{name} is {share:.0%} scenes from the single glacier {glacier}; "
                f"its score largely measures that one glacier"
            )

    return messages


def _split_scene_level(
    records: Sequence[SceneRecord],
    labels: Sequence[np.ndarray],
    config: TrainingConfig,
) -> tuple[list[int], list[int], list[int]]:
    """Scene-level split, optionally stratified per sensor.

    Retained as a configurable alternative and as the fallback when there are
    too few glaciers to split by glacier. It permits scenes from one glacier to
    land in different partitions.
    """
    import random

    rng = random.Random(config.split_seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    if config.split_by_sensor:
        by_sensor: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            by_sensor.setdefault(record.sensor, []).append(index)
        buckets = [sorted(by_sensor[sensor]) for sensor in sorted(by_sensor)]
    else:
        buckets = [list(range(len(records)))]

    for bucket in buckets:
        members = list(bucket)
        rng.shuffle(members)
        n = len(members)
        n_test = int(round(n * config.test_ratio)) if config.test_ratio > 0 else 0
        n_val = int(round(n * config.val_ratio)) if config.val_ratio > 0 else 0
        if n_test + n_val >= n:
            n_val = max(0, n - 1 - n_test)
        if n_test + n_val >= n:
            n_test = max(0, n - 1 - n_val)

        test.extend(members[:n_test])
        val.extend(members[n_test : n_test + n_val])
        train.extend(members[n_test + n_val :])

    return sorted(train), sorted(val), sorted(test)


# -- top level ---------------------------------------------------------------


def prepare_data(
    config: TrainingConfig,
    records: Optional[Sequence[SceneRecord]] = None,
) -> PreparedData:
    """Load the corpus, tile it, split it, and derive normalisation statistics.

    Order matters here. The split is computed first so that normalisation
    statistics and class weights can be derived from the *training* scenes
    only; computing them over the whole corpus would leak validation and test
    distributions into training.
    """
    if records is None:
        records = scan_scenes(
            config.train_root, skip_landsat7_gap=config.skip_landsat7_2003_2012
        )
    records = list(records)
    if config.max_scenes is not None:
        records = records[: int(config.max_scenes)]
    if not records:
        raise RuntimeError(f"no cached training scenes found under {config.train_root}")

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    boundary_labels: list[np.ndarray] = []
    kept: list[SceneRecord] = []

    for record in records:
        feature_array, label_array = load_scene_arrays(record, config.features)
        interior, boundary = boundary_band(label_array, config.ignore_boundary)
        if np.all(interior == IGNORE_INDEX):
            continue  # nothing annotated survives the boundary exclusion
        features.append(feature_array)
        labels.append(interior)
        boundary_labels.append(boundary)
        kept.append(record)

    if not kept:
        raise RuntimeError("every scene was empty after applying the boundary band")

    train_idx, val_idx, test_idx, warnings = split_scenes(kept, labels, config)
    for warning in warnings:
        print(f"[split] {warning}")

    tiles = collect_tiles(
        labels,
        patch_size=config.patch_size,
        stride=config.stride,
        min_valid=config.min_valid,
        include_rotations=config.rotations,
    )
    if not tiles:
        raise RuntimeError(
            f"no tile has at least {config.min_valid} annotated pixels at "
            f"patch_size={config.patch_size}, stride={config.stride}"
        )

    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    tiles_train = [t for t in tiles if t[0] in train_set]
    tiles_val = [t for t in tiles if t[0] in val_set]
    tiles_test = [t for t in tiles if t[0] in test_set]

    stats_indices = train_idx or list(range(len(features)))
    mean, std = compute_mean_std(features, stats_indices)

    per_sensor_norm, _, _ = config.sensor_adaptation
    sensor_norm_stats = (
        compute_mean_std_by_sensor(
            features, [r.sensor for r in kept], stats_indices
        )
        or None
        if per_sensor_norm
        else None
    )

    return PreparedData(
        features=features,
        labels=labels,
        boundary_labels=boundary_labels,
        scene_records=kept,
        mean=mean,
        std=std,
        sensor_norm_stats=sensor_norm_stats,
        tiles_train=tiles_train,
        tiles_val=tiles_val,
        tiles_test=tiles_test,
        class_counts=compute_class_counts(labels, stats_indices),
        patch_size=config.patch_size,
        stride=config.stride,
        scene_context=[None] * len(kept),
    )
