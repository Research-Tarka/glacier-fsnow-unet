"""PyTorch Dataset and samplers over the prepared tile lists.

`TileDataset` serves one 48x48 patch per index. It is deliberately free of
randomness: given an index it slices, optionally rotates by a fixed amount
recorded in the tile tuple, normalises, and returns. All the sampling
randomness lives in the samplers, which run in the main process. That keeps
worker processes from needing to agree on an RNG state, and means a batch is
determined entirely by the sampler's output.

Two samplers implement alternative epoch constructions, both off in the
reference configuration:

`SensorBalancedBatchSampler` gives every sensor a fixed quota in every batch,
so a corpus that is half Sentinel-2 does not produce batches that are half
Sentinel-2. This matters most for per-sensor BatchNorm, where a sensor absent
from a batch gets no gradient for its normalisation parameters.

`GlacierBalancedSampler` guarantees at least one tile from every glacier per
epoch, so small glaciers contributing a handful of tiles are not effectively
dropped.

Memory: scene features stay float16 for the whole corpus and are converted to
float32 per tile. On this corpus that is roughly 130 MB resident instead of
520 MB, and the conversion happens on a 48x48 window rather than a full scene.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, Sampler

from .config import IGNORE_INDEX, SENSOR_TO_IDX

__all__ = [
    "TileDataset",
    "GlacierBalancedSampler",
    "SensorBalancedBatchSampler",
    "build_glacier_tile_index",
    "build_sensor_tile_index",
    "worker_init_fn",
]

Tile = tuple[int, int, int, int]


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own derived seed.

    Torch already assigns each worker a distinct `torch.initial_seed()`, but
    NumPy's and Python's global RNGs are inherited identically by every worker
    on fork/spawn. Left alone, any future use of `np.random` inside the dataset
    would draw the same sequence in all eight workers.

    Nothing in `TileDataset` currently uses those RNGs, so this is a guard
    against a later change rather than a fix for a present bug — but it is the
    kind of bug that produces silently correlated batches and no error.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class TileDataset(Dataset):
    """Serves normalised `(features, labels, context)` triples for tiles.

    Args:
        tiles: `(scene_index, y, x, rotation)` tuples.
        features: one `(C, H, W)` float16 array per scene.
        labels: one `(H, W)` uint8 array per scene, `IGNORE_INDEX` for no-data.
        mean, std: global per-channel normalisation statistics.
        patch_size: tile side length.
        sensor_norm_stats: optional per-sensor statistics overriding the
            global ones, keyed by sensor name.
        scene_sensors: sensor per scene, for per-sensor normalisation and for
            the sensor index carried in the context vector.
        scene_context: optional per-scene `(x, y, year_norm, area_norm)` for
            FiLM conditioning.
        return_tile_meta: also return `(scene_index, y, x, rotation)`, needed
            when boundary metrics have to re-extract the matching band.
        return_scene_index: also return the scene index, for per-glacier
            metric aggregation.
    """

    def __init__(
        self,
        tiles: Sequence[Tile],
        features: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        mean: np.ndarray,
        std: np.ndarray,
        patch_size: int,
        sensor_norm_stats: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
        scene_sensors: Optional[Sequence[str]] = None,
        scene_context: Optional[Sequence[Optional[Mapping[str, float]]]] = None,
        return_tile_meta: bool = False,
        return_scene_index: bool = False,
    ) -> None:
        self.tiles = list(tiles)
        self.features = features
        self.labels = labels
        self.patch_size = int(patch_size)
        self.return_tile_meta = bool(return_tile_meta)
        self.return_scene_index = bool(return_scene_index)

        self.mean = np.asarray(mean, dtype=np.float32)
        # Fold the epsilon in once here rather than per fetch.
        self.inv_std = (1.0 / (np.asarray(std, dtype=np.float32) + 1e-6)).astype(np.float32)

        self.scene_sensors = [str(s or "unknown").lower() for s in (scene_sensors or [])]
        self.scene_sensor_ids = [
            SENSOR_TO_IDX.get(sensor, 0) for sensor in self.scene_sensors
        ]
        self.scene_context = list(scene_context or [])

        # Pre-resolve per-sensor statistics into per-scene arrays so the hot
        # path is an array lookup, not a dict lookup plus shape checks.
        self._scene_mean: Optional[list[np.ndarray]] = None
        self._scene_inv_std: Optional[list[np.ndarray]] = None
        if sensor_norm_stats:
            self._scene_mean = []
            self._scene_inv_std = []
            for sensor in self.scene_sensors:
                stats = sensor_norm_stats.get(sensor)
                if (
                    isinstance(stats, Mapping)
                    and np.shape(stats.get("mean")) == self.mean.shape
                ):
                    self._scene_mean.append(np.asarray(stats["mean"], dtype=np.float32))
                    self._scene_inv_std.append(
                        (1.0 / (np.asarray(stats["std"], dtype=np.float32) + 1e-6)).astype(
                            np.float32
                        )
                    )
                else:
                    self._scene_mean.append(self.mean)
                    self._scene_inv_std.append(self.inv_std)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int):
        scene_index, y0, x0, rotation = self.tiles[index]
        patch = self.patch_size

        feature = self.features[scene_index][:, y0 : y0 + patch, x0 : x0 + patch]
        label = self.labels[scene_index][y0 : y0 + patch, x0 : x0 + patch]

        # Tiles at a scene's right or bottom edge can fall short; pad features
        # with zeros and labels with no-data so the short region is ignored.
        if feature.shape[1] < patch or feature.shape[2] < patch:
            padded = np.zeros((feature.shape[0], patch, patch), dtype=feature.dtype)
            padded[:, : feature.shape[1], : feature.shape[2]] = feature
            feature = padded
            padded_label = np.full((patch, patch), IGNORE_INDEX, dtype=label.dtype)
            padded_label[: label.shape[0], : label.shape[1]] = label
            label = padded_label

        if rotation:
            feature = np.rot90(feature, rotation, axes=(1, 2))
            label = np.rot90(label, rotation, axes=(0, 1))

        mean = self.mean
        inv_std = self.inv_std
        if self._scene_mean is not None and scene_index < len(self._scene_mean):
            mean = self._scene_mean[scene_index]
            inv_std = self._scene_inv_std[scene_index]

        # float16 -> float32 happens here, on one 48x48 tile rather than on the
        # whole corpus. Multiplying by a precomputed reciprocal avoids a
        # per-element divide.
        normalised = (feature.astype(np.float32) - mean[:, None, None]) * inv_std[:, None, None]
        np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(np.ascontiguousarray(normalised))
        y = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))

        context = self._context_for(scene_index)

        if self.return_tile_meta:
            return x, y, context, int(scene_index), int(y0), int(x0), int(rotation)
        if self.return_scene_index:
            return x, y, context, int(scene_index)
        return x, y, context

    def _context_for(self, scene_index: int) -> torch.Tensor:
        """Build the `(x, y, year, area, sensor_id)` conditioning vector.

        The sensor index rides along as a fifth element so a single tensor
        carries everything the model's optional conditioning needs; the model
        splits it back off.
        """
        values = [0.0, 0.0, 0.0, 0.0]
        if scene_index < len(self.scene_context):
            meta = self.scene_context[scene_index]
            if meta is not None:
                values = [
                    float(meta.get("x", 0.0)),
                    float(meta.get("y", 0.0)),
                    float(meta.get("year_norm", 0.0)),
                    float(meta.get("area_norm", 0.0)),
                ]
        if scene_index < len(self.scene_sensor_ids):
            values.append(float(self.scene_sensor_ids[scene_index]))
        return torch.tensor(values, dtype=torch.float32)


def build_glacier_tile_index(
    tiles: Sequence[Tile],
    glacier_ids: Sequence[str],
) -> dict[str, list[int]]:
    """Map glacier ID to the positions of its tiles within `tiles`."""
    index: dict[str, list[int]] = {}
    for position, tile in enumerate(tiles):
        scene_index = int(tile[0])
        gid = (
            glacier_ids[scene_index]
            if scene_index < len(glacier_ids)
            else f"scene_{scene_index}"
        )
        index.setdefault(gid or f"scene_{scene_index}", []).append(position)
    return index


def build_sensor_tile_index(
    tiles: Sequence[Tile],
    scene_sensors: Sequence[str],
) -> dict[int, list[int]]:
    """Map sensor index to the positions of its tiles within `tiles`."""
    index: dict[int, list[int]] = {}
    for position, tile in enumerate(tiles):
        scene_index = int(tile[0])
        sensor = (
            scene_sensors[scene_index] if scene_index < len(scene_sensors) else "landsat"
        )
        index.setdefault(SENSOR_TO_IDX.get(str(sensor).lower(), 0), []).append(position)
    return {sid: positions for sid, positions in index.items() if positions}


class GlacierBalancedSampler(Sampler[int]):
    """Draws one tile from every glacier per epoch, then fills the rest randomly.

    Without this, glaciers contributing a handful of tiles are effectively
    invisible next to large glaciers contributing thousands. Guaranteeing one
    tile each puts a floor on every glacier's per-epoch contribution.

    Reseeded per epoch from `seed + epoch`, so epochs differ from each other
    but a whole run repeats exactly.
    """

    def __init__(
        self,
        glacier_to_tiles: Mapping[str, Sequence[int]],
        num_samples: int,
        seed: Optional[int] = None,
    ) -> None:
        # Sorted keys: the per-glacier draw order must not depend on hash order.
        self.glacier_to_tiles = {
            gid: list(tiles)
            for gid, tiles in sorted(glacier_to_tiles.items())
            if tiles
        }
        self.num_samples = int(num_samples)
        self.seed = int(seed) if seed is not None else None
        self._epoch = 0
        self._all_tiles = [t for tiles in self.glacier_to_tiles.values() for t in tiles]

    def __iter__(self) -> Iterator[int]:
        if not self._all_tiles:
            return iter(())

        rng = random if self.seed is None else random.Random(self.seed + self._epoch)
        if self.seed is not None:
            self._epoch += 1

        chosen = [rng.choice(tiles) for tiles in self.glacier_to_tiles.values()]

        shortfall = self.num_samples - len(chosen)
        if shortfall > 0:
            chosen.extend(rng.choice(self._all_tiles) for _ in range(shortfall))
        else:
            chosen = chosen[: self.num_samples]

        rng.shuffle(chosen)
        return iter(chosen)

    def __len__(self) -> int:
        return self.num_samples


class SensorBalancedBatchSampler(BatchSampler):
    """Builds batches with a near-equal share of every sensor.

    Each batch reserves `batch_size // n_sensors` slots per sensor, with the
    remainder distributed to the first few. Sensors are drawn from their own
    shuffled pools, which are reshuffled and reused when exhausted, so a sensor
    with few tiles is oversampled rather than dropped once its pool runs out.

    With `glacier_sensor_coverage` supplied, one anchor tile per
    (glacier, sensor) group is placed at the head of that sensor's pool, so
    every combination is visited at least once per epoch before any repetition.
    """

    def __init__(
        self,
        sensor_to_tiles: Mapping[int, Sequence[int]],
        batch_size: int,
        drop_last: bool = False,
        seed: Optional[int] = None,
        glacier_sensor_coverage: Optional[Mapping[tuple[str, str], Sequence[int]]] = None,
    ) -> None:
        self.sensor_to_tiles = {
            int(sid): list(tiles)
            for sid, tiles in sorted(sensor_to_tiles.items())
            if tiles
        }
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed) if seed is not None else None
        self._epoch = 0
        self._total = sum(len(tiles) for tiles in self.sensor_to_tiles.values())
        self._active = sorted(self.sensor_to_tiles)

        self._coverage: dict[int, list[list[int]]] = {}
        if glacier_sensor_coverage:
            for (_gid, sensor_name), tiles in sorted(glacier_sensor_coverage.items()):
                sid = SENSOR_TO_IDX.get(str(sensor_name).lower(), 0)
                if sid not in self.sensor_to_tiles:
                    continue
                allowed = set(self.sensor_to_tiles[sid])
                usable = [t for t in tiles if t in allowed]
                if usable:
                    self._coverage.setdefault(sid, []).append(usable)

    def __len__(self) -> int:
        if self._total <= 0:
            return 0
        if self.drop_last:
            return self._total // self.batch_size
        return int(math.ceil(self._total / self.batch_size))

    def _quotas(self) -> dict[int, int]:
        """Per-sensor slots in one batch, distributing the remainder."""
        n = len(self._active)
        if n <= 0:
            return {}
        base, remainder = divmod(self.batch_size, n)
        return {sid: base + (1 if i < remainder else 0) for i, sid in enumerate(self._active)}

    def __iter__(self) -> Iterator[list[int]]:
        if self._total <= 0 or not self._active:
            return iter(())

        rng = random if self.seed is None else random.Random(self.seed + self._epoch)
        if self.seed is not None:
            self._epoch += 1

        pools: dict[int, list[int]] = {}
        cursors: dict[int, int] = {}
        for sid in self._active:
            tiles = list(self.sensor_to_tiles[sid])
            anchors = [rng.choice(group) for group in self._coverage.get(sid, ())]
            if anchors:
                anchor_set = set(anchors)
                rest = [t for t in tiles if t not in anchor_set]
                rng.shuffle(rest)
                pools[sid] = anchors + rest
            else:
                rng.shuffle(tiles)
                pools[sid] = tiles
            cursors[sid] = 0

        quotas = self._quotas()
        batches: list[list[int]] = []
        for _ in range(len(self)):
            batch: list[int] = []
            for sid in self._active:
                need = quotas.get(sid, 0)
                if need <= 0:
                    continue
                pool = pools[sid]
                cursor = cursors[sid]
                for _slot in range(need):
                    if cursor >= len(pool):
                        rng.shuffle(pool)
                        cursor = 0
                    batch.append(pool[cursor])
                    cursor += 1
                cursors[sid] = cursor

            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                continue
            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]
            if batch:
                batches.append(batch)

        return iter(batches)
