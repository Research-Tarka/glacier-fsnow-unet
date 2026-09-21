"""Balanced distribution of glaciers across N processing batches (splits).

Purpose
-------
The scene-download and inference stages are run as N parallel batches, each
against its own Google Earth Engine project and output root. This module
decides which glacier goes in which batch, balancing both the **glacier count**
and the **total glacier area** across batches.

Inputs
------
- The glacier registry (stage 1), or an existing on-disk split layout.

Outputs
-------
- A split assignment table (``glims_id`` -> split index), optionally applied by
  moving the per-glacier directories on disk.

Algorithm
---------
Longest Processing Time (LPT) greedy: sort glaciers by descending area, then
assign each in turn to the currently least-loaded batch. LPT is the classic
4/3-approximation for makespan on identical machines, and area is the best
available proxy for per-glacier processing cost (scene footprint and pixel
count both scale with it).

"Least loaded" uses a combined score that balances both criteria at once::

    score = n_glaciers + total_area / mean_glacier_area

The second term expresses area in units of "average glaciers", so a batch that
is light on count but heavy on area is not treated as free capacity. Without
that normalisation, pure-area LPT would pile many tiny glaciers into one batch
(equal area, far more per-glacier overhead), and pure-count LPT would put all
the large glaciers together.

Design note (ROADMAP decision #5)
---------------------------------
The number of batches is a **parameter** here (``config.scene_download.splits``
determines it) rather than a hardcoded constant, so it can be tuned to however
many parallel GEE projects/output roots are actually available. Every
glacier's area is read from the registry Parquet in a single load, rather than
opening each glacier's ``metadata.json`` from disk individually, and the
filesystem is only touched when actually moving directories.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SplitLoad:
    """Running load of one batch during LPT assignment."""

    index: int
    name: str
    path: Optional[Path] = None
    count: int = 0
    total_area: float = 0.0
    glacier_ids: list[str] = field(default_factory=list)

    def score(self, mean_area: float) -> float:
        """Combined count + area-normalised load (lower is less loaded)."""
        area_term = (self.total_area / mean_area) if mean_area > 0 else 0.0
        return self.count + area_term

    def add(self, glims_id: str, area: float) -> None:
        self.count += 1
        self.total_area += float(area)
        self.glacier_ids.append(glims_id)


def assign_lpt(
    glacier_ids: Sequence[str],
    areas: Sequence[float],
    n_splits: int,
    split_names: Optional[Sequence[str]] = None,
    initial_loads: Optional[Sequence[SplitLoad]] = None,
) -> list[SplitLoad]:
    """Assign glaciers to ``n_splits`` batches by descending-area LPT.

    ``initial_loads`` lets an incremental run start from the batches' existing
    occupancy, so newly added glaciers land where they even things out rather
    than being distributed as if the batches were empty.

    Ties in score are broken by split index, making the assignment fully
    deterministic for a given input ordering.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if len(glacier_ids) != len(areas):
        raise ValueError(
            f"glacier_ids and areas must be the same length "
            f"({len(glacier_ids)} vs {len(areas)})"
        )

    if initial_loads is not None:
        loads = list(initial_loads)
        if len(loads) != n_splits:
            raise ValueError(
                f"initial_loads has {len(loads)} entries but n_splits is {n_splits}"
            )
    else:
        names = list(split_names) if split_names else [f"Split{i + 1}" for i in range(n_splits)]
        loads = [SplitLoad(index=i, name=names[i]) for i in range(n_splits)]

    area_arr = np.asarray(areas, dtype=float)
    if area_arr.size == 0:
        return loads

    # Mean over the glaciers being assigned, used to normalize the area term
    # of the load score (see module docstring).
    mean_area = float(area_arr.mean()) if area_arr.size else 1.0

    # Descending area (LPT). argsort on the negated array is a stable
    # descending sort, so equal areas keep their input order.
    order = np.argsort(-area_arr, kind="stable")

    for idx in order:
        target = min(loads, key=lambda s: (s.score(mean_area), s.index))
        target.add(str(glacier_ids[idx]), float(area_arr[idx]))

    return loads


def assignment_table(loads: Sequence[SplitLoad]) -> pd.DataFrame:
    """Flatten LPT loads into a ``glims_id`` -> split assignment table."""
    rows = []
    for load in loads:
        for gid in load.glacier_ids:
            rows.append(
                {"glims_id": gid, "split_index": load.index, "split_name": load.name}
            )
    return pd.DataFrame(rows, columns=["glims_id", "split_index", "split_name"])


def balance_report(loads: Sequence[SplitLoad]) -> pd.DataFrame:
    """Per-split count/area summary, for logging and for balance assertions."""
    return pd.DataFrame(
        [
            {
                "split_index": load.index,
                "split_name": load.name,
                "n_glaciers": load.count,
                "total_area_km2": load.total_area,
            }
            for load in loads
        ]
    ).sort_values("split_index", ignore_index=True)


def balance_metrics(loads: Sequence[SplitLoad]) -> dict:
    """Spread statistics used to judge (and test) balance quality."""
    counts = np.array([load.count for load in loads], dtype=float)
    areas = np.array([load.total_area for load in loads], dtype=float)

    def _spread(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "range": 0.0, "rel_range": 0.0}
        mean = float(values.mean())
        value_range = float(values.max() - values.min())
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": mean,
            "range": value_range,
            "rel_range": (value_range / mean) if mean > 0 else 0.0,
        }

    return {"count": _spread(counts), "area_km2": _spread(areas)}


def read_area_from_metadata(glacier_dir: Path) -> float:
    """Read ``metrics.area_km2`` from a glacier directory's metadata.json."""
    meta = glacier_dir / "metadata.json"
    if not meta.exists():
        return 0.0
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return float(data.get("metrics", {}).get("area_km2", 0.0))
    except Exception:
        return 0.0


def inventory_splits(split_dirs: Sequence[Path]) -> dict[str, Path]:
    """Return ``{glims_id: split_dir}`` for every glacier already on disk."""
    existing: dict[str, Path] = {}
    for split_dir in split_dirs:
        if not split_dir.exists():
            continue
        for sub in split_dir.iterdir():
            if sub.is_dir():
                existing[sub.name] = split_dir
    return existing


def loads_from_disk(split_dirs: Sequence[Path], areas: Optional[dict] = None) -> list[SplitLoad]:
    """Build the current per-split load from the on-disk layout."""
    areas = areas or {}
    loads: list[SplitLoad] = []
    for i, split_dir in enumerate(split_dirs):
        load = SplitLoad(index=i, name=split_dir.name, path=split_dir)
        if split_dir.exists():
            for sub in sorted(split_dir.iterdir()):
                if not sub.is_dir():
                    continue
                area = areas.get(sub.name)
                if area is None:
                    area = read_area_from_metadata(sub)
                load.add(sub.name, float(area))
        loads.append(load)
    return loads


def move_glacier(src: Path, dst_dir: Path, dry_run: bool = False) -> None:
    """Move a glacier directory into ``dst_dir``, merging if it already exists.

    When the target already exists, only files missing from the target are
    copied across (never overwriting newer work already done in the target),
    and the source is then removed once nothing is left to move -- so
    re-running a split assignment after a partial download never destroys
    work already completed for a glacier at its new location.
    """
    target = dst_dir / src.name
    if not target.exists():
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst_dir))
        return

    for item in src.rglob("*"):
        destination = target / item.relative_to(src)
        if item.is_dir():
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)
    if not dry_run:
        shutil.rmtree(src)


def plan_moves(
    current: dict[str, Path],
    target: dict[str, Path],
) -> list[tuple[str, Path, Path]]:
    """Return ``(glims_id, src_split_dir, dst_split_dir)`` for glaciers that move."""
    moves = []
    for gid, src_dir in current.items():
        dst_dir = target.get(gid)
        if dst_dir is not None and Path(dst_dir) != Path(src_dir):
            moves.append((gid, Path(src_dir), Path(dst_dir)))
    return moves


def apply_moves(
    moves: Iterable[tuple[str, Path, Path]],
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Execute planned moves. Returns the number of failures."""
    errors = 0
    for gid, src_dir, dst_dir in moves:
        if verbose:
            print(f"  {gid}: {src_dir.name} -> {dst_dir.name}")
        try:
            move_glacier(src_dir / gid, dst_dir, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {gid}: {exc}")
            errors += 1
    return errors
