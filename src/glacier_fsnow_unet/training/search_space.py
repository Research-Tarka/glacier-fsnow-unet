"""Hyperparameter search-space sampling and normalisation.

A search space maps hyperparameter names to `SearchSpaceEntry` descriptors
(log-uniform, uniform, or categorical). This module turns those into concrete
values — either by direct sampling or by asking Optuna to suggest one — and
then sanitises the result.

Sanitisation is not cosmetic. Optuna proposes floats; several hyperparameters
must be positive integers, and `stride` must not exceed `patch_size` or the
tiler produces nothing. Clamping here means an out-of-range proposal costs a
clamp rather than a crashed trial partway through a sweep.

Only continuous hyperparameters are searched in the reference protocol —
learning rate, weight decay, dropout. The structural ones (base channels,
patch size, stride, batch size) are fixed a priori, because varying them
changes the receptive field and the tile population at the same time and makes
trials incomparable.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Optional, Sequence

from .config import SearchSpaceEntry

__all__ = [
    "sanitise_overrides",
    "sample_search_space",
    "suggest_with_optuna",
    "boundary_combinations",
]

# Hyperparameters that must be positive integers, with their floors.
_INTEGER_FLOORS: Mapping[str, int] = {
    "base_channels": 1,
    "patch_size": 8,
    "stride": 1,
    "batch_size": 1,
    "epochs": 1,
    "patience": 1,
    "min_valid": 0,
}

# Hyperparameters clamped into a range.
_FLOAT_BOUNDS: Mapping[str, tuple[float, float]] = {
    "lr": (1e-8, 1.0),
    "weight_decay": (0.0, 1.0),
    "dropout_p": (0.0, 0.95),
}


def _round_value(value: Any, digits: int = 4) -> Any:
    """Round a float for logging without destroying small magnitudes.

    A weight decay near 1e-5 rounded to four decimals becomes 0.0, which is a
    different hyperparameter. Values below 1e-3 keep ten decimals instead.
    """
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 10) if abs(value) < 1e-3 else round(value, digits)
    return value


def sanitise_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce sampled hyperparameters into legal values.

    Integers are rounded and floored, floats clamped, and the
    `stride <= patch_size` invariant is restored last so it holds regardless of
    the order the two were sampled in.
    """
    out: dict[str, Any] = dict(overrides or {})

    for name, floor in _INTEGER_FLOORS.items():
        if name in out:
            try:
                out[name] = max(floor, int(round(float(out[name]))))
            except (TypeError, ValueError):
                del out[name]

    for name, (low, high) in _FLOAT_BOUNDS.items():
        if name in out:
            try:
                out[name] = min(high, max(low, float(out[name])))
            except (TypeError, ValueError):
                del out[name]

    if "patch_size" in out and "stride" in out:
        out["stride"] = min(out["stride"], out["patch_size"])

    return out


def _sample_entry(entry: SearchSpaceEntry, rng: random.Random) -> Any:
    """Draw one value from a single hyperparameter's distribution."""
    kind = entry.type.strip().lower()

    if kind == "log_uniform":
        if not entry.bounds:
            raise ValueError("log_uniform search-space entry needs bounds")
        low, high = entry.bounds
        if low <= 0:
            raise ValueError(f"log_uniform lower bound must be positive, got {low}")
        return _round_value(10 ** rng.uniform(math.log10(low), math.log10(high)))

    if kind == "uniform":
        if not entry.bounds:
            raise ValueError("uniform search-space entry needs bounds")
        return _round_value(rng.uniform(*entry.bounds))

    if kind == "choice":
        if not entry.values:
            raise ValueError("choice search-space entry needs values")
        return _round_value(rng.choice(list(entry.values)))

    raise ValueError(f"unknown search-space entry type {entry.type!r}")


def sample_search_space(
    space: Optional[Mapping[str, SearchSpaceEntry]],
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Draw one hyperparameter set. Used for random search and for tests.

    Names are iterated in sorted order so a given seed reproduces the same
    draw regardless of how the mapping was built.
    """
    if not space:
        return {}
    rng = random.Random(seed)
    return sanitise_overrides(
        {name: _sample_entry(space[name], rng) for name in sorted(space)}
    )


def suggest_with_optuna(
    trial: Any,
    space: Optional[Mapping[str, SearchSpaceEntry]],
) -> dict[str, Any]:
    """Ask an Optuna trial for one value per hyperparameter.

    Sorted iteration keeps the parameter registration order stable, which
    matters for resuming a study from persistent storage.
    """
    if not space:
        return {}

    suggested: dict[str, Any] = {}
    for name in sorted(space):
        entry = space[name]
        kind = entry.type.strip().lower()
        if kind == "choice":
            if not entry.values:
                raise ValueError(f"choice entry {name!r} needs values")
            suggested[name] = trial.suggest_categorical(name, list(entry.values))
        elif kind in ("log_uniform", "uniform"):
            if not entry.bounds:
                raise ValueError(f"{kind} entry {name!r} needs bounds")
            low, high = entry.bounds
            suggested[name] = trial.suggest_float(
                name, float(low), float(high), log=(kind == "log_uniform")
            )
        else:
            raise ValueError(f"unknown search-space entry type {entry.type!r} for {name!r}")

    return sanitise_overrides(suggested)


def boundary_combinations(
    space: Optional[Mapping[str, SearchSpaceEntry]],
) -> list[dict[str, Any]]:
    """Every corner of the search space, as an exhaustive sanity sweep.

    With three hyperparameters this is 8 configurations. Useful to confirm the
    space is well-formed and that its extremes train at all before spending a
    50-trial budget inside it.
    """
    if not space:
        return []

    from itertools import product

    names = sorted(space)
    corners: list[Sequence[Any]] = []
    for name in names:
        entry = space[name]
        if entry.bounds:
            corners.append([_round_value(entry.bounds[0]), _round_value(entry.bounds[1])])
        elif entry.values:
            numeric = [v for v in entry.values if isinstance(v, (int, float))]
            corners.append(
                [min(numeric), max(numeric)]
                if numeric
                else [entry.values[0], entry.values[-1]]
            )
        else:
            return []

    return [
        sanitise_overrides(dict(zip(names, combination)))
        for combination in product(*corners)
    ]
