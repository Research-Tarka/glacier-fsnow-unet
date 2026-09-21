"""End-of-season snow fraction (F_snow).

Purpose
-------
Compute the paper's headline metric (Eq. 1/6, Section 4.7):

.. math::
    F_{snow}(g, t) = \\frac{S_{snow}}{S_{snow} + S_{ice}}

where ``S_snow`` and ``S_ice`` are the counts of Snow- and Ice-classified pixels
**within the VGS mask** of glacier ``g`` in year ``t``. Both counts exclude
Cloud pixels, so the denominator is the resolved (non-cloud, non-nodata)
glacier surface rather than the whole mask.

Validity gate
-------------
"A year is included in the F_snow time series only if fewer than 5% of VGS
pixels remain unresolved (persistently cloud-covered across all composited
acquisitions, or otherwise affected by residual noise or classification error)
after multi-date compositing."

Unresolved = Cloud or NoData inside the VGS.

Inputs
------
- An annual worst-state composite ``(H, W)`` of class codes.
- The glacier's VGS mask.

Outputs
-------
- :class:`FsnowResult` per glacier-year, or a tidy DataFrame for a time series.

Vectorization
-------------
Counting is done with boolean array reductions. :func:`fsnow_series` accepts a
stacked ``(T, H, W)`` array of annual composites and computes the whole time
series with reductions over the year axis -- no Python loop over years or
pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .classes import CLOUD, ICE, NODATA, OTHER, SNOW

#: Maximum share of VGS pixels allowed to remain unresolved (paper: 5%).
DEFAULT_MAX_UNRESOLVED_FRACTION = 0.05


@dataclass
class FsnowResult:
    """F_snow for one glacier-year, with the counts behind it."""

    year: Optional[int]
    fsnow: float
    snow_px: int
    ice_px: int
    cloud_px: int
    nodata_px: int
    other_px: int
    vgs_px: int
    unresolved_fraction: float
    valid: bool
    reason: str = "ok"

    @property
    def resolved_px(self) -> int:
        """Snow + Ice pixels, i.e. the denominator of F_snow."""
        return self.snow_px + self.ice_px


def compute_fsnow(
    state_map: np.ndarray,
    vgs_mask: np.ndarray,
    year: Optional[int] = None,
    max_unresolved_fraction: float = DEFAULT_MAX_UNRESOLVED_FRACTION,
) -> FsnowResult:
    """Compute F_snow for one glacier-year.

    Parameters
    ----------
    state_map
        Annual worst-state composite of class codes.
    vgs_mask
        Boolean VGS mask defining the spatial support.
    year
        Year label, carried through to the result.
    max_unresolved_fraction
        Validity gate: the year is rejected if at least this share of VGS pixels
        is Cloud or NoData (paper: 0.05).

    Returns
    -------
    FsnowResult
        ``valid=False`` (with ``fsnow=nan``) when the year fails the gate or
        has no resolved Snow/Ice pixels at all.
    """
    state = np.asarray(state_map, dtype=np.uint8)
    vgs = np.asarray(vgs_mask, dtype=bool)

    if state.shape != vgs.shape:
        raise ValueError(
            f"state_map {state.shape} and vgs_mask {vgs.shape} must have the same shape."
        )

    vgs_px = int(vgs.sum())
    if vgs_px == 0:
        return FsnowResult(
            year=year, fsnow=float("nan"), snow_px=0, ice_px=0, cloud_px=0,
            nodata_px=0, other_px=0, vgs_px=0, unresolved_fraction=1.0,
            valid=False, reason="empty VGS mask",
        )

    inside = state[vgs]
    snow_px = int((inside == SNOW).sum())
    ice_px = int((inside == ICE).sum())
    cloud_px = int((inside == CLOUD).sum())
    other_px = int((inside == OTHER).sum())
    nodata_px = int((inside == NODATA).sum())

    unresolved = (cloud_px + nodata_px) / vgs_px

    if unresolved >= max_unresolved_fraction:
        return FsnowResult(
            year=year, fsnow=float("nan"), snow_px=snow_px, ice_px=ice_px,
            cloud_px=cloud_px, nodata_px=nodata_px, other_px=other_px,
            vgs_px=vgs_px, unresolved_fraction=unresolved, valid=False,
            reason=(
                f"{unresolved:.1%} of VGS pixels unresolved "
                f"(threshold {max_unresolved_fraction:.0%})"
            ),
        )

    denominator = snow_px + ice_px
    if denominator == 0:
        return FsnowResult(
            year=year, fsnow=float("nan"), snow_px=snow_px, ice_px=ice_px,
            cloud_px=cloud_px, nodata_px=nodata_px, other_px=other_px,
            vgs_px=vgs_px, unresolved_fraction=unresolved, valid=False,
            reason="no Snow or Ice pixels inside the VGS",
        )

    return FsnowResult(
        year=year, fsnow=snow_px / denominator, snow_px=snow_px, ice_px=ice_px,
        cloud_px=cloud_px, nodata_px=nodata_px, other_px=other_px,
        vgs_px=vgs_px, unresolved_fraction=unresolved, valid=True,
    )


def fsnow_series(
    annual_states: Mapping[int, np.ndarray] | Iterable[tuple[int, np.ndarray]],
    vgs_mask: np.ndarray,
    glims_id: Optional[str] = None,
    max_unresolved_fraction: float = DEFAULT_MAX_UNRESOLVED_FRACTION,
) -> pd.DataFrame:
    """Compute the F_snow time series for one glacier as a tidy DataFrame.

    Columns: ``glims_id`` (if given), ``year``, ``fsnow``, ``snow_px``,
    ``ice_px``, ``cloud_px``, ``nodata_px``, ``other_px``, ``vgs_px``,
    ``unresolved_fraction``, ``valid``, ``reason``.
    """
    items = (
        sorted(annual_states.items())
        if isinstance(annual_states, Mapping)
        else sorted(annual_states, key=lambda kv: kv[0])
    )

    rows = []
    for year, state_map in items:
        result = compute_fsnow(
            state_map, vgs_mask, year=year,
            max_unresolved_fraction=max_unresolved_fraction,
        )
        row = {
            "year": result.year,
            "fsnow": result.fsnow,
            "snow_px": result.snow_px,
            "ice_px": result.ice_px,
            "cloud_px": result.cloud_px,
            "nodata_px": result.nodata_px,
            "other_px": result.other_px,
            "vgs_px": result.vgs_px,
            "unresolved_fraction": result.unresolved_fraction,
            "valid": result.valid,
            "reason": result.reason,
        }
        if glims_id is not None:
            row = {"glims_id": glims_id, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def fsnow_stack(
    states: np.ndarray,
    vgs_mask: np.ndarray,
    max_unresolved_fraction: float = DEFAULT_MAX_UNRESOLVED_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized F_snow over a stacked ``(T, H, W)`` array of annual composites.

    Computes the whole time series with reductions over the year axis, with no
    Python loop over years. Useful when many years are held in memory at once
    (e.g. a zarr time slab).

    Returns
    -------
    (fsnow, valid)
        ``fsnow`` is a float array of length T (NaN where invalid); ``valid`` is
        the matching boolean array.
    """
    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 3:
        raise ValueError(f"Expected a (T, H, W) stack, got shape {states.shape}")

    vgs = np.asarray(vgs_mask, dtype=bool)
    if states.shape[1:] != vgs.shape:
        raise ValueError(
            f"states {states.shape[1:]} and vgs_mask {vgs.shape} must have the same shape."
        )

    vgs_px = int(vgs.sum())
    if vgs_px == 0:
        n = states.shape[0]
        return np.full(n, np.nan), np.zeros(n, dtype=bool)

    inside = states[:, vgs]  # (T, n_vgs_px)
    snow = (inside == SNOW).sum(axis=1)
    ice = (inside == ICE).sum(axis=1)
    cloud = (inside == CLOUD).sum(axis=1)
    nodata = (inside == NODATA).sum(axis=1)

    unresolved = (cloud + nodata) / vgs_px
    denominator = snow + ice

    valid = (unresolved < max_unresolved_fraction) & (denominator > 0)

    fsnow = np.full(states.shape[0], np.nan, dtype=float)
    np.divide(snow, denominator, out=fsnow, where=valid)
    fsnow[~valid] = np.nan
    return fsnow, valid
