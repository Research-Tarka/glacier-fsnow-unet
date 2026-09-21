"""Climatology, anomalies, and priority-based merging of climate sources.

Purpose
-------
Turn per-source absolute annual values into the delivered feature parquets:
``clim_{var}`` / ``anom_{var}`` plus the mandatory traceability fields
``{var}_source`` / ``{var}_quality_flag`` / ``{var}_coverage_flag``
(DESIGN_FEATURES.md sections 1.2-1.4).

Inputs
------
- Per-source tidy frames: ``id_glims``, ``year``, and one column per variable.

Outputs
-------
- Merged frames carrying, for each variable, the climatology, the anomaly, and
  which source supplied each glacier-year.

Reference window
----------------
1991-2020 by default (DESIGN_FEATURES.md section 1.3). Climatology is the mean
over that window **per glacier**; the anomaly is the year's value minus that
glacier's climatology.

Priority
--------
Merging fills each glacier-year from the highest-priority source that has a
value: Daymet V4 > TerraClimate > ERA5-Land > ERA5 (see
``source_defs.SOURCE_PRIORITY``).

Vectorization
-------------
All operations are pandas groupby/where expressions over whole columns; there
is no Python loop over glaciers or years.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .source_defs import CLIMATOLOGY_REFERENCE, SOURCE_PRIORITY

#: Quality flags recorded per variable per glacier-year.
QUALITY_OBSERVED = "observed"      # value came from the source directly
QUALITY_FILLED = "filled"          # value came from a lower-priority source
QUALITY_MISSING = "missing"        # no source had a value

#: Coverage flags describing the climatology's support.
COVERAGE_FULL = "full"             # every reference year present
COVERAGE_PARTIAL = "partial"       # some reference years present
COVERAGE_NONE = "none"             # no reference year present

#: Minimum share of reference-window years needed for a "full" climatology.
FULL_COVERAGE_THRESHOLD = 0.8


def compute_climatology(
    frame: pd.DataFrame,
    variables: Sequence[str],
    reference: tuple[int, int] = CLIMATOLOGY_REFERENCE,
    group_key: str = "id_glims",
    year_key: str = "year",
) -> pd.DataFrame:
    """Per-glacier mean of each variable over the reference window.

    Returns one row per glacier with columns ``clim_{var}`` and
    ``{var}_coverage_flag``.
    """
    start, end = reference
    in_window = frame[frame[year_key].between(start, end)]
    n_reference_years = end - start + 1

    if in_window.empty:
        out = pd.DataFrame({group_key: frame[group_key].unique()})
        for var in variables:
            out[f"clim_{var}"] = np.nan
            out[f"{var}_coverage_flag"] = COVERAGE_NONE
        return out

    grouped = in_window.groupby(group_key)
    means = grouped[list(variables)].mean()
    counts = grouped[list(variables)].count()

    out = pd.DataFrame(index=means.index)
    for var in variables:
        out[f"clim_{var}"] = means[var]
        ratio = counts[var] / float(n_reference_years)
        out[f"{var}_coverage_flag"] = np.where(
            counts[var] == 0,
            COVERAGE_NONE,
            np.where(ratio >= FULL_COVERAGE_THRESHOLD, COVERAGE_FULL, COVERAGE_PARTIAL),
        )
    return out.reset_index()


def compute_anomalies(
    frame: pd.DataFrame,
    variables: Sequence[str],
    climatology: Optional[pd.DataFrame] = None,
    reference: tuple[int, int] = CLIMATOLOGY_REFERENCE,
    group_key: str = "id_glims",
    year_key: str = "year",
) -> pd.DataFrame:
    """Attach ``clim_{var}`` and ``anom_{var}`` to a per-glacier-year frame.

    The anomaly is the absolute value minus that glacier's climatology, so it
    is comparable across glaciers with very different absolute climates.
    """
    if climatology is None:
        climatology = compute_climatology(
            frame, variables, reference, group_key=group_key, year_key=year_key
        )

    out = frame.merge(climatology, on=group_key, how="left")
    for var in variables:
        out[f"anom_{var}"] = out[var] - out[f"clim_{var}"]
    return out


def merge_by_priority(
    sources: Mapping[str, pd.DataFrame],
    variable: str,
    priority: Sequence[str] = SOURCE_PRIORITY,
    group_key: str = "id_glims",
    year_key: str = "year",
) -> pd.DataFrame:
    """Fill one variable from the highest-priority source that has a value.

    Each source frame must carry ``group_key``, ``year_key``, and ``variable``.
    Sources absent from ``sources`` are skipped; sources present but lacking
    the column are skipped too.

    Returns
    -------
    pd.DataFrame
        ``group_key``, ``year_key``, ``variable``, ``{var}_source``,
        ``{var}_quality_flag``.
    """
    ordered = [
        name for name in priority if name in sources and variable in sources[name].columns
    ]
    if not ordered:
        return pd.DataFrame(
            columns=[group_key, year_key, variable, f"{variable}_source",
                     f"{variable}_quality_flag"]
        )

    # Union of all (glacier, year) keys across the contributing sources.
    index_frames = [
        sources[name][[group_key, year_key]].drop_duplicates() for name in ordered
    ]
    result = pd.concat(index_frames, ignore_index=True).drop_duplicates()
    result = result.sort_values([group_key, year_key], ignore_index=True)

    result[variable] = np.nan
    result[f"{variable}_source"] = pd.Series([None] * len(result), dtype=object)

    for rank, name in enumerate(ordered):
        candidate = (
            sources[name][[group_key, year_key, variable]]
            .dropna(subset=[variable])
            .drop_duplicates(subset=[group_key, year_key])
            .rename(columns={variable: "_candidate"})
        )
        if candidate.empty:
            continue
        result = result.merge(candidate, on=[group_key, year_key], how="left")

        # Fill only where nothing higher-priority has already supplied a value.
        unfilled = result[variable].isna() & result["_candidate"].notna()
        result.loc[unfilled, variable] = result.loc[unfilled, "_candidate"]
        result.loc[unfilled, f"{variable}_source"] = name
        result = result.drop(columns=["_candidate"])

    top_source = ordered[0]
    result[f"{variable}_quality_flag"] = np.where(
        result[variable].isna(),
        QUALITY_MISSING,
        np.where(result[f"{variable}_source"] == top_source, QUALITY_OBSERVED, QUALITY_FILLED),
    )
    return result


def merge_variables(
    sources: Mapping[str, pd.DataFrame],
    variables: Sequence[str],
    priority: Sequence[str] = SOURCE_PRIORITY,
    reference: tuple[int, int] = CLIMATOLOGY_REFERENCE,
    group_key: str = "id_glims",
    year_key: str = "year",
) -> pd.DataFrame:
    """Merge several variables and attach climatology, anomaly and traceability.

    The climatology is recomputed on the **merged** series, as
    DESIGN_FEATURES.md section 1.3 requires ("la climatologie finale est
    recalculee sur la serie fusionnee").

    Output columns per variable:
    ``clim_{var}``, ``anom_{var}``, ``{var}_source``, ``{var}_quality_flag``,
    ``{var}_coverage_flag``.
    """
    merged: Optional[pd.DataFrame] = None

    for variable in variables:
        part = merge_by_priority(
            sources, variable, priority=priority, group_key=group_key, year_key=year_key
        )
        if part.empty:
            continue
        part = compute_anomalies(
            part, [variable], reference=reference, group_key=group_key, year_key=year_key
        )
        # The absolute value is not delivered (section 1.2): keep clim/anom only.
        part = part.drop(columns=[variable])
        merged = part if merged is None else merged.merge(
            part, on=[group_key, year_key], how="outer"
        )

    if merged is None:
        return pd.DataFrame(columns=[group_key, year_key])
    return merged.sort_values([group_key, year_key], ignore_index=True)


def traceability_columns(variable: str) -> tuple[str, str, str]:
    """The three mandatory traceability column names for a variable."""
    return (
        f"{variable}_source",
        f"{variable}_quality_flag",
        f"{variable}_coverage_flag",
    )


def validate_traceability(frame: pd.DataFrame, variables: Sequence[str]) -> list[str]:
    """Return the traceability columns missing from ``frame``.

    DESIGN_FEATURES.md section 1.4 makes these mandatory on every delivered
    temporal parquet, so this is used as a release gate.
    """
    missing: list[str] = []
    for variable in variables:
        for column in traceability_columns(variable):
            if column not in frame.columns:
                missing.append(column)
    return missing
