"""The per-glacier statistics tables.

Purpose
-------
The tail of the inference chain (stage 10): produce the three delivered
parquets:

* ``glacier_ref.parquet``      -- one row per glacier: the VGS reference
  (reference year, area, RGI ratio).
* ``stats_etat.parquet``       -- one row per glacier-year: class counts
  inside the VGS and F_snow.
* ``stats_normalized.parquet`` -- the same, normalized by VGS area.

Inputs
------
- Annual worst-state composites and the VGS mask, per glacier.

Outputs
-------
- ``pandas`` frames written as Parquet.

Vectorization
-------------
Class counting over a whole time series is a single reduction over the year
axis (``states[:, vgs]`` then per-class sums), never a loop over years or
pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .classes import CLOUD, ICE, NODATA, OTHER, SNOW
from .fsnow import DEFAULT_MAX_UNRESOLVED_FRACTION

GLACIER_REF_PARQUET = "glacier_ref.parquet"
STATS_ETAT_PARQUET = "stats_etat.parquet"
STATS_NORMALIZED_PARQUET = "stats_normalized.parquet"


def class_counts_in_mask(state_map: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    """Per-class pixel counts inside a mask."""
    inside = np.asarray(state_map, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    return {
        "cloud_px": int((inside == CLOUD).sum()),
        "snow_px": int((inside == SNOW).sum()),
        "ice_px": int((inside == ICE).sum()),
        "other_px": int((inside == OTHER).sum()),
        "nodata_px": int((inside == NODATA).sum()),
    }


def build_glacier_ref_row(
    glims_id: str,
    vgs_mask: np.ndarray,
    reference_year: Optional[int],
    rgi_area_km2: float,
    pixel_area_m2: float = 900.0,
    extra: Optional[Mapping] = None,
) -> dict:
    """One ``glacier_ref.parquet`` row: this glacier's VGS reference.

    Distance to the nearest coastline is a retrieved external value (from a
    user-supplied coastline shapefile), not something this pipeline
    produces, so it is not a column here -- it lives in
    ``static_features.parquet`` alongside the other retrieved static sources
    (see ``inference/coastline.py`` and ``docs/decisions/static_feature_sources.md``).
    """
    vgs_px = int(np.asarray(vgs_mask, dtype=bool).sum()) if vgs_mask is not None else 0
    row = {
        "id_glims": str(glims_id),
        "sgv_ref_year": reference_year,
        "sgv_ref_px": vgs_px,
        "sgv_ref_area_km2": vgs_px * pixel_area_m2 / 1e6,
        "rgi_area_km2": float(rgi_area_km2),
        "sgv_ref_to_rgi_ratio": (
            (vgs_px * pixel_area_m2 / 1e6) / rgi_area_km2 if rgi_area_km2 > 0 else np.nan
        ),
    }
    if extra:
        row.update(dict(extra))
    return row


def build_stats_etat(
    glims_id: str,
    annual_states: Mapping[int, np.ndarray] | Iterable[tuple[int, np.ndarray]],
    vgs_mask: np.ndarray,
    pixel_area_m2: float = 900.0,
    max_unresolved_fraction: float = DEFAULT_MAX_UNRESOLVED_FRACTION,
) -> pd.DataFrame:
    """One ``stats_etat.parquet`` row per glacier-year.

    Carries the class counts inside the VGS, F_snow, and the validity flag.
    """
    from .fsnow import compute_fsnow

    items = (
        sorted(annual_states.items())
        if isinstance(annual_states, Mapping)
        else sorted(annual_states, key=lambda kv: kv[0])
    )
    vgs = np.asarray(vgs_mask, dtype=bool)
    vgs_px = int(vgs.sum())

    rows = []
    for year, state_map in items:
        counts = class_counts_in_mask(state_map, vgs)
        result = compute_fsnow(
            state_map, vgs, year=year, max_unresolved_fraction=max_unresolved_fraction
        )

        rows.append(
            {
                "id_glims": str(glims_id),
                "year": int(year),
                "vgs_px": vgs_px,
                "vgs_area_km2": vgs_px * pixel_area_m2 / 1e6,
                **counts,
                "fsnow": result.fsnow,
                "unresolved_fraction": result.unresolved_fraction,
                "valid": result.valid,
                "reason": result.reason,
            }
        )

    return pd.DataFrame(rows)


def normalize_stats(stats_etat: pd.DataFrame) -> pd.DataFrame:
    """Build ``stats_normalized.parquet`` from ``stats_etat``.

    Every pixel count becomes a fraction of the VGS area, so glaciers of very
    different sizes are directly comparable.
    """
    if stats_etat.empty:
        return stats_etat.copy()

    out = stats_etat[["id_glims", "year", "vgs_px", "vgs_area_km2"]].copy()
    denominator = stats_etat["vgs_px"].replace(0, np.nan)

    for column in ("cloud_px", "snow_px", "ice_px", "other_px", "nodata_px"):
        if column in stats_etat.columns:
            out[column.replace("_px", "_frac")] = stats_etat[column] / denominator

    for column in ("fsnow", "unresolved_fraction", "valid"):
        if column in stats_etat.columns:
            out[column] = stats_etat[column]

    return out


def write_stats(
    output_root: str | Path,
    glacier_ref: pd.DataFrame,
    stats_etat: pd.DataFrame,
    stats_normalized: Optional[pd.DataFrame] = None,
) -> dict[str, Path]:
    """Write the three delivered statistics parquets."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    if stats_normalized is None:
        stats_normalized = normalize_stats(stats_etat)

    paths = {
        "glacier_ref": root / GLACIER_REF_PARQUET,
        "stats_etat": root / STATS_ETAT_PARQUET,
        "stats_normalized": root / STATS_NORMALIZED_PARQUET,
    }
    glacier_ref.to_parquet(paths["glacier_ref"], index=False)
    stats_etat.to_parquet(paths["stats_etat"], index=False)
    stats_normalized.to_parquet(paths["stats_normalized"], index=False)
    return paths
