"""Point-sampling helpers for static (non-temporal) glacier features.

Purpose
-------
Static sources (climatology normals, biome classification, permafrost
extent) describe one fixed value per glacier rather than a per-year series,
so they are sampled once at each glacier's centroid instead of going through
``gee_runner``'s per-year reduction. This module holds the two sampling
patterns every static source in ``static_source_defs.py`` needs: reading a
value out of a raster at a point, and joining a glacier's centroid against a
vector layer's polygons.

Inputs / outputs
-----------------
Pure numpy/pandas/rasterio/geopandas; no network calls.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def sample_rasters_at_centroids(
    glaciers: pd.DataFrame,
    variable_to_raster: dict[str, Path],
) -> pd.DataFrame:
    """Sample one value per raster at each glacier's centroid.

    ``glaciers`` must have ``glims_id``, ``centroid_lon``, ``centroid_lat``
    columns (WGS84). A raster's own nodata value is mapped to NaN.
    """
    existing = {name: path for name, path in variable_to_raster.items() if Path(path).exists()}
    if not existing:
        raise FileNotFoundError("No raster files resolved for this static source.")

    rows: list[dict] = []
    for row in glaciers.itertuples(index=False):
        out = {"glims_id": str(row.glims_id)}
        xy = [(float(row.centroid_lon), float(row.centroid_lat))]
        for name, path in existing.items():
            with rasterio.open(path) as ds:
                value = next(ds.sample(xy))[0]
                nodata = ds.nodata
                if nodata is not None and np.isfinite(nodata) and np.isfinite(value) and float(value) == float(nodata):
                    value = np.nan
                out[name] = float(value) if np.isfinite(value) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def sample_vector_attributes_at_centroids(
    glaciers: pd.DataFrame,
    vector_path: Path,
    column_map: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Spatially join each glacier's centroid against a vector layer.

    ``column_map`` maps an output column name to a tuple of candidate source
    column names to try, in order (vector schemas vary release to release).
    """
    gdf = gpd.read_file(vector_path)
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    points = gpd.GeoDataFrame(
        glaciers[["glims_id"]].copy(),
        geometry=gpd.points_from_xy(glaciers["centroid_lon"], glaciers["centroid_lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, gdf, how="left", predicate="intersects")
    joined = joined.drop_duplicates(subset=["glims_id"], keep="first")

    out = joined[["glims_id"]].reset_index(drop=True)
    for out_name, candidates in column_map.items():
        source_col = next((c for c in candidates if c in joined.columns), None)
        out[out_name] = joined[source_col].to_numpy() if source_col else pd.NA
    return out
