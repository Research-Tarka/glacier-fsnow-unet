"""Distance from each glacier to the nearest coastline.

Purpose
-------
Compute ``distance_to_nearest_coast_km`` per glacier, one of the sources
``scripts/13_download_static_sources.py`` (stage 13) merges into
``static_features.parquet``. This variable separates maritime from
continental glaciers, a distinction the paper leans on repeatedly when
interpreting F_snow (maritime south-east Alaska vs continental
Yukon/Rockies). It is computed from a user-supplied coastline shapefile -- a
retrieved external layer, not something this pipeline produces -- so it is
merged alongside the other retrieved static sources (RGI structural,
Koppen-Geiger, WorldClim, permafrost) rather than living in
``glacier_ref.parquet``; see ``docs/decisions/static_feature_sources.md``.

Inputs
------
- A coastline shapefile (or zipped shapefile), passed via ``--coastline``.
- The glacier table (``glims_id``/``centroid_lon``/``centroid_lat``) stage
  12 already loads from the registry for every static source.

Outputs
-------
- A two-column ``glims_id``/``distance_to_nearest_coast_km`` DataFrame; the
  caller merges it into ``static_features.parquet``, same as every other
  static source's builder output.

Vectorization
-------------
Distances are computed with a single vectorized ``GeoSeries.distance`` against
the unioned coastline, in an equal-distance projection, rather than a Python
loop over glaciers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Equidistant conic projection suitable for distance work over Alaska/NW Canada.
#: Distances are computed in metres in this CRS.
DEFAULT_DISTANCE_CRS = "EPSG:3338"  # NAD83 / Alaska Albers


class CoastlineError(Exception):
    """Raised when the coastline layer cannot be read or is unusable."""


def load_coastline(path: str | Path, target_crs: str = DEFAULT_DISTANCE_CRS):
    """Load a coastline shapefile and return its unioned geometry.

    Accepts a plain ``.shp`` or a ``.zip`` (read through GDAL's ``/vsizip/``).
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    source = Path(path)
    if not source.exists():
        raise CoastlineError(f"Coastline file not found: {source}")

    try:
        if source.suffix.lower() == ".zip":
            gdf = gpd.read_file(f"zip://{source}")
        else:
            gdf = gpd.read_file(source)
    except Exception as exc:
        raise CoastlineError(f"Could not read the coastline from '{source}': {exc}") from exc

    if gdf.empty:
        raise CoastlineError(f"Coastline layer is empty: {source}")
    if gdf.crs is None:
        raise CoastlineError(
            f"Coastline layer has no CRS: {source}. Assign one before using it."
        )

    gdf = gdf.to_crs(target_crs)
    return unary_union(gdf.geometry.values)


def distances_to_coast_km(
    lons: np.ndarray,
    lats: np.ndarray,
    coastline,
    distance_crs: str = DEFAULT_DISTANCE_CRS,
) -> np.ndarray:
    """Great-arc-free planar distance from each point to the coastline, in km.

    Points are projected into ``distance_crs`` (metric) and measured against the
    unioned coastline geometry in one vectorized call.
    """
    import geopandas as gpd

    points = gpd.GeoSeries(
        gpd.points_from_xy(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)),
        crs="EPSG:4326",
    ).to_crs(distance_crs)

    return (points.distance(coastline).to_numpy() / 1000.0).astype(float)


def build_coastline_distance(glaciers, coastline_path: str | Path, distance_crs: str = DEFAULT_DISTANCE_CRS):
    """Compute ``distance_to_nearest_coast_km`` for each glacier.

    ``glaciers`` needs ``glims_id``/``centroid_lon``/``centroid_lat`` columns
    (the same shape ``scripts/13_download_static_sources.py`` passes to every
    static source's builder). Returns a two-column ``pandas.DataFrame`` ready
    to merge into ``static_features.parquet`` alongside the other static
    sources, rather than writing the file itself -- the calling script's
    existing merge-and-write logic handles that uniformly for every source.
    """
    import pandas as pd

    lons = glaciers["centroid_lon"].to_numpy()
    lats = glaciers["centroid_lat"].to_numpy()
    if np.isnan(lons).all():
        raise CoastlineError(
            "No usable glacier coordinates were found; cannot compute distances."
        )

    coastline = load_coastline(coastline_path, target_crs=distance_crs)
    distances = distances_to_coast_km(lons, lats, coastline, distance_crs=distance_crs)

    return pd.DataFrame(
        {
            "glims_id": glaciers["glims_id"].astype(str).to_numpy(),
            "distance_to_nearest_coast_km": distances,
        }
    )
