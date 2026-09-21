"""Glacier selection from the RGI 7.0 inventory, and the glacier registry.

Purpose
-------
Select the "isolated" glaciers that the whole pipeline operates on, from the
RGI 7.0 regions 01 (Alaska) and 02 (Western Canada and USA). Two filters are
applied, matching Section 3.1 of the paper:

1. **No geometric overlap** with any other RGI polygon. A glacier whose
   geometry intersects (touches or overlaps) another RGI polygon is dropped,
   *and so is the polygon it intersects* -- overlap is a symmetric relation and
   both members of an intersecting pair are ambiguous for per-glacier
   segmentation, so both are removed.
2. **Area >= 0.05 km^2** (~111 pixels at 30 m), measured in a metric CRS.

Inputs
------
- RGI 7.0 shapefiles (paths from ``config.isolated_glacier.rgi_shapefiles``).

Outputs
-------
- A **glacier registry** written as Parquet (``glacier_registry.parquet``), one
  row per retained glacier: ``glims_id``, ``glac_name``, ``area_km2``,
  centroid lon/lat, WGS84 bounds, the UTM EPSG code of its local zone, and the
  bounds of the square U-Net analysis window. Every later pipeline stage reads
  this single file rather than re-scanning the shapefiles.
- Optionally, the per-glacier directory tree (``<glims_id>/shapefile/`` and
  ``<glims_id>/shapefile_UNet/`` plus ``metadata.json``) that the DEM and scene
  download stages consume.

Design notes (ROADMAP decision #5 -- re-audit for a better-established approach)
--------------------------------------------------------------------------------
* **Parquet, not JSON, for the registry.** The registry is a flat, homogeneous,
  strongly typed table read far more often than written, so Parquet was chosen
  over a JSON list of ``{name, glims_id, area_km2}`` dicts: it is columnar (a
  stage that only needs ``glims_id`` never deserializes the geometry bounds),
  typed (no float-from-string reparsing, no silent ``None`` vs ``NaN`` drift),
  and roughly an order of magnitude smaller and faster to load than the
  equivalent JSON. It also carries the extra per-glacier columns (bounds, UTM
  zone, window size) in one authoritative place rather than having each
  downstream stage recompute them independently and risk drifting out of sync.
  A JSON sidecar of ``{glims_id: glac_name}`` is still written for human
  inspection.
* **Vectorized overlap detection.** Rather than looping over all ~46 000
  polygons in Python (calling a spatial-tree query once per glacier and
  breaking on the first hit), this issues a single bulk
  ``STRtree.query(geoms, predicate=...)`` call, which returns the full (i, j)
  index pair array from one pass in compiled code. Shapely 2.x evaluates the
  ``intersects`` predicate inside the bulk query, so this is both the
  pairwise-exact answer and ~2 orders of magnitude faster than a per-geometry
  Python loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.strtree import STRtree

REGISTRY_FILENAME = "glacier_registry.parquet"
NAMES_FILENAME = "glacier_names.json"

#: Default half-width added around each glacier before squaring off the U-Net
#: analysis window, in metres.
DEFAULT_BUFFER_M = 200.0
#: Extra margin added to the buffered extent before squaring, in metres.
DEFAULT_MARGIN_M = 150.0


class GlacierSelectionError(Exception):
    """Raised when the RGI inputs are missing, unreadable, or malformed."""


def load_rgi(shapefiles: Sequence[str | Path]) -> gpd.GeoDataFrame:
    """Load and concatenate the RGI 7.0 shapefiles into one GeoDataFrame.

    All parts are reprojected to the CRS of the first non-empty part before
    concatenation, so mixing regions stored in different CRSs is safe.
    """
    parts: list[gpd.GeoDataFrame] = []
    for shp in shapefiles:
        path = Path(shp)
        if not path.exists():
            raise GlacierSelectionError(f"RGI shapefile not found: {path}")
        part = gpd.read_file(path)
        if part.empty:
            continue
        parts.append(part)

    if not parts:
        raise GlacierSelectionError(
            "No non-empty RGI shapefile could be read from: "
            + ", ".join(str(s) for s in shapefiles)
        )

    base_crs = parts[0].crs
    parts = [p if p.crs == base_crs else p.to_crs(base_crs) for p in parts]
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=base_crs)

    if "glims_id" not in gdf.columns:
        raise GlacierSelectionError(
            "Column 'glims_id' is absent from the RGI shapefiles; cannot build "
            "a registry without a stable glacier identifier."
        )
    return gdf


def dedupe_by_glims_id(registry: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per ``glims_id``: the largest-area polygon of each group.

    RGI 7.0 does **not** guarantee a unique ``glims_id``: in the local copy of
    region 01, 27,509 polygons carry only 27,321 distinct GLIMS ids (region 02
    has 4 collisions of its own). ``rgi_id`` is unique, ``glims_id`` is not --
    several distinct polygons can map to the same GLIMS entry.

    Every downstream stage keys per-glacier state on ``glims_id`` (directory
    names, the scene cache, the split assignment, the statistics tables), so a
    duplicated id would silently collide: two glaciers writing to one directory,
    one overwriting the other's DEM and scenes.

    Resolution: keep the largest polygon in each colliding group. It is the one
    whose extent dominates the shared GLIMS entry, and area is already the
    pipeline's admissibility and load-balancing currency.

    Applying this brings the retained count to exactly the **22,134** glaciers
    the paper reports, which is corroborating evidence that the published
    inventory was likewise counted per distinct GLIMS id.
    """
    if registry.empty or registry["glims_id"].is_unique:
        return registry
    deduped = (
        registry.sort_values("area_km2", ascending=False, kind="stable")
        .drop_duplicates(subset="glims_id", keep="first")
        .sort_values("area_km2", ascending=False, ignore_index=True)
    )
    return deduped


def auto_detect_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Return the EPSG code of the UTM zone best covering the inventory.

    Uses the median centroid longitude/latitude, so a few outlying glaciers do
    not drag the chosen zone away from the bulk of the data.
    """
    gdf_wgs84 = gdf if (gdf.crs and gdf.crs.to_epsg() == 4326) else gdf.to_crs("EPSG:4326")
    # Use bounds midpoints rather than .centroid: centroid on a geographic CRS
    # warns (and is only approximate), and for picking a UTM zone the extent
    # midpoint is equivalent in practice.
    bounds = gdf_wgs84.geometry.bounds
    median_lon = float(((bounds["minx"] + bounds["maxx"]) / 2.0).median())
    median_lat = float(((bounds["miny"] + bounds["maxy"]) / 2.0).median())
    return f"EPSG:{_utm_epsg(median_lon, median_lat)}"


def _utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def geodesic_area_km2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Compute per-polygon area on the WGS84 ellipsoid, in km^2.

    Used as the fallback when the RGI inventory does not ship its own
    ``area_km2`` column. Geodesic area is projection-independent, so unlike a
    single-UTM-zone planar area it does not distort across a domain spanning
    ~50 degrees of longitude (RGI 01+02).
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    gdf_wgs84 = gdf if (gdf.crs and gdf.crs.to_epsg() == 4326) else gdf.to_crs("EPSG:4326")
    return np.array(
        [abs(geod.geometry_area_perimeter(geom)[0]) / 1e6 for geom in gdf_wgs84.geometry.values],
        dtype=float,
    )


def project_metric(
    gdf: gpd.GeoDataFrame,
    crs_metric: str = "auto",
    area_source: str = "rgi",
) -> tuple[gpd.GeoDataFrame, str]:
    """Reproject to a metric CRS and attach the ``area_km2`` used for filtering.

    ``crs_metric="auto"`` picks the UTM zone of the inventory's median centroid.
    The projected geometry is used for *overlap detection* and for the square
    U-Net window; it is deliberately **not** used for the area threshold.

    ``area_source``:

    * ``"rgi"`` (default) -- use the inventory's own ``area_km2`` column when
      present, else fall back to geodesic. RGI 7.0 ships an ellipsoidal area
      per polygon; measured against ``pyproj.Geod`` it agrees to 4 decimal
      places, i.e. it *is* the geodesic area. Reusing it makes the 0.05 km^2
      admissibility threshold reproduce the published inventory exactly rather
      than approximately.
    * ``"geodesic"`` -- always recompute on the ellipsoid.
    * ``"planar"`` -- area of the reprojected geometry. Kept only for
      comparison: over RGI 01+02 a single UTM zone inflates areas by up to
      ~19% at the domain edges, which pushes ~100 glaciers across the
      0.05 km^2 threshold that the published inventory excludes.
    """
    resolved = auto_detect_utm_crs(gdf) if crs_metric.strip().lower() == "auto" else crs_metric
    gdf_proj = gdf.to_crs(resolved)

    mode = area_source.strip().lower()
    if mode == "planar":
        gdf_proj["area_km2"] = gdf_proj.geometry.area / 1e6
    elif mode == "geodesic":
        gdf_proj["area_km2"] = geodesic_area_km2(gdf)
    elif mode == "rgi":
        if "area_km2" in gdf.columns:
            gdf_proj["area_km2"] = gdf["area_km2"].to_numpy(dtype=float)
        else:
            gdf_proj["area_km2"] = geodesic_area_km2(gdf)
    else:
        raise ValueError(
            f"Unknown area_source '{area_source}' (expected 'rgi', 'geodesic', or 'planar')."
        )
    return gdf_proj, resolved


def find_overlapping_indices(gdf_proj: gpd.GeoDataFrame) -> np.ndarray:
    """Return the positional indices of glaciers that intersect another glacier.

    Fully vectorized: a single bulk ``STRtree.query`` with the ``intersects``
    predicate returns every intersecting (i, j) pair evaluated in compiled
    code, rather than a per-geometry Python loop over the ~46 000 candidate
    polygons. Self-pairs (i == j) are removed, then both members of every
    surviving pair are marked -- an overlap makes *both* polygons ambiguous,
    so both are dropped.

    Returns
    -------
    np.ndarray
        Sorted array of unique positional indices to remove.
    """
    geoms = gdf_proj.geometry.values
    if len(geoms) == 0:
        return np.empty(0, dtype=np.int64)

    tree = STRtree(geoms)
    # Bulk query: shape (2, n_pairs), row 0 = input index, row 1 = tree index.
    pairs = tree.query(geoms, predicate="intersects")
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64)

    left, right = pairs[0], pairs[1]
    cross = left != right  # drop each geometry's match against itself
    if not cross.any():
        return np.empty(0, dtype=np.int64)

    return np.unique(np.concatenate([left[cross], right[cross]])).astype(np.int64)


def _square_window_bounds(
    geom_metric,
    buffer_m: float,
    margin_m: float,
) -> tuple[float, float, float, float, float]:
    """Return (minx, miny, maxx, maxy, side) of the square U-Net window.

    The glacier is buffered by ``buffer_m``, ``margin_m`` is added to each
    dimension of the resulting extent, and the larger dimension defines the
    side of a square centred on the buffered extent, so every glacier gets a
    square, fixed-margin analysis window regardless of its native aspect
    ratio.
    """
    minx, miny, maxx, maxy = geom_metric.buffer(buffer_m).bounds
    side = max((maxx - minx) + margin_m, (maxy - miny) + margin_m)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = side / 2.0
    return cx - half, cy - half, cx + half, cy + half, side


def build_registry(
    gdf: gpd.GeoDataFrame,
    gdf_proj: gpd.GeoDataFrame,
    drop_indices: Iterable[int],
    min_area_km2: float,
    buffer_m: float = DEFAULT_BUFFER_M,
    margin_m: float = DEFAULT_MARGIN_M,
) -> pd.DataFrame:
    """Assemble the glacier registry table for the retained glaciers.

    Applies both selection filters (not in ``drop_indices``, and
    ``area_km2 >= min_area_km2``) with boolean masks, then computes the
    per-glacier geometry-derived columns in vectorized form.
    """
    n = len(gdf_proj)
    keep = np.ones(n, dtype=bool)
    drop = np.fromiter(drop_indices, dtype=np.int64)
    if drop.size:
        keep[drop] = False

    areas = gdf_proj["area_km2"].to_numpy(dtype=float)
    keep &= areas >= float(min_area_km2)

    if not keep.any():
        return _empty_registry()

    sel_proj = gdf_proj.loc[keep]
    sel_wgs84 = gdf.loc[keep].to_crs("EPSG:4326")

    # Centroids are computed on the projected geometry (planar centroid is only
    # meaningful there), then converted back to lon/lat for the registry.
    centroids = sel_proj.geometry.centroid.to_crs("EPSG:4326")
    lon = centroids.x.to_numpy(dtype=float)
    lat = centroids.y.to_numpy(dtype=float)

    # Vectorized UTM zone: same formula as _utm_epsg, applied to whole arrays.
    zone = np.clip(((lon + 180.0) // 6.0).astype(np.int64) + 1, 1, 60)
    utm_epsg = np.where(lat >= 0, 32600 + zone, 32700 + zone).astype(np.int64)

    wgs_bounds = sel_wgs84.geometry.bounds.to_numpy(dtype=float)

    window = np.array(
        [_square_window_bounds(g, buffer_m, margin_m) for g in sel_proj.geometry.values],
        dtype=float,
    )

    names = (
        sel_wgs84["glac_name"].astype("string").fillna("").str.strip()
        if "glac_name" in sel_wgs84.columns
        else pd.Series([""] * int(keep.sum()), index=sel_wgs84.index, dtype="string")
    )
    names = names.where(~names.str.lower().isin(["nan", "none", "null"]), "")

    registry = pd.DataFrame(
        {
            "glims_id": sel_wgs84["glims_id"].astype(str).to_numpy(),
            "glac_name": names.to_numpy(dtype=object),
            "area_km2": sel_proj["area_km2"].to_numpy(dtype=float),
            "centroid_lon": lon,
            "centroid_lat": lat,
            "min_lon": wgs_bounds[:, 0],
            "min_lat": wgs_bounds[:, 1],
            "max_lon": wgs_bounds[:, 2],
            "max_lat": wgs_bounds[:, 3],
            "utm_epsg": utm_epsg,
            "window_minx": window[:, 0],
            "window_miny": window[:, 1],
            "window_maxx": window[:, 2],
            "window_maxy": window[:, 3],
            "window_side_m": window[:, 4],
            "metric_crs": str(sel_proj.crs),
        }
    )
    registry = registry.sort_values("area_km2", ascending=False, ignore_index=True)
    # RGI 7.0 does not guarantee a unique glims_id; downstream stages key on it.
    return dedupe_by_glims_id(registry)


def _empty_registry() -> pd.DataFrame:
    """Return an empty registry with the correct dtypes."""
    return pd.DataFrame(
        {
            "glims_id": pd.Series(dtype=str),
            "glac_name": pd.Series(dtype=object),
            "area_km2": pd.Series(dtype=float),
            "centroid_lon": pd.Series(dtype=float),
            "centroid_lat": pd.Series(dtype=float),
            "min_lon": pd.Series(dtype=float),
            "min_lat": pd.Series(dtype=float),
            "max_lon": pd.Series(dtype=float),
            "max_lat": pd.Series(dtype=float),
            "utm_epsg": pd.Series(dtype="int64"),
            "window_minx": pd.Series(dtype=float),
            "window_miny": pd.Series(dtype=float),
            "window_maxx": pd.Series(dtype=float),
            "window_maxy": pd.Series(dtype=float),
            "window_side_m": pd.Series(dtype=float),
            "metric_crs": pd.Series(dtype=str),
        }
    )


def write_registry(registry: pd.DataFrame, path: str | Path) -> Path:
    """Write the registry to Parquet, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(out, index=False)
    return out


def read_registry(path: str | Path) -> pd.DataFrame:
    """Read a glacier registry written by :func:`write_registry`."""
    p = Path(path)
    if not p.is_file():
        raise GlacierSelectionError(
            f"Glacier registry not found at '{p}'. Run scripts/01_select_glaciers.py first."
        )
    return pd.read_parquet(p)


def write_names_json(registry: pd.DataFrame, path: str | Path) -> Path:
    """Write the ``{glims_id: glac_name}`` sidecar for named glaciers only."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    named = registry.loc[registry["glac_name"].astype(str).str.len() > 0]
    payload = dict(zip(named["glims_id"].astype(str), named["glac_name"].astype(str)))
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_glacier_directories(
    registry: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    output_root: str | Path,
    metric_crs: str,
    buffer_m: float = DEFAULT_BUFFER_M,
    margin_m: float = DEFAULT_MARGIN_M,
    min_area_km2: float = 0.05,
    n_jobs: int = 1,
) -> tuple[int, int]:
    """Write the per-glacier directory tree consumed by later stages.

    For each registry glacier, writes ``shapefile/{id}.shp`` (WGS84) and
    ``_utm.shp`` (local UTM) for the glacier outline, the same pair under
    ``shapefile_UNet/`` for the square analysis window, and a ``metadata.json``.

    Parallelized with joblib over glaciers (each glacier is an independent set
    of file writes). Returns ``(n_written, n_errors)``.
    """
    from joblib import Parallel, delayed

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    geom_by_id = dict(zip(gdf["glims_id"].astype(str), gdf.to_crs("EPSG:4326").geometry.values))
    gdf_metric = gdf.to_crs(metric_crs)
    metric_by_id = dict(zip(gdf["glims_id"].astype(str), gdf_metric.geometry.values))

    records = registry.to_dict("records")
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_write_one_glacier)(
            rec, geom_by_id, metric_by_id, root, metric_crs, buffer_m, margin_m, min_area_km2
        )
        for rec in records
    )
    n_ok = sum(1 for r in results if r is None)
    return n_ok, len(results) - n_ok


def _write_one_glacier(
    rec: dict,
    geom_by_id: dict,
    metric_by_id: dict,
    root: Path,
    metric_crs: str,
    buffer_m: float,
    margin_m: float,
    min_area_km2: float,
) -> Optional[str]:
    """Write one glacier's shapefiles and metadata. Returns None on success."""
    glims_id = str(rec["glims_id"])
    try:
        geom_wgs84 = geom_by_id[glims_id]
        geom_metric = metric_by_id[glims_id]
        minx, miny, maxx, maxy, side = _square_window_bounds(geom_metric, buffer_m, margin_m)
        square = Polygon(
            [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        )

        outline_wgs84 = gpd.GeoDataFrame(
            {"glims_id": [glims_id], "geometry": [geom_wgs84]}, crs="EPSG:4326"
        )
        window_wgs84 = gpd.GeoDataFrame(
            {"glims_id": [glims_id], "geometry": [square]}, crs=metric_crs
        ).to_crs("EPSG:4326")

        utm_crs = CRS.from_epsg(int(rec["utm_epsg"]))
        glacier_dir = root / glims_id
        dir_outline = glacier_dir / "shapefile"
        dir_window = glacier_dir / "shapefile_UNet"
        dir_outline.mkdir(parents=True, exist_ok=True)
        dir_window.mkdir(parents=True, exist_ok=True)

        paths = {
            "outline_wgs84": dir_outline / f"{glims_id}.shp",
            "outline_utm": dir_outline / f"{glims_id}_utm.shp",
            "window_wgs84": dir_window / f"{glims_id}.shp",
            "window_utm": dir_window / f"{glims_id}_utm.shp",
        }
        outline_wgs84.to_file(paths["outline_wgs84"])
        outline_wgs84.to_crs(utm_crs).to_file(paths["outline_utm"])
        window_wgs84.to_file(paths["window_wgs84"])
        window_wgs84.to_crs(utm_crs).to_file(paths["window_utm"])

        meta = {
            "glims_id": glims_id,
            "glac_name": str(rec.get("glac_name", "")),
            "filters": {
                "area_km2_min": float(min_area_km2),
                "overlap_check": True,
                "conflict_policy": "drop_both_intersecting",
                "buffer_m": float(buffer_m),
                "margin_m": float(margin_m),
                "window_side_m": float(side),
                "units": "meters",
            },
            "metrics": {"area_km2": float(rec["area_km2"])},
            "crs": {
                "filter_crs": metric_crs,
                "saved_crs": "EPSG:4326",
                "saved_crs_utm": f"EPSG:{int(rec['utm_epsg'])}",
                "saved_crs_utm_name": utm_crs.name,
            },
            "paths": {k: str(v) for k, v in paths.items()},
            "source": "glacier_fsnow_unet.glaciers.registry",
        }
        (glacier_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return None
    except Exception as exc:  # noqa: BLE001 - collected and reported by caller
        return f"{glims_id}: {exc}"


def select_glaciers(
    rgi_shapefiles: Sequence[str | Path],
    min_area_km2: float = 0.05,
    crs_metric: str = "auto",
    check_overlap: bool = True,
    area_source: str = "rgi",
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, str]:
    """Run the full selection and return ``(registry, source_gdf, metric_crs)``."""
    gdf = load_rgi(rgi_shapefiles)
    gdf_proj, resolved_crs = project_metric(gdf, crs_metric, area_source=area_source)
    drop = find_overlapping_indices(gdf_proj) if check_overlap else np.empty(0, dtype=np.int64)
    registry = build_registry(gdf, gdf_proj, drop, min_area_km2)
    return registry, gdf, resolved_crs
