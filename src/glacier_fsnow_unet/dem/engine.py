"""Per-glacier DEM extraction orchestration.

Purpose
-------
For each glacier in the registry: derive the buffered, squared analysis window,
fetch the best available source DEM (ArcticDEM v4.1, else Copernicus GLO-30),
mask it to the window, and write every output resolution plus provenance and
coverage statistics into the glacier's zarr store.

Inputs
------
- The glacier registry (stage 1) or an existing per-glacier directory tree.

Outputs
-------
- ``<glacier_dir>/<glims_id>.zarr`` -- ``dem`` group holding ``elev_10m``,
  ``elev_15m``, ``elev_30m``, ``elev_30m_interp`` (float32, EPSG:3413, NaN
  nodata) plus provenance/coverage attrs. This is the same store
  :mod:`glacier_fsnow_unet.scenes.zarr_store` writes scene TOA/RGB into (see
  ``docs/decisions/dem_backend.md``).
- ``<glacier_dir>/extract_dem.done`` -- completion flag (enables resume)
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  -- registers the .rio accessor
from shapely.geometry import box
from shapely.ops import unary_union

from .processing import (
    MARGIN_METERS,
    MAX_ZERO_PCT,
    OUTPUT_RESOLUTIONS_M,
    RESAMPLE_METHOD,
    apply_mask_rules,
    build_output_resolutions,
    interpolate_interior_holes,
    polygon_stats,
    rasterize_polygon_mask,
    square_bounds,
)
from .sources import (
    ARCTICDEM_BASE_RES_M,
    ARCTICDEM_COLLECTION,
    ARCTICDEM_VERSION,
    PGC_STAC_URL,
    TARGET_EPSG,
    load_best_dem,
)
from .zarr_store import write_dem, write_interpolated_30m

DONE_FLAG = "extract_dem.done"


def find_window_shapefile(glacier_dir: Path) -> Optional[Path]:
    """Return the analysis-window shapefile inside a glacier directory."""
    unet_dir = glacier_dir / "shapefile_UNet"
    if not unet_dir.is_dir():
        return None
    candidates = sorted(unet_dir.glob("*.shp"))
    # Prefer the WGS84 variant (no _utm suffix) for a stable reprojection path.
    for candidate in candidates:
        if not candidate.stem.endswith("_utm"):
            return candidate
    return candidates[0] if candidates else None


def window_polygon_from_registry(row) -> "gpd.GeoSeries":
    """Build the EPSG:3413 analysis-window polygon from a registry row."""
    square = box(
        float(row["window_minx"]),
        float(row["window_miny"]),
        float(row["window_maxx"]),
        float(row["window_maxy"]),
    )
    series = gpd.GeoSeries([square], crs=str(row["metric_crs"]))
    return series.to_crs(TARGET_EPSG)


def window_polygon_from_shapefile(path: Path):
    """Load and dissolve an analysis-window shapefile into one EPSG:3413 polygon."""
    gdf = gpd.read_file(path)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    if gdf.empty:
        return None
    if gdf.crs is None:
        raise ValueError(f"Analysis-window shapefile has no CRS: {path}")
    merged = unary_union(gdf.geometry.tolist())
    return gpd.GeoSeries([merged], crs=gdf.crs).to_crs(TARGET_EPSG).iloc[0]


def process_one_glacier(
    glacier_dir: Path,
    polygon_3413=None,
    overwrite: bool = False,
    copernicus_fallback: bool = True,
    stac_url: str = PGC_STAC_URL,
    collection: str = ARCTICDEM_COLLECTION,
    resolutions: tuple[int, ...] = OUTPUT_RESOLUTIONS_M,
    force_copernicus: bool = False,
) -> tuple[str, str]:
    """Extract and write the DEM products for one glacier.

    Returns ``(glacier_id, status)`` where status is ``"ok"``, ``"skip (...)"``
    or ``"error (...)"``. Never raises: failures are reported per glacier so a
    parallel sweep over 22k glaciers is not aborted by one bad tile.
    """
    gid = glacier_dir.name
    try:
        if not overwrite and (glacier_dir / DONE_FLAG).exists():
            return gid, "skip (already done)"

        if polygon_3413 is None:
            shp = find_window_shapefile(glacier_dir)
            if shp is None:
                return gid, "skip (no shapefile_UNet)"
            polygon_3413 = window_polygon_from_shapefile(shp)
            if polygon_3413 is None:
                return gid, "skip (empty analysis window)"

        buffered = (
            polygon_3413.buffer(MARGIN_METERS) if MARGIN_METERS > 0 else polygon_3413
        )
        bounds = square_bounds(buffered.bounds, ARCTICDEM_BASE_RES_M)

        dem, source_label, base_res, valid_pct = load_best_dem(
            bounds,
            use_copernicus_fallback=copernicus_fallback,
            stac_url=stac_url,
            collection=collection,
            force_copernicus=force_copernicus,
        )
        if dem is None:
            return gid, "skip (no DEM coverage from ArcticDEM or Copernicus GLO-30)"

        mask = rasterize_polygon_mask(dem, polygon_3413)
        if int(np.asarray(mask.values).sum()) == 0:
            return gid, "skip (empty mask: window outside the source DEM grid)"

        dem_masked = apply_mask_rules(dem, mask)
        stats = polygon_stats(dem_masked, mask)
        if stats["total_px"] == 0:
            return gid, "skip (no pixels inside the polygon after rasterization)"
        if stats["zero_pct"] > MAX_ZERO_PCT:
            return gid, f"skip (>{MAX_ZERO_PCT:.0f}% zero-filled pixels: {stats['zero_pct']}%)"

        outputs = build_output_resolutions(dem_masked, base_res, resolutions)

        glacier_dir.mkdir(parents=True, exist_ok=True)

        crs_wkt = dem_masked.rio.crs.to_wkt()
        resolution_arrays: dict[int, np.ndarray] = {}
        resolution_transforms: dict[int, object] = {}
        for res, arr in sorted(outputs.items()):
            arr.rio.write_nodata(np.nan, inplace=True)
            resolution_arrays[int(res)] = np.asarray(arr.values, dtype=np.float32)
            resolution_transforms[int(res)] = arr.rio.transform(recalc=False)

        interp_status = "skip (30 m product absent)"
        if 30 in resolution_arrays:
            filled = interpolate_interior_holes(resolution_arrays[30])
            interp_status = "ok"

        sidecar = {
            "glims_id": gid,
            "source_dem": source_label,
            "source_base_resolution_m": base_res,
            "arcticdem_version": ARCTICDEM_VERSION,
            "crs": f"EPSG:{TARGET_EPSG}",
            "outputs_m": list(resolutions),
            "resample_method": RESAMPLE_METHOD,
            "margin_m": MARGIN_METERS,
            "window_bounds_3413": list(bounds),
            "dem_30_interp_status": interp_status,
            "source_window_valid_pct": round(valid_pct, 2),
            "polygon_stats": stats,
            "mask_px": int(np.asarray(mask.values).sum()),
            "rules": {
                "outside_polygon": "NaN",
                "inside_polygon_nonfinite_to": 0.0,
                "keep_inside_values": True,
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_dem(glacier_dir, resolution_arrays, resolution_transforms, crs_wkt, attrs=sidecar)
        if 30 in resolution_arrays:
            write_interpolated_30m(glacier_dir, filled, status=interp_status)

        (glacier_dir / DONE_FLAG).write_text(
            json.dumps({"resolutions": list(resolutions)}), encoding="utf-8"
        )
        return gid, "ok"

    except Exception as exc:  # noqa: BLE001 - reported per glacier, never fatal
        tb = traceback.format_exc().splitlines()
        where = tb[-2].strip() if len(tb) >= 2 else "?"
        return gid, f"error ({type(exc).__name__}: {exc} @ {where})"


def run_dem_extraction(
    glacier_dirs: list[Path],
    polygons: Optional[dict] = None,
    n_jobs: int = 8,
    overwrite: bool = False,
    copernicus_fallback: bool = True,
    stac_url: str = PGC_STAC_URL,
    collection: str = ARCTICDEM_COLLECTION,
    force_copernicus: bool = False,
) -> list[tuple[str, str]]:
    """Run DEM extraction over many glaciers in parallel.

    Threads (not processes) are used because the work is dominated by network
    range-requests and GDAL/rasterio I/O, both of which release the GIL.

    ``force_copernicus`` skips ArcticDEM for every glacier in this call and
    goes straight to Copernicus GLO-30, so the fallback path can be exercised
    deliberately rather than only where ArcticDEM happens to have no coverage.
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    polygons = polygons or {}
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_one_glacier)(
            gdir,
            polygon_3413=polygons.get(gdir.name),
            overwrite=overwrite,
            copernicus_fallback=copernicus_fallback,
            stac_url=stac_url,
            collection=collection,
            force_copernicus=force_copernicus,
        )
        for gdir in tqdm(glacier_dirs, desc="DEM extraction")
    )
