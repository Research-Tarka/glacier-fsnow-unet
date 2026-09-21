"""DEM post-processing: masking, resampling, hole interpolation.

Purpose
-------
Turn a raw source DEM window (ArcticDEM 10 m or Copernicus GLO-30) into the
per-glacier products the pipeline consumes: masked to the glacier's analysis
window and written at each of the output resolutions [10, 15, 30] m.

Inputs
------
- A source DEM ``xarray.DataArray`` in EPSG:3413.
- The glacier's analysis-window polygon (EPSG:3413).

Outputs
-------
- One masked, resampled ``DataArray`` per output resolution.

Masking rules
-------------
- Outside the polygon: NaN.
- Inside the polygon but non-finite in the input DEM -- 0.0 (flagged in the
  sidecar's ``zero_pct``; a glacier whose window is >50% zeros is rejected).
- Inside and finite: the source value, unchanged.

The 0.0 fill (rather than NaN) for non-finite-but-in-polygon pixels keeps
downstream resampling and coarsening numerically simple (no NaN-propagation
edge cases through bilinear/mean kernels) while the ``zero_pct`` sidecar still
lets callers detect and reject windows where that fill dominates.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.geometry import mapping

from .sources import TARGET_EPSG, _wrap_array

#: Output resolutions written per glacier, in metres: 10 m native ArcticDEM,
#: 30 m to match the coarsest sensor grid, 15 m as the Sentinel-2/L8-9
#: intermediate grid used by several spectral bands.
OUTPUT_RESOLUTIONS_M = (10, 15, 30)
#: Buffer added around the glacier window before fetching the source DEM.
MARGIN_METERS = 150
#: Resampling kernel used for every resolution change.
RESAMPLE_METHOD = "bilinear"
#: Reject a glacier whose masked window is more than this share of filled zeros.
MAX_ZERO_PCT = 50.0


def square_bounds(
    bounds: tuple[float, float, float, float],
    pixel_size_m: Optional[float] = None,
) -> tuple[float, float, float, float]:
    """Expand ``bounds`` to a square, optionally snapped up to a pixel multiple."""
    xmin, ymin, xmax, ymax = bounds
    side = max(xmax - xmin, ymax - ymin)
    if pixel_size_m and pixel_size_m > 0:
        side = math.ceil(side / pixel_size_m) * pixel_size_m
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half = side / 2.0
    return cx - half, cy - half, cx + half, cy + half


def rasterize_polygon_mask(dem: xr.DataArray, polygon) -> xr.DataArray:
    """Rasterize ``polygon`` onto the DEM grid as a uint8 in/out mask."""
    poly = polygon if polygon.is_valid else polygon.buffer(0)
    transform = dem.rio.transform(recalc=False)
    mask = rasterize(
        [(mapping(poly), 1)],
        out_shape=(dem.sizes["y"], dem.sizes["x"]),
        transform=transform,
        all_touched=True,
        dtype="uint8",
    )
    out = xr.DataArray(mask, coords={"y": dem["y"], "x": dem["x"]}, dims=("y", "x"))
    out.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    out.rio.write_crs(TARGET_EPSG, inplace=True)
    out.rio.write_transform(transform, inplace=True)
    return out


def apply_mask_rules(dem: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Apply the in/out masking rules. Fully vectorized (no per-pixel loop)."""
    values = np.asarray(dem.values, dtype=np.float32)
    inside = np.asarray(mask.values) == 1

    out = np.where(inside, values, np.nan).astype(np.float32)
    # Inside the polygon, a non-finite source value becomes 0.0.
    out = np.where(inside & ~np.isfinite(out), np.float32(0.0), out).astype(np.float32)

    result = _wrap_array(out, dem.rio.transform(recalc=False), TARGET_EPSG)
    return result


def polygon_stats(dem_masked: xr.DataArray, mask: xr.DataArray) -> dict:
    """Coverage statistics inside the polygon, used for quality gating."""
    inside = np.asarray(mask.values) == 1
    values = np.asarray(dem_masked.values)

    total = int(inside.sum())
    finite_inside = np.isfinite(values) & inside
    zeros = int((finite_inside & (values == 0)).sum())
    good = int((finite_inside & (values != 0)).sum())
    return {
        "total_px": total,
        "zeros_px": zeros,
        "good_px": good,
        "zero_pct": round(100.0 * zeros / total, 2) if total else 0.0,
        "good_pct": round(100.0 * good / total, 2) if total else 0.0,
    }


def _pad_to_multiple(arr: np.ndarray, factor: int) -> np.ndarray:
    """Pad the bottom/right edges with NaN so both dims divide by ``factor``."""
    h, w = arr.shape
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor
    if not pad_h and not pad_w:
        return arr
    return np.pad(
        arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan
    )


def coarsen_3x3(dem10: xr.DataArray) -> xr.DataArray:
    """Aggregate a 10 m DEM to 30 m by 3x3 blocks, ignoring zeros and NaNs.

    Block rule:

    * all 9 values non-finite -> NaN;
    * finite values present but all zero -> 0.0;
    * otherwise -> mean of the finite, non-zero values.

    Fully vectorized: the array is reshaped to ``(h//3, 3, w//3, 3)`` and the
    whole reduction is a handful of whole-array numpy operations over the two
    block axes, rather than a per-cell Python loop (``h//3 * w//3`` iterations
    each slicing and ravelling a block), which would dominate runtime on large
    DEM windows.
    """
    transform = dem10.rio.transform(recalc=False)
    res_x, res_y = abs(float(transform.a)), abs(float(transform.e))
    if not (9.5 <= res_x <= 10.5 and 9.5 <= res_y <= 10.5):
        raise ValueError(f"coarsen_3x3 expects a 10 m grid, got {res_x} x {res_y} m")

    data = _pad_to_multiple(np.asarray(dem10.values, dtype=np.float32), 3)
    h, w = data.shape
    blocks = data.reshape(h // 3, 3, w // 3, 3)

    finite = np.isfinite(blocks)
    nonzero = finite & (blocks != 0)

    # Sum/count of finite non-zero values per block, over the two block axes.
    contrib = np.where(nonzero, blocks, np.float32(0.0))
    total = contrib.sum(axis=(1, 3))
    count = nonzero.sum(axis=(1, 3))
    any_finite = finite.any(axis=(1, 3))

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.float32(0.0))

    out = np.where(any_finite, mean, np.nan).astype(np.float32)

    new_transform = transform * transform.scale(3, 3)
    result = _wrap_array(out, new_transform, TARGET_EPSG)
    return result


def resample_to(
    dem: xr.DataArray, target_res_m: float, method: str = RESAMPLE_METHOD
) -> xr.DataArray:
    """Reproject/resample a DEM to ``target_res_m`` within EPSG:3413."""
    resampling = (
        Resampling.bilinear if (method or "bilinear").lower() == "bilinear"
        else Resampling.nearest
    )
    out = dem.rio.reproject(
        dst_crs=f"EPSG:{TARGET_EPSG}",
        resolution=target_res_m,
        resampling=resampling,
        nodata=np.nan,
    )
    out.rio.write_nodata(np.nan, inplace=True)
    return out.astype("float32")


def interpolate_interior_holes(dem: np.ndarray) -> np.ndarray:
    """Fill NaN holes that are fully enclosed by valid data.

    Only *interior* holes are filled: ``binary_fill_holes`` on the validity mask
    identifies gaps surrounded by data, so the NaN margin outside the glacier
    polygon is left untouched. Linear interpolation, with nearest-neighbour for
    any point the linear pass cannot reach.
    """
    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import binary_fill_holes
    except ImportError:
        return dem.copy()

    valid = np.isfinite(dem)
    if valid.all() or not valid.any():
        return dem.copy()

    holes = binary_fill_holes(valid) & ~valid
    if not holes.any():
        return dem.copy()

    yy, xx = np.nonzero(valid)
    points = np.column_stack([yy, xx])
    values = dem[valid]
    hy, hx = np.nonzero(holes)
    targets = np.column_stack([hy, hx])

    filled = griddata(points, values, targets, method="linear", fill_value=np.nan)
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = griddata(points, values, targets[missing], method="nearest")

    out = dem.copy()
    out[holes] = filled.astype(np.float32, copy=False)
    return out


def build_output_resolutions(
    dem_masked: xr.DataArray,
    base_resolution_m: int,
    resolutions: tuple[int, ...] = OUTPUT_RESOLUTIONS_M,
) -> dict[int, xr.DataArray]:
    """Derive every requested output resolution from the masked source DEM.

    From a 10 m source, the 30 m product uses the zero-aware 3x3 block
    aggregation (falling back to bilinear if the grid is not exactly 10 m);
    every other resolution is a bilinear resample. From a 30 m source, 30 m is
    passed through and the finer resolutions are bilinear upsamples.
    """
    outputs: dict[int, xr.DataArray] = {}
    for res in resolutions:
        if res == base_resolution_m:
            outputs[res] = dem_masked
        elif res == 30 and base_resolution_m == 10:
            try:
                outputs[30] = coarsen_3x3(dem_masked)
            except Exception:
                outputs[30] = resample_to(dem_masked, 30)
        else:
            outputs[res] = resample_to(dem_masked, res)
    return outputs
