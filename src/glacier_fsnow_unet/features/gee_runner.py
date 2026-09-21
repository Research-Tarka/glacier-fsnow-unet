"""Generic Earth Engine runner for the declarative climate source specs.

Purpose
-------
One runner that can execute any :class:`~glacier_fsnow_unet.features.source_defs.SourceSpec`,
rather than a separate script per climate variable. All five sources share the
same shape (image collection -> per-glacier reduction -> yearly Parquet), so
one declarative-spec-driven runner covers all of them without duplicating the
Earth Engine plumbing five times.

Inputs
------
- A ``SourceSpec`` and the glacier registry.

Outputs
-------
- One cached Parquet per (source, year) under the raw cache root, then one
  consolidated absolute-value frame per source.

Caching
-------
Each (source, year) result is cached to Parquet and reused unless ``force`` is
set, so an interrupted multi-year download resumes without re-querying Earth
Engine.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

#: Max bands composed into one reduceRegions call. A prior session's live
#: probe blamed "User memory limit exceeded" on wide multi-band requests and
#: added chunking here, but a clean-shell re-test (no other GEE-calling
#: processes present) showed every chunk failing identically, including a
#: single-band chunk -- disproving that hypothesis. Issuing one reduceRegions
#: per source-year with every band composed into one image (no chunking) is
#: what this value now matches; kept as a knob (not removed) in case a
#: genuinely wide source needs it later, not because 2 is known-correct.
MAX_BANDS_PER_REQUEST = 64

import numpy as np
import pandas as pd

from .source_defs import SEASON_MONTHS, BandRequest, SourceSpec

log = logging.getLogger(__name__)


def cache_path(root: Path, slug: str, year: int) -> Path:
    """Path of the per-(source, year) cache file."""
    return Path(root) / slug / f"{slug}__{int(year)}.parquet"


def empty_frame(spec: SourceSpec) -> pd.DataFrame:
    """An empty frame with this source's full output schema."""
    return pd.DataFrame(columns=["id_glims", "year", *spec.value_columns])


def build_reduced_image(ee: Any, spec: SourceSpec, year: int, requests: Optional[Sequence[BandRequest]] = None):
    """Build one multi-band image whose bands are (a subset of) this source's output variables.

    Each :class:`BandRequest` becomes one band: the collection is filtered to
    the request's temporal window, reduced over time with the requested
    reducer, then scaled and offset into physical units. ``requests`` defaults
    to every request the source declares; pass a smaller slice to build a
    chunk of the full image (see :func:`run_source_year`'s band-chunking,
    needed for wide/hourly collections that exceed Earth Engine's per-request
    compute memory budget when every band is composed into one image).
    """
    images = []
    for request in requests if requests is not None else spec.requests:
        months = (
            (request.month,)
            if request.window == "monthly"
            else SEASON_MONTHS[request.window]
        )
        collection = _filter_collection(ee, spec, year, months).select([request.band])

        if request.reducer == "sum":
            reduced = collection.sum()
        elif request.reducer == "min":
            reduced = collection.min()
        elif request.reducer == "max":
            reduced = collection.max()
        else:
            reduced = collection.mean()

        if request.scale != 1.0:
            reduced = reduced.multiply(request.scale)
        if request.offset != 0.0:
            reduced = reduced.add(request.offset)

        images.append(reduced.rename([request.out]))

    if not images:
        raise ValueError(f"Source '{spec.slug}' declares no band requests.")

    image = images[0]
    for extra in images[1:]:
        image = image.addBands(extra)
    return image


#: Collections whose per-request compute cost scales with raw image count
#: rather than band count or output size -- confirmed live for
#: ECMWF/ERA5/HOURLY, where reducing a full calendar year (~8760 hourly
#: images) in one reduceRegions call reproducibly fails with "User memory
#: limit exceeded" even for a single band on a brand-new project, while the
#: same reduction over a single week (168 images) succeeds in seconds. A
#: 4-source reproducibility check (ERA5_Land, TerraClimate, Daymet_V4,
#: MERRA2_Aerosols) ruled out transient GEE load as the explanation: those
#: sources' occasional failures always succeeded on an immediate retry with
#: no code change, whereas this collection failed identically three times
#: in a row. Sources on collections in this set are aggregated month by
#: month (see :func:`_reduce_month_chunked`) instead of over the whole
#: requested window in one call.
HOURLY_COLLECTIONS: frozenset[str] = frozenset({"ECMWF/ERA5/HOURLY"})


def _month_windows(year: int, months: Sequence[int]) -> list[tuple[int, str, str]]:
    """Split a (possibly winter-spanning) month set into per-month (year, start, end) windows."""
    month_set = sorted(set(int(m) for m in months))
    is_winter = set(month_set) == {12, 1, 2}
    windows = []
    for month in month_set:
        # Winter (DJF) is December of the *previous* year plus January and
        # February of this one -- the same convention _filter_collection
        # uses. Annual (all 12 months) keeps December in the current year.
        window_year = year - 1 if month == 12 and is_winter else year
        start = f"{window_year}-{month:02d}-01"
        end_year = window_year + 1 if month == 12 else window_year
        end_month = 1 if month == 12 else month + 1
        end = f"{end_year}-{end_month:02d}-01"
        windows.append((month, start, end))
    return windows


def _reduce_month_chunked(
    ee: Any,
    spec: SourceSpec,
    request: BandRequest,
    year: int,
    glaciers: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one band request month by month, combined client-side.

    Each month is its own ``reduceRegions`` call (small enough to never hit
    the hourly-collection compute ceiling), then combined into the same
    annual/seasonal aggregate a single-call reduction would have produced:
    weighted by each month's actual image count for ``mean`` (hourly image
    counts per month are not equal -- February has fewer hours than any
    other month, so weighting by calendar days would be wrong), and by plain
    summation for ``sum``. ``min``/``max`` reduce further over the monthly
    extrema.
    """
    months = (
        (request.month,) if request.window == "monthly" else SEASON_MONTHS[request.window]
    )
    windows = _month_windows(year, months)

    monthly_frames = []
    monthly_counts = []
    for month, start, end in windows:
        collection = ee.ImageCollection(spec.collection).filterDate(start, end).select([request.band])
        count = int(collection.size().getInfo())
        if count == 0:
            continue
        if request.reducer == "sum":
            reduced = collection.sum()
        elif request.reducer == "min":
            reduced = collection.min()
        elif request.reducer == "max":
            reduced = collection.max()
        else:
            reduced = collection.mean()
        reduced = reduced.rename([request.out])
        frame = reduce_regions_to_frame(ee, reduced, glaciers, year, spec.scale_m, [request.out])
        monthly_frames.append(frame.set_index("id_glims")[request.out])
        monthly_counts.append(count)

    if not monthly_frames:
        return pd.DataFrame(columns=["id_glims", "year", request.out])

    stacked = pd.concat(monthly_frames, axis=1)
    stacked.columns = range(len(monthly_frames))
    weights = pd.Series(monthly_counts, dtype=float)

    if request.reducer == "sum":
        combined = stacked.sum(axis=1)
    elif request.reducer == "min":
        combined = stacked.min(axis=1)
    elif request.reducer == "max":
        combined = stacked.max(axis=1)
    else:
        combined = (stacked * weights.values).sum(axis=1) / weights.sum()

    if request.scale != 1.0:
        combined = combined * request.scale
    if request.offset != 0.0:
        combined = combined + request.offset

    out = combined.rename(request.out).reset_index()
    out.insert(1, "year", int(year))
    return out


def _filter_collection(ee: Any, spec: SourceSpec, year: int, months: Sequence[int]):
    """Filter a collection to the given months of a year.

    Winter (Dec, Jan, Feb) is treated as December of the *previous* year plus
    January and February of this one, so a winter aggregate is a contiguous
    season rather than a calendar-year mixture.
    """
    collection = ee.ImageCollection(spec.collection)
    month_set = set(int(m) for m in months)

    if month_set == {12, 1, 2}:
        window = collection.filterDate(f"{year - 1}-12-01", f"{year}-03-01")
        return window
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    filtered = collection.filterDate(start, end)
    if month_set != set(range(1, 13)):
        filtered = filtered.filter(
            ee.Filter.calendarRange(min(month_set), max(month_set), "month")
        )
    return filtered


def apply_derived_columns(frame: pd.DataFrame, spec: SourceSpec) -> pd.DataFrame:
    """Compute the arithmetic columns a source declares as ``derived``.

    Kept out of the Earth Engine graph because they are cheap local arithmetic
    over already-reduced values, and easier to test here.
    """
    out = frame.copy()
    columns = set(out.columns)

    def _has(*names: str) -> bool:
        return all(name in columns for name in names)

    if "terraclimate_tavg_annual" in spec.derived and _has(
        "terraclimate_tmin_annual", "terraclimate_tmax_annual"
    ):
        out["terraclimate_tavg_annual"] = (
            out["terraclimate_tmin_annual"] + out["terraclimate_tmax_annual"]
        ) / 2.0
        out["terraclimate_diurnal_range_annual"] = (
            out["terraclimate_tmax_annual"] - out["terraclimate_tmin_annual"]
        )

    if "daymet_tavg_annual" in spec.derived and _has(
        "daymet_tmin_annual", "daymet_tmax_annual"
    ):
        out["daymet_tavg_annual"] = (
            out["daymet_tmin_annual"] + out["daymet_tmax_annual"]
        ) / 2.0
        out["daymet_diurnal_range_annual"] = (
            out["daymet_tmax_annual"] - out["daymet_tmin_annual"]
        )

    if "wind_speed_summer" in spec.derived and _has("u10_summer", "v10_summer"):
        out["wind_speed_summer"] = np.hypot(out["u10_summer"], out["v10_summer"])

    if "era5_wind_speed_summer" in spec.derived and _has(
        "era5_u10_summer", "era5_v10_summer"
    ):
        out["era5_wind_speed_summer"] = np.hypot(
            out["era5_u10_summer"], out["era5_v10_summer"]
        )

    if "diurnal_range_annual" in spec.derived and _has("t2m_max_annual", "t2m_min_annual"):
        out["diurnal_range_annual"] = out["t2m_max_annual"] - out["t2m_min_annual"]

    if "dewpoint_gap_summer" in spec.derived and _has("t2m_summer", "dewpoint_summer"):
        out["dewpoint_gap_summer"] = out["t2m_summer"] - out["dewpoint_summer"]

    if "vpd_summer" in spec.derived and _has("t2m_summer", "dewpoint_summer"):
        # Vapour-pressure deficit from Magnus-form saturation vapour pressure,
        # evaluated at air temperature and at dewpoint (both in degrees C, kPa).
        out["vpd_summer"] = _saturation_vapour_pressure_kpa(
            out["t2m_summer"]
        ) - _saturation_vapour_pressure_kpa(out["dewpoint_summer"])

    return out


def _saturation_vapour_pressure_kpa(temperature_c: pd.Series) -> pd.Series:
    """Magnus-Tetens saturation vapour pressure, in kPa, from degrees C."""
    return 0.6108 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))


def reduce_regions_to_frame(
    ee: Any,
    image,
    glaciers: pd.DataFrame,
    year: int,
    scale_m: float,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Reduce a multi-band image over each glacier's centroid, in one call.

    Builds one ``FeatureCollection`` of glacier centroid points and issues a
    single ``reduceRegions``, rather than one request per glacier. A point
    (not the glacier's full analysis-window rectangle) is deliberate: these
    climate sources are coarse-resolution/global (11 km-55 km native pixel
    size for the reanalysis products, hourly time steps for ERA5 Reanalysis
    and MERRA-2), so a single representative point per glacier is the
    physically meaningful sample and keeps the server-side reduction cheap.
    Reducing over the full multi-km-wide rectangle instead measurably risks
    Earth Engine's "User memory limit exceeded" on the hourly collections,
    confirmed live in this session on ``ECMWF/ERA5/HOURLY``.
    """
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row["centroid_lon"]), float(row["centroid_lat"])]),
            {"id_glims": str(row["glims_id"])},
        )
        for _, row in glaciers.iterrows()
    ]
    if not features:
        return pd.DataFrame(columns=["id_glims", "year", *value_columns])

    collection = ee.FeatureCollection(features)
    # tileScale=2 tells Earth Engine to split the computation into smaller
    # server-side tiles, keeping a safety margin against per-tile memory limits.
    reduced = image.reduceRegions(
        collection=collection, reducer=ee.Reducer.mean(), scale=float(scale_m),
        tileScale=2.0,
    )

    info = reduced.getInfo()
    rows = []
    for feature in info.get("features", []):
        properties = feature.get("properties", {}) or {}
        # reduceRegions on a single-band image names the reduced property
        # after the reducer (e.g. "mean"), not the band -- only a multi-band
        # image gets one output property per band name. Read back whatever
        # properties actually came back rather than looking each expected
        # column up by name, then align to value_columns positionally when
        # a single-band request produced the generic "mean" key.
        row = {"id_glims": properties.get("id_glims"), "year": int(year)}
        non_id_values = {k: v for k, v in properties.items() if k != "id_glims"}
        if len(value_columns) == 1 and list(non_id_values) == ["mean"]:
            non_id_values = {value_columns[0]: non_id_values["mean"]}
        for column in value_columns:
            value = non_id_values.get(column)
            row[column] = float(value) if value is not None else np.nan
        rows.append(row)

    return pd.DataFrame(rows, columns=["id_glims", "year", *value_columns])


def run_source_year(
    ee: Any,
    spec: SourceSpec,
    glaciers: pd.DataFrame,
    year: int,
    cache_root: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch (or load from cache) one source-year for all glaciers."""
    path = cache_path(cache_root, spec.slug, year)
    if path.exists() and not force:
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache unreadable at %s (%s); refetching.", path, exc)

    chunks: list[pd.DataFrame] = []
    requests = list(spec.requests)

    if spec.collection in HOURLY_COLLECTIONS:
        # This collection's per-request compute cost scales with raw image
        # count, not band count -- chunking by band (the else branch) still
        # reduces the whole requested window (up to a full year, ~8760
        # hourly images) in one server-side call per chunk, which is exactly
        # what fails here regardless of how few bands are in that chunk (see
        # HOURLY_COLLECTIONS). Reduce month by month instead.
        for request in requests:
            chunks.append(_reduce_month_chunked(ee, spec, request, year, glaciers))
    else:
        for start in range(0, len(requests), MAX_BANDS_PER_REQUEST):
            chunk_requests = requests[start : start + MAX_BANDS_PER_REQUEST]
            chunk_columns = [request.out for request in chunk_requests]
            image = build_reduced_image(ee, spec, year, requests=chunk_requests)
            chunks.append(
                reduce_regions_to_frame(ee, image, glaciers, year, spec.scale_m, chunk_columns)
            )

    frame = chunks[0]
    for chunk in chunks[1:]:
        frame = frame.merge(chunk.drop(columns=["year"]), on="id_glims", how="outer")
    frame = apply_derived_columns(frame, spec)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def run_source(
    ee: Any,
    spec: SourceSpec,
    glaciers: pd.DataFrame,
    cache_root: Path,
    years: Optional[Sequence[int]] = None,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch every year of one source and return the consolidated frame."""
    target = list(years) if years is not None else spec.years()
    parts: list[pd.DataFrame] = []

    for year in target:
        if verbose:
            print(f"    {spec.slug} {year} ...", flush=True)
        try:
            parts.append(
                run_source_year(ee, spec, glaciers, year, cache_root, force=force)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("%s %s failed: %s", spec.slug, year, exc)

    if not parts:
        return empty_frame(spec)
    return pd.concat(parts, ignore_index=True)


#: Per-unit timeout for the process-pool path (seconds). A single stalled
#: HTTP request inside earthengine-api/GDAL was confirmed, live, to hang a
#: worker indefinitely with no exception and near-zero CPU use -- neither
#: earthengine-api nor its underlying googleapiclient/httplib2 stack applies
#: a reliable timeout by default. A dead worker process is killed and its
#: unit reported failed rather than freezing the whole run forever.
WORK_UNIT_TIMEOUT_S = 900


def _run_source_year_worker(
    result_queue,
    project: str,
    spec: SourceSpec,
    year: int,
    glaciers: pd.DataFrame,
    cache_root: Path,
    force: bool,
) -> None:
    """Top-level (picklable) worker body: runs in its own OS process.

    Does its own ``ee`` import and ``ee.Initialize`` -- there is no shared
    Python state with the parent or sibling workers, unlike a thread pool
    where the ``ee`` module's authenticated project is one process-wide
    mutable value every thread shares and re-mutates. Puts its result (or
    exception) on ``result_queue`` rather than returning it, since a real
    ``multiprocessing.Process`` (not an executor) is used so the parent can
    ``terminate()`` a specific hung worker without tearing down the whole
    pool -- see :func:`run_sources_multi_project`.
    """
    import ee as ee_module

    try:
        ee_module.Initialize(project=project)
        frame = run_source_year(ee_module, spec, glaciers, year, cache_root, force=force)
        result_queue.put(("ok", frame))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("error", str(exc)))


#: Per-unit timeout (seconds). A single stalled HTTP request inside
#: earthengine-api/GDAL was confirmed, live, to hang a worker indefinitely
#: with no exception and near-zero CPU use -- neither earthengine-api nor
#: its underlying googleapiclient/httplib2 stack applies a reliable timeout
#: by default, and a hung worker never posts a result. A worker that misses
#: this deadline is force-killed and its unit reported failed, rather than
#: freezing the whole run forever.
WORK_UNIT_TIMEOUT_S = 900


def run_sources_multi_project(
    specs: Sequence[SourceSpec],
    glaciers: pd.DataFrame,
    cache_root: Path,
    projects: Sequence[str],
    years: Optional[Sequence[int]] = None,
    force: bool = False,
    max_workers: int = 1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch every (source, year) unit across multiple GEE projects in parallel.

    Uses one OS process per GEE project, sized by ``max_workers`` instead of
    one hardcoded script per project: up to ``min(max_workers, len(projects), len(units))``
    ``multiprocessing.Process`` workers run concurrently, each handling one
    (source, year) work unit round-robined across ``projects``, each doing
    its own ``ee.Initialize`` -- no shared, mutable process-wide ``ee``
    state the way a thread pool has.

    A prior thread-pool implementation of this function was live-tested
    against a real multi-year, multi-source run and found to hang
    indefinitely partway through, the whole process stuck at near-zero CPU
    with no exception -- consistent with one thread's stalled network call
    blocking every thread sharing that process (GIL contention plus a
    shared ``ee``/HTTP client state), a failure mode true OS-level process
    isolation does not have: a hung sibling process cannot block others,
    and it alone can be force-killed. A plain ``ProcessPoolExecutor`` was
    tried first and rejected: ``Future.result(timeout=...)`` only stops the
    *parent* from waiting, it does not kill the still-hung worker process
    underneath, leaking it. Real ``multiprocessing.Process`` objects are
    used instead so a timed-out worker can be ``.terminate()``-d for real.
    """
    import multiprocessing as mp

    if not projects:
        raise ValueError("run_sources_multi_project requires at least one GEE project id.")

    units: list[tuple[SourceSpec, int]] = []
    target_years_by_spec = {
        spec.slug: (list(years) if years is not None else spec.years()) for spec in specs
    }
    for spec in specs:
        for year in target_years_by_spec[spec.slug]:
            units.append((spec, year))

    max_concurrent = min(max(1, int(max_workers)), max(1, len(projects)), max(1, len(units)))
    ctx = mp.get_context("spawn")

    results: dict[str, list[pd.DataFrame]] = {spec.slug: [] for spec in specs}

    pending = list(enumerate(units))
    running: dict[int, tuple] = {}  # unit_index -> (process, queue, slug, year, started_at)

    def _launch(unit_index: int, spec: SourceSpec, year: int) -> None:
        project = projects[unit_index % len(projects)]
        if verbose:
            print(f"    {spec.slug} {year} [project={project}] ...", flush=True)
        queue = ctx.Queue()
        process = ctx.Process(
            target=_run_source_year_worker,
            args=(queue, project, spec, year, glaciers, cache_root, force),
            daemon=True,
        )
        process.start()
        running[unit_index] = (process, queue, spec.slug, year, time.time())

    while pending or running:
        while pending and len(running) < max_concurrent:
            unit_index, (spec, year) = pending.pop(0)
            _launch(unit_index, spec, year)

        finished = []
        for unit_index, (process, queue, slug, year, started_at) in list(running.items()):
            if process.is_alive():
                if time.time() - started_at > WORK_UNIT_TIMEOUT_S:
                    log.warning(
                        "%s %s timed out after %ds; terminating worker process",
                        slug, year, WORK_UNIT_TIMEOUT_S,
                    )
                    process.terminate()
                    process.join(timeout=10)
                    finished.append(unit_index)
                continue

            try:
                status, payload = queue.get_nowait()
                if status == "ok":
                    results[slug].append(payload)
                else:
                    log.warning("%s %s failed: %s", slug, year, payload)
            except Exception:  # noqa: BLE001 -- queue empty: process died without posting a result
                log.warning("%s %s failed: worker process exited without a result", slug, year)
            process.join(timeout=10)
            finished.append(unit_index)

        for unit_index in finished:
            del running[unit_index]

        if running and not finished:
            time.sleep(0.5)

    return {
        spec.slug: (
            pd.concat(results[spec.slug], ignore_index=True)
            if results[spec.slug]
            else empty_frame(spec)
        )
        for spec in specs
    }
