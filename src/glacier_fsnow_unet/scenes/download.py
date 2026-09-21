"""Scene discovery and download from Google Earth Engine.

Purpose
-------
For each glacier, sensor, and year in the ablation window, find the available
scenes in the relevant GEE collection, discard those already downloaded (per
the SQLite cache) or too cloudy, and fetch the remaining band stacks.

Inputs
------
- The glacier registry (stage 1) and the split assignment (stage 3).
- Earth Engine credentials and a project id (see ``gee_auth``).

Outputs
-------
- Per-glacier scene arrays written under the split's data root.
- Cache entries recording every scene as downloaded or failed.

Design notes
------------
Five sensors (Landsat 5/7/8/9/Sentinel-2) share the same search-filter-download
shape, differing only in collection id, band names, year range, and cloud
logic. Rather than one near-duplicate module per sensor, this module keeps
**one** loop parameterized by
:class:`~glacier_fsnow_unet.scenes.sensors.SensorSpec`, so a change to the
retention rules applies to all five sensors at once.

Parallel output splits (independent GEE projects/output roots) are handled by
a single entry point, ``scripts/04_download_scenes.py --split N``, rather than
one script per split index.

Re-audit note: within one split, ``run_split`` iterates glaciers x sensors
strictly sequentially. ``config.scene_download.download_workers`` and
``glacier_workers`` are declared but not read anywhere in this module --
parallelism across splits exists (separate GEE projects/processes), but not
within a split. This was checked, not overlooked: unlike the DEM stage's
threaded per-glacier fetch (independent public COG reads with no shared
state), threading glacier x sensor downloads here would run concurrent
writes against the *same* glacier's zarr store from multiple sensors, and
concurrent ``getInfo()``/export calls against one GEE project, both of which
are already rate-limited server-side per project. ``scenes/zarr_store.py``'s
per-path lock would keep the store consistent, but a bounded worker pool
would still need care around GEE quota backoff that this repo has not yet
validated live. Left sequential deliberately so the small real fixture download this repo
runs against live GEE is guaranteed correct, with the unused config knobs
flagged here rather than silently removed, in case a future session wires
them up once quota backoff is validated live.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .cache import SceneCache
from .sensors import SENSOR_ORDER, SensorSpec, get_sensor, is_year_allowed, season_bounds


@dataclass
class SceneRecord:
    """One scene discovered in an Earth Engine collection."""

    scene_id: str
    sensor: str
    year: int
    glims_id: str
    cloud_fraction: float = 0.0
    aoi_coverage: float = 1.0
    properties: dict = field(default_factory=dict)


@dataclass
class DownloadStats:
    """Tallies for one download run, printed as the end-of-run summary."""

    discovered: int = 0
    downloaded: int = 0
    skipped_cached: int = 0
    skipped_cloud: int = 0
    skipped_coverage: int = 0
    failed: int = 0
    glaciers: int = 0

    def merge(self, other: "DownloadStats") -> None:
        self.discovered += other.discovered
        self.downloaded += other.downloaded
        self.skipped_cached += other.skipped_cached
        self.skipped_cloud += other.skipped_cloud
        self.skipped_coverage += other.skipped_coverage
        self.failed += other.failed
        self.glaciers += other.glaciers

    def summary(self) -> str:
        return (
            f"glaciers={self.glaciers} discovered={self.discovered} "
            f"downloaded={self.downloaded} cached={self.skipped_cached} "
            f"cloudy={self.skipped_cloud} low_coverage={self.skipped_coverage} "
            f"failed={self.failed}"
        )


def build_aoi(ee, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    """Build the Earth Engine geometry of a glacier's analysis window."""
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], "EPSG:4326", False)


def search_scenes(
    ee,
    spec: SensorSpec,
    aoi,
    year: int,
    glims_id: str,
    max_cloud_fraction: float = 1.0,
) -> list[SceneRecord]:
    """List the scenes of one sensor/year intersecting a glacier's window.

    Applies the ablation-season date filter (1 July - 30 September) and the
    spatial filter, then reads back the per-scene metadata needed to decide
    retention. Cloud fraction uses the collection's scene-level property, which
    is a coarse pre-filter; the paper's authoritative 30% test is applied to the
    *model's own* cloud classification inside the RGI polygon, downstream.
    """
    start, end = season_bounds(year)
    collection = (
        ee.ImageCollection(spec.collection).filterBounds(aoi).filterDate(start, end)
    )

    cloud_property = "CLOUDY_PIXEL_PERCENTAGE" if spec.is_sentinel else "CLOUD_COVER"

    try:
        size = int(collection.size().getInfo())
        # ee.Collection.toList(count) rejects count=0 ("must be positive"), so
        # an empty collection (no scenes this glacier/sensor/year) must short
        # -circuit here rather than call toList at all -- this is the normal,
        # expected case for most glacier/sensor/year combinations, not an error.
        info = collection.toList(size).getInfo() if size > 0 else []
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Earth Engine query failed for {spec.key} {year} ({glims_id}): {exc}"
        ) from exc

    records: list[SceneRecord] = []
    for item in info:
        properties = item.get("properties", {}) or {}
        scene_id = str(
            properties.get("system:index") or item.get("id", "") or ""
        )
        raw_cloud = properties.get(cloud_property)
        cloud = float(raw_cloud) / 100.0 if raw_cloud is not None else 0.0
        records.append(
            SceneRecord(
                scene_id=scene_id,
                sensor=spec.key,
                year=year,
                glims_id=glims_id,
                cloud_fraction=cloud,
                properties=properties,
            )
        )
    return records


def filter_scenes(
    records: Sequence[SceneRecord],
    cache: Optional[SceneCache],
    max_cloud_fraction: float = 1.0,
    min_aoi_coverage: float = 0.0,
    max_scenes_per_year: int = 9999,
    stats: Optional[DownloadStats] = None,
) -> list[SceneRecord]:
    """Drop already-downloaded, too-cloudy, or poorly-covering scenes.

    The cache check is the resume mechanism: a scene recorded ``OK`` is skipped,
    so an interrupted run resumes exactly where it stopped, at scene
    granularity, without any manually maintained resume point.
    """
    stats = stats or DownloadStats()
    kept: list[SceneRecord] = []

    for record in records:
        if cache is not None and cache.is_done(
            record.glims_id, record.sensor, record.year, record.scene_id
        ):
            stats.skipped_cached += 1
            continue
        if record.cloud_fraction > max_cloud_fraction:
            stats.skipped_cloud += 1
            continue
        if record.aoi_coverage < min_aoi_coverage:
            stats.skipped_coverage += 1
            continue
        kept.append(record)

    if max_scenes_per_year and len(kept) > max_scenes_per_year:
        # Keep the least cloudy scenes when the per-year budget is exceeded.
        kept = sorted(kept, key=lambda r: r.cloud_fraction)[:max_scenes_per_year]

    return kept


def download_glacier_sensor(
    ee,
    spec: SensorSpec,
    glacier: dict,
    years: Sequence[int],
    output_root: Path,
    cache: Optional[SceneCache] = None,
    max_cloud_fraction: float = 1.0,
    min_aoi_coverage: float = 0.0,
    max_scenes_per_year: int = 9999,
    fetch: Optional[Callable[[SceneRecord, SensorSpec, dict, Path], None]] = None,
    verbose: bool = True,
) -> DownloadStats:
    """Discover and download every scene of one sensor for one glacier.

    ``fetch`` performs the actual raster retrieval and write; it is injected so
    the discovery/filter/cache logic above can be exercised without network
    access (see ``tests/unit/test_scene_download.py``). When omitted, only
    discovery and filtering run and nothing is written -- a useful dry run.
    """
    stats = DownloadStats(glaciers=1)
    glims_id = str(glacier["glims_id"])

    if cache is not None and cache.is_sensor_done(glims_id, spec.key):
        if verbose:
            print(f"    [{spec.key}] {glims_id}: already complete, skipping")
        return stats

    aoi = build_aoi(
        ee,
        float(glacier["min_lon"]),
        float(glacier["min_lat"]),
        float(glacier["max_lon"]),
        float(glacier["max_lat"]),
    )

    glacier_dir = Path(output_root) / glims_id
    for year in years:
        if not is_year_allowed(spec.key, year):
            continue

        records = search_scenes(
            ee, spec, aoi, year, glims_id, max_cloud_fraction=max_cloud_fraction
        )
        stats.discovered += len(records)

        selected = filter_scenes(
            records,
            cache,
            max_cloud_fraction=max_cloud_fraction,
            min_aoi_coverage=min_aoi_coverage,
            max_scenes_per_year=max_scenes_per_year,
            stats=stats,
        )

        for record in selected:
            if fetch is None:
                continue
            try:
                fetch(record, spec, glacier, glacier_dir)
                if cache is not None:
                    cache.set_ok(glims_id, spec.key, year, record.scene_id)
                stats.downloaded += 1
            except Exception as exc:  # noqa: BLE001
                if cache is not None:
                    cache.set_failed(
                        glims_id, spec.key, year, record.scene_id, str(exc)
                    )
                stats.failed += 1
                if verbose:
                    print(f"    [{spec.key}] {glims_id} {year} {record.scene_id}: {exc}")

    # Only mark a sensor "done" once real fetches were actually attempted --
    # with fetch=None (a discovery/dry-run caller, or a caller that forgot to
    # wire one up) nothing was ever written, and marking it done anyway would
    # make every later real run silently skip this glacier/sensor forever.
    # A prior version of this repo's own CLI hit exactly this: fetch=None was
    # the default and every scene was "discovered" and "filtered" but never
    # downloaded, yet the cache still recorded the sensor as complete.
    if cache is not None and stats.failed == 0 and fetch is not None:
        cache.set_sensor_done(glims_id, spec.key)

    return stats


def run_split(
    ee,
    glaciers: Iterable[dict],
    output_root: Path,
    cache: Optional[SceneCache] = None,
    sensors: Sequence[str] = SENSOR_ORDER,
    until_year: Optional[int] = None,
    max_cloud_fraction: float = 1.0,
    min_aoi_coverage: float = 0.0,
    max_scenes_per_year: int = 9999,
    fetch: Optional[Callable] = None,
    verbose: bool = True,
) -> DownloadStats:
    """Download every sensor for every glacier of one split.

    Sensors are processed in :data:`SENSOR_ORDER` (oldest first), so an
    interrupted run resumes in a predictable order.
    """
    total = DownloadStats()
    glacier_list = list(glaciers)

    for index, glacier in enumerate(glacier_list, start=1):
        glims_id = str(glacier["glims_id"])
        if verbose:
            print(f"  [{index}/{len(glacier_list)}] {glims_id}")

        for sensor_key in sensors:
            spec = get_sensor(sensor_key)
            years = spec.years(until=until_year)
            stats = download_glacier_sensor(
                ee,
                spec,
                glacier,
                years,
                output_root,
                cache=cache,
                max_cloud_fraction=max_cloud_fraction,
                min_aoi_coverage=min_aoi_coverage,
                max_scenes_per_year=max_scenes_per_year,
                fetch=fetch,
                verbose=verbose,
            )
            stats.glaciers = 0  # counted once per glacier below
            total.merge(stats)

        total.glaciers += 1

    return total


def default_fetch(ee_module, record: SceneRecord, spec: SensorSpec, glacier: dict, glacier_dir: Path) -> None:
    """The real per-scene pixel fetch: bridges discovery (this module, GEE
    metadata only) to ``gee_fetch.fetch_and_store_scene`` (actual pixels,
    including Landsat 7/8/9 pansharpening).

    A plain top-level function (not a closure capturing ``ee``) so it is
    picklable and safe to reference by name across a process boundary --
    see :func:`run_glaciers_multi_project`, which runs each glacier in its
    own OS process, each with its own freshly-imported ``ee``.
    """
    import datetime as _dt

    from ..dem.engine import window_polygon_from_registry
    from .gee_fetch import fetch_and_store_scene
    from .zarr_store import group_for_sensor, zarr_path_for_glacier

    zarr_path = zarr_path_for_glacier(glacier_dir)
    group = group_for_sensor(record.sensor)
    window = window_polygon_from_registry(glacier).total_bounds
    date_acquired = record.properties.get("DATE_ACQUIRED")
    if not date_acquired:
        time_start_ms = record.properties.get("system:time_start")
        date_acquired = (
            _dt.datetime.utcfromtimestamp(time_start_ms / 1000.0).strftime("%Y-%m-%d")
            if time_start_ms is not None
            else f"{record.year}-01-01"
        )
    fetch_and_store_scene(
        ee_module,
        zarr_path,
        group,
        spec.name,
        record.scene_id,
        record.year,
        date_acquired,
        tuple(float(v) for v in window),
        spec,
    )


def _run_glacier_worker(
    result_queue,
    project: str,
    index: int,
    glacier: dict,
    output_root: Path,
    sensors: Sequence[str],
    until_year: Optional[int],
    max_cloud_fraction: float,
    min_aoi_coverage: float,
    max_scenes_per_year: int,
    use_fetch: bool,
    cache_path: Optional[Path],
    verbose: bool,
) -> None:
    """Top-level (picklable) worker body: downloads one glacier's scenes in its own OS process.

    Does its own ``ee`` import and ``ee.Initialize`` -- no shared,
    process-wide ``ee`` state the way a thread pool would have (see
    :func:`run_glaciers_multi_project` for why that mattered live). Opens
    its own :class:`SceneCache` connection onto the same on-disk database
    (SQLite WAL mode is safe for concurrent writers across processes).
    """
    import ee as ee_module

    glims_id = str(glacier["glims_id"])
    if verbose:
        print(f"  [{index + 1}] {glims_id} [project={project}]", flush=True)

    try:
        ee_module.Initialize(project=project)
        cache = SceneCache(cache_path) if cache_path is not None else None
        fetch = default_fetch if use_fetch else None
        try:
            glacier_total = DownloadStats()
            for sensor_key in sensors:
                spec = get_sensor(sensor_key)
                years = spec.years(until=until_year)
                stats = download_glacier_sensor(
                    ee_module,
                    spec,
                    glacier,
                    years,
                    output_root,
                    cache=cache,
                    max_cloud_fraction=max_cloud_fraction,
                    min_aoi_coverage=min_aoi_coverage,
                    max_scenes_per_year=max_scenes_per_year,
                    fetch=(
                        (lambda record, spec, glacier, glacier_dir: fetch(ee_module, record, spec, glacier, glacier_dir))
                        if fetch is not None
                        else None
                    ),
                    verbose=verbose,
                )
                stats.glaciers = 0
                glacier_total.merge(stats)
            glacier_total.glaciers = 1
        finally:
            if cache is not None:
                cache.close()
        result_queue.put(("ok", glacier_total))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("error", str(exc)))


#: Per-glacier timeout (seconds). Mirrors features.gee_runner.WORK_UNIT_TIMEOUT_S
#: -- a stalled GEE/network call was confirmed live to hang a worker
#: indefinitely with no exception and near-zero CPU use.
GLACIER_WORK_TIMEOUT_S = 1800


def run_glaciers_multi_project(
    glaciers: Iterable[dict],
    output_root: Path,
    projects: Sequence[str],
    cache: Optional[SceneCache] = None,
    sensors: Sequence[str] = SENSOR_ORDER,
    until_year: Optional[int] = None,
    max_cloud_fraction: float = 1.0,
    min_aoi_coverage: float = 0.0,
    max_scenes_per_year: int = 9999,
    fetch: Optional[Callable] = None,
    max_workers: int = 1,
    verbose: bool = True,
) -> DownloadStats:
    """Download every glacier's scenes with one OS process per glacier,
    round-robined across ``projects``.

    Unlike :func:`run_split`, the unit of parallelism here is one glacier,
    not one whole split: up to ``min(max_workers, len(projects),
    len(glaciers))`` ``multiprocessing.Process`` workers run concurrently,
    each downloading every sensor for one glacier -- the same per-glacier
    sensor loop :func:`run_split` already uses, just moved into a worker
    process. This never assigns two workers to the same glacier, so there
    is no concurrent write to one glacier's zarr store; the on-disk
    :class:`SceneCache` database (SQLite WAL mode) is safe for concurrent
    writers across real OS processes.

    Real processes, not threads: a live multi-year run of the equivalent
    function in ``features.gee_runner`` hung indefinitely under a
    ``ThreadPoolExecutor`` (one thread's stalled network call blocked every
    thread sharing that process, via the GIL and shared ``ee``/HTTP client
    state) -- see that module's docstring for the full account. This
    function mirrors the same fix. ``fetch`` must be ``None`` (discovery
    only) or exactly :data:`default_fetch` -- a plain top-level function
    reference, picklable across the process boundary; an arbitrary closure
    (as a thread-based caller could pass) cannot be shipped to a separate
    process.
    """
    if not projects:
        raise ValueError("run_glaciers_multi_project requires at least one GEE project id.")
    if fetch is not None and fetch is not default_fetch:
        raise ValueError(
            "run_glaciers_multi_project's fetch must be None or scenes.download.default_fetch "
            "-- an arbitrary closure cannot be pickled across the process boundary."
        )

    import multiprocessing as mp

    glacier_list = list(glaciers)
    max_concurrent = min(
        max(1, int(max_workers)), max(1, len(projects)), max(1, len(glacier_list))
    )
    ctx = mp.get_context("spawn")
    cache_path = cache.path if cache is not None else None

    total = DownloadStats()
    pending = list(enumerate(glacier_list))
    running: dict[int, tuple] = {}

    def _launch(index: int, glacier: dict) -> None:
        project = projects[index % len(projects)]
        queue = ctx.Queue()
        process = ctx.Process(
            target=_run_glacier_worker,
            args=(
                queue, project, index, glacier, output_root, sensors, until_year,
                max_cloud_fraction, min_aoi_coverage, max_scenes_per_year,
                fetch is not None, cache_path, verbose,
            ),
            daemon=True,
        )
        process.start()
        running[index] = (process, queue, str(glacier["glims_id"]), time.time())

    while pending or running:
        while pending and len(running) < max_concurrent:
            index, glacier = pending.pop(0)
            _launch(index, glacier)

        finished = []
        for index, (process, queue, glims_id, started_at) in list(running.items()):
            if process.is_alive():
                if time.time() - started_at > GLACIER_WORK_TIMEOUT_S:
                    print(
                        f"  [ERROR] {glims_id} timed out after {GLACIER_WORK_TIMEOUT_S}s; "
                        f"terminating worker process"
                    )
                    process.terminate()
                    process.join(timeout=10)
                    finished.append(index)
                continue

            try:
                status, payload = queue.get_nowait()
                if status == "ok":
                    total.merge(payload)
                else:
                    print(f"  [ERROR] {glims_id} failed: {payload}")
            except Exception:  # noqa: BLE001 -- queue empty: process died without posting a result
                print(f"  [ERROR] {glims_id} failed: worker process exited without a result")
            process.join(timeout=10)
            finished.append(index)

        for index in finished:
            del running[index]

        if running and not finished:
            time.sleep(0.5)

    return total
