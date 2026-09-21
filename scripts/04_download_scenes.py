#!/usr/bin/env python3
"""Stage 4 -- download satellite scenes from Google Earth Engine.

Purpose
-------
For each glacier of a split, download the Landsat 5/7/8/9 and Sentinel-2 scenes
of the 1 July - 30 September ablation window (paper, Section 3.2).

One script taking ``--split N`` (or ``--split all``) covers every parallel
output split, rather than one entry point per split index. ``--split all
--parallel`` spawns one OS subprocess per configured split, each
authenticating against that split's own ``scene_download.splits[i].ee_project``
-- real load spreading across separate GEE project quotas, not just
sequential iteration in one process.

Inputs
------
- The glacier registry (stage 1) and split assignment (stage 3).
- Earth Engine credentials plus a project id, from ``--ee-project`` (overrides
  every split), each split's own ``ee_project`` in ``scene_download.splits``,
  or ``config.scene_download.ee_project_default`` as the last resort.

Outputs
-------
- Per-glacier scene data under the split's data root.
- ``scene_cache_split<N>.db`` -- the SQLite record of every downloaded scene,
  which is also the resume mechanism (see docs/decisions/scene_download_resume.md).

Credentials
-----------
This stage cannot run without Earth Engine credentials. It fails immediately,
with a message naming the fix, rather than hanging or erroring deep in a loop.

Usage
-----
    python scripts/04_download_scenes.py --split 7
    python scripts/04_download_scenes.py --split all --sensors L8,L9,S2
    python scripts/04_download_scenes.py --split all --parallel   # one process per project
    python scripts/04_download_scenes.py --split 1 --dry-run   # no credentials needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.scenes.cache import SceneCache  # noqa: E402
from glacier_fsnow_unet.scenes.download import default_fetch, run_split  # noqa: E402
from glacier_fsnow_unet.scenes.gee_auth import GeeAuthError, initialize_ee  # noqa: E402
from glacier_fsnow_unet.scenes.sensors import SENSOR_ORDER  # noqa: E402


def make_fetch(ee_module):
    """Build the ``fetch`` callback for the sequential (``run_split``) code path.

    Only valid for the single-process path: ``run_glaciers_multi_project``
    (multiple OS processes) must use ``scenes.download.default_fetch``
    directly instead -- a closure capturing ``ee_module`` cannot be pickled
    across a process boundary. Kept as a thin adapter here (rather than
    passing ``ee`` through every call) since ``run_split`` genuinely shares
    one ``ee`` instance across its whole sequential loop.
    """
    def _fetch(record, spec, glacier: dict, glacier_dir) -> None:
        default_fetch(ee_module, record, spec, glacier, glacier_dir)

    return _fetch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Landsat/Sentinel-2 scenes from Google Earth Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--split",
        default="all",
        help="Split number (1-based), or 'all' to iterate over every split (default)",
    )
    parser.add_argument(
        "--registry", default=None, help="Path to glacier_registry.parquet"
    )
    parser.add_argument(
        "--assignment", default=None, help="Path to split_assignment.parquet"
    )
    parser.add_argument("--ee-project", default=None, help="Earth Engine project id")
    parser.add_argument(
        "--service-account-key",
        default=None,
        help="Path to a service-account JSON key, for unattended runs",
    )
    parser.add_argument(
        "--sensors",
        default=",".join(SENSOR_ORDER),
        help="Comma-separated sensors to download",
    )
    parser.add_argument("--only-id", default=None, help="Process a single glacier id")
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N glaciers (0 = all)"
    )
    parser.add_argument(
        "--until-year",
        type=int,
        default=None,
        help="Latest acquisition year to consider (default: current year)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory holding scene_cache_split<N>.db (default: the split root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without contacting Earth Engine",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce per-glacier logging")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "With --split all, run every split as its own OS subprocess "
            "(one per GEE project) instead of sequentially in this process."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Download N glaciers concurrently (one thread per glacier), "
            "round-robined across every configured scene_download.splits "
            "project -- never two threads on the same glacier. Alternative "
            "to --parallel: one process, thread-per-glacier, instead of "
            "one subprocess per split. Default: config.scene_download."
            "glacier_workers."
        ),
    )
    return parser


def _strip_split_args(argv: list[str]) -> list[str]:
    """Remove --split/--split=X and --parallel/-p from argv, keeping everything else."""
    out: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--split":
            skip_next = True
            continue
        if token.startswith("--split="):
            continue
        if token == "--parallel":
            continue
        out.append(token)
    return out


def _run_splits_parallel(argv: list[str], splits: list[int]) -> int:
    """Spawn one OS subprocess per split, each running this same script.

    One process per project: each subprocess authenticates against its own
    split's GEE project (see main()'s per-split ee.Initialize), so N splits
    genuinely spread load across N separate project quotas instead of
    contending for one. Tested at 19 splits, ~1200 glaciers each.
    """
    import subprocess

    base_args = _strip_split_args(argv)
    script = str(Path(__file__).resolve())
    processes: dict[int, subprocess.Popen] = {}
    for split_num in splits:
        cmd = [sys.executable, script, "--split", str(split_num), *base_args]
        print(f"[PARALLEL] launching split {split_num}: {' '.join(cmd[1:])}")
        processes[split_num] = subprocess.Popen(cmd)

    exit_code = 0
    for split_num, proc in processes.items():
        code = proc.wait()
        if code != 0:
            print(f"[PARALLEL] split {split_num} exited with code {code}", file=sys.stderr)
            exit_code = exit_code or code

    return exit_code


def _run_multi_worker(args, cfg, sd, splits, registry, assignment, sensors, max_workers) -> int:
    """Thread-per-glacier download, round-robined across the selected splits'
    GEE projects -- see ``scenes.download.run_glaciers_multi_project``.

    All selected splits' glaciers are pooled into one list and distributed by
    a single ThreadPoolExecutor, rather than processed split-by-split, so the
    concurrency unit is the glacier, not the split. Every selected split must
    share one output root and one cache file, since scene output is keyed by
    glacier id, not by split, once the download is no longer partitioned by
    split directory.
    """
    from glacier_fsnow_unet.scenes.download import run_glaciers_multi_project

    projects = [sd.splits[i - 1].ee_project for i in splits]
    roots = {sd.splits[i - 1].root for i in splits}
    if len(roots) > 1:
        print(
            "[ERROR] --max-workers pools glaciers across the selected splits into one "
            "run, which requires them to share one output root; the selected splits "
            f"have {len(roots)} different roots: {sorted(roots)}. Select splits that "
            "share a root, or run them with --parallel instead.",
            file=sys.stderr,
        )
        return 2
    output_root = Path(next(iter(roots)))

    if assignment is not None:
        ids = set(
            assignment.loc[
                assignment["split_index"].isin([i - 1 for i in splits]), "glims_id"
            ].astype(str)
        )
        subset = registry[registry["glims_id"].astype(str).isin(ids)]
    else:
        subset = registry

    if args.only_id:
        subset = subset[subset["glims_id"].astype(str) == args.only_id]
    if args.limit:
        subset = subset.head(args.limit)

    print(f"  {len(subset)} glacier(s) -> {output_root}, "
          f"{len(projects)} project(s), max_workers={max_workers}")
    if subset.empty:
        print("\n[DONE] Stage 4 complete.")
        return 0

    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"scene_cache_split{splits[0]}.db"

    # Each worker process authenticates on its own (see
    # scenes.download.run_glaciers_multi_project) -- no shared ee.Initialize
    # needed here, unlike the sequential run_split path below.
    with SceneCache(cache_path) as cache:
        print(f"  cache: {cache_path} ({cache.count()} entries)")
        try:
            stats = run_glaciers_multi_project(
                subset.to_dict("records"),
                output_root,
                projects=projects,
                cache=cache,
                sensors=sensors,
                until_year=args.until_year,
                max_cloud_fraction=sd.max_cloud_fraction,
                min_aoi_coverage=sd.min_aoi_coverage,
                max_scenes_per_year=sd.max_scenes_per_year,
                fetch=default_fetch,
                max_workers=max_workers,
                verbose=not args.quiet,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Multi-worker download failed: {exc}", file=sys.stderr)
            return 1

    print(f"  {stats.summary()}")
    print("\n[DONE] Stage 4 complete.")
    return 1 if stats.failed else 0


def resolve_splits(arg: str, n_configured: int) -> list[int]:
    """Turn --split into a list of 1-based split numbers."""
    if str(arg).strip().lower() == "all":
        return list(range(1, n_configured + 1))
    try:
        value = int(arg)
    except ValueError:
        raise SystemExit(f"[ERROR] --split must be an integer or 'all', got '{arg}'")
    if not 1 <= value <= n_configured:
        raise SystemExit(
            f"[ERROR] --split {value} is out of range; "
            f"{n_configured} split(s) are configured in scene_download.splits."
        )
    return [value]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    sd = cfg.scene_download
    splits = resolve_splits(args.split, len(sd.splits))

    if args.parallel:
        if len(splits) < 2:
            print(
                "[ERROR] --parallel requires --split all with 2+ splits configured "
                "in scene_download.splits.",
                file=sys.stderr,
            )
            return 2
        return _run_splits_parallel(argv or sys.argv[1:], splits)

    sensors = [s.strip().upper() for s in args.sensors.split(",") if s.strip()]
    unknown = [s for s in sensors if s not in SENSOR_ORDER]
    if unknown:
        print(
            f"[ERROR] Unknown sensor(s) {unknown}; valid: {', '.join(SENSOR_ORDER)}",
            file=sys.stderr,
        )
        return 2

    registry_path = Path(
        args.registry or (Path(cfg.isolated_glacier.output_json).parent / REGISTRY_FILENAME)
    )
    try:
        registry = read_registry(registry_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    assignment_path = Path(
        args.assignment or (registry_path.parent / "split_assignment.parquet")
    )
    assignment = None
    if assignment_path.is_file():
        import pandas as pd

        assignment = pd.read_parquet(assignment_path)
    elif len(sd.splits) > 1:
        print(
            f"[ERROR] Split assignment not found at {assignment_path}.\n"
            f"        Run scripts/03_split_glaciers.py first, or configure a "
            f"single split.",
            file=sys.stderr,
        )
        return 2

    print("=" * 78)
    print(f"Stage 4: scene download (splits {splits}, sensors {', '.join(sensors)})")
    print("=" * 78)

    if args.dry_run:
        print("  [DRY-RUN] Earth Engine will not be contacted")

    max_workers = args.max_workers if args.max_workers is not None else sd.glacier_workers
    if max_workers > 1 and not args.dry_run:
        return _run_multi_worker(
            args, cfg, sd, splits, registry, assignment, sensors, max_workers
        )

    exit_code = 0
    for split_num in splits:
        split_cfg = sd.splits[split_num - 1]
        split_root = Path(split_cfg.root)

        # Each split authenticates against its own configured GEE project
        # (--ee-project overrides every split's project if given; otherwise
        # this split's own project id, not a single shared default, is used)
        # so multiple splits genuinely spread load across separate GEE
        # project quotas rather than all sharing one project's budget.
        ee = None
        if not args.dry_run:
            try:
                ee = initialize_ee(
                    project=args.ee_project or split_cfg.ee_project,
                    config_default=sd.ee_project_default,
                    service_account_key=args.service_account_key,
                )
            except GeeAuthError as exc:
                print(
                    f"\n[ERROR] Google Earth Engine is not available for split "
                    f"{split_num} (project '{split_cfg.ee_project}').\n\n{exc}\n",
                    file=sys.stderr,
                )
                exit_code = 3
                continue
            print(f"  Split {split_num}: Earth Engine authenticated (project={split_cfg.ee_project})")

        if assignment is not None:
            ids = set(
                assignment.loc[
                    assignment["split_index"] == (split_num - 1), "glims_id"
                ].astype(str)
            )
            subset = registry[registry["glims_id"].astype(str).isin(ids)]
        else:
            subset = registry

        if args.only_id:
            subset = subset[subset["glims_id"].astype(str) == args.only_id]
        if args.limit:
            subset = subset.head(args.limit)

        print(f"\n-- Split {split_num}: {len(subset)} glaciers -> {split_root}")
        if subset.empty:
            continue

        cache_dir = Path(args.cache_dir) if args.cache_dir else split_root
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"scene_cache_split{split_num}.db"

        with SceneCache(cache_path) as cache:
            print(f"   cache: {cache_path} ({cache.count()} entries)")
            if args.dry_run:
                print("   [DRY-RUN] skipping downloads")
                continue
            try:
                stats = run_split(
                    ee,
                    subset.to_dict("records"),
                    split_root,
                    cache=cache,
                    sensors=sensors,
                    until_year=args.until_year,
                    max_cloud_fraction=sd.max_cloud_fraction,
                    min_aoi_coverage=sd.min_aoi_coverage,
                    max_scenes_per_year=sd.max_scenes_per_year,
                    fetch=make_fetch(ee),
                    verbose=not args.quiet,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"   [ERROR] Split {split_num} failed: {exc}", file=sys.stderr)
                exit_code = 1
                continue
            print(f"   [SPLIT {split_num}] {stats.summary()}")
            if stats.failed:
                exit_code = 1

    print("\n[DONE] Stage 4 complete.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
