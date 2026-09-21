#!/usr/bin/env python3
"""Stage 11 -- download per-glacier climate/environmental features.

Purpose
-------
For each active source, aggregate its variables over every glacier footprint,
per year, and write one absolute-value Parquet per source.

Only five sources are supported (ROADMAP decision #4), each contributing a
distinct physical signal: ERA5_Land, ERA5_Reanalysis_CDS, TerraClimate,
Daymet_V4, MERRA2_Aerosols.

Inputs
------
- The glacier registry (stage 1).
- Earth Engine credentials and a project id.

Outputs
-------
- ``<features_root>/cache/<slug>/<slug>__<year>.parquet`` -- per-year cache.
- ``<features_root>/raw_sources/<slug>.parquet`` -- consolidated absolutes.

Usage
-----
    python scripts/11_download_features.py --config configs/config.yaml
    python scripts/11_download_features.py --sources Daymet_V4 --years 2000-2005
    python scripts/11_download_features.py --list-sources
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.features.gee_runner import (  # noqa: E402
    run_source,
    run_sources_multi_project,
)
from glacier_fsnow_unet.features.source_defs import SOURCES, get_source  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.scenes.gee_auth import GeeAuthError, initialize_ee  # noqa: E402


def parse_years(text: str | None) -> list[int] | None:
    """Parse '2000-2005' or '2000,2003,2010' into a list of years."""
    if not text:
        return None
    if "-" in text and "," not in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download per-glacier climate/environmental features from GEE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--registry", default=None, help="Path to glacier_registry.parquet"
    )
    parser.add_argument(
        "--features-root",
        default=None,
        help="Root for feature outputs (default: <data_root>/features)",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated sources (default: config.features.sources)",
    )
    parser.add_argument(
        "--years", default=None, help="Years as 'START-END' or a comma-separated list"
    )
    parser.add_argument("--ee-project", default=None, help="Earth Engine project id")
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N glaciers (0 = all)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore the per-year cache and refetch"
    )
    parser.add_argument(
        "--list-sources", action="store_true", help="List the active sources and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without contacting Earth Engine",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_sources:
        print("Active climate/environmental sources:")
        for name, spec in SOURCES.items():
            print(
                f"  {name:<22} {spec.collection:<34} "
                f"{spec.first_year}-{spec.last_year}  "
                f"{len(spec.value_columns)} variables"
            )
        return 0

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    source_names = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else list(cfg.features.sources)
    )
    try:
        specs = [get_source(name) for name in source_names]
    except KeyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    registry_path = Path(
        args.registry or (Path(cfg.isolated_glacier.output_json).parent / REGISTRY_FILENAME)
    )
    try:
        registry = read_registry(registry_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.limit:
        registry = registry.head(args.limit)

    features_root = Path(
        args.features_root or (Path(cfg.paths.data_root) / "features")
    )
    cache_root = features_root / "cache"
    raw_root = features_root / "raw_sources"
    raw_root.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.years)

    print("=" * 78)
    print("Stage 11: climate/environmental feature download")
    print("=" * 78)
    print(f"  glaciers: {len(registry)} | sources: {', '.join(source_names)}")
    print(f"  output:   {features_root}")

    if args.dry_run:
        for spec in specs:
            target = years or spec.years()
            print(
                f"  [DRY-RUN] {spec.slug}: {len(target)} years "
                f"({min(target)}-{max(target)}), {len(spec.value_columns)} variables"
            )
        return 0

    projects = [split.ee_project for split in cfg.scene_download.splits]
    max_workers = max(1, int(cfg.features.max_workers))

    if max_workers > 1 and not args.ee_project:
        print(
            f"  Earth Engine: {len(projects)} project(s) configured, "
            f"max_workers={max_workers} -- running (source, year) units in parallel"
        )
        exit_code = 0
        try:
            per_source = run_sources_multi_project(
                specs,
                registry,
                cache_root,
                projects=projects,
                years=years,
                force=args.force,
                max_workers=max_workers,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Multi-project run failed: {exc}", file=sys.stderr)
            return 1

        for spec in specs:
            frame = per_source[spec.slug]
            out_path = raw_root / f"{spec.slug}.parquet"
            frame.to_parquet(out_path, index=False)
            print(f"   {spec.slug}: rows={len(frame)} -> {out_path}")

        print("\n[DONE] Stage 11 complete.")
        return exit_code

    try:
        ee = initialize_ee(
            project=args.ee_project,
            config_default=cfg.scene_download.ee_project_default,
        )
    except GeeAuthError as exc:
        print(f"\n[ERROR] Google Earth Engine is not available.\n\n{exc}\n",
              file=sys.stderr)
        return 3
    print("  Earth Engine: authenticated and reachable")

    exit_code = 0
    for spec in specs:
        print(f"\n-- {spec.slug} ({spec.collection})")
        try:
            frame = run_source(
                ee, spec, registry, cache_root, years=years, force=args.force
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   [ERROR] {spec.slug} failed: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        out_path = raw_root / f"{spec.slug}.parquet"
        frame.to_parquet(out_path, index=False)
        print(f"   rows={len(frame)} -> {out_path}")

    print("\n[DONE] Stage 11 complete.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
