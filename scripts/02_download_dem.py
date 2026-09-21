#!/usr/bin/env python3
"""Stage 2 -- download and prepare a DEM for every glacier.

Purpose
-------
For each glacier, fetch the best available elevation model over its buffered
analysis window and write it at [10, 15, 30] m in EPSG:3413.

Sources, in priority order:
1. ArcticDEM v4.1 10 m mosaics, via the PGC STAC API + windowed COG reads
   (see docs/decisions/dem_backend.md for why this is direct STAC/COG access
   rather than a bundled tile index).
2. Copernicus DEM GLO-30, via the public AWS COG bucket, where ArcticDEM has
   no coverage.

Inputs
------
- The glacier registry from stage 1, or an existing per-glacier directory tree
  under ``config.dem.glacier_dem_root``.

Outputs
-------
- ``<glacier_dir>/<glims_id>.zarr`` -- ``dem`` group with ``elev_10m``,
  ``elev_15m``, ``elev_30m``, ``elev_30m_interp`` plus provenance/coverage
  attrs (see ``glacier_fsnow_unet.dem.zarr_store``).
- ``<glacier_dir>/extract_dem.done`` (resume flag).

Usage
-----
    python scripts/02_download_dem.py --config configs/config.yaml --jobs 8
    python scripts/02_download_dem.py --only-id G214501E60996N --overwrite
    python scripts/02_download_dem.py --limit 10 --from-registry
    python scripts/02_download_dem.py --only-id G214501E60996N --force-copernicus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.dem.engine import (  # noqa: E402
    run_dem_extraction,
    window_polygon_from_registry,
)
from glacier_fsnow_unet.dem.processing import OUTPUT_RESOLUTIONS_M  # noqa: E402
from glacier_fsnow_unet.dem.sources import (  # noqa: E402
    ARCTICDEM_COLLECTION,
    ARCTICDEM_VERSION,
    PGC_STAC_URL,
)
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch ArcticDEM (or Copernicus GLO-30) per glacier and write DEM products.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to glacier_registry.parquet (default: next to config output_json)",
    )
    parser.add_argument(
        "--glacier-root",
        default=None,
        help="Root of the per-glacier directory tree (default: config.dem.glacier_dem_root)",
    )
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="Take the analysis window from the registry instead of shapefile_UNet/*.shp",
    )
    parser.add_argument("--only-id", default=None, help="Process a single glacier id")
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N glaciers (0 = all)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers (default: config.dem.max_workers)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-extract glaciers already marked done"
    )
    parser.add_argument(
        "--no-copernicus-fallback",
        action="store_true",
        help="Fail instead of falling back to Copernicus GLO-30 outside ArcticDEM coverage",
    )
    parser.add_argument(
        "--force-copernicus",
        action="store_true",
        help="Skip ArcticDEM entirely and use Copernicus GLO-30, even where ArcticDEM "
             "would succeed (exercises the fallback path deliberately)",
    )
    parser.add_argument("--stac-url", default=None, help="PGC STAC API root URL")
    parser.add_argument(
        "--collection", default=ARCTICDEM_COLLECTION, help="ArcticDEM STAC collection id"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    stac_url = args.stac_url or cfg.dem.arctic_dem_stac_url or PGC_STAC_URL
    if "s3" in stac_url and stac_url.endswith("catalog.json"):
        print(
            f"[WARN] Config points at a static S3 catalog ({stac_url}) which PGC no "
            f"longer publishes; using the STAC API at {PGC_STAC_URL} instead.",
            file=sys.stderr,
        )
        stac_url = PGC_STAC_URL

    jobs = args.jobs if args.jobs is not None else cfg.dem.max_workers
    glacier_root = Path(args.glacier_root or cfg.dem.glacier_dem_root)
    registry_path = Path(
        args.registry or (Path(cfg.isolated_glacier.output_json).parent / REGISTRY_FILENAME)
    )

    print("=" * 78)
    print("Stage 2: DEM extraction")
    print("=" * 78)
    print(f"  ArcticDEM {ARCTICDEM_VERSION} ({args.collection}) via {stac_url}")
    print(
        f"  fallback: {'Copernicus GLO-30' if not args.no_copernicus_fallback else 'disabled'}"
        f" | outputs: {list(OUTPUT_RESOLUTIONS_M)} m"
    )

    polygons: dict = {}
    if args.from_registry:
        try:
            registry = read_registry(registry_path)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        if args.only_id:
            registry = registry[registry["glims_id"] == args.only_id]
        if args.limit:
            registry = registry.head(args.limit)
        if registry.empty:
            print("[ERROR] No glaciers selected from the registry.", file=sys.stderr)
            return 2
        glacier_dirs = [glacier_root / gid for gid in registry["glims_id"]]
        polygons = {
            str(row["glims_id"]): window_polygon_from_registry(row).iloc[0]
            for _, row in registry.iterrows()
        }
    else:
        if not glacier_root.is_dir():
            print(
                f"[ERROR] Glacier root not found: {glacier_root}\n"
                f"        Run scripts/01_select_glaciers.py --write-dirs first, "
                f"or pass --from-registry.",
                file=sys.stderr,
            )
            return 2
        if args.only_id:
            glacier_dirs = [glacier_root / args.only_id]
        else:
            glacier_dirs = sorted(p for p in glacier_root.iterdir() if p.is_dir())
        if args.limit:
            glacier_dirs = glacier_dirs[: args.limit]

    print(f"  glaciers: {len(glacier_dirs)} | workers: {jobs}")

    if args.force_copernicus:
        print("  [INFO] --force-copernicus: skipping ArcticDEM for every glacier in this run.")

    results = run_dem_extraction(
        glacier_dirs,
        polygons=polygons,
        n_jobs=jobs,
        overwrite=args.overwrite,
        copernicus_fallback=not args.no_copernicus_fallback,
        stac_url=stac_url,
        collection=args.collection,
        force_copernicus=args.force_copernicus,
    )

    ok = [r for r in results if r[1] == "ok"]
    skipped = [r for r in results if r[1].startswith("skip")]
    errors = [r for r in results if r[1].startswith("error")]

    print(f"\n[DONE] ok={len(ok)} | skipped={len(skipped)} | errors={len(errors)}")
    for label, rows in (("Skipped", skipped), ("Errors", errors)):
        if rows:
            print(f"  {label} (first 10):")
            for gid, status in rows[:10]:
                print(f"    - {gid}: {status}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
