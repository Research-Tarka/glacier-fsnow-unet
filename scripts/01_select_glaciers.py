#!/usr/bin/env python3
"""Stage 1 -- select isolated glaciers from RGI 7.0 and write the registry.

Purpose
-------
Filter the RGI 7.0 regions 01 (Alaska) and 02 (Western Canada and USA) down to
the glaciers the pipeline operates on: no geometric overlap with any other RGI
polygon, and area >= 0.05 km^2 (paper, Section 3.1).

Inputs
------
- ``config.isolated_glacier.rgi_shapefiles`` -- the RGI 7.0 shapefiles.
- ``config.isolated_glacier.min_area_km2`` -- area threshold (default 0.05).

Outputs
-------
- ``<output_json parent>/glacier_registry.parquet`` -- the glacier registry
  consumed by every later stage (one row per retained glacier).
- ``glacier_names.json`` -- ``{glims_id: glac_name}`` sidecar for named glaciers.
- With ``--write-dirs``: the per-glacier directory tree under
  ``config.isolated_glacier.output_root``.

Usage
-----
    python scripts/01_select_glaciers.py --config configs/config.yaml
    python scripts/01_select_glaciers.py --write-dirs --jobs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import (  # noqa: E402
    NAMES_FILENAME,
    REGISTRY_FILENAME,
    build_registry,
    find_overlapping_indices,
    load_rgi,
    project_metric,
    write_glacier_directories,
    write_names_json,
    write_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select isolated RGI 7.0 glaciers and write the glacier registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the registry (default: parent of config output_json)",
    )
    parser.add_argument(
        "--min-area-km2",
        type=float,
        default=None,
        help="Override the minimum glacier area in km^2",
    )
    parser.add_argument(
        "--crs-metric",
        default="auto",
        help="Metric CRS for area/overlap computation ('auto' = UTM of median centroid)",
    )
    parser.add_argument(
        "--area-source",
        choices=["rgi", "geodesic", "planar"],
        default="rgi",
        help=(
            "Which area to apply the threshold to: 'rgi' = the inventory's own "
            "ellipsoidal area_km2 column (recommended), 'geodesic' = recompute on "
            "the WGS84 ellipsoid, 'planar' = area in the metric CRS (kept for "
            "comparison only; inflates areas near the domain edges of a single "
            "UTM zone)"
        ),
    )
    parser.add_argument(
        "--no-overlap-check",
        action="store_true",
        help="Skip the geometric-overlap filter (keeps every glacier above the area threshold)",
    )
    parser.add_argument(
        "--no-write-dirs",
        dest="write_dirs",
        action="store_false",
        help="Skip writing the per-glacier shapefile/metadata directory tree "
             "(later stages such as 02 and 09 require it, so this is on by default)",
    )
    parser.set_defaults(write_dirs=True)
    parser.add_argument(
        "--jobs", type=int, default=1, help="Parallel workers for directory writing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    ig = cfg.isolated_glacier
    min_area = args.min_area_km2 if args.min_area_km2 is not None else ig.min_area_km2
    out_dir = Path(args.output_dir) if args.output_dir else Path(ig.output_json).parent

    print("=" * 78)
    print("Stage 1: glacier selection (RGI 7.0, isolated + area filter)")
    print("=" * 78)

    print("[1/4] Loading RGI shapefiles ...")
    gdf = load_rgi(ig.rgi_shapefiles)
    print(f"      loaded {len(gdf)} glacier polygons")

    print("[2/4] Reprojecting to a metric CRS ...")
    gdf_proj, metric_crs = project_metric(gdf, args.crs_metric, area_source=args.area_source)
    print(f"      metric CRS: {metric_crs} (area_source={args.area_source})")

    print("[3/4] Detecting geometric overlaps ...")
    if args.no_overlap_check:
        drop = []
        print("      overlap check disabled (--no-overlap-check)")
    else:
        drop = find_overlapping_indices(gdf_proj)
        print(f"      {len(drop)} glaciers dropped for intersecting another RGI polygon")

    print(f"[4/4] Applying area filter (>= {min_area} km^2) and writing registry ...")
    registry = build_registry(gdf, gdf_proj, drop, min_area)
    registry_path = write_registry(registry, out_dir / REGISTRY_FILENAME)
    names_path = write_names_json(registry, out_dir / NAMES_FILENAME)

    print(f"      retained {len(registry)} glaciers "
          f"({100.0 * len(registry) / max(len(gdf), 1):.1f}% of the inventory)")
    print(f"      total area: {registry['area_km2'].sum():,.1f} km^2")
    print(f"      registry -> {registry_path}")
    print(f"      names    -> {names_path}")

    if not registry.empty:
        print("      largest glaciers:")
        for row in registry.head(5).itertuples():
            label = row.glac_name or row.glims_id
            print(f"        {label}: {row.area_km2:,.2f} km^2")

    if args.write_dirs:
        print(f"[+]   Writing per-glacier directories under {ig.output_root} ...")
        n_ok, n_err = write_glacier_directories(
            registry, gdf, ig.output_root, metric_crs,
            min_area_km2=min_area, n_jobs=args.jobs,
        )
        print(f"      wrote {n_ok} glacier directories, {n_err} errors")
        if n_err:
            return 1

    print("[DONE] Stage 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
