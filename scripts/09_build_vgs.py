#!/usr/bin/env python3
"""Stage 9 -- build the Visible Glacier Surface (VGS/SGV) reference mask.

Purpose
-------
For each glacier, select the earliest year whose annual state map satisfies the
four simultaneous VGS criteria of the paper (Section 4.7), and build the
reference mask from it.

Criteria:
  1. primary 8-connected component intersecting the RGI polygon;
  2. Snow+Ice coverage within the RGI polygon >= 70%;
  3. primary component area <= 1.5 x the RGI area;
  4. cloud fraction <= 10%.

Inputs
------
- Annual state maps from stage 8, and the rasterized RGI polygons.

Outputs
-------
- One boolean VGS mask per glacier, plus the reference year and per-year
  diagnostics explaining why each candidate year passed or failed.

Usage
-----
    python scripts/09_build_vgs.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.inference.vgs import (  # noqa: E402
    DEFAULT_CLOSING_RADIUS,
    DEFAULT_MAX_AREA_RATIO,
    DEFAULT_MAX_CLOUD_FRACTION,
    DEFAULT_MIN_SNOW_ICE_COVERAGE,
    select_vgs,
)
from glacier_fsnow_unet.inference.zarr_store import (  # noqa: E402
    read_annual_composite,
    read_available_years,
    read_year_data_transform_crs,
    write_vgs_reference,
)
from glacier_fsnow_unet.scenes.zarr_store import zarr_path_for_glacier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the VGS reference mask from the four paper criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--split", default="all", help="Split number, or 'all' (default)")
    parser.add_argument("--only-id", default=None, help="Process a single glacier id")
    parser.add_argument("--limit", type=int, default=0, help="Max glaciers (0 = all)")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        help="Min Snow+Ice coverage of the RGI polygon, as a fraction "
             "(default: config sgv_ref_min_glacier_coverage_pct / 100)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=None,
        help="Max primary-component area as a multiple of the RGI area "
             "(default: 1 + config sgv_ref_max_overshoot_pct / 100)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=None,
        help="Max cloud fraction (default: config sgv_ref_max_cloud_pct / 100)",
    )
    parser.add_argument("--closing-radius", type=int, default=DEFAULT_CLOSING_RADIUS)
    parser.add_argument(
        "--glacier-root",
        default=None,
        help="Root of the per-glacier directory tree (default: the split's root)",
    )
    parser.add_argument(
        "--registry", default=None, help="Path to glacier_registry.parquet"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _rasterize_outline_mask(glacier_dir: Path, ref_shape: tuple[int, int], ref_transform, ref_crs_wkt: str):
    """Rasterize a glacier's outline shapefile onto the annual-composite grid.

    ``scripts/01_select_glaciers.py --write-dirs`` writes
    ``<glacier_dir>/shapefile/<glims_id>.shp`` (WGS84); this reprojects it to
    the composite's own CRS before rasterizing so the mask lines up with
    ``ref_transform`` regardless of which CRS the composite happens to use.
    """
    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    glims_id = glacier_dir.name
    outline_path = glacier_dir / "shapefile" / f"{glims_id}.shp"
    if not outline_path.is_file():
        return None

    gdf = gpd.read_file(outline_path)
    if gdf.empty:
        return None
    dst_crs = CRS.from_user_input(ref_crs_wkt) if ref_crs_wkt else gdf.crs
    gdf = gdf.to_crs(dst_crs)
    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union

    return rasterize(
        [(mapping(geom), 1)],
        out_shape=ref_shape,
        transform=ref_transform,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    ms = cfg.map_sgv_stats
    min_coverage = (
        args.min_coverage
        if args.min_coverage is not None
        else ms.sgv_ref_min_glacier_coverage_pct / 100.0
    )
    max_area_ratio = (
        args.max_area_ratio
        if args.max_area_ratio is not None
        else 1.0 + ms.sgv_ref_max_overshoot_pct / 100.0
    )
    max_cloud = (
        args.max_cloud
        if args.max_cloud is not None
        else ms.sgv_ref_max_cloud_pct / 100.0
    )

    print("=" * 78)
    print("Stage 9: VGS reference mask")
    print("=" * 78)
    print(f"  criterion 2 -- min Snow+Ice coverage: {min_coverage:.0%}"
          f"   (paper: {DEFAULT_MIN_SNOW_ICE_COVERAGE:.0%})")
    print(f"  criterion 3 -- max area ratio:        {max_area_ratio:.2f}x"
          f" (paper: {DEFAULT_MAX_AREA_RATIO:.2f}x)")
    print(f"  criterion 4 -- max cloud fraction:    {max_cloud:.0%}"
          f"   (paper: {DEFAULT_MAX_CLOUD_FRACTION:.0%})")

    if abs(max_cloud - DEFAULT_MAX_CLOUD_FRACTION) > 1e-9:
        print(
            f"  [WARN] Configured cloud threshold ({max_cloud:.0%}) differs from the "
            f"paper's {DEFAULT_MAX_CLOUD_FRACTION:.0%}. See "
            f"docs/decisions/inference_chain.md.",
            file=sys.stderr,
        )

    registry_path = Path(
        args.registry or (Path(cfg.isolated_glacier.output_json).parent / REGISTRY_FILENAME)
    )
    try:
        registry = read_registry(registry_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    sd = cfg.scene_download
    split_indices = (
        list(range(1, len(sd.splits) + 1))
        if str(args.split).strip().lower() == "all"
        else [int(args.split)]
    )

    assignment_path = registry_path.parent / "split_assignment.parquet"
    assignment = None
    if assignment_path.is_file():
        import pandas as pd

        assignment = pd.read_parquet(assignment_path)

    n_glaciers = 0
    n_found = 0
    n_not_found = 0
    n_skipped = 0

    for split_num in split_indices:
        glacier_root = Path(args.glacier_root) if args.glacier_root else Path(
            sd.splits[split_num - 1].root
        )

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
            subset = subset.head(max(0, args.limit - n_glaciers))

        print(f"\n-- Split {split_num}: {len(subset)} glacier(s) -> {glacier_root}")

        for glims_id in subset["glims_id"].astype(str):
            glacier_dir = glacier_root / glims_id
            zarr_path = zarr_path_for_glacier(glacier_dir)
            if not zarr_path.exists():
                print(f"   [skip] {glims_id}: no zarr store at {zarr_path}")
                continue

            years = read_available_years(zarr_path)
            if not years:
                print(f"   [skip] {glims_id}: no annual composites (run stage 8 first)")
                n_skipped += 1
                continue

            transform, crs_wkt = read_year_data_transform_crs(zarr_path)
            first_composite = read_annual_composite(zarr_path, years[0])
            ref_shape = first_composite.shape

            rgi_mask = _rasterize_outline_mask(glacier_dir, ref_shape, transform, crs_wkt)
            if rgi_mask is None:
                print(
                    f"   [skip] {glims_id}: no outline shapefile at "
                    f"{glacier_dir / 'shapefile' / (glims_id + '.shp')} "
                    f"(run scripts/01_select_glaciers.py --write-dirs first)"
                )
                n_skipped += 1
                continue

            annual_states = {year: read_annual_composite(zarr_path, year) for year in years}
            result = select_vgs(
                annual_states,
                rgi_mask,
                min_snow_ice_coverage=min_coverage,
                max_area_ratio=max_area_ratio,
                max_cloud_fraction=max_cloud,
                closing_radius=args.closing_radius,
            )
            n_glaciers += 1

            if result.mask is None:
                n_not_found += 1
                print(f"   [{glims_id}] no year satisfied all 4 VGS criteria "
                      f"({len(result.diagnostics)} candidate year(s) checked)")
                continue

            write_vgs_reference(zarr_path, result.mask, result.year, crs_wkt, transform)
            n_found += 1
            print(f"   [{glims_id}] VGS reference year={result.year} "
                  f"coverage_px={int(result.mask.sum())}")

            if args.limit and n_glaciers >= args.limit:
                break
        if args.limit and n_glaciers >= args.limit:
            break

    print(f"\n[DONE] glaciers={n_glaciers} vgs_found={n_found} "
          f"no_qualifying_year={n_not_found} skipped={n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
