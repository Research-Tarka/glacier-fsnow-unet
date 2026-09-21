#!/usr/bin/env python3
"""Stage 10 -- compute the delivered per-glacier statistics tables.

Purpose
-------
Produce the three delivered parquets of the pipeline:

  * ``glacier_ref.parquet``      -- one row per glacier: the VGS reference
                                    (reference year, area, RGI ratio).
  * ``stats_etat.parquet``       -- one row per glacier-year: class counts
                                    inside the VGS, F_snow, validity.
  * ``stats_normalized.parquet`` -- the same, normalized by VGS area.

Inputs
------
- Annual state maps (stage 8), the VGS masks (stage 9), and the glacier
  registry (stage 1).

Outputs
-------
- The three parquets under ``config.paths.output_root``.

Usage
-----
    python scripts/10_compute_stats.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.inference.fsnow import DEFAULT_MAX_UNRESOLVED_FRACTION  # noqa: E402
from glacier_fsnow_unet.inference.stats import (  # noqa: E402
    GLACIER_REF_PARQUET,
    STATS_ETAT_PARQUET,
    STATS_NORMALIZED_PARQUET,
    build_glacier_ref_row,
    build_stats_etat,
    normalize_stats,
    write_stats,
)
from glacier_fsnow_unet.inference.zarr_store import (  # noqa: E402
    read_annual_composite,
    read_available_years,
    read_year_data_attrs,
    read_vgs_reference,
)
from glacier_fsnow_unet.scenes.zarr_store import zarr_path_for_glacier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute glacier_ref / stats_etat / stats_normalized parquets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--split", default="all", help="Split number, or 'all' (default)")
    parser.add_argument(
        "--glacier-root",
        default=None,
        help="Root of the per-glacier directory tree (default: the split's root)",
    )
    parser.add_argument("--registry", default=None, help="Path to glacier_registry.parquet")
    parser.add_argument("--only-id", default=None, help="Process a single glacier id")
    parser.add_argument("--output-dir", default=None, help="Where to write the parquets")
    parser.add_argument("--limit", type=int, default=0, help="Max glaciers (0 = all)")
    parser.add_argument(
        "--max-unresolved",
        type=float,
        default=DEFAULT_MAX_UNRESOLVED_FRACTION,
        help="Reject a glacier-year at or above this unresolved VGS fraction",
    )
    parser.add_argument("--pixel-size-m", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir or cfg.paths.output_root)

    print("=" * 78)
    print("Stage 10: statistics tables")
    print("=" * 78)
    print(f"  output dir:     {out_dir}")
    print(f"  max unresolved: {args.max_unresolved:.0%} of VGS pixels")
    print(f"  pixel area:     {args.pixel_size_m ** 2:.0f} m^2")
    print(f"  will write:     {GLACIER_REF_PARQUET}, {STATS_ETAT_PARQUET}, "
          f"{STATS_NORMALIZED_PARQUET}")

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

    pixel_area_m2 = args.pixel_size_m ** 2
    glacier_ref_rows: list[dict] = []
    stats_etat_frames: list = []
    n_glaciers = 0
    n_no_vgs = 0

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

        for row in subset.itertuples():
            glims_id = str(row.glims_id)
            glacier_dir = glacier_root / glims_id
            zarr_path = zarr_path_for_glacier(glacier_dir)
            if not zarr_path.exists():
                print(f"   [skip] {glims_id}: no zarr store at {zarr_path}")
                continue

            years = read_available_years(zarr_path)
            if not years:
                print(f"   [skip] {glims_id}: no annual composites (run stage 8 first)")
                continue

            vgs_mask = read_vgs_reference(zarr_path)
            reference_year = read_year_data_attrs(zarr_path).get("vgs_ref_year")
            n_glaciers += 1

            glacier_ref_rows.append(
                build_glacier_ref_row(
                    glims_id,
                    vgs_mask,
                    reference_year,
                    rgi_area_km2=float(row.area_km2),
                    pixel_area_m2=pixel_area_m2,
                )
            )

            if vgs_mask is None:
                n_no_vgs += 1
                print(f"   [{glims_id}] no VGS reference (run stage 9 first) -- "
                      f"glacier_ref row written, no stats_etat rows")
                continue

            annual_states = {year: read_annual_composite(zarr_path, year) for year in years}
            frame = build_stats_etat(
                glims_id,
                annual_states,
                vgs_mask,
                pixel_area_m2=pixel_area_m2,
                max_unresolved_fraction=args.max_unresolved,
            )
            stats_etat_frames.append(frame)
            print(f"   [{glims_id}] {len(frame)} glacier-year row(s), "
                  f"{int(frame['valid'].sum())} valid")

            if args.limit and n_glaciers >= args.limit:
                break
        if args.limit and n_glaciers >= args.limit:
            break

    import pandas as pd

    glacier_ref = pd.DataFrame(glacier_ref_rows)
    stats_etat = (
        pd.concat(stats_etat_frames, ignore_index=True) if stats_etat_frames else pd.DataFrame()
    )
    stats_normalized = normalize_stats(stats_etat)

    paths = write_stats(out_dir, glacier_ref, stats_etat, stats_normalized)

    print(f"\n[DONE] glaciers={n_glaciers} no_vgs={n_no_vgs} "
          f"glacier_ref_rows={len(glacier_ref)} stats_etat_rows={len(stats_etat)}")
    for name, path in paths.items():
        print(f"  {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
