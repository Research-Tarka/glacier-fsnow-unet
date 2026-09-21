#!/usr/bin/env python3
"""Stage 8 -- build the annual worst-state composite per glacier-year.

Purpose
-------
Reduce all retained scenes of one ablation season to a single annual state map,
using the pixel-wise worst-state rule of the paper (Section 3.2): priority
Other > Ice > Snow > Cloud, with a state retained as robust when observed in at
least two scenes.

Inputs
------
- Per-scene class maps from stage 7 (``inference``/``inference_done`` in the
  glacier's zarr store), grouped by glacier and year.

Outputs
-------
- One annual state map per glacier-year, written into ``year_data/map_etat``
  in the same ``<glims_id>.zarr`` store (see
  ``glacier_fsnow_unet.inference.zarr_store``).

Usage
-----
    python scripts/08_annual_state_map.py --config configs/config.yaml
    python scripts/08_annual_state_map.py --split 1 --only-id G214501E60996N
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.inference.classes import (  # noqa: E402
    DEFAULT_MIN_SCENE_AGREEMENT,
    PRIORITY_ORDER,
    CLASS_NAMES,
)
from glacier_fsnow_unet.inference.engine import build_annual_composites_for_glacier  # noqa: E402
from glacier_fsnow_unet.scenes.zarr_store import zarr_path_for_glacier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the annual worst-state composite per glacier-year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--split", default="all", help="Split number, or 'all' (default)")
    parser.add_argument(
        "--registry", default=None, help="Path to glacier_registry.parquet"
    )
    parser.add_argument(
        "--glacier-root",
        default=None,
        help="Root of the per-glacier directory tree (default: the split's root)",
    )
    parser.add_argument("--only-id", default=None, help="Process a single glacier id")
    parser.add_argument("--limit", type=int, default=0, help="Max glaciers (0 = all)")
    parser.add_argument(
        "--min-scene-agreement",
        type=int,
        default=DEFAULT_MIN_SCENE_AGREEMENT,
        help="Scenes that must agree for a robust annual assignment",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    priority = " > ".join(CLASS_NAMES[c] for c in PRIORITY_ORDER)

    print("=" * 78)
    print("Stage 8: annual worst-state composite")
    print("=" * 78)
    print(f"  priority:  {priority}")
    print(f"  agreement: >= {args.min_scene_agreement} scenes")

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

    totals = {"years_written": 0, "years_skipped_existing": 0}
    n_glaciers = 0

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

            counters = build_annual_composites_for_glacier(
                zarr_path,
                min_scene_agreement=args.min_scene_agreement,
                overwrite=args.overwrite,
            )
            for key in totals:
                totals[key] += counters[key]
            n_glaciers += 1
            print(f"   [{glims_id}] years_written={counters['years_written']} "
                  f"already_had={counters['years_skipped_existing']}")

            if args.limit and n_glaciers >= args.limit:
                break
        if args.limit and n_glaciers >= args.limit:
            break

    print(f"\n[DONE] glaciers={n_glaciers} years_written={totals['years_written']} "
          f"already_had={totals['years_skipped_existing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
