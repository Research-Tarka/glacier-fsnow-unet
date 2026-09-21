#!/usr/bin/env python3
"""Stage 3 -- distribute glaciers across N processing batches (splits).

Purpose
-------
Assign every glacier to one of N parallel processing batches, balanced by both
glacier count and total area, using a Longest-Processing-Time (LPT) greedy
assignment. Optionally move the per-glacier directories on disk to match.

The number of batches comes from ``config.scene_download.splits``, so it can be
tuned to however many parallel GEE projects/output roots are actually
available, rather than a fixed batch count.

Inputs
------
- The glacier registry from stage 1 (for areas), and/or the existing on-disk
  split layout under each ``config.scene_download.splits[].root``.

Outputs
-------
- ``split_assignment.parquet`` -- ``glims_id`` -> ``split_index``/``split_name``.
- With ``--apply``: glacier directories moved into their assigned split root.

Note on ``--apply``
--------------------
Stages 4 and 5 no longer require glaciers to physically live under their
assigned split's root: both read ``split_assignment.parquet`` directly and
round-robin GEE project ids as a pure scheduling decision (see
``scenes.download.run_glaciers_multi_project`` /
``features.gee_runner.run_sources_multi_project``), writing into one shared
glacier directory tree. ``--apply`` is therefore optional legacy behaviour
for anyone who still wants split-partitioned directories on disk; the LPT
assignment table remains the source of truth for scheduling either way.

Modes
-----
- default    : assign glaciers not yet in any split; glaciers already placed
               stay where they are (incremental, so re-running after adding
               new glaciers does not reshuffle work already done).
- --rebalance: reassign *every* glacier across all splits from scratch, which
               is what you want after adding empty splits.

Usage
-----
    python scripts/03_split_glaciers.py --config configs/config.yaml --dry-run
    python scripts/03_split_glaciers.py --rebalance --apply
    python scripts/03_split_glaciers.py --n-splits 19 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.glaciers.splitter import (  # noqa: E402
    SplitLoad,
    apply_moves,
    assign_lpt,
    assignment_table,
    balance_metrics,
    balance_report,
    inventory_splits,
    loads_from_disk,
    plan_moves,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distribute glaciers across N batches, balanced by count and area (LPT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--registry", default=None, help="Path to glacier_registry.parquet"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Number of batches (default: len(config.scene_download.splits))",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write split_assignment.parquet (default: next to the registry)",
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Reassign every glacier from scratch instead of only the unassigned ones",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move glacier directories on disk (default: plan only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --apply, print the moves without touching the filesystem",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    split_roots = [Path(s.root) for s in cfg.scene_download.splits]
    n_splits = args.n_splits or len(split_roots)
    if n_splits < 1:
        print("[ERROR] Number of splits must be >= 1.", file=sys.stderr)
        return 2

    # If --n-splits exceeds the configured roots, synthesise sibling names so a
    # planning-only run can explore a different batch count.
    if n_splits > len(split_roots):
        if not split_roots:
            print(
                "[ERROR] --n-splits exceeds the configured roots, but "
                "config.scene_download.splits is empty so no base root is "
                "available to derive sibling names from.",
                file=sys.stderr,
            )
            return 2
        base = split_roots[0].parent
        split_roots = list(split_roots) + [
            base / f"Split{i + 1}" for i in range(len(split_roots), n_splits)
        ]
    split_roots = split_roots[:n_splits]

    registry_path = Path(
        args.registry or (Path(cfg.isolated_glacier.output_json).parent / REGISTRY_FILENAME)
    )
    try:
        registry = read_registry(registry_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    areas = dict(zip(registry["glims_id"].astype(str), registry["area_km2"].astype(float)))

    print("=" * 78)
    print(f"Stage 3: split glaciers across {n_splits} batches"
          f"{' (REBALANCE)' if args.rebalance else ''}")
    print("=" * 78)
    print(f"  registry: {registry_path} ({len(registry)} glaciers)")
    for root in split_roots:
        print(f"    - {root}")

    existing = inventory_splits(split_roots)
    print(f"  already placed on disk: {len(existing)}")

    if args.rebalance:
        # Reassign everything: known glaciers plus anything on disk.
        ids = list(registry["glims_id"].astype(str))
        known = set(ids)
        for gid in existing:
            if gid not in known:
                ids.append(gid)
                areas.setdefault(gid, 0.0)
        loads = assign_lpt(
            ids,
            [areas.get(gid, 0.0) for gid in ids],
            n_splits,
            split_names=[p.name for p in split_roots],
        )
    else:
        # Incremental: seed from disk, then place only the unassigned glaciers.
        loads = loads_from_disk(split_roots, areas=areas)
        for i, load in enumerate(loads):
            load.index = i
        pending = [
            gid for gid in registry["glims_id"].astype(str) if gid not in existing
        ]
        print(f"  to assign: {len(pending)}")
        loads = assign_lpt(
            pending,
            [areas.get(gid, 0.0) for gid in pending],
            n_splits,
            initial_loads=loads,
        )

    for i, load in enumerate(loads):
        load.path = split_roots[i]
        load.name = split_roots[i].name

    report = balance_report(loads)
    metrics = balance_metrics(loads)

    print("\nPlanned distribution:")
    for row in report.itertuples():
        print(f"  {row.split_name:<12} {row.n_glaciers:>7} glaciers  "
              f"{row.total_area_km2:>12,.1f} km^2")
    print(f"\n  count spread: {metrics['count']['min']:.0f}-{metrics['count']['max']:.0f} "
          f"(relative range {metrics['count']['rel_range']:.2%})")
    print(f"  area  spread: {metrics['area_km2']['min']:,.1f}-"
          f"{metrics['area_km2']['max']:,.1f} km^2 "
          f"(relative range {metrics['area_km2']['rel_range']:.2%})")

    table = assignment_table(loads)
    out_path = Path(args.output) if args.output else registry_path.parent / "split_assignment.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    print(f"\n  assignment -> {out_path}")

    if args.apply:
        target = {
            gid: split_roots[load.index]
            for load in loads
            for gid in load.glacier_ids
        }
        moves = plan_moves(existing, target)
        print(f"\nMoves required: {len(moves)}")
        if args.dry_run:
            print("[DRY-RUN] no files will be moved")
        errors = apply_moves(moves, dry_run=args.dry_run)
        if errors:
            print(f"[DONE] {len(moves) - errors} moved, {errors} errors")
            return 1
        print(f"[DONE] {len(moves)} moved, 0 errors")
    else:
        print("\n(planning only -- pass --apply to move directories on disk)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
