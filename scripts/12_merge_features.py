#!/usr/bin/env python3
"""Stage 12 -- merge climate sources into the delivered feature parquets.

Purpose
-------
Combine the per-source absolute values from stage 11 into one merged table per
variable family, filling each glacier-year from the highest-priority source
that has a value, then recomputing the 1991-2020 climatology and anomalies on
the merged series (DESIGN_FEATURES.md sections 1.2-1.4).

Priority: Daymet V4 > TerraClimate > ERA5-Land > ERA5.

Inputs
------
- ``<features_root>/raw_sources/<slug>.parquet`` from stage 11.

Outputs
-------
- ``<features_root>/merged/features_temporal.parquet`` with, per variable:
  ``clim_{var}``, ``anom_{var}``, ``{var}_source``, ``{var}_quality_flag``,
  ``{var}_coverage_flag``. Absolute values are deliberately not delivered.

Usage
-----
    python scripts/12_merge_features.py --config configs/config.yaml
    python scripts/12_merge_features.py --variables t2m_summer,precip_annual_mm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.features.climatology import (  # noqa: E402
    merge_variables,
    validate_traceability,
)
from glacier_fsnow_unet.features.source_defs import (  # noqa: E402
    CLIMATOLOGY_REFERENCE,
    SOURCE_PRIORITY,
    SOURCES,
    get_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge climate sources by priority and compute clim/anom features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument(
        "--features-root",
        default=None,
        help="Root for feature outputs (default: <data_root>/features)",
    )
    parser.add_argument(
        "--variables",
        default=None,
        help="Comma-separated variables to merge (default: every shared variable)",
    )
    parser.add_argument(
        "--reference",
        default=f"{CLIMATOLOGY_REFERENCE[0]}-{CLIMATOLOGY_REFERENCE[1]}",
        help="Climatology reference window as 'START-END'",
    )
    parser.add_argument(
        "--output", default=None, help="Output Parquet path for the merged features"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    features_root = Path(
        args.features_root or (Path(cfg.paths.data_root) / "features")
    )
    raw_root = features_root / "raw_sources"
    if not raw_root.is_dir():
        print(
            f"[ERROR] No source parquets found at {raw_root}.\n"
            f"        Run scripts/11_download_features.py first.",
            file=sys.stderr,
        )
        return 2

    try:
        start, end = (int(part) for part in args.reference.split("-", 1))
    except ValueError:
        print(f"[ERROR] --reference must be 'START-END', got '{args.reference}'",
              file=sys.stderr)
        return 2

    print("=" * 78)
    print("Stage 12: merge climate features")
    print("=" * 78)
    print(f"  priority:  {' > '.join(SOURCE_PRIORITY)}")
    print(f"  reference: {start}-{end}")

    sources: dict[str, pd.DataFrame] = {}
    for name in SOURCES:
        path = raw_root / f"{get_source(name).slug}.parquet"
        if not path.is_file():
            print(f"  [skip] {name}: {path.name} not found")
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            print(f"  [skip] {name}: empty")
            continue
        sources[name] = frame
        print(f"  [load] {name}: {len(frame)} rows, {len(frame.columns)} columns")

    if not sources:
        print("[ERROR] No usable source parquet was loaded.", file=sys.stderr)
        return 2

    if args.variables:
        variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    else:
        # Every variable that at least one source provides, excluding keys.
        seen: set[str] = set()
        for frame in sources.values():
            seen.update(frame.columns)
        variables = sorted(seen - {"id_glims", "year"})

    print(f"  variables: {len(variables)}")

    merged = merge_variables(
        sources, variables, priority=SOURCE_PRIORITY, reference=(start, end)
    )

    if merged.empty:
        print("[ERROR] Merge produced no rows.", file=sys.stderr)
        return 1

    merged_variables = [
        v for v in variables if f"clim_{v}" in merged.columns
    ]
    missing = validate_traceability(merged, merged_variables)
    if missing:
        print(
            f"[WARN] {len(missing)} traceability column(s) missing, e.g. {missing[:5]}",
            file=sys.stderr,
        )

    out_dir = features_root / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "features_temporal.parquet"
    merged.to_parquet(out_path, index=False)

    print(f"\n  merged rows: {len(merged)}")
    print(f"  glaciers:    {merged['id_glims'].nunique()}")
    print(f"  years:       {int(merged['year'].min())}-{int(merged['year'].max())}")
    print(f"  variables:   {len(merged_variables)} with clim/anom + traceability")
    print(f"  output ->    {out_path}")
    print("\n[DONE] Stage 12 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
