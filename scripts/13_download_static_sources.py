#!/usr/bin/env python3
"""Stage 13 -- fetch static (non-temporal) per-glacier feature sources.

Purpose
-------
Sample five sources that describe one fixed value per glacier rather than a
1984-2025 time series: RGI structural attributes, Koppen-Geiger climate
classification, WorldClim 1970-2000 climatology, circum-Arctic permafrost
extent, and distance to the nearest coastline. Kept in its own stage and its
own output file, separate from stage 11/12's per-year climate sources and
from every other produced pipeline output (VGS, F_snow, annual composites),
because all of those are data this pipeline *produces* while every source
here is *retrieved* unmodified from an external archive or file the user
supplies -- keeping them apart makes that distinction visible on disk
instead of folding everything into one table.

Every source is independently selectable via ``--sources``, so a user who
only wants, say, Koppen-Geiger and coastline distance does not have to fetch
the (large) WorldClim archive or provide RGI region paths.

Inputs
------
- The glacier registry (stage 1): ``glims_id``, ``centroid_lon``, ``centroid_lat``.
- RGI structural attributes: local ``*-attributes.csv`` files shipped with
  the RGI region shapefile packages this repo already reads in stage 1
  (``--rgi-region-root``, repeatable, or ``STATIC_RGI_REGION_ROOTS`` in
  ``.env``).
- Koppen-Geiger / WorldClim / Permafrost: fetched over HTTP/FTP on first run
  and cached under ``<features_root>/static_cache/<slug>/``.
- Coastline: a shapefile (or zipped shapefile) via ``--coastline``, required
  only when ``Coastline`` is one of the selected ``--sources``.

Outputs
-------
- ``<features_root>/raw_sources/static_features.parquet`` -- one row per
  glacier, one column per retrieved static variable, plus a
  ``<variable>_source`` column naming which external source it came from.

Usage
-----
    python scripts/13_download_static_sources.py --config configs/config.yaml --coastline path/to/coastline.shp
    python scripts/13_download_static_sources.py --sources KoppenGeiger,WorldClim
    python scripts/13_download_static_sources.py --sources Coastline --coastline coast.zip
    python scripts/13_download_static_sources.py --rgi-region-root D:/.../RGI2000-v7.0-G-01_alaska
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.features.static_source_defs import (  # noqa: E402
    STATIC_SOURCES,
    build_rgi_structural,
    get_static_source,
)
from glacier_fsnow_unet.glaciers.registry import REGISTRY_FILENAME, read_registry  # noqa: E402
from glacier_fsnow_unet.inference.coastline import (  # noqa: E402
    DEFAULT_DISTANCE_CRS,
    CoastlineError,
    build_coastline_distance,
)

#: Sources with no remote fetch or no uniform StaticSourceSpec builder --
#: handled as special cases in main() rather than through STATIC_SOURCES.
_SPECIAL_SOURCES = ("RGI_Structural", "Coastline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and sample static (non-temporal) glacier feature sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--registry", default=None, help="Path to glacier_registry.parquet")
    parser.add_argument(
        "--features-root",
        default=None,
        help="Root for feature outputs (default: <data_root>/features)",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated sources (default: all -- "
             f"{','.join(_SPECIAL_SOURCES[:1])},{','.join(STATIC_SOURCES)},{_SPECIAL_SOURCES[1]})",
    )
    parser.add_argument(
        "--rgi-region-root",
        action="append",
        default=None,
        help="Local RGI region folder (repeatable). Default: env STATIC_RGI_REGION_ROOTS, "
             "or the parent folders of config.isolated_glacier.rgi_shapefiles.",
    )
    parser.add_argument(
        "--coastline",
        default=None,
        help="Coastline shapefile or zipped shapefile (required if 'Coastline' is selected)",
    )
    parser.add_argument(
        "--distance-crs",
        default=DEFAULT_DISTANCE_CRS,
        help="Metric CRS used for the coastline distance computation",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached downloads and refetch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config, args.env_file)
    except ConfigError as exc:
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

    glaciers = registry[["glims_id", "centroid_lon", "centroid_lat"]].copy()
    glaciers["glims_id"] = glaciers["glims_id"].astype(str)

    features_root = Path(args.features_root or (Path(cfg.paths.data_root) / "features"))
    cache_root = features_root / "static_cache"
    out_dir = features_root / "raw_sources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "static_features.parquet"

    available = ["RGI_Structural", *STATIC_SOURCES, "Coastline"]
    if args.sources:
        selected = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        # Coastline needs --coastline, which most runs won't have on hand
        # (it isn't bundled -- see docs/decisions/static_feature_sources.md).
        # Default to it only when the user actually supplied one, so a plain
        # `--sources`-less run never fails just because no shapefile is
        # configured.
        selected = [s for s in available if s != "Coastline" or args.coastline]

    print("=" * 78)
    print("Stage 13: static feature sources")
    print("=" * 78)
    print(f"  glaciers: {len(glaciers)} | sources: {', '.join(selected)}")
    print(f"  output:   {out_path}")

    frames = []
    for name in selected:
        if name not in available:
            print(f"  [skip] {name}: unknown source (available: {', '.join(available)})", file=sys.stderr)
            continue

        print(f"    {name} ...", flush=True)
        try:
            if name == "RGI_Structural":
                roots_raw = args.rgi_region_root or (
                    os.environ.get("STATIC_RGI_REGION_ROOTS", "").split(os.pathsep)
                    if os.environ.get("STATIC_RGI_REGION_ROOTS")
                    else [str(Path(p).parent) for p in cfg.isolated_glacier.rgi_shapefiles]
                )
                roots = [Path(r) for r in roots_raw if str(r).strip()]
                frame = build_rgi_structural(glaciers, roots)
            elif name == "Coastline":
                if not args.coastline:
                    raise CoastlineError("--coastline is required to select the Coastline source")
                frame = build_coastline_distance(glaciers, args.coastline, distance_crs=args.distance_crs)
            else:
                spec = get_static_source(name)
                cache_dir = cache_root / spec.cache_subdir
                frame = spec.builder(glaciers, cache_dir)

            value_cols = [c for c in frame.columns if c != "glims_id"]
            for col in value_cols:
                frame[f"{col}_source"] = name
            frames.append(frame)
            print(f"    {name}: {len(frame)} row(s), {len(value_cols)} variable(s)")
        except Exception as exc:  # noqa: BLE001
            print(f"    {name} failed: {exc}", file=sys.stderr)

    if not frames:
        print("[ERROR] No static source produced any rows.", file=sys.stderr)
        return 1

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="glims_id", how="outer")

    if out_path.is_file():
        # A run selecting only a subset of --sources (or --force-refetching
        # just one of them) must not erase columns from other sources
        # already written to this output file -- merge onto the existing
        # table instead of overwriting it. --force only controls whether a
        # selected source's own cached download is refetched (see spec
        # builders above), never whether this file-level merge happens.
        existing = pd.read_parquet(out_path)
        new_cols = [c for c in merged.columns if c == "glims_id" or c not in existing.columns]
        merged = existing.merge(merged[new_cols], on="glims_id", how="outer")

    merged.to_parquet(out_path, index=False)
    print(f"\n  rows: {len(merged)} | columns: {len(merged.columns)}")
    print(f"  output -> {out_path}")
    print("\n[DONE] Stage 13 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
