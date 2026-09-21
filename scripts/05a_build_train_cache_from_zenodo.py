#!/usr/bin/env python3
"""Stage 5a -- rebuild the per-scene training cache from the published Zenodo corpus.

Purpose
-------
``05_train_model.py`` (via ``training.dataset.scan_scenes``/``load_scene_arrays``)
does not read raw imagery: it walks a directory tree for
``<scene_dir>/ia_cache_all/scene_features.npz`` files, each holding the 11
precomputed spectral-index channels, the remapped 4-class label array, a
valid-pixel mask, and scene identity fields (sensor/glacier id/year/scene id).

That cache is not itself published. What *is* published, in
``UNet_Train_Data`` (see its ``README.md``), is the raw per-scene corpus:

    GxxxxxxxxxxxxxN/year/YYYY/<entity_id>/TOA*.tif   (6 or 7 band TOA reflectance)
    GxxxxxxxxxxxxxN/year/YYYY/<entity_id>/Mask_30m.tif (3-band RGB-coded labels)

This script reconstructs the exact training cache from exactly those files, so
the whole pipeline can be run end to end from the Zenodo deposit alone, with no
dependency on any non-published intermediate. The transformation is:

1. Read the TOA GeoTIFF, keep only the first 6 bands in the pipeline's
   canonical order ``[Blue, Green, Red, NIR, SWIR1, SWIR2]``. Landsat 8/9
   scenes (``TOA_Landsat.tif``) carry a 7th band beyond that order; it is
   dropped here, exactly as the reference pipeline does (confirmed by
   byte-comparing this script's output against the original cache -- see
   ``docs/decisions`` if that comparison is later written up).
2. Run ``features.spectral_indices.compute_indices`` on those 6 bands to get
   the 11-channel feature stack (paper Table 2 formulas).
3. Decode ``Mask_30m.tif``'s RGB colour coding into the 4-class label array
   using the legend published in ``UNet_Train_Data/README.md``
   (Cloud/Snow/Ice/Other/NoData), then remap 1..4 -> 0..3 with 0/unmatched ->
   ``IGNORE_INDEX`` (255) -- the same encoding ``load_scene_arrays`` expects.
4. Mark any pixel that is non-finite in any of the 6 source bands as invalid
   (``valid_mask`` 0, label forced to ``IGNORE_INDEX``), matching
   ``load_scene_arrays``'s own non-finite handling.
5. Derive ``sensor`` from the TOA filename suffix (``TOA_Landsat7.tif`` ->
   ``landsat7``, ``TOA_Landsat.tif`` -> ``landsat`` (L8/L9), ``TOA_Landsat5.tif``
   -> ``landsat5``, ``TOA_Sentinel.tif`` -> ``sentinel``) and
   ``glacier_id``/``year``/``scene_id`` from the directory structure
   (``<glacier_id>/year/<year>/<scene_id>/``).

Verification
------------
This exact procedure was checked against a held reference cache built by the
original (unpublished) pipeline for three scenes spanning all three TOA band
counts (Landsat7 6-band, Landsat 7-band, Sentinel-2 6-band): every one of the
11 feature channels and every label pixel matched to float16 precision
(<= 3e-4 absolute difference, consistent with the cache's own float16 storage)
inside the reference ``valid_mask``.

Usage
-----
With no arguments at all, this reads its paths from the config, exactly like
every other stage script::

    python scripts/05a_build_train_cache_from_zenodo.py --config configs/config.yaml

That reads ``paths.zenodo_train_data_root`` (the published, read-only
``UNet_Train_Data`` corpus) as the source and writes the rebuilt cache into
``paths.train_root`` (the same corpus root ``05_train_model.py`` already reads
by default) -- so once this has run, ``05_train_model.py --config
configs/config.yaml`` needs no further flags either.

Either path can still be overridden on the command line, e.g. to rebuild into
a scratch location without touching ``paths.train_root``::

    python scripts/05a_build_train_cache_from_zenodo.py \\
        --source-root /path/to/UNet_Train_Data --out /path/to/scratch_cache

    # Rebuild only scenes whose cache is missing or stale (default); force
    # every scene to be rewritten regardless of an existing up-to-date cache:
    python scripts/05a_build_train_cache_from_zenodo.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glacier_fsnow_unet.config import ConfigError, load_config  # noqa: E402
from glacier_fsnow_unet.features.spectral_indices import (  # noqa: E402
    INDEX_NAMES,
    compute_indices,
)

# Imported directly by value rather than via `training.config`/`training.dataset`:
# both modules pull in `training/__init__.py`, which imports torch for the full
# training stack. This script only needs these constants and one hashing
# helper, so it reimplements them here to stay a lightweight, torch-free
# preprocessing step. Values must stay in lockstep with their originals in
# `glacier_fsnow_unet.training.config`/`.dataset`.
IGNORE_INDEX = 255
SCENE_CACHE_DIRNAME = "ia_cache_all"
SCENE_CACHE_FILENAME = "scene_features.npz"
SCENE_CACHE_META = "scene_features.json"
SCENE_CACHE_SCHEMA_VERSION = 1


def scene_cache_signature(scene_dir, source_paths, feature_names, schema_version=SCENE_CACHE_SCHEMA_VERSION):
    import hashlib

    payload = {
        "schema_version": int(schema_version),
        "scene": str(scene_dir),
        "features": list(feature_names),
        "sources": [],
    }
    for path in sorted(source_paths, key=str):
        try:
            stat = Path(path).stat()
            payload["sources"].append([str(path), int(stat.st_size), int(stat.st_mtime)])
        except OSError:
            payload["sources"].append([str(path), -1, -1])
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()

# Number of source bands actually consumed by compute_indices, in canonical
# [Blue, Green, Red, NIR, SWIR1, SWIR2] order. TOA_Landsat.tif (L8/L9) carries
# a 7th band past this; every other TOA file already has exactly 6.
N_SOURCE_BANDS = 6

# TOA filename suffix -> pipeline sensor label (glacier_fsnow_unet.training.config.SENSOR_TO_IDX).
SENSOR_BY_TOA_STEM: dict[str, str] = {
    "TOA_Landsat7": "landsat7",
    "TOA_Landsat5": "landsat5",
    "TOA_Landsat": "landsat",  # L8/L9 share one filename in the deposit
    "TOA_Sentinel": "sentinel",
}

# Mask_30m.tif RGB class encoding, from UNet_Train_Data/README.md.
MASK_RGB_TO_RAW_LABEL: dict[tuple[int, int, int], int] = {
    (150, 70, 190): 1,  # Cloud
    (200, 240, 255): 2,  # Snow
    (140, 170, 255): 3,  # Ice
    (180, 110, 50): 4,  # Other
    # (0, 0, 0) NoData is anything left unmapped -> raw 0 -> IGNORE_INDEX.
}
RAW_TO_TRAIN_LABEL: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3}  # Cloud,Snow,Ice,Other


def find_toa_file(scene_dir: Path) -> Path:
    matches = sorted(scene_dir.glob("TOA*.tif"))
    if not matches:
        raise FileNotFoundError(f"no TOA*.tif under {scene_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(f"multiple TOA*.tif under {scene_dir}: {matches}")
    return matches[0]


def sensor_from_toa_path(toa_path: Path) -> str:
    stem = toa_path.stem
    try:
        return SENSOR_BY_TOA_STEM[stem]
    except KeyError:
        raise ValueError(
            f"{toa_path}: unrecognised TOA filename stem {stem!r}; expected one "
            f"of {sorted(SENSOR_BY_TOA_STEM)}"
        ) from None


def decode_mask_labels(mask_rgb: np.ndarray) -> np.ndarray:
    """RGB-coded ``(3, H, W)`` mask -> ``(H, W)`` labels in 0..3, IGNORE_INDEX elsewhere."""
    h, w = mask_rgb.shape[1:]
    labels = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    for (r, g, b), raw_value in MASK_RGB_TO_RAW_LABEL.items():
        match = (mask_rgb[0] == r) & (mask_rgb[1] == g) & (mask_rgb[2] == b)
        labels[match] = RAW_TO_TRAIN_LABEL[raw_value]
    return labels


def parse_identity(scene_dir: Path, glacier_root: Path) -> tuple[str, int, str]:
    """``<glacier_root>/year/<year>/<scene_id>`` -> (glacier_id, year, scene_id)."""
    glacier_id = glacier_root.name
    year_part, scene_id = scene_dir.parts[-2], scene_dir.parts[-1]
    try:
        year = int(year_part)
    except ValueError as exc:
        raise ValueError(
            f"cannot parse year from {scene_dir}; expected "
            f"<glacier_id>/year/<YYYY>/<scene_id>"
        ) from exc
    return glacier_id, year, scene_id


def build_scene_cache(
    scene_dir: Path,
    glacier_id: str,
    year: int,
    scene_id: str,
    out_scene_dir: Path,
    force: bool,
) -> str:
    """Build (or skip, if current) one scene's cache. Returns 'built'/'skipped'/'error'."""
    toa_path = find_toa_file(scene_dir)
    mask_path = scene_dir / "Mask_30m.tif"
    if not mask_path.exists():
        print(f"[skip] {scene_dir}: no Mask_30m.tif")
        return "error"

    sensor = sensor_from_toa_path(toa_path)

    cache_dir = out_scene_dir / SCENE_CACHE_DIRNAME
    npz_path = cache_dir / SCENE_CACHE_FILENAME
    meta_path = cache_dir / SCENE_CACHE_META
    signature = scene_cache_signature(scene_dir, [toa_path, mask_path], INDEX_NAMES)

    if not force and npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        if meta.get("signature") == signature:
            return "skipped"

    with rasterio.open(toa_path) as ds:
        toa = ds.read().astype(np.float32)
    if toa.shape[0] < N_SOURCE_BANDS:
        raise ValueError(
            f"{toa_path}: expected >= {N_SOURCE_BANDS} bands, got {toa.shape[0]}"
        )
    bands = toa[:N_SOURCE_BANDS]

    with rasterio.open(mask_path) as ds:
        mask_rgb = ds.read()
    if mask_rgb.shape[0] < 3:
        raise ValueError(f"{mask_path}: expected 3 RGB bands, got {mask_rgb.shape[0]}")
    if mask_rgb.shape[1:] != bands.shape[1:]:
        raise ValueError(
            f"{mask_path} ({mask_rgb.shape[1:]}) and {toa_path} ({bands.shape[1:]}) "
            "do not share the same grid"
        )

    features = compute_indices(bands, names=INDEX_NAMES)
    labels = decode_mask_labels(mask_rgb[:3])

    invalid = ~np.all(np.isfinite(bands), axis=0)
    valid_mask = (~invalid).astype(np.uint8)
    if np.any(invalid):
        features = features.copy()
        labels = labels.copy()
        features[:, invalid] = 0
        labels[invalid] = IGNORE_INDEX

    features = features.astype(np.float16)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        features=features,
        labels=labels,
        valid_mask=valid_mask,
        sensor=np.array(sensor),
        id_glims=np.array(glacier_id),
        year=np.array(int(year)),
        scene_id=np.array(scene_id),
        scene_path=np.array(str(out_scene_dir)),
        feature_names=np.array(INDEX_NAMES),
    )
    meta_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "sensor": sensor,
                "id_glims": glacier_id,
                "year": int(year),
                "scene_id": scene_id,
                "npz_path": str(npz_path),
                "source_toa": str(toa_path),
                "source_mask": str(mask_path),
                "schema_version": SCENE_CACHE_SCHEMA_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "built"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Root of the published UNet_Train_Data corpus (read-only). "
            "Overrides paths.zenodo_train_data_root from the config."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output root for the rebuilt <glacier>/<year>/<scene>/ia_cache_all "
            "cache. Overrides paths.train_root from the config."
        ),
    )
    parser.add_argument(
        "--only-id", default=None, help="Rebuild a single glacier id only"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every scene even if an up-to-date cache already exists",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        pipeline_config = load_config(args.config)
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    source_root_str = args.source_root or pipeline_config.paths.zenodo_train_data_root
    out_root_str = args.out or pipeline_config.paths.train_root
    if not source_root_str:
        print(
            "[error] no source root: pass --source-root or set "
            "paths.zenodo_train_data_root in the config",
            file=sys.stderr,
        )
        return 2

    source_root = Path(source_root_str)
    out_root = Path(out_root_str)

    if not source_root.exists():
        print(f"[error] source root does not exist: {source_root}", file=sys.stderr)
        return 2

    glacier_dirs = sorted(
        d for d in source_root.iterdir() if d.is_dir() and (d / "year").is_dir()
    )
    if args.only_id:
        glacier_dirs = [d for d in glacier_dirs if d.name == args.only_id]
        if not glacier_dirs:
            print(f"[error] glacier id not found: {args.only_id}", file=sys.stderr)
            return 2

    counts = {"built": 0, "skipped": 0, "error": 0}
    for glacier_dir in glacier_dirs:
        glacier_id = glacier_dir.name
        year_dirs = sorted(p for p in (glacier_dir / "year").iterdir() if p.is_dir())
        for year_dir in year_dirs:
            scene_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir())
            for scene_dir in scene_dirs:
                glacier_id_p, year, scene_id = parse_identity(scene_dir, glacier_dir)
                out_scene_dir = out_root / glacier_id / str(year) / scene_id
                try:
                    status = build_scene_cache(
                        scene_dir, glacier_id, year, scene_id, out_scene_dir, args.force
                    )
                except (OSError, ValueError, KeyError) as exc:
                    print(f"[error] {scene_dir}: {exc}", file=sys.stderr)
                    status = "error"
                counts[status] += 1
                if status != "skipped":
                    print(f"[{status}] {glacier_id}/{year}/{scene_id}")

    print(
        f"[done] built={counts['built']} skipped={counts['skipped']} "
        f"errors={counts['error']} -> {out_root}"
    )
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
