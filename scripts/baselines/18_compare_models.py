#!/usr/bin/env python3
"""Statistical comparison between two (or more) model score series.

Two modes:

    scores    Generic: compare arbitrary per-seed test-score series read
              from JSON or CSV files, one file per model/condition. Welch's
              t-test (or a paired t-test with --paired) plus a bootstrap CI
              on the mean difference between every pair of series.
    checkpoints
              Reads `scene_confusions` / `scene_splits` / `scene_identities`
              from a family of trained U-Net checkpoints (glacier-level and
              scene-level split, several seeds each) and reproduces the four
              leakage-diagnostic comparisons from the paper's Table
              tab:leakage: per-scene seen-vs-unseen glaciers, per-tile pooled
              seen-vs-unseen, glacier-strict vs scene-unseen, and the
              headline glacier-vs-scene comparison.

Input file format for `scores` (JSON):

    {"name": "DeepLabv3", "test_miou": [0.612, 0.618, 0.609, 0.615, 0.611]}

or a bare list `[0.612, 0.618, ...]` (the file's stem becomes the name), or
CSV with a single numeric column (any header). Every guard already used
elsewhere in this repo's comparisons is preserved here: with fewer than 2
seeds in either group, `significant_at_0.05` is `None`, never a
silently-wrong `False`.

Examples:

    python scripts/baselines/18_compare_models.py scores \\
        --input _work/baselines/deeplab/deeplab_confirmation_summary.json:test_miou_per_seed \\
        --input _work/baselines/rf/rf_baseline_results.json:seed_sensitivity.per_seed[*].test_miou_macro \\
        --reference-unet

    python scripts/baselines/18_compare_models.py checkpoints \\
        --checkpoint-dir _work/rse_sigtest/checkpoints

The `checkpoints` mode is deliberately NOT a generic reader: it depends on
a `meta` schema (`scene_confusions` keyed by split family and partition,
`scene_splits`, `scene_identities`) that only the U-Net training pipeline's
checkpoints carry, and on domain-specific bookkeeping (seen vs unseen
glacier membership, per-scene vs per-tile pooling) that has no equivalent
for a Random Forest or DeepLab confusion matrix. Generalising it further
than this would mean inventing a "seen/unseen glacier" concept for models
that were never trained with a scene-level split in the first place --
so it stays a dedicated mode reading exactly that checkpoint family,
rather than a fully generic pathway.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import UNET_TEST_MIOU_5SEEDS, compare_series  # noqa: E402


# ---------------------------------------------------------------------------
# mode: scores
# ---------------------------------------------------------------------------


def _resolve_path_expr(obj: Any, expr: str) -> Any:
    """A tiny subset of JSONPath: dotted attribute access, `[*]` to map over
    a list, and `[i]` for a literal index. Enough to reach into the nested
    JSON summaries these scripts already produce without a new dependency.
    """
    if not expr:
        return obj
    tokens = re.findall(r"[^.\[\]]+|\[\*\]|\[\d+\]", expr)
    current = obj
    for tok in tokens:
        if tok == "[*]":
            if not isinstance(current, list):
                raise ValueError(f"'[*]' applied to non-list at {tok!r} in {expr!r}")
            current = current  # expand on the next token, applied per-element
            continue
        if tok.startswith("[") and tok.endswith("]"):
            idx = int(tok[1:-1])
            current = current[idx]
            continue
        if isinstance(current, list):
            current = [_resolve_path_expr(item, tok) for item in current]
        else:
            current = current[tok]
    return current


def load_score_series(spec: str) -> tuple[str, list[float]]:
    """Parse a `--input` spec: `path.json` (whole file), `path.json:field`,
    `path.json:a.b[*].c` (dotted / list-expansion path into the JSON), or a
    plain `.csv` with one numeric column.
    """
    if ":" in spec and not spec[1:3] == ":\\":  # avoid splitting a Windows drive letter
        path_str, field_expr = spec.split(":", 1)
    else:
        path_str, field_expr = spec, ""
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")

    if path.suffix.lower() == ".csv":
        values = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header_is_numeric = rows and _try_float(rows[0][0]) is not None
        data_rows = rows if header_is_numeric else rows[1:]
        for row in data_rows:
            if row:
                values.append(float(row[0]))
        name = path.stem
        return name, values

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return path.stem, [float(v) for v in raw]

    name = raw.get("name", path.stem) if isinstance(raw, dict) else path.stem
    if not field_expr:
        # No field given: look for a conventional key.
        for key in ("test_miou_per_seed", "test_miou", "scores", "values"):
            if isinstance(raw, dict) and key in raw:
                field_expr = key
                break
        if not field_expr:
            raise ValueError(
                f"{path}: no field given and none of the conventional keys "
                f"(test_miou_per_seed, test_miou, scores, values) were found"
            )
    resolved = _resolve_path_expr(raw, field_expr)
    flat = _flatten_numeric(resolved)
    return name, flat


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _flatten_numeric(obj: Any) -> list[float]:
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, list):
        out: list[float] = []
        for item in obj:
            out.extend(_flatten_numeric(item))
        return out
    raise ValueError(f"could not extract numeric values from {obj!r}")


def run_scores_mode(args) -> int:
    series: dict[str, list[float]] = {}
    for spec in args.input:
        name, values = load_score_series(spec)
        if name in series:
            name = f"{name}_{len(series)}"
        series[name] = values
        print(f"[scores] {name}: n={len(values)} values={values}")

    if args.reference_unet:
        series["UNet_published"] = list(UNET_TEST_MIOU_5SEEDS)
        print(f"[scores] UNet_published: n={len(UNET_TEST_MIOU_5SEEDS)} values={list(UNET_TEST_MIOU_5SEEDS)}")

    names = list(series.keys())
    if len(names) < 2:
        print("[error] need at least 2 score series to compare", file=sys.stderr)
        return 2

    import numpy as np

    summary_rows = []
    for name in names:
        values = np.asarray(series[name], dtype=np.float64)
        summary_rows.append({
            "name": name, "n_seeds": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        })

    comparisons = []
    pairs = (
        [(names[0], b) for b in names[1:]]
        if args.pairwise == "first_vs_rest"
        else [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    )
    for a, b in pairs:
        result = compare_series(
            series[a], series[b], paired=args.paired,
            n_boot=args.n_boot, boot_seed=args.boot_seed,
        )
        comparisons.append({"candidate": a, "reference": b, **result})
        sig = result["significant_at_0.05"]
        sig_str = "yes" if sig else ("no" if sig is False else "undetermined (n<2)")
        print(
            f"[compare] {a} vs {b}: mean_diff={result['mean_diff_candidate_minus_reference']:.4f} "
            f"p={result['p_value']:.5f} 95% CI={result['bootstrap_ci_95_diff']} significant={sig_str}"
        )

    output = {"summary": summary_rows, "comparisons": comparisons}
    out_path = Path(args.out) if args.out else REPO_ROOT / "_work" / "baselines" / "compare_scores_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"[done] {out_path}")
    return 0


# ---------------------------------------------------------------------------
# mode: checkpoints
# ---------------------------------------------------------------------------


def _miou_from_confusion(cm) -> float:
    import numpy as np

    cm = np.asarray(cm, dtype=np.float64)
    ious = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else float("nan")


def _load_meta(checkpoint_dir: Path, split_family: str, seed: int) -> dict:
    import torch

    path = checkpoint_dir / split_family / f"seed_{seed}_model.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["meta"]


def _scene_level_seen_unseen_series(checkpoint_dir: Path, seed: int) -> dict:
    import numpy as np

    meta = _load_meta(checkpoint_dir, "scene_split", seed)
    identities = meta["scene_identities"]
    splits = meta["scene_splits"]
    confusions = meta["scene_confusions"]["valid"]

    train_glaciers = {identities[i]["id_glims"] for i, s in enumerate(splits) if s == "train"}
    valid_global_idxs = [int(k) for k in confusions.keys()]

    seen_pooled = np.zeros((4, 4), dtype=np.int64)
    unseen_pooled = np.zeros((4, 4), dtype=np.int64)
    seen_scene_mious: list[float] = []
    unseen_scene_mious: list[float] = []

    for gidx in valid_global_idxs:
        cm = np.array(confusions[str(gidx)], dtype=np.int64)
        if cm.sum() == 0:
            continue
        glacier_id = identities[gidx]["id_glims"]
        scene_miou = _miou_from_confusion(cm)
        if glacier_id in train_glaciers:
            seen_pooled += cm
            if np.isfinite(scene_miou):
                seen_scene_mious.append(scene_miou)
        else:
            unseen_pooled += cm
            if np.isfinite(scene_miou):
                unseen_scene_mious.append(scene_miou)

    return {
        "seed": seed,
        "seen_tile_miou": _miou_from_confusion(seen_pooled),
        "unseen_tile_miou": _miou_from_confusion(unseen_pooled),
        "seen_scene_miou_mean": float(np.mean(seen_scene_mious)) if seen_scene_mious else float("nan"),
        "unseen_scene_miou_mean": float(np.mean(unseen_scene_mious)) if unseen_scene_mious else float("nan"),
        "n_seen_scenes": len(seen_scene_mious),
        "n_unseen_scenes": len(unseen_scene_mious),
    }


def _pooled_valid_miou(checkpoint_dir: Path, split_family: str, seed: int) -> float:
    import numpy as np

    meta = _load_meta(checkpoint_dir, split_family, seed)
    confusions = meta["scene_confusions"]["valid"]
    pooled = np.zeros((4, 4), dtype=np.int64)
    for cm in confusions.values():
        pooled += np.array(cm, dtype=np.int64)
    return _miou_from_confusion(pooled)


def run_checkpoints_mode(args) -> int:
    import numpy as np

    checkpoint_dir = Path(args.checkpoint_dir)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    per_seed_scene_level = [_scene_level_seen_unseen_series(checkpoint_dir, s) for s in seeds]
    glacier_level_valid = np.array(
        [_pooled_valid_miou(checkpoint_dir, "glacier_split", s) for s in seeds]
    )
    scene_level_valid = np.array(
        [_pooled_valid_miou(checkpoint_dir, "scene_split", s) for s in seeds]
    )
    seen_tile = np.array([r["seen_tile_miou"] for r in per_seed_scene_level])
    unseen_tile = np.array([r["unseen_tile_miou"] for r in per_seed_scene_level])
    seen_scene = np.array([r["seen_scene_miou_mean"] for r in per_seed_scene_level])
    unseen_scene = np.array([r["unseen_scene_miou_mean"] for r in per_seed_scene_level])

    report: dict[str, Any] = {"per_seed_raw": per_seed_scene_level}

    c1 = compare_series(seen_scene, unseen_scene, paired=True, boot_seed=1)
    c2 = compare_series(seen_tile, unseen_tile, paired=True, boot_seed=2)
    b = compare_series(glacier_level_valid, unseen_tile, paired=False, boot_seed=3)
    a = compare_series(glacier_level_valid, scene_level_valid, paired=False, boot_seed=4)

    report["C1_per_scene_seen_vs_unseen"] = c1
    report["C2_per_tile_seen_vs_unseen"] = c2
    report["B_glacier_strict_vs_scene_unseen"] = {
        **b,
        "note": "no pre-registered equivalence margin; a non-significant "
                "difference is reported as such, not as proven equivalence",
    }
    report["A_headline_glacier_vs_scene"] = {
        **a,
        "caveat": "independently re-optimised HPO per run; not a controlled comparison",
    }

    print(f"[C1] per-scene seen vs unseen: diff={c1['mean_diff_candidate_minus_reference']:.4f} "
          f"p={c1['p_value']:.5f} significant={c1['significant_at_0.05']}")
    print(f"[C2] per-tile pooled seen vs unseen: diff={c2['mean_diff_candidate_minus_reference']:.4f} "
          f"p={c2['p_value']:.5f} significant={c2['significant_at_0.05']}")
    print(f"[B] glacier-strict vs scene-unseen: diff={b['mean_diff_candidate_minus_reference']:.4f} "
          f"p={b['p_value']:.5f} significant={b['significant_at_0.05']}")
    print(f"[A] headline glacier vs scene: diff={a['mean_diff_candidate_minus_reference']:.4f} "
          f"p={a['p_value']:.5f} significant={a['significant_at_0.05']}")

    out_path = Path(args.out) if args.out else REPO_ROOT / "_work" / "baselines" / "compare_checkpoints_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[done] {out_path}")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scores = subparsers.add_parser("scores", help="compare arbitrary per-seed score series")
    scores.add_argument(
        "--input", action="append", required=True,
        help="path[:field] to a JSON/CSV file of per-seed scores; repeat for each series",
    )
    scores.add_argument(
        "--reference-unet", action="store_true",
        help="also include the published U-Net's 5-seed test mIoU as a series",
    )
    scores.add_argument("--paired", action="store_true", help="use a paired t-test (equal-length, matched series)")
    scores.add_argument(
        "--pairwise", choices=("first_vs_rest", "all"), default="first_vs_rest",
        help="compare the first series against every other (default), or every pair",
    )
    scores.add_argument("--n-boot", type=int, default=10_000)
    scores.add_argument("--boot-seed", type=int, default=0)
    scores.add_argument("--out", default=None, help="output JSON path")

    checkpoints = subparsers.add_parser(
        "checkpoints", help="leakage-diagnostic comparisons from a glacier_split/scene_split checkpoint family"
    )
    checkpoints.add_argument(
        "--checkpoint-dir", required=True,
        help="directory containing glacier_split/ and scene_split/ subdirs, each with seed_N_model.pt",
    )
    checkpoints.add_argument("--n-seeds", type=int, default=5)
    checkpoints.add_argument("--seed-start", type=int, default=0)
    checkpoints.add_argument("--out", default=None, help="output JSON path")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "scores":
        return run_scores_mode(args)
    return run_checkpoints_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
