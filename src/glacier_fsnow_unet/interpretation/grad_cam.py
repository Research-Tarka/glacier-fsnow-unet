"""Grad-CAM localisation maps, one per class, for a segmentation U-Net.

What it answers
---------------
The attribution in :mod:`glacier_fsnow_unet.training.shap_importance` answers
"which *bands* did this decision use". Grad-CAM answers the orthogonal
question: "which *part of the tile* did it look at". For a segmentation model
that is not the tautology it might sound like -- the network's receptive field
is far wider than one pixel, so a pixel labelled Ice can be driven by evidence
from terrain several tens of metres away, and seeing that spatial support is
what makes an implausible classification diagnosable.

The construction
----------------
For a chosen class `c` and a chosen convolutional stage, take the activations
`A_k` of every channel `k` at that stage and the gradient of the class score
with respect to them. Average each channel's gradient over space to get its
weight, then form the weighted channel sum and keep its positive part:

    L_c = relu( sum_k  mean_spatial(dS_c / dA_k) * A_k )

The ReLU is not cosmetic: negative contributions are evidence *against* class
`c`, and mixing them into a map read as "where the evidence was" would show
the two with opposite meaning at the same intensity.

The score `S_c` is the class logit averaged over the tile's pixels. A dense
`(B, C, H, W)` output has no single scalar to differentiate, and this is the
same reduction the expected-gradients estimator in `shap_importance` takes, so
the two analyses answer their different questions about the same quantity
rather than about two subtly different ones.

Which layer
-----------
The deepest encoder stage (`enc4` at the bottleneck) by default. It carries the
coarsest grid and the most semantic channels, and sits upstream of every
decoder skip connection, so its map reflects what the network *decided* rather
than what the final upsampling stage merely reconstructed. Any named module can
be substituted; the code resolves it by attribute path.

Why hand-rolled rather than a library
-------------------------------------
`pytorch-grad-cam` and `captum.LayerGradCam` both implement this, and both
assume a classification head: one scalar per sample, with the class chosen by
an index into a `(B, C)` output. Adapting either to a dense `(B, C, H, W)`
output means supplying exactly the spatial-mean reduction written above and
then unpicking their batching to keep the per-class maps separate. The
mechanism itself is a forward hook, a backward hook and four lines of tensor
arithmetic, so wrapping a library around it would add a dependency and an
adapter layer without removing any of the code that actually needs review. It
also keeps the class-conditioning policy -- only explaining a class on tiles
that genuinely contain it -- under this module's control, which is the part
that is specific to this task rather than generic to Grad-CAM.

Persistence
-----------
Maps are per-pixel arrays aligned to a tile's grid, so they are written to a
zarr store (see :func:`write_maps_to_zarr`), never to loose image or `.npz`
files. The HTML report is a rendering of that data, not the data itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "GradCamResult",
    "resolve_layer",
    "compute_grad_cam",
    "write_maps_to_zarr",
]


@dataclass
class GradCamResult:
    """Per-class Grad-CAM maps and the tiles they were computed on.

    `maps` is `(n_classes, n_tiles, H, W)`, normalised to [0, 1] per tile.
    `tile_has_class` is `(n_classes, n_tiles)` and records whether that tile
    actually contained enough pixels of the class to be worth explaining;
    entries where it is False hold zeros.
    """

    class_names: tuple[str, ...]
    maps: np.ndarray
    tile_has_class: np.ndarray
    target_layer: str
    n_tiles: int

    def class_mean_map(self, class_index: int) -> np.ndarray:
        """Mean map for one class, over the tiles that contained it."""
        selected = self.tile_has_class[class_index]
        if not np.any(selected):
            return np.zeros(self.maps.shape[2:], dtype=np.float32)
        return self.maps[class_index][selected].mean(axis=0).astype(np.float32)

    def summary(self) -> list[dict[str, Any]]:
        """One record per class, for the JSON summary."""
        records = []
        for index, name in enumerate(self.class_names):
            selected = self.tile_has_class[index]
            values = self.maps[index][selected] if np.any(selected) else np.zeros(1)
            records.append(
                {
                    "class": index,
                    "class_name": name,
                    "n_tiles_explained": int(selected.sum()),
                    "mean_activation": float(values.mean()),
                    "peak_activation": float(values.max()) if values.size else 0.0,
                }
            )
        return records


def resolve_layer(model: Any, path: str) -> Any:
    """Resolve a dotted module path (``"enc4"``, ``"enc4.block"``) on a model.

    Unwraps a ``DataParallel``/``compile`` wrapper first, so a path written
    against the plain architecture keeps working on a wrapped model.
    """
    target = getattr(model, "module", model)
    target = getattr(target, "_orig_mod", target)
    for part in str(path).split("."):
        if not hasattr(target, part):
            available = ", ".join(
                name for name, _ in getattr(target, "named_children", lambda: [])()
            )
            raise AttributeError(
                f"model has no module '{path}' (stopped at '{part}'; "
                f"available here: {available or 'none'})"
            )
        target = getattr(target, part)
    return target


def _normalise(array: "np.ndarray") -> np.ndarray:
    """Rescale one map to [0, 1]; an all-constant map becomes all zeros."""
    lowest = float(array.min())
    highest = float(array.max())
    if not np.isfinite(lowest) or not np.isfinite(highest) or highest <= lowest:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lowest) / (highest - lowest)).astype(np.float32)


def compute_grad_cam(
    model: Any,
    loader: Iterable[Any],
    device: Any,
    class_names: Sequence[str],
    target_layer: str = "enc4",
    n_samples: int = 48,
    batch_size: int = 8,
    class_conditioned: bool = True,
    min_class_pixels: int = 256,
    ignore_index: int = 255,
) -> GradCamResult:
    """Grad-CAM maps for every class over up to ``n_samples`` tiles.

    Parameters
    ----------
    class_conditioned
        Only explain class `c` on tiles holding at least ``min_class_pixels``
        ground-truth pixels of it. Explaining Ice on a tile with no ice
        produces a map of where the model would *hypothetically* look, which
        averages into noise and is not what the report claims to show.
    min_class_pixels
        Threshold for the above. Ignored when ``class_conditioned`` is False.

    Returns
    -------
    GradCamResult
        Maps upsampled back to the input tile's resolution.
    """
    import torch
    import torch.nn.functional as functional

    model.eval()
    layer = resolve_layer(model, target_layer)

    activations: dict[str, Any] = {}
    gradients: dict[str, Any] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        # Retaining the grad on the output tensor itself is more robust than a
        # full backward hook, which fires per-input on modules with several.
        output.retain_grad()

    handle = layer.register_forward_hook(forward_hook)

    tiles: list[Any] = []
    labels: list[Any] = []
    collected = 0
    try:
        for batch in loader:
            x = batch[0]
            y = batch[1]
            take = min(int(x.size(0)), max(0, int(n_samples) - collected))
            if take <= 0:
                break
            tiles.append(x[:take].detach())
            labels.append(y[:take].detach())
            collected += take
            if collected >= int(n_samples):
                break

        if not tiles:
            raise ValueError("the loader yielded no tiles for Grad-CAM")

        samples = torch.cat(tiles, dim=0)
        truths = torch.cat(labels, dim=0)
        n_tiles = int(samples.size(0))
        n_classes = len(class_names)
        height, width = int(samples.size(2)), int(samples.size(3))

        maps = np.zeros((n_classes, n_tiles, height, width), dtype=np.float32)
        has_class = np.zeros((n_classes, n_tiles), dtype=bool)

        # Which (class, tile) pairs are worth a backward pass at all.
        truth_np = truths.cpu().numpy()
        for class_index in range(n_classes):
            if class_conditioned:
                counts = (
                    (truth_np == class_index) & (truth_np != ignore_index)
                ).reshape(n_tiles, -1).sum(axis=1)
                has_class[class_index] = counts >= int(min_class_pixels)
            else:
                has_class[class_index] = True

        step = max(1, int(batch_size))
        for start in range(0, n_tiles, step):
            stop = min(start + step, n_tiles)
            x = samples[start:stop].to(device).float()

            for class_index in range(n_classes):
                wanted = has_class[class_index, start:stop]
                if not wanted.any():
                    continue

                model.zero_grad(set_to_none=True)
                activations.pop("value", None)
                gradients.pop("value", None)

                logits = model(x)
                # One scalar per class: the class logit averaged over pixels,
                # summed over the batch so a single backward pass yields every
                # tile's gradient independently (the tiles do not interact).
                score = logits[:, class_index].mean(dim=(1, 2)).sum()
                score.backward()

                feature = activations.get("value")
                if feature is None or feature.grad is None:
                    continue

                # (B, K, h, w): channel weights are the spatially-averaged
                # gradients; the map is their weighted sum, positive part only.
                weights = feature.grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * feature.detach()).sum(dim=1, keepdim=True))
                cam = functional.interpolate(
                    cam, size=(height, width), mode="bilinear", align_corners=False
                )
                cam_np = cam.squeeze(1).detach().cpu().numpy()

                for offset in range(cam_np.shape[0]):
                    if not wanted[offset]:
                        continue
                    maps[class_index, start + offset] = _normalise(cam_np[offset])
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    return GradCamResult(
        class_names=tuple(str(name) for name in class_names),
        maps=maps,
        tile_has_class=has_class,
        target_layer=str(target_layer),
        n_tiles=n_tiles,
    )


def write_maps_to_zarr(
    result: GradCamResult,
    store_path: str | Path,
    overwrite: bool = True,
) -> Path:
    """Persist the per-pixel maps in a zarr store.

    Grad-CAM maps are per-pixel arrays on a real pixel grid, and this repo
    keeps every such array in zarr rather than in image or ``.npz`` files, so
    that a later reader gets chunked access and the array's own metadata
    instead of a filename convention.

    Layout: one ``grad_cam`` array of shape ``(n_classes, n_tiles, H, W)``
    chunked one tile at a time, plus a ``tile_has_class`` mask and the class
    names and target layer as attributes.
    """
    import zarr

    path = Path(store_path)
    if overwrite and path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="a")
    n_classes, n_tiles, height, width = result.maps.shape
    store.create_dataset(
        "grad_cam",
        data=result.maps.astype(np.float32),
        chunks=(1, 1, height, width),
        dtype="f4",
        overwrite=True,
    )
    store.create_dataset(
        "tile_has_class",
        data=result.tile_has_class.astype(np.uint8),
        chunks=(n_classes, n_tiles),
        dtype="u1",
        overwrite=True,
    )
    store.attrs["class_names"] = list(result.class_names)
    store.attrs["target_layer"] = result.target_layer
    store.attrs["n_tiles"] = int(result.n_tiles)
    store.attrs["normalisation"] = "per-tile min-max to [0, 1]"
    return path


def read_maps_from_zarr(store_path: str | Path) -> Optional[GradCamResult]:
    """Read back what :func:`write_maps_to_zarr` wrote, or None if absent."""
    import zarr

    path = Path(store_path)
    if not path.exists():
        return None
    store = zarr.open_group(str(path), mode="r")
    return GradCamResult(
        class_names=tuple(store.attrs.get("class_names", ())),
        maps=np.asarray(store["grad_cam"]),
        tile_has_class=np.asarray(store["tile_has_class"]).astype(bool),
        target_layer=str(store.attrs.get("target_layer", "")),
        n_tiles=int(store.attrs.get("n_tiles", 0)),
    )
