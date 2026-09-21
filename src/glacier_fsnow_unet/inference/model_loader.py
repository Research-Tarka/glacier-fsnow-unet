"""Loading the trained U-Net for inference.

Purpose
-------
Bridge between the inference chain (stage 7) and the training module's model
definition (``glacier_fsnow_unet.training.model_architectures``).

Inputs
------
- A checkpoint path (``model.pt``) under ``config.paths.model_inference_root``.

Outputs
-------
- A ``torch.nn.Module`` in eval mode on the requested device.

Model interface
----------------
``glacier_fsnow_unet.training.model_architectures.build_model`` constructs a
four-level attention U-Net (channel progression 48 -> 96 -> 192 -> 384 at the
reference ``base_ch=48``) with attention-gated skip connections. The forward
pass takes ``(N, 11, H, W)`` float32 patches of the 11 spectral indices and
returns ``(N, 4, H, W)`` logits, ordered by the paper's training labels:
Cloud 0, Snow 1, Ice 2, Other 3.

Checkpoints are accepted either as a raw ``state_dict`` or as a dict with a
``"model"`` / ``"state_dict"`` / ``"model_state_dict"`` key plus metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

#: Number of input channels: the 11 spectral indices (paper Table 2).
DEFAULT_IN_CHANNELS = 11
#: Number of output classes: Cloud, Snow, Ice, Other.
DEFAULT_NUM_CLASSES = 4
#: First-level channel count (paper Section 4.2).
DEFAULT_BASE_CHANNELS = 48

_MISSING_TRAINING_MODULE_MESSAGE = (
    "glacier_fsnow_unet.training.model_architectures could not be imported.\n"
    "Install the training extras (see requirements-torch.txt) and re-run this "
    "stage."
)


class ModelNotAvailableError(NotImplementedError):
    """Raised when the training module or a checkpoint is unavailable."""


def build_model(
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    dropout_p: float = 0.0,
) -> Any:
    """Construct an untrained U-Net via the training module's architecture.

    Raises
    ------
    ModelNotAvailableError
        If ``training.model_architectures`` cannot be imported.
    """
    try:
        from glacier_fsnow_unet.training.model_architectures import build_model as _build
    except ImportError as exc:
        raise ModelNotAvailableError(_MISSING_TRAINING_MODULE_MESSAGE) from exc

    return _build(
        model_type="unet",
        in_channels=in_channels,
        num_classes=num_classes,
        base_ch=base_channels,
        dropout_p=dropout_p,
    )


def find_checkpoint(model_root: str | Path, name: str = "model.pt") -> Path:
    """Locate the inference checkpoint under ``model_root``.

    Looks for ``<model_root>/<name>``, then ``<model_root>/ensemble/<name>``
    (the multi-seed ensemble produced by the bootstrap protocol), then any
    ``seed*/`` subdirectory.
    """
    root = Path(model_root)
    candidates = [root / name, root / "ensemble" / name, *sorted(root.glob(f"seed*/{name}"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise ModelNotAvailableError(
        f"No trained model checkpoint found under '{root}'.\n"
        f"Looked for: {', '.join(str(c) for c in candidates[:3])}"
    )


def load_model(
    checkpoint_path: Optional[str | Path] = None,
    model_root: Optional[str | Path] = None,
    device: str = "cuda",
    base_channels: int = DEFAULT_BASE_CHANNELS,
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Any:
    """Load the trained U-Net, ready for inference.

    Falls back to CPU with a warning when CUDA is requested but unavailable.

    Raises
    ------
    ModelNotAvailableError
        If no checkpoint exists at the given path.
    """
    try:
        import torch
    except ImportError as exc:
        raise ModelNotAvailableError(
            "PyTorch is not installed; it is required for inference.\n"
            "Install it with: pip install -r requirements-torch.txt"
        ) from exc

    if checkpoint_path is None:
        if model_root is None:
            raise ValueError("Provide either checkpoint_path or model_root.")
        checkpoint_path = find_checkpoint(model_root)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ModelNotAvailableError(f"Checkpoint not found: {checkpoint_path}")

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    model = build_model(
        in_channels=in_channels, num_classes=num_classes, base_channels=base_channels
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
