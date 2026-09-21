"""SegFormer adapted for 11-channel, 48x48 glacier tiles, built on
HuggingFace `transformers.SegformerForSemanticSegmentation` (Xie et al. 2021).

Scale check: the standard MiT encoder has 4 stages, strides (4, 2, 2, 2)
from the input, for a total downsampling of x32. Applied to a 48x48 input
that gives stage sizes 12x12, 6x6, 3x3, 2x2 -- the last stage collapses
almost immediately to a near-global pool, losing the hierarchical
multi-scale property the all-MLP decoder depends on.

`SegformerConfig` is therefore instantiated with 3 encoder stages instead
of 4 (`num_encoder_blocks=3`, strides `(4, 2, 2)`, total x16: 12x12, 6x6,
3x3), dropping the fourth stage rather than shrinking it to a degenerate
2x2. Hidden sizes/heads/sr_ratios are truncated to the first 3 entries of
the standard MiT-B0 schedule. Every other design choice (overlapping patch
embeddings, spatial-reduction-ratio attention, Mix-FFN, the all-MLP decode
head) is HuggingFace's standard implementation, unmodified.

`num_channels=11` and `num_labels=4` adapt the first patch-embedding conv
and the final classifier conv; weights are randomly initialised (no
ImageNet pretraining, for the same reason as DeepLab -- mismatched channel
count and non-photographic input statistics).

The model's raw `.logits` are at stage-1 resolution (12x12 for a 48x48
input); this wrapper bilinearly upsamples to full input resolution, to
match what DeepLab's `["out"]` already returns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerGlacier(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        num_classes: int = 4,
        hidden_sizes=(32, 64, 128),
        num_heads=(1, 2, 4),
        depths=(2, 2, 2),
        sr_ratios=(4, 2, 1),
        decoder_hidden_size: int = 128,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        n_stages = len(hidden_sizes)
        config = SegformerConfig(
            num_channels=in_channels,
            num_labels=num_classes,
            num_encoder_blocks=n_stages,
            depths=tuple(depths),
            hidden_sizes=tuple(hidden_sizes),
            patch_sizes=(7, 3, 3)[:n_stages],
            strides=(4, 2, 2)[:n_stages],
            sr_ratios=tuple(sr_ratios),
            num_attention_heads=tuple(num_heads),
            mlp_ratios=(4,) * n_stages,
            decoder_hidden_size=decoder_hidden_size,
            classifier_dropout_prob=dropout_p,
            hidden_dropout_prob=dropout_p,
            attention_probs_dropout_prob=dropout_p,
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_h, target_w = x.shape[-2], x.shape[-1]
        logits = self.model(pixel_values=x).logits  # (B, num_classes, H/4, W/4)
        return F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)


def build_segformer_glacier(
    in_channels: int = 11,
    num_classes: int = 4,
    hidden_sizes=(32, 64, 128),
    num_heads=(1, 2, 4),
    depths=(2, 2, 2),
    sr_ratios=(4, 2, 1),
    decoder_hidden_size: int = 128,
    dropout_p: float = 0.1,
) -> nn.Module:
    return SegFormerGlacier(
        in_channels=in_channels, num_classes=num_classes, hidden_sizes=hidden_sizes,
        num_heads=num_heads, depths=depths, sr_ratios=sr_ratios,
        decoder_hidden_size=decoder_hidden_size, dropout_p=dropout_p,
    )


if __name__ == "__main__":
    m = build_segformer_glacier(dropout_p=0.1)
    x = torch.randn(2, 11, 48, 48)
    out = m(x)
    print("output shape:", out.shape)
    assert out.shape == (2, 4, 48, 48), f"unexpected output shape {out.shape}"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"total params: {n_params:,}")
