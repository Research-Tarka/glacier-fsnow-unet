"""DeepLabv3 (ResNet50 backbone) adapted for 11-channel, 48x48 glacier tiles.

Two adaptations from torchvision's stock `deeplabv3_resnet50`:

1. The first conv is replaced to accept 11 channels instead of 3. No
   ImageNet pretraining is used: the mismatched channel count and the
   non-photographic (normalised spectral index) input statistics make
   transfer weights not obviously useful here.
2. ASPP dilation rates are reduced from the default (12, 24, 36) to (2, 4,
   6): at patch_size=48 the ResNet50/output_stride=8 backbone produces only
   a 6x6 feature grid, on which the default high dilation rates would
   sample almost entirely zero-padding.

This is torchvision's `deeplabv3_resnet50` (ASPP head, no low-level-feature
decoder skip) rather than a literal DeepLabv3+ reproduction -- torchvision
does not ship one. Cite it as "a DeepLabv3-style ResNet50/ASPP model".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP


def _replace_aspp_rates(model: nn.Module, rates=(2, 4, 6)) -> None:
    head = model.classifier
    old_aspp = head[0]
    assert isinstance(old_aspp, ASPP), f"expected ASPP at classifier[0], got {type(old_aspp)}"
    in_channels = old_aspp.convs[0][0].in_channels  # 2048 for resnet50
    out_channels = old_aspp.convs[0][0].out_channels  # 256
    head[0] = ASPP(in_channels, list(rates), out_channels=out_channels)


def build_deeplabv3_glacier(
    in_channels: int = 11,
    num_classes: int = 4,
    aspp_rates=(2, 4, 6),
    dropout_p: float = 0.1,
) -> nn.Module:
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)

    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.backbone.conv1 = new_conv

    model.aux_classifier = None
    _replace_aspp_rates(model, rates=aspp_rates)

    # DeepLabHead is Sequential(ASPP, Conv2d(256,256,3,pad=1), BatchNorm2d,
    # ReLU, Conv2d(256, num_classes, 1)); splice dropout before the last conv.
    if dropout_p > 0:
        head = model.classifier
        last_conv = head[-1]
        head[-1] = nn.Sequential(nn.Dropout2d(p=dropout_p), last_conv)

    return model


if __name__ == "__main__":
    m = build_deeplabv3_glacier(dropout_p=0.2)
    x = torch.randn(2, 11, 48, 48)
    out = m(x)["out"]
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"total params: {n_params:,}")
