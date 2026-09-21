"""U-Net with attention-gated skip connections for four-class glacier
surface segmentation.

The network is a four-level U-Net over 48x48 patches of 11 normalised spectral
indices, producing per-pixel logits for Cloud / Snow / Ice / Other. Several
optional mechanisms sit on top of the plain architecture; every one of them is
off in the reference configuration, which is what the published checkpoint's
weights load into, but all are kept fully implemented:

*Per-sensor Multi-BatchNorm.* Convolution weights are shared across all five
sensors while each sensor gets its own BatchNorm (independent affine
parameters and running statistics). The intent is to let one network learn
common spatial features while normalising each sensor's radiometric
distribution separately. Enabled by `num_sensors > 1`.

*Spatial context gating.* A FiLM-style modulation of the first encoder stage,
conditioned on (x, y, year, area) and optionally a learned sensor embedding.
Enabled by `use_spatial_context`.

*GroupNorm in place of BatchNorm.* `norm_type="group"` swaps every
normalisation layer. BatchNorm's statistics are a property of the batch, and a
batch here mixes sensors and glaciers; its running mean and variance therefore
end up describing whatever composition dominated training, which is not what an
under-represented sensor sees at eval time. GroupNorm normalises within each
sample and keeps no running statistics at all, so train-time and eval-time
behaviour are identical regardless of what the batch contained. Unsupported
together with `num_sensors > 1` — see `_check_norm_and_sensors`.

*Bottleneck self-attention.* `bottleneck_attention` inserts multi-head
self-attention over the bottleneck's spatial grid, giving every position direct
access to every other rather than only to its convolutional neighbourhood. See
`BottleneckAttention` for why this is cheap at this depth specifically.

*Deep supervision.* `deep_supervision` attaches an auxiliary logit head to each
intermediate decoder level, so those levels receive gradient from their own loss
term rather than only through the levels above them. The heads are training
machinery: the forward pass returns the same final logits with or without them
unless a caller passes `return_aux`.

Two details differ from the architecture description in the accompanying
paper, and are correct as written here — the trained checkpoints have the
corresponding parameters, so changing either would make them unloadable:

- Upsampling is a learned `ConvTranspose2d`, not bilinear interpolation.
- Every skip connection passes through an additive attention gate before
  concatenation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_SENSORS, SENSOR_TO_IDX

__all__ = [
    "SpatialContextGating",
    "DoubleConv",
    "AttentionGate",
    "BottleneckAttention",
    "UNet",
    "build_model",
    "make_norm",
    "SENSOR_TO_IDX",
    "NUM_SENSORS",
    "GROUP_NORM_CHANNELS_PER_GROUP",
]

#: Channels per group when `norm_type="group"`. GroupNorm is parameterised by a
#: group *count*, but a fixed count divides the four channel widths here
#: (48/96/192/384) badly — the usual default of 32 does not divide 48 at all.
#: Fixing the group *size* instead and deriving the count gives 3/6/12/24
#: groups, which divides every width cleanly and keeps the statistic computed
#: over the same number of channels at every depth. Sixteen is small enough
#: that each group's mean and variance are over 16 x H x W values (4608 at the
#: first level's 48x48, 576 at the 6x6 bottleneck) — enough samples to be a
#: stable estimate, which is the property that makes GroupNorm batch-size
#: independent in the first place.
GROUP_NORM_CHANNELS_PER_GROUP: int = 16


def make_norm(channels: int, norm_type: str = "batch") -> nn.Module:
    """One normalisation layer for `channels` channels.

    `"batch"` is the reference setting and matches the published checkpoint's
    parameters. `"group"` substitutes GroupNorm, which normalises over channel
    groups within each sample and so does not depend on batch composition at
    all — no running mean or variance is kept, and eval-time behaviour is
    identical to train-time behaviour.

    A channel count not divisible by `GROUP_NORM_CHANNELS_PER_GROUP` falls back
    to the largest divisor that is no larger than it, so an unusual `base_ch`
    still builds rather than raising. The single-channel attention-gate output
    lands on one group, which is LayerNorm over that map — the only sensible
    reading of "group-normalise one channel".
    """
    kind = str(norm_type).strip().lower()
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        size = min(GROUP_NORM_CHANNELS_PER_GROUP, channels)
        while size > 1 and channels % size != 0:
            size -= 1
        return nn.GroupNorm(max(1, channels // size), channels)
    raise ValueError(f"unknown norm_type {norm_type!r} (expected 'batch' or 'group')")


def _route_by_sensor(
    x: torch.Tensor,
    sensor_ids: Optional[torch.Tensor],
    banks: nn.ModuleList,
) -> torch.Tensor:
    """Apply each batch item's own normalisation layer from `banks`.

    With a single bank, or with no sensor ids supplied, this is just
    `banks[0](x)` and costs nothing. Otherwise each sensor's slice of the batch
    goes through its own layer.

    Only sensors actually present in the batch are visited, so a batch drawn
    from one sensor costs one BatchNorm call rather than `len(banks)`.
    """
    if len(banks) == 1 or sensor_ids is None:
        return banks[0](x)

    out = torch.empty_like(x)
    # unique() bounds the loop by the sensors present, not by the bank size.
    for sensor in torch.unique(sensor_ids).tolist():
        index = int(sensor)
        if not 0 <= index < len(banks):
            index = 0
        mask = sensor_ids == sensor
        out[mask] = banks[index](x[mask])
    return out


class SpatialContextGating(nn.Module):
    """FiLM modulation of a feature map from spatio-temporal context.

    Produces a per-channel scale and shift from a context vector
    `(x, y, year_norm, area_norm)`, optionally concatenated with a learned
    sensor embedding, and applies `out = x * scale + shift`. The scale is
    passed through a sigmoid, so it stays in [0, 1] and cannot blow up early
    in training.

    Applied only to the first encoder stage. Cloud appearance does not depend
    on geography or year, so gating deeper stages would inject a spurious
    location prior into the one class that should not have it.
    """

    def __init__(
        self,
        feat_channels: int,
        context_dim: int = 4,
        num_sensors: int = 1,
        use_sensor_film: bool = False,
    ) -> None:
        super().__init__()
        self.num_sensors = max(1, int(num_sensors))
        self.use_sensor_film = bool(use_sensor_film and self.num_sensors > 1)

        sensor_embed_dim = 16 if self.use_sensor_film else 0
        self.sensor_embedding = (
            nn.Embedding(self.num_sensors, sensor_embed_dim)
            if self.use_sensor_film
            else None
        )

        self.context_embedding = nn.Sequential(
            nn.Linear(context_dim + sensor_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_channels * 2),  # scale and shift, concatenated
        )

        # Retained so a checkpoint's parameter set is stable across versions.
        self.register_parameter("scale_init", nn.Parameter(torch.ones(feat_channels)))
        self.register_parameter("shift_init", nn.Parameter(torch.zeros(feat_channels)))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        sensor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        channels = x.shape[1]

        if self.use_sensor_film and self.sensor_embedding is not None and sensor_ids is not None:
            ids = sensor_ids.long().clamp(0, self.num_sensors - 1)
            context = torch.cat([context, self.sensor_embedding(ids)], dim=1)

        embedded = self.context_embedding(context)
        scale = embedded[:, :channels].sigmoid().unsqueeze(-1).unsqueeze(-1)
        shift = embedded[:, channels:].unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift


def _check_norm_and_sensors(norm_type: str, num_sensors: int) -> None:
    """Reject the one combination that cannot mean anything.

    The per-sensor bank exists so each sensor accumulates its own running mean
    and variance over its own radiometric distribution. GroupNorm keeps no
    running statistics — it normalises within each sample at both train and
    eval time — so a bank of GroupNorms would differ only in their affine
    parameters, which is a per-sensor rescaling wearing a normalisation's name.
    That is a defensible thing to want, but it is not what the caller asked for
    and it would silently score differently from what they expect, so it raises
    rather than quietly doing something adjacent.
    """
    if str(norm_type).strip().lower() == "group" and int(num_sensors) > 1:
        raise ValueError(
            "norm_type='group' with num_sensors>1 is unsupported: GroupNorm "
            "keeps no running statistics, so there is nothing for a per-sensor "
            "bank to accumulate. Use norm_type='batch' for per-sensor "
            "normalisation, or num_sensors=1 with GroupNorm."
        )


class DoubleConv(nn.Module):
    """Conv-Norm-ReLU twice, with optional per-sensor normalisation banks.

    `num_sensors=1` gives an ordinary double convolution block.

    `norm_type="batch"` is the reference setting and what the published
    checkpoint's parameters are. `norm_type="group"` swaps in GroupNorm and is
    incompatible with `num_sensors > 1`; see `_check_norm_and_sensors`.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout_p: float = 0.0,
        num_sensors: int = 1,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        _check_norm_and_sensors(norm_type, num_sensors)
        self.num_sensors = int(num_sensors)
        self.norm_type = str(norm_type).strip().lower()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.ModuleList(
            make_norm(out_ch, norm_type) for _ in range(num_sensors)
        )
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.ModuleList(
            make_norm(out_ch, norm_type) for _ in range(num_sensors)
        )
        self.dropout = nn.Dropout2d(p=float(dropout_p)) if dropout_p > 0 else None

    def forward(
        self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.relu(_route_by_sensor(self.conv1(x), sensor_ids, self.bn1), inplace=True)
        x = F.relu(_route_by_sensor(self.conv2(x), sensor_ids, self.bn2), inplace=True)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class AttentionGate(nn.Module):
    """Additive attention gate for a skip connection.

    Computes `psi = sigmoid(W_psi . relu(W_g g + W_x x))`, a single-channel
    spatial attention map, and returns `x * psi`. The decoder's gating signal
    `g` therefore decides which parts of the encoder feature map `x` survive
    into the concatenation, rather than the whole map being passed through.
    """

    def __init__(
        self,
        f_gate: int,
        f_skip: int,
        f_inter: int,
        dropout_p: float = 0.2,
        num_sensors: int = 1,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        _check_norm_and_sensors(norm_type, num_sensors)
        self.num_sensors = int(num_sensors)
        self.norm_type = str(norm_type).strip().lower()

        def _drop() -> nn.Module:
            return nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        def _norms(channels: int) -> nn.ModuleList:
            return nn.ModuleList(
                make_norm(channels, norm_type) for _ in range(num_sensors)
            )

        self.W_g_conv = nn.Conv2d(f_gate, f_inter, kernel_size=1, bias=True)
        self.W_g_bn = _norms(f_inter)
        self.W_g_drop = _drop()

        self.W_x_conv = nn.Conv2d(f_skip, f_inter, kernel_size=1, bias=True)
        self.W_x_bn = _norms(f_inter)
        self.W_x_drop = _drop()

        self.psi_conv = nn.Conv2d(f_inter, 1, kernel_size=1, bias=True)
        # One channel: GroupNorm here is a LayerNorm over the attention map.
        self.psi_bn = _norms(1)
        self.psi_drop = _drop()

    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor,
        sensor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g1 = self.W_g_drop(_route_by_sensor(self.W_g_conv(g), sensor_ids, self.W_g_bn))
        x1 = self.W_x_drop(_route_by_sensor(self.W_x_conv(x), sensor_ids, self.W_x_bn))
        psi = F.relu(g1 + x1, inplace=True)
        psi = _route_by_sensor(self.psi_conv(psi), sensor_ids, self.psi_bn)
        return x * torch.sigmoid(self.psi_drop(psi))


class BottleneckAttention(nn.Module):
    """Multi-head self-attention over the bottleneck's spatial positions.

    The encoder's receptive field grows only by convolution and pooling, so at
    the bottleneck every position still summarises a bounded neighbourhood.
    Flattening the `H x W` bottleneck grid into a token sequence and running
    self-attention over it lets any position read any other directly, which is
    what a decision like "this bright patch sits in a glacier interior, not near
    the margin" needs and a stack of 3x3 convolutions can only approximate.

    Cheap at this depth, which is the reason it is worth trying at all. With the
    reference 48x48 patch and three poolings the bottleneck is 6x6 — 36 tokens.
    The attention matrix is 36x36 per head; the quadratic cost that rules
    self-attention out at full resolution is not a factor here.

    **Kept deliberately minimal.** One pre-norm attention block with a residual,
    and no feedforward sublayer. A transformer block's feedforward is where most
    of its parameters live, and over a 36-token grid whose channels have just
    passed through two 3x3 convolutions there is little reason to expect a
    position-wise MLP to add anything the surrounding convolutions cannot. The
    residual path means an untrained or unhelpful attention block degrades
    toward the identity rather than toward noise, so switching it on cannot make
    the architecture worse than not having it except through what training does
    with it.

    LayerNorm rather than the network's configured 2D norm: this operates on a
    token sequence, where LayerNorm over the channel dimension is what the
    attention formulation assumes, and it is batch-independent either way.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout_p: float = 0.0) -> None:
        super().__init__()
        heads = max(1, int(num_heads))
        # An indivisible head count would fail inside MultiheadAttention with a
        # message about embed_dim; fall back to the largest divisor instead, so
        # an unusual base_ch still builds.
        while heads > 1 and channels % heads != 0:
            heads -= 1
        self.num_heads = heads
        self.channels = int(channels)

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=float(dropout_p),
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, C, H, W)` in, the same shape out."""
        batch, channels, height, width = x.shape
        # (B, C, H, W) -> (B, H*W, C): one token per spatial position.
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm(tokens)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class UNet(nn.Module):
    """Four-level attention U-Net.

    Encoder channels progress `base_ch -> 2x -> 4x -> 8x` (48 -> 96 -> 192 ->
    384 at the reference `base_ch=48`), each level followed by 2x2 max-pooling.
    The decoder mirrors it with transposed convolutions, attention-gating each
    skip connection before concatenation. A 1x1 convolution produces the
    per-pixel class logits.

    Three options extend this, all off by default because the published
    checkpoint has none of them:

    *`norm_type="group"`* replaces every BatchNorm with a GroupNorm.

    *`bottleneck_attention`* inserts self-attention over the bottleneck grid
    between the last encoder stage and the first upsampling.

    *`deep_supervision`* attaches a 1x1 logit head to each intermediate decoder
    level. The heads exist only to receive their own loss term during training;
    the forward pass returns the same final logits either way unless
    `return_aux` asks for them, so nothing downstream of the model has to know
    whether deep supervision is on.

    *`use_attention_gates`* (default `True`, matching the published checkpoint)
    toggles the additive attention gate on every skip connection. Set to
    `False` for the ablation described in the paper: skip connections then
    carry the raw encoder feature map straight into the decoder's
    concatenation, exactly as if `AttentionGate.forward` had returned `x`
    unchanged. Off removes the three gates' parameters entirely rather than
    leaving them unused, so the ablation's "no attention gates" condition is
    not just gated to the identity but genuinely absent from the model.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 4,
        base_ch: int = 48,
        dropout_p: float = 0.0,
        use_spatial_context: bool = False,
        num_sensors: int = 1,
        use_sensor_film: bool = False,
        norm_type: str = "batch",
        bottleneck_attention: bool = False,
        bottleneck_attention_heads: int = 8,
        deep_supervision: bool = False,
        use_attention_gates: bool = True,
    ) -> None:
        super().__init__()
        _check_norm_and_sensors(norm_type, num_sensors)
        self.num_sensors = int(num_sensors)
        self.use_sensor_film = bool(use_sensor_film and num_sensors > 1)
        self.use_spatial_context = bool(use_spatial_context)
        self.norm_type = str(norm_type).strip().lower()
        self.deep_supervision = bool(deep_supervision)
        self.use_attention_gates = bool(use_attention_gates)

        block = lambda i, o: DoubleConv(
            i, o, dropout_p=dropout_p, num_sensors=num_sensors, norm_type=norm_type
        )

        self.enc1 = block(in_channels, base_ch)
        self.enc2 = block(base_ch, base_ch * 2)
        self.enc3 = block(base_ch * 2, base_ch * 4)
        self.enc4 = block(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck_attention = (
            BottleneckAttention(
                base_ch * 8, num_heads=bottleneck_attention_heads, dropout_p=dropout_p
            )
            if bottleneck_attention
            else None
        )

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = block(base_ch * 2, base_ch)

        if self.use_attention_gates:
            self.att3 = AttentionGate(
                base_ch * 4, base_ch * 4, base_ch * 2, dropout_p, num_sensors, norm_type
            )
            self.att2 = AttentionGate(
                base_ch * 2, base_ch * 2, base_ch, dropout_p, num_sensors, norm_type
            )
            self.att1 = AttentionGate(
                base_ch, base_ch, max(1, base_ch // 2), dropout_p, num_sensors, norm_type
            )
        else:
            self.att3 = self.att2 = self.att1 = None

        if use_spatial_context:
            self.context_gate = SpatialContextGating(
                base_ch,
                context_dim=4,
                num_sensors=num_sensors,
                use_sensor_film=self.use_sensor_film,
            )

        self.out_conv = nn.Conv2d(base_ch, num_classes, 1)

        # Auxiliary heads, coarsest first: dec3 at 1/4 resolution and dec2 at
        # 1/2. dec1 is the final level and already has `out_conv`, so giving it
        # a second head would only duplicate the primary loss.
        self.aux_heads = (
            nn.ModuleList(
                [
                    nn.Conv2d(base_ch * 4, num_classes, 1),
                    nn.Conv2d(base_ch * 2, num_classes, 1),
                ]
            )
            if deep_supervision
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        sensor_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Args:
            x: features, (B, C, H, W).
            context: (B, 4) as `[x, y, year_norm, area_norm]`, or (B, 5) with
                the sensor index appended as a float — in which case it is
                split off automatically when `sensor_ids` is not given.
            sensor_ids: (B,) int64 sensor indices. Takes precedence over a
                sensor index carried in `context`.
            return_aux: also return the deep-supervision heads' logits. Only
                the training loop asks for these; every other caller gets the
                same single tensor whether deep supervision is on or off, so
                evaluation, export and inference need no branch for it.

        Returns:
            Logits, (B, num_classes, H, W). With `return_aux`, a
            `(logits, [aux_coarse, aux_mid])` pair — the auxiliary list is
            empty when deep supervision is off, and each auxiliary tensor is at
            its own decoder level's resolution, not upsampled.
        """
        if sensor_ids is None and context is not None and context.shape[-1] >= 5:
            sensor_ids = context[:, 4].long()
            context = context[:, :4]

        e1 = self.enc1(x, sensor_ids=sensor_ids)
        if self.use_spatial_context and context is not None:
            e1 = self.context_gate(e1, context, sensor_ids=sensor_ids)

        e2 = self.enc2(self.pool(e1), sensor_ids=sensor_ids)
        e3 = self.enc3(self.pool(e2), sensor_ids=sensor_ids)
        e4 = self.enc4(self.pool(e3), sensor_ids=sensor_ids)

        if self.bottleneck_attention is not None:
            e4 = self.bottleneck_attention(e4)

        d3 = self.up3(e4)
        skip3 = self.att3(g=d3, x=e3, sensor_ids=sensor_ids) if self.att3 is not None else e3
        d3 = self.dec3(torch.cat([d3, skip3], dim=1), sensor_ids=sensor_ids)

        d2 = self.up2(d3)
        skip2 = self.att2(g=d2, x=e2, sensor_ids=sensor_ids) if self.att2 is not None else e2
        d2 = self.dec2(torch.cat([d2, skip2], dim=1), sensor_ids=sensor_ids)

        d1 = self.up1(d2)
        skip1 = self.att1(g=d1, x=e1, sensor_ids=sensor_ids) if self.att1 is not None else e1
        d1 = self.dec1(torch.cat([d1, skip1], dim=1), sensor_ids=sensor_ids)

        logits = self.out_conv(d1)
        if not return_aux:
            return logits

        aux = (
            [head(features) for head, features in zip(self.aux_heads, (d3, d2))]
            if self.aux_heads is not None
            else []
        )
        return logits, aux


def build_model(
    model_type: str,
    in_channels: int,
    num_classes: int,
    base_ch: int = 48,
    dropout_p: float = 0.0,
    use_spatial_context: bool = False,
    num_sensors: int = 1,
    use_sensor_film: bool = False,
    norm_type: str = "batch",
    bottleneck_attention: bool = False,
    bottleneck_attention_heads: int = 8,
    deep_supervision: bool = False,
    use_attention_gates: bool = True,
) -> nn.Module:
    """Construct a segmentation model by name.

    `num_sensors=1` (the reference setting) collapses the per-sensor
    normalisation banks to a single shared layer per block.

    The last five arguments default to the published checkpoint's architecture:
    BatchNorm, no bottleneck self-attention, no deep supervision, attention
    gates on every skip connection. Leaving them alone builds exactly the
    network the reference weights load into.
    """
    if model_type.lower() != "unet":
        raise ValueError(f"unknown model_type {model_type!r} (only 'unet' is supported)")

    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_ch=base_ch,
        dropout_p=dropout_p,
        use_spatial_context=use_spatial_context,
        num_sensors=num_sensors,
        use_sensor_film=use_sensor_film,
        norm_type=norm_type,
        bottleneck_attention=bottleneck_attention,
        bottleneck_attention_heads=bottleneck_attention_heads,
        deep_supervision=deep_supervision,
        use_attention_gates=use_attention_gates,
    )
