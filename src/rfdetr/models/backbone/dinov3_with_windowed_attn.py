# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from HuggingFace Transformers (https://github.com/huggingface/transformers)
# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved. (DINOv3)
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
"""DINOv3 ViT backbone with Swin-style windowed self-attention.

This module reuses RF-DETR's existing windowing scaffold (a batch-dimension window
fold, with most layers attending locally within a window and a few "output" layers
attending globally) but builds it on top of the ``transformers`` DINOv3 ViT blocks.

The key difference from the DINOv2 windowed backbone
([dinov2_with_windowed_attn.py][rfdetr.models.backbone.dinov2_with_windowed_attn]) is
positional encoding. DINOv2 adds a *learned absolute* position embedding once, before
the window fold, so windowing is position-agnostic data movement. DINOv3 has no learned
position table; its only positional signal is 2D-axial **RoPE applied inside every
attention layer**. We therefore use the "Option A" strategy:

1. Compute the stock DINOv3 global ``cos``/``sin`` table once, then fold it into
   window-major order with the *same* reshape applied to the patch tokens, so each
   window's patches carry their **true global coordinates** (see [`fold_to_windows`][]).
2. Apply RoPE per-window while tokens are still in the folded ``(B*nw**2, ...)`` layout.
   Because rotation happens before any window merge, the stock trailing-slice rotary
   apply is correct per window and only a *single* folded table is needed.
3. For global ("full attention") layers, merge ``q``/``k``/``v`` across windows
   **after** rotation, run one dense attention, then split back to the folded layout.

RoPE is translation-invariant for relative attention, so a window keeping its true
global spacing yields identical local-attention scores; and because every patch is
rotated by its global coordinate before the global merge, cross-window attention in the
output layers is also correct. With ``num_windows == 1`` this module reduces exactly to
``transformers``' ``DINOv3ViTBackbone`` (verified to fp32 tolerance in the tests).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from transformers.backbone_utils import BackboneMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BackboneOutput
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    ALL_ATTENTION_FUNCTIONS,
    DINOv3ViTAttention,
    DINOv3ViTEmbeddings,
    DINOv3ViTGatedMLP,
    DINOv3ViTLayerScale,
    DINOv3ViTMLP,
    DINOv3ViTPreTrainedModel,
    DINOv3ViTRopePositionEmbedding,
    eager_attention_forward,
    rotate_half,
)

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Apply Stochastic Depth (drop paths) per sample on the residual branch.

    Vendored from ``transformers`` to avoid depending on the DINOv3 ViT ``DropPath``
    class, whose name varies across supported ``transformers`` versions
    (``DINOv3ViTDropPath`` vs ``Dinov3ViTDropPath``).

    Args:
        input: Input tensor.
        drop_prob: Probability of dropping the path.
        training: Whether the module is in training mode.

    Returns:
        The (possibly scaled and masked) tensor; the input unchanged when
        ``drop_prob == 0`` or not training.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    return input.div(keep_prob) * random_tensor


class Dinov3DropPath(nn.Module):
    """Stochastic Depth module wrapping [`drop_path`][]."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


@strict
class WindowedDinov3ViTConfig(DINOv3ViTConfig):
    """DINOv3 ViT configuration extended with Swin-style windowed-attention fields.

    Inherits all DINOv3 architecture / RoPE fields (``rope_theta``, ``patch_size``,
    ``num_register_tokens``, ``use_gated_mlp``, asymmetric ``key_bias=False``, ...).

    Attributes:
        num_windows: Number of windows per spatial axis (the image is split into
            ``num_windows ** 2`` windows). ``1`` disables windowing.
        window_block_indexes: Indices of layers that attend *locally* (within a window).
            Layers absent from this list run global attention. Defaults to every layer
            (fully local) when ``None``.
        gradient_checkpointing: Whether transformer layers use gradient checkpointing.
    """

    model_type = "windowed_dinov3_vit"

    num_windows: int = 1
    window_block_indexes: list[int] | None = None
    gradient_checkpointing: bool = False

    def __post_init__(self, **kwargs) -> None:
        super().__post_init__(**kwargs)
        if self.window_block_indexes is None:
            self.window_block_indexes = list(range(self.num_hidden_layers))


def fold_to_windows(tensor: torch.Tensor, num_h: int, num_w: int, num_windows: int) -> torch.Tensor:
    """Fold a flat spatial sequence into window-major windows along the batch axis.

    This is the identical reshape/permute the DINOv2 windowed backbone applies to its
    patch tokens, so it works for patch embeddings ``(n, num_h * num_w, dim)`` and for
    RoPE ``cos``/``sin`` tables ``(1, num_h * num_w, dim)`` alike.

    Args:
        tensor: Tensor of shape ``(n, num_h * num_w, dim)`` in row-major (h-major) order.
        num_h: Number of patches along the height axis.
        num_w: Number of patches along the width axis.
        num_windows: Number of windows per axis.

    Returns:
        Tensor of shape ``(n * num_windows ** 2, (num_h // num_windows) *
        (num_w // num_windows), dim)``. The leading axis is window-major: index
        ``image * num_windows ** 2 + window_row * num_windows + window_col``.
    """
    n, _, dim = tensor.shape
    nw = num_windows
    h_per_window, w_per_window = num_h // nw, num_w // nw
    tensor = tensor.view(n, num_h, num_w, dim)
    tensor = tensor.reshape(n * nw, h_per_window, nw, w_per_window, dim).permute(0, 2, 1, 3, 4)
    tensor = tensor.reshape(n * nw * nw, h_per_window * w_per_window, dim)
    return tensor


def apply_windowed_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_windows_squared: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to patch tokens only, using per-window ``cos``/``sin`` in folded layout.

    Prefix tokens (CLS + register) are left unrotated, matching the trailing-slice
    convention of ``transformers``' ``apply_rotary_pos_emb``. The per-window tables carry
    each window's true global coordinates, so rotating here (before any window merge)
    is correct for both local and subsequently-merged global attention.

    Args:
        query: Query tensor of shape ``(N, heads, seq, head_dim)`` with
            ``N = batch * num_windows ** 2`` and ``seq = num_prefix + tokens_per_window``.
        key: Key tensor of the same shape as ``query``.
        cos: Cosine table of shape ``(num_windows ** 2, tokens_per_window, head_dim)``.
        sin: Sine table of the same shape as ``cos``.
        num_windows_squared: ``num_windows ** 2``.

    Returns:
        The rotated ``(query, key)`` tensors, same shapes as the inputs.
    """
    tokens_per_window = cos.shape[-2]
    n, heads, seq, head_dim = query.shape
    nw2 = num_windows_squared
    batch = n // nw2
    num_prefix = seq - tokens_per_window

    query_prefix, query_patches = query.split((num_prefix, tokens_per_window), dim=-2)
    key_prefix, key_patches = key.split((num_prefix, tokens_per_window), dim=-2)

    # (N, heads, Tpw, hd) -> (B, nw2, heads, Tpw, hd); cos/sin -> (1, nw2, 1, Tpw, hd)
    query_patches = query_patches.view(batch, nw2, heads, tokens_per_window, head_dim)
    key_patches = key_patches.view(batch, nw2, heads, tokens_per_window, head_dim)
    cos = cos.view(1, nw2, 1, tokens_per_window, head_dim)
    sin = sin.view(1, nw2, 1, tokens_per_window, head_dim)

    query_patches = (query_patches * cos) + (rotate_half(query_patches) * sin)
    key_patches = (key_patches * cos) + (rotate_half(key_patches) * sin)
    query_patches = query_patches.reshape(n, heads, tokens_per_window, head_dim)
    key_patches = key_patches.reshape(n, heads, tokens_per_window, head_dim)

    query = torch.cat((query_prefix, query_patches), dim=-2)
    key = torch.cat((key_prefix, key_patches), dim=-2)
    return query, key


class WindowedDinov3RopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    """Computes the stock DINOv3 global ``cos``/``sin`` table, then folds it window-major.

    Reusing the parent ``forward`` keeps the exact stock numerics (float32 angle
    computation, training-time coordinate augmentation) and then folds the result so each
    window's rows carry that window's patches' true global coordinates.
    """

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = super().forward(pixel_values)  # (num_patches, head_dim) each
        num_windows = self.config.num_windows
        cos, sin = cos[None], sin[None]  # (1, num_patches, head_dim)
        if num_windows == 1:
            return cos, sin
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        num_h, num_w = height // patch_size, width // patch_size
        cos = fold_to_windows(cos, num_h, num_w, num_windows)
        sin = fold_to_windows(sin, num_h, num_w, num_windows)
        return cos, sin


class WindowedDinov3ViTEmbeddings(DINOv3ViTEmbeddings):
    """DINOv3 patch / CLS / register embeddings followed by the window fold.

    There is no learned position table (RoPE replaces it). When ``num_windows > 1`` the
    patch tokens are folded into the batch axis and a CLS + register prefix is prepended
    per window, producing ``(batch * num_windows ** 2, num_register + 1 + tokens_per_window,
    hidden_size)``.
    """

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        num_windows = self.config.num_windows
        patch_size = self.config.patch_size
        target_dtype = self.patch_embeddings.weight.dtype
        patches = self.patch_embeddings(pixel_values.to(target_dtype))
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, C)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patches.dtype)
            patches = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patches)

        _, _, height, width = pixel_values.shape
        num_h, num_w = height // patch_size, width // patch_size
        if num_windows > 1:
            patches = fold_to_windows(patches, num_h, num_w, num_windows)

        n = patches.shape[0]
        cls_token = self.cls_token.expand(n, -1, -1)
        register_tokens = self.register_tokens.expand(n, -1, -1)
        return torch.cat([cls_token, register_tokens, patches], dim=1)


class WindowedDinov3ViTAttention(DINOv3ViTAttention):
    """DINOv3 multi-head attention with windowed RoPE and an optional global merge.

    RoPE is applied per-window while tokens are folded. For global ("full attention")
    layers the rotated ``q``/``k``/``v`` are merged across windows into a single sequence,
    attended densely, then split back to the folded layout so residual shapes are
    preserved throughout the layer.
    """

    @staticmethod
    def _merge_windows(tensor: torch.Tensor, batch: int, nw2: int) -> torch.Tensor:
        # (batch * nw2, heads, seq, hd) -> (batch, heads, nw2 * seq, hd)
        _, heads, seq, head_dim = tensor.shape
        return (
            tensor.view(batch, nw2, heads, seq, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, heads, nw2 * seq, head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        run_full_attention: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Run windowed (or globally-merged) attention.

        Args:
            hidden_states: Folded tokens of shape ``(N, seq, hidden_size)`` where
                ``N = batch * num_windows ** 2``.
            position_embeddings: ``(cos, sin)`` per-window RoPE tables.
            run_full_attention: If ``True``, merge windows for a global attention pass.
            **kwargs: Forwarded to the selected attention interface.

        Returns:
            A ``(attention_output, None)`` tuple; the output has the folded input shape.
        """
        cos, sin = position_embeddings
        n, seq, _ = hidden_states.shape
        nw2 = self.config.num_windows**2

        query = self.q_proj(hidden_states).view(n, seq, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(n, seq, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(n, seq, self.num_heads, self.head_dim).transpose(1, 2)

        query, key = apply_windowed_rotary(query, key, cos, sin, nw2)

        if run_full_attention:
            batch = n // nw2
            query = self._merge_windows(query, batch, nw2)
            key = self._merge_windows(key, batch, nw2)
            value = self._merge_windows(value, batch, nw2)
            eff_batch, eff_seq = batch, nw2 * seq
        else:
            eff_batch, eff_seq = n, seq

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            self,
            query,
            key,
            value,
            None,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(eff_batch, eff_seq, -1)
        if run_full_attention:
            attn_output = attn_output.view(eff_batch, nw2, seq, -1).reshape(n, seq, -1)
        return self.o_proj(attn_output), None


class WindowedDinov3ViTLayer(GradientCheckpointingLayer):
    """A DINOv3 transformer block whose attention is window-aware.

    Identical to ``transformers``' ``DINOv3ViTLayer`` (same submodule names, so stock
    weights load 1:1) except the attention call receives ``run_full_attention``.
    """

    def __init__(self, config: WindowedDinov3ViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = WindowedDinov3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = Dinov3DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTGatedMLP(config) if config.use_gated_mlp else DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        run_full_attention: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Apply attention and MLP sub-blocks with residual connections.

        Args:
            hidden_states: Folded tokens ``(N, seq, hidden_size)``.
            position_embeddings: ``(cos, sin)`` per-window RoPE tables.
            run_full_attention: Whether this layer attends globally across windows.
            **kwargs: Forwarded to the attention module.

        Returns:
            Updated hidden states with the same shape as the input.
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            position_embeddings=position_embeddings,
            run_full_attention=run_full_attention,
            **kwargs,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        return self.drop_path(hidden_states) + residual


class WindowedDinov3ViTEncoder(nn.Module):
    """Stack of windowed DINOv3 layers routing each layer to local or global attention."""

    def __init__(self, config: WindowedDinov3ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([WindowedDinov3ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Run the encoder, collecting per-stage hidden states.

        Args:
            hidden_states: Folded embedding output ``(N, seq, hidden_size)``.
            position_embeddings: ``(cos, sin)`` per-window RoPE tables.

        Returns:
            A tuple of hidden states aligned with ``stage_names`` (``stem`` first, then one
            entry per layer). Computation stops after the last requested output stage.
        """
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        last_out_index = int(self.config._out_features[-1][len("stage") :])
        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            if i > last_out_index:
                break
            run_full_attention = i not in self.config.window_block_indexes
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
                run_full_attention=run_full_attention,
            )
        all_hidden_states = all_hidden_states + (hidden_states,)
        return all_hidden_states


class WindowedDinov3ViTBackbone(BackboneMixin, DINOv3ViTPreTrainedModel):
    """DINOv3 ViT backbone with windowed attention, for use with detectors like RF-DETR.

    Submodule names mirror ``transformers``' ``DINOv3ViTBackbone`` (``embeddings``,
    ``rope_embeddings``, ``model``, ``norm``) so pretrained ``facebook/dinov3-*`` weights
    load directly via ``from_pretrained`` / ``load_state_dict`` (RoPE has no learned
    parameters). Feature maps are produced for the configured ``out_features`` stages.
    """

    config_class = WindowedDinov3ViTConfig

    def __init__(self, config: WindowedDinov3ViTConfig) -> None:
        super().__init__(config)
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = WindowedDinov3ViTEmbeddings(config)
        self.rope_embeddings = WindowedDinov3RopePositionEmbedding(config)
        self.model = WindowedDinov3ViTEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_register_tokens = config.num_register_tokens
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BackboneOutput:
        """Extract multi-scale feature maps from an image batch.

        Args:
            pixel_values: Image tensor ``(batch, num_channels, height, width)``. Both
                ``height`` and ``width`` must be divisible by ``patch_size * num_windows``.
            **kwargs: Accepted for interface compatibility; ignored.

        Returns:
            A ``BackboneOutput`` whose ``feature_maps`` holds one ``(batch, hidden_size,
            height // patch_size, width // patch_size)`` tensor per configured output stage.

        Raises:
            ValueError: If ``height`` or ``width`` is not divisible by
                ``patch_size * num_windows``.
        """
        num_windows, patch_size = self.config.num_windows, self.config.patch_size
        batch, _, height, width = pixel_values.shape
        divisor = patch_size * num_windows
        if height % divisor or width % divisor:
            raise ValueError(
                f"Input height and width must be divisible by patch_size * num_windows "
                f"({patch_size} * {num_windows} = {divisor}), but got {(height, width)}."
            )
        num_h, num_w = height // patch_size, width // patch_size

        embeddings = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)
        hidden_states = self.model(embeddings, position_embeddings)

        feature_maps: list[torch.Tensor] = []
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage not in self.out_features:
                continue
            if self.config.apply_layernorm:
                hidden_state = self.norm(hidden_state)
            if self.config.reshape_hidden_states:
                hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                if num_windows > 1:
                    nw2 = num_windows * num_windows
                    batch_windows, tokens_per_window, channels = hidden_state.shape
                    h_per_window, w_per_window = num_h // num_windows, num_w // num_windows
                    hidden_state = hidden_state.reshape(batch_windows // nw2, nw2 * tokens_per_window, channels)
                    hidden_state = hidden_state.reshape(
                        (batch_windows // nw2) * num_windows,
                        num_windows,
                        h_per_window,
                        w_per_window,
                        channels,
                    ).permute(0, 2, 1, 3, 4)
                hidden_state = hidden_state.reshape(batch, num_h, num_w, -1)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
            feature_maps.append(hidden_state)

        return BackboneOutput(feature_maps=tuple(feature_maps))


__all__ = [
    "WindowedDinov3ViTConfig",
    "WindowedDinov3ViTBackbone",
    "fold_to_windows",
    "apply_windowed_rotary",
]
