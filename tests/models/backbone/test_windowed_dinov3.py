# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the windowed-attention DINOv3 ViT backbone prototype.

The keystone test is :func:`TestStockEquivalence.test_num_windows_1_matches_stock_dinov3`:
with ``num_windows=1`` the windowed backbone must reduce *exactly* to the upstream
``transformers`` ``DINOv3ViTBackbone``, proving the block port, RoPE handling, and weight
layout are all faithful. The remaining tests cover the RoPE window-fold, local/global
window routing, shapes, and the divisibility constraint.
"""

import pytest
import torch
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTBackbone,
    DINOv3ViTRopePositionEmbedding,
)

from rfdetr.models.backbone.dinov3_with_windowed_attn import (
    WindowedDinov3ViTBackbone,
    WindowedDinov3ViTConfig,
    fold_to_windows,
)

# Shared small-model hyperparameters. eager attention keeps the stock-equivalence
# comparison bitwise-deterministic (SDPA and eager are not bit-identical).
_BASE = dict(
    hidden_size=32,
    num_hidden_layers=6,
    num_attention_heads=4,
    intermediate_size=64,
    num_register_tokens=2,
    patch_size=16,
    image_size=64,
    hidden_act="gelu",
    attention_dropout=0.0,
    drop_path_rate=0.0,
    _attn_implementation="eager",
)


def _base(**overrides) -> dict:
    return {**_BASE, **overrides}


def _windowed(out_features, num_windows, window_block_indexes, **overrides) -> WindowedDinov3ViTBackbone:
    config = WindowedDinov3ViTConfig(
        out_features=out_features,
        num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        **_base(**overrides),
    )
    return WindowedDinov3ViTBackbone(config).eval()


class TestStockEquivalence:
    """With ``num_windows=1`` the backbone must equal upstream ``DINOv3ViTBackbone``."""

    def test_pretrained_state_dict_loads_cleanly(self):
        """A stock DINOv3 ViT state_dict loads with no missing/unexpected keys.

        RoPE's ``inv_freq`` is a non-persistent buffer absent from both state dicts, so a
        clean load confirms the parameter layout matches the pretrained checkpoints.
        """
        stock = DINOv3ViTBackbone(DINOv3ViTConfig(out_features=["stage3", "stage6"], **_base()))
        windowed = _windowed(["stage3", "stage6"], num_windows=1, window_block_indexes=list(range(6)))
        missing, unexpected = windowed.load_state_dict(stock.state_dict(), strict=False)
        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"

    def test_num_windows_1_matches_stock_dinov3(self):
        """Feature maps are bit-identical (fp32 tolerance) to upstream at ``num_windows=1``."""
        stock = DINOv3ViTBackbone(DINOv3ViTConfig(out_features=["stage3", "stage6"], **_base())).eval()
        windowed = _windowed(["stage3", "stage6"], num_windows=1, window_block_indexes=list(range(6)))
        windowed.load_state_dict(stock.state_dict(), strict=False)

        pixel_values = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            expected = stock(pixel_values).feature_maps
            actual = windowed(pixel_values).feature_maps

        assert len(actual) == len(expected) == 2
        for stage_expected, stage_actual in zip(expected, actual):
            assert torch.allclose(stage_expected, stage_actual, atol=1e-5)


class TestRopeFold:
    """The folded per-window RoPE table must carry each window's true global coords."""

    def test_window_zero_equals_global_slice(self):
        config = WindowedDinov3ViTConfig(
            out_features=["stage6"], num_windows=2, window_block_indexes=list(range(6)), **_base()
        )
        rope = DINOv3ViTRopePositionEmbedding(config)
        pixel_values = torch.randn(1, 3, 64, 64)  # 4x4 patch grid, 2x2 windows
        cos_global, _ = rope(pixel_values)  # (16, head_dim)
        cos_folded = fold_to_windows(cos_global[None], 4, 4, 2)  # (4, 4, head_dim)

        # Window 0 = top-left 2x2 block: global flat patch indices 0, 1, 4, 5.
        assert torch.equal(cos_folded[0], cos_global[[0, 1, 4, 5]])


class TestWindowRouting:
    """Local layers isolate windows; global layers mix them."""

    def test_local_blocks_isolate_windows(self):
        """With every layer local, perturbing one window cannot change another's output."""
        windowed = _windowed(["stage6"], num_windows=2, window_block_indexes=list(range(6)))
        pixel_values = torch.randn(1, 3, 64, 64)
        perturbed = pixel_values.clone()
        perturbed[:, :, :32, :32] += 5.0  # top-left window only

        with torch.no_grad():
            base = windowed(pixel_values).feature_maps[0]
            after = windowed(perturbed).feature_maps[0]

        other_window_delta = (base[:, :, 2:, 2:] - after[:, :, 2:, 2:]).abs().max().item()
        perturbed_window_delta = (base[:, :, :2, :2] - after[:, :, :2, :2]).abs().max().item()
        assert other_window_delta < 1e-6, "local blocks leaked across windows"
        assert perturbed_window_delta > 1e-3, "perturbation had no effect"

    def test_global_blocks_mix_windows(self):
        """With every layer global, perturbing one window must change another's output."""
        windowed = _windowed(["stage6"], num_windows=2, window_block_indexes=[])
        pixel_values = torch.randn(1, 3, 64, 64)
        perturbed = pixel_values.clone()
        perturbed[:, :, :32, :32] += 5.0  # top-left window only

        with torch.no_grad():
            base = windowed(pixel_values).feature_maps[0]
            after = windowed(perturbed).feature_maps[0]

        other_window_delta = (base[:, :, 2:, 2:] - after[:, :, 2:, 2:]).abs().max().item()
        assert other_window_delta > 1e-3, "global blocks did not propagate across windows"


class TestShapes:
    """Feature-map shapes and the divisibility constraint."""

    @pytest.mark.parametrize(
        "height, width, num_windows",
        [
            pytest.param(64, 64, 2, id="square"),
            pytest.param(128, 128, 2, id="larger-square"),
            pytest.param(64, 96, 2, id="rectangular"),
            pytest.param(64, 64, 1, id="no-windowing"),
        ],
    )
    def test_feature_map_shapes(self, height: int, width: int, num_windows: int):
        windowed = _windowed(
            ["stage2", "stage4", "stage6"],
            num_windows=num_windows,
            window_block_indexes=[0, 1, 2, 3],
            image_size=max(height, width),
        )
        with torch.no_grad():
            feature_maps = windowed(torch.randn(2, 3, height, width)).feature_maps
        assert len(feature_maps) == 3
        for feature_map in feature_maps:
            assert tuple(feature_map.shape) == (2, 32, height // 16, width // 16)

    @pytest.mark.parametrize(
        "height, width, num_windows, should_raise",
        [
            pytest.param(64, 64, 2, False, id="valid-square"),
            pytest.param(64, 96, 2, False, id="valid-rectangular"),
            pytest.param(96, 96, 4, True, id="not-divisible-by-64"),
            pytest.param(80, 64, 2, True, id="height-not-divisible"),
        ],
    )
    def test_divisibility_constraint(self, height: int, width: int, num_windows: int, should_raise: bool):
        windowed = _windowed(
            ["stage6"],
            num_windows=num_windows,
            window_block_indexes=list(range(6)),
            image_size=max(height, width),
        )
        pixel_values = torch.randn(1, 3, height, width)
        if should_raise:
            with pytest.raises(ValueError, match="divisible"):
                windowed(pixel_values)
        else:
            with torch.no_grad():
                windowed(pixel_values)
