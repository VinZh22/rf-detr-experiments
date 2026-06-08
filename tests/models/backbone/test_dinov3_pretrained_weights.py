# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Real-download tests for loading gated DINOv3 weights into the windowed backbone.

These hit the network and require a HuggingFace token with access to the gated
``facebook/dinov3-*`` repos (accept the DINOv3 License on the model page first). They are
skipped automatically unless a token is present in the environment, so they never run in
offline CI. To run locally::

    export HF_TOKEN=hf_...                       # token that has accepted the DINOv3 license
    uv run --no-sync pytest tests/models/backbone/test_dinov3_pretrained_weights.py -v -o addopts=""

Set ``RFDETR_DINOV3_HUB_SIZE`` (``small`` | ``base`` | ``large``) to choose the variant
(default ``small`` for speed).
"""

import os

import pytest
import torch

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

pytestmark = pytest.mark.skipif(
    not _HF_TOKEN,
    reason="no HF token in environment (HF_TOKEN / HUGGING_FACE_HUB_TOKEN); skipping gated DINOv3 download",
)

_SIZE = os.environ.get("RFDETR_DINOV3_HUB_SIZE", "small")
_HUB_NAMES = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
# Per-variant patch grid the model was pretrained at is irrelevant (RoPE is dynamic); we
# pick small backbone-agnostic inputs divisible by patch_size * num_windows.


def _out_feature_indexes(size: str) -> list[int]:
    # ViT-S/B have 12 layers; ViT-L has 24.
    return [3, 6, 9, 12] if size != "large" else [6, 12, 18, 24]


class TestPretrainedWeightLoad:
    """Load real gated DINOv3 weights and verify they map cleanly and match upstream."""

    def test_from_pretrained_loads_and_matches_stock(self):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTBackbone

        from rfdetr.models.backbone.dinov3 import DinoV3, size_to_hub_name

        hub_name = size_to_hub_name[_SIZE]
        out_features = [f"stage{i}" for i in _out_feature_indexes(_SIZE)]

        # 1. Our windowed wrapper downloads + loads the gated checkpoint (num_windows=1).
        wrapper = DinoV3(
            size=_SIZE,
            out_feature_indexes=_out_feature_indexes(_SIZE),
            use_windowed_attn=False,  # -> num_windows = 1
            load_dinov3_weights=True,
            patch_size=16,
            num_windows=1,
        ).eval()

        # 2. Upstream backbone from the same checkpoint, same output stages.
        stock = DINOv3ViTBackbone.from_pretrained(hub_name, out_features=out_features).eval()

        # 3. Clean key mapping: every stock parameter is present in our backbone with the
        #    same shape (RoPE inv_freq is a non-persistent buffer, absent from both).
        windowed_state = wrapper.encoder.state_dict()
        for key, tensor in stock.state_dict().items():
            assert key in windowed_state, f"stock key missing from windowed backbone: {key}"
            assert windowed_state[key].shape == tensor.shape, f"shape mismatch at {key}"

        # 4. Numerical equivalence at num_windows=1 with the REAL pretrained weights.
        pixel_values = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = stock(pixel_values).feature_maps
            actual = wrapper(pixel_values)
        assert len(actual) == len(expected)
        for stage_expected, stage_actual in zip(expected, actual):
            assert torch.allclose(stage_expected, stage_actual, atol=1e-4), (
                (stage_expected - stage_actual).abs().max().item()
            )

    def test_windowed_load_and_forward(self):
        """The real weights also load into the genuinely-windowed config and run a forward."""
        from rfdetr.models.backbone.dinov3 import DinoV3

        wrapper = DinoV3(
            size=_SIZE,
            out_feature_indexes=_out_feature_indexes(_SIZE),
            use_windowed_attn=True,
            load_dinov3_weights=True,
            patch_size=16,
            num_windows=2,
        ).eval()
        # 224 is divisible by patch_size * num_windows = 32.
        with torch.no_grad():
            feature_maps = wrapper(torch.randn(1, 3, 224, 224))
        assert len(feature_maps) == 4
        width = wrapper._out_feature_channels[0]
        for feature_map in feature_maps:
            assert tuple(feature_map.shape) == (1, width, 14, 14)
