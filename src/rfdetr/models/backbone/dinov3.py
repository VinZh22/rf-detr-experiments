# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import os

import torch
import torch.nn as nn

from rfdetr.models.backbone.dinov3_with_windowed_attn import (
    WindowedDinov3ViTBackbone,
    WindowedDinov3ViTConfig,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()

size_to_width = {
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov3_small.json",
    "base": "dinov3_base.json",
    "large": "dinov3_large.json",
}

# Pretrained checkpoints on the HuggingFace Hub (gated under Meta's DINOv3 License).
size_to_hub_name = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def get_config(size: str) -> dict:
    """Load the DINOv3 ViT config preset for the given size.

    Args:
        size: One of ``"small"``, ``"base"``, ``"large"``.

    Returns:
        The parsed config dict, with non-constructor metadata keys removed.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "dinov3_configs", size_to_config[size])
    with open(config_path, "r") as f:
        dino_config = json.load(f)
    # Drop metadata that is not a constructor argument; ``model_type`` is fixed by the
    # windowed config class and ``out_features``/``out_indices`` are set per-instance below.
    for key in ("model_type", "architectures", "out_features", "out_indices"):
        dino_config.pop(key, None)
    return dino_config


class DinoV3(nn.Module):
    """Windowed DINOv3 ViT backbone wrapper, mirroring the [`DinoV2`][rfdetr.models.backbone.dinov2.DinoV2] interface.

    Unlike DINOv2, DINOv3 uses RoPE rather than a learned position table, so there is no
    positional-embedding interpolation and ``export`` is a no-op (RoPE handles arbitrary
    resolutions natively). ``num_windows == 1`` disables windowing.
    """

    def __init__(
        self,
        shape: tuple[int, int] = (640, 640),
        out_feature_indexes: list[int] = [3, 6, 9, 12],
        size: str = "base",
        use_windowed_attn: bool = True,
        gradient_checkpointing: bool = False,
        load_dinov3_weights: bool = True,
        patch_size: int = 16,
        num_windows: int = 2,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows if use_windowed_attn else 1

        dino_config = get_config(size)
        dino_config["patch_size"] = patch_size
        dino_config["drop_path_rate"] = drop_path_rate

        # Layers absent from window_block_indexes run global attention; mirror the DINOv2
        # schedule (every layer up to the last output stage is windowed except the
        # output stages themselves).
        window_block_indexes = set(range(out_feature_indexes[-1] + 1))
        window_block_indexes.difference_update(out_feature_indexes)
        window_block_indexes = list(window_block_indexes)

        config = WindowedDinov3ViTConfig(
            **dino_config,
            out_features=[f"stage{i}" for i in out_feature_indexes],
            num_windows=self.num_windows,
            window_block_indexes=window_block_indexes,
            gradient_checkpointing=gradient_checkpointing,
        )

        if load_dinov3_weights:
            self.encoder = WindowedDinov3ViTBackbone.from_pretrained(
                size_to_hub_name[size],
                config=config,
            )
        else:
            self.encoder = WindowedDinov3ViTBackbone(config)

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self) -> None:
        """No-op export hook.

        DINOv3's RoPE is computed from the input resolution at runtime, so unlike the
        DINOv2 backbone there is no position-embedding parameter to interpolate or
        monkeypatch ahead of tracing.
        """
        self._export = True

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        )
        return list(self.encoder(x).feature_maps)
