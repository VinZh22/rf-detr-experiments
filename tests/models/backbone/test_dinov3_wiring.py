# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Integration tests for selecting the windowed DINOv3 backbone from config.

These exercise the wiring path ``config.encoder`` -> ``Backbone`` factory -> ``DinoV3``
wrapper -> ``WindowedDinov3ViTBackbone``, plus a full ``build_model_from_config`` build.
All tests run with backbone weight-loading disabled so they never touch the gated
``facebook/dinov3-*`` Hub repos.
"""

import warnings

import pytest
import torch

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov3 import DinoV3
from rfdetr.utilities.tensors import NestedTensor


class TestBackboneFactoryRouting:
    """The ``Backbone`` factory must route ``dinov3_*`` names to the DINOv3 wrapper."""

    @pytest.mark.parametrize(
        "name, width",
        [
            pytest.param("dinov3_windowed_small", 384, id="small"),
            pytest.param("dinov3_windowed_base", 768, id="base"),
        ],
    )
    def test_routes_to_dinov3_and_projects(self, name: str, width: int):
        backbone = Backbone(
            name=name,
            out_feature_indexes=[3, 6, 9, 12],
            projector_scale=["P4"],
            num_windows=2,
            patch_size=16,
            load_dinov2_weights=False,  # random init, no Hub download
            target_shape=(64, 64),
            out_channels=256,
        ).eval()
        assert isinstance(backbone.encoder, DinoV3)
        assert backbone.encoder._out_feature_channels == [width] * 4

        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        with torch.no_grad():
            out = backbone(NestedTensor(torch.randn(1, 3, 64, 64), mask))
        assert len(out) == 1  # one projector scale
        assert tuple(out[0].tensors.shape) == (1, 256, 4, 4)

    def test_export_hook_is_noop(self):
        """The DINOv3 wrapper's export hook must be callable and a no-op (RoPE needs no PE surgery)."""
        backbone = Backbone(
            name="dinov3_windowed_small",
            out_feature_indexes=[3, 6, 9, 12],
            projector_scale=["P4"],
            num_windows=2,
            patch_size=16,
            load_dinov2_weights=False,
            target_shape=(64, 64),
            out_channels=256,
        )
        backbone.encoder.export()
        assert backbone.encoder._export is True


class TestBuildModelFromConfig:
    """A full RF-DETR model must build and run with the DINOv3 config."""

    def test_builds_and_runs_forward(self):
        from rfdetr.config import RFDETRDinov3BaseConfig
        from rfdetr.models import build_model_from_config
        from rfdetr.models.lwdetr import LWDETR

        # Dummy pretrain_weights -> load_dinov2_weights=False -> random backbone init, no Hub.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = RFDETRDinov3BaseConfig(pretrain_weights="dummy.pth", resolution=64, num_classes=2)
            model = build_model_from_config(config).eval()

        assert isinstance(model, LWDETR)
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        assert tuple(output["pred_logits"].shape) == (1, 300, 3)  # num_classes + 1
        assert tuple(output["pred_boxes"].shape) == (1, 300, 4)


class TestConfigDefaults:
    """The DINOv3 config defaults must be self-consistent."""

    def test_resolution_divisible_by_block_size(self):
        from rfdetr.config import RFDETRDinov3BaseConfig

        config = RFDETRDinov3BaseConfig()
        assert config.encoder == "dinov3_windowed_base"
        assert config.resolution % (config.patch_size * config.num_windows) == 0
        assert config.pretrain_weights is None
