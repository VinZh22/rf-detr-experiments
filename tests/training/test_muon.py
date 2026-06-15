# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.muon import Muon, zeropower_via_newtonschulz5


class TestNewtonSchulz:
    """Newton-Schulz orthogonalization produces near-orthogonal matrices."""

    def test_pushes_singular_values_toward_one(self) -> None:
        torch.manual_seed(0)
        g = torch.randn(16, 16)
        raw_sv = torch.linalg.svdvals(g)
        q_sv = torch.linalg.svdvals(zeropower_via_newtonschulz5(g, steps=5).float())
        # Orthogonalization collapses the wide singular spectrum of a random matrix toward 1.
        assert q_sv.min() > 0.5 and q_sv.max() < 1.5
        assert (q_sv.max() - q_sv.min()) < 0.25 * (raw_sv.max() - raw_sv.min())

    def test_preserves_shape_for_rectangular(self) -> None:
        g = torch.randn(8, 32)
        q = zeropower_via_newtonschulz5(g, steps=5)
        assert q.shape == g.shape


class TestMuonStep:
    """A few Muon steps reduce a simple quadratic loss for both branches."""

    def test_muon_branch_decreases_loss(self) -> None:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(8, 8))
        target = torch.randn(8, 8)
        opt = Muon([{"params": weight, "use_muon": True, "lr": 0.05}], lr=0.05)
        losses = []
        for _ in range(25):
            opt.zero_grad()
            loss = ((weight - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_adamw_branch_decreases_loss(self) -> None:
        torch.manual_seed(0)
        bias = torch.nn.Parameter(torch.randn(8))
        target = torch.randn(8)
        opt = Muon([{"params": bias, "use_muon": False, "lr": 0.1}], lr=0.1)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((bias - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.5 * losses[0]

    def test_missing_use_muon_flag_raises(self) -> None:
        weight = torch.nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="use_muon"):
            Muon([{"params": weight, "lr": 0.01}], lr=0.01)


class TestMuonParamClassification:
    """The hybrid split routes matrices to Muon and everything else to AdamW."""

    @pytest.mark.parametrize(
        ("name", "ndim", "expected"),
        [
            pytest.param("backbone.0.blocks.3.attn.qkv.weight", 2, True, id="attn-matrix"),
            pytest.param("transformer.decoder.layers.0.linear1.weight", 2, True, id="decoder-matrix"),
            pytest.param("backbone.0.pos_embed", 2, False, id="pos-embed"),
            pytest.param("class_embed.weight", 2, False, id="class-head"),
            pytest.param("transformer.decoder.norm.weight", 1, False, id="norm-1d"),
            pytest.param("backbone.0.blocks.3.attn.qkv.bias", 1, False, id="bias-1d"),
        ],
    )
    def test_is_muon_param(self, name: str, ndim: int, expected: bool) -> None:
        param = torch.zeros([8, 8] if ndim == 2 else [8])
        assert RFDETRModelModule._is_muon_param(name, param) is expected
