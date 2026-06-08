# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training smoke test for the windowed DINOv3 backbone.

Overfits a tiny windowed DINOv3 backbone to a fixed target feature map for a handful of
steps and asserts the training mechanics hold: the loss drops sharply, gradients reach the
backbone (through the window fold + RoPE apply), backbone parameters update, and the RoPE
``inv_freq`` buffer stays frozen (it is a non-persistent buffer, not a trainable parameter).

A tiny config keeps this CPU-fast; the full ViT-B detector was separately verified to train
end-to-end (gradients flow, parameters update, RoPE frozen) but is too slow for CI.
"""

import torch
import torch.nn as nn

from rfdetr.models.backbone.dinov3_with_windowed_attn import (
    WindowedDinov3ViTBackbone,
    WindowedDinov3ViTConfig,
)

_INV_FREQ_BUFFER = "rope_embeddings.inv_freq"


def _tiny_backbone() -> WindowedDinov3ViTBackbone:
    config = WindowedDinov3ViTConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        num_register_tokens=2,
        patch_size=16,
        image_size=64,
        out_features=["stage4"],
        num_windows=2,
        window_block_indexes=[0, 1, 2, 3],
        _attn_implementation="eager",
    )
    return WindowedDinov3ViTBackbone(config)


def test_backbone_overfits_and_keeps_rope_frozen():
    torch.manual_seed(0)
    backbone = _tiny_backbone().train()

    pixel_values = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 32, 4, 4)  # fixed target feature map to overfit
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)

    # inv_freq must be a buffer (frozen), never a trainable parameter.
    assert not any("inv_freq" in name for name, _ in backbone.named_parameters())
    inv_freq_before = dict(backbone.named_buffers())[_INV_FREQ_BUFFER].detach().clone()

    # A parameter inside a windowed (local) layer's attention — its gradient proves
    # backprop flows through the window fold and the RoPE apply.
    tracked = "model.layer.0.attention.q_proj.weight"
    param_before = dict(backbone.named_parameters())[tracked].detach().clone()

    losses: list[float] = []
    grad_reached_backbone = False
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        output = backbone(pixel_values).feature_maps[0]
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        grad = dict(backbone.named_parameters())[tracked].grad
        if grad is not None and grad.abs().sum() > 0:
            grad_reached_backbone = True
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(value)) for value in losses), "non-finite loss during training"
    assert losses[-1] < 0.3 * losses[0], f"loss did not drop enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert grad_reached_backbone, "no gradient reached the backbone attention"

    param_after = dict(backbone.named_parameters())[tracked].detach()
    assert not torch.equal(param_before, param_after), "backbone parameter did not update"

    inv_freq_after = dict(backbone.named_buffers())[_INV_FREQ_BUFFER].detach()
    assert torch.equal(inv_freq_before, inv_freq_after), "RoPE inv_freq buffer changed during training"
