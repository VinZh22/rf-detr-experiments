# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the opt-in Matchability-Aware Loss (MAL, DEIM arXiv:2412.04234).

MAL is an IoU-aware classification loss: at a matched (query, class) pair the target is
``q**gamma`` (q = IoU of the matched pred/GT box) giving BCE
``-(q**g*log p + (1-q**g)*log(1-p))``; every other entry is a focal-weighted negative
``-(p**g*log(1-p))``. These tests verify the implementation matches that formula exactly
and that the ``mal_loss`` flag routes through config -> criterion (and wins over the
default ``ia_bce_loss=True``).
"""

import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.utilities import box_ops


def _make_inputs():
    torch.manual_seed(0)
    batch, queries, num_classes = 1, 5, 4
    logits = torch.randn(batch, queries, num_classes)
    boxes = torch.rand(batch, queries, 4) * 0.4 + 0.3  # cxcywh in a sane mid-range
    target_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    target_labels = torch.tensor([1])
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    targets = [{"labels": target_labels, "boxes": target_boxes}]
    # match query 2 -> target 0
    indices = [(torch.tensor([2]), torch.tensor([0]))]
    return outputs, targets, indices, num_classes


def _criterion(num_classes, *, mal=False, mal_gamma=1.5, ia_bce=True):
    return SetCriterion(
        num_classes=num_classes,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=["labels"],
        group_detr=1,
        ia_bce_loss=ia_bce,
        mal_loss=mal,
        mal_gamma=mal_gamma,
    )


class TestMalLoss:
    def test_matches_closed_form(self):
        """The fused MAL loss must equal the explicit per-element formula."""
        outputs, targets, indices, num_classes = _make_inputs()
        gamma = 1.5
        criterion = _criterion(num_classes, mal=True, mal_gamma=gamma)

        got = criterion.loss_labels(outputs, targets, indices, num_boxes=1, log=False)["loss_ce"]

        logits = outputs["pred_logits"]
        prob = logits.sigmoid()
        # negatives everywhere: -(p**g * log(1-p))
        reference = -(prob**gamma) * torch.log(1 - prob)
        # positive at (batch=0, query=2, class=1): -(q**g*log p + (1-q**g)*log(1-p))
        iou = torch.diag(
            box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][0, [2]]),
                box_ops.box_cxcywh_to_xyxy(targets[0]["boxes"]),
            )[0]
        )[0]
        p = prob[0, 2, 1]
        q = iou.clamp(0, 1)
        reference[0, 2, 1] = -((q**gamma) * torch.log(p) + (1 - q**gamma) * torch.log(1 - p))
        expected = reference.sum()

        assert torch.allclose(got, expected, atol=1e-5), f"{got.item()} vs {expected.item()}"

    def test_differs_from_ia_bce(self):
        """MAL and IA-BCE must produce different values (it is a real loss swap, not a no-op)."""
        outputs, targets, indices, num_classes = _make_inputs()
        mal = _criterion(num_classes, mal=True)
        ia_bce = _criterion(num_classes, mal=False, ia_bce=True)
        mal_loss = mal.loss_labels(outputs, targets, indices, num_boxes=1, log=False)["loss_ce"]
        ia_loss = ia_bce.loss_labels(outputs, targets, indices, num_boxes=1, log=False)["loss_ce"]
        assert not torch.allclose(mal_loss, ia_loss)

    def test_gradient_flows_to_logits_only(self):
        """Loss is differentiable w.r.t. logits; the IoU target carries no gradient."""
        outputs, targets, indices, num_classes = _make_inputs()
        outputs["pred_logits"].requires_grad_(True)
        outputs["pred_boxes"].requires_grad_(True)
        criterion = _criterion(num_classes, mal=True)
        loss = criterion.loss_labels(outputs, targets, indices, num_boxes=1, log=False)["loss_ce"]
        loss.backward()
        assert outputs["pred_logits"].grad is not None
        assert outputs["pred_logits"].grad.abs().sum() > 0
        # IoU target is detached -> no gradient path through boxes from the classification loss
        assert outputs["pred_boxes"].grad is None or outputs["pred_boxes"].grad.abs().sum() == 0


class TestMalConfigRouting:
    def test_flag_wins_over_ia_bce_default(self):
        """RFDETRDinov3BaseConfig(mal_loss=True) -> criterion.mal_loss True even though ia_bce_loss defaults True."""
        from rfdetr.config import RFDETRDinov3BaseConfig, TrainConfig
        from rfdetr.models import build_criterion_from_config

        mc = RFDETRDinov3BaseConfig(mal_loss=True, num_classes=2)
        tc = TrainConfig(dataset_dir=".", output_dir=".")
        criterion, _ = build_criterion_from_config(mc, tc)
        assert criterion.mal_loss is True
        assert criterion.ia_bce_loss is True  # still set, but MAL branch is checked first

    def test_default_is_ia_bce(self):
        """Without the flag, mal_loss is off and IA-BCE remains the active branch."""
        from rfdetr.config import RFDETRDinov3BaseConfig, TrainConfig
        from rfdetr.models import build_criterion_from_config

        mc = RFDETRDinov3BaseConfig(num_classes=2)
        tc = TrainConfig(dataset_dir=".", output_dir=".")
        criterion, _ = build_criterion_from_config(mc, tc)
        assert criterion.mal_loss is False
