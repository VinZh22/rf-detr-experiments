# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Muon optimizer. Newton-Schulz orthogonalization adapted from Keller Jordan's
# reference implementation (https://github.com/KellerJordan/Muon, MIT License);
# RMS-matched update scaling follows the Moonshot "Muon is Scalable" formulation
# (https://github.com/MoonshotAI/Moonlight) so AdamW-tuned learning rates transfer.
# ------------------------------------------------------------------------
"""Single-device Muon optimizer with an AdamW fallback for non-matrix parameters.

Muon (MomentUm Orthogonalized by Newton-schulz) replaces each 2D weight matrix's momentum update
with its closest orthogonal matrix, computed via a fixed-step Newton-Schulz iteration. Orthogonal
updates spread learning signal evenly across singular directions, which empirically accelerates
convergence on transformer hidden weights.

Newton-Schulz orthogonalization is only defined for matrices, so this optimizer is *hybrid*: param
groups tagged ``use_muon=True`` (2D hidden weights) use the Muon rule, while groups tagged
``use_muon=False`` (embeddings, output heads, biases, norms — anything 1D or an embedding/head) fall
back to a standard decoupled-weight-decay AdamW step. Each group keeps its own ``lr`` so an external
scheduler (e.g. ``LambdaLR``) drives both branches uniformly.

The Muon update is scaled by ``0.2 * sqrt(max(fan_out, fan_in))`` so its root-mean-square magnitude
matches a typical AdamW update. This lets a run reuse its AdamW-tuned learning rate unchanged, which
is what makes an AdamW-vs-Muon comparison an apples-to-apples test of the update rule alone.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional

import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize a matrix via a quintic Newton-Schulz iteration.

    Computes an approximate ``U @ V.T`` of the SVD ``grad = U @ S @ V.T`` (i.e. ``grad`` with all
    singular values set to ~1), using a fixed quintic iteration with coefficients tuned to converge
    from a spectral-norm-normalized start. Runs in bfloat16 for speed; the slight inexactness at the
    singular-value extremes is harmless for an optimizer update.

    Args:
        grad: Matrix to orthogonalize, shape ``(..., m, n)`` with at least 2 dims.
        steps: Number of Newton-Schulz iterations (5 is the standard choice).

    Returns:
        Orthogonalized matrix with the same shape and dtype as ``grad``.
    """
    assert grad.ndim >= 2, "Newton-Schulz orthogonalization requires a matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    transposed = grad.size(-2) > grad.size(-1)
    if transposed:
        x = x.mT
    # Normalize so the spectral norm is <= 1, the basin where the iteration converges.
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        ax = x @ x.mT
        bx = b * ax + c * ax @ ax
        x = a * x + bx @ x
    if transposed:
        x = x.mT
    return x.to(grad.dtype)


class Muon(torch.optim.Optimizer):
    """Hybrid Muon optimizer: Muon for tagged matrix groups, AdamW for the rest.

    Each param group must set ``use_muon`` (bool). Muon groups apply Nesterov momentum followed by
    Newton-Schulz orthogonalization and RMS-matched scaling; AdamW groups apply a standard
    decoupled-weight-decay Adam step. Per-group ``lr`` and ``weight_decay`` override the optimizer
    defaults, so a layer-wise-decay param dict and an external LR scheduler both work unchanged.

    Args:
        param_groups: Iterable of param groups; each dict should contain ``params`` and ``use_muon``,
            and may override ``lr`` / ``weight_decay``.
        lr: Default learning rate for groups that do not set their own.
        weight_decay: Default decoupled weight decay.
        momentum: Momentum coefficient for the Muon branch.
        nesterov: Whether the Muon branch uses Nesterov momentum.
        ns_steps: Newton-Schulz iteration count for the Muon branch.
        adamw_betas: ``(beta1, beta2)`` for the AdamW branch.
        adamw_eps: Epsilon for the AdamW branch.
    """

    def __init__(
        self,
        param_groups: Iterable[Dict[str, Any]],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if "use_muon" not in group:
                raise ValueError("every Muon param group must set 'use_muon' (bool)")

    @staticmethod
    def _muon_lr_scale(param: torch.Tensor) -> float:
        """Return the RMS-matching scale so a Muon update matches AdamW's update magnitude.

        Args:
            param: The 2D parameter being updated.

        Returns:
            Multiplicative scale ``0.2 * sqrt(max(fan_out, fan_in))``.
        """
        fan_out, fan_in = param.shape[0], param.shape[1]
        return 0.2 * math.sqrt(max(fan_out, fan_in))

    def _step_muon(self, group: Dict[str, Any]) -> None:
        """Apply one Muon update to every parameter in a ``use_muon=True`` group."""
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.ndim > 2:
                grad = grad.view(grad.size(0), -1)
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
            update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
            if weight_decay != 0:
                param.data.mul_(1 - lr * weight_decay)
            param.data.add_(update.view_as(param), alpha=-lr * self._muon_lr_scale(param))

    def _step_adamw(self, group: Dict[str, Any]) -> None:
        """Apply one decoupled-weight-decay AdamW update to a ``use_muon=False`` group."""
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            if weight_decay != 0:
                param.data.mul_(1 - lr * weight_decay)
            param.data.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step over all param groups.

        Args:
            closure: Optional callable that reevaluates the model and returns the loss.

        Returns:
            The loss returned by ``closure`` if provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss
