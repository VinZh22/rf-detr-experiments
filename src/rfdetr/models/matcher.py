# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""Modules to compute the matching cost and solve the corresponding LSAP."""

import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.optimize import linear_sum_assignment
from torch import nn

from rfdetr.models.heads.segmentation import point_sample
from rfdetr.utilities.box_ops import batch_dice_loss, batch_sigmoid_ce_loss, box_cxcywh_to_xyxy, generalized_box_iou
from rfdetr.utilities.logger import get_logger

logger = get_logger()
_SANITIZED_COST_MARGIN = 1.0

#: Environment-variable fallback for ``HungarianMatcher(gpu_finite_check=...)``.  Lets the
#: cost-matrix finiteness check run on-device (before the GPU→CPU transfer) without any config
#: plumbing, so a profiling A/B can toggle it per process.  Default (unset) preserves the legacy
#: CPU-side check.  See :meth:`HungarianMatcher.forward`.
_GPU_FINITE_CHECK_ENV = "RFDETR_MATCHER_GPU_FINITE_CHECK"


def _env_flag(name: str) -> bool:
    """Return ``True`` when environment variable *name* is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: Environment-variable fallback for ``HungarianMatcher(solver=...)``.  Selects the assignment
#: backend without config plumbing so a profiling A/B can switch it per process.  See section 5/Tier-2
#: of ``training_profiling.md``.
_MATCHER_SOLVER_ENV = "RFDETR_MATCHER_SOLVER"
#: ``scipy`` — exact LSAP via SciPy (default, legacy).  ``lap`` — exact LSAP via ``lap.lapjv`` (faster
#: CPU solver).  ``sinkhorn`` — approximate, GPU-resident entropic-OT assignment (no cost-matrix
#: transfer; see :meth:`HungarianMatcher._assign_sinkhorn`).  ``cuda_lap`` — exact, GPU-resident batched
#: LSAP via ``torch_linear_assignment.batch_linear_assignment`` (a CUDA build; see
#: :meth:`HungarianMatcher._assign_cuda_lap`).
_VALID_SOLVERS = ("scipy", "lap", "sinkhorn", "cuda_lap")


def _env_solver() -> str | None:
    """Return the solver named by ``RFDETR_MATCHER_SOLVER`` (validated), or ``None`` when unset."""
    value = os.environ.get(_MATCHER_SOLVER_ENV, "").strip().lower()
    if not value:
        return None
    if value not in _VALID_SOLVERS:
        raise ValueError(f"{_MATCHER_SOLVER_ENV}={value!r} not in {_VALID_SOLVERS}")
    return value


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network For efficiency reasons,
    the targets don't include the no_object.

    Because of this, in general, there are more predictions than targets. In this case, we do a 1-to-1 matching of the
    best predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,  # reserved for future use; not yet implemented
        use_position_modulated_cost: bool = False,  # reserved for future use; not yet implemented
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
        gpu_finite_check: bool | None = None,
        solver: str | None = None,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 50,
    ):
        """Creates the matcher.

        Args:
            cost_class: Relative weight of the classification error in the matching cost.
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates.
            cost_giou: Relative weight of the GIoU loss of the bounding box.
            focal_alpha: Alpha parameter for focal loss used in the classification cost.
            use_pos_only: Reserved for future use; currently has no effect.
            use_position_modulated_cost: Reserved for future use; currently has no effect.
            mask_point_sample_ratio: Downsampling ratio for mask point sampling.
            cost_mask_ce: Relative weight of the binary cross-entropy mask cost.
            cost_mask_dice: Relative weight of the Dice mask cost.
            gpu_finite_check: When ``True``, run the cost-matrix finiteness check on-device before
                the GPU→CPU transfer instead of on the host (numerically identical, but moves the
                full-matrix reduction off the CPU critical path).  ``None`` (default) reads the
                ``RFDETR_MATCHER_GPU_FINITE_CHECK`` environment variable, defaulting to the legacy
                CPU-side check when unset.  Ignored by the ``sinkhorn`` solver (which never transfers
                the cost matrix).
            solver: Assignment backend — ``"scipy"`` (exact LSAP, default), ``"lap"`` (exact LSAP via
                ``lap.lapjv``, faster on small matrices), or ``"sinkhorn"`` (approximate GPU-resident
                entropic-OT assignment).  ``None`` reads ``RFDETR_MATCHER_SOLVER``, defaulting to
                ``"scipy"``.
            sinkhorn_eps: Entropic-regularisation strength for the ``sinkhorn`` solver (smaller →
                closer to the hard optimum, but more iterations / less numerically stable).
            sinkhorn_iters: Number of Sinkhorn iterations for the ``sinkhorn`` solver.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice
        self._warned_non_finite_costs = False
        self.gpu_finite_check = _env_flag(_GPU_FINITE_CHECK_ENV) if gpu_finite_check is None else bool(gpu_finite_check)
        resolved_solver = _env_solver() if solver is None else str(solver).lower()
        if resolved_solver is None:
            resolved_solver = "scipy"
        if resolved_solver not in _VALID_SOLVERS:
            raise ValueError(f"Unknown matcher solver {resolved_solver!r}; expected one of {_VALID_SOLVERS}")
        self.solver = resolved_solver
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters

    @staticmethod
    def _sanitize_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
        """Replace non-finite cost entries with a large finite sentinel.

        >>> HungarianMatcher._sanitize_cost_matrix(
        ...     torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
        ... ).tolist()
        [[1.0, 4.0], [4.0, -2.0]]

        Args:
            cost_matrix: Cost matrix to sanitize before Hungarian assignment.

        Returns:
            Cost matrix with all non-finite entries replaced by a finite sentinel that is no smaller than any valid
            entry.
        """
        finite_mask = torch.isfinite(cost_matrix)
        if finite_mask.all():
            return cost_matrix

        dtype_info = torch.finfo(cost_matrix.dtype)
        if finite_mask.any():
            finite_costs = cost_matrix[finite_mask]
            max_cost = finite_costs.max()
            # Add the largest absolute finite cost so the replacement stays
            # strictly larger than every valid entry, even if all costs are negative.
            replacement_cost = max_cost + finite_costs.abs().max() + _SANITIZED_COST_MARGIN
            # Guard against overflow to inf/NaN and clamp to the maximum finite value.
            if not torch.isfinite(replacement_cost):
                replacement_cost = cost_matrix.new_tensor(dtype_info.max)
            else:
                replacement_cost = torch.clamp(replacement_cost, max=dtype_info.max)
        else:
            # If all entries are non-finite, fall back to a large finite sentinel.
            replacement_cost = cost_matrix.new_tensor(dtype_info.max)

        sanitized_cost_matrix = cost_matrix.clone()
        sanitized_cost_matrix[~finite_mask] = replacement_cost
        return sanitized_cost_matrix

    def _solve_2d(self, cost: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Solve one ``[num_queries, num_targets]`` assignment, dispatching on ``self.solver``.

        Args:
            cost: 2-D CPU float tensor of matching costs (queries × targets).

        Returns:
            ``(row_ind, col_ind)`` arrays of query indices and the target each is matched to — the
            same contract as :func:`scipy.optimize.linear_sum_assignment`.
        """
        if self.solver == "lap":
            return self._lapjv_2d(cost)
        return linear_sum_assignment(cost)

    @staticmethod
    def _lapjv_2d(cost: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Exact LAP via ``lap.lapjv`` — a faster drop-in for SciPy on the small per-image matrices.

        Args:
            cost: 2-D CPU float tensor (queries × targets), ``num_queries >> num_targets``.

        Returns:
            ``(row_ind, col_ind)``: the query matched to each target, and the target indices.
        """
        try:
            import lap
        except ImportError as err:
            raise ImportError("matcher solver='lap' requires the 'lap' package (pip install lapx).") from err
        num_targets = cost.shape[1]
        if num_targets == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        # lapjv minimises; extend_cost squares the rectangular [Q, T] (Q >> T) matrix.  y[j] is the
        # row (query) assigned to column (target) j; the first T entries cover the real targets.
        cost_np = np.ascontiguousarray(cost.numpy(), dtype=np.float64)
        _, _, y = lap.lapjv(cost_np, extend_cost=True)
        return y[:num_targets], np.arange(num_targets)

    def _assign_sinkhorn(
        self, cost_matrix: torch.Tensor, targets: list, group_detr: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Approximate GPU-resident assignment via entropic OT (Sinkhorn) + per-target argmax rounding.

        Fully batched: all ``bs × group_detr`` per-image/per-group cost blocks are padded to the batch's
        max target count ``T_max`` (extra columns masked out), stacked, and solved by a **single** batched
        Sinkhorn; the rounded query indices are moved to the host in **one** transfer.  This avoids the
        per-image Python loop / per-image sync that made the unbatched variant launch-bound.  Each group
        independently assigns all of its image's targets (Group-DETR semantics); rounding picks, per
        (group, target), the argmax query over the soft plan, so two targets within a group may collide on
        a query (the approximation) — quantified in ``tests/models/test_matcher.py``.

        Args:
            cost_matrix: ``[bs, num_queries, total_gt]`` float tensor on the compute device.
            targets: Per-image target dicts (used for the per-image box counts).
            group_detr: Number of query groups.

        Returns:
            Per-image ``(query_idx, target_idx)`` int64 CPU tensors, concatenated across groups.
        """
        bs, num_queries, _ = cost_matrix.shape
        g_num_queries = num_queries // group_detr
        usable = g_num_queries * group_detr
        device = cost_matrix.device
        sizes = [len(v["boxes"]) for v in targets]
        t_max = max(sizes) if sizes else 0
        if t_max == 0:  # no targets anywhere → every image is empty
            empty = torch.empty(0, dtype=torch.int64)
            return [(empty, empty) for _ in range(bs)]

        # Pack the block-diagonal per-image cost blocks into a dense [bs, G, Qg, T_max] tensor with a
        # column-validity mask (padded target columns are excluded from the Sinkhorn solve).  The fill
        # loop is bs iterations of GPU slice-copies — no host sync.
        offsets = np.cumsum([0, *sizes])
        cost_b = cost_matrix.new_zeros(bs, group_detr, g_num_queries, t_max)
        col_valid = torch.zeros(bs, t_max, dtype=torch.bool, device=device)
        for i, num_targets in enumerate(sizes):
            if num_targets == 0:
                continue
            # Image i's queries vs its own targets, split into groups; group g owns global queries
            # [g*Qg : (g+1)*Qg], matching the scipy/lap path's query split.
            cost_b[i, :, :, :num_targets] = cost_matrix[i, :usable, offsets[i] : offsets[i + 1]].reshape(
                group_detr, g_num_queries, num_targets
            )
            col_valid[i, :num_targets] = True

        plan = self._sinkhorn_log_plan(
            cost_b.reshape(bs * group_detr, g_num_queries, t_max),
            col_mask=col_valid.repeat_interleave(group_detr, dim=0),
        )
        local_query = plan.argmax(dim=1).reshape(bs, group_detr, t_max)  # [bs, G, T_max]
        group_offset = (torch.arange(group_detr, device=device) * g_num_queries).view(1, group_detr, 1)
        # Single host transfer of the small index tensor for the whole batch.
        global_query = (local_query + group_offset).to("cpu", torch.int64)  # [bs, G, T_max]

        results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, num_targets in enumerate(sizes):
            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.int64)
                results.append((empty, empty))
                continue
            query_idx = global_query[i, :, :num_targets].reshape(-1)  # group-major [G*T]
            target_idx = torch.arange(num_targets, dtype=torch.int64).repeat(group_detr)  # [G*T]
            results.append((query_idx, target_idx))
        return results

    def _sinkhorn_log_plan(self, cost: torch.Tensor, col_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Log-domain entropic-OT plan for a batched cost tensor ``[..., Qg, T]`` (minimisation).

        Sinkhorn with column marginal 1 (each target fully assigned) and a uniform row (query) marginal.
        The row/column log-marginals are constant shifts on the dual potentials, so they do not change the
        per-target ``argmax`` over queries used for rounding and are omitted.  When *col_mask* is given,
        padded target columns are clamped out of the row update each iteration so they never draw mass.

        Args:
            cost: Batched cost tensor; the last two dims are (queries, targets).
            col_mask: Optional ``[..., T]`` bool mask (``True`` = real target). Padded columns are excluded.

        Returns:
            Log transport plan of the same shape; ``argmax`` over the query axis gives each target's best query.
        """
        log_k = -cost / self.sinkhorn_eps  # [..., Qg, T]
        f = torch.zeros(cost.shape[:-1], device=cost.device, dtype=cost.dtype)  # [..., Qg]
        g = torch.zeros(cost.shape[:-2] + cost.shape[-1:], device=cost.device, dtype=cost.dtype)  # [..., T]
        neg = cost.new_full((), -1e9)  # masked-out columns: excluded from the logsumexp (exp → 0)
        for _ in range(self.sinkhorn_iters):
            g = -torch.logsumexp(log_k + f.unsqueeze(-1), dim=-2)  # [..., T]; column log-marginal 0
            if col_mask is not None:
                g = torch.where(col_mask, g, neg)
            f = -torch.logsumexp(log_k + g.unsqueeze(-2), dim=-1)  # [..., Qg]; row log-marginal 0
        return f.unsqueeze(-1) + log_k + g.unsqueeze(-2)

    def _assign_cuda_lap(
        self, cost_matrix: torch.Tensor, targets: list, group_detr: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Exact GPU-resident batched LSAP via ``torch_linear_assignment.batch_linear_assignment``.

        Solves all ``bs × group_detr`` per-image/per-group sub-problems in a single batched on-device
        call — no ``.cpu()`` of the cost matrix and no Python solver loop — yet returns the **exact**
        optimal one-to-one assignment (verified equal to SciPy in ``tests/models/test_matcher.py``).
        Per-image target counts are padded to the batch max ``T_max`` with a constant cost; because the
        pad cost is constant across queries (workers), the solver assigns padded columns only to spare
        queries and never perturbs the real assignment, and those pairs are filtered out by index.

        Args:
            cost_matrix: ``[bs, num_queries, total_gt]`` float tensor on the compute device.
            targets: Per-image target dicts (used for per-image box counts).
            group_detr: Number of query groups.

        Returns:
            Per-image ``(query_idx, target_idx)`` int64 CPU tensors, concatenated across groups.
        """
        try:
            from torch_linear_assignment import batch_linear_assignment
        except ImportError as err:
            raise ImportError(
                "matcher solver='cuda_lap' requires the 'torch-linear-assignment' package built against "
                "your CUDA toolkit (needs nvcc)."
            ) from err

        bs, num_queries, _ = cost_matrix.shape
        g_num_queries = num_queries // group_detr
        usable = g_num_queries * group_detr
        sizes = [len(v["boxes"]) for v in targets]
        t_max = max(sizes) if sizes else 0
        if t_max == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return [(empty, empty) for _ in range(bs)]

        # Pack block-diagonal cost into [bs, G, Qg, T_max]; padded target columns stay 0 (constant
        # across workers ⇒ they take spare queries without disturbing the real optimum).
        offsets = np.cumsum([0, *sizes])
        cost_b = cost_matrix.new_zeros(bs, group_detr, g_num_queries, t_max)
        for i, num_targets in enumerate(sizes):
            if num_targets:
                cost_b[i, :, :, :num_targets] = cost_matrix[i, :usable, offsets[i] : offsets[i + 1]].reshape(
                    group_detr, g_num_queries, num_targets
                )

        # assign[b, q] = target assigned to query q for batch element b (−1 if unassigned).
        assign = batch_linear_assignment(cost_b.reshape(bs * group_detr, g_num_queries, t_max).contiguous())
        assign = assign.reshape(bs, group_detr, g_num_queries).to("cpu")  # single host transfer
        group_base = torch.arange(group_detr, dtype=torch.int64) * g_num_queries  # [G]

        results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, num_targets in enumerate(sizes):
            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.int64)
                results.append((empty, empty))
                continue
            assigned = assign[i]  # [G, Qg]
            valid = (assigned >= 0) & (assigned < num_targets)  # drop padded-target assignments
            gq = valid.nonzero(as_tuple=False)  # [K, 2] → (group, local-query), group-major
            group_ids, local_q = gq[:, 0], gq[:, 1]
            query_idx = (local_q + group_base[group_ids]).to(torch.int64)
            target_idx = assigned[group_ids, local_q].to(torch.int64)
            results.append((query_idx, target_idx))
        return results

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates "masks": Tensor of
                 dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_prob = flat_pred_logits.sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        masks_present = "masks" in targets[0]

        # Compute the giou cost between boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0

        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # we refactor these with logsigmoid for numerical stability
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])

            if isinstance(outputs["pred_masks"], torch.Tensor):
                out_masks = outputs["pred_masks"].flatten(0, 1)

                num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

                point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
                pred_masks_logits = point_sample(
                    out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False
                ).squeeze(1)
            else:
                spatial_features = outputs["pred_masks"]["spatial_features"]
                query_features = outputs["pred_masks"]["query_features"]
                bias = outputs["pred_masks"]["bias"]

                num_points = spatial_features.shape[-2] * spatial_features.shape[-1] // self.mask_point_sample_ratio
                point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
                pred_masks_logits = point_sample(
                    spatial_features, point_coords.repeat(spatial_features.shape[0], 1, 1), align_corners=False
                )
                # print(f"pred_masks_logits.shape: {pred_masks_logits.shape}")
                pred_masks_logits = torch.einsum("bcp,bnc->bnp", pred_masks_logits, query_features) + bias
                pred_masks_logits = pred_masks_logits.flatten(0, 1)

            tgt_masks = tgt_masks.to(pred_masks_logits.dtype)
            tgt_masks_flat = point_sample(
                tgt_masks.unsqueeze(1),
                point_coords.repeat(tgt_masks.shape[0], 1, 1),
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

            # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

            # Dice loss cost (1 - dice coefficient)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if masks_present:
            cost_matrix = cost_matrix + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        # Cast to float32 on-device (bfloat16 doesn't play nicely with the CPU LSAP solver).
        cost_matrix = cost_matrix.view(bs, num_queries, -1).float()

        # GPU-resident solvers stay on-device: sanitize on the GPU and never transfer the full cost
        # matrix — only the small index tensors come back to the host.
        if self.solver == "sinkhorn":
            return self._assign_sinkhorn(self._sanitize_cost_matrix(cost_matrix), targets, group_detr)
        if self.solver == "cuda_lap":
            return self._assign_cuda_lap(self._sanitize_cost_matrix(cost_matrix), targets, group_detr)

        # The full-matrix finiteness reduction is the dominant CPU cost of matching.  With
        # gpu_finite_check it runs on-device *before* the transfer (only a scalar bool is synced);
        # otherwise it runs on the host after .cpu() (legacy default).  The two paths are
        # numerically identical: casting bfloat16→float32 neither creates nor removes non-finite
        # values and the host copy is exact, so the same entries are flagged either way.
        if self.gpu_finite_check:
            has_non_finite = not bool(torch.isfinite(cost_matrix).all().item())
            cost_matrix = cost_matrix.cpu()
        else:
            cost_matrix = cost_matrix.cpu()
            has_non_finite = not bool(torch.isfinite(cost_matrix).all())

        # We assume any good match will not cause NaN or Inf, so replace invalid
        # entries with a finite value that is larger than every valid cost.
        if has_non_finite:
            if not self._warned_non_finite_costs:
                logger.warning(
                    "Non-finite values detected in matcher cost matrix; "
                    "replacing with finite sentinel. "
                    "Check for numerical instability."
                )
                self._warned_non_finite_costs = True
            cost_matrix = self._sanitize_cost_matrix(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        cost_matrix_list = cost_matrix.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            grouped_cost_matrix = cost_matrix_list[g_i]
            indices_g = [self._solve_2d(c[i]) for i, c in enumerate(grouped_cost_matrix.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )
