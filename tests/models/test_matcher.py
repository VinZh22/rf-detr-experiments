# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models import matcher as matcher_module
from rfdetr.models.matcher import HungarianMatcher


@pytest.fixture()
def matcher() -> HungarianMatcher:
    """Shared HungarianMatcher instance."""
    return HungarianMatcher()


@pytest.fixture()
def standard_target() -> dict[str, torch.Tensor]:
    """Single-class target with one box at (0.5, 0.5, 0.2, 0.2)."""
    return {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
    }


class TestHungarianMatcherNonFiniteCosts:
    """Tests for non-finite cost matrix sanitization in the Hungarian matcher."""

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="-inf"),
        ],
    )
    def test_replaces_non_finite_costs_before_assignment(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
        invalid_value: float,
    ) -> None:
        """Matcher should sanitize non-finite costs so assignment still succeeds."""
        outputs = {
            "pred_logits": torch.tensor([[[0.0], [10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [invalid_value, 0.5, 0.2, 0.2],
                        [0.5, 0.5, 0.2, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        matched_queries, matched_targets = matcher(outputs, [standard_target])[0]

        assert matched_queries.tolist() == [1]
        assert matched_targets.tolist() == [0]

    def test_all_nonfinite_produces_valid_assignment(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """When ALL costs are non-finite, the fallback sentinel (``dtype_info.max``)
        should allow ``linear_sum_assignment`` to complete with a valid 1-to-1
        assignment: exactly one match, query index in [0, num_queries), target index 0.

        This exercises the ``else: replacement_cost = C.new_tensor(dtype_info.max)`` branch.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor([[[nan], [nan]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],
                        [nan, nan, nan, nan],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        matched_queries, matched_targets = matcher(outputs, [standard_target])[0]

        assert len(matched_queries) == len(matched_targets) == 1
        assert 0 <= matched_queries.item() < 2
        assert matched_targets.item() == 0

    def test_negative_costs_with_nan_selects_valid_query(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """Regression test: when all finite costs are negative and one query
        produces NaN, the matcher must select the valid query, not the NaN one.

        This guards against the bug where ``max_cost * 2`` (the old replacement formula) could be smaller than
        ``max_cost`` when all costs are negative, causing the NaN query to appear cheaper than valid queries.
        """
        nan = float("nan")
        # Query 0: NaN box coordinates -> produces non-finite costs
        # Query 1: valid box, low logit -> all-negative but finite costs
        outputs = {
            "pred_logits": torch.tensor([[[0.0], [-10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],
                        [0.5, 0.5, 0.2, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        matched_queries, matched_targets = matcher(outputs, [standard_target])[0]

        # The valid query (index 1) must be matched, not the NaN query.
        assert matched_queries.tolist() == [1]
        assert matched_targets.tolist() == [0]

    @pytest.mark.parametrize(
        "image_idx, expected_query_idx",
        [
            pytest.param(0, 1, id="image0"),
            pytest.param(1, 0, id="image1"),
        ],
    )
    def test_batch_size_greater_than_one(
        self,
        matcher: HungarianMatcher,
        image_idx: int,
        expected_query_idx: int,
    ) -> None:
        """Exercises the ``C.split(sizes, -1)`` loop with batch_size > 1.

        Each image has 2 queries and 1 target. One query per image has NaN coordinates; the matcher must select the
        valid query in each case.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor(
                [
                    [[0.0], [10.0]],  # image 0: query 1 is valid
                    [[10.0], [0.0]],  # image 1: query 0 is valid
                ],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, 0.5, 0.2, 0.2],  # image 0, query 0: NaN
                        [0.5, 0.5, 0.2, 0.2],  # image 0, query 1: valid
                    ],
                    [
                        [0.5, 0.5, 0.2, 0.2],  # image 1, query 0: valid
                        [nan, 0.5, 0.2, 0.2],  # image 1, query 1: NaN
                    ],
                ],
                dtype=torch.float32,
            ),
        }
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
        ]

        results = matcher(outputs, targets)

        assert len(results) == 2

        matched_queries, matched_targets = results[image_idx]
        assert matched_queries.tolist() == [expected_query_idx]
        assert matched_targets.tolist() == [0]

    def test_group_detr_with_nonfinite_costs(
        self,
        matcher: HungarianMatcher,
        standard_target: dict[str, torch.Tensor],
    ) -> None:
        """Sanitization runs on the full cost matrix before splitting by group, so non-finite entries must be handled
        correctly when ``group_detr > 1``.

        4 queries, 2 groups of 2. Query 0 has a NaN box; query 2 (the best valid match in group 1) must be selected
        across groups.
        """
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor(
                [[[0.0], [10.0], [0.0], [10.0]]],
                dtype=torch.float32,
            ),
            "pred_boxes": torch.tensor(
                [
                    [
                        [nan, nan, nan, nan],  # group 0, query 0: NaN
                        [0.5, 0.5, 0.2, 0.2],  # group 0, query 1: valid
                        [nan, nan, nan, nan],  # group 1, query 0: NaN
                        [0.5, 0.5, 0.2, 0.2],  # group 1, query 1: valid
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        results = matcher(outputs, [standard_target], group_detr=2)

        assert len(results) == 1
        matched_queries, matched_targets = results[0]
        # Each group contributes one match; both must map to target 0
        assert matched_targets.tolist() == [0, 0]
        # The valid query in each group (indices 1 and 3) must be selected
        assert set(matched_queries.tolist()) == {1, 3}

    def test_warns_once_per_matcher_instance(
        self, standard_target: dict[str, torch.Tensor], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-finite-cost warning should be emitted once per matcher instance."""
        expected_warning = (
            "Non-finite values detected in matcher cost matrix; "
            "replacing with finite sentinel. "
            "Check for numerical instability."
        )
        warning_messages: list[str] = []

        def record_warning(msg: str, *args: object, **kwargs: object) -> None:
            warning_messages.append(msg)

        monkeypatch.setattr(matcher_module.logger, "warning", record_warning)

        outputs = {
            "pred_logits": torch.tensor([[[0.0], [10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor(
                [
                    [
                        [float("nan"), 0.5, 0.2, 0.2],
                        [0.5, 0.5, 0.2, 0.2],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

        first_matcher = HungarianMatcher()
        second_matcher = HungarianMatcher()

        first_matcher(outputs, [standard_target])
        first_matcher(outputs, [standard_target])
        second_matcher(outputs, [standard_target])

        assert warning_messages == [expected_warning, expected_warning]


class TestHungarianMatcherGpuFiniteCheck:
    """The ``gpu_finite_check`` toggle must change only *where* the finiteness reduction runs,
    never the resulting matches.  These run on CPU (where ``.cpu()`` is a no-op), so they verify
    the two branches are logically equivalent; on GPU the only difference is the device the
    reduction executes on, which cannot change the boolean outcome."""

    @staticmethod
    def _finite_problem() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        """Deterministic batch=2 / 6-query / 2-target-per-image problem with all-finite costs."""
        torch.manual_seed(0)
        outputs = {
            "pred_logits": torch.randn(2, 6, 3),
            "pred_boxes": torch.rand(2, 6, 4),
        }
        targets = [
            {"labels": torch.tensor([0, 2], dtype=torch.int64), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([1, 0], dtype=torch.int64), "boxes": torch.rand(2, 4)},
        ]
        return outputs, targets

    @staticmethod
    def _assert_same_matches(results_a: list, results_b: list) -> None:
        assert len(results_a) == len(results_b)
        for (q_a, t_a), (q_b, t_b) in zip(results_a, results_b):
            assert q_a.tolist() == q_b.tolist()
            assert t_a.tolist() == t_b.tolist()

    @pytest.mark.parametrize("group_detr", [1, 2, 3])
    def test_finite_costs_identical_matches(self, group_detr: int) -> None:
        """With finite costs, on-device and host finiteness checks yield identical assignments."""
        outputs, targets = self._finite_problem()
        legacy = HungarianMatcher(gpu_finite_check=False)
        on_device = HungarianMatcher(gpu_finite_check=True)
        self._assert_same_matches(
            legacy(outputs, targets, group_detr=group_detr),
            on_device(outputs, targets, group_detr=group_detr),
        )

    def test_nonfinite_costs_identical_matches(self) -> None:
        """The sanitize path is reached identically under both finiteness-check locations."""
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor([[[0.0], [10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor([[[nan, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
        }
        target = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        self._assert_same_matches(
            HungarianMatcher(gpu_finite_check=False)(outputs, target),
            HungarianMatcher(gpu_finite_check=True)(outputs, target),
        )

    def test_env_var_sets_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``gpu_finite_check=None`` (default) reads ``RFDETR_MATCHER_GPU_FINITE_CHECK``."""
        monkeypatch.setenv("RFDETR_MATCHER_GPU_FINITE_CHECK", "1")
        assert HungarianMatcher().gpu_finite_check is True
        monkeypatch.setenv("RFDETR_MATCHER_GPU_FINITE_CHECK", "0")
        assert HungarianMatcher().gpu_finite_check is False
        monkeypatch.delenv("RFDETR_MATCHER_GPU_FINITE_CHECK", raising=False)
        assert HungarianMatcher().gpu_finite_check is False

    def test_constructor_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit constructor value takes precedence over the environment variable."""
        monkeypatch.setenv("RFDETR_MATCHER_GPU_FINITE_CHECK", "1")
        assert HungarianMatcher(gpu_finite_check=False).gpu_finite_check is False


class TestHungarianMatcherSolvers:
    """Tier-2 solver backends: ``lap`` must reproduce scipy's optimal assignment exactly; ``sinkhorn``
    (approximate, GPU-resident) must produce structurally valid assignments and recover the optimum on
    well-separated problems. Solver selection is wired via constructor arg / ``RFDETR_MATCHER_SOLVER``."""

    @staticmethod
    def _random_problem(seed: int) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        torch.manual_seed(seed)
        outputs = {"pred_logits": torch.randn(2, 8, 4), "pred_boxes": torch.rand(2, 8, 4)}
        targets = [
            {"labels": torch.tensor([0, 2, 3]), "boxes": torch.rand(3, 4)},
            {"labels": torch.tensor([1, 0]), "boxes": torch.rand(2, 4)},
        ]
        return outputs, targets

    @staticmethod
    def _pair_sets(result: list) -> list[set[tuple[int, int]]]:
        """Per-image set of (query, target) pairs — order-independent comparison."""
        return [set(zip(q.tolist(), t.tolist())) for q, t in result]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_lap_matches_scipy_exactly(self, seed: int) -> None:
        """`lap` solves the same LSAP as scipy → identical (query, target) pairs (unique optimum)."""
        outputs, targets = self._random_problem(seed)
        scipy_res = HungarianMatcher(solver="scipy")(outputs, targets)
        lap_res = HungarianMatcher(solver="lap")(outputs, targets)
        assert self._pair_sets(lap_res) == self._pair_sets(scipy_res)

    def test_lap_handles_nonfinite_costs(self) -> None:
        """The CPU sanitize path runs before the solver, so `lap` tolerates non-finite costs."""
        nan = float("nan")
        outputs = {
            "pred_logits": torch.tensor([[[0.0], [10.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor([[[nan, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
        }
        target = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        q, t = HungarianMatcher(solver="lap")(outputs, target)[0]
        assert q.tolist() == [1] and t.tolist() == [0]

    def test_sinkhorn_recovers_obvious_optimum(self) -> None:
        """With queries 0–2 exact copies of targets 0–2, Sinkhorn's argmax rounding recovers the
        optimal one-to-one assignment (same as scipy)."""
        boxes = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.3, 0.1, 0.3]])
        labels = torch.tensor([0, 1, 2])
        far = torch.tensor([[0.1, 0.9, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05]])
        logits = torch.full((1, 6, 4), -5.0)
        for j in range(3):
            logits[0, j, labels[j]] = 5.0
        outputs = {"pred_logits": logits, "pred_boxes": torch.cat([boxes, far]).unsqueeze(0)}
        targets = [{"labels": labels, "boxes": boxes}]
        scipy_res = HungarianMatcher(solver="scipy")(outputs, targets)
        sink_res = HungarianMatcher(solver="sinkhorn")(outputs, targets)
        assert self._pair_sets(sink_res) == self._pair_sets(scipy_res) == [{(0, 0), (1, 1), (2, 2)}]

    def test_sinkhorn_structure_with_groups(self) -> None:
        """Group-DETR semantics: each of the `group_detr` groups assigns every target to a distinct
        in-range query, so totals and per-target multiplicity are exact even though rounding is approximate."""
        outputs, targets = self._random_problem(0)  # num_queries=8, group_detr=2 → 4 queries/group
        result = HungarianMatcher(solver="sinkhorn")(outputs, targets, group_detr=2)
        for (q, t), n in zip(result, [len(x["boxes"]) for x in targets]):
            assert len(q) == len(t) == 2 * n  # group_detr * n_targets
            assert q.min() >= 0 and q.max() < 8
            assert torch.equal(torch.bincount(t, minlength=n), torch.full((n,), 2))  # each target hit once/group

    def test_solver_env_and_ctor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`solver=None` (default) reads `RFDETR_MATCHER_SOLVER`; an explicit ctor value overrides it."""
        monkeypatch.setenv("RFDETR_MATCHER_SOLVER", "lap")
        assert HungarianMatcher().solver == "lap"
        assert HungarianMatcher(solver="sinkhorn").solver == "sinkhorn"
        monkeypatch.delenv("RFDETR_MATCHER_SOLVER", raising=False)
        assert HungarianMatcher().solver == "scipy"

    def test_invalid_solver_raises(self) -> None:
        with pytest.raises(ValueError, match="solver"):
            HungarianMatcher(solver="bogus")

    @pytest.mark.gpu
    def test_cuda_lap_matches_scipy_exactly(self) -> None:
        """The GPU-resident exact solver must reproduce scipy's assignment bit-for-bit (Tier-3)."""
        pytest.importorskip("torch_linear_assignment")
        if not torch.cuda.is_available():
            pytest.skip("cuda_lap requires CUDA")
        outputs, targets = self._random_problem(0)
        outputs = {k: v.cuda() for k, v in outputs.items()}
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        scipy_res = HungarianMatcher(solver="scipy").cuda()(outputs, targets, group_detr=2)
        cuda_res = HungarianMatcher(solver="cuda_lap").cuda()(outputs, targets, group_detr=2)
        assert self._pair_sets(cuda_res) == self._pair_sets(scipy_res)


class TestHungarianMatcherSanitization:
    """Unit tests for the private matcher cost sanitization helper."""

    def test_sanitize_cost_matrix_replaces_non_finite_entries(self) -> None:
        """Non-finite entries should be replaced with a larger finite sentinel."""
        cost_matrix = torch.tensor(
            [
                [1.0, float("nan")],
                [float("inf"), -2.0],
            ],
            dtype=torch.float32,
        )

        sanitized = HungarianMatcher._sanitize_cost_matrix(cost_matrix)

        assert torch.isfinite(sanitized).all()
        assert sanitized[0, 1] == 4.0
        assert sanitized[1, 0] == 4.0
        assert sanitized[0, 0] == 1.0
        assert sanitized[1, 1] == -2.0

    def test_sanitize_cost_matrix_all_non_finite_fallback(self) -> None:
        """All-non-finite matrices should fall back to the dtype maximum."""
        cost_matrix = torch.tensor(
            [
                [float("nan"), float("inf")],
                [float("-inf"), float("nan")],
            ],
            dtype=torch.float32,
        )

        sanitized = HungarianMatcher._sanitize_cost_matrix(cost_matrix)

        assert torch.isfinite(sanitized).all()
        assert torch.all(sanitized == torch.finfo(cost_matrix.dtype).max)

    def test_sanitize_cost_matrix_clamps_overflowing_replacement_cost(self) -> None:
        """Overflow in the computed replacement cost should clamp to dtype max."""
        dtype_max = torch.finfo(torch.float32).max
        cost_matrix = torch.tensor(
            [
                [dtype_max, float("nan")],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        sanitized = HungarianMatcher._sanitize_cost_matrix(cost_matrix)

        assert torch.isfinite(sanitized).all()
        assert sanitized[0, 1] == dtype_max
