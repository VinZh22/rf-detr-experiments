# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Profile an RF-DETR training run to locate CPU/GPU bottlenecks.

Runs the *real* training stack (``RFDETRModelModule`` + ``RFDETRDataModule`` + ``build_trainer``)
so precision (bf16), EMA, the COCO-eval callback, and the data pipeline all behave exactly as in a
production run — then bounds it to a short smoke test and attaches two complementary profilers:

* **PyTorch Lightning ``SimpleProfiler``** — wall-clock per training-loop hook.  The key rows are
  ``[_TrainingEpochLoop].train_dataloader_next`` (time *waiting* on the DataLoader → the data/CPU-bound
  signal), ``run_training_batch`` / ``[LightningModule]training_step`` (forward+loss), ``[Strategy]backward``,
  ``optimizer_step``, and the validation hooks.  If ``train_dataloader_next`` is large relative to
  ``run_training_batch`` the run is input-bound; if the batch hooks dominate it is GPU-compute-bound.

* **CUDA-event region timer** (this file) — forward hooks on ``backbone`` / ``transformer`` /
  ``segmentation_head`` / ``criterion`` / ``matcher`` record per-region GPU time (same CUDA-event style as
  ``bench_backbone_latency.py``), so you see *where inside DINOv3* the GPU spends its time.  The matcher's
  cost is largely CPU (``scipy.linear_sum_assignment`` after a ``.cpu()`` sync); its GPU-event time is small
  by design, and the gap between ``training_step`` wall time and the summed region GPU time is the CPU-side
  cost.  Use ``--profiler pytorch`` for an op-level table that attributes that CPU time directly.

The build is wired by monkeypatching ``rfdetr.training.build_trainer`` to inject the profiler, the smoke-test
batch limits, and the region-timer callback — so we reuse ``model.train()``'s full setup (dataset adaptation,
num-classes alignment, weight loading) verbatim rather than re-implementing it.

Example (GPU 5, DINOv3 ViT-B, default smoke size):
    CUDA_VISIBLE_DEVICES=5 RF_HOME=/workspace/rf_home/models python scripts/profile_training.py \
        --dataset-dir datasets/DocLayNetReduced --model dinov3-base --device cuda:0 \
        --batch-size 16 --num-workers 16 --use-ema

    # A/B torch.compile (same flags + --compile), or op-level CPU detail (--profiler pytorch).
"""

import argparse
import os
import sys
from pathlib import Path

# Reuse the dataset adapter from the sibling training script (scripts/ is sys.path[0] when run as
# ``python scripts/profile_training.py``; insert explicitly so it also works from other CWDs).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rfdetr_dataset import adapt_to_roboflow_yolo

# Variant name -> constructor symbol (mirrors train_rfdetr_dataset.MODEL_CHOICES).
MODEL_CHOICES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "dinov3-base": "RFDETRDinov3Base",
}


def _build_region_timer_callback(warmup_steps: int):
    """Build a Lightning ``Callback`` that times model sub-regions with CUDA events.

    Imported lazily (inside ``main``) so ``--help`` does not require the training extras.

    Args:
        warmup_steps: Number of initial optimizer steps to skip before recording (lets CUDA caching
            allocator / cuDNN autotuner / any torch.compile graph settle).

    Returns:
        An instantiated ``RegionTimerCallback``.
    """
    import torch
    from pytorch_lightning import Callback

    class RegionTimerCallback(Callback):
        """Accumulate per-region GPU time over training steps via paired CUDA events.

        Hooks are registered on the live model's submodules in ``on_fit_start``.  Each region's
        forward-pre hook records a start event and its forward hook records an end event; the
        ``(start, end)`` pairs are summed once at fit end (a single ``synchronize``), so per-step
        timing is never perturbed by mid-loop syncs.  Recording is gated to training-step forwards
        only (``_in_train_step``), so validation / EMA-eval forwards are excluded.
        """

        # Top-level regions are mutually exclusive and partition the forward+loss.  "matcher" is a
        # *sub-region* of "criterion" (reported separately, labelled as such) and fires once per
        # decoder layer, so its time is summed across those calls within a step.
        _TOP_LEVEL = ("backbone", "transformer", "segmentation_head", "criterion")

        def __init__(self, warmup_steps: int) -> None:
            super().__init__()
            self._warmup = max(0, int(warmup_steps))
            self._regions = (*self._TOP_LEVEL, "matcher")
            self._events: dict[str, list] = {r: [] for r in self._regions}
            self._open: dict[str, object] = {}
            self._handles: list = []
            self._in_train_step = False
            self._recording = False
            self._step = 0
            self._recorded_steps = 0
            self._cuda = torch.cuda.is_available()

        def _pre_hook(self, region: str):
            def hook(module, inputs):
                if not (self._recording and self._in_train_step and self._cuda):
                    return
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                # A region's module may be re-entered (matcher fires per decoder layer): flush any
                # already-open span for it before opening a new one would lose the prior end, so we
                # only keep one open span per region and rely on post-hook to close it first.
                self._open[region] = start

            return hook

        def _post_hook(self, region: str):
            def hook(module, inputs, output):
                start = self._open.pop(region, None)
                if start is None:
                    return
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._events[region].append((start, end))

            return hook

        def on_fit_start(self, trainer, pl_module) -> None:
            # Unwrap torch.compile's OptimizedModule so hooks land on the real submodules.
            model = getattr(pl_module.model, "_orig_mod", pl_module.model)
            targets = {
                "backbone": getattr(model, "backbone", None),
                "transformer": getattr(model, "transformer", None),
                "segmentation_head": getattr(model, "segmentation_head", None),
                "criterion": getattr(pl_module, "criterion", None),
            }
            criterion = targets["criterion"]
            targets["matcher"] = getattr(criterion, "matcher", None) if criterion is not None else None
            for region, module in targets.items():
                if module is None:
                    continue
                self._handles.append(module.register_forward_pre_hook(self._pre_hook(region)))
                self._handles.append(module.register_forward_hook(self._post_hook(region)))

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            self._in_train_step = True

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            self._in_train_step = False
            self._step += 1
            if self._step >= self._warmup:
                self._recording = True
                self._recorded_steps += 1

        def on_fit_end(self, trainer, pl_module) -> None:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self.print_summary()

        def print_summary(self) -> None:
            if self._cuda:
                torch.cuda.synchronize()
            steps = max(1, self._recorded_steps)
            print("\n" + "=" * 70)
            print(f"  MODEL REGION GPU TIME  (CUDA events, mean over {steps} recorded train steps)")
            print("=" * 70)
            print(f"  {'region':<22}{'GPU ms/step':>14}{'calls/step':>14}")
            print("  " + "-" * 50)
            top_total = 0.0
            for region in self._regions:
                pairs = self._events[region]
                total_ms = sum(s.elapsed_time(e) for s, e in pairs)
                per_step = total_ms / steps
                calls = len(pairs) / steps
                if region in self._TOP_LEVEL:
                    top_total += per_step
                label = region if region in self._TOP_LEVEL else f"  └─ {region} (subset of criterion)"
                print(f"  {label:<22}{per_step:>14.2f}{calls:>14.2f}")
            print("  " + "-" * 50)
            print(f"  {'sum(top-level regions)':<22}{top_total:>14.2f}")
            print("=" * 70)
            print(
                "  Note: matcher GPU time is small by design — its real cost is the CPU\n"
                "  scipy.linear_sum_assignment after a .cpu() sync. Compare sum(top-level)\n"
                "  against the SimpleProfiler 'training_step' wall time below: the gap is\n"
                "  CPU-side (matcher + Python). Use --profiler pytorch to attribute it.\n"
            )

    return RegionTimerCallback(warmup_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="datasets/DocLayNetReduced", help="YOLO dataset root")
    parser.add_argument(
        "--adapted-dir", default=None, help="symlinked Roboflow-YOLO view (default: /tmp/<name>_rf_prof)"
    )
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="dinov3-base")
    parser.add_argument("--output-dir", default=None, help="profiler/checkpoint output (default: /tmp/<model>_profile)")
    parser.add_argument("--device", default="cuda:0", help="mask the physical GPU with CUDA_VISIBLE_DEVICES")
    parser.add_argument("--resolution", type=int, default=None, help="default: the variant's native resolution")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=60, help="limit_train_batches for the smoke test")
    parser.add_argument("--val-batches", type=int, default=20, help="limit_val_batches for the smoke test")
    parser.add_argument("--warmup-steps", type=int, default=10, help="region-timer steps to skip before recording")
    parser.add_argument("--aug-backend", default="cpu", choices=["cpu", "auto", "gpu"])
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--mal-loss", action="store_true")
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile (A/B the GPU-side speedup)")
    parser.add_argument(
        "--matcher-gpu-finite-check",
        action="store_true",
        help="Tier-1 matcher opt: run the cost-matrix finiteness check on-device before the .cpu() "
        "transfer (sets RFDETR_MATCHER_GPU_FINITE_CHECK=1). Run with and without to A/B the speedup.",
    )
    parser.add_argument(
        "--matcher-solver",
        default=None,
        choices=["scipy", "lap", "sinkhorn", "cuda_lap"],
        help="matcher assignment backend (sets RFDETR_MATCHER_SOLVER): scipy=exact CPU (default), "
        "lap=exact CPU (lapjv), sinkhorn=approx GPU, cuda_lap=exact GPU batched (torch-linear-assignment).",
    )
    parser.add_argument("--profiler", default="simple", choices=["simple", "pytorch"], help="PTL loop-level profiler")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Must be set before rfdetr is imported / the matcher is constructed (it reads the env at init).
    if args.matcher_gpu_finite_check:
        os.environ["RFDETR_MATCHER_GPU_FINITE_CHECK"] = "1"
    if args.matcher_solver is not None:
        os.environ["RFDETR_MATCHER_SOLVER"] = args.matcher_solver

    src = Path(args.dataset_dir).resolve()
    adapted_dir = Path(args.adapted_dir).resolve() if args.adapted_dir else Path(f"/tmp/{src.name}_rf_prof")
    output_dir = args.output_dir or f"/tmp/{args.model}_profile"
    num_classes = adapt_to_roboflow_yolo(src, adapted_dir)
    print(f"Adapted {src} -> {adapted_dir} ({num_classes} classes)", flush=True)

    if args.seed >= 0:
        import random

        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler

    import rfdetr
    import rfdetr.training as training_pkg

    on_gpu = args.device.startswith("cuda")
    init_kwargs = dict(num_classes=num_classes, amp=on_gpu, fused_optimizer=on_gpu, mal_loss=args.mal_loss)
    if args.resolution is not None:
        init_kwargs["resolution"] = args.resolution
    if args.compile:
        init_kwargs["compile"] = True
    model = getattr(rfdetr, MODEL_CHOICES[args.model])(**init_kwargs)

    # --- Profiler (loop-level) ---
    if args.profiler == "simple":
        profiler = SimpleProfiler(dirpath=output_dir, filename="ptl_simple_profile")
    else:
        # torch.profiler over a short active window; skips the first `train-batches`-bounded steps via
        # its own wait/warmup schedule. Emits a key_averages table (CPU + CUDA self time) + chrome trace.
        # Sort by total CPU time: the matcher bottleneck is CPU-side (scipy LAP + the cudaStreamSynchronize
        # forced by cost_matrix.cpu()), so cpu_time_total surfaces it at the top. (PTL validates this key
        # against torch's allowed set; "self_cuda_time_total" is not accepted by this PTL version.)
        profiler = PyTorchProfiler(
            dirpath=output_dir,
            filename="ptl_torch_profile",
            export_to_chrome=True,
            row_limit=25,
            sort_by_key="cpu_time_total",
        )

    region_cb = _build_region_timer_callback(args.warmup_steps)

    # --- Inject profiler + smoke limits + region callback by wrapping build_trainer ---
    # detr.train() does `from rfdetr.training import build_trainer` at call time, so patching the
    # package attribute is picked up. We keep all of build_trainer's real callbacks (EMA, COCO eval,
    # checkpointing) and only add the region timer + bound the run.
    _orig_build_trainer = training_pkg.build_trainer

    def _patched_build_trainer(train_config, model_config, **trainer_kwargs):
        trainer_kwargs.update(
            dict(
                profiler=profiler,
                limit_train_batches=args.train_batches,
                limit_val_batches=args.val_batches,
                max_epochs=1,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
            )
        )
        trainer = _orig_build_trainer(train_config, model_config, **trainer_kwargs)
        trainer.callbacks.append(region_cb)  # dispatched live from trainer.callbacks at hook time
        return trainer

    training_pkg.build_trainer = _patched_build_trainer

    print(
        f"Profiling {args.model} | classes={num_classes} | bs={args.batch_size} | workers={args.num_workers} | "
        f"compile={args.compile} | ema={args.use_ema} | aug={args.aug_backend} | profiler={args.profiler} | "
        f"matcher_gpu_finite_check={args.matcher_gpu_finite_check} | matcher_solver={args.matcher_solver or 'scipy'}",
        flush=True,
    )
    try:
        model.train(
            dataset_dir=str(adapted_dir),
            dataset_file="yolo",
            output_dir=output_dir,
            epochs=1,
            batch_size=args.batch_size,
            grad_accum_steps=1,
            num_workers=args.num_workers,
            lr=1e-4,
            warmup_epochs=0.0,
            device=args.device,
            multi_scale=args.multi_scale,
            use_ema=args.use_ema,
            augmentation_backend=args.aug_backend,
            tensorboard=False,
            checkpoint_interval=10_000,  # large => no periodic-archive checkpoint fires in a 1-epoch smoke
            seed=args.seed if args.seed >= 0 else None,
        )
    finally:
        training_pkg.build_trainer = _orig_build_trainer

    print(f"\nProfiler artifacts written under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
