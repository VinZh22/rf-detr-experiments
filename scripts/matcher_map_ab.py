# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Convergence (mAP) A/B between matcher solvers — does the approximate Sinkhorn matcher hurt accuracy?

Trains the *same* model (same seed → identical init + data order) once per ``--solvers`` entry, changing
**only** the matcher backend (``RFDETR_MATCHER_SOLVER``), then reports val mAP/F1 on a fixed val subset so
the arms are directly comparable. This is the accuracy check that the speed A/B in ``training_profiling.md``
cannot answer: Sinkhorn + argmax rounding is approximate, so a throughput win is only real if mAP holds.

Bounded for a short wall-clock budget via ``--epochs`` / ``--train-batches`` (steps per epoch) /
``--val-batches`` (the shared val subset). Defaults target ~40 min total for two arms on one H200.

Both arms reuse ``model.train()``'s real pipeline; ``build_trainer`` is monkeypatched only to inject the
epoch/step bounds and to capture the trainer so the final ``val/mAP_50_95`` can be read back.

Example (GPU 3, scipy vs batched Sinkhorn, ~40 min):
    CUDA_VISIBLE_DEVICES=3 python scripts/matcher_map_ab.py \
        --dataset-dir datasets/DocLayNetReduced --model dinov3-base --device cuda:0 \
        --solvers scipy sinkhorn --epochs 4 --train-batches 250 --val-batches 40
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rfdetr_dataset import adapt_to_roboflow_yolo

MODEL_CHOICES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "dinov3-base": "RFDETRDinov3Base",
}
_METRIC_KEYS = ("val/mAP_50_95", "val/mAP_50", "val/mAP_75", "val/F1")


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_arm(solver: str, args: argparse.Namespace, adapted_dir: Path, num_classes: int) -> dict:
    """Train one arm with the given matcher solver and return its final val metrics.

    Args:
        solver: Matcher backend (``scipy`` / ``lap`` / ``sinkhorn``).
        args: Parsed CLI args.
        adapted_dir: Roboflow-YOLO view of the dataset.
        num_classes: Number of dataset classes.

    Returns:
        Mapping of metric name → value for the metrics in ``_METRIC_KEYS`` that were logged.
    """
    # Set before model construction: the matcher reads these at __init__.  gpu_finite_check is the
    # best scipy baseline and is ignored by sinkhorn, so it is safe to set for every arm.
    os.environ["RFDETR_MATCHER_SOLVER"] = solver
    os.environ["RFDETR_MATCHER_GPU_FINITE_CHECK"] = "1"
    _seed_everything(args.seed)  # identical init + data order across arms → only the matcher differs

    import torch

    import rfdetr
    import rfdetr.training as training_pkg

    model = getattr(rfdetr, MODEL_CHOICES[args.model])(
        num_classes=num_classes, amp=True, fused_optimizer=True, resolution=args.resolution
    )

    captured: dict = {}
    orig_build_trainer = training_pkg.build_trainer

    def _patched_build_trainer(train_config, model_config, **trainer_kwargs):
        trainer_kwargs.update(
            dict(
                max_epochs=args.epochs,
                limit_train_batches=args.train_batches,
                limit_val_batches=args.val_batches,
                num_sanity_val_steps=0,
            )
        )
        trainer = orig_build_trainer(train_config, model_config, **trainer_kwargs)
        captured["trainer"] = trainer
        return trainer

    training_pkg.build_trainer = _patched_build_trainer
    start = time.time()
    try:
        model.train(
            dataset_dir=str(adapted_dir),
            dataset_file="yolo",
            output_dir=f"{args.output_root}/{solver}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            lr_scheduler="cosine",
            warmup_epochs=args.warmup_epochs,
            device=args.device,
            # Pin a fixed resolution: TrainConfig defaults multi_scale=True, which (with
            # square_resize_div_64) collapses to a single *larger* scale (e.g. 736 px at res 576),
            # changing the compute regime and the time budget. Off ⇒ train at exactly --resolution,
            # matching the speed A/B conditions.
            multi_scale=False,
            use_ema=False,
            compute_val_loss=False,  # mAP is the signal; skip the extra matcher pass on val
            eval_interval=args.eval_interval,
            tensorboard=False,
            checkpoint_interval=10_000,  # avoid periodic ~1.5 GB archives in a short run
            seed=args.seed,
        )
    finally:
        training_pkg.build_trainer = orig_build_trainer
    elapsed = time.time() - start

    cm = captured["trainer"].callback_metrics
    metrics = {k: float(cm[k]) for k in _METRIC_KEYS if k in cm}
    metrics["_minutes"] = elapsed / 60.0
    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="datasets/DocLayNetReduced")
    parser.add_argument("--adapted-dir", default=None)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="dinov3-base")
    parser.add_argument("--output-root", default="/tmp/matcher_map_ab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--solvers", nargs="+", default=["scipy", "sinkhorn"], choices=["scipy", "lap", "sinkhorn"])
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=250, help="steps per epoch (bounds wall-clock)")
    parser.add_argument("--val-batches", type=int, default=40, help="shared val subset (batches) for the mAP estimate")
    parser.add_argument("--eval-interval", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.dataset_dir).resolve()
    adapted_dir = Path(args.adapted_dir).resolve() if args.adapted_dir else Path(f"/tmp/{src.name}_rf_mapab")
    num_classes = adapt_to_roboflow_yolo(src, adapted_dir)
    print(
        f"mAP A/B | model={args.model} classes={num_classes} res={args.resolution} | "
        f"epochs={args.epochs} steps/epoch={args.train_batches} val_batches={args.val_batches} | "
        f"solvers={args.solvers}",
        flush=True,
    )

    results: dict[str, dict] = {}
    for solver in args.solvers:
        print(f"\n========== ARM: {solver} ==========", flush=True)
        results[solver] = run_arm(solver, args, adapted_dir, num_classes)
        print(f"[{solver}] {results[solver]}", flush=True)

    # --- Comparison table ---
    print("\n" + "=" * 78)
    print("  MATCHER mAP A/B  (same seed/init/data; identical except the matcher solver)")
    print("=" * 78)
    header = f"  {'solver':<12}" + "".join(f"{k.split('/')[-1]:>12}" for k in _METRIC_KEYS) + f"{'minutes':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for solver in args.solvers:
        m = results[solver]
        row = f"  {solver:<12}" + "".join(f"{m.get(k, float('nan')):>12.4f}" for k in _METRIC_KEYS)
        row += f"{m.get('_minutes', float('nan')):>10.1f}"
        print(row)
    print("=" * 78)
    if "scipy" in results and "sinkhorn" in results:
        base = results["scipy"].get("val/mAP_50_95", float("nan"))
        sink = results["sinkhorn"].get("val/mAP_50_95", float("nan"))
        delta = sink - base
        rel = (delta / base * 100.0) if base else float("nan")
        print(
            f"  sinkhorn vs scipy mAP_50_95: {sink:.4f} vs {base:.4f}  "
            f"(Δ {delta:+.4f}, {rel:+.1f}%)  — speed-only adoption requires this to be ≈ 0 or positive.",
            flush=True,
        )


if __name__ == "__main__":
    main()
