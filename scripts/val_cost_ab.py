# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Wall-clock A/B for the validation-side throughput levers ``compute_val_loss`` and ``eval_interval``.

Validation in RF-DETR is expensive: with ``compute_val_loss=True`` it runs the full criterion (incl. the
Hungarian matcher) on every val batch, and with ``eval_interval=1`` it does so every epoch (see
``training_profiling.md`` — validation_step was ~592 ms/step in the original profile). These two knobs are
**pure throughput levers** — they change *how much* and *how often* you pay for validation, never the
training computation or the model weights.

This harness runs three short arms that are **identical except the val knobs**, so the wall-clock deltas
attribute each lever exactly (training is the same across arms → all differences are validation):

* ``baseline``      — ``compute_val_loss=True,  eval_interval=1``
* ``no_val_loss``   — ``compute_val_loss=False, eval_interval=1``   (isolates the val-loss/matcher pass)
* ``no_vl+ei=N``    — ``compute_val_loss=False, eval_interval=epochs`` (also skips intermediate evals)

Train is bounded small (we measure validation cost, not convergence); val uses a fixed subset.  EMA is off
to isolate the levers (EMA adds a *second* val forward pass on top, compounding the win).

Example (GPU 3):
    CUDA_VISIBLE_DEVICES=3 python scripts/val_cost_ab.py --dataset-dir datasets/DocLayNetReduced \
        --model dinov3-base --device cuda:0 --epochs 3 --train-batches 40 --val-batches 64
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


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_arm(
    label: str,
    compute_val_loss: bool,
    eval_interval: int,
    args: argparse.Namespace,
    adapted_dir: Path,
    num_classes: int,
) -> float:
    """Train one bounded arm with the given validation knobs and return wall-clock minutes."""
    _seed_everything(args.seed)
    import torch

    import rfdetr
    import rfdetr.training as training_pkg

    model = getattr(rfdetr, MODEL_CHOICES[args.model])(
        num_classes=num_classes, amp=True, fused_optimizer=True, resolution=args.resolution
    )

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
        return orig_build_trainer(train_config, model_config, **trainer_kwargs)

    training_pkg.build_trainer = _patched_build_trainer
    start = time.time()
    try:
        model.train(
            dataset_dir=str(adapted_dir),
            dataset_file="yolo",
            output_dir=f"{args.output_root}/{label}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            warmup_epochs=0.0,
            device=args.device,
            multi_scale=False,  # pin fixed resolution (see matcher_map_ab.py)
            use_ema=False,  # isolate the levers; EMA adds a second val forward on top
            eval_interval=eval_interval,
            compute_val_loss=compute_val_loss,
            tensorboard=False,
            checkpoint_interval=10_000,
            seed=args.seed,
        )
    finally:
        training_pkg.build_trainer = orig_build_trainer
    minutes = (time.time() - start) / 60.0
    del model
    torch.cuda.empty_cache()
    return minutes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="datasets/DocLayNetReduced")
    parser.add_argument("--adapted-dir", default=None)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="dinov3-base")
    parser.add_argument("--output-root", default="/tmp/val_cost_ab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batches", type=int, default=40, help="small: we measure val cost, not convergence")
    parser.add_argument("--val-batches", type=int, default=64, help="fixed val subset (batches) per eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.dataset_dir).resolve()
    adapted_dir = Path(args.adapted_dir).resolve() if args.adapted_dir else Path(f"/tmp/{src.name}_rf_valab")
    num_classes = adapt_to_roboflow_yolo(src, adapted_dir)
    print(
        f"val-cost A/B | model={args.model} res={args.resolution} | epochs={args.epochs} "
        f"train_batches={args.train_batches} val_batches={args.val_batches}",
        flush=True,
    )

    arms = [
        ("baseline", True, 1),
        ("no_val_loss", False, 1),
        (f"no_vl+ei{args.epochs}", False, args.epochs),
    ]
    minutes: dict[str, float] = {}
    for label, cvl, ei in arms:
        print(f"\n========== ARM: {label} (compute_val_loss={cvl}, eval_interval={ei}) ==========", flush=True)
        minutes[label] = run_arm(label, cvl, ei, args, adapted_dir, num_classes)
        print(f"[{label}] {minutes[label]:.2f} min", flush=True)

    # --- Attribution (train is identical across arms → deltas are pure validation) ---
    base = minutes["baseline"]
    no_vl = minutes["no_val_loss"]
    no_vl_ei = minutes[f"no_vl+ei{args.epochs}"]
    n_eval_base = args.epochs  # eval_interval=1 → one eval per epoch
    print("\n" + "=" * 70)
    print("  VALIDATION-COST A/B  (identical training; deltas are pure validation)")
    print("=" * 70)
    print(f"  {'arm':<20}{'minutes':>10}")
    print("  " + "-" * 30)
    for label, _, _ in arms:
        print(f"  {label:<20}{minutes[label]:>10.2f}")
    print("  " + "-" * 30)
    if n_eval_base:
        print(
            f"  compute_val_loss cost   : {(base - no_vl) / n_eval_base * 60:>6.1f} s / eval  "
            f"(matcher+loss pass over {args.val_batches} val batches)"
        )
    print(
        f"  per-eval cost (no vl)   : {(no_vl - no_vl_ei) / max(1, args.epochs - 1) * 60:>6.1f} s / eval  "
        f"(eval_interval skips these)"
    )
    saved = base - no_vl_ei
    print(
        f"  both levers, {args.epochs} epochs    : {saved:.2f} min saved "
        f"({saved / base * 100:.0f}% of baseline) — scales with val-set size and epoch count."
    )
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
