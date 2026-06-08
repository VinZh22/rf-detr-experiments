# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR with the windowed DINOv3 ViT-B/16 backbone (prototype).

By default this generates a tiny synthetic COCO dataset and runs a short CPU training
loop — a pipeline smoke test, not a convergence run. Point ``--dataset-dir`` at a real
COCO/Roboflow dataset (``train/`` and ``valid/`` each with ``_annotations.coco.json``) and
use a GPU box for a real run.

The DINOv3 backbone weights are gated on the HuggingFace Hub: the running account must have
accepted the DINOv3 License and a token must be available (``HF_TOKEN`` env var or
``hf auth login``).

``multi_scale`` and ``use_ema`` default to off here only to keep the CPU smoke test fast and
deterministic; both are safe and recommended for real convergence runs (enable with
``--multi-scale --use-ema``). Multi-scale is safe for the windowed backbone because
``compute_multi_scale_scales`` only ever returns sides that are multiples of
``patch_size * num_windows`` (16 * 2 = 32), and the DataLoader collate pads each batch up to
that same block size — so the backbone's divisibility requirement holds either way.

Example (real run on GPU):
    python scripts/train_dinov3_demo.py --dataset-dir /data/my-coco --output-dir runs/dinov3 \
        --epochs 100 --batch-size 4 --resolution 576 --device cuda --multi-scale --use-ema
"""

import argparse
import json
import os
from pathlib import Path


def build_synthetic_dataset(dataset_dir: Path, num_images: int, img_size: int) -> None:
    """Generate a tiny synthetic COCO dataset (train/ + valid/) if not already present."""
    if (dataset_dir / "train" / "_annotations.coco.json").exists():
        return
    from rfdetr.datasets.synthetic import DatasetSplitRatios, generate_coco_dataset

    generate_coco_dataset(
        output_dir=str(dataset_dir),
        num_images=num_images,
        img_size=img_size,
        class_mode="shape",
        min_objects=1,
        max_objects=3,
        split_ratios=DatasetSplitRatios(train=0.75, val=0.25, test=0.0),
    )
    # RF-DETR expects the validation split under "valid/"; generate_coco_dataset writes "val/".
    val_dir, valid_dir = dataset_dir / "val", dataset_dir / "valid"
    if val_dir.exists() and not valid_dir.exists():
        val_dir.rename(valid_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="/tmp/dinov3_coco_demo")
    parser.add_argument("--output-dir", default="/tmp/dinov3_train_out")
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="base input resolution; the collate step pads batches up to a multiple of "
        "patch_size*num_windows (32), so any positive value works (multiples of 32 avoid padding).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic-images", type=int, default=8)
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="enable multi-scale training (safe: scales are multiples of 32); recommended for real runs",
    )
    parser.add_argument(
        "--use-ema", action="store_true", help="enable EMA weights; recommended for real convergence runs"
    )
    args = parser.parse_args()

    from rfdetr import RFDETRDinov3Base

    dataset_dir = Path(args.dataset_dir)
    build_synthetic_dataset(dataset_dir, args.synthetic_images, img_size=max(64, args.resolution))

    with open(dataset_dir / "train" / "_annotations.coco.json") as f:
        num_classes = len(json.load(f)["categories"])

    # pretrain_weights defaults to None -> the windowed backbone initializes from the gated
    # facebook/dinov3-vitb16 weights; amp off for CPU. fused_optimizer is disabled because the
    # fused-AdamW gate probes CUDA bf16 support, which crashes on a stale/unusable GPU driver;
    # on a healthy GPU box you can leave it at its default (True).
    model = RFDETRDinov3Base(num_classes=num_classes, resolution=args.resolution, amp=False, fused_optimizer=False)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        num_workers=0,
        lr=args.lr,
        device=args.device,
        multi_scale=args.multi_scale,  # safe for the windowed backbone; off by default for a fast smoke test
        use_ema=args.use_ema,
        tensorboard=False,
        checkpoint_interval=1,
    )
    print("TRAINING RUN COMPLETE; checkpoints in", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
