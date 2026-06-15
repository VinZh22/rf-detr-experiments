# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR (windowed DINOv3 ViT-B/16) on COCO128.

COCO128 ships in the Ultralytics YOLO layout (``images/<split>``, ``labels/<split>``,
``coco128.yaml``, with ``val`` pointing at the same dir as ``train``). RF-DETR's YOLO loader
instead expects the Roboflow export layout (``data.yaml`` + ``train/{images,labels}`` +
``valid/{images,labels}``). Rather than fork the (well-tested) loader, ``adapt_ultralytics_to_roboflow_yolo``
builds that layout as a view of the source dataset using symlinks (no image copying), then we
train through the standard ``RFDETRDinov3Base.train(..., dataset_file="yolo")`` path.

DINOv3 weights are gated on the Hub (account must accept the license; ``HF_TOKEN`` required).

Example:
    python scripts/train_dinov3_coco128.py --epochs 1 --resolution 64 --device cpu
    # GPU convergence run:
    python scripts/train_dinov3_coco128.py --epochs 100 --resolution 640 --device cuda --multi-scale --use-ema
"""

import argparse
import shutil
from pathlib import Path

import yaml

# NOTE: this container's /dev/shm is only 64 MB, which makes PyTorch DataLoader workers crash
# with "unable to allocate shared memory" (neither the fd nor file_system sharing strategy
# avoids it here). Use --num-workers 0, which does no inter-process tensor sharing.


def adapt_ultralytics_to_roboflow_yolo(src: Path, dst: Path) -> int:
    """Build a Roboflow-YOLO-layout view of an Ultralytics YOLO dataset via symlinks.

    Resolves the ``train``/``val`` image dirs from the dataset YAML (relative to the YAML's
    directory), derives the matching label dirs by the Ultralytics ``images`` -> ``labels``
    convention, and links them under ``dst/{train,valid}/{images,labels}`` plus a ``data.yaml``
    carrying the class ``names`` (the only key RF-DETR's YOLO loader reads).

    Args:
        src: Source dataset root containing the ``*.yaml`` and ``images/``/``labels/`` dirs.
        dst: Destination root for the adapted (symlinked) Roboflow-YOLO view.

    Returns:
        Number of classes declared in the dataset YAML.
    """
    yaml_path = next(p for p in src.glob("*.yaml"))
    spec = yaml.safe_load(yaml_path.read_text())
    names = spec["names"]
    num_classes = len(names)

    # Roboflow "train"/"valid" splits <- Ultralytics "train"/"val" image dirs (resolved
    # relative to the YAML dir; COCO128 sets val == train).
    split_map = {"train": spec["train"], "valid": spec.get("val", spec["train"])}
    for split, rel_images in split_map.items():
        images_src = (yaml_path.parent / rel_images).resolve()
        labels_src = Path(str(images_src).replace("/images", "/labels"))
        images_dst, labels_dst = dst / split / "images", dst / split / "labels"
        images_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)
        for image in sorted(images_src.glob("*.jpg")):
            link = images_dst / image.name
            if not link.exists():
                link.symlink_to(image)
        for label in sorted(labels_src.glob("*.txt")):
            link = labels_dst / label.name
            if not link.exists():
                link.symlink_to(label)

    shutil.copy(yaml_path, dst / "data.yaml")
    return num_classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/workspace/rf-detr-experiments/datasets/coco128")
    parser.add_argument("--adapted-dir", default="/tmp/coco128_rf")
    parser.add_argument("--output-dir", default="/tmp/coco128_dinov3_out")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="epochs between periodic full checkpoints (~1.5 GB each); best_regular/best_ema are always kept",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--mal-loss", action="store_true", help="use DEIM's Matchability-Aware Loss instead of IA-BCE")
    parser.add_argument(
        "--aug-backend",
        default="cpu",
        choices=["cpu", "auto", "gpu"],
        help="'gpu' (kornia) offloads resize/normalize/aug to the GPU — much faster when /dev/shm forces num_workers=0",
    )
    args = parser.parse_args()

    num_classes = adapt_ultralytics_to_roboflow_yolo(Path(args.src), Path(args.adapted_dir))
    print(f"Adapted {args.src} -> {args.adapted_dir} ({num_classes} classes)")

    from rfdetr import RFDETRDinov3Base

    # AMP and the fused optimizer are GPU features; keep them off on CPU.
    on_gpu = args.device.startswith("cuda")
    model = RFDETRDinov3Base(
        num_classes=num_classes,
        resolution=args.resolution,
        amp=on_gpu,
        fused_optimizer=on_gpu,
        mal_loss=args.mal_loss,
    )
    model.train(
        dataset_dir=args.adapted_dir,
        dataset_file="yolo",
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=1,
        num_workers=args.num_workers,
        lr=args.lr,
        device=args.device,
        multi_scale=args.multi_scale,
        use_ema=args.use_ema,
        augmentation_backend=args.aug_backend,
        tensorboard=False,
        checkpoint_interval=args.checkpoint_interval,
    )
    print("COCO128 TRAINING RUN COMPLETE; checkpoints in", args.output_dir)


if __name__ == "__main__":
    main()
