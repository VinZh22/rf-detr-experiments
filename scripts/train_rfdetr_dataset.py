# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train any RF-DETR variant on any YOLO-format dataset.

Generalizes ``train_dinov3_coco128.py`` (which was hard-wired to COCO128 + the windowed
DINOv3 ViT-B backbone) along three axes:

* **dataset** — accepts both the Roboflow export layout (``train/{images,labels}`` +
  ``valid/...`` + ``test/...``) *and* the Ultralytics layout (``images/<split>`` +
  ``labels/<split>``). The class count is read from the dataset YAML, so it is dataset-agnostic.
* **variant** — ``--model`` selects any registered RF-DETR variant (DINOv2 ``nano/small/medium/
  base/large`` or the windowed-DINOv3 ``dinov3-base`` prototype). This lets the same harness run a
  DINOv3 arm and a DINOv2 baseline arm on identical data.
* **loss** — ``--mal-loss`` swaps DEIM's Matchability-Aware Loss in for the default IA-BCE.

RF-DETR's YOLO loader (:func:`rfdetr.datasets.yolo.build_roboflow_from_yolo`) requires the
Roboflow split names ``train``/``valid``/``test``; many datasets (COCO128, DocLayNet exports) ship
a ``val`` split and/or the Ultralytics ``images/<split>`` layout instead. Rather than fork the
well-tested loader, :func:`adapt_to_roboflow_yolo` builds a symlinked *view* in the Roboflow layout
(no image copying) and we train through the standard ``model.train(..., dataset_file="yolo")`` path.

DINOv3 weights are gated on the HF Hub (the account must accept the license; ``HF_TOKEN`` required).

``--multi-scale`` is intentionally omitted below: with RF-DETR's default ``square_resize_div_64=True``
and no random-resize, ``multi_scale`` collapses to a single *largest* scale (e.g. 736 px at res 576)
with no jitter — it just trains at a bigger fixed resolution. Passing a fixed ``--resolution`` keeps
train and val at the same size and avoids the ~1.6x compute surprise. Each arm gets its own
``--adapted-dir`` so the two processes never race on the same symlink view at startup.

Example:
    # DINOv3 ViT-B arm (GPU 1):
    CUDA_VISIBLE_DEVICES=1 RF_HOME=/workspace/rf_home/models python scripts/train_rfdetr_dataset.py \
        --dataset-dir datasets/DocLayNetReduced --model dinov3-base --device cuda \
        --epochs 12 --resolution 576 --batch-size 16 --num-workers 16 --use-ema \
        --adapted-dir /workspace/doclaynet_runs/dinov3_base/_rf \
        --output-dir /workspace/doclaynet_runs/dinov3_base

    # DINOv2 baseline arm (GPU 4):
    CUDA_VISIBLE_DEVICES=4 RF_HOME=/workspace/rf_home/models python scripts/train_rfdetr_dataset.py \
        --dataset-dir datasets/DocLayNetReduced --model medium --device cuda \
        --epochs 12 --resolution 576 --batch-size 16 --num-workers 16 --use-ema \
        --adapted-dir /workspace/doclaynet_runs/medium/_rf \
        --output-dir /workspace/doclaynet_runs/medium
"""

import argparse
import shutil
from pathlib import Path

import yaml

# Variant name -> (import symbol). Imported lazily inside main() so --help stays fast and does not
# require the training extras to be installed.
MODEL_CHOICES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "dinov3-base": "RFDETRDinov3Base",
    "dinov3-small": "RFDETRDinov3Small",
}


def adapt_to_roboflow_yolo(src: Path, dst: Path) -> int:
    """Build a Roboflow-YOLO-layout view of a YOLO dataset via directory symlinks.

    Reads the dataset YAML (``*.yaml``/``*.yml``), resolves each split's image directory relative to
    the YAML's location, derives the matching label directory by the YOLO ``images`` -> ``labels``
    convention, and links them under ``dst/{train,valid,test}/{images,labels}`` plus a ``data.yaml``
    carrying the class ``names`` (the only key RF-DETR's YOLO loader reads).

    Handles both common layouts transparently:

    * Roboflow export (``train/images``, ``val/images``, ...) — the ``val`` split is mapped to the
      Roboflow ``valid`` name the loader expects.
    * Ultralytics (``images/train``, ``images/val``, ...) — datasets where ``val`` may alias ``train``.

    Args:
        src: Source dataset root containing the ``*.yaml`` and the split image/label directories.
        dst: Destination root for the adapted (symlinked) Roboflow-YOLO view. Created if absent.

    Returns:
        Number of classes declared in the dataset YAML.

    Raises:
        FileNotFoundError: If no ``*.yaml``/``*.yml`` file is found under ``src`` or a resolved image
            directory does not exist.
    """
    yaml_path = next((p for p in sorted(src.glob("*.yaml")) + sorted(src.glob("*.yml"))), None)
    if yaml_path is None:
        raise FileNotFoundError(f"No data.yaml/.yml found under {src}")
    spec = yaml.safe_load(yaml_path.read_text())
    names = spec["names"]
    num_classes = len(names)

    # Roboflow "train"/"valid"/"test" splits <- dataset "train"/"val"/"test" image dirs (resolved
    # relative to the YAML dir). "val" is the conventional key; fall back to "valid" then "train".
    split_map = {
        "train": spec.get("train"),
        "valid": spec.get("val") or spec.get("valid") or spec.get("train"),
        "test": spec.get("test"),
    }
    for split, rel_images in split_map.items():
        if rel_images is None:
            continue
        images_src = (yaml_path.parent / rel_images).resolve()
        if not images_src.is_dir():
            raise FileNotFoundError(f"Resolved image dir for split {split!r} does not exist: {images_src}")
        # YOLO convention: the label dir mirrors the image dir with "images" -> "labels".
        labels_src = Path(str(images_src).replace("/images", "/labels"))
        split_dst = dst / split
        split_dst.mkdir(parents=True, exist_ok=True)
        # Whole-directory symlinks (not per-file): cheap and layout-agnostic. ``-sfn`` semantics.
        _relink(split_dst / "images", images_src)
        if labels_src.is_dir():
            _relink(split_dst / "labels", labels_src)

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(yaml_path, dst / "data.yaml")
    return num_classes


def _relink(link: Path, target: Path) -> None:
    """Create or replace ``link`` as a symlink to ``target`` (idempotent)."""
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="YOLO dataset root (Roboflow or Ultralytics layout)")
    parser.add_argument(
        "--adapted-dir",
        default=None,
        help="where to build the symlinked Roboflow-YOLO view (default: <dataset-dir>_rf next to source)",
    )
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="dinov3-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="epochs between periodic full checkpoints (~1.5 GB each); best_regular/best_ema are always kept",
    )
    parser.add_argument("--resolution", type=int, default=None, help="default: the variant's native resolution")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        default="cosine",
        choices=["step", "cosine"],
        help="'cosine' anneals over the whole run; 'step' (RF-DETR default) only drops after lr_drop=100 epochs, so it "
        "is a flat LR for short runs",
    )
    parser.add_argument(
        "--warmup-epochs", type=float, default=0.5, help="LR warmup; helps a from-scratch detector head"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seeded before model construction so the detector-head init is reproducible; pass <0 to leave unseeded",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="run validation/mAP every N epochs (the last epoch is always evaluated). >1 cuts the "
        "per-epoch validation cost on large val sets; throughput lever (no effect on training).",
    )
    parser.add_argument(
        "--compute-val-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-compute-val-loss skips the loss/matcher pass on validation batches (mAP is still "
        "computed); throughput lever for runs where only mAP is monitored.",
    )
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="path to a PTL .ckpt (e.g. <output-dir>/last.ckpt) to resume full training state "
        "(weights, optimizer, epoch, scheduler, EMA) from; None starts fresh.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="set pretrain_weights=None: train the detector head from random init on the variant's "
        "SSL backbone, instead of loading the released RF-DETR checkpoint. Use for fair backbone A/Bs.",
    )
    parser.add_argument("--mal-loss", action="store_true", help="use DEIM's Matchability-Aware Loss instead of IA-BCE")
    parser.add_argument(
        "--group-detr",
        type=int,
        default=None,
        help="number of Group DETR query groups (variant default if unset; set 1 to disable Group DETR, "
        "e.g. when relying on Dense O2O instead). Only safe with --from-scratch.",
    )
    parser.add_argument(
        "--dense-o2o",
        action="store_true",
        help="enable DEIM Dense O2O (mosaic+mixup) augmentation on the train split: densifies positive "
        "supervision on the data side instead of via Group DETR query groups",
    )
    parser.add_argument("--mosaic-prob", type=float, default=0.5, help="per-sample mosaic probability (Dense O2O)")
    parser.add_argument(
        "--mixup-prob", type=float, default=0.5, help="mixup probability given mosaic fired (Dense O2O)"
    )
    parser.add_argument(
        "--close-mosaic-epochs",
        type=int,
        default=5,
        help="trailing epochs with mosaic/mixup disabled so the model fine-tunes on clean images",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
        help="'muon' uses hybrid Muon (Newton-Schulz orthogonalized updates on 2D hidden weights, "
        "AdamW on embeddings/heads/1D params) with RMS-matched scaling so the same LRs apply",
    )
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--aug-backend",
        default="cpu",
        choices=["cpu", "auto", "gpu"],
        help="'gpu' (kornia) offloads resize/normalize/aug to the GPU; 'cpu' is best with many DataLoader workers",
    )
    args = parser.parse_args()

    src = Path(args.dataset_dir).resolve()
    adapted_dir = Path(args.adapted_dir).resolve() if args.adapted_dir else src.parent / f"{src.name}_rf"
    num_classes = adapt_to_roboflow_yolo(src, adapted_dir)
    print(f"Adapted {src} -> {adapted_dir} ({num_classes} classes)", flush=True)

    # Seed BEFORE constructing the model so the from-scratch detector-head init is pinned too (the
    # PTL seed_everything in on_fit_start only fires after construction). Same seed across both arms
    # gives a reproducible, comparable run; note a single seed still cannot separate a small mAP
    # delta from seed noise (run multiple seeds before trusting sub-~2-mAP differences).
    if args.seed is not None and args.seed >= 0:
        import random

        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    import rfdetr

    model_cls = getattr(rfdetr, MODEL_CHOICES[args.model])

    # AMP and the fused optimizer are GPU features; keep them off on CPU.
    on_gpu = args.device.startswith("cuda")
    # Construction kwargs shared by all variants. ``resolution`` is omitted when not given so each
    # variant keeps its native default (e.g. dinov3-base = 576, medium = 576).
    init_kwargs = dict(
        num_classes=num_classes,
        amp=on_gpu,
        fused_optimizer=on_gpu,
        mal_loss=args.mal_loss,
    )
    if args.resolution is not None:
        init_kwargs["resolution"] = args.resolution
    if args.group_detr is not None:
        # group_detr is an architecture field; changing it from the variant default is only
        # weight-compatible when training the head from scratch (no released checkpoint to load).
        init_kwargs["group_detr"] = args.group_detr
    if args.from_scratch:
        # pretrain_weights=None skips the released RF-DETR detector checkpoint and trains the detector
        # head from random init on top of the variant's self-supervised backbone (DINOv2/DINOv3).
        # Needed for a *fair backbone A/B*: dinov3-base is from-scratch by default, so the DINOv2
        # baseline (medium) must be too, instead of loading the COCO-pretrained rf-detr-medium.pth.
        init_kwargs["pretrain_weights"] = None
    model = model_cls(**init_kwargs)

    group_detr = args.group_detr if args.group_detr is not None else "variant-default"
    matching = f"DenseO2O(mosaic={args.mosaic_prob},mixup={args.mixup_prob})" if args.dense_o2o else "none"
    print(
        f"Training {args.model} ({MODEL_CHOICES[args.model]}) on {num_classes} classes | "
        f"loss={'MAL' if args.mal_loss else 'IA-BCE'} | group_detr={group_detr} | dense_o2o={matching} | "
        f"epochs={args.epochs} | bs={args.batch_size} | workers={args.num_workers} | "
        f"multi_scale={args.multi_scale} | ema={args.use_ema}",
        flush=True,
    )
    model.train(
        dataset_dir=str(adapted_dir),
        dataset_file="yolo",
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        lr=args.lr,
        optimizer=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        multi_scale=args.multi_scale,
        use_ema=args.use_ema,
        eval_interval=args.eval_interval,
        compute_val_loss=args.compute_val_loss,
        augmentation_backend=args.aug_backend,
        dense_o2o=args.dense_o2o,
        mosaic_prob=args.mosaic_prob,
        mixup_prob=args.mixup_prob,
        close_mosaic_epochs=args.close_mosaic_epochs,
        tensorboard=args.tensorboard,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        **({"seed": args.seed} if args.seed is not None and args.seed >= 0 else {}),
    )
    print(f"TRAINING RUN COMPLETE ({args.model}); checkpoints in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
