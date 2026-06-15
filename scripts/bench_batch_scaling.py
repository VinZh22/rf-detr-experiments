# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Sweep inference batch size and report per-image latency / throughput.

Complements ``scripts/profile_inference.py`` (which fixes batch size and decomposes a single
predict() call) by answering the orthogonal question: how does latency-per-image and throughput
scale as the batch grows? Three measurement modes isolate different effects:

* ``raw_forward_fp32`` — ``model.model(batch)`` only, timed with CUDA events on a pre-built GPU
  batch. No Python/predict() overhead, no preprocessing — the cleanest view of GPU compute
  scaling for the detector itself.
* ``raw_forward_jit_fp16`` — same, but on the ``optimize_for_inference`` jit-traced fp16 module
  (re-traced per batch size, since the trace fixes the batch dimension).
* ``e2e_predict_numpy_uint8`` — full ``predict()`` on uint8 NumPy inputs (the post-fast-path
  realistic path), wall-clock timed. Includes per-image CPU decode/snapshot + GPU preprocessing.

For each batch size B and mode it reports ms/batch, ms/image (= ms/batch / B) and images/sec
(= 1000 * B / ms_batch). The ms/image curve shows the fixed-overhead amortization; the img/s
curve shows where the GPU saturates.

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/bench_batch_scaling.py \
        --checkpoint /workspace/doclaynet_ab/dinov3_small/checkpoint_best_total.pth \
        --model dinov3-small --images-dir datasets/DocLayNetReduced/val/images \
        --batch-sizes 1 2 4 8 16 32 64 128 --output batch_scaling_results.json
"""

import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image

import rfdetr as rfdetr_pkg

_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "dinov3-base": "RFDETRDinov3Base",
    "dinov3-small": "RFDETRDinov3Small",
}


def _stats(samples_ms: list[float]) -> dict[str, float]:
    """Summarize millisecond samples (median/mean/std/min)."""
    s = sorted(samples_ms)
    return {
        "median_ms": statistics.median(s),
        "mean_ms": statistics.mean(s),
        "std_ms": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min_ms": s[0],
        "n": len(s),
    }


def bench_wall(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    """Wall-clock benchmark with a CUDA sync inside the timed region."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return _stats(samples)


def bench_cuda_events(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    """CUDA-event benchmark (same convention as bench_backbone_latency.py)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return _stats(samples)


def _per_image(entry: dict[str, float], bs: int) -> dict[str, float]:
    """Augment a stats dict with per-image latency and throughput for batch size ``bs``."""
    med = entry["median_ms"]
    return {**entry, "ms_per_image": med / bs, "img_per_s": 1000.0 * bs / med}


def load_image_pool(images_dir: Path, count: int) -> list[np.ndarray]:
    """Load up to ``count`` images as uint8 HWC arrays, cycling if the directory is smaller."""
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    arrays = []
    for i in range(count):
        im = Image.open(paths[i % len(paths)])
        im.load()
        arrays.append(np.asarray(im))
    return arrays


def main() -> None:
    """Run the batch-size sweep and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="dinov3-small", choices=list(_MODEL_CLASSES))
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--skip-jit", action="store_true", help="Skip the jit-fp16 mode (faster sweep)")
    parser.add_argument("--output", type=Path, default=Path("batch_scaling_results.json"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required (set CUDA_VISIBLE_DEVICES to pick the GPU)"
    device = torch.device("cuda")

    model = getattr(rfdetr_pkg, _MODEL_CLASSES[args.model])(pretrain_weights=str(args.checkpoint))
    resolution = model.model.resolution
    # The module is built on CPU and only moved to the device lazily on the first predict();
    # the raw-forward modes call the module directly, so move it here.
    model.model.model = model.model.model.to(device)
    net = model.model.model
    net.eval()

    max_bs = max(args.batch_sizes)
    arrays = load_image_pool(args.images_dir, max_bs)
    means, stds = model.means, model.stds

    # Pre-build one preprocessed GPU batch of size max_bs; slice it for each batch size so the
    # raw-forward timing excludes all preprocessing and host transfer.
    pre = []
    for arr in arrays:
        t = torch.from_numpy(arr).to(device).permute(2, 0, 1).to(torch.get_default_dtype())
        t = t.div_(torch.tensor(255.0, device=device))
        t = tvf.resize(t, [resolution, resolution])
        t = tvf.normalize(t, means, stds)
        pre.append(t)
    full_batch = torch.stack(pre)
    torch.cuda.synchronize()

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "model_class": type(model).__name__,
        "resolution": resolution,
        "torch": torch.__version__,
        "batch_sizes": args.batch_sizes,
        "results": {},
    }
    print(f"GPU={report['gpu']} model={report['model_class']} res={resolution}", flush=True)
    print(f"{'bs':>4} {'mode':22} {'ms/batch':>10} {'ms/img':>9} {'img/s':>9}", flush=True)

    # Warm up cudnn/cublas autotuner once at the largest shape.
    with torch.no_grad():
        net(full_batch)
    torch.cuda.synchronize()

    for bs in args.batch_sizes:
        batch = full_batch[:bs].contiguous()
        entry: dict[str, Any] = {}

        # Mode 1: raw forward, eager fp32.
        def fwd() -> Any:
            with torch.no_grad():
                return net(batch)

        entry["raw_forward_fp32"] = _per_image(bench_cuda_events(fwd, args.warmup, args.iters), bs)

        # Mode 2: raw forward, jit-traced fp16 (re-trace per batch size).
        if not args.skip_jit:
            try:
                model.optimize_for_inference(compile=True, batch_size=bs, dtype=torch.float16)
                inf = model.model.inference_model
                batch_h = batch.to(torch.float16)

                def fwd_jit() -> Any:
                    with torch.no_grad():
                        return inf(batch_h)

                entry["raw_forward_jit_fp16"] = _per_image(bench_cuda_events(fwd_jit, args.warmup, args.iters), bs)
            except Exception as exc:
                entry["raw_forward_jit_fp16"] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                with contextlib.suppress(Exception):
                    model.remove_optimized_model()

        # Mode 3: full predict() e2e on uint8 numpy inputs (fast path).
        imgs = [arrays[i % len(arrays)] for i in range(bs)]

        def e2e() -> Any:
            return model.predict(imgs, threshold=args.threshold, include_source_image=False)

        entry["e2e_predict_numpy_uint8"] = _per_image(bench_wall(e2e, args.warmup, args.iters), bs)

        report["results"][str(bs)] = entry
        for mode, e in entry.items():
            if "error" in e:
                print(f"{bs:>4} {mode:22} {'ERROR':>10}", flush=True)
            else:
                print(
                    f"{bs:>4} {mode:22} {e['median_ms']:>10.2f} {e['ms_per_image']:>9.3f} {e['img_per_s']:>9.1f}",
                    flush=True,
                )

    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWritten to {args.output}", flush=True)


if __name__ == "__main__":
    main()
