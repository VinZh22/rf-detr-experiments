# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Profile end-to-end RF-DETR inference latency: preprocess + model forward + postprocess.

Unlike ``scripts/bench_backbone_latency.py`` (backbone-only) and ``scripts/profile_training.py``
(training loop), this script measures the *user-facing* ``RFDETR.predict()`` path, attributing
wall-clock time to every stage a caller actually pays for:

1. image decode / source-image snapshot (CPU, PIL)
2. ``F.to_tensor`` HWC uint8 -> CHW float32 (CPU)
3. pixel-range validation ``(img > 1).any()`` / ``(img < 0).any()``
4. host-to-device transfer
5. resize + normalize (GPU)
6. model forward (with per-region breakdown: ViT encoder / projector / transformer / heads)
7. ``PostProcess`` (sigmoid + top-k + box scaling, GPU)
8. threshold filter + device-to-host sync + ``sv.Detections`` construction

It then A/B-tests the acceleration levers available without code changes
(``optimize_for_inference`` jit-trace, fp16/bf16, batching) plus autocast on the raw forward.

Stage attribution inserts ``torch.cuda.synchronize()`` between stages, which slightly inflates
the stage *sum* relative to the un-instrumented end-to-end timing also reported (the e2e number
is the ground truth; the stage shares show where the time goes).

Example:
    CUDA_VISIBLE_DEVICES=4 python scripts/profile_inference.py \
        --checkpoint /workspace/doclaynet_ab/dinov3_small/checkpoint_best_total.pth \
        --images-dir datasets/DocLayNetReduced/val/images \
        --output inference_profile_results.json
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

from rfdetr import RFDETR


def _stats(samples_ms: list[float]) -> dict[str, float]:
    """Summarize a list of millisecond samples.

    Args:
        samples_ms: Raw per-iteration latencies in milliseconds.

    Returns:
        Dict with median, mean, std, p95, min and the sample count.
    """
    s = sorted(samples_ms)
    return {
        "median_ms": statistics.median(s),
        "mean_ms": statistics.mean(s),
        "std_ms": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "p95_ms": s[int(0.95 * (len(s) - 1))],
        "min_ms": s[0],
        "n": len(s),
    }


def bench_wall(fn: Callable[[int], Any], warmup: int, iters: int) -> dict[str, float]:
    """Wall-clock benchmark of ``fn`` (called with the iteration index).

    ``torch.cuda.synchronize()`` is called inside the timed region after ``fn`` so async GPU
    work is fully accounted for.

    Args:
        fn: Callable invoked once per iteration with the iteration index.
        warmup: Untimed warmup iterations.
        iters: Timed iterations.

    Returns:
        Latency statistics in milliseconds (see :func:`_stats`).
    """
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    samples = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(i)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return _stats(samples)


def bench_cuda_events(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    """CUDA-event benchmark for GPU-side work (same convention as bench_backbone_latency.py).

    Args:
        fn: Callable performing GPU work.
        warmup: Untimed warmup iterations.
        iters: Timed iterations.

    Returns:
        Latency statistics in milliseconds.
    """
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


def load_images(images_dir: Path, num_images: int) -> list[Path]:
    """Pick a deterministic sample of image paths from a directory.

    Args:
        images_dir: Directory containing ``.jpg``/``.png`` images.
        num_images: Number of images to use.

    Returns:
        Sorted list of selected image paths.
    """
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    step = max(1, len(paths) // num_images)
    return paths[::step][:num_images]


def section_e2e_by_input_type(
    model: RFDETR, image_paths: list[Path], threshold: float, warmup: int, iters: int, device: torch.device
) -> dict[str, Any]:
    """Benchmark full ``predict()`` for each supported input type at batch size 1."""
    results: dict[str, Any] = {}

    pils = []
    for p in image_paths:
        im = Image.open(p)
        im.load()
        pils.append(im)
    arrays = [np.asarray(im) for im in pils]
    gpu_tensors = [tvf.to_tensor(im).to(device) for im in pils]
    torch.cuda.synchronize()

    n = len(image_paths)
    results["path_str"] = bench_wall(
        lambda i: model.predict(str(image_paths[i % n]), threshold=threshold), warmup, iters
    )
    results["pil_preloaded"] = bench_wall(lambda i: model.predict(pils[i % n], threshold=threshold), warmup, iters)
    results["numpy_array"] = bench_wall(lambda i: model.predict(arrays[i % n], threshold=threshold), warmup, iters)
    results["gpu_tensor_no_source"] = bench_wall(
        lambda i: model.predict(gpu_tensors[i % n], threshold=threshold, include_source_image=False), warmup, iters
    )
    return results


def section_stage_breakdown(
    model: RFDETR, image_paths: list[Path], threshold: float, warmup: int, iters: int, device: torch.device
) -> dict[str, Any]:
    """Instrumented mirror of ``RFDETR.predict()`` (PIL path-input case) with per-stage timing.

    Mirrors src/rfdetr/detr.py:1300-1442 for a single path input with
    ``include_source_image=True`` (the default). Each stage boundary synchronizes CUDA so GPU
    work is attributed to the stage that launched it.
    """
    import supervision as sv

    resolution = model.model.resolution
    means, stds = model.means, model.stds
    stage_names = [
        "decode+source_snapshot (CPU)",
        "to_tensor (CPU)",
        "range_validation (CPU)",
        "H2D transfer",
        "resize+normalize (GPU)",
        "model_forward (GPU)",
        "postprocess (GPU)",
        "filter+D2H+Detections",
    ]
    acc: dict[str, list[float]] = {k: [] for k in stage_names}
    n = len(image_paths)

    def run(i: int, record: bool) -> sv.Detections:
        times: dict[str, float] = {}

        def timed(name: str, f: Callable[[], Any]) -> Any:
            t0 = time.perf_counter()
            out = f()
            torch.cuda.synchronize()
            times[name] = (time.perf_counter() - t0) * 1000
            return out

        path = image_paths[i % n]

        def _decode() -> tuple[Image.Image, np.ndarray]:
            im = Image.open(path)
            src = np.array(im)
            return im, src

        img, _src = timed(stage_names[0], _decode)
        tensor = timed(stage_names[1], lambda: tvf.to_tensor(img))

        def _validate() -> None:
            if (tensor > 1).any() or (tensor < 0).any():
                raise ValueError("image not in [0,1]")

        timed(stage_names[2], _validate)
        h, w = tensor.shape[1:]
        t_dev = timed(stage_names[3], lambda: tensor.to(device))

        def _resize_norm() -> torch.Tensor:
            x = tvf.resize(t_dev, [resolution, resolution])
            x = tvf.normalize(x, means, stds)
            return torch.stack([x])

        batch = timed(stage_names[4], _resize_norm)

        def _forward() -> dict[str, torch.Tensor]:
            with torch.no_grad():
                return model.model.model(batch)

        preds = timed(stage_names[5], _forward)

        def _postprocess() -> list[dict[str, torch.Tensor]]:
            with torch.no_grad():
                target_sizes = torch.tensor([(h, w)], device=device)
                return model.model.postprocess(preds, target_sizes=target_sizes)

        res = timed(stage_names[6], _postprocess)

        def _to_detections() -> sv.Detections:
            r = res[0]
            keep = r["scores"] > threshold
            return sv.Detections(
                xyxy=r["boxes"][keep].float().cpu().numpy(),
                confidence=r["scores"][keep].float().cpu().numpy(),
                class_id=r["labels"][keep].cpu().numpy(),
            )

        dets = timed(stage_names[7], _to_detections)
        if record:
            for k, v in times.items():
                acc[k].append(v)
        return dets

    for i in range(warmup):
        run(i, record=False)
    for i in range(iters):
        run(i, record=True)

    # Fidelity check: instrumented mirror must reproduce the real predict() output.
    ref = model.predict(str(image_paths[0]), threshold=threshold)
    mine = run(0, record=False)
    order_a, order_b = np.argsort(-ref.confidence), np.argsort(-mine.confidence)
    matches = (
        len(ref) == len(mine)
        and np.allclose(ref.xyxy[order_a], mine.xyxy[order_b], atol=1e-3)
        and np.allclose(ref.confidence[order_a], mine.confidence[order_b], atol=1e-5)
    )

    return {
        "stages": {k: _stats(v) for k, v in acc.items()},
        "matches_real_predict": bool(matches),
        "num_detections_sample": len(ref),
    }


def section_forward_regions(model: RFDETR, batch: torch.Tensor, warmup: int, iters: int) -> dict[str, Any]:
    """Forward-pass region breakdown via forward hooks + paired CUDA events.

    Regions: ViT encoder, multi-scale projector, transformer (two-stage head + decoder).
    The residual (heads, padding/flatten glue) is total minus the sum of regions.
    """
    net = model.model.model
    regions = {
        "vit_encoder": net.backbone[0].encoder,
        "projector": net.backbone[0].projector,
        "transformer": net.transformer,
    }
    pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {k: [] for k in regions}
    handles = []

    def make_pre(name: str) -> Callable:
        def pre_hook(module: torch.nn.Module, args: tuple) -> None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            pairs[name].append((ev, None))

        return pre_hook

    def make_post(name: str) -> Callable:
        def post_hook(module: torch.nn.Module, args: tuple, output: Any) -> None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            start, _ = pairs[name][-1]
            pairs[name][-1] = (start, ev)

        return post_hook

    for name, mod in regions.items():
        handles.append(mod.register_forward_pre_hook(make_pre(name)))
        handles.append(mod.register_forward_hook(make_post(name)))

    total = bench_cuda_events(lambda: net(batch), warmup, iters)
    torch.cuda.synchronize()
    for h in handles:
        h.remove()

    out: dict[str, Any] = {"total_forward": total}
    region_medians = {}
    for name, evs in pairs.items():
        timed_pairs = evs[-iters:]
        samples = [s.elapsed_time(e) for s, e in timed_pairs if e is not None]
        out[name] = _stats(samples)
        region_medians[name] = out[name]["median_ms"]
    out["heads_and_glue (residual)"] = {"median_ms": total["median_ms"] - sum(region_medians.values())}
    return out


def section_torch_profiler(model: RFDETR, image_path: Path, threshold: float) -> dict[str, Any]:
    """One torch.profiler pass over predict() to surface kernels and hidden syncs."""
    from torch.profiler import ProfilerActivity, profile

    model.predict(str(image_path), threshold=threshold)  # warm
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            model.predict(str(image_path), threshold=threshold)
    ka = prof.key_averages()
    sync_rows = [
        {"name": ev.key, "count": ev.count, "cpu_time_total_ms": ev.cpu_time_total / 1000}
        for ev in ka
        if "Synchronize" in ev.key or "Memcpy" in ev.key or "memcpy" in ev.key or ev.key == "aten::item"
    ]
    return {
        "top_cuda": ka.table(sort_by="self_cuda_time_total", row_limit=15),
        "top_cpu": ka.table(sort_by="self_cpu_time_total", row_limit=15),
        "sync_and_copy_rows": sync_rows,
    }


def section_acceleration(
    model: RFDETR,
    image_paths: list[Path],
    threshold: float,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    """A/B acceleration levers: autocast on raw forward; optimize_for_inference variants e2e."""
    results: dict[str, Any] = {}
    resolution = model.model.resolution
    net = model.model.model

    pils = []
    for p in image_paths:
        im = Image.open(p)
        im.load()
        pils.append(im)
    gpu_tensors = [tvf.to_tensor(im).to(device) for im in pils]
    batch = torch.stack([tvf.normalize(tvf.resize(gpu_tensors[0], [resolution, resolution]), model.means, model.stds)])
    torch.cuda.synchronize()
    n = len(gpu_tensors)

    def fwd_eager() -> Any:
        with torch.no_grad():
            return net(batch)

    results["forward_eager_fp32_tf32"] = bench_cuda_events(fwd_eager, warmup, iters)
    for dtype_name, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:

        def fwd_amp() -> Any:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                return net(batch)

        results[f"forward_eager_autocast_{dtype_name}"] = bench_cuda_events(fwd_amp, warmup, iters)

    def e2e(i: int) -> Any:
        return model.predict(gpu_tensors[i % n], threshold=threshold, include_source_image=False)

    results["e2e_baseline_eager_fp32"] = bench_wall(e2e, warmup, iters)

    for label, kwargs in [
        ("e2e_optimized_nocompile_fp32", {"compile": False, "dtype": torch.float32}),
        ("e2e_optimized_jit_fp32", {"compile": True, "batch_size": 1, "dtype": torch.float32}),
        ("e2e_optimized_jit_fp16", {"compile": True, "batch_size": 1, "dtype": torch.float16}),
    ]:
        try:
            model.optimize_for_inference(**kwargs)
            results[label] = bench_wall(e2e, warmup, iters)
        except Exception as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            with contextlib.suppress(Exception):
                model.remove_optimized_model()
    return results


def section_batch_scaling(
    model: RFDETR,
    image_paths: list[Path],
    threshold: float,
    warmup: int,
    iters: int,
    device: torch.device,
    batch_sizes: list[int],
) -> dict[str, Any]:
    """Throughput vs batch size for eager fp32 and jit fp16 predict()."""
    results: dict[str, Any] = {}
    pils = []
    for p in image_paths:
        im = Image.open(p)
        im.load()
        pils.append(im)
    gpu_tensors = [tvf.to_tensor(im).to(device) for im in pils]
    torch.cuda.synchronize()
    n = len(gpu_tensors)

    for bs in batch_sizes:

        def e2e_batch(i: int) -> Any:
            imgs = [gpu_tensors[(i * bs + j) % n] for j in range(bs)]
            return model.predict(imgs, threshold=threshold, include_source_image=False)

        eager = bench_wall(e2e_batch, warmup, iters)
        entry: dict[str, Any] = {
            "eager_fp32": {
                **eager,
                "ms_per_image": eager["median_ms"] / bs,
                "img_per_s": 1000 * bs / eager["median_ms"],
            }
        }
        try:
            model.optimize_for_inference(compile=True, batch_size=bs, dtype=torch.float16)
            jit = bench_wall(e2e_batch, warmup, iters)
            entry["jit_fp16"] = {
                **jit,
                "ms_per_image": jit["median_ms"] / bs,
                "img_per_s": 1000 * bs / jit["median_ms"],
            }
        except Exception as exc:
            entry["jit_fp16"] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            with contextlib.suppress(Exception):
                model.remove_optimized_model()
        results[f"batch_{bs}"] = entry
    return results


def main() -> None:
    """Run all profiling sections and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to RF-DETR training checkpoint (.pth)")
    parser.add_argument(
        "--model",
        default="dinov3-small",
        choices=["nano", "small", "medium", "base", "large", "dinov3-base", "dinov3-small"],
        help="Variant class to instantiate (RFDETR.from_checkpoint cannot resolve the dinov3 classes: "
        "they are missing from _CHECKPOINT_MODEL_NAME_CLASS_SYMBOLS)",
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of benchmark images")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--output", type=Path, default=Path("inference_profile_results.json"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required (set CUDA_VISIBLE_DEVICES to pick the GPU)"
    device = torch.device("cuda")
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(args.checkpoint),
        "torch": torch.__version__,
    }

    print(f"Loading {args.model} from {args.checkpoint} ...", flush=True)
    import rfdetr as rfdetr_pkg

    class_names = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
        "dinov3-base": "RFDETRDinov3Base",
        "dinov3-small": "RFDETRDinov3Small",
    }
    model = getattr(rfdetr_pkg, class_names[args.model])(pretrain_weights=str(args.checkpoint))
    report["model_class"] = type(model).__name__
    report["resolution"] = model.model.resolution

    image_paths = load_images(args.images_dir, args.num_images)
    sample = Image.open(image_paths[0])
    report["image_size_sample"] = sample.size
    print(f"Using {len(image_paths)} images from {args.images_dir} (sample size {sample.size})", flush=True)

    # Cold start: first predict pays the lazy CPU->GPU weight move + cudnn/cublas init.
    t0 = time.perf_counter()
    model.predict(str(image_paths[0]), threshold=args.threshold)
    report["cold_start_first_predict_ms"] = (time.perf_counter() - t0) * 1000
    print(f"Cold-start first predict: {report['cold_start_first_predict_ms']:.0f} ms", flush=True)

    print("\n[1/6] End-to-end predict() by input type ...", flush=True)
    report["e2e_by_input_type"] = section_e2e_by_input_type(
        model, image_paths, args.threshold, args.warmup, args.iters, device
    )
    print(json.dumps(report["e2e_by_input_type"], indent=2), flush=True)

    print("\n[2/6] Stage breakdown (path input, instrumented mirror) ...", flush=True)
    report["stage_breakdown"] = section_stage_breakdown(
        model, image_paths, args.threshold, args.warmup, args.iters, device
    )
    print(json.dumps(report["stage_breakdown"], indent=2), flush=True)

    print("\n[3/6] Forward-pass region breakdown ...", flush=True)
    im0 = Image.open(image_paths[0])
    batch0 = torch.stack(
        [
            tvf.normalize(
                tvf.resize(tvf.to_tensor(im0).to(device), [model.model.resolution] * 2), model.means, model.stds
            )
        ]
    )
    report["forward_regions"] = section_forward_regions(model, batch0, args.warmup, args.iters)
    print(json.dumps(report["forward_regions"], indent=2), flush=True)

    print("\n[4/6] torch.profiler pass ...", flush=True)
    prof_out = section_torch_profiler(model, image_paths[0], args.threshold)
    report["profiler_sync_and_copy_rows"] = prof_out["sync_and_copy_rows"]
    print(prof_out["top_cuda"], flush=True)
    print(prof_out["top_cpu"], flush=True)
    print(json.dumps(prof_out["sync_and_copy_rows"], indent=2), flush=True)

    print("\n[5/6] Acceleration A/B ...", flush=True)
    report["acceleration"] = section_acceleration(model, image_paths, args.threshold, args.warmup, args.iters, device)
    print(json.dumps(report["acceleration"], indent=2), flush=True)

    print("\n[6/6] Batch scaling ...", flush=True)
    report["batch_scaling"] = section_batch_scaling(
        model, image_paths, args.threshold, args.warmup, args.iters, device, args.batch_sizes
    )
    print(json.dumps(report["batch_scaling"], indent=2), flush=True)

    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
