# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Quick latency comparison: windowed DINOv2 vs windowed DINOv3 backbones.

Controlled apples-to-apples: same size, resolution, num_windows, AND patch size, so the
comparison isolates the architecture (DINOv2 abs-pos blocks vs DINOv3 RoPE blocks + the
windowing implementation) rather than just token count. Weights are random (latency does
not depend on weights), so no gated download is needed.

Note: DINOv2 is natively patch-14 and DINOv3 patch-16; at a fixed resolution DINOv2 would
otherwise have ~(16/14)^2 ≈ 1.3x more tokens. Forcing both to patch-16 removes that, leaving
only the architectural delta.
"""

import argparse
import statistics

import torch

from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3


def _time(fn, *, warmup: int, iters: int, device: str) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for callable ``fn`` with CUDA-event timing."""
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        import time

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), statistics.pstdev(times)


def _benchmark(name: str, model, x, *, device: str, amp: bool, warmup: int, iters: int) -> dict:
    model.eval()
    autocast = torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=amp
    )

    def fwd():
        with torch.no_grad(), autocast:
            model(x)

    fwd_mean, fwd_std = _time(fwd, warmup=warmup, iters=iters, device=device)

    model.train()

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        with autocast:
            feats = model(x)
            loss = sum(f.float().mean() for f in feats)
        loss.backward()

    bwd_mean, bwd_std = _time(fwd_bwd, warmup=warmup, iters=iters, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "name": name,
        "params_M": n_params / 1e6,
        "fwd_ms": fwd_mean,
        "fwd_std": fwd_std,
        "fwd_bwd_ms": bwd_mean,
        "fwd_bwd_std": bwd_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", default="base", choices=["small", "base", "large"])
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--num-windows", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=16, help="DINOv3 patch size (also DINOv2 unless overridden)")
    parser.add_argument(
        "--dinov2-patch-size",
        type=int,
        default=None,
        help="override DINOv2 patch size for a native comparison (e.g. 14); res must divide patch*num_windows",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    ofi = [3, 6, 9, 12] if args.size != "large" else [6, 12, 18, 24]
    v2_patch = args.dinov2_patch_size or args.patch_size
    base = dict(
        shape=(args.resolution, args.resolution),
        out_feature_indexes=ofi,
        size=args.size,
        use_windowed_attn=True,
        num_windows=args.num_windows,
    )
    # use_registers=False matches RF-DETR's shipped dinov2_windowed_* encoders; the ~4 register
    # tokens DINOv3 carries are negligible vs ~1600 patch tokens for a latency comparison.
    dinov2 = DinoV2(
        **base,
        patch_size=v2_patch,
        use_registers=False,
        load_dinov2_weights=False,
        positional_encoding_size=args.resolution // v2_patch,
    ).to(args.device)
    dinov3 = DinoV3(**base, patch_size=args.patch_size, load_dinov3_weights=False).to(args.device)

    x = torch.randn(args.batch_size, 3, args.resolution, args.resolution, device=args.device)
    v2_tok = (args.resolution // v2_patch) ** 2
    v3_tok = (args.resolution // args.patch_size) ** 2
    print(
        f"size={args.size} res={args.resolution} num_windows={args.num_windows} batch={args.batch_size} "
        f"amp={'bf16' if not args.no_amp else 'off'} | "
        f"DINOv2 patch={v2_patch} ({v2_tok} tok), DINOv3 patch={args.patch_size} ({v3_tok} tok)"
    )

    results = []
    for name, model in [("DINOv2-windowed", dinov2), ("DINOv3-windowed", dinov3)]:
        results.append(
            _benchmark(name, model, x, device=args.device, amp=not args.no_amp, warmup=args.warmup, iters=args.iters)
        )

    print(f"\n{'backbone':<18}{'params(M)':>11}{'fwd ms':>14}{'fwd+bwd ms':>16}")
    for r in results:
        print(
            f"{r['name']:<18}{r['params_M']:>11.1f}"
            f"{r['fwd_ms']:>9.2f}±{r['fwd_std']:<4.2f}{r['fwd_bwd_ms']:>11.2f}±{r['fwd_bwd_std']:<4.2f}"
        )
    v2, v3 = results
    print(
        f"\nDINOv3 vs DINOv2:  fwd {v3['fwd_ms'] / v2['fwd_ms']:.2f}x   "
        f"fwd+bwd {v3['fwd_bwd_ms'] / v2['fwd_bwd_ms']:.2f}x  (>1 = DINOv3 slower)"
    )


if __name__ == "__main__":
    main()
