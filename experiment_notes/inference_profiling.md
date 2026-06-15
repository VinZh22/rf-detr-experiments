# RF-DETR Inference Profiling — where the time goes and how to make it faster

**Setup:** `RFDETRDinov3Small` (DocLayNet checkpoint `/workspace/doclaynet_ab/dinov3_small/checkpoint_best_total.pth`, 11 classes, resolution 576), H200 GPU (`CUDA_VISIBLE_DEVICES=4`), torch 2.10.0+cu128, DocLayNet val images (1025×1025 JPEG), threshold 0.5, 15 warmup / 50 timed iters, medians reported. Reproduce with `scripts/profile_inference.py`:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/profile_inference.py \
    --checkpoint /workspace/doclaynet_ab/dinov3_small/checkpoint_best_total.pth \
    --model dinov3-small --images-dir datasets/DocLayNetReduced/val/images
```

> **Caveat:** two training runs were saturating ~93/128 CPU cores during measurement, so the CPU-stage numbers are somewhat inflated vs an idle box. GPU 4 itself was idle. The *structure* of the result (CPU preprocessing ≫ GPU forward) is robust — the heavy stages are O(megapixel) single-image CPU ops that cost tens of ms even unloaded.

## Headline: `predict()` is ~75–80% CPU preprocessing, not model forward

End-to-end `predict()` latency at batch 1 by input type:

| Input type | e2e median | img/s |
| --- | --- | --- |
| file path (str) | 85.8 ms | 11.7 |
| PIL (pre-loaded) | 83.4 ms | 12.0 |
| numpy uint8 array | 79.0 ms | 12.7 |
| GPU tensor (`[0,1]` float) + `include_source_image=False` | 16.1 ms | 62 |

Stage breakdown (path input, instrumented mirror of `predict()`, verified to reproduce the real output exactly; stages sum to 80 ms):

| Stage | median | share | where |
| --- | --- | --- | --- |
| **range validation `(img>1).any()` / `(img<0).any()`** | **35.1 ms** | **44%** | CPU, `detr.py:1323-1330` |
| **`F.to_tensor` (uint8 HWC → float32 CHW ÷255)** | **22.7 ms** | **28%** | CPU, `detr.py:1319` |
| model forward | 14.8 ms | 18% | GPU |
| decode + `source_image` snapshot (`np.array(img)`) | 4.9 ms | 6% | CPU, `detr.py:1315` |
| H2D transfer (12.6 MB float32) | 1.5 ms | 2% | `detr.py:1342` |
| resize + normalize | 0.4 ms | 0.5% | GPU, `detr.py:1344-1345` |
| PostProcess (sigmoid + top-300 + box scaling) | 0.41 ms | 0.5% | GPU |
| threshold filter + D2H + `sv.Detections` | 0.26 ms | 0.3% | |

Pre-process ≈ 64 ms, forward ≈ 15 ms, post-process < 1 ms. The two dominant stages are full passes over the 1025×1025×3 float32 image on the CPU; both happen *before* the GPU sees anything.

Forward-pass internals (CUDA-event hooks, eager fp32+TF32, 15.9 ms total): ViT encoder 7.4 ms (47%), transformer (two-stage head + 4 deformable decoder layers) 5.6 ms (35%), heads/glue 1.8 ms, projector 1.0 ms. torch.profiler shows ~17 `cudaStreamSynchronize` + 8 `aten::item` per predict (the per-decoder-layer Python `assert` in `MSDeformAttn.forward` (`ms_deform_attn.py:132`) plus the range checks), but at batch 1 these cost ~1 ms total — real, not dominant.

Cold start (lazy weight H2D + cuDNN/cuBLAS init on first predict): 1.8 s.

## Acceleration A/B results (measured)

| Config | forward | e2e (GPU-tensor input) |
| --- | --- | --- |
| eager fp32 (TF32 on, default) | 13.7 ms | 17.1 ms |
| eager + autocast bf16 | 17.9 ms ❌ slower | — |
| eager + autocast fp16 | 17.7 ms ❌ slower | — |
| eager bf16 *weights* (no autocast) | 12.5 ms (= fp32) | — |
| `optimize_for_inference(compile=False)` (export path) | — | 16.5 ms |
| `optimize_for_inference()` jit-trace fp32 | — | **10.7 ms** |
| `optimize_for_inference(dtype=fp16)` jit-trace | — | 11.2 ms |

Precision does nothing here: the model is small (32.9 M params, 576px) and kernel-launch-bound on an H200, and TF32 already runs the matmuls on tensor cores. Autocast actively hurts (per-op cast overhead). The jit-trace win (−37% e2e) comes from fusion/Python-overhead removal, not dtype.

Batch scaling (`predict()` with a list of images, per-image throughput):

| Batch | eager fp32 | jit fp16 |
| --- | --- | --- |
| 1 | 54 img/s | 92 img/s |
| 4 | 194 img/s | 297 img/s |
| 8 | 303 img/s | 551 img/s |
| 16 | 397 img/s | **752 img/s** |

## Verified prototype: GPU-side preprocessing

`F.to_tensor`'s uint8 path divides by 255, so the output is *mathematically guaranteed* to be in `[0,1]` — the 35 ms range scan proves nothing for PIL/uint8-numpy inputs (it only protects float/tensor inputs). Prototype: ship the uint8 bytes to the GPU (3 MB instead of 12 MB) and do convert/scale/resize/normalize there, skipping validation for uint8 provenance:

```python
t = torch.from_numpy(arr).to(device, non_blocking=True)   # uint8 HWC
t = t.permute(2, 0, 1).float().div_(255)
t = tvf.resize(t, [res, res])
batch = ((t - means) / stds).unsqueeze(0)
```

Measured (numpy uint8 input, same images, parity with `predict()` output verified — identical 15 detections, boxes within 0.5 px):

| Pipeline | e2e median | vs baseline |
| --- | --- | --- |
| current `predict(numpy)` | 79.0 ms | 1× |
| GPU preprocess + eager forward | 14.1 ms | **5.6×** |
| GPU preprocess + jit fp32 forward | **7.1 ms** | **11×** |

Preprocessing itself drops from ~63 ms (CPU) to **0.52 ms** (GPU).

## Recommendations, ranked by measured impact

1. **Move preprocessing to GPU for uint8 inputs and skip the range scan when provenance is uint8** (`detr.py:1313-1345`). 79 → 14 ms e2e; the single biggest lever (5.6×). For float/tensor inputs, keep validation but fuse the two `.any()` passes into one `torch.aminmax`.
2. **Call `optimize_for_inference()`** (already in the public API, just not on by default): another ~2× on the forward → 7.1 ms e2e combined (11×). Fixed batch size + resolution after tracing; use `compile=False` if you need flexible shapes (worth only ~0.6 ms).
3. **Batch requests** when throughput matters: 752 img/s at batch 16 (jit) vs 12 img/s with today's single-image path-input flow.
4. **Pass `include_source_image=False`** when the source array isn't needed: saves the ~5 ms `np.array(img)` snapshot + 3 MB/image.
5. **Don't bother with fp16/bf16/autocast on H200** for this model — measured zero-to-negative gain; TF32 fp32 is already optimal. (May differ on smaller GPUs.)
6. Minor: replace the Python `assert` in `MSDeformAttn.forward` (`ms_deform_attn.py:132`) with `torch._assert` at eval (the export path already does this) — removes 4 GPU→CPU syncs/forward, ~0.5–1 ms at batch 1, more under concurrency.

## Update (2026-06-12): recommendation 1 implemented and A/B-verified

`predict()` now ships uint8 PIL/NumPy pixels to the device as raw bytes (4× less H2D traffic), does the `÷255` + permute there, and skips the range scan for uint8 provenance; float/tensor inputs keep validation as a single fused `torch.aminmax` pass (with a NaN-aware fallback preserving the old elementwise semantics). Two parity subtleties found and fixed during verification:

- CUDA's scalar-division kernel multiplies by the reciprocal, rounding differently from CPU division for 126/256 uint8 values (1 ulp) — enough to chaotically shift borderline confidences by up to 0.12 through TF32 matmuls. Dividing by a 0-dim *tensor* uses true IEEE division; verified bit-exact over all 256 uint8 values.
- `to_tensor` honors `torch.get_default_dtype()`; the fast path now does too.

A/B on GPU 3 (same 16 DocLayNet val images, threshold 0.4, 282 detections): **outputs bit-identical** for all five input types (max box/confidence diff 0.00, classes equal). Latency (median, before → after): numpy uint8 **117.7 → 18.8 ms (6.3×)**, PIL **125.3 → 19.7 ms (6.4×)**, file path **115.1 → 23.8 ms (4.8×)**, float32 numpy 98.5 → 43.8 ms (2.3×), CPU tensor 73.7 → 31.9 ms (2.3×), GPU tensor unchanged. Tests: `tests/models/test_predict.py::TestPredictUint8FastPath` (12 tests); full `tests/models/` suite green.

## Batch-size sweep (2026-06-12, GPU 3, idle box)

How per-image latency and throughput scale with batch size, via `scripts/bench_batch_scaling.py` (medians, 10 warmup / 40 iters). Three modes isolate the effects: **raw forward** (`model.model(batch)` only, CUDA-event timed on a pre-built GPU batch — pure GPU compute, no preprocessing or Python overhead) in eager fp32 and jit-fp16, and **e2e `predict()`** on uint8 NumPy (realistic, includes per-image CPU decode + GPU preprocess).

| bs | raw fp32 ms/img | raw fp32 img/s | jit fp16 ms/img | jit fp16 img/s | e2e ms/img | e2e img/s |
|---|---|---|---|---|---|---|
| 1 | 12.66 | 79 | 6.15 | 163 | 16.52 | 61 |
| 2 | 6.71 | 149 | 3.22 | 310 | 9.00 | 111 |
| 4 | 3.36 | 298 | 1.94 | 515 | 4.98 | 201 |
| 8 | 2.27 | 440 | 0.81 | 1241 | 3.54 | 283 |
| 16 | 1.80 | 555 | 0.67 | 1493 | 2.91 | 344 |
| 32 | 1.54 | 648 | 0.61 | 1644 | 2.56 | 390 |
| 64 | 1.42 | 707 | 0.58 | 1733 | 2.50 | 400 |
| 128 | 1.38 | 726 | 0.56 | 1781 | 2.45 | 409 |

Raw data: `batch_scaling_results.json`.

**Two regimes, and a wide "free" zone.** The model is small enough to be launch-bound at low batch: processing 2–4 images costs the *same wall-clock* as 1 (eager fp32 total batch time: bs1 12.66 ms → bs4 13.43 ms, i.e. **4× the work for +0.1%** time). The jit-fp16 model is free out to **batch 8** (bs1 6.15 → bs8 6.45 ms total). Past the knee (≈bs4 eager, ≈bs8 jit) it turns compute-bound: each doubling adds ~+60→95% time, so per-image cost flattens and throughput saturates.

**Per-image latency collapses ~9–11×, then plateaus.** Raw fp32: 12.66 → 1.38 ms/img (9.2×). jit fp16: 6.15 → 0.56 ms/img (11×). Almost all of the gain is captured by **bs8–16**; bs32→128 adds <15% more. Throughput plateaus at ~726 img/s (eager fp32) and ~1781 img/s (jit fp16) on one H200.

**E2e plateaus much lower (~409 img/s) — the bottleneck moved to the CPU.** After batching amortizes the GPU, the residual ~1.07 ms/img gap between e2e (2.45 ms/img) and raw fp32 (1.38 ms/img) at bs128 is per-image CPU work (PIL decode, source-image handling, the Python preprocessing loop), which runs sequentially and does *not* amortize with batch size. To push e2e past ~400 img/s you'd parallelize decode across workers or pass pre-decoded GPU tensors, not grow the batch.

**Practical guidance:** for latency-sensitive single requests, bs1 jit-fp16 (~6 ms). For throughput, **bs8–16 is the sweet spot** (80–90% of peak img/s at a fraction of the batch latency and memory); bs≥32 only helps offline bulk jobs. Numbers are for ViT-S @576 on H200 — larger backbones/resolutions hit the compute-bound knee at smaller batch.

## Bug found along the way

`RFDETR.from_checkpoint()` cannot load DINOv3 checkpoints even though they store `model_name='RFDETRDinov3Small'`: the DINOv3 classes are missing from `_CHECKPOINT_MODEL_NAME_CLASS_SYMBOLS` (`detr.py:336-360`). Workaround: instantiate `RFDETRDinov3Small(pretrain_weights=...)` directly (what `scripts/profile_inference.py --model dinov3-small` does).

Raw numbers: [`inference_profile_results.json`](inference_profile_results.json) (this folder). A fresh
`scripts/profile_inference.py` run writes it to the current working directory by default (`--output`).
