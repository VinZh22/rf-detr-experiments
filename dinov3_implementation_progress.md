# Windowed DINOv3 Backbone — Implementation Progress

Tracker for swapping RF-DETR's windowed DINOv2 backbone for a windowed **DINOv3 ViT** backbone.
Design rationale and the Option-A derivation live in [`dinov3_windowed_prototype.md`](dinov3_windowed_prototype.md).

**Status:** prototype complete through Phase 3 (trains end-to-end). All work is currently
**untracked on `develop`** — not yet committed. Scope is **ViT-S / ViT-B (plain MLP)** only.

Legend: ✅ done · 🔜 next · ⬜ not started

---

## Phase 0 — Feasibility & design ✅
- ✅ Confirmed `transformers` ships DINOv3 (`dinov3_vit`) with `AutoBackbone`.
- ✅ Identified the core problem: DINOv2 adds a learned absolute position table once before the
  window fold; DINOv3 has none — only per-layer **RoPE inside attention**.
- ✅ Chose **Option A**: apply RoPE per-window in the folded layout, then merge q/k/v across
  windows *after* rotation for global blocks (single folded cos/sin table; reuses the DINOv2
  window scaffold; preserves the `~num_windows²×` local-attention FLOP savings).
- ✅ Rejected mask/flex-attention (fails the latency bar; not ONNX-exportable).

## Phase 1 — Windowed DINOv3 backbone module ✅
- ✅ `src/rfdetr/models/backbone/dinov3_with_windowed_attn.py` — `WindowedDinov3ViTConfig` +
  `WindowedDinov3ViTBackbone` (composes stock DINOv3 ViT blocks; names mirror upstream so
  pretrained weights map 1:1).
- ✅ `src/rfdetr/models/backbone/dinov3_configs/dinov3_{small,base,large}.json`.
- ✅ `tests/models/backbone/test_windowed_dinov3.py` — **keystone:** `num_windows=1` is
  bit-identical to upstream `DINOv3ViTBackbone`; plus RoPE-fold identity, local/global window
  routing, shapes, divisibility.

## Phase 2 — Integration wiring ✅
- ✅ `DinoV3` wrapper (`dinov3.py`) mirroring `DinoV2`'s interface; `export()` is a no-op (RoPE
  needs no PE surgery).
- ✅ `Backbone` factory routes `dinov2_*` / `dinov3_*` by family.
- ✅ `config.py`: widened `EncoderName` (+`dinov3_windowed_small/base`); added
  `RFDETRDinov3BaseConfig`.
- ✅ Public variant `RFDETRDinov3Base` (registered in `variants.py`, `detr.py`, `__init__.py`).
- ✅ `weights.py` PE interpolation is a safe no-op for RoPE checkpoints (no change needed).
- ✅ `tests/models/backbone/test_dinov3_wiring.py` — factory routing + full `build_model_from_config`.
- ✅ Regression: 242 tests pass, no breakage to the DINOv2 path.

## Phase 3 — Pretrained weights + trainability ✅
- ✅ Real gated `facebook/dinov3-vit{s,b}16` load verified: all params map (0 missing / 0 extra),
  `num_windows=1` bit-identical with **real** weights
  (`tests/models/backbone/test_dinov3_pretrained_weights.py`, token-gated, auto-skips offline).
- ✅ Trainability proven: `tests/models/backbone/test_dinov3_training_smoke.py` (loss ↓, grads
  flow through window-fold + RoPE, backbone updates, RoPE `inv_freq` frozen).
- ✅ Real training run end-to-end via `scripts/train_dinov3_demo.py`: gated ViT-B → synthetic
  COCO → PTL `Trainer.fit` (97.3M params, CPU) → checkpoint → reloaded via `from_checkpoint`.
- ✅ Adversarial wiring review; fixed the 3 confirmed findings (all in the demo script —
  `multi_scale` is in fact **safe** to enable; added `--multi-scale`/`--use-ema` opt-ins).

## Phase 4 — Real convergence run 🔜
- ⬜ Train on a real COCO/Roboflow dataset on a **GPU** (with `--multi-scale --use-ema`,
  `fused_optimizer` default) to a target mAP; sanity-check vs the DINOv2 baseline.
- ⬜ Quick latency/throughput comparison vs the DINOv2 windowed backbone at matched resolution.

## Phase 5 — Land it ⬜
- ⬜ Commit to a branch; open the maintainer issue (per `AGENTS.md`, new backbone family needs sign-off).
- ⬜ `pre-commit run --all-files`; ensure the token-gated test stays CI-safe (skips w/o `HF_TOKEN`).
- ⬜ PR with the design report + benchmark results.

## Out of scope (deferred) ⬜
- ⬜ SwiGLU / ViT-S+ / ViT-H+ variants (gated-MLP path exists but untested).
- ⬜ 24-layer ViT-L (needs an `out_feature_indexes` / window schedule decision).

---

### Environment notes (this box)
- Train deps installed via `uv pip install pytorch_lightning "torchmetrics[detection]" pycocotools scipy albumentations faster-coco-eval` (the lockfile `sync` fails on the `tflite` extra).
- No usable GPU (stale CUDA driver): run CPU with `CUDA_VISIBLE_DEVICES=""` and `fused_optimizer=False`; neither is needed on a healthy GPU box.
- Gated weights need `HF_TOKEN` (account must accept the DINOv3 license per repo). Token kept in gitignored `.env`.
