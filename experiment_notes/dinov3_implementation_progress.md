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

## Phase 4 — Real training on COCO128 (GPU) ✅
- ✅ Trained on **COCO128** (real data) on an H200. COCO128 is Ultralytics-YOLO layout, which
  RF-DETR's YOLO loader doesn't read directly, so `scripts/train_dinov3_coco128.py` adapts it to
  the Roboflow-YOLO layout via symlinks (`adapt_ultralytics_to_roboflow_yolo`) — reuses the
  existing loader, no fork. Verified category-id alignment (model 0–79 ↔ GT coco cats 0–79).
- ✅ **Converges:** windowed DINOv3 ViT-B (res 640, bs 8, multi_scale + EMA), EMA mAP climbs
  0 → ~0.25 over 30 epochs (early-epoch 0.0 is normal DETR warmup, not a bug).
- ✅ **MAL vs IA-BCE A/B** (identical config, COCO128): MAL converges **~2–3 epochs faster** at
  every mAP threshold and ends slightly higher (EMA 0.264 vs 0.253). Matches DEIM's claim;
  caveat = COCO128 is val==train, single seed → suggestive, not a generalization result.
- ✅ **Latency** (`scripts/bench_backbone_latency.py`, ViT-B, H200, bf16, nw=2, batch 8) — windowed
  DINOv3 vs windowed DINOv2:
  - matched patch-16 (same 1600 tok): fwd **1.59×**, fwd+bwd **1.35×** slower (RoPE-per-layer cost).
  - native patches (DINOv2 p14=2304 tok, DINOv3 p16=1764 tok, res 672): fwd **1.27×**, fwd+bwd
    **1.08×** — DINOv2's extra tokens offset most of it; **~+8% per training step in practice**.
  - DINOv3 backbone is an unoptimized prototype (rotary apply + window merge/split are plain
    reshapes) → the gap is an upper bound, partly reclaimable.
- ✅ Real **generalization** run on **DocLayNetReduced** (11-class doc-layout; genuine train 4983 / val 4992 / test 4994
  split — *not* COCO128's val==train). Both arms via the new generalized harness `scripts/train_rfdetr_dataset.py`
  (any YOLO dataset / any variant / DINOv2|DINOv3 / MAL); identical config: res 576, bs 16, 16 workers, **cosine LR +
  0.5-ep warmup**, EMA, seed 42, 12 epochs, H200 (GPU 1 = DINOv3, GPU 4 = baseline).
  - **DINOv3 ViT-B** (from-scratch detector): best EMA mAP@50:95 **0.293** (ep 11, still creeping). Trains end-to-end,
    converges smoothly 0→0.29, no instability/crash — **the windowed-DINOv3 backbone generalizes beyond COCO128.**
  - **Baseline RFDETR-Medium** (DINOv2 ViT-S, COCO-pretrained): **0.382** (peaked ep 9).
  - The Medium win is **confounded**, not a backbone verdict: (1) Medium's detector head is COCO-pretrained vs DINOv3's
    from-scratch (huge head-start at only 12 ep — DETR heads converge slowly from scratch); (2) capacity differs
    (ViT-B 86M vs ViT-S 22M); (3) at patch16 the DINOv2 backbone never loads SSL weights (`dinov2.py` disables it),
    so it rides entirely on the COCO checkpoint. The trajectory tells the real story: DINOv3 gained ~2×/epoch and
    closed the gap 0.232→0.090 in 6 epochs, then both plateaued. Per-class: Medium leads on every class, but tail
    classes are floor for both (`formula` has **0** train instances; `footnote`/`title` ≈ 41/89 imgs → noise).
  - Read: *"as-shipped RFDETR-Medium > from-scratch-detector RFDETR-DINOv3-Base at 12 ep on DocLayNet."* Checkpoints:
    `/workspace/doclaynet_runs/{dinov3_base,medium}/checkpoint_best_ema.pth`.
- 🔜 For a clean **backbone** verdict: match the detector init (give DINOv3 a COCO-pretrained head, or train both
  from-scratch much longer) and capacity (DINOv3 ViT-S `dinov3_windowed_small` vs DINOv2 ViT-S). The **MAL vs IA-BCE**
  A/B on a real train≠val split is also still open (this run compared backbones, not losses).

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
- **GPU:** 8× H200, but `uv pip install -e .` pulls `torch 2.12+cu130` which can't init the box's CUDA 12.8 driver. Fix = the repo's own `ci-gpu-pin`: `uv pip install "torch<2.11" "torchvision<0.26" --index-url https://download.pytorch.org/whl/cu128` → torch 2.10+cu128, CUDA works. Pick a free GPU with `CUDA_VISIBLE_DEVICES=N`.
- **Checkpoint disk:** `/tmp` is on the tiny overlay `/` (3.8 T, ~near-full); ViT-B checkpoints are ~1.5 GB (full .ckpt) so `checkpoint_interval=1` fills it and crashes (`Errno 28`). Use `checkpoint_interval≈10`. `/workspace` has 21 T if more room is needed.
- **DataLoader:** `/dev/shm` was raised to **16 GB** (was 64 MB), so `num_workers>0` now works — the DocLayNet runs
  used `num_workers=16` per arm (2 arms = 32 workers) with no shm errors. The old `num_workers=0` workaround is obsolete.
- **Model-weight cache:** set `RF_HOME=/workspace/rf_home/models` — `~/.roboflow/models` isn't creatable here, so
  `download_pretrain_weights` otherwise dumps `rf-detr-*.pth` into the CWD (repo root). Also set `MPLCONFIGDIR=/tmp/mpl`
  to silence matplotlib's `~/.config` permission warning.
- **`--multi-scale` footgun:** with `square_resize_div_64=True` (default) and no random-resize, `multi_scale` collapses to
  a single *largest* scale (e.g. 736 px at res 576) with no jitter — it just trains at a bigger fixed res (~1.6× compute)
  and train/val resolutions diverge. Use a fixed `--resolution` instead unless you also pass `do_random_resize_via_padding`.
- **LR schedule:** the default `step` scheduler has `lr_drop=100`, so for short (≤12-ep) runs the LR never drops (flat,
  unconverged). Pass `lr_scheduler=cosine` (+ small `warmup_epochs`) for short runs — both DocLayNet arms used cosine.
- Gated weights need `HF_TOKEN` (account must accept the DINOv3 license per repo). Token kept in gitignored `.env`; source it per run: `set -a; . ./.env; set +a`.
- Related thread: MAL (DEIM) loss is implemented opt-in (`mal_loss` flag) — see `deim_rfdetr_review.md`.
