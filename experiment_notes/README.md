# Experiment notes

Standalone write-ups of the profiling and A/B experiments run on this RF-DETR fork. Each is
self-contained, written course-note style (build up the concept, then the implementation, then the
*measured* result), and ends with a **Reproduce** section. All commands assume you run from the **repo
root** after `uv sync --all-groups`.

## Start here

| If you want to… | Read |
|---|---|
| understand the windowed **DINOv3** backbone design (how windowed + global attention and RoPE are built) | [dinov3_windowed_prototype.md](dinov3_windowed_prototype.md) |
| track the implementation status of the DINOv2 → DINOv3 backbone swap | [dinov3_implementation_progress.md](dinov3_implementation_progress.md) |
| understand where training time goes and why the matcher was the bottleneck | [training_profiling.md](training_profiling.md) |
| understand the GPU-resident exact matcher that fixed it (with a correctness proof) | [cuda_lap_matcher.md](cuda_lap_matcher.md) |
| see what makes DINOv3 better than DINOv2, and MAL vs IA-BCE | [doclaynet_ab_results.md](doclaynet_ab_results.md) |
| understand the Muon optimizer and how it compared to AdamW here | [muon_optimizer.md](muon_optimizer.md) |
| make inference faster (CPU preprocessing is the real cost) | [inference_profiling.md](inference_profiling.md) |
| see whether DEIM's ideas (MAL, Dense O2O) fit RF-DETR | [deim_rfdetr_review.md](deim_rfdetr_review.md) |
| see whether DEIM's **Dense O2O** can replace **Group DETR** | [denseo2o_vs_groupdetr_ab.md](denseo2o_vs_groupdetr_ab.md) |

## The thread connecting them

0. **The backbone** ([dinov3_windowed_prototype.md](dinov3_windowed_prototype.md) →
   [dinov3_implementation_progress.md](dinov3_implementation_progress.md)) is the foundation of this
   branch: swapping RF-DETR's windowed DINOv2 ViT for a windowed **DINOv3** ViT (windowed + periodic
   global attention, RoPE). The prototype design report is adversarially verified; the progress tracker
   records what's implemented. Everything below builds on having this backbone.
1. **Profiling** ([training_profiling.md](training_profiling.md)) found the Hungarian **matcher** ate
   ~85% of each training step while the GPU sat idle.
2. That motivated **`cuda_lap`** ([cuda_lap_matcher.md](cuda_lap_matcher.md)) — an exact, GPU-resident,
   batched assignment solver: same matches as SciPy, ~27–50% faster steps, GPU finally busy.
3. With training cheap enough to iterate, we ran **convergence A/Bs**
   ([doclaynet_ab_results.md](doclaynet_ab_results.md)): DINOv3 vs DINOv2 backbones (size-matched), and
   MAL vs IA-BCE loss.
4. Then an **optimizer A/B** ([muon_optimizer.md](muon_optimizer.md)): AdamW vs Muon on the best
   small-model config.
5. Separately, **inference profiling** ([inference_profiling.md](inference_profiling.md)) showed
   `predict()` is ~75–80% CPU preprocessing, not model forward — a 5–11× win by moving preprocessing to
   the GPU.
6. [deim_rfdetr_review.md](deim_rfdetr_review.md) is the feasibility study that led to adding the MAL loss;
   [denseo2o_vs_groupdetr_ab.md](denseo2o_vs_groupdetr_ab.md) then implemented DEIM's **Dense O2O**
   (mosaic+mixup) and A/B'd it as a *replacement* for Group DETR (crossed with the loss).

## Key results at a glance

- **Backbone (prototype):** a windowed **DINOv3** ViT backbone (windowed + periodic global attention,
  RoPE) trains end-to-end in RF-DETR — the prototype that the convergence A/Bs below run on.
- **Matcher:** `cuda_lap` (exact, GPU) is ~27% faster than the best CPU variant and the only config that
  makes the step GPU-bound (83% util). Convergence-safe by construction (bit-identical to SciPy).
- **Backbone (DINOv2 vs DINOv3):** size-matched DINOv3-S beats DINOv2-S by **+0.107 mAP**; the 3.7× jump
  to DINOv3-B adds only +0.020 → DINOv3's advantage is ~84% backbone SSL/architecture, ~16% size.
- **Loss:** MAL beats IA-BCE by a steady ~+0.01 mAP (holds under both Group DETR and Dense O2O).
- **Dense O2O vs Group DETR:** budget-dependent. At **20ep** Group DETR (g=13) beats Dense O2O
  (mosaic+mixup, g=1) by ~0.01–0.02. At **40ep the ranking flips for the MAL pairing**: Dense O2O+MAL
  **0.351** is the best arm, edging Group DETR (0.343) — but Dense O2O+IA-BCE (0.334) is the *worst*, so
  Dense O2O's win hinges entirely on MAL (+0.017) handling the low-quality matches mosaic creates. Group
  DETR converges much faster (use it for short budgets); Dense O2O+MAL has the higher ceiling given a
  long schedule + close-mosaic fine-tune.
- **Optimizer:** drop-in Muon trails *tuned* AdamW by ~0.010 mAP on this small detector — expected
  outside Muon's large-scale-pretraining sweet spot, and Muon was not equally tuned.
- **Inference:** GPU-side preprocessing + jit-trace → up to **11×** faster `predict()`; fp16/bf16/autocast
  are useless on H200 for this model.

> These are research notes on an experiments fork, not user documentation — see
> [`../README.md`](../README.md) and [`../AGENTS.md`](../AGENTS.md) for the project proper.
