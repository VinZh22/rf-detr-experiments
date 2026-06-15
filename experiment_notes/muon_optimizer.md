# The Muon optimizer in RF-DETR — explained, wired, and A/B'd

*Course-note style. Assumes you know what an optimizer is (SGD/Adam) and roughly how a training loop
works, but **not** what "Muon", "orthogonalization", or "Newton-Schulz" mean. We build those up, show
exactly how it's wired into RF-DETR, then report a controlled AdamW-vs-Muon experiment.*

Code: [`src/rfdetr/training/muon.py`](../src/rfdetr/training/muon.py) ·
wiring in [`src/rfdetr/training/module_model.py`](../src/rfdetr/training/module_model.py) ·
tests in [`tests/training/test_muon.py`](../tests/training/test_muon.py).

---

## 0. TL;DR

- **Muon** is an optimizer for the **2D weight matrices** of a network. Instead of stepping along the raw
  (momentum-smoothed) gradient like SGD/Adam, it first **orthogonalizes** that update — roughly, it makes
  the update matrix have all singular values ≈ 1 — so learning is spread evenly across all directions of
  the weight matrix instead of being dominated by a few.
- Orthogonalization is only defined for matrices, so Muon is always **hybrid**: matrices use Muon,
  everything else (embeddings, the detection heads, all 1D biases/norms/gains) uses **AdamW**.
- We scale Muon's update so its size matches a typical AdamW update (the **RMS-matching** trick). That
  lets Muon **reuse AdamW's tuned learning rates and schedule unchanged** — which is what makes an
  AdamW-vs-Muon comparison a fair test of *the update rule alone*.
- **Result on this task** (DINOv3-small detector, DocLayNet, from scratch): **AdamW 0.347 vs Muon 0.337
  mAP** — AdamW wins by ~0.010. This is *expected*: a small detector on a small dataset with small batches
  is outside Muon's sweet spot (large-scale transformer pretraining). And Muon ran at AdamW's tuned
  operating point, so the number understates its potential (see §6).

---

## 1. The idea: why orthogonalize the update?

Picture the weights of one linear layer as a matrix `W` of shape `[d_out, d_in]`. Gradient descent nudges
`W` along `−∇W` (or a momentum-smoothed version of it). The problem: that gradient matrix is usually
**ill-conditioned** — a few of its singular directions are huge and dominate the step, while many small
directions barely move. So most of the "learning capacity" of the matrix is wasted each step.

**Muon's fix:** take the momentum-smoothed gradient `M`, compute its closest **orthogonal matrix**
(the `U Vᵀ` from its SVD `M = U Σ Vᵀ` — i.e. set every singular value to 1), and step along *that*
instead. Every direction now gets an equal-magnitude push. Intuitively it's **steepest descent measured
in the spectral norm** rather than the Euclidean norm. Empirically this accelerates convergence on
transformer hidden weights and underlies several recent LLM-pretraining speed records.

### 1.1 Newton-Schulz: orthogonalizing without an SVD

A real SVD every step would be far too slow. Muon instead runs a fixed **quintic Newton-Schulz
iteration** — five passes of `X ← aX + b(XXᵀ)X + c(XXᵀ)²X` with hand-tuned coefficients — that drives the
singular values toward 1 using only matrix multiplies (GPU-friendly), in **bfloat16**. It's approximate
(singular values land in ~`[0.7, 1.3]`, not exactly 1) but that inexactness is harmless for an optimizer
step. See [`zeropower_via_newtonschulz5`](../src/rfdetr/training/muon.py).

A quick sanity demo of "it pulls the spectrum toward 1":

```python
import torch
from rfdetr.training.muon import zeropower_via_newtonschulz5

g = torch.randn(16, 16)
print(torch.linalg.svdvals(g).min().item(),  torch.linalg.svdvals(g).max().item())     # wide, e.g. 0.05 .. 6.5
q = zeropower_via_newtonschulz5(g, steps=5).float()
print(torch.linalg.svdvals(q).min().item(),  torch.linalg.svdvals(q).max().item())     # tight, ~0.7 .. 1.3
```

---

## 2. Why Muon must be *hybrid* (and what goes where)

Newton-Schulz needs a matrix. So Muon can only govern **2D weight matrices** — the attention and MLP
linears in the ViT backbone and the DETR decoder. Everything else falls back to **AdamW**:

| parameter kind | optimizer | why |
|---|---|---|
| attention / MLP weight matrices (2D) | **Muon** | orthogonalization is defined here; this is where it helps |
| token / positional / query embeddings | AdamW | not a "hidden transform"; orthogonalizing them is meaningless |
| class / bbox **heads** (`*_embed`) | AdamW | output layers; standard Muon recipe keeps them on Adam |
| biases, LayerNorm/RMSNorm gains (1D) | AdamW | 1D — no matrix structure |

In RF-DETR this split is decided by
[`RFDETRModelModule._is_muon_param`](../src/rfdetr/training/module_model.py): a parameter goes to Muon iff
it is 2D **and** its name doesn't contain an embedding/head/norm marker (`embed`, `token`, `rel_pos`,
`reference_point`, `enc_out`, `norm`, `bias`). On `dinov3-small` this routes **107 of 497** parameter
groups to Muon and the rest to AdamW (printed at startup):

```
Muon optimizer: 107/497 param groups on Muon (2D hidden weights), rest on AdamW.
```

> **Detector caveat:** an LLM is ~95% matmul, so Muon governs almost everything. A *detector* carries a
> lot of non-matmul structure (heads, embeddings, and any conv neck / deformable-attention offsets), so a
> smaller fraction rides Muon — diluting its effect. Keep this in mind when reading §5.

---

## 3. RMS-matching: how Muon reuses AdamW's learning rate

A naive obstacle: AdamW and Muon updates have *completely different magnitudes*, so AdamW's tuned
learning rate (here `lr≈1e-4` with layer-wise decay) would be wildly wrong for Muon. Re-tuning the LR
from scratch would also confound any comparison — you'd be testing "Muon at LR X" vs "AdamW at LR Y".

The fix (the **Moonlight / "Muon is scalable" formulation**): scale each orthogonalized update so its
root-mean-square magnitude matches a typical AdamW update:

```
effective_step = lr · 0.2 · √(max(d_out, d_in))      # per matrix
```

With this, the **same per-group learning rates and the same schedule transfer from AdamW to Muon
unchanged**. That is exactly what makes our A/B clean: only the *update rule* changes, not the LR, the
warmup, or the cosine schedule. See [`Muon._muon_lr_scale`](../src/rfdetr/training/muon.py).

> This is also the most important caveat to keep in mind: RMS-matching gives the *base LR* a principled
> transfer, but **weight decay, layer-wise LR decay, and momentum were tuned for AdamW** and were *not*
> re-tuned for Muon (§6).

---

## 4. How it's wired into RF-DETR

Three small pieces:

1. **Optimizer** — [`src/rfdetr/training/muon.py`](../src/rfdetr/training/muon.py): a single-device
   `Muon(torch.optim.Optimizer)` holding both kinds of groups. Each group carries a `use_muon` flag;
   `use_muon=True` groups run momentum → Newton-Schulz → RMS-matched step, `use_muon=False` groups run a
   standard decoupled-weight-decay AdamW step. Per-group `lr` is honored, so the existing layer-wise-decay
   param dict and the external LR scheduler work without modification.
2. **Config** — [`TrainConfig`](../src/rfdetr/config.py) gains `optimizer: Literal["adamw","muon"] =
   "adamw"` and `muon_momentum: float = 0.95`.
3. **Selection** — [`RFDETRModelModule.configure_optimizers`](../src/rfdetr/training/module_model.py)
   branches on `optimizer`; `_build_muon_optimizer` tags each param group via `_is_muon_param` and
   constructs `Muon`. Fused-AdamW is disabled when `optimizer=="muon"`.

CLI: [`scripts/train_rfdetr_dataset.py`](../scripts/train_rfdetr_dataset.py) exposes `--optimizer
{adamw,muon}`.

**Correctness tests** ([`tests/training/test_muon.py`](../tests/training/test_muon.py), 11 cases): Newton-
Schulz collapses a random matrix's singular spectrum toward 1; both the Muon and AdamW branches reduce a
quadratic loss; the hybrid split routes six representative parameter types correctly; a group missing the
`use_muon` flag raises.

---

## 5. The experiment: AdamW vs Muon (measured)

A controlled A/B isolating *only the optimizer*. Both arms are **identical** except `--optimizer`:
`dinov3-small`, MAL loss, **from scratch**, 40 epochs, fixed resolution, EMA, `cuda_lap` matcher (exact,
so it doesn't bias convergence), seed 42, eval every 5 epochs. Run detached on two H200s (GPU 5 = AdamW,
GPU 6 = Muon). Metric: **EMA mAP@[.50:.95]** on the DocLayNet val split.

| epoch | AdamW | Muon | gap (AdamW − Muon) |
|---:|---:|---:|---:|
| 4  | 0.2494 | 0.2233 | +0.026 |
| 9  | 0.3027 | 0.2824 | +0.020 |
| 14 | 0.3243 | 0.3110 | +0.013 |
| 19 | 0.3389 | 0.3224 | +0.017 |
| 24 | 0.3444 | 0.3322 | +0.012 |
| 29 | 0.3461 | 0.3359 | +0.010 |
| 34 | 0.3471 | 0.3360 | +0.011 |
| **39** | **0.3473** | **0.3370** | **+0.010** |

**Read:** AdamW leads throughout. The gap is widest at the first eval (+0.026 — Muon's characteristic
slow warmup) and settles to a steady **~+0.010** by mid/late training. Muon tracks roughly *parallel* to
AdamW rather than crossing over. (Muon also covers epochs slightly slower in wall-clock, because Newton-
Schulz adds a few matmuls per matrix per step.)

### Why AdamW wins *here* (and why that's expected)

This task sits **outside Muon's sweet spot** on every axis that matters:

- **Small detector**, hidden dims 256–384 — orthogonalization helps most on *wide* matrices.
- **Low matmul-fraction** — only 107/497 groups ride Muon; the rest (heads, embeddings, norms) are on
  AdamW anyway, and a from-scratch detector head does a lot of its learning there.
- **Small physical batch** (eff. ~64) — Muon, like all matrix-structure methods, prefers cleaner
  large-batch gradient statistics.
- **Small dataset** — fast optimization can't separate from overfitting risk.

Muon's documented wins are on **large-scale transformer pretraining** (matmul-dominated, big matrices,
large batches, long schedules). None of those hold here, so a modest Muon loss on this benchmark says
little about that regime. See the companion discussion in
[`doclaynet_ab_results.md`](doclaynet_ab_results.md).

---

## 6. The fairness caveat (important)

AdamW had a **double home-field advantage**: its hyperparameters (LR, weight decay, layer-wise decay,
warmup, schedule) are the product of deliberate tuning *for AdamW on this architecture*, **and** the
regime suits it. Muon got exactly **one** principled adaptation — RMS-matched LR scaling — and inherited
everything else:

| hyperparameter | transferred to Muon? | note |
|---|---|---|
| base LR | ~yes | that's what RMS-matching is for — probably in the right ballpark |
| weight decay | **no** | Muon's update geometry differs; optimal WD likely differs |
| layer-wise LR decay (0.8 / 0.7) | **questionable** | tuned for Adam's per-element scaling; Muon already self-normalizes per matrix — possibly redundant |
| momentum (0.95) | untuned | a common Muon default, not swept |

So the honest framing of §5 is **"drop-in Muon with transferred hyperparameters trails *tuned* AdamW by
~0.010 here"**, not "Muon is worse." A fair follow-up is a small Muon-only sweep (base LR 0.5/1/2×; weight
decay; **layer-wise decay on vs off**; momentum 0.9 vs 0.95) — expected to bring Muon toward parity, but
not to a blowout in this regime.

---

## 7. When to actually reach for Muon

- **Use Muon** for **from-scratch transformer pretraining** dominated by 2D weight matrices (LLMs, large
  ViTs), especially at scale with large batches and long schedules — it's faster per step and uses *less*
  optimizer memory than Adam (one momentum buffer per matrix vs Adam's two moments).
- **Stick with AdamW** for small models, small/heterogeneous architectures (lots of non-matmul params),
  small batches, fine-tuning, and small datasets — i.e. settings like this one.
- It's a *drop-in* thanks to RMS-matching, so trying it costs little: A/B it early, and if it isn't
  clearly ahead by mid-training in your regime, keep AdamW.

---

## 8. Reproduce

From the repo root (`uv sync --all-groups` first). Both arms detached so they survive closing the
session (see [`scripts/run_detached.sh`](../scripts/run_detached.sh)):

```bash
DS=datasets/DocLayNetReduced
common="--dataset-dir $DS --model dinov3-small --epochs 40 --batch-size 16 \
        --num-workers 16 --from-scratch --mal-loss --use-ema --eval-interval 5 --seed 42"

# Arm A — AdamW (baseline)
CUDA_VISIBLE_DEVICES=5 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh \
  /workspace/doclaynet_ab/dinov3small_mal_adamw.log \
  .venv/bin/python scripts/train_rfdetr_dataset.py $common \
  --output-dir /workspace/doclaynet_ab/dinov3small_mal_adamw --optimizer adamw

# Arm B — Muon (identical except --optimizer)
CUDA_VISIBLE_DEVICES=6 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh \
  /workspace/doclaynet_ab/dinov3small_mal_muon.log \
  .venv/bin/python scripts/train_rfdetr_dataset.py $common \
  --output-dir /workspace/doclaynet_ab/dinov3small_mal_muon --optimizer muon
```

Read the mAP curve from each run's `metrics.csv` (column `ema_mAP_50_95`). Confirm the hybrid split in
the log (`Muon optimizer: 107/497 …`). Unit tests: `pytest tests/training/test_muon.py`.

> `RFDETR_MATCHER_SOLVER=cuda_lap` requires a CUDA toolkit (`nvcc`); drop it to fall back to the SciPy
> matcher. It only affects *speed*, not the result — see [`cuda_lap_matcher.md`](cuda_lap_matcher.md).
