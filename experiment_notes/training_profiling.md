# Training pipeline profiling — DINOv3 RF-DETR

Profiling of the RF-DETR training step to locate CPU/GPU bottlenecks, the resulting
recommendations, and the measured result of the first optimization (moving the matcher's
finiteness check onto the GPU).

**Headline:** at `batch_size=16` on an object-dense dataset, ~**85% of each training step is the
Hungarian matcher** (CPU-side `scipy.linear_sum_assignment` + a finiteness check + the cost-matrix
`.cpu()` transfer), and the GPU sits **~87% idle**. The data loader and the DINOv3 backbone are
**not** the bottleneck. Tier-1 (finiteness check on GPU) gives a measured **~13% end-to-end
step speedup at zero accuracy cost**.

- Measured: 2026-06-09/10, single **NVIDIA H200** (GPU 5).
- Model: `RFDETRDinov3Base` — windowed DINOv3 ViT-B/16, res 576, 97.4M params,
  `num_queries=300`, `group_detr=13`, `dec_layers=4`, two-stage, bf16-mixed AMP.
- Data: `DocLayNetReduced` (11 classes, 4 983 train images; document layouts → many boxes/image).
- Config: `batch_size=16`, `num_workers=16`, CPU augmentation.

---

## 1. How to reproduce

The profiler runs the **real** training stack (`RFDETRModelModule` + `RFDETRDataModule` +
`build_trainer`) — so precision, EMA, the COCO-eval callback and the data pipeline behave exactly
as in production — then bounds it to a short smoke test.

```bash
# Loop-level + model-region breakdown (low overhead; the authoritative throughput numbers)
CUDA_VISIBLE_DEVICES=5 python scripts/profile_training.py \
    --dataset-dir datasets/DocLayNetReduced --model dinov3-base --device cuda:0 \
    --batch-size 16 --num-workers 16 --use-ema --train-batches 60 --val-batches 20

# Op-level CPU/CUDA attribution (heavier; confirms *what* in the matcher is slow)
CUDA_VISIBLE_DEVICES=5 python scripts/profile_training.py ... --profiler pytorch --val-batches 0

# A/B the Tier-1 matcher optimization (run with and without the flag)
CUDA_VISIBLE_DEVICES=5 python scripts/profile_training.py ... --matcher-gpu-finite-check
```

See [`scripts/profile_training.py`](../scripts/profile_training.py). It attaches two complementary
profilers:

1. **PyTorch Lightning `SimpleProfiler`** — wall-clock per training-loop hook. Key rows:
   `[_TrainingEpochLoop].train_dataloader_next` (time *waiting* on the DataLoader → the data/CPU
   signal), `[...].training_step` (forward+loss), `[...].backward`, `run_training_batch` (full
   step). If `train_dataloader_next` dominates you're input-bound; if the batch hooks dominate
   you're GPU-compute-bound.
2. **CUDA-event region timer** — forward hooks on `backbone` / `transformer` / `criterion` /
   `matcher` recording per-region GPU-stream time, so you see *where inside the model* the time
   goes. (Same CUDA-event style as [`scripts/bench_backbone_latency.py`](../scripts/bench_backbone_latency.py).)

---

## 2. Where the training step goes

Representative step (`bs=16`, EMA on, eval every epoch). Two independent methods agree:

| Phase | Time/step | % of step | Notes |
|---|---:|---:|---|
| **criterion / Hungarian matcher** | **~810–840 ms** | **~76%** | the bottleneck (CPU-bound) |
| backbone forward (DINOv3 ViT-B) | ~30 ms | ~3% | |
| transformer (decoder) forward | ~22–26 ms | ~2% | |
| backward (whole model) | ~86 ms | ~8% | |
| EMA update | ~48 ms | ~4% | only with `--use-ema` |
| **dataloader wait** | **~24 ms** | **~1%** | **not** the bottleneck |
| full step (`run_training_batch`) | ~1.1 s | 100% | |

Real GPU compute is ~142 ms (fwd 56 + bwd 86) out of a ~1.1 s step → the **H200 is ~13% utilized**.
The remaining ~0.9 s the GPU is idle, waiting on the CPU-side matcher.

**Validation is also expensive:** `validation_step` ≈ 592 ms/step because `compute_val_loss=True`
runs the matcher during validation too, and with EMA the COCO-eval callback runs a *second* forward
pass per batch. With `eval_interval=1` this happens every epoch.

> **Measurement variance:** the matcher's absolute cost is **data-dependent** (it scales with the
> number of ground-truth boxes per batch) and sensitive to GPU co-tenancy. Across runs we saw the
> criterion region range ~550–840 ms/step. Therefore **comparisons use a same-seed, back-to-back
> A/B** (section 5), not cross-run absolute numbers.

---

## 3. Op-level attribution (torch.profiler)

Sorted by CPU time, the criterion ≈ 43% of all CPU time, almost entirely the matcher (≈ 41%).
Per step the criterion calls the matcher **~5×** (1 main + 3 aux for `dec_layers=4` + 1 encoder for
two-stage). Each call, on a `[bs, num_queries, total_gt]` cost matrix:

1. `cost_matrix.float().cpu()` — the GPU→CPU transfer (`cudaMemcpyAsync`): **~13%** of step.
2. `torch.isfinite(cost_matrix).all()` on the **CPU** matrix: **~19%** of step.
3. `bs × group_detr` (16×13 = 208) `scipy.linear_sum_assignment` calls in a Python loop.

See [`HungarianMatcher.forward`](../src/rfdetr/models/matcher.py).

### Why it's the matcher

- **`group_detr=13`** (Group DETR): training runs `num_queries × group_detr = 300 × 13 = 3 900`
  queries split into 13 groups, each Hungarian-matched independently against the same GT (denser
  positive supervision → faster convergence; inference uses only 300 queries). This makes every
  cost matrix 13× taller and multiplies the solver-call count by 13.
- **Object-dense data** (document layouts) inflates `total_gt`, so both the finiteness reduction
  and the per-image solver grow.
- `scipy.optimize.linear_sum_assignment` is the LSAP solver (modern SciPy uses Jonker–Volgenant,
  not classical Hungarian). The cost here is **Python/overhead + GPU→CPU sync**, not the solver's
  asymptotic math (matrices are small).

---

## 4. Recommendations (measured priority)

The measurement **reordered** the initial code-review priorities:

| Idea | Initial guess | After measuring | Verdict |
|---|---|---|---|
| **Speed up the matcher** | #4 | **#1 by far (~85% of step)** | do this |
| `torch.compile` | #1 | only touches the ~56 ms forward (~5%) | **not worth it yet** |
| CPU augmentation PIL round-trips | #2 | dataloader wait is ~1% at bs16/16 workers | deprioritize |
| `compute_val_loss=False` + `eval_interval>1` | #3 | removes the matcher from every val batch | easy win |

**Matcher fixes, tiered:**

- **Tier 1 — finiteness check on GPU (done; section 5).** Zero-risk, identical matches.
- **Tier 2 — faster LAP solver.** Replace `scipy` with `lap.lapjv` / `lapsolver` (10–100× faster
  per call, less Python overhead). Same optimal assignment (differs only on arbitrary ties).
- **Tier 3 — fully-GPU batched assignment.** Auction algorithm / `torch-linear-assignment` solves
  all `bs×group` matrices on-device, eliminating the `.cpu()` transfer and the Python loop. Biggest
  win but it's an *approximate* solver → must A/B mAP/convergence vs the SciPy baseline before
  trusting it.
- **Leave `group_detr=13` as-is** (convergence-critical); optionally expose it as a documented
  throughput↔convergence knob.

---

## 5. Tier-1 result: finiteness check on GPU

### The change

`HungarianMatcher.forward` previously moved the cost matrix to CPU and *then* ran
`torch.isfinite(cost_matrix).all()` — a reduction over a ~40M-element tensor on the CPU. The check
now runs **on-device before `.cpu()`**, so only a scalar bool is synced. The two paths are
numerically identical (casting bf16→float32 neither creates nor removes non-finite values, and the
host copy is exact), proven by [`tests/models/test_matcher.py`](../tests/models/test_matcher.py)
(`TestHungarianMatcherGpuFiniteCheck`).

It is **opt-in and defaults to the legacy behavior** so nothing changes unless you ask for it:

- Constructor: `HungarianMatcher(gpu_finite_check=True)`
- Env var: `RFDETR_MATCHER_GPU_FINITE_CHECK=1`
- Profiler flag: `--matcher-gpu-finite-check`

### A/B (controlled, same seed, back-to-back; `bs=16`, no val)

| Per-step metric | Baseline (CPU check) | Tier-1 (GPU check) | Δ |
|---|---:|---:|---:|
| **full step** (`run_training_batch`) | 1197 ms | 1044 ms | **−153 ms (−13%)** |
| `training_step` (fwd+loss) | 1014 ms | 796 ms | −218 ms (−21%) |
| criterion region (CUDA events) | 554 ms | 292 ms | −262 ms (−47%) |
| └─ matcher | 534 ms | 274 ms | −260 ms (−49%) |
| backbone / transformer fwd | 30 / 21 ms | 30 / 22 ms | ~0 (untouched — sanity check) |

Throughput **0.84 → 0.96 step/s (~+15%)**, matches identical.

### Why the region drops 260 ms but the wall clock only drops 153 ms

CUDA is asynchronous, and part of the matcher's CPU work was overlapping GPU work — so removing it
doesn't shorten the wall clock one-for-one. The CUDA-event region timer brackets the matcher, but
inside that bracket the GPU is mostly **idle** while the CPU runs `isfinite`/scipy; the bracket
therefore measures GPU-idle, and shrinking the CPU work shrinks that idle by 260 ms. The wall clock
only improves by the part that was on the critical path with nothing to overlap it.

The smoking gun is `backward`, which "rose" 114 → 153 ms (**+39 ms**): in the baseline, GPU
backward/forward kernels were finishing *during* the long matcher CPU stall (hidden under it); once
the stall shrinks, that work surfaces in the open. The arithmetic closes:

```
training_step saved:        −218 ms
re-exposed overlapped work:  +65 ms   (backward +39, other GPU ops +26)
                            ───────
net full-step saving:       −153 ms   ✓ (1197 → 1044)
```

**Trust `run_training_batch` (−13%)** as the real throughput win. The region timer's −47% is the
*mechanism* (GPU stall removed) and overstates the wall-clock benefit. The same caveat applies to
Tier 2/3: the remaining ~274 ms matcher region won't all convert to wall-clock, because the
`.cpu()` transfer and scipy loop are similarly partly-overlapped. The only metric that settles a
matcher optimization is re-running the same `run_training_batch` A/B.

---

## 6. Tier-2 result: alternative solvers (`lap`, `sinkhorn`)

Pluggable matcher backend, selected via `HungarianMatcher(solver=...)` /
`RFDETR_MATCHER_SOLVER` / `--matcher-solver`: `scipy` (default), `lap` (exact, `lap.lapjv`), or
`sinkhorn` (approximate, GPU-resident entropic OT + argmax rounding). Correctness is covered by
[`tests/models/test_matcher.py`](../tests/models/test_matcher.py) (`lap` reproduces scipy's matches
exactly; `sinkhorn` recovers the optimum on well-separated problems and yields valid Group-DETR
structure).

A/B/C on GPU 3, all with Tier-1 on, validation off, same seed (`bs=16`, 60 train batches):

| solver (+ Tier-1) | matcher region (GPU ms/step) | training_step | **full step** | vs scipy |
|---|---:|---:|---:|---:|
| **scipy** | 282 ms | 848 ms | **1101 ms** | — |
| `lap` (lapjv) | 692 ms | 1147 ms | 1312 ms | **+19% slower** |
| `sinkhorn` (naive) | 631 ms | 1135 ms | 1308 ms | **+19% slower** |

**Both alternatives are slower than `scipy` + Tier-1.** Mechanisms:

- **`lap`:** the cost submatrices are extremely rectangular (`[300 queries × ~tens of targets]`).
  `lap.lapjv` has no rectangular mode; `extend_cost=True` **square-pads to `[300×300]`**, so JV solves
  a ~10× larger problem. SciPy's `linear_sum_assignment` handles the rectangular shape natively. `lap`
  is the wrong tool when queries ≫ targets.
- **`sinkhorn` (this impl):** GPU-resident and avoids the `.cpu()` cost-matrix transfer, but it loops
  over `bs` images × ~5 matcher calls (~80 iterations/step), each launching ~100 tiny `logsumexp`
  kernels (50 Sinkhorn iters) on small `[13, 300, T]` tensors plus a per-image `.cpu()` sync (~80
  syncs/step). It's **kernel-launch- and sync-bound** — the H200 idles on tiny ops, the same failure
  mode as the original matcher, relocated.

(Sanity check: `backward` measured *lower* for the slower solvers — scipy 156 ms vs lap 102 / sinkhorn
107 — the same overlap effect from section 5: a slower matcher hides more backward GPU work.)

**Verdict (Tier-2):** `lap` is fundamentally mismatched to the shape. `sinkhorn`'s slowness was an
*implementation artifact* (per-image Python loop + ~80 syncs), not fundamental — fixed in Tier-2b.

### Tier-2b: batched single-sync Sinkhorn (the win)

The Sinkhorn solver was rewritten to **batch every `(image, group)` block into one masked solve**: pad
each image's targets to the batch's `T_max` (extra columns excluded by a validity mask in the row
update), stack to `[bs·G, 300, T_max]`, run **one** batched Sinkhorn, and move the rounded query
indices to the host in **one** transfer. This removes the per-image Python loop and the ~80 syncs/step.

A/B on GPU 3, same seed/session, all with Tier-1 (`bs=16`, 60 train batches, val off):

| solver (+ Tier-1) | matcher region (GPU ms/step) | training_step | **full step** | vs scipy |
|---|---:|---:|---:|---:|
| `scipy` | 321 ms | 865 ms | **1114 ms** | — |
| **`sinkhorn` (batched)** | **143 ms** | 681 ms | **1005 ms** | **−109 ms (−10%)** |
| `sinkhorn` (naive, §6) | 631 ms | 1135 ms | 1308 ms | +17% |

Batching turned Sinkhorn from **+17% slower into ~10% faster** than `scipy`+Tier-1, and it's now the
fastest matcher (region 143 ms vs scipy 321 ms; the end-to-end −10% is smaller than the −55% region
delta because of the §5 async-overlap effect). The run trains cleanly (finite losses, no NaN).

### Tier-2b convergence A/B — Sinkhorn rejected ❌

The speed win above is only real if accuracy holds, so we ran a same-seed mAP A/B (identical init + data
order, **only** the matcher differs) via [`scripts/matcher_map_ab.py`](../scripts/matcher_map_ab.py) on GPU 3:
`dinov3-base`, 576 px, 4 epochs × 250 steps (~1000 steps), mAP on a fixed 640-image val subset.

| solver (1000 steps, 576 px) | mAP_50_95 | mAP_50 | mAP_75 | F1 | minutes |
|---|---:|---:|---:|---:|---:|
| **`scipy`** (exact) | **0.0122** | 0.0367 | 0.0059 | 0.0565 | 10.6 |
| `sinkhorn` (batched, approx) | 0.0060 | 0.0191 | 0.0023 | 0.0296 | 8.5 |
| Δ | **−51%** | −48% | −61% | −48% | −20% (faster) |

**Sinkhorn reaches ~half of scipy's mAP at the same step budget**, consistently across *every* metric
(so it's not seed noise). The approximate argmax rounding produces noisy/colliding query→target
assignments, degrading the one-to-one supervision DETR relies on → slower convergence. The ~20% per-step
speed advantage is moot if you need ~2× the steps to match quality.

> **Verdict: do not adopt Sinkhorn as the matcher.** The validated win is **`scipy` + Tier-1** — exact
> matches (identical convergence to the original) *and* −13% step time. The Sinkhorn solver stays in the
> codebase behind its opt-in flag for experimentation (e.g. tightening `sinkhorn_eps` / `sinkhorn_iters`
> or adding collision-free rounding, then re-running this A/B), but it is **not** recommended for training.
>
> Caveat: single seed, tiny scale (~1000 steps, mAP ~0.01) — absolute numbers are noisy, but the ~2× gap
> across all metrics is well beyond seed noise.

### Tier-3: exact GPU batched solver (`cuda_lap`) — fastest, exact ✅

> 📖 Beginner-friendly deep-dive on how `cuda_lap` works (the assignment problem, the batching/padding
> trick, the correctness proof): [`cuda_lap_matcher.md`](cuda_lap_matcher.md).

The "missing quadrant": exact *and* on-GPU *and* batched. Implemented via
[`torch-linear-assignment`](https://github.com/ivan-chai/torch-linear-assignment)'s
`batch_linear_assignment` (a CUDA JV/auction kernel) behind `solver="cuda_lap"` — all `bs × group_detr`
sub-problems solved in one on-device call, no cost-matrix `.cpu()`, no Python loop. It is **exact**:
[`tests/models/test_matcher.py`](../tests/models/test_matcher.py) (GPU-marked) confirms it returns
**bit-identical** assignments to scipy, so convergence matches the baseline by construction (no mAP A/B
needed). Requires a CUDA toolkit (`nvcc`) to build — installed here via `cuda-nvcc-12-8` (matching `cu128`).
`cuda_lap` ignores the Tier-1 toggle (it sanitizes on the GPU regardless), so it subsumes Tier-1.

**GPU-utilization sweep** (GPU 3 verified idle, `nvidia-smi` sampled during training, `bs=16`, 80 steps;
step times are 3-rep means, reproducible to a few ms):

| config | matcher region | full step | GPU util (median) |
|---|---:|---:|---:|
| `scipy`, **no** Tier-1 (original) | 535 ms | 773 ms | **19%** |
| `scipy` + Tier-1 | 308 ms | 527 ms | **36%** |
| `cuda_lap` (Tier-1 irrelevant) | 153 ms | **384 ms** | **83%** |

`cuda_lap` is **~27% faster than scipy+Tier-1 and ~50% faster than the original**, and it's the only
config that makes the step **GPU-bound** (83% util). Tier-1 alone lifts util just 19%→36% — scipy+Tier-1
is **still matcher-CPU-bound** (GPU idle ~64%, stalling on the scipy Python loop + per-call cost `.cpu()`).
`cuda_lap` removes that stall entirely.

> **Correction:** earlier A/B runs (§ git history) showed `cuda_lap` ~6% *slower*. Those were **GPU-contention
> artifacts** — `cuda_lap` is GPU-bound, so a transient co-tenant on the shared H200 slows it far more than
> CPU-bound scipy. A util-instrumented run on a verified-idle GPU + a 3-rep same-seed confirmation
> (scipy 511/528/543 ms vs cuda_lap 383/385/383 ms) overturned that. Caveat: `cuda_lap`'s cost is more
> **data-sensitive** than scipy (auction iterations scale with target density), so the exact margin varies
> with the dataset; and it's contention-sensitive, so co-located jobs erode the win.

> **Verdict:** when a CUDA toolkit is available, **`cuda_lap` is the best matcher** — exact (identical
> convergence) *and* fastest (~27% over scipy+Tier-1), and it makes the step GPU-bound. It needs a compiled
> CUDA dependency (`nvcc`) and a reasonably exclusive GPU. **Fallback when no toolkit / shared GPU: scipy +
> Tier-1.**

---

## 7. Validation throughput levers (`compute_val_loss`, `eval_interval`) — the safe wins ✅

Independent of the matcher work, validation itself is expensive (validation_step ~592 ms/step in §2),
partly because `compute_val_loss=True` runs the full criterion — **including the Hungarian matcher** — on
every val batch, and `eval_interval=1` does it every epoch. Both are **pure throughput levers**: they
change how much / how often you pay for validation, never the training compute or the weights (no accuracy
risk). Now wired into [`scripts/train_rfdetr_dataset.py`](../scripts/train_rfdetr_dataset.py) as
`--eval-interval` / `--(no-)compute-val-loss`.

A/B via [`scripts/val_cost_ab.py`](../scripts/val_cost_ab.py) (GPU 3, dinov3-base, 576 px, 3 epochs, identical
training, 64-batch val subset — so all deltas are pure validation):

| arm | config | minutes |
|---|---|---:|
| baseline | `compute_val_loss=True, eval_interval=1` | 4.85 |
| no_val_loss | `compute_val_loss=False, eval_interval=1` | 3.79 |
| no_vl + ei=3 | `compute_val_loss=False, eval_interval=3` | 3.43 |

- **`compute_val_loss=False` — the bigger lever: ~21 s/eval saved.** Per-eval cost drops ~32 s → ~11 s, so
  the val-loss pass (matcher on val) is ~⅔ of each eval — the same matcher bottleneck, on the val side.
- **`eval_interval` — ~10.8 s per skipped eval** (linear; eval N× less often).
- Combined: **−29%** on this bounded run, and it **scales with val-set size × epoch count**. On a real
  full-val run (DocLayNet val ≈ 4992 imgs ≈ 5× this subset, 12 epochs) the saving is on the order of tens
  of minutes — far larger than here.

> Recommendation for throughput runs: `--no-compute-val-loss` and `--eval-interval 3` (or higher). Keep
> them at the defaults only when you actually monitor val loss / need per-epoch mAP. (EMA adds a *second*
> val forward pass on top, so these levers compound further when `--use-ema` is on.)

---

## 8. Live utilization sweep — batch size, model size, and the util-vs-throughput gap

A fresh `nvidia-smi`-sampled sweep (median GPU util over the steady-state window, sampled every 0.5 s
while the real profiler runs 80 train steps, val off) on an idle GPU. This adds two things the §6 sweep
didn't isolate: a **small model** (`dinov3-small`, 32.9 M) and a **bigger batch** (32), plus wall-clock
step time alongside utilization so the two can be compared directly.

Both runs sampled live on an idle GPU 1, 80 train steps, val off, EMA off. Each config got its own
output dir so all three wall-clock step times are captured in one session (`run_training_batch` mean from
the PTL `SimpleProfiler`).

### 8.1 `dinov3-small` (32.9 M), bs=32

| config | GPU util (median) | util p90 | wall step | matcher GPU region | peak mem |
|---|---:|---:|---:|---:|---:|
| `scipy` (original) | **5%** | 78% | 2.14 s | 1779 ms (idle stall) | 44 GB |
| `scipy` + Tier-1 | 30% | 95% | 1.54 s | 1208 ms | 44 GB |
| **`cuda_lap`** | **92%** | 98% | **0.78 s** | 441 ms (real GPU work) | 44 GB |

### 8.2 `dinov3-base` (97 M), bs=32

| config | GPU util (median) | util p90 | wall step | matcher GPU region | peak mem |
|---|---:|---:|---:|---:|---:|
| `scipy` (original) | **5.5%** | 92% | 2.23 s | 1789 ms (idle stall) | 54 GB |
| `scipy` + Tier-1 | 27.5% | 98% | 1.60 s | 1175 ms | 55 GB |
| **`cuda_lap`** | **93%** | 98% | **0.82 s** | 426 ms (real GPU work) | 53 GB |

### 8.3 What this says

- **`cuda_lap` is a ~2.7× step speedup at bs=32 for *both* model sizes** (small 2.14→0.78 s, base
  2.23→0.82 s) and takes utilization from ~5% (GPU sitting *kernel-less* during the CPU matcher) to
  **~92–93%**. Here util and throughput **agree** — the GPU is genuinely busy *and* the step is far
  faster. Tier-1 alone is the intermediate point (~−28%).
- **The matcher is still the single largest GPU consumer — relocated, not removed.** In the `cuda_lap`
  forward, the matcher is **441 ms of the 548 ms top-level region** (small) — i.e. the exact assignment
  dwarfs the actual detection compute (backbone 30 ms + transformer 38 ms). It's ~55% of the whole
  ~0.8 s step. So "93% util" is real, but roughly *half* of that busy time is the (exact, necessary)
  assignment, not convnet/transformer math. To go faster still, the next lever is the matcher itself
  (or `group_detr`), not the backbone.
- **Model size barely matters at this batch** because the matcher cost is model-independent (same
  dataset/batch → same assignment problem; 441 vs 426 ms) and dominates. The 97 M base costs almost the
  same per step as the 33 M small (0.82 vs 0.78 s) — the extra backbone compute is small next to the
  matcher.
- **bs=32 fits comfortably** (44 GB small, 54 GB base of 143 GB), so there's headroom for larger batches.

> **Correction to an earlier reading:** a first pass reported the small model at only ~8% faster with
> `cuda_lap` (2.05 s). That was a contaminated measurement (three configs sharing one overwritten output
> dir across separate invocations) — a *smaller* model cannot be slower than base at the same matcher
> cost. The clean single-session sweep above (per-config dirs) shows the expected ~2.7×, and the GPU-time
> budget now reconciles with wall-clock (fwd 548 + bwd ~122 ≈ 670 ms of GPU work in a 780 ms step ≈ the
> measured 92% util).

---

## 9. Status

- [x] Profiler added: [`scripts/profile_training.py`](../scripts/profile_training.py)
- [x] Tier 1 (finiteness check on GPU) — implemented, opt-in, tested, **−13% step**
- [x] Tier 2 (`lap` / `sinkhorn` solvers) — implemented, opt-in, tested; `lap` and *naive* `sinkhorn`
  both slower than scipy+Tier-1 for this matrix shape (see §6)
- [x] Tier 2b (batched single-sync Sinkhorn) — implemented, tested, **~10–20% faster than scipy+Tier-1**
- [x] mAP/convergence A/B (scipy vs Sinkhorn) — **Sinkhorn rejected: ~half the mAP at equal steps** (§6)
- [x] Validation levers wired (`--eval-interval` / `--(no-)compute-val-loss`) + A/B — **−29% bounded,
  scales to tens of minutes on full-val runs**; no accuracy risk (§7)
- [ ] **mAP/convergence A/B** for batched Sinkhorn vs scipy — required before using it for real training
- [x] Tier 3 (`cuda_lap`, exact GPU batched solver) — implemented, **bit-identical to scipy**, GPU-tested;
  **~27% faster than scipy+Tier-1, ~50% vs original, GPU-bound (83% util)** on an exclusive GPU → best
  matcher when a CUDA toolkit is available (§6). (Earlier "slower" reading was GPU contention.)
- [x] GPU-bound verification — scipy+Tier-1 is **only 36% util (still matcher-CPU-bound)**; only `cuda_lap`
  reaches GPU-bound (83%) (§6)
- [x] Validation throughput levers (`--eval-interval` / `--(no-)compute-val-loss`) wired + A/B'd — **−29%
  on a bounded run, scales to tens of minutes on full-val runs**, no accuracy risk (§7)
