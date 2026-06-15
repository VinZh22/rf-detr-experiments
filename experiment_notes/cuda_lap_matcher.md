# A GPU-resident exact matcher for RF-DETR (`cuda_lap`) — explained from scratch

*Course-note style. Assumes you know basic Python/PyTorch and roughly what object detection is, but
**not** what "linear assignment", "Hungarian matcher", or "Group-DETR" mean. We build those up first,
then dissect the implementation and the one genuinely clever trick in it.*

Companion to [`training_profiling.md`](training_profiling.md) (the measurements) and the code in
[`src/rfdetr/models/matcher.py`](../src/rfdetr/models/matcher.py).

---

## 0. TL;DR

RF-DETR spends a large fraction of each training step inside the **matcher** — the part that decides
*which predicted box should be compared against which ground-truth box*. The classic implementation
does that on the **CPU** (SciPy), which stalls the GPU. `cuda_lap` does the *exact same computation*
**entirely on the GPU, for the whole batch at once**, using one call to a CUDA solver. The result is
**bit-identical** to the CPU version (so training is unaffected) but the GPU stops idling.

The one idea you should remember: *you can solve many differently-sized assignment problems in a single
batched GPU call by padding them to a common size with a **constant** cost, and then throwing away the
padded matches afterwards — and this provably does not change the real answer.*

---

## 1. The problem the matcher solves

### 1.1 Detection as "match, then score"

A DETR-style detector emits a **fixed set of predictions** — say 300 "object queries", each producing a
box + class scores. An image has some number of **ground-truth (GT) objects** — say 7 boxes. Before we
can compute a loss, we must decide **which query is responsible for which GT box**. We want a
**one-to-one** matching: each GT box is explained by exactly one query, and each query explains at most
one GT box (the rest are "no object").

We pick the matching that is *cheapest*, where the "cost" of pairing query *i* with GT box *j* combines:

- how different their class predictions are,
- how far apart the boxes are (L1 distance),
- how poorly they overlap (1 − GIoU).

So we build a **cost matrix** `C` of shape `[num_queries, num_gt]`, where `C[i, j]` = cost of matching
query *i* to GT box *j*. Low cost = good match.

### 1.2 This is the "Linear Sum Assignment Problem" (LSAP)

> **Definition.** Given a cost matrix `C` of shape `[N, M]`, choose at most one cell per row and at most
> one per column, covering all `min(N, M)` of the smaller side, so that the **sum of chosen cells is
> minimized**.

That's it. It's a classic combinatorial optimization problem. The textbook algorithm is the
**Hungarian algorithm** (Kuhn–Munkres); modern solvers (SciPy's `linear_sum_assignment`, the `lap`
library, GPU "auction" solvers) use faster variants (Jonker–Volgenant, auction) but compute the **same
optimal answer**. DETR named its module `HungarianMatcher` for this reason.

**Worked example.** 3 queries, 2 GT boxes (`N=3 > M=2`, so 2 pairs are chosen, 1 query is left over):

```
        gt0   gt1
q0   [  9.0   1.0 ]
q1   [  2.0   8.0 ]
q2   [  7.0   6.0 ]
```

The optimal assignment is `q1→gt0` (2.0) and `q0→gt1` (1.0), total 3.0. `q2` is unmatched (background).
No other choice is cheaper.

**Key shape fact for us:** there are *many more queries than GT boxes* (300 vs ~7). So the cost matrix is
extremely **tall and thin**, and the solver assigns *every GT box* to a distinct query, leaving most
queries unmatched. Remember this — it drives the whole design.

---

## 2. Two complications specific to RF-DETR

### 2.1 Group-DETR: several query groups at once

To converge faster, RF-DETR trains with **`group_detr` independent copies** of the query set (13 of
them). During training the model emits `num_queries × group_detr` queries (e.g. `300 × 13 = 3900`), split
into 13 groups of 300. **Each group independently solves its own assignment against the same GT boxes.**
So every GT box gets matched 13 times per image (once per group) → denser training signal. (At inference
only one group is used, so deployment cost is unchanged.)

Consequence: per image we don't solve one assignment — we solve `group_detr` of them.

### 2.2 A batch has many images, each with a different number of GT boxes

We train on `batch_size` images at once (say 16). Image 0 might have 7 GT boxes, image 1 might have 60,
image 2 might have 0. Each (image, group) pair is its **own** little assignment problem.

So per **training step** we solve:

```
batch_size × group_detr  =  16 × 13  =  208  separate assignment problems
```

each of shape `[300 queries, T_i targets]`, where `T_i` varies per image. And the criterion runs the
matcher ~5 times per step (once for the final layer, once per auxiliary decoder layer, once for the
two-stage encoder), so it's really ~**1000 assignment problems per step**.

---

## 3. Why this was the bottleneck

Profiling DINOv3 RF-DETR (details in [`training_profiling.md`](training_profiling.md)) showed the matcher
was **~85% of each training step**, with the GPU sitting **~13–36% utilized** — i.e. mostly *idle*. Why?

The classic recipe per (image, group) problem is:

1. Compute `C` on the **GPU** (fast — it's just arithmetic on tensors).
2. Copy `C` to the **CPU** (`.cpu()`), which **forces a synchronization**: the CPU blocks until the GPU
   finishes, and while the CPU then runs the solver the **GPU does nothing**.
3. Run SciPy `linear_sum_assignment` on the CPU, in a **Python loop over all 208 sub-problems**.

Steps 2–3 are pure CPU/transfer work. During them the expensive GPU (an H200) is idle. The classic matcher
is **CPU-bound**, and it starves the GPU.

> There's a whole design space here — a matcher can be **exact or approximate**, and run on **CPU or GPU**.
> SciPy = exact + CPU (the bottleneck). Sinkhorn = approximate + GPU (fast but it *hurt accuracy by ~half*,
> so it was rejected). `cuda_lap` is the **missing quadrant: exact + GPU**.

---

## 4. The `cuda_lap` idea

Keep everything on the GPU and solve all 208 problems **in one batched GPU call**, with an **exact**
solver, so the answer is identical to SciPy but the GPU never idles.

We use the [`torch-linear-assignment`](https://github.com/ivan-chai/torch-linear-assignment) library,
which provides:

```python
batch_linear_assignment(cost)  # cost: [B, W, T] float on GPU
# returns: assignment [B, W] int64, where assignment[b, w] = the task assigned to worker w
#          (or -1 if that worker got no task). Minimizes total cost. Runs as one CUDA kernel.
```

In its vocabulary, **workers = queries**, **tasks = GT boxes**. `B` is the batch of *independent*
problems. Because we have far more workers (queries) than tasks (GT boxes), exactly `T` workers get a
task and the rest get `-1`. We verified its output is **numerically identical to SciPy** (Section 7).

Two obstacles stand between "nice API" and "use it for all 208 problems at once":

1. **The problems have different sizes** (`T_i` varies per image), but a single batched call needs one
   common shape `[B, W, T]`.
2. We must map the result back to the (query, target) index pairs the loss expects, and do it without
   re-introducing per-problem CPU syncs.

Sections 5–6 solve these.

---

## 5. The cost tensor: from one big matrix to a clean batch

### 5.1 What we start with

The model hands the criterion a cost matrix for the *whole batch* of shape:

```
C_all : [bs, num_queries, total_gt]
        num_queries = 300 × group_detr (training)
        total_gt    = sum of all images' GT counts, concatenated along the columns
```

Only the **block-diagonal** part is meaningful: image *i*'s queries should only be matched against image
*i*'s **own** GT boxes. Picture `total_gt` columns split into per-image chunks:

```
            img0 cols   img1 cols   img2 cols
          ┌───────────┬───────────┬──────────┐
 img0 Q   │  USE THIS │   ignore  │  ignore  │
 img1 Q   │  ignore   │ USE THIS  │  ignore  │
 img2 Q   │  ignore   │  ignore   │ USE THIS │
          └───────────┴───────────┴──────────┘
```

We only ever use the diagonal blocks `C_all[i, :, cols_of_image_i]`, each `[num_queries, T_i]`.

### 5.2 Reshape: peel out the groups

Within image *i*, the `num_queries` rows are the 13 groups stacked: rows `0..299` = group 0, `300..599`
= group 1, etc. So a diagonal block `[num_queries, T_i]` reshapes to `[group_detr, 300, T_i]` — i.e.
group `g` owns rows `g·300 … (g+1)·300−1`. Each `[300, T_i]` slice is one independent assignment problem.

### 5.3 Pad to a common size

Different images have different `T_i`. To stack them into one batched tensor we pad every block's task
dimension to `T_max = max(T_i)` and fill the extra columns with **0** (a constant):

```
real targets (T_i = 2)        padded to T_max = 4
┌──────┬──────┐               ┌──────┬──────┬──────┬──────┐
│ c00  │ c01  │   ──────►     │ c00  │ c01  │  0   │  0   │
│ c10  │ c11  │               │ c10  │ c11  │  0   │  0   │
│ ...  │ ...  │               │ ...  │ ...  │  0   │  0   │
└──────┴──────┘               └──────┴──────┴──────┴──────┘
                                              ↑ padded "fake" tasks
```

We do this for all `bs × group_detr` blocks and stack them into:

```
cost_b : [bs · group_detr, 300, T_max]      ← one tensor, 208 independent problems
```

This is exactly what `batch_linear_assignment` wants.

### 5.4 The catch — and why padding with a constant is safe

Because there are far more queries (300) than tasks (`T_max`), the solver **assigns every task column**,
**including the padded fake ones**. So some queries will be "matched" to fake targets. We must (a) make
sure the fake tasks don't *steal the good queries* from the real targets, and (b) discard the fake matches
afterward.

(b) is easy: a fake match has a target index `≥ T_i`, so we filter by `target < T_i`.

(a) is the subtle part, and it's why we pad with a **constant**:

> **Lemma (constant padding doesn't change the real answer).** If every padded column has the *same* cost
> for *every* query (here, 0), then the optimal assignment gives the real targets exactly the queries they
> would have gotten with no padding at all.
>
> **Why.** Total cost = (cost of real-target matches) + (cost of fake-target matches). A fake column costs
> the same constant *no matter which query it's assigned to*, so the fake part contributes a **fixed
> total** (number-of-fakes × constant) regardless of which queries the fakes grab. Minimizing the total is
> therefore the same as minimizing just the real part. And could a fake column "want" a query that a real
> target needs? Moving a fake column off that query never changes the fake cost (it's constant) but frees
> the query for the real target, which can only *lower or equal* the real cost. With 300 queries and only
> `T_max` tasks there are always ~293 spare queries for the fakes to sit on. So at the optimum the real
> targets keep their best queries. ∎

Intuition: the fakes are "indifferent" tenants — they'll take whatever empty seat is left, and there are
plenty of empty seats, so they never bump a real passenger.

---

## 6. Solve once, unpack once

```python
# one CUDA kernel call solves all 208 problems in parallel, on the GPU:
assign = batch_linear_assignment(cost_b)            # [bs·group_detr, 300]
assign = assign.reshape(bs, group_detr, 300).to("cpu")   # the ONLY host sync
```

`assign[i, g, q]` = the target that query *q* of group *g* in image *i* was matched to (or `-1`).

Then, on the CPU (cheap — these are tiny integer tensors, no GPU work, no further syncs), we turn it into
the `(query_index, target_index)` pairs the loss expects, per image, concatenated over groups:

```python
for i, T_i in enumerate(sizes):                     # per image
    a = assign[i]                                   # [group_detr, 300]
    valid = (a >= 0) & (a < T_i)                    # drop the padded/fake matches
    g, q = valid.nonzero(as_tuple=False).unbind(1)  # which (group, local-query) are real matches
    query_index  = q + g * 300                      # local query → global query index
    target_index = a[g, q]                          # the GT box it matched
```

Two design points worth calling out:

- **Exactly one host transfer per matcher call.** The big cost tensor never leaves the GPU. We only copy
  the small `assign` tensor (`[208, 300]` int64 ≈ 0.5 MB) once. Contrast with the classic path: a `.cpu()`
  of the full cost matrix *plus* a Python loop of 208 SciPy calls.
- **The result is the same shape/meaning as every other solver** (a list of `(query_idx, target_idx)`
  int64 CPU tensors, group-major), so nothing downstream in the loss has to change. `cuda_lap` is a
  drop-in.

### 6.1 The whole thing, annotated

This is [`HungarianMatcher._assign_cuda_lap`](../src/rfdetr/models/matcher.py), lightly trimmed:

```python
def _assign_cuda_lap(self, cost_matrix, targets, group_detr):
    from torch_linear_assignment import batch_linear_assignment

    bs, num_queries, _ = cost_matrix.shape
    g_num_queries = num_queries // group_detr          # 300
    usable = g_num_queries * group_detr
    sizes = [len(t["boxes"]) for t in targets]         # T_i per image
    t_max = max(sizes) if sizes else 0
    if t_max == 0:                                     # no GT anywhere → all background
        empty = torch.empty(0, dtype=torch.int64)
        return [(empty, empty) for _ in range(bs)]

    # (5) pack the block-diagonal cost into [bs, G, 300, T_max], padded with 0
    offsets = np.cumsum([0, *sizes])
    cost_b = cost_matrix.new_zeros(bs, group_detr, g_num_queries, t_max)
    for i, n in enumerate(sizes):
        if n:
            cost_b[i, :, :, :n] = cost_matrix[
                i, :usable, offsets[i]:offsets[i + 1]
            ].reshape(group_detr, g_num_queries, n)

    # (6) one batched GPU solve + one host transfer
    assign = batch_linear_assignment(
        cost_b.reshape(bs * group_detr, g_num_queries, t_max).contiguous()
    ).reshape(bs, group_detr, g_num_queries).to("cpu")

    group_base = torch.arange(group_detr, dtype=torch.int64) * g_num_queries
    results = []
    for i, n in enumerate(sizes):
        if n == 0:
            empty = torch.empty(0, dtype=torch.int64)
            results.append((empty, empty)); continue
        a = assign[i]                                   # [G, 300]
        valid = (a >= 0) & (a < n)                      # drop fake/padded matches
        gq = valid.nonzero(as_tuple=False)
        g_idx, q_idx = gq[:, 0], gq[:, 1]
        query_idx  = (q_idx + group_base[g_idx]).to(torch.int64)
        target_idx = a[g_idx, q_idx].to(torch.int64)
        results.append((query_idx, target_idx))
    return results
```

(One detail from elsewhere in the file: before this runs, the cost matrix is passed through
`_sanitize_cost_matrix`, which replaces any non-finite entries — `nan`/`inf` from edge cases — with a
large finite value, **on the GPU**. The solver needs finite inputs. This is the same safety step the
CPU path does, just kept on-device.)

---

## 7. Is it really exact? (Yes — and we test it)

A faster matcher is worthless if it changes *which* boxes match, because that changes the training signal.
Unlike the approximate Sinkhorn solver (which reached only ~half the mAP and was rejected),
`batch_linear_assignment` computes the **true optimum**, so by construction it produces the same matches
as SciPy (up to ties, which are arbitrary anyway and don't affect the permutation-invariant loss).

We don't take that on faith — there's a GPU test in
[`tests/models/test_matcher.py`](../tests/models/test_matcher.py) that builds random problems and asserts the
`cuda_lap` matches are **bit-identical** to SciPy's, including the Group-DETR structure and an
empty-target image. It passes across seeds. So adopting `cuda_lap` is **convergence-safe by
construction** — no accuracy A/B needed.

---

## 8. Does it actually help? (Measured)

On a **dedicated** H200 (GPU 3), `dinov3-base`, `bs=16`, 576 px (3-run means, reproducible to a few ms):

| matcher | matcher time / step | full step | GPU utilization |
|---|---:|---:|---:|
| SciPy (original) | 535 ms | 773 ms | 19% |
| SciPy + "Tier-1" (finiteness check moved to GPU) | 308 ms | 527 ms | 36% |
| **`cuda_lap`** | **153 ms** | **384 ms** | **83%** |

`cuda_lap` is ~**27% faster than the best CPU variant** and ~**50% faster than the original**, and it's
the only version that actually makes the step **GPU-bound** (83% utilization vs the GPU idling most of
the time before). That matches the theory: we moved the assignment off the stalling CPU path and onto the
GPU.

### 8.1 Honest caveats

- **It needs a CUDA toolkit to build.** `torch-linear-assignment` compiles a CUDA kernel, so the machine
  needs `nvcc` (a `-devel`/NGC container, or `apt install cuda-nvcc-…`). A plain runtime image won't build
  it. Where it's unavailable, the **fallback is SciPy + Tier-1**.
- **It's contention-sensitive.** Because it's GPU-bound, a second job sharing the GPU slows it much more
  than the CPU-bound SciPy matcher. Early measurements on a *shared* GPU even made it look ~6% *slower* —
  that was contention, not the algorithm; a verified-idle GPU showed the real ~27% win. Measure on an
  exclusive GPU.
- **Its cost is data-dependent.** The GPU auction solver does more work when images have more GT boxes, so
  the exact speedup varies with the dataset's object density.

---

## 9. Mental-model summary

- The matcher answers "which prediction is graded against which ground-truth box" by solving a
  **minimum-cost one-to-one assignment** (LSAP / Hungarian).
- RF-DETR solves **hundreds of these per step** (one per image × per query-group × per decoder layer),
  and the classic SciPy path does them on the **CPU**, stalling the GPU.
- `cuda_lap` does the **identical, exact** assignment **on the GPU for the whole batch in one call**.
- The enabling trick: **pad** every problem to a common size with a **constant** cost, solve them all at
  once, then **discard the padded matches** — provably without changing the real answer.
- Result: same training, ~27–50% faster steps, GPU finally busy — *when* you have a CUDA toolkit and a
  reasonably exclusive GPU.

---

### Appendix: the design space at a glance

| solver | exact? | runs on | per-step cost | verdict |
|---|---|---|---|---|
| SciPy (`scipy`) | ✅ | CPU | high (stalls GPU) | baseline / fallback |
| `lap` (lapjv) | ✅ | CPU | higher (mis-fit to tall matrices) | ❌ |
| Sinkhorn | ❌ (approximate) | GPU | low | ❌ −half mAP |
| **`cuda_lap`** | ✅ | **GPU** | **lowest, GPU-bound** | ✅ best (with a CUDA toolkit) |
