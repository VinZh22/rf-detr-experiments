# DocLayNet A/B: Dense O2O vs Group DETR × IA-BCE vs MAL

A controlled 2×2 on DocLayNet testing whether DEIM's **Dense O2O** (data-side densification via
mosaic+mixup) is a viable *replacement* for RF-DETR's **Group DETR** (query-group densification),
crossed with the two IoU-aware classification losses (IA-BCE vs MAL).

Both methods attack the same problem — the sparse positive supervision of one-to-one Hungarian
matching that slows DETR convergence — from different sides: Group DETR adds K=13 query groups per
target; Dense O2O keeps one-to-one matching but packs ~4–8× more targets into each image via
mosaic+mixup. They are alternatives, so the arms treat them as **mutually exclusive**: the Dense O2O
arms set `group_detr=1` (Group DETR off), the Group DETR arms use no mosaic. The question is whether,
at equal budget, replacing Group DETR with Dense O2O holds or loses accuracy.

All arms: dinov3-small from scratch, EMA, seed 42, fixed (native) resolution, `multi_scale=False`,
[`cuda_lap`](cuda_lap_matcher.md) matcher (matching identical to scipy, just faster). Dense O2O:
mosaic p=0.8, mixup p=0.5, closed for the final 1/4 of training (clean-image fine-tune). Validation
every 5 epochs. Metric is **EMA mAP@[.50:.95]**.

This was run at **two budgets** — first 20 epochs (close-mosaic at epoch 15), then a 40-epoch rerun
(close-mosaic at epoch 30) because Dense O2O had not converged at 20ep. **The verdict flips between
the two**, so read both. TL;DR: at 20ep Group DETR wins; at 40ep **Dense O2O + MAL wins overall** but
Dense O2O + IA-BCE still loses — i.e. Dense O2O can replace (and beat) Group DETR, but only with MAL
and a long-enough schedule.

> Background on why Dense O2O and Group DETR are alternatives (not complementary), and the chronology
> (Group DETR ICCV'23 predates DEIM CVPR'25, which positions Dense O2O as a substitute for the
> one-to-many family incl. Group DETR): [`deim_rfdetr_review.md`](deim_rfdetr_review.md).

## Runs (final EMA mAP@[.50:.95])

| run | matching | group_detr | loss | **40ep** | 20ep |
|---|---|---|---|---|---|
| `denseo2o_mal`   | Dense O2O  | 1  | MAL    | **0.3507** 🏆 | 0.2946 |
| `group_mal`      | Group DETR | 13 | MAL    | 0.3431 | 0.3069 |
| `group_iabce`    | Group DETR | 13 | IA-BCE | 0.3417 | 0.3049 |
| `denseo2o_iabce` | Dense O2O  | 1  | IA-BCE | 0.3341 | 0.2856 |

## Convergence curves (EMA mAP@[.50:.95])

**40-epoch** (close-mosaic at epoch 30):

| epoch | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| denseo2o_mal   | 0.177 | 0.250 | 0.285 | 0.304 | 0.318 | 0.325 | 0.350 | **0.351** |
| group_mal      | 0.246 | 0.298 | 0.320 | 0.333 | 0.338 | 0.343 | 0.342 | **0.343** |
| group_iabce    | 0.237 | 0.295 | 0.321 | 0.337 | 0.340 | 0.341 | 0.342 | **0.342** |
| denseo2o_iabce | 0.169 | 0.241 | 0.275 | 0.287 | 0.300 | 0.307 | 0.332 | **0.334** |

**20-epoch** (close-mosaic at epoch 15):

| epoch | 4 | 9 | 14 | 19 |
|---|---:|---:|---:|---:|
| group_mal      | 0.248 | 0.292 | 0.306 | **0.307** |
| group_iabce    | 0.237 | 0.287 | 0.304 | **0.305** |
| denseo2o_mal   | 0.175 | 0.239 | 0.264 | **0.295** |
| denseo2o_iabce | 0.163 | 0.228 | 0.254 | **0.286** |

## Findings

### 1. The verdict flips with budget: at 40ep, Dense O2O + MAL is the best arm

At **20ep**, Group DETR wins both losses (swapping in Dense O2O costs −0.012 MAL / −0.019 IA-BCE). At
**40ep the ranking reverses for the MAL pairing**: `denseo2o_mal` **0.351** beats `group_mal` **0.343**
(+0.008) and `group_iabce` 0.342 (+0.009). So Dense O2O *can* replace — and beat — Group DETR, but it
needs a long-enough schedule to do so. The 20ep result was a budget artifact, not Dense O2O's ceiling.

### 2. ...but Dense O2O's win depends entirely on MAL

The loss is not a side-knob for Dense O2O — it decides the outcome:

- **Dense O2O:** MAL → 0.351, IA-BCE → 0.334 (**+0.017** from the loss alone).
- **Group DETR:** MAL → 0.343, IA-BCE → 0.342 (**+0.001** — loss barely matters).

So `denseo2o_iabce` (0.334) is the *worst* 40ep arm, below both Group arms. Dense mosaic+mixup
manufactures many low-quality matches; MAL was designed in DEIM precisely to down-weight overconfident
low-IoU boxes, so it is doing the heavy lifting that makes Dense O2O viable. **Dense O2O without MAL is
a net loss vs Group DETR.** The DEIM recipe pairs them for a reason.

### 3. Group DETR converges much faster; Dense O2O pays an early tax but a higher ceiling

Group DETR is far ahead early (epoch-9 ~0.30 vs Dense ~0.24) and plateaus by ~epoch 19–24 (group_mal
0.333→0.343 over its last 20 epochs). Dense O2O starts slow — mosaic+mixup makes every image much
harder — and is still climbing late; the close-mosaic fine-tune (epochs 30→40 on clean images) is the
decisive kick: `denseo2o_mal` jumps **0.325→0.351 (+0.026)** there, overtaking the (already plateaued)
Group arms. Practical implication: if you can only afford a short schedule, use Group DETR; if you can
train long and use MAL, Dense O2O edges ahead.

### 4. MAL ≥ IA-BCE everywhere, but the margin is regime-dependent

MAL ≥ IA-BCE in all four comparisons (20ep & 40ep, both densifiers), consistent with the ~+0.01 edge
from the [backbone/loss study](doclaynet_ab_results.md). The margin is tiny under Group DETR (+0.001
@40ep) and large under Dense O2O (+0.017 @40ep) — MAL matters most exactly where low-quality matches
are most abundant.

## Caveats

- **Single seed per arm.** The decisive 40ep gap (denseo2o_mal 0.351 vs group_mal 0.343 = +0.008) is
  small enough to sit within plausible single-seed noise — treat it as "Dense O2O+MAL is competitive
  with / slightly ahead of Group DETR at 40ep," not a guaranteed win. The larger effects (the −0.017
  IA-BCE penalty for Dense O2O; Group's faster convergence) are robust across the whole curve.
- **DocLayNet only**, document layout. Mosaic stitches document pages into 2×2 collages — a fairly
  unnatural input for this domain; the early-convergence tax may be larger here than on natural images.
- Dense O2O `mosaic_prob=0.8`/`mixup_prob=0.5`/`close=1/4` are untuned first guesses; the schedule is a
  knob that materially affects the Dense arms (the close-mosaic phase alone is worth ~+0.026 mAP).
- The 40ep Group numbers (0.342–0.343) match the prior dinov3-small 40ep baseline (0.343), confirming
  the harness reproduces the known result; the Dense O2O+MAL arm exceeds it.

## Reproduce

Two launchers, both detached, both sourcing `HF_TOKEN` from `.env` and pinning `TMPDIR=/workspace/tmp`
so the atomic checkpoint temp stages on the 21 TB volume rather than the tiny container root overlay (a
`/tmp`-on-`/` ENOSPC crashed the first attempt at the epoch-5 eval — both GPUs staged ~473 MB
checkpoints into a `/` with <1 GB free):

```bash
bash scripts/dense_o2o_ab.sh        # 20ep, 4 arms across GPU 3 & 4 (Dense arm first per GPU), ~3.5h
bash scripts/dense_o2o_ab_40ep.sh   # 40ep, all 4 arms sequential on GPU 4, ~13h overnight
# monitor: tail -f /workspace/doclaynet_runs/denseo2o_ab{,_40ep}/gpu*.log
# results: <arm>/metrics.csv  column val/ema_mAP_50_95
```

Single arm directly (the Dense O2O flags are new; see `scripts/train_rfdetr_dataset.py`):

```bash
# Dense O2O arm (Group DETR off): mosaic+mixup, group_detr=1
RFDETR_MATCHER_SOLVER=cuda_lap TMPDIR=/workspace/tmp .venv/bin/python scripts/train_rfdetr_dataset.py \
  --dataset-dir datasets/DocLayNetReduced --model dinov3-small --from-scratch --use-ema \
  --epochs 20 --eval-interval 5 --seed 42 --batch-size 16 --num-workers 16 --checkpoint-interval 10000 \
  --group-detr 1 --dense-o2o --mosaic-prob 0.8 --mixup-prob 0.5 --close-mosaic-epochs 5 \
  --mal-loss --output-dir /workspace/doclaynet_runs/denseo2o_ab/denseo2o_mal

# Group DETR baseline arm: variant-default group_detr=13, no mosaic
RFDETR_MATCHER_SOLVER=cuda_lap TMPDIR=/workspace/tmp .venv/bin/python scripts/train_rfdetr_dataset.py \
  --dataset-dir datasets/DocLayNetReduced --model dinov3-small --from-scratch --use-ema \
  --epochs 20 --eval-interval 5 --seed 42 --batch-size 16 --num-workers 16 --checkpoint-interval 10000 \
  --mal-loss --output-dir /workspace/doclaynet_runs/denseo2o_ab/group_mal
```

Dense O2O is implemented in `src/rfdetr/datasets/dense_o2o.py` (mosaic+mixup pre-transform, with a
`multiprocessing.Value` epoch gate for the close-mosaic schedule driven by `CloseMosaicCallback`);
tests in `tests/datasets/test_dense_o2o.py`.
