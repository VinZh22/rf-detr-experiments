# DocLayNet A/B: backbone & loss convergence study

A controlled 40-epoch comparison on DocLayNet isolating two questions:

1. **Backbone** — how much of DINOv3's advantage over DINOv2 is *size* vs *SSL/architecture*?
2. **Loss** — does MAL beat IA-BCE?

All runs use the same detector config, same data pipeline, same seed, fixed resolution
(`multi_scale=False`), EMA, and the exact [`cuda_lap`](cuda_lap_matcher.md) GPU matcher (so matching is
identical to scipy — it only affects speed, not which boxes get matched). Validation every 5 epochs.
Metric below is **EMA mAP@[.50:.95]** on the val split.

> A third study on the same model/data — **AdamW vs Muon optimizer** — lives in its own write-up:
> [muon_optimizer.md](muon_optimizer.md).

## Runs

| run | backbone | params | SSL init | loss | final EMA mAP |
|---|---|---|---|---|---|
| `base_iabce`    | DINOv2-S (ViT-S/14) | ~23.5M | ✅ real DINOv2-S SSL | IA-BCE | 0.2356 |
| `base_mal`      | DINOv2-S (ViT-S/14) | ~23.5M | ✅ real DINOv2-S SSL | MAL    | 0.2461 |
| `dinov3_small`  | DINOv3-S (ViT-S/16) | ~23.0M | ✅ DINOv3-S SSL      | IA-BCE | 0.3434 |
| `dinov3_iabce`  | DINOv3-B (ViT-B/16) | ~87.5M | ✅ DINOv3-B SSL      | IA-BCE | 0.3631 |

Heads trained from scratch in every run; only the backbone carries SSL pretraining.

## Convergence curves (EMA mAP@[.50:.95])

| epoch | DINOv2-S IA-BCE | DINOv2-S MAL | DINOv3-S IA-BCE | DINOv3-B IA-BCE |
|---:|---:|---:|---:|---:|
| 4  | 0.100 | 0.109 | 0.240 | 0.269 |
| 9  | 0.154 | 0.163 | 0.298 | 0.329 |
| 14 | 0.189 | 0.200 | 0.324 | 0.348 |
| 19 | 0.211 | 0.220 | 0.338 | 0.359 |
| 24 | 0.225 | 0.235 | 0.339 | 0.362 |
| 29 | 0.232 | 0.243 | 0.343 | 0.364 |
| 34 | 0.234 | 0.244 | 0.343 | 0.364 |
| 39 | 0.236 | 0.246 | 0.343 | 0.363 |

## Findings

### 1. The DINOv3 advantage is the backbone (SSL/architecture), not size

The clean, size-matched comparison is **DINOv3-S vs DINOv2-S** — both ~23M params, both from
scratch, same loss (IA-BCE), same everything but the backbone:

- **DINOv3-S 0.343** vs **DINOv2-S 0.236** → **+0.107 mAP at matched size.**
- DINOv3-S already beats DINOv2-S's *final* mAP by epoch 4 (0.240 vs 0.236).

Adding 3.7× more parameters on top (DINOv3-S → DINOv3-B) only buys a further **+0.020**
(0.343 → 0.363). So of the headline +0.128 gap (DINOv3-B over DINOv2-S), **~84% comes from the
backbone's SSL/architecture and only ~16% from the 3.7× size increase.**

```
DINOv2-S  ──(+0.107: SSL/arch)──▶  DINOv3-S  ──(+0.020: 3.7× size)──▶  DINOv3-B
  0.236                              0.343                              0.363
```

### 2. DINOv3 also converges far faster

DINOv3-S reaches 0.240 at epoch 4 — a level DINOv2-S never reaches in 40 epochs. Both DINOv3
variants are essentially converged by epoch ~19–24, while DINOv2-S is still crawling upward at
epoch 39. The SSL features are immediately useful to the detector heads.

### 3. MAL > IA-BCE, consistently but modestly

On the DINOv2-S backbone, **MAL beats IA-BCE at every checkpoint** by a steady ~+0.01
(final 0.246 vs 0.236). Small but monotonic and noise-free across the whole curve — MAL is the
better default loss. (Loss A/B was run on DINOv2-S; the DINOv3 runs both use IA-BCE, so the MAL
gain is expected to stack on top of the DINOv3 numbers rather than overlap with them.)

## Caveats

- Single seed per arm; the ~+0.01 MAL gain is consistent across 8 checkpoints but within
  plausible run-to-run noise for a single seed — treat it as "MAL ≥ IA-BCE, likely better,"
  not a precise effect size.
- DINOv2 uses ViT/14 patches, DINOv3 uses ViT/16 — a native property of each SSL family, not a
  knob we set. The size-matched arms equalize *parameter count*, not patch size.
- DocLayNet only; document layout is a domain where strong general-purpose visual features
  (DINOv3) may help more than on natural-image detection. The *direction* should generalize; the
  *magnitude* may not.
- mAP is EMA-weights val mAP at fixed resolution; absolute numbers are not comparable to
  multi-scale / TTA leaderboard figures.

## Reproduce

From the repo root after `uv sync --all-groups`. Each arm is one detached run (see
[`scripts/run_detached.sh`](../scripts/run_detached.sh)); they differ only in `--model` and `--mal-loss`.

```bash
DS=datasets/DocLayNetReduced
common="--dataset-dir $DS --epochs 40 --batch-size 16 --num-workers 16 \
        --from-scratch --use-ema --eval-interval 5 --seed 42"

# Backbone arms (loss = IA-BCE, the default): DINOv2-small vs DINOv3-small vs DINOv3-base
CUDA_VISIBLE_DEVICES=3 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh out/base_iabce.log \
  .venv/bin/python scripts/train_rfdetr_dataset.py $common --model base         --output-dir out/base_iabce
CUDA_VISIBLE_DEVICES=4 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh out/dinov3_small.log \
  .venv/bin/python scripts/train_rfdetr_dataset.py $common --model dinov3-small --output-dir out/dinov3_small
CUDA_VISIBLE_DEVICES=5 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh out/dinov3_base.log \
  .venv/bin/python scripts/train_rfdetr_dataset.py $common --model dinov3-base  --output-dir out/dinov3_base

# Loss arm: same DINOv2-small backbone, MAL instead of IA-BCE (add --mal-loss)
CUDA_VISIBLE_DEVICES=6 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh out/base_mal.log \
  .venv/bin/python scripts/train_rfdetr_dataset.py $common --model base --mal-loss --output-dir out/base_mal
```

`--model base` loads the real DINOv2-small SSL backbone (patch-14) from scratch — the correct DINOv2 arm
(a patch-16 `medium` would get a *random* backbone, see git history). Read each run's mAP curve from
`metrics.csv` (column `ema_mAP_50_95`). Drop `RFDETR_MATCHER_SOLVER=cuda_lap` if you have no CUDA toolkit
(SciPy matcher; same result, slower — see [cuda_lap_matcher.md](cuda_lap_matcher.md)).
