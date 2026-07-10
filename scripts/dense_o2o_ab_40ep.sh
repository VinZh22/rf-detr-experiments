#!/usr/bin/env bash
# ------------------------------------------------------------------------
# RF-DETR — Dense O2O vs Group DETR x IA-BCE vs MAL  — 40-epoch rerun, ONE GPU
#
# Follow-up to the 20-epoch 2x2 (see experiment_notes/denseo2o_vs_groupdetr_ab.md): there the Dense
# O2O arms were still rising steeply at epoch 19 while the Group DETR arms had plateaued by epoch 14,
# so the short budget structurally favored the faster-converging Group DETR. This rerun gives Dense
# O2O room to converge (40 epochs, cosine over the full run) for a fair ceiling comparison.
#
#   arm             | matching strategy          | loss
#   ----------------|----------------------------|------
#   denseo2o_mal    | Dense O2O  (group_detr=1)  | MAL      (run first — the open question)
#   denseo2o_iabce  | Dense O2O  (group_detr=1)  | IA-BCE
#   group_mal       | Group DETR (group_detr=13) | MAL
#   group_iabce     | Group DETR (group_detr=13) | IA-BCE
#
# All arms: dinov3-small from scratch, EMA, 40 epochs, seed 42, native resolution, cuda_lap matcher,
# eval every 5 epochs. Dense O2O: mosaic p=0.8, mixup p=0.5, closed for the final 10 epochs (1/4 of
# training, same proportion as the 20ep run's close=5).
#
# Single GPU (GPU 4), all 4 arms SEQUENTIAL, fully detached (survives logout). ~3.5h/arm => ~14h total.
# Separate output dir from the 20ep run so those results are preserved.
#
# Monitor:  tail -f /workspace/doclaynet_runs/denseo2o_ab_40ep/gpu4.log
# Results:  <arm>/metrics.csv  column val/ema_mAP_50_95
# ------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

# HF_TOKEN for the gated DINOv3 backbone (cached locally, so this is a no-op if already downloaded).
[ -f .env ] && set -a && . ./.env && set +a
export RF_HOME="${RF_HOME:-/workspace/rf_home/models}"

# Keep ALL scratch (atomic checkpoint temp, eval temp, mpl cache) on the 21 TB /workspace volume — the
# container root overlay ('/') runs ~full and a /tmp-on-/ ENOSPC crashed the first 20ep attempt.
export TMPDIR=/workspace/tmp
export MPLCONFIGDIR=/workspace/tmp/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

DS=datasets/DocLayNetReduced
OUT=/workspace/doclaynet_runs/denseo2o_ab_40ep
PY=.venv/bin/python
COMMON="--dataset-dir $DS --model dinov3-small --from-scratch --use-ema --epochs 40 --eval-interval 5 --seed 42 --batch-size 16 --num-workers 16 --checkpoint-interval 10000"
DENSE="--group-detr 1 --dense-o2o --mosaic-prob 0.8 --mixup-prob 0.5 --close-mosaic-epochs 10"
mkdir -p "$OUT"
for a in group_iabce group_mal denseo2o_iabce denseo2o_mal; do rm -rf "${OUT:?}/$a"; done

arm() { local n="$1"; shift; echo "$PY scripts/train_rfdetr_dataset.py $COMMON --adapted-dir $OUT/$n/_rf --output-dir $OUT/$n $*"; }

# One GPU, all four arms sequential. '; ' (not '&&') so a failure in one arm does not block the rest.
CUDA_VISIBLE_DEVICES=4 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh "$OUT/gpu4.log" bash -lc "
  $(arm denseo2o_mal   $DENSE --mal-loss) ;
  $(arm denseo2o_iabce $DENSE) ;
  $(arm group_mal      --mal-loss) ;
  $(arm group_iabce)
"

echo "launched 4 arms (40ep, sequential) on GPU 4 -> $OUT"
echo "monitor: tail -f $OUT/gpu4.log     results: $OUT/<arm>/metrics.csv (val/ema_mAP_50_95)"
