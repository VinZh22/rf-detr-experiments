#!/usr/bin/env bash
# ------------------------------------------------------------------------
# RF-DETR — Dense O2O vs Group DETR  x  IA-BCE vs MAL   (2x2 A/B)
#
# Tests whether DEIM's Dense O2O (data-side densification via mosaic+mixup) is a
# viable *replacement* for Group DETR (query-group densification), crossed with the
# two IoU-aware classification losses. Hypothesis under test: "no point running Dense
# O2O and Group DETR together" — so the Dense O2O arms DROP group_detr to 1, while the
# Group DETR arms keep the variant default group_detr=13 with no mosaic.
#
#   arm             | matching strategy          | loss
#   ----------------|----------------------------|------
#   group_iabce     | Group DETR (group_detr=13) | IA-BCE   (= current RF-DETR default)
#   group_mal       | Group DETR (group_detr=13) | MAL
#   denseo2o_iabce  | Dense O2O  (group_detr=1)  | IA-BCE
#   denseo2o_mal    | Dense O2O  (group_detr=1)  | MAL
#
# All arms: dinov3-small from scratch, EMA, 20 epochs, seed 42, native resolution,
# cuda_lap matcher. Dense O2O: mosaic p=0.8, mixup p=0.5, closed for the final 5 epochs.
#
# Layout: 4 arms across 2 GPUs, one Dense + one Group arm per GPU (sequential, to
# balance wall-clock since Dense arms are heavier). Dense arm runs FIRST on each GPU so
# the newer code path surfaces any error early. Fully detached — survives logout.
#   GPU 3:  denseo2o_mal    -> group_iabce
#   GPU 4:  denseo2o_iabce  -> group_mal
#
# Monitor:  tail -f /workspace/doclaynet_runs/denseo2o_ab/gpu3.log
# Results:  <output-dir>/metrics.csv  column ema_mAP_50_95
# ------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

# DINOv3 backbone weights are gated on the HF Hub; export HF_TOKEN from .env (the weights are also
# cached locally, so this is a no-op if already downloaded). RF_HOME matches the prior DocLayNet runs.
[ -f .env ] && set -a && . ./.env && set +a
export RF_HOME="${RF_HOME:-/workspace/rf_home/models}"

# The container root overlay ('/') is tiny and runs ~full; keep ALL scratch (atomic checkpoint temp,
# pycocotools eval temp, matplotlib cache) on the 21 TB /workspace volume so training never hits ENOSPC.
export TMPDIR=/workspace/tmp
export MPLCONFIGDIR=/workspace/tmp/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

DS=datasets/DocLayNetReduced
OUT=/workspace/doclaynet_runs/denseo2o_ab
PY=.venv/bin/python
COMMON="--dataset-dir $DS --model dinov3-small --from-scratch --use-ema --epochs 20 --eval-interval 5 --seed 42 --batch-size 16 --num-workers 16 --checkpoint-interval 10000"
DENSE="--group-detr 1 --dense-o2o --mosaic-prob 0.8 --mixup-prob 0.5 --close-mosaic-epochs 5"
mkdir -p "$OUT"

# Start each arm fresh (a previous crashed run leaves a partial checkpoint + metrics.csv that PTL
# would otherwise append to / warn about). Keep the top-level launcher logs.
for a in group_iabce group_mal denseo2o_iabce denseo2o_mal; do rm -rf "${OUT:?}/$a"; done

# One arm as a single-line command (its python harness prints a per-arm banner).
arm() { local n="$1"; shift; echo "$PY scripts/train_rfdetr_dataset.py $COMMON --adapted-dir $OUT/$n/_rf --output-dir $OUT/$n $*"; }

# GPU 3 chain: denseo2o_mal ; group_iabce   ('; ' so arm2 runs even if arm1 fails)
CUDA_VISIBLE_DEVICES=3 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh "$OUT/gpu3.log" \
  bash -lc "$(arm denseo2o_mal $DENSE --mal-loss) ; $(arm group_iabce)"

# GPU 4 chain: denseo2o_iabce ; group_mal
CUDA_VISIBLE_DEVICES=4 RFDETR_MATCHER_SOLVER=cuda_lap scripts/run_detached.sh "$OUT/gpu4.log" \
  bash -lc "$(arm denseo2o_iabce $DENSE) ; $(arm group_mal --mal-loss)"

echo "launched 4 arms across GPU 3 & 4 -> $OUT"
echo "monitor: tail -f $OUT/gpu3.log $OUT/gpu4.log"
