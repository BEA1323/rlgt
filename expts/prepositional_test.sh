#!/usr/bin/env bash
set -e  # exit immediately on any error

############################
# Config — edit as needed
############################
DATASET="rel-f1"
TASK="driver-top3"

GPU_ID=0
PORT=29130

BATCH_SIZE=64
NUM_NEIGHBORS=100
NUM_LAYERS=1
CHANNELS=64
GT_CONV_TYPE="full"

EPOCHS=10
MAX_STEPS_PER_EPOCH=300
LR=0.0001
FF_DROPOUT=0.3
ATTN_DROPOUT=0.3
NUM_WORKERS=1
SEED=0

RUN_NAME="relgt-${GT_CONV_TYPE}-l${NUM_LAYERS}-${DATASET}-${TASK}"
OUT_DIR="results/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

############################
# Launch
############################
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching ${DATASET}(${TASK}) on GPU ${GPU_ID}"

CUDA_VISIBLE_DEVICES=${GPU_ID} \
torchrun \
    --nproc_per_node=1 \
    --master_port="${PORT}" \
    main_node.py \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --precompute \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_neighbors "${NUM_NEIGHBORS}" \
    --num_layers "${NUM_LAYERS}" \
    --gt_conv_type "${GT_CONV_TYPE}" \
    --channels "${CHANNELS}" \
    --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}" \
    --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --ff_dropout "${FF_DROPOUT}" \
    --attn_dropout "${ATTN_DROPOUT}" \
    --run_name "${RUN_NAME}" \
    --out_dir "${OUT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run finished."