#!/usr/bin/env bash
set -e  # exit immediately on any error

############################
# Config 
############################

GPU_ID=0
PORT=29130


DATASET=("rel-f1")
TASK=("driver-position")
FF_DROPOUT=(0.2 0.3 0.5 0.2 0.3 0.5 0.3 0.5 )
BATCH_SIZE=(128 128 128 64 64 64 32 32)
MAX_STEPS_PER_EPOCH=(10 10 10 10 10 10 10 10 10)
EPOCHS=(10 10 10 10 10 10 10 10 10)
ATTN_DROPOUT=(0.2 0.3 0.5 0.2 0.3 0.5 0.3 0.5 )
NUM_NEIGHBORS=100
NUM_LAYERS=1
CHANNELS=64
GT_CONV_TYPE="full"

LR=0.0001
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