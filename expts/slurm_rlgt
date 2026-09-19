#!/usr/bin/env bash
set -e

source .venv/bin/activate

DATASET="rel-f1"
TASK="driver-position"

LR=0.0001
NUM_NEIGHBORS=300
NUM_GLOBAL_CENTROIDS=4096
NUM_WORKERS=1
SEED=0

GT_CONV_TYPE="full"
CHANNELS=64
BATCH_SIZE=256

EPOCHS=100
MAX_STEPS_PER_EPOCH=100

# Bu ikisini her job için değiştir
NUM_LAYERS=$1
DROPOUT=$2

RUN_NAME="relgt-full-l${NUM_LAYERS}-64-dro${DROPOUT}-BS256-MAX100"

OUT_DIR="results/${RUN_NAME}"

mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=1 main_node.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --precompute \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --num_neighbors "$NUM_NEIGHBORS" \
    --num_layers "$NUM_LAYERS" \
    --gt_conv_type "$GT_CONV_TYPE" \
    --channels "$CHANNELS" \
    --num_centroids "$NUM_GLOBAL_CENTROIDS" \
    --max_steps_per_epoch "$MAX_STEPS_PER_EPOCH" \
    --num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --ff_dropout "$DROPOUT" \
    --attn_dropout "$DROPOUT" \
    --run_name "$RUN_NAME" \
    --out_dir "$OUT_DIR"
