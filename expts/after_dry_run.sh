#!/usr/bin/env bash

GPU_ID=0
BASE_PORT=29130

DATASET="rel-f1"
TASK="driver-top3"
ABLATE="none"

CHANNELS=32
NUM_LAYERS=1
NUM_HEADS=4
NUM_NEIGHBORS=100

BATCH_SIZE=64
EPOCHS=10
MAX_STEPS=100

LR=0.0001
WARMUP_STEPS=10
WEIGHT_DECAY=0.00001

DROPOUT=0.3

GT_CONV_TYPE="global"

RUN_NAME="relgt-ablate-${ABLATE}-l${NUM_LAYERS}-${CHANNELS}-dropout${DROPOUT//./}-BS${BATCH_SIZE}-MAX${MAX_STEPS}"
OUT_DIR="results/${RUN_NAME}"

mkdir -p "$OUT_DIR"

echo "========================================"
echo "Running: $RUN_NAME"
echo "Dataset: $DATASET"
echo "Task:    $TASK"
echo "Ablate:  $ABLATE"
echo "========================================"

CUDA_VISIBLE_DEVICES=$GPU_ID \
torchrun \
    --nproc_per_node=1 \
    --master_port=$BASE_PORT \
    main_node.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --precompute \
    --seed 0 \
    --batch_size "$BATCH_SIZE" \
    --num_neighbors "$NUM_NEIGHBORS" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --gt_conv_type "$GT_CONV_TYPE" \
    --ablate "$ABLATE" \
    --channels "$CHANNELS" \
    --max_steps_per_epoch "$MAX_STEPS" \
    --num_workers 1 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --ff_dropout "$DROPOUT" \
    --attn_dropout "$DROPOUT" \
    --weight_decay "$WEIGHT_DECAY" \
    --run_name "$RUN_NAME" \
    --out_dir "$OUT_DIR"

if [ $? -eq 0 ]; then
    echo "Finished successfully: $RUN_NAME"
else
    echo "Experiment FAILED: $RUN_NAME"
    exit 1
fi