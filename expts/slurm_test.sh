```bash
#!/usr/bin/env bash
set -e


source .venv/bin/activate


# ============================================================
# Fixed experiment setup
# ============================================================

DATASET="rel-f1"
TASK="driver-position"

LR=0.0001

# Local neighbors used during token preparation
NUM_NEIGHBORS=300

# Global centroid/token count
NUM_GLOBAL_CENTROIDS=4096

NUM_WORKERS=1
SEED=0

GT_CONV_TYPE="full"
CHANNELS=64


# ============================================================
# Dataset size
#
# Set this according to the number of training nodes:
#
#   SMALL -> < 1M training nodes
#   LARGE -> > 1M training nodes
# ============================================================

DATASET_SIZE="SMALL"


# ============================================================
# Experiment configurations
#
# SMALL:
#   Layers  = 1, 4, 8
#   Dropout = 0.3, 0.4, 0.5
#   Batch   = 256
#
# LARGE:
#   Layers  = 4
#   Dropout = 0.3, 0.4, 0.5
#   Batch   = 1024
# ============================================================

if [[ "${DATASET_SIZE}" == "SMALL" ]]; then

    NUM_LAYERS=(
        1
        1
        1
        4
        4
        4
        8
        8
        8
    )

    DROPOUT=(
        0.3
        0.4
        0.5
        0.3
        0.4
        0.5
        0.3
        0.4
        0.5
    )

    BATCH_SIZE=(
        256
        256
        256
        256
        256
        256
        256
        256
        256
    )

else

    NUM_LAYERS=(
        4
        4
        4
    )

    DROPOUT=(
        0.3
        0.4
        0.5
    )

    BATCH_SIZE=(
        1024
        1024
        1024
    )

fi


# ============================================================
# Training budget
#
# Set these however you want for your experiments.
# ============================================================

MAX_STEPS_PER_EPOCH=100
EPOCHS=100


# ============================================================
# Run experiments sequentially
# ============================================================

NUM_EXPERIMENTS=${#NUM_LAYERS[@]}

echo "=========================================="
echo "Dataset:        ${DATASET}"
echo "Task:           ${TASK}"
echo "Dataset size:   ${DATASET_SIZE}"
echo "Experiments:    ${NUM_EXPERIMENTS}"
echo "K neighbors:    ${NUM_NEIGHBORS}"
echo "B centroids:    ${NUM_GLOBAL_CENTROIDS}"
echo "Learning rate:  ${LR}"
echo "=========================================="


for ((i=0; i<NUM_EXPERIMENTS; i++)); do

    num_layers=${NUM_LAYERS[$i]}
    dropout=${DROPOUT[$i]}
    batch_size=${BATCH_SIZE[$i]}

    ff_dropout="${dropout}"
    attn_dropout="${dropout}"

    dro=${dropout//./}

    RUN_NAME="relgt-${GT_CONV_TYPE}-l${num_layers}-${CHANNELS}-dro${dro}-BS${batch_size}-MAX${MAX_STEPS_PER_EPOCH}"

    OUT_DIR="results/${RUN_NAME}"

    mkdir -p "${OUT_DIR}"

    echo ""
    echo "=========================================="
    echo "Experiment $((i + 1)) / ${NUM_EXPERIMENTS}"
    echo "=========================================="
    echo "Layers:          ${num_layers}"
    echo "Dropout:         ${dropout}"
    echo "Batch size:      ${batch_size}"
    echo "K neighbors:     ${NUM_NEIGHBORS}"
    echo "B centroids:     ${NUM_GLOBAL_CENTROIDS}"
    echo "Learning rate:   ${LR}"
    echo "Run name:        ${RUN_NAME}"
    echo "=========================================="


    torchrun \
        --nproc_per_node=1 \
        main_node.py \
        --dataset "${DATASET}" \
        --task "${TASK}" \
        --precompute \
        --seed "${SEED}" \
        --batch_size "${batch_size}" \
        --num_neighbors "${NUM_NEIGHBORS}" \
        --num_layers "${num_layers}" \
        --gt_conv_type "${GT_CONV_TYPE}" \
        --channels "${CHANNELS}" \
        --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}" \
        --num_workers "${NUM_WORKERS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --ff_dropout "${ff_dropout}" \
        --attn_dropout "${attn_dropout}" \
        --run_name "${RUN_NAME}" \
        --out_dir "${OUT_DIR}"

    echo "Experiment $((i + 1)) finished."

done


echo ""
echo "=========================================="
echo "All ${NUM_EXPERIMENTS} experiments finished."
echo "=========================================="
```
