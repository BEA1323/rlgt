```bash
#!/usr/bin/env bash
set -e

cd ~/burak/rlgt-dryrun
source .venv/bin/activate

# ============================================================
# Fixed parameters
# ============================================================

DATASET="rel-f1"
TASK="driver-position"

NUM_NEIGHBORS=100
NUM_LAYERS=1
CHANNELS=64
GT_CONV_TYPE="full"

LR=0.0001
NUM_WORKERS=1
SEED=0


# ============================================================
# Experiment parameters
#
# Add/remove values here.
# All arrays must have the SAME length.
# ============================================================

FF_DROPOUT=(
    0.2
    0.3
    0.5
)

ATTN_DROPOUT=(
    0.2
    0.3
    0.5
)

BATCH_SIZE=(
    128
    64
    32
)

MAX_STEPS_PER_EPOCH=(
    10
    10
    10
)

EPOCHS=(
    100
    100
    100
)


# ============================================================
# Number of experiments
# ============================================================

NUM_EXPERIMENTS=${#FF_DROPOUT[@]}

echo "Running ${NUM_EXPERIMENTS} experiments"
echo "===================================="


# ============================================================
# Run experiments sequentially
# ============================================================

for ((i=0; i<NUM_EXPERIMENTS; i++)); do

    ff_dropout=${FF_DROPOUT[$i]}
    attn_dropout=${ATTN_DROPOUT[$i]}
    batch_size=${BATCH_SIZE[$i]}
    max_steps=${MAX_STEPS_PER_EPOCH[$i]}
    epochs=${EPOCHS[$i]}

    ff_dro=${ff_dropout//./}
    attn_dro=${attn_dropout//./}

    RUN_NAME="relgt-${GT_CONV_TYPE}-l${NUM_LAYERS}-${CHANNELS}-ff${ff_dro}-attn${attn_dro}-BS${batch_size}-MAX${max_steps}"

    OUT_DIR="results/${RUN_NAME}"

    mkdir -p "${OUT_DIR}"

    echo ""
    echo "===================================="
    echo "Experiment $((i + 1)) / ${NUM_EXPERIMENTS}"
    echo "Run name:       ${RUN_NAME}"
    echo "FF dropout:     ${ff_dropout}"
    echo "Attn dropout:   ${attn_dropout}"
    echo "Batch size:     ${batch_size}"
    echo "Max steps:      ${max_steps}"
    echo "Epochs:         ${epochs}"
    echo "===================================="

    torchrun \
        --nproc_per_node=1 \
        main_node.py \
        --dataset "${DATASET}" \
        --task "${TASK}" \
        --precompute \
        --seed "${SEED}" \
        --batch_size "${batch_size}" \
        --num_neighbors "${NUM_NEIGHBORS}" \
        --num_layers "${NUM_LAYERS}" \
        --gt_conv_type "${GT_CONV_TYPE}" \
        --channels "${CHANNELS}" \
        --max_steps_per_epoch "${max_steps}" \
        --num_workers "${NUM_WORKERS}" \
        --epochs "${epochs}" \
        --lr "${LR}" \
        --ff_dropout "${ff_dropout}" \
        --attn_dropout "${attn_dropout}" \
        --run_name "${RUN_NAME}" \
        --out_dir "${OUT_DIR}"

    echo "Experiment $((i + 1)) finished."

done

echo ""
echo "===================================="
echo "All ${NUM_EXPERIMENTS} experiments finished."
echo "===================================="
```
