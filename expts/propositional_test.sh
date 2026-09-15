#!/usr/bin/env bash
set -e

############################
# Config
############################

BASE_PORT=29130

# GPUs available on this node
GPU_NODES=(0 )

############################
# Fixed experiment settings
############################

DATASET="rel-f1"
TASK="driver-position"

NUM_NEIGHBORS=100
NUM_LAYERS=1
CHANNELS=64
GT_CONV_TYPE="full"

LR=0.0001
NUM_WORKERS=1
SEED=0

############################
# Experiment sweep
############################

FF_DROPOUT=(0.2 0.3 0.5 0.2 0.3 0.5 0.3 0.5)
ATTN_DROPOUT=(0.2 0.3 0.5 0.2 0.3 0.5 0.3 0.5)
BATCH_SIZE=(128 128 128 64 64 64 32 32)

MAX_STEPS_PER_EPOCH=(10 10 10 10 10 10 10 10)
EPOCHS=(10 10 10 10 10 10 10 10)

############################
# Check array lengths
############################

NUM_EXPERIMENTS=${#FF_DROPOUT[@]}

if [ "${#ATTN_DROPOUT[@]}" -ne "$NUM_EXPERIMENTS" ]; then
    echo "Error: ATTN_DROPOUT has wrong length."
    exit 1
fi

if [ "${#BATCH_SIZE[@]}" -ne "$NUM_EXPERIMENTS" ]; then
    echo "Error: BATCH_SIZE has wrong length."
    exit 1
fi

if [ "${#MAX_STEPS_PER_EPOCH[@]}" -ne "$NUM_EXPERIMENTS" ]; then
    echo "Error: MAX_STEPS_PER_EPOCH has wrong length."
    exit 1
fi

if [ "${#EPOCHS[@]}" -ne "$NUM_EXPERIMENTS" ]; then
    echo "Error: EPOCHS has wrong length."
    exit 1
fi

############################
# Build experiment configs
############################

EXPT_CONFIGS=()

for i in "${!FF_DROPOUT[@]}"; do

    EXPT_CONFIGS+=(
        "${FF_DROPOUT[$i]}|${ATTN_DROPOUT[$i]}|${BATCH_SIZE[$i]}|${MAX_STEPS_PER_EPOCH[$i]}|${EPOCHS[$i]}"
    )

done

TOTAL_EXPERIMENTS=${#EXPT_CONFIGS[@]}
NUM_GPUS=${#GPU_NODES[@]}

echo "=========================================="
echo "Total experiments : ${TOTAL_EXPERIMENTS}"
echo "Available GPUs    : ${NUM_GPUS}"
echo "Dataset           : ${DATASET}"
echo "Task              : ${TASK}"
echo "=========================================="

############################
# GPU check
############################

is_gpu_in_use() {

    local gpu_id=$1

    if nvidia-smi \
        -i "$gpu_id" \
        --query-compute-apps=pid \
        --format=csv,noheader 2>/dev/null \
        | grep -q "[0-9]"; then

        return 0

    else

        return 1

    fi
}

############################
# Logging
############################

log_message() {

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"

}

############################
# GPU status
############################

log_gpu_status() {

    log_message "----------- GPU STATUS -----------"

    for gpu_id in "${GPU_NODES[@]}"; do

        if is_gpu_in_use "$gpu_id"; then

            pid=$(
                nvidia-smi \
                    -i "$gpu_id" \
                    --query-compute-apps=pid \
                    --format=csv,noheader \
                    | head -n1
            )

            mem=$(
                nvidia-smi \
                    -i "$gpu_id" \
                    --query-compute-apps=used_memory \
                    --format=csv,noheader \
                    | head -n1
            )

            log_message "GPU ${gpu_id}: BUSY | PID=${pid} | Memory=${mem}"

        else

            log_message "GPU ${gpu_id}: FREE"

        fi

    done

    log_message "----------------------------------"

}

############################
# Launch experiment
############################

launch_experiment() {

    local config="$1"
    local gpu_id="$2"
    local port="$3"
    local exp_id="$4"

    ############################
    # Parse configuration
    ############################

    IFS='|' read -r \
        ff_dropout \
        attn_dropout \
        batch_size \
        max_steps \
        epochs \
        <<< "$config"

    ############################
    # Format dropout for run name
    ############################

    local ff_dro="${ff_dropout//./}"
    local attn_dro="${attn_dropout//./}"

    ############################
    # Run name
    ############################

    local run_name="relgt-${GT_CONV_TYPE}-l${NUM_LAYERS}-${CHANNELS}-ff${ff_dro}-attn${attn_dro}-BS${batch_size}-MAX${max_steps}"

    local out_dir="results/${run_name}"

    mkdir -p "${out_dir}"

    log_message "Launching experiment ${exp_id}"
    log_message "GPU          : ${gpu_id}"
    log_message "FF dropout   : ${ff_dropout}"
    log_message "Attn dropout : ${attn_dropout}"
    log_message "Batch size   : ${batch_size}"
    log_message "Max steps    : ${max_steps}"
    log_message "Epochs       : ${epochs}"
    log_message "Run name     : ${run_name}"

    ############################
    # Launch
    ############################

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    torchrun \
        --nproc_per_node=1 \
        --master_port="${port}" \
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
        --run_name "${run_name}" \
        --out_dir "${out_dir}" &

    ############################
    # Give process time to start
    ############################

    sleep 10

    if ! is_gpu_in_use "$gpu_id"; then

        log_message "[WARNING] GPU ${gpu_id} does not appear to be running the experiment."

    else

        log_message "[GPU ${gpu_id}] Experiment ${exp_id} successfully started."

    fi

}

############################
# Main scheduler
############################

current_expt=0

log_message "Starting experiment scheduler..."

############################
# Initial wave
############################

for gpu_id in "${GPU_NODES[@]}"; do

    if (( current_expt >= TOTAL_EXPERIMENTS )); then
        break
    fi

    if ! is_gpu_in_use "$gpu_id"; then

        port=$((BASE_PORT + current_expt))

        launch_experiment \
            "${EXPT_CONFIGS[$current_expt]}" \
            "$gpu_id" \
            "$port" \
            "$current_expt"

        ((current_expt+=1))

    else

        log_message "[WARNING] GPU ${gpu_id} is already in use. Skipping."

    fi

done

log_gpu_status

############################
# Monitor GPUs
############################

while (( current_expt < TOTAL_EXPERIMENTS )); do

    log_message "Waiting for a GPU to become available..."

    sleep 30

    for gpu_id in "${GPU_NODES[@]}"; do

        if (( current_expt >= TOTAL_EXPERIMENTS )); then
            break
        fi

        if ! is_gpu_in_use "$gpu_id"; then

            log_message "GPU ${gpu_id} is free."

            # Small delay to make sure GPU resources are released
            sleep 5

            port=$((BASE_PORT + current_expt))

            launch_experiment \
                "${EXPT_CONFIGS[$current_expt]}" \
                "$gpu_id" \
                "$port" \
                "$current_expt"

            ((current_expt+=1))

        fi

    done

    log_gpu_status

    log_message "Progress: ${current_expt}/${TOTAL_EXPERIMENTS} experiments launched."

done

############################
# Wait for all jobs
############################

log_message "All experiments have been launched."
log_message "Waiting for remaining experiments to finish..."

while true; do

    all_free=true

    for gpu_id in "${GPU_NODES[@]}"; do

        if is_gpu_in_use "$gpu_id"; then

            all_free=false
            break

        fi

    done

    if $all_free; then
        break
    fi

    log_gpu_status

    sleep 60

done

############################
# Finished
############################

log_message "=========================================="
log_message "All experiments finished."
log_message "=========================================="