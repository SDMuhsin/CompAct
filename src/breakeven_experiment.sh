#!/bin/bash
# Breakeven experiment: measure peak memory for 4 adapters × 5 seq_lengths × 2 modes (baseline vs FlashFFN)
# Uses both GPUs to run 2 experiments in parallel.
# Results logged to results/breakeven/

set -e
cd /workspace/CompAct
source env/bin/activate
export HF_HOME=./data TORCH_HOME=./data PYTHONPATH=src

RESULTS_DIR="./results/breakeven"
mkdir -p "$RESULTS_DIR"

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH=16
STEPS=20
TASK="sst2"
DTYPE="bfloat16"
SEQ_LENGTHS="64 128 256 512 1024"

# Common args
COMMON="--model_name_or_path $MODEL --task_name $TASK --per_device_train_batch_size $BATCH --total_batch_size $BATCH --max_train_steps $STEPS --dtype $DTYPE --pad_to_max_length --num_train_epochs 99 --lr 5e-4 --output_dir $RESULTS_DIR --mo5 false --seed 41"

run_experiment() {
    local GPU=$1
    local NAME=$2
    local EXTRA_ARGS=$3
    local SEQ=$4
    local FLASH=$5

    local SUFFIX=""
    local FLASH_ARGS=""
    if [ "$FLASH" = "flash" ]; then
        SUFFIX="_flash"
        FLASH_ARGS="--flash_ffn --flash_ffn_k_fraction 0.3"
    fi

    local RUN_NAME="${NAME}_seq${SEQ}${SUFFIX}"
    local LOG="$RESULTS_DIR/${RUN_NAME}.log"

    echo "[$(date +%H:%M:%S)] Starting $RUN_NAME on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python src/train_glue.py \
        $COMMON \
        --name "$RUN_NAME" \
        --max_length $SEQ \
        $EXTRA_ARGS \
        $FLASH_ARGS \
        > "$LOG" 2>&1

    # Extract peak memory from log
    local PEAK=$(grep -oP "Overall Peak GPU Memory.*?: \K[0-9.]+" "$LOG" 2>/dev/null || grep -oP "Peak Memory: \K[0-9.]+" "$LOG" 2>/dev/null || echo "N/A")
    echo "$RUN_NAME,$PEAK" >> "$RESULTS_DIR/summary.csv"
    echo "[$(date +%H:%M:%S)] Done $RUN_NAME → Peak: ${PEAK} MiB"
}

# Initialize summary CSV
echo "run_name,peak_memory_mib" > "$RESULTS_DIR/summary.csv"

# Adapter configurations
# LoRA (AdapterHub) - activations mode
LORA_ARGS="--optimizer adamw-lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.0"
# DoRA (PEFT) - recompute mode (effective weight)
DORA_ARGS="--optimizer adamw-dora --dora_r 16 --dora_alpha 32 --dora_dropout 0.0"
# AdaLoRA (PEFT) - recompute mode (effective weight)
ADALORA_ARGS="--optimizer adamw-adalora --adalora_init_r 16 --adalora_target_r 8 --adalora_alpha 16 --adalora_dropout 0.0 --adalora_tinit 5 --adalora_tfinal 5 --adalora_deltaT 2"
# DyLoRA (custom) - recompute mode (effective weight)
DYLORA_ARGS="--optimizer adamw-dylora --dylora_r 16 --dylora_alpha 32 --dylora_dropout 0.0"

ADAPTERS=("lora" "dora" "adalora" "dylora")
ADAPTER_ARGS=("$LORA_ARGS" "$DORA_ARGS" "$ADALORA_ARGS" "$DYLORA_ARGS")

echo "============================================"
echo "Breakeven Experiment"
echo "Adapters: ${ADAPTERS[*]}"
echo "Seq lengths: $SEQ_LENGTHS"
echo "Batch: $BATCH, Steps: $STEPS"
echo "Breakeven prediction (TinyLlama, k=0.3): N ≈ 2024"
echo "============================================"

# Run experiments: 2 at a time (one per GPU)
for i in "${!ADAPTERS[@]}"; do
    ADAPTER=${ADAPTERS[$i]}
    ARGS="${ADAPTER_ARGS[$i]}"

    for SEQ in $SEQ_LENGTHS; do
        # Run baseline on GPU 0, FlashFFN on GPU 1 in parallel
        run_experiment 0 "$ADAPTER" "$ARGS" "$SEQ" "baseline" &
        PID0=$!
        run_experiment 1 "$ADAPTER" "$ARGS" "$SEQ" "flash" &
        PID1=$!

        wait $PID0
        wait $PID1
    done
done

echo ""
echo "============================================"
echo "All experiments complete. Summary:"
echo "============================================"
cat "$RESULTS_DIR/summary.csv"
