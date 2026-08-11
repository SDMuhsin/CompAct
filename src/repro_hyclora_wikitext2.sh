#!/bin/bash
# Gate 3: reproduce HyC-LoRA's published TinyLlama-1.1B / WikiText-2 numbers (their Table 1).
#
# Target (paper Table 1, TinyLlama-1.1B, "W" column = WikiText-2 perplexity, lower better):
#   Baseline (QLoRA, 16-bit activations)          8.24
#   HyC-LoRA @4-bit, intra+inter                  8.25
#   HyC-LoRA @2-bit, intra+inter                  8.32
#   (their spread across ALL HyC-LoRA settings is 8.25 - 8.39)
#
# Hyperparameters mirror upstream run_wikitext2.sh: rank 16, alpha 16 (r == alpha is REQUIRED by
# their fused kernels), lr 3e-4, block_size 1024, 3 epochs, per-device batch 2, effective batch
# 256, lora_dropout 0, NF4 base weights (their baseline IS QLoRA).
#
# The "Baseline" arm uses --hyclora_layer_type baseline: their own fused path with compression
# disabled. Gate 2 showed it matches stock HF to fwd cos 0.9999997, so this isolates the
# compression effect while holding the code path fixed.

set -u
cd /workspace/CompAct
source env/bin/activate
export HF_HOME=$(pwd)/data TORCH_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1

CSV=results/hyclora/repro_wikitext2.csv
LOGDIR=results/hyclora/repro_logs
mkdir -p "$LOGDIR"

common=(
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0
  --task_name wikitext2
  --optimizer adamw-lora
  --learning_rate 3e-4
  --lora_r 16 --lora_alpha 16 --lora_dropout 0.0
  --num_train_epochs 3
  --per_device_train_batch_size 2
  --total_batch_size 256
  --max_length 1024
  --dtype bfloat16
  --seeds 41
  --hyclora --hyclora_base nf4
  --hyclora_iteration_threshold 5
  --hyclora_layernorm_outlier_ratio 0.005
  --results_csv "$CSV"
)

run_arm () {   # $1=gpu  $2=arm-name  $3...=extra flags
  local gpu=$1; local name=$2; shift 2
  echo "[$(date +%H:%M:%S)] launching $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -W ignore src/train_glue.py \
    "${common[@]}" --name "$name" "$@" > "$LOGDIR/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] $name exit=$?"
}

# baseline: compression off (softmax ratio irrelevant, guard only applies to compressed types)
run_arm 0 hyclora_baseline \
    --hyclora_layer_type baseline --hyclora_q_bit 8 --hyclora_softmax_outlier_ratio 0.05 &
P0=$!

run_arm 1 hyclora_intra_inter_q4 \
    --hyclora_layer_type intra_inter --hyclora_q_bit 4 --hyclora_softmax_outlier_ratio 0.05 &
P1=$!

wait $P0 $P1

run_arm 0 hyclora_intra_inter_q2 \
    --hyclora_layer_type intra_inter --hyclora_q_bit 2 --hyclora_softmax_outlier_ratio 0.05

echo "=== DONE -> $CSV ==="
