#!/bin/bash
# Head-to-head: HyC-LoRA (MLSys 2025) vs FlashFFN v2/v3, under a MATCHED configuration.
#
# The config is dictated by HyC-LoRA's constraints, because they are the strictest:
#   * lora_alpha == r        (their fused kernels apply x@A@B with no alpha/r factor)
#   * lora_dropout == 0      (their fused path ignores dropout)
#   * LoRA on all 7 projections (q,k,v,o,gate,up,down)
#   * attn_implementation=eager (their fused attention needs the explicit 4D additive mask)
# FlashFFN supports all of these, so every arm below runs the SAME config and differs ONLY in
# the activation-compression method.
#
# NOTE ON ATTENTION: forcing eager on every arm is what makes the memory comparison meaningful.
# Under sdpa the attention matrix is never materialised, which would hand FlashFFN a large memory
# advantage unrelated to FFN compression. The last two arms measure that separately, because
# "composes with fused attention" is itself a real difference between the methods.

set -u
cd /workspace/CompAct
source env/bin/activate
export HF_HOME=$(pwd)/data TORCH_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1

CSV=results/hyclora/headtohead.csv
LOGDIR=results/hyclora/h2h_logs
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
  --results_csv "$CSV"
)

# name | extra flags
ARMS=(
  "h2h_baseline_eager|--attn_implementation eager"
  "h2h_hyclora_q4|--attn_implementation eager --hyclora --hyclora_base bf16 --hyclora_layer_type intra_inter --hyclora_q_bit 4 --hyclora_softmax_outlier_ratio 0.05 --hyclora_layernorm_outlier_ratio 0.005"
  "h2h_hyclora_q2|--attn_implementation eager --hyclora --hyclora_base bf16 --hyclora_layer_type intra_inter --hyclora_q_bit 2 --hyclora_softmax_outlier_ratio 0.05 --hyclora_layernorm_outlier_ratio 0.005"
  "h2h_v3_recompute|--attn_implementation eager --flash_ffn_v3 recompute"
  "h2h_v3_int4|--attn_implementation eager --flash_ffn_v3 int4"
  "h2h_v2_flash|--attn_implementation eager --flash_ffn --flash_ffn_k_fraction 0.3"
  "h2h_baseline_sdpa|--attn_implementation sdpa"
  "h2h_v3_recompute_sdpa|--attn_implementation sdpa --flash_ffn_v3 recompute"
)

run_one () {  # $1=gpu $2=spec
  local gpu=$1 spec=$2
  local name="${spec%%|*}" flags="${spec#*|}"
  echo "[$(date +%H:%M:%S)] START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu python -W ignore src/train_glue.py \
     "${common[@]}" --name "$name" $flags > "$LOGDIR/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name exit=$?"
}

# two-slot scheduler: alternate GPUs, two arms in flight
i=0
while [ $i -lt ${#ARMS[@]} ]; do
  run_one 0 "${ARMS[$i]}" &
  P0=$!
  if [ $((i+1)) -lt ${#ARMS[@]} ]; then
    run_one 1 "${ARMS[$((i+1))]}" &
    P1=$!
    wait $P0 $P1
  else
    wait $P0
  fi
  i=$((i+2))
done

echo "=== DONE -> $CSV ==="
