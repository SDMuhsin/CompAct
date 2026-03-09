#!/bin/bash
# ============================================================================
# TinyLlama FlashFFN Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits Mo5 (median-of-five, seeds 41-45) benchmarks for PEFT methods with
# and without FlashFFN on TinyLlama-1.1B-Chat-v1.0 (SwiGLU, FlashFFN-compatible).
#
# 8 PEFT baselines + 6 FlashFFN variants = 14 techniques
# FlashFFN only for methods with effective-weight paths:
#   Full FT, LoRA, DoRA, AdaLoRA, DyLoRA, VeRA
# FourierFT and Spectral fall back to gradient checkpointing (NOT FlashFFN).
#
# Usage:
#   ./sbatch/run_tinyllama.sh
#   ./sbatch/run_tinyllama.sh --account def-myprof
#   ./sbatch/run_tinyllama.sh --local    # Run locally (no SLURM)
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT="def-seokbum"
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--local]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_SHORT="tinyllama"
DTYPE="bfloat16"
# NOTE: No --adapter_target_modules passed. train_glue.py defaults for LLaMA:
# PEFT methods: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj (all 7)
# This is required for FlashFFN effective-weight mode on MLP layers.

# Techniques to benchmark (comment/uncomment as needed)
techniques=(
    "base"
    #"lora"
    #"dora"
    #"adalora"
    #"dylora"
    #"vera"
    #"fourierft"
    #"spectral"
    # FlashFFN variants (only for methods with effective-weight paths)
    #"base_flash"
    #"lora_flash"
    #"dora_flash"
    #"adalora_flash"
    #"dylora_flash"
    #"vera_flash"
)

# Tasks to evaluate
tasks=(
    "cola"
    #"mrpc"
    #"sst2"
    #"rte"
    #"qnli"
    #"stsb"
    #"boolq"
    #"cb"
    #"wikitext2"
)

# ============================================================================
# HYPERPARAMETERS (validated on TinyLlama Mo5 benchmarks)
# ============================================================================

# --- Shared across all techniques ---
# batch=32, seq=128 → N=4096 (2× breakeven for FlashFFN recompute mode)
BATCH_SIZE=32
EVAL_BATCH_SIZE=32
TOTAL_BATCH_SIZE=64
WEIGHT_DECAY=0.01
LR_SCHEDULER="linear"
GRAD_CLIP=1.0
MAX_LENGTH=128
PAD_TO_MAX="--pad_to_max_length"

# --- WikiText-2 overrides ---
WT2_BATCH_SIZE=8
WT2_MAX_LENGTH=512

# --- Full fine-tuning ---
BASE_LR="2e-5"
BASE_EPOCHS=3

# --- PEFT epochs ---
PEFT_EPOCHS=10

# --- WikiText-2 epochs (all methods) ---
WT2_EPOCHS=1

# --- LoRA (PEFT library, r=8) ---
LORA_LR="5e-4"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.0

# --- DoRA (PEFT library, r=8) ---
DORA_LR="5e-4"
DORA_R=8
DORA_ALPHA=16
DORA_DROPOUT=0.05

# --- AdaLoRA (PEFT library) ---
ADALORA_LR="5e-4"
ADALORA_INIT_R=8
ADALORA_TARGET_R=4
ADALORA_ALPHA=16

# --- DyLoRA (custom) ---
DYLORA_LR="5e-4"
DYLORA_R=8
DYLORA_ALPHA=16

# --- VeRA (PEFT library, r=256) ---
VERA_LR="1e-3"
VERA_R=256
VERA_D_INITIAL=0.1
VERA_DROPOUT=0.0

# --- FourierFT (PEFT library) ---
# NOTE: Not yet tuned for TinyLlama. Using BERT defaults as starting point.
FOURIERFT_LR="5e-2"
FOURIERFT_N=256
FOURIERFT_SCALING=150.0

# --- Spectral Adapter ---
# NOTE: Not yet tuned for TinyLlama. Using BERT defaults as starting point.
# Per-task config selection follows BERT reference.
SPECTRAL_LR="2e-2"
SPECTRAL_SCALING=1.0
SPECTRAL_DROPOUT=0.0
SPECTRAL_FREQ_MODE="contiguous"
# Mode A — Dense defaults (boolq, stsb, mrpc)
SPECTRAL_DENSE_P=16
SPECTRAL_DENSE_Q=16
SPECTRAL_DENSE_D_INITIAL=0.01
# Mode B — RTE special: dense p=16 but d=0.07 and scaling=2.0
SPECTRAL_RTE_D_INITIAL=0.07
SPECTRAL_RTE_SCALING=2.0
# Mode C/D — Factored defaults (cola, cb, sst2, qnli)
SPECTRAL_FACTORED_P=32
SPECTRAL_FACTORED_Q=32
SPECTRAL_FACTORED_D_INITIAL=0.07
SPECTRAL_FACTORED_RANK=4
# Per-task mode selection
SPECTRAL_FACTORED_LEARN_TASKS="cola cb"    # Mode C: factored + learn_scaling
SPECTRAL_FACTORED_TASKS="sst2 qnli"       # Mode D: factored, no learn_scaling

# --- FlashFFN ---
FLASH_FFN_K_FRACTION=0.3

# ============================================================================
# END CONFIGURATION
# ============================================================================

job_count=0
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

is_wikitext_task() {
    case $1 in
        wikitext2|wikitext103) return 0 ;;
        *) return 1 ;;
    esac
}

get_job_resources() {
    # Sets: gpu_type, gpu_mem
    # TinyLlama (1.1B) fits comfortably in a 4g.40gb MIG slice.
    local technique=$1
    local base_tech=$(get_base_technique "$technique")

    gpu_type="nvidia_h100_80gb_hbm3_4g.40gb:1"
    gpu_mem="40000M"
}

is_flash_technique() {
    [[ "$1" == *"_flash" ]]
}

get_base_technique() {
    # Strip _flash suffix
    echo "${1%_flash}"
}

get_epochs() {
    local technique=$1
    local task=$2
    if is_wikitext_task "$task"; then
        echo "$WT2_EPOCHS"
    elif [[ "$(get_base_technique "$technique")" == "base" ]]; then
        echo "$BASE_EPOCHS"
    else
        echo "$PEFT_EPOCHS"
    fi
}

get_time_limit() {
    # Returns SLURM time string.
    # Mo5 (5 seeds): estimates based on TinyLlama (1.1B) throughput.
    local technique=$1
    local task=$2
    local minutes=0

    local base_tech=$(get_base_technique "$technique")

    if is_wikitext_task "$task"; then
        # WikiText-2: 1 epoch, 5 seeds, seq=512
        if [[ "$base_tech" == "base" ]]; then
            minutes=120     # Full FT, larger memory
        else
            minutes=60      # PEFT, frozen backbone
        fi
    elif [[ "$base_tech" == "base" ]]; then
        # Full FT: 3 epochs, 5 seeds
        case $task in
            mrpc|rte|stsb|cb)  minutes=60   ;;
            cola)              minutes=90   ;;
            boolq)             minutes=180  ;;
            sst2)              minutes=360  ;;
            qnli)              minutes=540  ;;
        esac
    else
        # PEFT: 10 epochs, 5 seeds
        case $task in
            rte|cb)        minutes=120  ;;
            mrpc|stsb)     minutes=150  ;;
            cola)          minutes=240  ;;
            boolq)         minutes=600  ;;
            sst2)          minutes=900  ;;
            qnli)          minutes=1200 ;;
        esac
    fi

    # FlashFFN recompute mode adds ~50% overhead for top-K
    if is_flash_technique "$technique" && [[ "$base_tech" != "base" ]]; then
        minutes=$((minutes * 3 / 2))
    fi

    # Format as D-HH:MM:SS or H:MM:SS
    local hours=$((minutes / 60))
    local mins=$((minutes % 60))
    if [[ $hours -ge 24 ]]; then
        local days=$((hours / 24))
        hours=$((hours % 24))
        printf "%d-%02d:%02d:00" "$days" "$hours" "$mins"
    else
        printf "%d:%02d:00" "$hours" "$mins"
    fi
}

build_python_cmd() {
    local technique=$1
    local task=$2
    local epochs=$3
    local run_name=$4

    local base_tech=$(get_base_technique "$technique")
    local flash=""
    if is_flash_technique "$technique"; then
        flash=" --flash_ffn --flash_ffn_k_fraction $FLASH_FFN_K_FRACTION"
    fi

    # Base command
    local common="python src/train_glue.py"
    common+=" --model_name_or_path $MODEL"
    common+=" --task_name $task"
    common+=" --num_train_epochs $epochs"
    common+=" --total_batch_size $TOTAL_BATCH_SIZE"
    common+=" --weight_decay $WEIGHT_DECAY"
    common+=" --lr_scheduler_type $LR_SCHEDULER"
    common+=" --grad_clipping $GRAD_CLIP"
    common+=" --dtype $DTYPE"
    common+=" --name $run_name"

    # Task-specific overrides
    if is_wikitext_task "$task"; then
        common+=" --per_device_train_batch_size $WT2_BATCH_SIZE"
        common+=" --per_device_eval_batch_size $WT2_BATCH_SIZE"
        common+=" --max_length $WT2_MAX_LENGTH"
    else
        common+=" --per_device_train_batch_size $BATCH_SIZE"
        common+=" --per_device_eval_batch_size $EVAL_BATCH_SIZE"
        common+=" --max_length $MAX_LENGTH"
        common+=" $PAD_TO_MAX"
    fi

    case $base_tech in
        base)
            echo "$common --optimizer adamw --learning_rate $BASE_LR$flash"
            ;;
        lora)
            echo "$common --optimizer adamw-lora --learning_rate $LORA_LR --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT$flash"
            ;;
        dora)
            echo "$common --optimizer adamw-dora --learning_rate $DORA_LR --dora_r $DORA_R --dora_alpha $DORA_ALPHA --dora_dropout $DORA_DROPOUT$flash"
            ;;
        adalora)
            echo "$common --optimizer adamw-adalora --learning_rate $ADALORA_LR --adalora_init_r $ADALORA_INIT_R --adalora_target_r $ADALORA_TARGET_R --adalora_alpha $ADALORA_ALPHA$flash"
            ;;
        dylora)
            echo "$common --optimizer adamw-dylora --learning_rate $DYLORA_LR --dylora_r $DYLORA_R --dylora_alpha $DYLORA_ALPHA$flash"
            ;;
        vera)
            echo "$common --optimizer adamw-vera --learning_rate $VERA_LR --vera_r $VERA_R --vera_d_initial $VERA_D_INITIAL --vera_dropout $VERA_DROPOUT$flash"
            ;;
        fourierft)
            echo "$common --optimizer adamw-fourierft --learning_rate $FOURIERFT_LR --fourierft_n_frequency $FOURIERFT_N --fourierft_scaling $FOURIERFT_SCALING$flash"
            ;;
        spectral)
            local cmd="$common --optimizer adamw-spectral --learning_rate $SPECTRAL_LR --spectral_dropout $SPECTRAL_DROPOUT --spectral_freq_mode $SPECTRAL_FREQ_MODE"
            # Per-task config selection (4 modes, following BERT reference)
            if [[ " $SPECTRAL_FACTORED_LEARN_TASKS " == *" $task "* ]]; then
                # Mode C: Factored + learn_scaling (cola, cb)
                cmd+=" --spectral_p $SPECTRAL_FACTORED_P --spectral_q $SPECTRAL_FACTORED_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_FACTORED_D_INITIAL"
                cmd+=" --spectral_factored_rank $SPECTRAL_FACTORED_RANK"
                cmd+=" --spectral_scaling $SPECTRAL_SCALING"
                cmd+=" --spectral_learn_scaling"
            elif [[ " $SPECTRAL_FACTORED_TASKS " == *" $task "* ]]; then
                # Mode D: Factored, no learn_scaling (sst2, qnli)
                cmd+=" --spectral_p $SPECTRAL_FACTORED_P --spectral_q $SPECTRAL_FACTORED_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_FACTORED_D_INITIAL"
                cmd+=" --spectral_factored_rank $SPECTRAL_FACTORED_RANK"
                cmd+=" --spectral_scaling $SPECTRAL_SCALING"
            elif [[ "$task" == "rte" ]]; then
                # Mode B: Dense + scaling=2.0 (rte only)
                cmd+=" --spectral_p $SPECTRAL_DENSE_P --spectral_q $SPECTRAL_DENSE_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_RTE_D_INITIAL"
                cmd+=" --spectral_scaling $SPECTRAL_RTE_SCALING"
            else
                # Mode A: Dense default (boolq, stsb, mrpc, wikitext2, and any others)
                cmd+=" --spectral_p $SPECTRAL_DENSE_P --spectral_q $SPECTRAL_DENSE_Q"
                cmd+=" --spectral_d_initial $SPECTRAL_DENSE_D_INITIAL"
                cmd+=" --spectral_scaling $SPECTRAL_SCALING"
            fi
            echo "$cmd$flash"
            ;;
    esac
}

get_technique_desc() {
    local tech=$1
    local base_tech=$(get_base_technique "$tech")
    local flash_suffix=""
    if is_flash_technique "$tech"; then
        flash_suffix=" + FlashFFN(k=$FLASH_FFN_K_FRACTION)"
    fi
    case $base_tech in
        base)      echo "Full FT (lr=$BASE_LR)${flash_suffix}" ;;
        lora)      echo "LoRA (r=$LORA_R, a=$LORA_ALPHA, lr=$LORA_LR)${flash_suffix}" ;;
        dora)      echo "DoRA (r=$DORA_R, a=$DORA_ALPHA, lr=$DORA_LR)${flash_suffix}" ;;
        adalora)   echo "AdaLoRA (r=$ADALORA_INIT_R→$ADALORA_TARGET_R, a=$ADALORA_ALPHA, lr=$ADALORA_LR)${flash_suffix}" ;;
        dylora)    echo "DyLoRA (r=$DYLORA_R, a=$DYLORA_ALPHA, lr=$DYLORA_LR)${flash_suffix}" ;;
        vera)      echo "VeRA (r=$VERA_R, d=$VERA_D_INITIAL, lr=$VERA_LR)${flash_suffix}" ;;
        fourierft) echo "FourierFT (n=$FOURIERFT_N, s=$FOURIERFT_SCALING, lr=$FOURIERFT_LR)${flash_suffix}" ;;
        spectral)  echo "Spectral (per-task config, lr=$SPECTRAL_LR)${flash_suffix}" ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

echo "============================================"
echo "TinyLlama FlashFFN Benchmark Suite"
echo "============================================"
echo "Model:      $MODEL"
echo "Techniques: ${techniques[*]}"
echo "Tasks:      ${tasks[*]}"
echo "Dtype:      $DTYPE"
echo "Target:     architecture defaults (all 7 for LLaMA)"
echo "============================================"
echo ""

for technique in "${techniques[@]}"; do
    technique_desc=$(get_technique_desc "$technique")

    for task in "${tasks[@]}"; do
        epochs=$(get_epochs "$technique" "$task")
        time_limit=$(get_time_limit "$technique" "$task")
        job_name="${MODEL_SHORT}_${technique}_${task}"
        run_name="${technique}_${MODEL_SHORT}_${task}"

        get_job_resources "$technique"
        python_cmd=$(build_python_cmd "$technique" "$task" "$epochs" "$run_name")

        if [[ "$LOCAL_MODE" == true ]]; then
            echo "========================================"
            echo "Running locally: $job_name"
            echo "Config: $technique_desc"
            echo "Command: $python_cmd"
            echo "========================================"
            eval "$python_cmd"
            ((job_count++))
            continue
        fi

        account_line=""
        if [[ -n "$ACCOUNT" ]]; then
            account_line="#SBATCH --account=$ACCOUNT"
        fi

        sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH --gres=gpu:$gpu_type
#SBATCH --mem=$gpu_mem
#SBATCH --cpus-per-task=4
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export HF_DATASETS_CACHE=\$(pwd)/data
export TRANSFORMERS_CACHE=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
mkdir -p \$HF_HOME

echo '========================================'
echo "Job: $job_name"
echo "Model: $MODEL"
echo "Technique: $technique"
echo "Config: $technique_desc"
echo "Task: $task"
echo "Epochs: $epochs"
echo "Dtype: $DTYPE"
echo "Time limit: $time_limit"
echo "Cache: \$HF_HOME"
echo "Started: \$(date)"
echo '========================================'
nvidia-smi
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src
$python_cmd
echo '========================================'
echo "Finished: \$(date)"
echo '========================================'
EOF
)
        echo "  [$sbatch_id] $job_name  ($technique_desc, ${task}, ${epochs}ep, ${time_limit})"
        ((job_count++))
    done
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          ./results/mo53_glue.csv"
echo "Logs directory:       ./logs/"
echo "============================================"
