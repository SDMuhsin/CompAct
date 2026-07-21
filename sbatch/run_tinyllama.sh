#!/bin/bash
# ============================================================================
# TinyLlama FlashFFN Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits Mo5 (median-of-five, seeds 41-45) benchmarks for PEFT methods with
# and without FlashFFN on TinyLlama-1.1B-Chat-v1.0 (SwiGLU, FlashFFN-compatible).
#
# 8 PEFT baselines + 6 FlashFFN (v2) variants + 6x|V3_MODES| FlashFFNV3 variants.
# With the active list below (fourierft/spectral commented out) and the default
# V3_MODES="recompute": 6 + 6 + 6 = 18 techniques x 6 active tasks.
# Job accounting: non-split tasks submit 1 job each; split-by-seed tasks (qnli
# active) submit 2 (array + agg) -> 18 x (5 + 2) = 126 submissions by default.
# (V3_MODES="recompute int4" adds the int4 speed arm -> 24 techniques -> 168.)
# FlashFFN (v2) only for methods with effective-weight paths:
#   Full FT, LoRA, DoRA, AdaLoRA, DyLoRA, VeRA
# FourierFT and Spectral fall back to gradient checkpointing (NOT FlashFFN).
# FlashFFNV3 arms (<tech>_flashffnv3_<mode>) cover the SAME 6 methods —
# train_glue.py raises ValueError for FourierFT/Spectral with --flash_ffn_v3
# (by design, NO v3 arms for those). v3 arms pass --flash_ffn_v3 <mode> and
# NEVER --flash_ffn/--flash_ffn_k_fraction (mutually exclusive). Set the
# V3_MODES env var to trim modes, or V3_MODES="" to disable all v3 arms.
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
    "lora"
    "dora"
    "adalora"
    "dylora"
    "vera"
    #"fourierft"
    #"spectral"
    # FlashFFN variants (only for methods with effective-weight paths)
    "base_flash"
    "lora_flash"
    "dora_flash"
    "adalora_flash"
    "dylora_flash"
    "vera_flash"
)

# --- FlashFFNV3 arms (env-overridable) ---
# V3_MODES: space-separated subset of {int4 int8 bf16 recompute}. The submitter
# can trim to one mode (V3_MODES="int4" ./sbatch/run_tinyllama.sh) or set
# V3_MODES="" to disable all FlashFFNV3 arms. NOTE: "-" (not ":-") expansion so
# an explicitly empty V3_MODES="" disables v3 instead of re-enabling the default.
# DEFAULT = recompute: exact gradients (cos>=0.9999), lightest memory, and the actual
# novel win (factored adapter math, NO merged-weight storage). int4 is an OPT-IN speed
# knob (skips the backward recompute GEMMs via generic activation quantization, ~tiny
# grad error) — it does NOT add memory savings over recompute. Add it via
# V3_MODES="recompute int4" if you want the speed arm too.
V3_MODES="${V3_MODES-recompute}"
# v3-capable techniques = the same 6 effective-weight methods as the v2 list
# (FourierFT/Spectral raise ValueError in train_glue.py with --flash_ffn_v3).
V3_TECHNIQUES=(base lora dora adalora dylora vera)

# --- V3_ONLY: submit ONLY the FlashFFNV3 arms (skip baselines + v2 _flash) ---
# Default 0 = full sweep (baselines + v2 _flash + v3, unchanged behaviour). Set
# V3_ONLY=1 when the baseline and v2 _flash rows are ALREADY in
# results/mo53_glue.csv from an earlier sweep and you only want to ADD the new
# v3 rows without re-spending compute re-running arms that already completed.
# Non-destructive: each v3 arm writes a NEW row keyed by its distinct run name
# (flash_ffn=v3:<mode>); existing baseline (flash_ffn=False) and v2
# (flash_ffn=True) rows are left untouched. Pair with V3_MODES to pick modes:
#   V3_ONLY=1 ./sbatch/run_tinyllama.sh                 # add recompute arms only
#   V3_ONLY=1 V3_MODES="recompute int4" ./sbatch/run_tinyllama.sh
V3_ONLY="${V3_ONLY-0}"
if [[ "$V3_ONLY" == "1" ]]; then
    techniques=()   # drop baselines + v2 _flash; the loop below appends only v3
fi

# --- RERUN_ONLY: submit ONLY the <technique>:<task> pairs listed in RERUN_PAIRS ---
# Default 0 = submit the full technique x task cross-product (unchanged). Set
# RERUN_ONLY=1 to resubmit a hand-picked set of arms after a partial sweep,
# without re-spending compute on the arms that already landed in the CSV.
# The filter is applied on top of V3_ONLY / V3_MODES, so an entry whose technique
# is not generated by the current V3_MODES is simply never reached.
#
# RERUN_PAIRS below is the exact set of arms LOST in the 2026-07-19 V3_ONLY sweep
# (18 of 72). All three causes are fixed in this commit:
#   * 14 x SLURM time-limit kill  -> root cause was a ~44 min/seed stall inside
#     evaluate.load() probing an unreachable Hub. Fixed by HF_EVALUATE_OFFLINE=1
#     in the job env block + src/train_glue.py mirroring HF_HUB_OFFLINE onto it,
#     plus MO5_OVERHEAD_MIN so small tasks get a fixed-overhead allowance.
#   *  4 x AdaLoRA ValueError "schedule values don't allow for a budgeting phase"
#     (rte=390 steps, cb=40 steps vs tinit=tfinal=200) -> fixed by the
#     short-schedule rescale in src/train_glue.py.
# Verify/refresh this list against results/mo53_glue.csv before resubmitting.
RERUN_ONLY="${RERUN_ONLY-0}"
RERUN_PAIRS=(
    # --- CB: all 12 v3 arms lost (10 time-limit, 2 AdaLoRA crash) ---
    "base_flashffnv3_recompute:cb"     "base_flashffnv3_int4:cb"
    "lora_flashffnv3_recompute:cb"     "lora_flashffnv3_int4:cb"
    "dora_flashffnv3_recompute:cb"     "dora_flashffnv3_int4:cb"
    "adalora_flashffnv3_recompute:cb"  "adalora_flashffnv3_int4:cb"
    "dylora_flashffnv3_recompute:cb"   "dylora_flashffnv3_int4:cb"
    "vera_flashffnv3_recompute:cb"     "vera_flashffnv3_int4:cb"
    # --- RTE: Full FT (time-limit, died at seed 43/5) + AdaLoRA (crash) ---
    "base_flashffnv3_recompute:rte"    "base_flashffnv3_int4:rte"
    "adalora_flashffnv3_recompute:rte" "adalora_flashffnv3_int4:rte"
    # --- MRPC: Full FT (time-limit, died at seed 44/5) ---
    "base_flashffnv3_recompute:mrpc"   "base_flashffnv3_int4:mrpc"
)

should_submit_pair() {
    # $1=technique $2=task. Always true unless RERUN_ONLY=1.
    [[ "$RERUN_ONLY" != "1" ]] && return 0
    local pair="$1:$2"
    for p in "${RERUN_PAIRS[@]}"; do
        [[ "$p" == "$pair" ]] && return 0
    done
    return 1
}

for _v3_mode in $V3_MODES; do
    case $_v3_mode in
        int4|int8|bf16|recompute) ;;
        *) echo "ERROR: invalid V3_MODES entry '$_v3_mode' (allowed: int4 int8 bf16 recompute)"; exit 1 ;;
    esac
    for _v3_tech in "${V3_TECHNIQUES[@]}"; do
        techniques+=("${_v3_tech}_flashffnv3_${_v3_mode}")
    done
done

# Tasks to evaluate
tasks=(
    #"cola"
    "mrpc"
    #"sst2"
    "rte"
    "qnli"
    "stsb"
    "boolq"
    "cb"
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

# --- Per-seed splitting (large tasks only) ---
# Tasks listed here have their Mo5 run as 5 independent per-seed jobs (a SLURM
# array 0-4) plus one tiny CPU aggregation job, instead of a single multi-day
# job. This keeps every job far under the wall-time cap, and a failed/timed-out
# seed costs one seed instead of all five. Reporting is unchanged: the
# aggregation job writes the same single Mo5 row to the CSV. Submit the script
# exactly as before — the split is fully transparent.
SPLIT_BY_SEED_TASKS="sst2 qnli"
SEEDS_LIST=(41 42 43 44 45)
SEED_OVERHEAD_MIN=30   # fixed per-seed overhead (model load, tokenize, eval) added on top of time/5

# Fixed overhead for a NON-split Mo5 job, in minutes. get_time_limit() below
# scales only with optimizer-step count, so before this term a small task got a
# budget with ZERO room for the 5x (model load + tokenise + metric load + eval)
# that every Mo5 job pays regardless of dataset size. That is why CB — 40 steps,
# ~2 min of actual compute — died on a 45 min wall in the 2026-07-19 sweep.
# 12 min/seed x 5 seeds, sized to absorb one recurrence of the evaluate.load()
# Hub-probe stall on the smallest tasks even if HF_EVALUATE_OFFLINE were to fail.
MO5_OVERHEAD_MIN=60

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

should_split_by_seed() {
    # Large tasks → split Mo5 into a per-seed array job + aggregation (see SPLIT_BY_SEED_TASKS).
    [[ " $SPLIT_BY_SEED_TASKS " == *" $1 "* ]]
}

get_job_resources() {
    # Sets: gpu_type, gpu_mem
    # TinyLlama (1.1B) fits comfortably in a 4g.40gb MIG slice.
    local technique=$1
    local base_tech=$(get_base_technique "$technique")

    gpu_type="h100_3g.40gb:1"
    gpu_mem="40000M"
}

is_flash_technique() {
    [[ "$1" == *"_flash" ]]
}

is_v3_technique() {
    # FlashFFNV3 arm: <technique>_flashffnv3_<cache_mode>
    [[ "$1" == *"_flashffnv3_"* ]]
}

get_v3_mode() {
    # lora_flashffnv3_int4 -> int4
    echo "${1##*_flashffnv3_}"
}

get_base_technique() {
    # Strip _flash / _flashffnv3_<mode> suffix
    local t="${1%%_flashffnv3_*}"
    echo "${t%_flash}"
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
    # Mo5 (5 seeds) on MIG h100_3g.40gb slice.
    # Calibrated from CoLA Full FT empirical run: ~360 min for 2010 optimizer steps
    # → ~10.8s/step. Scaled proportionally by dataset size and epochs.
    # Dataset sizes: CB=250, RTE=2490, MRPC=3668, STSB=5749, CoLA=8551,
    #   BoolQ=9427, SST2=67349, QNLI=104743, WT2≈36k chunks (seq=512,batch=8)
    local technique=$1
    local task=$2
    local per_seed=${3:-0}   # 1 = return per-seed time for a split (array) job
    local minutes=0

    local base_tech=$(get_base_technique "$technique")

    if is_wikitext_task "$task"; then
        # WikiText-2: 1 epoch, 5 seeds, seq=512, batch=8
        if [[ "$base_tech" == "base" ]]; then
            minutes=480
        else
            minutes=240
        fi
    elif [[ "$base_tech" == "base" ]]; then
        # Full FT: 3 epochs, 5 seeds, ~10.8s/optimizer_step
        case $task in
            cb)                minutes=30    ;;
            rte)               minutes=120   ;;
            mrpc)              minutes=180   ;;
            stsb)              minutes=300   ;;
            cola)              minutes=360   ;;
            boolq)             minutes=480   ;;
            sst2)              minutes=2880  ;;
            qnli)              minutes=4320  ;;
        esac
    else
        # PEFT: 10 epochs, 5 seeds, ~7s/optimizer_step (less backward compute)
        case $task in
            cb)            minutes=30    ;;
            rte)           minutes=240   ;;
            mrpc)          minutes=360   ;;
            stsb)          minutes=540   ;;
            cola)          minutes=720   ;;
            boolq)         minutes=840   ;;
            sst2)          minutes=5760  ;;
            qnli)          minutes=8640  ;;
        esac
    fi

    # Per-seed split: divide the full Mo5 allocation across the 5 seeds and add
    # fixed per-job overhead (model load, tokenize, eval). Applied before the
    # FlashFFN multiplier so the buffer compounds correctly.
    # Non-split jobs pay that same fixed overhead FIVE times (once per seed) and
    # previously got no allowance for it at all — see MO5_OVERHEAD_MIN.
    if [[ "$per_seed" == "1" ]]; then
        minutes=$(( (minutes + 4) / 5 + SEED_OVERHEAD_MIN ))
    else
        minutes=$(( minutes + MO5_OVERHEAD_MIN ))
    fi

    # FlashFFN v2 recompute mode adds ~50% overhead for top-K.
    # FlashFFNV3: measured s/step <= the technique's baseline AND <= its v2
    # _flash arm in all our data (results/v3_adapters/, results/v3_validation/),
    # but stay conservative: v3 limit = max(baseline limit, _flash limit). That
    # equals the same 1.5x bump as _flash for PEFT methods, and the plain
    # baseline limit for base (whose _flash arm gets no bump either).
    if { is_flash_technique "$technique" || is_v3_technique "$technique"; } && [[ "$base_tech" != "base" ]]; then
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
    if is_v3_technique "$technique"; then
        # FlashFFNV3: pass ONLY --flash_ffn_v3 <mode>; never --flash_ffn /
        # --flash_ffn_k_fraction (mutually exclusive in train_glue.py).
        flash=" --flash_ffn_v3 $(get_v3_mode "$technique")"
    elif is_flash_technique "$technique"; then
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
    if is_v3_technique "$tech"; then
        flash_suffix=" + FlashFFNV3($(get_v3_mode "$tech"))"
    elif is_flash_technique "$tech"; then
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
echo "V3 modes:   ${V3_MODES:-<disabled>} (FlashFFNV3 arms: <tech>_flashffnv3_<mode>)"
echo "V3 only:    ${V3_ONLY} ($([[ "$V3_ONLY" == "1" ]] && echo "v3 arms ONLY — baselines + v2 _flash skipped" || echo "full sweep: baselines + v2 _flash + v3"))"
echo "Rerun only: ${RERUN_ONLY} ($([[ "$RERUN_ONLY" == "1" ]] && echo "restricted to the ${#RERUN_PAIRS[@]} pairs in RERUN_PAIRS (2026-07-19 lost arms)" || echo "full technique x task cross-product"))"
echo "============================================"
echo ""

account_line=""
if [[ -n "$ACCOUNT" ]]; then
    account_line="#SBATCH --account=$ACCOUNT"
fi

for technique in "${techniques[@]}"; do
    technique_desc=$(get_technique_desc "$technique")

    for task in "${tasks[@]}"; do
        should_submit_pair "$technique" "$task" || continue
        epochs=$(get_epochs "$technique" "$task")
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

        if should_split_by_seed "$task"; then
            # ---- Per-seed array (0-4) + dependent CPU aggregation job ----
            # Each array task trains ONE seed and writes a JSON partial; the
            # aggregation job (afterany) recombines them into the same Mo5 row.
            partial_dir="./results/_partial/${run_name}"
            seed_time=$(get_time_limit "$technique" "$task" 1)
            rm -rf "$partial_dir"   # start clean so a resubmit never mixes stale seeds

            array_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_seed
#SBATCH --output=./logs/${job_name}_seed_%A_%a.out
#SBATCH --error=./logs/${job_name}_seed_%A_%a.err
#SBATCH --time=$seed_time
#SBATCH --gpus=$gpu_type
#SBATCH --mem=$gpu_mem
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1   # the evaluate lib ignores HF_HUB_OFFLINE; without this, evaluate.load probes the Hub and stalls ~44 min PER SEED
mkdir -p \$HF_HOME
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src

SEEDS_ARR=(${SEEDS_LIST[*]})
SEED=\${SEEDS_ARR[\$SLURM_ARRAY_TASK_ID]}
echo '========================================'
echo "Job: $job_name (seed \$SEED, array idx \$SLURM_ARRAY_TASK_ID)"
echo "Config: $technique_desc | Task: $task | Epochs: $epochs"
echo "Time limit: $seed_time | Started: \$(date)"
echo '========================================'
nvidia-smi
$python_cmd --seeds \$SEED --partial_dir $partial_dir
echo '========================================'
echo "Finished seed \$SEED: \$(date)"
echo '========================================'
EOF
)

            agg_id=$(sbatch --parsable --dependency=afterany:${array_id} <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_agg
#SBATCH --output=./logs/${job_name}_agg_%j.out
#SBATCH --error=./logs/${job_name}_agg_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1   # the evaluate lib ignores HF_HUB_OFFLINE; without this, evaluate.load probes the Hub and stalls ~44 min PER SEED
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src

echo '========================================'
echo "Aggregating Mo5: $run_name  (depends on array $array_id)"
echo "Started: \$(date)"
echo '========================================'
$python_cmd --aggregate --partial_dir $partial_dir
echo '========================================'
echo "Finished aggregation: \$(date)"
echo '========================================'
EOF
)
            echo "  [array ${array_id} (0-4, ${seed_time}/seed) -> agg ${agg_id}] $job_name  ($technique_desc, ${task}, ${epochs}ep, SPLIT-BY-SEED)"
            job_count=$((job_count + 2))
        else
            # ---- Single job: all 5 seeds in one process (original behaviour) ----
            time_limit=$(get_time_limit "$technique" "$task")
            sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH --gpus=$gpu_type
#SBATCH --mem=$gpu_mem
#SBATCH --cpus-per-task=4
$account_line

module load gcc arrow scipy-stack cuda cudnn
source ./env/bin/activate

export HF_HOME=\$(pwd)/data
export TORCH_HOME=\$(pwd)/data
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1   # the evaluate lib ignores HF_HUB_OFFLINE; without this, evaluate.load probes the Hub and stalls ~44 min PER SEED
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
        fi
    done
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          ./results/mo53_glue.csv"
echo "Logs directory:       ./logs/"
echo "============================================"
