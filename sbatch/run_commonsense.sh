#!/bin/bash
# ============================================================================
# Commonsense Multiple-Choice Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Trains ONCE on the LLM-Adapters Commonsense-170K mixture and, in the SAME job,
# evaluates on all 8 native commonsense sets (BoolQ, PIQA, SIQA, HellaSwag,
# WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA), writing ONE Mo5 CSV row per
# eval set (task_name="commonsense_<set>"). See CONTEXT.md section 26 and
# src/commonsense_mc.py.
#
# This is a SEPARATE script from run_tinyllama.sh on purpose: commonsense is a
# THIRD paradigm (P3 multiple-choice) with a single 170K "task" that fans out to
# 8 result rows, a much larger train set (~170K rows vs <105K for the GLUE
# sweep's biggest task), and its own time budget. Mixing it into run_tinyllama.sh
# (which loops technique x task over GLUE/SuperGLUE) would not fit cleanly.
#
# Model: TinyLlama-1.1B-Chat-v1.0 (SwiGLU, FlashFFN-compatible).
#
# 6 baselines + 6 FlashFFN (v2) variants + 6x|V3_MODES| FlashFFNV3 variants
# = 12 + 6 = 18 techniques with the default V3_MODES="recompute", each a
# Mo5 (seeds 41-45) -> 18 x {array 0-4 + CPU agg} = 36 SLURM submissions.
# (V3_MODES="recompute int4" -> 24 techniques -> 48 submissions.)
# FlashFFN only for methods with effective-weight paths:
#   Full FT, LoRA, DoRA, AdaLoRA, DyLoRA, VeRA
# (FourierFT/Spectral have NO FlashFFN effective-weight path -> excluded; the
#  same 6-method restriction holds for FlashFFNV3, which raises ValueError in
#  train_glue.py for any other PEFT wrapper on the MLP projections.)
# FlashFFNV3 arms (<tech>_flashffnv3_<mode>) pass --flash_ffn_v3 <mode> and
# NEVER --flash_ffn/--flash_ffn_k_fraction (mutually exclusive). Set the
# V3_MODES env var to trim modes, or V3_MODES="" to disable all v3 arms.
#
# Per-seed splitting is ALWAYS on here (commonsense is the heaviest task in the
# project): every technique submits a SLURM array (0-4, one seed each) plus a
# tiny CPU aggregation job, so no single job runs multi-day and a failed/timed-out
# seed costs one seed, not five. Reporting is unchanged: the aggregation job emits
# the same 8 Mo5 rows (one per eval set). See run_tinyllama.sh section 23 in
# CONTEXT.md for the per-seed-split machinery.
#
# Usage:
#   ./sbatch/run_commonsense.sh
#   ./sbatch/run_commonsense.sh --account def-myprof
#   ./sbatch/run_commonsense.sh --local    # Run locally (no SLURM)
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
TASK="commonsense"          # single task; trains on 170K, evals 8 sets -> 8 rows
# NOTE: No --adapter_target_modules passed. train_glue.py defaults for LLaMA wrap
# all 7 linear layers (q/k/v/o_proj + gate/up/down_proj). The MLP projections MUST
# be wrapped for FlashFFN effective-weight (recompute) mode to engage.

# Techniques to benchmark (comment/uncomment as needed)
techniques=(
    "base"
    "lora"
    "dora"
    "adalora"
    "dylora"
    "vera"
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
# can trim to one mode (V3_MODES="int4" ./sbatch/run_commonsense.sh) or set
# V3_MODES="" to disable all FlashFFNV3 arms. NOTE: "-" (not ":-") expansion so
# an explicitly empty V3_MODES="" disables v3 instead of re-enabling the default.
# DEFAULT = recompute: exact gradients (cos>=0.9999), lightest memory, and the actual
# novel win (factored adapter math, NO merged-weight storage). int4 is an OPT-IN speed
# knob (skips the backward recompute GEMMs via generic activation quantization, ~tiny
# grad error) — it does NOT add memory savings over recompute. Add it via
# V3_MODES="recompute int4" if you want the speed arm too.
V3_MODES="${V3_MODES-recompute}"
# v3-capable techniques = the same 6 effective-weight methods as the v2 list.
V3_TECHNIQUES=(base lora dora adalora dylora vera)

# --- V3_ONLY: submit ONLY the FlashFFNV3 arms (skip baselines + v2 _flash) ---
# Default 0 = full sweep (baselines + v2 _flash + v3, unchanged behaviour). Set
# V3_ONLY=1 when the baseline and v2 _flash rows are ALREADY in
# results/mo53_glue.csv from an earlier sweep and you only want to ADD the new
# v3 rows without re-spending compute re-running arms that already completed.
# Non-destructive: each v3 arm writes a NEW row keyed by its distinct run name
# (flash_ffn=v3:<mode>); existing baseline (flash_ffn=False) and v2
# (flash_ffn=True) rows are left untouched. Pair with V3_MODES to pick modes:
#   V3_ONLY=1 ./sbatch/run_commonsense.sh               # add recompute arms only
#   V3_ONLY=1 V3_MODES="recompute int4" ./sbatch/run_commonsense.sh
V3_ONLY="${V3_ONLY-0}"
if [[ "$V3_ONLY" == "1" ]]; then
    techniques=()   # drop baselines + v2 _flash; the loop below appends only v3
fi

for _v3_mode in $V3_MODES; do
    case $_v3_mode in
        int4|int8|bf16|recompute) ;;
        *) echo "ERROR: invalid V3_MODES entry '$_v3_mode' (allowed: int4 int8 bf16 recompute)"; exit 1 ;;
    esac
    for _v3_tech in "${V3_TECHNIQUES[@]}"; do
        techniques+=("${_v3_tech}_flashffnv3_${_v3_mode}")
    done
done

# ============================================================================
# HYPERPARAMETERS (reuse the validated TinyLlama Mo5 configs; CONTEXT.md section 26
# explicitly DEFERS per-LLM-Adapters-paper HP tuning, so we reuse the LoRA-family
# configs already validated in results/mo53_glue.csv.)
# ============================================================================

# --- Shared across all techniques ---
# batch=8, seq=256 -> per micro-batch N = 8 x ~2.76 choices x 256. MC multiplies the
# token count by #choices, so batch is kept at 8 (not 32 like the GLUE sweep) to stay
# safely within a 40 GB slice. total_batch=64 -> grad_accum=8.
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
TOTAL_BATCH_SIZE=64
MAX_LENGTH=256
WEIGHT_DECAY=0.01
LR_SCHEDULER="linear"
GRAD_CLIP=1.0

# --- Epochs ---
# ALL techniques use 3 epochs on commonsense (NOT the GLUE sweep's 3/10 base/PEFT
# split). The 170K mixture is ~20x the largest GLUE task; 3 epochs is the standard
# LLM-Adapters commonsense training budget. 10 PEFT epochs over 170K x 5 seeds would
# be infeasible.
CS_EPOCHS=3

# --- Full fine-tuning ---
BASE_LR="2e-5"

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
# tinit/tfinal use the PEFT defaults (200/200); with ~2663 steps/epoch x 3 epochs
# = 7989 steps there is ample room for the SVD warmup + pruning schedule.
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

# --- FlashFFN ---
FLASH_FFN_K_FRACTION=0.3

# ============================================================================
# TIME BUDGET CALIBRATION
# ============================================================================
# Per-seed wall-time is calibrated from EMPIRICAL dev-box measurements (NVIDIA A40,
# TinyLlama, commonsense MC, batch=8, seq=256, total_batch=64). Each array task runs
# ONE seed = a full 3-epoch pass over the 170K train set + per-epoch eval on all 8
# sets. The measured steady-state seconds/optimizer-step (S_STEP below) drives the
# estimate; FlashFFN recompute-mode top-K overhead is measured directly per variant.
#
#   steps/epoch   = ceil(170420 / 64)            = 2663
#   total steps   = 2663 * 3 epochs              = 7989   (per seed)
#   train seconds = 7989 * S_STEP[tech]
#   per-seed min  = ceil(train_sec/60 * SAFETY) + EVAL_MIN*epochs + OVERHEAD_MIN
#
# SAFETY (1.5x) covers: dev-box A40 -> HPC h100_3g.40gb MIG-slice hardware delta,
# run-to-run variance, and scheduler jitter. Over-allocating SLURM time is harmless
# (only affects queue priority); under-allocating kills a multi-hour job, so we round
# UP everywhere. EVAL_MIN and OVERHEAD_MIN are deliberately generous (full eval on the
# 8 sets measured ~4-5 min/epoch; one-time model load + 170K parse/tokenise ~2-3 min).
TRAIN_SIZE=170420
STEPS_PER_EPOCH=$(( (TRAIN_SIZE + TOTAL_BATCH_SIZE - 1) / TOTAL_BATCH_SIZE ))   # 2663
TOTAL_STEPS=$(( STEPS_PER_EPOCH * CS_EPOCHS ))                                  # 7989
TIME_SAFETY="1.5"        # training-time multiplier (A40 -> MIG + variance)
EVAL_MIN_PER_EPOCH=6     # generous; full 8-set eval measured ~4-5 min/epoch
OVERHEAD_MIN=12          # model load + load/parse/tokenise 170K + 8 eval sets

# Measured steady-state seconds/optimizer-step on the A40 dev box (NVIDIA A40,
# TinyLlama, commonsense MC, batch=8/seq=256/total_batch=64). See tracker
# llmdocs/trackers/commonsense_mc.md for the full calibration table + peak memory.
# base/lora/dora/dora_flash/base_flash/lora_flash/adalora/dylora/vera measured
# DIRECTLY; the three remaining flash variants (adalora/dylora/vera) use baseline x1.5
# (a deliberately conservative flash-overhead factor -- over-allocating SLURM time is
# harmless, under-allocating kills a multi-day job).
declare -A S_STEP=(
    [base]=3.14          [base_flash]=3.62
    [lora]=4.20          [lora_flash]=4.48
    [dora]=13.00         [dora_flash]=6.97
    [adalora]=7.60       [adalora_flash]=11.40
    [dylora]=3.28        [dylora_flash]=4.92
    [vera]=5.55          [vera_flash]=8.33
)

# FlashFFNV3 time rule (conservative): every v3 arm is allocated
#   s_step = max( S_STEP[<base>], S_STEP[<base>_flash] )
# i.e. v3 time limit = max(that technique's baseline limit, its _flash limit).
# Measured v3 s/step is <= BOTH reference arms in all our data — per-adapter
# 30-step smokes (results/v3_adapters/smoke_v3_adapters.json, median step-s,
# v2 -> int4 / recompute): adalora 1.146 -> 0.983/1.098, dylora 0.865 ->
# 0.696/0.824, vera 1.273 -> 0.927/1.062, dora 1.642 -> 1.661/1.625 (int4
# ~parity, +1% within noise) — and v3 beat the BASELINE wall-time on CoLA/WT2
# (results/v3_validation/summary.md). Over-allocating SLURM time is harmless;
# under-allocating kills a multi-hour seed. Effective v3 s/step used:
#   base 3.62  lora 4.48  dora 13.00  adalora 11.40  dylora 4.92  vera 8.33

# --- Per-seed splitting (ALWAYS on for commonsense) ---
SEEDS_LIST=(41 42 43 44 45)

# ============================================================================
# END CONFIGURATION
# ============================================================================

job_count=0
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

get_job_resources() {
    # Sets: gpu_type (the --gpus request, which determines GPU memory) and gpu_mem
    # (the --mem host-RAM request).
    #
    # Measured A40 overall-peak GPU memory at batch=8/seq=256 MC (GB; see tracker):
    #   base 24.6  lora 26.4  vera 27.7  dylora 17.3  adalora ~17  -> all fit 40 GB
    #   (the heaviest, vera 27.7, keeps >5 GB margin even with a +25% worst-case-batch
    #   allowance). DoRA baseline OOM'd at 43+ GB even on the 45 GB-free A40 -> exceeds
    #   a 40 GB slice.
    # DoRA's effective-weight baseline is the only high-water mark; route BOTH its
    # arms (baseline AND flash) to a full 80 GB H100 so the A/B throughput comparison
    # stays on identical hardware. Everything else uses the cost-efficient 40 GB MIG
    # slice (same as run_tinyllama.sh).
    #
    # FlashFFNV3 arms ALWAYS use the 40 GB MIG slice — the same resource class as
    # the technique's v2 _flash arm. v3 peak memory is strictly below v2-flash in
    # every per-adapter measurement (results/v3_adapters/smoke_v3_adapters.json,
    # 30-step smokes; peak MiB v2 -> v3-int4 / v3-recompute):
    #   dora    19,515 -> 15,142 / 14,584   (-22% / -25%)
    #   adalora 12,013 -> 10,523 /  9,986   (-12% / -17%)
    #   dylora   9,821 ->  8,338 /  7,791   (-15% / -21%)
    #   vera    11,568 ->  9,932 /  9,384   (-14% / -19%)
    # and v3 never materializes W_eff (the cause of the DoRA-baseline 43+ GB
    # high-water mark). Only the DoRA BASELINE (plus its v2 _flash A/B partner)
    # needs the 80 GB H100; dora_flashffnv3_* goes to the MIG slice.
    local technique=$1
    local base_tech=$(get_base_technique "$technique")

    if [[ "$base_tech" == "dora" ]] && ! is_v3_technique "$technique"; then
        gpu_type="h100:1"          # full H100, 80 GB
        gpu_mem="64000M"
    else
        gpu_type="h100_3g.40gb:1"  # MIG slice, 40 GB
        gpu_mem="40000M"
    fi
}

get_per_seed_time() {
    # Returns a SLURM time string D-HH:MM:SS for ONE seed (full 3-epoch run + eval).
    local technique=$1
    local s_step
    if is_v3_technique "$technique"; then
        # FlashFFNV3 rule: max(baseline, _flash) s/step — see comment at S_STEP.
        local v3_base=$(get_base_technique "$technique")
        local s_base=${S_STEP[$v3_base]}
        local s_flash=${S_STEP[${v3_base}_flash]}
        if [[ -z "$s_base" || -z "$s_flash" ]]; then
            echo "ERROR: no S_STEP calibration for v3 technique '$technique' (base '$v3_base')" >&2
            exit 1
        fi
        s_step=$(awk -v a="$s_base" -v b="$s_flash" 'BEGIN { print (a > b) ? a : b }')
    else
        s_step=${S_STEP[$technique]}
    fi
    if [[ -z "$s_step" || "$s_step" == __* ]]; then
        echo "ERROR: no S_STEP calibration for technique '$technique'" >&2
        exit 1
    fi
    # minutes = ceil(TOTAL_STEPS * s_step / 60 * SAFETY) + EVAL_MIN*epochs + OVERHEAD
    local minutes
    minutes=$(awk -v ts="$TOTAL_STEPS" -v s="$s_step" -v safety="$TIME_SAFETY" \
                  -v ev="$EVAL_MIN_PER_EPOCH" -v ep="$CS_EPOCHS" -v oh="$OVERHEAD_MIN" \
                  'BEGIN { m = (ts*s/60.0)*safety + ev*ep + oh; printf("%d", int(m)+1) }')
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
    local run_name=$2

    local base_tech=$(get_base_technique "$technique")
    local flash=""
    if is_v3_technique "$technique"; then
        # FlashFFNV3: pass ONLY --flash_ffn_v3 <mode>; never --flash_ffn /
        # --flash_ffn_k_fraction (mutually exclusive in train_glue.py).
        flash=" --flash_ffn_v3 $(get_v3_mode "$technique")"
    elif is_flash_technique "$technique"; then
        flash=" --flash_ffn --flash_ffn_k_fraction $FLASH_FFN_K_FRACTION"
    fi

    local common="python src/train_glue.py"
    common+=" --model_name_or_path $MODEL"
    common+=" --task_name $TASK"
    common+=" --num_train_epochs $CS_EPOCHS"
    common+=" --per_device_train_batch_size $BATCH_SIZE"
    common+=" --per_device_eval_batch_size $EVAL_BATCH_SIZE"
    common+=" --total_batch_size $TOTAL_BATCH_SIZE"
    common+=" --max_length $MAX_LENGTH"
    common+=" --weight_decay $WEIGHT_DECAY"
    common+=" --lr_scheduler_type $LR_SCHEDULER"
    common+=" --grad_clipping $GRAD_CLIP"
    common+=" --dtype $DTYPE"
    common+=" --name $run_name"

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
        base)    echo "Full FT (lr=$BASE_LR)${flash_suffix}" ;;
        lora)    echo "LoRA (r=$LORA_R, a=$LORA_ALPHA, lr=$LORA_LR)${flash_suffix}" ;;
        dora)    echo "DoRA (r=$DORA_R, a=$DORA_ALPHA, lr=$DORA_LR)${flash_suffix}" ;;
        adalora) echo "AdaLoRA (r=$ADALORA_INIT_R->$ADALORA_TARGET_R, a=$ADALORA_ALPHA, lr=$ADALORA_LR)${flash_suffix}" ;;
        dylora)  echo "DyLoRA (r=$DYLORA_R, a=$DYLORA_ALPHA, lr=$DYLORA_LR)${flash_suffix}" ;;
        vera)    echo "VeRA (r=$VERA_R, d=$VERA_D_INITIAL, lr=$VERA_LR)${flash_suffix}" ;;
    esac
}

# ============================================================================
# MAIN LOOP
# ============================================================================

echo "============================================"
echo "Commonsense MC Benchmark Suite"
echo "============================================"
echo "Model:      $MODEL"
echo "Task:       $TASK (train 170K -> eval 8 sets -> 8 rows/technique)"
echo "Techniques: ${techniques[*]}"
echo "Dtype:      $DTYPE"
echo "Epochs:     $CS_EPOCHS | batch=$BATCH_SIZE seq=$MAX_LENGTH total_batch=$TOTAL_BATCH_SIZE"
echo "Per-seed:   array 0-4 + CPU aggregation (always split)"
echo "V3 modes:   ${V3_MODES:-<disabled>} (FlashFFNV3 arms: <tech>_flashffnv3_<mode>)"
echo "V3 only:    ${V3_ONLY} ($([[ "$V3_ONLY" == "1" ]] && echo "v3 arms ONLY — baselines + v2 _flash skipped" || echo "full sweep: baselines + v2 _flash + v3"))"
echo "============================================"
echo ""

account_line=""
if [[ -n "$ACCOUNT" ]]; then
    account_line="#SBATCH --account=$ACCOUNT"
fi

for technique in "${techniques[@]}"; do
    technique_desc=$(get_technique_desc "$technique")
    job_name="${MODEL_SHORT}_cs_${technique}"
    run_name="${technique}_${MODEL_SHORT}_commonsense"

    get_job_resources "$technique"
    python_cmd=$(build_python_cmd "$technique" "$run_name")

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

    # ---- Per-seed array (0-4) + dependent CPU aggregation job ----
    # Each array task trains ONE seed and writes JSON partials (one per eval set is
    # carried inside multi_eval); the aggregation job (afterany) recombines them into
    # the 8 Mo5 rows. The partial dir is cleared at submit so a resubmit never mixes
    # stale seeds.
    partial_dir="./results/_partial/${run_name}"
    seed_time=$(get_per_seed_time "$technique")
    rm -rf "$partial_dir"

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
# Reduce CUDA allocator fragmentation on the long (multi-day), memory-tight 40 GB
# runs. Harmless on the 80 GB DoRA jobs. Keeps peak well under the slice limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p \$HF_HOME
export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src

SEEDS_ARR=(${SEEDS_LIST[*]})
SEED=\${SEEDS_ARR[\$SLURM_ARRAY_TASK_ID]}
echo '========================================'
echo "Job: $job_name (seed \$SEED, array idx \$SLURM_ARRAY_TASK_ID)"
echo "Config: $technique_desc | Task: $TASK | Epochs: $CS_EPOCHS"
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
echo "Aggregating Mo5 (8 rows): $run_name  (depends on array $array_id)"
echo "Started: \$(date)"
echo '========================================'
$python_cmd --aggregate --partial_dir $partial_dir
echo '========================================'
echo "Finished aggregation: \$(date)"
echo '========================================'
EOF
)
    echo "  [array ${array_id} (0-4, ${seed_time}/seed) -> agg ${agg_id}] $job_name  ($technique_desc)"
    job_count=$((job_count + 2))
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results CSV:          ./results/mo53_glue.csv"
echo "Logs directory:       ./logs/"
echo "============================================"
