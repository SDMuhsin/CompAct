#!/bin/bash
# ============================================================================
# 05_sweep_task.sh — the camera-ready sweep: ALL baselines, ONE model, ONE task.
# ============================================================================
#   bash sbatch/fir/05_sweep_task.sh                    # submit (task=boolq)
#   P_TASK=rte bash sbatch/fir/05_sweep_task.sh         # a different task
#   bash sbatch/fir/05_sweep_task.sh --dry-run          # print the manifest, submit nothing
#   bash sbatch/fir/05_sweep_task.sh --report           # re-run the gate over what exists
#   bash sbatch/fir/05_sweep_task.sh --local            # run the whole grid here (salloc)
#
# WHAT THIS SWEEPS
# ----------------
# Every (method, with_fb) pair the registry PERMITS, x every seed. The arm list is
# NOT hardcoded here -- it comes from `validate_glue_runner.arms()`, which is the
# same function the validator uses, so adding a method to REGISTRY adds it to this
# sweep and there is no second list to drift. Today that is 13 arms x 3 seeds = 39
# cells, in two regimes that are NOT comparable with each other:
#
#   peft_lora (7)  baseline, baseline_fb (ours), qlora, minis, streambp, alst, zero3
#   full_ft   (6)  galore, galore_fb, lomo, lomo_fb, adalomo, adalomo_fb
#
# ONE JOB PER CELL, AS A SLURM ARRAY -- deliberately, for three reasons:
#   1. A cell that dies takes down one cell. A monolithic job loses the whole grid,
#      and this grid is ~14 GPU-hours.
#   2. `run_production.write_row` is a lock-protected atomic upsert built for exactly
#      this ("N of these can run concurrently" -- its docstring), so the CSV is safe.
#   3. Re-running the script SKIPS cells already clean in the CSV, so a partial sweep
#      resumes instead of restarting. Nothing is ever recomputed for free.
#
# SOME CELLS WILL NOT PRODUCE A ROW, AND THAT CAN BE THE CORRECT OUTCOME. The script
# RECORDS which, from the run's own output -- it does NOT predict them:
#
#   REFUSED  illegal by construction. Confirmed for `streambp` on a classification
#            head: StreamBP's mechanism IS the chunked LM head (it reads
#            `model.lm_head.weight.grad`, stream_model.py:622-629), and
#            LlamaForSequenceClassification has `score`, emitting one logit vector per
#            sequence with nothing to chunk. run_production exits 0, writes no row.
#   INERT    built, but proved no work, so `verify_engagement` refused the row.
#            EXPECTED for `alst` here: their rule is num_shards = ceil(seq/hidden),
#            which is 1 on a 2048-hidden model at any GLUE length -> no tiling.
#            ⚠ NOT YET OBSERVED -- the first sweep's alst cells died earlier, on a
#            missing PYTHONPATH, so this remains a prediction until a cell reaches it.
#   ERROR    anything else. A broken cell. Fails the sweep, loudly.
#
# **Never "fix" a REFUSED or INERT cell with --allow_inert** -- that writes a row
# claiming a method ran when it did not. Report it as inapplicable at this shape,
# citing the method's own rule.
#
# ⚠ REGIME. GLUE runs on a sequence-classification head, so `--flce` is meaningless
#   and refused (CONTEXT.md §8). Every row here is regime A by construction and the
#   `flce` column will read 0. Do not compare these against a regime-B memory table.
# ============================================================================
set -uo pipefail

FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_sweep_task "$@"        # full transcript -> ./logs/fir_sweep_task_<UTC>.log
mkdir -p logs results/production results/production/_sweep

MODE="submit"
case "${1:-}" in
    --dry-run) MODE="dry" ;;
    --report)  MODE="report" ;;
    --local)   MODE="local" ;;
    "")        ;;
    *) echo "unknown flag ${1}; see the header"; exit 2 ;;
esac

# --- the cell grid. CAMERA-READY values, matched to the pilot that validated them. --------
P_MODEL="${P_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
# TASK FAMILY. `glue` = sequence-classification head; `lm` = causal-LM head over a real corpus.
# ⚠ THE FAMILY CHANGES WHICH BASELINES ARE MEASURABLE AT ALL -- see the header. One script serves
#   both deliberately: every hard-won behaviour here (recorded outcomes, run_id scoping,
#   needs_pythonpath, per-method LR/clip, the wstream guard, the fb_variant axis) is family-neutral,
#   and a forked script would drift from it.
P_FAMILY="${P_FAMILY:-glue}"
case "$P_FAMILY" in
    glue) P_TASK="${P_TASK:-boolq}" ;;
    lm)   P_TASK="${P_TASK:-wikitext2}" ;;
    *) echo "P_FAMILY must be glue or lm, got '$P_FAMILY'"; exit 2 ;;
esac
P_SEEDS="${P_SEEDS:-41 42 43}"
P_EPOCHS="${P_EPOCHS:-3}"
# ⚠ EMPTY ON PURPOSE: `run_production.resolve_seq` applies the measured per-task policy
#   (128 everywhere, 384 for boolq, because at 128 boolq truncates 61.4% of its examples).
#   The row records `seq_source` either way, so an override is always visible in the data.
P_SEQ="${P_SEQ:-}"
# ⚠ PER-FAMILY BATCH. GLUE sequences are 128-384 tokens, so batch 16 is ~6k tokens/step. An LM
#   block is `seq` tokens by construction, so batch 16 at seq 2048 would be 32k tokens/step -- 16x
#   the §8 protocol shape and a different experiment. batch 4 x seq 2048 = 8192 tokens/step keeps
#   the step size in the same range as the GLUE cells and near the memory table's operating point.
case "$P_FAMILY" in
    lm) P_BATCH="${P_BATCH:-4}" ;;
    *)  P_BATCH="${P_BATCH:-16}" ;;
esac
P_LR="${P_LR:-2e-4}"
# ⚠ RUN_ID SCOPES RESUME AND THE REPORT, AND THAT IS LOAD-BEARING -- BUMP IT WHEN THE
#   CONFIGURATION CHANGES. `check_row` validates a row's NUMBERS, not the configuration that
#   produced it, so a row can be perfectly clean and still scientifically invalid. Sweep 1's
#   galore/qlora/adalomo rows all PASS `check_row` while having no gradient checkpointing and the
#   wrong LR -- verified. Without run_id scoping, resume would mark those cells done and **skip
#   exactly the arms that most need re-running**. Old rows are never deleted (§1); they stay under
#   their own run_id and are ignored by both resume and the report.
#   `_v2` = matched gc + published per-method LR + published clipping + the fb_variant axis.
# ⚠ THE `glue` FORM IS UNCHANGED ON PURPOSE. A glue sweep is IN FLIGHT on fir under run_id
#   `camera_ready_boolq_v2`; adding a family prefix to it would make resume and the report scope to
#   a run_id no existing row carries, silently orphaning that sweep's results. New families get the
#   prefix; glue keeps the name its rows were written with.
case "$P_FAMILY" in
    glue) P_RUN_ID="${P_RUN_ID:-camera_ready_${P_TASK}_v2}" ;;
    *)    P_RUN_ID="${P_RUN_ID:-camera_ready_${P_FAMILY}_${P_TASK}_v2}" ;;
esac
# Per-family CSV. `glue` resolves to the same path it always did, so the in-flight sweep's rows
# stay where they are; `lm` gets its own file rather than putting causal-LM rows in a file named
# for GLUE. The `task` column distinguishes them if they are ever concatenated.
P_CSV="${P_CSV:-results/production/camera_ready_${P_FAMILY}.csv}"

# Per-cell wall time. The pilot's heaviest arm (zero3, the only DeepSpeed-engine arm) took
# 1243 s for 3 epochs of boolq -- the largest SuperGLUE split. 3 h is ~8.7x that, which covers
# the full-FT arms training 1.1B parameters instead of 16M without a per-arm table.
P_TIME="${P_TIME:-3:00:00}"
# Array throttle. A shared cluster and a shared CSV lock; 8 concurrent is plenty to finish
# 39 cells in about 3 waves and is polite about the account's GPU share.
P_THROTTLE="${P_THROTTLE:-8}"

MANIFEST="results/production/_sweep/${P_RUN_ID}_manifest.tsv"
# Recorded (not predicted) outcomes for cells that produced no row. See run_cell's classifier.
OUTCOMES="results/production/_sweep/${P_RUN_ID}_outcomes.tsv"

# ---------------------------------------------------------------------------
# PER-METHOD LEARNING RATE — the authors' own published value, cited.
#
# ⚠ ONE LR FOR EVERY ARM IS DETUNING, and the first sweep (54846646) did exactly that: it ran
#   `--lr 2e-4` everywhere, which is a LoRA LR. What that did to the full-FT arms was NOT what I
#   first assumed. LOMO is SGD-like -- its update is a fused `p.add_(grad, alpha=-self.lr)`
#   (temp/lomo/lomo/src/lomo.py:101) -- so its published LR is 0.03, **150x LARGER** than what we
#   gave it. Its collapse to the majority class was UNDER-training, not divergence.
#
# Citations, all verified in the vendored upstream repos under temp/:
#   lomo     0.03    temp/lomo/lomo/config/args_lomo.yaml:20   (LLaMA-7B, SuperGLUE, full-param)
#   adalomo  5e-4    temp/lomo/adalomo/instruction-tuning/train.py:145   (LLaMA-7B, Alpaca-GPT4)
#   qlora    2e-4    temp/qlora/qlora.py:203 + scripts/finetune_guanaco_7b.sh:38  (<=13B default)
#   galore   3e-5    temp/galore/README.md:146  (GaLore on GLUE MRPC, RoBERTa-base)
#
# ⚠ THREE CAVEATS THAT MUST TRAVEL WITH THIS TABLE.
#   1. GaLore's repo contains NO full-fine-tuning LR at all. Its LLaMA-1B value (0.01,
#      scripts/benchmark_c4/llama_1b.sh:4) is a PRETRAINING LR and must never be reused as a
#      fine-tuning LR. 3e-5 is the closest fine-tuning number they publish and it is on a 125M
#      RoBERTa, not a 1.1B decoder. Disclose this in any table that quotes GaLore's quality.
#   2. NO repo publishes a ~1B-scale fine-tuning LR. Every value above is transplanted from 7B
#      (or from 125M for GaLore). That is a disclosed limitation, not a hidden one.
#   3. ⚠ THE LR IS NOT THE WHOLE SETTING. Clipping now travels with it (see `cell_clip`), but the
#      SCHEDULERS still do not: LOMO ships `warmup: 0.1` + linear (args_lomo.yaml:23-24) and QLoRA
#      ships constant + `warmup_ratio 0.03`. `run_production --lr_scheduler/--warmup_ratio` exist
#      and RAISE on the DeepSpeed and LOMO arms, which own their own update path -- so those two
#      recipes are reproduced on LR and clipping but not on schedule. State that in the table.
# ---------------------------------------------------------------------------
cell_lr() {
    case "$1" in
        lomo)    echo "${P_LR_LOMO:-0.03}"    ;;
        adalomo) echo "${P_LR_ADALOMO:-5e-4}" ;;
        galore)  echo "${P_LR_GALORE:-3e-5}"  ;;
        *)       echo "$P_LR"                 ;;   # peft arms + qlora: 2e-4, their own default
    esac
}

# Gradient-norm clipping, same sourcing rule as the LR. Empty = off, which is how every row
# before 2026-08-15 was measured.
#
# ⚠ LOMO's 1.0 IS EXPENSIVE AND THAT IS THEIRS, NOT OURS. Their clipping is a two-pass protocol
#   (`grad_norm(loss)` + a second forward before `fused_backward`, lomo_trainer.py:105-172), which
#   MEASURED +74% peak on this artifact (8305 -> 14469 MiB). It is in their published config
#   (args_lomo.yaml:25), so running without it is also a deviation -- just a cheaper one. Whichever
#   way this is set, the lomo row must say which, because it moves the memory column a lot.
#   Default here is ON, i.e. faithful to their config; set P_CLIP_LOMO= to turn it off.
# ⚠ REGIME B, AND IT ONLY EXISTS IN THE `lm` FAMILY. §8: "B is where competitor claims are
#   adjudicated" -- the LM-head + fp32-CE stack is 875 MiB at seq 1024/batch 2, larger than
#   everything the block saves, and unmatched it decides the comparison on its own. `--flce` is
#   REFUSED on a GLUE head (no LM head, no vocab-sized logits), so every glue row is regime A by
#   construction; a causal-LM head is the only place the choice exists.
#   Skipped for `minis`/`streambp` (FLCE_FORBIDDEN): their own mechanism IS a chunked/fused LM head
#   and loss, so stacking Liger's on top would be two implementations of one optimisation and would
#   measure neither. That makes regime B the MATCHED control it is meant to be -- every arm gets a
#   fused CE, either Liger's or its own -- rather than an asymmetry. Say so in the table.
#   Measured on the dev box: baseline_fb lm:wikitext2 seq 2048 goes 3249.14 -> 2482.24 MiB with
#   identical perplexity, and 2482.24 reproduces §4.1's published fb_min 2482.02 to 0.2 MiB.
cell_flce() {
    [ "$P_FAMILY" = "lm" ] || { echo ""; return; }
    [ "${P_FLCE:-1}" = "1" ] || { echo ""; return; }
    case "$1" in
        minis|streambp) echo "" ;;            # FLCE_FORBIDDEN -- they ship their own fused head
        *)              echo "--flce" ;;
    esac
}

cell_clip() {
    case "$1" in
        lomo|adalomo) echo "${P_CLIP_LOMO-1.0}" ;;   # args_lomo.yaml:25
        qlora)        echo "${P_CLIP_QLORA-0.3}" ;;  # temp/qlora/qlora.py:205
        *)            echo "${P_CLIP-}"          ;;
    esac
}

# ---------------------------------------------------------------------------
# Build the manifest: one line per cell that still needs running.
# ---------------------------------------------------------------------------
# ⚠ `$FIR_VENV/bin/python`, NOT `python`. This runs on the LOGIN NODE at submit time, where no
#   venv is active and `python` may not exist at all -- the interpreter has to be named explicitly.
#   It also needs pandas, which lives in the venv and not in the bare module stack.
build_manifest() {
    P_TASK="$P_TASK" P_SEEDS="$P_SEEDS" P_CSV="$P_CSV" P_SEQ="$P_SEQ" P_RUN_ID="$P_RUN_ID" \
    P_FAMILY="$P_FAMILY" \
    MANIFEST="$MANIFEST" "$FIR_VENV/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, "src")
from validate_glue_runner import arms, check_row
from experiment_registry import FULL_FT_REGIME, LOSSY, resolve_fb_variant, CombinationRefused
from run_production import TASK_MAX_LENGTH, GLUE_DEFAULT_SEQ, LM_DEFAULT_SEQ

task     = os.environ["P_TASK"]
seeds    = [int(s) for s in os.environ["P_SEEDS"].split()]
csv_path = os.environ["P_CSV"]
run_id   = os.environ["P_RUN_ID"]
manifest = os.environ["MANIFEST"]
family   = os.environ["P_FAMILY"]
seq      = int(os.environ["P_SEQ"]) if os.environ.get("P_SEQ") else (
           LM_DEFAULT_SEQ if family == "lm" else TASK_MAX_LENGTH.get(task, GLUE_DEFAULT_SEQ))

# ⚠ SCOPED TO run_id. `check_row` validates NUMBERS, not the configuration that produced them, so
# a row can be clean and still invalid -- sweep 1's galore/qlora/adalomo rows all pass it while
# having no gradient checkpointing and the wrong LR. Matching on (method, fb, variant, seed) alone
# would mark those done and skip the very arms that need re-running. Bump P_RUN_ID on any
# configuration change; the old rows stay in the CSV under their own run_id.
done_cells, dirty = set(), 0
if os.path.exists(csv_path):
    import pandas as pd
    d = pd.read_csv(csv_path)
    d = d[(d.task == f"{family}:{task}") & (d.run_id == run_id)]
    for _, r in d.iterrows():
        r = r.to_dict()
        key = (str(r.get("method")), int(r.get("with_fb") or 0),
               str(r.get("fb_variant") or "none"), int(r.get("seed")))
        if check_row(r):
            dirty += 1                      # leave it in the CSV; the upsert will replace it
        else:
            done_cells.add(key)

# ⚠ NO REFUSAL PREDICTION HERE ANY MORE. An earlier revision marked alst `expect=refusal` from
# ALST's `num_shards = ceil(seq/hidden)` rule, and `run_cell` then accepted ANY non-zero exit as
# that refusal -- which silently absorbed a completely unrelated environment bug (missing
# PYTHONPATH) and reported it as an expected methodological outcome. Refusals are now RECORDED by
# `run_cell` from the run's own output, never predicted. Cells that refuse cost seconds, so there
# is nothing to save by guessing in advance.
# ⚠ `fb_variant` IS A THIRD AXIS AND `arms()` DOES NOT COVER IT. `arms()` enumerates
# method x with_fb only, which is why sweep 1 silently contained NO `wstream` cell at all -- the
# project's best peak-memory result (floor 2188.66 -> 88.66, a flat -1932.00 MiB) was missing from
# a sweep whose whole point is peak memory. Derived from `resolve_fb_variant`, not hardcoded: it
# declines wstream on every full-fine-tuning arm, because WP-E streams the FROZEN base weights and
# a trainable base weight makes saving them as `None` unsound. Today that means exactly one arm --
# `baseline_fb` -- carries both variants, so this adds 3 cells.
def variants_for(method, fb):
    if not fb:
        return ["none"]                      # non-fb rows record fb_variant="none"
    out = ["min"]
    try:
        resolve_fb_variant(method, fb, "wstream")
        out.append("wstream")
    except CombinationRefused:
        pass                                 # structurally inapplicable; a correct exclusion
    return out

lines, skipped = [], 0
for method, fb in arms():
    for variant in variants_for(method, fb):
        for seed in seeds:
            if (method, int(fb), variant, seed) in done_cells:
                skipped += 1
                continue
            regime = "full_ft" if method in FULL_FT_REGIME else "peft_lora"
            note   = "LOSSY" if method in LOSSY else "-"
            lines.append(f"{method}\t{int(fb)}\t{seed}\t{variant}\t{regime}\t{note}")

with open(manifest, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))

print(f"  task           {family}:{task}   seq={seq}")
print(f"  seeds          {seeds}")
print(f"  cells total    {sum(len(variants_for(m, f)) for m, f in arms()) * len(seeds)}")
print(f"  already clean  {len(done_cells)} (skipped)")
print(f"  dirty rows     {dirty} (will be re-run and upserted)" if dirty else "  dirty rows     0")
print(f"  to run         {len(lines)}")
print(f"  manifest       {manifest}")
PY
}

# ---------------------------------------------------------------------------
# Run ONE cell. Exported into the array job verbatim.
# ---------------------------------------------------------------------------
run_cell() {
    local method="$1" fb="$2" seed="$3" variant="${4:-min}"
    local fbflag=""; [ "$fb" = "1" ] && fbflag="--fb --fb_variant $variant"
    local tag="${method}$([ "$fb" = "1" ] && echo "_fb")$([ "$variant" = "wstream" ] && echo "+ws")"

    fir_load_modules_gpu || return 1
    # shellcheck disable=SC1091
    source "$FIR_VENV/bin/activate" || return 1
    fir_export_offline

    # ⚠ HONOUR `needs_pythonpath`. THIS WAS MISSING AND IT COST THE FIRST SWEEP ALL THREE alst
    #   CELLS (array 54846646 tasks 6-8, 2026-08-15). ALST needs DeepSpeed >= 0.17 from the
    #   side-by-side prefix `temp/ds_alst`; without it the venv's deliberately-pinned 0.16.5 loads
    #   and `build_alst_model` raises. `validate_glue_runner.run_arm:113-115` already did this and
    #   this function did not.
    #   ⚠ APPEND, NEVER ASSIGN: on Alliance clusters numpy comes from the scipy-stack module via
    #   PYTHONPATH, so `PYTHONPATH=temp/ds_alst` deletes numpy and deepspeed dies with
    #   `ModuleNotFoundError: No module named 'numpy'` (hpc_fir.md §3 failure 6).
    local need
    need=$("$FIR_VENV/bin/python" -c "
import sys; sys.path.insert(0, 'src')
from experiment_registry import REGISTRY
print(REGISTRY['$method'].needs_pythonpath or '')" 2>/dev/null)
    if [ -n "$need" ]; then
        export PYTHONPATH="$(pwd)/$need:${PYTHONPATH:-}"
        echo "  needs_pythonpath: prepended $(pwd)/$need"
    fi

    # ⚠ PER-CELL TORCH EXTENSION DIR. DeepSpeed JIT-builds `fused_adam` (62 s, seen in pilot
    #   54770596) into $TORCH_EXTENSIONS_DIR. Array tasks run CONCURRENTLY ON DIFFERENT NODES
    #   against one shared filesystem, and DeepSpeed's build baton does not make that safe across
    #   nodes -- two cells building into one directory can read a half-written .so. Isolating per
    #   cell costs the rebuild (~62 s, and ONLY on the 3 zero3 cells; no other arm builds an op)
    #   and removes the race outright. Do not "optimise" this into a shared directory.
    export TORCH_EXTENSIONS_DIR="${FIR_SCRATCH_ROOT}/.torch_ext/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "$TORCH_EXTENSIONS_DIR"

    # ⚠ NODE IDENTITY AND A t-1s CONTEXT PROBE. Job 54759272 died in `torch.cuda.set_device()`
    #   with `CUDA error: out of memory` -- CONTEXT CREATION, nothing of ours allocated -- on a
    #   card its own nvidia-smi reported as free, and the log could not say which node. Across 39
    #   cells this WILL recur; when it does, these two lines make it attributable to a named node
    #   instead of costing another round trip. (It has not yet fired as a discriminator: the probe
    #   has passed on every green run so far.)
    echo "node: $(hostname)  JOB=${SLURM_JOB_ID:-<none>}  ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-<none>}"
    python - <<'PROBE'
import torch
torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info(0)
print(f"  context OK on {torch.cuda.get_device_name(0)}: "
      f"free={free/2**20:.0f} MiB / total={total/2**20:.0f} MiB", flush=True)
PROBE
    local prc=$?
    [ $prc -eq 0 ] || echo "  ⚠ CONTEXT PROBE FAILED (exit $prc) on $(hostname) -- see CONTEXT.md §15.2"

    echo "--- cell: $tag x $P_FAMILY:$P_TASK seed=$seed lr=$(cell_lr "$method") clip=$(cell_clip "$method") epochs=$P_EPOCHS ---"
    local t0 rc=0 out
    t0=$(date +%s)
    # ⚠ NO --allow_inert, EVER. It writes a row for a method that proved no work, which is the
    #   one failure this whole registry exists to prevent.
    # Output is teed so the CLASSIFIER below can read it while the log still shows everything live.
    local _clip; _clip=$(cell_clip "$method")
    out=$(python src/run_production.py \
        --method "$method" $fbflag --task "$P_FAMILY:$P_TASK" --seed "$seed" \
        --model "$P_MODEL" --epochs "$P_EPOCHS" ${P_SEQ:+--seq "$P_SEQ"} \
        --batch "$P_BATCH" --lr "$(cell_lr "$method")" ${_clip:+--max_grad_norm "$_clip"} \
        $(cell_flce "$method") \
        --device cuda:0 --warmup_steps 8 \
        --run_id "$P_RUN_ID" --out_csv "$P_CSV" 2>&1 | tee /dev/stderr) || rc=1
    echo "--- $tag seed=$seed wall-clock: $(( $(date +%s) - t0 ))s  rc=$rc ---"

    # ------------------------------------------------------------------------
    # CLASSIFY WHAT ACTUALLY HAPPENED. Do NOT predict it.
    #
    # ⚠ THIS REPLACES AN `expect=refusal` PREDICTION THAT PRODUCED A SILENT FAILURE. The first
    #   sweep (54846646) marked the alst cells `expect=refusal` and then treated ANY non-zero exit
    #   as that expected refusal -- so a missing PYTHONPATH, which is an ENVIRONMENT BUG, was
    #   reported as "alst has nothing to do at this sequence length. OK." A prediction that
    #   swallows its own falsification is worse than no check. Three outcomes, distinguished by
    #   evidence from the run itself:
    #
    #     REFUSED  `CombinationRefused` -- the pair is illegal BY CONSTRUCTION (e.g. streambp on a
    #              classification head: its mechanism IS the chunked LM head and there is none).
    #              run_production exits 0 and writes no row. A correct outcome.
    #     INERT    `EngagementFailure` -- the method built but proved no work, so the registry
    #              refused to write a row claiming it ran. Also a correct outcome, and the one
    #              alst is EXPECTED to reach once its PYTHONPATH is right.
    #     ERROR    anything else. NOT correct. Must be loud.
    # ------------------------------------------------------------------------
    local status="ok"
    if [ $rc -eq 0 ] && grep -q "^REFUSED " <<<"$out"; then
        status="REFUSED"
    elif grep -q "EngagementFailure" <<<"$out"; then
        status="INERT"
    elif [ $rc -ne 0 ]; then
        status="ERROR"
    fi

    if [ "$status" != "ok" ]; then
        # One line per non-ok cell, read back by run_report. Appended under flock because array
        # tasks land here concurrently.
        local reason
        reason=$(grep -m1 -E "^REFUSED |EngagementFailure|^[A-Za-z_.]*Error" <<<"$out" | tr '\t\n' '  ' | cut -c1-300)
        mkdir -p "$(dirname "$OUTCOMES")"
        flock "$OUTCOMES.lock" -c \
            "printf '%s\t%s\t%s\t%s\t%s\t%s\n' '$method' '$fb' '$seed' '$variant' '$status' '${reason//\'/}' >> '$OUTCOMES'"
        echo "  OUTCOME=$status  $reason"
    fi

    # REFUSED and INERT are correct outcomes and must not fail the cell. ERROR must.
    [ "$status" = "ERROR" ] && return 1
    return 0
}

# ---------------------------------------------------------------------------
# The camera-ready report: apply the anomaly gate to every cell of the grid.
# ---------------------------------------------------------------------------
# Named interpreter for the same reason as `build_manifest`: this is invoked both from the login
# node (--report) and from a CPU-only report job, neither of which has an activated venv.
run_report() {
    P_TASK="$P_TASK" P_SEEDS="$P_SEEDS" P_CSV="$P_CSV" P_SEQ="$P_SEQ" OUTCOMES="$OUTCOMES" \
    P_FAMILY="$P_FAMILY" \
    P_RUN_ID="$P_RUN_ID" "$FIR_VENV/bin/python" - <<'PY'
import os, sys, json
sys.path.insert(0, "src")
import pandas as pd
from validate_glue_runner import arms, check_row
from experiment_registry import FULL_FT_REGIME, resolve_fb_variant, CombinationRefused
from run_production import TASK_MAX_LENGTH, GLUE_DEFAULT_SEQ, LM_DEFAULT_SEQ

def variants_for(method, fb):
    if not fb:
        return ["none"]
    out = ["min"]
    try:
        resolve_fb_variant(method, fb, "wstream"); out.append("wstream")
    except CombinationRefused:
        pass
    return out

task  = os.environ["P_TASK"]
seeds = [int(s) for s in os.environ["P_SEEDS"].split()]
csv_path = os.environ["P_CSV"]
family = os.environ["P_FAMILY"]
seq = int(os.environ["P_SEQ"]) if os.environ.get("P_SEQ") else (
      LM_DEFAULT_SEQ if family == "lm" else TASK_MAX_LENGTH.get(task, GLUE_DEFAULT_SEQ))

# RECORDED outcomes, not predicted ones. `run_cell` appends a line for every cell that did not
# produce a row, tagged REFUSED (illegal by construction) / INERT (built but proved no work) /
# ERROR (anything else). A cell with no row and no recorded outcome is genuinely MISSING -- it
# never ran, or it died before it could classify itself.
outcomes = {}
opath = os.environ.get("OUTCOMES", "")
if opath and os.path.exists(opath):
    for line in open(opath):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 5:
            outcomes[(p[0], int(p[1]), int(p[2]), p[3])] = (p[4], p[5] if len(p) > 5 else "")

if not os.path.exists(csv_path):
    print(f"  no CSV at {csv_path}"); sys.exit(1)
d = pd.read_csv(csv_path)
# Scoped to run_id for the same reason resume is -- a clean row from a superseded configuration
# must not be reported as a result of THIS sweep.
d = d[(d.task == f"{family}:{task}") & (d.run_id == os.environ["P_RUN_ID"])]

rows, n_ok, n_bad, n_missing, n_refused, n_error = [], 0, 0, 0, 0, 0
for method, fb in arms():
  for variant in variants_for(method, fb):
    tag = method + ("_fb" if fb else "") + ("+ws" if variant == "wstream" else "")
    regime = "full_ft" if method in FULL_FT_REGIME else "peft_lora"
    for seed in seeds:
        c = d[(d.method == method) & (d.with_fb == int(fb)) & (d.seed == seed)
              & (d.fb_variant.astype(str) == variant)]
        if not len(c):
            oc, reason = outcomes.get((method, int(fb), seed, variant), (None, ""))
            if oc in ("REFUSED", "INERT"):
                status = f"{oc}: {reason[:60]}"
                n_refused += 1
            elif oc == "ERROR":
                status = f"ERROR: {reason[:60]}"
                n_error += 1
            else:
                status, n_missing = "MISSING (never ran)", n_missing + 1
            rows.append((regime, tag, seed, status, None, None, None))
            continue
        r = c.iloc[-1].to_dict()
        bad = check_row(r)
        if bad:
            status, n_bad = "ANOMALY: " + "; ".join(bad)[:70], n_bad + 1
        else:
            status, n_ok = "ok", n_ok + 1
        rows.append((regime, tag, seed, status,
                     r.get("task_metric"), r.get("train_step_peak_alloc_mib"),
                     r.get("ms_per_step_median")))

for regime in ("peft_lora", "full_ft"):
    sel = [x for x in rows if x[0] == regime]
    if not sel:
        continue
    print(f"\n=== {regime} — {family}:{task}, seq {seq} ===")
    print(f"  {'arm':14} {'seed':>4}  {'metric':>8} {'peak MiB':>10} {'ms/step':>9}  status")
    for _, tag, seed, status, m, p, ms in sel:
        f = lambda v, w, d=4: (f"{float(v):{w}.{d}f}" if v is not None and v == v else " " * w)
        print(f"  {tag:14} {seed:>4}  {f(m,8)} {f(p,10,2)} {f(ms,9,2)}  {status}")

print(f"\n  ok {n_ok} | anomalies {n_bad} | missing {n_missing} | "
      f"refused/inert {n_refused} | ERRORS {n_error}")
# REFUSED and INERT are correct outcomes and do NOT fail the sweep -- they are findings about
# where a method applies. Anomalies, gaps and ERRORS do fail it.
if n_error:
    print("  ⚠ ERRORS ABOVE ARE NOT METHODOLOGICAL REFUSALS -- they are broken cells. Fix them.")
if n_bad or n_missing or n_error:
    print("  SWEEP INCOMPLETE -- re-run this script to fill gaps; investigate anomalies first.")
    sys.exit(1)
print("  SWEEP CLEAN -- every permitted cell either has a clean row or a recorded refusal.")
PY
}

# ---------------------------------------------------------------------------
echo "=== camera-ready sweep: all baselines x $P_FAMILY:$P_TASK ==="
echo "  model $P_MODEL   epochs $P_EPOCHS   batch $P_BATCH   lr $P_LR"
echo "  csv   $P_CSV   run_id $P_RUN_ID"
echo

if [ "$MODE" = "report" ]; then
    run_report; exit $?
fi

build_manifest || { echo "manifest build FAILED"; exit 1; }
N=$(wc -l < "$MANIFEST" 2>/dev/null || echo 0)
echo
if [ "$N" -eq 0 ]; then
    echo "  nothing to run -- every cell is already clean. Running the report instead."
    run_report; exit $?
fi

if [ "$MODE" = "dry" ]; then
    echo "  --- manifest (method / fb / seed / variant / regime / note) ---"
    nl -ba "$MANIFEST" | sed 's/^/    /'
    echo
    echo "  would submit: array 0-$((N-1))%$P_THROTTLE, ${P_TIME} per cell"
    exit 0
fi

if [ "$MODE" = "local" ]; then
    rc=0
    while IFS=$'\t' read -r method fb seed variant regime note; do
        run_cell "$method" "$fb" "$seed" "$variant" || rc=1
    done < "$MANIFEST"
    echo; run_report || rc=1
    exit $rc
fi

# --- fail on the LOGIN node, not 39 times on compute nodes -------------------
fir_assert_env cpu || { echo "environment is not sane -- refusing to submit $N cells"; exit 1; }

jid=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=fir_sweep_${P_FAMILY}_${P_TASK}
#SBATCH --output=./logs/fir_sweep_${P_FAMILY}_${P_TASK}_%A_%a.out
#SBATCH --error=./logs/fir_sweep_${P_FAMILY}_${P_TASK}_%A_%a.err
#SBATCH --time=$P_TIME
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --cpus-per-task=4
#SBATCH --account=$FIR_ACCOUNT_GPU
#SBATCH --array=0-$((N-1))%$P_THROTTLE

cd "\$SLURM_SUBMIT_DIR"
source sbatch/fir/fir_env.sh
export P_MODEL="$P_MODEL" P_FAMILY=$P_FAMILY P_TASK=$P_TASK P_EPOCHS=$P_EPOCHS P_SEQ="$P_SEQ"
export P_FLCE="${P_FLCE:-1}"
export P_BATCH=$P_BATCH P_LR=$P_LR P_RUN_ID=$P_RUN_ID P_CSV=$P_CSV
export OUTCOMES="$OUTCOMES"
line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=\$'\t' read -r method fb seed variant regime note <<< "\$line"
echo "array task \$SLURM_ARRAY_TASK_ID -> \$method fb=\$fb variant=\$variant seed=\$seed (\$regime)"
$(declare -f cell_lr)
$(declare -f cell_clip)
$(declare -f cell_flce)
$(declare -f run_cell)
run_cell "\$method" "\$fb" "\$seed" "\$variant"
EOF
)
echo "submitted sweep array $jid  ($N cells, throttle %$P_THROTTLE, $P_TIME each)"

# The report runs after the LAST cell finishes, whatever its exit status -- a sweep with a
# failed cell is exactly when the summary is most useful. `afterany` on the array as a whole.
rid=$(sbatch --parsable --dependency=afterany:$jid <<EOF
#!/bin/bash
#SBATCH --job-name=fir_sweep_${P_FAMILY}_${P_TASK}_report
#SBATCH --output=./logs/fir_sweep_${P_FAMILY}_${P_TASK}_report_%j.out
#SBATCH --error=./logs/fir_sweep_${P_FAMILY}_${P_TASK}_report_%j.err
#SBATCH --time=0:20:00
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=2
#SBATCH --account=$FIR_ACCOUNT_CPU

cd "\$SLURM_SUBMIT_DIR"
source sbatch/fir/fir_env.sh
fir_load_modules_cpu
export P_FAMILY=$P_FAMILY P_TASK=$P_TASK P_SEEDS="$P_SEEDS" P_CSV=$P_CSV P_SEQ="$P_SEQ" OUTCOMES="$OUTCOMES"
$(declare -f run_report)
run_report
EOF
)
echo "submitted report job $rid (runs after the array, whatever it exits with)"
echo
echo "  watch:   squeue -j $jid"
echo "  cells:   tail -f logs/fir_sweep_${P_FAMILY}_${P_TASK}_${jid}_0.out"
echo "  report:  cat logs/fir_sweep_${P_FAMILY}_${P_TASK}_report_${rid}.out"
echo "  resume:  re-run this script -- clean cells are skipped, gaps are refilled"
