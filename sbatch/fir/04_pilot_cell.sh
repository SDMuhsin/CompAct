#!/bin/bash
# ============================================================================
# 04_pilot_cell.sh — ONE cell of the camera-ready GLUE sweep, submitted alone.
# ============================================================================
#   bash sbatch/fir/04_pilot_cell.sh            # submit
#   bash sbatch/fir/04_pilot_cell.sh --local    # run here (inside salloc)
#
# WHY ONE CELL, AND WHY THIS ONE
# ------------------------------
# The sweep is 8 tasks x 3 seeds x ~7 usable arms. Submitting it before a single
# cell has ever run on fir would spend hundreds of GPU-hours discovering a
# missing cache entry. So: run the MOST TROUBLESOME cell first, at the REAL
# configuration, and size the rest from its wall-clock.
#
#   task   = boolq   the most troublesome task. It is the ONLY task whose dataset
#                    is reached through a fallback: datasets>=4 dropped script
#                    loading and this cache is split across two repo names --
#                    `boolq` under `aps/super_glue`, `cb` under `super_glue`.
#                    ⚠ Its dataset was NEVER cached by 02_download_cache.sh until
#                    2026-08-14 (only the METRIC was), so this cell is the one
#                    that proves the offline cache is actually complete. It is
#                    also the largest SuperGLUE split (9,427 train / 3,270 eval).
#
#   method = zero3   the most troublesome method. It is the only arm that hands
#                    the step to a DeepSpeed ENGINE -- `engine.backward()` /
#                    `engine.step()` instead of `loss.backward()` / `opt.step()`
#                    (profile_hyclora.step:1559) -- so it exercises a code path no
#                    other arm touches, it initialises a process group on a compute
#                    node, and it was the slowest and heaviest arm in the dev-box
#                    table (768 ms/step, 6792 MiB peak). If anything hangs or OOMs
#                    on fir, it hangs here first.
#
# Between them this one job exercises: the fir venv, the temp/ clones, an OFFLINE
# dataset load through the fallback, the SuperGLUE metric, a non-standard backward,
# the engagement gate, and the lock-protected CSV write. That is the whole path.
#
# ⚠ DO NOT SUBMIT THE FULL SWEEP UNTIL THIS PRINTS `PILOT OK`.
# ============================================================================
set -uo pipefail

FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_pilot_cell "$@"        # full transcript -> ./logs/fir_pilot_cell_<UTC>.log
mkdir -p logs results/production

LOCAL=false
[ "${1:-}" = "--local" ] && LOCAL=true

# The cell. These are the CAMERA-READY values, not smoke values -- the point is a
# usable wall-clock estimate for the other cells.
P_METHOD="${P_METHOD:-zero3}"
P_TASK="${P_TASK:-boolq}"
P_SEED="${P_SEED:-41}"
P_EPOCHS="${P_EPOCHS:-3}"
# ⚠ EMPTY BY DEFAULT ON PURPOSE: let `run_production.resolve_seq` apply the measured per-task
#   policy (128 everywhere, 384 for boolq, because at 128 boolq truncates 61.4% of its examples).
#   Set P_SEQ only to override it deliberately; the row records `seq_source` either way.
P_SEQ="${P_SEQ:-}"
P_BATCH="${P_BATCH:-16}"
P_LR="${P_LR:-2e-4}"
P_CSV="${P_CSV:-results/production/camera_ready_glue.csv}"

run_pilot() {
    fir_load_modules_gpu || return 1
    # shellcheck disable=SC1091
    source "$FIR_VENV/bin/activate" || return 1
    fir_export_offline
    fir_assert_env gpu || return 1

    # ⚠ IDENTIFY THE NODE. Job 54759272 failed at CUDA context creation on a card that reported
    #   itself free, and the log could not say WHICH node it was -- so a rerun could not tell a bad
    #   node from a systematic fault. Print it.
    echo "node: $(hostname)   SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}  NODELIST=${SLURM_JOB_NODELIST:-<none>}"
    echo; nvidia-smi; echo

    # A CUDA-context probe in its OWN process, immediately before the run. `fir_assert_env gpu`
    # already creates a context, but it did so 25 s before 54759272 failed; this narrows the window
    # to ~1 s. If this PASSES and the run then fails the same way, something is taking the GPU
    # between two processes on a card we were allocated exclusively -- which is the finding, not
    # a detail.
    echo "--- CUDA context probe (t-1s) ---"
    python - <<'PROBE'
import torch
torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info(0)
print(f"  context OK on {torch.cuda.get_device_name(0)}: "
      f"free={free/2**20:.0f} MiB / total={total/2**20:.0f} MiB", flush=True)
PROBE
    echo "  probe exit=$?"

    echo "--- pilot cell: $P_METHOD on glue:$P_TASK, seed $P_SEED, ${P_EPOCHS} epochs ---"
    local t0 rc=0
    t0=$(date +%s)
    python src/run_production.py \
        --method "$P_METHOD" --task "glue:$P_TASK" --seed "$P_SEED" \
        --epochs "$P_EPOCHS" ${P_SEQ:+--seq "$P_SEQ"} --batch "$P_BATCH" --lr "$P_LR" \
        --device cuda:0 --warmup_steps 8 \
        --run_id camera_ready_pilot --out_csv "$P_CSV" || rc=1
    echo "--- cell wall-clock: $(( $(date +%s) - t0 ))s ---"
    [ $rc -eq 0 ] || { echo "PILOT FAILED: the run itself errored (above)"; return 1; }

    echo; echo "--- anomaly gate on the row it wrote ---"
    P_CSV="$P_CSV" P_METHOD="$P_METHOD" P_TASK="$P_TASK" python - <<'PY' || rc=1
import os, sys
sys.path.insert(0, "src")
import pandas as pd
from validate_glue_runner import check_row

csv, method, task = os.environ["P_CSV"], os.environ["P_METHOD"], os.environ["P_TASK"]
d = pd.read_csv(csv)
d = d[(d.method == method) & (d.task == f"glue:{task}")]
if not len(d):
    print("  NO ROW WRITTEN -- the run reported success but the CSV has no cell.")
    sys.exit(1)
r = d.iloc[-1].to_dict()
bad = check_row(r)
for k in ("task_metric_name", "task_metric", "epochs", "steps_per_epoch", "best_epoch",
          "train_step_peak_alloc_mib", "resident_floor_mib", "ms_per_step_median",
          "seq_len", "seq_source", "n_train_examples", "n_eval_examples", "dataset_source",
          "pred_distribution",
          "engagement_ok", "flce"):
    print(f"  {k:28} {r.get(k)}")
if bad:
    print("\n  ANOMALIES:")
    for b in bad:
        print(f"    - {b}")
    print("  PILOT FAILED")
    sys.exit(1)
print("\n  PILOT OK -- the cell is clean. Size the sweep from the wall-clock above.")
PY
    return $rc
}

if $LOCAL; then
    run_pilot
    exit $?
fi

jid=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=fir_pilot_cell
#SBATCH --output=./logs/fir_pilot_cell_%j.out
#SBATCH --error=./logs/fir_pilot_cell_%j.err
#SBATCH --time=2:00:00
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --cpus-per-task=4
#SBATCH --account=$FIR_ACCOUNT_GPU

cd "\$SLURM_SUBMIT_DIR"
source sbatch/fir/fir_env.sh
export P_METHOD=$P_METHOD P_TASK=$P_TASK P_SEED=$P_SEED P_EPOCHS=$P_EPOCHS
export P_SEQ="$P_SEQ" P_BATCH=$P_BATCH P_LR=$P_LR P_CSV=$P_CSV
$(declare -f run_pilot)
run_pilot
EOF
)
echo "submitted pilot cell job $jid"
echo "  cell:   $P_METHOD x glue:$P_TASK, seed $P_SEED, $P_EPOCHS epochs, seq ${P_SEQ:-<task policy>} batch $P_BATCH"
echo "  watch:  tail -f logs/fir_pilot_cell_${jid}.out"
echo "  ⚠ DO NOT submit the full sweep until this prints 'PILOT OK'."
