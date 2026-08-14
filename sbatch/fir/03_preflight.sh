#!/bin/bash
# ============================================================================
# 03_preflight.sh — ONE tiny GPU job that proves the stack works on a fir COMPUTE
# node before the camera-ready sweep is submitted.
# ============================================================================
#   bash sbatch/fir/03_preflight.sh              # submit
#   bash sbatch/fir/03_preflight.sh --local      # run here (inside salloc)
#
# WHY THIS EXISTS
# ---------------
# Everything up to here was validated on a LOGIN node. A compute node differs in
# the ways that have historically cost this project the most: no route to the
# internet, a different visible GPU, and the offline env actually enforced. This
# job costs ~10 GPU-minutes and answers "would the sweep have worked?" before the
# sweep is queued. It is deliberately the SMALLEST thing that exercises the whole
# path: build a method, verify engagement, train, evaluate, write a locked CSV row.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
source sbatch/fir/fir_env.sh
mkdir -p logs results/production

LOCAL=false
[ "${1:-}" = "--local" ] && LOCAL=true

# The preflight matrix: cheap, and between them they touch every risky mechanism.
#   baseline      plain LoRA + gc            — the control
#   baseline --fb our fused block            — Triton kernels compile on this GPU
#   qlora         bitsandbytes 4-bit          — the lossy path + a non-torch dep
#   lomo --fb     full-FT + fused composition — a custom backward AND a composition
PREFLIGHT_ARMS=("baseline:" "baseline:--fb" "qlora:" "lomo:--fb")

run_preflight() {
    fir_load_modules_gpu || return 1
    # shellcheck disable=SC1091
    source "$FIR_VENV/bin/activate" || return 1
    fir_export_offline
    fir_assert_env gpu || return 1

    echo; nvidia-smi; echo
    local csv="results/production/fir_preflight.csv" rc=0
    for spec in "${PREFLIGHT_ARMS[@]}"; do
        local m="${spec%%:*}" fb="${spec#*:}"
        echo "--- preflight: $m ${fb:-(no fb)} ---"
        python src/run_production.py --method "$m" $fb \
            --device cuda:0 --seq 512 --batch 2 --train_steps 4 --warmup_steps 4 \
            --flce --run_id fir_preflight --out_csv "$csv" || { echo "  FAILED: $m $fb"; rc=1; }
    done

    echo; echo "--- CSV integrity ---"
    python - <<'PY' || rc=1
import pandas as pd, os, sys
# ⚠ REPORT THE REAL FAILURE, NOT A TRACEBACK ON TOP OF IT. When all four arms died
# (job 54306984) the CSV was never created and this block ended the log with a
# 16-line pandas FileNotFoundError — burying the three ModuleNotFoundErrors that
# actually explained the job. The last thing in a log is what gets read first.
p = "results/production/fir_preflight.csv"
if not os.path.exists(p):
    print(f"  NO CSV AT {p} — every arm failed before writing a row.")
    print("  PREFLIGHT FAILED: read the per-arm tracebacks ABOVE, not this line.")
    sys.exit(1)
d = pd.read_csv(p)
dups = d.duplicated(subset=["method","with_fb","task","seed","seq_len","batch_size","lr"]).sum()
print(f"  rows={len(d)} cols={len(d.columns)} dup_comb_keys={dups} all_engaged={int(d.engagement_ok.all())}")
print(d[["method","with_fb","engagement_ok","train_step_peak_alloc_mib","resident_floor_mib"]].to_string(index=False))
if dups or not d.engagement_ok.all() or len(d) == 0:
    print("  PREFLIGHT FAILED"); sys.exit(1)
print("  PREFLIGHT OK — safe to submit the sweep")
PY
    return $rc
}

if $LOCAL; then
    run_preflight
    exit $?
fi

jid=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=fir_preflight
#SBATCH --output=./logs/fir_preflight_%j.out
#SBATCH --error=./logs/fir_preflight_%j.err
#SBATCH --time=0:40:00
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --cpus-per-task=4
#SBATCH --account=$FIR_ACCOUNT_GPU

cd "\$SLURM_SUBMIT_DIR"
source sbatch/fir/fir_env.sh
$(declare -f run_preflight)
PREFLIGHT_ARMS=(${PREFLIGHT_ARMS[*]@Q})
run_preflight
EOF
)
echo "submitted preflight job $jid"
echo "  watch:  tail -f logs/fir_preflight_${jid}.out"
echo "  ⚠ DO NOT submit 10_submit_camera_ready.sh until this job prints 'PREFLIGHT OK'."
