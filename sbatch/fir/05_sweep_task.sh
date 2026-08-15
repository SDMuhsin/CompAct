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
# ⚠ ONE CELL IS EXPECTED TO REFUSE, AND THAT IS THE CORRECT OUTCOME.
#   ALST's own rule is `num_shards = ceil(seq / hidden)`. On a 2048-hidden model at
#   any GLUE sequence length (128, or 384 for boolq) that is 1, i.e. NO TILING: their
#   method configures itself to do nothing. `_eng_alst` reports `alst_tiling_active: 0`,
#   the `_active` liveness rule fires, and `verify_engagement` refuses to write a row
#   that would claim a method which did not run. The manifest marks these cells
#   `expect=refusal` and the report counts them separately from breakage. **Do not
#   "fix" this by passing --allow_inert** -- that writes a row claiming ALST ran.
#   ALST is a long-sequence method and this task is not where it has anything to say;
#   report it as inapplicable at this shape, with their rule as the reason.
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
P_TASK="${P_TASK:-boolq}"
P_SEEDS="${P_SEEDS:-41 42 43}"
P_EPOCHS="${P_EPOCHS:-3}"
# ⚠ EMPTY ON PURPOSE: `run_production.resolve_seq` applies the measured per-task policy
#   (128 everywhere, 384 for boolq, because at 128 boolq truncates 61.4% of its examples).
#   The row records `seq_source` either way, so an override is always visible in the data.
P_SEQ="${P_SEQ:-}"
P_BATCH="${P_BATCH:-16}"
P_LR="${P_LR:-2e-4}"
P_RUN_ID="${P_RUN_ID:-camera_ready_${P_TASK}}"
P_CSV="${P_CSV:-results/production/camera_ready_glue.csv}"

# Per-cell wall time. The pilot's heaviest arm (zero3, the only DeepSpeed-engine arm) took
# 1243 s for 3 epochs of boolq -- the largest SuperGLUE split. 3 h is ~8.7x that, which covers
# the full-FT arms training 1.1B parameters instead of 16M without a per-arm table.
P_TIME="${P_TIME:-3:00:00}"
# Array throttle. A shared cluster and a shared CSV lock; 8 concurrent is plenty to finish
# 39 cells in about 3 waves and is polite about the account's GPU share.
P_THROTTLE="${P_THROTTLE:-8}"

MANIFEST="results/production/_sweep/${P_RUN_ID}_manifest.tsv"

# ---------------------------------------------------------------------------
# Build the manifest: one line per cell that still needs running.
# ---------------------------------------------------------------------------
# ⚠ `$FIR_VENV/bin/python`, NOT `python`. This runs on the LOGIN NODE at submit time, where no
#   venv is active and `python` may not exist at all -- the interpreter has to be named explicitly.
#   It also needs pandas, which lives in the venv and not in the bare module stack.
build_manifest() {
    P_TASK="$P_TASK" P_SEEDS="$P_SEEDS" P_CSV="$P_CSV" P_SEQ="$P_SEQ" \
    MANIFEST="$MANIFEST" "$FIR_VENV/bin/python" - <<'PY'
import os, sys, math
sys.path.insert(0, "src")
from validate_glue_runner import arms, check_row
from experiment_registry import FULL_FT_REGIME, LOSSY
from run_production import TASK_MAX_LENGTH, GLUE_DEFAULT_SEQ

task     = os.environ["P_TASK"]
seeds    = [int(s) for s in os.environ["P_SEEDS"].split()]
csv_path = os.environ["P_CSV"]
manifest = os.environ["MANIFEST"]
seq      = int(os.environ["P_SEQ"]) if os.environ.get("P_SEQ") else \
           TASK_MAX_LENGTH.get(task, GLUE_DEFAULT_SEQ)

# Which cells are ALREADY clean? A row that exists but fails the anomaly gate is NOT clean --
# it must be re-run, because a bad row in a camera-ready CSV is worse than a missing one.
done_cells, dirty = set(), 0
if os.path.exists(csv_path):
    import pandas as pd
    d = pd.read_csv(csv_path)
    d = d[d.task == f"glue:{task}"]
    for _, r in d.iterrows():
        r = r.to_dict()
        key = (str(r.get("method")), int(r.get("with_fb") or 0), int(r.get("seed")))
        if check_row(r):
            dirty += 1                      # leave it in the CSV; the upsert will replace it
        else:
            done_cells.add(key)

# ALST's own rule: num_shards = ceil(seq/hidden). TinyLlama hidden = 2048. shards==1 -> no tiling
# -> `alst_tiling_active: 0` -> verify_engagement refuses the row. Mark it, do not hide it.
HIDDEN = 2048
alst_inapplicable = math.ceil(seq / HIDDEN) <= 1

lines, skipped = [], 0
for method, fb in arms():
    for seed in seeds:
        if (method, int(fb), seed) in done_cells:
            skipped += 1
            continue
        regime = "full_ft" if method in FULL_FT_REGIME else "peft_lora"
        expect = "refusal" if (method == "alst" and alst_inapplicable) else "ok"
        note   = "LOSSY" if method in LOSSY else "-"
        lines.append(f"{method}\t{int(fb)}\t{seed}\t{regime}\t{expect}\t{note}")

with open(manifest, "w") as f:
    f.write("\n".join(lines) + ("\n" if lines else ""))

print(f"  task           glue:{task}   seq={seq} (hidden={HIDDEN})")
print(f"  seeds          {seeds}")
print(f"  cells total    {len(arms()) * len(seeds)}")
print(f"  already clean  {len(done_cells)} (skipped)")
print(f"  dirty rows     {dirty} (will be re-run and upserted)" if dirty else "  dirty rows     0")
print(f"  to run         {len(lines)}")
if alst_inapplicable:
    print(f"  ⚠ alst        ceil({seq}/{HIDDEN})=1 shards -> NO TILING; its cells are expect=refusal")
print(f"  manifest       {manifest}")
PY
}

# ---------------------------------------------------------------------------
# Run ONE cell. Exported into the array job verbatim.
# ---------------------------------------------------------------------------
run_cell() {
    local method="$1" fb="$2" seed="$3" expect="$4"
    local fbflag=""; [ "$fb" = "1" ] && fbflag="--fb"
    local tag="${method}$([ "$fb" = "1" ] && echo "_fb")"

    fir_load_modules_gpu || return 1
    # shellcheck disable=SC1091
    source "$FIR_VENV/bin/activate" || return 1
    fir_export_offline

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

    echo "--- cell: $tag x glue:$P_TASK seed=$seed epochs=$P_EPOCHS expect=$expect ---"
    local t0 rc=0
    t0=$(date +%s)
    # ⚠ NO --allow_inert, EVER. It writes a row for a method that proved no work, which is the
    #   one failure this whole registry exists to prevent.
    python src/run_production.py \
        --method "$method" $fbflag --task "glue:$P_TASK" --seed "$seed" \
        --model "$P_MODEL" --epochs "$P_EPOCHS" ${P_SEQ:+--seq "$P_SEQ"} \
        --batch "$P_BATCH" --lr "$P_LR" --device cuda:0 --warmup_steps 8 \
        --run_id "$P_RUN_ID" --out_csv "$P_CSV" || rc=1
    echo "--- $tag seed=$seed wall-clock: $(( $(date +%s) - t0 ))s  rc=$rc ---"

    if [ "$expect" = "refusal" ]; then
        # An engagement refusal is the CORRECT outcome here and must not look like a green cell.
        [ $rc -ne 0 ] \
            && { echo "EXPECTED REFUSAL -- $tag has nothing to do at this sequence length. OK."; return 0; } \
            || { echo "⚠ $tag was expected to refuse and did NOT. Check its liveness counters."; return 1; }
    fi
    return $rc
}

# ---------------------------------------------------------------------------
# The camera-ready report: apply the anomaly gate to every cell of the grid.
# ---------------------------------------------------------------------------
# Named interpreter for the same reason as `build_manifest`: this is invoked both from the login
# node (--report) and from a CPU-only report job, neither of which has an activated venv.
run_report() {
    P_TASK="$P_TASK" P_SEEDS="$P_SEEDS" P_CSV="$P_CSV" P_SEQ="$P_SEQ" \
    "$FIR_VENV/bin/python" - <<'PY'
import os, sys, math, json
sys.path.insert(0, "src")
import pandas as pd
from validate_glue_runner import arms, check_row
from experiment_registry import FULL_FT_REGIME
from run_production import TASK_MAX_LENGTH, GLUE_DEFAULT_SEQ

task  = os.environ["P_TASK"]
seeds = [int(s) for s in os.environ["P_SEEDS"].split()]
csv_path = os.environ["P_CSV"]
seq = int(os.environ["P_SEQ"]) if os.environ.get("P_SEQ") else \
      TASK_MAX_LENGTH.get(task, GLUE_DEFAULT_SEQ)
alst_inapplicable = math.ceil(seq / 2048) <= 1

if not os.path.exists(csv_path):
    print(f"  no CSV at {csv_path}"); sys.exit(1)
d = pd.read_csv(csv_path)
d = d[d.task == f"glue:{task}"]

rows, n_ok, n_bad, n_missing, n_refused = [], 0, 0, 0, 0
for method, fb in arms():
    tag = method + ("_fb" if fb else "")
    regime = "full_ft" if method in FULL_FT_REGIME else "peft_lora"
    expect_refusal = (method == "alst" and alst_inapplicable)
    for seed in seeds:
        c = d[(d.method == method) & (d.with_fb == int(fb)) & (d.seed == seed)]
        if not len(c):
            if expect_refusal:
                status, n_refused = "REFUSED (by their own rule)", n_refused + 1
            else:
                status, n_missing = "MISSING", n_missing + 1
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
    print(f"\n=== {regime} — glue:{task}, seq {seq} ===")
    print(f"  {'arm':14} {'seed':>4}  {'metric':>8} {'peak MiB':>10} {'ms/step':>9}  status")
    for _, tag, seed, status, m, p, ms in sel:
        f = lambda v, w, d=4: (f"{float(v):{w}.{d}f}" if v is not None and v == v else " " * w)
        print(f"  {tag:14} {seed:>4}  {f(m,8)} {f(p,10,2)} {f(ms,9,2)}  {status}")

print(f"\n  ok {n_ok} | anomalies {n_bad} | missing {n_missing} | refused-by-design {n_refused}")
# Refusals are a correct outcome and do NOT fail the sweep. Anomalies and gaps do.
if n_bad or n_missing:
    print("  SWEEP INCOMPLETE -- re-run this script to fill gaps; investigate anomalies first.")
    sys.exit(1)
print("  SWEEP CLEAN -- every permitted cell is present and passes the anomaly gate.")
PY
}

# ---------------------------------------------------------------------------
echo "=== camera-ready sweep: all baselines x glue:$P_TASK ==="
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
    echo "  --- manifest (method / fb / seed / regime / expect / note) ---"
    nl -ba "$MANIFEST" | sed 's/^/    /'
    echo
    echo "  would submit: array 0-$((N-1))%$P_THROTTLE, ${P_TIME} per cell"
    exit 0
fi

if [ "$MODE" = "local" ]; then
    rc=0
    while IFS=$'\t' read -r method fb seed regime expect note; do
        run_cell "$method" "$fb" "$seed" "$expect" || rc=1
    done < "$MANIFEST"
    echo; run_report || rc=1
    exit $rc
fi

# --- fail on the LOGIN node, not 39 times on compute nodes -------------------
fir_assert_env cpu || { echo "environment is not sane -- refusing to submit $N cells"; exit 1; }

jid=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=fir_sweep_${P_TASK}
#SBATCH --output=./logs/fir_sweep_${P_TASK}_%A_%a.out
#SBATCH --error=./logs/fir_sweep_${P_TASK}_%A_%a.err
#SBATCH --time=$P_TIME
#SBATCH --gpus=$FIR_GPU_FULL
#SBATCH --mem=$FIR_GPU_MEM
#SBATCH --cpus-per-task=4
#SBATCH --account=$FIR_ACCOUNT_GPU
#SBATCH --array=0-$((N-1))%$P_THROTTLE

cd "\$SLURM_SUBMIT_DIR"
source sbatch/fir/fir_env.sh
export P_MODEL="$P_MODEL" P_TASK=$P_TASK P_EPOCHS=$P_EPOCHS P_SEQ="$P_SEQ"
export P_BATCH=$P_BATCH P_LR=$P_LR P_RUN_ID=$P_RUN_ID P_CSV=$P_CSV
line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
IFS=\$'\t' read -r method fb seed regime expect note <<< "\$line"
echo "array task \$SLURM_ARRAY_TASK_ID -> \$method fb=\$fb seed=\$seed (\$regime, expect=\$expect)"
$(declare -f run_cell)
run_cell "\$method" "\$fb" "\$seed" "\$expect"
EOF
)
echo "submitted sweep array $jid  ($N cells, throttle %$P_THROTTLE, $P_TIME each)"

# The report runs after the LAST cell finishes, whatever its exit status -- a sweep with a
# failed cell is exactly when the summary is most useful. `afterany` on the array as a whole.
rid=$(sbatch --parsable --dependency=afterany:$jid <<EOF
#!/bin/bash
#SBATCH --job-name=fir_sweep_${P_TASK}_report
#SBATCH --output=./logs/fir_sweep_${P_TASK}_report_%j.out
#SBATCH --error=./logs/fir_sweep_${P_TASK}_report_%j.err
#SBATCH --time=0:20:00
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=2
#SBATCH --account=$FIR_ACCOUNT_CPU

cd "\$SLURM_SUBMIT_DIR"
source sbatch/fir/fir_env.sh
fir_load_modules_cpu
export P_TASK=$P_TASK P_SEEDS="$P_SEEDS" P_CSV=$P_CSV P_SEQ="$P_SEQ"
$(declare -f run_report)
run_report
EOF
)
echo "submitted report job $rid (runs after the array, whatever it exits with)"
echo
echo "  watch:   squeue -j $jid"
echo "  cells:   tail -f logs/fir_sweep_${P_TASK}_${jid}_0.out"
echo "  report:  cat logs/fir_sweep_${P_TASK}_report_${rid}.out"
echo "  resume:  re-run this script -- clean cells are skipped, gaps are refilled"
