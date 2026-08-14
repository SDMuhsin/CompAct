#!/bin/bash
# ============================================================================
# 02_download_cache.sh — pre-download every model, dataset and metric.
# RUN ON A FIR LOGIN NODE (compute nodes have NO route to the internet).
# ============================================================================
#   bash sbatch/fir/02_download_cache.sh 2>&1 | tee logs/fir_download.txt
#
# Derived from sbatch/download_cache.sh (rorqual) with three changes, all forced:
#   1. paths/modules come from fir_env.sh — the cache lands on /scratch via the
#      ./data symlink, never on /project (inode budget, see fir_env.sh)
#   2. adds MMLU (`cais/mmlu`, config "all"), which the commonsense-MC paradigm
#      now supports as an extra eval set
#   3. SuperGLUE is NOT downloaded — dropped from scope on 2026-08-11
#
# ⚠ THIS MUST COMPLETE BEFORE ANY JOB IS SUBMITTED. A compute node with
#   HF_HUB_OFFLINE=1 and a cold cache does not fail fast: `evaluate.load()` alone
#   stalls ~44 minutes per seed probing a Hub it cannot reach.
# ============================================================================
set -uo pipefail

# FIR_SELF must be resolved BEFORE the cd: $0 is relative to the invocation
# directory, and fir_log_to re-execs this script from the repo root.
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_download_cache "$@"        # full transcript -> ./logs/fir_download_cache_<UTC>.log

echo "############ fir download cache — $(date -u +%FT%TZ) ############"

# CPU module set on purpose: no GPU is needed to download, and cuda/cudnn only
# slow the environment down here. (Same choice rorqual's download_cache.sh makes.)
fir_load_modules_cpu || { echo "FAIL: module load '$FIR_MODULES_CPU'"; exit 1; }
fir_link_scratch || exit 1
# shellcheck disable=SC1091
if [ ! -f "$FIR_VENV/bin/activate" ]; then
    echo "FAIL: no usable venv at $(readlink -f "$FIR_VENV" 2>/dev/null || echo "$FIR_VENV")"
    echo "  bin/python:   $([ -x "$FIR_VENV/bin/python" ] && echo present || echo missing)"
    echo "  bin/activate: missing"
    echo "  -> run:  bash sbatch/fir/01_setup_venv.sh"
    echo "     (if bin/python is present but activate is missing, the venv is HALF-BUILT from an"
    echo "      interrupted create; 01_setup_venv.sh now detects that and rebuilds automatically.)"
    exit 1
fi
# shellcheck disable=SC1091
source "$FIR_VENV/bin/activate" || exit 1
fir_export_online                       # ONLINE: no *_OFFLINE flags here

echo "cache -> $(readlink -f "$FIR_DATA")"
df -h "$(readlink -f "$FIR_DATA")" 2>/dev/null | tail -1

python - <<'PY' || exit 1
import time, sys
from huggingface_hub import snapshot_download
from datasets import load_dataset
import evaluate

def retry(fn, label, n=5):
    """Multi-shard pulls hit transient ChunkedEncodingError / IncompleteRead. Each
    attempt RESUMES from already-cached shards, so retrying makes progress rather
    than restarting. Without this one hiccup aborts the whole prep."""
    for a in range(1, n + 1):
        try:
            print(f"[{a}/{n}] {label} ...", flush=True)
            out = fn()
            print(f"   OK  {label}", flush=True)
            return out
        except Exception as e:
            print(f"   attempt {a} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            if a == n:
                raise
            time.sleep(10 * a)

print("=== models ===")
for m in ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "huggyllama/llama-7b", "Qwen/Qwen2-1.5B"]:
    retry(lambda m=m: snapshot_download(repo_id=m), m)

print("\n=== GLUE ===")
for t in ["cola", "mrpc", "sst2", "rte", "qnli", "stsb"]:
    retry(lambda t=t: load_dataset("glue", t), f"glue/{t}")

print("\n=== language modelling ===")
retry(lambda: load_dataset("wikitext", "wikitext-2-raw-v1"), "wikitext-2")

print("\n=== commonsense: train mixture + the 8 eval sets ===")
retry(lambda: load_dataset("zwhe99/commonsense_170k"), "commonsense_170k (170,420 rows)")
for args, label in [
    (("google/boolq",), "boolq"),
    (("nthngdy/piqa",), "piqa (ybisk/piqa ships a script -> mirror)"),
    (("lighteval/siqa",), "siqa (social_i_qa ships a script -> mirror)"),
    (("Rowan/hellaswag",), "hellaswag"),
    (("allenai/winogrande", "winogrande_xl"), "winogrande"),
    (("allenai/ai2_arc", "ARC-Easy"), "arc_e"),
    (("allenai/ai2_arc", "ARC-Challenge"), "arc_c"),
    (("allenai/openbookqa", "main"), "obqa"),
]:
    retry(lambda a=args: load_dataset(*a), label)

print("\n=== MMLU (extra MC eval set) ===")
# ⚠ Reported here 0-shot through a trained MC head, NOT the literature's 5-shot
# prompted protocol. Comparable ACROSS our arms; not comparable with published
# MMLU numbers. The column is named `mmlu_0shot_mc` for that reason.
retry(lambda: load_dataset("cais/mmlu", "all"), "cais/mmlu all (14,042 test rows)")

print("\n=== metrics ===")
# `evaluate` ignores HF_HUB_OFFLINE at run time, so anything not cached here
# costs ~44 min per seed on a compute node. SuperGLUE is out of scope.
for m in ["glue", "accuracy", "f1", "matthews_correlation", "pearsonr", "spearmanr", "perplexity"]:
    try:
        retry(lambda m=m: evaluate.load(m), f"metric:{m}", n=3)
    except Exception as e:
        print(f"   ⚠ metric {m} unavailable: {type(e).__name__}")
PY

echo
echo "=== cache size ==="
du -sh "$(readlink -f "$FIR_DATA")" 2>/dev/null

echo
echo "############ verifying the cache satisfies an OFFLINE load ############"
# The real test: re-load with every offline flag set, exactly as a compute node
# will. Passing here is what makes it safe to submit jobs.
( fir_export_offline
  python - <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
import evaluate
fails = []
try:
    AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("  model+tokenizer offline: OK")
except Exception as e: fails.append(f"model: {type(e).__name__}: {str(e)[:120]}")
for args, label in [(("glue","rte"),"glue/rte"), (("cais/mmlu","all"),"mmlu"),
                    (("Rowan/hellaswag",),"hellaswag"), (("allenai/ai2_arc","ARC-Easy"),"arc_e")]:
    try:
        load_dataset(*args); print(f"  {label} offline: OK")
    except Exception as e: fails.append(f"{label}: {type(e).__name__}: {str(e)[:120]}")
try:
    evaluate.load("accuracy"); print("  evaluate/accuracy offline: OK")
except Exception as e: fails.append(f"evaluate: {type(e).__name__}: {str(e)[:120]}")
if fails:
    print("\n  OFFLINE VERIFICATION FAILED:"); [print("   -", f) for f in fails]
    sys.exit(1)
print("\n  ALL OFFLINE LOADS OK — safe to submit jobs")
PY
) || { echo "############ DO NOT SUBMIT JOBS YET ############"; exit 1; }

echo
echo "NEXT: bash sbatch/fir/03_preflight.sh   (one tiny GPU job before the sweep)"
