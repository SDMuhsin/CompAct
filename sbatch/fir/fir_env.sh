#!/bin/bash
# ============================================================================
# fir_env.sh — THE SINGLE SOURCE OF TRUTH for the fir environment.
# ============================================================================
# Sourced by every other script in this directory. Nothing else defines a module
# line, an offline flag, or a path. If a value is wrong, it is wrong in exactly
# one place.
#
#   source sbatch/fir/fir_env.sh          # sets vars, does NOT load modules
#   fir_load_modules_gpu                  # compute-node module set
#   fir_load_modules_cpu                  # login/CPU set (no cuda/cudnn)
#   fir_export_offline                    # the offline env, compute nodes
#   fir_export_online                     # the online env, login node downloads
#   fir_assert_env                        # FAIL FAST — run before any real work
#
# ⚠ EVERY VALUE MARKED `PROBE:` IS CARRIED OVER FROM RORQUAL AND IS UNVERIFIED ON
#   FIR. Run `sbatch/fir/00_probe_fir.sh` and correct them here BEFORE submitting
#   anything. They are deliberately NOT guessed a second time somewhere else.
# ============================================================================

# ---------------------------------------------------------------------------
# Module set. ⚠ ORDER IS LOAD-BEARING — do not "tidy" it.
#   gcc          toolchain first; arrow/scipy-stack are built against it
#   arrow        pyarrow for `datasets`
#   scipy-stack  supplies NUMPY. The venv is built with --system-site-packages,
#                so WITHOUT this `import datasets` dies with
#                "ModuleNotFoundError: No module named 'numpy'"
#   cuda cudnn   GPU only; omitted on the download/login path on purpose
# PROBE: confirm these names/versions exist on fir and that this order works.
# ---------------------------------------------------------------------------
FIR_MODULES_CPU="${FIR_MODULES_CPU:-gcc arrow scipy-stack}"
FIR_MODULES_GPU="${FIR_MODULES_GPU:-gcc arrow scipy-stack cuda cudnn}"

# PROBE: rorqual used these. fir's GPU names and MIG slice names may differ.
FIR_GPU_FULL="${FIR_GPU_FULL:-h100:1}"
FIR_GPU_MIG="${FIR_GPU_MIG:-h100_3g.40gb:1}"
FIR_ACCOUNT="${FIR_ACCOUNT:-def-seokbum}"        # PROBE: confirm on fir

# Repo-relative paths. Every script runs from the repo root.
FIR_VENV="${FIR_VENV:-./env}"
FIR_DATA="${FIR_DATA:-$(pwd)/data}"
FIR_DS_ALST="${FIR_DS_ALST:-$(pwd)/temp/ds_alst}"   # DeepSpeed>=0.17, ALST only

fir_load_modules_cpu() { module load $FIR_MODULES_CPU; }
fir_load_modules_gpu() { module load $FIR_MODULES_GPU; }

fir_export_online() {
    export HF_HOME="$FIR_DATA"
    export TORCH_HOME="$FIR_DATA"
    export HF_HUB_DISABLE_XET=1      # xet backend has caused stalled/partial pulls
    mkdir -p "$HF_HOME"
}

fir_export_offline() {
    export HF_HOME="$FIR_DATA"
    export TORCH_HOME="$FIR_DATA"
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    # ⚠ MANDATORY AND NOT REDUNDANT: `evaluate` ignores HF_HUB_OFFLINE. Without
    # this, evaluate.load() probes the Hub from a compute node with no route and
    # stalls ~44 MINUTES PER SEED before timing out. This one line has cost this
    # project more compute than any other.
    export HF_EVALUATE_OFFLINE=1
    # Long, memory-tight runs fragment the caching allocator.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
    mkdir -p "$HF_HOME"
}

# ---------------------------------------------------------------------------
# fir_assert_env — fail on the LOGIN NODE, not 40 minutes into a GPU allocation.
# Every check here corresponds to a failure mode that has actually happened.
# ---------------------------------------------------------------------------
fir_assert_env() {
    local want_gpu="${1:-gpu}" rc=0
    echo "--- fir_assert_env ($want_gpu) ---"

    [ -d ./src ] || { echo "FAIL: not in the repo root (no ./src)"; return 1; }
    [ -x "$FIR_VENV/bin/python" ] || { echo "FAIL: no venv at $FIR_VENV — run 01_setup_venv.sh"; return 1; }

    "$FIR_VENV/bin/python" - <<'PY' || rc=1
import importlib, sys
need = ["numpy", "torch", "transformers", "peft", "datasets", "accelerate",
        "filelock", "pandas", "evaluate"]
bad = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m} ({type(e).__name__})")
print("  core imports:", "OK" if not bad else "MISSING -> " + ", ".join(bad))
if bad:
    sys.exit(1)
import transformers
if transformers.__version__ != "4.51.3":
    print(f"  ⚠ transformers is {transformers.__version__}, the measured stack is 4.51.3.")
    print("    Every published number in this repo is on 4.51.3; a different version is a")
    print("    DIFFERENT EXPERIMENT, not a detail. Pin it or record the change deliberately.")
PY

    if [ "$want_gpu" = "gpu" ]; then
        "$FIR_VENV/bin/python" - <<'PY' || rc=1
import torch, sys
ok = torch.cuda.is_available()
print(f"  torch {torch.__version__} cuda={torch.version.cuda} available={ok}")
if not ok:
    print("  FAIL: no CUDA device visible")
    sys.exit(1)
print(f"  device: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
PY
    fi

    # Offline correctness: the cache must satisfy a load with the Hub blocked.
    ( fir_export_offline
      "$FIR_VENV/bin/python" - <<'PY' || exit 1
import sys
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("  offline model cache: OK")
except Exception as e:
    print(f"  FAIL offline model cache: {type(e).__name__}: {str(e)[:150]}")
    print("    -> run 02_download_cache.sh on a LOGIN node first")
    sys.exit(1)
PY
    ) || rc=1

    [ $rc -eq 0 ] && echo "--- fir_assert_env PASSED ---" || echo "--- fir_assert_env FAILED ---"
    return $rc
}
