#!/bin/bash
# ============================================================================
# 01_setup_venv.sh — build the venv on a FIR LOGIN NODE. Internet required.
# ============================================================================
#   bash sbatch/fir/01_setup_venv.sh 2>&1 | tee logs/fir_setup_venv.txt
#
# Idempotent: re-running verifies and repairs rather than rebuilding, unless you
# pass --fresh.
#
# WHY THIS FIGHTS THE WHEELHOUSE
# ------------------------------
# `avail_wheels` on fir offers torch 2.13.0 / transformers 5.14.1 / datasets 5.0.0.
# Every number in CONTEXT.md was measured on torch 2.10.0+cu128 / transformers
# 4.51.3 / datasets 4.5.0. transformers 5.x is a MAJOR bump: the fused block
# patches LlamaDecoderLayer and LlamaRMSNorm internals, the eleven architecture
# guards key on 4.51.3 structure, and both the HyC-LoRA port and StreamBP import
# `transformers.modeling_flash_attention_utils._flash_attention_forward`. Taking
# the wheelhouse would not be an upgrade, it would be a different experiment whose
# numbers cannot be compared with anything already published in this repo.
#
# PyPI is reachable from the fir login node (probe §7: HTTP 200), so the pinned
# stack is installable. Each stage VERIFIES before the next begins, so a failure
# costs a login-node minute rather than a GPU allocation.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1          # repo root
source sbatch/fir/fir_env.sh

FRESH=false
BUILD_FA=false
for a in "$@"; do
    case "$a" in
        --fresh)            FRESH=true ;;
        --with-flash-attn)  BUILD_FA=true ;;   # attempt the ~35 min source build
        *) echo "unknown option: $a"; echo "usage: $0 [--fresh] [--with-flash-attn]"; exit 1 ;;
    esac
done

echo "############ fir venv setup — $(date -u +%FT%TZ) ############"
echo "repo (on /project): $(pwd)"

# --- 0. modules FIRST, in the load-bearing order (cudnn is only visible after cuda)
fir_load_modules_gpu || { echo "FAIL: module load '$FIR_MODULES_GPU'"; exit 1; }
echo "python: $(python -V 2>&1) at $(command -v python)"

# --- 1. scratch targets + symlinks (venv/data must NOT consume /project inodes)
echo; echo "--- linking venv/data onto /scratch ---"
fir_link_scratch || exit 1

# --- 2. create the venv
# ⚠ A HALF-BUILT VENV MUST BE DETECTED, NOT INHERITED.
# CPython's venv runs `_setup_pip` (ensurepip) BEFORE `setup_scripts` (which writes
# bin/activate), so a Ctrl-C during ensurepip — which is exactly what happens when
# the slow Lustre/CVMFS create looks hung — leaves `bin/python` present and
# `bin/activate` absent. Checking only for bin/python then "finds" a venv, skips
# creation, and every later step dies on `./env/bin/activate: No such file`.
# Observed on fir 2026-08-11. Validate by RUNNING the interpreter, not by stat.
venv_is_healthy() {
    [ -x "$FIR_VENV/bin/python" ] && [ -f "$FIR_VENV/bin/activate" ] \
        && "$FIR_VENV/bin/python" -c "import sys, ensurepip" >/dev/null 2>&1
}

if $FRESH && [ -d "$FIR_VENV_REAL" ]; then
    echo "--fresh: removing $FIR_VENV_REAL"; rm -rf "$FIR_VENV_REAL"
fi
if ! venv_is_healthy; then
    if [ -e "$FIR_VENV_REAL/bin" ]; then
        echo; echo "--- found a BROKEN/partial venv at $FIR_VENV_REAL — removing and rebuilding ---"
        echo "    (bin/python:  $([ -x "$FIR_VENV/bin/python" ] && echo present || echo missing))"
        echo "    (bin/activate:$([ -f "$FIR_VENV/bin/activate" ] && echo present || echo missing))"
        rm -rf "$FIR_VENV_REAL"
    fi
    mkdir -p "$FIR_VENV_REAL"
    echo; echo "--- creating venv at $FIR_VENV_REAL ---"
    echo "    NOTE: MEASURED on fir — virtualenv ~10s, python -m venv ~15s. Not minutes."
    echo "    If it sits far longer, something IS wrong. Re-running repairs a partial venv."
    # --system-site-packages is REQUIRED, not stylistic: numpy comes from the
    # scipy-stack module, and the venv is built against it. Without it
    # `import datasets` dies with "ModuleNotFoundError: No module named 'numpy'"
    # (the rorqual scripts carry the same note). Our pins still shadow the system copy.
    #
    # ⚠ CREATION IS A CASCADE, because both seeders fail in different ways on fir.
    #
    #   1. `virtualenv --no-download` is Alliance's recommended path and is the
    #      fastest, BUT its via_app_data seeder extracts bundled pip/setuptools
    #      wheels that live in CVMFS, and CVMFS returns EIO on a cold/transient
    #      read. Observed on fir 2026-08-11:
    #        RuntimeError: failed to build image setuptools, pip because:
    #        OSError: [Errno 5] Input/output error   (in zip_ref.extractall)
    #      That is a filesystem hiccup, not a broken environment, so it is worth
    #      one retry before giving up on the fast path.
    #   2. `python -m venv` runs ensurepip instead. It is SLOW here (1-3 min on
    #      Lustre) but it does not read CVMFS zips, so it survives exactly the
    #      failure above. This is the path that was already working before it was
    #      interrupted for looking hung.
    #
    # App-data is redirected onto /scratch so virtualenv's cache never touches
    # /home (48 GiB) or /project (inode-bound).
    export VIRTUALENV_OVERRIDE_APP_DATA="$FIR_SCRATCH_ROOT/.virtualenv_appdata"
    mkdir -p "$VIRTUALENV_OVERRIDE_APP_DATA"

    created=false
    if command -v virtualenv >/dev/null 2>&1; then
        for attempt in 1 2; do
            echo "    [$attempt/2] virtualenv --no-download --system-site-packages"
            if virtualenv --no-download --system-site-packages "$FIR_VENV_REAL"; then
                created=true; break
            fi
            echo "    virtualenv attempt $attempt failed (CVMFS EIO is the known cause)"
            rm -rf "$FIR_VENV_REAL" "$VIRTUALENV_OVERRIDE_APP_DATA"
            mkdir -p "$FIR_VENV_REAL" "$VIRTUALENV_OVERRIDE_APP_DATA"
            sleep 5
        done
    fi
    if ! $created; then
        echo "    falling back to: python -m venv --system-site-packages (~15s, measured)"
        echo "    (ensurepip instead of the CVMFS app-data zip extract — survives the EIO above)"
        rm -rf "$FIR_VENV_REAL"; mkdir -p "$FIR_VENV_REAL"
        python -m venv --system-site-packages "$FIR_VENV_REAL" || {
            echo "FAIL: both virtualenv and python -m venv failed to create $FIR_VENV_REAL"
            echo "  If this was another Errno 5, the CVMFS/Lustre mount is unhealthy right now —"
            echo "  wait a few minutes and re-run; nothing here is corrupted."
            exit 1; }
    fi
    venv_is_healthy || { echo "FAIL: venv still not healthy after creation"; exit 1; }
    echo "    venv created OK"
fi
# shellcheck disable=SC1091
source "$FIR_VENV/bin/activate" || exit 1
echo "venv python: $(python -V 2>&1) at $(command -v python)"

python -m pip install -q --upgrade pip setuptools wheel packaging ninja || exit 1

# ⚠ THE TORCH GUARD. A pip stage can silently DOWNGRADE torch by pulling a
# dependency that pins an older one — which is exactly what happened on fir
# 2026-08-11: flash-attn's Alliance wheel is `2.8.3+torch29.computecanada`, it
# requires torch 2.9.x, and installing it without --no-deps took torch from the
# pinned 2.10.0 down to 2.9.1. Every earlier stage had already verified and
# passed, so nothing complained until the preflight job failed on a compute node.
# Re-assert the pin after EVERY stage so the culprit is named at the moment it acts.
assert_torch_pin() {
    local where="$1"
    python - <<PY || { echo "FAIL: torch pin broken by: $where"; exit 1; }
import sys, torch
want, got = "${FIR_PIN_TORCH}", torch.__version__
if not got.startswith(want):
    print(f"  ⚠⚠ TORCH DOWNGRADED to {got} (pinned {want}) by: $where")
    print( "     Some dependency pinned an older torch. Re-install that package with")
    print( "     --no-deps, or constrain it. Continuing would measure a DIFFERENT stack.")
    sys.exit(1)
PY
}

stage() {   # stage <label> <import-check> -- <pip args...>
    local label="$1" check="$2"; shift 3
    echo; echo "--- $label ---"
    python -m pip install "$@" || { echo "FAIL: pip install for $label"; exit 1; }
    python -c "$check" || { echo "FAIL: post-install verification for $label"; exit 1; }
    assert_torch_pin "$label"
}

# --- 3. torch FIRST. Everything else compiles/links against it, and flash-attn
#        must see the final torch or it builds an incompatible extension.
#
# ⚠ NO --index-url. MEASURED on fir (00c_probe_deps.sh): Alliance's pip config
#   (/cvmfs/.../pip-x86-64-v4-gentoo2023.conf) adds `find-links` to the CVMFS
#   wheelhouse in [install]/[download]/[wheel] with `prefer-binary = true`, so the
#   LOCAL wheel wins no matter what index is passed — `--index-url ...cu128` and
#   even `--isolated` both still resolved to `torch-2.10.0+computecanada`. Passing
#   the PyTorch index is therefore pure noise; dropping it keeps the resolution
#   honest and uses the fast local wheels.
#
# ⚠ RECORD THIS DEVIATION: fir gives `2.10.0+computecanada`, the dev box measured
#   `2.10.0+cu128`. Same upstream version, DIFFERENT BUILD (Alliance compile it
#   against their own CUDA/toolchain). Memory numbers are allocator-sensitive, so
#   fir figures are comparable with each other but should not be quoted
#   interchangeably with CONTEXT.md's dev-box numbers without saying so.
stage "torch ${FIR_PIN_TORCH} (Alliance wheelhouse)" \
      "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)" -- \
      -q "torch==${FIR_PIN_TORCH}"

# --- 4. the pinned HF stack. Installed together so pip resolves them as a set
#        rather than letting a later package silently upgrade an earlier one.
stage "pinned HF stack" \
      "import transformers,datasets,peft,accelerate;print('  transformers',transformers.__version__,'datasets',datasets.__version__,'peft',peft.__version__,'accelerate',accelerate.__version__)" -- \
      -q "transformers==${FIR_PIN_TRANSFORMERS}" "datasets==${FIR_PIN_DATASETS}" \
         "peft==${FIR_PIN_PEFT}" "accelerate==${FIR_PIN_ACCELERATE}" \
         evaluate sentencepiece filelock pandas scikit-learn

# --- 5. per-method extras. Each is needed ONLY by its own arms, so a failure here
#        costs those arms and nothing else.
stage "bitsandbytes (qlora arms)" \
      "import bitsandbytes;print('  bitsandbytes',bitsandbytes.__version__)" -- \
      -q "bitsandbytes==${FIR_PIN_BNB}"

stage "deepspeed ${FIR_PIN_DEEPSPEED} (zero3 arms)" \
      "import deepspeed;print('  deepspeed',deepspeed.__version__)" -- \
      -q "deepspeed==${FIR_PIN_DEEPSPEED}"

stage "tensorly (galore arms — their own requirements.txt)" \
      "import tensorly;print('  tensorly',tensorly.__version__)" -- \
      -q --no-deps tensorly

# --- 6. DeepSpeed >= 0.17 SIDE-BY-SIDE for ALST only. Installed to a separate
#        prefix and put on PYTHONPATH for ALST arms alone, because upgrading in
#        place would silently change every zero3 number.
#
# ⚠⚠ ALWAYS APPEND TO PYTHONPATH, NEVER ASSIGN IT. On fir, numpy is supplied by the
#    `scipy-stack` MODULE, which puts it on PYTHONPATH — so `PYTHONPATH=/some/dir`
#    silently removes numpy from the interpreter and deepspeed dies with
#    `ModuleNotFoundError: No module named 'numpy'` from `deepspeed/utils/timer.py`.
#    Observed on fir 2026-08-11. This applies to the ALST ARM AT RUN TIME too:
#    use `PYTHONPATH=$FIR_DS_ALST:$PYTHONPATH`, not `PYTHONPATH=$FIR_DS_ALST`.
echo; echo "--- deepspeed ${FIR_PIN_DEEPSPEED_ALST} side-by-side -> $FIR_DS_ALST ---"
DS_BUILD_OPS=0 python -m pip install -q --no-deps --target "$FIR_DS_ALST" \
    "deepspeed==${FIR_PIN_DEEPSPEED_ALST}" \
  && PYTHONPATH="$FIR_DS_ALST:$PYTHONPATH" python -c "
import importlib.util as u, sys
sys.path.insert(0,'$FIR_DS_ALST')
ok = u.find_spec('deepspeed.runtime.sequence_parallel') is not None
print('  sequence_parallel present:', ok)
sys.exit(0 if ok else 1)" \
  || echo "  ⚠ ALST side-by-side deepspeed FAILED -> alst arms unavailable (nothing else affected)"

# --- 7. flash-attn LAST, and allowed to fail. This is the ONE genuinely risky
#        dependency: on the dev box no wheel matched torch 2.10+cu128 and it had
#        to be built from source (~35 min). fir HAS a cp311 wheel for 2.8.3, but
#        it is built against fir's torch, so pairing it with pinned torch 2.10 can
#        give the classic ABI break:
#           undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
#        Try the wheel, verify by RUNNING a kernel (importing is not enough), and
#        fall back to a source build. If both fail, ONLY the streambp arms are lost.
#
# ⚠ MEASURED on fir: the wheelhouse offers `flash_attn-2.8.3+torch29.computecanada`
#   — built against TORCH 2.9, while we pin torch 2.10. That is precisely the ABI
#   mismatch that produced `undefined symbol: _ZN3c104cuda29c10_cuda_check_...`
#   on the dev box. So the wheel is EXPECTED to fail here and the source build is
#   the likely path. This is why the check below RUNS a kernel instead of trusting
#   a successful import, and why failure costs only the streambp arms.
echo; echo "--- flash-attn (streambp arms only; failure is survivable) ---"
echo "    NOTE: fir's wheel is built for torch 2.9 (+torch29) against our pinned 2.10 —"
echo "    expect the wheel to fail its kernel check and the source build (~35 min) to run."
FA_OK=false
# ⚠ --no-deps IS MANDATORY HERE. fir's wheel is `flash_attn-2.8.3+torch29...`,
# which REQUIRES torch 2.9.x; without --no-deps pip happily downgrades our pinned
# torch 2.10.0 to satisfy it and every other package then sits on the wrong ABI.
# Observed on fir 2026-08-11: torch silently became 2.9.1 and peft/evaluate stopped
# importing. With --no-deps the wheel either works against our torch or fails its
# kernel check below and is removed — it can no longer damage the rest of the stack.
# ⚠ THE CHECK MUST NOT NEED A GPU. This script runs on a LOGIN NODE, which has
# none, so a `device='cuda'` test would fail for every candidate — condemning a
# perfectly good wheel and then burning 35 minutes on a source build whose check
# fails the same way. Importing the extension is the right test anyway: the ABI
# break we are guarding against (`undefined symbol: _ZN3c104cuda29c10_cuda_...`)
# surfaces when the .so is LOADED, not when a kernel runs. The actual kernel is
# exercised later by 03_preflight.sh, on a node that has a GPU.
fa_check() {
    python - <<'PY'
import sys
try:
    import flash_attn
    import flash_attn_2_cuda            # loads the compiled extension -> resolves symbols
    import torch
    print(f"  flash_attn {flash_attn.__version__} imports cleanly against torch {torch.__version__}")
except Exception as e:
    print(f"  flash_attn unusable: {type(e).__name__}: {str(e)[:200]}")
    sys.exit(1)
PY
}

if python -m pip install -q --no-deps flash-attn --no-build-isolation 2>&1 | tail -3; then :; fi
if fa_check; then
    FA_OK=true
elif $BUILD_FA; then
    echo "  wheel unusable (expected: built for torch 2.9). Removing it and building from SOURCE."
    python -m pip uninstall -y -q flash-attn 2>/dev/null
    echo "  source build: ~35 min, needs nvcc from the cuda module. MAX_JOBS=8."
    export MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE
    # ⚠ `--no-binary flash-attn` IS THE LOAD-BEARING FLAG. Without it pip finds the
    #   SAME broken wheel in Alliance's find-links wheelhouse and reinstalls it —
    #   observed on fir 2026-08-11, where the "source build" printed
    #   "Processing /cvmfs/.../flash_attn-2.8.3+torch29...whl" and produced the
    #   identical undefined-symbol failure in seconds. FLASH_ATTENTION_FORCE_BUILD
    #   never applies because pip never reaches an sdist.
    # --no-deps again: the sdist declares a torch requirement that would undo the pin.
    if python -m pip install --no-deps --no-build-isolation --no-binary flash-attn flash-attn \
       && fa_check; then
        FA_OK=true
    fi
else
    echo "  wheel unusable (expected: built for torch 2.9)."
    echo "  SKIPPING the ~35 min source build. Re-run with --with-flash-attn to attempt it."
fi
if ! $FA_OK; then
    # Leave NOTHING broken behind: a half-installed flash_attn makes `import
    # streambp` fail in a confusing way at job time instead of the arm simply
    # being absent.
    python -m pip uninstall -y -q flash-attn 2>/dev/null
    echo "  ⚠ flash-attn UNAVAILABLE -> streambp arms disabled. Every other method is fine."
fi
assert_torch_pin "flash-attn stage"

# --- 8. final gate
echo; echo "############ verifying ############"
fir_assert_env cpu || { echo "SETUP INCOMPLETE — fix the above before submitting anything"; exit 1; }
echo
echo "venv:  $(readlink -f "$FIR_VENV")"
echo "data:  $(readlink -f "$FIR_DATA")  (empty until 02_download_cache.sh runs)"
echo "flash-attn: $($FA_OK && echo available || echo 'UNAVAILABLE (streambp arms disabled)')"
echo "NEXT: bash sbatch/fir/02_download_cache.sh"
