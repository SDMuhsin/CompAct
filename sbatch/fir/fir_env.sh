#!/bin/bash
# ============================================================================
# fir_env.sh — THE SINGLE SOURCE OF TRUTH for the fir environment.
# ============================================================================
# Sourced by every other script here. Nothing else defines a module line, an
# offline flag, a GPU string, an account or a path.
#
#   source sbatch/fir/fir_env.sh
#   fir_load_modules_gpu / fir_load_modules_cpu
#   fir_export_offline  / fir_export_online
#   fir_assert_env [gpu|cpu]      # FAIL FAST before any real work
#
# ALL VALUES BELOW ARE MEASURED ON FIR by `00_probe_fir.sh` (2026-08-11,
# login1.int.fir.alliancecan.ca). Where fir differs from rorqual the rorqual
# value is kept in a comment, because silently dropping it loses the reason.
# ============================================================================

# ---------------------------------------------------------------------------
# MODULES — ⚠ ORDER IS LOAD-BEARING, AND FIR PROVES IT.
#
#   `module avail cudnn` on fir returns "No module(s) found" ON ITS OWN.
#   cudnn is CUDA-dependent in the Lmod hierarchy and only becomes visible AFTER
#   `cuda` is loaded. So `... cuda cudnn` works (probe: exit=0) and any reorder
#   that puts cudnn before cuda FAILS. This is exactly why the rorqual order is
#   copied verbatim rather than tidied.
#
# VERIFIED on fir: the rorqual line loads cleanly and yields
#   cuda/12.6, cudnn/9.10.0.56, nccl/2.26.2, python/3.11.5, numpy 2.4.2
# Note fir ALREADY loads gcc/12.3, arrow/25.0.0, scipy-stack/2026a, python/3.11.5
# by default; loading them explicitly is a no-op that also protects against a
# future default change. Keep it.
# ---------------------------------------------------------------------------
FIR_MODULES_CPU="${FIR_MODULES_CPU:-gcc arrow scipy-stack}"
FIR_MODULES_GPU="${FIR_MODULES_GPU:-gcc arrow scipy-stack cuda cudnn}"

# ---------------------------------------------------------------------------
# GPUS — ⚠ FIR'S NAMES ARE NOT RORQUAL'S. Using rorqual's string fails outright.
#   fir Gres (probe §8):  gpu:h100:4
#                         gpu:nvidia_h100_80gb_hbm3_3g.40gb:4
#                         gpu:nvidia_h100_80gb_hbm3_2g.20gb:4
#                         gpu:nvidia_h100_80gb_hbm3_1g.10gb:8
#   rorqual was:          h100:1  and  h100_3g.40gb:1
#
# ⚠⚠ AND THE MIG SLICES ARE INTERACTIVE-ONLY ON FIR. Every MIG gres lives on the
#    `gpubase_interac` partition (a single node, fc11020). The batch partitions
#    (gpubase_bynode_b*) carry FULL h100:4 only. So a BATCH job must request a
#    full H100 — rorqual's habit of submitting 40 GB MIG slices to batch does not
#    transfer, and requesting one would sit in the queue forever.
# ---------------------------------------------------------------------------
FIR_GPU_FULL="${FIR_GPU_FULL:-h100:1}"                                   # batch: use this
FIR_GPU_MIG40="${FIR_GPU_MIG40:-nvidia_h100_80gb_hbm3_3g.40gb:1}"        # interactive ONLY
FIR_GPU_MEM="${FIR_GPU_MEM:-64000M}"

# ---------------------------------------------------------------------------
# ACCOUNTS — ⚠ fir splits CPU and GPU allocations. `def-seokbum` alone is NOT a
# valid account here; probe §9 shows only `def-seokbum_cpu` and `def-seokbum_gpu`.
# ---------------------------------------------------------------------------
FIR_ACCOUNT_GPU="${FIR_ACCOUNT_GPU:-def-seokbum_gpu}"
FIR_ACCOUNT_CPU="${FIR_ACCOUNT_CPU:-def-seokbum_cpu}"

# ---------------------------------------------------------------------------
# PATHS — ⚠ DO NOT PUT ANYTHING ON /project. Probe §10: it is at
#   938 GiB / 954 GiB (98%) and 486K / 500K files — effectively full on BOTH
#   space and inode count. /scratch is 41 GiB / 19 TiB and is where the repo
#   already lives. /home is only 48 GiB.
# The HF cache alone is ~10 GB (PG-19 is ~7 GB of it) and the venv is several GB.
# ---------------------------------------------------------------------------
# The REPO lives on /project (code only). The VENV and the HF CACHE must live on
# /scratch and are reached through SYMLINKS, so that every existing `./env/bin/...`
# and `$(pwd)/data` reference in the repo keeps working untouched.
#
# ⚠ THE INODE BUDGET IS THE BINDING CONSTRAINT, NOT THE SPACE. /project is at
#   486K / 500K files. A virtualenv is tens of thousands of files on its own and
#   the HF cache is tens of thousands more; either one lands the project over its
#   file quota, at which point EVERY write on /project fails, including jobs that
#   have nothing to do with this repo.
FIR_SCRATCH_ROOT="${FIR_SCRATCH_ROOT:-${SCRATCH:-/scratch/$USER}/CompAct}"
FIR_VENV_REAL="${FIR_VENV_REAL:-$FIR_SCRATCH_ROOT/env}"
FIR_DATA_REAL="${FIR_DATA_REAL:-$FIR_SCRATCH_ROOT/data}"
FIR_TEMP_REAL="${FIR_TEMP_REAL:-$FIR_SCRATCH_ROOT/temp}"
FIR_VENV="${FIR_VENV:-./env}"                       # symlink -> $FIR_VENV_REAL
FIR_DATA="${FIR_DATA:-$(pwd)/data}"                 # symlink -> $FIR_DATA_REAL
FIR_TEMP="${FIR_TEMP:-$(pwd)/temp}"                 # symlink -> $FIR_TEMP_REAL

# ⚠ `temp/` IS NOT SCRATCH SPACE — IT IS A CODE DEPENDENCY, AND IT IS GITIGNORED.
#   `.gitignore:8` excludes it, so neither git nor a repo-only rsync carries it, and
#   fir got a repo whose baseline arms could not import their own methods. Every
#   published baseline of the §16 campaign is a CLONE living there, reached by an
#   explicit `sys.path.insert` at a HARDCODED relative path:
#       temp/lomo              profile_hyclora.py:671   (lomo, adalomo)
#       temp/galore            profile_hyclora.py:724   (galore)
#       temp/minis             profile_hyclora.py:631   (mini-sequence)
#       temp/ds_alst           profile_hyclora.py:798   (alst — deepspeed >= 0.17)
#       temp/arctic/...        profile_hyclora.py:815   (alst tiled_compute)
#       temp/unsloth_pkgs      profile_unsloth.py:39    (PYTHONPATH, not sys.path)
#       temp/liger_pkgs                                 (PYTHONPATH)
#   `01c_stage_repos.sh` recreates it on fir from pinned upstream commits.
#
# ⚠ AND ds_alst MUST BE INSIDE IT. An earlier revision put the side-by-side
#   DeepSpeed at $FIR_SCRATCH_ROOT/ds_alst while profile_hyclora.py:798 computes
#   `<repo>/temp/ds_alst` from `__file__` and has no env override — so the setup
#   script "verified sequence_parallel present: True" against a directory no ALST
#   arm would ever look in. Route it through the temp symlink so both agree.
FIR_DS_ALST="${FIR_DS_ALST:-$FIR_TEMP/ds_alst}"     # DeepSpeed>=0.17, ALST arms only

# Create the scratch targets and the symlinks. Idempotent; refuses to clobber a
# real directory that is not already a symlink.
fir_link_scratch() {
    mkdir -p "$FIR_VENV_REAL" "$FIR_DATA_REAL" "$FIR_TEMP_REAL"
    local pair
    for pair in "./env:$FIR_VENV_REAL" "./data:$FIR_DATA_REAL" "./temp:$FIR_TEMP_REAL"; do
        local link="${pair%%:*}" target="${pair#*:}"
        if [ -L "$link" ]; then
            [ "$(readlink -f "$link")" = "$(readlink -f "$target")" ] || {
                echo "FAIL: $link points at $(readlink -f "$link"), expected $target"; return 1; }
        elif [ -e "$link" ]; then
            echo "FAIL: $link exists and is NOT a symlink. On fir it must live on /scratch."
            echo "      Move it aside, then re-run: mv $link ${link}.local"
            return 1
        else
            ln -s "$target" "$link"
        fi
        echo "  $link -> $(readlink -f "$link")"
    done
}

# ---------------------------------------------------------------------------
# THE PINNED STACK. ⚠ READ THIS BEFORE CHANGING A VERSION.
#
# Every number in CONTEXT.md was measured on the LEFT column. fir's wheelhouse
# offers the RIGHT column, and the two are NOT interchangeable:
#
#   package        measured (dev box)   fir `avail_wheels`   verdict
#   python         3.10.12              3.11.5 (module)      venv MUST be rebuilt; wheels are cp311
#   torch          2.10.0+cu128         2.13.0               different allocator/kernels
#   transformers   4.51.3               5.14.1               ⚠ MAJOR BUMP — see below
#   datasets       4.5.0                5.0.0                script-dataset loading already dropped at 4.x
#   peft           0.18.1               0.19.1
#   accelerate     1.12.0               1.14.0
#   deepspeed      0.16.5 (+0.17.6)     0.18.1
#   bitsandbytes   0.49.2               0.49.2               match
#   tensorly       0.9.0                0.9.0                match
#   triton         3.6.0                3.6.0                match
#   flash_attn     built from source    2.8.3 +torch29       ⚠ built for torch 2.9, we pin 2.10
#
# ⚠ MEASURED 2026-08-11 (00c_probe_deps.sh), and it overturns the row above:
#   EVERY pin resolves from Alliance's OWN wheelhouse as `<version>+computecanada`
#   — torch 2.10.0, transformers 4.51.3, datasets 4.5.0, peft 0.18.1,
#   accelerate 1.12.0, bitsandbytes 0.49.2, tensorly 0.9.0, evaluate 0.4.6.
#   `avail_wheels` showed only the DEFAULT (torch 2.13.0); the older pins are there
#   too. deepspeed 0.16.5/0.17.6 come from PyPI (no +computecanada suffix).
#   So NO --index-url is needed anywhere, and `pip index versions` is an unreliable
#   experimental command that reported "No matching distribution" for a package
#   that installs fine — do not use it to conclude anything.
#
# ⚠ THE BUILD DIFFERS EVEN WHERE THE VERSION MATCHES: fir installs
#   `torch 2.10.0+computecanada`, the dev box measured `2.10.0+cu128`. Same
#   upstream version, different compile. Peak-memory numbers are allocator- and
#   kernel-sensitive, so fir results are internally comparable but must not be
#   quoted interchangeably with CONTEXT.md's dev-box table without saying so.
#
# ⚠ transformers 5.x WILL NOT SILENTLY WORK. The fused block patches
#   LlamaDecoderLayer / LlamaRMSNorm internals and the eleven architecture guards
#   key on 4.51.3 structure; HyC-LoRA's port and StreamBP both import
#   `transformers.modeling_flash_attention_utils._flash_attention_forward`. A major
#   bump is a DIFFERENT EXPERIMENT, not an upgrade.
#
# PyPI is reachable from the fir LOGIN node (probe §7: HTTP 200), so the pinned
# versions are installable. `01_setup_venv.sh` pins them and verifies.
# ---------------------------------------------------------------------------
FIR_PIN_TORCH="${FIR_PIN_TORCH:-2.10.0}"
FIR_PIN_TRANSFORMERS="${FIR_PIN_TRANSFORMERS:-4.51.3}"
FIR_PIN_DATASETS="${FIR_PIN_DATASETS:-4.5.0}"
FIR_PIN_PEFT="${FIR_PIN_PEFT:-0.18.1}"
FIR_PIN_ACCELERATE="${FIR_PIN_ACCELERATE:-1.12.0}"
FIR_PIN_DEEPSPEED="${FIR_PIN_DEEPSPEED:-0.16.5}"
FIR_PIN_DEEPSPEED_ALST="${FIR_PIN_DEEPSPEED_ALST:-0.17.6}"
FIR_PIN_BNB="${FIR_PIN_BNB:-0.49.2}"

fir_load_modules_cpu() { module load $FIR_MODULES_CPU; }
fir_load_modules_gpu() { module load $FIR_MODULES_GPU; }

# ---------------------------------------------------------------------------
# LOGGING — every fir script writes a FULL transcript to ./logs/, automatically.
#
# Not a convenience. Diagnosing a fir failure from this repo means reading the
# transcript, and asking for `2>&1 | tee ...` by convention has already produced
# hand-pasted terminal scrollback that was TRUNCATED mid-line at the exact point
# the outcome would have appeared (2026-08-14). A script that only prints is a
# script whose failure cannot be handed to anyone.
#
# Usage — right after `cd` to the repo root, before any real work:
#     FIR_SELF="$(readlink -f "$0")"      # BEFORE the cd; $0 is relative
#     cd ...; source sbatch/fir/fir_env.sh
#     fir_log_to fir_setup_venv "$@"
# It re-execs the script once with stdout+stderr teed to the log, and the child
# short-circuits on FIR_LOGGING. Exit status is preserved via PIPESTATUS.
# ---------------------------------------------------------------------------
fir_log_to() {   # fir_log_to <tag> "$@"
    [ -n "${FIR_LOGGING:-}" ] && return 0
    local tag="$1"; shift
    [ -n "${FIR_SELF:-}" ] || { echo "fir_log_to: FIR_SELF unset — not logging"; return 0; }
    mkdir -p ./logs
    local f="./logs/${tag}_$(date -u +%Y%m%dT%H%M%SZ).log"
    export FIR_LOGGING=1
    echo "### transcript -> $f"
    "$FIR_SELF" "$@" 2>&1 | tee "$f"
    local rc=${PIPESTATUS[0]}
    echo "### $tag exit=$rc" | tee -a "$f"
    echo "### transcript: $f"
    exit $rc
}

fir_export_online() {
    export HF_HOME="$FIR_DATA"
    export TORCH_HOME="$FIR_DATA"
    export HF_HUB_DISABLE_XET=1      # xet backend has produced stalled/partial pulls
    mkdir -p "$HF_HOME"
}

fir_export_offline() {
    export HF_HOME="$FIR_DATA"
    export TORCH_HOME="$FIR_DATA"
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    # ⚠ MANDATORY AND NOT REDUNDANT: `evaluate` ignores HF_HUB_OFFLINE. Without
    # this, evaluate.load() probes the Hub from a node with no route and stalls
    # ~44 MINUTES PER SEED. This single line has cost this project more compute
    # than anything else.
    export HF_EVALUATE_OFFLINE=1
    # torch >= 2.9 renamed this and warns on every process start; older torch only
    # knows the old name. Set BOTH so the setting actually applies either way and
    # the logs stay clean. (fir warns: "PYTORCH_CUDA_ALLOC_CONF is deprecated, use
    # PYTORCH_ALLOC_CONF instead".)
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
    mkdir -p "$HF_HOME"
}

# ---------------------------------------------------------------------------
# fir_assert_env — fail on the LOGIN NODE, not 40 minutes into an allocation.
# ---------------------------------------------------------------------------
fir_assert_env() {
    local want="${1:-gpu}" rc=0
    echo "--- fir_assert_env ($want) ---"
    [ -d ./src ] || { echo "FAIL: not in repo root (no ./src)"; return 1; }
    [ -x "$FIR_VENV/bin/python" ] || { echo "FAIL: no venv at $FIR_VENV — run 01_setup_venv.sh"; return 1; }

    # ⚠⚠ THIS LIST MUST COVER WHAT THE RUN ACTUALLY IMPORTS, NOT WHAT LOOKS CORE.
    # fir preflight 54306984 (2026-08-12) printed `fir_assert_env PASSED` and then
    # lost 4/4 arms to `ModuleNotFoundError` — triton, galore_torch, lomo_optim. The
    # gate checked nine packages and none of the three that decided the job. A gate
    # that passes a broken environment is worse than no gate: it moves the failure
    # from a free login-node second to a GPU allocation.
    # Rule: a package belongs here iff SOME arm imports it at MODULE scope. `triton`
    # (flashffn.py:20) and `galore_torch` (train_glue.py:82, reached by every arm via
    # run_production.write_row) are not optional at all.
    "$FIR_VENV/bin/python" - <<PY || rc=1
import importlib, sys
bad = []
for m in ["numpy","torch","transformers","peft","datasets","accelerate",
          "filelock","pandas","evaluate",
          "triton",          # flashffn.py:20 -- EVERY fb_* arm. NOT transitive on fir.
          "galore_torch",    # train_glue.py:82 -- module scope, so EVERY arm's CSV write
          "bitsandbytes",    # qlora arms
          "deepspeed",       # zero3 arms
          "tensorly"]:       # galore arms
    try: importlib.import_module(m)
    except Exception as e:
        # ⚠ PRINT THE MESSAGE, NOT JUST THE TYPE. "peft (RuntimeError)" is
        # unactionable and cost a round trip on fir 2026-08-11; the message names
        # the actual conflict (usually a version mismatch after a silent downgrade).
        bad.append(m)
        print(f"  IMPORT FAILED {m}: {type(e).__name__}: {str(e)[:400]}")
print("  core imports:", "OK" if not bad else "MISSING -> " + ", ".join(bad))
if bad: sys.exit(1)
import transformers, datasets, torch
pins = {"transformers": ("$FIR_PIN_TRANSFORMERS", transformers.__version__),
        "datasets": ("$FIR_PIN_DATASETS", datasets.__version__)}
drift = {k: v for k, v in pins.items() if v[0] and not v[1].startswith(v[0])}
if not torch.__version__.startswith("$FIR_PIN_TORCH"):
    drift["torch"] = ("$FIR_PIN_TORCH", torch.__version__)
for k, (want, got) in drift.items():
    print(f"  ⚠ {k}: pinned {want}, installed {got} — this is a DIFFERENT EXPERIMENT")
    print(f"    from every number in CONTEXT.md. Fix the pin or record the change deliberately.")
if drift: sys.exit(1)
print("  pinned stack: OK")
PY

    if [ "$want" = "gpu" ]; then
        "$FIR_VENV/bin/python" - <<'PY' || rc=1
import torch, sys
if not torch.cuda.is_available():
    print("  FAIL: no CUDA device visible"); sys.exit(1)
print(f"  torch {torch.__version__} cuda={torch.version.cuda} "
      f"dev={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
PY
    fi

    # --- the temp/ clones. Missing ones do not break the fused block, but they DO
    # silently remove published-baseline arms from a §16 comparison, which is the
    # worse failure: the sweep completes, the table is short, and nothing said so.
    #
    # ⚠ ORDERING — `FIR_ASSERT_SKIP_TEMP=1` EXISTS FOR A REAL DEPENDENCY, NOT AS AN
    #   ESCAPE HATCH. `01c_stage_repos.sh` needs the venv (it pip-installs into
    #   --target prefixes), so it can only run AFTER `01_setup_venv.sh`. Without this
    #   flag 01's own closing gate demands artifacts only 01c can create and a correct
    #   fresh setup ALWAYS ends "SETUP INCOMPLETE" — observed on fir 2026-08-14.
    #   01 sets it; 01c and 03_preflight do NOT, so nothing reaches a GPU unchecked.
    #
    # Each probe is the EXACT artifact the code loads, not the directory it sits in —
    # a clone that fetched but left the pin unchecked-out would pass a bare -d test.
    if [ -n "${FIR_ASSERT_SKIP_TEMP:-}" ]; then
        echo "  temp/ baseline clones: SKIPPED (FIR_ASSERT_SKIP_TEMP -> run 01c_stage_repos.sh next)"
    else
        local miss=""
        local probe
        for probe in "lomo/lomo_optim" "galore/galore_torch" "minis/minis/mini_sequence.py" \
                     "streambp/src/streambp/stream_model.py" \
                     "arctic/arctic_training/model/tiled_compute.py" "ds_alst/deepspeed" \
                     "liger_pkgs/liger_kernel" "unsloth_pkgs/unsloth" "HyC-LoRA-release"; do
            [ -e "./temp/${probe}" ] || miss="$miss ${probe%%/*}"
        done
        if [ -n "$miss" ]; then
            echo "  ⚠ temp/ clones MISSING ->$miss"
            echo "    Those arms cannot run. temp/ is gitignored (.gitignore:8), so a repo"
            echo "    sync does NOT carry it -> run: bash sbatch/fir/01c_stage_repos.sh"
            rc=1
        else
            echo "  temp/ baseline clones: OK"
        fi
    fi

    ( fir_export_offline
      "$FIR_VENV/bin/python" - <<'PY' || exit 1
import sys
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("  offline model cache: OK")
except Exception as e:
    print(f"  FAIL offline model cache: {type(e).__name__}: {str(e)[:140]}")
    print("    -> run 02_download_cache.sh on a LOGIN node first")
    sys.exit(1)
PY
    ) || rc=1

    [ $rc -eq 0 ] && echo "--- fir_assert_env PASSED ---" || echo "--- fir_assert_env FAILED ---"
    return $rc
}
