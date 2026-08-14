#!/bin/bash
# ============================================================================
# 01c_stage_repos.sh — recreate `temp/` on a FIR LOGIN NODE. Internet required.
# ============================================================================
#   bash sbatch/fir/01c_stage_repos.sh 2>&1 | tee logs/fir_stage_repos.txt
#
# WHY THIS EXISTS
# ---------------
# `temp/` holds every PUBLISHED BASELINE of the §16 campaign as a real clone of the
# authors' own repository (CONTEXT.md §16: "do not reinvent them from scratch, but
# clone their repo to ./temp/"). `src/profile_hyclora.py` and `src/profile_unsloth.py`
# reach them by HARDCODED relative path — see the map in fir_env.sh.
#
# ⚠ AND `temp/` IS GITIGNORED (`.gitignore:8`). Neither `git push` nor a repo rsync
#   carries it. fir therefore received a repo whose baseline arms could not import
#   the methods they measure; preflight job 54306984 lost the `lomo --fb` arm to
#   `ModuleNotFoundError: No module named 'lomo_optim'` for exactly this reason.
#
# COMMITS ARE PINNED to what the dev-box measurements were taken against. A baseline
# that moves between clusters is not a fair comparison — it is two experiments.
#
# ⚠ INODES: these clones plus the two --target prefixes are ~7,900 files. /project is
#   at 486K/500K files, so `temp/` lives on /scratch behind a symlink exactly like
#   `env/` and `data/` (fir_link_scratch). Do NOT materialise it on /project.
# ============================================================================
set -uo pipefail

# FIR_SELF must be resolved BEFORE the cd: $0 is relative to the invocation
# directory, and fir_log_to re-execs this script from the repo root.
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1          # repo root
source sbatch/fir/fir_env.sh
fir_log_to fir_stage_repos "$@"        # full transcript -> ./logs/fir_stage_repos_<UTC>.log

FRESH=false
[ "${1:-}" = "--fresh" ] && FRESH=true

echo "############ fir temp/ staging — $(date -u +%FT%TZ) ############"
fir_load_modules_gpu || { echo "FAIL: module load '$FIR_MODULES_GPU'"; exit 1; }
fir_link_scratch || exit 1
[ -x "$FIR_VENV/bin/python" ] || { echo "FAIL: no venv — run 01_setup_venv.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$FIR_VENV/bin/activate" || exit 1
fir_export_online

rc=0

# --- 1. the author clones, at the commits the dev box measured -----------------
# name                 url                                                    commit
CLONES=(
  "lomo|https://github.com/OpenLMLab/LOMO.git|45d4bac"
  "galore|https://github.com/jiaweizzhao/GaLore.git|2cc66f8"
  "streambp|https://github.com/Ledzy/StreamBP.git|3b56c16"
  "qlora|https://github.com/artidoro/qlora.git|7f4e95a"
  "HyC-LoRA-release|https://github.com/thu-ee-acts-lab/HyC-LoRA-release.git|62ac396"
  "minis|https://github.com/wdlctc/mini-s.git|797c54b"
  "arctic|https://github.com/snowflakedb/ArcticTraining.git|b28638e"
)
for spec in "${CLONES[@]}"; do
    name="${spec%%|*}"; rest="${spec#*|}"; url="${rest%%|*}"; commit="${rest##*|}"
    dst="./temp/$name"
    echo; echo "--- $name @ $commit ---"
    if $FRESH && [ -d "$dst" ]; then rm -rf "$dst"; fi
    if [ -d "$dst/.git" ]; then
        have=$(git -C "$dst" rev-parse --short HEAD 2>/dev/null)
        if [ "$have" = "$commit" ]; then echo "  already at $commit"; continue; fi
        echo "  present at $have, want $commit -> fetching"
        git -C "$dst" fetch --quiet origin || { echo "  FAIL: fetch $name"; rc=1; continue; }
    else
        # Full clone, not --depth 1: the pinned commit may predate the fetch tip.
        git clone --quiet "$url" "$dst" || { echo "  FAIL: clone $name"; rc=1; continue; }
    fi
    git -C "$dst" checkout --quiet "$commit" \
        || { echo "  FAIL: checkout $commit — the pin is what makes this comparison fair,"
             echo "        so DO NOT fall back to the branch tip. Record a new pin deliberately."
             rc=1; continue; }
    echo "  ok: $(git -C "$dst" rev-parse --short HEAD)"
done

# --- 2. the --target prefixes ---------------------------------------------------
# These are pip installs into their own directory rather than into the venv, so that
# a competitor's package can be put on PYTHONPATH for ITS arm alone and cannot
# perturb any other arm's measurement. --no-deps for the same reason: unsloth pulls
# a datasets/trl stack that would otherwise redefine the shared environment.
#
# ⚠⚠ THE VERIFICATION MUST NOT EXECUTE THE PACKAGE. THIS SCRIPT RUNS ON A LOGIN
#    NODE, WHICH HAS NO GPU. An earlier revision checked unsloth with a plain
#    `import unsloth_zoo` and it raised, from a correctly installed package:
#       unsloth_zoo/device_type.py:235
#       NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
#    unsloth_zoo evaluates `DEVICE_TYPE = get_device_type()` AT IMPORT TIME, so
#    importing it anywhere without a GPU is fatal by design. pip had succeeded; the
#    check condemned it. This is the same trap as flash-attn (§3.5 of hpc_fir.md) and
#    it has now cost two round trips.
#    `importlib.util.find_spec("<toplevel>")` resolves the package through the real
#    PYTHONPATH — proving the --target prefix is importable — WITHOUT executing its
#    module body. Whether the package actually runs is a GPU question, and
#    03_preflight.sh is the GPU job that answers it.
#    Two edge cases, both handled below rather than discovered on fir:
#      - find_spec on a DOTTED name raises ModuleNotFoundError (it does not return
#        None) when a parent is missing, so it must be wrapped;
#      - a namespace package has `origin is None`, so the origin must be str()'d.
find_spec_check() { printf '%s' "
import importlib.util as u, sys
n = '$1'
try:
    s = u.find_spec(n)
except Exception as e:
    print(f'  {n} NOT IMPORTABLE: {type(e).__name__}: {str(e)[:200]}'); sys.exit(1)
print(f'  {n} -> {s.origin}' if s else f'  {n} NOT FOUND')
sys.exit(0 if s else 1)
"; }

target_stage() {   # target_stage <dir> <toplevel-module> <pkg...>
    local dir="$1" mod="$2"; shift 2
    echo; echo "--- $dir <- $* ---"
    # ⚠ --upgrade IS REQUIRED. `pip install --target` with an existing directory
    #   prints "Target directory ... already exists. Specify --upgrade to force
    #   replacement." and SKIPS the package — so a re-run silently keeps whatever is
    #   already there and reports success. Observed on fir 2026-08-14 (ds_alst).
    python -m pip install -q --no-deps --upgrade --target "./temp/$dir" "$@" \
        || { echo "  FAIL: pip install --target ./temp/$dir"; rc=1; return 1; }
    PYTHONPATH="$(pwd)/temp/$dir:${PYTHONPATH:-}" python -c "$(find_spec_check "$mod")" \
        || { echo "  FAIL: post-install verification for ./temp/$dir"; rc=1; return 1; }
}

target_stage liger_pkgs liger_kernel "liger_kernel==0.5.10"

target_stage unsloth_pkgs unsloth_zoo \
    "unsloth==2026.8.1" "unsloth_zoo==2026.8.1" "trl==0.24.0" "datasets==4.3.0" \
    "hf_transfer" "structlog" "typer" "tyro" "nest_asyncio" "pillow"

# ⚠ ds_alst GOES INSIDE temp/, NOT beside it. profile_hyclora.py:798 computes
#   `<repo>/temp/ds_alst` from __file__ and offers no env override; an earlier
#   revision installed it at $SCRATCH/CompAct/ds_alst and verified it there, so the
#   check passed against a directory no ALST arm would ever consult. It is staged
#   HERE and no longer in 01_setup_venv.sh — two installers into one prefix is how
#   the "already exists, specify --upgrade" skip above went unnoticed.
#   DS_BUILD_OPS=0 keeps this a pure-python install (no nvcc on the login node).
DS_BUILD_OPS=0 target_stage ds_alst deepspeed "deepspeed==${FIR_PIN_DEEPSPEED_ALST}"
# The ALST arms need one SUBmodule, not just the package. find_spec on a dotted name
# imports the PARENT, and deepspeed's __init__ is safe on CPU (it prints "Setting
# accelerator to CPU"), so this is login-node-safe — unlike unsloth_zoo above.
PYTHONPATH="$(pwd)/temp/ds_alst:${PYTHONPATH:-}" python -c "$(find_spec_check deepspeed.runtime.sequence_parallel)" \
    || { echo "  FAIL: deepspeed at temp/ds_alst lacks runtime.sequence_parallel -> ALST arms unavailable"; rc=1; }

# --- 3. gate --------------------------------------------------------------------
echo; echo "############ verifying ############"
# ⚠ `find -L`, NOT `find`. ./temp is a SYMLINK to /scratch, and find does not follow
#   symlink arguments by default — so plain `find ./temp -type f` matches only the
#   link itself (type l, not f) and reports `files=0` immediately after staging ~7,900
#   of them. Printed exactly that on fir 2026-08-14, contradicting the clone check two
#   lines below it. Cosmetic, but a status line that lies is worse than none.
echo "temp: $(readlink -f ./temp)   files=$(find -L ./temp -type f 2>/dev/null | wc -l)"
fir_assert_env cpu || rc=1
[ $rc -eq 0 ] && echo "--- 01c_stage_repos OK ---" \
              || echo "--- 01c_stage_repos INCOMPLETE — arms above are unavailable ---"
exit $rc
