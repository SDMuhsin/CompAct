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

cd "$(dirname "$0")/../.." || exit 1          # repo root
source sbatch/fir/fir_env.sh

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
target_stage() {   # target_stage <dir> <import-check> <pkg...>
    local dir="$1" check="$2"; shift 2
    echo; echo "--- $dir <- $* ---"
    python -m pip install -q --no-deps --target "./temp/$dir" "$@" \
        || { echo "  FAIL: pip install --target ./temp/$dir"; rc=1; return 1; }
    PYTHONPATH="$(pwd)/temp/$dir:${PYTHONPATH:-}" python -c "$check" \
        || { echo "  FAIL: post-install verification for ./temp/$dir"; rc=1; return 1; }
}

target_stage liger_pkgs \
    "import liger_kernel;print('  liger_kernel OK')" "liger_kernel==0.5.10"

target_stage unsloth_pkgs \
    "import unsloth_zoo;print('  unsloth OK')" \
    "unsloth==2026.8.1" "unsloth_zoo==2026.8.1" "trl==0.24.0" "datasets==4.3.0" \
    "hf_transfer" "structlog" "typer" "tyro" "nest_asyncio" "pillow"

# ⚠ ds_alst GOES INSIDE temp/, NOT beside it. profile_hyclora.py:798 computes
#   `<repo>/temp/ds_alst` from __file__ and offers no env override; an earlier
#   revision installed it at $SCRATCH/CompAct/ds_alst and verified it there, so the
#   check passed against a directory no ALST arm would ever consult. DS_BUILD_OPS=0
#   keeps this a pure-python install (no nvcc on the login node).
DS_BUILD_OPS=0 target_stage ds_alst \
    "import importlib.util as u,sys;sys.exit(0 if u.find_spec('deepspeed.runtime.sequence_parallel') else 1)" \
    "deepspeed==${FIR_PIN_DEEPSPEED_ALST}" \
    && echo "  sequence_parallel present (ALST arms available)"

# --- 3. gate --------------------------------------------------------------------
echo; echo "############ verifying ############"
echo "temp: $(readlink -f ./temp)   files=$(find ./temp -type f 2>/dev/null | wc -l)"
fir_assert_env cpu || rc=1
[ $rc -eq 0 ] && echo "--- 01c_stage_repos OK ---" \
              || echo "--- 01c_stage_repos INCOMPLETE — arms above are unavailable ---"
exit $rc
