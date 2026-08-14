#!/bin/bash
# ============================================================================
# 00c_probe_deps.sh — settle WHICH INDEX serves each pinned package, in ONE trip.
# ============================================================================
#   bash sbatch/fir/00c_probe_deps.sh 2>&1 | tee dep_probe.txt
#
# Run on a FIR LOGIN NODE from the repo root. Send back `dep_probe.txt`.
#
# WHY: the venv probe showed PyPI works from inside a venv (filelock installed),
# but `pip index versions torch --index-url https://download.pytorch.org/whl/cu128`
# returned "No matching distribution found for torch". Three candidate causes,
# and they need DIFFERENT fixes, so guessing costs a round trip:
#
#   (a) `pip index` is an experimental command and misreports flat PEP-503 indexes
#       like download.pytorch.org  -> harmless, install would have worked
#   (b) download.pytorch.org is not reachable / is proxied differently to pypi.org
#       -> must install torch from PyPI instead
#   (c) Alliance's PIP_CONFIG_FILE pins an index or sets no-index, overriding
#       --index-url -> must pass --index-url AND neutralise the config
#
# This resolves EVERY pinned package WITHOUT downloading gigabytes: `--dry-run`
# does full dependency resolution and reports what it WOULD install. The only
# real download is a ~1 MB sdist-free package used as a control.
#
# Nothing is installed into the real venv. A throwaway venv is used and removed.
# ============================================================================
set -uo pipefail

# FIR_SELF must be resolved BEFORE the cd: $0 is relative to the invocation
# directory, and fir_log_to re-execs this script from the repo root.
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_probe_deps "$@"        # full transcript -> ./logs/fir_probe_deps_<UTC>.log

module load $FIR_MODULES_CPU 2>/dev/null
PROBE_VENV="${FIR_SCRATCH_ROOT}/dep_probe_env"
TORCH_IDX="https://download.pytorch.org/whl/cu128"

echo "############ DEPENDENCY PROBE — $(date -u +%FT%TZ) ############"

echo; echo "=== 0. pip configuration (candidate cause (c)) ==="
echo "PIP_CONFIG_FILE : ${PIP_CONFIG_FILE:-<unset>}"
[ -n "${PIP_CONFIG_FILE:-}" ] && [ -f "${PIP_CONFIG_FILE}" ] && {
    echo "--- contents ---"; sed 's/^/    /' "${PIP_CONFIG_FILE}"; }
echo "PIP_INDEX_URL   : ${PIP_INDEX_URL:-<unset>}"
echo "PIP_EXTRA_INDEX_URL: ${PIP_EXTRA_INDEX_URL:-<unset>}"
echo "PIP_NO_INDEX    : ${PIP_NO_INDEX:-<unset>}"
echo "PIP_FIND_LINKS  : ${PIP_FIND_LINKS:-<unset>}"
for f in /etc/pip.conf "$HOME/.pip/pip.conf" "$HOME/.config/pip/pip.conf"; do
    [ -f "$f" ] && { echo "--- $f ---"; sed 's/^/    /' "$f"; }
done

echo; echo "=== 1. raw reachability of the two indexes (candidate cause (b)) ==="
for u in "https://pypi.org/simple/torch/" "$TORCH_IDX/torch/"; do
    printf "  %-58s " "$u"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$u" 2>/dev/null)
    echo "HTTP $code"
done

echo; echo "=== 2. throwaway venv ==="
rm -rf "$PROBE_VENV"
virtualenv --no-download --system-site-packages "$PROBE_VENV" >/dev/null 2>&1 \
    || python -m venv --system-site-packages "$PROBE_VENV" >/dev/null 2>&1 \
    || { echo "  FAIL: could not create probe venv"; exit 1; }
# shellcheck disable=SC1091
source "$PROBE_VENV/bin/activate"
python -m pip install -q --upgrade pip >/dev/null 2>&1
echo "  pip: $(python -m pip --version)"

resolve() {   # resolve <label> <pip-args...>   — dry run, no download
    local label="$1"; shift
    echo; echo "--- $label ---"
    echo "    pip install --dry-run --no-deps $*"
    local out rc
    out=$(python -m pip install --dry-run --no-deps "$@" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
        echo "$out" | grep -Ei "Would install|Requirement already" | head -3 | sed 's/^/    OK   /'
        [ -z "$(echo "$out" | grep -Ei 'Would install|Requirement already')" ] && echo "$out" | tail -3 | sed 's/^/    OK   /'
    else
        echo "$out" | tail -6 | sed 's/^/    FAIL /'
    fi
    return $rc
}

echo; echo "=== 3. TORCH — the open question, tried four ways ==="
resolve "3a. torch==${FIR_PIN_TORCH} from the cu128 index"        "torch==${FIR_PIN_TORCH}" --index-url "$TORCH_IDX"
resolve "3b. torch==${FIR_PIN_TORCH} from PyPI (default index)"   "torch==${FIR_PIN_TORCH}"
resolve "3c. torch (any version) from the cu128 index"            "torch" --index-url "$TORCH_IDX"
resolve "3d. torch==${FIR_PIN_TORCH} with pip config neutralised" "torch==${FIR_PIN_TORCH}" --index-url "$TORCH_IDX" --isolated

echo; echo "=== 4. what torch versions does each index actually offer? ==="
echo "--- PyPI ---"
python -m pip index versions torch 2>&1 | head -4 | sed 's/^/    /'
echo "--- cu128 index (flat PEP-503; pip index is unreliable here, so also curl) ---"
python -m pip index versions torch --index-url "$TORCH_IDX" 2>&1 | head -4 | sed 's/^/    /'
curl -s --max-time 25 "$TORCH_IDX/torch/" 2>/dev/null \
  | grep -o 'torch-2\.1[0-9][^"]*cp311[^"]*\.whl' | head -5 | sed 's/^/    wheel: /' \
  || echo "    (could not list wheels via curl)"

echo; echo "=== 5. the rest of the pinned stack (PyPI) ==="
resolve "transformers==${FIR_PIN_TRANSFORMERS}" "transformers==${FIR_PIN_TRANSFORMERS}"
resolve "datasets==${FIR_PIN_DATASETS}"         "datasets==${FIR_PIN_DATASETS}"
resolve "peft==${FIR_PIN_PEFT}"                 "peft==${FIR_PIN_PEFT}"
resolve "accelerate==${FIR_PIN_ACCELERATE}"     "accelerate==${FIR_PIN_ACCELERATE}"
resolve "deepspeed==${FIR_PIN_DEEPSPEED}"       "deepspeed==${FIR_PIN_DEEPSPEED}"
resolve "deepspeed==${FIR_PIN_DEEPSPEED_ALST} (ALST side-by-side)" "deepspeed==${FIR_PIN_DEEPSPEED_ALST}"
resolve "bitsandbytes==${FIR_PIN_BNB}"          "bitsandbytes==${FIR_PIN_BNB}"
resolve "tensorly"                              "tensorly"
resolve "evaluate / sentencepiece / scikit-learn" evaluate sentencepiece scikit-learn

echo; echo "=== 6. flash-attn (streambp only; failure here is survivable) ==="
resolve "flash-attn (wheel resolvable?)" "flash-attn"
echo "    NOTE: resolving is NOT enough — a wheel built against a different torch"
echo "    still fails at import with 'undefined symbol: _ZN3c104cuda29c10_cuda...'."
echo "    01_setup_venv.sh verifies by RUNNING a kernel, and falls back to a source build."

echo; echo "=== 7. cleanup ==="
deactivate 2>/dev/null || true
rm -rf "$PROBE_VENV"
echo "  removed $PROBE_VENV"

echo; echo "############ DEPENDENCY PROBE COMPLETE ############"
echo "Send back this whole file. Nothing was installed into the real venv."
