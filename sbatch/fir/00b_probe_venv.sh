#!/bin/bash
# ============================================================================
# 00b_probe_venv.sh — settle HOW to build a venv on fir, in ONE round trip.
# ============================================================================
#   bash sbatch/fir/00b_probe_venv.sh 2>&1 | tee venv_probe.txt
#
# Run on a FIR LOGIN NODE, from the repo root. Then send back `venv_probe.txt`.
#
# WHY: two creation methods have now failed in two different ways
#   - `python -m venv`            : reached ensurepip, interrupted (never proven)
#   - `virtualenv --no-download`  : OSError [Errno 5] reading a CVMFS seed zip
# Guessing a third would be a third round trip. This tries EVERY candidate,
# INDEPENDENTLY, does not stop at the first failure, and reports what works.
#
# Each attempt builds into its own throwaway directory under /scratch and is
# removed afterwards, EXCEPT the winner, which is left at
# $FIR_SCRATCH_ROOT/env_probe_winner so it can simply be renamed into place.
# Nothing existing is modified. Total runtime: a few minutes, mostly waiting on
# whichever method is slow.
# ============================================================================
set -uo pipefail

# FIR_SELF must be resolved BEFORE the cd: $0 is relative to the invocation
# directory, and fir_log_to re-execs this script from the repo root.
FIR_SELF="$(readlink -f "$0")"
cd "$(dirname "$FIR_SELF")/../.." || exit 1
source sbatch/fir/fir_env.sh
fir_log_to fir_probe_venv "$@"        # full transcript -> ./logs/fir_probe_venv_<UTC>.log

PROBE_ROOT="${FIR_SCRATCH_ROOT}/venv_probe"
rm -rf "$PROBE_ROOT"; mkdir -p "$PROBE_ROOT"
WINNER=""

echo "############ VENV PROBE — $(date -u +%FT%TZ) ############"
echo "repo      : $(pwd)"
echo "probe root: $PROBE_ROOT"

echo; echo "=== 0. context ==="
module load $FIR_MODULES_CPU 2>/dev/null
echo "python    : $(python -V 2>&1) at $(command -v python)"
echo "virtualenv: $(command -v virtualenv || echo '<not on PATH>') $(virtualenv --version 2>/dev/null | head -1)"
echo "pip(sys)  : $(python -m pip --version 2>&1 | head -1)"
echo "TMPDIR    : ${TMPDIR:-<unset>}"
echo "XDG_DATA_HOME: ${XDG_DATA_HOME:-<unset>}"
echo "virtualenv app-data default: ${HOME}/.local/share/virtualenv"
ls -ld "${HOME}/.local/share/virtualenv" 2>/dev/null || echo "  (app-data dir does not exist yet)"

echo; echo "=== 1. can we even read virtualenv's bundled seed wheels from CVMFS? ==="
# This is the exact read that threw Errno 5. If it fails here, the fault is the
# CVMFS mount, not the venv tooling, and NO virtualenv seeder will work.
python - <<'PY'
import glob, zipfile, os, sys
try:
    import virtualenv
    base = os.path.join(os.path.dirname(virtualenv.__file__), "seed", "wheels", "embed")
    whls = sorted(glob.glob(os.path.join(base, "*.whl")))
    print(f"  seed wheel dir: {base}")
    print(f"  found {len(whls)} wheels")
    for w in whls:
        try:
            with zipfile.ZipFile(w) as z:
                n = len(z.namelist())
                tot = sum(len(z.read(m)) for m in z.namelist()[:40])   # force real reads
            print(f"    OK   {os.path.basename(w)}  ({n} members, first-40 bytes read {tot})")
        except Exception as e:
            print(f"    FAIL {os.path.basename(w)}: {type(e).__name__}: {e}")
except Exception as e:
    print(f"  could not inspect virtualenv seed wheels: {type(e).__name__}: {e}")
PY

echo; echo "=== 2. /scratch write sanity (many small files, like a venv) ==="
T="$PROBE_ROOT/writetest"; mkdir -p "$T"
if ( for i in $(seq 1 500); do echo x > "$T/f$i"; done ) 2>&1; then
    echo "  wrote 500 small files OK"
else
    echo "  FAILED writing small files to /scratch"
fi
rm -rf "$T"

# ---------------------------------------------------------------------------
try_method() {   # try_method <label> <dir> <command...>
    local label="$1" dir="$2"; shift 2
    echo; echo "=== $label ==="
    echo "  cmd: $*"
    rm -rf "$dir"; mkdir -p "$(dirname "$dir")"
    local t0=$SECONDS out rc
    out=$("$@" 2>&1); rc=$?
    local dt=$((SECONDS - t0))
    if [ $rc -ne 0 ]; then
        echo "  RESULT: FAILED (exit $rc, ${dt}s)"
        echo "$out" | tail -25 | sed 's/^/    /'
        return 1
    fi
    # health: bin/python runs, bin/activate exists, pip works, numpy visible
    local ok=true
    [ -f "$dir/bin/activate" ] || { echo "  RESULT: FAILED — no bin/activate (${dt}s)"; ok=false; }
    if $ok; then
        "$dir/bin/python" -c "import sys; print('    python', sys.version.split()[0])" || ok=false
        "$dir/bin/python" -m pip --version 2>&1 | sed 's/^/    pip: /' || ok=false
        "$dir/bin/python" -c "import numpy; print('    numpy', numpy.__version__, '(from system site-packages)')" 2>&1 | sed 's/^/  /' || ok=false
    fi
    if $ok; then
        echo "  RESULT: OK (${dt}s)"
        [ -z "$WINNER" ] && WINNER="$label|$dir"
        return 0
    fi
    echo "  RESULT: created but UNHEALTHY (${dt}s)"
    return 1
}

# ---------------------------------------------------------------------------
# The candidates, in the order we would prefer them.
# ---------------------------------------------------------------------------
export VIRTUALENV_OVERRIDE_APP_DATA="$PROBE_ROOT/appdata"
mkdir -p "$VIRTUALENV_OVERRIDE_APP_DATA"

if command -v virtualenv >/dev/null 2>&1; then
    try_method "A. virtualenv --no-download (app-data on /scratch)" \
        "$PROBE_ROOT/A" virtualenv --no-download --system-site-packages "$PROBE_ROOT/A"

    try_method "B. virtualenv --seeder pip (avoids the app-data zip extract entirely)" \
        "$PROBE_ROOT/B" virtualenv --seeder pip --system-site-packages "$PROBE_ROOT/B"

    try_method "C. virtualenv --no-seed (no pip seeded; we'd bootstrap pip after)" \
        "$PROBE_ROOT/C" virtualenv --no-seed --system-site-packages "$PROBE_ROOT/C"
else
    echo; echo "=== A/B/C skipped: virtualenv not on PATH ==="
fi

# D is the one that was already working before it was interrupted. It is SLOW
# (ensurepip on Lustre) — this is the run that proves whether it completes.
try_method "D. python -m venv --system-site-packages (slow; ensurepip)" \
    "$PROBE_ROOT/D" python -m venv --system-site-packages "$PROBE_ROOT/D"

try_method "E. python -m venv --without-pip (fast; pip bootstrapped separately)" \
    "$PROBE_ROOT/E" python -m venv --without-pip --system-site-packages "$PROBE_ROOT/E"

# ---------------------------------------------------------------------------
echo; echo "=== 3. does the winner's pip actually reach PyPI? ==="
if [ -n "$WINNER" ]; then
    wdir="${WINNER#*|}"
    echo "  winner: ${WINNER%%|*}  ($wdir)"
    if [ -x "$wdir/bin/python" ] && "$wdir/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "  --- pip install filelock (small, harmless, real network test) ---"
        "$wdir/bin/python" -m pip install --quiet --no-cache-dir filelock 2>&1 | tail -8 | sed 's/^/    /'
        "$wdir/bin/python" -c "import filelock; print('    filelock', filelock.__version__, '-> PyPI reachable from the venv')" 2>&1
        echo "  --- can pip see the pinned torch wheel? (resolve only, NO download) ---"
        "$wdir/bin/python" -m pip index versions torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | head -3 | sed 's/^/    /'
    else
        echo "  winner has no working pip — would need bootstrapping"
    fi
    mv "$wdir" "${FIR_SCRATCH_ROOT}/env_probe_winner" 2>/dev/null && \
        echo "  kept winner at ${FIR_SCRATCH_ROOT}/env_probe_winner"
else
    echo "  !!! NO METHOD PRODUCED A HEALTHY VENV — send this file, do not retry blindly."
fi

echo; echo "=== 4. cleanup ==="
rm -rf "$PROBE_ROOT"
echo "  removed $PROBE_ROOT (the winner, if any, was moved out first)"

echo; echo "############ VENV PROBE COMPLETE ############"
echo "Send back this whole file."
