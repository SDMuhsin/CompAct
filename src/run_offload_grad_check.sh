#!/usr/bin/env bash
# Correctness battery for the `o_h` offload (route_b.md section 5b).
#
# ONE VARIANT PER PROCESS, always.  Variants run in sequence inside one process share
# caching-allocator state, and that is precisely the state this defect lives in -- two earlier
# diagnoses were wrong because of it.
#
# The control is offload OFF against offload OFF: the FlashAttention backward is nondeterministic at
# ~1.5e-02 on these gradients, so "correct" means "at the control's floor", never "bitwise".
#
# Usage:  bash src/run_offload_grad_check.sh [GPU] [SEQ]
set -u
GPU=${1:-0}
SEQ=${2:-1024}
R=results/hyclora/offload/seq${SEQ}
mkdir -p "$R"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

run () {  # run <outfile> <args...>
    local out=$1; shift
    python src/probe_offload_grads.py --seq "$SEQ" --batch 2 --out "$R/$out" "$@" \
        2>&1 | grep -E '^run|stats:' | sed "s|^|  [$out] |"
}

echo "=== control: offload OFF, three independent processes (the noise floor) ==="
for r in a b c; do run "off_$r.pt" --run off; done

echo "=== defect: landing buffer taken from the COMPUTE stream's allocator pool ==="
for r in a b c; do run "bad_$r.pt" --run on --alloc-stream compute; done

echo "=== fix: landing buffer taken from the COPY stream's allocator pool ==="
for r in a b c; do run "fix_$r.pt" --run on --alloc-stream copy; done

echo "=== fix, no lookahead (synchronous round trip) ==="
run "fix_nolook.pt" --run on --alloc-stream copy --no-lookahead

echo "=== fix, three optimizer-free steps (pool reaches steady state and recycles) ==="
run "fix_steps3.pt" --run on --alloc-stream copy --steps 3
run "off_steps3.pt" --run off --steps 3

echo "=== independent second fix: copy stream waits on compute before each H2D ==="
run "fix_waitalloc.pt" --run on --alloc-stream compute --wait-alloc

echo
echo "=== comparison (reference = off_a) ==="
python src/probe_offload_grads.py --compare "$R"/off_a.pt "$R"/off_b.pt "$R"/off_c.pt \
    "$R"/bad_a.pt "$R"/bad_b.pt "$R"/bad_c.pt \
    "$R"/fix_a.pt "$R"/fix_b.pt "$R"/fix_c.pt \
    "$R"/fix_nolook.pt "$R"/fix_waitalloc.pt
echo
echo "=== multi-step comparison (reference = off_steps3) ==="
python src/probe_offload_grads.py --compare "$R"/off_steps3.pt "$R"/fix_steps3.pt
