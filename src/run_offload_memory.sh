#!/usr/bin/env bash
# Peak memory for the offloaded arm across the sequence sweep, regime B.
#
# ONE ARM PER PROCESS.  The landing-buffer pool is process-global, and a buffer left over from an
# offloaded arm stays resident while the next arm is measured -- 64.00 MiB of inflation at seq 4096,
# measured, which is more than enough to move a column.  `fb_offload.reset()` now frees the pools
# and `apply_family` calls it, but one arm per process removes the question entirely.
#
# Memory survives a contended box (CONTEXT.md section 33.12); THROUGHPUT DOES NOT.  The ms/step in
# these rows is recorded for the receipt and is NOT quotable unless `nvidia-smi` showed the box
# quiet -- check `gpu_state_before` in the JSON before using any time from here.
#
# Usage:  bash src/run_offload_memory.sh [GPU] [SEQ...]
set -u
GPU=${1:-0}; shift || true
SEQS=${*:-"1024 2048 4096 8192 16384"}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
OUT=results/hyclora/offload
mkdir -p "$OUT"

for s in $SEQS; do
    for arm in fb_attn_offload_fnorm_sdpa fb_attn_fnorm_sdpa fb_min_fnorm_sdpa; do
        echo "=== seq $s  $arm ==="
        python src/profile_unsloth.py --arm "$arm" --flce --seq "$s" --batch 2 --steps 12 \
            --out "$OUT/mem_B_seq${s}_${arm}.json" 2>&1 \
            | grep -E 'peak\(alloc\)|offload staged' | sed 's|^|  |'
    done
done
