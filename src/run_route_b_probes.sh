#!/usr/bin/env bash
# Route B falsification probes (llmdocs/trackers/route_b.md sections 4 and 2.7).
#
# Two independent questions, both answered before any kernel is written:
#   1. Can layer i's recompute hide behind layer i+1's backward on a second stream?
#      -- predicted NO (both sides are tensor-core bound; the grids already exceed 84 SMs).
#   2. Can `o_h` hide behind the idle COPY ENGINES instead?
#      -- predicted YES (unsloth_offload already moves 5376 MiB/step at seq 16384 for +0.001%).
#   3. Does the pinned round trip survive allocator pressure bitwise?
#      -- `fb_offload.py`'s own adversarial self-test, which churns allocations between stash and
#         fetch specifically to expose a missing `record_stream` or recycle event.
#
# ONE GPU, nothing else timed at the same time (protocol A.3.1 / section 33.3).
# Usage:  bash src/run_route_b_probes.sh [GPU]
set -u
cd "$(dirname "$0")/.."
source env/bin/activate
GPU="${1:-1}"
OUT=results/hyclora/frontier
mkdir -p "$OUT"

echo "############ GPU state before"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

for S in 4096 16384; do
  echo "############ stream/DMA overlap ceiling, seq $S"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python src/probe_stream_overlap.py \
      --seq "$S" --batch 2 --reps $([ "$S" -ge 16384 ] && echo 10 || echo 30) \
      --out "$OUT/stream_overlap_seq$S.json" 2>&1 | tail -25
done

echo "############ fb_offload round-trip self-test (adversarial: churns the allocator)"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python src/fb_offload.py 2>&1 | tail -10

echo "############ DONE"
