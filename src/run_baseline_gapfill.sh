#!/usr/bin/env bash
# Baseline-completeness gap-fill (llmdocs/trackers/baseline_completeness.md).
#
# Every run here is a MEMORY measurement plus a protocol-A.3.6 GPU-kernel-time measurement.
# Neither is invalidated by a co-tenant: peak allocated is per-process, and kernel time is the
# figure the protocol names for a box that cannot be quiesced.  Wall-clock ms/step from these runs
# is VOID and must not be quoted -- see the tracker.
#
# One arm at a time on ONE GPU (protocol A.3.1).  `|| true` on each run because the point of some
# of these rows is to establish WHERE an arm OOMs; the harness records the error inside the JSON.
#
# Usage:  bash src/run_baseline_gapfill.sh [GPU] [PHASE]
#           GPU    physical device index, default 0
#           PHASE  all | r3 | hyclora | regimeA | reserved   (default all)
set -u
cd "$(dirname "$0")/.."
source env/bin/activate

GPU="${1:-0}"
PHASE="${2:-all}"
OUT=results/hyclora/frontier
mkdir -p "$OUT"

SEQS="1024 2048 4096 8192 16384"
run()   { echo "=== $* ==="; CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src \
            python src/profile_unsloth.py --device cuda:0 --batch 2 --steps 12 "$@" 2>&1 | tail -6 || true; }
run_u() { echo "=== $* ==="; CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=temp/unsloth_pkgs:src \
            python src/profile_unsloth.py --device cuda:0 --batch 2 --steps 12 \
            --unsloth_bf16_adapters "$@" 2>&1 | tail -6 || true; }

# --- G4 / G14: the no-compression rows (protocol R3) and each competitor's best-on-axis config ---
# R3 is binding: "LoRA + FlashAttention with no activation compression is a row in every table."
# D.0 step 3 is also binding: the competitor's best configuration ON THE AXIS BEING CLAIMED must be
# identified and swept.  unsloth's best-for-memory (unsloth_offload) is already swept at all five
# lengths; their best-for-THROUGHPUT is `unsloth_nogc`, which had only ever been run at seq 1024 --
# where it is 21% faster than our fastest arm.  A throughput claim cannot stand on an unswept
# competitor optimum.  These arms are heavy and are EXPECTED to OOM at long sequence; that is a
# reportable row, not a failure.
if [ "$PHASE" = all ] || [ "$PHASE" = r3 ]; then
  for S in $SEQS; do
    run   --arm baseline_sdpa --flce --seq "$S" --kernel_time --out "$OUT/gapfill_baseline_seq$S.json"
    run   --arm liger_sdpa           --seq "$S" --kernel_time --out "$OUT/gapfill_liger_nogc_seq$S.json"
    run_u --arm unsloth_nogc         --seq "$S" --kernel_time --out "$OUT/gapfill_unsloth_nogc_seq$S.json"
  done
fi

# --- G5: HyC-LoRA's COMPRESSED arms, at every length -------------------------------------------
# §33.11 caveat 2: `hyclora_flash_nc` runs with compression BYPASSED, and its 43.76-81.79% margin
# must never be quoted alone.  The pairing -- their real compressed method -- exists only at seq
# 1024.  Their FlashAttention compressed path is VOID upstream (it does not train; do NOT repair
# it, that would change their method), so the compressed arm is the eager one.
if [ "$PHASE" = all ] || [ "$PHASE" = hyclora ]; then
  for S in $SEQS; do
    run --arm hyclora_q2 --flce --seq "$S" --kernel_time --out "$OUT/gapfill_hyclora_q2_seq$S.json"
    run --arm hyclora_q4 --flce --seq "$S" --kernel_time --out "$OUT/gapfill_hyclora_q4_seq$S.json"
  done
fi

# --- G6: regime A, re-measured after the in-place GLU kernel ------------------------------------
# The protocol requires both regimes.  Ours date from before the kernel landed and are stale by
# -44 MiB at 1024 rising to -704 MiB at 16384.
if [ "$PHASE" = all ] || [ "$PHASE" = regimeA ]; then
  for S in $SEQS; do
    run --arm fb_min_fnorm_sdpa,fb_attn_fnorm_sdpa,fb_auto_fnorm_sdpa --seq "$S" \
        --out "$OUT/gapfill_regimeA_seq$S.json"
  done
fi

# --- G7: reserved memory, one arm per process ---------------------------------------------------
# Protocol A.1: reserved may only be quoted from a dedicated single-arm process.  A pooled reserved
# figure was published once from a multi-arm run and was wrong by 67%.
if [ "$PHASE" = all ] || [ "$PHASE" = reserved ]; then
  for S in $SEQS; do
    for A in fb_min_fnorm_sdpa fb_attn_fnorm_sdpa gc_manual_sdpa liger_gc_sdpa; do
      FL=--flce; case "$A" in liger*) FL="";; esac
      run --arm "$A" $FL --seq "$S" --out "$OUT/solo_${A}_seq$S.json"
    done
    run_u --arm unsloth_gc      --seq "$S" --out "$OUT/solo_unsloth_gc_seq$S.json"
    run_u --arm unsloth_offload --seq "$S" --memcpy --out "$OUT/solo_unsloth_offload_seq$S.json"
  done
fi

echo "gap-fill phase '$PHASE' complete on GPU $GPU"
