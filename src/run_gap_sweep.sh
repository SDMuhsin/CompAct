#!/usr/bin/env bash
# THE THROUGHPUT-GAP SWEEP (2026-08-03) -- re-measure after `keep='attn'` removes the block's only
# O(S^2) recompute.
#
# WHAT CHANGED AND WHY THIS SWEEP EXISTS
# --------------------------------------
# `holes_closed.md` section 3.3 measured `fb_min_fnorm` 4.5% slower than `hyclora_flash_nc` at seq
# 8192 and 9.1% slower at 16384, outside the project's 5% throughput budget.  The diagnosis
# (`src/diag_fb_breakdown.py`, kineto): `keep='min'` recomputes the whole block forward including
# the FlashAttention forward, which is O(S^2); its recompute half alone is 461.7 ms/step at seq
# 16384, i.e. 121% of the entire 382 ms deficit.  `keep='attn'` stores FA's output (o_h + lse,
# 8.25 MiB/layer at seq 1024 batch 2) and never re-runs it.  This sweep measures what that costs
# and what it buys, against the same four competitors, one arm at a time.
#
# BOTH HALVES AT EVERY SHAPE.  A throughput fix that surrenders the memory story is not a fix, so
# every shape reports peak allocated, peak reserved and ms/step for every arm.
#
# ARMS
#   gc_manual_sdpa      LoRA + sdpa + torch.utils.checkpoint      (also a MACHINE-STATE CONTROL:
#                                                                  touches none of the code under
#                                                                  test)
#   fb_min_fnorm_sdpa   the OLD headline arm, keep pinned to 'min' -- the before number, measured
#                       in this session, and a regression check that `min` itself did not move
#   fb_attn_fnorm_sdpa  the NEW level, keep pinned to 'attn'
#   fb_auto_fnorm_sdpa  the POLICY (min below 4096 tokens, attn at/above), run at one short and
#                       one long shape to prove it resolves to the pinned arm's number to the byte
#   hyclora_flash_nc    HyC-LoRA's own fused FlashAttention layer, codecs bypassed -- their only
#                       valid exact arm and the one `run_longseq.sh` ships (hyclora_flash_variant
#                       section 9.8: the compressed FA arms do not train and must not be used)
#   unsloth_gc / unsloth_offload   separate process: `import unsloth` patches transformers, peft
#                       and torch.utils.checkpoint GLOBALLY.
#
# REGIME B ONLY (Liger FusedLinearCrossEntropy in every non-unsloth arm).  Unsloth ships its own
# fused linear CE and cannot carry HF's, so B is the only regime in which an unsloth row means
# anything, and it is the regime `holes_closed.md` adjudicates the 5% budget in.
#
# RESERVED MEMORY comes from DEDICATED SINGLE-ARM PROCESSES only (phase 6) -- `holes_closed.md`
# section 3.5 showed a 4426-vs-2646 MiB caching-allocator artifact from arm ordering.  The
# in-sweep reserved column is recorded but nothing rests on it.
#
# CONTROLS at the start, the middle and the end.  Established seq-1024 regime-B anchors:
# `baseline_sdpa`+FLCE 6156.39 MiB / ~231.3 ms, `gc_manual_sdpa`+FLCE 2593.80 / ~318.7.
# More than ~1% of drift on time means the box was not quiesced: discard the window and re-run.
set -u
cd /workspace/CompAct
source env/bin/activate
export HF_HOME=$(pwd)/data TORCH_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=
export CUDA_VISIBLE_DEVICES=1

OUT=results/hyclora/gap
LOG=temp/gap_logs
mkdir -p "$OUT" "$LOG"

# `hyclora_flash_nc` needs ~20 GiB at 8192 and ~38 GiB at 16384 on a 46 GiB card, so it goes LAST
# in any process it shares, and solo at 16384.
SMALL="gc_manual_sdpa,fb_min_fnorm_sdpa,fb_attn_fnorm_sdpa,hyclora_flash_nc"
HUGE="gc_manual_sdpa,fb_min_fnorm_sdpa,fb_attn_fnorm_sdpa"

run () {   # run <tag> <extra-args...>
    local tag=$1; shift
    echo "### $tag  $(date +%H:%M:%S)"
    timeout 10800 python -W ignore src/profile_unsloth.py "$@" > "$LOG/$tag.log" 2>&1
    local rc=$?
    grep -E "^===|peak\(alloc\)|fb_policy|ERROR|Error" "$LOG/$tag.log" | sed 's/^/    /' | cut -c1-165
    echo "    [exit=$rc]"
}

uns () {   # uns <tag> <seq> <steps>
    local tag=$1 s=$2 st=$3
    echo "### $tag  $(date +%H:%M:%S)"
    PYTHONPATH=temp/unsloth_pkgs timeout 10800 python -W ignore src/profile_unsloth.py \
        --arm unsloth_gc,unsloth_offload --unsloth_bf16_adapters \
        --seq "$s" --batch 2 --steps "$st" --interleave 2 \
        --out "$OUT/uns_seq$s.json" > "$LOG/$tag.log" 2>&1
    local rc=$?
    grep -E "^===|peak\(alloc\)|ERROR" "$LOG/$tag.log" | sed 's/^/    /' | cut -c1-165
    echo "    [exit=$rc]"
}

# ===================== phase 0: control -- is the box quiet? ==========================
run "ctrl_start" --arm baseline_sdpa,gc_manual_sdpa --flce --seq 1024 --batch 2 \
    --steps 12 --interleave 2 --out "$OUT/control_start.json"

# ===================== phase 1: the shapes that are outside budget ====================
run "B_seq8192"  --arm "$SMALL" --flce --seq 8192  --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/B_seq8192.json"
run "B_seq16384" --arm "$HUGE"  --flce --seq 16384 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/B_seq16384.json"
# 37.8 GiB = 82% of the card; it cannot be interleaved with anything.
run "B_seq16384_nc" --arm hyclora_flash_nc --flce --seq 16384 --batch 2 --steps 12 \
    --interleave 2 --out "$OUT/B_seq16384_nc.json"

# ===================== control: still quiet after the 16384 rows? =====================
run "ctrl_mid" --arm baseline_sdpa,gc_manual_sdpa --flce --seq 1024 --batch 2 \
    --steps 12 --interleave 1 --out "$OUT/control_mid.json"

# ===================== phase 2: the rest of the sweep =================================
run "B_seq4096" --arm "$SMALL" --flce --seq 4096 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/B_seq4096.json"
run "B_seq2048" --arm "$SMALL" --flce --seq 2048 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/B_seq2048.json"
run "B_seq1024" --arm "$SMALL" --flce --seq 1024 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/B_seq1024.json"

# ===================== phase 3: the POLICY reproduces the pinned arm ==================
# `fb_auto` must land on `min` at 1024 and on `attn` at 8192, and must reproduce that arm's
# allocated peak TO THE BYTE.  If it does not, the policy is doing something nobody asked for.
run "B_auto_seq1024" --arm fb_auto_fnorm_sdpa --flce --seq 1024 --batch 2 --steps 12 \
    --interleave 2 --out "$OUT/B_auto_seq1024.json"
run "B_auto_seq8192" --arm fb_auto_fnorm_sdpa --flce --seq 8192 --batch 2 --steps 12 \
    --interleave 2 --out "$OUT/B_auto_seq8192.json"

# ===================== phase 4: unsloth (their own fused CE == regime B) ==============
uns "C_seq1024"  1024  12
uns "C_seq2048"  2048  12
uns "C_seq4096"  4096  12
uns "C_seq8192"  8192  12
uns "C_seq16384" 16384 12

# ===================== phase 5: control -- bracket the whole sweep ====================
run "ctrl_end" --arm baseline_sdpa,gc_manual_sdpa,fb_min_fnorm_sdpa --flce --seq 1024 \
    --batch 2 --steps 12 --interleave 1 --out "$OUT/control_end.json"

# ===================== phase 6: RESERVED, from dedicated single-arm processes =========
# `--steps 4` because these rows are quoted for memory only; the peak protocol is unchanged.
for S in 1024 2048 4096 8192 16384; do
  for A in gc_manual_sdpa fb_min_fnorm_sdpa fb_attn_fnorm_sdpa hyclora_flash_nc; do
    run "solo_${A}_$S" --arm "$A" --flce --seq "$S" --batch 2 --steps 4 --interleave 1 \
        --out "$OUT/solo_${A}_seq$S.json"
  done
  PYTHONPATH=temp/unsloth_pkgs timeout 10800 python -W ignore src/profile_unsloth.py \
      --arm unsloth_gc --unsloth_bf16_adapters --seq "$S" --batch 2 --steps 4 --interleave 1 \
      --out "$OUT/solo_unsloth_gc_seq$S.json" > "$LOG/solo_unsloth_gc_$S.log" 2>&1
  PYTHONPATH=temp/unsloth_pkgs timeout 10800 python -W ignore src/profile_unsloth.py \
      --arm unsloth_offload --unsloth_bf16_adapters --seq "$S" --batch 2 --steps 4 --interleave 1 \
      --out "$OUT/solo_unsloth_offload_seq$S.json" > "$LOG/solo_unsloth_offload_$S.log" 2>&1
  echo "### solo unsloth seq=$S done  $(date +%H:%M:%S)"
done

echo "### DONE $(date +%H:%M:%S)"
