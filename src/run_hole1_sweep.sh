#!/usr/bin/env bash
# HOLE 1 -- the sequence-length crossover, re-measured against the VALID HyC-LoRA arm.
#
# `fused_block_verification.md` section 7 measured the crossover against `hyclora_flash_q2`, which
# `hyclora_flash_variant.md` section 9 has since shown does not train.  That result is void.  This
# script re-runs the sweep against `hyclora_flash_nc` -- upstream's shipped `layer_type=baseline`,
# exact, and what `run_longseq.sh` actually drives at seq 8192 -- plus unsloth's two checkpointing
# configurations.
#
# TWO CROSS-ENTROPY REGIMES, both measured, because the CE term scales with S and would otherwise
# dominate every long-sequence row:
#   A  stock HF cross-entropy in every non-unsloth arm  (the regime the void section-7 sweep used;
#      anchors against the established seq-1024 numbers)
#   B  Liger FusedLinearCrossEntropy in every non-unsloth arm (unsloth ships its own fused CE, so
#      B is the only regime in which an unsloth row means anything)
#
# CONTROL ARMS (measurement rule): `baseline_sdpa` and `gc_manual_sdpa` touch none of the code
# under test.  Established seq-1024 values -- A: 6952.40 MiB / ~230.2 ms and 3256.93 / ~317.7;
# B: 6156.39 / ~230.8 and 2593.80 / ~318.2.  A deviation of more than ~1% on time means the box
# was not quiesced and the window must be discarded.  A third-party tenant took 24 GiB of GPU 0
# during this session, so the controls are re-run between phases, not just at the ends.
#
# ORDERING is by value, so that an interrupted run still yields the load-bearing shapes:
# seq 1024 (the anchor) -> 8192 (the shape `run_longseq.sh` ships) -> 4096 -> 2048 -> 16384.
#
# One arm at a time, one GPU, arms interleaved A,B,C,D,A,B,C,D inside each process.
set -u
cd /workspace/CompAct
source env/bin/activate
export HF_HOME=$(pwd)/data TORCH_HOME=$(pwd)/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=
export CUDA_VISIBLE_DEVICES=0

OUT=results/hyclora/holes
LOG=temp/hole_logs
mkdir -p "$OUT" "$LOG"

# `baseline_sdpa` needs ~40 GiB at seq 8192 and ~78 GiB at 16384 on a 46 GiB card, and
# `hyclora_flash_nc` ~50 GiB at 16384, so those cells are expected to OOM and are recorded as
# such.  Arm order inside a process puts the OOM-prone arm LAST at the shapes where it is at risk.
SMALL="baseline_sdpa,gc_manual_sdpa,fb_min_fnorm_sdpa,hyclora_flash_nc"
LARGE="gc_manual_sdpa,fb_min_fnorm_sdpa,hyclora_flash_nc,baseline_sdpa"
HUGE="gc_manual_sdpa,fb_min_fnorm_sdpa"

run () {   # run <tag> <extra-args...>
    local tag=$1; shift
    echo "### $tag  $(date +%H:%M:%S)"
    timeout 7200 python -W ignore src/profile_unsloth.py "$@" > "$LOG/$tag.log" 2>&1
    local rc=$?
    grep -E "^===|peak\(alloc\)|ERROR" "$LOG/$tag.log" | sed 's/^/    /' | cut -c1-150
    echo "    [exit=$rc]"
}

uns () {   # uns <tag> <seq> <steps>
    local tag=$1 s=$2 st=$3
    echo "### $tag  $(date +%H:%M:%S)"
    PYTHONPATH=temp/unsloth_pkgs timeout 7200 python -W ignore src/profile_unsloth.py \
        --arm unsloth_gc,unsloth_offload --unsloth_bf16_adapters \
        --seq "$s" --batch 2 --steps "$st" --interleave 2 \
        --out "$OUT/sweepC_unsloth_seq$s.json" > "$LOG/$tag.log" 2>&1
    local rc=$?
    grep -E "^===|peak\(alloc\)|ERROR" "$LOG/$tag.log" | sed 's/^/    /' | cut -c1-150
    echo "    [exit=$rc]"
}

# ============================ phase 1: the anchor, both regimes, seq 1024 ============================
run "A_seq1024" --arm "$SMALL" --seq 1024 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/sweepA_stockce_seq1024.json"
run "B_seq1024" --arm "$SMALL" --flce --seq 1024 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/sweepB_flce_seq1024.json"

# ============================ phase 2: seq 8192 -- the shape run_longseq.sh ships ====================
run "A_seq8192" --arm "$LARGE" --seq 8192 --batch 2 --steps 10 --interleave 2 \
    --out "$OUT/sweepA_stockce_seq8192.json"
run "B_seq8192" --arm "$LARGE" --flce --seq 8192 --batch 2 --steps 10 --interleave 2 \
    --out "$OUT/sweepB_flce_seq8192.json"

# ============================ phase 3: fill in 4096 and 2048 ========================================
run "A_seq4096" --arm "$LARGE" --seq 4096 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/sweepA_stockce_seq4096.json"
run "B_seq4096" --arm "$LARGE" --flce --seq 4096 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/sweepB_flce_seq4096.json"
run "A_seq2048" --arm "$SMALL" --seq 2048 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/sweepA_stockce_seq2048.json"
run "B_seq2048" --arm "$SMALL" --flce --seq 2048 --batch 2 --steps 12 --interleave 2 \
    --out "$OUT/sweepB_flce_seq2048.json"

# ============================ control: is the box still quiet? ======================================
run "ctrl_mid" --arm baseline_sdpa,gc_manual_sdpa --seq 1024 --batch 2 --steps 12 --interleave 1 \
    --out "$OUT/control_mid.json"

# ============================ phase 4: unsloth (their own fused CE == regime B) =====================
# Separate process: `import unsloth` patches transformers/peft/torch.utils.checkpoint GLOBALLY, so
# no non-unsloth arm may share a process with it.  `--unsloth_bf16_adapters` is the matched-adapter
# configuration (protocol B.7); it is also the one that FAVOURS unsloth (-72.19 MiB of floor).
uns "C_seq1024" 1024 12
uns "C_seq8192" 8192 10
uns "C_seq4096" 4096 12
uns "C_seq2048" 2048 12
uns "C_seq16384" 16384 8

# ============================ phase 5: seq 16384, the arms that fit =================================
run "B_seq16384" --arm "$HUGE" --flce --seq 16384 --batch 2 --steps 8 --interleave 2 \
    --out "$OUT/sweepB_flce_seq16384.json"
run "A_seq16384" --arm "$HUGE" --seq 16384 --batch 2 --steps 8 --interleave 2 \
    --out "$OUT/sweepA_stockce_seq16384.json"
run "B_seq16384_nc" --arm hyclora_flash_nc --flce --seq 16384 --batch 2 --steps 8 --interleave 1 \
    --out "$OUT/sweepB_flce_seq16384_nc.json"

# ============================ control: bracket the whole sweep ======================================
run "ctrl_end" --arm baseline_sdpa,gc_manual_sdpa,fb_min_fnorm_sdpa --seq 1024 --batch 2 \
    --steps 12 --interleave 1 --out "$OUT/control_end.json"

echo "### DONE $(date +%H:%M:%S)"
