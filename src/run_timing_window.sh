#!/usr/bin/env bash
# G12 -- the timing window (llmdocs/trackers/baseline_completeness.md section 5.3, CONTEXT.md 34.9.1).
#
# WHY THIS EXISTS.  Every ms/step in CONTEXT.md 33.4 for OUR arms predates the in-place GLU kernel
# of 33.6(b), and every competitor time is the best of its own archived run, measured in a
# different window on a different day.  No single window has ever contained every arm with the
# current code.  33.5 projects that the kernel turns seq 2048 from +0.52% into a WIN against
# `hyclora_flash_nc`; that projection has never been measured.
#
# PROTOCOL (fair_comparison_protocol.md A.3, binding):
#   * ONE arm at a time, ONE GPU, never two timed processes concurrently.
#   * `baseline_sdpa` rides along as the control at start / middle / end of every window -- it
#     touches none of the code under test.  It needs 33.9 GB at 8192, so at 8192/16384 the control
#     falls back to `gc_manual_sdpa`, which touches none of our block either.  Discard the window
#     if the control moves more than ~1% across reps.
#   * `--interleave 3` gives each arm three reps in rotation, so a drift that develops mid-window
#     hits every arm equally instead of penalising whichever arm ran last.
#   * unsloth needs its own PYTHONPATH and therefore its own process.
#   * liger_gc_sdpa is LAST and ALONE.  Its patcher is a global monkey-patch with no unpatch
#     (34.6): any arm measured after it in the same process silently inherits Liger's kernels.
#
# Usage:  bash src/run_timing_window.sh [GPU] [SEQ...]      default: GPU 0, seqs 2048 4096
#         FFN_WINDOW_REPS=8 FFN_WINDOW_STEPS=16 bash src/run_timing_window.sh 0 4096
set -u
cd "$(dirname "$0")/.."
source env/bin/activate

GPU="${1:-0}"; shift || true
SEQS="${*:-2048 4096}"
# REPS / STEPS are overridable so a cell whose effect is smaller than the estimator's spread can be
# attacked with more reps rather than by waiting for a quieter box.  CONTEXT.md 45.7a: seq 4096 has
# now returned "NOT RESOLVED, spread exceeds the effect" in two separate windows, one of them on an
# IDLE box -- so the binding constraint there is estimator variance, not contention.  Raising
# --interleave costs wall-clock linearly and shrinks the paired spread.  Defaults reproduce the
# historical protocol exactly, so every archived window remains comparable.
REPS=${FFN_WINDOW_REPS:-3}
STEPS=${FFN_WINDOW_STEPS:-12}
OUT=results/hyclora/timing
mkdir -p "$OUT"

for S in $SEQS; do
  # The control arm has to fit.  baseline_sdpa stores every activation and is the reason G4 rows
  # OOM above 4096; above that the control is the lightest arm that is still outside our code.
  if [ "$S" -le 4096 ]; then CTRL=baseline_sdpa; else CTRL=gc_manual_sdpa; fi

  echo "############ seq $S -- window 1/3: ours + HyC-LoRA + stock, interleaved (control $CTRL)"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python src/profile_unsloth.py --device cuda:0 \
      --arm "$CTRL,fb_min_fnorm_sdpa,hyclora_flash_nc,fb_attn_fnorm_sdpa,gc_manual_sdpa" \
      --flce --seq "$S" --batch 2 --steps "$STEPS" --interleave "$REPS" \
      --out "$OUT/window_main_seq$S.json" 2>&1 | tail -30

  echo "############ seq $S -- window 2/3: unsloth (own PYTHONPATH, own process)"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=temp/unsloth_pkgs:src python src/profile_unsloth.py \
      --device cuda:0 --arm unsloth_gc --seq "$S" --batch 2 --steps "$STEPS" --interleave "$REPS" \
      --unsloth_bf16_adapters --out "$OUT/window_unsloth_seq$S.json" 2>&1 | tail -20

  echo "############ seq $S -- window 3/3: liger_gc_sdpa (LAST, ALONE -- global patch, no unpatch)"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python src/profile_unsloth.py --device cuda:0 \
      --arm liger_gc_sdpa --seq "$S" --batch 2 --steps "$STEPS" --interleave "$REPS" \
      --out "$OUT/window_liger_seq$S.json" 2>&1 | tail -20
done

echo "############ DONE -- windows under $OUT"
