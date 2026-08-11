#!/usr/bin/env bash
# Fire a timed measurement the moment a GPU goes quiet.  CONTEXT.md 37.5 item 6 / 38.4.
#
# WHY THIS EXISTS.  The mandate of CONTEXT.md 37 is a THROUGHPUT mandate, and 17.6 h of
# `results/gpu_watch/util_log.csv` (1059 samples, 2026-08-05) contains ZERO minutes with both A40s
# below 50% and only ~6 minutes where either card did.  That is a 0.6% duty cycle.  Two consecutive
# sessions (36.4, 38.4) ended with no throughput number at all because the window never opened while
# an agent was watching.  Waiting at the keyboard is not a plan; this waits instead.
#
# WHAT IT DOES.  Polls `nvidia-smi` every PERIOD seconds.  When ONE card has been below THRESH%
# utilisation for NEED consecutive samples AND has FREE_MIB free, it takes a lock and runs CMD
# pinned to that card, then exits.  The lock is the protocol guard: 37.7 forbids two timed arms
# running concurrently, including against your own background jobs, so anything else that times
# must take `results/gpu_watch/timed.lock` too.
#
# The adjudicator (`src/agg_timing_window.py`) still decides whether the window was valid -- this
# script only decides when to TRY.  A window that opens and then closes mid-sweep is caught there,
# by the control arm, exactly as protocol A.3.4 requires.  Firing on a false positive costs a
# discarded window; missing a true one costs a session.
#
# Usage:
#   bash src/opportunistic_window.sh                      # default sweep, seqs 2048 4096
#   bash src/opportunistic_window.sh "bash src/run_timing_window.sh {GPU} 8192 16384"
#   FFN_WINDOW_UTIL=15 FFN_WINDOW_SAMPLES=6 bash src/opportunistic_window.sh
# `{GPU}` in CMD is replaced by the index of the card that went quiet.
#
# Stop it with:  touch results/gpu_watch/opportunistic.stop
set -u
cd "$(dirname "$0")/.."

THRESH=${FFN_WINDOW_UTIL:-25}          # percent utilisation that counts as "quiet"
NEED=${FFN_WINDOW_SAMPLES:-4}          # consecutive quiet samples before firing
PERIOD=${FFN_WINDOW_PERIOD:-30}        # seconds between polls
# Free MiB the card must have.  Sized for the DEFAULT sweep (seqs 2048/4096, whose heaviest arm is
# `hyclora_flash_nc` at 11.1 GB allocated plus the control and the floor).  A foreign tenant that
# goes idle without exiting still holds its allocation -- both cards currently show ~27 GB free with
# 18 GB held by someone else -- so this must NOT be set to "the whole card" or it will never fire.
# Raise it to ~40000 for an 8192/16384 sweep, where `hyclora_flash_nc` alone needs 37.8 GB.
FREE_MIB=${FFN_WINDOW_FREE_MIB:-22000}
MAXWAIT=${FFN_WINDOW_MAXWAIT:-0}       # seconds to wait before giving up; 0 = forever

WATCH_DIR=results/gpu_watch
LOG="$WATCH_DIR/opportunistic.log"
LOCK="$WATCH_DIR/timed.lock"
STOP="$WATCH_DIR/opportunistic.stop"
mkdir -p "$WATCH_DIR"

CMD=${1:-"bash src/run_timing_window.sh {GPU} 2048 4096"}

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

say "START watching: thresh=${THRESH}% need=${NEED} period=${PERIOD}s free>=${FREE_MIB}MiB"
say "START cmd: $CMD"
rm -f "$STOP"

declare -A streak
streak[0]=0; streak[1]=0
elapsed=0

while :; do
  if [ -f "$STOP" ]; then say "STOP file present -- exiting without firing"; exit 0; fi
  if [ "$MAXWAIT" -gt 0 ] && [ "$elapsed" -ge "$MAXWAIT" ]; then
    say "MAXWAIT ${MAXWAIT}s reached -- exiting without firing"; exit 3
  fi

  # index, util%, free MiB -- one line per card.
  while IFS=, read -r idx util free; do
    idx=$(echo "$idx" | tr -d ' '); util=$(echo "$util" | tr -d ' %'); free=$(echo "$free" | tr -d ' MiB')
    [ -z "${idx:-}" ] && continue
    if [ "$util" -lt "$THRESH" ] && [ "$free" -ge "$FREE_MIB" ]; then
      streak[$idx]=$(( ${streak[$idx]} + 1 ))
    else
      streak[$idx]=0
    fi

    if [ "${streak[$idx]}" -ge "$NEED" ]; then
      say "WINDOW OPEN on GPU $idx (util=${util}% free=${free}MiB, ${streak[$idx]} consecutive quiet samples)"
      # 37.7: never two timed arms at once.  `set -o noclobber` makes this an atomic test-and-set.
      if ( set -o noclobber; echo "$$ gpu$idx $(date -u +%FT%TZ)" > "$LOCK" ) 2>/dev/null; then
        trap 'rm -f "$LOCK"' EXIT
        RUN=${CMD//\{GPU\}/$idx}
        say "FIRING: $RUN"
        # nvidia-smi ordering is PCI-bus order; run_timing_window.sh pins with CUDA_VISIBLE_DEVICES,
        # which defaults to the same order, so the index carries through.
        eval "$RUN" >> "$LOG" 2>&1
        rc=$?
        say "FINISHED rc=$rc -- adjudicate with: PYTHONPATH=src python src/agg_timing_window.py"
        rm -f "$LOCK"; trap - EXIT
        exit "$rc"
      else
        say "LOCK held by $(cat "$LOCK" 2>/dev/null) -- another timed job is running; standing down"
        streak[$idx]=0
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader)

  sleep "$PERIOD"
  elapsed=$(( elapsed + PERIOD ))
done
