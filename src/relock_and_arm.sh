#!/usr/bin/env bash
# Wait for the in-flight timed job to exit, release `timed.lock`, then re-arm the opportunistic
# runner.  CONTEXT.md 45.8 / 39.7 / 42.7.  Exists so the lock and the runner do not depend on an
# agent still being alive at the end of a session -- 44.8 handed forward exactly that debt.
set -u
cd "$(dirname "$0")/.."
PID=${1:?usage: relock_and_arm.sh <pid-to-wait-for>}
LOG=results/gpu_watch/opportunistic.log
while [ -d "/proc/$PID" ]; do sleep 20; done
echo "$(date -u +%FT%TZ) relock_and_arm: pid $PID exited; releasing timed.lock" >> "$LOG"
rm -f results/gpu_watch/timed.lock
rm -f results/gpu_watch/opportunistic.stop
# Sized for a seq-4096 sweep (heaviest arm hyclora_flash_nc at 11.1 GB) plus headroom.
FFN_WINDOW_FREE_MIB=22000 nohup bash src/opportunistic_window.sh \
    "bash src/run_timing_window.sh {GPU} 2048 4096" >/dev/null 2>&1 &
echo "$(date -u +%FT%TZ) relock_and_arm: opportunistic runner re-armed" >> "$LOG"
