#!/usr/bin/env bash
# Append one row per GPU per PERIOD seconds to results/gpu_watch/util_log.csv.
# CONTEXT.md 39.8 / 44.8 / 45.8.
#
# WHY THIS EXISTS AS A FILE.  This watcher has been started inline (`while true; do ... done &`)
# three times and has died with its session three times -- 44.8 and 45.8 both hand forward "the
# watcher is dead" as an operational debt, and 45.8's restart was itself dead within an hour
# (last row 2026-08-09T17:45:55Z).  An inline loop is a child of the agent's shell; a setsid'd
# script is not.  T-7 also wants any long-running job to be resumable from a fresh context.
#
# Row format (unchanged from 45.8):  timestamp,index,util,used,total
# NOTE the pre-45.8 file had BOTH cards on one line; anything parsing this must handle both.
# `src/opportunistic_window.sh` does NOT read this file (it polls nvidia-smi directly).
#
# Usage:
#   setsid nohup bash src/gpu_util_watch.sh >/dev/null 2>&1 &     # survives the session
#   FFN_WATCH_PERIOD=30 bash src/gpu_util_watch.sh                # faster sampling
# Stop it with:  touch results/gpu_watch/util_watch.stop
set -u
cd "$(dirname "$0")/.."
PERIOD=${FFN_WATCH_PERIOD:-60}
OUT=results/gpu_watch/util_log.csv
STOP=results/gpu_watch/util_watch.stop
mkdir -p results/gpu_watch
rm -f "$STOP"
while [ ! -e "$STOP" ]; do
    ts=$(date -u +%FT%TZ)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r line; do echo "$ts,$line"; done >> "$OUT"
    sleep "$PERIOD"
done
echo "$(date -u +%FT%TZ) gpu_util_watch: stopped via $STOP" >> results/gpu_watch/opportunistic.log
