#!/usr/bin/env python
"""Sequential driver for the padded/ragged workload sweep.

One arm at a time, one GPU, `train_glue.py` in a fresh process per cell, everything held
identical except the arm and the padding regime.  Writes one JSON per cell plus a rolling
summary JSON, so a killed sweep can be resumed (`--resume`).

See `llmdocs/trackers/padded_workloads.md`.  Measurement rules (binding):
  * `fair_comparison_protocol.md` §A.3 -- one arm at a time, same physical GPU, never two timed
    arms concurrently; nvidia-smi recorded either side of every measured window.
  * the sweep ALWAYS contains `lora_sdpa`, an arm that touches none of the code under test; if
    its train-window peak or train time moves by more than ~1% between repeats the block is void.
  * `train_peak_alloc_mib` / `train_peak_resv_mib` come from the training window only (epoch 0,
    before any eval pass) -- NOT the eval-clamped whole-run `peak_mem_mib` CSV column.

Usage
-----
  python src/run_padded_workloads.py --tasks cola --arms all --regimes ragged,padded \
      --seeds 41 --epochs 1 --tag calib --max_train_samples 256
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Adapter config is IDENTICAL in every arm and is dictated by the strictest competitor:
# HyC-LoRA's fused kernels compute x@A@B with no alpha/r factor, so lora_alpha must equal r,
# and they require all seven projections (src/hyclora/patch.py).  bf16 adapters everywhere.
COMMON = (
    f"--model_name_or_path {MODEL} --optimizer adamw-lora "
    "--lora_r 16 --lora_alpha 16 --lora_dropout 0.0 "
    "--dtype bfloat16 --grad_clipping 1.0 --lr_scheduler_type linear"
)

# --- arms -------------------------------------------------------------------------------------
# `lora_sdpa` and `lora_eager` execute NO code under test; they are the machine-state controls.
ARMS = {
    "fb_min_fnorm":     "--attn_implementation sdpa --fused_block min --fused_block_final_norm",
    # `_ad` = adapters cast to the run dtype, matching what the fused-block and HyC-LoRA arms do
    # by necessity.  These are the arms a memory table may quote.  The three without the suffix
    # keep PEFT's shipped fp32-adapter default and are reported separately, clearly labelled.
    "lora_sdpa_ad":     "--attn_implementation sdpa --cast_adapter_dtype",
    "lora_sdpa_gc_ad":  "--attn_implementation sdpa --gradient_checkpointing --cast_adapter_dtype",
    "lora_eager_ad":    "--attn_implementation eager --cast_adapter_dtype",
    "lora_sdpa":        "--attn_implementation sdpa",
    "lora_sdpa_gc":     "--attn_implementation sdpa --gradient_checkpointing",
    "lora_eager":       "--attn_implementation eager",
    "hyclora_ii_q4":    ("--attn_implementation eager --hyclora --cast_adapter_dtype "
                         "--hyclora_layer_type intra_inter --hyclora_q_bit 4"),
    "hyclora_ii_q2":    ("--attn_implementation eager --hyclora --cast_adapter_dtype "
                         "--hyclora_layer_type intra_inter --hyclora_q_bit 2"),
    # Their FlashAttention layer.  Expected to RAISE on any padded batch in this port
    # (src/hyclora/patch.py::_validate_flash_mask); run anyway so the failure is on the record.
    "hyclora_flash_nc": ("--attn_implementation sdpa --hyclora "
                         "--hyclora_layer_type intra_inter_flash --hyclora_no_compress"),
}
DEFAULT_ARMS = ["fb_min_fnorm", "lora_sdpa", "lora_sdpa_gc", "lora_eager", "hyclora_ii_q4"]

# --- tasks ------------------------------------------------------------------------------------
# max_length is the task's own padding target; batch size is held at 8 everywhere so the only
# thing that changes between tasks is the sequence-length distribution.
TASKS = {
    "cola": "--task_name cola --max_length 128 --learning_rate 5e-4",
    "sst2": "--task_name sst2 --max_length 128 --learning_rate 5e-4",
    "mrpc": "--task_name mrpc --max_length 128 --learning_rate 5e-4",
    "rte":  "--task_name rte  --max_length 256 --learning_rate 5e-4",
    "commonsense": "--task_name commonsense --max_length 256 --learning_rate 3e-4",
    # Reference point with ZERO padding waste, in this same harness: WikiText-2 is
    # concatenate-and-chunked into fixed-length blocks, so `--pad_to_max_length` is a no-op and
    # every token is valid. Run it at `ragged` only.  NOTE it is the causal-LM paradigm, so it
    # carries an LM head + fp32 cross-entropy stack that the classification tasks do not.
    "wikitext2_128": "--task_name wikitext2 --max_length 128 --learning_rate 2e-4",
    "wikitext2_256": "--task_name wikitext2 --max_length 256 --learning_rate 2e-4",
}
REGIMES = {"ragged": "", "padded": "--pad_to_max_length"}


def gpu_state() -> dict:
    """State of EVERY physical GPU, keyed by index, plus which one this sweep is pinned to.

    `nvidia-smi` is not affected by CUDA_VISIBLE_DEVICES, so row 0 is physical GPU 0 -- reading
    only row 0 while running on GPU 1 reports the co-tenant's state, which is worse than useless.
    """
    out = {"visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        for row in q.stdout.strip().splitlines():
            if not row.strip():
                continue
            v = [float(x) for x in row.split(",")]
            out[f"gpu{int(v[0])}"] = dict(zip(
                ["mem_used_mib", "util_pct", "sm_clock_mhz", "temp_c", "power_w"], v[1:]))
    except Exception as e:  # pragma: no cover
        out["error"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="cola")
    ap.add_argument("--arms", default="default")
    ap.add_argument("--regimes", default="ragged,padded")
    ap.add_argument("--seeds", default="41")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--total_batch_size", type=int, default=16)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results/padded")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose per-cell JSON already exists")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    arms = DEFAULT_ARMS if args.arms == "default" else (
        list(ARMS) if args.arms == "all" else args.arms.split(","))
    tasks = args.tasks.split(",")
    regimes = args.regimes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    outdir = REPO / args.outdir / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "_summary.json"
    summary = json.loads(summary_path.read_text()) if (args.resume and summary_path.exists()) else []

    env = dict(os.environ)
    env.update({
        "HF_HOME": str(REPO / "data"), "TORCH_HOME": str(REPO / "data"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_EVALUATE_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "",           # pinned empty for every arm
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
        "TOKENIZERS_PARALLELISM": "false",
    })

    # Seed-outer / arm-inner ordering: every arm is revisited once per seed, so slow thermal
    # drift cannot align with the arm variable and the control arm is re-measured 3x.
    cells = [(seed, task, regime, arm)
             for seed in seeds for task in tasks for regime in regimes for arm in arms]
    print(f"[driver] {len(cells)} cells -> {outdir}", flush=True)

    for i, (seed, task, regime, arm) in enumerate(cells, 1):
        cell = f"{task}__{regime}__{arm}__s{seed}"
        cell_json = outdir / f"{cell}.json"
        if args.resume and cell_json.exists():
            print(f"[{i}/{len(cells)}] SKIP (resume) {cell}", flush=True)
            continue

        cmd = (
            f"{sys.executable} -W ignore {REPO/'src'/'train_glue.py'} {COMMON} "
            f"{TASKS[task]} {REGIMES[regime]} {ARMS[arm]} "
            f"--num_train_epochs {args.epochs} "
            f"--per_device_train_batch_size {args.batch_size} "
            f"--per_device_eval_batch_size {args.batch_size} "
            f"--total_batch_size {args.total_batch_size} "
            f"--seeds {seed} --name pw_{args.tag}_{cell} "
            f"--results_csv {outdir/'runs.csv'} --run_json {cell_json}"
        )
        if args.max_train_samples:
            cmd += f" --max_train_samples {args.max_train_samples}"
        if args.max_eval_samples:
            cmd += f" --max_eval_samples {args.max_eval_samples}"

        rec = {"cell": cell, "task": task, "regime": regime, "arm": arm, "seed": seed,
               "cmd": cmd, "gpu_before": gpu_state()}
        print(f"[{i}/{len(cells)}] RUN  {cell}", flush=True)
        if args.dry_run:
            print("      " + cmd, flush=True)
            continue

        t0 = time.time()
        log_path = outdir / f"{cell}.log"
        with open(log_path, "w") as lf:
            p = subprocess.run(shlex.split(cmd), env=env, stdout=lf,
                               stderr=subprocess.STDOUT, timeout=args.timeout)
        rec["returncode"] = p.returncode
        rec["wall_sec"] = round(time.time() - t0, 2)
        rec["gpu_after"] = gpu_state()

        if p.returncode == 0 and cell_json.exists():
            r = json.loads(cell_json.read_text())["per_seed"][0]
            rec.update({
                "ok": True,
                "metric": r.get("best_metric_dict", {}),
                "train_peak_alloc_mib": r.get("train_peak_alloc_mib"),
                "train_peak_resv_mib": r.get("train_peak_resv_mib"),
                "train_epoch_times_sec": r.get("train_epoch_times_sec"),
                "epoch_train_loss_mean": r.get("epoch_train_loss_mean"),
                "n_micro_steps": r.get("n_micro_steps"),
                "n_opt_steps": r.get("n_opt_steps"),
                "honesty_counters": r.get("honesty_counters"),
                "dtype_receipt": r.get("dtype_receipt"),
                "token_stats": r.get("token_stats"),
                "whole_run_peak_mib": r.get("peak_mem_mib"),
            })
            ep = rec["train_epoch_times_sec"] or [float("nan")]
            n = max(1, rec["n_micro_steps"] // max(1, len(ep)))
            rec["ms_per_micro_step"] = round(1000.0 * ep[-1] / n, 3)
            print(f"      OK  peak={rec['train_peak_alloc_mib']:.2f} MiB "
                  f"resv={rec['train_peak_resv_mib']:.0f} "
                  f"{rec['ms_per_micro_step']:.1f} ms/micro-step  {rec['metric']}", flush=True)
        else:
            tail = log_path.read_text()[-4000:]
            err = [l for l in tail.splitlines()
                   if "Error" in l or "error" in l or "raise" in l or "Traceback" in l]
            rec.update({"ok": False, "error_tail": err[-8:] or tail.splitlines()[-8:]})
            print(f"      FAIL rc={p.returncode}: {rec['error_tail'][-1] if rec['error_tail'] else ''}",
                  flush=True)

        summary = [s for s in summary if s.get("cell") != cell] + [rec]
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"[driver] done -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
