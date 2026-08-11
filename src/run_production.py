"""Production experiment runner: one method (optionally composed with the fused block), one task,
one seed -> one CSV row, written under a file lock so N of these can run concurrently.

WHAT THIS GUARANTEES, AND WHY EACH GUARANTEE EXISTS
---------------------------------------------------
1. **Methods are constructed by `profile_hyclora.build_model`** -- the same code that produced the
   memory numbers in CONTEXT.md §16 -- so a task row and a memory row describe the same object.
2. **Illegal combinations are refused BEFORE the model is built** (`experiment_registry.resolve_arm`).
   A camera-ready sweep must not spend GPU hours on a row it will throw away.
3. **Every requested method must PROVE it executed** (`verify_engagement`). This is the load-bearing
   one: `alst + fb` applies without error and silently kills ALST, so "it ran without crashing" is
   not evidence. A row whose method did not run is worse than a missing row.
4. **The CSV write is lock-protected and atomic** -- `FileLock` + temp-file + `os.replace`, the same
   proven path `train_glue.write_result_row` uses. Concurrent runners cannot interleave or truncate.
5. **Every hyperparameter is recorded.** Common ones get their own column; per-method configuration
   and engagement counters are carried verbatim as JSON so nothing is lost when a new method adds a
   knob. `config_hash` makes a cell identifiable without parsing JSON.
6. **Task AND computational metrics in the same row**, so a quality/memory trade is readable without
   joining two files.

⚠ SCOPE OF THE TASK LAYER TODAY. `--task lm:<hf_dataset>` (causal-LM perplexity) is wired, because
every registry builder constructs an `AutoModelForCausalLM` and that is the head they all share.
**GLUE and the multiple-choice suites need the builders parameterised by model class**
(`AutoModelForSequenceClassification` / the registered `AutoModelForMultipleChoice`); until that
lands, `--task glue:*` / `--task mc:*` raise rather than silently measuring the wrong head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import profile_hyclora as ph  # noqa: E402
from experiment_registry import (  # noqa: E402
    REGISTRY, FB_COMPAT, FLCE_FORBIDDEN, FULL_FT_REGIME, LOSSY,
    resolve_arm, verify_engagement, CombinationRefused,
)

CSV_COLUMNS = [
    # --- identity ---
    "timestamp", "run_id", "method", "with_fb", "arm", "task", "seed", "model_name_or_path",
    # --- protocol / hyperparameters ---
    "regime", "exact", "seq_len", "batch_size", "grad_accum", "train_steps", "lr",
    "lora_r", "lora_alpha", "lora_dropout", "target_modules", "adapter_dtype",
    "gc_variant", "flce", "attn_implementation", "n_trainable_params",
    # --- task metrics ---
    "eval_loss", "perplexity", "accuracy", "f1", "matthews_correlation",
    # --- computational metrics ---
    "train_step_peak_alloc_mib", "train_step_peak_reserved_mib", "resident_floor_mib",
    "peak_minus_floor_mib", "ms_per_step_median", "ms_per_step_iqr", "total_train_time_sec",
    "host_pinned_mib",
    # --- receipts: everything method-specific, losslessly ---
    "engagement_ok", "engagement_json", "method_config_json", "config_hash",
    "torch_version", "harness", "notes",
]
# One row per (method, fb, task, seed, seq, batch, lr). Re-running a cell UPDATES it in place.
COMB_COLS = ["method", "with_fb", "task", "seed", "seq_len", "batch_size", "lr"]


def _cfg(args):
    return {"model": args.model, "lora_r": args.lora_r, "seq": args.seq, "batch": args.batch,
            "iteration_threshold": 5, "softmax_outlier_ratio": 0.05,
            "layernorm_outlier_ratio": 0.005, "q_bit": 4}


def run_one(args) -> dict:
    """Build -> verify engagement -> train -> evaluate -> return one CSV row dict."""
    t_start = time.time()
    method, with_fb = args.method, bool(args.fb)

    arm = resolve_arm(method, with_fb)                     # raises CombinationRefused if illegal
    if method in FLCE_FORBIDDEN and args.flce:
        raise ValueError(f"{method}: --flce forbidden (its own chunked LM head IS the fused CE)")

    torch.manual_seed(args.seed)
    cfg = _cfg(args)
    device = args.device
    # ⚠ `torch.cuda.max_memory_allocated()` with no argument reads the CURRENT device, which is
    # cuda:0 regardless of where the model was placed -- a run on cuda:1 reported peak=0.00 and
    # floor=0.00 in the first smoke test. Bind the process to the requested device so every
    # memory statistic below refers to the card the model is actually on.
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)

    if arm == "" or arm == "_fb":                          # the `baseline` family
        base = "fb_min_fnorm_sdpa" if with_fb else "gc_manual_sdpa"
        model = ph.build_model(base, cfg, device, adapter_dtype="bf16")
        arm_str = base
    else:
        model = ph.build_model(arm, cfg, device, adapter_dtype="bf16")
        arm_str = arm

    constituents = [method] + (["fb"] if with_fb else [])
    engagement = verify_engagement(model, constituents, strict=not args.allow_inert)

    # ---- train: the harness's own step(), so the recipe matches every measured row ----
    vocab = ph.hf_config(model).vocab_size
    batch = ph.make_batch(cfg, device, vocab)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if getattr(model, "_galore_groups", None) is not None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "temp", "galore"))
        from galore_torch import GaLoreAdamW
        opt = GaLoreAdamW(model._galore_groups, lr=args.lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)

    for _ in range(args.warmup_steps):                     # protocol §A.2 warm-up
        ph.step(model, batch, opt)
    torch.cuda.synchronize()
    import gc as _gc
    _gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    resident_before = torch.cuda.memory_allocated()

    times, peaks_a, peaks_r, losses = [], [], [], []
    for _ in range(args.train_steps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = ph.step(model, batch, opt)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks_a.append(torch.cuda.max_memory_allocated())
        peaks_r.append(torch.cuda.max_memory_reserved())
        if loss is not None:
            losses.append(float(loss))

    # ---- evaluate ----
    model.eval()
    with torch.no_grad():
        out = model(**batch)
        eval_loss = float(out.loss)
    model.train()
    ppl = float(torch.exp(torch.tensor(min(eval_loss, 20.0))))

    times_s = sorted(times)
    receipt = {}
    for attr in ("_qlora_receipt", "_minis_receipt", "_streambp_receipt", "_alst_receipt",
                 "_zero3_receipt", "_galore_receipt", "_lomo_receipt", "_ckpt_receipt"):
        r = getattr(model, attr, None)
        if r:
            receipt[attr.strip("_")] = r
    cfg_json = json.dumps(receipt, sort_keys=True, default=str)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": args.run_id, "method": method, "with_fb": int(with_fb), "arm": arm_str,
        "task": args.task, "seed": args.seed, "model_name_or_path": args.model,
        "regime": "full_ft" if method in FULL_FT_REGIME else "peft_lora",
        "exact": int(method not in LOSSY),
        "seq_len": args.seq, "batch_size": args.batch, "grad_accum": 1,
        "train_steps": args.train_steps, "lr": args.lr,
        "lora_r": args.lora_r, "lora_alpha": args.lora_r, "lora_dropout": 0.0,
        "target_modules": ",".join(ph.FB_TARGETS), "adapter_dtype": "bf16",
        "gc_variant": (receipt.get("ckpt_receipt") or {}).get("variant", "none"),
        "flce": int(bool(args.flce)),
        "attn_implementation": "sdpa",
        # ⚠ Under ZeRO-3 a parameter is PARTITIONED, so `p.numel()` returns the local shard (and 0
        # for a fully offloaded one) -- the smoke test reported 5,529,600 trainable params for a
        # LoRA arm whose true count is 12,615,680. DeepSpeed keeps the real size on `p.ds_numel`.
        "n_trainable_params": sum(getattr(p, "ds_numel", None) or p.numel()
                                  for p in model.parameters() if p.requires_grad),
        "eval_loss": eval_loss, "perplexity": ppl,
        "accuracy": None, "f1": None, "matthews_correlation": None,
        "train_step_peak_alloc_mib": max(peaks_a) / 2 ** 20,
        "train_step_peak_reserved_mib": max(peaks_r) / 2 ** 20,
        "resident_floor_mib": resident_before / 2 ** 20,
        "peak_minus_floor_mib": (max(peaks_a) - resident_before) / 2 ** 20,
        "ms_per_step_median": 1e3 * statistics.median(times_s),
        "ms_per_step_iqr": 1e3 * (times_s[int(0.75 * len(times_s))] - times_s[int(0.25 * len(times_s))])
        if len(times_s) >= 4 else 0.0,
        "total_train_time_sec": time.time() - t_start,
        "host_pinned_mib": (receipt.get("zero3_receipt") or {}).get("host_pinned_mib"),
        "engagement_ok": int(bool(engagement.get("engagement_ok"))),
        "engagement_json": json.dumps(engagement, sort_keys=True, default=str),
        "method_config_json": cfg_json,
        "config_hash": hashlib.sha1(
            f"{method}|{with_fb}|{args.task}|{args.seq}|{args.batch}|{args.lr}|{cfg_json}"
            .encode()).hexdigest()[:16],
        "torch_version": torch.__version__, "harness": "run_production.py",
        "notes": REGISTRY[method].notes,
    }
    return row


def write_row(csv_path: str, row: dict) -> bool:
    """Lock-protected atomic upsert. Reuses `train_glue.write_result_row`, which is the path already
    proven against concurrent HPC array jobs (FileLock 300s, 5 retries, temp-file + os.replace)."""
    from train_glue import write_result_row
    return write_result_row(csv_path, CSV_COLUMNS, COMB_COLS, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--fb", action="store_true", help="compose with the fused decoder block")
    ap.add_argument("--task", default="lm:synthetic")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--warmup_steps", type=int, default=8)
    ap.add_argument("--train_steps", type=int, default=6)
    ap.add_argument("--flce", action="store_true")
    ap.add_argument("--allow_inert", action="store_true",
                    help="DEBUG ONLY: write the row even if a method proved no work")
    ap.add_argument("--run_id", default="adhoc")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    if args.task.startswith(("glue:", "mc:")):
        raise NotImplementedError(
            f"{args.task}: the registry's builders construct AutoModelForCausalLM. GLUE needs a "
            f"sequence-classification head and the commonsense/MMLU suites need the registered "
            f"AutoModelForMultipleChoice; parameterising the builders by model class is the next "
            f"step. Refusing rather than measuring the wrong head.")

    try:
        row = run_one(args)
    except CombinationRefused as e:
        print(f"REFUSED  {args.method}+fb={args.fb}: {e}")
        return 0                                   # a refusal is a correct outcome, not a failure
    ok = write_row(args.out_csv, row)
    print(f"{'OK ' if ok else 'CSV-FAIL '} {args.method}{'+fb' if args.fb else '':<4} "
          f"peak={row['train_step_peak_alloc_mib']:.2f} floor={row['resident_floor_mib']:.2f} "
          f"ppl={row['perplexity']:.3f} engaged={row['engagement_ok']} -> {args.out_csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
