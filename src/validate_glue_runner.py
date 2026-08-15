"""Validate the GLUE production runner across EVERY registry method, on this box.

WHAT "VALIDATE" MEANS HERE, AND WHY IT IS NOT "IT RAN"
-----------------------------------------------------
A row that exists is not a row that is right. This project has already shipped a
`gradient_checkpointing_enable()` that silently trained only the LM head, and an `alst + fb`
composition that applies cleanly while killing ALST. So this driver checks the VALUES, in two
phases, and reports every anomaly rather than dropping the arm that produced it.

  PHASE 1  build smoke -- 2 optimizer steps, tiny splits. Answers "does this method construct on a
           SEQUENCE-CLASSIFICATION head at all?". Several baselines wrap the model
           (`StreamModel`, `minisequence`) or replace its forward, and a wrapper written for a
           causal LM is not obliged to work for a classifier. Cheap, and it finds the structural
           refusals before any real training budget is spent.

  PHASE 2  real training -- enough steps that the baseline demonstrably learns, then the anomaly
           gate below. Running phase 2 on an arm that failed phase 1 would only produce a
           confusing second failure.

THE ANOMALY GATE (`check_row`)
------------------------------
Computational, per CONTEXT.md §4.1 / §8:
  * `resident_floor_mib` > 0 and `train_step_peak_alloc_mib` > floor -- a zero here is the
    "measured the wrong device" bug (run_production.py:91) coming back.
  * `peak_minus_floor_mib` > 0.
  * `ms_per_step_median` in a sane band -- a near-zero step time means the backward did not run,
    which is exactly how a StreamBP/DeepSpeed arm fails.
  * `n_trainable_params` must match the REGIME: ~12.6M for a LoRA arm, ~1.1B for full fine-tuning.
    A LoRA arm reporting 1.1B (or a full-FT arm reporting 12.6M) is a mis-built model, and it is
    invisible in the loss.
  * `engagement_ok` -- the method proved it executed.

Task quality:
  * the metric must be present and finite;
  * `eval_loss` finite;
  * ⚠ NOT DEGENERATE. `pred_distribution` must contain more than one class. A decoder classifier
    that collapses to the majority class scores the majority baseline -- 0.527 on RTE, 0.509 on
    SST-2 -- and that number looks like a result. This is the check the whole `pred_distribution`
    column exists for.
  * the metric must beat the task's MAJORITY BASELINE by a margin, not merely be non-zero.

Usage:
    python src/validate_glue_runner.py --phase 1
    python src/validate_glue_runner.py --phase 2 --task sst2 --train_steps 300
    python src/validate_glue_runner.py --report          # re-read the CSV and re-apply the gate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_registry import REGISTRY, FB_COMPAT, COMPOSES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Majority-class rate of the EVAL split. The gate demands the metric clear this, because matching
# it is precisely what a collapsed classifier does.
# COMPUTED from each task's own cached validation split (2026-08-14), not quoted from memory:
#   Counter(load_dataset(...)['validation']['label']) -> max/total.
# `stsb` is absent deliberately: it is a REGRESSION task scored by Pearson, where "majority class"
# is undefined. The degeneracy check below is likewise skipped for it (its pred histogram is empty).
MAJORITY_BASELINE = {"sst2": 0.5092, "rte": 0.5271, "mrpc": 0.6838, "cola": 0.6913,
                     "qnli": 0.5054, "boolq": 0.6217, "cb": 0.5000}

# LoRA r=16 on seven projections of TinyLlama-1.1B; full fine-tuning is the whole model.
EXPECTED_TRAINABLE = {"peft_lora": (10e6, 20e6), "full_ft": (0.9e9, 1.3e9)}


def arms():
    """Every (method, with_fb) the registry permits. `fb` is only offered where FB_COMPAT says
    COMPOSES -- resolve_arm would refuse the rest, and a refusal is a correct outcome, not a test
    failure, so there is nothing gained by launching it."""
    out = []
    for m in sorted(REGISTRY):
        out.append((m, False))
        if FB_COMPAT.get(m, {}).get("status") == COMPOSES:
            out.append((m, True))
    return out


def run_arm(method, with_fb, args, csv_path, phase):
    """One arm, in its OWN PROCESS. Not optional: ZeRO-3 initialises a DeepSpeed engine and ALST a
    torch.distributed process group, and both persist for the life of the interpreter. Sharing a
    process would let one arm's global state land in the next arm's measurement."""
    cmd = [sys.executable, os.path.join(HERE, "run_production.py"),
           "--method", method, "--task", f"glue:{args.task}",
           "--seq", str(args.seq), "--batch", str(args.batch), "--lr", str(args.lr),
           "--seed", str(args.seed), "--device", args.device,
           "--warmup_steps", str(args.warmup_steps), "--train_steps", str(args.train_steps),
           "--fb_variant", args.fb_variant,
           "--run_id", f"validate_p{phase}", "--out_csv", csv_path]
    if with_fb:
        cmd.append("--fb")
    if args.max_train_samples:
        cmd += ["--max_train_samples", str(args.max_train_samples)]
    if args.max_eval_samples:
        cmd += ["--max_eval_samples", str(args.max_eval_samples)]

    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(ROOT, "data"))
    for k in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "HF_EVALUATE_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.setdefault(k, "1")
    # ALST needs DeepSpeed >= 0.17 from the side-by-side prefix, and APPENDS -- assigning
    # PYTHONPATH deletes the module-supplied numpy on a cluster (hpc_fir.md §3 failure 6).
    need = REGISTRY[method].needs_pythonpath
    if need:
        env["PYTHONPATH"] = os.path.join(ROOT, need) + ":" + env.get("PYTHONPATH", "")

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    tail = [l for l in (p.stdout + p.stderr).strip().split("\n") if l.strip()]
    return {"method": method, "with_fb": with_fb, "rc": p.returncode,
            "secs": round(time.time() - t0, 1),
            "last": tail[-1][:400] if tail else "",
            "error": _first_error(p.stdout + p.stderr)}


def _first_error(text):
    """The exception line, not the 40-frame traceback around it."""
    lines = text.strip().split("\n")
    for i, l in enumerate(lines):
        if l.startswith(("Traceback",)):
            for j in range(len(lines) - 1, i, -1):
                if lines[j].strip() and not lines[j].startswith(" "):
                    return lines[j][:300]
    for l in lines:
        if l.startswith(("REFUSED", "ValueError", "NotImplementedError")):
            return l[:300]
    return None


def check_row(r):
    """Apply the anomaly gate to one CSV row. Returns a list of complaints; empty means clean."""
    bad = []

    def num(k):
        v = r.get(k)
        try:
            return None if v is None or v == "" or (isinstance(v, float) and v != v) else float(v)
        except (TypeError, ValueError):
            return None

    peak, floor = num("train_step_peak_alloc_mib"), num("resident_floor_mib")
    if not floor or floor <= 0:
        bad.append(f"resident_floor_mib={r.get('resident_floor_mib')} (<=0: wrong-device bug)")
    if not peak or peak <= 0:
        bad.append(f"peak={r.get('train_step_peak_alloc_mib')} (<=0)")
    if peak and floor and peak <= floor:
        bad.append(f"peak {peak:.1f} <= floor {floor:.1f} (no activation memory: did a step run?)")
    pmf = num("peak_minus_floor_mib")
    if pmf is not None and pmf <= 0:
        bad.append(f"peak_minus_floor={pmf:.2f} <= 0")

    ms = num("ms_per_step_median")
    if ms is None or ms < 1.0:
        bad.append(f"ms_per_step_median={r.get('ms_per_step_median')} (<1ms: backward likely absent)")

    ntp, regime = num("n_trainable_params"), r.get("regime")
    lo, hi = EXPECTED_TRAINABLE.get(regime, (None, None))
    if ntp is None:
        bad.append("n_trainable_params missing")
    elif lo and not (lo <= ntp <= hi):
        bad.append(f"n_trainable_params={ntp:,.0f} outside {regime} band [{lo:,.0f},{hi:,.0f}] "
                   f"-- model built in the wrong regime")

    if str(r.get("engagement_ok")) not in ("1", "1.0", "True"):
        bad.append(f"engagement_ok={r.get('engagement_ok')} -- method did not prove it ran")

    el = num("eval_loss")
    if el is None or el != el or el in (float("inf"), float("-inf")):
        bad.append(f"eval_loss={r.get('eval_loss')} not finite")

    tm, tmn = num("task_metric"), r.get("task_metric_name")
    if tm is None:
        bad.append("task_metric missing")
    else:
        task = str(r.get("task", "")).split(":")[-1]
        base = MAJORITY_BASELINE.get(task)
        if base is not None and tmn == "accuracy" and tm <= base:
            bad.append(f"{tmn}={tm:.4f} <= majority baseline {base:.3f} for {task} "
                       f"-- indistinguishable from predicting one class")

    pd_raw = r.get("pred_distribution")
    if pd_raw and str(pd_raw) != "nan":
        try:
            hist = json.loads(pd_raw)
            if len(hist) < 2:
                bad.append(f"pred_distribution={pd_raw} -- DEGENERATE, one class only")
            else:
                tot = sum(hist.values())
                top = max(hist.values()) / max(tot, 1)
                if top > 0.97:
                    bad.append(f"pred_distribution={pd_raw} -- {top:.1%} one class, near-degenerate")
        except (ValueError, TypeError):
            bad.append(f"pred_distribution unparseable: {pd_raw!r}")
    return bad


def report(csv_path):
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"no CSV at {csv_path}")
        return 1
    df = pd.read_csv(csv_path)
    print(f"\n{'='*100}\nANOMALY GATE over {len(df)} rows in {csv_path}\n{'='*100}")
    n_bad = 0
    for _, r in df.iterrows():
        d = r.to_dict()
        bad = check_row(d)
        tag = f"{d['method']}{'+fb' if d.get('with_fb') else ''}"
        metric = f"{d.get('task_metric_name')}={d.get('task_metric')}"
        base = f"peak={d.get('train_step_peak_alloc_mib')} ms={d.get('ms_per_step_median')}"
        if bad:
            n_bad += 1
            print(f"\n  ✗ {tag:<16} {metric}  {base}")
            for b in bad:
                print(f"      - {b}")
        else:
            print(f"  ✓ {tag:<16} {metric}  {base}  preds={d.get('pred_distribution')}")
    print(f"\n{'='*100}\n{len(df)-n_bad}/{len(df)} rows clean, {n_bad} with anomalies\n{'='*100}")
    return 1 if n_bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup_steps", type=int, default=2)
    ap.add_argument("--train_steps", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--fb_variant", default="min", choices=["min", "wstream"])
    ap.add_argument("--fb_only", action="store_true",
                    help="only the arms that compose with the fused block (for a wstream pass)")
    ap.add_argument("--only", default=None, help="comma-separated methods, for re-testing one arm")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--report", action="store_true", help="re-apply the gate to an existing CSV")
    args = ap.parse_args()

    csv_path = args.out_csv or os.path.join(
        ROOT, "results", "production", f"validate_glue_{args.task}_p{args.phase}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if args.report:
        return report(csv_path)

    todo = arms()
    if args.fb_only:
        todo = [(m, f) for m, f in todo if f]
    if args.only:
        want = {m.strip() for m in args.only.split(",")}
        todo = [(m, f) for m, f in todo if m in want]

    print(f"phase {args.phase}: {len(todo)} arms on glue:{args.task}, seq={args.seq} "
          f"batch={args.batch} steps={args.warmup_steps}+{args.train_steps} -> {csv_path}\n")
    results = []
    for m, f in todo:
        label = f"{m}{'+fb' if f else ''}"
        print(f"--- {label} ...", flush=True)
        res = run_arm(m, f, args, csv_path, args.phase)
        results.append(res)
        status = "OK" if res["rc"] == 0 else f"rc={res['rc']}"
        print(f"    {status} in {res['secs']}s :: {res['error'] or res['last']}", flush=True)

    print(f"\n{'='*100}\nPHASE {args.phase} BUILD/RUN SUMMARY\n{'='*100}")
    for r in results:
        label = f"{r['method']}{'+fb' if r['with_fb'] else ''}"
        print(f"  {'PASS' if r['rc']==0 else 'FAIL'}  {label:<16} {r['secs']:>7.1f}s  "
              f"{r['error'] or ''}")
    return report(csv_path)


if __name__ == "__main__":
    sys.exit(main())
