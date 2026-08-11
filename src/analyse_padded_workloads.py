#!/usr/bin/env python
"""Turn the per-cell JSONs written by `run_padded_workloads.py` into the tables of
`llmdocs/trackers/padded_workloads.md`.

Nothing here touches the GPU; it is pure aggregation.  Two things it deliberately does NOT do:
  * average memory across seeds -- it reports the per-seed spread, because on ragged batches the
    peak depends on which long example landed in which batch (protocol A.4);
  * use `avg_step_time` (optimizer.step() only) or `peak_mem_mib` (eval-clamped) for anything.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

METRIC_KEY = {
    "cola": "matthews_correlation", "sst2": "accuracy", "mrpc": "accuracy",
    "rte": "accuracy", "commonsense": "accuracy",
    "wikitext2_128": "perplexity", "wikitext2_256": "perplexity",
}
ARM_ORDER = ["fb_min_fnorm", "lora_sdpa_ad", "lora_sdpa_gc_ad", "lora_eager_ad",
             "hyclora_ii_q4", "hyclora_ii_q2", "hyclora_flash_nc",
             "lora_sdpa", "lora_sdpa_gc", "lora_eager"]
CONTROL_ARM = "lora_sdpa_ad"


def load(tags):
    recs = []
    for tag in tags:
        d = REPO / "results" / "padded" / tag
        p = d / "_summary.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            r["tag"] = tag
            cj = d / f"{r['cell']}.json"
            if cj.exists():                      # fields the driver does not copy
                s = json.loads(cj.read_text())["per_seed"][0]
                for k in ("param_mem_mib", "opt_mem_mib", "runtime_mem_mib", "first_micro_loss"):
                    r[k] = s.get(k)
            recs.append(r)
    return recs


def ms_per_micro(r):
    """Steady-state ms per micro-step: the LAST epoch's train-only wall clock.

    Epoch 0 carries first-step warm-up, HyC-LoRA's calibration window and (for the eager HyC-LoRA
    layers on ragged batches) a Triton recompile for every new sequence length; the last epoch
    carries none of them.
    """
    ep = r.get("train_epoch_times_sec") or []
    n = r.get("n_micro_steps") or 0
    if not ep or not n:
        return float("nan")
    return 1000.0 * ep[-1] / (n / len(ep))


def fmt(v, nd=1):
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="g1,g2,g3")
    args = ap.parse_args()
    recs = load(args.tags.split(","))

    by = defaultdict(list)
    for r in recs:
        by[(r["task"], r["regime"], r["arm"])].append(r)

    tasks = sorted({r["task"] for r in recs})
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in recs)]

    # ---------------- token / padding-waste statistics ----------------
    print("\n## Padding-waste statistics (train split, epoch 0, seed 41)\n")
    print("| task | regime | samples | valid tok/sample mean | p50 | p95 | max | "
          "valid tok/batch mean | p50 | p95 | max | padded tok/batch mean | max | pad waste |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        for reg in ("ragged", "padded"):
            cand = [r for r in recs
                    if r["task"] == t and r["regime"] == reg and r.get("ok") and r.get("token_stats")
                    and r["seed"] == 41]
            if not cand:
                continue
            s = cand[0]["token_stats"]
            print(f"| {t} | {reg} | {s['n_samples']} | {s['sample_valid_mean']} | "
                  f"{s['sample_valid_p50']:.0f} | {s['sample_valid_p95']:.0f} | {s['sample_valid_max']:.0f} | "
                  f"{s['batch_valid_mean']:.1f} | {s['batch_valid_p50']:.0f} | {s['batch_valid_p95']:.0f} | "
                  f"{s['batch_valid_max']:.0f} | {s['batch_padded_mean']:.1f} | {s['batch_padded_max']:.0f} | "
                  f"**{100*s['pad_waste_frac']:.1f}%** |")

    # ---------------- main per-task table ----------------
    for t in tasks:
        mk = METRIC_KEY.get(t, "accuracy")
        print(f"\n## {t}\n")
        print("| arm | regime | peak alloc MiB (med [min,max]) | peak resv MiB | ms/micro-step "
              "(med [min,max]) | metric med [min,max] | n seeds |")
        print("|---|---|---|---|---|---|---|")
        for a in arms:
            for reg in ("ragged", "padded"):
                rs = [r for r in by[(t, reg, a)] if r.get("ok")]
                fails = [r for r in by[(t, reg, a)] if not r.get("ok")]
                if not rs:
                    if fails:
                        print(f"| `{a}` | {reg} | **RAISES** | — | — | — | {len(fails)} |")
                    continue
                pk = sorted(r["train_peak_alloc_mib"] for r in rs)
                rv = sorted(r["train_peak_resv_mib"] for r in rs)
                ms = sorted(ms_per_micro(r) for r in rs)
                mt = sorted(r["metric"].get(mk, float("nan")) for r in rs)
                print(f"| `{a}` | {reg} | {statistics.median(pk):.2f} [{pk[0]:.2f}, {pk[-1]:.2f}] "
                      f"| {statistics.median(rv):.0f} [{rv[0]:.0f}, {rv[-1]:.0f}] "
                      f"| {statistics.median(ms):.1f} [{ms[0]:.1f}, {ms[-1]:.1f}] "
                      f"| {statistics.median(mt):.4f} [{mt[0]:.4f}, {mt[-1]:.4f}] | {len(rs)} |")

        # ratio block
        print(f"\n**{t}: padded / ragged, per arm (the whole point)**\n")
        print("| arm | ragged peak | padded peak | padded/ragged | vs `fb_min_fnorm` ragged | "
              "vs `fb_min_fnorm` padded |")
        print("|---|---|---|---|---|---|")
        fb = {reg: statistics.median([r["train_peak_alloc_mib"] for r in by[(t, reg, "fb_min_fnorm")]
                                      if r.get("ok")] or [float("nan")])
              for reg in ("ragged", "padded")}
        for a in arms:
            g = {reg: [r["train_peak_alloc_mib"] for r in by[(t, reg, a)] if r.get("ok")]
                 for reg in ("ragged", "padded")}
            if not g["ragged"] or not g["padded"]:
                continue
            rg, pd_ = statistics.median(g["ragged"]), statistics.median(g["padded"])
            print(f"| `{a}` | {rg:.2f} | {pd_:.2f} | **{pd_/rg:.3f}x** | "
                  f"{rg/fb['ragged']:.3f}x | **{pd_/fb['padded']:.3f}x** |")

    # ---------------- throughput in useful tokens ----------------
    print("\n## Throughput in VALID tokens/s (seed 41, last epoch, train only)\n")
    print("A padded workload makes the two throughput denominators diverge: `padded tok/s` is "
          "what the GPU chewed, `valid tok/s` is what the user paid for.\n")
    print("| task | arm | regime | ms/micro-step | padded tok/s | **valid tok/s** | "
          "valid tok/s vs ragged |")
    print("|---|---|---|---|---|---|---|")
    for t in tasks:
        for a in arms:
            base = None
            for reg in ("ragged", "padded"):
                rs = [r for r in by[(t, reg, a)] if r.get("ok") and r["seed"] == 41]
                if not rs:
                    continue
                r = rs[0]
                s = r.get("token_stats") or {}
                ep = r.get("train_epoch_times_sec") or []
                if not s or not ep:
                    continue
                secs = ep[-1]
                vt, pt = s["total_valid_tokens"] / secs, s["total_padded_tokens"] / secs
                if reg == "ragged":
                    base = vt
                rel = f"{vt/base:.3f}x" if base else "—"
                print(f"| {t} | `{a}` | {reg} | {ms_per_micro(r):.1f} | {pt:,.0f} | "
                      f"**{vt:,.0f}** | {rel} |")

    # ---------------- activation term: peak above the resident floor ----------------
    print("\n## Activation memory only (peak alloc − resident floor), seed 41\n")
    print("The resident floor is `param_mem_mib + opt_mem_mib`, read off live parameters and "
          "optimizer state; it is method-independent and identical in every arm of a task.\n")
    print("| task | arm | floor MiB | ragged act MiB | padded act MiB | padded/ragged | "
          "act vs `fb_min_fnorm` (padded) |")
    print("|---|---|---|---|---|---|---|")
    for t in tasks:
        base = None
        rows = []
        for a in arms:
            g = {}
            for reg in ("ragged", "padded"):
                rs = [r for r in by[(t, reg, a)] if r.get("ok") and r["seed"] == 41]
                if rs:
                    r = rs[0]
                    fl = (r.get("param_mem_mib") or 0) + (r.get("opt_mem_mib") or 0)
                    g[reg] = (fl, r["train_peak_alloc_mib"] - fl)
            if len(g) == 2:
                rows.append((a, g))
                if a == "fb_min_fnorm":
                    base = g["padded"][1]
        for a, g in rows:
            fl = g["ragged"][0]
            ar, ap = g["ragged"][1], g["padded"][1]
            rel = f"{ap/base:.2f}x" if base else "—"
            print(f"| {t} | `{a}` | {fl:.1f} | {ar:.1f} | {ap:.1f} | **{ap/ar:.2f}x** | {rel} |")

    # ---------------- padding-invariance ----------------
    print("\n## Padding-invariance of the training computation "
          "(mean train loss per epoch, identical data order)\n")
    print("| task | arm | seed | 1st micro-batch loss ragged | padded | Δ | "
          "ragged epoch-mean trace | padded epoch-mean trace | max |Δ| |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        for a in arms:
            for seed in sorted({r["seed"] for r in recs}):
                gr = [r for r in by[(t, "ragged", a)] if r.get("ok") and r["seed"] == seed]
                gp = [r for r in by[(t, "padded", a)] if r.get("ok") and r["seed"] == seed]
                if not gr or not gp:
                    continue
                lr = gr[0].get("epoch_train_loss_mean") or []
                lp = gp[0].get("epoch_train_loss_mean") or []
                if not lr or not lp or len(lr) != len(lp):
                    continue
                d = max(abs(x - y) for x, y in zip(lr, lp))
                f1, f2 = gr[0].get("first_micro_loss"), gp[0].get("first_micro_loss")
                fs = (f"{f1:.8f} | {f2:.8f} | **{abs(f1-f2):.2e}**"
                      if f1 is not None and f2 is not None else "— | — | —")
                print(f"| {t} | `{a}` | {seed} | {fs} | {', '.join(f'{x:.6f}' for x in lr)} | "
                      f"{', '.join(f'{x:.6f}' for x in lp)} | **{d:.2e}** |")

    # ---------------- receipts ----------------
    print("\n## Receipts\n")
    ok = [r for r in recs if r.get("ok")]
    dts = {json.dumps(r.get("dtype_receipt"), sort_keys=True) for r in ok}
    print(f"- adapter/base dtype receipt distinct values across {len(ok)} successful cells: "
          f"{len(dts)}")
    for d in sorted(dts):
        print(f"  - `{d}`")
    hs = defaultdict(set)
    for r in ok:
        for k, v in (r.get("honesty_counters") or {}).items():
            hs[k].add(json.dumps({kk: (">0" if vv else "ZERO") for kk, vv in v.items()},
                                 sort_keys=True))
    for k, v in hs.items():
        print(f"- honesty counters `{k}`: {sorted(v)}")
    tokid = defaultdict(set)
    for r in ok:
        s = r.get("token_stats") or {}
        if s:
            tokid[(r["task"], r["regime"], r["seed"])].add(
                (s["total_valid_tokens"], s["total_padded_tokens"]))
    bad = {k: v for k, v in tokid.items() if len(v) > 1}
    print(f"- identical data across arms (total valid/padded tokens per task/regime/seed): "
          f"{'ALL MATCH' if not bad else f'MISMATCH {bad}'}")

    # repeat-run noise floor: same cell, same seed, run twice in different sweeps
    rep = defaultdict(list)
    for r in recs:
        if r.get("ok"):
            rep[(r["task"], r["regime"], r["arm"], r["seed"])].append(r)
    dup = {k: v for k, v in rep.items() if len(v) > 1}
    if dup:
        print("\n## Run-to-run noise floor (identical cell, repeated)\n")
        print("| task | regime | arm | seed | 1st micro-batch loss | Δ | epoch-mean traces | max |Δ| "
              "| peak alloc MiB | ms/micro-step |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for (t, reg, a, s), v in sorted(dup.items()):
            f = [x.get("first_micro_loss") for x in v]
            tr = [x.get("epoch_train_loss_mean") or [] for x in v]
            dl = (max(abs(x - y) for x, y in zip(tr[0], tr[1]))
                  if len(tr[0]) == len(tr[1]) and tr[0] else float("nan"))
            df = abs(f[0] - f[1]) if None not in f[:2] else float("nan")
            print(f"| {t} | {reg} | `{a}` | {s} | "
                  f"{' / '.join('n/a' if x is None else f'{x:.8f}' for x in f)} | **{df:.2e}** | "
                  f"{' / '.join('[' + ', '.join(f'{y:.6f}' for y in x) + ']' for x in tr)} | "
                  f"**{dl:.2e}** | "
                  f"{' / '.join('%.2f' % x['train_peak_alloc_mib'] for x in v)} | "
                  f"{' / '.join(f'{ms_per_micro(x):.1f}' for x in v)} |")

    # control-arm stability
    print("\n## Control-arm stability (`lora_sdpa_ad`, touches none of the code under test)\n")
    print("| task | regime | seed | peak alloc MiB | ms/micro-step | dev from median |")
    print("|---|---|---|---|---|---|")
    for t in tasks:
        for reg in ("ragged", "padded"):
            rs = sorted([r for r in by[(t, reg, CONTROL_ARM)] if r.get("ok")],
                        key=lambda r: r["seed"])
            if not rs:
                continue
            med = statistics.median([ms_per_micro(r) for r in rs])
            for r in rs:
                m = ms_per_micro(r)
                print(f"| {t} | {reg} | {r['seed']} | {r['train_peak_alloc_mib']:.2f} | {m:.1f} | "
                      f"{100*(m-med)/med:+.2f}% |")

    # failures
    bad = [r for r in recs if not r.get("ok")]
    if bad:
        print("\n## Cells that did not run\n")
        for r in bad:
            print(f"- `{r['cell']}`: {(r.get('error_tail') or ['?'])[-1][:220]}")


if __name__ == "__main__":
    main()
