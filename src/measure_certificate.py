"""Gate NR-2 / clauses F4-F5: what does the rematerialisation certificate cost?

Companion to `llmdocs/trackers/remat_certificate.md`. This file answers exactly two questions and
deliberately nothing else:

  F4  is the certificate inside CONTEXT.md §31.5 NR-2 -- peak allocated <= +0.5%, ms/step <=
      +1.0%, peak reserved <= +1.0% -- against the same block with it switched off?
  F5  is FUSING the witnesses into kernels that already read the tensors what makes it
      affordable?  The `standalone` arm is the framework-level cost baseline: the same three
      witnesses, computed by separate kernels that each make their own pass, which is all a
      framework could do because it does not own the kernels.

MEASUREMENT DISCIPLINE (`fair_comparison_protocol.md` section A, binding)
------------------------------------------------------------------------
* The measurement itself is `profile_hyclora.measure_headline`, imported and called unmodified,
  so the recipe (warm-up, one `empty_cache`, >=3 measured steps, allocated AND reserved, dtype
  receipt read off live parameters, gradient-liveness gate) is byte-identical to every other
  number this project has published.  Protocol A.0: one harness.
* Arms are run ONE AT A TIME, in one process, INTERLEAVED (A,B,C,A,B,C,...) rather than in
  blocks, so thermal drift cannot align with the arm variable (A.3.4).
* A CONTROL arm that touches none of the code under test runs first, in the middle and last.  If
  it moves by more than `--control_tol` the window is contaminated and the sweep says so instead
  of reporting a number (A.3.3).  **This project's dev box is shared and this rule has caught
  contamination twice.**
* `nvidia-smi` clocks/temperature/power are recorded per arm by the harness itself (A.3.5).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -W ignore src/measure_certificate.py \
        --seq 1024 --reps 3 --out results/certificate/cost_seq1024.json
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                          # noqa: E402
import flashffn                                                       # noqa: E402
from profile_hyclora import measure_headline                          # noqa: E402

CFG = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "lora_r": 16, "q_bit": 2,
       "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
       "iteration_threshold": 5, "n_layers": 22}

# (label, arm, certificate mode). The control touches none of the certificate code.
MODES = [
    ("control_gc_manual", "gc_manual_sdpa", None),
    ("cert_off",          "fb_min_fnorm_sdpa", "off"),
    ("cert_fused",        "fb_min_fnorm_sdpa", "fused"),
    ("cert_standalone",   "fb_min_fnorm_sdpa", "standalone"),
]


class _Args:
    """The subset of `profile_hyclora.main`'s argparse namespace that `measure_headline` reads."""
    def __init__(self, steps, adapter_dtype="bf16"):
        self.steps = steps
        self.adapter_dtype = adapter_dtype
        # the autograd-graph walk holds a second reference to every stored tensor and perturbs
        # the very peak this file is measuring, so it is off for every arm here, identically.
        self.inventory = False


def set_mode(mode):
    if mode is None or mode == "off":
        flashffn.fb_certify_disable()
    else:
        flashffn.fb_certify_enable(fused=(mode == "fused"))


def run_one(label, arm, mode, cfg, device, args):
    set_mode(mode)
    row = measure_headline(arm, cfg, device, args)
    row["label"] = label
    row["certificate_mode"] = mode or "n/a"
    row["certificate_report"] = flashffn.fb_certify_report()
    return row


def pct(new, base):
    return 100.0 * (new - base) / base


def kernel_time_us(arm, mode, cfg, device, adapter_dtype, steps=6, warm=8):
    """Total GPU kernel time per training step, summed from the profiler.

    `fair_comparison_protocol.md` A.3.6: when the box cannot be quiesced -- and this one is shared
    with tenants outside our PID namespace -- GPU kernel time is the PRIMARY throughput figure and
    wall clock is secondary, because kernel time is far more robust to a co-tenant. Only
    `DeviceType.CUDA` rows are summed, and only SELF device time, which is the pitfall recorded in
    HYCLORA_PROFILE §1: `device_time_total` double-counts every parent annotation.
    """
    from torch.profiler import profile, ProfilerActivity
    from profile_hyclora import build_model, make_batch, step, self_dev_time

    set_mode(mode)
    torch.manual_seed(41)
    model = build_model(arm, cfg, device, adapter_dtype=adapter_dtype, use_cache=False)
    batch = make_batch(cfg, device, model.config.vocab_size)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(warm):
        step(model, batch, opt)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step(model, batch, opt)
        torch.cuda.synchronize()
    tot = sum(self_dev_time(e) for e in prof.key_averages()
              if str(getattr(e, "device_type", "")).endswith("CUDA"))
    per_step = tot / steps
    # per-kernel breakdown, so the certificate's own cost is attributable rather than inferred
    by_name = {}
    for e in prof.key_averages():
        if str(getattr(e, "device_type", "")).endswith("CUDA"):
            by_name[e.key] = by_name.get(e.key, 0.0) + self_dev_time(e) / steps
    top = dict(sorted(by_name.items(), key=lambda kv: -kv[1])[:18])
    del model, opt, batch, prof
    import gc as _gc
    _gc.collect(); torch.cuda.empty_cache()
    return {"kernel_us_per_step": per_step, "n_steps": steps, "top_kernels_us": top}


def microbench(seq, batch, hidden=2048, inter=5632, n_layers=22, iters=300):
    """Attributable cost: time the ONLY two kernels the certificate changes, at the exact shapes.

    Why this exists.  The dev box is shared with tenants outside our PID namespace, and under
    time-slicing even GPU *kernel* durations inflate, so a whole-step wall-clock delta of under
    1% cannot be resolved here (`fair_comparison_protocol.md` A.3.3 voids the window and this
    project's control arm has caught contamination three times now).  But the certificate touches
    exactly two kernels -- `_fb_rmsnorm_fwd_kernel` and `_silu_mul_fwd_kernel` -- and leaves every
    other kernel in the step byte-identical.  So the per-step cost is a sum of measured per-kernel
    deltas times known call counts, and each delta can be measured as a MINIMUM over many trials,
    which is the standard contention-robust estimator: a co-tenant can only ever make a kernel
    slower, so the minimum is the tightest available estimate of the uncontended time.

    Call counts per step at `keep='min'` (from the Function itself, not assumed):
      forward   2 norms (W0 on `x`, W1 on `x_mid`) + 1 GLU (W2)
      backward  2 norms recomputed                 + 1 GLU
      => 4 norm-with-witness + 2 GLU-with-witness per layer per step.
    """
    from profile_hyclora import _time_cuda
    dev = torch.device("cuda")
    m = batch * seq
    x = torch.randn(m, hidden, device=dev, dtype=torch.bfloat16)
    w = torch.ones(hidden, device=dev, dtype=torch.bfloat16)
    hg = torch.randn(m, inter, device=dev, dtype=torch.bfloat16)
    hu = torch.randn(m, inter, device=dev, dtype=torch.bfloat16)

    def best(fn, trials=5):
        return min(_time_cuda(fn, iters) for _ in range(trials)) * 1e6      # us

    flashffn._FB_CERT["fused"] = True
    norm_off = best(lambda: flashffn.fb_rmsnorm_forward(x, w, 1e-5))
    norm_fus = best(lambda: flashffn.fb_rmsnorm_forward(x, w, 1e-5, digest=True))
    glu_off = best(lambda: flashffn.triton_silu_mul(hg, hu))
    glu_fus = best(lambda: flashffn.triton_silu_mul(hg, hu, digest=True))
    flashffn._FB_CERT["fused"] = False
    norm_std = best(lambda: flashffn.fb_rmsnorm_forward(x, w, 1e-5, digest=True))
    glu_std = best(lambda: flashffn.triton_silu_mul(hg, hu, digest=True))
    flashffn._FB_CERT["fused"] = True

    per_step_fused = n_layers * (4 * (norm_fus - norm_off) + 2 * (glu_fus - glu_off))
    per_step_std = n_layers * (4 * (norm_std - norm_off) + 2 * (glu_std - glu_off))
    return {
        "shapes": {"norm": [m, hidden], "glu": [m, inter]},
        "iters": iters, "estimator": "min over 5 x median-of-%d" % iters,
        "norm_us": {"off": norm_off, "fused": norm_fus, "standalone": norm_std,
                    "d_fused_pct": pct(norm_fus, norm_off),
                    "d_standalone_pct": pct(norm_std, norm_off)},
        "glu_us": {"off": glu_off, "fused": glu_fus, "standalone": glu_std,
                   "d_fused_pct": pct(glu_fus, glu_off),
                   "d_standalone_pct": pct(glu_std, glu_off)},
        "calls_per_layer_per_step": {"norm_with_witness": 4, "glu_with_witness": 2},
        "added_us_per_step_fused": per_step_fused,
        "added_us_per_step_standalone": per_step_std,
        "added_ms_per_step_fused": per_step_fused / 1e3,
        "added_ms_per_step_standalone": per_step_std / 1e3,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12, help="timed steady-state steps per arm")
    ap.add_argument("--reps", type=int, default=3, help="interleaved repetitions per arm")
    ap.add_argument("--control_tol", type=float, default=1.0,
                    help="%% drift of the control arm above which the window is declared void")
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--kernel_time", action="store_true",
                    help="also measure GPU kernel time per step (protocol A.3.6, the "
                         "co-tenant-robust throughput figure)")
    ap.add_argument("--micro", action="store_true",
                    help="attributable per-kernel cost (contention-robust); implies no arm sweep")
    ap.add_argument("--out", default="results/certificate/cost.json")
    a = ap.parse_args()

    if a.micro:
        cfgm = dict(CFG, seq=a.seq, batch=a.batch)
        mb = microbench(a.seq, a.batch)
        print(f"== attributable per-kernel cost, seq {a.seq} batch {a.batch} ==")
        for k in ("norm_us", "glu_us"):
            r = mb[k]
            print(f"  {k:8s} off {r['off']:9.2f} us | fused {r['fused']:9.2f} "
                  f"({r['d_fused_pct']:+6.2f}%) | standalone {r['standalone']:9.2f} "
                  f"({r['d_standalone_pct']:+6.2f}%)")
        print(f"  added per step:  fused {mb['added_ms_per_step_fused']:7.4f} ms | "
              f"standalone {mb['added_ms_per_step_standalone']:7.4f} ms")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": cfgm, "microbench": mb}, open(a.out, "w"), indent=2, default=str)
        print("->", a.out)
        sys.exit(0)

    cfg = dict(CFG, seq=a.seq, batch=a.batch)
    device = torch.device("cuda")
    args = _Args(a.steps, a.adapter_dtype)

    rows, controls = [], []
    print(f"== certificate cost, seq {a.seq} batch {a.batch}, {a.reps} interleaved reps ==",
          flush=True)
    for rep in range(a.reps):
        for label, arm, mode in MODES:
            if label.startswith("control") and rep not in (0, a.reps // 2, a.reps - 1):
                continue
            r = run_one(label, arm, mode, cfg, device, args)
            r["rep"] = rep
            (controls if label.startswith("control") else rows).append(r)
            print(f"  rep{rep} {label:17s} peak {r['train_step_peak_alloc_mib']:9.2f} MiB  "
                  f"resv {r['train_step_peak_reserved_mib']:9.2f}  "
                  f"{r['ms_per_step_median']:8.2f} ms  "
                  f"cert={r['certificate_report']['witnesses_compared']:4d} cmp / "
                  f"{r['certificate_report']['witnesses_mismatched']} bad", flush=True)

    # ---- control drift: the gate on whether ANY number here may be quoted ----
    c_ms = [c["ms_per_step_median"] for c in controls]
    c_pk = [c["train_step_peak_alloc_mib"] for c in controls]
    drift_ms = 100.0 * (max(c_ms) - min(c_ms)) / min(c_ms) if c_ms else float("nan")
    drift_pk = 100.0 * (max(c_pk) - min(c_pk)) / min(c_pk) if c_pk else float("nan")
    contaminated = bool(drift_ms > a.control_tol)

    # ---- per-mode medians across reps ----
    agg = {}
    for label, arm, mode in MODES:
        rs = [r for r in rows if r["label"] == label]
        if not rs:
            continue
        agg[label] = {
            "n_reps": len(rs),
            "peak_alloc_mib": statistics.median(r["train_step_peak_alloc_mib"] for r in rs),
            "peak_alloc_all": [r["train_step_peak_alloc_mib"] for r in rs],
            "peak_reserved_mib": statistics.median(r["train_step_peak_reserved_mib"] for r in rs),
            "ms_per_step": statistics.median(r["ms_per_step_median"] for r in rs),
            "ms_per_step_all": [r["ms_per_step_median"] for r in rs],
            "witnesses_compared": rs[-1]["certificate_report"]["witnesses_compared"],
            "witnesses_mismatched": rs[-1]["certificate_report"]["witnesses_mismatched"],
        }

    verdict = {}
    if "cert_off" in agg:
        base = agg["cert_off"]
        for label in ("cert_fused", "cert_standalone"):
            if label not in agg:
                continue
            g = agg[label]
            d = {"d_peak_alloc_pct": pct(g["peak_alloc_mib"], base["peak_alloc_mib"]),
                 "d_peak_reserved_pct": pct(g["peak_reserved_mib"], base["peak_reserved_mib"]),
                 "d_ms_per_step_pct": pct(g["ms_per_step"], base["ms_per_step"])}
            # CONTEXT.md §31.5 NR-2
            d["NR2_peak_alloc_le_0p5"] = bool(d["d_peak_alloc_pct"] <= 0.5)
            d["NR2_ms_per_step_le_1p0"] = bool(d["d_ms_per_step_pct"] <= 1.0)
            d["NR2_peak_reserved_le_1p0"] = bool(d["d_peak_reserved_pct"] <= 1.0)
            d["NR2_PASS"] = bool(d["NR2_peak_alloc_le_0p5"] and d["NR2_ms_per_step_le_1p0"]
                                 and d["NR2_peak_reserved_le_1p0"])
            verdict[label] = d

    # ---- kernel time: interleaved, same discipline, robust to the co-tenant ----
    ktime = {}
    if a.kernel_time:
        print("\n-- GPU kernel time (protocol A.3.6) --", flush=True)
        for rep in range(a.reps):
            for label, arm, mode in MODES:
                if label.startswith("control"):
                    continue
                k = kernel_time_us(arm, mode, cfg, device, a.adapter_dtype)
                ktime.setdefault(label, []).append(k["kernel_us_per_step"])
                if rep == a.reps - 1:
                    ktime.setdefault("_top", {})[label] = k["top_kernels_us"]
                print(f"  rep{rep} {label:17s} {k['kernel_us_per_step'] / 1e3:8.3f} ms of "
                      f"GPU kernel time/step", flush=True)
        base_k = statistics.median(ktime["cert_off"])
        for label in ("cert_fused", "cert_standalone"):
            if label in ktime:
                d = pct(statistics.median(ktime[label]), base_k)
                verdict.setdefault(label, {})["d_kernel_time_pct"] = d
                verdict[label]["NR2_kernel_time_le_1p0"] = bool(d <= 1.0)
                print(f"  {label:17s} {d:+7.3f}% GPU kernel time vs cert_off", flush=True)

    out = {"config": cfg, "reps": a.reps, "timed_steps_per_arm": a.steps, "kernel_time": ktime,
           "control": {"arm": MODES[0][1], "n": len(controls),
                       "ms_per_step": c_ms, "peak_alloc_mib": c_pk,
                       "drift_ms_pct": drift_ms, "drift_peak_pct": drift_pk,
                       "tolerance_pct": a.control_tol,
                       "CONTAMINATED": contaminated},
           "aggregate": agg, "NR2": verdict, "rows": rows, "control_rows": controls}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=str)

    print("\n-- control --", flush=True)
    print(f"  {MODES[0][1]}: ms/step {['%.2f' % v for v in c_ms]}  drift {drift_ms:.2f}% "
          f"(tolerance {a.control_tol}%)  ->  "
          f"{'CONTAMINATED, NUMBERS VOID' if contaminated else 'clean'}", flush=True)
    print("-- NR-2 --", flush=True)
    for label, d in verdict.items():
        print(f"  {label:17s} peak {d['d_peak_alloc_pct']:+7.3f}%  "
              f"resv {d['d_peak_reserved_pct']:+7.3f}%  "
              f"time {d['d_ms_per_step_pct']:+7.3f}%   "
              f"NR-2 {'PASS' if d['NR2_PASS'] else 'FAIL'}", flush=True)
    if contaminated:
        print("\n  The control arm moved more than the tolerance, so another tenant was active "
              "during this window. Per fair_comparison_protocol.md A.3.3 the TIMING numbers "
              "above are void and must be re-run on a quiet box. The MEMORY numbers are not "
              "affected by co-tenancy and remain valid.", flush=True)
    print("->", a.out, flush=True)
