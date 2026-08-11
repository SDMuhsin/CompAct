"""Paired A/B microbenchmark: does folding the LoRA down-projection into the base GEMM
along the OUTPUT dimension pay, and is it bitwise-safe?

Context (llmdocs/CONTEXT.md 39, llmdocs/trackers/throughput_mechanism.md).
`_fb_proj` (src/flashffn.py:3009-3013) issues the base projection and the adapter
down-projection as two separate cuBLAS calls that read the SAME input:

    y  = F.linear(x2, w)      # [M,K] x [K,N]   -> [M,N]     wide, tensor-core bound
    xa = F.linear(x2, a)      # [M,K] x [K,r]   -> [M,r]     thin, ~9.5% machine occupancy

Because they share x2 and contract over the same dimension, they can be issued as ONE
GEMM against [w ; a] concatenated along the OUTPUT dimension.  The backward twin is
    dx   = dy @ w             # contract over `out`
    dxa  = dy @ b             # contract over `out`
-> one GEMM against [w | b] concatenated along the INPUT dimension.

This probe answers three questions, per shape actually observed in
results/recon/step_breakdown_seq{1024,4096}.json:

  Q1  time(sep) vs time(cat)  -- paired, interleaved, with a control arm (33.9)
  Q2  are the base columns of the fused output BITWISE identical to the separate GEMM?
      (gate B3 needs every `keep` level to agree with the others; B2 vs stock HF is a
       tolerance gate, so a last-bit change here is legal but must be measured, not assumed)
  Q3  which cuBLAS kernel does each variant select?  (a different kernel is the mechanism
      by which Q2 can fail)

The box is shared and has been at 99-100% all week.  Absolute ms are therefore NOT
quotable; the paired ratio with a passing control arm is.  Every window carries a control
GEMM that touches none of the code under test, measured at start/middle/end, and the
window is reported as VOID if the control drifts more than 1%.

Usage:
    PYTHONPATH=src python src/probe_fused_downproj.py --reps 60 \
        --out results/recon/fused_downproj.json
"""

import argparse
import json
import os
import statistics
import time

import torch

# (label, M, K, N)  -- M is tokens, K is in_features, N is out_features.
# Drawn from the measured gemm_shape_table; seq1024/batch2 -> M=2048, seq4096 -> M=8192.
SHAPES = [
    # label                         M      K      N
    ("s1024_qo_2048x2048x2048",   2048,  2048,  2048),
    ("s1024_gateup_2048x2048x5632", 2048, 2048,  5632),
    ("s1024_down_2048x5632x2048",  2048,  5632,  2048),
    ("s1024_kv_2048x2048x256",     2048,  2048,   256),
    ("s4096_qo_8192x2048x2048",    8192,  2048,  2048),
    ("s4096_gateup_8192x2048x5632", 8192, 2048,  5632),
    ("s4096_down_8192x5632x2048",  8192,  5632,  2048),
]

R = 16  # LoRA rank


def _sync():
    torch.cuda.synchronize()


def _time_fn(fn, reps, warmup=8):
    for _ in range(warmup):
        fn()
    _sync()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    per = []
    for _ in range(reps):
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        per.append(ev0.elapsed_time(ev1))
    return per


def _time_rotation(fns, reps, warmup=8):
    """Time k variants in one rotation, rotating which goes first each rep.

    Contention on this box drifts on a 10-100 ms scale; putting all k variants inside one
    ~1 ms window makes each rep its own control.  Rotating the order removes any
    systematic first/last-in-window bias.  Returns {name: [per-rep ms]}.
    """
    names = list(fns.keys())
    k = len(names)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    _sync()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * k)]
    per = {n: [] for n in names}
    for i in range(reps):
        order = names[i % k:] + names[: i % k]
        for j, n in enumerate(order):
            ev[2 * j].record()
            fns[n]()
            ev[2 * j + 1].record()
        ev[2 * k - 1].synchronize()
        for j, n in enumerate(order):
            per[n].append(ev[2 * j].elapsed_time(ev[2 * j + 1]))
    return per


def _time_paired(fn_a, fn_b, reps, warmup=8):
    """Interleave A and B rep by rep so tenant contention drifts through BOTH arms.

    Returns (per_rep_a, per_rep_b, per_rep_pct_gain).  A block-structured A-then-B
    comparison is not usable on this box: the first run of this probe voided 5 of 7
    windows on control drift alone (2.1-6.7%).  Interleaving makes each pair its own
    control; the spread of the per-rep gains is then the honest error bar.
    """
    for _ in range(warmup):
        fn_a()
        fn_b()
    _sync()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    pa, pb = [], []
    for i in range(reps):
        # alternate which arm goes first, so neither systematically pays cold-cache cost
        first_a = (i % 2) == 0
        f0, f1 = (fn_a, fn_b) if first_a else (fn_b, fn_a)
        ev[0].record()
        f0()
        ev[1].record()
        ev[2].record()
        f1()
        ev[3].record()
        ev[3].synchronize()
        t0 = ev[0].elapsed_time(ev[1])
        t1 = ev[2].elapsed_time(ev[3])
        if first_a:
            pa.append(t0)
            pb.append(t1)
        else:
            pa.append(t1)
            pb.append(t0)
    gains = [100.0 * (a - b) / a for a, b in zip(pa, pb)]
    return pa, pb, gains


def _median(xs):
    return statistics.median(xs)


def run_shape(label, M, K, N, reps, dev, dtype):
    x = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(N, K, device=dev, dtype=dtype) * 0.02
    a = torch.randn(R, K, device=dev, dtype=dtype) * 0.02
    b = torch.randn(N, R, device=dev, dtype=dtype) * 0.02

    # --- FORWARD variant -------------------------------------------------
    # cat_fwd holds [w ; a] contiguously: one buffer, no duplicated bytes in the
    # real integration (w and a would be views into it).
    cat_fwd = torch.empty(N + R, K, device=dev, dtype=dtype)
    cat_fwd[:N].copy_(w)
    cat_fwd[N:].copy_(a)

    # Pre-allocated outputs: the two largest shapes were bimodal (frac_positive 0.50,
    # +-60% spread) on the first paired run because each rep allocated an 88 MiB output
    # and the caching allocator churned.  `out=` removes that from the comparison.
    wt, at, catf_t = w.t(), a.t(), cat_fwd.t()
    o_base = torch.empty(M, N, device=dev, dtype=dtype)
    o_thin = torch.empty(M, R, device=dev, dtype=dtype)
    o_cat = torch.empty(M, N + R, device=dev, dtype=dtype)

    def base_only_fwd():
        torch.mm(x, wt, out=o_base)

    def thin_only_fwd():
        torch.mm(x, at, out=o_thin)

    def sep_fwd():
        torch.mm(x, wt, out=o_base)
        torch.mm(x, at, out=o_thin)

    def cat_fwd_fn():
        torch.mm(x, catf_t, out=o_cat)

    # --- BACKWARD variant ------------------------------------------------
    # dx = dy @ w   [M,N] x [N,K] -> [M,K];   dxa = dy @ b  [M,N] x [N,R] -> [M,R]
    dy = torch.randn(M, N, device=dev, dtype=dtype)
    cat_bwd = torch.empty(N, K + R, device=dev, dtype=dtype)
    cat_bwd[:, :K].copy_(w)
    cat_bwd[:, K:].copy_(b)

    o_dx = torch.empty(M, K, device=dev, dtype=dtype)
    o_dxa = torch.empty(M, R, device=dev, dtype=dtype)
    o_dcat = torch.empty(M, K + R, device=dev, dtype=dtype)

    def base_only_bwd():
        torch.mm(dy, w, out=o_dx)

    def thin_only_bwd():
        torch.mm(dy, b, out=o_dxa)

    def sep_bwd():
        torch.mm(dy, w, out=o_dx)
        torch.mm(dy, b, out=o_dxa)

    def cat_bwd_fn():
        torch.mm(dy, cat_bwd, out=o_dcat)

    # --- control arm: touches none of the code under test ----------------
    cx = torch.randn(1024, 1024, device=dev, dtype=dtype)
    cw = torch.randn(1024, 1024, device=dev, dtype=dtype)

    def control():
        return cx @ cw

    # --- Q2: bitwise ------------------------------------------------------
    sep_fwd()
    y_s, xa_s = o_base.clone(), o_thin.clone()
    cat_fwd_fn()
    y_c, xa_c = o_cat[:, :N].clone(), o_cat[:, N:].clone()
    sep_bwd()
    dx_s, dxa_s = o_dx.clone(), o_dxa.clone()
    cat_bwd_fn()
    dx_c, dxa_c = o_dcat[:, :K].clone(), o_dcat[:, K:].clone()
    _sync()
    bitwise = {
        "fwd_base_bitwise": bool(torch.equal(y_s, y_c)),
        "fwd_base_max_abs_diff": float((y_s.float() - y_c.float()).abs().max()),
        "fwd_xa_bitwise": bool(torch.equal(xa_s, xa_c)),
        "fwd_xa_max_abs_diff": float((xa_s.float() - xa_c.float()).abs().max()),
        "bwd_base_bitwise": bool(torch.equal(dx_s, dx_c)),
        "bwd_base_max_abs_diff": float((dx_s.float() - dx_c.float()).abs().max()),
        "bwd_dxa_bitwise": bool(torch.equal(dxa_s, dxa_c)),
        "bwd_dxa_max_abs_diff": float((dxa_s.float() - dxa_c.float()).abs().max()),
        "fwd_base_rel_l2": float(
            (y_s.float() - y_c.float()).norm() / y_s.float().norm().clamp_min(1e-30)
        ),
        "bwd_base_rel_l2": float(
            (dx_s.float() - dx_c.float()).norm() / dx_s.float().norm().clamp_min(1e-30)
        ),
    }
    del y_s, xa_s, y_c, xa_c, dx_s, dxa_s, dx_c, dxa_c

    # --- Q1: rotation timing, all four variants inside one window --------
    ctrl_start = _median(_time_fn(control, reps))
    rf = _time_rotation(
        {"base": base_only_fwd, "thin": thin_only_fwd, "sep": sep_fwd, "cat": cat_fwd_fn},
        reps,
    )
    ctrl_mid = _median(_time_fn(control, reps))
    rb = _time_rotation(
        {"base": base_only_bwd, "thin": thin_only_bwd, "sep": sep_bwd, "cat": cat_bwd_fn},
        reps,
    )
    ctrl_end = _median(_time_fn(control, reps))

    fwd_sep, fwd_cat = rf["sep"], rf["cat"]
    bwd_sep, bwd_cat = rb["sep"], rb["cat"]
    fwd_gains = [100.0 * (s - c) / s for s, c in zip(fwd_sep, fwd_cat)]
    bwd_gains = [100.0 * (s - c) / s for s, c in zip(bwd_sep, bwd_cat)]

    ctrls = [ctrl_start, ctrl_mid, ctrl_end]
    drift = (max(ctrls) - min(ctrls)) / min(ctrls)

    def summ(xs):
        s = sorted(xs)
        return {
            "median_ms": _median(xs),
            "min_ms": s[0],
            "p10_ms": s[int(0.1 * len(s))],
            "p90_ms": s[int(0.9 * len(s))],
        }

    def gsumm(gs):
        s = sorted(gs)
        return {
            "median_pct": _median(gs),
            "p10_pct": s[int(0.1 * len(s))],
            "p90_pct": s[int(0.9 * len(s))],
            "frac_positive": sum(1 for g in gs if g > 0) / len(gs),
        }

    fs, fc = _median(fwd_sep), _median(fwd_cat)
    bs, bc = _median(bwd_sep), _median(bwd_cat)

    return {
        "label": label,
        "M": M,
        "K": K,
        "N": N,
        "r": R,
        "reps": reps,
        "control_median_ms": ctrls,
        "control_drift": drift,
        "control_ok": bool(drift <= 0.01),
        "fwd_base_only": summ(rf["base"]),
        "fwd_thin_only": summ(rf["thin"]),
        "bwd_base_only": summ(rb["base"]),
        "bwd_thin_only": summ(rb["thin"]),
        # is the FUSED gemm more expensive than the base gemm alone?  (the +16 columns)
        "fwd_cat_vs_base_pct": 100.0
        * (_median(fwd_cat) - _median(rf["base"]))
        / _median(rf["base"]),
        "bwd_cat_vs_base_pct": 100.0
        * (_median(bwd_cat) - _median(rb["base"]))
        / _median(rb["base"]),
        "fwd_thin_share_of_sep_pct": 100.0 * _median(rf["thin"]) / _median(fwd_sep),
        "bwd_thin_share_of_sep_pct": 100.0 * _median(rb["thin"]) / _median(bwd_sep),
        "fwd_sep": summ(fwd_sep),
        "fwd_cat": summ(fwd_cat),
        "bwd_sep": summ(bwd_sep),
        "bwd_cat": summ(bwd_cat),
        # paired per-rep gain: the drift-immune estimator
        "fwd_paired_gain": gsumm(fwd_gains),
        "bwd_paired_gain": gsumm(bwd_gains),
        # unpaired medians, kept for cross-check only
        "fwd_speedup_pct": 100.0 * (fs - fc) / fs,
        "bwd_speedup_pct": 100.0 * (bs - bc) / bs,
        "fwd_speedup_pct_min": 100.0 * (min(fwd_sep) - min(fwd_cat)) / min(fwd_sep),
        "bwd_speedup_pct_min": 100.0 * (min(bwd_sep) - min(bwd_cat)) / min(bwd_sep),
        "bitwise": bitwise,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="results/recon/fused_downproj.json")
    ap.add_argument("--shapes", default="all")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    dev = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    shapes = SHAPES
    if args.shapes != "all":
        want = set(args.shapes.split(","))
        shapes = [s for s in SHAPES if s[0] in want]

    smi = os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    ).read()

    out = {
        "WARNING": (
            "Shared box. Absolute ms are NOT quotable. The paired sep-vs-cat ratio is, "
            "and only when control_ok is true."
        ),
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "nvidia_smi_before": smi,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reps": args.reps,
        "results": [],
    }
    for label, M, K, N in shapes:
        print(f"[probe] {label} M={M} K={K} N={N} ...", flush=True)
        res = run_shape(label, M, K, N, args.reps, dev, dtype)
        out["results"].append(res)
        fg, bg = res["fwd_paired_gain"], res["bwd_paired_gain"]
        print(
            f"    fwd {fg['median_pct']:+6.2f}% [{fg['p10_pct']:+6.2f},{fg['p90_pct']:+6.2f}] "
            f"pos={fg['frac_positive']:.2f}   "
            f"bwd {bg['median_pct']:+6.2f}% [{bg['p10_pct']:+6.2f},{bg['p90_pct']:+6.2f}] "
            f"pos={bg['frac_positive']:.2f}   "
            f"ctrl_drift {res['control_drift']*100:.2f}%  "
            f"bitwise fwd={res['bitwise']['fwd_base_bitwise']} bwd={res['bitwise']['bwd_base_bitwise']}",
            flush=True,
        )
        print(
            f"      decomposed: fwd thin={res['fwd_thin_share_of_sep_pct']:.2f}% of sep, "
            f"cat vs base-alone {res['fwd_cat_vs_base_pct']:+.2f}%   |   "
            f"bwd thin={res['bwd_thin_share_of_sep_pct']:.2f}% of sep, "
            f"cat vs base-alone {res['bwd_cat_vs_base_pct']:+.2f}%",
            flush=True,
        )
        torch.cuda.empty_cache()

    out["nvidia_smi_after"] = os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    ).read()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
