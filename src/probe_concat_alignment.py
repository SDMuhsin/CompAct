"""Does PADDING the concatenated dimension remove the cuBLAS tile-alignment penalty?

Background (results/recon/fused_downproj.json, src/probe_fused_downproj.py):
issuing the base projection `x @ W^T` ([M,K]x[K,N]) together with the LoRA
down-projection `x @ A^T` ([M,K]x[K,16]) as ONE GEMM against a concatenated
[W ; A] of shape [N+16, K] is bitwise-identical on the base columns and usually
faster -- but at some shapes the fused GEMM is MUCH slower than the base GEMM
alone, apparently because N+16 breaks the cuBLAS output-tile decomposition:

    M=2048 K=2048 N=256   -> cat costs +55.17% vs base alone
    M=8192 K=2048 N=2048  -> +12.66%
    M=8192 K=5632 N=2048  -> +5.65%   (backward twin, concatenating along K)
    M=8192 K=2048 N=5632  -> +4.24%

This probe sweeps the concatenated width: instead of N+16 we build N+P for
P in {16,32,64,128,256}.  The first N rows are W, the next 16 are A, and the
remaining P-16 rows are ZERO padding that is computed and thrown away.  P=16 is
the unpadded control (identical to probe_fused_downproj).  The question is
whether buying back tile alignment with wasted FLOPs is net positive, and where.

Backward twin: dy [M,N] @ W [N,K] plus dy @ B [N,16] versus dy @ cat [N, K+P].

Methodology is copied from probe_fused_downproj.py:
  * pre-allocated outputs, torch.mm(..., out=buf)  -- no allocator churn
  * ALL variants for a shape/pass timed inside ONE rotation, rotating which goes
    first each rep, so tenant contention drifts through every arm equally
  * headline estimator is the MIN (least-contended sample); median also reported
  * control arm (fixed 1024x1024 matmul) at start/middle/end, drift reported

The box is SHARED and pinned at ~99% by foreign tenants.  Absolute ms are NOT
quotable; only the paired/min-based comparisons within a rotation are.

Usage:
    PYTHONPATH=src python src/probe_concat_alignment.py --reps 150 \
        --out results/recon/concat_alignment.json
"""

import argparse
import json
import os
import statistics
import time

import torch

# (label, M, K, N) -- M is tokens, K is in_features, N is out_features.
SHAPES = [
    ("m2048_k2048_n2048", 2048, 2048, 2048),
    ("m2048_k2048_n5632", 2048, 2048, 5632),
    ("m2048_k5632_n2048", 2048, 5632, 2048),
    ("m2048_k2048_n256", 2048, 2048, 256),
    ("m8192_k2048_n2048", 8192, 2048, 2048),
    ("m8192_k2048_n5632", 8192, 2048, 5632),
    ("m8192_k5632_n2048", 8192, 5632, 2048),
    ("m8192_k2048_n256", 8192, 2048, 256),
]

R = 16  # LoRA rank
PADS = [16, 32, 64, 128, 256]  # P=16 == unpadded control


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
    """Time k variants in one rotation, rotating which goes first each rep."""
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


def _median(xs):
    return statistics.median(xs)


def _summ(xs):
    s = sorted(xs)
    return {
        "median_ms": _median(xs),
        "min_ms": s[0],
        "p10_ms": s[int(0.1 * len(s))],
        "p90_ms": s[int(0.9 * len(s))],
        "reps": len(xs),
    }


def _pass_stats(per, pads, base_key="base", thin_key="thin", sep_key="sep"):
    """Turn per-rep timings into the reported per-pad table."""
    base_s = _summ(per[base_key])
    thin_s = _summ(per[thin_key])
    sep_s = _summ(per[sep_key])
    pads_out = {}
    for p in pads:
        cs = _summ(per[f"cat_p{p}"])
        pads_out[str(p)] = {
            "pad": p,
            "cat": cs,
            # is the fused GEMM more expensive than the base GEMM alone?
            "cat_vs_base_pct_min": 100.0 * (cs["min_ms"] - base_s["min_ms"]) / base_s["min_ms"],
            "cat_vs_base_pct_median": 100.0
            * (cs["median_ms"] - base_s["median_ms"])
            / base_s["median_ms"],
            # net gain of the fused GEMM over issuing base+thin separately
            "net_gain_vs_sep_pct_min": 100.0 * (sep_s["min_ms"] - cs["min_ms"]) / sep_s["min_ms"],
            "net_gain_vs_sep_pct_median": 100.0
            * (sep_s["median_ms"] - cs["median_ms"])
            / sep_s["median_ms"],
        }
    best = min(pads, key=lambda p: pads_out[str(p)]["cat"]["min_ms"])
    return {
        "base_only": base_s,
        "thin_only": thin_s,
        "sep": sep_s,
        "pads": pads_out,
        "best_pad_by_min": best,
        "best_cat_min_ms": pads_out[str(best)]["cat"]["min_ms"],
        "best_cat_vs_base_pct_min": pads_out[str(best)]["cat_vs_base_pct_min"],
        "best_net_gain_vs_sep_pct_min": pads_out[str(best)]["net_gain_vs_sep_pct_min"],
        "any_pad_beats_sep": any(
            pads_out[str(p)]["net_gain_vs_sep_pct_min"] > 0.0 for p in pads
        ),
    }


def run_shape(label, M, K, N, reps, dev, dtype, pads):
    x = torch.randn(M, K, device=dev, dtype=dtype)
    w = torch.randn(N, K, device=dev, dtype=dtype) * 0.02
    a = torch.randn(R, K, device=dev, dtype=dtype) * 0.02
    b = torch.randn(N, R, device=dev, dtype=dtype) * 0.02
    dy = torch.randn(M, N, device=dev, dtype=dtype)

    # ---------------- FORWARD: concat along OUTPUT dim ----------------------
    # cat_fwd[P] is [N+P, K]: rows [0,N) = W, [N,N+16) = A, [N+16,N+P) = 0.
    cat_fwd, catf_t, o_cat_f = {}, {}, {}
    for p in pads:
        c = torch.zeros(N + p, K, device=dev, dtype=dtype)
        c[:N].copy_(w)
        c[N:N + R].copy_(a)
        cat_fwd[p] = c
        catf_t[p] = c.t()
        o_cat_f[p] = torch.empty(M, N + p, device=dev, dtype=dtype)

    wt, at = w.t(), a.t()
    o_base_f = torch.empty(M, N, device=dev, dtype=dtype)
    o_thin_f = torch.empty(M, R, device=dev, dtype=dtype)

    def base_fwd():
        torch.mm(x, wt, out=o_base_f)

    def thin_fwd():
        torch.mm(x, at, out=o_thin_f)

    def sep_fwd():
        torch.mm(x, wt, out=o_base_f)
        torch.mm(x, at, out=o_thin_f)

    def _mk_cat_fwd(p):
        tp, op = catf_t[p], o_cat_f[p]

        def f():
            torch.mm(x, tp, out=op)

        return f

    # ---------------- BACKWARD: concat along INPUT dim ----------------------
    # cat_bwd[P] is [N, K+P]: cols [0,K) = W, [K,K+16) = B, [K+16,K+P) = 0.
    cat_bwd, o_cat_b = {}, {}
    for p in pads:
        c = torch.zeros(N, K + p, device=dev, dtype=dtype)
        c[:, :K].copy_(w)
        c[:, K:K + R].copy_(b)
        cat_bwd[p] = c
        o_cat_b[p] = torch.empty(M, K + p, device=dev, dtype=dtype)

    o_base_b = torch.empty(M, K, device=dev, dtype=dtype)
    o_thin_b = torch.empty(M, R, device=dev, dtype=dtype)

    def base_bwd():
        torch.mm(dy, w, out=o_base_b)

    def thin_bwd():
        torch.mm(dy, b, out=o_thin_b)

    def sep_bwd():
        torch.mm(dy, w, out=o_base_b)
        torch.mm(dy, b, out=o_thin_b)

    def _mk_cat_bwd(p):
        cp, op = cat_bwd[p], o_cat_b[p]

        def f():
            torch.mm(dy, cp, out=op)

        return f

    # ---------------- control arm (touches nothing under test) --------------
    cx = torch.randn(1024, 1024, device=dev, dtype=dtype)
    cw = torch.randn(1024, 1024, device=dev, dtype=dtype)
    c_out = torch.empty(1024, 1024, device=dev, dtype=dtype)

    def control():
        torch.mm(cx, cw, out=c_out)

    # ---------------- bitwise verification, EVERY (shape, pad) --------------
    sep_fwd()
    y_ref, xa_ref = o_base_f.clone(), o_thin_f.clone()
    sep_bwd()
    dx_ref, dxa_ref = o_base_b.clone(), o_thin_b.clone()
    _sync()

    bitwise = {}
    for p in pads:
        _mk_cat_fwd(p)()
        _mk_cat_bwd(p)()
        _sync()
        yf = o_cat_f[p][:, :N]
        af = o_cat_f[p][:, N:N + R]
        padf = o_cat_f[p][:, N + R:]
        yb = o_cat_b[p][:, :K]
        ab = o_cat_b[p][:, K:K + R]
        padb = o_cat_b[p][:, K + R:]
        bitwise[str(p)] = {
            "pad": p,
            "fwd_base_bitwise": bool(torch.equal(y_ref, yf)),
            "fwd_base_max_abs_diff": float((y_ref.float() - yf.float()).abs().max()),
            "fwd_base_rel_l2": float(
                (y_ref.float() - yf.float()).norm() / y_ref.float().norm().clamp_min(1e-30)
            ),
            "fwd_adapter_bitwise": bool(torch.equal(xa_ref, af)),
            "fwd_adapter_max_abs_diff": float((xa_ref.float() - af.float()).abs().max()),
            "fwd_pad_cols_all_zero": bool(padf.numel() == 0 or torch.equal(padf, torch.zeros_like(padf))),
            "bwd_base_bitwise": bool(torch.equal(dx_ref, yb)),
            "bwd_base_max_abs_diff": float((dx_ref.float() - yb.float()).abs().max()),
            "bwd_base_rel_l2": float(
                (dx_ref.float() - yb.float()).norm() / dx_ref.float().norm().clamp_min(1e-30)
            ),
            "bwd_adapter_bitwise": bool(torch.equal(dxa_ref, ab)),
            "bwd_adapter_max_abs_diff": float((dxa_ref.float() - ab.float()).abs().max()),
            "bwd_pad_cols_all_zero": bool(padb.numel() == 0 or torch.equal(padb, torch.zeros_like(padb))),
        }
    del y_ref, xa_ref, dx_ref, dxa_ref

    # ---------------- timing ------------------------------------------------
    ctrl_start = _median(_time_fn(control, reps))

    fwd_fns = {"base": base_fwd, "thin": thin_fwd, "sep": sep_fwd}
    for p in pads:
        fwd_fns[f"cat_p{p}"] = _mk_cat_fwd(p)
    rf = _time_rotation(fwd_fns, reps)

    ctrl_mid = _median(_time_fn(control, reps))

    bwd_fns = {"base": base_bwd, "thin": thin_bwd, "sep": sep_bwd}
    for p in pads:
        bwd_fns[f"cat_p{p}"] = _mk_cat_bwd(p)
    rb = _time_rotation(bwd_fns, reps)

    ctrl_end = _median(_time_fn(control, reps))

    ctrls = [ctrl_start, ctrl_mid, ctrl_end]
    drift = (max(ctrls) - min(ctrls)) / min(ctrls)

    # wasted-FLOP accounting: the P-16 zero rows/cols are computed and discarded
    fwd_pad_flop_overhead = {
        str(p): 100.0 * (p - R) / N for p in pads
    }
    bwd_pad_flop_overhead = {
        str(p): 100.0 * (p - R) / K for p in pads
    }

    return {
        "label": label,
        "M": M,
        "K": K,
        "N": N,
        "r": R,
        "pads": pads,
        "reps": reps,
        "control_median_ms": ctrls,
        "control_drift": drift,
        "control_ok": bool(drift <= 0.01),
        "fwd": _pass_stats(rf, pads),
        "bwd": _pass_stats(rb, pads),
        "fwd_pad_flop_overhead_pct_vs_base": fwd_pad_flop_overhead,
        "bwd_pad_flop_overhead_pct_vs_base": bwd_pad_flop_overhead,
        "bitwise": bitwise,
        "raw_fwd_per_rep_ms": {k: v for k, v in rf.items()},
        "raw_bwd_per_rep_ms": {k: v for k, v in rb.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--out", default="results/recon/concat_alignment.json")
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
            "Shared box at ~99% from foreign tenants. Absolute ms are NOT quotable. "
            "Only within-rotation comparisons are; headline estimator is the MIN."
        ),
        "question": (
            "Does padding the concatenated dim to N+P (P in 16/32/64/128/256, first N rows W, "
            "next 16 A, rest zero) remove the cuBLAS tile-alignment regression seen at N+16?"
        ),
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "nvidia_smi_before": smi,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reps": args.reps,
        "rank": R,
        "pad_widths": PADS,
        "results": [],
    }

    for label, M, K, N in shapes:
        print(f"[probe] {label} M={M} K={K} N={N} ...", flush=True)
        res = run_shape(label, M, K, N, args.reps, dev, dtype, PADS)
        out["results"].append(res)
        for ps in ("fwd", "bwd"):
            s = res[ps]
            cells = " ".join(
                f"P{p}:{s['pads'][str(p)]['cat_vs_base_pct_min']:+6.2f}%" for p in PADS
            )
            bw_ok = all(
                res["bitwise"][str(p)][f"{ps}_base_bitwise"]
                and res["bitwise"][str(p)][f"{ps}_adapter_bitwise"]
                for p in PADS
            )
            print(
                f"    {ps}  base_min={s['base_only']['min_ms']:.4f} "
                f"sep_min={s['sep']['min_ms']:.4f} "
                f"best_pad={s['best_pad_by_min']} "
                f"cat_min={s['best_cat_min_ms']:.4f} "
                f"vs_base={s['best_cat_vs_base_pct_min']:+.2f}% "
                f"net_vs_sep={s['best_net_gain_vs_sep_pct_min']:+.2f}% "
                f"bitwise={bw_ok}",
                flush=True,
            )
            print(f"      cat_vs_base by pad: {cells}", flush=True)
        print(f"      control drift {res['control_drift']*100:.2f}%", flush=True)
        torch.cuda.empty_cache()

    out["nvidia_smi_after"] = os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    ).read()
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
