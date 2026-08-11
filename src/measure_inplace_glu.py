"""Attributable per-kernel cost of the in-place GLU backward window.

WHY PER-KERNEL RATHER THAN WHOLE-STEP.  `fair_comparison_protocol.md` A.3.6 and CONTEXT.md section
33.10 both allow a per-kernel measurement when the box cannot be quiesced, and it is the right
instrument here anyway: the change is local and replaces a known set of kernel launches with
another.  Whole-step wall clock on a contended box has already produced a physically impossible
result in this project (a strictly-more-work arm measured 18% FASTER), so it is not used here.

WHAT IS COMPARED.  Per decoder block per step, the FFN backward runs either

  OLD:  triton_swiglu_backward(grad_h_mid, h_gate, h_up) -> 2 fresh [N, I] allocations
        triton_silu_mul(h_gate, h_up)                    -> 1 fresh [N, I] allocation
        i.e. TWO kernel launches, THREE allocations, and 5 x [N, I] live at the crossing point

  NEW:  triton_swiglu_backward_glu(..., inplace=True)    -> ONE launch, ZERO allocations,
        3 x [N, I] live

The two are timed alternately in the same process on the same buffers, so any drift from another
tenant lands on both arms rather than on one, and the ratio survives what the absolute numbers do
not.  A CONTROL kernel that touches none of the code under test is timed in the same interleave; if
its own median moves by more than a few percent between the first and last block of reps the window
is reported as suspect.

Usage:
  python src/measure_inplace_glu.py --seq 1024 --batch 2 --reps 200
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402


def time_kernel(fn, reps, warmup=20):
    """Median GPU time per call, from CUDA events, one event pair per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts


def summarise(ts):
    return {
        "median_ms": ts[len(ts) // 2],
        "min_ms": ts[0],
        "p25_ms": ts[int(0.25 * len(ts))],
        "p75_ms": ts[int(0.75 * len(ts))],
        "iqr_ms": ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))],
        "n": len(ts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--inter", type=int, default=5632, help="LLaMA intermediate size")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=4, help="interleaved measurement blocks")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/hyclora/frontier/inplace_glu_kernel_cost.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    from flashffn import (triton_swiglu_backward, triton_silu_mul,
                          triton_swiglu_backward_glu)

    n = args.batch * args.seq
    i = args.inter
    torch.manual_seed(41)
    h_gate = torch.randn(n, i, device=device, dtype=torch.bfloat16)
    h_up = torch.randn(n, i, device=device, dtype=torch.bfloat16)
    g_mid = torch.randn(n, i, device=device, dtype=torch.bfloat16)
    # Scratch copies, so the in-place arm does not consume the inputs the other arm needs.
    sg = torch.empty_like(h_gate)
    su = torch.empty_like(h_up)
    sm = torch.empty_like(g_mid)
    ctrl_a = torch.randn(n, i, device=device, dtype=torch.bfloat16)
    ctrl_b = torch.empty_like(ctrl_a)

    # The OLD path never mutates its inputs, so it needs no refill and is timed as it runs.
    # The NEW path consumes them, so its buffers must be refilled between reps or the kernel would
    # be re-applied to its own output and would very quickly be timed on Inf/NaN data.  The refill
    # is therefore timed SEPARATELY and subtracted, rather than left in one arm only -- in the real
    # backward no copy exists at all, because the buffers ARE the block's freshly recomputed
    # tensors and are consumed exactly once.
    def old():
        gg, gu = triton_swiglu_backward(g_mid, h_gate, h_up)
        hm = triton_silu_mul(h_gate, h_up)
        return gg, gu, hm

    def refill():
        sg.copy_(h_gate)
        su.copy_(h_up)
        sm.copy_(g_mid)

    def new():
        refill()
        return triton_swiglu_backward_glu(sm, sg, su, True, inplace=True)

    def control():
        torch.mul(ctrl_a, 1.0, out=ctrl_b)

    res = {"seq": args.seq, "batch": args.batch, "inter": args.inter,
           "N": n, "tensor_MiB": n * i * 2 / 2 ** 20,
           "reps": args.reps, "blocks": args.blocks,
           "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
           "arms": {}, "control_by_block": []}

    acc = {"old": [], "new_plus_refill": [], "refill_only": []}
    for blk in range(args.blocks):
        res["control_by_block"].append(summarise(time_kernel(control, args.reps))["median_ms"])
        acc["old"] += time_kernel(old, args.reps)
        acc["new_plus_refill"] += time_kernel(new, args.reps)
        acc["refill_only"] += time_kernel(refill, args.reps)
    res["control_by_block"].append(summarise(time_kernel(control, args.reps))["median_ms"])

    for k, v in acc.items():
        v.sort()
        res["arms"][k] = summarise(v)

    c = res["control_by_block"]
    res["control_drift_pct"] = 100.0 * (max(c) - min(c)) / max(min(c), 1e-12)
    om = res["arms"]["old"]["median_ms"]
    nm = (res["arms"]["new_plus_refill"]["median_ms"]
          - res["arms"]["refill_only"]["median_ms"])
    res["arms"]["new_derived"] = {"median_ms": nm,
                                  "note": "new_plus_refill minus refill_only"}
    res["delta_ms_per_block"] = nm - om
    res["delta_pct"] = 100.0 * res["delta_ms_per_block"] / om
    res["n_launches_old"] = 2
    res["n_launches_new"] = 1
    res["allocations_old_MiB"] = 3 * n * i * 2 / 2 ** 20
    res["allocations_new_MiB"] = 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"\nseq {args.seq} batch {args.batch}  ([{n}, {i}] bf16 = "
          f"{res['tensor_MiB']:.2f} MiB per tensor)")
    for k in ("old", "new_plus_refill", "refill_only"):
        a = res["arms"][k]
        print(f"  {k:<16} median {a['median_ms']:8.4f} ms   IQR {a['iqr_ms']:7.4f}   "
              f"min {a['min_ms']:8.4f}")
    print(f"  {'new (derived)':<16} median "
          f"{res['arms']['new_derived']['median_ms']:8.4f} ms")
    print(f"  control medians by block: "
          f"{', '.join(f'{x:.4f}' for x in res['control_by_block'])}  "
          f"(drift {res['control_drift_pct']:.2f}%)")
    print(f"\n  per block per step: {res['delta_ms_per_block']:+.4f} ms "
          f"({res['delta_pct']:+.2f}%), launches {res['n_launches_old']} -> "
          f"{res['n_launches_new']}, allocations "
          f"{res['allocations_old_MiB']:.1f} -> {res['allocations_new_MiB']:.1f} MiB")
    print(f"  x22 layers: {22 * res['delta_ms_per_block']:+.3f} ms/step")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
