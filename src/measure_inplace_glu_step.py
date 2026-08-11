"""Whole-step ON-vs-OFF cost of the in-place GLU window, paired at STEP granularity.

WHY THIS DESIGN.  The dev box is shared and has been at 97-100% utilisation from another tenant for
this entire session, which `fair_comparison_protocol.md` and CONTEXT.md section 33.10 both say voids a
whole-step timing window -- one such window in this project produced the physically impossible
result that a strictly-more-work arm was 18% FASTER.  Two separate processes, or two blocks of
steps, inherit that problem in full.

So neither arm gets its own run.  `_FB_INPLACE_GLU` is read inside `backward`, so it can be flipped
BETWEEN STEPS ON ONE MODEL: the same weights, the same batch, the same optimiser, the same
allocator state, alternating A/B/A/B.  Tenant interference then lands on both arms in proportion
and cancels in the paired difference, which is the quantity reported.  The absolute ms/step is
still contaminated and is NOT a protocol number -- it is printed only so the contamination is
visible.

The paired statistic is the median of the per-PAIR differences (ON_i - OFF_i), not the difference
of the medians, so a drifting baseline cannot manufacture a delta.  A control that touches none of
the code under test is timed in the same interleave.

Usage:
  PYTHONPATH=src python src/measure_inplace_glu_step.py --seq 1024 --pairs 40 --flce
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_WANT_LIGER = "--flce" in sys.argv
if _WANT_LIGER:
    _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "temp", "liger_pkgs")
    if os.path.isdir(_lp) and _lp not in sys.path:
        sys.path.insert(0, _lp)

import torch  # noqa: E402

import profile_unsloth as pu  # noqa: E402
from profile_hyclora import make_batch, step, RESULTS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--flce", action="store_true")
    ap.add_argument("--no_head", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    import flashffn

    cfg = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "batch": args.batch, "seq": args.seq,
        "hidden": 2048, "intermediate": 5632, "n_heads": 32, "n_kv_heads": 4,
        "head_dim": 64, "n_layers": 22, "lora_r": 16, "q_bit": 4,
        "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
        "iteration_threshold": 5, "attn_nnz": 0,
    }

    torch.manual_seed(41)
    model = pu.build(args.arm, cfg, device, args)
    batch = make_batch(cfg, device, model.config.vocab_size)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    for _ in range(8):
        step(model, batch, opt)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    def timed_step():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(model, batch, opt)
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - t0)

    def peak_for(on):
        flashffn.fb_inplace_glu_enable(on)
        for _ in range(2):
            step(model, batch, opt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step(model, batch, opt)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2 ** 20

    peak_off = peak_for(False)
    peak_on = peak_for(True)

    on_t, off_t, diffs = [], [], []
    for i in range(args.pairs):
        # Alternate which of the pair goes first, so a monotone drift within a pair cannot bias
        # the sign of the difference.
        if i % 2 == 0:
            flashffn.fb_inplace_glu_enable(True)
            a = timed_step()
            flashffn.fb_inplace_glu_enable(False)
            b = timed_step()
        else:
            flashffn.fb_inplace_glu_enable(False)
            b = timed_step()
            flashffn.fb_inplace_glu_enable(True)
            a = timed_step()
        on_t.append(a)
        off_t.append(b)
        diffs.append(a - b)

    flashffn.fb_inplace_glu_enable(True)
    on_s, off_s, d_s = sorted(on_t), sorted(off_t), sorted(diffs)
    med_off = off_s[len(off_s) // 2]
    res = {
        "arm": args.arm, "seq": args.seq, "batch": args.batch, "flce": bool(args.flce),
        "pairs": args.pairs,
        "WARNING": "absolute ms/step is contaminated by a shared box and is NOT a protocol number; "
                   "only the paired difference is reported as a result",
        "ms_on_median": on_s[len(on_s) // 2],
        "ms_off_median": med_off,
        "paired_diff_median_ms": d_s[len(d_s) // 2],
        "paired_diff_mean_ms": statistics.fmean(diffs),
        "paired_diff_p25_ms": d_s[int(0.25 * len(d_s))],
        "paired_diff_p75_ms": d_s[int(0.75 * len(d_s))],
        "paired_diff_pct_of_step": 100.0 * d_s[len(d_s) // 2] / max(med_off, 1e-9),
        "n_pairs_on_faster": sum(1 for d in diffs if d < 0),
        "peak_off_MiB": peak_off, "peak_on_MiB": peak_on,
        "peak_saved_MiB": peak_off - peak_on,
        "torch": torch.__version__,
    }
    if args.arm.startswith("fb_"):
        from flashffn import fb_policy_report
        res["fb_policy"] = fb_policy_report()

    out = args.out or os.path.join(RESULTS_DIR, f"inplace_glu_step_{args.arm}_seq{args.seq}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"\narm {args.arm}  seq {args.seq}  flce={args.flce}  pairs={args.pairs}")
    print(f"  peak  OFF {peak_off:.2f} -> ON {peak_on:.2f} MiB  "
          f"(saved {peak_off - peak_on:.2f})")
    print(f"  ms/step (CONTAMINATED, not a protocol number): "
          f"ON {res['ms_on_median']:.2f}  OFF {res['ms_off_median']:.2f}")
    print(f"  PAIRED diff (ON - OFF): median {res['paired_diff_median_ms']:+.3f} ms "
          f"[{res['paired_diff_p25_ms']:+.3f}, {res['paired_diff_p75_ms']:+.3f}]  "
          f"= {res['paired_diff_pct_of_step']:+.2f}% of step")
    print(f"  ON faster in {res['n_pairs_on_faster']}/{args.pairs} pairs")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
