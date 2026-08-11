"""Probe 10 -- gate Q-1 for the rematerialisation certificate.

Q-1 asks for Delta-quality within +/-2e-3 of the matched baseline. For this component the honest
answer is stronger than a measurement and has to be argued, then checked:

  The certificate changes NO arithmetic. It reads bytes that already exist, mixes them into an
  int64, and compares two int64s. Every tensor the block computes is bit-for-bit what it was.
  Gate NR-1 measures exactly that: gradients bitwise identical 308/308 with it off, fused and
  standalone, and the whole `verify_fused_block.py` gate-suite JSON identical to the leaf.

So Delta-ppl is not "within 2e-3", it is EXACTLY ZERO, on any input, by construction.

The trap this probe exists to avoid: a naive check runs a 40-step trajectory with the certificate
off, another with it on, and compares. That test cannot work, and running it proved it -- the
`gc_manual_sdpa` and `hyclora_flash_q2` arms in the same sweep, which the certificate cannot
possibly touch, moved by the same order (0.003086 -> 0.001570 and 0.761378 -> 0.822501). The
harness is nondeterministic run to run, because FlashAttention's BACKWARD is not reproducible
against itself, and over 40 AdamW steps that compounds.

So the measurement is paired, in ONE process, against a measured noise floor -- the structure gate
B6 already uses -- and with `--reps` runs of BOTH arms rather than one of each:

    within-arm  distances (off vs off, on vs on)  -> the noise floor, as a RANGE
    between-arm distances (off vs on, all pairs)  -> the certificate's effect, if any

Q-1 passes iff the between-arm distances are NOT separated from the within-arm ones, i.e. the
certificate is indistinguishable from re-running the same arm.

A NOTE ON THE CRITERION, because the first version of this probe got it wrong. With one run per
arm there are two between-distances and one floor, and it is possible to "pass" by taking the
minimum of the two -- which is cherry-picking, and it is what this file did until every pair was
enumerated instead. Three runs per arm gives 3 within-off, 3 within-on and 9 between distances,
and the comparison is range against range.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -W ignore src/verification/probe10_certificate_quality.py \
        --steps 40 --reps 3 --out results/certificate/probe10_quality.json
"""

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import flashffn                                                       # noqa: E402
from profile_hyclora import build_model, make_batch                   # noqa: E402

CFG = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "batch": 2, "seq": 512, "lora_r": 16,
       "q_bit": 2, "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
       "iteration_threshold": 5, "n_layers": 22}


def trajectory(mode, steps, lr=1e-4):
    """`steps` AdamW steps on one fixed batch. Returns the per-step losses."""
    dev = torch.device("cuda")
    if mode == "off":
        flashffn.fb_certify_disable()
    else:
        flashffn.fb_certify_enable(fused=(mode == "fused"))
    torch.manual_seed(41)
    m = build_model("fb_min_fnorm_sdpa", CFG, dev, adapter_dtype="bf16")
    g = torch.Generator(device=dev).manual_seed(7)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02, generator=g)
    b = make_batch(CFG, dev, m.config.vocab_size)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        out = m(**b)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
        del out
    rep = flashffn.fb_certify_report()
    del m, b, opt
    gc.collect(); torch.cuda.empty_cache()
    return losses, rep


def maxabs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3,
                    help="runs per arm; >=3 so the noise floor is a RANGE, not one pair")
    ap.add_argument("--out", default="results/certificate/probe10_quality.json")
    a = ap.parse_args()
    CFG["seq"] = a.seq

    print(f"== Q-1: paired trajectories, {a.steps} AdamW steps, seq {a.seq}, "
          f"{a.reps} runs per arm ==", flush=True)
    offs, ons, rep_on = [], [], None
    for i in range(a.reps):
        t, _ = trajectory("off", a.steps)
        offs.append(t)
        print(f"  off#{i + 1} done", flush=True)
    for i in range(a.reps):
        t, rep_on = trajectory("fused", a.steps)
        ons.append(t)
        print(f"  on#{i + 1} done", flush=True)

    # WITHIN-arm distances are the noise floor; BETWEEN-arm distances are the effect, if any.
    # Comparing one pair against one pair -- and worse, taking the min of two -- would be
    # cherry-picking, so every pair is enumerated and the two RANGES are compared.
    within_off = [maxabs(offs[i], offs[j])
                  for i in range(a.reps) for j in range(i + 1, a.reps)]
    within_on = [maxabs(ons[i], ons[j])
                 for i in range(a.reps) for j in range(i + 1, a.reps)]
    between = [maxabs(o, n) for o in offs for n in ons]
    floor_hi = max(within_off + within_on)
    # Pass iff the between-arm distances are NOT separated from the within-arm ones: the
    # certificate must be indistinguishable from re-running the same arm.
    separated = min(between) > floor_hi
    res = {
        "config": dict(CFG), "steps": a.steps, "reps": a.reps,
        "within_off_max_abs_dloss": within_off,
        "within_on_max_abs_dloss": within_on,
        "between_off_and_on_max_abs_dloss": between,
        "noise_floor_hi": floor_hi,
        "between_min": min(between), "between_max": max(between),
        "separated_from_noise": bool(separated),
        "Q1_PASS": bool(not separated),
        "any_pair_bitwise": {"within_off": any(offs[i] == offs[j] for i in range(a.reps)
                                               for j in range(i + 1, a.reps)),
                             "between": any(o == n for o in offs for n in ons)},
        "final_losses": {"off": [t[-1] for t in offs], "on": [t[-1] for t in ons]},
        "certificate_report_on": rep_on,
        "trajectories": {"off": offs, "on": ons},
        "argument": (
            "The LOAD-BEARING evidence for Q-1 is not this probe. The certificate changes no "
            "arithmetic: gate NR-1 measures gradients bitwise identical 308/308 with it off, "
            "fused and standalone, and the entire verify_fused_block.py gate-suite JSON is "
            "byte-identical. Delta-ppl is therefore exactly 0 by construction, not 'within 2e-3'. "
            "This probe exists only to confirm that a multi-step trajectory shows no effect "
            "ABOVE the run-to-run nondeterminism of FlashAttention's backward, which is present "
            "between two identical runs anyway and which compounds over AdamW steps."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=str)

    print(f"\n  within-arm |dloss| (the noise floor) : "
          f"off {['%.3e' % v for v in within_off]}  on {['%.3e' % v for v in within_on]}")
    print(f"  between-arm |dloss| (off vs on)      : {['%.3e' % v for v in between]}")
    print(f"  floor hi = {floor_hi:.3e}   between range = [{min(between):.3e}, "
          f"{max(between):.3e}]")
    print(f"  Q-1: {'PASS' if res['Q1_PASS'] else 'FAIL'} -- the certificate is "
          f"{'indistinguishable from' if res['Q1_PASS'] else 'SEPARATED FROM'} re-running the "
          f"same arm")
    print(f"  witnesses compared during the ON trajectories: "
          f"{rep_on['witnesses_compared']}, mismatched: {rep_on['witnesses_mismatched']}")
    print("->", a.out)
