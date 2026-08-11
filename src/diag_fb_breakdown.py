#!/usr/bin/env python
"""Per-kernel CUDA-time breakdown of a training step, bucketed by category.

Purpose (2026-08-03): attribute `fb_min_fnorm`'s throughput deficit against `hyclora_flash_nc`
at long sequence.  The hypothesis under test is that `keep='min'` recomputes the whole block
forward INCLUDING the FlashAttention forward, which is O(S^2), while every other recomputed
operation is O(S) -- so the deficit should grow like S^2 while the memory lead stays flat.

The FA forward is called 22x per step in a non-recomputing arm and 44x per step in `fb_min`
(22 forward + 22 recompute) at identical shapes, so the recompute share is exactly half of the
`flash_fwd` bucket.  That is read off the kineto call counts rather than assumed.

Everything is imported from `profile_unsloth` / `profile_hyclora` so the model construction is
byte-identical to the measurement harness.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import profile_unsloth as pu  # noqa: E402
import profile_hyclora as ph  # noqa: E402
from profile_hyclora import make_batch, step  # noqa: E402

# Kernel-name -> category.  Ordered; first match wins.  `other` is printed in full so nothing
# large can hide in it.
_BUCKETS = [
    ("flash_fwd", r"flash_fwd|flash::.*fwd"),
    ("flash_bwd", r"flash_bwd|flash::.*bwd|flash.*dot_do_o|flash.*convert_dq|clear_dkvaccum"),
    ("rmsnorm_fused", r"_fb_rmsnorm"),
    ("rope_fused", r"_fb_rope"),
    ("head2hidden", r"_fb_head_to_hidden"),
    ("swiglu_fused", r"_silu_mul|_swiglu"),
    ("hyclora_kernels", r"quant|dequant|_hy|outlier|rms_norm_(forward|backward)|"
                        r"rope_embedding|_ln_|triton_"),
    ("ce_logits", r"softmax|nll_loss|cross_entropy|liger|fused_linear"),
    ("optimizer", r"adam|multi_tensor|foreach|amp_"),
    ("gemm", r"gemm|cutlass|ampere_|sm80_|sm86_|s16816|s1688|volta_|turing_"),
    ("copy_cast", r"CatArray|copy|Copy|cast|convert|contiguous|transpose|permute"),
    ("elementwise", r"elementwise|vectorized_|unrolled_|reduce_kernel|fill|Fill|index"),
]


def bucket(name):
    for cat, pat in _BUCKETS:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


def breakdown(arm, cfg, device, args, n_prof_steps=2):
    torch.manual_seed(41)
    model = pu.build(arm, cfg, device, args)
    vocab = model.config.vocab_size
    batch = make_batch(cfg, device, vocab)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4)

    for _ in range(max(4, cfg["iteration_threshold"] + 3)):
        step(model, batch, opt)
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(n_prof_steps):
            step(model, batch, opt)
        torch.cuda.synchronize()

    per_kernel = defaultdict(lambda: [0.0, 0])   # name -> [us, count]
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0.0)
        if dev <= 0:
            continue
        if str(getattr(e, "device_type", "")).endswith("CUDA") or e.key.startswith("void ") \
                or True:
            per_kernel[e.key][0] += float(dev)
            per_kernel[e.key][1] += int(e.count)

    # kineto reports CPU-op self device time too on some builds; keep only true kernels by
    # filtering the ones that also have CPU time attributed as "self" is ambiguous.  Instead use
    # the event list restricted to device events.
    per_kernel = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if str(e.device_type) not in ("DeviceType.CUDA", "DeviceType.PrivateUse1"):
            continue
        dur = float(getattr(e, "device_time", 0.0) or getattr(e, "cuda_time", 0.0) or 0.0)
        if dur <= 0:
            dur = float(getattr(e, "duration", 0.0) or 0.0)
        per_kernel[e.name][0] += dur
        per_kernel[e.name][1] += 1

    cats = defaultdict(lambda: [0.0, 0])
    for name, (us, n) in per_kernel.items():
        c = bucket(name)
        cats[c][0] += us
        cats[c][1] += n

    total = sum(v[0] for v in cats.values())
    res = {
        "arm": arm, "seq": cfg["seq"], "batch": cfg["batch"], "n_prof_steps": n_prof_steps,
        "total_device_ms_per_step": total / 1e3 / n_prof_steps,
        "categories": {c: {"ms_per_step": v[0] / 1e3 / n_prof_steps,
                           "calls_per_step": v[1] / n_prof_steps,
                           "pct": 100.0 * v[0] / total if total else 0.0}
                       for c, v in sorted(cats.items(), key=lambda kv: -kv[1][0])},
        "top_kernels": [{"name": k, "ms_per_step": v[0] / 1e3 / n_prof_steps,
                         "calls_per_step": v[1] / n_prof_steps, "bucket": bucket(k)}
                        for k, v in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:35]],
    }
    del model, opt, batch
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapter_dtype", default="bf16")
    ap.add_argument("--flce", action="store_true")
    ap.add_argument("--no_head", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmpdir", default="/tmp")
    args = ap.parse_args()

    cfg = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "batch": args.batch, "seq": args.seq,
        "hidden": 2048, "intermediate": 5632, "n_heads": 32, "n_kv_heads": 4,
        "head_dim": 64, "n_layers": 22, "lora_r": 16, "q_bit": 4,
        "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
        "iteration_threshold": 5, "attn_nnz": 0,
    }
    device = torch.device(args.device)
    out = []
    for arm in args.arm.split(","):
        r = breakdown(arm.strip(), cfg, device, args, n_prof_steps=args.steps)
        out.append(r)
        print(f"\n=== {r['arm']}  seq={r['seq']}  total_device={r['total_device_ms_per_step']:.1f} "
              f"ms/step")
        for c, v in r["categories"].items():
            print(f"    {c:18s} {v['ms_per_step']:9.2f} ms  {v['pct']:5.1f}%  "
                  f"{v['calls_per_step']:8.1f} calls")
        print("    -- top kernels --")
        for k in r["top_kernels"][:14]:
            print(f"      {k['ms_per_step']:8.2f} ms  {k['calls_per_step']:7.1f}x  "
                  f"[{k['bucket']}] {k['name'][:96]}")
        sys.stdout.flush()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
