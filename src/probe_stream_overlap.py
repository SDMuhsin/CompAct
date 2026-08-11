#!/usr/bin/env python
"""Does a second CUDA stream buy anything?  The falsification test for Route B's named mechanism.

CONTEXT.md 33.5 / 34.9.4 names ONE mechanism for making `keep='min'` faster: layer i's recompute
depends only on the stored `x_i`, so it is independent of layer i+1's backward and can run on a
second stream.  That is a correctness argument.  It is NOT a performance argument, and the
difference has never been measured here.

The performance argument requires the GPU to have idle capacity while the backward runs.  Two
kernels on two streams do not add throughput on a device whose SMs are already full -- they
time-slice, and the literature on overlapping compute with communication reports contention
SLOWDOWNS of up to 1.3x on exactly this hardware pattern.  So before any kernel is written, this
script measures the ceiling directly:

    run op A alone,  run op B alone,  run A and B concurrently on two streams.

    overlap_gain = (t_A + t_B - t_concurrent) / min(t_A, t_B)

      1.00  perfect: the shorter op became free (this is what Route B needs)
      0.00  none:    concurrency bought exactly nothing
     <0.00  contention: two streams are SLOWER than one

The ops are the real ones at real shapes, taken from the live model config, not synthetic:
  A-side (what the recompute does, and what we want to hide)
      `flash_fwd`   the O(S^2) FlashAttention forward -- 462 ms/step of a 4596 ms step at seq 16384
      `gate_up`     the two [N, H] x [H, I] GLU projections re-run at keep='min'
  B-side (what it would hide behind: the backward of the NEXT layer)
      `ffn_bwd`     grad_h_mid = gy @ wd, then the two [N, I] x [I, H] grad_xn2 GEMMs
      `flash_bwd`   the FlashAttention backward

The measurement is a ceiling, not an implementation: it puts A and B on two streams with no
dependency between them at all, which is the most favourable case any real scheduler could reach.
If the ceiling is near zero, the mechanism is dead and no amount of kernel work rescues it.

Usage:
    PYTHONPATH=src python src/probe_stream_overlap.py --seq 4096 --batch 2 --reps 30
    PYTHONPATH=src python src/probe_stream_overlap.py --seq 16384 --batch 2 --reps 10
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import profile_hyclora as ph  # noqa: E402


def _gpu_state():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], text=True, timeout=20)
        return [l.strip() for l in out.strip().splitlines()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


def make_ops(cfg, device):
    """Real shapes from the model config.  Returns {name: (callable, side)}."""
    from transformers import AutoConfig
    mc = AutoConfig.from_pretrained(cfg["model"])
    H = mc.hidden_size
    I = mc.intermediate_size
    nh = mc.num_attention_heads
    nkv = getattr(mc, "num_key_value_heads", nh)
    hd = H // nh
    B, S = cfg["batch"], cfg["seq"]
    N = B * S
    dt = torch.bfloat16

    q = torch.randn(B, nh, S, hd, device=device, dtype=dt)
    k = torch.randn(B, nkv, S, hd, device=device, dtype=dt)
    v = torch.randn(B, nkv, S, hd, device=device, dtype=dt)
    # A separate q/k/v set that DOES require grad, for the backward-side op.  Kept distinct so the
    # forward-side op never builds a graph and the two sides stay independent.
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)

    # TinyLlama is GQA (32 query heads, 4 KV heads) and `scaled_dot_product_attention` will not
    # broadcast them for you -- it needs `enable_gqa=True`, which is also what the fused block's
    # own `_fb_flash_fwd` relies on ("GQA-native: k/v keep num_key_value_heads").  Passing the
    # unexpanded tensors without it raises, which is how this was caught.
    def _sdpa(qq, kk, vv):
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, enable_gqa=(nkv != nh))

    o_ref = _sdpa(qg, kg, vg)
    go = torch.randn_like(o_ref)

    xn2 = torch.randn(N, H, device=device, dtype=dt)
    wg = torch.randn(I, H, device=device, dtype=dt)
    wu = torch.randn(I, H, device=device, dtype=dt)
    wd = torch.randn(H, I, device=device, dtype=dt)
    gy = torch.randn(N, H, device=device, dtype=dt)
    ghg = torch.randn(N, I, device=device, dtype=dt)
    ghu = torch.randn(N, I, device=device, dtype=dt)

    def flash_fwd():
        with torch.no_grad():
            _sdpa(q, k, v)

    def gate_up():
        with torch.no_grad():
            F.linear(xn2, wg)
            F.linear(xn2, wu)

    def flash_bwd():
        # Re-run the forward under grad and backward it.  The forward part is common to both arms
        # of the comparison, so it inflates t_B equally in the sequential and concurrent legs and
        # cannot manufacture an overlap gain.
        o = _sdpa(qg, kg, vg)
        torch.autograd.grad(o, (qg, kg, vg), go, retain_graph=False)

    def ffn_bwd():
        with torch.no_grad():
            gy @ wd                 # grad_h_mid
            ghg @ wg                # grad_xn2 (gate half)
            ghu @ wu                # grad_xn2 (up half)

    # ---- the OTHER idle resource: the copy engines --------------------------------------------
    # If SM-vs-SM overlap is dead (the prediction), the question becomes whether the DMA engines
    # can carry `o_h` to and from pinned host memory while the SMs stay busy.  That would buy
    # `keep='attn'`'s speed at `keep='min'`'s GPU memory, which is exactly the both-axes point.
    # `o_h` is [B, S, hidden] bf16 -- the same shape as the block input, 128 MiB/layer at seq
    # 16384.  Sized from the live config, not assumed.
    o_h_dev = torch.empty(N, H, device=device, dtype=dt)
    o_h_host = torch.empty(N, H, device="cpu", dtype=dt, pin_memory=True)
    o_h_mib = o_h_dev.numel() * o_h_dev.element_size() / 2**20

    def d2h():
        o_h_host.copy_(o_h_dev, non_blocking=True)

    def h2d():
        o_h_dev.copy_(o_h_host, non_blocking=True)

    return ({"flash_fwd": (flash_fwd, "A"), "gate_up": (gate_up, "A"),
             "flash_bwd": (flash_bwd, "B"), "ffn_bwd": (ffn_bwd, "B")},
            {"d2h_o_h": d2h, "h2d_o_h": h2d}, o_h_mib)


def time_alone(fn, reps, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def time_concurrent(fn_a, fn_b, reps, warmup=5):
    """A on stream 1, B on stream 2, no dependency either way -- the best case for overlap."""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    for _ in range(warmup):
        with torch.cuda.stream(s1):
            fn_a()
        with torch.cuda.stream(s2):
            fn_b()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        with torch.cuda.stream(s1):
            fn_a()
        with torch.cuda.stream(s2):
            fn_b()
    # The end event must observe BOTH streams, so join them into the default stream first.
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--model", default=ph.DEFAULT_MODEL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = ph.make_cfg(args.batch, args.seq, model=args.model)
    state_before = _gpu_state()
    ops, dma, o_h_mib = make_ops(cfg, device)

    alone = {}
    for name, (fn, _side) in ops.items():
        alone[name] = time_alone(fn, args.reps)
        print(f"  {name:<12} alone {alone[name]:8.3f} ms")
    for name, fn in dma.items():
        alone[name] = time_alone(fn, args.reps)
        print(f"  {name:<12} alone {alone[name]:8.3f} ms  "
              f"({o_h_mib:.1f} MiB, {o_h_mib / 2**10 / (alone[name] / 1e3):.1f} GiB/s)")

    pairs = [(a, b) for a, (_, sa) in ops.items() if sa == "A"
             for b, (_, sb) in ops.items() if sb == "B"]
    results = []
    print(f"\n  {'A (recompute)':<14} {'B (backward)':<14} {'t_A':>8} {'t_B':>8} "
          f"{'seq':>9} {'conc':>9} {'gain':>8}")
    for a, b in pairs:
        t_conc = time_concurrent(ops[a][0], ops[b][0], args.reps)
        t_seq = alone[a] + alone[b]
        gain = (t_seq - t_conc) / min(alone[a], alone[b])
        results.append({"a": a, "b": b, "t_a_ms": alone[a], "t_b_ms": alone[b],
                        "t_sequential_ms": t_seq, "t_concurrent_ms": t_conc,
                        "overlap_gain": gain})
        print(f"  {a:<14} {b:<14} {alone[a]:>8.3f} {alone[b]:>8.3f} "
              f"{t_seq:>9.3f} {t_conc:>9.3f} {gain:>7.3f}")

    best = max(r["overlap_gain"] for r in results)
    print(f"\n  best SM-vs-SM overlap gain {best:.3f}  "
          f"(1.00 = the recompute becomes free; 0.00 = a second stream buys nothing)")

    # ---- mechanism 2: the copy engines --------------------------------------------------------
    # Same estimator, but now A is a PCIe transfer and B is compute.  A gain near 1.00 here means
    # `o_h` can be moved to and from the host for free while the SMs work -- which converts the
    # O(S^2) FlashAttention forward recompute into a transfer that costs no GPU time and no GPU
    # memory beyond a double buffer.
    dma_results = []
    print(f"\n  {'A (DMA)':<14} {'B (compute)':<14} {'t_A':>8} {'t_B':>8} "
          f"{'seq':>9} {'conc':>9} {'gain':>8}")
    for a, fn_a in dma.items():
        for b in ("flash_bwd", "ffn_bwd"):
            t_conc = time_concurrent(fn_a, ops[b][0], args.reps)
            t_seq = alone[a] + alone[b]
            gain = (t_seq - t_conc) / min(alone[a], alone[b])
            dma_results.append({"a": a, "b": b, "t_a_ms": alone[a], "t_b_ms": alone[b],
                                "t_sequential_ms": t_seq, "t_concurrent_ms": t_conc,
                                "overlap_gain": gain})
            print(f"  {a:<14} {b:<14} {alone[a]:>8.3f} {alone[b]:>8.3f} "
                  f"{t_seq:>9.3f} {t_conc:>9.3f} {gain:>7.3f}")
    best_dma = max(r["overlap_gain"] for r in dma_results)
    print(f"\n  best DMA-vs-compute overlap gain {best_dma:.3f}  "
          f"({o_h_mib:.1f} MiB of o_h per layer at this shape)")

    out = {"cfg": cfg, "alone_ms": alone, "pairs": results, "best_overlap_gain": best,
           "dma_pairs": dma_results, "best_dma_overlap_gain": best_dma, "o_h_mib": o_h_mib,
           "gpu_state_before": state_before, "gpu_state_after": _gpu_state(),
           "torch": torch.__version__}
    path = args.out or f"results/hyclora/frontier/stream_overlap_seq{args.seq}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
