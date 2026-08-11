"""Probe 4 (attack items 3 + 7): independent throughput, one arm at a time, and the
sequence-length sweep that locates the fb_min_fnorm / hyclora_flash_q2 crossover.

Own timing loop (not `measure_headline`), own warm-up, own peak measurement.  Arms are
interleaved rep-by-rep and never run concurrently.
"""
import argparse, gc, json, os, statistics, subprocess, sys, time
import torch

sys.path.insert(0, "/workspace/CompAct/src")
from profile_hyclora import build_model, make_batch, step  # noqa

MiB = 2 ** 20
BASE = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "lora_r": 16, "q_bit": 2,
        "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
        "iteration_threshold": 5, "n_layers": 22}


def clocks():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10).stdout.strip().splitlines()
        return o
    except Exception:
        return []


def run_arm(arm, seq, batch, warm, timed):
    cfg = dict(BASE, seq=seq, batch=batch)
    dev = torch.device("cuda")
    torch.manual_seed(41)
    # "+fnorm" = give ANY arm the same fused model-level final RMSNorm that `fb_min_fnorm` gets.
    # `apply_flash_final_norm` is orthogonal to the block, so a like-for-like memory comparison
    # against gradient checkpointing has to offer it to the checkpointing arm too.
    extra_fnorm = arm.endswith("+fnorm")
    real = arm[:-len("+fnorm")] if extra_fnorm else arm
    m = build_model(real, cfg, dev, adapter_dtype="bf16")
    if extra_fnorm:
        from flashffn import apply_flash_final_norm
        apply_flash_final_norm(m)
    b = make_batch(cfg, dev, m.config.vocab_size)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(warm):
        step(m, b, opt)
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()
    floor = torch.cuda.memory_allocated()
    peaks = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        step(m, b, opt)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    ts = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = step(m, b, opt)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    counters = None
    if real.startswith("fb_"):
        from flashffn import fb_get_counters
        counters = fb_get_counters()
        assert counters["forward"] > 0 and counters["backward"] > 0, counters
        if "fb_min" in real:
            assert counters["recompute"] > 0, counters
    elif real.startswith("hyclora"):
        from hyclora.patch import get_counters
        counters = get_counters()
    r = {"arm": arm, "seq": seq, "batch": batch,
         "peak_alloc_mib": max(peaks) / MiB, "floor_mib": floor / MiB,
         "peak_minus_floor_mib": (max(peaks) - floor) / MiB,
         "ms_median": 1e3 * ts[len(ts) // 2], "ms_min": 1e3 * ts[0],
         "ms_iqr": 1e3 * (ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]),
         "loss": float(loss), "counters": counters, "clocks": clocks()}
    del m, b, opt
    gc.collect(); torch.cuda.empty_cache()
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fb_min_fnorm_sdpa,hyclora_flash_q2")
    ap.add_argument("--seqs", default="1024")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--timed", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = a.arms.split(",")
    rows = []
    for seq in [int(s) for s in a.seqs.split(",")]:
        for rep in range(a.reps):
            for arm in arms:
                try:
                    r = run_arm(arm, seq, a.batch, a.warm, a.timed)
                except torch.cuda.OutOfMemoryError as e:
                    r = {"arm": arm, "seq": seq, "batch": a.batch, "oom": str(e)[:120]}
                    gc.collect(); torch.cuda.empty_cache()
                r["rep"] = rep
                rows.append(r)
                print(f"seq={seq:6d} rep={rep} {arm:22s} "
                      f"peak={r.get('peak_alloc_mib', float('nan')):9.2f} MiB  "
                      f"ms={r.get('ms_median', float('nan')):8.2f} "
                      f"(iqr {r.get('ms_iqr', float('nan')):.2f})  {r.get('clocks')}", flush=True)
                json.dump(rows, open(a.out, "w"), indent=2)
    # summary
    print("\n== summary (min over reps of the median) ==")
    for seq in sorted({r["seq"] for r in rows}):
        vals = {}
        for arm in arms:
            rs = [r for r in rows if r["seq"] == seq and r["arm"] == arm and "ms_median" in r]
            if rs:
                vals[arm] = (min(r["ms_median"] for r in rs), max(r["peak_alloc_mib"] for r in rs))
        line = f"seq {seq:6d}: " + "  ".join(
            f"{k}={v[0]:.2f}ms/{v[1]:.0f}MiB" for k, v in vals.items())
        if len(vals) >= 2:
            ks = list(vals)
            a0, a1 = vals[ks[0]], vals[ks[1]]
            line += (f"   d_ms={100 * (a0[0] - a1[0]) / a1[0]:+.2f}%  "
                     f"d_mem={100 * (a0[1] - a1[1]) / a1[1]:+.2f}%")
        print(line)
