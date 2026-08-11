"""Where the high-water mark actually is, as a CURVE rather than a single point.

CONTEXT.md section 33.5 asks which allocation sets the peak.  `profile_hyclora.peak_composition`
answers that for one instant.  This probe answers the follow-up question that decides whether any
proposed saving is worth implementing: **how far does the second-highest water mark sit below the
first?**  A 44 MiB cut at the peak is worth 44 MiB only if nothing else comes within 44 MiB of it.

Method: record the CUDA allocator's own history over one clean training step, replay it to get the
running live-bytes curve, and report

  * the global maximum and the python frame that allocated there,
  * the local maxima of that curve, separated by a caller-supplied drop, each with its frame,
  * `headroom_MiB` -- global max minus the highest local maximum that is NOT in the same window,
    which is the most any single-window optimisation can possibly return.

Regime B (Liger FusedLinearCrossEntropy) is applied with `--flce`, matching
`fair_comparison_protocol.md`; without it the LM-head/CE stack sits ~875 MiB above everything in
the decoder stack and the curve says nothing about the block.

Usage:
  python src/probe_highwater.py --arm fb_min_fnorm_sdpa --flce --seq 1024 --out results/....json
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_WANT_LIGER = "--flce" in sys.argv
if _WANT_LIGER:
    _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "temp", "liger_pkgs")
    if os.path.isdir(_lp) and _lp not in sys.path:
        sys.path.insert(0, _lp)

import torch  # noqa: E402

import profile_hyclora as ph  # noqa: E402
import profile_unsloth as pu  # noqa: E402
from profile_hyclora import make_batch, step, _frame_label, RESULTS_DIR  # noqa: E402


def live_curve(device_index, baseline_bytes):
    """Replay the allocator history into a running live-bytes curve.

    Returns (curve, events) where curve[i] is the live byte count after event i and events[i] is
    the raw allocator event, so a frame label can be attached to any index.
    """
    snap = torch.cuda.memory._snapshot()
    tr = (snap.get("device_traces") or [])[device_index]
    curve, cur, live = [], 0, {}
    for ev in tr:
        a = ev.get("action")
        if a == "alloc":
            live[ev.get("addr")] = ev
            cur += ev.get("size", 0)
        elif a == "free_completed":
            e = live.pop(ev.get("addr"), None)
            if e is not None:
                cur -= e.get("size", 0)
        curve.append(cur + baseline_bytes)
    return curve, tr


def local_maxima(curve, events, min_drop_bytes, top_k):
    """Peaks of the curve separated by a drop of at least `min_drop_bytes`.

    A plain argmax-per-window would report forty adjacent allocations inside one GEMM as forty
    peaks.  Requiring the curve to fall by `min_drop` before a new peak is admitted makes each
    reported row a genuinely distinct high-water *window*.
    """
    peaks = []
    best_i, best_v, since = -1, -1, 0
    for i, v in enumerate(curve):
        if v > best_v:
            best_v, best_i = v, i
        if best_v - v >= min_drop_bytes:
            peaks.append((best_i, best_v))
            best_v, best_i = v, i
        since = i
    if best_i >= 0:
        peaks.append((best_i, best_v))
    peaks.sort(key=lambda p: -p[1])
    out = []
    for i, v in peaks[:top_k]:
        out.append({
            "index": i,
            "live_MiB": v / 2 ** 20,
            "frame": _frame_label(events[i].get("frames")),
            "alloc_MiB": events[i].get("size", 0) / 2 ** 20,
        })
    return out, since


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"])
    # `build_unsloth` reads this off `args` via getattr and silently defaults to False, which
    # leaves PEFT's fp32 adapters in place -- the landmine of `fair_comparison_protocol.md` B.7,
    # worth +70.69 MiB of resident floor at seq1024/batch2.  Without this flag the probe cannot
    # reproduce the protocol-matched unsloth arm at all.
    ap.add_argument("--unsloth_bf16_adapters", action="store_true",
                    help="cast unsloth's adapters to bf16 (required for a matched comparison)")
    ap.add_argument("--flce", action="store_true")
    ap.add_argument("--no_head", action="store_true")
    ap.add_argument("--model", default=None,
                    help="HF model id; every shape field is derived from its config")
    ap.add_argument("--min_drop_MiB", type=float, default=8.0)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ph._HEADLESS["on"] = bool(args.no_head)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cfg = ph.make_cfg(args.batch, args.seq, model=args.model or ph.DEFAULT_MODEL)

    torch.manual_seed(41)
    model = pu.build(args.arm, cfg, device, args)
    vocab = model.config.vocab_size
    batch = make_batch(cfg, device, vocab)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    for _ in range(max(3, cfg["iteration_threshold"] + 3)):
        step(model, batch, opt)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.memory._record_memory_history(context="all", stacks="python", max_entries=2000000)
    base = torch.cuda.memory_allocated()
    out = model(**batch)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    measured_peak = torch.cuda.max_memory_allocated()
    curve, events = live_curve(device.index or 0, base)
    torch.cuda.memory._record_memory_history(enabled=None)
    del out

    peaks, n = local_maxima(curve, events, int(args.min_drop_MiB * 2 ** 20), args.top_k)
    gmax = max(curve)

    # The ceiling on any single-window optimisation: for every allocating frame, the highest the
    # curve ever reached while THAT frame was allocating.  If the peak window is shaved, the peak
    # falls to the highest row here that belongs to a different window -- so this table, not the
    # peak alone, is what says whether a proposed saving is worth implementing.
    by_frame = {}
    for i, ev in enumerate(events):
        if ev.get("action") != "alloc":
            continue
        f = _frame_label(ev.get("frames"))
        if curve[i] > by_frame.get(f, 0):
            by_frame[f] = curve[i]
    max_live_by_frame = [{"frame": k, "max_live_MiB": v / 2 ** 20}
                         for k, v in sorted(by_frame.items(), key=lambda kv: -kv[1])[:40]]
    res = {
        "arm": args.arm, "seq": args.seq, "batch": args.batch, "flce": bool(args.flce),
        "headless": bool(args.no_head),
        "baseline_resident_MiB": base / 2 ** 20,
        "measured_peak_alloc_MiB": measured_peak / 2 ** 20,
        "replayed_peak_MiB": gmax / 2 ** 20,
        "replay_error_MiB": (measured_peak - gmax) / 2 ** 20,
        "n_alloc_events": n + 1,
        "min_drop_MiB": args.min_drop_MiB,
        "local_maxima": peaks,
        "max_live_by_frame": max_live_by_frame,
        "second_window_gap_MiB": (peaks[0]["live_MiB"] - peaks[1]["live_MiB"]) if len(peaks) > 1
                                 else None,
        "fb_policy": None,
    }
    if args.arm.startswith("fb_"):
        from flashffn import fb_policy_report
        res["fb_policy"] = fb_policy_report()

    out_path = args.out or os.path.join(RESULTS_DIR, f"highwater_{args.arm}_seq{args.seq}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"\narm {args.arm}  seq {args.seq}  flce={args.flce}")
    print(f"  measured peak {measured_peak / 2**20:.2f} MiB | replayed {gmax / 2**20:.2f} "
          f"| floor {base / 2**20:.2f} | replay err {(measured_peak - gmax) / 2**20:+.3f}")
    print(f"  distinct high-water windows (drop >= {args.min_drop_MiB} MiB):")
    for p in peaks:
        print(f"    {p['live_MiB']:9.2f} MiB  (-{gmax / 2**20 - p['live_MiB']:7.2f})  "
              f"alloc {p['alloc_MiB']:6.2f}  {p['frame']}")
    print(f"  highest the curve reaches under each allocating frame:")
    for r in max_live_by_frame[:18]:
        print(f"    {r['max_live_MiB']:9.2f} MiB  (-{gmax / 2**20 - r['max_live_MiB']:7.2f})  "
              f"{r['frame']}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
