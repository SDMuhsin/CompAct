"""The peak-memory BUDGET: what every byte of the high-water mark is, as a percentage.

`probe_highwater.py` answers "which frame allocated at the peak, and how far below it is the next
window?".  That is the right question for *shaving* the peak.  This probe answers the prior
question -- **where is the peak's mass?** -- which is what decides whether shaving is worth doing
at all.

The peak is split three ways, and every term is measured, not modelled:

  peak = resident floor  +  forward-stored state live at the peak instant  +  backward transient

* **floor** -- what `torch.cuda.memory_allocated()` reads before the step begins: base weights,
  adapter parameters, optimizer state, the input batch.  Broken down by walking
  `named_parameters()` and the optimizer's state dict, so the categories sum to the measured
  number rather than to an estimate.
* **forward-stored** -- blocks allocated before `.backward()` was entered and still live when the
  curve reaches its maximum.  These are the rematerialisation checkpoints plus whatever the head /
  loss stack is holding.
* **backward transient** -- blocks allocated after `.backward()` was entered and live at the peak.

The forward/backward cut is exact, not inferred from frame names: a uniquely-sized sentinel tensor
is allocated immediately before `.backward()`, and its index in the allocator's event trace is the
boundary.  Frame labels cannot do this job -- `_fb_proj` runs in the forward, in the recompute and
in the gradient pass, so its blocks land on both sides of the cut.

Regime B (`--flce`) is required for any number that is compared against the tables in
CONTEXT.md 33.4 / 37.2; without it the fp32-logits CE stack sits above the whole decoder stack and
the budget describes the loss function rather than the method.

Usage:
  PYTHONPATH=src python src/probe_mem_budget.py --arm fb_min_fnorm_sdpa --flce --seq 1024
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict

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

MIB = 2 ** 20
# Chosen so no real tensor in the step collides with it: not a multiple of 512, so the caching
# allocator rounds it to a size no ordinary activation lands on.
SENTINEL_ELEMS = 7_777_777


def param_categories(model):
    """Resident parameter bytes bucketed by role.  Sums to the model's own parameter footprint."""
    cats = defaultdict(lambda: {"bytes": 0, "n": 0})
    for name, p in model.named_parameters():
        nb = p.numel() * p.element_size()
        if "lora_" in name or "vera_" in name or "lora_magnitude" in name:
            k = "adapter (trainable)"
        elif "embed_tokens" in name:
            k = "base: embedding"
        elif "lm_head" in name:
            k = "base: lm_head"
        elif ".mlp." in name:
            k = "base: FFN (gate/up/down)"
        elif ".self_attn." in name:
            k = "base: attention (q/k/v/o)"
        elif "norm" in name:
            k = "base: norms"
        else:
            k = "base: other"
        cats[k]["bytes"] += nb
        cats[k]["n"] += 1
    return cats


def optimizer_bytes(opt):
    tot, n = 0, 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v) and v.is_cuda:
                tot += v.numel() * v.element_size()
                n += 1
    return tot, n


def trace(device_index):
    snap = torch.cuda.memory._snapshot()
    return (snap.get("device_traces") or [])[device_index]


def replay(tr, baseline_bytes, boundary):
    """Running live-bytes curve, plus the live block set at two instants.

    Two maxima are tracked, not one: the global maximum (which lands in the backward), and the
    maximum reached while still in the FORWARD phase, i.e. at event index < `boundary`.  The gap
    between them is the entire headroom any backward-side saving can return -- shave the backward
    transient and the peak simply falls onto the forward's own high-water mark.
    """
    curve, cur, live = [], 0, {}
    peak_v, peak_i, live_at_peak = -1, -1, None
    fwd_v, fwd_i, live_at_fwd = -1, -1, None
    for i, ev in enumerate(tr):
        a = ev.get("action")
        if a == "alloc":
            live[ev.get("addr")] = (i, ev)
            cur += ev.get("size", 0)
        elif a == "free_completed":
            e = live.pop(ev.get("addr"), None)
            if e is not None:
                cur -= e[1].get("size", 0)
        curve.append(cur + baseline_bytes)
        if cur > peak_v:
            peak_v, peak_i, live_at_peak = cur, i, dict(live)
        if boundary is not None and i < boundary and cur > fwd_v:
            fwd_v, fwd_i, live_at_fwd = cur, i, dict(live)
    return (curve, peak_i, peak_v + baseline_bytes, (live_at_peak or {}),
            fwd_i, fwd_v + baseline_bytes, (live_at_fwd or {}))


def find_sentinel(tr, nbytes):
    """Event index of the sentinel allocation -- the exact forward/backward boundary."""
    for i, ev in enumerate(tr):
        if ev.get("action") == "alloc" and ev.get("size") == nbytes:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--unsloth_bf16_adapters", action="store_true")
    ap.add_argument("--flce", action="store_true")
    ap.add_argument("--no_head", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--top_k", type=int, default=22)
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

    pcats = param_categories(model)
    opt_b, opt_n = optimizer_bytes(opt)
    batch_b = sum(v.numel() * v.element_size() for v in batch.values() if torch.is_tensor(v))

    torch.cuda.memory._record_memory_history(context="all", stacks="python", max_entries=2000000)
    base = torch.cuda.memory_allocated()

    # `--no_head` must be honoured HERE, not only in the warm-up above.  The warm-up calls
    # `ph.step`, which branches on `ph._HEADLESS["on"]`; this measured window used to call
    # `model(**batch)` / `out.loss.backward()` unconditionally, so with `--no_head` the warm-up ran
    # decoder-only while the measured step ran with the head -- a SILENT no-op that mislabelled the
    # measurement as headless.  Found 2026-08-09 (CONTEXT.md §46).  Any "headless" number produced
    # by this probe before that date is really a WITH-head number; re-take it.
    # The branch mirrors `ph.step` exactly (same synthetic gradient, same seed) so the two agree.
    if ph._HEADLESS["on"]:
        inner = ph._inner_llama(model)
        hs = inner(input_ids=batch["input_ids"]).last_hidden_state
        g = ph._HEADLESS["grad"]
        if g is None or tuple(g.shape) != tuple(hs.shape) or g.dtype != hs.dtype:
            gen = torch.Generator(device="cpu").manual_seed(41)
            g = (torch.randn(tuple(hs.shape), generator=gen, dtype=torch.float32) * 1e-3
                 ).to(device=hs.device, dtype=hs.dtype)
            ph._HEADLESS["grad"] = g
        out = hs
        # Exact forward/backward cut -- see the comment on the with-head path below.
        sentinel = torch.empty(SENTINEL_ELEMS, dtype=torch.uint8, device=device)
        sent_bytes = sentinel.numel()
        hs.backward(g)
    else:
        out = model(**batch)
        # Exact forward/backward cut.  Allocated (and kept alive) across `.backward()` so it appears
        # in the trace at the boundary and never gets recycled into a real tensor mid-pass.
        sentinel = torch.empty(SENTINEL_ELEMS, dtype=torch.uint8, device=device)
        sent_bytes = sentinel.numel()
        out.loss.backward()
    torch.cuda.synchronize()
    measured_peak = torch.cuda.max_memory_allocated()

    tr = trace(device.index or 0)
    bnd = find_sentinel(tr, sent_bytes)
    (curve, peak_i, replayed_peak, live_at_peak,
     fwd_i, fwd_phase_max, live_at_fwd) = replay(tr, base, bnd)
    torch.cuda.memory._record_memory_history(enabled=None)
    del sentinel, out
    opt.zero_grad(set_to_none=True)

    # --- the three-way split, and the per-frame table underneath each side -------------------
    def split(live_set):
        fb = bb = 0
        ff = defaultdict(lambda: {"bytes": 0, "n": 0})
        bf = defaultdict(lambda: {"bytes": 0, "n": 0})
        blks = []
        for _addr, (i, ev) in live_set.items():
            nb = ev.get("size", 0)
            if nb == sent_bytes:
                continue                  # the sentinel is measurement scaffolding, not the model
            f = _frame_label(ev.get("frames"))
            in_fwd = bnd is not None and i < bnd
            if in_fwd:
                fb += nb
                ff[f]["bytes"] += nb
                ff[f]["n"] += 1
            else:
                bb += nb
                bf[f]["bytes"] += nb
                bf[f]["n"] += 1
            blks.append({"MiB": nb / MIB, "phase": "fwd" if in_fwd else "bwd", "frame": f})
        return fb, bb, ff, bf, blks

    fwd_b, bwd_b, fwd_frames, bwd_frames, blocks = split(live_at_peak)

    # At the FORWARD phase's own maximum every live block was allocated in the forward, so the
    # phase cut says nothing there.  The cut that does: which blocks SURVIVE to the backward peak.
    # Those are the persistent rematerialisation checkpoints; the rest is the forward's own
    # transient working set.  Keyed on allocation event index, which is unique -- addresses are
    # recycled by the caching allocator and would alias.
    peak_idx = {i for (i, _ev) in live_at_peak.values()}
    persist_b = ftrans_b = 0
    ftrans_frames = defaultdict(lambda: {"bytes": 0, "n": 0})
    for _addr, (i, ev) in live_at_fwd.items():
        nb = ev.get("size", 0)
        if nb == sent_bytes:
            continue
        if i in peak_idx:
            persist_b += nb
        else:
            ftrans_b += nb
            f = _frame_label(ev.get("frames"))
            ftrans_frames[f]["bytes"] += nb
            ftrans_frames[f]["n"] += 1

    def table(d):
        return [{"frame": k, "MiB": v["bytes"] / MIB, "n": v["n"]}
                for k, v in sorted(d.items(), key=lambda kv: -kv[1]["bytes"])[:args.top_k]]

    # The sentinel is live from just before `.backward()` to the end, so it raises the whole
    # backward half of the curve by a constant and must come out of the denominator.  It can only
    # MOVE the peak if the forward's own maximum sits within `sent_bytes` of the backward's, so
    # that margin is measured and reported rather than assumed.
    peak = measured_peak - sent_bytes
    # The peak is genuinely in the backward only if it still clears the forward's own maximum once
    # the sentinel is removed from it.  The forward-phase max never contains the sentinel, so this
    # is the exact test, and the margin IS the headroom of every backward-side saving.
    peak_is_in_backward = (replayed_peak - sent_bytes) > fwd_phase_max
    headroom = (replayed_peak - sent_bytes) - fwd_phase_max
    res = {
        "arm": args.arm, "seq": args.seq, "batch": args.batch, "flce": bool(args.flce),
        # Recorded so a reader can tell which window a number came from.  Before 2026-08-09 this
        # probe accepted `--no_head` and ignored it in the measured window (CONTEXT.md §46), and
        # emitted no receipt either way -- so an old artifact cannot be classified after the fact.
        "headless": bool(ph._HEADLESS["on"]),
        "model": cfg["model"],
        "peak_alloc_MiB": peak / MIB,
        "measured_peak_incl_sentinel_MiB": measured_peak / MIB,
        "sentinel_MiB": sent_bytes / MIB,
        "replayed_peak_MiB": replayed_peak / MIB,
        "replay_error_MiB": (measured_peak - replayed_peak) / MIB,
        "fwd_phase_max_MiB": fwd_phase_max / MIB,
        "peak_is_in_backward": bool(peak_is_in_backward),
        "backward_side_headroom_MiB": headroom / MIB,
        "budget_sums_to_peak_MiB": (base + fwd_b + bwd_b) / MIB,
        "at_fwd_phase_max": {
            "index": fwd_i,
            "persistent_MiB": persist_b / MIB,
            "fwd_transient_MiB": ftrans_b / MIB,
            "fwd_transient_frames": [{"frame": k, "MiB": v["bytes"] / MIB, "n": v["n"]}
                                     for k, v in sorted(ftrans_frames.items(),
                                                        key=lambda kv: -kv[1]["bytes"])[:args.top_k]],
        },
        "floor_MiB": base / MIB,
        "fwd_stored_live_at_peak_MiB": fwd_b / MIB,
        "bwd_transient_at_peak_MiB": bwd_b / MIB,
        "sentinel_index": bnd, "peak_index": peak_i, "n_events": len(tr),
        "peak_after_backward_entered": (bnd is not None and peak_i > bnd),
        "floor_breakdown": {k: {"MiB": v["bytes"] / MIB, "n": v["n"]}
                            for k, v in sorted(pcats.items(), key=lambda kv: -kv[1]["bytes"])},
        "optimizer_state_MiB": opt_b / MIB, "optimizer_state_tensors": opt_n,
        "input_batch_MiB": batch_b / MIB,
        "fwd_frames_at_peak": table(fwd_frames),
        "bwd_frames_at_peak": table(bwd_frames),
        "largest_blocks_at_peak": sorted(blocks, key=lambda b: -b["MiB"])[:args.top_k],
        "fb_policy": None,
    }
    if args.arm.startswith("fb_"):
        from flashffn import fb_policy_report
        res["fb_policy"] = fb_policy_report()

    out_path = args.out or os.path.join(RESULTS_DIR, f"membudget_{args.arm}_seq{args.seq}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)

    pk = peak / MIB
    print(f"\n=== {args.arm}  seq {args.seq}  batch {args.batch}  flce={args.flce} ===")
    print(f"  peak {pk:.2f} MiB (net of the {sent_bytes / MIB:.2f} MiB sentinel) "
          f"| budget sums to {(base + fwd_b + bwd_b) / MIB:.2f}")
    print(f"  peak event {peak_i} of {len(tr)}; backward entered at event {bnd}; "
          f"peak_in_backward={peak_is_in_backward}")
    print(f"  forward-phase max {fwd_phase_max / MIB:.2f} MiB "
          f"=> BACKWARD-SIDE HEADROOM {headroom / MIB:.2f} MiB ({100 * headroom / peak:.3f}% of peak)")
    print(f"    at that forward instant: persistent {persist_b / MIB:.2f} MiB "
          f"+ forward transient {ftrans_b / MIB:.2f} MiB")
    for r in res["at_fwd_phase_max"]["fwd_transient_frames"][:8]:
        print(f"      {r['MiB']:9.2f} MiB  x{r['n']:<4} {r['frame']}")
    print(f"\n  THE BUDGET")
    print(f"    resident floor            {base / MIB:9.2f} MiB  {100 * base / peak:6.2f}%")
    print(f"    forward-stored @ peak     {fwd_b / MIB:9.2f} MiB  {100 * fwd_b / peak:6.2f}%")
    print(f"    backward transient @ peak {bwd_b / MIB:9.2f} MiB  {100 * bwd_b / peak:6.2f}%")
    print(f"\n  FLOOR ({base / MIB:.2f} MiB = {100 * base / peak:.2f}% of peak)")
    for k, v in sorted(pcats.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"    {v['bytes'] / MIB:9.2f} MiB  {100 * v['bytes'] / peak:6.2f}%  {k} ({v['n']})")
    print(f"    {opt_b / MIB:9.2f} MiB  {100 * opt_b / peak:6.2f}%  AdamW state ({opt_n})")
    print(f"    {batch_b / MIB:9.2f} MiB  {100 * batch_b / peak:6.2f}%  input batch")
    print(f"\n  FORWARD-STORED, live at the peak instant, by allocating frame")
    for r in res["fwd_frames_at_peak"]:
        print(f"    {r['MiB']:9.2f} MiB  {100 * r['MiB'] * MIB / peak:6.2f}%  x{r['n']:<4} {r['frame']}")
    print(f"\n  BACKWARD TRANSIENT, live at the peak instant, by allocating frame")
    for r in res["bwd_frames_at_peak"]:
        print(f"    {r['MiB']:9.2f} MiB  {100 * r['MiB'] * MIB / peak:6.2f}%  x{r['n']:<4} {r['frame']}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
