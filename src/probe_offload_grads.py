"""Ground truth for the `o_h` offload: end-to-end gradients, ONE VARIANT PER PROCESS.

Why this file exists.  `route_b.md` §5b records that every convenient diagnostic for the offload
defect is unreliable -- a `torch.equal` round-trip verifier at fetch time reports 22/22 correct
while the FlashAttention backward downstream reads a stale buffer, because the synchronisation the
verifier performs is itself the fix.  It also records that running several variants in sequence in
one process shares caching-allocator state and produces answers that do not reproduce.  So:

  * the only ground truth is end-to-end gradients,
  * the only honest reference is another run of the SAME configuration (the FlashAttention backward
    is nondeterministic at ~1.5e-02 at these shapes, so a bitwise test is meaningless), and
  * each run gets its own process, writing to disk, with the comparison done offline.

Usage:
    python src/probe_offload_grads.py --run off --out results/.../off_a.pt
    python src/probe_offload_grads.py --run off --out results/.../off_b.pt     # the control
    python src/probe_offload_grads.py --run on  --out results/.../on.pt
    python src/probe_offload_grads.py --compare off_a.pt off_b.pt on.pt

`--digest` takes a 64-bit witness over `o_h` in the forward and again at fetch, accumulating
disagreements into a device counter read once at the end -- deliberately never returning a Python
value mid-step, so that it cannot hide the race the way a `torch.equal` verifier does.

**IT HIDES THE RACE ANYWAY, and that is itself a result worth keeping.**  With the defect restored
it reports `mismatch_at_fetch: 0` AND the gradients come out clean, where the same configuration
without `--digest` corrupts.  The witness allocates ~32 MiB of `int64` temporaries per call, and
that is enough allocator churn to change which block the landing buffer gets.  So route_b.md
section 5b's trap 1 is broader than it was written: it is not only *synchronising* reads that hide
this defect, it is any extra device work at the fetch point.  Keep the flag for exactly this
demonstration; do not use it to certify anything.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build(seq, batch, device, keep="attn", lora_r=16):
    """A LoRA-patched TinyLlama plus a fixed batch, identical in every process.

    Adapters are overwritten from a deterministic CPU generator, exactly as
    `probe_inplace_glu.build` does, so PEFT's own RNG use cannot make two processes disagree.
    """
    from transformers import AutoModelForCausalLM
    import flashffn
    from flashffn import apply_flash_block, apply_flash_final_norm
    from profile_hyclora import apply_family

    torch.manual_seed(41)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    m.config.use_cache = False
    m = apply_family(m, "lora", lora_r)
    m.to(device)
    m.train()

    g = torch.Generator(device="cpu").manual_seed(1234)
    for n, p in sorted(m.named_parameters()):
        if p.requires_grad and ("lora_" in n or "vera_" in n):
            p.data.copy_((torch.randn(tuple(p.shape), generator=g, dtype=torch.float32) * 0.02)
                         .to(device=p.device, dtype=p.dtype))

    apply_flash_block(m, keep=keep, verbose=False)
    apply_flash_final_norm(m)
    flashffn.fb_reset_counters()

    vocab = m.config.vocab_size
    gb = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, vocab, (batch, seq), generator=gb).to(device)
    return m, {"input_ids": ids, "labels": ids.clone(),
               "attention_mask": torch.ones_like(ids)}


def stress(m, batch, trials, churn):
    """Repeat the same step in ONE process and flag any step that disagrees with the first.

    The defect is a race and fires intermittently -- 1 of 3 processes at seq 1024 -- so a handful of
    cross-process runs has no power to confirm a fix.  This mode buys power cheaply: the only thing
    that has to be detected is a DISAGREEMENT between steps of the identical configuration, and
    corruption is ~1e+04 against a ~1.5e-02 FlashAttention-backward noise floor, six orders apart.
    No reference process is needed and nothing is read from the device mid-step.

    `churn` drops the landing-buffer pool between steps.  Without it the pool is warm after step 1
    and `prefetch` never calls `torch.empty` again, so every step after the first stops exercising
    the allocation path the defect lives in -- the test would look clean for the wrong reason.
    """
    import fb_offload
    ref, out = None, []
    for t in range(trials):
        if churn and t:
            fb_offload.reset()          # also drops the buffer pools
        grads, loss = one_step(m, batch)
        if ref is None:
            ref, ref_loss = grads, loss
            out.append(0.0)
            continue
        worst = max((((ref[k] - grads[k]).norm() / ref[k].norm()).item()
                     if ref[k].norm() > 0 else 0.0) for k in ref)
        out.append(worst)
        if loss != ref_loss:
            print(f"  !! step {t}: loss moved {ref_loss!r} -> {loss!r} "
                  f"(a backward-side defect cannot do this)")
    return out


def one_step(m, batch):
    random.seed(1234)
    torch.manual_seed(1234)
    m.zero_grad(set_to_none=True)
    out = m(**batch)
    out.loss.backward()
    grads = {n: p.grad.detach().float().to("cpu", copy=True)
             for n, p in m.named_parameters() if p.grad is not None}
    loss = float(out.loss.detach())
    del out
    return grads, loss


def _layer_of(name):
    """`...model.layers.13.self_attn...` -> 13, else None."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def compare(paths):
    ref = torch.load(paths[0], map_location="cpu", weights_only=False)
    print(f"reference: {paths[0]}  loss={ref['loss']!r}  offload={ref['offload']}")
    for p in paths[1:]:
        cand = torch.load(p, map_location="cpu", weights_only=False)
        a, b = ref["grads"], cand["grads"]
        keys = sorted(set(a) & set(b))
        worst, worst_k, per_layer = 0.0, "-", {}
        for k in keys:
            x, y = a[k], b[k]
            den = x.norm().item()
            rel = ((x - y).norm().item() / den) if den > 0 else 0.0
            if rel > worst:
                worst, worst_k = rel, k
            li = _layer_of(k)
            if li is not None:
                per_layer[li] = max(per_layer.get(li, 0.0), rel)
        same_loss = (cand["loss"] == ref["loss"])
        print(f"\n{p}  offload={cand['offload']}")
        print(f"  loss identical to reference: {same_loss}   "
              f"({ref['loss']!r} vs {cand['loss']!r})")
        print(f"  tensors compared: {len(keys)}   max rel-L2: {worst:.3e}  ({worst_k})")
        if cand.get("stats"):
            s = cand["stats"]
            print(f"  offload stats: stashed={s['stashed']} fetched={s['fetched']} "
                  f"blocked_waits={s['blocked_waits']} "
                  f"D2H={s['bytes_d2h'] / 2 ** 20:.0f} MiB H2D={s['bytes_h2d'] / 2 ** 20:.0f} MiB")
        if cand.get("digest"):
            print(f"  digest: {cand['digest']}")
        print("  per-layer max rel-L2 (layer: value), backward order is 21 -> 0:")
        line = "   "
        for li in sorted(per_layer, reverse=True):
            line += f" {li}:{per_layer[li]:.2e}"
            if len(line) > 100:
                print(line)
                line = "   "
        if line.strip():
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["on", "off"])
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--keep", default="attn")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--trials", type=int, default=0,
                    help="stress mode: N steps in one process, each compared against step 0")
    ap.add_argument("--no-churn", action="store_true",
                    help="stress mode: keep the landing-buffer pool warm between steps (then only "
                         "step 1 exercises the allocation path, so the test loses its teeth)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="stress mode: rel-L2 above which a step counts as corrupted; the floor is "
                         "~1.5e-02 and corruption is ~1e+04, so anything in between separates them")
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--no-lookahead", action="store_true")
    ap.add_argument("--barrier", default=None, choices=["event", "stream"])
    ap.add_argument("--alloc-stream", default=None, choices=["copy", "compute"],
                    help="'compute' reproduces the route_b.md section 5b defect exactly: the "
                         "landing buffer is taken from the compute stream's allocator pool, so the "
                         "block's previous tenant may still have kernels in flight over it")
    ap.add_argument("--wait-alloc", action="store_true",
                    help="blunt alternative fix: make the copy stream wait for the compute stream "
                         "before every H2D")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    import flashffn  # noqa: F401
    import fb_offload

    if args.digest:
        fb_offload.fb_offload_digest(True)
    if args.no_lookahead:
        fb_offload.fb_offload_lookahead(False)
    if args.barrier:
        fb_offload.fb_offload_barrier(args.barrier)
    if args.alloc_stream:
        fb_offload.fb_offload_alloc_stream(args.alloc_stream)
    if args.wait_alloc:
        fb_offload.fb_offload_wait_alloc(True)
    fb_offload.fb_offload_enable(args.run == "on")
    cfg = (f"alloc_stream={fb_offload._FB_OFFLOAD['alloc_stream']} "
           f"barrier={fb_offload._FB_OFFLOAD['barrier']} "
           f"wait_alloc={fb_offload._FB_OFFLOAD['wait_alloc']} "
           f"lookahead={fb_offload._FB_OFFLOAD['lookahead']}")

    m, batch = build(args.seq, args.batch, device, keep=args.keep)
    if args.trials:
        devs = stress(m, batch, args.trials, churn=not args.no_churn)
        worst = max(devs)
        n_bad = sum(1 for d in devs if d > args.threshold)
        print(f"stress offload={args.run} keep={args.keep} seq={args.seq} "
              f"trials={args.trials} churn={not args.no_churn}  {cfg}")
        print(f"  max deviation from step 0: {worst:.3e}   "
              f"steps over threshold {args.threshold:g}: {n_bad}/{len(devs) - 1}")
        print("  per-step: " + " ".join(f"{d:.1e}" for d in devs))
        print(f"  stats: {fb_offload.fb_offload_stats()}")
        print("  VERDICT: " + ("CORRUPTION DETECTED" if n_bad else "no corruption detected"))
        return
    for _ in range(args.steps):
        grads, loss = one_step(m, batch)
    stats = fb_offload.fb_offload_stats()
    digest = fb_offload.fb_offload_digest_report() if args.digest else None
    print(f"run offload={args.run} keep={args.keep} seq={args.seq} loss={loss!r}  {cfg}")
    print(f"  stats: {stats}")
    if digest is not None:
        print(f"  digest: {digest}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({"grads": grads, "loss": loss, "offload": args.run, "cfg": cfg,
                    "seq": args.seq, "keep": args.keep, "stats": stats,
                    "digest": digest}, args.out)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
