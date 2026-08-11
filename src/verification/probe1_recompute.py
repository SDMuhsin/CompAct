"""Probe 1: is the fused block ACTUALLY recomputing, and is nothing silently retained?

Independent of the project's own harness.  Measures, on the live model:
  (a) kernel-level call counts in forward vs backward for each `keep` level,
  (b) retained bytes after forward-before-backward (allocator truth, catches ctx-parked
      tensors that a saved_tensors walk would miss),
  (c) the tensors actually reachable from the fused node (saved_tensors + ctx attrs),
  (d) resident-floor drift across 12 steps (leak check).
"""
import gc, os, sys, json
import torch

sys.path.insert(0, "/workspace/CompAct/src")
import flashffn
from profile_hyclora import build_model, make_batch, step  # noqa

MiB = 2 ** 20
CFG = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "batch": 2, "seq": 1024, "lora_r": 16,
       "q_bit": 2, "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
       "iteration_threshold": 5, "n_layers": 22}

CALLS = {}


def instrument():
    """Wrap the leaf kernels so we can count them per phase."""
    names = ["fb_rmsnorm_forward", "fb_rmsnorm_backward", "_fb_flash_fwd", "_fb_flash_bwd",
             "_fb_proj", "_fb_rope_bsh", "_fb_head_to_hidden", "triton_silu_mul"]
    for n in names:
        orig = getattr(flashffn, n)
        if getattr(orig, "_probed", False):
            continue

        def mk(orig=orig, n=n):
            def f(*a, **k):
                CALLS[n] = CALLS.get(n, 0) + 1
                return orig(*a, **k)
            f._probed = True
            return f
        setattr(flashffn, n, mk())


def phase_counts(arm, seq=1024, batch=2):
    cfg = dict(CFG, seq=seq, batch=batch)
    torch.manual_seed(41)
    m = build_model(arm, cfg, torch.device("cuda"), adapter_dtype="bf16")
    b = make_batch(cfg, torch.device("cuda"), m.config.vocab_size)
    out = m(**b)
    CALLS.clear()
    out = m(**b)
    fwd = dict(CALLS)
    CALLS.clear()
    out.loss.backward()
    bwd = dict(CALLS)
    m.zero_grad(set_to_none=True)
    del out, m, b
    gc.collect(); torch.cuda.empty_cache()
    return fwd, bwd


def retained_after_forward(arm, seq=1024, batch=2, n_warm=8):
    """Bytes still live between forward and backward = the true stored-activation cost."""
    cfg = dict(CFG, seq=seq, batch=batch)
    torch.manual_seed(41)
    dev = torch.device("cuda")
    m = build_model(arm, cfg, dev, adapter_dtype="bf16")
    b = make_batch(cfg, dev, m.config.vocab_size)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(n_warm):
        step(m, b, opt)
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    floor = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = m(**b)
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    # drop the logits/loss graph tail so we can isolate the decoder-stack retention
    torch.cuda.reset_peak_memory_stats()
    out.loss.backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated()
    m.zero_grad(set_to_none=True)
    del out
    gc.collect(); torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    # leak check: 12 more steps, floor must not drift
    floors = []
    for i in range(12):
        step(m, b, opt)
        if i % 4 == 3:
            torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
            floors.append(torch.cuda.memory_allocated() / MiB)
    del m, b, opt
    gc.collect(); torch.cuda.empty_cache()
    return {"floor_MiB": floor / MiB, "retained_after_fwd_MiB": (after_fwd - floor) / MiB,
            "peak_fwd_MiB": peak_fwd / MiB, "peak_bwd_MiB": peak_bwd / MiB,
            "resident_after_bwd_MiB": (after - floor) / MiB,
            "floors_over_12_steps_MiB": floors}


def node_inventory(arm, seq=1024, batch=2):
    """What the fused node itself holds: saved_tensors + every tensor on ctx."""
    cfg = dict(CFG, seq=seq, batch=batch)
    torch.manual_seed(41)
    dev = torch.device("cuda")
    m = build_model(arm, cfg, dev, adapter_dtype="bf16")
    b = make_batch(cfg, dev, m.config.vocab_size)
    out = m(**b)
    # walk graph, find fused nodes
    seen, stack, nodes = set(), [out.loss.grad_fn], []
    while stack:
        n = stack.pop()
        if n is None or id(n) in seen:
            continue
        seen.add(id(n))
        if "FusedLoRABlock" in type(n).__name__:
            nodes.append(n)
        for nxt, _ in n.next_functions:
            stack.append(nxt)
    info = {"n_fused_nodes": len(nodes)}
    if nodes:
        n0 = nodes[0]
        storages, rows = {}, []
        try:
            sts = n0.saved_tensors
        except Exception as e:  # noqa
            sts = []
            info["saved_tensors_error"] = repr(e)
        for i, t in enumerate(sts):
            if t is None:
                continue
            key = t.untyped_storage().data_ptr()
            storages[key] = t.untyped_storage().nbytes()
            rows.append({"i": i, "shape": list(t.shape), "dtype": str(t.dtype),
                         "MiB": t.untyped_storage().nbytes() / MiB, "is_param": t.requires_grad})
        info["saved"] = rows
        info["unique_saved_MiB"] = sum(storages.values()) / MiB
        # ctx-parked attributes
        extra = {}
        for k in ("flash_meta", "shape", "heads", "scales", "eps", "keep", "scale"):
            v = getattr(n0, k, "<absent>")
            if isinstance(v, tuple):
                v = [(("T", list(x.shape), str(x.dtype), x.numel() * x.element_size())
                      if torch.is_tensor(x) else x) for x in v]
            extra[k] = str(v)
        info["ctx_attrs"] = extra
    out.loss.backward()
    del out, m, b
    gc.collect(); torch.cuda.empty_cache()
    return info


if __name__ == "__main__":
    res = {}
    instrument()
    for arm in ("fb_full_sdpa", "fb_glu_sdpa", "fb_min_fnorm_sdpa"):
        f, bw = phase_counts(arm)
        res[arm] = {"kernel_calls_forward": f, "kernel_calls_backward": bw}
        print(arm, "FWD", f, "\n      BWD", bw, flush=True)
    for arm in ("fb_min_fnorm_sdpa", "fb_full_sdpa"):
        res[arm]["node_inventory"] = node_inventory(arm)
        print(arm, json.dumps(res[arm]["node_inventory"], indent=1)[:2000], flush=True)
    for arm in ("baseline_sdpa", "gc_manual_sdpa", "hyclora_flash_q2", "fb_min_fnorm_sdpa"):
        r = retained_after_forward(arm)
        res.setdefault(arm, {})["retention"] = r
        print(arm, json.dumps(r), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe1.json"), "w") as fh:
        json.dump(res, fh, indent=2)
