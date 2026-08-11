"""Probe 2 (attack item 5): is the seq-1024 divergence really FA-backward nondeterminism,
or a systematic recompute error hiding under a noise floor?

Test A -- decisive: capture EVERY intermediate the forward produces and EVERY intermediate the
          keep='min' backward recomputes, and compare them bitwise, tensor by tensor.
          If they are bitwise identical the recompute cannot be the source of any divergence.
Test B -- torch.use_deterministic_algorithms(True): does FlashAttention's backward become
          reproducible, and if so do fb_full and fb_min agree bitwise at seq 1024?
"""
import gc, os, sys, json, argparse
import torch

sys.path.insert(0, "/workspace/CompAct/src")
import flashffn
from profile_hyclora import build_model, make_batch

CFG = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "batch": 2, "seq": 1024, "lora_r": 16,
       "q_bit": 2, "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
       "iteration_threshold": 5, "n_layers": 22}
PHASE = {"p": "fwd"}
LOG = {"fwd": {}, "bwd": {}}


def _rec(name, outs):
    d = LOG[PHASE["p"]].setdefault(name, [])
    d.append(tuple((t.detach().clone() if torch.is_tensor(t) else t) for t in outs))


_INSTRUMENTED = {"done": False}


def instrument():
    # Idempotent: `test_A` may be called more than once per process (--keep min,attn), and
    # wrapping the already-wrapped kernels would record every call twice and mis-index every
    # forward<->backward pair.
    if _INSTRUMENTED["done"]:
        return
    _INSTRUMENTED["done"] = True
    o_norm, o_proj, o_rope, o_flash, o_h2h, o_silu = (
        flashffn.fb_rmsnorm_forward, flashffn._fb_proj, flashffn._fb_rope_bsh,
        flashffn._fb_flash_fwd, flashffn._fb_head_to_hidden, flashffn.triton_silu_mul)

    def norm(*a, **k):
        r = o_norm(*a, **k); _rec("norm", r); return r

    def proj(*a, **k):
        r = o_proj(*a, **k); _rec("proj", r); return r

    def rope(*a, **k):
        r = o_rope(*a, **k); _rec("rope", (r,)); return r

    def flash(*a, **k):
        r = o_flash(*a, **k); _rec("flash", (r[0], r[1])); return r

    def h2h(*a, **k):
        r = o_h2h(*a, **k); _rec("h2h", (r,)); return r

    def silu(*a, **k):
        r = o_silu(*a, **k); _rec("silu", (r,)); return r

    flashffn.fb_rmsnorm_forward = norm
    flashffn._fb_proj = proj
    flashffn._fb_rope_bsh = rope
    flashffn._fb_flash_fwd = flash
    flashffn._fb_head_to_hidden = h2h
    flashffn.triton_silu_mul = silu


def cmp_pairs(fwd, bwd, per_layer_f, per_layer_b, n_layers=22, fwd_offset=0):
    """backward call j of layer group -> forward call index."""
    bad, n = [], 0
    for j, tb in enumerate(bwd):
        lb, within = divmod(j, per_layer_b)
        layer = n_layers - 1 - lb
        fi = fwd_offset + layer * per_layer_f + within
        tf = fwd[fi]
        for a, b in zip(tf, tb):
            if not torch.is_tensor(a):
                continue
            n += 1
            if not torch.equal(a, b):
                bad.append({"j": j, "layer": layer, "within": within,
                            "max_abs": float((a.float() - b.float()).abs().max()),
                            "shape": list(a.shape)})
    return n, bad


def test_A(seq, keep="min", family="lora"):
    """Every tensor the forward produces vs every tensor the backward RECOMPUTES, bitwise.

    `keep='min'` recomputes all five kernel families and yields 462 compared tensors.
    `keep='attn'` (2026-08-03) stores FlashAttention's output instead of re-running the O(S^2)
    forward, so `flash` and the recompute call of `h2h` are simply absent from backward -- their
    66 tensors move from "recomputed and compared" to "stored verbatim", which is bitwise by
    construction.  The remaining 396 must still be 0-mismatch, and that is what this asserts.
    """
    cfg = dict(CFG, seq=seq)
    dev = torch.device("cuda")
    torch.manual_seed(41)
    m = build_model(f"fb_{keep}_sdpa+{family}", cfg, dev, adapter_dtype="bf16")
    # perturb whichever factor the family zero-initialises, so every path is live
    g = torch.Generator(device=dev).manual_seed(7)
    for n_, p in m.named_parameters():
        if any(t in n_ for t in ("lora_B", "lora_E", "vera_lambda_b")):
            p.data.normal_(0, 0.02, generator=g)
    b = make_batch(cfg, dev, m.config.vocab_size)
    out = m(**b)                      # warm (triton autotune) - not logged
    out.loss.backward(); m.zero_grad(set_to_none=True); del out
    gc.collect(); torch.cuda.empty_cache()

    instrument()
    PHASE["p"] = "fwd"; LOG["fwd"].clear(); LOG["bwd"].clear()
    out = m(**b)
    PHASE["p"] = "bwd"
    out.loss.backward()
    res = {}
    # norm: fwd 2/layer (+1 final norm only in _fnorm arms, absent here), bwd 2/layer
    # DoRA's magnitude gradient needs the PRE-scale down projection, which nothing else in the
    # recompute produces, so its backward makes 7 projection calls per layer instead of 6.
    proj_pb = 7 if family == "dora" else 6
    families = [("norm", 2, 2), ("proj", 7, proj_pb), ("rope", 2, 2), ("flash", 1, 1),
                ("h2h", 1, 1)]
    if keep == "attn":
        # `attn` never re-runs the attention forward, so `flash` has no backward call and `h2h`'s
        # only backward call is `grad_v`'s layout fold -- not a recompute of anything.
        families = [f for f in families if f[0] not in ("flash", "h2h")]
    for key, pf, pb in families:
        f, bw = LOG["fwd"].get(key, []), LOG["bwd"].get(key, [])
        if key == "h2h":
            # backward calls h2h twice per layer: once in the recompute, once for grad_v.
            bw = [bw[i] for i in range(len(bw)) if i % 2 == 0]
        n, bad = cmp_pairs(f, bw, pf, pb)
        res[key] = {"n_fwd_calls": len(f), "n_bwd_calls": len(bw), "n_tensors_compared": n,
                    "n_mismatched": len(bad), "examples": bad[:4]}
        print(f"  {key:6s} fwd={len(f):4d} bwd={len(bw):4d} compared={n:4d} "
              f"MISMATCH={len(bad)}", flush=True)
    m.zero_grad(set_to_none=True)
    del out, m, b
    LOG["fwd"].clear(); LOG["bwd"].clear()
    gc.collect(); torch.cuda.empty_cache()
    return res


def grads_of(arm, cfg, ids_state):
    dev = torch.device("cuda")
    torch.manual_seed(41)
    m = build_model(arm, cfg, dev, adapter_dtype="bf16")
    g = torch.Generator(device=dev).manual_seed(7)
    for n_, p in m.named_parameters():
        if "lora_B" in n_:
            p.data.normal_(0, 0.02, generator=g)
    b = ids_state
    out = m(**b)
    out.loss.backward()
    gr = {n_: p.grad.detach().clone() for n_, p in m.named_parameters() if p.requires_grad}
    lg = out.logits.detach().clone()
    m.zero_grad(set_to_none=True)
    del out, m
    gc.collect(); torch.cuda.empty_cache()
    return gr, lg


def relL2(a, b):
    import statistics
    v = [float((a[n].float() - b[n].float()).norm()) / (float(a[n].float().norm()) + 1e-30)
         for n in a]
    return {"median": statistics.median(v), "max": max(v),
            "n_bitwise_equal": sum(1 for n in a if torch.equal(a[n], b[n])), "n": len(v)}


def test_B(seq, deterministic):
    cfg = dict(CFG, seq=seq)
    dev = torch.device("cuda")
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    gen = torch.Generator(device="cpu").manual_seed(41)
    ids = torch.randint(0, 32000, (cfg["batch"], cfg["seq"]), generator=gen).to(dev)
    batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
    runs = {}
    for tag, arm in (("full1", "fb_full_sdpa"), ("full2", "fb_full_sdpa"),
                     ("min1", "fb_min_sdpa"), ("min2", "fb_min_sdpa"),
                     ("glu1", "fb_glu_sdpa")):
        runs[tag] = grads_of(arm, cfg, batch)
        print(f"  ran {tag}", flush=True)
    out = {}
    for a, b in (("full1", "full2"), ("min1", "min2"), ("full1", "min1"), ("full2", "min2"),
                 ("full1", "min2"), ("full2", "min1"), ("full1", "glu1")):
        r = relL2(runs[a][0], runs[b][0])
        r["logits_bitwise"] = bool(torch.equal(runs[a][1], runs[b][1]))
        out[f"{a}_vs_{b}"] = r
        print(f"  {a:6s} vs {b:6s}  relL2 med={r['median']:.4e} max={r['max']:.4e} "
              f"bitwise {r['n_bitwise_equal']}/{r['n']} logits_bw={r['logits_bitwise']}",
              flush=True)
    if deterministic:
        torch.use_deterministic_algorithms(False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--tests", default="A,B,Bdet")
    ap.add_argument("--family", default="lora",
                    help="adapter family for Test A: lora|dora|adalora|dylora|vera|full|frozen")
    ap.add_argument("--keep", default="min", help="which keep level Test A instruments "
                                                  "(comma-separated: min,attn)")
    a = ap.parse_args()
    res = {"seq": a.seq}
    if "A" in a.tests.split(","):
        for _keep in a.keep.split(","):
            print(f"== TEST A: recompute vs forward, bitwise, seq {a.seq}, keep={_keep} ==",
                  flush=True)
            key = "A_recompute_bitwise" + ("" if _keep == "min" else f"_{_keep}")
            res[key] = test_A(a.seq, keep=_keep, family=a.family)
    if "B" in a.tests.split(","):
        print(f"== TEST B: nondeterministic baseline, seq {a.seq} ==", flush=True)
        res["B_nondet"] = test_B(a.seq, False)
    if "Bdet" in a.tests.split(","):
        print(f"== TEST B': use_deterministic_algorithms(True), seq {a.seq} ==", flush=True)
        res["B_det"] = test_B(a.seq, True)
    _sfx = "" if a.family == "lora" else f"_{a.family}"
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"probe2_seq{a.seq}{_sfx}.json")
    json.dump(res, open(p, "w"), indent=2)
    print("->", p)
