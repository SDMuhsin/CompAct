"""Probe 8 -- rematerialisation is a SILENT correctness hazard, measured.

Companion to `llmdocs/trackers/remat_certificate.md`.  Nothing here is about our method; this
probe establishes the *problem*, on this machine, with artifacts on disk.

The premise every rematerialising system rests on is that the recomputed forward equals the
forward that actually ran.  `torch.utils.checkpoint`, unsloth's block checkpointer, HyC-LoRA's
fused decoder layer, NVIDIA Transformer Engine and `torch.compile`'s min-cut partitioner all
assume it and none of them checks it at the value level.  PyTorch ships the *shape* of a check
and explicitly declines to make it a value check:

    torch/utils/checkpoint.py:1063   _default_meta_extractor -> {"shape", "dtype", "device"}
    torch/utils/checkpoint.py:461    "compares the shapes, dtypes, and devices of the recomputed
                                      tensors against those the saved tensors ...
                                      Please open an issue if you would like to see more
                                      determinism checks."
    torch/utils/checkpoint.py:466    "... if ``use_reentrant=True``, the determinism check is
                                      always disabled."

So a divergence that preserves shape/dtype/device is invisible, and under the reentrant variant
every divergence is invisible.  The cases below measure what that costs.

CASES
-----
S1  same-shape value divergence inside a checkpointed region, NON-reentrant with the shipped
    `determinism_check="default"`.  Does it raise?  How wrong is the gradient?
S2  the same question for RNG state: dropout inside the region with `preserve_rng_state=False`
    -- the setting unsloth passes at `unsloth/models/llama.py`'s checkpoint call site.
S3  model level: TinyLlama-1.1B + PEFT LoRA with `lora_dropout>0`, decoder-layer checkpointing,
    `preserve_rng_state` False vs True vs an unwrapped reference.  Per-parameter gradient damage,
    and a loss trajectory showing the corrupted run still trains and never raises.
S4  model level: DyLoRA under REENTRANT checkpointing.  `fused_block.md` §9.6 records that the
    non-reentrant variant catches this by accident (the sampled rank changes the *shape*); the
    reentrant variant is what `train_glue.py --gradient_checkpointing` actually uses and has no
    check at all.

Every case reports whether an exception was raised, whether the loss stayed finite, and how far
the gradients moved -- because "silent" is the whole claim and it has to be demonstrated, not
asserted.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -W ignore src/verification/probe8_remat_divergence.py \
        --out results/certificate/probe8_remat_divergence.json
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "batch": 2, "seq": 256, "lora_r": 16,
       "n_layers": 22}
FB_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def grad_delta(a, b):
    """rel-L2 and cosine of two {name: tensor} gradient dicts, keyed on `a`."""
    rel, cos, bitwise = [], [], 0
    for n in a:
        if n not in b or a[n] is None or b[n] is None:
            continue
        x, y = a[n].float().flatten(), b[n].float().flatten()
        rel.append(float((x - y).norm() / (x.norm() + 1e-30)))
        cos.append(float(F.cosine_similarity(x, y, dim=0)))
        bitwise += int(torch.equal(a[n], b[n]))
    return {"n": len(rel), "rel_l2_median": statistics.median(rel), "rel_l2_max": max(rel),
            "cos_min": min(cos), "cos_median": statistics.median(cos),
            "n_bitwise_equal": bitwise}


def _wrap_layers(model, reentrant, preserve_rng):
    """Decoder-layer granularity `torch.utils.checkpoint`, with both knobs exposed.

    `profile_hyclora._wrap_forward_checkpoint` hard-codes PyTorch's defaults; this probe has to
    vary `preserve_rng_state`, which is the whole point of S2/S3, so it wraps here instead of
    importing that one.
    """
    from profile_hyclora import _decoder_layers
    layers = [m for _n, m in _decoder_layers(model)]
    n = 0
    for layer in layers:
        orig = layer.forward

        def fwd(*a, _o=orig, **k):
            if not (torch.is_grad_enabled()):
                return _o(*a, **k)

            # The reentrant variant takes positional args only, so the keyword arguments a
            # `LlamaDecoderLayer` is called with (attention_mask, position_embeddings, ...) are
            # captured in a closure -- exactly what `transformers`' own `create_custom_forward`
            # does. None of them requires grad, so nothing is lost by keeping them off the
            # checkpoint boundary.
            def run(*args, _k=k):
                return _o(*args, **_k)
            return cp.checkpoint(run, *a, use_reentrant=reentrant,
                                 preserve_rng_state=preserve_rng)
        layer.forward = fwd
        n += 1
    # CONTEXT.md §31.9 / §14.5: under the REENTRANT variant the embedding output does not require
    # grad, so every checkpointed block is detached and only the head trains -- or, if nothing
    # downstream requires grad at all, backward raises "element 0 of tensors does not require
    # grad". Walk to whichever module actually owns the method; a bare DyLoRA-patched model is not
    # a PeftModel and does not forward it.
    hooked = False
    for m in ([model] + [mm for _n, mm in model.named_modules()]):
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
            hooked = True
            break
    if not hooked:
        raise RuntimeError("no module exposes enable_input_require_grads(); reentrant "
                           "checkpointing would silently train only the head")
    return n


def _build(family, dropout, device, seed=41):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_pretrained(CFG["model"], attn_implementation="sdpa",
                                             torch_dtype=torch.bfloat16)
    m.config.use_cache = False
    if family == "lora":
        from peft import get_peft_model, LoraConfig
        m = get_peft_model(m, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_r"],
                                         lora_dropout=dropout, bias="none",
                                         task_type="CAUSAL_LM", target_modules=FB_TARGETS))
    elif family == "dylora":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "src"))
        from dylora import get_dylora_model
        m = get_dylora_model(m, FB_TARGETS, r=CFG["lora_r"], alpha=CFG["lora_r"],
                             dropout=dropout)
    m.to(device=device, dtype=torch.bfloat16)
    m.train()
    # lora_B initialises to zero, so every downstream gradient would be structurally zero and a
    # corruption would be invisible for the wrong reason.  Perturb it, deterministically.
    g = torch.Generator(device=device).manual_seed(7)
    for n_, p in m.named_parameters():
        if "lora_B" in n_:
            p.data.normal_(0, 0.02, generator=g)
    return m


def _batch(device, vocab):
    gen = torch.Generator(device="cpu").manual_seed(41)
    ids = torch.randint(0, vocab, (CFG["batch"], CFG["seq"]), generator=gen).to(device)
    return {"input_ids": ids, "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)}


def _grads(model, batch, torch_seed=None, py_seed=None):
    """One forward+backward, returning the trainable gradients, the loss, and whether it raised."""
    import random
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    if py_seed is not None:
        random.seed(py_seed)
    model.zero_grad(set_to_none=True)
    raised = None
    try:
        out = model(**batch)
        out.loss.backward()
    except Exception as e:                                   # noqa: BLE001 -- the point is to see it
        return None, None, f"{type(e).__name__}: {str(e)[:200]}"
    gr = {n: p.grad.detach().clone() for n, p in model.named_parameters()
          if p.requires_grad and p.grad is not None}
    loss = float(out.loss)
    model.zero_grad(set_to_none=True)
    del out
    return gr, loss, raised


# ------------------------------------------------------------------------------------------
# S1 -- a same-shape value divergence is invisible to PyTorch's determinism check
# ------------------------------------------------------------------------------------------
def case_s1(device):
    torch.manual_seed(0)
    x = torch.randn(8, 16, device=device)
    w = torch.randn(16, 16, device=device, requires_grad=True)
    st = {"n": 0}

    def f(x, w):
        # Identical shape, dtype and device on both passes; only the VALUE differs.  This is the
        # shape of every real recompute-divergence bug: a schedule counter, a resampled mask, a
        # calibration threshold that moved, a codec whose state advanced.
        st["n"] += 1
        return (x * (1.0 if st["n"] == 1 else 1.5)) @ w

    res = {}
    for tag, reentrant in (("non_reentrant", False), ("reentrant", True)):
        st["n"] = 0
        w.grad = None
        raised = None
        try:
            kw = {} if reentrant else {"determinism_check": "default"}
            y = cp.checkpoint(f, x, w, use_reentrant=reentrant, **kw)
            y.sum().backward()
        except Exception as e:                               # noqa: BLE001
            raised = f"{type(e).__name__}: {str(e)[:160]}"
        res[tag] = {"raised": raised,
                    "grad_norm": None if w.grad is None else float(w.grad.norm())}
    st["n"] = 0
    w.grad = None
    f(x, w).sum().backward()
    res["true_grad_norm"] = float(w.grad.norm())
    for tag in ("non_reentrant", "reentrant"):
        gn = res[tag]["grad_norm"]
        res[tag]["rel_error_vs_true"] = (
            None if gn is None else abs(gn - res["true_grad_norm"]) / res["true_grad_norm"])
    return res


# ------------------------------------------------------------------------------------------
# S2 -- RNG-state divergence, the `preserve_rng_state=False` case
# ------------------------------------------------------------------------------------------
def case_s2(device):
    torch.manual_seed(0)
    x = torch.randn(8, 16, device=device)
    w = torch.randn(16, 16, device=device, requires_grad=True)

    def g(x, w):
        return F.dropout(x, p=0.5, training=True) @ w

    out = {}
    for tag, preserve in (("preserve_rng_state_False", False),
                          ("preserve_rng_state_True", True)):
        w.grad = None
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        raised = None
        try:
            y = cp.checkpoint(g, x, w, use_reentrant=True, preserve_rng_state=preserve)
            y.sum().backward()
        except Exception as e:                               # noqa: BLE001
            raised = f"{type(e).__name__}: {str(e)[:160]}"
        out[tag] = {"raised": raised, "grad": w.grad.detach().clone()}
    a, b = out["preserve_rng_state_False"]["grad"], out["preserve_rng_state_True"]["grad"]
    return {"preserve_rng_state_False_raised": out["preserve_rng_state_False"]["raised"],
            "preserve_rng_state_True_raised": out["preserve_rng_state_True"]["raised"],
            "grad_rel_l2_False_vs_True": float((a - b).norm() / b.norm()),
            "grad_cos_False_vs_True": float(F.cosine_similarity(
                a.flatten(), b.flatten(), dim=0))}


# ------------------------------------------------------------------------------------------
# S3 -- model level: LoRA dropout under decoder-layer checkpointing
# ------------------------------------------------------------------------------------------
def case_s3(device, dropout=0.1, traj_steps=20):
    ref = _build("lora", dropout, device)
    batch = _batch(device, ref.config.vocab_size)
    g_ref, loss_ref, r_ref = _grads(ref, batch, torch_seed=1234)
    del ref
    gc.collect(); torch.cuda.empty_cache()

    out = {"dropout": dropout, "reference": {"loss": loss_ref, "raised": r_ref}}
    traj = {}
    for tag, preserve in (("ckpt_preserve_True", True), ("ckpt_preserve_False", False)):
        m = _build("lora", dropout, device)
        n_wrapped = _wrap_layers(m, reentrant=True, preserve_rng=preserve)
        g, loss, raised = _grads(m, batch, torch_seed=1234)
        row = {"n_wrapped": n_wrapped, "loss": loss, "raised": raised,
               "loss_finite": bool(loss is not None and loss == loss)}
        if g is not None and g_ref is not None:
            row["vs_unwrapped_reference"] = grad_delta(g_ref, g)
        # does the corrupted arm still *train*?  a silent bug that visibly diverges is not silent.
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        losses = []
        for i in range(traj_steps):
            torch.manual_seed(1234 + i)
            o = m(**batch)
            o.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(o.loss))
            del o
        row["loss_trajectory"] = losses
        row["loss_decreased"] = bool(losses[-1] < losses[0])
        traj[tag] = row
        del m, opt
        gc.collect(); torch.cuda.empty_cache()
    out.update(traj)
    return out


# ------------------------------------------------------------------------------------------
# S4 -- model level: DyLoRA under reentrant checkpointing (no check exists there at all)
# ------------------------------------------------------------------------------------------
def case_s4(device):
    out = {}
    ref = _build("dylora", 0.0, device)
    batch = _batch(device, ref.config.vocab_size)
    g_ref, loss_ref, r_ref = _grads(ref, batch, torch_seed=1234, py_seed=99)
    out["reference_no_checkpoint"] = {"loss": loss_ref, "raised": r_ref}
    del ref
    gc.collect(); torch.cuda.empty_cache()

    for tag, reentrant in (("ckpt_non_reentrant", False), ("ckpt_reentrant", True)):
        m = _build("dylora", 0.0, device)
        _wrap_layers(m, reentrant=reentrant, preserve_rng=True)
        g, loss, raised = _grads(m, batch, torch_seed=1234, py_seed=99)
        row = {"raised": raised, "loss": loss}
        if g is not None and g_ref is not None:
            row["vs_unwrapped_reference"] = grad_delta(g_ref, g)
        out[tag] = row
        del m
        gc.collect(); torch.cuda.empty_cache()

    # the fused block on the same construction, as the control
    from flashffn import apply_flash_block, fb_reset_counters, fb_get_counters
    m = _build("dylora", 0.0, device)
    fb_reset_counters()
    apply_flash_block(m, keep="min", verbose=False, announce=False)
    g, loss, raised = _grads(m, batch, torch_seed=1234, py_seed=99)
    row = {"raised": raised, "loss": loss, "counters": fb_get_counters()}
    if g is not None and g_ref is not None:
        row["vs_unwrapped_reference"] = grad_delta(g_ref, g)
    out["fused_block_keep_min"] = row
    del m
    gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="S1,S2,S3,S4")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="results/certificate/probe8_remat_divergence.json")
    a = ap.parse_args()
    CFG["seq"] = a.seq

    dev = torch.device("cuda")
    want = a.cases.split(",")
    res = {"config": dict(CFG), "torch": torch.__version__,
           "pytorch_determinism_check": "metadata only "
                                        "(torch/utils/checkpoint.py:1063 _default_meta_extractor "
                                        "-> shape/dtype/device); disabled entirely when "
                                        "use_reentrant=True (checkpoint.py:466)"}
    if "S1" in want:
        print("== S1: same-shape value divergence ==", flush=True)
        res["S1_same_shape_divergence"] = case_s1(dev)
        print(json.dumps(res["S1_same_shape_divergence"], indent=2), flush=True)
    if "S2" in want:
        print("== S2: RNG-state divergence ==", flush=True)
        res["S2_rng_state_divergence"] = case_s2(dev)
        print(json.dumps(res["S2_rng_state_divergence"], indent=2), flush=True)
    if "S3" in want:
        print("== S3: LoRA dropout under decoder-layer checkpointing ==", flush=True)
        res["S3_lora_dropout"] = case_s3(dev, dropout=a.dropout)
        print(json.dumps({k: v for k, v in res["S3_lora_dropout"].items()
                          if k != "loss_trajectory"}, indent=2, default=str)[:2000], flush=True)
    if "S4" in want:
        print("== S4: DyLoRA under reentrant checkpointing ==", flush=True)
        res["S4_dylora"] = case_s4(dev)
        print(json.dumps(res["S4_dylora"], indent=2, default=str)[:2000], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=str)
    print("->", a.out)
