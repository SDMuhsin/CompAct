"""Probe 9 -- does the rematerialisation certificate actually catch anything?

Companion to `llmdocs/trackers/remat_certificate.md`. Probe 8 established that recompute
divergence is real, silent and damaging. This one establishes that the certificate detects it,
that it does not fire when nothing is wrong, and that the two witness implementations agree.

A detector evaluated only on faults its own author injected is worth little (this is the standing
objection to the whole ABFT genre), so the injections here are chosen to be *mechanically* the
failure modes documented in other people's bug trackers, not shapes invented to be catchable:

  C1  storage corruption of a saved activation -- one bit, at unpack time.  This is the
      offload/ECC/allocator class, and it is the one unsloth's CPU-offloaded checkpointer
      (`Unsloth_Offloaded_Gradient_Checkpointer`) exposes itself to by moving the block input
      over PCIe and back.  Nothing in any framework checks it.
  C2  a recompute that computes a different value -- the gate projection is perturbed on the
      backward pass only.  Mechanically this is pytorch#186572 (Inductor autotuning picking a
      different reduction tree for the AC-recompute graph than for the forward graph: same
      shape, same dtype, same device, different bf16 values) and the lossy-codec class
      (ActNN/GACT/HyC-LoRA dequantising a saved activation).
  C3  ONE bit, in ONE element, of one tensor -- the sensitivity floor.  This is the silent-data-
      corruption class; `arXiv:2605.04213` measures NaN/±INF at 1.01% of SDC outcomes, so a
      detector that only catches NaNs catches almost nothing.

Each injection reports BOTH whether the certificate fired AND the gradient damage it prevented,
because "it fired" is only interesting next to "and here is what would otherwise have shipped".

Controls:
  N0  no injection, several steps -- the false-positive rate must be exactly zero, which is a
      claim the block can make only because its recompute is bitwise (`probe2_seq1024.json`).
  N1  the fused and standalone witnesses must produce IDENTICAL digests on the same tensors, or
      the cheap implementation is not computing the same function as the reference one.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -W ignore src/verification/probe9_certificate.py \
        --out results/certificate/probe9_certificate.json
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

import flashffn                                                    # noqa: E402
from flashffn import RematerialisationError                        # noqa: E402
from profile_hyclora import build_model, make_batch                # noqa: E402

CFG = {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "batch": 2, "seq": 256, "lora_r": 16,
       "q_bit": 2, "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
       "iteration_threshold": 5, "n_layers": 22}


def build(keep="min", family="lora", cert=True, fused=True):
    dev = torch.device("cuda")
    torch.manual_seed(41)
    if cert:
        flashffn.fb_certify_enable(fused=fused)
    else:
        flashffn.fb_certify_disable()
    m = build_model(f"fb_{keep}_sdpa+{family}", CFG, dev, adapter_dtype="bf16")
    g = torch.Generator(device=dev).manual_seed(7)
    for n, p in m.named_parameters():
        if any(t in n for t in ("lora_B", "lora_E", "vera_lambda_b")):
            p.data.normal_(0, 0.02, generator=g)
    b = make_batch(CFG, dev, m.config.vocab_size)
    return m, b


def step(m, b):
    """One forward+backward. Returns (grads, fired, message)."""
    m.zero_grad(set_to_none=True)
    out = m(**b)
    out.loss.backward()
    gr = {n: p.grad.detach().clone() for n, p in m.named_parameters()
          if p.requires_grad and p.grad is not None}
    # read the tallies BEFORE checking: `fb_certify_check` zeroes the counter when it raises, so
    # a report taken afterwards would show 0 mismatches on the very step that failed.
    rep = flashffn.fb_certify_report()
    fired, msg = False, None
    try:
        flashffn.fb_certify_check()
    except RematerialisationError as e:
        fired, msg = True, str(e)[:110]
    m.zero_grad(set_to_none=True)
    del out
    return gr, fired, msg, rep


def damage(ref, got):
    rel = [float((ref[n].float() - got[n].float()).norm() / (ref[n].float().norm() + 1e-30))
           for n in ref if n in got]
    return {"n": len(rel), "rel_l2_median": statistics.median(rel), "rel_l2_max": max(rel),
            "n_bitwise_equal": sum(1 for n in ref if n in got and torch.equal(ref[n], got[n]))}


# ------------------------------------------------------------------------------------------
# C1 -- storage corruption of a saved activation, one bit, at unpack
# ------------------------------------------------------------------------------------------
def case_c1(hidden, nbits=1):
    """`saved_tensors_hooks` corrupts the block input on the way OUT of the saved-tensor store.

    That is exactly the shape of an offload round-trip fault or an uncorrected ECC event: the
    forward saw one value, the backward reads another, and no framework anywhere looks.
    """
    m, b = build()
    ref, _, _, _ = step(m, b)                       # clean reference from the same process

    state = {"hits": 0}

    def pack(t):
        return t

    def unpack(t):
        # target the block input: [B, S, hidden] bf16. One flip, on the first block only.
        if (t.dtype == torch.bfloat16 and t.dim() == 3 and t.shape[-1] == hidden
                and state["hits"] < 1):
            state["hits"] += 1
            c = t.clone()
            v = c.view(torch.int16).reshape(-1)
            for i in range(nbits):
                v[7919 + i] ^= 1                    # 1 ULP, deterministic index
            return c
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        got, fired, msg, rep = step(m, b)
    out = {"bits_flipped": nbits, "tensors_corrupted": state["hits"],
           "certificate_fired": fired, "message": msg,
           "witnesses_mismatched": rep["witnesses_mismatched"],
           "witnesses_compared": rep["witnesses_compared"],
           "gradient_damage_that_would_have_shipped": damage(ref, got)}
    del m, b
    gc.collect(); torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------
# C2 -- the recompute computes a different value (pytorch#186572 / lossy-codec class)
# ------------------------------------------------------------------------------------------
def case_c2(scale=1.00390625):       # 1 + 2^-8 == exactly ONE bf16 ULP, the smallest
                                     # perturbation this dtype can even represent
    """Perturb the gate projection on the BACKWARD pass only.

    Shape, dtype and device are untouched, so `torch.utils.checkpoint`'s `determinism_check`
    would pass this without comment (probe8 S1 measures that directly).

    `scale` must be exactly representable in bf16 and at least one ULP: the first version of this
    probe used 1 + 2^-10, which bf16 (7 explicit mantissa bits, ULP 2^-8) rounds straight back to
    1.0, so the "injection" was a no-op and the case reported a false negative. The receipt below
    counts the elements that actually changed, so that cannot recur silently.
    """
    m, b = build()
    ref, _, _, _ = step(m, b)

    orig = flashffn._fb_proj
    state = {"phase": "fwd", "n": 0, "changed": 0, "numel": 0}

    def patched(x2, w, a, bb, s, bias=None, c=None, keep_raw=False):
        y, xa, raw = orig(x2, w, a, bb, s, bias, c, keep_raw)
        # gate/up live at hidden*4-ish width; perturb only during backward, only the wide ones
        if state["phase"] == "bwd" and y.shape[-1] > x2.shape[-1]:
            state["n"] += 1
            if state["n"] == 1:
                y2 = y * scale
                state["changed"] = int((y2 != y).sum())
                state["numel"] = y.numel()
                y = y2
        return y, xa, raw

    flashffn._fb_proj = patched
    try:
        m.zero_grad(set_to_none=True)
        state["phase"] = "fwd"
        out = m(**b)
        state["phase"] = "bwd"
        out.loss.backward()
        got = {n: p.grad.detach().clone() for n, p in m.named_parameters()
               if p.requires_grad and p.grad is not None}
        rep = flashffn.fb_certify_report()
        fired, msg = False, None
        try:
            flashffn.fb_certify_check()
        except RematerialisationError as e:
            fired, msg = True, str(e)[:110]
    finally:
        flashffn._fb_proj = orig
    res = {"scale": scale, "perturbed_calls": state["n"] and 1,
           "elements_actually_changed": state["changed"], "elements_in_tensor": state["numel"],
           "certificate_fired": fired, "message": msg,
           "witnesses_mismatched": rep["witnesses_mismatched"],
           "gradient_damage_that_would_have_shipped": damage(ref, got)}
    del m, b, out
    gc.collect(); torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------
# N0 -- the false-positive control
# ------------------------------------------------------------------------------------------
def case_n0(steps=8, keep="min", family="lora", fused=True):
    m, b = build(keep=keep, family=family, fused=fused)
    fired = 0
    for _ in range(steps):
        _, f, _, rep = step(m, b)
        fired += int(f)
    del m, b
    gc.collect(); torch.cuda.empty_cache()
    return {"keep": keep, "family": family, "fused": fused, "steps": steps,
            "times_fired": fired, "witnesses_compared": rep["witnesses_compared"],
            "witnesses_mismatched": rep["witnesses_mismatched"],
            "blocks_certified": rep["blocks_certified"]}


# ------------------------------------------------------------------------------------------
# N1 -- fused and standalone witnesses must compute the SAME function
# ------------------------------------------------------------------------------------------
def case_n1():
    dev = torch.device("cuda")
    torch.manual_seed(3)
    rows = []
    for shape in ((2048, 2048), (2048, 5632), (512, 2048), (3, 4097)):
        t = torch.randn(*shape, device=dev, dtype=torch.bfloat16)
        w = torch.ones(shape[-1], device=dev, dtype=torch.bfloat16)
        flashffn._FB_CERT["fused"] = True
        _, _, df = flashffn.fb_rmsnorm_forward(t, w, 1e-5, digest=True)
        flashffn._FB_CERT["fused"] = False
        _, _, ds = flashffn.fb_rmsnorm_forward(t, w, 1e-5, digest=True)
        direct = flashffn.fb_digest(t)
        rows.append({"shape": list(shape), "fused": int(df), "standalone": int(ds),
                     "direct": int(direct),
                     "agree": bool(int(df) == int(ds) == int(direct))})
    flashffn._FB_CERT["fused"] = True
    return rows


# ------------------------------------------------------------------------------------------
# C3 -- single-bit sensitivity of the digest itself, over many trials
# ------------------------------------------------------------------------------------------
def case_c3(trials=256, n=2048 * 5632):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(11)
    t = torch.randn(n, device=dev, dtype=torch.bfloat16, generator=g)
    base = int(flashffn.fb_digest(t))
    idx = torch.randint(0, n, (trials,), generator=torch.Generator().manual_seed(5)).tolist()
    detected = 0
    for i in idx:
        c = t.clone()
        c.view(torch.int16)[i] ^= 1
        detected += int(int(flashffn.fb_digest(c)) != base)
    return {"elements": n, "trials": trials, "single_bit_flips_detected": detected,
            "detection_rate": detected / trials}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="N1,C3,N0,C1,C2")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", default="results/certificate/probe9_certificate.json")
    a = ap.parse_args()
    CFG["seq"] = a.seq
    want = a.cases.split(",")
    res = {"config": dict(CFG)}

    if "N1" in want:
        print("== N1: fused vs standalone vs direct digest ==", flush=True)
        res["N1_witness_agreement"] = case_n1()
        for r in res["N1_witness_agreement"]:
            print(f"  {str(r['shape']):16s} agree={r['agree']}", flush=True)
    if "C3" in want:
        print("== C3: single-bit sensitivity ==", flush=True)
        res["C3_single_bit"] = case_c3()
        print("  ", res["C3_single_bit"], flush=True)
    if "N0" in want:
        print("== N0: false-positive control ==", flush=True)
        res["N0_control"] = [case_n0(keep=k, family=f, fused=fu)
                             for k, f, fu in (("min", "lora", True), ("min", "lora", False),
                                              ("attn", "lora", True), ("full", "lora", True),
                                              ("min", "dora", True), ("min", "dylora", True))]
        for r in res["N0_control"]:
            print(f"  keep={r['keep']:5s} family={r['family']:7s} fused={r['fused']} "
                  f"steps={r['steps']} compared={r['witnesses_compared']:4d} "
                  f"FIRED={r['times_fired']}", flush=True)
    if "C1" in want:
        print("== C1: one-bit storage corruption of a saved activation ==", flush=True)
        res["C1_storage_corruption"] = case_c1(hidden=2048)
        r = res["C1_storage_corruption"]
        print(f"  fired={r['certificate_fired']} mismatched={r['witnesses_mismatched']} "
              f"damage={r['gradient_damage_that_would_have_shipped']}", flush=True)
    if "C2" in want:
        print("== C2: recompute computes a different value ==", flush=True)
        res["C2_recompute_divergence"] = case_c2()
        r = res["C2_recompute_divergence"]
        print(f"  fired={r['certificate_fired']} mismatched={r['witnesses_mismatched']} "
              f"damage={r['gradient_damage_that_would_have_shipped']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=str)
    print("->", a.out)
