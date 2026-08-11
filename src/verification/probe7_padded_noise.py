"""Probe 7: is `verify_fused_block.py --seq 1024`'s B8 `keep_exactness_*` failure a real error?

B8 (the padded-batch gate) compares the `keep` levels' gradients on a PADDED batch with
`torch.equal` and NO noise-floor fallback -- it was written at, and is documented at, seq 256,
where FlashAttention's backward is bitwise reproducible.  B3 has such a fallback and passes at
seq 1024; B8 does not and fails there.

The decisive test is to run the SAME arm twice on the SAME padded batch and compare it with
itself.  If one arm does not reproduce itself, the check is measuring the FA-backward atomics,
not the keep levels.

    python -W ignore src/verification/probe7_padded_noise.py --seq 1024 --batch 2
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flashffn                                                          # noqa: E402
from verify_fused_block import MODEL, TARGETS, _edge_batch, init_state   # noqa: E402


def build(keep, dtype, device, state):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa", torch_dtype=dtype)
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                                     task_type="CAUSAL_LM", target_modules=TARGETS))
    m.to(device=device, dtype=dtype)
    m.train()
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.data.copy_(state[n].to(dtype))
    flashffn.fb_reset_counters()
    flashffn.apply_flash_block(m, keep=keep)
    return m


def grads_of(keep, device, state, batch):
    m = build(keep, torch.bfloat16, device, state)
    out = m(**batch)
    out.loss.backward()
    g = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.requires_grad}
    lg = out.logits.detach().float().clone()
    del m, out
    torch.cuda.empty_cache()
    return g, lg


def rel(a, b):
    v = [float((a[n] - b[n]).norm()) / (float(a[n].norm()) + 1e-30) for n in a]
    return {"relL2_median": statistics.median(v), "relL2_max": max(v),
            "n_bitwise": sum(1 for n in a if torch.equal(a[n], b[n])), "n": len(v),
            "max_abs": max(float((a[n] - b[n]).abs().max()) for n in a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(41)
    state = init_state(dev)
    pads = [(32, 0), (7, 0)][: a.batch]
    batch, _keep = _edge_batch(dev, a.batch, a.seq, pads)
    runs = {}
    for tag, keep in (("min1", "min"), ("min2", "min"), ("full1", "full"), ("full2", "full")):
        runs[tag] = grads_of(keep, dev, state, batch)
        print(f"  ran {tag}", flush=True)
    res = {"seq": a.seq, "batch": a.batch, "pads": pads}
    for x, y in (("min1", "min2"), ("full1", "full2"), ("full1", "min1"), ("full2", "min2")):
        r = rel(runs[x][0], runs[y][0])
        r["logits_bitwise"] = bool(torch.equal(runs[x][1], runs[y][1]))
        res[f"{x}_vs_{y}"] = r
        print(f"  {x:6s} vs {y:6s}  bitwise {r['n_bitwise']:3d}/{r['n']}  relL2 med="
              f"{r['relL2_median']:.4e} max_abs={r['max_abs']:.3e}  logits_bw={r['logits_bitwise']}",
              flush=True)
    same = res["min1_vs_min2"]["n_bitwise"] < res["min1_vs_min2"]["n"]
    res["same_arm_is_not_bitwise_reproducible"] = bool(same)
    res["VERDICT"] = ("B8's bitwise keep-exactness check at this shape is measuring the "
                      "FlashAttention backward's atomics: the SAME arm does not reproduce itself."
                      if same else
                      "The same arm IS bitwise reproducible here, so a keep-level difference "
                      "would be a real error.")
    print(res["VERDICT"])
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
