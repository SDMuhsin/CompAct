"""Diagnostic: why do HyC-LoRA gradient cosines collapse in the synthetic layer gate?

Compares per-tensor LoRA gradient fidelity under two input regimes:
  (a) synthetic  -- randn*0.1 hidden states, random-logit (flat) attention
  (b) real       -- hidden states captured from a real TinyLlama forward pass on WikiText-2

Hypothesis: HyC-LoRA's compression is built around properties of REAL LLM activations (heavy
per-channel outliers, peaked attention maps). Synthetic gaussian input has neither, so the
outlier/softmax-sparsification paths behave pathologically. If (b) is healthy and (a) is not,
the layer gate's input distribution was the problem, not the port.
"""

import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hyclora.patch import HyCLoRAConfig, apply_hyclora, reset_counters

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYER_IDX = 11
DEV = "cuda:0"
DTYPE = torch.bfloat16


def cos_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def build_model(seed=1234):
    # MUST be deterministic: get_peft_model initialises lora_A from the global RNG, so without
    # reseeding, the reference model and each variant would carry DIFFERENT adapter weights and
    # every gradient cosine would be meaningless.
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager")
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16,
                     lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(m, cfg)
    with torch.no_grad():
        torch.manual_seed(seed + 1)
        for n, p in sorted(m.named_parameters()):
            if "lora_B" in n:
                p.normal_(0, 0.02)
    return m.to(DEV).to(DTYPE)


def lora_params(layer):
    out = []
    for pname, mod in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                       ("v", layer.self_attn.v_proj), ("o", layer.self_attn.o_proj),
                       ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj),
                       ("down", layer.mlp.down_proj)):
        out.append((f"{pname}.lora_A", mod.lora_A.default.weight))
        out.append((f"{pname}.lora_B", mod.lora_B.default.weight))
    return out


def capture_real_hidden(model, bsz=2, seq=128):
    """Run real WikiText-2 text through the model, capture layer LAYER_IDX's input."""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: bsz * seq].view(bsz, seq).to(DEV)

    base = model.base_model.model.model
    grabbed = {}

    def hook(mod, inp, out):
        grabbed["h"] = inp[0].detach().clone()

    hnd = base.layers[LAYER_IDX].register_forward_hook(hook, with_kwargs=False)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids)
    hnd.remove()
    return grabbed["h"], ids


def run_layer(layer, hidden, mask, pos_emb, position_ids):
    out = layer(hidden, attention_mask=mask, position_ids=position_ids, past_key_value=None,
                output_attentions=False, use_cache=False,
                cache_position=torch.arange(hidden.shape[1], device=hidden.device),
                position_embeddings=pos_emb)
    return out[0] if isinstance(out, tuple) else out


def measure(hidden, label, results):
    bsz, seq, _ = hidden.shape
    model = build_model()
    base = model.base_model.model.model
    layer = base.layers[LAYER_IDX]
    position_ids = torch.arange(seq, device=DEV).unsqueeze(0).expand(bsz, -1)
    cos, sin = base.rotary_emb(hidden, position_ids)
    mask = torch.full((seq, seq), torch.finfo(DTYPE).min, device=DEV, dtype=DTYPE)
    mask = torch.triu(mask, diagonal=1)[None, None].expand(bsz, 1, seq, seq).contiguous()

    model.train()
    h = hidden.clone().requires_grad_(True)
    y = run_layer(layer, h, mask, (cos, sin), position_ids)
    gy = torch.randn_like(y)
    for _, p in lora_params(layer):
        p.grad = None
    y.backward(gy)
    ref = {n: p.grad.detach().clone() for n, p in lora_params(layer)}
    del model
    torch.cuda.empty_cache()

    # also report how "peaked" the attention is and how outlier-y the channels are
    for variant in [("baseline", 8, 0.0, 0.0), ("intra", 8, 0.05, 0.005),
                    ("intra", 4, 0.05, 0.005), ("intra_inter", 4, 0.05, 0.005),
                    ("intra_inter", 2, 0.05, 0.005),
                    ("intra_inter", 4, 0.0, 0.005)]:
        lt, qb, sr, lr = variant
        model = build_model()
        base = model.base_model.model.model
        layer = base.layers[LAYER_IDX]
        reset_counters()
        apply_hyclora(model, HyCLoRAConfig(True, lt, qb, sr, lr, 5), verbose=False)
        model.train()
        h2 = hidden.clone().requires_grad_(True)
        y2 = run_layer(layer, h2, mask, (cos, sin), position_ids)
        for _, p in lora_params(layer):
            p.grad = None
        y2.backward(gy)
        g = {n: p.grad.detach().clone() for n, p in lora_params(layer)}
        per = {n: round(cos_sim(g[n], ref[n]), 4) for n in ref}
        key = f"{label}|{lt}@{qb}|smax={sr}"
        results[key] = {"fwd_cos": round(cos_sim(y2, y), 6), "per_tensor": per,
                        "worst": min(per.values()), "worst_tensor": min(per, key=per.get)}
        print(f"  {key:46s} fwd={results[key]['fwd_cos']:.6f} "
              f"worst={results[key]['worst']:+.4f} ({results[key]['worst_tensor']})")
        del model
        torch.cuda.empty_cache()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {}

    print("=== REAL captured activations (WikiText-2, layer %d) ===" % LAYER_IDX)
    m = build_model()
    real_h, _ = capture_real_hidden(m)
    del m
    torch.cuda.empty_cache()
    print(f"  real hidden: shape={tuple(real_h.shape)} absmax={real_h.float().abs().max():.3f} "
          f"std={real_h.float().std():.3f} "
          f"kurtosis_per_channel_max={((real_h.float()-real_h.float().mean())**4).mean(dim=(0,1)).max():.1f}")
    measure(real_h, "real", results)

    print("\n=== SYNTHETIC activations (randn*0.1) ===")
    torch.manual_seed(0)
    synth = torch.randn_like(real_h) * 0.1
    print(f"  synth hidden: absmax={synth.float().abs().max():.3f} std={synth.float().std():.3f}")
    measure(synth, "synth", results)

    os.makedirs("results/hyclora", exist_ok=True)
    with open("results/hyclora/grad_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n-> results/hyclora/grad_diagnostic.json")


if __name__ == "__main__":
    main()
