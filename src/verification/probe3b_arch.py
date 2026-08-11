"""Probe 3b: architecture variants the fused block might silently mis-handle."""
import gc, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/CompAct/src")
import flashffn
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

DEV = torch.device("cuda")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RES = {}


def build(cfg_over, seed=0):
    torch.manual_seed(seed)
    c = LlamaConfig(hidden_size=256, intermediate_size=688, num_hidden_layers=2,
                    num_attention_heads=8, num_key_value_heads=4, vocab_size=1024,
                    rms_norm_eps=1e-5, max_position_embeddings=2048)
    for k, v in cfg_over.items():
        setattr(c, k, v)
    c._attn_implementation = "sdpa"
    m = LlamaForCausalLM(c).to(torch.bfloat16)
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
                                     task_type="CAUSAL_LM", target_modules=TARGETS))
    m.to(device=DEV, dtype=torch.bfloat16)
    g = torch.Generator(device=DEV).manual_seed(7)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02, generator=g)
    m.train()
    return m


def run(m, ids):
    out = m(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
    out.loss.backward()
    gr = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.requires_grad}
    lg = out.logits.detach().float().clone()
    m.zero_grad(set_to_none=True)
    return float(out.loss), lg, gr


def case(tag, cfg_over):
    import statistics
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1, 1024, (2, 128), generator=g).to(DEV)
    ref = build(cfg_over)
    state = {n: p.detach().clone() for n, p in ref.named_parameters()}
    l0, lg0, g0 = run(ref, ids)
    del ref; gc.collect(); torch.cuda.empty_cache()
    m = build(cfg_over)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(state[n])
    try:
        flashffn.apply_flash_block(m, keep="min")
        l1, lg1, g1 = run(m, ids)
        rel = [float((g1[n] - g0[n]).norm()) / (float(g0[n].norm()) + 1e-30) for n in g0]
        r = {"raised": False, "dloss": l1 - l0,
             "logit_cos": float(F.cosine_similarity(lg1.flatten(), lg0.flatten(), dim=0)),
             "grad_relL2_med": statistics.median(rel)}
        print(f"  {tag:26s} cos={r['logit_cos']:.6f} dloss={r['dloss']:+.5f} "
              f"grad relL2={r['grad_relL2_med']:.3e}", flush=True)
    except Exception as e:
        r = {"raised": True, "msg": f"{type(e).__name__}: {str(e)[:150]}"}
        print(f"  {tag:26s} raised {r['msg'][:110]}", flush=True)
    RES[tag] = r
    del m; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    case("control_gqa_8_4", {})
    case("mha_kv==heads", {"num_key_value_heads": 8})
    case("mqa_kv==1", {"num_key_value_heads": 1})
    case("head_dim!=h/heads", {"head_dim": 64})
    case("rope_theta_1e6", {"rope_theta": 1e6})
    case("rope_scaling_llama3", {"rope_scaling": {"rope_type": "llama3", "factor": 8.0,
                                                  "low_freq_factor": 1.0,
                                                  "high_freq_factor": 4.0,
                                                  "original_max_position_embeddings": 64}})
    json.dump(RES, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "vfy_probe3b.json"), "w"), indent=2)
