"""Probe 3 (attack item 4): correctness beyond the happy path.

Every case compares the fused block against stock HF (same weights, sdpa/FA2, bf16) or asserts
that it raises.  A case that silently produces wrong numbers is a defect.
"""
import gc, os, sys, json, math, traceback
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/CompAct/src")
import flashffn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEV = torch.device("cuda")
RES = {}


def build(attn="sdpa", dtype=torch.bfloat16, dropout=0.0, alpha=16, r=16, targets=None,
         cfg_over=None, tiny=False):
    torch.manual_seed(1234)          # PEFT kaiming-inits lora_A from the global RNG
    torch.cuda.manual_seed_all(1234)
    if tiny:
        c = LlamaConfig(hidden_size=256, intermediate_size=688, num_hidden_layers=2,
                        num_attention_heads=8, num_key_value_heads=4, vocab_size=1024,
                        rms_norm_eps=1e-5, max_position_embeddings=2048)
        for k, v in (cfg_over or {}).items():
            setattr(c, k, v)
        c._attn_implementation = attn
        torch.manual_seed(0)
        m = LlamaForCausalLM(c)
        m = m.to(dtype)
    else:
        m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation=attn,
                                                 torch_dtype=dtype)
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                                     task_type="CAUSAL_LM",
                                     target_modules=targets or TARGETS))
    m.to(device=DEV, dtype=dtype)
    g = torch.Generator(device=DEV).manual_seed(7)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02, generator=g)
    m.train()
    return m


def fwdbwd(m, batch):
    out = m(**batch)
    out.loss.backward()
    gr = {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.requires_grad}
    lg = out.logits.detach().float().clone()
    loss = float(out.loss)
    m.zero_grad(set_to_none=True)
    return loss, lg, gr


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def gradstats(a, b):
    import statistics
    v = [float((a[n] - b[n]).norm()) / (float(b[n].norm()) + 1e-30) for n in b]
    c = [cos(a[n], b[n]) for n in b]
    return {"relL2_med": statistics.median(v), "relL2_max": max(v), "cos_min": min(c)}


def mkbatch(batch, seq, vocab=32000, seed=41, pad_left=0, pad_right=0, pad_id=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, seq), generator=g).to(DEV)
    am = torch.ones_like(ids)
    if pad_left:
        ids[:, :pad_left] = pad_id
        am[:, :pad_left] = 0
    if pad_right:
        ids[:, seq - pad_right:] = pad_id
        am[:, seq - pad_right:] = 0
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": am}


def case_shape(batch, seq, tag=None, alpha=16, targets=None):
    tag = tag or f"b{batch}_s{seq}"
    ref = build(alpha=alpha, targets=targets)
    b = mkbatch(batch, seq)
    l0, lg0, g0 = fwdbwd(ref, b)
    del ref; gc.collect(); torch.cuda.empty_cache()
    m = build(alpha=alpha, targets=targets)
    flashffn.fb_reset_counters()
    flashffn.apply_flash_block(m, keep="min")
    flashffn.apply_flash_final_norm(m)
    l1, lg1, g1 = fwdbwd(m, b)
    c = flashffn.fb_get_counters()
    del m; gc.collect(); torch.cuda.empty_cache()
    r = {"loss_hf": l0, "loss_fb": l1, "dloss": l1 - l0, "logit_cos": cos(lg1, lg0),
         "logit_maxabs": float((lg1 - lg0).abs().max()), **gradstats(g1, g0),
         "counters": c, "finite": bool(math.isfinite(l1))}
    print(f"  {tag:22s} dloss={r['dloss']:+.5f} logit_cos={r['logit_cos']:.7f} "
          f"grad relL2 med={r['relL2_med']:.3e} cos_min={r['cos_min']:.5f} "
          f"recompute={c.get('recompute')}", flush=True)
    RES[tag] = r


def case_raises(tag, fn):
    try:
        fn()
        RES[tag] = {"raised": False}
        print(f"  {tag:22s} DID NOT RAISE  <-- possible defect", flush=True)
    except Exception as e:
        RES[tag] = {"raised": True, "type": type(e).__name__, "msg": str(e)[:180]}
        print(f"  {tag:22s} raised {type(e).__name__}: {str(e)[:110]}", flush=True)


def case_mask_after_warm(pad_left, pad_right, seq=256, batch=2):
    """The one-shot `checked['mask']` flag: does a padded batch that ARRIVES SECOND get checked?"""
    tag = f"mask_2nd_L{pad_left}R{pad_right}"
    ref = build()
    warm = mkbatch(batch, seq, seed=1)
    padded = mkbatch(batch, seq, seed=2, pad_left=pad_left, pad_right=pad_right)
    _ = fwdbwd(ref, warm)
    l0, lg0, g0 = fwdbwd(ref, padded)
    del ref; gc.collect(); torch.cuda.empty_cache()
    m = build()
    flashffn.fb_reset_counters()
    flashffn.apply_flash_block(m, keep="min")
    err = None
    try:
        _ = fwdbwd(m, warm)          # unpadded first -> mask is None -> check passes
        l1, lg1, g1 = fwdbwd(m, padded)   # padded second -> is it checked?
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"
    del m; gc.collect(); torch.cuda.empty_cache()
    if err:
        RES[tag] = {"raised": True, "msg": err}
        print(f"  {tag:22s} raised {err[:110]}", flush=True)
    else:
        valid = slice(pad_left, seq - pad_right if pad_right else seq)
        r = {"raised": False, "dloss": l1 - l0,
             "logit_cos_all": cos(lg1, lg0),
             "logit_cos_valid": cos(lg1[:, valid], lg0[:, valid]),
             "logit_maxabs_valid": float((lg1[:, valid] - lg0[:, valid]).abs().max()),
             **gradstats(g1, g0)}
        print(f"  {tag:22s} NO RAISE  dloss={r['dloss']:+.4f} "
              f"cos(valid)={r['logit_cos_valid']:.6f} grad relL2={r['relL2_med']:.3e} "
              f"cos_min={r['cos_min']:.4f}", flush=True)
        RES[tag] = r


def case_accum(micro=4, batch=2, seq=512):
    """Gradient accumulation: 4 micro-batches, grads summed."""
    bs = [mkbatch(batch, seq, seed=100 + i) for i in range(micro)]

    def run(patch):
        m = build()
        if patch:
            flashffn.fb_reset_counters()
            flashffn.apply_flash_block(m, keep="min")
            flashffn.apply_flash_final_norm(m)
        for b in bs:
            out = m(**b)
            (out.loss / micro).backward()
        gr = {n: p.grad.detach().float().clone()
              for n, p in m.named_parameters() if p.requires_grad}
        c = flashffn.fb_get_counters() if patch else None
        m.zero_grad(set_to_none=True)
        del m, out
        gc.collect(); torch.cuda.empty_cache()
        return gr, c
    g0, _ = run(False)
    g1, c = run(True)
    r = {**gradstats(g1, g0), "counters": c}
    print(f"  {'grad_accum_x4':22s} relL2 med={r['relL2_med']:.3e} cos_min={r['cos_min']:.6f} "
          f"recompute={c.get('recompute')}", flush=True)
    RES["grad_accum_x4"] = r


def case_bias_control():
    """Same tiny-model harness with NO bias: establishes what 'agrees' looks like."""
    case_bias(cases=(({}, "cfg_nobias_control"),))


def case_bias(cases=None):
    """attention_bias / mlp_bias: `_fb_proj` is F.linear(x, W) with NO bias term."""
    for over, tag in (cases or (({"attention_bias": True}, "cfg_attention_bias"),
                                ({"mlp_bias": True}, "cfg_mlp_bias"))):
        torch.manual_seed(0)
        ref = build(tiny=True, cfg_over=over)
        for n, p in ref.named_parameters():
            if n.endswith("bias"):
                p.data.normal_(0, 0.05)
        state = {n: p.detach().clone() for n, p in ref.named_parameters()}
        b = mkbatch(2, 64, vocab=1024, seed=3)
        l0, lg0, g0 = fwdbwd(ref, b)
        del ref; gc.collect(); torch.cuda.empty_cache()
        torch.manual_seed(0)
        m = build(tiny=True, cfg_over=over)
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(state[n])
        try:
            flashffn.fb_reset_counters()
            flashffn.apply_flash_block(m, keep="min")
            l1, lg1, g1 = fwdbwd(m, b)
            r = {"raised": False, "dloss": l1 - l0, "logit_cos": cos(lg1, lg0),
                 "logit_maxabs": float((lg1 - lg0).abs().max()), **gradstats(g1, g0)}
            print(f"  {tag:22s} NO RAISE dloss={r['dloss']:+.4f} cos={r['logit_cos']:.6f} "
                  f"grad relL2={r['relL2_med']:.3e}", flush=True)
        except Exception as e:
            r = {"raised": True, "msg": f"{type(e).__name__}: {str(e)[:160]}"}
            print(f"  {tag:22s} raised {r['msg'][:110]}", flush=True)
        RES[tag] = r
        del m; gc.collect(); torch.cuda.empty_cache()


def case_qk_norm():
    """Qwen3/OLMo2-style q_norm/k_norm inside self_attn: silently ignored?"""
    tag = "cfg_qk_norm_present"
    torch.manual_seed(0)
    m = build(tiny=True)
    import torch.nn as nn
    base = m.base_model.model.model
    for lyr in base.layers:
        lyr.self_attn.q_norm = nn.RMSNorm(32).to(DEV, torch.bfloat16)
        lyr.self_attn.k_norm = nn.RMSNorm(32).to(DEV, torch.bfloat16)
    try:
        flashffn.apply_flash_block(m, keep="min")
        RES[tag] = {"raised": False}
        print(f"  {tag:22s} DID NOT RAISE (extra sub-modules silently ignored)", flush=True)
    except Exception as e:
        RES[tag] = {"raised": True, "msg": f"{type(e).__name__}: {str(e)[:160]}"}
        print(f"  {tag:22s} raised {str(e)[:110]}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()


def case_sliding_window():
    tag = "cfg_sliding_window"
    torch.manual_seed(0)
    m = build(tiny=True, cfg_over={"sliding_window": 16, "layer_types": None})
    try:
        flashffn.apply_flash_block(m, keep="min")
        RES[tag] = {"raised": False}
        print(f"  {tag:22s} DID NOT RAISE (sliding window silently ignored)", flush=True)
    except Exception as e:
        RES[tag] = {"raised": True, "msg": f"{type(e).__name__}: {str(e)[:160]}"}
        print(f"  {tag:22s} raised {str(e)[:110]}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()


def case_eval_fallthrough():
    m = build()
    flashffn.fb_reset_counters()
    flashffn.apply_flash_block(m, keep="min")
    flashffn.apply_flash_final_norm(m)
    b = mkbatch(2, 256, seed=9)
    m.eval()
    with torch.no_grad():
        l_eval = float(m(**b).loss)
    c_eval = flashffn.fb_get_counters()
    m.train()
    del m; gc.collect(); torch.cuda.empty_cache()
    ref = build()
    ref.eval()
    with torch.no_grad():
        l_ref = float(ref(**b).loss)
    del ref; gc.collect(); torch.cuda.empty_cache()
    RES["eval_fallthrough"] = {"loss_fb_eval": l_eval, "loss_hf_eval": l_ref,
                               "dloss": l_eval - l_ref, "counters": c_eval}
    print(f"  {'eval_fallthrough':22s} dloss={l_eval - l_ref:+.6f} fwd_counter="
          f"{c_eval['forward']} (0 == fell through to stock HF)", flush=True)


def _patch(m):
    flashffn.apply_flash_block(m, keep="min")
    return m


if __name__ == "__main__":
    print("== shapes ==", flush=True)
    for bsz, s in ((1, 1024), (3, 512), (2, 1000), (2, 777), (2, 129), (2, 33), (2, 4096)):
        try:
            case_shape(bsz, s)
        except Exception:
            traceback.print_exc()
            RES[f"b{bsz}_s{s}"] = {"error": traceback.format_exc()[-400:]}
    print("== lora_alpha != r (scaling 2.0) ==", flush=True)
    case_shape(2, 512, tag="alpha32_r16", alpha=32)
    print("== LoRA on a SUBSET of projections (q,v only) ==", flush=True)
    case_shape(2, 512, tag="lora_qv_only", targets=["q_proj", "v_proj"])
    print("== padding ==", flush=True)
    case_mask_after_warm(0, 32)
    case_mask_after_warm(32, 0)
    case_raises("padded_first_batch",
                lambda: fwdbwd(_patch(build()), mkbatch(2, 256, pad_left=32)))
    print("== gradient accumulation ==", flush=True)
    case_accum()
    print("== dropout ==", flush=True)
    case_raises("lora_dropout_0.1_patchtime",
                lambda: flashffn.apply_flash_block(build(dropout=0.1)))
    case_raises("lora_dropout_0.1_forward",
                lambda: fwdbwd(_patch(build(dropout=0.1)), mkbatch(2, 128)))
    print("== architecture guards ==", flush=True)
    case_bias_control()
    case_bias()
    case_qk_norm()
    case_sliding_window()
    print("== eval fall-through ==", flush=True)
    case_eval_fallthrough()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe3.json")
    json.dump(RES, open(p, "w"), indent=2, default=str)
    print("->", p)


def _patch(m):
    flashffn.apply_flash_block(m, keep="min")
    return m
