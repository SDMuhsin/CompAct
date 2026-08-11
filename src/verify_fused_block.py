"""Verification gates for the fused LoRA decoder block (`flashffn.FusedLoRABlockFunction`).

Companion to `llmdocs/trackers/fused_block.md`. Every number quoted there for fidelity comes from
this script; nothing here measures performance (that is `profile_hyclora.py --headline`, and
protocol §A.4 forbids computing a fidelity number and a throughput number in the same process).

GATES
-----
B1  forward parity   -- logit cosine / max-abs-error against stock HuggingFace.
B2  gradient fidelity-- rel-L2 and cosine of EVERY trainable gradient against
                        (i) an fp32 stock-HF reference, which is the ground truth, and
                        (ii) the bf16 stock-HF arm on the SAME attention backend, which isolates
                        the port from the bf16-vs-fp32 and eager-vs-flash differences.
B3  recompute exactness -- the four `keep` levels (`full`/`glu`/`attn`/`min`) must produce
                        BITWISE-identical gradients. This is the load-bearing exactness statement:
                        recomputation in this block is not an approximation, it re-executes the
                        same kernels on the same inputs. `attn` (2026-08-03) stores
                        FlashAttention's output instead of re-running its O(S^2) forward, and is
                        gated exactly like the other three.
B4  gradient liveness-- every trainable tensor in every layer has a non-zero gradient
                        (the `gradient_checkpointing_enable()`-without-`enable_input_require_grads()`
                        trap, CONTEXT.md §14.5, has already shipped once in this repo).
B5  adapter dtype    -- live `p.dtype` totals per arm; fp32 adapters are worth ~2150 MiB and make
                        PEFT allocate three fp32 copies of the same activation (protocol §E.1).
B6  noise floor      -- FlashAttention's backward accumulates dk/dv with atomics and is not
                        bitwise reproducible; the same arm is run twice so every number above can
                        be read against its own noise.
B7  honesty counters -- the fused path must actually have executed.

EDGE GATES (`--edges`, on by default; the three defects `fused_block_verification.md` §4 found)
----------------------------------------------------------------------------------------------
B8  padded batches   -- DEFECT A. The mask used to be validated once per layer and the flag
                        latched, so a padded batch arriving after an unpadded one was silently
                        ignored (measured: left-pad logit cosine 0.519, grad rel-L2 3.82). The
                        gate feeds left-/right-/mixed-padded batches BOTH first and second and
                        requires (a) valid-position logits at the unpadded control's fidelity,
                        (b) gradients at the unpadded control's fidelity once pad positions carry
                        no label, and (c) arriving-second to be BITWISE identical to
                        arriving-first, which is what a latch of any kind would break.
B9  projection bias  -- DEFECT B. `_fb_proj` dropped `nn.Linear.bias` (measured logit cosine 0.485
                        with `attention_bias=True`, 0.384 with `mlp_bias=True`). The gate builds
                        biased models with non-zero biases and requires the no-bias control's
                        fidelity, and requires a TRAINABLE bias to raise.
B10 architecture     -- DEFECT C. q_norm/k_norm, sliding-window attention, a non-SiLU activation,
                        a Gemma-style `(1+w)` RMSNorm, attention dropout and extra layer
                        sub-modules must all raise at patch time, not be silently ignored.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -W ignore src/verify_fused_block.py \
        --out results/hyclora/fused_block_gates.json
    CUDA_VISIBLE_DEVICES=1 python -W ignore src/verify_fused_block.py --edges_only \
        --out results/hyclora/fused_block_edge_gates.json
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flashffn  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build(attn, dtype, device):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation=attn, torch_dtype=dtype)
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                                     task_type="CAUSAL_LM", target_modules=TARGETS))
    m.to(device=device, dtype=dtype)
    m.train()
    return m


def init_state(device):
    """A reproducible non-trivial adapter state. PEFT zero-inits lora_B, which makes every
    lora_A gradient identically zero and its cosine undefined -- so lora_B is perturbed first,
    exactly as `boring_baseline_frontier.md` §4.2 does."""
    m = build("eager", torch.float32, device)
    g = torch.Generator(device=device).manual_seed(7)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02, generator=g)
    state = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    del m
    torch.cuda.empty_cache()
    return state


def load_state(m, state, dtype):
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.data.copy_(state[n].to(dtype))


def run(m, ids):
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = {n: p.grad.detach().float().clone()
             for n, p in m.named_parameters() if p.requires_grad}
    logits = out.logits.detach().float().clone()
    loss = float(out.loss)
    m.zero_grad(set_to_none=True)
    return loss, logits, grads


def compare(grads, ref, hi=False):
    """`hi=True` does the reductions in float64.

    Only the full-fine-tuning / frozen families need it, and they need it badly: their gradient
    tensors reach 32000 x 2048 elements and their comparison runs on the host (the GPU cannot
    hold eight arms' worth of 1.1B fp32 gradients), where an fp32 dot product over 65M nearly
    identical terms loses enough to report a "cosine" above 1. Every adapter family keeps the
    default fp32 path, so their numbers stay directly comparable with the ones already published
    in `fused_block.md` §3.2."""
    cos, rel, dead = [], [], []
    for n, g in grads.items():
        r = ref[n]
        if float(g.abs().sum()) == 0.0:
            dead.append(n)
        if hi:
            gf, rf = g.flatten().double(), r.flatten().double()
            cos.append(float((gf @ rf) / (gf.norm() * rf.norm() + 1e-300)))
            rel.append(float((rf - gf).norm()) / (float(rf.norm()) + 1e-300))
        else:
            cos.append(float(torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(),
                                                                   dim=0)))
            rel.append(float((r - g).norm()) / (float(r.norm()) + 1e-30))
    return {"n": len(cos), "cos_min": min(cos), "cos_median": statistics.median(cos),
            "relL2_median": statistics.median(rel), "relL2_max": max(rel),
            "n_dead": len(dead), "dead": dead[:5], "fp64_math": bool(hi)}


def dtype_receipt(m):
    from collections import defaultdict
    d = defaultdict(lambda: [0, 0])
    for n, p in m.named_parameters():
        key = ("adapter" if "lora_" in n else "base") + "/" + str(p.dtype)
        d[key][0] += 1
        d[key][1] += p.numel() * p.element_size()
    return {k: {"n_tensors": v[0], "MiB": round(v[1] / 2 ** 20, 3)} for k, v in sorted(d.items())}


def layer_liveness(grads, n_layers):
    live = 0
    for i in range(n_layers):
        tag = f"layers.{i}."
        ts = [g for n, g in grads.items() if tag in n]
        if ts and all(float(g.abs().sum()) > 0 for g in ts):
            live += 1
    return live


def trajectory(seq, batch, steps, out_path, family="lora"):
    """§3.7's sanity check: overfit a fixed batch for `steps` AdamW steps and compare the loss
    trajectory of every arm against uncompressed `LoRA + sdpa`. Not a gate -- a cheap end-to-end
    check that the block trains, in the same harness the headline numbers come from."""
    from profile_hyclora import build_model, make_batch
    cfg = {"model": MODEL, "batch": batch, "seq": seq, "lora_r": 16, "q_bit": 4,
           "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
           "iteration_threshold": 5, "n_layers": 22}
    device = torch.device("cuda")
    out = {}
    # HyC-LoRA's port is LoRA-only, so it is in the comparison only for the LoRA family.
    arms = ["baseline_sdpa", "fb_min_fnorm_sdpa", "gc_manual_sdpa"]
    if family == "lora":
        arms.append("hyclora_flash_q2")
    sfx = "" if family == "lora" else f"+{family}"
    for arm in [a + sfx for a in arms]:
        torch.manual_seed(41)
        m = build_model(arm, dict(cfg), device, adapter_dtype="bf16")
        b = make_batch(cfg, device, m.config.vocab_size)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        losses = []
        for _ in range(steps):
            o = m(**b)
            o.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(float(o.loss), 6))
        out[arm] = losses
        print(f"{arm:22s} {losses[0]:.5f} -> {losses[steps // 4]:.5f} -> "
              f"{losses[steps // 2]:.5f} -> {losses[-1]:.5f}", flush=True)
        del m, opt
        torch.cuda.empty_cache()
    ref = out["baseline_sdpa" + sfx]
    res = {"seq": seq, "batch": batch, "steps": steps, "family": family, "losses": out,
           "max_abs_dloss_vs_baseline":
               {k: max(abs(a - c) for a, c in zip(ref, v)) for k, v in out.items()}}
    for k, v in res["max_abs_dloss_vs_baseline"].items():
        print(f"  max |dloss| vs uncompressed {family}: {k:26s} {v:.6f}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {out_path}")


# =============================================================================================
# EDGE GATES B8 / B9 / B10 -- regressions for the three defects in
# `llmdocs/trackers/fused_block_verification.md` §4.  Each one FAILED before the 2026-08-03 fix
# and each one is a silent wrong answer if it comes back.
# =============================================================================================

def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _grad_stats(a, b):
    v = [float((a[n] - b[n]).norm()) / (float(b[n].norm()) + 1e-30) for n in b]
    c = [_cos(a[n], b[n]) for n in b]
    return {"relL2_med": statistics.median(v), "relL2_max": max(v), "cos_min": min(c)}


def _edge_batch(device, batch, seq, pads, vocab=32000, seed=41):
    """Padded batch with the ONLY label convention that is meaningful under any FlashAttention
    varlen implementation: no loss on a pad position, and no loss on a target whose *query*
    position is a pad (the causal shift makes the row before the first real token a pad row)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, seq), generator=g).to(device)
    am = torch.ones_like(ids)
    for i, (pl, pr) in enumerate(pads):
        if pl:
            ids[i, :pl] = 0
            am[i, :pl] = 0
        if pr:
            ids[i, seq - pr:] = 0
            am[i, seq - pr:] = 0
    keep = am.bool()
    lab = ids.clone()
    lab[~keep] = -100
    lab[:, 1:][~keep[:, :-1]] = -100
    return {"input_ids": ids, "labels": lab, "attention_mask": am}, keep


def _edge_run(m, batches):
    """Forward+backward each batch in order; return (logits, grads) of the LAST one only."""
    for bt in batches:
        m.zero_grad(set_to_none=True)
        out = m(**bt)
        out.loss.backward()
    grads = {n: p.grad.detach().float().clone()
             for n, p in m.named_parameters() if p.requires_grad}
    lg = out.logits.detach().float().clone()
    loss = float(out.loss)
    m.zero_grad(set_to_none=True)
    return loss, lg, grads


def gate_B8_padding(device, state, seq=256, batch=2, floor_reps=3):
    """DEFECT A regression: the mask must be honoured on EVERY forward, not just the first."""
    import flashffn as _ff

    def build_patched():
        m = build("sdpa", torch.bfloat16, device)
        load_state(m, state, torch.bfloat16)
        _ff.fb_reset_counters()
        _ff.apply_flash_block(m, keep="min")
        return m

    def build_ref():
        m = build("sdpa", torch.bfloat16, device)
        load_state(m, state, torch.bfloat16)
        return m

    warm, _ = _edge_batch(device, batch, seq, [(0, 0)] * batch, seed=1)
    cases = {"unpadded_control": [(0, 0)] * batch,
             "left_pad": [(32, 0), (7, 0)],
             "right_pad": [(0, 32), (0, 7)],
             "mixed": [(11, 5), (0, 19)]}
    out = {}
    for tag, pads in cases.items():
        bt, keep = _edge_batch(device, batch, seq, pads, seed=2)
        r = build_ref()
        _, lg_ref, g_ref = _edge_run(r, [warm, bt])
        del r
        torch.cuda.empty_cache()
        m2 = build_patched()                                 # padded batch arrives SECOND
        _, lg2, g2 = _edge_run(m2, [warm, bt])
        c2 = _ff.fb_get_counters()
        del m2
        torch.cuda.empty_cache()
        m1 = build_patched()                                 # padded batch arrives FIRST
        _, lg1, g1 = _edge_run(m1, [bt])
        del m1
        torch.cuda.empty_cache()
        row = {
            "pads": pads,
            "logit_cos_valid": _cos(lg2[keep], lg_ref[keep]),
            "logit_maxabs_valid": float((lg2[keep] - lg_ref[keep]).abs().max()),
            "logits_first_vs_second_bitwise": bool(torch.equal(lg1, lg2)),
            "grads_vs_hf_sdpa": _grad_stats(g2, g_ref),
            "counters": c2,
        }
        out[tag] = row
        print(f"  B8 {tag:18s} cos(valid)={row['logit_cos_valid']:.7f} "
              f"maxabs(valid)={row['logit_maxabs_valid']:.4f} "
              f"grad relL2 med={row['grads_vs_hf_sdpa']['relL2_med']:.3e} "
              f"cos_min={row['grads_vs_hf_sdpa']['cos_min']:.6f} "
              f"1st==2nd bitwise={row['logits_first_vs_second_bitwise']}", flush=True)

    # An unpadded batch AFTER a padded one, and a DIFFERENT padded batch after a padded one:
    # both are the same latch failure mode in the other direction (a memoised plan going stale).
    b_pad1, _ = _edge_batch(device, batch, seq, [(32, 0), (7, 0)], seed=2)
    b_pad2, keep2 = _edge_batch(device, batch, seq, [(0, 40), (13, 6)], seed=3)
    b_none, keep_n = _edge_batch(device, batch, seq, [(0, 0)] * batch, seed=4)
    for tag, seqn, tgt_keep in (("stale_plan_unpadded_after_padded", [b_pad1, b_none], keep_n),
                                ("stale_plan_padded_after_padded", [b_pad1, b_pad2], keep2)):
        r = build_ref()
        _, lg_ref, g_ref = _edge_run(r, seqn)
        del r
        torch.cuda.empty_cache()
        m2 = build_patched()
        _, lg2, g2 = _edge_run(m2, seqn)
        del m2
        torch.cuda.empty_cache()
        m1 = build_patched()
        _, lg1, g1 = _edge_run(m1, [seqn[-1]])
        del m1
        torch.cuda.empty_cache()
        out[tag] = {"logit_cos_valid": _cos(lg2[tgt_keep], lg_ref[tgt_keep]),
                    "logit_maxabs_valid": float((lg2[tgt_keep] - lg_ref[tgt_keep]).abs().max()),
                    "logits_first_vs_second_bitwise": bool(torch.equal(lg1, lg2)),
                    "grads_vs_hf_sdpa": _grad_stats(g2, g_ref)}
        print(f"  B8 {tag:18s} cos(valid)={out[tag]['logit_cos_valid']:.7f} "
              f"grad relL2 med={out[tag]['grads_vs_hf_sdpa']['relL2_med']:.3e} "
              f"1st==2nd bitwise={out[tag]['logits_first_vs_second_bitwise']}", flush=True)

    # B3 (recompute exactness) must survive on the varlen path too: all four `keep` levels have
    # to agree on a mixed-length padded batch.
    #
    # LOGITS are required BITWISE unconditionally -- the forward is deterministic at every shape,
    # and that is the part of this gate that actually tests the recompute.
    #
    # GRADIENTS are required bitwise OR at/below the SAME-ARM noise floor, which is exactly the
    # allowance B3 already makes on the dense path (`fused_block.md` section 3.1/3.3) and which
    # this gate was missing.  FlashAttention's backward accumulates dk/dv across query blocks with
    # atomics and stops being bitwise reproducible somewhere above seq 256 -- a property of the
    # kernel, not of any keep level -- so `full` is run TWICE here and its disagreement with
    # itself is the floor every keep level is read against.  Without this the gate reports FAIL at
    # seq 1024 for `glu`, `attn` and `min` alike, i.e. it fails levels that store MORE and
    # recompute nothing, which is not a statement about the recompute at all.
    bt, _ = _edge_batch(device, 3, seq, [(37, 0), (0, 21), (9, 13)], seed=5)
    warm3, _ = _edge_batch(device, 3, seq, [(0, 0)] * 3, seed=6)
    lv = {}
    for keep in ("full", "full_rep", "glu", "attn", "min"):
        m = build("sdpa", torch.bfloat16, device)
        load_state(m, state, torch.bfloat16)
        _ff.fb_reset_counters()
        _ff.apply_flash_block(m, keep=keep.replace("_rep", ""))
        _, lg, gr = _edge_run(m, [warm3, bt])
        lv[keep] = (lg, gr, _ff.fb_get_counters())
        del m
        torch.cuda.empty_cache()
    ref_lg, ref_gr, _ = lv["full"]

    def _maxabs(a, b):
        return max(float((a[n] - b[n]).abs().max()) for n in b)

    # The floor is an estimate of a MAXIMUM (the worst self-disagreement this kernel produces at
    # this shape), and a single draw is the wrong estimator for one.  At seq 1024 the FA backward
    # reproduces itself bitwise only 12/308 of the time and the single-sample floor draws 2^-8 or
    # 2^-7 run to run, so the gate flipped between runs in BOTH the enabled and the disabled
    # configuration -- i.e. its failures carried no signal about the code under test.  Take the max
    # over `floor_reps` independent repeats instead.  These repeats run AFTER glu/attn/min so the
    # build/run order of the levels being judged is byte-for-byte what it was before this change.
    #
    # Each repeat is a whole extra model build, and it happens at the run's high-water mark: `lv`
    # is still holding the logits and gradients of all five levels.  On a shared box that is enough
    # to OOM -- observed, on a card with 1.69 MiB free while tenants held 44 GB, dying on a 22 MiB
    # allocation.  Losing the ENTIRE edge-gate suite because the third floor sample could not be
    # drawn is the wrong trade in both directions: a floor from two samples is still a better
    # estimator than one, and B9/B10 have nothing to do with this loop at all.  So a failed repeat
    # is recorded and the loop stops, rather than propagating.
    floor_samples = [_maxabs(ref_gr, lv["full_rep"][1])]
    floor_bitwise_samples = [bool(all(torch.equal(ref_gr[n], lv["full_rep"][1][n]) for n in ref_gr))]
    floor_incomplete = None
    for _i in range(max(0, floor_reps - 1)):
        try:
            m = build("sdpa", torch.bfloat16, device)
            load_state(m, state, torch.bfloat16)
            _ff.fb_reset_counters()
            _ff.apply_flash_block(m, keep="full")
            _, _lg_r, _gr_r = _edge_run(m, [warm3, bt])
            floor_samples.append(_maxabs(ref_gr, _gr_r))
            floor_bitwise_samples.append(bool(all(torch.equal(ref_gr[n], _gr_r[n])
                                                  for n in ref_gr)))
            del m, _gr_r, _lg_r
        except torch.OutOfMemoryError as exc:
            floor_incomplete = f"repeat {_i + 2}/{floor_reps} OOM: {str(exc)[:120]}"
            print(f"  B8 noise-floor {floor_incomplete}\n"
                  f"  B8 continuing with {len(floor_samples)} sample(s); a short floor is a LOWER "
                  f"estimate of the maximum, so the gate is CONSERVATIVE, not wrong.", flush=True)
            m = _gr_r = _lg_r = None
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    floor_abs = max(floor_samples)
    floor_bitwise = all(floor_bitwise_samples)
    keep_rows = {}
    for keep in ("glu", "attn", "min"):
        lg, gr, c = lv[keep]
        bw = bool(all(torch.equal(ref_gr[n], gr[n]) for n in gr))
        mx = _maxabs(ref_gr, gr)
        keep_rows[f"full_vs_{keep}"] = {
            "logits_bitwise": bool(torch.equal(ref_lg, lg)),
            "grads_bitwise": bw,
            "max_abs_diff": mx,
            "noise_floor_max_abs_diff": floor_abs,
            "noise_floor_is_bitwise": floor_bitwise,
            "at_or_below_noise_floor": bool(mx <= floor_abs * 1.5 + 1e-12),
            "counters": c,
        }
        print(f"  B8 keep-exactness on a padded batch: full vs {keep:4s} "
              f"logits_bitwise={keep_rows[f'full_vs_{keep}']['logits_bitwise']} "
              f"grads_bitwise={bw} max_abs={mx:.3e} floor={floor_abs:.3e}", flush=True)
    out["keep_level_exactness_padded"] = keep_rows
    out["keep_level_noise_floor"] = {
        "max_abs_diff": floor_abs, "bitwise": floor_bitwise,
        "n_samples": len(floor_samples), "n_samples_requested": floor_reps,
        "samples": floor_samples,
        "sample_spread": (max(floor_samples) - min(floor_samples)) if floor_samples else 0.0,
        # non-None => the floor is short and therefore a LOWER estimate of the maximum; the gate
        # stays conservative but the run should not be quoted as a k-sample floor
        "incomplete": floor_incomplete,
    }

    ctrl = out["unpadded_control"]
    lim_cos = ctrl["logit_cos_valid"] - 2e-5
    lim_abs = 2.5 * ctrl["logit_maxabs_valid"]
    lim_rel = 2.5 * ctrl["grads_vs_hf_sdpa"]["relL2_med"]
    checks = {}
    for tag, row in out.items():
        if "logit_cos_valid" not in row:
            continue
        checks[tag] = bool(row["logit_cos_valid"] >= lim_cos
                           and row["logit_maxabs_valid"] <= lim_abs
                           and row["grads_vs_hf_sdpa"]["relL2_med"] <= lim_rel
                           and row["logits_first_vs_second_bitwise"])
    for tag, row in keep_rows.items():
        checks[f"keep_exactness_{tag}"] = bool(
            row["logits_bitwise"]
            and (row["grads_bitwise"] or row["at_or_below_noise_floor"]))
    out["_thresholds"] = {"logit_cos_valid_min": lim_cos, "logit_maxabs_valid_max": lim_abs,
                          "grad_relL2_med_max": lim_rel}
    out["_checks"] = checks
    out["PASS"] = bool(all(checks.values()))
    return out


def _tiny_model(device, cfg_over=None, bias_sigma=0.0, dtype=torch.bfloat16):
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    c = LlamaConfig(hidden_size=256, intermediate_size=688, num_hidden_layers=2,
                    num_attention_heads=8, num_key_value_heads=4, vocab_size=1024,
                    rms_norm_eps=1e-5, max_position_embeddings=2048)
    for k, v in (cfg_over or {}).items():
        setattr(c, k, v)
    c._attn_implementation = "sdpa"
    torch.manual_seed(0)
    m = LlamaForCausalLM(c).to(dtype)
    m.config.use_cache = False
    torch.manual_seed(1234)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                                     task_type="CAUSAL_LM", target_modules=TARGETS))
    m.to(device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(7)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02, generator=g)
    if bias_sigma:
        gg = torch.Generator(device=device).manual_seed(11)
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.data.normal_(0, bias_sigma, generator=gg)
    m.train()
    return m


def gate_B9_bias(device):
    """DEFECT B regression: `nn.Linear.bias` on any of the seven projections must be COMPUTED."""
    import flashffn as _ff
    out = {}
    for tag, over in (("no_bias_control", {}),
                      ("attention_bias", {"attention_bias": True}),
                      ("mlp_bias", {"mlp_bias": True}),
                      ("both_biases", {"attention_bias": True, "mlp_bias": True})):
        sig = 0.05 if over else 0.0
        bt, _ = _edge_batch(device, 2, 64, [(0, 0), (0, 0)], vocab=1024, seed=3)
        ref = _tiny_model(device, over, sig)
        state = {n: p.detach().clone() for n, p in ref.named_parameters()}
        _, lg0, g0 = _edge_run(ref, [bt])
        del ref
        torch.cuda.empty_cache()
        m = _tiny_model(device, over, sig)
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(state[n])
        _ff.fb_reset_counters()
        _ff.apply_flash_block(m, keep="min")
        _, lg1, g1 = _edge_run(m, [bt])
        n_bias = sum(1 for n, _ in m.named_parameters() if n.endswith("bias"))
        del m
        torch.cuda.empty_cache()
        out[tag] = {"n_bias_params": n_bias, "logit_cos": _cos(lg1, lg0),
                    "logit_maxabs": float((lg1 - lg0).abs().max()), **_grad_stats(g1, g0)}
        print(f"  B9 {tag:18s} n_bias={n_bias:3d} cos={out[tag]['logit_cos']:.7f} "
              f"grad relL2 med={out[tag]['relL2_med']:.3e}", flush=True)

    # a TRAINABLE bias must raise -- the fused Function returns no gradient for it
    def _trainable_bias():
        m = _tiny_model(device, {"attention_bias": True}, 0.05)
        n_unfrozen = 0
        for lyr in m.base_model.model.model.layers:
            q = lyr.self_attn.q_proj
            base = q.get_base_layer() if hasattr(q, "get_base_layer") else q
            base.bias.requires_grad_(True)
            n_unfrozen += 1
        assert n_unfrozen > 0
        _ff.apply_flash_block(m, keep="min")
    out["trainable_bias_raises"] = _expect_raise(_trainable_bias)
    print(f"  B9 {'trainable_bias':18s} raises={out['trainable_bias_raises']['raised']} "
          f"({out['trainable_bias_raises'].get('type')})", flush=True)

    ctrl = out["no_bias_control"]
    checks = {}
    for tag in ("attention_bias", "mlp_bias", "both_biases"):
        checks[tag] = bool(out[tag]["n_bias_params"] > 0
                           and out[tag]["logit_cos"] >= ctrl["logit_cos"] - 2e-5
                           and out[tag]["relL2_med"] <= 2.5 * ctrl["relL2_med"])
    checks["trainable_bias_raises"] = out["trainable_bias_raises"]["raised"]
    out["_checks"] = checks
    out["PASS"] = bool(all(checks.values()))
    return out


def _expect_raise(fn):
    try:
        fn()
        return {"raised": False}
    except Exception as e:
        return {"raised": True, "type": type(e).__name__, "msg": str(e)[:200]}


def gate_B10_arch(device):
    """DEFECT C regression: architectural features the fused forward does not implement must
    raise at PATCH time rather than be silently ignored."""
    import torch.nn as nn
    import flashffn as _ff

    def qk_norm():
        m = _tiny_model(device)
        for lyr in m.base_model.model.model.layers:
            lyr.self_attn.q_norm = nn.RMSNorm(32).to(device, torch.bfloat16)
            lyr.self_attn.k_norm = nn.RMSNorm(32).to(device, torch.bfloat16)
        _ff.apply_flash_block(m, keep="min")

    def sliding():
        _ff.apply_flash_block(
            _tiny_model(device, {"sliding_window": 16, "layer_types": None}), keep="min")

    def layer_types_sliding():
        _ff.apply_flash_block(
            _tiny_model(device, {"sliding_window": 16,
                                 "layer_types": ["sliding_attention", "full_attention"]}),
            keep="min")

    def attn_dropout():
        _ff.apply_flash_block(_tiny_model(device, {"attention_dropout": 0.1}), keep="min")

    def gelu_act():
        _ff.apply_flash_block(_tiny_model(device, {"hidden_act": "gelu"}), keep="min")

    def gemma_norm():
        m = _tiny_model(device)

        class _GemmaLike(nn.Module):
            def __init__(self, w, eps):
                super().__init__()
                self.weight = nn.Parameter(w.detach().clone(), requires_grad=False)
                self.eps = eps

            def forward(self, x):
                f = x.float()
                y = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)
                return (y * (1.0 + self.weight.float())).to(x.dtype)
        for lyr in m.base_model.model.model.layers:
            lyr.input_layernorm = _GemmaLike(lyr.input_layernorm.weight,
                                             1e-5).to(device, torch.bfloat16)
        _ff.apply_flash_block(m, keep="min")

    def layernorm_instead_of_rmsnorm():
        m = _tiny_model(device)
        for lyr in m.base_model.model.model.layers:
            ln = nn.LayerNorm(256, eps=1e-5, bias=False).to(device, torch.bfloat16)
            ln.weight.data.copy_(lyr.input_layernorm.weight.data)
            ln.weight.requires_grad_(False)
            lyr.input_layernorm = ln
        _ff.apply_flash_block(m, keep="min")

    def wrong_epsilon():
        m = _tiny_model(device)
        m.base_model.model.config.rms_norm_eps = 0.5      # far from the module's own 1e-5
        _ff.apply_flash_block(m, keep="min")

    def extra_submodule():
        m = _tiny_model(device)
        for lyr in m.base_model.model.model.layers:
            lyr.pre_feedforward_layernorm = nn.RMSNorm(256).to(device, torch.bfloat16)
        _ff.apply_flash_block(m, keep="min")

    def lora_dropout_patchtime():
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        c = LlamaConfig(hidden_size=256, intermediate_size=688, num_hidden_layers=2,
                        num_attention_heads=8, num_key_value_heads=4, vocab_size=1024)
        c._attn_implementation = "sdpa"
        m = LlamaForCausalLM(c).to(torch.bfloat16)
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, bias="none",
                                         task_type="CAUSAL_LM", target_modules=TARGETS))
        m.to(device=device, dtype=torch.bfloat16)
        _ff.apply_flash_block(m, keep="min")

    def bad_mask_shape():
        m = _tiny_model(device)
        _ff.apply_flash_block(m, keep="min")
        ids = torch.randint(1, 1024, (2, 32), device=device)
        # a sliding-window-shaped 4-D mask: causal AND within a window of 8
        base = torch.tril(torch.ones(32, 32, dtype=torch.bool, device=device))
        band = torch.triu(base, -7)
        am = torch.zeros(2, 1, 32, 32, dtype=torch.bfloat16, device=device)
        am.masked_fill_(~band, torch.finfo(torch.bfloat16).min)
        lyr = m.base_model.model.model.layers[0]
        cos_ = torch.ones(32, 32, device=device, dtype=torch.bfloat16)
        sin_ = torch.zeros(32, 32, device=device, dtype=torch.bfloat16)
        h = torch.randn(2, 32, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
        lyr.forward(h, attention_mask=am, position_embeddings=(cos_, sin_))
        del ids

    out = {}
    for tag, fn in (("qk_norm", qk_norm), ("sliding_window", sliding),
                    ("layer_types_sliding", layer_types_sliding),
                    ("attention_dropout", attn_dropout), ("non_silu_activation", gelu_act),
                    ("gemma_style_rmsnorm", gemma_norm),
                    ("layernorm_not_rmsnorm", layernorm_instead_of_rmsnorm),
                    ("wrong_rms_norm_eps", wrong_epsilon),
                    ("extra_layer_submodule", extra_submodule),
                    ("lora_dropout_at_patch_time", lora_dropout_patchtime),
                    ("non_causal_4d_mask", bad_mask_shape)):
        out[tag] = _expect_raise(fn)
        torch.cuda.empty_cache()
        print(f"  B10 {tag:26s} raises={out[tag]['raised']:d} "
              f"{out[tag].get('type', '')}: {out[tag].get('msg', '')[:70]}", flush=True)
    out["_checks"] = {k: v["raised"] for k, v in out.items() if isinstance(v, dict)
                      and "raised" in v}
    out["PASS"] = bool(all(out["_checks"].values()))
    return out


# =================================================================================================
# MULTI-ADAPTER-FAMILY GATES (2026-08-03)
#
# The fused block used to cover plain LoRA only; `_fb_factors` now reduces DoRA, AdaLoRA, DyLoRA,
# VeRA, full fine-tuning and the frozen/raw-weight case to the same factored five-slot form. Every
# gate above is therefore re-run PER FAMILY, plus three that only exist because a family is not a
# pure function of its inputs:
#
# B11 DyLoRA rank sampling -- DyLoRA draws its rank with `random.randint` on EVERY forward, and
#                             CONTEXT §14.6 records that gradient checkpointing's RNG preservation
#                             does not cover the `random` module, so a recomputing scheme silently
#                             gets a different rank in the second pass. The gate counts the draws
#                             (must be exactly one per projection per forward, and ZERO added by
#                             the keep='min' recompute) and instruments the projection call to
#                             record the rank actually multiplied in each pass.
# B12 AdaLoRA rank mask    -- `lora_E` is re-masked by the RankAllocator on a schedule. The gate
#                             mutates `lora_E` BETWEEN the forward and the backward and requires
#                             the gradients to be unchanged, i.e. the recompute reads the same
#                             mask the output pass did.
# B13 unsupported families -- FourierFT and the Spectral (truncated-DCT) adapter must RAISE, at
#                             patch time, with no silent fallback to gradient checkpointing.
# =================================================================================================

FAMILIES = ("lora", "dora", "adalora", "dylora", "vera", "full", "frozen")
# Tensors PEFT zero-initialises; left at zero they make their partner factor's gradient
# identically zero and its cosine undefined (the same reason `init_state` perturbs `lora_B`).
_PERTURB = ("lora_B", "lora_E", "vera_lambda_b")
# full FT / frozen carry ~1.1B trainable elements; their gradients are held on the host so that
# eight arms' worth can coexist.
_HEAVY = ("full", "frozen")


def build_fam(family, attn, dtype, device, r=16):
    """Stock HuggingFace + the family's own uncompressed reference implementation."""
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from profile_hyclora import apply_family
    m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation=attn, torch_dtype=dtype)
    m.config.use_cache = False
    m = apply_family(m, family, r)
    m.to(device=device, dtype=dtype)
    m.train()
    return m


def init_state_fam(family, device, r=16):
    """A reproducible non-trivial adapter state for one family, in fp32.

    Returns None for full/frozen: those have no zero-initialised factor to perturb and their
    weights come from the checkpoint identically in every arm, so nothing has to be synced (and
    a 4.4 GB state clone is avoided)."""
    if family in _HEAVY:
        return None
    m = build_fam(family, "eager", torch.float32, device, r)
    g = torch.Generator(device=device).manual_seed(7)
    for n, p in m.named_parameters():
        if any(t in n for t in _PERTURB):
            p.data.normal_(0, 0.02, generator=g)
    state = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    del m
    torch.cuda.empty_cache()
    return state


def run_fam(m, ids, cpu=False):
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            g = p.grad.detach().float()
            grads[n] = g.cpu() if cpu else g.clone()
    logits = out.logits.detach().float()
    logits = logits.cpu() if cpu else logits.clone()
    loss = float(out.loss)
    m.zero_grad(set_to_none=True)
    del out
    return loss, logits, grads


def dtype_receipt_fam(m):
    from collections import defaultdict
    from profile_hyclora import is_adapter_param
    d = defaultdict(lambda: [0, 0])
    for n, p in m.named_parameters():
        key = ("adapter" if is_adapter_param(n) else "base") + "/" + str(p.dtype)
        d[key][0] += 1
        d[key][1] += p.numel() * p.element_size()
    return {k: {"n_tensors": v[0], "MiB": round(v[1] / 2 ** 20, 3)} for k, v in sorted(d.items())}


def _cos_hi(a, b):
    """float64 cosine. See `compare(hi=...)`: at 16M logit elements an fp32 host-side dot product
    is not accurate enough to be quoted (it reports values above 1)."""
    af, bf = a.flatten().double(), b.flatten().double()
    return float((af @ bf) / (af.norm() * bf.norm() + 1e-300))


def family_gates(device, family, seq=256, batch=2, r=16):
    """B1/B2/B3/B4/B5/B6/B7 for one adapter family."""
    import random as _random
    cpu = family in _HEAVY
    torch.manual_seed(41)
    gen = torch.Generator(device="cpu").manual_seed(41)
    ids = torch.randint(0, 32000, (batch, seq), generator=gen).to(device)
    state = init_state_fam(family, device, r)
    res = {"family": family, "seq": seq, "batch": batch, "arms": {}}

    def load(m, dtype):
        if state is None:
            return
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.data.copy_(state[n].to(dtype))

    # ---- ground truth: fp32, EAGER. sdpa's backward is nondeterministic at ~4e-3 (MEMORY.md),
    #      which is larger than every effect measured here, so the reference must be eager.
    m = build_fam(family, "eager", torch.float32, device, r)
    load(m, torch.float32)
    ref_loss, ref_logits, ref_grads = run_fam(m, ids, cpu=cpu)
    res["fp32_reference"] = {"loss": ref_loss, "n_trainable": len(ref_grads),
                             "dtype_receipt": dtype_receipt_fam(m)}
    del m
    torch.cuda.empty_cache()

    arms = [("hf_eager_bf16", "eager", None), ("hf_sdpa_bf16", "sdpa", None),
            ("hf_sdpa_bf16_rep", "sdpa", None),
            ("fb_full", "sdpa", "full"), ("fb_glu", "sdpa", "glu"),
            ("fb_attn", "sdpa", "attn"),
            ("fb_min", "sdpa", "min"), ("fb_min_rep", "sdpa", "min"),
            ("fb_min_fnorm", "sdpa", "min+fnorm")]
    grads_by_arm, logits_by_arm = {}, {}
    for tag, attn, keep in arms:
        # DyLoRA samples a rank per forward from the `random` module; seeding it per arm is what
        # makes the arms comparable at all, and is itself part of B11's statement.
        _random.seed(1234)
        m = build_fam(family, attn, torch.bfloat16, device, r)
        load(m, torch.bfloat16)
        counters = None
        if keep:
            flashffn.fb_reset_counters()
            flashffn.apply_flash_block(m, keep=keep.split("+")[0])
            if "fnorm" in keep:
                flashffn.apply_flash_final_norm(m)
        loss, logits, grads = run_fam(m, ids, cpu=cpu)
        if keep:
            counters = flashffn.fb_get_counters()
            if counters["forward"] == 0 or counters["backward"] == 0:
                raise RuntimeError(f"{family}/{tag}: fused block never ran ({counters})")
        n_layers = m.config.num_hidden_layers
        row = {
            "loss": loss, "dloss_vs_fp32": loss - ref_loss,
            "B1_logit_cos_vs_fp32": (_cos_hi if cpu else _cos)(logits, ref_logits),
            "B1_logit_maxabs_vs_fp32": float((logits - ref_logits).abs().max()),
            "B2_grads_vs_fp32": compare(grads, ref_grads, hi=cpu),
            "B4_layers_all_grads_live": layer_liveness(grads, n_layers),
            "B4_expected_layers": 0 if family == "frozen" else n_layers,
            "B4_n_trainable": len(grads),
            "B5_dtype_receipt": dtype_receipt_fam(m),
            "B7_counters": counters,
        }
        grads_by_arm[tag] = grads
        logits_by_arm[tag] = logits
        res["arms"][tag] = row
        del m
        torch.cuda.empty_cache()
        print(f"  {family:8s} {tag:16s} loss={loss:.8f} logit_cos={row['B1_logit_cos_vs_fp32']:.8f}"
              f" grad relL2 med={row['B2_grads_vs_fp32']['relL2_median']:.3e}"
              f" cos_min={row['B2_grads_vs_fp32']['cos_min']:.6f}"
              f" live={row['B4_layers_all_grads_live']}/{row['B4_expected_layers']}"
              f" ntr={len(grads)}", flush=True)

    for tag in ("fb_full", "fb_attn", "fb_min", "fb_min_fnorm"):
        res["arms"][tag]["B2_grads_vs_hf_sdpa_bf16"] = compare(grads_by_arm[tag],
                                                               grads_by_arm["hf_sdpa_bf16"], hi=cpu)
        res["arms"][tag]["B1_logit_cos_vs_hf_sdpa_bf16"] = (_cos_hi if cpu else _cos)(
            logits_by_arm[tag], logits_by_arm["hf_sdpa_bf16"])
    res["B6_noise_floor"] = {
        "hf_sdpa_bf16_self": compare(grads_by_arm["hf_sdpa_bf16_rep"],
                                     grads_by_arm["hf_sdpa_bf16"], hi=cpu),
        "fb_min_self": compare(grads_by_arm["fb_min_rep"], grads_by_arm["fb_min"], hi=cpu),
    }

    floor = res["B6_noise_floor"]["fb_min_self"]
    b3 = {}
    for tag in ("fb_glu", "fb_attn", "fb_min"):
        ga, gb = grads_by_arm["fb_full"], grads_by_arm[tag]
        cmp = compare(gb, ga, hi=cpu)
        row = {
            "n_grads": len(gb),
            "bitwise_identical": all(torch.equal(ga[n], gb[n]) for n in gb),
            "n_bitwise": sum(1 for n in gb if torch.equal(ga[n], gb[n])),
            "max_abs_diff": max(float((ga[n] - gb[n]).abs().max()) for n in gb),
            "logits_bitwise_identical": bool(torch.equal(logits_by_arm["fb_full"],
                                                         logits_by_arm[tag])),
            "relL2_median": cmp["relL2_median"], "relL2_max": cmp["relL2_max"],
            "cos_min": cmp["cos_min"],
            "noise_floor_relL2_median": floor["relL2_median"],
            "noise_floor_relL2_max": floor["relL2_max"],
        }
        row["at_or_below_noise_floor"] = bool(
            row["relL2_median"] <= floor["relL2_median"] * 1.5 + 1e-12
            and row["relL2_max"] <= floor["relL2_max"] * 1.5 + 1e-12)
        row["PASS"] = bool(row["logits_bitwise_identical"]
                           and (row["bitwise_identical"] or row["at_or_below_noise_floor"]))
        b3[f"fb_full_vs_{tag}"] = row
    res["B3_recompute_exactness"] = b3

    res["PASS"] = bool(
        all(v["PASS"] for v in b3.values())
        and all(r["B4_layers_all_grads_live"] == r["B4_expected_layers"]
                for r in res["arms"].values())
        and all(r["B2_grads_vs_fp32"]["n_dead"] == 0 for r in res["arms"].values()))
    print(f"  == family {family}: {'PASS' if res['PASS'] else 'FAIL'} ==", flush=True)
    return res


def gate_B11_dylora_rank(device, seq=128, batch=2, r=8):
    """DyLoRA's rank must be drawn ONCE per projection per forward, and the recompute must
    multiply the SAME rank.  Both halves are measured, not assumed."""
    import random as _random
    seen = {"fwd": [], "bwd": []}
    phase = {"p": "fwd"}
    orig = flashffn._fb_proj

    def spy(x2, w, a, b, s, bias=None, c=None, keep_raw=False):
        seen[phase["p"]].append(int(a.shape[0]) if a is not None else -1)
        return orig(x2, w, a, b, s, bias, c, keep_raw)

    _random.seed(99)
    torch.manual_seed(41)
    m = build_fam("dylora", "sdpa", torch.bfloat16, device, r)
    flashffn.fb_reset_counters()
    flashffn.apply_flash_block(m, keep="min")
    ids = torch.randint(0, 32000, (batch, seq), device=device)
    m(input_ids=ids, labels=ids).loss.backward()          # warm-up, not instrumented
    m.zero_grad(set_to_none=True)
    flashffn.fb_reset_counters()
    flashffn._fb_proj = spy
    try:
        out = m(input_ids=ids, labels=ids)
        c_fwd = flashffn.fb_get_counters()["dylora_rank_draws"]
        phase["p"] = "bwd"
        out.loss.backward()
        c_all = flashffn.fb_get_counters()["dylora_rank_draws"]
    finally:
        flashffn._fb_proj = orig
    n_layers = m.config.num_hidden_layers
    fwd, bwd = seen["fwd"], seen["bwd"]
    # forward:  7 projections per layer, in order q k v o gate up down
    # backward: the keep='min' recompute redoes 6 of them (the down projection is never needed)
    per_f, per_b = 7, 6
    ok_pairs, bad = 0, []
    for j, rb in enumerate(bwd):
        lb, within = divmod(j, per_b)
        layer = n_layers - 1 - lb
        rf = fwd[layer * per_f + within]
        if rf == rb:
            ok_pairs += 1
        else:
            bad.append({"layer": layer, "within": within, "fwd_rank": rf, "bwd_rank": rb})
    res = {
        "n_layers": n_layers,
        "draws_during_forward": c_fwd,
        "draws_after_backward": c_all,
        "draws_added_by_recompute": c_all - c_fwd,
        "expected_draws_per_forward": 7 * n_layers,
        "n_distinct_ranks_sampled": len(set(fwd)),
        "rank_range": [min(fwd), max(fwd)],
        "n_recompute_projections_checked": len(bwd),
        "n_rank_matches": ok_pairs,
        "mismatches": bad[:5],
        "PASS": bool(c_fwd == 7 * n_layers and c_all == c_fwd
                     and len(bwd) == per_b * n_layers and ok_pairs == len(bwd)
                     and len(set(fwd)) > 1),
    }
    del m
    torch.cuda.empty_cache()
    print(f"  B11 dylora: draws fwd={c_fwd} (expect {7 * n_layers}), added by recompute="
          f"{res['draws_added_by_recompute']}, ranks {res['rank_range']} "
          f"({res['n_distinct_ranks_sampled']} distinct), recompute rank matches "
          f"{ok_pairs}/{len(bwd)} -> {'PASS' if res['PASS'] else 'FAIL'}", flush=True)
    return res


def gate_B12_adalora_mask(device, seq=128, batch=2, r=8):
    """AdaLoRA's `lora_E` rank mask is re-written by the RankAllocator between optimizer steps.
    THE question this gate answers: does the `keep='min'` recompute multiply the same mask the
    output pass multiplied?

    It is answered directly rather than statistically: `lora_E` is ZEROED between the forward and
    the backward -- a maximally destructive stand-in for a re-masking -- and the `A` factor that
    each projection actually multiplies is captured in both passes and compared bitwise.  Because
    `lora_A * lora_E` is composed OUTSIDE the Function, what `save_for_backward` holds is the
    product, so the mutation cannot reach the recompute.

    A second, HONEST half: PyTorch's `MulBackward` keeps a reference to the leaf `lora_E`, so
    `grad_lora_A = grad_a_eff * lora_E` is read at backward time and DOES move under this
    mutation.  That is a property of the reference implementation, not of this block, so the same
    mutation is applied to stock HF + PEFT AdaLoRA and the two are required to move the SAME set
    of tensors by the SAME construction.
    """
    ids = torch.randint(0, 32000, (batch, seq), device=device)

    def build(fused):
        torch.manual_seed(41)
        m = build_fam("adalora", "sdpa", torch.bfloat16, device, r)
        g = torch.Generator(device=device).manual_seed(7)
        for n, p in m.named_parameters():
            if any(t in n for t in _PERTURB):
                p.data.normal_(0, 0.02, generator=g)
        if fused:
            flashffn.fb_reset_counters()
            flashffn.apply_flash_block(m, keep="min")
        return m

    def run(fused, mutate):
        m = build(fused)
        out = m(input_ids=ids, labels=ids)
        n_mut = 0
        if mutate:
            with torch.no_grad():               # what RankAllocator does between optimizer steps
                for n, p in m.named_parameters():
                    if "lora_E" in n:
                        p.data.zero_()
                        n_mut += 1
        out.loss.backward()
        gr = {n: p.grad.detach().float().clone()
              for n, p in m.named_parameters() if p.requires_grad}
        del m, out
        torch.cuda.empty_cache()
        return gr, n_mut

    # ---- (1) the decisive test: the A factor the recompute multiplies, bitwise ----
    seen = {"fwd": [], "bwd": []}
    phase = {"p": "fwd"}
    orig = flashffn._fb_proj

    def spy(x2, w, a, b, s, bias=None, c=None, keep_raw=False):
        seen[phase["p"]].append(None if a is None else a.detach().clone())
        return orig(x2, w, a, b, s, bias, c, keep_raw)

    m = build(True)
    m(input_ids=ids, labels=ids).loss.backward()        # warm-up (triton autotune)
    m.zero_grad(set_to_none=True)
    flashffn._fb_proj = spy
    try:
        out = m(input_ids=ids, labels=ids)
        phase["p"] = "bwd"
        n_mut = 0
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_E" in n:
                    p.data.zero_()
                    n_mut += 1
        out.loss.backward()
    finally:
        flashffn._fb_proj = orig
    n_layers = m.config.num_hidden_layers
    del m, out
    torch.cuda.empty_cache()
    fwd, bwd = seen["fwd"], seen["bwd"]
    per_f, per_b = 7, 6
    n_ok, n_cmp = 0, 0
    for j, ab in enumerate(bwd):
        lb, within = divmod(j, per_b)
        af = fwd[(n_layers - 1 - lb) * per_f + within]
        if af is None or ab is None:
            continue
        n_cmp += 1
        n_ok += int(torch.equal(af, ab))

    # ---- (2) the honest half: stock PEFT moves under the same mutation, identically ----
    ref_fused, _ = run(True, False)
    mut_fused, _ = run(True, True)
    ref_stock, _ = run(False, False)
    mut_stock, _ = run(False, True)

    def kind_of(n):
        for t in ("lora_A", "lora_B", "lora_E"):
            if t in n:
                return t
        return n

    def kinds(a, b):
        return sorted({kind_of(n) for n in a
                       if "lora_" in n and not torch.equal(a[n], b[n])})

    kinds_fused = kinds(ref_fused, mut_fused)
    kinds_stock = kinds(ref_stock, mut_stock)
    res = {
        "n_lora_E_tensors_zeroed_between_fwd_and_bwd": n_mut,
        "n_recompute_A_factors_compared": n_cmp,
        "n_recompute_A_factors_bitwise_identical_to_forward": n_ok,
        "grad_kinds_moved_by_the_mutation_fused": kinds_fused,
        "grad_kinds_moved_by_the_mutation_stock_peft": kinds_stock,
        "note": ("grad_lora_A moves in BOTH arms: MulBackward reads the leaf lora_E at backward "
                 "time. That is the reference implementation's behaviour, not the block's; what "
                 "the block controls -- the A factor its recompute multiplies -- is bitwise "
                 "frozen at its forward value."),
        "PASS": bool(n_mut > 0 and n_cmp == per_b * n_layers and n_ok == n_cmp
                     and kinds_fused == kinds_stock),
    }
    print(f"  B12 adalora: zeroed {n_mut} lora_E between fwd and bwd; recompute A factors "
          f"bitwise identical to forward {n_ok}/{n_cmp}; grads moved -- fused {kinds_fused} vs "
          f"stock PEFT {kinds_stock} -> {'PASS' if res['PASS'] else 'FAIL'}", flush=True)
    return res


def gate_B13_unsupported(device):
    """FourierFT and the Spectral adapter must RAISE at patch time -- no silent grad-checkpoint
    fallback, matching FlashFFNv3AdapterFunction's behaviour."""
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    out = {}

    def load():
        m = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa",
                                                 torch_dtype=torch.bfloat16)
        m.config.use_cache = False
        return m

    def fourierft():
        from peft import FourierFTConfig, get_peft_model
        m = get_peft_model(load(), FourierFTConfig(
            n_frequency=200, target_modules=TARGETS, task_type="CAUSAL_LM", init_weights=True))
        m.to(device=device, dtype=torch.bfloat16)
        m.train()
        flashffn.apply_flash_block(m, keep="min")

    def spectral():
        from spectral_adapter import get_spectral_adapter_model
        m = get_spectral_adapter_model(model=load(), target_modules=TARGETS, p=8, q=8)
        m.to(device=device, dtype=torch.bfloat16)
        m.train()
        flashffn.apply_flash_block(m, keep="min")

    for tag, fn in (("fourierft", fourierft), ("spectral", spectral)):
        out[tag] = _expect_raise(fn)
        torch.cuda.empty_cache()
        print(f"  B13 {tag:12s} raises={out[tag]['raised']:d} {out[tag].get('type', '')}: "
              f"{out[tag].get('msg', '')[:90]}", flush=True)
    out["PASS"] = bool(all(v["raised"] for v in out.values() if isinstance(v, dict)))
    return out


def edge_gates(device, state=None, seq=256, batch=2, b8_floor_reps=3):
    if state is None:
        state = init_state(device)
    res = {}
    print("== B8 padded batches (DEFECT A) ==", flush=True)
    res["B8_padding"] = gate_B8_padding(device, state, seq=seq, batch=batch,
                                        floor_reps=b8_floor_reps)
    print("== B9 projection biases (DEFECT B) ==", flush=True)
    res["B9_bias"] = gate_B9_bias(device)
    print("== B10 architecture guards (DEFECT C) ==", flush=True)
    res["B10_arch"] = gate_B10_arch(device)
    res["PASS"] = bool(all(v["PASS"] for v in res.values() if isinstance(v, dict)))
    print(f"EDGE GATES {'PASS' if res['PASS'] else 'FAIL'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="results/hyclora/fused_block_gates.json")
    ap.add_argument("--b8_floor_reps", type=int, default=3,
                    help="B8's self-disagreement floor estimates a MAXIMUM; take the max over "
                         "this many repeats. 1 reproduces the old single-draw behaviour, whose "
                         "seq-1024 failures were not signal (CONTEXT.md 33.7).")
    ap.add_argument("--trajectory", type=int, default=0, metavar="STEPS",
                    help="instead of the gates, run the §3.7 loss-trajectory sanity check for "
                         "STEPS optimizer steps on a fixed batch")
    ap.add_argument("--no_edges", action="store_true",
                    help="skip the B8/B9/B10 edge gates (the three §4 defect regressions)")
    ap.add_argument("--edges_only", action="store_true",
                    help="run ONLY the B8/B9/B10 edge gates")
    ap.add_argument("--families", default=None,
                    help="run the per-adapter-family gates instead: a comma list from "
                         f"{','.join(FAMILIES)}, or 'all'. Adds B11 (DyLoRA rank sampling), "
                         "B12 (AdaLoRA rank mask) and B13 (FourierFT/Spectral must raise).")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--fb_certify", default="off", choices=["off", "fused", "standalone"],
                    help="run every gate with the rematerialisation certificate on. NR-3 requires "
                         "that turning it on changes no gate outcome, and NR-1 requires that "
                         "turning it off reproduces the base bitwise.")
    ap.add_argument("--fb_inplace_glu", default="on", choices=["on", "off"],
                    help="the in-place GLU-backward window (CONTEXT.md section 33.5). 'off' "
                         "restores the two-kernel path exactly, which is what an NR-1-style "
                         "control run needs.")
    ap.add_argument("--fb_offload", default="off", choices=["off", "on"],
                    help="stage `o_h` through pinned host memory at keep='attn' (route_b.md). The "
                         "certificate adjudicates this for free: W1 rides on `x_mid`, which at "
                         "keep='attn' is recomputed from the RELOADED `o_h`, so a bad reload makes "
                         "B14 fire. Pair it with --fb_certify fused.")
    ap.add_argument("--fb_offload_alloc_stream", default="copy", choices=["copy", "compute"],
                    help="'compute' restores the route_b.md section 5b defect, so the suite can be "
                         "shown to catch it rather than assumed to.")
    args = ap.parse_args()

    import flashffn as _ff
    import fb_offload as _fbo
    _fbo.fb_offload_alloc_stream(args.fb_offload_alloc_stream)
    _fbo.fb_offload_enable(args.fb_offload == "on")
    if args.fb_offload == "on":
        print(f"[verify] o_h offload ON (alloc_stream={args.fb_offload_alloc_stream})", flush=True)
    _ff.fb_inplace_glu_enable(args.fb_inplace_glu == "on")
    print(f"[verify] in-place GLU backward {args.fb_inplace_glu.upper()}", flush=True)
    if args.fb_certify == "off":
        _ff.fb_certify_disable()
    else:
        _ff.fb_certify_enable(fused=(args.fb_certify == "fused"))
        print(f"[verify] rematerialisation certificate ON ({args.fb_certify})", flush=True)

    if args.trajectory:
        trajectory(args.seq, args.batch, args.trajectory, args.out,
                   family=(args.families or "lora").split(",")[0])
        return

    device = torch.device("cuda")
    if args.families:
        fams = FAMILIES if args.families == "all" else tuple(
            f for f in args.families.split(",") if f)
        res = {"seq": args.seq, "batch": args.batch, "model": MODEL, "lora_r": args.lora_r,
               "torch": torch.__version__, "families": {}}
        for fam in fams:
            print(f"== FAMILY {fam} ==", flush=True)
            res["families"][fam] = family_gates(device, fam, seq=args.seq, batch=args.batch,
                                                r=args.lora_r)
        print("== B11/B12/B13 the non-pure-forward traps and the refusals ==", flush=True)
        res["B11_dylora_rank_sampling"] = gate_B11_dylora_rank(device, seq=min(args.seq, 128),
                                                               batch=args.batch)
        res["B12_adalora_rank_mask"] = gate_B12_adalora_mask(device, seq=min(args.seq, 128),
                                                             batch=args.batch)
        res["B13_unsupported_raise"] = gate_B13_unsupported(device)
        res["PASS"] = bool(all(v["PASS"] for v in res["families"].values())
                           and res["B11_dylora_rank_sampling"]["PASS"]
                           and res["B12_adalora_rank_mask"]["PASS"]
                           and res["B13_unsupported_raise"]["PASS"])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nFAMILY GATES {'PASS' if res['PASS'] else 'FAIL'}  -> {args.out}")
        return
    if args.edges_only:
        torch.manual_seed(41)
        res = edge_gates(device, seq=args.seq, batch=args.batch,
                         b8_floor_reps=args.b8_floor_reps)
        res.update({"seq": args.seq, "batch": args.batch, "model": MODEL,
                    "torch": torch.__version__})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nEDGE GATES {'PASS' if res['PASS'] else 'FAIL'}  -> {args.out}")
        return

    torch.manual_seed(41)
    gen = torch.Generator(device="cpu").manual_seed(41)
    ids = torch.randint(0, 32000, (args.batch, args.seq), generator=gen).to(device)
    state = init_state(device)
    res = {"seq": args.seq, "batch": args.batch, "model": MODEL,
           "torch": torch.__version__, "arms": {}}

    # ---- ground truth: fp32, eager (deterministic) ----
    m = build("eager", torch.float32, device)
    load_state(m, state, torch.float32)
    ref_loss, ref_logits, ref_grads = run(m, ids)
    res["fp32_reference"] = {"loss": ref_loss, "dtype_receipt": dtype_receipt(m)}
    del m
    torch.cuda.empty_cache()

    # `fb_min_fnorm` is the headline arm of `fused_block.md`, so it is gated too: the final
    # RMSNorm is routed through the same fused kernel, which changes the model's numerics (one
    # bf16 rounding instead of an fp32 upcast) and must not be quoted unverified.
    arms = [("hf_eager_bf16", "eager", None), ("hf_sdpa_bf16", "sdpa", None),
            ("hf_sdpa_bf16_rep", "sdpa", None),
            ("fb_full", "sdpa", "full"), ("fb_glu", "sdpa", "glu"),
            ("fb_attn", "sdpa", "attn"),
            ("fb_min", "sdpa", "min"), ("fb_min_rep", "sdpa", "min"),
            ("fb_min_fnorm", "sdpa", "min+fnorm"),
            ("fb_attn_fnorm", "sdpa", "attn+fnorm")]
    grads_by_arm, logits_by_arm = {}, {}
    for tag, attn, keep in arms:
        m = build(attn, torch.bfloat16, device)
        load_state(m, state, torch.bfloat16)
        counters = None
        if keep:
            flashffn.fb_reset_counters()
            flashffn.apply_flash_block(m, keep=keep.split("+")[0])
            if "fnorm" in keep:
                flashffn.apply_flash_final_norm(m)
        loss, logits, grads = run(m, ids)
        if keep:
            counters = flashffn.fb_get_counters()
            if counters["forward"] == 0 or counters["backward"] == 0:
                raise RuntimeError(f"{tag}: fused block never ran ({counters})")
        n_layers = m.config.num_hidden_layers
        row = {
            "loss": loss, "dloss_vs_fp32": loss - ref_loss,
            "B1_logit_cos_vs_fp32": float(torch.nn.functional.cosine_similarity(
                logits.flatten(), ref_logits.flatten(), dim=0)),
            "B1_logit_maxabs_vs_fp32": float((logits - ref_logits).abs().max()),
            "B2_grads_vs_fp32": compare(grads, ref_grads),
            "B4_layers_all_adapter_grads_live": layer_liveness(grads, n_layers),
            "B4_expected_layers": n_layers,
            "B5_dtype_receipt": dtype_receipt(m),
            "B7_counters": counters,
        }
        grads_by_arm[tag] = grads
        logits_by_arm[tag] = logits
        res["arms"][tag] = row
        del m
        torch.cuda.empty_cache()
        print(f"{tag:16s} loss={loss:.8f} logit_cos={row['B1_logit_cos_vs_fp32']:.8f} "
              f"grad relL2 med={row['B2_grads_vs_fp32']['relL2_median']:.3e} "
              f"cos_min={row['B2_grads_vs_fp32']['cos_min']:.6f} "
              f"live={row['B4_layers_all_adapter_grads_live']}/{n_layers}", flush=True)

    # ---- B2 (ii): backend-matched, dtype-matched control ----
    for tag in ("fb_full", "fb_attn", "fb_min", "fb_min_fnorm", "fb_attn_fnorm"):
        res["arms"][tag]["B2_grads_vs_hf_sdpa_bf16"] = compare(grads_by_arm[tag],
                                                               grads_by_arm["hf_sdpa_bf16"])
        res["arms"][tag]["B1_logit_cos_vs_hf_sdpa_bf16"] = float(
            torch.nn.functional.cosine_similarity(
                logits_by_arm[tag].flatten(), logits_by_arm["hf_sdpa_bf16"].flatten(), dim=0))

    # ---- B2 (iii): the FIDELITY GATE.  Added 2026-08-09 (CONTEXT.md 46.11, 37.9). ----
    #
    # WHY THIS EXISTS.  Until today B1/B2 computed the fused block's error against the fp32
    # reference across all logits and all 308 gradients and then THREW THE VERDICT AWAY:
    # `res["PASS"]` below looked only at `n_dead == 0`, so the suite printed ALL GATES PASS while a
    # fidelity regression of any magnitude would have gone unnoticed.  That is the silent failure
    # the project's absolute rules forbid, and it matters now because the exactness audit (46.11)
    # established that NO other gate constrains the block's arithmetic against anything OUTSIDE
    # FlashFFN -- B3/B4/B6/B7/B8b/B11/B12(a)/B14 are all internal self-consistency, and no gate
    # would catch a deterministic arithmetic change applied uniformly across the four keep levels.
    #
    # WHY THE BAR IS CONTROL-RELATIVE.  The control is `hf_sdpa_bf16`: stock HuggingFace at the same
    # dtype and the same attention backend, scored against the same fp32 reference.  An absolute
    # threshold would freeze one shape's bf16 rounding into a constant and would have to be
    # re-tuned per sequence length; the control-relative form asks the question actually at issue,
    # "is the fused block LESS faithful to fp32 than stock bf16 is?", and is shape-robust.  B8/B9
    # are already control-relative for the same reason, and this reuses their 2.5x multiplier.
    #
    # MEASURED HEADROOM when the gate was written (seq 128, batch 2; control relL2_median
    # 2.2199e-02, relL2_max 5.8646e-02, cos_min 0.999020):
    #     every fb_* arm   relL2_median 2.0328e-02 (0.916x -- BETTER than the control)
    #                      relL2_max    6.1479e-02 (1.048x)
    #                      cos_min      0.998973   (deficit 4.7e-05)
    #     the fnorm arms   relL2_max    6.1887e-02 (1.055x), cos_min 0.998937 (deficit 8.3e-05)
    # so the 2.5x bars carry ~2.4x headroom.  COS_ABS_SLACK is 1e-4 rather than B8/B9's 2e-5
    # because this gate scores a whole model against an fp32 reference rather than one block
    # against a same-precision control, and because the `fnorm` arms deliberately trade an fp32
    # upcast for one bf16 rounding (see the `arms` comment above) -- 2e-5 would fail them by
    # construction, which would be a false alarm, not a finding.
    #
    # These thresholds are calibrated FROM MEASUREMENT, not chosen a priori.  If you change the
    # block's arithmetic and this gate fires, re-read 46.11 before you widen it: widening this bar
    # to make a change pass is precisely the failure it was added to prevent.
    FIDELITY_RATIO_MAX = 2.5
    COS_ABS_SLACK = 1e-4
    ctrl_b2 = res["arms"]["hf_sdpa_bf16"]["B2_grads_vs_fp32"]
    b2fid = {}
    for tag in [t for t in res["arms"] if t.startswith("fb_")]:
        arm_b2 = res["arms"][tag]["B2_grads_vs_fp32"]
        # The control is bf16-vs-fp32, so these are ~1e-2 and never zero; guard anyway rather than
        # let a degenerate control turn the gate into a silent pass.
        den_med = ctrl_b2["relL2_median"] or float("nan")
        den_max = ctrl_b2["relL2_max"] or float("nan")
        row = {
            "control": "hf_sdpa_bf16",
            "relL2_median": arm_b2["relL2_median"],
            "relL2_median_ratio": arm_b2["relL2_median"] / den_med,
            "relL2_max": arm_b2["relL2_max"],
            "relL2_max_ratio": arm_b2["relL2_max"] / den_max,
            "cos_min": arm_b2["cos_min"],
            "cos_min_deficit_vs_control": ctrl_b2["cos_min"] - arm_b2["cos_min"],
            "ratio_max_allowed": FIDELITY_RATIO_MAX,
            "cos_abs_slack": COS_ABS_SLACK,
        }
        row["PASS"] = bool(row["relL2_median_ratio"] <= FIDELITY_RATIO_MAX
                           and row["relL2_max_ratio"] <= FIDELITY_RATIO_MAX
                           and row["cos_min_deficit_vs_control"] <= COS_ABS_SLACK)
        b2fid[tag] = row
    res["B2b_fidelity_vs_control"] = b2fid
    for tag, row in b2fid.items():
        print(f"B2b {tag:16s} relL2 med x{row['relL2_median_ratio']:.3f} "
              f"max x{row['relL2_max_ratio']:.3f} "
              f"cos_deficit={row['cos_min_deficit_vs_control']:.2e} "
              f"{'PASS' if row['PASS'] else 'FAIL'}", flush=True)

    # ---- B6 noise floor: the same arm, twice ----
    res["B6_noise_floor"] = {
        "hf_sdpa_bf16_self": compare(grads_by_arm["hf_sdpa_bf16_rep"],
                                     grads_by_arm["hf_sdpa_bf16"]),
        "fb_min_self": compare(grads_by_arm["fb_min_rep"], grads_by_arm["fb_min"]),
    }

    # ---- B3 recompute exactness: keep levels must agree BITWISE, or -- where FlashAttention's
    # backward is not reproducible (its dk/dv accumulate across query blocks with atomics, which
    # bites from roughly seq 512 upward) -- within the SAME-ARM noise floor measured in B6. The
    # forward is deterministic at every shape, so the LOGITS must be bitwise identical
    # unconditionally; that is the part of the gate that actually tests the recompute.
    floor = res["B6_noise_floor"]["fb_min_self"]
    b3 = {}
    for tag in ("fb_glu", "fb_attn", "fb_min"):
        ga, gb = grads_by_arm["fb_full"], grads_by_arm[tag]
        cmp = compare(gb, ga)
        row = {
            "bitwise_identical": all(torch.equal(ga[n], gb[n]) for n in gb),
            "max_abs_diff": max(float((ga[n] - gb[n]).abs().max()) for n in gb),
            "logits_bitwise_identical": bool(torch.equal(logits_by_arm["fb_full"],
                                                         logits_by_arm[tag])),
            "relL2_median": cmp["relL2_median"], "relL2_max": cmp["relL2_max"],
            "cos_min": cmp["cos_min"],
            "noise_floor_relL2_median": floor["relL2_median"],
            "noise_floor_relL2_max": floor["relL2_max"],
        }
        row["at_or_below_noise_floor"] = bool(
            row["relL2_median"] <= floor["relL2_median"] * 1.5 + 1e-12
            and row["relL2_max"] <= floor["relL2_max"] * 1.5 + 1e-12)
        row["PASS"] = bool(row["logits_bitwise_identical"]
                           and (row["bitwise_identical"] or row["at_or_below_noise_floor"]))
        b3[f"fb_full_vs_{tag}"] = row
    res["B3_recompute_exactness"] = b3

    res["PASS"] = bool(
        all(v["PASS"] for v in b3.values())
        and all(r["B4_layers_all_adapter_grads_live"] == r["B4_expected_layers"]
                for r in res["arms"].values())
        and all(r["B2_grads_vs_fp32"]["n_dead"] == 0 for r in res["arms"].values())
        # B2b, added 2026-08-09: external-reference fidelity is now ASSERTED, not merely measured.
        # Without this clause the suite reported ALL GATES PASS while the only numbers that
        # constrain the block against anything outside FlashFFN were computed and discarded.
        and all(v["PASS"] for v in res["B2b_fidelity_vs_control"].values())
    )

    if not args.no_edges:
        res["edge_gates"] = edge_gates(device, state=state, seq=args.seq, batch=args.batch,
                                       b8_floor_reps=args.b8_floor_reps)
        res["PASS"] = bool(res["PASS"] and res["edge_gates"]["PASS"])

    # B14 -- the certificate's OWN honesty counter (protocol §B line 20, generalised).
    # NR-3's evidence is that this JSON is unchanged when the certificate is on; that evidence is
    # worthless unless the certificate actually RAN, and an unengaged component produces exactly
    # the same unchanged JSON. So the receipt is recorded, and a run that asked for the
    # certificate and made no comparisons FAILS rather than reporting a clean sweep.
    res["B14_certificate"] = dict(_ff.fb_certify_report(), requested=args.fb_certify)
    if args.fb_certify != "off":
        engaged = (res["B14_certificate"]["witnesses_compared"] > 0
                   and res["B14_certificate"]["on"])
        res["B14_certificate"]["engaged"] = bool(engaged)
        res["B14_certificate"]["PASS"] = bool(
            engaged and res["B14_certificate"]["witnesses_mismatched"] == 0)
        res["PASS"] = bool(res["PASS"] and res["B14_certificate"]["PASS"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res["B3_recompute_exactness"], indent=2))
    print(json.dumps(res["B6_noise_floor"], indent=2))
    print(f"\nALL FUSED-BLOCK GATES {'PASS' if res['PASS'] else 'FAIL'}  -> {args.out}")


if __name__ == "__main__":
    main()
