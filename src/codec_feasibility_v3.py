"""FlashFFN v3 codec feasibility study.

Measures stored-bytes vs gradient-fidelity trade-offs for compressing the
SwiGLU FFN pair (h_gate, h_up) on REAL TinyLlama activations + REAL upstream
gradients (wikitext-2 causal LM loss), compared against the current v2
approach (exact recompute of h_gate/h_up + top-30% |h_mid| for grad_w_down).

This is a pure-PyTorch MEASUREMENT study (no Triton, no training).

Pipeline:
  Step 1: capture x, h_gate, h_up, grad_y at layers {0,5,11,21} for 4 batches
          (batch=4, seq=512 -> N=2048 tokens/batch) + activation statistics.
          h_gate/h_up are recomputed from x in bf16 and VERIFIED against the
          module's actual intermediates (hooks on gate_proj/up_proj/act_fn).
  Step 2: fp32 ground-truth grads {grad_w_gate, grad_w_up, grad_w_down,
          grad_x} from the SAME bf16 inputs the codecs see (isolates codec
          error from bf16 error). TF32 disabled.
  Step 3: codecs C0..C7 reconstruct (h_act, silu', h_up); grads recomputed
          from reconstructions; ALL metadata bytes counted (values, fp16
          group scales, 1-bit bitmaps, int16 indices).
  Step 4: cosine sim + relative Frobenius error per grad, per layer, per
          batch; stochastic-rounding bias check (16-draw averaged grad).
  Step 5: results/v3_codec/codec_results.json + summary.md.

Run from repo root:
  source env/bin/activate
  HF_HOME=./data TORCH_HOME=./data CUDA_VISIBLE_DEVICES=<idle GPU> \
      python src/codec_feasibility_v3.py
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GROUP = 64                      # quantization group size along D_int
TOPK_FRACTION = 0.3             # matches v2 k_fraction (per-row, dim=-1)
SILU_NEG_MIN = -0.278           # fixed quantizer range for silu(g), g<0 (spec)
SILUP_NEG_LO, SILUP_NEG_HI = -0.1, 0.5  # fixed range for silu'(g), g<0
FP8_MAX = 448.0                 # float8_e4m3fn max normal
PCTS = [0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0]

GRAD_NAMES = ["grad_w_gate", "grad_w_up", "grad_w_down", "grad_x"]


def silu_prime(g):
    s = torch.sigmoid(g)
    return s * (1.0 + g * (1.0 - s))


# --------------------------------------------------------------------------
# Step 1: capture real activations + upstream gradients
# --------------------------------------------------------------------------

def build_batches(tokenizer, n_batches, batch_size, seq_len):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    # ~600k chars -> ~150k tokens, far more than n_batches*batch_size*seq_len.
    ids = tokenizer(text[:600_000], return_tensors="pt").input_ids[0]
    need = n_batches * batch_size * seq_len
    assert ids.numel() >= need, f"not enough tokens: {ids.numel()} < {need}"
    blocks = ids[:need].view(n_batches, batch_size, seq_len)
    return blocks  # distinct, non-overlapping, deterministic batches


def capture(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()  # no dropout; grads still flow
    model.config.use_cache = False

    batches = build_batches(tokenizer, args.n_batches, args.batch_size, args.seq_len)

    layers = args.layers
    cap = {L: [dict() for _ in range(args.n_batches)] for L in layers}
    verify = {L: {} for L in layers}  # module intermediates, batch 0 only
    state = {"b": 0}
    handles = []

    def mlp_fwd_hook(L):
        def hook(module, inputs, output):
            b = state["b"]
            cap[L][b]["x"] = inputs[0].detach().to("cpu", copy=True)

            def grad_hook(grad, L=L, b=b):
                cap[L][b]["grad_y"] = grad.detach().to("cpu", copy=True)

            output.register_hook(grad_hook)
        return hook

    def save_out_hook(dst, key):
        def hook(module, inputs, output):
            if state["b"] == 0:  # verification batch only
                dst[key] = output.detach().to("cpu", copy=True)
        return hook

    for L in layers:
        mlp = model.model.layers[L].mlp
        handles.append(mlp.register_forward_hook(mlp_fwd_hook(L)))
        handles.append(mlp.gate_proj.register_forward_hook(save_out_hook(verify[L], "gate_out")))
        handles.append(mlp.up_proj.register_forward_hook(save_out_hook(verify[L], "up_out")))
        handles.append(mlp.act_fn.register_forward_hook(save_out_hook(verify[L], "act_out")))

    losses = []
    for b in range(args.n_batches):
        state["b"] = b
        input_ids = batches[b].to(device)
        out = model(input_ids=input_ids, labels=input_ids)
        losses.append(out.loss.item())
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        del out
        torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    # Per-layer bf16 weights (needed to recompute h_gate/h_up and for grads).
    weights = {}
    for L in layers:
        mlp = model.model.layers[L].mlp
        weights[L] = {
            "w_gate": mlp.gate_proj.weight.detach().to("cpu", copy=True),
            "w_up": mlp.up_proj.weight.detach().to("cpu", copy=True),
            "w_down": mlp.down_proj.weight.detach().to("cpu", copy=True),
        }
    del model
    torch.cuda.empty_cache()

    # Recompute h_gate/h_up from x in bf16 and VERIFY vs module intermediates.
    verification = {}
    for L in layers:
        wg = weights[L]["w_gate"].to(device)
        wu = weights[L]["w_up"].to(device)
        for b in range(args.n_batches):
            x = cap[L][b]["x"].to(device)
            h_gate = x @ wg.T  # bf16, same math as gate_proj's F.linear
            h_up = x @ wu.T
            cap[L][b]["h_gate"] = h_gate.to("cpu")
            cap[L][b]["h_up"] = h_up.to("cpu")
            if b == 0:
                g_mod = verify[L]["gate_out"].to(device)
                u_mod = verify[L]["up_out"].to(device)
                a_mod = verify[L]["act_out"].to(device)
                checks = {
                    "h_gate_vs_gate_proj": (h_gate, g_mod),
                    "h_up_vs_up_proj": (h_up, u_mod),
                    "silu(h_gate)_vs_act_fn": (F.silu(h_gate), a_mod),
                }
                vrec = {}
                for name, (a, bt) in checks.items():
                    af, bf = a.float(), bt.float()
                    max_abs = (af - bf).abs().max().item()
                    ok = torch.allclose(af, bf, rtol=2e-2, atol=2e-2)
                    assert ok, f"layer {L}: {name} mismatch, max_abs_diff={max_abs}"
                    vrec[name] = {
                        "max_abs_diff": max_abs,
                        "frac_bitwise_equal": (a == bt).float().mean().item(),
                        "allclose_bf16_tol": ok,
                    }
                verification[L] = vrec
        del wg, wu
        torch.cuda.empty_cache()

    # Activation statistics, pooled over all batches per layer.
    stats = {}
    for L in layers:
        g = torch.cat([cap[L][b]["h_gate"].view(-1, cap[L][b]["h_gate"].shape[-1])
                       for b in range(args.n_batches)]).to(device).float()
        u = torch.cat([cap[L][b]["h_up"].view(-1, cap[L][b]["h_up"].shape[-1])
                       for b in range(args.n_batches)]).to(device).float()
        mid = F.silu(g) * u

        def pcts(t):
            v = t.flatten().sort().values
            n = v.numel()
            return {f"p{p:g}": v[min(n - 1, int(round(p / 100 * (n - 1))))].item()
                    for p in PCTS}

        stats[L] = {
            "frac_h_gate_lt_0": (g < 0).float().mean().item(),
            "frac_h_gate_lt_-1": (g < -1).float().mean().item(),
            "frac_h_gate_lt_-2": (g < -2).float().mean().item(),
            "frac_h_gate_lt_-3": (g < -3).float().mean().item(),
            "h_gate_pcts": pcts(g),
            "h_up_pcts": pcts(u),
            "h_mid_pcts": pcts(mid),
        }
        del g, u, mid
        torch.cuda.empty_cache()

    meta = {
        "model": MODEL_NAME,
        "dataset": "wikitext-2-raw-v1 (train split, cached)",
        "losses": losses,
        "layers": layers,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_batch": args.batch_size * args.seq_len,
        "shapes": {k: list(cap[layers[0]][0][k].shape)
                   for k in ("x", "h_gate", "h_up", "grad_y")},
    }
    return cap, weights, stats, verification, meta


# --------------------------------------------------------------------------
# Quantizer building blocks (all byte counts include metadata)
# --------------------------------------------------------------------------

def group_absmax_quant(t32, bits, stochastic=False, gen=None, mask=None,
                       count_scales=True):
    """Symmetric absmax per-group quantization (group=GROUP along last dim).

    Scales are stored fp16 (2 bytes each); int range is [-qmax, qmax] with
    qmax = 2^(bits-1)-1. If `mask` is given, only masked entries are stored
    (scale computed over masked entries; unmasked reconstruct to 0).
    Returns (recon_fp32, bytes, n_stored_entries).
    """
    N, D = t32.shape
    tg = t32.view(N, D // GROUP, GROUP)
    if mask is not None:
        mg = mask.view(N, D // GROUP, GROUP)
        absmax = (tg.abs() * mg).amax(dim=-1, keepdim=True)
    else:
        absmax = tg.abs().amax(dim=-1, keepdim=True)
    qmax = float(2 ** (bits - 1) - 1)
    scale = (absmax / qmax).half().float()  # honest: quantize WITH fp16 scale
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    r = tg / safe
    if stochastic:
        q = torch.floor(r + torch.rand(r.shape, device=t32.device, generator=gen))
    else:
        q = torch.round(r)
    q = q.clamp_(-qmax, qmax)
    recon = (q * scale).view(N, D)
    if mask is not None:
        recon = recon * mask
        n_stored = int(mask.sum().item())
    else:
        n_stored = N * D
    n_groups = N * (D // GROUP)
    nbytes = n_stored * bits / 8 + (n_groups * 2 if count_scales else 0)
    return recon, nbytes, n_stored


def fixed_range_quant(v32, lo, hi, bits):
    """Uniform quantizer over a FIXED absolute range [lo, hi]: no per-group
    metadata (range is a compile-time constant; 8 bytes counted once)."""
    nlev = 2 ** bits - 1
    step = (hi - lo) / nlev
    q = torch.round((v32.clamp(lo, hi) - lo) / step)
    return lo + q * step


def topk_rowwise(score, k):
    """Per-row top-k indices along dim=-1 (matches v2 torch.topk semantics)."""
    _, idx = torch.topk(score, k, dim=-1, sorted=False)
    return idx


# --------------------------------------------------------------------------
# Codecs: fn(ctx, gen) -> dict with reconstruction + exact byte counts.
#   ctx: g16,u16 (bf16 codec inputs), g32,u32,act32,sp32,mid32 (fp32 truth
#        from the same bf16 inputs), N, D
# Returns {"act","sp","u"} fp32 reconstructions (or "mode":"v2" with sparse
# mid for grad_w_down only), plus "bytes" dict.
# --------------------------------------------------------------------------

def bitmap_bytes(n_entries):
    return math.ceil(n_entries / 8)


def codec_c0a_v2_topk(ctx, gen):
    """v2 reference: exact recompute of h_gate/h_up (from stored x) for
    grad_w_gate/up/x; grad_w_down from top-30% |h_mid| (bf16 values + int16
    per-row indices). Bytes = x + topk values + indices."""
    N, D = ctx["N"], ctx["D"]
    k = max(1, int(D * TOPK_FRACTION))
    mid16 = F.silu(ctx["g16"]) * ctx["u16"]          # bf16 forward values
    idx = topk_rowwise(mid16.abs().float(), k)
    mid_sparse = torch.zeros_like(ctx["mid32"])
    mid_sparse.scatter_(1, idx, mid16.gather(1, idx).float())
    x_bytes = N * ctx["D_hid"] * 2
    b = {"x_bf16": x_bytes, "topk_values_bf16": N * k * 2,
         "topk_indices_int16": N * k * 2}
    return {"mode": "v2", "mid_sparse": mid_sparse, "bytes": b}


def codec_c0b_exact_recompute(ctx, gen):
    N = ctx["N"]
    return {"mode": "exact", "bytes": {"x_bf16": N * ctx["D_hid"] * 2}}


def codec_c0c_store_full(ctx, gen):
    N, D = ctx["N"], ctx["D"]
    return {"act": ctx["act32"], "sp": ctx["sp32"], "u": ctx["u32"],
            "bytes": {"h_gate_bf16": N * D * 2, "h_up_bf16": N * D * 2}}


def make_codec_int(bits, stochastic):
    def fn(ctx, gen):
        g_rec, gb, _ = group_absmax_quant(ctx["g32"], bits, stochastic, gen)
        u_rec, ub, _ = group_absmax_quant(ctx["u32"], bits, stochastic, gen)
        return {"act": F.silu(g_rec), "sp": silu_prime(g_rec), "u": u_rec,
                "bytes": {"h_gate_q": gb, "h_up_q": ub}}
    return fn


def make_codec_gate_aware(b_neg, b_pos, b_up_neg):
    """C3: 1-bit sign bitmap. Negative branch stores quantized silu(g) and
    silu'(g) at b_neg bits each over FIXED ranges (no per-group scales).
    Positive branch stores g at b_pos bits (per-group absmax over positive
    entries). h_up: positive entries at b_pos bits, negative entries at
    b_up_neg bits (0 = zero them), each branch with its OWN per-group scales.
    """
    def fn(ctx, gen):
        g32, u32 = ctx["g32"], ctx["u32"]
        N, D = ctx["N"], ctx["D"]
        neg = g32 < 0
        pos = ~neg
        n_neg = int(neg.sum().item())
        n_pos = N * D - n_neg
        n_groups = N * (D // GROUP)
        # negative branch: silu(g), silu'(g) at fixed absolute ranges
        act_n = fixed_range_quant(ctx["act32"], SILU_NEG_MIN, 0.0, b_neg)
        sp_n = fixed_range_quant(ctx["sp32"], SILUP_NEG_LO, SILUP_NEG_HI, b_neg)
        # positive branch: quantize g itself, per-group absmax over positives
        g_p, g_p_bytes, _ = group_absmax_quant(g32, b_pos, mask=pos)
        act = torch.where(neg, act_n, F.silu(g_p))
        sp = torch.where(neg, sp_n, silu_prime(g_p))
        # h_up: separate per-branch scales (avoids cross-branch clipping)
        u_p, u_p_bytes, _ = group_absmax_quant(u32, b_pos, mask=pos)
        if b_up_neg > 0:
            u_n, u_n_bytes, _ = group_absmax_quant(u32, b_up_neg, mask=neg)
            u = u_p + u_n
        else:
            u, u_n_bytes = u_p, 0
        b = {
            "sign_bitmap": bitmap_bytes(N * D),
            "neg_act_q": n_neg * b_neg / 8,
            "neg_silup_q": n_neg * b_neg / 8,
            "neg_fixed_range_consts": 8,
            "pos_g_q_incl_scales": g_p_bytes,
            "pos_up_q_incl_scales": u_p_bytes,
            "neg_up_q_incl_scales": u_n_bytes,
        }
        return {"act": act, "sp": sp, "u": u, "bytes": b,
                "extra": {"frac_neg": n_neg / (N * D)}}
    return fn


def make_codec_deadband(tau, bits=4, stochastic=False):
    """C4: drop entries with h_gate < tau entirely (bitmap + nothing);
    INT-`bits` per-group for live entries of BOTH tensors. Dead entries
    reconstruct as act=0, silu'=0, h_up=0 (per spec)."""
    def fn(ctx, gen):
        g32, u32 = ctx["g32"], ctx["u32"]
        N, D = ctx["N"], ctx["D"]
        live = g32 >= tau
        g_rec, gb, n_live = group_absmax_quant(g32, bits, stochastic, gen, mask=live)
        u_rec, ub, _ = group_absmax_quant(u32, bits, stochastic, gen, mask=live)
        act = torch.where(live, F.silu(g_rec), torch.zeros_like(g_rec))
        sp = torch.where(live, silu_prime(g_rec), torch.zeros_like(g_rec))
        b = {"live_bitmap": bitmap_bytes(N * D),
             "h_gate_q_incl_scales": gb, "h_up_q_incl_scales": ub}
        return {"act": act, "sp": sp, "u": u_rec, "bytes": b,
                "extra": {"frac_dead": 1.0 - n_live / (N * D)}}
    return fn


def codec_c5_topk_factors(ctx, gen):
    """C5: per-row top-30% of |h_mid|; store h_gate AND h_up bf16 at those
    positions + int16 indices; others reconstruct as 0/0."""
    N, D = ctx["N"], ctx["D"]
    k = max(1, int(D * TOPK_FRACTION))
    mid16 = F.silu(ctx["g16"]) * ctx["u16"]
    idx = topk_rowwise(mid16.abs().float(), k)
    g_rec = torch.zeros_like(ctx["g32"])
    u_rec = torch.zeros_like(ctx["u32"])
    g_rec.scatter_(1, idx, ctx["g16"].gather(1, idx).float())
    u_rec.scatter_(1, idx, ctx["u16"].gather(1, idx).float())
    # u=0 off-support kills the gate path; silu(0)=0 kills the up path.
    return {"act": F.silu(g_rec), "sp": silu_prime(g_rec), "u": u_rec,
            "bytes": {"g_values_bf16": N * k * 2, "u_values_bf16": N * k * 2,
                      "indices_int16": N * k * 2}}


def codec_c6_fp8(ctx, gen):
    """C6: float8_e4m3fn with a per-tensor fp16 scale (1 byte/entry/tensor)."""
    N, D = ctx["N"], ctx["D"]

    def q(t32):
        scale = (t32.abs().max() / FP8_MAX).half().float().clamp(min=1e-12)
        t8 = (t32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return t8.float() * scale

    g_rec, u_rec = q(ctx["g32"]), q(ctx["u32"])
    return {"act": F.silu(g_rec), "sp": silu_prime(g_rec), "u": u_rec,
            "bytes": {"h_gate_fp8": N * D, "h_up_fp8": N * D,
                      "per_tensor_scales_fp16": 4}}


def build_codecs():
    codecs = []  # (name, family, fn, is_stochastic)
    codecs.append(("c0a_v2_topk30", "C0-reference", codec_c0a_v2_topk, False))
    codecs.append(("c0b_exact_recompute", "C0-reference", codec_c0b_exact_recompute, False))
    codecs.append(("c0c_store_full_bf16", "C0-reference", codec_c0c_store_full, False))
    for bits in (2, 3, 4, 8):
        fam = "C1" if bits == 4 else "C2"
        codecs.append((f"{fam.lower()}_int{bits}_det", fam, make_codec_int(bits, False), False))
        codecs.append((f"{fam.lower()}_int{bits}_stoch", fam, make_codec_int(bits, True), True))
    for b_neg in (2, 3):
        for b_pos in (4, 8):
            for b_up_neg in (0, 2, 4):
                name = f"c3_gateaware_bneg{b_neg}_bpos{b_pos}_bupneg{b_up_neg}"
                codecs.append((name, "C3", make_codec_gate_aware(b_neg, b_pos, b_up_neg), False))
    for tau in (-1.0, -2.0, -3.0):
        codecs.append((f"c4_deadband_tau{int(tau)}_int4", "C4",
                       make_codec_deadband(tau, 4, False), False))
    codecs.append(("c5_topk30_factors_bf16", "C5", codec_c5_topk_factors, False))
    codecs.append(("c6_fp8_e4m3", "C6", codec_c6_fp8, False))
    # C7: dead-band tau=-2 + INT4 is exactly c4_deadband_tau-2_int4; add the
    # stochastic-rounding variant as the distinct C7 datapoint.
    codecs.append(("c7_deadband_tau-2_int4_stoch", "C7",
                   make_codec_deadband(-2.0, 4, True), True))
    return codecs


# --------------------------------------------------------------------------
# Steps 2-4: ground truth, reconstruction grads, metrics
# --------------------------------------------------------------------------

def grads_from(act, sp, u, ghm, x32, gy32, Wg32, Wu32):
    """The 4 FFN grads from (h_act, silu', h_up) reconstructions, fp32."""
    ghg = ghm * u * sp           # grad_h_gate
    ghu = ghm * act              # grad_h_up
    return {
        "grad_w_gate": ghg.T @ x32,
        "grad_w_up": ghu.T @ x32,
        "grad_w_down": gy32.T @ (act * u),
        "grad_x": ghg @ Wg32 + ghu @ Wu32,
    }


def compare(a, b):
    a64, b64 = a.double().flatten(), b.double().flatten()
    cos = F.cosine_similarity(a64, b64, dim=0).item()
    rel = ((a64 - b64).norm() / b64.norm()).item()
    return {"cos": cos, "relerr": rel}


def make_ctx(rec, weights_L, device):
    g16 = rec["h_gate"].to(device).flatten(0, -2)
    u16 = rec["h_up"].to(device).flatten(0, -2)
    x16 = rec["x"].to(device).flatten(0, -2)
    gy16 = rec["grad_y"].to(device).flatten(0, -2)
    g32, u32 = g16.float(), u16.float()
    x32, gy32 = x16.float(), gy16.float()
    Wg32 = weights_L["w_gate"].to(device).float()
    Wu32 = weights_L["w_up"].to(device).float()
    Wd32 = weights_L["w_down"].to(device).float()
    act32, sp32 = F.silu(g32), silu_prime(g32)
    ctx = {
        "g16": g16, "u16": u16, "g32": g32, "u32": u32,
        "x32": x32, "gy32": gy32, "act32": act32, "sp32": sp32,
        "mid32": act32 * u32, "Wg32": Wg32, "Wu32": Wu32, "Wd32": Wd32,
        "N": g16.shape[0], "D": g16.shape[1], "D_hid": x16.shape[1],
        "ghm": gy32 @ Wd32,  # grad_h_mid: exact in ALL codecs (from grad_y)
    }
    gt = grads_from(act32, sp32, u32, ctx["ghm"], x32, gy32, Wg32, Wu32)
    return ctx, gt


def eval_codec(fn, ctx, gt, gen):
    out = fn(ctx, gen)
    nbytes = float(sum(out["bytes"].values()))
    mode = out.get("mode", "recon")
    if mode == "exact":
        metrics = {g: {"cos": 1.0, "relerr": 0.0} for g in GRAD_NAMES}
    elif mode == "v2":
        metrics = {g: {"cos": 1.0, "relerr": 0.0}
                   for g in ("grad_w_gate", "grad_w_up", "grad_x")}
        gwd = ctx["gy32"].T @ out["mid_sparse"]
        metrics["grad_w_down"] = compare(gwd, gt["grad_w_down"])
    else:
        cg = grads_from(out["act"], out["sp"], out["u"], ctx["ghm"],
                        ctx["x32"], ctx["gy32"], ctx["Wg32"], ctx["Wu32"])
        metrics = {g: compare(cg[g], gt[g]) for g in GRAD_NAMES}
    return metrics, nbytes, out["bytes"], out.get("extra", {})


def run_eval(cap, weights, args, device):
    codecs = build_codecs()
    results = {name: {"family": fam, "stochastic": st, "per_layer": {}}
               for name, fam, _, st in codecs}
    x_bytes_ref = None
    for L in args.layers:
        for b in range(args.n_batches):
            ctx, gt = make_ctx(cap[L][b], weights[L], device)
            x_bytes_ref = ctx["N"] * ctx["D_hid"] * 2
            for name, fam, fn, stoch in codecs:
                gen = torch.Generator(device=device)
                gen.manual_seed(args.seed * 100003 + L * 1009 + b * 101
                                + abs(hash(name)) % 100000)
                m, nbytes, bb, extra = eval_codec(fn, ctx, gt, gen)
                rec = results[name]["per_layer"].setdefault(str(L), [])
                rec.append({"batch": b, "metrics": m, "bytes": nbytes,
                            "bytes_breakdown": bb, "extra": extra})
            del ctx, gt
            torch.cuda.empty_cache()
        print(f"  layer {L} done ({time.strftime('%H:%M:%S')})", flush=True)
    return results, x_bytes_ref


def bias_check(cap, weights, args, device):
    """Average reconstructed grads over 16 independent stochastic-rounding
    draws (one layer, one batch); cos/relerr of the AVERAGED grad vs ground
    truth. Unbiased noise => relerr of 16-draw mean ~ single-draw/4."""
    L = args.layers[len(args.layers) // 2]  # middle hooked layer (11)
    ctx, gt = make_ctx(cap[L][0], weights[L], device)
    variants = {f"c{'1' if bits == 4 else '2'}_int{bits}_stoch":
                make_codec_int(bits, True) for bits in (2, 3, 4, 8)}
    variants["c7_deadband_tau-2_int4_stoch"] = make_codec_deadband(-2.0, 4, True)
    n_draws = 16
    out = {}
    for name, fn in variants.items():
        sums = {g: torch.zeros_like(gt[g], dtype=torch.float64) for g in GRAD_NAMES}
        single = []
        for d in range(n_draws):
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed * 7919 + d)
            r = fn(ctx, gen)
            cg = grads_from(r["act"], r["sp"], r["u"], ctx["ghm"],
                            ctx["x32"], ctx["gy32"], ctx["Wg32"], ctx["Wu32"])
            for g in GRAD_NAMES:
                sums[g] += cg[g].double()
            single.append({g: compare(cg[g], gt[g]) for g in GRAD_NAMES})
        rec = {}
        for g in GRAD_NAMES:
            avg = compare(sums[g] / n_draws, gt[g])
            mean_single_rel = sum(s[g]["relerr"] for s in single) / n_draws
            ratio = avg["relerr"] / mean_single_rel if mean_single_rel > 0 else 0.0
            # unbiased noise scales ~1/sqrt(16)=0.25; systematic bias doesn't
            verdict = ("unbiased (noise-dominated)" if ratio < 0.40 else
                       "mostly unbiased" if ratio < 0.70 else "BIASED (systematic)")
            rec[g] = {"avg16_cos": avg["cos"], "avg16_relerr": avg["relerr"],
                      "mean_single_relerr": mean_single_rel,
                      "avg16_over_single_ratio": ratio, "verdict": verdict}
        out[name] = {"layer": L, "batch": 0, "n_draws": n_draws, "grads": rec}
        print(f"  bias check {name}: gwd ratio="
              f"{rec['grad_w_down']['avg16_over_single_ratio']:.3f} "
              f"({rec['grad_w_down']['verdict']})", flush=True)
    return out


# --------------------------------------------------------------------------
# Step 5: aggregation + report
# --------------------------------------------------------------------------

def aggregate(results, x_bytes_ref):
    """Per codec: per-layer batch-mean cos/relerr -> min & median over layers
    (never average away a bad layer); bytes averaged (+max) over layer/batch."""
    agg = {}
    for name, r in results.items():
        layer_means = {g: [] for g in GRAD_NAMES}
        layer_worst = {g: [] for g in GRAD_NAMES}
        all_bytes = []
        for L, recs in r["per_layer"].items():
            for g in GRAD_NAMES:
                cosv = [x["metrics"][g]["cos"] for x in recs]
                layer_means[g].append(sum(cosv) / len(cosv))
                layer_worst[g].append(min(cosv))
            all_bytes += [x["bytes"] for x in recs]
        per_grad = {}
        for g in GRAD_NAMES:
            ms = sorted(layer_means[g])
            per_grad[g] = {
                "min_over_layers_cos": min(layer_means[g]),
                "median_over_layers_cos": ms[len(ms) // 2],
                "worst_single_batch_cos": min(layer_worst[g]),
            }
        mean_bytes = sum(all_bytes) / len(all_bytes)
        agg[name] = {
            "family": r["family"],
            "stochastic": r["stochastic"],
            "per_grad": per_grad,
            "codec_bytes_mean": mean_bytes,
            "codec_bytes_max": max(all_bytes),
            "total_bytes_with_x": None,  # filled by fix_totals()
            "overall_min_cos": min(per_grad[g]["min_over_layers_cos"] for g in GRAD_NAMES),
        }
    return agg


def fix_totals(agg, x_bytes_ref):
    """total_bytes_with_x: codec bytes + x (x is required for grad_w_gate/up
    in ALL trainable-weight scenarios). C0a/C0b already include x in their
    own counts; C0c (activations mode) gets x added for the trainable-case
    comparison column as well, with a note in the report."""
    for name, a in agg.items():
        if name in ("c0a_v2_topk30", "c0b_exact_recompute"):
            a["total_bytes_with_x"] = a["codec_bytes_mean"]  # x already counted
        else:
            a["total_bytes_with_x"] = a["codec_bytes_mean"] + x_bytes_ref
    return agg


def pareto_frontier(agg):
    """Frontier over v3 STORAGE codecs only (C0a needs the v2 top-K pass and
    C0b needs two recompute GEMMs in backward -- they are the incumbents the
    codecs are compared against, not points on the storage-codec frontier)."""
    pts = sorted(((a["total_bytes_with_x"], a["overall_min_cos"], n)
                  for n, a in agg.items()
                  if n not in ("c0a_v2_topk30", "c0b_exact_recompute")))
    frontier, best = [], -2.0
    for nbytes, cos, name in pts:
        if cos > best:
            frontier.append({"codec": name, "total_bytes_with_x": nbytes,
                             "overall_min_cos": cos})
            best = cos
    return frontier


def fmt_bytes(b):
    return f"{b / 1024 / 1024:.2f} MiB"


def write_summary(path, meta, stats, verification, agg, bias, frontier,
                  results, x_bytes_ref):
    n_pairs = meta["tokens_per_batch"] * 5632  # N * D_int entries per tensor
    lines = []
    w = lines.append
    w("# FlashFFN v3 Codec Feasibility Study")
    w("")
    w(f"- Model: {meta['model']} (bf16), dataset: {meta['dataset']}")
    w(f"- Hooked layers: {meta['layers']}; {meta['n_batches']} batches x "
      f"(batch={meta['batch_size']}, seq={meta['seq_len']}) -> N={meta['tokens_per_batch']} tokens/batch")
    w(f"- CLM losses per batch: {[round(l, 4) for l in meta['losses']]}")
    w(f"- Ground truth: fp32 grads from the SAME bf16 tensors the codecs see (TF32 off).")
    w(f"- Bytes are per layer per batch at N={meta['tokens_per_batch']} "
      f"(the 'N=2048 equivalent' directly). x (bf16) = {fmt_bytes(x_bytes_ref)}; "
      f"store-full bf16 pair = {fmt_bytes(2 * n_pairs * 2)}.")
    w("")
    w("## 1. Activation statistics per layer")
    w("")
    w("| layer | frac g<0 | frac g<-1 | frac g<-2 | frac g<-3 | g p1/p50/p99 | up p1/p50/p99 | mid p1/p50/p99 |")
    w("|---|---|---|---|---|---|---|---|")
    for L, s in stats.items():
        gp, up, mp = s["h_gate_pcts"], s["h_up_pcts"], s["h_mid_pcts"]
        w(f"| {L} | {s['frac_h_gate_lt_0']:.3f} | {s['frac_h_gate_lt_-1']:.3f} "
          f"| {s['frac_h_gate_lt_-2']:.3f} | {s['frac_h_gate_lt_-3']:.3f} "
          f"| {gp['p1']:.2f}/{gp['p50']:.2f}/{gp['p99']:.2f} "
          f"| {up['p1']:.2f}/{up['p50']:.2f}/{up['p99']:.2f} "
          f"| {mp['p1']:.2f}/{mp['p50']:.2f}/{mp['p99']:.2f} |")
    w("")
    w("Full percentiles (p0..p100) are in codec_results.json.")
    w("")
    w("## 2. Recompute verification (honesty check)")
    w("")
    for L, v in verification.items():
        parts = [f"{k}: max|diff|={d['max_abs_diff']:.2e}, "
                 f"bitwise-equal={d['frac_bitwise_equal'] * 100:.1f}%"
                 for k, d in v.items()]
        w(f"- layer {L}: " + "; ".join(parts))
    w("")
    w("## 3. Decision table")
    w("")
    w(f"Bytes at N={meta['tokens_per_batch']}; 'total' includes stored x "
      f"({fmt_bytes(x_bytes_ref)}) needed for grad_w_gate/up in any "
      "trainable-weight scenario (C0a/C0b counts already include x; for "
      "C0c/frozen-weights x would not be needed). min cos = min over layers "
      "of the batch-mean cosine (worst layer is NOT averaged away).")
    w("")
    w("| codec | codec bytes | total (+x) | bits/pair | min-cos gw_gate | min-cos gw_up | min-cos gw_down | min-cos gx | bias |")
    w("|---|---|---|---|---|---|---|---|---|")
    order = sorted(agg.items(), key=lambda kv: kv[1]["total_bytes_with_x"])
    for name, a in order:
        bias_str = "-"
        if name in bias:
            bias_str = bias[name]["grads"]["grad_w_down"]["verdict"]
        elif a["stochastic"]:
            bias_str = "(stoch)"
        pg = a["per_grad"]
        bits_pair = a["codec_bytes_mean"] * 8 / n_pairs
        w(f"| {name} | {fmt_bytes(a['codec_bytes_mean'])} | "
          f"{fmt_bytes(a['total_bytes_with_x'])} | {bits_pair:.2f} | "
          + " | ".join(f"{pg[g]['min_over_layers_cos']:.6f}" for g in GRAD_NAMES)
          + f" | {bias_str} |")
    w("")
    w("(bits/pair = stored bits per (h_gate, h_up) element pair, excluding x;"
      " bf16 store-full = 32.)")
    w("")
    w("## 4. grad_x focus (propagates upstream; v2 has it EXACT)")
    w("")
    w("| codec | min-cos grad_x (layer mean) | median-cos grad_x | worst single batch cos |")
    w("|---|---|---|---|")
    for name, a in order:
        gx = a["per_grad"]["grad_x"]
        w(f"| {name} | {gx['min_over_layers_cos']:.6f} | "
          f"{gx['median_over_layers_cos']:.6f} | "
          f"{gx['worst_single_batch_cos']:.6f} |")
    w("")
    w("## 5. Stochastic-rounding bias check (16-draw averaged grad, layer "
      f"{list(bias.values())[0]['layer'] if bias else '-'}, batch 0)")
    w("")
    w("| variant | grad | mean single relerr | 16-draw-avg relerr | ratio (0.25 = pure noise) | verdict |")
    w("|---|---|---|---|---|---|")
    for name, rec in bias.items():
        for g in GRAD_NAMES:
            d = rec["grads"][g]
            w(f"| {name} | {g} | {d['mean_single_relerr']:.5f} | "
              f"{d['avg16_relerr']:.5f} | {d['avg16_over_single_ratio']:.3f} | {d['verdict']} |")
    w("")
    w("## 6. Pareto frontier (total bytes incl. x vs overall min-cos)")
    w("")
    w("Frontier over v3 storage codecs (C1-C7 + store-full). Reference "
      "points: c0b_exact_recompute = "
      f"{fmt_bytes(agg['c0b_exact_recompute']['total_bytes_with_x'])} at cos "
      "1.0 but pays 2 recompute GEMMs in backward; c0a_v2_topk30 = "
      f"{fmt_bytes(agg['c0a_v2_topk30']['total_bytes_with_x'])} at overall "
      f"min-cos {agg['c0a_v2_topk30']['overall_min_cos']:.6f} "
      "(gw_down only approx).")
    w("")
    w("| codec | total bytes | overall min-cos (worst grad, worst layer) |")
    w("|---|---|---|")
    for p in frontier:
        w(f"| {p['codec']} | {fmt_bytes(p['total_bytes_with_x'])} | {p['overall_min_cos']:.6f} |")
    w("")
    w("## 7. Takeaways")
    w("")
    w("(Filled in by analysis -- see final report.)")
    w("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11, 21])
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out_dir", default="results/v3_codec")
    ap.add_argument("--quick", action="store_true",
                    help="1 layer, 1 batch smoke test")
    args = ap.parse_args()
    if args.quick:
        args.layers, args.n_batches = [11], 1

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    print("Step 1: capturing real activations + gradients ...", flush=True)
    cap, weights, stats, verification, meta = capture(args, device)
    print(f"  capture done in {time.time() - t0:.0f}s; losses={meta['losses']}",
          flush=True)
    for L, s in stats.items():
        print(f"  layer {L}: frac(g<0)={s['frac_h_gate_lt_0']:.3f} "
              f"frac(g<-1)={s['frac_h_gate_lt_-1']:.3f} "
              f"frac(g<-2)={s['frac_h_gate_lt_-2']:.3f} "
              f"frac(g<-3)={s['frac_h_gate_lt_-3']:.3f}", flush=True)

    print("Steps 2-4: ground truth + codec evaluation ...", flush=True)
    results, x_bytes_ref = run_eval(cap, weights, args, device)

    print("Step 4b: stochastic-rounding bias check ...", flush=True)
    bias = bias_check(cap, weights, args, device)

    agg = fix_totals(aggregate(results, x_bytes_ref), x_bytes_ref)
    frontier = pareto_frontier(agg)

    out = {
        "meta": meta,
        "config": vars(args),
        "activation_stats": {str(k): v for k, v in stats.items()},
        "recompute_verification": {str(k): v for k, v in verification.items()},
        "x_bytes_per_layer": x_bytes_ref,
        "per_codec_raw": results,
        "aggregate": agg,
        "bias_check": bias,
        "pareto_frontier": frontier,
    }
    jpath = os.path.join(args.out_dir, "codec_results.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    mpath = os.path.join(args.out_dir, "summary.md")
    write_summary(mpath, meta, stats, verification, agg, bias, frontier,
                  results, x_bytes_ref)
    print(f"Done in {time.time() - t0:.0f}s. Wrote {jpath} and {mpath}",
          flush=True)


if __name__ == "__main__":
    main()
