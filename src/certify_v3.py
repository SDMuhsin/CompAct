"""Certified error bounds for the FlashFFN v3 quantized GLU-cache backward.

Derives and validates COMPUTABLE worst-case bounds on every gradient tensor
produced by FlashFFNv3Function.backward when (h_gate, h_up) are stored under
the v3 symmetric absmax per-group quantizer (group=64, fp16 scales, bf16
dequant) instead of exactly. Full derivation: llmdocs/trackers/v3_bounds.md.

Pipeline (single process):
  Stage 0  Constants: L_silu = sup|silu'| (bisection on silu''=0, fp64) and
           L_silu' = sup|silu''| = 1/2 (proved analytically; grid-verified).
  Stage 1  Quantizer mirror gate: a generic-bits mirror of
           flashffn.v3_quantize/dequantize_group_absmax must be BITWISE equal
           to the real functions at bits in {4, 8} (mirror is used for 2/3).
  Stage 2  Calibration capture: TinyLlama-1.1B-Chat (bf16) + PEFT LoRA r=16
           on gate/up/down of all 22 MLPs, wikitext-2 (cached), 3 batches of
           (batch=4, seq=512) -> N=2048 tokens each. Batches 0,1 calibrate;
           batch 2 is held out. Hooks capture x, h_gate, h_up, grad_y,
           grad(MLP input), grad(block input) per layer per batch.
  Stage 3  Per-layer per-bit bound calibration (fp64): quantization envelopes
           -> elementwise backward envelopes -> per-tensor Frobenius bounds
           (spectral / Frobenius / per-token-rowwise variants) -> rho tables.
  Stage 4  Local validation (layers {0,5,11,21} x bits {2,3,4,8} x 2 batches):
           measured ||Delta T||_F / ||T||_F from the ACTUAL v3 quantizer vs
           the bound. REQUIREMENT: bound >= measured in 100% of cases.
  Stage 5  Depth composition: real full-model backward, all layers int4
           (actual FlashFFNv3Function) vs exact (bf16 cache mode), adapter
           grads at layers {0,5,11,21} vs the composed semi-empirical bound.
  Stage 6  Certified allocation choose_bits(eps): per-layer cache_mode in
           {recompute, int2, int3, int4, int8} minimizing stored bytes s.t.
           composed bound <= eps (recompute excluded from the byte-greedy --
           it has 0 cache bytes and rho=0, so unconstrained byte minimization
           degenerates to all-recompute; it is admitted only when quantization
           alone is infeasible, lexicographically minimizing recompute count).
  Stage 7  Frontier: measured bytes-vs-fidelity for uniform int2/3/4/8 and
           the certified allocations (eps in {0.02, 0.05, 0.10}).

All bound arithmetic in fp64. Measured deltas in Stage 4 are computed with
fp64 implementations of the EXACT formulas in FlashFFNv3Function.backward,
fed the bf16 tensors the real kernels see (the bound is a statement about
exact arithmetic on the bf16 saved tensors; Stage 5 additionally exercises
the real bf16/triton kernel path end to end).

Run from repo root:
  source env/bin/activate
  HF_HOME=./data TORCH_HOME=./data CUDA_VISIBLE_DEVICES=1 \
      python src/certify_v3.py [--quick]
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import argparse
import json
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashffn  # noqa: E402
from flashffn import (  # noqa: E402
    make_v3_forward,
    triton_silu_mul,
    v3_dequantize_group_absmax,
    v3_quantize_group_absmax,
    v3_reset_counters,
    _V3_COUNTERS,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GROUP = 64
RANK, ALPHA = 16, 32
BITS_ALL = (2, 3, 4, 8)
VAL_LAYERS = (0, 5, 11, 21)
ADAPTER_TENSORS = ("grad_b_gate", "grad_a_gate", "grad_b_up", "grad_a_up",
                   "grad_b_down", "grad_a_down")
DENSE_TENSORS = ("grad_w_gate", "grad_w_up", "grad_w_down")
ALL_TENSORS = ADAPTER_TENSORS + ("grad_x",) + DENSE_TENSORS

# Unit roundoffs (round-to-nearest): |fl(x) - x| <= u*|x|.
U_BF16 = 2.0 ** -8      # bf16: 8-bit significand (7 stored + implicit)
U_FP16 = 2.0 ** -11     # fp16: 11-bit significand
U_FP32 = 2.0 ** -24     # fp32: 24-bit significand
# Expressed against the ROUNDED value: |fl(x) - x| <= U/(1-U) * |fl(x)|.
BF16_OUT = U_BF16 / (1.0 - U_BF16)

_CERT_COUNTERS = {"forward": 0, "backward": 0}


def log(msg):
    print(msg, flush=True)


# ===========================================================================
# Stage 0 -- exact SiLU Lipschitz constants
# ===========================================================================

def compute_silu_constants():
    """L1 = sup_z |silu'(z)|, L2 = sup_z |silu''(z)|.

    silu'(z)  = sig(z) * (1 + z*(1 - sig(z)))
    silu''(z) = sig(z)*(1-sig(z)) * (2 + z*(1 - 2*sig(z)))   [even function]

    L2 = 1/2 EXACTLY: for the bracket b(z) = 2 - z*(2*sig(z)-1) >= 0 (z >= 0),
    |silu''| <= (1/4)*2 = 1/2 with equality at z=0; where b(z) < 0 (z > z*),
    |silu''| <= (1-sig(z))*z <= z*exp(-z) <= 0.22 < 1/2. By evenness, global.
    L1 is attained at the positive root z* of b(z) = 0 (transcendental);
    computed by bisection in fp64, then a +1e-9 safety margin is added.
    Both are verified on a dense fp64 grid (tails decay like z*exp(-|z|)).
    """
    def sig(z):
        return 1.0 / (1.0 + math.exp(-z))

    def bracket(z):
        return 2.0 + z * (1.0 - 2.0 * sig(z))

    lo, hi = 2.0, 3.0
    assert bracket(lo) > 0 > bracket(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bracket(mid) > 0:
            lo = mid
        else:
            hi = mid
    z_star = 0.5 * (lo + hi)
    l1_exact = sig(z_star) * (1.0 + z_star * (1.0 - sig(z_star)))
    L1 = l1_exact + 1e-9
    L2 = 0.5

    z = torch.linspace(-60.0, 60.0, 4_000_001, dtype=torch.float64)
    s = torch.sigmoid(z)
    sp = s * (1.0 + z * (1.0 - s))
    spp = s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))
    grid_l1 = float(sp.abs().max())
    grid_l2 = float(spp.abs().max())
    assert grid_l1 <= L1, (grid_l1, L1)
    assert grid_l2 <= L2 + 1e-12, (grid_l2, L2)
    return {"z_star": z_star, "L1_sup_abs_silu_prime": L1,
            "L1_exact_value": l1_exact, "L2_sup_abs_silu_second": L2,
            "grid_max_abs_silu_prime": grid_l1,
            "grid_max_abs_silu_second": grid_l2}


# ===========================================================================
# Stage 1 -- generic-bits quantizer mirror (+ certified per-entry envelope)
# ===========================================================================

def quant_any(t, bits, group_size=GROUP):
    """Mirror of flashffn.v3_quantize_group_absmax for any bits >= 2.

    Identical semantics (fp16 scales, round-to-nearest, clamp to +-qmax);
    codes returned unpacked as int8. Gated bitwise against the real functions
    at bits in {4, 8} (see mirror_gate)."""
    N, D = t.shape
    assert D % group_size == 0
    G = D // group_size
    qmax = float(2 ** (bits - 1) - 1)
    tg = t.reshape(N, G, group_size).float()
    absmax = tg.abs().amax(dim=-1, keepdim=True)
    scales = (absmax / qmax).to(torch.float16)
    safe = scales.float()
    safe = torch.where(safe > 0, safe, torch.ones_like(safe))
    q = torch.round(tg / safe).clamp_(-qmax, qmax)
    return q.reshape(N, D).to(torch.int8), scales.reshape(N, G)


def deq_any(q, scales, group_size=GROUP, dtype=torch.bfloat16):
    N, D = q.shape
    G = D // group_size
    deq = q.to(torch.float32).reshape(N, G, group_size) \
        * scales.reshape(N, G, 1).to(torch.float32)
    return deq.reshape(N, D).to(dtype)


def quantize_v3(h16, bits, group_size=GROUP):
    """Quantize+dequantize h16 (bf16 [N, D]) with v3 codec semantics.

    bits in {4, 8}: the REAL flashffn functions. bits in {2, 3}: the gated
    mirror. Returns (h_hat bf16, scales fp16 [N, G])."""
    if bits in (4, 8):
        payload, scales = v3_quantize_group_absmax(h16, bits, group_size)
        h_hat = v3_dequantize_group_absmax(payload, scales, bits, group_size,
                                           torch.bfloat16)
    else:
        q, scales = quant_any(h16, bits, group_size)
        h_hat = deq_any(q, scales, group_size, torch.bfloat16)
    return h_hat, scales


def quant_envelope(h16, bits, group_size=GROUP):
    """Certified per-entry quantization-error envelope (fp64).

    Returns (h_hat bf16 [N,D], env fp64 [N,D]) with the guarantee
        |h_hat - h16| <= env   entrywise,
    accounting for: round-to-nearest (1/2 code), fp16 scale rounding /
    clamp overshoot (per-group o = max(0, absmax/s - qmax), exact), fp32
    division rounding ((qmax+1)*2^-24 codes), and the bf16 output cast
    (BF16_OUT * |h_hat|). q*s is exact in fp32 (<= 18 significand bits).
    Zero-scale groups reconstruct to 0; their envelope is absmax itself.
    """
    N, D = h16.shape
    G = D // group_size
    qmax = float(2 ** (bits - 1) - 1)
    h_hat, scales = quantize_v3(h16, bits, group_size)
    s64 = scales.to(torch.float64)                                  # [N, G]
    absmax64 = h16.to(torch.float64).abs().reshape(N, G, group_size).amax(-1)
    safe = torch.where(s64 > 0, s64, torch.ones_like(s64))
    overshoot = (absmax64 / safe - qmax).clamp_(min=0.5)            # max(1/2, o)
    base = s64 * (overshoot + (qmax + 1.0) * U_FP32)                # [N, G]
    base = torch.where(s64 > 0, base, absmax64)
    env = base.reshape(N, G, 1).expand(N, G, group_size).reshape(N, D).contiguous()
    env += BF16_OUT * h_hat.to(torch.float64).abs()
    return h_hat, env


def mirror_gate(caps, device):
    """Mirror must be BITWISE equal to the real v3 codec at bits 4 and 8,
    on synthetic AND real captured tensors. Also: per-entry envelope must
    hold on every tested tensor for all bits."""
    torch.manual_seed(123)
    tensors = [torch.randn(256, 512, device=device).to(torch.bfloat16) * 3.0,
               torch.zeros(64, 128, device=device, dtype=torch.bfloat16)]
    t = torch.randn(64, 128, device=device) * 1e-6                  # tiny scales
    tensors.append(t.to(torch.bfloat16))
    for cell in caps.values():
        tensors.append(cell[0]["h_gate"].to(device).reshape(-1, 5632))
        break
    rep = {"bitwise_equal_4": True, "bitwise_equal_8": True,
           "envelope_holds": True, "max_env_slack": 0.0}
    for t16 in tensors:
        for bits in (4, 8):
            p_real, s_real = v3_quantize_group_absmax(t16, bits, GROUP)
            d_real = v3_dequantize_group_absmax(p_real, s_real, bits, GROUP,
                                                torch.bfloat16)
            q_mir, s_mir = quant_any(t16, bits, GROUP)
            d_mir = deq_any(q_mir, s_mir, GROUP, torch.bfloat16)
            ok = torch.equal(d_real, d_mir) and torch.equal(s_real, s_mir)
            rep[f"bitwise_equal_{bits}"] &= ok
            assert ok, f"mirror mismatch at bits={bits}"
        for bits in BITS_ALL:
            h_hat, env = quant_envelope(t16, bits, GROUP)
            delta = (h_hat.to(torch.float64) - t16.to(torch.float64)).abs()
            slack = float((delta - env).max())
            rep["max_env_slack"] = max(rep["max_env_slack"], slack)
            if slack > 0:
                rep["envelope_holds"] = False
            assert slack <= 0, f"quant envelope violated: bits={bits} by {slack}"
    return rep


# ===========================================================================
# Stage 2 -- model + capture
# ===========================================================================

def build_model(seed, device):
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # attn_implementation="eager": SDPA's fused attention BACKWARD is
    # non-deterministic (atomics); run-to-run jitter on adapter grads reaches
    # 4e-3 absolute, which would contaminate exact-vs-quantized comparisons.
    # Eager attention makes the full backward bitwise deterministic.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    model.eval()
    model.config.use_cache = False

    cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0,
                     target_modules=["gate_proj", "up_proj", "down_proj"],
                     bias="none")
    model = inject_adapter_in_model(cfg, model)

    torch.manual_seed(seed)
    for layer in model.model.layers:
        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj,
                     layer.mlp.down_proj):
            name = proj.active_adapters[0]
            b = proj.lora_B[name].weight
            with torch.no_grad():                       # CPU RNG, seeded
                b.data.copy_((torch.randn(b.shape) * 0.02).to(b.device))
    model = model.to(torch.bfloat16)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    return model, tok


def build_batches(tok, n_batches, batch_size, seq_len):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text[:900_000], return_tensors="pt").input_ids[0]
    need = n_batches * batch_size * seq_len
    assert ids.numel() >= need
    return ids[:need].view(n_batches, batch_size, seq_len)


def iter_chunks(batch_ids, micro_bs):
    """Split one batch [B, S] into micro-chunks of micro_bs sequences with the
    loss weight that makes accumulated grads equal the full-batch-mean grads
    (CE token-mean with equal-length sequences: weight = chunk_B / B)."""
    B = batch_ids.shape[0]
    mb = min(micro_bs, B)
    for c0 in range(0, B, mb):
        chunk = batch_ids[c0:c0 + mb]
        yield chunk, chunk.shape[0] / B


def capture(model, batches, device, micro_bs):
    """Per layer L, per batch b: x, h_gate, h_up (bf16, WITH LoRA), grad_y
    (MLP output grad), grad_v (MLP input grad), grad_u (block input grad).

    Forward/backward runs in micro-chunks (loss scaled by chunk/B) so that
    eager attention fits the shared GPU; captures are concatenated along the
    batch dim — the analyzed gradient IS the full-batch-mean gradient (the
    backward formulas are token sums). The same chunking is used by run_arm,
    so calibration and e2e arms see identical semantics."""
    n_layers = len(model.model.layers)
    n_b = batches.shape[0]
    caps = {L: [dict() for _ in range(n_b)] for L in range(n_layers)}
    state = {"b": 0}
    handles = []

    emb = model.get_input_embeddings()
    handles.append(emb.register_forward_hook(
        lambda m, i, o: o.requires_grad_(True)))

    def stash(L, b, key, t):
        caps[L][b].setdefault(key, []).append(t.detach().to("cpu", copy=True))

    def mlp_hook(L):
        def hook(module, inputs, output):
            b = state["b"]
            stash(L, b, "x", inputs[0])
            inputs[0].register_hook(
                lambda g, L=L, b=b: stash(L, b, "grad_v", g))
            output.register_hook(
                lambda g, L=L, b=b: stash(L, b, "grad_y", g))
        return hook

    def proj_hook(L, key):
        def hook(module, inputs, output):
            stash(L, state["b"], key, output)
        return hook

    def block_pre_hook(L):
        def hook(module, args):
            b = state["b"]
            args[0].register_hook(
                lambda g, L=L, b=b: stash(L, b, "grad_u", g))
        return hook

    for L, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(mlp_hook(L)))
        handles.append(layer.mlp.gate_proj.register_forward_hook(
            proj_hook(L, "h_gate")))
        handles.append(layer.mlp.up_proj.register_forward_hook(
            proj_hook(L, "h_up")))
        handles.append(layer.register_forward_pre_hook(block_pre_hook(L)))

    losses = []
    for b in range(n_b):
        state["b"] = b
        total = 0.0
        for chunk, w in iter_chunks(batches[b], micro_bs):
            ids = chunk.to(device)
            out = model(input_ids=ids, labels=ids)
            total += float(out.loss.detach()) * w
            (out.loss * w).backward()
            del out
            torch.cuda.empty_cache()
        losses.append(total)
        model.zero_grad(set_to_none=True)
    for h in handles:
        h.remove()
    # Concatenate chunk captures along the batch dim (grad hooks fire during
    # each chunk's backward, immediately after its forward -> lists aligned).
    for L in range(n_layers):
        for b in range(n_b):
            for key in list(caps[L][b].keys()):
                caps[L][b][key] = torch.cat(caps[L][b][key], dim=0)

    # Honesty check: hooked h_gate/h_up must match an offline bf16 recompute
    # through the same factored ops the v3 kernels use.
    verif = {}
    for L in (0, n_layers // 2, n_layers - 1):
        P = layer_params(model, L)
        x = caps[L][0]["x"].to(device).reshape(-1, x_dim(caps[L][0]["x"]))
        hg = F.linear(x, P["w_gate"])
        hg = hg + P["s_gate"] * F.linear(F.linear(x, P["a_gate"]), P["b_gate"])
        hu = F.linear(x, P["w_up"])
        hu = hu + P["s_up"] * F.linear(F.linear(x, P["a_up"]), P["b_up"])
        got_g = caps[L][0]["h_gate"].to(device).reshape(hg.shape)
        got_u = caps[L][0]["h_up"].to(device).reshape(hu.shape)
        verif[L] = {
            "h_gate_bitwise": float((hg == got_g).float().mean()),
            "h_up_bitwise": float((hu == got_u).float().mean()),
            "h_gate_max_absdiff": float((hg.float() - got_g.float()).abs().max()),
            "h_up_max_absdiff": float((hu.float() - got_u.float()).abs().max()),
        }
        assert torch.allclose(hg.float(), got_g.float(), rtol=2e-2, atol=2e-2)
    return caps, losses, verif


def x_dim(t):
    return t.shape[-1]


def layer_params(model, L):
    mlp = model.model.layers[L].mlp
    out = {}
    for key, proj in (("gate", mlp.gate_proj), ("up", mlp.up_proj),
                      ("down", mlp.down_proj)):
        w, a, b, s = flashffn._v3_extract_proj(proj)
        out[f"w_{key}"], out[f"a_{key}"], out[f"b_{key}"] = w, a, b
        out[f"s_{key}"] = s
    return out


# ===========================================================================
# fp64 helpers
# ===========================================================================

def lift(t, device):
    return t.to(device).to(torch.float64)


def spec_norm_ub(M, inflate=1e-8):
    """Upper estimate of sigma_max(M) via eigvalsh(M^T M) in fp64 (+ relative
    inflation covering backward-stable eigensolver rounding)."""
    if M.shape[0] < M.shape[1]:
        M = M.T
    S = M.T @ M
    S = (S + S.T) * 0.5
    lam = float(torch.linalg.eigvalsh(S)[-1].clamp(min=0.0))
    return math.sqrt(lam) * (1.0 + inflate)


def power_iter_est(M, iters=8, seed=0):
    """Plain power iteration estimate (LOWER estimate; diagnostic only)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(M.shape[1], generator=g).to(M.device, M.dtype)
    v = v / v.norm()
    for _ in range(iters):
        w = M.T @ (M @ v)
        v = w / w.norm()
    return float((M @ v).norm())


def ffn_backward_fp64(x64, gy64, hg64, hu64, P64, ghm64=None):
    """fp64 implementation of the EXACT formulas in FlashFFNv3Function.backward
    (LoRA on all three projections + dense full-FT grads). All inputs fp64."""
    s_g, s_u, s_d = P64["s_gate"], P64["s_up"], P64["s_down"]
    if ghm64 is None:
        ghm64 = gy64 @ P64["w_down"] \
            + s_d * ((gy64 @ P64["b_down"]) @ P64["a_down"])
    sig = torch.sigmoid(hg64)
    act = hg64 * sig
    sp = sig * (1.0 + hg64 * (1.0 - sig))
    phi_g = ghm64 * hu64 * sp                       # grad_h_gate
    phi_u = ghm64 * act                             # grad_h_up
    hmid = act * hu64
    gyB_d = gy64 @ P64["b_down"]
    out = {
        "grad_b_gate": s_g * (phi_g.T @ P64["xa_g"]),
        "grad_a_gate": s_g * ((phi_g @ P64["b_gate"]).T @ x64),
        "grad_b_up": s_u * (phi_u.T @ P64["xa_u"]),
        "grad_a_up": s_u * ((phi_u @ P64["b_up"]).T @ x64),
        "grad_b_down": s_d * (gy64.T @ P64["ha_d"]),
        "grad_a_down": s_d * (gyB_d.T @ hmid),
        "grad_x": phi_g @ P64["w_gate"]
        + s_g * ((phi_g @ P64["b_gate"]) @ P64["a_gate"])
        + phi_u @ P64["w_up"]
        + s_u * ((phi_u @ P64["b_up"]) @ P64["a_up"]),
        "grad_w_gate": phi_g.T @ x64,
        "grad_w_up": phi_u.T @ x64,
        "grad_w_down": gy64.T @ hmid,
    }
    return out, phi_g, phi_u, hmid


# ===========================================================================
# Stage 3/4 -- bounds + local validation
# ===========================================================================

def backward_envelopes(g_abs, hg_hat64, hu_hat64, env_g, env_u, L1, L2):
    """Elementwise certified envelopes through the SwiGLU backward.

    E_gg >= |Delta grad_h_gate|, E_gu >= |Delta grad_h_up|,
    E_hm >= |Delta h_mid_hat|, entrywise, in exact arithmetic.
    Anchored at the DEQUANTIZED point (hat values are known at calibration),
    which removes the second-order term from E_gg entirely; E_hm keeps its
    (rigorous) second-order term L1*env_g*env_u.
    """
    abs_hu_hat = hu_hat64.abs()
    act_hat_abs = (hg_hat64 * torch.sigmoid(hg_hat64)).abs()
    E_gg = g_abs * (L2 * abs_hu_hat * env_g + L1 * env_u)
    E_gu = g_abs * (L1 * env_g)
    E_hm = L1 * abs_hu_hat * env_g + (act_hat_abs + L1 * env_g) * env_u
    return E_gg, E_gu, E_hm


def row_norms(t):
    return t.square().sum(dim=1).sqrt()


def tensor_bounds(E_gg, E_gu, E_hm, comp, P64):
    """Per-tensor Frobenius bounds, three rigorous variants each:
      spec : ||E||_F * sigma_max(companion)        (submultiplicative, sharp norm)
      frob : ||E||_F * ||companion||_F             (looser, no spectral norm)
      row  : per-token Cauchy-Schwarz  sum_i ||E_i|| * ||companion_i||
    Operative bound = min(spec, row) (frob reported). grad_b_down is EXACT
    in v3 (it uses only saved exact tensors) -> bound 0."""
    s_g, s_u, s_d = P64["s_gate"], P64["s_up"], P64["s_down"]
    f_gg, f_gu, f_hm = float(E_gg.norm()), float(E_gu.norm()), float(E_hm.norm())
    r_gg, r_gu, r_hm = row_norms(E_gg), row_norms(E_gu), row_norms(E_hm)
    b = {}

    def pack(spec, frob, row=None):
        op = spec if row is None else min(spec, row)
        return {"spec": spec, "frob": frob,
                "row": row if row is not None else spec, "op": op}

    b["grad_b_gate"] = pack(
        s_g * f_gg * comp["sig_xa_g"], s_g * f_gg * comp["fro_xa_g"],
        s_g * float((r_gg * comp["rows_xa_g"]).sum()))
    b["grad_a_gate"] = pack(
        s_g * comp["sig_b_gate"] * f_gg * comp["sig_x"],
        s_g * comp["fro_b_gate"] * f_gg * comp["fro_x"],
        s_g * comp["sig_b_gate"] * float((r_gg * comp["rows_x"]).sum()))
    b["grad_b_up"] = pack(
        s_u * f_gu * comp["sig_xa_u"], s_u * f_gu * comp["fro_xa_u"],
        s_u * float((r_gu * comp["rows_xa_u"]).sum()))
    b["grad_a_up"] = pack(
        s_u * comp["sig_b_up"] * f_gu * comp["sig_x"],
        s_u * comp["fro_b_up"] * f_gu * comp["fro_x"],
        s_u * comp["sig_b_up"] * float((r_gu * comp["rows_x"]).sum()))
    b["grad_b_down"] = pack(0.0, 0.0, 0.0)
    b["grad_a_down"] = pack(
        s_d * comp["sig_gyBd"] * f_hm, s_d * comp["fro_gyBd"] * f_hm,
        s_d * float((comp["rows_gyBd"] * r_hm).sum()))
    b["grad_x"] = pack(
        f_gg * comp["sig_weff_g"] + f_gu * comp["sig_weff_u"],
        f_gg * comp["fro_weff_g"] + f_gu * comp["fro_weff_u"])
    b["grad_w_gate"] = pack(
        f_gg * comp["sig_x"], f_gg * comp["fro_x"],
        float((r_gg * comp["rows_x"]).sum()))
    b["grad_w_up"] = pack(
        f_gu * comp["sig_x"], f_gu * comp["fro_x"],
        float((r_gu * comp["rows_x"]).sum()))
    b["grad_w_down"] = pack(
        comp["sig_gy"] * f_hm, comp["fro_gy"] * f_hm,
        float((comp["rows_gy"] * r_hm).sum()))
    return b


def calibrate_and_validate(model, caps, args, device, consts):
    """Stage 3 (all layers, all bits, calib batches) + Stage 4 (VAL_LAYERS)."""
    L1, L2 = consts["L1_sup_abs_silu_prime"], consts["L2_sup_abs_silu_second"]
    n_layers = len(model.model.layers)
    calib_b = list(range(args.n_calib))
    rho = {}            # rho[L][bits][tensor] = [per-batch rho]
    bounds_abs = {}     # bounds_abs[L][bits][tensor] = [per-batch op bound]
    norms_T = {}        # norms_T[L][tensor] = [per-batch ||T||_F]
    rho_x = {}          # rho_x[L][bits] = [per-batch bound_grad_x / ||grad_v||]
    grad_norms = {}     # grad_norms[L] = per-batch {v, u, gy} norms
    companions_log = {}
    validation = []     # Stage 4 records
    env_checks = []     # per-entry envelope validity records
    spec_pi_log = []    # power-iteration vs eigvalsh diagnostic

    for L in range(n_layers):
        t0 = time.time()
        P = layer_params(model, L)
        P64 = {k: (lift(v, device) if torch.is_tensor(v) else v)
               for k, v in P.items()}
        weff_g = P64["w_gate"] + P64["s_gate"] * (P64["b_gate"] @ P64["a_gate"])
        weff_u = P64["w_up"] + P64["s_up"] * (P64["b_up"] @ P64["a_up"])
        layer_comp = {
            "sig_weff_g": spec_norm_ub(weff_g), "fro_weff_g": float(weff_g.norm()),
            "sig_weff_u": spec_norm_ub(weff_u), "fro_weff_u": float(weff_u.norm()),
            "sig_b_gate": spec_norm_ub(P64["b_gate"]),
            "fro_b_gate": float(P64["b_gate"].norm()),
            "sig_b_up": spec_norm_ub(P64["b_up"]),
            "fro_b_up": float(P64["b_up"].norm()),
        }
        spec_pi_log.append({
            "layer": L, "matrix": "weff_gate",
            "eigvalsh_ub": layer_comp["sig_weff_g"],
            "power_iter_8": power_iter_est(weff_g, iters=8, seed=L),
            "frobenius": layer_comp["fro_weff_g"]})
        del weff_g, weff_u

        rho[L] = {bits: {t: [] for t in ALL_TENSORS} for bits in args.bits}
        bounds_abs[L] = {bits: {t: [] for t in ALL_TENSORS} for bits in args.bits}
        norms_T[L] = {t: [] for t in ALL_TENSORS}
        rho_x[L] = {bits: [] for bits in args.bits}
        grad_norms[L] = []

        for b in calib_b:
            cell = caps[L][b]
            H = x_dim(cell["x"])
            x16 = cell["x"].to(device).reshape(-1, H)
            gy16 = cell["grad_y"].to(device).reshape(-1, H)
            hg16 = cell["h_gate"].to(device).reshape(x16.shape[0], -1)
            hu16 = cell["h_up"].to(device).reshape(x16.shape[0], -1)
            xa_g16 = F.linear(x16, P["a_gate"])         # bf16, same op as kernel
            xa_u16 = F.linear(x16, P["a_up"])
            hmid16 = triton_silu_mul(hg16, hu16)
            ha_d16 = F.linear(hmid16, P["a_down"])

            x64, gy64 = x16.to(torch.float64), gy16.to(torch.float64)
            hg64, hu64 = hg16.to(torch.float64), hu16.to(torch.float64)
            P64b = dict(P64)
            P64b["xa_g"] = xa_g16.to(torch.float64)
            P64b["xa_u"] = xa_u16.to(torch.float64)
            P64b["ha_d"] = ha_d16.to(torch.float64)

            ghm64 = gy64 @ P64["w_down"] \
                + P64["s_down"] * ((gy64 @ P64["b_down"]) @ P64["a_down"])
            T_exact, _, _, _ = ffn_backward_fp64(x64, gy64, hg64, hu64, P64b,
                                                 ghm64)
            for t in ALL_TENSORS:
                norms_T[L][t].append(float(T_exact[t].norm()))

            gyBd64 = gy64 @ P64["b_down"]
            comp = dict(layer_comp)
            comp.update({
                "sig_x": spec_norm_ub(x64), "fro_x": float(x64.norm()),
                "rows_x": row_norms(x64),
                "sig_gy": spec_norm_ub(gy64), "fro_gy": float(gy64.norm()),
                "rows_gy": row_norms(gy64),
                "sig_xa_g": spec_norm_ub(P64b["xa_g"]),
                "fro_xa_g": float(P64b["xa_g"].norm()),
                "rows_xa_g": row_norms(P64b["xa_g"]),
                "sig_xa_u": spec_norm_ub(P64b["xa_u"]),
                "fro_xa_u": float(P64b["xa_u"].norm()),
                "rows_xa_u": row_norms(P64b["xa_u"]),
                "sig_gyBd": spec_norm_ub(gyBd64),
                "fro_gyBd": float(gyBd64.norm()),
                "rows_gyBd": row_norms(gyBd64),
            })
            if b == 0:
                companions_log[L] = {k: v for k, v in comp.items()
                                     if not torch.is_tensor(v)}
            v_norm = float(lift(cell["grad_v"], device).norm())
            grad_norms[L].append({
                "v_mlp_in": v_norm,
                "u_block_in": float(lift(cell["grad_u"], device).norm()),
                "gy_mlp_out": float(gy64.norm()),
            })

            g_abs = ghm64.abs()
            for bits in args.bits:
                hg_hat, env_g = quant_envelope(hg16, bits)
                hu_hat, env_u = quant_envelope(hu16, bits)
                hg_hat64 = hg_hat.to(torch.float64)
                hu_hat64 = hu_hat.to(torch.float64)
                E_gg, E_gu, E_hm = backward_envelopes(
                    g_abs, hg_hat64, hu_hat64, env_g, env_u, L1, L2)
                bnd = tensor_bounds(E_gg, E_gu, E_hm, comp, P64)
                for t in ALL_TENSORS:
                    bound_op = bnd[t]["op"]
                    denom = norms_T[L][t][b]
                    bounds_abs[L][bits][t].append(bound_op)
                    rho[L][bits][t].append(bound_op / denom if denom > 0 else 0.0)
                rho_x[L][bits].append(bnd["grad_x"]["op"] / v_norm)

                if L in args.val_layers:
                    T_q, phi_g_q, phi_u_q, hmid_q = ffn_backward_fp64(
                        x64, gy64, hg_hat64, hu_hat64, P64b, ghm64)
                    _, phi_g_e, phi_u_e, hmid_e = ffn_backward_fp64(
                        x64, gy64, hg64, hu64, P64b, ghm64)
                    for name, dE, dM in (
                            ("E_gg", E_gg, (phi_g_q - phi_g_e).abs()),
                            ("E_gu", E_gu, (phi_u_q - phi_u_e).abs()),
                            ("E_hm", E_hm, (hmid_q - hmid_e).abs())):
                        slack = float((dM - dE * (1 + 1e-12)).max())
                        env_checks.append({
                            "layer": L, "batch": b, "bits": bits, "env": name,
                            "holds": slack <= 0.0, "max_violation": max(slack, 0.0),
                            "max_ratio": float(
                                (dM / dE.clamp(min=1e-300)).max())})
                    for t in ALL_TENSORS:
                        dT = float((T_q[t] - T_exact[t]).norm())
                        denom = norms_T[L][t][b]
                        rec = {
                            "layer": L, "batch": b, "bits": bits, "tensor": t,
                            "measured_abs": dT,
                            "measured_rel": dT / denom if denom > 0 else 0.0,
                            "bound_spec": bnd[t]["spec"],
                            "bound_frob": bnd[t]["frob"],
                            "bound_row": bnd[t]["row"],
                            "bound_op": bnd[t]["op"],
                            "rho": rho[L][bits][t][b],
                        }
                        rec["holds"] = rec["bound_op"] >= dT * (1 - 1e-9)
                        rec["tightness"] = (rec["bound_op"] / dT
                                            if dT > 0 else float("nan"))
                        validation.append(rec)
                    del T_q, phi_g_q, phi_u_q, hmid_q, phi_g_e, phi_u_e, hmid_e
                del E_gg, E_gu, E_hm, env_g, env_u, hg_hat64, hu_hat64
            del x64, gy64, hg64, hu64, ghm64, g_abs, T_exact, comp
            torch.cuda.empty_cache()
        del P64
        torch.cuda.empty_cache()
        log(f"  layer {L:2d} calibrated in {time.time() - t0:.1f}s")
    return {"rho": rho, "bounds_abs": bounds_abs, "norms_T": norms_T,
            "rho_x": rho_x, "grad_norms": grad_norms,
            "companions": companions_log, "validation": validation,
            "env_checks": env_checks, "spec_pi_log": spec_pi_log}


# ===========================================================================
# Stage 5/6/7 -- end-to-end arms (real v3 path + generic-bits replica)
# ===========================================================================

class CertV3Function(torch.autograd.Function):
    """Bit-generalized replica of flashffn.FlashFFNv3Function, quantized cache
    modes only. bits in {4,8}: REAL v3 quantizer; {2,3}: gated mirror.
    Backward math and Triton kernels identical to FlashFFNv3Function
    (verified bitwise at int4/int8 by replica_gate)."""

    @staticmethod
    def forward(ctx, x, w_gate, a_gate, b_gate, w_up, a_up, b_up,
                w_down, a_down, b_down, s_gate=0.0, s_up=0.0, s_down=0.0,
                bits=4, group_size=GROUP):
        _CERT_COUNTERS["forward"] += 1
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        x_2d = x.reshape(-1, hidden_dim)
        h_gate = F.linear(x_2d, w_gate)
        xa_g = None
        if a_gate is not None:
            xa_g = F.linear(x_2d, a_gate)
            h_gate = h_gate + s_gate * F.linear(xa_g, b_gate)
        h_up = F.linear(x_2d, w_up)
        xa_u = None
        if a_up is not None:
            xa_u = F.linear(x_2d, a_up)
            h_up = h_up + s_up * F.linear(xa_u, b_up)
        h_mid = triton_silu_mul(h_gate, h_up)
        y = F.linear(h_mid, w_down)
        ha_d = None
        if a_down is not None:
            ha_d = F.linear(h_mid, a_down)
            y = y + s_down * F.linear(ha_d, b_down)
        if bits in (4, 8):
            c_gate, qs_gate = v3_quantize_group_absmax(h_gate, bits, group_size)
            c_up, qs_up = v3_quantize_group_absmax(h_up, bits, group_size)
        else:
            c_gate, qs_gate = quant_any(h_gate, bits, group_size)
            c_up, qs_up = quant_any(h_up, bits, group_size)
        del h_gate, h_up, h_mid
        ctx.save_for_backward(x, w_gate, a_gate, b_gate, w_up, a_up, b_up,
                              w_down, a_down, b_down, xa_g, xa_u, ha_d,
                              c_gate, qs_gate, c_up, qs_up)
        ctx.s_gate, ctx.s_up, ctx.s_down = s_gate, s_up, s_down
        ctx.bits = bits
        ctx.group_size = group_size
        ctx.orig_shape = orig_shape
        ctx.in_dtype = x.dtype
        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        _CERT_COUNTERS["backward"] += 1
        (x, w_gate, a_gate, b_gate, w_up, a_up, b_up,
         w_down, a_down, b_down, xa_g, xa_u, ha_d,
         c_gate, qs_gate, c_up, qs_up) = ctx.saved_tensors
        s_gate, s_up, s_down = ctx.s_gate, ctx.s_up, ctx.s_down
        bits = ctx.bits
        orig_shape = ctx.orig_shape
        hidden_dim = orig_shape[-1]
        ng = ctx.needs_input_grad
        (need_x, need_wg, need_ag, need_bg, need_wu, need_au, need_bu,
         need_wd, need_ad, need_bd) = ng[:10]
        gy = grad_y.reshape(-1, hidden_dim)
        x_2d = x.reshape(-1, hidden_dim)
        need_gate_path = (need_x or need_wg or need_ag or need_bg
                          or need_wu or need_au or need_bu)
        need_h_mid = need_wd or (a_down is not None and need_ad)
        if bits in (4, 8):
            hg = v3_dequantize_group_absmax(c_gate, qs_gate, bits,
                                            ctx.group_size, ctx.in_dtype)
            hu = v3_dequantize_group_absmax(c_up, qs_up, bits,
                                            ctx.group_size, ctx.in_dtype)
        else:
            hg = deq_any(c_gate, qs_gate, ctx.group_size, ctx.in_dtype)
            hu = deq_any(c_up, qs_up, ctx.group_size, ctx.in_dtype)
        grad_w_down = grad_a_down = grad_b_down = None
        gyB_d = None
        if a_down is not None and (need_gate_path or need_ad):
            gyB_d = gy @ b_down
        if need_h_mid:
            h_mid_hat = triton_silu_mul(hg, hu)
            if need_wd:
                grad_w_down = gy.t() @ h_mid_hat
            if need_ad:
                grad_a_down = s_down * (gyB_d.t() @ h_mid_hat)
            del h_mid_hat
        if need_bd:
            grad_b_down = s_down * (gy.t() @ ha_d)
        grad_x = None
        grad_w_gate = grad_a_gate = grad_b_gate = None
        grad_w_up = grad_a_up = grad_b_up = None
        if need_gate_path:
            grad_h_mid = gy @ w_down
            if a_down is not None:
                grad_h_mid = grad_h_mid + s_down * (gyB_d @ a_down)
            grad_h_gate, grad_h_up = flashffn.triton_swiglu_backward(
                grad_h_mid, hg, hu)
            del grad_h_mid
            ghgB = None
            if a_gate is not None and (need_x or need_ag):
                ghgB = grad_h_gate @ b_gate
            ghuB = None
            if a_up is not None and (need_x or need_au):
                ghuB = grad_h_up @ b_up
            if need_x:
                grad_x = grad_h_gate @ w_gate
                if a_gate is not None:
                    grad_x = grad_x + s_gate * (ghgB @ a_gate)
                grad_x = grad_x + grad_h_up @ w_up
                if a_up is not None:
                    grad_x = grad_x + s_up * (ghuB @ a_up)
            if need_bg:
                grad_b_gate = s_gate * (grad_h_gate.t() @ xa_g)
            if need_ag:
                grad_a_gate = s_gate * (ghgB.t() @ x_2d)
            if need_bu:
                grad_b_up = s_up * (grad_h_up.t() @ xa_u)
            if need_au:
                grad_a_up = s_up * (ghuB.t() @ x_2d)
            if need_wg:
                grad_w_gate = grad_h_gate.t() @ x_2d
            if need_wu:
                grad_w_up = grad_h_up.t() @ x_2d
        if need_x and len(orig_shape) == 3:
            grad_x = grad_x.view(orig_shape)
        return (grad_x,
                grad_w_gate, grad_a_gate, grad_b_gate,
                grad_w_up, grad_a_up, grad_b_up,
                grad_w_down, grad_a_down, grad_b_down,
                None, None, None, None, None)


def make_replica_forward(mlp, bits, group_size=GROUP):
    def fwd(x):
        wg, ag, bg, sg = flashffn._v3_extract_proj(mlp.gate_proj)
        wu, au, bu, su = flashffn._v3_extract_proj(mlp.up_proj)
        wd, ad, bd, sd = flashffn._v3_extract_proj(mlp.down_proj)
        dt = x.dtype

        def cast(t):
            return t if (t is None or t.dtype == dt) else t.to(dt)

        return CertV3Function.apply(
            x, cast(wg), cast(ag), cast(bg), cast(wu), cast(au), cast(bu),
            cast(wd), cast(ad), cast(bd), sg, su, sd, bits, group_size)
    return fwd


def patch_model(model, modes):
    """modes: list per layer in {'bf16','int4','int8','recompute',
    'q2','q3','q4','q8'} (q* = generic-bits replica)."""
    for L, layer in enumerate(model.model.layers):
        m = modes[L]
        if m in ("bf16", "int4", "int8", "recompute"):
            layer.mlp.forward = make_v3_forward(layer.mlp, m, GROUP)
        elif m.startswith("q"):
            layer.mlp.forward = make_replica_forward(layer.mlp, int(m[1:]))
        else:
            raise ValueError(m)


def unpatch_model(model):
    for layer in model.model.layers:
        if "forward" in layer.mlp.__dict__:
            del layer.mlp.__dict__["forward"]


def adapter_param_index(model):
    idx = []
    for L, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for key, proj in (("gate", mlp.gate_proj), ("up", mlp.up_proj),
                          ("down", mlp.down_proj)):
            name = proj.active_adapters[0]
            idx.append((L, f"grad_a_{key}", proj.lora_A[name].weight))
            idx.append((L, f"grad_b_{key}", proj.lora_B[name].weight))
    return idx


def run_arm(model, modes, batches, device, use_v3_counters, micro_bs):
    """Forward+backward per batch (micro-chunked exactly like capture) with
    the given per-layer cache modes; returns (grads[b][(L, tensor)] cpu fp32,
    losses[b] = list of per-chunk losses)."""
    patch_model(model, modes)
    v3_reset_counters()
    _CERT_COUNTERS["forward"] = _CERT_COUNTERS["backward"] = 0
    idx = adapter_param_index(model)
    grads, losses = [], []
    n_chunk_calls = 0
    for b in range(batches.shape[0]):
        model.zero_grad(set_to_none=True)
        chunk_losses = []
        for chunk, w in iter_chunks(batches[b], micro_bs):
            ids = chunk.to(device)
            out = model(input_ids=ids, labels=ids)
            chunk_losses.append(float(out.loss.detach()))
            (out.loss * w).backward()
            n_chunk_calls += 1
            del out
            torch.cuda.empty_cache()
        losses.append(chunk_losses)
        grads.append({(L, t): p.grad.detach().float().cpu()
                      for (L, t, p) in idx})
    model.zero_grad(set_to_none=True)
    unpatch_model(model)
    torch.cuda.empty_cache()
    n_real = sum(m in ("bf16", "int4", "int8", "recompute") for m in modes)
    n_rep = len(modes) - n_real
    if use_v3_counters:
        assert _V3_COUNTERS["backward"] == n_real * n_chunk_calls, \
            "v3 path did not run"
        assert _CERT_COUNTERS["backward"] == n_rep * n_chunk_calls, \
            "replica count off"
    return grads, losses


def compare_grads(grads_a, grads_e):
    """Per-tensor relerr and cosine of arm A vs exact arm E (fp64)."""
    out = {}
    for key in grads_e:
        a = grads_a[key].to(torch.float64).flatten()
        e = grads_e[key].to(torch.float64).flatten()
        d = float((a - e).norm())
        en = float(e.norm())
        cos = float(F.cosine_similarity(a, e, dim=0)) if en > 0 else 1.0
        out[key] = {"relerr": d / en if en > 0 else 0.0, "cos": cos}
    return out


def replica_gate(model, batches, device, micro_bs):
    """Replica (generic-bits path) must reproduce the REAL FlashFFNv3 int4/int8
    end-to-end adapter grads bitwise."""
    rep = {}
    one = batches[:1]
    for bits, real_mode in ((4, "int4"), (8, "int8")):
        n_layers = len(model.model.layers)
        g_real, l_real = run_arm(model, [real_mode] * n_layers, one, device,
                                 use_v3_counters=False, micro_bs=micro_bs)
        g_rep, l_rep = run_arm(model, [f"q{bits}"] * n_layers, one, device,
                               use_v3_counters=False, micro_bs=micro_bs)
        bitwise = all(torch.equal(g_real[0][k], g_rep[0][k]) for k in g_real[0])
        max_rel = max(compare_grads(g_rep[0], g_real[0])[k]["relerr"]
                      for k in g_real[0])
        rep[f"int{bits}"] = {"bitwise": bitwise, "max_relerr_vs_real": max_rel,
                             "loss_equal": l_real[0] == l_rep[0]}
        assert max_rel < 1e-6, f"replica mismatch at int{bits}: {max_rel}"
    return rep


# ===========================================================================
# Composition + allocation
# ===========================================================================

def agg_rho(cal, args, reducer=max):
    """rho_local[L][option][tensor], rho_x[L][option] aggregated over calib
    batches (max = conservative). option in {'recompute','int2','int3',...}."""
    n_layers = len(cal["rho"])
    opts = ["recompute"] + [f"int{b}" for b in args.bits]
    rho_local, rho_x = {}, {}
    for L in range(n_layers):
        rho_local[L], rho_x[L] = {}, {}
        for o in opts:
            if o == "recompute":
                rho_local[L][o] = {t: 0.0 for t in ADAPTER_TENSORS}
                rho_x[L][o] = 0.0
            else:
                bits = int(o[3:])
                rho_local[L][o] = {t: reducer(cal["rho"][L][bits][t])
                                   for t in ADAPTER_TENSORS}
                rho_x[L][o] = reducer(cal["rho_x"][L][bits])
    return rho_local, rho_x


def composed_rhos(assign, rho_local, rho_x, mode="first_order"):
    """Composed per-(layer, tensor) bound under the directional-transport
    convention (see writeup): upstream relative errors ADD.

    mode="first_order": rho_local + sum_{l'>l} rho_x  (second-order
      envelope-perturbation terms dropped -- they are second order in the
      TRUE errors; multiplying bound-by-bound instead is mode="worst_case").
    mode="worst_case": (rho_local + sum) * prod(1 + rho_x). Fully
      conservative but explodes multiplicatively once any rho_x >~ 1
      (vacuous at int4 over 22 layers -- reported, not used for allocation).
    """
    n_layers = len(assign)
    rx = [rho_x[L][assign[L]] for L in range(n_layers)]
    out = {}
    suffix_sum = 0.0
    suffix_prod = 1.0
    for L in range(n_layers - 1, -1, -1):
        infl = suffix_prod if mode == "worst_case" else 1.0
        for t in ADAPTER_TENSORS:
            out[(L, t)] = (rho_local[L][assign[L]][t] + suffix_sum) * infl
        suffix_sum += rx[L]
        suffix_prod *= (1.0 + rx[L])
    return out


def cache_bytes(option, n_tok, d_int):
    """Stored cache bytes per layer for the (h_gate, h_up) pair (payload +
    fp16 group scales). recompute stores neither. x ([N, D_hid] bf16) is
    stored in EVERY mode and excluded (constant)."""
    if option == "recompute":
        return 0
    bits = int(option[3:])
    payload = 2 * n_tok * d_int * bits / 8
    scales = 2 * n_tok * (d_int // GROUP) * 2
    return int(payload + scales)


def choose_bits(eps, rho_local, rho_x, n_layers, bits_opts, n_tok, d_int):
    """Greedy byte-minimal allocation s.t. max composed rho <= eps.

    Quantized options only (recompute would degenerate the byte objective:
    0 bytes AND rho=0). If even all-int8 is infeasible, recompute layers are
    added one at a time (lexicographic: minimize #recompute, then bytes)."""
    ladder = [f"int{b}" for b in bits_opts]               # ascending bytes

    def max_rho(assign):
        return max(composed_rhos(assign, rho_local, rho_x).values())

    recompute_set = set()
    while True:
        top = ["int8" if L not in recompute_set else "recompute"
               for L in range(n_layers)]
        if max_rho(top) <= eps:
            break
        if len(recompute_set) == n_layers:
            return None
        best_L, best_val = None, float("inf")
        for L in range(n_layers):
            if L in recompute_set:
                continue
            cand = list(top)
            cand[L] = "recompute"
            v = max_rho(cand)
            if v < best_val:
                best_val, best_L = v, L
        recompute_set.add(best_L)

    assign = ["recompute" if L in recompute_set else ladder[0]
              for L in range(n_layers)]
    while True:
        cur = max_rho(assign)
        if cur <= eps:
            break
        best = None                                       # (score, L, new_opt)
        for L in range(n_layers):
            if L in recompute_set or assign[L] == ladder[-1]:
                continue
            nxt = ladder[ladder.index(assign[L]) + 1]
            cand = list(assign)
            cand[L] = nxt
            new_max = max_rho(cand)
            dbytes = (cache_bytes(nxt, n_tok, d_int)
                      - cache_bytes(assign[L], n_tok, d_int))
            score = (cur - new_max) / max(dbytes, 1)
            if best is None or score > best[0]:
                best = (score, L, nxt, new_max)
        if best is None:
            return None                                   # unreachable: top feasible
        assign[best[1]] = best[2]

    changed = True                                        # downgrade pass
    while changed:
        changed = False
        order = sorted(range(n_layers),
                       key=lambda L: -cache_bytes(assign[L], n_tok, d_int))
        for L in order:
            if L in recompute_set or assign[L] == ladder[0]:
                continue
            prev = ladder[ladder.index(assign[L]) - 1]
            cand = list(assign)
            cand[L] = prev
            if max_rho(cand) <= eps:
                assign[L] = prev
                changed = True
    return assign


# ===========================================================================
# main
# ===========================================================================

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def save_json(out_dir, name, obj):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    log(f"  [saved {path}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--n_batches", type=int, default=3,
                    help="total batches; first n_calib calibrate, rest held out")
    ap.add_argument("--n_calib", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--micro_bs", type=int, default=2,
                    help="micro-chunk size for fwd/bwd (memory)")
    ap.add_argument("--out_dir", default="results/v3_bounds")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.bits = BITS_ALL
    args.val_layers = VAL_LAYERS
    args.eps_list = (0.02, 0.05, 0.10)
    # Achievable-certificate sweep: deterministic worst-case bounds composed
    # over 22 layers cannot reach 0.02-0.10 with quantization only (see
    # Stage 6 floor log); these levels exercise meaningful MIXED allocations.
    args.eps_extended = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
    if args.quick:
        args.n_batches, args.n_calib = 1, 1
        args.bits = (4, 8)
        args.val_layers = (11,)
        args.eps_list = (0.05,)
        args.eps_extended = (8.0,)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    log("Stage 0: SiLU Lipschitz constants ...")
    consts = compute_silu_constants()
    log(f"  L1 = sup|silu'| = {consts['L1_exact_value']:.10f} (z* = "
        f"{consts['z_star']:.6f}), L2 = sup|silu''| = 0.5 (exact)")

    log("Stage 2: building model + capture (capture before mirror gate; the "
        "gate also wants a real tensor) ...")
    model, tok = build_model(args.seed, device)
    n_layers = len(model.model.layers)
    d_int = model.config.intermediate_size
    d_hid = model.config.hidden_size
    n_tok = args.batch_size * args.seq_len
    batches = build_batches(tok, args.n_batches, args.batch_size, args.seq_len)
    caps, cap_losses, cap_verif = capture(model, batches, device,
                                          args.micro_bs)
    log(f"  losses {[round(l, 4) for l in cap_losses]}; recompute verification "
        f"{cap_verif}")

    log("Stage 1: quantizer mirror + envelope gate ...")
    mirror_rep = mirror_gate(caps, device)
    log(f"  {mirror_rep}")

    log("Stage 3+4: calibration bounds + local validation ...")
    cal = calibrate_and_validate(model, caps, args, device, consts)

    val = cal["validation"]
    n_holds = sum(r["holds"] for r in val)
    env_holds = sum(r["holds"] for r in cal["env_checks"])
    log(f"  LOCAL VALIDITY: {n_holds}/{len(val)} tensor cases hold; "
        f"{env_holds}/{len(cal['env_checks'])} envelope checks hold")
    tight = {}
    for t in ALL_TENSORS:
        ratios = sorted(r["tightness"] for r in val
                        if r["tensor"] == t and r["measured_abs"] > 0)
        if ratios:
            tight[t] = {"median": ratios[len(ratios) // 2],
                        "min": ratios[0], "max": ratios[-1]}
    for t, d in tight.items():
        log(f"    tightness {t:13s}: median {d['median']:9.1f}x  "
            f"min {d['min']:7.1f}x  max {d['max']:9.1f}x")

    save_json(args.out_dir, "validation.json", {
        "constants": consts, "mirror_gate": mirror_rep,
        "capture_verification": cap_verif,
        "n_cases": len(val), "n_holds": n_holds,
        "env_checks_total": len(cal["env_checks"]),
        "env_checks_hold": env_holds,
        "tightness_by_tensor": tight, "cases": val,
        "env_checks": cal["env_checks"]})
    save_json(args.out_dir, "rho_tables.json", {
        "meta": {"model": MODEL_NAME, "n_layers": n_layers,
                 "rank": RANK, "alpha": ALPHA, "group": GROUP,
                 "n_tok": n_tok, "bits": list(args.bits),
                 "seed": args.seed, "losses": cap_losses},
        "rho": cal["rho"], "rho_x": cal["rho_x"],
        "bounds_abs": cal["bounds_abs"], "norms_T": cal["norms_T"],
        "grad_norms": cal["grad_norms"], "companions": cal["companions"],
        "spec_powiter_diagnostic": cal["spec_pi_log"]})
    if n_holds != len(val) or env_holds != len(cal["env_checks"]):
        log("!! LOCAL VALIDITY FAILED -- aborting before e2e stages")
        sys.exit(1)

    log("Stage 5a: replica gate (generic-bits path vs real v3, e2e) ...")
    rep_gate = replica_gate(model, batches, device, args.micro_bs)
    log(f"  {rep_gate}")

    log("Stage 5b: e2e arms (exact + uniform int2/3/4/8) ...")
    arms = {"exact": ["bf16"] * n_layers}
    for b in args.bits:
        arms[f"uniform_int{b}"] = ([f"int{b}"] * n_layers if b in (4, 8)
                                   else [f"q{b}"] * n_layers)
    arm_grads, arm_losses, arm_metrics = {}, {}, {}
    for name, modes in arms.items():
        g, l = run_arm(model, modes, batches, device, use_v3_counters=True,
                       micro_bs=args.micro_bs)
        arm_grads[name], arm_losses[name] = g, l
        log(f"  arm {name}: losses {[[round(x, 4) for x in lb] for lb in l]}")
    for name in arms:
        if name == "exact":
            continue
        assert arm_losses[name] == arm_losses["exact"], \
            f"forward changed in arm {name} (must be cache-only)"
        arm_metrics[name] = [compare_grads(arm_grads[name][b],
                                           arm_grads["exact"][b])
                             for b in range(args.n_batches)]

    log("Stage 5c: depth-composition check (uniform int4, real kernels) ...")
    rho_local, rho_x = agg_rho(cal, args)
    composition = []
    if 4 in args.bits:
        cr_fo, cr_wc = [], []
        for b in range(args.n_calib):
            rl = {L: {"int4": {t: cal["rho"][L][4][t][b]
                               for t in ADAPTER_TENSORS}} for L in range(n_layers)}
            rx = {L: {"int4": cal["rho_x"][L][4][b]} for L in range(n_layers)}
            cr_fo.append(composed_rhos(["int4"] * n_layers, rl, rx,
                                       "first_order"))
            cr_wc.append(composed_rhos(["int4"] * n_layers, rl, rx,
                                       "worst_case"))
        for L in args.val_layers:
            for t in ADAPTER_TENSORS:
                for b in range(args.n_calib):
                    meas = arm_metrics["uniform_int4"][b][(L, t)]["relerr"]
                    p_fo, p_wc = cr_fo[b][(L, t)], cr_wc[b][(L, t)]
                    composition.append({
                        "layer": L, "tensor": t, "batch": b,
                        "measured_rel": meas,
                        "composed_first_order": p_fo,
                        "composed_worst_case": p_wc,
                        "holds_first_order": p_fo >= meas,
                        "holds_worst_case": p_wc >= meas,
                        "tightness_first_order":
                            p_fo / meas if meas > 0 else float("nan")})
        c_fo = sum(r["holds_first_order"] for r in composition)
        c_wc = sum(r["holds_worst_case"] for r in composition)
        log(f"  COMPOSITION: first-order {c_fo}/{len(composition)} hold, "
            f"worst-case {c_wc}/{len(composition)} hold "
            f"(semi-empirical: directional-transport assumption)")
    save_json(args.out_dir, "composition.json", {
        "replica_gate": rep_gate, "cases": composition,
        "grad_norm_chain": cal["grad_norms"]})

    log("Stage 6: certified allocation (constraint: FIRST-ORDER composed "
        "bound; worst-case reported alongside) ...")
    # Certifiable floor: best possible composed bound with quantization only.
    floor_all_int8 = max(composed_rhos(["int8"] * n_layers, rho_local,
                                       rho_x).values())
    log(f"  quantized-only certifiable floor (all-int8, first-order composed):"
        f" {floor_all_int8:.2f} -> requested eps in {list(args.eps_list)} "
        "force recompute (honest finding: deterministic worst-case bounds at "
        "depth 22 cannot certify <=0.1 without recompute)")
    all_eps = sorted(set(list(args.eps_list) + list(args.eps_extended)))
    allocations = {}
    for eps in all_eps:
        assign = choose_bits(eps, rho_local, rho_x, n_layers, args.bits,
                             n_tok, d_int)
        if assign is None:
            allocations[str(eps)] = {"feasible": False}
            log(f"  eps={eps}: INFEASIBLE even with recompute everywhere")
            continue
        cr = composed_rhos(assign, rho_local, rho_x)
        cr_wc = composed_rhos(assign, rho_local, rho_x, "worst_case")
        total = sum(cache_bytes(o, n_tok, d_int) for o in assign)
        allocations[str(eps)] = {
            "feasible": True, "assign": assign,
            "requested_target": eps in args.eps_list,
            "predicted_max_composed_rho": max(cr.values()),
            "predicted_max_composed_rho_worst_case": max(cr_wc.values()),
            "cache_bytes_total": total,
            "n_recompute_layers": sum(o == "recompute" for o in assign),
        }
        log(f"  eps={eps}: " + " ".join(
            o.replace("recompute", "RC").replace("int", "") for o in assign))
        log(f"    bytes {total / 2**20:.1f} MiB, n_recompute "
            f"{sum(o == 'recompute' for o in assign)}, predicted max rho "
            f"{max(cr.values()):.4f}")
    uniform_bytes = {f"int{b}": cache_bytes(f"int{b}", n_tok, d_int) * n_layers
                     for b in args.bits}
    bf16_bytes = 2 * n_tok * d_int * 2 * n_layers
    log(f"  uniform bytes/batch: " + ", ".join(
        f"{k}={v / 2**20:.1f}MiB" for k, v in uniform_bytes.items())
        + f", bf16={bf16_bytes / 2**20:.1f}MiB")

    log("Stage 7: frontier (measured fidelity per arm) ...")
    frontier = []
    cert_arms = {}
    seen_assigns = {}
    for eps in all_eps:
        a = allocations[str(eps)]
        if not a.get("feasible"):
            continue
        key = tuple(a["assign"])
        if key in seen_assigns:           # identical allocation, reuse arm
            a["same_as"] = seen_assigns[key]
            continue
        seen_assigns[key] = f"certified_eps{eps}"
        modes = []
        for o in a["assign"]:
            if o == "recompute":
                modes.append("recompute")
            elif o in ("int4", "int8"):
                modes.append(o)
            else:
                modes.append(f"q{int(o[3:])}")
        cert_arms[f"certified_eps{eps}"] = (modes, a)
    for name, (modes, a) in cert_arms.items():
        g, l = run_arm(model, modes, batches, device, use_v3_counters=True,
                       micro_bs=args.micro_bs)
        assert l == arm_losses["exact"]
        arm_metrics[name] = [compare_grads(g[b], arm_grads["exact"][b])
                             for b in range(args.n_batches)]

    def arm_point(name, bytes_total, predicted, certified, eps=None):
        ms = arm_metrics[name]
        calib = [m for b, m in enumerate(ms) if b < args.n_calib]
        held = [m for b, m in enumerate(ms) if b >= args.n_calib]

        def stats(batch_list):
            if not batch_list:
                return None
            rel = [v["relerr"] for m in batch_list for v in m.values()]
            cos = [v["cos"] for m in batch_list for v in m.values()]
            rel_s = sorted(rel)
            return {"max_relerr": max(rel), "median_relerr":
                    rel_s[len(rel_s) // 2], "min_cos": min(cos)}
        pt = {"name": name, "cache_bytes_total": bytes_total,
              "bytes_MiB": bytes_total / 2**20,
              "predicted_max_composed_rho": predicted,
              "certified": certified, "eps": eps,
              "measured_calib": stats(calib),
              "measured_heldout": stats(held)}
        if certified and eps is not None and pt["measured_calib"]:
            pt["promise_held_calib"] = pt["measured_calib"]["max_relerr"] <= eps
            if pt["measured_heldout"]:
                pt["promise_held_heldout"] = \
                    pt["measured_heldout"]["max_relerr"] <= eps
        return pt

    for b in args.bits:
        nm = f"uniform_int{b}"
        assign_u = [f"int{b}"] * n_layers
        pred = max(composed_rhos(assign_u, rho_local, rho_x).values())
        pt = arm_point(nm, uniform_bytes[f"int{b}"], pred, False)
        pt["predicted_max_composed_rho_worst_case"] = max(
            composed_rhos(assign_u, rho_local, rho_x, "worst_case").values())
        frontier.append(pt)
    for name, (modes, a) in cert_arms.items():
        frontier.append(arm_point(name, a["cache_bytes_total"],
                                  a["predicted_max_composed_rho"], True,
                                  eps=float(name.split("eps")[1])))
    frontier.sort(key=lambda p: p["cache_bytes_total"])
    for p in frontier:
        mc = p["measured_calib"]
        log(f"  {p['name']:22s} {p['bytes_MiB']:8.1f} MiB  "
            f"pred<= {p['predicted_max_composed_rho']:8.4f}  "
            f"meas max-rel {mc['max_relerr']:.5f}  min-cos {mc['min_cos']:.6f}"
            + (f"  promise_held={p.get('promise_held_calib')}"
               if p["certified"] else ""))

    save_json(args.out_dir, "allocation.json", {
        "eps_list": list(args.eps_list),
        "eps_extended": list(args.eps_extended),
        "composed_mode_for_constraint": "first_order",
        "quantized_only_floor_all_int8": floor_all_int8,
        "allocations": allocations,
        "uniform_bytes": uniform_bytes, "bf16_cache_bytes": bf16_bytes,
        "n_tok": n_tok, "d_int": d_int, "d_hid": d_hid,
        "note": "bytes = (h_gate,h_up) cache per layer-batch of N=2048 tokens "
                "(payload + fp16 scales); x excluded (stored in all modes)"})
    save_json(args.out_dir, "frontier.json", {
        "points": frontier,
        "arm_losses": arm_losses,
        "note": "measured vs exact-arm (bf16 cache) adapter grads, real "
                "v3 kernels; calib batches 0..n_calib-1, rest held out"})
    save_json(args.out_dir, "meta.json", {
        "args": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in vars(args).items()},
        "model": MODEL_NAME, "n_layers": n_layers,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "elapsed_sec": time.time() - t_start})
    log(f"Done in {time.time() - t_start:.0f}s.")


if __name__ == "__main__":
    main()
