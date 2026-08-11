"""Gate 1 for the HyC-LoRA port: Triton kernel correctness under triton 3.6 / torch 2.10.

Upstream HyC-LoRA pins triton==2.3.0 / torch==2.3.0. We run triton 3.6 / torch 2.10, so every
vendored kernel is re-validated against an independent PyTorch reference before any training
number is trusted. A silently-wrong kernel here would surface as an "anomalous result" much
later, so each kernel gets an explicit numeric gate.

Usage:  python src/verify_hyclora_kernels.py [--device cuda:0]
Exit 0 iff every gate passes.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyclora.operators.rmsnorm_kernels import rmsnorm_forward, rmsnorm_backward
from hyclora.operators.silu_kernels import silu_backward, silu_hadamard
from hyclora.operators.rope_kernels import rope_forward, rope_backward, calculate_settings
from hyclora.operators.softmax_kernels import softmax_backward
from hyclora.operators.compress_function import (
    compression_pack_channel_base,
    compression_pack_quant_base,
    compression_pack_softmax_base,
    outlier_addition_fuse_decompression_dequantization,
    decompression_dequantization,
)

RESULTS = []


def record(name, ok, detail):
    RESULTS.append({"gate": name, "pass": bool(ok), **detail})
    status = "\033[1;32mPASS\033[0m" if ok else "\033[1;31mFAIL\033[0m"
    print(f"[{status}] {name}: {detail}")
    return ok


def rel_err(a, b):
    a, b = a.float(), b.float()
    denom = b.abs().max().clamp_min(1e-12)
    return ((a - b).abs().max() / denom).item()


def cos_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# --------------------------------------------------------------------------------------
# K1: RMSNorm forward/backward
# --------------------------------------------------------------------------------------
def gate_rmsnorm(dev, dtype):
    torch.manual_seed(0)
    B, S, D = 2, 128, 2048
    eps = 1e-5
    x = torch.randn(B, S, D, device=dev, dtype=dtype)
    w = torch.randn(D, device=dev, dtype=dtype)

    # signature order is (y, mean, rstd, BLOCK_SIZE, num_warps) -- backward takes them swapped
    y, mean, rstd, BLOCK_SIZE, num_warps = rmsnorm_forward(x, w, eps=eps)

    # independent reference
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()
    fwd_err = rel_err(y, ref.to(dtype))

    dy = torch.randn_like(y)
    xr = x.clone().float().requires_grad_(True)
    ref_y = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + eps)) * w.float()
    ref_y.backward(dy.float())

    # NOTE: upstream rmsnorm_backward returns (dx, None) -- it deliberately never produces a
    # norm-weight gradient, because under LoRA the RMSNorm weights are frozen. Only dx is gated.
    dx, dw = rmsnorm_backward(dy, x, w, mean, rstd, True, eps, num_warps, BLOCK_SIZE)
    dx_cos = cos_sim(dx, xr.grad)

    ok = fwd_err < 2e-2 and dx_cos > 0.999 and dw is None
    return record("K1_rmsnorm", ok,
                  {"fwd_rel_err": round(fwd_err, 6), "dx_cos": round(dx_cos, 6),
                   "dw_is_none_by_design": dw is None,
                   "BLOCK_SIZE": BLOCK_SIZE, "num_warps": num_warps})


# --------------------------------------------------------------------------------------
# K2: SiLU-Hadamard forward (h = silu(x) * g) and its backward
# --------------------------------------------------------------------------------------
def gate_silu(dev, dtype):
    torch.manual_seed(1)
    B, S, D = 2, 128, 5632
    x = torch.randn(B, S, D, device=dev, dtype=dtype)
    g = torch.randn(B, S, D, device=dev, dtype=dtype)

    h = silu_hadamard(x, g)
    ref_h = (torch.nn.functional.silu(x.float()) * g.float()).to(dtype)
    fwd_err = rel_err(h, ref_h)

    # silu_backward(x, g) returns d/dx [silu(x)] * g  (the grad wrt the silu input, given g)
    got = silu_backward(x, g)
    xr = x.clone().float().requires_grad_(True)
    ref_out = torch.nn.functional.silu(xr)
    ref_out.backward(g.float())
    bwd_cos = cos_sim(got, xr.grad)

    ok = fwd_err < 2e-2 and bwd_cos > 0.999
    return record("K2_silu_hadamard", ok,
                  {"fwd_rel_err": round(fwd_err, 6), "bwd_cos": round(bwd_cos, 6)})


# --------------------------------------------------------------------------------------
# K3: RoPE forward/backward round-trip
# --------------------------------------------------------------------------------------
def gate_rope(dev, dtype):
    torch.manual_seed(2)
    # upstream rope_forward expects [batch, seq_len, n_heads, head_dim] (call sites pass
    # q.transpose(1,2)) and returns a single tensor, mutating its input in place.
    B, H, S, Dh = 2, 32, 128, 64
    q_bhsd = torch.randn(B, H, S, Dh, device=dev, dtype=dtype)
    # upstream cos/sin layout: [seq_len, head_dim] (position-independent, positions 0..S-1)
    inv = 1.0 / (10000 ** (torch.arange(0, Dh, 2, device=dev).float() / Dh))
    t = torch.arange(S, device=dev).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)

    q_rope = rope_forward(q_bhsd.clone().transpose(1, 2), cos, sin).transpose(1, 2)

    # reference rotate_half RoPE
    def rotate_half(v):
        v1, v2 = v[..., : v.shape[-1] // 2], v[..., v.shape[-1] // 2:]
        return torch.cat((-v2, v1), dim=-1)

    c = cos.float().view(1, 1, S, Dh)
    s = sin.float().view(1, 1, S, Dh)
    ref = (q_bhsd.float() * c + rotate_half(q_bhsd.float()) * s).to(dtype)
    fwd_err = rel_err(q_rope, ref)

    # backward applies the inverse rotation, so fwd->bwd must recover the original tensor.
    # N_GROUPS=128 mirrors the (over-provisioned but harmless) constant used at the call site.
    BLOCK, num_warps = calculate_settings(Dh // 2)
    back = rope_backward(q_rope.clone().transpose(1, 2), cos, sin, 128, BLOCK,
                         num_warps).transpose(1, 2)
    roundtrip = rel_err(back, q_bhsd)

    ok = fwd_err < 3e-2 and roundtrip < 3e-2
    return record("K3_rope", ok,
                  {"fwd_rel_err": round(fwd_err, 6), "roundtrip_rel_err": round(roundtrip, 6)})


# --------------------------------------------------------------------------------------
# K4: softmax backward
# --------------------------------------------------------------------------------------
def gate_softmax(dev, dtype):
    torch.manual_seed(3)
    B, H, S = 2, 8, 256
    logits = torch.randn(B, H, S, S, device=dev, dtype=dtype)
    y = torch.softmax(logits.float(), dim=-1).to(dtype)
    gy = torch.randn_like(y)

    got = softmax_backward(y, gy)

    yr = y.clone().float().requires_grad_(True)
    # reference: dx = y * (gy - sum(gy*y))
    ref = yr * (gy.float() - (gy.float() * yr).sum(-1, keepdim=True))
    c = cos_sim(got, ref)
    ok = c > 0.999
    return record("K4_softmax_backward", ok, {"cos": round(c, 6)})


# --------------------------------------------------------------------------------------
# K5: quantisation codec round-trip (the heart of HyC-LoRA's compression)
# --------------------------------------------------------------------------------------
def gate_codec(dev, dtype):
    torch.manual_seed(4)
    B, S, D = 2, 256, 2048
    x = torch.randn(B, S, D, device=dev, dtype=dtype)

    def reference_codec(t, bits):
        """Independent per-channel symmetric absmax quantise/dequantise, in bf16."""
        flat = t.reshape(-1, t.shape[-1]).float()
        qmax = 2 ** (bits - 1) - 1
        absmax = flat.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)
        scale = absmax / qmax
        codes = torch.clamp(torch.round(flat / scale), -qmax - 1, qmax)
        return (codes * scale).to(t.dtype).reshape(t.shape)

    out = {}
    ok_all = True
    for q_bit in (8, 4, 2):
        static = {"scale": None}
        xq, scale = compression_pack_quant_base(
            x=x.clone(), q_bit=q_bit, q_method="per-channel",
            it_num=0, it_num_thd=5, static_value=static,
        )
        xhat = decompression_dequantization(xq, scale, quantize_bit=q_bit, dtype=dtype)
        xhat = xhat.reshape(x.shape)
        c = cos_sim(xhat, x)
        # Gate against an INDEPENDENT reference codec at the same bit width rather than an
        # arbitrary constant: their codec must be at least as faithful as a textbook
        # per-channel absmax codec (minus a small tolerance for scale/packing differences).
        ref = reference_codec(x, q_bit)
        ref_c = cos_sim(ref, x)
        out[f"q{q_bit}_cos"] = round(c, 5)
        out[f"q{q_bit}_ref_cos"] = round(ref_c, 5)
        out[f"q{q_bit}_dtype"] = str(xq.dtype)
        ok_all &= c >= ref_c - 0.02
        # and the packed payload must actually be uint8-backed (real memory saving)
        ok_all &= xq.dtype == torch.uint8
    # monotonicity: more bits must not be worse
    mono = out["q8_cos"] >= out["q4_cos"] >= out["q2_cos"]
    out["monotonic_in_bits"] = bool(mono)
    ok_all &= mono
    return record("K5_quant_codec", ok_all, out)


# --------------------------------------------------------------------------------------
# K6: channel-outlier codec round-trip (intra-operator hybrid compression)
# --------------------------------------------------------------------------------------
def gate_outlier_codec(dev, dtype):
    torch.manual_seed(5)
    B, S, D = 2, 256, 2048
    x = torch.randn(B, S, D, device=dev, dtype=dtype)
    # plant heavy per-channel outliers, which is what the intra-op path exists to handle
    x[:, :, ::128] *= 25.0

    res = {}
    ok_all = True
    for q_bit in (4, 2):
        for ratio in (0.0, 0.005):
            static = {"outlier_channel_index": None, "scale": None}
            xo, xq, idx, scale = compression_pack_channel_base(
                x=x.clone(), o_ratio=ratio, q_bit=q_bit, q_method="per-channel",
                it_num=0, it_num_thd=5, static_value=static,
            )
            xhat = outlier_addition_fuse_decompression_dequantization(
                xq, scale, xo, idx, quantize_bit=q_bit, dtype=dtype
            ).reshape(x.shape)
            c = cos_sim(xhat, x)
            res[f"q{q_bit}_o{ratio}"] = round(c, 5)
            ok_all &= c > 0.5  # sanity only; the point is the ordering assert below
        # outlier storage must IMPROVE reconstruction, else intra-op compression is a no-op
        better = res[f"q{q_bit}_o0.005"] > res[f"q{q_bit}_o0.0"]
        res[f"q{q_bit}_outlier_helps"] = bool(better)
        ok_all &= better
    return record("K6_outlier_codec", ok_all, res)


# --------------------------------------------------------------------------------------
# K7: softmax sparse codec (attention map outlier storage)
# --------------------------------------------------------------------------------------
def gate_softmax_codec(dev, dtype):
    torch.manual_seed(6)
    B, H, S = 1, 4, 128
    logits = torch.randn(B, H, S, S, device=dev, dtype=dtype)
    a = torch.softmax(logits.float(), dim=-1).to(dtype)
    static = {"outlier": None}
    packed, thr = compression_pack_softmax_base(
        x=a.clone(), o_ratio=0.05, it_num=0, it_num_thd=5, static_value=static
    )
    dense = packed.to_dense() if packed.is_sparse else packed
    kept = (dense != 0).float().mean().item()
    retained = (dense.float().sum() / a.float().sum()).item()
    # The meaningful property is SELECTIVITY: keeping ~o_ratio of the entries must retain far
    # more than o_ratio of the probability mass, i.e. it keeps the largest entries rather than
    # arbitrary ones. (Absolute mass depends on how peaked the attention is, so gating on a
    # fixed mass fraction would only be testing the synthetic input distribution.)
    selectivity = retained / max(kept, 1e-9)
    ok = 0.0 < kept < 0.2 and selectivity > 3.0
    return record("K7_softmax_codec", ok,
                  {"nonzero_frac": round(kept, 5), "mass_retained": round(retained, 5),
                   "selectivity_x": round(selectivity, 2),
                   "threshold": float(thr) if thr is not None else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/hyclora/kernel_gates.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device={dev} torch={torch.__version__} dtype={dtype}")
    import triton
    print(f"triton={triton.__version__}\n")

    gates = [gate_rmsnorm, gate_silu, gate_rope, gate_softmax,
             gate_codec, gate_outlier_codec, gate_softmax_codec]
    ok = True
    for g in gates:
        try:
            ok &= g(dev, dtype)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok &= record(g.__name__, False, {"exception": f"{type(e).__name__}: {e}"})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"all_pass": ok, "gates": RESULTS}, f, indent=2)
    print(f"\n{'ALL KERNEL GATES PASS' if ok else 'KERNEL GATES FAILED'} -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
