"""FlashFFN v3 adapter-factored paths (DoRA / AdaLoRA / DyLoRA / VeRA) —
numerical verification.

Per adapter x cache_mode {recompute, int4} at TinyLlama dims (N=2048,
D_hid=2048, D_int=5632), bf16:

  FORWARD  v3 (make_v3_forward -> FlashFFNv3AdapterFunction) vs the actual
           reference module forward (PEFT lora.Linear use_dora=True / PEFT
           SVDLinear / PEFT VeRA Linear / custom DyLoRALinear), training-mode
           semantics, dropout 0. Report max|diff| + allclose verdict at
           tol = max(2e-2, 4*ULP_bf16(max|y_ref|)) (bf16 tolerance; v3 uses the
           fused triton silu*mul and a different-but-equivalent association of
           the bypass, so a few bf16 ULP are expected — same situation as the
           v3 LoRA V1 gate). Also reports each arm's max diff vs an fp64-free
           fp32 reference for attribution.

  GRADS    v3 backward grads (collected from the live module parameters) vs an
           fp32 plain-autograd reference of the SAME semantics built from the
           SAME bf16 numeric values (TF32 off). Cosine for EVERY trainable
           tensor + grad_x.
           Targets: recompute >= 0.999 everywhere;
                    int4      >= 0.98 adapter grads / >= 0.99 grad_x.

DyLoRA's stochastic rank b is seeded identically (random.seed) between the
reference module forward, the fp32 reference and every v3 run; the projections
sample in the same order (gate, up, down) in all three.

Honesty: flashffn._V3_COUNTERS['adapter_forward'/'adapter_backward'] are
asserted to advance for every v3 case (no silent fallback to the LoRA fast
path or anything else); thresholds are never relaxed.

Run:  source env/bin/activate
      CUDA_VISIBLE_DEVICES=1 python src/verify_v3_adapters.py
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import argparse
import gc
import json
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashffn  # noqa: E402
from flashffn import (  # noqa: E402
    make_v3_forward,
    _v3_proj_kind,
    _v3_dora_col_norm,
    v3_reset_counters,
    _V3_COUNTERS,
)
from dylora import DyLoRALinear  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

D_HID, D_INT = 2048, 5632
N = 2048
RANK, ALPHA = 8, 16            # LoRA-family rank (DoRA/AdaLoRA/DyLoRA)
VERA_R = 256
DTYPE = torch.bfloat16
SEED = 41
DYLORA_SEED = 1234             # seeds random.randint for the rank sampling
ADAPTERS = ("vera", "adalora", "dylora", "dora")
CACHE_MODES = ("recompute", "int4")
OUT_DIR = "results/v3_adapters"
PROJS = ("gate_proj", "up_proj", "down_proj")


def log(msg):
    print(msg, flush=True)


def cos(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    return float(F.cosine_similarity(a, b, dim=0))


class SwiGLUMLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def build_wrapped_mlp(adapter, device):
    """Build a bf16 SwiGLU MLP wrapped by `adapter` with all adapter params
    randomized to non-degenerate values (B=0 / lambda_b=0 / E=0 inits would
    make half the grads identically zero)."""
    from peft import LoraConfig, AdaLoraConfig, VeraConfig, inject_adapter_in_model

    torch.manual_seed(SEED)
    mlp = SwiGLUMLP(D_HID, D_INT)
    for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        nn.init.normal_(lin.weight, std=0.02)
    mlp = mlp.to(device=device, dtype=DTYPE)
    targets = list(PROJS)

    if adapter == "dora":
        cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0,
                         target_modules=targets, bias="none", use_dora=True)
        mlp = inject_adapter_in_model(cfg, mlp)
        with torch.no_grad():
            for p in PROJS:
                proj = getattr(mlp, p)
                nn.init.normal_(proj.lora_B["default"].weight, std=0.02)
                m = proj.lora_magnitude_vector["default"].weight
                m.mul_(1.0 + 0.2 * torch.randn_like(m))  # decouple c from ~1
    elif adapter == "adalora":
        cfg = AdaLoraConfig(init_r=RANK, target_r=4, lora_alpha=ALPHA,
                            lora_dropout=0.0, target_modules=targets,
                            total_step=1000)
        mlp = inject_adapter_in_model(cfg, mlp)
        with torch.no_grad():
            for p in PROJS:
                proj = getattr(mlp, p)
                nn.init.normal_(proj.lora_B["default"], std=0.02)
                nn.init.normal_(proj.lora_E["default"], std=0.5)
    elif adapter == "vera":
        cfg = VeraConfig(r=VERA_R, target_modules=targets, vera_dropout=0.0,
                         d_initial=0.1)
        mlp = inject_adapter_in_model(cfg, mlp)
        with torch.no_grad():
            for p in PROJS:
                proj = getattr(mlp, p)
                lam_d = proj.vera_lambda_d["default"]
                lam_b = proj.vera_lambda_b["default"]
                lam_d.add_(0.05 * torch.randn_like(lam_d))
                lam_b.normal_(0.0, 0.05)
    elif adapter == "dylora":
        for p in PROJS:
            setattr(mlp, p, DyLoRALinear(getattr(mlp, p), r=RANK, alpha=ALPHA))
        with torch.no_grad():
            for p in PROJS:
                nn.init.normal_(getattr(mlp, p).lora_B, std=0.02)
    else:
        raise ValueError(adapter)

    mlp = mlp.to(device=device, dtype=DTYPE)
    mlp.train()
    kinds = [_v3_proj_kind(getattr(mlp, p)) for p in PROJS]
    assert kinds == [adapter] * 3, kinds
    return mlp


def collect_trainables(mlp, adapter):
    """name -> live parameter tensor (the ones v3 must produce grads for)."""
    out = {}
    for p in PROJS:
        proj = getattr(mlp, p)
        if adapter == "dora":
            out[f"{p}.A"] = proj.lora_A["default"].weight
            out[f"{p}.B"] = proj.lora_B["default"].weight
            out[f"{p}.m"] = proj.lora_magnitude_vector["default"].weight
        elif adapter == "adalora":
            assert not proj.ranknum["default"].requires_grad
            out[f"{p}.A"] = proj.lora_A["default"]
            out[f"{p}.E"] = proj.lora_E["default"]
            out[f"{p}.B"] = proj.lora_B["default"]
        elif adapter == "vera":
            out[f"{p}.lam_d"] = proj.vera_lambda_d["default"]
            out[f"{p}.lam_b"] = proj.vera_lambda_b["default"]
        elif adapter == "dylora":
            out[f"{p}.A"] = proj.lora_A
            out[f"{p}.B"] = proj.lora_B
    for t in out.values():
        assert t.requires_grad, "trainable expected"
    return out


# ---------------------------------------------------------------------------
# fp32 autograd reference (same semantics, same bf16 numeric values)
# ---------------------------------------------------------------------------

def build_ref_params(mlp, adapter):
    """fp32 leaf clones of every tensor the reference needs."""
    def leaf(t):
        return t.detach().float().clone().requires_grad_(True)

    P = {}
    for p in PROJS:
        proj = getattr(mlp, p)
        d = {"kind": adapter}
        if adapter == "dora":
            d["w"] = proj.get_base_layer().weight.detach().float()
            d["a"] = leaf(proj.lora_A["default"].weight)
            d["b"] = leaf(proj.lora_B["default"].weight)
            d["m"] = leaf(proj.lora_magnitude_vector["default"].weight)
            d["s"] = float(proj.scaling["default"])
        elif adapter == "adalora":
            d["w"] = proj.get_base_layer().weight.detach().float()
            d["a"] = leaf(proj.lora_A["default"])
            d["e"] = leaf(proj.lora_E["default"])
            d["b"] = leaf(proj.lora_B["default"])
            d["s_eff"] = float(proj.scaling["default"]) / (
                float(proj.ranknum["default"]) + 1e-5)
        elif adapter == "vera":
            d["w"] = proj.get_base_layer().weight.detach().float()
            d["a_frozen"] = proj.vera_A["default"][:, :proj.in_features].detach().float()
            d["b_frozen"] = proj.vera_B["default"][:proj.out_features, :].detach().float()
            d["lam_d"] = leaf(proj.vera_lambda_d["default"])
            d["lam_b"] = leaf(proj.vera_lambda_b["default"])
        elif adapter == "dylora":
            d["w"] = proj.base_layer.weight.detach().float()
            d["a"] = leaf(proj.lora_A)
            d["b"] = leaf(proj.lora_B)
            d["s"] = float(proj.scaling)
        P[p] = d
    return P


def ref_proj_forward(z, d, b_rank=None):
    kind = d["kind"]
    if kind == "dora":
        # PEFT semantics: c = m / ||W + s*B@A||_dim1.DETACHED; out = c * raw
        merged = d["w"] + d["s"] * (d["b"] @ d["a"])
        nu = merged.detach().norm(p=2, dim=1)
        c = d["m"] / nu
        raw = F.linear(z, d["w"]) + d["s"] * F.linear(F.linear(z, d["a"]), d["b"])
        return raw * c.unsqueeze(0)
    if kind == "adalora":
        a_eff = d["a"] * d["e"]
        return F.linear(z, d["w"]) + d["s_eff"] * F.linear(
            F.linear(z, a_eff), d["b"])
    if kind == "vera":
        u = F.linear(z, d["a_frozen"]) * d["lam_d"]
        return F.linear(z, d["w"]) + F.linear(u, d["b_frozen"]) * d["lam_b"]
    if kind == "dylora":
        a = d["a"][:b_rank, :]
        b = d["b"][:, :b_rank]
        return F.linear(z, d["w"]) + d["s"] * F.linear(F.linear(z, a), b)
    raise ValueError(kind)


def ref_forward(x32, P, b_ranks):
    hg = ref_proj_forward(x32, P["gate_proj"], b_ranks[0])
    hu = ref_proj_forward(x32, P["up_proj"], b_ranks[1])
    hmid = F.silu(hg) * hu
    return ref_proj_forward(hmid, P["down_proj"], b_ranks[2])


def ref_grads(mlp, adapter, x, gy, b_ranks):
    P = build_ref_params(mlp, adapter)
    x32 = x.detach().float().clone().requires_grad_(True)
    y32 = ref_forward(x32, P, b_ranks)
    y32.backward(gy.float())
    grads = {"x": x32.grad.detach()}
    name_map = {"a": "A", "e": "E", "b": "B", "m": "m",
                "lam_d": "lam_d", "lam_b": "lam_b"}
    for p in PROJS:
        for k, t in P[p].items():
            if isinstance(t, torch.Tensor) and t.requires_grad:
                grads[f"{p}.{name_map[k]}"] = t.grad.detach()
    y32_d = y32.detach()
    del P, x32, y32
    return grads, y32_d


# ---------------------------------------------------------------------------
# main per-case runner
# ---------------------------------------------------------------------------

def sample_b_ranks(adapter):
    """Replicate the exact random.randint sequence the v3 closure (and the
    reference modules) consume: one draw per projection, order gate/up/down."""
    if adapter != "dylora":
        return (None, None, None)
    random.seed(DYLORA_SEED)
    return tuple(random.randint(1, RANK) for _ in range(3))


def run_adapter(adapter, device, results):
    log(f"\n=== {adapter.upper()} (N={N}, dims {D_HID}x{D_INT}, bf16) ===")
    mlp = build_wrapped_mlp(adapter, device)
    trainables = collect_trainables(mlp, adapter)

    torch.manual_seed(SEED + 1)
    x = torch.randn(N, D_HID, device=device, dtype=DTYPE)
    gy = torch.randn(N, D_HID, device=device, dtype=DTYPE)

    b_ranks = sample_b_ranks(adapter)
    if adapter == "dylora":
        log(f"  DyLoRA sampled ranks (gate/up/down): {b_ranks}")

    # --- reference module forward (the actual PEFT / DyLoRA modules) ---
    with torch.no_grad():
        random.seed(DYLORA_SEED)
        y_mod = mlp(x)

    # --- fp32 autograd reference (same numeric values, same b) ---
    grads_ref, y32 = ref_grads(mlp, adapter, x, gy, b_ranks)

    # DoRA extra: Gram-identity norm vs reference fp32 norm
    if adapter == "dora":
        max_nu_rel = 0.0
        for p in PROJS:
            proj = getattr(mlp, p)
            w = proj.get_base_layer().weight
            a = proj.lora_A["default"].weight
            b = proj.lora_B["default"].weight
            s = float(proj.scaling["default"])
            with torch.no_grad():
                nu_gram = _v3_dora_col_norm(w, a, b, s)
                merged = w.float() + s * (b.float() @ a.float())
                nu_ref = merged.norm(p=2, dim=1)
            max_nu_rel = max(max_nu_rel,
                             float(((nu_gram - nu_ref).abs() / nu_ref).max()))
        log(f"  DoRA Gram-identity norm vs direct fp32 norm: "
            f"max rel diff {max_nu_rel:.3e}")
        results[f"{adapter}.nu_max_rel_diff"] = max_nu_rel

    mod_vs_32 = float((y_mod.double() - y32.double()).abs().max())

    for mode in CACHE_MODES:
        v3_reset_counters()
        fwd = make_v3_forward(mlp, cache_mode=mode)
        for t in trainables.values():
            t.grad = None
        x_v3 = x.detach().clone().requires_grad_(True)
        random.seed(DYLORA_SEED)
        y_v3 = fwd(x_v3)
        assert _V3_COUNTERS["adapter_forward"] == 1, _V3_COUNTERS
        y_v3.backward(gy)
        assert _V3_COUNTERS["adapter_backward"] == 1, _V3_COUNTERS

        # forward check
        max_diff = float((y_v3.detach().double() - y_mod.double()).abs().max())
        v3_vs_32 = float((y_v3.detach().double() - y32.double()).abs().max())
        ymax = float(y_mod.abs().max())
        tol = max(2e-2, 4 * (2.0 ** -8) * ymax)
        fwd_ok = max_diff <= tol
        log(f"  [{mode:9s}] forward: max|v3-module|={max_diff:.5f} "
            f"(tol {tol:.5f} -> {'PASS' if fwd_ok else 'FAIL'}) | "
            f"max|v3-fp32|={v3_vs_32:.5f} max|module-fp32|={mod_vs_32:.5f}")

        # grad check
        case = {"forward_max_diff": max_diff, "forward_tol": tol,
                "forward_ok": fwd_ok, "v3_vs_fp32": v3_vs_32,
                "module_vs_fp32": mod_vs_32, "cos": {}}
        thr_adapter = 0.999 if mode == "recompute" else 0.98
        thr_x = 0.999 if mode == "recompute" else 0.99
        all_ok = fwd_ok
        cos_x = cos(x_v3.grad, grads_ref["x"])
        case["cos"]["grad_x"] = cos_x
        ok = cos_x >= thr_x
        all_ok &= ok
        log(f"             grad_x cos={cos_x:.5f} (thr {thr_x}) "
            f"{'PASS' if ok else 'FAIL'}")
        for nm, t in trainables.items():
            assert t.grad is not None, f"v3 produced no grad for {nm}"
            c = cos(t.grad, grads_ref[nm])
            case["cos"][nm] = c
            ok = c >= thr_adapter
            all_ok &= ok
            log(f"             {nm:16s} cos={c:.5f} (thr {thr_adapter}) "
                f"{'PASS' if ok else 'FAIL'}")
        case["pass"] = bool(all_ok)
        results[f"{adapter}.{mode}"] = case

        del x_v3, y_v3
        for t in trainables.values():
            t.grad = None
        gc.collect()
        torch.cuda.empty_cache()

    del mlp, trainables, grads_ref, y_mod, y32, x, gy
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=list(ADAPTERS))
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for adapter in args.adapters:
        run_adapter(adapter, device, results)

    # summary table
    log("\n" + "=" * 100)
    log(f"{'adapter':9s} {'mode':10s} {'fwd max|diff|':>13s} {'fwd':>5s} "
        f"{'min adapter cos':>16s} {'grad_x cos':>11s} {'verdict':>8s}")
    overall = True
    for adapter in args.adapters:
        for mode in CACHE_MODES:
            c = results[f"{adapter}.{mode}"]
            ad_cos = [v for k, v in c["cos"].items() if k != "grad_x"]
            verdict = "PASS" if c["pass"] else "FAIL"
            overall &= c["pass"]
            log(f"{adapter:9s} {mode:10s} {c['forward_max_diff']:13.5f} "
                f"{'ok' if c['forward_ok'] else 'FAIL':>5s} "
                f"{min(ad_cos):16.5f} {c['cos']['grad_x']:11.5f} {verdict:>8s}")
    log("=" * 100)
    log(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    with open(os.path.join(OUT_DIR, "verify_v3_adapters.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)
    log(f"[saved {OUT_DIR}/verify_v3_adapters.json]")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
