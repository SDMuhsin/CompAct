"""FlashFFN v3 prototype — verification gates V1..V5 and benchmarks.

V1  Forward exactness vs a PEFT-LoRA-wrapped SwiGLU MLP (bf16, TinyLlama dims).
V2  Gradient correctness vs an fp32 plain-autograd reference built from the
    SAME bf16 inputs (synthetic Gaussian + REAL TinyLlama layer-11 capture).
V3  Memory honesty: non-parameter saved-for-backward bytes via
    torch.autograd.graph.saved_tensors_hooks.
V4  Speed: CUDA events, interleaved arms, p10 + median.
V5  E2E smoke: TinyLlama + PEFT LoRA r8 (MLP projections), wikitext-2,
    30 optimizer steps, bf16.

Run from repo root:
  source env/bin/activate
  HF_HOME=./data TORCH_HOME=./data CUDA_VISIBLE_DEVICES=<idle GPU> \
      python src/verify_flashffn_v3.py [--gates v1 v2 ...] [--quick]

Honesty rules implemented here:
  - flashffn._V3_COUNTERS asserted > 0 after every v3 test (and == 0 for
    non-v3 arms), so a silent fallback can never produce a reported number.
  - No thresholds are relaxed; failures are reported as failures.
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 1 was the idler at design time
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import argparse
import gc
import json
import math
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashffn  # noqa: E402
from flashffn import (  # noqa: E402
    FlashFFNFunction,
    FlashFFNv3Function,
    make_v3_forward,
    v3_quantize_group_absmax,
    v3_dequantize_group_absmax,
    v3_reset_counters,
    _V3_COUNTERS,
    _V3_SKELETON,
)

D_HID, D_INT, RANK, ALPHA = 2048, 5632, 8, 16
SCALING = ALPHA / RANK
DTYPE = torch.bfloat16
OUT_DIR = "results/v3_prototype"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CACHE_MODES = ("int4", "int8", "bf16", "recompute")
ADAPTER_GRADS = ("a_gate", "b_gate", "a_up", "b_up", "a_down", "b_down")
WEIGHT_GRADS = ("w_gate", "w_up", "w_down")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def save_json(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    log(f"  [saved {path}]")


def tensor_cos(a, b):
    return float(F.cosine_similarity(a.double().flatten(),
                                     b.double().flatten(), dim=0))


def rel_frob(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


class SwiGLUMLP(nn.Module):
    """LLaMA-style SwiGLU MLP (reference forward = HF op order)."""

    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def build_mlp(config, seed, device):
    """config: 'all' | 'gateup' | 'none' | 'fullft'. Returns a (possibly
    PEFT-wrapped) SwiGLUMLP in bf16 with adapters forced to bf16 and
    lora_B randomized (nonzero, otherwise grad_A == 0 makes V2 degenerate)."""
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(seed)
    mlp = SwiGLUMLP(D_HID, D_INT)
    for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        nn.init.normal_(lin.weight, std=0.02)
    mlp = mlp.to(device=device, dtype=DTYPE)

    if config in ("all", "gateup"):
        targets = ["gate_proj", "up_proj", "down_proj"] if config == "all" \
            else ["gate_proj", "up_proj"]
        cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0,
                         target_modules=targets, bias="none")
        mlp = inject_adapter_in_model(cfg, mlp)
        for t in targets:
            proj = getattr(mlp, t)
            name = proj.active_adapters[0]
            a = proj.lora_A[name].weight
            b = proj.lora_B[name].weight
            with torch.no_grad():
                b.data.copy_(torch.randn(b.shape) * 0.02)   # CPU RNG, seeded
            proj.lora_A[name].weight.data = a.data.to(DTYPE)
            proj.lora_B[name].weight.data = b.data.to(DTYPE)
        for n, p in mlp.named_parameters():
            p.requires_grad = "lora_" in n
    elif config == "fullft":
        for p in mlp.parameters():
            p.requires_grad = True
    else:  # none
        for p in mlp.parameters():
            p.requires_grad = False
    return mlp


def extract_param_set(mlp, config):
    """Pull (W, A, B, s) per projection out of a built mlp into a dict of
    tensors (still the module's tensors; not copies)."""
    ps = {}
    for key, proj in (("gate", mlp.gate_proj), ("up", mlp.up_proj),
                      ("down", mlp.down_proj)):
        w, a, b, s = flashffn._v3_extract_proj(proj)
        ps[f"w_{key}"], ps[f"a_{key}"], ps[f"b_{key}"] = w, a, b
        ps[f"s_{key}"] = s
    ps["train_w"] = (config == "fullft")
    return ps


def synthetic_param_set(config, seed, device):
    """Standalone leaf-tensor param set (no module) for V2."""
    torch.manual_seed(seed)
    ps = {"train_w": config == "fullft"}
    ps["w_gate"] = (torch.randn(D_INT, D_HID) * 0.02).to(device, DTYPE)
    ps["w_up"] = (torch.randn(D_INT, D_HID) * 0.02).to(device, DTYPE)
    ps["w_down"] = (torch.randn(D_HID, D_INT) * 0.02).to(device, DTYPE)
    _add_synthetic_adapters(ps, config, device)
    return ps


def _add_synthetic_adapters(ps, config, device):
    has_gu = config in ("all", "gateup")
    has_d = config == "all"
    for key, d_in in (("gate", D_HID), ("up", D_HID)):
        if has_gu:
            ps[f"a_{key}"] = (torch.randn(RANK, d_in) / math.sqrt(d_in)).to(device, DTYPE)
            ps[f"b_{key}"] = (torch.randn(D_INT, RANK) * 0.02).to(device, DTYPE)
            ps[f"s_{key}"] = SCALING
        else:
            ps[f"a_{key}"] = ps[f"b_{key}"] = None
            ps[f"s_{key}"] = 0.0
    if has_d:
        ps["a_down"] = (torch.randn(RANK, D_INT) / math.sqrt(D_INT)).to(device, DTYPE)
        ps["b_down"] = (torch.randn(D_HID, RANK) * 0.02).to(device, DTYPE)
        ps["s_down"] = SCALING
    else:
        ps["a_down"] = ps["b_down"] = None
        ps["s_down"] = 0.0


# ---------------------------------------------------------------------------
# Real activation capture (TinyLlama layer 11, wikitext-2) — reuses the
# capture approach from src/codec_feasibility_v3.py
# ---------------------------------------------------------------------------

def build_batches(tokenizer, n_batches, batch_size, seq_len):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text[:1_200_000], return_tensors="pt").input_ids[0]
    need = n_batches * batch_size * seq_len
    assert ids.numel() >= need, f"not enough tokens: {ids.numel()} < {need}"
    return ids[:need].view(n_batches, batch_size, seq_len)


def get_real_capture(device, layer_idx=11):
    path = os.path.join(OUT_DIR, f"real_capture_layer{layer_idx}.pt")
    if os.path.exists(path):
        log(f"  [V2] using cached real capture: {path}")
        return torch.load(path)
    log("  [V2] capturing real TinyLlama activations (wikitext-2) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    batch = build_batches(tok, 1, 4, 512)[0].to(device)

    cap = {}
    mlp = model.model.layers[layer_idx].mlp

    def fwd_hook(module, inputs, output):
        cap["x"] = inputs[0].detach().to("cpu", copy=True)
        output.register_hook(
            lambda g: cap.__setitem__("grad_y", g.detach().to("cpu", copy=True)))

    h = mlp.register_forward_hook(fwd_hook)
    out = model(input_ids=batch, labels=batch)
    loss = float(out.loss.detach())
    out.loss.backward()
    h.remove()
    data = {
        "x": cap["x"], "grad_y": cap["grad_y"],
        "w_gate": mlp.gate_proj.weight.detach().cpu(),
        "w_up": mlp.up_proj.weight.detach().cpu(),
        "w_down": mlp.down_proj.weight.detach().cpu(),
        "loss": loss, "layer": layer_idx,
    }
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(data, path)
    log(f"  [V2] capture done (clm loss {loss:.4f}); cached to {path}")
    return data


def real_param_set(config, seed, device):
    cap = get_real_capture(device)
    torch.manual_seed(seed)
    ps = {"train_w": config == "fullft"}
    for k in ("w_gate", "w_up", "w_down"):
        ps[k] = cap[k].to(device)
    _add_synthetic_adapters(ps, config, device)
    x = cap["x"].to(device).reshape(-1, D_HID)
    gy = cap["grad_y"].to(device).reshape(-1, D_HID)
    return ps, x, gy


# ---------------------------------------------------------------------------
# V2 runners
# ---------------------------------------------------------------------------

def run_v3_grads(ps, x16, gy16, mode, device):
    """Run FlashFFNv3Function fwd+bwd in bf16 on leaf clones; return grads."""
    def leaf(t, rg):
        if t is None:
            return None
        return t.detach().clone().requires_grad_(rg)

    train_w = ps["train_w"]
    xl = x16.detach().clone().requires_grad_(True)
    tensors = {}
    for key in ("gate", "up", "down"):
        tensors[f"w_{key}"] = leaf(ps[f"w_{key}"], train_w)
        has_ad = ps[f"a_{key}"] is not None
        tensors[f"a_{key}"] = leaf(ps[f"a_{key}"], has_ad)
        tensors[f"b_{key}"] = leaf(ps[f"b_{key}"], has_ad)

    v3_reset_counters()
    y = FlashFFNv3Function.apply(
        xl,
        tensors["w_gate"], tensors["a_gate"], tensors["b_gate"],
        tensors["w_up"], tensors["a_up"], tensors["b_up"],
        tensors["w_down"], tensors["a_down"], tensors["b_down"],
        ps["s_gate"], ps["s_up"], ps["s_down"], mode, 64)
    y.backward(gy16)
    assert _V3_COUNTERS["forward"] == 1 and _V3_COUNTERS["backward"] == 1, \
        "v3 code path did not run"

    grads = {"grad_x": xl.grad}
    for k, t in tensors.items():
        if t is not None and t.requires_grad:
            grads[k] = t.grad
    return grads, y.detach()


def run_ref_grads_fp32(ps, x16, gy16):
    """fp32 plain-autograd reference of the SAME factored computation, from
    the SAME bf16 inputs upcast. TF32 must be globally disabled."""
    def leaf(t, rg):
        if t is None:
            return None
        return t.detach().float().requires_grad_(rg)

    train_w = ps["train_w"]
    xl = x16.detach().float().requires_grad_(True)
    t = {}
    for key in ("gate", "up", "down"):
        t[f"w_{key}"] = leaf(ps[f"w_{key}"], train_w)
        has_ad = ps[f"a_{key}"] is not None
        t[f"a_{key}"] = leaf(ps[f"a_{key}"], has_ad)
        t[f"b_{key}"] = leaf(ps[f"b_{key}"], has_ad)

    h_gate = F.linear(xl, t["w_gate"])
    if t["a_gate"] is not None:
        h_gate = h_gate + ps["s_gate"] * F.linear(F.linear(xl, t["a_gate"]), t["b_gate"])
    h_up = F.linear(xl, t["w_up"])
    if t["a_up"] is not None:
        h_up = h_up + ps["s_up"] * F.linear(F.linear(xl, t["a_up"]), t["b_up"])
    h_mid = F.silu(h_gate) * h_up
    y = F.linear(h_mid, t["w_down"])
    if t["a_down"] is not None:
        y = y + ps["s_down"] * F.linear(F.linear(h_mid, t["a_down"]), t["b_down"])
    y.backward(gy16.float())

    grads = {"grad_x": xl.grad}
    for k, tt in t.items():
        if tt is not None and tt.requires_grad:
            grads[k] = tt.grad
    return grads


# ---------------------------------------------------------------------------
# V1 — forward exactness
# ---------------------------------------------------------------------------

def _v1_attribution(mlp, x, y_ref, y_v3):
    """Attribute any y_v3-vs-y_ref difference.

    semi-reference: identical to the PEFT reference EXCEPT h_mid is computed
    with the (spec-mandated) fused triton_silu_mul (single fp32 rounding)
    instead of eager bf16(silu(g)) * u (double rounding). If y_v3 is bitwise
    equal to the semi-reference, the entire difference is the silu*mul
    rounding convention — every LoRA-factored path is bitwise-exact vs PEFT.
    Also compares both against an fp64 ground truth of the same factored
    computation: v3 should be no farther from the truth than the reference.
    """
    from flashffn import triton_silu_mul

    with torch.no_grad():
        hg = mlp.gate_proj(x)
        hu = mlp.up_proj(x)
        y_semi = mlp.down_proj(triton_silu_mul(hg, hu))

        x64 = x.reshape(-1, D_HID).double()

        def proj64(t, proj):
            w, a, b, s = flashffn._v3_extract_proj(proj)
            out = F.linear(t, w.double())
            if a is not None:
                out = out + s * F.linear(F.linear(t, a.double()), b.double())
            return out

        hg64 = proj64(x64, mlp.gate_proj)
        hu64 = proj64(x64, mlp.up_proj)
        y64 = proj64(F.silu(hg64) * hu64, mlp.down_proj).reshape(y_ref.shape)

    err_ref = (y_ref.double() - y64)
    err_v3 = (y_v3.double() - y64)
    return {
        "v3_bitwise_eq_semi_ref": bool(torch.equal(y_v3, y_semi)),
        "max_err_ref_vs_fp64": float(err_ref.abs().max()),
        "max_err_v3_vs_fp64": float(err_v3.abs().max()),
        "rms_err_ref_vs_fp64": float(err_ref.pow(2).mean().sqrt()),
        "rms_err_v3_vs_fp64": float(err_v3.pow(2).mean().sqrt()),
    }


def gate_v1(args, device):
    log("\n=== V1: forward exactness (bf16, TinyLlama dims) ===")
    results = []
    strict_all = True
    attributed_all = True
    cases = [(c, N) for c in ("all", "gateup", "none", "fullft")
             for N in (512, 2048)]
    cases.append(("all", "3d"))  # HF models pass [B, S, H]
    for config, N in cases:
        mlp = build_mlp(config, args.seed, device)
        if N == "3d":
            torch.manual_seed(args.seed + 3)
            x = torch.randn(4, 512, D_HID).to(device, DTYPE)
        else:
            torch.manual_seed(args.seed + N)
            x = torch.randn(N, D_HID).to(device, DTYPE)
        with torch.no_grad():
            y_ref = mlp(x)
        ys = {}
        for mode in CACHE_MODES:
            v3_reset_counters()
            y = make_v3_forward(mlp, mode)(x)
            assert _V3_COUNTERS["forward"] == 1, "v3 forward did not run"
            ys[mode] = y.detach()
        # forward output must be identical across cache modes (caching only
        # affects what is SAVED, never the computed y)
        modes_equal = all(torch.equal(ys["int4"], ys[m]) for m in CACHE_MODES)
        y_v3 = ys["int4"]
        max_abs = float((y_v3 - y_ref).abs().max())
        strict = bool(torch.allclose(y_v3.float(), y_ref.float(),
                                     atol=2e-2, rtol=0.0)) and modes_equal \
            and y_v3.shape == y_ref.shape
        bitwise = float((y_v3 == y_ref).float().mean())
        attribution = None
        attributed = strict
        if not strict:
            # do NOT relax the threshold — attribute the difference instead
            attribution = _v1_attribution(mlp, x, y_ref, y_v3)
            attributed = (modes_equal
                          and attribution["v3_bitwise_eq_semi_ref"]
                          and attribution["rms_err_v3_vs_fp64"]
                          <= attribution["rms_err_ref_vs_fp64"] * 1.05)
        strict_all &= strict
        attributed_all &= attributed
        rec = {"config": config, "N": N, "max_abs_diff": max_abs,
               "allclose_atol2e-2_strict": strict,
               "bitwise_equal_frac": bitwise,
               "cache_modes_identical": modes_equal,
               "attribution": attribution, "attributed_pass": attributed}
        results.append(rec)
        status = "PASS" if strict else (
            "FAIL(strict)/PASS(attributed: fused silu*mul rounding only)"
            if attributed else "FAIL")
        log(f"  {config:7s} N={str(N):5s}: max|diff|={max_abs:.5f} "
            f"strict_allclose={strict} bitwise={bitwise * 100:.1f}% "
            f"modes_identical={modes_equal} -> {status}")
        if attribution is not None:
            log(f"          attribution: v3==semi_ref(bitwise)="
                f"{attribution['v3_bitwise_eq_semi_ref']}, vs fp64 truth "
                f"max err v3={attribution['max_err_v3_vs_fp64']:.5f} <= "
                f"ref={attribution['max_err_ref_vs_fp64']:.5f}, rms "
                f"v3={attribution['rms_err_v3_vs_fp64']:.6f} <= "
                f"ref={attribution['rms_err_ref_vs_fp64']:.6f}")
        del mlp, x, y_ref, ys
        gc.collect()
        torch.cuda.empty_cache()

    save_json("v1_forward.json", {"pass_strict": strict_all,
                                  "pass_attributed": attributed_all,
                                  "cases": results})
    log(f"V1 OVERALL: strict={'PASS' if strict_all else 'FAIL'}, "
        f"attributed={'PASS' if attributed_all else 'FAIL'}")
    return attributed_all, {"strict": strict_all, "attributed": attributed_all,
                            "cases": results}


# ---------------------------------------------------------------------------
# V2 — gradient correctness
# ---------------------------------------------------------------------------

def _judge_v2(config, mode, metrics):
    """Apply the spec pass criteria. Returns (judged: bool|None, detail)."""
    cos = {k: v["cos"] for k, v in metrics.items()}
    if mode in ("recompute", "bf16"):
        return all(c >= 0.999 for c in cos.values()), "all cos >= 0.999"
    if mode == "int4":
        if config in ("all", "gateup"):
            ad = [c for k, c in cos.items() if k in ADAPTER_GRADS]
            ok = all(c >= 0.98 for c in ad) and cos["grad_x"] >= 0.99
            return ok, "adapter grads >= 0.98, grad_x >= 0.99"
        return None, "full-FT int4: reported, no spec threshold"
    return None, "int8: reported, no spec threshold"


def gate_v2(args, device):
    log("\n=== V2: gradient correctness vs fp32 reference ===")
    N = 2048
    results = []
    ok_all = True
    datasets = ["synthetic", "real"]
    for data in datasets:
        for config in ("all", "gateup", "fullft"):
            if data == "real" and config == "gateup":
                continue  # partial-adapter path covered on synthetic
            if data == "synthetic":
                ps = synthetic_param_set(config, args.seed, device)
                torch.manual_seed(args.seed + 7)
                x16 = torch.randn(N, D_HID).to(device, DTYPE)
                gy16 = torch.randn(N, D_HID).to(device, DTYPE)
            else:
                ps, x16, gy16 = real_param_set(config, args.seed, device)
            ref = run_ref_grads_fp32(ps, x16, gy16)
            for mode in CACHE_MODES:
                grads, _ = run_v3_grads(ps, x16, gy16, mode, device)
                assert set(grads.keys()) == set(ref.keys()), \
                    f"grad key mismatch: {sorted(grads)} vs {sorted(ref)}"
                metrics = {k: {"cos": tensor_cos(grads[k], ref[k]),
                               "rel_frob": rel_frob(grads[k], ref[k])}
                           for k in sorted(ref.keys())}
                judged, crit = _judge_v2(config, mode, metrics)
                if judged is not None:
                    ok_all &= judged
                worst = min(metrics.items(), key=lambda kv: kv[1]["cos"])
                results.append({"data": data, "config": config, "mode": mode,
                                "metrics": metrics, "criterion": crit,
                                "pass": judged})
                status = ("PASS" if judged else "FAIL") if judged is not None \
                    else "report"
                log(f"  {data:9s} {config:7s} {mode:9s}: worst cos "
                    f"{worst[0]}={worst[1]['cos']:.6f} "
                    f"(relF={worst[1]['rel_frob']:.5f}) [{status}]")
                del grads
                gc.collect()
                torch.cuda.empty_cache()
            del ref, ps
            gc.collect()
            torch.cuda.empty_cache()

    save_json("v2_grads.json", {"pass": ok_all, "N": N, "cases": results})
    log(f"V2 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, results


# ---------------------------------------------------------------------------
# V3 — memory honesty (saved-for-backward bytes)
# ---------------------------------------------------------------------------

def count_saved_bytes(forward_fn, param_ptrs):
    """Non-parameter bytes saved for backward, deduped by storage pointer."""
    seen = set()
    entries = []
    total = 0

    def pack(t):
        nonlocal total
        key = (t.data_ptr(), t.nbytes)
        if t.data_ptr() in param_ptrs or key in seen:
            return t
        seen.add(key)
        total += t.nbytes
        entries.append({"shape": list(t.shape), "dtype": str(t.dtype),
                        "bytes": t.nbytes})
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = forward_fn()
    del y
    return total, entries


def lora_eff_weight(proj):
    """v2 effective-weight path (mirrors train_glue._compute_lora_effective_weight)."""
    name = proj.active_adapters[0]
    w = proj.get_base_layer().weight
    a = proj.lora_A[name].weight
    b = proj.lora_B[name].weight
    s = proj.scaling[name]
    return (w + s * (b @ a)).to(w.dtype)


def gate_v3(args, device):
    log("\n=== V3: memory honesty (saved-for-backward bytes) ===")
    mlp = build_mlp("all", args.seed, device)
    param_ptrs = {p.data_ptr() for p in mlp.parameters()}
    mib = 1024 * 1024
    table = {}
    ok_all = True
    for N in (2048, 8192):
        torch.manual_seed(args.seed + N)
        x_vals = torch.randn(N, D_HID).to(device, DTYPE)
        row = {}

        def fresh_x():
            return x_vals.detach().clone().requires_grad_(True)

        # standard PEFT-LoRA MLP
        x = fresh_x()
        b, e = count_saved_bytes(lambda: mlp(x), param_ptrs | {x.data_ptr()})
        row["standard_peft"] = {"bytes": b + x.nbytes, "n_tensors": len(e) + 1,
                                "entries": e}
        # v2 effective-weight FlashFFN (W_eff are non-leaf -> counted)
        x = fresh_x()
        b, e = count_saved_bytes(
            lambda: FlashFFNFunction.apply(
                x, lora_eff_weight(mlp.gate_proj), lora_eff_weight(mlp.up_proj),
                lora_eff_weight(mlp.down_proj), 0.3),
            param_ptrs | {x.data_ptr()})
        row["v2_eff_weight"] = {"bytes": b + x.nbytes, "n_tensors": len(e) + 1,
                                "entries": e}
        # v3 modes
        for mode in ("int4", "int8", "bf16", "recompute"):
            x = fresh_x()
            v3_reset_counters()
            b, e = count_saved_bytes(lambda: make_v3_forward(mlp, mode)(x),
                                     param_ptrs | {x.data_ptr()})
            assert _V3_COUNTERS["forward"] == 1
            row[f"v3_{mode}"] = {"bytes": b + x.nbytes,
                                 "n_tensors": len(e) + 1, "entries": e}
        # NOTE: x is excluded from the hook count (it is a leaf input saved by
        # several ops) and added back exactly once per arm — identical
        # accounting for every arm.

        x_bytes = N * D_HID * 2
        expected_int4 = (x_bytes + 2 * (N * D_INT // 2)
                         + 2 * (N * (D_INT // 64) * 2) + 3 * (N * RANK * 2))
        eps = 256 * 1024
        got = row["v3_int4"]["bytes"]
        tgt_ok = (got <= expected_int4 + eps
                  and got < row["v2_eff_weight"]["bytes"]
                  and got < row["standard_peft"]["bytes"])
        ok_all &= tgt_ok
        row["int4_target"] = {"expected_bytes": expected_int4, "eps": eps,
                              "measured": got, "pass": tgt_ok}
        table[str(N)] = row
        log(f"  N={N}:")
        for arm in ("standard_peft", "v2_eff_weight", "v3_bf16", "v3_int8",
                    "v3_int4", "v3_recompute"):
            d = row[arm]
            log(f"    {arm:14s}: {d['bytes'] / mib:8.2f} MiB "
                f"({d['n_tensors']} tensors)")
        log(f"    int4 target: measured {got / mib:.2f} <= expected "
            f"{(expected_int4 + eps) / mib:.2f} MiB and < v2/std -> "
            f"{'PASS' if tgt_ok else 'FAIL'}")
        del x, x_vals
        gc.collect()
        torch.cuda.empty_cache()

    save_json("v3_memory.json", {"pass": ok_all, "table": table})
    log(f"V3 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, table


# ---------------------------------------------------------------------------
# V4 — speed
# ---------------------------------------------------------------------------

def gate_v4(args, device):
    log("\n=== V4: speed (CUDA events, interleaved arms) ===")
    warmup, iters = (3, 10) if args.quick else (10, 60)
    mlp = build_mlp("all", args.seed, device)
    adapter_params = [p for p in mlp.parameters() if p.requires_grad]
    out = {}
    ok_all = True
    n_list = (2048, 8192) if args.quick else (2048, 8192, 16384)
    for N in n_list:
        torch.manual_seed(args.seed + N)
        x_vals = torch.randn(N, D_HID).to(device, DTYPE)
        gy = torch.randn(N, D_HID).to(device, DTYPE)

        arm_x = {}

        def make_x(name):
            arm_x[name] = x_vals.detach().clone().requires_grad_(True)
            return arm_x[name]

        def step_standard(x=make_x("standard_peft")):
            mlp(x).backward(gy)

        def step_v2(x=make_x("v2_eff_weight")):
            FlashFFNFunction.apply(
                x, lora_eff_weight(mlp.gate_proj), lora_eff_weight(mlp.up_proj),
                lora_eff_weight(mlp.down_proj), 0.3).backward(gy)

        v3_fwds = {m: make_v3_forward(mlp, m) for m in CACHE_MODES}

        def make_v3_step(mode):
            x = make_x(f"v3_{mode}")

            def step():
                v3_fwds[mode](x).backward(gy)
            return step

        def step_skeleton(x=make_x("v3_int4_skeleton")):
            _V3_SKELETON["enabled"] = True
            try:
                v3_fwds["int4"](x).backward(gy)
            finally:
                _V3_SKELETON["enabled"] = False

        arms = {
            "standard_peft": step_standard,
            "v2_eff_weight": step_v2,
            "v3_int4": make_v3_step("int4"),
            "v3_recompute": make_v3_step("recompute"),
            "v3_bf16": make_v3_step("bf16"),
            "v3_int4_skeleton": step_skeleton,
        }

        events = {a: [] for a in arms}
        v3_reset_counters()
        for it in range(warmup + iters):
            for name, step in arms.items():
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                step()
                e1.record()
                if it >= warmup:
                    events[name].append((e0, e1))
                # uniform cleanup outside the timed window
                arm_x[name if name in arm_x else name].grad = None
                for p in adapter_params:
                    p.grad = None
            torch.cuda.synchronize()
        assert _V3_COUNTERS["forward"] >= 4 * (warmup + iters), \
            "v3 arms did not run"

        stats = {}
        for name, evs in events.items():
            ts = sorted(e0.elapsed_time(e1) for e0, e1 in evs)
            stats[name] = {
                "p10_ms": ts[max(0, int(0.10 * len(ts)) - 1)] if len(ts) >= 10 else ts[0],
                "median_ms": statistics.median(ts),
                "n": len(ts),
            }

        # quant/dequant standalone cost (the eager-PyTorch overhead int4 pays)
        h = torch.randn(N, D_INT).to(device, DTYPE)
        qd = {}
        for label, fn in (
            ("quant_int4", lambda: v3_quantize_group_absmax(h, 4, 64)),
            ("dequant_int4", None),
        ):
            if label == "dequant_int4":
                payload, scales = v3_quantize_group_absmax(h, 4, 64)
                fn = lambda: v3_dequantize_group_absmax(payload, scales, 4, 64, DTYPE)  # noqa: E731
            ts = []
            for it in range(5 + 30):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                if it >= 5:
                    ts.append(e0.elapsed_time(e1))
            qd[label] = statistics.median(ts)
        qd["per_step_quant_overhead_ms"] = 2 * (qd["quant_int4"]
                                                + qd["dequant_int4"])
        del h

        row = {"stats": stats, "quant_breakdown_ms": qd}
        if N == 8192:
            for kind in ("median_ms", "p10_ms"):
                r_v2 = stats["v3_int4"][kind] / stats["v2_eff_weight"][kind]
                r_std = stats["v3_int4"][kind] / stats["standard_peft"][kind]
                row[f"target_{kind}"] = {
                    "int4_over_v2": r_v2, "int4_over_std": r_std,
                    "pass": r_v2 <= 0.70 and r_std <= 1.35,
                }
            ok_all &= row["target_median_ms"]["pass"]
        out[str(N)] = row

        log(f"  N={N} (median / p10 ms):")
        for name in arms:
            s = stats[name]
            log(f"    {name:17s}: {s['median_ms']:8.3f} / {s['p10_ms']:8.3f}")
        log(f"    quant int4 {qd['quant_int4']:.3f} ms, dequant "
            f"{qd['dequant_int4']:.3f} ms -> per-step quant overhead "
            f"{qd['per_step_quant_overhead_ms']:.3f} ms")
        if N == 8192:
            t = row["target_median_ms"]
            log(f"    targets @8192 (median): int4/v2={t['int4_over_v2']:.3f} "
                f"(<=0.70), int4/std={t['int4_over_std']:.3f} (<=1.35) -> "
                f"{'PASS' if t['pass'] else 'FAIL'}")
            t = row["target_p10_ms"]
            log(f"    targets @8192 (p10):    int4/v2={t['int4_over_v2']:.3f}, "
                f"int4/std={t['int4_over_std']:.3f} -> "
                f"{'PASS' if t['pass'] else 'FAIL'}")
        del x_vals, gy, arm_x
        gc.collect()
        torch.cuda.empty_cache()

    save_json("v4_speed.json", {"pass": ok_all, "warmup": warmup,
                                "iters": iters, "results": out})
    log(f"V4 OVERALL (prototype targets, median @8192): "
        f"{'PASS' if ok_all else 'FAIL'}")
    return ok_all, out


# ---------------------------------------------------------------------------
# V5 — end-to-end smoke (TinyLlama + PEFT LoRA, wikitext-2, 30 steps)
# ---------------------------------------------------------------------------

def gate_v5(args, device):
    log("\n=== V5: e2e smoke (TinyLlama + LoRA r8 MLP-only, wikitext-2) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    n_steps = 5 if args.quick else 30
    bs, seq = 8, 512
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    batches = build_batches(tok, n_steps, bs, seq)

    arm_results = {}
    for arm in ("baseline", "v3_int4", "v3_recompute"):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
        model.config.use_cache = False
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=RANK,
                         lora_alpha=ALPHA, lora_dropout=0.0,
                         target_modules=["gate_proj", "up_proj", "down_proj"])
        # adapters kept bf16 so the only difference between arms is the FFN path
        model = get_peft_model(model, cfg, autocast_adapter_dtype=False)
        model.train()

        n_patched = 0
        if arm != "baseline":
            mode = "int4" if arm == "v3_int4" else "recompute"
            for _, module in model.named_modules():
                if module.__class__.__name__ == "LlamaMLP":
                    module.forward = make_v3_forward(module, mode)
                    n_patched += 1
            assert n_patched == model.config.num_hidden_layers, n_patched

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=5e-4)
        v3_reset_counters()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses, step_times = [], []
        for step in range(n_steps):
            ids = batches[step].to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=ids, labels=ids)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach()))
        peak = torch.cuda.max_memory_allocated()
        counters = dict(_V3_COUNTERS)
        if arm == "baseline":
            assert counters["forward"] == 0, "v3 ran in baseline arm!"
        else:
            expected = n_patched * n_steps
            assert counters["forward"] == expected \
                and counters["backward"] == expected, \
                f"v3 path incomplete: {counters} vs {expected}"
        arm_results[arm] = {
            "losses": losses,
            "loss_sum_10": sum(losses[:10]),
            "loss_sum_20": sum(losses[:20]),
            "loss_sum_30": sum(losses[:n_steps]),
            "peak_mem_mib": peak / (1024 * 1024),
            "median_step_s": statistics.median(step_times),
            "mean_step_s": statistics.fmean(step_times),
            "step_times": step_times,
            "v3_counters": counters,
            "n_patched": n_patched,
        }
        r = arm_results[arm]
        log(f"  {arm:13s}: loss sums 10/20/30 = {r['loss_sum_10']:.4f} / "
            f"{r['loss_sum_20']:.4f} / {r['loss_sum_30']:.4f} | peak "
            f"{r['peak_mem_mib']:.0f} MiB | {r['median_step_s']:.3f} s/step")
        del model, opt, out, loss
        gc.collect()
        torch.cuda.empty_cache()

    base = arm_results["baseline"]
    ok_all = True
    checks = {}
    for arm in ("v3_int4", "v3_recompute"):
        r = arm_results[arm]
        loss_rel = abs(r["loss_sum_30"] - base["loss_sum_30"]) / base["loss_sum_30"]
        mem_save = 1 - r["peak_mem_mib"] / base["peak_mem_mib"]
        ok = loss_rel < 0.02 and r["peak_mem_mib"] < base["peak_mem_mib"]
        ok_all &= ok
        checks[arm] = {"loss_sum30_rel_diff": loss_rel,
                       "peak_mem_savings": mem_save, "pass": ok}
        log(f"  {arm}: loss diff {loss_rel * 100:.2f}% | peak savings "
            f"{mem_save * 100:.1f}% -> {'PASS' if ok else 'FAIL'}")

    save_json("v5_e2e.json", {"pass": ok_all, "config": {
        "model": MODEL_NAME, "steps": n_steps, "batch": bs, "seq": seq,
        "lr": 5e-4, "optimizer": "AdamW", "dtype": "bf16",
        "lora": {"r": RANK, "alpha": ALPHA, "dropout": 0.0,
                 "targets": ["gate_proj", "up_proj", "down_proj"]}},
        "arms": arm_results, "checks": checks})
    log(f"V5 OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, arm_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="+",
                    default=["v1", "v2", "v3", "v4", "v5"])
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run benchmarks even if V1/V2 fail")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = "cuda"
    log(f"device: {torch.cuda.get_device_name(0)} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
        f"torch {torch.__version__}")

    summary = {}
    t0 = time.time()
    correctness_ok = True
    for gate in args.gates:
        if gate in ("v3", "v4", "v5") and not correctness_ok and not args.force:
            log(f"\nSKIPPING {gate}: V1/V2 failed (use --force to override)")
            summary[gate] = "skipped (V1/V2 failed)"
            continue
        fn = {"v1": gate_v1, "v2": gate_v2, "v3": gate_v3,
              "v4": gate_v4, "v5": gate_v5}[gate]
        ok, detail = fn(args, device)
        if gate == "v1" and ok and not detail["strict"]:
            summary[gate] = "FAIL(strict atol)/PASS(attributed)"
        else:
            summary[gate] = "PASS" if ok else "FAIL"
        if gate in ("v1", "v2") and not ok:
            correctness_ok = False

    log("\n" + "=" * 60)
    log("GATE SUMMARY: " + " | ".join(f"{g.upper()}={s}"
                                      for g, s in summary.items()))
    log(f"total wall time: {time.time() - t0:.0f}s")
    save_json("gate_summary.json", {"summary": summary,
                                    "args": vars(args),
                                    "wall_s": time.time() - t0})


if __name__ == "__main__":
    main()
