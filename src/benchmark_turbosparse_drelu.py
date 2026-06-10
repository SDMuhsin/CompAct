"""
TurboSparse-Mistral (dReLU / Bamboo) FlashFFN benchmark + anomaly suite.

Isolated FFN-level comparison of the real masked-dReLU MLP vs FlashDReLUFFN at
the actual TurboSparse-Mistral-Instruct dimensions, using REAL activations
captured from the loaded 7B model (so the predictor sparsity is realistic).

Measures, per N = batch*seq:
  - peak memory (forward+backward), activation bytes stored for backward, time
  - gradient cosine similarity (grad_x / grad_w_gate / grad_w_up exact;
    grad_w_down approximate via top-K reconstruction)
Anomaly checks throughout: forward equivalence vs the real BambooMLP, NaN/inf
in every gradient, and the measured dReLU + hard-mask sparsity.

The predictor is a FROZEN sparsity oracle in both arms. The baseline replicates
BambooMLP.forward EXACTLY (incl. straight-through round + predictor graph);
FlashDReLU recomputes the mask in backward and stop-grads the predictor path.

Run:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 HF_HOME=./data TORCH_HOME=./data PYTHONPATH=src \
        python src/benchmark_turbosparse_drelu.py
"""

import sys, gc, json, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from flashffn import FlashDReLUFFNFunction, compute_bamboo_mask

MODEL = "PowerInfer/TurboSparse-Mistral-Instruct"
DTYPE = torch.bfloat16
K_FRACTION = 0.3
CAPTURE_LAYER = 8                 # mid-stack layer for representative sparsity
N_VALUES = [2048, 4096, 8192, 16384]
RESULTS = "results/turbosparse_drelu"


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def bad(t):
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def baseline_drelu(x, before_norm, wg, wu, wd, pfc1, pfc2):
    """Exact replica of BambooMLP.forward (masked dReLU + straight-through mask).

    Predictor weights pfc1/pfc2 are frozen; before_norm carries grad, so the
    predictor graph + mask are stored for backward exactly like the real model.
    """
    h = F.relu(F.linear(x, wg)) * F.relu(F.linear(x, wu))
    m = torch.sigmoid(F.linear(F.relu(F.linear(before_norm, pfc1)), pfc2))
    hard = torch.round(m)
    mask = m + (hard - m).detach()       # straight-through estimator
    h = mask * h
    return F.linear(h, wd)


def flash_drelu(x, before_norm, wg, wu, wd, pfc1, pfc2):
    return FlashDReLUFFNFunction.apply(x, wg, wu, wd, K_FRACTION, before_norm, pfc1, pfc2)


def measure(fn, x, before_norm, wg, wu, wd, pfc1, pfc2, do_backward=True):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**2
    t0 = time.time()
    y = fn(x, before_norm, wg, wu, wd, pfc1, pfc2)
    torch.cuda.synchronize()
    mem_after_fwd = torch.cuda.memory_allocated() / 1024**2
    peak_fwd = torch.cuda.max_memory_allocated() / 1024**2
    grads = None
    if do_backward:
        g = torch.ones_like(y)
        y.backward(g)
        torch.cuda.synchronize()
        grads = {
            "x": x.grad.detach().clone(),
            "wg": wg.grad.detach().clone(),
            "wu": wu.grad.detach().clone(),
            "wd": wd.grad.detach().clone(),
        }
    peak_total = torch.cuda.max_memory_allocated() / 1024**2
    dt = time.time() - t0
    out = y.detach().clone()
    return {
        "stored_mb": mem_after_fwd - mem_before,
        "peak_fwd_mb": peak_fwd,
        "peak_total_mb": peak_total,
        "time_s": dt,
        "y": out,
        "grads": grads,
        "y_bad": bad(out),
    }


def fresh_leaf(t):
    return t.detach().clone().requires_grad_(True)


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); sys.exit(1)
    os.makedirs(RESULTS, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("TurboSparse-Mistral dReLU FlashFFN benchmark")
    print("=" * 78)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=DTYPE, attn_implementation="eager"
    ).to("cuda").eval()
    cfg = model.config
    H, I = cfg.hidden_size, cfg.intermediate_size
    print(f"model={type(model).__name__}  hidden={H}  intermediate={I}  layers={cfg.num_hidden_layers}")

    # freeze predictors (sparsity oracle)
    nfz = 0
    for n, p in model.named_parameters():
        if ".predictor." in n:
            p.requires_grad_(False); nfz += 1
    print(f"froze {nfz} predictor params")

    mlp = model.model.layers[CAPTURE_LAYER].mlp
    print(f"capture layer {CAPTURE_LAYER}: {type(mlp).__name__}, predictor hidden="
          f"{mlp.predictor.fc1.weight.shape[0]}")

    # ---- capture REAL (x, before_norm) inputs to this MLP ----
    captured = {}
    def pre_hook(module, args):
        captured["x"] = args[0].detach()
        captured["before_norm"] = args[1].detach()
    h = mlp.register_forward_pre_hook(pre_hook)
    text = ("In a shocking finding, scientists discovered a herd of unicorns living in a "
            "remote, previously unexplored valley in the Andes Mountains. Even more "
            "surprising to the researchers was the fact that the unicorns spoke perfect "
            "English. The scientists named the population after the valley.") * 4
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    enc = {k: v.repeat(2, 1) for k, v in enc.items()}     # batch=2
    with torch.no_grad():
        model(**enc)
    h.remove()
    x_cap = captured["x"].reshape(-1, H)                   # [n_tok, H]
    bn_cap = captured["before_norm"].reshape(-1, H)
    n_tok = x_cap.shape[0]
    print(f"captured {n_tok} real tokens  (x {tuple(x_cap.shape)}, before_norm {tuple(bn_cap.shape)})")

    # clone to independent tensors so they survive deleting the 7B model
    wg0 = mlp.gate_proj.weight.detach().clone()
    wu0 = mlp.up_proj.weight.detach().clone()
    wd0 = mlp.down_proj.weight.detach().clone()
    pfc1 = mlp.predictor.fc1.weight.detach().clone()
    pfc2 = mlp.predictor.fc2.weight.detach().clone()
    x_cap = x_cap.clone(); bn_cap = bn_cap.clone()

    # ---- real sparsity on captured tokens ----
    with torch.no_grad():
        hmid = F.relu(F.linear(x_cap, wg0)) * F.relu(F.linear(x_cap, wu0))
        mask = compute_bamboo_mask(bn_cap, pfc1, pfc2)
        drelu_sp = (hmid == 0).float().mean().item()
        masked_sp = ((hmid * mask) == 0).float().mean().item()
        k = max(1, int(I * K_FRACTION))
        nnz = (hmid * mask != 0).sum(-1)
        captured_by_topk = (nnz <= k).float().mean().item()
    print(f"\nReal sparsity @ layer {CAPTURE_LAYER}: dReLU zeros={drelu_sp:.1%}  "
          f"after hard mask={masked_sp:.1%}  | top-K k={k}  max nnz/row={nnz.max().item()}  "
          f"rows fully captured={captured_by_topk:.1%}")

    # ---- forward equivalence: real BambooMLP vs replica vs flash ----
    print("\n[Anomaly] Forward equivalence (real BambooMLP vs replica vs flash)")
    xb = x_cap.reshape(2, -1, H); bnb = bn_cap.reshape(2, -1, H)
    with torch.no_grad():
        y_real = mlp(xb, bnb).reshape(-1, H)
        y_repl = baseline_drelu(x_cap, bn_cap, wg0, wu0, wd0, pfc1, pfc2)
        y_flash = flash_drelu(x_cap, bn_cap, wg0, wu0, wd0, pfc1, pfc2)
    d_repl = (y_real - y_repl).abs().max().item()
    d_flash = (y_real - y_flash).abs().max().item()
    rel = (y_real - y_flash).abs().max().item() / (y_real.abs().max().item() + 1e-9)
    print(f"    max|real-replica| = {d_repl:.3e}   (replica faithfulness)")
    print(f"    max|real-flash|   = {d_flash:.3e}   rel={rel:.3e}   "
          f"NaN(flash)={bad(y_flash)}  -> {'PASS' if (d_flash < 0.05 and not bad(y_flash)) else 'FAIL'}")

    # free the 7B model so isolated FFN peak memory is not dominated by its weights
    del mlp, model, y_real, y_repl, y_flash, hmid, mask
    gc.collect(); torch.cuda.empty_cache()

    # ---- memory / speed / grad sweep ----
    print(f"\n[Benchmark] masked dReLU FFN, model freed  (H={H}, I={I}, dtype={DTYPE})")
    header = (f"{'N':>7} | {'base peak':>10} {'flash peak':>10} {'save%':>6} | "
              f"{'base stored':>11} {'flash stored':>12} {'save%':>6} | "
              f"{'base t':>7} {'flash t':>7} | {'gx':>6} {'wg':>6} {'wu':>6} {'wd':>6} | anom")
    print(header); print("-" * len(header))
    rows = []
    for N in N_VALUES:
        # tile real captured tokens up to N (keeps realistic sparsity)
        reps = (N + n_tok - 1) // n_tok
        xt = x_cap.repeat(reps, 1)[:N].contiguous()
        bnt = bn_cap.repeat(reps, 1)[:N].contiguous()

        # baseline
        xb_ = fresh_leaf(xt); bn_b = fresh_leaf(bnt)
        wgb, wub, wdb = fresh_leaf(wg0), fresh_leaf(wu0), fresh_leaf(wd0)
        try:
            rb = measure(baseline_drelu, xb_, bn_b, wgb, wub, wdb, pfc1, pfc2)
        except torch.cuda.OutOfMemoryError:
            print(f"{N:>7} | baseline OOM"); rows.append({"N": N, "baseline_oom": True});
            gc.collect(); torch.cuda.empty_cache(); continue

        # flash
        xf_ = fresh_leaf(xt); bn_f = fresh_leaf(bnt)
        wgf, wuf, wdf = fresh_leaf(wg0), fresh_leaf(wu0), fresh_leaf(wd0)
        rf = measure(flash_drelu, xf_, bn_f, wgf, wuf, wdf, pfc1, pfc2)

        gx = cos(rf["grads"]["x"], rb["grads"]["x"])
        gwg = cos(rf["grads"]["wg"], rb["grads"]["wg"])
        gwu = cos(rf["grads"]["wu"], rb["grads"]["wu"])
        gwd = cos(rf["grads"]["wd"], rb["grads"]["wd"])
        anom = (rf["y_bad"] or any(bad(v) for v in rf["grads"].values())
                or bad(rb["grads"]["wd"]))
        peak_save = 100 * (1 - rf["peak_total_mb"] / rb["peak_total_mb"])
        stored_save = 100 * (1 - rf["stored_mb"] / rb["stored_mb"]) if rb["stored_mb"] > 0 else float('nan')
        print(f"{N:>7} | {rb['peak_total_mb']:>10.0f} {rf['peak_total_mb']:>10.0f} {peak_save:>5.1f}% | "
              f"{rb['stored_mb']:>11.0f} {rf['stored_mb']:>12.0f} {stored_save:>5.1f}% | "
              f"{rb['time_s']:>7.3f} {rf['time_s']:>7.3f} | "
              f"{gx:>6.3f} {gwg:>6.3f} {gwu:>6.3f} {gwd:>6.3f} | {'BAD' if anom else 'ok'}")
        rows.append({
            "N": N, "base_peak_mb": rb["peak_total_mb"], "flash_peak_mb": rf["peak_total_mb"],
            "peak_save_pct": peak_save, "base_stored_mb": rb["stored_mb"],
            "flash_stored_mb": rf["stored_mb"], "stored_save_pct": stored_save,
            "base_time_s": rb["time_s"], "flash_time_s": rf["time_s"],
            "cos_grad_x": gx, "cos_grad_wgate": gwg, "cos_grad_wup": gwu, "cos_grad_wdown": gwd,
            "anomaly": anom,
        })
        del rb, rf, xb_, bn_b, wgb, wub, wdb, xf_, bn_f, wgf, wuf, wdf
        gc.collect(); torch.cuda.empty_cache()

    # ---- projection: stored-activation memory across ALL layers (where peak savings live) ----
    L = cfg.num_hidden_layers
    print(f"\n[Projection] stored backward-activation memory across all {L} layers:")
    print(f"{'N':>7} | {'baseline GB':>12} {'flash GB':>10} {'saved GB':>9} {'save%':>6}")
    proj = []
    for r in rows:
        if r.get("baseline_oom"):
            continue
        b = r["base_stored_mb"] * L / 1024
        f = r["flash_stored_mb"] * L / 1024
        print(f"{r['N']:>7} | {b:>12.2f} {f:>10.2f} {b-f:>9.2f} {100*(1-f/b):>5.1f}%")
        proj.append({"N": r["N"], "base_stored_all_layers_gb": b,
                     "flash_stored_all_layers_gb": f, "saved_gb": b - f})

    summary = {
        "model": MODEL, "hidden": H, "intermediate": I, "num_layers": L, "dtype": str(DTYPE),
        "k_fraction": K_FRACTION, "capture_layer": CAPTURE_LAYER,
        "drelu_sparsity": drelu_sp, "masked_sparsity": masked_sp,
        "rows_fully_captured_by_topk": captured_by_topk,
        "forward_max_abs_diff_real_vs_flash": d_flash,
        "results": rows,
        "all_layer_projection": proj,
    }
    out = os.path.join(RESULTS, "benchmark.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
