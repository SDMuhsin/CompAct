"""
Phase: v3 profiling — WHERE TIME AND MEMORY GO in FlashFFN recompute mode.

Produces a precise per-op breakdown of FlashFFNFunction (recompute mode) vs the
standard SwiGLU FFN to identify optimization targets.

Experiments
-----------
E1  Per-op time breakdown of one fwd+bwd through FlashFFNFunction in RECOMPUTE
    mode, via a faithfully replicated op sequence (verified bitwise/allclose
    against the real FlashFFNFunction.apply before timings are trusted), plus
    the real apply end-to-end and StandardFFN as baseline.
E2  Isolated cost of abs + torch.topk + gather + int16 cast on a precomputed
    h_mid (fraction of the total flash step).
E3  Stored-for-backward bytes (torch.autograd.graph.saved_tensors_hooks around
    the REAL apply) + peak transient memory during backward, flash vs standard.
E4  LoRA effective-weight scenario: W_eff = W_base + 0.5*(B@A), r=8, gradients
    through B/A only. Counts the 3 extra weight-sized W_eff tensors saved and
    compares against a PEFT-style LoRA baseline's saved activations.
E5  (separate invocation: --append-e5) parses the train_glue.py A/B CSV and
    appends an E5 section to the markdown summary.

Usage (from project root, env activated, HF_HOME/TORCH_HOME -> ./data):
    CUDA_VISIBLE_DEVICES=<idle gpu> python src/profile_flashffn_v3.py
    python src/profile_flashffn_v3.py --append-e5 results/v3_profiling/e5.csv

Outputs: results/v3_profiling/microbench.json, results/v3_profiling/summary.md
"""

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
from contextlib import contextmanager
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashffn import (  # noqa: E402
    FlashFFNFunction,
    triton_silu_mul,
    triton_swiglu_backward,
)

DTYPE = torch.bfloat16
K_FRACTION = 0.3
SHAPES = [
    # (label, hidden_dim, intermediate_dim)
    ("tinyllama", 2048, 5632),
    ("llama7b", 4096, 11008),
]
N_TOKENS = [2048, 8192, 16384]
WARMUP_ITERS = 15
TIMED_ITERS = 150
NOISY_STD_FRAC = 0.10  # flag measurements with std/mean > 10%

# NOTE on noise handling: this dev box is SHARED — both GPUs run a bursty
# co-tenant workload (sampled 0-100% util in 30 s). Headline numbers therefore
# use the MEDIAN over many iterations (robust to contention bursts) and we
# additionally report min / p10 (uncontended floor) and mean/std. Ops where
# even the interquartile range exceeds 10% of the median are flagged NOISY.

OUT_DIR = os.path.join("results", "v3_profiling")


# =============================================================================
# Timing utilities
# =============================================================================

class EventTimer:
    """CUDA-event based per-op timer. Events are pre-allocated; results are
    collected after a single torch.cuda.synchronize()."""

    def __init__(self, names, iters):
        self.names = list(names)
        self.iters = iters
        self.events = {
            n: [(torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for n in self.names
        }
        self.i = -1

    def next_iter(self):
        self.i += 1

    @contextmanager
    def time(self, name):
        start, end = self.events[name][self.i]
        start.record()
        yield
        end.record()

    def results(self):
        torch.cuda.synchronize()
        out = {}
        for n in self.names:
            ts = sorted(s.elapsed_time(e) for s, e in self.events[n][: self.i + 1])
            mean = statistics.mean(ts)
            std = statistics.stdev(ts) if len(ts) > 1 else 0.0
            med = statistics.median(ts)
            p10 = ts[max(0, int(0.10 * len(ts)) - 1)] if len(ts) >= 10 else ts[0]
            p25 = ts[max(0, int(0.25 * len(ts)) - 1)]
            p75 = ts[min(len(ts) - 1, int(0.75 * len(ts)))]
            iqr_frac = (p75 - p25) / med if med > 0 else 0.0
            out[n] = {
                "median_ms": med,
                "min_ms": ts[0],
                "p10_ms": p10,
                "mean_ms": mean,
                "std_ms": std,
                "iqr_over_median": iqr_frac,
                "std_over_mean": (std / mean) if mean > 0 else 0.0,
                "noisy": bool(iqr_frac > NOISY_STD_FRAC),
                "n": len(ts),
            }
        return out


def fmt_bytes(b):
    return f"{b / 1024**2:.2f} MiB"


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Replica of FlashFFNFunction recompute-mode op sequence
# (faithful copy of src/flashffn.py lines 169-191 fwd / 210-289 bwd;
#  verified against the real implementation before timing is trusted)
# =============================================================================

def replica_forward(x_2d, w_gate, w_up, w_down, k_fraction, timer=None):
    """Exact op sequence of FlashFFNFunction.forward (recompute mode), 2-D x.
    abs() is timed separately from topk but is mathematically identical to the
    inline h_mid.abs() in the source."""
    intermediate_dim = w_gate.shape[0]

    def t(name):
        return timer.time(name) if timer is not None else _nullctx()

    with t("fwd.gate_linear"):
        h_gate = F.linear(x_2d, w_gate)              # src L172
    with t("fwd.up_linear"):
        h_up = F.linear(x_2d, w_up)                  # src L173
    with t("fwd.silu_mul_triton"):
        h_mid = triton_silu_mul(h_gate, h_up)        # src L174
    with t("fwd.down_linear"):
        y = F.linear(h_mid, w_down)                  # src L176
    # recompute-mode save path (src L186-190)
    k = max(1, int(intermediate_dim * k_fraction))
    with t("fwd.abs"):
        h_abs = h_mid.abs()
    with t("fwd.topk"):
        _, top_indices = torch.topk(h_abs, k, dim=-1, sorted=False)
    with t("fwd.gather_values"):
        top_values = torch.gather(h_mid, dim=-1, index=top_indices)
    with t("fwd.cast_int16"):
        all_indices = top_indices.to(torch.int16)
    return y, (top_values, all_indices), k


def replica_backward(grad_output_2d, x_2d, w_gate, w_up, w_down,
                     values, indices, intermediate_dim, timer=None):
    """Exact op sequence of FlashFFNFunction.backward (recompute mode), with
    all needs_input_grad True. indices.long() is timed separately but is the
    same inline op as in the source (src L256)."""
    batch_seq = x_2d.shape[0]

    def t(name):
        return timer.time(name) if timer is not None else _nullctx()

    # --- grad_w_down via sparse h_mid reconstruction (src L252-257) ---
    with t("bwd.zeros_hmid_sparse"):
        h_mid_sparse = torch.zeros(
            batch_seq, intermediate_dim, device=x_2d.device, dtype=x_2d.dtype
        )
    with t("bwd.indices_long"):
        idx_long = indices.long()
    with t("bwd.scatter"):
        h_mid_sparse.scatter_(dim=-1, index=idx_long, src=values)
    with t("bwd.grad_w_down_mm"):
        grad_w_down = grad_output_2d.t() @ h_mid_sparse

    # --- exact gradients via forward recomputation (src L263-280) ---
    with t("bwd.recompute_gate_linear"):
        h_gate = F.linear(x_2d, w_gate)
    with t("bwd.recompute_up_linear"):
        h_up = F.linear(x_2d, w_up)
    with t("bwd.grad_h_mid_mm"):
        grad_h_mid = grad_output_2d @ w_down
    with t("bwd.triton_swiglu_bwd"):
        grad_h_gate, grad_h_up = triton_swiglu_backward(grad_h_mid, h_gate, h_up)
    with t("bwd.grad_w_gate_mm"):
        grad_w_gate = grad_h_gate.t() @ x_2d
    with t("bwd.grad_w_up_mm"):
        grad_w_up = grad_h_up.t() @ x_2d
    with t("bwd.grad_x_mms"):
        grad_x = grad_h_gate @ w_gate + grad_h_up @ w_up
    return grad_x, grad_w_gate, grad_w_up, grad_w_down


@contextmanager
def _nullctx():
    yield


FWD_OPS = [
    "fwd.gate_linear", "fwd.up_linear", "fwd.silu_mul_triton",
    "fwd.down_linear", "fwd.abs", "fwd.topk", "fwd.gather_values",
    "fwd.cast_int16",
]
BWD_OPS = [
    "bwd.zeros_hmid_sparse", "bwd.indices_long", "bwd.scatter",
    "bwd.grad_w_down_mm", "bwd.recompute_gate_linear",
    "bwd.recompute_up_linear", "bwd.grad_h_mid_mm", "bwd.triton_swiglu_bwd",
    "bwd.grad_w_gate_mm", "bwd.grad_w_up_mm", "bwd.grad_x_mms",
]


# =============================================================================
# Test data / verification
# =============================================================================

def make_tensors(n_tokens, d_hid, d_int, device, requires_grad=True):
    torch.manual_seed(41)
    x = torch.randn(n_tokens, d_hid, device=device, dtype=DTYPE) * 0.5
    w_gate = torch.randn(d_int, d_hid, device=device, dtype=DTYPE) * (1.0 / math.sqrt(d_hid))
    w_up = torch.randn(d_int, d_hid, device=device, dtype=DTYPE) * (1.0 / math.sqrt(d_hid))
    w_down = torch.randn(d_hid, d_int, device=device, dtype=DTYPE) * (1.0 / math.sqrt(d_int))
    g = torch.randn(n_tokens, d_hid, device=device, dtype=DTYPE)
    if requires_grad:
        x.requires_grad_(True)
        for w in (w_gate, w_up, w_down):
            w.requires_grad_(True)
    return x, w_gate, w_up, w_down, g


def verify_replica(x, w_gate, w_up, w_down, g, d_int):
    """Assert the replica op sequence reproduces FlashFFNFunction.apply
    (forward bitwise-equal; gradients bitwise or allclose). Returns dict."""
    # Real path (recompute mode: weights require grad)
    assert w_gate.requires_grad and w_up.requires_grad and w_down.requires_grad
    for t in (x, w_gate, w_up, w_down):
        t.grad = None
    y_real = FlashFFNFunction.apply(x, w_gate, w_up, w_down, K_FRACTION)
    # Confirm RECOMPUTE mode actually triggered (honesty check on mode select)
    saved = y_real.grad_fn.saved_tensors
    assert len(saved) == 6, (
        f"Expected 6 saved tensors in recompute mode, got {len(saved)} — "
        "mode-selection logic may have changed!"
    )
    y_real.backward(g)
    real_grads = {
        "x": x.grad.detach().clone(),
        "w_gate": w_gate.grad.detach().clone(),
        "w_up": w_up.grad.detach().clone(),
        "w_down": w_down.grad.detach().clone(),
    }

    # Replica path (no autograd)
    with torch.no_grad():
        y_rep, (values, indices), _k = replica_forward(
            x.detach(), w_gate.detach(), w_up.detach(), w_down.detach(), K_FRACTION
        )
        gx, gwg, gwu, gwd = replica_backward(
            g, x.detach(), w_gate.detach(), w_up.detach(), w_down.detach(),
            values, indices, d_int,
        )
    rep_grads = {"x": gx, "w_gate": gwg, "w_up": gwu, "w_down": gwd}

    fwd_bitwise = torch.equal(y_real, y_rep)
    assert fwd_bitwise, "Replica forward is NOT bitwise-equal to FlashFFNFunction.apply"

    grad_report = {}
    for name in real_grads:
        bitwise = torch.equal(real_grads[name], rep_grads[name])
        if not bitwise:
            a = real_grads[name].float()
            b = rep_grads[name].float()
            ok = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
            max_abs = (a - b).abs().max().item()
            assert ok, f"Replica grad_{name} not allclose to real (max abs diff {max_abs})"
            grad_report[name] = {"bitwise": False, "max_abs_diff": max_abs}
        else:
            grad_report[name] = {"bitwise": True}
    for t in (x, w_gate, w_up, w_down):
        t.grad = None
    return {"forward_bitwise": fwd_bitwise, "grads": grad_report}


# =============================================================================
# E1: per-op timing
# =============================================================================

def run_e1(x, w_gate, w_up, w_down, g, d_int, device):
    """Per-op replica timing + real apply + StandardFFN baseline.

    All three variants are INTERLEAVED inside the same iteration so that
    co-tenant contention bursts hit them equally (keeps ratios honest)."""
    n_tokens = x.shape[0]
    xd, wg, wu, wd = x.detach(), w_gate.detach(), w_up.detach(), w_down.detach()

    def standard_forward(xi):
        h_gate = F.linear(xi, w_gate)
        h_up = F.linear(xi, w_up)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up
        return F.linear(h_mid, w_down)

    def clear_grads():
        for t in (x, w_gate, w_up, w_down):
            t.grad = None  # avoid grad-accumulation add kernels

    names = (FWD_OPS + BWD_OPS
             + ["replica.fwd_total", "replica.bwd_total",
                "real.fwd", "real.bwd", "std.fwd", "std.bwd"])
    timer = EventTimer(names, TIMED_ITERS)

    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            y, (v, i), _ = replica_forward(xd, wg, wu, wd, K_FRACTION)
            replica_backward(g, xd, wg, wu, wd, v, i, d_int)
        clear_grads()
        y = FlashFFNFunction.apply(x, w_gate, w_up, w_down, K_FRACTION)
        y.backward(g)
        clear_grads()
        y = standard_forward(x)
        y.backward(g)
    torch.cuda.synchronize()

    for _ in range(TIMED_ITERS):
        timer.next_iter()
        # (1) replica per-op + bracketed totals (pure kernels, no autograd)
        with torch.no_grad():
            with timer.time("replica.fwd_total"):
                y, (v, i), _ = replica_forward(xd, wg, wu, wd, K_FRACTION, timer=timer)
            with timer.time("replica.bwd_total"):
                replica_backward(g, xd, wg, wu, wd, v, i, d_int, timer=timer)
            del y, v, i
        # (2) real FlashFFNFunction.apply end-to-end (ground truth)
        clear_grads()
        with timer.time("real.fwd"):
            y = FlashFFNFunction.apply(x, w_gate, w_up, w_down, K_FRACTION)
        with timer.time("real.bwd"):
            y.backward(g)
        del y
        # (3) StandardFFN baseline (plain autograd SwiGLU)
        clear_grads()
        with timer.time("std.fwd"):
            y = standard_forward(x)
        with timer.time("std.bwd"):
            y.backward(g)
        del y

    res = timer.results()
    replica_res = {n: res[n] for n in FWD_OPS + BWD_OPS
                   + ["replica.fwd_total", "replica.bwd_total"]}
    real_res = {n: res[n] for n in ["real.fwd", "real.bwd"]}
    std_res = {n: res[n] for n in ["std.fwd", "std.bwd"]}
    cleanup()

    flash_total = real_res["real.fwd"]["median_ms"] + real_res["real.bwd"]["median_ms"]
    std_total = std_res["std.fwd"]["median_ms"] + std_res["std.bwd"]["median_ms"]
    flash_total_p10 = real_res["real.fwd"]["p10_ms"] + real_res["real.bwd"]["p10_ms"]
    std_total_p10 = std_res["std.fwd"]["p10_ms"] + std_res["std.bwd"]["p10_ms"]
    replica_total = (replica_res["replica.fwd_total"]["median_ms"]
                     + replica_res["replica.bwd_total"]["median_ms"])
    op_sum = sum(replica_res[n]["median_ms"] for n in FWD_OPS + BWD_OPS)

    return {
        "n_tokens": n_tokens,
        "replica_ops": replica_res,
        "real": real_res,
        "standard": std_res,
        "totals": {
            "flash_real_total_ms": flash_total,
            "standard_total_ms": std_total,
            "flash_real_total_p10_ms": flash_total_p10,
            "standard_total_p10_ms": std_total_p10,
            "replica_total_ms": replica_total,
            "replica_op_sum_ms": op_sum,
            "flash_over_standard": flash_total / std_total,
            "flash_over_standard_p10": flash_total_p10 / std_total_p10,
            "replica_vs_real_ratio": replica_total / flash_total,
        },
    }


# =============================================================================
# E2: isolated topk pipeline cost on precomputed h_mid
# =============================================================================

def run_e2(x, w_gate, w_up, d_int, flash_total_ms):
    with torch.no_grad():
        h_gate = F.linear(x.detach(), w_gate.detach())
        h_up = F.linear(x.detach(), w_up.detach())
        h_mid = triton_silu_mul(h_gate, h_up)
        del h_gate, h_up
    k = max(1, int(d_int * K_FRACTION))

    names = ["e2.abs", "e2.topk", "e2.gather", "e2.cast_int16"]
    timer = EventTimer(names, TIMED_ITERS)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            h_abs = h_mid.abs()
            _, idx = torch.topk(h_abs, k, dim=-1, sorted=False)
            vals = torch.gather(h_mid, dim=-1, index=idx)
            idx.to(torch.int16)
        torch.cuda.synchronize()
        for _ in range(TIMED_ITERS):
            timer.next_iter()
            with timer.time("e2.abs"):
                h_abs = h_mid.abs()
            with timer.time("e2.topk"):
                _, idx = torch.topk(h_abs, k, dim=-1, sorted=False)
            with timer.time("e2.gather"):
                vals = torch.gather(h_mid, dim=-1, index=idx)
            with timer.time("e2.cast_int16"):
                idx16 = idx.to(torch.int16)
            del h_abs, idx, vals, idx16
    res = timer.results()
    del h_mid
    cleanup()

    pipeline_ms = sum(res[n]["median_ms"] for n in names)
    return {
        "k": k,
        "ops": res,
        "pipeline_ms": pipeline_ms,
        "topk_only_ms": res["e2.topk"]["median_ms"],
        "fraction_of_flash_total_step": pipeline_ms / flash_total_ms,
        "topk_fraction_of_flash_total_step": res["e2.topk"]["median_ms"] / flash_total_ms,
    }


# =============================================================================
# E3: stored-for-backward bytes + peak transient backward memory
# =============================================================================

def measure_saved_tensors(build_fn):
    """Run build_fn() (a forward producing y) under saved_tensors_hooks and
    return (y, records). Each record: shape/dtype/bytes/ptr."""
    records = []

    def pack(t):
        records.append({
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "bytes": t.numel() * t.element_size(),
            "ptr": t.data_ptr(),
        })
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = build_fn()
    assert len(records) > 0, "saved_tensors_hooks captured nothing — introspection failed"
    return y, records


def classify_records(records, ptr_labels):
    """Dedupe by data ptr; label tensors using known data ptrs."""
    seen = {}
    for r in records:
        if r["ptr"] not in seen:
            label = ptr_labels.get(r["ptr"], "other")
            seen[r["ptr"]] = {**r, "label": label}
    per_label = {}
    for r in seen.values():
        per_label.setdefault(r["label"], {"bytes": 0, "count": 0, "shapes": []})
        per_label[r["label"]]["bytes"] += r["bytes"]
        per_label[r["label"]]["count"] += 1
        per_label[r["label"]]["shapes"].append([list(r["shape"]), r["dtype"]])
    total_unique = sum(r["bytes"] for r in seen.values())
    raw_total = sum(r["bytes"] for r in records)
    return {
        "per_label": per_label,
        "total_unique_bytes": total_unique,
        "raw_total_bytes_with_duplicates": raw_total,
        "n_saved_calls": len(records),
        "n_unique_tensors": len(seen),
    }


def run_e3(x, w_gate, w_up, w_down, g, d_int, device):
    param_ptrs = {
        w_gate.data_ptr(): "weight:w_gate",
        w_up.data_ptr(): "weight:w_up",
        w_down.data_ptr(): "weight:w_down",
    }

    # ---------- FlashFFN recompute mode ----------
    ptr_labels = dict(param_ptrs)
    ptr_labels[x.data_ptr()] = "x"
    y, recs = measure_saved_tensors(
        lambda: FlashFFNFunction.apply(x, w_gate, w_up, w_down, K_FRACTION)
    )
    # label top-K values/indices by dtype (the only bf16/int16 non-x tensors)
    for r in recs:
        if r["ptr"] not in ptr_labels:
            if r["dtype"] == "torch.int16":
                ptr_labels[r["ptr"]] = "topk_indices_int16"
            elif r["dtype"] == str(DTYPE):
                ptr_labels[r["ptr"]] = "topk_values"
    flash_saved = classify_records(recs, ptr_labels)

    # peak transient during backward
    for t in (x, w_gate, w_up, w_down):
        t.grad = None
    torch.cuda.synchronize()
    alloc_before_bwd = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    y.backward(g)
    torch.cuda.synchronize()
    flash_bwd_peak = torch.cuda.max_memory_allocated(device)
    flash_mem = {
        "saved": flash_saved,
        "alloc_before_bwd_bytes": alloc_before_bwd,
        "bwd_peak_bytes": flash_bwd_peak,
        "bwd_transient_bytes": flash_bwd_peak - alloc_before_bwd,
    }
    del y
    for t in (x, w_gate, w_up, w_down):
        t.grad = None
    cleanup()

    # ---------- StandardFFN ----------
    def std_fwd():
        h_gate = F.linear(x, w_gate)
        h_up = F.linear(x, w_up)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up
        return F.linear(h_mid, w_down)

    ptr_labels = dict(param_ptrs)
    ptr_labels[x.data_ptr()] = "x"
    y, recs = measure_saved_tensors(std_fwd)
    # remaining tensors are the 4 intermediates; label by save order:
    # linear(gate): x,w | linear(up): x,w | silu: h_gate | mul: h_act,h_up | linear(down): h_mid,w
    inter_names = iter(["h_gate", "h_act_or_h_up", "h_act_or_h_up", "h_mid"])
    for r in recs:
        if r["ptr"] not in ptr_labels:
            ptr_labels[r["ptr"]] = "intermediate:" + next(inter_names, "extra")
    std_saved = classify_records(recs, ptr_labels)

    for t in (x, w_gate, w_up, w_down):
        t.grad = None
    torch.cuda.synchronize()
    alloc_before_bwd = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    y.backward(g)
    torch.cuda.synchronize()
    std_bwd_peak = torch.cuda.max_memory_allocated(device)
    std_mem = {
        "saved": std_saved,
        "alloc_before_bwd_bytes": alloc_before_bwd,
        "bwd_peak_bytes": std_bwd_peak,
        "bwd_transient_bytes": std_bwd_peak - alloc_before_bwd,
    }
    del y
    for t in (x, w_gate, w_up, w_down):
        t.grad = None
    cleanup()

    def nonparam_bytes(saved):
        return sum(v["bytes"] for k, v in saved["per_label"].items()
                   if not k.startswith("weight:"))

    return {
        "flash": flash_mem,
        "standard": std_mem,
        "flash_nonparam_saved_bytes": nonparam_bytes(flash_saved),
        "standard_nonparam_saved_bytes": nonparam_bytes(std_saved),
    }


# =============================================================================
# E4: LoRA effective-weight scenario
# =============================================================================

def run_e4(n_tokens, d_hid, d_int, device, lora_r=8, scaling=0.5):
    torch.manual_seed(41)
    x = torch.randn(n_tokens, d_hid, device=device, dtype=DTYPE) * 0.5
    x.requires_grad_(True)
    wb_gate = torch.randn(d_int, d_hid, device=device, dtype=DTYPE) * (1.0 / math.sqrt(d_hid))
    wb_up = torch.randn(d_int, d_hid, device=device, dtype=DTYPE) * (1.0 / math.sqrt(d_hid))
    wb_down = torch.randn(d_hid, d_int, device=device, dtype=DTYPE) * (1.0 / math.sqrt(d_int))
    # frozen base weights (parameters in the real model — no requires_grad)
    lora = {}
    for name, (dout, din) in [("gate", (d_int, d_hid)), ("up", (d_int, d_hid)),
                              ("down", (d_hid, d_int))]:
        A = (torch.randn(lora_r, din, device=device, dtype=DTYPE) * 0.02).requires_grad_(True)
        B = (torch.randn(dout, lora_r, device=device, dtype=DTYPE) * 0.02).requires_grad_(True)
        lora[name] = (A, B)
    g = torch.randn(n_tokens, d_hid, device=device, dtype=DTYPE)

    base_ptrs = {
        wb_gate.data_ptr(): "frozen_base:w_gate",
        wb_up.data_ptr(): "frozen_base:w_up",
        wb_down.data_ptr(): "frozen_base:w_down",
    }
    lora_ptrs = {}
    for name, (A, B) in lora.items():
        lora_ptrs[A.data_ptr()] = f"lora_param:A_{name}"
        lora_ptrs[B.data_ptr()] = f"lora_param:B_{name}"

    # ---------- (a) FlashFFN with effective weights ----------
    def flash_eff_fwd():
        w_eff = {}
        for name, wb in [("gate", wb_gate), ("up", wb_up), ("down", wb_down)]:
            A, B = lora[name]
            w_eff[name] = wb + scaling * (B @ A)
        return FlashFFNFunction.apply(
            x, w_eff["gate"], w_eff["up"], w_eff["down"], K_FRACTION
        ), w_eff

    ptr_labels = {**base_ptrs, **lora_ptrs, x.data_ptr(): "x"}
    records = []

    def pack(t):
        records.append({"shape": tuple(t.shape), "dtype": str(t.dtype),
                        "bytes": t.numel() * t.element_size(), "ptr": t.data_ptr()})
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y, w_eff = flash_eff_fwd()

    # sanity: recompute mode must have triggered (W_eff requires grad)
    assert all(w.requires_grad for w in w_eff.values())
    assert len(y.grad_fn.saved_tensors) == 6, "recompute mode did not trigger for W_eff"

    weff_shapes = {(d_int, d_hid): "w_eff(gate/up-sized)", (d_hid, d_int): "w_eff(down-sized)"}
    for r in records:
        if r["ptr"] not in ptr_labels:
            if r["shape"] in weff_shapes:
                ptr_labels[r["ptr"]] = "W_eff_materialized"
            elif r["dtype"] == "torch.int16":
                ptr_labels[r["ptr"]] = "topk_indices_int16"
            elif len(r["shape"]) == 2 and r["shape"][1] == max(1, int(d_int * K_FRACTION)):
                ptr_labels[r["ptr"]] = "topk_values"
    # match W_eff ptrs explicitly
    for name, w in w_eff.items():
        ptr_labels[w.data_ptr()] = f"W_eff:{name}"
    flash_eff_saved = classify_records(records, ptr_labels)
    y.backward(g)
    # honesty check: gradients flowed to LoRA params
    assert all(A.grad is not None and B.grad is not None for A, B in lora.values())
    del y, w_eff
    x.grad = None
    for A, B in lora.values():
        A.grad = None
        B.grad = None
    cleanup()

    # ---------- (b) PEFT-style LoRA baseline (no effective weight) ----------
    def lora_linear(inp, wb, A, B):
        return F.linear(inp, wb) + scaling * F.linear(F.linear(inp, A), B)

    def baseline_fwd():
        h_gate = lora_linear(x, wb_gate, *lora["gate"])
        h_up = lora_linear(x, wb_up, *lora["up"])
        h_mid = F.silu(h_gate) * h_up
        return lora_linear(h_mid, wb_down, *lora["down"])

    ptr_labels = {**base_ptrs, **lora_ptrs, x.data_ptr(): "x"}
    records = []
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = baseline_fwd()
    for r in records:
        if r["ptr"] not in ptr_labels:
            if len(r["shape"]) == 2 and r["shape"][1] == lora_r:
                ptr_labels[r["ptr"]] = "lora_intermediate(N,r)"
            else:
                ptr_labels[r["ptr"]] = "activation_intermediate"
    baseline_saved = classify_records(records, ptr_labels)
    y.backward(g)
    assert all(A.grad is not None and B.grad is not None for A, B in lora.values())
    del y
    cleanup()

    def bucket(saved, prefixes):
        return sum(v["bytes"] for k, v in saved["per_label"].items()
                   if any(k.startswith(p) for p in prefixes))

    weff_bytes = bucket(flash_eff_saved, ["W_eff"])
    flash_nonparam = sum(
        v["bytes"] for k, v in flash_eff_saved["per_label"].items()
        if not (k.startswith("frozen_base") or k.startswith("lora_param"))
    )
    baseline_nonparam = sum(
        v["bytes"] for k, v in baseline_saved["per_label"].items()
        if not (k.startswith("frozen_base") or k.startswith("lora_param"))
    )
    return {
        "n_tokens": n_tokens,
        "lora_r": lora_r,
        "flash_eff_saved": flash_eff_saved,
        "baseline_lora_saved": baseline_saved,
        "w_eff_extra_bytes": weff_bytes,
        "flash_nonparam_saved_bytes": flash_nonparam,
        "baseline_nonparam_saved_bytes": baseline_nonparam,
        "net_bytes_flash_minus_baseline": flash_nonparam - baseline_nonparam,
    }


# =============================================================================
# Markdown rendering
# =============================================================================

def render_markdown(results):
    L = []
    L.append("# FlashFFN v3 Profiling — recompute mode vs standard SwiGLU FFN")
    L.append("")
    L.append(f"Generated: {results['meta']['timestamp']}  ")
    L.append(f"GPU: {results['meta']['gpu']} | dtype: bf16 | k_fraction: {K_FRACTION} | "
             f"warmup {WARMUP_ITERS}, timed {TIMED_ITERS} iters (CUDA events)")
    L.append("")
    L.append("**Noise caveat:** the dev box is shared; both GPUs run a bursty co-tenant "
             "workload. Headline numbers are MEDIANS over iterations (robust to bursts); "
             "p10 is reported as the near-uncontended floor. Ops with IQR > 10% of the "
             "median are flagged NOISY.")
    L.append("")
    L.append("All replica timings were verified against `FlashFFNFunction.apply` "
             "(forward bitwise-equal; per-config grad equality reported below).")
    L.append("")

    if results.get("key_findings"):
        L.append("## Key findings")
        L.append("")
        for kf in results["key_findings"]:
            L.append(f"- {kf}")
        L.append("")

    for cfg in results["configs"]:
        label = cfg["label"]
        L.append(f"## {label}  (D_hid={cfg['d_hid']}, D_int={cfg['d_int']}, "
                 f"N={cfg['n_tokens']}, k={cfg['k']})")
        L.append("")
        v = cfg["verification"]
        grads_bw = all(gr.get("bitwise") for gr in v["grads"].values())
        L.append(f"Verification: forward bitwise-equal = {v['forward_bitwise']}, "
                 f"all grads bitwise-equal = {grads_bw}"
                 + ("" if grads_bw else f" (details: {v['grads']})"))
        L.append("")

        e1 = cfg["e1"]
        tot = e1["totals"]
        rep = e1["replica_ops"]
        op_sum = tot["replica_op_sum_ms"]
        op_sum_p10 = sum(rep[n]["p10_ms"] for n in FWD_OPS + BWD_OPS)
        L.append("### E1 — per-op time breakdown (replica, verified)")
        L.append("")
        L.append("Primary column is p10 (near-uncontended floor); medians are inflated "
                 "by co-tenant bursts on matmul-heavy ops.")
        L.append("")
        L.append("| op | p10 ms | % of step (p10) | median ms | IQR/median |")
        L.append("|---|---|---|---|---|")
        for n in FWD_OPS + BWD_OPS:
            r = rep[n]
            noisy = " (NOISY)" if r["noisy"] else ""
            L.append(f"| {n} | {r['p10_ms']:.3f} | {100*r['p10_ms']/op_sum_p10:.1f}% | "
                     f"{r['median_ms']:.3f} | {r['iqr_over_median']*100:.1f}%{noisy} |")
        L.append(f"| **sum of ops** | **{op_sum_p10:.3f}** | 100% | **{op_sum:.3f}** | |")
        L.append("")
        L.append(f"- Replica bracketed totals (median): fwd {rep['replica.fwd_total']['median_ms']:.3f} ms, "
                 f"bwd {rep['replica.bwd_total']['median_ms']:.3f} ms "
                 f"(sum {tot['replica_total_ms']:.3f} ms)")
        L.append(f"- REAL `FlashFFNFunction.apply` (median): fwd {e1['real']['real.fwd']['median_ms']:.3f} ms, "
                 f"bwd {e1['real']['real.bwd']['median_ms']:.3f} ms "
                 f"(total {tot['flash_real_total_ms']:.3f} ms; replica/real = "
                 f"{tot['replica_vs_real_ratio']:.3f})")
        L.append(f"- StandardFFN (median): fwd {e1['standard']['std.fwd']['median_ms']:.3f} ms, "
                 f"bwd {e1['standard']['std.bwd']['median_ms']:.3f} ms "
                 f"(total {tot['standard_total_ms']:.3f} ms)")
        L.append(f"- **flash_total / standard_total = {tot['flash_over_standard']:.3f} (median)**, "
                 f"{tot['flash_over_standard_p10']:.3f} (p10 floor)")
        L.append("")

        e2 = cfg["e2"]
        L.append("### E2 — isolated top-K pipeline (on precomputed h_mid)")
        L.append("")
        L.append(f"abs {e2['ops']['e2.abs']['p10_ms']:.3f} ms | "
                 f"topk {e2['ops']['e2.topk']['p10_ms']:.3f} ms | "
                 f"gather {e2['ops']['e2.gather']['p10_ms']:.3f} ms | "
                 f"int16 cast {e2['ops']['e2.cast_int16']['p10_ms']:.3f} ms (p10)")
        L.append("")
        e2_names = ["e2.abs", "e2.topk", "e2.gather", "e2.cast_int16"]
        pipeline_p10 = sum(e2["ops"][n]["p10_ms"] for n in e2_names)
        flash_p10 = tot["flash_real_total_p10_ms"]
        L.append(f"- p10 basis: topk alone = **{100*e2['ops']['e2.topk']['p10_ms']/flash_p10:.1f}%** "
                 f"of total flash fwd+bwd step; full abs+topk+gather+cast pipeline = "
                 f"**{100*pipeline_p10/flash_p10:.1f}%**")
        L.append(f"- median basis: topk alone = {100*e2['topk_fraction_of_flash_total_step']:.1f}%; "
                 f"pipeline = {100*e2['fraction_of_flash_total_step']:.1f}%")
        L.append("")

        e3 = cfg["e3"]
        L.append("### E3 — stored-for-backward bytes (saved_tensors_hooks, deduped)")
        L.append("")
        L.append("| tensor | FlashFFN | StandardFFN |")
        L.append("|---|---|---|")
        all_labels = sorted(set(list(e3["flash"]["saved"]["per_label"].keys())
                                + list(e3["standard"]["saved"]["per_label"].keys())))
        for lab in all_labels:
            fb = e3["flash"]["saved"]["per_label"].get(lab, {}).get("bytes", 0)
            sb = e3["standard"]["saved"]["per_label"].get(lab, {}).get("bytes", 0)
            L.append(f"| {lab} | {fmt_bytes(fb) if fb else '—'} | {fmt_bytes(sb) if sb else '—'} |")
        L.append(f"| **total (unique)** | **{fmt_bytes(e3['flash']['saved']['total_unique_bytes'])}** "
                 f"| **{fmt_bytes(e3['standard']['saved']['total_unique_bytes'])}** |")
        L.append(f"| total excl. weights (params) | {fmt_bytes(e3['flash_nonparam_saved_bytes'])} "
                 f"| {fmt_bytes(e3['standard_nonparam_saved_bytes'])} |")
        L.append("")
        L.append(f"- Backward transient peak (over alloc before bwd): "
                 f"flash {fmt_bytes(e3['flash']['bwd_transient_bytes'])}, "
                 f"standard {fmt_bytes(e3['standard']['bwd_transient_bytes'])}")
        L.append("")

        e4 = cfg["e4"]
        L.append("### E4 — LoRA effective-weight scenario (r=8)")
        L.append("")
        L.append(f"- W_eff materialized tensors saved by FlashFFN: "
                 f"{e4['flash_eff_saved']['per_label'].get('W_eff:gate', {}).get('count', 0) + e4['flash_eff_saved']['per_label'].get('W_eff:up', {}).get('count', 0) + e4['flash_eff_saved']['per_label'].get('W_eff:down', {}).get('count', 0)} "
                 f"tensors, **{fmt_bytes(e4['w_eff_extra_bytes'])}** extra")
        L.append(f"- FlashFFN(W_eff) non-param saved: {fmt_bytes(e4['flash_nonparam_saved_bytes'])}; "
                 f"PEFT-style LoRA baseline non-param saved: {fmt_bytes(e4['baseline_nonparam_saved_bytes'])}")
        net = e4["net_bytes_flash_minus_baseline"]
        L.append(f"- **Net flash − baseline = {'+' if net >= 0 else '−'}{fmt_bytes(abs(net))}** "
                 f"({'flash COSTS more' if net >= 0 else 'flash SAVES'} at N={cfg['n_tokens']})")
        L.append("")

    if results.get("discrepancies"):
        L.append("## Discrepancies vs task description")
        L.append("")
        for d in results["discrepancies"]:
            L.append(f"- {d}")
        L.append("")

    return "\n".join(L) + "\n"


# =============================================================================
# E5 append (parses train_glue.py results CSV)
# =============================================================================

def append_e5(csv_path, md_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    L = ["", "## E5 — end-to-end reality anchor (train_glue.py, TinyLlama, wikitext2)", ""]
    L.append("| run | flash_ffn | peak_mem (MiB) | perplexity | total_time (s) |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r.get('name','?')} | {r.get('flash_ffn','?')} | "
                 f"{r.get('peak_mem_mib','?')} | {r.get('perplexity','?')} | "
                 f"{r.get('total_training_time_sec','?')} |")
    if len(rows) == 2:
        try:
            m0, m1 = float(rows[0]["peak_mem_mib"]), float(rows[1]["peak_mem_mib"])
            L.append("")
            L.append(f"- peak-mem ratio (row2/row1): {m1/m0:.3f}")
        except (KeyError, ValueError):
            pass
    L.append("")
    L.append("- CAVEAT (pre-existing train_glue.py artifact): the CSV column "
             "`avg_step_time` brackets ONLY `optimizer.step()` with perf_counter and "
             "no cuda.synchronize — it is async CPU launch time of the AdamW update, "
             "NOT training throughput. Use wall step time from the run logs instead.")
    with open(md_path, "a") as f:
        f.write("\n".join(L) + "\n")
    print(f"Appended E5 section to {md_path}")
    # also drop raw rows into the JSON
    json_path = os.path.join(os.path.dirname(md_path), "microbench.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        data["e5_rows"] = rows
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append-e5", type=str, default=None,
                    help="Path to e5.csv — append E5 section to existing summary.md and exit")
    ap.add_argument("--rerender", action="store_true",
                    help="Regenerate summary.md from existing microbench.json and exit")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "summary.md")

    if args.append_e5:
        append_e5(args.append_e5, md_path)
        return

    if args.rerender:
        with open(os.path.join(args.out_dir, "microbench.json")) as f:
            results = json.load(f)
        with open(md_path, "w") as f:
            f.write(render_markdown(results))
        print(f"Re-rendered {md_path}")
        return

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "dtype": str(DTYPE),
            "k_fraction": K_FRACTION,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
        },
        "configs": [],
        "discrepancies": [],
    }

    # Code-reading notes surfaced as data (checked manually against source):
    results["notes"] = [
        "Recompute-mode save set is (x, w_gate, w_up, w_down, top_values, all_indices)"
        " — weights ARE in the save set but are parameter references (no extra allocation"
        " when they are real parameters; they DO allocate when they are materialized W_eff).",
        "indices are stored as int16 and upcast with .long() inside backward —"
        " a transient int64 tensor of 4x the stored index bytes is allocated every backward.",
        "torch.topk is called on h_mid.abs() — the abs() materializes a full (N, D_int) copy.",
    ]

    for label, d_hid, d_int in SHAPES:
        for n_tokens in N_TOKENS:
            cfg_label = f"{label}_N{n_tokens}"
            print(f"\n=== {cfg_label} (D_hid={d_hid}, D_int={d_int}) ===", flush=True)
            cleanup()
            x, w_gate, w_up, w_down, g = make_tensors(n_tokens, d_hid, d_int, device)
            k = max(1, int(d_int * K_FRACTION))

            print("  verifying replica vs FlashFFNFunction.apply ...", flush=True)
            verification = verify_replica(x, w_gate, w_up, w_down, g, d_int)
            print(f"  verification: {verification}", flush=True)

            print("  E1: per-op timing ...", flush=True)
            e1 = run_e1(x, w_gate, w_up, w_down, g, d_int, device)
            print(f"    flash {e1['totals']['flash_real_total_ms']:.3f} ms vs "
                  f"standard {e1['totals']['standard_total_ms']:.3f} ms "
                  f"(ratio {e1['totals']['flash_over_standard']:.3f})", flush=True)

            print("  E2: isolated topk ...", flush=True)
            e2 = run_e2(x, w_gate, w_up, d_int, e1["totals"]["flash_real_total_ms"])
            print(f"    topk {e2['topk_only_ms']:.3f} ms "
                  f"({100*e2['topk_fraction_of_flash_total_step']:.1f}% of flash step)", flush=True)

            print("  E3: saved bytes + bwd transient ...", flush=True)
            e3 = run_e3(x, w_gate, w_up, w_down, g, d_int, device)
            print(f"    flash saved {fmt_bytes(e3['flash']['saved']['total_unique_bytes'])} "
                  f"vs std {fmt_bytes(e3['standard']['saved']['total_unique_bytes'])}", flush=True)

            del x, w_gate, w_up, w_down, g
            cleanup()

            print("  E4: LoRA effective-weight ...", flush=True)
            e4 = run_e4(n_tokens, d_hid, d_int, device)
            print(f"    W_eff extra {fmt_bytes(e4['w_eff_extra_bytes'])}, "
                  f"net vs baseline {fmt_bytes(abs(e4['net_bytes_flash_minus_baseline']))} "
                  f"{'COST' if e4['net_bytes_flash_minus_baseline'] >= 0 else 'SAVED'}", flush=True)
            cleanup()

            results["configs"].append({
                "label": cfg_label,
                "d_hid": d_hid,
                "d_int": d_int,
                "n_tokens": n_tokens,
                "k": k,
                "verification": verification,
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "e4": e4,
            })

    json_path = os.path.join(args.out_dir, "microbench.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(render_markdown(results))
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
