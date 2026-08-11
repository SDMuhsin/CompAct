#!/usr/bin/env python
"""Per-kernel GPU-time attribution of ONE training step of the `fb_min_fnorm_sdpa` arm.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a **measurement-only** reconnaissance script.  It answers "what fraction of the step's GPU
time is spent in which class of work", on a box whose two A40s are pegged at ~99-100% by other
tenants.  **The absolute millisecond numbers it prints are inflated by that contention and are NOT
throughput results.**  The quantities that survive a loud box are the *shares*, the *launch counts*
and the *per-call means* -- `measurement-gotchas-shared-box` and `fair_comparison_protocol.md`
A.3.6 both say so, and the shares are ratios inside a single window so a uniform slowdown cancels.

WHY IT IS A NEW FILE RATHER THAN A FLAG ON `diag_fb_breakdown.py`
----------------------------------------------------------------
`src/diag_fb_breakdown.py` already buckets kernels, but it (a) runs with `record_shapes=False`, so
it cannot separate the base-weight GEMMs from the LoRA GEMMs -- which is the single most useful cut
of the GEMM bucket, since they are the same cuBLAS kernels at wildly different arithmetic
intensity; (b) reports only the naive SUM of kernel durations, which overcounts whenever two
streams run concurrently and therefore cannot yield a GPU-idle fraction; and (c) keeps no timeline,
so it cannot report inter-kernel gaps.  This script parses the kineto chrome trace directly, which
gives all three.

METHOD
------
1. Build the arm through `profile_unsloth.build` (which delegates to `profile_hyclora.build_model`)
   so the model, the adapters, the batch and the `step()` body are byte-identical to the harness
   every published number in this project came from.  Nothing is re-implemented here.
2. Warm up (Triton autotune, the fused block's policy resolution, one optimizer step so the Adam
   moments already exist), then profile N steady-state full steps with
   `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)`.
3. Export the chrome trace and parse it.  Every GPU kernel carries a `correlation` id shared with
   its `cudaLaunchKernel` runtime event; that runtime event is nested inside the ATen op that
   issued it, and with `record_shapes=True` that op carries `Input Dims`.  Rebuilding the CPU
   ancestor stack per thread therefore attributes every kernel to a named op WITH SHAPES.  That is
   what makes `gemm_big` vs `gemm_lora` a measured split rather than a guess.
4. GPU-busy time is the **union of kernel intervals**, not their sum, so concurrent streams and
   the copy engines are counted once.  Idle fraction = (wall - busy)/wall.

CLASSES (task-specified)
------------------------
  gemm_big          cuBLAS/cutlass GEMM whose K or N >= 512 (the base-weight projections)
  gemm_lora         GEMM one of whose dims is the LoRA rank (16)
  gemm_small        GEMM with no dim >= 512 and no dim == 16 (reported separately, never hidden)
  gemm_unattributed GEMM whose issuing op could not be resolved (should be 0; reported if not)
  flash_fwd         scaled-dot-product FlashAttention forward
  flash_bwd         ... backward (incl. its dot_do_o / convert_dq / clear_dkv preambles)
  triton            every Triton kernel, listed individually by name
  elementwise_other aten elementwise / copy / reduction kernels
  optimizer         the optimizer's kernels (foreach / fused adam)
  memcpy_memset     DMA and memset activity (kept out of `other` so it cannot hide)
  other             everything else

NOTE ON THE LoRA-RANK TEST.  `16` is used as the LoRA-rank discriminator.  On TinyLlama-1.1B that
is unambiguous: hidden=2048, intermediate=5632, heads=32, kv_heads=4, head_dim=64, batch=2, and
seq is 1024/4096 -- no other shape in the step is 16.  The check is asserted against the live
config at start-up so it cannot silently become wrong on another model.

Usage:
  python src/profile_step_breakdown.py --seq 1024 --steps 8 --flce
  python src/profile_step_breakdown.py --seq 4096 --steps 8 --flce
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import profile_unsloth as pu  # noqa: E402
import profile_hyclora as ph  # noqa: E402
from profile_hyclora import make_batch, step  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "recon")


# ---------------------------------------------------------------------------------------------
# GPU state receipt -- mandatory on this box
# ---------------------------------------------------------------------------------------------
def nvidia_smi_full():
    try:
        q = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        return q.stdout
    except Exception as exc:                                            # pragma: no cover
        return f"error: {exc}"


# ---------------------------------------------------------------------------------------------
# Kernel classification
# ---------------------------------------------------------------------------------------------
_FLASH_FWD = re.compile(r"flash_fwd", re.I)
_FLASH_BWD = re.compile(r"flash_bwd|dot_do_o|convert_dq|clear_dkvaccum|clear_dKVaccum", re.I)
# cuBLAS / cutlass naming across the generations that can appear on an A40 (sm_86).
_GEMM_NAME = re.compile(
    r"gemm|cutlass|ampere_|turing_|volta_|sm80_|sm86_|s16816|s1688|h1688|"
    r"splitKreduce|gemv|nn_align|tn_align|nt_align|dot_kernel|GemmEx", re.I)
_GEMM_OPS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul",
             "aten::linear", "aten::_scaled_mm", "aten::einsum")
_OPT_ANCESTOR = re.compile(r"Optimizer\.step|aten::_foreach|adam|_fused_adam", re.I)
_OPT_NAME = re.compile(r"multi_tensor_apply|fused_adam|adamw?_kernel|amp_", re.I)
_ELEMENTWISE = re.compile(
    r"elementwise|vectorized_|unrolled_|reduce_kernel|CatArray|copy|Copy|fill|Fill|index|"
    r"transpose|permute|contiguous|cast|convert|at::native::|scatter|gather|sum_kernel|"
    r"norm_kernel|nll|softmax|cross_entropy|embedding|masked|arange|clamp|where|"
    r"reduce_?scatter|BinaryOp|UnaryOp|distribution|randperm|cub::", re.I)
# A Triton JIT kernel appears in kineto under its *python function name*: a plain C identifier,
# with no `void ` prefix and no C++ template/namespace punctuation.  Checked LAST among the
# name-based rules so that plain-identifier cuBLAS names (`ampere_bf16_s16816gemm_...`) are
# already gone.
_PLAIN_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KNOWN_TRITON = re.compile(
    r"^_fb_|^_silu_mul|^_swiglu|^_drelu|^_v3_|^triton_|liger|^_efficient|^rms_norm|^_rms|"
    r"^_ln_|^rope_|^_layer_norm|kernel$", re.I)


# --- phase attribution -------------------------------------------------------------------------
# The fused block is ONE `torch.autograd.Function`, so kineto's CPU stack labels every kernel it
# issues with either `FusedLoRABlockFunction` (the forward) or `FusedLoRABlockFunctionBackward`
# (which contains BOTH the `keep='min'` recompute of the block forward AND the backward proper).
# That is the finest forward/backward cut this window can make without instrumenting the kernel,
# and it is what lets the result be checked against `route_b.md` section 2.1's recompute
# decomposition instead of duplicating it.
_PHASES = [
    ("optimizer", re.compile(r"Optimizer\.step|aten::_foreach|_fused_adam", re.I)),
    ("block_bwd", re.compile(r"FusedLoRABlockFunctionBackward")),
    ("block_fwd", re.compile(r"FusedLoRABlockFunction(?!Backward)")),
    ("final_norm_bwd", re.compile(r"_FinalNorm\w*Backward")),
    ("final_norm_fwd", re.compile(r"_FinalNorm")),
    ("loss_ce", re.compile(r"CrossEntropy|cross_entropy", re.I)),
    ("autograd_other", re.compile(r"autograd::engine::evaluate_function|torch/autograd")),
]


def phase_of(ancestors):
    anc = " | ".join(ancestors or ())
    for name, pat in _PHASES:
        if pat.search(anc):
            return name
    return "fwd_other"


def _gemm_dims(shapes):
    """(M, K, N) for an mm/addmm/bmm-shaped `Input Dims` list, or None."""
    mats = [s for s in shapes if isinstance(s, (list, tuple)) and len(s) >= 2]
    if len(mats) < 2:
        return None
    a, b = mats[-2], mats[-1]
    try:
        return int(a[-2]), int(a[-1]), int(b[-1])
    except Exception:
        return None


def classify(name, op_name, shapes, cat, lora_r, ancestors):
    """-> (class, detail).  `detail` carries the resolved GEMM dims when there are any."""
    if cat in ("gpu_memcpy", "gpu_memset", "Memcpy", "Memset"):
        return "memcpy_memset", None
    anc = " | ".join(ancestors or ())
    if _OPT_ANCESTOR.search(anc) or _OPT_NAME.search(name):
        return "optimizer", None
    if _FLASH_FWD.search(name):
        return "flash_fwd", None
    if _FLASH_BWD.search(name):
        return "flash_bwd", None

    is_gemm = bool(_GEMM_NAME.search(name)) or (op_name in _GEMM_OPS)
    if is_gemm:
        dims = _gemm_dims(shapes) if shapes else None
        if dims is None:
            return "gemm_unattributed", None
        m, k, n = dims
        if lora_r in (m, k, n):
            return "gemm_lora", dims
        if k >= 512 or n >= 512:
            return "gemm_big", dims
        return "gemm_small", dims

    if _KNOWN_TRITON.search(name) or (_PLAIN_IDENT.match(name) and not name.startswith("void")):
        return "triton", None
    if _ELEMENTWISE.search(name):
        return "elementwise_other", None
    return "other", None


# ---------------------------------------------------------------------------------------------
# Chrome-trace parsing
# ---------------------------------------------------------------------------------------------
_STACK_CATS = ("cpu_op", "user_annotation", "python_function")
_RUNTIME_CATS = ("cuda_runtime", "cuda_driver")
_DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset", "Kernel", "Memcpy", "Memset")


def parse_trace(path, lora_r):
    with open(path) as f:
        trace = json.load(f)
    evs = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]

    # --- 1. correlation -> issuing ATen op (name + Input Dims), via the per-thread CPU stack ---
    per_tid = defaultdict(list)
    for e in evs:
        cat = e.get("cat")
        if cat in _STACK_CATS or cat in _RUNTIME_CATS:
            per_tid[(e.get("pid"), e.get("tid"))].append(e)

    corr_info = {}
    for _tid, lst in per_tid.items():
        # outer-first at equal start time
        lst.sort(key=lambda e: (e.get("ts", 0.0), -e.get("dur", 0.0)))
        stack = []
        for e in lst:
            ts = e.get("ts", 0.0)
            while stack and (stack[-1].get("ts", 0.0) + stack[-1].get("dur", 0.0)) <= ts:
                stack.pop()
            if e.get("cat") in _RUNTIME_CATS:
                corr = (e.get("args") or {}).get("correlation")
                if corr is None:
                    continue
                op_name, shapes = None, None
                for anc in reversed(stack):
                    if anc.get("cat") != "cpu_op":
                        continue
                    if op_name is None:
                        op_name = anc.get("name")
                        shapes = (anc.get("args") or {}).get("Input Dims")
                    if shapes:
                        op_name = anc.get("name")
                        break
                corr_info[corr] = (op_name, shapes,
                                   tuple(a.get("name", "") for a in stack)[-12:])
            else:
                stack.append(e)

    # --- 2. device events ---
    per_kernel = {}
    intervals = []
    n_dev = 0
    unresolved = 0
    streams = defaultdict(int)
    for e in evs:
        cat = e.get("cat")
        if cat not in _DEVICE_CATS:
            continue
        args = e.get("args") or {}
        dur = float(e.get("dur", 0.0) or 0.0)
        ts = float(e.get("ts", 0.0) or 0.0)
        name = e.get("name", "?")
        corr = args.get("correlation")
        op_name, shapes, ancestors = corr_info.get(corr, (None, None, ()))
        if op_name is None:
            unresolved += 1
        cls, dims = classify(name, op_name, shapes, cat, lora_r, ancestors)
        pha = phase_of(ancestors)
        n_dev += 1
        intervals.append((ts, ts + dur))
        streams[args.get("stream")] += 1
        d = per_kernel.get(name)
        if d is None:
            d = per_kernel[name] = {"name": name, "cat": cat,
                                    "total_us": 0.0, "count": 0,
                                    "by_class": defaultdict(lambda: [0.0, 0]),
                                    "by_phase": defaultdict(lambda: [0.0, 0]),
                                    "by_op": defaultdict(lambda: [0.0, 0]),
                                    "streams": defaultdict(int)}
        d["total_us"] += dur
        d["count"] += 1
        # EXACT per-instance class accounting.  One cuBLAS kernel name legitimately serves both a
        # base-weight GEMM and a LoRA GEMM (e.g. `ampere_bf16_s1688gemm_..._tn` runs both
        # [8192,2048]x[2048,5632] and [8192,16]x[16,5632]); apportioning that kernel's time to the
        # two classes BY CALL COUNT would be badly wrong, because their per-call costs differ by an
        # order of magnitude.  So the class is resolved per launch and the duration is added to
        # that class directly.
        d["by_class"][cls][0] += dur
        d["by_class"][cls][1] += 1
        d["by_phase"][pha][0] += dur
        d["by_phase"][pha][1] += 1
        d["streams"][args.get("stream")] += 1
        key = json.dumps([op_name, shapes, dims, cls, pha], default=str)
        d["by_op"][key][0] += dur
        d["by_op"][key][1] += 1

    return per_kernel, intervals, n_dev, unresolved, dict(streams)


def union_busy(intervals):
    """Union length of the kernel timeline, plus the gap statistics between merged runs."""
    if not intervals:
        return {"busy_us": 0.0, "span_us": 0.0, "n_merged": 0, "gaps_us": []}
    iv = sorted(intervals)
    merged = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    busy = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    return {"busy_us": busy, "span_us": merged[-1][1] - merged[0][0],
            "n_merged": len(merged), "gaps_us": gaps,
            "t0": merged[0][0], "t1": merged[-1][1]}


# ---------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------
def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = ph.make_cfg(args.batch, args.seq, model=args.model)

    # The LoRA-rank discriminator must be unambiguous on THIS config, or `gemm_lora` is a guess.
    other_shapes = {cfg["hidden"], cfg["intermediate"], cfg["n_heads"], cfg["n_kv_heads"],
                    cfg["head_dim"], cfg["batch"], cfg["seq"], cfg["batch"] * cfg["seq"],
                    cfg["n_kv_heads"] * cfg["head_dim"]}
    if cfg["lora_r"] in other_shapes:
        raise RuntimeError(
            f"lora_r={cfg['lora_r']} collides with another shape in this config {sorted(other_shapes)}; "
            f"the gemm_lora split would be wrong. Refusing to measure.")

    smi_before = nvidia_smi_full()
    gpu_before = ph._gpu_state()

    torch.manual_seed(41)
    model = pu.build(args.arm, cfg, device, args)
    vocab = model.config.vocab_size
    batch = make_batch(cfg, device, vocab)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4)

    # ---- warm-up: Triton autotune, the fused block's policy resolution, Adam moments ----
    for _ in range(args.warmup):
        step(model, batch, opt)
    torch.cuda.synchronize()

    # ---- engagement receipt: the fused block must have run, at the level the arm names ----
    from flashffn import fb_get_counters, fb_policy_report
    fbc = fb_get_counters()
    fbp = fb_policy_report()
    if fbc["forward"] == 0 or fbc["backward"] == 0:
        raise RuntimeError(f"{args.arm}: fused block never executed ({fbc})")

    gc.collect()
    torch.cuda.empty_cache()

    # ---- clean (un-profiled) wall clock, for reference only ----
    clean = []
    for _ in range(max(3, args.steps // 2)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(model, batch, opt)
        torch.cuda.synchronize()
        clean.append(time.perf_counter() - t0)
    clean.sort()

    # ---- the profiled window ----
    os.makedirs(args.tmpdir, exist_ok=True)
    trace_path = os.path.join(args.tmpdir, f"step_breakdown_seq{cfg['seq']}_trace.json")
    from torch.profiler import profile, ProfilerActivity, record_function

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=False, profile_memory=False) as prof:
        torch.cuda.synchronize()
        t_wall0 = time.perf_counter()
        for i in range(args.steps):
            with record_function(f"STEP{i}"):
                step(model, batch, opt)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_wall0
    peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20
    prof.export_chrome_trace(trace_path)
    trace_bytes = os.path.getsize(trace_path)

    smi_after = nvidia_smi_full()
    gpu_after = ph._gpu_state()

    per_kernel, intervals, n_dev, unresolved, streams = parse_trace(trace_path, cfg["lora_r"])
    tl = union_busy(intervals)
    n = float(args.steps)

    sum_us = sum(d["total_us"] for d in per_kernel.values())
    wall_us = wall * 1e6

    cls_tot = defaultdict(lambda: [0.0, 0])
    pha_tot = defaultdict(lambda: [0.0, 0])
    cls_x_pha = defaultdict(lambda: [0.0, 0])
    for d in per_kernel.values():
        for c, (us, cnt) in d["by_class"].items():
            cls_tot[c][0] += us
            cls_tot[c][1] += cnt
        for p, (us, cnt) in d["by_phase"].items():
            pha_tot[p][0] += us
            pha_tot[p][1] += cnt
        for key, (us, cnt) in d["by_op"].items():
            _o, _s, _dm, c, p = json.loads(key)
            cls_x_pha[(c, p)][0] += us
            cls_x_pha[(c, p)][1] += cnt
        # the kernel's headline class is the one holding most of ITS time
        d["class"] = max(d["by_class"].items(), key=lambda kv: kv[1][0])[0]

    def _kernel_row(d):
        top = sorted(d["by_op"].items(), key=lambda kv: -kv[1][0])[:6]
        return {
            "name": d["name"], "cat": d["cat"], "class": d["class"],
            "class_mix_ms_per_step": {c: v[0] / 1e3 / n
                                      for c, v in sorted(d["by_class"].items(),
                                                         key=lambda kv: -kv[1][0])},
            "class_mix_calls_per_step": {c: v[1] / n for c, v in d["by_class"].items()},
            "phase_mix_ms_per_step": {p: v[0] / 1e3 / n
                                      for p, v in sorted(d["by_phase"].items(),
                                                         key=lambda kv: -kv[1][0])},
            "phase_mix_calls_per_step": {p: v[1] / n for p, v in d["by_phase"].items()},
            "total_us": d["total_us"], "ms_per_step": d["total_us"] / 1e3 / n,
            "count": d["count"], "calls_per_step": d["count"] / n,
            "mean_us_per_call": d["total_us"] / max(d["count"], 1),
            "pct_of_sum_device": 100.0 * d["total_us"] / sum_us if sum_us else 0.0,
            "streams": {str(k): v for k, v in d["streams"].items()},
            "top_calls": [{"op": json.loads(k)[0], "class": json.loads(k)[3],
                           "input_dims": json.loads(k)[1], "gemm_MKN": json.loads(k)[2],
                           "us": v[0], "count": v[1],
                           "mean_us": v[0] / max(v[1], 1)} for k, v in top],
        }

    rows = sorted(per_kernel.values(), key=lambda d: -d["total_us"])
    gaps = tl["gaps_us"]
    gaps_sorted = sorted(gaps)

    # --- GEMM shape table: every GEMM launch aggregated by (class, issuing op, M, K, N).
    # This is what makes the gemm_big / gemm_lora split auditable rather than asserted.
    # PRECEDENCE NOTE: a LoRA GEMM such as [N,2048]@[2048,16] satisfies BOTH task definitions
    # (K >= 512 and a dim == r).  `gemm_lora` wins, because the intent of the split is
    # "base-weight projection" vs "adapter", and every adapter GEMM has a 512+ contraction.
    gemm_shapes = defaultdict(lambda: [0.0, 0])
    for d in per_kernel.values():
        for key, (us, cnt) in d["by_op"].items():
            op_name, _shapes, dims, cls, pha = json.loads(key)
            if not str(cls).startswith("gemm"):
                continue
            gemm_shapes[(cls, pha, op_name, tuple(dims) if dims else None)][0] += us
            gemm_shapes[(cls, pha, op_name, tuple(dims) if dims else None)][1] += cnt
    gemm_table = [{"class": c, "phase": p, "op": op, "M": (mkn or (None,) * 3)[0],
                   "K": (mkn or (None,) * 3)[1], "N": (mkn or (None,) * 3)[2],
                   "ms_per_step": v[0] / 1e3 / n, "calls_per_step": v[1] / n,
                   "mean_us": v[0] / max(v[1], 1)}
                  for (c, p, op, mkn), v in sorted(gemm_shapes.items(), key=lambda kv: -kv[1][0])]

    res = {
        "WARNING": ("Absolute times are from a CONTENDED box (both A40s at ~99-100% from "
                    "processes outside this project). They are NOT throughput results. Only the "
                    "SHARES, the launch counts and the per-call means are quotable."),
        "arm": args.arm, "flce": bool(args.flce),
        "seq": cfg["seq"], "batch": cfg["batch"], "model": cfg["model"],
        "lora_r": cfg["lora_r"], "n_layers": cfg["n_layers"],
        "adapter_dtype": args.adapter_dtype,
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "warmup_steps": args.warmup, "n_profiled_steps": args.steps,
        "fb_counters": fbc, "fb_policy": fbp,
        "engagement": getattr(model, "_engagement", None),
        "peak_alloc_mib_during_profile": peak_mib,

        "wall_ms_per_step_profiled": wall * 1e3 / n,
        "wall_ms_per_step_unprofiled_median": 1e3 * clean[len(clean) // 2],
        "wall_ms_unprofiled_all": [1e3 * c for c in clean],

        "device_time_sum_ms_per_step": sum_us / 1e3 / n,
        "device_busy_union_ms_per_step": tl["busy_us"] / 1e3 / n,
        "kernel_timeline_span_ms_per_step": tl["span_us"] / 1e3 / n,
        "gpu_idle_fraction_vs_wall": (wall_us - tl["busy_us"]) / wall_us if wall_us else None,
        "gpu_idle_fraction_vs_kernel_span": ((tl["span_us"] - tl["busy_us"]) / tl["span_us"]
                                             if tl["span_us"] else None),
        "concurrency_overcount_ratio": sum_us / tl["busy_us"] if tl["busy_us"] else None,

        "n_device_events_total": n_dev,
        "launches_per_step": n_dev / n,
        "n_distinct_kernels": len(per_kernel),
        "n_unresolved_issuing_op": unresolved,
        "streams_used": {str(k): v for k, v in streams.items()},
        "gap_stats_us": {
            "n_gaps": len(gaps),
            "gaps_per_step": len(gaps) / n,
            "mean_gap_us": (sum(gaps) / len(gaps)) if gaps else None,
            "median_gap_us": gaps_sorted[len(gaps) // 2] if gaps else None,
            "p90_gap_us": gaps_sorted[int(0.9 * len(gaps))] if gaps else None,
            "max_gap_us": gaps_sorted[-1] if gaps else None,
            "total_idle_in_span_ms_per_step": (tl["span_us"] - tl["busy_us"]) / 1e3 / n,
            "mean_gap_per_launch_us": ((tl["span_us"] - tl["busy_us"]) / (n_dev - 1)
                                       if n_dev > 1 else None),
        },

        "classes": {c: {"ms_per_step": v[0] / 1e3 / n,
                        "pct_of_sum_device": 100.0 * v[0] / sum_us if sum_us else 0.0,
                        "launches": v[1], "launches_per_step": v[1] / n,
                        "mean_us_per_launch": v[0] / v[1] if v[1] else None}
                    for c, v in sorted(cls_tot.items(), key=lambda kv: -kv[1][0])},

        "phases": {p: {"ms_per_step": v[0] / 1e3 / n,
                       "pct_of_sum_device": 100.0 * v[0] / sum_us if sum_us else 0.0,
                       "launches_per_step": v[1] / n}
                   for p, v in sorted(pha_tot.items(), key=lambda kv: -kv[1][0])},
        "class_x_phase": [{"class": c, "phase": p, "ms_per_step": v[0] / 1e3 / n,
                           "pct_of_sum_device": 100.0 * v[0] / sum_us if sum_us else 0.0,
                           "launches_per_step": v[1] / n}
                          for (c, p), v in sorted(cls_x_pha.items(), key=lambda kv: -kv[1][0])],

        "gemm_shape_table": gemm_table,
        "triton_kernels": [_kernel_row(d) for d in rows if d["class"] == "triton"],
        "top_kernels": [_kernel_row(d) for d in rows[:40]],
        "all_kernels": [_kernel_row(d) for d in rows],

        "gpu_state_before": gpu_before, "gpu_state_after": gpu_after,
        "nvidia_smi_before": smi_before, "nvidia_smi_after": smi_after,
        "trace_path": trace_path, "trace_bytes": trace_bytes,
    }

    del model, opt, batch, prof
    gc.collect()
    torch.cuda.empty_cache()
    if not args.keep_trace:
        os.remove(trace_path)
        res["trace_path"] = f"{trace_path} (deleted; rerun with --keep_trace)"
    return res


def print_report(res):
    print("\n" + "=" * 100)
    print(f"  {res['arm']}  seq {res['seq']}  batch {res['batch']}  "
          f"{res['n_profiled_steps']} profiled steps  --  CONTENDED BOX, SHARES ONLY")
    print("=" * 100)
    print(f"  wall/step (profiled)      {res['wall_ms_per_step_profiled']:.2f} ms")
    print(f"  wall/step (unprofiled)    {res['wall_ms_per_step_unprofiled_median']:.2f} ms")
    print(f"  device time SUM/step      {res['device_time_sum_ms_per_step']:.2f} ms")
    print(f"  device BUSY (union)/step  {res['device_busy_union_ms_per_step']:.2f} ms")
    print(f"  GPU idle fraction vs wall {res['gpu_idle_fraction_vs_wall'] * 100:.2f}%")
    print(f"  launches/step             {res['launches_per_step']:.1f}  "
          f"({res['n_distinct_kernels']} distinct kernels)")
    g = res["gap_stats_us"]
    print(f"  inter-kernel gap          mean {g['mean_gap_us']:.2f} us, "
          f"median {g['median_gap_us']:.2f} us, p90 {g['p90_gap_us']:.2f} us")
    print("\n  {:<20} {:>11} {:>8} {:>11} {:>12}".format(
        "class", "ms/step", "%", "launches", "us/launch"))
    print("  " + "-" * 66)
    for c, v in res["classes"].items():
        print("  {:<20} {:>11.3f} {:>8.2f} {:>11.1f} {:>12.2f}".format(
            c, v["ms_per_step"], v["pct_of_sum_device"], v["launches_per_step"],
            v["mean_us_per_launch"] or 0.0))
    print("\n  {:<20} {:>11} {:>8} {:>11}".format("autograd phase", "ms/step", "%", "launches"))
    print("  " + "-" * 53)
    for p, v in res["phases"].items():
        print("  {:<20} {:>11.3f} {:>8.2f} {:>11.1f}".format(
            p, v["ms_per_step"], v["pct_of_sum_device"], v["launches_per_step"]))
    print("\n  TOP 20 KERNELS")
    print("  {:<58} {:>18} {:>10} {:>7} {:>9} {:>10}".format(
        "kernel", "class", "ms/step", "%", "calls/st", "us/call"))
    print("  " + "-" * 116)
    for k in res["top_kernels"][:20]:
        mixed = "*" if len(k["class_mix_ms_per_step"]) > 1 else ""
        print("  {:<58} {:>18} {:>10.3f} {:>7.2f} {:>9.1f} {:>10.2f}".format(
            k["name"][:58], k["class"] + mixed, k["ms_per_step"], k["pct_of_sum_device"],
            k["calls_per_step"], k["mean_us_per_call"]))
    print("  (* = this kernel name serves more than one class; per-launch class accounting is "
          "exact, see class_mix_ms_per_step in the JSON)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=ph.DEFAULT_MODEL)
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--flce", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmpdir", default="/tmp")
    ap.add_argument("--keep_trace", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    res = run(args)
    out = args.out or os.path.join(OUT_DIR, f"step_breakdown_seq{res['seq']}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print_report(res)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
