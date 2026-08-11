"""Ground-level tensor-op profiling of HyC-LoRA (MLSys 2025), and of our own arms for reference.

WHY THIS EXISTS
---------------
CONTEXT.md §29 establishes that HyC-LoRA beats FlashFFN v3 on end-to-end peak memory at seq 1024
(-66.8% vs -16.8%) while being 27% faster than stock HF eager. Those are whole-run aggregates.
They say nothing about WHERE the time and the bytes actually go, and therefore nothing about what
the *next* bottleneck is -- which is the question that decides whether their approach has headroom
left or is already at its floor.

HyC-LoRA implements an entire decoder layer as ONE `torch.autograd.Function`. That is deliberate
(it is how they control which buffers survive into backward) but it makes the layer opaque to
every standard profiler: module hooks see one call, autograd sees one node. `hyclora/prof.py`
adds zero-cost `record_function` regions inside the layer body so the ~70 tensor ops become
individually visible; this script drives them.

WHAT IT MEASURES
----------------
1. Achievable machine peaks on THIS GPU (bf16 tensor-core TFLOP/s, HBM GB/s, kernel launch
   latency), measured rather than taken from a spec sheet, to serve as roofline denominators.
2. Per-op device time, over steady-state steps only (iteration >= iteration_threshold, so the
   one-off calibration kernels are excluded -- and reported separately).
3. Per-CUDA-kernel time and launch counts.
4. An analytic FLOP and compulsory-byte model per op, giving arithmetic intensity, achieved
   TFLOP/s, achieved GB/s, % of the measured peaks, and a memory/compute/latency-bound verdict.
5. GPU busy time vs wall-clock, i.e. how much of the step is not on the GPU at all.
6. Exact saved-tensor accounting per layer, read off the real autograd graph, plus a peak-memory
   timeline across forward/backward/step.

METHODOLOGY NOTES (read before quoting any number)
--------------------------------------------------
* `ncu` is unavailable on this box (ERR_NVGPUCTRPERM; lifting it needs an NVIDIA kernel-module
  reload, unacceptable on a shared machine). So DRAM traffic is not hardware-measured. The byte
  model below is COMPULSORY traffic: every input read once, every output written once. For the
  streaming/elementwise/quantisation kernels that dominate this workload that is exact (there is
  no reuse to exploit). For GEMMs it is a lower bound, so their "achieved GB/s" is a lower bound
  too -- which is harmless, because GEMMs are classified by FLOP throughput, not bandwidth.
* Any op whose derived bandwidth EXCEEDS the measured HBM peak is flagged: that means either an
  L2 hit (the A40 has 6 MB of L2) or a modelling error, and it is called out rather than hidden.
* Timings come from a profiler pass; the wall-clock ground truth comes from a separate clean pass
  with profiling fully disabled, and the two are cross-checked.

Usage:
  python src/profile_hyclora.py --arm hyclora_q4
  python src/profile_hyclora.py --arm all
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "hyclora", "profile")


# ----------------------------------------------------------------------------------------------
# 1. Machine peaks -- measured, not assumed
# ----------------------------------------------------------------------------------------------

def _time_cuda(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]  # ms
    times.sort()
    return times[len(times) // 2] / 1e3  # median, seconds


def measure_peaks(device):
    """Sustained bf16 GEMM throughput, HBM bandwidth, and kernel launch latency on this GPU."""
    out = {}

    # --- bf16 tensor-core GEMM peak ---
    n = 8192
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    t = _time_cuda(lambda: torch.mm(a, b), iters=30)
    out["gemm_bf16_tflops"] = (2.0 * n ** 3) / t / 1e12
    del a, b

    # --- HBM bandwidth: pure streaming copy (1 read + 1 write) ---
    nelem = 1 << 27  # 128M bf16 = 256 MiB per buffer
    x = torch.randn(nelem, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    t = _time_cuda(lambda: y.copy_(x), iters=30)
    out["hbm_copy_gbs"] = (2.0 * nelem * 2) / t / 1e9

    # --- HBM bandwidth: 2 reads + 1 write (closer to the shape of most ops here) ---
    z = torch.empty_like(x)
    t = _time_cuda(lambda: torch.add(x, y, out=z), iters=30)
    out["hbm_add_gbs"] = (3.0 * nelem * 2) / t / 1e9
    out["hbm_peak_gbs"] = max(out["hbm_copy_gbs"], out["hbm_add_gbs"])
    del x, y, z

    # --- per-op CPU dispatch cost: a 1-element op is entirely dispatch+launch bound, so the
    #     amortised time of a long unsynchronised run of them is the CPU cost of issuing one op.
    #     This is the number that matters when a workload is launch-bound rather than GPU-bound.
    tiny = torch.ones(1, device=device)
    for _ in range(200):
        tiny.add_(1.0)
    torch.cuda.synchronize()
    n_iss = 5000
    t0 = time.perf_counter()
    for _ in range(n_iss):
        tiny.add_(1.0)
    torch.cuda.synchronize()
    out["cpu_dispatch_us_per_op"] = (time.perf_counter() - t0) / n_iss * 1e6

    # --- minimum GPU-side kernel duration (fixed cost of any kernel, however trivial) ---
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as _p:
        for _ in range(200):
            tiny.add_(1.0)
        torch.cuda.synchronize()
    _kt = [(getattr(e, "self_device_time_total", 0.0), e.count) for e in _p.key_averages()
           if str(getattr(e, "device_type", "")) == "DeviceType.CUDA"
           and getattr(e, "self_device_time_total", 0.0) > 0]
    out["gpu_min_kernel_us"] = (sum(a for a, _ in _kt) / sum(c for _, c in _kt)) if _kt else float("nan")
    del tiny

    torch.cuda.empty_cache()
    out["ridge_point_flops_per_byte"] = (out["gemm_bf16_tflops"] * 1e12) / (out["hbm_peak_gbs"] * 1e9)
    return out


# ----------------------------------------------------------------------------------------------
# 2. Analytic FLOP / compulsory-byte model, per instrumented region, per layer, per step
# ----------------------------------------------------------------------------------------------

def gemm(m, k, n, e=2):
    """A dense GEMM [m,k]@[k,n]. Bytes = compulsory operand traffic (lower bound; GEMMs reuse)."""
    return {"flops": 2.0 * m * k * n, "bytes": float(e) * (m * k + k * n + m * n), "gemm_flops": 2.0 * m * k * n}


def ew(nelem, reads=1, writes=1, flops_per=1.0, e=2, extra_bytes=0.0):
    """An elementwise/streaming kernel over `nelem` elements."""
    return {"flops": float(flops_per) * nelem, "bytes": float(e) * nelem * (reads + writes) + extra_bytes,
            "gemm_flops": 0.0}


def raw(flops=0.0, bytes_=0.0):
    return {"flops": float(flops), "bytes": float(bytes_), "gemm_flops": 0.0}


def merge(*parts):
    o = {"flops": 0.0, "bytes": 0.0, "gemm_flops": 0.0}
    for p in parts:
        for k in o:
            o[k] += p.get(k, 0.0)
    return o


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def make_cfg(batch, seq, model=DEFAULT_MODEL, lora_r=16, q_bit=4, attn_nnz=0):
    """The measurement cfg, with every shape field READ FROM THE MODEL CONFIG.

    Three call sites (`profile_hyclora.main`, `profile_unsloth.main`, `probe_highwater.main`) each
    carried their own copy of this dict with TinyLlama's 2048/5632/32/4/64/22 written out as
    literals.  That is why one model family has ever been measured: pointing the harness at a
    second model would have silently kept TinyLlama's shapes and mis-reported every derived
    quantity (`attn_nnz`, the analytic op model, `stored_all`, the per-layer retention report).

    `head_dim` is taken from the config when present -- Qwen3 and Gemma set it independently of
    `hidden_size / num_attention_heads`, and deriving it by division is wrong for them.
    """
    from transformers import AutoConfig

    hc = AutoConfig.from_pretrained(model)
    hc = getattr(hc, "text_config", hc)
    n_heads = hc.num_attention_heads
    hidden = hc.hidden_size
    head_dim = getattr(hc, "head_dim", None) or hidden // n_heads
    return {
        "model": model,
        "batch": batch, "seq": seq,
        "hidden": hidden,
        "intermediate": hc.intermediate_size,
        "n_heads": n_heads,
        "n_kv_heads": getattr(hc, "num_key_value_heads", None) or n_heads,
        "head_dim": head_dim,
        "n_layers": hc.num_hidden_layers,
        "lora_r": lora_r, "q_bit": q_bit,
        "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
        "iteration_threshold": 5,
        "attn_nnz": attn_nnz,
    }


def comparison_key(model, cfg, args, device):
    """The `fair_comparison_protocol.md` section B held-identical checklist, OBSERVED at runtime.

    B closes with: *"Add a `--assert_matched` mode to the runner that hashes lines 1-20 into a
    comparison key and refuses to write a row whose key differs from the declared cell. Every
    failure in section E would have been caught by this."*  It was specified and never built, and
    the failures kept coming: section E.1 (unmatched adapter dtype, worth 1149-2150 MiB) is the
    largest of them, and this session added another -- `probe_highwater.py` had no
    `--unsloth_bf16_adapters` flag, `build_unsloth` reads it off `args` with `getattr(..., False)`,
    so the probe silently measured unsloth with PEFT's fp32 adapters and a floor 70.69 MiB above
    ours.  Nothing raised.

    Every field here is read from the LIVE model or the live process, never from a flag, because a
    flag records what was requested and the bug is always that the request did not take effect.
    Fields that are legitimately arm-dependent (the arm name, the attention backend, whether the CE
    is fused, gradient checkpointing) are deliberately EXCLUDED from the hash and reported beside
    it -- those are the independent variable.  `PYTORCH_CUDA_ALLOC_CONF` and the tf32 flags are in
    the hash because they change the allocator and the GEMM kernel respectively.
    """
    import hashlib
    import transformers

    dtypes = {}
    for n, p in model.named_parameters():
        bucket = "adapter" if is_adapter_param(n) else "base"
        dtypes.setdefault(bucket, {}).setdefault(str(p.dtype), 0)
        dtypes[bucket][str(p.dtype)] += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    fields = {
        "model": cfg["model"],
        "model_revision": _resolved_revision(cfg["model"]),
        "n_layers": cfg["n_layers"], "hidden": cfg["hidden"],
        "intermediate": cfg["intermediate"],
        "seq": cfg["seq"], "batch": cfg["batch"],
        "lora_r": cfg["lora_r"],
        "param_dtypes": {k: dict(sorted(v.items())) for k, v in sorted(dtypes.items())},
        "n_trainable_params": trainable,
        "use_cache": bool(getattr(model.config, "use_cache", False)),
        "headless": bool(_HEADLESS["on"]),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    blob = json.dumps(fields, sort_keys=True, default=str)
    return {
        "key": hashlib.sha1(blob.encode()).hexdigest()[:16],
        "fields": fields,
        # reported, NOT hashed: these are what a head-to-head is allowed to vary
        "arm_dependent": {
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "flce": bool(getattr(args, "flce", False)),
            "adapter_dtype_requested": getattr(args, "adapter_dtype", None),
        },
    }


def _resolved_revision(model_id):
    """The commit hash the HF cache actually resolved, per protocol B.1 ("log the resolved commit
    hash from the HF cache")."""
    try:
        from huggingface_hub import snapshot_download
        return os.path.basename(os.path.realpath(snapshot_download(model_id, local_files_only=True)))
    except Exception as exc:                                     # offline node, or a local path
        return f"<unresolved: {type(exc).__name__}>"


def build_op_model(cfg):
    """Per-layer, per-step analytic cost of every instrumented region, in STEADY STATE.

    Steady state means iteration >= iteration_threshold, so the calibration paths
    (kthvalue / topk / min-max reductions in get_statistics_*) do not run. Those are measured
    separately; they are one-off per layer for the first `iteration_threshold` steps.
    """
    B, S = cfg["batch"], cfg["seq"]
    H, I = cfg["hidden"], cfg["intermediate"]
    nh, nkv, hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    r, qb, e = cfg["lora_r"], cfg["q_bit"], 2
    N = B * S
    Dkv = nkv * hd
    Amap = B * nh * S * S                     # attention map elements
    nc = int(H * cfg["layernorm_outlier_ratio"])  # outlier channels kept in bf16
    nnz = cfg["attn_nnz"]                     # MEASURED nnz of the sparsified attention map
    qbytes = qb / 8.0                         # bytes per element once packed

    def quant(nelem):
        """compression_quantization: read bf16, write packed q_bit."""
        return raw(flops=3.0 * nelem, bytes_=e * nelem + qbytes * nelem)

    def dequant(nelem):
        """decompression_dequantization: read packed q_bit, write bf16."""
        return raw(flops=2.0 * nelem, bytes_=qbytes * nelem + e * nelem)

    def lora_fwd(n, din, dout):
        """x@W + (x@A)@B, then the add. Three GEMMs + one elementwise add."""
        return merge(gemm(n, din, dout), gemm(n, din, r), gemm(n, r, dout), ew(n * dout, 2, 1, 1.0))

    def lora_bwd(n, din, dout):
        """grad_medium, grad_w_lora_a, grad_w_lora_b, grad_x, and the += fold-in."""
        return merge(gemm(n, dout, r), gemm(din, n, r), gemm(r, n, dout),
                     gemm(n, dout, din), gemm(n, r, din), ew(n * din, 2, 1, 1.0))

    m = {}
    # ---------------- forward ----------------
    m["F01.rmsnorm_1"] = ew(N * H, 1, 1, 5.0)
    m["F02.clone_x"] = ew(N * H, 1, 1, 0.0)
    m["F03.compress_x_chan"] = merge(quant(N * H), ew(N * nc, 2, 2, 0.0))
    m["F04.lora_fwd_q"] = lora_fwd(N, H, H)
    m["F05.lora_fwd_k"] = lora_fwd(N, H, Dkv)
    m["F06.lora_fwd_v"] = lora_fwd(N, H, Dkv)
    m["F07.compress_x_norm_1"] = quant(N * H)
    m["F08.compress_q_main"] = quant(N * H)
    m["F09.compress_k_main"] = quant(N * Dkv)
    m["F10.compress_v_main"] = quant(N * Dkv)
    m["F11.reshape_qkv"] = raw()                       # pure views
    m["F12.rope_qk"] = merge(ew(N * H, 1, 1, 6.0), ew(N * Dkv, 1, 1, 6.0))
    m["F13.repeat_kv_k"] = raw(bytes_=e * (N * Dkv + N * H))
    # bmm, then a SEPARATE elementwise divide over the whole [B,nh,S,S] map
    m["F14.qk_gemm"] = merge(
        {"flops": 2.0 * B * nh * S * S * hd, "gemm_flops": 2.0 * B * nh * S * S * hd,
         "bytes": e * (2 * N * H + Amap)},
        ew(Amap, 1, 1, 1.0))
    m["F15.add_mask"] = ew(Amap, 1, 1, 1.0, extra_bytes=e * B * S * S)   # + broadcast mask read
    m["F16.softmax"] = ew(Amap, 1, 1, 5.0)
    # mask=(a>thr) ; a*mask ; flatten().to_sparse() -> nonzero scan + index/value gather
    m["F17.compress_attn_map"] = merge(
        raw(flops=Amap, bytes_=e * Amap + 1.0 * Amap),                    # comparison -> bool mask
        raw(flops=Amap, bytes_=e * Amap + 1.0 * Amap + e * Amap),         # a * mask
        raw(flops=Amap, bytes_=e * Amap + nnz * (8.0 + e)))               # to_sparse (int64 idx)
    m["F18.repeat_kv_v"] = raw(bytes_=e * (N * Dkv + N * H))
    m["F19.av_gemm"] = {"flops": 2.0 * B * nh * S * S * hd, "gemm_flops": 2.0 * B * nh * S * S * hd,
                        "bytes": e * (Amap + N * H + N * H)}
    m["F20.reshape_o"] = ew(N * H, 1, 1, 0.0)          # .contiguous() copy
    m["F21.lora_fwd_o"] = lora_fwd(N, H, H)
    m["F22.compress_o"] = quant(N * H)
    m["F23.residual_1"] = ew(N * H, 2, 1, 1.0)
    m["F24.rmsnorm_2"] = ew(N * H, 1, 1, 5.0)
    m["F25.clone_x_medium"] = ew(N * H, 1, 1, 0.0)
    m["F26.compress_x_medium_chan"] = merge(quant(N * H), ew(N * nc, 2, 2, 0.0))
    m["F27.lora_fwd_gate"] = lora_fwd(N, H, I)
    m["F28.lora_fwd_up"] = lora_fwd(N, H, I)
    m["F29.compress_x_norm_2"] = quant(N * H)
    m["F30.compress_gate_main"] = quant(N * I)
    m["F31.compress_up_main"] = quant(N * I)
    m["F32.silu_hadamard"] = ew(N * I, 2, 1, 5.0)
    m["F33.lora_fwd_down"] = lora_fwd(N, I, H)
    m["F34.residual_2"] = ew(N * H, 2, 1, 1.0)

    # ---------------- backward ----------------
    m["B01.dequant_gate_main"] = dequant(N * I)
    m["B02.readd_gate_lora"] = merge(gemm(N, r, I), ew(N * I, 2, 1, 1.0))
    m["B03.silu_recompute"] = ew(N * I, 1, 1, 5.0)
    m["B04.dequant_up_main"] = dequant(N * I)
    m["B05.readd_up_lora"] = merge(gemm(N, r, I), ew(N * I, 2, 1, 1.0))
    m["B06.hadamard_recompute"] = ew(N * I, 2, 1, 1.0)
    m["B07.lora_bwd_down"] = lora_bwd(N, I, H)
    m["B08.hadamard_bwd"] = merge(ew(N * I, 2, 1, 1.0), ew(N * I, 2, 1, 1.0))
    m["B09.silu_bwd"] = ew(N * I, 2, 1, 8.0)
    m["B10.dequant_x_norm_2"] = dequant(N * H)
    m["B11.lora_bwd_gate"] = lora_bwd(N, H, I)
    m["B12.lora_bwd_up"] = merge(lora_bwd(N, H, I), ew(N * H, 2, 1, 1.0))
    m["B13.dequant_x_medium"] = merge(dequant(N * H), ew(N * nc, 1, 1, 0.0))
    m["B14.rmsnorm_bwd_2"] = ew(N * H, 2, 1, 10.0)
    m["B15.residual_bwd_2"] = ew(N * H, 2, 1, 1.0)
    m["B16.dequant_o"] = dequant(N * H)
    m["B17.lora_bwd_o"] = lora_bwd(N, H, H)
    m["B18.reshape_grad_o"] = raw()                    # view
    m["B19.attn_map_to_dense"] = raw(flops=nnz, bytes_=e * Amap + nnz * (8.0 + e) + e * nnz)
    m["B20.dequant_v_rebuild"] = merge(dequant(N * Dkv), gemm(N, r, Dkv), ew(N * Dkv, 2, 1, 1.0),
                                       raw(bytes_=e * (N * Dkv + N * H)))
    m["B21.av_gemm_bwd"] = merge(
        {"flops": 2.0 * B * nh * S * S * hd, "gemm_flops": 2.0 * B * nh * S * S * hd,
         "bytes": e * (Amap + N * H + N * H)},
        {"flops": 2.0 * B * nh * S * S * hd, "gemm_flops": 2.0 * B * nh * S * S * hd,
         "bytes": e * (N * H + N * H + Amap)},
        raw(flops=N * H, bytes_=e * (N * H + N * Dkv)))   # repeat_kv_backward sum
    m["B22.softmax_bwd"] = ew(Amap, 2, 1, 5.0)
    m["B23.scale_grad_s"] = ew(Amap, 1, 1, 1.0)
    m["B24.dequant_q_rebuild"] = merge(dequant(N * H), gemm(N, r, H), ew(N * H, 2, 1, 1.0),
                                       ew(N * H, 1, 1, 6.0))
    m["B25.qk_gemm_bwd_k"] = {"flops": 2.0 * B * nh * S * S * hd, "gemm_flops": 2.0 * B * nh * S * S * hd,
                              "bytes": e * (Amap + N * H + N * H)}
    m["B26.dequant_k_rebuild"] = merge(dequant(N * Dkv), gemm(N, r, Dkv), ew(N * Dkv, 2, 1, 1.0),
                                       ew(N * Dkv, 1, 1, 6.0),
                                       raw(bytes_=e * (N * Dkv + N * H)),      # repeat_kv k
                                       raw(flops=N * H, bytes_=e * (N * H + N * Dkv)))  # repeat_kv_bwd grad_k
    m["B27.qk_gemm_bwd_q"] = {"flops": 2.0 * B * nh * S * S * hd, "gemm_flops": 2.0 * B * nh * S * S * hd,
                              "bytes": e * (Amap + N * H + N * H)}
    m["B28.rope_bwd"] = merge(ew(N * H, 1, 1, 6.0), ew(N * Dkv, 1, 1, 6.0))
    # head_to_hidden_shape here undoes a transpose that is already present, so the reshape is a
    # view and the trailing .contiguous() is a no-op. Measured at 6.8 us/layer, i.e. no copy.
    m["B29.reshape_grad_qkv"] = raw()
    m["B30.dequant_x_norm_1"] = dequant(N * H)
    m["B31.lora_bwd_q"] = lora_bwd(N, H, H)
    m["B32.lora_bwd_k"] = merge(lora_bwd(N, H, Dkv), ew(N * H, 2, 1, 1.0))
    m["B33.lora_bwd_v"] = merge(lora_bwd(N, H, Dkv), ew(N * H, 2, 1, 1.0))
    m["B34.dequant_x"] = merge(dequant(N * H), ew(N * nc, 1, 1, 0.0))
    m["B35.rmsnorm_bwd_1"] = ew(N * H, 2, 1, 10.0)
    m["B36.residual_bwd_1"] = ew(N * H, 2, 1, 1.0)
    return m


# ----------------------------------------------------------------------------------------------
# 3. Model construction -- identical to the §29 head-to-head arm
# ----------------------------------------------------------------------------------------------

FB_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FB_FAMILIES = ("lora", "dora", "adalora", "dylora", "vera", "full", "frozen")


def arm_family(arm):
    """Arm grammar extension for the multi-adapter sweep: `<arm>+<family>`.

    `+` never appears in an existing arm name, so every previously-published arm string still
    resolves to `family='lora'` and reproduces its published number byte-for-byte. A sweep can
    therefore carry a plain-LoRA CONTROL arm (touching none of the code under test) alongside
    per-family arms in the same process.
    """
    base, _, fam = arm.partition("+")
    fam = fam or "lora"
    if fam not in FB_FAMILIES:
        raise ValueError(f"unknown adapter family {fam!r} in arm {arm!r}; expected one of "
                         f"{FB_FAMILIES}")
    return base, fam


def is_adapter_param(name):
    """Trainable-adapter name predicate that covers all five families, not just `lora_`.

    The dtype receipt and the gradient-liveness assert both key on this; classifying VeRA's
    `vera_lambda_*` as a base tensor would silently hide an unmatched adapter dtype, which
    protocol §E.1 is explicit is worth ~1150 MiB."""
    return ("lora_" in name or "vera_lambda" in name)


def apply_family(model, family, r):
    """Attach one of the seven parameterisations to a freshly loaded base model."""
    from peft import get_peft_model
    if family in ("lora", "dora"):
        from peft import LoraConfig
        return get_peft_model(model, LoraConfig(
            r=r, lora_alpha=r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=FB_TARGETS, use_dora=(family == "dora")))
    if family == "adalora":
        from peft import AdaLoraConfig
        return get_peft_model(model, AdaLoraConfig(
            init_r=r, target_r=max(r // 2, 4), lora_alpha=r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=FB_TARGETS,
            total_step=1000, tinit=20, tfinal=20, deltaT=10))
    if family == "vera":
        from peft import VeraConfig
        return get_peft_model(model, VeraConfig(
            r=r, target_modules=FB_TARGETS, vera_dropout=0.0, d_initial=0.1, bias="none",
            task_type="CAUSAL_LM"))
    if family == "dylora":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dylora import get_dylora_model
        return get_dylora_model(model, FB_TARGETS, r=r, alpha=r, dropout=0.0)
    if family == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return model
    if family == "frozen":
        # Everything in the decoder frozen; only the input embedding trains. This is the
        # raw-weight mode (CONTEXT §12 mode 7) and it is the case that exercises the fused
        # block's "no adapter factors, no dense weight grad" path while still forcing grad_x
        # through all 22 blocks.
        for p in model.parameters():
            p.requires_grad_(False)
        model.get_input_embeddings().weight.requires_grad_(True)
        return model
    raise ValueError(family)


def build_qlora_model(arm, cfg, device, use_cache=False):
    """QLoRA (Dettmers et al., NeurIPS 2023) -- NOT reimplemented.

    This is a transcription of the released construction in `temp/qlora/qlora.py`, which is the
    authors' own code, so that the row measures their method and not our reading of their paper
    (protocol §C R6/R7: state the implementation, cite file and line, re-check against the release).

      * `qlora.py:309-326` -- `from_pretrained(..., quantization_config=BitsAndBytesConfig(...))`
      * `qlora.py:376`     -- `prepare_model_for_kbit_training(...)`
      * `qlora.py:380-394` -- `LoraConfig` over `find_all_linear_names`
      * `qlora.py:396-405` -- the post-hoc dtype pass (LoRA->bf16, *norm*->fp32, head/embed->bf16)

    Their published defaults, read from the release, not from memory: `quant_type='nf4'`,
    `double_quant=True`, `bits=4`, compute dtype bf16 under `--bf16`.

    THREE DELIBERATE DEVIATIONS, each forced by the held-identical checklist (§B), each logged in
    the receipt so the row can be audited:

    1. **`use_gradient_checkpointing=False`.** §B line 13 binds gc OFF in both arms unless the
       paper is about gc; every arm in our table runs without it. Their script defaults it ON, so
       a gc=True ablation is a separate row, never this one.
    2. **`target_modules=FB_TARGETS` instead of `find_all_linear_names(model)`.** These are the
       SAME SEVEN for LLaMA -- their helper collects every `bnb.nn.Linear4bit` basename and drops
       `lm_head` (`qlora.py:248-259`) -- so this is a pin, not a change. It is pinned anyway
       because it must provably match on a non-LLaMA model too.
    3. **`r = alpha = cfg['lora_r']` (16), not their r=64/alpha=16.** §B line 9 binds rank and
       alpha across arms. Their published rank is a separate, clearly-labelled row.

    THE fp32-NORM FORK -- read this before quoting any QLoRA row.

    Their dtype pass upcasts every module whose name contains `norm` to fp32 (45 tensors here),
    and `prepare_model_for_kbit_training` does the same. HF's `LlamaRMSNorm` returns
    `self.weight * hidden.to(input_dtype)`, so an fp32 *weight* promotes every norm OUTPUT to fp32
    -- which then meets a bf16 `lm_head`. Their script survives this only because HF
    `Trainer(bf16=True)` wraps the step in `torch.autocast`, which re-casts at each matmul. Our
    harness runs pure-bf16 with no autocast, so the recipe cannot be transplanted unchanged.

    Two rows, because one row cannot answer both questions:

      `qlora_nf4`          -- norms held at bf16, matched to every other arm. This ISOLATES the
                              4-bit base weight, which is the mechanism under comparison, and is
                              the row that belongs beside our arms in a memory table.
      `qlora_nf4_norm32`   -- their fp32 norms intact, run under autocast exactly as their Trainer
                              does. This is QLoRA-as-published.

    The gap between the two rows is the cost of their dtype recipe rather than of quantisation, and
    reporting it is what §C R4 means by carrying the decomposition. **Quality runs must use
    `_norm32`** -- the fp32 norms are a stability provision for 4-bit training, and stripping them
    for a perplexity number would be measuring our variant of their method.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.tuners.lora import LoraLayer

    quant_type = "fp4" if "_fp4" in arm else "nf4"
    double_quant = "_nodq" not in arm
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        attn_implementation="sdpa",          # §C R1: matched backend against our `_sdpa` arms
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
        ),
        device_map={"": device},             # 4-bit params cannot be `.to()`-moved after load
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_r"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=FB_TARGETS))

    norm32 = "_norm32" in arm
    for name, module in model.named_modules():      # qlora.py:396-405, verbatim in effect
        if isinstance(module, LoraLayer):
            module.to(torch.bfloat16)
        if "norm" in name:
            module.to(torch.float32 if norm32 else torch.bfloat16)
        if ("lm_head" in name or "embed_tokens" in name) and hasattr(module, "weight"):
            if module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)

    # `prepare_model_for_kbit_training` upcasts EVERY non-quantised parameter to fp32, not only the
    # norms (peft: "cast all non INT8 parameters to fp32"). On the matched row that would leave a
    # silent fp32 tail -- e.g. biases -- so sweep whatever the loop above did not reach. Verified by
    # `dtype_receipt` in the result row, which is what the reader audits.
    if not norm32:
        for _n, _p in model.named_parameters():
            if _p.dtype == torch.float32 and not is_adapter_param(_n):
                _p.data = _p.data.to(torch.bfloat16)

    if norm32:
        # Their fp32 norms make every RMSNorm OUTPUT fp32, which cannot meet a bf16 `lm_head`
        # unaided. Their script survives it because HF `Trainer(bf16=True)` runs the step inside
        # `torch.autocast`, which re-casts at each matmul. Reproduce that, and ONLY that -- this is
        # transcribing their execution environment, not repairing their method. The matched row
        # (`qlora_nf4`) does not take this path at all.
        _orig_forward = model.forward

        def _autocast_forward(*a, **kw):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _orig_forward(*a, **kw)

        model.forward = _autocast_forward

    model.config.use_cache = bool(use_cache)
    model.train()
    return model, {"method": "qlora@artidoro/qlora",
                   "quant_type": quant_type, "double_quant": double_quant,
                   "bits": 4, "compute_dtype": "bfloat16",
                   "gradient_checkpointing": False, "norm_dtype": "fp32" if norm32 else "bf16",
                   "autocast": bool(norm32),
                   "target_modules": list(FB_TARGETS),
                   "r": cfg["lora_r"], "alpha": cfg["lora_r"]}


def build_minis_model(arm, cfg, device, use_cache=False):
    """Mini-Sequence Transformer -- NOT reimplemented; their `minisequence` wrapper is applied.

    Source: `temp/minis` (github.com/wdlctc/mini-s). `minis/mini_sequence.py:282` walks the module
    tree, replaces every `LlamaMLP` with `LlamaMLPWarpper` and `LlamaForCausalLM` with
    `LlamaForCausalLMWarpper` (a chunked LM head, `_LM_head` at :69), then calls
    `gradient_checkpointing_enable()` itself at :291. Canonical usage is their README:51-62.

    TWO THINGS THAT MUST TRAVEL WITH THIS ROW:

    1. **This arm must be run WITHOUT `--flce`.** Their `LMheadWarpper`/`_LM_head` IS their
       fused/chunked linear-cross-entropy; stacking Liger's FLCE on top would be two
       implementations of one optimisation and would measure neither. Regime B's requirement is
       that the head and loss are inside the measured region (§B line 14) -- they are, via their
       kernel instead of Liger's. The `--flce` guard below enforces this rather than trusting the
       caller.
    2. **`enable_input_require_grads()` is called before wrapping.** Their code targets full
       fine-tuning and calls `gradient_checkpointing_enable()` with no kwargs, i.e. transformers'
       REENTRANT default. Under PEFT that detaches every checkpointed block and silently trains
       only the LM head -- the trap documented at `apply_checkpointing`, which every `gc_*` arm in
       this harness already guards. Adding the guard makes their arm CORRECT under LoRA; omitting
       it would hand them a fake memory win from a model that is not training. The liveness gate
       (`grad_liveness.PASS`, `n_dead_gradients`) is the receipt either way.
    """
    import torch
    from transformers import AutoModelForCausalLM

    _minis = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "minis")
    if _minis not in sys.path:
        sys.path.insert(0, _minis)
    from minis.mini_sequence import minisequence

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = apply_family(model, "lora", cfg["lora_r"])
    model.to(device=device, dtype=torch.bfloat16)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()          # see (2) above -- mandatory under PEFT
    model = minisequence(model)                     # their wrapper; enables gc itself
    model.config = getattr(model.module, "config", None)
    model.config.use_cache = bool(use_cache)
    model.train()
    return model, {"method": "mini-sequence@wdlctc/mini-s",
                   "wrapped": "LlamaMLP->LlamaMLPWarpper, LlamaForCausalLM->LlamaForCausalLMWarpper",
                   "gradient_checkpointing": "enabled by their wrapper (mini_sequence.py:291)",
                   "flce": False, "enable_input_require_grads": True}


def build_lomo_model(arm, cfg, device, use_cache=False):
    """LOMO / AdaLomo -- NOT reimplemented; their optimizers from `temp/lomo/lomo_optim`.

    ⚠⚠ **DIFFERENT REGIME, same caveat as GaLore**: full-parameter fine-tuning of all ~1.1B
    parameters, against 12.6M for every LoRA arm. `n_trainable_params` is in the receipt.

    LOMO **fuses the parameter update into the backward** so gradients never all materialise and
    there is no optimizer state at all; `AdaLomo` adds a second moment. Their call is
    `optimizer.fused_backward(loss, lr)` in place of `loss.backward()` + `opt.step()`
    (`lomo/src/lomo_trainer.py:172`), which is why `step()` needs a branch — the third baseline in
    this campaign to own its own backward, after DeepSpeed and StreamBP.

    Arms: `lomo` (no optimizer state -- the aggressive memory claim) and `adalomo` (their README:27
    recommendation, "AdaLomo without gradnorm ... better performance and higher throughput").
    """
    import torch
    from transformers import AutoModelForCausalLM

    _l = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "lomo")
    if _l not in sys.path:
        sys.path.insert(0, _l)
    from lomo_optim import Lomo, AdaLomo

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)                        # full fine-tuning -- their regime
    model.to(device=device, dtype=torch.bfloat16)
    _ck = apply_checkpointing(model, arm)
    if _ck:
        model._ckpt_receipt = _ck
    model.config.use_cache = bool(use_cache)
    model.train()

    # `..._fb` composes our fused decoder block with their optimizer. The pools are disjoint in
    # principle -- they remove OPTIMIZER STATE (and, for LOMO, gradient residency), we remove
    # ACTIVATIONS -- so this is the composition question worth measuring rather than arguing about.
    if "_fb" in arm:
        from flashffn import apply_flash_block, apply_flash_final_norm
        apply_flash_block(model, keep="min")
        apply_flash_final_norm(model)

    lr = 3e-4
    cls = AdaLomo if arm.startswith("adalomo") else Lomo
    model._lomo_opt = cls(model, lr=lr)               # gradnorm left off, per README:27
    model._lomo_lr = lr
    return model, {"method": f"{cls.__name__.lower()}@OpenLMLab/LOMO",
                   "regime": "FULL fine-tuning", "fused_backward": True, "lr": lr,
                   "optimizer_state": "none" if cls is Lomo else "second moment only",
                   "n_trainable_params": sum(p.numel() for p in model.parameters()
                                             if p.requires_grad)}


def build_galore_model(arm, cfg, device, use_cache=False):
    """GaLore -- NOT reimplemented; their `GaLoreAdamW` from `temp/galore/galore_torch`.

    ⚠⚠ **THIS IS A DIFFERENT TRAINING REGIME AND ITS ROW MUST NEVER SIT UNLABELLED BESIDE A LoRA
    ARM.** GaLore is FULL-PARAMETER fine-tuning with a low-rank *projection of the gradient* so the
    optimizer moments are small. It trains **all ~1.1B parameters**; every other arm in this
    campaign trains 12.6M adapter parameters. It answers "can I full-finetune in LoRA-like memory?",
    not "whose activation cache is smaller". Quality is likewise not comparable at matched steps.
    The row carries `n_trainable_params` so the asymmetry is visible in the data, not just prose.

    Their published defaults, from README:52-55: 2-D weights get
    `rank=128, update_proj_gap=200, scale=0.25, proj_type='std'`; everything else (norms,
    embeddings, biases) goes in a plain group. `GaLoreAdamW` is then constructed over both.
    """
    import torch
    from transformers import AutoModelForCausalLM

    _g = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "galore")
    if _g not in sys.path:
        sys.path.insert(0, _g)
    from galore_torch import GaLoreAdamW  # noqa: F401  (constructed at the optimizer site)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)                       # full fine-tuning -- their regime
    model.to(device=device, dtype=torch.bfloat16)
    _ck = apply_checkpointing(model, arm)
    if _ck:
        model._ckpt_receipt = _ck
    model.config.use_cache = bool(use_cache)
    model.train()

    # Their split: 2-D weights are projected, everything else is not (README:52-55).
    target = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # `..._fb` composes the fused decoder block with GaLore. GaLore replaces only the OPTIMIZER, so
    # the module tree is untouched and the block applies cleanly over the full-FT model.
    if "_fb" in arm:
        from flashffn import apply_flash_block, apply_flash_final_norm
        apply_flash_block(model, keep="min")
        apply_flash_final_norm(model)

    galore_p, other_p = [], []
    for n, p in model.named_parameters():
        (galore_p if (p.dim() == 2 and any(t in n for t in target)) else other_p).append(p)
    model._galore_groups = [
        {"params": other_p},
        {"params": galore_p, "rank": 128, "update_proj_gap": 200, "scale": 0.25,
         "proj_type": "std"},
    ]
    return model, {"method": "galore@jiaweizzhao/GaLore", "regime": "FULL fine-tuning",
                   "rank": 128, "update_proj_gap": 200, "scale": 0.25, "proj_type": "std",
                   "n_galore_tensors": len(galore_p), "n_plain_tensors": len(other_p),
                   "n_trainable_params": sum(p.numel() for p in model.parameters()
                                             if p.requires_grad)}


def build_alst_model(arm, cfg, device, use_cache=False):
    """ALST / TiledMLP (Arctic Long Sequence Training) -- NOT reimplemented.

    Their `enable_tiled_mlp_compute` is called verbatim from
    `temp/arctic/arctic_training/model/tiled_compute.py:84`, which sets
    `LlamaMLP.forward = tiled_mlp_forward_common` (`:48`), which shards the MLP along the sequence
    axis and calls **`deepspeed.runtime.sequence_parallel.ulysses_sp.TiledMLP`** -- the mechanism
    itself lives in DeepSpeed, not in ArcticTraining.

    TWO INFRASTRUCTURE NOTES:

    * **DeepSpeed 0.17.6 is installed SIDE-BY-SIDE at `temp/ds_alst`** and prepended to `sys.path`
      for this arm only. The env's DeepSpeed is 0.16.5, which predates `sequence_parallel` entirely;
      upgrading in place would have silently changed the already-measured `zero3_*` rows. Same
      pattern as `temp/unsloth_pkgs` / `temp/liger_pkgs`.
    * **Their forward `all_reduce`s the shard count across ranks**, so a process group is required
      even on one GPU.

    ⚠ **THEIR `num_shards="auto"` RULE MAKES THIS A NO-OP AT SHORT SEQUENCE.** `tiled_compute.py:54`
    computes `num_shards = ceil(seqlen / hidden)`; with TinyLlama's hidden=2048 that is **1 shard for
    seq 1024 and 2048** — i.e. no tiling at all — 2 at 4096, 4 at 8192, 8 at 16384. This is their own
    rule, unmodified, and the short-sequence rows must be reported as "their policy declines to
    tile", NOT as a method that tiled and did not help.

    ⚠ The patch is applied to the `LlamaMLP` CLASS and therefore persists for the whole process.
    Run this arm in its own process; do not batch it behind another arm with `--arm a,b`.
    """
    import torch
    import torch.distributed as dist
    import importlib.util
    from transformers import AutoModelForCausalLM

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_alst = os.path.join(root, "temp", "ds_alst")
    if ds_alst not in sys.path:
        sys.path.insert(0, ds_alst)
    # `deepspeed` is already in sys.modules by this point (transformers/accelerate import it at
    # startup), so the path insert above CANNOT swap the version -- it must be on PYTHONPATH before
    # the interpreter starts, exactly as `temp/unsloth_pkgs` is. Fail loudly with the fix rather
    # than let the import error surface 40 frames down.
    import deepspeed as _ds
    if not os.path.abspath(_ds.__file__).startswith(os.path.abspath(ds_alst)):
        raise RuntimeError(
            f"ALST needs DeepSpeed >= 0.17 for `runtime.sequence_parallel.ulysses_sp.TiledMLP`, but "
            f"the loaded deepspeed is {_ds.__version__} from {os.path.dirname(_ds.__file__)}. "
            f"Re-run with:  PYTHONPATH=temp/ds_alst python src/profile_unsloth.py --arm {arm} ... "
            f"(the env's 0.16.5 is kept deliberately so the measured `zero3_*` rows stay valid).")
    _tc = os.path.join(root, "temp", "arctic", "arctic_training", "model", "tiled_compute.py")
    _spec = importlib.util.spec_from_file_location("_alst_tiled_compute", _tc)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    _mod.enable_tiled_mlp_compute(cfg["model"])     # their entry point, patches the class
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = apply_family(model, "lora", cfg["lora_r"])
    model.to(device=device, dtype=torch.bfloat16)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    _ck = apply_checkpointing(model, arm)
    if _ck:
        model._ckpt_receipt = _ck
    model.config.use_cache = bool(use_cache)
    model.train()
    seq = int(cfg.get("seq", 1024))
    hidden = hf_config(model).hidden_size
    import math as _math
    return model, {"method": "alst-tiledmlp@snowflakedb/ArcticTraining",
                   "kernel": "deepspeed.runtime.sequence_parallel.ulysses_sp.TiledMLP",
                   "deepspeed_for_this_arm": "0.17.6 (temp/ds_alst)",
                   "num_shards_auto": _math.ceil(seq / hidden),
                   "tiling_active": _math.ceil(seq / hidden) > 1}


def build_streambp_model(arm, cfg, device, use_cache=False):
    """StreamBP (2025) -- NOT reimplemented; their `StreamModel` wrapper is applied.

    Source: `temp/streambp` (github.com/Ledzy/StreamBP). Canonical construction is their
    `scripts/test_bp.py:38-40` and readme:98-107:

        StreamModel(model, gradient_accumulation_steps=..., logits_chunk_size=100,
                    checkpoint_chunk_size=<seq/3>, stream_checkpoint=True)
        model.gradient_checkpointing_enable()

    Their published knobs are used verbatim: `logits_chunk_size=100` and
    **`checkpoint_chunk_size = sequence_len // 3`, which is the value their own readme recommends**
    ("Suggested value: sequence_len/3"). A `fb_streambp:<k>` suffix overrides it for a sweep.

    THREE ADAPTATIONS, each forced and each disclosed:

    1. **The raw `LlamaForCausalLM` is handed to `StreamModel`, not the `PeftModel`.**
       `_setup_stream_forward` (`stream_model.py:436`) indexes `self.model.model.layers`, which
       assumes a bare HF CausalLM; under PEFT that path is `PeftModel.model.model.model.layers` and
       raises. `get_peft_model` MUTATES the base model in place (it swaps target `nn.Linear` for
       `lora.Linear`), so the raw object already carries the adapters and plain-LoRA's
       `PeftModelForCausalLM.forward` is a pass-through to it. **This is a wiring fix for a
       PEFT/StreamBP incompatibility, not a change to their algorithm.**
    2. **`attn_implementation="flash_attention_2"`.** Their `StreamAttention` calls
       `_flash_attention_forward` directly. Protocol §C R1 explicitly contemplates this: the
       head-to-head backend is FlashAttention, "flash_attn varlen for their path, sdpa/FA for ours",
       and our `_sdpa` arms dispatch to the same vendored FA-2 kernel.
    3. **Must run WITHOUT `--flce`** -- `logits_chunk_size` IS their chunked LM head, exactly as
       with Mini-Sequence. Guarded in `profile_unsloth.build_other`.
    """
    import torch
    from transformers import AutoModelForCausalLM

    # `streambp/__init__.py` imports their DPO/GRPO/SFT trainers, which import `trl`. Installing
    # trl here is not safe: it pins/upgrades `transformers`, and every published number in this
    # repo is on 4.51.3. Load their module file directly instead -- `stream_model.py` has no
    # relative imports, so it stands alone and their algorithm is untouched.
    import importlib.util
    _sbp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "temp", "streambp", "src", "streambp", "stream_model.py")
    _spec = importlib.util.spec_from_file_location("_streambp_stream_model", _sbp)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    StreamModel = _mod.StreamModel

    seq = int(cfg.get("seq", 1024))
    _, _, tail = arm.partition(":")
    chunk = int(tail) if tail.isdigit() else max(1, seq // 3)

    llama = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    llama.config.use_cache = False
    apply_family(llama, "lora", cfg["lora_r"])       # mutates `llama` in place; see (1)
    llama.to(device=device, dtype=torch.bfloat16)
    if hasattr(llama, "enable_input_require_grads"):
        llama.enable_input_require_grads()           # same PEFT+checkpointing trap as Mini-Sequence
    engine = StreamModel(llama, gradient_accumulation_steps=1, logits_chunk_size=100,
                         checkpoint_chunk_size=chunk, stream_checkpoint=True)
    engine.gradient_checkpointing_enable()
    engine.train()
    return engine, {"method": "streambp@Ledzy/StreamBP",
                    "logits_chunk_size": 100, "checkpoint_chunk_size": chunk,
                    "chunk_rule": "seq//3 (their readme)", "stream_checkpoint": True,
                    "attn_implementation": "flash_attention_2", "flce": False}


def hf_config(model):
    """Resolve the HuggingFace `PretrainedConfig` through any number of wrapper layers.

    Baseline integrations wrap the model -- a DeepSpeed engine, StreamBP's `StreamModel`,
    Mini-Sequence's `minisequence` -- and some put their OWN object on `.config`: a
    `DeepSpeedEngine.config` is the DeepSpeed JSON, so `model.config.vocab_size` raises there.
    Walk `.module` / `.model` until a real HF config appears.
    """
    from transformers import PretrainedConfig
    m, seen = model, set()
    for _ in range(8):
        c = getattr(m, "config", None)
        if isinstance(c, PretrainedConfig):
            return c
        nxt = getattr(m, "module", None)
        if nxt is None:
            nxt = getattr(m, "model", None)
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        m = nxt
    raise AttributeError(f"no HuggingFace config reachable from {type(model).__name__}")


def build_zero3_model(arm, cfg, device, use_cache=False):
    """DeepSpeed ZeRO-3 with CPU parameter offload -- the published owner of WP-E's mechanism.

    NOT reimplemented: this is `deepspeed.initialize` with a stage-3 config. ZeRO-Offload
    (ATC'21) offloads OPTIMIZER STATE and GRADIENTS, which under LoRA is ~48 MiB and therefore
    not the comparison we need; **`offload_param` (ZeRO-Infinity, SC'21) is the one that moves the
    frozen base weights**, which is exactly what `fb_min_wstream` does. That is the row.

    Arms:
      `zero3`          -- stage 3, params resident (partitioning is a no-op on one GPU)
      `zero3_offload`  -- stage 3 + `offload_param={"device":"cpu","pin_memory":true}`  <- the one

    Fairness notes, all forced by §B:
      * bf16 engine, LoRA r=alpha=16 over the same seven projections, dropout 0, no gc -- matched.
      * DeepSpeed owns the optimizer; `step()` routes through `engine.backward`/`engine.step`
        because ZeRO's hooks fire there and not from `loss.backward()`.
      * `train_micro_batch_size_per_gpu` must equal the harness batch or DS silently rescales the
        loss, which would move the quality column without moving the memory column.
    """
    import torch
    from transformers import AutoModelForCausalLM
    import deepspeed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29529")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        deepspeed.init_distributed(dist_backend="nccl")

    offload = "offload" in arm
    ds_config = {
        "train_micro_batch_size_per_gpu": cfg.get("batch", 2),
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_max_live_parameters": 1e8,
        },
        "optimizer": {"type": "AdamW", "params": {"lr": 2e-4}},
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
    }
    if offload:
        ds_config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = apply_family(model, "lora", cfg["lora_r"])
    model.config.use_cache = bool(use_cache)
    # `..._fb` composes the fused decoder block with ZeRO-3. It must be applied BEFORE
    # `deepspeed.initialize`, for the same reason checkpointing must be: afterwards the engine owns
    # the decoder layers and the patcher would not see them.
    if "_fb" in arm:
        from flashffn import apply_flash_block, apply_flash_final_norm
        apply_flash_block(model, keep="min")
        apply_flash_final_norm(model)
    # Checkpointing must be applied to the HF module BEFORE `deepspeed.initialize` wraps it --
    # afterwards the decoder layers are owned by the engine and the wrapper would not see them.
    # ZeRO-3 partitions/offloads PARAMETERS and does nothing about activations, so without this a
    # `zero3_*` row is a no-checkpointing arm and is not comparable with `fb_min` or `liger_gc_sdpa`
    # (§B line 13). `zero3_offload_gc_manual` is therefore the row that belongs in the table.
    _ck = apply_checkpointing(model, arm)
    engine, _opt, _, _ = deepspeed.initialize(
        model=model, model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config)
    engine.train()
    return engine, {"method": "deepspeed-zero3@deepspeedai/DeepSpeed",
                    "version": deepspeed.__version__, "stage": 3,
                    "offload_param": bool(offload), "bf16": True,
                    "checkpointing": _ck}


def build_model(arm, cfg, device, adapter_dtype="bf16", use_cache=False):
    """`adapter_dtype`:
        'bf16' -- the whole model (adapters included) is cast to bf16. This is what this harness
                  has always done, and what HyC-LoRA's own patch does to the adapters anyway.
        'fp32' -- adapters left in PEFT's default fp32, which is what `train_glue.py` actually
                  runs for every non-HyC-LoRA arm (it never re-casts after `get_peft_model`).
    """
    from transformers import AutoModelForCausalLM

    arm, family = arm_family(arm)
    if arm.startswith("qlora"):
        if family != "lora":
            raise ValueError(f"{arm!r}: QLoRA is defined over plain LoRA only; "
                             f"family={family!r} would be our extension of their method, not theirs")
        model, _qreceipt = build_qlora_model(arm, cfg, device, use_cache=use_cache)
        # Route through the SAME checkpointing tail as every other arm, so that
        # `qlora_nf4_gc_manual` gets byte-identical treatment to `gc_manual_sdpa` and
        # `liger_gc_sdpa` and the gc axis is a one-variable change (§B line 13).
        _ck = apply_checkpointing(model, arm)
        if _ck:
            model._ckpt_receipt = _ck
        model._qlora_receipt = _qreceipt
        return model
    if arm.startswith("minis"):
        if family != "lora":
            raise ValueError(f"{arm!r}: Mini-Sequence is applied over plain LoRA here; "
                             f"family={family!r} would be our extension, not theirs")
        model, _mreceipt = build_minis_model(arm, cfg, device, use_cache=use_cache)
        model._minis_receipt = _mreceipt
        return model
    if arm.startswith("lomo") or arm.startswith("adalomo"):
        model, _lreceipt = build_lomo_model(arm, cfg, device, use_cache=use_cache)
        model._lomo_receipt = _lreceipt
        return model
    if arm.startswith("galore"):
        if family != "lora":
            raise ValueError(f"{arm!r}: GaLore is a full-fine-tuning method; the `+family` grammar "
                             f"does not apply (got {family!r})")
        model, _greceipt = build_galore_model(arm, cfg, device, use_cache=use_cache)
        model._galore_receipt = _greceipt
        return model
    if arm.startswith("alst"):
        if family != "lora":
            raise ValueError(f"{arm!r}: ALST row is defined over plain LoRA; family={family!r} "
                             f"would be our extension, not theirs")
        model, _areceipt = build_alst_model(arm, cfg, device, use_cache=use_cache)
        model._alst_receipt = _areceipt
        return model
    if arm.startswith("streambp"):
        if family != "lora":
            raise ValueError(f"{arm!r}: StreamBP row is defined over plain LoRA; family={family!r} "
                             f"would be our extension, not theirs")
        engine, _sreceipt = build_streambp_model(arm, cfg, device, use_cache=use_cache)
        engine._streambp_receipt = _sreceipt
        return engine
    if arm.startswith("zero3"):
        if family != "lora":
            raise ValueError(f"{arm!r}: ZeRO-3 row is defined over plain LoRA; family={family!r} "
                             f"would be our extension, not theirs")
        engine, _zreceipt = build_zero3_model(arm, cfg, device, use_cache=use_cache)
        engine._zero3_receipt = _zreceipt
        return engine
    # Clear the `o_h` staging state for EVERY arm, not just fused-block ones.  `fb_offload`'s
    # buffer pools are process-global, so a landing buffer left over from an offloaded arm stays
    # resident while the next arm is measured and lands in its peak -- 64.00 MiB on both `fb_attn`
    # and `fb_min` at seq 4096 before this existed, which is enough to move a published column.
    import fb_offload as _fbo
    _fbo.reset()
    _fbo.fb_offload_enable(False)
    # HyC-LoRA's FA variant expresses causality with `is_causal`, so it must NOT be handed a 4-D
    # additive mask; loading with sdpa makes transformers pass `attention_mask=None` for an
    # unpadded batch, which is exactly what that path wants. (`patch.py` validates this and
    # raises if a real padding pattern ever shows up.)
    attn_impl = ("sdpa" if (arm.endswith("_sdpa") or arm.startswith("hyclora_flash")
                            or arm.startswith("fb_")) else "eager")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation=attn_impl, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = apply_family(model, family, cfg["lora_r"])
    model.to(device=device, dtype=torch.bfloat16)
    if adapter_dtype == "fp32":
        for _n, _p in model.named_parameters():
            if is_adapter_param(_n):
                _p.data = _p.data.to(torch.float32)
    model.config.use_cache = bool(use_cache)
    model.train()

    if arm.startswith("hyclora"):
        from hyclora.patch import HyCLoRAConfig, apply_hyclora, reset_counters
        reset_counters()
        # q_bit comes from the arm name (hyclora_q4 / hyclora_q2 / hyclora_q8), not from the
        # shared cfg dict -- otherwise every hyclora arm silently runs at the same bit width.
        # Matched by regex, not by `split("_q")[-1]`, so a suffixed arm name such as
        # `hyclora_q2_gc_manual` (their method stacked with checkpointing, which their own paper
        # Table 5(c) advocates) still resolves the bit width instead of raising.
        _m = re.search(r"_q(\d+)", arm)
        qb = int(_m.group(1)) if _m else cfg["q_bit"]
        cfg["q_bit"] = qb
        layer_type = "intra_inter_flash" if arm.startswith("hyclora_flash") else "intra_inter"
        # `hyclora_flash_nc` = their FUSED FlashAttention layer with every codec bypassed. It is
        # the control that separates "fusion deletes framework waste" from "their compression is
        # better": same layer boundaries, same FA2 kernel, same recompute of the SwiGLU
        # intermediates, zero quantisation. (6952 - nc) is what any fused implementation gets for
        # free on this stack; (nc - q2) is what their compression actually buys.
        no_compress = arm.endswith("_nc")
        hc = HyCLoRAConfig(
            use_hyclora=True, layer_type=layer_type, q_bit=qb,
            softmax_outlier_ratio=cfg["softmax_outlier_ratio"],
            layernorm_outlier_ratio=cfg["layernorm_outlier_ratio"],
            iteration_threshold=cfg["iteration_threshold"],
            no_compress=no_compress,
        )
        apply_hyclora(model, hc, verbose=False)
    elif arm.startswith("v3"):
        mode = "int4" if "int4" in arm else "recompute"
        from train_glue import apply_flash_ffn_v3
        apply_flash_ffn_v3(model, mode)
    elif arm.startswith("fb_"):
        # Fused decoder-block arms (`llmdocs/trackers/fused_block.md`). Grammar:
        #   fb_<keep>[_fnorm]_sdpa    keep in {min, glu, full}
        # `_fnorm` additionally routes the model-level final RMSNorm through the same fused
        # kernel: stock LlamaRMSNorm saves a [B,S,H] fp32 upcast (16 MiB at seq1024/batch2) where
        # the fused form saves the bf16 input + rstd (8 MiB). It is the SAME class-(a) elimination
        # the block does, applied to the one norm that lives outside every decoder block, and it
        # is a separate arm so the two effects are never conflated.
        from flashffn import apply_flash_block, apply_flash_final_norm, fb_reset_counters
        fb_reset_counters()
        # `fb_auto[:<seq>]` pins the auto policy's threshold, e.g. `fb_auto:2048_fnorm_sdpa`;
        # `fb_attn` / `fb_min` / `fb_glu` / `fb_full` pin the level outright.  The policy and every
        # resolution it makes are printed and land in the result row (`fb_policy`), so no memory or
        # throughput number in this project can come from a keep level nobody can see.
        keep = next((k for k in ("full", "glu", "attn", "min", "auto") if f"fb_{k}" in arm), None)
        if keep is None:
            raise ValueError(f"fused-block arm {arm!r} must name a keep level: "
                             f"fb_min/fb_attn/fb_glu/fb_full/fb_auto")
        auto_seq = None
        if keep == "auto":
            _m = re.search(r"fb_auto:(\d+)", arm)
            auto_seq = int(_m.group(1)) if _m else None
        apply_flash_block(model, keep=keep, auto_seq=auto_seq, verbose=True)
        if "fnorm" in arm:
            apply_flash_final_norm(model)
        # `_offload` stages `o_h` through pinned host memory over the otherwise-idle copy engines
        # (`route_b.md` §2.6; the mechanism is MEMO's, arXiv 2407.12117, and must be cited as
        # theirs).  It only does anything at keep='attn' -- that is the only level that stores
        # `o_h` -- so naming it with any other level is a configuration error, not a silent no-op.
        import fb_offload
        want_offload = "offload" in arm
        if want_offload and keep not in ("attn", "auto"):
            raise ValueError(f"fused-block arm {arm!r} asks for the o_h offload at keep={keep!r}; "
                             f"only keep='attn' stores o_h, so the flag would do nothing")
        fb_offload.fb_offload_enable(want_offload)     # `build_model` already reset the pools
        if want_offload:
            print(f"    o_h offload ON (pinned host staging, keep={keep})")

    # Checkpointing is applied LAST so it wraps whatever compression the arm also uses. The
    # returned dict is a receipt that goes into the result row: an arm that silently failed to
    # get checkpointed would otherwise look like an ordinary (and wrong) memory number.
    ck = apply_checkpointing(model, arm)
    if ck:
        model._ckpt_receipt = ck
    return model


# ----------------------------------------------------------------------------------------------
# 3b. The "boring baselines": gradient checkpointing and selective activation checkpointing
# ----------------------------------------------------------------------------------------------
#
# WHY THESE ARMS EXIST
# --------------------
# Plain gradient checkpointing (Chen et al. 2016) stores only each block's input and recomputes
# the rest in backward. It is bit-exact, it is one line of code, and at TinyLlama/seq1024/batch2
# it stores ~8 MiB/layer -- LESS than HyC-LoRA's FlashAttention variant is estimated to store at
# q_bit=2. If it is also within 5% of HyC-LoRA on throughput then "beat HyC-LoRA on memory while
# staying exact and within 5% on speed" is already solved by a flag, and any new method has to
# beat THIS, not the uncompressed baseline. Selective activation checkpointing (SAC) is the
# stronger form and the one Megatron/torchtitan actually ship, so it is the real frontier.
#
# Five granularities are implemented, all at bf16 adapters, sdpa attention, dropout 0:
#   gc_hf     -- `model.gradient_checkpointing_enable()` at transformers' default (REENTRANT)
#                plus `enable_input_require_grads()`. This is verbatim what `train_glue.py`
#                does (L1895-1915) and it carries the documented PEFT-detachment trap.
#   gc_full   -- the same but `use_reentrant=False`, torch's current recommendation.
#   gc_manual -- our own wrapper around `torch.utils.checkpoint` at decoder-layer granularity,
#                non-reentrant. Exists so that SAC-vs-full-checkpointing is a one-variable
#                comparison (same wrapper, only the policy changes).
#   sac_dots  -- the same wrapper plus `create_selective_checkpoint_contexts` with a
#                `dots_saveable`-style policy: MUST_SAVE the output of every matmul and of SDPA,
#                PREFER_RECOMPUTE everything else (norms, SiLU, the GLU product, RoPE, residual
#                adds). This is JAX's `checkpoint_dots` / torchtitan's `_save_list`.
#   gc_attn / gc_ffn -- checkpoint only the attention sub-block, or only the FFN sub-block.
#                Block-granularity selectivity, i.e. Korthikanti-style partial recomputation.

_SAC_SAVE_OPS = None


def _sac_save_ops():
    """The `dots_saveable` save-list: matmul-family and SDPA outputs are kept, everything else in
    the checkpointed region is recomputed. Names are resolved defensively because the SDPA
    overloads that exist depend on the torch build."""
    global _SAC_SAVE_OPS
    if _SAC_SAVE_OPS is not None:
        return _SAC_SAVE_OPS
    a = torch.ops.aten
    wanted = [
        "mm.default", "addmm.default", "bmm.default", "baddbmm.default", "matmul.default",
        "_scaled_dot_product_efficient_attention.default",
        "_scaled_dot_product_flash_attention.default",
        "_scaled_dot_product_cudnn_attention.default",
        "_scaled_dot_product_attention_math.default",
    ]
    ops = set()
    for w in wanted:
        base, ov = w.split(".")
        try:
            ops.add(getattr(getattr(a, base), ov))
        except AttributeError:
            pass
    _SAC_SAVE_OPS = ops
    return ops


# Diagnostic: which ops the policy actually saved / recomputed on the FIRST forward it saw. If
# this comes back empty the policy never fired and the "SAC" arm is really plain checkpointing.
SAC_STATS = defaultdict(int)


def _sac_context_fn():
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    save_ops = _sac_save_ops()

    def policy(ctx, func, *args, **kwargs):
        keep = func in save_ops
        if not ctx.is_recompute and len(SAC_STATS) < 400:
            SAC_STATS[("SAVE " if keep else "RECOMP ") + str(func)] += 1
        return CheckpointPolicy.MUST_SAVE if keep else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


def _decoder_layers(model):
    """The `nn.ModuleList` of decoder blocks, however deep PEFT has buried it."""
    for _n, m in model.named_modules():
        if type(m).__name__.endswith("DecoderLayer"):
            parent = m
            break
    else:
        raise RuntimeError("no DecoderLayer found")
    out = []
    for n, m in model.named_modules():
        if type(m).__name__ == type(parent).__name__:
            out.append((n, m))
    return out


def _wrap_forward_checkpoint(mod, context_fn=None, reentrant=False):
    """Route a module's forward through `torch.utils.checkpoint`.

    `preserve_rng_state=False` is safe and is used here because every arm runs with
    `lora_dropout=0` and `attention_dropout=0`, so the checkpointed region consumes no RNG; the
    recompute is bit-identical without state capture. (torchtitan's SAC wrapper does the same.)
    Eval/no-grad forwards bypass checkpointing entirely, which is what every framework does.
    """
    orig = mod.forward

    def fwd(*args, **kwargs):
        if not (mod.training and torch.is_grad_enabled()):
            return orig(*args, **kwargs)
        kw = {"use_reentrant": reentrant, "preserve_rng_state": False}
        if context_fn is not None:
            kw["context_fn"] = context_fn
        return torch.utils.checkpoint.checkpoint(orig, *args, **kw, **kwargs)

    mod.forward = fwd


# name-fragment -> (mechanism, granularity, reentrant, policy). Longest fragment wins so that
# `gc_full` is not shadowed by `gc_f`.
CKPT_SPECS = {
    "gc_hf":     ("hf",   "layer", True,  None),
    "gc_full":   ("hf",   "layer", False, None),
    "gc_manual": ("wrap", "layer", False, None),
    "sac_dots":  ("wrap", "layer", False, "dots"),
    "gc_attn":   ("wrap", "attn",  False, None),
    "gc_ffn":    ("wrap", "mlp",   False, None),
}


def checkpoint_spec(arm):
    for frag in sorted(CKPT_SPECS, key=len, reverse=True):
        if frag in str(arm):
            return frag, CKPT_SPECS[frag]
    return None, None


def apply_checkpointing(model, arm):
    """Apply the checkpointing variant named by the arm. Returns a receipt, or None."""
    frag, spec = checkpoint_spec(arm)
    if spec is None:
        return None
    mech, where, reentrant, policy = spec
    layers = _decoder_layers(model)
    receipt = {"variant": frag, "mechanism": mech, "granularity": where,
               "use_reentrant": reentrant, "policy": policy or "full",
               "n_decoder_layers": len(layers), "n_wrapped": 0,
               "enable_input_require_grads": False}

    if mech == "hf":
        # Exactly what train_glue.py:1895-1915 does. `enable_input_require_grads()` is MANDATORY
        # for the reentrant variant under PEFT: without it the embedding output does not require
        # grad, every checkpointed block is detached, and only the LM head trains -- a silent
        # bug that already shipped once in this repo (CONTEXT.md §14.5). Applied for both
        # variants so the two rows differ in exactly one thing.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": bool(reentrant)})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            receipt["enable_input_require_grads"] = True
        receipt["n_wrapped"] = len(layers)
        # verify transformers really turned it on, rather than trusting the call
        flags = [bool(getattr(m, "gradient_checkpointing", False))
                 for _n, m in model.named_modules()
                 if hasattr(m, "gradient_checkpointing")]
        receipt["gradient_checkpointing_flags_set"] = sum(1 for f in flags if f)
        if not any(flags):
            raise RuntimeError("gradient_checkpointing_enable() did not set any flag")
        return receipt

    if policy == "dots":
        SAC_STATS.clear()
    ctx_fn = _sac_context_fn if policy == "dots" else None
    for _n, layer in layers:
        if where == "layer":
            _wrap_forward_checkpoint(layer, context_fn=ctx_fn, reentrant=reentrant)
            receipt["n_wrapped"] += 1
        elif where == "attn":
            _wrap_forward_checkpoint(layer.self_attn, context_fn=ctx_fn, reentrant=reentrant)
            receipt["n_wrapped"] += 1
        elif where == "mlp":
            _wrap_forward_checkpoint(layer.mlp, context_fn=ctx_fn, reentrant=reentrant)
            receipt["n_wrapped"] += 1
    if receipt["n_wrapped"] == 0:
        raise RuntimeError(f"checkpoint wrapper matched no modules for arm {arm}")
    return receipt


def is_checkpointed(arm):
    return checkpoint_spec(arm)[1] is not None


def is_nonreentrant_checkpointed(arm):
    """Non-reentrant checkpointing replaces saved tensors with holders; READING a graph node's
    `_saved_*` property triggers the recompute. So the generic graph-walking inventory and the
    per-aten-op allocator timeline must not be run on these arms -- they would both report
    recomputed tensors as if they were stored AND perturb the memory they are measuring."""
    _f, spec = checkpoint_spec(arm)
    return bool(spec) and not spec[2]


# ----------------------------------------------------------------------------------------------
# 3c. Per-layer RETAINED bytes -- the one stored-bytes definition that is arm-independent
# ----------------------------------------------------------------------------------------------
#
# The graph-walking inventory in §4b cannot be used on a checkpointed arm (see above), and the
# `saved_tensors` walk in §4 only works on HyC-LoRA's fused Function. This probe works on every
# arm because it does not look at the graph at all: it reads `torch.cuda.memory_allocated()` at
# each decoder-layer boundary during a real grad-enabled forward. The delta across one layer is,
# by construction, exactly the bytes that layer still holds when its forward returns -- whether
# those bytes live in `save_for_backward`, in a checkpoint frame's captured inputs, in a
# quantised buffer, or on a ctx attribute. Transients are already freed at the boundary.

def per_layer_retained_report(model, batch):
    layers = _decoder_layers(model)
    marks, handles = [], []

    def post(mod, inp, out):
        marks.append(torch.cuda.memory_allocated())

    for _n, l in layers:
        handles.append(l.register_forward_hook(post))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    try:
        out = model(**batch)
        torch.cuda.synchronize()
        after_fwd = torch.cuda.memory_allocated()
    finally:
        for h in handles:
            h.remove()
    if len(marks) < 3:
        del out
        return {"error": f"only {len(marks)} layer boundaries seen"}
    deltas = [marks[i] - marks[i - 1] for i in range(1, len(marks))]
    srt = sorted(deltas)
    rep = {
        "n_layer_boundaries": len(marks),
        "retained_MiB_per_layer_median": srt[len(srt) // 2] / 2 ** 20,
        "retained_MiB_per_layer_min": srt[0] / 2 ** 20,
        "retained_MiB_per_layer_max": srt[-1] / 2 ** 20,
        "retained_MiB_layer0_incl_embedding": (marks[0] - before) / 2 ** 20,
        "retained_MiB_all_layers": (marks[-1] - before) / 2 ** 20,
        "retained_MiB_whole_forward_incl_head": (after_fwd - before) / 2 ** 20,
        "per_layer_MiB": [d / 2 ** 20 for d in deltas],
    }
    # drain the graph so nothing survives into the next measurement
    out.loss.backward()
    del out
    for p in model.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    return rep


def adapter_grad_liveness(model, batch, n_layers):
    """THE anti-trap check. `gradient_checkpointing_enable()` without `enable_input_require_grads()`
    silently detaches every checkpointed block under PEFT, so only the LM head trains -- and an
    arm that trains only the head shows a spectacular, entirely fake memory win. This asserts that
    EVERY decoder layer has at least one adapter tensor with a strictly non-zero gradient norm."""
    if getattr(model, "_lomo_opt", None) is not None:
        # LOMO applies the update INSIDE the backward and FREES each gradient as soon as it is
        # consumed -- that is the whole mechanism. So `p.grad` is None for every parameter
        # afterwards and a grad-norm gate reports 100% dead, which is a FALSE NEGATIVE.
        # Verify the thing the gate actually cares about instead: that the step CHANGES the
        # weights. Sampled over the decoder projections, which is where a detached-graph failure
        # would show up.
        # 2-D projection weights ONLY. A 1-D LayerNorm weight sits at 1.0, where bf16's ulp is
        # ~0.0078; an lr=3e-4 update rounds to NO CHANGE and the tensor reads as "dead" while
        # training perfectly well. Measured: exactly the two norm weights in a 24-tensor sample
        # failed this way before the filter. Projection weights have no such degeneracy.
        names = [n for n, p in model.named_parameters()
                 if p.requires_grad and ".layers." in n and n.endswith(".weight") and p.dim() == 2]
        watch = names[::max(1, len(names) // 24)][:24]
        before = {n: p.detach().clone() for n, p in model.named_parameters() if n in watch}
        out = model(**batch)
        model._lomo_opt.fused_backward(out.loss, model._lomo_lr)
        moved = [n for n, p in model.named_parameters()
                 if n in before and not torch.equal(p.detach(), before[n])]
        n_layers_moved = len({n.split(".layers.")[1].split(".")[0] for n in moved})
        return {"loss": float(out.loss), "n_dead_trainable_grads": len(watch) - len(moved),
                "dead_examples": [n for n in watch if n not in moved][:5],
                "n_layers_with_adapter_grads": n_layers_moved,
                "n_layers_all_adapter_grads_live": n_layers_moved,
                "expected_layers": n_layers, "n_watched": len(watch), "n_moved": len(moved),
                "criterion": "LOMO fuses+frees grads; verified by PARAMETER CHANGE, not grad norm",
                "PASS": len(moved) == len(watch) and len(watch) > 0}
    for p in model.parameters():
        p.grad = None
    out = model(**batch)
    if type(model).__name__ == "StreamModel":
        # StreamBP fuses the backward into the forward (see `step`), so the gradients this gate
        # inspects already exist and `loss.backward()` would raise. Without this branch the gate
        # throws, the receipt comes back EMPTY, and the arm looks measured while nothing verified
        # that it trains -- the precise failure this function exists to prevent.
        if out.loss.requires_grad:
            raise RuntimeError("StreamModel loss still requires grad; fused backward did not run")
    elif type(model).__module__.startswith("deepspeed"):
        model.backward(out.loss)
    else:
        out.loss.backward()
    per_layer = defaultdict(lambda: {"n": 0, "live": 0, "sum_norm": 0.0})
    dead = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nz = p.grad is not None and float(p.grad.float().norm()) > 0.0
        if not nz:
            dead.append(n)
        # Bucket EVERY trainable tensor that lives inside a decoder layer, not just the ones whose
        # name says "lora". Under any adapter family those are exactly the adapter tensors, so no
        # previously published number moves; under full fine-tuning it covers the projection and
        # norm weights too, and under `frozen` the bucket is legitimately empty.
        if True:
            m = re.search(r"\.layers\.(\d+)\.", n)
            if m:
                d = per_layer[int(m.group(1))]
                d["n"] += 1
                d["live"] += int(nz)
                d["sum_norm"] += float(p.grad.float().norm()) if p.grad is not None else 0.0
    loss_val = float(out.loss.detach())
    del out
    for p in model.parameters():
        p.grad = None
    layers_all_live = sum(1 for d in per_layer.values() if d["n"] > 0 and d["live"] == d["n"])
    layers_seen = len(per_layer)
    res = {
        "loss": loss_val,
        "n_dead_trainable_grads": len(dead),
        "dead_examples": dead[:8],
        "n_layers_with_adapter_grads": layers_seen,
        "n_layers_all_adapter_grads_live": layers_all_live,
        "expected_layers": n_layers,
        # `layers_seen == 0` is the `frozen` family: nothing inside a decoder layer is trainable,
        # so there is no per-layer liveness to assert -- the whole-model dead-gradient count is.
        "PASS": bool(len(dead) == 0 and (layers_seen == 0 or layers_all_live == n_layers)),
        "per_layer_grad_norm_sum": {str(k): round(v["sum_norm"], 6)
                                    for k, v in sorted(per_layer.items())},
    }
    gc.collect()
    return res


def make_batch(cfg, device, vocab):
    g = torch.Generator(device="cpu").manual_seed(41)
    ids = torch.randint(0, vocab, (cfg["batch"], cfg["seq"]), generator=g).to(device)
    return {"input_ids": ids, "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)}


# `--no_head`: run the decoder stack only and start backward from a FIXED synthetic gradient on
# the last hidden state, i.e. remove the LM head, the [N, V] bf16 logits and the fp32
# cross-entropy stack from the measured window. That stack is 875 MiB at seq 1024 / batch 2 --
# larger than every stored activation in the model combined -- and it is entirely
# method-independent (protocol §D.4), so at this scale it hides the quantity being compared.
# Removing it identically in every arm is the only way to see the activation term; it is reported
# ALONGSIDE the with-head number, never instead of it.
_HEADLESS = {"on": False, "grad": None}


def _inner_llama(model):
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    return getattr(base, "model", base)


def step(model, batch, opt):
    if _HEADLESS["on"]:
        inner = _inner_llama(model)
        hs = inner(input_ids=batch["input_ids"]).last_hidden_state
        g = _HEADLESS["grad"]
        if g is None or tuple(g.shape) != tuple(hs.shape) or g.dtype != hs.dtype:
            gen = torch.Generator(device="cpu").manual_seed(41)
            g = (torch.randn(tuple(hs.shape), generator=gen, dtype=torch.float32) * 1e-3
                 ).to(device=hs.device, dtype=hs.dtype)
            _HEADLESS["grad"] = g
        receipt = hs.detach().norm()
        hs.backward(g)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return receipt
    out = model(**batch)
    if type(model).__module__.startswith("deepspeed"):
        # A DeepSpeed engine owns its own optimizer and its ZeRO hooks fire from `engine.backward`
        # / `engine.step`, not from `loss.backward()` / `opt.step()`. Calling the plain pair on a
        # stage-3 engine silently skips parameter gather/partition and measures a broken arm.
        # Routing it here keeps this the ONE harness (protocol §A.0) instead of forking a second
        # one that would then need cross-validation before either could be quoted.
        model.backward(out.loss)
        model.step()
        return out.loss.detach()
    if getattr(model, "_lomo_opt", None) is not None:
        # LOMO fuses the parameter update INTO the backward: `fused_backward(loss, lr)` replaces
        # `loss.backward()` + `opt.step()` entirely (lomo/src/lomo_trainer.py:172). There is no
        # optimizer state to step and no gradient buffer to zero.
        model._lomo_opt.fused_backward(out.loss, model._lomo_lr)
        return out.loss.detach()
    if type(model).__name__ == "StreamModel":
        # StreamBP FUSES the backward into the forward -- their readme tells integrators to replace
        # `accelerator.backward` with a no-op, and their own driver guards with
        # `if output.loss.requires_grad: output.loss.backward()` (scripts/test_bp.py:60-61).
        # The guard is gated on the arm rather than applied blanket: a bare
        # `if requires_grad` everywhere would silently swallow a genuinely detached graph, which is
        # exactly the PEFT+checkpointing failure this harness exists to catch.
        if out.loss.requires_grad:
            raise RuntimeError(
                "StreamModel returned a loss that still requires grad -- the fused backward did "
                "not run. Refusing to measure; this would report a forward-only step.")
        opt.step()
        opt.zero_grad(set_to_none=True)
        return out.loss.detach()
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return out.loss


# ----------------------------------------------------------------------------------------------
# 4. Saved-tensor accounting, read off the real autograd graph
# ----------------------------------------------------------------------------------------------

SAVED_NAMES = [
    "x_o", "x_q", "x_scale", "mean_1", "rstd_1", "x_norm_1_q", "x_norm_1_scale",
    "cos", "sin", "q_main_q", "q_main_scale", "k_main_q", "k_main_scale",
    "v_main_q", "v_main_scale", "a_o", "a_threshold", "o_q", "o_scale",
    "q_lora_a", "k_lora_a", "v_lora_a", "o_final_lora_a",
    "mean_2", "rstd_2", "x_medium_o", "x_medium_q", "x_medium_scale",
    "x_norm_2_q", "x_norm_2_scale", "gate_main_q", "gate_main_scale",
    "up_main_q", "up_main_scale", "gate_lora_a", "up_lora_a", "down_lora_a",
    "norm_weight_1", "norm_bias_1",
    "w_q", "b_q", "w_q_lora_a", "w_q_lora_b",
    "w_k", "b_k", "w_k_lora_a", "w_k_lora_b",
    "w_v", "b_v", "w_v_lora_a", "w_v_lora_b",
    "w_o", "b_o", "w_o_lora_a", "w_o_lora_b",
    "norm_weight_2", "norm_bias_2",
    "w_gate", "b_gate", "w_gate_lora_a", "w_gate_lora_b",
    "w_up", "b_up", "w_up_lora_a", "w_up_lora_b",
    "w_down", "b_down", "w_down_lora_a", "w_down_lora_b",
]

# Everything after this index is a WEIGHT (a reference to an existing parameter), not a
# per-token activation buffer. Weights cost no incremental memory: they are already resident.
FIRST_WEIGHT_IDX = SAVED_NAMES.index("norm_weight_1")

# The FlashAttention variant (`layer_type='intra_inter_flash'`). Differences from the eager list:
# `a_o`/`a_threshold` are GONE (no attention map is stored at all), and q/k carry a zero point.
SAVED_NAMES_FLASH = [
    "x_o", "x_q", "x_scale", "mean_1", "rstd_1", "x_norm_1_q", "x_norm_1_scale",
    "cos", "sin",
    "q_main_q", "q_main_scale", "q_main_zero_point",
    "k_main_q", "k_main_scale", "k_main_zero_point",
    "v_main_q", "v_main_scale", "o_q", "o_scale",
    "q_lora_a", "k_lora_a", "v_lora_a", "o_final_lora_a",
    "mean_2", "rstd_2", "x_medium_o", "x_medium_q", "x_medium_scale",
    "x_norm_2_q", "x_norm_2_scale", "gate_main_q", "gate_main_scale",
    "up_main_q", "up_main_scale", "gate_lora_a", "up_lora_a", "down_lora_a",
    "norm_weight_1", "norm_bias_1",
    "w_q", "b_q", "w_q_lora_a", "w_q_lora_b",
    "w_k", "b_k", "w_k_lora_a", "w_k_lora_b",
    "w_v", "b_v", "w_v_lora_a", "w_v_lora_b",
    "w_o", "b_o", "w_o_lora_a", "w_o_lora_b",
    "norm_weight_2", "norm_bias_2",
    "w_gate", "b_gate", "w_gate_lora_a", "w_gate_lora_b",
    "w_up", "b_up", "w_up_lora_a", "w_up_lora_b",
    "w_down", "b_down", "w_down_lora_a", "w_down_lora_b",
]

# Tensors the fused Function parks on `ctx` instead of `save_for_backward`. They are just as
# resident as anything in `saved_tensors` but a walk of `saved_tensors` alone MISSES them, so the
# byte table would silently under-report. Upstream's FA layer does exactly this with `softmax_lse`.
CTX_HELD_TENSORS = ("softmax_lse",)


def _fused_node_name(arm):
    return ("FusedLlamaLayerIntraInterFlashFuncBackward"
            if str(arm).startswith("hyclora_flash") else "FusedLlamaLayerIntraInterFuncBackward")


def _saved_names_for(arm):
    return (SAVED_NAMES_FLASH if str(arm).startswith("hyclora_flash") else SAVED_NAMES)


def _tensor_bytes(t):
    if t is None:
        return 0
    if t.is_sparse:
        return (t._indices().numel() * t._indices().element_size()
                + t._values().numel() * t._values().element_size())
    return t.numel() * t.element_size()


def saved_tensor_report(loss, target_name="FusedLlamaLayerIntraInterFuncBackward",
                        names=None, first_weight_idx=None):
    """Walk the autograd graph and read the actual saved tensors of every fused layer node."""
    # NOTE: autograd graph nodes are materialised as fresh Python wrapper objects on every
    # `.next_functions` access. If we only record id()s, a wrapper can be garbage-collected and
    # its id reused by a later node, which then looks "already seen" and truncates the walk (in
    # practice it stopped after 2 nodes). `keepalive` holds a strong reference to every node so
    # ids stay unique for the duration of the traversal.
    seen, nodes, keepalive = set(), [], []
    stack = [loss.grad_fn]
    while stack:
        n = stack.pop()
        if n is None or id(n) in seen:
            continue
        seen.add(id(n))
        keepalive.append(n)
        if type(n).__name__.startswith(target_name):
            nodes.append(n)
        for nxt, _ in getattr(n, "next_functions", ()):
            stack.append(nxt)
    if not nodes:
        return None

    per_name = defaultdict(float)
    detail = {}
    node = nodes[0]
    try:
        saved = node.saved_tensors
    except Exception as exc:                                  # pragma: no cover
        return {"error": f"could not read saved_tensors: {exc}"}
    names = names if names is not None else SAVED_NAMES
    if first_weight_idx is None:
        first_weight_idx = names.index("norm_weight_1")
    for i, t in enumerate(saved):
        name = names[i] if i < len(names) else f"idx{i}"
        b = _tensor_bytes(t)
        per_name[name] += b
        kind = "weight" if i >= first_weight_idx else "activation"
        detail[name] = {
            "bytes": b, "kind": kind,
            "shape": (list(t.shape) if t is not None and not t.is_sparse
                      else (f"sparse nnz={t._nnz()}" if t is not None else None)),
            "dtype": str(t.dtype) if t is not None else None,
        }
    # ctx-parked tensors (`softmax_lse` on the FA path). For a Python autograd.Function the
    # grad_fn object IS the ctx, so attributes set in forward are readable straight off the node.
    for extra in CTX_HELD_TENSORS:
        t = getattr(node, extra, None)
        if torch.is_tensor(t):
            detail[extra] = {
                "bytes": _tensor_bytes(t), "kind": "activation",
                "shape": list(t.shape), "dtype": str(t.dtype),
                "note": "held on ctx, NOT in save_for_backward",
            }
    act = sum(v["bytes"] for v in detail.values() if v["kind"] == "activation")
    wgt = sum(v["bytes"] for v in detail.values() if v["kind"] == "weight")
    return {"n_fused_nodes": len(nodes), "per_tensor": detail,
            "activation_bytes_per_layer": act, "weight_ref_bytes_per_layer": wgt}


# ----------------------------------------------------------------------------------------------
# 4b. GENERIC saved-tensor inventory -- works for ANY arm, read off the real autograd graph
# ----------------------------------------------------------------------------------------------
#
# `saved_tensor_report` above only works for HyC-LoRA, whose whole layer is one autograd.Function
# with a known positional signature. Our own arms (`v3_recompute_sdpa`, baselines) are ordinary
# graphs of hundreds of nodes, so the inventory has to be built by walking every node.
#
# Two independent mechanisms are combined:
#   * a `saved_tensors_hooks` pack hook that records WHICH MODULE saved each tensor. The hook
#     returns the tensor object unchanged, so the graph holds exactly the objects it would
#     otherwise hold -- no numerical and no allocation effect. Only metadata is retained.
#   * a walk of the live graph after forward, reading `_saved_*` properties (C++ nodes) and
#     `saved_tensors` (Python Functions). This is what makes the inventory HONEST for sdpa:
#     the flash/mem-efficient backend's retained buffers (q/k/v/out/logsumexp/philox) are
#     invisible in source but are plain `_saved_*` entries on the graph node.
#
# Every tensor is deduplicated by STORAGE, so aliases/views are counted once, and storages that
# belong to parameters or buffers are reported separately because they cost no incremental memory.

import re


class SavedNamer:
    """Module attribution for saved tensors. Numerically and allocation-wise inert."""

    def __init__(self, model):
        self.stack, self.records, self.handles = [], {}, []
        for name, mod in model.named_modules():
            self.handles.append(mod.register_forward_pre_hook(self._mk_pre(name)))
            self.handles.append(mod.register_forward_hook(self._post))

    def _mk_pre(self, name):
        def pre(mod, inp):
            self.stack.append(name)
        return pre

    def _post(self, mod, inp, out):
        if self.stack:
            self.stack.pop()

    def pack(self, t):
        # MUST NOT raise: this hook sits on every save_for_backward in the model. HyC-LoRA saves
        # a SPARSE tensor (`a_o`), which has no storage and no data_ptr at all.
        try:
            key, _nb = _storage_key_bytes(t)
            self.records[key] = {"module": self.stack[-1] if self.stack else "<root>",
                                 "depth": len(self.stack)}
        except Exception:
            pass
        return t

    @staticmethod
    def unpack(t):
        return t

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _storage_key_bytes(t):
    """(dedup key, incremental bytes) for a tensor. Sparse tensors carry two storages and have
    neither `untyped_storage()` nor `data_ptr()`, so they are keyed by their component pointers."""
    if getattr(t, "is_sparse", False) or getattr(t, "layout", None) is not torch.strided:
        try:
            idx, val = t._indices(), t._values()
            return (("sp", idx.data_ptr(), val.data_ptr()),
                    idx.numel() * idx.element_size() + val.numel() * val.element_size())
        except Exception:
            return ("opaque", id(t)), t.numel() * t.element_size()
    try:
        st = t.untyped_storage()
        return st.data_ptr(), st.nbytes()
    except Exception:
        try:
            return t.data_ptr(), t.numel() * t.element_size()
        except Exception:
            return ("opaque", id(t)), t.numel() * t.element_size()


def _node_tensors(node):
    """Every tensor a graph node holds: `_saved_*` (C++ nodes) + `saved_tensors` (Python)."""
    out = []
    for attr in dir(node):
        if not attr.startswith("_saved_"):
            continue
        try:
            v = getattr(node, attr)
        except Exception:
            continue
        nm = attr[len("_saved_"):]
        if isinstance(v, torch.Tensor):
            out.append((nm, v))
        elif isinstance(v, (list, tuple)):
            for i, e in enumerate(v):
                if isinstance(e, torch.Tensor):
                    out.append((f"{nm}[{i}]", e))
    try:
        st = node.saved_tensors
    except Exception:
        st = None
    if st:
        for i, t in enumerate(st):
            if isinstance(t, torch.Tensor):
                out.append((f"saved[{i}]", t))
    # tensors stashed as plain ctx attributes by a Python Function (not save_for_backward)
    try:
        d = vars(node)
    except Exception:
        d = {}
    for k, v in (d or {}).items():
        if isinstance(v, torch.Tensor):
            out.append((f"ctx.{k}", v))
    return out


def walk_graph_nodes(loss):
    """All autograd nodes reachable from the loss. `keep` must stay alive during traversal:
    node wrappers are recreated on every `.next_functions` access and their id()s get reused."""
    seen, nodes, keep = set(), [], []
    stack = [loss.grad_fn]
    while stack:
        n = stack.pop()
        if n is None or id(n) in seen:
            continue
        seen.add(id(n))
        keep.append(n)
        nodes.append(n)
        for nxt, _ in getattr(n, "next_functions", ()):
            stack.append(nxt)
    return nodes, keep


def _canon_module(name):
    """model.layers.7.self_attn -> layers.*.self_attn (so 22 layers collapse into one row)."""
    if not name:
        return "<root>"
    n = re.sub(r"\.\d+(\.|$)", r".*\1", name)
    n = re.sub(r"^model\.model\.|^base_model\.model\.model\.|^model\.", "", n)
    return n or "<root>"


def node_signature(nodes, match):
    """FULL `_saved_*` signature of the first node whose type matches -- tensors AND scalars.

    This is the only way to see what a fused backend actually retains: the sdpa/flash backward
    kernel's requirements are invisible in our source and are not all tensors (rng state,
    is_causal, scale, dropout_p all live on the node too).
    """
    for node in nodes:
        if match not in type(node).__name__:
            continue
        sig = []
        for attr in sorted(dir(node)):
            if not attr.startswith("_saved_"):
                continue
            nm = attr[len("_saved_"):]
            try:
                v = getattr(node, attr)
            except Exception as exc:
                sig.append({"name": nm, "type": "<unreadable>", "value": str(exc)[:80]})
                continue
            if isinstance(v, torch.Tensor):
                try:
                    _k, nb = _storage_key_bytes(v)
                except Exception:
                    nb = v.numel() * v.element_size()
                sig.append({"name": nm, "type": "Tensor",
                            "dtype": str(v.dtype).replace("torch.", ""),
                            "shape": list(v.shape), "device": str(v.device),
                            "MiB": nb / 2 ** 20, "numel": v.numel()})
            else:
                sig.append({"name": nm, "type": type(v).__name__, "value": str(v)[:60]})
        return {"node": type(node).__name__, "saved": sig}
    return None


def generic_saved_report(loss, model, namer, n_layers):
    """Per-layer saved-tensor inventory for an arbitrary arm, read off the real graph."""
    owned = {}
    for nm, p in model.named_parameters():
        k, _ = _storage_key_bytes(p)
        owned[k] = ("param", nm)
    for nm, b in model.named_buffers():
        k, _ = _storage_key_bytes(b)
        owned.setdefault(k, ("buffer", nm))

    nodes, _keep = walk_graph_nodes(loss)
    # Collect every (node, attr, tensor) first, then dedup in a DETERMINISTIC order. A storage
    # saved by more than one node (common: an RMSNorm output feeding q/k/v) can only be charged
    # once; charging it to the alphabetically-first group makes the table reproducible, and the
    # `refs` count shows where sharing happened.
    cand = []
    for node in nodes:
        ntype = type(node).__name__.replace("Backward0", "").replace("Backward", "")
        for attr, t in _node_tensors(node):
            if t is None or t.numel() == 0:
                continue
            key, nb = _storage_key_bytes(t)
            kind, owner = owned.get(key, (None, None))
            mod = _canon_module((namer.records.get(key, {}) or {}).get("module", ""))
            gk = f"{mod}::{ntype}.{attr}"
            cand.append((gk, key, nb, kind or "activation", owner, mod, ntype, attr,
                         str(t.dtype).replace("torch.", ""),
                         (f"sparse nnz={t._nnz()}" if t.is_sparse else list(t.shape))))
    cand.sort(key=lambda c: (c[0], c[1]))

    uniq = {}
    for (gk, key, nb, kind, owner, mod, ntype, attr, dt, sh) in cand:
        if key in uniq:
            uniq[key]["refs"] += 1
            continue
        uniq[key] = {"gk": gk, "bytes": nb, "refs": 1, "kind": kind, "owner": owner,
                     "module": mod, "node": ntype, "attr": attr, "dtype": dt, "shape": sh}

    groups = defaultdict(lambda: {"bytes": 0.0, "n": 0, "big": -1, "dtype": "", "shape": None,
                                  "kind": "", "module": "", "node": "", "attr": "", "members": []})
    for rec in uniq.values():
        # kind is part of the key: a group must never mix parameter aliases with activations
        gk = f"{rec['gk']}" + ("" if rec["kind"] == "activation" else f" [{rec['kind']}]")
        g = groups[gk]
        g["bytes"] += rec["bytes"]
        g["n"] += 1
        g["kind"] = rec["kind"]
        g["module"], g["node"], g["attr"] = rec["module"], rec["node"], rec["attr"]
        if len(g["members"]) < 64:
            g["members"].append(rec["bytes"])
        if rec["bytes"] > g["big"]:                     # describe the LARGEST member
            g["big"] = rec["bytes"]
            g["dtype"], g["shape"] = rec["dtype"], rec["shape"]

    per_layer, model_level, params = [], [], []
    for gk, g in groups.items():
        mem = sorted(g["members"])
        row = {"key": gk, "n_tensors": g["n"], "bytes_total": g["bytes"],
               "MiB_total": g["bytes"] / 2 ** 20, "dtype": g["dtype"], "shape": g["shape"],
               "kind": g["kind"], "module": g["module"],
               "member_MiB_min": (mem[0] / 2 ** 20) if mem else 0.0,
               "member_MiB_max": (mem[-1] / 2 ** 20) if mem else 0.0,
               "member_MiB_median": (mem[len(mem) // 2] / 2 ** 20) if mem else 0.0}
        if g["kind"] in ("param", "buffer"):
            params.append(row)
        elif "layers.*" in g["module"]:
            row["MiB_per_layer"] = g["bytes"] / 2 ** 20 / n_layers
            per_layer.append(row)
        else:
            model_level.append(row)
    per_layer.sort(key=lambda r: -r["MiB_total"])
    model_level.sort(key=lambda r: -r["MiB_total"])
    params.sort(key=lambda r: -r["MiB_total"])

    act_layer = sum(r["bytes_total"] for r in per_layer)
    act_model = sum(r["bytes_total"] for r in model_level)
    return {
        "n_graph_nodes": len(nodes),
        "n_unique_storages": len(uniq),
        "sdpa_backward_signature": node_signature(nodes, "ScaledDotProduct"),
        "node_type_counts": dict(sorted(
            ((k, v) for k, v in
             __import__("collections").Counter(type(n).__name__ for n in nodes).items()),
            key=lambda kv: -kv[1])[:25]),
        "per_layer_rows": per_layer,
        "model_level_rows": model_level,
        "param_alias_rows": params[:20],
        "activation_bytes_per_layer": act_layer / n_layers,
        "activation_bytes_all_layers": act_layer,
        "activation_bytes_model_level": act_model,
        "param_alias_bytes": sum(r["bytes_total"] for r in params),
    }


# ----------------------------------------------------------------------------------------------
# 4c. GENERIC allocator timeline -- per aten-op net allocation, forward AND backward
# ----------------------------------------------------------------------------------------------
#
# `hyclora/prof.py`'s memtrace only covers the annotated fused layer. For an ordinary graph the
# same information comes from a TorchDispatchMode that samples `torch.cuda.memory_allocated()`
# around every aten call (the allocator counter is maintained by the issuing thread in program
# order and does not synchronise, so this is faithful and cheap). Backward ops are attributed to
# the autograd node that is executing, via `Node.register_prehook` (returns None -> inert).
#
# This runs only in an explicitly-enabled measurement pass; nothing is left installed.

_CUR_LABEL = {"v": "fwd:<pre>"}


class AllocTraceMode(object):
    """Instantiated lazily so importing this module does not require TorchDispatchMode."""

    def __new__(cls, trace, namer):
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                before = torch.cuda.memory_allocated()
                out = func(*args, **(kwargs or {}))
                after = torch.cuda.memory_allocated()
                if before != after or len(trace) < 200000:
                    trace.append((_CUR_LABEL["v"], str(func).split(".")[-2]
                                  if "." in str(func) else str(func), before, after))
                return out

        return _Mode()


def install_fwd_labels(model):
    """Forward-phase labels: innermost module currently executing."""
    handles, stack = [], []

    def mk_pre(name):
        def pre(mod, inp):
            stack.append(name)
            _CUR_LABEL["v"] = "fwd:" + _canon_module(name)
        return pre

    def post(mod, inp, out):
        if stack:
            stack.pop()
        _CUR_LABEL["v"] = "fwd:" + (_canon_module(stack[-1]) if stack else "<root>")

    for name, mod in model.named_modules():
        handles.append(mod.register_forward_pre_hook(mk_pre(name)))
        handles.append(mod.register_forward_hook(post))
    return handles


def install_bwd_labels(loss, namer):
    """Backward-phase labels: the autograd node currently executing, plus its module."""
    nodes, keep = walk_graph_nodes(loss)
    handles = []
    for node in nodes:
        ntype = type(node).__name__.replace("Backward0", "").replace("Backward", "")
        mod = ""
        for _attr, t in _node_tensors(node):
            if t is None:
                continue
            key, _nb = _storage_key_bytes(t)
            rec = namer.records.get(key)
            if rec:
                mod = rec["module"]
                break
        label = f"bwd:{_canon_module(mod)}::{ntype}"

        def mk(lbl):
            def hook(grads):
                _CUR_LABEL["v"] = lbl
                return None
            return hook
        try:
            handles.append(node.register_prehook(mk(label)))
        except Exception:
            pass
    return handles, keep


def analyse_alloc_trace(trace, n_layers):
    """§9-format: net MiB retained per op-label, and where the high-water mark lands."""
    if not trace:
        return None
    hi = max(range(len(trace)), key=lambda i: trace[i][3])
    net = defaultdict(float)
    for lbl, op, b0, b1 in trace:
        net[f"{lbl}|{op}"] += (b1 - b0)
    rows = sorted(net.items(), key=lambda kv: -kv[1])
    # Only per-layer labels may be divided by the layer count; a model-level op (lm_head, CE)
    # happens once per step and dividing it by 22 would be a lie.
    return {
        "n_ops_traced": len(trace),
        "peak_label": trace[hi][0],
        "peak_op": trace[hi][1],
        "peak_MiB": trace[hi][3] / 2 ** 20,
        "peak_index": hi,
        "net_alloc_MiB_per_layer": {k: (v / 2 ** 20 / n_layers if "layers.*" in k else v / 2 ** 20)
                                    for k, v in rows[:30]
                                    if abs(v) / 2 ** 20 / (n_layers if "layers.*" in k else 1) > 0.05},
        "net_alloc_is_per_layer": {k: ("layers.*" in k) for k, v in rows[:30]},
        "net_alloc_MiB_total": {k: v / 2 ** 20 for k, v in rows[:30]},
        "context_at_peak": [{"label": t[0], "op": t[1], "after_MiB": t[3] / 2 ** 20,
                             "delta_MiB": (t[3] - t[2]) / 2 ** 20}
                            for t in trace[max(0, hi - 12):hi + 6]],
    }


# ----------------------------------------------------------------------------------------------
# 4d. What is LIVE at the high-water mark -- replayed from the CUDA allocator's own history
# ----------------------------------------------------------------------------------------------

_FRAME_SKIP = ("/torch/", "profile_hyclora.py", "_python_dispatch", "<built-in>", "_ops.py",
               "_tensor.py", "overrides.py")


def _frame_label(frames):
    """Innermost frame that is real user/library code, not torch plumbing or this profiler."""
    if not frames:
        return "<no frames>"
    for f in frames:
        fn = f.get("filename", "")
        if not any(s in fn for s in _FRAME_SKIP):
            return f"{os.path.basename(fn)}:{f.get('line','')}:{f.get('name','')}"
    f = frames[0]
    return f"{os.path.basename(f.get('filename',''))}:{f.get('line','')}:{f.get('name','')}"


def peak_composition(device_index=0, baseline_bytes=0):
    """Replay the recorded allocator history: exact peak, and the live-block composition there.

    `baseline_bytes` is what was already allocated when recording began (parameters, optimizer
    state, the input batch). The recorder only sees allocations made after it was switched on, so
    without this offset the replayed peak is relative, not the absolute high-water mark.
    """
    try:
        snap = torch.cuda.memory._snapshot()
    except Exception as exc:
        return {"error": str(exc)}
    traces = snap.get("device_traces") or []
    if device_index >= len(traces):
        return {"error": "no device trace"}
    tr = traces[device_index]

    cur = peak = 0
    peak_i = -1
    for i, ev in enumerate(tr):
        a = ev.get("action")
        if a == "alloc":
            cur += ev.get("size", 0)
            if cur > peak:
                peak, peak_i = cur, i
        elif a in ("free_completed", "free_requested"):
            if a == "free_completed":
                cur -= ev.get("size", 0)
    live = {}
    cur = 0
    for i, ev in enumerate(tr[:peak_i + 1]):
        a = ev.get("action")
        if a == "alloc":
            live[ev.get("addr")] = ev
            cur += ev.get("size", 0)
        elif a == "free_completed":
            e = live.pop(ev.get("addr"), None)
            if e is not None:
                cur -= e.get("size", 0)
    by = defaultdict(lambda: {"bytes": 0, "n": 0})
    for ev in live.values():
        k = _frame_label(ev.get("frames"))
        by[k]["bytes"] += ev.get("size", 0)
        by[k]["n"] += 1
    rows = sorted(by.items(), key=lambda kv: -kv[1]["bytes"])[:40]
    return {
        "peak_MiB_absolute": (peak + baseline_bytes) / 2 ** 20,
        "peak_MiB_above_baseline": peak / 2 ** 20,
        "baseline_MiB": baseline_bytes / 2 ** 20,
        "n_live_blocks_at_peak": len(live),
        "allocating_frame_at_peak": _frame_label(tr[peak_i].get("frames")) if peak_i >= 0 else None,
        "alloc_size_at_peak_MiB": tr[peak_i].get("size", 0) / 2 ** 20 if peak_i >= 0 else 0,
        "live_by_frame": [{"frame": k, "MiB": v["bytes"] / 2 ** 20, "n_blocks": v["n"]}
                          for k, v in rows],
    }


# ----------------------------------------------------------------------------------------------
# 5. Profiling driver
# ----------------------------------------------------------------------------------------------

def dev_time(evt):
    for attr in ("device_time_total", "cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def self_dev_time(evt):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def measure_peak_variant(arm, cfg, device, *, adapter_dtype="bf16", use_cache=False,
                         warm_steps=8, fresh_opt=False, accum=1, empty_cache=True,
                         label=""):
    """One peak-memory number under ONE explicit definition, with the harness conditions that
    differ between `profile_hyclora.py` and `train_glue.py` exposed as toggles.

    `train_glue.py`'s reported figure is `max_memory_allocated` over the whole FIRST optimizer
    step: gradient_accumulation_steps micro-batches with grads live across them, the optimizer
    state materialised inside the window, and no prior warm-up. This harness's figure is
    `max_memory_allocated` over ONE warm fwd+bwd+step with the optimizer already materialised.
    """
    torch.manual_seed(41)
    model = build_model(arm, cfg, device, adapter_dtype=adapter_dtype, use_cache=use_cache)
    vocab = hf_config(model).vocab_size
    batch = make_batch(cfg, device, vocab)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(warm_steps):
        step(model, batch, opt)
    torch.cuda.synchronize()
    if fresh_opt:
        del opt
        gc.collect()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    if empty_cache:
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for i in range(accum):
        out = model(**batch)
        (out.loss / accum).backward()
        del out
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    res = {"label": label or f"{arm}|{adapter_dtype}|warm{warm_steps}|accum{accum}"
                             f"|fresh_opt={fresh_opt}|use_cache={use_cache}",
           "arm": arm, "adapter_dtype": adapter_dtype, "warm_steps": warm_steps,
           "accum": accum, "fresh_opt": fresh_opt, "use_cache": use_cache,
           "baseline_resident_MiB": base / 2 ** 20, "peak_MiB": peak / 2 ** 20}
    del model, opt, batch
    gc.collect()
    torch.cuda.empty_cache()
    return res


def _gpu_state():
    """Clock / temperature / power receipt, per fair_comparison_protocol.md A.3.5."""
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return [r.strip() for r in q.stdout.strip().splitlines()]
    except Exception as exc:                                       # pragma: no cover
        return [f"error: {exc}"]


def measure_headline(arm, cfg, device, args):
    """THE headline number, measured exactly as `llmdocs/trackers/fair_comparison_protocol.md`
    section A requires.

      * peak = `max_memory_allocated` over a full steady-state training step (fwd + bwd +
        optimizer.step + zero_grad), never the whole-run maximum (eval-clamped, §A.1) and never
        the first step (optimizer-state-biased);
      * warm-up clears Triton autotune, HyC-LoRA's `iteration_threshold` calibration window and
        at least one optimizer step, so Adam moments already exist;
      * ONE `empty_cache()`, in the quiesce block only, so `max_memory_reserved` is comparable;
      * >= 3 measured steps, max AND median reported, allocated AND reserved reported;
      * the adapter dtype is READ OFF THE LIVE PARAMETERS, not trusted from a flag (§E.1: an
        unmatched adapter dtype is worth ~2150 MiB and it has already bitten this project once).
    """
    torch.manual_seed(41)
    model = build_model(arm, cfg, device, adapter_dtype=args.adapter_dtype, use_cache=False)
    # Every downstream `arm.startswith/endswith` test must see the BASE arm name; the row keeps
    # the full `<arm>+<family>` string so a sweep's JSON distinguishes the families.
    arm_full = arm
    arm, family = arm_family(arm)
    vocab = hf_config(model).vocab_size
    batch = make_batch(cfg, device, vocab)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if getattr(model, "_galore_groups", None) is not None:
        # GaLore replaces the OPTIMIZER, not the model, so it is constructed here rather than in
        # `build_model`. Their param-group split (2-D projected weights vs everything else) and
        # their published hyper-parameters travel on the model from `build_galore_model`.
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "galore"))
        from galore_torch import GaLoreAdamW
        opt = GaLoreAdamW(model._galore_groups, lr=3e-4)
    else:
        opt = torch.optim.AdamW(trainable, lr=3e-4)

    # ---- dtype receipt, from the live parameters ----
    dt = defaultdict(lambda: [0, 0])
    for n, p in model.named_parameters():
        key = ("adapter" if is_adapter_param(n) else "base") + "/" + str(p.dtype)
        dt[key][0] += 1
        dt[key][1] += p.numel() * p.element_size()
    dtype_receipt = {k: {"n_tensors": v[0], "MiB": round(v[1] / 2 ** 20, 2)}
                     for k, v in sorted(dt.items())}
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    opt_mem = sum(p.numel() * p.element_size() for p in trainable) * 2 / 2 ** 20
    adapter_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters()
                             if is_adapter_param(n)})

    # ---- warm-up (protocol A.2) ----
    warm = max(3, cfg["iteration_threshold"] + 3)
    for _ in range(warm):
        step(model, batch, opt)
    torch.cuda.synchronize()

    # ---- quiesce; the ONLY legitimate empty_cache() in the measurement ----
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resident_before = torch.cuda.memory_allocated()

    gpu_before = _gpu_state()
    peaks_alloc, peaks_resv, losses = [], [], []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        loss = step(model, batch, opt)
        torch.cuda.synchronize()
        peaks_alloc.append(torch.cuda.max_memory_allocated())
        peaks_resv.append(torch.cuda.max_memory_reserved())
        losses.append(float(loss))

    # ---- steady-state timing: individually timed steps, median + IQR ----
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(model, batch, opt)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    gpu_after = _gpu_state()

    # ---- gradient liveness (protocol D.8, generalised) + the checkpointing/PEFT-detachment
    #      trap: every trainable grad non-zero AND every decoder layer's adapters live.
    liveness = adapter_grad_liveness(model, batch, cfg["n_layers"])
    dead = liveness["dead_examples"]
    loss_val = liveness["loss"]
    opt.zero_grad(set_to_none=True)

    # ---- per-layer RETAINED bytes, from the allocator, works on EVERY arm ----
    retained = per_layer_retained_report(model, batch)
    opt.zero_grad(set_to_none=True)

    # ---- per-layer stored bytes, read off the REAL autograd graph ----
    stored = None
    if arm.startswith("hyclora"):
        out = model(**batch)
        stored = saved_tensor_report(out.loss, target_name=_fused_node_name(arm),
                                     names=_saved_names_for(arm))
        out.loss.backward()
        opt.zero_grad(set_to_none=True)
        del out
    generic = None
    # A non-reentrant checkpointed graph must NOT be walked: reading `_saved_*` fires the unpack
    # hook, which RE-RUNS the checkpointed region. The result would be the uncompressed byte
    # count plus a large memory perturbation -- i.e. exactly the wrong answer, silently.
    if getattr(args, "inventory", True) and not is_nonreentrant_checkpointed(arm):
        namer = SavedNamer(model)
        with torch.autograd.graph.saved_tensors_hooks(namer.pack, namer.unpack):
            out = model(**batch)
        try:
            generic = generic_saved_report(out.loss, model, namer, cfg["n_layers"])
        finally:
            namer.close()
        out.loss.backward()
        opt.zero_grad(set_to_none=True)
        del out
    torch.cuda.synchronize()

    counters = None
    fb_policy = None
    if arm.startswith("hyclora"):
        from hyclora.patch import get_counters
        counters = get_counters()
    elif arm.startswith("fb_"):
        from flashffn import fb_get_counters, fb_policy_report
        counters = fb_get_counters()
        # Fail loud rather than silently reporting baseline numbers under a fused-block name
        # (the honesty-counter convention v3 and the HyC-LoRA port both use).
        if counters["forward"] == 0 or counters["backward"] == 0:
            raise RuntimeError(f"{arm}: fused block never executed ({counters})")
        fb_policy = fb_policy_report()
        # An `attn`/`glu`/`full` row that re-ran the O(S^2) FlashAttention forward, or a `min` row
        # that did not, is self-refuting -- assert rather than report it.
        _lv = fb_policy["forwards_by_level"]
        if _lv.get("min", 0) > 0 and counters["flash_recompute"] == 0:
            raise RuntimeError(f"{arm}: keep='min' forwards ran but flash_recompute==0 ({counters})")
        if _lv.get("min", 0) == 0 and counters["flash_recompute"] > 0:
            raise RuntimeError(f"{arm}: no keep='min' forward, yet the O(S^2) attention forward "
                               f"was recomputed {counters['flash_recompute']}x ({fb_policy})")

    res = {
        "arm": arm_full,
        "family": family,
        "attn_implementation": ("sdpa" if (arm.endswith("_sdpa")
                                           or arm.startswith("hyclora_flash")) else "eager"),
        "seq": cfg["seq"], "batch": cfg["batch"],
        "adapter_dtype_requested": args.adapter_dtype,
        "adapter_dtype_measured": adapter_dtypes,
        "dtype_receipt": dtype_receipt,
        "param_mem_mib": round(param_mem, 2), "opt_mem_mib": round(opt_mem, 2),
        "warmup_steps": warm,
        "train_step_peak_alloc_mib": max(peaks_alloc) / 2 ** 20,
        "train_step_peak_alloc_median_mib": sorted(peaks_alloc)[1] / 2 ** 20,
        "train_step_peak_reserved_mib": max(peaks_resv) / 2 ** 20,
        "resident_floor_mib": resident_before / 2 ** 20,
        "peak_minus_floor_mib": (max(peaks_alloc) - resident_before) / 2 ** 20,
        "all_peaks_mib": [p / 2 ** 20 for p in peaks_alloc],
        "ms_per_step_median": 1e3 * times[len(times) // 2],
        "ms_per_step_iqr": 1e3 * (times[int(0.75 * len(times))] - times[int(0.25 * len(times))]),
        "ms_per_step_min": 1e3 * times[0],
        "n_timed_steps": len(times),
        "loss": loss_val, "losses_during_measured_steps": losses,
        "loss_finite": bool(math.isfinite(loss_val)),
        "n_dead_gradients": liveness["n_dead_trainable_grads"], "dead_gradients": dead,
        "grad_liveness": liveness,
        "retained": retained,
        "checkpointing": getattr(model, "_ckpt_receipt", None),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
        "gpu_state_before": gpu_before, "gpu_state_after": gpu_after,
        "hyclora_counters": counters,
        "fb_policy": fb_policy,
        "headless": bool(_HEADLESS["on"]),
    }
    if checkpoint_spec(arm)[0] == "sac_dots":
        res["sac_policy_ops"] = dict(sorted(SAC_STATS.items(), key=lambda kv: -kv[1])[:40])
        res["sac_n_saved_ops"] = sum(v for k, v in SAC_STATS.items() if k.startswith("SAVE "))
        res["sac_n_recomputed_ops"] = sum(v for k, v in SAC_STATS.items()
                                          if k.startswith("RECOMP "))
    if stored:
        acts = {k: v for k, v in stored["per_tensor"].items() if v["kind"] == "activation"}
        res["stored_per_layer"] = {
            "activation_bytes_per_layer": stored["activation_bytes_per_layer"],
            "activation_MiB_per_layer": stored["activation_bytes_per_layer"] / 2 ** 20,
            "activation_MiB_all_layers": (stored["activation_bytes_per_layer"]
                                          * cfg["n_layers"] / 2 ** 20),
            "n_fused_nodes": stored["n_fused_nodes"],
            "per_tensor": {k: {"MiB": v["bytes"] / 2 ** 20, "dtype": v["dtype"],
                               "shape": v["shape"], **({"note": v["note"]} if "note" in v else {})}
                           for k, v in sorted(acts.items(), key=lambda kv: -kv[1]["bytes"])},
        }
    if generic:
        res["saved_graph"] = generic

    del model, opt, batch
    gc.collect()
    torch.cuda.empty_cache()
    return res


def profile_arm(arm, cfg, device, args):
    from hyclora import prof as hprof

    torch.manual_seed(41)
    model = build_model(arm, cfg, device, adapter_dtype=getattr(args, "adapter_dtype", "bf16"))
    vocab = hf_config(model).vocab_size
    batch = make_batch(cfg, device, vocab)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    res = {"arm": arm}

    # ---- warmup: past triton autotune AND past the calibration window ----
    warm = max(cfg["iteration_threshold"] + 3, 8)
    for _ in range(warm):
        step(model, batch, opt)
    torch.cuda.synchronize()

    # ---- measure the actual nnz of the sparsified attention map (needed by the byte model) ----
    if arm.startswith("hyclora"):
        out = model(**batch)
        rep = saved_tensor_report(out.loss)
        if rep and "per_tensor" in rep:
            a_o = rep["per_tensor"].get("a_o", {})
            sh = a_o.get("shape") or ""
            if isinstance(sh, str) and "nnz=" in sh:
                cfg["attn_nnz"] = int(sh.split("nnz=")[1])
            res["saved_tensors"] = rep
        out.loss.backward()
        opt.zero_grad(set_to_none=True)
        del out
        torch.cuda.synchronize()

    # ---- GENERIC saved-tensor inventory, read off the real graph (every arm) ----
    # ...except a non-reentrant checkpointed one, where reading `_saved_*` fires the recompute.
    if getattr(args, "inventory", True) and not is_nonreentrant_checkpointed(arm):
        namer = SavedNamer(model)
        with torch.autograd.graph.saved_tensors_hooks(namer.pack, namer.unpack):
            out = model(**batch)
        try:
            res["saved_graph"] = generic_saved_report(out.loss, model, namer, cfg["n_layers"])
        finally:
            namer.close()
        out.loss.backward()
        opt.zero_grad(set_to_none=True)
        del out
        gc.collect()
        torch.cuda.synchronize()

    # ---- clean wall-clock pass (NO profiler, NO record_function) ----
    hprof.enable(False)
    gc.collect()
    torch.cuda.synchronize()
    if getattr(args, "mem_only", False):
        wall = float("nan")
    else:
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step(model, batch, opt)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / args.steps
    res["wall_s_per_step_clean"] = wall

    # ---- peak-memory timeline ----
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = model(**batch)
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    out.loss.backward()
    torch.cuda.synchronize()
    after_bwd = torch.cuda.memory_allocated()
    peak_bwd = torch.cuda.max_memory_allocated()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak_step = torch.cuda.max_memory_allocated()
    res["memory"] = {
        "baseline_resident_MiB": base / 2 ** 20,
        "after_fwd_MiB": after_fwd / 2 ** 20,
        "live_activations_after_fwd_MiB": (after_fwd - base) / 2 ** 20,
        "peak_during_fwd_MiB": peak_fwd / 2 ** 20,
        "peak_during_bwd_MiB": peak_bwd / 2 ** 20,
        "peak_overall_MiB": peak_step / 2 ** 20,
        "after_bwd_MiB": after_bwd / 2 ** 20,
        "fwd_headroom_over_live_MiB": (peak_fwd - after_fwd) / 2 ** 20,
        "reserved_MiB": torch.cuda.memory_reserved() / 2 ** 20,
    }
    del out
    gc.collect()
    torch.cuda.empty_cache()

    # ---- per-op allocator timeline: which single op sets the high-water mark? ----
    if arm.startswith("hyclora"):
        hprof.enable_memtrace(True)
        out = model(**batch)
        out.loss.backward()
        opt.zero_grad(set_to_none=True)
        tr = hprof.get_memtrace()
        hprof.enable_memtrace(False)
        del out
        if tr:
            # peak point over the whole trace, and net allocation per op summed over layers
            hi_i = max(range(len(tr)), key=lambda i: tr[i][2])
            net = defaultdict(float)
            for nm, b0, b1 in tr:
                net[nm] += (b1 - b0)
            n_layers = max(1, sum(1 for nm, _, _ in tr if nm == "F01.rmsnorm_1"))
            res["memory_trace"] = {
                "peak_op": tr[hi_i][0],
                "peak_MiB_at_that_op": tr[hi_i][2] / 2 ** 20,
                "n_layer_invocations_in_trace": n_layers,
                "net_alloc_MiB_per_layer": {k: v / 2 ** 20 / n_layers
                                            for k, v in sorted(net.items(), key=lambda kv: -kv[1])
                                            if abs(v) / 2 ** 20 / n_layers > 0.05},
                "timeline_first_layer": [
                    {"op": nm, "after_MiB": b1 / 2 ** 20, "delta_MiB": (b1 - b0) / 2 ** 20}
                    for nm, b0, b1 in tr[:70]],
            }
        gc.collect()
        torch.cuda.empty_cache()

    # ---- GENERIC per-aten-op allocator timeline + high-water composition (every arm) ----
    if getattr(args, "timeline", True):
        # ---- pass 1: allocator history on a CLEAN step (no dispatch mode, so the recorded
        #      python frames are the real call sites -- transformers / flashffn / hyclora) ----
        try:
            torch.cuda.memory._record_memory_history(context="all", stacks="python",
                                                     max_entries=400000)
            base_bytes = torch.cuda.memory_allocated()
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            res["peak_composition"] = peak_composition(device.index or 0, base_bytes)
            torch.cuda.memory._record_memory_history(enabled=None)
            del out
        except Exception as exc:                                   # pragma: no cover
            res["peak_composition"] = {"error": str(exc)}
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

        # ---- pass 2: per-aten-op allocator timeline via TorchDispatchMode ----
        # `install_bwd_labels` walks the graph and reads node tensors, which on a non-reentrant
        # checkpointed graph triggers the recompute. Pass 1 above (allocator history replay) is
        # safe on every arm because it never touches the graph; pass 2 is not.
    if getattr(args, "timeline", True) and is_nonreentrant_checkpointed(arm):
        res["alloc_timeline"] = {"skipped": "non-reentrant checkpointing: walking the graph "
                                            "would trigger the recompute"}
    elif getattr(args, "timeline", True):
        trace = []
        namer2 = SavedNamer(model)
        fwd_h = install_fwd_labels(model)
        mode = AllocTraceMode(trace, namer2)
        with mode:
            with torch.autograd.graph.saved_tensors_hooks(namer2.pack, namer2.unpack):
                out = model(**batch)
        for h in fwd_h:
            h.remove()
        bwd_h, _keep = install_bwd_labels(out.loss, namer2)
        _CUR_LABEL["v"] = "bwd:<start>"
        with mode:
            out.loss.backward()
        _CUR_LABEL["v"] = "opt:step"
        with mode:
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        for h in bwd_h:
            h.remove()
        namer2.close()
        del out, _keep
        res["alloc_timeline"] = analyse_alloc_trace(trace, cfg["n_layers"])
        if res["alloc_timeline"] is not None:
            res["alloc_timeline"]["timeline_head"] = [
                {"label": t[0], "op": t[1], "after_MiB": t[3] / 2 ** 20,
                 "delta_MiB": (t[3] - t[2]) / 2 ** 20}
                for t in trace[:400] if abs(t[3] - t[2]) > 0.25 * 2 ** 20]
        del trace
        gc.collect()
        torch.cuda.empty_cache()

    # ---- profiler pass ----
    if getattr(args, "mem_only", False):
        res["total_device_us_per_step"] = float("nan")
        res["gpu_busy_fraction"] = float("nan")
        res["n_kernel_launches_per_step"] = float("nan")
        del model, opt, batch
        gc.collect()
        torch.cuda.empty_cache()
        return res
    if arm.startswith("hyclora"):
        hprof.enable(True)
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False) as p:
        for _ in range(args.prof_steps):
            step(model, batch, opt)
        torch.cuda.synchronize()
    hprof.enable(False)

    evts = p.key_averages()
    nsteps = args.prof_steps

    # Regions: a record_function emits TWO rows -- a CPU row, and a device-side annotation
    # ("gpu_user_annotation") whose self time is the GPU-timeline span covered by the region.
    #
    # We use the DEVICE annotation as the primary measure. The CPU row's `device_time_total`
    # was tried first and is not trustworthy here: it returns 0 for many regions (any region
    # whose kernels are launched from inside a nested aten op that the linker does not walk
    # back to the annotation), which silently zeroes real work. The device annotation is
    # measured on the GPU timeline itself and has no such gap. Because the regions are strictly
    # sequential on one stream, the span is kernel time plus any intra-region idle, so summing
    # spans slightly exceeds pure kernel time -- reported and cross-checked below.
    regions, cpu_side = {}, {}
    for e in evts:
        if not ("." in e.key and e.key[0] in "FB" and e.key[1:3].isdigit()):
            continue
        if str(getattr(e, "device_type", "")) == "DeviceType.CUDA":
            regions.setdefault(e.key, {})["device_us_per_step"] = self_dev_time(e) / nsteps
            regions[e.key]["count_per_step"] = e.count / nsteps
        else:
            cpu_side[e.key] = {"cpu_us_per_step": e.cpu_time_total / nsteps,
                               "linked_device_us_per_step": dev_time(e) / nsteps}
    for k, v in regions.items():
        v.update(cpu_side.get(k, {}))
    res["regions_raw"] = regions

    # kernels
    # Real kernels only: DeviceType.CUDA and NOT a user annotation. Two sources of double-count
    # are excluded here -- the CPU op that launched the kernel (which also reports the kernel's
    # time), and our own record_function regions (which emit a device-side annotation row).
    kernels = []
    total_dev = 0.0
    for e in evts:
        if str(getattr(e, "device_type", "")) != "DeviceType.CUDA":
            continue
        if getattr(e, "is_user_annotation", False):
            continue
        sdt = self_dev_time(e)
        if sdt <= 0:
            continue
        total_dev += sdt
        kernels.append({"name": e.key, "self_device_us_per_step": sdt / nsteps,
                        "count_per_step": e.count / nsteps})
    kernels.sort(key=lambda k: -k["self_device_us_per_step"])
    res["kernels_top"] = kernels[:45]
    res["total_device_us_per_step"] = total_dev / nsteps
    res["gpu_busy_fraction"] = (total_dev / nsteps / 1e6) / wall
    res["n_kernel_launches_per_step"] = sum(k["count_per_step"] for k in kernels)

    del model, opt, batch
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------------------------
# 6. Roofline join + reporting
# ----------------------------------------------------------------------------------------------

CATEGORY = {
    # the GEMMs a stock LoRA forward/backward would also have to do -- irreducible math
    "F04.lora_fwd_q": "proj_gemm", "F05.lora_fwd_k": "proj_gemm", "F06.lora_fwd_v": "proj_gemm",
    "F21.lora_fwd_o": "proj_gemm", "F27.lora_fwd_gate": "proj_gemm", "F28.lora_fwd_up": "proj_gemm",
    "F33.lora_fwd_down": "proj_gemm",
    "B07.lora_bwd_down": "proj_gemm", "B11.lora_bwd_gate": "proj_gemm", "B12.lora_bwd_up": "proj_gemm",
    "B17.lora_bwd_o": "proj_gemm", "B31.lora_bwd_q": "proj_gemm", "B32.lora_bwd_k": "proj_gemm",
    "B33.lora_bwd_v": "proj_gemm",
    # attention score/context GEMMs
    "F14.qk_gemm": "attn_gemm", "F19.av_gemm": "attn_gemm",
    "B21.av_gemm_bwd": "attn_gemm", "B25.qk_gemm_bwd_k": "attn_gemm", "B27.qk_gemm_bwd_q": "attn_gemm",
    # O(S^2) elementwise traffic over the materialised attention map
    "F15.add_mask": "attn_map_ew", "F16.softmax": "attn_map_ew",
    "F17.compress_attn_map": "attn_map_ew", "B19.attn_map_to_dense": "attn_map_ew",
    "B22.softmax_bwd": "attn_map_ew", "B23.scale_grad_s": "attn_map_ew",
    # HyC-LoRA's compression overhead
    "F03.compress_x_chan": "quantize", "F07.compress_x_norm_1": "quantize",
    "F08.compress_q_main": "quantize", "F09.compress_k_main": "quantize",
    "F10.compress_v_main": "quantize", "F22.compress_o": "quantize",
    "F26.compress_x_medium_chan": "quantize", "F29.compress_x_norm_2": "quantize",
    "F30.compress_gate_main": "quantize", "F31.compress_up_main": "quantize",
    "B01.dequant_gate_main": "dequantize", "B04.dequant_up_main": "dequantize",
    "B10.dequant_x_norm_2": "dequantize", "B13.dequant_x_medium": "dequantize",
    "B16.dequant_o": "dequantize", "B20.dequant_v_rebuild": "dequantize",
    "B24.dequant_q_rebuild": "dequantize", "B26.dequant_k_rebuild": "dequantize",
    "B30.dequant_x_norm_1": "dequantize", "B34.dequant_x": "dequantize",
    # rebuilding gate/up/hadamard in backward instead of storing them
    "B02.readd_gate_lora": "recompute", "B03.silu_recompute": "recompute",
    "B05.readd_up_lora": "recompute", "B06.hadamard_recompute": "recompute",
    # norms
    "F01.rmsnorm_1": "norm", "F24.rmsnorm_2": "norm",
    "B14.rmsnorm_bwd_2": "norm", "B35.rmsnorm_bwd_1": "norm",
    # rope
    "F12.rope_qk": "rope", "B28.rope_bwd": "rope",
    # plain elementwise on [N,H] or [N,I]
    "F23.residual_1": "elementwise", "F32.silu_hadamard": "elementwise",
    "F34.residual_2": "elementwise", "B08.hadamard_bwd": "elementwise",
    "B09.silu_bwd": "elementwise", "B15.residual_bwd_2": "elementwise",
    "B36.residual_bwd_1": "elementwise",
    # pure data movement forced by layout/GQA
    "F02.clone_x": "layout_copy", "F11.reshape_qkv": "layout_copy", "F13.repeat_kv_k": "layout_copy",
    "F18.repeat_kv_v": "layout_copy", "F20.reshape_o": "layout_copy", "F25.clone_x_medium": "layout_copy",
    "B18.reshape_grad_o": "layout_copy", "B29.reshape_grad_qkv": "layout_copy",
}


def classify(ai, ach_tflops, ach_gbs, peaks, dev_us, launch_us, n_launch):
    frac_mem = ach_gbs / peaks["hbm_peak_gbs"]
    frac_cmp = ach_tflops / peaks["gemm_bf16_tflops"]
    launch_frac = (launch_us * n_launch) / dev_us if dev_us > 0 else 0.0
    if frac_mem > 1.15:
        verdict = "CACHE/MODEL-CHECK"
    elif launch_frac > 0.5:
        verdict = "LAUNCH-BOUND"
    elif frac_cmp >= frac_mem and frac_cmp > 0.25:
        verdict = "COMPUTE-BOUND"
    elif frac_mem > 0.55:
        verdict = "MEMORY-BOUND"
    elif frac_mem > 0.25:
        verdict = "MEMORY-BOUND(partial)"
    else:
        verdict = "LATENCY/OCCUPANCY-BOUND"
    return verdict, frac_mem, frac_cmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hyclora_q4")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--prof_steps", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id. Every shape field in the cfg is derived from its config, "
                         "so a second family needs no other change here.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"],
                    help="bf16 = this harness's historical behaviour (whole model cast, which is "
                         "also what HyC-LoRA's patch does to adapters); fp32 = PEFT's default, "
                         "which is what train_glue.py actually runs for non-HyC-LoRA arms.")
    ap.add_argument("--mem_only", action="store_true",
                    help="skip the clean timing pass and the profiler pass (memory numbers only)")
    ap.add_argument("--no_inventory", dest="inventory", action="store_false", default=True)
    ap.add_argument("--no_timeline", dest="timeline", action="store_false", default=True)
    ap.add_argument("--peak_variants", action="store_true",
                    help="peak-memory matrix isolating every harness difference between this "
                         "script and train_glue.py (resolves the 5015-vs-6160 divergence)")
    ap.add_argument("--skip_roofline", action="store_true")
    ap.add_argument("--headline", action="store_true",
                    help="protocol-compliant head-to-head: training-step peak (max+median, "
                         "allocated+reserved), steady-state ms/step, per-layer stored bytes, "
                         "adapter-dtype receipt and gradient-liveness assert, for every --arm. "
                         "This is the mode that produces a quotable number.")
    ap.add_argument("--no_head", action="store_true",
                    help="measure the decoder stack only: skip the LM head / logits / "
                         "cross-entropy (875 MiB at seq1024/batch2, method-independent) and start "
                         "backward from a fixed synthetic grad on the last hidden state. Applied "
                         "identically to every arm; reported alongside the with-head number.")
    ap.add_argument("--interleave", type=int, default=1,
                    help="--headline only: repeat the arm sequence N times (A,B,A,B,...) so "
                         "thermal drift cannot align with the arm variable (protocol A.3.4)")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _HEADLESS["on"] = bool(getattr(args, "no_head", False))

    cfg = make_cfg(args.batch, args.seq, model=args.model)
    # replaced by measurement; the head count must come from the cfg, not from a literal 32
    cfg["attn_nnz"] = int(0.05 * args.batch * cfg["n_heads"] * args.seq * args.seq)

    if args.skip_roofline:
        peaks = {"gemm_bf16_tflops": float("nan"), "hbm_peak_gbs": float("nan"),
                 "cpu_dispatch_us_per_op": float("nan"), "gpu_min_kernel_us": float("nan"),
                 "ridge_point_flops_per_byte": float("nan")}
    else:
        print("=== measuring machine peaks ===", flush=True)
        peaks = measure_peaks(device)
        for k, v in peaks.items():
            print(f"  {k}: {v:,.2f}")

    # `frontier` = the boring-baseline frontier: everything a reviewer would benchmark us against
    # before reading our method section, plus the competitor and our own best arm, in ONE harness
    # with ONE peak definition on ONE GPU. See llmdocs/trackers/boring_baseline_frontier.md.
    FRONTIER = ["baseline_sdpa", "gc_hf_sdpa", "gc_full_sdpa", "gc_manual_sdpa",
                "sac_dots_sdpa", "gc_attn_sdpa", "gc_ffn_sdpa",
                "v3_recompute_sdpa", "v3_recompute_gc_full_sdpa", "v3_recompute_gc_hf_sdpa",
                "hyclora_q4", "hyclora_q2", "baseline_eager"]
    if args.arm == "all":
        arms = ["hyclora_q4", "hyclora_q2", "baseline_eager", "v3_recompute", "v3_recompute_sdpa"]
    elif args.arm == "frontier":
        arms = list(FRONTIER)
    else:
        arms = [a for a in args.arm.split(",") if a]

    allres = {"peaks": peaks, "cfg": cfg, "adapter_dtype": args.adapter_dtype, "arms": {}}

    # ---- protocol-compliant head-to-head (fair_comparison_protocol.md §A) ----
    if args.headline:
        rows, runs = {}, []
        for rep in range(args.interleave):
            for arm in arms:
                print(f"\n=== headline [{arm}] rep {rep + 1}/{args.interleave} ===", flush=True)
                try:
                    r = measure_headline(arm, cfg, device, args)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    r = {"arm": arm, "error": str(exc)}
                r["rep"] = rep
                runs.append(r)
                if "error" not in r:
                    gl = r.get("grad_liveness", {})
                    rt = r.get("retained", {}) or {}
                    print(f"    peak(alloc) {r['train_step_peak_alloc_mib']:.2f} MiB | "
                          f"reserved {r['train_step_peak_reserved_mib']:.2f} | "
                          f"floor {r['resident_floor_mib']:.2f} | "
                          f"{r['ms_per_step_median']:.1f} ms/step | "
                          f"retained/layer {rt.get('retained_MiB_per_layer_median', float('nan')):.2f} MiB | "
                          f"adapters {r['adapter_dtype_measured']} | "
                          f"grad-liveness {'PASS' if gl.get('PASS') else 'FAIL'} "
                          f"({gl.get('n_layers_all_adapter_grads_live')}/{gl.get('expected_layers')} "
                          f"layers, {r['n_dead_gradients']} dead)", flush=True)
                # keep the rep with the LOWEST median step time (least co-tenant interference)
                if arm not in rows or (
                        "error" not in r
                        and r.get("ms_per_step_median", 1e9) < rows[arm].get("ms_per_step_median", 1e9)):
                    rows[arm] = r
        allres["headline"] = rows
        allres["headline_all_reps"] = runs
        out = args.out or os.path.join(RESULTS_DIR, f"headline_seq{cfg['seq']}.json")
        with open(out, "w") as f:
            json.dump(allres, f, indent=2, default=str)
        print(f"\n-> {out}")
        hdr = (f"{'arm':<26} {'attn':<6} {'ckpt':<10} {'peak alloc':>11} {'peak resv':>10} "
               f"{'floor':>9} {'peak-floor':>11} {'ms/step':>9} {'ret MiB/lyr':>12} "
               f"{'saved MiB/lyr':>14} {'grads':>6}")
        print(f"\n{'=' * len(hdr)}\nHEADLINE  seq {cfg['seq']} batch {cfg['batch']}  "
              f"adapters={args.adapter_dtype}  alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}"
              f"\n{'=' * len(hdr)}")
        print(hdr)
        for arm in arms:
            r = rows.get(arm, {})
            if "error" in r:
                print(f"{arm:<26} ERROR {r['error'][:70]}")
                continue
            st = r.get("stored_per_layer", {}).get("activation_MiB_per_layer")
            if st is None:
                sg = r.get("saved_graph") or {}
                if sg.get("activation_bytes_per_layer"):
                    st = sg["activation_bytes_per_layer"] / 2 ** 20
            rt = (r.get("retained") or {}).get("retained_MiB_per_layer_median")
            ck = (r.get("checkpointing") or {}).get("variant", "-")
            gl = r.get("grad_liveness", {})
            print(f"{arm:<26} {r['attn_implementation']:<6} {ck:<10} "
                  f"{r['train_step_peak_alloc_mib']:>11.2f} "
                  f"{r['train_step_peak_reserved_mib']:>10.2f} "
                  f"{r['resident_floor_mib']:>9.2f} "
                  f"{r['peak_minus_floor_mib']:>11.2f} "
                  f"{r['ms_per_step_median']:>9.2f} "
                  f"{(f'{rt:.3f}' if rt is not None else '-'):>12} "
                  f"{(f'{st:.3f}' if st else '-'):>14} "
                  f"{('PASS' if gl.get('PASS') else 'FAIL'):>6}")

        # ---- Pareto frontier over (training-step peak, ms/step) ----
        pts = [(a, rows[a]["train_step_peak_alloc_mib"], rows[a]["ms_per_step_median"])
               for a in arms if "error" not in rows.get(a, {"error": 1})]
        front = []
        for a, m, t in sorted(pts, key=lambda p: (p[1], p[2])):
            if not front or t < front[-1][2] - 1e-9:
                front.append((a, m, t))
        print(f"\nPARETO FRONTIER (min peak, min ms/step) -- a new method must land "
              f"BELOW-AND-LEFT of every one of these:")
        for a, m, t in front:
            print(f"    {a:<28}{m:>10.1f} MiB{t:>10.2f} ms/step")
        return

    # ---- harness-difference matrix: ONE arm, one peak definition, one toggle at a time ----
    if args.peak_variants:
        variants = []
        for arm in arms:
            base_kw = dict(warm_steps=8, accum=1, fresh_opt=False, use_cache=False)
            grid = [
                ("A profile-harness default (bf16 adapters, warm, accum=1)",
                 dict(adapter_dtype="bf16")),
                ("B PEFT fp32 adapters (what train_glue.py actually runs)",
                 dict(adapter_dtype="fp32")),
                ("C fp32 adapters + grads live across 2 accumulation micro-batches",
                 dict(adapter_dtype="fp32", accum=2)),
                ("D fp32 adapters + optimizer state created inside the window",
                 dict(adapter_dtype="fp32", fresh_opt=True)),
                ("E fp32 adapters + COLD first step (no warm-up at all)",
                 dict(adapter_dtype="fp32", warm_steps=0, fresh_opt=True)),
                ("F fp32 adapters + use_cache left at the HF default",
                 dict(adapter_dtype="fp32", use_cache=True)),
                ("G bf16 adapters + cold + fresh optimizer",
                 dict(adapter_dtype="bf16", warm_steps=0, fresh_opt=True)),
                ("H fp32 + cold + fresh opt + accum=8 (closest emulation of train_glue)",
                 dict(adapter_dtype="fp32", warm_steps=0, fresh_opt=True, accum=8)),
                ("I bf16 + cold + fresh opt + accum=8 (same, but bf16 adapters)",
                 dict(adapter_dtype="bf16", warm_steps=0, fresh_opt=True, accum=8)),
            ]
            for lab, kw in grid:
                kws = dict(base_kw)
                kws.update(kw)
                print(f"\n=== peak variant [{arm}] {lab} ===", flush=True)
                try:
                    v = measure_peak_variant(arm, cfg, device, label=f"{arm} | {lab}", **kws)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    v = {"label": f"{arm} | {lab}", "error": str(exc)}
                variants.append(v)
                print(f"    peak {v.get('peak_MiB', float('nan')):.2f} MiB "
                      f"(resident {v.get('baseline_resident_MiB', float('nan')):.2f})", flush=True)
        allres["peak_variants"] = variants
        out = args.out or os.path.join(RESULTS_DIR, "peak_variants.json")
        with open(out, "w") as f:
            json.dump(allres, f, indent=2, default=str)
        print(f"\n-> {out}")
        print(f"\n{'='*100}\nPEAK-MEMORY VARIANT MATRIX (seq {cfg['seq']}, batch {cfg['batch']})\n{'='*100}")
        for v in variants:
            print(f"  {v['label']:<78} {v.get('peak_MiB', float('nan')):>10.2f} MiB")
        return
    for arm in arms:
        print(f"\n=== profiling {arm} ===", flush=True)
        try:
            r = profile_arm(arm, cfg, device, args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            allres["arms"][arm] = {"error": str(exc)}
            continue
        allres["arms"][arm] = r
        print(f"  wall {r['wall_s_per_step_clean']*1e3:.1f} ms/step | "
              f"gpu busy {r['gpu_busy_fraction']*100:.1f}% | "
              f"launches {r['n_kernel_launches_per_step']:.0f}/step | "
              f"peak {r['memory']['peak_overall_MiB']:.0f} MiB")

    allres["cfg"]["attn_nnz_measured"] = cfg["attn_nnz"]

    # ---- join measured region times against the analytic roofline model ----
    model = build_op_model(cfg)
    for arm, r in allres["arms"].items():
        if "regions_raw" not in r:
            continue
        rows, unmapped = [], []
        for name, meas in sorted(r["regions_raw"].items()):
            cnt = meas["count_per_step"] or 1.0
            us_inv = meas["device_us_per_step"] / cnt          # per layer-invocation
            spec = model.get(name)
            if spec is None:
                unmapped.append(name)
                continue
            secs = us_inv / 1e6
            fl, by, gfl = spec["flops"], spec["bytes"], spec["gemm_flops"]
            ach_tf = (fl / secs / 1e12) if secs > 0 else 0.0
            ach_gb = (by / secs / 1e9) if secs > 0 else 0.0
            ai = (fl / by) if by > 0 else float("inf")
            # kernels launched inside this region, approximated by count of aten ops -> use 1 as
            # a floor; the launch-bound test uses the measured dispatch cost against region time.
            verdict, fm, fc = classify(ai, ach_tf, ach_gb, peaks, us_inv,
                                       peaks["gpu_min_kernel_us"], 1)
            rows.append({
                "op": name, "category": CATEGORY.get(name, "other"),
                "us_per_layer_inv": us_inv,
                "us_per_step_all_layers": meas["device_us_per_step"],
                "invocations_per_step": cnt,
                "GFLOP": fl / 1e9, "MiB_moved": by / 2 ** 20,
                "arith_intensity": ai, "achieved_TFLOPs": ach_tf, "achieved_GBs": ach_gb,
                "pct_hbm_peak": 100 * fm, "pct_gemm_peak": 100 * fc, "verdict": verdict,
                "gemm_fraction_of_flops": (gfl / fl) if fl > 0 else 0.0,
            })
        rows.sort(key=lambda x: -x["us_per_step_all_layers"])
        r["ops"] = rows
        r["unmapped_regions"] = unmapped
        r["region_device_us_per_step_total"] = sum(x["us_per_step_all_layers"] for x in rows)

        cat = defaultdict(lambda: {"us": 0.0, "GFLOP": 0.0, "MiB": 0.0, "n": 0})
        for x in rows:
            c = cat[x["category"]]
            c["us"] += x["us_per_step_all_layers"]
            c["GFLOP"] += x["GFLOP"] * x["invocations_per_step"]
            c["MiB"] += x["MiB_moved"] * x["invocations_per_step"]
            c["n"] += 1
        tot = sum(c["us"] for c in cat.values()) or 1.0
        for c in cat.values():
            c["pct_of_layer_time"] = 100 * c["us"] / tot
        r["by_category"] = dict(cat)

    out = args.out or os.path.join(RESULTS_DIR, "profile_raw.json")
    with open(out, "w") as f:
        json.dump(allres, f, indent=2, default=str)
    print(f"\n-> {out}")

    # ---- console summary ----
    for arm, r in allres["arms"].items():
        if not r.get("ops"):
            continue
        print(f"\n{'='*110}\n{arm}: per-op, steady state, per layer-invocation "
              f"(x{r['ops'][0]['invocations_per_step']:.0f} layers/step)\n{'='*110}")
        print(f"{'op':<28}{'cat':<14}{'us':>8}{'%lyr':>7}{'GFLOP':>8}{'MiB':>9}"
              f"{'AI':>8}{'TF/s':>8}{'GB/s':>8}{'%HBM':>7}  verdict")
        tot = r["region_device_us_per_step_total"]
        for x in r["ops"]:
            print(f"{x['op']:<28}{x['category']:<14}{x['us_per_layer_inv']:>8.1f}"
                  f"{100*x['us_per_step_all_layers']/tot:>7.2f}{x['GFLOP']:>8.2f}"
                  f"{x['MiB_moved']:>9.1f}{x['arith_intensity']:>8.1f}"
                  f"{x['achieved_TFLOPs']:>8.1f}{x['achieved_GBs']:>8.0f}"
                  f"{x['pct_hbm_peak']:>7.0f}  {x['verdict']}")
        print(f"\n{arm}: by category")
        for c, v in sorted(r["by_category"].items(), key=lambda kv: -kv[1]["us"]):
            print(f"  {c:<16} {v['us']/1e3:>8.2f} ms/step  {v['pct_of_layer_time']:>6.2f}%  "
                  f"{v['GFLOP']:>9.1f} GFLOP  {v['MiB']/1024:>8.2f} GiB moved")

    # ---- time budget + memory ----
    print(f"\n{'='*110}\nTIME BUDGET AND MEMORY\n{'='*110}")
    for arm, r in allres["arms"].items():
        if "wall_s_per_step_clean" not in r:
            continue
        wall_us = r["wall_s_per_step_clean"] * 1e6
        dev = r["total_device_us_per_step"]
        reg = r.get("region_device_us_per_step_total", 0.0)
        n_l = r["n_kernel_launches_per_step"]
        print(f"\n{arm}")
        print(f"  wall clock          {wall_us/1e3:>9.1f} ms/step")
        print(f"  GPU kernel time     {dev/1e3:>9.1f} ms/step  ({100*dev/wall_us:.1f}% of wall)")
        print(f"  GPU idle (gaps)     {(wall_us-dev)/1e3:>9.1f} ms/step  ({100*(wall_us-dev)/wall_us:.1f}% of wall)")
        if reg:
            print(f"  ...inside fused layers {reg/1e3:>6.1f} ms/step  ({100*reg/dev:.1f}% of GPU time)")
        print(f"  kernel launches     {n_l:>9.0f} /step")
        print(f"  CPU dispatch cost   {n_l*peaks['cpu_dispatch_us_per_op']/1e3:>9.1f} ms/step "
              f"(= launches x {peaks['cpu_dispatch_us_per_op']:.2f} us; "
              f"{100*n_l*peaks['cpu_dispatch_us_per_op']/wall_us:.1f}% of wall)")
        m = r["memory"]
        print(f"  peak overall        {m['peak_overall_MiB']:>9.0f} MiB   "
              f"(fwd {m['peak_during_fwd_MiB']:.0f} / bwd {m['peak_during_bwd_MiB']:.0f})")
        print(f"  live after fwd      {m['after_fwd_MiB']:>9.0f} MiB   "
              f"(of which activations {m['live_activations_after_fwd_MiB']:.0f})")
        st = r.get("saved_tensors")
        if st and "per_tensor" in st:
            print(f"  saved per layer: activations {st['activation_bytes_per_layer']/2**20:.2f} MiB"
                  f"  (x{cfg['n_layers']} = {st['activation_bytes_per_layer']*cfg['n_layers']/2**20:.0f} MiB)"
                  f"  + weight refs {st['weight_ref_bytes_per_layer']/2**20:.2f} MiB (no incremental cost)")
            acts = [(v["bytes"], k, v) for k, v in st["per_tensor"].items() if v["kind"] == "activation"]
            acts.sort(reverse=True)
            print(f"    {'buffer':<20}{'MiB':>9}{'%':>7}  dtype / shape")
            tota = st["activation_bytes_per_layer"] or 1
            for b, k, v in acts:
                if b == 0:
                    continue
                print(f"    {k:<20}{b/2**20:>9.3f}{100*b/tota:>7.1f}  {v['dtype']} {v['shape']}")
            stored_all = st["activation_bytes_per_layer"] * cfg["n_layers"] / 2 ** 20
            print(f"  STORED vs TRANSIENT: saved-for-backward {stored_all:.0f} MiB "
                  f"({100*stored_all/m['peak_overall_MiB']:.0f}% of peak); "
                  f"the other {m['peak_overall_MiB']-stored_all:.0f} MiB is model/optimizer "
                  f"state plus the transient working set")
        mt = r.get("memory_trace")
        if mt:
            print(f"  allocator high-water inside a layer: at op '{mt['peak_op']}' "
                  f"({mt['peak_MiB_at_that_op']:.0f} MiB live)")
            print("    net allocation retained per layer (MiB):")
            for k, v in list(mt["net_alloc_MiB_per_layer"].items())[:10]:
                print(f"      {k:<28}{v:>8.2f}")

        # ---- generic graph inventory (every arm) ----
        sg = r.get("saved_graph")
        if sg:
            pl = sg["activation_bytes_per_layer"] / 2 ** 20
            print(f"\n  --- saved-tensor inventory from the real autograd graph "
                  f"({sg['n_graph_nodes']} nodes, {sg['n_unique_storages']} unique storages) ---")
            print(f"  per-layer activations {pl:.3f} MiB  x{cfg['n_layers']} = "
                  f"{sg['activation_bytes_all_layers']/2**20:.0f} MiB; "
                  f"model-level (embed/lm_head/CE) {sg['activation_bytes_model_level']/2**20:.0f} MiB; "
                  f"parameter/buffer aliases {sg['param_alias_bytes']/2**20:.0f} MiB (no incremental cost)")
            print(f"    {'buffer':<62}{'MiB/lyr':>9}{'%':>7}  dtype  shape")
            for row in sg["per_layer_rows"][:28]:
                if row["MiB_per_layer"] < 0.001:
                    continue
                spread = ("" if row["member_MiB_max"] - row["member_MiB_min"] < 0.01
                          else f"  [min {row['member_MiB_min']:.2f} / med "
                               f"{row['member_MiB_median']:.2f} / max {row['member_MiB_max']:.2f}]")
                print(f"    {row['key'][:60]:<62}{row['MiB_per_layer']:>9.3f}"
                      f"{100*row['bytes_total']/max(1.0, sg['activation_bytes_all_layers']):>7.1f}"
                      f"  {row['dtype']:<9}{row['shape']}{spread}")
            sig = sg.get("sdpa_backward_signature")
            if sig:
                print(f"    -- what the sdpa backward node ({sig['node']}) actually retains --")
                for e in sig["saved"]:
                    if e["type"] == "Tensor":
                        print(f"      {e['name']:<22}{e['MiB']:>9.3f} MiB  {e['dtype']:<9}"
                              f"{e['shape']} on {e['device']}")
                    else:
                        print(f"      {e['name']:<22}{'':>13}  {e['type']} = {e['value']}")
            print(f"    {'-- model-level --':<62}")
            for row in sg["model_level_rows"][:10]:
                if row["MiB_total"] < 0.5:
                    continue
                print(f"    {row['key'][:60]:<62}{row['MiB_total']:>9.3f}{'':>7}"
                      f"  {row['dtype']:<9}{row['shape']}")

        at = r.get("alloc_timeline")
        if at:
            print(f"\n  --- allocator timeline ({at['n_ops_traced']} aten ops traced) ---")
            print(f"  high-water at: {at['peak_label']} | {at['peak_op']}  -> {at['peak_MiB']:.0f} MiB live")
            print("    net MiB retained per layer ('(step)' rows are whole-step, not per layer):")
            for k, v in list(at["net_alloc_MiB_per_layer"].items())[:18]:
                tag = "" if at["net_alloc_is_per_layer"].get(k, False) else "  (step)"
                print(f"      {k[:60]:<62}{v:>8.2f}{tag}")
        pc = r.get("peak_composition")
        if pc and "error" not in pc:
            print(f"\n  --- what is LIVE at the high-water mark (allocator history replay) ---")
            print(f"  peak {pc['peak_MiB_absolute']:.0f} MiB = baseline {pc['baseline_MiB']:.0f} "
                  f"+ {pc['peak_MiB_above_baseline']:.0f} over {pc['n_live_blocks_at_peak']} live "
                  f"blocks; the allocation that set it: {pc['alloc_size_at_peak_MiB']:.1f} MiB at "
                  f"{pc['allocating_frame_at_peak']}")
            for row in pc["live_by_frame"][:14]:
                print(f"      {row['frame'][:66]:<68}{row['MiB']:>9.1f} MiB  ({row['n_blocks']} blocks)")


if __name__ == "__main__":
    main()
