"""unsloth / torch.compile / Liger arms, measured under `fair_comparison_protocol.md` section A.

WHY THIS IS A SEPARATE FILE FROM `profile_hyclora.py`
----------------------------------------------------
`import unsloth` monkey-patches `transformers`, `peft`, `trl`, `torch.utils.checkpoint` and
`transformers.modeling_utils.checkpoint` GLOBALLY, at import time, and it must run BEFORE
`transformers` is imported.  Putting that import inside `profile_hyclora.py` would silently change
every other arm in that harness.  So the unsloth arms live here, in their own process, and the
measurement code is imported from `profile_hyclora.py` so that the recipe is byte-identical:
same `step()`, same `make_batch()`, same warm-up, same peak definition, same liveness gate.

ARMS
----
  unsloth_nogc      FastLanguageModel + get_peft_model(use_gradient_checkpointing=False)
  unsloth_gc        ... use_gradient_checkpointing=True     (standard HF/torch checkpointing)
  unsloth_offload   ... use_gradient_checkpointing="unsloth" (their advocated best: the block
                        input is saved to PINNED CPU, so GPU-resident stored state per block ~ 0)
  compile_default   stock HF+PEFT LoRA, sdpa, torch.compile(model)
  compile_maxauto   ... torch.compile(model, mode="max-autotune")
  liger_sdpa        stock HF+PEFT LoRA, sdpa, apply_liger_kernel_to_llama() (all op kernels)
  liger_gc_sdpa     ... plus decoder-layer gradient checkpointing

FAIRNESS CONTROLS THIS FILE IMPLEMENTS
--------------------------------------
1. `--flce` applies Liger `FusedLinearCrossEntropy` to a NON-unsloth arm.  This matters because
   **unsloth already runs a fused linear cross-entropy by default** (`unsloth/models/llama.py`
   L1483-1495: `unsloth_fused_ce_loss`, taken whenever `labels is not None` and
   `UNSLOTH_RETURN_LOGITS != "1"`).  The LM-head/CE stack is 875 MiB at seq1024/batch2 -- 26.9% of
   our peak -- so an unsloth-vs-us peak comparison WITHOUT this control is measuring the CE
   implementation, not the activation cache.  `--force_logits` does the opposite (forces unsloth
   to materialise logits) purely as a diagnostic of how large that term is inside their stack.
2. `--memcpy` profiles one step and sums H2D / D2H bytes from the kineto trace.  Required for
   `unsloth_offload`: a GPU-peak win bought with PCIe traffic is a different trade, and the
   protocol requires it be disclosed.
3. Engagement receipts.  Every fused call site is wrapped with a counter AFTER unsloth has done
   its patching, so the result row proves their kernels ran rather than that the import succeeded.

Usage:
  PYTHONPATH=temp/unsloth_pkgs python src/profile_unsloth.py --arm unsloth_offload --headline
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ARMS_NEEDING_UNSLOTH = ("unsloth_nogc", "unsloth_gc", "unsloth_offload")

# ---------------------------------------------------------------------------------------------
# unsloth must be imported before transformers.  Do it here, guarded on the requested arm, so a
# torch.compile / Liger / fb_* run in this same file is NOT contaminated by unsloth's patches.
# ---------------------------------------------------------------------------------------------
_WANT_UNSLOTH = any(a in " ".join(sys.argv) for a in _ARMS_NEEDING_UNSLOTH)
_WANT_LIGER = ("liger" in " ".join(sys.argv)) or ("--flce" in sys.argv)

if _WANT_LIGER:
    _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "temp", "liger_pkgs")
    if os.path.isdir(_lp) and _lp not in sys.path:
        sys.path.insert(0, _lp)

UNSLOTH_VERSION = None
if _WANT_UNSLOTH:
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    import unsloth  # noqa: F401  (must precede transformers)
    from unsloth import FastLanguageModel
    import unsloth.kernels.fast_lora as _fast_lora
    try:
        UNSLOTH_VERSION = unsloth.__version__
    except AttributeError:
        from importlib.metadata import version as _v
        UNSLOTH_VERSION = _v("unsloth")

import torch  # noqa: E402

import profile_hyclora as ph  # noqa: E402
from profile_hyclora import (  # noqa: E402
    make_batch, step, adapter_grad_liveness, per_layer_retained_report, _gpu_state,
    _HEADLESS, _inner_llama, RESULTS_DIR,
)

FUSED_CALLS = defaultdict(int)


# ---------------------------------------------------------------------------------------------
# Engagement receipts -- prove the competitor's kernels are in the graph
# ---------------------------------------------------------------------------------------------
def _instrument_unsloth(model):
    """Wrap every fused call site AFTER unsloth patched it.  If a site was never patched the
    wrapper is still installed but the underlying callable is stock, and `_receipt` records that.
    """
    FUSED_CALLS.clear()
    inner = model.model.model  # PeftModel -> FastLlamaModel(LlamaForCausalLM) -> LlamaModel
    receipt = {"n_layers": len(inner.layers), "mlp_fused": 0, "qkv_fused": 0, "o_fused": 0}

    def _count(fn, key):
        def wrapped(*a, **kw):
            FUSED_CALLS[key] += 1
            return fn(*a, **kw)
        wrapped.__wrapped_name__ = getattr(fn, "__name__", str(fn))
        return wrapped

    for layer in inner.layers:
        mlp_fwd = layer.mlp.forward
        name = getattr(mlp_fwd, "__func__", mlp_fwd).__name__
        if "apply_lora_mlp" in name:
            receipt["mlp_fused"] += 1
        layer.mlp.forward = _count(mlp_fwd, f"mlp:{name}")

        qkv = getattr(layer.self_attn, "apply_qkv", None)
        if qkv is not None and "apply_lora_qkv" in getattr(qkv, "__name__", ""):
            receipt["qkv_fused"] += 1
            layer.self_attn.apply_qkv = _count(qkv, f"qkv:{qkv.__name__}")
        o = getattr(layer.self_attn, "apply_o", None)
        if o is not None and "apply_lora_o" in getattr(o, "__name__", ""):
            receipt["o_fused"] += 1
            layer.self_attn.apply_o = _count(o, f"o:{o.__name__}")

    ckpt = torch.utils.checkpoint.checkpoint
    receipt["torch_checkpoint_impl"] = getattr(ckpt, "__name__", str(ckpt))
    receipt["torch_checkpoint_module"] = getattr(ckpt, "__module__", "?")
    receipt["unsloth_gc_flag"] = getattr(model, "_unsloth_gradient_checkpointing", "?")
    try:
        import unsloth_zoo.gradient_checkpointing as gcmod
        receipt["zoo_classes"] = [c for c in dir(gcmod) if "Checkpoint" in c]
    except Exception as exc:  # pragma: no cover
        receipt["zoo_classes"] = f"err:{exc}"
    return receipt


def _graph_nodes(loss, limit=20000):
    """Node names in the autograd graph -- the direct proof of which Functions are in the graph.

    `keep` exists because grad_fn accessors are freshly-constructed Python wrappers: if the only
    reference is dropped the object is freed and CPython REUSES its id, so an `id()`-keyed visited
    set silently prunes most of the graph.  Holding a reference to every visited node fixes it.
    """
    seen, keep, out, stack = set(), [], defaultdict(int), [loss.grad_fn]
    while stack and len(seen) < limit:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keep.append(fn)
        out[type(fn).__name__] += 1
        for nxt, _ in getattr(fn, "next_functions", ()):
            stack.append(nxt)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------------------------
def _apply_flce(model_id=None):
    """Liger FusedLinearCrossEntropy ONLY -- every other Liger kernel left off, so this control
    changes exactly one thing (the LM-head/CE stack) and nothing about the decoder blocks.

    THE PATCHER IS PER-ARCHITECTURE, and getting that wrong is silent.  Until 2026-08-03 this
    called `apply_liger_kernel_to_llama` unconditionally.  On the first Qwen2 run the flag
    therefore did **nothing**: the row still reported `flce: true`, but the traceback showed
    `transformers/models/qwen2/modeling_qwen2.py:843 -> ForCausalLMLoss -> logits.float()`, i.e.
    stock fp32-logits cross-entropy.  On Qwen2's 151936-token vocabulary that is a **1.16 GiB**
    allocation, so the arm was measured in regime A while being labelled regime B, and the CE stack
    swamped the decoder difference the comparison exists to measure.  Resolve the patcher from the
    model's own config and raise when Liger has none, rather than mislabel a regime.

    THE `torch.compile` FIX (`holes_closed.md` HOLE 2).  Without the `torch._dynamo.disable`
    below, `torch.compile` composed with Liger FLCE dies with a reproducible
    `CUDA error: an illegal memory access was encountered`.  Mechanism, read off the traceback:
    Dynamo traces INTO the body of `LigerFusedLinearCrossEntropyFunction.forward`, hits the
    `target_mask.sum().item()` at `ops/fused_linear_cross_entropy.py:56`, breaks the graph, and
    then compiles the RESUMED fragment -- Liger's Python chunk loop, which slices `_input` and
    writes in place into `grad_input`/`grad_weight`.  Inductor emits `triton_poi_fused_4` for
    that fragment and the crash lands in its autotuner (`benchmark_all_configs`).

    Disabling Dynamo on the FLCE entry point puts exactly ONE graph break at the loss and runs
    Liger's kernel eagerly -- which is what every non-compiled FLCE arm already does.  It is
    therefore not a detune of the compiled arm: the LM head + CE stack is byte-identical across
    all FLCE arms, and the only thing `compile_gc` still compiles is the decoder stack, which is
    the region the comparison is about.  On a non-compiled arm the wrapper is a no-op.
    """
    import liger_kernel.transformers.monkey_patch as _lk
    from transformers import AutoConfig

    mt = "llama"
    if model_id is not None:
        mt = getattr(AutoConfig.from_pretrained(model_id), "model_type", "llama")
    patcher = getattr(_lk, f"apply_liger_kernel_to_{mt}", None)
    if patcher is None:
        raise NotImplementedError(
            f"--flce requested but liger_kernel has no `apply_liger_kernel_to_{mt}`, so regime B "
            f"cannot be constructed for model_type={mt!r}. Report the missing regime; do NOT fall "
            f"back to the llama patcher, which is a silent no-op here.")
    patcher(rope=False, swiglu=False, rms_norm=False, cross_entropy=False,
            fused_linear_cross_entropy=True)
    import liger_kernel.transformers.functional as _lf
    fn = _lf.liger_fused_linear_cross_entropy
    if not getattr(fn, "_dynamo_disabled_here", False):
        wrapped = torch._dynamo.disable(fn)
        wrapped._dynamo_disabled_here = True
        _lf.liger_fused_linear_cross_entropy = wrapped
    return f"liger.fused_linear_cross_entropy+dynamo_disable@{mt}"


def build_unsloth(arm, cfg, device, args):
    gc_mode = {"unsloth_nogc": False, "unsloth_gc": True, "unsloth_offload": "unsloth"}[arm]
    model, _tok = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=cfg["seq"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        device_map={"": 0},
        # Unsloth's `mapper.py` silently rewrites `TinyLlama/TinyLlama-1.1B-Chat-v1.0` to their
        # own mirror `unsloth/tinyllama-chat`.  Different weights = a different arm.  Every other
        # arm in this project loads the original repo, so pin it.
        use_exact_model_name=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"], lora_alpha=cfg["lora_r"], lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=gc_mode,
        random_state=41, use_rslora=False, loftq_config=None,
        max_seq_length=cfg["seq"],
    )
    # Unsloth leaves LoRA in fp32 (48.12 MiB of adapters + 96.24 MiB of Adam moments).  Every
    # other arm in this project runs bf16 adapters, and protocol section E.1 makes an unmatched
    # adapter dtype a hard error in a head-to-head.  `--unsloth_bf16_adapters` produces the
    # MATCHED arm; the default (fp32) is unsloth's own configuration and is reported too.
    if getattr(args, "unsloth_bf16_adapters", False):
        for _n, _p in model.named_parameters():
            if "lora_" in _n and _p.dtype == torch.float32:
                _p.data = _p.data.to(torch.bfloat16)
    model.config.use_cache = False
    model.train()
    receipt = _instrument_unsloth(model)
    receipt["requested_gc"] = str(gc_mode)
    receipt["unsloth_version"] = UNSLOTH_VERSION
    receipt["attn_impl"] = getattr(model.config, "_attn_implementation", "?")
    receipt["fused_ce_default"] = os.environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1"
    model._engagement = receipt
    return model


def build_other(arm, cfg, device, args):
    """Everything that is NOT unsloth: stock HF + PEFT, optionally Liger-patched / compiled."""
    liger_receipt = None
    if arm.startswith("streambp") and getattr(args, "flce", False):
        raise ValueError(
            "arm 'streambp' must not be run with --flce: their `logits_chunk_size` IS their chunked "
            "LM head (readme:104). Re-run without --flce; the head and loss stay inside the measured "
            "region (protocol B line 14) via their implementation instead of Liger's.")
    if arm.startswith("minis") and getattr(args, "flce", False):
        # Mini-Sequence ships its own chunked LM head + loss (`minis/mini_sequence.py:69,131`).
        # Stacking Liger's FLCE on top is two implementations of one optimisation; the row would
        # measure neither. Refuse rather than silently produce it.
        raise ValueError(
            "arm 'minis' must not be run with --flce: their `_LM_head`/`LMheadWarpper` IS their "
            "fused chunked cross-entropy. Re-run without --flce; the head and loss are still "
            "inside the measured region (protocol B line 14), via their kernel instead of Liger's.")
    if arm.startswith("liger"):
        # The patcher is per-architecture: `apply_liger_kernel_to_llama` does nothing for a Qwen2
        # model, and calling it there would produce a silently UNPATCHED arm that still reports
        # `liger` in its receipt.  Resolve it from the model's own config, and raise if Liger has
        # no patcher for this family rather than measuring a stock model under a Liger label.
        # imported from `monkey_patch` rather than the package root: the root re-exports lazily and
        # `getattr(pkg, name, None)` there returns None for patchers that do exist.
        import liger_kernel.transformers.monkey_patch as _lk
        from transformers import AutoConfig

        mt = getattr(AutoConfig.from_pretrained(cfg["model"]), "model_type", "llama")
        fn = getattr(_lk, f"apply_liger_kernel_to_{mt}", None)
        if fn is None:
            raise NotImplementedError(
                f"liger_kernel has no `apply_liger_kernel_to_{mt}`; a Liger arm cannot be "
                f"constructed for {cfg['model']}. Report the missing row, do not substitute the "
                f"llama patcher.")
        fn(rope=True, swiglu=True, rms_norm=True, cross_entropy=False,
           fused_linear_cross_entropy=True)
        liger_receipt = f"liger.full(rope,swiglu,rms_norm,fused_linear_ce)@{mt}"
    elif getattr(args, "flce", False):
        liger_receipt = _apply_flce(cfg["model"])

    base_arm = {
        "compile_default": "baseline_sdpa", "compile_maxauto": "baseline_sdpa",
        "compile_maxauto_nocg": "baseline_sdpa", "compile_gc": "gc_manual_sdpa",
        "compile_gc_maxauto_nocg": "gc_manual_sdpa", "compile_gc_blocks": "gc_manual_sdpa",
        "liger_sdpa": "baseline_sdpa", "liger_gc_sdpa": "gc_manual_sdpa",
    }.get(arm, arm)

    model = ph.build_model(base_arm, cfg, device,
                           adapter_dtype=args.adapter_dtype, use_cache=False)
    receipt = {"liger": liger_receipt, "base_arm": base_arm}
    if arm.startswith("compile"):
        # `max-autotune` turns on CUDA graphs, whose private memory pool is NOT counted by
        # `max_memory_allocated` -- that arm reports an allocated peak far below its true
        # footprint (measured: 2298 MiB allocated against 6206 MiB reserved).  The
        # `_nocg` variant is the one whose allocated peak is comparable with every other row.
        mode = {"compile_maxauto": "max-autotune",
                "compile_maxauto_nocg": "max-autotune-no-cudagraphs",
                "compile_gc_maxauto_nocg": "max-autotune-no-cudagraphs"}.get(arm)
        torch._dynamo.config.cache_size_limit = 64
        if arm == "compile_gc_blocks":
            # torchtitan-style: compile each decoder block INSIDE its checkpoint wrapper, leaving
            # the embedding, the final norm, the LM head and the loss outside every compiled
            # region.  A second, structurally different way to get a matched-CE number for
            # `torch.compile o checkpointing`, in case the whole-model compile is leaving memory
            # on the table.
            n = 0
            for _n, layer in ph._decoder_layers(model):
                layer.forward = torch.compile(layer.forward, dynamic=False)
                n += 1
            receipt["compile_mode"] = f"per-block default (n={n})"
        else:
            model = torch.compile(model, mode=mode, dynamic=False)
            receipt["compile_mode"] = mode or "default"
    model._engagement = receipt
    return model


def build(arm, cfg, device, args):
    if arm in _ARMS_NEEDING_UNSLOTH:
        return build_unsloth(arm, cfg, device, args)
    if getattr(args, "flce", False) and not arm.startswith("liger"):
        _apply_flce(cfg["model"])
    return build_other(arm, cfg, device, args)


# ---------------------------------------------------------------------------------------------
# H2D / D2H traffic (required for the offloaded arm)
# ---------------------------------------------------------------------------------------------
def measure_memcpy(model, batch, opt, tmpdir, n_steps=2):
    """Sum kineto `gpu_memcpy` bytes over `n_steps` steady-state steps.  Reported per step."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(2):
        step(model, batch, opt)
    torch.cuda.synchronize()
    path = os.path.join(tmpdir, "memcpy_trace.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(n_steps):
            step(model, batch, opt)
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    tot = defaultdict(lambda: [0, 0])  # name -> [bytes, count]
    for ev in trace.get("traceEvents", []):
        if ev.get("cat") not in ("gpu_memcpy", "Memcpy", "gpu_memset"):
            continue
        nb = (ev.get("args") or {}).get("bytes", 0)
        tot[ev.get("name", "?")][0] += nb
        tot[ev.get("name", "?")][1] += 1
    os.remove(path)
    h2d = sum(v[0] for k, v in tot.items() if "HtoD" in k)
    d2h = sum(v[0] for k, v in tot.items() if "DtoH" in k)
    dtod = sum(v[0] for k, v in tot.items() if "DtoD" in k)
    return {
        "n_steps": n_steps,
        "h2d_MiB_per_step": h2d / n_steps / 2 ** 20,
        "d2h_MiB_per_step": d2h / n_steps / 2 ** 20,
        "dtod_MiB_per_step": dtod / n_steps / 2 ** 20,
        "total_pcie_MiB_per_step": (h2d + d2h) / n_steps / 2 ** 20,
        "by_kind": {k: {"MiB_per_step": v[0] / n_steps / 2 ** 20, "n_per_step": v[1] / n_steps}
                    for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0])},
    }


# ---------------------------------------------------------------------------------------------
# The measurement -- protocol section A, verbatim, same shape as
# `profile_hyclora.measure_headline`
# ---------------------------------------------------------------------------------------------
def measure(arm, cfg, device, args):
    torch.manual_seed(41)
    model = build(arm, cfg, device, args)
    vocab = ph.hf_config(model).vocab_size
    batch = make_batch(cfg, device, vocab)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if getattr(model, "_galore_groups", None) is not None:
        # GaLore replaces the OPTIMIZER, not the model. THIS harness has its own `measure()` with
        # its own optimizer construction, so patching only `profile_hyclora`'s site left the sweep
        # silently running plain AdamW -- caught because `galore_gc_manual` and the full-FT AdamW
        # control came back IDENTICAL to 0.01 MiB at all five lengths.
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "galore"))
        from galore_torch import GaLoreAdamW
        opt = GaLoreAdamW(model._galore_groups, lr=3e-4)
    else:
        opt = torch.optim.AdamW(trainable, lr=3e-4)

    dt = defaultdict(lambda: [0, 0])
    for n, p in model.named_parameters():
        key = ("adapter" if "lora_" in n else "base") + "/" + str(p.dtype)
        dt[key][0] += 1
        dt[key][1] += p.numel() * p.element_size()
    dtype_receipt = {k: {"n_tensors": v[0], "MiB": round(v[1] / 2 ** 20, 2)}
                     for k, v in sorted(dt.items())}
    adapter_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
    n_adapters = sum(1 for n, _ in model.named_parameters() if "lora_" in n)

    # ---- warm-up (protocol A.2).  torch.compile needs a much longer one. ----
    warm = max(3, cfg["iteration_threshold"] + 3)
    if arm.startswith("compile"):
        warm = max(warm, 6)
    t_warm = time.perf_counter()
    for _ in range(warm):
        step(model, batch, opt)
    torch.cuda.synchronize()
    warm_sec = time.perf_counter() - t_warm

    # ---- quiesce; the ONLY legitimate empty_cache() ----
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

    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(model, batch, opt)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    gpu_after = _gpu_state()

    n_fused_calls = dict(FUSED_CALLS)

    # Which `keep` level the fused block actually ran, and whether the O(S^2) attention forward
    # was recomputed.  A row is not quotable without this: `fb_auto` chooses its level from the
    # shape, so the level has to travel with the number.  Both directions are asserted.
    fb_policy = None
    if arm.startswith("fb_"):
        from flashffn import fb_get_counters, fb_policy_report
        fb_policy = fb_policy_report()
        fbc = fb_get_counters()
        if fbc["forward"] == 0 or fbc["backward"] == 0:
            raise RuntimeError(f"{arm}: fused block never executed ({fbc})")
        lv = fb_policy["forwards_by_level"]
        if (lv.get("min", 0) > 0) != (fbc["flash_recompute"] > 0):
            raise RuntimeError(f"{arm}: keep levels {lv} disagree with flash_recompute="
                               f"{fbc['flash_recompute']} -- one of them is lying.")
        fb_policy["counters"] = fbc
        # The offload receipt, for the same reason the keep level is one: an arm named `_offload`
        # whose transfers silently never happened would be an ordinary -- and wrong -- number.
        # `blocked_waits` also travels, because a run that stalled on every fetch is a degraded run
        # and must not be reported as an overlapped one.
        import fb_offload as _fbo
        fb_policy["offload"] = _fbo.fb_offload_stats()
        fb_policy["offload"]["enabled"] = _fbo.fb_offload_enabled()
        if ("offload" in arm) != _fbo.fb_offload_enabled():
            raise RuntimeError(f"{arm}: offload flag and arm name disagree ({fb_policy['offload']})")
        if _fbo.fb_offload_enabled() and fb_policy["offload"]["fetched"] == 0:
            raise RuntimeError(f"{arm}: offload enabled but nothing was ever staged "
                               f"({fb_policy['offload']})")
        print(f"    fb_policy {fb_policy['policy']!r} -> {fb_policy['resolved_by_seq']} | "
              f"levels {lv} | flash_recompute {fbc['flash_recompute']}", flush=True)
        if _fbo.fb_offload_enabled():
            o = fb_policy["offload"]
            print(f"    offload staged {o['fetched']} tensors | blocked_waits {o['blocked_waits']}"
                  f" | D2H {o['bytes_d2h'] / 2 ** 20:.0f} MiB | H2D {o['bytes_h2d'] / 2 ** 20:.0f}"
                  f" MiB", flush=True)

    # ---- WHERE the high-water mark lands, and what is live there (CONTEXT.md section 33.5) ----
    # `profile_hyclora.py` records this for regime A only; regime B (Liger FLCE) is the regime
    # competitor claims are adjudicated in, and the FLCE stack is ~875 MiB smaller, so the peak
    # can sit on a completely different op there.  Replaying the CUDA allocator's own history is
    # safe on EVERY arm (it never walks the autograd graph, so it cannot trigger a non-reentrant
    # recompute the way a saved-tensor-hook pass would).
    peak_comp = None
    if getattr(args, "peak_composition", False):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history(context="all", stacks="python",
                                                     max_entries=800000)
            base_bytes = torch.cuda.memory_allocated()
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak_comp = ph.peak_composition(device.index or 0, base_bytes)
            del out
        except Exception as exc:
            peak_comp = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    # ---- GPU kernel time: the protocol's throughput figure for a box that cannot be quiesced ----
    # `fair_comparison_protocol.md` A.3.6 is explicit: "If the box cannot be quiesced, report GPU
    # kernel time ... as the PRIMARY throughput figure and wall clock as secondary."  This box is
    # shared with tenants outside our PID namespace and today's liger_gc_sdpa run clocked 1069.85
    # ms/step against ~250 archived, so wall clock here is worthless.  `measure_certificate.py`
    # implements this correctly but only over `profile_hyclora.build_model`, which cannot construct
    # the unsloth / Liger / torch.compile arms -- so the one measurement that survives a loud box
    # was unavailable for exactly the arms the mandate is judged against.
    #
    # Sum SELF device time over `DeviceType.CUDA` rows only.  `device_time_total` double-counts
    # every parent annotation (HYCLORA_PROFILE section 1).  Timed in a SEPARATE pass, after the
    # clean un-annotated wall-clock window, so profiler overhead never contaminates ms/step.
    kernel_time = None
    if getattr(args, "kernel_time", False):
        try:
            from torch.profiler import profile as _tprofile, ProfilerActivity
            n_kt = int(getattr(args, "kernel_time_steps", 6))
            for _ in range(2):
                step(model, batch, opt)
            torch.cuda.synchronize()
            with _tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as _prof:
                for _ in range(n_kt):
                    step(model, batch, opt)
                torch.cuda.synchronize()
            evts = [e for e in _prof.key_averages()
                    if str(getattr(e, "device_type", "")).endswith("CUDA")]
            tot = sum(ph.self_dev_time(e) for e in evts)
            by_name = {}
            for e in evts:
                by_name[e.key] = by_name.get(e.key, 0.0) + ph.self_dev_time(e) / n_kt
            kernel_time = {
                "kernel_us_per_step": tot / n_kt,
                "n_steps": n_kt,
                "top_kernels_us": dict(sorted(by_name.items(), key=lambda kv: -kv[1])[:20]),
                "gpu_state": _gpu_state(),
            }
            del _prof
            print(f"    kernel time {kernel_time['kernel_us_per_step'] / 1e3:.3f} ms/step "
                  f"(protocol A.3.6 primary figure on a contended box)", flush=True)
        except Exception as exc:
            kernel_time = {"error": f"{type(exc).__name__}: {exc}"}
        gc.collect()
        torch.cuda.empty_cache()
        opt.zero_grad(set_to_none=True)

    memcpy = None
    if getattr(args, "memcpy", False):
        try:
            memcpy = measure_memcpy(model, batch, opt, args.tmpdir)
        except Exception as exc:
            memcpy = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- gradient liveness ----
    try:
        liveness = adapter_grad_liveness(model, batch, cfg["n_layers"])
        loss_val = liveness["loss"]
        n_dead = liveness["n_dead_trainable_grads"]
    except Exception as exc:
        liveness = {"error": f"{type(exc).__name__}: {exc}"}
        loss_val, n_dead = float(losses[-1]), -1
    opt.zero_grad(set_to_none=True)

    try:
        retained = per_layer_retained_report(model, batch)
    except Exception as exc:
        retained = {"error": f"{type(exc).__name__}: {exc}"}
    opt.zero_grad(set_to_none=True)

    graph = None
    if not arm.startswith("compile"):
        try:
            out = model(**batch)
            graph = _graph_nodes(out.loss)
            out.loss.backward()
            opt.zero_grad(set_to_none=True)
            del out
        except Exception as exc:
            graph = {"error": f"{type(exc).__name__}: {exc}"}
    torch.cuda.synchronize()

    res = {
        "arm": arm,
        "seq": cfg["seq"], "batch": cfg["batch"],
        "adapter_dtype_requested": args.adapter_dtype,
        "adapter_dtype_measured": adapter_dtypes,
        "n_adapter_tensors": n_adapters,
        "dtype_receipt": dtype_receipt,
        "warmup_steps": warm, "warmup_sec": round(warm_sec, 2),
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
        "n_dead_gradients": n_dead, "grad_liveness": liveness,
        "retained": retained,
        "engagement": getattr(model, "_engagement", None),
        "n_fused_calls_during_measurement": n_fused_calls,
        "fb_policy": fb_policy,
        "autograd_graph_nodes": graph,
        "peak_composition": peak_comp,
        "kernel_time": kernel_time,
        "memcpy": memcpy,
        "flce": bool(getattr(args, "flce", False)),
        "force_logits": bool(getattr(args, "force_logits", False)),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
        "gpu_state_before": gpu_before, "gpu_state_after": gpu_after,
        "headless": bool(_HEADLESS["on"]),
        "torch": torch.__version__,
        "comparison_key": ph.comparison_key(model, cfg, args, device),
    }
    del model, opt, batch
    gc.collect()
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------------------------
# Exactness: adapter-gradient rel-L2 against an fp32 / eager reference
# ---------------------------------------------------------------------------------------------
#
# Two problems have to be solved before any two arms are comparable:
#
#   1. **The adapters must be identical.**  PEFT's LoRA init depends on the RNG state at
#      `get_peft_model` time, and unsloth's `random_state=41` is not the same draw as ours.
#      Different A => different grad => a meaningless "difference".  `_canon_adapters` overwrites
#      every LoRA tensor from a name-sorted deterministic stream, so all arms start identical.
#      **B must be non-zero** (PEFT initialises it to zeros, which makes grad_A identically zero
#      and the comparison vacuous), so it is filled too, at a small scale.
#   2. **SDPA's backward is nondeterministic** on this box -- ~4e-3 absolute jitter on adapter
#      grads, larger than a whole int8 quantisation effect.  So the reported rel-L2 must be read
#      against a measured NOISE FLOOR, which `--exactness` obtains by running the reference twice.
_ADAPTER_SCALE = 0.02


def _canon_adapters(model):
    g = torch.Generator(device="cpu").manual_seed(1234)
    names = sorted(n for n, _ in model.named_parameters() if "lora_" in n)
    for n in names:
        p = dict(model.named_parameters())[n]
        v = (torch.randn(tuple(p.shape), generator=g, dtype=torch.float32) * _ADAPTER_SCALE)
        p.data.copy_(v.to(device=p.device, dtype=p.dtype))
    return len(names)


def _canon_name(n):
    """`torch.compile(model)` returns an `OptimizedModule` that prefixes every parameter name with
    `_orig_mod.`.  Without stripping it, `_rel_l2`'s key intersection with the reference is EMPTY
    and the arm silently reports `rel_l2_global = 0.0` -- a perfect score that means "nothing was
    compared".  (Observed exactly that: `n_tensors_compared = 0, n_missing = 308`.)"""
    while n.startswith("_orig_mod."):
        n = n[len("_orig_mod."):]
    return n.replace("._orig_mod.", ".")


def _adapter_grads(model, batch):
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    grads = {_canon_name(n): p.grad.detach().float().cpu().clone()
             for n, p in model.named_parameters() if "lora_" in n and p.grad is not None}
    loss = float(out.loss.detach())
    model.zero_grad(set_to_none=True)
    del out
    return grads, loss


def build_fp32_reference(cfg, device):
    """fp32 weights, fp32 adapters, EAGER attention -- the closest thing to ground truth that
    fits on this GPU, and eager because SDPA's backward is nondeterministic."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained(
        cfg["model"], attn_implementation="eager", torch_dtype=torch.float32)
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_r"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    m.to(device=device, dtype=torch.float32)
    m.train()
    return m


def _rel_l2(a, b):
    num = den = 0.0
    per = {}
    for k in sorted(set(a) & set(b)):
        d = (a[k] - b[k]).double()
        num += float((d * d).sum())
        den += float((b[k].double() ** 2).sum())
        pd = float(d.norm()) / max(float(b[k].double().norm()), 1e-30)
        per[k] = pd
    worst = max(per.items(), key=lambda kv: kv[1]) if per else ("-", float("nan"))
    return {
        "rel_l2_global": (num ** 0.5) / max(den ** 0.5, 1e-30),
        "n_tensors_compared": len(per),
        "n_missing": len(set(b) - set(a)),
        "worst_tensor": worst[0], "worst_rel_l2": worst[1],
        "median_rel_l2": sorted(per.values())[len(per) // 2] if per else float("nan"),
    }


def run_exactness(arms, cfg, device, args):
    # `import unsloth` patches transformers' Llama classes GLOBALLY, so an fp32 "reference" built
    # inside an unsloth process is not a reference at all.  `--ref_out` dumps it from a clean
    # process; `--ref_in` loads it in the unsloth process.
    torch.manual_seed(41)
    if getattr(args, "ref_in", None):
        blob = torch.load(args.ref_in, map_location="cpu", weights_only=False)
        ref_g, ref_loss, noise, n = (blob["grads"], blob["loss"], blob["noise"], blob["n"])
        vocab = blob["vocab"]
        batch = make_batch(cfg, device, vocab)
    else:
        ref = build_fp32_reference(cfg, device)
        n = _canon_adapters(ref)
        vocab = ref.config.vocab_size
        batch = make_batch(cfg, device, vocab)
        ref_g, ref_loss = _adapter_grads(ref, batch)
        ref_g2, _ = _adapter_grads(ref, batch)
        noise = _rel_l2(ref_g2, ref_g)
        del ref
        gc.collect()
        torch.cuda.empty_cache()
        if getattr(args, "ref_out", None):
            torch.save({"grads": ref_g, "loss": ref_loss, "noise": noise, "n": n,
                        "vocab": vocab}, args.ref_out)
            print(f"  [reference dumped to {args.ref_out}]", flush=True)

    out = {"n_adapter_tensors": n, "ref_loss": ref_loss,
           "reference": "fp32 weights + fp32 adapters + eager attention",
           "fp32_reference_selfnoise": noise, "arms": {}}
    for arm in arms:
        try:
            torch.manual_seed(41)
            m = build(arm, cfg, device, args)
            n_can = _canon_adapters(m)
            if n_can != n:
                raise RuntimeError(f"{arm}: canonicalised {n_can} adapters, reference has {n}")
            # Clear HyC-LoRA's `iteration_threshold=5` calibration window before reading any
            # gradient: inside it their codec is still fitting scales and the grads are not
            # representative (measured: rel-L2 1.1e+10 without this warm-up).  Uniform across
            # arms so it cannot advantage anyone.
            for _ in range(6):
                m.zero_grad(set_to_none=True)
                _o = m(**batch)
                _o.loss.backward()
                del _o
            m.zero_grad(set_to_none=True)
            g, loss = _adapter_grads(m, batch)
            r = _rel_l2(g, ref_g)
            r["loss"] = loss
            r["loss_abs_err_vs_fp32"] = abs(loss - ref_loss)
            # self-noise of the arm itself (same arm, two backwards)
            g2, _ = _adapter_grads(m, batch)
            r["self_noise_rel_l2"] = _rel_l2(g2, g)["rel_l2_global"]
            out["arms"][arm] = r
            print(f"  {arm:<22} rel-L2 {r['rel_l2_global']:.3e}  "
                  f"self-noise {r['self_noise_rel_l2']:.3e}  "
                  f"loss {loss:.6f} (fp32 {ref_loss:.6f})", flush=True)
            del m
        except Exception as exc:
            import traceback
            traceback.print_exc()
            out["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
        gc.collect()
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="unsloth_offload")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--adapter_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--model", default=ph.DEFAULT_MODEL,
                    help="HF model id; every shape field is derived from its config")
    ap.add_argument("--assert_matched", default=None, metavar="KEY",
                    help="protocol section B pre-flight: refuse to write unless the comparison key "
                         "observed from the live model equals KEY. Arms measured in separate "
                         "processes may only be pooled into one table when their keys agree.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flce", action="store_true",
                    help="apply Liger FusedLinearCrossEntropy (non-unsloth arms; unsloth has its "
                         "own by default)")
    ap.add_argument("--force_logits", action="store_true",
                    help="unsloth only: UNSLOTH_RETURN_LOGITS=1, i.e. DISABLE their fused CE. "
                         "Diagnostic for the size of the LM-head/CE term inside their stack.")
    ap.add_argument("--memcpy", action="store_true", help="measure H2D/D2H bytes per step")
    ap.add_argument("--kernel_time", action="store_true",
                    help="sum GPU kernel time per step from the profiler. Protocol A.3.6 makes "
                         "this the PRIMARY throughput figure when the box cannot be quiesced; "
                         "measured in a separate pass so ms/step stays un-annotated.")
    ap.add_argument("--kernel_time_steps", type=int, default=6)
    ap.add_argument("--peak_composition", action="store_true",
                    help="replay the allocator history to locate the high-water op and the "
                         "live-block composition there (CONTEXT.md section 33.5)")
    ap.add_argument("--ref_out", default=None, help="--exactness: dump the fp32 reference grads")
    ap.add_argument("--ref_in", default=None, help="--exactness: load a dumped fp32 reference")
    ap.add_argument("--exactness", action="store_true",
                    help="adapter-gradient rel-L2 against an fp32+eager reference (no timing)")
    ap.add_argument("--unsloth_bf16_adapters", action="store_true",
                    help="cast unsloth's fp32 LoRA to bf16, matching every other arm (protocol E.1)")
    ap.add_argument("--no_head", action="store_true")
    ap.add_argument("--interleave", type=int, default=1)
    ap.add_argument("--tmpdir", default="/tmp")
    args = ap.parse_args()

    if args.force_logits:
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _HEADLESS["on"] = bool(args.no_head)

    cfg = ph.make_cfg(args.batch, args.seq, model=args.model)

    arms = [a for a in args.arm.split(",") if a]

    if args.exactness:
        res = run_exactness(arms, cfg, device, args)
        out = args.out or os.path.join(RESULTS_DIR, f"unsloth_exactness_seq{cfg['seq']}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\n-> {out}")
        return

    rows, runs = {}, []
    for rep in range(args.interleave):
        for arm in arms:
            print(f"\n=== {arm} rep {rep + 1}/{args.interleave} ===", flush=True)
            try:
                r = measure(arm, cfg, device, args)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                r = {"arm": arm, "error": f"{type(exc).__name__}: {exc}"}
                # An OOM leaves the model alive in the traceback's frames, so the NEXT arm in the
                # process would start with several GiB already resident and OOM in turn.  Drop the
                # exception state explicitly before continuing.
                sys.exc_info()
                traceback.clear_frames(exc.__traceback__)
                del exc
                gc.collect()
                torch.cuda.empty_cache()
            r["rep"] = rep
            runs.append(r)
            if "error" not in r:
                eng = r.get("engagement") or {}
                print(f"    peak(alloc) {r['train_step_peak_alloc_mib']:.2f} MiB | "
                      f"reserved {r['train_step_peak_reserved_mib']:.2f} | "
                      f"floor {r['resident_floor_mib']:.2f} | "
                      f"{r['ms_per_step_median']:.2f} ms/step | "
                      f"adapters {r['adapter_dtype_measured']} ({r['n_adapter_tensors']}) | "
                      f"engaged {eng}", flush=True)
                if r.get("memcpy"):
                    m = r["memcpy"]
                    print(f"    memcpy: H2D {m.get('h2d_MiB_per_step', -1):.2f} MiB/step, "
                          f"D2H {m.get('d2h_MiB_per_step', -1):.2f} MiB/step", flush=True)
            if arm not in rows or (
                    "error" not in r
                    and r.get("ms_per_step_median", 1e9) < rows[arm].get("ms_per_step_median", 1e9)):
                rows[arm] = r

    # Protocol section B pre-flight.  Two checks, both refusing to write rather than warning:
    #   * every arm in THIS file agrees on the held-identical fields, and
    #   * that key equals the cell declared on the command line, so rows measured in separate
    #     processes (unsloth needs its own) can be pooled into one table without hand-checking.
    keys = {a: (r.get("comparison_key") or {}).get("key")
            for a, r in rows.items() if "error" not in r}
    if len(set(keys.values())) > 1:
        for a, r in rows.items():
            if "error" not in r:
                print(f"  !! {a}: key={keys[a]} "
                      f"fields={json.dumps((r['comparison_key'])['fields'], sort_keys=True)}")
        raise SystemExit(f"ABORT: arms in this run are not a matched cell: {keys}")
    observed = next(iter(keys.values()), None)
    if args.assert_matched and observed and args.assert_matched != observed:
        bad = next(r for r in rows.values() if "error" not in r)
        print(json.dumps(bad["comparison_key"], indent=2, sort_keys=True))
        raise SystemExit(f"ABORT: comparison key {observed} != declared cell "
                         f"{args.assert_matched}; refusing to write {args.out}")
    if observed:
        print(f"\ncomparison key (protocol B, observed from the live model): {observed}")

    tag = "_flce" if args.flce else ""
    tag += "_nohead" if args.no_head else ""
    tag += "_logits" if args.force_logits else ""
    tag += "_bf16ad" if args.unsloth_bf16_adapters else ""
    out = args.out or os.path.join(RESULTS_DIR, f"unsloth_seq{cfg['seq']}{tag}.json")
    with open(out, "w") as f:
        json.dump({"cfg": cfg, "rows": rows, "all_reps": runs,
                   "torch": torch.__version__,
                   "unsloth": UNSLOTH_VERSION}, f, indent=2, default=str)
    print(f"\n-> {out}")

    hdr = (f"{'arm':<20} {'peak alloc':>11} {'peak resv':>10} {'floor':>9} {'peak-floor':>11} "
           f"{'ms/step':>9} {'ret MiB/lyr':>12} {'H2D MiB':>9} {'D2H MiB':>9}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for arm in arms:
        r = rows.get(arm, {})
        if "error" in r:
            print(f"{arm:<20} ERROR {r['error'][:80]}")
            continue
        rt = (r.get("retained") or {}).get("retained_MiB_per_layer_median")
        m = r.get("memcpy") or {}
        h2d, d2h = m.get("h2d_MiB_per_step"), m.get("d2h_MiB_per_step")
        s_rt = f"{rt:.3f}" if isinstance(rt, float) else "-"
        s_h2d = f"{h2d:.1f}" if isinstance(h2d, float) else "-"
        s_d2h = f"{d2h:.1f}" if isinstance(d2h, float) else "-"
        print(f"{arm:<20} {r['train_step_peak_alloc_mib']:>11.2f} "
              f"{r['train_step_peak_reserved_mib']:>10.2f} {r['resident_floor_mib']:>9.2f} "
              f"{r['peak_minus_floor_mib']:>11.2f} {r['ms_per_step_median']:>9.2f} "
              f"{s_rt:>12} {s_h2d:>9} {s_d2h:>9}")


if __name__ == "__main__":
    main()
