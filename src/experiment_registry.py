"""Method registry for production/camera-ready experiments — one place that knows how to build
every benchmarked method, which combinations are legal, and how to PROVE each one engaged.

WHY THIS FILE EXISTS
--------------------
`profile_hyclora.build_model` knows how to construct each published baseline (that is where the
campaign's integrations live, and reusing it is what makes a production row comparable with the
memory numbers in CONTEXT.md §16). `train_glue.py` knows how to train and evaluate a task and how to
write a CSV row under a file lock. Neither knows which METHOD COMBINATIONS are meaningful, and that
is the question a camera-ready sweep gets wrong silently.

THE GOVERNING RULE, MEASURED NOT ASSUMED
----------------------------------------
**A combination is legal only if EVERY constituent can be proved to have executed.** This is not
defensive programming; it is the difference between a result and a fiction:

  * `alst_tiledmlp` + `fb`  APPLIES WITHOUT ERROR AND IS A LIE. ALST patches `LlamaMLP.forward`;
    the fused block replaces the entire decoder-layer forward, so `LlamaMLP.forward` is never
    called. The run completes, the memory looks like the fused block's, and the row claims to be
    ALST. Measured 2026-08-11.
  * `qlora` + `fb`, `minis` + `fb`, `streambp` + `fb` all REFUSE LOUDLY (GUARD 12 / "no decoder
    layers" / "decoder layer carries sub-module(s) ['base_layer']"). Those are safe failures.

So every `MethodSpec` carries an `engagement` callable returning a counter dict, and
`verify_engagement` raises unless every declared method reports non-zero work. A method that cannot
prove itself must not produce a CSV row.

COMPATIBILITY IS DATA, NOT INFERENCE
------------------------------------
`FB_COMPAT` below records the MEASURED outcome for each method against the fused block, with the
observed error text or the reason the pair is meaningless. Do not "fix" a `REFUSES` entry by
loosening a guard; the guard is the finding.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------------
# Compatibility with the fused decoder block, measured 2026-08-11. See the module docstring.
# ---------------------------------------------------------------------------------------------
COMPOSES = "COMPOSES"          # both mechanisms provably run
REFUSES = "REFUSES"            # raises at patch time -- a SAFE failure
SILENT_NOOP = "SILENT_NOOP"    # applies but one mechanism is dead -- MUST be blocked
NOT_APPLICABLE = "N/A"         # the pair is meaningless (e.g. nothing to compose)

FB_COMPAT: Dict[str, Dict] = {
    "baseline":  {"status": COMPOSES, "why": "plain LoRA; the fused block IS the change under test"},
    "qlora":     {"status": REFUSES,
                  "why": "GUARD 12: base weight is a packed Params4bit (uint8, flat), not a dense "
                         "[out,in] float matrix. The block multiplies it with F.linear; consuming a "
                         "packed weight needs a dequantise-in-kernel path that does not exist."},
    "minis":     {"status": REFUSES,
                  "why": "'no decoder layers on minisequence' -- their wrapper hides the layer stack."},
    "streambp":  {"status": REFUSES,
                  "why": "'decoder layer carries sub-module(s) [base_layer]' -- StreamDecoderLayer "
                         "wraps the original layer, which the fused block does not implement."},
    "alst":      {"status": SILENT_NOOP,
                  "why": "⚠ APPLIES WITHOUT ERROR AND ALST GOES DEAD. ALST patches "
                         "LlamaMLP.forward; the fused block replaces the whole decoder-layer "
                         "forward, so their tiling is never called. BLOCKED, not because it fails, "
                         "but because it succeeds while meaning nothing."},
    "zero3":     {"status": REFUSES,
                  "why": "⚠ CORRECTED 2026-08-11 — this entry said COMPOSES on INFERENCE and the "
                         "smoke test refuted it. ZeRO-3 `offload_param` gathers each parameter via "
                         "**module forward hooks**; the fused block replaces the decoder layer's "
                         "forward with its own autograd Function and reads `proj.weight` directly, "
                         "so the gather hook never fires and the weight is still on CPU when the "
                         "Triton kernel dereferences it: `ValueError: Pointer argument (at 0) "
                         "cannot be accessed from Triton (cpu tensor?)`. Composing them needs the "
                         "block to participate in ZeRO's gather protocol. Structural, not a bug."},
    "galore":    {"status": COMPOSES,
                  "why": "GaLore replaces only the OPTIMIZER; the module tree is untouched. Full "
                         "fine-tuning regime, so the block runs with family='full'."},
    "lomo":      {"status": COMPOSES,
                  "why": "MEASURED ADDITIVE: seq 16384 full-FT, LOMO 8867.55 + fb alone 11142.78 "
                         "vs control 13188.91 -> composed 6819.43; sum-of-parts 6367.49 against "
                         "measured 6369.48, i.e. additive to 1.99 MiB (0.03%)."},
    "adalomo":   {"status": COMPOSES, "why": "as `lomo`."},
}

# Methods whose own mechanism already implements a chunked/fused LM head + loss. Running Liger's
# FLCE on top would be two implementations of one optimisation and would measure neither.
FLCE_FORBIDDEN = ("minis", "streambp")

# ---------------------------------------------------------------------------------------------
# HEAD compatibility. MEASURED 2026-08-14 by running all 13 arms on `glue:sst2` (phase 1 of
# `validate_glue_runner.py`); 12 of 13 built and trained on a sequence-classification head.
#
# ⚠ THIS IS A METHOD-LEVEL FACT, NOT A MISSING FEATURE. Declaring it here means the sweep refuses
#   the cell before building a model, the same way FB_COMPAT does -- rather than discovering it as
#   an AttributeError 40 frames down, after the dataset has been tokenised and the model loaded.
# ---------------------------------------------------------------------------------------------
SEQ_CLS_REFUSES = {
    "streambp":
        "StreamBP's mechanism IS the chunked LM head: `stream_model.py:622-629` reads "
        "`model.lm_head.weight.grad` and evaluates `model.lm_head(hidden[:, start:end, :])` "
        "chunk by chunk along the SEQUENCE. `LlamaForSequenceClassification` has no `lm_head` -- "
        "it has `score`, and it emits ONE logit vector per sequence, not one per token, so there "
        "is nothing to chunk. Observed: `AttributeError: 'LlamaForSequenceClassification' object "
        "has no attribute 'lm_head'`. This is not an unimplemented feature; the method does not "
        "apply to sentence classification and must be reported absent, with this reason.",
}


def resolve_fb_variant(method: str, with_fb: bool, fb_variant: str) -> None:
    """Raise `CombinationRefused` if the fused-block VARIANT cannot apply to this method.

    ⚠ WP-E (`wstream`) STREAMS THE **FROZEN** BASE WEIGHTS, so it is structurally incompatible with
    every full-fine-tuning arm. MEASURED 2026-08-14: `lomo_fb_wstream`, `galore_fb_wstream` and
    `adalomo_fb_wstream` all reach `fb_wstream.install()` and are declined with
    `'layers [0..21] have TRAINABLE base weights'`.

    The refusal is not incidental -- it is what makes the mechanism correct. The block saves the
    seven base weights as `None` and re-acquires them in the backward, which is sound ONLY because
    nothing writes them (no version counter needed) and they carry no gradient. Under full FT the
    optimizer writes them and they do produce gradients, so streaming them would be wrong, not
    merely unsupported.
    """
    if with_fb and fb_variant == "wstream" and method in FULL_FT_REGIME:
        raise CombinationRefused(
            f"{method} + fb_variant=wstream: WP-E streams the FROZEN base weights, and {method} is "
            f"a FULL fine-tuning method whose base weights are all trainable "
            f"(fb_wstream.install declines: 'layers have TRAINABLE base weights'). The frozen-ness "
            f"is what makes saving them as None and re-acquiring them in the backward correct, so "
            f"this is a structural incompatibility, not a missing feature. Use fb_variant=min.")


def resolve_head(method: str, head: str) -> None:
    """Raise `CombinationRefused` if `method` cannot be built on `head`. Called before any model
    is constructed, so a refused cell costs nothing."""
    if head == "seq_cls" and method in SEQ_CLS_REFUSES:
        raise CombinationRefused(f"{method} + sequence-classification head: {SEQ_CLS_REFUSES[method]}")

# Methods that train ALL parameters. Their rows answer "can I full-finetune cheaply?", NOT "whose
# activation cache is smaller", and must never be tabulated beside a LoRA arm without this flag.
FULL_FT_REGIME = ("galore", "lomo", "adalomo")

# Methods that are LOSSY. The exactness column in any table must be driven from this, not prose.
LOSSY = ("qlora",)


@dataclass
class MethodSpec:
    """One benchmarked method: how to name it as an arm, and how to prove it ran."""
    key: str
    arm_template: str                       # `{fb}` expands to the fused-block infix, if any
    engagement: Callable[[object], Dict]    # model -> counter dict; empty/zero means DID NOT RUN
    needs_pythonpath: Optional[str] = None  # e.g. ALST needs temp/ds_alst before interpreter start
    notes: str = ""
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Engagement probes. Each returns a dict of counters that must be non-zero for the row to be valid.
# These read the receipts the campaign's integrations already attach to the model, so they prove
# the mechanism EXECUTED rather than that the import succeeded.
# ---------------------------------------------------------------------------------------------
def _eng_fb(model) -> Dict:
    """The fused block: counters live on the module that `flashffn` patched."""
    try:
        from flashffn import fb_get_counters
        c = dict(fb_get_counters() or {})
    except Exception:
        c = {}
    return {"fb_forward": c.get("forward", 0), "fb_backward": c.get("backward", 0),
            "fb_patched_layers": c.get("patched_layers", 0)}


def _eng_lora_baseline(model) -> Dict:
    """The `baseline` method is plain LoRA -- its engagement is that adapters EXIST AND TRAIN.

    ⚠ Do NOT give this method the fused-block probe. `baseline` without fb is `gc_manual_sdpa`,
    which has no fb counters, so an fb probe reports zero and the gate refuses a perfectly valid
    row. (Caught by the first all-combinations smoke test.) When fb IS requested it is verified
    separately, as its own constituent.
    """
    n_ad = sum(1 for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    return {"lora_trainable_tensors": n_ad}


def _eng_qlora(model) -> Dict:
    n4 = sum(1 for _n, m in model.named_modules() if type(m).__name__ in ("Linear4bit", "Params4bit"))
    u8 = sum(1 for _n, p in model.named_parameters() if str(p.dtype) == "torch.uint8")
    return {"linear4bit_modules": n4, "uint8_param_tensors": u8}


def _eng_minis(model) -> Dict:
    """⚠ COUNT THE TWO HALVES SEPARATELY. Mini-Sequence has two mechanisms -- per-token MLP tiling
    (`LlamaMLPWarpper`) and chunking of the LM head + loss (`LlamaForCausalLMWarpper`/`LMheadWarpper`,
    their headline memory claim). A single total hides which one ran.

    MEASURED 2026-08-14 on `glue:sst2`: the total was 22 = the 22 decoder MLPs and NOTHING else,
    because the model is a `LlamaForSequenceClassification` and their LM-head wrapper patches
    `LlamaForCausalLM`. So on a classification head Mini-Sequence is HALF-ENGAGED: the MLP tiling is
    genuinely theirs and genuinely running, the LM-head chunking cannot apply because there is no LM
    head. That is a caveat a GLUE table must carry, and it is only visible if the counts are split.
    """
    mlp = sum(1 for _n, m in model.named_modules() if type(m).__name__ == "LlamaMLPWarpper")
    head = sum(1 for _n, m in model.named_modules()
               if type(m).__name__ in ("LlamaForCausalLMWarpper", "LMheadWarpper"))
    return {"minis_mlp_wrappers": mlp, "minis_lm_head_wrappers": head,
            "minis_wrapped_modules": mlp + head}


def _eng_streambp(model) -> Dict:
    n = sum(1 for _n, m in model.named_modules() if type(m).__name__ == "StreamDecoderLayer")
    return {"stream_decoder_layers": n}


def _eng_alst(model) -> Dict:
    """⚠ Class-level patch, so presence is NOT enough -- it must be CALLED. `tiled_mlp_forward_common`
    is what ALST installs; if the fused block took over the decoder forward this stays zero, which is
    exactly the silent no-op this registry exists to block."""
    import transformers.models.llama.modeling_llama as ML
    name = getattr(ML.LlamaMLP.forward, "__name__", "")
    return {"alst_mlp_forward_patched": int(name == "tiled_mlp_forward_common"),
            "alst_num_shards": (getattr(model, "_alst_receipt", {}) or {}).get("num_shards_auto", 0),
            "alst_tiling_active": int(bool((getattr(model, "_alst_receipt", {}) or {})
                                           .get("tiling_active", False)))}


def _eng_zero3(model) -> Dict:
    r = getattr(model, "_zero3_receipt", {}) or {}
    return {"zero3_engine": int(type(model).__module__.startswith("deepspeed")),
            "zero3_stage": r.get("stage", 0), "zero3_offload_param": int(bool(r.get("offload_param")))}


def _eng_galore(model) -> Dict:
    g = getattr(model, "_galore_groups", None)
    return {"galore_projected_tensors": len(g[1]["params"]) if g else 0}


def _eng_lomo(model) -> Dict:
    return {"lomo_optimizer": int(getattr(model, "_lomo_opt", None) is not None)}


REGISTRY: Dict[str, MethodSpec] = {
    "baseline": MethodSpec("baseline", "{fb}", _eng_lora_baseline,
                           notes="plain LoRA; with fb this is our own arm"),
    "qlora":    MethodSpec("qlora", "qlora_nf4{fb}", _eng_qlora,
                           notes="4-bit NF4 base, double-quant, bf16 compute. LOSSY. "
                                 "`_norm32` variant reproduces their fp32-norm recipe under autocast "
                                 "and is REQUIRED for quality runs."),
    "minis":    MethodSpec("minis", "minis{fb}", _eng_minis, notes="no --flce (own chunked LM head)"),
    "streambp": MethodSpec("streambp", "streambp{fb}", _eng_streambp,
                           notes="no --flce; fuses backward into forward; chunk = seq//3 (their rule)"),
    "alst":     MethodSpec("alst", "alst_tiledmlp{fb}", _eng_alst,
                           needs_pythonpath="temp/ds_alst",
                           notes="num_shards=ceil(seq/hidden) -> NO tiling below seq 4096 on a "
                                 "2048-hidden model; that is their rule, report it as such"),
    "zero3":    MethodSpec("zero3", "zero3_offload{fb}", _eng_zero3,
                           notes="use gc_hf; ZeRO-3 is incompatible with non-reentrant checkpointing"),
    "galore":   MethodSpec("galore", "galore{fb}", _eng_galore, notes="FULL FT regime"),
    "lomo":     MethodSpec("lomo", "lomo{fb}", _eng_lomo, notes="FULL FT regime; no optimizer state"),
    "adalomo":  MethodSpec("adalomo", "adalomo{fb}", _eng_lomo, notes="FULL FT regime"),
}


class CombinationRefused(RuntimeError):
    """Raised for a combination that is illegal by construction, BEFORE any GPU time is spent."""


class EngagementFailure(RuntimeError):
    """Raised when a method that was requested cannot be proved to have executed."""


def resolve_arm(method: str, with_fb: bool, keep: str = "min") -> str:
    """Arm string for `profile_hyclora.build_model`, or raise if the pair is illegal.

    Refusing here — before the model is built — is deliberate: a camera-ready sweep must not spend
    GPU hours producing a row it will have to throw away.
    """
    if method not in REGISTRY:
        raise KeyError(f"unknown method {method!r}; known: {sorted(REGISTRY)}")
    if with_fb:
        compat = FB_COMPAT.get(method, {})
        if compat.get("status") in (REFUSES, SILENT_NOOP, NOT_APPLICABLE):
            raise CombinationRefused(
                f"{method} + fused-block is {compat.get('status')}: {compat.get('why')}")
    spec = REGISTRY[method]
    return spec.arm_template.format(fb="_fb" if with_fb else "")


def verify_engagement(model, methods, strict: bool = True) -> Dict:
    """Prove every requested method actually ran. Returns the merged counter dict.

    `methods` is the list of constituents, e.g. ['lomo', 'fb']. A zero counter is a hard error:
    a row whose method did not execute is worse than a missing row, because it looks like data.

    ⚠ THE `_active` CONVENTION, ADDED 2026-08-14 AFTER IT LET AN INERT ARM THROUGH.
    `any(counters)` treats every counter as equally good evidence, so a counter meaning "the patch
    is INSTALLED" outvotes one meaning "the patch DID SOMETHING". Measured: `alst` on `glue:sst2`
    at seq 128 reported `alst_mlp_forward_patched: 1, alst_num_shards: 1, alst_tiling_active: 0`
    and passed -- ALST's own rule is `num_shards = ceil(seq/hidden)`, which is 1 on a 2048-hidden
    model at any GLUE sequence length, i.e. their method configured itself to do nothing and we
    wrote the row anyway. `_eng_alst`'s own docstring warned about exactly this failure and the
    aggregator did not honour it.

    **A counter whose name ends in `_active` is a LIVENESS counter: if it is present and zero, the
    method is dead regardless of what any other counter says.**
    """
    counters: Dict = {}
    dead = []
    for m in methods:
        probe = _eng_fb if m == "fb" else REGISTRY[m].engagement
        c = probe(model)
        counters.update(c)
        liveness = {k: v for k, v in c.items() if k.endswith("_active")}
        if liveness:
            if not all(liveness.values()):
                dead.append(m)
        elif not any(v for v in c.values()):
            dead.append(m)
    counters["engagement_ok"] = not dead
    if dead and strict:
        raise EngagementFailure(
            f"requested method(s) {dead} report ZERO work: {counters}. This row would claim a "
            f"method that did not run. Refusing to write it.")
    return counters
