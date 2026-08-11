"""Port of HyC-LoRA (MLSys 2025) onto transformers 4.51.3.

Upstream ships a fork of transformers 4.41's `modeling_llama.py` and swaps in a fused
whole-decoder-layer autograd Function during training. We cannot vendor that fork (our stack is
transformers 4.51.3), but we do not need to: their Function is pure tensor code with no
transformers internals. This module reproduces only the *call site*, against the 4.51 API.

What upstream's decoder layer did (models/llama/modeling_llama.py L745-790):

    if self.training and self.fused_llama_layer.use_hyclora:
        cos, sin = self.self_attn.rotary_emb.forward(x_fake, seq_len)   # 4.41 API
        out = self.fused_llama_layer.forward(hidden_states, ln1.weight, None, cos, sin,
                                             q.base_layer, q.lora_A, q.lora_B, ..., mask,
                                             num_heads, head_dim, num_kv_heads)
        return (out,)
    else:
        return self.forward_old(...)

Differences handled here, and why:

1. Rotary embeddings. In 4.41 `rotary_emb.forward(x, seq_len)` returned cos/sin of shape
   [seq_len, head_dim]. In 4.51 they are computed once at model level and passed into the layer
   as `position_embeddings`, shaped [batch, seq_len, head_dim]. Their Triton RoPE kernel calls
   `cos.squeeze()` and then indexes with `cos.stride(0)`, which is only correct for a 2D
   [seq_len, head_dim] tensor. We therefore slice the batch dimension off -- after asserting
   every row is identical, since upstream's formulation inherently assumes every sequence in the
   batch shares positions 0..S-1.

2. Attention mask. Their Function does `s = s + attention_mask` before the softmax, so it needs
   a 4D *additive* float mask. With `attn_implementation="sdpa"`/`"flash_attention_2"` the mask
   handed to the layer can be None (causality expressed via `is_causal`), which would silently
   make their attention NON-CAUSAL. We hard-require an eager-style 4D additive mask and raise
   otherwise -- a silent failure here would corrupt every number downstream.

3. Base weights. Upstream is QLoRA-only and calls `dequantize_nf4` unconditionally. To also
   support a bf16 base (needed for an apples-to-apples comparison against our bf16 arms) we
   attach `quant_state = None` to un-quantised base weights; `compute_utils._base_weight_t`
   then uses the raw weight. The vendored algorithm itself is untouched.

4. LoRA scaling. Their Function applies `x @ A @ B` with no `alpha/r` factor, so it is only
   correct when `lora_alpha == r`. Upstream enforces this in their runner; we enforce it here.
"""

import dataclasses
from typing import Optional, Tuple

import torch

from .layers.fused_llama_layer_baseline import FusedLlamaLayerBaseline
from .layers.fused_llama_layer_intra import FusedLlamaLayerIntra
from .layers.fused_llama_layer_intra_inter import FusedLlamaLayerIntraInter
from .layers.fused_llama_layer_intra_inter_flash import FusedLlamaLayerIntraInterFlash
from .layers.fused_llama_layer_intra_inter_full_fuse import FusedLlamaLayerIntraInterFullFuse

LAYER_TYPES = {
    "baseline": FusedLlamaLayerBaseline,
    "intra": FusedLlamaLayerIntra,
    "intra_inter": FusedLlamaLayerIntraInter,
    "intra_inter_flash": FusedLlamaLayerIntraInterFlash,
    "intra_inter_full_fuse": FusedLlamaLayerIntraInterFullFuse,
}

# Layer types whose attention block is FlashAttention. They store NO attention map, express
# causality with `is_causal` rather than an additive mask, and therefore have a completely
# different set of call-site requirements from the eager layers -- see `_make_forward`.
FLASH_LAYER_TYPES = {"intra_inter_flash"}

# Layer types that materialise the S x S attention map and sparsify it. These are the ones for
# which `softmax_outlier_ratio` is meaningful (and for which 0 is a measured trap).
ATTN_MAP_LAYER_TYPES = {"intra", "intra_inter", "intra_inter_full_fuse"}

# Fail-loud honesty counters, mirroring the FlashFFN v3 convention: if the fused path never
# executes, the run must error rather than silently report baseline numbers under a HyC-LoRA name.
_HYCLORA_COUNTERS = {"forward": 0, "backward": 0, "patched_layers": 0}


def reset_counters():
    _HYCLORA_COUNTERS["forward"] = 0
    _HYCLORA_COUNTERS["backward"] = 0
    _HYCLORA_COUNTERS["patched_layers"] = 0


def get_counters():
    return dict(_HYCLORA_COUNTERS)


@dataclasses.dataclass
class HyCLoRAConfig:
    """Mirrors upstream's `hyclora_args` dataclass field-for-field."""
    use_hyclora: bool = True
    layer_type: str = "intra_inter"
    q_bit: int = 4
    softmax_outlier_ratio: float = 0.05
    layernorm_outlier_ratio: float = 0.005
    iteration_threshold: int = 5
    # VERIFICATION ONLY (currently honoured by `intra_inter_flash`): bypass every codec and keep
    # the dataflow otherwise identical, so the fused layer can be compared against stock HF for
    # exactness. Never set by a measurement arm.
    no_compress: bool = False

    def validate(self):
        if self.layer_type not in LAYER_TYPES:
            raise ValueError(
                f"unknown hyclora layer_type {self.layer_type!r}; "
                f"expected one of {sorted(LAYER_TYPES)}"
            )
        if self.q_bit not in (2, 4, 8):
            raise ValueError(f"hyclora q_bit must be one of 2/4/8, got {self.q_bit}")
        # Measured trap (results/hyclora/grad_diagnostic.json): with layer_type intra/intra_inter
        # a softmax_outlier_ratio of 0 sets the sparsification threshold to the attention max, so
        # essentially the whole attention map is discarded and grad_q/grad_k come back EXACTLY
        # zero -- training silently does nothing on the attention path. Upstream always runs 0.05.
        #
        # The FlashAttention layer types never materialise the map, so they have no
        # `softmax_outlier_ratio` at all and this guard does not apply to them. Their analogous
        # silent-failure knob is checked separately -- see `verify_hyclora_layer.py`; empirically
        # there is none: with q_bit=2 and layernorm_outlier_ratio=0 every LoRA gradient is still
        # non-zero, because grad_q/grad_k come out of the FA kernel rather than out of a
        # thresholded map that can be emptied.
        if self.layer_type in ATTN_MAP_LAYER_TYPES:
            if not self.softmax_outlier_ratio > 0:
                raise ValueError(
                    f"hyclora softmax_outlier_ratio must be > 0 for layer_type={self.layer_type!r} "
                    f"(got {self.softmax_outlier_ratio}). At 0 the attention map is fully "
                    f"sparsified and grad_q/grad_k are identically zero. Upstream uses 0.05."
                )


def _lora_triplet(module, name):
    """Return (base_layer, lora_A, lora_B) for a PEFT-wrapped projection, or raise."""
    for attr in ("base_layer", "lora_A", "lora_B"):
        if not hasattr(module, attr):
            raise ValueError(
                f"HyC-LoRA requires every projection to be PEFT-LoRA wrapped, but {name} "
                f"has no .{attr}. Upstream targets q,k,v,o,gate,up,down -- pass all seven in "
                f"--adapter_target_modules (or leave the default)."
            )
    if "default" not in module.lora_A:
        raise ValueError(f"{name}: expected PEFT adapter named 'default', got {list(module.lora_A)}")
    return module.base_layer, module.lora_A, module.lora_B


def _check_scaling(module, name):
    scaling = module.scaling.get("default", 1.0)
    if abs(scaling - 1.0) > 1e-6:
        raise ValueError(
            f"{name}: HyC-LoRA's fused kernels compute x@A@B with NO alpha/r scaling, so they "
            f"are only correct when lora_alpha == r (scaling == 1.0); got scaling={scaling}. "
            f"Upstream enforces the same constraint (run_gsm8k.py L373)."
        )
    # lora_dropout is an nn.ModuleDict (no .get); scaling above is a plain dict.
    dropout = module.lora_dropout["default"] if "default" in module.lora_dropout else None
    p = getattr(dropout, "p", 0.0)
    if p and p > 0.0:
        raise ValueError(
            f"{name}: HyC-LoRA's fused path ignores LoRA dropout, but lora_dropout={p}. "
            f"Upstream runs dropout 0.0; set --lora_dropout 0 for a faithful comparison."
        )


def _prepare_base_weight(linear, name):
    """Ensure `.weight.quant_state` exists so compute_utils can branch NF4 vs raw."""
    w = linear.weight
    if not hasattr(w, "quant_state"):
        # plain nn.Linear (bf16/fp16/fp32 base) -> signal "no NF4" to _base_weight_t
        w.quant_state = None
    if w.quant_state is None and w.dim() != 2:
        raise ValueError(f"{name}: expected a 2D base weight, got shape {tuple(w.shape)}")
    return linear


def _check_adapter_dtype(layer, hidden_dtype, name):
    """Trap #3 from the eager port, re-checked for the FA path.

    `compute_utils.lora_backward` computes `grad_y.to(w_dequant.dtype) @ w_lora_b.mT` -- the
    *uncast* `w_lora_b`. With PEFT's default `autocast_adapter_dtype=True` the adapters stay fp32
    while the base is bf16, and that matmul is a dtype mismatch. It also silently changes the
    memory profile: `peft/tuners/lora/layer.py` casts the input activation to the adapter dtype,
    so fp32 adapters make every LoRA-branch activation fp32 (~2150 MiB at seq 1024 batch 2).
    """
    sa, mlp = layer.self_attn, layer.mlp
    for pname, mod in (("q_proj", sa.q_proj), ("k_proj", sa.k_proj), ("v_proj", sa.v_proj),
                       ("o_proj", sa.o_proj), ("gate_proj", mlp.gate_proj),
                       ("up_proj", mlp.up_proj), ("down_proj", mlp.down_proj)):
        for which, d in (("lora_A", mod.lora_A["default"].weight.dtype),
                         ("lora_B", mod.lora_B["default"].weight.dtype)):
            if d != hidden_dtype:
                raise RuntimeError(
                    f"HyC-LoRA port ({name}): {pname}.{which} is {d} but the activations are "
                    f"{hidden_dtype}. Their fused kernels use `w_lora_b.mT` uncast, and PEFT's "
                    f"fp32-adapter default additionally makes every LoRA-branch activation fp32 "
                    f"(worth ~2150 MiB of training-step peak at seq 1024 / batch 2). Cast the "
                    f"adapters to the base dtype in EVERY arm being compared, not just this one."
                )


def _validate_flash_mask(attention_mask, seq_len, device):
    """Trap #2, inverted, for the FlashAttention path.

    On the *eager* path the trap is a `None` mask: their kernel adds `attention_mask` to the
    scores, so `None` silently makes attention non-causal. On the *FA* path the trap is the
    opposite. Upstream's `models/llama_flash_attn/modeling_llama.py` hands the fused layer a 4-D
    additive causal mask, which their FA forward then feeds to `unpad_input` as if it were a 2-D
    0/1 key-padding mask (`attention_mask.sum(dim=-1)` over an additive float mask). We express
    causality with `is_causal=True` instead, and refuse anything that is not a plain, unpadded
    causal mask -- padding would need varlen packing, which we do not have without flash_attn.
    """
    if attention_mask is None:
        return                                   # sdpa/flash path, unpadded -> pure causal
    if attention_mask.dim() == 2:
        if not bool(attention_mask.all()):
            raise RuntimeError(
                "HyC-LoRA FA port: the batch contains padding. Upstream handles this with "
                "flash_attn's varlen unpad/pad, which is unavailable here (the Dao-AILab "
                "package does not build against torch 2.10). Use fixed-length unpadded blocks.")
        return
    if attention_mask.dim() != 4:
        raise RuntimeError(
            f"HyC-LoRA FA port: unexpected attention_mask rank {attention_mask.dim()}.")
    m = attention_mask[:, :, :, :seq_len]
    keep = (m == 0)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    if not bool((keep == causal).all()):
        raise RuntimeError(
            "HyC-LoRA FA port: the 4-D additive mask is not a plain causal mask (padding or a "
            "custom pattern). FlashAttention can only express `is_causal`; running anyway would "
            "silently ignore the extra masking.")


def _make_forward(layer, fused, cfg, n_heads, head_dim, n_kv_heads):
    orig_forward = layer.forward
    is_flash = cfg.layer_type in FLASH_LAYER_TYPES
    checked = {"dtype": False, "mask": False}

    def hyclora_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # Upstream gates the fused path on training mode; evaluation runs the stock HF path,
        # so reported eval metrics are computed by unmodified transformers.
        if not (layer.training and cfg.use_hyclora):
            return orig_forward(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )

        if position_embeddings is None:
            raise RuntimeError(
                "HyC-LoRA port: position_embeddings is None. transformers>=4.48 must supply "
                "(cos, sin) to the decoder layer; this model appears to use an older path."
            )
        cos, sin = position_embeddings
        if cos.dim() == 3:
            # [B, S, Dh] -> [S, Dh]; valid only if every sequence shares positions 0..S-1,
            # which is what upstream's 4.41 rotary call assumed. Verify rather than assume.
            if cos.shape[0] > 1 and not torch.equal(cos[0], cos[-1]):
                raise RuntimeError(
                    "HyC-LoRA port: per-sample position_ids differ across the batch. Upstream's "
                    "RoPE kernel takes a single [seq_len, head_dim] cos/sin table and cannot "
                    "represent that. Use uniform positions (standard packed LM / padded batches)."
                )
            cos, sin = cos[0], sin[0]
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        if not checked["dtype"]:
            _check_adapter_dtype(layer, hidden_states.dtype, cfg.layer_type)
            checked["dtype"] = True

        kv_len = hidden_states.shape[1]
        if is_flash:
            # FlashAttention expresses causality with `is_causal`, not with an additive mask.
            # Validate once (the mask is identical every step) and then drop it entirely.
            if not checked["mask"]:
                _validate_flash_mask(attention_mask, kv_len, hidden_states.device)
                checked["mask"] = True
            attention_mask = None
        else:
            if attention_mask is None or attention_mask.dim() != 4:
                raise RuntimeError(
                    "HyC-LoRA port: the fused attention adds `attention_mask` to the pre-softmax "
                    f"scores, so it needs a 4D additive mask; got "
                    f"{None if attention_mask is None else tuple(attention_mask.shape)}. Load the "
                    "model with attn_implementation='eager' -- with sdpa/flash the mask may be None "
                    "and attention would silently become non-causal."
                )
            # transformers >=4.48 allocates the causal mask with target_length =
            # past_seen_tokens + seq_len + 1, i.e. one column wider than the keys. Stock
            # LlamaAttention slices it (`attention_mask[:, :, :, : key_states.shape[-2]]`) before
            # adding it to the scores; upstream HyC-LoRA was written against 4.41 where the mask
            # already arrived square, so we reproduce HF's slice here. Without it the add fails
            # (512 vs 513) -- or worse, would silently broadcast on a different shape combination.
            if attention_mask.shape[-1] != kv_len:
                attention_mask = attention_mask[:, :, :, :kv_len]
            attention_mask = attention_mask.to(hidden_states.dtype)

        sa, mlp = layer.self_attn, layer.mlp
        out = fused.forward(
            hidden_states,
            layer.input_layernorm.weight, None,
            cos, sin,
            *_lora_triplet(sa.q_proj, "q_proj"),
            *_lora_triplet(sa.k_proj, "k_proj"),
            *_lora_triplet(sa.v_proj, "v_proj"),
            *_lora_triplet(sa.o_proj, "o_proj"),
            layer.post_attention_layernorm.weight, None,
            *_lora_triplet(mlp.gate_proj, "gate_proj"),
            *_lora_triplet(mlp.up_proj, "up_proj"),
            *_lora_triplet(mlp.down_proj, "down_proj"),
            attention_mask,
            n_heads, head_dim, n_kv_heads,
        )

        _HYCLORA_COUNTERS["forward"] += 1
        if out.requires_grad:
            out.register_hook(lambda g: _HYCLORA_COUNTERS.__setitem__(
                "backward", _HYCLORA_COUNTERS["backward"] + 1) or g)
        return (out,)

    return hyclora_forward


def apply_hyclora(model, cfg: HyCLoRAConfig, verbose: bool = True) -> int:
    """Swap every Llama/Mistral decoder layer onto HyC-LoRA's fused training path.

    Returns the number of patched layers. Raises rather than silently falling back.
    """
    cfg.validate()
    layer_cls = LAYER_TYPES[cfg.layer_type]

    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)          # peft wrapper
    inner = getattr(base, "model", base)         # LlamaForCausalLM -> LlamaModel
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            f"HyC-LoRA: could not locate decoder layers on {type(model).__name__}. "
            "Only Llama/Mistral-family causal LMs are supported."
        )

    config = inner.config
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_heads)

    patched = 0
    for layer in layers:
        sa, mlp = layer.self_attn, layer.mlp
        for name, mod in (("q_proj", sa.q_proj), ("k_proj", sa.k_proj), ("v_proj", sa.v_proj),
                          ("o_proj", sa.o_proj), ("gate_proj", mlp.gate_proj),
                          ("up_proj", mlp.up_proj), ("down_proj", mlp.down_proj)):
            _lora_triplet(mod, name)
            _check_scaling(mod, name)
            _prepare_base_weight(mod.base_layer, name)

        fused = layer_cls()
        fused.set_hyclora_config(cfg)
        layer._hyclora_fused = fused
        layer.forward = _make_forward(layer, fused, cfg, n_heads, head_dim, n_kv_heads)
        patched += 1

    if patched == 0:
        raise ValueError("HyC-LoRA: 0 decoder layers patched")
    _HYCLORA_COUNTERS["patched_layers"] = patched

    if verbose:
        print("\033[1;34m********** HyC-LoRA Configuration **********\033[0m")
        print(f"\033[1;32m[INFO] layer_type: {cfg.layer_type}\033[0m")
        print(f"\033[1;32m[INFO] q_bit: {cfg.q_bit}\033[0m")
        print(f"\033[1;32m[INFO] softmax_outlier_ratio: {cfg.softmax_outlier_ratio}\033[0m")
        print(f"\033[1;32m[INFO] layernorm_outlier_ratio: {cfg.layernorm_outlier_ratio}\033[0m")
        print(f"\033[1;32m[INFO] iteration_threshold: {cfg.iteration_threshold}\033[0m")
        print(f"\033[1;32m[INFO] patched {patched} decoder layer(s)\033[0m")
        print("\033[1;34m********************************************\033[0m")
    return patched
