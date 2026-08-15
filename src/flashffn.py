"""
FlashFFN: Memory-efficient FFN with activation compression.

Computes SwiGLU FFN exactly (bit-close to standard FFN) in the forward pass,
but stores only compressed (top-K) intermediate activations for the backward pass.

Key optimizations (v2 — fused Triton kernels, no Python tiling loop):
- cuBLAS for all matmuls (via F.linear / torch.mm) — no custom Triton matmuls
- Fused Triton kernels for element-wise SiLU*mul (forward) and SwiGLU backward
- No Python tiling loop: full vectorized computation in a single pass
- torch.topk for top-K compression (well-optimized CUDA implementation)

Forward: 3 cuBLAS + 1 Triton + topK ops = ~8 kernel launches (was ~80 with tiling)
Backward: 8 cuBLAS + 1 Triton + scatter = ~10 kernel launches (was ~120 with tiling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional
import math
import os
import weakref

# Pinned-host staging for `o_h` at `keep='attn'` (route_b.md sections 2.6/2.7).  Kept in its own
# module because it is pure CUDA-stream/buffer plumbing with no autograd in it, and because the
# mechanism is MEMO's (arXiv 2407.12117) rather than ours -- the separation makes that boundary
# visible.  Imported unconditionally: it touches no CUDA API at import time.
import fb_offload as _fb_offload  # noqa: E402
from fb_offload import (  # noqa: E402  -- re-exported so callers use one module
    fb_offload_enable, fb_offload_enabled, fb_offload_stats,
)
# Pinned-host residency for the FROZEN BASE WEIGHTS (WP-E).  Same separation of concerns as
# `fb_offload`: pure stream/buffer plumbing, no autograd, and the mechanism is DeepSpeed
# ZeRO-Offload's / accelerate's rather than ours.  Direction is the opposite one -- weights coming
# IN, H2D only, same bytes every step, never modified.
import fb_wstream as _fb_wstream  # noqa: E402
from fb_wstream import (  # noqa: E402
    fb_wstream_enable, fb_wstream_enabled, fb_wstream_alloc_stream, fb_wstream_lookahead,
    fb_wstream_bwd, fb_wstream_parts, fb_wstream_fault, fb_wstream_stats,
)


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def _silu_mul_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    dig_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    HAS_DIG: tl.constexpr = False,
):
    """Fused SiLU(gate) * up → out.  Element-wise, 1-D grid.

    `HAS_DIG` folds witness W2 over the OUTPUT into the epilogue.  The `tl.store` above is left
    exactly as it was: the digest reads a separately-cast copy of the same register, so with the
    certificate off this kernel is instruction-identical to its pre-certificate form, and with it
    on the stored bytes are unchanged.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    # silu(gate) = gate * sigmoid(gate)
    act = gate * tl.sigmoid(gate)
    result = act * up

    tl.store(out_ptr + offsets, result, mask=mask)
    if HAS_DIG:
        bits = result.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        t = _fb_mix32(bits ^ (offsets.to(tl.uint32) * 0x9E3779B1))
        tl.store(dig_ptr + pid, tl.sum(tl.where(mask, t, 0).to(tl.int64), axis=0))


@triton.jit
def _swiglu_bwd_kernel(
    # Inputs (read)
    grad_hmid_ptr,
    hgate_ptr,
    hup_ptr,
    # Outputs (write)
    grad_hgate_ptr,
    grad_hup_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused backward through h_mid = SiLU(h_gate) * h_up.

    Computes:
      grad_h_up   = grad_h_mid * silu(h_gate)
      grad_h_gate = grad_h_mid * h_up * silu'(h_gate)
    where silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_hmid = tl.load(grad_hmid_ptr + offsets, mask=mask).to(tl.float32)
    hgate = tl.load(hgate_ptr + offsets, mask=mask).to(tl.float32)
    hup = tl.load(hup_ptr + offsets, mask=mask).to(tl.float32)

    # Forward recomputation
    sig = tl.sigmoid(hgate)
    h_act = hgate * sig  # silu(h_gate)

    # Backward through h_mid = h_act * h_up
    grad_h_act = grad_hmid * hup
    grad_h_up = grad_hmid * h_act

    # Backward through silu: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    dsilu = sig * (1.0 + hgate * (1.0 - sig))
    grad_h_gate = grad_h_act * dsilu

    tl.store(grad_hgate_ptr + offsets, grad_h_gate, mask=mask)
    tl.store(grad_hup_ptr + offsets, grad_h_up, mask=mask)


@triton.jit
def _swiglu_bwd_glu_kernel(
    # Inputs (read)
    grad_hmid_ptr,
    hgate_ptr,
    hup_ptr,
    # Outputs (write) -- ANY of these may alias ANY of the inputs above
    grad_hgate_ptr,
    grad_hup_ptr,
    hmid_ptr,
    dig_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    HAS_HMID: tl.constexpr = False,
    HAS_DIG: tl.constexpr = False,
):
    """`_swiglu_bwd_kernel` and `_silu_mul_fwd_kernel` in ONE pass, with caller-chosen destinations.

    WHY THIS EXISTS.  The block's backward needs three [N, I] tensors out of three [N, I] tensors:
    `grad_h_gate`, `grad_h_up` and (for the down projection's adapter/weight gradients) a
    rematerialised `h_mid`, from `h_gate`, `h_up` and `grad_h_mid`.  Computing them with the two
    separate kernels makes all six coexist -- and `probe_highwater.py` measures that instant as
    the model's high-water mark (CONTEXT.md section 33.5).  Fusing them lets every output be written
    over an input that has already been read, so the window costs three tensors instead of six.

    SAFETY OF THE ALIASING.  Program `pid` owns exactly the half-open index range
    `[pid*BLOCK_SIZE, (pid+1)*BLOCK_SIZE)` and no other program touches it, so the only ordering
    that matters is within one program -- and there every `tl.store` below is data-dependent on
    every `tl.load` above it.  A store can therefore never clobber a value another program still
    has to read.  This is what makes the caller's `out=in` choice sound rather than merely lucky.

    ARITHMETIC.  Every expression is transcribed from the two kernels it replaces, in their order:
    `sig`/`h_act`/`grad_h_act`/`grad_h_up`/`dsilu`/`grad_h_gate` from `_swiglu_bwd_kernel`, and
    `result = act * up` from `_silu_mul_fwd_kernel` -- where that kernel's `act = gate *
    tl.sigmoid(gate)` is the same expression as this one's `h_act`, so `h_mid` comes out bitwise
    equal to the forward's and gate B3 keeps its bitwise agreement across `keep` levels.

    `HAS_DIG` folds witness W2 over `h_mid` in exactly the convention `_silu_mul_fwd_kernel` uses
    -- same `BLOCK_SIZE`, same grid, same GLOBAL `offsets` in the mix -- because the certificate's
    digest is index-keyed and a different tiling would make the forward and the recompute disagree
    on correct code (CONTEXT.md section 33.10).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_hmid = tl.load(grad_hmid_ptr + offsets, mask=mask).to(tl.float32)
    hgate = tl.load(hgate_ptr + offsets, mask=mask).to(tl.float32)
    hup = tl.load(hup_ptr + offsets, mask=mask).to(tl.float32)

    sig = tl.sigmoid(hgate)
    h_act = hgate * sig  # silu(h_gate)

    grad_h_act = grad_hmid * hup
    grad_h_up = grad_hmid * h_act

    dsilu = sig * (1.0 + hgate * (1.0 - sig))
    grad_h_gate = grad_h_act * dsilu

    # Every load above has happened; from here the buffers may be overwritten.
    if HAS_HMID:
        result = h_act * hup
        tl.store(hmid_ptr + offsets, result, mask=mask)
        if HAS_DIG:
            bits = result.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
            t = _fb_mix32(bits ^ (offsets.to(tl.uint32) * 0x9E3779B1))
            tl.store(dig_ptr + pid, tl.sum(tl.where(mask, t, 0).to(tl.int64), axis=0))

    tl.store(grad_hgate_ptr + offsets, grad_h_gate, mask=mask)
    tl.store(grad_hup_ptr + offsets, grad_h_up, mask=mask)


# =============================================================================
# PyTorch Wrappers for Triton Kernels
# =============================================================================

def triton_silu_mul(h_gate: torch.Tensor, h_up: torch.Tensor, digest=False):
    """Fused SiLU(h_gate) * h_up using Triton. Replaces F.silu + mul.

    `digest=True` also returns witness W2 over the output (and is the only caller that ever passes
    it; every other call site keeps the original single-return signature)."""
    out = torch.empty_like(h_gate)
    n_elements = h_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused = digest and _FB_CERT["fused"] and out.element_size() == 2
    dig_buf = torch.empty(grid, dtype=torch.int64, device=h_gate.device) if fused else None
    _silu_mul_fwd_kernel[grid](h_gate, h_up, out, dig_buf, n_elements, BLOCK_SIZE,
                               HAS_DIG=fused)
    if not digest:
        return out
    return out, (dig_buf.sum() if fused else fb_digest(out))


def triton_swiglu_backward(
    grad_h_mid: torch.Tensor,
    h_gate: torch.Tensor,
    h_up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused backward through SwiGLU using Triton."""
    grad_h_gate = torch.empty_like(h_gate)
    grad_h_up = torch.empty_like(h_up)
    n_elements = h_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_bwd_kernel[grid](
        grad_h_mid, h_gate, h_up,
        grad_h_gate, grad_h_up,
        n_elements, BLOCK_SIZE,
    )
    return grad_h_gate, grad_h_up


# `fb_inplace_glu_enable(False)` restores the two-kernel path exactly.  It exists so the in-place
# restructure can be shown to change nothing but the destination buffers: same gate suite, same
# gradients, bitwise, with the only difference in the memory number.
_FB_INPLACE_GLU = {"on": True}


def fb_inplace_glu_enable(on=True):
    """Toggle the fused GLU-backward + rematerialise path. Returns the previous setting."""
    prev = _FB_INPLACE_GLU["on"]
    _FB_INPLACE_GLU["on"] = bool(on)
    return prev


def fb_inplace_glu_enabled():
    return _FB_INPLACE_GLU["on"]


def triton_swiglu_backward_glu(grad_h_mid, h_gate, h_up, want_h_mid, inplace, digest=False):
    """SwiGLU backward and (optionally) the `h_mid` rematerialisation, in one pass.

    `inplace=True` writes `grad_h_gate` over `h_gate`, `grad_h_up` over `h_up` and `h_mid` over
    `grad_h_mid`, so the call allocates nothing at all.  The caller must have established that the
    three inputs are private to this backward -- they are, at every `keep` level except `full`,
    where `h_gate`/`h_up` come out of `save_for_backward` and must not be mutated.

    Returns `(grad_h_gate, grad_h_up, h_mid, w2)`; `h_mid` is None when `want_h_mid` is False and
    `w2` is None unless `digest` is set and the witness could be fused into this kernel.
    """
    n_elements = h_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    if inplace:
        grad_h_gate, grad_h_up = h_gate, h_up
        h_mid = grad_h_mid if want_h_mid else None
    else:
        grad_h_gate = torch.empty_like(h_gate)
        grad_h_up = torch.empty_like(h_up)
        h_mid = torch.empty_like(h_gate) if want_h_mid else None
    fused_dig = bool(digest and want_h_mid and _FB_CERT["fused"] and h_gate.element_size() == 2)
    dig_buf = torch.empty(grid, dtype=torch.int64, device=h_gate.device) if fused_dig else None
    _swiglu_bwd_glu_kernel[grid](
        grad_h_mid, h_gate, h_up,
        grad_h_gate, grad_h_up, h_mid if want_h_mid else grad_h_gate, dig_buf,
        n_elements, BLOCK_SIZE,
        HAS_HMID=want_h_mid, HAS_DIG=fused_dig,
    )
    w2 = None
    if digest and want_h_mid:
        # The unfused witness has to be taken AFTER the kernel, from the tensor it just wrote --
        # `fb_digest` over `h_mid` -- which is what the standalone certificate path already does.
        w2 = dig_buf.sum() if fused_dig else fb_digest(h_mid)
    return grad_h_gate, grad_h_up, h_mid, w2


# =============================================================================
# Core FlashFFN Autograd Function
# =============================================================================

class FlashFFNFunction(torch.autograd.Function):
    """
    Custom autograd function for FlashFFN with activation compression.

    Dual-mode forward/backward optimized for both training regimes:

    **Recompute mode** (weights need gradients — full fine-tuning):
      Forward:  cuBLAS matmuls + fused Triton SiLU*mul + top-K compression
      Backward: sparse h_mid for grad_w_down (approximate), recompute
                h_gate/h_up for exact grad_x/grad_w_gate/grad_w_up
      Saves: x + compressed top-K  (~22 MB/layer at TinyLlama scale)

    **Activations mode** (weights frozen — LoRA / adapter training):
      Forward:  cuBLAS matmuls + fused Triton SiLU*mul (NO top-K overhead)
      Backward: direct backward using saved h_gate/h_up (no recomputation)
      Saves: h_gate + h_up  (~46 MB/layer — 33% less than standard autograd
             which saves h_gate + h_act + h_up = ~69 MB/layer)

    Mode is selected automatically based on weight requires_grad status.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        k_fraction: float = 0.3,
    ) -> torch.Tensor:
        orig_shape = x.shape
        batch_seq = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        hidden_dim = x.shape[-1]
        intermediate_dim = w_gate.shape[0]

        x_2d = x.reshape(batch_seq, hidden_dim)

        # --- Full forward: cuBLAS matmuls + fused Triton SiLU*mul ---
        h_gate = F.linear(x_2d, w_gate)              # (N, D) cuBLAS
        h_up = F.linear(x_2d, w_up)                  # (N, D) cuBLAS
        h_mid = triton_silu_mul(h_gate, h_up)         # (N, D) fused Triton

        y = F.linear(h_mid, w_down)                   # (N, H) cuBLAS

        # --- Choose save mode based on weight requires_grad ---
        weights_need_grad = (
            w_gate.requires_grad or w_up.requires_grad or w_down.requires_grad
        )

        if weights_need_grad:
            # RECOMPUTE MODE: save x + compressed top-K for backward
            del h_gate, h_up
            k = max(1, int(intermediate_dim * k_fraction))
            _, top_indices = torch.topk(h_mid.abs(), k, dim=-1, sorted=False)
            top_values = torch.gather(h_mid, dim=-1, index=top_indices)
            all_indices = top_indices.to(torch.int16)
            del h_mid, top_indices
            ctx.save_for_backward(x, w_gate, w_up, w_down, top_values, all_indices)
            ctx.save_mode = 'recompute'
            ctx.k = k
            ctx.intermediate_dim = intermediate_dim
        else:
            # ACTIVATIONS MODE: save h_gate/h_up directly (fast backward)
            del h_mid
            ctx.save_for_backward(h_gate, h_up, w_gate, w_up, w_down)
            ctx.save_mode = 'activations'

        ctx.orig_shape = orig_shape

        # Reshape output
        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        orig_shape = ctx.orig_shape
        batch_seq = orig_shape[0] * orig_shape[1] if len(orig_shape) == 3 else orig_shape[0]
        hidden_dim = orig_shape[-1]

        grad_output_2d = grad_output.reshape(batch_seq, hidden_dim)

        if ctx.save_mode == 'activations':
            # --- ACTIVATIONS MODE: fast backward, no recomputation ---
            h_gate, h_up, w_gate, w_up, w_down = ctx.saved_tensors

            # grad through output projection
            grad_h_mid = grad_output_2d @ w_down               # (N, D) cuBLAS

            # Fused backward through SiLU-mul using saved h_gate, h_up
            grad_h_gate, grad_h_up = triton_swiglu_backward(grad_h_mid, h_gate, h_up)
            del h_gate, h_up, grad_h_mid

            # Input gradient only (weights are frozen)
            grad_x = grad_h_gate @ w_gate + grad_h_up @ w_up   # (N, H)
            del grad_h_gate, grad_h_up

            if len(orig_shape) == 3:
                grad_x = grad_x.view(orig_shape)

            return grad_x, None, None, None, None

        else:
            # --- RECOMPUTE MODE: full gradient computation ---
            x, w_gate, w_up, w_down, values, indices = ctx.saved_tensors
            k = ctx.k
            intermediate_dim = ctx.intermediate_dim

            x_2d = x.reshape(batch_seq, hidden_dim)

            # Check which inputs need gradients
            need_x_grad = ctx.needs_input_grad[0]
            need_wgate_grad = ctx.needs_input_grad[1]
            need_wup_grad = ctx.needs_input_grad[2]
            need_wdown_grad = ctx.needs_input_grad[3]

            # grad_w_down via sparse h_mid reconstruction
            if need_wdown_grad:
                h_mid_sparse = torch.zeros(
                    batch_seq, intermediate_dim, device=x.device, dtype=x.dtype
                )
                h_mid_sparse.scatter_(dim=-1, index=indices.long(), src=values)
                grad_w_down = grad_output_2d.t() @ h_mid_sparse
                del h_mid_sparse
            else:
                grad_w_down = None

            # Exact gradients via forward recomputation
            if need_x_grad or need_wgate_grad or need_wup_grad:
                h_gate = F.linear(x_2d, w_gate)
                h_up = F.linear(x_2d, w_up)
                grad_h_mid = grad_output_2d @ w_down

                grad_h_gate, grad_h_up = triton_swiglu_backward(
                    grad_h_mid, h_gate, h_up
                )
                del h_gate, h_up, grad_h_mid

                grad_w_gate = grad_h_gate.t() @ x_2d if need_wgate_grad else None
                grad_w_up = grad_h_up.t() @ x_2d if need_wup_grad else None

                if need_x_grad:
                    grad_x = grad_h_gate @ w_gate + grad_h_up @ w_up
                else:
                    grad_x = None
                del grad_h_gate, grad_h_up
            else:
                grad_x = None
                grad_w_gate = None
                grad_w_up = None

            if need_x_grad and len(orig_shape) == 3:
                grad_x = grad_x.view(orig_shape)

            return grad_x, grad_w_gate, grad_w_up, grad_w_down, None


# =============================================================================
# Module Wrappers
# =============================================================================

class FlashFFN(nn.Module):
    """
    Flash Feed-Forward Network with activation compression.

    Implements SwiGLU FFN with memory-efficient activation storage:
    y = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    During forward pass, only top-K intermediate activations are stored
    for the backward pass, dramatically reducing memory footprint.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        k_fraction: float = 0.3,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.k_fraction = k_fraction

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FlashFFNFunction.apply(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            self.k_fraction,
        )


class StandardFFN(nn.Module):
    """Standard SwiGLU FFN for comparison."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_gate = self.gate_proj(x)
        h_up = self.up_proj(x)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up
        y = self.down_proj(h_mid)
        return y


class FlashFFNNoInputSaveFunction(torch.autograd.Function):
    """
    FlashFFN variant that doesn't save input x for backward.

    This is for use with model-level gradient checkpointing where x will be
    recomputed from the previous layer. Achieves maximum memory savings.

    WARNING: Only use this with gradient checkpointing enabled at model level.
    Without recomputing x, grad_w_gate and grad_w_up will be zeros!
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        k_fraction: float = 0.3,
    ) -> torch.Tensor:
        orig_shape = x.shape
        batch_seq = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        hidden_dim = x.shape[-1]
        intermediate_dim = w_gate.shape[0]

        x_2d = x.reshape(batch_seq, hidden_dim)

        # Full forward: cuBLAS + fused Triton (no tiling)
        h_gate = F.linear(x_2d, w_gate)
        h_up = F.linear(x_2d, w_up)
        h_mid = triton_silu_mul(h_gate, h_up)
        del h_gate, h_up

        y = F.linear(h_mid, w_down)

        # Compress
        k = max(1, int(intermediate_dim * k_fraction))
        _, top_indices = torch.topk(h_mid.abs(), k, dim=-1, sorted=False)
        top_values = torch.gather(h_mid, dim=-1, index=top_indices)
        all_indices = top_indices.to(torch.int16)
        del h_mid, top_indices

        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)

        # Save ONLY compressed representation + weights (NOT x!)
        ctx.save_for_backward(w_gate, w_up, w_down, top_values, all_indices)
        ctx.k = k
        ctx.intermediate_dim = intermediate_dim
        ctx.orig_shape = orig_shape

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using ONLY compressed activations.

        NOTE: This only computes grad_w_down accurately.
        grad_x, grad_w_gate, grad_w_up are zeros (rely on model-level checkpointing).
        """
        w_gate, w_up, w_down, values, indices = ctx.saved_tensors
        k = ctx.k
        intermediate_dim = ctx.intermediate_dim
        orig_shape = ctx.orig_shape

        batch_seq = orig_shape[0] * orig_shape[1] if len(orig_shape) == 3 else orig_shape[0]
        hidden_dim = orig_shape[-1]

        grad_output_2d = grad_output.view(batch_seq, hidden_dim)

        # Reconstruct sparse h_mid
        h_mid_sparse = torch.zeros(batch_seq, intermediate_dim, device=values.device, dtype=values.dtype)
        h_mid_sparse.scatter_(dim=-1, index=indices.long(), src=values)

        # grad_w_down = grad_output.T @ h_mid_sparse (accurate using sparse h_mid)
        grad_w_down = grad_output_2d.t() @ h_mid_sparse

        # Return zeros for other gradients (model-level checkpointing will recompute)
        grad_w_gate = torch.zeros_like(w_gate)
        grad_w_up = torch.zeros_like(w_up)
        grad_x = torch.zeros(orig_shape, device=grad_output.device, dtype=grad_output.dtype)

        return grad_x, grad_w_gate, grad_w_up, grad_w_down, None


class FlashFFNUltraLight(nn.Module):
    """
    Ultra memory-efficient FlashFFN that only saves compressed h_mid.

    Must be used with model-level gradient checkpointing!
    Achieves ~90% activation memory reduction.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        k_fraction: float = 0.3,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.k_fraction = k_fraction

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FlashFFNNoInputSaveFunction.apply(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            self.k_fraction,
        )


# =============================================================================
# dReLU FlashFFN  (TurboSparse-Mistral / "Bamboo" architecture)
# =============================================================================
#
# TurboSparse models (arXiv 2406.05955) replace SwiGLU's SiLU gate with dReLU,
# which applies ReLU to BOTH the gate and up projections before the product:
#
#     h_mid = ReLU(x @ W_gate.T) * ReLU(x @ W_up.T)
#
# In the released checkpoint (PowerInfer/TurboSparse-Mistral-Instruct, the
# "Bamboo" custom architecture), h_mid is additionally multiplied by a hard
# 0/1 sparsity mask produced by a small per-layer "predictor" MLP that reads
# the *pre-norm residual* (before_norm) of the FFN block:
#
#     m    = sigmoid( W_p2 @ ReLU(W_p1 @ before_norm) )      # predictor MLP
#     mask = round(m)                                        # straight-through
#     h_mid = mask * h_mid
#     y     = h_mid @ W_down.T
#
# The predictor is treated as a FROZEN sparsity oracle: its hard mask is
# computed WITHOUT building an autograd graph and is RECOMPUTED in the backward
# pass (FlashFFN style) rather than stored, so no [N, intermediate] mask tensor
# ever persists across forward->backward.  This keeps the FlashFFN memory
# advantage intact on the dReLU architecture.
#
# dReLU backward (h_mid = ReLU(g) * ReLU(u)):
#     grad_g = grad_h_mid * ReLU(u) * 1[g > 0]
#     grad_u = grad_h_mid * ReLU(g) * 1[u > 0]
# (ReLU' is a 0/1 step; much cheaper than SiLU'.)


@triton.jit
def _drelu_mul_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU(gate) * ReLU(up) -> out.  Element-wise, 1-D grid."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    rg = tl.where(gate > 0.0, gate, 0.0)
    ru = tl.where(up > 0.0, up, 0.0)

    tl.store(out_ptr + offsets, rg * ru, mask=mask)


@triton.jit
def _drelu_bwd_kernel(
    grad_hmid_ptr,
    gate_ptr,
    up_ptr,
    grad_gate_ptr,
    grad_up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused backward through h_mid = ReLU(h_gate) * ReLU(h_up).

    Computes:
      grad_h_gate = grad_h_mid * ReLU(h_up)   * 1[h_gate > 0]
      grad_h_up   = grad_h_mid * ReLU(h_gate) * 1[h_up   > 0]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_hmid = tl.load(grad_hmid_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    rg = tl.where(gate > 0.0, gate, 0.0)
    ru = tl.where(up > 0.0, up, 0.0)
    step_g = tl.where(gate > 0.0, 1.0, 0.0)
    step_u = tl.where(up > 0.0, 1.0, 0.0)

    grad_h_gate = grad_hmid * ru * step_g
    grad_h_up = grad_hmid * rg * step_u

    tl.store(grad_gate_ptr + offsets, grad_h_gate, mask=mask)
    tl.store(grad_up_ptr + offsets, grad_h_up, mask=mask)


def triton_drelu_mul(h_gate: torch.Tensor, h_up: torch.Tensor) -> torch.Tensor:
    """Fused ReLU(h_gate) * ReLU(h_up) using Triton. Replaces F.relu*F.relu*mul."""
    out = torch.empty_like(h_gate)
    n_elements = h_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _drelu_mul_fwd_kernel[grid](h_gate, h_up, out, n_elements, BLOCK_SIZE)
    return out


def triton_drelu_backward(
    grad_h_mid: torch.Tensor,
    h_gate: torch.Tensor,
    h_up: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused backward through dReLU using Triton."""
    grad_h_gate = torch.empty_like(h_gate)
    grad_h_up = torch.empty_like(h_up)
    n_elements = h_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _drelu_bwd_kernel[grid](
        grad_h_mid, h_gate, h_up,
        grad_h_gate, grad_h_up,
        n_elements, BLOCK_SIZE,
    )
    return grad_h_gate, grad_h_up


def compute_bamboo_mask(
    before_norm_2d: torch.Tensor,
    pred_fc1_w: torch.Tensor,
    pred_fc2_w: torch.Tensor,
) -> torch.Tensor:
    """Recompute the TurboSparse/Bamboo predictor hard mask (0/1).

    mask = round(sigmoid(pred_fc2_w @ ReLU(pred_fc1_w @ before_norm)))

    Matches the predictor MLP in modeling_bamboo.py EXACTLY: fc1 -> ReLU ->
    fc2 -> sigmoid, both Linear with bias=False, all computed in the input's
    native dtype (the real model runs the predictor in bf16). Computing in the
    same dtype reproduces the hard 0/1 mask bit-for-bit, including neurons whose
    predictor logit sits near the sigmoid=0.5 rounding boundary.
    """
    h = F.relu(F.linear(before_norm_2d, pred_fc1_w))
    m = torch.sigmoid(F.linear(h, pred_fc2_w))
    return torch.round(m)


class FlashDReLUFFNFunction(torch.autograd.Function):
    """
    FlashFFN autograd for the dReLU FFN (TurboSparse-Mistral / Bamboo).

    Forward (exact, matches StandardDReLUFFN bit-for-bit up to fp accumulation):
        h_mid = ReLU(x @ W_gate.T) * ReLU(x @ W_up.T)
        if predictor provided:  h_mid *= round(sigmoid(predictor(before_norm)))
        y     = h_mid @ W_down.T

    Always uses recompute-mode backward: stores only x (+ before_norm + tiny
    predictor weights when masked) and the top-K compressed h_mid.  h_gate /
    h_up / mask are recomputed in backward.  grad_w_down uses the sparse top-K
    reconstruction of h_mid (approximate, but ~exact here because the hard mask
    makes h_mid >=90% zeros, so the top-K keeps every non-zero); grad_x /
    grad_w_gate / grad_w_up are exact.

    Inputs (5 required + 3 optional for the predictor mask):
        x, w_gate, w_up, w_down, k_fraction,
        before_norm=None, pred_fc1_w=None, pred_fc2_w=None
    Pass the three predictor tensors as None for a plain (maskless) dReLU FFN.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        k_fraction: float = 0.3,
        before_norm: Optional[torch.Tensor] = None,
        pred_fc1_w: Optional[torch.Tensor] = None,
        pred_fc2_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        batch_seq = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        hidden_dim = x.shape[-1]
        intermediate_dim = w_gate.shape[0]

        x_2d = x.reshape(batch_seq, hidden_dim)

        h_gate = F.linear(x_2d, w_gate)               # (N, D) cuBLAS
        h_up = F.linear(x_2d, w_up)                   # (N, D) cuBLAS
        h_mid = triton_drelu_mul(h_gate, h_up)        # (N, D) fused Triton
        del h_gate, h_up

        use_mask = (
            before_norm is not None
            and pred_fc1_w is not None
            and pred_fc2_w is not None
        )
        if use_mask:
            bn_2d = before_norm.reshape(batch_seq, hidden_dim)
            with torch.no_grad():
                mask = compute_bamboo_mask(bn_2d, pred_fc1_w, pred_fc2_w).to(h_mid.dtype)
            h_mid = h_mid * mask
            del mask

        y = F.linear(h_mid, w_down)                   # (N, H) cuBLAS

        # RECOMPUTE MODE: save x (+ predictor recompute inputs) + compressed top-K
        k = max(1, int(intermediate_dim * k_fraction))
        # int16 indices require intermediate_dim < 32768
        if intermediate_dim >= 32768:
            raise ValueError(
                f"FlashDReLUFFN int16 indices require intermediate_dim < 32768, "
                f"got {intermediate_dim}"
            )
        _, top_indices = torch.topk(h_mid.abs(), k, dim=-1, sorted=False)
        top_values = torch.gather(h_mid, dim=-1, index=top_indices)
        all_indices = top_indices.to(torch.int16)
        del h_mid, top_indices

        if use_mask:
            ctx.save_for_backward(
                x, w_gate, w_up, w_down, top_values, all_indices,
                before_norm, pred_fc1_w, pred_fc2_w,
            )
        else:
            ctx.save_for_backward(x, w_gate, w_up, w_down, top_values, all_indices)
        ctx.use_mask = use_mask
        ctx.k = k
        ctx.intermediate_dim = intermediate_dim
        ctx.orig_shape = orig_shape

        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        orig_shape = ctx.orig_shape
        batch_seq = orig_shape[0] * orig_shape[1] if len(orig_shape) == 3 else orig_shape[0]
        hidden_dim = orig_shape[-1]

        grad_output_2d = grad_output.reshape(batch_seq, hidden_dim)

        if ctx.use_mask:
            (x, w_gate, w_up, w_down, values, indices,
             before_norm, pred_fc1_w, pred_fc2_w) = ctx.saved_tensors
        else:
            x, w_gate, w_up, w_down, values, indices = ctx.saved_tensors

        intermediate_dim = ctx.intermediate_dim
        x_2d = x.reshape(batch_seq, hidden_dim)

        need_x_grad = ctx.needs_input_grad[0]
        need_wgate_grad = ctx.needs_input_grad[1]
        need_wup_grad = ctx.needs_input_grad[2]
        need_wdown_grad = ctx.needs_input_grad[3]

        # grad_w_down via sparse h_mid reconstruction (h_mid already mask-applied)
        if need_wdown_grad:
            h_mid_sparse = torch.zeros(
                batch_seq, intermediate_dim, device=x.device, dtype=x.dtype
            )
            h_mid_sparse.scatter_(dim=-1, index=indices.long(), src=values)
            grad_w_down = grad_output_2d.t() @ h_mid_sparse
            del h_mid_sparse
        else:
            grad_w_down = None

        if need_x_grad or need_wgate_grad or need_wup_grad:
            h_gate = F.linear(x_2d, w_gate)
            h_up = F.linear(x_2d, w_up)
            grad_h_mid = grad_output_2d @ w_down

            # Re-apply the (recomputed) hard sparsity mask to grad_h_mid: it
            # gates which neurons contributed to the forward output.
            if ctx.use_mask:
                bn_2d = before_norm.reshape(batch_seq, hidden_dim)
                with torch.no_grad():
                    mask = compute_bamboo_mask(bn_2d, pred_fc1_w, pred_fc2_w).to(grad_h_mid.dtype)
                grad_h_mid = grad_h_mid * mask
                del mask

            grad_h_gate, grad_h_up = triton_drelu_backward(grad_h_mid, h_gate, h_up)
            del h_gate, h_up, grad_h_mid

            grad_w_gate = grad_h_gate.t() @ x_2d if need_wgate_grad else None
            grad_w_up = grad_h_up.t() @ x_2d if need_wup_grad else None

            if need_x_grad:
                grad_x = grad_h_gate @ w_gate + grad_h_up @ w_up
            else:
                grad_x = None
            del grad_h_gate, grad_h_up
        else:
            grad_x = None
            grad_w_gate = None
            grad_w_up = None

        if need_x_grad and len(orig_shape) == 3:
            grad_x = grad_x.view(orig_shape)

        # 8 inputs -> 8 grads (k_fraction + 3 predictor tensors get None)
        return grad_x, grad_w_gate, grad_w_up, grad_w_down, None, None, None, None


class StandardDReLUFFN(nn.Module):
    """Reference (non-compressed) dReLU FFN, optionally with a Bamboo predictor.

    Mirrors BambooMLP's math for verification:
        h = ReLU(gate(x)) * ReLU(up(x))
        if mask given:  h = mask * h
        y = down(h)
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.relu(self.gate_proj(x)) * F.relu(self.up_proj(x))
        if mask is not None:
            h = h * mask
        return self.down_proj(h)


class FlashDReLUFFN(nn.Module):
    """Drop-in dReLU FlashFFN module (no predictor mask).

    For the masked Bamboo variant the model integration passes the predictor
    weights and the pre-norm residual directly to FlashDReLUFFNFunction.apply;
    this module is the maskless convenience wrapper used by tests/benchmarks.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        k_fraction: float = 0.3,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.k_fraction = k_fraction
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FlashDReLUFFNFunction.apply(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            self.k_fraction,
        )


# =============================================================================
# FlashFFN v3 — LoRA-factored, quantized GLU-cache (prototype, additive only)
# =============================================================================
#
# Setting: SwiGLU FFN where each projection P in {gate, up, down} may carry a
# LoRA adapter, computed in FACTORED form (no merged weights anywhere):
#
#     out_P = x @ W_P.T + s_P * (x @ A_P.T) @ B_P.T
#
# with W_P frozen (or trainable for full-FT, then no adapters), A_P [r, in],
# B_P [out, r] trainable, s_P a python float (lora_alpha / r).
#
# Saved for backward (cache_mode):
#   'int4' / 'int8' : x + Q(h_gate) + Q(h_up); Q = symmetric absmax per-group
#                     quantization (group_size along last dim), fp16 scales.
#                     INT4 is REALLY packed two-per-byte into uint8.
#   'bf16'          : x + h_gate + h_up (uncompressed reference mode)
#   'recompute'     : x only; h_gate/h_up recomputed in backward including the
#                     LoRA bypass (bitwise exact: same kernels, same inputs).
# All modes additionally save the thin [N, r] mats xA_g = x@A_g.T, xA_u and
# hA_d = h_mid@A_d.T (tiny; avoid extra thin recompute) when adapters present.
#
# Backward (reassociated; verified numerically in src/verify_flashffn_v3.py):
#   grad_h_mid = grad_y @ W_d + s_d*(grad_y @ B_d) @ A_d            (exact)
#   (grad_h_gate, grad_h_up) = triton_swiglu_backward(grad_h_mid, hg^, hu^)
#   grad_x   = grad_h_gate@W_g + s_g*(grad_h_gate@B_g)@A_g
#            + grad_h_up @W_u + s_u*(grad_h_up @B_u)@A_u
#   grad_B_g = s_g * grad_h_gate.T @ xA_g                  [D_int, r]
#   grad_A_g = s_g * (grad_h_gate @ B_g).T @ x             [r, D_hid]
#   grad_B_d = s_d * grad_y.T @ hA_d                       [D_hid, r]
#   grad_A_d = s_d * (grad_y @ B_d).T @ h_mid^             [r, D_int]
#   full-FT: grad_W_g = grad_h_gate.T@x, grad_W_u = grad_h_up.T@x,
#            grad_W_d = grad_y.T @ h_mid^   (dense, unavoidable)
# INVARIANT: when weights are frozen no [D_out, D_in]-shaped tensor is created
# in backward (dense weight grads exist only inside needs_input_grad guards;
# asserted at the end of backward).
#
# Quantization is plain PyTorch (no new Triton in this prototype phase).

import warnings

# Honesty instrumentation: proves the v3 code path actually ran.
# triton_quant_fwd / triton_dequant_bwd count executions of the FUSED v3
# kernels (K1/K2) specifically, so an eager fallback can never masquerade as
# a Triton benchmark number.
_V3_COUNTERS = {"forward": 0, "backward": 0,
                "triton_quant_fwd": 0, "triton_dequant_bwd": 0,
                "adapter_forward": 0, "adapter_backward": 0}

# GEMM-skeleton timing mode: quant/dequant become no-ops (cache stores raw
# bf16). ONLY for the V4 kernel-phase-headroom measurement — never for
# correctness or memory numbers.
_V3_SKELETON = {"enabled": False}


def v3_reset_counters():
    for k in _V3_COUNTERS:
        _V3_COUNTERS[k] = 0


def v3_quantize_group_absmax(t: torch.Tensor, bits: int, group_size: int = 64):
    """Symmetric absmax per-group quantization along the last dim.

    t: [N, D] floating tensor, D divisible by group_size.
    Returns (payload, scales):
      bits=4: payload uint8 [N, D/2] (two offset-8 nibble codes per byte,
              even column in low nibble, odd column in high nibble)
      bits=8: payload int8 [N, D]
      scales: fp16 [N, D/group_size]
    Quantization divides by the fp16-ROUNDED scale (the same scale dequant
    will use), so the codec is self-consistent.
    In skeleton mode returns (t, None) — a no-op for timing only.
    """
    if _V3_SKELETON["enabled"]:
        return t, None
    assert bits in (4, 8), f"unsupported bits={bits}"
    N, D = t.shape
    assert D % group_size == 0, f"D={D} not divisible by group_size={group_size}"
    G = D // group_size
    qmax = float(2 ** (bits - 1) - 1)
    tg = t.reshape(N, G, group_size).float()
    absmax = tg.abs().amax(dim=-1, keepdim=True)
    scales = (absmax / qmax).to(torch.float16)            # stored metadata
    safe = scales.float()
    safe = torch.where(safe > 0, safe, torch.ones_like(safe))
    q = torch.round(tg / safe).clamp_(-qmax, qmax)        # [N, G, group]
    if bits == 4:
        q8 = (q.reshape(N, D) + 8.0).to(torch.uint8)      # codes in [1, 15]
        payload = torch.bitwise_or(
            q8[:, 0::2], torch.bitwise_left_shift(q8[:, 1::2], 4)
        )                                                  # uint8 [N, D/2]
    else:
        payload = q.reshape(N, D).to(torch.int8)           # int8 [N, D]
    return payload, scales.reshape(N, G)


def v3_dequantize_group_absmax(payload, scales, bits, group_size=64,
                               dtype=torch.bfloat16):
    """Inverse of v3_quantize_group_absmax. Returns [N, D] in `dtype`.

    If scales is None (skeleton mode) the payload is returned unchanged.
    """
    if scales is None:
        return payload
    if bits == 4:
        N, half = payload.shape
        D = half * 2
        lo = torch.bitwise_and(payload, 0x0F).to(torch.float32) - 8.0
        hi = torch.bitwise_right_shift(payload, 4).to(torch.float32) - 8.0
        q = torch.stack((lo, hi), dim=-1).reshape(N, D)
    else:
        N, D = payload.shape
        q = payload.to(torch.float32)
    G = D // group_size
    deq = q.reshape(N, G, group_size) * scales.reshape(N, G, 1).to(torch.float32)
    return deq.reshape(N, D).to(dtype)


# -----------------------------------------------------------------------------
# v3 fused Triton kernels (K1/K2)
#
# K1 _v3_silu_mul_quant_fwd_kernel: one pass over h_gate/h_up that
#     (a) computes h_mid = SiLU(h_gate) * h_up with the EXACT op order of
#         _silu_mul_fwd_kernel (bitwise-equal h_mid), and
#     (b) quantizes h_gate and h_up with the EXACT math of
#         v3_quantize_group_absmax (per-group absmax, scale = absmax/qmax
#         rounded to fp16 BEFORE use, round-to-nearest-even codes, int4
#         packed two-per-byte: even column in low nibble, odd in high).
# K2 _v3_dequant_swiglu_bwd_kernel: one pass that dequantizes (codes * fp16
#     scale in fp32, rounded through bf16 exactly like
#     v3_dequantize_group_absmax) and applies the _swiglu_bwd_kernel math;
#     optionally also writes h_mid^ = SiLU(hg^)*hu^ (needed for the
#     down-projection weight/adapter grads).
#
# Bit-exactness notes (verified in src/verify_v3_kernels.py gate G1/G2):
#   - Triton's `/` is NOT IEEE-rounded (div.full.f32); tl.math.div_rn is used
#     everywhere a division must match the PyTorch reference.
#   - libdevice.rint == torch.round (round-half-to-even).
#   - .to(tl.float16) / .to(tl.bfloat16) use round-to-nearest-even, matching
#     torch casts.
# Constraints: D % group_size == 0 (asserted, same as the eager codec),
# group_size a power of two, bf16 inputs. Ragged last groups are NOT
# supported — the wrappers assert cleanly, mirroring v3_quantize_group_absmax.
# -----------------------------------------------------------------------------

from triton.language.extra import libdevice as _v3_libdevice  # noqa: E402

# Launch configuration for the fused kernels (overridable for tuning).
# Tuned on A40 @ N=8192, D=5632 (sweep over block in {256,512,1024,2048} x
# warps in {2,4,8}); both kernels are within ~10% of the bandwidth floor.
_V3_TRITON_CFG = {"fwd_block": 2048, "fwd_warps": 8,
                  "bwd_block": 2048, "bwd_warps": 2}


@triton.jit
def _v3_quant_store(vals, q_ptr, s_ptr, q_row, s_row, pid_d, D, G,
                    BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                    BLOCK_D: tl.constexpr):
    """Quantize one [BLOCK_D] fp32 block (whole groups only) and store
    payload + fp16 scales. Math identical to v3_quantize_group_absmax."""
    GROUPS: tl.constexpr = BLOCK_D // GROUP_SIZE
    HALF: tl.constexpr = BLOCK_D // 2
    if BITS == 4:
        QMAX = 7.0
    else:
        QMAX = 127.0

    v2d = tl.reshape(vals, (GROUPS, GROUP_SIZE))
    absmax = tl.max(tl.abs(v2d), axis=1)                     # fp32 [GROUPS]
    qmax_t = tl.full([GROUPS], QMAX, tl.float32)
    scale16 = tl.math.div_rn(absmax, qmax_t).to(tl.float16)
    s_col = pid_d * GROUPS + tl.arange(0, GROUPS)
    tl.store(s_ptr + s_row + s_col, scale16, mask=s_col < G)

    safe = scale16.to(tl.float32)
    safe = tl.where(safe > 0.0, safe, 1.0)
    safe2d = tl.broadcast_to(safe[:, None], (GROUPS, GROUP_SIZE))
    q = _v3_libdevice.rint(tl.math.div_rn(v2d, safe2d))
    q = tl.clamp(q, -QMAX, QMAX)
    if BITS == 4:
        codes = (q + 8.0).to(tl.int32)                       # [GROUPS, GS] 1..15
        pair = tl.reshape(codes, (HALF, 2))                  # row-major pairs
        nib = tl.where(tl.arange(0, 2) == 0, 1, 16)          # lo | hi << 4
        byte = tl.sum(pair * nib[None, :], axis=1)           # [HALF]
        q_col = pid_d * HALF + tl.arange(0, HALF)
        tl.store(q_ptr + q_row + q_col, byte.to(tl.uint8),
                 mask=q_col < (D // 2))
    else:
        q_col = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        tl.store(q_ptr + q_row + q_col,
                 tl.reshape(q, (BLOCK_D,)).to(tl.int8), mask=q_col < D)


@triton.jit
def _v3_silu_mul_quant_fwd_kernel(
    gate_ptr, up_ptr,            # in:  bf16 [N, D] contiguous
    mid_ptr,                     # out: bf16 [N, D]  SiLU(gate) * up
    qgate_ptr, qup_ptr,          # out: uint8 [N, D//2] (BITS=4) | int8 [N, D]
    sgate_ptr, sup_ptr,          # out: fp16 [N, G]
    D, G,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """K1: fused SiLU*mul forward + per-group absmax quantization of BOTH
    h_gate and h_up. Grid: (N, cdiv(D, BLOCK_D)); BLOCK_D % GROUP_SIZE == 0
    and D % GROUP_SIZE == 0, so block boundaries never split a group."""
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    HALF: tl.constexpr = BLOCK_D // 2

    col = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = col < D
    row = pid_n.to(tl.int64) * D
    g = tl.load(gate_ptr + row + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + row + col, mask=mask, other=0.0).to(tl.float32)

    # identical op order to _silu_mul_fwd_kernel => bitwise-equal h_mid
    act = g * tl.sigmoid(g)
    result = act * u
    tl.store(mid_ptr + row + col, result, mask=mask)

    s_row = pid_n.to(tl.int64) * G
    if BITS == 4:
        q_row = pid_n.to(tl.int64) * (D // 2)
    else:
        q_row = row
    _v3_quant_store(g, qgate_ptr, sgate_ptr, q_row, s_row, pid_d, D, G,
                    BITS, GROUP_SIZE, BLOCK_D)
    _v3_quant_store(u, qup_ptr, sup_ptr, q_row, s_row, pid_d, D, G,
                    BITS, GROUP_SIZE, BLOCK_D)


@triton.jit
def _v3_dequant_block(q_ptr, s_ptr, q_row, s_row, pid_d, D, G,
                      BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    """Dequantize one [BLOCK_D] block to fp32 *rounded through bf16*, exactly
    matching v3_dequantize_group_absmax(..., dtype=bf16) -> .to(fp32)."""
    GROUPS: tl.constexpr = BLOCK_D // GROUP_SIZE
    HALF: tl.constexpr = BLOCK_D // 2
    if BITS == 4:
        q_col = pid_d * HALF + tl.arange(0, HALF)
        b = tl.load(q_ptr + q_row + q_col,
                    mask=q_col < (D // 2), other=0).to(tl.int32)
        lo = (b & 15).to(tl.float32) - 8.0
        hi = (b >> 4).to(tl.float32) - 8.0
        codes = tl.interleave(lo, hi)                        # [BLOCK_D]
    else:
        q_col = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        codes = tl.load(q_ptr + q_row + q_col,
                        mask=q_col < D, other=0).to(tl.float32)
    s_col = pid_d * GROUPS + tl.arange(0, GROUPS)
    s = tl.load(s_ptr + s_row + s_col,
                mask=s_col < G, other=0.0).to(tl.float32)
    deq = tl.reshape(codes, (GROUPS, GROUP_SIZE)) * s[:, None]
    # bf16 round-trip: the eager reference materializes bf16 tensors before
    # _swiglu_bwd_kernel upcasts them to fp32 again.
    return tl.reshape(deq, (BLOCK_D,)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _v3_dequant_swiglu_bwd_kernel(
    gmid_ptr,                    # in:  bf16 [N, D] grad wrt h_mid
    qgate_ptr, sgate_ptr,        # in:  quantized h_gate payload + scales
    qup_ptr, sup_ptr,            # in:  quantized h_up payload + scales
    ggate_ptr, gup_ptr,          # out: bf16 [N, D] grad_h_gate / grad_h_up
    hmid_ptr,                    # out: bf16 [N, D] h_mid^ (only if WRITE_HMID)
    D, G,
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WRITE_HMID: tl.constexpr,
):
    """K2: fused dequantize + SwiGLU backward (math identical to
    _swiglu_bwd_kernel on the bf16-dequantized inputs), optionally emitting
    h_mid^ = SiLU(hg^)*hu^ (identical to _silu_mul_fwd_kernel)."""
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    col = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = col < D
    row = pid_n.to(tl.int64) * D
    s_row = pid_n.to(tl.int64) * G
    if BITS == 4:
        q_row = pid_n.to(tl.int64) * (D // 2)
    else:
        q_row = row

    hgate = _v3_dequant_block(qgate_ptr, sgate_ptr, q_row, s_row, pid_d, D, G,
                              BITS, GROUP_SIZE, BLOCK_D)
    hup = _v3_dequant_block(qup_ptr, sup_ptr, q_row, s_row, pid_d, D, G,
                            BITS, GROUP_SIZE, BLOCK_D)
    grad_hmid = tl.load(gmid_ptr + row + col, mask=mask, other=0.0).to(tl.float32)

    # identical math/op order to _swiglu_bwd_kernel
    sig = tl.sigmoid(hgate)
    h_act = hgate * sig
    grad_h_act = grad_hmid * hup
    grad_h_up = grad_hmid * h_act
    dsilu = sig * (1.0 + hgate * (1.0 - sig))
    grad_h_gate = grad_h_act * dsilu
    tl.store(ggate_ptr + row + col, grad_h_gate, mask=mask)
    tl.store(gup_ptr + row + col, grad_h_up, mask=mask)
    if WRITE_HMID:
        result = h_act * hup                                 # == SiLU(hg^)*hu^
        tl.store(hmid_ptr + row + col, result, mask=mask)


def _v3_check_quant_shapes(t, bits, group_size):
    assert bits in (4, 8), f"unsupported bits={bits}"
    assert t.dim() == 2 and t.is_contiguous(), "expected contiguous [N, D]"
    assert t.dtype == torch.bfloat16, f"fused v3 kernels are bf16-only, got {t.dtype}"
    N, D = t.shape
    assert D % group_size == 0, f"D={D} not divisible by group_size={group_size}"
    assert group_size > 0 and (group_size & (group_size - 1)) == 0, \
        f"group_size={group_size} must be a power of two"
    return N, D, D // group_size


def triton_silu_mul_quant(h_gate, h_up, bits, group_size=64):
    """K1 wrapper: returns (h_mid, q_gate, scales_gate, q_up, scales_up).

    h_mid is bitwise-identical to triton_silu_mul(h_gate, h_up); payloads and
    scales are bitwise-identical to v3_quantize_group_absmax (verified in
    src/verify_v3_kernels.py G1/G4)."""
    N, D, G = _v3_check_quant_shapes(h_gate, bits, group_size)
    assert h_up.shape == h_gate.shape and h_up.is_contiguous() \
        and h_up.dtype == h_gate.dtype
    dev = h_gate.device
    h_mid = torch.empty_like(h_gate)
    s_gate = torch.empty((N, G), dtype=torch.float16, device=dev)
    s_up = torch.empty((N, G), dtype=torch.float16, device=dev)
    if bits == 4:
        q_gate = torch.empty((N, D // 2), dtype=torch.uint8, device=dev)
        q_up = torch.empty((N, D // 2), dtype=torch.uint8, device=dev)
    else:
        q_gate = torch.empty((N, D), dtype=torch.int8, device=dev)
        q_up = torch.empty((N, D), dtype=torch.int8, device=dev)
    block = max(group_size, _V3_TRITON_CFG["fwd_block"])
    grid = (N, triton.cdiv(D, block))
    _v3_silu_mul_quant_fwd_kernel[grid](
        h_gate, h_up, h_mid, q_gate, q_up, s_gate, s_up, D, G,
        BITS=bits, GROUP_SIZE=group_size, BLOCK_D=block,
        num_warps=_V3_TRITON_CFG["fwd_warps"])
    _V3_COUNTERS["triton_quant_fwd"] += 1
    return h_mid, q_gate, s_gate, q_up, s_up


def triton_dequant_swiglu_backward(grad_h_mid, q_gate, s_gate, q_up, s_up,
                                   bits, group_size=64, want_h_mid=False):
    """K2 wrapper: returns (grad_h_gate, grad_h_up, h_mid_hat | None).

    Bitwise-identical to: hg^ = v3_dequantize_group_absmax(q_gate, s_gate),
    hu^ likewise, triton_swiglu_backward(grad_h_mid, hg^, hu^) and (when
    want_h_mid) triton_silu_mul(hg^, hu^) — without materializing hg^/hu^."""
    assert grad_h_mid.dim() == 2 and grad_h_mid.is_contiguous()
    assert grad_h_mid.dtype == torch.bfloat16, \
        f"fused v3 kernels are bf16-only, got {grad_h_mid.dtype}"
    N, D = grad_h_mid.shape
    assert D % group_size == 0, f"D={D} not divisible by group_size={group_size}"
    G = D // group_size
    assert s_gate.shape == (N, G) and s_up.shape == (N, G)
    expect_payload = (N, D // 2) if bits == 4 else (N, D)
    assert q_gate.shape == expect_payload and q_up.shape == expect_payload, \
        f"payload shape {q_gate.shape} != expected {expect_payload}"
    grad_h_gate = torch.empty_like(grad_h_mid)
    grad_h_up = torch.empty_like(grad_h_mid)
    h_mid_hat = torch.empty_like(grad_h_mid) if want_h_mid else None
    block = max(group_size, _V3_TRITON_CFG["bwd_block"])
    grid = (N, triton.cdiv(D, block))
    _v3_dequant_swiglu_bwd_kernel[grid](
        grad_h_mid, q_gate, s_gate, q_up, s_up,
        grad_h_gate, grad_h_up,
        h_mid_hat if want_h_mid else grad_h_gate,  # dummy ptr, never stored
        D, G, BITS=bits, GROUP_SIZE=group_size, BLOCK_D=block,
        WRITE_HMID=want_h_mid,
        num_warps=_V3_TRITON_CFG["bwd_warps"])
    _V3_COUNTERS["triton_dequant_bwd"] += 1
    return grad_h_gate, grad_h_up, h_mid_hat


class FlashFFNv3Function(torch.autograd.Function):
    """LoRA-factored SwiGLU FFN with quantized GLU-cache backward.

    apply(x,
          w_gate, a_gate, b_gate,
          w_up,   a_up,   b_up,
          w_down, a_down, b_down,
          s_gate, s_up, s_down,
          cache_mode, group_size)

    a_*/b_* may be None (adapter absent; pass s_*=0.0). Weights may be
    trainable (full-FT) or frozen. Scalars are python floats (no grad).
    cache_mode: 'int4' | 'int8' | 'bf16' | 'recompute'.
    use_triton: for 'int4'/'int8', use the fused K1/K2 Triton kernels
    (silu*mul+quant forward, dequant+swiglu backward). Falls back to the
    eager PyTorch codec when False, when skeleton mode is on, or when the
    activations are not bf16 (the fused kernels are bf16-only).
    """

    @staticmethod
    def forward(ctx, x,
                w_gate, a_gate, b_gate,
                w_up, a_up, b_up,
                w_down, a_down, b_down,
                s_gate=0.0, s_up=0.0, s_down=0.0,
                cache_mode="int4", group_size=64, use_triton=True):
        assert cache_mode in ("int4", "int8", "bf16", "recompute"), cache_mode
        _V3_COUNTERS["forward"] += 1

        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        x_2d = x.reshape(-1, hidden_dim)

        # ---- factored forward (PEFT op order: base, then + s * B(A(x))) ----
        h_gate = F.linear(x_2d, w_gate)
        xa_g = None
        if a_gate is not None:
            xa_g = F.linear(x_2d, a_gate)                       # [N, r]
            h_gate = h_gate + s_gate * F.linear(xa_g, b_gate)

        h_up = F.linear(x_2d, w_up)
        xa_u = None
        if a_up is not None:
            xa_u = F.linear(x_2d, a_up)                         # [N, r]
            h_up = h_up + s_up * F.linear(xa_u, b_up)

        # ---- fused K1 path: silu*mul + quantize in one Triton kernel ----
        c_gate = qs_gate = c_up = qs_up = None
        fused_quant = (cache_mode in ("int4", "int8") and use_triton
                       and not _V3_SKELETON["enabled"]
                       and h_gate.dtype == torch.bfloat16)
        if fused_quant:
            bits = 4 if cache_mode == "int4" else 8
            h_mid, c_gate, qs_gate, c_up, qs_up = triton_silu_mul_quant(
                h_gate, h_up, bits, group_size)
            del h_gate, h_up
        else:
            h_mid = triton_silu_mul(h_gate, h_up)

        y = F.linear(h_mid, w_down)
        ha_d = None
        if a_down is not None:
            ha_d = F.linear(h_mid, a_down)                      # [N, r]
            y = y + s_down * F.linear(ha_d, b_down)

        # ---- build the backward cache (eager codec path) ----
        if not fused_quant:
            if cache_mode in ("int4", "int8"):
                bits = 4 if cache_mode == "int4" else 8
                c_gate, qs_gate = v3_quantize_group_absmax(h_gate, bits, group_size)
                c_up, qs_up = v3_quantize_group_absmax(h_up, bits, group_size)
                del h_gate, h_up
            elif cache_mode == "bf16":
                c_gate, c_up = h_gate, h_up
            else:  # recompute
                del h_gate, h_up
        del h_mid

        ctx.save_for_backward(
            x, w_gate, a_gate, b_gate, w_up, a_up, b_up,
            w_down, a_down, b_down, xa_g, xa_u, ha_d,
            c_gate, qs_gate, c_up, qs_up,
        )
        ctx.s_gate, ctx.s_up, ctx.s_down = s_gate, s_up, s_down
        ctx.cache_mode = cache_mode
        ctx.group_size = group_size
        ctx.use_triton = fused_quant
        ctx.orig_shape = orig_shape
        ctx.in_dtype = x.dtype

        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        _V3_COUNTERS["backward"] += 1
        (x, w_gate, a_gate, b_gate, w_up, a_up, b_up,
         w_down, a_down, b_down, xa_g, xa_u, ha_d,
         c_gate, qs_gate, c_up, qs_up) = ctx.saved_tensors
        s_gate, s_up, s_down = ctx.s_gate, ctx.s_up, ctx.s_down
        mode = ctx.cache_mode
        orig_shape = ctx.orig_shape
        hidden_dim = orig_shape[-1]

        ng = ctx.needs_input_grad
        (need_x, need_wg, need_ag, need_bg, need_wu, need_au, need_bu,
         need_wd, need_ad, need_bd) = ng[:10]

        gy = grad_y.reshape(-1, hidden_dim)
        x_2d = x.reshape(-1, hidden_dim)

        need_gate_path = (need_x or need_wg or need_ag or need_bg
                          or need_wu or need_au or need_bu)
        need_h_mid = need_wd or (a_down is not None and need_ad)
        need_recon = need_gate_path or need_h_mid

        # K2 (fused dequant+swiglu-bwd) handles quantized caches written by K1;
        # it reconstructs everything in-kernel, so hg/hu are never materialized.
        use_k2 = (mode in ("int4", "int8")
                  and getattr(ctx, "use_triton", False) and need_gate_path)

        # ---- reconstruct h_gate / h_up from the cache (eager paths) ----
        hg = hu = None
        if need_recon and not use_k2:
            if mode == "recompute":
                # Same kernels on the same inputs as forward -> bitwise exact.
                hg = F.linear(x_2d, w_gate)
                if a_gate is not None:
                    hg = hg + s_gate * F.linear(xa_g, b_gate)
                hu = F.linear(x_2d, w_up)
                if a_up is not None:
                    hu = hu + s_up * F.linear(xa_u, b_up)
            elif mode == "bf16":
                hg, hu = c_gate, c_up
            else:
                bits = 4 if mode == "int4" else 8
                hg = v3_dequantize_group_absmax(
                    c_gate, qs_gate, bits, ctx.group_size, ctx.in_dtype)
                hu = v3_dequantize_group_absmax(
                    c_up, qs_up, bits, ctx.group_size, ctx.in_dtype)

        grad_w_down = grad_a_down = grad_b_down = None
        gyB_d = None
        if a_down is not None and (need_gate_path or need_ad):
            gyB_d = gy @ b_down                                    # [N, r]

        # ---- fused K2 path: dequant + SwiGLU backward (+ optional h_mid^) ----
        grad_h_gate = grad_h_up = h_mid_hat = None
        if use_k2:
            bits = 4 if mode == "int4" else 8
            grad_h_mid = gy @ w_down                                # [N, D]
            if a_down is not None:
                grad_h_mid = grad_h_mid + s_down * (gyB_d @ a_down)
            grad_h_gate, grad_h_up, h_mid_hat = triton_dequant_swiglu_backward(
                grad_h_mid, c_gate, qs_gate, c_up, qs_up, bits,
                ctx.group_size, want_h_mid=need_h_mid)
            del grad_h_mid

        # ---- down projection grads (h_mid^ only when actually needed) ----
        if need_h_mid:
            if h_mid_hat is None:
                h_mid_hat = triton_silu_mul(hg, hu)
            if need_wd:
                grad_w_down = gy.t() @ h_mid_hat                   # [H, D] dense
            if need_ad:
                grad_a_down = s_down * (gyB_d.t() @ h_mid_hat)     # [r, D]
            del h_mid_hat
        if need_bd:
            grad_b_down = s_down * (gy.t() @ ha_d)                 # [H, r]

        # ---- gate/up path ----
        grad_x = None
        grad_w_gate = grad_a_gate = grad_b_gate = None
        grad_w_up = grad_a_up = grad_b_up = None
        if need_gate_path:
            if grad_h_gate is None:  # eager path (K2 already produced these)
                grad_h_mid = gy @ w_down                            # [N, D]
                if a_down is not None:
                    grad_h_mid = grad_h_mid + s_down * (gyB_d @ a_down)
                grad_h_gate, grad_h_up = triton_swiglu_backward(
                    grad_h_mid, hg, hu)
                del grad_h_mid

            ghgB = None
            if a_gate is not None and (need_x or need_ag):
                ghgB = grad_h_gate @ b_gate                         # [N, r]
            ghuB = None
            if a_up is not None and (need_x or need_au):
                ghuB = grad_h_up @ b_up                             # [N, r]

            if need_x:
                grad_x = grad_h_gate @ w_gate
                if a_gate is not None:
                    grad_x = grad_x + s_gate * (ghgB @ a_gate)
                grad_x = grad_x + grad_h_up @ w_up
                if a_up is not None:
                    grad_x = grad_x + s_up * (ghuB @ a_up)

            if need_bg:
                grad_b_gate = s_gate * (grad_h_gate.t() @ xa_g)     # [D, r]
            if need_ag:
                grad_a_gate = s_gate * (ghgB.t() @ x_2d)            # [r, H]
            if need_bu:
                grad_b_up = s_up * (grad_h_up.t() @ xa_u)           # [D, r]
            if need_au:
                grad_a_up = s_up * (ghuB.t() @ x_2d)                # [r, H]

            if need_wg:
                grad_w_gate = grad_h_gate.t() @ x_2d                # dense (full-FT)
            if need_wu:
                grad_w_up = grad_h_up.t() @ x_2d                    # dense (full-FT)

        # INVARIANT: frozen weights => no dense [D_out, D_in] grads were built.
        if not (need_wg or need_wu or need_wd):
            assert grad_w_gate is None and grad_w_up is None \
                and grad_w_down is None, \
                "FlashFFNv3 invariant violated: dense weight grad with frozen weights"

        if need_x and len(orig_shape) == 3:
            grad_x = grad_x.view(orig_shape)

        return (grad_x,
                grad_w_gate, grad_a_gate, grad_b_gate,
                grad_w_up, grad_a_up, grad_b_up,
                grad_w_down, grad_a_down, grad_b_down,
                ) + (None,) * (len(ctx.needs_input_grad) - 10)


def _v3_extract_proj(proj: nn.Module):
    """Extract (W, A, B, s) from a plain nn.Linear or a PEFT LoRA lora.Linear.

    Plain Linear (full-FT or frozen): returns (weight, None, None, 0.0).
    PEFT LoRA: returns (base weight, lora_A [r,in], lora_B [out,r], scaling).
    Rejects DoRA / AdaLoRA / multi-adapter setups (prototype = plain LoRA only).
    LoRA dropout is ignored; warns if p > 0.
    """
    is_lora = (hasattr(proj, "base_layer")
               and isinstance(getattr(proj, "lora_A", None), nn.ModuleDict)
               and len(proj.lora_A) > 0)
    if not is_lora:
        return proj.weight, None, None, 0.0
    if len(getattr(proj, "lora_magnitude_vector", {}) or {}) > 0:
        raise NotImplementedError("FlashFFNv3 prototype does not support DoRA")
    if hasattr(proj, "lora_E"):
        raise NotImplementedError("FlashFFNv3 prototype does not support AdaLoRA")
    assert len(proj.active_adapters) == 1, "exactly one active adapter expected"
    name = proj.active_adapters[0]
    drop = proj.lora_dropout[name]
    p = float(getattr(drop, "p", 0.0))
    if p > 0.0:
        warnings.warn(
            f"FlashFFNv3 ignores LoRA dropout (p={p}); results differ from PEFT "
            "forward when training with dropout.")
    w = proj.get_base_layer().weight
    a = proj.lora_A[name].weight
    b = proj.lora_B[name].weight
    s = float(proj.scaling[name])
    return w, a, b, s


def make_v3_forward(mlp: nn.Module, cache_mode: str = "int4",
                    group_size: int = 64, use_triton: bool = True):
    """Build a drop-in replacement for a LLaMA-style MLP forward.

    `mlp` must expose gate_proj / up_proj / down_proj, each either a plain
    nn.Linear or a PEFT-LoRA-wrapped lora.Linear. Factors are re-extracted on
    every call (parameters may be updated by the optimizer between steps);
    adapters that live in a different dtype than x are cast through autograd
    (grads still reach the original parameters).
    use_triton: use the fused K1/K2 kernels for 'int4'/'int8' caches
    (default); False keeps the eager PyTorch codec for cross-checking.

    Adapter routing (additive, 2026-06-12): projections wrapped by DoRA /
    AdaLoRA / DyLoRA / VeRA (possibly mixed with plain/LoRA) are routed to
    make_v3_adapter_forward (sibling Function, shared cache/kernels). The
    plain/LoRA fast path below is byte-identical to before. Unknown wrappers
    raise loudly — no silent fallback.
    """
    kinds = tuple(_v3_proj_kind(getattr(mlp, p))
                  for p in ("gate_proj", "up_proj", "down_proj"))
    if any(k == "unknown" for k in kinds):
        raise NotImplementedError(
            f"FlashFFNv3: unsupported projection wrapper(s) on MLP: {kinds}")
    if any(k not in ("plain", "lora") for k in kinds):
        return make_v3_adapter_forward(mlp, cache_mode, group_size, use_triton)

    def v3_forward(x):
        wg, ag, bg, sg = _v3_extract_proj(mlp.gate_proj)
        wu, au, bu, su = _v3_extract_proj(mlp.up_proj)
        wd, ad, bd, sd = _v3_extract_proj(mlp.down_proj)
        dt = x.dtype

        def cast(t):
            return t if (t is None or t.dtype == dt) else t.to(dt)

        return FlashFFNv3Function.apply(
            x,
            cast(wg), cast(ag), cast(bg),
            cast(wu), cast(au), cast(bu),
            cast(wd), cast(ad), cast(bd),
            sg, su, sd, cache_mode, group_size, use_triton)
    return v3_forward


class FlashFFNv3(nn.Module):
    """Thin module wrapper around make_v3_forward for an existing MLP."""

    def __init__(self, mlp: nn.Module, cache_mode: str = "int4",
                 group_size: int = 64, use_triton: bool = True):
        super().__init__()
        self.mlp = mlp
        self.cache_mode = cache_mode
        self.group_size = group_size
        self.use_triton = use_triton
        self._fwd = make_v3_forward(mlp, cache_mode, group_size, use_triton)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fwd(x)


# =============================================================================
# FlashFFN v3 — adapter-factored extension: DoRA / AdaLoRA / DyLoRA / VeRA
# =============================================================================
#
# Extends the v3 factored design (no merged [D_out, D_in] weight anywhere) to
# the four remaining adapter families. The GLU cache layer (int4/int8/bf16/
# recompute, fused K1/K2 kernels) is SHARED with FlashFFNv3Function; what
# changes per adapter is the projection forward and the thin gradient algebra.
# The plain/LoRA fast path (FlashFFNv3Function) is untouched; this section adds
# a sibling Function. Per-projection mixing (e.g. DoRA gate + plain up) is
# supported.
#
# ---------------------------------------------------------------------------
# DERIVATIONS (on paper, verified numerically in src/verify_v3_adapters.py)
# ---------------------------------------------------------------------------
# Notation per projection: input z [N, in], frozen base W [out, in], upstream
# grad g = dL/d(out) [N, out]. LoRA-style bypass factors A [r, in], B [out, r],
# scale s (python float). zA := z @ A.T [N, r] (thin, saved).
#
# (0) LoRA core (already in FlashFFNv3Function; reused by all four below):
#       out    = z@W.T + s*(zA)@B.T
#       grad_z = g@W + s*(g@B)@A
#       grad_B = s * g.T @ zA            [out, r]   thin
#       grad_A = s * (g@B).T @ z         [r, in]    thin
#
# (1) VeRA (PEFT): out = z@W.T + lam_b * ((z@A.T) * lam_d) @ B.T
#     with A [r,in], B [out,r] FROZEN shared buffers; lam_d [r], lam_b [out]
#     trainable. Equivalent factored form (matches the repo's v2
#     _compute_vera_effective_weight  W_eff = W + (lam_b[:,None]*B)@(lam_d[:,None]*A)):
#       A_eff = lam_d[:,None] * A        [r, in]   thin elementwise, autograd-tracked
#       B_eff = lam_b[:,None] * B        [out, r]  thin elementwise, autograd-tracked
#       out   = z@W.T + 1.0*(z@A_eff.T)@B_eff.T    -> LoRA core with s=1
#     Thin lambda grads (rowsums of thin products; what autograd's MulBackward
#     computes from the LoRA-core grads — no [out,in] tensor anywhere):
#       grad_lam_d = rowsum(grad_A_eff * A)  = ((g*lam_b)@B * zA).sum(0)    [r]
#       grad_lam_b = rowsum(grad_B_eff * B)  = (g * ((zA*lam_d)@B.T)).sum(0) [out]
#
# (2) AdaLoRA (PEFT SVDLinear): out = z@W.T + (z@(A*E).T)@B.T * s/(ranknum+1e-5)
#     with A [r,in], E [r,1], B [out,r] trainable; ranknum requires_grad=False.
#       A_eff = A * E                    [r, in]   thin elementwise, autograd-tracked
#       s_eff = s / (float(ranknum) + 1e-5)        python float (no grad, matches v2)
#       out   = LoRA core with (A_eff, B, s_eff)
#     Thin grads through the A*E product (autograd MulBackward on [r,in]):
#       grad_A = grad_A_eff * E          (row-broadcast)
#       grad_E = rowsum(grad_A_eff * A, dim=1, keepdim=True)   [r, 1]
#     Rank-pruned rows (E_j = 0) contribute 0 to forward; their grads follow
#     PEFT reference behaviour exactly (nonzero grad_E possible; the allocator
#     re-masks after the step).
#
# (3) DyLoRA (src/dylora.py DyLoRALinear): per-forward (per projection,
#     independently) sample b ~ U{1..r} in training, b = r in eval:
#       out = z@W.T + s*(z@A[:b,:].T)@B[:,:b].T ,  s = alpha/r (fixed)
#     Implemented by passing the SLICES A[:b,:], B[:,:b] into the Function:
#     the slices are autograd views, so the same b is used in backward (the
#     saved tensors ARE the slices), and SliceBackward scatters the returned
#     thin grads into full-size zero-padded .grad — identical to DyLoRA's own
#     autograd behaviour (rows/cols >= b get zero grad, not "untouched").
#
# (4) DoRA (PEFT, hardest): column(=per-output-row in [out,in] layout) norm:
#       nu_j = || W_j + s*(B@A)_j ||_2   over dim=1 (in-features), DETACHED
#       c_j  = m_j / nu_j                m [out] trainable magnitude
#       out  = c * (z@W.T + s*zA@B.T) = c * raw     (c broadcast over rows N)
#     (confirmed vs PEFT lora.dora.DoraLinearLayer.forward: result =
#      base + (c-1)*base_result + c*lora_result*s = c*(base_result + s*lora_result),
#      weight_norm over dim=1 then .detach(); and vs the repo's v2
#      _compute_dora_effective_weight.)
#     FACTORED norm without materializing B@A — Gram identity (related work on
#     factored DoRA norms: "Scaling DoRA", arXiv 2603.22276):
#       nu_j^2 = ||W_j||^2 + 2s*sum_r B_jr*(W@A.T)_jr + s^2*sum_{r,r'} B_jr*(A@A.T)_{rr'}*B_jr'
#     i.e. w2 = rowsum(W*W); WA = W@A.T (ONE tall-thin [out,r] GEMM, fp32);
#     cross = 2s*rowsum(B*WA); quad = s^2*rowsum((B@G)*B) with G = A@A.T [r,r].
#     Backward (c is constant w.r.t. raw because nu is detached; only the m
#     factor of c carries grad):
#       dL/draw  = g * c[None,:]   =: g_eff      -> LoRA core grads with g_eff:
#       grad_z = g_eff@W + s*(g_eff@B)@A
#       grad_B = s * g_eff.T @ zA ;  grad_A = s * (g_eff@B).T @ z
#       grad_m_j = sum_n g[n,j] * raw[n,j] / nu_j          (uses RAW g, not g_eff)
#     Obtaining raw in backward without storing it ([N,out]-sized):
#       - recompute mode: raw is recomputed anyway (h = c*raw replays forward
#         bitwise), so grad_m = (g_h (.) raw).sum(0) / nu   where g_h is the
#         SwiGLU-backward grad w.r.t. h (gate/up) or grad_y (down).
#       - cached modes (int4/int8/bf16): the cache holds h^ = Q(c*raw), so
#         raw^ = h^/c and  grad_m = (g_h (.) h^).sum(0) / m   (since c*nu = m;
#         the absmax-quant error scales with |h| ~ m, so the division does not
#         amplify relative error; m_j == 0 exactly would give 0/0 = NaN, loud
#         by design — m is initialised to ||W_j|| > 0).
#       - down projection: y is never cached, so raw_d is recomputed thin+dense:
#         raw_d = h_mid^@W_d.T + s*ha_d@B_d.T (ONE extra dense GEMM, honest
#         cost, only when grad_m_down is needed).
#     All sums use fp32 accumulation (.sum(0, dtype=float32)).
#
# Cache layer: IDENTICAL to FlashFFNv3Function. h_gate/h_up are cached POST-c
# for DoRA (that is what SwiGLU backward needs); K1/K2 kernels are reused
# unmodified. In the K2 path the magnitude grads additionally need h^ itself
# (K2 never materializes it), so h_gate/h_up are eager-dequantized for DoRA —
# bitwise-identical codec to K2's in-kernel dequant (verified G1/G2).
# =============================================================================

import random as _v3_random


def _v3_proj_kind(proj: nn.Module) -> str:
    """Structurally classify one MLP projection module.

    Returns 'dora' | 'adalora' | 'vera' | 'dylora' | 'lora' | 'plain' |
    'unknown'. Mirrors train_glue.py's _is_*_wrapped helpers (duplicated here
    because flashffn must not import train_glue — circular)."""
    if hasattr(proj, "base_layer"):
        if len(getattr(proj, "lora_magnitude_vector", None) or {}) > 0:
            return "dora"
        if len(getattr(proj, "lora_E", None) or {}) > 0:
            return "adalora"
        if len(getattr(proj, "vera_lambda_d", None) or {}) > 0:
            return "vera"
        if (isinstance(getattr(proj, "lora_A", None), nn.Parameter)
                and isinstance(getattr(proj, "lora_B", None), nn.Parameter)
                and hasattr(proj, "r") and hasattr(proj, "scaling")):
            return "dylora"  # custom DyLoRALinear (src/dylora.py)
        if (isinstance(getattr(proj, "lora_A", None), nn.ModuleDict)
                and len(proj.lora_A) > 0):
            return "lora"
        return "unknown"  # some other PEFT wrapper
    if hasattr(proj, "weight"):
        return "plain"
    return "unknown"


@torch.no_grad()
def _v3_dora_col_norm(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                      s: float) -> torch.Tensor:
    """DoRA column norm ||W + s*B@A||_2 over dim=1 WITHOUT materializing B@A.

    Gram identity (see derivation block above; related work for the factored-
    norm idea: "Scaling DoRA", arXiv 2603.22276):
      nu_j^2 = ||W_j||^2 + 2s*rowsum(B * (W@A.T))_j + s^2*rowsum((B@(A@A.T)) * B)_j
    Only thin GEMMs: W@A.T is [out, r] (tall-thin), A@A.T is [r, r]. Computed
    in fp32 for stability; result is detached (PEFT detaches the norm)."""
    wf, af, bf = w.float(), a.float(), b.float()
    w2 = (wf * wf).sum(dim=1)                       # [out]
    wa = wf @ af.t()                                # [out, r] tall-thin fp32 GEMM
    cross = (2.0 * s) * (bf * wa).sum(dim=1)        # [out]
    gram = af @ af.t()                              # [r, r]
    quad = (s * s) * ((bf @ gram) * bf).sum(dim=1)  # [out]
    return (w2 + cross + quad).clamp_min_(0.0).sqrt_()


def _v3_warn_dropout(kind: str, p: float):
    if p > 0.0:
        warnings.warn(
            f"FlashFFNv3 ignores {kind} dropout (p={p}); results differ from the "
            "reference forward when training with dropout (same trade-off as the "
            "v2 effective-weight path).")


def _v3_extract_proj_adapter(proj: nn.Module, dt: torch.dtype):
    """Per-forward extraction of one projection's factored parameterization.

    Returns (w, a, b, s, m, nu):
      w  [out, in] base weight (frozen or trainable full-FT)
      a  [r, in] / b [out, r] LoRA-style bypass factors (None if absent).
         For AdaLoRA a = lora_A*lora_E; for VeRA a = lam_d[:,None]*A_frozen and
         b = lam_b[:,None]*B_frozen — composed OUTSIDE the autograd.Function so
         autograd's MulBackward executes the derived thin chain rules
         (grad_E = rowsum(grad_A_eff*A) etc.) on [r,in]/[out,r] tensors only.
         For DyLoRA a/b are the rank-b slices (per-forward sampled b in
         training; SliceBackward zero-pads the grads like DyLoRA itself).
      s  python float bypass scale.
      m / nu: DoRA magnitude [out] (trainable) and DETACHED column norm [out]
         (fp32 Gram identity, cast to dt); None for non-DoRA.
    Must be called once per forward (DyLoRA samples b; AdaLoRA ranknum and all
    trainables move between steps)."""
    kind = _v3_proj_kind(proj)

    def cast(t):
        return t if (t is None or t.dtype == dt) else t.to(dt)

    if kind == "plain":
        return cast(proj.weight), None, None, 0.0, None, None

    if kind == "lora":
        w, a, b, s = _v3_extract_proj(proj)
        return cast(w), cast(a), cast(b), s, None, None

    if kind == "dylora":
        _v3_warn_dropout("DyLoRA", float(getattr(proj.dropout, "p", 0.0)))
        b_rank = _v3_random.randint(1, proj.r) if proj.training else proj.r
        return (cast(proj.base_layer.weight),
                cast(proj.lora_A[:b_rank, :]),
                cast(proj.lora_B[:, :b_rank]),
                float(proj.scaling), None, None)

    if kind == "adalora":
        name = proj.active_adapters[0]
        _v3_warn_dropout("AdaLoRA", float(getattr(proj.lora_dropout[name], "p", 0.0)))
        a_eff = proj.lora_A[name] * proj.lora_E[name]          # [r, in] thin
        s_eff = float(proj.scaling[name]) / (float(proj.ranknum[name]) + 1e-5)
        return (cast(proj.get_base_layer().weight), cast(a_eff),
                cast(proj.lora_B[name]), s_eff, None, None)

    if kind == "vera":
        name = proj.active_adapters[0]
        _v3_warn_dropout("VeRA", float(getattr(proj.vera_dropout[name], "p", 0.0)))
        lam_d = proj.vera_lambda_d[name]                       # [r] trainable
        lam_b = proj.vera_lambda_b[name]                       # [out] trainable
        a_frozen = proj.vera_A[name][:, : proj.in_features]    # [r, in] buffer
        b_frozen = proj.vera_B[name][: proj.out_features, :]   # [out, r] buffer
        a_eff = lam_d.unsqueeze(-1) * a_frozen                 # thin, tracked
        b_eff = lam_b.unsqueeze(-1) * b_frozen                 # thin, tracked
        return (cast(proj.get_base_layer().weight), cast(a_eff), cast(b_eff),
                1.0, None, None)

    if kind == "dora":
        name = proj.active_adapters[0]
        _v3_warn_dropout("DoRA", float(getattr(proj.lora_dropout[name], "p", 0.0)))
        w = cast(proj.get_base_layer().weight)
        a = cast(proj.lora_A[name].weight)
        b = cast(proj.lora_B[name].weight)
        s = float(proj.scaling[name])
        m = cast(proj.lora_magnitude_vector[name].weight)      # [out] trainable
        nu = _v3_dora_col_norm(w, a, b, s).to(dt)              # [out] DETACHED
        return w, a, b, s, m, nu

    raise NotImplementedError(
        f"FlashFFNv3 adapter path: unsupported projection wrapper "
        f"({type(proj).__name__}).")


class FlashFFNv3AdapterFunction(torch.autograd.Function):
    """Sibling of FlashFFNv3Function adding per-projection DoRA magnitude
    scaling (m, nu) on top of the LoRA-style factored form. AdaLoRA / VeRA /
    DyLoRA enter through composed/sliced thin factors (see
    _v3_extract_proj_adapter), so this Function only knows two shapes of
    projection: LoRA-core and LoRA-core-with-magnitude.

    apply(x,
          w_gate, a_gate, b_gate, m_gate, nu_gate,
          w_up,   a_up,   b_up,   m_up,   nu_up,
          w_down, a_down, b_down, m_down, nu_down,
          s_gate, s_up, s_down, cache_mode, group_size, use_triton)

    a_*/b_* may be None (no bypass); m_*/nu_* may be None (no magnitude).
    cache_mode / group_size / use_triton: identical semantics and identical
    kernels (K1/K2) to FlashFFNv3Function. nu_* are detached constants and
    never receive grads.
    """

    @staticmethod
    def forward(ctx, x,
                w_gate, a_gate, b_gate, m_gate, nu_gate,
                w_up, a_up, b_up, m_up, nu_up,
                w_down, a_down, b_down, m_down, nu_down,
                s_gate=0.0, s_up=0.0, s_down=0.0,
                cache_mode="int4", group_size=64, use_triton=True):
        assert cache_mode in ("int4", "int8", "bf16", "recompute"), cache_mode
        _V3_COUNTERS["forward"] += 1
        _V3_COUNTERS["adapter_forward"] += 1

        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        x_2d = x.reshape(-1, hidden_dim)

        # ---- factored forward (same op order as FlashFFNv3Function, plus the
        # per-output-channel DoRA scale c = m/nu applied AFTER the bypass) ----
        h_gate = F.linear(x_2d, w_gate)
        xa_g = None
        if a_gate is not None:
            xa_g = F.linear(x_2d, a_gate)                       # [N, r]
            h_gate = h_gate + s_gate * F.linear(xa_g, b_gate)
        if m_gate is not None:
            h_gate = h_gate * (m_gate / nu_gate).unsqueeze(0)

        h_up = F.linear(x_2d, w_up)
        xa_u = None
        if a_up is not None:
            xa_u = F.linear(x_2d, a_up)                         # [N, r]
            h_up = h_up + s_up * F.linear(xa_u, b_up)
        if m_up is not None:
            h_up = h_up * (m_up / nu_up).unsqueeze(0)

        # ---- shared cache layer: fused K1 or eager (identical to v3) ----
        c_gate = qs_gate = c_up = qs_up = None
        fused_quant = (cache_mode in ("int4", "int8") and use_triton
                       and not _V3_SKELETON["enabled"]
                       and h_gate.dtype == torch.bfloat16)
        if fused_quant:
            bits = 4 if cache_mode == "int4" else 8
            h_mid, c_gate, qs_gate, c_up, qs_up = triton_silu_mul_quant(
                h_gate, h_up, bits, group_size)
            del h_gate, h_up
        else:
            h_mid = triton_silu_mul(h_gate, h_up)

        y = F.linear(h_mid, w_down)
        ha_d = None
        if a_down is not None:
            ha_d = F.linear(h_mid, a_down)                      # [N, r]
            y = y + s_down * F.linear(ha_d, b_down)
        if m_down is not None:
            y = y * (m_down / nu_down).unsqueeze(0)

        if not fused_quant:
            if cache_mode in ("int4", "int8"):
                bits = 4 if cache_mode == "int4" else 8
                c_gate, qs_gate = v3_quantize_group_absmax(h_gate, bits, group_size)
                c_up, qs_up = v3_quantize_group_absmax(h_up, bits, group_size)
                del h_gate, h_up
            elif cache_mode == "bf16":
                c_gate, c_up = h_gate, h_up
            else:  # recompute
                del h_gate, h_up
        del h_mid

        ctx.save_for_backward(
            x, w_gate, a_gate, b_gate, m_gate, nu_gate,
            w_up, a_up, b_up, m_up, nu_up,
            w_down, a_down, b_down, m_down, nu_down,
            xa_g, xa_u, ha_d, c_gate, qs_gate, c_up, qs_up,
        )
        ctx.s_gate, ctx.s_up, ctx.s_down = s_gate, s_up, s_down
        ctx.cache_mode = cache_mode
        ctx.group_size = group_size
        ctx.use_triton = fused_quant
        ctx.orig_shape = orig_shape
        ctx.in_dtype = x.dtype

        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        _V3_COUNTERS["backward"] += 1
        _V3_COUNTERS["adapter_backward"] += 1
        (x, w_gate, a_gate, b_gate, m_gate, nu_gate,
         w_up, a_up, b_up, m_up, nu_up,
         w_down, a_down, b_down, m_down, nu_down,
         xa_g, xa_u, ha_d, cq_gate, qs_gate, cq_up, qs_up) = ctx.saved_tensors
        s_gate, s_up, s_down = ctx.s_gate, ctx.s_up, ctx.s_down
        mode = ctx.cache_mode
        orig_shape = ctx.orig_shape
        hidden_dim = orig_shape[-1]

        ng = ctx.needs_input_grad
        need_x = ng[0]
        need_wg, need_ag, need_bg, need_mg = ng[1], ng[2], ng[3], ng[4]
        need_wu, need_au, need_bu, need_mu = ng[6], ng[7], ng[8], ng[9]
        need_wd, need_ad, need_bd, need_md = ng[11], ng[12], ng[13], ng[14]

        gy = grad_y.reshape(-1, hidden_dim)
        x_2d = x.reshape(-1, hidden_dim)

        # DoRA scales — recomputed with the SAME op as forward (bitwise equal).
        cg = (m_gate / nu_gate) if m_gate is not None else None
        cu = (m_up / nu_up) if m_up is not None else None
        cd = (m_down / nu_down) if m_down is not None else None

        need_gate_path = (need_x or need_wg or need_ag or need_bg or need_mg
                          or need_wu or need_au or need_bu or need_mu)
        need_h_mid = need_wd or need_md or (a_down is not None and need_ad)
        need_recon = need_gate_path or need_h_mid
        use_k2 = (mode in ("int4", "int8")
                  and getattr(ctx, "use_triton", False) and need_gate_path)

        # ---- reconstruct h_gate / h_up (POST-c) from the cache ----
        hg = hu = raw_g = raw_u = None
        if need_recon and not use_k2:
            if mode == "recompute":
                # Same kernels on the same inputs as forward -> bitwise exact.
                raw_g = F.linear(x_2d, w_gate)
                if a_gate is not None:
                    raw_g = raw_g + s_gate * F.linear(xa_g, b_gate)
                hg = raw_g * cg.unsqueeze(0) if cg is not None else raw_g
                raw_u = F.linear(x_2d, w_up)
                if a_up is not None:
                    raw_u = raw_u + s_up * F.linear(xa_u, b_up)
                hu = raw_u * cu.unsqueeze(0) if cu is not None else raw_u
            elif mode == "bf16":
                hg, hu = cq_gate, cq_up
            else:
                bits = 4 if mode == "int4" else 8
                hg = v3_dequantize_group_absmax(
                    cq_gate, qs_gate, bits, ctx.group_size, ctx.in_dtype)
                hu = v3_dequantize_group_absmax(
                    cq_up, qs_up, bits, ctx.group_size, ctx.in_dtype)
        elif use_k2 and (need_mg or need_mu):
            # K2 reconstructs h^ in-kernel without materializing it; the DoRA
            # magnitude grads need h^ itself, so dequantize eagerly (codec is
            # bitwise-identical to K2's in-kernel dequant, verified G1/G2).
            bits = 4 if mode == "int4" else 8
            if need_mg:
                hg = v3_dequantize_group_absmax(
                    cq_gate, qs_gate, bits, ctx.group_size, ctx.in_dtype)
            if need_mu:
                hu = v3_dequantize_group_absmax(
                    cq_up, qs_up, bits, ctx.group_size, ctx.in_dtype)

        # ---- down projection ----
        # y = c_d * raw_d  =>  dL/draw_d = gy * c_d  =: g_eff_y (LoRA core grads
        # use g_eff_y; grad_m_down uses RAW gy — see derivation block).
        g_eff_y = gy * cd.unsqueeze(0) if cd is not None else gy
        gyB_d = None
        if a_down is not None and (need_gate_path or need_ad):
            gyB_d = g_eff_y @ b_down                               # [N, r]

        # ---- fused K2 path: dequant + SwiGLU backward (+ optional h_mid^) ----
        grad_h_gate = grad_h_up = h_mid_hat = None
        if use_k2:
            bits = 4 if mode == "int4" else 8
            grad_h_mid = g_eff_y @ w_down                          # [N, D]
            if a_down is not None:
                grad_h_mid = grad_h_mid + s_down * (gyB_d @ a_down)
            grad_h_gate, grad_h_up, h_mid_hat = triton_dequant_swiglu_backward(
                grad_h_mid, cq_gate, qs_gate, cq_up, qs_up, bits,
                ctx.group_size, want_h_mid=need_h_mid)
            del grad_h_mid

        grad_w_down = grad_a_down = grad_b_down = grad_m_down = None
        if need_h_mid:
            if h_mid_hat is None:
                h_mid_hat = triton_silu_mul(hg, hu)
            if need_wd:
                grad_w_down = g_eff_y.t() @ h_mid_hat              # [H, D] dense
            if need_ad:
                grad_a_down = s_down * (gyB_d.t() @ h_mid_hat)     # [r, D]
            if need_md:
                # raw_d recomputed (never stored): one extra dense GEMM.
                raw_d = F.linear(h_mid_hat, w_down)
                if a_down is not None:
                    raw_d = raw_d + s_down * F.linear(ha_d, b_down)
                grad_m_down = ((gy * raw_d).sum(0, dtype=torch.float32)
                               / nu_down.float()).to(m_down.dtype)
                del raw_d
            del h_mid_hat
        if need_bd:
            grad_b_down = s_down * (g_eff_y.t() @ ha_d)            # [H, r]

        # ---- gate/up path ----
        grad_x = None
        grad_w_gate = grad_a_gate = grad_b_gate = grad_m_gate = None
        grad_w_up = grad_a_up = grad_b_up = grad_m_up = None
        if need_gate_path:
            if grad_h_gate is None:  # eager path (K2 already produced these)
                grad_h_mid = g_eff_y @ w_down                       # [N, D]
                if a_down is not None:
                    grad_h_mid = grad_h_mid + s_down * (gyB_d @ a_down)
                grad_h_gate, grad_h_up = triton_swiglu_backward(
                    grad_h_mid, hg, hu)
                del grad_h_mid

            # grad_m (gate/up): recompute mode has raw; cached modes use h^/m.
            if need_mg:
                if raw_g is not None:
                    grad_m_gate = ((grad_h_gate * raw_g).sum(0, dtype=torch.float32)
                                   / nu_gate.float()).to(m_gate.dtype)
                else:
                    grad_m_gate = ((grad_h_gate * hg).sum(0, dtype=torch.float32)
                                   / m_gate.float()).to(m_gate.dtype)
            if need_mu:
                if raw_u is not None:
                    grad_m_up = ((grad_h_up * raw_u).sum(0, dtype=torch.float32)
                                 / nu_up.float()).to(m_up.dtype)
                else:
                    grad_m_up = ((grad_h_up * hu).sum(0, dtype=torch.float32)
                                 / m_up.float()).to(m_up.dtype)

            # h = c * raw  =>  dL/draw = grad_h * c
            g_eff_g = grad_h_gate * cg.unsqueeze(0) if cg is not None else grad_h_gate
            g_eff_u = grad_h_up * cu.unsqueeze(0) if cu is not None else grad_h_up

            ghgB = None
            if a_gate is not None and (need_x or need_ag):
                ghgB = g_eff_g @ b_gate                             # [N, r]
            ghuB = None
            if a_up is not None and (need_x or need_au):
                ghuB = g_eff_u @ b_up                               # [N, r]

            if need_x:
                grad_x = g_eff_g @ w_gate
                if a_gate is not None:
                    grad_x = grad_x + s_gate * (ghgB @ a_gate)
                grad_x = grad_x + g_eff_u @ w_up
                if a_up is not None:
                    grad_x = grad_x + s_up * (ghuB @ a_up)

            if need_bg:
                grad_b_gate = s_gate * (g_eff_g.t() @ xa_g)         # [D, r]
            if need_ag:
                grad_a_gate = s_gate * (ghgB.t() @ x_2d)            # [r, H]
            if need_bu:
                grad_b_up = s_up * (g_eff_u.t() @ xa_u)             # [D, r]
            if need_au:
                grad_a_up = s_up * (ghuB.t() @ x_2d)                # [r, H]

            if need_wg:
                grad_w_gate = g_eff_g.t() @ x_2d                    # dense (full-FT)
            if need_wu:
                grad_w_up = g_eff_u.t() @ x_2d                      # dense (full-FT)

        # INVARIANT: frozen weights => no dense [D_out, D_in] grads were built.
        if not (need_wg or need_wu or need_wd):
            assert grad_w_gate is None and grad_w_up is None \
                and grad_w_down is None, \
                "FlashFFNv3 invariant violated: dense weight grad with frozen weights"

        if need_x and len(orig_shape) == 3:
            grad_x = grad_x.view(orig_shape)

        return (grad_x,
                grad_w_gate, grad_a_gate, grad_b_gate, grad_m_gate, None,
                grad_w_up, grad_a_up, grad_b_up, grad_m_up, None,
                grad_w_down, grad_a_down, grad_b_down, grad_m_down, None,
                ) + (None,) * (len(ctx.needs_input_grad) - 16)


def make_v3_adapter_forward(mlp: nn.Module, cache_mode: str = "int4",
                            group_size: int = 64, use_triton: bool = True):
    """Adapter-factored v3 forward for an MLP whose projections may be wrapped
    by DoRA / AdaLoRA / DyLoRA / VeRA (or PEFT LoRA / plain Linear, mixed).

    Factors (incl. composed thin factors and the DoRA Gram-identity norm) are
    re-extracted on EVERY call — parameters move between optimizer steps, the
    DoRA norm tracks A/B, DyLoRA samples its rank per forward."""
    def v3_adapter_forward(x):
        dt = x.dtype
        wg, ag, bg, sg, mg, ng_ = _v3_extract_proj_adapter(mlp.gate_proj, dt)
        wu, au, bu, su, mu, nu_ = _v3_extract_proj_adapter(mlp.up_proj, dt)
        wd, ad, bd, sd, md, nd_ = _v3_extract_proj_adapter(mlp.down_proj, dt)
        return FlashFFNv3AdapterFunction.apply(
            x,
            wg, ag, bg, mg, ng_,
            wu, au, bu, mu, nu_,
            wd, ad, bd, md, nd_,
            sg, su, sd, cache_mode, group_size, use_triton)
    return v3_adapter_forward


# =============================================================================
# Fused LoRA decoder block  (FlashFFN "block" path)
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# `llmdocs/trackers/hyclora_flash_variant.md` §7 measured the decomposition of our memory gap
# against HyC-LoRA's FlashAttention variant and found that **107.9% of their lead is fusion, not
# compression**: at equal compression (none) their fused whole-layer autograd Function stores
# 94.97 MiB/layer against our 87.99, yet their *peak* is far lower because a fused Function never
# pays the 81 MiB/layer of framework tax that a stock HuggingFace decoder layer spends before any
# codec runs (`v3_memory_profile.md` §3, class (a) = 1172 MiB: fp32 RMSNorm upcasts, `repeat_kv`
# GQA pre-expansion, and a duplicated attention-output layout copy).
#
# This Function is the answer to that: one autograd node per decoder block that
#   * owns its RMSNorms  -> stores `rstd` [M] fp32 and recomputes, never an fp32 [B,S,H] upcast;
#   * feeds 4-head k/v straight into FlashAttention-2 -> no `repeat_kv` pre-expansion;
#   * owns its layouts   -> exactly one bf16 copy of the attention output exists;
#   * reuses the FACTORED adapter math of FlashFFNv3Function verbatim (thin [N, r] matmuls, the
#     merged weight W + s*B@A is never materialised) for all SEVEN projections, not just the FFN.
#
# WHAT IS STORED (`keep` levels; MiB/layer at TinyLlama seq 1024 batch 2)
# ----------------------------------------------------------------------
#   'full' 94.25 : x, x_norm_1, q_rope, k_rope, v, o_hidden, lse, x_mid, x_norm_2, h_gate, h_up
#                  -> recomputes only SiLU*mul. The fidelity reference and the speed ceiling;
#                     directly comparable with HyC-LoRA's `no_compress` 94.97.
#   'glu'  50.25 : the same minus h_gate/h_up -> recomputes the two [H, I] GLU GEMMs.
#                  This is FlashFFN-v3-`recompute` living inside a fused block, i.e. the
#                  "projected fused-exact-v3 ~= 4189 MiB" row of hyclora_flash_variant.md §7.4,
#                  now measured rather than projected.
#   'attn' 16.25 : x, o_hidden, lse. `min` plus FlashAttention's OUTPUT, and nothing else -- so
#                  the recompute re-runs every O(S) operation `min` re-runs (both norms, the six
#                  GEMMs, both RoPEs) but NEVER the O(S^2) FlashAttention forward. The one place
#                  in the block where O(S) bytes buy O(S^2) time; see the note above the counters.
#   'min'   8.00 : x ONLY. Everything else is recomputed in backward. This is the block-minimal
#                  point and it stores exactly what `torch.utils.checkpoint` at decoder-layer
#                  granularity stores -- the difference is that the recompute runs through THIS
#                  code (fused norms, native GQA, factored adapters, and the down-projection GEMM
#                  skipped because backward never needs the block output) instead of through the
#                  stock module graph.
#   'auto'       : resolve `min` below `_FB_AUTO_SEQ` tokens and `attn` at or above it, per
#                  forward, from the batch's own S. Announced, recorded, and pinnable.
#
# EXACTNESS
# ---------
# Every `keep` level runs the *same kernels on the same inputs* in the recompute as in the
# forward, so all four produce **bitwise-identical gradients to each other** (gate B3 in
# `src/verify_fused_block.py`). Recomputation here is not an approximation; the only deviation
# from stock HuggingFace is bf16 rounding order inside the fused RMSNorm and the factored adapter
# form, which is measured against an eager HF reference (gates B1/B2) rather than assumed.
#
# NOT REUSING HyC-LoRA'S KERNELS
# ------------------------------
# `hyclora/operators/rmsnorm_kernels.py`'s backward carries a LayerNorm mean term
# (`c2 = sum(wdy)/N`) that RMSNorm does not have, and accumulates a `dw` it does not need through
# an atomic lock. Both are wrong/wasteful for a frozen-norm LoRA setup, so the two kernels below
# are written from scratch: dx-only, no lock, no mean.

# THE O(S^2) RECOMPUTE, AND THE `attn` LEVEL THAT REMOVES IT  (added 2026-08-03)
# ------------------------------------------------------------------------------
# `keep='min'` re-executes the ENTIRE block forward in backward, and one operation in that forward
# is not O(S): the FlashAttention forward is O(S^2).  Every other recomputed operation -- the six
# GEMMs, the two norms, the two RoPEs, the layout fold -- is O(S).  Measured on TinyLlama-1.1B,
# batch 2, per training step (`src/diag_fb_breakdown.py`, kineto device time; the `flash_fwd`
# kernel is called 44x/step at `min` -- 22 output pass + 22 recompute -- at identical shapes, so
# the recompute share is exactly half):
#
#     seq            2048      8192     16384
#     flash_fwd      16.90    230.91    923.38  ms/step  (44 calls)
#     of which recompute  8.45    115.46    461.69       <- 4.00x per doubling above 8192
#
# At seq 16384 that 461.69 ms is 121% of our entire 382 ms throughput deficit against
# `hyclora_flash_nc`; excluding it we are 84 ms FASTER than that arm.  It is the whole crossover.
#
# `keep='attn'` stores FlashAttention's OUTPUT -- `o_h` and `lse`, 8.25 MiB/layer on top of `min`'s
# 8.00 -- and recomputes everything else exactly as `min` does.  q/k/v are still recomputed (they
# are O(S) and bitwise-identical, so FA's backward receives the same bits it would at `full`), but
# the O(S^2) forward is never re-run.  This is the "spend O(S) bytes to save O(S^2) time" trade,
# and it is the only place in the block where that trade exists.
_FB_COUNTERS = {"forward": 0, "backward": 0, "patched_layers": 0, "recompute": 0,
                "flash_recompute": 0, "dylora_rank_draws": 0,
                "keep_min": 0, "keep_attn": 0, "keep_glu": 0, "keep_full": 0}
_FB_KEEP_LEVELS = ("full", "glu", "attn", "min")            # what the Function executes
_FB_KEEP_POLICIES = _FB_KEEP_LEVELS + ("auto",)             # what `apply_flash_block` accepts

# `keep='auto'`: resolve `min` below this sequence length and `attn` at or above it, on EVERY
# forward, from the batch's own S.
#
# THE RULE, stated so it is not a fit to the measured shapes: switch at the largest power of two
# that is at most HALF the sequence length at which `min` leaves the 5% throughput budget.
# Measured (`holes_closed.md` 2026-08-03, regime B, batch 2, against `hyclora_flash_nc`), `min`'s
# deficit is -1.09% at seq 1024, +0.52% at 2048, +1.20% at 4096, +4.59% at 8192 and +8.98% at
# 16384, so it crosses +5% at seq ~= 8900 -- and 4096 <= 8900/2.  A threshold of 8192 would also
# meet the budget at every measured shape but leaves only 0.4 points of margin for an unmeasured
# shape just below it, which is less than the 0.5-0.7% session-to-session drift this project has
# measured; 4096 leaves 3.8.
#
# Overridable per call (`apply_flash_block(..., auto_seq=N)`) or by `FLASHFFN_FB_AUTO_SEQ` so the
# threshold can be pinned or swept for measurement without editing code.
_FB_AUTO_SEQ = int(os.environ.get("FLASHFFN_FB_AUTO_SEQ", "4096"))

# What the policy actually did, for the harness to print and record.  Never a silent choice.
_FB_POLICY = {"policy": None, "auto_seq": _FB_AUTO_SEQ, "resolved": {}, "announce": True}

# Adapter families the fused block executes, all through the SAME factored five-slot path.
# Anything else -- FourierFT, the Spectral (truncated-DCT) adapter, IA3, prefix tuning, an
# unrecognised PEFT wrapper -- RAISES.  There is deliberately no grad-checkpoint fallback: the v2
# path had one and it was mistaken for the real method once already (CONTEXT.md §16).
_FB_SUPPORTED_KINDS = ("plain", "lora", "dora", "adalora", "vera", "dylora")

# `FusedLoRABlockFunction.apply` argument layout, named once so `backward`'s return tuple and its
# `needs_input_grad` lookups can never drift out of step with `forward`'s signature:
#   [0..4]   x, w_norm1, w_norm2, cos, sin
#   [5..39]  seven projections (q k v o gate up down) x five slots (w, a, b, m, nu)
#   [40..46] seven projection biases
#   [47..53] seven bypass scales
#   [54..59] n_heads, n_kv_heads, head_dim, eps, keep, plan
_FB_W0 = 5
_FB_NPROJ = 7
_FB_NARGS = 5 + 5 * _FB_NPROJ + 7 + 7 + 6      # == 60
_FB_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fb_reset_counters():
    for k in _FB_COUNTERS:
        _FB_COUNTERS[k] = 0


def fb_get_counters():
    return dict(_FB_COUNTERS)


# ---------------------------------------------------------------------------------------------
# THE REMATERIALISATION CERTIFICATE  (`llmdocs/trackers/remat_certificate.md`)
# ---------------------------------------------------------------------------------------------
#
# Rematerialisation is redundant execution.  This block recomputes most of its own forward inside
# `backward` and then throws the comparison away.  The certificate recovers it.
#
# WHAT IS ASSUMED, EVERYWHERE, AND CHECKED NOWHERE
# ------------------------------------------------
# Every rematerialising system assumes the recomputed forward equals the executed one.  When that
# breaks -- RNG state, schedule state, autocast/precision state, a nondeterministic kernel, a lossy
# codec, or transient hardware corruption -- the gradients are wrong with no exception, no NaN and
# a still-decreasing loss.  Measured on this machine (`probe8_remat_divergence.py`): a same-shape
# value divergence costs +50.0% on a gradient and raises nothing; LoRA `dropout=0.1` under
# decoder-layer checkpointing with `preserve_rng_state=False` costs rel-L2 median 0.221 with the
# 20-step loss moving by 0.006; DyLoRA under the REENTRANT checkpointing HuggingFace defaults to
# costs rel-L2 median 0.749 at cosine 0.061 with the loss agreeing to four decimals.
#
# The only shipped checker anywhere is PyTorch's, and it compares METADATA:
# `torch/utils/checkpoint.py:1063` `_default_meta_extractor` -> {"shape","dtype","device"}, and it
# is disabled outright under `use_reentrant=True` (`:466`).  NVIDIA Transformer Engine has none and
# disclaims the hazard to the user (`pytorch/distributed.py:666-671`).  unsloth's checkpointer is a
# raw `autograd.Function`, so no framework check applies to it at all.
#
# THE WITNESS
# -----------
# Three digests per block, on a cut of the dataflow that dominates everything the recompute
# produces.  Each rides on a kernel that ALREADY reads the tensor, so the certificate adds no
# memory pass at all:
#
#   W0  `x`      block input        -- storage integrity of the one tensor `keep='min'` retains
#   W1  `x_mid`  attention residual -- rmsnorm1, q/k/v (+adapter factors), RoPE, FlashAttention,
#                                      the head->hidden fold, o_proj and the residual add
#   W2  `h_mid`  the GLU product    -- rmsnorm2, gate/up (+adapter factors), the SwiGLU kernel
#
# W0 and W1 are the two INPUTS of the two `fb_rmsnorm_forward` calls; W2 is the OUTPUT of
# `triton_silu_mul`.  All three kernels are ours and already hold the bytes in registers.
#
# WHY IT CAN BE EXACT RATHER THAN STATISTICAL
# -------------------------------------------
# The recomputed forward is BITWISE reproducible -- `probe2_seq1024.json`: 462 recomputed
# intermediates at seq 1024, 0 mismatches, and 3,014 across the seven adapter families.  (The
# shape-dependent noise floor recorded elsewhere in this repo is on GRADIENTS and is caused by
# FlashAttention's BACKWARD, which is downstream of every quantity compared here.)  So the
# comparison is exact equality of a digest: no threshold, no calibration, and a false-positive
# rate of zero.  `related_work.md` P4.4c.
#
# THE DIGEST
# ----------
# `sum_i mix32(bits_i XOR (i * PHI32))`, accumulated in int64.  Addition is commutative, so the
# value does not depend on Triton's block decomposition -- verified identical across BLOCK
# 256/1024/4096.  `mix32` (the murmur3 finaliser) is a bijection on uint32, so a change to ONE
# element always changes exactly one term of the sum and is detected with probability 1; a
# multi-element change collides only if the terms cancel exactly.  8 bytes per witness, i.e.
# 528 bytes for a 22-layer model.
#
# COST DISCIPLINE
# ---------------
# Digests are compared ON DEVICE into a counter tensor.  Nothing here reads a device value, so the
# certificate adds NO synchronisation per block.  The counter is read once per forward by the
# pre-hook `apply_flash_block` installs -- one sync per step -- which is also what makes a
# violation impossible to ignore: there is deliberately no mode in which a mismatch is recorded and
# never surfaced.  That would be precisely the silent failure this component exists to remove.
_FB_CERT = {
    "on": False,
    "fused": True,          # False = standalone digest kernels, the framework-level cost baseline
    "witnesses": ("x", "x_mid", "h_mid"),
}

# Device-side tallies.  `mismatch` is the only one whose value can fail a run.
_FB_CERT_STATE = {"mismatch": None, "compared": None, "device": None,
                  "blocks_certified": 0, "checks": 0, "raised": 0}

_FB_PHI32 = 0x9E3779B1


def fb_certify_enable(fused=True):
    """Turn the rematerialisation certificate on. Off by default; NR-1 requires that OFF is
    byte-identical to the uncertified block, so every line of this machinery is gated."""
    _FB_CERT["on"] = True
    _FB_CERT["fused"] = bool(fused)
    _FB_CERT_STATE.update(mismatch=None, compared=None, device=None,
                          blocks_certified=0, checks=0, raised=0)


def fb_certify_disable():
    # Tallies are cleared as well: a report taken after disabling must not show the PREVIOUS
    # arm's comparison counts, which is how an interleaved sweep ends up attributing 1650
    # comparisons to an arm that made none.
    _FB_CERT["on"] = False
    _FB_CERT_STATE.update(mismatch=None, compared=None, device=None,
                          blocks_certified=0, checks=0, raised=0)


def _fb_cert_tallies(device):
    if _FB_CERT_STATE["mismatch"] is None or _FB_CERT_STATE["device"] != device:
        _FB_CERT_STATE["mismatch"] = torch.zeros((), dtype=torch.int64, device=device)
        _FB_CERT_STATE["compared"] = torch.zeros((), dtype=torch.int64, device=device)
        _FB_CERT_STATE["device"] = device
    return _FB_CERT_STATE["mismatch"], _FB_CERT_STATE["compared"]


def fb_certify_report():
    """Read the tallies. Synchronises once."""
    m, c = _FB_CERT_STATE["mismatch"], _FB_CERT_STATE["compared"]
    return {"on": _FB_CERT["on"], "fused": _FB_CERT["fused"],
            "witnesses_compared": 0 if c is None else int(c),
            "witnesses_mismatched": 0 if m is None else int(m),
            "blocks_certified": _FB_CERT_STATE["blocks_certified"],
            "checks": _FB_CERT_STATE["checks"], "raised": _FB_CERT_STATE["raised"]}


class RematerialisationError(RuntimeError):
    """A recomputed or stored activation did not match the value the forward produced.

    The gradients of the step that raised this are wrong. Nothing about the loss, its finiteness,
    or its trajectory would have told you (`probe8_remat_divergence.py`)."""


def fb_certify_check():
    """Raise if any witness has mismatched since the last check. One synchronisation.

    Called automatically by the pre-hook `apply_flash_block` installs, so a mismatch surfaces at
    the next forward at the latest. It is deliberately not possible to run the certificate and
    silently discard its verdict."""
    if not _FB_CERT["on"] or _FB_CERT_STATE["mismatch"] is None:
        return 0
    _FB_CERT_STATE["checks"] += 1
    n = int(_FB_CERT_STATE["mismatch"])
    if n:
        _FB_CERT_STATE["raised"] += 1
        _FB_CERT_STATE["mismatch"].zero_()
        raise RematerialisationError(
            f"{n} activation witness(es) disagreed between the executed forward and the "
            f"backward's recompute. The gradients produced since the last check are WRONG. "
            f"Causes seen in practice: RNG state not restored across the recompute (dropout, "
            f"adapter dropout), a schedule or rank resampled per forward (DyLoRA, AdaLoRA), "
            f"autocast/precision state differing between the passes, a nondeterministic kernel "
            f"on the recompute path, a lossy codec on a saved activation, or transient memory "
            f"corruption. Re-run with flashffn.fb_certify_enable(fused=False) and "
            f"src/verification/probe2_exactness.py to localise the diverging tensor.")
    return 0


@triton.jit
def _fb_mix32(h):
    """murmur3 finaliser on uint32 -- a bijection, so no single-element change can be absorbed."""
    h = h ^ (h >> 16)
    h = h * 0x85EBCA6B
    h = h ^ (h >> 13)
    h = h * 0xC2B2AE35
    h = h ^ (h >> 16)
    return h


@triton.jit
def _fb_digest_kernel(P, OUT, n, BLOCK: tl.constexpr):
    """Standalone digest over a flat int16 view. This is the UNFUSED arm: it exists to measure
    what the same certificate costs a framework that cannot ride an existing kernel."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    # `P` is an int16 view, so the loaded value is SIGNED: a plain `.to(tl.uint32)` would
    # sign-extend and disagree with the fused witnesses, which bitcast from bf16 and zero-extend.
    # Caught by gate N1 (probe9), which exists precisely to stop the cheap implementation and the
    # reference one silently computing different functions.
    bits = tl.load(P + off, mask=m, other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
    t = _fb_mix32(bits ^ (off.to(tl.uint32) * 0x9E3779B1))
    tl.store(OUT + pid, tl.sum(tl.where(m, t, 0).to(tl.int64), axis=0))


def fb_digest(t):
    """Order-independent int64 digest of a contiguous 2-byte tensor. Returns a 0-dim device
    tensor; reading it would synchronise, and nothing on the hot path does."""
    assert t.is_contiguous(), "fb_digest needs a contiguous tensor"
    assert t.element_size() == 2, f"fb_digest is for 2-byte dtypes, got {t.dtype}"
    flat = t.view(torch.int16).reshape(-1)
    n = flat.numel()
    grid = triton.cdiv(n, 1024)
    out = torch.empty((grid,), dtype=torch.int64, device=t.device)
    _fb_digest_kernel[(grid,)](flat, out, n, BLOCK=1024)
    return out.sum()


def _fb_norm_w(x, w, eps, cert):
    """`fb_rmsnorm_forward` returning a third slot for the witness (None when off)."""
    if cert:
        return fb_rmsnorm_forward(x, w, eps, digest=True)
    y, rstd = fb_rmsnorm_forward(x, w, eps)
    return y, rstd, None


def _fb_glu_w(h_gate, h_up, cert):
    """`triton_silu_mul` returning a second slot for the witness (None when off)."""
    if cert:
        return triton_silu_mul(h_gate, h_up, digest=True)
    return triton_silu_mul(h_gate, h_up), None


def _fb_cert_compare(fwd, bwd):
    """Accumulate a device-side mismatch tally. No host read, no synchronisation."""
    if fwd is None or bwd is None:
        return
    mism, comp = _fb_cert_tallies(fwd.device)
    mism += (fwd != bwd).to(torch.int64)
    comp += 1


@triton.jit
def _fb_rmsnorm_fwd_kernel(X, Y, W, RSTD, DIG, stride, N, eps, BLOCK: tl.constexpr,
                           HAS_DIG: tl.constexpr):
    """y = w * bf16(x * rsqrt(mean(x^2) + eps)); stores rstd [M] fp32 only.

    The intermediate `x_hat` is rounded to bf16 BEFORE the weight multiply, which is what
    `LlamaRMSNorm.forward` does (`self.weight * hidden_states.to(input_dtype)`); matching that
    rounding point is what keeps the forward-parity gate at cosine ~1.

    `HAS_DIG` additionally folds a witness over the kernel's INPUT into the pass it is already
    making over it (witnesses W0 `x` and W1 `x_mid`, one per call site).  It is a `constexpr`, so
    with the certificate off Triton compiles this kernel to exactly the instructions it compiled
    before the certificate existed -- which is what NR-1 requires.
    """
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    xb = tl.load(X + cols, mask=mask, other=0.0)
    x = xb.to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(RSTD + row, rstd)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    xhat = (x * rstd).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + cols, (xhat * w).to(Y.dtype.element_ty), mask=mask)
    if HAS_DIG:
        # `row * N + cols` is the element's index in the flattened tensor, so a witness taken here
        # is bit-identical to one taken by the standalone `_fb_digest_kernel` -- which is what
        # lets the fused and unfused arms be cross-validated against each other.
        bits = xb.to(tl.uint16, bitcast=True).to(tl.uint32)
        t = _fb_mix32(bits ^ ((row * N + cols).to(tl.uint32) * 0x9E3779B1))
        tl.store(DIG + row, tl.sum(tl.where(mask, t, 0).to(tl.int64), axis=0))


@triton.jit
def _fb_rmsnorm_bwd_kernel(DX, DY, X, W, RSTD, stride, N, BLOCK: tl.constexpr):
    """dx = rstd * (w*dy - xhat * mean(xhat * w*dy)).

    RMSNorm has no mean-subtraction, hence no `c2 = mean(w*dy)` term. dw is not produced: under
    LoRA the norm weights are frozen, and the atomic-lock reduction that would compute it is the
    single most expensive part of a generic fused-RMSNorm backward.
    """
    row = tl.program_id(0)
    X += row * stride
    DY += row * stride
    DX += row * stride
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    xhat = x * rstd
    wdy = w * dy
    c1 = tl.sum(tl.where(mask, xhat * wdy, 0.0), axis=0) / N
    dx = (wdy - xhat * c1) * rstd
    tl.store(DX + cols, dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def _fb_rmsnorm_dw_kernel(DWP, DY, X, RSTD, stride, M, N, BLOCK: tl.constexpr):
    """Partial reduction for dw = sum_rows(xhat * dy), xhat rounded through bf16 exactly as the
    forward kernel rounds it.

    Only needed for FULL fine-tuning (under every adapter family the norm weights are frozen and
    this kernel never runs). Written as a grid-stride partial reduction into a [G, N] fp32 buffer
    rather than the atomic-lock accumulation a generic fused RMSNorm backward uses: no lock, no
    serialisation, and the final [G, N] -> [N] sum is 512 KiB of traffic at G=128.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    row = pid
    while row < M:
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)
        acc += (x * rstd).to(tl.bfloat16).to(tl.float32) * dy
        row += nprog
    tl.store(DWP + pid * N + cols, acc, mask=mask)


def _fb_norm_settings(n):
    block = triton.next_power_of_2(n)
    if block > 65536 // 2:
        raise RuntimeError(f"fused block RMSNorm: hidden dim {n} too large")
    return block, min(max(block // 256, 1), 16)


def fb_rmsnorm_forward(x, w, eps, digest=False):
    """x [..., N] contiguous -> (y same shape/dtype, rstd [M] fp32).

    `digest=True` additionally returns an int64 witness over `x`.  When the certificate is in
    `fused` mode it comes out of the kernel above at no extra memory pass; in `standalone` mode it
    costs a separate full read, which is the framework-level cost baseline this component is
    measured against.
    """
    x2 = x.reshape(-1, x.shape[-1])
    m, n = x2.shape
    y = torch.empty_like(x)
    rstd = torch.empty((m,), dtype=torch.float32, device=x.device)
    block, warps = _fb_norm_settings(n)
    fused = digest and _FB_CERT["fused"] and x.element_size() == 2 and x2.stride(0) == n
    dig_buf = torch.empty((m,), dtype=torch.int64, device=x.device) if fused else None
    _fb_rmsnorm_fwd_kernel[(m,)](x2, y.reshape(-1, n), w, rstd, dig_buf, x2.stride(0), n, eps,
                                 BLOCK=block, num_warps=warps, HAS_DIG=fused)
    if not digest:
        return y, rstd
    return y, rstd, (dig_buf.sum() if fused else fb_digest(x2))


def fb_rmsnorm_backward(dy, x, w, rstd, need_dw=False):
    """dx (and, only under full fine-tuning, dw). `need_dw=False` is the LoRA-family path and is
    byte-for-byte the kernel sequence this function has always run."""
    x2 = x.reshape(-1, x.shape[-1])
    m, n = x2.shape
    dx = torch.empty_like(x2)
    block, warps = _fb_norm_settings(n)
    dy2 = dy.reshape(-1, n)
    _fb_rmsnorm_bwd_kernel[(m,)](dx, dy2, x2, w, rstd, x2.stride(0), n,
                                 BLOCK=block, num_warps=warps)
    if not need_dw:
        return dx.view_as(x)
    groups = min(m, 128)
    dwp = torch.empty((groups, n), dtype=torch.float32, device=x.device)
    _fb_rmsnorm_dw_kernel[(groups,)](dwp, dy2, x2, rstd, x2.stride(0), m, n,
                                     BLOCK=block, num_warps=warps)
    return dx.view_as(x), dwp.sum(0).to(w.dtype)


@triton.jit
def _fb_rope_kernel(X, Y, COS, SIN, sx0, sx1, sx2, sy0, sy1, sy2,
                    n_elem, S, H, D: tl.constexpr, HALF: tl.constexpr,
                    BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    """RoPE on half-pairs, with an arbitrary in-layout and an arbitrary out-layout.

    Element index runs over (b, s, h, c) with c < D/2, so the SAME kernel serves both
    [B, S, H, D] (contiguous, what a projection produces) and [B, H, S, D] (what FlashAttention
    produces) by choosing strides -- which is how the RoPE and the head<->hidden layout change are
    fused into one pass in backward instead of being a rotate + a transposing copy.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    c = idx % HALF
    t = idx // HALF
    h = t % H
    t2 = t // H
    s = t2 % S
    b = t2 // S
    i1 = b * sx0 + s * sx1 + h * sx2 + c
    o1 = b * sy0 + s * sy1 + h * sy2 + c
    x1 = tl.load(X + i1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + i1 + HALF, mask=mask, other=0.0).to(tl.float32)
    cs = s * D + c
    c1 = tl.load(COS + cs, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cs + HALF, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cs, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cs + HALF, mask=mask, other=0.0).to(tl.float32)
    if BACKWARD:
        # adjoint of (y1, y2) = (x1*c1 - x2*s1, x2*c2 + x1*s2)
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    tl.store(Y + o1, y1.to(Y.dtype.element_ty), mask=mask)
    tl.store(Y + o1 + HALF, y2.to(Y.dtype.element_ty), mask=mask)


def _fb_rope_bsh(x, cos, sin):
    """x [B, S, H, D] contiguous -> new [B, S, H, D] contiguous. `cos`/`sin` are [S, D].

    Working in the [B, S, H, D] layout (not [B, H, S, D]) is deliberate: it is exactly what the
    q/k projection produces, so RoPE needs no contiguous copy, and FlashAttention takes the
    `.transpose(1, 2)` VIEW of the result -- the same thing stock transformers hands to sdpa.
    """
    b, s, h, d = x.shape
    y = torch.empty_like(x)
    n = b * s * h * (d // 2)
    BLOCK = 512
    _fb_rope_kernel[(triton.cdiv(n, BLOCK),)](
        x, y, cos, sin, s * h * d, h * d, d, s * h * d, h * d, d,
        n, s, h, D=d, HALF=d // 2, BACKWARD=False, BLOCK=BLOCK, num_warps=4)
    return y


def _fb_rope_bwd_to_bsh(g, cos, sin, b, s, h, d):
    """g is FlashAttention's [B, H, S, D] gradient; returns [B*S, H*D] contiguous.

    The rotate-adjoint and the head->hidden layout change happen in ONE kernel, so the 8 MiB
    transposing copy that a `.transpose(1,2).reshape(...)` would allocate never exists.
    """
    out = torch.empty((b * s, h * d), dtype=g.dtype, device=g.device)
    n = b * s * h * (d // 2)
    BLOCK = 512
    gs = g.stride()
    _fb_rope_kernel[(triton.cdiv(n, BLOCK),)](
        g, out, cos, sin, gs[0], gs[2], gs[1], s * h * d, h * d, d,
        n, s, h, D=d, HALF=d // 2, BACKWARD=True, BLOCK=BLOCK, num_warps=4)
    return out


@triton.jit
def _fb_head_to_hidden_kernel(X, Y, sx0, sx1, sx2, S, H, D: tl.constexpr, n_elem,
                              BLOCK: tl.constexpr):
    """[B, H, S, D] (any strides) -> [B*S, H*D] contiguous. Used for v's gradient, which needs
    the layout change but not the rotation."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elem
    c = idx % D
    t = idx // D
    h = t % H
    t2 = t // H
    s = t2 % S
    b = t2 // S
    val = tl.load(X + b * sx0 + s * sx1 + h * sx2 + c, mask=mask, other=0.0)
    tl.store(Y + (b * S + s) * (H * D) + h * D + c, val, mask=mask)


def _fb_head_to_hidden(x, b, s, h, d):
    out = torch.empty((b * s, h * d), dtype=x.dtype, device=x.device)
    n = b * s * h * d
    BLOCK = 1024
    xs = x.stride()
    _fb_head_to_hidden_kernel[(triton.cdiv(n, BLOCK),)](
        x, out, xs[0], xs[2], xs[1], s, h, d, n, BLOCK=BLOCK, num_warps=4)
    return out


def _fb_flash_fwd(q, k, v, scale):
    """PyTorch's vendored FlashAttention-2 forward, GQA-native (k/v keep num_key_value_heads).

    Same entry point `hyclora/layers/fused_llama_layer_intra_inter_flash.py` uses, so the two
    fused arms are attention-kernel-matched by construction (protocol §C.2 R1).
    """
    return torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True, False,
                                                              scale=scale)


def _fb_flash_bwd(go, q, k, v, o, lse, cum_q, cum_k, max_q, max_k, seed, off, scale):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        go, q, k, v, o, lse, cum_q, cum_k, max_q, max_k, 0.0, True, seed, off, scale=scale)


# ---------------------------------------------------------------------------------------------
# Padded / ragged batches: FlashAttention-2 VARLEN.
#
# The dense `_scaled_dot_product_flash_attention` entry point above expresses causality with a
# flag and has nowhere to put per-sample lengths, so a padded batch cannot be handled by it -- the
# reason the first version of this block rejected padded batches outright.  `aten::
# _flash_attention_forward` is the same kernel with the varlen signature (cu_seqlens + packed
# [T, H, D] tensors), which is exactly how FlashAttention-2 is meant to consume ragged batches.
#
# Everything else in the block is per-token (RMSNorm, all seven projections, SiLU*mul, both
# residuals), so ONLY the attention call needs to know about padding: q/k/v are gathered down to
# the valid tokens, the varlen kernel runs, and the output is scattered back into a zero-filled
# [B*S, H*D] buffer.  Pad rows therefore carry a zero attention output instead of the arbitrary
# value stock HF produces for them (transformers' `_unmask_unattended` lets a fully-masked query
# row attend to everything to dodge a NaN).  That difference is confined to pad positions and can
# never reach a valid one: a valid query attends only to valid keys, so no gradient flows back
# into a pad position unless the caller puts a LABEL there.  See §"padding" in
# `llmdocs/trackers/fused_block.md`.
# ---------------------------------------------------------------------------------------------

class _FbVarlenPlan:
    """Gather/scatter plan for one padded batch. Built from the attention mask, once per mask."""
    __slots__ = ("idx", "cu", "max_len", "total", "lens")

    def __init__(self, idx, cu, max_len, total, lens):
        self.idx = idx            # [T] int64 flat indices into [B*S]
        self.cu = cu              # [B+1] int32 cumulative sequence lengths
        self.max_len = max_len    # int
        self.total = total        # int, == T
        self.lens = lens          # [B] int32 (diagnostics / error messages)


def _fb_varlen_fwd(q, k, v, scale, plan):
    """Packed FlashAttention-2 forward. q/k/v are [B, S, H, D] contiguous.

    Returns (o_h [B*S, H*D] with zeros on pad rows, lse, meta) so the caller cannot tell the
    varlen path from the dense one by shape.
    """
    b, s, h, d = q.shape
    idx, cu, max_len = plan.idx, plan.cu, plan.max_len
    qp = q.reshape(b * s, h, d).index_select(0, idx)
    kp = k.reshape(b * s, k.shape[2], d).index_select(0, idx)
    vp = v.reshape(b * s, v.shape[2], d).index_select(0, idx)
    o_p, lse, seed, off, _ = torch.ops.aten._flash_attention_forward(
        qp, kp, vp, cu, cu, max_len, max_len, 0.0, True, False, scale=scale)
    o_full = q.new_zeros((b * s, h, d))
    o_full.index_copy_(0, idx, o_p)
    return o_full.view(b * s, h * d), lse, (cu, cu, max_len, max_len, seed, off)


def _fb_varlen_bwd(grad_o_h, q, k, v, o_h, lse, meta, scale, plan):
    """Packed FlashAttention-2 backward. Returns dq/dk/dv as [B, H, S, D] (a permuted VIEW of a
    [B, S, H, D] buffer), i.e. the same layout the dense kernel returns, so the RoPE-adjoint and
    head->hidden kernels downstream are shared verbatim."""
    b, s, h, d = q.shape
    kvh = k.shape[2]
    cum_q, cum_k, max_q, max_k, seed, off = meta
    idx = plan.idx
    qp = q.reshape(b * s, h, d).index_select(0, idx)
    kp = k.reshape(b * s, kvh, d).index_select(0, idx)
    vp = v.reshape(b * s, kvh, d).index_select(0, idx)
    op = o_h.reshape(b * s, h, d).index_select(0, idx)
    gop = grad_o_h.reshape(b * s, h, d).index_select(0, idx)
    gq, gk, gv = torch.ops.aten._flash_attention_backward(
        gop, qp, kp, vp, op, lse, cum_q, cum_k, max_q, max_k, 0.0, True, seed, off, scale=scale)

    def scatter(t, nh):
        out = t.new_zeros((b * s, nh, d))
        out.index_copy_(0, idx, t)
        return out.view(b, s, nh, d).permute(0, 2, 1, 3)
    return scatter(gq, h), scatter(gk, kvh), scatter(gv, kvh)


def _fb_build_mask_plan(mask, b, s_len, device):
    """Turn an attention mask into either None (plain unpadded causal -> dense path) or a
    `_FbVarlenPlan`.  Raises on anything that is not `causal AND key-is-valid`.

    A 4-D additive mask is validated *structurally*: the key-validity vector is read off the last
    query row and the whole mask is then required to equal `tril & key_valid` on every query row
    that is not fully masked (fully-masked rows are the pad rows, whose content transformers
    rewrites via `_unmask_unattended`; they are required to be all-attend or all-masked and are
    otherwise ignored).  A sliding-window mask, a packed/block-diagonal mask, a per-head mask or a
    prefix-LM mask all fail this test and raise -- none of them can be silently accepted.
    """
    if not torch.is_tensor(mask):
        raise RuntimeError(
            f"FusedLoRABlock: attention_mask is a {type(mask).__name__}, not a Tensor "
            f"(a flex-attention BlockMask, for instance). The fused block only implements a "
            f"dense `causal AND key-validity` mask.")
    if mask.dim() == 2:
        if tuple(mask.shape) != (b, s_len):
            raise RuntimeError(f"FusedLoRABlock: 2-D attention_mask {tuple(mask.shape)} does not "
                               f"match the batch {(b, s_len)}.")
        keep = mask.to(torch.bool)
    elif mask.dim() == 4:
        if mask.shape[0] != b or mask.shape[2] != s_len:
            raise RuntimeError(f"FusedLoRABlock: 4-D attention_mask {tuple(mask.shape)} does not "
                               f"match the batch {(b, s_len)}.")
        if mask.shape[1] != 1:
            raise RuntimeError(f"FusedLoRABlock: per-head attention mask (shape "
                               f"{tuple(mask.shape)}); the fused block only implements a shared "
                               f"causal+padding mask.")
        if mask.shape[3] != s_len:
            raise RuntimeError(f"FusedLoRABlock: kv length {mask.shape[3]} != query length "
                               f"{s_len}; a KV cache / static cache is not supported in training.")
        allowed = mask[:, 0] if mask.dtype == torch.bool else (mask[:, 0] == 0)
        keep = allowed[:, -1, :].clone()
        causal = torch.tril(torch.ones(s_len, s_len, dtype=torch.bool, device=device))
        expect = causal.unsqueeze(0) & keep.unsqueeze(1)
        live = expect.any(-1)                                  # [B, S] non-empty query rows
        row_ok = (allowed == expect).all(-1)                   # exact match where it matters
        dead_ok = allowed.all(-1) | ~allowed.any(-1)           # pad rows: all-attend or all-masked
        if not bool((row_ok | (~live & dead_ok)).all()):
            raise RuntimeError(
                "FusedLoRABlock: the 4-D attention mask is not `causal AND key-validity`. "
                "Sliding-window, block-diagonal/packed, prefix-LM and per-head masks are not "
                "implemented by the fused block and must not be silently ignored.")
    else:
        raise RuntimeError(f"FusedLoRABlock: unexpected attention_mask rank {mask.dim()}.")

    if bool(keep.all()):
        return None                                            # unpadded -> dense causal path
    lens = keep.sum(1).to(torch.int32)
    if int(lens.min()) == 0:
        raise RuntimeError("FusedLoRABlock: a sample in this batch has zero unmasked tokens; "
                           "FlashAttention-2 varlen cannot represent an empty sequence.")
    cu = torch.zeros(b + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens, 0, out=cu[1:])
    idx = keep.reshape(-1).nonzero(as_tuple=True)[0]
    return _FbVarlenPlan(idx, cu, int(lens.max()), int(idx.numel()), lens)


# One-entry memo so the plan is built once per mask TENSOR rather than once per layer. The key is
# the tensor's identity AND its autograd version counter, so a different mask -- or the same
# object mutated in place -- always rebuilds. This is emphatically NOT the "validate once per
# layer and latch" bug it replaces: nothing here survives a change of mask.
_FB_MASK_MEMO = {"ref": None, "key": None, "plan": None}


def _fb_mask_plan(mask, b, s_len, device):
    """Called on EVERY fused forward. `mask is None` (the unpadded case, which is what
    transformers hands the layer when the batch has no padding) costs one `is None` test."""
    if mask is None:
        return None
    memo = _FB_MASK_MEMO
    ref = memo["ref"]
    key = (b, s_len, mask._version)
    if ref is not None and ref() is mask and memo["key"] == key:
        return memo["plan"]
    plan = _fb_build_mask_plan(mask, b, s_len, device)
    try:
        memo["ref"] = weakref.ref(mask)
    except TypeError:                                          # non-weakref-able tensor subclass
        memo["ref"] = None
    memo["key"] = key
    memo["plan"] = plan
    return plan


def _fb_dora_scale(m, nu):
    """DoRA's per-output-channel scale c = m / ||W + s*B@A||.

    Computed from the two SAVED tensors with the SAME op in forward and in backward, so the two
    are bitwise equal (the same construction `FlashFFNv3AdapterFunction` uses). `nu` arrives
    already detached -- PEFT detaches the column norm, and the Gram identity that produces it
    (`_v3_dora_col_norm`) runs under `no_grad` -- so no gradient flows through it.
    """
    return None if m is None else (m / nu)


def _fb_proj(x2, w, a, b, s, bias=None, c=None, keep_raw=False):
    """Factored projection: y = x@W^T + s*(x@A^T)@B^T. Returns (y, xa, raw) with xa the thin [N, r]
    intermediate that `grad_b` needs. The merged weight W + s*B@A is never formed.

    `c` is DoRA's per-output-channel magnitude scale (None for every other family): the projection
    output becomes `c * (x@W^T + bias + s*(x@A^T)@B^T)`, which is what PEFT's `DoraLinearLayer`
    computes once its `(mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result` is
    collected onto the base result. It is applied AFTER the bypass and after the bias, matching
    PEFT.  `keep_raw=True` additionally returns the PRE-`c` value, which is the only thing DoRA's
    magnitude gradient needs (`dL/dm_j = sum_rows(dL/dh_j * raw_j) / nu_j`); the pre-`c` value is
    never recoverable from the post-`c` one for q/k/v because RoPE mixes channels afterwards, so
    it is recomputed rather than divided back out -- identically at every `keep` level, which is
    what keeps gate B3's bitwise agreement across keep levels true for DoRA as well.

    The bypass is folded in with an in-place `addmm_` rather than `y = y + s * F.linear(xa, b)`.
    At [N, 5632] that replaces three full-size allocations and three passes over the tensor with
    one read-modify-write, which is worth ~9 ms/step across the model. With `s == 1` (the
    lora_alpha == r case every arm here runs) the two are also *more* accurate, not just faster:
    the bypass product is accumulated in fp32 and rounded to bf16 once instead of twice.

    `bias` is the projection's own (frozen) bias, threaded through so biased architectures --
    Qwen2's q/k/v bias, any `attention_bias=True` / `mlp_bias=True` config -- are computed, not
    silently dropped. It plays no part in backward: d(x@W^T + bias)/dx is W, and the bias itself
    is required to be frozen at patch time.
    """
    y = F.linear(x2, w, bias)
    xa = None
    if a is not None:
        xa = F.linear(x2, a)
        y.addmm_(xa, b.t(), alpha=s)
    raw = None
    if c is None:
        if keep_raw:
            raw = y
    elif keep_raw:
        raw = y
        y = y * c.unsqueeze(0)
    else:
        y = y.mul_(c.unsqueeze(0))
    return y, xa, raw


def _fb_acc(acc, m1, m2, alpha=1.0):
    """acc += alpha * (m1 @ m2), in place, allocating only on the first term."""
    if acc is None:
        return (m1 @ m2) if alpha == 1.0 else alpha * (m1 @ m2)
    acc.addmm_(m1, m2, alpha=alpha)
    return acc


class FusedLoRABlockFunction(torch.autograd.Function):
    """One autograd node for a whole SwiGLU/GQA LLaMA decoder block under any of the six
    supported adapter families.

    apply(x, w_norm1, w_norm2, cos, sin,
          wq, aq, bq, mq, nuq,  wk, ak, bk, mk, nuk,  wv, av, bv, mv, nuv,
          wo, ao, bo, mo, nuo,  wg, ag, bg, mg, nug,  wu, au, bu, mu, nuu,
          wd, ad, bd, md, nud,
          biq, bik, biv, bio, big, biu, bid,
          sq, sk, sv, so, sg, su, sd,
          n_heads, n_kv_heads, head_dim, eps, keep, plan)

    Each of the seven projections is described by five slots -- base weight `w`, thin bypass
    factors `a` [r, in] / `b` [out, r], DoRA magnitude `m` [out] and its DETACHED column norm
    `nu` [out] -- and one scalar bypass scale `s`.  Every adapter family reduces to that shape
    OUTSIDE this Function (`_fb_factors`), which is what keeps one code path for all of them:

      LoRA      a = lora_A,             b = lora_B,             s = alpha/r
      DoRA      the same, plus m = magnitude and nu = ||W + s*B@A|| (Gram identity, no B@A)
      AdaLoRA   a = lora_A * lora_E,    b = lora_B,             s = scaling/(ranknum + 1e-5)
      DyLoRA    a = lora_A[:b_r, :],    b = lora_B[:, :b_r],    s = alpha/r   (b_r sampled ONCE)
      VeRA      a = lam_d[:,None]*A_f,  b = lam_b[:,None]*B_f,  s = 1
      full FT / frozen                  a = b = None, and `w` itself may require grad

    Composing outside means autograd runs the derived thin chain rules (grad_E, grad_lambda,
    DyLoRA's zero-padding SliceBackward) on [r, in] / [out, r] tensors only, and -- load-bearing
    for this block -- it means the sampled DyLoRA rank and the current AdaLoRA rank mask are
    baked into the SAVED tensors, so the `keep='min'` recompute cannot draw a different rank or
    read a re-masked `lora_E` than the output pass did.

    Gradients are produced for `a`/`b` (all families), `m` (DoRA), `w` (full FT only, and then
    the dense [out, in] grad IS the parameter's own gradient, not an intermediate) and the two
    norm weights (full FT only).  The seven `bi*` projection biases and `nu` never receive one --
    biases are required to be frozen at patch time, `nu` is detached by construction.
    """

    @staticmethod
    def forward(ctx, x, w_norm1, w_norm2, cos, sin,
                wq, aq, bq, mq, nuq, wk, ak, bk, mk, nuk, wv, av, bv, mv, nuv,
                wo, ao, bo, mo, nuo, wg, ag, bg, mg, nug, wu, au, bu, mu, nuu,
                wd, ad, bd, md, nud,
                biq, bik, biv, bio, big, biu, bid,
                sq, sk, sv, so, sg, su, sd,
                n_heads, n_kv_heads, head_dim, eps, keep, plan):
        assert keep in _FB_KEEP_LEVELS, keep
        _FB_COUNTERS["forward"] += 1
        b, s_len, hidden = x.shape
        scale = head_dim ** -0.5
        kv_dim = n_kv_heads * head_dim
        cq, ck, cv, co = (_fb_dora_scale(mq, nuq), _fb_dora_scale(mk, nuk),
                          _fb_dora_scale(mv, nuv), _fb_dora_scale(mo, nuo))
        cg, cu, cd = (_fb_dora_scale(mg, nug), _fb_dora_scale(mu, nuu),
                      _fb_dora_scale(md, nud))

        # The three witnesses (`_FB_CERT`) ride on kernels this forward already runs, so with the
        # certificate off `cert` is False and every call below is the call it always was.
        cert = _FB_CERT["on"]

        # ---------------------------------------------------------------- attention
        xn1, rstd1, d_x = _fb_norm_w(x, w_norm1, eps, cert)          # W0: the block input
        xn1_2 = xn1.view(-1, hidden)
        q, xa_q, _ = _fb_proj(xn1_2, wq, aq, bq, sq, biq, cq)
        k, xa_k, _ = _fb_proj(xn1_2, wk, ak, bk, sk, bik, ck)
        v, xa_v, _ = _fb_proj(xn1_2, wv, av, bv, sv, biv, cv)
        # RoPE runs in the [B, S, H, D] layout the projection already produced, so it needs no
        # contiguous copy; FlashAttention then takes the transposed VIEW, which is exactly what
        # stock transformers hands to sdpa (minus the `repeat_kv` pre-expansion and the
        # `query.contiguous()` that `sdpa_attention.py` adds).
        q = _fb_rope_bsh(q.view(b, s_len, n_heads, head_dim), cos, sin)
        k = _fb_rope_bsh(k.view(b, s_len, n_kv_heads, head_dim), cos, sin)
        v = v.view(b, s_len, n_kv_heads, head_dim)
        if plan is None:
            o, lse, cum_q, cum_k, max_q, max_k, seed, off, _ = _fb_flash_fwd(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale)
            # ONE bf16 copy of the attention output ever exists: FA's [B,H,S,D] result is folded
            # into the [B*S, H*D] layout by a single kernel and immediately released.
            o_h = _fb_head_to_hidden(o, b, s_len, n_heads, head_dim)
            del o
            flash_meta = (cum_q, cum_k, max_q, max_k, seed, off)
        else:
            o_h, lse, flash_meta = _fb_varlen_fwd(q, k, v, scale, plan)
        # ------------------------------------------------- `o_h` offload (route_b.md sections 2.6/2.7)
        # `keep='attn'` buys the O(S^2) FlashAttention forward out of the backward by STORING `o_h`,
        # and that storage is what makes it lose the memory axis.  Measured on this box
        # (`probe_stream_overlap.py`): the SMs have no headroom to hide the recompute in (all four
        # stream pairings NEGATIVE at seq 16384) while the copy engines are idle and carry `o_h` for
        # free (all four pairings >= 0.989).  So `o_h` goes to pinned host memory here and comes
        # back one layer ahead of where the backward needs it.
        #
        # THE MECHANISM IS MEMO'S (arXiv 2407.12117), which offloads exactly this tensor by exactly
        # this reasoning.  It is cited in the method section, not related work.  See
        # `related_work.md` SCOPING PASS 5.
        #
        # Issued HERE rather than beside `save_for_backward` so the transfer has the whole rest of
        # the block's forward to drain in.  `o_h` is still read by `_fb_proj` on the next line --
        # safe, because both streams are only READING it and `fb_offload.stash` marks it with
        # `record_stream` so the allocator cannot recycle it under either.
        o_h_handle = None
        if keep == "attn" and _fb_offload.fb_offload_enabled():
            o_h_handle = _fb_offload.stash(o_h)
        attn_out, xa_o, _ = _fb_proj(o_h, wo, ao, bo, so, bio, co)
        x_mid = x + attn_out.view(b, s_len, hidden)
        del attn_out

        # ---------------------------------------------------------------- FFN
        # W1: `x_mid` is the whole attention half's output, so this one witness covers rmsnorm1,
        # q/k/v and their adapter factors, RoPE, FlashAttention, the head->hidden fold, o_proj and
        # the residual add -- and it costs nothing, because rmsnorm2 is already reading it.
        xn2, rstd2, d_xmid = _fb_norm_w(x_mid, w_norm2, eps, cert)
        xn2_2 = xn2.view(-1, hidden)
        h_gate, xa_g, _ = _fb_proj(xn2_2, wg, ag, bg, sg, big, cg)
        h_up, xa_u, _ = _fb_proj(xn2_2, wu, au, bu, su, biu, cu)
        h_mid, d_hmid = _fb_glu_w(h_gate, h_up, cert)                 # W2: the GLU product
        y, xa_d, _ = _fb_proj(h_mid, wd, ad, bd, sd, bid, cd)
        del h_mid
        x_out = x_mid + y.view(b, s_len, hidden)
        del y

        keep_path = keep in ("full", "glu")        # the whole attention PATH is stored
        keep_out = keep in ("full", "glu", "attn")  # FlashAttention's OUTPUT (o_h, lse) is stored
        if keep != "full":
            del h_gate, h_up
            h_gate = h_up = None
        if not keep_path:
            del xn1, q, k, v, xn2, x_mid
            xn1 = q = k = v = xn2 = x_mid = None
            rstd1 = rstd2 = None
            xa_q = xa_k = xa_v = xa_o = xa_g = xa_u = xa_d = None
        if keep_out:
            # `attn` keeps exactly these two on top of `min`'s single tensor: 8.25 MiB/layer at
            # seq 1024 batch 2, which is what buys the O(S^2) forward out of the backward.
            ctx.flash_meta = flash_meta
            if o_h_handle is not None:
                # Offloaded: `lse` stays resident (it is 0.25 of the 8.25 MiB/layer) and only `o_h`
                # travels, so what `save_for_backward` receives below is None -- exactly what
                # `keep='min'` already passes.  The handle rides on `ctx`, like `flash_meta`: it is
                # not an autograd input or output and must not join the version-counter bookkeeping.
                o_h = None
        else:
            del o_h, lse
            o_h = lse = None
            ctx.flash_meta = None

        # ------------------------------------------------- WP-E: base weights are NOT saved when
        # they are being streamed.  `SavedVariable` takes a shallow copy that holds the STORAGE, so
        # a staged landing slab saved here would stay alive from this forward to this block's
        # backward -- all 22 layers resident, i.e. no saving at all.  The seven frozen `w` are
        # therefore saved as `None` and the backward re-acquires them from `fb_wstream`, which is
        # sound precisely because they are FROZEN: nothing writes them (so no version counter is
        # needed) and they produce no gradient.  `install()` refuses a model with a trainable base
        # weight, so there is no configuration in which this drops a `w` that backward needs to
        # differentiate.
        _wsc = _fb_wstream.current()
        ctx.wstream = _wsc
        if _wsc is None:
            _sq, _sk, _sv, _so, _sg, _su, _sd = wq, wk, wv, wo, wg, wu, wd
        else:
            _sq = _sk = _sv = _so = _sg = _su = _sd = None
        ctx.save_for_backward(
            x, w_norm1, w_norm2, cos, sin,
            _sq, aq, bq, mq, nuq, _sk, ak, bk, mk, nuk, _sv, av, bv, mv, nuv,
            _so, ao, bo, mo, nuo, _sg, ag, bg, mg, nug, _su, au, bu, mu, nuu,
            _sd, ad, bd, md, nud,
            biq, bik, biv, bio, big, biu, bid,
            xa_q, xa_k, xa_v, xa_o, xa_g, xa_u, xa_d,
            xn1, q, k, v, o_h, lse, x_mid, xn2, h_gate, h_up, rstd1, rstd2,
        )
        # 24 bytes per block. Held on `ctx` rather than in `save_for_backward` because these are
        # not inputs and must not participate in the autograd version-counter bookkeeping.
        ctx.cert = (d_x, d_xmid, d_hmid) if cert else None
        ctx.o_h_handle = o_h_handle
        ctx.shape = (b, s_len, hidden, kv_dim)
        ctx.heads = (n_heads, n_kv_heads, head_dim)
        ctx.scales = (sq, sk, sv, so, sg, su, sd)
        ctx.eps = eps
        ctx.keep = keep
        ctx.scale = scale
        ctx.plan = plan
        return x_out

    @staticmethod
    def backward(ctx, grad_out):
        _FB_COUNTERS["backward"] += 1
        (x, w_norm1, w_norm2, cos, sin,
         wq, aq, bq, mq, nuq, wk, ak, bk, mk, nuk, wv, av, bv, mv, nuv,
         wo, ao, bo, mo, nuo, wg, ag, bg, mg, nug, wu, au, bu, mu, nuu,
         wd, ad, bd, md, nud,
         biq, bik, biv, bio, big, biu, bid,
         xa_q, xa_k, xa_v, xa_o, xa_g, xa_u, xa_d,
         xn1, q, k, v, o_h, lse, x_mid, xn2, h_gate, h_up, rstd1, rstd2) = ctx.saved_tensors
        b, s_len, hidden, kv_dim = ctx.shape
        n_heads, n_kv_heads, head_dim = ctx.heads
        sq, sk, sv, so, sg, su, sd = ctx.scales
        keep, scale, eps = ctx.keep, ctx.scale, ctx.eps
        plan = ctx.plan

        # WP-E: the seven frozen base weights were saved as `None`; bring this layer's slab back
        # before anything reads them.  `bwd_enter` also issues the H2D for the NEXT block the
        # backward will reach (layers run in reverse, so that is layer i-1), which is what gives
        # the transfer a whole layer backward to arrive in.  If it has not arrived, the compute
        # stream waits on the event -- a slowdown, not a wrong answer -- and
        # `fb_wstream_stats()['blocked_acquires']` counts every time it happens, so a serialised
        # run cannot be reported as an overlapped one.
        _ws = getattr(ctx, "wstream", None)
        if _ws is not None:
            _wsx, _wsi = _ws
            wq, wk, wv, wo, wg, wu, wd = _wsx.bwd_enter(_wsi)

        ngi = ctx.needs_input_grad
        # slot layout: 5 leading, then 5 slots (w, a, b, m, nu) per projection in q k v o g u d
        nw = [ngi[_FB_W0 + 5 * p] for p in range(7)]      # dense weight grad  (full FT only)
        nm = [ngi[_FB_W0 + 3 + 5 * p] for p in range(7)]  # DoRA magnitude grad
        ndw1, ndw2 = ngi[1], ngi[2]                       # norm weights       (full FT only)
        cq, ck, cv, co = (_fb_dora_scale(mq, nuq), _fb_dora_scale(mk, nuk),
                          _fb_dora_scale(mv, nuv), _fb_dora_scale(mo, nuo))
        cg, cu, cd = (_fb_dora_scale(mg, nug), _fb_dora_scale(mu, nuu),
                      _fb_dora_scale(md, nud))

        # ------------------------------------------------------------- recompute (keep='min')
        # Same kernels, same inputs, same order as forward => bitwise-identical intermediates
        # (gate B3). The DOWN projection is deliberately absent: backward needs `h_mid` but never
        # the block output, and `down` is 26% of the block's GEMM work.
        raw_q = raw_k = raw_v = raw_o = raw_g = raw_u = None
        # `ctx.cert` is the forward's witness triple, or None. It is read here and compared against
        # witnesses taken by the SAME kernels on the recompute -- the comparison the whole
        # rematerialisation literature discards.
        cert_fwd = getattr(ctx, "cert", None)
        cert = cert_fwd is not None
        # Bring `o_h` back before anything reads it.  `fetch` also issues the H2D for the NEXT block
        # the backward will reach (layers run in reverse, so that is the entry before this one), which
        # is what gives the transfer a whole layer backward -- ~190 ms at seq 16384 against a measured
        # 5.6 ms transfer -- to arrive in.  If it has not arrived this blocks, which is a slowdown and
        # not a wrong answer; `fb_offload_stats()['blocked_waits']` counts every time it happens so a
        # degraded run cannot be reported as an overlapped one.
        o_h_handle = getattr(ctx, "o_h_handle", None)
        if o_h_handle is not None:
            o_h = _fb_offload.fetch(o_h_handle)
        if keep in ("min", "attn"):
            _FB_COUNTERS["recompute"] += 1
            xn1, rstd1, d_x = _fb_norm_w(x, w_norm1, eps, cert)
            if cert:
                _fb_cert_compare(cert_fwd[0], d_x)
            xn1_2 = xn1.view(-1, hidden)
            q, xa_q, raw_q = _fb_proj(xn1_2, wq, aq, bq, sq, biq, cq, nm[0])
            k, xa_k, raw_k = _fb_proj(xn1_2, wk, ak, bk, sk, bik, ck, nm[1])
            v, xa_v, raw_v = _fb_proj(xn1_2, wv, av, bv, sv, biv, cv, nm[2])
            q = _fb_rope_bsh(q.view(b, s_len, n_heads, head_dim), cos, sin)
            k = _fb_rope_bsh(k.view(b, s_len, n_kv_heads, head_dim), cos, sin)
            v = v.view(b, s_len, n_kv_heads, head_dim)
            if keep == "attn":
                # THE fix: `o_h` and `lse` came out of `save_for_backward`, so the O(S^2)
                # FlashAttention forward is NOT re-run.  q/k/v above are still recomputed and are
                # bitwise identical to the output pass's (probe2 Test A), so FA's backward is fed
                # exactly the bits it would be fed at `keep='full'`.
                flash_meta = ctx.flash_meta
            elif plan is None:
                _FB_COUNTERS["flash_recompute"] += 1
                o, lse, cum_q, cum_k, max_q, max_k, seed, off, _ = _fb_flash_fwd(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale)
                o_h = _fb_head_to_hidden(o, b, s_len, n_heads, head_dim)
                del o
                flash_meta = (cum_q, cum_k, max_q, max_k, seed, off)
            else:
                _FB_COUNTERS["flash_recompute"] += 1
                o_h, lse, flash_meta = _fb_varlen_fwd(q, k, v, scale, plan)
            attn_out, xa_o, raw_o = _fb_proj(o_h, wo, ao, bo, so, bio, co, nm[3])
            x_mid = x + attn_out.view(b, s_len, hidden)
            del attn_out
            xn2, rstd2, d_xmid = _fb_norm_w(x_mid, w_norm2, eps, cert)
            if cert:
                _fb_cert_compare(cert_fwd[1], d_xmid)
        else:
            flash_meta = ctx.flash_meta
        cum_q, cum_k, max_q, max_k, seed, off = flash_meta
        xn1_2 = xn1.view(-1, hidden)
        xn2_2 = xn2.view(-1, hidden)
        if h_gate is None:
            h_gate, xa_g, raw_g = _fb_proj(xn2_2, wg, ag, bg, sg, big, cg, nm[4])
            h_up, xa_u, raw_u = _fb_proj(xn2_2, wu, au, bu, su, biu, cu, nm[5])
        if _ws is not None and _fb_wstream._WS["bwd"] == "split":
            # The recompute is done and the dgrad has not started.  `split` drops the slab here and
            # re-acquires it below, so the layer is streamed THREE times per step instead of twice.
            # It buys nothing: both halves read all seven weights, so the maximum number of
            # concurrently-live slabs is unchanged and only the H2D bill goes up.  Kept because the
            # measurement is the answer to "stream three times or hold across the backward".
            _wsx.bwd_split_release(_wsi)
            wq, wk, wv, wo, wg, wu, wd = _wsx.bwd_split_acquire(_wsi)

        def _raw(stored, x2, w_, a_, b_, s_, bias_, c_):
            """The PRE-DoRA-scale projection output, for `dL/dm = sum_rows(dL/dh * raw)/nu`.

            Either it fell out of the recompute above, or it is recomputed here by the SAME
            `_fb_proj(..., c, keep_raw=True)` call the recompute makes -- same kernel, same
            inputs, same rounding -- so the two are bitwise identical and grad_m agrees across
            `keep` levels (gate B3) even though only `min` gets the raw value for free.  The
            recompute form (rather than `h / c`) is required for q/k/v: RoPE mixes channel pairs
            whose `c` entries differ, so the pre-`c` value is not recoverable from the post-`c`
            one.  Producing `y` as well is what lets the bitwise-intermediate probe pair this
            call with the forward's, at the cost of one elementwise multiply on a DoRA arm.
            """
            if stored is not None:
                return stored
            return _fb_proj(x2, w_, a_, b_, s_, bias_, c_, True)[2]

        # ------------------------------------------------------------- FFN backward
        gy = grad_out.reshape(-1, hidden)
        grad_m_down = None
        need_h_mid = (ad is not None) or nw[6] or nm[6]
        # `h_mid` is normally materialised AFTER `grad_h_mid` has been consumed, so the two never
        # coexist; only DoRA's down-magnitude gradient needs it earlier, and only that arm pays.
        h_mid = None
        d_hmid = None
        if nm[6]:
            h_mid, d_hmid = _fb_glu_w(h_gate, h_up, cert)
            raw_d = _raw(None, h_mid, wd, ad, bd, sd, bid, cd)
            grad_m_down = ((gy * raw_d).sum(0, dtype=torch.float32)
                           / nud.float()).to(md.dtype)
            del raw_d
        # y = c_d * raw_d  =>  dL/draw_d = gy * c_d. Everything downstream of the down projection
        # uses that effective gradient; only grad_m above uses the raw one.
        g_eff_y = gy * cd.unsqueeze(0) if cd is not None else gy
        gyB = g_eff_y @ bd if ad is not None else None
        grad_h_mid = g_eff_y @ wd
        if ad is not None:
            grad_h_mid.addmm_(gyB, ad, alpha=sd)
        # THE HIGH-WATER WINDOW (CONTEXT.md section 33.5, measured by `src/probe_highwater.py`).
        # Three [N, I] tensors go in -- `h_gate`, `h_up`, `grad_h_mid` -- and three come out --
        # `grad_h_gate`, `grad_h_up`, `h_mid`.  The two-kernel form makes all six coexist, and that
        # instant IS the model's peak allocation: `flashffn.py:3163:backward`, 110 MiB of 5632-wide
        # tensors out of a 169.4 MiB transient.  One kernel writing each output over an
        # already-read input costs the window three tensors instead of six.
        #
        # `keep == 'full'` is excluded because there `h_gate`/`h_up` come from `save_for_backward`:
        # mutating them would corrupt a second backward under `retain_graph` and would bump a
        # version counter autograd is entitled to check.  The values written are bitwise identical
        # either way (the kernel transcribes both original kernels expression for expression), so
        # gate B3's agreement ACROSS `keep` levels survives the exclusion.
        remat_h_mid = bool(need_h_mid and h_mid is None)
        if _FB_INPLACE_GLU["on"]:
            grad_h_gate, grad_h_up, _hm, _d = triton_swiglu_backward_glu(
                grad_h_mid, h_gate, h_up, remat_h_mid,
                inplace=(keep != "full"), digest=cert)
            if remat_h_mid:
                h_mid, d_hmid = _hm, _d
            _hm = None
            # The buffers now hold the gradients (or are dead); nothing below reads the forward
            # values, and dropping the names here is what lets `keep='full'`'s saved tensors go.
            h_gate = h_up = None
        else:
            grad_h_gate, grad_h_up = triton_swiglu_backward(grad_h_mid, h_gate, h_up)
        del grad_h_mid
        grad_a_down = grad_b_down = grad_w_down = None
        if need_h_mid:
            if h_mid is None:
                h_mid, d_hmid = _fb_glu_w(h_gate, h_up, cert)
            if ad is not None:
                if xa_d is None:                   # 'min': the thin [N, r] tensor is recomputed
                    xa_d = F.linear(h_mid, ad)
                grad_a_down = sd * (gyB.t() @ h_mid)
                grad_b_down = sd * (g_eff_y.t() @ xa_d)
            if nw[6]:
                grad_w_down = g_eff_y.t() @ h_mid  # dense: this IS the parameter's own gradient
            del h_mid, gyB, xa_d
        if cert:
            # W2 is live only where the backward actually forms `h_mid`. Under the fully-frozen
            # family it never does, so the witness is absent rather than silently passing -- the
            # tally counts comparisons MADE, and coverage is reported from it, not assumed.
            _fb_cert_compare(cert_fwd[2], d_hmid)
            _FB_CERT_STATE["blocks_certified"] += 1
        del h_gate, h_up, gy

        grad_m_gate = grad_m_up = None
        if nm[4]:
            rg = _raw(raw_g, xn2_2, wg, ag, bg, sg, big, cg)
            grad_m_gate = ((grad_h_gate * rg).sum(0, dtype=torch.float32)
                           / nug.float()).to(mg.dtype)
            del rg
        if nm[5]:
            ru = _raw(raw_u, xn2_2, wu, au, bu, su, biu, cu)
            grad_m_up = ((grad_h_up * ru).sum(0, dtype=torch.float32) / nuu.float()).to(mu.dtype)
            del ru
        del raw_g, raw_u
        # in place: `triton_swiglu_backward` just produced these and nothing else holds them.
        if cg is not None:
            grad_h_gate.mul_(cg.unsqueeze(0))
        if cu is not None:
            grad_h_up.mul_(cu.unsqueeze(0))

        ghgB = grad_h_gate @ bg if ag is not None else None
        ghuB = grad_h_up @ bu if au is not None else None
        grad_xn2 = grad_h_gate @ wg
        if ag is not None:
            grad_xn2.addmm_(ghgB, ag, alpha=sg)
        grad_xn2.addmm_(grad_h_up, wu)
        if au is not None:
            grad_xn2.addmm_(ghuB, au, alpha=su)
        grad_a_gate = grad_b_gate = grad_a_up = grad_b_up = None
        grad_w_gate = grad_w_up = None
        if ag is not None:
            grad_b_gate = sg * (grad_h_gate.t() @ xa_g)
            grad_a_gate = sg * (ghgB.t() @ xn2_2)
        if au is not None:
            grad_b_up = su * (grad_h_up.t() @ xa_u)
            grad_a_up = su * (ghuB.t() @ xn2_2)
        if nw[4]:
            grad_w_gate = grad_h_gate.t() @ xn2_2
        if nw[5]:
            grad_w_up = grad_h_up.t() @ xn2_2
        del grad_h_gate, grad_h_up, ghgB, ghuB, xn2_2

        grad_dw2 = None
        if ndw2:
            grad_x_mid, grad_dw2 = fb_rmsnorm_backward(
                grad_xn2.view(b, s_len, hidden), x_mid, w_norm2, rstd2, need_dw=True)
        else:
            grad_x_mid = fb_rmsnorm_backward(grad_xn2.view(b, s_len, hidden), x_mid, w_norm2,
                                             rstd2)
        del grad_xn2, xn2
        grad_x_mid += grad_out
        del x_mid

        # ------------------------------------------------------------- attention backward
        gm = grad_x_mid.view(-1, hidden)
        grad_m_o = None
        if nm[3]:
            ro = _raw(raw_o, o_h, wo, ao, bo, so, bio, co)
            grad_m_o = ((gm * ro).sum(0, dtype=torch.float32) / nuo.float()).to(mo.dtype)
            del ro
        del raw_o
        g_eff_o = gm * co.unsqueeze(0) if co is not None else gm
        gmB = g_eff_o @ bo if ao is not None else None
        grad_o_h = g_eff_o @ wo
        if ao is not None:
            grad_o_h.addmm_(gmB, ao, alpha=so)
        grad_a_o = grad_b_o = grad_w_o = None
        if ao is not None:
            grad_a_o = so * (gmB.t() @ o_h)
            grad_b_o = so * (g_eff_o.t() @ xa_o)
            del gmB
        if nw[3]:
            grad_w_o = g_eff_o.t() @ o_h
        del gm, g_eff_o

        # FA's backward wants [B, H, S, D]; both `o` and its gradient are held in the [B*S, H*D]
        # layout, so the head view is a stride-only VIEW rather than a second stored copy.
        if plan is None:
            o_heads = o_h.view(b, s_len, n_heads, head_dim).transpose(1, 2)
            grad_o_heads = grad_o_h.view(b, s_len, n_heads, head_dim).transpose(1, 2)
            grad_q, grad_k, grad_v = _fb_flash_bwd(
                grad_o_heads, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                o_heads, lse, cum_q, cum_k, max_q, max_k, seed, off, scale)
            del grad_o_heads, o_heads
        else:
            grad_q, grad_k, grad_v = _fb_varlen_bwd(
                grad_o_h, q, k, v, o_h, lse, flash_meta, scale, plan)
        del grad_o_h, o_h, q, k, v, lse
        if o_h_handle is not None:
            # Both buffers go back to the pools here -- the pinned one immediately, the GPU landing
            # buffer behind a compute-stream event so the next H2D cannot overwrite it while this
            # layer's FlashAttention backward is still reading it (`fb_offload._Pool.take_dev`).
            _fb_offload.release(o_h_handle)
            ctx.o_h_handle = o_h_handle = None

        # rotate-adjoint + head->hidden layout change fused into one kernel each
        grad_q = _fb_rope_bwd_to_bsh(grad_q, cos, sin, b, s_len, n_heads, head_dim)
        grad_k = _fb_rope_bwd_to_bsh(grad_k, cos, sin, b, s_len, n_kv_heads, head_dim)
        grad_v = _fb_head_to_hidden(grad_v, b, s_len, n_kv_heads, head_dim)

        grad_xn1 = None
        out_qkv = []
        for j, (gproj, w_, a_, b_, m_, nu_, sc, xa, c_, rw, bi_) in enumerate((
                (grad_q, wq, aq, bq, mq, nuq, sq, xa_q, cq, raw_q, biq),
                (grad_k, wk, ak, bk, mk, nuk, sk, xa_k, ck, raw_k, bik),
                (grad_v, wv, av, bv, mv, nuv, sv, xa_v, cv, raw_v, biv))):
            g_m = None
            if nm[j]:
                rr = _raw(rw, xn1_2, w_, a_, b_, sc, bi_, c_)
                g_m = ((gproj * rr).sum(0, dtype=torch.float32) / nu_.float()).to(m_.dtype)
                del rr
            if c_ is not None:
                gproj.mul_(c_.unsqueeze(0))    # in place: the rope adjoint just produced it
            gB = gproj @ b_ if a_ is not None else None
            grad_xn1 = _fb_acc(grad_xn1, gproj, w_)
            if a_ is not None:
                grad_xn1.addmm_(gB, a_, alpha=sc)
                out_qkv.append((sc * (gB.t() @ xn1_2), sc * (gproj.t() @ xa), g_m,
                                (gproj.t() @ xn1_2) if nw[j] else None))
            else:
                out_qkv.append((None, None, g_m, (gproj.t() @ xn1_2) if nw[j] else None))
        del grad_q, grad_k, grad_v, raw_q, raw_k, raw_v

        grad_dw1 = None
        if ndw1:
            grad_x, grad_dw1 = fb_rmsnorm_backward(grad_xn1.view(b, s_len, hidden), x, w_norm1,
                                                   rstd1, need_dw=True)
        else:
            grad_x = fb_rmsnorm_backward(grad_xn1.view(b, s_len, hidden), x, w_norm1, rstd1)
        del grad_xn1, xn1, xn1_2
        grad_x += grad_x_mid

        (ga_q, gb_q, gm_q, gw_q), (ga_k, gb_k, gm_k, gw_k), (ga_v, gb_v, gm_v, gw_v) = out_qkv
        gw = [gw_q, gw_k, gw_v, grad_w_o, grad_w_gate, grad_w_up, grad_w_down]
        # INVARIANT (mirrors FlashFFNv3AdapterFunction's): a frozen base weight must never have
        # produced a dense [out, in] gradient. Under every adapter family all seven are frozen,
        # so this asserts that the factored path is the one that ran.
        for p in range(7):
            assert nw[p] or gw[p] is None, \
                "FusedLoRABlock invariant violated: dense weight grad with a frozen base weight"

        # 5 leading + 7 projections x 5 slots + 7 biases + 7 scales + 6 trailing.  A `None` here
        # for a tensor that DID require grad is exactly the silent wrong answer this block must
        # not have, so the biases (frozen, enforced at patch time and on every forward) and `nu`
        # (detached by construction) are the only tensor slots that are unconditionally None.
        out = [None] * _FB_NARGS
        out[0] = grad_x
        out[1] = grad_dw1
        out[2] = grad_dw2
        for p, (ga, gb, g_m) in enumerate((
                (ga_q, gb_q, gm_q), (ga_k, gb_k, gm_k), (ga_v, gb_v, gm_v),
                (grad_a_o, grad_b_o, grad_m_o), (grad_a_gate, grad_b_gate, grad_m_gate),
                (grad_a_up, grad_b_up, grad_m_up), (grad_a_down, grad_b_down, grad_m_down))):
            out[_FB_W0 + 5 * p] = gw[p]
            out[_FB_W0 + 1 + 5 * p] = ga
            out[_FB_W0 + 2 + 5 * p] = gb
            out[_FB_W0 + 3 + 5 * p] = g_m
        if _ws is not None:
            # Every read of this layer's slab has now been ISSUED on the compute stream, which is
            # the edge `bwd_exit`'s `record_stream` records.  Nothing in `out` references a base
            # weight (all seven `gw[p]` are None under a frozen base, asserted above).
            _wsx.bwd_exit(_wsi)
        return tuple(out)


def _fb_unsupported_msg(name, kind, proj):
    """One message, used at patch time and on every forward, that names the family if it can."""
    t = type(proj).__name__
    hint = ""
    if "Fourier" in t or len(getattr(proj, "fourierft_spectrum", None) or {}) > 0:
        hint = (" FourierFT parameterises dW by an inverse-DCT of a sparse spectrum; there is no "
                "[r, in]/[out, r] factorisation for the fused path to multiply, so it raises here "
                "exactly as it does in FlashFFNv3AdapterFunction.")
    elif "Spectral" in t or hasattr(proj, "dct_in"):
        hint = (" The Spectral (truncated-DCT) adapter's bypass is x @ C_in^T @ S^T @ C_out, which "
                "is a three-factor form the five-slot path does not express; it raises exactly as "
                "it does in FlashFFNv3AdapterFunction.")
    return (f"FusedLoRABlock: {name} is a {kind!r} projection ({t}). The fused block covers "
            f"{', '.join(_FB_SUPPORTED_KINDS)} and refuses everything else rather than falling "
            f"back to gradient checkpointing, which would not be this method.{hint}")


def _fb_base_layer(proj):
    """The wrapped `nn.Linear` under any of the supported wrappers.

    PEFT exposes `get_base_layer()`; the repo's own `DyLoRALinear` (src/dylora.py) exposes only
    `base_layer`. Reading the bias off the WRAPPER instead would silently return None for a
    DyLoRA-wrapped biased projection -- Defect B all over again, for one family.
    """
    if hasattr(proj, "get_base_layer"):
        return proj.get_base_layer()
    if hasattr(proj, "base_layer"):
        return proj.base_layer
    return proj


def _fb_bias(proj, name, dt):
    """The projection's own additive bias, or None.

    Read off the INSTANTIATED module rather than `config.attention_bias` / `config.mlp_bias`,
    because the two can disagree (a config flag is not load-bearing; `nn.Linear.bias` is). The
    bias must be frozen: the fused Function returns no gradient for it, and a trainable bias with
    a `None` gradient is a silent wrong answer.
    """
    base = _fb_base_layer(proj)
    bias = getattr(base, "bias", None)
    if bias is None:
        return None
    if bias.requires_grad:
        raise RuntimeError(
            f"FusedLoRABlock: {name}.bias requires grad. The fused block computes the bias in its "
            f"forward but produces no bias gradient (LoRA freezes it); freeze it or use the stock "
            f"layer.")
    return bias if bias.dtype == dt else bias.to(dt)


_FB_DROPOUT_ATTR = {"lora": "lora_dropout", "dora": "lora_dropout", "adalora": "lora_dropout",
                    "vera": "vera_dropout"}


def _fb_dropout_p(proj, kind):
    """The adapter's dropout probability, whatever the family calls it. Any non-zero value is a
    hard error in this block (not a warning as in the v3 path): the recompute re-executes the
    whole forward and there is no RNG state to replay, so a dropout mask would differ between the
    output pass and the recompute -- silently wrong gradients, the exact failure mode §14.6
    records for DyLoRA's rank sampling."""
    if kind == "dylora":
        return float(getattr(getattr(proj, "dropout", None), "p", 0.0) or 0.0)
    attr = _FB_DROPOUT_ATTR.get(kind)
    if attr is None:
        return 0.0
    d = getattr(proj, attr, None)
    if d is None or len(d) == 0:
        return 0.0
    return float(getattr(d[proj.active_adapters[0]], "p", 0.0) or 0.0)


def _fb_factors(proj, name, dt):
    """(W, A, B, s, bias, m, nu) for ONE projection, for any supported adapter family.

    This is the whole of the multi-family support: every family is reduced here, outside the
    autograd Function, to the same five-slot shape the Function knows (see its docstring).  The
    composition is deliberately done with ordinary autograd ops on THIN tensors -- `lora_A *
    lora_E` is [r, in], `lam_d[:,None] * A` is [r, in], DyLoRA's slice is [b_r, in] -- so PyTorch
    derives grad_E / grad_lambda / DyLoRA's zero-padded grads for free and the fused block needs
    no per-family backward.

    Two non-pure-forward traps are closed by the same property, and both are closed HERE rather
    than inside the Function:

      * **DyLoRA samples its rank with `random.randint` on every forward** (CONTEXT §14.6).  It is
        sampled exactly ONCE, here, and the resulting slices are handed to `.apply()`, so they
        are what `save_for_backward` stores and what the `keep='min'` recompute re-multiplies.
        The recompute therefore cannot draw a second, different rank -- unlike
        `torch.utils.checkpoint`, whose RNG preservation does not cover the `random` module.
      * **AdaLoRA's rank mask moves on a schedule.**  `lora_A * lora_E` and `ranknum` are read
        once, here; the composed [r, in] tensor is saved, so a `RankAllocator` update landing
        between the forward and the backward cannot change what the recompute sees.

    DoRA additionally returns (m, nu): the trainable magnitude and the DETACHED column norm
    ||W + s*B@A||, computed by the Gram identity `_v3_dora_col_norm` -- thin GEMMs only, the
    merged weight is never formed.  It is computed once per forward and passed in as a tensor,
    so the recompute re-uses the same bits rather than recomputing the norm.
    """
    kind = _v3_proj_kind(proj)
    bias = _fb_bias(proj, name, dt)

    def cast(t):
        return t if (t is None or t.dtype == dt) else t.to(dt)

    if kind not in _FB_SUPPORTED_KINDS:
        raise NotImplementedError(_fb_unsupported_msg(name, kind, proj))
    p = _fb_dropout_p(proj, kind)
    if p > 0.0:
        raise ValueError(f"FusedLoRABlock: {name} has adapter dropout={p}; the fused path has no "
                         f"RNG state to replay in the recompute. Use dropout 0.")

    if kind == "plain":
        base = proj
        if not isinstance(base, nn.Linear):
            raise NotImplementedError(
                f"FusedLoRABlock: {name} is a {type(base).__name__}, not an nn.Linear. The fused "
                f"projection is `F.linear(x, W, bias)`.")
        return cast(base.weight), None, None, 0.0, bias, None, None

    if kind == "lora":
        aname = proj.active_adapters[0]
        if getattr(proj.lora_B[aname], "bias", None) is not None:
            raise NotImplementedError(
                f"FusedLoRABlock: {name} uses PEFT `lora_bias=True`; that bias is TRAINABLE and "
                f"the fused Function produces no gradient for it.")
        w, a, b, s = _v3_extract_proj(proj)
        return cast(w), cast(a), cast(b), s, bias, None, None

    if kind == "dora":
        aname = proj.active_adapters[0]
        w = cast(proj.get_base_layer().weight)
        a = cast(proj.lora_A[aname].weight)
        b = cast(proj.lora_B[aname].weight)
        s = float(proj.scaling[aname])
        m = cast(proj.lora_magnitude_vector[aname].weight)          # [out], trainable
        nu = _v3_dora_col_norm(w, a, b, s).to(dt)                   # [out], DETACHED
        return w, a, b, s, bias, m, nu

    if kind == "adalora":
        aname = proj.active_adapters[0]
        a_eff = cast(proj.lora_A[aname] * proj.lora_E[aname])       # [r, in], thin, tracked
        s = float(proj.scaling[aname]) / (float(proj.ranknum[aname]) + 1e-5)
        return (cast(proj.get_base_layer().weight), a_eff, cast(proj.lora_B[aname]), s,
                bias, None, None)

    if kind == "vera":
        aname = proj.active_adapters[0]
        lam_d = proj.vera_lambda_d[aname]                           # [r],   trainable
        lam_b = proj.vera_lambda_b[aname]                           # [out], trainable
        a_eff = lam_d.unsqueeze(-1) * proj.vera_A[aname][:, : proj.in_features]
        b_eff = lam_b.unsqueeze(-1) * proj.vera_B[aname][: proj.out_features, :]
        return (cast(proj.get_base_layer().weight), cast(a_eff), cast(b_eff), 1.0,
                bias, None, None)

    if kind == "dylora":
        # ONE draw per forward, and the slices it produces are what the recompute re-uses.
        b_rank = _v3_random.randint(1, proj.r) if proj.training else proj.r
        _FB_COUNTERS["dylora_rank_draws"] += 1
        return (cast(proj.base_layer.weight),
                cast(proj.lora_A[:b_rank, :]), cast(proj.lora_B[:, :b_rank]),
                float(proj.scaling), bias, None, None)

    raise NotImplementedError(_fb_unsupported_msg(name, kind, proj))


class FusedLoRABlock(nn.Module):
    """Holds the `keep` policy for one patched decoder layer (no parameters of its own).

    `keep` is either one of the four executable levels (`full`/`glu`/`attn`/`min`), in which case
    `resolve` is the identity and the level is PINNED, or the string `auto`, in which case the
    level is chosen from the batch's own sequence length on every forward.  EVERY resolution is
    recorded in `_FB_POLICY['resolved']` and readable from `fb_policy_report()`; the log line is
    printed whenever the resolved LEVEL changes (not per distinct S -- a ragged GLUE collator
    hands the block a different S on every step, and one line per length would be noise, not
    evidence).  It is never silent, because a shape-dependent storage policy that nobody can see
    in the log is exactly the kind of thing a reviewer is right to distrust in a memory table.
    """

    def __init__(self, keep="min", eps=1e-5, auto_seq=None):
        super().__init__()
        self.keep = keep
        self.eps = eps
        self.auto_seq = int(_FB_AUTO_SEQ if auto_seq is None else auto_seq)

    def resolve(self, s_len):
        if self.keep != "auto":
            return self.keep
        return "attn" if s_len >= self.auto_seq else "min"


def fb_policy_report():
    """What the keep policy actually resolved to -- for the harness to print and to record.

    `resolved` maps each sequence length seen to the level chosen for it; `flash_recompute` counts
    how many times the O(S^2) FlashAttention forward was actually re-executed in a backward, so a
    row claiming `attn` at seq 8192 and carrying a non-zero `flash_recompute` is self-refuting.
    """
    return {
        "policy": _FB_POLICY["policy"],
        "auto_seq": _FB_POLICY["auto_seq"],
        "resolved_by_seq": dict(_FB_POLICY["resolved"]),
        "forwards_by_level": {k[5:]: v for k, v in _FB_COUNTERS.items() if k.startswith("keep_")},
        "recompute": _FB_COUNTERS["recompute"],
        "flash_recompute": _FB_COUNTERS["flash_recompute"],
    }


def _fb_make_forward(layer, cfg, n_heads, n_kv_heads, head_dim):
    orig_forward = layer.forward

    def fused_forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                      output_attentions=False, use_cache=False, cache_position=None,
                      position_embeddings=None, **kwargs):
        # WP-E.  Resolved here, not latched: `fb_wstream_enable` may be toggled between forwards by
        # a bisection probe, and a cached decision would be exactly the kind of silent
        # configuration drift this project has been bitten by.
        wsx = _fb_wstream.streamer()
        li = getattr(cfg, "layer_idx", None)
        if wsx is not None and li is None:
            raise RuntimeError("fb_wstream: patched layer has no layer_idx; re-patch the model.")
        if not (layer.training and torch.is_grad_enabled()):
            # The unfused fall-back reads the base weights straight off the modules, and under
            # streaming those live on the host.  Stage the layer for the call and hand it back --
            # `no_grad`, so nothing can have saved the slab and it dies at `fwd_exit`.
            if wsx is not None:
                wsx.fwd_enter(li)
            try:
                return orig_forward(
                    hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_value, output_attentions=output_attentions,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs)
            finally:
                if wsx is not None:
                    wsx.fwd_exit(li, for_backward=False)
        if position_embeddings is None:
            raise RuntimeError("FusedLoRABlock: transformers must supply position_embeddings.")
        cos, sin = position_embeddings
        if cos.dim() == 3:
            if cos.shape[0] > 1 and not torch.equal(cos[0], cos[-1]):
                raise RuntimeError("FusedLoRABlock: per-sample position_ids differ across the "
                                   "batch; the shared [S, D] rotary table cannot represent that.")
            cos, sin = cos[0], sin[0]
        dt = hidden_states.dtype
        hidden_states = hidden_states.contiguous()
        cos = cos.to(dt)
        sin = sin.to(dt)
        # EVERY forward, never latched. `attention_mask is None` -- the unpadded case, which is
        # what transformers hands the layer when nothing in the batch is padded -- costs one
        # `is None` test, so the measured arm pays nothing for this. A padded batch that arrives
        # after an unpadded one is planned here exactly as if it had arrived first.
        plan = _fb_mask_plan(attention_mask, hidden_states.shape[0], hidden_states.shape[1],
                             hidden_states.device)

        # WP-E: the slab must be resident BEFORE `_fb_factors` runs -- it reads `base.weight` off
        # the module (and DoRA computes its column norm from it), so the acquire has to precede the
        # factor extraction, not just the `.apply()`.
        if wsx is not None:
            wsx.fwd_enter(li)
        sa, mlp = layer.self_attn, layer.mlp
        # Re-extracted on EVERY forward, never cached: parameters move between optimizer steps,
        # DoRA's column norm tracks A/B, AdaLoRA's mask and ranknum move on a schedule, and
        # DyLoRA samples a rank. A cache here would be the §14.6 bug with a different name.
        f = [_fb_factors(getattr(sa if i < 4 else mlp, nm), nm, dt)
             for i, nm in enumerate(_FB_PROJ_NAMES)]
        # also every forward: adapters can be re-cast and norms can be unfrozen between steps,
        # and both would be silent wrong answers rather than errors.
        for nm, t in zip(_FB_PROJ_NAMES, (row[1] for row in f)):
            if t is not None and t.dtype != dt:
                raise RuntimeError(f"FusedLoRABlock: {nm}'s A factor is {t.dtype}, activations "
                                   f"are {dt}. Cast adapters in EVERY arm being compared.")

        args = [hidden_states, layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight, cos, sin]
        for w_, a_, b_, s_, bi_, m_, nu_ in f:
            args += [w_, a_, b_, m_, nu_]
        args += [row[4] for row in f]                       # 7 biases
        args += [row[3] for row in f]                       # 7 scales
        # Resolved from THIS batch's S, every forward, and announced once per distinct shape.
        keep = cfg.resolve(hidden_states.shape[1])
        _FB_COUNTERS["keep_" + keep] += 1
        s_key = int(hidden_states.shape[1])
        _new_shape = _FB_POLICY["resolved"].get(s_key) != keep
        if _new_shape:
            _FB_POLICY["resolved"][s_key] = keep
            if _FB_POLICY["announce"] and _FB_POLICY.get("last_level") != keep:
                _FB_POLICY["last_level"] = keep
                _why = {"min": "ON (block-minimal state, 8.00 MiB/layer @seq1024b2)",
                        "attn": "OFF -- o_h+lse stored (16.25 MiB/layer @seq1024b2)",
                        "glu": "OFF -- whole attention path stored (50.70)",
                        "full": "OFF -- nothing is recomputed but SiLU*mul (94.70)"}[keep]
                print(f"[FusedLoRABlock] keep policy {_FB_POLICY['policy']!r} -> {keep!r} "
                      f"for seq={s_key} (auto_seq={cfg.auto_seq}); "
                      f"O(S^2) attention recompute: {_why}", flush=True)
        args += [n_heads, n_kv_heads, head_dim, cfg.eps, keep, plan]
        assert len(args) == _FB_NARGS, (len(args), _FB_NARGS)
        # `(streamer, layer_idx)` is handed to the Function through a module global rather than an
        # argument slot, because `_FB_NARGS` is a published contract and every arg slot maps to a
        # gradient slot.  It is set immediately before `.apply()` and cleared immediately after, in
        # a `finally`, so a raising forward cannot leave a stale pair for the next layer -- and
        # `forward` reads it synchronously, on the same host thread, one statement later.
        _fb_wstream.set_current(None if wsx is None else (wsx, li))
        try:
            out = FusedLoRABlockFunction.apply(*args)
        finally:
            _fb_wstream.set_current(None)
            if wsx is not None:
                wsx.fwd_exit(li)
        return (out,)

    return fused_forward


# ---------------------------------------------------------------------------------------------
# Architecture guards.
#
# The fused forward hard-codes exactly one architecture: pre-norm RMSNorm -> GQA self-attention
# with RoPE and full causal masking -> pre-norm RMSNorm -> SwiGLU MLP -> residual. Every feature a
# real decoder layer might carry that this does NOT implement has to raise at patch time, because
# ignoring one produces a plausible loss and wrong numbers. The two checks that matter most are
# behavioural rather than name-based: `mlp.act_fn` is *evaluated* and required to be SiLU (a GELU
# GLU would be silently wrong), and each norm module is *evaluated* and required to agree with the
# fused RMSNorm kernel (Gemma's `(1 + w) * x_hat` RMSNorm and a plain LayerNorm both fail here,
# and so does a mismatched epsilon).
# ---------------------------------------------------------------------------------------------

_FB_LAYER_CHILDREN = {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
_FB_ATTN_CHILDREN = {"q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb"}
_FB_MLP_CHILDREN = {"gate_proj", "up_proj", "down_proj", "act_fn", "activation_fn"}


def _fb_sliding_active(config, layer, i):
    """True if THIS layer uses sliding-window / local attention. The fused path always calls
    FlashAttention with full causal masking, so any local-attention layer must raise."""
    types = getattr(config, "layer_types", None)
    if types is not None and i < len(types):
        return "sliding" in str(types[i]) or "local" in str(types[i])
    sa_win = getattr(layer.self_attn, "sliding_window", "missing")
    if sa_win != "missing":
        return sa_win is not None
    win = getattr(config, "sliding_window", None)
    return win is not None and bool(getattr(config, "use_sliding_window", True))


def _fb_check_proj(proj, name, i):
    """Patch-time version of the per-forward projection checks. `_fb_factors` catches all of
    these on the first forward, but that is after the whole data pipeline has run (audit §4.4);
    the same conditions are static, so they are checked here as well."""
    kind = _v3_proj_kind(proj)
    if kind not in _FB_SUPPORTED_KINDS:
        raise NotImplementedError(f"layer {i}: " + _fb_unsupported_msg(name, kind, proj))
    p = _fb_dropout_p(proj, kind)
    if p > 0.0:
        raise ValueError(
            f"FusedLoRABlock: layer {i} {name} has adapter dropout={p}; the fused path has no "
            f"RNG state to replay in the recompute. Use dropout 0.")
    if kind in ("lora", "dora"):
        aname = proj.active_adapters[0]
        if getattr(proj.lora_B[aname], "bias", None) is not None:
            raise NotImplementedError(
                f"FusedLoRABlock: layer {i} {name} uses PEFT `lora_bias=True`; that bias is "
                f"TRAINABLE and the fused Function produces no gradient for it.")
    base = _fb_base_layer(proj)
    if not isinstance(base, nn.Linear):
        raise NotImplementedError(
            f"FusedLoRABlock: layer {i} {name}'s base layer is a {type(base).__name__}, not an "
            f"nn.Linear; the fused projection is `F.linear(x, W, bias)`.")
    # GUARD 12 -- a QUANTISED base weight. `isinstance(base, nn.Linear)` above does NOT catch it:
    # `bitsandbytes.nn.Linear4bit` and `Linear8bitLt` SUBCLASS `nn.Linear`, so a QLoRA model walked
    # straight through every patch-time check and then died deep in the first GEMM with
    # `mat1 and mat2 shapes cannot be multiplied (512x2048 and 1x2097152)` -- the packed
    # `Params4bit` is a flat uint8 blob, not an [out, in] matrix. That is a confusing runtime shape
    # error where this block's contract is a loud refusal at patch time (found 2026-08-11 while
    # attempting to compose QLoRA with the fused block).
    w = getattr(base, "weight", None)
    if w is not None and (not w.is_floating_point() or w.dim() != 2
                          or tuple(w.shape) != (base.out_features, base.in_features)):
        raise NotImplementedError(
            f"FusedLoRABlock: layer {i} {name}'s base weight is {type(w).__name__} "
            f"dtype={w.dtype} shape={tuple(w.shape)}, not a dense [out, in] float matrix "
            f"({base.out_features}, {base.in_features}). This is a QUANTISED base (QLoRA / "
            f"bitsandbytes / GPTQ). The fused block multiplies the base weight directly with "
            f"`F.linear`; consuming a packed weight would need a dequantise-in-kernel path that "
            f"does not exist. Composing the two is real kernel work, not a config change.")
    bias = getattr(base, "bias", None)
    if bias is not None and bias.requires_grad:
        raise RuntimeError(
            f"FusedLoRABlock: layer {i} {name}.bias requires grad. The fused block computes the "
            f"bias but produces no bias gradient; freeze it or use the stock layer.")


def _fb_check_layer(layer, i, config, eps):
    """Raise on any architectural feature the fused block does not implement."""
    sa = layer.self_attn
    for nm, owner in (("q_proj", sa), ("k_proj", sa), ("v_proj", sa), ("o_proj", sa),
                      ("gate_proj", layer.mlp), ("up_proj", layer.mlp),
                      ("down_proj", layer.mlp)):
        _fb_check_proj(getattr(owner, nm), nm, i)
    for nm in ("q_norm", "k_norm"):
        if getattr(sa, nm, None) is not None:
            raise NotImplementedError(
                f"FusedLoRABlock: layer {i} self_attn has a {nm} (Qwen3 / OLMo2 style per-head "
                f"QK-norm). The fused forward applies RoPE straight to the projection output and "
                f"would silently skip it.")
    if _fb_sliding_active(config, layer, i):
        raise NotImplementedError(
            f"FusedLoRABlock: layer {i} uses sliding-window / local attention. The fused forward "
            f"always calls FlashAttention with full causal masking.")
    drop = float(getattr(config, "attention_dropout", 0.0) or 0.0)
    if drop > 0.0:
        raise NotImplementedError(
            f"FusedLoRABlock: config.attention_dropout={drop}. FlashAttention is called with "
            f"dropout 0.0 and the recompute has no RNG state to replay.")
    for owner, nm, allowed in ((layer, "decoder layer", _FB_LAYER_CHILDREN),
                               (sa, "self_attn", _FB_ATTN_CHILDREN),
                               (layer.mlp, "mlp", _FB_MLP_CHILDREN)):
        extra = {n for n, m in owner.named_children() if m is not None} - allowed
        if extra:
            raise NotImplementedError(
                f"FusedLoRABlock: layer {i} {nm} carries sub-module(s) {sorted(extra)} that the "
                f"fused forward does not execute (e.g. Gemma2's pre/post-feedforward norms). "
                f"Refusing to patch rather than silently ignore them.")

    dev = layer.input_layernorm.weight.device
    dt = layer.input_layernorm.weight.dtype
    with torch.no_grad():
        # own generator: patching must not perturb the global RNG stream, or every seeded
        # harness in this repo would draw a different batch after the patch.
        gen = torch.Generator(device=dev).manual_seed(0)
        probe = torch.randn(8, config.hidden_size, device=dev, dtype=dt, generator=gen)
        # half the rows carry a deliberate mean offset: RMSNorm ignores the mean and LayerNorm
        # subtracts it, so a LayerNorm masquerading as the block's norm is separated by O(1)
        # rather than by O(1/sqrt(hidden)), which at hidden=2048 would hide inside the tolerance.
        probe[4:] += 3.0
        act = getattr(layer.mlp, "act_fn", None)
        if act is None:
            raise NotImplementedError(f"FusedLoRABlock: layer {i} mlp has no act_fn.")
        a_got = act(probe.float())
        if not torch.allclose(a_got, F.silu(probe.float()), atol=1e-5, rtol=1e-4):
            raise NotImplementedError(
                f"FusedLoRABlock: layer {i} mlp.act_fn is not SiLU (config.hidden_act="
                f"{getattr(config, 'hidden_act', None)!r}). The fused GLU kernel is SiLU*mul.")
        # The tolerance is keyed on BF16 regardless of `dt`, because the fused kernel always
        # rounds `x_hat` through bf16 (matching `LlamaRMSNorm`'s own rounding point). That keeps
        # an fp32 norm weight over bf16 activations -- a legitimate mixed-precision setup --
        # passing, while leaving a (1+w) / LayerNorm / wrong-epsilon norm far outside.
        rel = 4.0 * float(torch.finfo(torch.bfloat16).eps)
        for nm in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, nm)
            got = mod(probe)
            want, _ = fb_rmsnorm_forward(probe.contiguous(), mod.weight, eps)
            tol = rel * float(want.abs().max()) + 1e-4
            if float((got.float() - want.float()).abs().max()) > tol:
                raise NotImplementedError(
                    f"FusedLoRABlock: layer {i} {nm} ({type(mod).__name__}) does not match the "
                    f"fused RMSNorm kernel (Gemma-style `(1+w)` RMSNorm, LayerNorm, or a "
                    f"different epsilon). The fused block would compute a different norm.")


def apply_flash_block(model, keep="min", verbose=False, auto_seq=None, announce=True):
    """Swap every LLaMA/Mistral decoder layer onto the fused block Function.

    `keep` is one of the four executable levels -- `full` (94.70 MiB/layer), `glu` (50.70),
    `attn` (16.25) or `min` (8.00), at TinyLlama seq 1024 batch 2 -- or `auto`, which resolves
    `min` below `auto_seq` tokens and `attn` at or above it, per forward, from the batch's own S.
    `attn` is `min` plus FlashAttention's OUTPUT (`o_h`, `lse`); it is the level that removes the
    only O(S^2) term from the recompute, and `auto_seq` is where that term stops being noise.
    Every resolution is printed once per distinct sequence length (`announce=False` to silence)
    and is readable afterwards from `fb_policy_report()`.

    Requires: one of LoRA / DoRA / AdaLoRA / DyLoRA / VeRA on all seven projections, or plain
    `nn.Linear` (full fine-tuning or frozen); adapter dropout 0; frozen projection biases; bf16
    activations. Norm weights may be trainable (full FT) -- the fused RMSNorm backward produces
    `dw` in that case and only in that case. FourierFT and the Spectral adapter RAISE.

    Padded batches ARE supported: the mask is re-planned on every forward and a ragged batch is
    routed through FlashAttention-2's varlen entry point. Only `causal AND key-validity` masks are
    accepted; anything else (sliding window, packed/block-diagonal, prefix-LM, per-head) raises.
    Attention output at PAD positions is zero rather than whatever stock HF's
    `_unmask_unattended` produces there -- valid positions are unaffected, but a caller that puts
    a real label on a pad position will not reproduce stock HF's gradient (mask pad labels with
    -100, as every collator in this repo does).
    """
    if keep not in _FB_KEEP_POLICIES:
        raise ValueError(f"keep must be one of {_FB_KEEP_POLICIES}, got {keep!r}")
    auto_seq = int(_FB_AUTO_SEQ if auto_seq is None else auto_seq)
    _FB_POLICY["policy"] = keep
    _FB_POLICY["auto_seq"] = auto_seq
    _FB_POLICY["resolved"] = {}
    _FB_POLICY["last_level"] = None
    _FB_POLICY["announce"] = bool(announce)
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    inner = getattr(base, "model", base)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(f"FusedLoRABlock: no decoder layers on {type(model).__name__}")
    config = inner.config
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_heads)
    eps = getattr(config, "rms_norm_eps", 1e-5)

    patched = 0
    for i, layer in enumerate(layers):
        _fb_check_layer(layer, i, config, eps)
        cfg = FusedLoRABlock(keep=keep, eps=eps, auto_seq=auto_seq)
        cfg.layer_idx = i                     # WP-E: which slab this layer streams
        layer._flash_block = cfg
        layer.forward = _fb_make_forward(layer, cfg, n_heads, n_kv_heads, head_dim)
        patched += 1
    if patched == 0:
        raise ValueError("FusedLoRABlock: 0 decoder layers patched")
    _FB_COUNTERS["patched_layers"] = patched
    # WP-E.  Installed LAST, after every architecture guard has passed, so a model that refuses to
    # patch never has 2 GiB of its weights pinned as a side effect.
    if _fb_wstream.fb_wstream_enabled():
        _fb_wstream.install(model, inner, layers, _FB_PROJ_NAMES, _fb_base_layer,
                            verbose=True)
    if _FB_CERT["on"]:
        # A detector whose verdict can go unread is the silent failure it exists to remove. The
        # per-block comparisons are device-side and never synchronise; this hook is the ONE read,
        # once per forward, and it turns any mismatch from the previous step into an exception.
        # There is no mode in which the certificate runs and its verdict is discarded.
        if getattr(model, "_fb_cert_hook", None) is None:
            def _cert_pre_hook(_m, _a):
                fb_certify_check()
                return None          # a pre-hook returning non-None REPLACES the forward's args
            model._fb_cert_hook = model.register_forward_pre_hook(_cert_pre_hook)
        print(f"[FusedLoRABlock] rematerialisation certificate ON "
              f"({'fused' if _FB_CERT['fused'] else 'standalone'} witnesses; "
              f"W0=x, W1=x_mid, W2=h_mid; {8 * 3 * patched} bytes)", flush=True)
    if verbose or keep == "auto":
        print(f"[FusedLoRABlock] patched {patched} layers, keep={keep}"
              + (f", auto_seq={auto_seq} (min below, attn at/above)" if keep == "auto" else ""),
              flush=True)
    return patched


def fb_adapter_families(model):
    """{family: n_projections} across every patched decoder layer, read off the live modules.

    An honesty receipt, not a config echo: an arm that claims `--adapter_method dora` but whose
    projections structurally classify as `lora` (a `use_dora=False` typo, a merged adapter, a
    reload that dropped the magnitude vector) is exactly the kind of thing that produces a
    plausible number under the wrong name. It is printed at patch time and stored in the run's
    honesty counters. `full_ft` and `frozen` are separated by whether the base weight requires a
    gradient, which is the only thing that distinguishes them structurally.
    """
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    inner = getattr(base, "model", base)
    layers = getattr(inner, "layers", None) or []
    out = {}
    for layer in layers:
        if not hasattr(layer, "_flash_block"):
            continue
        for i, nm in enumerate(_FB_PROJ_NAMES):
            proj = getattr(layer.self_attn if i < 4 else layer.mlp, nm)
            kind = _v3_proj_kind(proj)
            if kind == "plain":
                kind = "full_ft" if _fb_base_layer(proj).weight.requires_grad else "frozen"
            out[kind] = out.get(kind, 0) + 1
    return dict(sorted(out.items()))


def apply_flash_final_norm(model):
    """Route `model.norm` (the final RMSNorm) through the same fused kernel.

    Not part of the block, but the same class-(a) elimination: stock `LlamaRMSNorm` saves a
    [B, S, H] **fp32** upcast (16.00 MiB at seq 1024 batch 2) where the fused form saves the bf16
    input plus `rstd` (8.00 MiB). Reported as a separate line item so it is never confused with
    the per-layer result.
    """
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    inner = getattr(base, "model", base)
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise ValueError("FusedLoRABlock: no final `norm` module found")
    eps = float(getattr(norm, "variance_epsilon", 1e-5))
    w = norm.weight

    class _FinalNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            y, rstd = fb_rmsnorm_forward(x, weight, eps)
            ctx.save_for_backward(x, weight, rstd)
            return y

        @staticmethod
        def backward(ctx, g):
            x, weight, rstd = ctx.saved_tensors
            if ctx.needs_input_grad[1]:            # full fine-tuning trains this norm
                dx, dw = fb_rmsnorm_backward(g.contiguous(), x, weight, rstd, need_dw=True)
                return dx, dw
            return fb_rmsnorm_backward(g.contiguous(), x, weight, rstd), None

    orig = norm.forward

    def fwd(x):
        if not (norm.training and torch.is_grad_enabled()):
            return orig(x)
        return _FinalNorm.apply(x.contiguous(), w)

    norm.forward = fwd
    return True


# =============================================================================
# Testing and Verification
# =============================================================================

def test_forward_correctness():
    """Verify FlashFFN produces same output as standard FFN."""
    print("Testing forward correctness...")

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 512, 1024
    intermediate_dim = 2816

    # Create both modules with same weights
    flash_ffn = FlashFFN(hidden_dim, intermediate_dim, k_fraction=0.3).to(device=device, dtype=dtype)
    standard_ffn = StandardFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)

    # Copy weights
    standard_ffn.gate_proj.weight.data = flash_ffn.gate_proj.weight.data.clone()
    standard_ffn.up_proj.weight.data = flash_ffn.up_proj.weight.data.clone()
    standard_ffn.down_proj.weight.data = flash_ffn.down_proj.weight.data.clone()

    # Test input
    x = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype)

    # Forward pass
    y_flash = flash_ffn(x)
    y_standard = standard_ffn(x)

    # Compare
    max_diff = (y_flash - y_standard).abs().max().item()
    mean_diff = (y_flash - y_standard).abs().mean().item()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Output matches: {max_diff < 1e-2}")  # FP16 tolerance

    return max_diff < 1e-2


def test_backward_correctness():
    """Verify FlashFFN gradients are close to standard FFN."""
    print("\nTesting backward correctness...")

    device = "cuda"
    dtype = torch.float32  # Use FP32 for gradient comparison

    batch, seq, hidden_dim = 2, 128, 512
    intermediate_dim = 1408

    # Create both modules
    flash_ffn = FlashFFN(hidden_dim, intermediate_dim, k_fraction=0.3).to(device=device, dtype=dtype)
    standard_ffn = StandardFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)

    # Copy weights
    standard_ffn.gate_proj.weight.data = flash_ffn.gate_proj.weight.data.clone()
    standard_ffn.up_proj.weight.data = flash_ffn.up_proj.weight.data.clone()
    standard_ffn.down_proj.weight.data = flash_ffn.down_proj.weight.data.clone()

    # Test input (requires grad)
    x_flash = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    x_standard = x_flash.clone().detach().requires_grad_(True)

    # Forward + backward
    y_flash = flash_ffn(x_flash)
    y_standard = standard_ffn(x_standard)

    loss_flash = y_flash.sum()
    loss_standard = y_standard.sum()

    loss_flash.backward()
    loss_standard.backward()

    # Compare gradients
    def cosine_similarity(a, b):
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    grad_x_sim = cosine_similarity(x_flash.grad, x_standard.grad)
    grad_wgate_sim = cosine_similarity(flash_ffn.gate_proj.weight.grad, standard_ffn.gate_proj.weight.grad)
    grad_wup_sim = cosine_similarity(flash_ffn.up_proj.weight.grad, standard_ffn.up_proj.weight.grad)
    grad_wdown_sim = cosine_similarity(flash_ffn.down_proj.weight.grad, standard_ffn.down_proj.weight.grad)

    print(f"  grad_x cosine similarity: {grad_x_sim:.6f}")
    print(f"  grad_w_gate cosine similarity: {grad_wgate_sim:.6f}")
    print(f"  grad_w_up cosine similarity: {grad_wup_sim:.6f}")
    print(f"  grad_w_down cosine similarity: {grad_wdown_sim:.6f}")

    all_good = all([
        grad_x_sim > 0.99,  # x gradient should be exact (recomputed)
        grad_wgate_sim > 0.99,  # gate gradient should be exact (recomputed)
        grad_wup_sim > 0.99,  # up gradient should be exact (recomputed)
        grad_wdown_sim > 0.95,  # down gradient uses sparse h_mid, may be slightly off
    ])

    print(f"  Gradients acceptable: {all_good}")

    return all_good


def test_memory_savings():
    """Measure actual memory savings from FlashFFN."""
    print("\nTesting memory savings...")

    import gc

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008

    # Test standard FFN memory
    print("\n  Standard FFN:")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    standard_ffn = StandardFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype, requires_grad=True)

    mem_before = torch.cuda.memory_allocated() / 1024**2
    y = standard_ffn(x)
    mem_after_fwd = torch.cuda.memory_allocated() / 1024**2
    peak_fwd = torch.cuda.max_memory_allocated() / 1024**2

    y.sum().backward()
    peak_total = torch.cuda.max_memory_allocated() / 1024**2

    print(f"    Memory before forward: {mem_before:.1f} MB")
    print(f"    Memory after forward: {mem_after_fwd:.1f} MB")
    print(f"    Peak during forward: {peak_fwd:.1f} MB")
    print(f"    Peak total: {peak_total:.1f} MB")

    del standard_ffn, x, y
    gc.collect()
    torch.cuda.empty_cache()

    # Test FlashFFN memory
    print("\n  FlashFFN (k=30%):")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    flash_ffn = FlashFFN(hidden_dim, intermediate_dim, k_fraction=0.3).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype, requires_grad=True)

    mem_before_flash = torch.cuda.memory_allocated() / 1024**2
    y = flash_ffn(x)
    mem_after_fwd_flash = torch.cuda.memory_allocated() / 1024**2
    peak_fwd_flash = torch.cuda.max_memory_allocated() / 1024**2

    y.sum().backward()
    peak_total_flash = torch.cuda.max_memory_allocated() / 1024**2

    print(f"    Memory before forward: {mem_before_flash:.1f} MB")
    print(f"    Memory after forward: {mem_after_fwd_flash:.1f} MB")
    print(f"    Peak during forward: {peak_fwd_flash:.1f} MB")
    print(f"    Peak total: {peak_total_flash:.1f} MB")

    # Calculate savings
    fwd_savings = (1 - peak_fwd_flash / peak_fwd) * 100
    total_savings = (1 - peak_total_flash / peak_total) * 100

    print(f"\n  Forward peak savings: {fwd_savings:.1f}%")
    print(f"  Total peak savings: {total_savings:.1f}%")

    return fwd_savings > 0 and total_savings > 0


def test_memory_detailed():
    """Detailed memory analysis showing all savings."""
    print("\n" + "=" * 60)
    print("DETAILED MEMORY ANALYSIS")
    print("=" * 60)

    import gc
    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008

    def measure(ffn_class, name, k_fraction=0.3):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if 'UltraLight' in name or 'Flash' in name:
            ffn = ffn_class(hidden_dim, intermediate_dim, k_fraction=k_fraction).to(device=device, dtype=dtype)
        else:
            ffn = ffn_class(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)

        x = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype, requires_grad=True)

        mem_before = torch.cuda.memory_allocated() / 1024**2
        y = ffn(x)
        mem_after_fwd = torch.cuda.memory_allocated() / 1024**2
        peak_fwd = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        y.sum().backward()
        peak_bwd = torch.cuda.max_memory_allocated() / 1024**2

        stored = mem_after_fwd - mem_before

        print(f"\n{name}:")
        print(f"  Stored for backward: {stored:.1f} MB")
        print(f"  Peak forward: {peak_fwd:.1f} MB")
        print(f"  Peak backward: {peak_bwd:.1f} MB")

        del ffn, x, y
        gc.collect()
        torch.cuda.empty_cache()

        return stored, peak_fwd, peak_bwd

    std_stored, std_peak_fwd, std_peak_bwd = measure(StandardFFN, "Standard FFN")
    flash_stored, flash_peak_fwd, flash_peak_bwd = measure(FlashFFN, "FlashFFN (k=30%)")
    ultra_stored, ultra_peak_fwd, ultra_peak_bwd = measure(FlashFFNUltraLight, "FlashFFN UltraLight (k=30%)")

    print("\n" + "=" * 60)
    print("SAVINGS SUMMARY:")
    print("=" * 60)
    print(f"\nStored for backward (per layer):")
    print(f"  Standard:   {std_stored:.1f} MB")
    print(f"  FlashFFN:   {flash_stored:.1f} MB ({(1-flash_stored/std_stored)*100:.1f}% savings)")
    print(f"  UltraLight: {ultra_stored:.1f} MB ({(1-ultra_stored/std_stored)*100:.1f}% savings)")

    print(f"\nFor 32-layer model:")
    print(f"  Standard:   {std_stored*32/1024:.1f} GB")
    print(f"  FlashFFN:   {flash_stored*32/1024:.1f} GB")
    print(f"  UltraLight: {ultra_stored*32/1024:.1f} GB")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("FlashFFN Verification Tests")
    print("=" * 60)

    fwd_ok = test_forward_correctness()
    bwd_ok = test_backward_correctness()
    mem_ok = test_memory_savings()
    test_memory_detailed()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Forward correctness: {'PASS' if fwd_ok else 'FAIL'}")
    print(f"  Backward correctness: {'PASS' if bwd_ok else 'FAIL'}")
    print(f"  Memory savings: {'PASS' if mem_ok else 'FAIL'}")
    print("=" * 60)
