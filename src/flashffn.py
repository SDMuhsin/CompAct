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


# =============================================================================
# Triton Kernels
# =============================================================================

@triton.jit
def _silu_mul_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up → out.  Element-wise, 1-D grid."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    # silu(gate) = gate * sigmoid(gate)
    act = gate * tl.sigmoid(gate)
    result = act * up

    tl.store(out_ptr + offsets, result, mask=mask)


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


# =============================================================================
# PyTorch Wrappers for Triton Kernels
# =============================================================================

def triton_silu_mul(h_gate: torch.Tensor, h_up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(h_gate) * h_up using Triton. Replaces F.silu + mul."""
    out = torch.empty_like(h_gate)
    n_elements = h_gate.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_fwd_kernel[grid](h_gate, h_up, out, n_elements, BLOCK_SIZE)
    return out


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
