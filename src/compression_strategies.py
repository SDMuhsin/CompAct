"""
Phase 3: FFN Activation Compression Strategies

Implements multiple strategies for compressing FFN intermediate activations
during training while preserving gradient computation.

CRITICAL: These are ACTIVATION compression techniques, not weight/gradient compression.
The goal is to reduce memory for tensors stored during forward pass for backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import gc
import math


# ==============================================================================
# Strategy A: Selective Top-K Storage
# ==============================================================================

class TopKActivationCompression(torch.autograd.Function):
    """
    Compress activations by storing only top-K values by magnitude.
    During backward, reconstruct a sparse approximation.

    Memory savings: (1 - k/n) * original_size - overhead_for_indices
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k_fraction: float = 0.3):
        """
        Args:
            x: Activation tensor of shape (batch, seq, dim)
            k_fraction: Fraction of values to keep (0.3 = keep top 30%)
        """
        original_shape = x.shape
        x_flat = x.view(-1)
        n = x_flat.numel()
        k = max(1, int(n * k_fraction))

        # Find top-K values by magnitude
        abs_x = x_flat.abs()
        _, indices = torch.topk(abs_x, k, sorted=False)
        values = x_flat[indices]

        # Save compressed representation for backward
        ctx.save_for_backward(indices, values)
        ctx.original_shape = original_shape
        ctx.n = n

        # Return original for forward computation (no change to forward pass)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, values = ctx.saved_tensors
        original_shape = ctx.original_shape
        n = ctx.n

        # Reconstruct sparse activation for gradient computation
        # For h_mid in down_proj backward: grad_weight = grad_output.T @ h_mid
        # We approximate h_mid with sparse version

        grad_output_flat = grad_output.view(-1)

        # The gradient w.r.t. input is just grad_output (identity for this wrapper)
        # But downstream layers use our sparse reconstruction

        return grad_output, None


class SparseLinearBackward(torch.autograd.Function):
    """
    Linear layer that stores compressed activations for backward pass.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, k_fraction: float = 0.3):
        """
        Forward: y = x @ weight.T
        Stores compressed x for backward pass.
        """
        # Compute output normally
        output = F.linear(x, weight)

        # Compress x for storage
        original_shape = x.shape
        x_flat = x.view(-1)
        n = x_flat.numel()
        k = max(1, int(n * k_fraction))

        # Top-K by magnitude
        abs_x = x_flat.abs()
        _, indices = torch.topk(abs_x, k, sorted=False)
        values = x_flat[indices]

        # Save compressed activation (much smaller than full x)
        ctx.save_for_backward(indices, values, weight)
        ctx.original_shape = original_shape
        ctx.n = n
        ctx.k_fraction = k_fraction

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, values, weight = ctx.saved_tensors
        original_shape = ctx.original_shape
        n = ctx.n

        # Reconstruct sparse x
        x_sparse = torch.zeros(n, device=values.device, dtype=values.dtype)
        x_sparse[indices] = values
        x_reconstructed = x_sparse.view(original_shape)

        # Gradient computations
        # grad_x = grad_output @ weight
        grad_x = grad_output @ weight

        # grad_weight = grad_output.T @ x (using sparse approximation)
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        x_2d = x_reconstructed.view(-1, original_shape[-1])
        grad_weight = grad_output_2d.t() @ x_2d

        return grad_x, grad_weight, None


# ==============================================================================
# Strategy B: Quantized Activation Storage
# ==============================================================================

class QuantizedActivationStorage(torch.autograd.Function):
    """
    Store activations in INT8 for backward pass.
    Reduces memory by 2x (FP16 -> INT8) or 4x (FP32 -> INT8).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int = 8):
        """
        Quantize activation for storage, return original for forward.
        """
        # Compute quantization parameters
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (2**bits - 1)
        zero_point = -x_min / scale

        # Quantize
        x_quantized = torch.clamp(
            torch.round((x - x_min) / scale),
            0, 2**bits - 1
        ).to(torch.uint8)

        # Save quantized representation
        ctx.save_for_backward(x_quantized)
        ctx.scale = scale
        ctx.x_min = x_min
        ctx.original_dtype = x.dtype

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_quantized, = ctx.saved_tensors

        # Dequantize for gradient computation
        x_dequant = x_quantized.to(ctx.original_dtype) * ctx.scale + ctx.x_min

        return grad_output, None


class QuantizedLinearBackward(torch.autograd.Function):
    """
    Linear layer that stores INT8 quantized activations for backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bits: int = 8):
        output = F.linear(x, weight)

        # Quantize x for storage
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / (2**bits - 1) + 1e-10

        x_quantized = torch.clamp(
            torch.round((x - x_min) / scale),
            0, 2**bits - 1
        ).to(torch.uint8)

        ctx.save_for_backward(x_quantized, weight)
        ctx.scale = scale
        ctx.x_min = x_min
        ctx.original_dtype = x.dtype
        ctx.original_shape = x.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_quantized, weight = ctx.saved_tensors

        # Dequantize
        x_dequant = x_quantized.to(ctx.original_dtype) * ctx.scale + ctx.x_min

        grad_x = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        x_2d = x_dequant.view(-1, ctx.original_shape[-1])
        grad_weight = grad_output_2d.t() @ x_2d

        return grad_x, grad_weight, None


# ==============================================================================
# Strategy C: Low-Rank Activation Approximation
# ==============================================================================

class LowRankActivationCompression(torch.autograd.Function):
    """
    Project activations to a lower-dimensional space for storage.
    Memory savings proportional to rank_fraction.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, projection: torch.Tensor):
        """
        Args:
            x: Activation tensor (batch, seq, dim)
            projection: Random projection matrix (dim, low_rank)
        """
        # Project to lower dimension for storage
        x_2d = x.view(-1, x.shape[-1])
        x_low_rank = x_2d @ projection  # (batch*seq, low_rank)

        ctx.save_for_backward(x_low_rank, projection)
        ctx.original_shape = x.shape

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_low_rank, projection = ctx.saved_tensors
        original_shape = ctx.original_shape

        # Reconstruct approximate x from low-rank representation
        # x_approx = x_low_rank @ projection.T (pseudo-inverse or transpose)
        x_reconstructed = x_low_rank @ projection.t()
        x_reconstructed = x_reconstructed.view(original_shape)

        return grad_output, None


class LowRankLinearBackward(torch.autograd.Function):
    """
    Linear with low-rank activation storage for backward.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                projection: torch.Tensor, rank_fraction: float = 0.25):
        output = F.linear(x, weight)

        # Project x to lower dimension
        x_2d = x.view(-1, x.shape[-1])
        x_low_rank = x_2d @ projection

        ctx.save_for_backward(x_low_rank, weight, projection)
        ctx.original_shape = x.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_low_rank, weight, projection = ctx.saved_tensors
        original_shape = ctx.original_shape

        # Reconstruct x
        x_reconstructed = x_low_rank @ projection.t()
        x_reconstructed = x_reconstructed.view(original_shape)

        grad_x = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        x_2d = x_reconstructed.view(-1, original_shape[-1])
        grad_weight = grad_output_2d.t() @ x_2d

        return grad_x, grad_weight, None, None


# ==============================================================================
# Compressed SwiGLU FFN Modules
# ==============================================================================

class CompressedSwiGLUFFN(nn.Module):
    """
    SwiGLU FFN with compressed activation storage.
    Supports multiple compression strategies.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        compression_strategy: str = "topk",  # topk, quantize, lowrank, none
        compression_ratio: float = 0.3,  # Fraction to keep (topk) or rank ratio (lowrank)
        quantization_bits: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.compression_strategy = compression_strategy
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        # Low-rank projection matrices (if using low-rank compression)
        if compression_strategy == "lowrank":
            low_rank = int(intermediate_dim * compression_ratio)
            self.register_buffer(
                'projection_mid',
                torch.randn(intermediate_dim, low_rank) / math.sqrt(intermediate_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_gate = self.gate_proj(x)
        h_up = self.up_proj(x)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up

        if self.compression_strategy == "topk":
            # Apply top-K compression to h_mid before down_proj
            y = CompressedDownProj.apply(
                h_mid, self.down_proj.weight, self.compression_ratio
            )
        elif self.compression_strategy == "quantize":
            y = QuantizedDownProj.apply(
                h_mid, self.down_proj.weight, self.quantization_bits
            )
        elif self.compression_strategy == "lowrank":
            y = LowRankDownProj.apply(
                h_mid, self.down_proj.weight, self.projection_mid
            )
        else:
            y = self.down_proj(h_mid)

        return y


class CompressedDownProj(torch.autograd.Function):
    """Down projection with top-K compressed h_mid storage."""

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor, k_fraction: float):
        output = F.linear(h_mid, weight)

        # Compress h_mid for storage
        original_shape = h_mid.shape
        h_flat = h_mid.view(-1)
        n = h_flat.numel()
        k = max(1, int(n * k_fraction))

        abs_h = h_flat.abs()
        _, indices = torch.topk(abs_h, k, sorted=False)
        values = h_flat[indices]

        ctx.save_for_backward(indices, values, weight)
        ctx.original_shape = original_shape
        ctx.n = n

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        indices, values, weight = ctx.saved_tensors
        original_shape = ctx.original_shape
        n = ctx.n

        # Reconstruct sparse h_mid
        h_sparse = torch.zeros(n, device=values.device, dtype=values.dtype)
        h_sparse[indices] = values
        h_reconstructed = h_sparse.view(original_shape)

        grad_h_mid = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_reconstructed.view(-1, original_shape[-1])
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None


class QuantizedDownProj(torch.autograd.Function):
    """Down projection with INT8 quantized h_mid storage."""

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor, bits: int):
        output = F.linear(h_mid, weight)

        h_min = h_mid.min()
        h_max = h_mid.max()
        scale = (h_max - h_min) / (2**bits - 1) + 1e-10

        h_quantized = torch.clamp(
            torch.round((h_mid - h_min) / scale),
            0, 2**bits - 1
        ).to(torch.uint8)

        ctx.save_for_backward(h_quantized, weight)
        ctx.scale = scale
        ctx.h_min = h_min
        ctx.original_dtype = h_mid.dtype
        ctx.original_shape = h_mid.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h_quantized, weight = ctx.saved_tensors

        h_dequant = h_quantized.to(ctx.original_dtype) * ctx.scale + ctx.h_min

        grad_h_mid = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_dequant.view(-1, ctx.original_shape[-1])
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None


class LowRankDownProj(torch.autograd.Function):
    """Down projection with low-rank h_mid storage."""

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor, projection: torch.Tensor):
        output = F.linear(h_mid, weight)

        h_2d = h_mid.view(-1, h_mid.shape[-1])
        h_low_rank = h_2d @ projection

        ctx.save_for_backward(h_low_rank, weight, projection)
        ctx.original_shape = h_mid.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h_low_rank, weight, projection = ctx.saved_tensors

        h_reconstructed = h_low_rank @ projection.t()
        h_reconstructed = h_reconstructed.view(ctx.original_shape)

        grad_h_mid = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_reconstructed.view(-1, ctx.original_shape[-1])
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None


# ==============================================================================
# Baseline FFN for comparison
# ==============================================================================

class BaselineSwiGLUFFN(nn.Module):
    """Standard SwiGLU FFN without compression."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_gate = self.gate_proj(x)
        h_up = self.up_proj(x)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up
        y = self.down_proj(h_mid)
        return y


# ==============================================================================
# Memory and Gradient Measurement Utilities
# ==============================================================================

def measure_memory_usage(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cuda"
) -> Dict:
    """Measure peak memory usage during forward/backward."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    x = x.clone().requires_grad_(True)

    if device == "cuda":
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

    y = model(x)

    if device == "cuda":
        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated()

    loss = y.sum()
    loss.backward()

    if device == "cuda":
        torch.cuda.synchronize()
        mem_after_backward = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        return {
            "mem_before_mb": mem_before / (1024**2),
            "mem_after_forward_mb": mem_after_forward / (1024**2),
            "mem_after_backward_mb": mem_after_backward / (1024**2),
            "peak_mem_mb": peak_mem / (1024**2),
            "forward_activation_mb": (mem_after_forward - mem_before) / (1024**2),
        }
    else:
        return {"note": "CPU mode - no memory tracking"}


def measure_gradient_error(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    x: torch.Tensor,
) -> Dict:
    """
    Measure gradient approximation error between baseline and compressed models.
    """
    # Copy weights from baseline to compressed
    with torch.no_grad():
        compressed_model.gate_proj.weight.copy_(baseline_model.gate_proj.weight)
        compressed_model.up_proj.weight.copy_(baseline_model.up_proj.weight)
        compressed_model.down_proj.weight.copy_(baseline_model.down_proj.weight)

    # Baseline gradients
    x_baseline = x.clone().requires_grad_(True)
    y_baseline = baseline_model(x_baseline)
    loss_baseline = y_baseline.sum()
    loss_baseline.backward()

    baseline_grads = {
        "gate": baseline_model.gate_proj.weight.grad.clone(),
        "up": baseline_model.up_proj.weight.grad.clone(),
        "down": baseline_model.down_proj.weight.grad.clone(),
    }

    # Clear gradients
    baseline_model.zero_grad()

    # Compressed gradients
    x_compressed = x.clone().requires_grad_(True)
    y_compressed = compressed_model(x_compressed)
    loss_compressed = y_compressed.sum()
    loss_compressed.backward()

    compressed_grads = {
        "gate": compressed_model.gate_proj.weight.grad.clone(),
        "up": compressed_model.up_proj.weight.grad.clone(),
        "down": compressed_model.down_proj.weight.grad.clone(),
    }

    # Compute errors
    errors = {}
    for name in ["gate", "up", "down"]:
        diff = (baseline_grads[name] - compressed_grads[name]).abs()
        baseline_norm = baseline_grads[name].abs()

        errors[f"{name}_mae"] = diff.mean().item()
        errors[f"{name}_max_error"] = diff.max().item()
        errors[f"{name}_relative_error"] = (diff / (baseline_norm + 1e-8)).mean().item()

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            baseline_grads[name].flatten().unsqueeze(0),
            compressed_grads[name].flatten().unsqueeze(0)
        ).item()
        errors[f"{name}_cosine_sim"] = cos_sim

    return errors


# ==============================================================================
# Main evaluation
# ==============================================================================

def evaluate_compression_strategies(
    hidden_dim: int = 4096,
    intermediate_dim: int = 11008,
    batch_size: int = 4,
    seq_len: int = 2048,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Evaluate all compression strategies."""

    print("=" * 70)
    print("PHASE 3: COMPRESSION STRATEGY EVALUATION")
    print("=" * 70)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        dtype = torch.float32
        print("CUDA not available, using CPU")

    print(f"\nConfiguration:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  intermediate_dim: {intermediate_dim}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  device: {device}")

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    # Theoretical memory for h_mid
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    h_mid_memory_mb = batch_size * seq_len * intermediate_dim * bytes_per_element / (1024**2)
    print(f"\nTheoretical h_mid memory: {h_mid_memory_mb:.2f} MB")

    results = {}

    # ==== Baseline ====
    print("\n" + "-" * 70)
    print("BASELINE (No Compression)")
    print("-" * 70)

    baseline = BaselineSwiGLUFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)
    baseline_mem = measure_memory_usage(baseline, x, device)

    print(f"  Forward activation memory: {baseline_mem.get('forward_activation_mb', 'N/A'):.2f} MB")
    print(f"  Peak memory: {baseline_mem.get('peak_mem_mb', 'N/A'):.2f} MB")
    results["baseline"] = baseline_mem

    # ==== Top-K Compression ====
    for k_frac in [0.5, 0.3, 0.2, 0.1]:
        print(f"\n" + "-" * 70)
        print(f"TOP-K COMPRESSION (k={k_frac*100:.0f}%)")
        print("-" * 70)

        compressed = CompressedSwiGLUFFN(
            hidden_dim, intermediate_dim,
            compression_strategy="topk",
            compression_ratio=k_frac
        ).to(device=device, dtype=dtype)

        mem = measure_memory_usage(compressed, x, device)
        errors = measure_gradient_error(baseline, compressed, x)

        print(f"  Forward activation memory: {mem.get('forward_activation_mb', 'N/A'):.2f} MB")
        print(f"  Peak memory: {mem.get('peak_mem_mb', 'N/A'):.2f} MB")

        if 'forward_activation_mb' in baseline_mem and 'forward_activation_mb' in mem:
            savings = (1 - mem['forward_activation_mb'] / baseline_mem['forward_activation_mb']) * 100
            print(f"  Memory savings: {savings:.1f}%")

        print(f"  Gradient errors:")
        print(f"    down_proj cosine sim: {errors['down_cosine_sim']:.4f}")
        print(f"    down_proj relative error: {errors['down_relative_error']:.4f}")

        results[f"topk_{k_frac}"] = {"memory": mem, "gradient_errors": errors}

    # ==== Quantization Compression ====
    for bits in [8, 4]:
        print(f"\n" + "-" * 70)
        print(f"QUANTIZATION ({bits}-bit)")
        print("-" * 70)

        compressed = CompressedSwiGLUFFN(
            hidden_dim, intermediate_dim,
            compression_strategy="quantize",
            quantization_bits=bits
        ).to(device=device, dtype=dtype)

        mem = measure_memory_usage(compressed, x, device)
        errors = measure_gradient_error(baseline, compressed, x)

        print(f"  Forward activation memory: {mem.get('forward_activation_mb', 'N/A'):.2f} MB")
        print(f"  Peak memory: {mem.get('peak_mem_mb', 'N/A'):.2f} MB")
        print(f"  Gradient errors:")
        print(f"    down_proj cosine sim: {errors['down_cosine_sim']:.4f}")
        print(f"    down_proj relative error: {errors['down_relative_error']:.4f}")

        results[f"quant_{bits}bit"] = {"memory": mem, "gradient_errors": errors}

    # ==== Low-Rank Compression ====
    for rank_frac in [0.5, 0.25, 0.1]:
        print(f"\n" + "-" * 70)
        print(f"LOW-RANK COMPRESSION (rank={rank_frac*100:.0f}%)")
        print("-" * 70)

        compressed = CompressedSwiGLUFFN(
            hidden_dim, intermediate_dim,
            compression_strategy="lowrank",
            compression_ratio=rank_frac
        ).to(device=device, dtype=dtype)

        mem = measure_memory_usage(compressed, x, device)
        errors = measure_gradient_error(baseline, compressed, x)

        print(f"  Forward activation memory: {mem.get('forward_activation_mb', 'N/A'):.2f} MB")
        print(f"  Peak memory: {mem.get('peak_mem_mb', 'N/A'):.2f} MB")
        print(f"  Gradient errors:")
        print(f"    down_proj cosine sim: {errors['down_cosine_sim']:.4f}")
        print(f"    down_proj relative error: {errors['down_relative_error']:.4f}")

        results[f"lowrank_{rank_frac}"] = {"memory": mem, "gradient_errors": errors}

    return results


if __name__ == "__main__":
    import json
    import os

    os.makedirs("/workspace/actcomp/results", exist_ok=True)

    # Run with smaller dimensions for faster testing, then scale up
    print("Running compression strategy evaluation...")

    # First test with smaller dimensions
    results_small = evaluate_compression_strategies(
        hidden_dim=2048,
        intermediate_dim=5504,
        batch_size=2,
        seq_len=1024,
    )

    # Then test with LLaMA-7B dimensions
    print("\n\n" + "=" * 70)
    print("FULL-SCALE EVALUATION (LLaMA-7B dimensions)")
    print("=" * 70)

    results_full = evaluate_compression_strategies(
        hidden_dim=4096,
        intermediate_dim=11008,
        batch_size=4,
        seq_len=2048,
    )

    # Save results
    all_results = {
        "small_scale": results_small,
        "llama_scale": results_full,
    }

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    with open("/workspace/actcomp/results/phase3_compression_results.json", "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print("\n\nResults saved to /workspace/actcomp/results/phase3_compression_results.json")
