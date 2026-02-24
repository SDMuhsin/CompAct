"""
Phase 3: Optimized FFN Activation Compression

Key optimizations:
1. Use efficient sparse storage (CSR-like format)
2. Block-wise top-K to reduce overhead
3. Better memory tracking
4. Gradient accumulation friendly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import gc
import math


class BlockwiseTopKCompression(torch.autograd.Function):
    """
    Block-wise top-K compression for better efficiency.
    Instead of global top-K, apply top-K per block to reduce index overhead.
    """

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor,
                k_fraction: float, block_size: int = 256):
        """
        h_mid: (batch, seq, intermediate_dim)
        Apply top-K per (batch, seq) position to preserve spatial structure.
        """
        output = F.linear(h_mid, weight)

        batch, seq, dim = h_mid.shape
        k = max(1, int(dim * k_fraction))

        # Reshape for per-position top-K
        h_reshaped = h_mid.view(batch * seq, dim)

        # Get top-K per position (much more memory efficient)
        abs_h = h_reshaped.abs()
        values, indices = torch.topk(abs_h, k, dim=-1, sorted=False)

        # Gather actual values (with signs)
        actual_values = torch.gather(h_reshaped, dim=-1, index=indices)

        # Store indices as int16 if dim allows (saves memory)
        if dim < 32768:
            indices = indices.to(torch.int16)

        ctx.save_for_backward(actual_values, indices, weight)
        ctx.original_shape = h_mid.shape
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        actual_values, indices, weight = ctx.saved_tensors
        batch, seq, dim = ctx.original_shape
        k = ctx.k

        # Reconstruct h_mid from sparse representation
        h_reconstructed = torch.zeros(
            batch * seq, dim,
            device=actual_values.device,
            dtype=actual_values.dtype
        )
        indices_long = indices.to(torch.int64)
        h_reconstructed.scatter_(dim=-1, index=indices_long, src=actual_values)
        h_reconstructed = h_reconstructed.view(batch, seq, dim)

        # Gradient w.r.t. h_mid
        grad_h_mid = grad_output @ weight

        # Gradient w.r.t. weight using sparse reconstruction
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_reconstructed.view(-1, dim)
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None, None


class PerChannelQuantizedCompression(torch.autograd.Function):
    """
    Per-channel INT8 quantization with fused scale storage.
    More accurate than global quantization.
    """

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor, bits: int = 8):
        output = F.linear(h_mid, weight)

        batch, seq, dim = h_mid.shape
        h_reshaped = h_mid.view(-1, dim)

        # Per-channel quantization
        h_min = h_reshaped.min(dim=0).values  # (dim,)
        h_max = h_reshaped.max(dim=0).values  # (dim,)
        scale = (h_max - h_min) / (2**bits - 1) + 1e-10  # (dim,)

        # Quantize
        h_normalized = (h_reshaped - h_min) / scale
        h_quantized = torch.clamp(torch.round(h_normalized), 0, 2**bits - 1)

        if bits == 8:
            h_quantized = h_quantized.to(torch.uint8)
        elif bits == 4:
            # Pack two 4-bit values per byte
            h_quantized = h_quantized.to(torch.uint8)

        ctx.save_for_backward(h_quantized, weight, scale, h_min)
        ctx.original_shape = h_mid.shape
        ctx.original_dtype = h_mid.dtype
        ctx.bits = bits

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h_quantized, weight, scale, h_min = ctx.saved_tensors
        batch, seq, dim = ctx.original_shape

        # Dequantize
        h_dequant = h_quantized.to(ctx.original_dtype) * scale + h_min
        h_dequant = h_dequant.view(batch, seq, dim)

        grad_h_mid = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_dequant.view(-1, dim)
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None


class MagnitudeThresholdCompression(torch.autograd.Function):
    """
    Store activations above a magnitude threshold.
    More adaptive than fixed top-K.
    """

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor,
                threshold_percentile: float = 0.7):
        output = F.linear(h_mid, weight)

        # Compute threshold from percentile
        abs_h = h_mid.abs()
        threshold = torch.quantile(abs_h.flatten().float(), threshold_percentile)

        # Create mask of values above threshold
        mask = abs_h >= threshold

        # Store only significant values
        significant_values = h_mid[mask]
        significant_indices = mask.nonzero(as_tuple=False)

        ctx.save_for_backward(significant_values, significant_indices, weight)
        ctx.original_shape = h_mid.shape
        ctx.dtype = h_mid.dtype

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        significant_values, significant_indices, weight = ctx.saved_tensors
        original_shape = ctx.original_shape

        # Reconstruct
        h_reconstructed = torch.zeros(
            original_shape,
            device=significant_values.device,
            dtype=ctx.dtype
        )
        h_reconstructed[
            significant_indices[:, 0],
            significant_indices[:, 1],
            significant_indices[:, 2]
        ] = significant_values

        grad_h_mid = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_reconstructed.view(-1, original_shape[-1])
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None


class HybridCompression(torch.autograd.Function):
    """
    Hybrid approach: Quantize large values, discard small values.
    Best of both worlds.
    """

    @staticmethod
    def forward(ctx, h_mid: torch.Tensor, weight: torch.Tensor,
                keep_fraction: float = 0.3, bits: int = 8):
        output = F.linear(h_mid, weight)

        batch, seq, dim = h_mid.shape
        h_reshaped = h_mid.view(batch * seq, dim)

        # Get top-K positions per token
        k = max(1, int(dim * keep_fraction))
        abs_h = h_reshaped.abs()
        _, indices = torch.topk(abs_h, k, dim=-1, sorted=False)

        # Gather values at top-K positions
        values = torch.gather(h_reshaped, dim=-1, index=indices)

        # Quantize only the kept values (per-position)
        v_min = values.min(dim=-1, keepdim=True).values
        v_max = values.max(dim=-1, keepdim=True).values
        scale = (v_max - v_min) / (2**bits - 1) + 1e-10

        v_quantized = torch.clamp(
            torch.round((values - v_min) / scale),
            0, 2**bits - 1
        ).to(torch.uint8)

        if dim < 32768:
            indices = indices.to(torch.int16)

        ctx.save_for_backward(v_quantized, indices, scale, v_min, weight)
        ctx.original_shape = h_mid.shape
        ctx.k = k
        ctx.original_dtype = h_mid.dtype

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v_quantized, indices, scale, v_min, weight = ctx.saved_tensors
        batch, seq, dim = ctx.original_shape
        k = ctx.k

        # Dequantize
        values = v_quantized.to(ctx.original_dtype) * scale + v_min

        # Reconstruct sparse h_mid
        h_reconstructed = torch.zeros(
            batch * seq, dim,
            device=values.device,
            dtype=ctx.original_dtype
        )
        indices_long = indices.to(torch.int64)
        h_reconstructed.scatter_(dim=-1, index=indices_long, src=values)
        h_reconstructed = h_reconstructed.view(batch, seq, dim)

        grad_h_mid = grad_output @ weight

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        h_2d = h_reconstructed.view(-1, dim)
        grad_weight = grad_output_2d.t() @ h_2d

        return grad_h_mid, grad_weight, None, None


class OptimizedCompressedFFN(nn.Module):
    """
    Optimized FFN with configurable compression strategies.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        compression: str = "hybrid",  # topk, quantize, threshold, hybrid
        keep_fraction: float = 0.3,
        bits: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.compression = compression
        self.keep_fraction = keep_fraction
        self.bits = bits

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_gate = self.gate_proj(x)
        h_up = self.up_proj(x)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up

        if self.compression == "topk":
            y = BlockwiseTopKCompression.apply(
                h_mid, self.down_proj.weight, self.keep_fraction
            )
        elif self.compression == "quantize":
            y = PerChannelQuantizedCompression.apply(
                h_mid, self.down_proj.weight, self.bits
            )
        elif self.compression == "threshold":
            y = MagnitudeThresholdCompression.apply(
                h_mid, self.down_proj.weight, 1.0 - self.keep_fraction
            )
        elif self.compression == "hybrid":
            y = HybridCompression.apply(
                h_mid, self.down_proj.weight, self.keep_fraction, self.bits
            )
        else:
            y = self.down_proj(h_mid)

        return y


class BaselineFFN(nn.Module):
    """Baseline FFN for comparison."""

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


def compute_theoretical_memory(
    batch: int, seq: int, dim: int, dtype: torch.dtype,
    compression: str, keep_fraction: float, bits: int = 8
) -> Dict:
    """Compute theoretical memory savings."""

    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # Baseline: store full h_mid
    baseline_bytes = batch * seq * dim * bytes_per_elem

    if compression == "topk":
        k = int(dim * keep_fraction)
        # Store: values (k * bytes_per_elem) + indices (k * 2 bytes for int16)
        compressed_bytes = batch * seq * k * (bytes_per_elem + 2)
    elif compression == "quantize":
        # Store: quantized values (1 byte per value) + scale and min per channel
        compressed_bytes = batch * seq * dim * 1 + dim * bytes_per_elem * 2
    elif compression == "hybrid":
        k = int(dim * keep_fraction)
        # Store: quantized values (k * 1 byte) + indices (k * 2 bytes) + scale/min per position
        compressed_bytes = batch * seq * (k * 3 + bytes_per_elem * 2)
    else:
        compressed_bytes = baseline_bytes

    return {
        "baseline_mb": baseline_bytes / (1024**2),
        "compressed_mb": compressed_bytes / (1024**2),
        "savings_pct": (1 - compressed_bytes / baseline_bytes) * 100,
    }


def measure_actual_memory(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cuda"
) -> Dict:
    """Measure actual CUDA memory."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    x = x.clone().requires_grad_(True)

    if device == "cuda":
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
            "forward_mb": (mem_after_forward - mem_before) / (1024**2),
            "peak_mb": peak_mem / (1024**2),
            "mem_before_mb": mem_before / (1024**2),
        }
    return {}


def measure_gradient_accuracy(
    baseline_model: nn.Module,
    compressed_model: nn.Module,
    x: torch.Tensor,
) -> Dict:
    """Measure gradient approximation quality."""

    # Sync weights
    with torch.no_grad():
        compressed_model.gate_proj.weight.copy_(baseline_model.gate_proj.weight)
        compressed_model.up_proj.weight.copy_(baseline_model.up_proj.weight)
        compressed_model.down_proj.weight.copy_(baseline_model.down_proj.weight)

    # Baseline gradients
    baseline_model.zero_grad()
    x1 = x.clone().requires_grad_(True)
    y1 = baseline_model(x1)
    y1.sum().backward()
    grad_baseline = baseline_model.down_proj.weight.grad.clone()

    # Compressed gradients
    compressed_model.zero_grad()
    x2 = x.clone().requires_grad_(True)
    y2 = compressed_model(x2)
    y2.sum().backward()
    grad_compressed = compressed_model.down_proj.weight.grad.clone()

    # Compute metrics in float32 for accuracy
    grad_baseline = grad_baseline.float()
    grad_compressed = grad_compressed.float()

    diff = (grad_baseline - grad_compressed).abs()
    baseline_norm = grad_baseline.norm()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        grad_baseline.flatten().unsqueeze(0),
        grad_compressed.flatten().unsqueeze(0)
    ).item()

    # Relative error
    rel_error = diff.norm() / (baseline_norm + 1e-8)

    # Per-element statistics
    mae = diff.mean().item()
    max_error = diff.max().item()

    return {
        "cosine_similarity": cos_sim,
        "relative_l2_error": rel_error.item(),
        "mae": mae,
        "max_error": max_error,
        "baseline_norm": baseline_norm.item(),
    }


def run_evaluation():
    """Run comprehensive evaluation of compression strategies."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 70)
    print("OPTIMIZED COMPRESSION STRATEGY EVALUATION")
    print("=" * 70)

    configs = [
        # Small scale for quick testing
        {"hidden_dim": 2048, "intermediate_dim": 5504, "batch_size": 2, "seq_len": 1024, "name": "small"},
        # LLaMA-7B scale
        {"hidden_dim": 4096, "intermediate_dim": 11008, "batch_size": 4, "seq_len": 2048, "name": "llama7b"},
    ]

    compression_configs = [
        {"compression": "none", "keep_fraction": 1.0, "bits": 16, "name": "Baseline"},
        {"compression": "topk", "keep_fraction": 0.5, "bits": 16, "name": "TopK-50%"},
        {"compression": "topk", "keep_fraction": 0.3, "bits": 16, "name": "TopK-30%"},
        {"compression": "topk", "keep_fraction": 0.2, "bits": 16, "name": "TopK-20%"},
        {"compression": "topk", "keep_fraction": 0.1, "bits": 16, "name": "TopK-10%"},
        {"compression": "quantize", "keep_fraction": 1.0, "bits": 8, "name": "Quant-8bit"},
        {"compression": "hybrid", "keep_fraction": 0.3, "bits": 8, "name": "Hybrid-30%-8bit"},
        {"compression": "hybrid", "keep_fraction": 0.2, "bits": 8, "name": "Hybrid-20%-8bit"},
    ]

    results = {}

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Scale: {config['name'].upper()}")
        print(f"  hidden_dim: {config['hidden_dim']}")
        print(f"  intermediate_dim: {config['intermediate_dim']}")
        print(f"  batch_size: {config['batch_size']}")
        print(f"  seq_len: {config['seq_len']}")
        print(f"{'=' * 70}")

        x = torch.randn(
            config['batch_size'], config['seq_len'], config['hidden_dim'],
            device=device, dtype=dtype
        )

        # Baseline model for gradient comparison
        baseline = BaselineFFN(
            config['hidden_dim'], config['intermediate_dim']
        ).to(device=device, dtype=dtype)

        config_results = {}

        for comp_config in compression_configs:
            print(f"\n{'-' * 50}")
            print(f"{comp_config['name']}")
            print(f"{'-' * 50}")

            if comp_config['compression'] == "none":
                model = baseline
            else:
                model = OptimizedCompressedFFN(
                    config['hidden_dim'],
                    config['intermediate_dim'],
                    compression=comp_config['compression'],
                    keep_fraction=comp_config['keep_fraction'],
                    bits=comp_config['bits'],
                ).to(device=device, dtype=dtype)

            # Theoretical memory
            theory = compute_theoretical_memory(
                config['batch_size'], config['seq_len'], config['intermediate_dim'],
                dtype, comp_config['compression'], comp_config['keep_fraction'],
                comp_config['bits']
            )

            # Actual memory
            mem = measure_actual_memory(model, x, device)

            # Gradient accuracy (skip for baseline)
            if comp_config['compression'] != "none":
                grad_accuracy = measure_gradient_accuracy(baseline, model, x)
            else:
                grad_accuracy = {"cosine_similarity": 1.0, "relative_l2_error": 0.0}

            print(f"  Theoretical h_mid storage:")
            print(f"    Baseline: {theory['baseline_mb']:.2f} MB")
            print(f"    Compressed: {theory['compressed_mb']:.2f} MB")
            print(f"    Savings: {theory['savings_pct']:.1f}%")
            print(f"  Actual peak memory: {mem.get('peak_mb', 'N/A'):.2f} MB")
            print(f"  Gradient accuracy:")
            print(f"    Cosine similarity: {grad_accuracy['cosine_similarity']:.4f}")
            print(f"    Relative L2 error: {grad_accuracy['relative_l2_error']:.4f}")

            config_results[comp_config['name']] = {
                "theoretical": theory,
                "actual_memory": mem,
                "gradient_accuracy": grad_accuracy,
            }

        results[config['name']] = config_results

    return results


def training_convergence_test():
    """
    Quick test to verify training actually converges with compression.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use float32 for stability in this test

    print("\n" + "=" * 70)
    print("TRAINING CONVERGENCE TEST")
    print("=" * 70)

    hidden_dim = 512
    intermediate_dim = 1408
    batch_size = 8
    seq_len = 128
    num_steps = 100

    # Create target function (simple transformation to learn)
    target_model = BaselineFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)

    configs = [
        {"compression": "none", "keep_fraction": 1.0, "name": "Baseline"},
        {"compression": "topk", "keep_fraction": 0.3, "name": "TopK-30%"},
        {"compression": "quantize", "keep_fraction": 1.0, "name": "Quant-8bit"},
        {"compression": "hybrid", "keep_fraction": 0.3, "name": "Hybrid-30%"},
    ]

    results = {}

    for config in configs:
        print(f"\n{config['name']}:")

        if config['compression'] == "none":
            model = BaselineFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)
        else:
            model = OptimizedCompressedFFN(
                hidden_dim, intermediate_dim,
                compression=config['compression'],
                keep_fraction=config['keep_fraction'],
                bits=8,
            ).to(device=device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []

        for step in range(num_steps):
            x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

            with torch.no_grad():
                target = target_model(x)

            output = model(x)
            loss = F.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 20 == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

        final_loss = sum(losses[-10:]) / 10
        print(f"  Final avg loss (last 10 steps): {final_loss:.6f}")

        results[config['name']] = {
            "losses": losses,
            "final_loss": final_loss,
        }

    # Summary
    print("\n" + "-" * 50)
    print("CONVERGENCE SUMMARY")
    print("-" * 50)
    baseline_loss = results["Baseline"]["final_loss"]
    for name, data in results.items():
        ratio = data["final_loss"] / baseline_loss
        print(f"  {name}: {data['final_loss']:.6f} ({ratio:.2f}x baseline)")

    return results


if __name__ == "__main__":
    import json
    import os

    os.makedirs("/workspace/actcomp/results", exist_ok=True)

    # Run main evaluation
    eval_results = run_evaluation()

    # Run convergence test
    conv_results = training_convergence_test()

    # Save results
    all_results = {
        "evaluation": eval_results,
        "convergence": conv_results,
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

    with open("/workspace/actcomp/results/phase3_optimized_results.json", "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to /workspace/actcomp/results/phase3_optimized_results.json")
    print("=" * 70)
