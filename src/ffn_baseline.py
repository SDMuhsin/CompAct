"""
Phase 1: Minimal FFN Block Reproduction and Activation Memory Measurement

This module creates a minimal SwiGLU FFN block matching LLaMA-7B dimensions
and measures the activation memory stored for backward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import gc


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN block as used in LLaMA models.

    Forward pass:
        h_gate = x @ W_gate          # Stored for backward
        h_up = x @ W_up              # Stored for backward
        h_act = silu(h_gate)         # Stored for backward
        h_mid = h_act * h_up         # Stored for backward
        y = h_mid @ W_down
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # Gate projection (for SiLU activation)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        # Up projection (multiplied with activated gate)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        # Down projection (back to hidden dim)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)
        Returns:
            Output tensor of shape (batch, seq, hidden_dim)
        """
        # These intermediate tensors are stored for backward pass
        h_gate = self.gate_proj(x)      # (batch, seq, intermediate_dim)
        h_up = self.up_proj(x)          # (batch, seq, intermediate_dim)
        h_act = F.silu(h_gate)          # (batch, seq, intermediate_dim)
        h_mid = h_act * h_up            # (batch, seq, intermediate_dim)
        y = self.down_proj(h_mid)       # (batch, seq, hidden_dim)
        return y


class ActivationMemoryTracker:
    """
    Tracks activation memory using PyTorch hooks.
    Registers hooks on tensors to measure what PyTorch stores for backward.
    """

    def __init__(self):
        self.activation_memory: Dict[str, int] = {}
        self.activation_shapes: Dict[str, Tuple] = {}
        self.hooks: List = []
        self.saved_tensors: Dict[str, torch.Tensor] = {}

    def _create_hook(self, name: str):
        """Create a forward hook that logs activation info."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.requires_grad:
                # Memory for this activation
                mem_bytes = output.element_size() * output.numel()
                self.activation_memory[name] = mem_bytes
                self.activation_shapes[name] = tuple(output.shape)
                # Keep a reference to understand saved tensors
                self.saved_tensors[name] = output
        return hook

    def register_hooks(self, model: nn.Module, prefix: str = ""):
        """Register hooks on all modules to track activations."""
        for name, module in model.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            hook = module.register_forward_hook(self._create_hook(full_name))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_report(self) -> str:
        """Generate a report of activation memory usage."""
        lines = ["=" * 60, "Activation Memory Report", "=" * 60]

        total_mem = 0
        for name, mem in sorted(self.activation_memory.items()):
            shape = self.activation_shapes.get(name, "unknown")
            mem_mb = mem / (1024 * 1024)
            total_mem += mem
            lines.append(f"{name:40s} | {str(shape):30s} | {mem_mb:8.2f} MB")

        lines.append("-" * 60)
        lines.append(f"{'TOTAL':40s} | {'':30s} | {total_mem / (1024 * 1024):8.2f} MB")
        lines.append("=" * 60)

        return "\n".join(lines)


def measure_activation_memory_detailed(
    hidden_dim: int = 4096,
    intermediate_dim: int = 11008,
    batch_size: int = 4,
    seq_len: int = 2048,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict:
    """
    Measure FFN activation memory in detail using PyTorch's memory tracking.

    This function measures the EXACT memory consumed by intermediate activations
    that PyTorch stores for the backward pass.
    """

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Clear any cached memory
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Create model and move to device
    model = SwiGLUFFN(hidden_dim, intermediate_dim).to(device=device, dtype=dtype)

    # Theoretical memory calculations
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    tokens = batch_size * seq_len

    # Each intermediate tensor: (batch, seq, intermediate_dim)
    intermediate_tensor_bytes = tokens * intermediate_dim * bytes_per_element
    intermediate_tensor_mb = intermediate_tensor_bytes / (1024 * 1024)

    # Tensors stored for backward in SwiGLU:
    # 1. Input x (for gate_proj, up_proj backward)
    # 2. h_gate (for SiLU backward)
    # 3. h_act (for elementwise multiply backward)
    # 4. h_up (for elementwise multiply backward)
    # 5. h_mid (for down_proj backward)

    # Input tensor memory
    input_tensor_bytes = tokens * hidden_dim * bytes_per_element
    input_tensor_mb = input_tensor_bytes / (1024 * 1024)

    print("=" * 70)
    print("THEORETICAL ACTIVATION MEMORY ANALYSIS")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  hidden_dim:       {hidden_dim}")
    print(f"  intermediate_dim: {intermediate_dim} ({intermediate_dim / hidden_dim:.1f}x expansion)")
    print(f"  batch_size:       {batch_size}")
    print(f"  seq_len:          {seq_len}")
    print(f"  total tokens:     {tokens:,}")
    print(f"  dtype:            {dtype}")
    print(f"  bytes/element:    {bytes_per_element}")
    print()
    print("Tensor Memory Breakdown:")
    print(f"  Input x:          {input_tensor_mb:8.2f} MB  (shape: {batch_size}, {seq_len}, {hidden_dim})")
    print(f"  h_gate:           {intermediate_tensor_mb:8.2f} MB  (shape: {batch_size}, {seq_len}, {intermediate_dim})")
    print(f"  h_up:             {intermediate_tensor_mb:8.2f} MB  (shape: {batch_size}, {seq_len}, {intermediate_dim})")
    print(f"  h_act (silu):     {intermediate_tensor_mb:8.2f} MB  (shape: {batch_size}, {seq_len}, {intermediate_dim})")
    print(f"  h_mid:            {intermediate_tensor_mb:8.2f} MB  (shape: {batch_size}, {seq_len}, {intermediate_dim})")
    print("-" * 70)

    # Total activation memory for FFN
    total_activation_mb = input_tensor_mb + 4 * intermediate_tensor_mb
    print(f"  TOTAL FFN Activations: {total_activation_mb:8.2f} MB")
    print()

    # Weight memory for comparison
    weight_bytes = (
        hidden_dim * intermediate_dim * bytes_per_element +  # gate_proj
        hidden_dim * intermediate_dim * bytes_per_element +  # up_proj
        intermediate_dim * hidden_dim * bytes_per_element    # down_proj
    )
    weight_mb = weight_bytes / (1024 * 1024)
    print(f"  Weight memory:    {weight_mb:8.2f} MB")
    print(f"  Activation/Weight ratio: {total_activation_mb / weight_mb:.2f}x")
    print()

    # Now measure ACTUAL memory during forward/backward
    print("=" * 70)
    print("ACTUAL MEMORY MEASUREMENT")
    print("=" * 70)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)

    if device == "cuda":
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward pass
        y = model(x)
        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated()

        # Backward pass
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()
        mem_after_backward = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        forward_mem = mem_after_forward - mem_before
        forward_mem_mb = forward_mem / (1024 * 1024)

        print(f"Memory before forward:     {mem_before / (1024**2):8.2f} MB")
        print(f"Memory after forward:      {mem_after_forward / (1024**2):8.2f} MB")
        print(f"Memory for forward pass:   {forward_mem_mb:8.2f} MB")
        print(f"Memory after backward:     {mem_after_backward / (1024**2):8.2f} MB")
        print(f"Peak memory:               {peak_mem / (1024**2):8.2f} MB")

    else:
        # CPU mode - can't use CUDA memory tracking
        print("CPU mode: Using theoretical calculations only")
        forward_mem_mb = total_activation_mb

    print()
    print("=" * 70)
    print("SUMMARY: FFN ACTIVATION MEMORY BOTTLENECK")
    print("=" * 70)
    print(f"For ONE FFN layer with {tokens:,} tokens:")
    print(f"  - Theoretical activation memory: {total_activation_mb:.2f} MB")
    print(f"  - 4 intermediate tensors at {intermediate_dim} dim dominate")
    print(f"  - Expansion ratio: {intermediate_dim / hidden_dim:.1f}x")
    print()

    # Scale to full LLaMA-7B (32 layers)
    num_layers = 32
    full_model_activation_mb = total_activation_mb * num_layers
    print(f"Scaled to 32-layer model:")
    print(f"  - Total FFN activation memory: {full_model_activation_mb:.2f} MB = {full_model_activation_mb / 1024:.2f} GB")
    print()

    return {
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype": str(dtype),
        "input_activation_mb": input_tensor_mb,
        "intermediate_activation_mb": intermediate_tensor_mb,
        "total_ffn_activation_mb": total_activation_mb,
        "weight_mb": weight_mb,
        "activation_weight_ratio": total_activation_mb / weight_mb,
        "full_model_activation_gb": full_model_activation_mb / 1024,
    }


def measure_with_custom_autograd(
    hidden_dim: int = 4096,
    intermediate_dim: int = 11008,
    batch_size: int = 4,
    seq_len: int = 2048,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict:
    """
    Measure exactly what PyTorch's autograd stores by using custom autograd functions.
    This gives us ground truth on saved tensors.
    """

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    class InstrumentedSiLU(torch.autograd.Function):
        """SiLU that reports what it saves for backward."""
        saved_info = {}

        @staticmethod
        def forward(ctx, x):
            sigmoid_x = torch.sigmoid(x)
            result = x * sigmoid_x
            # SiLU backward needs both x and sigmoid(x)
            ctx.save_for_backward(x, sigmoid_x)
            InstrumentedSiLU.saved_info['silu_saved'] = {
                'x': x.element_size() * x.numel(),
                'sigmoid_x': sigmoid_x.element_size() * sigmoid_x.numel(),
            }
            return result

        @staticmethod
        def backward(ctx, grad_output):
            x, sigmoid_x = ctx.saved_tensors
            # d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            #                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            grad = sigmoid_x * (1 + x * (1 - sigmoid_x))
            return grad_output * grad

    class InstrumentedLinear(torch.autograd.Function):
        """Linear that reports what it saves for backward."""
        saved_info = {}

        @staticmethod
        def forward(ctx, x, weight, name):
            ctx.save_for_backward(x, weight)
            ctx.name = name
            result = x @ weight.t()
            InstrumentedLinear.saved_info[f'{name}_saved'] = {
                'x': x.element_size() * x.numel(),
                'weight': weight.element_size() * weight.numel(),
            }
            return result

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            grad_x = grad_output @ weight
            grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
            return grad_x, grad_weight, None

    class InstrumentedMul(torch.autograd.Function):
        """Element-wise multiply that reports what it saves for backward."""
        saved_info = {}

        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            result = a * b
            InstrumentedMul.saved_info['mul_saved'] = {
                'a': a.element_size() * a.numel(),
                'b': b.element_size() * b.numel(),
            }
            return result

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            return grad_output * b, grad_output * a

    # Create weights
    W_gate = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    W_up = torch.randn(intermediate_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    W_down = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)

    # Forward pass with instrumented operations
    h_gate = InstrumentedLinear.apply(x, W_gate, 'gate')
    h_up = InstrumentedLinear.apply(x, W_up, 'up')
    h_act = InstrumentedSiLU.apply(h_gate)
    h_mid = InstrumentedMul.apply(h_act, h_up)
    y = InstrumentedLinear.apply(h_mid, W_down, 'down')

    # Report what was saved
    print("=" * 70)
    print("AUTOGRAD SAVED TENSORS ANALYSIS")
    print("=" * 70)

    total_saved = 0

    print("\nLinear (gate_proj) saves:")
    for k, v in InstrumentedLinear.saved_info.get('gate_saved', {}).items():
        print(f"  {k}: {v / (1024**2):.2f} MB")
        if 'x' in k:  # Only count activations, not weights
            total_saved += v

    print("\nLinear (up_proj) saves:")
    for k, v in InstrumentedLinear.saved_info.get('up_saved', {}).items():
        print(f"  {k}: {v / (1024**2):.2f} MB")
        # Note: x is the same tensor, so we don't double count

    print("\nSiLU saves:")
    for k, v in InstrumentedSiLU.saved_info.get('silu_saved', {}).items():
        print(f"  {k}: {v / (1024**2):.2f} MB")
        total_saved += v

    print("\nElementwise multiply saves:")
    for k, v in InstrumentedMul.saved_info.get('mul_saved', {}).items():
        print(f"  {k}: {v / (1024**2):.2f} MB")
        total_saved += v

    print("\nLinear (down_proj) saves:")
    for k, v in InstrumentedLinear.saved_info.get('down_saved', {}).items():
        print(f"  {k}: {v / (1024**2):.2f} MB")
        if 'x' in k:  # Only count activations, not weights
            total_saved += v

    print("\n" + "=" * 70)
    print(f"TOTAL ACTIVATION MEMORY SAVED FOR BACKWARD: {total_saved / (1024**2):.2f} MB")
    print("=" * 70)

    return {
        "gate_saved": InstrumentedLinear.saved_info.get('gate_saved', {}),
        "up_saved": InstrumentedLinear.saved_info.get('up_saved', {}),
        "silu_saved": InstrumentedSiLU.saved_info.get('silu_saved', {}),
        "mul_saved": InstrumentedMul.saved_info.get('mul_saved', {}),
        "down_saved": InstrumentedLinear.saved_info.get('down_saved', {}),
        "total_activation_mb": total_saved / (1024**2),
    }


if __name__ == "__main__":
    import json
    import os

    # Create results directory
    os.makedirs("/workspace/actcomp/results", exist_ok=True)

    print("Phase 1.1 & 1.2: FFN Activation Memory Measurement")
    print("=" * 70)
    print()

    # LLaMA-7B style dimensions
    config = {
        "hidden_dim": 4096,
        "intermediate_dim": 11008,
        "batch_size": 4,
        "seq_len": 2048,
        "dtype": torch.float16,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print(f"Running on: {config['device']}")
    print()

    # Measure activation memory
    results1 = measure_activation_memory_detailed(**config)

    print("\n" + "=" * 70 + "\n")

    # Detailed autograd analysis
    results2 = measure_with_custom_autograd(**config)

    # Save results
    combined_results = {
        "config": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in config.items()},
        "theoretical_analysis": results1,
        "autograd_analysis": {k: v for k, v in results2.items() if not isinstance(v, dict) or all(isinstance(vv, (int, float)) for vv in v.values())},
    }

    with open("/workspace/actcomp/results/phase1_measurements.json", "w") as f:
        json.dump(combined_results, f, indent=2, default=str)

    print("\nResults saved to /workspace/actcomp/results/phase1_measurements.json")
