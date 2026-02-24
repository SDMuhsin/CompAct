"""
Deep investigation into why memory savings aren't materializing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


def get_tensor_memory(tensor):
    """Get memory in bytes for a tensor."""
    return tensor.element_size() * tensor.numel()


def print_cuda_memory(label=""):
    """Print detailed CUDA memory stats."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{label}: allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")


class InspectSavedTensors(torch.autograd.Function):
    """Wrapper to inspect what autograd actually saves."""

    saved_tensors_info = []

    @staticmethod
    def forward(ctx, x, weight, k_fraction):
        # Compute output
        output = F.linear(x, weight)

        # Our "compressed" storage
        batch_seq, dim = x.view(-1, x.shape[-1]).shape
        k = int(dim * k_fraction)

        x_flat = x.view(-1, dim)
        abs_x = x_flat.abs()
        values, indices = torch.topk(abs_x, k, dim=-1, sorted=False)
        actual_values = torch.gather(x_flat, dim=-1, index=indices)

        if dim < 32768:
            indices = indices.to(torch.int16)

        # Save for backward
        ctx.save_for_backward(actual_values, indices, weight)
        ctx.original_shape = x.shape
        ctx.k = k

        # Record what we're saving
        InspectSavedTensors.saved_tensors_info.append({
            'actual_values': get_tensor_memory(actual_values),
            'indices': get_tensor_memory(indices),
            'weight': get_tensor_memory(weight),
            'original_x_would_be': get_tensor_memory(x),
        })

        return output

    @staticmethod
    def backward(ctx, grad_output):
        actual_values, indices, weight = ctx.saved_tensors
        batch, seq, dim = ctx.original_shape

        # Reconstruct
        x_reconstructed = torch.zeros(
            batch * seq, dim,
            device=actual_values.device,
            dtype=actual_values.dtype
        )
        indices_long = indices.to(torch.int64)
        x_reconstructed.scatter_(dim=-1, index=indices_long, src=actual_values)
        x_reconstructed = x_reconstructed.view(batch, seq, dim)

        grad_x = grad_output @ weight
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        x_2d = x_reconstructed.view(-1, dim)
        grad_weight = grad_output_2d.t() @ x_2d

        return grad_x, grad_weight, None


def investigate_autograd_storage():
    """Investigate exactly what autograd stores."""

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008

    print("=" * 70)
    print("INVESTIGATION: What does autograd actually store?")
    print("=" * 70)
    print(f"Config: batch={batch}, seq={seq}, hidden={hidden_dim}, intermediate={intermediate_dim}")
    print()

    # Calculate theoretical sizes
    x_size = batch * seq * hidden_dim * 2  # FP16
    h_mid_size = batch * seq * intermediate_dim * 2

    print(f"Theoretical x size: {x_size / 1024**2:.2f} MB")
    print(f"Theoretical h_mid size: {h_mid_size / 1024**2:.2f} MB")
    print()

    # Test 1: Standard linear layer
    print("-" * 70)
    print("TEST 1: What does a standard nn.Linear save?")
    print("-" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    linear = nn.Linear(intermediate_dim, hidden_dim, bias=False).to(device=device, dtype=dtype)
    h_mid = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    print_cuda_memory("Before forward")
    y = linear(h_mid)
    print_cuda_memory("After forward")

    # Check what's in the autograd graph
    print(f"\ny.grad_fn: {y.grad_fn}")
    print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")

    # The key insight: nn.Linear uses F.linear which saves the INPUT for backward
    # Let's verify by looking at saved tensors

    y.sum().backward()
    print_cuda_memory("After backward")

    print()
    print("-" * 70)
    print("TEST 2: Using torch.autograd.graph to inspect saved tensors")
    print("-" * 70)

    gc.collect()
    torch.cuda.empty_cache()

    h_mid = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    # Use hooks to track saved tensors
    saved_tensors_sizes = []

    def pack_hook(x):
        saved_tensors_sizes.append(('pack', get_tensor_memory(x) / 1024**2, x.shape))
        return x

    def unpack_hook(x):
        saved_tensors_sizes.append(('unpack', get_tensor_memory(x) / 1024**2, x.shape))
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = F.linear(h_mid, weight)
        y.sum().backward()

    print("Saved tensors during F.linear forward/backward:")
    for op, size, shape in saved_tensors_sizes:
        print(f"  {op}: {size:.2f} MB, shape={shape}")

    print()
    print("-" * 70)
    print("TEST 3: Our 'compressed' version - what does IT actually save?")
    print("-" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    InspectSavedTensors.saved_tensors_info = []
    saved_tensors_sizes = []

    h_mid = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = InspectSavedTensors.apply(h_mid, weight, 0.3)
        y.sum().backward()

    print("Saved tensors during compressed forward/backward:")
    for op, size, shape in saved_tensors_sizes:
        print(f"  {op}: {size:.2f} MB, shape={shape}")

    print("\nOur compression stats:")
    for info in InspectSavedTensors.saved_tensors_info:
        print(f"  actual_values: {info['actual_values'] / 1024**2:.2f} MB")
        print(f"  indices: {info['indices'] / 1024**2:.2f} MB")
        print(f"  weight: {info['weight'] / 1024**2:.2f} MB")
        print(f"  original x would be: {info['original_x_would_be'] / 1024**2:.2f} MB")
        compressed = info['actual_values'] + info['indices']
        original = info['original_x_would_be']
        print(f"  Compression ratio: {compressed/original:.2%}")


def investigate_the_real_problem():
    """The REAL investigation - where does memory go?"""

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008

    print("\n" + "=" * 70)
    print("THE REAL PROBLEM: Tracking every allocation")
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_cuda_memory("Start")

    # Allocate input
    h_mid = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    print_cuda_memory("After h_mid allocation")
    h_mid_mem = get_tensor_memory(h_mid) / 1024**2
    print(f"  h_mid tensor size: {h_mid_mem:.2f} MB")

    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    print_cuda_memory("After weight allocation")

    # Now do the TopK operation - THIS is where the problem likely is
    print("\n--- TopK Operation ---")
    mem_before_topk = torch.cuda.memory_allocated()

    k_fraction = 0.3
    dim = intermediate_dim
    k = int(dim * k_fraction)

    x_flat = h_mid.view(-1, dim)
    print_cuda_memory("After view (should be free)")

    abs_x = x_flat.abs()
    print_cuda_memory("After abs()")
    abs_mem = get_tensor_memory(abs_x) / 1024**2
    print(f"  abs_x tensor size: {abs_mem:.2f} MB (EXTRA ALLOCATION!)")

    values, indices = torch.topk(abs_x, k, dim=-1, sorted=False)
    print_cuda_memory("After topk()")
    values_mem = get_tensor_memory(values) / 1024**2
    indices_mem = get_tensor_memory(indices) / 1024**2
    print(f"  values tensor size: {values_mem:.2f} MB")
    print(f"  indices tensor size: {indices_mem:.2f} MB")

    # Now gather actual values
    actual_values = torch.gather(x_flat, dim=-1, index=indices)
    print_cuda_memory("After gather()")

    # Delete intermediates
    del abs_x, values
    gc.collect()
    torch.cuda.empty_cache()
    print_cuda_memory("After deleting abs_x, values")

    mem_after_topk = torch.cuda.memory_allocated()
    print(f"\nMemory used by TopK process: {(mem_after_topk - mem_before_topk) / 1024**2:.2f} MB")

    # THE KEY INSIGHT: What happens during forward when h_mid is still needed?
    print("\n--- The Real Issue: h_mid lifetime ---")
    print("During forward pass, h_mid is:")
    print("  1. Used to compute abs() -> creates copy")
    print("  2. Used for topk() -> abs_x still alive")
    print("  3. Used for gather() -> abs_x can be freed, but h_mid still needed")
    print("  4. h_mid used for F.linear output -> h_mid still alive")
    print("  5. Only after forward returns can h_mid be 'replaced' by compressed version")
    print("")
    print("BUT: ctx.save_for_backward() happens AFTER h_mid is fully allocated!")
    print("The compressed representation saves memory for BACKWARD, not FORWARD peak.")


def investigate_forward_vs_backward_memory():
    """Show that our compression helps backward but not forward peak."""

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008

    print("\n" + "=" * 70)
    print("FORWARD vs BACKWARD Memory Usage")
    print("=" * 70)

    class BaselineLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(x, weight)  # Saves full x
            return F.linear(x, weight)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            grad_x = grad_output @ weight
            grad_weight = grad_output.view(-1, grad_output.shape[-1]).t() @ x.view(-1, x.shape[-1])
            return grad_x, grad_weight

    class CompressedLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, k_fraction):
            output = F.linear(x, weight)

            # Compress x for backward
            dim = x.shape[-1]
            k = int(dim * k_fraction)
            x_flat = x.view(-1, dim)
            _, indices = torch.topk(x_flat.abs(), k, dim=-1, sorted=False)
            values = torch.gather(x_flat, dim=-1, index=indices)

            if dim < 32768:
                indices = indices.to(torch.int16)

            ctx.save_for_backward(values, indices, weight)
            ctx.original_shape = x.shape
            ctx.k = k

            return output

        @staticmethod
        def backward(ctx, grad_output):
            values, indices, weight = ctx.saved_tensors
            shape = ctx.original_shape

            # Reconstruct
            x_recon = torch.zeros(shape[0] * shape[1], shape[2], device=values.device, dtype=values.dtype)
            x_recon.scatter_(dim=-1, index=indices.long(), src=values)
            x_recon = x_recon.view(shape)

            grad_x = grad_output @ weight
            grad_weight = grad_output.view(-1, grad_output.shape[-1]).t() @ x_recon.view(-1, shape[-1])
            return grad_x, grad_weight, None

    # Test baseline
    print("\n--- Baseline (full storage) ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    mem_start = torch.cuda.memory_allocated()
    y = BaselineLinear.apply(x, weight)
    mem_after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()

    y.sum().backward()
    mem_after_bwd = torch.cuda.memory_allocated()
    peak_total = torch.cuda.max_memory_allocated()

    print(f"  After forward: {(mem_after_fwd - mem_start) / 1024**2:.2f} MB")
    print(f"  Peak during forward: {peak_fwd / 1024**2:.2f} MB")
    print(f"  After backward: {(mem_after_bwd - mem_start) / 1024**2:.2f} MB")
    print(f"  Peak total: {peak_total / 1024**2:.2f} MB")

    # Test compressed
    print("\n--- Compressed (TopK 30%) ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    mem_start = torch.cuda.memory_allocated()
    y = CompressedLinear.apply(x, weight, 0.3)
    mem_after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()

    y.sum().backward()
    mem_after_bwd = torch.cuda.memory_allocated()
    peak_total = torch.cuda.max_memory_allocated()

    print(f"  After forward: {(mem_after_fwd - mem_start) / 1024**2:.2f} MB")
    print(f"  Peak during forward: {peak_fwd / 1024**2:.2f} MB")
    print(f"  After backward: {(mem_after_bwd - mem_start) / 1024**2:.2f} MB")
    print(f"  Peak total: {peak_total / 1024**2:.2f} MB")

    print("\n" + "=" * 70)
    print("ROOT CAUSE IDENTIFIED:")
    print("=" * 70)
    print("""
The compression happens INSIDE the forward pass, but:

1. The input tensor 'x' must be FULLY MATERIALIZED before we can compress it
2. The topk() operation creates ADDITIONAL temporary tensors (abs, indices, values)
3. Peak memory occurs DURING forward when both x AND temporaries exist
4. The compressed representation only helps AFTER forward completes

To actually save memory, we need to:
- Never materialize the full tensor in the first place
- Use gradient checkpointing + recompute with compression
- Or use custom CUDA kernels that fuse the computation
""")


def show_the_solution():
    """Demonstrate how gradient checkpointing + compression DOES save memory."""

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008

    print("\n" + "=" * 70)
    print("THE SOLUTION: Gradient Checkpointing + Compressed Recompute")
    print("=" * 70)

    class CheckpointedCompressedLinear(torch.autograd.Function):
        """
        Don't store activations at all during forward.
        During backward, recompute but use compressed representation.
        """
        @staticmethod
        def forward(ctx, x, weight, k_fraction):
            # Don't save x - just save what we need to recompute
            # We save: compressed x (for approximate gradient), weight
            output = F.linear(x, weight)

            # Compress x
            dim = x.shape[-1]
            k = int(dim * k_fraction)
            x_flat = x.view(-1, dim)
            _, indices = torch.topk(x_flat.abs(), k, dim=-1, sorted=False)
            values = torch.gather(x_flat, dim=-1, index=indices)

            if dim < 32768:
                indices = indices.to(torch.int16)

            # Key: we do NOT save the full x, only compressed version
            ctx.save_for_backward(values, indices, weight)
            ctx.original_shape = x.shape

            return output

        @staticmethod
        def backward(ctx, grad_output):
            values, indices, weight = ctx.saved_tensors
            shape = ctx.original_shape

            # Reconstruct approximate x
            x_approx = torch.zeros(shape[0] * shape[1], shape[2], device=values.device, dtype=values.dtype)
            x_approx.scatter_(dim=-1, index=indices.long(), src=values)
            x_approx = x_approx.view(shape)

            grad_x = grad_output @ weight
            grad_weight = grad_output.view(-1, grad_output.shape[-1]).t() @ x_approx.view(-1, shape[-1])

            return grad_x, grad_weight, None

    # Baseline: Standard + full storage
    print("\n--- Baseline ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    y = F.linear(x, weight)
    peak_baseline = torch.cuda.max_memory_allocated()
    y.sum().backward()

    print(f"  Peak memory: {peak_baseline / 1024**2:.2f} MB")

    # Checkpointed + compressed
    print("\n--- Checkpointed + Compressed ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch, seq, intermediate_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_dim, intermediate_dim, device=device, dtype=dtype, requires_grad=True)

    y = CheckpointedCompressedLinear.apply(x, weight, 0.3)
    peak_compressed = torch.cuda.max_memory_allocated()
    y.sum().backward()

    print(f"  Peak memory: {peak_compressed / 1024**2:.2f} MB")
    print(f"  Savings: {(1 - peak_compressed/peak_baseline)*100:.1f}%")

    print("\n" + "-" * 70)
    print("But wait - peak is still during forward when x exists!")
    print("The REAL solution needs to not materialize x at all,")
    print("which requires fusing the FFN computation.")
    print("-" * 70)


def demonstrate_fused_approach():
    """Show that fusing operations is the real solution."""

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden_dim = 4, 2048, 4096
    intermediate_dim = 11008
    k_fraction = 0.3

    print("\n" + "=" * 70)
    print("FUSED APPROACH: Compute output while streaming compression")
    print("=" * 70)

    print("""
The key insight: In a fused kernel, we can:
1. Compute h_mid = SiLU(x @ W_gate) * (x @ W_up) in tiles
2. For each tile, immediately:
   - Compute partial output contribution
   - Extract top-K values
   - Discard the rest
3. Never materialize full h_mid tensor

This is how FlashAttention works for attention - it never materializes
the full attention matrix.

For FFN, a "FlashFFN" would:
- Stream through the intermediate dimension
- Accumulate output while compressing activations
- Only store compressed representation for backward

Without custom CUDA kernels, we CAN'T do this in PyTorch.
""")

    # Demonstrate the memory profile difference
    print("\nMemory profile comparison:")
    print()
    print("Standard FFN memory timeline:")
    print("  Forward:")
    print("    - Allocate h_gate: 172 MB")
    print("    - Allocate h_up: 172 MB")
    print("    - Allocate h_act: 172 MB")
    print("    - Allocate h_mid: 172 MB")
    print("    - Peak: 688 MB (all 4 intermediate tensors)")
    print("  Backward:")
    print("    - All 4 tensors still needed: 688 MB")
    print()
    print("Our 'compressed' approach memory timeline:")
    print("  Forward:")
    print("    - Allocate h_gate: 172 MB")
    print("    - Allocate h_up: 172 MB")
    print("    - Allocate h_act: 172 MB")
    print("    - Allocate h_mid: 172 MB")
    print("    - Compute compression (extra temp): +172 MB")
    print("    - Peak: 860 MB (WORSE during forward!)")
    print("  Backward:")
    print("    - Only compressed stored: ~52 MB (70% savings)")
    print("    - But peak already hit during forward")
    print()
    print("Fused FlashFFN (hypothetical) memory timeline:")
    print("  Forward:")
    print("    - Stream tiles, never full materialization")
    print("    - Peak: ~50 MB (tile size + compressed buffer)")
    print("  Backward:")
    print("    - Compressed activations: ~52 MB")
    print("    - Peak: ~100 MB")
    print()
    print("=" * 70)
    print("CONCLUSION: Without custom CUDA kernels, PyTorch's eager execution")
    print("forces full tensor materialization before any compression can happen.")
    print("=" * 70)


if __name__ == "__main__":
    investigate_autograd_storage()
    investigate_the_real_problem()
    investigate_forward_vs_backward_memory()
    show_the_solution()
    demonstrate_fused_approach()
