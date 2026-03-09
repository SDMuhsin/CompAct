"""
Comprehensive verification tests for FlashFFN.

Tests:
1. Output correctness across various configurations
2. Gradient accuracy analysis
3. Memory savings verification
4. Numerical stability tests
5. Edge case handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import json
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, 'src')
from flashffn import FlashFFN, StandardFFN, FlashFFNUltraLight


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative L2 error."""
    return (a - b).norm().item() / (b.norm().item() + 1e-8)


def test_output_correctness_various_configs():
    """Test output correctness across various configurations."""
    print("\n" + "=" * 70)
    print("TEST 1: Output Correctness Across Configurations")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float16

    configs = [
        # (batch, seq, hidden, intermediate, name)
        (1, 128, 512, 1408, "Small"),
        (2, 256, 768, 2048, "Medium"),
        (4, 512, 1024, 2816, "Large"),
        (4, 2048, 4096, 11008, "LLaMA-7B scale"),
        (8, 1024, 2048, 5632, "Wide batch"),
        (1, 4096, 1024, 2816, "Long sequence"),
    ]

    k_fractions = [0.1, 0.2, 0.3, 0.5]

    results = []

    for batch, seq, hidden, intermediate, name in configs:
        for k_frac in k_fractions:
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # Create modules
                flash_ffn = FlashFFN(hidden, intermediate, k_fraction=k_frac).to(device=device, dtype=dtype)
                standard_ffn = StandardFFN(hidden, intermediate).to(device=device, dtype=dtype)

                # Copy weights
                standard_ffn.gate_proj.weight.data = flash_ffn.gate_proj.weight.data.clone()
                standard_ffn.up_proj.weight.data = flash_ffn.up_proj.weight.data.clone()
                standard_ffn.down_proj.weight.data = flash_ffn.down_proj.weight.data.clone()

                # Test input
                x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)

                # Forward
                y_flash = flash_ffn(x)
                y_standard = standard_ffn(x)

                # Compare
                max_diff = (y_flash - y_standard).abs().max().item()
                mean_diff = (y_flash - y_standard).abs().mean().item()
                cos_sim = cosine_similarity(y_flash, y_standard)

                result = {
                    'config': name,
                    'shape': f"({batch}, {seq}, {hidden}, {intermediate})",
                    'k_fraction': k_frac,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'cosine_sim': cos_sim,
                    'passed': max_diff < 0.01,  # FP16 tolerance
                }
                results.append(result)

                status = "PASS" if result['passed'] else "FAIL"
                print(f"  {name} (k={k_frac}): max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f} [{status}]")

                del flash_ffn, standard_ffn, x, y_flash, y_standard

            except torch.cuda.OutOfMemoryError:
                print(f"  {name} (k={k_frac}): OOM - skipped")
                results.append({
                    'config': name,
                    'k_fraction': k_frac,
                    'passed': None,
                    'error': 'OOM',
                })

    passed = sum(1 for r in results if r.get('passed') == True)
    failed = sum(1 for r in results if r.get('passed') == False)
    skipped = sum(1 for r in results if r.get('passed') is None)

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    return results


def test_gradient_accuracy():
    """Test gradient accuracy across configurations."""
    print("\n" + "=" * 70)
    print("TEST 2: Gradient Accuracy Analysis")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float32  # Use FP32 for gradient comparison

    configs = [
        (2, 128, 512, 1408, "Small"),
        (2, 256, 768, 2048, "Medium"),
        (2, 512, 1024, 2816, "Large"),
    ]

    k_fractions = [0.1, 0.2, 0.3, 0.5]

    results = []

    for batch, seq, hidden, intermediate, name in configs:
        for k_frac in k_fractions:
            gc.collect()
            torch.cuda.empty_cache()

            # Create modules
            flash_ffn = FlashFFN(hidden, intermediate, k_fraction=k_frac).to(device=device, dtype=dtype)
            standard_ffn = StandardFFN(hidden, intermediate).to(device=device, dtype=dtype)

            # Copy weights
            standard_ffn.gate_proj.weight.data = flash_ffn.gate_proj.weight.data.clone()
            standard_ffn.up_proj.weight.data = flash_ffn.up_proj.weight.data.clone()
            standard_ffn.down_proj.weight.data = flash_ffn.down_proj.weight.data.clone()

            # Test input
            x_flash = torch.randn(batch, seq, hidden, device=device, dtype=dtype, requires_grad=True)
            x_standard = x_flash.clone().detach().requires_grad_(True)

            # Forward + backward
            y_flash = flash_ffn(x_flash)
            y_standard = standard_ffn(x_standard)

            y_flash.sum().backward()
            y_standard.sum().backward()

            # Compare gradients
            grad_x_sim = cosine_similarity(x_flash.grad, x_standard.grad)
            grad_wgate_sim = cosine_similarity(flash_ffn.gate_proj.weight.grad, standard_ffn.gate_proj.weight.grad)
            grad_wup_sim = cosine_similarity(flash_ffn.up_proj.weight.grad, standard_ffn.up_proj.weight.grad)
            grad_wdown_sim = cosine_similarity(flash_ffn.down_proj.weight.grad, standard_ffn.down_proj.weight.grad)

            result = {
                'config': name,
                'k_fraction': k_frac,
                'grad_x_sim': grad_x_sim,
                'grad_wgate_sim': grad_wgate_sim,
                'grad_wup_sim': grad_wup_sim,
                'grad_wdown_sim': grad_wdown_sim,
                'min_sim': min(grad_x_sim, grad_wgate_sim, grad_wup_sim, grad_wdown_sim),
            }
            results.append(result)

            print(f"  {name} (k={k_frac}): x={grad_x_sim:.4f}, gate={grad_wgate_sim:.4f}, "
                  f"up={grad_wup_sim:.4f}, down={grad_wdown_sim:.4f}")

            del flash_ffn, standard_ffn, x_flash, x_standard

    print(f"\n  Summary:")
    for k_frac in k_fractions:
        k_results = [r for r in results if r['k_fraction'] == k_frac]
        avg_down = sum(r['grad_wdown_sim'] for r in k_results) / len(k_results)
        min_down = min(r['grad_wdown_sim'] for r in k_results)
        print(f"    k={k_frac}: avg grad_w_down similarity = {avg_down:.4f}, min = {min_down:.4f}")

    return results


def test_memory_savings_detailed():
    """Detailed memory savings analysis."""
    print("\n" + "=" * 70)
    print("TEST 3: Memory Savings Analysis")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float16

    configs = [
        (4, 512, 1024, 2816, "Small model"),
        (4, 1024, 2048, 5632, "Medium model"),
        (4, 2048, 4096, 11008, "LLaMA-7B scale"),
    ]

    results = []

    for batch, seq, hidden, intermediate, name in configs:
        print(f"\n  {name} ({batch}x{seq}x{hidden}, intermediate={intermediate}):")

        def measure_memory(ffn_class, ffn_name, k_frac=0.3):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if 'Standard' in ffn_name:
                ffn = ffn_class(hidden, intermediate).to(device=device, dtype=dtype)
            else:
                ffn = ffn_class(hidden, intermediate, k_fraction=k_frac).to(device=device, dtype=dtype)

            x = torch.randn(batch, seq, hidden, device=device, dtype=dtype, requires_grad=True)

            mem_before = torch.cuda.memory_allocated()
            y = ffn(x)
            mem_after_fwd = torch.cuda.memory_allocated()
            peak_fwd = torch.cuda.max_memory_allocated()

            torch.cuda.reset_peak_memory_stats()
            y.sum().backward()
            peak_bwd = torch.cuda.max_memory_allocated()

            stored = (mem_after_fwd - mem_before) / 1024**2

            del ffn, x, y
            gc.collect()
            torch.cuda.empty_cache()

            return {
                'stored_mb': stored,
                'peak_fwd_mb': peak_fwd / 1024**2,
                'peak_bwd_mb': peak_bwd / 1024**2,
            }

        try:
            std = measure_memory(StandardFFN, "Standard")
            flash = measure_memory(FlashFFN, "FlashFFN")

            stored_savings = (1 - flash['stored_mb'] / std['stored_mb']) * 100
            fwd_savings = (1 - flash['peak_fwd_mb'] / std['peak_fwd_mb']) * 100

            print(f"    Standard: stored={std['stored_mb']:.1f}MB, peak_fwd={std['peak_fwd_mb']:.1f}MB")
            print(f"    FlashFFN: stored={flash['stored_mb']:.1f}MB, peak_fwd={flash['peak_fwd_mb']:.1f}MB")
            print(f"    Savings: stored={stored_savings:.1f}%, peak_fwd={fwd_savings:.1f}%")

            results.append({
                'config': name,
                'standard': std,
                'flashffn': flash,
                'stored_savings_pct': stored_savings,
                'peak_fwd_savings_pct': fwd_savings,
            })

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM - skipped")
            results.append({'config': name, 'error': 'OOM'})

    return results


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\n" + "=" * 70)
    print("TEST 4: Numerical Stability")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float16

    batch, seq, hidden, intermediate = 2, 128, 512, 1408

    flash_ffn = FlashFFN(hidden, intermediate, k_fraction=0.3).to(device=device, dtype=dtype)
    standard_ffn = StandardFFN(hidden, intermediate).to(device=device, dtype=dtype)

    standard_ffn.gate_proj.weight.data = flash_ffn.gate_proj.weight.data.clone()
    standard_ffn.up_proj.weight.data = flash_ffn.up_proj.weight.data.clone()
    standard_ffn.down_proj.weight.data = flash_ffn.down_proj.weight.data.clone()

    test_cases = [
        ("Normal input", torch.randn(batch, seq, hidden, device=device, dtype=dtype)),
        ("Small values", torch.randn(batch, seq, hidden, device=device, dtype=dtype) * 0.001),
        ("Large values", torch.randn(batch, seq, hidden, device=device, dtype=dtype) * 10),
        ("Sparse input", torch.randn(batch, seq, hidden, device=device, dtype=dtype) * (torch.rand(batch, seq, hidden, device=device, dtype=dtype) > 0.9)),
        ("Near-zero", torch.zeros(batch, seq, hidden, device=device, dtype=dtype) + 1e-4),
    ]

    results = []

    for name, x in test_cases:
        y_flash = flash_ffn(x)
        y_standard = standard_ffn(x)

        max_diff = (y_flash - y_standard).abs().max().item()
        has_nan = torch.isnan(y_flash).any().item()
        has_inf = torch.isinf(y_flash).any().item()

        status = "PASS" if (max_diff < 0.1 and not has_nan and not has_inf) else "FAIL"
        print(f"  {name}: max_diff={max_diff:.6f}, nan={has_nan}, inf={has_inf} [{status}]")

        results.append({
            'test': name,
            'max_diff': max_diff,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'passed': status == "PASS",
        })

    return results


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("TEST 5: Edge Cases")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float16

    test_cases = [
        ("Batch=1", 1, 128, 512, 1408, 0.3),
        ("Seq=1", 4, 1, 512, 1408, 0.3),
        ("Very long seq", 1, 4096, 512, 1408, 0.3),
        ("Small hidden", 4, 128, 64, 176, 0.3),
        ("k=0.1 (very sparse)", 4, 128, 512, 1408, 0.1),
        ("k=0.5 (half)", 4, 128, 512, 1408, 0.5),
    ]

    results = []

    for name, batch, seq, hidden, intermediate, k_frac in test_cases:

        gc.collect()
        torch.cuda.empty_cache()

        try:
            flash_ffn = FlashFFN(hidden, intermediate, k_fraction=k_frac).to(device=device, dtype=dtype)
            standard_ffn = StandardFFN(hidden, intermediate).to(device=device, dtype=dtype)

            standard_ffn.gate_proj.weight.data = flash_ffn.gate_proj.weight.data.clone()
            standard_ffn.up_proj.weight.data = flash_ffn.up_proj.weight.data.clone()
            standard_ffn.down_proj.weight.data = flash_ffn.down_proj.weight.data.clone()

            x = torch.randn(batch, seq, hidden, device=device, dtype=dtype, requires_grad=True)

            y_flash = flash_ffn(x.clone().detach().requires_grad_(True))
            y_standard = standard_ffn(x)

            max_diff = (y_flash - y_standard).abs().max().item()
            passed = max_diff < 0.01

            print(f"  {name}: shape=({batch},{seq},{hidden},{intermediate}), k={k_frac}, "
                  f"max_diff={max_diff:.6f} [{'PASS' if passed else 'FAIL'}]")

            results.append({
                'test': name,
                'shape': (batch, seq, hidden, intermediate),
                'k_fraction': k_frac,
                'max_diff': max_diff,
                'passed': passed,
            })

            del flash_ffn, standard_ffn, x

        except Exception as e:
            print(f"  {name}: ERROR - {str(e)[:50]}")
            results.append({
                'test': name,
                'error': str(e),
                'passed': False,
            })

    return results


def run_all_tests():
    """Run all verification tests and save results."""
    print("=" * 70)
    print("FLASHFFN COMPREHENSIVE VERIFICATION")
    print("=" * 70)

    all_results = {}

    all_results['output_correctness'] = test_output_correctness_various_configs()
    all_results['gradient_accuracy'] = test_gradient_accuracy()
    all_results['memory_savings'] = test_memory_savings_detailed()
    all_results['numerical_stability'] = test_numerical_stability()
    all_results['edge_cases'] = test_edge_cases()

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    # Count passes/fails for each test category
    for test_name, results in all_results.items():
        if isinstance(results, list):
            passed = sum(1 for r in results if r.get('passed') == True)
            failed = sum(1 for r in results if r.get('passed') == False)
            total = len(results)
            print(f"  {test_name}: {passed}/{total} passed")

    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, float):
            if obj != obj:  # NaN check
                return "NaN"
            elif obj == float('inf'):
                return "Inf"
            elif obj == float('-inf'):
                return "-Inf"
            return obj
        else:
            return obj

    with open("/workspace/CompAct/results/flashffn_verification.json", "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print("\n  Results saved to /workspace/actcomp/results/flashffn_verification.json")

    return all_results


if __name__ == "__main__":
    run_all_tests()
