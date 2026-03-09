"""
Verify PEFT LoRA + FlashFFN effective-weight integration.

Tests:
1. Forward output equivalence (LoRA standard vs LoRA+FlashFFN)
2. Gradient flow to all trainable params (lora_A and lora_B)
3. Gradient cosine similarity (standard LoRA vs FlashFFN gradients)
4. Training step produces parameter updates
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType
from flashffn import FlashFFNFunction


def _is_lora_wrapped(linear_module):
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'lora_A')
            and len(getattr(linear_module, 'lora_A', {})) > 0
            and hasattr(linear_module, 'lora_B')
            and len(getattr(linear_module, 'lora_B', {})) > 0
            and len(getattr(linear_module, 'lora_magnitude_vector', {})) == 0
            and not hasattr(linear_module, 'lora_E'))


def _compute_lora_effective_weight(lora_linear):
    adapter_name = lora_linear.active_adapters[0]
    base_weight = lora_linear.get_base_layer().weight
    lora_A_weight = lora_linear.lora_A[adapter_name].weight
    lora_B_weight = lora_linear.lora_B[adapter_name].weight
    scaling = lora_linear.scaling[adapter_name]
    delta_w = lora_B_weight @ lora_A_weight
    w_eff = base_weight + scaling * delta_w
    w_eff = w_eff.to(base_weight.dtype)
    return w_eff


def cosine_sim(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _make_peft_lora_model(device, dtype, num_layers=2, target_modules=None):
    """Create a TinyLlama model with PEFT LoRA applied."""
    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=num_layers,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = PeftLoraConfig(
        r=8, lora_alpha=16, target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS, use_dora=False,
    )
    model = get_peft_model(model, peft_config)
    return model


def test_forward_equivalence():
    """Test that LoRA+FlashFFN produces the same output as standard LoRA."""
    print("=" * 60)
    print("TEST 1: Forward Output Equivalence")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    model = _make_peft_lora_model(device, dtype)
    model.eval()

    mlp = model.base_model.model.model.layers[0].mlp
    assert _is_lora_wrapped(mlp.gate_proj), "gate_proj should be LoRA-wrapped"
    assert _is_lora_wrapped(mlp.up_proj), "up_proj should be LoRA-wrapped"
    assert _is_lora_wrapped(mlp.down_proj), "down_proj should be LoRA-wrapped"

    x = torch.randn(2, 64, 2048, device=device, dtype=dtype)

    # Standard LoRA forward (through PEFT modules)
    with torch.no_grad():
        h_gate_std = mlp.gate_proj(x)
        h_up_std = mlp.up_proj(x)
        h_mid_std = F.silu(h_gate_std) * h_up_std
        y_std = mlp.down_proj(h_mid_std)

    # FlashFFN LoRA forward (effective weight approach)
    with torch.no_grad():
        w_gate_eff = _compute_lora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_lora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_lora_effective_weight(mlp.down_proj)
        y_flash = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    max_diff = (y_std - y_flash).abs().max().item()
    mean_diff = (y_std - y_flash).abs().mean().item()
    cos = cosine_sim(y_std, y_flash)

    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Cosine:    {cos:.8f}")

    ok = cos > 0.9999
    print(f"  PASS: {ok}")
    return ok


def test_gradient_flow():
    """Test that gradients flow to all LoRA trainable params through FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Flow")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    model = _make_peft_lora_model(device, dtype)
    model.train()

    mlp = model.base_model.model.model.layers[0].mlp

    x = torch.randn(2, 32, 2048, device=device, dtype=dtype, requires_grad=True)
    w_gate_eff = _compute_lora_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_lora_effective_weight(mlp.up_proj)
    w_down_eff = _compute_lora_effective_weight(mlp.down_proj)
    y = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    loss = y.sum()
    loss.backward()

    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, proj_name)
        adapter_name = proj.active_adapters[0]

        # LoRA A — at init, lora_B=0, so grad_lora_A = lora_B.T @ grad = 0. Expected.
        lora_a = proj.lora_A[adapter_name].weight
        has_grad_a = lora_a.grad is not None
        grad_a_max = lora_a.grad.abs().max().item() if has_grad_a else -1
        print(f"  {proj_name}.lora_A grad: {'EXISTS' if has_grad_a else 'MISSING'}"
              f" (max={grad_a_max:.4e}, zero expected at init since lora_B=0)")
        all_ok = all_ok and has_grad_a

        # LoRA B — should have nonzero grad (from grad_w_eff @ lora_A.T... but lora_A is random, so B gets real grad)
        lora_b = proj.lora_B[adapter_name].weight
        has_grad_b = lora_b.grad is not None and lora_b.grad.abs().max().item() > 0
        grad_b_max = lora_b.grad.abs().max().item() if lora_b.grad is not None else -1
        print(f"  {proj_name}.lora_B grad: {'YES' if has_grad_b else 'NO'} (max={grad_b_max:.4e})")
        all_ok = all_ok and has_grad_b

        # Base weight should NOT have grad (frozen)
        base_w = proj.get_base_layer().weight
        has_grad_base = base_w.grad is not None and base_w.grad.abs().max().item() > 0
        print(f"  {proj_name}.base_weight grad: {'NO (frozen)' if not has_grad_base else 'YES (BAD!)'}")
        all_ok = all_ok and (not has_grad_base)

    # Input gradient
    has_grad_x = x.grad is not None and x.grad.abs().max().item() > 0
    print(f"  input x grad: {'YES' if has_grad_x else 'NO'}")
    all_ok = all_ok and has_grad_x

    print(f"  PASS: {all_ok}")
    return all_ok


def test_gradient_similarity():
    """Compare gradients from standard LoRA vs LoRA+FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Cosine Similarity (Standard LoRA vs FlashFFN)")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    target_modules = ["gate_proj", "up_proj", "down_proj"]

    # --- Run 1: Standard LoRA ---
    model1 = _make_peft_lora_model(device, dtype, target_modules=target_modules)
    model1.train()

    # --- Run 2: LoRA+FlashFFN (clone weights from model1) ---
    model2 = _make_peft_lora_model(device, dtype, target_modules=target_modules)
    model2.train()

    # Copy all trainable weights from model1 to model2
    state1 = {n: p.data.clone() for n, p in model1.named_parameters() if p.requires_grad}
    for n, p in model2.named_parameters():
        if n in state1:
            p.data.copy_(state1[n])

    # Same input
    torch.manual_seed(42)
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype)

    # Standard LoRA forward + backward
    mlp1 = model1.base_model.model.model.layers[0].mlp
    h_gate = mlp1.gate_proj(x)
    h_up = mlp1.up_proj(x)
    h_mid = F.silu(h_gate) * h_up
    y1 = mlp1.down_proj(h_mid)
    y1.sum().backward()

    # FlashFFN LoRA forward + backward
    mlp2 = model2.base_model.model.model.layers[0].mlp
    w_gate_eff = _compute_lora_effective_weight(mlp2.gate_proj)
    w_up_eff = _compute_lora_effective_weight(mlp2.up_proj)
    w_down_eff = _compute_lora_effective_weight(mlp2.down_proj)
    y2 = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)
    y2.sum().backward()

    # Compare output
    cos_output = cosine_sim(y1.detach(), y2.detach())
    print(f"  Output cosine:  {cos_output:.8f}")

    # Compare gradients for each projection
    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj1 = getattr(mlp1, proj_name)
        proj2 = getattr(mlp2, proj_name)
        adapter_name = proj1.active_adapters[0]

        # lora_B grad comparison
        g1_b = proj1.lora_B[adapter_name].weight.grad
        g2_b = proj2.lora_B[adapter_name].weight.grad
        if g1_b is not None and g2_b is not None:
            cos_b = cosine_sim(g1_b, g2_b)
            print(f"  {proj_name}.lora_B grad cosine: {cos_b:.6f}")
            # gate_proj and up_proj should be exact (grad via tiled recompute)
            # down_proj is approximate (uses sparse top-K h_mid)
            if proj_name in ('gate_proj', 'up_proj'):
                all_ok = all_ok and (cos_b > 0.99)
            else:
                all_ok = all_ok and (cos_b > 0.95)

        # lora_A grad comparison — at init, lora_B=0, so both standard and
        # FlashFFN produce zero grad for lora_A. Skip cosine check if both zero.
        g1_a = proj1.lora_A[adapter_name].weight.grad
        g2_a = proj2.lora_A[adapter_name].weight.grad
        if g1_a is not None and g2_a is not None:
            both_zero = g1_a.abs().max().item() < 1e-8 and g2_a.abs().max().item() < 1e-8
            if both_zero:
                print(f"  {proj_name}.lora_A grad cosine: SKIP (both zero at init, lora_B=0)")
            else:
                cos_a = cosine_sim(g1_a, g2_a)
                print(f"  {proj_name}.lora_A grad cosine: {cos_a:.6f}")
                if proj_name in ('gate_proj', 'up_proj'):
                    all_ok = all_ok and (cos_a > 0.99)
                else:
                    all_ok = all_ok and (cos_a > 0.95)

    print(f"  PASS: {all_ok}")
    return all_ok


def test_training_step():
    """Test that an optimizer step updates all trainable params."""
    print("\n" + "=" * 60)
    print("TEST 4: Training Step (Parameter Updates)")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    model = _make_peft_lora_model(device, dtype)
    model.train()

    # Snapshot initial weights
    initial_state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            initial_state[n] = p.data.clone()

    # Forward through FlashFFN
    mlp = model.base_model.model.model.layers[0].mlp
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype)
    w_gate_eff = _compute_lora_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_lora_effective_weight(mlp.up_proj)
    w_down_eff = _compute_lora_effective_weight(mlp.down_proj)
    y = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    loss = y.sum()
    loss.backward()

    # Optimizer step
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    optimizer.step()

    # Check which params changed — only for layer 0 MLP (where we computed forward)
    updated = 0
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n in initial_state and "layers.0.mlp" in n:
            total += 1
            diff = (p.data - initial_state[n]).abs().max().item()
            if diff > 0:
                updated += 1
                print(f"    Updated: {n} (diff={diff:.4e})")
            else:
                print(f"    Not updated: {n} (zero grad at init expected for lora_A since lora_B=0)")

    print(f"  Layer 0 MLP trainable params: {total}")
    print(f"  Layer 0 MLP updated params:   {updated}")
    # lora_B params should update (6 total: 3 projections × lora_A + lora_B,
    # but lora_A grad is zero at init because lora_B=0, so only lora_B updates → 3)
    ok = updated >= 3  # At least lora_B for gate/up/down should update
    print(f"  PASS: {ok} ({updated}/{total} updated, need >= 3)")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("Forward Equivalence", test_forward_equivalence()))
    torch.cuda.empty_cache()
    results.append(("Gradient Flow", test_gradient_flow()))
    torch.cuda.empty_cache()
    results.append(("Gradient Similarity", test_gradient_similarity()))
    torch.cuda.empty_cache()
    results.append(("Training Step", test_training_step()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        all_pass = all_pass and ok

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
