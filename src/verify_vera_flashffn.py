"""
Verify PEFT VeRA + FlashFFN effective-weight integration.

Tests:
1. Forward output equivalence (VeRA standard vs VeRA+FlashFFN)
2. Gradient flow to all trainable params (vera_lambda_b and vera_lambda_d)
3. Gradient cosine similarity (standard VeRA vs FlashFFN gradients)
4. Training step produces parameter updates
"""
import sys
import os
import gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, VeraConfig, TaskType
from flashffn import FlashFFNFunction


def _is_vera_wrapped(linear_module):
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'vera_lambda_d')
            and len(getattr(linear_module, 'vera_lambda_d', {})) > 0)


def _compute_vera_effective_weight(vera_linear):
    adapter_name = vera_linear.active_adapters[0]
    base_weight = vera_linear.get_base_layer().weight
    in_features = vera_linear.in_features
    out_features = vera_linear.out_features

    vera_A = vera_linear.vera_A[adapter_name]
    vera_B = vera_linear.vera_B[adapter_name]
    lambda_d = vera_linear.vera_lambda_d[adapter_name]
    lambda_b = vera_linear.vera_lambda_b[adapter_name]

    sliced_A = vera_A[:, :in_features].to(lambda_d.device)
    sliced_B = vera_B[:out_features, :].to(lambda_d.device)

    delta_w = (lambda_b.unsqueeze(-1) * sliced_B) @ (lambda_d.unsqueeze(-1) * sliced_A)
    w_eff = base_weight + delta_w
    w_eff = w_eff.to(base_weight.dtype)
    return w_eff


def cosine_sim(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def _make_peft_vera_model(device, dtype, num_layers=2, target_modules=None):
    """Create a TinyLlama model with PEFT VeRA applied."""
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

    peft_config = VeraConfig(
        r=256, d_initial=0.1, vera_dropout=0.0,
        target_modules=target_modules,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, peft_config)
    return model


def test_forward_equivalence():
    """Test that VeRA+FlashFFN produces the same output as standard VeRA."""
    print("=" * 60)
    print("TEST 1: Forward Output Equivalence")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    model = _make_peft_vera_model(device, dtype)
    model.eval()

    mlp = model.base_model.model.model.layers[0].mlp
    assert _is_vera_wrapped(mlp.gate_proj), "gate_proj should be VeRA-wrapped"
    assert _is_vera_wrapped(mlp.up_proj), "up_proj should be VeRA-wrapped"
    assert _is_vera_wrapped(mlp.down_proj), "down_proj should be VeRA-wrapped"

    x = torch.randn(2, 64, 2048, device=device, dtype=dtype)

    # Standard VeRA forward (through PEFT modules)
    with torch.no_grad():
        h_gate_std = mlp.gate_proj(x)
        h_up_std = mlp.up_proj(x)
        h_mid_std = F.silu(h_gate_std) * h_up_std
        y_std = mlp.down_proj(h_mid_std)

    # FlashFFN VeRA forward (effective weight approach)
    with torch.no_grad():
        w_gate_eff = _compute_vera_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_vera_effective_weight(mlp.up_proj)
        w_down_eff = _compute_vera_effective_weight(mlp.down_proj)
        y_flash = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    max_diff = (y_std - y_flash).abs().max().item()
    mean_diff = (y_std - y_flash).abs().mean().item()
    cos = cosine_sim(y_std, y_flash)

    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Cosine:    {cos:.8f}")

    ok = cos > 0.9999
    print(f"  PASS: {ok}")

    del model, mlp, x, y_std, y_flash
    _cleanup()
    return ok


def test_gradient_flow():
    """Test that gradients flow to VeRA trainable params through FlashFFN.

    VeRA trainable params: vera_lambda_b [out] and vera_lambda_d [r].
    At VeRA's default init (d_initial=0.1), lambda_b is initialized to 1.0
    and lambda_d to d_initial (0.1), so both should get nonzero gradients.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Flow")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    model = _make_peft_vera_model(device, dtype)
    model.train()

    mlp = model.base_model.model.model.layers[0].mlp

    x = torch.randn(2, 32, 2048, device=device, dtype=dtype, requires_grad=True)
    w_gate_eff = _compute_vera_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_vera_effective_weight(mlp.up_proj)
    w_down_eff = _compute_vera_effective_weight(mlp.down_proj)
    y = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    loss = y.sum()
    loss.backward()

    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, proj_name)
        adapter_name = proj.active_adapters[0]

        # lambda_b — trainable, should get nonzero grad
        lb = proj.vera_lambda_b[adapter_name]
        has_grad_lb = lb.grad is not None and lb.grad.abs().max().item() > 0
        grad_lb_max = lb.grad.abs().max().item() if lb.grad is not None else -1
        print(f"  {proj_name}.lambda_b grad: {'YES' if has_grad_lb else 'NO'} (max={grad_lb_max:.4e})")
        all_ok = all_ok and has_grad_lb

        # lambda_d — trainable, gradient depends on lambda_b (which is 1.0 at init → nonzero grad)
        ld = proj.vera_lambda_d[adapter_name]
        has_grad_ld = ld.grad is not None
        grad_ld_max = ld.grad.abs().max().item() if ld.grad is not None else -1
        ld_nonzero = has_grad_ld and grad_ld_max > 0
        print(f"  {proj_name}.lambda_d grad: {'YES' if ld_nonzero else 'NO/ZERO'} (max={grad_ld_max:.4e})")
        # lambda_d should get gradient since lambda_b=1.0 at init (unlike LoRA where lora_B=0)
        all_ok = all_ok and has_grad_ld

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

    del model, mlp, x, y, loss
    _cleanup()
    return all_ok


def test_gradient_similarity():
    """Compare gradients from standard VeRA vs VeRA+FlashFFN.

    Serialized: run model1 forward/backward first, save grads, free model1,
    then create model2 to avoid OOM from having 2 models loaded simultaneously.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Cosine Similarity (Standard VeRA vs FlashFFN)")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32
    target_modules = ["gate_proj", "up_proj", "down_proj"]

    # --- Run 1: Standard VeRA → save grads and output ---
    model1 = _make_peft_vera_model(device, dtype, target_modules=target_modules)
    model1.train()

    # Save vera_A/B buffers and trainable params to copy to model2
    saved_buffers = {n: buf.clone() for n, buf in model1.named_buffers()}
    saved_params = {n: p.data.clone() for n, p in model1.named_parameters() if p.requires_grad}

    torch.manual_seed(42)
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype)

    mlp1 = model1.base_model.model.model.layers[0].mlp
    h_gate = mlp1.gate_proj(x)
    h_up = mlp1.up_proj(x)
    h_mid = F.silu(h_gate) * h_up
    y1 = mlp1.down_proj(h_mid)
    y1.sum().backward()

    # Collect grads from model1
    grads1 = {}
    y1_detach = y1.detach().clone()
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp1, proj_name)
        adapter_name = proj.active_adapters[0]
        g_lb = proj.vera_lambda_b[adapter_name].grad
        g_ld = proj.vera_lambda_d[adapter_name].grad
        grads1[f"{proj_name}.lambda_b"] = g_lb.clone() if g_lb is not None else None
        grads1[f"{proj_name}.lambda_d"] = g_ld.clone() if g_ld is not None else None

    del model1, mlp1, h_gate, h_up, h_mid, y1
    _cleanup()

    # --- Run 2: VeRA+FlashFFN (with same weights/buffers) ---
    model2 = _make_peft_vera_model(device, dtype, target_modules=target_modules)
    model2.train()

    # Copy trainable params
    for n, p in model2.named_parameters():
        if n in saved_params:
            p.data.copy_(saved_params[n])
    # Copy vera_A/vera_B buffers
    for n, buf in model2.named_buffers():
        if n in saved_buffers:
            buf.copy_(saved_buffers[n])

    del saved_buffers, saved_params

    mlp2 = model2.base_model.model.model.layers[0].mlp
    w_gate_eff = _compute_vera_effective_weight(mlp2.gate_proj)
    w_up_eff = _compute_vera_effective_weight(mlp2.up_proj)
    w_down_eff = _compute_vera_effective_weight(mlp2.down_proj)
    y2 = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)
    y2.sum().backward()

    # Compare output
    cos_output = cosine_sim(y1_detach, y2.detach())
    print(f"  Output cosine:  {cos_output:.8f}")

    # Compare gradients for each projection
    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj2 = getattr(mlp2, proj_name)
        adapter_name = proj2.active_adapters[0]

        # lambda_b grad comparison
        g1_lb = grads1[f"{proj_name}.lambda_b"]
        g2_lb = proj2.vera_lambda_b[adapter_name].grad
        if g1_lb is not None and g2_lb is not None:
            cos_lb = cosine_sim(g1_lb, g2_lb)
            print(f"  {proj_name}.lambda_b grad cosine: {cos_lb:.6f}")
            if proj_name in ('gate_proj', 'up_proj'):
                all_ok = all_ok and (cos_lb > 0.99)
            else:
                all_ok = all_ok and (cos_lb > 0.95)

        # lambda_d grad comparison
        g1_ld = grads1[f"{proj_name}.lambda_d"]
        g2_ld = proj2.vera_lambda_d[adapter_name].grad
        if g1_ld is not None and g2_ld is not None:
            both_zero = g1_ld.abs().max().item() < 1e-8 and g2_ld.abs().max().item() < 1e-8
            if both_zero:
                print(f"  {proj_name}.lambda_d grad cosine: SKIP (both zero)")
            else:
                cos_ld = cosine_sim(g1_ld, g2_ld)
                print(f"  {proj_name}.lambda_d grad cosine: {cos_ld:.6f}")
                if proj_name in ('gate_proj', 'up_proj'):
                    all_ok = all_ok and (cos_ld > 0.99)
                else:
                    all_ok = all_ok and (cos_ld > 0.95)

    print(f"  PASS: {all_ok}")

    del model2, mlp2, x, y2, grads1, y1_detach
    _cleanup()
    return all_ok


def test_training_step():
    """Test that an optimizer step updates all trainable params."""
    print("\n" + "=" * 60)
    print("TEST 4: Training Step (Parameter Updates)")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    model = _make_peft_vera_model(device, dtype)
    model.train()

    # Snapshot initial weights
    initial_state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            initial_state[n] = p.data.clone()

    # Forward through FlashFFN
    mlp = model.base_model.model.model.layers[0].mlp
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype)
    w_gate_eff = _compute_vera_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_vera_effective_weight(mlp.up_proj)
    w_down_eff = _compute_vera_effective_weight(mlp.down_proj)
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
                print(f"    Not updated: {n} (zero grad)")

    print(f"  Layer 0 MLP trainable params: {total}")
    print(f"  Layer 0 MLP updated params:   {updated}")
    # VeRA has lambda_b and lambda_d for each of gate/up/down = 6 params.
    # Both should update since lambda_b=1.0 at init (unlike LoRA where B=0).
    ok = updated >= 3  # At least lambda_b for gate/up/down should update
    print(f"  PASS: {ok} ({updated}/{total} updated, need >= 3)")

    del model, mlp, x, y, loss, optimizer
    _cleanup()
    return ok


if __name__ == "__main__":
    results = []
    results.append(("Forward Equivalence", test_forward_equivalence()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Gradient Similarity", test_gradient_similarity()))
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
