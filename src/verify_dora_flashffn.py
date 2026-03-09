"""
Verify DoRA + FlashFFN effective-weight integration.

Tests:
1. Forward output equivalence (DoRA standard vs DoRA+FlashFFN)
2. Gradient flow to all trainable params (LoRA A/B + magnitude)
3. Gradient cosine similarity
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


def _is_dora_wrapped(linear_module):
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'lora_magnitude_vector')
            and len(linear_module.lora_magnitude_vector) > 0)


def _compute_dora_effective_weight(dora_linear):
    adapter_name = dora_linear.active_adapters[0]
    base_weight = dora_linear.get_base_layer().weight
    lora_A_weight = dora_linear.lora_A[adapter_name].weight
    lora_B_weight = dora_linear.lora_B[adapter_name].weight
    scaling = dora_linear.scaling[adapter_name]
    lora_delta = lora_B_weight @ lora_A_weight
    merged_weight = base_weight + scaling * lora_delta
    weight_norm = merged_weight.detach().norm(p=2, dim=1)
    magnitude = dora_linear.lora_magnitude_vector[adapter_name].weight
    w_eff = (magnitude / weight_norm).unsqueeze(1) * merged_weight
    w_eff = w_eff.to(base_weight.dtype)
    return w_eff


def cosine_sim(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def test_forward_equivalence():
    """Test that DoRA+FlashFFN produces the same output as standard DoRA."""
    print("=" * 60)
    print("TEST 1: Forward Output Equivalence")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    # Load a small model with DoRA
    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,  # Only 2 layers for speed
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    # Apply DoRA via PEFT
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = PeftLoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.0,  # No dropout for exact comparison
        bias="none", task_type=TaskType.SEQ_CLS, use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    model.eval()  # Eval mode for deterministic comparison

    # Get one MLP layer
    mlp = model.base_model.model.model.layers[0].mlp
    assert _is_dora_wrapped(mlp.gate_proj), "gate_proj should be DoRA-wrapped"

    # Test input
    x = torch.randn(2, 64, 2048, device=device, dtype=dtype)

    # Standard DoRA forward (through PEFT modules)
    with torch.no_grad():
        h_gate_std = mlp.gate_proj(x)
        h_up_std = mlp.up_proj(x)
        h_mid_std = F.silu(h_gate_std) * h_up_std
        y_std = mlp.down_proj(h_mid_std)

    # FlashFFN DoRA forward (effective weight approach)
    with torch.no_grad():
        w_gate_eff = _compute_dora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_dora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_dora_effective_weight(mlp.down_proj)
        y_flash = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    max_diff = (y_std - y_flash).abs().max().item()
    mean_diff = (y_std - y_flash).abs().mean().item()
    cos = cosine_sim(y_std, y_flash)

    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Cosine:    {cos:.8f}")

    # BF16 tolerance — FlashFFN uses Triton kernel (fp32 internally) while
    # standard uses PyTorch's F.silu + mul. Small numerical diffs expected.
    # Cosine similarity is the reliable metric here.
    ok = cos > 0.9999
    print(f"  PASS: {ok}")
    return ok


def test_gradient_flow():
    """Test that gradients flow to all DoRA trainable params through FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Flow")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32  # FP32 for gradient precision

    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = PeftLoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS, use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    model.train()

    mlp = model.base_model.model.model.layers[0].mlp

    # Forward through FlashFFN with effective weights
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype, requires_grad=True)
    w_gate_eff = _compute_dora_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_dora_effective_weight(mlp.up_proj)
    w_down_eff = _compute_dora_effective_weight(mlp.down_proj)
    y = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    loss = y.sum()
    loss.backward()

    # Check gradient flow to all trainable params
    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, proj_name)
        adapter_name = proj.active_adapters[0]

        # LoRA A — NOTE: lora_B is initialized to zeros in PEFT, so at init
        # grad_lora_A = lora_B.T @ grad_delta = 0. This is expected.
        # After first optimizer step (lora_B nonzero), lora_A gets real grads.
        lora_a = proj.lora_A[adapter_name].weight
        has_grad_a = lora_a.grad is not None
        grad_a_max = lora_a.grad.abs().max().item() if has_grad_a else -1
        print(f"  {proj_name}.lora_A grad: {'EXISTS' if has_grad_a else 'MISSING'}"
              f" (max={grad_a_max:.4e}, zero expected at init since lora_B=0)")
        # lora_A grad being zero at init is OK (lora_B is initialized to zeros)
        all_ok = all_ok and has_grad_a

        # LoRA B
        lora_b = proj.lora_B[adapter_name].weight
        has_grad_b = lora_b.grad is not None and lora_b.grad.abs().max().item() > 0
        grad_b_max = lora_b.grad.abs().max().item() if lora_b.grad is not None else -1
        print(f"  {proj_name}.lora_B grad: {'YES' if has_grad_b else 'NO'} (max={grad_b_max:.4e})")
        all_ok = all_ok and has_grad_b

        # Magnitude vector
        mag = proj.lora_magnitude_vector[adapter_name].weight
        has_grad_m = mag.grad is not None and mag.grad.abs().max().item() > 0
        grad_m_max = mag.grad.abs().max().item() if mag.grad is not None else -1
        print(f"  {proj_name}.magnitude grad: {'YES' if has_grad_m else 'NO'} (max={grad_m_max:.4e})")
        all_ok = all_ok and has_grad_m

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
    """Compare gradients from standard DoRA vs DoRA+FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Cosine Similarity (Standard DoRA vs FlashFFN)")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    # --- Run 1: Standard DoRA ---
    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,
    )
    model1 = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    target_modules = ["gate_proj", "up_proj", "down_proj"]
    peft_config = PeftLoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS, use_dora=True,
    )
    model1 = get_peft_model(model1, peft_config)
    model1.train()

    # --- Run 2: DoRA+FlashFFN (clone weights from model1) ---
    config2 = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,
    )
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config2,
        torch_dtype=dtype,
    ).to(device)
    model2 = get_peft_model(model2, peft_config)
    model2.train()

    # Copy all trainable weights from model1 to model2
    state1 = {n: p.data.clone() for n, p in model1.named_parameters() if p.requires_grad}
    for n, p in model2.named_parameters():
        if n in state1:
            p.data.copy_(state1[n])

    # Same input
    torch.manual_seed(42)
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype)

    # Standard DoRA forward for one MLP layer
    mlp1 = model1.base_model.model.model.layers[0].mlp

    x1 = x.clone().requires_grad_(True)
    h_gate1 = mlp1.gate_proj(x1)
    h_up1 = mlp1.up_proj(x1)
    h_mid1 = F.silu(h_gate1) * h_up1
    y1 = mlp1.down_proj(h_mid1)
    y1.sum().backward()

    # FlashFFN DoRA forward for one MLP layer
    mlp2 = model2.base_model.model.model.layers[0].mlp

    x2 = x.clone().requires_grad_(True)
    w_gate_eff = _compute_dora_effective_weight(mlp2.gate_proj)
    w_up_eff = _compute_dora_effective_weight(mlp2.up_proj)
    w_down_eff = _compute_dora_effective_weight(mlp2.down_proj)
    y2 = FlashFFNFunction.apply(x2, w_gate_eff, w_up_eff, w_down_eff, 0.3)
    y2.sum().backward()

    # Compare output
    output_cos = cosine_sim(y1.detach(), y2.detach())
    print(f"  Output cosine sim:      {output_cos:.8f}")

    # Compare input gradients
    grad_x_cos = cosine_sim(x1.grad, x2.grad)
    print(f"  grad_x cosine sim:      {grad_x_cos:.8f}")

    # Compare DoRA param gradients
    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj1 = getattr(mlp1, proj_name)
        proj2 = getattr(mlp2, proj_name)
        adapter_name = proj1.active_adapters[0]

        # LoRA A — both are zero at init (lora_B=0), so cosine is undefined
        ga1 = proj1.lora_A[adapter_name].weight.grad
        ga2 = proj2.lora_A[adapter_name].weight.grad
        if ga1 is not None and ga2 is not None:
            if ga1.abs().max().item() == 0 and ga2.abs().max().item() == 0:
                print(f"  {proj_name}.lora_A grad: both zero (expected, lora_B=0 at init)")
            else:
                sim = cosine_sim(ga1, ga2)
                print(f"  {proj_name}.lora_A grad cos: {sim:.6f}")
                all_ok = all_ok and sim > 0.90
        else:
            print(f"  {proj_name}.lora_A grad: MISSING")
            all_ok = False

        # LoRA B — down_proj uses sparse h_mid so gradient is approximate
        gb1 = proj1.lora_B[adapter_name].weight.grad
        gb2 = proj2.lora_B[adapter_name].weight.grad
        if gb1 is not None and gb2 is not None:
            sim = cosine_sim(gb1, gb2)
            # down_proj lora_B grad is affected by sparse h_mid (FlashFFN top-K)
            threshold = 0.95 if proj_name == 'down_proj' else 0.99
            print(f"  {proj_name}.lora_B grad cos: {sim:.6f}")
            all_ok = all_ok and sim > threshold
        else:
            print(f"  {proj_name}.lora_B grad: MISSING")
            all_ok = False

        # Magnitude — down_proj uses sparse h_mid so gradient is approximate
        gm1 = proj1.lora_magnitude_vector[adapter_name].weight.grad
        gm2 = proj2.lora_magnitude_vector[adapter_name].weight.grad
        if gm1 is not None and gm2 is not None:
            sim = cosine_sim(gm1, gm2)
            # down_proj magnitude grad is affected by sparse h_mid (FlashFFN top-K)
            threshold = 0.95 if proj_name == 'down_proj' else 0.99
            print(f"  {proj_name}.magnitude grad cos: {sim:.6f}")
            all_ok = all_ok and sim > threshold
        else:
            print(f"  {proj_name}.magnitude grad: MISSING")
            all_ok = False

    all_ok = all_ok and output_cos > 0.999 and grad_x_cos > 0.95
    print(f"  PASS: {all_ok}")
    return all_ok


def test_training_step():
    """Test that a full training step updates DoRA params via FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 4: Training Step Updates Parameters")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,
    )
    config.pad_token_id = config.eos_token_id  # Fix pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = PeftLoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS, use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    model.train()

    # Apply FlashFFN to MLP layers
    from train_glue import apply_flash_ffn
    n = apply_flash_ffn(model, k_fraction=0.3)
    print(f"  Converted {n} MLP layers to FlashFFN (DoRA mode)")

    # Save param snapshot
    trainable_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Fake input
    x = torch.randint(0, 32000, (2, 64), device=device)
    labels = torch.randint(0, 2, (2,), device=device)

    # Forward + backward
    outputs = model(input_ids=x, labels=labels)
    loss = outputs.loss
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()

    # Optimizer step
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    optimizer.step()

    # Check params changed
    changed = 0
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n in trainable_params:
            total += 1
            if not torch.equal(p.data, trainable_params[n]):
                changed += 1

    print(f"  Parameters changed: {changed}/{total}")
    ok = changed > 0 and changed == total
    print(f"  PASS: {ok}")
    return ok


if __name__ == "__main__":
    print("DoRA + FlashFFN Verification")
    print("=" * 60)

    results = {}
    results['forward'] = test_forward_equivalence()

    torch.cuda.empty_cache()
    results['gradient_flow'] = test_gradient_flow()

    torch.cuda.empty_cache()
    results['gradient_similarity'] = test_gradient_similarity()

    torch.cuda.empty_cache()
    results['training_step'] = test_training_step()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
