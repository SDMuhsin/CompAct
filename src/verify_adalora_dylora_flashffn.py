"""
Verify AdaLoRA + DyLoRA FlashFFN effective-weight integration.

Tests:
1. Forward output equivalence (AdaLoRA standard vs AdaLoRA+FlashFFN)
2. Forward output equivalence (DyLoRA standard vs DyLoRA+FlashFFN)
3. Gradient flow for AdaLoRA (lora_A, lora_B, lora_E get gradients)
4. Gradient flow for DyLoRA (lora_A, lora_B get gradients)
5. Gradient cosine similarity (AdaLoRA standard vs FlashFFN)
6. Gradient cosine similarity (DyLoRA standard vs FlashFFN)
7. Training step with AdaLoRA+FlashFFN (params actually update)
8. Training step with DyLoRA+FlashFFN (params actually update)
"""
import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, TaskType
from peft import AdaLoraConfig
from flashffn import FlashFFNFunction
from dylora import DyLoRALinear, get_dylora_model


# ── Helpers copied from train_glue.py (standalone test, no train_glue import
#    for the unit-level tests 1-6; tests 7-8 do import apply_flash_ffn) ──────

def _is_adalora_wrapped(linear_module):
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'lora_E')
            and len(getattr(linear_module, 'lora_E', {})) > 0)


def _compute_adalora_effective_weight(svd_linear):
    adapter_name = svd_linear.active_adapters[0]
    base_weight = svd_linear.get_base_layer().weight
    lora_A = svd_linear.lora_A[adapter_name]
    lora_E = svd_linear.lora_E[adapter_name]
    lora_B = svd_linear.lora_B[adapter_name]
    scaling = svd_linear.scaling[adapter_name]
    ranknum = svd_linear.ranknum[adapter_name]
    delta_w = lora_B @ (lora_A * lora_E)
    w_eff = base_weight + delta_w * scaling / (ranknum + 1e-5)
    w_eff = w_eff.to(base_weight.dtype)
    return w_eff


def _is_dylora_wrapped(linear_module):
    return isinstance(linear_module, DyLoRALinear)


def _compute_dylora_effective_weight(dylora_linear):
    base_weight = dylora_linear.base_layer.weight
    lora_A = dylora_linear.lora_A
    lora_B = dylora_linear.lora_B
    scaling = dylora_linear.scaling
    if dylora_linear.training:
        b = random.randint(1, dylora_linear.r)
    else:
        b = dylora_linear.r
    delta_w = lora_B[:, :b] @ lora_A[:b, :]
    w_eff = base_weight + scaling * delta_w
    w_eff = w_eff.to(base_weight.dtype)
    return w_eff


def cosine_sim(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# ── Test 1: AdaLoRA Forward Equivalence ─────────────────────────────────────

def test_adalora_forward_equivalence():
    """Test that AdaLoRA+FlashFFN produces the same output as standard AdaLoRA."""
    print("=" * 60)
    print("TEST 1: AdaLoRA Forward Output Equivalence")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

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
    peft_config = AdaLoraConfig(
        init_r=12, target_r=4, lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
        total_step=1000,
    )
    model = get_peft_model(model, peft_config)
    model.eval()

    mlp = model.base_model.model.model.layers[0].mlp
    assert _is_adalora_wrapped(mlp.gate_proj), "gate_proj should be AdaLoRA-wrapped"

    x = torch.randn(2, 64, 2048, device=device, dtype=dtype)

    # Standard AdaLoRA forward (through PEFT modules)
    with torch.no_grad():
        h_gate_std = mlp.gate_proj(x)
        h_up_std = mlp.up_proj(x)
        h_mid_std = F.silu(h_gate_std) * h_up_std
        y_std = mlp.down_proj(h_mid_std)

    # FlashFFN AdaLoRA forward (effective weight approach)
    with torch.no_grad():
        w_gate_eff = _compute_adalora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_adalora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_adalora_effective_weight(mlp.down_proj)
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


# ── Test 2: DyLoRA Forward Equivalence ──────────────────────────────────────

def test_dylora_forward_equivalence():
    """Test that DyLoRA+FlashFFN produces the same output as standard DyLoRA."""
    print("\n" + "=" * 60)
    print("TEST 2: DyLoRA Forward Output Equivalence")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

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
    model = get_dylora_model(model, target_modules=target_modules, r=8, alpha=16, dropout=0.0)
    model.eval()  # Full rank for deterministic comparison

    mlp = model.model.model.layers[0].mlp
    assert _is_dylora_wrapped(mlp.gate_proj), "gate_proj should be DyLoRA-wrapped"

    x = torch.randn(2, 64, 2048, device=device, dtype=dtype)

    # Standard DyLoRA forward (through DyLoRALinear modules)
    with torch.no_grad():
        h_gate_std = mlp.gate_proj(x)
        h_up_std = mlp.up_proj(x)
        h_mid_std = F.silu(h_gate_std) * h_up_std
        y_std = mlp.down_proj(h_mid_std)

    # FlashFFN DyLoRA forward (effective weight approach, eval mode → full rank)
    with torch.no_grad():
        w_gate_eff = _compute_dylora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_dylora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_dylora_effective_weight(mlp.down_proj)
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


# ── Test 3: AdaLoRA Gradient Flow ───────────────────────────────────────────

def test_adalora_gradient_flow():
    """Test that gradients flow to all AdaLoRA trainable params through FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 3: AdaLoRA Gradient Flow")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

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
    peft_config = AdaLoraConfig(
        init_r=12, target_r=4, lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
        total_step=1000,
    )
    model = get_peft_model(model, peft_config)
    model.train()

    mlp = model.base_model.model.model.layers[0].mlp

    x = torch.randn(2, 32, 2048, device=device, dtype=dtype, requires_grad=True)
    w_gate_eff = _compute_adalora_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_adalora_effective_weight(mlp.up_proj)
    w_down_eff = _compute_adalora_effective_weight(mlp.down_proj)
    y = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    loss = y.sum()
    loss.backward()

    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, proj_name)
        adapter_name = proj.active_adapters[0]

        # lora_A — at init lora_E=0, so grad_A = 0 (expected)
        lora_a = proj.lora_A[adapter_name]
        has_grad_a = lora_a.grad is not None
        grad_a_max = lora_a.grad.abs().max().item() if has_grad_a else -1
        print(f"  {proj_name}.lora_A grad: {'EXISTS' if has_grad_a else 'MISSING'}"
              f" (max={grad_a_max:.4e}, zero expected at init since lora_E=0)")
        all_ok = all_ok and has_grad_a

        # lora_B — at init lora_E=0, so grad_B = 0 (expected)
        lora_b = proj.lora_B[adapter_name]
        has_grad_b = lora_b.grad is not None
        grad_b_max = lora_b.grad.abs().max().item() if lora_b.grad is not None else -1
        print(f"  {proj_name}.lora_B grad: {'EXISTS' if has_grad_b else 'MISSING'}"
              f" (max={grad_b_max:.4e}, zero expected at init since lora_E=0)")
        all_ok = all_ok and has_grad_b

        # lora_E (singular values — key AdaLoRA param, MUST have nonzero grad)
        lora_e = proj.lora_E[adapter_name]
        has_grad_e = lora_e.grad is not None and lora_e.grad.abs().max().item() > 0
        grad_e_max = lora_e.grad.abs().max().item() if lora_e.grad is not None else -1
        print(f"  {proj_name}.lora_E grad: {'YES' if has_grad_e else 'NO'} (max={grad_e_max:.4e})")
        all_ok = all_ok and has_grad_e

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


# ── Test 4: DyLoRA Gradient Flow ────────────────────────────────────────────

def test_dylora_gradient_flow():
    """Test that gradients flow to all DyLoRA trainable params through FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 4: DyLoRA Gradient Flow")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

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
    model = get_dylora_model(model, target_modules=target_modules, r=8, alpha=16, dropout=0.0)
    model.train()

    mlp = model.model.model.layers[0].mlp

    # Use full rank for deterministic gradient check
    # (Set eval temporarily to use full rank, then manually set training back
    # — but we want requires_grad=True on effective weight so use train mode
    # and fix the random seed for reproducibility)
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype, requires_grad=True)

    # Fix random state so rank sampling is deterministic
    random.seed(42)
    w_gate_eff = _compute_dylora_effective_weight(mlp.gate_proj)
    w_up_eff = _compute_dylora_effective_weight(mlp.up_proj)
    w_down_eff = _compute_dylora_effective_weight(mlp.down_proj)
    y = FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, 0.3)

    loss = y.sum()
    loss.backward()

    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(mlp, proj_name)

        # lora_A — may be zero for slices not used in this forward
        lora_a = proj.lora_A
        has_grad_a = lora_a.grad is not None
        grad_a_max = lora_a.grad.abs().max().item() if has_grad_a else -1
        print(f"  {proj_name}.lora_A grad: {'EXISTS' if has_grad_a else 'MISSING'} (max={grad_a_max:.4e})")
        all_ok = all_ok and has_grad_a

        # lora_B — init is zeros so at init lora_B.grad may be zero
        # (since delta = B @ A and B=0 → grad_A = B.T @ grad_delta = 0)
        # But lora_A is nonzero (Kaiming init), so grad_B = grad_delta @ A.T should be nonzero
        lora_b = proj.lora_B
        has_grad_b = lora_b.grad is not None
        grad_b_max = lora_b.grad.abs().max().item() if has_grad_b else -1
        # At init, lora_B=0, so the LoRA delta is zero regardless of lora_A.
        # grad_B comes from grad_delta @ A_trunc.T. grad_delta = grad_W_eff for the
        # delta_w component, which should be nonzero. So grad_B should be nonzero.
        print(f"  {proj_name}.lora_B grad: {'EXISTS' if has_grad_b else 'MISSING'} (max={grad_b_max:.4e})")
        all_ok = all_ok and has_grad_b

        # Base weight should NOT have grad (frozen)
        base_w = proj.base_layer.weight
        has_grad_base = base_w.grad is not None and base_w.grad.abs().max().item() > 0
        print(f"  {proj_name}.base_weight grad: {'NO (frozen)' if not has_grad_base else 'YES (BAD!)'}")
        all_ok = all_ok and (not has_grad_base)

    # Input gradient
    has_grad_x = x.grad is not None and x.grad.abs().max().item() > 0
    print(f"  input x grad: {'YES' if has_grad_x else 'NO'}")
    all_ok = all_ok and has_grad_x

    print(f"  PASS: {all_ok}")
    return all_ok


# ── Test 5: AdaLoRA Gradient Cosine Similarity ──────────────────────────────

def test_adalora_gradient_similarity():
    """Compare gradients from standard AdaLoRA vs AdaLoRA+FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 5: AdaLoRA Gradient Cosine Similarity")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    # --- Run 1: Standard AdaLoRA ---
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
    peft_config = AdaLoraConfig(
        init_r=12, target_r=4, lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
        total_step=1000,
    )
    model1 = get_peft_model(model1, peft_config)
    model1.train()

    # --- Run 2: AdaLoRA+FlashFFN (clone weights from model1) ---
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

    # Standard AdaLoRA forward for one MLP layer
    mlp1 = model1.base_model.model.model.layers[0].mlp

    x1 = x.clone().requires_grad_(True)
    h_gate1 = mlp1.gate_proj(x1)
    h_up1 = mlp1.up_proj(x1)
    h_mid1 = F.silu(h_gate1) * h_up1
    y1 = mlp1.down_proj(h_mid1)
    y1.sum().backward()

    # FlashFFN AdaLoRA forward for one MLP layer
    mlp2 = model2.base_model.model.model.layers[0].mlp

    x2 = x.clone().requires_grad_(True)
    w_gate_eff = _compute_adalora_effective_weight(mlp2.gate_proj)
    w_up_eff = _compute_adalora_effective_weight(mlp2.up_proj)
    w_down_eff = _compute_adalora_effective_weight(mlp2.down_proj)
    y2 = FlashFFNFunction.apply(x2, w_gate_eff, w_up_eff, w_down_eff, 0.3)
    y2.sum().backward()

    # Compare output
    output_cos = cosine_sim(y1.detach(), y2.detach())
    print(f"  Output cosine sim:      {output_cos:.8f}")

    # Compare input gradients
    grad_x_cos = cosine_sim(x1.grad, x2.grad)
    print(f"  grad_x cosine sim:      {grad_x_cos:.8f}")

    # Compare AdaLoRA param gradients
    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj1 = getattr(mlp1, proj_name)
        proj2 = getattr(mlp2, proj_name)
        adapter_name = proj1.active_adapters[0]

        # lora_A — both are zero at init (lora_E=0), so cosine is undefined
        ga1 = proj1.lora_A[adapter_name].grad
        ga2 = proj2.lora_A[adapter_name].grad
        if ga1 is not None and ga2 is not None:
            if ga1.abs().max().item() == 0 and ga2.abs().max().item() == 0:
                print(f"  {proj_name}.lora_A grad: both zero (expected, lora_E=0 at init)")
            else:
                sim = cosine_sim(ga1, ga2)
                print(f"  {proj_name}.lora_A grad cos: {sim:.6f}")
                all_ok = all_ok and sim > 0.90
        else:
            print(f"  {proj_name}.lora_A grad: MISSING")
            all_ok = False

        # lora_B — both are zero at init (lora_E=0), so cosine is undefined
        gb1 = proj1.lora_B[adapter_name].grad
        gb2 = proj2.lora_B[adapter_name].grad
        if gb1 is not None and gb2 is not None:
            if gb1.abs().max().item() == 0 and gb2.abs().max().item() == 0:
                print(f"  {proj_name}.lora_B grad: both zero (expected, lora_E=0 at init)")
            else:
                sim = cosine_sim(gb1, gb2)
                threshold = 0.95 if proj_name == 'down_proj' else 0.99
                print(f"  {proj_name}.lora_B grad cos: {sim:.6f}")
                all_ok = all_ok and sim > threshold
        else:
            print(f"  {proj_name}.lora_B grad: MISSING")
            all_ok = False

        # lora_E — down_proj uses sparse h_mid so gradient is approximate
        ge1 = proj1.lora_E[adapter_name].grad
        ge2 = proj2.lora_E[adapter_name].grad
        if ge1 is not None and ge2 is not None:
            sim = cosine_sim(ge1, ge2)
            threshold = 0.95 if proj_name == 'down_proj' else 0.99
            print(f"  {proj_name}.lora_E grad cos: {sim:.6f}")
            all_ok = all_ok and sim > threshold
        else:
            print(f"  {proj_name}.lora_E grad: MISSING")
            all_ok = False

    all_ok = all_ok and output_cos > 0.999 and grad_x_cos > 0.95
    print(f"  PASS: {all_ok}")
    return all_ok


# ── Test 6: DyLoRA Gradient Cosine Similarity ───────────────────────────────

def test_dylora_gradient_similarity():
    """Compare gradients from standard DyLoRA vs DyLoRA+FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 6: DyLoRA Gradient Cosine Similarity")
    print("=" * 60)

    device = "cuda"
    dtype = torch.float32

    # --- Run 1: Standard DyLoRA ---
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
    model1 = get_dylora_model(model1, target_modules=target_modules, r=8, alpha=16, dropout=0.0)
    model1.eval()  # Full rank for deterministic comparison

    # --- Run 2: DyLoRA+FlashFFN ---
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
    model2 = get_dylora_model(model2, target_modules=target_modules, r=8, alpha=16, dropout=0.0)
    model2.eval()  # Full rank

    # Copy all trainable weights from model1 to model2
    state1 = {n: p.data.clone() for n, p in model1.named_parameters() if p.requires_grad}
    for n, p in model2.named_parameters():
        if n in state1:
            p.data.copy_(state1[n])

    # Same input
    torch.manual_seed(42)
    x = torch.randn(2, 32, 2048, device=device, dtype=dtype)

    # Standard DyLoRA forward for one MLP layer
    mlp1 = model1.model.model.layers[0].mlp

    x1 = x.clone().requires_grad_(True)
    h_gate1 = mlp1.gate_proj(x1)
    h_up1 = mlp1.up_proj(x1)
    h_mid1 = F.silu(h_gate1) * h_up1
    y1 = mlp1.down_proj(h_mid1)
    y1.sum().backward()

    # FlashFFN DyLoRA forward for one MLP layer
    mlp2 = model2.model.model.layers[0].mlp

    x2 = x.clone().requires_grad_(True)
    w_gate_eff = _compute_dylora_effective_weight(mlp2.gate_proj)
    w_up_eff = _compute_dylora_effective_weight(mlp2.up_proj)
    w_down_eff = _compute_dylora_effective_weight(mlp2.down_proj)
    y2 = FlashFFNFunction.apply(x2, w_gate_eff, w_up_eff, w_down_eff, 0.3)
    y2.sum().backward()

    # Compare output
    output_cos = cosine_sim(y1.detach(), y2.detach())
    print(f"  Output cosine sim:      {output_cos:.8f}")

    # Compare input gradients
    grad_x_cos = cosine_sim(x1.grad, x2.grad)
    print(f"  grad_x cosine sim:      {grad_x_cos:.8f}")

    # Compare DyLoRA param gradients
    all_ok = True
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj1 = getattr(mlp1, proj_name)
        proj2 = getattr(mlp2, proj_name)

        # lora_A
        ga1 = proj1.lora_A.grad
        ga2 = proj2.lora_A.grad
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

        # lora_B — down_proj uses sparse h_mid so gradient is approximate
        gb1 = proj1.lora_B.grad
        gb2 = proj2.lora_B.grad
        if gb1 is not None and gb2 is not None:
            sim = cosine_sim(gb1, gb2)
            threshold = 0.95 if proj_name == 'down_proj' else 0.99
            print(f"  {proj_name}.lora_B grad cos: {sim:.6f}")
            all_ok = all_ok and sim > threshold
        else:
            print(f"  {proj_name}.lora_B grad: MISSING")
            all_ok = False

    all_ok = all_ok and output_cos > 0.999 and grad_x_cos > 0.95
    print(f"  PASS: {all_ok}")
    return all_ok


# ── Test 7: AdaLoRA Training Step ───────────────────────────────────────────

def test_adalora_training_step():
    """Test that a full training step updates AdaLoRA params via FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 7: AdaLoRA Training Step Updates Parameters")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,
    )
    config.pad_token_id = config.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = AdaLoraConfig(
        init_r=12, target_r=4, lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
        total_step=1000,
    )
    model = get_peft_model(model, peft_config)
    model.train()

    # Apply FlashFFN to MLP layers
    from train_glue import apply_flash_ffn
    n = apply_flash_ffn(model, k_fraction=0.3)
    print(f"  Converted {n} MLP layers to FlashFFN (AdaLoRA mode)")

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


# ── Test 8: DyLoRA Training Step ────────────────────────────────────────────

def test_dylora_training_step():
    """Test that a full training step updates DyLoRA params via FlashFFN."""
    print("\n" + "=" * 60)
    print("TEST 8: DyLoRA Training Step Updates Parameters")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=2,
        num_hidden_layers=2,
    )
    config.pad_token_id = config.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        config=config,
        torch_dtype=dtype,
    ).to(device)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_dylora_model(model, target_modules=target_modules, r=8, alpha=16, dropout=0.0)
    model.train()

    # Apply FlashFFN to MLP layers
    from train_glue import apply_flash_ffn
    n = apply_flash_ffn(model, k_fraction=0.3)
    print(f"  Converted {n} MLP layers to FlashFFN (DyLoRA mode)")

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
    # At init, lora_B=0 so lora_A gets zero gradients and won't change.
    # Only lora_B params + classifier head should update.
    changed = 0
    total = 0
    lora_a_unchanged = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n in trainable_params:
            total += 1
            if not torch.equal(p.data, trainable_params[n]):
                changed += 1
            elif 'lora_A' in n:
                lora_a_unchanged += 1

    print(f"  Parameters changed: {changed}/{total}")
    print(f"  lora_A params unchanged (expected at init, lora_B=0): {lora_a_unchanged}")
    # At least lora_B params and classifier should change
    ok = changed > 0 and (changed + lora_a_unchanged) == total
    print(f"  PASS: {ok}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("AdaLoRA + DyLoRA + FlashFFN Verification")
    print("=" * 60)

    results = {}

    results['adalora_forward'] = test_adalora_forward_equivalence()
    torch.cuda.empty_cache()

    results['dylora_forward'] = test_dylora_forward_equivalence()
    torch.cuda.empty_cache()

    results['adalora_gradient_flow'] = test_adalora_gradient_flow()
    torch.cuda.empty_cache()

    results['dylora_gradient_flow'] = test_dylora_gradient_flow()
    torch.cuda.empty_cache()

    results['adalora_gradient_similarity'] = test_adalora_gradient_similarity()
    torch.cuda.empty_cache()

    results['dylora_gradient_similarity'] = test_dylora_gradient_similarity()
    torch.cuda.empty_cache()

    results['adalora_training_step'] = test_adalora_training_step()
    torch.cuda.empty_cache()

    results['dylora_training_step'] = test_dylora_training_step()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
