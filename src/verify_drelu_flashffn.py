"""
Verification suite for dReLU FlashFFN (TurboSparse-Mistral / Bamboo).

Validates FlashDReLUFFNFunction against a reference dReLU FFN at the
TurboSparse-Mistral dimensions (hidden=4096, intermediate=14336), covering:

  1. Forward equivalence (maskless)               -- bf16 + fp32
  2. Forward equivalence (with Bamboo predictor mask)
  3. Gradient similarity (grad_x / grad_w_gate / grad_w_up exact;
     grad_w_down approximate via top-K reconstruction)        -- maskless + masked
  4. Training step (params actually update, no NaN/inf)
  5. Anomaly checks (NaN/inf in forward + every gradient)
  6. Sparsity stats (dReLU + hard-mask sparsity; whether top-K captures
     all non-zeros, which makes grad_w_down ~exact)

Run:
    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/verify_drelu_flashffn.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from flashffn import (
    FlashDReLUFFNFunction,
    StandardDReLUFFN,
    compute_bamboo_mask,
)

# TurboSparse-Mistral-Instruct ("Bamboo") dimensions
HIDDEN = 4096
INTER = 14336
PRED_HIDDEN = 1536          # representative predictor hidden (weight_list[0])
K_FRACTION = 0.3


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def has_bad(t):
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _make_weights(device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    w_gate = (torch.randn(INTER, HIDDEN, device=device, dtype=dtype, generator=g) * 0.02)
    w_up = (torch.randn(INTER, HIDDEN, device=device, dtype=dtype, generator=g) * 0.02)
    w_down = (torch.randn(HIDDEN, INTER, device=device, dtype=dtype, generator=g) * 0.02)
    return w_gate, w_up, w_down


def _make_predictor(device, dtype, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    # bias=False, matches modeling_bamboo.MLP (fc1 -> relu -> fc2 -> sigmoid)
    fc1 = torch.randn(PRED_HIDDEN, HIDDEN, device=device, dtype=dtype, generator=g) * 0.02
    fc2 = torch.randn(INTER, PRED_HIDDEN, device=device, dtype=dtype, generator=g) * 0.02
    return fc1, fc2


# ---------------------------------------------------------------------------
# Test 1: forward equivalence, maskless
# ---------------------------------------------------------------------------
def test_forward_maskless():
    print("\n[1] Forward equivalence (maskless)")
    device = "cuda"
    ok = True
    for dtype, tol in [(torch.float32, 1e-3), (torch.bfloat16, 5e-2)]:
        batch, seq = 2, 512
        w_gate, w_up, w_down = _make_weights(device, dtype, seed=7)
        ref = StandardDReLUFFN(HIDDEN, INTER).to(device=device, dtype=dtype)
        ref.gate_proj.weight.data.copy_(w_gate)
        ref.up_proj.weight.data.copy_(w_up)
        ref.down_proj.weight.data.copy_(w_down)

        x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
        y_flash = FlashDReLUFFNFunction.apply(x, w_gate, w_up, w_down, K_FRACTION)
        y_ref = ref(x)

        maxd = (y_flash - y_ref).abs().max().item()
        meand = (y_flash - y_ref).abs().mean().item()
        bad = has_bad(y_flash)
        passed = (maxd < tol) and not bad
        ok = ok and passed
        print(f"    {str(dtype):16s} max={maxd:.2e} mean={meand:.2e} "
              f"NaN/Inf={bad}  -> {'PASS' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test 2: forward equivalence WITH predictor mask
# ---------------------------------------------------------------------------
def test_forward_masked():
    print("\n[2] Forward equivalence (with Bamboo predictor mask)")
    device = "cuda"
    ok = True
    for dtype, tol in [(torch.float32, 1e-3), (torch.bfloat16, 5e-2)]:
        batch, seq = 2, 512
        w_gate, w_up, w_down = _make_weights(device, dtype, seed=7)
        fc1, fc2 = _make_predictor(device, dtype, seed=3)
        ref = StandardDReLUFFN(HIDDEN, INTER).to(device=device, dtype=dtype)
        ref.gate_proj.weight.data.copy_(w_gate)
        ref.up_proj.weight.data.copy_(w_up)
        ref.down_proj.weight.data.copy_(w_down)

        x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
        before_norm = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

        # Reference uses the identical recomputed hard mask
        mask = compute_bamboo_mask(
            before_norm.reshape(-1, HIDDEN), fc1, fc2
        ).to(dtype).reshape(batch, seq, INTER)
        sparsity = 1.0 - mask.float().mean().item()

        y_flash = FlashDReLUFFNFunction.apply(
            x, w_gate, w_up, w_down, K_FRACTION, before_norm, fc1, fc2
        )
        y_ref = ref(x, mask=mask)

        maxd = (y_flash - y_ref).abs().max().item()
        meand = (y_flash - y_ref).abs().mean().item()
        bad = has_bad(y_flash)
        passed = (maxd < tol) and not bad
        ok = ok and passed
        print(f"    {str(dtype):16s} mask_sparsity={sparsity:.1%} "
              f"max={maxd:.2e} mean={meand:.2e} NaN/Inf={bad} -> {'PASS' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test 3: gradient similarity (fp32), maskless + masked
# ---------------------------------------------------------------------------
def _grad_compare(masked: bool):
    device = "cuda"
    dtype = torch.float32
    batch, seq = 2, 256
    w_gate, w_up, w_down = _make_weights(device, dtype, seed=11)

    # two independent param sets (flash vs ref) sharing the same init
    def mk_params():
        wg = w_gate.clone().requires_grad_(True)
        wu = w_up.clone().requires_grad_(True)
        wd = w_down.clone().requires_grad_(True)
        return wg, wu, wd

    wg_f, wu_f, wd_f = mk_params()
    ref = StandardDReLUFFN(HIDDEN, INTER).to(device=device, dtype=dtype)
    ref.gate_proj.weight.data.copy_(w_gate)
    ref.up_proj.weight.data.copy_(w_up)
    ref.down_proj.weight.data.copy_(w_down)

    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
    x_f = x.clone().requires_grad_(True)
    x_r = x.clone().requires_grad_(True)

    if masked:
        fc1, fc2 = _make_predictor(device, dtype, seed=5)
        before_norm = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
        mask = compute_bamboo_mask(before_norm.reshape(-1, HIDDEN), fc1, fc2).reshape(batch, seq, INTER)
        y_f = FlashDReLUFFNFunction.apply(x_f, wg_f, wu_f, wd_f, K_FRACTION, before_norm, fc1, fc2)
        y_r = ref(x_r, mask=mask)
    else:
        y_f = FlashDReLUFFNFunction.apply(x_f, wg_f, wu_f, wd_f, K_FRACTION)
        y_r = ref(x_r)

    g = torch.randn_like(y_f)
    y_f.backward(g)
    y_r.backward(g)

    sims = {
        "grad_x": cos(x_f.grad, x_r.grad),
        "grad_w_gate": cos(wg_f.grad, ref.gate_proj.weight.grad),
        "grad_w_up": cos(wu_f.grad, ref.up_proj.weight.grad),
        "grad_w_down": cos(wd_f.grad, ref.down_proj.weight.grad),
    }
    bad = any(has_bad(t) for t in [x_f.grad, wg_f.grad, wu_f.grad, wd_f.grad])
    return sims, bad


def test_gradients():
    print("\n[3] Gradient similarity (fp32)")
    ok = True
    for masked in [False, True]:
        sims, bad = _grad_compare(masked)
        # exact paths must be ~1.0; grad_w_down is top-K approximate
        passed = (sims["grad_x"] > 0.99 and sims["grad_w_gate"] > 0.99
                  and sims["grad_w_up"] > 0.99 and sims["grad_w_down"] > 0.95
                  and not bad)
        ok = ok and passed
        tag = "masked" if masked else "maskless"
        print(f"    {tag:9s} grad_x={sims['grad_x']:.5f} w_gate={sims['grad_w_gate']:.5f} "
              f"w_up={sims['grad_w_up']:.5f} w_down={sims['grad_w_down']:.5f} "
              f"NaN/Inf={bad} -> {'PASS' if passed else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: training step (params update, stable)
# ---------------------------------------------------------------------------
def test_training_step():
    print("\n[4] Training step (params update, stable)")
    device = "cuda"
    dtype = torch.float32
    batch, seq = 2, 128
    w_gate, w_up, w_down = _make_weights(device, dtype, seed=21)
    wg = w_gate.clone().requires_grad_(True)
    wu = w_up.clone().requires_grad_(True)
    wd = w_down.clone().requires_grad_(True)
    fc1, fc2 = _make_predictor(device, dtype, seed=9)   # frozen predictor

    opt = torch.optim.SGD([wg, wu, wd], lr=1e-2)
    before = [p.detach().clone() for p in (wg, wu, wd)]

    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
    before_norm = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
    target = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

    losses = []
    for _ in range(3):
        opt.zero_grad()
        y = FlashDReLUFFNFunction.apply(x, wg, wu, wd, K_FRACTION, before_norm, fc1, fc2)
        loss = F.mse_loss(y, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    updated = sum(int(not torch.equal(b, p.detach())) for b, p in zip(before, (wg, wu, wd)))
    stable = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses)
    passed = updated == 3 and stable
    print(f"    params_updated={updated}/3 losses={['%.4f' % l for l in losses]} "
          f"stable={stable} -> {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Test 5: sparsity stats + top-K coverage (why grad_w_down is ~exact when masked)
# ---------------------------------------------------------------------------
def test_sparsity_stats():
    print("\n[5] Sparsity stats + top-K coverage")
    device = "cuda"
    dtype = torch.float32
    batch, seq = 2, 256
    w_gate, w_up, w_down = _make_weights(device, dtype, seed=33)
    fc1, fc2 = _make_predictor(device, dtype, seed=13)
    x = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)
    before_norm = torch.randn(batch, seq, HIDDEN, device=device, dtype=dtype)

    N = batch * seq
    h_gate = F.linear(x.reshape(N, HIDDEN), w_gate)
    h_up = F.linear(x.reshape(N, HIDDEN), w_up)
    h_mid_drelu = F.relu(h_gate) * F.relu(h_up)
    mask = compute_bamboo_mask(before_norm.reshape(N, HIDDEN), fc1, fc2)
    h_mid_masked = h_mid_drelu * mask

    drelu_sparsity = (h_mid_drelu == 0).float().mean().item()
    masked_sparsity = (h_mid_masked == 0).float().mean().item()
    k = max(1, int(INTER * K_FRACTION))
    nnz_per_row = (h_mid_masked != 0).sum(-1)
    captured = (nnz_per_row <= k).float().mean().item()  # fraction of rows fully captured by top-K
    max_nnz = nnz_per_row.max().item()

    print(f"    dReLU natural sparsity (zeros)      : {drelu_sparsity:.1%}")
    print(f"    after hard mask (zeros)             : {masked_sparsity:.1%}")
    print(f"    top-K k={k} ({K_FRACTION:.0%} of {INTER}); max non-zeros/row={max_nnz}")
    print(f"    rows fully captured by top-K        : {captured:.1%}  "
          f"(=> grad_w_down exact for these rows)")
    # Sanity: this is informational; pass if no NaN and numbers are well-formed
    passed = not (has_bad(h_mid_masked)) and 0.0 <= masked_sparsity <= 1.0
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)
    print("=" * 68)
    print("dReLU FlashFFN Verification  (TurboSparse-Mistral / Bamboo dims)")
    print(f"hidden={HIDDEN} intermediate={INTER} k_fraction={K_FRACTION}")
    print("=" * 68)
    results = {
        "forward_maskless": test_forward_maskless(),
        "forward_masked": test_forward_masked(),
        "gradients": test_gradients(),
        "training_step": test_training_step(),
        "sparsity_stats": test_sparsity_stats(),
    }
    print("\n" + "=" * 68)
    print("SUMMARY")
    for k, v in results.items():
        print(f"  {k:20s} {'PASS' if v else 'FAIL'}")
    allok = all(results.values())
    print("=" * 68)
    print("ALL PASS" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
