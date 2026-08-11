"""Can the dual-output fused GLU GEMM be made to MATCH cuBLAS?  (CONTEXT.md section 42.4 step 2)

WHY THIS EXISTS.  `src/probe_fused_glu_fwd.py` established the prize and the blocker:

  * the forward/recompute materialisation asymmetry is worth **+7.7 .. +11.8%** of the FFN forward
    gate/up+SwiGLU block at CONSTANT GEMM (`triton_fused` vs `triton_mat`), bitwise, 60/60 cells;
  * but the Triton dual-output GEMM trails cuBLAS by **+13.1% at M=4096 and +11.9% at M=8192**, so
    the net regresses seq 4096 and section 42.4 step 1 forbids landing it.

Section 42.4 step 2 is therefore "close the GEMM gap or abandon", and section 42.5 offers a
structural reason to expect it cannot be closed:

  > "a fused dual-GEMM structurally buys half the tile area per register of the two separate GEMMs
  >  it replaces, and tile area is what buys arithmetic intensity."

THIS PROBE TESTS THAT CLAIM, because on paper it does not hold.  Per K-step a single-output tile
BMxBN loads BM*BK + BK*BN and does BM*BN*BK MACs; a dual-output tile sharing its x tile loads
BM*BK + 2*BK*BN and does 2*BM*BN*BK MACs.  So

    I_single(BM,BN) = BM*BN / (BM+BN)          I_dual(BM,BN) = 2*BM*BN / (BM+2*BN)

and at EQUAL ACCUMULATOR BUDGET the dual form is never worse:

    32768 fp32 accumulators : single 256x128 -> 85.3   ==   dual 128x128 -> 85.3
    65536 fp32 accumulators : single 512x128 -> 102.4  <    dual 256x128 -> 128.0

Tile *area* per register is indeed halved; arithmetic *intensity* is not, because the dual form
shares the x tile across two weight tiles exactly as a taller single tile shares its weight tile
across two x halves.  If that algebra is right, the measured gap is Triton codegen, not structure,
and the fix is a wider config space rather than CUTLASS.

Two things the prior probe could not distinguish are therefore separated here:

  1. a SINGLE-output Triton GEMM is timed against a SINGLE cuBLAS GEMM at the same shapes.  That is
     the Triton-vs-cuBLAS tax with no dual-output structure in it at all.  If a single-output Triton
     GEMM also trails by ~12%, the dual form is exonerated and CUTLASS is the only route.
  2. the dual config space is widened to the tiles the register algebra says should exist --
     num_warps=16 with 256x128 and 128x256 dual tiles (128 regs/thread of accumulator, the same as
     a 256x128 single-output tile at 8 warps), 256x64, and a BLOCK_K sweep.  The prior probe topped
     out at 128x128 and never tried 16 warps.

AND IT ADDS A THIRD DESIGN POINT the prior probe did not consider: `half_fused`.  Instead of
computing both accumulators at once, SERIALISE them --

    h_gate = cuBLAS(x @ wg^T)                        (materialised, as today)
    h_mid  = triton_single_glu(x, wu, h_gate)        (h_up lives only in registers)

The second kernel is an ordinary SINGLE-output GEMM with a heavier epilogue: it loads the h_gate
tile, rounds its own accumulator to bf16 and back, and stores only h_mid.  It gets the full 256x128
tile budget, has no dual accumulator, and still removes the h_up store, the h_up load and the
standalone SwiGLU pass -- 2 of the 4 DRAM passes full fusion removes, with (possibly) none of the
GEMM tax.  Half the prize at no cost beats the whole prize at a 12% cost.

THE BITWISE CONTRACT IS UNCHANGED AND IS THE POINT (section 42.3.1).  Every fused epilogue here
rounds each fp32 accumulator to bf16 and casts it back to fp32 *before* the activation, reproducing
in registers the rounding the separate path performs when it stores h_gate/h_up as bf16 and
`_silu_mul_fwd_kernel` (`flashffn.py:62-67`) loads them back.  Omit those casts and the fused arm is
MORE accurate and gate B3 fails.

METHOD (section 40.6, and this is where the prior probe was weakest).  Its config selection ran 20
reps with 5 warmup and no control on a 100%-loaded box, and picked a 64x128x64/num_warps=4/stages=2
tile at M=8192 -- 4 warps and one CTA per SM.  That selection is noise.  Here EVERY config is an arm
in ONE rotation window alongside the control, at the full rep count, MIN estimator.

NOT A NOVELTY CLAIM.  Inside the owned fusion family (section 40.1/C12: MoMoE, CODA, SonicMoE own
"inline the rematerialisation so intermediates never reach DRAM"; TiledMLP/ALST and Liger own
token-tiled MLP; CUTLASS example 45 `DualGemm` and xformers' SwiGLU own the fused dual GEMM itself).
A NUMBERS detour, authorised as such by section 42.0.  It may ship; it may never be the claim.

Usage:
    PYTHONPATH=src python src/probe_dual_gemm_close.py --reps 200 \
        --out results/recon/dual_gemm_close.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

SHAPES = [
    ("seq1024_b2", 2048, 2048, 5632),
    ("seq2048_b2", 4096, 2048, 5632),
    ("seq4096_b2", 8192, 2048, 5632),
    ("seq8192_b2", 16384, 2048, 5632),
]

GROUP_M = 8

# ---------------------------------------------------------------------------------------------
# Config spaces.  `smem/stage` is (BM*BK + n_w*BK*BN)*2 bytes with n_w = 1 (single) or 2 (dual);
# the A40's limit is 101376 B/CTA.  `acc regs/thread` is n_w*BM*BN*4 / (32*num_warps) / 4.
# Infeasible configs are not pruned by hand -- they are launched, and an OutOfResources is recorded
# as an explicit `err` in the artifact so the sweep's coverage is auditable.
# ---------------------------------------------------------------------------------------------
DUAL_CONFIGS = [
    # --- what the prior probe swept (kept so the two artifacts are comparable) ---
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=4),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=2),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=5),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=4),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4),
    # --- NEW: the tiles the register algebra says should exist.  16 warps puts a 256x128 dual
    #     tile at 128 accumulator registers/thread -- the same budget an 8-warp 256x128 single
    #     -output tile already spends -- and its arithmetic intensity is 128 vs that tile's 85.3.
    dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, num_warps=16, num_stages=3),   # smem 98304
    dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=16, num_warps=16, num_stages=6),   # smem 98304
    dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=16, num_warps=16, num_stages=4),   # smem 65536
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, num_warps=16, num_stages=2),   # smem 81920
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=16, num_stages=4),
    # --- NEW: 256xBN dual tiles at 8 warps (128 acc regs/thread), M-heavy for tall problems ---
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=32, num_warps=8, num_stages=4),     # smem 98304
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=2),     # smem 98304
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=16, num_warps=8, num_stages=6),     # smem 73728
    # --- NEW: deeper pipelines at small BLOCK_K, which the smem bound now allows ---
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=16, num_warps=8, num_stages=8),    # smem 98304
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=16, num_warps=8, num_stages=6),    # smem 73728
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=5),
]

SINGLE_CONFIGS = [
    dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=4),    # smem 98304
    dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3),    # smem 73728
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, num_warps=8, num_stages=4),
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=2),     # smem 81920
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3),    # smem 98304
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=5),    # smem 81920
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=5),
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4),
]


# =============================================================================================
# Kernels
# =============================================================================================
@triton.jit
def _dual_gemm_glu_kernel(
    x_ptr, wg_ptr, wu_ptr,
    hg_ptr, hu_ptr, hm_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, FUSE: tl.constexpr,
):
    """Byte-for-byte the kernel of `probe_fused_glu_fwd.py`, so the two artifacts compare directly.

    h_gate = x @ wg^T and h_up = x @ wu^T sharing one load of the x tile.
    FUSE=True stores only h_mid; FUSE=False stores h_gate and h_up as the shipped path does.
    The bf16 round-trip in the FUSE branch is THE BITWISE CONTRACT (section 42.3.1).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wg_ptrs = wg_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    wu_ptrs = wu_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k0 * BLOCK_K
        xt = tl.load(x_ptrs, mask=m_mask & (offs_k[None, :] < k_rem), other=0.0)
        wgt = tl.load(wg_ptrs, mask=n_mask & (offs_k[:, None] < k_rem), other=0.0)
        wut = tl.load(wu_ptrs, mask=n_mask & (offs_k[:, None] < k_rem), other=0.0)
        acc_g += tl.dot(xt, wgt)
        acc_u += tl.dot(xt, wut)
        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    out_mask = m_mask & n_mask
    if FUSE:
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        res = (g * tl.sigmoid(g)) * u
        hm_ptrs = hm_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(hm_ptrs, res.to(tl.bfloat16), mask=out_mask)
    else:
        hg_ptrs = hg_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        hu_ptrs = hu_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(hg_ptrs, acc_g.to(tl.bfloat16), mask=out_mask)
        tl.store(hu_ptrs, acc_u.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _single_gemm_kernel(
    x_ptr, w_ptr, hg_ptr, hm_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, GLU: tl.constexpr,
):
    """One output.  GLU=False: h_gate = x @ w^T, stored (the Triton-vs-cuBLAS reference arm).

    GLU=True: this is the `half_fused` second kernel.  The accumulator is h_UP; the h_gate tile is
    read back from DRAM (it was produced by cuBLAS in the first kernel) and only h_mid is stored.
    h_up never reaches DRAM.

    THE BITWISE CONTRACT, same as the dual kernel: the accumulator is rounded to bf16 and cast back
    to fp32 before the activation, reproducing the store/load round-trip the separate path performs.
    The h_gate operand needs no such treatment -- it is already the bf16 tensor the separate path
    would have loaded, so `.to(tl.float32)` on it is exactly `_silu_mul_fwd_kernel`'s own load.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k0 * BLOCK_K
        xt = tl.load(x_ptrs, mask=m_mask & (offs_k[None, :] < k_rem), other=0.0)
        wt = tl.load(w_ptrs, mask=n_mask & (offs_k[:, None] < k_rem), other=0.0)
        acc += tl.dot(xt, wt)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_mask = m_mask & n_mask
    if GLU:
        hg_ptrs = hg_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        g = tl.load(hg_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        u = acc.to(tl.bfloat16).to(tl.float32)
        res = (g * tl.sigmoid(g)) * u
        hm_ptrs = hm_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(hm_ptrs, res.to(tl.bfloat16), mask=out_mask)
    else:
        hg_ptrs = hg_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(hg_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def _launch_dual(cfg, x, wg, wu, hg, hu, hm, fuse):
    M, K = x.shape
    N = wg.shape[0]
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _dual_gemm_glu_kernel[grid](
        x, wg, wu, hg, hu, hm, M, N, K,
        x.stride(0), x.stride(1), wg.stride(0), wg.stride(1), hm.stride(0), hm.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg.get("GROUP_M", GROUP_M), FUSE=fuse,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def _launch_single(cfg, x, w, hg, hm, glu):
    M, K = x.shape
    N = w.shape[0]
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _single_gemm_kernel[grid](
        x, w, hg, hm, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), hg.stride(0), hg.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg.get("GROUP_M", GROUP_M), GLU=glu,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


# =============================================================================================
# Timing (section 40.6): all arms in ONE window, rotate the lead, pre-allocated `out=`, MIN.
# =============================================================================================
def _time_rotation(fns, reps, warmup=12):
    names = list(fns.keys())
    k = len(names)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * k)]
    per = {n: [] for n in names}
    for i in range(reps):
        order = names[i % k:] + names[: i % k]
        for j, n in enumerate(order):
            ev[2 * j].record()
            fns[n]()
            ev[2 * j + 1].record()
        ev[2 * k - 1].synchronize()
        for j, n in enumerate(order):
            per[n].append(ev[2 * j].elapsed_time(ev[2 * j + 1]))
    return per


def _drift(control_times, reps):
    third = max(1, reps // 3)
    parts = [min(control_times[:third]), min(control_times[third:2 * third]),
             min(control_times[2 * third:])]
    return parts, 100.0 * (max(parts) - min(parts)) / min(parts)


def _intensity(bm, bn, dual):
    return (2.0 * bm * bn / (bm + 2.0 * bn)) if dual else (1.0 * bm * bn / (bm + bn))


def _cfg_tag(cfg, dual):
    return (f"{'D' if dual else 'S'}{cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']}"
            f"_w{cfg['num_warps']}s{cfg['num_stages']}")


def run_shape(label, M, K, N, reps, sel_reps, dev, silu_mul):
    dt = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    wg = (torch.randn(N, K, device=dev, dtype=torch.float32, generator=g) * 0.02).to(dt).contiguous()
    wu = (torch.randn(N, K, device=dev, dtype=torch.float32, generator=g) * 0.02).to(dt).contiguous()

    hg_c = torch.empty(M, N, device=dev, dtype=dt)
    hu_c = torch.empty(M, N, device=dev, dtype=dt)
    hm_c = torch.empty(M, N, device=dev, dtype=dt)
    hg_t = torch.empty(M, N, device=dev, dtype=dt)
    hu_t = torch.empty(M, N, device=dev, dtype=dt)
    hm_t = torch.empty(M, N, device=dev, dtype=dt)
    hm_f = torch.empty(M, N, device=dev, dtype=dt)
    hg_s = torch.empty(M, N, device=dev, dtype=dt)
    hm_h = torch.empty(M, N, device=dev, dtype=dt)

    cx = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cw = torch.randn(N, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cout = torch.empty(M, N, device=dev, dtype=dt)
    wgT, wuT, cwT = wg.t(), wu.t(), cw.t()

    def arm_control():
        torch.mm(cx, cwT, out=cout)

    # -------- PHASE A: config selection, every config an arm in ONE window with the control -----
    # The prior probe selected at 20 reps / 5 warmup with no control and picked a 4-warp,
    # 1-CTA-per-SM tile at M=8192.  That is the number this phase exists to not reproduce.
    sel_fns = {"control": arm_control}
    feasible, errors = [], []
    for cfg in DUAL_CONFIGS:
        tag = _cfg_tag(cfg, True)
        try:
            _launch_dual(cfg, x, wg, wu, hg_t, hu_t, hm_f, True)
            torch.cuda.synchronize()
        except Exception as e:                                            # noqa: BLE001
            errors.append(dict(tag=tag, kind="dual", err=f"{type(e).__name__}: {e}"[:200]))
            continue
        feasible.append((tag, "dual", cfg))
        sel_fns[tag] = (lambda c=cfg: _launch_dual(c, x, wg, wu, hg_t, hu_t, hm_f, True))
    for cfg in SINGLE_CONFIGS:
        tag = _cfg_tag(cfg, False)
        try:
            _launch_single(cfg, x, wg, hg_s, hm_h, False)
            torch.cuda.synchronize()
        except Exception as e:                                            # noqa: BLE001
            errors.append(dict(tag=tag, kind="single", err=f"{type(e).__name__}: {e}"[:200]))
            continue
        feasible.append((tag, "single", cfg))
        sel_fns[tag] = (lambda c=cfg: _launch_single(c, x, wg, hg_s, hm_h, False))
        tag2 = tag + "_glu"
        feasible.append((tag2, "single_glu", cfg))
        sel_fns[tag2] = (lambda c=cfg: _launch_single(c, x, wu, hg_c, hm_h, True))

    sel_per = _time_rotation(sel_fns, reps=sel_reps)
    sel_parts, sel_drift = _drift(sel_per["control"], sel_reps)
    sel_table = []
    for tag, kind, cfg in feasible:
        sel_table.append(dict(
            tag=tag, kind=kind, cfg=cfg, min_ms=min(sel_per[tag]),
            median_ms=statistics.median(sel_per[tag]),
            acc_regs_per_thread=(2 if kind == "dual" else 1) * cfg["BLOCK_M"] * cfg["BLOCK_N"]
            / (32.0 * cfg["num_warps"]),
            smem_bytes=((cfg["BLOCK_M"] * cfg["BLOCK_K"]
                         + (2 if kind == "dual" else 1) * cfg["BLOCK_K"] * cfg["BLOCK_N"]) * 2
                        * cfg["num_stages"]),
            arith_intensity=_intensity(cfg["BLOCK_M"], cfg["BLOCK_N"], kind == "dual"),
        ))
    sel_table.sort(key=lambda r: r["min_ms"])

    def _best(kind):
        c = [r for r in sel_table if r["kind"] == kind]
        return c[0] if c else None

    best_dual = _best("dual")
    best_single = _best("single")
    best_glu = _best("single_glu")
    if best_dual is None or best_single is None or best_glu is None:
        raise RuntimeError(f"no feasible config for some kind at {label}")
    cd, cs, cgl = best_dual["cfg"], best_single["cfg"], best_glu["cfg"]

    # ------------------------------------- PHASE B: the window ------------------------------
    def arm_cublas_1gemm():
        torch.mm(x, wgT, out=hg_c)

    def arm_cublas_sep():
        torch.mm(x, wgT, out=hg_c)
        torch.mm(x, wuT, out=hu_c)
        silu_mul(hg_c, hu_c, hm_c)

    def arm_triton_single():
        _launch_single(cs, x, wg, hg_s, hm_h, False)

    def arm_dual_mat():
        _launch_dual(cd, x, wg, wu, hg_t, hu_t, hm_t, False)
        silu_mul(hg_t, hu_t, hm_t)

    def arm_dual_fused():
        _launch_dual(cd, x, wg, wu, hg_t, hu_t, hm_f, True)

    def arm_half_fused():
        torch.mm(x, wgT, out=hg_c)
        _launch_single(cgl, x, wu, hg_c, hm_h, True)

    # -------- bitwise checks BEFORE timing (they mutate the buffers) --------
    arm_cublas_sep(); torch.cuda.synchronize()
    hg_cub, hu_cub, hm_cub = hg_c.clone(), hu_c.clone(), hm_c.clone()
    arm_dual_mat(); torch.cuda.synchronize()
    hg_dm, hm_dm = hg_t.clone(), hm_t.clone()
    arm_dual_fused(); torch.cuda.synchronize()
    hm_df = hm_f.clone()
    arm_triton_single(); torch.cuda.synchronize()
    hg_ts = hg_s.clone()
    arm_half_fused(); torch.cuda.synchronize()
    hm_hf = hm_h.clone()

    def maxabs(a, b):
        return float((a.float() - b.float()).abs().max())

    checks = {
        # the dual path's gating check, re-run at the NEW winning config
        "dual_fused_eq_dual_mat": bool(torch.equal(hm_df, hm_dm)),
        "dual_fused_vs_mat_maxabs": maxabs(hm_df, hm_dm),
        "dual_hg_eq_cublas_hg": bool(torch.equal(hg_dm, hg_cub)),
        # the half-fusion path's gating check: it must reproduce the SHIPPED h_mid exactly, which
        # additionally requires Triton's h_up accumulator to equal cuBLAS's bit for bit.
        "half_fused_hm_eq_cublas_hm": bool(torch.equal(hm_hf, hm_cub)),
        "half_fused_vs_cublas_maxabs": maxabs(hm_hf, hm_cub),
        "single_hg_eq_cublas_hg": bool(torch.equal(hg_ts, hg_cub)),
        "single_hg_vs_cublas_maxabs": maxabs(hg_ts, hg_cub),
        "rel_l2_half_fused_vs_cublas": float(
            (hm_hf.float() - hm_cub.float()).norm() / hm_cub.float().norm()),
    }

    # -------- ADVERSARIAL sweep over the bitwise claims (section 42.3.2 is explicit that the
    # Triton==cuBLAS accumulator equality must be RE-VERIFIED for every shape and config shipped,
    # and section 40.3.2 is the retraction that happened when a rule was claimed on 14 points).
    # `half_fused` leans on it hardest: its h_mid mixes a cuBLAS h_gate with a Triton h_up, so it
    # equals the shipped h_mid only if Triton's h_up accumulator equals cuBLAS's bit for bit.
    cells = []
    for seed in (11, 22, 33, 44, 55):
        for scale in (0.005, 0.02, 0.08, 0.3):
            gg = torch.Generator(device=dev).manual_seed(seed)
            xs = torch.randn(M, K, device=dev, dtype=torch.float32, generator=gg).to(dt)
            ws = (torch.randn(N, K, device=dev, dtype=torch.float32, generator=gg)
                  * scale).to(dt).contiguous()
            ws2 = (torch.randn(N, K, device=dev, dtype=torch.float32, generator=gg)
                   * scale).to(dt).contiguous()
            torch.mm(xs, ws.t(), out=hg_c)
            torch.mm(xs, ws2.t(), out=hu_c)
            silu_mul(hg_c, hu_c, hm_c)
            ref_mid = hm_c.clone()
            ref_hu = hu_c.clone()
            # half-fusion: cuBLAS h_gate already in hg_c; Triton computes h_up in registers.
            _launch_single(cgl, xs, ws2, hg_c, hm_h, True)
            eq_half = bool(torch.equal(hm_h, ref_mid))
            # the single-output Triton GEMM's own accumulator, against cuBLAS's
            _launch_single(cs, xs, ws2, hg_s, hm_h, False)
            eq_single = bool(torch.equal(hg_s, ref_hu))
            # the dual path, at the NEW winning config
            _launch_dual(cd, xs, ws, ws2, hg_t, hu_t, hm_t, False)
            eq_dual_gemm = bool(torch.equal(hg_t, hg_c))
            silu_mul(hg_t, hu_t, hm_t)
            ref_dual_mid = hm_t.clone()
            _launch_dual(cd, xs, ws, ws2, hg_t, hu_t, hm_f, True)
            eq_dual_fuse = bool(torch.equal(hm_f, ref_dual_mid))
            cells.append(dict(seed=seed, scale=scale,
                              half_fused_eq_cublas=eq_half,
                              single_eq_cublas=eq_single,
                              dual_gemm_eq_cublas=eq_dual_gemm,
                              dual_fused_eq_dual_mat=eq_dual_fuse,
                              half_maxabs=maxabs(hm_h, ref_mid)))
            del xs, ws, ws2, ref_mid, ref_hu, ref_dual_mid
    checks["sweep_cells"] = len(cells)
    checks["sweep_half_fused_eq_cublas_pass"] = sum(c["half_fused_eq_cublas"] for c in cells)
    checks["sweep_single_eq_cublas_pass"] = sum(c["single_eq_cublas"] for c in cells)
    checks["sweep_dual_gemm_eq_cublas_pass"] = sum(c["dual_gemm_eq_cublas"] for c in cells)
    checks["sweep_dual_fused_eq_mat_pass"] = sum(c["dual_fused_eq_dual_mat"] for c in cells)
    checks["sweep_detail"] = cells
    torch.cuda.empty_cache()

    per = _time_rotation({
        "cublas_1gemm": arm_cublas_1gemm,
        "triton_single": arm_triton_single,
        "cublas_sep": arm_cublas_sep,
        "triton_dual_mat": arm_dual_mat,
        "triton_dual_fused": arm_dual_fused,
        "half_fused": arm_half_fused,
        "control": arm_control,
    }, reps=reps)

    def stats(v):
        return dict(min=min(v), median=statistics.median(v), mean=sum(v) / len(v), max=max(v))

    st = {k: stats(v) for k, v in per.items()}
    parts, drift = _drift(per["control"], reps)

    single_tax = 100.0 * (st["triton_single"]["min"] - st["cublas_1gemm"]["min"]) \
        / st["cublas_1gemm"]["min"]
    dual_tax = 100.0 * (st["triton_dual_mat"]["min"] - st["cublas_sep"]["min"]) \
        / st["cublas_sep"]["min"]
    prize = 100.0 * (st["triton_dual_mat"]["min"] - st["triton_dual_fused"]["min"]) \
        / st["triton_dual_mat"]["min"]
    half_net = 100.0 * (st["cublas_sep"]["min"] - st["half_fused"]["min"]) / st["cublas_sep"]["min"]
    dual_net = 100.0 * (st["cublas_sep"]["min"] - st["triton_dual_fused"]["min"]) \
        / st["cublas_sep"]["min"]

    return dict(
        label=label, M=M, K=K, N=N, reps=reps, sel_reps=sel_reps,
        best_dual_cfg=cd, best_single_cfg=cs, best_glu_cfg=cgl,
        best_dual_tag=best_dual["tag"], best_single_tag=best_single["tag"],
        best_glu_tag=best_glu["tag"],
        selection=dict(table=sel_table, errors=errors,
                       control_thirds_min_ms=sel_parts, control_drift_pct=sel_drift,
                       control_ok=bool(sel_drift <= 1.0)),
        times_ms=st,
        control_thirds_min_ms=parts, control_drift_pct=drift, control_ok=bool(drift <= 1.0),
        bitwise=checks,
        # THE FOUR NUMBERS THIS PROBE EXISTS FOR
        triton_single_tax_vs_cublas_pct=single_tax,
        triton_dual_tax_vs_cublas_pct=dual_tax,
        fusion_prize_at_constant_gemm_pct=prize,
        half_fused_net_vs_cublas_sep_pct=half_net,
        dual_fused_net_vs_cublas_sep_pct=dual_net,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sel-reps", type=int, default=60)
    ap.add_argument("--shapes", default="seq1024_b2,seq2048_b2,seq4096_b2")
    ap.add_argument("--out", default="results/recon/dual_gemm_close.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 2
    dev = "cuda"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flashffn                                                        # noqa: E402

    def silu_mul(hg, hu, out):
        """The SHIPPED forward SwiGLU, called exactly as the block calls it."""
        n = hg.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        flashffn._silu_mul_fwd_kernel[grid](hg, hu, out, out, n, BLOCK_SIZE=BLOCK, HAS_DIG=False)

    want = [s for s in SHAPES if s[0] in args.shapes.split(",")]
    gpu_before = os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader").read()

    out = dict(
        note=("Section 42.4 step 2. Separates the Triton-vs-cuBLAS tax (single-output) from the "
              "dual-output structure tax, widens the dual config space to the 16-warp 256x128 "
              "tiles the register algebra predicts, and adds the serialised `half_fused` design "
              "point. Rotation window, MIN estimator, control gates every comparison. NUMBERS "
              "probe inside the owned fusion family (section 40.1/C12) -- not a novelty claim."),
        torch=torch.__version__, triton=triton.__version__,
        device=torch.cuda.get_device_name(0),
        gpu_state_before=gpu_before.strip().splitlines(),
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        shapes=[],
    )
    for label, M, K, N in want:
        print(f"=== {label}  M={M} K={K} N={N} ===", flush=True)
        r = run_shape(label, M, K, N, args.reps, args.sel_reps, dev, silu_mul)
        out["shapes"].append(r)
        t = r["times_ms"]
        print(f"  selection: control drift {r['selection']['control_drift_pct']:.2f}% "
              f"({'VALID' if r['selection']['control_ok'] else 'VOID'}), "
              f"{len(r['selection']['table'])} feasible, {len(r['selection']['errors'])} failed")
        print(f"    best dual   {r['best_dual_tag']}")
        print(f"    best single {r['best_single_tag']}   best glu {r['best_glu_tag']}")
        for k in ("cublas_1gemm", "triton_single", "cublas_sep", "triton_dual_mat",
                  "triton_dual_fused", "half_fused", "control"):
            print(f"    {k:18s} min {t[k]['min']:.4f} ms")
        print(f"  control drift {r['control_drift_pct']:.2f}% -> "
              f"{'VALID' if r['control_ok'] else 'VOID'}")
        print(f"  TRITON TAX single-output {r['triton_single_tax_vs_cublas_pct']:+.2f}%   "
              f"dual-output {r['triton_dual_tax_vs_cublas_pct']:+.2f}%")
        print(f"  fusion prize at constant GEMM {r['fusion_prize_at_constant_gemm_pct']:+.2f}%")
        print(f"  NET vs cublas_sep:  dual_fused {r['dual_fused_net_vs_cublas_sep_pct']:+.2f}%   "
              f"half_fused {r['half_fused_net_vs_cublas_sep_pct']:+.2f}%")
        b = r["bitwise"]
        print(f"  BITWISE dual_fused==dual_mat {b['dual_fused_eq_dual_mat']}   "
              f"half_fused==cublas {b['half_fused_hm_eq_cublas_hm']}   "
              f"single_hg==cublas_hg {b['single_hg_eq_cublas_hg']}")
        print(f"  sweep {b['sweep_cells']} cells: half_fused=={b['sweep_half_fused_eq_cublas_pass']}"
              f"  single=={b['sweep_single_eq_cublas_pass']}"
              f"  dual_gemm=={b['sweep_dual_gemm_eq_cublas_pass']}"
              f"  dual_fused=={b['sweep_dual_fused_eq_mat_pass']}", flush=True)

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
