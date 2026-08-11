"""Why does the Triton dual-GEMM's tax GROW with M, and can a better tile schedule remove it?

`src/probe_dual_gemm_close.py` closed most of section 42.4 step 2: with a widened config space and a
protocol-faithful selection window the dual-output tax against cuBLAS fell from +0.50/+13.14/+11.88%
to +0.10/+4.02/+8.21% at M=2048/4096/8192, and the fused arm went from LOSING 3.23% at M=8192 to
winning 4.83%.  What is left is a residue that still grows with M -- and growth with M is the exact
signature section 42.6 records for the FIRST defect of this detour: a tile schedule that re-reads the
weight matrices once per tile-row group.  That defect was fixed by adding the standard GROUP_M=8
swizzle; GROUP_M was then never swept.

The arithmetic says it should matter here.  Both weights together are N*K*2*2 = 46 MiB against the
A40's 6 MiB L2, so a group of GROUP_M tile-rows streams all 46 MiB, and the number of groups is
M / (GROUP_M * BLOCK_M).  At BLOCK_M=256, GROUP_M=8 that is 4 groups at M=8192 and 8 at M=16384 --
184 MiB and 368 MiB of avoidable weight traffic, growing linearly in M exactly as the residue does.

THIS PROBE VARIES ONLY THE SCHEDULE, never the arithmetic.  Two knobs:

  1. GROUP_M in {1, 4, 8, 16, 32} at the winning tile shapes.  The CTA->tile map cannot move a bit;
     the bitwise checks below assert that rather than assuming it.
  2. A PERSISTENT variant: grid = number of SMs, each program walking a strided sequence of tiles.
     This is the "persistent tiles" section 42.4 names as a reason to reach for CUTLASS -- it is
     available in Triton for the cost of one loop, so it should be priced before CUTLASS is.

Method is section 40.6 unchanged: every schedule is an arm in ONE rotation window with the control,
MIN estimator, `out=` pre-allocated.

NOT A NOVELTY CLAIM -- section 42.0.  Tile scheduling is CUTLASS/cuBLAS ground and this is a numbers
detour inside the owned fusion family.

Usage:
    PYTHONPATH=src python src/probe_dual_gemm_sched.py --reps 200 \
        --shapes seq4096_b2,seq8192_b2 --out results/recon/dual_gemm_sched.json
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

from probe_dual_gemm_close import (SHAPES, _dual_gemm_glu_kernel, _drift, _launch_dual,
                                   _time_rotation)

# The tile shapes that won `probe_dual_gemm_close.py`'s selection window, plus the two runners-up
# worth re-testing once the schedule is fixed (a bad schedule can hide a better tile).
TILES = [
    # the three that led `probe_dual_gemm_close.py`'s selection at every valid shape
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=2),
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=32, num_warps=8, num_stages=4),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=4),
    # OCCUPANCY.  Every config above spends 98304 of the A40's 101376 B/CTA of shared memory, so it
    # runs ONE CTA per SM -- 8 warps out of the 48 an SM can hold, with nothing to hide a global
    # miss behind.  That is a second, independent explanation for a tax that grows with M, and it
    # is not tested by varying GROUP_M.  These halve smem to fit TWO CTAs per SM.
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=32, num_warps=8, num_stages=2),    # smem 49152
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=2),   # smem 49152
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3),    # smem 49152
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2),    # smem 65536
]
GROUPS = [4, 8, 16]


@triton.jit
def _dual_gemm_glu_persistent(
    x_ptr, wg_ptr, wu_ptr,
    hg_ptr, hu_ptr, hm_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    NUM_TILES,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, FUSE: tl.constexpr,
):
    """The same arithmetic as `_dual_gemm_glu_kernel`, walked by a persistent grid.

    One program per SM, each stepping through tiles `pid, pid+G, pid+2G, ...`.  Consecutive tiles in
    one program are near each other in the grouped order, so a program's weight tiles stay resident
    across iterations instead of being re-fetched by a fresh CTA.  The K loop, the accumulator dtype
    and the epilogue -- including the bf16 round-trip that IS the bitwise contract (section 42.3.1)
    -- are copied verbatim from the non-persistent kernel.
    """
    start = tl.program_id(0)
    step = tl.num_programs(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    for tile in range(start, NUM_TILES, step):
        group_id = tile // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((tile % num_pid_in_group) % group_size_m)
        pid_n = (tile % num_pid_in_group) // group_size_m

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


def _launch_persistent(cfg, x, wg, wu, hg, hu, hm, fuse, num_sms):
    M, K = x.shape
    N = wg.shape[0]
    tiles = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
    grid = (min(num_sms * cfg.get("CTA_PER_SM", 1), tiles),)
    _dual_gemm_glu_persistent[grid](
        x, wg, wu, hg, hu, hm, M, N, K,
        x.stride(0), x.stride(1), wg.stride(0), wg.stride(1), hm.stride(0), hm.stride(1),
        tiles,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg.get("GROUP_M", 8), FUSE=fuse,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def _tag(cfg, kind):
    return (f"{kind}{cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']}"
            f"_w{cfg['num_warps']}s{cfg['num_stages']}_g{cfg.get('GROUP_M', 8)}"
            + (f"c{cfg['CTA_PER_SM']}" if 'CTA_PER_SM' in cfg else ""))


def _parse_fixed(spec):
    """"256x64x64_w8s2_g8,128x64x64_w4s2_g8" -> config dicts, for the confirmation mode.

    WHY THIS MODE EXISTS.  The selection phase is the LONG phase -- 49 arms -- and on a contended
    box it is the one that voids: at seq 2048 and seq 4096 its control drifted 1.53% and 5.04% while
    the short main window drifted 0.64% and 0.34%.  A void selection then picks a config that the
    valid window shows underperforming, which is the same failure the whole probe was written to
    avoid, arriving by a different route.  With the shortlist already known, skipping selection puts
    every candidate into the SHORT window instead, where each gets a comparison the control can
    actually gate.
    """
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        tile, warps, group = tok.split("_")
        bm, bn, bk = (int(v) for v in tile.split("x"))
        nw, ns = warps[1:].split("s")
        out.append(dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=int(nw),
                        num_stages=int(ns), GROUP_M=int(group[1:])))
    return out


def run_fixed(label, M, K, N, reps, dev, silu_mul, fixed_cfgs):
    """Confirmation window: every candidate config contributes BOTH its arms, no selection phase."""
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
    cx = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cw = torch.randn(N, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cout = torch.empty(M, N, device=dev, dtype=dt)
    wgT, wuT, cwT = wg.t(), wu.t(), cw.t()

    fns = {
        "control": lambda: torch.mm(cx, cwT, out=cout),
        "cublas_sep": None, "cublas_2gemm": None,
    }

    def arm_cublas_sep():
        torch.mm(x, wgT, out=hg_c)
        torch.mm(x, wuT, out=hu_c)
        silu_mul(hg_c, hu_c, hm_c)

    def arm_cublas_2gemm():
        torch.mm(x, wgT, out=hg_c)
        torch.mm(x, wuT, out=hu_c)

    fns["cublas_sep"] = arm_cublas_sep
    fns["cublas_2gemm"] = arm_cublas_2gemm

    tags, errors = [], []
    for cfg in fixed_cfgs:
        tag = _tag(cfg, "N")
        try:
            _launch_dual(cfg, x, wg, wu, hg_t, hu_t, hm_f, True)
            _launch_dual(cfg, x, wg, wu, hg_t, hu_t, hm_t, False)
            torch.cuda.synchronize()
        except Exception as e:                                            # noqa: BLE001
            errors.append(dict(tag=tag, err=f"{type(e).__name__}: {e}"[:200]))
            continue
        tags.append((tag, cfg))
        fns[tag + ":fused"] = (lambda c=cfg: _launch_dual(c, x, wg, wu, hg_t, hu_t, hm_f, True))
        fns[tag + ":mat"] = (lambda c=cfg: _launch_dual(c, x, wg, wu, hg_t, hu_t, hm_t, False))

    # bitwise, per config, before timing
    arm_cublas_sep(); torch.cuda.synchronize()
    hg_cub, hm_cub = hg_c.clone(), hm_c.clone()
    bw = {}
    for tag, cfg in tags:
        _launch_dual(cfg, x, wg, wu, hg_t, hu_t, hm_t, False); torch.cuda.synchronize()
        eq_gemm = bool(torch.equal(hg_t, hg_cub))
        silu_mul(hg_t, hu_t, hm_t)
        ref = hm_t.clone()
        _launch_dual(cfg, x, wg, wu, hg_t, hu_t, hm_f, True); torch.cuda.synchronize()
        bw[tag] = dict(gemm_eq_cublas=eq_gemm, fused_eq_mat=bool(torch.equal(hm_f, ref)),
                       fused_eq_cublas_hm=bool(torch.equal(hm_f, hm_cub)))
        del ref

    per = _time_rotation(fns, reps=reps)

    def stats(v):
        return dict(min=min(v), median=statistics.median(v), mean=sum(v) / len(v), max=max(v))

    st = {k: stats(v) for k, v in per.items()}
    parts, drift = _drift(per["control"], reps)
    fwd_cublas, rec_cublas = st["cublas_sep"]["min"], st["cublas_2gemm"]["min"]

    rows = []
    for tag, cfg in tags:
        f, m = st[tag + ":fused"]["min"], st[tag + ":mat"]["min"]
        rows.append(dict(
            tag=tag, cfg=cfg, fused_ms=f, mat_gemm_only_ms=m,
            gemm_tax_pct=100.0 * (m - rec_cublas) / rec_cublas,
            fused_net_vs_cublas_sep_pct=100.0 * (fwd_cublas - f) / fwd_cublas,
            landing_option_a_pct=100.0 * ((fwd_cublas + rec_cublas) - (f + m))
            / (fwd_cublas + rec_cublas),
            landing_option_b_pct=100.0 * ((fwd_cublas + rec_cublas) - (f + rec_cublas))
            / (fwd_cublas + rec_cublas),
            bitwise=bw[tag]))
    rows.sort(key=lambda r: -r["landing_option_a_pct"])

    return dict(label=label, M=M, K=K, N=N, reps=reps, mode="fixed",
                times_ms=st, control_thirds_min_ms=parts, control_drift_pct=drift,
                control_ok=bool(drift <= 1.0), errors=errors, configs=rows)


def run_shape(label, M, K, N, reps, sel_reps, dev, silu_mul, num_sms):
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
    cx = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cw = torch.randn(N, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cout = torch.empty(M, N, device=dev, dtype=dt)
    wgT, wuT, cwT = wg.t(), wu.t(), cw.t()

    def arm_control():
        torch.mm(cx, cwT, out=cout)

    # ---------------- PHASE A: every schedule an arm in ONE selection window ----------------
    sel_fns = {"control": arm_control}
    feasible, errors = [], []
    for tile in TILES:
        for gm in GROUPS:
            cfg = dict(tile, GROUP_M=gm)
            tag = _tag(cfg, "N")
            try:
                _launch_dual(cfg, x, wg, wu, hg_t, hu_t, hm_f, True)
                torch.cuda.synchronize()
            except Exception as e:                                        # noqa: BLE001
                errors.append(dict(tag=tag, err=f"{type(e).__name__}: {e}"[:200]))
                continue
            feasible.append((tag, "nonpersistent", cfg))
            sel_fns[tag] = (lambda c=cfg: _launch_dual(c, x, wg, wu, hg_t, hu_t, hm_f, True))
        for gm in (8, 16):
            for cta in (1, 2):
                cfg = dict(tile, GROUP_M=gm, CTA_PER_SM=cta)
                tag = _tag(cfg, "P")
                try:
                    _launch_persistent(cfg, x, wg, wu, hg_t, hu_t, hm_f, True, num_sms)
                    torch.cuda.synchronize()
                except Exception as e:                                    # noqa: BLE001
                    errors.append(dict(tag=tag, err=f"{type(e).__name__}: {e}"[:200]))
                    continue
                feasible.append((tag, "persistent", cfg))
                sel_fns[tag] = (lambda c=cfg: _launch_persistent(
                    c, x, wg, wu, hg_t, hu_t, hm_f, True, num_sms))

    sel_per = _time_rotation(sel_fns, reps=sel_reps)
    sel_parts, sel_drift = _drift(sel_per["control"], sel_reps)
    sel_table = sorted(
        [dict(tag=t, kind=k, cfg=c, min_ms=min(sel_per[t]),
              median_ms=statistics.median(sel_per[t])) for t, k, c in feasible],
        key=lambda r: r["min_ms"])
    best = sel_table[0]
    bc, bkind = best["cfg"], best["kind"]

    def _fused():
        if bkind == "persistent":
            _launch_persistent(bc, x, wg, wu, hg_t, hu_t, hm_f, True, num_sms)
        else:
            _launch_dual(bc, x, wg, wu, hg_t, hu_t, hm_f, True)

    def _mat_gemm_only():
        if bkind == "persistent":
            _launch_persistent(bc, x, wg, wu, hg_t, hu_t, hm_t, False, num_sms)
        else:
            _launch_dual(bc, x, wg, wu, hg_t, hu_t, hm_t, False)

    def arm_mat():
        _mat_gemm_only()
        silu_mul(hg_t, hu_t, hm_t)

    def arm_cublas_sep():
        torch.mm(x, wgT, out=hg_c)
        torch.mm(x, wuT, out=hu_c)
        silu_mul(hg_c, hu_c, hm_c)

    def arm_cublas_2gemm():
        """The RECOMPUTE's actual shape of work: two GEMMs, NO SwiGLU.

        `flashffn.py:3306-3312` shows the recompute normally does not form `h_mid` -- that is folded
        into `triton_swiglu_backward_glu`.  So the price the recompute pays for leaving cuBLAS must
        be measured against two bare GEMMs, not against `cublas_sep`.
        """
        torch.mm(x, wgT, out=hg_c)
        torch.mm(x, wuT, out=hu_c)

    # -------------------------- bitwise: the schedule must move no bit --------------------------
    arm_cublas_sep(); torch.cuda.synchronize()
    hg_cub, hm_cub = hg_c.clone(), hm_c.clone()
    arm_mat(); torch.cuda.synchronize()
    hg_best, hm_best = hg_t.clone(), hm_t.clone()
    _fused(); torch.cuda.synchronize()
    hm_fused = hm_f.clone()
    ref_cfg = dict(TILES[0], GROUP_M=8)
    _launch_dual(ref_cfg, x, wg, wu, hg_t, hu_t, hm_t, False); torch.cuda.synchronize()
    hg_ref = hg_t.clone()

    checks = dict(
        best_fused_eq_best_mat=bool(torch.equal(hm_fused, hm_best)),
        best_gemm_eq_cublas=bool(torch.equal(hg_best, hg_cub)),
        best_gemm_eq_group8_reference=bool(torch.equal(hg_best, hg_ref)),
        best_mat_hm_eq_cublas_hm=bool(torch.equal(hm_best, hm_cub)),
        maxabs_fused_vs_cublas=float((hm_fused.float() - hm_cub.float()).abs().max()),
    )

    per = _time_rotation({
        "cublas_sep": arm_cublas_sep,
        "cublas_2gemm": arm_cublas_2gemm,
        "best_mat": arm_mat,
        "best_mat_gemm_only": _mat_gemm_only,
        "best_fused": _fused,
        "control": arm_control,
    }, reps=reps)

    def stats(v):
        return dict(min=min(v), median=statistics.median(v), mean=sum(v) / len(v), max=max(v))

    st = {k: stats(v) for k, v in per.items()}
    parts, drift = _drift(per["control"], reps)

    gemm_tax = 100.0 * (st["best_mat_gemm_only"]["min"] - st["cublas_2gemm"]["min"]) \
        / st["cublas_2gemm"]["min"]
    prize = 100.0 * (st["best_mat"]["min"] - st["best_fused"]["min"]) / st["best_mat"]["min"]
    net_fwd = 100.0 * (st["cublas_sep"]["min"] - st["best_fused"]["min"]) / st["cublas_sep"]["min"]
    # THE LANDING ARITHMETIC.  Gate B3 makes the four `keep` levels agree bitwise WITH EACH OTHER,
    # so if the keep='min' FORWARD fuses on Triton, every other level's gate/up GEMM must use the
    # same kernel -- including the recompute, which pays `gemm_tax` on two bare GEMMs.  That is
    # option (a), agreement by construction.  Option (b) leaves the recompute on cuBLAS and leans on
    # Triton's accumulator equalling cuBLAS's, which section 42.3.2 forbids promoting to a rule.
    fwd_cublas, rec_cublas = st["cublas_sep"]["min"], st["cublas_2gemm"]["min"]
    opt_a = 100.0 * ((fwd_cublas + rec_cublas)
                     - (st["best_fused"]["min"] + st["best_mat_gemm_only"]["min"])) \
        / (fwd_cublas + rec_cublas)
    opt_b = 100.0 * ((fwd_cublas + rec_cublas) - (st["best_fused"]["min"] + rec_cublas)) \
        / (fwd_cublas + rec_cublas)

    return dict(
        label=label, M=M, K=K, N=N, reps=reps, sel_reps=sel_reps, num_sms=num_sms,
        best_tag=best["tag"], best_kind=bkind, best_cfg=bc,
        selection=dict(table=sel_table, errors=errors, control_thirds_min_ms=sel_parts,
                       control_drift_pct=sel_drift, control_ok=bool(sel_drift <= 1.0)),
        times_ms=st, control_thirds_min_ms=parts, control_drift_pct=drift,
        control_ok=bool(drift <= 1.0), bitwise=checks,
        dual_gemm_tax_vs_cublas_pct=gemm_tax,
        fusion_prize_at_constant_gemm_pct=prize,
        fused_net_vs_cublas_sep_pct=net_fwd,
        landing_option_a_pct=opt_a,
        landing_option_b_pct=opt_b,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--sel-reps", type=int, default=60)
    ap.add_argument("--shapes", default="seq2048_b2,seq4096_b2,seq8192_b2")
    ap.add_argument("--fixed", default="",
                    help="skip selection; time these configs directly, e.g. "
                         "'256x64x64_w8s2_g8,128x64x64_w4s2_g8'")
    ap.add_argument("--out", default="results/recon/dual_gemm_sched.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 2
    dev = "cuda"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flashffn                                                        # noqa: E402

    def silu_mul(hg, hu, out):
        n = hg.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        flashffn._silu_mul_fwd_kernel[grid](hg, hu, out, out, n, BLOCK_SIZE=BLOCK, HAS_DIG=False)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    want = [s for s in SHAPES if s[0] in args.shapes.split(",")]
    out = dict(
        note=("Section 42.4 step 2, second pass: vary ONLY the tile schedule (GROUP_M, persistent "
              "grid). The residue being attacked grows with M, which is the signature of the "
              "weight-refetch defect section 42.6 records. NUMBERS probe, owned family."),
        torch=torch.__version__, triton=triton.__version__,
        device=torch.cuda.get_device_name(0), num_sms=num_sms,
        gpu_state_before=os.popen(
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader"
        ).read().strip().splitlines(),
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        shapes=[],
    )
    fixed_cfgs = _parse_fixed(args.fixed) if args.fixed else None
    for label, M, K, N in want:
        print(f"=== {label}  M={M} K={K} N={N} ===", flush=True)
        if fixed_cfgs:
            r = run_fixed(label, M, K, N, args.reps, dev, silu_mul, fixed_cfgs)
            out["shapes"].append(r)
            t = r["times_ms"]
            print(f"    cublas_sep {t['cublas_sep']['min']:.4f}   "
                  f"cublas_2gemm {t['cublas_2gemm']['min']:.4f}   "
                  f"control {t['control']['min']:.4f}")
            print(f"  control drift {r['control_drift_pct']:.2f}% -> "
                  f"{'VALID' if r['control_ok'] else 'VOID'}")
            for row in r["configs"]:
                print(f"    {row['tag']:26s} fused {row['fused_ms']:7.4f}  "
                      f"mat {row['mat_gemm_only_ms']:7.4f}  tax {row['gemm_tax_pct']:+6.2f}%  "
                      f"opt_a {row['landing_option_a_pct']:+6.2f}%  "
                      f"opt_b {row['landing_option_b_pct']:+6.2f}%  "
                      f"bw {int(row['bitwise']['gemm_eq_cublas'])}"
                      f"{int(row['bitwise']['fused_eq_mat'])}"
                      f"{int(row['bitwise']['fused_eq_cublas_hm'])}")
            print("", flush=True)
            continue
        r = run_shape(label, M, K, N, args.reps, args.sel_reps, dev, silu_mul, num_sms)
        out["shapes"].append(r)
        t = r["times_ms"]
        print(f"  selection drift {r['selection']['control_drift_pct']:.2f}% "
              f"({len(r['selection']['table'])} feasible, {len(r['selection']['errors'])} failed)")
        print(f"  BEST SCHEDULE: {r['best_tag']}  ({r['best_kind']})")
        for row in r["selection"]["table"][:6]:
            print(f"      {row['tag']:34s} {row['min_ms']:.4f} ms")
        for k in ("cublas_sep", "cublas_2gemm", "best_mat", "best_mat_gemm_only", "best_fused",
                  "control"):
            print(f"    {k:20s} min {t[k]['min']:.4f} ms")
        print(f"  control drift {r['control_drift_pct']:.2f}% -> "
              f"{'VALID' if r['control_ok'] else 'VOID'}")
        print(f"  DUAL GEMM TAX vs 2 cuBLAS GEMMs {r['dual_gemm_tax_vs_cublas_pct']:+.2f}%   "
              f"prize {r['fusion_prize_at_constant_gemm_pct']:+.2f}%")
        print(f"  LANDING  option a (recompute also Triton, bitwise by construction) "
              f"{r['landing_option_a_pct']:+.2f}%")
        print(f"           option b (recompute stays cuBLAS) {r['landing_option_b_pct']:+.2f}%")
        print(f"  BITWISE {r['bitwise']}", flush=True)

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
