"""Does the FORWARD/RECOMPUTE materialisation asymmetry buy anything?  (CONTEXT.md section 42)

THE QUESTION.  At `keep='min'` the forward and the recompute run *identical* code, but they have
different requirements:

  * the FORWARD's only surviving output is `x_{k+1}` -- `h_gate`/`h_up`/`h_mid` are written to DRAM
    and thrown away;
  * the RECOMPUTE must materialise `h_gate`/`h_up`, because the gradient pass reads them at G4
    (`flashffn.py:3338`).

So the forward could in principle keep `h_gate`/`h_up` in registers and never write them, while the
recompute keeps writing them.  Gate B3 survives *if and only if* the two paths produce bitwise
identical values, which requires (a) the same GEMM tiling / K-reduction order in both, and (b) the
fused epilogue to reproduce the bf16 rounding that the separate path performs when it stores
`h_gate`/`h_up` and loads them back.  (b) is the subtle one and is handled explicitly below.

WHAT THIS MEASURES.  Four arms at the real FFN shapes, in ONE rotation window (section 40.6: all
variants inside one window, rotate which leads each rep, pre-allocated `out=`, >=150 reps, report the
MIN, carry a control):

  cublas_sep   two cuBLAS GEMMs -> h_gate, h_up  +  the shipped Triton SwiGLU -> h_mid   [today]
  triton_fused ONE Triton dual-GEMM with a SwiGLU epilogue -> h_mid, h_gate/h_up never stored
  triton_mat   the SAME Triton kernel with FUSE=False -> h_gate, h_up  +  Triton SwiGLU -> h_mid
  control      a fixed cuBLAS GEMM none of the arms touch

`triton_mat` is the arm that decides the direction, and it is easy to forget: if the fused forward
uses a Triton GEMM, then B3 forces the RECOMPUTE onto the same Triton GEMM, so the recompute pays
`triton_mat - cublas_sep`.  The net is therefore

    (cublas_sep - triton_fused)        [forward saves]
  - (triton_mat - cublas_sep)          [recompute pays]

and it is negative whenever Triton's GEMM is worse than cuBLAS's by more than the fusion saves.
That is the whole bet, and it is why this probe exists before any kernel is landed in the block.

NOT A NOVELTY CLAIM.  This is inside the fusion family (CONTEXT.md section 40.1/C12 -- MoMoE, CODA,
SonicMoE own "inline the rematerialisation so intermediates never reach DRAM"; TiledMLP/ALST and
Liger own token-tiled MLP).  It is a NUMBERS detour, explicitly authorised as such.  Do not let it
drift into a contribution.

Usage:
    PYTHONPATH=src python src/probe_fused_glu_fwd.py --reps 200 \
        --out results/recon/fused_glu_fwd.json
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

# Real FFN shapes: M = batch*seq tokens, K = hidden, N = intermediate (TinyLlama-1.1B, batch 2).
SHAPES = [
    ("seq1024_b2", 2048, 2048, 5632),
    ("seq2048_b2", 4096, 2048, 5632),
    ("seq4096_b2", 8192, 2048, 5632),
]

# Tile configs swept for the Triton dual-GEMM.  Both FUSE=True and FUSE=False are always run with
# the SAME config, which is what makes their intermediate values bitwise identical by construction.
#
# TWO CONSTRAINTS SHAPE THIS LIST, and both are structural to a DUAL-output GEMM rather than
# accidents of this implementation:
#
#  (1) SHARED MEMORY.  Each pipeline stage stages one x tile AND TWO weight tiles, so smem per
#      stage is (BM*BK + 2*BK*BN)*2 bytes -- 1.5-2x a single-output GEMM's.  The A40's limit is
#      101376 B/CTA, which is why the first pass of this probe saw every BLOCK_N=128, stages>=3,
#      BLOCK_K=64 config die with `OutOfResources: Required 122880`.
#  (2) REGISTERS.  Two fp32 accumulators cost 2*BM*BN*4 bytes/CTA.  At 8 warps a 128x128 dual tile
#      already needs ~128 registers/thread for accumulators alone; 256x128 needs ~256 and cannot
#      compile.  A single-output GEMM gets 256x128 for the same register budget.
#
# So a fused dual-GEMM structurally buys HALF the tile area per register of the two separate GEMMs
# it replaces -- and tile area is what buys arithmetic intensity.  That is the real tension, and it
# is why the configs below top out at 128x128.
CONFIGS = [
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=4),   # smem 98304
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=2),   # smem 98304
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3),    # smem 98304
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=5),    # smem 81920
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2),    # smem 81920
    dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=4),    # smem 81920
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4),
    dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=6),
]

# Grouped tile ordering (the standard Triton matmul swizzle).  Without it the (pid_m, pid_n)
# traversal is row-major and every tile column re-reads both weight matrices from DRAM, which is
# why the first pass of this probe degraded from +19% to +182% vs cuBLAS as M grew 4x.
GROUP_M = 8


@triton.jit
def _dual_gemm_glu_kernel(
    x_ptr, wg_ptr, wu_ptr,
    hg_ptr, hu_ptr, hm_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FUSE: tl.constexpr,
):
    """h_gate = x @ wg^T and h_up = x @ wu^T, sharing one load of the x tile.

    FUSE=True   -> apply SiLU(h_gate)*h_up in the epilogue and store ONLY h_mid.
    FUSE=False  -> store h_gate and h_up, exactly as the shipped path does.

    THE BITWISE CONTRACT.  With FUSE=True the epilogue rounds each accumulator to bf16 and casts it
    back to fp32 *before* the activation.  That reproduces, in registers, precisely the rounding the
    separate path performs when it stores `h_gate`/`h_up` as bf16 and `_silu_mul_fwd_kernel`
    (`flashffn.py:62-67`) loads them back with `.to(tl.float32)`.  Without those two casts the fused
    arm is MORE accurate and therefore NOT bitwise equal, and gate B3 fails -- which is the whole
    reason the forward and the recompute may differ in materialisation but not in arithmetic.
    """
    # Grouped ordering for L2 reuse: walk GROUP_M tile-rows at a time so consecutive CTAs share
    # weight tiles.  Same tiles, same K-loop, same arithmetic -- only the CTA->tile map changes,
    # so this cannot move a single bit of the result.
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
        # See THE BITWISE CONTRACT above: bf16 round-trip in registers, then the shipped expression.
        g = acc_g.to(tl.bfloat16).to(tl.float32)
        u = acc_u.to(tl.bfloat16).to(tl.float32)
        act = g * tl.sigmoid(g)
        res = act * u
        hm_ptrs = hm_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(hm_ptrs, res.to(tl.bfloat16), mask=out_mask)
    else:
        hg_ptrs = hg_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        hu_ptrs = hu_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(hg_ptrs, acc_g.to(tl.bfloat16), mask=out_mask)
        tl.store(hu_ptrs, acc_u.to(tl.bfloat16), mask=out_mask)


def _launch(cfg, x, wg, wu, hg, hu, hm, fuse):
    M, K = x.shape
    N = wg.shape[0]
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _dual_gemm_glu_kernel[grid](
        x, wg, wu, hg, hu, hm,
        M, N, K,
        x.stride(0), x.stride(1),
        wg.stride(0), wg.stride(1),
        hm.stride(0), hm.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=GROUP_M,
        FUSE=fuse,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )


def _sync():
    torch.cuda.synchronize()


def _time_rotation(fns, reps, warmup=12):
    """All variants inside ONE window, rotating which leads each rep (section 40.6)."""
    names = list(fns.keys())
    k = len(names)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    _sync()
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


def run_shape(label, M, K, N, reps, dev, silu_mul):
    dt = torch.bfloat16
    g = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    wg = torch.randn(N, K, device=dev, dtype=torch.float32, generator=g).to(dt) * 0.02
    wu = torch.randn(N, K, device=dev, dtype=torch.float32, generator=g).to(dt) * 0.02
    wg = wg.contiguous()
    wu = wu.contiguous()

    # Pre-allocated outputs, one set per arm -- `out=` everywhere so no arm pays an allocation.
    hg_c = torch.empty(M, N, device=dev, dtype=dt)
    hu_c = torch.empty(M, N, device=dev, dtype=dt)
    hm_c = torch.empty(M, N, device=dev, dtype=dt)
    hg_t = torch.empty(M, N, device=dev, dtype=dt)
    hu_t = torch.empty(M, N, device=dev, dtype=dt)
    hm_t = torch.empty(M, N, device=dev, dtype=dt)
    hm_f = torch.empty(M, N, device=dev, dtype=dt)
    # control: same-ish cost, touched by nobody else
    cx = torch.randn(M, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cw = torch.randn(N, K, device=dev, dtype=torch.float32, generator=g).to(dt)
    cout = torch.empty(M, N, device=dev, dtype=dt)

    wgT = wg.t()
    wuT = wu.t()
    cwT = cw.t()

    def arm_cublas():
        torch.mm(x, wgT, out=hg_c)
        torch.mm(x, wuT, out=hu_c)
        silu_mul(hg_c, hu_c, hm_c)

    def arm_control():
        torch.mm(cx, cwT, out=cout)

    # ---- pick the best Triton config, timed the same way, so the comparison is not rigged ----
    cfg_times = []
    for cfg in CONFIGS:
        try:
            _launch(cfg, x, wg, wu, hg_t, hu_t, hm_f, True)
            _sync()
        except Exception as e:                                   # noqa: BLE001
            cfg_times.append((float("inf"), cfg, f"{type(e).__name__}: {e}"))
            continue
        per = _time_rotation({"f": lambda c=cfg: _launch(c, x, wg, wu, hg_t, hu_t, hm_f, True)},
                             reps=20, warmup=5)
        cfg_times.append((min(per["f"]), cfg, None))
    cfg_times.sort(key=lambda t: t[0])
    best_ms, best_cfg, _ = cfg_times[0]
    if best_cfg is None or best_ms == float("inf"):
        raise RuntimeError("every Triton config failed for %s" % label)

    def arm_fused():
        _launch(best_cfg, x, wg, wu, hg_t, hu_t, hm_f, True)

    def arm_mat():
        _launch(best_cfg, x, wg, wu, hg_t, hu_t, hm_t, False)
        silu_mul(hg_t, hu_t, hm_t)

    # ---------------- bitwise checks, before timing (they mutate the buffers) ----------------
    arm_cublas(); _sync()
    hg_cub = hg_c.clone(); hu_cub = hu_c.clone(); hm_cub = hm_c.clone()
    arm_mat(); _sync()
    hg_tri = hg_t.clone(); hu_tri = hu_t.clone(); hm_mat = hm_t.clone()
    arm_fused(); _sync()
    hm_fus = hm_f.clone()

    def maxabs(a, b):
        return float((a.float() - b.float()).abs().max())

    checks = {
        # THE GATING CHECK: can the forward fuse while the recompute materialises, bitwise?
        "fused_hm_eq_mat_hm": bool(torch.equal(hm_fus, hm_mat)),
        "fused_hm_vs_mat_hm_maxabs": maxabs(hm_fus, hm_mat),
        # Whether the recompute must ALSO leave cuBLAS.  If Triton != cuBLAS bitwise it must.
        "triton_hg_eq_cublas_hg": bool(torch.equal(hg_tri, hg_cub)),
        "triton_hg_vs_cublas_hg_maxabs": maxabs(hg_tri, hg_cub),
        "fused_hm_eq_cublas_hm": bool(torch.equal(hm_fus, hm_cub)),
        "fused_hm_vs_cublas_hm_maxabs": maxabs(hm_fus, hm_cub),
        "rel_l2_fused_vs_cublas_hm": float(
            (hm_fus.float() - hm_cub.float()).norm() / hm_cub.float().norm()),
    }

    # ---- ADVERSARIAL sweep over the bitwise claims (section 40.3.2: state the measured RANGE,
    # not a rule).  A single draw already cost this project one retracted "invariant" on 14 points.
    cells = []
    for seed in (11, 22, 33, 44, 55):
        for scale in (0.005, 0.02, 0.08, 0.3):
            gg = torch.Generator(device=dev).manual_seed(seed)
            xs = torch.randn(M, K, device=dev, dtype=torch.float32, generator=gg).to(dt)
            ws = (torch.randn(N, K, device=dev, dtype=torch.float32, generator=gg)
                  * scale).to(dt).contiguous()
            ws2 = (torch.randn(N, K, device=dev, dtype=torch.float32, generator=gg)
                   * scale).to(dt).contiguous()
            a = torch.empty(M, N, device=dev, dtype=dt)
            b = torch.empty(M, N, device=dev, dtype=dt)
            m1 = torch.empty(M, N, device=dev, dtype=dt)
            m2 = torch.empty(M, N, device=dev, dtype=dt)
            torch.mm(xs, ws.t(), out=a)
            _launch(best_cfg, xs, ws, ws2, b, m1, m2, False)
            eq_gemm = bool(torch.equal(a, b))
            _launch(best_cfg, xs, ws, ws2, b, m1, m2, False)
            silu_mul(b, m1, m2)
            ref_mid = m2.clone()
            _launch(best_cfg, xs, ws, ws2, b, m1, m2, True)
            eq_fuse = bool(torch.equal(m2, ref_mid))
            cells.append(dict(seed=seed, scale=scale,
                              triton_gemm_eq_cublas=eq_gemm, fused_eq_mat=eq_fuse,
                              gemm_maxabs=maxabs(a, b)))
            del xs, ws, ws2, a, b, m1, m2, ref_mid
    checks["sweep_cells"] = len(cells)
    checks["sweep_fused_eq_mat_pass"] = sum(c["fused_eq_mat"] for c in cells)
    checks["sweep_triton_eq_cublas_pass"] = sum(c["triton_gemm_eq_cublas"] for c in cells)
    checks["sweep_detail"] = cells
    torch.cuda.empty_cache()

    # ------------------------------- the timing window -------------------------------
    per = _time_rotation(
        {"cublas_sep": arm_cublas, "triton_fused": arm_fused,
         "triton_mat": arm_mat, "control": arm_control},
        reps=reps)

    def stats(v):
        return dict(min=min(v), median=statistics.median(v), mean=sum(v) / len(v), max=max(v))

    st = {k: stats(v) for k, v in per.items()}
    # Control drift measured the way section 40.6 requires: thirds of the window, on the MIN.
    third = max(1, reps // 3)
    cparts = [min(per["control"][:third]), min(per["control"][third:2 * third]),
              min(per["control"][2 * third:])]
    drift = 100.0 * (max(cparts) - min(cparts)) / min(cparts)

    fwd_saves = st["cublas_sep"]["min"] - st["triton_fused"]["min"]
    bwd_pays = st["triton_mat"]["min"] - st["cublas_sep"]["min"]
    # THE TWO NUMBERS THAT MATTER, separated.  `triton_fused` vs `triton_mat` holds the GEMM
    # constant, so their difference is the fusion prize ALONE, uncontaminated by kernel quality;
    # `triton_mat` vs `cublas_sep` is the price of leaving cuBLAS.  Reporting only the net would
    # hide that a better GEMM (CUTLASS) changes the verdict without changing the prize.
    fusion_prize_pct = 100.0 * (st["triton_mat"]["min"] - st["triton_fused"]["min"]) \
        / st["triton_mat"]["min"]
    gemm_penalty_pct = 100.0 * (st["triton_mat"]["min"] - st["cublas_sep"]["min"]) \
        / st["cublas_sep"]["min"]

    return dict(
        label=label, M=M, K=K, N=N, reps=reps,
        best_cfg=best_cfg,
        cfg_sweep=[dict(min_ms=t, cfg=c, err=e) for t, c, e in cfg_times],
        times_ms=st,
        control_thirds_min_ms=cparts,
        control_drift_pct=drift,
        control_ok=bool(drift <= 1.0),
        bitwise=checks,
        forward_saves_ms=fwd_saves,
        recompute_pays_ms=bwd_pays,
        net_per_layer_per_step_ms=fwd_saves - bwd_pays,
        forward_saves_pct_of_cublas=100.0 * fwd_saves / st["cublas_sep"]["min"],
        recompute_pays_pct_of_cublas=100.0 * bwd_pays / st["cublas_sep"]["min"],
        fusion_prize_at_constant_gemm_pct=fusion_prize_pct,
        triton_gemm_penalty_vs_cublas_pct=gemm_penalty_pct,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--out", default="results/recon/fused_glu_fwd.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 2
    dev = "cuda"

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import flashffn                                                    # noqa: E402

    def silu_mul(hg, hu, out):
        """The SHIPPED forward SwiGLU, called exactly as the block calls it."""
        n = hg.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        flashffn._silu_mul_fwd_kernel[grid](hg, hu, out, out, n, BLOCK_SIZE=BLOCK, HAS_DIG=False)

    want = SHAPES if args.shapes == "all" else [s for s in SHAPES if s[0] in args.shapes.split(",")]
    gpu_before = os.popen(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader").read()

    out = dict(
        note=("Rotation timing on a possibly contended box: the MIN over reps is the estimator "
              "(section 40.6). `control_ok` gates every comparison. This is a NUMBERS probe inside "
              "the owned fusion family (section 40.1/C12) -- not a novelty claim."),
        torch=torch.__version__, triton=triton.__version__,
        device=torch.cuda.get_device_name(0),
        gpu_state_before=gpu_before.strip().splitlines(),
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        shapes=[],
    )
    for label, M, K, N in want:
        print(f"=== {label}  M={M} K={K} N={N} ===", flush=True)
        r = run_shape(label, M, K, N, args.reps, dev, silu_mul)
        out["shapes"].append(r)
        t = r["times_ms"]
        print(f"  cublas_sep   min {t['cublas_sep']['min']:.4f} ms")
        print(f"  triton_fused min {t['triton_fused']['min']:.4f} ms   "
              f"(forward saves {r['forward_saves_pct_of_cublas']:+.2f}%)")
        print(f"  triton_mat   min {t['triton_mat']['min']:.4f} ms   "
              f"(recompute pays {r['recompute_pays_pct_of_cublas']:+.2f}%)")
        print(f"  control drift {r['control_drift_pct']:.2f}%  -> "
              f"{'VALID' if r['control_ok'] else 'VOID'}")
        print(f"  BITWISE fused==mat : {r['bitwise']['fused_hm_eq_mat_hm']}   "
              f"triton==cublas h_gate : {r['bitwise']['triton_hg_eq_cublas_hg']}")
        print(f"  FUSION PRIZE at constant GEMM: {r['fusion_prize_at_constant_gemm_pct']:+.2f}%"
              f"   TRITON GEMM penalty vs cuBLAS: {r['triton_gemm_penalty_vs_cublas_pct']:+.2f}%")
        print(f"  bitwise sweep: fused==mat {r['bitwise']['sweep_fused_eq_mat_pass']}"
              f"/{r['bitwise']['sweep_cells']}   triton==cublas "
              f"{r['bitwise']['sweep_triton_eq_cublas_pass']}/{r['bitwise']['sweep_cells']}")
        print(f"  net per layer per step: {r['net_per_layer_per_step_ms']:+.4f} ms", flush=True)

    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
