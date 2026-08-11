#!/usr/bin/env python
"""Can a bandwidth-bound kernel hide inside a tensor-core-bound one?  The roofline pairing test.

`src/probe_stream_overlap.py` closed one question: two DENSE bf16 GEMMs do not co-schedule
profitably on this A40.  That is the expected answer -- both saturate the same resource (the
tensor cores), so a second stream can only time-slice them.

It never tested the one pairing class the A40's roofline actually leaves room for:

    a MEMORY-BANDWIDTH-bound kernel run concurrently with a TENSOR-CORE-bound kernel.

The arithmetic on this card:  a dense bf16 GEMM at these shapes runs near 115 TFLOP/s with an
arithmetic intensity around 1436 FLOP/byte, i.e. it draws only ~82 GB/s of the ~696 GB/s HBM
bandwidth -- about 12% of the memory system.  A pointwise Triton kernel is the mirror image: it
saturates HBM and leaves the tensor cores at ~0%.  On paper the two are complementary and the
smaller one should be nearly free.  Whether the hardware's warp scheduler, L2 and CTA dispatch
actually deliver that is an empirical question that has never been asked here.

The estimator is the same one the earlier probe used:

    run op A alone,  run op B alone,  run A and B concurrently on two streams

    overlap_gain    = (t_A + t_B) / t_concurrent - 1
    fraction_hidden = (t_A + t_B - t_concurrent) / min(t_A, t_B)

      fraction_hidden = 1.00  the smaller op became completely free  (what we want)
                        0.00  concurrency bought exactly nothing
                       <0.00  contention: two streams are SLOWER than one

`fraction_hidden` is the headline: it is normalised by the op that could in principle disappear,
so it is comparable across pairs whose absolute times differ by 50x.

CONTROLS (the earlier probe had none, which weakened its result):

  1. Every `alone` time is measured TWICE -- once before the concurrent phase and once after it --
     and the drift is reported per op.  On a box shared with other tenants at 100% utilisation, a
     drifting baseline can manufacture or destroy an apparent overlap.  Any op drifting more than
     `--drift_tol` (default 2%) sets `control_ok: false` for the whole file.
  2. Two NULL pairings are measured alongside the real ones:
         `gemm_gate x gemm_dx`   tensor-core x tensor-core -- must reproduce ~0 or negative gain
         `silu_mul  x elt_add`   bandwidth  x bandwidth    -- must also show ~0 gain
     If either null shows a large positive `fraction_hidden`, the harness is measuring something
     other than co-scheduling and EVERY number in the file is void.
  3. `nvidia-smi` state is captured before, midway through the pair sweep, and after.

THE CHUNKING VARIANT (`--chunks K`): with K>1 the filler op is issued as K sequential smaller
pieces split along the token/row dimension, inside both the alone leg and the concurrent leg.  The
hypothesis is that one large filler grid queues behind the cover kernel's CTAs, whereas K small
grids can trickle into the tail of each cover wave.  Both legs are chunked so the comparison stays
one-variable.

Every op is a REAL kernel from `src/flashffn.py` at the real model shapes -- no stand-ins:

    cover  (tensor-core-bound)   gemm_gate, gemm_dx, flash_bwd
    filler (bandwidth-bound)     silu_mul, glu_bwd, rmsnorm, rope, lora_down, lora_up, elt_add

Usage:
    PYTHONPATH=src python src/probe_roofline_overlap.py --seq 4096 --chunks 1,8
    PYTHONPATH=src python src/probe_roofline_overlap.py --seq 16384 --chunks 1,8 --reps 5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import profile_hyclora as ph  # noqa: E402
from flashffn import (  # noqa: E402
    triton_silu_mul,
    triton_swiglu_backward_glu,
    fb_rmsnorm_forward,
    _fb_rope_bsh,
)


def _gpu_state():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], text=True, timeout=20)
        return [l.strip() for l in out.strip().splitlines()]
    except Exception as exc:
        return [f"unavailable: {exc}"]


COVERS = ("gemm_gate", "gemm_dx", "flash_bwd")
FILLERS = ("silu_mul", "glu_bwd", "rmsnorm", "rope", "lora_down", "lora_up", "elt_add")
NULLS = (("gemm_gate", "gemm_dx"), ("silu_mul", "elt_add"))


def _split(n, k):
    """k contiguous row ranges covering [0, n).  The last piece absorbs the remainder."""
    step = n // k
    bounds = [(i * step, (i + 1) * step) for i in range(k)]
    bounds[-1] = (bounds[-1][0], n)
    return bounds


class Bench:
    """Allocates every tensor once; `ops(K)` builds the callables for a chunk setting.

    Filler tensors and cover tensors are DISJOINT.  Nothing the filler writes is read by the
    cover and vice versa, so the two streams carry no data dependency at all -- this is the
    ceiling case, the most favourable a real scheduler could ever reach.
    """

    def __init__(self, cfg, device):
        dt = torch.bfloat16
        self.dt = dt
        self.device = device
        H = self.H = cfg["hidden"]
        I = self.I = cfg["intermediate"]
        nh = self.nh = cfg["n_heads"]
        nkv = self.nkv = cfg["n_kv_heads"]
        hd = self.hd = cfg["head_dim"]
        B = self.B = cfg["batch"]
        S = self.S = cfg["seq"]
        N = self.N = B * S
        r = self.r = cfg.get("lora_r", 16)
        self.eps = cfg.get("rms_eps", 1e-5)

        g = torch.Generator(device=device).manual_seed(1234)

        def rnd(*shape, scale=1.0):
            t = torch.empty(*shape, device=device, dtype=dt)
            t.normal_(0.0, 1.0, generator=g)
            if scale != 1.0:
                t.mul_(scale)
            return t

        # ---- cover side (tensor-core-bound) -------------------------------------------------
        self.x_c = rnd(N, H)
        self.wg = rnd(I, H)
        self.gh_c = rnd(N, I)

        # attention: `flash_bwd` copied verbatim from probe_stream_overlap.py
        self.q = rnd(B, nh, S, hd)
        self.k = rnd(B, nkv, S, hd)
        self.v = rnd(B, nkv, S, hd)
        self.qg = self.q.clone().requires_grad_(True)
        self.kg = self.k.clone().requires_grad_(True)
        self.vg = self.v.clone().requires_grad_(True)
        o_ref = self._sdpa(self.qg, self.kg, self.vg)
        self.go = torch.randn_like(o_ref)
        del o_ref

        # ---- filler side (bandwidth-bound) --------------------------------------------------
        self.s_gate = rnd(N, I, scale=0.5)          # silu_mul inputs
        self.s_up = rnd(N, I, scale=0.5)
        self.b_gm = rnd(N, I, scale=0.5)            # glu_bwd inputs (MUTATED in place)
        self.b_gate = rnd(N, I, scale=0.5)
        self.b_up = rnd(N, I, scale=0.5)
        self.n_x = rnd(B, S, H)                     # rmsnorm
        self.n_w = rnd(H)
        self.rope_q = rnd(B, S, nh, hd)             # rope
        self.rope_cos = rnd(N, hd)                  # N rows so the chunked view can index it
        self.rope_sin = rnd(N, hd)
        self.ld_x = rnd(N, H)                       # lora_down
        self.ld_a = rnd(r, H)
        self.lu_y = rnd(N, I, scale=0.01)           # lora_up (read-modify-write)
        self.lu_xa = rnd(N, r, scale=0.01)
        self.lu_b = rnd(I, r, scale=0.01)
        self.e_x = rnd(B, S, H, scale=1e-3)         # elt_add
        self.e_y = rnd(B, S, H, scale=1e-3)
        self._g = g

    def _sdpa(self, qq, kk, vv):
        # TinyLlama is GQA (32 query heads, 4 KV heads); sdpa needs enable_gqa, exactly as the
        # fused block's own `_fb_flash_fwd` relies on.
        return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True,
                                              enable_gqa=(self.nkv != self.nh))

    # ------------------------------------------------------------------ op construction
    def ops(self, K):
        H, I, N, B, S = self.H, self.I, self.N, self.B, self.S
        nh, nkv, hd, r = self.nh, self.nkv, self.hd, self.r
        es = 2  # bf16
        out = {}

        # ---------------- covers (never chunked) ----------------
        def gemm_gate():
            with torch.no_grad():
                F.linear(self.x_c, self.wg)

        def gemm_dx():
            with torch.no_grad():
                self.gh_c @ self.wg

        def flash_bwd():
            o = self._sdpa(self.qg, self.kg, self.vg)
            torch.autograd.grad(o, (self.qg, self.kg, self.vg), self.go, retain_graph=False)

        out["gemm_gate"] = dict(fn=gemm_gate, kind="cover", chunks=1,
                                flops=2 * N * H * I,
                                bytes=(N * H + I * H + N * I) * es,
                                shape=f"[{N},{H}] x [{I},{H}]^T -> [{N},{I}]")
        out["gemm_dx"] = dict(fn=gemm_dx, kind="cover", chunks=1,
                              flops=2 * N * I * H,
                              bytes=(N * I + I * H + N * H) * es,
                              shape=f"[{N},{I}] x [{I},{H}] -> [{N},{H}]")
        # FA FLOP model: causal halves both the QK^T and the PV GEMM; the backward is the usual
        # 2.5x of the forward, and this op re-runs the forward under grad, so 3.5x total.
        fa_fwd = 2 * B * nh * S * S * hd  # = 2 * (2*B*nh*S^2*hd) * 0.5
        out["flash_bwd"] = dict(fn=flash_bwd, kind="cover", chunks=1,
                                flops=int(3.5 * fa_fwd), bytes=None,
                                flops_model="3.5 x causal fwd (fwd + 2.5x bwd); approximate",
                                shape=f"q[{B},{nh},{S},{hd}] kv[{B},{nkv},{S},{hd}] causal")

        # ---------------- fillers (chunked along rows/tokens) ----------------
        # silu_mul
        if K == 1:
            sm_pieces = [(self.s_gate, self.s_up)]
        else:
            sm_pieces = [(self.s_gate[a:b], self.s_up[a:b]) for a, b in _split(N, K)]

        def silu_mul():
            for hg, hu in sm_pieces:
                triton_silu_mul(hg, hu)

        def silu_mul_reset():
            self.s_gate.normal_(0.0, 0.5, generator=self._g)
            self.s_up.normal_(0.0, 0.5, generator=self._g)

        out["silu_mul"] = dict(fn=silu_mul, reset=silu_mul_reset, kind="filler", chunks=K,
                               flops=4 * N * I, bytes=3 * N * I * es,
                               shape=f"2 reads + 1 write of [{N},{I}]")

        # glu_bwd -- the in-place variant, as flashffn.py:3338 calls it in the LoRA family
        # (want_h_mid=True because the down-adapter's gradient needs h_mid).  It MUTATES all
        # three inputs, so `reset` refills them before every timed block.
        if K == 1:
            gb_pieces = [(self.b_gm, self.b_gate, self.b_up)]
        else:
            gb_pieces = [(self.b_gm[a:b], self.b_gate[a:b], self.b_up[a:b])
                         for a, b in _split(N, K)]

        def glu_bwd():
            for gm, hg, hu in gb_pieces:
                triton_swiglu_backward_glu(gm, hg, hu, True, inplace=True)

        def glu_bwd_reset():
            self.b_gm.normal_(0.0, 0.5, generator=self._g)
            self.b_gate.normal_(0.0, 0.5, generator=self._g)
            self.b_up.normal_(0.0, 0.5, generator=self._g)

        out["glu_bwd"] = dict(fn=glu_bwd, reset=glu_bwd_reset, kind="filler", chunks=K,
                              flops=10 * N * I, bytes=6 * N * I * es,
                              shape=f"3 reads + 3 in-place writes of [{N},{I}]")

        # rmsnorm
        if K == 1:
            rn_pieces = [self.n_x]
        else:
            x2 = self.n_x.view(N, H)
            rn_pieces = [x2[a:b] for a, b in _split(N, K)]

        def rmsnorm():
            for xp in rn_pieces:
                fb_rmsnorm_forward(xp, self.n_w, self.eps)

        out["rmsnorm"] = dict(fn=rmsnorm, kind="filler", chunks=K,
                              flops=5 * N * H, bytes=2 * N * H * es + 4 * N,
                              shape=f"[{B},{S},{H}] -> y + rstd[{N}]")

        # rope -- at K=1 the real [B,S,nh,hd] call with cos/sin[:S]; at K>1 the same kernel over
        # [1, n_chunk, nh, hd] views (identical grid, strides and byte traffic; only the cos row
        # range differs, and cos is 4 MiB and L2-resident).
        if K == 1:
            rp_pieces = [(self.rope_q, self.rope_cos[:S], self.rope_sin[:S])]
        else:
            q3 = self.rope_q.view(N, nh, hd)
            rp_pieces = []
            for a, b in _split(N, K):
                n = b - a
                rp_pieces.append((q3[a:b].view(1, n, nh, hd),
                                  self.rope_cos[:n], self.rope_sin[:n]))

        def rope():
            for xq, c, s in rp_pieces:
                _fb_rope_bsh(xq, c, s)

        out["rope"] = dict(fn=rope, kind="filler", chunks=K,
                           flops=3 * N * nh * hd, bytes=2 * N * nh * hd * es,
                           shape=f"q[{B},{S},{nh},{hd}]")

        # lora_down -- the A-side GEMM: 2*N*H*r FLOP over N*H*2 bytes -> AI ~= r = 16 FLOP/byte
        if K == 1:
            ld_pieces = [self.ld_x]
        else:
            ld_pieces = [self.ld_x[a:b] for a, b in _split(N, K)]

        def lora_down():
            with torch.no_grad():
                for xp in ld_pieces:
                    F.linear(xp, self.ld_a)

        out["lora_down"] = dict(fn=lora_down, kind="filler", chunks=K,
                                flops=2 * N * H * r,
                                bytes=(N * H + r * H + N * r) * es,
                                shape=f"[{N},{H}] x [{r},{H}]^T -> [{N},{r}]")

        # lora_up -- the B-side read-modify-write on the full [N, I] activation
        bt = self.lu_b.t()
        if K == 1:
            lu_pieces = [(self.lu_y, self.lu_xa)]
        else:
            lu_pieces = [(self.lu_y[a:b], self.lu_xa[a:b]) for a, b in _split(N, K)]

        def lora_up():
            with torch.no_grad():
                for yp, xp in lu_pieces:
                    yp.addmm_(xp, bt)

        def lora_up_reset():
            self.lu_y.normal_(0.0, 0.01, generator=self._g)

        out["lora_up"] = dict(fn=lora_up, reset=lora_up_reset, kind="filler", chunks=K,
                              flops=2 * N * r * I,
                              bytes=(2 * N * I + N * r + I * r) * es,
                              shape=f"[{N},{I}].addmm_([{N},{r}], [{r},{I}])")

        # elt_add
        if K == 1:
            ea_pieces = [(self.e_x, self.e_y)]
        else:
            x2 = self.e_x.view(N, H)
            y2 = self.e_y.view(N, H)
            ea_pieces = [(x2[a:b], y2[a:b]) for a, b in _split(N, K)]

        def elt_add():
            for xp, yp in ea_pieces:
                xp.add_(yp)

        def elt_add_reset():
            self.e_x.normal_(0.0, 1e-3, generator=self._g)

        out["elt_add"] = dict(fn=elt_add, reset=elt_add_reset, kind="filler", chunks=K,
                              flops=N * H, bytes=3 * N * H * es,
                              shape=f"[{B},{S},{H}].add_")

        return out


# ---------------------------------------------------------------------- timing harness
# ONE pair of side streams for the whole run.  Creating a fresh `torch.cuda.Stream()` per pair
# leaves each pair's freed blocks in a dead stream's allocator pool, so later default-stream
# allocations have to `cudaMalloc` fresh -- which showed up as a spurious 30x "drift" on the
# allocating ops in the first smoke run.  Persistent streams remove that artifact entirely.
_STREAMS = []
# Events are reused for the same reason plus one of its own: `torch.cuda.Event` creates the
# underlying `cudaEvent` LAZILY, on first `record()`.  Fresh Event objects per timing block put
# that one-off cost inside the CPU-issue window and, at `reps=2`, inflated the measured launch
# cost by ~25 us/rep -- enough to flag ops as launch-bound that are not.
_EVENTS = []


def _streams():
    if not _STREAMS:
        _STREAMS.extend([torch.cuda.Stream(), torch.cuda.Stream()])
    return _STREAMS


def _events():
    if not _EVENTS:
        _EVENTS.extend([torch.cuda.Event(True), torch.cuda.Event(True)])
        # Force the lazy cudaEventCreate out of every subsequent measurement window.
        _EVENTS[0].record()
        _EVENTS[1].record()
        torch.cuda.synchronize()
    return _EVENTS


def time_alone(op, reps, warmup):
    """Returns (gpu_ms_per_rep, wall_ms_per_rep, launch_ms_per_rep).

    `launch_ms` is the CPU time to ENQUEUE one rep, measured in the same loop before any sync.
    If `launch_ms >= gpu_ms` the measurement is launch-bound, not GPU-bound, and any apparent
    overlap is Python filling GPU gaps rather than the hardware co-scheduling anything.  It is the
    third control, and it is what makes the null pairings interpretable.
    """
    if op.get("reset"):
        op["reset"]()
    fn = op["fn"]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = _events()
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (ev0.elapsed_time(ev1) / reps, (t2 - t0) * 1e3 / reps, (t1 - t0) * 1e3 / reps)


def time_concurrent(op_a, op_b, reps, warmup):
    """A on stream 1, B on stream 2, no dependency either way -- the best case for overlap."""
    for op in (op_a, op_b):
        if op.get("reset"):
            op["reset"]()
    fn_a, fn_b = op_a["fn"], op_b["fn"]
    s1, s2 = _streams()
    for _ in range(warmup):
        with torch.cuda.stream(s1):
            fn_a()
        with torch.cuda.stream(s2):
            fn_b()
    torch.cuda.synchronize()
    ev0, ev1 = _events()
    cur = torch.cuda.current_stream()
    t0 = time.perf_counter()
    ev0.record()
    # The side streams must not be allowed to start BEFORE the start event is processed; without
    # this barrier they have no dependency on the default stream at all and, at low `reps`, the
    # measured window can begin after work has already run.
    s1.wait_stream(cur)
    s2.wait_stream(cur)
    for _ in range(reps):
        with torch.cuda.stream(s1):
            fn_a()
        with torch.cuda.stream(s2):
            fn_b()
    # The end event must observe BOTH streams, so join them into the default stream first.
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)
    ev1.record()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (ev0.elapsed_time(ev1) / reps, (t2 - t0) * 1e3 / reps, (t1 - t0) * 1e3 / reps)


def _stats(xs):
    """`spread_pct` is MEDIAN-over-min, not max-over-min.

    The estimator is the min, so one catastrophic block (another tenant grabbing the card for a
    whole slice) must not disqualify a measurement -- and it was doing exactly that: single
    outliers pushed max/min to 3000% on ops whose median sat 1% above their min.  Median/min asks
    the question that matters: were MOST blocks clean?  The max is kept for the record.
    """
    ys = sorted(xs)
    n = len(ys)
    lo, med, hi = ys[0], ys[n // 2], ys[-1]
    return {"min": lo, "median": med, "max": hi,
            "spread_pct": (med - lo) / lo * 100.0 if lo > 0 else None,
            "max_pct": (hi - lo) / lo * 100.0 if lo > 0 else None}


def measure_pair(op_a, op_b, reps, warmup, blocks):
    """Interleave the three legs and take the MIN over blocks.

    The dev box is shared with two other tenants and sits at 100% utilisation, so every timing is
    an uncontended time PLUS a non-negative, time-varying interference term.  Two consequences:

      * the three legs must be measured ADJACENT IN TIME, or a slow patch lands on one leg and
        manufactures (or destroys) an overlap that is not there.  Each block runs A, B, A||B
        back to back, and `blocks` blocks are run.
      * the MIN over blocks is the right estimator, not the mean: interference only ever ADDS
        time, so the minimum is the closest thing to the uncontended number.  The full
        min/median/max is recorded so the reader can see how loud the box was.

    A previous non-interleaved, mean-based version of this same probe produced `fraction_hidden`
    values of +1.78 and -5.46 on ops that cannot physically do either.  That is what this replaces.
    """
    ta, tb, tc, la, lb, lc = [], [], [], [], [], []
    for i in range(blocks):
        w = warmup if i == 0 else 1
        # ALTERNATE the order of the three legs.  With a fixed A, B, A||B order the concurrent
        # leg always sits last in the block and inherits any monotonic within-block effect
        # (clock ramp, cache state, the allocator settling).  That is not hypothetical: the fixed
        # order produced `rmsnorm x flash_bwd  fraction_hidden = +3.87` at seq 2048 -- a saving
        # larger than the whole of the op being hidden, which is physically impossible and can
        # only mean the cover itself was timed faster in the concurrent leg.  Reversing on odd
        # blocks makes the position effect cancel in the median.
        legs = [("a", op_a), ("b", op_b), ("c", None)]
        if i % 2:
            legs.reverse()
        for tag, op in legs:
            if tag == "c":
                t, _, l = time_concurrent(op_a, op_b, reps, w)
                tc.append(t)
                lc.append(l)
            else:
                t, _, l = time_alone(op, reps, w)
                (ta if tag == "a" else tb).append(t)
                (la if tag == "a" else lb).append(l)
    return {"ta": ta, "tb": tb, "tc": tc, "la": min(la), "lb": min(lb), "lc": min(lc)}


def _q(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))]


def paired_metrics(raw):
    """THE estimator.  Form the ratio INSIDE each block, then take the median across blocks.

    An aggregate-then-divide estimator (min-of-A + min-of-B over min-of-C, or the same with
    medians) is biased on a contended card, and the bias has a sign: the concurrent leg is the
    LONGEST of the three windows, so it straddles more of the driver's time slices and its
    aggregate is contaminated more than either `alone` leg's.  Measured at seq 4096, that bias
    alone turned +0.5% overlap into -21.6% "contention" on `lora_down x gemm_dx`.

    Forming `gain_i = (ta_i + tb_i)/tc_i - 1` within a block and taking the median cancels the
    common-mode contention: all three legs of a block are measured within tens of milliseconds of
    each other and see the same neighbours.  The interquartile range across blocks is reported as
    the honest uncertainty, and it -- not a max/min spread -- decides whether a row is quotable.
    """
    ta, tb, tc = raw["ta"], raw["tb"], raw["tc"]
    gains = [(a + b) / c - 1.0 for a, b, c in zip(ta, tb, tc)]
    fhids = [(a + b - c) / min(a, b) for a, b, c in zip(ta, tb, tc)]
    return {
        "overlap_gain": _q(gains, 0.5), "fraction_hidden": _q(fhids, 0.5),
        "overlap_gain_p25": _q(gains, 0.25), "overlap_gain_p75": _q(gains, 0.75),
        "fraction_hidden_p25": _q(fhids, 0.25), "fraction_hidden_p75": _q(fhids, 0.75),
        "overlap_gain_iqr": _q(gains, 0.75) - _q(gains, 0.25),
        "fraction_hidden_iqr": _q(fhids, 0.75) - _q(fhids, 0.25),
    }


def measure_alone(op, reps, warmup, blocks):
    ts, ls = [], []
    for i in range(blocks):
        w = warmup if i == 0 else 1
        t, _, l = time_alone(op, reps, w)
        ts.append(t)
        ls.append(l)
    return _stats(ts), min(ls)


def _rate(op, t_ms):
    gb_s = None if op["bytes"] is None else op["bytes"] / (t_ms * 1e-3) / 1e9
    tf_s = None if op["flops"] is None else op["flops"] / (t_ms * 1e-3) / 1e12
    return gb_s, tf_s


def auto_reps(seq):
    """Deliberately TINY.

    Our process shares both A40s with two other tenants that hold the cards at 100% utilisation,
    and CUDA-event timing measures wall time on the device -- including the intervals in which our
    context is descheduled by the driver's time slicer.  A long rep loop is therefore GUARANTEED to
    straddle a slice boundary and the measurement is contaminated no matter how many times it is
    repeated.  A short loop can fit inside one slice, so with enough blocks the MIN finds clean
    ones.  Measured effect at seq 1024: reps=8/blocks=15 gave the cover GEMMs a 26% run-to-run
    drift and `fraction_hidden` values up to +3.03; reps=2/blocks=40 gave <1% drift, 106 TFLOP/s
    on the GEMMs (vs the 149 TFLOP/s A40 bf16 peak) and 560 GB/s on `silu_mul` (vs 696 GB/s peak).
    """
    return 2 if seq <= 4096 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reps", type=int, default=0, help="0 = auto from --seq")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5,
                    help="interleaved A/B/A||B blocks per pair; the min over blocks is reported")
    ap.add_argument("--chunks", default="1", help="comma-separated K list, e.g. 1,8")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--drift_tol", type=float, default=0.02)
    ap.add_argument("--iqr_tol", type=float, default=0.06,
                    help="max interquartile range of the per-block overlap_gain for a row to be "
                         "`reliable` (the honest uncertainty on this shared box)")
    ap.add_argument("--model", default=ph.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    reps = args.reps or auto_reps(args.seq)
    ks = [int(x) for x in args.chunks.split(",") if x.strip()]

    cfg = ph.make_cfg(args.batch, args.seq, model=args.model, lora_r=args.lora_r)
    from transformers import AutoConfig
    hc = AutoConfig.from_pretrained(args.model)
    hc = getattr(hc, "text_config", hc)
    cfg["rms_eps"] = getattr(hc, "rms_norm_eps", 1e-5)
    cfg["reps"] = reps
    cfg["warmup"] = args.warmup
    cfg["blocks"] = args.blocks
    cfg["estimator"] = "min over interleaved A/B/A||B blocks"
    cfg["chunk_settings"] = ks

    state_before = _gpu_state()
    print(f"gpu before: {state_before}")
    bench = Bench(cfg, device)
    torch.cuda.synchronize()
    setup_mib = torch.cuda.memory_allocated(device) / 2**20

    pairs_all = [(f, c) for f in FILLERS for c in COVERS]
    total_units = len(ks) * (len(pairs_all) + len(NULLS))
    done_units = 0
    state_mid = None

    per_k = {}
    control_ok = True
    for K in ks:
        print(f"\n===== chunks K={K}  (seq {args.seq}, reps {reps}, warmup {args.warmup}) =====")
        ops = bench.ops(K)

        alone1, launch1, errs = {}, {}, {}
        for name, op in ops.items():
            try:
                st, ln = measure_alone(op, reps, args.warmup, args.blocks)
            except Exception as exc:
                errs[name] = f"{type(exc).__name__}: {exc}"
                print(f"  !! {name} FAILED alone-1: {errs[name]}")
                continue
            alone1[name], launch1[name] = st, ln
            t = st["min"]
            gb, tf = _rate(op, t)
            lb = "  LAUNCH-BOUND" if ln >= t else ""
            print(f"  {name:<11} [{op['kind']:<6} K={op['chunks']}] alone {t:9.4f} ms "
                  f"(spread {st['spread_pct']:5.1f}%) "
                  f"{('%8.1f GB/s' % gb) if gb else '        n/a':>13}  "
                  f"{('%7.2f TF/s' % tf) if tf else '     n/a':>12}  "
                  f"launch {ln:7.4f}{lb}")

        rows = []
        pair_list = [(f, c, "real") for f, c in pairs_all] + \
                    [(a, b, "null") for a, b in NULLS]
        print(f"\n  {'filler':<11} {'cover':<11} {'t_a':>9} {'t_b':>9} {'sum':>9} "
              f"{'conc':>9} {'gain':>7} {'f_hid':>7} {'IQR':>7}")
        for a, b, tag in pair_list:
            if a in errs or b in errs:
                rows.append({"a": a, "b": b, "tag": tag,
                             "error": f"skipped: {errs.get(a) or errs.get(b)}"})
                print(f"  {a:<11} {b:<11}  SKIPPED (an alone timing failed)")
                done_units += 1
                continue
            try:
                raw = measure_pair(ops[a], ops[b], reps, args.warmup, args.blocks)
            except Exception as exc:
                rows.append({"a": a, "b": b, "tag": tag,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {a:<11} {b:<11}  FAILED: {type(exc).__name__}: {exc}")
                done_units += 1
                continue
            sa, sb, sc = _stats(raw["ta"]), _stats(raw["tb"]), _stats(raw["tc"])
            pm = paired_metrics(raw)
            gain, f_hid = pm["overlap_gain"], pm["fraction_hidden"]
            # Reported times are the per-block MEDIANS; the achieved GB/s and TFLOP/s below use
            # the MINIMA instead, because the roofline question ("did this kernel reach the
            # card's peak?") wants the least-contended observation, not the typical one.
            ta, tb, t_conc = sa["median"], sb["median"], sc["median"]
            gb_a, tf_a = _rate(ops[a], sa["min"])
            gb_b, tf_b = _rate(ops[b], sb["min"])
            l_conc = raw["lc"]
            # Launch-bound: Python could not keep both streams fed, so t_concurrent is a CPU
            # number and any "overlap" is Python filling GPU gaps.  Flagged, never hidden.
            lbound = l_conc >= t_conc
            # `fraction_hidden > 1` says more time was saved than the smaller op takes, i.e. the
            # COVER ran faster under concurrency than alone.  Nothing physical does that; it is
            # always a measurement fault, and it must disqualify the row rather than be reported
            # as a spectacular result.
            impossible = f_hid > 1.05
            reliable = bool(pm["overlap_gain_iqr"] <= args.iqr_tol
                            and not lbound and not impossible)
            rows.append({"a": a, "b": b, "tag": tag,
                         "a_kind": ops[a]["kind"], "b_kind": ops[b]["kind"],
                         "t_a_alone_ms": ta, "t_b_alone_ms": tb,
                         "t_sequential_ms": ta + tb, "t_concurrent_ms": t_conc,
                         "t_a_stats": sa, "t_b_stats": sb, "t_conc_stats": sc,
                         "launch_ms": l_conc, "launch_bound": bool(lbound),
                         "physically_impossible": bool(impossible),
                         "reliable": reliable,
                         "a_gb_s": gb_a, "a_tflop_s": tf_a,
                         "b_gb_s": gb_b, "b_tflop_s": tf_b,
                         **pm})
            print(f"  {a:<11} {b:<11} {ta:>9.4f} {tb:>9.4f} {ta+tb:>9.4f} "
                  f"{t_conc:>9.4f} {gain:>7.3f} {f_hid:>7.3f} "
                  f"{pm['overlap_gain_iqr']:>7.3f}"
                  + ("  LB" if lbound else ("  XX" if impossible else "    "))
                  + ("  ok" if reliable else "  --")
                  + ("   <-- NULL" if tag == "null" else ""))
            done_units += 1
            if state_mid is None and done_units >= total_units // 2:
                state_mid = _gpu_state()
                print(f"  gpu mid: {state_mid}")

        # ---- control 1: re-measure every alone time AFTER the concurrent phase
        # Drift is reported on the MEDIAN (the statistic the pair table reports) and on the MIN
        # (the statistic the roofline rates use).  Both are kept; `control_ok` is the strict
        # reading -- either one exceeding the tolerance fails it.
        alone2, drift, drift_min = {}, {}, {}
        for name, op in ops.items():
            if name in errs:
                continue
            try:
                st, _l = measure_alone(op, reps, args.warmup, args.blocks)
            except Exception as exc:
                errs[name] = f"alone-2 {type(exc).__name__}: {exc}"
                continue
            alone2[name] = st
            drift[name] = (st["median"] - alone1[name]["median"]) / alone1[name]["median"]
            drift_min[name] = (st["min"] - alone1[name]["min"]) / alone1[name]["min"]
        k_ok = (all(abs(d) <= args.drift_tol for d in drift.values())
                and all(abs(d) <= args.drift_tol for d in drift_min.values()) and not errs)
        control_ok = control_ok and k_ok
        print(f"\n  control drift (alone-2 vs alone-1), tol +-{args.drift_tol*100:.0f}%  "
              f"[median | min]:")
        for name in sorted(drift, key=lambda n: -max(abs(drift[n]), abs(drift_min[n]))):
            flag = "  DRIFT" if max(abs(drift[name]), abs(drift_min[name])) > args.drift_tol \
                else ""
            print(f"    {name:<11} {alone1[name]['median']:9.4f} -> "
                  f"{alone2[name]['median']:9.4f}  {drift[name]*100:+7.2f}% | "
                  f"{drift_min[name]*100:+7.2f}%{flag}")
        print(f"  control_ok(K={K}) = {k_ok}")

        # ---- control 2: null-pairing verdict
        nulls = {f"{r['a']}x{r['b']}": {"fraction_hidden": r.get("fraction_hidden"),
                                        "overlap_gain": r.get("overlap_gain")}
                 for r in rows if r["tag"] == "null"}
        null_ok = all(v["fraction_hidden"] is not None and v["fraction_hidden"] < 0.15
                      for v in nulls.values())
        if not null_ok:
            print("  *** NULL PAIRING ANOMALY: a same-resource pairing shows real overlap. ***")
            print("  Two readings, and the sequence-length trend decides between them: either the")
            print("  harness is measuring something other than co-scheduling (in which case every")
            print("  number at this K is VOID), or the 'saturating' op does not actually fill the")
            print("  machine at this shape -- wave quantisation leaves a tail that ANY second")
            print("  kernel can occupy.  The second predicts the null SHRINKS as seq grows and the")
            print("  GEMMs go from ~3 waves to ~49; the first predicts it does not.")
            print("  Whichever it is, the null is the FLOOR: no real pair may be credited with a")
            print("  gain it does not clear.")
        best_real = max((r for r in rows if r["tag"] == "real" and r.get("reliable")),
                        key=lambda r: r["fraction_hidden"], default=None)
        best_any = max((r for r in rows
                        if r["tag"] == "real" and "fraction_hidden" in r),
                       key=lambda r: r["fraction_hidden"], default=None)
        n_reliable = sum(1 for r in rows if r["tag"] == "real" and r.get("reliable"))
        print(f"  reliable real pairs: {n_reliable}/21")
        if best_any and best_any is not best_real:
            print(f"  (best UNRELIABLE real f_hid = {best_any['fraction_hidden']:.3f} "
                  f"({best_any['a']} x {best_any['b']}) -- NOT quotable)")
        null_floor_gain = max((v["overlap_gain"] for v in nulls.values()
                               if v["overlap_gain"] is not None), default=None)
        n_clear = None
        if null_floor_gain is not None:
            n_clear = sum(1 for r in rows if r["tag"] == "real"
                          and r.get("overlap_gain", -9) > null_floor_gain)
        if best_real:
            print(f"  best REAL (bw x tc) fraction_hidden = {best_real['fraction_hidden']:.3f} "
                  f"({best_real['a']} x {best_real['b']}, "
                  f"overlap_gain {best_real['overlap_gain']:+.3f})")
            print(f"  null floor (max null overlap_gain) = {null_floor_gain:+.3f}; "
                  f"{n_clear}/21 real pairs clear it")

        per_k[str(K)] = {
            "chunks": K,
            "alone1_ms": {n: st["median"] for n, st in alone1.items()},
            "alone1_min_ms": {n: st["min"] for n, st in alone1.items()},
            "alone2_ms": {n: st["median"] for n, st in alone2.items()},
            "alone1_stats": alone1, "alone2_stats": alone2,
            "alone1_launch_ms": launch1,
            "launch_bound_ops": [n for n in alone1 if launch1[n] >= alone1[n]["min"]],
            "drift_median": drift, "drift_min": drift_min, "drift": drift,
            "errors": errs,
            "op_meta": {n: {kk: vv for kk, vv in o.items()
                            if kk not in ("fn", "reset")} for n, o in ops.items()},
            "pairs": rows,
            "control_drift_ok": k_ok,
            "nulls": nulls, "null_ok": null_ok,
            "null_floor_overlap_gain": null_floor_gain,
            "real_pairs_clearing_null_floor": n_clear,
            "n_reliable_real_pairs": n_reliable,
            "best_real_pair_any": (f"{best_any['a']}x{best_any['b']}" if best_any else None),
            "best_real_fraction_hidden_any": (best_any["fraction_hidden"] if best_any else None),
            "best_real_fraction_hidden": (best_real["fraction_hidden"] if best_real else None),
            "best_real_overlap_gain": (best_real["overlap_gain"] if best_real else None),
            "best_real_pair": (f"{best_real['a']}x{best_real['b']}" if best_real else None),
        }

    if state_mid is None:
        state_mid = _gpu_state()
    null_ok_all = all(v["null_ok"] for v in per_k.values())
    out = {
        "cfg": cfg, "torch": torch.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "setup_alloc_mib": setup_mib,
        "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "per_chunks": per_k,
        "control_ok": bool(control_ok),
        "null_pairings_ok": bool(null_ok_all),
        "gpu_state_before": state_before,
        "gpu_state_mid": state_mid,
        "gpu_state_after": _gpu_state(),
    }
    path = args.out or f"results/recon/roofline_overlap_seq{args.seq}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\ncontrol_ok={control_ok}  null_pairings_ok={null_ok_all}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
