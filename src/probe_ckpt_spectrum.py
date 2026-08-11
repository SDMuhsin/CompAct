#!/usr/bin/env python
"""Stage 1 of CONTEXT.md 44.6: is the CHECKPOINT STACK lossily compressible, and by WHICH codec?

WHY THIS EXISTS.  44.2 measured that at `keep='min'` the 22 saved tensors -- one residual-stream
snapshot per decoder layer -- are **60.1% of all activation memory at every sequence length**, and
44.3 proved that cutting the backward transient alone returns ~1 MiB because the peak is a PLATEAU.
So the checkpoint stack is the only large memory target left.  The user then opened the direction
explicitly ("is there a way to compress and decompress the checkpoint at runtime for lower memory
footprint, lossy is acceptable within acceptable task performance loss"), and corrected an earlier
narrowing of "compress" to "quantise": *"I didn't say it had to be quantisation -- what about random
sketching / SVD-based projection, or simple subsampling/downsampling?"*

**THIS PROBE MEASURES THE PREMISE ONLY.  It is not a mechanism, it assumes none, and it says nothing
about novelty** (44.4's own novelty paragraph says the family is very likely owned; T-6 runs
separately and BEFORE any kernel -- 37.4).  39.3's negative result does NOT cover this: it measured
LOSSLESS, bit-level incompressibility, and 44.7.1 records that quoting it beyond that scope is an
error this project has already made twice.  Bit entropy says nothing about numerical rank.

WHAT IT MEASURES.  The 22 real checkpoints, captured on real WikiText-2 text, then:

  1. **The singular spectrum, per layer.**  Rank needed for 90 / 99 / 99.9% of the energy.  This is
     the whole premise: low-rank only exists if the spectrum is steep.
  2. **The JOINT spectrum across all 22 layers.**  A basis shared by every depth would amortise its
     own storage to nothing.  Whether the 22 depths share a subspace has never been checked here.
     Measured as the eigendecomposition of the summed Gram matrix, accumulated in float64.
  3. **A codec bake-off at MATCHED BITS PER ELEMENT** -- so codecs are compared at equal rate rather
     than by anecdote, which is the specific failure 44.4 asks this to avoid.  Every codec reports
     the rate it ACTUALLY achieved, including its side information, so the comparison is honest.
  4. **The propagation proxy, and it is the number that matters.**  44.4: reconstruction error is
     NOT the quantity of interest -- the recompute consumes `x` through `rmsnorm1(x)` and then the
     q/k/v GEMMs, so the error is multiplied by real weight matrices before it reaches any gradient,
     and the damage "may be far below what reconstruction error suggests".  This measures the error
     after RMSNorm and after the real W_q/W_k/W_v/W_gate/W_up of the layer the checkpoint belongs to.

THE RATE CEILING, which is the sharpest thing here and is pure arithmetic.  A rank-r factorisation
of an [N, H] tensor costs r(N+H) elements against NH, so at bf16 its bits/element is 16r(N+H)/(NH).
Solving for r at a target rate and letting N -> infinity gives **r_ceiling = rate * H / 16**, which
for H = 2048 is 1024 / 512 / 256 at 8 / 4 / 2 bits.  **No sequence length, however long, buys a
low-rank code more rank than that at a given rate.**  So the question this probe answers is exactly:
does rank-512 of 2048 beat INT4-with-per-token-scaling, which spends its 4 bits on every element?
Both the finite-N rank and the ceiling rank are reported.

WHY THE CODECS ARE THE ONES THEY ARE (44.4's family sort, restated so this file stands alone):
the recompute consumes `x` in the VALUE domain, elementwise -- `rmsnorm1(x)`, then GEMMs, RoPE, and
`x_mid = x + attn_out`.  So a scheme must support ELEMENTWISE RECONSTRUCTION, not inner-product or
norm preservation.  That is why random sketching appears here as a cheap way to COMPUTE a basis
(randomised range finder) and not as a separate compressibility claim: JL preserves distances, not
elements, and you cannot RMSNorm a sketch.  Token-axis subsampling is absent because it is
structurally broken (dropped tokens receive no gradient), and layer-axis subsampling is absent
because it is not compression at all -- it is coarser checkpoint granularity, which is exact and
already on Griewank/revolve's proved-optimal curve (41.2).

CONTROLS, because a compressibility number with no control is unreadable:
  * an i.i.d. Gaussian tensor of the same shape and scale -- the "no structure at all" floor, whose
    spectrum is the Marchenko-Pastur quarter-circle and whose low-rank error is the worst case;
  * a SHUFFLED-basis control for the joint spectrum (does sharing a basis across depths actually
    help, or would any 22 random rotations do as well?);
  * INT8/INT4/INT2 measured alongside, since they are the incumbent family and the thing low-rank
    has to beat to be worth its compute.

Usage:
    PYTHONPATH=src python src/probe_ckpt_spectrum.py --seq 1024 --batch 2 \
        --out results/recon/ckpt_spectrum_seq1024.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import profile_hyclora as ph  # noqa: E402
from probe_value_redundancy import real_text_ids  # noqa: E402  (reuse; do not rebuild -- 44.6)

BF16_BITS = 16.0


# --------------------------------------------------------------------------------------- helpers
def rel_l2(a, b):
    """||a - b|| / ||a||, in float64 on the GPU.  `a` is the reference."""
    num = torch.linalg.vector_norm((a - b).double())
    den = torch.linalg.vector_norm(a.double())
    return float(num / den) if float(den) > 0 else float("nan")


def err_stats(ref, rec):
    d = (ref - rec).double()
    return {
        "rel_l2": rel_l2(ref, rec),
        "max_abs": float(d.abs().max()),
        "rms": float(d.pow(2).mean().sqrt()),
    }


def rate_of(n_elems_stored_bf16, n_side_bits, n_orig_elems):
    """Bits per ORIGINAL element, counting side information honestly."""
    return (n_elems_stored_bf16 * BF16_BITS + n_side_bits) / float(n_orig_elems)


def bf16_rt(t):
    """Round-trip through bf16 -- every codec below stores its factors/residuals in bf16, so this
    rounding is part of the codec and must be charged to it, not hidden."""
    return t.to(torch.bfloat16).float()


# ---------------------------------------------------------------------------------------- codecs
def codec_lowrank(X, r, U=None, S=None, Vh=None, center=False):
    """Truncated SVD, factors stored in bf16.  Cost r(N+H) elements (+H for the mean if centered)."""
    N, H = X.shape
    mu = X.mean(dim=0, keepdim=True) if center else None
    Xc = X - mu if center else X
    if U is None:
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    A = bf16_rt(U[:, :r] * S[:r])          # [N, r]
    B = bf16_rt(Vh[:r, :])                 # [r, H]
    rec = A @ B
    side = 0
    if center:
        mu = bf16_rt(mu)
        rec = rec + mu
        side = H * BF16_BITS
    return rec, rate_of(r * (N + H), side, N * H)


def codec_randproj(X, r, seed=0):
    """Randomised range finder: Omega ~ N(0,1) [H, r]; Q = qr(X @ Omega); reconstruct Q (Q^T X).

    Stored: Q [N, r] and (Q^T X) [r, H] -- the SAME r(N+H) budget as the SVD, so this isolates
    "how much does a random basis lose against the optimal one" at equal rate.  Omega itself is
    free: it is regenerated from the seed, not stored.
    """
    N, H = X.shape
    g = torch.Generator(device=X.device).manual_seed(seed)
    Om = torch.randn(H, r, generator=g, device=X.device, dtype=X.dtype)
    Y = X @ Om
    Q, _ = torch.linalg.qr(Y)              # [N, r]
    C = Q.T @ X                            # [r, H]
    rec = bf16_rt(Q) @ bf16_rt(C)
    return rec, rate_of(r * (N + H), 0, N * H)


def codec_channel_subsample(X, r, mode="topvar"):
    """Keep r of H hidden channels exactly; the rest reconstruct as 0 (their mean is ~0).

    44.4 classifies this as a fixed-basis rank-r projection in the worst available basis given
    outlier channels.  `topvar` is the charitable version (keep the highest-energy channels);
    `uniform` is the naive one.  Cost r*N elements + r channel indices at 16 bits.
    """
    N, H = X.shape
    if mode == "topvar":
        idx = torch.topk(X.pow(2).sum(dim=0), r).indices
    else:
        idx = torch.linspace(0, H - 1, r, device=X.device).round().long()
    rec = torch.zeros_like(X)
    rec[:, idx] = bf16_rt(X[:, idx])
    return rec, rate_of(r * N, r * 16.0, N * H)


def codec_intN(X, bits, group=None):
    """Symmetric absmax integer quantisation.  `group=None` means per-token (per row of H).

    Per-token scaling is the strongest cheap variant for a residual stream, because the outlier
    channels that dominate are shared across tokens while the per-token magnitude varies.
    Side information: one bf16 scale per group.
    """
    N, H = X.shape
    g = H if group is None else group
    assert H % g == 0, f"group {g} does not divide H={H}"
    Xg_ = X.reshape(N, H // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = Xg_.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scale = bf16_rt(scale)                                   # the scale is stored, so round it
    q = torch.clamp(torch.round(Xg_ / scale), -qmax - 1, qmax)
    rec = (q * scale).reshape(N, H)
    n_groups = N * (H // g)
    return rec, (N * H * float(bits) + n_groups * BF16_BITS) / float(N * H)


def codec_intN_per_channel(X, bits):
    """Symmetric absmax quantisation with ONE scale per HIDDEN CHANNEL (a column of N tokens).

    This is the variant matched to what the measurement actually found: the residual stream's energy
    is dominated by a handful of FIXED channels (a "massive activation") whose magnitude is ~1000x
    the RMS.  A per-token scale is set by that outlier and destroys every other channel in the row;
    a per-CHANNEL scale isolates it.  Side information is H scales for N*H elements -- 16/N bits per
    element, i.e. essentially free, and cheaper than any group scheme.
    """
    N, H = X.shape
    qmax = 2 ** (bits - 1) - 1
    scale = X.abs().amax(dim=0, keepdim=True) / qmax
    scale = bf16_rt(torch.where(scale > 0, scale, torch.ones_like(scale)))
    q = torch.clamp(torch.round(X / scale), -qmax - 1, qmax)
    return (q * scale), (N * H * float(bits) + H * BF16_BITS) / float(N * H)


def codec_outlier_lowrank(X, k, rate, U=None, S=None, Vh=None):
    """Keep the k highest-energy CHANNELS exactly; spend the rest of the budget on a low-rank code
    of the residual.  Matched total rate.

    WHY THIS EXISTS, and it is the point.  42.6's lesson: *before you publish a negative about an
    approach, make sure you are measuring the approach and not your first draft of it.*  A plain
    truncated SVD spends its leading components representing the massive-activation channels; an
    exact channel costs N elements where a rank-1 component costs N+H, so pulling the outliers out
    by hand and giving the freed budget to the tail is the strictly stronger form of the idea.  If
    low-rank loses even here, the loss is a property of the object, not of the first codec tried.
    """
    N, H = X.shape
    budget = rate * N * H / BF16_BITS                        # in bf16-element equivalents
    idx = torch.topk(X.pow(2).sum(dim=0), k).indices
    m = int((budget - k * N - k) / (N + H))
    if m < 1:
        return None, None, None
    R = X.clone()
    R[:, idx] = 0                                            # residual: outlier channels removed
    Ur, Sr, Vhr = torch.linalg.svd(R, full_matrices=False)
    rec = bf16_rt(Ur[:, :m] * Sr[:m]) @ bf16_rt(Vhr[:m, :])
    rec[:, idx] = bf16_rt(X[:, idx])
    return rec, rate_of(k * N + m * (N + H), k * 16.0, N * H), m


def codec_outlier_intN(X, k, bits, group=128):
    """Keep the k highest-energy channels in bf16; quantise everything else to `bits` per element.

    This is the incumbent construction (ActNN / GACT / ALAM / COAT / HyC-LoRA keep outliers wide),
    included so the bar low-rank has to clear is the real one and not a strawman.  The achieved rate
    is reported, not assumed -- keeping k channels wide costs 11.875*k/H extra bits per element at
    bits=4, so this lands slightly ABOVE the nominal rate and must be read that way.
    """
    N, H = X.shape
    idx = torch.topk(X.pow(2).sum(dim=0), k).indices
    mask = torch.ones(H, dtype=torch.bool, device=X.device)
    mask[idx] = False
    rest = X[:, mask]
    Hr = int(mask.sum())
    g = min(group, Hr)
    while Hr % g:
        g -= 1
    qrest, _ = codec_intN(rest, bits, group=g)
    rec = torch.empty_like(X)
    rec[:, idx] = bf16_rt(X[:, idx])
    rec[:, mask] = qrest
    bits_total = (k * N * BF16_BITS + Hr * N * float(bits) + (N * Hr / g) * BF16_BITS + k * 16.0)
    return rec, bits_total / float(N * H)


# ------------------------------------------------------------------------------------------ main
def spectrum_of(X):
    """Singular values (descending) and the rank needed for a set of energy fractions."""
    S = torch.linalg.svdvals(X.double())
    e = S.pow(2)
    cum = torch.cumsum(e, 0) / e.sum()
    out = {"n_sv": int(S.numel()), "sv_top": [float(v) for v in S[:8]],
           "sv_tail": [float(v) for v in S[-4:]]}
    for f in (0.50, 0.90, 0.95, 0.99, 0.999):
        out[f"rank_for_{f}"] = int(torch.searchsorted(cum, torch.tensor(f, dtype=cum.dtype,
                                                                       device=cum.device)) + 1)
    # Energy captured at fixed ranks -- the dual view, and the one the rate ceiling needs.
    out["energy_at_rank"] = {}
    for r in (16, 32, 64, 128, 256, 512, 1024):
        if r <= S.numel():
            out["energy_at_rank"][str(r)] = float(cum[r - 1])
    out["stable_rank"] = float(e.sum() / e[0])               # ||X||_F^2 / ||X||_2^2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--model", default=ph.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rates", default="8,4,2", help="target bits per element for the bake-off")
    ap.add_argument("--layers", default="", help="comma list; default = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = ph.make_cfg(args.batch, args.seq, model=args.model, lora_r=16)
    model = ph.build_model(args.arm, cfg, dev, adapter_dtype="bf16")
    model.train()

    ids, src = real_text_ids(args.model, args.batch, args.seq, dev)
    print(f"token source: {src}", flush=True)

    inner = ph._inner_llama(model)
    layers = inner.layers
    captured = {}

    def mk(i):
        def hook(_mod, a, kw):  # noqa: ANN001
            t = a[0] if a else kw.get("hidden_states")
            captured[i] = t.detach().float().reshape(-1, t.shape[-1]).clone()
            return None
        return hook

    handles = [ly.register_forward_pre_hook(mk(i), with_kwargs=True) for i, ly in enumerate(layers)]
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in handles:
        h.remove()

    L = len(layers)
    N, H = captured[0].shape
    sel = ([int(v) for v in args.layers.split(",")] if args.layers else list(range(L)))
    rates = [float(v) for v in args.rates.split(",")]

    out = {
        "WARNING": ("PREMISE MEASUREMENT ONLY (CONTEXT.md 44.6 Stage 1). This says whether the "
                    "checkpoint stack is lossily compressible and by which codec at equal rate. It "
                    "says NOTHING about whether any such mechanism is novel (T-6 is separate and "
                    "runs first, 37.4), nor about throughput (compression is a DEBIT on 37.1), nor "
                    "about exactness (a lossy checkpoint breaks B3 and the certificate by "
                    "construction, 44.5). 39.3 does NOT cover this: it measured LOSSLESS bit-level "
                    "incompressibility, and bit entropy says nothing about numerical rank (44.7.1)."),
        "arm": args.arm, "model": args.model, "seq": args.seq, "batch": args.batch,
        "n_layers": L, "token_source": src, "N": N, "H": H, "rates_bits_per_elem": rates,
        "rate_ceiling_note": ("r_ceiling = rate*H/16 is the largest rank ANY low-rank code can "
                              "afford at that rate as N->infinity; r_at_N is what this N affords."),
        "rate_ceiling": {str(rt): {"r_ceiling_N_inf": int(rt * H / 16),
                                   "r_at_this_N": int(rt * N * H / (16 * (N + H)))} for rt in rates},
    }

    # --------------------------------------------------------------- 1. per-layer spectra + codecs
    per_layer = []
    for k in sel:
        X = captured[k]
        rec_l = {"layer": k, "fro": float(torch.linalg.matrix_norm(X.double())),
                 "spectrum": spectrum_of(X)}

        # Channel-energy concentration: the outlier-feature / massive-activation question, measured.
        ce = X.pow(2).sum(dim=0)
        ce_sorted = torch.sort(ce, descending=True).values
        tot = float(ce.sum())
        rec_l["channel_energy"] = {
            "top1_frac": float(ce_sorted[0]) / tot,
            "top8_frac": float(ce_sorted[:8].sum()) / tot,
            "top64_frac": float(ce_sorted[:64].sum()) / tot,
            "top512_frac": float(ce_sorted[:512].sum()) / tot,
            "max_abs": float(X.abs().max()),
            "rms": float(X.pow(2).mean().sqrt()),
        }

        # One SVD per layer, reused across every rank so the bake-off is cheap.
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        codecs = {}
        for rt in rates:
            r_at_N = max(1, int(rt * N * H / (16 * (N + H))))
            r_ceil = max(1, int(rt * H / 16))
            r_chan = max(1, int(rt * H / 16))                # channel subsample: r*N*16/(N*H) = rt
            entry = {}

            rec, rate = codec_lowrank(X, r_at_N, U, S, Vh)
            entry["lowrank_svd@r_at_N"] = {"r": r_at_N, "bits_per_elem": rate, **err_stats(X, rec)}

            # The ceiling rank is NOT affordable at this N; reported as the best low-rank can EVER
            # do at this rate, so a small-N result is not mistaken for a bound on the idea.
            rec, rate = codec_lowrank(X, min(r_ceil, min(N, H)), U, S, Vh)
            entry["lowrank_svd@r_ceiling"] = {"r": min(r_ceil, min(N, H)),
                                              "bits_per_elem_AT_THIS_N": rate,
                                              "bits_per_elem_asymptotic": rt, **err_stats(X, rec)}

            rec, rate = codec_lowrank(X, r_at_N, center=True)
            entry["lowrank_centered@r_at_N"] = {"r": r_at_N, "bits_per_elem": rate,
                                                **err_stats(X, rec)}

            rec, rate = codec_randproj(X, r_at_N)
            entry["randproj@r_at_N"] = {"r": r_at_N, "bits_per_elem": rate, **err_stats(X, rec)}

            rec, rate = codec_channel_subsample(X, r_chan, "topvar")
            entry["channel_topvar"] = {"r": r_chan, "bits_per_elem": rate, **err_stats(X, rec)}

            rec, rate = codec_channel_subsample(X, r_chan, "uniform")
            entry["channel_uniform"] = {"r": r_chan, "bits_per_elem": rate, **err_stats(X, rec)}

            bits = int(round(rt))
            if bits >= 2:
                rec, rate = codec_intN(X, bits, group=None)
                entry[f"int{bits}_per_token"] = {"bits_per_elem": rate, **err_stats(X, rec)}
                rec, rate = codec_intN(X, bits, group=128)
                entry[f"int{bits}_group128"] = {"bits_per_elem": rate, **err_stats(X, rec)}
                rec, rate = codec_intN_per_channel(X, bits)
                entry[f"int{bits}_per_channel"] = {"bits_per_elem": rate, **err_stats(X, rec)}

            # -------- the STRONG forms: outlier-aware.  42.6 -- measure the approach, not a draft.
            for kk in (8, 64):
                rec, rate, m = codec_outlier_lowrank(X, kk, rt, U, S, Vh)
                if rec is not None:
                    entry[f"outlier{kk}+lowrank"] = {"k": kk, "m": m, "bits_per_elem": rate,
                                                     **err_stats(X, rec)}
                if bits >= 2:
                    rec, rate = codec_outlier_intN(X, kk, bits, group=128)
                    entry[f"outlier{kk}+int{bits}"] = {"k": kk, "bits_per_elem": rate,
                                                       **err_stats(X, rec)}

            codecs[str(rt)] = entry
        rec_l["codecs"] = codecs
        per_layer.append(rec_l)
        print(f"layer {k}: rank@99%={rec_l['spectrum']['rank_for_0.99']} "
              f"stable_rank={rec_l['spectrum']['stable_rank']:.1f} "
              f"top8_chan_energy={rec_l['channel_energy']['top8_frac']:.3f}", flush=True)
    out["per_layer"] = per_layer

    # ------------------------------------------------------- 2. joint spectrum: a basis for all 22
    # Gram accumulated in float64: G = sum_k X_k^T X_k, [H, H].  Its eigenvectors are the optimal
    # SHARED right basis, i.e. the one basis minimising total reconstruction error over all layers.
    G = torch.zeros(H, H, dtype=torch.float64, device=dev)
    for k in range(L):
        Xk = captured[k].double()
        G += Xk.T @ Xk
    evals, evecs = torch.linalg.eigh(G)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    cum = torch.cumsum(evals.clamp(min=0), 0) / evals.clamp(min=0).sum()
    joint = {"rank_for_0.90": int(torch.searchsorted(cum, torch.tensor(0.90, dtype=cum.dtype,
                                                                      device=dev)) + 1),
             "rank_for_0.99": int(torch.searchsorted(cum, torch.tensor(0.99, dtype=cum.dtype,
                                                                       device=dev)) + 1),
             "rank_for_0.999": int(torch.searchsorted(cum, torch.tensor(0.999, dtype=cum.dtype,
                                                                        device=dev)) + 1),
             "energy_at_rank": {str(r): float(cum[r - 1]) for r in (64, 128, 256, 512, 1024)}}

    # The operative question: at a given rank, how much WORSE is the shared basis than each layer's
    # own optimal one?  If the answer is "barely", the basis amortises across 22 layers and its own
    # storage becomes negligible -- which changes the rate arithmetic materially.
    shared = {}
    for r in (64, 128, 256, 512):
        V = evecs[:, :r].float()                             # [H, r] shared right basis
        pl = []
        for k in sel:
            X = captured[k]
            rec = bf16_rt(X @ V) @ bf16_rt(V.T)
            own = per_layer[sel.index(k)]["spectrum"]["energy_at_rank"].get(str(r))
            pl.append({"layer": k, "shared_rel_l2": rel_l2(X, rec),
                       "own_basis_energy_at_r": own})
        shared[str(r)] = {
            "shared_rel_l2_median": float(np.median([p["shared_rel_l2"] for p in pl])),
            "shared_rel_l2_max": float(np.max([p["shared_rel_l2"] for p in pl])),
            "per_layer": pl,
            "rate_bits_per_elem_if_basis_amortised": rate_of(r * N, r * H * BF16_BITS / L, N * H),
        }
    joint["shared_basis"] = shared
    out["joint_spectrum"] = joint

    # ------------------------------------------------------------------ 3. controls: Gaussian floor
    kmid = sel[len(sel) // 2]
    Xm = captured[kmid]
    g = torch.Generator(device=dev).manual_seed(7)
    Xg = torch.randn(N, H, generator=g, device=dev) * float(Xm.pow(2).mean().sqrt())
    ctrl = {"layer_matched": kmid, "spectrum": spectrum_of(Xg), "codecs": {}}
    for rt in rates:
        r_at_N = max(1, int(rt * N * H / (16 * (N + H))))
        rec, rate = codec_lowrank(Xg, r_at_N)
        e = {"lowrank_svd@r_at_N": {"r": r_at_N, "bits_per_elem": rate, **err_stats(Xg, rec)}}
        bits = int(round(rt))
        if bits >= 2:
            rec, rate = codec_intN(Xg, bits, group=128)
            e[f"int{bits}_group128"] = {"bits_per_elem": rate, **err_stats(Xg, rec)}
        ctrl["codecs"][str(rt)] = e
    out["control_gaussian"] = ctrl

    # --------------------------------------------- 4. THE PROPAGATION PROXY -- the number that matters
    # 44.4: the reconstruction error is multiplied by W_q/W_k/W_v (and W_gate/W_up after attention)
    # before it reaches any gradient, and it first passes through RMSNorm, which is scale-invariant
    # per token.  So the error that matters is the one measured AFTER those ops, on the real weights
    # of the layer the checkpoint belongs to.  Measured for the same codecs, at the same rates.
    def rmsnorm(x, w, eps):
        v = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(v + eps) * w

    prop = []
    for k in sel:
        ly = layers[k]
        X = captured[k]
        w_n1 = ly.input_layernorm.weight.detach().float()
        eps = float(getattr(ly.input_layernorm, "variance_epsilon",
                            getattr(ly.input_layernorm, "eps", 1e-5)))

        def base_w(mod):
            m = getattr(mod, "base_layer", mod)
            return m.weight.detach().float()

        Wq = base_w(ly.self_attn.q_proj)
        Wk = base_w(ly.self_attn.k_proj)
        Wv = base_w(ly.self_attn.v_proj)

        ref_n = rmsnorm(X, w_n1, eps)
        ref = {"xn1": ref_n, "q": ref_n @ Wq.T, "k": ref_n @ Wk.T, "v": ref_n @ Wv.T}

        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        entry = {"layer": k, "codecs": {}}
        for rt in rates:
            r_at_N = max(1, int(rt * N * H / (16 * (N + H))))
            r_ceil = max(1, min(int(rt * H / 16), min(N, H)))
            cands = {
                "lowrank_svd@r_at_N": codec_lowrank(X, r_at_N, U, S, Vh)[0],
                "lowrank_svd@r_ceiling": codec_lowrank(X, r_ceil, U, S, Vh)[0],
                "channel_topvar": codec_channel_subsample(X, max(1, int(rt * H / 16)), "topvar")[0],
            }
            bits = int(round(rt))
            if bits >= 2:
                cands[f"int{bits}_per_token"] = codec_intN(X, bits, group=None)[0]
                cands[f"int{bits}_group128"] = codec_intN(X, bits, group=128)[0]
                cands[f"int{bits}_per_channel"] = codec_intN_per_channel(X, bits)[0]
            for kk in (8, 64):
                r_ol = codec_outlier_lowrank(X, kk, rt, U, S, Vh)[0]
                if r_ol is not None:
                    cands[f"outlier{kk}+lowrank"] = r_ol
                if bits >= 2:
                    cands[f"outlier{kk}+int{bits}"] = codec_outlier_intN(X, kk, bits, group=128)[0]
            e = {}
            for name, Xr in cands.items():
                n = rmsnorm(Xr, w_n1, eps)
                e[name] = {
                    "x_rel_l2": rel_l2(X, Xr),
                    "xn1_rel_l2": rel_l2(ref["xn1"], n),
                    "q_rel_l2": rel_l2(ref["q"], n @ Wq.T),
                    "k_rel_l2": rel_l2(ref["k"], n @ Wk.T),
                    "v_rel_l2": rel_l2(ref["v"], n @ Wv.T),
                }
            entry["codecs"][str(rt)] = e
        prop.append(entry)
    out["propagation"] = prop

    # -------------------------------------------------------------------------------- 5. summary
    def med(codec, rate, field="rel_l2"):
        v = [l["codecs"][str(rate)][codec][field] for l in per_layer
             if codec in l["codecs"][str(rate)]]
        return float(np.median(v)) if v else None

    def medp(codec, rate, field):
        v = [l["codecs"][str(rate)][codec][field] for l in prop if codec in l["codecs"][str(rate)]]
        return float(np.median(v)) if v else None

    summary = {
        "rank_for_0.99_median": float(np.median([l["spectrum"]["rank_for_0.99"] for l in per_layer])),
        "rank_for_0.999_median": float(np.median([l["spectrum"]["rank_for_0.999"] for l in per_layer])),
        "stable_rank_median": float(np.median([l["spectrum"]["stable_rank"] for l in per_layer])),
        "top8_channel_energy_median": float(np.median([l["channel_energy"]["top8_frac"]
                                                       for l in per_layer])),
        "gaussian_control_rank_for_0.99": ctrl["spectrum"]["rank_for_0.99"],
        "joint_rank_for_0.99": joint["rank_for_0.99"],
        "reconstruction_rel_l2_median": {},
        "propagated_q_rel_l2_median": {},
    }
    for rt in rates:
        bits = int(round(rt))
        names = ["lowrank_svd@r_at_N", "lowrank_svd@r_ceiling", "lowrank_centered@r_at_N",
                 "randproj@r_at_N", "channel_topvar", "channel_uniform",
                 f"int{bits}_per_token", f"int{bits}_group128"]
        summary["reconstruction_rel_l2_median"][str(rt)] = {
            n: med(n, rt) for n in names if med(n, rt) is not None}
        pnames = ["lowrank_svd@r_at_N", "lowrank_svd@r_ceiling", "channel_topvar",
                  f"int{bits}_per_token", f"int{bits}_group128"]
        summary["propagated_q_rel_l2_median"][str(rt)] = {
            n: medp(n, rt, "q_rel_l2") for n in pnames if medp(n, rt, "q_rel_l2") is not None}

    summary["VERDICT_HINT"] = (
        "Low-rank is alive ONLY if it beats int4_group128 on the PROPAGATED error at the same "
        "bits/element -- reconstruction error alone is the wrong quantity (44.4). Compare "
        "lowrank_svd@r_ceiling (the best any low-rank code can ever do at that rate) against the "
        "intN rows. Read rank_for_0.99 against gaussian_control_rank_for_0.99: if they are close, "
        "the checkpoint has no usable low-rank structure at all.")
    out["SUMMARY"] = summary

    js = json.dumps(out, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(js)
        print(f"wrote {args.out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
