#!/usr/bin/env python
"""Exact-value redundancy census of the block's CHECKPOINTED state.  CONTEXT.md §37 mandate.

WHY THIS EXISTS.  Every mechanism this project has considered -- across §§30-38 and three prior-art
sweeps -- attacks the SCHEDULE: when work runs (overlap), where it runs (offload), or how many
kernels it runs in (fusion).  §38.2's map closes all three, every one with a named owner.  Nobody has
ever looked at the BITS.

At `keep='min'` the fused block stores exactly one tensor per layer: `x`, the decoder layer's input.
So the entire checkpoint IS the residual stream, sampled 22 times.  And the residual stream is
built by accumulation -- `x_{k+1} = x_k + attn_out_k + ffn_out_k` -- so consecutive snapshots are
*a priori* correlated.  If that correlation is large, it is an EXACT structure: XOR on the bf16 bit
patterns is exactly invertible, so exploiting it costs no accuracy at all and is invisible to
T-3/T-5.  If it is small, the direction is dead and this probe kills it in twenty minutes.

**This probe measures the PREMISE ONLY.  It is not a mechanism and it does not assume one.**
§38.6's first recorded lesson is that a physical premise was asserted from a spec sheet and a plan
was allowed to grow on it before it was measured; the premise turned out to be false and the probe
that killed it cost one agent-hour.  This is that probe, run first.

WHAT IT MEASURES, per consecutive pair (x_k, x_{k+1}) and per tensor alone:
  * share of elements whose sign+exponent (top 9 bf16 bits) agree -- the structural precondition
  * the full leading-zero histogram of the 16-bit XOR
  * `blockpack` bits/element: a REALISTIC exact scheme -- per 32-element block store a 4-bit width
    header plus 32 fixed-width residuals, which decodes branch-free and in parallel on a GPU.
    Quoted against the 16 bits/element the checkpoint costs today.
  * the zeroth-order entropy bound, as the floor no such scheme can beat
  * zlib -9 on the raw bytes and on the XOR bytes, as an independent check that is not my own
    arithmetic
  * ||delta|| / ||x||, so the bit result can be read against the numerical one

CONTROLS, because a redundancy number with no control is unreadable:
  * the SAME statistics for a SHUFFLED pairing (x_k against a random other layer) -- if adjacency
    carries no information, the shuffled control matches the real pairing and the premise is dead
  * the same statistics for `x_k` alone (no delta), so we can see whether the delta earns its keep
  * an i.i.d. Gaussian tensor of the same shape and scale, as the "no structure at all" floor

REAL TEXT, NOT RANDOM IDS.  `profile_hyclora.make_batch` draws token ids uniformly from the
vocabulary.  A residual stream driven by uniform-random tokens is not the object we are trying to
characterise, and a premise measured on it would be worthless.  This reads WikiText-2 from the local
cache (`HF_HOME=./data`) and falls back to random ids only with a loud flag in the JSON.

Usage:
    PYTHONPATH=src python src/probe_value_redundancy.py --seq 1024 --batch 2 \
        --out results/recon/value_redundancy_seq1024.json
"""
import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import profile_hyclora as ph  # noqa: E402


def real_text_ids(model_name, batch, seq, device):
    """WikiText-2 from the local cache, concatenated and chunked.  Returns (ids, source_str)."""
    try:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")  # mandatory offline (CONTEXT.md §33.12)
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        need = batch * seq + 64
        buf, i = [], 0
        while len(buf) < need and i < len(ds):
            t = ds[i]["text"]
            if t.strip():
                buf.extend(tok(t, add_special_tokens=False)["input_ids"])
            i += 1
        if len(buf) < need:
            raise RuntimeError(f"only {len(buf)} tokens available, need {need}")
        ids = torch.tensor(buf[: batch * seq], dtype=torch.long).view(batch, seq).to(device)
        return ids, "wikitext-2-raw-v1/train"
    except Exception as e:  # noqa: BLE001
        print(f"!! REAL TEXT UNAVAILABLE ({type(e).__name__}: {e}) -- falling back to random ids",
              flush=True)
        g = torch.Generator(device="cpu").manual_seed(41)
        from transformers import AutoConfig
        v = AutoConfig.from_pretrained(model_name).vocab_size
        return torch.randint(0, v, (batch, seq), generator=g).to(device), "RANDOM_IDS_FALLBACK"


def bits_stats(u16, block=32):
    """Exact-coding statistics for a uint16 array (a bf16 bit pattern, or an XOR of two).

    Returns leading-zero histogram, the zeroth-order entropy bound, and the bits/element of a
    realistic per-block fixed-width packing (4-bit width header + `block` residuals).
    """
    u16 = np.ascontiguousarray(u16.ravel())
    n = u16.size
    # Leading zeros of a 16-bit word: 16 for zero, else 15 - floor(log2(v)).
    nlz = np.full(n, 16, dtype=np.int32)
    nz = u16 != 0
    if nz.any():
        # `np.log2` on uint16 is exact for these magnitudes; use bit_length via searchsorted-free math
        nlz[nz] = 15 - np.floor(np.log2(u16[nz].astype(np.float64))).astype(np.int32)
    hist = np.bincount(nlz, minlength=17)[:17]

    # Zeroth-order entropy of the 16-bit symbol distribution -- the floor for any memoryless code.
    cnt = np.bincount(u16, minlength=65536).astype(np.float64)
    p = cnt[cnt > 0] / n
    entropy_bits = float(-(p * np.log2(p)).sum())

    # Realistic exact scheme: per block of `block`, width = 16 - min(nlz), plus a 4-bit header.
    nb = (n + block - 1) // block
    pad = nb * block - n
    nlz_p = np.concatenate([nlz, np.full(pad, 16, dtype=np.int32)]) if pad else nlz
    w = 16 - nlz_p.reshape(nb, block).min(axis=1)
    w = np.maximum(w, 0)
    blockpack_bits = float((w.astype(np.float64).sum() * block + 4 * nb) / n)

    return {
        "n": int(n),
        "nlz_hist": [int(x) for x in hist],
        "mean_nlz": float(nlz.mean()),
        "entropy_bits_per_elem": entropy_bits,
        "blockpack_bits_per_elem": blockpack_bits,
        "blockpack_ratio_vs_bf16": 16.0 / blockpack_bits if blockpack_bits > 0 else float("inf"),
    }


def zlib_bits_per_elem(u16, cap_elems=4_000_000):
    """Independent check that is not my own arithmetic.  Subsampled contiguously for runtime."""
    a = np.ascontiguousarray(u16.ravel()[:cap_elems])
    return len(zlib.compress(a.tobytes(), 9)) * 8.0 / a.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--arm", default="fb_min_fnorm_sdpa")
    ap.add_argument("--model", default=ph.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-elems", type=int, default=4_000_000,
                    help="elements per tensor used for the bit statistics (0 = all)")
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
            captured[i] = t.detach().to("cpu", copy=True)
            return None
        return hook

    handles = [ly.register_forward_pre_hook(mk(i), with_kwargs=True) for i, ly in enumerate(layers)]
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for h in handles:
        h.remove()

    L = len(layers)
    xs = [captured[i] for i in range(L)]
    assert all(t.dtype == torch.bfloat16 for t in xs), "expected bf16 residual stream"

    def u16_of(t):
        a = t.reshape(-1).view(torch.int16).numpy().astype(np.uint16)
        return a[: args.sample_elems] if args.sample_elems else a

    us = [u16_of(t) for t in xs]
    fl = [t.reshape(-1).float().numpy()[: args.sample_elems or None] for t in xs]

    out = {
        "WARNING": ("This is a PREMISE measurement, not a mechanism. It says whether exact "
                    "inter-layer redundancy exists in the checkpointed residual stream. It does "
                    "NOT say any mechanism exploiting it is novel, useful, or net-positive on "
                    "throughput -- see CONTEXT.md 37.4 (T-6 runs before kernels)."),
        "arm": args.arm, "model": args.model, "seq": args.seq, "batch": args.batch,
        "n_layers": L, "token_source": src, "sample_elems": args.sample_elems,
        "dtype": "bfloat16", "blockpack_block": 32,
    }

    # ---- alone: how compressible is a single checkpoint, with no delta at all? ----
    alone = []
    for k in (0, L // 2, L - 1):
        s = bits_stats(us[k])
        s.update({"layer": k, "zlib_bits_per_elem": zlib_bits_per_elem(us[k])})
        alone.append(s)
    out["alone"] = alone

    # ---- control: i.i.d. Gaussian at the same scale = the "no structure" floor ----
    g = torch.randn(min(args.sample_elems or us[0].size, us[0].size),
                    generator=torch.Generator().manual_seed(7)) * float(xs[L // 2].float().std())
    gu = g.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
    gs = bits_stats(gu)
    gs["zlib_bits_per_elem"] = zlib_bits_per_elem(gu)
    out["control_gaussian"] = gs

    # ---- the real question: consecutive-layer XOR deltas ----
    pairs = []
    for k in range(L - 1):
        d = us[k] ^ us[k + 1]
        se = float(np.mean((us[k] >> 7) == (us[k + 1] >> 7)))  # sign+exponent agree
        s = bits_stats(d)
        num = float(np.linalg.norm(fl[k + 1] - fl[k]))
        den = float(np.linalg.norm(fl[k])) or 1.0
        s.update({
            "pair": f"x{k}^x{k+1}",
            "sign_exp_agree_frac": se,
            "rel_delta_l2": num / den,
            "zlib_bits_per_elem": zlib_bits_per_elem(d),
        })
        pairs.append(s)
    out["consecutive_xor"] = pairs

    # ---- control: SHUFFLED pairing. If adjacency carries no information this matches. ----
    rng = np.random.default_rng(11)
    perm = rng.permutation(L)
    shuf = []
    for k in range(L - 1):
        j = int(perm[k])
        if j == k:
            j = int(perm[(k + 1) % L])
        d = us[k] ^ us[j]
        s = bits_stats(d)
        s.update({"pair": f"x{k}^x{j}(shuffled)",
                  "sign_exp_agree_frac": float(np.mean((us[k] >> 7) == (us[j] >> 7))),
                  "zlib_bits_per_elem": zlib_bits_per_elem(d)})
        shuf.append(s)
    out["shuffled_xor_control"] = shuf

    # ---- duplicate rows: identical token ids give identical layer-0 inputs (embedding lookup) ----
    x0 = xs[0].reshape(-1, xs[0].shape[-1])
    uniq = np.unique(x0.view(torch.int16).numpy(), axis=0).shape[0]
    out["layer0_duplicate_rows"] = {
        "rows": int(x0.shape[0]), "unique_rows": int(uniq),
        "duplicate_frac": 1.0 - uniq / float(x0.shape[0]),
        "note": ("Layer 0's input is a pure embedding lookup, so repeated token ids give bitwise "
                 "identical rows. RoPE is positional and applied inside attention, so this "
                 "degenerates immediately at layer 1 -- reported to bound the idea, not to sell it."),
    }

    def summ(rows, key="blockpack_bits_per_elem"):
        v = [r[key] for r in rows]
        return {"min": min(v), "median": float(np.median(v)), "max": max(v), "mean": float(np.mean(v))}

    out["SUMMARY"] = {
        "alone_blockpack_bits": summ(out["alone"]),
        "consecutive_xor_blockpack_bits": summ(pairs),
        "shuffled_xor_blockpack_bits": summ(shuf),
        "gaussian_blockpack_bits": gs["blockpack_bits_per_elem"],
        "consecutive_xor_zlib_bits": summ(pairs, "zlib_bits_per_elem"),
        "shuffled_xor_zlib_bits": summ(shuf, "zlib_bits_per_elem"),
        "alone_zlib_bits": summ(out["alone"], "zlib_bits_per_elem"),
        "sign_exp_agree_frac_median": float(np.median([p["sign_exp_agree_frac"] for p in pairs])),
        "rel_delta_l2_median": float(np.median([p["rel_delta_l2"] for p in pairs])),
        "VERDICT_HINT": ("Compare consecutive_xor against BOTH shuffled_xor (does adjacency matter?) "
                         "and alone (does the delta beat just coding the tensor?). The premise is "
                         "alive only if consecutive beats both by a wide margin."),
    }

    js = json.dumps(out, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(js)
        print(f"wrote {args.out}", flush=True)
    print(json.dumps(out["SUMMARY"], indent=2), flush=True)


if __name__ == "__main__":
    main()
