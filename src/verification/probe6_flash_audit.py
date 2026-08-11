"""Probe 6 -- fairness audit of the HyC-LoRA FlashAttention port (`intra_inter_flash`).

Question: is `hyclora_flash_q2` as we run it a faithful implementation of HyC-LoRA's FA variant,
or is our port broken?  Two anomalies motivate it: (a) +3.61 WikiText-2 ppl where their *eager*
implementation costs +0.043 at the same bit width, (b) a 1.8e+10 adapter-gradient rel-L2 in the
matched-CE table.

The controlling comparison is EAGER-vs-FLASH at identical settings.  The FFN half of the two
fused layers is byte-identical (verified by source diff against upstream), so any difference in
the FFN adapter gradients is noise and any difference in the ATTENTION adapter gradients is
attributable to the attention block alone.

Modes
-----
grads   : whole-model adapter-gradient fidelity vs an exact reference, on real WikiText-2 tokens,
          with PARTIALLY-TRAINED adapters (not N(0,0.02) noise -- that was the caveat on the
          1.8e+10 number).  Split by attention-adapter group vs FFN-adapter group.
calib   : dump the frozen quantiser statistics from every layer of a calibrated model and check
          they are finite, non-degenerate, and identical in construction between eager and FA.
famech  : standalone mechanism probe on the FlashAttention backward -- feed it quantised q/k, a
          quantised `out`, or both, and see which input drives the error.  Uses real tensors
          captured from a live layer.
"""
import argparse, gc, json, math, os, re, sys, time
import torch

sys.path.insert(0, "/workspace/CompAct/src")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEV = torch.device("cuda")
ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_KEYS = ("gate_proj", "up_proj", "down_proj")


def data(block=1024):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    t = raw.map(lambda ex: tok(ex["text"], return_attention_mask=False),
                batched=True, remove_columns=raw["train"].column_names, desc="tok")

    def grp(ex):
        cc = {k: sum(ex[k], []) for k in ex}
        n = (len(cc["input_ids"]) // block) * block
        r = {k: [cc[k][i:i + block] for i in range(0, n, block)] for k in cc}
        return r
    p = t.map(grp, batched=True, desc="chunk")
    return torch.tensor(p["train"]["input_ids"], dtype=torch.long)


def build(arm, seed, seq):
    from profile_hyclora import build_model
    cfg = {"model": MODEL, "batch": 2, "seq": seq, "lora_r": 16, "q_bit": 2,
           "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
           "iteration_threshold": 5, "n_layers": 22}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return build_model(arm, cfg, DEV, adapter_dtype="bf16")


def adapter_sd(m):
    return {n: p.detach().to(torch.bfloat16).clone()
            for n, p in m.named_parameters() if p.requires_grad}


def pretrain_adapters(tr, seed, seq, micro, accum, lr):
    """A short *real* fine-tune with the uncompressed baseline, so the adapters we probe are on
    the activation distribution a calibrated codec would actually see."""
    m = build("baseline_eager", seed, seq)
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(tr.shape[0], generator=g)
    for i in range(micro):
        ids = tr[order[i * 2:(i + 1) * 2]].to(DEV)
        out = m(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
        (out.loss / accum).backward()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
    sd = adapter_sd(m)
    loss = float(out.loss)
    del m, opt, trainable
    gc.collect(); torch.cuda.empty_cache()
    return sd, loss


def grads_for(arm, sd, warm_batches, probe_batch, seed, seq):
    m = build(arm, seed, seq)
    miss = m.load_state_dict(sd, strict=False)
    assert not miss.unexpected_keys, miss.unexpected_keys[:5]
    # Warm-up: >= iteration_threshold(5) forwards so every quantiser statistic is calibrated and
    # frozen, exactly as it would be during a real fine-tune.
    for ids in warm_batches:
        out = m(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
        out.loss.backward()
        m.zero_grad(set_to_none=True)
    out = m(input_ids=probe_batch, labels=probe_batch,
            attention_mask=torch.ones_like(probe_batch))
    out.loss.backward()
    g = {n: p.grad.detach().float().clone() for n, p in m.named_parameters()
         if p.requires_grad and p.grad is not None}
    loss = float(out.loss)
    nz = sum(1 for v in g.values() if float(v.norm()) == 0.0)
    nonfinite = sum(1 for v in g.values() if not bool(torch.isfinite(v).all()))
    del m
    gc.collect(); torch.cuda.empty_cache()
    return g, loss, nz, nonfinite


def compare(ref, got):
    """rel-L2 over the concatenated gradient, plus per-tensor cosines split by block."""
    names = sorted(set(ref) & set(got))
    num = den = 0.0
    per = {}
    for n in names:
        a, b = ref[n], got[n]
        num += float(((b - a) ** 2).sum())
        den += float((a ** 2).sum())
        ca = a.flatten(); cb = b.flatten()
        cos = float(torch.dot(ca, cb) / (ca.norm() * cb.norm() + 1e-30))
        rl2 = float((cb - ca).norm() / (ca.norm() + 1e-30))
        per[n] = (cos, rl2)

    def grp(keys):
        v = [per[n] for n in names if any(k in n for k in keys)]
        if not v:
            return {}
        cos = sorted(x[0] for x in v)
        rl2 = sorted(x[1] for x in v)
        return {"n": len(v), "cos_min": cos[0], "cos_med": cos[len(cos) // 2],
                "rl2_med": rl2[len(rl2) // 2], "rl2_max": rl2[-1]}
    worst = min(names, key=lambda n: per[n][0])
    return {"rel_l2_global": math.sqrt(num / den), "n_tensors": len(names),
            "attn": grp(ATTN_KEYS), "ffn": grp(FFN_KEYS),
            "worst_tensor": worst, "worst_cos": per[worst][0],
            "worst_rl2_tensor": max(names, key=lambda n: per[n][1]),
            "worst_rl2": max(per[n][1] for n in names)}


def mode_grads(a):
    tr = data(a.seq)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(tr.shape[0], generator=g)
    print(f"pre-training adapters: {a.pretrain} micro steps on the uncompressed baseline",
          flush=True)
    t0 = time.time()
    sd, ploss = pretrain_adapters(tr, a.seed, a.seq, a.pretrain, 8, a.lr)
    print(f"  done in {time.time()-t0:.0f}s, last loss {ploss:.4f}", flush=True)

    base = a.pretrain
    warm = [tr[order[(base + i) * 2:(base + i + 1) * 2]].to(DEV) for i in range(a.warm)]
    probe = tr[order[(base + a.warm) * 2:(base + a.warm + 1) * 2]].to(DEV)

    rows = []
    ref, rloss, _, _ = grads_for("baseline_eager", sd, warm, probe, a.seed, a.seq)
    ref2, _, _, _ = grads_for("baseline_eager", sd, warm, probe, a.seed, a.seq)
    noise = compare(ref, ref2)
    print(f"reference self-noise: rel-L2 {noise['rel_l2_global']:.3e} (loss {rloss:.6f})",
          flush=True)
    rows.append({"arm": "REFERENCE_self_noise", "loss": rloss, **noise})

    for arm in a.arms.split(","):
        g_, loss, nz, nf = grads_for(arm, sd, warm, probe, a.seed, a.seq)
        r = compare(ref, g_)
        r.update({"arm": arm, "loss": loss, "n_zero_grad": nz, "n_nonfinite_grad": nf})
        rows.append(r)
        print(f"{arm:22s} loss {loss:.6f}  relL2 {r['rel_l2_global']:.4e}  "
              f"ATTN cos_med {r['attn'].get('cos_med', float('nan')):.5f} "
              f"cos_min {r['attn'].get('cos_min', float('nan')):.5f} | "
              f"FFN cos_med {r['ffn'].get('cos_med', float('nan')):.5f} "
              f"cos_min {r['ffn'].get('cos_min', float('nan')):.5f}  "
              f"worst={r['worst_tensor'].split('base_model.model.model.')[-1]} "
              f"{r['worst_cos']:.4f}  zero={nz} nonfinite={nf}", flush=True)
        del g_
        gc.collect(); torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


def mode_calib(a):
    """Dump every frozen quantiser statistic, eager vs FA, and check it is sane."""
    tr = data(a.seq)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(tr.shape[0], generator=g)
    batches = [tr[order[i * 2:(i + 1) * 2]].to(DEV) for i in range(a.warm)]
    rows = {}
    for arm in a.arms.split(","):
        m = build(arm, a.seed, a.seq)
        for ids in batches:
            out = m(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
            out.loss.backward()
            m.zero_grad(set_to_none=True)
        base = m.base_model.model.model
        stats = []
        for li, layer in enumerate(base.layers):
            f = layer._hyclora_fused
            e = {"layer": li, "iteration": f.iteration}
            for k, d in f.static_value.items():
                for sk, v in d.items():
                    if v is None:
                        e[f"{k}.{sk}"] = None
                    elif torch.is_tensor(v):
                        vf = v.float()
                        e[f"{k}.{sk}"] = {
                            "dtype": str(v.dtype), "shape": list(v.shape),
                            "min": float(vf.min()), "max": float(vf.max()),
                            "mean": float(vf.mean()),
                            "n_zero": int((vf == 0).sum()),
                            "n_nonfinite": int((~torch.isfinite(vf)).sum()),
                        }
            stats.append(e)
        rows[arm] = stats
        it = {s["iteration"] for s in stats}
        print(f"{arm}: iterations seen per layer = {it} (threshold 5)")
        for key in ("x.scale", "x_norm_1.scale", "q.scale", "k.scale", "v.scale", "o.scale",
                    "x_medium.scale", "x_norm_2.scale", "gate.scale", "up.scale",
                    "q.zero_point", "k.zero_point"):
            vals = [s[key] for s in stats if s.get(key)]
            if not vals:
                print(f"    {key:22s} ABSENT")
                continue
            print(f"    {key:22s} dtype {vals[0]['dtype']:15s} "
                  f"min {min(v['min'] for v in vals):+.5g} max {max(v['max'] for v in vals):+.5g} "
                  f"zeros {sum(v['n_zero'] for v in vals)} "
                  f"nonfinite {sum(v['n_nonfinite'] for v in vals)}")
        del m
        gc.collect(); torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


def mode_famech(a):
    """Which quantised input corrupts the FlashAttention backward?

    Captures real q/k/v/o/lse from a calibrated FA layer, then re-runs the FA backward with each
    input independently replaced by its 2-bit dequantised version, exactly as the layer does.
    """
    from hyclora.operators.compress_function import (
        compression_pack_quant_base, compression_pack_quant_zp_base,
        decompression_dequantization, decompression_dequantization_with_zero_point)
    from hyclora.layers.fused_llama_layer_intra_inter_flash import _flash_fwd, _flash_bwd
    from hyclora.compute_utils import hidden_to_head_shape, head_to_hidden_shape

    tr = data(a.seq)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(tr.shape[0], generator=g)
    ids = tr[order[:2]].to(DEV)

    # Grab real q/k/v out of a stock model's layer 0 by hooking the projections.
    m = build("baseline_sdpa", a.seed, a.seq)
    cap = {}
    lay = m.base_model.model.model.layers[a.layer]
    hs = {}
    h = lay.register_forward_pre_hook(lambda mod, args: hs.__setitem__("x", args[0]))
    with torch.no_grad():
        m(input_ids=ids, attention_mask=torch.ones_like(ids))
    h.remove()
    with torch.no_grad():
        xn = lay.input_layernorm(hs["x"])
        cap["q"] = lay.self_attn.q_proj(xn)
        cap["k"] = lay.self_attn.k_proj(xn)
        cap["v"] = lay.self_attn.v_proj(xn)
    nh = m.config.num_attention_heads
    nkv = m.config.num_key_value_heads
    hd = m.config.hidden_size // nh
    cos, sin = m.base_model.model.model.rotary_emb(
        hs["x"], torch.arange(a.seq, device=DEV).unsqueeze(0))
    cos, sin = cos[0].to(torch.bfloat16), sin[0].to(torch.bfloat16)
    del m, hs
    gc.collect(); torch.cuda.empty_cache()

    from hyclora.operators.rope_kernels import rope_forward

    def to_heads(t, n):
        return hidden_to_head_shape(t, n)

    q0 = rope_forward(to_heads(cap["q"], nh).transpose(1, 2), cos, sin).transpose(1, 2)
    k0 = rope_forward(to_heads(cap["k"], nkv).transpose(1, 2), cos, sin).transpose(1, 2)
    v0 = to_heads(cap["v"], nkv)
    scale = hd ** -0.5
    o, lse, cq, ck, mq, mk, ps, po, _ = _flash_fwd(q0, k0, v0, scale)
    o_hidden = head_to_hidden_shape(o)
    torch.manual_seed(0)
    grad_o = torch.randn_like(o) * 0.01

    def quant_zp(x, bits):
        qq, s, z = compression_pack_quant_zp_base(x.clone(), bits, 'per-channel', 99, 5,
                                                  _stat_zp(x, bits))
        return decompression_dequantization_with_zero_point(qq, s, z, bits)

    def quant(x, bits):
        qq, s = compression_pack_quant_base(x.clone(), bits, 'per-channel', 99, 5, _stat(x, bits))
        return decompression_dequantization(qq, s, bits)

    from hyclora.operators.compress_function import (get_statistics_only_quant,
                                                     get_statistics_only_quant_zero_point)

    def _stat(x, bits):
        return {"scale": get_statistics_only_quant(x, bits, 'per-channel')}

    def _stat_zp(x, bits):
        s, z = get_statistics_only_quant_zero_point(x, bits, 'per-channel')
        return {"scale": s, "zero_point": z}

    ref = _flash_bwd(grad_o, q0, k0, v0, o, lse, cq, ck, mq, mk, ps, po, scale)

    rows = []
    for bits in [int(b) for b in a.bits.split(",")]:
        qd = to_heads(quant_zp(head_to_hidden_shape(to_heads(cap["q"], nh)), bits), nh)
        qd = rope_forward(qd.transpose(1, 2), cos, sin).transpose(1, 2)
        kd = to_heads(quant_zp(head_to_hidden_shape(to_heads(cap["k"], nkv)), bits), nkv)
        kd = rope_forward(kd.transpose(1, 2), cos, sin).transpose(1, 2)
        vd = to_heads(quant(head_to_hidden_shape(to_heads(cap["v"], nkv)), bits), nkv)
        od = to_heads(quant(o_hidden, bits), nh)
        cases = {
            "exact": (q0, k0, v0, o),
            "quant_o_only": (q0, k0, v0, od),
            "quant_qk_only": (qd, kd, v0, o),
            "quant_v_only": (q0, k0, vd, o),
            "quant_all (= HyC-LoRA FA)": (qd, kd, vd, od),
        }
        for name, (qq, kk, vv, oo) in cases.items():
            got = _flash_bwd(grad_o, qq, kk, vv, oo, lse, cq, ck, mq, mk, ps, po, scale)
            r = {"bits": bits, "case": name}
            for lbl, x, y in zip(("dq", "dk", "dv"), ref, got):
                xf, yf = x.float().flatten(), y.float().flatten()
                r[f"{lbl}_cos"] = float(torch.dot(xf, yf) / (xf.norm() * yf.norm() + 1e-30))
                r[f"{lbl}_rl2"] = float((yf - xf).norm() / (xf.norm() + 1e-30))
            rows.append(r)
            print(f"  q{bits} {name:28s} dq cos {r['dq_cos']:.5f} rl2 {r['dq_rl2']:.4f} | "
                  f"dk cos {r['dk_cos']:.5f} rl2 {r['dk_rl2']:.4f} | "
                  f"dv cos {r['dv_cos']:.5f} rl2 {r['dv_rl2']:.4f}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


def mode_depth(a):
    """Gradient-norm amplification as a function of depth.

    If the FA backward's error is multiplicative per layer (P = exp(S~ - lse_exact) is not row
    stochastic, so grad_x is rescaled by a per-row factor), the ratio ||g_arm|| / ||g_ref|| must
    grow geometrically from the last decoder layer towards the first.  A codec error that is
    merely additive would show a flat ratio.
    """
    tr = data(a.seq)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(tr.shape[0], generator=g)
    sd, _ = pretrain_adapters(tr, a.seed, a.seq, a.pretrain, 8, a.lr)
    warm = [tr[order[(a.pretrain + i) * 2:(a.pretrain + i + 1) * 2]].to(DEV) for i in range(a.warm)]
    probe = tr[order[(a.pretrain + a.warm) * 2:(a.pretrain + a.warm + 1) * 2]].to(DEV)

    ref, _, _, _ = grads_for("baseline_eager", sd, warm, probe, a.seed, a.seq)
    rows = {}
    for arm in a.arms.split(","):
        got, _, _, _ = grads_for(arm, sd, warm, probe, a.seed, a.seq)
        per = {}
        for n in got:
            m = re.search(r"layers\.(\d+)\.", n)
            if not m:
                continue
            li = int(m.group(1))
            per.setdefault(li, [0.0, 0.0])
            per[li][0] += float(got[n].pow(2).sum())
            per[li][1] += float(ref[n].pow(2).sum())
        ratio = {li: math.sqrt(v[0] / v[1]) for li, v in sorted(per.items())}
        rows[arm] = ratio
        s = "  ".join(f"L{li}:{r:.3g}" for li, r in sorted(ratio.items()) if li % 3 == 0 or li == 21)
        print(f"{arm:20s} ||g||/||g_ref|| by layer  {s}", flush=True)
        del got
        gc.collect(); torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


def mode_lse(a):
    """Is the blow-up intrinsic to the FA-backward *algorithm*, or to our aten substitution?

    Every FlashAttention-2 backward (Dao-AILab's included) reconstructs the probabilities as
        P = exp(Q K^T * scale - lse)
    with `lse` carried over from the forward.  HyC-LoRA's FA layer hands that backward a
    *dequantised* Q and K while `lse` still describes the *exact* Q and K, so P is no longer a
    row-stochastic matrix.  This probe measures that directly, in plain PyTorch, with no kernel
    involved -- if the row sums explode here, the defect is in the construction and not in the
    kernel we chose.
    """
    from hyclora.operators.compress_function import (
        compression_quantization_with_zero_point, decompression_dequantization_with_zero_point,
        get_statistics_only_quant_zero_point)
    from hyclora.compute_utils import hidden_to_head_shape
    from hyclora.operators.rope_kernels import rope_forward

    tr = data(a.seq)
    g = torch.Generator().manual_seed(a.seed)
    ids = tr[torch.randperm(tr.shape[0], generator=g)[:2]].to(DEV)
    m = build("baseline_sdpa", a.seed, a.seq)
    lay = m.base_model.model.model.layers[a.layer]
    hs = {}
    h = lay.register_forward_pre_hook(lambda mod, args: hs.__setitem__("x", args[0]))
    with torch.no_grad():
        m(input_ids=ids, attention_mask=torch.ones_like(ids))
    h.remove()
    with torch.no_grad():
        xn = lay.input_layernorm(hs["x"])
        q_m, k_m = lay.self_attn.q_proj(xn), lay.self_attn.k_proj(xn)
    nh, nkv = m.config.num_attention_heads, m.config.num_key_value_heads
    hd = m.config.hidden_size // nh
    cos, sin = m.base_model.model.model.rotary_emb(
        hs["x"], torch.arange(a.seq, device=DEV).unsqueeze(0))
    cos, sin = cos[0].to(torch.bfloat16), sin[0].to(torch.bfloat16)
    del m, hs
    gc.collect(); torch.cuda.empty_cache()

    def rope(t, n):
        return rope_forward(hidden_to_head_shape(t, n).transpose(1, 2), cos, sin).transpose(1, 2)

    rows = []
    S = a.seq
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=DEV))
    scale = hd ** -0.5
    q0, k0 = rope(q_m, nh), rope(k_m, nkv)
    kx = k0.repeat_interleave(nh // nkv, dim=1)
    S0 = (q0.float() @ kx.float().transpose(-1, -2)) * scale
    S0 = S0.masked_fill(~causal, float("-inf"))
    lse = torch.logsumexp(S0, dim=-1)                      # the forward's exact lse

    for bits in [int(b) for b in a.bits.split(",")]:
        st_q = get_statistics_only_quant_zero_point(q_m, bits, 'per-channel')
        st_k = get_statistics_only_quant_zero_point(k_m, bits, 'per-channel')
        qd_h = decompression_dequantization_with_zero_point(
            compression_quantization_with_zero_point(q_m.clone(), st_q[0], st_q[1], bits),
            st_q[0], st_q[1], bits)
        kd_h = decompression_dequantization_with_zero_point(
            compression_quantization_with_zero_point(k_m.clone(), st_k[0], st_k[1], bits),
            st_k[0], st_k[1], bits)
        codec_q = float((qd_h.float() - q_m.float()).norm() / q_m.float().norm())
        codec_k = float((kd_h.float() - k_m.float()).norm() / k_m.float().norm())
        zp_q, zp_k = st_q[1], st_k[1]
        # int8 zero point: the true (pre-cast) value is -round(min/scale) - 2^(b-1); check whether
        # casting it to int8 wrapped anything around.
        true_zp = (-torch.round(q_m[0].min(dim=-2, keepdim=True).values / st_q[0].float())
                   - 2 ** (bits - 1))
        n_wrap = int((true_zp != zp_q.float()).sum())

        qd, kd = rope(qd_h, nh), rope(kd_h, nkv)
        kdx = kd.repeat_interleave(nh // nkv, dim=1)
        Sq = (qd.float() @ kdx.float().transpose(-1, -2)) * scale
        Sq = Sq.masked_fill(~causal, float("-inf"))
        dS = (Sq - S0)[causal.expand_as(Sq)]
        P_fa = torch.exp(Sq - lse.unsqueeze(-1))           # what any FA-2 backward rebuilds
        rowsum = P_fa.sum(-1)
        r = {"bits": bits,
             "codec_relerr_q": codec_q, "codec_relerr_k": codec_k,
             "zp_int8_wrapped_channels": n_wrap, "zp_total_channels": int(zp_q.numel()),
             "score_err_absmax": float(dS.abs().max()), "score_err_rms": float(dS.pow(2).mean().sqrt()),
             "P_rowsum_mean": float(rowsum.mean()), "P_rowsum_max": float(rowsum.max()),
             "P_rowsum_min": float(rowsum.min()),
             "P_max": float(P_fa.max())}
        rows.append(r)
        print(f"q{bits}: codec relerr q {codec_q:.4f} k {codec_k:.4f} | "
              f"int8 zero-point wrapped {n_wrap}/{r['zp_total_channels']} channels | "
              f"|dS| max {r['score_err_absmax']:.2f} rms {r['score_err_rms']:.3f} | "
              f"row-sum of exp(S~ - lse_exact): mean {r['P_rowsum_mean']:.3e} "
              f"max {r['P_rowsum_max']:.3e} min {r['P_rowsum_min']:.3e} "
              f"(exact construction = 1.0)", flush=True)
        del P_fa, Sq
        gc.collect(); torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grads", choices=["grads", "calib", "famech", "lse", "depth"])
    ap.add_argument("--arms", default="hyclora_q2,hyclora_flash_q2,hyclora_q4,hyclora_flash_q4,"
                                      "hyclora_flash_nc")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--pretrain", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--bits", default="8,4,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"grads": mode_grads, "calib": mode_calib, "famech": mode_famech,
     "lse": mode_lse, "depth": mode_depth}[a.mode](a)
