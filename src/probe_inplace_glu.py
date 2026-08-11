"""ON-vs-OFF equivalence for the in-place GLU backward window (CONTEXT.md section 33.5/33.6).

The restructure in `flashffn.triton_swiglu_backward_glu` changes only WHERE three tensors are
written, never what is computed.  This probe is the direct evidence for that claim, and the direct
evidence for the memory it returns:

  * every trainable gradient, bitwise, with the path enabled and disabled,
  * the loss and the output logits, bitwise,
  * training-step `max_memory_allocated` both ways, on the same process and the same batch,

for each `keep` level and each adapter family asked for.  `fb_inplace_glu_enable(False)` restores
the two-kernel path exactly, so an OFF row is the pre-change artifact and any difference in the
gradient columns is a defect in the restructure.

THE PEAK COLUMN HERE IS A CONTROL, NOT THE RESULT.  This probe runs stock HF cross-entropy at
short sequences, where the fp32-logits/CE stack -- not the decoder stack -- sets the high-water
mark, so the window this change shrinks is not what is being measured and the column reads
0.00 MiB saved.  That is the correct answer to "does it cost anything anywhere", and it is not the
memory result; the memory result is `profile_unsloth.py --flce` at seq >= 1024, under
`fair_comparison_protocol.md`.

Attention is EAGER by default: SDPA/FlashAttention's backward is not bitwise reproducible against
itself at these shapes (~3-4.5e-3 on adapter grads), which is larger than the effect under test and
would make a bitwise claim meaningless (CONTEXT.md section 33.10).

Usage:
  python src/probe_inplace_glu.py --keep min,attn,glu,full --families lora,dora --seq 256
"""

import argparse
import gc
import random
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build(keep, family, seq, batch, device, attn_impl, lora_r=16):
    """A patched model plus a fixed batch. Adapters are overwritten from a deterministic stream so
    the two runs being compared start from identical parameters regardless of PEFT's RNG use."""
    from transformers import AutoModelForCausalLM
    import flashffn
    from flashffn import apply_flash_block, apply_flash_final_norm

    # The repo's own family builder, so this probe cannot disagree with the gate suite about what
    # "dylora" or "frozen" means (DyLoRA is a local implementation, not a PEFT config, and the
    # frozen family trains only the input embedding -- without which the loss has no `grad_fn`).
    from profile_hyclora import apply_family

    torch.manual_seed(41)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation=attn_impl, torch_dtype=torch.bfloat16)
    m.config.use_cache = False
    m = apply_family(m, family, lora_r)
    m.to(device)
    m.train()

    g = torch.Generator(device="cpu").manual_seed(1234)
    for n, p in sorted(m.named_parameters()):
        if p.requires_grad and ("lora_" in n or "vera_" in n):
            p.data.copy_((torch.randn(tuple(p.shape), generator=g, dtype=torch.float32) * 0.02)
                         .to(device=p.device, dtype=p.dtype))

    apply_flash_block(m, keep=keep, verbose=False)
    apply_flash_final_norm(m)
    flashffn.fb_reset_counters()

    vocab = m.config.vocab_size
    gb = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, vocab, (batch, seq), generator=gb).to(device)
    return m, {"input_ids": ids, "labels": ids.clone(),
               "attention_mask": torch.ones_like(ids)}


def grads_of(model, batch):
    """Everything is moved to the HOST before returning.

    Python's `random` and torch's global RNG are re-seeded on entry because DyLoRA draws its rank
    with `random.randint` on EVERY forward (`src/dylora.py:73`).  Without this the two runs being
    compared use different ranks and disagree on all 308 gradients AND on the loss -- and the loss
    is the tell, since a backward-only change cannot move a forward quantity.  (Observed exactly
    that before this line existed.)

    The comparison holds one run's results while the other run executes, and a GPU-resident copy of
    308 gradients plus a [B, S, vocab] logits tensor is ~80 MiB -- which lands in the second run's
    `max_memory_allocated` and inverts the sign of the memory column.  (Observed exactly that: the
    enabled arm appeared 80 MiB HEAVIER, identically at all four keep levels, which is the
    signature of a fixed retained baseline rather than of anything the change does.)"""
    random.seed(1234)
    torch.manual_seed(1234)
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    g = {n: p.grad.detach().to("cpu", copy=True) for n, p in model.named_parameters()
         if p.grad is not None}
    loss = out.loss.detach().to("cpu", copy=True)
    logits = (out.logits.detach().to("cpu", copy=True)
              if getattr(out, "logits", None) is not None else None)
    model.zero_grad(set_to_none=True)
    del out
    return g, loss, logits


def _bits(t):
    """Raw bytes of any tensor, including 0-dim ones (`view(uint8)` refuses those directly)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def bitwise_eq(x, y):
    return bool(_bits(x).eq(_bits(y)).all().item()) if x.shape == y.shape else False


def compare(a, b):
    """Bitwise over the raw bytes -- `torch.equal` on floats would call two NaNs unequal and two
    differently-signed zeros equal, and neither is what 'the same bytes were written' means."""
    keys = sorted(set(a) & set(b))
    n_diff, worst, worst_k = 0, 0.0, "-"
    for k in keys:
        x, y = a[k], b[k]
        same = bitwise_eq(x, y)
        if not same:
            n_diff += 1
            d = (x.float() - y.float()).abs().max().item()
            if d > worst:
                worst, worst_k = d, k
    return {"n_tensors": len(keys), "n_bitwise_differing": n_diff,
            "worst_abs_diff": worst, "worst_tensor": worst_k,
            "n_missing": len(set(a) ^ set(b)),
            "BITWISE_IDENTICAL": n_diff == 0 and len(set(a) ^ set(b)) == 0}


def peak_of(model, batch):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        opt.step()
        del out
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    opt.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated()
    del out, opt
    return p / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default="min,attn,glu,full")
    ap.add_argument("--families", default="lora")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--out", default="results/hyclora/frontier/inplace_glu_equiv.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    import flashffn

    rows, all_ok = [], True
    for family in args.families.split(","):
        for keep in args.keep.split(","):
            row = {"family": family, "keep": keep, "seq": args.seq, "attn": args.attn}
            try:
                res = {}
                for on in (False, True):
                    flashffn.fb_inplace_glu_enable(on)
                    m, batch = build(keep, family, args.seq, args.batch, device, args.attn)
                    g, loss, logits = grads_of(m, batch)
                    res[on] = {"g": g, "loss": loss, "logits": logits,
                               "peak": peak_of(m, batch),
                               "n_trainable": sum(1 for p in m.parameters() if p.requires_grad)}
                    del m
                    gc.collect()
                    torch.cuda.empty_cache()
                row.update(compare(res[True]["g"], res[False]["g"]))
                row["loss_off"] = float(res[False]["loss"])
                row["loss_on"] = float(res[True]["loss"])
                row["loss_bitwise"] = bitwise_eq(res[True]["loss"], res[False]["loss"])
                if res[False]["logits"] is not None:
                    row["logits_bitwise"] = bitwise_eq(res[True]["logits"], res[False]["logits"])
                row["peak_off_MiB"] = res[False]["peak"]
                row["peak_on_MiB"] = res[True]["peak"]
                row["peak_saved_MiB"] = res[False]["peak"] - res[True]["peak"]
                row["n_trainable"] = res[True]["n_trainable"]
                row["PASS"] = bool(row["BITWISE_IDENTICAL"] and row["loss_bitwise"])
                del res
            except Exception as exc:
                import traceback
                traceback.print_exc()
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["PASS"] = False
            all_ok &= bool(row.get("PASS"))
            rows.append(row)
            print(f"  {family:<8} keep={keep:<5} "
                  f"bitwise={row.get('BITWISE_IDENTICAL')} "
                  f"({row.get('n_bitwise_differing', '?')}/{row.get('n_tensors', '?')} differ) "
                  f"loss_bw={row.get('loss_bitwise')} "
                  f"peak {row.get('peak_off_MiB', float('nan')):.2f} -> "
                  f"{row.get('peak_on_MiB', float('nan')):.2f} MiB "
                  f"(-{row.get('peak_saved_MiB', float('nan')):.2f})", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    flashffn.fb_inplace_glu_enable(True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "ALL_PASS": all_ok, "attn": args.attn,
                   "seq": args.seq, "batch": args.batch}, f, indent=2, default=str)
    print(f"\n{'ALL PASS' if all_ok else 'FAILURES PRESENT'} -> {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
