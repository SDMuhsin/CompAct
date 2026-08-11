"""Probe 5 (attack item 6): real WikiText-2 LoRA fine-tune, >=3 seeds, per-arm perplexity.

Everything except the training-time layer implementation is held fixed: same data, same order,
same LoRA init (seeded), same optimiser/schedule, same eval.  Perplexity is ALWAYS measured on a
freshly built, unpatched, uncompressed stock-HF bf16 model into which the trained adapter weights
are loaded -- so no arm can win or lose at eval time, only at training time.
"""
import argparse, gc, json, math, os, sys, time
import torch

sys.path.insert(0, "/workspace/CompAct/src")

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEV = torch.device("cuda")


def data(block=1024):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tk(ex):
        return tok(ex["text"], return_attention_mask=False)
    t = raw.map(tk, batched=True, remove_columns=raw["train"].column_names, desc="tok")

    def grp(ex):
        cc = {k: sum(ex[k], []) for k in ex}
        n = (len(cc["input_ids"]) // block) * block
        r = {k: [cc[k][i:i + block] for i in range(0, n, block)] for k in cc}
        r["labels"] = r["input_ids"].copy()
        return r
    p = t.map(grp, batched=True, desc="chunk")
    tr = torch.tensor(p["train"]["input_ids"], dtype=torch.long)
    ev = torch.tensor(p["validation"]["input_ids"], dtype=torch.long)
    te = torch.tensor(p["test"]["input_ids"], dtype=torch.long)
    return tr, ev, te


def build(arm, seed, seq):
    from profile_hyclora import build_model
    cfg = {"model": MODEL, "batch": 2, "seq": seq, "lora_r": 16, "q_bit": 2,
           "softmax_outlier_ratio": 0.05, "layernorm_outlier_ratio": 0.005,
           "iteration_threshold": 5, "n_layers": 22}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return build_model(arm, cfg, DEV, adapter_dtype="bf16")


@torch.no_grad()
def perplexity(model, blocks, batch=4):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, blocks.shape[0], batch):
        ids = blocks[i:i + batch].to(DEV)
        out = model(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
        k = ids.numel()
        tot += float(out.loss) * k
        n += k
    model.train()
    return math.exp(tot / n), tot / n


def eval_with_clean_model(sd, ev, te, seed, seq):
    """Identical eval path for every arm: stock HF bf16 + LoRA, no patching."""
    m = build("baseline_sdpa", seed, seq)
    missing = m.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys[:5]
    p_ev, l_ev = perplexity(m, ev)
    p_te, l_te = perplexity(m, te)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return {"val_ppl": p_ev, "val_loss": l_ev, "test_ppl": p_te, "test_loss": l_te}


def train_arm(arm, seed, tr, ev, te, args):
    t0 = time.time()
    m = build(arm, seed, args.seq)
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(tr.shape[0], generator=g)
    n_micro = min(args.max_micro, (tr.shape[0] // args.batch))
    n_opt = n_micro // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) *
        max(0.0, 1.0 - max(0, s - args.warmup) / max(1, n_opt - args.warmup)))
    losses, k = [], 0
    for i in range(n_micro):
        idx = order[i * args.batch:(i + 1) * args.batch]
        ids = tr[idx].to(DEV)
        out = m(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids))
        (out.loss / args.accum).backward()
        losses.append(float(out.loss))
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            k += 1
        if (i + 1) % 200 == 0:
            print(f"    [{arm} s{seed}] micro {i + 1}/{n_micro} opt {k} "
                  f"loss(ma100)={sum(losses[-100:]) / len(losses[-100:]):.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    counters = None
    if arm.startswith("fb_"):
        from flashffn import fb_get_counters
        counters = fb_get_counters()
        assert counters["forward"] > 0 and counters["backward"] > 0 and (
            counters["recompute"] > 0 if "fb_min" in arm else True), counters
    elif arm.startswith("hyclora"):
        from hyclora.patch import get_counters
        counters = get_counters()
    sd = {n: p.detach().to(torch.bfloat16).clone() for n, p in m.named_parameters()
          if p.requires_grad}
    del m, opt
    gc.collect(); torch.cuda.empty_cache()
    r = eval_with_clean_model(sd, ev, te, seed, args.seq)
    r.update({"arm": arm, "seed": seed, "n_micro": n_micro, "n_opt_steps": k,
              "train_loss_last100": sum(losses[-100:]) / len(losses[-100:]),
              "train_loss_first100": sum(losses[:100]) / len(losses[:100]),
              "counters": counters, "sec": time.time() - t0})
    print(f"  == {arm:20s} seed {seed}  val_ppl={r['val_ppl']:.4f}  test_ppl={r['test_ppl']:.4f}"
          f"  train_loss(last100)={r['train_loss_last100']:.4f}  {r['sec']:.0f}s", flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline_sdpa,fb_min_fnorm_sdpa,hyclora_flash_q2")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max_micro", type=int, default=100000)
    ap.add_argument("--out", required=True)
    # T-5 ENFORCEMENT, added 2026-08-09 (CONTEXT.md 46.11 / 37.9).  Until today this probe PRINTED
    # the paired delta and always exited 0: CONTEXT.md 37.5.2's quality clause ("paired d(ppl)
    # within +-2e-3") had no threshold, no PASS/FAIL and no non-zero exit anywhere in the repo, so
    # T-5 was a manual protocol that nothing could fail.  That matters because the exactness audit
    # (46.11) found no gate in verify_fused_block.py constrains the block's arithmetic against an
    # external reference either -- leaving quality as the ONLY check on an arithmetic change, and
    # it was unenforced.
    ap.add_argument("--t5_tol", type=float, default=2e-3,
                    help="CONTEXT.md 37.5.2's bar on the paired per-seed d(test ppl). "
                         "Do NOT widen this to make a change pass.")
    ap.add_argument("--t5_gate_arms", default="",
                    help="comma-separated arms to GATE; default = every arm whose name starts "
                         "with 'fb_'. Competitor arms (e.g. hyclora_flash_q2) are reported but not "
                         "gated -- they are lossy by construction and would fail by design.")
    ap.add_argument("--t5_require_seeds", default="41,42,43",
                    help="seeds that must be present for a PASS; an incomplete run FAILS rather "
                         "than silently passing. Empty string disables the completeness check.")
    ap.add_argument("--no_t5_gate", action="store_true",
                    help="report only, exit 0 regardless. For exploratory runs; never for T-5.")
    a = ap.parse_args()
    tr, ev, te = data(a.seq)
    print(f"blocks: train {tr.shape} val {ev.shape} test {te.shape}", flush=True)
    rows = []
    if os.path.exists(a.out):
        rows = json.load(open(a.out))
    done = {(r["arm"], r["seed"]) for r in rows}
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in a.arms.split(","):
            if (arm, seed) in done:
                continue
            rows.append(train_arm(arm, seed, tr, ev, te, a))
            json.dump(rows, open(a.out, "w"), indent=2)
    print("\n== summary ==")
    import statistics
    for arm in a.arms.split(","):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        v = [r["val_ppl"] for r in rs]
        t = [r["test_ppl"] for r in rs]
        print(f"{arm:22s} n={len(rs)}  val ppl {statistics.mean(v):.4f} "
              f"+- {(statistics.stdev(v) if len(v) > 1 else 0):.4f}  "
              f"test ppl {statistics.mean(t):.4f} "
              f"+- {(statistics.stdev(t) if len(t) > 1 else 0):.4f}  {[round(x, 4) for x in t]}")
    base = [r for r in rows if r["arm"] == "baseline_sdpa"]
    for arm in a.arms.split(","):
        if arm == "baseline_sdpa":
            continue
        d = []
        for r in [x for x in rows if x["arm"] == arm]:
            b = next((x for x in base if x["seed"] == r["seed"]), None)
            if b:
                d.append(r["test_ppl"] - b["test_ppl"])
        if d:
            print(f"paired d(test ppl) vs baseline_sdpa: {arm:22s} "
                  f"mean {statistics.mean(d):+.4f}  per-seed {[round(x, 4) for x in d]}")

    # ---- T-5 VERDICT (see --t5_tol above for why this block exists) ----
    gate_arms = ([s for s in a.t5_gate_arms.split(",") if s]
                 or [s for s in a.arms.split(",") if s.startswith("fb_")])
    need_seeds = [int(s) for s in a.t5_require_seeds.split(",") if s]
    print("\n== T-5 verdict ==")
    t5 = {"tol": a.t5_tol, "gated_arms": gate_arms, "required_seeds": need_seeds, "arms": {}}
    failures = []
    for arm in gate_arms:
        per_seed = {}
        for r in [x for x in rows if x["arm"] == arm]:
            b = next((x for x in base if x["seed"] == r["seed"]), None)
            if b:
                per_seed[r["seed"]] = r["test_ppl"] - b["test_ppl"]
        missing = [s for s in need_seeds if s not in per_seed]
        breached = {s: v for s, v in per_seed.items() if abs(v) > a.t5_tol}
        ok = not missing and not breached and bool(per_seed)
        t5["arms"][arm] = {"per_seed_delta": per_seed, "missing_seeds": missing,
                           "breached": breached, "PASS": ok}
        if not ok:
            why = []
            if not per_seed:
                why.append("no paired seeds at all")
            if missing:
                why.append(f"missing seeds {missing}")
            if breached:
                why.append("|d| > tol at " +
                           ", ".join(f"seed {s}: {v:+.6f}" for s, v in sorted(breached.items())))
            failures.append(f"{arm}: " + "; ".join(why))
        print(f"  {arm:22s} {'PASS' if ok else 'FAIL'}  tol=+-{a.t5_tol:g}  "
              f"per-seed {{{', '.join(f'{s}: {v:+.6f}' for s, v in sorted(per_seed.items()))}}}"
              + (f"  MISSING {missing}" if missing else ""))
    ungated = [s for s in a.arms.split(",") if s and s != "baseline_sdpa" and s not in gate_arms]
    if ungated:
        print(f"  (reported but NOT gated: {', '.join(ungated)})")
    t5["PASS"] = not failures
    res_path = os.path.splitext(a.out)[0] + "_t5verdict.json"
    json.dump(t5, open(res_path, "w"), indent=2)
    print(f"  -> {res_path}")
    if failures:
        print("T-5 FAIL: " + " | ".join(failures))
        if a.no_t5_gate:
            print("  (--no_t5_gate set: exiting 0 anyway -- this run does NOT satisfy T-5)")
        else:
            sys.exit(1)
    else:
        print("T-5 PASS")
