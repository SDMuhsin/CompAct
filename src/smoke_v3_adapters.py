"""FlashFFN v3 adapter paths — 30-step e2e smoke (TinyLlama, wikitext-2).

Per adapter in {dora, adalora, dylora, vera}: three arms that differ ONLY in
the FFN path —
  v2            train_glue.apply_flash_ffn(model, k_fraction=0.3)
                (effective-weight recompute mode, the production v2 path)
  v3_int4       train_glue.apply_flash_ffn_v3(model, cache_mode='int4')
  v3_recompute  train_glue.apply_flash_ffn_v3(model, cache_mode='recompute')

Config mirrors the production train_glue path (CONTEXT.md section 22 validated
TinyLlama hyperparameters): target modules q/k/v/o + gate/up/down, DoRA
r8/a16/do0.05 lr5e-4, AdaLoRA r8->4/a16 lr5e-4, DyLoRA r8/a16 lr5e-4, VeRA
r256/d0.1 lr1e-3; AdamW, grad-clip 1.0, bf16 base, seed 41, wikitext-2 train
chunks (dataset capped at 1000 chunks; first steps*batch used, identical
across arms), 30 optimizer steps, eval ppl on the first N wikitext-2
validation chunks.

Reported per arm: loss sums @10/20/30 (loss-curve tracking), final eval ppl
(metric), peak CUDA MiB, median s/step and total wall time. Seeding notes:
torch+random seeded identically per arm; DyLoRA's per-forward rank sampling
consumes random.randint in the same order in every arm (attention DyLoRA
modules sample inside their own forward in all arms; MLP draws happen in the
v2/v3 closures, 3 per MLP in gate/up/down order), so the sampled ranks are
identical across arms. AdaLoRA's allocator is not stepped (30 < tinit=200, a
no-op in production too); orth-reg is applied by PEFT identically in all arms.

Honesty: v3 arms assert _V3_COUNTERS adapter_forward/adapter_backward advanced
to the exact expected counts; the v2 arm asserts the v3 counters stayed at 0.

Run:  source env/bin/activate
      CUDA_VISIBLE_DEVICES=1 python src/smoke_v3_adapters.py [--adapters dora ...]
"""

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HOME", "./data")
os.environ.setdefault("TORCH_HOME", "./data")

import argparse
import gc
import json
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashffn  # noqa: E402
from flashffn import _V3_COUNTERS, v3_reset_counters  # noqa: E402
import train_glue  # noqa: E402  (apply_flash_ffn / apply_flash_ffn_v3)
from dylora import get_dylora_model  # noqa: E402

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]
ADAPTERS = ("dora", "adalora", "dylora", "vera")
ARMS = ("v2", "v3_int4", "v3_recompute")
LRS = {"dora": 5e-4, "adalora": 5e-4, "dylora": 5e-4, "vera": 1e-3}
OUT_DIR = "results/v3_adapters"


def log(msg):
    print(msg, flush=True)


def build_chunks(tok, n_chunks, seq, split="train", max_chunks_pool=1000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text[:6_000_000], return_tensors="pt").input_ids[0]
    pool = min(max_chunks_pool, ids.numel() // seq)
    chunks = ids[: pool * seq].view(pool, seq)
    assert n_chunks <= pool, (n_chunks, pool)
    return chunks[:n_chunks]


def build_model(adapter, device, seed):
    from transformers import AutoModelForCausalLM
    from peft import (LoraConfig, AdaLoraConfig, VeraConfig, TaskType,
                      get_peft_model)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False

    if adapter == "dora":
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                         target_modules=TARGETS, bias="none",
                         task_type=TaskType.CAUSAL_LM, use_dora=True)
        model = get_peft_model(model, cfg)
    elif adapter == "adalora":
        cfg = AdaLoraConfig(init_r=8, target_r=4, lora_alpha=16,
                            lora_dropout=0.0, target_modules=TARGETS,
                            bias="none", task_type=TaskType.CAUSAL_LM,
                            total_step=1000, tinit=200, tfinal=200,
                            deltaT=10, orth_reg_weight=0.5)
        model = get_peft_model(model, cfg)
    elif adapter == "vera":
        cfg = VeraConfig(r=256, target_modules=TARGETS, vera_dropout=0.0,
                         bias="none", task_type=TaskType.CAUSAL_LM,
                         save_projection=True, projection_prng_key=0,
                         d_initial=0.1)
        model = get_peft_model(model, cfg)
    elif adapter == "dylora":
        model = get_dylora_model(model=model, target_modules=TARGETS,
                                 r=8, alpha=16, dropout=0.0)
    else:
        raise ValueError(adapter)
    model.train()
    return model


@torch.no_grad()
def eval_ppl(model, eval_chunks, device, batch):
    import math
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(0, eval_chunks.shape[0], batch):
        ids = eval_chunks[i:i + batch].to(device)
        out = model(input_ids=ids, labels=ids)
        n = ids.numel()
        total_loss += float(out.loss) * n
        total_tok += n
    model.train()
    return math.exp(min(total_loss / total_tok, 100))


def run_arm(adapter, arm, train_chunks, eval_chunks, device, args):
    seed = args.seed
    random.seed(seed)
    model = build_model(adapter, device, seed)

    n_layers = (model.config.num_hidden_layers
                if hasattr(model, "config")
                else model.model.config.num_hidden_layers)

    v3_reset_counters()
    if arm == "v2":
        n_patched = train_glue.apply_flash_ffn(model, k_fraction=0.3)
    else:
        mode = "int4" if arm == "v3_int4" else "recompute"
        n_patched = train_glue.apply_flash_ffn_v3(model, cache_mode=mode)
    assert n_patched == n_layers, (n_patched, n_layers)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LRS[adapter])

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses, step_times = [], []
    t_start = time.perf_counter()
    for step in range(args.steps):
        ids = train_chunks[step * args.batch:(step + 1) * args.batch].to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
    total_train_s = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024**2

    # honesty: the intended path must actually have run
    counters = dict(_V3_COUNTERS)
    if arm == "v2":
        assert counters["forward"] == 0 and counters["adapter_forward"] == 0, \
            f"v3 ran in the v2 arm! {counters}"
    else:
        expected = n_patched * args.steps
        assert counters["adapter_forward"] == expected \
            and counters["adapter_backward"] == expected, \
            f"v3 adapter path incomplete: {counters} vs expected {expected}"

    ppl = eval_ppl(model, eval_chunks, device, args.batch)

    res = {
        "losses": losses,
        "loss_sum_10": sum(losses[:10]),
        "loss_sum_20": sum(losses[:20]),
        "loss_sum_30": sum(losses[:args.steps]),
        "eval_ppl": ppl,
        "peak_mem_mib": peak,
        "median_step_s": statistics.median(step_times),
        "total_train_s": total_train_s,
        "n_patched": n_patched,
        "v3_counters": counters,
    }
    log(f"  {arm:13s}: loss 10/20/30 = {res['loss_sum_10']:.4f} / "
        f"{res['loss_sum_20']:.4f} / {res['loss_sum_30']:.4f} | "
        f"ppl {ppl:.4f} | peak {peak:.0f} MiB | "
        f"{res['median_step_s']:.3f} s/step | {total_train_s:.1f} s total")

    del model, opt, params
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=list(ADAPTERS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max_train_samples", type=int, default=1000)
    ap.add_argument("--eval_chunks", type=int, default=48)
    args = ap.parse_args()

    device = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    need = args.steps * args.batch
    assert need <= args.max_train_samples
    train_chunks = build_chunks(tok, need, args.seq, "train",
                                args.max_train_samples)
    eval_chunks = build_chunks(tok, args.eval_chunks, args.seq, "validation",
                               args.eval_chunks)
    log(f"data: {train_chunks.shape[0]} train chunks (pool capped at "
        f"{args.max_train_samples}), {eval_chunks.shape[0]} eval chunks, "
        f"seq={args.seq}, batch={args.batch}")

    results = {}
    out_path = os.path.join(OUT_DIR, "smoke_v3_adapters.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    for adapter in args.adapters:
        log(f"\n=== {adapter.upper()} (lr {LRS[adapter]}, seed {args.seed}, "
            f"{args.steps} steps) ===")
        for arm in args.arms:
            key = f"{adapter}.{arm}"
            res = run_arm(adapter, arm, train_chunks, eval_chunks, device, args)
            res_slim = dict(res)
            results[key] = res_slim
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1, default=str)

    # summary
    log("\n" + "=" * 110)
    log(f"{'adapter':9s} {'arm':13s} {'loss@10':>9s} {'loss@20':>9s} "
        f"{'loss@30':>9s} {'eval ppl':>9s} {'peak MiB':>9s} {'s/step':>7s} "
        f"{'total s':>8s}")
    for adapter in args.adapters:
        for arm in args.arms:
            r = results.get(f"{adapter}.{arm}")
            if r is None:
                continue
            log(f"{adapter:9s} {arm:13s} {r['loss_sum_10']:9.4f} "
                f"{r['loss_sum_20']:9.4f} {r['loss_sum_30']:9.4f} "
                f"{r['eval_ppl']:9.4f} {r['peak_mem_mib']:9.0f} "
                f"{r['median_step_s']:7.3f} {r['total_train_s']:8.1f}")
    log("=" * 110)
    log(f"[saved {out_path}]")


if __name__ == "__main__":
    main()
