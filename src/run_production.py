"""Production experiment runner: one method (optionally composed with the fused block), one task,
one seed -> one CSV row, written under a file lock so N of these can run concurrently.

WHAT THIS GUARANTEES, AND WHY EACH GUARANTEE EXISTS
---------------------------------------------------
1. **Methods are constructed by `profile_hyclora.build_model`** -- the same code that produced the
   memory numbers in CONTEXT.md §16 -- so a task row and a memory row describe the same object.
2. **Illegal combinations are refused BEFORE the model is built** (`experiment_registry.resolve_arm`).
   A camera-ready sweep must not spend GPU hours on a row it will throw away.
3. **Every requested method must PROVE it executed** (`verify_engagement`). This is the load-bearing
   one: `alst + fb` applies without error and silently kills ALST, so "it ran without crashing" is
   not evidence. A row whose method did not run is worse than a missing row.
4. **The CSV write is lock-protected and atomic** -- `FileLock` + temp-file + `os.replace`, the same
   proven path `train_glue.write_result_row` uses. Concurrent runners cannot interleave or truncate.
5. **Every hyperparameter is recorded.** Common ones get their own column; per-method configuration
   and engagement counters are carried verbatim as JSON so nothing is lost when a new method adds a
   knob. `config_hash` makes a cell identifiable without parsing JSON.
6. **Task AND computational metrics in the same row**, so a quality/memory trade is readable without
   joining two files.

SCOPE OF THE TASK LAYER
-----------------------
* `--task lm:synthetic` -- causal-LM perplexity on a fixed random batch. Computational metrics only;
  the "perplexity" of random tokens is a receipt that the step ran, NOT a quality number.
* `--task glue:<config>` -- **real GLUE/SuperGLUE training and evaluation** on a
  sequence-classification head, with the task's own official metric. This is the quality half of
  the §16 mandate. Enabled by `profile_hyclora.load_base_model`, which parameterises all eight
  builders by head; before that every builder hardcoded `AutoModelForCausalLM` and this runner
  refused `glue:*` rather than measure the wrong head.
* `--task mc:*` -- still refused. The commonsense/MMLU suites need the registered
  `AutoModelForMultipleChoice`, which `load_base_model` does not yet construct.

⚠ REGIME. `--flce` applies Liger `FusedLinearCrossEntropy` (regime B, CONTEXT.md §8) via
`profile_unsloth._apply_flce`, and it must be passed BEFORE the model is built because it is a
class-level monkey-patch. **It is meaningless on a GLUE head** -- there is no LM head and no
vocab-sized logits to fuse -- so `--flce` with `--task glue:*` is refused rather than recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import profile_hyclora as ph  # noqa: E402
from experiment_registry import (  # noqa: E402
    REGISTRY, FB_COMPAT, FLCE_FORBIDDEN, FULL_FT_REGIME, LOSSY,
    resolve_arm, resolve_head, resolve_fb_variant, verify_engagement,
    CombinationRefused, EngagementFailure,
)

CSV_COLUMNS = [
    # --- identity ---
    "timestamp", "run_id", "method", "with_fb", "arm", "task", "seed", "model_name_or_path",
    # --- protocol / hyperparameters ---
    "regime", "exact", "seq_len", "batch_size", "grad_accum", "train_steps", "lr",
    "max_grad_norm", "lr_scheduler", "warmup_ratio",
    "epochs", "steps_per_epoch", "best_epoch", "seq_source",
    "lora_r", "lora_alpha", "lora_dropout", "target_modules", "adapter_dtype",
    "gc_variant", "fb_variant", "wstream_json", "flce", "attn_implementation",
    "n_trainable_params",
    # --- task metrics ---
    # `task_metric_name`/`task_metric` carry the task's OWN headline metric (MCC for cola, F1 for
    # mrpc/qqp, Pearson for stsb, accuracy elsewhere), so a sweep is readable without knowing which
    # task filled which column. The named columns stay populated too, for joins with mo53_glue.csv.
    "eval_loss", "perplexity", "accuracy", "f1", "matthews_correlation", "pearson", "spearmanr",
    "task_metric_name", "task_metric", "task_metric_json",
    # `pred_distribution` is the degeneracy receipt -- see evaluate_glue(). `n_train_examples` and
    # `n_eval_examples` make "trained on how much?" answerable from the row alone.
    "pred_distribution", "n_train_examples", "n_eval_examples", "eval_split", "dataset_source",
    "train_steps_total",
    # --- computational metrics ---
    "train_step_peak_alloc_mib", "train_step_peak_reserved_mib", "resident_floor_mib",
    "peak_minus_floor_mib", "ms_per_step_median", "ms_per_step_iqr", "total_train_time_sec",
    "host_pinned_mib",
    # --- receipts: everything method-specific, losslessly ---
    "engagement_ok", "engagement_json", "method_config_json", "config_hash",
    "torch_version", "harness", "notes",
]
# One row per (method, fb, fb_variant, task, seed, seq, batch, lr). Re-running a cell UPDATES it
# in place. ⚠ `fb_variant` is in the key deliberately: `fb_min` and `fb_min_wstream` are different
# arms sharing every other coordinate, so without it the second run silently overwrites the first.
COMB_COLS = ["method", "with_fb", "fb_variant", "task", "seed", "seq_len", "batch_size", "lr"]


def _model_dtype(model):
    """The dtype the model actually computes in, read off a live parameter rather than a flag --
    the same discipline the protocol demands for adapter dtype (CONTEXT.md 8)."""
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _wstream_stats(with_fb, fb_variant):
    """The streamer's own receipt -- host bytes pinned, H2D per step, blocked acquires. A saving
    with an unchanged floor is a bug, so the row must carry the evidence, not just the arm name."""
    if not (with_fb and fb_variant == "wstream"):
        return None
    import fb_wstream as _fbw
    return _fbw.fb_wstream_stats()


def _cfg(args):
    # ⚠ `lr` IS IN cfg BECAUSE TWO BUILDERS OWN THEIR OWN OPTIMIZER AND WERE IGNORING `--lr`.
    #   `build_lomo_model` hardcoded `lr = 3e-4` (profile_hyclora.py:751) and `build_zero3_model`
    #   hardcoded `"lr": 2e-4` in its DeepSpeed config, while this runner wrote `args.lr` into the
    #   CSV's `lr` column for every row. So the lomo/adalomo rows of sweep 54846646 record 2e-4 and
    #   were actually trained at 3e-4. That is the same defect class as the `flce=1`-without-FLCE
    #   bug in §8: A VALUE THAT WAS RECORDED BUT NEVER REACHED THE KERNEL. Builders now read
    #   `cfg["lr"]`, so there is one source of truth. **Verify the receipt, never the flag.**
    return {"model": args.model, "lora_r": args.lora_r, "seq": args.seq, "batch": args.batch,
            "iteration_threshold": 5, "softmax_outlier_ratio": 0.05,
            "layernorm_outlier_ratio": 0.005, "q_bit": 4,
            "lr": args.lr, "max_grad_norm": args.max_grad_norm}


# ==============================================================================================
# GLUE / SuperGLUE task layer
# ==============================================================================================
# ⚠ ROUTING IS COPIED FROM `train_glue.py`, NOT INVENTED, so a row here and a row there describe
#   the same task. `boolq`/`cb` live in SuperGLUE and take the `super_glue` dataset AND metric
#   (train_glue.py:2340); everything else is GLUE (train_glue.py:2344). Getting this wrong does not
#   crash -- it silently scores against the wrong metric.
SUPER_GLUE_TASKS = ("boolq", "cb")

# ---------------------------------------------------------------------------------------------
# PER-TASK SEQUENCE LENGTH. ⚠ MEASURED with the model's OWN tokenizer (TinyLlama), 4,000 train
# examples per task, on 2026-08-14 -- not chosen by eye:
#
#   task    mean   p50   p90   p95   p99   max   | complete at 128 | at 384 | at 512
#   boolq  161.7   149   261   311   422  1041   |     38.6%       | 98.3%  | 99.6%
#   qnli    56.9    53    84    96   121   663   |     99.2%       |  100%  |  100%
#   sst2    14.9    11    30    37    51    68   |      100%       |  100%  |  100%
#
# **At 128, boolq loses 61.4% of its examples to truncation** -- it is question+passage, and the
# passage is where the answer is. That is not a small effect: the dev-box pilot scored 0.6664
# against a 0.6217 majority baseline with 92% of predictions on one class, which is what a model
# reading truncated passages should look like. 384 keeps 98.3% intact for 3x the tokens; 512 would
# buy the last 1.3%. Every other task is complete at 128, so raising them would cost memory and
# time for literally nothing.
#
# ⚠ THE PRICE, AND IT MUST BE STATED WHEREVER THIS TABLE IS: boolq's memory and step-time columns
#   are NO LONGER COMPARABLE with the other tasks' -- both scale with sequence length. `seq_len` is
#   in COMB_COLS so the row is keyed by it, and `seq_source` records whether the value came from
#   this policy or an explicit flag. Compare boolq across ARMS, never across TASKS.
GLUE_DEFAULT_SEQ = 128                      # train_glue.py's own --max_length default
TASK_MAX_LENGTH = {"boolq": 384}            # everything else takes GLUE_DEFAULT_SEQ


# ==============================================================================================
# CAUSAL-LM CORPORA — the benchmark family where EVERY baseline is actually engaged.
# ==============================================================================================
# ⚠ WHY THIS EXISTS, AND WHY A GLUE-ONLY TABLE IS NOT ENOUGH. A sequence-classification head
#   structurally disables or degrades THREE of the eight published baselines, and it is not a
#   fixable limitation of our integration -- it is what those methods are:
#
#     streambp  REFUSED outright. Its mechanism IS the chunked LM head; `LlamaForSequenceClassification`
#               has `score`, one logit vector per sequence, nothing to chunk (SEQ_CLS_REFUSES).
#     minis     HALF-ENGAGED. `_eng_minis` measured 22 MLP wrappers and ZERO LM-head wrappers on
#               glue:sst2 -- their headline memory claim patches `LlamaForCausalLM` and cannot fire.
#     alst      INERT. num_shards = ceil(seq/hidden); at GLUE lengths (128, or 384 for boolq) on a
#               2048-hidden model that is 1, i.e. no tiling at all.
#
#   And `--flce` is refused on a GLUE head, so every GLUE row is regime A -- while §8 says regime B
#   is "where competitor claims are adjudicated". A causal-LM corpus at a long sequence fixes all
#   four at once: there is an `lm_head` to chunk, seq > hidden so ALST tiles, and FLCE applies.
#
#   wstream is unaffected either way -- it needs only FROZEN base weights, so it runs on every LoRA
#   arm here exactly as it does on GLUE.
LM_CORPORA = {
    # name -> (hub id, config, text column, eval split)
    # Both verified to load OFFLINE from ./data on 2026-08-15.
    "wikitext2": ("wikitext", "wikitext-2-raw-v1", "text", "validation"),
    "pg19":      ("emozilla/pg19", None, "text", "validation"),
}

# ⚠⚠ 2048, AND ALST CANNOT BE RESCUED ON THIS MODEL. MEASURED 2026-08-15, DO NOT RE-DERIVE.
#
#   I first set this to 4096 so that ALST would tile (their rule is `num_shards = ceil(seq/hidden)`,
#   so seq must EXCEED the 2048 hidden size for a second shard to exist). That is invalid, because
#   TinyLlama's `max_position_embeddings` is **also 2048** -- the two numbers coincide exactly --
#   and running past it makes RoPE extrapolate into nonsense. Base-model WikiText-2 perplexity,
#   no training, packed blocks:
#
#       seq  512 -> 10.634      seq 2048 -> 8.171
#       seq 1024 ->  9.092      seq 4096 -> 149.918   <-- past the context limit
#
#   So on TinyLlama-1.1B the ALST-tiling requirement (seq > 2048) and the valid-perplexity
#   requirement (seq <= 2048) are MUTUALLY EXCLUSIVE. ALST is therefore inapplicable to any
#   quality benchmark on this model -- not through a task choice we could change, but because the
#   model's context window is exactly its tiling threshold. Report it as such, citing both numbers.
#   Measuring ALST's tiling needs a longer-context model; that is a model decision, not a task one.
#
#   ⚠ This limit binds QUALITY only. The §4.1 memory table legitimately runs to seq 16384 because
#   it uses `lm:synthetic` random tokens, where the docstring already says the "perplexity" is a
#   receipt that the step ran and NOT a quality number. Do not use this note to question those.
LM_DEFAULT_SEQ = 2048


def resolve_seq(explicit, is_glue, glue_name, is_lm_corpus=False):
    """(seq, source). `--seq` still wins when given; the policy fills in when it is not."""
    if explicit is not None:
        return explicit, "explicit"
    if is_glue:
        return TASK_MAX_LENGTH.get(glue_name, GLUE_DEFAULT_SEQ), "task_policy"
    if is_lm_corpus:
        return LM_DEFAULT_SEQ, "lm_policy"
    return 1024, "default"


def build_lm_data(name, tokenizer, args):
    """Load one causal-LM corpus and pack it into fixed `args.seq` blocks.

    Returns (train_dl, eval_dl, sizes).

    ⚠ CONCATENATE-AND-CHUNK, NOT PAD-AND-TRUNCATE. Perplexity is only comparable across arms if
    every arm sees the same tokens in the same blocks, and a padded corpus would (a) make the
    number depend on document boundaries and (b) put a variable amount of PAD in the loss. Packing
    is what the wikitext-2 perplexity numbers in the literature assume, and it makes `seq` a
    controlled variable rather than an upper bound -- which matters here because `seq` is the axis
    ALST and the memory table both live on.

    Labels are the input ids; the model's own shift-by-one supplies causality (LlamaForCausalLM
    does the shift internally), so no manual shifting here -- doing it twice is a classic silent
    off-by-one that leaves the loss plausible and the perplexity wrong.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset

    if name not in LM_CORPORA:
        raise KeyError(f"unknown lm corpus {name!r}; known: {sorted(LM_CORPORA)}")
    hub_id, cfg_name, text_col, eval_split = LM_CORPORA[name]
    raw = load_dataset(hub_id, cfg_name) if cfg_name else load_dataset(hub_id)

    def pack(split, max_docs=None):
        # ⚠ `"\n\n".join(non-blank rows)`, TOKENISED AS ONE STREAM. This is not a style choice --
        #   it is the convention four already-validated probes in this repo use
        #   (`smoke_v3_adapters.py:73`, `diag_hyclora_grads.py:71`, `codec_feasibility_v3.py:64`,
        #   `probe_value_redundancy.py`) and it is what produces CONTEXT.md §5.1's WikiText-2
        #   figure of 7.19 for this model.
        #   The first version here tokenised ROW BY ROW and appended `eos_token_id` after each.
        #   WikiText's rows are partial paragraphs, not documents, so that injected ~36k spurious
        #   EOS into the stream and measured **perplexity 90.60 instead of ~7.2** -- a 12x error
        #   that looks entirely plausible in isolation. A perplexity is only comparable against
        #   published numbers if the token stream is built the same way.
        ds = raw[split]
        if max_docs:
            ds = ds.select(range(min(max_docs, len(ds))))
        texts = [t for t in ds[text_col] if t and t.strip()]
        ids = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
        n_blocks = len(ids) // args.seq
        if n_blocks == 0:
            raise ValueError(
                f"lm:{name}/{split} yielded {len(ids)} tokens, fewer than one block of "
                f"{args.seq}. Lower --seq or raise --max_train_samples.")
        buf = torch.tensor(ids[:n_blocks * args.seq], dtype=torch.long).view(n_blocks, args.seq)
        return buf

    # ⚠ PG-19 IS ENORMOUS (28,602 books). Reading it whole to tokenise would take longer than the
    #   training run and would OOM the host. `--max_train_samples`/`--max_eval_samples` cap the
    #   DOCUMENT count here (they cap examples on GLUE), and the row records the block counts, so
    #   what was actually trained on is always readable from the data.
    train_blocks = pack("train", args.max_train_samples or (200 if name == "pg19" else None))
    eval_blocks = pack(eval_split, args.max_eval_samples or (20 if name == "pg19" else None))

    train_dl = DataLoader(TensorDataset(train_blocks), batch_size=args.batch,
                          shuffle=True, drop_last=True)
    eval_dl = DataLoader(TensorDataset(eval_blocks), batch_size=args.batch, drop_last=False)
    # ⚠ `n_train`/`n_eval` are the names the GLUE path uses and the row reads. Blocks, not tokens
    #   -- an "example" here is one packed block of `args.seq` tokens. Token counts travel too,
    #   because "how much text did it see?" is not answerable from block count without `seq`.
    sizes = {"n_train": int(train_blocks.shape[0]),
             "n_eval": int(eval_blocks.shape[0]),
             "n_train_tokens": int(train_blocks.numel()),
             "n_eval_tokens": int(eval_blocks.numel()),
             "dataset_source": hub_id, "eval_split": eval_split}
    return train_dl, eval_dl, sizes


@torch.no_grad()
def evaluate_lm(model, eval_dl, device):
    """Token-weighted mean NLL -> perplexity. Returns (ppl, mean_nll, n_tokens).

    ⚠ TOKEN-WEIGHTED, NOT BATCH-AVERAGED. A mean of per-batch losses silently over-weights a short
    final batch. Blocks are equal-length here so the two agree, but the eval loader keeps its
    remainder (`drop_last=False`), so they would diverge the moment anyone changed the packing.
    """
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    for (ids,) in eval_dl:
        ids = ids.to(device)
        out = model(input_ids=ids, labels=ids)
        # HF averages over the (seq-1) predicted positions per sequence.
        n_pred = ids.shape[0] * (ids.shape[1] - 1)
        total_nll += float(out.loss) * n_pred
        total_tok += n_pred
    if was_training:
        model.train()
    mean_nll = total_nll / max(total_tok, 1)
    return math.exp(min(mean_nll, 20.0)), mean_nll, total_tok

# `stsb` is a REGRESSION task: one output, Pearson/Spearman, not accuracy. num_labels=1 is what
# switches LlamaForSequenceClassification's loss to MSE (train_glue.py:1614).
REGRESSION_TASKS = ("stsb",)


def _glue_metric_key(name):
    """The metric a task is scored by. Used to fill the single `task_metric_*` pair so a sweep can
    be read without knowing which task produced which column."""
    return {"cola": "matthews_correlation", "stsb": "pearson",
            "mrpc": "f1", "qqp": "f1"}.get(name, "accuracy")


def build_glue_data(name, tokenizer, args):
    """Load + tokenize one GLUE/SuperGLUE config. Returns (train_dl, eval_dl, num_labels, sizes)."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    from train_glue import task_to_keys                      # ONE definition of the key mapping

    if name not in task_to_keys:
        raise KeyError(f"unknown task {name!r}; known: {sorted(task_to_keys)}")
    s1, s2 = task_to_keys[name]
    if s1 is None:
        raise ValueError(f"glue:{name} is not a sentence-classification task "
                         f"(task_to_keys says {s1, s2}) -- use --task lm:* for the causal-LM sets")

    if name in SUPER_GLUE_TASKS:
        # ⚠ SuperGLUE's cache is SPLIT ACROSS TWO REPO NAMES on this project's disk, and neither
        #   resolves both configs: `boolq` is under `aps/super_glue`, `cb` under `super_glue`.
        #   datasets>=4 dropped script-based loading, so the bare name fails with
        #   "Couldn't find a module script" wherever only the script repo was cached. Try the
        #   candidates in order and RECORD which one answered, so the row says where its data
        #   came from instead of leaving it to be re-derived.
        raw = last = None
        for cand in ("aps/super_glue", "super_glue", "super_glue_mirror"):
            try:
                raw = load_dataset(cand, name)
                sg_source = cand
                break
            except Exception as e:                                  # noqa: BLE001
                last = f"{cand}: {type(e).__name__}: {str(e)[:120]}"
        if raw is None:
            raise RuntimeError(f"glue:{name} is SuperGLUE and no cached mirror resolved it. "
                               f"Last error -- {last}. Cache it with 02_download_cache.sh.")
    else:
        raw = load_dataset("glue", name)
        sg_source = "glue"
    is_regression = name in REGRESSION_TASKS
    num_labels = 1 if is_regression else len(raw["train"].features["label"].names)

    def _tok(batch):
        args_ = (batch[s1],) if s2 is None else (batch[s1], batch[s2])
        # ⚠ padding="max_length", NOT dynamic padding. Two reasons, both load-bearing here:
        #   (1) a memory/throughput number is only comparable across arms at a FIXED shape, and this
        #       runner reports peak memory in the same row as the metric;
        #   (2) several arms fix their shape at build time -- StreamBP's chunk is seq//3 and ALST's
        #       num_shards is ceil(seq/hidden) -- so a ragged batch would change the method's own
        #       configuration from step to step.
        out = tokenizer(*args_, padding="max_length", max_length=args.seq, truncation=True)
        out["labels"] = batch["label"]
        return out

    keep = ["input_ids", "attention_mask", "labels"]
    enc = raw.map(_tok, batched=True, remove_columns=raw["train"].column_names,
                  desc=f"tokenizing {name}")
    enc = enc.remove_columns([c for c in enc["train"].column_names if c not in keep])
    enc.set_format("torch")

    eval_split = "validation_matched" if name == "mnli" else "validation"
    train_ds, eval_ds = enc["train"], enc[eval_split]
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g,
                          collate_fn=default_data_collator, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch, shuffle=False,
                         collate_fn=default_data_collator)
    return train_dl, eval_dl, num_labels, {"n_train": len(train_ds), "n_eval": len(eval_ds),
                                           "eval_split": eval_split, "is_regression": is_regression,
                                           "dataset_source": sg_source}


def _to_device(batch, device, is_regression, dtype):
    """Move a batch to the device, casting REGRESSION labels to the model's dtype.

    ⚠ stsb's labels are `Value('float32')` while the model runs in bf16, and
    `LlamaForSequenceClassification` with num_labels==1 takes the MSELoss branch, which compares
    them directly: `RuntimeError: Found dtype Float but expected BFloat16`. Classification labels
    are int64 and must NOT be touched -- cross-entropy wants integer targets.
    """
    out = {k: v.to(device) for k, v in batch.items()}
    if is_regression and "labels" in out:
        out["labels"] = out["labels"].to(dtype)
    return out


def evaluate_glue(model, eval_dl, name, is_regression, device):
    """Run the eval split and score it with the task's OWN official metric.

    Also returns the PREDICTION DISTRIBUTION. That is not decoration: the classic silent failure of
    a decoder classifier on a small GLUE task is a model that collapses to one class and scores the
    majority baseline (0.527 on RTE, 0.684 on MRPC) -- a number that looks like learning. A
    degenerate predictor is only visible in the label histogram, so it travels in the row.
    """
    import evaluate as hf_evaluate
    from collections import Counter

    metric = hf_evaluate.load("super_glue" if name in SUPER_GLUE_TASKS else "glue", name)
    model.eval()
    preds_hist, n, loss_sum = Counter(), 0, 0.0
    with torch.no_grad():
        for batch in eval_dl:
            batch = _to_device(batch, device, is_regression, _model_dtype(model))
            out = model(**batch)
            logits = out.logits
            preds = logits.squeeze(-1).float() if is_regression else logits.argmax(dim=-1)
            metric.add_batch(predictions=preds, references=batch["labels"])
            if not is_regression:
                preds_hist.update(preds.tolist())
            if getattr(out, "loss", None) is not None:
                loss_sum += float(out.loss) * batch["labels"].shape[0]
            n += batch["labels"].shape[0]
    model.train()
    scores = metric.compute()
    return scores, (loss_sum / max(n, 1)), dict(sorted(preds_hist.items())), n


def run_one(args) -> dict:
    """Build -> verify engagement -> train -> evaluate -> return one CSV row dict."""
    t_start = time.time()
    method, with_fb = args.method, bool(args.fb)

    arm = resolve_arm(method, with_fb)                     # raises CombinationRefused if illegal
    if method in FLCE_FORBIDDEN and args.flce:
        raise ValueError(f"{method}: --flce forbidden (its own chunked LM head IS the fused CE)")

    is_glue = args.task.startswith("glue:")
    glue_name = args.task.split(":", 1)[1] if is_glue else None
    # `lm:synthetic` stays what it always was -- a fixed random batch, computational metrics only.
    # `lm:<corpus>` is REAL text with a real perplexity. Both are causal-LM heads.
    lm_name = args.task.split(":", 1)[1] if args.task.startswith("lm:") else None
    is_lm_corpus = lm_name is not None and lm_name != "synthetic"
    if lm_name is not None and not is_lm_corpus and lm_name != "synthetic":
        raise KeyError(f"unknown lm task {lm_name!r}; known: synthetic, {sorted(LM_CORPORA)}")
    if is_lm_corpus and lm_name not in LM_CORPORA:
        raise KeyError(f"unknown lm corpus {lm_name!r}; known: {sorted(LM_CORPORA)}")
    # Refuse an impossible head BEFORE tokenising a dataset or loading a model (registry
    # SEQ_CLS_REFUSES). A refusal is a correct, cheap outcome; an AttributeError 40 frames into
    # StreamBP's backward is not.
    resolve_head(method, "seq_cls" if is_glue else "causal_lm")
    resolve_fb_variant(method, with_fb, args.fb_variant)
    if is_glue and args.flce:
        raise ValueError(
            "--flce is meaningless on a GLUE head: Liger FusedLinearCrossEntropy fuses the LM head "
            "with a vocab-sized cross-entropy, and a sequence-classification head has neither. "
            "Refusing rather than recording flce=1 on a run that cannot use it.")

    args.seq, seq_source = resolve_seq(args.seq, is_glue, glue_name, is_lm_corpus)

    # ⚠ A QUALITY BENCHMARK MAY NOT EXCEED THE MODEL'S CONTEXT WINDOW. Past it, RoPE extrapolates
    #   and the perplexity is nonsense while looking like an ordinary number: measured 149.918 at
    #   seq 4096 against 8.171 at seq 2048 on TinyLlama (max_position_embeddings=2048), untrained.
    #   Nothing else in the harness catches this -- the run trains, the loss is finite, the row
    #   passes every anomaly check. Refuse instead, BEFORE the model is built.
    #   `lm:synthetic` is exempt: it measures memory and time on random tokens and explicitly does
    #   not report a quality number, which is what makes the §4.1 table's seq 16384 rows valid.
    if is_lm_corpus:
        from transformers import AutoConfig
        _maxpos = getattr(AutoConfig.from_pretrained(args.model), "max_position_embeddings", None)
        if _maxpos and args.seq > _maxpos:
            raise ValueError(
                f"--task lm:{lm_name} at --seq {args.seq} exceeds {args.model}'s "
                f"max_position_embeddings={_maxpos}. The perplexity would be a RoPE-extrapolation "
                f"artifact, not a quality measurement (measured on TinyLlama: 149.918 at 4096 vs "
                f"8.171 at 2048, untrained). Use --seq <= {_maxpos}, or a longer-context model. "
                f"For memory/throughput at longer shapes use `--task lm:synthetic`, which reports "
                f"no quality number.")
    torch.manual_seed(args.seed)
    cfg = _cfg(args)
    device = args.device
    # ⚠ `torch.cuda.max_memory_allocated()` with no argument reads the CURRENT device, which is
    # cuda:0 regardless of where the model was placed -- a run on cuda:1 reported peak=0.00 and
    # floor=0.00 in the first smoke test. Bind the process to the requested device so every
    # memory statistic below refers to the card the model is actually on.
    if str(device).startswith('cuda'):
        # ⚠ REPORT THE STATE OF THE CARD IF THIS FAILS. `torch.cuda.set_device` only CREATES THE
        #   CONTEXT -- it allocates nothing of ours -- so a `CUDA error: out of memory` here is not
        #   this run exhausting the device. It is the context itself failing to be created, and the
        #   causes are environmental: another tenant holding the GPU (processes outside our PID
        #   namespace are invisible to `nvidia-smi`, §1), a cgroup/host-memory limit, an exclusive
        #   compute mode, or a driver/runtime mismatch. It happened on fir job 54759272 at t+4s on a
        #   card `nvidia-smi` had reported as 5 MiB used with no processes, 25 s after
        #   `fir_assert_env gpu` had successfully created a context in another process on the same
        #   node -- i.e. it was NOT reproducible from the log alone. A bare traceback there costs a
        #   whole submit/queue round trip, so print what a diagnosis needs, then re-raise unchanged.
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            import subprocess
            print(f"\nCUDA INIT FAILED on {device}: {type(e).__name__}: {e}", flush=True)
            print(f"  visible devices     : {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
            try:
                print(f"  torch.cuda.device_count: {torch.cuda.device_count()}", flush=True)
            except Exception as e2:
                print(f"  device_count unavailable: {type(e2).__name__}: {e2}", flush=True)
            print(f"  PYTORCH_ALLOC_CONF  : {os.environ.get('PYTORCH_ALLOC_CONF', '<unset>')}", flush=True)
            print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}",
                  flush=True)
            print(f"  torch {torch.__version__} / cuda {torch.version.cuda}", flush=True)
            try:
                free, total = torch.cuda.mem_get_info(0)
                print(f"  mem_get_info(0)     : free={free/2**20:.0f} MiB total={total/2**20:.0f} MiB",
                      flush=True)
            except Exception as e2:
                print(f"  mem_get_info unavailable: {type(e2).__name__}: {e2}", flush=True)
            for cmd in (["nvidia-smi", "--query-gpu=index,memory.used,memory.total,compute_mode",
                         "--format=csv,noheader"],
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]):
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
                    print(f"  {' '.join(cmd[1:2])}: {out or '<none>'}", flush=True)
                except Exception:
                    pass
            raise

    # ---- task data FIRST: the head's shape is a property of the task ----
    tokenizer = train_dl = eval_dl = None
    task_info = {}
    if is_glue:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dl, eval_dl, num_labels, task_info = build_glue_data(glue_name, tokenizer, args)
        cfg.update({"head": "seq_cls", "num_labels": num_labels, "task_name": glue_name,
                    "pad_token_id": tokenizer.pad_token_id})
    elif is_lm_corpus:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dl, eval_dl, task_info = build_lm_data(lm_name, tokenizer, args)
        cfg.update({"head": "causal_lm", "task_name": lm_name})

    # ---- regime B, if asked for. ⚠ MUST PRECEDE build_model: `_apply_flce` monkey-patches the
    #      model CLASS, so a model already constructed keeps the stock fp32-logits CE.
    #      ⚠⚠ THIS CALL WAS MISSING UNTIL 2026-08-14. `--flce` was parsed, checked against
    #      FLCE_FORBIDDEN and written into the row as `flce=1`, but never applied -- so every row
    #      this runner had produced, including the fir preflight's four, was regime A wearing a
    #      regime B label. CONTEXT.md §8: regime B is where competitor claims are adjudicated, and
    #      the CE stack is 875 MiB at seq 1024/batch 2, larger than everything the block saves.
    flce_receipt = None
    if args.flce:
        # ⚠ `liger_kernel` is installed into the `temp/liger_pkgs` --target prefix, NOT the venv,
        #   so that a competitor's package cannot perturb any other arm's measurement. Put it on
        #   the path here rather than let `_apply_flce` die on ModuleNotFoundError.
        _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "temp", "liger_pkgs")
        if os.path.isdir(_lp) and _lp not in sys.path:
            sys.path.insert(0, _lp)
        try:
            from profile_unsloth import _apply_flce
            flce_receipt = _apply_flce(args.model)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"--flce needs liger_kernel, which lives in {_lp} (a pip --target prefix, not the "
                f"venv). Missing: {e}. Run sbatch/fir/01c_stage_repos.sh, or drop --flce and say "
                f"regime A in the writeup -- do NOT record flce=1 on a run that did not use it.")

    # The fused-block variant. `fb_min` is the shipped artifact; `wstream` (WP-E) additionally
    # streams the FROZEN base weights from pinned host memory, which removes a flat ~1932 MiB from
    # the resident floor at every sequence length. On GLUE shapes the floor IS the peak (87.6% of
    # it at seq 128), so this is the variant a peak-memory comparison wants. Costs ~2 GiB of pinned
    # HOST memory and single-digit % step time. Authority: memory_compression.md WP-E.
    fb_variant = args.fb_variant

    if arm == "" or arm == "_fb":                          # the `baseline` family
        if with_fb:
            base = ("fb_min_wstream_fnorm_sdpa" if fb_variant == "wstream"
                    else "fb_min_fnorm_sdpa")
        else:
            base = "gc_manual_sdpa"
        model = ph.build_model(base, cfg, device, adapter_dtype="bf16")
        arm_str = base
    else:
        # A composed arm carries the variant in its own name; `build_model` parses `wstream` at the
        # top of the dispatcher, so `lomo_fb_wstream` reaches `apply_flash_block` with it enabled.
        if with_fb and fb_variant == "wstream":
            arm = arm + "_wstream"
        model = ph.build_model(arm, cfg, device, adapter_dtype="bf16")
        arm_str = arm
    if with_fb and fb_variant == "wstream":
        import fb_wstream as _fbw
        _st = _fbw.fb_wstream_stats()
        # ⚠ VERIFY FROM THE STREAMER, NEVER FROM THE ARM NAME (memory_compression.md). An arm
        #   called `_wstream` whose streamer never installed is a full-residency run wearing a
        #   streaming label, and its floor would silently be 2 GiB too high.
        if not (_st.get("installed") and _st.get("on")):
            raise EngagementFailure(
                f"fb_variant=wstream requested but the streamer did not install: {_st}. "
                f"This row would claim a memory mechanism that never ran.")

    constituents = [method] + (["fb"] if with_fb else [])
    engagement = verify_engagement(model, constituents, strict=not args.allow_inert)

    # ---- train: the harness's own step(), so the recipe matches every measured row ----
    if is_glue or is_lm_corpus:
        batch = None                                       # real batches come from the dataloader
    else:
        vocab = ph.hf_config(model).vocab_size
        batch = ph.make_batch(cfg, device, vocab)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if getattr(model, "_galore_groups", None) is not None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "temp", "galore"))
        from galore_torch import GaLoreAdamW
        opt = GaLoreAdamW(model._galore_groups, lr=args.lr)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)

    # Clipping is read by `ph.step` off the model, so it reaches every arm through the one harness
    # rather than a second code path. lomo/adalomo consume it in `build_lomo_model` instead (their
    # clipping is inside the fused backward) and DeepSpeed takes it via `gradient_clipping`.
    model._max_grad_norm = args.max_grad_norm

    def _batches():
        """The step source. `lm:synthetic` re-uses ONE fixed batch (that is deliberate -- it makes
        the computational measurement shape-exact and method-independent). GLUE streams real
        examples and re-opens the loader when the epoch ends."""
        if not (is_glue or is_lm_corpus):
            while True:
                yield batch
        if is_lm_corpus:
            # Packed blocks: input_ids ARE the labels; LlamaForCausalLM shifts internally.
            #
            # ⚠ `attention_mask` FOR StreamBP ONLY, AND THAT IS THEIR CONTRACT, NOT A WORKAROUND.
            #   StreamBP records the mask through a wrapper that only fires when the caller passes
            #   one (`stream_model.py:427-428`), so omitting it leaves `stream_buffer` without the
            #   key and their attention raises `KeyError: 'attention_mask'` at `:353`. Their own
            #   driver passes `torch.ones_like(input_ids)` (`scripts/test_bp.py:27,52`).
            #   ⚠ NOT passed to the other arms on purpose: these blocks are packed and unpadded, so
            #   an all-ones mask is semantically a no-op, and `build_model` deliberately relies on
            #   transformers handing `attention_mask=None` to the sdpa path -- "exactly what that
            #   path wants" for the `hyclora_flash`/`fb_*` arms. Passing one everywhere would
            #   perturb arms that are already measured, to no benefit.
            _needs_mask = type(model).__name__ == "StreamModel"
            while True:
                for (ids,) in train_dl:
                    ids = ids.to(device)
                    b = {"input_ids": ids, "labels": ids}
                    if _needs_mask:
                        b["attention_mask"] = torch.ones_like(ids)
                    yield b
        while True:
            for b in train_dl:
                yield _to_device(b, device, task_info.get("is_regression", False), _mdtype)

    _mdtype = _model_dtype(model)
    src = _batches()

    # ---- the step budget -------------------------------------------------------------------
    # `--epochs` is the camera-ready control: a task is trained for E passes over its OWN train
    # split, so a small task (cb, 250 rows) and a large one (qnli, 105k) each get a comparable
    # amount of learning instead of an arbitrary shared step cap. `--train_steps` remains the
    # explicit override for smoke tests and for `lm:synthetic`, where there is no epoch.
    steps_per_epoch = len(train_dl) if (is_glue or is_lm_corpus) else 0
    if (is_glue or is_lm_corpus) and args.epochs:
        total_steps = max(1, steps_per_epoch * args.epochs)
    else:
        total_steps = args.train_steps
    eval_every = steps_per_epoch if ((is_glue or is_lm_corpus) and args.epochs) else 0

    # ⚠ WHAT `warmup_steps` MEANS DIFFERS BY TASK, AND THE DIFFERENCE IS DELIBERATE.
    #   On `lm:synthetic` these are throwaway steps on a fixed batch, purely to reach steady state
    #   before timing (protocol §A.2). On GLUE they are REAL OPTIMIZATION STEPS on real examples --
    #   discarding them would discard training. They are excluded from the timing/memory statistics
    #   and counted in `train_steps_total`, so the quality number reflects every step taken.
    # ---- LR schedule -----------------------------------------------------------------------
    # ⚠ Built over `total_steps` ONLY -- the `warmup_steps` above are measurement warm-up on
    #   lm:synthetic and real optimization on GLUE, but in neither case are they part of the
    #   published recipe's schedule. `constant` with `warmup_ratio=0` is a no-op and leaves the
    #   optimizer exactly as every previously-measured row had it.
    sched = None
    if args.lr_scheduler != "constant" or args.warmup_ratio > 0:
        if type(model).__module__.startswith("deepspeed") or getattr(model, "_lomo_opt", None):
            # Both own their update path; a torch LRScheduler wrapped around `opt` would not be
            # consulted. Refuse rather than silently schedule nothing -- that is exactly the
            # "recorded but never reached the kernel" failure this file keeps hitting.
            raise NotImplementedError(
                f"--lr_scheduler/--warmup_ratio are not wired for {method!r}: it owns its own "
                f"optimizer (DeepSpeed engine / LOMO fused backward), so a torch scheduler over "
                f"`opt` would be recorded and never applied. Implement it in that arm's builder "
                f"or leave the schedule constant and disclose it.")
        from transformers import get_scheduler
        sched = get_scheduler(args.lr_scheduler, optimizer=opt,
                              num_warmup_steps=int(args.warmup_ratio * total_steps),
                              num_training_steps=total_steps)

    for _ in range(args.warmup_steps):
        ph.step(model, next(src), opt)
    torch.cuda.synchronize()
    import gc as _gc
    _gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    resident_before = torch.cuda.memory_allocated()

    times, peaks_a, peaks_r, losses = [], [], [], []
    best = None
    _metric_key = _glue_metric_key(glue_name) if is_glue else None
    for _i in range(total_steps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = ph.step(model, next(src), opt)
        if sched is not None:
            sched.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks_a.append(torch.cuda.max_memory_allocated())
        peaks_r.append(torch.cuda.max_memory_reserved())
        if loss is not None:
            losses.append(float(loss))
        # ⚠ EVALUATE AT EACH EPOCH BOUNDARY AND KEEP THE BEST. End-of-training is the wrong
        #   estimator on the small GLUE tasks -- rte (2.5k) and cb (250) overfit within a couple of
        #   epochs, so the last epoch is routinely worse than the third. `train_glue.py` reports a
        #   best-over-epochs metric, and a row here that reported the final epoch would not be
        #   comparable with it. The eval runs under no_grad and is excluded from `times`.
        if eval_every and (_i + 1) % eval_every == 0 and (_i + 1) < total_steps:
            if is_lm_corpus:
                # ⚠ PERPLEXITY IS LOWER-IS-BETTER. The GLUE path keeps the MAXIMUM metric; using
                #   that comparator here would keep the WORST epoch and report it as the best.
                _p, _el, _n = evaluate_lm(model, eval_dl, device)
                if best is None or _p < best[0]:
                    best = (_p, {"perplexity": _p}, _el, None, _n, (_i + 1) // eval_every)
            else:
                _sc, _el, _ph_, _n = evaluate_glue(model, eval_dl, glue_name,
                                                   task_info["is_regression"], device)
                _v = _sc.get(_metric_key)
                if _v is not None and (best is None or _v > best[0]):
                    best = (_v, _sc, _el, _ph_, _n, (_i + 1) // eval_every)

    # ---- evaluate ----
    task_scores, pred_hist, ppl = {}, None, None
    if is_glue:
        task_scores, eval_loss, pred_hist, n_scored = evaluate_glue(
            model, eval_dl, glue_name, task_info["is_regression"], device)
        final_v = task_scores.get(_metric_key)
        if best is not None and final_v is not None and best[0] > final_v:
            # An earlier epoch was better: report it, and record WHICH, so the row says whether the
            # number came from the end of training or from a peak the run then walked away from.
            task_scores, eval_loss, pred_hist, n_scored = best[1], best[2], best[3], best[4]
            task_info["best_epoch"] = best[5]
        else:
            task_info["best_epoch"] = (total_steps // eval_every) if eval_every else None
        task_info["n_scored"] = n_scored
        task_info["steps_per_epoch"] = steps_per_epoch
    elif is_lm_corpus:
        ppl, eval_loss, n_scored = evaluate_lm(model, eval_dl, device)
        if best is not None and best[0] < ppl:
            ppl, eval_loss, n_scored = best[0], best[2], best[4]
            task_info["best_epoch"] = best[5]
        else:
            task_info["best_epoch"] = (total_steps // eval_every) if eval_every else None
        # The corpus's OWN headline metric, so a sweep reads without knowing the task family.
        task_scores = {"perplexity": ppl, "eval_nll": eval_loss,
                       "n_eval_tokens_scored": n_scored,
                       "n_train_tokens": task_info.get("n_train_tokens")}
        task_info["steps_per_epoch"] = steps_per_epoch
    else:
        model.eval()
        with torch.no_grad():
            out = model(**batch)
            eval_loss = float(out.loss)
        model.train()
        ppl = float(torch.exp(torch.tensor(min(eval_loss, 20.0))))

    times_s = sorted(times)
    receipt = {}
    for attr in ("_qlora_receipt", "_minis_receipt", "_streambp_receipt", "_alst_receipt",
                 "_zero3_receipt", "_galore_receipt", "_lomo_receipt", "_ckpt_receipt"):
        r = getattr(model, attr, None)
        if r:
            receipt[attr.strip("_")] = r
    cfg_json = json.dumps(receipt, sort_keys=True, default=str)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": args.run_id, "method": method, "with_fb": int(with_fb), "arm": arm_str,
        "task": args.task, "seed": args.seed, "model_name_or_path": args.model,
        "regime": "full_ft" if method in FULL_FT_REGIME else "peft_lora",
        "exact": int(method not in LOSSY),
        "seq_len": args.seq, "batch_size": args.batch, "grad_accum": 1,
        # ⚠ `lr` HERE IS THE REQUESTED VALUE. For arms that own their optimizer the AUTHORITATIVE
        #   value is the builder's receipt (`lomo_receipt.lr`, `zero3_receipt`), which is why both
        #   now read cfg["lr"] -- before that they hardcoded their own and this column lied.
        "train_steps": total_steps, "lr": args.lr,
        "max_grad_norm": args.max_grad_norm, "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs if (is_glue or is_lm_corpus) else None,
        "seq_source": seq_source,
        "steps_per_epoch": task_info.get("steps_per_epoch"),
        "best_epoch": task_info.get("best_epoch"),
        "lora_r": args.lora_r, "lora_alpha": args.lora_r, "lora_dropout": 0.0,
        "target_modules": ",".join(ph.FB_TARGETS), "adapter_dtype": "bf16",
        "gc_variant": (receipt.get("ckpt_receipt") or {}).get("variant", "none"),
        "fb_variant": fb_variant if with_fb else "none",
        "wstream_json": json.dumps(_wstream_stats(with_fb, fb_variant), sort_keys=True, default=str),
        "flce": int(bool(args.flce)),
        "attn_implementation": "sdpa",
        # ⚠ Under ZeRO-3 a parameter is PARTITIONED, so `p.numel()` returns the local shard (and 0
        # for a fully offloaded one) -- the smoke test reported 5,529,600 trainable params for a
        # LoRA arm whose true count is 12,615,680. DeepSpeed keeps the real size on `p.ds_numel`.
        "n_trainable_params": sum(getattr(p, "ds_numel", None) or p.numel()
                                  for p in model.parameters() if p.requires_grad),
        "eval_loss": eval_loss, "perplexity": ppl,
        "accuracy": task_scores.get("accuracy"),
        "f1": task_scores.get("f1"),
        "matthews_correlation": task_scores.get("matthews_correlation"),
        "pearson": task_scores.get("pearson"),
        "spearmanr": task_scores.get("spearmanr"),
        # ⚠ `perplexity` here is LOWER-IS-BETTER, unlike every GLUE metric that shares this
        #   column. Anything that ranks or gates on `task_metric` must branch on
        #   `task_metric_name`; `check_row`'s majority-class check already guards on
        #   `task_metric_name == "accuracy"` and so is unaffected.
        "task_metric_name": (_glue_metric_key(glue_name) if is_glue
                             else ("perplexity" if is_lm_corpus else None)),
        "task_metric": (task_scores.get(_glue_metric_key(glue_name)) if is_glue
                        else (ppl if is_lm_corpus else None)),
        "task_metric_json": (json.dumps(task_scores, sort_keys=True, default=str)
                             if (is_glue or is_lm_corpus) else None),
        "pred_distribution": json.dumps(pred_hist) if pred_hist is not None else None,
        "n_train_examples": task_info.get("n_train"),
        "n_eval_examples": task_info.get("n_scored", task_info.get("n_eval")),
        "eval_split": task_info.get("eval_split"),
        "dataset_source": task_info.get("dataset_source"),
        "train_steps_total": args.warmup_steps + total_steps,
        "train_step_peak_alloc_mib": max(peaks_a) / 2 ** 20,
        "train_step_peak_reserved_mib": max(peaks_r) / 2 ** 20,
        "resident_floor_mib": resident_before / 2 ** 20,
        "peak_minus_floor_mib": (max(peaks_a) - resident_before) / 2 ** 20,
        "ms_per_step_median": 1e3 * statistics.median(times_s),
        "ms_per_step_iqr": 1e3 * (times_s[int(0.75 * len(times_s))] - times_s[int(0.25 * len(times_s))])
        if len(times_s) >= 4 else 0.0,
        "total_train_time_sec": time.time() - t_start,
        "host_pinned_mib": (receipt.get("zero3_receipt") or {}).get("host_pinned_mib"),
        "engagement_ok": int(bool(engagement.get("engagement_ok"))),
        "engagement_json": json.dumps(engagement, sort_keys=True, default=str),
        "method_config_json": cfg_json,
        "config_hash": hashlib.sha1(
            f"{method}|{with_fb}|{args.task}|{args.seq}|{args.batch}|{args.lr}|{cfg_json}"
            .encode()).hexdigest()[:16],
        "torch_version": torch.__version__, "harness": "run_production.py",
        "notes": REGISTRY[method].notes,
    }
    return row


def write_row(csv_path: str, row: dict) -> bool:
    """Lock-protected atomic upsert (FileLock 300s, 5 retries, temp-file + os.replace).

    ⚠ Imported from `results_csv`, NOT from `train_glue`. Importing `train_glue` here pulled
    galore_torch / lion_pytorch / adapters into the write path and destroyed three fir preflight
    jobs whose training had already succeeded. Read `results_csv`'s docstring before fanning
    hundreds of writers at one CSV -- the lock is `fcntl.flock` and Lustre can make it node-local."""
    from results_csv import write_result_row
    return write_result_row(csv_path, CSV_COLUMNS, COMB_COLS, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--fb", action="store_true", help="compose with the fused decoder block")
    ap.add_argument("--task", default="lm:synthetic")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--device", default="cuda:0")
    # ⚠ default None, resolved by `resolve_seq`: 1024 for lm:*, and the per-task GLUE policy
    #   above otherwise (128, or 384 for boolq). Passing --seq explicitly still wins.
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--warmup_steps", type=int, default=8)
    # ⚠ THESE DEFAULT TO "OFF", WHICH IS THE BEHAVIOUR EVERY ALREADY-MEASURED ROW WAS TAKEN
    #   UNDER. They exist because a published LR is not a published recipe: LOMO ships
    #   `clip_grad_norm 1.0` + `warmup 0.1` + linear (args_lomo.yaml:23-25) and QLoRA ships
    #   `max_grad_norm 0.3` + constant (qlora.py:205,208). Quoting their LR while silently
    #   dropping the rest is the detuning §8 forbids. Set them per-arm; record them in the row.
    ap.add_argument("--max_grad_norm", type=float, default=None,
                    help="gradient-norm clipping. ⚠ On lomo/adalomo this enables THEIR two-pass "
                         "protocol (an extra backward per step, lomo.py:165-168) -- it is their "
                         "recipe, but it changes the throughput and memory columns, so say so.")
    ap.add_argument("--lr_scheduler", default="constant",
                    choices=["constant", "linear", "cosine"],
                    help="LR schedule over the training steps. `constant` reproduces every row "
                         "measured before this flag existed.")
    ap.add_argument("--warmup_ratio", type=float, default=0.0,
                    help="fraction of total steps spent warming the LR up from 0")
    ap.add_argument("--train_steps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=None,
                    help="GLUE: train for E passes over the task's own train split (camera-ready "
                         "control). Overrides --train_steps. Evaluates each epoch, keeps the best.")
    ap.add_argument("--fb_variant", default="min", choices=["min", "wstream"],
                    help="which fused-block variant --fb selects: `min` (shipped artifact) or "
                         "`wstream` (WP-E, streams the frozen base weights; best peak memory)")
    ap.add_argument("--flce", action="store_true",
                    help="regime B: Liger FusedLinearCrossEntropy. Causal-LM tasks only.")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="truncate the GLUE train split (smoke tests)")
    ap.add_argument("--max_eval_samples", type=int, default=None,
                    help="truncate the GLUE eval split (smoke tests)")
    ap.add_argument("--allow_inert", action="store_true",
                    help="DEBUG ONLY: write the row even if a method proved no work")
    ap.add_argument("--run_id", default="adhoc")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    if args.task.startswith("mc:"):
        raise NotImplementedError(
            f"{args.task}: the commonsense/MMLU suites need the registered "
            f"AutoModelForMultipleChoice, which `profile_hyclora.load_base_model` does not yet "
            f"construct. GLUE is wired (`--task glue:<config>`); this is the same change one head "
            f"further. Refusing rather than measuring the wrong head.")

    try:
        row = run_one(args)
    except CombinationRefused as e:
        print(f"REFUSED  {args.method}+fb={args.fb}: {e}")
        return 0                                   # a refusal is a correct outcome, not a failure
    ok = write_row(args.out_csv, row)
    score = (f"{row['task_metric_name']}={row['task_metric']:.4f}"
             if row.get("task_metric") is not None else
             f"ppl={row['perplexity']:.3f}" if row.get("perplexity") is not None else "score=n/a")
    print(f"{'OK ' if ok else 'CSV-FAIL '} {args.method}{'+fb' if args.fb else '':<4} "
          f"peak={row['train_step_peak_alloc_mib']:.2f} floor={row['resident_floor_mib']:.2f} "
          f"{score} preds={row.get('pred_distribution')} "
          f"engaged={row['engagement_ok']} -> {args.out_csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
