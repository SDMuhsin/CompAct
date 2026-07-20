"""
Standalone verification for the commonsense multiple-choice integration
(src/commonsense_mc.py). No training script, no full model download required
(uses a tiny random LLaMA for the head/collator test).

Run:
    source env/bin/activate
    HF_HOME=./data HF_DATASETS_CACHE=./data PYTHONPATH=src python src/verify_commonsense_mc.py
"""
import sys
import collections

import torch

import commonsense_mc as cm


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def test_parse_full_170k():
    banner("TEST 1: parse ALL of Commonsense-170K (expect 0 failures)")
    from datasets import load_dataset
    ds = load_dataset("zwhe99/commonsense_170k")["train"]
    instrs = ds["instruction"]
    answers = ds["answer"]
    n = len(instrs)
    fam_count = collections.Counter()
    choice_count = collections.Counter()
    label_ok = 0
    failures = []
    for i in range(n):
        try:
            ctx, chs, lab = cm.parse_commonsense_instruction(instrs[i], answers[i])
        except Exception as e:
            failures.append((i, answers[i], str(e)[:140]))
            continue
        import re
        fam = re.sub(r"\d+$", "", str(answers[i]).strip())
        fam_count[fam] += 1
        choice_count[len(chs)] += 1
        if 0 <= lab < len(chs):
            label_ok += 1
    print(f"rows parsed: {n - len(failures)}/{n}")
    print("family counts:", dict(fam_count))
    print("choice-count distribution:", dict(sorted(choice_count.items())))
    print("labels in range:", label_ok)
    if failures:
        print("FAILURES (first 10):")
        for f in failures[:10]:
            print("  ", f)
    assert not failures, f"{len(failures)} parse failures!"
    assert label_ok == n, "some labels out of range"
    # spot-check a few parsed examples
    print("\nSpot-check parsed examples:")
    for idx in [0, 9427, 60000, 130000, 169000]:
        ctx, chs, lab = cm.parse_commonsense_instruction(instrs[idx], answers[idx])
        print(f"  [{idx}] answer={answers[idx]!r} label={lab} #ch={len(chs)}")
        print(f"        ctx={ctx[:80]!r}")
        print(f"        gold={chs[lab][:80]!r}")
    print("TEST 1 PASS")


def test_eval_sets():
    banner("TEST 2: load all 8 eval sets -> (context, choices, label) columns")
    expected_nchoices = {"boolq": {2}, "piqa": {2}, "siqa": {3}, "hellaswag": {4},
                         "winogrande": {2}, "arc_e": {3, 4, 5}, "arc_c": {3, 4, 5}, "obqa": {4}}
    for name in cm.COMMONSENSE_EVAL_SETS:
        cols = cm._extract_eval_columns(name)
        nrows = len(cols["label"])
        ncset = collections.Counter(len(c) for c in cols["choices"])
        labmin, labmax = min(cols["label"]), max(cols["label"])
        # every label must index into its own choice list
        bad = [i for i in range(nrows) if not (0 <= cols["label"][i] < len(cols["choices"][i]))]
        assert not bad, f"{name}: {len(bad)} labels out of range"
        observed = set(ncset.keys())
        assert observed <= expected_nchoices[name], f"{name}: unexpected #choices {observed}"
        print(f"  {name:11s} rows={nrows:5d} nchoices={dict(sorted(ncset.items()))} "
              f"label_range=[{labmin},{labmax}]")
        print(f"       ex: ctx={cols['context'][0][:70]!r} gold={cols['choices'][0][cols['label'][0]][:50]!r}")
    print("TEST 2 PASS")


def test_head_and_collator():
    banner("TEST 3: MC head + collator (tiny random LLaMA, mixed #choices)")
    from transformers import LlamaConfig, AutoModelForMultipleChoice
    cm.register_mc_models()
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=512,
                      pad_token_id=0)
    model = AutoModelForMultipleChoice.from_config(cfg)
    print("resolved class:", type(model).__name__)
    assert type(model).__name__ == "LlamaForMultipleChoice"

    # Build a batch with DIFFERENT #choices per example (2, 4, 3) to exercise padding+masking.
    class _Tok:
        pad_token_id = 0
    feats = [
        {"input_ids": [[1, 2, 3], [1, 2, 4]], "attention_mask": [[1, 1, 1], [1, 1, 1]], "label": 1},
        {"input_ids": [[5, 6], [5, 7], [5, 8], [5, 9]], "attention_mask": [[1, 1], [1, 1], [1, 1], [1, 1]], "label": 3},
        {"input_ids": [[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], "attention_mask": [[1, 1, 1, 1]] * 3, "label": 0},
    ]
    collator = cm.DataCollatorForMultipleChoice(_Tok())
    batch = collator(feats)
    print("input_ids:", tuple(batch["input_ids"].shape), "-> [B, max_choices, max_len]")
    assert batch["input_ids"].shape == (3, 4, 4)
    assert batch["attention_mask"].shape == (3, 4, 4)
    # example 0 has only 2 real choices -> choices 2,3 must be all-pad
    assert batch["attention_mask"][0, 2].sum() == 0 and batch["attention_mask"][0, 3].sum() == 0
    assert batch["attention_mask"][2, 3].sum() == 0  # example 2 has 3 real choices

    out = model(**batch)
    logits = out.logits
    print("logits:", tuple(logits.shape), "loss:", float(out.loss))
    assert logits.shape == (3, 4)
    assert torch.isfinite(out.loss), "loss not finite"
    # padded choices must be masked to a very negative logit (never argmax-selected)
    assert logits[0, 2].item() < -1e30 and logits[0, 3].item() < -1e30
    assert logits[2, 3].item() < -1e30
    preds = logits.argmax(dim=-1)
    assert (preds < torch.tensor([2, 4, 3])).all(), "argmax selected a padded choice!"

    # backward -> the score head must receive gradient
    out.loss.backward()
    g = model.score.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "no grad to score head"
    print("score.weight grad norm:", float(g.norm()))
    print("TEST 3 PASS")


def test_tokenize_roundtrip():
    banner("TEST 4: tokenisation roundtrip with a real tokenizer (tiny eval subset)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = cm.load_commonsense_eval("siqa", tok, max_length=128, n_samples=8)
    print("columns:", ds.column_names, "rows:", len(ds))
    ex = ds[0]
    print("example #choices:", len(ex["input_ids"]), "label:", ex["label"])
    assert len(ex["input_ids"]) == 3  # siqa has 3 choices
    assert all(len(x) == len(m) for x, m in zip(ex["input_ids"], ex["attention_mask"]))
    collator = cm.DataCollatorForMultipleChoice(tok)
    batch = collator([ds[i] for i in range(4)])
    print("collated batch input_ids:", tuple(batch["input_ids"].shape))
    assert batch["input_ids"].shape[0] == 4 and batch["input_ids"].shape[1] == 3
    print("TEST 4 PASS")


if __name__ == "__main__":
    test_parse_full_170k()
    test_eval_sets()
    test_head_and_collator()
    test_tokenize_roundtrip()
    banner("ALL COMMONSENSE-MC VERIFICATION TESTS PASSED")
