"""
LLM-Adapters commonsense-reasoning suite as a MULTIPLE-CHOICE paradigm.

This module provides everything `train_glue.py` needs to train ONCE on the
Commonsense-170K mixture and evaluate (in the same job) on all 8 commonsense
benchmarks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge,
OpenBookQA), writing one result per eval set.

Why a custom multiple-choice head?
    transformers 4.51.3 ships NO `LlamaForMultipleChoice` / `MistralForMultipleChoice`
    (`MODEL_FOR_MULTIPLE_CHOICE_MAPPING` maps llama/mistral -> None). Only encoder
    models (BERT/RoBERTa) have MC heads, and those are not SwiGLU so FlashFFN cannot
    apply to them. We therefore implement a decoder-LM multiple-choice head here
    (last-token pooling + a single scalar score per choice, softmax over choices)
    and REGISTER it with `AutoModelForMultipleChoice`, so the literal
    `AutoModelForMultipleChoice.from_pretrained(...)` API works on a FlashFFN-compatible
    SwiGLU backbone (TinyLlama / LLaMA-7B / Mistral).

Data:
    Train  : `zwhe99/commonsense_170k` (170,420 rows, LLM-Adapters instruction format).
             Parsed per answer-family into (context, choices, label).
    Eval   : native parquet HF datasets (no loading scripts -> load under datasets 4.5.0),
             mapped to the SAME (context, choices, label) schema.

All datasets here are parquet/json-backed (NO python loading script), so they load
under datasets==4.5.0 and can be pre-cached for offline HPC nodes via
`sbatch/download_cache.sh`.
"""

import re
from typing import List, Tuple, Dict

import torch
from torch import nn

from datasets import load_dataset, Dataset
from transformers import LlamaConfig, MistralConfig, AutoModelForMultipleChoice
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralModel, MistralPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


###############################################################################
#                       Decoder-LM multiple-choice head                       #
###############################################################################
def _make_mc_class(pretrained_cls, base_model_cls, cls_name):
    """Build a `<Model>ForMultipleChoice` class for a decoder-LM backbone.

    Encodes each (context, choice) pair through the backbone, pools the LAST
    non-pad token's hidden state, projects it to a scalar via `self.score`, and
    softmaxes the per-choice scores. Padding choices (used to batch examples with
    different #choices) are detected as choices whose attention mask is all-zero
    and have their logit set to -inf BEFORE the cross-entropy / argmax, so they
    can never be selected and do not contribute to the loss.
    """

    class _ForMultipleChoice(pretrained_cls):
        def __init__(self, config):
            super().__init__(config)
            # attribute name "model" matches the *ForCausalLM checkpoint prefix
            # ("model.*"), so backbone weights load cleanly; only `score` is new.
            self.model = base_model_cls(config)
            self.score = nn.Linear(config.hidden_size, 1, bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def forward(self, input_ids=None, attention_mask=None, labels=None,
                    position_ids=None, inputs_embeds=None, return_dict=True, **kwargs):
            # input_ids: [B, C, L]  (C = num choices, padded per batch)
            if input_ids is not None:
                num_choices = input_ids.shape[1]
                flat_input_ids = input_ids.reshape(-1, input_ids.size(-1))
            else:
                num_choices = inputs_embeds.shape[1]
                flat_input_ids = None
            flat_attn = (attention_mask.reshape(-1, attention_mask.size(-1))
                         if attention_mask is not None else None)
            flat_pos = (position_ids.reshape(-1, position_ids.size(-1))
                        if position_ids is not None else None)
            flat_embeds = (inputs_embeds.reshape(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
                           if inputs_embeds is not None else None)

            outputs = self.model(
                input_ids=flat_input_ids,
                attention_mask=flat_attn,
                position_ids=flat_pos,
                inputs_embeds=flat_embeds,
            )
            hidden = outputs.last_hidden_state  # [B*C, L, H]

            # Pool the last NON-pad token (right padding -> index = #real_tokens - 1).
            if flat_attn is not None:
                lengths = (flat_attn.sum(dim=-1) - 1).clamp(min=0).long()  # [B*C]
            else:
                lengths = torch.full((hidden.size(0),), hidden.size(1) - 1,
                                     device=hidden.device, dtype=torch.long)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]  # [B*C, H]

            logits = self.score(pooled).squeeze(-1).view(-1, num_choices)  # [B, C]

            # Mask padding choices (all-pad sequence -> attention sums to 0).
            if flat_attn is not None:
                choice_real = (flat_attn.sum(dim=-1) > 0).view(-1, num_choices)  # [B, C]
                neg = torch.finfo(logits.dtype).min
                logits = torch.where(choice_real, logits, torch.full_like(logits, neg))

            loss = None
            if labels is not None:
                # CE in fp32 for numerical stability under bf16/fp16.
                loss = nn.functional.cross_entropy(logits.float(), labels.view(-1).long())

            return MultipleChoiceModelOutput(loss=loss, logits=logits)

    _ForMultipleChoice.__name__ = cls_name
    _ForMultipleChoice.__qualname__ = cls_name
    return _ForMultipleChoice


LlamaForMultipleChoice = _make_mc_class(LlamaPreTrainedModel, LlamaModel, "LlamaForMultipleChoice")
MistralForMultipleChoice = _make_mc_class(MistralPreTrainedModel, MistralModel, "MistralForMultipleChoice")


def register_mc_models():
    """Register the decoder-LM MC heads so `AutoModelForMultipleChoice` resolves
    LLaMA/Mistral configs to them. Idempotent (exist_ok=True)."""
    AutoModelForMultipleChoice.register(LlamaConfig, LlamaForMultipleChoice, exist_ok=True)
    AutoModelForMultipleChoice.register(MistralConfig, MistralForMultipleChoice, exist_ok=True)


###############################################################################
#                       Multiple-choice data collator                         #
###############################################################################
class DataCollatorForMultipleChoice:
    """Pads a batch of variable-#choice, variable-length MC examples.

    Each feature: {"input_ids": List[List[int]], "attention_mask": List[List[int]],
    "label": int}, where the outer list is over choices (length varies across the
    suite: 2 for boolq/piqa/winogrande, 3 for siqa, 4 for hellaswag/obqa, 3-5 for arc).

    Pads choices to the per-batch max with all-pad sequences (which the MC head
    detects and masks), and right-pads each sequence to the per-batch max length.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, features):
        labels = [f["label"] for f in features]
        choices = [f["input_ids"] for f in features]      # per-example: list[list[int]]
        masks = [f["attention_mask"] for f in features]

        batch_max_choices = max(len(c) for c in choices)
        max_len = max((len(seq) for c in choices for seq in c), default=1)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        B = len(features)
        input_ids = torch.full((B, batch_max_choices, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, batch_max_choices, max_len), dtype=torch.long)
        for i, (cc, mm) in enumerate(zip(choices, masks)):
            for j, (seq, ms) in enumerate(zip(cc, mm)):
                L = len(seq)
                input_ids[i, j, :L] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, j, :L] = torch.tensor(ms, dtype=torch.long)
            # choices j >= len(cc) stay all-pad (attention all 0) -> masked by the head.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


###############################################################################
#                 Commonsense-170K instruction-format parser                  #
###############################################################################
# The 170K mixture is in LLM-Adapters instruction format. Each family uses a fixed,
# machine-generated template; every instruction ends with "Answer format: ..." which
# is a universal split point separating the body from the answer-format hint.
_FAMILY_PREFIX = {
    "tf":       "Please answer the following question with true or false, question: ",
    "solution": "Please choose the correct solution to the question: ",
    "answer":   "Please choose the correct answer to the question: ",
    "ending":   "Please choose the correct ending to complete the given sentence: ",
    "option":   "Please choose the correct answer to fill in the blank to complete the given sentence: ",
}
_FAMILY_MARKER = {"solution": "Solution", "answer": "Answer", "ending": "Ending", "option": "Option"}


def parse_commonsense_instruction(instruction: str, answer: str) -> Tuple[str, List[str], int]:
    """Parse one Commonsense-170K row into (context, choices, label_idx).

    Deterministic; raises ValueError on any structural anomaly (no silent
    corruption). Families:
      true/false (boolq, 2 choices = ["true","false"], question-only/no passage),
      solution   (piqa, 2),
      answer     (siqa/arc/obqa, 3-5),
      ending     (hellaswag, 4),
      option     (winogrande, 2).
    """
    answer = answer.strip()
    fam = re.sub(r"\d+$", "", answer)  # true/false/solution/answer/ending/option

    if "Answer format:" not in instruction:
        raise ValueError(f"No 'Answer format:' marker in instruction: {instruction[:120]!r}")
    body = instruction.rsplit("Answer format:", 1)[0].rstrip()

    if fam in ("true", "false"):
        prefix = _FAMILY_PREFIX["tf"]
        if not body.startswith(prefix):
            raise ValueError(f"true/false body missing prefix: {body[:120]!r}")
        context = body[len(prefix):].strip()
        choices = ["true", "false"]
        label = 0 if fam == "true" else 1
    else:
        if fam not in _FAMILY_PREFIX:
            raise ValueError(f"Unknown answer family {fam!r} (answer={answer!r})")
        prefix = _FAMILY_PREFIX[fam]
        marker = _FAMILY_MARKER[fam]
        if not body.startswith(prefix):
            raise ValueError(f"{fam} body missing prefix: {body[:120]!r}")
        rest = body[len(prefix):]
        first_marker = f"{marker}1:"
        idx = rest.find(first_marker)
        if idx < 0:
            raise ValueError(f"{fam}: no '{first_marker}' found in {rest[:160]!r}")
        context = rest[:idx].strip()
        opts_block = rest[idx:]
        # Option markers are preceded by whitespace (inline " Answer2:" / newline
        # "\n\nSolution2:") or sit at the start of the block.
        positions = [(m.start(), int(m.group(1)))
                     for m in re.finditer(rf"(?:(?<=\s)|^){re.escape(marker)}(\d+):", opts_block)]
        if not positions:
            raise ValueError(f"{fam}: no option markers in {opts_block[:160]!r}")
        choices = []
        for k, (pos, num) in enumerate(positions):
            if num != k + 1:
                raise ValueError(f"{fam}: non-contiguous option index {num} at slot {k+1}")
            hdr = re.match(rf"{re.escape(marker)}\d+:\s*", opts_block[pos:])
            txt_start = pos + hdr.end()
            txt_end = positions[k + 1][0] if k + 1 < len(positions) else len(opts_block)
            choices.append(opts_block[txt_start:txt_end].strip())
        label = int(answer[len(fam):]) - 1

    # Structural validation (raise, never silently mislabel).
    if not context:
        raise ValueError(f"Empty context for answer={answer!r}: {instruction[:120]!r}")
    if len(choices) < 2:
        raise ValueError(f"<2 choices for answer={answer!r}: {choices}")
    if any(not c for c in choices):
        raise ValueError(f"Empty choice text for answer={answer!r}: {choices}")
    if not (0 <= label < len(choices)):
        raise ValueError(f"label {label} out of range for {len(choices)} choices (answer={answer!r})")
    return context, choices, label


###############################################################################
#                          Tokenisation (shared)                              #
###############################################################################
def _tokenize_mc_dataset(raw: Dataset, tokenizer, max_length: int, desc: str) -> Dataset:
    """Map a (context, choices, label) Dataset -> (input_ids, attention_mask, label),
    where input_ids/attention_mask are per-choice token lists (variable length)."""
    def _tok(examples):
        out_ids, out_mask = [], []
        for ctx, chs in zip(examples["context"], examples["choices"]):
            ids, masks = [], []
            for ch in chs:
                enc = tokenizer(ctx, ch, truncation=True, max_length=max_length)
                ids.append(enc["input_ids"])
                masks.append(enc["attention_mask"])
            out_ids.append(ids)
            out_mask.append(masks)
        return {"input_ids": out_ids, "attention_mask": out_mask, "label": examples["label"]}

    return raw.map(_tok, batched=True, remove_columns=["context", "choices"], desc=desc)


def load_commonsense_train(tokenizer, max_length: int, n_samples=None, shuffle_seed=None) -> Dataset:
    """Load + parse Commonsense-170K into a tokenised MC train set.

    `shuffle_seed`/`n_samples`: the 170K is ordered by family (boolq first ~9.4K),
    so for a smoke-test subset we shuffle BEFORE truncating to cover all families.
    """
    ds = load_dataset("zwhe99/commonsense_170k")["train"]
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _parse(examples):
        ctxs, chss, labs = [], [], []
        for instr, ans in zip(examples["instruction"], examples["answer"]):
            c, ch, lab = parse_commonsense_instruction(instr, ans)
            ctxs.append(c)
            chss.append(ch)
            labs.append(lab)
        return {"context": ctxs, "choices": chss, "label": labs}

    parsed = ds.map(_parse, batched=True, remove_columns=ds.column_names,
                    desc="Parsing commonsense-170k")
    return _tokenize_mc_dataset(parsed, tokenizer, max_length, desc="Tokenising commonsense-170k")


###############################################################################
#                      Native eval sets -> MC schema                          #
###############################################################################
# Eval-set order matches the LLM-Adapters commonsense table.
COMMONSENSE_EVAL_SETS = ["boolq", "piqa", "siqa", "hellaswag",
                         "winogrande", "arc_e", "arc_c", "obqa"]


def _extract_eval_columns(name: str) -> Dict[str, list]:
    """Load one native eval dataset (correct split) and map it to columns
    context / choices / label. Contexts mirror the 170K stem formatting so train
    and eval are featurised consistently."""
    if name == "boolq":
        d = load_dataset("google/boolq")["validation"]
        ctx = [ex["question"] for ex in d]                       # question only (no passage; matches 170K)
        chs = [["true", "false"] for _ in d]
        lab = [0 if ex["answer"] else 1 for ex in d]             # True->"true"(0), False->"false"(1)
    elif name == "piqa":
        d = load_dataset("nthngdy/piqa")["validation"]
        ctx = [ex["goal"] for ex in d]
        chs = [[ex["sol1"], ex["sol2"]] for ex in d]
        lab = [int(ex["label"]) for ex in d]                     # 0/1
    elif name == "siqa":
        d = load_dataset("lighteval/siqa")["validation"]
        ctx = [f'{ex["context"]} {ex["question"]}' for ex in d]
        chs = [[ex["answerA"], ex["answerB"], ex["answerC"]] for ex in d]
        lab = [int(ex["label"]) - 1 for ex in d]                 # "1"/"2"/"3" -> 0/1/2
    elif name == "hellaswag":
        d = load_dataset("Rowan/hellaswag")["validation"]
        ctx = [f'{ex["activity_label"]}: {ex["ctx"]}' for ex in d]
        chs = [list(ex["endings"]) for ex in d]
        lab = [int(ex["label"]) for ex in d]                     # "0".."3"
    elif name == "winogrande":
        d = load_dataset("allenai/winogrande", "winogrande_xl")["validation"]
        ctx = [ex["sentence"] for ex in d]                       # contains "_" blank
        chs = [[ex["option1"], ex["option2"]] for ex in d]
        lab = [int(ex["answer"]) - 1 for ex in d]                # "1"/"2" -> 0/1
    elif name in ("arc_e", "arc_c"):
        cfg = "ARC-Easy" if name == "arc_e" else "ARC-Challenge"
        d = load_dataset("allenai/ai2_arc", cfg)["test"]
        ctx, chs, lab = [], [], []
        for ex in d:
            texts, labels, ak = ex["choices"]["text"], ex["choices"]["label"], ex["answerKey"]
            if ak not in labels:
                raise ValueError(f"{name}: answerKey {ak!r} not in choice labels {labels}")
            ctx.append(ex["question"])
            chs.append(list(texts))
            lab.append(labels.index(ak))                         # letter/digit -> index via choices.label
    elif name == "obqa":
        d = load_dataset("allenai/openbookqa", "main")["test"]
        ctx, chs, lab = [], [], []
        for ex in d:
            texts, labels, ak = ex["choices"]["text"], ex["choices"]["label"], ex["answerKey"]
            if ak not in labels:
                raise ValueError(f"obqa: answerKey {ak!r} not in choice labels {labels}")
            ctx.append(ex["question_stem"])
            chs.append(list(texts))
            lab.append(labels.index(ak))
    else:
        raise ValueError(f"Unknown commonsense eval set: {name!r}")
    return {"context": ctx, "choices": chs, "label": lab}


def load_commonsense_eval(name: str, tokenizer, max_length: int, n_samples=None) -> Dataset:
    """Load one of the 8 commonsense eval sets as a tokenised MC dataset."""
    cols = _extract_eval_columns(name)
    if n_samples is not None:
        cols = {k: v[:n_samples] for k, v in cols.items()}
    raw = Dataset.from_dict(cols)
    return _tokenize_mc_dataset(raw, tokenizer, max_length, desc=f"Tokenising {name}")
