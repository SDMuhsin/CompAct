"""
Breakeven Experiment: Empirically verify the FlashFFN memory breakeven formula.

Breakeven formula (recompute mode):
  N > (3 * D_int * D_hid) / (D_int * (4 - 2*k) - D_hid)
  where N = batch * seq_len, k = k_fraction

For TinyLlama (D_int=5632, D_hid=2048, k=0.3): breakeven N ≈ 2024

This script runs 4 adapters (LoRA, DoRA, AdaLoRA, DyLoRA) × 5 seq lengths × 2 modes
(baseline vs FlashFFN) and measures peak GPU memory after 20 training steps.
"""

import argparse
import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import adapters
from adapters import LoRAConfig
from peft import (
    LoraConfig as PeftLoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from dylora import get_dylora_model, DyLoRALinear
from flashffn import FlashFFN, FlashFFNFunction

# Import apply_flash_ffn from train_glue
from train_glue import apply_flash_ffn


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BATCH_SIZE = 16
NUM_STEPS = 20
K_FRACTION = 0.3
SEED = 41
DTYPE = torch.bfloat16
SEQ_LENGTHS = [64, 128, 256, 512, 1024]

# TinyLlama dims
D_HID = 2048
D_INT = 5632


def compute_breakeven(d_hid, d_int, k):
    """Theoretical breakeven N = batch * seq_len."""
    return (3 * d_int * d_hid) / (d_int * (4 - 2 * k) - d_hid)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_gpu():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_base_model(num_labels=2):
    """Load TinyLlama for sequence classification."""
    config = AutoConfig.from_pretrained(
        MODEL_NAME, num_labels=num_labels, finetuning_task="sst2",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(dtype=DTYPE)
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA via AdapterHub (activations mode)."""
    adapters.init(model)
    adapter_config = LoRAConfig(r=16, alpha=32, dropout=0.0)
    model.add_adapter("lora_adapter", config=adapter_config)
    model.train_adapter("lora_adapter")
    model.set_active_adapters("lora_adapter")
    model.to(dtype=DTYPE)
    return model


def apply_dora(model):
    """Apply DoRA via PEFT (recompute mode via effective weight)."""
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = PeftLoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
        use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    return model


def apply_adalora(model):
    """Apply AdaLoRA via PEFT (recompute mode via effective weight)."""
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = AdaLoraConfig(
        init_r=16, target_r=8, lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
        total_step=NUM_STEPS,
        tinit=5, tfinal=5, deltaT=2, orth_reg_weight=0.5,
    )
    model = get_peft_model(model, peft_config)
    return model


def apply_dylora(model):
    """Apply DyLoRA via custom implementation (recompute mode via effective weight)."""
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_dylora_model(
        model=model, target_modules=target_modules,
        r=16, alpha=32, dropout=0.0,
    )
    return model


def make_dummy_batch(seq_len, batch_size, vocab_size=32000, num_labels=2):
    """Create a dummy batch of input_ids, attention_mask, labels."""
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, num_labels, (batch_size,), dtype=torch.long)
    return input_ids, attention_mask, labels


def run_single_experiment(adapter_name, apply_adapter_fn, seq_len, use_flash, device):
    """Run a single experiment: load model, apply adapter, optionally FlashFFN, train N steps, return peak memory."""
    set_seed(SEED)
    clear_gpu()

    # Load fresh model
    model, tokenizer = load_base_model()

    # Apply adapter
    model = apply_adapter_fn(model)
    model.to(device)

    # Apply FlashFFN if requested
    flash_info = ""
    if use_flash:
        n_converted = apply_flash_ffn(model, k_fraction=K_FRACTION)
        flash_info = f" [FlashFFN: {n_converted} layers, k={K_FRACTION}]"

    model.train()

    # Create optimizer (only over trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4)

    # Reset memory stats after setup
    torch.cuda.reset_peak_memory_stats(device)

    # Training loop
    try:
        for step in range(NUM_STEPS):
            input_ids, attention_mask, labels = make_dummy_batch(seq_len, BATCH_SIZE)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mib = peak_mem_bytes / (1024 ** 2)
    except torch.cuda.OutOfMemoryError:
        peak_mem_mib = float('inf')  # OOM sentinel

    # Clean up
    del model, optimizer, trainable_params
    clear_gpu()

    return peak_mem_mib


def main():
    parser = argparse.ArgumentParser(description="FlashFFN breakeven experiment")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Run only this adapter (lora/dora/adalora/dylora). If None, run all.")
    parser.add_argument("--seq_lengths", type=str, default=None,
                        help="Comma-separated seq lengths. Default: 64,128,256,512,1024")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    seq_lengths = SEQ_LENGTHS
    if args.seq_lengths:
        seq_lengths = [int(s) for s in args.seq_lengths.split(",")]

    adapters_map = {
        "lora": ("LoRA (AdapterHub)", apply_lora, "activations"),
        "dora": ("DoRA (PEFT)", apply_dora, "recompute"),
        "adalora": ("AdaLoRA (PEFT)", apply_adalora, "recompute"),
        "dylora": ("DyLoRA (custom)", apply_dylora, "recompute"),
    }

    if args.adapter:
        if args.adapter not in adapters_map:
            print(f"Unknown adapter: {args.adapter}. Choose from: {list(adapters_map.keys())}")
            return
        adapters_map = {args.adapter: adapters_map[args.adapter]}

    breakeven_n = compute_breakeven(D_HID, D_INT, K_FRACTION)
    print(f"{'='*80}")
    print(f"FlashFFN Breakeven Experiment")
    print(f"Model: {MODEL_NAME} (D_hid={D_HID}, D_int={D_INT})")
    print(f"Batch: {BATCH_SIZE}, Steps: {NUM_STEPS}, k_fraction: {K_FRACTION}")
    print(f"Theoretical breakeven N: {breakeven_n:.0f}")
    print(f"{'='*80}")

    results = []

    for adapter_key, (adapter_label, apply_fn, mode) in adapters_map.items():
        print(f"\n{'─'*60}")
        print(f"Adapter: {adapter_label} (FlashFFN mode: {mode})")
        print(f"{'─'*60}")

        for seq_len in seq_lengths:
            N = BATCH_SIZE * seq_len
            predicted = "SAVE" if N > breakeven_n else "COST" if N < breakeven_n else "EVEN"

            # Baseline
            print(f"  seq={seq_len:>5}, N={N:>6} ({predicted:>4}) ... ", end="", flush=True)
            mem_base = run_single_experiment(adapter_key, apply_fn, seq_len, use_flash=False, device=device)
            base_str = "OOM" if mem_base == float('inf') else f"{mem_base:.1f}"
            print(f"baseline={base_str:>10}", end="", flush=True)

            # FlashFFN
            mem_flash = run_single_experiment(adapter_key, apply_fn, seq_len, use_flash=True, device=device)
            flash_str = "OOM" if mem_flash == float('inf') else f"{mem_flash:.1f}"

            if mem_base == float('inf') and mem_flash == float('inf'):
                print(f"  flash={flash_str:>10}  [BOTH OOM]")
                observed = "BOTH_OOM"
                delta = float('nan')
                delta_pct = float('nan')
            elif mem_base == float('inf'):
                print(f"  flash={flash_str:>10} MiB  [BASELINE OOM - FlashFFN fits!]")
                observed = "SAVED"
                delta = float('-inf')
                delta_pct = float('-inf')
            elif mem_flash == float('inf'):
                print(f"  flash={flash_str:>10}  [FLASH OOM]")
                observed = "COST"
                delta = float('inf')
                delta_pct = float('inf')
            else:
                delta = mem_flash - mem_base
                delta_pct = (delta / mem_base) * 100
                observed = "SAVED" if delta < 0 else "COST"
                print(f"  flash={mem_flash:>8.1f} MiB  Δ={delta:>+8.1f} MiB ({delta_pct:>+5.1f}%) [{observed}]")

            results.append({
                "adapter": adapter_key,
                "adapter_label": adapter_label,
                "mode": mode,
                "seq_len": seq_len,
                "N": N,
                "predicted": predicted,
                "baseline_mib": round(mem_base, 1) if mem_base != float('inf') else "OOM",
                "flash_mib": round(mem_flash, 1) if mem_flash != float('inf') else "OOM",
                "delta_mib": round(delta, 1) if not (math.isnan(delta) or math.isinf(delta)) else str(delta),
                "delta_pct": round(delta_pct, 2) if not (math.isnan(delta_pct) or math.isinf(delta_pct)) else str(delta_pct),
                "observed": observed,
            })

    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"SUMMARY TABLE")
    print(f"Breakeven N = {breakeven_n:.0f} (batch={BATCH_SIZE})")
    print(f"{'='*100}")
    print(f"{'Adapter':<20} {'seq':>5} {'N':>6} {'Predict':>8} {'Baseline':>10} {'FlashFFN':>10} {'Delta':>10} {'%':>7} {'Result':>8}")
    print(f"{'─'*20} {'─'*5} {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*7} {'─'*8}")

    for r in results:
        predicted = r["predicted"]
        observed = r["observed"]
        # For LoRA (activations mode), breakeven formula doesn't apply — always saves
        if r["mode"] == "activations":
            match = "✓" if observed == "SAVED" else "~" if observed == "BOTH_OOM" else "✗"
        else:
            match = "✓" if predicted == observed else "~" if "OOM" in observed else "✗"

        base_str = f"{r['baseline_mib']:>9.1f}M" if isinstance(r['baseline_mib'], (int, float)) else f"{'OOM':>10}"
        flash_str = f"{r['flash_mib']:>9.1f}M" if isinstance(r['flash_mib'], (int, float)) else f"{'OOM':>10}"
        delta_str = f"{r['delta_mib']:>+9.1f}M" if isinstance(r['delta_mib'], (int, float)) else f"{str(r['delta_mib']):>10}"
        pct_str = f"{r['delta_pct']:>+6.1f}%" if isinstance(r['delta_pct'], (int, float)) else f"{str(r['delta_pct']):>7}"

        print(f"{r['adapter_label']:<20} {r['seq_len']:>5} {r['N']:>6} {predicted:>8} "
              f"{base_str} {flash_str} {delta_str} {pct_str} {match:>8}")

    # Save results
    results_dir = Path("./results/breakeven")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "breakeven_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_NAME,
                "batch_size": BATCH_SIZE,
                "num_steps": NUM_STEPS,
                "k_fraction": K_FRACTION,
                "d_hid": D_HID,
                "d_int": D_INT,
                "breakeven_n": breakeven_n,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
