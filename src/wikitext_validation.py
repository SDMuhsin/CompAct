"""
WikiText-2 Training Validation for FlashFFN

Proper validation on real data with perplexity measurement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import gc
import time
import json
import math
import os
from typing import Dict, List, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, 'src')
from flashffn import FlashFFN, StandardFFN

# Set cache directories
os.environ['HF_HOME'] = '/workspace/actcomp/data'
os.environ['TORCH_HOME'] = '/workspace/actcomp/data'


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(S)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary(q, k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int,
                 max_seq_len: int = 2048, use_flash_ffn: bool = False, k_fraction: float = 0.3):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attention = Attention(hidden_dim, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(hidden_dim)
        if use_flash_ffn:
            self.ffn = FlashFFN(hidden_dim, intermediate_dim, k_fraction=k_fraction)
        else:
            self.ffn = StandardFFN(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SmallLM(nn.Module):
    """Small language model for WikiText validation."""
    def __init__(self, vocab_size: int, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, intermediate_dim: int = 1408, max_seq_len: int = 256,
                 use_flash_ffn: bool = False, k_fraction: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, intermediate_dim, max_seq_len, use_flash_ffn, k_fraction)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class WikiTextDataset(Dataset):
    """WikiText-2 dataset for language modeling."""
    def __init__(self, tokenizer, split: str = 'train', max_length: int = 256):
        # Map split names
        split_map = {'train': 'train', 'validation': 'valid', 'test': 'test'}
        split_name = split_map.get(split, split)

        # Load from local files
        data_dir = '/workspace/actcomp/data/wikitext'
        file_path = f'{data_dir}/wiki.{split_name}.txt'

        print(f"Loading WikiText-2 {split} split from {file_path}...")

        with open(file_path, 'r') as f:
            text = f.read()

        # Tokenize
        print("Tokenizing...")
        all_tokens = tokenizer.encode(text, add_special_tokens=False)

        # Create chunks
        self.data = []
        for i in range(0, len(all_tokens) - max_length, max_length):
            chunk = all_tokens[i:i + max_length + 1]
            if len(chunk) == max_length + 1:
                self.data.append(chunk)

        print(f"Created {len(self.data)} samples from {len(all_tokens):,} tokens")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long),
        }


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """Evaluate model and compute perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {'loss': avg_loss, 'perplexity': perplexity}


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, device: str) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    steps = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if steps % 50 == 0:
            pbar.set_postfix({'loss': f'{total_loss/steps:.4f}'})

    return total_loss / steps


def run_wikitext_validation():
    """Run full WikiText-2 validation."""
    from transformers import AutoTokenizer

    print("=" * 70)
    print("WIKITEXT-2 VALIDATION")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float32

    # Model config
    hidden_dim = 512
    num_layers = 6
    num_heads = 8
    intermediate_dim = 1408
    max_seq_len = 256

    # Training config
    batch_size = 16
    num_epochs = 3
    learning_rate = 5e-4

    print(f"\nModel: {num_layers} layers, {hidden_dim} hidden, {intermediate_dim} intermediate")
    print(f"Training: {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Load datasets
    train_dataset = WikiTextDataset(tokenizer, split='train', max_length=max_seq_len)
    val_dataset = WikiTextDataset(tokenizer, split='validation', max_length=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Configurations to test
    configs = [
        {'name': 'Standard FFN', 'use_flash_ffn': False, 'k_fraction': 1.0},
        {'name': 'FlashFFN k=0.3', 'use_flash_ffn': True, 'k_fraction': 0.3},
    ]

    results = {}

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Training: {config['name']}")
        print("=" * 70)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        model = SmallLM(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            max_seq_len=max_seq_len,
            use_flash_ffn=config['use_flash_ffn'],
            k_fraction=config['k_fraction'],
        ).to(device=device, dtype=dtype)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params/1e6:.1f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        train_losses = []
        val_metrics = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = train_epoch(model, train_loader, optimizer, device)
            train_losses.append(train_loss)

            val_result = evaluate(model, val_loader, device)
            val_metrics.append(val_result)

            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_result['loss']:.4f}, Val PPL: {val_result['perplexity']:.2f}")

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        results[config['name']] = {
            'train_losses': train_losses,
            'val_losses': [m['loss'] for m in val_metrics],
            'val_perplexities': [m['perplexity'] for m in val_metrics],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_metrics[-1]['loss'],
            'final_val_ppl': val_metrics[-1]['perplexity'],
            'peak_memory_gb': peak_memory,
        }

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Train Loss':<12} {'Val Loss':<12} {'Val PPL':<12} {'Memory'}")
    print("-" * 70)

    for name, r in results.items():
        print(f"{name:<20} {r['final_train_loss']:<12.4f} {r['final_val_loss']:<12.4f} "
              f"{r['final_val_ppl']:<12.2f} {r['peak_memory_gb']:.2f} GB")

    # Compute degradation
    std = results['Standard FFN']
    flash = results['FlashFFN k=0.3']

    ppl_diff = flash['final_val_ppl'] - std['final_val_ppl']
    ppl_pct = (ppl_diff / std['final_val_ppl']) * 100
    mem_savings = (1 - flash['peak_memory_gb'] / std['peak_memory_gb']) * 100

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)
    print(f"Perplexity difference: {ppl_diff:+.2f} ({ppl_pct:+.1f}%)")
    print(f"Memory savings: {mem_savings:.1f}%")
    print(f"\nAcceptable degradation (PPL diff < 5%)? {abs(ppl_pct) < 5}")

    # Save results
    os.makedirs('/workspace/actcomp/results', exist_ok=True)
    with open('/workspace/actcomp/results/wikitext_validation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to /workspace/actcomp/results/wikitext_validation.json")

    return results


if __name__ == "__main__":
    results = run_wikitext_validation()
