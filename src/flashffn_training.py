"""
FlashFFN Training Validation

Train a small transformer model with FlashFFN and compare:
1. Loss convergence vs standard FFN
2. Memory usage during training
3. Throughput (tokens/second)
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
from typing import Dict, List
from tqdm import tqdm

# Import our implementations
import sys
sys.path.insert(0, 'src')
from flashffn import FlashFFN, StandardFFN


class RMSNorm(nn.Module):
    """RMSNorm as used in LLaMA."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings."""
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
    """Multi-head attention with RoPE."""
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
    """Transformer block with configurable FFN."""
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_seq_len: int = 2048,
        use_flash_ffn: bool = False,
        k_fraction: float = 0.3,
    ):
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


class SmallTransformer(nn.Module):
    """Small transformer for training validation."""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 2048,
        max_seq_len: int = 512,
        use_flash_ffn: bool = False,
        k_fraction: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, intermediate_dim,
                max_seq_len, use_flash_ffn, k_fraction
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SyntheticDataset(Dataset):
    """Synthetic dataset for quick training validation."""
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        # Pre-generate all data for consistency
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:],
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_interval: int = 50,
) -> Dict:
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0
    step = 0
    losses = []
    throughputs = []

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        batch_start = time.time()

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_time = time.time() - batch_start
        tokens_per_sec = input_ids.numel() / batch_time

        total_loss += loss.item()
        step += 1

        if step % log_interval == 0:
            avg_loss = total_loss / step
            losses.append(avg_loss)
            throughputs.append(tokens_per_sec)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'tok/s': f'{tokens_per_sec:.0f}'})

    training_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    return {
        'final_loss': total_loss / step,
        'losses': losses,
        'peak_memory_gb': peak_memory,
        'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
        'training_time_s': training_time,
        'total_steps': step,
    }


def run_training_comparison():
    """Compare standard FFN vs FlashFFN in training."""
    print("=" * 70)
    print("FLASHFFN TRAINING VALIDATION")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float32  # Use FP32 for stable training comparison

    # Model config (small model for quick validation)
    vocab_size = 10000
    hidden_dim = 512
    num_layers = 6
    num_heads = 8
    intermediate_dim = 1408
    max_seq_len = 256

    # Training config
    batch_size = 8
    num_samples = 2000  # Enough for meaningful comparison
    learning_rate = 3e-4

    print(f"\nModel config:")
    print(f"  hidden_dim={hidden_dim}, layers={num_layers}, intermediate={intermediate_dim}")
    print(f"\nTraining config:")
    print(f"  batch_size={batch_size}, samples={num_samples}, lr={learning_rate}")

    # Create dataset
    dataset = SyntheticDataset(vocab_size, max_seq_len, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    results = {}

    configs = [
        {'name': 'Standard FFN', 'use_flash_ffn': False, 'k_fraction': 1.0},
        {'name': 'FlashFFN k=0.3', 'use_flash_ffn': True, 'k_fraction': 0.3},
        {'name': 'FlashFFN k=0.5', 'use_flash_ffn': True, 'k_fraction': 0.5},
    ]

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Training: {config['name']}")
        print("=" * 70)

        gc.collect()
        torch.cuda.empty_cache()

        model = SmallTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            max_seq_len=max_seq_len,
            use_flash_ffn=config['use_flash_ffn'],
            k_fraction=config['k_fraction'],
        ).to(device=device, dtype=dtype)

        print(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        metrics = train_epoch(model, dataloader, optimizer, device)

        print(f"\nResults:")
        print(f"  Final loss: {metrics['final_loss']:.4f}")
        print(f"  Peak memory: {metrics['peak_memory_gb']:.2f} GB")
        print(f"  Avg throughput: {metrics['avg_throughput']:.0f} tokens/sec")
        print(f"  Training time: {metrics['training_time_s']:.1f}s")

        results[config['name']] = metrics

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary comparison
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON SUMMARY")
    print("=" * 70)

    baseline = results['Standard FFN']

    print(f"\n{'Method':<20} {'Loss':<10} {'Memory':<12} {'Throughput':<15} {'vs Baseline'}")
    print("-" * 75)

    for name, metrics in results.items():
        loss_ratio = metrics['final_loss'] / baseline['final_loss']
        mem_ratio = metrics['peak_memory_gb'] / baseline['peak_memory_gb']
        tp_ratio = metrics['avg_throughput'] / baseline['avg_throughput']

        print(f"{name:<20} {metrics['final_loss']:<10.4f} {metrics['peak_memory_gb']:<12.2f}GB "
              f"{metrics['avg_throughput']:<15.0f} "
              f"loss:{loss_ratio:.2f}x mem:{mem_ratio:.2f}x tp:{tp_ratio:.2f}x")

    # Verify success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)

    flash_results = results['FlashFFN k=0.3']

    loss_converges = abs(flash_results['final_loss'] - baseline['final_loss']) < 0.5
    memory_reduction = (1 - flash_results['peak_memory_gb'] / baseline['peak_memory_gb']) * 100
    throughput_penalty = (1 - flash_results['avg_throughput'] / baseline['avg_throughput']) * 100

    print(f"\n1. Loss convergence: {flash_results['final_loss']:.4f} vs baseline {baseline['final_loss']:.4f}")
    print(f"   {'PASS' if loss_converges else 'FAIL'} - Training converges similarly")

    print(f"\n2. Memory reduction: {memory_reduction:.1f}%")
    print(f"   {'PASS' if memory_reduction > 10 else 'FAIL'} - Target: >10% (note: 6 layers, larger models will show more savings)")

    print(f"\n3. Throughput penalty: {throughput_penalty:.1f}%")
    print(f"   {'PASS' if throughput_penalty < 20 else 'FAIL'} - Target: <20%")

    return results


def run_extended_training():
    """Extended training with more epochs to verify convergence."""
    print("\n" + "=" * 70)
    print("EXTENDED TRAINING TEST (3 epochs)")
    print("=" * 70)

    device = "cuda"
    dtype = torch.float32

    vocab_size = 10000
    hidden_dim = 512
    num_layers = 6
    num_heads = 8
    intermediate_dim = 1408
    max_seq_len = 256
    batch_size = 8
    num_samples = 5000
    num_epochs = 3

    dataset = SyntheticDataset(vocab_size, max_seq_len, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    configs = [
        {'name': 'Standard', 'use_flash_ffn': False},
        {'name': 'FlashFFN', 'use_flash_ffn': True, 'k_fraction': 0.3},
    ]

    all_losses = {c['name']: [] for c in configs}

    for config in configs:
        print(f"\n{config['name']}:")

        gc.collect()
        torch.cuda.empty_cache()

        model = SmallTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            max_seq_len=max_seq_len,
            use_flash_ffn=config.get('use_flash_ffn', False),
            k_fraction=config.get('k_fraction', 0.3),
        ).to(device=device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for epoch in range(num_epochs):
            metrics = train_epoch(model, dataloader, optimizer, device, log_interval=100)
            all_losses[config['name']].extend(metrics['losses'])
            print(f"  Epoch {epoch+1}: loss={metrics['final_loss']:.4f}")

        del model, optimizer

    # Compare final losses
    print("\n" + "-" * 50)
    print("Loss comparison after 3 epochs:")
    for name, losses in all_losses.items():
        if losses:
            print(f"  {name}: final={losses[-1]:.4f}")

    return all_losses


if __name__ == "__main__":
    results = run_training_comparison()

    # Also run extended test
    extended_results = run_extended_training()

    # Save all results
    all_results = {
        'single_epoch': {k: {
            'final_loss': v['final_loss'],
            'peak_memory_gb': v['peak_memory_gb'],
            'avg_throughput': v['avg_throughput'],
            'training_time_s': v['training_time_s'],
            'losses': v['losses'],
        } for k, v in results.items()},
        'extended_training': extended_results,
    }

    os.makedirs('/workspace/actcomp/results', exist_ok=True)
    with open('/workspace/actcomp/results/flashffn_training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to /workspace/actcomp/results/flashffn_training_results.json")
    print("=" * 70)
