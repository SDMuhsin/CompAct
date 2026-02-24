"""
Phase 4: Training Validation

Train a small transformer (125M params) on real data to validate:
1. Compression strategies preserve training convergence
2. Memory reduction is achieved
3. Throughput penalty is acceptable

Dataset: WikiText-103 or C4 subset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import math
import time
import gc
import os
import json
from tqdm import tqdm

# Import compression strategies
from optimized_compression import (
    BlockwiseTopKCompression,
    PerChannelQuantizedCompression,
    HybridCompression,
)


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
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        # Precompute cos/sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, hidden_dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.o_proj(attn_output)


class CompressedSwiGLUMLP(nn.Module):
    """
    SwiGLU MLP with configurable activation compression.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        compression: str = "none",
        compression_ratio: float = 0.3,
        quantization_bits: int = 8,
    ):
        super().__init__()
        self.compression = compression
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_gate = self.gate_proj(x)
        h_up = self.up_proj(x)
        h_act = F.silu(h_gate)
        h_mid = h_act * h_up

        if self.compression == "topk":
            y = BlockwiseTopKCompression.apply(
                h_mid, self.down_proj.weight, self.compression_ratio
            )
        elif self.compression == "quantize":
            y = PerChannelQuantizedCompression.apply(
                h_mid, self.down_proj.weight, self.quantization_bits
            )
        elif self.compression == "hybrid":
            y = HybridCompression.apply(
                h_mid, self.down_proj.weight, self.compression_ratio, self.quantization_bits
            )
        else:
            y = self.down_proj(h_mid)

        return y


class TransformerBlock(nn.Module):
    """Single transformer block with compressed FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        compression: str = "none",
        compression_ratio: float = 0.3,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attention = Attention(hidden_dim, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.mlp = CompressedSwiGLUMLP(
            hidden_dim, intermediate_dim, compression, compression_ratio
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class SmallLLaMA(nn.Module):
    """
    Small LLaMA-style model for training validation.

    ~125M parameters with:
    - 12 layers
    - 768 hidden dim
    - 12 attention heads
    - 2048 intermediate dim
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 2048,
        max_seq_len: int = 512,
        compression: str = "none",
        compression_ratio: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, intermediate_dim,
                compression, compression_ratio, max_seq_len
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextDataset(Dataset):
    """Simple text dataset for training."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Concatenate all texts and tokenize
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        # Create chunks of max_length + 1 (for next token prediction)
        for i in range(0, len(all_tokens) - max_length - 1, max_length):
            chunk = all_tokens[i:i + max_length + 1]
            if len(chunk) == max_length + 1:
                self.data.append(chunk)

        print(f"Created {len(self.data)} training samples from {len(all_tokens)} tokens")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def load_wikitext_data():
    """Load WikiText dataset with fallback to synthetic data."""
    try:
        from datasets import load_dataset

        print("Loading WikiText-2 dataset...")
        # Use wikitext-2 which is more reliable
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        texts = [item['text'] for item in dataset if len(item['text']) > 100]
        print(f"Loaded {len(texts)} text samples from WikiText-2")
        return texts
    except Exception as e:
        print(f"Could not load WikiText: {e}")
        print("Using synthetic text data instead...")

        # Generate synthetic language-like data
        words = ["the", "a", "is", "are", "was", "were", "have", "has", "been",
                 "being", "to", "from", "with", "and", "or", "but", "if", "then",
                 "this", "that", "these", "those", "it", "they", "we", "you", "he",
                 "she", "can", "could", "would", "should", "will", "shall", "may",
                 "might", "must", "need", "want", "like", "love", "hate", "think",
                 "know", "see", "hear", "feel", "make", "take", "give", "get", "go",
                 "come", "run", "walk", "talk", "say", "tell", "ask", "answer",
                 "one", "two", "three", "four", "five", "six", "seven", "eight",
                 "big", "small", "large", "tiny", "huge", "great", "good", "bad",
                 "new", "old", "young", "first", "last", "next", "long", "short",
                 "high", "low", "right", "left", "same", "different", "other",
                 "only", "just", "also", "very", "really", "quite", "too", "so",
                 "world", "time", "year", "people", "way", "day", "man", "woman",
                 "child", "life", "hand", "part", "place", "case", "week", "company",
                 "system", "program", "question", "work", "government", "number",
                 "night", "point", "home", "water", "room", "mother", "area", "money",
                 "story", "fact", "month", "lot", "right", "study", "book", "eye",
                 "job", "word", "business", "issue", "side", "kind", "head", "house"]

        import random
        texts = []
        # Generate longer texts to ensure we have enough tokens
        for _ in range(10000):
            length = random.randint(200, 500)
            text = " ".join(random.choices(words, k=length))
            texts.append(text)

        print(f"Generated {len(texts)} synthetic text samples")
        return texts


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    log_interval: int = 50,
) -> Dict:
    """Train model and return metrics."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    total_loss = 0
    step = 0
    losses = []
    throughputs = []

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch in epoch_pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            batch_start = time.time()

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
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
                epoch_pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "tok/s": f"{tokens_per_sec:.0f}"
                })

    training_time = time.time() - start_time

    # Get memory stats
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_memory = 0

    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

    return {
        "final_loss": total_loss / step,
        "losses": losses,
        "peak_memory_gb": peak_memory,
        "avg_throughput": avg_throughput,
        "training_time_s": training_time,
        "total_steps": step,
    }


def run_training_comparison():
    """Run training comparison between baseline and compressed models."""
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 for training stability, then measure with float16 separately
    dtype = torch.float32

    print("=" * 70)
    print("PHASE 4: TRAINING VALIDATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype} (for stability)")

    # Load data
    texts = load_wikitext_data()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=256)
    print(f"Dataset size: {len(dataset)} samples")

    # Training config
    batch_size = 8
    num_epochs = 1
    learning_rate = 3e-4

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Model configs
    model_configs = [
        {"compression": "none", "compression_ratio": 1.0, "name": "Baseline"},
        {"compression": "topk", "compression_ratio": 0.3, "name": "TopK-30%"},
        {"compression": "quantize", "compression_ratio": 1.0, "name": "Quant-8bit"},
        {"compression": "hybrid", "compression_ratio": 0.3, "name": "Hybrid-30%"},
    ]

    results = {}

    for config in model_configs:
        print(f"\n{'=' * 70}")
        print(f"Training: {config['name']}")
        print("=" * 70)

        # Create model
        model = SmallLLaMA(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            intermediate_dim=2048,
            max_seq_len=256,
            compression=config['compression'],
            compression_ratio=config['compression_ratio'],
        )

        print(f"Model parameters: {model.count_parameters() / 1e6:.1f}M")

        # Move model to device (keep float32 for stability)
        model = model.to(device)

        # Train
        metrics = train_model(
            model,
            dataloader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            log_interval=50,
        )

        print(f"\nResults for {config['name']}:")
        print(f"  Final loss: {metrics['final_loss']:.4f}")
        print(f"  Peak memory: {metrics['peak_memory_gb']:.2f} GB")
        print(f"  Avg throughput: {metrics['avg_throughput']:.0f} tokens/sec")
        print(f"  Training time: {metrics['training_time_s']:.1f}s")

        results[config['name']] = metrics

        # Clean up
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary comparison
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON SUMMARY")
    print("=" * 70)

    baseline_loss = results["Baseline"]["final_loss"]
    baseline_memory = results["Baseline"]["peak_memory_gb"]
    baseline_throughput = results["Baseline"]["avg_throughput"]

    print(f"\n{'Method':<20} {'Loss':<12} {'Memory (GB)':<15} {'Throughput':<15} {'vs Baseline'}")
    print("-" * 80)

    for name, metrics in results.items():
        loss_ratio = metrics["final_loss"] / baseline_loss
        mem_ratio = metrics["peak_memory_gb"] / baseline_memory if baseline_memory > 0 else 1.0
        throughput_ratio = metrics["avg_throughput"] / baseline_throughput if baseline_throughput > 0 else 1.0

        print(f"{name:<20} {metrics['final_loss']:<12.4f} {metrics['peak_memory_gb']:<15.2f} "
              f"{metrics['avg_throughput']:<15.0f} loss:{loss_ratio:.2f}x mem:{mem_ratio:.2f}x tp:{throughput_ratio:.2f}x")

    return results


def quick_validation():
    """Quick validation run with synthetic data for testing."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 for stability in quick validation
    dtype = torch.float32

    print("=" * 70)
    print("QUICK VALIDATION (Synthetic Data)")
    print("=" * 70)

    vocab_size = 1000
    batch_size = 4
    seq_len = 128
    num_steps = 50

    configs = [
        {"compression": "none", "name": "Baseline"},
        {"compression": "topk", "compression_ratio": 0.3, "name": "TopK-30%"},
        {"compression": "quantize", "name": "Quant-8bit"},
        {"compression": "hybrid", "compression_ratio": 0.3, "name": "Hybrid-30%"},
    ]

    results = {}

    for config in configs:
        print(f"\n{config['name']}:")

        model = SmallLLaMA(
            vocab_size=vocab_size,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            intermediate_dim=1408,
            max_seq_len=seq_len,
            compression=config.get('compression', 'none'),
            compression_ratio=config.get('compression_ratio', 0.3),
        ).to(device=device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR for stability

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        losses = []
        start_time = time.time()

        for step in range(num_steps):
            # Use language modeling style data (next token prediction)
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
            labels = input_ids[:, 1:].clone()
            input_ids = input_ids[:, :-1]

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % 10 == 0:
                print(f"  Step {step}: loss = {loss.item():.4f}")

        training_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0

        final_loss = sum(losses[-10:]) / 10
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  Time: {training_time:.1f}s")

        results[config['name']] = {
            "final_loss": final_loss,
            "peak_memory_gb": peak_memory,
            "training_time_s": training_time,
        }

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "-" * 50)
    print("QUICK VALIDATION SUMMARY")
    print("-" * 50)
    baseline_loss = results["Baseline"]["final_loss"]
    for name, data in results.items():
        ratio = data["final_loss"] / baseline_loss
        print(f"  {name}: loss={data['final_loss']:.4f} ({ratio:.2f}x), mem={data['peak_memory_gb']:.2f}GB")

    return results


if __name__ == "__main__":
    import os
    os.makedirs("/workspace/actcomp/results", exist_ok=True)

    # First run quick validation
    quick_results = quick_validation()

    # Then run full training comparison
    print("\n\n")
    full_results = run_training_comparison()

    # Save results
    all_results = {
        "quick_validation": quick_results,
        "full_training": full_results,
    }

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    with open("/workspace/actcomp/results/phase4_training_results.json", "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to /workspace/actcomp/results/phase4_training_results.json")
    print("=" * 70)
