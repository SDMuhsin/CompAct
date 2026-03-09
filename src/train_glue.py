# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Finetuning 🤗 Transformers models for sequence-classification on GLUE, running the
same training five times with seeds 41-45 (“median-of-five”, Mo5).
After the five runs finish we log **only the median** task-performance numbers
to `./results/mo5_glue.csv`; ancillary metrics (memory, timing, …) come from the
**first** seed’s run. The “seed” column in the CSV is literally the string
`"41,42,43,44,45"`.
"""
import argparse
import builtins
import copy
import csv
import gc
import json
import logging
import math
import operator
import os
import random
import statistics
import time
from functools import reduce
from pathlib import Path
from typing import Dict, List

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Import GaLore optimizers (standard ones)
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import bitsandbytes as bnb

# Try to import GALE optimizers (optional, from custom fork)
try:
    from galore_torch import GALE_AdamW, GALE_Adafactor, GALE_AdamW8bit, SwiftGaLoreAdamW, GALE_Lion
    GALE_AVAILABLE = True
except ImportError:
    GALE_AVAILABLE = False
    # Provide dummy classes for when GALE is not available
    class _DummyOptimizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("GALE optimizers not available. Install galore-torch from custom fork.")
    GALE_AdamW = GALE_Adafactor = GALE_AdamW8bit = SwiftGaLoreAdamW = GALE_Lion = _DummyOptimizer

# Import Lion optimizer
from lion_pytorch import Lion

# Import AdapterHub
import adapters
from adapters import LoRAConfig, IA3Config, PrefixTuningConfig
from filelock import FileLock, Timeout

# Import PEFT library for DoRA, VeRA, FourierFT, and AdaLoRA
from peft import (
    LoraConfig as PeftLoraConfig,
    VeraConfig,
    FourierFTConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType
)

# Import GB-VeRA (our gradient-balanced VeRA implementation)
from gbvera import get_gbvera_model, GBVeraModel

# Import Spectral Adapter (Truncated DCT Factored Adaptation)
from spectral_adapter import get_spectral_adapter_model, SpectralAdapterModel

# Import DyLoRA (Dynamic Low-Rank Adaptation)
from dylora import get_dylora_model, DyLoRAModel, DyLoRALinear

# Import FlashFFN (activation-compressed SwiGLU FFN)
from flashffn import FlashFFNFunction



torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

###############################################################################
#                                   constants                                 #
###############################################################################
SEEDS: List[int] = [41, 42, 43, 44, 45]
CAUSAL_LM_TASKS = {"wikitext2", "wikitext103"}
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DEFAULT_RESULTS_FILE = os.path.join(RESULTS_DIR, "mo53_glue.csv")
_METRIC_FOR_TASK = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
    "cb": "f1",
    "wikitext2": "perplexity",
    "wikitext103": "perplexity",
}

###############################################################################
#                                   helpers                                   #
###############################################################################
logger = logging.getLogger(__name__)


def _primary_metric(task_name: str, metric_dict: dict) -> float:
    key = _METRIC_FOR_TASK.get(task_name, "accuracy")
    val = metric_dict.get(key, float("-inf"))
    if key == "perplexity":
        return -val if val != float("-inf") else float("-inf")
    return val


def _load_results_df(results_file: str, columns: List[str]) -> pd.DataFrame:
    if os.path.isfile(results_file):
        df = pd.read_csv(results_file)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns]
    return pd.DataFrame(columns=columns)


def _upsert_result(df: pd.DataFrame, comb_cols: List[str], row_dict: Dict) -> pd.DataFrame:
    mask = reduce(
        operator.and_, [(df[col] == row_dict[col]) for col in comb_cols], pd.Series(True, index=df.index)
    )
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    return df


###############################################################################
#                                   FlashFFN                                  #
###############################################################################
def _is_peft_wrapped(linear_module: nn.Module) -> bool:
    """Check if a linear module has been wrapped by PEFT (LoRA, DoRA, etc.)."""
    module_type = type(linear_module).__name__
    # PEFT wraps nn.Linear with classes like: Linear (peft.tuners.lora.layer),
    # AdaLoraLayer, etc. Check for base_layer attribute (PEFT convention).
    return hasattr(linear_module, 'base_layer')


def _make_flash_forward_raw(mlp: nn.Module, kf: float):
    """FlashFFN forward using raw weights (maximum optimization).

    Used when MLP projections are plain nn.Linear (no PEFT adapters).
    Routes through FlashFFNFunction custom autograd for compressed
    backward storage + tiled forward.
    """
    def flash_forward(x):
        return FlashFFNFunction.apply(
            x,
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
            kf,
        )
    return flash_forward


def _is_dora_wrapped(linear_module: nn.Module) -> bool:
    """Check if a linear module is wrapped with PEFT DoRA specifically."""
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'lora_magnitude_vector')
            and len(linear_module.lora_magnitude_vector) > 0)


def _compute_dora_effective_weight(dora_linear: nn.Module) -> torch.Tensor:
    """Compute the effective weight of a DoRA-wrapped linear layer.

    Produces W_eff = (m / ||W + s*BA||_detach) * (W + s*BA) where:
      - W is the frozen base weight
      - B, A are the LoRA low-rank matrices (trainable)
      - s is the LoRA scaling factor
      - m is the DoRA magnitude vector (trainable)
      - ||...||_detach means the norm is detached (matching PEFT's design)

    Gradients flow through m, B, and A. W is frozen.
    """
    adapter_name = dora_linear.active_adapters[0]

    # Base weight (frozen)
    base_weight = dora_linear.get_base_layer().weight          # [out, in]

    # LoRA delta: B @ A, scaled
    lora_A_weight = dora_linear.lora_A[adapter_name].weight    # [r, in]
    lora_B_weight = dora_linear.lora_B[adapter_name].weight    # [out, r]
    scaling = dora_linear.scaling[adapter_name]                 # scalar

    lora_delta = lora_B_weight @ lora_A_weight                 # [out, in]
    merged_weight = base_weight + scaling * lora_delta          # [out, in]

    # Weight norm (DETACHED — matches PEFT's DoRA implementation)
    weight_norm = merged_weight.detach().norm(p=2, dim=1)      # [out,]

    # Magnitude vector (trainable)
    magnitude = dora_linear.lora_magnitude_vector[adapter_name].weight  # [out,]

    # Effective weight (cast to base_weight dtype for cuBLAS compatibility)
    w_eff = (magnitude / weight_norm).unsqueeze(1) * merged_weight
    w_eff = w_eff.to(base_weight.dtype)

    return w_eff


def _make_flash_forward_dora(mlp: nn.Module, kf: float):
    """FlashFFN forward for DoRA-wrapped MLP projections.

    Computes DoRA effective weights (preserving gradients through LoRA A/B
    and magnitude vector), then passes them to FlashFFNFunction.apply() for
    full FlashFFN benefit: fused Triton SiLU*mul + top-K compressed backward.

    This uses FlashFFN's recompute mode (since effective weights have
    requires_grad=True), which saves x + compressed top-K activations.
    """
    def flash_forward(x):
        w_gate_eff = _compute_dora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_dora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_dora_effective_weight(mlp.down_proj)
        return FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, kf)
    return flash_forward


def _is_lora_wrapped(linear_module: nn.Module) -> bool:
    """Check if a linear module is wrapped with PEFT LoRA (but NOT DoRA, AdaLoRA, or DyLoRA).

    PEFT LoRA modules have base_layer and lora_A/lora_B as nn.ModuleDict.
    DoRA has non-empty lora_magnitude_vector. AdaLoRA has lora_E.
    DyLoRA (custom) stores lora_A/lora_B as nn.Parameter (not ModuleDict).
    We exclude all three.
    """
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'lora_A')
            and isinstance(getattr(linear_module, 'lora_A', None), nn.ModuleDict)
            and len(getattr(linear_module, 'lora_A', {})) > 0
            and hasattr(linear_module, 'lora_B')
            and isinstance(getattr(linear_module, 'lora_B', None), nn.ModuleDict)
            and len(getattr(linear_module, 'lora_B', {})) > 0
            and len(getattr(linear_module, 'lora_magnitude_vector', {})) == 0
            and not hasattr(linear_module, 'lora_E'))


def _compute_lora_effective_weight(lora_linear: nn.Module) -> torch.Tensor:
    """Compute the effective weight of a PEFT LoRA layer.

    Produces W_eff = base_weight + scaling * lora_B @ lora_A
    where:
      - base_weight is the frozen base weight [out, in]
      - lora_A [r, in], lora_B [out, r] are trainable
      - scaling = lora_alpha / r

    Gradients flow through lora_A and lora_B.
    """
    adapter_name = lora_linear.active_adapters[0]

    base_weight = lora_linear.get_base_layer().weight              # [out, in]
    lora_A_weight = lora_linear.lora_A[adapter_name].weight        # [r, in]
    lora_B_weight = lora_linear.lora_B[adapter_name].weight        # [out, r]
    scaling = lora_linear.scaling[adapter_name]                     # scalar

    delta_w = lora_B_weight @ lora_A_weight                        # [out, in]
    w_eff = base_weight + scaling * delta_w
    w_eff = w_eff.to(base_weight.dtype)

    return w_eff


def _make_flash_forward_lora(mlp: nn.Module, kf: float):
    """FlashFFN forward for PEFT LoRA-wrapped MLP projections.

    Computes LoRA effective weights (preserving gradients through lora_A and
    lora_B), then passes them to FlashFFNFunction.apply() for full FlashFFN
    benefit: fused Triton SiLU*mul + top-K compressed backward.

    Uses FlashFFN's recompute mode (since effective weights have
    requires_grad=True), which saves x + compressed top-K activations.
    """
    def flash_forward(x):
        w_gate_eff = _compute_lora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_lora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_lora_effective_weight(mlp.down_proj)
        return FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, kf)
    return flash_forward


def _is_adalora_wrapped(linear_module: nn.Module) -> bool:
    """Check if a linear module is wrapped with PEFT AdaLoRA (SVDLinear).

    AdaLoRA uses SVD-parameterized LoRA: lora_A (right singular vectors),
    lora_E (singular values), lora_B (left singular vectors).  The
    distinguishing attribute is lora_E, which regular LoRA/DoRA don't have.
    """
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'lora_E')
            and len(getattr(linear_module, 'lora_E', {})) > 0)


def _compute_adalora_effective_weight(svd_linear: nn.Module) -> torch.Tensor:
    """Compute the effective weight of an AdaLoRA SVDLinear layer.

    Produces W_eff = base_weight + lora_B @ (lora_A * lora_E) * scaling / ranknum
    where:
      - base_weight is the frozen base weight [out, in]
      - lora_A [r, in], lora_E [r, 1], lora_B [out, r] are trainable
      - ranknum [1] tracks effective rank (non-trainable)
      - scaling is lora_alpha (scalar)

    Gradients flow through lora_A, lora_B, and lora_E.  When AdaLoRA's
    rank allocator zeros entries in lora_E, those components vanish from
    delta_w automatically.
    """
    adapter_name = svd_linear.active_adapters[0]

    base_weight = svd_linear.get_base_layer().weight            # [out, in]
    lora_A = svd_linear.lora_A[adapter_name]                    # nn.Parameter [r, in]
    lora_E = svd_linear.lora_E[adapter_name]                    # nn.Parameter [r, 1]
    lora_B = svd_linear.lora_B[adapter_name]                    # nn.Parameter [out, r]
    scaling = svd_linear.scaling[adapter_name]                   # scalar
    ranknum = svd_linear.ranknum[adapter_name]                   # nn.Parameter [1]

    delta_w = lora_B @ (lora_A * lora_E)                        # [out, in]
    w_eff = base_weight + delta_w * scaling / (ranknum + 1e-5)
    w_eff = w_eff.to(base_weight.dtype)

    return w_eff


def _make_flash_forward_adalora(mlp: nn.Module, kf: float):
    """FlashFFN forward for AdaLoRA-wrapped MLP projections.

    Computes AdaLoRA effective weights (preserving gradients through lora_A,
    lora_B, and lora_E), then passes them to FlashFFNFunction.apply() for
    full FlashFFN benefit: fused Triton SiLU*mul + top-K compressed backward.

    Uses FlashFFN's recompute mode (since effective weights have
    requires_grad=True), which saves x + compressed top-K activations.
    """
    def flash_forward(x):
        w_gate_eff = _compute_adalora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_adalora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_adalora_effective_weight(mlp.down_proj)
        return FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, kf)
    return flash_forward


def _is_dylora_wrapped(linear_module: nn.Module) -> bool:
    """Check if a linear module is a DyLoRA adapter (our custom DyLoRALinear)."""
    return isinstance(linear_module, DyLoRALinear)


def _compute_dylora_effective_weight(dylora_linear) -> torch.Tensor:
    """Compute the effective weight of a DyLoRALinear layer.

    Produces W_eff = base_weight + scaling * lora_B[:, :b] @ lora_A[:b, :]
    where:
      - base_weight is the frozen base weight [out, in]
      - lora_A [r, in], lora_B [out, r] are trainable
      - b = random rank sample during training, b = r during eval

    Gradients flow through lora_A and lora_B (the sliced portions).
    """
    base_weight = dylora_linear.base_layer.weight               # [out, in]
    lora_A = dylora_linear.lora_A                               # nn.Parameter [r, in]
    lora_B = dylora_linear.lora_B                               # nn.Parameter [out, r]
    scaling = dylora_linear.scaling                              # scalar

    if dylora_linear.training:
        b = random.randint(1, dylora_linear.r)
    else:
        b = dylora_linear.r

    delta_w = lora_B[:, :b] @ lora_A[:b, :]                    # [out, in]
    w_eff = base_weight + scaling * delta_w
    w_eff = w_eff.to(base_weight.dtype)

    return w_eff


def _make_flash_forward_dylora(mlp: nn.Module, kf: float):
    """FlashFFN forward for DyLoRA-wrapped MLP projections.

    Computes DyLoRA effective weights (with rank sampling during training),
    then passes them to FlashFFNFunction.apply() for full FlashFFN benefit:
    fused Triton SiLU*mul + top-K compressed backward.

    Each projection independently samples its own rank b, matching the
    original DyLoRA behavior.  FlashFFNFunction saves the computed effective
    weights, so backward uses the exact same weights (no re-sampling issue).
    """
    def flash_forward(x):
        w_gate_eff = _compute_dylora_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_dylora_effective_weight(mlp.up_proj)
        w_down_eff = _compute_dylora_effective_weight(mlp.down_proj)
        return FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, kf)
    return flash_forward


def _is_vera_wrapped(linear_module: nn.Module) -> bool:
    """Check if a linear module is wrapped with PEFT VeRA.

    VeRA layers have vera_lambda_d and vera_lambda_b (trainable scaling vectors)
    plus vera_A and vera_B (shared frozen random matrices stored as BufferDicts).
    The vera_lambda_d attribute is unique to VeRA — no other PEFT adapter has it.
    """
    return (hasattr(linear_module, 'base_layer')
            and hasattr(linear_module, 'vera_lambda_d')
            and len(getattr(linear_module, 'vera_lambda_d', {})) > 0)


def _compute_vera_effective_weight(vera_linear: nn.Module) -> torch.Tensor:
    """Compute the effective weight of a PEFT VeRA layer.

    Produces W_eff = base_weight + (lambda_b * sliced_B) @ (lambda_d * sliced_A)
    where:
      - base_weight is the frozen base weight [out, in]
      - vera_A [r, max_in] and vera_B [max_out, r] are shared frozen random matrices
      - sliced_A = vera_A[:, :in_features]  [r, in]
      - sliced_B = vera_B[:out_features, :] [out, r]
      - lambda_d [r] and lambda_b [out] are the trainable scaling vectors

    Gradients flow through lambda_d and lambda_b.  vera_A and vera_B are buffers
    (non-trainable).  VeRA dropout is NOT applied (same tradeoff as DoRA;
    default vera_dropout=0.0).
    """
    adapter_name = vera_linear.active_adapters[0]

    base_weight = vera_linear.get_base_layer().weight             # [out, in]
    in_features = vera_linear.in_features
    out_features = vera_linear.out_features

    vera_A = vera_linear.vera_A[adapter_name]                     # [r, max_in]
    vera_B = vera_linear.vera_B[adapter_name]                     # [max_out, r]
    lambda_d = vera_linear.vera_lambda_d[adapter_name]            # [r]
    lambda_b = vera_linear.vera_lambda_b[adapter_name]            # [out]

    sliced_A = vera_A[:, :in_features].to(lambda_d.device)        # [r, in]
    sliced_B = vera_B[:out_features, :].to(lambda_d.device)       # [out, r]

    delta_w = (lambda_b.unsqueeze(-1) * sliced_B) @ (lambda_d.unsqueeze(-1) * sliced_A)  # [out, in]
    w_eff = base_weight + delta_w
    w_eff = w_eff.to(base_weight.dtype)

    return w_eff


def _make_flash_forward_vera(mlp: nn.Module, kf: float):
    """FlashFFN forward for VeRA-wrapped MLP projections.

    Computes VeRA effective weights (preserving gradients through lambda_d and
    lambda_b), then passes them to FlashFFNFunction.apply() for full FlashFFN
    benefit: fused Triton SiLU*mul + top-K compressed backward.

    Uses FlashFFN's recompute mode (since effective weights have
    requires_grad=True), which saves x + compressed top-K activations.
    """
    def flash_forward(x):
        w_gate_eff = _compute_vera_effective_weight(mlp.gate_proj)
        w_up_eff = _compute_vera_effective_weight(mlp.up_proj)
        w_down_eff = _compute_vera_effective_weight(mlp.down_proj)
        return FlashFFNFunction.apply(x, w_gate_eff, w_up_eff, w_down_eff, kf)
    return flash_forward


def _make_flash_forward_module(mlp: nn.Module, kf: float):
    """FlashFFN forward using module calls (PEFT-compatible fallback).

    Uses gradient checkpointing to avoid storing MLP intermediate
    activations (h_gate, h_up, h_act, h_mid) for the backward pass.
    Only the MLP input is saved; intermediates are recomputed during
    backward.  This preserves adapter gradient flow while reducing
    peak memory by ~4×intermediate_dim×batch_seq×dtype_bytes per layer.

    Note: k_fraction (kf) is not used here — top-K compression is only
    available in raw-weight mode (FlashFFNFunction) or DoRA mode.
    """
    def _mlp_fn(x_2d):
        h_gate = mlp.gate_proj(x_2d)
        h_up = mlp.up_proj(x_2d)
        h_act = torch.nn.functional.silu(h_gate)
        h_mid = h_act * h_up
        return mlp.down_proj(h_mid)

    def flash_forward(x):
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        batch_seq = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]

        x_2d = x.reshape(batch_seq, hidden_dim)
        y = torch.utils.checkpoint.checkpoint(
            _mlp_fn, x_2d, use_reentrant=False
        )

        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], hidden_dim)
        return y

    return flash_forward


def apply_flash_ffn(model: nn.Module, k_fraction: float = 0.3) -> int:
    """
    Replace the forward method of SwiGLU MLP modules with FlashFFN.

    Walks the model tree looking for modules that have gate_proj, up_proj, and
    down_proj attributes (the signature of a SwiGLU MLP). For each one found,
    the module's forward method is monkey-patched to use tiled processing.

    Seven modes (checked in this order):
      1. **DoRA mode** (PEFT DoRA-wrapped projections): Computes DoRA effective
         weights (magnitude * normalized merged weight) preserving gradients,
         then passes to FlashFFNFunction.apply. Full FlashFFN benefit.
      2. **AdaLoRA mode** (PEFT SVDLinear projections): Computes AdaLoRA
         effective weights (lora_B @ (lora_A * lora_E) * scaling / ranknum)
         preserving gradients, then passes to FlashFFNFunction.apply.
         Full FlashFFN benefit (Triton kernels + top-K compression).
      3. **LoRA mode** (PEFT LoRA-wrapped projections): Computes LoRA effective
         weights (base + scaling * B @ A) preserving gradients, then passes to
         FlashFFNFunction.apply. Full FlashFFN benefit.
      4. **DyLoRA mode** (custom DyLoRALinear projections): Computes DyLoRA
         effective weights (with rank sampling) preserving gradients, then
         passes to FlashFFNFunction.apply. Full FlashFFN benefit.
      5. **VeRA mode** (PEFT VeRA-wrapped projections): Computes VeRA effective
         weights (base + (lambda_b * B) @ (lambda_d * A)) preserving gradients,
         then passes to FlashFFNFunction.apply. Full FlashFFN benefit.
      6. **Grad-checkpoint fallback** (other PEFT-wrapped projections): Wraps
         the MLP forward in torch.utils.checkpoint. Does NOT use FlashFFN's
         top-K compression. Fallback for unsupported PEFT methods.
      7. **Raw-weight mode** (plain nn.Linear projections): Uses
         FlashFFNFunction.apply for full optimization — fused Triton SiLU*mul +
         compressed (top-K) backward storage. Fastest path.

    Args:
        model: The HuggingFace model (possibly wrapped in PEFT).
        k_fraction: Fraction of intermediate activations to keep (0..1).

    Returns:
        Number of MLP modules converted.
    """
    converted = 0
    for name, module in model.named_modules():
        if (hasattr(module, 'gate_proj') and hasattr(module, 'up_proj')
                and hasattr(module, 'down_proj')):
            # Detect wrapping type for each projection
            dora_wrapped = (
                _is_dora_wrapped(module.gate_proj)
                and _is_dora_wrapped(module.up_proj)
                and _is_dora_wrapped(module.down_proj)
            )
            adalora_wrapped = (
                _is_adalora_wrapped(module.gate_proj)
                and _is_adalora_wrapped(module.up_proj)
                and _is_adalora_wrapped(module.down_proj)
            )
            lora_wrapped = (
                _is_lora_wrapped(module.gate_proj)
                and _is_lora_wrapped(module.up_proj)
                and _is_lora_wrapped(module.down_proj)
            )
            dylora_wrapped = (
                _is_dylora_wrapped(module.gate_proj)
                and _is_dylora_wrapped(module.up_proj)
                and _is_dylora_wrapped(module.down_proj)
            )
            vera_wrapped = (
                _is_vera_wrapped(module.gate_proj)
                and _is_vera_wrapped(module.up_proj)
                and _is_vera_wrapped(module.down_proj)
            )
            peft_wrapped = (
                _is_peft_wrapped(module.gate_proj)
                or _is_peft_wrapped(module.up_proj)
                or _is_peft_wrapped(module.down_proj)
            )

            if dora_wrapped:
                module.forward = _make_flash_forward_dora(module, k_fraction)
                logger.info(f"FlashFFN applied (DoRA mode, k={k_fraction}) to: {name}")
            elif adalora_wrapped:
                module.forward = _make_flash_forward_adalora(module, k_fraction)
                logger.info(f"FlashFFN applied (AdaLoRA mode, k={k_fraction}) to: {name}")
            elif lora_wrapped:
                module.forward = _make_flash_forward_lora(module, k_fraction)
                logger.info(f"FlashFFN applied (LoRA mode, k={k_fraction}) to: {name}")
            elif dylora_wrapped:
                module.forward = _make_flash_forward_dylora(module, k_fraction)
                logger.info(f"FlashFFN applied (DyLoRA mode, k={k_fraction}) to: {name}")
            elif vera_wrapped:
                module.forward = _make_flash_forward_vera(module, k_fraction)
                logger.info(f"FlashFFN applied (VeRA mode, k={k_fraction}) to: {name}")
            elif peft_wrapped:
                module.forward = _make_flash_forward_module(module, k_fraction)
                logger.info(f"FlashFFN applied (grad-checkpoint fallback) to: {name}")
            else:
                module.forward = _make_flash_forward_raw(module, k_fraction)
                logger.info(f"FlashFFN applied (raw-weight mode) to: {name}")

            converted += 1
    return converted


###############################################################################
#                                   data-keys                                 #
###############################################################################
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "anli_r1": ("premise", "hypothesis"),
    "wikitext2": (None, None),
    "wikitext103": (None, None),
}

###############################################################################
#                                  arg-parsing                                #
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a GLUE task (Mo5 variant)")

    # Model and Data Arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--load_pretrained_model", type=str, default=None, help="Path to a checkpoint to load model weights from.")
    parser.add_argument("--task_name", type=str, required=True, choices=list(task_to_keys.keys()),
        help="Task name: GLUE/SuperGLUE task or causal LM task (wikitext2, wikitext103).")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")

    # Training Hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use (e.g., 'adamw', 'galore_adamw', 'adamw-lora').")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per-device batch size for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=None, help="Effective total batch size. Overrides gradient_accumulation_steps if set.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum learning rate as a ratio of the max learning rate.")
    parser.add_argument("--grad_clipping", type=float, default=1.0, help="Gradient clipping value. 0.0 to disable.")
    parser.add_argument("--beta1", type=float, default=0.0, help="Beta1 for Adam-like optimizers (e.g., Adafactor).")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32", help="Data type for model training (bfloat16, float16, float32).")

    # GaLore / GALE Specific Arguments
    parser.add_argument("--rank", type=int, default=128, help="Rank for GaLore/GALE projection matrices.")
    parser.add_argument("--update_proj_gap", type=int, default=50, help="Frequency of updating GaLore/GALE projection matrices.")
    parser.add_argument("--galore_scale", type=float, default=1.0, help="Scaling factor for GaLore.")
    parser.add_argument("--proj_type", type=str, default="std", help="Projection type for GaLore.")

    # LoRA / AdapterHub Specific Arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (used by PEFT LoRA via --optimizer adamw-lora).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--prefix_bottleneck_size", type=int, default=256, help="Prefix Tuning bottleneck size.")
    parser.add_argument("--lora_all_modules", action="store_true", help="Apply LoRA to all supported linear layers.")

    # PEFT-specific Arguments (DoRA, VeRA, FourierFT)
    parser.add_argument("--dora_r", type=int, default=16, help="DoRA rank (typically half of LoRA rank).")
    parser.add_argument("--dora_alpha", type=int, default=32, help="DoRA alpha scaling parameter.")
    parser.add_argument("--dora_dropout", type=float, default=0.05, help="DoRA dropout probability.")
    parser.add_argument("--vera_r", type=int, default=256, help="VeRA rank (typically higher than LoRA).")
    parser.add_argument("--vera_dropout", type=float, default=0.0, help="VeRA dropout probability.")
    parser.add_argument("--vera_d_initial", type=float, default=0.1, help="VeRA initial value for scaling vectors.")
    parser.add_argument("--vera_projection_prng_key", type=int, default=0, help="VeRA random seed for projection initialization.")
    parser.add_argument("--fourierft_n_frequency", type=int, default=1000, help="FourierFT number of learnable frequency components.")
    parser.add_argument("--fourierft_scaling", type=float, default=150.0, help="FourierFT scaling parameter (100-150 for GLUE/NLU, 300 for LLaMA/ViT).")
    parser.add_argument("--fourierft_random_loc_seed", type=int, default=777, help="FourierFT random seed for frequency selection.")

    # GB-VeRA Specific Arguments
    parser.add_argument("--gbvera_r", type=int, default=256, help="GB-VeRA rank (typically same as VeRA, default 256).")
    parser.add_argument("--gbvera_d_initial", type=float, default=0.1, help="GB-VeRA initial value for λ_d (default 0.1).")
    parser.add_argument("--gbvera_b_initial", type=float, default=0.01, help="GB-VeRA initial value for λ_b (default 0.01, non-zero to fix bootstrap).")
    parser.add_argument("--gbvera_dropout", type=float, default=0.0, help="GB-VeRA dropout probability.")
    parser.add_argument("--gbvera_projection_prng_key", type=int, default=0, help="GB-VeRA random seed for projection initialization.")

    # AdaLoRA Specific Arguments
    parser.add_argument("--adalora_init_r", type=int, default=12, help="AdaLoRA initial rank (before pruning).")
    parser.add_argument("--adalora_target_r", type=int, default=4, help="AdaLoRA target rank (after pruning).")
    parser.add_argument("--adalora_alpha", type=int, default=8, help="AdaLoRA alpha scaling parameter.")
    parser.add_argument("--adalora_dropout", type=float, default=0.0, help="AdaLoRA dropout probability.")
    parser.add_argument("--adalora_tinit", type=int, default=200, help="AdaLoRA: initial warmup steps (no pruning). Paper default=200.")
    parser.add_argument("--adalora_tfinal", type=int, default=200, help="AdaLoRA: final steps (no pruning). Paper default=200.")
    parser.add_argument("--adalora_deltaT", type=int, default=10, help="AdaLoRA: interval between rank allocation steps. Paper default=10.")
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=0.5, help="AdaLoRA: orthogonality regularization weight.")

    # DyLoRA Specific Arguments
    parser.add_argument("--dylora_r", type=int, default=8, help="DyLoRA max rank (trains across ranks 1..r).")
    parser.add_argument("--dylora_alpha", type=int, default=16, help="DyLoRA alpha scaling parameter.")
    parser.add_argument("--dylora_dropout", type=float, default=0.0, help="DyLoRA dropout probability.")

    # Spectral Adapter (Truncated DCT Factored Adaptation) Arguments
    parser.add_argument("--spectral_p", type=int, default=32, help="Spectral adapter: number of DCT basis vectors for output dimension.")
    parser.add_argument("--spectral_q", type=int, default=32, help="Spectral adapter: number of DCT basis vectors for input dimension.")
    parser.add_argument("--spectral_scaling", type=float, default=1.0, help="Spectral adapter: scaling factor for adapter output.")
    parser.add_argument("--spectral_dropout", type=float, default=0.0, help="Spectral adapter: dropout probability.")
    parser.add_argument("--spectral_d_initial", type=float, default=0.0, help="Spectral adapter: if > 0, initialize coefficients with N(0, d_initial) instead of zeros.")
    parser.add_argument("--spectral_target_modules", type=str, default=None, help="Spectral adapter: comma-separated list of target module names (e.g., 'query,value'). If None, uses architecture defaults.")
    parser.add_argument("--spectral_freq_mode", type=str, default="contiguous", choices=["contiguous", "geometric", "geometric_half", "hybrid"], help="Spectral adapter: frequency selection strategy. 'contiguous' uses [0..k-1], 'geometric' uses power-spaced indices over [0, d//2], 'hybrid' uses 3k/4 contiguous low + k/4 geometric high.")
    parser.add_argument("--spectral_freq_exponent", type=float, default=2.0, help="Spectral adapter: exponent for geometric spacing (default 2.0=quadratic). 1.0=linear/uniform, 3.0=cubic/denser low-freq.")
    parser.add_argument("--spectral_factored_rank", type=int, default=0, help="Spectral adapter: if > 0, factor S = A(p,r)@B(r,q) for wider freq coverage. Params per module = p*r + r*q. 0 = dense S (default).")
    parser.add_argument("--spectral_learn_scaling", action="store_true", default=False, help="Spectral adapter: if set, each module gets a learnable log-space scaling parameter (+1 param/module).")
    parser.add_argument("--freeze_classifier_dense", action="store_true", default=False, help="Freeze classifier.dense layer to prevent gradient collapse on RoBERTa-like models where the large randomly-initialized dense layer overwhelms the small adapter.")
    parser.add_argument("--classifier_lr", type=float, default=None, help="Separate learning rate for classifier head params. If set, creates separate optimizer param groups for classifier (at this LR) and adapter (at --learning_rate). Prevents the fast-classifier/slow-adapter race condition.")

    # Generic target-module override (applies to all adapter methods)
    parser.add_argument("--adapter_target_modules", type=str, default=None,
        help="Comma-separated target module names, overrides architecture defaults")

    # Execution & Benchmarking Arguments
    parser.add_argument("--name", type=str, default="glue_finetuning_run", help="A name for this training run.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="(ignored, script uses fixed seeds 41-45)")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--per_layer_opt", action="store_true", help="Enable per-layer optimization (no retaining grad mode) where gradients are applied immediately layer by layer.")

    # Hub / Checkpointing Arguments
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory at the cost of slower backward pass.")

    # FlashFFN Arguments
    parser.add_argument("--flash_ffn", action="store_true", help="Enable FlashFFN activation compression for SwiGLU MLP blocks. Only works with SwiGLU models (LLaMA, Mistral, etc.).")
    parser.add_argument("--flash_ffn_k_fraction", type=float, default=0.3, help="Fraction of intermediate activations to keep in FlashFFN (default: 0.3 = top 30%%).")

    # Results CSV
    parser.add_argument("--results_csv", type=str, default=DEFAULT_RESULTS_FILE, help="Path to the results CSV file (default: ./results/mo53_glue.csv).")

    args = parser.parse_args()

    if args.total_batch_size:
        assert args.total_batch_size % args.per_device_train_batch_size == 0, "total_batch_size must be divisible by per_device_train_batch_size"
        args.gradient_accumulation_steps = args.total_batch_size // args.per_device_train_batch_size
    # Note: final total_batch_size is calculated in run_single_seed

    # Handle AdapterHub/PEFT/Optimizer method detection
    args.adapter_method = None
    args.optimizer_base = args.optimizer.lower()
    # AdapterHub methods: ia3, prefix
    # PEFT methods: lora, dora, vera, fourierft, adalora
    # Custom methods: gbvera (gradient-balanced VeRA)
    adapter_methods = ['lora', 'ia3', 'prefix', 'dora', 'vera', 'fourierft', 'gbvera', 'spectral', 'adalora', 'dylora']
    for method in adapter_methods:
        suffix = f'-{method}'
        if args.optimizer.lower().endswith(suffix):
            args.adapter_method = method
            args.optimizer_base = args.optimizer.lower().replace(suffix, '')
            break

    return args

###############################################################################
#                               memory accounting                             #
###############################################################################
def mib(x: int) -> float:
    """Converts bytes to MiB."""
    return x / 1024 ** 2

def calculate_theoretical_memory(model: nn.Module, args: argparse.Namespace) -> float:
    """
    Calculates the theoretical memory usage in MiB for various optimizers including Adam(W), Adafactor,
    AdamW8bit, LoRA, GaLore, GALE, Lion, IA³, and Prefix-Tuning.
    Assumes bf16 (2 bytes per parameter) for model weights and optimizer states.
    Returns 0.0 for unsupported configurations as a placeholder.
    """
    # Supported optimizers and their variants
    is_galore_or_gale = 'galore' in args.optimizer_base or 'gale' in args.optimizer_base
    is_adam = args.optimizer_base in ['adam', 'adamw']
    is_adafactor = args.optimizer_base in ['adafactor']
    is_adamw8bit = args.optimizer_base in ['adam8bit', 'adamw8bit']
    is_lion = args.optimizer_base in ['lion']

    if not (is_galore_or_gale or is_adam or is_adafactor or is_adamw8bit or is_lion):
        return 0.0

    total_model_params = sum(p.numel() for p in model.parameters())
    optimizer_state_params = 0
    # For Adam/AdamW, optimizer states are 2x the number of trainable parameters (momentum + variance)
    # For Lion, optimizer states are 1x the number of trainable parameters (only momentum, exp_avg)
    optimizer_state_multiplier = 1 if is_lion else 2

    if is_galore_or_gale:
        # Check if this is GALE or GaLore
        is_gale = 'gale' in args.optimizer.lower()
        
        if is_gale:
            # GALE: Memory = Full model params + GALE optimizer states (stored in low-rank space)
            gale_param_ids = set()
            # For BERT/RoBERTa models, target attention and feedforward layers
            target_modules = ["attention", "intermediate", "output"] if "llama" not in args.model_name_or_path.lower() else ["attn", "mlp"]

            # Calculate GALE-specific optimizer state size
            # GALE stores optimizer states (exp_avg, exp_avg_sq) in low-rank space
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                    m, n = module.weight.shape
                    # GALE projects gradient to low-rank space and stores optimizer states there
                    # For matrix m×n, the low-rank gradient has dimensions:
                    # - If m >= n: gradient is m×r, so optimizer states are 2×(m×r) = 2mr
                    # - If m < n: gradient is r×n, so optimizer states are 2×(r×n) = 2rn
                    if m >= n:
                        low_rank_size = m * args.rank
                    else:
                        low_rank_size = args.rank * n
                    optimizer_state_params += optimizer_state_multiplier * low_rank_size
                    gale_param_ids.add(id(module.weight))
            
            # Add standard optimizer states for other trainable parameters (e.g., embeddings, LayerNorms)
            for p in model.parameters():
                if p.requires_grad and id(p) not in gale_param_ids:
                    optimizer_state_params += optimizer_state_multiplier * p.numel()
        else:
            # GaLore: Memory = Full model params + GaLore optimizer states (stored in low-rank space)
            # GaLore actually stores optimizer states in low-rank space, same as GALE
            galore_param_ids = set()
            # For BERT/RoBERTa models, target attention and feedforward layers
            target_modules = ["attention", "intermediate", "output"] if "llama" not in args.model_name_or_path.lower() else ["attn", "mlp"]

            # Calculate GaLore-specific optimizer state size
            # GaLore stores optimizer states (exp_avg, exp_avg_sq) in low-rank space
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                    m, n = module.weight.shape
                    # GaLore projects gradient to low-rank space and stores optimizer states there
                    # For matrix m×n, the low-rank gradient has dimensions:
                    # - If m >= n: gradient is m×r, so optimizer states are 2×(m×r) = 2mr
                    # - If m < n: gradient is r×n, so optimizer states are 2×(r×n) = 2rn
                    if m >= n:
                        low_rank_size = m * args.rank
                    else:
                        low_rank_size = args.rank * n
                    optimizer_state_params += optimizer_state_multiplier * low_rank_size
                    galore_param_ids.add(id(module.weight))
            
            # Add standard optimizer states for other trainable parameters (e.g., embeddings, LayerNorms)
            for p in model.parameters():
                if p.requires_grad and id(p) not in galore_param_ids:
                    optimizer_state_params += optimizer_state_multiplier * p.numel()

    elif is_adam or is_adafactor or is_adamw8bit or is_lion:
        # Handle different adapter methods for all supported optimizers
        if args.adapter_method == 'lora':
            # LoRA: Only LoRA parameters are trainable, very small memory footprint
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                # Adafactor can use factored second moments for 2D parameters
                # For simplicity, we assume non-factored mode (similar to Adam)
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                # AdamW8bit uses 8-bit quantized states, but we calculate in full precision equivalent
                # The actual memory usage is lower, but we use full precision for theoretical calculation
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'dora':
            # DoRA: Similar to LoRA but with magnitude decomposition
            # Memory is similar to LoRA with slightly more parameters for magnitude vectors
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'vera':
            # VeRA: Very few trainable parameters (only scaling vectors)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'gbvera':
            # GB-VeRA: Same parameter count as VeRA (μ_d and μ_b instead of λ_d and λ_b)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'fourierft':
            # FourierFT: Extremely few trainable parameters (spectral coefficients)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            elif is_adamw8bit:
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'adalora':
            # AdaLoRA: SVD-parameterized LoRA with adaptive rank allocation
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'dylora':
            # DyLoRA: Dynamic LoRA (same param count as LoRA at max rank)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer_state_params = optimizer_state_multiplier * trainable_params

        elif args.adapter_method == 'ia3':
            # IA³: Only scaling vectors are trainable, optimizer states for scaling vectors only
            ia3_optimizer_params = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    m, n = module.weight.shape
                    # IA³ introduces scaling vector of dimension n
                    # Optimizer states: 2 * n (momentum + variance for scaling vector)
                    ia3_optimizer_params += optimizer_state_multiplier * n
            optimizer_state_params = ia3_optimizer_params
            
        elif args.adapter_method == 'prefix':
            # Prefix-Tuning: Prefix parameters are trainable
            prefix_optimizer_params = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    m, n = module.weight.shape
                    p = args.prefix_bottleneck_size
                    # Prefix parameters: 2pn (for key and value prefixes)
                    # Optimizer states: 2 * (2pn) = 4pn (momentum + variance for prefix parameters)
                    prefix_optimizer_params += optimizer_state_multiplier * (2 * p * n)
            optimizer_state_params = prefix_optimizer_params
            
        else:
            # Full Fine-Tuning: All parameters are trainable
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if is_adafactor:
                # Adafactor memory calculation
                # For 2D parameters (matrices), Adafactor can use factored second moments
                # This reduces memory from O(mn) to O(m+n) for an m×n matrix
                # For 1D parameters, it uses standard second moments
                adafactor_state_params = 0
                for p in model.parameters():
                    if p.requires_grad:
                        if len(p.shape) >= 2:  # 2D or higher dimensional parameters
                            # Factored mode: row and column statistics
                            # exp_avg_sq_row: shape[:-1] elements
                            # exp_avg_sq_col: shape[:-2] + shape[-1:] elements
                            row_size = 1
                            for dim in p.shape[:-1]:
                                row_size *= dim
                            col_size = 1
                            for dim in p.shape[:-2]:
                                col_size *= dim
                            col_size *= p.shape[-1]
                            factored_size = row_size + col_size
                            
                            # Add first moment if beta1 is used
                            if hasattr(args, 'beta1') and args.beta1 and args.beta1 > 0:
                                adafactor_state_params += p.numel()  # exp_avg
                            adafactor_state_params += factored_size  # factored second moments
                        else:
                            # Non-factored mode for 1D parameters
                            if hasattr(args, 'beta1') and args.beta1 and args.beta1 > 0:
                                adafactor_state_params += p.numel()  # exp_avg
                            adafactor_state_params += p.numel()  # exp_avg_sq
                optimizer_state_params = adafactor_state_params
            elif is_adamw8bit:
                # AdamW8bit uses quantized states, but we calculate theoretical full precision
                optimizer_state_params = optimizer_state_multiplier * trainable_params
            else:  # is_adam or is_lion
                optimizer_state_params = optimizer_state_multiplier * trainable_params
    
    # Total parameters for memory calculation = model weights + optimizer states
    total_theoretical_params = total_model_params + optimizer_state_params
    
    # Convert to MiB assuming 2 bytes per parameter (BF16)
    bytes_per_mib = 1024**2
    memory_mib = (total_theoretical_params * 2) / bytes_per_mib
    
    return memory_mib

@torch.no_grad()
def get_memory_breakdown(model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer,
                           device: torch.device) -> dict:
    """
    Returns a breakdown of memory usage in MiB.
    """
    stats = {}
    if device.type == "cuda":
        # Model Parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        stats['param_mem_mib'] = mib(param_bytes)

        # Optimizer State
        opt_bytes = 0
        if optimizer and hasattr(optimizer, 'state') and optimizer.state:
            for state in optimizer.state.values():
                for t in state.values():
                    if torch.is_tensor(t):
                        opt_bytes += t.numel() * t.element_size()
        stats['opt_mem_mib'] = mib(opt_bytes)

        # CUDA Memory Stats
        stats['peak_memory_mib'] = mib(torch.cuda.max_memory_allocated(device))
        stats['allocated_memory_mib'] = mib(torch.cuda.memory_allocated(device))
    return stats

###############################################################################
#                             single-seed training loop                         #
###############################################################################
def run_single_seed(base_args: argparse.Namespace, seed: int):
    """
    Execute **one** full training run with the given `seed`.
    """
    args = copy.deepcopy(base_args)
    args.seed = seed
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, f"seed_{seed}")

    # --- Device and Seed Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Calculate total batch size
    if not args.total_batch_size:
        # Assuming a single device (num_processes = 1)
        args.total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"[seed {seed}] Running on device: {device}")
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.push_to_hub:
        repo_name = args.hub_model_id or Path(args.output_dir).absolute().name
        repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
        repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Data Loading ---
    is_causal_lm = args.task_name in CAUSAL_LM_TASKS
    is_regression = args.task_name == "stsb"

    if is_causal_lm:
        wikitext_config = "wikitext-2-raw-v1" if args.task_name == "wikitext2" else "wikitext-103-raw-v1"
        raw_datasets = load_dataset("wikitext", wikitext_config)
        num_labels = None
    elif args.task_name in ("boolq", "cb"):
        raw_datasets = load_dataset("super_glue", args.task_name)
    elif args.task_name == "anli_r1":
        _anli = load_dataset("facebook/anli")
        from datasets import DatasetDict
        raw_datasets = DatasetDict({
            "train": _anli["train_r1"],
            "validation": _anli["dev_r1"],
            "test": _anli["test_r1"],
        })
    else:
        raw_datasets = load_dataset("glue", args.task_name)

    if not is_causal_lm:
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1

    # --- Model Initialization ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if is_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )
    
    if (args.download_only):
       logger.info("DOWNLOAD ONLY (passed via --download_only flag") 
       exit()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Dtype, Adapter, and Device Setup ---
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32
    
    # Cast model to the correct dtype before adapter init or moving to device
    if dtype != torch.float32:
        model.to(dtype=dtype)
    
    if args.adapter_method:
        # PEFT methods: dora, vera, fourierft
        # Custom methods: gbvera, spectral
        peft_methods = ['dora', 'vera', 'fourierft', 'adalora', 'lora']
        custom_methods = ['gbvera', 'spectral', 'dylora']

        if args.adapter_method == 'spectral':
            # Use our Truncated DCT Factored Adaptation
            logger.info(f"Initializing model for Spectral Adapter (Truncated DCT) training...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif args.spectral_target_modules:
                target_modules = [m.strip() for m in args.spectral_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"Spectral adapter target modules: {target_modules}")

            model = get_spectral_adapter_model(
                model=model,
                target_modules=target_modules,
                p=args.spectral_p,
                q=args.spectral_q,
                scaling=args.spectral_scaling,
                dropout=args.spectral_dropout,
                d_initial=args.spectral_d_initial,
                freq_mode=args.spectral_freq_mode,
                freq_exponent=args.spectral_freq_exponent,
                factored_rank=args.spectral_factored_rank,
                learn_scaling=args.spectral_learn_scaling,
                freeze_classifier_dense=args.freeze_classifier_dense,
            )

            logger.info(f"Successfully applied Spectral Adapter to model.")
            model.print_trainable_parameters()

            # Mixed-precision: adapter params (coeffs, DCT basis) stay float32
            # even when base model is float16/bfloat16 for LLaMA-scale models
            if dtype != torch.float32:
                n_f32 = sum(1 for p in model.parameters() if p.requires_grad and p.dtype == torch.float32)
                n_base = sum(1 for p in model.parameters() if not p.requires_grad and p.dtype == dtype)
                logger.info(f"Mixed-precision: {n_f32} trainable params in float32, {n_base} frozen params in {dtype}")

        elif args.adapter_method == 'dylora':
            # Use our custom DyLoRA implementation
            logger.info(f"Initializing model for DyLoRA training (custom implementation)...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            model = get_dylora_model(
                model=model,
                target_modules=target_modules,
                r=args.dylora_r,
                alpha=args.dylora_alpha,
                dropout=args.dylora_dropout,
            )

            logger.info(f"Successfully applied DyLoRA to model.")
            model.print_trainable_parameters()

        elif args.adapter_method == 'gbvera':
            # Use our custom GB-VeRA implementation
            logger.info(f"Initializing model for GB-VeRA training (custom implementation)...")

            # Determine target modules based on model architecture
            if "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                target_modules = ["query", "value"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "v_proj"]
            elif "opt" in args.model_name_or_path.lower():
                target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]

            model = get_gbvera_model(
                model=model,
                target_modules=target_modules,
                r=args.gbvera_r,
                d_initial=args.gbvera_d_initial,
                b_initial=args.gbvera_b_initial,
                dropout=args.gbvera_dropout,
                projection_prng_key=args.gbvera_projection_prng_key,
            )

            logger.info(f"Successfully applied GB-VeRA to model.")
            model.print_trainable_parameters()

        elif args.adapter_method in peft_methods:
            logger.info(f"Initializing model for {args.adapter_method.upper()} training with PEFT library...")

            # Determine target modules: CLI override or architecture defaults
            if args.adapter_target_modules:
                target_modules = [m.strip() for m in args.adapter_target_modules.split(",")]
            elif "roberta" in args.model_name_or_path.lower() or "bert" in args.model_name_or_path.lower():
                # For BERT/RoBERTa models used in GLUE
                target_modules = ["query", "key", "value", "dense"]
            elif "gpt2" in args.model_name_or_path.lower() or "gpt-2" in args.model_name_or_path.lower():
                # For GPT-2 models
                target_modules = ["c_attn", "c_proj"]
            elif "llama" in args.model_name_or_path.lower():
                # For LLaMA models
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "opt" in args.model_name_or_path.lower():
                # For OPT models (separate Q/K/V/out_proj + FFN)
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            else:
                # Default: try common attention projection names
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            peft_task_type = TaskType.CAUSAL_LM if is_causal_lm else TaskType.SEQ_CLS
            peft_config = None
            if args.adapter_method == 'dora':
                peft_config = PeftLoraConfig(
                    r=args.dora_r,
                    lora_alpha=args.dora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.dora_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    use_dora=True  # Enable DoRA
                )
            elif args.adapter_method == 'vera':
                peft_config = VeraConfig(
                    r=args.vera_r,
                    target_modules=target_modules,
                    vera_dropout=args.vera_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    save_projection=True,
                    projection_prng_key=args.vera_projection_prng_key,
                    d_initial=args.vera_d_initial
                )
            elif args.adapter_method == 'fourierft':
                peft_config = FourierFTConfig(
                    n_frequency=args.fourierft_n_frequency,
                    target_modules=target_modules,
                    task_type=peft_task_type,
                    scaling=args.fourierft_scaling,
                    random_loc_seed=args.fourierft_random_loc_seed,
                    init_weights=True,  # PEFT defaults to False (random init), must be True for zero-delta start
                )
            elif args.adapter_method == 'adalora':
                # Pre-compute total training steps for AdaLoRA's rank allocation schedule.
                # For causal LM, raw_datasets["train"] has text lines (not training examples),
                # so use max_train_steps directly or defer to a rough estimate.
                if args.max_train_steps:
                    est_total_steps = args.max_train_steps
                else:
                    n_train = len(raw_datasets["train"])
                    est_steps_per_epoch = math.ceil(n_train / args.per_device_train_batch_size / args.gradient_accumulation_steps)
                    est_total_steps = est_steps_per_epoch * args.num_train_epochs
                logger.info(f"AdaLoRA: estimated total_step={est_total_steps} for rank allocation schedule")

                peft_config = AdaLoraConfig(
                    init_r=args.adalora_init_r,
                    target_r=args.adalora_target_r,
                    lora_alpha=args.adalora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.adalora_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    total_step=est_total_steps,
                    tinit=args.adalora_tinit,
                    tfinal=args.adalora_tfinal,
                    deltaT=args.adalora_deltaT,
                    orth_reg_weight=args.adalora_orth_reg_weight,
                )
            elif args.adapter_method == 'lora':
                peft_config = PeftLoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type=peft_task_type,
                    use_dora=False,
                )

            if peft_config:
                model = get_peft_model(model, peft_config)
                logger.info(f"Successfully applied {args.adapter_method.upper()} to model via PEFT.")
                model.print_trainable_parameters()
        else:
            # AdapterHub methods: ia3, prefix
            logger.info(f"Initializing model for {args.adapter_method.upper()} training with AdapterHub...")
            adapters.init(model)

            adapter_config = None
            if args.adapter_method == 'ia3':
                adapter_config = IA3Config()
            elif args.adapter_method == 'prefix':
                adapter_config = PrefixTuningConfig(bottleneck_size=args.prefix_bottleneck_size)

            if adapter_config:
                adapter_name = f"{args.adapter_method}_adapter"
                model.add_adapter(adapter_name, config=adapter_config)
                model.train_adapter(adapter_name)
                model.set_active_adapters(adapter_name)

                # Cast model again after adding adapters to ensure new params are also in the correct dtype
                if dtype != torch.float32:
                    model.to(dtype=dtype)
                logger.info(f"Successfully added and enabled {args.adapter_method.upper()} adapter for training.")

    model.to(device)

    # --- Enable Gradient Checkpointing ---
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info(f"[seed {seed}] Gradient checkpointing enabled")
        else:
            logger.warning(f"[seed {seed}] Model does not support gradient checkpointing, skipping")

    # --- FlashFFN Application ---
    if args.flash_ffn:
        n_converted = apply_flash_ffn(model, k_fraction=args.flash_ffn_k_fraction)
        if n_converted > 0:
            logger.info(f"[seed {seed}] FlashFFN enabled: converted {n_converted} SwiGLU MLP layers (k_fraction={args.flash_ffn_k_fraction})")
        else:
            raise ValueError(
                f"--flash_ffn was requested but no SwiGLU MLP modules (gate_proj/up_proj/down_proj) "
                f"were found in model '{args.model_name_or_path}'. FlashFFN only works with SwiGLU models "
                f"(LLaMA, Mistral, Qwen, Gemma, etc.), not BERT/RoBERTa/GPT-2/OPT."
            )

    # --- Dataset Preprocessing ---
    if is_causal_lm:
        # Causal LM: tokenize text, concatenate all tokens, chunk into fixed-length blocks
        block_size = args.max_length
        text_column = "text"

        def tokenize_function(examples):
            return tokenizer(examples[text_column], return_attention_mask=False)

        tokenized_datasets = raw_datasets.map(
            tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names,
            desc="Tokenising",
        )

        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = (len(concatenated["input_ids"]) // block_size) * block_size
            result = {k: [concatenated[k][i:i + block_size] for i in range(0, total_length, block_size)]
                      for k in concatenated.keys()}
            result["labels"] = result["input_ids"].copy()
            return result

        processed_datasets = tokenized_datasets.map(
            group_texts, batched=True, desc="Chunking",
        )
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        data_collator = default_data_collator
    else:
        # Classification: tokenize sentence pairs, add labels
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
            result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
            if "label" in examples:
                result["labels"] = examples["label"]
            return result

        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, desc="Tokenising",
        )
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

        data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_loader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # --- Optimizer and Scheduler Setup ---
    param_groups = None
    if 'galore' in args.optimizer_base or 'gale' in args.optimizer_base:
        method_name = "GaLore" if 'galore' in args.optimizer_base else "GALE"
        target_modules = ["attn", "mlp"] if "llama" in args.model_name_or_path.lower() else ["attention", "intermediate", "output"]
        
        low_rank_params = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in target_modules):
                logger.info(f"Enabling {method_name} for weights in module: {name}")
                low_rank_params.append(module.weight)

        id_low_rank_params = {id(p) for p in low_rank_params}
        regular_params = [p for p in model.parameters() if id(p) not in id_low_rank_params and p.requires_grad]
        low_rank_pg = {'params': low_rank_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale}
        if 'galore' in args.optimizer_base:
            low_rank_pg['proj_type'] = args.proj_type
        param_groups = [{'params': regular_params}, low_rank_pg]
    elif args.classifier_lr is not None:
        # Separate LR for classifier vs adapter params
        classifier_params = []
        adapter_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'classifier' in name or 'score' in name:
                classifier_params.append(p)
            else:
                adapter_params.append(p)
        param_groups = [
            {'params': adapter_params, 'lr': args.learning_rate},
            {'params': classifier_params, 'lr': args.classifier_lr},
        ]
        logger.info(f"Separate LR: adapter={args.learning_rate}, classifier={args.classifier_lr} "
                     f"({len(adapter_params)} adapter params, {len(classifier_params)} classifier params)")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    optimizer_classes = {
        'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'adam8bit': bnb.optim.Adam8bit,
        'adafactor': transformers.optimization.Adafactor, 'galore_adamw': GaLoreAdamW,
        'galore_adamw8bit': GaLoreAdamW8bit, 'galore_adafactor': GaLoreAdafactor,
        'swift_galore_adamw': SwiftGaLoreAdamW,
        'gale_adamw': GALE_AdamW, 'gale_adamw_fused': GALE_AdamW, 'gale_adamw_fused_approx': GALE_AdamW,
        'gale_adafactor': GALE_Adafactor, 'gale_adafactor_fused': GALE_Adafactor, 'gale_adafactor_fused_approx': GALE_Adafactor,
        'gale_adamw8bit': GALE_AdamW8bit, 'gale_adamw8bit_fused': GALE_AdamW8bit, 'gale_adamw8bit_fused_approx': GALE_AdamW8bit,
        'lion': Lion, 'gale_lion': GALE_Lion
    }
    optimizer_class = optimizer_classes[args.optimizer_base]
    optimizer_kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    
    if args.optimizer_base in ['adafactor', 'galore_adafactor']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False})
    elif args.optimizer_base in ['gale_adamw']:
        optimizer_kwargs['mode'] = 'native'
    elif args.optimizer_base in ['gale_adamw_fused']:
        optimizer_kwargs['mode'] = 'fused'
    elif args.optimizer_base in ['gale_adamw_fused_approx']:
        optimizer_kwargs['mode'] = 'approximate'
    elif args.optimizer_base in ['gale_adafactor']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'native'})
    elif args.optimizer_base in ['gale_adafactor_fused']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'fused'})
    elif args.optimizer_base in ['gale_adafactor_fused_approx']:
        optimizer_kwargs['beta1'] = None if args.beta1 == 0.0 else args.beta1
        optimizer_kwargs.update({'relative_step': False, 'scale_parameter': False, 'warmup_init': False, 'mode': 'approximate'})
    elif args.optimizer_base in ['gale_adamw8bit']:
        optimizer_kwargs['mode'] = 'native'
    elif args.optimizer_base in ['gale_adamw8bit_fused']:
        optimizer_kwargs['mode'] = 'fused'
    elif args.optimizer_base in ['gale_adamw8bit_fused_approx']:
        optimizer_kwargs['mode'] = 'approximate'
    
    optimizer = optimizer_class(param_groups, **optimizer_kwargs)
    
    # Calculate theoretical memory AFTER optimizer setup
    theoretical_mem_mib = calculate_theoretical_memory(model, args)
    logger.info(f"[seed {seed}] Theoretical Memory (BF16): {theoretical_mem_mib:.2f} MiB")
    
    # --- Training Loop Setup ---
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps,
    )
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if is_causal_lm:
        metric = None  # perplexity computed from loss directly
    elif args.task_name in ("boolq", "cb"):
        metric = evaluate.load("super_glue", args.task_name)
    elif args.task_name == "anli_r1":
        metric = evaluate.load("accuracy")
    else:
        metric = evaluate.load("glue", args.task_name)
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    
    logger.info(f"[seed {seed}] ***** Training *****")
    logger.info(f"[seed {seed}] Epochs={args.num_train_epochs} | Steps={args.max_train_steps} | Total batch={args.total_batch_size}")
    
    step_times: List[float] = []
    mem_stats_after_first_step = {}
    best_metric_val = float("-inf")
    best_metric_dict: Dict[str, float] = {}
    
    # --- Training Loop ---
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):

            # Move batch to device and cast to appropriate dtype
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
            }
            if is_regression and "labels" in batch:
                batch["labels"] = batch["labels"].to(dtype)

            outputs = model(**batch)
            loss = outputs.loss
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step == len(train_loader) - 1):
                if args.grad_clipping > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
                
                step_start_time = time.perf_counter()
                optimizer.step()
                step_times.append(time.perf_counter() - step_start_time)

                lr_scheduler.step()

                # AdaLoRA: update rank allocation BEFORE zero_grad (needs gradients)
                # Only run when actual pruning is configured (init_r > target_r)
                if args.adapter_method == 'adalora' and args.adalora_init_r > args.adalora_target_r:
                    model.base_model.update_and_allocate(completed_steps + 1)

                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps == 1 and device.type == "cuda":
                    torch.cuda.empty_cache()
                    mem_stats_after_first_step = get_memory_breakdown(model, optimizer, device)
                    logger.info(
                        "Memory breakdown after 1st optimizer step: | "
                        f"Param Memory: {mem_stats_after_first_step.get('param_mem_mib', 0):.2f} MiB | "
                        f"Optimizer Memory: {mem_stats_after_first_step.get('opt_mem_mib', 0):.2f} MiB | "
                        f"Allocated Memory: {mem_stats_after_first_step.get('allocated_memory_mib', 0):.2f} MiB | "
                        f"Peak Memory: {mem_stats_after_first_step.get('peak_memory_mib', 0):.2f} MiB"
                    )
            
            if completed_steps >= args.max_train_steps:
                break
        
        # --- Evaluation ---
        model.eval()
        if is_causal_lm:
            total_loss = 0.0
            total_tokens = 0
            for batch in eval_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                # outputs.loss is mean over tokens in the batch
                n_tokens = batch["labels"].numel()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            ppl = math.exp(min(avg_loss, 100))  # cap to avoid overflow
            eval_metric = {"perplexity": ppl, "loss": avg_loss}
        else:
            for batch in eval_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                refs = batch["labels"]
                metric.add_batch(predictions=preds.cpu(), references=refs.cpu())
            eval_metric = metric.compute()

        logger.info(f"[seed {seed}] epoch {epoch}: {eval_metric}")

        primary_val = _primary_metric(args.task_name, eval_metric)
        if primary_val > best_metric_val:
            best_metric_val = primary_val
            best_metric_dict = eval_metric.copy()
        
        if completed_steps >= args.max_train_steps:
            break
            
    # --- Final Benchmarks ---
    peak_mem_mib = mem_stats_after_first_step.get('peak_memory_mib', 0)
    if device.type == "cuda":
        final_peak_memory_mib = mib(torch.cuda.max_memory_allocated(device))
        logger.info(f"[seed {seed}] Overall Peak GPU Memory (whole run): {final_peak_memory_mib:.2f} MiB")
        peak_mem_mib = max(peak_mem_mib, final_peak_memory_mib)

    if not step_times:
        avg_step_time = std_step_time = np.nan
    else:
        avg_step_time = statistics.mean(step_times)
        std_step_time = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    # --- Cleanup ---
    del model, optimizer, train_loader, eval_loader, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "best_metric_dict": best_metric_dict,
        "param_mem_mib": mem_stats_after_first_step.get('param_mem_mib', 0),
        "opt_mem_mib": mem_stats_after_first_step.get('opt_mem_mib', 0),
        "runtime_mem_mib": mem_stats_after_first_step.get('allocated_memory_mib', 0),
        "peak_mem_mib": peak_mem_mib,
        "theoretical_mem_mib": theoretical_mem_mib,
        "avg_step_time": avg_step_time,
        "std_step_time": std_step_time,
    }

###############################################################################
#                                  entry-point                                #
###############################################################################
def main():
    args = parse_args()
    
    training_start_time = time.time()
    all_results: List[Dict] = []
    for idx, seed in enumerate(SEEDS):
        print("=" * 80, flush=True)
        print(f"Starting run {idx + 1}/{len(SEEDS)} with seed {seed}", flush=True)
        print("=" * 80, flush=True)
        res = run_single_seed(args, seed)
        all_results.append(res)
    
    total_training_time_sec = time.time() - training_start_time

    # --- Process and Save Results ---
    first_res = all_results[0]
    metric_keys = ["accuracy", "f1", "matthews_correlation", "pearson", "spearmanr", "perplexity"]
    median_metrics = {}
    for k in metric_keys:
        vals = [r["best_metric_dict"].get(k, np.nan) for r in all_results]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        median_metrics[k] = statistics.median(vals) if vals else np.nan

    all_columns = [
        "timestamp", "name", "model_name_or_path", "task_name", "optimizer",
        "lr", "per_device_train_batch_size", "total_batch_size", "num_train_epochs",
        "max_train_steps", "dtype", "adapter_method",
        "rank", "update_proj_gap", "galore_scale",
        "lora_r", "lora_alpha", "lora_dropout", "prefix_bottleneck_size",
        "dora_r", "dora_alpha", "dora_dropout",
        "vera_r", "vera_dropout", "vera_d_initial",
        "gbvera_r", "gbvera_d_initial", "gbvera_b_initial", "gbvera_dropout",
        "fourierft_n_frequency", "fourierft_scaling",
        "adalora_init_r", "adalora_target_r", "adalora_alpha", "adalora_dropout",
        "dylora_r", "dylora_alpha", "dylora_dropout",
        "spectral_p", "spectral_q", "spectral_scaling", "spectral_dropout", "spectral_d_initial", "spectral_freq_mode", "spectral_freq_exponent", "spectral_factored_rank", "spectral_learn_scaling",
        "per_layer_opt", "gradient_checkpointing", "flash_ffn", "flash_ffn_k_fraction",
        "accuracy", "f1", "matthews_correlation", "pearson", "spearmanr", "perplexity",
        "total_training_time_sec", "param_mem_mib", "opt_mem_mib", "runtime_mem_mib",
        "peak_mem_mib", "theoretical_mem_mib", "avg_step_time", "std_step_time", "seed"
    ]
    comb_cols = ["name", "model_name_or_path", "task_name", "optimizer", "lr", "total_batch_size"]

    is_galore_or_gale = 'galore' in args.optimizer.lower() or 'gale' in args.optimizer.lower()
    
    result_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": args.name,
        "model_name_or_path": args.model_name_or_path,
        "task_name": args.task_name,
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "total_batch_size": args.total_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "dtype": args.dtype,
        "adapter_method": args.adapter_method if args.adapter_method else 'N/A',
        "rank": args.rank if is_galore_or_gale else 'N/A',
        "update_proj_gap": args.update_proj_gap if is_galore_or_gale else 'N/A',
        "galore_scale": args.galore_scale if is_galore_or_gale else 'N/A',
        "lora_r": args.lora_r if args.adapter_method == 'lora' else 'N/A',
        "lora_alpha": args.lora_alpha if args.adapter_method == 'lora' else 'N/A',
        "lora_dropout": args.lora_dropout if args.adapter_method == 'lora' else 'N/A',
        "prefix_bottleneck_size": args.prefix_bottleneck_size if args.adapter_method == 'prefix' else 'N/A',
        "dora_r": args.dora_r if args.adapter_method == 'dora' else 'N/A',
        "dora_alpha": args.dora_alpha if args.adapter_method == 'dora' else 'N/A',
        "dora_dropout": args.dora_dropout if args.adapter_method == 'dora' else 'N/A',
        "vera_r": args.vera_r if args.adapter_method == 'vera' else 'N/A',
        "vera_dropout": args.vera_dropout if args.adapter_method == 'vera' else 'N/A',
        "vera_d_initial": args.vera_d_initial if args.adapter_method == 'vera' else 'N/A',
        "gbvera_r": args.gbvera_r if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_d_initial": args.gbvera_d_initial if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_b_initial": args.gbvera_b_initial if args.adapter_method == 'gbvera' else 'N/A',
        "gbvera_dropout": args.gbvera_dropout if args.adapter_method == 'gbvera' else 'N/A',
        "fourierft_n_frequency": args.fourierft_n_frequency if args.adapter_method == 'fourierft' else 'N/A',
        "fourierft_scaling": args.fourierft_scaling if args.adapter_method == 'fourierft' else 'N/A',
        "adalora_init_r": args.adalora_init_r if args.adapter_method == 'adalora' else 'N/A',
        "adalora_target_r": args.adalora_target_r if args.adapter_method == 'adalora' else 'N/A',
        "adalora_alpha": args.adalora_alpha if args.adapter_method == 'adalora' else 'N/A',
        "adalora_dropout": args.adalora_dropout if args.adapter_method == 'adalora' else 'N/A',
        "dylora_r": args.dylora_r if args.adapter_method == 'dylora' else 'N/A',
        "dylora_alpha": args.dylora_alpha if args.adapter_method == 'dylora' else 'N/A',
        "dylora_dropout": args.dylora_dropout if args.adapter_method == 'dylora' else 'N/A',
        "spectral_p": args.spectral_p if args.adapter_method == 'spectral' else 'N/A',
        "spectral_q": args.spectral_q if args.adapter_method == 'spectral' else 'N/A',
        "spectral_scaling": args.spectral_scaling if args.adapter_method == 'spectral' else 'N/A',
        "spectral_dropout": args.spectral_dropout if args.adapter_method == 'spectral' else 'N/A',
        "spectral_d_initial": args.spectral_d_initial if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_mode": args.spectral_freq_mode if args.adapter_method == 'spectral' else 'N/A',
        "spectral_freq_exponent": args.spectral_freq_exponent if args.adapter_method == 'spectral' else 'N/A',
        "spectral_factored_rank": args.spectral_factored_rank if args.adapter_method == 'spectral' else 'N/A',
        "spectral_learn_scaling": args.spectral_learn_scaling if args.adapter_method == 'spectral' else 'N/A',
        "per_layer_opt": args.per_layer_opt,
        "gradient_checkpointing": args.gradient_checkpointing,
        "flash_ffn": args.flash_ffn,
        "flash_ffn_k_fraction": args.flash_ffn_k_fraction if args.flash_ffn else 'N/A',
        "accuracy": median_metrics.get("accuracy", np.nan),
        "f1": median_metrics.get("f1", np.nan),
        "matthews_correlation": median_metrics.get("matthews_correlation", np.nan),
        "pearson": median_metrics.get("pearson", np.nan),
        "spearmanr": median_metrics.get("spearmanr", np.nan),
        "perplexity": median_metrics.get("perplexity", np.nan),
        "total_training_time_sec": round(total_training_time_sec, 2),
        "param_mem_mib": round(first_res["param_mem_mib"], 2),
        "opt_mem_mib": round(first_res["opt_mem_mib"], 2),
        "runtime_mem_mib": round(first_res["runtime_mem_mib"], 2),
        "peak_mem_mib": round(first_res["peak_mem_mib"], 2),
        "theoretical_mem_mib": round(first_res["theoretical_mem_mib"], 2),
        "avg_step_time": round(first_res["avg_step_time"], 4) if first_res["avg_step_time"] is not np.nan else np.nan,
        "std_step_time": round(first_res["std_step_time"], 4) if first_res["std_step_time"] is not np.nan else np.nan,
        "seed": ",".join(map(str, SEEDS)),
    }
    # --- Hardened locking for 100s of concurrent SLURM jobs ---
    results_file = args.results_csv
    lock_file = results_file + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    lock = FileLock(lock_file, timeout=300)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with lock:
                logger.info(f"Acquired lock on {lock_file} (attempt {attempt + 1}).")
                df_results = _load_results_df(results_file, all_columns)
                df_results = _upsert_result(df_results, comb_cols, result_row)
                # Atomic write: write to temp file then rename
                tmp_file = results_file + f".tmp.{os.getpid()}"
                df_results.to_csv(tmp_file, index=False)
                os.replace(tmp_file, results_file)
                logger.info(f"Released lock. Logged Mo5 median results to {results_file}")
            break
        except Timeout:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning(f"Lock timeout on attempt {attempt + 1}/{max_retries}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    else:
        logger.error(f"Failed to acquire lock on {lock_file} after {max_retries} attempts. Results NOT saved.")


if __name__ == "__main__":
    main()
