import torch
import torch.nn.functional as F
import bitsandbytes.functional as BF

# NOTE (FlashFFN port): upstream imports flash_attn + einops at module scope, but they are
# only needed by pad_input/unpad_input, which the fused-layer path never calls. Importing
# them eagerly would make the whole package unimportable on a box without flash_attn built.
# Made lazy; the two helpers raise a clear error if they are ever actually used.
def _require_flash_attn():
    from flash_attn.bert_padding import index_first_axis, index_put_first_axis
    from einops import rearrange
    return index_first_axis, index_put_first_axis, rearrange

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, _, seq_len, _ = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()


def _base_weight_t(w, w_quant_state):
    """Return the base weight in [in_features, out_features] layout.

    Upstream HyC-LoRA is QLoRA-only and always calls dequantize_nf4. We additionally accept
    a plain (bf16/fp16/fp32) weight, signalled by w_quant_state is None, so the method can be
    run on an un-quantised base model. That makes an apples-to-apples comparison against our
    bf16 FlashFFN arms possible; the NF4 path is bit-for-bit the upstream behaviour.
    """
    if w_quant_state is None:
        return w.t()
    return BF.dequantize_nf4(w, w_quant_state).t()


def lora_forward(w, w_quant_state, w_lora_a, w_lora_b, b, x):
    w_dequant = _base_weight_t(w, w_quant_state)
    x = x.to(w_dequant.dtype)
    x_main = x @ w_dequant + b.to(w_dequant.dtype) if b is not None else x @ w_dequant
    x_lora_a = x @ w_lora_a.to(w_dequant.dtype)
    x_lora = x_lora_a @ w_lora_b.to(w_dequant.dtype)
    x = x_main + x_lora
    return x, x_main, x_lora_a


def lora_backward(w, w_quant_state, w_lora_a, w_lora_b, x, x_lora_a, grad_y):
    w_dequant = _base_weight_t(w, w_quant_state)
    grad_medium = grad_y.to(w_dequant.dtype) @ w_lora_b.mT
    w_lora_a, w_lora_b = w_lora_a.to(w_dequant.dtype), w_lora_b.to(w_dequant.dtype)
    grad_w_lora_a = x.to(w_dequant.dtype).mT @ (grad_medium)
    grad_w_lora_b = (x_lora_a.mT @ grad_y.to(w_lora_b.dtype))
    grad_x = grad_y.to(w_dequant.dtype) @ w_dequant.T 
    grad_x += (grad_medium @ w_lora_a.T)
    return grad_w_lora_a, grad_w_lora_b, grad_x


def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_backward(grad_output: torch.Tensor, n_rep: int):
    batch, expand_num_key_value_heads, slen, head_dim = grad_output.shape
    num_key_value_heads = expand_num_key_value_heads // n_rep
    grad_output = grad_output.reshape(batch, num_key_value_heads, n_rep, slen, head_dim)
    return grad_output.sum(dim=2)


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    _, index_put_first_axis, rearrange = _require_flash_attn()
    dim = hidden_states.shape[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    index_first_axis, _, rearrange = _require_flash_attn()
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )