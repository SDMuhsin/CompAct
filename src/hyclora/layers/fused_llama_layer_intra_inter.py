import math
import torch
import typing
import bitsandbytes as bnb

from ..operators.rope_kernels import rope_forward, rope_backward, calculate_settings
from ..operators.silu_kernels import silu_backward, silu_hadamard
from ..operators.rmsnorm_kernels import rmsnorm_backward, rmsnorm_forward
from ..operators.softmax_kernels import softmax_backward
from ..operators.compress_function import (
    compression_pack_channel_base,
    compression_pack_quant_base,
    compression_pack_softmax_base,
    outlier_addition_fuse_decompression_dequantization,
    decompression_dequantization,
    update_dict,
)

from ..compute_utils import(
    hidden_to_head_shape, head_to_hidden_shape,
    lora_forward, lora_backward,
    repeat_kv, repeat_kv_backward
)

# Zero-cost unless hyclora.prof.enable() has been called. The whole decoder layer is a single
# autograd.Function, so without these regions no profiler can see the ~70 tensor ops inside it.
from ..prof import rf

class FusedLlamaLayerIntraInterFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        #############attention part#############
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        ####################################
        cos: torch.Tensor,
        sin: torch.Tensor,
        ####################################
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_q_quant_state: typing.Tuple,
        w_q_lora_a: torch.Tensor,
        w_q_lora_b: torch.Tensor,
        ####################################
        w_k: torch.Tensor,
        b_k: torch.Tensor,
        w_k_quant_state: typing.Tuple,
        w_k_lora_a: torch.Tensor,
        w_k_lora_b: torch.Tensor,
        ####################################
        w_v: torch.Tensor,
        b_v: torch.Tensor,
        w_v_quant_state: typing.Tuple,
        w_v_lora_a: torch.Tensor,
        w_v_lora_b: torch.Tensor,
        ####################################
        w_o: torch.Tensor,
        b_o: torch.Tensor,
        w_o_quant_state: typing.Tuple,
        w_o_lora_a: torch.Tensor,
        w_o_lora_b: torch.Tensor,
        #############mlp part#############
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ####################################
        w_gate: torch.Tensor,
        b_gate: torch.Tensor,
        w_gate_quant_state: typing.Tuple,
        w_gate_lora_a: torch.Tensor,
        w_gate_lora_b: torch.Tensor,
        ####################################
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_up_quant_state: typing.Tuple,
        w_up_lora_a: torch.Tensor,
        w_up_lora_b: torch.Tensor,
        ####################################
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        w_down_quant_state: typing.Tuple,
        w_down_lora_a: torch.Tensor,
        w_down_lora_b: torch.Tensor,
        ###############other################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int,
        ###############about statistics################
        iteration: int,
        iteration_threshold: int,
        static_value: dict,
        softmax_outlier_ratio: float,
        layernorm_outlier_ratio: float,
        q_bit: int,
    ):
        # layernorm or rmsnorm
        with rf("F01.rmsnorm_1"):
            x_norm_1, mean_1, rstd_1, _, _ = rmsnorm_forward(x, norm_weight_1, eps = 1e-5)

        #* compress the (copy of) x
        with rf("F02.clone_x"):
            x_copy = x.clone()
        with rf("F03.compress_x_chan"):
            x_o, x_q, x_channel_idx, x_scale = compression_pack_channel_base(
                x=x_copy, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['x']
            )

        # compute q,k,v
        # forward process: q_proj
        with rf("F04.lora_fwd_q"):
            q, q_main, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x_norm_1)

        # forward process: k_proj
        with rf("F05.lora_fwd_k"):
            k, k_main, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x_norm_1)

        # forward process: v_proj
        with rf("F06.lora_fwd_v"):
            v, v_main, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x_norm_1)

        #* compress x_norm_1
        with rf("F07.compress_x_norm_1"):
            x_norm_1_q, x_norm_1_scale = compression_pack_quant_base(
                x=x_norm_1, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['x_norm_1']
            )
            del x_norm_1

        #* quantize q_main, k_main and v_main in original version, then add lora and apply rope
        with rf("F08.compress_q_main"):
            q_main_q, q_main_scale = compression_pack_quant_base(
                x=q_main, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['q']
            )
        with rf("F09.compress_k_main"):
            k_main_q, k_main_scale = compression_pack_quant_base(
                x=k_main, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['k']
            )
        with rf("F10.compress_v_main"):
            v_main_q, v_main_scale = compression_pack_quant_base(
                x=v_main, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['v']
            )
            del q_main, k_main, v_main

        # reshape
        with rf("F11.reshape_qkv"):
            q = hidden_to_head_shape(q, num_heads)
            k = hidden_to_head_shape(k, num_key_value_heads)
            v = hidden_to_head_shape(v, num_key_value_heads)

        ctx.q_shape = q.shape

        with rf("F12.rope_qk"):
            q = rope_forward(q.transpose(1, 2), cos, sin).transpose(1, 2)
            k = rope_forward(k.transpose(1, 2), cos, sin).transpose(1, 2)

        # forward: S = Q @ K.T / sqrt(d_k)
        with rf("F13.repeat_kv_k"):
            if num_heads != num_key_value_heads:
                k = repeat_kv(k, n_rep=num_heads // num_key_value_heads)

        with rf("F14.qk_gemm"):
            s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
            del q, k

        # apply mask
        with rf("F15.add_mask"):
            if attention_mask is not None:
                s = s + attention_mask

        # forward: softmax
        with rf("F16.softmax"):
            a = torch.softmax(s, dim=-1, dtype=v.dtype)  # [bsz, num_heads, q_len, q_len]
            del s

        #* compress a
        with rf("F17.compress_attn_map"):
            a_o, a_threshold = compression_pack_softmax_base(
                x=a, o_ratio=softmax_outlier_ratio, it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['a']
            )
        ctx.a_shape = a.shape

        # forward: O = A @ V
        with rf("F18.repeat_kv_v"):
            if num_heads != num_key_value_heads:
                v = repeat_kv(v, n_rep=num_heads // num_key_value_heads)

        with rf("F19.av_gemm"):
            o = a @ v
            del a, v

        # reshape
        with rf("F20.reshape_o"):
            o = head_to_hidden_shape(o)

        # forward process: o_proj
        with rf("F21.lora_fwd_o"):
            o_final, _, o_final_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o)

        #* compress o
        with rf("F22.compress_o"):
            o_q, o_scale = compression_pack_quant_base(
                x=o, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['o']
            )
            del o

        # residual connection
        with rf("F23.residual_1"):
            x_medium = x + o_final
            del x, o_final

        # layernorm or rmsnorm
        with rf("F24.rmsnorm_2"):
            x_norm_2, mean_2, rstd_2, block_size, num_warps = rmsnorm_forward(x_medium, norm_weight_2, eps = 1e-5)

        #* compress the (copy of) x_medium
        with rf("F25.clone_x_medium"):
            x_medium_copy = x_medium.clone()
        with rf("F26.compress_x_medium_chan"):
            x_medium_o, x_medium_q, x_medium_channel_idx, x_medium_scale = compression_pack_channel_base(
                x=x_medium_copy, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['x_medium']
            )

        # forward process: gate_proj
        with rf("F27.lora_fwd_gate"):
            gate, gate_main, gate_lora_a = lora_forward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, b_gate, x_norm_2)
        with rf("F28.lora_fwd_up"):
            up, up_main, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_norm_2)

        #* compress the x_norm_2
        with rf("F29.compress_x_norm_2"):
            x_norm_2_q, x_norm_2_scale = compression_pack_quant_base(
                x=x_norm_2, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['x_norm_2']
            )
            del x_norm_2

        #* compress the gate_main
        with rf("F30.compress_gate_main"):
            gate_main_q, gate_main_scale = compression_pack_quant_base(
                x=gate_main, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['gate']
            )
            del gate_main

        #* compress the up_main
        with rf("F31.compress_up_main"):
            up_main_q, up_main_scale = compression_pack_quant_base(
                x=up_main, q_bit=q_bit,
                q_method='per-channel', it_num=iteration,
                it_num_thd=iteration_threshold, static_value=static_value['up']
            )
            del up_main

        # hadamard
        with rf("F32.silu_hadamard"):
            hadamard = silu_hadamard(gate, up)
            del gate, up

        # forward process: down_proj
        with rf("F33.lora_fwd_down"):
            down, _, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, hadamard)
            del hadamard

        # residual connection
        with rf("F34.residual_2"):
            x_out = x_medium + down
            ctx.seq_length = x_out.shape[1]
            del x_medium, down
        
        ctx.save_for_backward(
            ### buffered activation (attention) ###
            x_o, x_q, x_scale, # x
            mean_1, rstd_1, # buffer for rmsnorm
            x_norm_1_q, x_norm_1_scale, # x_norm_1
            cos, sin, # buffer for rope
            q_main_q, q_main_scale, # q
            k_main_q, k_main_scale, # k
            v_main_q, v_main_scale, # v
            a_o, a_threshold, # a
            o_q, o_scale, # o
            q_lora_a, k_lora_a, v_lora_a, # buffer for lora (qkv)
            o_final_lora_a, # buffer for lora (o)
            ### buffered activation (mlp) ###
            mean_2, rstd_2, # buffer for rmsnorm
            x_medium_o, x_medium_q, x_medium_scale, # x_medium
            x_norm_2_q, x_norm_2_scale, # x_norm_2
            gate_main_q, gate_main_scale, # gate
            up_main_q, up_main_scale, # up
            gate_lora_a, up_lora_a, down_lora_a,
            ### weights (attention) ###
            norm_weight_1, norm_bias_1,
            w_q, b_q, w_q_lora_a, w_q_lora_b,
            w_k, b_k, w_k_lora_a, w_k_lora_b,
            w_v, b_v, w_v_lora_a, w_v_lora_b,
            w_o, b_o, w_o_lora_a, w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2, norm_bias_2,
            w_gate, b_gate, w_gate_lora_a, w_gate_lora_b,
            w_up, b_up, w_up_lora_a, w_up_lora_b,
            w_down, b_down, w_down_lora_a, w_down_lora_b,
        )
        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_gate_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        )
        ctx.input_layernorm_channel = x_channel_idx 
        ctx.post_layernorm_channel = x_medium_channel_idx
        ctx.num_heads = num_heads
        ctx.num_key_value_heads = num_key_value_heads
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        ctx.q_bit = q_bit

        return x_out, x_channel_idx, x_scale, \
            x_norm_1_scale, \
            q_main_scale, k_main_scale, v_main_scale, \
            a_threshold, o_scale, \
            x_medium_channel_idx, x_medium_scale, \
            x_norm_2_scale, \
            gate_main_scale, up_main_scale
            
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_gate_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            ### buffered activation (attention) ###
            x_o, x_q, x_scale, # x
            mean_1, rstd_1, # buffer for rmsnorm
            x_norm_1_q, x_norm_1_scale, # x_norm_1
            cos, sin, # buffer for rope
            q_main_q, q_main_scale, # q
            k_main_q, k_main_scale, # k
            v_main_q, v_main_scale, # v
            a_o, a_threshold, # a
            o_q, o_scale, # o
            q_lora_a, k_lora_a, v_lora_a, # buffer for lora (qkv)
            o_final_lora_a, # buffer for lora (o)
            ### buffered activation (mlp) ###
            mean_2, rstd_2, # buffer for rmsnorm
            x_medium_o, x_medium_q, x_medium_scale, # x_medium
            x_norm_2_q, x_norm_2_scale, # x_norm_2
            gate_main_q, gate_main_scale, # gate
            up_main_q, up_main_scale, # up
            gate_lora_a, up_lora_a, down_lora_a,
            ### weights (attention) ###
            norm_weight_1, norm_bias_1,
            w_q, b_q, w_q_lora_a, w_q_lora_b,
            w_k, b_k, w_k_lora_a, w_k_lora_b,
            w_v, b_v, w_v_lora_a, w_v_lora_b,
            w_o, b_o, w_o_lora_a, w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2, norm_bias_2,
            w_gate, b_gate, w_gate_lora_a, w_gate_lora_b,
            w_up, b_up, w_up_lora_a, w_up_lora_b,
            w_down, b_down, w_down_lora_a, w_down_lora_b,
        ) = ctx.saved_tensors
        
        #! recompute flow, use gate and up to reconstruct fn and hadamard
        #* dequantize gate_main
        with rf("B01.dequant_gate_main"):
            gate_main = decompression_dequantization(gate_main_q, gate_main_scale, ctx.q_bit)
            del gate_main_q, gate_main_scale
        with rf("B02.readd_gate_lora"):
            gate = gate_main + (gate_lora_a.to(gate_main.dtype)) @ w_gate_lora_b.to(gate_main.dtype)
            del gate_main

        # TODO: write a fused silu-hadamard triton kernel
        with rf("B03.silu_recompute"):
            fn = torch.nn.functional.silu(gate)

        #* dequantize up_main
        with rf("B04.dequant_up_main"):
            up_main = decompression_dequantization(up_main_q, up_main_scale, ctx.q_bit)
            del up_main_q, up_main_scale
        with rf("B05.readd_up_lora"):
            up = up_main + (up_lora_a.to(up_main.dtype)) @ w_up_lora_b.to(up_main.dtype)
            del up_main

        with rf("B06.hadamard_recompute"):
            hadamard = up * fn

        # down proj part
        with rf("B07.lora_bwd_down"):
            grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, hadamard, down_lora_a, grad_output)
            del hadamard, down_lora_a

        # hadamard
        with rf("B08.hadamard_bwd"):
            grad_hadamard_1 = grad_down * up
            grad_hadamard_2 = grad_down * fn
            del grad_down, up, fn

        with rf("B09.silu_bwd"):
            grad_fn = silu_backward(gate, grad_hadamard_1)
            del gate, grad_hadamard_1

        #* dequantize x_norm_2
        with rf("B10.dequant_x_norm_2"):
            x_norm_2 = decompression_dequantization(x_norm_2_q, x_norm_2_scale, ctx.q_bit)
            del x_norm_2_q, x_norm_2_scale

        # gate proj part
        with rf("B11.lora_bwd_gate"):
            grad_w_gate_lora_a, grad_w_gate_lora_b, grad_gate = lora_backward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, x_norm_2, gate_lora_a, grad_fn)
            del gate_lora_a, grad_fn

        # up proj part
        with rf("B12.lora_bwd_up"):
            grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_norm_2, up_lora_a, grad_hadamard_2)
            grad_gate_up = grad_up + grad_gate
            del up_lora_a, grad_hadamard_2, grad_up, grad_gate, x_norm_2

        #* dequantize x_medium
        with rf("B13.dequant_x_medium"):
            x_medium = outlier_addition_fuse_decompression_dequantization(x_medium_q, x_medium_scale, x_medium_o, ctx.post_layernorm_channel, ctx.q_bit)
            del x_medium_q, x_medium_scale, x_medium_o

        # layernorm & rmsnorm backward
        with rf("B14.rmsnorm_bwd_2"):
            grad_norm_2, _ = rmsnorm_backward(
                grad_gate_up, x_medium, norm_weight_2, mean_2, rstd_2, # TODO: other params
                True, 1e-5, ctx.num_warps, ctx.block_size
            )
            del x_medium, grad_gate_up

        # residual connection
        with rf("B15.residual_bwd_2"):
            grad_medium = grad_norm_2 + grad_output
            del grad_norm_2, grad_output

        #* dequantize o
        with rf("B16.dequant_o"):
            o = decompression_dequantization(o_q, o_scale, ctx.q_bit)
            del o_q, o_scale

        # o part
        with rf("B17.lora_bwd_o"):
            grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_medium)
            del o, o_final_lora_a

        # reshape
        with rf("B18.reshape_grad_o"):
            grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)

        #* dequantize a, v
        with rf("B19.attn_map_to_dense"):
            a = a_o.to_dense()
            a = a.reshape(ctx.a_shape)
        with rf("B20.dequant_v_rebuild"):
            v_main = decompression_dequantization(v_main_q, v_main_scale, ctx.q_bit)
            v = v_main + (v_lora_a.to(v_main.dtype)) @ w_v_lora_b.to(v_main.dtype)
            v = hidden_to_head_shape(v, ctx.num_key_value_heads)
            if ctx.num_heads != ctx.num_key_value_heads:
                v = repeat_kv(v, n_rep=ctx.num_heads // ctx.num_key_value_heads)
            del a_o, v_main_q, v_main_scale, v_main

        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        with rf("B21.av_gemm_bwd"):
            grad_v = a.transpose(-2, -1) @ grad_o
            grad_a = grad_o @ v.transpose(-2, -1)
            if ctx.num_heads != ctx.num_key_value_heads:
                grad_v = repeat_kv_backward(grad_v, n_rep=ctx.num_heads // ctx.num_key_value_heads)
            del grad_o, v

        # backward of softmax
        with rf("B22.softmax_bwd"):
            grad_s = softmax_backward(a, grad_a)
            del a, grad_a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        with rf("B23.scale_grad_s"):
            grad_s = grad_s / math.sqrt(ctx.head_dim)

        #* dequantize q, k, then apply rope
        with rf("B24.dequant_q_rebuild"):
            q_main = decompression_dequantization(q_main_q, q_main_scale, ctx.q_bit)
            del q_main_q, q_main_scale
            q = q_main + (q_lora_a.to(q_main.dtype)) @ w_q_lora_b.to(q_main.dtype)
            q = hidden_to_head_shape(q, ctx.num_heads)
            del q_main
            q = rope_forward(q.transpose(1, 2), cos, sin).transpose(1, 2)
        with rf("B25.qk_gemm_bwd_k"):
            grad_k = grad_s.transpose(-2, -1) @ q

        with rf("B26.dequant_k_rebuild"):
            k_main = decompression_dequantization(k_main_q, k_main_scale, ctx.q_bit)
            del k_main_q, k_main_scale
            k = k_main + (k_lora_a.to(k_main.dtype)) @ w_k_lora_b.to(k_main.dtype)
            k = hidden_to_head_shape(k, ctx.num_key_value_heads)
            del k_main
            k = rope_forward(k.transpose(1, 2), cos, sin).transpose(1, 2)
            if ctx.num_heads != ctx.num_key_value_heads:
                k = repeat_kv(k, n_rep=ctx.num_heads // ctx.num_key_value_heads)
                grad_k = repeat_kv_backward(grad_k, n_rep=ctx.num_heads // ctx.num_key_value_heads)
        with rf("B27.qk_gemm_bwd_q"):
            grad_q = grad_s @ k
            del grad_s, k

        with rf("B28.rope_bwd"):
            BLOCK_SIZE, num_warps = calculate_settings(ctx.head_dim // 2)
            N_GROUPS = 128
            grad_q = rope_backward(grad_q.transpose(1, 2), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2) # TODO: other params
            grad_k = rope_backward(grad_k.transpose(1, 2), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2)

        with rf("B29.reshape_grad_qkv"):
            grad_q = head_to_hidden_shape(grad_q)
            grad_k = head_to_hidden_shape(grad_k)
            grad_v = head_to_hidden_shape(grad_v)

        #* dequantize x_norm_1
        with rf("B30.dequant_x_norm_1"):
            x_norm_1 = decompression_dequantization(x_norm_1_q, x_norm_1_scale, ctx.q_bit)
            del x_norm_1_q, x_norm_1_scale

        # backward of q_proj
        with rf("B31.lora_bwd_q"):
            grad_w_q_lora_a, grad_w_q_lora_b, grad_x = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x_norm_1, q_lora_a, grad_q)

        # backward of k_proj
        with rf("B32.lora_bwd_k"):
            grad_w_k_lora_a, grad_w_k_lora_b, grad_x_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x_norm_1, k_lora_a, grad_k)
            grad_x += grad_x_temp

        # backward of v_proj
        with rf("B33.lora_bwd_v"):
            grad_w_v_lora_a, grad_w_v_lora_b, grad_x_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x_norm_1, v_lora_a, grad_v)
            grad_x += grad_x_temp

        #* dequantize x
        with rf("B34.dequant_x"):
            x = outlier_addition_fuse_decompression_dequantization(x_q, x_scale, x_o, ctx.post_layernorm_channel, ctx.q_bit)
            del x_q, x_scale, x_o

        # layernorm or rmsnorm backward
        with rf("B35.rmsnorm_bwd_1"):
            grad_norm_1, _ = rmsnorm_backward(
                grad_x, x, norm_weight_1, mean_1, rstd_1, # TODO: other params
                True, 1e-5, ctx.num_warps, ctx.block_size
            )

        # residual connection
        with rf("B36.residual_bwd_1"):
            grad_input = grad_norm_1 + grad_medium
        
        return (
            grad_input,
            #############attention part#############
            None,
            None,
            ####################################
            None,
            None,
            ####################################
            None,
            None,
            None,
            grad_w_q_lora_a,
            grad_w_q_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_k_lora_a,
            grad_w_k_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_v_lora_a,
            grad_w_v_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_o_lora_a,
            grad_w_o_lora_b,
            ####################################
            None,
            None,
            ####################################
            None,
            None,
            None,
            grad_w_gate_lora_a,
            grad_w_gate_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_up_lora_a,
            grad_w_up_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_down_lora_a,
            grad_w_down_lora_b
        ) + (None,) * 10


class FusedLlamaLayerIntraInter(torch.nn.Module):
    def __init__(
        self,
    ):
        super(FusedLlamaLayerIntraInter, self).__init__()
        self.iteration = 0
        self.static_value = {
            'x': {'outlier_channel_index': None, 'scale': None},
            'x_norm_1': {'scale': None},
            'q': {'scale': None},
            'k': {'scale': None},
            'v': {'scale': None},
            'a': {'outlier': None},
            'o': {'scale': None},
            'x_medium': {'outlier_channel_index': None, 'scale': None},
            'x_norm_2': {'scale': None},
            'gate': {'scale': None},
            'up': {'scale': None},
        }


    def set_hyclora_config(self, hyclora_config):
        self.hyclora_config = hyclora_config
        self.use_hyclora = hyclora_config.use_hyclora
        self.iteration_threshold = hyclora_config.iteration_threshold
        self.softmax_outlier_ratio = hyclora_config.softmax_outlier_ratio
        self.layernorm_outlier_ratio = hyclora_config.layernorm_outlier_ratio
        self.q_bit = hyclora_config.q_bit
        
        
    def forward(
        self,
        input: torch.Tensor,
        ############################################
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_proj_base: bnb.nn.modules.Linear4bit,
        q_proj_lora_a: torch.nn.Linear,
        q_proj_lora_b: torch.nn.Linear,
        k_proj_base: bnb.nn.modules.Linear4bit,
        k_proj_lora_a: torch.nn.Linear,
        k_proj_lora_b: torch.nn.Linear,
        v_proj_base: bnb.nn.modules.Linear4bit,
        v_proj_lora_a: torch.nn.Linear,
        v_proj_lora_b: torch.nn.Linear,
        o_proj_base: bnb.nn.modules.Linear4bit,
        o_proj_lora_a: torch.nn.Linear,
        o_proj_lora_b: torch.nn.Linear,
        ############################################
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        gate_proj_base: bnb.nn.modules.Linear4bit,
        gate_proj_lora_a: torch.nn.Linear,
        gate_proj_lora_b: torch.nn.Linear,
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
        ############################################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int,
    ):
        y, x_channel_idx, x_scale, \
        x_norm_1_scale, \
        q_main_scale, k_main_scale, v_main_scale, \
        a_threshold, o_scale, \
        x_medium_channel_idx, x_medium_scale, \
        x_norm_2_scale, \
        gate_main_scale, up_main_scale = FusedLlamaLayerIntraInterFunc.apply(
            input,
            #############attention part#############
            norm_weight_1,
            norm_bias_1,
            ####################################
            cos,
            sin,
            ####################################
            q_proj_base.weight,
            q_proj_base.bias,
            q_proj_base.weight.quant_state,
            q_proj_lora_a.default.weight.T,
            q_proj_lora_b.default.weight.T,
            ####################################
            k_proj_base.weight,
            k_proj_base.bias,
            k_proj_base.weight.quant_state,
            k_proj_lora_a.default.weight.T,
            k_proj_lora_b.default.weight.T,
            ####################################
            v_proj_base.weight,
            v_proj_base.bias,
            v_proj_base.weight.quant_state,
            v_proj_lora_a.default.weight.T,
            v_proj_lora_b.default.weight.T,
            ####################################
            o_proj_base.weight,
            o_proj_base.bias,
            o_proj_base.weight.quant_state,
            o_proj_lora_a.default.weight.T,
            o_proj_lora_b.default.weight.T,
            #############mlp part#############
            norm_weight_2,
            norm_bias_2,
            ####################################
            gate_proj_base.weight,
            gate_proj_base.bias,
            gate_proj_base.weight.quant_state,
            gate_proj_lora_a.default.weight.T,
            gate_proj_lora_b.default.weight.T,
            ####################################
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.default.weight.T,
            up_proj_lora_b.default.weight.T,
            ####################################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.default.weight.T,
            down_proj_lora_b.default.weight.T,
            ####################################
            attention_mask,
            num_heads,
            head_dim,
            num_key_value_heads,
            ####################################
            self.iteration,
            self.iteration_threshold,
            self.static_value,
            self.softmax_outlier_ratio,
            self.layernorm_outlier_ratio,
            self.q_bit,
        )
        
        if self.iteration < self.iteration_threshold:
            self.static_value['x'] = update_dict(self.static_value['x'], {'outlier_channel_index': x_channel_idx, 'scale': x_scale}, self.iteration)
            self.static_value['x_norm_1'] = update_dict(self.static_value['x_norm_1'], {'scale': x_norm_1_scale}, self.iteration)
            self.static_value['q'] = update_dict(self.static_value['q'], {'scale': q_main_scale}, self.iteration)
            self.static_value['k'] = update_dict(self.static_value['k'], {'scale': k_main_scale}, self.iteration)
            self.static_value['v'] = update_dict(self.static_value['v'], {'scale': v_main_scale}, self.iteration)
            self.static_value['a'] = update_dict(self.static_value['a'], {'outlier': a_threshold}, self.iteration)
            self.static_value['o'] = update_dict(self.static_value['o'], {'scale': o_scale}, self.iteration)
            self.static_value['x_medium'] = update_dict(self.static_value['x_medium'], {'outlier_channel_index': x_medium_channel_idx, 'scale': x_medium_scale}, self.iteration)
            self.static_value['x_norm_2'] = update_dict(self.static_value['x_norm_2'], {'scale': x_norm_2_scale}, self.iteration)
            self.static_value['gate'] = update_dict(self.static_value['gate'], {'scale': gate_main_scale}, self.iteration)
            self.static_value['up'] = update_dict(self.static_value['up'], {'scale': up_main_scale}, self.iteration)
        
        self.iteration += 1
        
        return y