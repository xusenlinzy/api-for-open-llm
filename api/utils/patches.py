import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import rotate_half, LlamaAttention, repeat_kv

TRAINING_LENGTH = 4096
WINDOW_SIZE = 512
old_init = modeling_llama.LlamaRotaryEmbedding.__init__
old_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb


def ntk_rope_mixed_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    old_init(self, dim, max_position_embeddings, base, device)
    k, b = 12, 0.75
    max_position_embeddings = TRAINING_LENGTH * k
    a = np.log(k) / (dim / 2)**b
    inv_freq = base**(-torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq *= (-a * torch.arange(1, dim // 2 + 1).float().to(device)**b).exp()
    self.register_buffer('inv_freq', inv_freq)
    self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())


def apply_rotary_pos_emb_and_logn_scale(q, k, cos, sin, position_ids):
    q_embed, k_embed = old_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    scale = ((position_ids + 1)[:, None, :, None].log() / np.log(TRAINING_LENGTH)).clip(1)
    return q_embed * scale.to(q_embed.dtype), k_embed


def apply_ntk_scaling_patch(training_length: int = 4096):
    """ https://kexue.fm/archives/9706 """
    global TRAINING_LENGTH
    TRAINING_LENGTH = training_length
    modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_rope_mixed_init
    modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb_and_logn_scale
    logger.info(f"Apply NTK scaling with TRAINING_LENGTH={TRAINING_LENGTH}")


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos[:, :, -q.shape[2]:]) + (rotate_half(q) * sin[:, :, -q.shape[2]:]) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed


def forward_with_rerope(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    query_states *= ((position_ids + 1)[:, None, :, None].log() / np.log(TRAINING_LENGTH)).clip(1).to(
        query_states.dtype)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
        position_ids = torch.cat([past_key_value[2], position_ids], dim=1)

    past_key_value = (key_states, value_states, position_ids) if use_cache else None

    if q_len == 1:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        position_ids = (position_ids[:, -1] - position_ids).clip(max=WINDOW_SIZE)
        _, key_states = apply_rotary_pos_emb(None, key_states, cos, -sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, WINDOW_SIZE))
        query_states1, key_states1 = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states2, _ = apply_rotary_pos_emb(query_states, None, cos, sin, position_ids * 0 + WINDOW_SIZE)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states1 = repeat_kv(key_states1, self.num_key_value_groups)
        key_states2 = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights1 = torch.matmul(query_states1, key_states1.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights2 = torch.matmul(query_states2, key_states2.transpose(2, 3)) / math.sqrt(self.head_dim)
        rectified_mask = (position_ids[:, -q_len:, None] - position_ids[:, None]).abs() < WINDOW_SIZE
        attn_weights = torch.where(rectified_mask, attn_weights1, attn_weights2)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_rerope_patch(training_length: int = 4096, window_size: int = 512):
    """ https://spaces.ac.cn/archives/9708 """
    global TRAINING_LENGTH, WINDOW_SIZE
    TRAINING_LENGTH, WINDOW_SIZE = training_length, window_size
    LlamaAttention.forward = forward_with_rerope
    logger.info(f"Apply ReRoPE with TRAINING_LENGTH={TRAINING_LENGTH}")
