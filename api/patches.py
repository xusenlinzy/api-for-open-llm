from typing import Union

import torch
import transformers

ALPHA = 1.0
old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__


def adaptive_ntk_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    self.dim = dim
    self.alpha = ALPHA
    if isinstance(ALPHA, (float, int)):
        base = base * ALPHA ** (dim / (dim - 2))
        self.base = base
    elif ALPHA == 'auto':
        self.base = base
    else:
        raise ValueError(ALPHA)
    old_init(self, dim, max_position_embeddings, base, device)
    ntk_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    self.register_buffer("ntk_inv_freq", ntk_inv_freq, persistent=False)


def adaptive_ntk_forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len_cached:
        if isinstance(self.alpha, (float, int)):
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.ntk_inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.ntk_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
            return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )
        elif self.alpha == 'auto':
            t = torch.arange(seq_len, device=x.device, dtype=self.ntk_inv_freq.dtype)
            dim = self.dim
            alpha = (seq_len / 1024 - 1) * 1.1
            base = self.base * alpha ** (dim / (dim - 2))
            ntk_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(x.device) / dim))

            freqs = torch.einsum("i,j->ij", t, ntk_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            cos_cached = emb.cos()[None, None, :, :]
            sin_cached = emb.sin()[None, None, :, :]
            return (
                cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
            )
    else:
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )


def apply_ntk_scaling_patch(alpha: Union[float, str]):
    global ALPHA
    ALPHA = alpha
    try:
        ALPHA = float(ALPHA)
    except ValueError:
        if ALPHA != "auto":
            raise ValueError(f"Alpha can only be a float or 'auto', but given {ALPHA}")
    print(f"Apply NTK scaling with ALPHA={ALPHA}")

    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward
