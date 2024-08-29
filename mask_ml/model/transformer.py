

import torch
from torch import nn
from torch import Tensor
import numpy
import torch.nn.functional as F
import cv2
import requests
import math
from typing import Optional, Union, Tuple, List, Literal

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class TransformerConfig:
    embedded_size : int=  200
    attention_heads: int = 5
    mlp_hidden_size: int = 2
    mlp_layers: int = 2
    activation_function: str = "relu"
    dropout_prob : float = 0.2







class MLP(nn.Module):
    def __init__(self, activation_function: str, input_size: int, output_size: int, hidden_sizes: Optional[List[int]] = None):
        super(MLP, self).__init__()
        if hidden_sizes:
            layer_sizes = [input_size] + hidden_sizes + [output_size]
        else:
            layer_sizes = [input_size, output_size]
        
        # Initialize an empty ModuleList
        self.layers = nn.ModuleList()
        
        # Create layers dynamically and add them to the ModuleList
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Define the activation function based on the string input
        if activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_function == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through all layers except the last one with activation
        for layer in self.layers[:-1]: # type: ignore
            x = self.activation(layer(x))
        
        # No activation on the final layer
        x = self.layers[-1](x)
        return x


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Phi
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class MultiHeadAttention(nn.Module):
    def __init__(self,config: TransformerConfig, rotational_embeddings = False):
        super(MultiHeadAttention,self).__init__()
        self.config = config
        self.rotational_embeddings = rotational_embeddings
        self.flash = False
        self.c_attn = nn.Linear(config.embedded_size, 3 * config.embedded_size, bias=False)
        self.c_proj = nn.Linear(config.embedded_size,  config.embedded_size, bias=True)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)


        if self.rotational_embeddings:
            self.rot_embeddings = RotaryEmbedding(config.embedded_size)

    # Forward Attention by https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
    def forward(self, x, position_ids: Optional[Tensor] = None, return_attention_head = False)-> Union[Tensor, Tuple[Tensor, Tensor]]:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v   = self.c_attn(x).split(self.config.embedded_size, dim=2)
        k = k.view(B, T, self.config.attention_heads, C // self.config.attention_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.config.attention_heads, C // self.config.attention_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.config.attention_heads, C // self.config.attention_heads).transpose(1, 2) # (B, nh, T, hs)


        # Get the positional relative embeddings with  RoPE:
        if self.rotational_embeddings:
            cos, sin = self.rot_embeddings(x)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)


        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if return_attention_head:
            return y, att
        else:
            return y
    
    

    
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(config=config)
        self.norm_layer_1 = nn.LayerNorm(config.embedded_size)
        self.norm_layer_2 = nn.LayerNorm(config.embedded_size)
        hidden_layers = [config.mlp_hidden_size for _ in range(config.mlp_layers) ]
        self.feed_forward = MLP("relu", config.embedded_size, config.embedded_size,hidden_layers)
    def forward(self, x: Tensor, return_attention_head = False) -> Union[Tensor,Tuple[Tensor,Tensor]]:
        y, attention_head = x + self.attention(self.norm_layer_1(x), return_attention_head)
        y = y + self.feed_forward(self.norm_layer_2(y))
        if  return_attention_head:
            return y, attention_head
        else:
            return y

