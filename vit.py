import torch
from torch import nn
import numpy
import cv2
import requests
import math
from typing import Optional, Union, Tuple, List
from PIL import Image
import logging
import torch.functional as F
import numpy as np
from utils import read_yaml
from torchvision.transforms.functional import to_tensor


class TransformerConfig:
    def __init__(self) -> None:

        self.embedded_size=  200
        self.attention_heads = 5
        self.mlp_hidden_size = 2
        self.mlp_layers = 2
        self.dropout_prob = 0.2

class ViTConfig:
    def __init__(
        self,
        config_file_path: Optional[str] = None,
        transformer_config: TransformerConfig = TransformerConfig(),
        transformer_block=10,
        image_size=224,
        patch_size=16,
        num_channels=3,
        encoder_stride=16
    ):
        if config_file_path:
            self.load_config_file(config_file_path)

        self.transformer_config = transformer_config
        self.transformer_blocks = transformer_block,
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels= num_channels
        self.encoder_stride = encoder_stride

    def load_config_file(self, file_path: str):
        config_params = read_yaml(file_path)
        self.transformer_config.attention_heads = config_params['attention_heads']
        self.transformer_config.mlp_hidden_size = config_params['mlp_hidden_size']
        self.transformer_config.mlp_layers = config_params['mlp_layers']
        self.transformer_config.dropout_prob = config_params['dropout_prob']

        self.image_size = config_params['image_size']
        self.patch_size = config_params['patch_size']
        self.transformer_blocks = config_params['transformer_blocks']
        self.num_channels = config_params['num_channels']
        self.encoder_stride = config_params['encoder_stride']





class MultiHeadAttention(nn.Module):
    def __init__(self, heads, all_head_size, embedd_size, bias):
        super(self).__init__()
        self.attention_heads = heads
        self.all_head_size = all_head_size
        self.c_attn = nn.Linear(embedd_size, 3 * all_head_size, bias=bias)

    
    # Forward Attention by https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.attention_heads, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(self).__init__()
        self.hidden_size = config.hidden_size
        self.first_norm_layer = nn.LayerNorm(config.embedded_size)
        self.head_attention = MultiHeadAttention()
        self.norm_layer = nn.LayerNorm(config.embedded_size)
        self.feed_forward = [nn.Linear(config.embedded_size) for i in range(config.mlp_hidden_size)]

    def forward(self, x):
        x = self.head_attention(x)
        x = self.norm_layer(x)





class VITPatchEncoder(nn.Module):
    def __init__(self,config: ViTConfig):
        super(VITPatchEncoder, self).__init__()
        self.projection = nn.Conv2d(config.num_channels,config.transformer_config.embedded_size, config.patch_size, config.patch_size)
        # self.positional_embedding = torch.ParameterDict()
    def forward(self, images: Union[torch.Tensor, np.ndarray, List[Image.Image]]):

        if isinstance(images, list):
            images = to_tensor(images)
        elif isinstance(images, np.ndarray):
            images = to_tensor(images)
        embeddings = self.projection(images) #.flatten(2).transpose(1, 2)
        return embeddings

class VitModel(nn.Module):
    def __init__(self, config: ViTConfig):
        self.config = config
        self.embedded_layer = VITPatchEncoder(config)
        self.attention_blocks = [TransformerBlock(config.transformer_config) for i in range(self.config.transformer_block)]
    def load_config(self):
        pass

    def forward(self, x: Union[torch.tensor, np.ndarray, List[Image.Image]]):
        x = self.embedded_layer(x) # Size (Batch, sequence_length, embedded_size)
        for block in self.attention_blocks:
            x = block(x)
        return x
    
def train(model):
    steps = 0
    lr = 0.5
    lr_scheduling = 0.2
    momemtum = 0.2
    batch_size = 1
    gradients_history = 2
    gradient_clipping = 0.2
    vit_model = VitModel()
    optimizer = torch.optim.SGD(vit_model.parameters(), lr)
    for i in range(steps):
        
        print(i)
