import torch
from typing import Tuple
from torch import Tensor
import torch.nn as nn
from transformer import TransformerConfig, TransformerBlock
from torch.nn import functional as F
from utils import sinusoidal_positional_encoding

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderConfig:
    def __init__(self,
        positional_embedding,
        transformer_config: TransformerConfig,
        transformer_blocks: int,
        num_multimask_outputs: int ,
        iou_mlp_layer_depth: int,
                 ) -> None:
        self.positional_embedding = positional_embedding
        self.transformer_config = transformer_config
        self.transformer_blocks = transformer_blocks
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_mlp_layer_depth = iou_mlp_layer_depth



# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):   
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
        
class MaskDecoder:
    def __init__(self, config: MaskDecoderConfig):
        self.config = config
        self.num_multimask_outputs: int = 3
        activation= nn.GELU()
        self.mask_token = nn.Parameter(torch.zeros(config.transformer_config.embedded_size))
        self.image_positional_embedding = sinusoidal_positional_encoding(10,self.config.transformer_config.embedded_size)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.transformer_config) for _ in range(config.transformer_blocks)])

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(config.transformer_config.embedded_size, config.transformer_config.embedded_size // 4, kernel_size=2, stride=2),
            LayerNorm2d(config.transformer_config.embedded_size // 4),
            activation(),
            nn.ConvTranspose2d(config.transformer_config.embedded_size // 4, config.transformer_config.embedded_size // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.masks_mlp =  MLP(config.transformer_config.embedded_size, config.transformer_config.embedded_size,
                           config.transformer_config.embedded_size,
                            config.num_multimask_outputs)

        self.iou_mlp = MLP(config.transformer_config.embedded_size, config.transformer_config.embedded_size,
                           config.transformer_config.embedded_size,
                            config.num_multimask_outputs)
        

    def forward(self, image_embeddings: Tensor) -> Tuple[Tensor, Tensor]:

        batch_size,sequence_length, dimension_size = image_embeddings.shape # (Batch, Sequence Length, Dimension Model)
        expanded_mask = self.mask_token.expand(batch_size, sequence_length, -1)
        tokens = torch.cat([image_embeddings, expanded_mask], dim= 0)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(tokens)

        iou_token_out = x[:, 0, :]
        mask_tokens_out = x[:, 1 : (1 + self.config.num_multimask_outputs), :]\
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        iou_pred = self.iou_mlp(iou_token_out)
        mask_pred = self.masks_mlp(mask_tokens_out)

        return mask_pred, iou_pred
        





        