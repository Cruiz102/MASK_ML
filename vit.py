import torch
from torch import nn
from torch import Tensor
from typing import Optional, Union, Tuple, List, Literal
from PIL import Image
import logging
import numpy as np
from utils import read_yaml
from torchvision.transforms.functional import to_tensor
from utils import calculate_conv2d_output_dimensions, sinusoidal_positional_encoding
from transformer import  TransformerBlock, TransformerConfig

class ViTConfig:
    def __init__(
        self,
        config_file_path: Optional[str] = None,
        transformer_config: TransformerConfig = TransformerConfig(),
        transformer_blocks: int=10,
        image_size:int = 256,
        patch_size:int=16,
        num_channels:int=3,
        encoder_stride:int=16,
        positinal_embedding: Literal["sinusoidal", 'rotary', 'learned'] = 'sinusoidal' 
    ):
        if config_file_path:
            self.load_config_file(config_file_path)

        self.transformer_config = transformer_config
        self.transformer_blocks = transformer_blocks
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels= num_channels
        self.encoder_stride = encoder_stride
        self.positional_embedding = positinal_embedding

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




class VITPatchEncoder(nn.Module):
    def __init__(self,config: ViTConfig):
        super(VITPatchEncoder, self).__init__()
        self.config = config
        self.projection = nn.Conv2d(config.num_channels,config.transformer_config.embedded_size, config.patch_size, config.patch_size)
        self.pos_embed: Optional[nn.Parameter] = None
        height_feature_map, width_feature_map = calculate_conv2d_output_dimensions(self.config.image_size, self.config.image_size, config.encoder_stride, config.patch_size)
        self.sequence_length  = height_feature_map * width_feature_map
        if config.positional_embedding == "learned":
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros((1, self.sequence_length, self.config.transformer_config.embedded_size ))
            ) 
        elif config.positional_embedding == "sinusoidal":
            self.pos_embed = sinusoidal_positional_encoding(self.sequence_length, self.config.transformer_config.embedded_size)

        

        elif config.positional_embedding == "rotary":
            # The RoPE embeddings are not applied as a absolute vector to add but as a rotation matrix.
            # If using the rotation embeddings this will be executed in the attention Layer before doing 
            # the inner product between the Queries and the Keys.
            pass
    def forward(self, images: Union[torch.Tensor, np.ndarray, List[Image.Image]]):
        if isinstance(images, list):
            images = to_tensor(images)
        elif isinstance(images, np.ndarray):
            images = to_tensor(images)
        
        embeddings = self.projection(images).flatten(2).transpose(1, 2)
        if self.config.positional_embedding:
            embeddings += self.pos_embed
        return embeddings

class VitModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super(VitModel, self).__init__()
        self.config = config
        self.embedded_layer = VITPatchEncoder(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.transformer_config) for _ in range(self.config.transformer_blocks)])
    def load_config(self):
        pass

    def forward(self, x: Union[Tensor, np.ndarray, List[Image.Image]]):
        y = self.embedded_layer(x) # Size (Batch, sequence_length, embedded_size)
        for block in self.transformer_blocks:
            y = block(y)
            print(y.shape)
        return y
    


class VitClassificationHead(nn.Module):
    pass



if __name__ == "__main__":
    from datasets import CocoSegmentationDataset
    dataset  = CocoSegmentationDataset()
    image,  mask = dataset[0]
    image = image.unsqueeze(0)
    config = ViTConfig()
    model = VitModel(config)
    print(model(image.float()).shape)