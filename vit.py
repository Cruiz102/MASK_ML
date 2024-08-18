import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
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
        use_mask_token = False,
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
        self.use_mask_token = use_mask_token
        self.positional_embedding = positinal_embedding

    def load_config_file(self, file_path: str):
        config_params = read_yaml(file_path)
        if config_params:
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.transformer_config.embedded_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.transformer_config.embedded_size)) if use_mask_token else None
        self.projection = nn.Conv2d(config.num_channels,config.transformer_config.embedded_size, config.patch_size, config.patch_size)
        height_feature_map, width_feature_map = calculate_conv2d_output_dimensions(self.config.image_size, self.config.image_size, config.encoder_stride, config.patch_size)
        self.sequence_length  = height_feature_map * width_feature_map

        self.pos_embed: Optional[nn.Parameter] = None
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

        B,C, H, W = images.shape

        embeddings = self.projection(images).flatten(2).transpose(1, 2)
        if self.config.positional_embedding:
            embeddings += self.pos_embed
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        return embeddings

class VitModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super(VitModel, self).__init__()
        self.config = config
        self.embedded_layer = VITPatchEncoder(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config.transformer_config) for _ in range(self.config.transformer_blocks)])
    def load_config(self):
        pass

    def forward(self, x: Union[Tensor, np.ndarray, List[Image.Image]], attention_heads_idx: List[int]):
        y = self.embedded_layer(x) # Size (Batch, sequence_length, embedded_size)
        for block in self.transformer_blocks:
            y = block(y)
            print(y.shape)
        return y
    

class ClassificationConfig:
    def __init__(self,
                 model: nn.Module,
                 input_size: int,
                 num_classes : int,
                 labels:List[str]
                 ):
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.labels = labels
class VitClassificationHead(nn.Module):
    def __init__(self, config: ClassificationConfig):
        self.model = config.model
        self.config = config
        self.linear_classifier = nn.Linear(config.input_size, config.num_classes)
    def forward(self, x: Union[Tensor, np.ndarray, List[Image.Image]]):

        outputs = self.model()
        sequence_output = outputs[0]
        # The 0 index is the [CLS] Token.
        logits = self.linear_classifier(sequence_output[:, 0, :])

        probs = F.softmax(logits)
        return probs



if __name__ == "__main__":
    from datasets import CocoSegmentationDataset
    dataset  = CocoSegmentationDataset()
    image,  mask = dataset[0]
    image = image.unsqueeze(0)
    config = ViTConfig()
    model = VitModel(config)
    print(model(image.float()).shape)