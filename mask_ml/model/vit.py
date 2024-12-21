import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, List, Literal, Tuple
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
from mask_ml.utils.utils import calculate_conv2d_output_dimensions, sinusoidal_positional_encoding
from mask_ml.model.transformer import  TransformerBlock, TransformerConfig

from dataclasses import dataclass




@dataclass
class ViTConfig:
    # ViTConfig parameters
    transformer_blocks: int = 10
    image_size: int = 256
    patch_size: int = 16
    num_channels: int = 3
    encoder_stride: int = 16
    use_mask_token: bool = False
    positional_embedding: Literal["sinusoidal", "learned"] = "sinusoidal"
    interpolation: bool = False
    interpolation_scale: int = 1

    # TransformerConfig parameters
    embedded_size: int = 200
    attention_heads: int = 5
    mlp_hidden_size: int = 2
    mlp_layers: int = 2
    activation_function: str = "relu"
    dropout_prob: float = 0.2
    flash_attention = False
    rotary_relative_embeddings = False


@dataclass
class VitClassificationConfig:
        model_config : ViTConfig
        input_size : int
        num_classes : int


def find_divisors(n: int):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)  # Add the divisor
            if i != n // i:  # Avoid duplicates for perfect squares
                divisors.append(n // i)
    return sorted(divisors)

def generate_all_valid_configurations(image_size: int):
    """
    Given an image size (assuming a square image), generate 
    all possible valid configurations for the ViT model based on
    the divisors of the image size.
    """
    divisors_of_square_image = find_divisors(image_size)
    configurations = []

    for kernel_size in divisors_of_square_image:
            for stride_size in divisors_of_square_image:

                sequences = calculate_conv2d_output_dimensions(
                    H_in=image_size, W_in=image_size, K=kernel_size, S=kernel_size
                )
                sequences = sequences[0] * sequences[1]

                attention_heads = image_size // kernel_size

                configurations.append({
                    "sequence_per_image": sequences,
                    "attention_heads": attention_heads,
                    "kernel": kernel_size,
                    "stride": stride_size
                })

    return configurations

def check_vit_classification_configuration_validity(config: VitClassificationConfig):
    pass


class VITPatchEncoder(nn.Module):
    def __init__(self,config: ViTConfig):
        super(VITPatchEncoder, self).__init__()
        self.config = config
        self.interpolate_sequence = False
    


        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embedded_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embedded_size)) if self.config.use_mask_token else None
        self.projection = nn.Conv2d(in_channels=config.num_channels,out_channels=config.embedded_size, kernel_size=config.patch_size, stride=config.encoder_stride)
        self.height_feature_map, self.width_feature_map = calculate_conv2d_output_dimensions(
            H_in=self.config.image_size, W_in=self.config.image_size, S=config.encoder_stride, K=config.patch_size)
        self.sequence_length  = self.height_feature_map * self.width_feature_map

        self.pos_embed: Optional[nn.Parameter] = None
        if config.positional_embedding == "learned":
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros((1, self.sequence_length, self.config.embedded_size ))
            ) 
        elif config.positional_embedding == "sinusoidal":
            self.pos_embed = sinusoidal_positional_encoding(self.sequence_length, self.config.embedded_size)
        

        elif config.positional_embedding == "rotary":
            # The RoPE embeddings are not applied as a absolute vector to add but as a rotation matrix.
            # If using the rotation embeddings this will be executed in the attention Layer before doing 
            # the inner product between the Queries and the Keys.
            pass
    def forward(self, images: Union[torch.Tensor, List[Image.Image]]):
        if isinstance(images, list):
            images = to_tensor(images)
        elif isinstance(images, np.ndarray):
            images = to_tensor(images)

        B,C, H, W = images.shape
        embeddings = self.projection(images)  # Shape: (B, embedded_size, H', W')
        if self.config.interpolation:
            # Perform interpolation directly on the projected embeddings
            new_height = int(self.height_feature_map * self.config.interpolation_scale)
            new_width = int(self.width_feature_map * self.config.interpolation_scale)
            embeddings = F.interpolate(embeddings, size=(new_height, new_width), mode='bilinear', align_corners=False)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # Shape: (B, Sequence, embedded_size)
        if self.config.positional_embedding:
            embeddings += self.pos_embed.to(embeddings.device)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        return embeddings

class VitModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super(VitModel, self).__init__()
        self.config = config
        self.transformer_config = TransformerConfig(
            embedded_size=config.embedded_size,
            attention_heads= config.attention_heads,
            mlp_hidden_size=config.mlp_hidden_size,
            mlp_layers=config.mlp_layers,
            activation_function=config.activation_function,
            dropout_prob= config.dropout_prob
            )
        self.embedded_layer = VITPatchEncoder(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.transformer_config) for _ in range(self.config.transformer_blocks)])

    def forward(self, x: Union[Tensor, List[Image.Image]], attention_heads_idx: List[int] = [])-> Tuple[Tensor, List]:
        y = self.embedded_layer(x)  # Size (Batch, sequence_length, embedded_size)
        attention_heads = []
        for i, block in enumerate(self.transformer_blocks):
            y, attn_h = block(y)
            if i in attention_heads_idx:
                attention_heads.append(attn_h)
        return y, attention_heads
    

class VitClassificationHead(nn.Module):
    def __init__(self, config: VitClassificationConfig):
        super(VitClassificationHead, self).__init__()
        self.model = VitModel(config.model_config)
        self.config = config
        self.linear_classifier = nn.Linear(config.input_size, config.num_classes)
    def forward(self, x: Union[Tensor, List[Image.Image]], attention_heads_idx: List[int]=[]):
        outputs, attention_heads = self.model(x, attention_heads_idx)
        logits = self.linear_classifier(outputs[:, 0, :])# The 0 index is the [CLS] Token.
        probs = F.softmax(logits, dim=-1)
        return probs, attention_heads



if __name__ == "__main__":
    from utils.datasets import CocoSegmentationDataset
    dataset  = CocoSegmentationDataset()
    image,  mask = dataset[0]
    image = image.unsqueeze(0)
    config = ViTConfig()
    model = VitModel(config)
    print(model(image.float()).shape)