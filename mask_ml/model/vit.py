import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, List, Literal, Tuple
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
from mask_ml.utils.utils import calculate_conv2d_output_dimensions, sinusoidal_positional_encoding
from mask_ml.model.transformer import  TransformerBlock

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
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        num_channels: int = 3,
        encoder_stride: int = 16,
        embedded_size: int = 200,
        use_mask_token: bool = False,
        positional_embedding: str = "sinusoidal",
        interpolation: bool = False,
        interpolation_scale: int = 1
    ):
        super(VITPatchEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_stride = encoder_stride
        self.embedded_size = embedded_size
        self.use_mask_token = use_mask_token
        self.positional_embedding = positional_embedding
        self.interpolation = interpolation
        self.interpolation_scale = interpolation_scale

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedded_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedded_size)) if use_mask_token else None
        self.projection = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embedded_size,
            kernel_size=patch_size,
            stride=encoder_stride
        )
        
        self.height_feature_map, self.width_feature_map = calculate_conv2d_output_dimensions(
            H_in=image_size, W_in=image_size, S=encoder_stride, K=patch_size)
        self.sequence_length = self.height_feature_map * self.width_feature_map

        self.pos_embed: Optional[nn.Parameter] = None
        if positional_embedding == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros((1, self.sequence_length, embedded_size))
            )
        elif positional_embedding == "sinusoidal":
            self.pos_embed = sinusoidal_positional_encoding(self.sequence_length, embedded_size)
        elif positional_embedding == "rotary":
            # The RoPE embeddings are not applied as a absolute vector to add but as a rotation matrix.
            # If using the rotation embeddings this will be executed in the attention Layer before doing 
            # the inner product between the Queries and the Keys.
            pass

    def forward(self, images: Union[torch.Tensor, List[Image.Image]]):
        if isinstance(images, list):
            images = to_tensor(images)
        elif isinstance(images, np.ndarray):
            images = to_tensor(images)

        B, C, H, W = images.shape
        embeddings = self.projection(images)  # Shape: (B, embedded_size, H', W')
        if self.interpolation:
            # Perform interpolation directly on the projected embeddings
            new_height = int(self.height_feature_map * self.interpolation_scale)
            new_width = int(self.width_feature_map * self.interpolation_scale)
            embeddings = F.interpolate(embeddings, size=(new_height, new_width), mode='bilinear', align_corners=False)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # Shape: (B, Sequence, embedded_size)
        if self.positional_embedding:
            embeddings += self.pos_embed.to(embeddings.device)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        return embeddings

class VitModel(nn.Module):
    def __init__(
        self,
        transformer_blocks: int = 10,
        image_size: int = 256,
        patch_size: int = 16,
        num_channels: int = 3,
        encoder_stride: int = 16,
        embedded_size: int = 200,
        attention_heads: int = 5,
        mlp_hidden_size: int = 2,
        mlp_layers: int = 2,
        activation_function: str = "relu",
        dropout_prob: float = 0.2,
        use_mask_token: bool = False,
        positional_embedding: str = "sinusoidal",
        interpolation: bool = False,
        interpolation_scale: int = 1,
        flash_attention: bool = False,
        rotary_relative_embeddings: bool = False
    ):
        super(VitModel, self).__init__()
        
        self.embedded_layer = VITPatchEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            encoder_stride=encoder_stride,
            embedded_size=embedded_size,
            use_mask_token=use_mask_token,
            positional_embedding=positional_embedding,
            interpolation=interpolation,
            interpolation_scale=interpolation_scale
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedded_size=embedded_size,
                attention_heads=attention_heads,
                mlp_hidden_size=mlp_hidden_size,
                mlp_layers=mlp_layers,
                activation_function=activation_function,
                dropout_prob=dropout_prob,
                flash_attention=flash_attention,
                rotary_relative_embeddings=rotary_relative_embeddings
            ) for _ in range(transformer_blocks)
        ])

    def forward(self, x: Union[Tensor, List[Image.Image]], attention_heads_idx: List[int] = [])-> Tuple[Tensor, List]:
        y = self.embedded_layer(x)  # Size (Batch, sequence_length, embedded_size)
        attention_heads = []
        for i, block in enumerate(self.transformer_blocks):
            y, attn_h = block(y)
            if i in attention_heads_idx:
                attention_heads.append(attn_h)
        return y, attention_heads
    

class VitClassificationHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        **vit_kwargs
    ):
        super(VitClassificationHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = VitModel(image_size=input_size,**vit_kwargs)
        self.linear_classifier = nn.Linear(input_size, num_classes)

    def forward(self, x: Union[Tensor, List[Image.Image]], attention_heads_idx: List[int]=[]):
        outputs, attention_heads = self.model(x, attention_heads_idx)
        logits = self.linear_classifier(outputs[:, 0, :])
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