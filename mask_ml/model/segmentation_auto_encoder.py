import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List
import numpy as np
from PIL import Image
from mask_ml.model.vit import VitModel, ViTConfig
from mask_ml.model.mask_decoder import MaskDecoderConfig, MaskDecoder
from torchvision.transforms.functional import to_tensor

from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class SegmentationAutoEncoderConfig:
    # ViTConfig parameters
    encoder_config: ViTConfig
    decoder_config: MaskDecoderConfig

class SegmentationAutoEncoder(nn.Module):
    def __init__(self, config: SegmentationAutoEncoderConfig ) -> None:
        super(SegmentationAutoEncoder, self).__init__()
        self.config = config
        self.encoder = VitModel(config.encoder_config)
        self.decoder = MaskDecoder(config.decoder_config) 


    def forward(self, images: Union[Tensor, np.ndarray, List[Image.Image]], attention_heads_idx: List[int]):
        if isinstance(images, list):
            images = to_tensor(images)
        elif isinstance(images, np.ndarray):
            images = to_tensor(images)
        x = torch.stack([self.preprocess(img) for img in images], dim=0)

        y = self.encoder(x, attention_heads_idx)
        y = self.decoder(y)
        return y
    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x