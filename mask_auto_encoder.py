import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List
import numpy as np
from PIL import Image
from vit import VitModel
from mask_decoder import MaskDecoder

class SegmentationAutoEncoder(nn.Module):
    def __init__(self, encoder: VitModel, decoder: MaskDecoder ) -> None:
        super(SegmentationAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 


    def forward(self, x: Union[Tensor, np.ndarray, List[Image.Image]], attention_heads_idx: List[int]):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        y = self.encoder(x, attention_heads_idx)
        y = self.decoder(y, attention_heads_idx)
        return y
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x