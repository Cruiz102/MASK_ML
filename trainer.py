import torch
import torch.nn as nn
from typing import List

class TrainerConfig:
    image_augmentation: bool
    image_encoder: nn.Module
    mask_decoder: nn.Module
    learning_rate: float
    lr_scheduler: List[float]
    output_dir:str # Directory to save the Results of the Training

class Trainer:
    def __init__(self) -> None:
        pass
