import torch
import torch.nn as nn
from typing import List
from torch.utils.data import Dataset, DataLoader

class TrainerConfig:
    image_augmentation: bool
    image_encoder: nn.Module
    mask_decoder: nn.Module
    learning_rate: float
    lr_scheduler: List[float]
    checkpoint_steps :int = 10
    dataset : str

    output_dir:str # Directory to save the Results of the Training

class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config  = config
        if config.dataset 
        self.dataloader = DataLoader()
    def train(self):
        pass
