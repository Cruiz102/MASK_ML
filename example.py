import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from utils import read_yaml
from datasets import CocoSegmentationDataset
from typing import List
from vit import ViTConfig, VitModel

@hydra.main(config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run_evaluation()
