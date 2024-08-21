import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from utils.utils import read_yaml
from utils.datasets import CocoSegmentationDataset
from typing import List
from model.vit import ViTConfig, VitModel

@hydra.main(config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    print(cfg.model.model_name)


if __name__ == "__main__":
    run_evaluation()
