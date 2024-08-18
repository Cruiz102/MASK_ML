import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from utils import read_yaml
from datasets import CocoSegmentationDataset
from typing import List
from vit import ViTConfig, VitModel

@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))

    # Load dataset
    dataset = CocoSegmentationDataset(cfg.evaluation.dataset_path)
    dataloader = DataLoader(dataset, batch_size=cfg.evaluation.batch_size, shuffle=False)

    # Load the ViT configuration and model
    vit_config = ViTConfig(
        transformer_config=cfg.vit_model.transformer_config,
        transformer_blocks=cfg.vit_model.transformer_blocks,
        image_size=cfg.vit_model.image_size,
        patch_size=cfg.vit_model.patch_size,
        num_channels=cfg.vit_model.num_channels,
        encoder_stride=cfg.vit_model.encoder_stride,
        positinal_embedding=cfg.vit_model.positional_embedding
    )

    model = VitModel(vit_config)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images.float(), attention_heads_idx=[0])
            print(f"Output shape: {outputs.shape}")

if __name__ == "__main__":
    run_evaluation()
