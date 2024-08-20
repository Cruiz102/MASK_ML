import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from datasets import create_dataloader
from typing import List
from vit import ViTConfig, VitModel
from mask_auto_encoder import SegmentationAutoEncoder
import os
import tqdm

@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg['dataset']
    data_image_dir =  cfg[dataset][dataset]['base_dir'] + cfg[dataset][f'{dataset}_validation']['image_directory']
    data_annotation_dir = cfg[dataset][dataset]['base_dir'] + cfg[dataset][f'{dataset}_validation']['annotation_directory']
    batch_size = cfg['batch_size']
    task = cfg['task']
    attention_heads_to_visualize = cfg['visualization']['attention_heads']

    dataloader = create_dataloader(dataset, image_dir=data_image_dir, annotation_dir= data_annotation_dir,batch_size=batch_size)

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

   # Create a directory to save the outputs and visualizations
    output_dir = os.path.join(cfg.output_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    if task == 'classification':
        all_true_labels = []
        all_predicted_labels = []

    with torch.no_grad():
        for i, (imgs, label) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if isinstance(model, SegmentationAutoEncoder):
                is_encoder_transformer = isinstance(model.encoder, VitModel)
                is_decoder_transformer = isinstance(model.decoder, VitModel)
            if isinstance(model, VitModel) or is_encoder_transformer:
                outputs = model(imgs.float(), attention_heads_idx=attention_heads_to_visualize)
            print(f"Output shape: {outputs.shape}")
if __name__ == "__main__":
    run_evaluation()
