import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from utils.datasets import create_dataloader
from typing import List
from model.vit import ViTConfig, VitModel
from model.mask_decoder import MaskDecoderConfig
from model.segmentation_auto_encoder import SegmentationAutoEncoder, SegmentationAutoEncoderConfig
import os
import tqdm

@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))

    dataset_name = cfg['dataset']
    data_image_dir =  cfg['dataset'][dataset_name]['base_dir'] + cfg['dataset'][f'{dataset_name}_validation']['image_directory']
    data_annotation_dir = cfg['dataset'][dataset_name]['base_dir'] + cfg['dataset'][f'{dataset_name}_validation']['annotation_directory']
    batch_size = cfg['batch_size']
    task = cfg['task']
    attention_heads_to_visualize = cfg['visualization']['attention_heads']

    dataloader = create_dataloader(dataset_name, image_dir=data_image_dir, annotation_dir= data_annotation_dir,batch_size=batch_size, cfg['dataset'])

    model_name = cfg['model']['model_name']

    if model_name == 'vit_classification':
        model_config = ViTConfig(
            transformer_blocks=cfg.model.transformer_blocks,
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            num_channels=cfg.model.num_channels,
            encoder_stride=cfg.model.encoder_stride,
            use_mask_token=False,
            positional_embedding=cfg.model.positional_embedding,
            embedded_size=cfg.model.embedded_size,
            attention_heads=cfg.model.attention_heads,
            mlp_hidden_size=cfg.model.mlp_hidden_size,
            mlp_layers=cfg.model.mlp_layers,
            activation_function= cfg.model.activation_function,
            dropout_prob=cfg.model.dropout_prob)
        

        model = VitModel(model_config)
        
    elif model_name == "SegmentationAutoEncoder":
        encoder_config = ViTConfig(
            transformer_blocks=cfg.model.encoder.transformer_blocks,
            image_size=cfg.model.encoder.image_size,
            patch_size=cfg.model.encoder.patch_size,
            num_channels=cfg.model.encoder.num_channels,
            encoder_stride=cfg.model.encoder.encoder_stride,
            use_mask_token=True,
            positional_embedding=cfg.model.encoder.positional_embedding,
            embedded_size=cfg.model.encoder.embedded_size,
            attention_heads=cfg.model.encoder.attention_heads,
            mlp_hidden_size=cfg.model.encoder.mlp_hidden_size,
            mlp_layers=cfg.model.encoder.mlp_layers,
            activation_function= cfg.model.encoder.activation_function,
            dropout_prob=cfg.model.encoder.dropout_prob)
    
        decoder_config = MaskDecoderConfig(
            transformer_blocks=cfg.model.decoder.transformer_blocks,
            num_multimask_outputs=cfg.model.decoder.num_multitask_outputs,
            iou_mlp_layer_depth=cfg.model.decoder.iou_mlp_layer_depth,
            embedded_size= cfg.model.decoder.embedded_size,
            attention_heads=cfg.model.decoder.attention_heads,
            mlp_hidden_size=cfg.model.encoder.mlp_hidden_size,
            mlp_layers=cfg.model.encoder.mlp_layers,
            activation_function= cfg.model.encoder.activation_function,
            dropout_prob=cfg.model.encoder.dropout_prob)

        model_config = SegmentationAutoEncoderConfig(
            encoder_config=encoder_config,
            decoder_config= decoder_config
        )

        model = SegmentationAutoEncoder(model_config)









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
