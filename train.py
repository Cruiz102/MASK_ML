
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mask_ml.utils.datasets import create_dataloader
from typing import List
from mask_ml.model.vit import ViTConfig, VitModel,ClassificationConfig, VitClassificationHead
from mask_ml.model.mask_decoder import MaskDecoderConfig
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder, SegmentationAutoEncoderConfig
import os
import tqdm
import numpy as np
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from tqdm import tqdm
from eval import validation_test

def create_unique_experiment_dir(output_dir, experiment_name):
    # Generate the initial experiment directory path
    experiment_dir = os.path.join(output_dir, experiment_name)
    
    # If the directory already exists, append a number to the experiment name
    counter = 1
    while os.path.exists(experiment_dir):
        experiment_dir = os.path.join(output_dir, f"{experiment_name}_{counter}")
        counter += 1
    
    # Create the final directory
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


@hydra.main(version_base=None, config_path="config", config_name="training")
def run_training(cfg: DictConfig):

  # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    dataloader_train, dataloader_test = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg['model']['model_name']
    task = cfg.task
    lr = cfg.learning_rate
    epochs = cfg.epochs
    experiment_name = cfg.experiment_name
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    experiment_dir = create_unique_experiment_dir(output_dir, experiment_name)


    if task == 'classification':
        dataset_name = cfg['dataset']
        num_classes = cfg['datasets'][dataset_name]['num_classes']
        

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

        vit = VitModel(model_config)

        classification_config = ClassificationConfig(
            model=vit,
            input_size=cfg.model.embedded_size,
            num_classes=num_classes,
        )
        model = VitClassificationHead(classification_config)
        
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
            num_multimask_outputs=cfg.model.decoder.num_multimask_outputs,
            iou_mlp_layer_depth=cfg.model.decoder.iou_mlp_depth,
            mlp_hidden_size=cfg.model.encoder.mlp_hidden_size,
            embedded_size= cfg.model.decoder.embedded_size,
            attention_heads=cfg.model.decoder.attention_heads,
            mlp_layers=cfg.model.encoder.transformer_mlp_layers,
            activation_function= cfg.model.encoder.activation_function,
            dropout_prob=cfg.model.encoder.dropout_prob)

        model_config = SegmentationAutoEncoderConfig(
            encoder_config=encoder_config,
            decoder_config= decoder_config
        )

        model = SegmentationAutoEncoder(model_config)

    # if cfg.saved_weights_path:
    #     model.load_state_dict(torch.load(cfg.saved_weights_path))
    #     print(f"Loaded weights from {cfg.saved_weights_path}")

    optimizer = AdamW(model.parameters(), lr=lr)
    # Loss function
    criterion = nn.CrossEntropyLoss()


    loss_file = os.path.join(experiment_dir, "cumulative_losses.csv")
    model = model.to(device)
    with open(loss_file, 'w') as f:
        f.write("epoch,cumulative_loss\n")  # Header for the CSV file
    try:
        for i in range(epochs):
            cumulative_loss = 0.0
            for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {i+1}/{epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                y = model(inputs)
                loss = criterion(y, labels)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()
            
            model_save_path = os.path.join(experiment_dir, f"model_epoch_{i+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")


            validation_test(output_path=experiment_dir, model= model, dataloader=dataloader_test,task=task, attentions_heads=[1])

            # Save cumulative loss to the file
            with open(loss_file, 'a') as f:
                f.write(f"{i+1},{cumulative_loss}\n")

    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model weights...")
        latest_model_save_path = os.path.join(experiment_dir, "latest_model_interrupted.pth")
        torch.save(model.state_dict(), latest_model_save_path)
        print(f"Latest model saved at {latest_model_save_path}")

if __name__ == "__main__":
    run_training()
