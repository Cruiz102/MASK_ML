
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




@hydra.main(version_base=None, config_path="config", config_name="training")
def run_training(cfg: DictConfig):

  # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    dataloader = create_dataloader(cfg)

    model_name = cfg['model']['model_name']
    task = cfg['task']
    lr = cfg.learning_rate
    epochs = cfg.epochs

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

            # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(),lr=lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_scheduler[0])

    # Loss function
    criterion = nn.CrossEntropyLoss()


    for i in range(cfg.epochs):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            y = model(inputs)
            loss = criterion(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())



if __name__ == "__main__":
    run_training()
