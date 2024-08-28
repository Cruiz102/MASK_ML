
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mask_ml.utils.datasets import create_dataloader
from typing import List
from mask_ml.model.vit import ViTConfig, VitModel
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
from mask_ml.model.trainer import Trainer, TrainerConfig
class MultiClassInstanceSegmentationLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, weight_iou=1.0, num_classes=3):
        super(MultiClassInstanceSegmentationLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, outputs, labels, smooth=1e-6):
        dice = 0
        for c in range(self.num_classes):
            outputs_c = outputs[:, c]
            labels_c = (labels == c).float()
            intersection = (outputs_c * labels_c).sum(dim=(1, 2))
            union = outputs_c.sum(dim=(1, 2)) + labels_c.sum(dim=(1, 2))
            dice += (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def iou_loss(self, outputs, labels, smooth=1e-6):
        iou = 0
        for c in range(self.num_classes):
            outputs_c = outputs[:, c]
            labels_c = (labels == c).float()
            intersection = (outputs_c * labels_c).sum(dim=(1, 2))
            union = outputs_c.sum(dim=(1, 2)) + labels_c.sum(dim=(1, 2)) - intersection
            iou += (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()

    def forward(self, outputs, labels):
        ce_loss = self.ce_loss(outputs, labels)

        # Apply softmax to outputs for multi-class classification
        outputs = F.softmax(outputs, dim=1)

        dice_loss = self.dice_loss(outputs, labels)
        iou_loss = self.iou_loss(outputs, labels)

        total_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss) + (self.weight_iou * iou_loss)
        return total_loss

@hydra.main(version_base=None, config_path="config", config_name="training")
def run_training(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    dataset_name = cfg['dataset']
    data_image_dir =  cfg['datasets'][dataset_name]['base_dir'] + cfg['datasets'][dataset_name][f'{dataset_name}_validation']['image_dir']
    data_annotation_dir = cfg['datasets'][dataset_name]['base_dir'] + cfg['datasets'][dataset_name][f'{dataset_name}_validation']['annotation_dir']
    batch_size = cfg['batch_size']
    task = cfg['task']
    attention_heads_to_visualize = cfg['visualization']['attention_heads']

    trainer_config = TrainerConfig()
    trainer = Trainer(trainer_config)

    trainer.train()



def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    dataset_name = cfg['dataset']
    data_image_dir =  cfg['datasets'][dataset_name]['base_dir'] + cfg['datasets'][dataset_name][f'{dataset_name}_validation']['image_dir']
    data_annotation_dir = cfg['datasets'][dataset_name]['base_dir'] + cfg['datasets'][dataset_name][f'{dataset_name}_validation']['annotation_dir']
    batch_size = cfg['batch_size']
    task = cfg['task']
    attention_heads_to_visualize = cfg['visualization']['attention_heads']
