import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from typing import List
from torch.utils.data import Dataset, DataLoader
from utils.datasets import CocoSegmentationDataset, SA1BImageDataset
import cv2
import os
import logging
from mask_ml.utils.datasets import create_dataloader


def draw_mask_on_image(images, masks):
    pass

class TrainerConfig:
    def __init__(self,
            image_augmentation: bool,
            image_encoder: nn.Module,
            mask_decoder: nn.Module,
            learning_rate: float,
            batch_size: int,
            steps: int,
            lr_scheduler: List[float],
            checkpoint_steps :int = 10,
            dataset : str = "coco"
                 
                 ) -> None:
        self.image_augmentation = image_augmentation
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.steps = steps

        self.lr_scheduler = lr_scheduler
        self.checkpoint_steps= checkpoint_steps
        self.dataset  =  dataset


    output_dir:str # Directory to save the Results of the Training

def iou_evaluation(pred_mask, target_mask):
    # Calculate the intersection over union (IoU)
    intersection = (pred_mask & target_mask).float().sum((1, 2))
    union = (pred_mask | target_mask).float().sum((1, 2))
    iou = intersection / union
    return iou.mean().item()


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.image_encoder.to(self.device)
        self.config.mask_decoder.to(self.device)

    def train(self):
        # Set up optimizer and learning rate scheduler
        optimizer = AdamW(
            list(self.config.image_encoder.parameters()) + list(self.config.mask_decoder.parameters()),
            lr=self.config.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.config.lr_scheduler[0])

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for step in range(self.config.steps):


            for batch in self.dataloader:
                images = batch['image'].to(self.device)
                target_masks = batch['segmentation'].to(self.device)

                # Forward pass through the image encoder
                encoded_images = self.config.image_encoder(images)

                # Forward pass through the mask decoder
                predicted_masks = self.config.mask_decoder(encoded_images)

                # Calculate the loss
                loss = criterion(predicted_masks, target_masks.float())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Learning rate scheduling
            scheduler.step()

            # Save checkpoints
            if step % self.config.checkpoint_steps == 0 or step == self.config.steps - 1:
                checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_step_{step}.pth")
                torch.save({
                    'image_encoder_state_dict': self.config.image_encoder.state_dict(),
                    'mask_decoder_state_dict': self.config.mask_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                }, checkpoint_path)
                print(f"Checkpoint saved at step {step}")

            # Evaluate on a validation set or the current batch for IoU (this is just an example)
            self.config.image_encoder.eval()
            self.config.mask_decoder.eval()
            with torch.no_grad():
                for batch in self.dataloader:
                    images = batch['image'].to(self.device)
                    target_masks = batch['segmentation'].to(self.device)

                    encoded_images = self.config.image_encoder(images)
                    predicted_masks = torch.sigmoid(self.config.mask_decoder(encoded_images)) > 0.5

                    iou = iou_evaluation(predicted_masks, target_masks)
                    print(f"Step {step}, IoU: {iou}")

        print("Training complete.")