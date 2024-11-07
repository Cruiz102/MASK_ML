import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from mask_ml.utils.datasets import create_dataloader
from typing import List
from mask_ml.model.vit import VitClassificationHead
from mask_ml.model.mlp import MLPClassification
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder, SegmentationAutoEncoderConfig
from mask_ml.model.metrics import iou_score
import os
import time
import csv
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum

class Tasks(Enum):
    CLASSIFICATION = 1
    SEGMENTATION = 2


def validation_test(output_path: str, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, attentions_heads: list, log: bool = True) -> float:
    model.eval()  # Set the model to evaluation mode
    total_score = 0.0
    num_batches = len(dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_true_labels = []
    all_predicted_labels = []
    if log:
        csv_file = os.path.join(output_path, "validation_results.csv")
        os.makedirs(output_path, exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Batch Index', 'Loss/Metric', 'Time (seconds)', 'Output Shape'])

    with torch.no_grad():
        for i, (inputs, label) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            imgs = imgs.to(device)
            label = label.to(device)

            start_time = time.time()

            # Check if the model is a SegmentationAutoEncoder or VitModel
            if isinstance(model, SegmentationAutoEncoder):
                task = Tasks.SEGMENTATION
            elif isinstance(model,VitClassificationHead) or isinstance(model,MLPClassification ):
                task = Tasks.CLASSIFICATION  

            if task == Tasks.CLASSIFICATION:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_true_labels.extend(label.cpu().numpy())
                all_predicted_labels.extend(predicted.cpu().numpy())
                metric_score = (predicted == label).float().mean().item()
                total_score += metric_score


        avg_score = total_score / num_batches
        print(f"Validation complete. Average metric: {avg_score:.4f}")


    return avg_score

@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    dataloader = create_dataloader(cfg, train=False)


    model.eval()  # Set the model to evaluation mode

   # Create a directory to save the outputs and visualizations
    output_dir = os.path.join(cfg.output_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    validation_test(output_path=output_dir,dataloader=dataloader, model=model, task=task,attentions_heads=[1])

if __name__ == "__main__":
    run_evaluation()
