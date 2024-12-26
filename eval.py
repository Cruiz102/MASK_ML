import hydra
from omegaconf import DictConfig, OmegaConf
from mask_ml.utils.datasets import create_dataloader
from mask_ml.model.vit import VitClassificationHead
from mask_ml.model.mlp import MLPClassification
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder
from mask_ml.model.auto_encoder import ImageAutoEncoder
import os
from typing import List
from torchvision.transforms.functional import to_pil_image
import time
import csv
import torch
from tqdm import tqdm
from enum import Enum
from hydra.utils import instantiate
import random
# Data loading
from utils import create_unique_experiment_dir, visualize_attention_heads
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

class Tasks(Enum):
    CLASSIFICATION = 1
    SEGMENTATION = 2
    AUTOENCODER = 3  # Added new task type

def validation_test(
    output_path: str, 
    model: torch.nn.Module, 
    dataloader, 
    attentions_heads_idx: List[int], 
    samples_heads_indices_size: int = 5,
    log: bool = True
) -> float:
    """
    Evaluates the model on the given dataloader and logs the results.

    :param output_path: Directory to store validation logs/results.
    :param model: The PyTorch model to be evaluated.
    :param dataloader: A DataLoader for the validation/test set.
    :param attentions_heads: Not currently used in this example, but kept for consistency.
    :param log: Whether to log results to CSV.
    :return: Average classification accuracy (for classification models). 
             For segmentation models, you can replace or append metrics as needed.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_score = 0.0
    num_batches = len(dataloader)

    # Distinguish the task based on the model's type
    if isinstance(model, VitClassificationHead):
        task = Tasks.CLASSIFICATION
    elif isinstance(model, MLPClassification):
        task = Tasks.CLASSIFICATION
    elif isinstance(model, SegmentationAutoEncoder):
        task = Tasks.SEGMENTATION
    elif isinstance(model, ImageAutoEncoder):  # Check if it's an autoencoder
        task = Tasks.AUTOENCODER
    else:
        raise ValueError("Unknown model type provided to validation_test.")

    # Create directories for results
    os.makedirs(output_path, exist_ok=True)
    if attentions_heads_idx:
        attention_dir = os.path.join(output_path, "attention_heads")
        os.makedirs(attention_dir, exist_ok=True)
    if task == Tasks.AUTOENCODER:
        reconstruction_dir = os.path.join(output_path, "reconstructions")
        os.makedirs(reconstruction_dir, exist_ok=True)

    # Initialize metrics tracking
    total_loss = 0.0
    reconstruction_samples = []

    # Randomly select batch indices to visualize
    total_batches = len(dataloader)
    sampled_batch_indices = random.sample(range(total_batches), samples_heads_indices_size)

    all_true_labels = []
    all_predicted_labels = []

    if log:
        os.makedirs(output_path, exist_ok=True)
        csv_file = os.path.join(output_path, "validation_results.csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Batch Index', 'Metric Score', 'Inference Time (s)', 'Output Shape'])

    with torch.no_grad(), \
         open(os.path.join(output_path, "validation_results.csv"), mode='a', newline='') as file:
        writer = csv.writer(file)

        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            start_time = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)

            if task == Tasks.AUTOENCODER:
                # Forward pass for autoencoder
                reconstructed = model(inputs)
                loss = F.mse_loss(reconstructed, inputs)
                total_loss += loss.item()
                metric_score = loss.item()  # Use reconstruction loss as metric

                # Save reconstruction samples
                if batch_idx < samples_heads_indices_size:
                    # Save original and reconstructed images
                    comparison = torch.cat([inputs[:8], reconstructed[:8]])
                    save_image(comparison.cpu(),
                             os.path.join(reconstruction_dir, f'reconstruction_{batch_idx}.png'),
                             nrow=8)
                    
                    # Calculate and save reconstruction error heatmap
                    reconstruction_error = (inputs - reconstructed).abs()
                    error_map = reconstruction_error.mean(dim=1)  # Average across channels
                    for i in range(min(8, error_map.size(0))):
                        plt.figure(figsize=(10, 4))
                        plt.imshow(error_map[i].cpu().numpy(), cmap='hot')
                        plt.colorbar()
                        plt.title(f'Reconstruction Error Heatmap - Batch {batch_idx}, Sample {i}')
                        plt.savefig(os.path.join(reconstruction_dir, f'error_heatmap_b{batch_idx}_s{i}.png'))
                        plt.close()

            else:
                # Handle existing classification and attention visualization logic
                if batch_idx in sampled_batch_indices and attentions_heads_idx and isinstance(model, VitClassificationHead):
                    outputs, attention_heads = model(inputs, attentions_heads_idx)
                    # Save input images and visualize attention heads (existing code)
                    batch_dir = os.path.join(attention_dir, f"batch_{batch_idx}")
                    os.makedirs(batch_dir, exist_ok=True)
                    for i in range(inputs.size(0)):
                        input_image = to_pil_image(inputs[i].cpu())
                        input_image_path = os.path.join(batch_dir, f"input_image_{i}.png")
                        input_image.save(input_image_path)

                    visualize_attention_heads(attention_heads, os.path.join(batch_dir, "attention_maps"))

                    metadata_path = os.path.join(batch_dir, "metadata.txt")
                    with open(metadata_path, "w") as metadata_file:
                        metadata_file.write(f"Batch Index: {batch_idx}\n")
                        metadata_file.write(f"Batch Size: {inputs.size(0)}\n")
                        metadata_file.write(f"Attention Heads: {attentions_heads_idx}\n")
                        metadata_file.write(f"Labels: {labels.cpu().tolist()}\n")
                elif isinstance(model, VitClassificationHead):
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)

                if task == Tasks.CLASSIFICATION:
                    _, predicted = torch.max(outputs, 1)
                    all_true_labels.extend(labels.cpu().numpy())
                    all_predicted_labels.extend(predicted.cpu().numpy())
                    metric_score = (predicted == labels).float().mean().item()
                    total_score += metric_score
                elif task == Tasks.SEGMENTATION:
                    metric_score = 1.0  # placeholder for segmentation metric
                    total_score += metric_score

            # Log results
            end_time = time.time()
            elapsed_time = end_time - start_time
            if log:
                writer.writerow([
                    batch_idx,
                    f"{metric_score:.4f}",
                    f"{elapsed_time:.6f}",
                    list(outputs.shape if task != Tasks.AUTOENCODER else reconstructed.shape)
                ])

    # Calculate and return final metrics
    if task == Tasks.AUTOENCODER:
        avg_loss = total_loss / num_batches
        print(f"Validation complete. Average reconstruction loss: {avg_loss:.4f}")
        return avg_loss
    else:
        avg_score = total_score / num_batches if num_batches > 0 else 0
        print(f"Validation complete. Average metric score: {avg_score:.4f}")
        return avg_score


@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    """
    Runs the evaluation process for the model.
    Randomly samples batches for attention visualization if specified.
    """
    # Print the configuration for debugging/logging
    print(OmegaConf.to_yaml(cfg))

    # Create dataloaders for training and testing
    dataloader_train, dataloader_test = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directories
    experiment_name = cfg.experiment_name
    base_output_dir = cfg.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    experiment_dir = create_unique_experiment_dir(base_output_dir, experiment_name)

    # Instantiate the model directly using hydra
    model = instantiate(cfg.model)

    # Load transfer learning weights if specified
    if cfg.get("transfer_learning_weights"):
        state_dict = torch.load(cfg.transfer_learning_weights, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded transfer learning weights.")

    # Move model to the correct device
    model = model.to(device)
    print("attention::::", cfg.get('attention_heads'))

    # Call the validation_test function
    validation_test(
        output_path=experiment_dir,
        model=model,
        dataloader=dataloader_test,
        attentions_heads_idx=[0],
        samples_heads_indices_size=3,
        log=True,
    )

    print(f"Evaluation complete. Results saved to {experiment_dir}")


if __name__ == "__main__":
    run_evaluation()