import hydra
from omegaconf import DictConfig, OmegaConf
from mask_ml.utils.datasets import create_dataloader
from mask_ml.model.vit import VitClassificationHead
from mask_ml.model.mlp import MLPClassification
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder
from mask_ml.model.auto_encoder import ImageAutoEncoder, MaskedAutoEncoder
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
from utils import (create_unique_experiment_dir, visualize_attention_heads, get_layer_output, visualize_latent_space,
                    save_reconstruction_and_error_maps, save_masked_input_and_reconstructions,save_attention_overlay)
import torch.nn.functional as F
import numpy as np

class Tasks(Enum):
    CLASSIFICATION = 1
    SEGMENTATION = 2
    AUTOENCODER = 3 
    MASKAUTOENCODER = 4

def validation_test(
    output_path: str, 
    model: torch.nn.Module, 
    dataloader, 
    attentions_heads_idx: List[int], 
    samples_heads_indices_size: int,
    latent_space_visualization: bool = True,
    latent_sample_space_size: int = 100,
    latent_space_layer_name: str = 'encoder',
    n_components_pca: int = 2,
    log: bool = True
) -> float:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_score = 0.0
    num_batches = len(dataloader)
    flat_output_get_layer_output = False

    if isinstance(model, VitClassificationHead):
        task = Tasks.CLASSIFICATION
    elif isinstance(model, MLPClassification):
        task = Tasks.CLASSIFICATION
    elif isinstance(model, SegmentationAutoEncoder):
        task = Tasks.SEGMENTATION
    elif isinstance(model, ImageAutoEncoder): 
        task = Tasks.AUTOENCODER
        flat_output_get_layer_output = model.flatten

        
    elif isinstance(model, MaskedAutoEncoder):
        task = Tasks.MASKAUTOENCODER
    else:
        raise ValueError("Unknown model type provided to validation_test.")

    os.makedirs(output_path, exist_ok=True)
    if attentions_heads_idx and len(attentions_heads_idx) > 0:  
        attention_dir = os.path.join(output_path, "attention_heads")
        os.makedirs(attention_dir, exist_ok=True)
    if task == Tasks.AUTOENCODER:
        reconstruction_dir = os.path.join(output_path, "reconstructions")
        os.makedirs(reconstruction_dir, exist_ok=True)

    total_loss = 0.0

    total_batches = len(dataloader)
    sampled_batch_indices = random.sample(range(total_batches), samples_heads_indices_size)
    # Classification metrics
    all_true_labels = []
    all_predicted_labels = []

    # Latent space visualization
    latents = []
    latents_labels = []


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
            batch_size = inputs.shape[0]

            if latent_space_visualization and len(latents) < latent_sample_space_size:
                layer_output = get_layer_output(model, inputs, latent_space_layer_name, batch_size=batch_size, flatten= flat_output_get_layer_output)
                latents.append(layer_output.view(layer_output.size(0), -1).cpu().numpy())
                latents_labels.append(labels.cpu().numpy())

            if task == Tasks.AUTOENCODER:
                reconstructed = model(inputs)
                loss = F.mse_loss(reconstructed, inputs)
                total_loss += loss.item()
                metric_score = loss.item()  # Use reconstruction loss as metric
                if batch_idx < samples_heads_indices_size:
                    save_reconstruction_and_error_maps(inputs, reconstructed, reconstruction_dir, batch_idx, inputs.size(0))
            elif task == Tasks.MASKAUTOENCODER:
                reconstructed, masked_indices = model(inputs)
                loss = F.mse_loss(reconstructed, inputs)
                total_loss += loss.item()
                metric_score = loss.item()
                if batch_idx < samples_heads_indices_size:
                    save_masked_input_and_reconstructions(inputs, reconstructed, masked_indices, reconstruction_dir, batch_idx, inputs.size(0))

            else:
                if batch_idx in sampled_batch_indices and attentions_heads_idx and isinstance(model, VitClassificationHead):
                    outputs, attention_heads = model(inputs, attentions_heads_idx)
                    batch_dir = os.path.join(attention_dir, f"batch_{batch_idx}")
                    os.makedirs(batch_dir, exist_ok=True)
                    for i in range(inputs.size(0)):
                        input_image = to_pil_image(inputs[i].cpu())
                        input_image_path = os.path.join(batch_dir, f"input_image_{i}.png")
                        input_image.save(input_image_path)

                    visualize_attention_heads(attention_heads, os.path.join(batch_dir, "attention_maps"))
                    save_attention_overlay(attention_heads,inputs, os.path.join(batch_dir, "overlays"))


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


        # Visualize latent space if enabled
        if latent_space_visualization and len(latents) > 0:
            latents = np.concatenate(latents, axis=0)
            latents_labels = np.concatenate(latents_labels, axis=0)
            visualize_latent_space(
                latents, 
                latents_labels, 
                n_components=n_components_pca, 
                save_path=output_path
            )

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

    dataloader_train, dataloader_test = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directories
    experiment_name = cfg.experiment_name
    base_output_dir = cfg.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    experiment_dir = create_unique_experiment_dir(base_output_dir, experiment_name)

    model = instantiate(cfg.model)

    if cfg.get("transfer_learning_weights"):
        state_dict = torch.load(cfg.transfer_learning_weights, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded transfer learning weights.")

    model = model.to(device)

    validation_test(
        output_path=experiment_dir,
        model=model,
        dataloader=dataloader_test,
        attentions_heads_idx=cfg.get("attention_heads",[0]),
        samples_heads_indices_size=3,
        latent_space_visualization=cfg.get("latent_space_visualization", True),
        latent_sample_space_size=cfg.get("latent_sample_space_size", 100),
        latent_space_layer_name=cfg.get("latent_space_layer_name", 'encoder'),
        n_components_pca=cfg.get("n_components_pca", 2),
        log=True,
    )

    print(f"Evaluation complete. Results saved to {experiment_dir}")


if __name__ == "__main__":
    run_evaluation()