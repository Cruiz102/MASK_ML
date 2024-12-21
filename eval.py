import hydra
from omegaconf import DictConfig, OmegaConf
from mask_ml.utils.datasets import create_dataloader
from mask_ml.model.vit import VitClassificationHead, VitClassificationConfig
from mask_ml.model.mlp import MLPClassification, MLPClassificationConfig
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder, SegmentationAutoEncoderConfig
import os
from typing import List, Optional
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

class Tasks(Enum):
    CLASSIFICATION = 1
    SEGMENTATION = 2

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
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_score = 0.0
    num_batches = len(dataloader)

    # Distinguish the task based on the model's type
    if isinstance(model, SegmentationAutoEncoder):
        task = Tasks.SEGMENTATION
    elif isinstance(model, VitClassificationHead) or isinstance(model, MLPClassification):
        task = Tasks.CLASSIFICATION
    else:
        raise ValueError("Unknown model type provided to validation_test.")

    if attentions_heads_idx:     
        os.makedirs(output_path, exist_ok=True)
        attention_dir = os.path.join(output_path, "attention_heads")
        os.makedirs(attention_dir, exist_ok=True)

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
            # Visualize attention heads for randomly sampled batches
            if batch_idx in sampled_batch_indices and attentions_heads_idx and isinstance(model, VitClassificationHead):
                outputs, attention_heads = model(inputs, attentions_heads_idx)
                # Save input images
                batch_dir = os.path.join(attention_dir, f"batch_{batch_idx}")
                os.makedirs(batch_dir, exist_ok=True)
                for i in range(inputs.size(0)):
                    input_image = to_pil_image(inputs[i].cpu())
                    input_image_path = os.path.join(batch_dir, f"input_image_{i}.png")
                    input_image.save(input_image_path)

                # Visualize attention heads
                visualize_attention_heads(attention_heads, os.path.join(batch_dir, "attention_maps"))

                # Save metadata
                metadata_path = os.path.join(batch_dir, "metadata.txt")
                with open(metadata_path, "w") as metadata_file:
                    metadata_file.write(f"Batch Index: {batch_idx}\n")
                    metadata_file.write(f"Batch Size: {inputs.size(0)}\n")
                    metadata_file.write(f"Attention Heads: {attentions_heads_idx}\n")
                    metadata_file.write(f"Labels: {labels.cpu().tolist()}\n")
            elif isinstance(model, VitClassificationHead):
                outputs,_ = model(inputs)
            else:
                outputs = model(inputs)

            if task == Tasks.CLASSIFICATION:
                _, predicted = torch.max(outputs, 1)
                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_labels.extend(predicted.cpu().numpy())
                metric_score = (predicted == labels).float().mean().item()
                total_score += metric_score

            elif task == Tasks.SEGMENTATION:
                # For segmentation, you'd typically compute IoU, Dice, etc.
                # This is just a placeholder.
                # Example: let's do a trivial "score" of 1.0 for each batch (not real metric)
                metric_score = 1.0  
                total_score += metric_score
            else:
                metric_score = 0.0  # fallback if needed

            # Log the data
            end_time = time.time()
            elapsed_time = end_time - start_time
            if log:
                writer.writerow([
                    batch_idx, 
                    f"{metric_score:.4f}", 
                    f"{elapsed_time:.6f}", 
                    list(outputs.shape)
                ])

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

    # Instantiate the model based on configuration
    model_config = instantiate(cfg.model)

    if isinstance(model_config, VitClassificationConfig):
        model = VitClassificationHead(model_config)
    elif isinstance(model_config, SegmentationAutoEncoderConfig):
        model = SegmentationAutoEncoder(model_config)
    elif isinstance(model_config, MLPClassificationConfig):
        model = MLPClassification(model_config)
    else:
        raise ValueError(f"Unsupported model type: {type(model_config)}")

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