import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mask_ml.utils.datasets import create_dataloader
from typing import List
from mask_ml.model.vit import ViTConfig, VitModel, VitClassificationHead
from mask_ml.model.mask_decoder import MaskDecoderConfig
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder, SegmentationAutoEncoderConfig
from mask_ml.model.metrics import iou_score
import os
from tqdm import tqdm
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.nn.functional import one_hot

import torch
import time
import os
import csv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def validation_test(output_path: str, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, task: str, attentions_heads: list, log: bool = True) -> float:
    model.eval()  # Set the model to evaluation mode
    total_score = 0.0
    num_batches = len(dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if task == 'classification':
        all_true_labels = []
        all_predicted_labels = []

    if log:
        # Prepare for logging if required
        csv_file = os.path.join(output_path, "validation_results.csv")
        os.makedirs(output_path, exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Batch Index', 'Loss/Metric', 'Time (seconds)', 'Output Shape'])

    with torch.no_grad():
        for i, (imgs, label) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            imgs = imgs.to(device)
            label = label.to(device)

            start_time = time.time()

            # Check if the model is a SegmentationAutoEncoder or VitModel
            if isinstance(model, SegmentationAutoEncoder):
                is_encoder_transformer = isinstance(model.encoder, VitModel)
                is_decoder_transformer = isinstance(model.decoder, VitModel)
            else:
                is_encoder_transformer = False
                is_decoder_transformer = False

            if isinstance(model, VitModel) or (is_encoder_transformer and False):
                outputs = model(imgs.float(), attention_heads_idx=attentions_heads)
            else:
                outputs = model(imgs.float())  # For non-transformer models

            if task == 'classification':
                _, predicted = torch.max(outputs, 1)
                all_true_labels.extend(label.cpu().numpy())
                all_predicted_labels.extend(predicted.cpu().numpy())
                metric_score = (predicted == label).float().mean().item()
                total_score += metric_score

            elif task == 'segmentation':
                metric_score = iou_score(outputs, label).mean().item()  # Assuming iou_score is implemented
                total_score += metric_score

            end_time = time.time()
            batch_time = end_time - start_time

            # Log the results if logging is enabled
            if log:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([i + 1, metric_score, batch_time, list(outputs.shape)])

        avg_score = total_score / num_batches
        print(f"Validation complete. Average metric: {avg_score:.4f}")

        if task == 'classification':
            conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
            if log:
                # Save confusion matrix as a CSV
                conf_matrix_file = os.path.join(output_path, "confusion_matrix.csv")
                with open(conf_matrix_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["True Label", "Predicted Label", "Count"])
                    for true_label in range(conf_matrix.shape[0]):
                        for pred_label in range(conf_matrix.shape[1]):
                            writer.writerow([true_label, pred_label, conf_matrix[true_label][pred_label]])

                # Plot confusion matrix and save as image
                conf_matrix_img_path = os.path.join(output_path, "confusion_matrix.png")
                plt.figure(figsize=(10, 7))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=True, yticklabels=True)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title("Confusion Matrix")
                plt.savefig(conf_matrix_img_path)
                plt.close()

        print(f"Validation complete. Average score: {avg_score:.4f}")

    return avg_score

@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    dataloader = create_dataloader(cfg, train=False)

    model_name = cfg['model']['model_name']
    task = cfg['task']
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

    model.eval()  # Set the model to evaluation mode

   # Create a directory to save the outputs and visualizations
    output_dir = os.path.join(cfg.output_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    validation_test(output_path=output_dir,dataloader=dataloader, model=model, task=task,attentions_heads=[1])

if __name__ == "__main__":
    run_evaluation()
