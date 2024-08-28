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


def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = np.bincount(inds, minlength=n**2).reshape(n, n)

    return mat


# This func is deprecated since it's not memory efficient
def legacy_mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(
            results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    iou = np.diag(total_mat) / (
        total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    return all_acc, acc, iou

def validation_test(output_path: str, model: nn.Module, dataloader: DataLoader, loss_function: Callable, attentions_heads: List):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    csv_file = os.path.join(output_path, "validation_results.csv")
    
    # Create the CSV file and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Batch Index', 'Loss', 'Time (seconds)', 'Output Shape'])

    with torch.no_grad():
        for i, (imgs, label) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            start_time = time.time()  # Start timing

            if isinstance(model, SegmentationAutoEncoder):
                is_encoder_transformer = isinstance(model.encoder, VitModel)
                is_decoder_transformer = isinstance(model.decoder, VitModel)
            
            if isinstance(model, VitModel) or is_encoder_transformer:
                outputs = model(imgs.float(), attention_heads_idx=attentions_heads)
            else:
                outputs = model(imgs.float())  # For non-transformer models

            loss = loss_function(outputs, label)
            total_loss += loss.item()

            end_time = time.time()  # End timing
            batch_time = end_time - start_time

            print(f"Batch {i+1}/{num_batches}")
            print(f"Output shape: {outputs.shape}")
            print(f"Batch loss: {loss.item():.4f}")
            print(f"Batch processing time: {batch_time:.4f} seconds")
            print("-" * 40)

    avg_loss = total_loss / num_batches
    print(f"Validation complete. Average loss: {avg_loss:.4f}")

      # Write results to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, loss.item(), batch_time, outputs.shape])

    print(f"Batch {i + 1} - Loss: {loss.item():.4f}, Time: {batch_time:.4f} seconds")
    print(f"Output shape: {outputs.shape}")

    return avg_loss
@hydra.main(version_base=None, config_path="config", config_name="evaluation")
def run_evaluation(cfg: DictConfig):
    # Print the full configuration
    print(OmegaConf.to_yaml(cfg))
    dataset_name = cfg['dataset']
    data_image_dir =  cfg['datasets'][dataset_name]['base_dir'] + cfg['datasets'][dataset_name][f'{dataset_name}_validation']['image_dir']
    data_annotation_dir = cfg['datasets'][dataset_name]['base_dir'] + cfg['datasets'][dataset_name][f'{dataset_name}_validation']['annotation_dir']
    batch_size = cfg['batch_size']
    task = cfg['task']
    attention_heads_to_visualize = cfg['visualization']['attention_heads']

    dataloader = create_dataloader(dataset_name=dataset_name, image_dir=data_image_dir, annotation_dir= data_annotation_dir,batch_size=batch_size, )

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

    model.eval()  # Set the model to evaluation mode

   # Create a directory to save the outputs and visualizations
    output_dir = os.path.join(cfg.output_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    if task == 'classification':
        all_true_labels = []
        all_predicted_labels = []

    
    
    validation_test()

if __name__ == "__main__":
    run_evaluation()
