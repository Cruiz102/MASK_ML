import os
import matplotlib.pyplot as plt
from typing import List, Sequence
from torch import Tensor
import numpy as np
from sklearn.decomposition import PCA
import torch
from torchvision.utils import save_image
import cv2
from scipy.ndimage import zoom


def create_unique_experiment_dir(output_dir, experiment_name):
    experiment_dir = os.path.join(output_dir, experiment_name)
    counter = 1
    while os.path.exists(experiment_dir):
        experiment_dir = os.path.join(output_dir, f"{experiment_name}_{counter}")
        counter += 1
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def plot_loss_per_step(step_losses, output_path):
    plt.figure()
    plt.plot(range(1, len(step_losses) + 1), step_losses, marker='o', markersize=2)
    plt.title('Loss per Step (Batch)')
    plt.xlabel('Step (Batch)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'step_loss_plot.png'))
    plt.close()

def plot_recourses_per_step(output_path):
    plt.figure()
    plt.grid()
    plt.savefig(os.path.join(output_path,'resources_plot.png'))
    plt.close()


def visualize_latent_space(latent_vectors: np.ndarray, labels: np.ndarray, n_components: int = 2, save_path: str = None):
    """
    Visualizes latent representations using PCA.

    :param latent_vectors: NumPy array of latent representations.
    :param labels: NumPy array of labels corresponding to the latent vectors.
    :param n_components: Number of PCA components to project (2 or 3).
    :param save_path: Directory where the PCA plot will be saved.
    """
    # Fit PCA
    pca = PCA(n_components=n_components)
    latents_pca = pca.fit_transform(latent_vectors)

    # Plot PCA results
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            latents_pca[:, 0], 
            latents_pca[:, 1], 
            c=labels, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Labels')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('Latent Space (2D PCA)')
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'latent_space_2d_pca.png'))
        else:
            plt.show()

    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(
            latents_pca[:, 0], 
            latents_pca[:, 1], 
            latents_pca[:, 2], 
            c=labels, 
            cmap='viridis', 
            alpha=0.7
        )
        fig.colorbar(p, label='Labels')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        plt.title('Latent Space (3D PCA)')
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'latent_space_3d_pca.png'))
        else:
            plt.show()
    else:
        raise ValueError("n_components must be 2 or 3.")


def get_layer_output(model, x, layer_name: str, batch_size: int = 32, flatten: bool = False):
    """
    Returns the output of a specified layer within the model.

    :param model: The PyTorch model.
    :param x: Input tensor.
    :param layer_name: Name of the layer to capture output from.
    :return: Output tensor from the specified layer.
    """

    if flatten:
        x = x.view(batch_size, -1)
    if hasattr(model, layer_name):  # Generic case for named layers
        layer = getattr(model, layer_name)
        output = layer(x)
        #  This is for models that return more things than the output, Like Mask Indices or Attention Heads
        if  isinstance(output, Sequence):
            output = output[-1]
        
        return layer(x)
    else:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")
    



def save_reconstruction_and_error_maps(inputs, reconstructed, reconstruction_dir, batch_idx, bath_size):
    n_samples = min(inputs.size(0), bath_size)  # Limit to 8 samples per batch

    # Create comparison of original and reconstructed
    comparison = torch.cat([inputs[:n_samples], reconstructed[:n_samples]])
    save_image(comparison.cpu(),
               os.path.join(reconstruction_dir, f'reconstruction_batch_{batch_idx}.png'),
               nrow=n_samples)

    # Create error heatmaps for this batch
    reconstruction_error = (inputs[:n_samples] - reconstructed[:n_samples]).abs()
    error_maps = reconstruction_error.mean(dim=1)  # Average across channels

    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(error_maps[i].cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Sample {i}')
    plt.tight_layout()
    plt.savefig(os.path.join(reconstruction_dir, f'error_heatmaps_batch_{batch_idx}.png'))
    plt.close()



def save_masked_input_and_reconstructions(inputs, reconstructed, masked_indices, reconstruction_dir, batch_idx, batch_size):
    n_samples = min(inputs.size(0), batch_size)  # Limit to batch_size samples per batch

    # Create comparison of original and reconstructed
    comparison = torch.cat([inputs[:n_samples], reconstructed[:n_samples]])
    save_image(comparison.cpu(),
               os.path.join(reconstruction_dir, f'reconstruction_batch_{batch_idx}.png'),
               nrow=n_samples)

    # Create masked input images
    masked_inputs = inputs.clone()
    for idx in masked_indices:
        masked_inputs[:, :, idx // inputs.size(2), idx % inputs.size(3)] = 0  # Mask the patches

    # Save masked input images
    save_image(masked_inputs.cpu(),
               os.path.join(reconstruction_dir, f'masked_inputs_batch_{batch_idx}.png'),
               nrow=n_samples)

    # Create error heatmaps for this batch
    reconstruction_error = (inputs[:n_samples] - reconstructed[:n_samples]).abs()
    error_maps = reconstruction_error.mean(dim=1)  # Average across channels

    plt.figure(figsize=(20, 4))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(error_maps[i].cpu().numpy(), cmap='hot')
        plt.colorbar()
        plt.title(f'Sample {i}')
    plt.tight_layout()
    plt.savefig(os.path.join(reconstruction_dir, f'error_heatmaps_batch_{batch_idx}.png'))
    plt.close()

def save_attention_overlay(attention_heads: List[Tensor], image_batch: Tensor, save_path: str):
    """
    Visualizes and saves attention overlays on images in a batch.
    :param attention_heads: List of attention head tensors (one per layer).
    :param image_batch: A batch of image tensors (Batch, Channels, Height, Width).
    :param save_path: Directory where images will be saved.
    """
    os.makedirs(save_path, exist_ok=True)
    
    for layer_idx, attention_layer in enumerate(attention_heads):
        # Assuming attention_layer is of shape (Batch, Heads, Tokens, Tokens)
        for img_idx, image_tensor in enumerate(image_batch):
            head_avg_attention = attention_layer[img_idx].mean(dim=0)  # Average across heads (Shape: Tokens x Tokens)
            cls_attention = head_avg_attention[0, 1:]  # Ignore [CLS] to [CLS] attention
            
            image_height, image_width = image_tensor.shape[1], image_tensor.shape[2]
            num_patches_height = int(cls_attention.shape[0] ** 0.5)
            num_patches_width = int(cls_attention.shape[0] ** 0.5)
            cls_attention = cls_attention.reshape(num_patches_height, num_patches_width).detach().cpu().numpy()

            # Interpolate the attention map to match the image size
            attention_map = zoom(cls_attention, (image_height / num_patches_height, image_width / num_patches_width))
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            
            # Convert the image tensor to a NumPy array
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            image_np = (image_np * 255).astype(np.uint8)
            
            # Create a heatmap and overlay it on the image
            heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap, 0.6, image_np, 0.4, 0)
            
            # Save the overlay image
            plt.imsave(os.path.join(save_path, f"attention{img_idx + 1}_layer_{layer_idx + 1}.png"), attention_map)
            plt.imsave(os.path.join(save_path, f"overlay{img_idx + 1}_layer_{layer_idx + 1}.png"), overlay)

def visualize_attention_heads(attention_heads: List[Tensor], save_path: str):
    """
    Visualizes and saves attention head heatmaps.
    :param attention_heads: List of attention head tensors (one per layer).
    :param save_path: Directory where images will be saved.
    """
    os.makedirs(save_path, exist_ok=True)
    for layer_idx, head in enumerate(attention_heads):
        # Assuming `head` is of shape (Batch, Heads, Seq_len, Seq_len)
        for head_idx in range(head.shape[1]):
            plt.figure(figsize=(10, 10))
            plt.imshow(head[0, head_idx].cpu().detach().numpy(), cmap="viridis")
            plt.title(f"Layer {layer_idx + 1}, Head {head_idx + 1}")
            plt.colorbar()
            plt.savefig(os.path.join(save_path, f"layer_{layer_idx+1}_head_{head_idx+1}.png"))
            plt.close()

