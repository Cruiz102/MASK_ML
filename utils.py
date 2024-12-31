import os
import matplotlib.pyplot as plt
from typing import List
from torch import Tensor
import numpy as np
from sklearn.decomposition import PCA
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


def get_layer_output(model, x, layer_name: str, batch_size: int = 32, flatten: bool = True):
    """
    Returns the output of a specified layer within the model.

    :param model: The PyTorch model.
    :param x: Input tensor.
    :param layer_name: Name of the layer to capture output from.
    :return: Output tensor from the specified layer.
    """
    # Example for accessing a sub-layer by name
    if layer_name == 'encoder':
        x = x.view(batch_size, -1)
        return model.image_encoder(x)
    elif layer_name == 'decoder':
        encoded = model.image_encoder(x)
        return model.decoder(encoded)
    elif hasattr(model, layer_name):  # Generic case for named layers
        layer = getattr(model, layer_name)
        return layer(x)
    else:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")