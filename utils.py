import os
import matplotlib.pyplot as plt
from typing import List
from torch import Tensor
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
