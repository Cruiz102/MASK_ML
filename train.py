import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mask_ml.utils.datasets import create_dataloader
from typing import List
from mask_ml.model.vit import ViTConfig, VitModel, VitClassificationConfig, VitClassificationHead
from mask_ml.model.mask_decoder import MaskDecoderConfig
from mask_ml.model.segmentation_auto_encoder import SegmentationAutoEncoder, SegmentationAutoEncoderConfig
from mask_ml.model.mlp import MLPClassification, MLPClassificationConfig
from mask_ml.model.mask_decoder import MLP
import os
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from tqdm import tqdm
from torch.optim.adamw import AdamW
from eval import validation_test
from hydra.utils import instantiate
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
@hydra.main(version_base=None, config_path="config", config_name="training")
def run_training(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))

    dataloader_train, dataloader_test = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    task = cfg.task
    lr = cfg.learning_rate
    epochs = cfg.epochs
    experiment_name = cfg.experiment_name
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    experiment_dir = create_unique_experiment_dir(output_dir, experiment_name)
    
    # Mapping task to the correct model name
    if task == 'vit_classification':
        model_name = 'vit_classification'
    elif task == 'mlp_classification':
        model_name = 'mlp_classification'
    elif task == 'segmentation':
        model_name = 'segmentation_auto_encoder'
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Instantiate the correct model based on the task
    model_config = instantiate(cfg.model[model_name])

    if isinstance(model_config, MLPClassificationConfig):
        model = MLPClassification(model_config)

    if isinstance(model_config, VitClassificationConfig):
        model = VitClassificationHead(model_config)

    elif isinstance(model_config, SegmentationAutoEncoderConfig):
        model = SegmentationAutoEncoder(model_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_file = os.path.join(experiment_dir, "step_losses.csv")
    training_data_file = os.path.join(experiment_dir, 'training_data.txt')
    model = model.to(device)

    step_losses = []  # To store loss for each step (batch)

    with open(training_data_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    with open(loss_file, 'w') as f:
        f.write("step,loss\n")  # Header for the CSV file
    try:
        step_count = 1
        for epoch in range(epochs):
            for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                y = model(inputs)
                loss = criterion(y, labels)
                loss.backward()
                optimizer.step()

                step_losses.append(loss.item())  # Log the loss per step

                # Save step loss to the file
                with open(loss_file, 'a') as f:
                    f.write(f"{step_count},{loss.item()}\n")
                
                step_count += 1

            model_save_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

            validation_test(output_path=experiment_dir, model=model, dataloader=dataloader_test, task=task, attentions_heads=[1])

    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model weights...")
        latest_model_save_path = os.path.join(experiment_dir, "latest_model_interrupted.pth")
        torch.save(model.state_dict(), latest_model_save_path)
        print(f"Latest model saved at {latest_model_save_path}")

    # Plot the loss per step
    plot_loss_per_step(step_losses, experiment_dir)

if __name__ == "__main__":
    run_training()
