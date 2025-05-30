import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from mask_ml.utils.datasets import create_dataloader
from mask_ml.model.vit import   VitClassificationHead
from mask_ml.model.auto_encoder import ImageAutoEncoder, MaskedAutoEncoder, MaskAutoEncoderCriterion
from tqdm import tqdm
from torch.optim.adamw import AdamW
from eval import validation_test
from hydra.utils import instantiate
from mask_ml.utils.utils import monitor_resources
from utils import create_unique_experiment_dir, plot_loss_per_step
import os



@hydra.main(version_base=None, config_path="config", config_name="training")
def run_training(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dataloader_train, dataloader_test = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = cfg.learning_rate
    epochs = cfg.epochs
    experiment_name = cfg.experiment_name
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    experiment_dir = create_unique_experiment_dir(output_dir, experiment_name)
    
    model = instantiate(cfg.model)


    if cfg.transfer_learning_weights:
        state_dict = torch.load(cfg.transfer_learning_weights, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    
    criterion = instantiate(cfg.loss_function)
    if isinstance(model, MaskedAutoEncoder):
        criterion = MaskAutoEncoderCriterion(
            reconstruction_loss_fn=criterion)

    loss_file = os.path.join(experiment_dir, "step_losses.csv")
    training_data_file = os.path.join(experiment_dir, 'training_data.txt')
    dataset_file = os.path.join(experiment_dir,'dataset_configs.txt')
    resources_file = os.path.join(experiment_dir, 'resources.csv')
    model = model.to(device)

    step_losses = [] 
    step_count = 1
    step_loss_logging_rate = cfg.get("step_loss_logging_rate", 100)
    with open(dataset_file, 'w') as f:
        f.write(f"{dataloader_train.batch_size}")
        for i in range(1):
            for inputs, labels in dataloader_train:
                f.write(f"Input shape :{inputs.shape}")
                f.write(f"Output shape:{labels.shape} ")
                break
    with open(training_data_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    with open(loss_file, 'w') as f:
        f.write("step,loss\n") 
    try:
        for epoch in range(epochs):
            for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # FORWARD PASS DEPENDING ON THE MODEL
                if isinstance(model, VitClassificationHead):
                    y,_ = model(inputs)
                elif isinstance(model,MaskedAutoEncoder):
                    y, mask_indices = model(inputs)
                else:
                    y = model(inputs)


                # LOSS CALCULATION DEPENDING ON THE MODEL
                if isinstance(model, ImageAutoEncoder):
                    loss = criterion(y, inputs)
                elif isinstance(model, MaskedAutoEncoder):
                    loss = criterion(y,inputs,labels)
                else:
                    loss = criterion(y, labels)


                # BACKWARD PASS OPTIMATION
                loss.backward()
                optimizer.step()
                step_losses.append(loss.item())
                if step_count % step_loss_logging_rate == 0:
                    with open(loss_file, 'a') as f:
                        f.write(f"{step_count},{loss.item()}\n")
                monitor_resources(resources_file, step_count)
                step_count += 1

            model_save_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

            validation_test(
                output_path=experiment_dir, 
                model=model, 
                dataloader=dataloader_test, 
                attentions_heads_idx=cfg.attention_heads,
                samples_heads_indices_size=3,
                latent_space_visualization=cfg.get("latent_space_visualization", True),
                latent_sample_space_size=cfg.get("latent_sample_space_size", 100),
                latent_space_layer_name=cfg.get("latent_space_layer_name", 'encoder'),
                n_components_pca=cfg.get("latent_space_pca_components", 2)
            )

    except KeyboardInterrupt:
        print("Training interrupted. Saving latest model weights...")
        latest_model_save_path = os.path.join(experiment_dir, "latest_model_interrupted.pth")
        torch.save(model.state_dict(), latest_model_save_path)
        print(f"Latest model saved at {latest_model_save_path}")

    # Plot the loss per step
    plot_loss_per_step(step_losses, experiment_dir)

if __name__ == "__main__":
    run_training()