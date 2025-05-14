import torch
import pytest
from mask_ml.model.auto_encoder import ImageAutoEncoder
from mask_ml.model.mlp import MLP
from mask_ml.model.vit import VitModel, generate_all_valid_configurations
from utils import get_layer_output, visualize_latent_space
from mask_ml.utils.datasets import create_dataloader
from train import run_training
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import os
import tempfile
import torch.nn as nn
import shutil
import glob
import numpy as np
from pathlib import Path


# Fixture for creating a standard test configuration
@pytest.fixture
def base_config():
    """
    Creates a basic configuration for ImageAutoEncoder tests.
    This configuration can be modified for specific test cases.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = OmegaConf.create({
            "output_dir": temp_dir,
            "experiment_name": "test_autoencoder",
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "image_reshape": 28,
            "transfer_learning_weights": "",
            "datasets": {
                "name": "mnist_classification",
                "base_dir": "./datasets",
                "download": True,
                "train": True,
                "image_size": 28
            },
            "model": {
                "_target_": "mask_ml.model.auto_encoder.ImageAutoEncoder",
                "encoder": {
                    "_target_": "mask_ml.model.mlp.MLP",
                    "activation_function": "relu",
                    "input_size": 784,  # 28x28 flattened
                    "output_size": 32,
                    "hidden_sizes": [256, 128]
                },
                "decoder": {
                    "_target_": "mask_ml.model.mlp.MLP",
                    "activation_function": "relu",
                    "input_size": 32,
                    "output_size": 784,  # 28x28 flattened
                    "hidden_sizes": [128, 256]
                },
                "image_size": 28,
                "flatten": True
            },
            "loss_function": {
                "_target_": "torch.nn.MSELoss"
            },
            "attention_heads": [],
            "step_loss_logging_rate": 5,
            "latent_space_visualization": False,  # Disabled by default
            "latent_sample_space_size": 10,
            "latent_space_layer_name": 'encoder',
            "latent_space_pca_components": 2
        })
        
        yield cfg, temp_dir


def test_get_layer_for_latent_representation_pca():
    # Test Configuration
    ###################
    image_size = 28
    batch_size = 8
    latent_spaces = [10,20,30,32,50]
    for latent_dim in latent_spaces:    
        autoencoder = ImageAutoEncoder(
            image_size= image_size,
            flatten= False,
            encoder=MLP(
                activation_function="relu",
                input_size=image_size,
                output_size=latent_dim
            ),
            decoder=MLP(
                activation_function="relu",
                input_size=latent_dim,
                output_size=image_size
            )
            )
        image = torch.randn([batch_size,image_size])
        latent_representation = get_layer_output(autoencoder, image, 'encoder',batch_size, flatten=True) 
        assert (latent_representation.shape[0], latent_representation.shape[1]) == (batch_size, latent_dim), f"Expected latent vector to be ({batch_size}, {latent_dim}), got ({latent_representation.shape[0]}, {latent_representation[1]})"

    vit_configs = generate_all_valid_configurations(image_size)
    vit_configs = vit_configs[len(vit_configs)//2]
    vit_model = VitModel(
        transformer_blocks=1,
        image_size=image_size,
        patch_size=vit_configs["kernel"],
        num_channels= 3,
        encoder_stride=vit_configs['stride'],
        embedded_size= image_size,
        attention_heads=vit_configs['attention_heads'],
    )
    x = torch.randn([batch_size, 3, image_size, image_size])
    embedded_layer_vit = get_layer_output(vit_model, x, 'embedded_layer' )
    assert (embedded_layer_vit.shape[0], embedded_layer_vit.shape[1], embedded_layer_vit.shape[2]) == (batch_size,485,image_size), f"Expected latent vector to be ({batch_size}, {485}, {image_size}), got ({embedded_layer_vit.shape[0]}, {embedded_layer_vit.shape[1]}, {embedded_layer_vit.shape[2]})"


    print("Forward pass successful")


def test_files_generation(base_config):
    """
    Test that all expected files are correctly generated during training.
    This verifies that weights, loss logs, and other outputs are created.
    """
    cfg, temp_dir = base_config
    
    # Run the training pipeline
    try:
        run_training(cfg)
        
        # Verify output files were created
        experiment_dirs = glob.glob(os.path.join(temp_dir, "test_autoencoder*"))
        assert len(experiment_dirs) > 0, "No experiment directory created"
        
        experiment_dir = experiment_dirs[0]
        
        # Check that model weights were saved
        model_files = glob.glob(os.path.join(experiment_dir, "model_epoch_*.pth"))
        assert len(model_files) > 0, "No model weights saved"
        
        # Check that loss file was created
        loss_file = os.path.join(experiment_dir, "step_losses.csv")
        assert os.path.exists(loss_file), "Loss file not created"
        
        # Check if loss plot was created
        loss_plot = os.path.join(experiment_dir, "step_loss_plot.png")
        assert os.path.exists(loss_plot), "Loss plot not created"
        
        # Check other expected files
        validation_results = os.path.join(experiment_dir, "validation_results.csv")
        assert os.path.exists(validation_results), "Validation results file not created"
        
        # Check for reconstructions directory
        reconstruction_dir = os.path.join(experiment_dir, "reconstructions")
        assert os.path.exists(reconstruction_dir), "Reconstructions directory not created"
        
    except Exception as e:
        pytest.fail(f"run_training failed with error: {str(e)}")


def test_model_inference(base_config):
    """
    Test that a trained model can be loaded and perform inference correctly.
    """
    cfg, temp_dir = base_config
    
    # Run the training pipeline
    try:
        run_training(cfg)
        
        # Get the experiment directory
        experiment_dirs = glob.glob(os.path.join(temp_dir, "test_autoencoder*"))
        experiment_dir = experiment_dirs[0]
        
        # Get the latest model file
        model_files = glob.glob(os.path.join(experiment_dir, "model_epoch_*.pth"))
        latest_model_file = sorted(model_files)[-1]
        
        # Create device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(latest_model_file, map_location=device))
        model.to(device)
        model.eval()
        
        # Create a test dataloader
        test_cfg = OmegaConf.create({
            "datasets": cfg.datasets,
            "batch_size": 8
        })
        _, dataloader_test = create_dataloader(test_cfg)
        
        # Get a batch of test data
        test_batch, _ = next(iter(dataloader_test))
        test_batch = test_batch.to(device)
        
        # Run inference
        with torch.no_grad():
            reconstructed = model(test_batch)
            
            # Verify output shape
            assert reconstructed.shape == test_batch.shape, \
                f"Expected reconstruction shape {test_batch.shape}, got {reconstructed.shape}"
            
            # Check reasonable reconstruction error
            reconstruction_error = nn.MSELoss()(reconstructed, test_batch).item()
            assert reconstruction_error < 0.5, f"Reconstruction error too high: {reconstruction_error}"
        
    except Exception as e:
        pytest.fail(f"Model inference test failed with error: {str(e)}")


def test_latent_space_visualization(base_config):
    """
    Test that latent space visualization works correctly.
    """
    cfg, temp_dir = base_config
    
    # Enable latent space visualization
    cfg.latent_space_visualization = True
    cfg.latent_sample_space_size = 20
    cfg.latent_space_pca_components = 2
    
    # Run the training pipeline
    try:
        run_training(cfg)
        
        # Get the experiment directory
        experiment_dirs = glob.glob(os.path.join(temp_dir, "test_autoencoder*"))
        experiment_dir = experiment_dirs[0]
        
        # Check that latent space visualization files were created
        latent_space_2d = os.path.join(experiment_dir, "latent_space_2d_pca.png")
        assert os.path.exists(latent_space_2d), "2D latent space visualization not created"
        
        # Load the model for further testing
        model_files = glob.glob(os.path.join(experiment_dir, "model_epoch_*.pth"))
        latest_model_file = sorted(model_files)[-1]
        
        # Create device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(latest_model_file, map_location=device))
        model.to(device)
        model.eval()
        
        # Create a test dataloader
        test_cfg = OmegaConf.create({
            "datasets": cfg.datasets,
            "batch_size": 10
        })
        _, dataloader_test = create_dataloader(test_cfg)
        
        # Get a batch of test data
        test_batch, test_labels = next(iter(dataloader_test))
        test_batch = test_batch.to(device)
        
        # Get latent representations
        batch_size = test_batch.shape[0]
        with torch.no_grad():
            latent = get_layer_output(model, test_batch, 'encoder', batch_size, flatten=True)
        
        # Verify latent space dimensions
        expected_latent_dim = cfg.model.encoder.output_size
        assert latent.shape == (batch_size, expected_latent_dim), \
            f"Expected latent shape ({batch_size}, {expected_latent_dim}), got {latent.shape}"
        
        # Test manual visualization in a temp directory
        test_vis_dir = os.path.join(temp_dir, "test_visualization")
        os.makedirs(test_vis_dir, exist_ok=True)
        
        latents_np = latent.cpu().numpy()
        labels_np = test_labels.cpu().numpy()
        
        # Test visualization function directly
        visualize_latent_space(latents_np, labels_np, n_components=2, save_path=test_vis_dir)
        
        # Verify visualization file was created
        assert os.path.exists(os.path.join(test_vis_dir, "latent_space_2d_pca.png")), \
            "Manual latent space visualization failed"
        
    except Exception as e:
        pytest.fail(f"Latent space visualization test failed with error: {str(e)}")


def test_3d_latent_space_visualization(base_config):
    """
    Test that 3D latent space visualization works correctly.
    """
    cfg, temp_dir = base_config
    
    # Enable 3D latent space visualization
    cfg.latent_space_visualization = True
    cfg.latent_sample_space_size = 20
    cfg.latent_space_pca_components = 3  # Set to 3D
    
    # Run the training pipeline
    try:
        run_training(cfg)
        
        # Get the experiment directory
        experiment_dirs = glob.glob(os.path.join(temp_dir, "test_autoencoder*"))
        experiment_dir = experiment_dirs[0]
        
        # Check that 3D latent space visualization file was created
        latent_space_3d = os.path.join(experiment_dir, "latent_space_3d_pca.png")
        assert os.path.exists(latent_space_3d), "3D latent space visualization not created"
        
    except Exception as e:
        pytest.fail(f"3D latent space visualization test failed with error: {str(e)}")


def test_custom_latent_dimension(base_config):
    """
    Test training with custom latent dimensions.
    """
    cfg, temp_dir = base_config
    
    # Modify latent dimension (encoder output_size and decoder input_size)
    cfg.model.encoder.output_size = 10
    cfg.model.decoder.input_size = 10
    
    # Run the training pipeline
    try:
        run_training(cfg)
        
        # Get the experiment directory
        experiment_dirs = glob.glob(os.path.join(temp_dir, "test_autoencoder*"))
        experiment_dir = experiment_dirs[0]
        
        # Load the model
        model_files = glob.glob(os.path.join(experiment_dir, "model_epoch_*.pth"))
        latest_model_file = sorted(model_files)[-1]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(latest_model_file, map_location=device))
        model.to(device)
        model.eval()
        
        # Create a test dataloader and get a batch
        test_cfg = OmegaConf.create({
            "datasets": cfg.datasets,
            "batch_size": 8
        })
        _, dataloader_test = create_dataloader(test_cfg)
        test_batch, _ = next(iter(dataloader_test))
        test_batch = test_batch.to(device)
        
        # Verify latent dimension
        with torch.no_grad():
            batch_size = test_batch.shape[0]
            latent = get_layer_output(model, test_batch, 'encoder', batch_size, flatten=True)
            assert latent.shape == (batch_size, 10), f"Expected latent shape (8, 10), got {latent.shape}"
        
    except Exception as e:
        pytest.fail(f"Custom latent dimension test failed with error: {str(e)}")


# Run the tests with Pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])

